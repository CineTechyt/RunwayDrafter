//! Viewport tools: translate raw viewport input into camera moves, selection
//! and document edits.

use app_ui::{CanvasTool as Tool, EditorState, PointMarker, SewPick, ViewportInput};
use cad_core::geometry::Curve as _;
use cad_core::{
    ControlPoint, DocumentChange, DocumentSnapshot, EdgeRef, EdgeSpan, HandleRef, History,
    MoveHandle, Outline, PatternDocument, PieceId, PointKind, Segment,
};
use gfx_render::{Camera2D, OrbitCamera};
use glam::Vec2;
use topology_sync::{GarmentMesh, SurfaceLocation, TopologySync};

/// Pick radius around handles, in pixels.
const PICK_RADIUS_PX: f32 = 9.0;

struct HandleDrag {
    handle: HandleRef,
    /// Position when the drag began, for the undo record.
    from: Vec2,
    /// Handle position minus the grab point, so the handle does not jump to
    /// the cursor.
    grab_offset: Vec2,
}

/// A piece being dragged around the layout.
struct PieceDrag {
    piece: PieceId,
    /// The document before the drag, for the undo record.
    before: PatternDocument,
    /// Pointer position of the previous frame, in pattern space.
    last: Vec2,
    moved: bool,
}

/// A line being bent with the curve tool.
struct BendDrag {
    piece: PieceId,
    edge: u32,
    /// Curve parameter of the grabbed point.
    t: f32,
    before: PatternDocument,
    last: Vec2,
    moved: bool,
}

/// The 2D canvas tools: select and move pieces, edit points, sew lines.
#[derive(Default)]
pub struct CanvasTool {
    pub hovered: Option<HandleRef>,
    /// The line under the cursor.
    pub hovered_line: Option<(PieceId, EdgeSpan)>,
    drag: Option<HandleDrag>,
    piece_drag: Option<PieceDrag>,
    bend: Option<BendDrag>,
    panning: bool,
}

/// Cuts the line through `edge` into `parts` lines of equal length with
/// new turn points. Records nothing; returns whether it changed.
pub fn split_line(document: &mut PatternDocument, piece: PieceId, edge: u32, parts: u32) -> bool {
    let Some(outline) = document.piece(piece).map(|p| &p.outline) else {
        return false;
    };
    if parts < 2 || edge >= outline.edge_count() {
        return false;
    }
    let span = outline.line_span(edge);
    let mut cuts: Vec<(u32, f32)> = (1..parts)
        .map(|k| outline.span_location(span, k as f32 / parts as f32))
        .collect();
    // From the highest edge down, so earlier edges keep their numbers; a
    // second cut on an already cut edge is measured on its first half.
    cuts.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.total_cmp(&a.1)));
    let mut changed = false;
    let mut previous: Option<(u32, f32)> = None;
    for (e, t) in cuts {
        let local = match previous {
            Some((pe, pt)) if pe == e && pt > 0.0 => t / pt,
            _ => t,
        };
        changed |= document.insert_point(piece, e, local).is_some();
        previous = Some((e, t));
    }
    changed
}

/// Where the next sketch point goes for the cursor at `at`: straight
/// horizontal and vertical lines snap within 3°, Shift snaps to 15° steps,
/// and the first point attracts the cursor to close the piece.
fn sketch_point(points: &[Vec2], at: Vec2, shift: bool, radius: f32) -> Vec2 {
    let Some(&last) = points.last() else {
        return at;
    };
    if points.len() >= 3 && points[0].distance(at) <= radius * 2.0 {
        return points[0];
    }
    let d = at - last;
    let length = d.length();
    if length < 1e-6 {
        return at;
    }
    let angle = d.y.atan2(d.x).to_degrees();
    let step = if shift { 15.0 } else { 90.0 };
    let snapped = (angle / step).round() * step;
    if shift || (angle - snapped).abs() <= 3.0 {
        last + Vec2::from_angle(snapped.to_radians()) * length
    } else {
        at
    }
}

/// The line (run of edges between turn points) at `at`, if any.
pub fn pick_line(
    document: &PatternDocument,
    at: Vec2,
    radius: f32,
) -> Option<(PieceId, EdgeSpan, EdgeRef)> {
    let edge = document.pick_edge(at, radius)?;
    let piece = document.piece(edge.piece)?;
    Some((edge.piece, piece.outline.line_span(edge.edge), edge))
}

/// Whether any edge of the span is sewn already.
fn is_sewn(document: &PatternDocument, piece: PieceId, span: EdgeSpan) -> bool {
    let Some(p) = document.piece(piece) else {
        return true;
    };
    let n = p.outline.edge_count();
    span.edges(n)
        .any(|edge| document.seam_of(EdgeRef::new(piece, edge)).is_some())
}

fn overlaps(document: &PatternDocument, a: &SewPick, b: &SewPick) -> bool {
    let Some(piece) = document.piece(a.piece) else {
        return false;
    };
    let n = piece.outline.edge_count();
    a.piece == b.piece && a.span.edges(n).any(|edge| b.span.contains(edge, n))
}

/// Sews the two picks together. Each side starts at the end nearer to
/// where it was first clicked. Records one undo step.
pub fn sew(
    document: &mut PatternDocument,
    history: &mut History,
    a: SewPick,
    b: SewPick,
) -> Option<cad_core::SeamId> {
    if is_sewn(document, a.piece, a.span)
        || is_sewn(document, b.piece, b.span)
        || overlaps(document, &a, &b)
    {
        return None;
    }
    let near_end = |pick: &SewPick| {
        document
            .piece(pick.piece)
            .is_some_and(|p| p.outline.span_fraction(pick.span, pick.at) > 0.5)
    };
    // `reversed` means: the start of A meets the end of B.
    let reversed = near_end(&a) != near_end(&b);
    let before = document.clone();
    let id = document.add_seam_spans((a.piece, a.span), (b.piece, b.span), reversed);
    history.record(Box::new(DocumentSnapshot::new(
        "Sew",
        before,
        document.clone(),
    )));
    Some(id)
}

impl CanvasTool {
    /// The handle being dragged, if any.
    pub fn active(&self) -> Option<HandleRef> {
        self.drag.as_ref().map(|d| d.handle)
    }

    pub fn update(
        &mut self,
        input: &ViewportInput,
        camera: &mut Camera2D,
        document: &mut PatternDocument,
        history: &mut History,
        editor: &mut EditorState,
        changes: &mut Vec<DocumentChange>,
    ) {
        let viewport = Vec2::new(input.size[0] as f32, input.size[1] as f32);
        if let Some(pointer) = input.pointer
            && input.zoom != 1.0
        {
            camera.zoom_about(pointer, viewport, input.zoom);
        }
        if input.secondary.dragging || input.middle.dragging {
            camera.pan(input.drag_delta);
        }
        let radius = PICK_RADIUS_PX * camera.pixel_size();
        let world = |screen: Vec2, camera: &Camera2D| camera.screen_to_world(screen, viewport);
        let tool = editor.canvas_tool;
        if std::mem::take(&mut editor.sketch_finish)
            && matches!(tool, Tool::SketchLine | Tool::SketchSpline)
        {
            self.finish_sketch(document, history, editor, changes, tool);
        }
        if !matches!(tool, Tool::SketchLine | Tool::SketchSpline) {
            editor.sketch_preview = None;
        }
        if tool != Tool::AddPoint {
            editor.point_marker = None;
        }
        if tool != Tool::Sew {
            editor.sew_a = None;
            editor.sew_b = None;
        }

        // --- press -------------------------------------------------------------
        if input.primary.started {
            // Pick where the button went down, not where the drag threshold
            // was crossed.
            let origin = input
                .press_origin
                .or(input.pointer)
                .map(|p| world(p, camera));
            self.panning = true;
            if let Some(at) = origin {
                match tool {
                    Tool::Points => {
                        let handle = editor
                            .selected_piece
                            .and_then(|piece| pick_handle_in(document, piece, at, radius));
                        if let Some(handle) = handle {
                            let from = document.handle_position(handle).unwrap_or(at);
                            self.drag = Some(HandleDrag {
                                handle,
                                from,
                                grab_offset: from - at,
                            });
                            self.panning = false;
                        } else if let Some(piece) = pick_line(document, at, radius * 1.5)
                            .map(|l| l.0)
                            .or_else(|| document.pick_piece(at))
                        {
                            editor.selected_piece = Some(piece);
                            self.panning = false;
                        }
                    }
                    Tool::Select => {
                        let line = pick_line(document, at, radius * 1.5);
                        if let Some(piece) = line.map(|l| l.0).or_else(|| document.pick_piece(at)) {
                            editor.selected_piece = Some(piece);
                            editor.selected_edge = line.map(|l| l.2);
                            self.piece_drag = Some(PieceDrag {
                                piece,
                                before: document.clone(),
                                last: at,
                                moved: false,
                            });
                            self.panning = false;
                        }
                    }
                    Tool::Curve => {
                        if let Some(edge) = document.pick_edge(at, radius * 1.5)
                            && let Some(piece) = document.piece(edge.piece)
                        {
                            let t = piece.outline.nearest_on_edge(edge.edge, at);
                            editor.selected_piece = Some(edge.piece);
                            editor.selected_edge = Some(edge);
                            self.bend = Some(BendDrag {
                                piece: edge.piece,
                                edge: edge.edge,
                                t,
                                before: document.clone(),
                                last: at,
                                moved: false,
                            });
                            self.panning = false;
                        }
                    }
                    Tool::Sew | Tool::AddPoint => {}
                    Tool::SketchLine | Tool::SketchSpline => {}
                    Tool::Button | Tool::Buttonhole => {}
                    Tool::Pull => {}
                }
            }
        }

        // --- drag ----------------------------------------------------------------
        if input.primary.dragging {
            let pointer = input.pointer.map(|p| world(p, camera));
            if let (Some(drag), Some(at)) = (&self.drag, pointer) {
                let target = at + drag.grab_offset;
                let moved = document.handle_position(drag.handle) != Some(target)
                    && document.piece_mut(drag.handle.piece).is_some_and(|piece| {
                        piece.outline.set_handle_position(drag.handle.kind, target)
                    });
                if moved {
                    document.touch();
                    changes.push(DocumentChange::PieceGeometry(drag.handle.piece));
                }
            } else if let (Some(bend), Some(at)) = (&mut self.bend, pointer) {
                let delta = at - bend.last;
                if delta != Vec2::ZERO
                    && document
                        .piece_mut(bend.piece)
                        .is_some_and(|p| p.outline.bend_edge(bend.edge, bend.t, delta))
                {
                    document.touch();
                    changes.push(DocumentChange::PieceGeometry(bend.piece));
                    bend.last = at;
                    bend.moved = true;
                }
            } else if let (Some(drag), Some(at)) = (&mut self.piece_drag, pointer) {
                let delta = at - drag.last;
                if delta != Vec2::ZERO && document.move_piece(drag.piece, delta) {
                    changes.push(DocumentChange::PieceGeometry(drag.piece));
                    drag.last = at;
                    drag.moved = true;
                }
            } else if self.panning {
                camera.pan(input.drag_delta);
            }
        }

        // --- release -------------------------------------------------------------
        if input.primary.stopped {
            self.panning = false;
            // The document already holds the result; record one undo step
            // for the whole gesture.
            if let Some(drag) = self.drag.take()
                && let Some(to) = document.handle_position(drag.handle)
                && to != drag.from
            {
                history.record(Box::new(MoveHandle {
                    handle: drag.handle,
                    from: drag.from,
                    to,
                }));
            }
            if let Some(bend) = self.bend.take()
                && bend.moved
            {
                history.record(Box::new(DocumentSnapshot::new(
                    "Bend line",
                    bend.before,
                    document.clone(),
                )));
            }
            if let Some(drag) = self.piece_drag.take()
                && drag.moved
            {
                history.record(Box::new(DocumentSnapshot::new(
                    "Move piece",
                    drag.before,
                    document.clone(),
                )));
            }
        }

        // --- double-click with the curve tool: straighten ------------------------
        if input.double_clicked
            && tool == Tool::Curve
            && let Some(at) = input.pointer.map(|p| world(p, camera))
            && let Some(edge) = document.pick_edge(at, radius * 1.5)
        {
            let before = document.clone();
            if document
                .piece_mut(edge.piece)
                .is_some_and(|p| p.outline.straighten_edge(edge.edge))
            {
                document.touch();
                history.record(Box::new(DocumentSnapshot::new(
                    "Straighten line",
                    before,
                    document.clone(),
                )));
                changes.push(DocumentChange::PieceGeometry(edge.piece));
            }
        }

        // --- double-click: join two lines / split a line at a point ------------
        if input.double_clicked
            && tool == Tool::Points
            && let (Some(piece), Some(at)) = (
                editor.selected_piece,
                input.pointer.map(|p| world(p, camera)),
            )
            && let Some(HandleRef {
                kind: cad_core::HandleKind::Anchor(anchor),
                ..
            }) = pick_handle_in(document, piece, at, radius)
        {
            let before = document.clone();
            let current = document
                .piece(piece)
                .map(|p| p.outline.points()[anchor as usize].kind);
            let kind = if current == Some(PointKind::Corner) {
                PointKind::Smooth
            } else {
                PointKind::Corner
            };
            if document
                .piece_mut(piece)
                .is_some_and(|p| p.outline.set_point_kind(anchor, kind))
            {
                document.touch();
                let label = if kind == PointKind::Smooth {
                    "Join lines"
                } else {
                    "Split line"
                };
                history.record(Box::new(DocumentSnapshot::new(
                    label,
                    before,
                    document.clone(),
                )));
                changes.push(DocumentChange::PieceAnnotations(piece));
            }
        }

        // --- click -----------------------------------------------------------------
        if input.clicked
            && let Some(at) = input.pointer.map(|p| world(p, camera))
        {
            if matches!(tool, Tool::SketchLine | Tool::SketchSpline) {
                let at = sketch_point(&editor.sketch_points, at, input.shift, radius);
                let close = editor.sketch_points.len() >= 3
                    && editor
                        .sketch_points
                        .first()
                        .is_some_and(|first| first.distance(at) <= radius * 2.0);
                if close || input.double_clicked {
                    if !close {
                        editor.sketch_points.push(at);
                    }
                    self.finish_sketch(document, history, editor, changes, tool);
                } else {
                    editor.sketch_points.push(at);
                }
                editor.canvas_hint = self.hint(document, editor);
                return;
            }
            let line = pick_line(document, at, radius * 1.5);
            match tool {
                Tool::Select | Tool::Points | Tool::Pull | Tool::Curve => {
                    editor.selected_edge = line.map(|l| l.2);
                    editor.selected_piece = line.map(|l| l.0).or_else(|| document.pick_piece(at));
                }
                Tool::AddPoint => {
                    if let Some(marker) = self.marker(document, at, radius) {
                        let before = document.clone();
                        if document
                            .insert_point(marker.piece, marker.edge, marker.t)
                            .is_some()
                        {
                            history.record(Box::new(DocumentSnapshot::new(
                                "Add point",
                                before,
                                document.clone(),
                            )));
                            editor.selected_piece = Some(marker.piece);
                            editor.selected_edge = Some(EdgeRef::new(marker.piece, marker.edge));
                            changes.push(DocumentChange::Everything);
                        }
                    }
                }
                Tool::Sew => {
                    self.sew_click(document, history, editor, changes, line, at, input.shift)
                }
                Tool::SketchLine | Tool::SketchSpline => {}
                Tool::Button | Tool::Buttonhole => {
                    let Some(piece_id) = line.map(|l| l.0).or_else(|| document.pick_piece(at))
                    else {
                        return;
                    };
                    let before = document.clone();
                    if let Some(piece) = document.piece_mut(piece_id) {
                        if tool == Tool::Button {
                            piece.buttons.push(cad_core::Button::new(at, 18.0));
                        } else {
                            piece
                                .buttonholes
                                .push(cad_core::Buttonhole::for_button(at, 18.0));
                        }
                        document.touch();
                        editor.selected_piece = Some(piece_id);
                        history.record(Box::new(DocumentSnapshot::new(
                            if tool == Tool::Button {
                                "Add button"
                            } else {
                                "Add buttonhole"
                            },
                            before,
                            document.clone(),
                        )));
                        changes.push(DocumentChange::PieceAnnotations(piece_id));
                    }
                }
            }
        }

        // --- hover -----------------------------------------------------------------
        let pointer = input
            .pointer
            .filter(|_| input.hovered)
            .map(|p| world(p, camera));
        self.hovered = match (&self.drag, pointer) {
            (Some(drag), _) => Some(drag.handle),
            (None, Some(at)) if tool == Tool::Points => editor
                .selected_piece
                .and_then(|piece| pick_handle_in(document, piece, at, radius)),
            _ => None,
        };
        self.hovered_line = pointer
            .and_then(|at| pick_line(document, at, radius * 1.5))
            .map(|(piece, span, _)| (piece, span));
        match tool {
            Tool::SketchLine | Tool::SketchSpline => {
                editor.sketch_preview =
                    pointer.map(|at| sketch_point(&editor.sketch_points, at, input.shift, radius));
            }
            Tool::AddPoint => {
                editor.point_marker = pointer.and_then(|at| self.marker(document, at, radius));
            }
            _ => {}
        }
        editor.canvas_hint = self.hint(document, editor);
    }

    /// Where the add-point tool puts a point for the cursor at `at`: on the
    /// nearest line, snapped to its middle when close to it.
    fn marker(&self, document: &PatternDocument, at: Vec2, radius: f32) -> Option<PointMarker> {
        let edge = document.pick_edge(at, radius * 1.5)?;
        let outline = &document.piece(edge.piece)?.outline;
        let span = outline.line_span(edge.edge);
        let length = outline.span_length(span);
        let mut t = outline.nearest_on_edge(edge.edge, at);
        let mut on = edge.edge;
        let mut fraction = outline.span_fraction(span, outline.curve(on).eval(t));
        let middle = ((fraction - 0.5) * length).abs() <= radius * 1.5;
        if middle {
            (on, t) = outline.span_location(span, 0.5);
            fraction = 0.5;
        }
        Some(PointMarker {
            piece: edge.piece,
            edge: on,
            t,
            position: outline.curve(on).eval(t),
            from_start: fraction * length,
            to_end: (1.0 - fraction) * length,
            middle,
        })
    }

    fn finish_sketch(
        &mut self,
        document: &mut PatternDocument,
        history: &mut History,
        editor: &mut EditorState,
        changes: &mut Vec<DocumentChange>,
        tool: Tool,
    ) {
        let points = std::mem::take(&mut editor.sketch_points);
        if points.len() < 3 {
            return;
        }
        let Some(fabric) = document.fabrics().first().map(|f| f.id) else {
            return;
        };
        let before = document.clone();
        let control_points = points
            .into_iter()
            .map(|position| ControlPoint {
                position,
                kind: PointKind::Corner,
            })
            .collect::<Vec<_>>();
        let mut outline = Outline::new(
            control_points.clone(),
            vec![Segment::Line; control_points.len()],
        );
        if tool == Tool::SketchSpline {
            outline = outline.simplified(1.0, 18.0);
        }
        let id = document.add_piece("Sketched piece", outline, fabric);
        editor.selected_piece = Some(id);
        editor.selected_edge = None;
        history.record(Box::new(DocumentSnapshot::new(
            if tool == Tool::SketchSpline {
                "Draw spline"
            } else {
                "Draw polygon"
            },
            before,
            document.clone(),
        )));
        changes.push(DocumentChange::Everything);
    }

    fn sew_click(
        &mut self,
        document: &mut PatternDocument,
        history: &mut History,
        editor: &mut EditorState,
        changes: &mut Vec<DocumentChange>,
        line: Option<(PieceId, EdgeSpan, EdgeRef)>,
        at: Vec2,
        shift: bool,
    ) {
        let Some((piece, span, edge)) = line else {
            // A click into nothing starts over.
            editor.sew_a = None;
            editor.sew_b = None;
            return;
        };
        editor.selected_piece = Some(piece);
        editor.selected_edge = Some(edge);
        if is_sewn(document, piece, span) {
            return; // already sewn: just selected (remove it in the panel)
        }
        let pick = SewPick { piece, span, at };
        let edge_count = document.piece(piece).map_or(0, |p| p.outline.edge_count());
        // Shift adds a neighbouring line to the side being picked.
        let extend = |side: &mut SewPick| {
            if side.piece == piece
                && let Some(union) = side.span.union(&span, edge_count)
            {
                side.span = union;
                return true;
            }
            false
        };
        let Some(a) = &mut editor.sew_a else {
            editor.sew_a = Some(pick);
            return;
        };
        match &mut editor.sew_b {
            None => {
                if shift && extend(a) {
                    return;
                }
                if overlaps(document, a, &pick) {
                    return;
                }
                editor.sew_b = Some(pick);
                if shift {
                    return; // more lines for the second side may follow
                }
            }
            Some(b) => {
                if !extend(b) {
                    *b = pick;
                }
                if shift {
                    return;
                }
            }
        }
        let (a, b) = (editor.sew_a.take(), editor.sew_b.take());
        if let (Some(a), Some(b)) = (a, b)
            && sew(document, history, a, b).is_some()
        {
            changes.push(DocumentChange::Seams);
        }
    }

    /// What the active tool expects next.
    fn hint(&self, document: &PatternDocument, editor: &EditorState) -> String {
        let length = |pick: &SewPick| {
            document
                .piece(pick.piece)
                .map_or(0.0, |p| p.outline.span_length(pick.span))
        };
        match editor.canvas_tool {
            Tool::Select => "Drag a piece to move it · click a line to select it · Del deletes the piece".to_owned(),
            Tool::Points if editor.selected_piece.is_none() => "Click a piece to edit its points".to_owned(),
            Tool::Points => {
                "Drag points and curve handles · squares are turn points, dots are curve points · double-click a square to join the two lines there into one, a dot to split"
                    .to_owned()
            }
            Tool::Sew => match (&editor.sew_a, &editor.sew_b) {
                (None, _) => {
                    "Sewing: click the first line near the end where the seam starts · a sewn line + Del takes its seam out · Ctrl+Z undoes".to_owned()
                }
                (Some(a), None) => format!(
                    "First line {:.0} mm · Shift+click adds a neighbouring line · now click the matching end of the second line · Esc cancels",
                    length(a)
                ),
                (Some(a), Some(b)) => format!(
                    "{:.0} mm to {:.0} mm · Shift+click adds lines · click once more (without Shift) to sew",
                    length(a),
                    length(b)
                ),
            },
            Tool::SketchLine => format!("Click the corners of a new piece · type a length (and \", angle\") + Enter for an exact side · Shift snaps to 15° · Enter, double-click or the first point closes it · Esc cancels ({} points)", editor.sketch_points.len()),
            Tool::SketchSpline => format!("Click points along a curved piece · type a length + Enter for an exact step · Enter or double-click finishes and fits Bézier curves · Esc cancels ({} points)", editor.sketch_points.len()),
            Tool::AddPoint => "Click a line to add a turn point there, splitting it in two · near the middle it snaps to the exact middle · Ctrl+Z undoes".to_owned(),
            Tool::Curve => "Drag a line to bend it into a curve · double-click a curve to straighten it · Ctrl+Z undoes".to_owned(),
            Tool::Button => "Click a piece to place a button · edit or remove it in Pattern properties".to_owned(),
            Tool::Buttonhole => "Click a piece to place a buttonhole · edit or remove it in Pattern properties".to_owned(),
            Tool::Pull => "Drag a piece to pull it around the flat layout · Ctrl+Z undoes".to_owned(),
        }
    }
}

/// The handle of one piece nearest to `at` within `radius`.
fn pick_handle_in(
    document: &PatternDocument,
    piece: PieceId,
    at: Vec2,
    radius: f32,
) -> Option<HandleRef> {
    let p = document.piece(piece)?;
    p.outline
        .handles()
        .map(|(kind, position)| (kind, position.distance(at)))
        .filter(|(_, distance)| *distance <= radius)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(kind, _)| HandleRef { piece, kind })
}

/// The 3D viewport tool: orbit/pan/dolly and click-to-select through the
/// 3D → 2D mapping.
pub fn update_scene(
    input: &ViewportInput,
    camera: &mut OrbitCamera,
    topology: &impl TopologySync,
    editor: &mut EditorState,
) {
    let viewport = Vec2::new(input.size[0] as f32, input.size[1] as f32);
    if input.primary.dragging {
        camera.orbit(input.drag_delta);
    }
    if input.secondary.dragging || input.middle.dragging {
        camera.pan(input.drag_delta, viewport.y);
    }
    if input.zoom != 1.0 {
        camera.dolly(input.zoom);
    }
    if input.clicked
        && let Some(pointer) = input.pointer
    {
        let (origin, direction) = camera.ray(pointer, viewport);
        editor.selected_piece = pick_piece(topology, topology.mesh(), origin, direction);
    }
}

fn pick_piece(
    topology: &impl TopologySync,
    mesh: &GarmentMesh,
    origin: glam::Vec3,
    direction: glam::Vec3,
) -> Option<cad_core::PieceId> {
    let hit = mesh.raycast(origin, direction)?;
    let location = SurfaceLocation {
        triangle: hit.triangle,
        barycentric: hit.barycentric,
    };
    // `surface_to_pattern` also yields the exact pattern-space point under
    // the cursor: the hook for 3D tools such as drawing internal lines.
    topology.surface_to_pattern(location).map(|at| at.piece)
}

pub fn surface_hit(
    topology: &impl TopologySync,
    origin: glam::Vec3,
    direction: glam::Vec3,
) -> Option<(SurfaceLocation, glam::Vec3)> {
    let hit = topology.mesh().raycast(origin, direction)?;
    let triangle = topology.mesh().triangle(hit.triangle);
    let position = triangle
        .into_iter()
        .zip([hit.barycentric.x, hit.barycentric.y, hit.barycentric.z])
        .map(|(vertex, weight)| topology.mesh().positions[vertex as usize] * weight)
        .reduce(|a, b| a + b)?;
    Some((
        SurfaceLocation {
            triangle: hit.triangle,
            barycentric: hit.barycentric,
        },
        position,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_core::geometry::Curve;

    fn click(point: Vec2, shift: bool) -> ViewportInput {
        ViewportInput {
            size: [1200, 1100],
            hovered: true,
            pointer: Some(point),
            clicked: true,
            zoom: 1.0,
            shift,
            ..Default::default()
        }
    }

    /// A click on `edge` of `piece`, at fraction `t` along it.
    fn click_edge(
        document: &PatternDocument,
        camera: &Camera2D,
        piece: PieceId,
        edge: u32,
        t: f32,
        shift: bool,
    ) -> ViewportInput {
        let viewport = Vec2::new(1200.0, 1100.0);
        let point = document.piece(piece).unwrap().outline.curve(edge).eval(t);
        click(camera.world_to_screen(point, viewport), shift)
    }

    /// The demo top's front, a tool and the camera the clicks are made with.
    fn setup(tool: Tool) -> (PatternDocument, PieceId, EditorState, Camera2D) {
        let document = PatternDocument::demo_top();
        let front = document.pieces()[0].id;
        let editor = EditorState {
            canvas_tool: tool,
            ..Default::default()
        };
        let mut camera = Camera2D::default();
        let (lo, hi) = document.piece(front).unwrap().outline.bounds();
        camera.fit(lo, hi, Vec2::new(1200.0, 1100.0));
        (document, front, editor, camera)
    }

    #[test]
    fn clicking_near_the_middle_of_a_line_cuts_it_exactly_in_half() {
        let (mut document, front, mut editor, mut camera) = setup(Tool::AddPoint);
        let (mut tool, mut history, mut changes) = (CanvasTool::default(), History::default(), Vec::new());
        let outline = document.piece(front).unwrap().outline.clone();
        let span = outline.line_span(1);
        let length = outline.span_length(span);
        let lines = outline.lines().len();
        // A little off the middle: snaps to it.
        let (edge, t) = outline.span_location(span, 0.505);
        let viewport = Vec2::new(1200.0, 1100.0);
        let input = click(camera.world_to_screen(outline.curve(edge).eval(t), viewport), false);
        tool.update(&input, &mut camera, &mut document, &mut history, &mut editor, &mut changes);

        let outline = &document.piece(front).unwrap().outline;
        assert_eq!(outline.lines().len(), lines + 1);
        let first = outline.line_span(span.first);
        assert!((outline.span_length(first) - length / 2.0).abs() < 0.2, "{} of {length}", outline.span_length(first));
        assert_eq!(history.undo_label(), Some("Add point"));
    }

    #[test]
    fn dragging_a_line_with_the_curve_tool_bends_it() {
        let (mut document, front, mut editor, mut camera) = setup(Tool::Curve);
        let (mut tool, mut history, mut changes) = (CanvasTool::default(), History::default(), Vec::new());
        let viewport = Vec2::new(1200.0, 1100.0);
        let outline = document.piece(front).unwrap().outline.clone();
        let edge = (0..outline.edge_count())
            .find(|&e| outline.segments()[e as usize] == cad_core::Segment::Line)
            .unwrap();
        let grab = outline.curve(edge).eval(0.5);
        let start = camera.world_to_screen(grab, viewport);
        let press = ViewportInput {
            size: [1200, 1100],
            hovered: true,
            pointer: Some(start),
            press_origin: Some(start),
            primary: app_ui::ButtonDrag { started: true, dragging: true, stopped: false },
            zoom: 1.0,
            ..Default::default()
        };
        tool.update(&press, &mut camera, &mut document, &mut history, &mut editor, &mut changes);
        let to = start + Vec2::new(0.0, -40.0);
        let drag = ViewportInput {
            pointer: Some(to),
            primary: app_ui::ButtonDrag { started: false, dragging: true, stopped: false },
            ..press
        };
        tool.update(&drag, &mut camera, &mut document, &mut history, &mut editor, &mut changes);
        let release = ViewportInput {
            primary: app_ui::ButtonDrag { started: false, dragging: false, stopped: true },
            ..drag
        };
        tool.update(&release, &mut camera, &mut document, &mut history, &mut editor, &mut changes);

        let bent = &document.piece(front).unwrap().outline;
        assert!(matches!(bent.segments()[edge as usize], cad_core::Segment::Cubic { .. }));
        let target = camera.screen_to_world(to, viewport);
        assert!(bent.curve(edge).eval(0.5).distance(target) < 0.5);
        assert_eq!(history.undo_label(), Some("Bend line"));
    }

    #[test]
    fn sketch_lines_snap_straight_and_to_15_degrees_with_shift() {
        let points = [Vec2::ZERO];
        let level = sketch_point(&points, Vec2::new(100.0, 2.0), false, 1.0);
        assert!((level - Vec2::new(100.02, 0.0)).length() < 0.01, "{level}");
        let free = sketch_point(&points, Vec2::new(100.0, 30.0), false, 1.0);
        assert_eq!(free, Vec2::new(100.0, 30.0));
        let stepped = sketch_point(&points, Vec2::new(100.0, 30.0), true, 1.0);
        assert!((stepped.y.atan2(stepped.x).to_degrees() - 15.0).abs() < 0.01);
    }

    #[test]
    fn a_line_cut_into_three_gets_three_equal_parts() {
        let mut document = PatternDocument::demo_top();
        let front = document.pieces()[0].id;
        let outline = document.piece(front).unwrap().outline.clone();
        // The longest line, curved or not.
        let span = outline
            .lines()
            .into_iter()
            .max_by(|a, b| outline.span_length(*a).total_cmp(&outline.span_length(*b)))
            .unwrap();
        let length = outline.span_length(span);
        assert!(split_line(&mut document, front, span.first, 3));
        let outline = &document.piece(front).unwrap().outline;
        let mut anchor_edge = span.first;
        for _ in 0..3 {
            let part = outline.line_span(anchor_edge);
            assert!((outline.span_length(part) - length / 3.0).abs() < 0.3);
            anchor_edge = (part.first + part.count) % outline.edge_count();
        }
    }

    #[test]
    fn sewing_two_lines_takes_two_clicks_and_follows_the_clicked_ends() {
        let mut document = PatternDocument::demo_top();
        for seam in document.seams().iter().map(|s| s.id).collect::<Vec<_>>() {
            document.remove_seam(seam);
        }
        let (front, back) = (document.pieces()[0].id, document.pieces()[1].id);
        let mut editor = EditorState {
            canvas_tool: Tool::Sew,
            ..Default::default()
        };
        let (mut tool, camera, mut history, mut changes) = (
            CanvasTool::default(),
            Camera2D::default(),
            History::default(),
            Vec::new(),
        );
        let mut run = |input: ViewportInput,
                       document: &mut PatternDocument,
                       editor: &mut EditorState,
                       history: &mut History| {
            tool.update(
                &input,
                &mut camera.clone(),
                document,
                history,
                editor,
                &mut changes,
            );
        };

        // Front's right side near its start (bottom), back's left side near
        // its end (bottom): the seam must not be twisted.
        run(
            click_edge(&document, &camera, front, 1, 0.1, false),
            &mut document,
            &mut editor,
            &mut history,
        );
        assert!(editor.sew_a.is_some());
        run(
            click_edge(&document, &camera, back, 7, 0.9, false),
            &mut document,
            &mut editor,
            &mut history,
        );
        assert!(editor.sew_a.is_none() && editor.sew_b.is_none());
        assert_eq!(document.seams().len(), 1);
        let seam = document.seams()[0];
        assert_eq!(
            (seam.a, seam.b),
            (EdgeRef::new(front, 1), EdgeRef::new(back, 7))
        );
        assert!(
            seam.reversed,
            "the start of the front's side meets the end of the back's"
        );
        assert_eq!(history.undo_label(), Some("Sew"));

        // A sewn line cannot be sewn again.
        run(
            click_edge(&document, &camera, front, 1, 0.5, false),
            &mut document,
            &mut editor,
            &mut history,
        );
        assert!(editor.sew_a.is_none());
    }

    #[test]
    fn a_seam_is_undone_and_redone() {
        let mut document = PatternDocument::demo_top();
        for seam in document.seams().iter().map(|s| s.id).collect::<Vec<_>>() {
            document.remove_seam(seam);
        }
        let (front, back) = (document.pieces()[0].id, document.pieces()[1].id);
        let mut history = History::default();
        let at = |document: &PatternDocument, piece, edge| {
            document.piece(piece).unwrap().outline.curve(edge).eval(0.1)
        };
        let a = SewPick {
            piece: front,
            span: EdgeSpan::single(1),
            at: at(&document, front, 1),
        };
        let b = SewPick {
            piece: back,
            span: EdgeSpan::single(7),
            at: at(&document, back, 7),
        };
        sew(&mut document, &mut history, a, b).unwrap();
        assert_eq!(document.seams().len(), 1);
        history.undo(&mut document).unwrap().unwrap();
        assert!(document.seams().is_empty(), "the stitch is gone");
        history.redo(&mut document).unwrap().unwrap();
        assert_eq!(document.seams().len(), 1);
    }

    #[test]
    fn double_clicking_a_turn_point_joins_its_two_lines() {
        let mut document = PatternDocument::demo_top();
        let front = document.pieces()[0].id;
        let lines_before = document.piece(front).unwrap().outline.lines().len();
        let mut editor = EditorState {
            canvas_tool: Tool::Points,
            selected_piece: Some(front),
            ..Default::default()
        };
        let (mut tool, camera, mut history, mut changes) = (
            CanvasTool::default(),
            Camera2D::default(),
            History::default(),
            Vec::new(),
        );
        // Anchor 1: the corner between the hem and the right side.
        let point = document.piece(front).unwrap().outline.points()[1].position;
        let input = ViewportInput {
            double_clicked: true,
            ..click(
                camera.world_to_screen(point, Vec2::new(1200.0, 1100.0)),
                false,
            )
        };
        tool.update(
            &input,
            &mut camera.clone(),
            &mut document,
            &mut history,
            &mut editor,
            &mut changes,
        );
        let outline = &document.piece(front).unwrap().outline;
        assert_eq!(outline.lines().len(), lines_before - 1);
        assert_eq!(outline.line_span(0), EdgeSpan { first: 0, count: 2 });
        assert_eq!(history.undo_label(), Some("Join lines"));
    }

    #[test]
    fn shift_click_sews_several_lines_to_one() {
        let mut document = PatternDocument::demo_top();
        for seam in document.seams().iter().map(|s| s.id).collect::<Vec<_>>() {
            document.remove_seam(seam);
        }
        let (front, back) = (document.pieces()[0].id, document.pieces()[1].id);
        let mut editor = EditorState {
            canvas_tool: Tool::Sew,
            ..Default::default()
        };
        let (mut tool, camera, mut history, mut changes) = (
            CanvasTool::default(),
            Camera2D::default(),
            History::default(),
            Vec::new(),
        );
        for input in [
            click_edge(&document, &camera, front, 2, 0.1, false),
            click_edge(&document, &camera, front, 3, 0.5, true),
            click_edge(&document, &camera, back, 4, 0.5, false),
        ] {
            tool.update(
                &input,
                &mut camera.clone(),
                &mut document,
                &mut history,
                &mut editor,
                &mut changes,
            );
        }
        let seam = document.seams()[0];
        assert_eq!((seam.a_count, seam.b_count), (2, 1));
        assert!(changes.contains(&DocumentChange::Seams));
    }
}

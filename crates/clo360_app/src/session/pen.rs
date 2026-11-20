//! The 3D pen in the session: drawing lines on the avatar, selecting them,
//! and flattening the areas they close into pattern pieces.

use std::sync::Arc;

use app_ui::{SceneTool, ViewportInput};
use avatar::drawing;
use cad_core::{DocumentChange, DocumentSnapshot, FlatEdge, SkinPoint};
use glam::{Mat4, Vec2, Vec3};

use super::Session;
use crate::controllers;

/// The pen snaps to points and lines within this many pixels.
const SNAP_PX: f32 = 10.0;
/// A mirrored point this close to a drawn point or line joins it, metres.
const ATTACH: f32 = 0.004;
/// Lines lie this far off the skin so they are not hidden in it, metres.
const LIFT: f32 = 0.0012;

/// Traced lines and the like, kept while the document and body stay.
#[derive(Default)]
pub(super) struct PenCache {
    key: Option<(u64, avatar::BodyParams, avatar::PoseParams)>,
    traced: Vec<drawing::Traced>,
    adjacency: Arc<Vec<[u32; 3]>>,
    /// Where the pen points on the skin.
    hover: Option<SkinPoint>,
    /// The area under the pointer for Flatten: its seed triangle and
    /// document revision, and its boundary (or why there is none).
    area: Option<(u32, u64, Result<Vec<Vec3>, String>)>,
    /// Whether the garment was hidden in the last frame.
    pub(super) hid_garment: bool,
    /// A line point being dragged with Edit lines.
    drag: Option<PointDrag>,
}

/// A drawn point being moved along the body, with the points at the same
/// spot (where lines meet) and their mirror images.
struct PointDrag {
    /// (line, point index, is a mirror image).
    targets: Vec<(u32, usize, bool)>,
    /// The point lies on the centre line: it stays on it.
    centre: bool,
    before: cad_core::PatternDocument,
    moved: bool,
}

/// How far beside the body the pen still draws (on the skin nearest to
/// the pointer), metres.
const BESIDE: f32 = 0.12;

/// Where a click of the pen lands.
enum Snap {
    /// On a point of a drawn line (or of the line being drawn).
    Point(SkinPoint),
    /// On a drawn line between its points: a point is added to it there.
    OnLine {
        line: u32,
        at: usize,
        point: SkinPoint,
    },
    Skin(SkinPoint),
}

impl Snap {
    fn point(&self) -> SkinPoint {
        match self {
            Snap::Point(p) | Snap::Skin(p) | Snap::OnLine { point: p, .. } => *p,
        }
    }
}

impl Session {
    /// Whether the garment is hidden to draw on the avatar.
    pub fn garment_hidden(&self) -> bool {
        matches!(
            self.editor.pen.tool,
            SceneTool::Pen | SceneTool::EditLines | SceneTool::Flatten
        )
    }

    /// Traces the drawn lines over the current body when anything changed.
    fn refresh_pen(&mut self) -> bool {
        let Some(body) = &self.body else { return false };
        let Some(collider) = body.collider.clone() else {
            return false;
        };
        let key = (self.document.revision(), body.params, body.pose);
        if self.pen_cache.key != Some(key) {
            self.pen_cache.key = Some(key);
            self.pen_cache.traced = self
                .document
                .drawing()
                .lines
                .iter()
                .map(|line| drawing::trace_line(&body.mesh, &collider, line))
                .collect();
            self.pen_cache.area = None;
        }
        if self.pen_cache.adjacency.len() != body.mesh.skin().len() {
            self.pen_cache.adjacency = Arc::new(drawing::adjacency(&body.mesh));
        }
        true
    }

    /// Screen position of `p` (pixels), if in front of the camera and on the
    /// side of the body facing it.
    fn on_screen(
        view_proj: &Mat4,
        eye: Vec3,
        viewport: Vec2,
        p: Vec3,
        normal: Vec3,
    ) -> Option<Vec2> {
        if (eye - p).dot(normal) <= 0.0 {
            return None;
        }
        let ndc = view_proj.project_point3(p);
        (ndc.z > 0.0 && ndc.z < 1.0).then(|| {
            Vec2::new(
                (ndc.x + 1.0) * 0.5 * viewport.x,
                (1.0 - ndc.y) * 0.5 * viewport.y,
            )
        })
    }

    /// Where the pointer is on the avatar, snapped to the drawn points and
    /// lines near it.
    fn pen_snap(&self, pointer: Vec2, viewport: Vec2, lines_only: bool) -> Option<Snap> {
        let body = self.body.as_ref()?;
        let collider = body.collider.as_ref()?;
        let mesh = &body.mesh;
        let view_proj = self.scene_camera.view_proj(viewport.x / viewport.y);
        let eye = self.scene_camera.eye();
        let near = |p: &SkinPoint| {
            let at = drawing::position(mesh, p)?;
            let screen = Self::on_screen(&view_proj, eye, viewport, at, drawing::normal(mesh, p))?;
            Some(screen.distance(pointer))
        };
        if !lines_only {
            let points = self
                .document
                .drawing()
                .lines
                .iter()
                .flat_map(|l| l.points.iter())
                .chain(&self.editor.pen.drawing);
            let best = points
                .filter_map(|p| Some((near(p)?, *p)))
                .filter(|(d, _)| *d <= SNAP_PX)
                .min_by(|a, b| a.0.total_cmp(&b.0));
            if let Some((_, p)) = best {
                return Some(Snap::Point(p));
            }
        }
        let mut best: Option<(f32, u32, usize, SkinPoint)> = None;
        for traced in &self.pen_cache.traced {
            for (i, sample) in traced.samples.iter().enumerate() {
                if let Some(d) = near(sample)
                    && d <= SNAP_PX
                    && best.is_none_or(|b| d < b.0)
                {
                    // The line point before this sample: the new point goes
                    // after it.
                    let at = traced.points.iter().rposition(|&s| s <= i).unwrap_or(0) + 1;
                    best = Some((d, traced.line, at, *sample));
                }
            }
        }
        if let Some((_, line, at, point)) = best {
            return Some(Snap::OnLine { line, at, point });
        }
        if lines_only {
            return None;
        }
        let (origin, direction) = self.scene_camera.ray(pointer, viewport);
        drawing::pick(mesh, collider, origin, direction, BESIDE).map(Snap::Skin)
    }

    /// The drawn point under the pointer: (line, index, point).
    fn point_near(&self, pointer: Vec2, viewport: Vec2) -> Option<(u32, usize, SkinPoint)> {
        let body = self.body.as_ref()?;
        let mesh = &body.mesh;
        let view_proj = self.scene_camera.view_proj(viewport.x / viewport.y);
        let eye = self.scene_camera.eye();
        self.document
            .drawing()
            .lines
            .iter()
            .flat_map(|l| l.points.iter().enumerate().map(move |(k, p)| (l.id, k, *p)))
            .filter_map(|(line, k, p)| {
                let at = drawing::position(mesh, &p)?;
                let screen =
                    Self::on_screen(&view_proj, eye, viewport, at, drawing::normal(mesh, &p))?;
                Some((screen.distance(pointer), line, k, p))
            })
            .filter(|(d, ..)| *d <= SNAP_PX)
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, line, k, p)| (line, k, p))
    }

    /// Starts moving the drawn point `point` (of `line`, index `k`).
    fn start_point_drag(&mut self, point: SkinPoint) {
        let drawing = self.document.drawing();
        let mut targets: Vec<(u32, usize, bool)> = drawing
            .at_point(&point)
            .into_iter()
            .map(|(l, k)| (l, k, false))
            .collect();
        let mut centre = false;
        let own: Vec<(u32, usize)> = targets.iter().map(|t| (t.0, t.1)).collect();
        for (line, k) in own {
            let Some(twin) = drawing.twin(line) else {
                continue;
            };
            let Some(mirror) = drawing.line(twin).and_then(|t| t.points.get(k)).copied() else {
                continue;
            };
            if mirror.same(&point) {
                // On the centre line, shared with its own mirror image.
                centre = true;
                continue;
            }
            for (l, j) in drawing.at_point(&mirror) {
                if !targets.iter().any(|t| t.0 == l && t.1 == j) {
                    targets.push((l, j, true));
                }
            }
        }
        self.pen_cache.drag = Some(PointDrag {
            targets,
            centre,
            before: self.document.clone(),
            moved: false,
        });
    }

    /// Moves the dragged point to the skin under the pointer.
    fn drag_point(&mut self, pointer: Vec2, viewport: Vec2) {
        let Some(drag) = &self.pen_cache.drag else {
            return;
        };
        let Some(body) = &self.body else { return };
        let Some(collider) = body.collider.clone() else {
            return;
        };
        let mesh = &body.mesh;
        let (origin, direction) = self.scene_camera.ray(pointer, viewport);
        let Some(mut point) = drawing::pick(mesh, &collider, origin, direction, BESIDE) else {
            return;
        };
        let Some(mut at) = drawing::position(mesh, &point) else {
            return;
        };
        if drag.centre {
            at.x = 0.0;
            if let Some(p) = drawing::closest(mesh, &collider, at, 0.05) {
                point = p;
                at = drawing::position(mesh, &p).unwrap_or(at);
            }
        }
        let mirror = drawing::closest(mesh, &collider, Vec3::new(-at.x, at.y, at.z), 0.05);
        let targets = drag.targets.clone();
        let drawing_mut = self.document.drawing_mut();
        for (line, k, mirrored) in targets {
            let new = if mirrored {
                match mirror {
                    Some(m) => m,
                    None => continue,
                }
            } else {
                point
            };
            let Some(l) = drawing_mut.line_mut(line) else {
                continue;
            };
            let Some(slot) = l.points.get_mut(k) else {
                continue;
            };
            let old = *slot;
            *slot = new;
            // Flattened piece lines remember their ends: they move along.
            for flat in &mut drawing_mut.flat_edges {
                if flat.line == line {
                    if flat.from.same(&old) {
                        flat.from = new;
                    }
                    if flat.to.same(&old) {
                        flat.to = new;
                    }
                }
            }
        }
        if let Some(drag) = &mut self.pen_cache.drag {
            drag.moved = true;
        }
    }

    /// Removes point `k` of `line` (a line keeps at least two points, a
    /// closed one three).
    pub(super) fn remove_line_point(&mut self, line: u32, k: usize) -> bool {
        let before = self.document.clone();
        let Some(l) = self.document.drawing().line(line) else {
            return false;
        };
        let least = if l.closed { 3 } else { 2 };
        if l.points.len() <= least || k >= l.points.len() {
            self.editor.pen.status = Some(Err(
                "A line needs this point: delete the line instead".into()
            ));
            return false;
        }
        if let Some(l) = self.document.drawing_mut().line_mut(line) {
            l.points.remove(k);
        }
        self.history.record(Box::new(DocumentSnapshot::new(
            "Remove point",
            before,
            self.document.clone(),
        )));
        self.editor.pen.selected_point = None;
        true
    }

    /// Edit lines: drag points, click to select, double-click to add or
    /// remove a point.
    fn edit_lines(&mut self, input: &ViewportInput, pointer: Vec2, viewport: Vec2) {
        if input.primary.dragging && self.pen_cache.drag.is_some() {
            self.drag_point(pointer, viewport);
        }
        if input.primary.stopped
            && let Some(drag) = self.pen_cache.drag.take()
            && drag.moved
        {
            self.history.record(Box::new(DocumentSnapshot::new(
                "Move line point",
                drag.before,
                self.document.clone(),
            )));
        }
        if input.double_clicked {
            match self.point_near(pointer, viewport) {
                Some((line, k, _)) => {
                    self.remove_line_point(line, k);
                }
                None => {
                    if let Some(Snap::OnLine { line, at, point }) =
                        self.pen_snap(pointer, viewport, true)
                    {
                        let before = self.document.clone();
                        if let Some(l) = self.document.drawing_mut().line_mut(line) {
                            l.points.insert(at.min(l.points.len()), point);
                        }
                        self.history.record(Box::new(DocumentSnapshot::new(
                            "Add point to line",
                            before,
                            self.document.clone(),
                        )));
                        self.editor.pen.selected_line = Some(line);
                        self.editor.pen.selected_point = Some(at);
                    }
                }
            }
        } else if input.clicked {
            match self.point_near(pointer, viewport) {
                Some((line, k, _)) => {
                    self.editor.pen.selected_line = Some(line);
                    self.editor.pen.selected_point = Some(k);
                }
                None => {
                    self.editor.pen.selected_point = None;
                    self.editor.pen.selected_line = match self.pen_snap(pointer, viewport, true) {
                        Some(Snap::OnLine { line, .. }) => Some(line),
                        _ => None,
                    };
                }
            }
        }
    }

    /// One frame of the 3D view with a pen tool.
    pub(super) fn update_scene_pen(
        &mut self,
        input: &ViewportInput,
        changes: &mut Vec<DocumentChange>,
    ) {
        if self.imported.is_some() || !self.refresh_pen() {
            self.editor.pen.status = Some(Err("Drawing needs the built-in avatar".into()));
            return;
        }
        let viewport = Vec2::new(input.size[0] as f32, input.size[1] as f32);
        // A press on a drawn point with Edit lines moves the point, not the
        // camera.
        if self.editor.pen.tool == SceneTool::EditLines
            && input.primary.started
            && let Some(origin) = input.press_origin.or(input.pointer)
            && let Some((line, k, point)) = self.point_near(origin, viewport)
        {
            self.editor.pen.selected_line = Some(line);
            self.editor.pen.selected_point = Some(k);
            self.start_point_drag(point);
        }
        if self.editor.pen.tool != SceneTool::EditLines {
            self.pen_cache.drag = None;
        }
        // The camera still turns, pans and zooms.
        let dragging_point = self.pen_cache.drag.is_some();
        let camera_only = ViewportInput {
            clicked: false,
            double_clicked: false,
            primary: if dragging_point {
                Default::default()
            } else {
                input.primary
            },
            ..*input
        };
        controllers::update_scene(
            &camera_only,
            &mut self.scene_camera,
            &self.topology,
            &mut self.editor,
            &mut self.document,
        );
        if std::mem::take(&mut self.editor.pen.cancel) {
            self.editor.pen.drawing.clear();
        }
        if std::mem::take(&mut self.editor.pen.finish) {
            self.finish_pen_line(false);
        }
        let Some(pointer) = input.pointer.filter(|_| input.hovered || dragging_point) else {
            self.pen_cache.hover = None;
            return;
        };
        match self.editor.pen.tool {
            SceneTool::Pen => {
                let snap = self.pen_snap(pointer, viewport, false);
                self.pen_cache.hover = snap.as_ref().map(Snap::point);
                if input.clicked
                    && let Some(snap) = snap
                {
                    self.pen_click(snap, input.double_clicked);
                }
            }
            SceneTool::EditLines => self.edit_lines(input, pointer, viewport),
            SceneTool::Flatten => {
                let snap = self.pen_snap(pointer, viewport, true).is_none().then(|| {
                    let (origin, direction) = self.scene_camera.ray(pointer, viewport);
                    let body = self.body.as_ref()?;
                    drawing::pick(
                        &body.mesh,
                        body.collider.as_ref()?,
                        origin,
                        direction,
                        BESIDE,
                    )
                });
                let seed = snap.flatten();
                self.pen_cache.hover = seed;
                if let Some(seed) = seed {
                    let revision = self.document.revision();
                    let fresh = self
                        .pen_cache
                        .area
                        .as_ref()
                        .is_some_and(|(t, r, _)| *t == seed.triangle && *r == revision);
                    if !fresh {
                        let body = self.body.as_ref().expect("refreshed above");
                        let found = drawing::find_area(
                            &body.mesh,
                            body.collider.as_ref().expect("refreshed above"),
                            &self.pen_cache.adjacency,
                            &self.document.drawing().lines,
                            seed,
                        )
                        .map(|area| area.outline());
                        self.pen_cache.area = Some((seed.triangle, revision, found));
                    }
                    if input.clicked {
                        self.flatten_at(seed, changes);
                    }
                } else {
                    self.pen_cache.area = None;
                }
            }
            SceneTool::Orbit | SceneTool::Place => {}
        }
    }

    fn pen_click(&mut self, snap: Snap, double: bool) {
        let joined = !matches!(snap, Snap::Skin(_));
        let point = match snap {
            Snap::OnLine { line, at, point } => {
                // A point on the line there, for the new line to meet it.
                let before = self.document.clone();
                if let Some(l) = self.document.drawing_mut().line_mut(line) {
                    let at = at.min(l.points.len());
                    l.points.insert(at, point);
                }
                self.history.record(Box::new(DocumentSnapshot::new(
                    "Add point to line",
                    before,
                    self.document.clone(),
                )));
                point
            }
            Snap::Point(p) | Snap::Skin(p) => p,
        };
        let pen = &mut self.editor.pen;
        // The second click of a double-click lands on the first one.
        let repeated = pen.drawing.last().is_some_and(|last| {
            self.body.as_ref().is_some_and(|b| {
                match (
                    drawing::position(&b.mesh, last),
                    drawing::position(&b.mesh, &point),
                ) {
                    (Some(a), Some(c)) => a.distance(c) < 0.003,
                    _ => false,
                }
            })
        });
        if double || repeated {
            if !repeated {
                pen.drawing.push(point);
            }
            self.finish_pen_line(false);
            return;
        }
        if pen.drawing.len() >= 3 && pen.drawing[0].same(&point) {
            self.finish_pen_line(true);
            return;
        }
        pen.drawing.push(point);
        if joined && pen.drawing.len() >= 2 {
            self.finish_pen_line(false);
        }
    }

    /// The point of the skin at `p`, joined to a drawn point or line within
    /// [`ATTACH`] (a point is added to the line there).
    fn attach(&mut self, p: Vec3) -> Option<SkinPoint> {
        let body = self.body.as_ref()?;
        let collider = body.collider.clone()?;
        let mesh = &body.mesh;
        let nearest_point = self
            .document
            .drawing()
            .lines
            .iter()
            .flat_map(|l| l.points.iter())
            .filter_map(|q| Some((drawing::position(mesh, q)?.distance(p), *q)))
            .filter(|(d, _)| *d < ATTACH)
            .min_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((_, q)) = nearest_point {
            return Some(q);
        }
        let mut best: Option<(f32, u32, usize, SkinPoint)> = None;
        for traced in &self.pen_cache.traced {
            for (i, (sample, at)) in traced.samples.iter().zip(&traced.positions).enumerate() {
                let d = at.distance(p);
                if d < ATTACH && best.is_none_or(|b| d < b.0) {
                    let index = traced.points.iter().rposition(|&s| s <= i).unwrap_or(0) + 1;
                    best = Some((d, traced.line, index, *sample));
                }
            }
        }
        if let Some((_, line, index, point)) = best {
            if let Some(l) = self.document.drawing_mut().line_mut(line) {
                l.points.insert(index.min(l.points.len()), point);
            }
            return Some(point);
        }
        drawing::closest(mesh, &collider, p, 0.05)
    }

    /// Ends the line being drawn (closed: back to its first point), with
    /// its mirror image on the other side of the body.
    fn finish_pen_line(&mut self, closed: bool) {
        let points = std::mem::take(&mut self.editor.pen.drawing);
        if points.len() < 2 || (closed && points.len() < 3) {
            return;
        }
        let Some(body) = &self.body else { return };
        let positions: Vec<Vec3> = points
            .iter()
            .filter_map(|p| drawing::position(&body.mesh, p))
            .collect();
        let before = self.document.clone();
        let original = self.document.drawing_mut().add_line(points.clone(), closed);
        // A line across the middle is its own mirror image.
        let (low, high) = positions
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, h), p| (l.min(p.x), h.max(p.x)));
        let crosses = low < -0.01 && high > 0.01;
        let on_centre = low > -0.008 && high < 0.008;
        let mut mirrored = false;
        if self.editor.pen.mirror && !crosses && !on_centre {
            // Traced lines now include the new one.
            self.pen_cache.key = None;
            self.refresh_pen();
            let mirror: Option<Vec<SkinPoint>> = positions
                .iter()
                .map(|p| self.attach(Vec3::new(-p.x, p.y, p.z)))
                .collect();
            if let Some(mirror) = mirror {
                let twin = self.document.drawing_mut().add_line(mirror, closed);
                if let Some(l) = self.document.drawing_mut().line_mut(twin) {
                    l.mirror_of = Some(original);
                }
                mirrored = true;
            }
        }
        self.history.record(Box::new(DocumentSnapshot::new(
            "Draw on avatar",
            before,
            self.document.clone(),
        )));
        // Say when the lines now close an area: the next step is Flatten.
        self.pen_cache.key = None;
        self.refresh_pen();
        let areas = match &self.body {
            Some(body) => body.collider.as_ref().map_or(0, |collider| {
                drawing::closed_areas(
                    &body.mesh,
                    collider,
                    &self.pen_cache.adjacency,
                    &self.document.drawing().lines,
                )
                .len()
            }),
            None => 0,
        };
        self.editor.pen.status = Some(Ok(match (areas, mirrored) {
            (0, true) => "Line drawn, and mirrored".to_owned(),
            (0, false) => "Line drawn".to_owned(),
            (n, _) => format!(
                "{n} closed area(s): pick Flatten and click inside one, or press \"Flatten all to 2D\""
            ),
        }));
    }

    /// Flattens the area at `seed` into a new piece, sewn to the pieces
    /// already flattened along its lines.
    fn flatten_at(&mut self, seed: SkinPoint, changes: &mut Vec<DocumentChange>) {
        let Some(body) = &self.body else { return };
        let Some(collider) = body.collider.clone() else {
            return;
        };
        let lines = self.document.drawing().lines.clone();
        let flat = match drawing::flatten_area(
            &body.mesh,
            &collider,
            &self.pen_cache.adjacency,
            &lines,
            seed,
        ) {
            Ok(flat) => flat,
            Err(message) => {
                self.editor.pen.status = Some(Err(message));
                return;
            }
        };
        if self.already_flat(&flat) {
            self.editor.pen.status = Some(Err("This area is a pattern piece already".into()));
            return;
        }
        let message = self.add_flattened(flat, changes);
        self.editor.pen.status = Some(message);
    }

    /// Whether a piece was flattened from exactly this area before.
    fn already_flat(&self, flat: &drawing::Flattened) -> bool {
        let edges = &self.document.drawing().flat_edges;
        self.document.pieces().iter().any(|piece| {
            let mine: Vec<&FlatEdge> = edges.iter().filter(|e| e.piece == piece.id).collect();
            !mine.is_empty()
                && mine.len() == flat.runs.len()
                && flat.runs.iter().all(|run| {
                    mine.iter().any(|e| {
                        e.line == run.line
                            && ((e.from.same(&run.from) && e.to.same(&run.to))
                                || (e.from.same(&run.to) && e.to.same(&run.from)))
                    })
                })
        })
    }

    /// Flattens every closed area that is not a piece yet.
    pub(super) fn flatten_all(&mut self, changes: &mut Vec<DocumentChange>) {
        if !self.refresh_pen() {
            return;
        }
        let Some(body) = &self.body else { return };
        let Some(collider) = body.collider.clone() else {
            return;
        };
        let lines = self.document.drawing().lines.clone();
        let seeds = drawing::closed_areas(&body.mesh, &collider, &self.pen_cache.adjacency, &lines);
        let (mut made, mut problems) = (Vec::new(), Vec::new());
        for seed in seeds {
            let Some(body) = &self.body else { return };
            match drawing::flatten_area(
                &body.mesh,
                &collider,
                &self.pen_cache.adjacency,
                &lines,
                seed,
            ) {
                Ok(flat) if self.already_flat(&flat) => {}
                Ok(flat) => match self.add_flattened(flat, changes) {
                    Ok(message) => made.push(message),
                    Err(message) => {
                        made.push(message.clone());
                        problems.push(message);
                    }
                },
                Err(message) => problems.push(message),
            }
        }
        self.editor.pen.status = Some(match (made.len(), problems.first()) {
            (0, None) => Err("No new closed areas: draw lines that meet all round an area".into()),
            (0, Some(problem)) => Err(problem.clone()),
            (n, None) => Ok(format!("{n} new piece(s) in the 2D view")),
            (n, Some(problem)) => Err(format!("{n} new piece(s) in the 2D view; {problem}")),
        });
    }

    /// Adds a flattened area as a new piece; says how it went.
    fn add_flattened(
        &mut self,
        flat: drawing::Flattened,
        changes: &mut Vec<DocumentChange>,
    ) -> Result<String, String> {
        let Some(body) = &self.body else {
            return Err("No avatar".into());
        };
        let (mesh, measurements) = (body.mesh.clone(), body.measurements);
        let before = self.document.clone();
        let document = &mut self.document;
        // To the right of everything in the 2D layout.
        let mut outline = flat.outline.clone();
        let (low, _) = outline.bounds();
        let right = document
            .pieces()
            .iter()
            .map(|p| p.outline.bounds().1.x)
            .fold(f32::NEG_INFINITY, f32::max);
        let x = if right.is_finite() { right + 60.0 } else { 0.0 };
        outline.translate(Vec2::new(x - low.x, -low.y));
        let fabric = match document.fabrics().first() {
            Some(f) => f.id,
            None => document.add_fabric("Cotton", [0.75, 0.75, 0.72]),
        };
        let count = document.pieces().len() + 1;
        let name = format!("Flattened {count}");
        let id = document.add_piece(name.clone(), outline, fabric);
        if let Some(piece) = document.piece_mut(id) {
            // The direction of the area's middle from the body's vertical
            // axis (the arrangement wraps the piece round it there).
            piece.placement.angle = flat.centre.x.atan2(flat.centre.z).to_degrees();
        }
        // Sewn to the pieces flattened next to it.
        let mut sewn = 0;
        let new_outline = document
            .piece(id)
            .map(|p| p.outline.clone())
            .expect("just added");
        for run in &flat.runs {
            let partner = document.drawing().flat_edges.iter().find_map(|e| {
                if e.line != run.line || e.piece == id {
                    return None;
                }
                let other = document.piece(e.piece)?;
                if e.edge >= other.outline.edge_count() {
                    return None;
                }
                let same = e.from.same(&run.from) && e.to.same(&run.to);
                let opposite = e.from.same(&run.to) && e.to.same(&run.from);
                (same || opposite).then(|| (e.piece, other.outline.line_span(e.edge), opposite))
            });
            if let Some((piece, span, reversed)) = partner {
                document.add_seam_spans(
                    (piece, span),
                    (id, new_outline.line_span(run.edge)),
                    reversed,
                );
                sewn += 1;
            }
        }
        for run in &flat.runs {
            document.drawing_mut().flat_edges.push(FlatEdge {
                line: run.line,
                from: run.from,
                to: run.to,
                piece: id,
                edge: run.edge,
            });
        }
        // Around the body where it was drawn: the arrangement puts it at
        // its angle; its height is then corrected to where the area is.
        avatar::arrange_around(document, &mesh, &measurements);
        if let Some(piece) = document.piece(id) {
            let (lo, hi) = piece.outline.bounds();
            let got = piece.arrangement.place((lo + hi) * 0.5).y;
            let shift = (flat.centre.y - got) * 1000.0;
            if let Some(piece) = document.piece_mut(id) {
                piece.placement.height += shift;
            }
            avatar::arrange_around(document, &mesh, &measurements);
        }
        self.history.record(Box::new(DocumentSnapshot::new(
            "Flatten",
            before,
            self.document.clone(),
        )));
        self.editor.selected_piece = Some(id);
        self.editor.selected_edge = None;
        changes.push(DocumentChange::Everything);
        self.garment_dirty = true;
        // Show the new piece in the 2D view.
        self.fit_canvas = true;
        let sewn = match sewn {
            0 => String::new(),
            1 => ", sewn to 1 line".to_owned(),
            n => format!(", sewn along {n} lines"),
        };
        if flat.stretch > 0.06 {
            Err(format!(
                "{name}: the body is too curved here to lie flat ({:.0} % stretch){sewn}. Draw a dart (two lines \
                 from the edge meeting near the fullest point) and flatten again",
                flat.stretch * 100.0
            ))
        } else {
            Ok(format!(
                "{name} is in the 2D view: {:.1} % stretch to lie flat{sewn}",
                flat.stretch * 100.0
            ))
        }
    }

    /// The skin seen straight from the front at `at` (x, y in metres).
    fn front(&self, at: Vec2) -> Option<SkinPoint> {
        let body = self.body.as_ref()?;
        drawing::hit(
            &body.mesh,
            body.collider.as_ref()?,
            Vec3::new(at.x, at.y, 2.0),
            Vec3::NEG_Z,
        )
    }

    /// A scripted line (headless runs): through points seen from the
    /// front, joined to drawn points and lines it starts or ends on.
    pub(super) fn draw_line_from_front(&mut self, points: &[Vec2], closed: bool) {
        if !self.refresh_pen() {
            return;
        }
        let mut skin = Vec::new();
        for &at in points {
            let Some(p) = self.front(at) else { continue };
            let Some(position) = self
                .body
                .as_ref()
                .and_then(|b| drawing::position(&b.mesh, &p))
            else {
                continue;
            };
            if let Some(p) = self.attach(position) {
                skin.push(p);
            }
            self.pen_cache.key = None;
            self.refresh_pen();
        }
        self.editor.pen.drawing = skin;
        self.finish_pen_line(closed);
    }

    /// A scripted flatten (headless runs) of the area seen from the front.
    pub(super) fn flatten_from_front(&mut self, at: Vec2, changes: &mut Vec<DocumentChange>) {
        if !self.refresh_pen() {
            return;
        }
        match self.front(at) {
            Some(seed) => self.flatten_at(seed, changes),
            None => self.editor.pen.status = Some(Err("Nothing of the avatar there".into())),
        }
    }

    /// The drawn lines, their points, the line being drawn and the area to
    /// flatten, as thin tubes on the skin.
    pub fn pen_props(&mut self, props: &mut gfx_render::MeshData) {
        if self.document.drawing().lines.is_empty() && self.editor.pen.tool == SceneTool::Orbit {
            return;
        }
        if self.imported.is_some() || !self.refresh_pen() {
            return;
        }
        let Some(body) = &self.body else { return };
        let mesh = &body.mesh;
        let lifted = |p: &SkinPoint| -> Option<(Vec3, Vec3)> {
            let n = drawing::normal(mesh, p);
            Some((drawing::position(mesh, p)? + n * LIFT, n))
        };
        let selected = self.editor.pen.selected_line;
        let lines = &self.document.drawing().lines;
        for traced in &self.pen_cache.traced {
            let line = lines.iter().find(|l| l.id == traced.line);
            let color = if selected == Some(traced.line) {
                [1.0, 0.45, 0.05]
            } else if line.is_some_and(|l| l.straight) {
                [0.1, 0.6, 0.25]
            } else {
                [0.05, 0.25, 0.9]
            };
            let path: Vec<(Vec3, Vec3)> = traced.samples.iter().filter_map(lifted).collect();
            tube(props, &path, 0.0011, color);
            if let Some(line) = line {
                for (k, p) in line.points.iter().enumerate() {
                    let chosen =
                        selected == Some(line.id) && self.editor.pen.selected_point == Some(k);
                    if chosen {
                        dot(props, lifted(p), 0.0042, [1.0, 0.45, 0.05]);
                    } else {
                        dot(props, lifted(p), 0.0028, [0.95, 0.95, 0.95]);
                    }
                }
            }
        }
        // The line being drawn, on to the pointer.
        let drawing = &self.editor.pen.drawing;
        if !drawing.is_empty()
            && let Some(collider) = body.collider.as_ref()
        {
            let mut points = drawing.clone();
            if let Some(hover) = self.pen_cache.hover {
                points.push(hover);
            }
            let mut path = Vec::new();
            for pair in points.windows(2) {
                path.extend(
                    drawing::trace(mesh, collider, pair[0], pair[1])
                        .iter()
                        .filter_map(lifted),
                );
            }
            tube(props, &path, 0.0013, [0.9, 0.1, 0.1]);
            for p in drawing {
                dot(props, lifted(p), 0.003, [0.9, 0.1, 0.1]);
            }
        }
        if self.editor.pen.tool == SceneTool::Pen {
            dot(
                props,
                self.pen_cache.hover.as_ref().and_then(lifted),
                0.0025,
                [0.9, 0.1, 0.1],
            );
        }
        // The area Flatten would take.
        if self.editor.pen.tool == SceneTool::Flatten
            && let Some((_, _, Ok(outline))) = &self.pen_cache.area
        {
            let mut path: Vec<(Vec3, Vec3)> = outline
                .iter()
                .filter_map(|p| {
                    let found = body.collider.as_ref()?.closest_point((*p).into(), 0.01)?;
                    let skin = drawing::skin_point(mesh, found.triangle, found.point.into());
                    let n = drawing::normal(mesh, &skin);
                    Some((*p + n * LIFT * 2.0, n))
                })
                .collect();
            if let Some(&first) = path.first() {
                path.push(first);
            }
            tube(props, &path, 0.0022, [1.0, 0.8, 0.1]);
        }
    }
}

/// A tube along `path` (points with the skin's normal there).
fn tube(out: &mut gfx_render::MeshData, path: &[(Vec3, Vec3)], radius: f32, color: [f32; 3]) {
    use gfx_render::MeshVertex;
    if path.len() < 2 {
        return;
    }
    const SIDES: u32 = 4;
    let material = Vec2::new(0.6, 0.0);
    let base = out.vertices.len() as u32;
    for (k, &(p, up)) in path.iter().enumerate() {
        let next = path[(k + 1).min(path.len() - 1)].0;
        let previous = path[k.saturating_sub(1)].0;
        let along = (next - previous).normalize_or(Vec3::X);
        let side = up.cross(along).normalize_or(Vec3::Y);
        let up = along.cross(side);
        for s in 0..SIDES {
            let angle = (s as f32 + 0.5) / SIDES as f32 * std::f32::consts::TAU;
            let normal = side * angle.cos() + up * angle.sin();
            out.vertices
                .push(MeshVertex::new(p + normal * radius, normal, material).with_color(color));
        }
    }
    for k in 0..(path.len() as u32 - 1) {
        for s in 0..SIDES {
            let a = base + k * SIDES + s;
            let b = base + k * SIDES + (s + 1) % SIDES;
            let (c, d) = (a + SIDES, b + SIDES);
            out.indices.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }
}

/// A little cube standing on the skin, for a line's point.
fn dot(out: &mut gfx_render::MeshData, at: Option<(Vec3, Vec3)>, size: f32, color: [f32; 3]) {
    let Some((p, n)) = at else { return };
    let tangent = n.any_orthonormal_vector();
    let path = [(p - n * size * 0.2, tangent), (p + n * size * 0.8, tangent)];
    tube(out, &path, size, color);
}

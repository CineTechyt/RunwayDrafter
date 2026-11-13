//! Turns the pattern document into canvas geometry.

use std::collections::HashSet;

use cad_core::geometry::Curve;
use cad_core::{ButtonStyle, EdgeRef, EdgeSpan, HandleKind, HandleRef, NotchKind, PatternDocument, PatternPiece, PieceId, PointKind};
use glam::Vec2;
use topology_sync::GarmentMesh;

use super::batch::CanvasBatch;
use crate::camera::Camera2D;
use crate::srgba;

/// A line of a piece drawn emphasised: the selection, a side picked for
/// sewing, the line under the cursor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineHighlight {
    pub piece: PieceId,
    pub span: EdgeSpan,
    /// Linear RGBA.
    pub color: [f32; 4],
    /// Marks where the seam starts on this line: at its end (`true`) or
    /// its start (`false`).
    pub start_at_end: Option<bool>,
}

/// Editor state that affects how the canvas is drawn.
#[derive(Debug, Clone, PartialEq)]
pub struct CanvasOverlay {
    pub hovered: Option<HandleRef>,
    pub active: Option<HandleRef>,
    pub selected_piece: Option<PieceId>,
    /// The piece whose points and curve handles are shown for editing.
    pub handles_for: Option<PieceId>,
    pub highlights: Vec<LineHighlight>,
    pub show_grid: bool,
    pub show_mesh: bool,
    pub show_seam_allowance: bool,
    /// In-progress points from one of the sketch tools.
    pub sketch_points: Vec<Vec2>,
}

impl Default for CanvasOverlay {
    fn default() -> Self {
        Self {
            hovered: None,
            active: None,
            selected_piece: None,
            handles_for: None,
            highlights: Vec::new(),
            show_grid: true,
            show_mesh: false,
            show_seam_allowance: true,
            sketch_points: Vec::new(),
        }
    }
}

/// One colour per seam so both sewn edges are recognisable as a pair.
const SEAM_PALETTE: [[u8; 3]; 6] =
    [[0x4c, 0xc9, 0xf0], [0xf7, 0xb8, 0x01], [0x80, 0xed, 0x99], [0xf7, 0x25, 0x85], [0xb5, 0x8c, 0xff], [0xff, 0x8c, 0x42]];

/// Rebuilds `batch` for the current frame. Line widths and handle sizes are
/// given in pixels and converted with the camera so they stay constant on
/// screen at any zoom.
pub fn build(
    batch: &mut CanvasBatch,
    doc: &PatternDocument,
    mesh: Option<&GarmentMesh>,
    camera: &Camera2D,
    viewport: Vec2,
    overlay: &CanvasOverlay,
) {
    batch.clear();
    let px = camera.pixel_size();
    if overlay.show_grid {
        grid(batch, camera, viewport);
    }
    for piece in doc.pieces() {
        let selected = overlay.selected_piece == Some(piece.id);
        if let Some(mesh) = mesh {
            fill(batch, doc, piece, mesh, selected, overlay.show_mesh, px);
        }
        annotations(batch, piece, overlay, px);
        outline(batch, doc, piece, selected, px);
    }
    seam_connectors(batch, doc, overlay, px);
    for highlight in &overlay.highlights {
        highlight_line(batch, doc, highlight, px);
    }
    if overlay.sketch_points.len() >= 2 {
        batch.polyline(&overlay.sketch_points, false, 2.5 * px, srgba(80, 220, 255, 0.9));
    }
    for point in &overlay.sketch_points {
        batch.disc(*point, 4.0 * px, srgba(80, 220, 255, 1.0));
    }
    // Handles go last so they are never hidden behind another piece.
    if let Some(piece) = overlay.handles_for.and_then(|id| doc.piece(id)) {
        handles(batch, piece, overlay, px);
    }
}

/// For the seams of the selected piece, thin lines between the points
/// that are sewn together at either end of each seam: a twisted seam shows
/// as a cross.
fn seam_connectors(batch: &mut CanvasBatch, doc: &PatternDocument, overlay: &CanvasOverlay, px: f32) {
    let Some(selected) = overlay.selected_piece else { return };
    for (index, seam) in doc.seams().iter().enumerate() {
        if seam.a.piece != selected && seam.b.piece != selected {
            continue;
        }
        let (Some(a), Some(b)) = (doc.piece(seam.a.piece), doc.piece(seam.b.piece)) else { continue };
        let ends = |piece: &PatternPiece, span: EdgeSpan| {
            let points = piece.outline.span_points(span, 1.0);
            (points[0], points[points.len() - 1])
        };
        let (a_start, a_end) = ends(a, seam.a_span());
        let (b_start, b_end) = ends(b, seam.b_span());
        let (b_first, b_last) = if seam.reversed { (b_end, b_start) } else { (b_start, b_end) };
        let [r, g, bl] = SEAM_PALETTE[index % SEAM_PALETTE.len()];
        let color = srgba(r, g, bl, 0.75);
        batch.dashed_polyline(&[a_start, b_first], false, px, 5.0 * px, 4.0 * px, color);
        batch.dashed_polyline(&[a_end, b_last], false, px, 5.0 * px, 4.0 * px, color);
        batch.disc(a_start, 3.0 * px, color);
        batch.disc(b_first, 3.0 * px, color);
    }
}

fn highlight_line(batch: &mut CanvasBatch, doc: &PatternDocument, highlight: &LineHighlight, px: f32) {
    let Some(piece) = doc.piece(highlight.piece) else { return };
    let points = piece.outline.span_points(highlight.span, 0.5 * px);
    if points.len() < 2 {
        return;
    }
    batch.polyline(&points, false, 4.5 * px, highlight.color);
    if let Some(at_end) = highlight.start_at_end {
        // The end the seam starts at, with an arrow along the line.
        let (tip, next) = if at_end {
            (points[points.len() - 1], points[points.len().saturating_sub(4).max(0)])
        } else {
            (points[0], points[3.min(points.len() - 1)])
        };
        batch.disc(tip, 6.0 * px, highlight.color);
        batch.disc(tip, 3.0 * px, srgba(20, 22, 28, 1.0));
        let direction = (next - tip).normalize_or_zero();
        let head = tip + direction * 22.0 * px;
        batch.segment(tip, head, 2.0 * px, highlight.color);
        batch.segment(head, head - direction * 7.0 * px + direction.perp() * 5.0 * px, 2.0 * px, highlight.color);
        batch.segment(head, head - direction * 7.0 * px - direction.perp() * 5.0 * px, 2.0 * px, highlight.color);
    }
}

fn grid(batch: &mut CanvasBatch, camera: &Camera2D, viewport: Vec2) {
    let px = camera.pixel_size();
    let (min, max) = camera.visible(viewport);
    // 10 mm lines fade in once they are at least ~8 px apart; 100 mm lines
    // are always shown.
    for (step, color) in [(10.0f32, srgba(255, 255, 255, 0.035)), (100.0, srgba(255, 255, 255, 0.08))] {
        if step * camera.zoom < 8.0 {
            continue;
        }
        let mut x = (min.x / step).floor() * step;
        while x <= max.x {
            batch.segment(Vec2::new(x, min.y), Vec2::new(x, max.y), px, color);
            x += step;
        }
        let mut y = (min.y / step).floor() * step;
        while y <= max.y {
            batch.segment(Vec2::new(min.x, y), Vec2::new(max.x, y), px, color);
            y += step;
        }
    }
    let axis = srgba(255, 255, 255, 0.18);
    batch.segment(Vec2::new(0.0, min.y), Vec2::new(0.0, max.y), px, axis);
    batch.segment(Vec2::new(min.x, 0.0), Vec2::new(max.x, 0.0), px, axis);
}

fn fill(
    batch: &mut CanvasBatch,
    doc: &PatternDocument,
    piece: &PatternPiece,
    mesh: &GarmentMesh,
    selected: bool,
    wireframe: bool,
    px: f32,
) {
    let Some(span) = mesh.span(piece.id) else { return };
    let positions = &mesh.pattern_positions[span.vertices.start as usize..span.vertices.end as usize];
    let indices = &mesh.indices[span.indices.start as usize..span.indices.end as usize];
    let [r, g, b] = doc.fabric(piece.fabric).map_or([0.5; 3], |f| f.base_color);
    let alpha = if selected { 0.42 } else { 0.28 };
    batch.triangles(positions, indices.iter().map(|i| i - span.vertices.start), [r, g, b, alpha]);

    if wireframe {
        let color = srgba(255, 255, 255, 0.16);
        // Interior edges are shared by two triangles; draw each edge once.
        let edges: HashSet<(u32, u32)> = indices
            .chunks_exact(3)
            .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
            .map(|(a, b)| (a.min(b), a.max(b)))
            .collect();
        for (a, b) in edges {
            batch.segment(mesh.pattern_positions[a as usize], mesh.pattern_positions[b as usize], px, color);
        }
    }
}

fn annotations(batch: &mut CanvasBatch, piece: &PatternPiece, overlay: &CanvasOverlay, px: f32) {
    if overlay.show_seam_allowance
        && let Some(cut_line) = piece.seam_allowance_outline(0.25)
    {
        batch.dashed_polyline(&cut_line, true, px, 6.0 * px, 4.0 * px, srgba(220, 220, 230, 0.55));
    }

    // Grainline: a double-headed arrow.
    let color = srgba(235, 235, 245, 0.7);
    let (tail, tip) = piece.grainline.end_points();
    let dir = piece.grainline.direction();
    batch.segment(tail, tip, 1.5 * px, color);
    for (end, sign) in [(tip, 1.0), (tail, -1.0)] {
        let back = -dir * sign * 9.0 * px;
        batch.segment(end, end + back + dir.perp() * 4.0 * px, 1.5 * px, color);
        batch.segment(end, end + back - dir.perp() * 4.0 * px, 1.5 * px, color);
    }

    let color = srgba(255, 214, 102, 1.0);
    for notch in &piece.notches {
        let Some((at, inward)) = piece.notch_frame(notch) else { continue };
        let tip = at + inward * notch.depth;
        let half = inward.perp() * notch.depth * 0.5;
        match notch.kind {
            NotchKind::Slit => batch.segment(at, tip, 1.5 * px, color),
            NotchKind::V => {
                batch.segment(at - half, tip, 1.5 * px, color);
                batch.segment(at + half, tip, 1.5 * px, color);
            }
            NotchKind::T => {
                batch.segment(at, tip, 1.5 * px, color);
                batch.segment(tip - half, tip + half, 1.5 * px, color);
            }
        }
    }

    buttons(batch, piece, px);
}

/// Buttons in their own colour with their holes and thread, buttonholes
/// as a bound slit. A ring marks buttons that are done up.
fn buttons(batch: &mut CanvasBatch, piece: &PatternPiece, px: f32) {
    let edge = srgba(12, 14, 18, 1.0);
    let fastened = srgba(110, 200, 255, 1.0);
    for button in &piece.buttons {
        let (at, r) = (button.position, button.diameter * 0.5);
        let [cr, cg, cb] = button.color;
        let color = [cr, cg, cb, 1.0];
        let shade = |k: f32| [cr * k, cg * k, cb * k, 1.0];
        if button.fastened_to.is_some() {
            batch.disc(at, r + 3.0 * px, fastened);
        }
        batch.disc(at, r + 1.0 * px, edge);
        batch.disc(at, r, color);
        let [tr, tg, tb] = button.thread_color;
        let thread = [tr, tg, tb, 1.0];
        match button.style {
            ButtonStyle::FourHole | ButtonStyle::TwoHole => {
                // The dished face inside the rim.
                batch.disc(at, r * 0.78, shade(0.8));
                let spread = r * 0.3;
                let holes: Vec<Vec2> = if button.style == ButtonStyle::FourHole {
                    [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].iter().map(|&(x, y)| at + Vec2::new(x, y) * spread * 0.75).collect()
                } else {
                    vec![at - Vec2::X * spread, at + Vec2::X * spread]
                };
                let width = (r * 0.12).max(1.0 * px);
                if holes.len() == 4 {
                    batch.segment(holes[0], holes[2], width, thread);
                    batch.segment(holes[1], holes[3], width, thread);
                } else {
                    batch.segment(holes[0], holes[1], width, thread);
                }
                for hole in holes {
                    batch.disc(hole, r * 0.1, edge);
                }
            }
            ButtonStyle::Shank => {
                // Light falls on the dome.
                batch.disc(at + Vec2::new(-0.2, 0.2) * r, r * 0.45, shade(1.35));
            }
            ButtonStyle::Tack => {
                batch.disc(at, r * 0.72, shade(0.65));
                batch.disc(at, r * 0.55, color);
                batch.disc(at, r * 0.14, shade(0.5));
            }
        }
    }
    let thread = srgba(220, 225, 235, 1.0);
    for hole in &piece.buttonholes {
        let direction = Vec2::from_angle(hole.angle);
        let half = direction * (hole.length * 0.5);
        let (a, b) = (hole.position - half, hole.position + half);
        // Satin-stitched sides with bar tacks across the ends, the cut between.
        let side = direction.perp() * 1.6;
        batch.segment(a + side, b + side, 1.4 * px.max(0.5), thread);
        batch.segment(a - side, b - side, 1.4 * px.max(0.5), thread);
        batch.segment(a - side * 1.6, a + side * 1.6, 2.0 * px.max(0.6), thread);
        batch.segment(b - side * 1.6, b + side * 1.6, 2.0 * px.max(0.6), thread);
        batch.segment(a, b, 1.0 * px, edge);
    }
}

fn outline(batch: &mut CanvasBatch, doc: &PatternDocument, piece: &PatternPiece, selected: bool, px: f32) {
    for edge in 0..piece.outline.edge_count() {
        let width = if selected { 2.5 * px } else { 1.75 * px };
        let color = {
            match doc.seam_of(EdgeRef::new(piece.id, edge)) {
                Some(seam) => {
                    let index = doc.seams().iter().position(|s| s.id == seam.id).unwrap_or(0);
                    let [r, g, b] = SEAM_PALETTE[index % SEAM_PALETTE.len()];
                    srgba(r, g, b, 1.0)
                }
                None => srgba(236, 236, 240, 1.0),
            }
        };
        batch.polyline(&piece.outline.curve(edge).flatten(0.5 * px), false, width, color);
    }
}

fn handles(batch: &mut CanvasBatch, piece: &PatternPiece, overlay: &CanvasOverlay, px: f32) {
    let tangent_color = srgba(150, 160, 180, 0.8);
    for (kind, position) in piece.outline.handles() {
        if let HandleKind::Control { segment, slot } = kind
            && let Some(anchor) = piece.outline.control_anchor(segment, slot)
        {
            batch.segment(anchor, position, px, tangent_color);
        }
    }
    for (kind, position) in piece.outline.handles() {
        let handle = Some(HandleRef { piece: piece.id, kind });
        let (emphasis, color) = if overlay.active == handle {
            (1.5, srgba(255, 160, 60, 1.0))
        } else if overlay.hovered == handle {
            (1.35, srgba(255, 220, 120, 1.0))
        } else {
            (1.0, srgba(250, 250, 255, 1.0))
        };
        let border = srgba(20, 22, 28, 1.0);
        match kind {
            // Turn points are squares, curve points (smooth) are round.
            HandleKind::Anchor(i) if piece.outline.points()[i as usize].kind == PointKind::Corner => {
                batch.square(position, 4.5 * emphasis * px, border);
                batch.square(position, 3.25 * emphasis * px, color);
            }
            HandleKind::Anchor(_) => {
                batch.disc(position, 4.5 * emphasis * px, border);
                batch.disc(position, 3.25 * emphasis * px, color);
            }
            // Curve handles are small and hollow-looking.
            HandleKind::Control { .. } => {
                batch.disc(position, 3.2 * emphasis * px, border);
                batch.disc(position, 2.0 * emphasis * px, srgba(150, 200, 255, 1.0));
            }
        }
    }
}

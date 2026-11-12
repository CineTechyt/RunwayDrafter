//! Turns the pattern document into canvas geometry.

use std::collections::HashSet;

use cad_core::geometry::Curve;
use cad_core::{EdgeRef, HandleKind, HandleRef, NotchKind, PatternDocument, PatternPiece, PieceId};
use glam::Vec2;
use topology_sync::GarmentMesh;

use super::batch::CanvasBatch;
use crate::camera::Camera2D;
use crate::srgba;

/// Editor state that affects how the canvas is drawn.
#[derive(Debug, Clone, PartialEq)]
pub struct CanvasOverlay {
    pub hovered: Option<HandleRef>,
    pub active: Option<HandleRef>,
    pub selected_piece: Option<PieceId>,
    pub selected_edge: Option<EdgeRef>,
    pub sewing_edges_a: Vec<EdgeRef>,
    pub sewing_edges_b: Vec<EdgeRef>,
    pub show_grid: bool,
    pub show_mesh: bool,
    pub show_seam_allowance: bool,
}

impl Default for CanvasOverlay {
    fn default() -> Self {
        Self {
            hovered: None,
            active: None,
            selected_piece: None,
            selected_edge: None,
            sewing_edges_a: Vec::new(),
            sewing_edges_b: Vec::new(),
            show_grid: true,
            show_mesh: false,
            show_seam_allowance: true,
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
        outline(batch, doc, piece, selected, overlay, px);
    }
    // Handles go last so they are never hidden behind another piece.
    for piece in doc.pieces() {
        handles(batch, piece, overlay, px);
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
}

fn outline(batch: &mut CanvasBatch, doc: &PatternDocument, piece: &PatternPiece, selected: bool, overlay: &CanvasOverlay, px: f32) {
    for edge in 0..piece.outline.edge_count() {
        let edge_ref = EdgeRef::new(piece.id, edge);
        let is_selected_edge = overlay.selected_edge == Some(edge_ref);
        let sewing_side_a = overlay.sewing_edges_a.contains(&edge_ref);
        let sewing_side_b = overlay.sewing_edges_b.contains(&edge_ref);
        let width = if is_selected_edge || sewing_side_a || sewing_side_b {
            4.0 * px
        } else if selected {
            2.5 * px
        } else {
            1.75 * px
        };
        let color = if sewing_side_a {
            srgba(64, 210, 135, 1.0)
        } else if sewing_side_b {
            srgba(70, 180, 245, 1.0)
        } else if is_selected_edge {
            srgba(255, 185, 40, 1.0)
        } else {
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
            HandleKind::Anchor(_) => {
                batch.square(position, 4.5 * emphasis * px, border);
                batch.square(position, 3.25 * emphasis * px, color);
            }
            HandleKind::Control { .. } => {
                batch.disc(position, 4.5 * emphasis * px, border);
                batch.disc(position, 3.25 * emphasis * px, color);
            }
        }
    }
}

//! Places pattern pieces around a body, ready for sewing.
//!
//! Sewing pulls matching edges together along straight lines, so every
//! seam line must clear the body at the start. Two things make that work
//! for any body shape:
//!
//! * each piece is wrapped around a vertical cylinder so that its side
//!   edges end up just outside the torso, inside the arms;
//! * each piece is lifted until its top edge is above the shoulders.

use cad_core::glam::Vec3;
use cad_core::{DocumentChange, PatternDocument};

use crate::measure::Measurements;
use crate::model::BodyMesh;

/// Gap between the torso's side and a piece's side edge.
const SIDE_CLEARANCE: f32 = 0.035;
/// Gap between the torso's front/back and the centre of a piece.
const DEPTH_CLEARANCE: f32 = 0.07;
/// Height of a piece's top edge above the shoulders.
const SHOULDER_CLEARANCE: f32 = 0.025;
/// Extra distance from the body per garment layer above the first, so an
/// outer garment starts outside the one it is worn over.
const LAYER_SPACING: f32 = 0.03;

/// The radius of the cylinder that brings a point `half_width` along the
/// piece to a distance `reach` from the centre plane: solves
/// `r * sin(half_width / r) = reach`. `None` if the piece is narrow enough
/// to stay flat.
fn wrap_radius(half_width: f32, reach: f32) -> Option<f32> {
    if reach >= half_width {
        return None;
    }
    // Never wrap further than a half circle.
    let (mut low, mut high) = (half_width / std::f32::consts::FRAC_PI_2, 100.0 * half_width);
    if low * (half_width / low).sin() >= reach {
        return Some(low);
    }
    for _ in 0..40 {
        let mid = 0.5 * (low + high);
        if mid * (half_width / mid).sin() < reach { low = mid } else { high = mid }
    }
    Some(0.5 * (low + high))
}

/// Re-arranges every piece of `document` around `body`. Pieces keep their
/// side of the body (front or back) and their orientation.
pub fn arrange_around(document: &mut PatternDocument, body: &BodyMesh, measurements: &Measurements) -> Vec<DocumentChange> {
    let ids: Vec<_> = document.pieces().iter().map(|p| p.id).collect();
    let mut changes = Vec::new();
    for id in ids {
        let Some(piece) = document.piece_mut(id) else { continue };
        let (low, high) = piece.outline.bounds();
        let half_width = 0.5 * (high.x - low.x) * 1e-3;
        let side = if piece.arrangement.normal().z >= 0.0 { 1.0 } else { -1.0 };
        let outset = piece.layer.saturating_sub(1) as f32 * LAYER_SPACING;

        let mut arrangement = piece.arrangement;
        arrangement.anchor = 0.5 * (low + high);
        let radius = wrap_radius(half_width, measurements.torso_half_width + SIDE_CLEARANCE + outset);
        arrangement.curvature = radius.map_or(0.0, |r| 1.0 / r);
        // Wrapping moves the side edges towards the body by `sag`; keep
        // them on the piece's own side of the centre plane.
        let sag = radius.map_or(0.0, |r| r * (1.0 - (half_width / r).cos()));
        let depth = (body.torso_depth(side) + DEPTH_CLEARANCE + outset).max(sag + 0.03 + outset);
        arrangement.translation = Vec3::new(0.0, 0.0, side * depth);

        // Lift: every point near the top edge must clear the shoulders.
        let top_band = high.y - 0.15 * (high.y - low.y);
        let lift = piece
            .outline
            .points()
            .iter()
            .filter(|point| point.position.y >= top_band)
            .filter_map(|point| {
                let placed = arrangement.place(point.position);
                Some(body.shoulder_surface(placed.x.abs())? + SHOULDER_CLEARANCE + outset - placed.y)
            })
            .fold(f32::MIN, f32::max);
        arrangement.translation.y = if lift > f32::MIN {
            lift
        } else {
            // Nothing of the body below the top edge: hang it at the armpits.
            measurements.armpit_level - (high.y - arrangement.anchor.y) * 1e-3
        };

        if piece.arrangement != arrangement {
            piece.arrangement = arrangement;
            changes.push(DocumentChange::PieceArrangement(id));
        }
    }
    if !changes.is_empty() {
        document.touch();
    }
    changes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_radius_brings_the_edge_to_the_requested_reach() {
        assert_eq!(wrap_radius(0.2, 0.25), None);
        let radius = wrap_radius(0.25, 0.20).unwrap();
        assert!((radius * (0.25 / radius).sin() - 0.20).abs() < 1e-4);
        // Too tight to reach: capped at a half circle.
        let tight = wrap_radius(0.5, 0.05).unwrap();
        assert!((0.5 / tight - std::f32::consts::FRAC_PI_2).abs() < 1e-4);
    }
}

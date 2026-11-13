//! Places pattern pieces around a body, ready for sewing.
//!
//! Sewing pulls matching edges together along straight lines, so every
//! seam line must clear the body at the start. Two things make that work
//! for any body shape:
//!
//! * each piece is wrapped around a vertical cylinder so that its side
//!   edges end up just outside the torso, inside the arms;
//! * each piece is lifted until its top edge is above the shoulders.

use cad_core::glam::{Quat, Vec3};
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
/// Extra distance for the buttonhole side of a buttoned closure, so it
/// starts over the button side.
const OVERLAP_OUTSET: f32 = 0.015;

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
        if mid * (half_width / mid).sin() < reach {
            low = mid
        } else {
            high = mid
        }
    }
    Some(0.5 * (low + high))
}

/// How far the torso reaches from the body's axis in direction `d`
/// (horizontal unit vector), between hip and shoulder height. Arms are left
/// out: only points within the torso's half width count.
fn torso_reach(body: &BodyMesh, measurements: &Measurements, d: Vec3) -> f32 {
    let height = body.height();
    body.positions
        .iter()
        .filter(|p| {
            p.y > 0.45 * height
                && p.y < 0.85 * height
                && p.x.abs() <= measurements.torso_half_width * 1.05
        })
        .map(|p| p.x * d.x + p.z * d.z)
        .fold(0.0, f32::max)
}

/// Re-arranges every piece of `document` around `body`, each at its
/// [`cad_core::Placement`]: centre front and back pieces are wrapped so
/// their sides reach around the torso; pieces placed elsewhere (side
/// panels, a front half) are wrapped onto a cylinder around the body at
/// their angle.
pub fn arrange_around(
    document: &mut PatternDocument,
    body: &BodyMesh,
    measurements: &Measurements,
) -> Vec<DocumentChange> {
    let ids: Vec<_> = document.pieces().iter().map(|p| p.id).collect();
    // Pieces whose buttonholes are done up over another piece's buttons:
    // the overlapping side, which is worn on top.
    // (Unless its layer already puts it there.)
    let layer = |id| document.piece(id).map_or(1, |p| p.layer);
    let overlapping: Vec<_> = document
        .fastenings()
        .filter(|&(button_piece, _, hole_piece, _)| layer(hole_piece) <= layer(button_piece))
        .map(|(_, _, hole_piece, _)| hole_piece)
        .collect();
    let mut closures = Vec::new();
    for (_, _, hole_piece, _) in document.fastenings() {
        if !closures.contains(&hole_piece) {
            closures.push(hole_piece);
        }
    }
    let mut changes = Vec::new();
    for id in ids {
        let over = overlapping.contains(&id);
        let Some(piece) = document.piece_mut(id) else {
            continue;
        };
        let (low, high) = piece.outline.bounds();
        let half_width = 0.5 * (high.x - low.x) * 1e-3;
        // Two front halves sit on the same cylinder; where they overlap at
        // the button stand they would start inside each other, and nothing
        // would say which belongs on top.
        let outset = piece.layer.saturating_sub(1) as f32 * LAYER_SPACING
            + if over { OVERLAP_OUTSET } else { 0.0 };
        let angle = piece.placement.angle.to_radians();
        // Exact directions at quarter turns: sin(180°) is not quite zero
        // in floating point.
        let exact = |v: f32| if v.abs() < 1e-6 { 0.0 } else { v };
        let outward = Vec3::new(exact(angle.sin()), 0.0, exact(angle.cos()));
        // Centre front or centre back: the classic arrangement.
        let centred = outward.x.abs() < 1e-3;

        let mut arrangement = piece.arrangement;
        arrangement.anchor = 0.5 * (low + high);
        arrangement.rotation = Quat::from_rotation_y(angle);
        let depth = if centred {
            let side = outward.z.signum();
            let radius = wrap_radius(
                half_width,
                measurements.torso_half_width + SIDE_CLEARANCE + outset,
            );
            arrangement.curvature = radius.map_or(0.0, |r| 1.0 / r);
            // Wrapping moves the side edges towards the body by `sag`; keep
            // them on the piece's own side of the centre plane.
            let sag = radius.map_or(0.0, |r| r * (1.0 - (half_width / r).cos()));
            (body.torso_depth(side) + DEPTH_CLEARANCE + outset).max(sag + 0.03 + outset)
        } else {
            let depth = torso_reach(body, measurements, outward) + DEPTH_CLEARANCE + outset;
            arrangement.curvature = 1.0 / depth.max(0.05);
            depth
        };
        arrangement.translation = outward * depth;

        // Lift: every point near the top edge must clear the shoulders. Off
        // the centre line only points over the torso count; a side panel
        // ends under the arm.
        let top_band = high.y - 0.15 * (high.y - low.y);
        let reach = measurements.torso_half_width;
        let lift = piece
            .outline
            .points()
            .iter()
            .filter(|point| point.position.y >= top_band)
            .filter_map(|point| {
                let placed = arrangement.place(point.position);
                if !centred && placed.x.abs() > reach {
                    return None;
                }
                Some(
                    body.shoulder_surface(placed.x.abs())? + SHOULDER_CLEARANCE + outset - placed.y,
                )
            })
            .fold(f32::MIN, f32::max);
        arrangement.translation.y = if lift > f32::MIN {
            lift
        } else {
            // Nothing of the body above the top edge: hang it at the armpits.
            measurements.armpit_level - (high.y - arrangement.anchor.y) * 1e-3
        };
        arrangement.translation.y += piece.placement.height * 1e-3;

        if piece.arrangement != arrangement {
            piece.arrangement = arrangement;
            changes.push(DocumentChange::PieceArrangement(id));
        }
    }
    // Buttoned closures start closed: the buttonhole side is turned round
    // the body until its holes lie over the buttons, outside them. Left
    // to be pulled shut, two halves meet edge-on and either may end up on
    // top.
    for &hole_piece in &closures {
        let pairs: Vec<(Vec3, Vec3)> = document
            .fastenings()
            .filter(|&(_, _, h, _)| h == hole_piece)
            .filter_map(|(button_piece, button, _, hole)| {
                let b = document
                    .piece(button_piece)?
                    .arrangement
                    .place(button.position);
                let h = document.piece(hole_piece)?.arrangement.place(hole.position);
                Some((b, h))
            })
            .collect();
        if pairs.is_empty() {
            continue;
        }
        let heading = |p: Vec3| p.x.atan2(p.z);
        let turn = pairs
            .iter()
            .map(|(b, h)| wrap_angle(heading(*b) - heading(*h)))
            .sum::<f32>()
            / pairs.len() as f32;
        if turn.abs() < 1e-4 {
            continue;
        }
        let Some(piece) = document.piece_mut(hole_piece) else {
            continue;
        };
        let spin = Quat::from_rotation_y(turn);
        piece.arrangement.rotation = spin * piece.arrangement.rotation;
        let height = piece.arrangement.translation.y;
        piece.arrangement.translation = spin * piece.arrangement.translation;
        piece.arrangement.translation.y = height;
        if !changes.contains(&DocumentChange::PieceArrangement(hole_piece)) {
            changes.push(DocumentChange::PieceArrangement(hole_piece));
        }
    }
    if !changes.is_empty() {
        document.touch();
    }
    changes
}

/// `angle` brought into (−π, π].
fn wrap_angle(angle: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    angle - tau * ((angle + std::f32::consts::PI) / tau).floor()
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

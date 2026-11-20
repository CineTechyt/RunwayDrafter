//! Places pattern pieces around a body, ready for sewing.
//!
//! Sewing pulls matching edges together along straight lines, so every
//! seam line must clear the body at the start. Two things make that work
//! for any body shape:
//!
//! * each piece is wrapped around a vertical cylinder so that its side
//!   edges end up just outside the torso, inside the arms;
//! * each piece is lifted until its top edge is above the shoulders.

use cad_core::glam::{Mat3, Quat, Vec2, Vec3};
use cad_core::{Arrangement, BodyPart, DocumentChange, PatternDocument, PatternPiece};

use crate::measure::{Measure, Measurements};
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
/// Gap between a sleeve and the arm it is wrapped around.
const ARM_CLEARANCE: f32 = 0.03;
/// Gap between a neck band and the neck.
const NECK_CLEARANCE: f32 = 0.02;
/// Room around a thigh for a trouser leg.
const LEG_CLEARANCE: f32 = 0.025;
const WAIST_CLEARANCE: f32 = 0.015;
/// How far a sleeve's top reaches past the shoulder joint, along the arm.
const CAP_ABOVE_SHOULDER: f32 = 0.03;

/// A piece wrapped around an arm or the neck: the piece's width goes round
/// the limb, its height along it. A sleeve's top (the cap) is at the
/// shoulder and its middle faces away from the body, so its two side edges
/// meet under the arm. `None` when the body has no such limb (an imported
/// avatar) or the piece belongs on the torso.
fn limb_arrangement(
    piece: &PatternPiece,
    body: &BodyMesh,
    measurements: &Measurements,
) -> Option<Arrangement> {
    let (low, high) = piece.outline.bounds();
    let width = (high.x - low.x) * 1e-3;
    let girth = |measure: Measure, fallback: f32| {
        let g = measurements.get(measure);
        if g > 0.0 {
            g / std::f32::consts::TAU
        } else {
            fallback
        }
    };
    let (centre, along, outward, limb_radius, anchor) = match piece.placement.part {
        BodyPart::Torso => return None,
        BodyPart::LeftArm | BodyPart::RightArm => {
            let left = piece.placement.part == BodyPart::LeftArm;
            let (shoulder, elbow) = body.arm_axis(left)?;
            let down = (elbow - shoulder).try_normalize()?;
            let lateral = Vec3::X * if left { 1.0 } else { -1.0 };
            let outward = (lateral - down * lateral.dot(down)).try_normalize()?;
            (
                shoulder - down * CAP_ABOVE_SHOULDER,
                -down,
                outward,
                girth(Measure::UpperArm, 0.05) + ARM_CLEARANCE,
                Vec2::new(0.5 * (low.x + high.x), high.y),
            )
        }
        BodyPart::LeftLeg | BodyPart::RightLeg => {
            // Trouser legs: around the leg from the hip down, each half
            // (front, back) wrapped round half of it; the top at the waist.
            let left = piece.placement.part == BodyPart::LeftLeg;
            let (hip, _knee, ankle) = body.leg_axis(left)?;
            let up = (hip - ankle).try_normalize()?;
            let facing = piece.placement.angle.to_radians();
            let ahead = Vec3::new(facing.sin(), 0.0, facing.cos());
            let outward = (ahead - up * ahead.dot(up)).try_normalize()?;
            let waist = hip + up * ((measurements.waist_level - hip.y) / up.y.max(0.2));
            let thigh = girth(Measure::Thigh, 0.09) + LEG_CLEARANCE;
            // The leg's own part of the piece (below the crotch) centres on
            // the leg; the crotch extension wraps on towards the other leg.
            let lower: Vec<Vec2> = piece
                .outline
                .points()
                .iter()
                .map(|p| p.position)
                .filter(|p| p.y < low.y + 0.5 * (high.y - low.y))
                .collect();
            let centre_x = if lower.is_empty() {
                0.5 * (low.x + high.x)
            } else {
                let (a, b) = lower
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(a, b), p| (a.min(p.x), b.max(p.x)));
                0.5 * (a + b)
            };
            let radius = (width / std::f32::consts::PI).max(thigh);
            let across = up.cross(outward);
            let mut arrangement = piece.arrangement;
            arrangement.anchor = Vec2::new(centre_x, high.y);
            arrangement.curvature = 1.0 / radius;
            arrangement.rotation = Quat::from_mat3(&Mat3::from_cols(across, up, outward));
            arrangement.translation = waist
                + outward * (radius + piece.placement.depth * 1e-3)
                + up * (piece.placement.height * 1e-3);
            return Some(arrangement);
        }
        BodyPart::Waist => {
            // Waistbands: half round the waist, centred on `angle`.
            let facing = piece.placement.angle.to_radians();
            let outward = Vec3::new(facing.sin(), 0.0, facing.cos());
            let radius =
                (width / std::f32::consts::PI).max(girth(Measure::Waist, 0.14) + WAIST_CLEARANCE);
            let across = Vec3::Y.cross(outward);
            let mut arrangement = piece.arrangement;
            arrangement.anchor = 0.5 * (low + high);
            arrangement.curvature = 1.0 / radius;
            arrangement.rotation = Quat::from_mat3(&Mat3::from_cols(across, Vec3::Y, outward));
            arrangement.translation = Vec3::new(0.0, measurements.waist_level, 0.0)
                + outward * radius
                + Vec3::Y * (piece.placement.height * 1e-3 + 0.5 * (high.y - low.y) * 1e-3);
            return Some(arrangement);
        }
        BodyPart::Neck => {
            let base = body.neck_base()?;
            (
                base,
                Vec3::Y,
                Vec3::Z,
                girth(Measure::Neck, 0.06) + NECK_CLEARANCE,
                0.5 * (low + high),
            )
        }
    };
    // Wide enough pieces go once round; narrower ones keep their distance
    // and leave a gap that sewing closes.
    let radius = (width / std::f32::consts::TAU).max(limb_radius);
    let across = along.cross(outward);
    let mut arrangement = piece.arrangement;
    arrangement.anchor = anchor;
    arrangement.curvature = 1.0 / radius;
    arrangement.rotation = Quat::from_mat3(&Mat3::from_cols(across, along, outward));
    arrangement.translation = centre + outward * radius + along * (piece.placement.height * 1e-3);
    Some(arrangement)
}

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
    let sewn_together = same_garment(document);
    // A buttonhole piece on a higher layer than its buttons but sewn into
    // the same garment is only lapped over them: it starts where the
    // button piece's layer does (a whole layer further out, the closure
    // started turned and the garment draped lopsided).
    let lapped: Vec<(cad_core::PieceId, u32)> = document
        .fastenings()
        .filter(|&(b, _, h, _)| b != h && layer(h) > layer(b) && sewn_together(b, h))
        .map(|(b, _, h, _)| (h, layer(b)))
        .collect();
    let overlapping: Vec<_> = document
        .fastenings()
        .filter(|&(button_piece, _, hole_piece, _)| {
            layer(hole_piece) <= layer(button_piece) || lapped.iter().any(|&(h, _)| h == hole_piece)
        })
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
        let layer = lapped
            .iter()
            .find(|&&(h, _)| h == id)
            .map_or(piece.layer, |&(_, l)| l);
        let outset = layer.saturating_sub(1) as f32 * LAYER_SPACING
            + if over { OVERLAP_OUTSET } else { 0.0 };
        let angle = piece.placement.angle.to_radians();
        // Exact directions at quarter turns: sin(180°) is not quite zero
        // in floating point.
        let exact = |v: f32| if v.abs() < 1e-6 { 0.0 } else { v };
        let outward = Vec3::new(exact(angle.sin()), 0.0, exact(angle.cos()));
        // Centre front or centre back: the classic arrangement.
        let centred = outward.x.abs() < 1e-3;

        if let Some(arrangement) = limb_arrangement(piece, body, measurements) {
            if piece.arrangement != arrangement {
                piece.arrangement = arrangement;
                changes.push(DocumentChange::PieceArrangement(id));
            }
            continue;
        }
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
                let holed = document.piece(hole_piece)?;
                let h = holed.arrangement.place(hole.button_end(&holed.outline));
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
    // Pieces that lie under another start behind it, at their seam.
    let under: Vec<_> = document
        .pieces()
        .iter()
        .filter_map(|p| Some((p.id, p.placement.under?, p.placement.depth)))
        .collect();
    for (id, host_id, depth) in under {
        let Some(arrangement) = under_arrangement(document, id, host_id, depth) else {
            continue;
        };
        if let Some(piece) = document.piece_mut(id)
            && piece.arrangement != arrangement
        {
            piece.arrangement = arrangement;
            if !changes.contains(&DocumentChange::PieceArrangement(id)) {
                changes.push(DocumentChange::PieceArrangement(id));
            }
        }
    }
    if !changes.is_empty() {
        document.touch();
    }
    changes
}

/// Whether two pieces are sewn into one garment, directly or through
/// other pieces.
fn same_garment(
    document: &PatternDocument,
) -> impl Fn(cad_core::PieceId, cad_core::PieceId) -> bool + use<> {
    let ids: Vec<_> = document.pieces().iter().map(|p| p.id).collect();
    let mut group: Vec<usize> = (0..ids.len()).collect();
    fn root(group: &mut [usize], mut i: usize) -> usize {
        while group[i] != i {
            group[i] = group[group[i]];
            i = group[i];
        }
        i
    }
    for seam in document.seams() {
        let (Some(a), Some(b)) = (
            ids.iter().position(|&id| id == seam.a.piece),
            ids.iter().position(|&id| id == seam.b.piece),
        ) else {
            continue;
        };
        let (ra, rb) = (root(&mut group, a), root(&mut group, b));
        group[ra.max(rb)] = ra.min(rb);
    }
    let roots: Vec<usize> = (0..ids.len()).map(|i| root(&mut group, i)).collect();
    move |a, b| {
        let at = |id| ids.iter().position(|&p| p == id).map(|i| roots[i]);
        at(a).is_some() && at(a) == at(b)
    }
}

/// Where piece `id` starts when it lies under `host`: on the host's own
/// cylinder, `depth` mm off it (negative: inside), the middles of their
/// seam's two sides on top of each other. Both pieces must lie the same
/// way up in the pattern.
fn under_arrangement(
    document: &PatternDocument,
    id: cad_core::PieceId,
    host_id: cad_core::PieceId,
    depth: f32,
) -> Option<Arrangement> {
    use cad_core::Curve;
    let (piece, host) = (document.piece(id)?, document.piece(host_id)?);
    let middle = |p: &PatternPiece, span: cad_core::EdgeSpan| {
        let n = p.outline.edge_count();
        let edge = (span.first + span.count / 2) % n.max(1);
        let t = if span.count % 2 == 0 { 0.0 } else { 0.5 };
        p.outline.curve(edge).eval(t)
    };
    let (mine, theirs) = document.seams().iter().find_map(|seam| {
        if seam.a.piece == id && seam.b.piece == host_id {
            Some((middle(piece, seam.a_span()), middle(host, seam.b_span())))
        } else if seam.b.piece == id && seam.a.piece == host_id {
            Some((middle(piece, seam.b_span()), middle(host, seam.a_span())))
        } else {
            None
        }
    })?;
    let on = host.arrangement;
    let offset = depth * 1e-3;
    // The host's frame at the seam: turned round its cylinder's axis.
    let angle = (theirs.x - on.anchor.x) * 1e-3 * on.curvature;
    let rotation = on.rotation * Quat::from_rotation_y(angle);
    let outward = rotation * Vec3::Z;
    let mut arrangement = piece.arrangement;
    arrangement.anchor = mine;
    arrangement.rotation = rotation;
    arrangement.translation = on.place(theirs) + outward * offset;
    arrangement.curvature = if on.curvature.abs() > 1e-6 {
        1.0 / (1.0 / on.curvature + offset).max(0.02)
    } else {
        0.0
    };
    Some(arrangement)
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

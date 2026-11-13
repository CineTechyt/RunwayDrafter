//! Turning and mirroring whole pieces in the 2D layout.
//!
//! Everything that refers to a piece's edges follows along: seam
//! allowances per edge, notches and seams. Mirroring reverses the outline
//! (it must stay counter-clockwise), which renumbers the edges backwards
//! and runs every edge the other way; seams touching the piece are
//! re-pointed and turned around to match.

use glam::Vec2;

use crate::document::PatternDocument;
use crate::ids::PieceId;
use crate::pattern::PatternPiece;

/// A rigid change of a piece's 2D layout, about the centre of its bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PieceTransform {
    /// Quarter turn clockwise.
    RotateCw,
    /// Quarter turn counter-clockwise.
    RotateCcw,
    /// Mirror left ↔ right.
    FlipHorizontal,
    /// Mirror top ↔ bottom.
    FlipVertical,
}

impl PieceTransform {
    pub const ALL: [Self; 4] = [Self::RotateCcw, Self::RotateCw, Self::FlipHorizontal, Self::FlipVertical];

    pub fn label(self) -> &'static str {
        match self {
            Self::RotateCw => "Rotate clockwise",
            Self::RotateCcw => "Rotate counter-clockwise",
            Self::FlipHorizontal => "Mirror horizontally",
            Self::FlipVertical => "Mirror vertically",
        }
    }

    /// Whether the transform mirrors (and so reverses the outline).
    pub fn mirrors(self) -> bool {
        matches!(self, Self::FlipHorizontal | Self::FlipVertical)
    }

    /// The map of a point, about `centre`.
    pub fn point(self, centre: Vec2, p: Vec2) -> Vec2 {
        let d = p - centre;
        centre
            + match self {
                Self::RotateCw => Vec2::new(d.y, -d.x),
                Self::RotateCcw => Vec2::new(-d.y, d.x),
                Self::FlipHorizontal => Vec2::new(-d.x, d.y),
                Self::FlipVertical => Vec2::new(d.x, -d.y),
            }
    }

    /// The map of a direction given as an angle from +X, in radians.
    pub fn angle(self, angle: f32) -> f32 {
        use std::f32::consts::{FRAC_PI_2, PI, TAU};
        let mapped = match self {
            Self::RotateCw => angle - FRAC_PI_2,
            Self::RotateCcw => angle + FRAC_PI_2,
            Self::FlipHorizontal => PI - angle,
            Self::FlipVertical => -angle,
        };
        mapped.rem_euclid(TAU)
    }
}

impl PatternPiece {
    /// Applies `transform` about the centre of the piece's bounds. Returns
    /// `true` if the outline had to be reversed, i.e. edge `e` is now edge
    /// `n - 1 - e` and runs the other way.
    pub fn apply_transform(&mut self, transform: PieceTransform) -> bool {
        let (lo, hi) = self.outline.bounds();
        let centre = (lo + hi) * 0.5;
        self.outline.map_points(|p| transform.point(centre, p));
        let reversed = !self.outline.is_ccw();
        if reversed {
            let n = self.outline.edge_count();
            self.outline = self.outline.reverse();
            self.seam_allowance.per_edge =
                std::mem::take(&mut self.seam_allowance.per_edge).into_iter().map(|(edge, width)| (n - 1 - edge, width)).collect();
            for notch in &mut self.notches {
                notch.edge = n - 1 - notch.edge;
                notch.t = 1.0 - notch.t;
            }
        }
        self.grainline.origin = transform.point(centre, self.grainline.origin);
        self.grainline.angle = transform.angle(self.grainline.angle);
        self.uv.origin = transform.point(centre, self.uv.origin);
        self.uv.rotation = transform.angle(self.uv.rotation);
        self.arrangement.anchor = transform.point(centre, self.arrangement.anchor);
        reversed
    }
}

impl PatternDocument {
    /// Turns or mirrors a piece in place; seams stay attached to the same
    /// lines. Returns `false` if there is no such piece.
    pub fn transform_piece(&mut self, id: PieceId, transform: PieceTransform) -> bool {
        let Some(piece) = self.piece_mut(id) else { return false };
        let n = piece.outline.edge_count();
        let reversed = piece.apply_transform(transform);
        if reversed {
            self.reverse_seams_on(id, n);
        }
        self.touch();
        true
    }

    /// Re-points the seams on a piece whose outline was reversed.
    fn reverse_seams_on(&mut self, id: PieceId, n: u32) {
        for seam in self.seams_mut() {
            if seam.a.piece == id {
                seam.a.edge = n - 1 - seam.a_span().last(n);
                seam.reversed = !seam.reversed;
            }
            if seam.b.piece == id {
                seam.b.edge = n - 1 - seam.b_span().last(n);
                seam.reversed = !seam.reversed;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Curve;
    use crate::ids::EdgeRef;

    fn edge_midpoints(doc: &PatternDocument, edge: EdgeRef) -> Vec2 {
        doc.piece(edge.piece).unwrap().outline.curve(edge.edge).eval(0.5)
    }

    #[test]
    fn four_quarter_turns_are_the_identity() {
        let mut doc = PatternDocument::demo_top();
        let id = doc.pieces()[0].id;
        let before = doc.piece(id).unwrap().clone();
        for _ in 0..4 {
            assert!(doc.transform_piece(id, PieceTransform::RotateCw));
        }
        let after = doc.piece(id).unwrap();
        for (a, b) in before.outline.points().iter().zip(after.outline.points()) {
            assert!(a.position.distance(b.position) < 1e-3);
        }
        assert!((after.grainline.angle - before.grainline.angle).abs() < 1e-5);
    }

    #[test]
    fn a_quarter_turn_turns_the_grainline_with_the_piece() {
        let mut doc = PatternDocument::demo_top();
        let id = doc.pieces()[0].id;
        doc.transform_piece(id, PieceTransform::RotateCw);
        let piece = doc.piece(id).unwrap();
        // Vertical before, horizontal after.
        assert!(piece.grainline.direction().y.abs() < 1e-5);
        let (lo, hi) = piece.outline.bounds();
        assert!(hi.x - lo.x > hi.y - lo.y, "the tall piece now lies on its side");
    }

    #[test]
    fn mirroring_keeps_seams_on_the_same_lines_and_turns_them_around() {
        let mut doc = PatternDocument::demo_top();
        let front = doc.pieces()[0].id;
        let seam = doc.seams()[0];
        assert_eq!(seam.a.piece, front);
        let (lo, hi) = doc.piece(front).unwrap().outline.bounds();
        let centre_x = (lo.x + hi.x) * 0.5;
        let midpoint = edge_midpoints(&doc, seam.a);

        assert!(doc.transform_piece(front, PieceTransform::FlipHorizontal));
        let piece = doc.piece(front).unwrap();
        assert!(piece.outline.is_ccw());
        let moved = doc.seams().iter().find(|s| s.id == seam.id).unwrap();
        // The seam's edge is the mirror image of the one it was on…
        let mirrored = edge_midpoints(&doc, moved.a);
        assert!((mirrored - Vec2::new(2.0 * centre_x - midpoint.x, midpoint.y)).length() < 1e-2);
        // …which now runs the other way.
        assert_ne!(moved.reversed, seam.reversed);
        // Notches move along.
        for notch in &piece.notches {
            assert!(piece.notch_frame(notch).is_some());
        }
    }
}

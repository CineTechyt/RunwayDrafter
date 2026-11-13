//! Pattern lines: what a pattern maker sees as one line of a piece.
//!
//! An outline is stored as segments (edges), each a line or one Bézier. A
//! curve such as an armhole is often several segments joined smoothly. Its
//! anchors in between are [`PointKind::Smooth`] *curve points*, while
//! [`PointKind::Corner`] anchors are the *turn points* that separate one
//! line from the next. Selection and sewing work on lines, so a curve made
//! of three segments is still picked and sewn as one.

use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::fit;
use crate::geometry::{self, Curve};
use crate::pattern::{ControlPoint, Outline, PointKind, Segment};

/// Corners rounded off within this distance still count as corners …
const CORNER_WINDOW_MM: f32 = 4.0;
/// … if the outline turns by this many times the corner angle over it.
const WINDOW_TURN_FACTOR: f32 = 2.5;

/// A run of consecutive edges of one outline: `first`, `first + 1`, …,
/// wrapping around after the last edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeSpan {
    pub first: u32,
    pub count: u32,
}

impl EdgeSpan {
    pub const fn single(edge: u32) -> Self {
        Self { first: edge, count: 1 }
    }

    /// Whether `edge` is part of the span in an outline of `n` edges.
    pub fn contains(&self, edge: u32, n: u32) -> bool {
        n > 0 && (edge + n - self.first % n) % n < self.count
    }

    /// The edges in order.
    pub fn edges(&self, n: u32) -> impl Iterator<Item = u32> + '_ {
        let n = n.max(1);
        (0..self.count.min(n)).map(move |k| (self.first + k) % n)
    }

    pub fn last(&self, n: u32) -> u32 {
        (self.first + self.count.max(1) - 1) % n.max(1)
    }

    /// The smallest span covering both, if they are adjacent or overlap
    /// (`None` if there is a gap between them).
    pub fn union(&self, other: &Self, n: u32) -> Option<Self> {
        if n == 0 {
            return None;
        }
        let covered: Vec<bool> = (0..n).map(|e| self.contains(e, n) || other.contains(e, n)).collect();
        let count = covered.iter().filter(|&&c| c).count() as u32;
        if count == n {
            return Some(Self { first: self.first, count: n });
        }
        // The run starts after the one uncovered edge that precedes a covered one.
        let first = (0..n).find(|&e| covered[e as usize] && !covered[((e + n - 1) % n) as usize])?;
        let run = (0..n).take_while(|k| covered[((first + k) % n) as usize]).count() as u32;
        (run == count).then_some(Self { first, count })
    }
}

impl Outline {
    /// The line edge `edge` belongs to: the run of edges joined at curve
    /// points, between two turn points.
    pub fn line_span(&self, edge: u32) -> EdgeSpan {
        let n = self.edge_count();
        if n == 0 {
            return EdgeSpan::single(edge);
        }
        let smooth = |anchor: u32| self.points()[(anchor % n) as usize].kind == PointKind::Smooth;
        if (0..n).all(smooth) {
            return EdgeSpan { first: 0, count: n };
        }
        let mut first = edge % n;
        while smooth(first) {
            first = (first + n - 1) % n;
        }
        let mut count = 1;
        while smooth((first + count) % n) {
            count += 1;
        }
        EdgeSpan { first, count }
    }

    /// Every line of the outline, in order.
    pub fn lines(&self) -> Vec<EdgeSpan> {
        let n = self.edge_count();
        let mut lines: Vec<EdgeSpan> = Vec::new();
        for edge in 0..n {
            let span = self.line_span(edge);
            if !lines.contains(&span) {
                lines.push(span);
            }
        }
        lines
    }

    /// The polyline of a span, from its start to its end.
    pub fn span_points(&self, span: EdgeSpan, tolerance: f32) -> Vec<Vec2> {
        let mut points = Vec::new();
        for edge in span.edges(self.edge_count()) {
            let curve = self.curve(edge);
            if points.is_empty() {
                points.push(curve.eval(0.0));
            }
            curve.flatten_into(tolerance, &mut points);
        }
        points
    }

    pub fn span_length(&self, span: EdgeSpan) -> f32 {
        span.edges(self.edge_count()).map(|e| self.curve(e).length(0.1)).sum()
    }

    /// Arc-length position of `p` along a span, from 0 (start) to 1 (end).
    pub fn span_fraction(&self, span: EdgeSpan, p: Vec2) -> f32 {
        let points = self.span_points(span, 0.25);
        let total = geometry::polyline_length(&points).max(1e-6);
        let (mut best, mut best_distance, mut walked) = (0.0, f32::INFINITY, 0.0);
        for pair in points.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let d = b - a;
            let t = if d.length_squared() > 0.0 { ((p - a).dot(d) / d.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
            let distance = (a + d * t).distance(p);
            if distance < best_distance {
                best_distance = distance;
                best = (walked + d.length() * t) / total;
            }
            walked += d.length();
        }
        best
    }

    /// Rebuilds the outline with as few segments as fit within `tolerance`
    /// (mm): anchors that turn by more than `corner_degrees` stay as turn
    /// points; everything between them is refitted as one straight line or
    /// a chain of cubics joined at curve points.
    pub fn simplified(&self, tolerance: f32, corner_degrees: f32) -> Outline {
        let n = self.edge_count() as usize;
        if n < 3 {
            return self.clone();
        }
        // Drop segments too short to matter, then find the corners.
        let mut source = self.clone();
        source.remove_tiny_segments(tolerance.max(0.05));
        let n = source.edge_count() as usize;
        let corner_cos = corner_degrees.to_radians().cos();
        let tangent_in = |e: usize| source.curve(((e + n - 1) % n) as u32).end_direction();
        let tangent_out = |e: usize| source.curve(e as u32).start_direction();
        let windowed = source.windowed_turns(CORNER_WINDOW_MM);
        let wide_corner = (corner_degrees * WINDOW_TURN_FACTOR).to_radians();
        let is_corner: Vec<bool> = (0..n)
            .map(|anchor| {
                if source.points()[anchor].kind != PointKind::Corner {
                    return false;
                }
                // A sharp joint…
                if tangent_in(anchor).dot(tangent_out(anchor)) < corner_cos {
                    return true;
                }
                // …or the middle of a corner rounded off over a few short
                // segments, none of which turns much on its own: the
                // strongest turn among its close neighbours.
                windowed[anchor].0 > wide_corner
                    && (0..n).all(|other| {
                        other == anchor
                            || windowed[other].1.distance(windowed[anchor].1) > CORNER_WINDOW_MM
                            || windowed[other].0 < windowed[anchor].0
                            || (windowed[other].0 == windowed[anchor].0 && other > anchor)
                    })
            })
            .collect();
        let corners: Vec<usize> = (0..n).filter(|&a| is_corner[a]).collect();

        let mut points: Vec<ControlPoint> = Vec::new();
        let mut segments: Vec<Segment> = Vec::new();
        let mut emit_run = |start: usize, count: usize, start_kind: PointKind, closed_loop: bool| {
            let span = EdgeSpan { first: start as u32, count: count as u32 };
            let raw = source.span_points(span, tolerance * 0.25);
            let length = geometry::polyline_length(&raw);
            // Evenly spaced samples: the fitter weighs every sample alike.
            let samples = geometry::resample_polyline(&raw, ((length / 2.0).ceil() as u32).clamp(2, 4000));
            let mut t1 = tangent_out(start);
            let mut t2 = -source.curve(((start + count - 1) % n) as u32).end_direction();
            if closed_loop {
                // A loop without corners: one shared tangent where it closes.
                let shared = (tangent_in(start) + tangent_out(start)).normalize_or(tangent_out(start));
                t1 = shared;
                t2 = -shared;
            }
            points.push(ControlPoint { position: source.points()[start].position, kind: start_kind });
            if count == 1 && !closed_loop {
                // Already a single segment: nothing to simplify.
                segments.push(source.segments()[start]);
                return;
            }
            if fit::is_straight(&samples, tolerance) && !closed_loop {
                segments.push(Segment::Line);
                return;
            }
            let cubics = fit::fit_cubics(&samples, t1, t2, tolerance);
            for (k, cubic) in cubics.iter().enumerate() {
                if k > 0 {
                    points.push(ControlPoint { position: cubic.p0, kind: PointKind::Smooth });
                }
                segments.push(Segment::Cubic { ctrl1: cubic.p1, ctrl2: cubic.p2 });
            }
        };

        if corners.is_empty() {
            emit_run(0, n, PointKind::Smooth, true);
        } else {
            for (k, &start) in corners.iter().enumerate() {
                let next = corners[(k + 1) % corners.len()];
                let count = if next > start { next - start } else { next + n - start };
                emit_run(start, count, PointKind::Corner, false);
            }
        }
        let mut outline = Outline::new(points, segments);
        outline.ensure_ccw();
        outline
    }

    /// For every anchor: how much the outline turns between a few
    /// millimetres before and after it (radians), and where the anchor is.
    fn windowed_turns(&self, window: f32) -> Vec<(f32, Vec2)> {
        let flat = self.flatten(0.05);
        let points = &flat.points;
        let count = points.len();
        if count < 3 {
            return vec![(0.0, Vec2::ZERO); self.edge_count() as usize];
        }
        // Arc length at every polyline point, and the total.
        let mut along = Vec::with_capacity(count + 1);
        along.push(0.0f32);
        for i in 0..count {
            along.push(along[i] + points[i].distance(points[(i + 1) % count]));
        }
        let total = along[count];
        // The point at arc length `s` (wrapping).
        let at = |s: f32| {
            let s = s.rem_euclid(total);
            let i = along.partition_point(|&a| a <= s).saturating_sub(1).min(count - 1);
            let segment = along[i + 1] - along[i];
            let t = if segment > 0.0 { (s - along[i]) / segment } else { 0.0 };
            points[i].lerp(points[(i + 1) % count], t)
        };
        (0..self.edge_count())
            .map(|edge| {
                let index = flat.edge_of.iter().position(|&e| e == edge).unwrap_or(0);
                let (here, s) = (points[index], along[index]);
                let window = window.min(total / 8.0);
                let before = (here - at(s - window)).normalize_or_zero();
                let after = (at(s + window) - here).normalize_or_zero();
                (before.dot(after).clamp(-1.0, 1.0).acos(), here)
            })
            .collect()
    }

    /// Merges segments shorter than `min_length` into their neighbours.
    fn remove_tiny_segments(&mut self, min_length: f32) {
        loop {
            let n = self.edge_count() as usize;
            if n <= 3 {
                return;
            }
            let Some(mut edge) = (0..n).find(|&e| self.curve(e as u32).length(0.05) < min_length) else { return };
            let mut points = self.points().to_vec();
            let mut segments = self.segments().to_vec();
            if edge == n - 1 {
                // Keep the segment off the end, where its end anchor is anchor 0.
                points.rotate_left(1);
                segments.rotate_left(1);
                edge -= 1;
            }
            // Drop the segment's end anchor: the next segment now starts at
            // this segment's start.
            let next = edge + 1;
            let start = points[edge].position;
            if let Segment::Cubic { ctrl1, .. } = &mut segments[next]
                && points[next].position.distance(*ctrl1) < 1e-6
            {
                *ctrl1 = start;
            }
            points.remove(next);
            segments.remove(edge);
            // `points[edge]` now starts the former next segment.
            *self = Outline::new(points, segments);
        }
    }

    /// Marks anchors where the outline turns by less than `degrees` as curve
    /// points, so that the segments on either side form one line.
    pub fn mark_smooth_joins(&mut self, degrees: f32) {
        let n = self.edge_count() as usize;
        let cos = degrees.to_radians().cos();
        let kinds: Vec<PointKind> = (0..n)
            .map(|anchor| {
                let incoming = self.curve(((anchor + n - 1) % n) as u32).end_direction();
                let outgoing = self.curve(anchor as u32).start_direction();
                if incoming.dot(outgoing) >= cos { PointKind::Smooth } else { PointKind::Corner }
            })
            .collect();
        let mut points = self.points().to_vec();
        for (point, kind) in points.iter_mut().zip(kinds) {
            point.kind = kind;
        }
        *self = Outline::new(points, self.segments().to_vec());
    }

    /// Makes anchor `anchor` a turn point (`Corner`, splitting the line
    /// there) or a curve point (`Smooth`, joining the lines on either side
    /// into one). The shape does not change. Returns whether it changed.
    pub fn set_point_kind(&mut self, anchor: u32, kind: PointKind) -> bool {
        let mut points = self.points().to_vec();
        let Some(point) = points.get_mut(anchor as usize) else { return false };
        if point.kind == kind {
            return false;
        }
        point.kind = kind;
        *self = Outline::new(points, self.segments().to_vec());
        true
    }

    /// Applies `map` to every anchor and control point. A map that mirrors
    /// leaves the outline clockwise; see [`Outline::ensure_ccw`].
    pub fn map_points(&mut self, map: impl Fn(Vec2) -> Vec2) {
        let mut points = self.points().to_vec();
        for point in &mut points {
            point.position = map(point.position);
        }
        let segments = self
            .segments()
            .iter()
            .map(|segment| match *segment {
                Segment::Line => Segment::Line,
                Segment::Quadratic { ctrl } => Segment::Quadratic { ctrl: map(ctrl) },
                Segment::Cubic { ctrl1, ctrl2 } => Segment::Cubic { ctrl1: map(ctrl1), ctrl2: map(ctrl2) },
            })
            .collect();
        *self = Outline::new(points, segments);
    }
}

/// Unit start and end directions of a curve segment, robust to controls
/// that sit on their anchor.
trait Directions {
    fn start_direction(&self) -> Vec2;
    fn end_direction(&self) -> Vec2;
}

impl Directions for crate::geometry::CurveSegment {
    fn start_direction(&self) -> Vec2 {
        [0.0, 0.01, 0.1].iter().find_map(|&t| self.tangent(t).try_normalize()).unwrap_or(Vec2::X)
    }
    fn end_direction(&self) -> Vec2 {
        [1.0, 0.99, 0.9].iter().find_map(|&t| self.tangent(t).try_normalize()).unwrap_or(Vec2::X)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 200 × 100 rectangle whose top edge is drawn as 40 short lines
    /// following an arc that bulges 20 mm.
    fn bumpy() -> Outline {
        let mut points = vec![Vec2::new(0.0, 0.0), Vec2::new(200.0, 0.0)];
        for i in 0..=40 {
            let x = 200.0 - 5.0 * i as f32;
            let t = x / 200.0;
            points.push(Vec2::new(x, 100.0 + 80.0 * t * (1.0 - t)));
        }
        Outline::polygon(points.into_iter().filter(|p| *p != Vec2::new(0.0, 100.0)).chain([Vec2::new(0.0, 100.0)]))
    }

    #[test]
    fn simplifying_turns_a_polyline_curve_into_one_line_of_few_cubics() {
        let source = bumpy();
        assert!(source.edge_count() > 40);
        let simple = source.simplified(0.3, 15.0);
        assert!(simple.edge_count() <= 6, "{} edges", simple.edge_count());
        // Four turn points; the curve is one line.
        assert_eq!(simple.points().iter().filter(|p| p.kind == PointKind::Corner).count(), 4);
        assert_eq!(simple.lines().len(), 4);
        // Same shape: the areas agree.
        assert!((simple.signed_area() - source.signed_area()).abs() < 0.01 * source.signed_area());
    }

    #[test]
    fn a_corner_rounded_over_several_short_segments_stays_a_corner() {
        // A square whose bottom-right corner is a 3 mm fillet made of eight
        // little lines, each turning 11.25°.
        let mut points = vec![Vec2::ZERO];
        for k in 0..=8 {
            let a = (-90.0 + 11.25 * k as f32).to_radians();
            points.push(Vec2::new(97.0, 3.0) + Vec2::new(a.cos(), a.sin()) * 3.0);
        }
        points.extend([Vec2::new(100.0, 100.0), Vec2::new(0.0, 100.0)]);
        let simple = Outline::polygon(points).simplified(0.25, 15.0);
        assert_eq!(simple.lines().len(), 4, "four sides, the fillet does not merge two of them");
    }

    #[test]
    fn lines_follow_curve_points_and_spans_wrap() {
        let mut outline = bumpy().simplified(0.3, 15.0);
        let curve = outline.lines().into_iter().max_by_key(|s| s.count).unwrap();
        for edge in curve.edges(outline.edge_count()) {
            assert_eq!(outline.line_span(edge), curve);
        }
        // Mark everything as turn points: every edge is its own line again.
        let points: Vec<ControlPoint> =
            outline.points().iter().map(|p| ControlPoint { kind: PointKind::Corner, ..*p }).collect();
        outline = Outline::new(points, outline.segments().to_vec());
        assert_eq!(outline.lines().len() as u32, outline.edge_count());

        let span = EdgeSpan { first: 4, count: 3 };
        assert!(span.contains(0, 6) && span.contains(5, 6) && !span.contains(1, 6));
        assert_eq!(span.edges(6).collect::<Vec<_>>(), vec![4, 5, 0]);
        assert_eq!(span.union(&EdgeSpan::single(1), 6), Some(EdgeSpan { first: 4, count: 4 }));
        assert_eq!(span.union(&EdgeSpan::single(2), 6), None);
    }

    #[test]
    fn joining_two_lines_at_their_turn_point_makes_one() {
        let mut square = Outline::polygon([Vec2::ZERO, Vec2::new(100.0, 0.0), Vec2::new(100.0, 100.0), Vec2::new(0.0, 100.0)]);
        assert_eq!(square.lines().len(), 4);
        // Anchor 1 is where edge 0 (bottom) meets edge 1 (right side).
        assert!(square.set_point_kind(1, PointKind::Smooth));
        assert_eq!(square.line_span(0), EdgeSpan { first: 0, count: 2 });
        assert_eq!(square.lines().len(), 3);
        assert!(!square.set_point_kind(1, PointKind::Smooth), "nothing to change");
        assert!(square.set_point_kind(1, PointKind::Corner));
        assert_eq!(square.lines().len(), 4);
    }

    #[test]
    fn span_fraction_measures_along_the_line() {
        let outline = Outline::polygon([Vec2::ZERO, Vec2::new(100.0, 0.0), Vec2::new(100.0, 100.0), Vec2::new(0.0, 100.0)]);
        let span = EdgeSpan { first: 0, count: 2 };
        assert!((outline.span_fraction(span, Vec2::new(50.0, -3.0)) - 0.25).abs() < 1e-3);
        assert!((outline.span_fraction(span, Vec2::new(103.0, 50.0)) - 0.75).abs() < 1e-3);
    }
}

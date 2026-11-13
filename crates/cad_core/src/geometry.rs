//! Curve primitives and polygon utilities.

use glam::Vec2;
use serde::{Deserialize, Serialize};

/// A parametric 2D curve over `t ∈ [0, 1]`.
pub trait Curve {
    fn eval(&self, t: f32) -> Vec2;

    /// First derivative (not normalised).
    fn tangent(&self, t: f32) -> Vec2;

    /// Number of straight pieces needed to stay within `tolerance` (mm) of the
    /// true curve.
    fn subdivisions(&self, tolerance: f32) -> u32;

    /// Appends a polyline approximation to `out`. The start point is *not*
    /// emitted (so consecutive segments can be chained), the end point is.
    fn flatten_into(&self, tolerance: f32, out: &mut Vec<Vec2>) {
        let n = self.subdivisions(tolerance);
        for i in 1..=n {
            out.push(self.eval(i as f32 / n as f32));
        }
    }

    /// Polyline approximation including both end points.
    fn flatten(&self, tolerance: f32) -> Vec<Vec2> {
        let mut out = vec![self.eval(0.0)];
        self.flatten_into(tolerance, &mut out);
        out
    }

    /// Arc length, approximated to `tolerance`.
    fn length(&self, tolerance: f32) -> f32 {
        polyline_length(&self.flatten(tolerance))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Line {
    pub a: Vec2,
    pub b: Vec2,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuadBezier {
    pub p0: Vec2,
    pub p1: Vec2,
    pub p2: Vec2,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CubicBezier {
    pub p0: Vec2,
    pub p1: Vec2,
    pub p2: Vec2,
    pub p3: Vec2,
}

impl Curve for Line {
    fn eval(&self, t: f32) -> Vec2 {
        self.a.lerp(self.b, t)
    }
    fn tangent(&self, _t: f32) -> Vec2 {
        self.b - self.a
    }
    fn subdivisions(&self, _tolerance: f32) -> u32 {
        1
    }
}

impl Curve for QuadBezier {
    fn eval(&self, t: f32) -> Vec2 {
        let u = 1.0 - t;
        self.p0 * (u * u) + self.p1 * (2.0 * u * t) + self.p2 * (t * t)
    }
    fn tangent(&self, t: f32) -> Vec2 {
        (self.p1 - self.p0) * (2.0 * (1.0 - t)) + (self.p2 - self.p1) * (2.0 * t)
    }
    fn subdivisions(&self, tolerance: f32) -> u32 {
        // Wang's formula, degree 2.
        let m = (self.p0 - 2.0 * self.p1 + self.p2).length();
        wang(m / (4.0 * tolerance.max(1e-4)))
    }
}

impl Curve for CubicBezier {
    fn eval(&self, t: f32) -> Vec2 {
        let u = 1.0 - t;
        self.p0 * (u * u * u)
            + self.p1 * (3.0 * u * u * t)
            + self.p2 * (3.0 * u * t * t)
            + self.p3 * (t * t * t)
    }
    fn tangent(&self, t: f32) -> Vec2 {
        let u = 1.0 - t;
        (self.p1 - self.p0) * (3.0 * u * u)
            + (self.p2 - self.p1) * (6.0 * u * t)
            + (self.p3 - self.p2) * (3.0 * t * t)
    }
    fn subdivisions(&self, tolerance: f32) -> u32 {
        // Wang's formula, degree 3.
        let m = (self.p0 - 2.0 * self.p1 + self.p2)
            .length()
            .max((self.p1 - 2.0 * self.p2 + self.p3).length());
        wang(0.75 * m / tolerance.max(1e-4))
    }
}

fn wang(x: f32) -> u32 {
    (x.sqrt().ceil() as u32).clamp(1, 256)
}

/// A resolved outline segment (anchors and controls in absolute coordinates).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CurveSegment {
    Line(Line),
    Quad(QuadBezier),
    Cubic(CubicBezier),
}

impl Curve for CurveSegment {
    fn eval(&self, t: f32) -> Vec2 {
        match self {
            Self::Line(c) => c.eval(t),
            Self::Quad(c) => c.eval(t),
            Self::Cubic(c) => c.eval(t),
        }
    }
    fn tangent(&self, t: f32) -> Vec2 {
        match self {
            Self::Line(c) => c.tangent(t),
            Self::Quad(c) => c.tangent(t),
            Self::Cubic(c) => c.tangent(t),
        }
    }
    fn subdivisions(&self, tolerance: f32) -> u32 {
        match self {
            Self::Line(c) => c.subdivisions(tolerance),
            Self::Quad(c) => c.subdivisions(tolerance),
            Self::Cubic(c) => c.subdivisions(tolerance),
        }
    }
}

// ---------------------------------------------------------------------------
// Polyline / polygon helpers
// ---------------------------------------------------------------------------

pub fn polyline_length(points: &[Vec2]) -> f32 {
    points.windows(2).map(|w| w[0].distance(w[1])).sum()
}

/// Resamples an open polyline into `segments` pieces of equal arc length.
/// Returns `segments + 1` points; the first and last are copied exactly.
pub fn resample_polyline(points: &[Vec2], segments: u32) -> Vec<Vec2> {
    let segments = segments.max(1) as usize;
    let (Some(&first), Some(&last)) = (points.first(), points.last()) else {
        return Vec::new();
    };
    if points.len() < 2 {
        return vec![first; segments + 1];
    }
    let total = polyline_length(points);
    let mut out = Vec::with_capacity(segments + 1);
    out.push(first);

    let mut seg = 0usize; // current source segment
    let mut seg_start = 0.0f32; // arc length at points[seg]
    for i in 1..segments {
        let target = total * i as f32 / segments as f32;
        loop {
            let len = points[seg].distance(points[seg + 1]);
            if seg_start + len >= target || seg + 2 >= points.len() {
                let t = if len > 0.0 {
                    ((target - seg_start) / len).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                out.push(points[seg].lerp(points[seg + 1], t));
                break;
            }
            seg_start += len;
            seg += 1;
        }
    }
    out.push(last);
    out
}

/// Signed area of a closed polygon; positive when counter-clockwise (Y up).
pub fn signed_area(polygon: &[Vec2]) -> f32 {
    let n = polygon.len();
    let mut sum = 0.0;
    for i in 0..n {
        sum += polygon[i].perp_dot(polygon[(i + 1) % n]);
    }
    0.5 * sum
}

/// Even-odd point-in-polygon test.
pub fn point_in_polygon(p: Vec2, polygon: &[Vec2]) -> bool {
    let n = polygon.len();
    let mut inside = false;
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        let (a, b) = (polygon[i], polygon[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
        j = i;
    }
    inside
}

pub fn distance_to_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_squared();
    let t = if len2 > 0.0 {
        ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    p.distance(a + ab * t)
}

/// Distance from `p` to the boundary of a closed polygon.
pub fn distance_to_polygon(p: Vec2, polygon: &[Vec2]) -> f32 {
    let n = polygon.len();
    (0..n)
        .map(|i| distance_to_segment(p, polygon[i], polygon[(i + 1) % n]))
        .fold(f32::INFINITY, f32::min)
}

/// Offsets a closed polygon outwards. `width(i)` is the offset distance of the
/// polygon edge starting at vertex `i`, which lets every pattern edge carry its
/// own seam allowance. Corners are mitred, with the mitre length clamped.
pub fn offset_polygon(polygon: &[Vec2], width: impl Fn(usize) -> f32) -> Vec<Vec2> {
    const MITRE_LIMIT: f32 = 4.0;
    let n = polygon.len();
    if n < 3 {
        return polygon.to_vec();
    }
    // Outward is to the right of travel for a CCW polygon.
    let sign = if signed_area(polygon) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let normal = |i: usize| -> Vec2 {
        let d = (polygon[(i + 1) % n] - polygon[i]).normalize_or_zero();
        Vec2::new(d.y, -d.x) * sign
    };

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let prev = (i + n - 1) % n;
        let (n0, n1) = (normal(prev), normal(i));
        let (w0, w1) = (width(prev), width(i));
        // Intersect the two offset lines: find q with q·n0 = w0 and q·n1 = w1
        // (relative to the vertex).
        let det = n0.perp_dot(n1);
        let offset = if det.abs() < 1e-4 {
            // Collinear neighbours: no corner to mitre.
            n1 * w1.max(w0)
        } else {
            let q = Vec2::new(w0 * n1.y - w1 * n0.y, w1 * n0.x - w0 * n1.x) / det;
            q.clamp_length_max(MITRE_LIMIT * w0.max(w1))
        };
        out.push(polygon[i] + offset);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_hits_its_end_points() {
        let c = CubicBezier {
            p0: Vec2::ZERO,
            p1: Vec2::new(0.0, 10.0),
            p2: Vec2::new(10.0, 10.0),
            p3: Vec2::new(10.0, 0.0),
        };
        assert_eq!(c.eval(0.0), c.p0);
        assert_eq!(c.eval(1.0), c.p3);
        assert!(c.subdivisions(0.05) > 4);
    }

    #[test]
    fn resample_is_uniform() {
        let pts = [Vec2::ZERO, Vec2::new(3.0, 0.0), Vec2::new(10.0, 0.0)];
        let r = resample_polyline(&pts, 5);
        assert_eq!(r.len(), 6);
        for (i, p) in r.iter().enumerate() {
            assert!((p.x - 2.0 * i as f32).abs() < 1e-4);
        }
    }

    #[test]
    fn offset_square_grows_by_width() {
        let sq = [
            Vec2::ZERO,
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ];
        let off = offset_polygon(&sq, |_| 1.0);
        assert!((signed_area(&off) - 144.0).abs() < 1e-3);
        // Same result for the clockwise version.
        let cw: Vec<Vec2> = sq.iter().rev().copied().collect();
        assert!((signed_area(&offset_polygon(&cw, |_| 1.0)) + 144.0).abs() < 1e-3);
    }

    #[test]
    fn point_in_polygon_works() {
        let sq = [
            Vec2::ZERO,
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ];
        assert!(point_in_polygon(Vec2::new(5.0, 5.0), &sq));
        assert!(!point_in_polygon(Vec2::new(15.0, 5.0), &sq));
    }
}

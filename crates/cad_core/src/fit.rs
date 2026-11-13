//! Fitting Bézier curves to sampled points.
//!
//! Patterns imported from PDF or SVG arrive as long chains of short pieces:
//! a CAD export draws one armhole as a dozen little lines or cubics. To edit
//! them like a curve drawn by hand, every run of points between two corners
//! is replaced by as few cubic Béziers as fit within a tolerance, joined
//! with matching tangents (Schneider, "An Algorithm for Automatically
//! Fitting Digitized Curves", Graphics Gems 1990).

use glam::Vec2;

use crate::geometry::{CubicBezier, Curve, distance_to_segment};

/// Fits cubics to `points` (an open run, at least two points), so that no
/// point is further than `tolerance` from the result. `start_tangent` and
/// `end_tangent` are unit vectors pointing *into* the run at its start and
/// *back into* it at its end, i.e. the curve leaves `points[0]` along
/// `start_tangent` and arrives at the last point along `-end_tangent`.
///
/// Greedy: every cubic reaches as far along the points as it can while
/// still fitting (found by bisection), which comes close to the fewest
/// pieces. Neighbouring cubics share their joint and its tangent.
pub fn fit_cubics(
    points: &[Vec2],
    start_tangent: Vec2,
    end_tangent: Vec2,
    tolerance: f32,
) -> Vec<CubicBezier> {
    let mut out = Vec::new();
    let n = points.len();
    if n < 2 {
        return out;
    }
    let tolerance = tolerance.max(1e-4);
    // Tangent at an interior point, pointing back towards the start.
    let back = |i: usize| {
        if i + 1 >= n {
            end_tangent
        } else {
            (points[i - 1] - points[i + 1])
                .try_normalize()
                .unwrap_or_else(|| (points[i - 1] - points[i]).normalize_or_zero())
        }
    };
    let (mut start, mut leaving) = (0, start_tangent);
    while start + 1 < n {
        let fit = |end: usize| fit_single(&points[start..=end], leaving, back(end), tolerance);
        let end = match fit(n - 1) {
            Some(curve) => {
                out.push(curve);
                break;
            }
            None => {
                // The longest run from `start` that one cubic fits.
                let (mut good, mut bad) = (start + 1, n - 1);
                while bad - good > 1 {
                    let middle = (good + bad) / 2;
                    if fit(middle).is_some() {
                        good = middle
                    } else {
                        bad = middle
                    }
                }
                good
            }
        };
        out.push(
            fit(end).unwrap_or_else(|| straight(points[start], points[end], leaving, back(end))),
        );
        leaving = -back(end);
        start = end;
    }
    out
}

/// Whether every point lies within `tolerance` of the chord from the first
/// to the last point.
pub fn is_straight(points: &[Vec2], tolerance: f32) -> bool {
    let (Some(&a), Some(&b)) = (points.first(), points.last()) else {
        return true;
    };
    points
        .iter()
        .all(|&p| distance_to_segment(p, a, b) <= tolerance)
}

fn straight(first: Vec2, last: Vec2, t1: Vec2, t2: Vec2) -> CubicBezier {
    let third = first.distance(last) / 3.0;
    CubicBezier {
        p0: first,
        p1: first + t1 * third,
        p2: last + t2 * third,
        p3: last,
    }
}

/// One cubic through the run, if one fits within `tolerance`.
fn fit_single(points: &[Vec2], t1: Vec2, t2: Vec2, tolerance: f32) -> Option<CubicBezier> {
    let (first, last) = (points[0], points[points.len() - 1]);
    if points.len() == 2 {
        return Some(straight(first, last, t1, t2));
    }
    let mut parameters = chord_lengths(points);
    let mut curve = generate(points, &parameters, t1, t2);
    let (mut error, _) = max_error(points, &parameters, &curve);
    if error <= tolerance {
        return Some(curve);
    }
    // Chord-length parameters are poor for strongly curved runs, so even
    // a large first error may still converge.
    if error > 50.0 * tolerance {
        return None;
    }
    // Close: improve the parameters (closest points on the curve).
    for _ in 0..24 {
        reparameterize(points, &mut parameters, &curve);
        curve = generate(points, &parameters, t1, t2);
        (error, _) = max_error(points, &parameters, &curve);
        if error <= tolerance {
            return Some(curve);
        }
    }
    None
}

fn chord_lengths(points: &[Vec2]) -> Vec<f32> {
    let mut u = Vec::with_capacity(points.len());
    u.push(0.0);
    for pair in points.windows(2) {
        u.push(u.last().copied().unwrap_or(0.0) + pair[0].distance(pair[1]));
    }
    let total = u.last().copied().unwrap_or(0.0).max(1e-12);
    u.iter_mut().for_each(|v| *v /= total);
    u
}

fn bernstein(t: f32) -> [f32; 4] {
    let s = 1.0 - t;
    [s * s * s, 3.0 * s * s * t, 3.0 * s * t * t, t * t * t]
}

/// Least-squares cubic with fixed end points and end tangent directions.
fn generate(points: &[Vec2], parameters: &[f32], t1: Vec2, t2: Vec2) -> CubicBezier {
    let (first, last) = (points[0], points[points.len() - 1]);
    let (mut c00, mut c01, mut c11, mut x0, mut x1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (&p, &t) in points.iter().zip(parameters) {
        let b = bernstein(t);
        let (a0, a1) = (t1 * b[1], t2 * b[2]);
        c00 += a0.dot(a0);
        c01 += a0.dot(a1);
        c11 += a1.dot(a1);
        let rest = p - (first * (b[0] + b[1]) + last * (b[2] + b[3]));
        x0 += a0.dot(rest);
        x1 += a1.dot(rest);
    }
    let det = c00 * c11 - c01 * c01;
    let (mut alpha1, mut alpha2) = if det.abs() > 1e-12 {
        ((x0 * c11 - x1 * c01) / det, (c00 * x1 - c01 * x0) / det)
    } else {
        (0.0, 0.0)
    };
    // Degenerate or backwards handles: fall back to a third of the chord.
    let chord = first.distance(last);
    let epsilon = 1e-6 * chord;
    if alpha1 < epsilon || alpha2 < epsilon {
        alpha1 = chord / 3.0;
        alpha2 = chord / 3.0;
    }
    CubicBezier {
        p0: first,
        p1: first + t1 * alpha1,
        p2: last + t2 * alpha2,
        p3: last,
    }
}

/// Largest distance of a point from the curve at its parameter, and where.
fn max_error(points: &[Vec2], parameters: &[f32], curve: &CubicBezier) -> (f32, usize) {
    let mut worst = (0.0f32, points.len() / 2);
    for (i, (&p, &t)) in points
        .iter()
        .zip(parameters)
        .enumerate()
        .skip(1)
        .take(points.len().saturating_sub(2))
    {
        let distance = curve.eval(t).distance(p);
        if distance > worst.0 {
            worst = (distance, i);
        }
    }
    worst
}

/// One Newton step per point towards its closest point on the curve.
fn reparameterize(points: &[Vec2], parameters: &mut [f32], curve: &CubicBezier) {
    for (p, t) in points.iter().zip(parameters.iter_mut()) {
        let d = curve.eval(*t) - *p;
        let first = curve.tangent(*t);
        let s = 1.0 - *t;
        let second = (curve.p2 - 2.0 * curve.p1 + curve.p0) * (6.0 * s)
            + (curve.p3 - 2.0 * curve.p2 + curve.p1) * (6.0 * *t);
        let denominator = first.dot(first) + d.dot(second);
        if denominator.abs() > 1e-12 {
            *t = (*t - d.dot(first) / denominator).clamp(0.0, 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arc(points: usize, radius: f32, degrees: f32) -> Vec<Vec2> {
        (0..=points)
            .map(|i| {
                let a = (degrees * i as f32 / points as f32).to_radians();
                Vec2::new(radius * a.cos(), radius * a.sin())
            })
            .collect()
    }

    fn worst_distance(points: &[Vec2], curves: &[CubicBezier]) -> f32 {
        let flat: Vec<Vec2> = curves.iter().flat_map(|c| c.flatten(0.01)).collect();
        points
            .iter()
            .map(|&p| crate::geometry::distance_to_polygon(p, &flat))
            .fold(0.0, f32::max)
    }

    #[test]
    fn a_quarter_circle_needs_one_cubic() {
        let points = arc(64, 100.0, 90.0);
        let curves = fit_cubics(&points, Vec2::Y, Vec2::X, 0.1);
        assert_eq!(curves.len(), 1);
        assert!(worst_distance(&points, &curves) < 0.1);
    }

    #[test]
    fn an_s_curve_splits_with_matching_tangents() {
        let mut points = arc(48, 100.0, 120.0);
        let joint = *points.last().unwrap();
        let mirrored: Vec<Vec2> = arc(48, 100.0, 120.0)
            .iter()
            .map(|p| 2.0 * joint - *p)
            .rev()
            .skip(1)
            .collect();
        points.extend(mirrored);
        let start = (points[1] - points[0]).normalize();
        let end = (points[points.len() - 2] - points[points.len() - 1]).normalize();
        let curves = fit_cubics(&points, start, end, 0.2);
        assert!(
            curves.len() >= 2 && curves.len() <= 4,
            "{} cubics",
            curves.len()
        );
        // The flattened comparison curve adds its own 0.01 mm.
        assert!(worst_distance(&points, &curves) < 0.21);
        for pair in curves.windows(2) {
            assert_eq!(pair[0].p3, pair[1].p0);
            let incoming = (pair[0].p3 - pair[0].p2).normalize();
            let outgoing = (pair[1].p1 - pair[1].p0).normalize();
            assert!(incoming.dot(outgoing) > 0.999, "joints are smooth");
        }
    }

    #[test]
    fn straightness_is_judged_against_the_chord() {
        assert!(is_straight(
            &[Vec2::ZERO, Vec2::new(50.0, 0.05), Vec2::new(100.0, 0.0)],
            0.1
        ));
        assert!(!is_straight(
            &[Vec2::ZERO, Vec2::new(50.0, 2.0), Vec2::new(100.0, 0.0)],
            0.1
        ));
    }
}

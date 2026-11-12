//! Geometric tests between points, segments and triangles, static and
//! continuous (over a time step with linear motion).

use glam::{DVec3, Vec3A};

/// Where the segment `from → to` crosses triangle `abc`, as the fraction
/// along the segment, with the triangle's unit normal (right-handed for
/// `a, b, c`). Both faces count.
#[inline]
pub fn segment_triangle(from: Vec3A, to: Vec3A, a: Vec3A, b: Vec3A, c: Vec3A) -> Option<(f32, Vec3A)> {
    // Möller–Trumbore.
    let direction = to - from;
    let (e1, e2) = (b - a, c - a);
    let p = direction.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-14 {
        return None; // parallel to the triangle
    }
    let inv = 1.0 / det;
    let s = from - a;
    let u = s.dot(p) * inv;
    let q = s.cross(e1);
    let v = direction.dot(q) * inv;
    if u < 0.0 || v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    (0.0..=1.0).contains(&t).then(|| (t, e1.cross(e2).normalize_or_zero()))
}

/// The point of triangle `abc` closest to `p`, with its barycentric
/// weights. (Ericson, Real-Time Collision Detection, 5.1.5.)
#[inline]
pub fn closest_point_on_triangle(p: Vec3A, a: Vec3A, b: Vec3A, c: Vec3A) -> (Vec3A, [f32; 3]) {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, [1.0 - v, v, 0.0]);
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, [1.0 - w, 0.0, w]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, [0.0, 1.0 - w, w]);
    }
    let denominator = 1.0 / (va + vb + vc);
    let (v, w) = (vb * denominator, vc * denominator);
    (a + ab * v + ac * w, [1.0 - v - w, v, w])
}

/// Real roots of `c3·t³ + c2·t² + c1·t + c0` in `0..=1`, ascending.
///
/// The interval is cut at the extrema of the cubic; on each monotonic
/// piece a sign change brackets exactly one root, which bisection then
/// finds without any of the cancellation problems of closed formulas.
pub fn cubic_roots_in_unit_interval(c: [f64; 4]) -> ([f64; 3], usize) {
    let f = |t: f64| ((c[3] * t + c[2]) * t + c[1]) * t + c[0];
    // Extrema: 3·c3·t² + 2·c2·t + c1 = 0.
    let mut cuts = [0.0, 1.0, 1.0, 1.0];
    let mut cut_count = 1;
    let (qa, qb, qc) = (3.0 * c[3], 2.0 * c[2], c[1]);
    if qa.abs() > 1e-300 {
        let discriminant = qb * qb - 4.0 * qa * qc;
        if discriminant > 0.0 {
            let root = discriminant.sqrt();
            // The numerically stable pair of quadratic roots.
            let q = -0.5 * (qb + qb.signum() * root);
            let mut extrema = [q / qa, if q.abs() > 1e-300 { qc / q } else { f64::NAN }];
            if extrema[0] > extrema[1] {
                extrema.swap(0, 1);
            }
            for t in extrema {
                if t > 0.0 && t < 1.0 {
                    cuts[cut_count] = t;
                    cut_count += 1;
                }
            }
        }
    } else if qb.abs() > 1e-300 {
        let t = -qc / qb;
        if t > 0.0 && t < 1.0 {
            cuts[cut_count] = t;
            cut_count += 1;
        }
    }
    cuts[cut_count] = 1.0;

    let mut roots = [0.0; 3];
    let mut count = 0;
    for piece in 0..cut_count {
        let (mut low, mut high) = (cuts[piece], cuts[piece + 1]);
        let (f_low, f_high) = (f(low), f(high));
        if f_low == 0.0 {
            if count == 0 || roots[count - 1] != low {
                roots[count] = low;
                count += 1;
            }
            continue;
        }
        if f_low * f_high > 0.0 {
            continue;
        }
        if f_high == 0.0 && piece + 1 < cut_count {
            continue; // found as the start of the next piece
        }
        for _ in 0..60 {
            let middle = 0.5 * (low + high);
            if f(middle) * f_low > 0.0 { low = middle } else { high = middle }
        }
        if count < 3 {
            roots[count] = 0.5 * (low + high);
            count += 1;
        }
    }
    (roots, count)
}

/// Continuous vertex–triangle test. The vertex moves linearly from `p0` to
/// `p1` while the triangle's corners move from `a0, b0, c0` to
/// `a1, b1, c1`. Returns the first time in `0..=1` at which the vertex lies
/// in the triangle, i.e. passes through it.
///
/// The four points are coplanar when `(p − a) · ((b − a) × (c − a)) = 0`,
/// a cubic in time (Provot 1997); each root is then checked for the point
/// actually being inside the triangle. `tolerance` widens the triangle a
/// little (in units of length) so nothing slips between two neighbours.
pub fn vertex_triangle_impact(p: [Vec3A; 2], a: [Vec3A; 2], b: [Vec3A; 2], c: [Vec3A; 2], tolerance: f32) -> Option<f32> {
    let d = |v: Vec3A| DVec3::new(v.x as f64, v.y as f64, v.z as f64);
    // Everything relative to `a`.
    let (x0, e10, e20) = (d(p[0] - a[0]), d(b[0] - a[0]), d(c[0] - a[0]));
    let (dx, de1, de2) = (d(p[1] - a[1]) - x0, d(b[1] - a[1]) - e10, d(c[1] - a[1]) - e20);
    let n0 = e10.cross(e20);
    let n1 = de1.cross(e20) + e10.cross(de2);
    let n2 = de1.cross(de2);
    let coefficients = [x0.dot(n0), dx.dot(n0) + x0.dot(n1), dx.dot(n1) + x0.dot(n2), dx.dot(n2)];

    let (roots, count) = cubic_roots_in_unit_interval(coefficients);
    for &t in &roots[..count] {
        let tf = t as f32;
        let at = |v: [Vec3A; 2]| v[0] + (v[1] - v[0]) * tf;
        let (pt, at_, bt, ct) = (at(p), at(a), at(b), at(c));
        let (closest, _) = closest_point_on_triangle(pt, at_, bt, ct);
        if closest.distance_squared(pt) <= tolerance * tolerance {
            return Some(tf);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Vec3A = Vec3A::new(0.0, 0.0, 0.0);
    const B: Vec3A = Vec3A::new(1.0, 0.0, 0.0);
    const C: Vec3A = Vec3A::new(0.0, 1.0, 0.0);

    #[test]
    fn segment_hits_and_misses() {
        let (t, normal) =
            segment_triangle(Vec3A::new(0.2, 0.2, 1.0), Vec3A::new(0.2, 0.2, -1.0), A, B, C).expect("crosses");
        assert!((t - 0.5).abs() < 1e-6);
        assert!((normal - Vec3A::Z).length() < 1e-6);
        assert!(segment_triangle(Vec3A::new(0.9, 0.9, 1.0), Vec3A::new(0.9, 0.9, -1.0), A, B, C).is_none());
        assert!(segment_triangle(Vec3A::new(0.2, 0.2, 1.0), Vec3A::new(0.2, 0.2, 0.5), A, B, C).is_none());
    }

    #[test]
    fn closest_point_covers_face_edge_and_corner() {
        let (face, weights) = closest_point_on_triangle(Vec3A::new(0.2, 0.3, 2.0), A, B, C);
        assert!((face - Vec3A::new(0.2, 0.3, 0.0)).length() < 1e-6);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let (edge, _) = closest_point_on_triangle(Vec3A::new(0.5, -1.0, 0.0), A, B, C);
        assert!((edge - Vec3A::new(0.5, 0.0, 0.0)).length() < 1e-6);
        let (corner, _) = closest_point_on_triangle(Vec3A::new(2.0, -1.0, 0.0), A, B, C);
        assert!((corner - B).length() < 1e-6);
    }

    #[test]
    fn cubic_roots_are_found() {
        // (t − 0.2)(t − 0.5)(t − 0.9)
        let (roots, count) = cubic_roots_in_unit_interval([-0.09, 0.73, -1.6, 1.0]);
        assert_eq!(count, 3);
        for (found, expected) in roots.iter().zip([0.2, 0.5, 0.9]) {
            assert!((found - expected).abs() < 1e-9);
        }
        // Linear: 2t − 1.
        let (roots, count) = cubic_roots_in_unit_interval([-1.0, 2.0, 0.0, 0.0]);
        assert!(count == 1 && (roots[0] - 0.5).abs() < 1e-9);
        assert_eq!(cubic_roots_in_unit_interval([1.0, 0.0, 0.0, 1.0]).1, 0);
    }

    #[test]
    fn a_vertex_passing_through_a_moving_triangle_is_caught() {
        // The triangle rises by 1 while the vertex falls by 1: they meet
        // half way, even though start and end are 1 apart on either side.
        let up = Vec3A::Z;
        let p = [Vec3A::new(0.2, 0.2, 1.0), Vec3A::new(0.2, 0.2, 0.0)];
        let t = vertex_triangle_impact(p, [A, A + up], [B, B + up], [C, C + up], 1e-4).expect("passes through");
        assert!((t - 0.5).abs() < 1e-5);
        // Past the triangle's edge: coplanar at the same time, but outside.
        let beside = [Vec3A::new(0.9, 0.9, 1.0), Vec3A::new(0.9, 0.9, 0.0)];
        assert!(vertex_triangle_impact(beside, [A, A + up], [B, B + up], [C, C + up], 1e-4).is_none());
        // Moving along with the triangle: never meets it.
        let along = [Vec3A::new(0.2, 0.2, 0.5), Vec3A::new(0.2, 0.2, 1.5)];
        assert!(vertex_triangle_impact(along, [A, A + up], [B, B + up], [C, C + up], 1e-4).is_none());
    }

    #[test]
    fn a_rotating_triangle_sweeps_up_a_resting_vertex() {
        // The triangle swings about the x axis through 90°; a vertex at
        // rest in its path is hit although it never moves.
        let swung = |v: Vec3A| Vec3A::new(v.x, 0.0, v.y);
        let p = Vec3A::new(0.2, 0.3, 0.3);
        let t = vertex_triangle_impact([p, p], [A, swung(A)], [B, swung(B)], [C, swung(C)], 1e-4);
        assert!(t.is_some());
    }
}

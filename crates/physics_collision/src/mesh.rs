//! A static triangle mesh behind a BVH: the avatar as the cloth sees it.

use glam::Vec3A;
use rayon::prelude::*;

use crate::bvh::{Aabb, Bvh};
use crate::ccd::{closest_point_on_triangle, segment_triangle};

/// Where a ray or a moving point meets the surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceHit {
    pub triangle: u32,
    pub point: Vec3A,
    /// Unit normal of the triangle (counter-clockwise front).
    pub normal: Vec3A,
    /// Distance along the ray.
    pub distance: f32,
}

/// The surface point nearest to a query point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clearance {
    pub triangle: u32,
    pub point: Vec3A,
    pub distance: f32,
}

#[derive(Debug, Clone, Default)]
pub struct TriangleMesh {
    positions: Vec<Vec3A>,
    triangles: Vec<[u32; 3]>,
    bvh: Bvh,
}

impl TriangleMesh {
    /// `triangles` must be wound counter-clockwise seen from outside.
    pub fn new(positions: Vec<Vec3A>, triangles: Vec<[u32; 3]>) -> Self {
        let boxes: Vec<Aabb> =
            triangles.iter().map(|t| Aabb::from_points(t.map(|i| positions[i as usize]))).collect();
        Self { bvh: Bvh::build(&boxes), positions, triangles }
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    pub fn bvh(&self) -> &Bvh {
        &self.bvh
    }

    #[inline]
    fn corners(&self, triangle: u32) -> [Vec3A; 3] {
        self.triangles[triangle as usize].map(|i| self.positions[i as usize])
    }

    /// The first intersection of the ray with the surface within
    /// `max_distance`. `direction` must be a unit vector.
    pub fn raycast(&self, origin: Vec3A, direction: Vec3A, max_distance: f32) -> Option<SurfaceHit> {
        let end = origin + direction * max_distance;
        let (triangle, t) = self.bvh.raycast(origin, direction, max_distance, |triangle, _| {
            let [a, b, c] = self.corners(triangle);
            segment_triangle(origin, end, a, b, c).map(|(t, _)| t * max_distance)
        })?;
        let [a, b, c] = self.corners(triangle);
        Some(SurfaceHit {
            triangle,
            point: origin + direction * t,
            normal: (b - a).cross(c - a).normalize_or_zero(),
            distance: t,
        })
    }

    /// The surface point nearest to `p`, if within `max_distance`.
    pub fn closest_point(&self, p: Vec3A, max_distance: f32) -> Option<Clearance> {
        let (triangle, distance) = self.bvh.closest(p, max_distance, |triangle| {
            let [a, b, c] = self.corners(triangle);
            closest_point_on_triangle(p, a, b, c).0.distance_squared(p)
        })?;
        let [a, b, c] = self.corners(triangle);
        Some(Clearance { triangle, point: closest_point_on_triangle(p, a, b, c).0, distance })
    }

    /// Continuous collision of one moving point with the surface: if the
    /// path `from → to` enters the mesh through a front face, returns where
    /// the point must stop, `margin` off the surface, with the normal
    /// there. Paths that leave the mesh are let through, so a point that
    /// did end up inside can get out again.
    pub fn sweep(&self, from: Vec3A, to: Vec3A, margin: f32) -> Option<(Vec3A, Vec3A)> {
        let path = to - from;
        let length = path.length();
        if length < 1e-9 {
            return None;
        }
        let direction = path / length;
        let mut entry: Option<(f32, Vec3A)> = None;
        // Not `raycast`: the nearest hit may be an exit, with an entry
        // behind it.
        self.bvh.raycast(from, direction, length, |triangle, _| {
            let [a, b, c] = self.corners(triangle);
            if let Some((t, normal)) = segment_triangle(from, to, a, b, c)
                && normal.dot(direction) < 0.0
                && entry.is_none_or(|(best, _)| t < best)
            {
                entry = Some((t, normal));
            }
            None
        });
        entry.map(|(t, normal)| (from + path * t + normal * margin, normal))
    }

    /// [`TriangleMesh::sweep`] for every point of a cloth: `positions` are
    /// moved back out where their path from `start` went into the mesh.
    /// Returns the indices that were corrected and the surface normal for
    /// each.
    pub fn sweep_all(&self, start: &[Vec3A], positions: &mut [Vec3A], margin: f32) -> Vec<(u32, Vec3A)> {
        let fixes: Vec<(u32, Vec3A, Vec3A)> = start
            .par_iter()
            .zip(&*positions)
            .enumerate()
            .with_min_len(256)
            .filter_map(|(i, (&from, &to))| self.sweep(from, to, margin).map(|(point, normal)| (i as u32, point, normal)))
            .collect();
        fixes
            .into_iter()
            .map(|(i, point, normal)| {
                positions[i as usize] = point;
                (i, normal)
            })
            .collect()
    }

    /// Clearance between a garment point and the body, measured the way a
    /// fit map wants it: along the ray from the point straight towards the
    /// body (against the cloth's normal), falling back to the nearest
    /// surface point where that ray misses (e.g. cloth facing a gap between
    /// arm and torso). `None` beyond `max_distance`.
    pub fn clearance(&self, p: Vec3A, cloth_normal: Vec3A, max_distance: f32) -> Option<f32> {
        let nearest = self.closest_point(p, max_distance)?;
        if nearest.distance < 1e-6 {
            return Some(0.0);
        }
        // The cloth normal has no preferred side; aim at the body.
        let towards = (nearest.point - p) / nearest.distance;
        let Some(normal) = cloth_normal.try_normalize() else { return Some(nearest.distance) };
        let direction = if normal.dot(towards) >= 0.0 { normal } else { -normal };
        // A ray that grazes past the body would report the far side.
        let limit = (nearest.distance * 3.0).min(max_distance);
        Some(self.raycast(p, direction, limit).map_or(nearest.distance, |hit| hit.distance))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube centred on the origin, outward-facing.
    fn cube() -> TriangleMesh {
        let positions: Vec<Vec3A> = (0..8)
            .map(|i| Vec3A::new((i & 1) as f32 - 0.5, ((i >> 1) & 1) as f32 - 0.5, ((i >> 2) & 1) as f32 - 0.5))
            .collect();
        let quads = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
        let triangles = quads.iter().flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]).collect();
        TriangleMesh::new(positions, triangles)
    }

    #[test]
    fn rays_hit_the_near_face_with_an_outward_normal() {
        let hit = cube().raycast(Vec3A::new(0.1, 0.2, 3.0), -Vec3A::Z, 10.0).expect("hits the cube");
        assert!((hit.distance - 2.5).abs() < 1e-5);
        assert!((hit.normal - Vec3A::Z).length() < 1e-5);
        assert!(cube().raycast(Vec3A::new(0.1, 0.2, 3.0), Vec3A::Z, 10.0).is_none());
    }

    #[test]
    fn a_point_moving_into_the_mesh_is_stopped_at_the_surface() {
        let cube = cube();
        // Fast enough to end up on the far side: a discrete test at the
        // end position would see nothing wrong.
        let (stopped, normal) =
            cube.sweep(Vec3A::new(0.0, 0.0, 2.0), Vec3A::new(0.0, 0.0, -2.0), 0.01).expect("tunnels through");
        assert!((stopped - Vec3A::new(0.0, 0.0, 0.51)).length() < 1e-5);
        assert!((normal - Vec3A::Z).length() < 1e-5);
        // Leaving is allowed, passing by is not a hit.
        assert!(cube.sweep(Vec3A::ZERO, Vec3A::new(0.0, 0.0, 2.0), 0.01).is_none());
        assert!(cube.sweep(Vec3A::new(2.0, 0.0, 2.0), Vec3A::new(2.0, 0.0, -2.0), 0.01).is_none());
    }

    #[test]
    fn clearance_is_measured_along_the_cloth_normal() {
        let cube = cube();
        let p = Vec3A::new(0.0, 0.0, 0.53);
        assert!((cube.clearance(p, Vec3A::Z, 1.0).unwrap() - 0.03).abs() < 1e-5);
        // The normal's sign does not matter.
        assert!((cube.clearance(p, -Vec3A::Z, 1.0).unwrap() - 0.03).abs() < 1e-5);
        // Tilted cloth: the ray is longer than the shortest distance.
        let tilted = Vec3A::new(0.0, 0.6, 0.8);
        assert!((cube.clearance(p, tilted, 1.0).unwrap() - 0.03 / 0.8).abs() < 1e-5);
        assert!(cube.clearance(Vec3A::new(0.0, 0.0, 5.0), Vec3A::Z, 1.0).is_none());
    }
}

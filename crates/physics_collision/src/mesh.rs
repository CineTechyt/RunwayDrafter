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

/// How far in front of a skin point other skin makes it a crease, in
/// metres: about where cloth under tension stops following a fold.
const CREASE: f32 = 0.015;

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
    /// Area-weighted outward vertex normals.
    normals: Vec<Vec3A>,
    triangles: Vec<[u32; 3]>,
    bvh: Bvh,
}

impl TriangleMesh {
    /// `triangles` must be wound counter-clockwise seen from outside.
    pub fn new(positions: Vec<Vec3A>, triangles: Vec<[u32; 3]>) -> Self {
        let boxes: Vec<Aabb> = triangles
            .iter()
            .map(|t| Aabb::from_points(t.map(|i| positions[i as usize])))
            .collect();
        let mut normals = vec![Vec3A::ZERO; positions.len()];
        for t in &triangles {
            let [a, b, c] = t.map(|i| positions[i as usize]);
            let n = (b - a).cross(c - a);
            for &i in t {
                normals[i as usize] += n;
            }
        }
        normals.iter_mut().for_each(|n| *n = n.normalize_or_zero());
        Self {
            bvh: Bvh::build(&boxes),
            positions,
            normals,
            triangles,
        }
    }

    /// Vertex positions.
    pub fn positions(&self) -> &[Vec3A] {
        &self.positions
    }

    /// Area-weighted outward unit normals, one per vertex.
    pub fn vertex_normals(&self) -> &[Vec3A] {
        &self.normals
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
    pub fn raycast(
        &self,
        origin: Vec3A,
        direction: Vec3A,
        max_distance: f32,
    ) -> Option<SurfaceHit> {
        let end = origin + direction * max_distance;
        let (triangle, t) = self
            .bvh
            .raycast(origin, direction, max_distance, |triangle, _| {
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
        Some(Clearance {
            triangle,
            point: closest_point_on_triangle(p, a, b, c).0,
            distance,
        })
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
    pub fn sweep_all(
        &self,
        start: &[Vec3A],
        positions: &mut [Vec3A],
        margin: f32,
    ) -> Vec<(u32, Vec3A)> {
        let fixes: Vec<(u32, Vec3A, Vec3A)> = start
            .par_iter()
            .zip(&*positions)
            .enumerate()
            .with_min_len(256)
            .filter_map(|(i, (&from, &to))| {
                self.sweep(from, to, margin)
                    .map(|(point, normal)| (i as u32, point, normal))
            })
            .collect();
        fixes
            .into_iter()
            .map(|(i, point, normal)| {
                positions[i as usize] = point;
                (i, normal)
            })
            .collect()
    }

    /// The reverse of [`TriangleMesh::sweep_all`]: body features smaller
    /// than a cloth triangle (a collarbone, a nipple, the point of a
    /// shoulder) poke through the cloth between its vertices, which a test
    /// of cloth points against the body never sees. Every body vertex is
    /// tested against the cloth triangles near it; where one is closer
    /// than half of `margin` in front of it, or up to `depth` behind it, that
    /// triangle is pushed out along the body normal, the push shared over
    /// its corners by barycentric weight (Bridson et al. 2002). Returns the
    /// cloth vertices moved and how far.
    ///
    /// `field`, where given, is the body's distance field: distance and
    /// outward direction at a point. With it, skin in a crease (the top of
    /// a shoulder with the arm raised, an armpit, under the bust) pushes
    /// nothing: cloth bridges a fold rather than following it in, and
    /// cloth lying on one side of a fold is in front of skin points on the
    /// other, so pushing it along their normal would drive it into the
    /// body. A push is also only made where the field agrees which way is
    /// out.
    pub fn push_out_cloth(
        &self,
        cloth: &[[u32; 3]],
        positions: &mut [Vec3A],
        margin: f32,
        depth: f32,
        field: Option<&(dyn Fn(Vec3A) -> Option<(f32, Vec3A)> + Sync)>,
    ) -> Vec<(u32, Vec3A)> {
        if cloth.is_empty() || positions.is_empty() {
            return Vec::new();
        }
        let reach = margin.max(depth);
        let boxes: Vec<Aabb> = cloth
            .iter()
            .map(|t| Aabb::from_points(t.map(|i| positions[i as usize])).expanded(reach))
            .collect();
        let tree = Bvh::build(&boxes);
        let bounds = tree.bounds();
        let x = &*positions;
        // Per body vertex: the cloth triangle it pushes, the weights and
        // how far along which normal.
        let pushes: Vec<([u32; 3], [f32; 3], Vec3A)> = self
            .positions
            .par_iter()
            .zip(&self.normals)
            .with_min_len(512)
            .filter(|(p, n)| n.length_squared() > 0.5 && bounds.distance_squared(**p) == 0.0)
            .filter_map(|(&p, &n)| {
                // Another stretch of skin close in front: a crease.
                if let Some(field) = field
                    && field(p + n * CREASE).is_none_or(|(distance, _)| distance < CREASE * 0.6)
                {
                    return None;
                }
                let mut best: Option<([u32; 3], [f32; 3], f32)> = None;
                tree.query(&Aabb::from_points([p]), |triangle| {
                    let t = cloth[triangle as usize];
                    let [a, b, c] = t.map(|i| x[i as usize]);
                    let (q, w) = closest_point_on_triangle(p, a, b, c);
                    let offset = q - p;
                    let height = offset.dot(n);
                    // The body point must lie under the face, not beside
                    // it: what is off to the side is the vertices' job.
                    let under = w.iter().all(|&w| w > 0.0) || (offset - n * height).length() < 1e-4;
                    // Pushed to the margin, tested at half of it: a face
                    // just lifted does not count again.
                    if height < margin * 0.5 && height > -depth && under {
                        let need = margin - height;
                        if best.is_none_or(|(_, _, most)| need > most) {
                            best = Some((t, w, need));
                        }
                    }
                });
                let (t, w, need) = best?;
                if let Some(field) = field {
                    let [a, b, c] = t.map(|i| x[i as usize]);
                    let q = a * w[0] + b * w[1] + c * w[2];
                    if field(q).is_none_or(|(_, o)| o.dot(n) < 0.7) {
                        return None;
                    }
                }
                Some((t, w, n * need))
            })
            .collect();
        let mut moved = Vec::new();
        if pushes.is_empty() {
            return moved;
        }
        let mut sum = vec![Vec3A::ZERO; positions.len()];
        let mut count = vec![0u32; positions.len()];
        for (t, w, push) in &pushes {
            // The smallest corner moves that lift the face's point at `w`
            // by `push`: corner i moves w_i / Σw² of it.
            let norm = w.iter().map(|w| w * w).sum::<f32>().max(1e-6);
            for k in 0..3 {
                let i = t[k] as usize;
                if w[k] > 1e-3 {
                    sum[i] += *push * (w[k] / norm);
                    count[i] += 1;
                }
            }
        }
        for (i, (&s, &c)) in sum.iter().zip(&count).enumerate() {
            if c > 0 {
                // Averaged, so several body points under one face do not
                // add up to a launch.
                // At most `margin` per round: a face found deep under the skin
                // comes out over a few steps, not in a jolt that tears it
                // from its neighbours.
                let delta = (s / c as f32).clamp_length_max(margin);
                positions[i] += delta;
                moved.push((i as u32, delta));
            }
        }
        moved
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
        let Some(normal) = cloth_normal.try_normalize() else {
            return Some(nearest.distance);
        };
        let direction = if normal.dot(towards) >= 0.0 {
            normal
        } else {
            -normal
        };
        // A ray that grazes past the body would report the far side.
        let limit = (nearest.distance * 3.0).min(max_distance);
        Some(
            self.raycast(p, direction, limit)
                .map_or(nearest.distance, |hit| hit.distance),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube centred on the origin, outward-facing.
    fn cube() -> TriangleMesh {
        let positions: Vec<Vec3A> = (0..8)
            .map(|i| {
                Vec3A::new(
                    (i & 1) as f32 - 0.5,
                    ((i >> 1) & 1) as f32 - 0.5,
                    ((i >> 2) & 1) as f32 - 0.5,
                )
            })
            .collect();
        let quads = [
            [0, 2, 3, 1],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 4, 6, 2],
            [1, 3, 7, 5],
        ];
        let triangles = quads
            .iter()
            .flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
            .collect();
        TriangleMesh::new(positions, triangles)
    }

    #[test]
    fn rays_hit_the_near_face_with_an_outward_normal() {
        let hit = cube()
            .raycast(Vec3A::new(0.1, 0.2, 3.0), -Vec3A::Z, 10.0)
            .expect("hits the cube");
        assert!((hit.distance - 2.5).abs() < 1e-5);
        assert!((hit.normal - Vec3A::Z).length() < 1e-5);
        assert!(
            cube()
                .raycast(Vec3A::new(0.1, 0.2, 3.0), Vec3A::Z, 10.0)
                .is_none()
        );
    }

    #[test]
    fn a_point_moving_into_the_mesh_is_stopped_at_the_surface() {
        let cube = cube();
        // Fast enough to end up on the far side: a discrete test at the
        // end position would see nothing wrong.
        let (stopped, normal) = cube
            .sweep(Vec3A::new(0.0, 0.0, 2.0), Vec3A::new(0.0, 0.0, -2.0), 0.01)
            .expect("tunnels through");
        assert!((stopped - Vec3A::new(0.0, 0.0, 0.51)).length() < 1e-5);
        assert!((normal - Vec3A::Z).length() < 1e-5);
        // Leaving is allowed, passing by is not a hit.
        assert!(
            cube.sweep(Vec3A::ZERO, Vec3A::new(0.0, 0.0, 2.0), 0.01)
                .is_none()
        );
        assert!(
            cube.sweep(Vec3A::new(2.0, 0.0, 2.0), Vec3A::new(2.0, 0.0, -2.0), 0.01)
                .is_none()
        );
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
        assert!(
            cube.clearance(Vec3A::new(0.0, 0.0, 5.0), Vec3A::Z, 1.0)
                .is_none()
        );
    }

    #[test]
    fn body_corners_poking_through_a_coarse_cloth_lift_it_off() {
        let cube = cube();
        // One quad far wider than the cube, just under its top: none of
        // its vertices is inside, yet the top corners stick through.
        let mut sheet = vec![
            Vec3A::new(-2.0, -2.0, 0.49),
            Vec3A::new(2.0, -2.0, 0.49),
            Vec3A::new(2.0, 2.0, 0.49),
            Vec3A::new(-2.0, 2.0, 0.49),
        ];
        let faces = [[0, 1, 2], [0, 2, 3]];
        assert!(
            sheet
                .iter()
                .all(|&p| cube.sweep(p + Vec3A::Z, p, 0.0).is_none())
        );
        let mut rounds = 0;
        while !cube
            .push_out_cloth(&faces, &mut sheet, 0.005, 0.02, None)
            .is_empty()
        {
            rounds += 1;
            assert!(rounds < 20, "settles");
        }
        assert!(rounds > 0);
        // Above every top corner now, by the margin, and not flung away.
        for corner in cube.positions().iter().filter(|p| p.z > 0.0) {
            let up = cube.normals[cube.positions.iter().position(|p| p == corner).unwrap()];
            let hit = TriangleMesh::new(sheet.clone(), faces.to_vec()).raycast(
                *corner - up * 0.1,
                up,
                1.0,
            );
            assert!(
                hit.is_some_and(|h| h.distance > 0.1),
                "corner {corner} under the cloth"
            );
        }
        assert!(sheet.iter().all(|p| p.z < 0.6), "{sheet:?}");
    }
}

//! Cloth against cloth: self-collision and garment layers.
//!
//! Once per simulation step, [`ClothContacts::detect`] looks at every
//! vertex and the triangles near its path during that step, and produces
//! up to [`PLANES_PER_VERTEX`] *contact planes* per vertex: half-spaces the
//! vertex must stay in. The solver enforces them in every sub-step of the
//! next step, which costs a dot product per plane and runs unchanged on the
//! CPU and in the compute shader.
//!
//! Which side of a triangle a vertex belongs on is decided as follows.
//!
//! * **Same garment layer** — the side it is on, unless the continuous
//!   test ([`crate::ccd::vertex_triangle_impact`]) shows it passed through
//!   the triangle during the step; then the side it came from, and the
//!   vertex is put back at once.
//! * **Different layers** — the higher layer belongs further from the
//!   body. `outward` gives the direction away from the body, so a jacket
//!   vertex found under a shirt triangle is pushed out however it got
//!   there.
//!
//! Each vertex takes half of the correction: the triangle's own vertices
//! are in turn tested against the triangles around this vertex and take
//! the other half.
//!
//! Contacts are *speculative*: a vertex gets a plane for cloth that is
//! merely near (a few thicknesses away), placed so that the two may close
//! half of the spare distance during the next step and no more. Cloth
//! that approaches other cloth is therefore slowed down before it
//! arrives, instead of being found on the wrong side afterwards; the
//! continuous test remains as the safety net for what comes in faster
//! than that.

use glam::Vec3A;
use rayon::prelude::*;

use crate::bvh::{Aabb, Bvh};
use crate::ccd::{closest_point_on_triangle, vertex_triangle_impact};

pub const PLANES_PER_VERTEX: usize = 4;

/// Steps between full rebuilds of the hierarchy; in between it is refit.
const REBUILD_INTERVAL: u32 = 24;
/// The fabric thickness is capped at this fraction of the mean edge
/// length: beyond it, panels that meet at an angle along a seam would
/// count as touching.
const MAX_THICKNESS_PER_EDGE: f32 = 0.3;
/// Cloth within this many thicknesses gets a speculative contact …
const NEAR_THICKNESSES: f32 = 5.0;
/// … but never further than this fraction of the mean edge length, or a
/// vertex would be in "contact" with its neighbours in the mesh.
const MAX_NEAR_PER_EDGE: f32 = 0.6;
/// Cloth of the same piece this close along it, in mean edge lengths, is
/// not tested against: to lie on itself it would have to crease tighter
/// than the mesh can. Tested, cloth bunched at an ankle counts as stacked
/// layers, is held apart and gripped like them, and moves as one stiff
/// lump. (Along the piece only: across seams, where pieces meet at an
/// angle, the crotch, cloth still collides.)
const OWN_REACH: f32 = 2.0;


/// A vertex against a point of another triangle: the vertex is to keep
/// `normal · (x − Σ weights · triangle) ≥ target`. The plane of
/// [`ClothContacts::planes`] is this pair with the triangle held where it
/// was found; a solver that moves both sides at once uses the pair.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ContactPair {
    pub triangle: [u32; 3],
    pub weights: [f32; 3],
    pub normal: Vec3A,
    pub target: f32,
}

impl ContactPair {
    /// The point of the triangle the vertex faces, at positions `x`.
    #[inline]
    pub fn point(&self, x: &[Vec3A]) -> Vec3A {
        (0..3).fold(Vec3A::ZERO, |sum, k| {
            sum + x[self.triangle[k] as usize] * self.weights[k]
        })
    }

    /// How far vertex `p` is on the wrong side (negative: on the right one).
    #[inline]
    pub fn depth(&self, p: Vec3A, x: &[Vec3A]) -> f32 {
        self.target - self.normal.dot(p - self.point(x))
    }
}

/// The half-space `normal · x + offset ≥ 0`. All zero means unused. The
/// layout is a `vec4<f32>` for the compute shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ContactPlane {
    pub normal: [f32; 3],
    pub offset: f32,
}

impl ContactPlane {
    pub const NONE: Self = Self {
        normal: [0.0; 3],
        offset: 0.0,
    };

    #[inline]
    pub fn is_active(&self) -> bool {
        self.normal != [0.0; 3]
    }

    /// Signed distance of `x` into the allowed half-space.
    #[inline]
    pub fn distance(&self, x: Vec3A) -> f32 {
        Vec3A::from(self.normal).dot(x) + self.offset
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactParams {
    /// Distance kept between two cloth surfaces, in metres.
    pub thickness: f32,
    /// How far on the wrong side of another layer a vertex is still
    /// brought back, in metres. Beyond it the triangle is taken to belong
    /// to another part of the garment (the far side of a sleeve, say).
    pub layer_reach: f32,
    /// Whether a vertex found on the wrong side of another layer is moved
    /// to the right one at once. If not, it only gets the plane, and the
    /// solver brings it round.
    pub move_layers: bool,
}

impl Default for ContactParams {
    fn default() -> Self {
        Self {
            thickness: 0.0025,
            layer_reach: 0.02,
            move_layers: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ContactStats {
    /// Vertices closer to other cloth than the fabric thickness.
    pub contacts: usize,
    /// Vertices that had passed through cloth and were put back.
    pub crossings: usize,
}

#[derive(Debug, Clone, Default)]
pub struct ClothContacts {
    triangles: Vec<[u32; 3]>,
    /// Garment layer of every vertex.
    layers: Vec<u32>,
    /// Vertices sewn together share a group; a vertex is never tested
    /// against a triangle that contains a member of its own group.
    groups: Vec<u32>,
    /// Per vertex, the vertices of its own piece within [`OWN_REACH`]
    /// along it, sorted (`near[near_start[v]..near_start[v + 1]]`).
    near_start: Vec<u32>,
    near: Vec<u32>,
    layered: bool,
    mean_edge: f32,
    bvh: Bvh,
    boxes: Vec<Aabb>,
    steps_since_build: u32,
    planes: Vec<[ContactPlane; PLANES_PER_VERTEX]>,
    /// The two strongest planes per vertex as pairs (see [`ContactPair`]).
    pairs: Vec<[Option<ContactPair>; 2]>,
    /// Vertices the last detection moved.
    moved: Vec<u32>,
}

impl ClothContacts {
    /// `layers` holds one entry per vertex; `stitches` are the vertex pairs
    /// sewn together; `rest` are any positions with undistorted triangles
    /// (used for the mean edge length only).
    pub fn new(
        triangles: Vec<[u32; 3]>,
        layers: Vec<u32>,
        stitches: &[(u32, u32)],
        rest: &[Vec3A],
    ) -> Self {
        // Union-find over the stitches.
        let mut groups: Vec<u32> = (0..layers.len() as u32).collect();
        fn root(groups: &mut [u32], mut v: u32) -> u32 {
            while groups[v as usize] != v {
                groups[v as usize] = groups[groups[v as usize] as usize];
                v = groups[v as usize];
            }
            v
        }
        for &(a, b) in stitches {
            let (ra, rb) = (root(&mut groups, a), root(&mut groups, b));
            groups[ra.max(rb) as usize] = ra.min(rb);
        }
        for v in 0..groups.len() as u32 {
            groups[v as usize] = root(&mut groups, v);
        }

        let (mut total, mut count) = (0.0f64, 0u32);
        for t in &triangles {
            for k in 0..3 {
                total += rest[t[k] as usize].distance(rest[t[(k + 1) % 3] as usize]) as f64;
                count += 1;
            }
        }
        let layered = layers.iter().any(|&layer| layer != layers[0]);
        let mean = if count > 0 {
            (total / count as f64) as f32
        } else {
            0.0
        };
        let (near_start, near) =
            neighbourhoods(layers.len(), &triangles, rest, OWN_REACH * mean);
        let mean_edge = if count > 0 {
            (total / count as f64) as f32
        } else {
            0.0
        };
        Self {
            planes: vec![[ContactPlane::NONE; PLANES_PER_VERTEX]; layers.len()],
            pairs: vec![[None; 2]; layers.len()],
            mean_edge,
            near_start,
            near,
            triangles,
            layers,
            groups,
            layered,
            ..Self::default()
        }
    }

    /// The planes found by the last [`ClothContacts::detect`], per vertex.
    pub fn planes(&self) -> &[[ContactPlane; PLANES_PER_VERTEX]] {
        &self.planes
    }

    /// The same contacts as pairs of vertex and triangle, per vertex.
    pub fn pairs(&self) -> &[[Option<ContactPair>; 2]] {
        &self.pairs
    }

    /// The vertices whose position the last [`ClothContacts::detect`]
    /// corrected.
    pub fn moved(&self) -> &[u32] {
        &self.moved
    }

    /// Forgets all planes, e.g. after the cloth was moved from outside.
    /// Returns `true` if there were any.
    pub fn clear(&mut self) -> bool {
        let mut any = false;
        for planes in &mut self.planes {
            any |= planes[0].is_active();
            *planes = [ContactPlane::NONE; PLANES_PER_VERTEX];
        }
        self.pairs.fill([None; 2]);
        self.moved.clear();
        any
    }

    /// The cloth as a whole was moved by `delta` (it travels with its
    /// wearer): the planes go with it.
    pub fn translate(&mut self, delta: Vec3A) {
        for plane in self.planes.iter_mut().flatten() {
            if plane.is_active() {
                plane.offset -= Vec3A::from(plane.normal).dot(delta);
            }
        }
    }

    /// The fabric thickness actually used for a requested one.
    pub fn effective_thickness(&self, requested: f32) -> f32 {
        requested
            .min(self.mean_edge * MAX_THICKNESS_PER_EDGE)
            .max(0.0)
    }

    /// Finds the contacts for the step that moved the cloth from `start`
    /// to `positions`. Vertices that passed through cloth, or sit on the
    /// wrong side of another layer, are moved back in `positions`.
    ///
    /// `outward(p)` is the unit direction away from the body at `p`, where
    /// known; without it layers are told apart like one garment.
    pub fn detect(
        &mut self,
        start: &[Vec3A],
        positions: &mut [Vec3A],
        params: &ContactParams,
        outward: Option<&(dyn Fn(Vec3A) -> Option<Vec3A> + Sync)>,
    ) -> ContactStats {
        self.moved.clear();
        let vertex_count = self.layers.len();
        if vertex_count == 0 || start.len() != vertex_count || positions.len() != vertex_count {
            return ContactStats::default();
        }
        let thickness = self.effective_thickness(params.thickness);
        let near = (NEAR_THICKNESSES * thickness)
            .min(MAX_NEAR_PER_EDGE * self.mean_edge)
            .max(thickness);
        let reach = if self.layered && outward.is_some() {
            params.layer_reach.max(near)
        } else {
            near
        };

        // Boxes around each triangle's path during the step.
        self.boxes.clear();
        self.boxes.extend(self.triangles.iter().map(|t| {
            Aabb::from_points(
                t.iter()
                    .flat_map(|&i| [start[i as usize], positions[i as usize]]),
            )
        }));
        if self.bvh.primitive_count() != self.boxes.len()
            || self.steps_since_build >= REBUILD_INTERVAL
        {
            self.bvh = Bvh::build(&self.boxes);
            self.steps_since_build = 0;
        } else {
            self.bvh.refit(&self.boxes);
            self.steps_since_build += 1;
        }

        let end: &[Vec3A] = positions;
        let mut planes = std::mem::take(&mut self.planes);
        let mut pairs = std::mem::take(&mut self.pairs);
        let this = &*self;
        let mut touching = vec![false; vertex_count];
        let results: Vec<(Option<Vec3A>, bool)> = planes
            .par_iter_mut()
            .zip(&mut touching)
            .zip(&mut pairs)
            .enumerate()
            .with_min_len(128)
            .map(|(v, ((planes, touching), pairs))| {
                this.vertex_contacts(
                    v as u32,
                    start,
                    end,
                    [thickness, near, reach],
                    params,
                    outward,
                    planes,
                    pairs,
                    touching,
                )
            })
            .collect();
        self.planes = planes;
        self.pairs = pairs;

        let mut stats = ContactStats::default();
        for (v, (moved, crossed)) in results.into_iter().enumerate() {
            if let Some(position) = moved {
                positions[v] = position;
                self.moved.push(v as u32);
            }
            stats.crossings += usize::from(crossed);
            // Before the correction: the plane was violated.
            stats.contacts += usize::from(touching[v]);
        }
        stats
    }

    /// Fills `planes` for vertex `v`. Returns its corrected position if it
    /// has to be moved right away, and whether it had passed through cloth.
    #[allow(clippy::too_many_arguments)]
    fn vertex_contacts(
        &self,
        v: u32,
        start: &[Vec3A],
        end: &[Vec3A],
        [thickness, near, reach]: [f32; 3],
        params: &ContactParams,
        outward: Option<&(dyn Fn(Vec3A) -> Option<Vec3A> + Sync)>,
        planes: &mut [ContactPlane; PLANES_PER_VERTEX],
        pairs: &mut [Option<ContactPair>; 2],
        touching: &mut bool,
    ) -> (Option<Vec3A>, bool) {
        *planes = [ContactPlane::NONE; PLANES_PER_VERTEX];
        *pairs = [None; 2];
        let (p0, p1) = (start[v as usize], end[v as usize]);
        let group = self.groups[v as usize];
        let layer = self.layers[v as usize];
        // Violations of the planes kept so far, largest first. Negative
        // for speculative contacts: the spare distance left.
        let mut kept = [f32::NEG_INFINITY; PLANES_PER_VERTEX];
        // The vertex's own path, for the continuous test.
        let path = Aabb::from_points([p0, p1]).expanded(1e-4);
        let mut fix: Option<(f32, Vec3A)> = None;
        let mut crossed_any = false;

        let query = Aabb::from_points([p0, p1]).expanded(reach);
        self.bvh.query(&query, |triangle| {
            if !self.boxes[triangle as usize].overlaps(&query) {
                return;
            }
            let t = self.triangles[triangle as usize];
            if t.iter().any(|&u| self.groups[u as usize] == group) {
                return;
            }
            let own = &self.near
                [self.near_start[v as usize] as usize..self.near_start[v as usize + 1] as usize];
            if t.iter().any(|u| own.binary_search(u).is_ok()) {
                return;
            }
            let [a1, b1, c1] = t.map(|i| end[i as usize]);
            let Some(normal) = (b1 - a1).cross(c1 - a1).try_normalize() else {
                return;
            };
            let (closest, weights) = closest_point_on_triangle(p1, a1, b1, c1);
            let offset = p1 - closest;
            let distance = offset.length();
            let height = offset.dot(normal);

            // (unit normal of the allowed side, signed distance on it,
            //  distance to reach, move the vertex now?)
            let mut contact: Option<(Vec3A, f32, f32, bool)> = None;
            let other_layer = self.layers[t[0] as usize];
            let layer_side = (other_layer != layer)
                .then(|| outward.and_then(|outward| outward(closest)))
                .flatten()
                .map(|out| normal.dot(out))
                .filter(|facing| facing.abs() > 0.3);
            if let Some(facing) = layer_side {
                // The higher layer belongs on the triangle's outer face.
                let side = facing.signum() * if layer > other_layer { 1.0 } else { -1.0 };
                let signed = height * side;
                let lateral = (distance * distance - height * height).max(0.0).sqrt();
                if signed < near && signed > -params.layer_reach && lateral < thickness.max(1e-4) {
                    contact = Some((
                        normal * side,
                        signed,
                        (0.5 * (signed + thickness)).max(0.5 * thickness),
                        signed < 0.0 && params.move_layers,
                    ));
                }
            } else {
                let [a0, b0, c0] = t.map(|i| start[i as usize]);
                let start_height = (p0 - a0).dot((b0 - a0).cross(c0 - a0));
                let passed = start_height * height < 0.0
                    && self.boxes[triangle as usize].overlaps(&path)
                    && vertex_triangle_impact([p0, p1], [a0, a1], [b0, b1], [c0, c1], 1e-5)
                        .is_some();
                if passed {
                    // Back to the side it came from.
                    let side = start_height.signum();
                    contact = Some((normal * side, height * side, 0.5 * thickness, true));
                } else if distance < 1e-6 {
                    // Exactly on the triangle (both were stopped by the
                    // same obstacle, say): no direction to read off, so
                    // the side it came from decides.
                    if start_height != 0.0 {
                        contact =
                            Some((normal * start_height.signum(), 0.0, 0.5 * thickness, true));
                    }
                } else if distance < near {
                    // Resting (closer than the thickness: pushed apart) or
                    // speculative (further: may close half of the spare
                    // distance). Along the true direction, which is not
                    // the face normal next to an edge.
                    contact = Some((
                        offset / distance,
                        distance,
                        0.5 * (distance + thickness),
                        false,
                    ));
                }
            }

            let Some((allowed, signed, target, move_now)) = contact else {
                return;
            };
            let violation = target - signed;
            *touching |= violation > 0.0;
            crossed_any |= move_now;
            if move_now && fix.is_none_or(|(worst, _)| violation > worst) {
                fix = Some((violation, p1 + allowed * violation));
            }
            // allowed · (x − closest) ≥ target
            let plane = ContactPlane {
                normal: allowed.to_array(),
                offset: -allowed.dot(closest) - target,
            };
            let pair = Some(ContactPair {
                triangle: t,
                weights,
                normal: allowed,
                target,
            });
            if violation > kept[0] {
                kept[1] = kept[0];
                planes[1] = planes[0];
                pairs[1] = pairs[0];
                kept[0] = violation;
                planes[0] = plane;
                pairs[0] = pair;
            } else if violation > kept[1] {
                kept[1] = violation;
                planes[1] = plane;
                pairs[1] = pair;
            }
        });
        (fix.map(|(_, position)| position), crossed_any)
    }
}

/// For every vertex, the vertices within `reach` of it along the mesh
/// edges at their rest length, sorted.
fn neighbourhoods(
    count: usize,
    triangles: &[[u32; 3]],
    rest: &[Vec3A],
    reach: f32,
) -> (Vec<u32>, Vec<u32>) {
    if reach <= 0.0 || rest.len() < count {
        return (vec![0; count + 1], Vec::new());
    }
    let mut links: Vec<Vec<(u32, f32)>> = vec![Vec::new(); count];
    for t in triangles {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let length = rest[a as usize].distance(rest[b as usize]);
            links[a as usize].push((b, length));
            links[b as usize].push((a, length));
        }
    }
    let found: Vec<Vec<u32>> = (0..count as u32)
        .into_par_iter()
        .with_min_len(64)
        .map(|start| {
            // Dijkstra cut off at `reach`; the neighbourhoods are small.
            let mut best: Vec<(u32, f32)> = vec![(start, 0.0)];
            let mut open: Vec<(u32, f32)> = vec![(start, 0.0)];
            while let Some(index) = open
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.1.total_cmp(&b.1.1))
                .map(|(i, _)| i)
            {
                let (v, d) = open.swap_remove(index);
                for &(u, length) in &links[v as usize] {
                    let e = d + length;
                    if e > reach {
                        continue;
                    }
                    match best.iter_mut().find(|(w, _)| *w == u) {
                        Some((_, known)) if *known <= e => {}
                        Some((_, known)) => {
                            *known = e;
                            open.push((u, e));
                        }
                        None => {
                            best.push((u, e));
                            open.push((u, e));
                        }
                    }
                }
            }
            let mut near: Vec<u32> = best.into_iter().map(|(v, _)| v).collect();
            near.sort_unstable();
            near
        })
        .collect();
    let mut start = Vec::with_capacity(count + 1);
    let mut near = Vec::new();
    start.push(0);
    for list in found {
        near.extend(list);
        start.push(near.len() as u32);
    }
    (start, near)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two parallel squares (two triangles each) `gap` apart along z; the
    /// second one's vertices are 4..8. The squares are offset sideways so
    /// that vertices face triangle interiors.
    fn two_sheets(gap: f32) -> (Vec<Vec3A>, Vec<[u32; 3]>) {
        let square = |z: f32, shift: f32| {
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
                .map(|(x, y)| Vec3A::new(x + shift, y + 0.5 * shift, z))
        };
        let positions = [square(0.0, 0.0), square(gap, 0.3)].concat();
        (positions, vec![[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    }

    fn params() -> ContactParams {
        ContactParams {
            thickness: 0.05,
            layer_reach: 0.3,
            move_layers: true,
        }
    }

    #[test]
    fn sheets_closer_than_the_thickness_get_planes_that_separate_them() {
        let (mut positions, triangles) = two_sheets(0.02);
        let start = positions.clone();
        let mut contacts = ClothContacts::new(triangles, vec![0; 8], &[], &positions);
        let stats = contacts.detect(&start, &mut positions, &params(), None);
        assert_eq!(stats.crossings, 0);
        assert!(stats.contacts >= 2);
        assert_eq!(positions, start, "resting contacts move nothing at once");
        // Vertex 4 hovers over the first sheet: its plane points up and is
        // violated by half the missing distance.
        let plane = contacts.planes()[4][0];
        assert!(plane.is_active() && plane.normal[2] > 0.99);
        assert!((plane.distance(positions[4]) + 0.015).abs() < 1e-5);
        // Vertex 2 lies under the second sheet: pushed the other way.
        assert!(contacts.planes()[2][0].normal[2] < -0.99);
    }

    #[test]
    fn a_vertex_that_passed_through_is_put_back_on_its_side() {
        let (start, triangles) = two_sheets(0.2);
        // The upper sheet drops through the lower one within one step.
        let mut positions = start.clone();
        for p in &mut positions[4..] {
            p.z = -0.2;
        }
        let mut contacts = ClothContacts::new(triangles, vec![0; 8], &[], &start);
        let stats = contacts.detect(&start, &mut positions, &params(), None);
        assert!(stats.crossings >= 1);
        assert!(
            positions[4].z > 0.0,
            "vertex 4 is back above the lower sheet: {}",
            positions[4].z
        );
        assert!(contacts.planes()[4][0].normal[2] > 0.99);
    }

    #[test]
    fn stitched_vertices_do_not_collide_with_each_other() {
        let (mut positions, triangles) = two_sheets(0.02);
        let start = positions.clone();
        // Sew every vertex of the upper sheet to the lower sheet.
        let stitches: Vec<(u32, u32)> = (0..4).map(|i| (i, i + 4)).collect();
        let mut contacts = ClothContacts::new(triangles, vec![0; 8], &stitches, &positions);
        assert_eq!(
            contacts.detect(&start, &mut positions, &params(), None),
            ContactStats::default()
        );
    }

    #[test]
    fn the_higher_layer_is_moved_outside_the_lower_one() {
        // The "jacket" (layer 1, vertices 4..8) starts and stays *under*
        // the "shirt": no crossing happens, only the layer order tells.
        let (mut positions, triangles) = two_sheets(-0.03);
        let start = positions.clone();
        let layers = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let outward = |_: Vec3A| Some(Vec3A::Z);
        let mut contacts = ClothContacts::new(triangles.clone(), layers.clone(), &[], &positions);
        let stats = contacts.detect(&start, &mut positions, &params(), Some(&outward));
        assert!(stats.crossings >= 1);
        assert!(
            positions[4].z > 0.0,
            "jacket vertex is outside the shirt: {}",
            positions[4].z
        );
        // The shirt vertex below the jacket is pushed inwards.
        assert!(contacts.planes()[2][0].normal[2] < -0.99);

        // Without layers the same configuration is a resting contact and
        // nothing changes sides.
        let mut same = start.clone();
        let mut contacts = ClothContacts::new(triangles, vec![0; 8], &[], &start);
        contacts.detect(&start, &mut same, &params(), Some(&outward));
        assert!(same[4].z < 0.0);
    }
}

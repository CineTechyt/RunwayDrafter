//! Vertex Block Descent with Offset Geometric Contact.
//!
//! A second solver for the same [`Model`], built for contact:
//!
//! * **Vertex Block Descent** (Chen, Liu, Yang & Yuksel 2024). Implicit
//!   Euler, minimised one vertex at a time: each vertex takes a Newton step
//!   on its own 3×3 system (inertia, every constraint row that touches it,
//!   every contact it is in), Gauss–Seidel over vertex colours. The rows of
//!   the model are used as energies `½·k·C²` with `k = 1/compliance`.
//! * **Offset Geometric Contact** (Chen et al. 2025). Cloth has a
//!   thickness: every point, edge and face is surrounded by a shell of
//!   radius `r`, and contact is an energy that rises as two shells
//!   overlap (quadratic, turning into a log barrier at half overlap).
//!   Contact pairs are vertex–face (with the face's closest feature, so a
//!   vertex over an edge counts once, not once per face) and edge–edge.
//!   After every detection each vertex gets a *conservative bound*: less
//!   than half the distance from anything around it to anything near. No
//!   vertex moves further than its bound before the next detection, so no
//!   two parts of the cloth can pass through each other however the
//!   solver iterates, without any continuous collision test.
//!
//! The body is contact too, not a constraint on velocities: an energy
//! that holds cloth `margin` off the skin (through the distance field,
//! which knows inside from out) and an energy from skin points against
//! cloth faces, for features smaller than a triangle. Both act only when
//! touching, so cloth can lift off the body whenever inertia or gravity
//! take it away, and friction acts on the cloth's motion relative to the
//! moving skin, which is what makes it follow the body.

use glam::{Mat3A, Vec3A};
use physics_collision::ccd::closest_point_on_triangle;
use physics_collision::{Aabb, Bvh, TriangleMesh};
use rayon::prelude::*;
use topology_sync::DistanceField;

use crate::cpu::{Ellipsoid, State};
use crate::model::{INVALID, Model, ROW_ACTIVE, ROW_SEAM};

/// Fraction of the distance to the nearest other geometry a vertex may
/// move between detections (OGC's γ; below ½ keeps two approaching
/// primitives apart).
const BOUND_FRACTION: f32 = 0.45;
/// Smallest bound, in metres: cloth that already intersects itself (from
/// another solver, or a pattern placed through itself) is not frozen.
const MIN_BOUND: f32 = 2e-4;
/// Largest query radius, in metres: beyond it, cloth moving faster than
/// this per sub-step simply takes several detections.
const MAX_QUERY: f32 = 0.05;
/// Bounds are sized for this many sub-steps of the current motion.
const LOOKAHEAD: f32 = 3.0;
/// Detections between full rebuilds of the search trees; in between they
/// are refit.
const REBUILD_INTERVAL: u32 = 32;
/// Iterations added to a sub-step while vertices are held at their bounds.
const EXTRA_ITERATIONS: usize = 4;
/// Re-detections allowed per sub-step when vertices run into their bounds.
const MAX_DETECTIONS: usize = 6;
/// Stiffness of a closed stitch, N/m.
const SEAM_STIFFNESS: f32 = 1.0e4;
/// Contact stiffness between cloth surfaces, N/m.
const SELF_STIFFNESS: f32 = 2_000.0;
/// Contact stiffness between cloth and the body, N/m.
const BODY_STIFFNESS: f32 = 4_000.0;
/// Largest normal force (N) friction acts with at one vertex: some
/// twenty times a vertex's weight, enough for cloth carried by the body.
const MAX_FRICTION_LOAD: f32 = 0.01;
/// How far beyond the margin cloth still counts as resting on the body.
const GRIP_BAND: f32 = 0.0005;
/// Below this sliding speed (m/s) friction holds: static friction.
const STICK_SPEED: f32 = 0.01;
/// How far in front of a skin point other skin makes it a crease.
const CREASE: f32 = 0.015;
/// Fraction of the body margin within which skin points push cloth faces.
const SKIN_POINT_SHARE: f32 = 0.5;
/// Below this many vertices a loop runs on the calling thread.
const PARALLEL_THRESHOLD: usize = 48;

/// What one sub-step needs besides the state.
pub struct StepInput<'a> {
    pub h: f32,
    pub gravity: Vec3A,
    /// Velocity multiplier per sub-step.
    pub damping: f32,
    pub seam_gap: f32,
    pub friction: f32,
    /// Distance kept from the body.
    pub margin: f32,
    /// Shell radius of the cloth (half of the distance kept between two
    /// cloth surfaces would be the thickness; this is the full distance).
    pub thickness: f32,
    pub ground: f32,
    pub max_speed: f32,
    pub iterations: usize,
    /// Chebyshev acceleration of the iterations (the spectral radius
    /// estimate ρ, below 1); 0 turns it off.
    pub acceleration: f32,
    pub body: Option<&'a DistanceField>,
    pub body_mesh: Option<&'a TriangleMesh>,
    /// Cloth collides with cloth.
    pub self_collision: bool,
    /// Time from the end of this sub-step to the end of the step, whose
    /// body `body` and `body_mesh` show.
    pub body_lag: f32,
    pub colliders: &'a [Ellipsoid],
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OgcStats {
    /// Microseconds spent finding contacts and solving.
    pub detect_us: u64,
    pub solve_us: u64,
    /// Cloth contact pairs in touch.
    pub self_contacts: usize,
    /// Skin points touching cloth faces.
    pub body_contacts: usize,
    /// Vertex moves cut short by their bound.
    pub truncated: usize,
    pub detections: usize,
}

/// A contact pair: `Σ cₖ·xₖ` over up to four vertices is the vector from
/// one side to the other; its length is the distance.
#[derive(Debug, Clone, Copy)]
enum Pair {
    /// Vertex against a face (closest point anywhere on it).
    VertexFace { v: u32, face: u32 },
    /// Two edges, closest points inside both.
    EdgeEdge { a: u32, b: u32 },
}

/// Friction of one vertex against the body for a sub-step.
#[derive(Debug, Clone, Copy)]
struct Grip {
    normal: Vec3A,
    load: f32,
    velocity: Vec3A,
}

/// A skin point against a cloth face.
#[derive(Debug, Clone, Copy)]
struct BodyPair {
    /// Where the skin point is at the end of the step.
    point: Vec3A,
    normal: Vec3A,
    /// Its velocity, m/s.
    velocity: Vec3A,
    face: u32,
}

/// Compressed lists: `items[start[i]..start[i + 1]]` belong to `i`.
#[derive(Debug, Clone, Default)]
struct Csr<T> {
    start: Vec<u32>,
    items: Vec<T>,
}

impl<T: Copy> Csr<T> {
    fn build(count: usize, entries: impl Iterator<Item = (u32, T)> + Clone) -> Self {
        let mut start = vec![0u32; count + 1];
        for (i, _) in entries.clone() {
            start[i as usize + 1] += 1;
        }
        for i in 0..count {
            start[i + 1] += start[i];
        }
        let mut cursor = start.clone();
        let mut items: Vec<Option<T>> = vec![None; start[count] as usize];
        for (i, item) in entries {
            items[cursor[i as usize] as usize] = Some(item);
            cursor[i as usize] += 1;
        }
        Self {
            start,
            items: items.into_iter().map(Option::unwrap).collect(),
        }
    }

    #[inline]
    fn of(&self, i: usize) -> &[T] {
        if i + 1 >= self.start.len() {
            return &[];
        }
        &self.items[self.start[i] as usize..self.start[i + 1] as usize]
    }
}

/// Topology and the contacts of the last detection.
#[derive(Debug, Clone, Default)]
pub struct Ogc {
    triangles: Vec<[u32; 3]>,
    edges: Vec<[u32; 2]>,
    /// Per vertex, the sorted vertices within two hops (mesh edges and
    /// stitches): never in contact with it.
    near: Csr<u32>,
    mean_edge: f32,
    /// Vertex → (element, slot in the element).
    rows: Csr<(u32, u8)>,
    /// Vertices no two of which share an element.
    colours: Vec<Vec<u32>>,
    pairs: Vec<Pair>,
    /// Vertex → (pair, which coefficient is its own).
    incident: Csr<(u32, u8)>,
    body_pairs: Vec<BodyPair>,
    body_incident: Csr<(u32, u8)>,
    /// Positions at the last detection, and how far from them each vertex
    /// may go.
    anchor: Vec<Vec3A>,
    bound: Vec<f32>,
    /// The contacts must be found again before the next sub-step.
    stale: bool,
    /// The body moved: its pairs must be found again.
    body_stale: bool,
    face_tree: Bvh,
    edge_tree: Bvh,
    /// Detections since the trees were last built rather than refit.
    since_build: u32,
    pub stats: OgcStats,
}

impl Ogc {
    pub fn new(
        model: &Model,
        triangles: Vec<[u32; 3]>,
        stitches: &[(u32, u32)],
        rest: &[Vec3A],
    ) -> Self {
        let n = model.inverse_mass.len();
        let mut edges: Vec<[u32; 2]> = triangles
            .iter()
            .flat_map(|t| [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]])
            .map(|[a, b]| [a.min(b), a.max(b)])
            .collect();
        edges.sort_unstable();
        edges.dedup();
        let mean_edge = if edges.is_empty() {
            0.0
        } else {
            edges
                .iter()
                .map(|&[a, b]| rest[a as usize].distance(rest[b as usize]))
                .sum::<f32>()
                / edges.len() as f32
        };

        // Each vertex's neighbourhood: two hops over mesh edges and
        // stitches. Geometry this close can only meet by folding sharper
        // than bending allows, and moves with the vertex; counting it
        // would hold every vertex to a fraction of an edge per detection.
        let mut links: Vec<Vec<u32>> = vec![Vec::new(); n];
        for &[a, b] in edges.iter().chain(
            stitches
                .iter()
                .map(|(a, b)| [*a, *b])
                .collect::<Vec<_>>()
                .iter(),
        ) {
            if a != b {
                links[a as usize].push(b);
                links[b as usize].push(a);
            }
        }
        let near: Vec<Vec<u32>> = (0..n)
            .map(|v| {
                let mut set: Vec<u32> = vec![v as u32];
                for &u in &links[v] {
                    set.push(u);
                    set.extend_from_slice(&links[u as usize]);
                }
                set.sort_unstable();
                set.dedup();
                set
            })
            .collect();
        let near = Csr::build(
            n,
            near.iter()
                .enumerate()
                .flat_map(|(v, set)| set.iter().map(move |&u| (v as u32, u))),
        );

        let rows = Csr::build(
            n,
            model.elements.iter().enumerate().flat_map(|(e, element)| {
                (0..4)
                    .filter(move |&k| element.particles[k] != INVALID)
                    .map(move |k| (element.particles[k], (e as u32, k as u8)))
            }),
        );

        // Greedy vertex colouring over shared elements.
        let mut colour_of = vec![u32::MAX; n];
        let mut colours: Vec<Vec<u32>> = Vec::new();
        let mut taken: Vec<u32> = Vec::new();
        for v in 0..n {
            taken.clear();
            for &(e, _) in rows.of(v) {
                for &u in &model.elements[e as usize].particles {
                    if u != INVALID && u as usize != v && colour_of[u as usize] != u32::MAX {
                        taken.push(colour_of[u as usize]);
                    }
                }
            }
            // The least used colour still free: colours of even size keep
            // every sweep parallel (a small colour is mostly overhead).
            let colour = (0..colours.len() as u32)
                .filter(|c| !taken.contains(c))
                .min_by_key(|&c| colours[c as usize].len())
                .unwrap_or(colours.len() as u32);
            colour_of[v] = colour;
            if colours.len() <= colour as usize {
                colours.resize(colour as usize + 1, Vec::new());
            }
            colours[colour as usize].push(v as u32);
        }

        Self {
            triangles,
            edges,
            near,
            mean_edge,
            rows,
            colours,
            ..Self::default()
        }
    }

    /// Finds contacts afresh at the next sub-step (the body has moved, or
    /// the cloth was moved from outside).
    pub fn invalidate(&mut self) {
        self.stale = true;
        self.body_stale = true;
    }

    pub fn colour_count(&self) -> usize {
        self.colours.len()
    }

    /// Contact radius and query radius for a requested cloth thickness.
    fn radii(&self, thickness: f32) -> (f32, f32) {
        let r = thickness.min(0.3 * self.mean_edge).max(1e-4);
        (r, (0.5 * self.mean_edge).max(3.0 * r))
    }

    /// Finds the contact pairs around `x` and every vertex's bound.
    /// `reach` is how far each vertex wants to go: its query radius grows
    /// to that, so fast cloth is not held back by bounds that only reflect
    /// how little was looked at. Two primitives are tested within the
    /// larger of their radii; anything not found is further away than
    /// both can move together.
    fn detect(&mut self, x: &[Vec3A], input: &StepInput<'_>, reach: &[f32]) {
        let clock = std::time::Instant::now();
        self.detect_inner(x, input, reach);
        self.stats.detect_us += clock.elapsed().as_micros() as u64;
    }

    fn detect_inner(&mut self, x: &[Vec3A], input: &StepInput<'_>, reach: &[f32]) {
        let (_, base) = self.radii(input.thickness);
        let radius: Vec<f32> = reach
            .iter()
            .map(|r| base.max(r / BOUND_FRACTION * 1.1).min(MAX_QUERY))
            .collect();
        let n = x.len();
        let face_radius: Vec<f32> = self
            .triangles
            .iter()
            .map(|t| t.iter().map(|&i| radius[i as usize]).fold(0.0, f32::max))
            .collect();
        let face_boxes: Vec<Aabb> = self
            .triangles
            .iter()
            .zip(&face_radius)
            .map(|(t, &r)| Aabb::from_points(t.map(|i| x[i as usize])).expanded(r))
            .collect();
        let refit = self.face_tree.primitive_count() == face_boxes.len()
            && self.since_build < REBUILD_INTERVAL;
        if refit {
            self.face_tree.refit(&face_boxes);
            self.since_build += 1;
        } else {
            self.face_tree = Bvh::build(&face_boxes);
            self.since_build = 0;
        }
        let faces = &self.face_tree;
        let edge_radius: Vec<f32> = self
            .edges
            .iter()
            .map(|e| radius[e[0] as usize].max(radius[e[1] as usize]))
            .collect();
        let edge_boxes: Vec<Aabb> = self
            .edges
            .iter()
            .zip(&edge_radius)
            .map(|(e, &r)| Aabb::from_points(e.map(|i| x[i as usize])).expanded(r))
            .collect();
        if refit && self.edge_tree.primitive_count() == edge_boxes.len() {
            self.edge_tree.refit(&edge_boxes);
        } else {
            self.edge_tree = Bvh::build(&edge_boxes);
        }
        let edge_tree = &self.edge_tree;
        let near = &self.near;
        let close = |v: u32, u: u32| near.of(v as usize).binary_search(&u).is_ok();
        let (triangles, edges) = (&self.triangles, &self.edges);

        // Vertex against faces: one pair per closest feature.
        let cloth = input.self_collision;
        let vertex_pairs: Vec<Vec<(u32, f32)>> = (0..if cloth { n } else { 0 })
            .into_par_iter()
            .with_min_len(256)
            .map(|v| {
                let p = x[v];
                let mut found: Vec<(u32, f32, [u32; 2])> = Vec::new();
                let around = Aabb::from_points([p]).expanded(radius[v]);
                faces.query(&around, |face| {
                    if !face_boxes[face as usize].overlaps(&around) {
                        return;
                    }
                    let query = radius[v].max(face_radius[face as usize]);
                    let t = triangles[face as usize];
                    if t.iter().any(|&u| close(v as u32, u)) {
                        return;
                    }
                    let [a, b, c] = t.map(|i| x[i as usize]);
                    let (closest, w) = closest_point_on_triangle(p, a, b, c);
                    let distance = closest.distance(p);
                    if distance >= query {
                        return;
                    }
                    // The feature the closest point lies on: the face, one
                    // of its edges or one of its corners.
                    let zero: Vec<usize> = (0..3).filter(|&k| w[k] <= 1e-6).collect();
                    let feature = match zero.len() {
                        0 => [u32::MAX, face],
                        1 => {
                            let (i, j) = (t[(zero[0] + 1) % 3], t[(zero[0] + 2) % 3]);
                            [i.min(j), i.max(j)]
                        }
                        _ => {
                            let corner = (0..3).find(|k| !zero.contains(k)).unwrap_or(0);
                            [t[corner], t[corner]]
                        }
                    };
                    if !found.iter().any(|f| f.2 == feature) {
                        found.push((face, distance, feature));
                    }
                });
                found
                    .into_iter()
                    .map(|(face, distance, _)| (face, distance))
                    .collect()
            })
            .collect();

        // Edges against edges, closest points inside both; each pair once.
        let edge_pairs: Vec<Vec<(u32, f32)>> = (0..if cloth { edges.len() } else { 0 })
            .into_par_iter()
            .with_min_len(256)
            .map(|e| {
                let [a, b] = edges[e];
                let mut found = Vec::new();
                edge_tree.query(&edge_boxes[e], |other| {
                    if other as usize <= e || !edge_boxes[other as usize].overlaps(&edge_boxes[e]) {
                        return;
                    }
                    let query = edge_radius[e].max(edge_radius[other as usize]);
                    let [c, d] = edges[other as usize];
                    if [c, d].iter().any(|&u| close(a, u) || close(b, u)) {
                        return;
                    }
                    if let Some((s, t, distance)) =
                        segment_segment(x[a as usize], x[b as usize], x[c as usize], x[d as usize])
                        && s > 1e-4
                        && s < 1.0 - 1e-4
                        && t > 1e-4
                        && t < 1.0 - 1e-4
                        && distance < query
                    {
                        found.push((other, distance));
                    }
                });
                found
            })
            .collect();

        // Bounds: every primitive in a pair limits all of its vertices.
        let mut nearest = radius.clone();
        for (v, found) in vertex_pairs.iter().enumerate() {
            for &(face, distance) in found {
                nearest[v] = nearest[v].min(distance);
                for &u in &triangles[face as usize] {
                    nearest[u as usize] = nearest[u as usize].min(distance);
                }
            }
        }
        for (e, found) in edge_pairs.iter().enumerate() {
            for &(other, distance) in found {
                for &u in edges[e].iter().chain(&edges[other as usize]) {
                    nearest[u as usize] = nearest[u as usize].min(distance);
                }
            }
        }
        self.bound = nearest
            .iter()
            .map(|d| (BOUND_FRACTION * d).max(MIN_BOUND))
            .collect();
        // Only pairs that can come within the contact radius before the
        // next detection, their vertices staying within their bounds, are
        // worth evaluating while solving.
        let (r, _) = self.radii(input.thickness);
        let bound = &self.bound;
        let reach = |ids: &[u32]| {
            ids.iter()
                .map(|&u| bound[u as usize])
                .fold(0.0f32, f32::max)
        };
        self.pairs.clear();
        for (v, found) in vertex_pairs.iter().enumerate() {
            for &(face, distance) in found {
                if distance < r + bound[v] + reach(&triangles[face as usize]) {
                    self.pairs.push(Pair::VertexFace { v: v as u32, face });
                }
            }
        }
        for (e, found) in edge_pairs.iter().enumerate() {
            for &(other, distance) in found {
                if distance < r + reach(&edges[e]) + reach(&edges[other as usize]) {
                    self.pairs.push(Pair::EdgeEdge {
                        a: e as u32,
                        b: other,
                    });
                }
            }
        }
        self.anchor.clear();
        self.anchor.extend_from_slice(x);

        let pairs = &self.pairs;
        let vertices = |pair: &Pair| -> [u32; 4] {
            match *pair {
                Pair::VertexFace { v, face } => {
                    let t = triangles[face as usize];
                    [v, t[0], t[1], t[2]]
                }
                Pair::EdgeEdge { a, b } => {
                    let ([p, q], [r, s]) = (edges[a as usize], edges[b as usize]);
                    [p, q, r, s]
                }
            }
        };
        self.incident = Csr::build(
            n,
            pairs.iter().enumerate().flat_map(|(i, pair)| {
                vertices(pair)
                    .into_iter()
                    .enumerate()
                    .map(move |(k, v)| (v, (i as u32, k as u8)))
            }),
        );

        self.stats.detections += 1;
    }

    /// Skin points against cloth faces, at the start of a step: the body
    /// stands still within it.
    fn detect_body(&mut self, x: &[Vec3A], input: &StepInput<'_>) {
        let clock = std::time::Instant::now();
        let n = x.len();
        let triangles = &self.triangles;
        let face_boxes: Vec<Aabb> = triangles
            .iter()
            .map(|t| Aabb::from_points(t.map(|i| x[i as usize])))
            .collect();
        let faces = Bvh::build(&face_boxes);
        // Skin points against cloth faces.
        self.body_pairs.clear();
        if let Some(body) = input.body_mesh {
            let (margin, field) = (input.margin, input.body);
            let bounds = faces.bounds().expanded(margin);
            let skin = body.positions();
            let normals = body.vertex_normals();
            let found: Vec<BodyPair> = skin
                .par_iter()
                .zip(normals)
                .with_min_len(512)
                .filter(|(p, n)| n.length_squared() > 0.5 && bounds.distance_squared(**p) == 0.0)
                .filter_map(|(&p, &normal)| {
                    let mut best: Option<(u32, f32)> = None;
                    faces.query(&Aabb::from_points([p]).expanded(margin), |face| {
                        let [a, b, c] = triangles[face as usize].map(|i| x[i as usize]);
                        let (q, w) = closest_point_on_triangle(p, a, b, c);
                        // Under the face, not beside it.
                        if w.iter().any(|&w| w <= 0.0) {
                            return;
                        }
                        let height = (q - p).dot(normal);
                        // Only where the body agrees which way is out: in
                        // the gap between an arm and the torso, cloth on
                        // the torso is in front of the arm's skin too.
                        let agrees = field.is_none_or(|f| {
                            f.sample(q.into()).is_some_and(|(_, g)| {
                                Vec3A::from(g).normalize_or_zero().dot(normal) > 0.7
                            })
                        });
                        if height < margin
                            && height > -margin
                            && agrees
                            && best.is_none_or(|(_, h)| height < h)
                        {
                            best = Some((face, height));
                        }
                    });
                    let (face, _) = best?;
                    // A crease: cloth bridges it rather than following in.
                    // (Tested last: most skin points have no cloth face over
                    // them at all.)
                    if let Some(field) = field
                        && field
                            .sample((p + normal * CREASE).into())
                            .is_none_or(|(d, _)| d < CREASE * 0.6)
                    {
                        return None;
                    }
                    let velocity =
                        field.map_or(Vec3A::ZERO, |f| Vec3A::from(f.velocity_at(p.into())));
                    Some(BodyPair {
                        point: p,
                        normal,
                        velocity,
                        face,
                    })
                })
                .collect();
            self.body_pairs = found;
        }
        let body_pairs = &self.body_pairs;
        self.body_incident = Csr::build(
            n,
            body_pairs.iter().enumerate().flat_map(|(i, pair)| {
                triangles[pair.face as usize]
                    .into_iter()
                    .enumerate()
                    .map(move |(k, v)| (v, (i as u32, k as u8)))
            }),
        );
        self.body_stale = false;
        self.stats.detect_us += clock.elapsed().as_micros() as u64;
    }

    /// Distance vector of a pair at `x`: `(Σ cₖ·xₖ, c)`.
    #[inline]
    fn pair_vector(&self, pair: &Pair, x: &[Vec3A]) -> (Vec3A, [f32; 4]) {
        match *pair {
            Pair::VertexFace { v, face } => {
                let t = self.triangles[face as usize];
                let [a, b, c] = t.map(|i| x[i as usize]);
                let p = x[v as usize];
                let (q, w) = closest_point_on_triangle(p, a, b, c);
                (p - q, [1.0, -w[0], -w[1], -w[2]])
            }
            Pair::EdgeEdge { a, b } => {
                let ([p, q], [r, s]) = (self.edges[a as usize], self.edges[b as usize]);
                let (xp, xq, xr, xs) = (x[p as usize], x[q as usize], x[r as usize], x[s as usize]);
                let (s_, t_) = segment_segment(xp, xq, xr, xs).map_or((0.5, 0.5), |(s, t, _)| {
                    (s.clamp(0.0, 1.0), t.clamp(0.0, 1.0))
                });
                let c = [1.0 - s_, s_, -(1.0 - t_), -t_];
                (xp * c[0] + xq * c[1] + xr * c[2] + xs * c[3], c)
            }
        }
    }

    fn pair_vertices(&self, pair: &Pair) -> [u32; 4] {
        match *pair {
            Pair::VertexFace { v, face } => {
                let t = self.triangles[face as usize];
                [v, t[0], t[1], t[2]]
            }
            Pair::EdgeEdge { a, b } => {
                let ([p, q], [r, s]) = (self.edges[a as usize], self.edges[b as usize]);
                [p, q, r, s]
            }
        }
    }

    /// One sub-step of implicit Euler.
    pub fn substep(&mut self, model: &Model, state: &mut State, input: &StepInput<'_>) {
        let n = state.positions.len();
        if n == 0 {
            return;
        }
        let h = input.h;
        self.stats = OgcStats::default();
        let start: Vec<Vec3A> = state.positions.clone();
        // Inertia: where each vertex would go without any force but gravity.
        let inertia: Vec<Vec3A> = start
            .iter()
            .zip(&mut state.velocities)
            .zip(&model.inverse_mass)
            .map(|((&x, v), &w)| {
                if w <= 0.0 {
                    return x;
                }
                *v = (*v * input.damping).clamp_length_max(input.max_speed);
                x + *v * h + input.gravity * (h * h)
            })
            .collect();

        let reach = |target: &[Vec3A], from: &[Vec3A]| -> Vec<f32> {
            target
                .iter()
                .zip(from)
                .map(|(a, b)| a.distance(*b) * LOOKAHEAD)
                .collect()
        };
        let mut detections = 0;
        if self.body_stale || self.anchor.len() != n {
            self.detect_body(&state.positions, input);
        }
        // Contacts and bounds carry over between sub-steps until a vertex
        // reaches its bound or the body moves (a new step).
        // Also when the motion this sub-step would take any vertex past its
        // bound: clipping the inertial guess vertex by vertex would bend
        // the cloth and cost it its momentum.
        let outgrown = || {
            self.anchor.len() != n
                || inertia
                    .iter()
                    .zip(&self.anchor)
                    .zip(&self.bound)
                    .any(|((y, a), b)| y.distance_squared(*a) > b * b)
        };
        if self.stale || outgrown() {
            self.detect(&state.positions, input, &reach(&inertia, &start));
            detections += 1;
        }
        // Start from the inertial guess, as far as the bounds allow.
        let mut truncated = 0;
        for (i, (&target, &w)) in inertia.iter().zip(&model.inverse_mass).enumerate() {
            if w > 0.0 {
                let (moved, cut) = self.truncate(i, target);
                state.positions[i] = moved;
                truncated += usize::from(cut);
            }
        }

        // Friction against the body is set up once per sub-step, from where
        // the cloth starts it: normal, normal force and skin velocity. Taken
        // from the iterate instead, a vertex at the edge of contact switches
        // friction on and off between iterations and buzzes on the skin.
        let grip: Vec<Option<Grip>> = match input.body {
            Some(field) if input.friction > 0.0 => start
                .par_iter()
                .zip(&model.inverse_mass)
                .with_min_len(512)
                .map(|(&x, &w)| {
                    let velocity = Vec3A::from(field.velocity_at(x.into()));
                    let (distance, gradient) =
                        field.sample((x + velocity * (input.body_lag + input.h)).into())?;
                    // Cloth resting on the skin sits right at the margin,
                    // a hair outside it as often as inside.
                    if distance >= input.margin + GRIP_BAND || w <= 0.0 {
                        return None;
                    }
                    let normal = Vec3A::from(gradient).try_normalize()?;
                    // At least its own weight pressing on the surface: the
                    // load of cloth that lies on it.
                    let weight = (-input.gravity.dot(normal)).max(0.0) / w;
                    let load = (-body_energy(distance, input.margin).0)
                        .max(weight)
                        .min(MAX_FRICTION_LOAD);
                    Some(Grip {
                        normal,
                        load,
                        velocity,
                    })
                })
                .collect(),
            _ => Vec::new(),
        };
        let mut out: Vec<Vec3A> = Vec::new();
        let clock = std::time::Instant::now();
        let mut detect_before = self.stats.detect_us;
        let planned = input.iterations.max(1);
        let mut omega = 1.0f32;
        let mut older: Vec<Vec3A> = state.positions.clone();
        let mut iteration = 0;
        // Beyond the planned iterations, more while vertices are held at
        // their bounds (the body pushing cloth onto cloth, say): each with
        // a fresh detection, so they can follow.
        while iteration < planned || (truncated > 0 && iteration < planned + EXTRA_ITERATIONS) {
            iteration += 1;
            // Vertices stopped by their bounds have somewhere to go: look
            // again from where they are now.
            if truncated > 0 && detections < MAX_DETECTIONS && iteration > 1 {
                let x = state.positions.clone();
                self.detect(&x, input, &reach(&inertia, &x));
                detections += 1;
            }
            let before = state.positions.clone();
            truncated = self.sweep(
                model,
                &mut state.positions,
                &start,
                &inertia,
                &grip,
                input,
                &mut out,
            );
            // Chebyshev semi-iterative acceleration (Wang 2015, as in VBD):
            // extrapolate from the iterate before last.
            let rho = input.acceleration;
            if rho > 0.0 {
                omega = match iteration {
                    1 => 1.0,
                    2 => 2.0 / (2.0 - rho * rho),
                    _ => 4.0 / (4.0 - rho * rho * omega),
                };
                if iteration > 1 && omega > 1.0 {
                    for (i, ((x, &old), &w)) in state
                        .positions
                        .iter_mut()
                        .zip(&older)
                        .zip(&model.inverse_mass)
                        .enumerate()
                    {
                        if w > 0.0 {
                            *x = self.truncate(i, old + (*x - old) * omega).0;
                        }
                    }
                }
                older = before;
            }
        }
        self.stale = truncated > 0;
        detect_before = self.stats.detect_us - detect_before;
        self.stats.solve_us += (clock.elapsed().as_micros() as u64).saturating_sub(detect_before);
        self.stats.truncated = truncated;
        self.stats.detections = detections;

        // Rigid colliders (ellipsoids) and the floor as hard limits.
        for (i, x) in state.positions.iter_mut().enumerate() {
            if model.inverse_mass[i] <= 0.0 {
                continue;
            }
            for collider in input.colliders {
                if let Some(corrected) =
                    crate::cpu::push_out_of_ellipsoid(*x, collider, input.margin)
                {
                    *x = corrected;
                }
            }
        }
        for ((v, &x), &x0) in state
            .velocities
            .iter_mut()
            .zip(&state.positions)
            .zip(&start)
        {
            *v = ((x - x0) / h).clamp_length_max(input.max_speed);
        }
        state.previous.clone_from(&start);

        let (r, _) = self.radii(input.thickness);
        self.stats.self_contacts = self
            .pairs
            .iter()
            .filter(|p| self.pair_vector(p, &state.positions).0.length() < r)
            .count();
        self.stats.body_contacts = self
            .body_pairs
            .iter()
            .filter(|p| {
                let t = self.triangles[p.face as usize].map(|i| state.positions[i as usize]);
                let (q, _) = closest_point_on_triangle(p.point, t[0], t[1], t[2]);
                (q - p.point).dot(p.normal) < input.margin
            })
            .count();
    }

    /// One Gauss–Seidel sweep over all colours, each colour in parallel.
    /// Returns how many vertices were held at their bounds.
    #[allow(clippy::too_many_arguments)]
    fn sweep(
        &self,
        model: &Model,
        positions: &mut [Vec3A],
        start: &[Vec3A],
        inertia: &[Vec3A],
        grip: &[Option<Grip>],
        input: &StepInput<'_>,
        out: &mut Vec<Vec3A>,
    ) -> usize {
        let threads = rayon::current_num_threads();
        let largest = self.colours.iter().map(Vec::len).max().unwrap_or(0);
        if threads == 1 || positions.len() < PARALLEL_THRESHOLD * 4 {
            let mut truncated = 0;
            for vertices in &self.colours {
                out.clear();
                out.extend(vertices.iter().map(|&v| {
                    self.solve_vertex(v as usize, model, positions, start, inertia, grip, input)
                }));
                for (&v, &target) in vertices.iter().zip(out.iter()) {
                    let (moved, cut) = self.truncate(v as usize, target);
                    positions[v as usize] = moved;
                    truncated += usize::from(cut);
                }
            }
            return truncated;
        }
        let _ = largest;
        let mut truncated = 0;
        for vertices in &self.colours {
            out.clear();
            out.resize(vertices.len(), Vec3A::ZERO);
            let x = &*positions;
            out.par_iter_mut()
                .zip(vertices)
                .with_min_len(16)
                .for_each(|(slot, &v)| {
                    *slot = self.solve_vertex(v as usize, model, x, start, inertia, grip, input);
                });
            for (&v, &target) in vertices.iter().zip(out.iter()) {
                let (moved, cut) = self.truncate(v as usize, target);
                positions[v as usize] = moved;
                truncated += usize::from(cut);
            }
        }
        truncated
    }

    /// `target` limited to the vertex's bound around its anchor; whether
    /// it had to be cut short.
    #[inline]
    fn truncate(&self, i: usize, target: Vec3A) -> (Vec3A, bool) {
        let anchor = self.anchor[i];
        let offset = target - anchor;
        let bound = self.bound[i];
        if offset.length_squared() <= bound * bound {
            (target, false)
        } else {
            (anchor + offset * (bound / offset.length()), true)
        }
    }

    /// The Newton step of one vertex on its local energy.
    #[allow(clippy::too_many_arguments)]
    fn solve_vertex(
        &self,
        i: usize,
        model: &Model,
        x: &[Vec3A],
        start: &[Vec3A],
        inertia: &[Vec3A],
        grip: &[Option<Grip>],
        input: &StepInput<'_>,
    ) -> Vec3A {
        let xi = x[i];
        let w = model.inverse_mass[i];
        if w <= 0.0 {
            return xi;
        }
        let h = input.h;
        let mass = 1.0 / w;
        let inertia_stiffness = mass / (h * h);
        // Force (negative gradient) and Hessian of the local energy.
        let mut force = (inertia[i] - xi) * inertia_stiffness;
        let mut hessian = Mat3A::from_diagonal(glam::Vec3::splat(inertia_stiffness));

        // --- the model's rows: ½·k·(|Σ wₖ·xₖ| − rest)² -----------------------
        for &(e, slot) in self.rows.of(i) {
            let element = &model.elements[e as usize];
            // The element's positions, once for all its rows.
            let corners = element.particles.map(|p| {
                if p != INVALID {
                    x[p as usize]
                } else {
                    Vec3A::ZERO
                }
            });
            for row in &element.rows {
                if row.flags & ROW_ACTIVE == 0 {
                    continue;
                }
                let wi = row.weights[slot as usize];
                if wi == 0.0 {
                    continue;
                }
                let w = row.weights;
                let mut sum =
                    corners[0] * w[0] + corners[1] * w[1] + corners[2] * w[2] + corners[3] * w[3];
                // A done-up button (a seam row that closes to a gap): its hole
                // belongs outside it, a fabric thickness further from the
                // body. The row is measured from that point rather than
                // from the button, so the lap cannot end up underneath.
                let fastening = row.flags & ROW_SEAM != 0 && row.fold_limit > 0.0;
                let mut lap = false;
                if fastening
                    && let Some(field) = input.body
                    && let Some((_, gradient)) =
                        field.sample(x[element.particles[0] as usize].into())
                    && let Some(out) = Vec3A::from(gradient).try_normalize()
                {
                    // `sum` is button − hole; aim the hole at button + gap·out.
                    sum += out * row.fold_limit;
                    lap = true;
                }
                let length = sum.length();
                let mut compliance = row.compliance;
                if row.fold_limit() > 0.0 && length > row.fold_limit() {
                    let ratio = row.fold_limit() / length;
                    compliance *= ratio * ratio * ratio * ratio;
                }
                let mut k = 1.0 / compliance.max(1e-12);
                if row.flags & ROW_SEAM != 0 {
                    // A stitch as stiff as the explicit solver makes it
                    // (≈ rigid) leaves each vertex's Newton step to the
                    // stitch alone: seam vertices race ahead of the fabric
                    // and tear the seam line apart. This still holds a
                    // seam under a few newtons to a fraction of a mm.
                    k = k.min(SEAM_STIFFNESS);
                }
                let rest = if lap {
                    (row.rest - row.fold_limit).max(0.0) * input.seam_gap
                } else if row.flags & ROW_SEAM != 0 {
                    row.seam_rest(input.seam_gap)
                } else {
                    row.rest
                };
                if rest == 0.0 {
                    // ½·k·|s|²: exact, also through zero.
                    force -= sum * (k * wi);
                    hessian += Mat3A::from_diagonal(glam::Vec3::splat(k * wi * wi));
                    continue;
                }
                if length < 1e-9 {
                    continue;
                }
                let u = sum / length;
                let c = length - rest;
                force -= u * (k * c * wi);
                // k·wᵢ²·(u·uᵀ + g·(I − u·uᵀ)), the geometric term g only where
                // it keeps the system positive definite (stretched rows).
                let g = c.max(0.0) / length;
                let scale = k * wi * wi;
                hessian += outer(u) * (scale * (1.0 - g));
                if g > 0.0 {
                    hessian += Mat3A::from_diagonal(glam::Vec3::splat(scale * g));
                }
            }
        }

        let friction = input.friction.max(0.0);
        let stick = STICK_SPEED * h;
        // Friction against relative tangential motion `u` this sub-step,
        // with normal force `normal_force`, for a vertex whose share of the
        // contact point is `share`.
        let add_friction = |force: &mut Vec3A,
                            hessian: &mut Mat3A,
                            normal: Vec3A,
                            normal_force: f32,
                            motion: Vec3A,
                            share: f32| {
            if friction <= 0.0 || normal_force <= 0.0 {
                return;
            }
            // A body posed anew every frame jumps into the cloth by far
            // more than it presses on it; that overlap is not pressure.
            // Counted as such it would lock the cloth to a swinging limb.
            let normal_force = normal_force.min(MAX_FRICTION_LOAD);
            let tangential = motion - normal * motion.dot(normal);
            let length = tangential.length();
            let scale = friction * normal_force / length.max(stick);
            *force -= tangential * (scale * share);
            *hessian += (Mat3A::IDENTITY - outer(normal)) * (scale * share * share);
        };

        // --- cloth against cloth ------------------------------------------------
        let (r, _) = self.radii(input.thickness);
        for &(p, slot) in self.incident.of(i) {
            let pair = &self.pairs[p as usize];
            let (vector, c) = self.pair_vector(pair, x);
            let distance = vector.length();
            if distance >= r || distance < 1e-9 {
                continue;
            }
            let normal = vector / distance;
            let (slope, curvature) = capped_shell_energy(distance, r, SELF_STIFFNESS);
            let share = c[slot as usize];
            force -= normal * (slope * share);
            hessian += outer(normal) * (curvature * share * share);
            let ids = self.pair_vertices(pair);
            let motion = (0..4).fold(Vec3A::ZERO, |m, k| {
                m + (x[ids[k] as usize] - start[ids[k] as usize]) * c[k]
            });
            add_friction(&mut force, &mut hessian, normal, -slope, motion, share);
        }

        // --- the body through its distance field --------------------------------
        // The field holds the body at the end of the step; earlier in the
        // step its surface was further back along its velocity, so the
        // body sweeps through the step instead of arriving all at once.
        if let Some(field) = input.body {
            let velocity = Vec3A::from(field.velocity_at(xi.into()));
            if let Some((distance, gradient)) =
                field.sample((xi + velocity * input.body_lag).into())
                && distance < input.margin
                && let Some(normal) = Vec3A::from(gradient).try_normalize()
            {
                let (slope, curvature) = body_energy(distance, input.margin);
                force += normal * -slope;
                hessian += outer(normal) * curvature;
            }
        }
        if let Some(Some(grip)) = grip.get(i) {
            add_friction(
                &mut force,
                &mut hessian,
                grip.normal,
                grip.load,
                xi - start[i] - grip.velocity * h,
                1.0,
            );
        }
        // --- skin points against the faces this vertex is a corner of -----------
        for &(p, slot) in self.body_incident.of(i) {
            let pair = &self.body_pairs[p as usize];
            let t = self.triangles[pair.face as usize];
            let [a, b, c] = t.map(|v| x[v as usize]);
            let point = pair.point - pair.velocity * input.body_lag;
            let (q, weights) = closest_point_on_triangle(point, a, b, c);
            let height = (q - point).dot(pair.normal);
            // Only while the skin point is under this face (a hand sweeping
            // past has moved on to others) and not through it by more than
            // the margin (then it is on the far side, not poking through).
            // Half the margin: cloth lying on the skin is the distance
            // field's business. Measured along the skin's normal rather than
            // through the field, a full margin would disagree with it by a
            // fraction of a millimetre and the two would shake the cloth
            // between them; only skin that really pokes in acts here.
            let reach = SKIN_POINT_SHARE * input.margin;
            if height >= reach || height <= -input.margin || weights.iter().any(|&w| w <= 0.0) {
                continue;
            }
            let share = weights[slot as usize];
            let (slope, curvature) = body_energy(height, reach);
            let push = -slope;
            force += pair.normal * (push * share);
            hessian += outer(pair.normal) * (curvature * share * share);
            let motion = (0..3).fold(Vec3A::ZERO, |m, k| {
                m + (x[t[k] as usize] - start[t[k] as usize]) * weights[k]
            }) - pair.velocity * h;
            add_friction(&mut force, &mut hessian, pair.normal, push, motion, share);
        }
        // --- the floor -----------------------------------------------------------
        let floor = input.ground + input.margin;
        if xi.y < floor {
            let k = BODY_STIFFNESS;
            let push = k * (floor - xi.y);
            force += Vec3A::Y * push;
            hessian += outer(Vec3A::Y) * k;
            add_friction(&mut force, &mut hessian, Vec3A::Y, push, xi - start[i], 1.0);
        }

        if hessian.determinant().abs() < 1e-12 {
            return xi;
        }
        xi + hessian.inverse() * force
    }
}

#[inline]
fn outer(u: Vec3A) -> Mat3A {
    Mat3A::from_cols(u * u.x, u * u.y, u * u.z)
}

/// First and second derivative of the shell overlap energy at `distance`
/// for radius `r`: `½·k·(r − d)²` down to `r/2`, then a log barrier that
/// matches it to second order and grows without bound towards zero.
#[inline]
fn shell_energy(distance: f32, r: f32, k: f32) -> (f32, f32) {
    let tau = 0.5 * r;
    if distance >= tau {
        (-k * (r - distance), k)
    } else {
        let d = distance.max(1e-7);
        (-k * tau * (r - tau) / d, k * tau * (r - tau) / (d * d))
    }
}

/// [`shell_energy`] with its barrier continued as a straight slope below a
/// tenth of the radius: two layers the body squeezes together push back
/// hard, but finitely, so the body wins without the forces blowing up.
#[inline]
fn capped_shell_energy(distance: f32, r: f32, k: f32) -> (f32, f32) {
    let floor = 0.1 * r;
    if distance > floor {
        return shell_energy(distance, r, k);
    }
    let (slope, curvature) = shell_energy(floor, r, k);
    (slope - curvature * (floor - distance), curvature)
}

/// [`shell_energy`] against the body, `distance` being signed (negative
/// inside). Below a tenth of the margin the barrier continues as a
/// straight slope, so it stays finite and keeps pushing out of the skin.
#[inline]
fn body_energy(distance: f32, margin: f32) -> (f32, f32) {
    let floor = 0.1 * margin;
    if distance > floor {
        return shell_energy(distance, margin, BODY_STIFFNESS);
    }
    let (slope, curvature) = shell_energy(floor, margin, BODY_STIFFNESS);
    (slope - curvature * (floor - distance), curvature)
}

/// Closest points of segments `p0p1` and `q0q1`: parameters along each and
/// the distance. `None` for degenerate segments.
pub fn segment_segment(p0: Vec3A, p1: Vec3A, q0: Vec3A, q1: Vec3A) -> Option<(f32, f32, f32)> {
    let (d1, d2, r) = (p1 - p0, q1 - q0, p0 - q0);
    let (a, e, f) = (d1.dot(d1), d2.dot(d2), d2.dot(r));
    if a < 1e-14 || e < 1e-14 {
        return None;
    }
    let (b, c) = (d1.dot(d2), d1.dot(r));
    let denominator = a * e - b * b;
    let mut s = if denominator > 1e-12 {
        ((b * f - c * e) / denominator).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut t = (b * s + f) / e;
    if t < 0.0 {
        t = 0.0;
        s = (-c / a).clamp(0.0, 1.0);
    } else if t > 1.0 {
        t = 1.0;
        s = ((b - c) / a).clamp(0.0, 1.0);
    }
    let distance = (p0 + d1 * s).distance(q0 + d2 * t);
    Some((s, t, distance))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closest_points_of_crossing_segments() {
        let (s, t, d) = segment_segment(
            Vec3A::new(-1.0, 0.0, 0.0),
            Vec3A::new(1.0, 0.0, 0.0),
            Vec3A::new(0.0, -1.0, 0.5),
            Vec3A::new(0.0, 1.0, 0.5),
        )
        .unwrap();
        assert!((s - 0.5).abs() < 1e-6 && (t - 0.5).abs() < 1e-6 && (d - 0.5).abs() < 1e-6);
    }

    #[test]
    fn the_shell_energy_is_smooth_where_the_barrier_takes_over() {
        let (r, k) = (0.003, 1000.0);
        let below = shell_energy(0.5 * r - 1e-7, r, k);
        let above = shell_energy(0.5 * r + 1e-7, r, k);
        assert!((below.0 - above.0).abs() < 1e-3 * above.0.abs());
        assert!((below.1 - above.1).abs() < 1e-2 * above.1);
        assert!(shell_energy(1e-6, r, k).1 > 1e3 * k);
    }
}

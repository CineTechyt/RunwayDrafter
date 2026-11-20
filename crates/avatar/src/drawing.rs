//! Drawing on the avatar's skin, and flattening the areas the lines close
//! into pattern pieces (a 3D pen and Flatten, as in CLO).
//!
//! * **Lines** run over the skin between their points: the straight chord
//!   between two points is split until the pieces are short and each split
//!   point is pulled onto the skin ([`trace`]).
//! * **Areas**: from a clicked point, the skin's triangles are filled up to
//!   the triangles the lines run through. The lines touching the filled
//!   area, cut at the points where lines meet, are chained into its
//!   boundary ([`find_area`]).
//! * **Flattening**: the area's triangles (with the ones its boundary runs
//!   through) are laid flat with least squares conformal maps, then made
//!   as rigid as possible (ARAP: every triangle keeps its shape as well as
//!   the others let it), so lengths on the pattern are lengths on the body.
//!   The boundary is carried over by its barycentric weights, turned so that
//!   up on the body is up on the pattern, and fitted with Bézier curves:
//!   corners where lines meet or turn sharply ([`flatten_area`]).

use std::collections::HashMap;

use cad_core::{AvatarLine, ControlPoint, Outline, PointKind, Segment, SkinPoint};
use glam::{Vec2, Vec3};
use physics_collision::TriangleMesh;

use crate::model::BodyMesh;

/// Distance between samples along a drawn line, metres.
pub const STEP: f32 = 0.004;
/// Points closer than this are the same point (lines meeting), metres.
const SAME_POINT: f32 = 0.0015;
/// A drawn line turning by more than this at one of its points gives the
/// pattern a corner there.
const CORNER_DEGREES: f32 = 40.0;
/// How closely the pattern's curves follow the flattened boundary, mm.
const FIT_TOLERANCE: f32 = 0.8;

/// The skin point's place on the body.
pub fn position(mesh: &BodyMesh, p: &SkinPoint) -> Option<Vec3> {
    let t = mesh.triangles.get(p.triangle as usize)?;
    Some(
        (0..3)
            .map(|k| mesh.positions[t[k] as usize] * p.weights[k])
            .sum(),
    )
}

/// The skin's outward normal at the point (of its triangle).
pub fn normal(mesh: &BodyMesh, p: &SkinPoint) -> Vec3 {
    mesh.triangles
        .get(p.triangle as usize)
        .map_or(Vec3::Z, |t| {
            let [a, b, c] = t.map(|i| mesh.positions[i as usize]);
            (b - a).cross(c - a).normalize_or(Vec3::Z)
        })
}

/// `point` (on or near triangle `triangle`) as a skin point.
pub fn skin_point(mesh: &BodyMesh, triangle: u32, point: Vec3) -> SkinPoint {
    let [a, b, c] = mesh.triangles[triangle as usize].map(|i| mesh.positions[i as usize]);
    let (v0, v1, v2) = (b - a, c - a, point - a);
    let (d00, d01, d11) = (v0.dot(v0), v0.dot(v1), v1.dot(v1));
    let (d20, d21) = (v2.dot(v0), v2.dot(v1));
    let denom = d00 * d11 - d01 * d01;
    let (v, w) = if denom.abs() < 1e-20 {
        (0.0, 0.0)
    } else {
        (
            (d11 * d20 - d01 * d21) / denom,
            (d00 * d21 - d01 * d20) / denom,
        )
    };
    let mut weights = [1.0 - v - w, v, w].map(|x| x.clamp(0.0, 1.0));
    let sum: f32 = weights.iter().sum();
    for x in &mut weights {
        *x /= sum.max(1e-9);
    }
    SkinPoint { triangle, weights }
}

/// Barycentric weights of `point` in triangle `triangle`, not clamped
/// (outside the triangle some are negative): to extrapolate.
fn barycentric(mesh: &BodyMesh, triangle: u32, point: Vec3) -> [f32; 3] {
    let [a, b, c] = mesh.triangles[triangle as usize].map(|i| mesh.positions[i as usize]);
    let (v0, v1, v2) = (b - a, c - a, point - a);
    let (d00, d01, d11) = (v0.dot(v0), v0.dot(v1), v1.dot(v1));
    let (d20, d21) = (v2.dot(v0), v2.dot(v1));
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-20 {
        return [1.0, 0.0, 0.0];
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    [1.0 - v - w, v, w]
}

/// The skin point nearest to `p`, within `reach` metres.
pub fn closest(mesh: &BodyMesh, collider: &TriangleMesh, p: Vec3, reach: f32) -> Option<SkinPoint> {
    let found = collider.closest_point(p.into(), reach)?;
    Some(skin_point(mesh, found.triangle, found.point.into()))
}

/// Where a ray from `origin` along `direction` (unit) meets the skin first.
pub fn hit(
    mesh: &BodyMesh,
    collider: &TriangleMesh,
    origin: Vec3,
    direction: Vec3,
) -> Option<SkinPoint> {
    let found = collider.raycast(origin.into(), direction.into(), 100.0)?;
    Some(skin_point(mesh, found.triangle, found.point.into()))
}

/// The path over the skin from `a` to `b`: points about [`STEP`] apart,
/// `a` and `b` included.
pub fn trace(
    mesh: &BodyMesh,
    collider: &TriangleMesh,
    a: SkinPoint,
    b: SkinPoint,
) -> Vec<SkinPoint> {
    fn split(
        mesh: &BodyMesh,
        collider: &TriangleMesh,
        a: (SkinPoint, Vec3),
        b: (SkinPoint, Vec3),
        depth: u32,
        out: &mut Vec<SkinPoint>,
    ) {
        let gap = a.1.distance(b.1);
        if gap > STEP && depth < 14 {
            let middle = (a.1 + b.1) * 0.5;
            if let Some(m) = closest(mesh, collider, middle, gap.max(0.01))
                && let Some(mp) = position(mesh, &m)
            {
                split(mesh, collider, a, (m, mp), depth + 1, out);
                split(mesh, collider, (m, mp), b, depth + 1, out);
                return;
            }
        }
        out.push(b.0);
    }
    let (Some(pa), Some(pb)) = (position(mesh, &a), position(mesh, &b)) else {
        return vec![a, b];
    };
    let mut out = vec![a];
    split(mesh, collider, (a, pa), (b, pb), 0, &mut out);
    out
}

/// A drawn line traced over the skin.
#[derive(Debug, Clone)]
pub struct Traced {
    pub line: u32,
    pub samples: Vec<SkinPoint>,
    pub positions: Vec<Vec3>,
    /// Index into `samples` of each of the line's points (a closed line's
    /// first point also ends it: one more entry).
    pub points: Vec<usize>,
}

pub fn trace_line(mesh: &BodyMesh, collider: &TriangleMesh, line: &AvatarLine) -> Traced {
    let mut samples: Vec<SkinPoint> = Vec::new();
    let mut points = Vec::new();
    let mut controls: Vec<SkinPoint> = line.points.clone();
    if line.closed && controls.len() >= 2 {
        controls.push(controls[0]);
    }
    for (k, &p) in controls.iter().enumerate() {
        if k == 0 {
            samples.push(p);
            points.push(0);
            continue;
        }
        let path = trace(mesh, collider, controls[k - 1], p);
        samples.extend_from_slice(&path[1..]);
        points.push(samples.len() - 1);
    }
    let positions = samples
        .iter()
        .map(|s| position(mesh, s).unwrap_or(Vec3::ZERO))
        .collect();
    Traced {
        line: line.id,
        samples,
        positions,
        points,
    }
}

/// A stretch of drawn line between two points where lines meet (or end).
#[derive(Debug, Clone)]
struct Edge {
    line: usize,
    /// Sample range in the traced line, both ends included.
    first: usize,
    last: usize,
    from: usize,
    to: usize,
    /// The line's points at either end.
    from_point: SkinPoint,
    to_point: SkinPoint,
}

/// The lines cut into edges between nodes (points where lines meet or
/// end).
struct Graph {
    traced: Vec<Traced>,
    edges: Vec<Edge>,
}

fn graph(mesh: &BodyMesh, collider: &TriangleMesh, lines: &[AvatarLine]) -> Graph {
    let traced: Vec<Traced> = lines
        .iter()
        .map(|l| trace_line(mesh, collider, l))
        .collect();
    // Every line point with its position.
    let mut all: Vec<(usize, usize, Vec3)> = Vec::new();
    for (l, line) in lines.iter().enumerate() {
        for (k, p) in line.points.iter().enumerate() {
            all.push((l, k, position(mesh, p).unwrap_or(Vec3::ZERO)));
        }
    }
    // Nodes: points shared by two lines (or twice by one), ends of open
    // lines; one point of a closed line that meets nothing.
    let mut nodes: Vec<Vec3> = Vec::new();
    let mut node_of: HashMap<(usize, usize), usize> = HashMap::new();
    let node_at = |p: Vec3, nodes: &mut Vec<Vec3>| {
        nodes
            .iter()
            .position(|n| n.distance(p) < SAME_POINT)
            .unwrap_or_else(|| {
                nodes.push(p);
                nodes.len() - 1
            })
    };
    for &(l, k, p) in &all {
        let shared = all
            .iter()
            .any(|&(l2, k2, p2)| (l2, k2) != (l, k) && p2.distance(p) < SAME_POINT);
        let end = !lines[l].closed && (k == 0 || k + 1 == lines[l].points.len());
        if shared || end {
            node_of.insert((l, k), node_at(p, &mut nodes));
        }
    }
    for (l, line) in lines.iter().enumerate() {
        if line.closed
            && !(0..line.points.len()).any(|k| node_of.contains_key(&(l, k)))
            && !all.is_empty()
        {
            let p = all.iter().find(|a| a.0 == l).map_or(Vec3::ZERO, |a| a.2);
            node_of.insert((l, 0), node_at(p, &mut nodes));
        }
    }
    let mut edges = Vec::new();
    for (l, line) in lines.iter().enumerate() {
        let n = line.points.len();
        if n < 2 {
            continue;
        }
        // Points in order along the line (a closed line comes back to its
        // first point at the end).
        let order: Vec<usize> = if line.closed {
            (0..=n).collect()
        } else {
            (0..n).collect()
        };
        let at = |i: usize| i % n;
        let node_positions: Vec<usize> = order
            .iter()
            .enumerate()
            .filter(|&(_, &k)| node_of.contains_key(&(l, at(k))))
            .map(|(i, _)| i)
            .collect();
        // A closed line starts and ends at its first node.
        let rotated: Vec<usize> = if line.closed {
            match node_positions.first() {
                Some(&first) => {
                    let mut r: Vec<usize> = (first..n).chain(0..=first).collect();
                    r.dedup();
                    r
                }
                None => continue,
            }
        } else {
            order.clone()
        };
        let mut start: Option<usize> = None;
        for (i, &k) in rotated.iter().enumerate() {
            if !node_of.contains_key(&(l, at(k))) {
                continue;
            }
            if let Some(s) = start {
                let (ks, ke) = (rotated[s], k);
                // Samples from point ks to point ke, along the line.
                let first = traced[l].points[ks];
                let last = if ke <= ks && line.closed {
                    traced[l].points[n] + traced[l].points[ke]
                } else {
                    traced[l].points[ke]
                };
                edges.push(Edge {
                    line: l,
                    first,
                    last,
                    from: node_of[&(l, at(ks))],
                    to: node_of[&(l, at(ke))],
                    from_point: line.points[at(ks)],
                    to_point: line.points[at(ke)],
                });
            }
            start = Some(i);
        }
    }
    Graph { traced, edges }
}

impl Graph {
    /// An edge's samples in order (a closed line's edge may wrap round).
    fn samples(&self, edge: &Edge) -> Vec<(SkinPoint, Vec3)> {
        let t = &self.traced[edge.line];
        let len = t.samples.len();
        // A closed line's samples end with its first point again.
        let period = len.saturating_sub(1).max(1);
        (edge.first..=edge.last)
            .map(|i| {
                let i = if i >= len { i % period } else { i };
                (t.samples[i], t.positions[i])
            })
            .collect()
    }

    /// Indices (into the traced samples) of points of the line inside the
    /// edge where it turns sharply.
    fn sharp_turns(&self, edge: &Edge, samples: &[(SkinPoint, Vec3)]) -> Vec<usize> {
        let t = &self.traced[edge.line];
        let cos = CORNER_DEGREES.to_radians().cos();
        t.points
            .iter()
            .filter_map(|&s| {
                let local = if s >= edge.first {
                    s - edge.first
                } else {
                    s + t.samples.len() - 1 - edge.first
                };
                (local > 0 && local + 1 < samples.len()).then_some(local)
            })
            .filter(|&k| {
                // Directions over a few samples either side.
                let reach = 3.min(k).min(samples.len() - 1 - k).max(1);
                let into = (samples[k].1 - samples[k - reach].1).normalize_or_zero();
                let out = (samples[k + reach].1 - samples[k].1).normalize_or_zero();
                into.dot(out) < cos
            })
            .collect()
    }
}

/// Neighbouring triangles across each edge of the skin.
pub fn adjacency(mesh: &BodyMesh) -> Vec<[u32; 3]> {
    let skin = mesh.skin();
    let mut by_edge: HashMap<(u32, u32), (u32, u8)> = HashMap::with_capacity(skin.len() * 2);
    let mut out = vec![[u32::MAX; 3]; skin.len()];
    for (t, tri) in skin.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            let key = (a.min(b), a.max(b));
            if let Some((other, ok)) = by_edge.remove(&key) {
                out[t][k] = other;
                out[other as usize][ok as usize] = t as u32;
            } else {
                by_edge.insert(key, (t as u32, k as u8));
            }
        }
    }
    out
}

/// An area of the skin closed by drawn lines.
pub struct Area {
    /// The triangles filled from the clicked point.
    pub filled: Vec<u32>,
    /// The triangles the boundary runs through.
    pub rim: Vec<u32>,
    /// The boundary's edges in order round the area: (edge, forwards).
    chain: Vec<(usize, bool)>,
    graph: Graph,
}

impl Area {
    /// The boundary as points on the body, in order.
    pub fn outline(&self) -> Vec<Vec3> {
        let mut out = Vec::new();
        for &(e, forwards) in &self.chain {
            let mut s = self.graph.samples(&self.graph.edges[e]);
            if !forwards {
                s.reverse();
            }
            out.extend(s.iter().take(s.len() - 1).map(|x| x.1));
        }
        out
    }
}

/// The area around `seed` closed by `lines`.
/// The triangles every edge of the graph runs through, and all of them
/// together.
fn blocked(
    mesh: &BodyMesh,
    collider: &TriangleMesh,
    adjacency: &[[u32; 3]],
    graph: &Graph,
) -> (Vec<bool>, Vec<Vec<u32>>) {
    let skin = adjacency.len();
    let mut blocked = vec![false; skin];
    let mut through: Vec<Vec<u32>> = Vec::with_capacity(graph.edges.len());
    for edge in &graph.edges {
        let samples = graph.samples(edge);
        let mut tris: Vec<u32> = samples.iter().map(|s| s.0.triangle).collect();
        // Between two samples in triangles that are not neighbours, a few
        // points more, so the line leaves no gap for the fill.
        for pair in samples.windows(2) {
            let (a, b) = (pair[0].0.triangle, pair[1].0.triangle);
            if a != b && !adjacency.get(a as usize).is_some_and(|n| n.contains(&b)) {
                for f in [0.25, 0.5, 0.75] {
                    if let Some(p) = closest(mesh, collider, pair[0].1.lerp(pair[1].1, f), 0.02) {
                        tris.push(p.triangle);
                    }
                }
            }
        }
        tris.sort_unstable();
        tris.dedup();
        for &t in &tris {
            if (t as usize) < skin {
                blocked[t as usize] = true;
            }
        }
        through.push(tris);
    }
    (blocked, through)
}

/// Areas of fewer triangles are not offered by [`closed_areas`]: the
/// narrow wedge of a dart between its two lines.
const MIN_AREA_TRIANGLES: usize = 15;

/// A point inside every area the lines close (one per area), for
/// flattening them all at once. The skin outside all lines is left out.
pub fn closed_areas(
    mesh: &BodyMesh,
    collider: &TriangleMesh,
    adjacency: &[[u32; 3]],
    lines: &[AvatarLine],
) -> Vec<SkinPoint> {
    let graph = graph(mesh, collider, lines);
    let skin = adjacency.len();
    let (blocked, _) = blocked(mesh, collider, adjacency, &graph);
    let mut seen = blocked.clone();
    let mut seeds = Vec::new();
    for start in 0..skin {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut stack = vec![start as u32];
        let mut members = Vec::new();
        while let Some(t) = stack.pop() {
            members.push(t);
            for &n in &adjacency[t as usize] {
                if n != u32::MAX && !seen[n as usize] {
                    seen[n as usize] = true;
                    stack.push(n);
                }
            }
        }
        if members.len() >= MIN_AREA_TRIANGLES && members.len() * 2 < skin {
            // A triangle well inside: the one whose neighbours are all in.
            let set: std::collections::HashSet<u32> = members.iter().copied().collect();
            let inner = members
                .iter()
                .copied()
                .find(|&t| adjacency[t as usize].iter().all(|n| set.contains(n)))
                .unwrap_or(members[0]);
            seeds.push(SkinPoint {
                triangle: inner,
                weights: [1.0 / 3.0; 3],
            });
        }
    }
    seeds
}

/// Where a ray (unit `direction`) lands on the skin: where it hits it, or,
/// when it passes the body by, the skin nearest to it within `reach`
/// metres, so lines can be drawn a little beside the body too.
pub fn pick(
    mesh: &BodyMesh,
    collider: &TriangleMesh,
    origin: Vec3,
    direction: Vec3,
    reach: f32,
) -> Option<SkinPoint> {
    if let Some(p) = hit(mesh, collider, origin, direction) {
        return Some(p);
    }
    // Where the ray is within reach of the body's box.
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for t in mesh.skin() {
        for &v in t {
            let p = mesh.positions[v as usize];
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    let (lo, hi) = (lo - reach, hi + reach);
    let (mut t0, mut t1) = (0.0f32, 100.0f32);
    for k in 0..3 {
        let d = direction[k];
        if d.abs() < 1e-9 {
            if origin[k] < lo[k] || origin[k] > hi[k] {
                return None;
            }
            continue;
        }
        let (a, b) = ((lo[k] - origin[k]) / d, (hi[k] - origin[k]) / d);
        t0 = t0.max(a.min(b));
        t1 = t1.min(a.max(b));
    }
    if t0 > t1 {
        return None;
    }
    let mut best: Option<(f32, SkinPoint)> = None;
    let steps = ((t1 - t0) / 0.01).ceil().clamp(1.0, 400.0) as usize;
    for k in 0..=steps {
        let p = origin + direction * (t0 + (t1 - t0) * k as f32 / steps as f32);
        if let Some(found) = collider.closest_point(p.into(), reach)
            && best.as_ref().is_none_or(|b| found.distance < b.0)
        {
            best = Some((
                found.distance,
                skin_point(mesh, found.triangle, found.point.into()),
            ));
        }
    }
    best.map(|b| b.1)
}

pub fn find_area(
    mesh: &BodyMesh,
    collider: &TriangleMesh,
    adjacency: &[[u32; 3]],
    lines: &[AvatarLine],
    seed: SkinPoint,
) -> Result<Area, String> {
    let graph = graph(mesh, collider, lines);
    let skin = adjacency.len();
    let (blocked, through) = blocked(mesh, collider, adjacency, &graph);
    let start = seed.triangle as usize;
    if start >= skin {
        return Err("Click on the avatar".into());
    }
    if blocked[start] {
        return Err("That is on a line: click inside an area".into());
    }
    let mut filled = vec![false; skin];
    let mut stack = vec![start as u32];
    filled[start] = true;
    let mut count = 0usize;
    while let Some(t) = stack.pop() {
        count += 1;
        for &n in &adjacency[t as usize] {
            if n != u32::MAX && !filled[n as usize] && !blocked[n as usize] {
                filled[n as usize] = true;
                stack.push(n);
            }
        }
    }
    if count * 2 > skin {
        return Err(
            "This area is not closed: draw lines all the way round it, meeting at their ends"
                .into(),
        );
    }
    // The edges along the filled area: their middle stretch borders it.
    // (Near its ends, a line leading away from a corner of the area also
    // touches it; and only about every other triangle a line crosses has
    // its free side towards the area, so any contact counts.)
    let borders = |t: u32| {
        adjacency[t as usize]
            .iter()
            .any(|&n| n != u32::MAX && filled[n as usize])
    };
    let middle_touches: Vec<bool> = graph
        .edges
        .iter()
        .map(|edge| {
            let samples = graph.samples(edge);
            let n = samples.len();
            let (a, b) = if n >= 5 { (n / 5, n - n / 5) } else { (0, n) };
            samples[a..b].iter().any(|s| borders(s.0.triangle))
        })
        .collect();
    let touches = |e: usize| middle_touches[e];
    let mut boundary: Vec<usize> = (0..graph.edges.len()).filter(|&e| touches(e)).collect();
    // Lines that end inside the area are no part of its boundary.
    loop {
        let degree = |node: usize, boundary: &[usize]| {
            boundary
                .iter()
                .map(|&e| {
                    let edge = &graph.edges[e];
                    usize::from(edge.from == node) + usize::from(edge.to == node)
                })
                .sum::<usize>()
        };
        let before = boundary.len();
        let snapshot = boundary.clone();
        boundary.retain(|&e| {
            let edge = &graph.edges[e];
            degree(edge.from, &snapshot) >= 2 && degree(edge.to, &snapshot) >= 2
        });
        if boundary.len() == before {
            break;
        }
    }
    if boundary.is_empty() {
        return Err("No lines around this area".into());
    }
    // Chain them round.
    let mut chain = vec![(boundary[0], true)];
    let mut used = vec![boundary[0]];
    let start_node = graph.edges[boundary[0]].from;
    let mut node = graph.edges[boundary[0]].to;
    while node != start_node {
        let next = boundary
            .iter()
            .copied()
            .filter(|e| !used.contains(e))
            .find_map(|e| {
                let edge = &graph.edges[e];
                if edge.from == node {
                    Some((e, true))
                } else if edge.to == node {
                    Some((e, false))
                } else {
                    None
                }
            });
        let Some((e, forwards)) = next else {
            return Err(
                "The lines round this area do not close: let them meet at their points".into(),
            );
        };
        used.push(e);
        chain.push((e, forwards));
        let edge = &graph.edges[e];
        node = if forwards { edge.to } else { edge.from };
    }
    if used.len() != boundary.len() {
        return Err("More than one loop of lines round this area (lines crossing?): let lines meet only at their points".into());
    }
    let mut rim: Vec<u32> = chain
        .iter()
        .flat_map(|&(e, _)| through[e].iter().copied())
        .collect();
    rim.sort_unstable();
    rim.dedup();
    Ok(Area {
        filled: (0..skin as u32).filter(|&t| filled[t as usize]).collect(),
        rim,
        chain,
        graph,
    })
}

/// One stretch of the flattened piece's outline and where it came from.
#[derive(Debug, Clone, PartialEq)]
pub struct FlatRun {
    pub line: u32,
    pub from: SkinPoint,
    pub to: SkinPoint,
    /// The outline's first edge of the run.
    pub edge: u32,
}

/// An area laid flat.
#[derive(Debug, Clone)]
pub struct Flattened {
    /// In mm, counter-clockwise, up on the body is +y.
    pub outline: Outline,
    pub runs: Vec<FlatRun>,
    /// The area's middle on the body and the skin's direction there.
    pub centre: Vec3,
    pub normal: Vec3,
    /// How far the skin had to stretch or shrink to lie flat: the 95th
    /// percentile over its triangles' edges (0.02 = 2 %).
    pub stretch: f32,
}

/// Lays the area around `seed` flat.
pub fn flatten_area(
    mesh: &BodyMesh,
    collider: &TriangleMesh,
    adjacency: &[[u32; 3]],
    lines: &[AvatarLine],
    seed: SkinPoint,
) -> Result<Flattened, String> {
    let area = find_area(mesh, collider, adjacency, lines, seed)?;
    let skin = mesh.skin();
    // The triangles the boundary runs through belong to the area if their
    // middle is on its side of the line: a dart's wedge is left out, so
    // the dart can open when the area lies flat.
    let centroid = |t: u32| {
        skin[t as usize]
            .iter()
            .map(|&v| mesh.positions[v as usize])
            .sum::<Vec3>()
            / 3.0
    };
    let path: Vec<(Vec3, Vec3)> = {
        let mut out = Vec::new();
        for &(e, forwards) in &area.chain {
            let mut s = area.graph.samples(&area.graph.edges[e]);
            if !forwards {
                s.reverse();
            }
            for pair in s.windows(2) {
                let n = normal(mesh, &pair[0].0);
                let tangent = (pair[1].1 - pair[0].1).normalize_or_zero();
                out.push((pair[0].1, tangent.cross(n)));
            }
        }
        out
    };
    let side = |p: Vec3| -> f32 {
        path.iter()
            .min_by(|a, b| a.0.distance_squared(p).total_cmp(&b.0.distance_squared(p)))
            .map_or(0.0, |(at, across)| (p - *at).dot(*across).signum())
    };
    let filled_set: std::collections::HashSet<u32> = area.filled.iter().copied().collect();
    // Which side is inside: as the filled triangles next to the line say.
    let inside: f32 = area
        .rim
        .iter()
        .flat_map(|&t| adjacency[t as usize].iter().copied())
        .filter(|n| filled_set.contains(n))
        .map(|n| side(centroid(n)))
        .sum::<f32>()
        .signum();
    let mut tris: Vec<u32> = area
        .filled
        .iter()
        .copied()
        .chain(
            area.rim
                .iter()
                .copied()
                .filter(|&t| side(centroid(t)) == inside),
        )
        .collect();
    tris.sort_unstable();
    tris.dedup();
    // The area's own little mesh.
    let mut local: HashMap<u32, u32> = HashMap::new();
    let mut positions: Vec<Vec3> = Vec::new();
    let triangles: Vec<[u32; 3]> = tris
        .iter()
        .map(|&t| {
            skin[t as usize].map(|v| {
                *local.entry(v).or_insert_with(|| {
                    positions.push(mesh.positions[v as usize]);
                    positions.len() as u32 - 1
                })
            })
        })
        .collect();
    let mut uv = flatten(&positions, &triangles);
    let tri_index: HashMap<u32, usize> = tris.iter().enumerate().map(|(k, &t)| (t, k)).collect();
    // A boundary point in a triangle left out (a dart's wedge) is carried
    // over by the nearest triangle that was laid flat, extrapolated.
    let flat = |p: &SkinPoint, uv: &[Vec2]| -> Vec2 {
        if let Some(&k) = tri_index.get(&p.triangle) {
            return (0..3)
                .map(|j| uv[triangles[k][j] as usize] * p.weights[j])
                .sum();
        }
        let at = position(mesh, p).unwrap_or(Vec3::ZERO);
        let mut near: Vec<u32> = vec![p.triangle];
        for _ in 0..4 {
            let more: Vec<u32> = near
                .iter()
                .flat_map(|&t| adjacency[t as usize].iter().copied())
                .filter(|&n| n != u32::MAX)
                .collect();
            near.extend(more);
            near.sort_unstable();
            near.dedup();
            if near.iter().any(|t| tri_index.contains_key(t)) {
                break;
            }
        }
        let best = near
            .iter()
            .filter_map(|t| Some((tri_index.get(t)?, centroid(*t).distance_squared(at))))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        let Some((&k, _)) = best else {
            return Vec2::ZERO;
        };
        let w = barycentric(mesh, tris[k], at);
        (0..3).map(|j| uv[triangles[k][j] as usize] * w[j]).sum()
    };

    // Seen from outside (the skin's triangles turn counter-clockwise
    // there), the pattern must not come out mirrored.
    let signed = |uv: &[Vec2], t: &[u32; 3]| {
        let [a, b, c] = t.map(|i| uv[i as usize]);
        (b - a).perp_dot(c - a)
    };
    let flipped: f32 = triangles.iter().map(|t| signed(&uv, t)).sum();
    if flipped < 0.0 {
        for p in &mut uv {
            p.x = -p.x;
        }
    }
    // Up on the body is up on the pattern: the mean direction the height
    // grows in, over the triangles.
    let mut up = Vec2::ZERO;
    for t in &triangles {
        let [a, b, c] = t.map(|i| uv[i as usize]);
        let [ya, yb, yc] = t.map(|i| positions[i as usize].y);
        let (e1, e2) = (b - a, c - a);
        let det = e1.perp_dot(e2);
        if det.abs() < 1e-12 {
            continue;
        }
        // Solve g·e1 = yb - ya, g·e2 = yc - ya.
        let g = Vec2::new(
            ((yb - ya) * e2.y - (yc - ya) * e1.y) / det,
            ((yc - ya) * e1.x - (yb - ya) * e2.x) / det,
        );
        up += g * det.abs();
    }
    let angle = std::f32::consts::FRAC_PI_2 - up.y.atan2(up.x);
    let turn = Vec2::from_angle(angle);
    for p in &mut uv {
        *p = turn.rotate(*p) * 1000.0;
    }

    // The boundary, edge by edge, with its corners.
    let mut runs_points: Vec<(Vec<Vec2>, bool, FlatRun)> = Vec::new();
    for &(e, forwards) in &area.chain {
        let edge = &area.graph.edges[e];
        let line = &lines[edge.line];
        let mut samples = area.graph.samples(edge);
        let mut turns = area.graph.sharp_turns(edge, &samples);
        let (mut from, mut to) = (edge.from_point, edge.to_point);
        if !forwards {
            samples.reverse();
            turns = turns.iter().map(|&k| samples.len() - 1 - k).collect();
            std::mem::swap(&mut from, &mut to);
        }
        turns.sort_unstable();
        let mut cuts = vec![0];
        cuts.extend(turns);
        cuts.push(samples.len() - 1);
        cuts.dedup();
        for pair in cuts.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if b <= a {
                continue;
            }
            let points: Vec<Vec2> = samples[a..=b].iter().map(|s| flat(&s.0, &uv)).collect();
            let run_from = if a == 0 { from } else { samples[a].0 };
            let run_to = if b == samples.len() - 1 {
                to
            } else {
                samples[b].0
            };
            runs_points.push((
                points,
                line.straight,
                FlatRun {
                    line: line.id,
                    from: run_from,
                    to: run_to,
                    edge: 0,
                },
            ));
        }
    }
    // Counter-clockwise.
    let ring: Vec<Vec2> = runs_points
        .iter()
        .flat_map(|r| r.0[..r.0.len() - 1].iter().copied())
        .collect();
    if cad_core::geometry::signed_area(&ring) < 0.0 {
        runs_points.reverse();
        for run in &mut runs_points {
            run.0.reverse();
            std::mem::swap(&mut run.2.from, &mut run.2.to);
        }
    }
    let mut points: Vec<ControlPoint> = Vec::new();
    let mut segments: Vec<Segment> = Vec::new();
    let mut runs = Vec::new();
    for (pts, straight, mut run) in runs_points {
        run.edge = segments.len() as u32;
        runs.push(run);
        let (start, end) = (pts[0], pts[pts.len() - 1]);
        if start.distance(end) < 1e-3 && pts.len() < 3 {
            continue;
        }
        if straight || cad_core::fit::is_straight(&pts, FIT_TOLERANCE) || pts.len() < 3 {
            points.push(ControlPoint {
                position: start,
                kind: PointKind::Corner,
            });
            segments.push(Segment::Line);
            continue;
        }
        let into = (pts[1] - pts[0]).normalize_or(Vec2::X);
        let back = (pts[pts.len() - 2] - pts[pts.len() - 1]).normalize_or(-Vec2::X);
        let cubics = cad_core::fit::fit_cubics(&pts, into, back, FIT_TOLERANCE);
        for (k, cubic) in cubics.iter().enumerate() {
            points.push(ControlPoint {
                position: cubic.p0,
                kind: if k == 0 {
                    PointKind::Corner
                } else {
                    PointKind::Smooth
                },
            });
            segments.push(Segment::Cubic {
                ctrl1: cubic.p1,
                ctrl2: cubic.p2,
            });
        }
    }
    if points.len() < 3 {
        return Err("This area is too small to flatten".into());
    }
    let outline = Outline::new(points, segments);

    let area_of = |p: &[Vec3], t: &[u32; 3]| {
        let [a, b, c] = t.map(|i| p[i as usize]);
        0.5 * (b - a).cross(c - a).length()
    };
    let inside: Vec<u32> = area.filled.clone();
    let (mut centre, mut normal, mut area_body) = (Vec3::ZERO, Vec3::ZERO, 0.0);
    for &t in &inside {
        let tri = skin[t as usize];
        let a = area_of(&mesh.positions, &tri);
        let [p0, p1, p2] = tri.map(|i| mesh.positions[i as usize]);
        centre += (p0 + p1 + p2) / 3.0 * a;
        normal += (p1 - p0).cross(p2 - p0);
        area_body += a;
    }
    // Weighted by length: the many tiny triangles of a detail such as a
    // nipple (which cloth bridges anyway) do not outweigh the rest.
    let mut weighted: Vec<(f32, f32)> = triangles
        .iter()
        .flat_map(|t| {
            (0..3).map(|k| {
                let (i, j) = (t[k] as usize, t[(k + 1) % 3] as usize);
                let body = positions[i].distance(positions[j]) * 1000.0;
                ((uv[i].distance(uv[j]) / body.max(1e-6) - 1.0).abs(), body)
            })
        })
        .collect();
    weighted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f32 = weighted.iter().map(|w| w.1).sum();
    let percentile = |q: f32| {
        let mut walked = 0.0;
        for &(ratio, length) in &weighted {
            walked += length;
            if walked >= q * total {
                return ratio;
            }
        }
        weighted.last().map_or(0.0, |w| w.0)
    };
    let stretch = percentile(0.95);
    let ratios: Vec<f32> = weighted.iter().map(|w| w.0).collect();
    if std::env::var_os("CLO360_FLAT_DEBUG").is_some() {
        let pick = |q: usize| percentile(q as f32 / 100.0) * 100.0;
        let _ = &ratios;
        let flipped_count = triangles.iter().filter(|t| signed(&uv, t) < 0.0).count();
        eprintln!(
            "flatten: {} triangles ({} filled, {} rim), stretch p50 {:.1} % p90 {:.1} % p95 {:.1} % max {:.1} %, {} flipped",
            triangles.len(),
            area.filled.len(),
            area.rim.len(),
            pick(50),
            pick(90),
            pick(95),
            pick(99),
            flipped_count
        );
    }
    let _ = area_body;
    Ok(Flattened {
        outline,
        runs,
        centre: centre / area_body.max(1e-9),
        normal: normal.normalize_or(Vec3::Z),
        stretch,
    })
}

// --- flattening a triangle mesh --------------------------------------------------

/// A symmetric sparse matrix in rows of (column, value).
struct Sparse {
    rows: Vec<Vec<(u32, f32)>>,
}

impl Sparse {
    fn new(n: usize) -> Self {
        Self {
            rows: vec![Vec::new(); n],
        }
    }

    fn add(&mut self, i: usize, j: usize, v: f32) {
        let row = &mut self.rows[i];
        match row.iter_mut().find(|(c, _)| *c as usize == j) {
            Some(entry) => entry.1 += v,
            None => row.push((j as u32, v)),
        }
    }

    fn mul(&self, x: &[f32], out: &mut [f32]) {
        for (i, row) in self.rows.iter().enumerate() {
            out[i] = row.iter().map(|&(j, v)| v * x[j as usize]).sum();
        }
    }

    fn diagonal(&self) -> Vec<f32> {
        self.rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .find(|(j, _)| *j as usize == i)
                    .map_or(1.0, |e| e.1)
            })
            .collect()
    }
}

/// Solves `a x = b` for symmetric positive definite `a` by conjugate
/// gradients (Jacobi preconditioned), starting from `x`. Rows in `fixed`
/// keep their value.
fn conjugate_gradients(a: &Sparse, b: &[f32], x: &mut [f32], fixed: &[bool], iterations: usize) {
    let n = b.len();
    let diag = a.diagonal();
    let mut ax = vec![0.0; n];
    a.mul(x, &mut ax);
    let mut r: Vec<f32> = (0..n)
        .map(|i| if fixed[i] { 0.0 } else { b[i] - ax[i] })
        .collect();
    let mut z: Vec<f32> = (0..n).map(|i| r[i] / diag[i].max(1e-12)).collect();
    let mut p = z.clone();
    let mut rz: f32 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let norm_b: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
    let mut ap = vec![0.0; n];
    for _ in 0..iterations {
        a.mul(&p, &mut ap);
        for i in 0..n {
            if fixed[i] {
                ap[i] = 0.0;
            }
        }
        let pap: f32 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        if pap.abs() < 1e-30 {
            break;
        }
        let alpha = rz / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let norm_r: f32 = r.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm_r < 1e-6 * norm_b {
            break;
        }
        for i in 0..n {
            z[i] = r[i] / diag[i].max(1e-12);
        }
        let rz_new: f32 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rz_new / rz.max(1e-30);
        rz = rz_new;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
}

/// Each triangle in its own plane: corners at (0, 0), (|e1|, 0) and the
/// third above the first edge.
fn local_frames(positions: &[Vec3], triangles: &[[u32; 3]]) -> Vec<[Vec2; 3]> {
    triangles
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|i| positions[i as usize]);
            let e1 = b - a;
            let e2 = c - a;
            let x = e1.normalize_or(Vec3::X);
            let n = e1.cross(e2);
            let y = n.cross(e1).normalize_or(Vec3::Y);
            [
                Vec2::ZERO,
                Vec2::new(e1.length(), 0.0),
                Vec2::new(e2.dot(x), e2.dot(y)),
            ]
        })
        .collect()
}

/// Lays a disc-like triangle mesh flat with as little stretch as it allows
/// (LSCM, then ARAP). Units are kept.
pub fn flatten(positions: &[Vec3], triangles: &[[u32; 3]]) -> Vec<Vec2> {
    let n = positions.len();
    if n < 3 || triangles.is_empty() {
        return vec![Vec2::ZERO; n];
    }
    let frames = local_frames(positions, triangles);
    let mut uv = lscm(positions, triangles, &frames);
    arap(triangles, &frames, &mut uv, 25);
    uv
}

/// Least squares conformal map with two far-apart vertices pinned.
fn lscm(positions: &[Vec3], triangles: &[[u32; 3]], frames: &[[Vec2; 3]]) -> Vec<Vec2> {
    let n = positions.len();
    let far = |from: usize| {
        (0..n)
            .max_by(|&a, &b| {
                positions[a]
                    .distance_squared(positions[from])
                    .total_cmp(&positions[b].distance_squared(positions[from]))
            })
            .unwrap_or(0)
    };
    let p0 = far(0);
    let p1 = far(p0);
    // Unknowns: u of every vertex, then v. Normal equations of the
    // residuals Re(W·U), Im(W·U) per triangle, weighted by 1 / sqrt(2 area).
    let mut m = Sparse::new(2 * n);
    for (t, tri) in triangles.iter().enumerate() {
        let q = frames[t];
        let area2 = (q[1] - q[0]).perp_dot(q[2] - q[0]).abs();
        if area2 < 1e-14 {
            continue;
        }
        let w = 1.0 / area2.sqrt();
        // W_j = q_{j+2} - q_{j+1} as complex numbers.
        let ws: [Vec2; 3] = [q[2] - q[1], q[0] - q[2], q[1] - q[0]];
        // Row 1: Σ Wr u − Wi v; row 2: Σ Wi u + Wr v.
        let rows: [[(usize, f32); 6]; 2] = [
            std::array::from_fn(|k| {
                let j = k % 3;
                if k < 3 {
                    (tri[j] as usize, ws[j].x * w)
                } else {
                    (n + tri[j] as usize, -ws[j].y * w)
                }
            }),
            std::array::from_fn(|k| {
                let j = k % 3;
                if k < 3 {
                    (tri[j] as usize, ws[j].y * w)
                } else {
                    (n + tri[j] as usize, ws[j].x * w)
                }
            }),
        ];
        for row in &rows {
            for &(i, a) in row {
                for &(j, b) in row {
                    m.add(i, j, a * b);
                }
            }
        }
    }
    let mut x = vec![0.0; 2 * n];
    let length = positions[p0].distance(positions[p1]);
    x[p1] = length;
    let mut fixed = vec![false; 2 * n];
    for i in [p0, p1, n + p0, n + p1] {
        fixed[i] = true;
    }
    // Start from a projection onto the plane through the two pins and the
    // mean normal: close, so few iterations are needed.
    let axis = (positions[p1] - positions[p0]).normalize_or(Vec3::X);
    let mut normal = Vec3::ZERO;
    for t in triangles {
        let [a, b, c] = t.map(|i| positions[i as usize]);
        normal += (b - a).cross(c - a);
    }
    let side = normal.cross(axis).normalize_or(Vec3::Y);
    for i in 0..n {
        if !fixed[i] {
            let d = positions[i] - positions[p0];
            x[i] = d.dot(axis);
            x[n + i] = d.dot(side);
        }
    }
    let b = {
        // Move the pinned values to the right-hand side.
        let mut pinned = vec![0.0; 2 * n];
        for i in 0..2 * n {
            if fixed[i] {
                pinned[i] = x[i];
            }
        }
        let mut mp = vec![0.0; 2 * n];
        m.mul(&pinned, &mut mp);
        mp.iter().map(|v| -v).collect::<Vec<f32>>()
    };
    // Solve for the free unknowns only: the pinned columns are on the
    // right; the operator below ignores them.
    let mut free = Sparse::new(2 * n);
    for (i, row) in m.rows.iter().enumerate() {
        if fixed[i] {
            free.add(i, i, 1.0);
            continue;
        }
        for &(j, v) in row {
            if !fixed[j as usize] {
                free.add(i, j as usize, v);
            }
        }
    }
    conjugate_gradients(&free, &b, &mut x, &fixed, 4 * n + 200);
    (0..n).map(|i| Vec2::new(x[i], x[n + i])).collect()
}

/// As-rigid-as-possible: alternately the best rotation per triangle and
/// the positions that fit them best (cotangent Laplacian, one pinned
/// vertex).
fn arap(triangles: &[[u32; 3]], frames: &[[Vec2; 3]], uv: &mut [Vec2], iterations: usize) {
    let n = uv.len();
    // Cotangent weight of each triangle edge (opposite its corner).
    let cot = |a: Vec2, b: Vec2| {
        let cross = a.perp_dot(b).abs();
        if cross < 1e-12 { 0.0 } else { a.dot(b) / cross }
    };
    let weights: Vec<[f32; 3]> = frames
        .iter()
        .map(|q| {
            std::array::from_fn(|k| {
                // Edge (k+1, k+2), opposite corner k.
                let (i, j) = ((k + 1) % 3, (k + 2) % 3);
                (0.5 * cot(q[i] - q[k], q[j] - q[k])).max(1e-4)
            })
        })
        .collect();
    let mut laplacian = Sparse::new(n);
    for (t, tri) in triangles.iter().enumerate() {
        for k in 0..3 {
            let (i, j) = (tri[(k + 1) % 3] as usize, tri[(k + 2) % 3] as usize);
            let w = weights[t][k];
            laplacian.add(i, i, w);
            laplacian.add(j, j, w);
            laplacian.add(i, j, -w);
            laplacian.add(j, i, -w);
        }
    }
    let mut fixed = vec![false; n];
    fixed[0] = true;
    // Vertex 0 stays put: its row reads x0 = value, and its column moves
    // to the right-hand side.
    let mut column0: Vec<(usize, f32)> = Vec::new();
    for (i, row) in laplacian.rows.iter_mut().enumerate() {
        if i == 0 {
            *row = vec![(0, 1.0)];
        } else {
            if let Some(&(_, w)) = row.iter().find(|&&(j, _)| j == 0) {
                column0.push((i, w));
            }
            row.retain(|&(j, _)| j != 0);
        }
    }
    let (mut xs, mut ys) = (
        uv.iter().map(|p| p.x).collect::<Vec<_>>(),
        uv.iter().map(|p| p.y).collect::<Vec<_>>(),
    );
    for _ in 0..iterations {
        // Local: the rotation that fits each triangle best.
        let rotations: Vec<Vec2> = triangles
            .iter()
            .enumerate()
            .map(|(t, tri)| {
                let q = frames[t];
                let (mut a, mut b, mut c, mut d) = (0.0, 0.0, 0.0, 0.0);
                for k in 0..3 {
                    let (i, j) = ((k + 1) % 3, (k + 2) % 3);
                    let w = weights[t][k];
                    let du = Vec2::new(
                        xs[tri[i] as usize] - xs[tri[j] as usize],
                        ys[tri[i] as usize] - ys[tri[j] as usize],
                    );
                    let dq = q[i] - q[j];
                    a += w * du.x * dq.x;
                    b += w * du.x * dq.y;
                    c += w * du.y * dq.x;
                    d += w * du.y * dq.y;
                }
                let angle = (c - b).atan2(a + d);
                Vec2::from_angle(angle)
            })
            .collect();
        // Global: positions that fit the rotated triangles best.
        let mut bx = vec![0.0; n];
        let mut by = vec![0.0; n];
        for (t, tri) in triangles.iter().enumerate() {
            let q = frames[t];
            for k in 0..3 {
                let (i, j) = ((k + 1) % 3, (k + 2) % 3);
                let w = weights[t][k];
                let r = rotations[t].rotate(q[i] - q[j]) * w;
                let (vi, vj) = (tri[i] as usize, tri[j] as usize);
                bx[vi] += r.x;
                by[vi] += r.y;
                bx[vj] -= r.x;
                by[vj] -= r.y;
            }
        }
        bx[0] = xs[0];
        by[0] = ys[0];
        for &(i, w) in &column0 {
            bx[i] -= w * xs[0];
            by[i] -= w * ys[0];
        }
        conjugate_gradients(&laplacian, &bx, &mut xs, &fixed, 2 * n + 100);
        conjugate_gradients(&laplacian, &by, &mut ys, &fixed, 2 * n + 100);
    }
    for i in 0..n {
        uv[i] = Vec2::new(xs[i], ys[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A patch of a cylinder unrolls to a rectangle of the same size.
    #[test]
    fn a_cylinder_patch_unrolls_without_stretch() {
        let (radius, rows, cols) = (0.15, 12, 16);
        let angle = 1.6; // radians around
        let height = 0.3;
        let mut positions = Vec::new();
        for r in 0..=rows {
            for c in 0..=cols {
                let a = angle * c as f32 / cols as f32;
                positions.push(Vec3::new(
                    radius * a.sin(),
                    height * r as f32 / rows as f32,
                    radius * a.cos(),
                ));
            }
        }
        let mut triangles = Vec::new();
        let at = |r: u32, c: u32| r * (cols + 1) + c;
        for r in 0..rows {
            for c in 0..cols {
                triangles.push([at(r, c), at(r, c + 1), at(r + 1, c + 1)]);
                triangles.push([at(r, c), at(r + 1, c + 1), at(r + 1, c)]);
            }
        }
        let uv = flatten(&positions, &triangles);
        // Every mesh edge keeps its length.
        let mut worst: f32 = 0.0;
        for t in &triangles {
            for k in 0..3 {
                let (i, j) = (t[k] as usize, t[(k + 1) % 3] as usize);
                let d3 = positions[i].distance(positions[j]);
                let d2 = uv[i].distance(uv[j]);
                worst = worst.max((d2 / d3 - 1.0).abs());
            }
        }
        assert!(
            worst < 0.01,
            "edges stretched by up to {:.2} %",
            worst * 100.0
        );
        let width = uv[at(0, cols) as usize].distance(uv[at(0, 0) as usize]);
        assert!(
            (width - radius * angle).abs() < 0.002,
            "unrolled width {width}"
        );
    }
}

//! Meshing of a single pattern piece.
//!
//! The boundary is sampled edge by edge at uniform arc length, the interior is
//! filled with a hexagonal point lattice, and the lot is connected with a
//! constrained Delaunay triangulation. That gives the near-equilateral
//! triangles of uniform size ("particle distance") that cloth solvers want.

use cad_core::geometry::{self, Curve};
use cad_core::glam::Vec2;
use cad_core::{EdgeRef, PatternPiece};
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};

use crate::mesh::VertexOrigin;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshingParams {
    /// Target triangle edge length in mm.
    pub particle_distance: f32,
    /// Curve flattening tolerance in mm used before resampling.
    pub tolerance: f32,
}

impl Default for MeshingParams {
    fn default() -> Self {
        Self {
            particle_distance: 20.0,
            tolerance: 0.05,
        }
    }
}

impl MeshingParams {
    /// How many mesh edges a pattern edge of length `length` (mm) wants.
    pub fn subdivisions(&self, length: f32) -> u32 {
        ((length / self.particle_distance).round() as u32).max(1)
    }
}

/// A meshed piece in pattern space, with piece-local vertex indices.
#[derive(Debug, Clone, Default)]
pub struct PieceMesh {
    pub pattern_positions: Vec<Vec2>,
    pub origins: Vec<VertexOrigin>,
    pub indices: Vec<u32>,
    /// For every pattern edge, the ordered chain of vertices along it,
    /// including the corner vertices at both ends.
    pub edge_chains: Vec<Vec<u32>>,
    /// Mesh edges along the piece's fold lines: (vertex, vertex, fold
    /// angle in degrees).
    pub folds: Vec<(u32, u32, f32)>,
}

/// Arc length of every edge of a piece.
pub fn edge_lengths(piece: &PatternPiece, params: &MeshingParams) -> Vec<f32> {
    piece
        .outline
        .curves()
        .map(|c| c.length(params.tolerance))
        .collect()
}

/// Meshes `piece`. `subdivisions[e]` is the number of mesh edges along pattern
/// edge `e`; the caller chooses it so that both sides of a seam match.
pub fn mesh_piece(piece: &PatternPiece, subdivisions: &[u32], params: &MeshingParams) -> PieceMesh {
    let edge_count = piece.outline.edge_count() as usize;
    assert_eq!(subdivisions.len(), edge_count);
    let mut mesh = PieceMesh::default();
    if edge_count == 0 {
        return mesh;
    }

    // --- boundary ----------------------------------------------------------
    let mut starts = Vec::with_capacity(edge_count);
    for (e, curve) in piece.outline.curves().enumerate() {
        let n = subdivisions[e].max(1);
        let samples = geometry::resample_polyline(&curve.flatten(params.tolerance), n);
        starts.push(mesh.pattern_positions.len() as u32);
        for (k, &p) in samples[..n as usize].iter().enumerate() {
            mesh.pattern_positions.push(p);
            mesh.origins.push(if k == 0 {
                VertexOrigin::Corner {
                    piece: piece.id,
                    anchor: e as u32,
                }
            } else {
                VertexOrigin::Edge {
                    edge: EdgeRef::new(piece.id, e as u32),
                    t: k as f32 / n as f32,
                }
            });
        }
    }
    let boundary_len = mesh.pattern_positions.len() as u32;
    for e in 0..edge_count {
        let end = starts[(e + 1) % edge_count];
        let mut chain: Vec<u32> = (starts[e]..starts[e] + subdivisions[e].max(1)).collect();
        chain.push(end);
        mesh.edge_chains.push(chain);
    }
    let boundary: Vec<Vec2> = mesh.pattern_positions.clone();

    // --- fold lines --------------------------------------------------------
    // Vertices along every fold line, so that mesh edges lie on it and the
    // cloth can fold there and nowhere else.
    let h = params.particle_distance.max(1e-3);
    let mut fold_segments: Vec<(Vec2, Vec2)> = Vec::new();
    let mut fold_chains: Vec<(Vec<u32>, f32)> = Vec::new();
    for fold in &piece.folds {
        let length = fold.a.distance(fold.b);
        if length < 1e-3 {
            continue;
        }
        fold_segments.push((fold.a, fold.b));
        let n = (length / h).ceil().max(1.0) as u32;
        let mut chain = Vec::new();
        for k in 0..=n {
            let p = fold.a.lerp(fold.b, k as f32 / n as f32);
            // Inside, and clear of the outline (whose own vertices end it).
            if geometry::point_in_polygon(p, &boundary)
                && geometry::distance_to_polygon(p, &boundary) > 0.35 * h
            {
                chain.push(mesh.pattern_positions.len() as u32);
                mesh.pattern_positions.push(p);
                mesh.origins
                    .push(VertexOrigin::Interior { piece: piece.id });
            } else if !chain.is_empty() {
                fold_chains.push((std::mem::take(&mut chain), fold.angle));
            }
        }
        if !chain.is_empty() {
            fold_chains.push((chain, fold.angle));
        }
    }
    let near_fold = |p: Vec2| {
        fold_segments.iter().any(|&(a, b)| {
            let ab = b - a;
            let t = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
            p.distance(a + ab * t) < 0.5 * h
        })
    };

    // --- interior lattice --------------------------------------------------
    let (lo, hi) = boundary
        .iter()
        .fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(lo, hi), &p| {
            (lo.min(p), hi.max(p))
        });
    let row_height = h * 0.866_025_4;
    let rows = ((hi.y - lo.y) / row_height).ceil() as i32;
    let cols = ((hi.x - lo.x) / h).ceil() as i32;
    for row in 0..=rows {
        let shift = if row % 2 == 0 { 0.0 } else { 0.5 * h };
        for col in 0..=cols {
            let p = Vec2::new(
                lo.x + col as f32 * h + shift,
                lo.y + row as f32 * row_height,
            );
            // Keep clear of the boundary so no sliver triangles appear there.
            if geometry::point_in_polygon(p, &boundary)
                && geometry::distance_to_polygon(p, &boundary) > 0.6 * h
                && !near_fold(p)
            {
                mesh.pattern_positions.push(p);
                mesh.origins
                    .push(VertexOrigin::Interior { piece: piece.id });
            }
        }
    }

    // --- constrained Delaunay ----------------------------------------------
    let mut cdt = ConstrainedDelaunayTriangulation::<Point2<f64>>::new();
    // spade merges coincident points, so its vertex indices can diverge from
    // ours on degenerate input; keep an explicit mapping both ways.
    let mut spade_to_local: Vec<u32> = Vec::with_capacity(mesh.pattern_positions.len());
    let mut handles = Vec::with_capacity(mesh.pattern_positions.len());
    for (local, p) in mesh.pattern_positions.iter().enumerate() {
        let Ok(handle) = cdt.insert(Point2::new(p.x as f64, p.y as f64)) else {
            return PieceMesh::default(); // NaN / out-of-range coordinates
        };
        if handle.index() == spade_to_local.len() {
            spade_to_local.push(local as u32);
        }
        handles.push(handle);
    }
    for k in 0..boundary_len as usize {
        let (a, b) = (handles[k], handles[(k + 1) % boundary_len as usize]);
        // A self-intersecting outline (mid-drag, say) must not panic.
        if a != b && cdt.can_add_constraint(a, b) {
            cdt.add_constraint(a, b);
        }
    }
    for (chain, angle) in &fold_chains {
        for w in chain.windows(2) {
            let (a, b) = (handles[w[0] as usize], handles[w[1] as usize]);
            if a != b && cdt.can_add_constraint(a, b) {
                cdt.add_constraint(a, b);
                mesh.folds.push((w[0], w[1], *angle));
            }
        }
    }

    for face in cdt.inner_faces() {
        let [a, b, c] = face.vertices();
        let centroid = [a, b, c].iter().fold(Vec2::ZERO, |acc, v| {
            let p = v.position();
            acc + Vec2::new(p.x as f32, p.y as f32)
        }) / 3.0;
        // Three near-collinear boundary samples (on a straight edge) can
        // form a sliver with no area; it would only add an ill-conditioned
        // element to the simulation.
        let [pa, pb, pc] =
            [a, b, c].map(|v| Vec2::new(v.position().x as f32, v.position().y as f32));
        let sliver = (pb - pa).perp_dot(pc - pa).abs() < 1e-3 * h * h;
        if !sliver && geometry::point_in_polygon(centroid, &boundary) {
            // spade emits counter-clockwise faces, so every piece gets +Z
            // normals in pattern space regardless of its outline winding.
            mesh.indices
                .extend([a, b, c].map(|v| spade_to_local[v.fix().index()]));
        }
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_core::PatternDocument;

    #[test]
    fn meshes_the_demo_front() {
        let doc = PatternDocument::demo_top();
        let piece = &doc.pieces()[0];
        let params = MeshingParams::default();
        let plan: Vec<u32> = edge_lengths(piece, &params)
            .iter()
            .map(|&l| params.subdivisions(l))
            .collect();
        let mesh = mesh_piece(piece, &plan, &params);

        assert!(mesh.indices.len() > 300);
        assert_eq!(mesh.edge_chains.len(), 8);
        for (e, chain) in mesh.edge_chains.iter().enumerate() {
            assert_eq!(chain.len() as u32, plan[e] + 1);
            // Chains share their end points with the neighbouring edges.
            assert_eq!(*chain.last().unwrap(), mesh.edge_chains[(e + 1) % 8][0]);
        }

        // The triangles tile the outline: their area matches the polygon's.
        let area: f32 = mesh
            .indices
            .chunks_exact(3)
            .map(|t| {
                let [a, b, c] = [0, 1, 2].map(|k| mesh.pattern_positions[t[k] as usize]);
                0.5 * (b - a).perp_dot(c - a)
            })
            .sum();
        let boundary_len: u32 = plan.iter().sum();
        let polygon = &mesh.pattern_positions[..boundary_len as usize];
        let expected = geometry::signed_area(polygon);
        assert!(
            (area - expected).abs() < 1e-3 * expected,
            "{area} vs {expected}"
        );
    }
}

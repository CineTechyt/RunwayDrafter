//! The assembled garment mesh shared by the renderer and the cloth solver.

use std::collections::HashMap;
use std::ops::Range;

use cad_core::glam::{Vec2, Vec3};
use cad_core::{EdgeRef, PieceId, SeamId};

/// Where a mesh vertex came from in the pattern.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VertexOrigin {
    /// Sits on anchor point `anchor`; shared by two pattern edges.
    Corner { piece: PieceId, anchor: u32 },
    /// Sampled along a pattern edge at arc-length fraction `t`.
    Edge { edge: EdgeRef, t: f32 },
    /// Generated inside the piece.
    Interior { piece: PieceId },
}

impl VertexOrigin {
    pub fn piece(&self) -> PieceId {
        match *self {
            Self::Corner { piece, .. } | Self::Interior { piece } => piece,
            Self::Edge { edge, .. } => edge.piece,
        }
    }
}

/// The contiguous block of the global buffers that belongs to one piece.
#[derive(Debug, Clone, PartialEq)]
pub struct PieceSpan {
    pub piece: PieceId,
    pub vertices: Range<u32>,
    /// Range into [`GarmentMesh::indices`] (three entries per triangle).
    pub indices: Range<u32>,
}

/// Two vertices that a seam sews together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stitch {
    /// The seam, or [`Stitch::FASTENING`] for a button done up through
    /// its buttonhole.
    pub seam: SeamId,
    pub a: u32,
    pub b: u32,
}

impl Stitch {
    /// The `seam` of the stitches that hold buttons in their buttonholes.
    pub const FASTENING: SeamId = SeamId(u32::MAX);
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    pub triangle: u32,
    /// Weights of the triangle's three vertices.
    pub barycentric: Vec3,
    /// Distance along the ray.
    pub distance: f32,
}

/// Every piece is a manifold disc; seams connect discs through [`Stitch`]es
/// rather than by welding, so each piece keeps its own UV island.
#[derive(Debug, Clone, Default)]
pub struct GarmentMesh {
    /// World-space positions in metres. The cloth solver writes these.
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    /// Pattern-space position (mm) of each vertex: the inverse mapping.
    pub pattern_positions: Vec<Vec2>,
    pub origins: Vec<VertexOrigin>,
    pub indices: Vec<u32>,
    pub pieces: Vec<PieceSpan>,
    pub stitches: Vec<Stitch>,
    /// Bumped whenever the buffers are reassembled.
    pub generation: u64,
}

impl GarmentMesh {
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn triangle(&self, triangle: u32) -> [u32; 3] {
        let i = triangle as usize * 3;
        [self.indices[i], self.indices[i + 1], self.indices[i + 2]]
    }

    pub fn span(&self, piece: PieceId) -> Option<&PieceSpan> {
        self.pieces.iter().find(|s| s.piece == piece)
    }

    /// The vertex of `piece` closest to a pattern-space point (mm).
    pub fn nearest_vertex(&self, piece: PieceId, pattern_position: Vec2) -> Option<u32> {
        let span = self.span(piece)?;
        span.vertices.clone().min_by(|&a, &b| {
            let distance =
                |v: u32| self.pattern_positions[v as usize].distance_squared(pattern_position);
            distance(a).total_cmp(&distance(b))
        })
    }

    /// The piece a triangle belongs to.
    pub fn piece_of_triangle(&self, triangle: u32) -> Option<PieceId> {
        let first_index = triangle * 3;
        self.pieces
            .iter()
            .find(|s| s.indices.contains(&first_index))
            .map(|s| s.piece)
    }

    /// Area-weighted vertex normals from the current positions.
    pub fn recompute_normals(&mut self) {
        self.normals.clear();
        self.normals.resize(self.positions.len(), Vec3::ZERO);
        for tri in self.indices.chunks_exact(3) {
            let [a, b, c] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
            let n = (self.positions[b] - self.positions[a])
                .cross(self.positions[c] - self.positions[a]);
            self.normals[a] += n;
            self.normals[b] += n;
            self.normals[c] += n;
        }
        for n in &mut self.normals {
            *n = n.normalize_or(Vec3::Z);
        }
    }

    /// `true` if no edge is shared by more than two triangles and no two
    /// triangles traverse a shared edge in the same direction.
    pub fn is_manifold(&self) -> bool {
        let mut directed: HashMap<(u32, u32), u32> = HashMap::new();
        for tri in self.indices.chunks_exact(3) {
            for k in 0..3 {
                *directed.entry((tri[k], tri[(k + 1) % 3])).or_default() += 1;
            }
        }
        directed.values().all(|&count| count == 1)
    }

    /// Closest intersection of a ray with the garment (both faces).
    pub fn raycast(&self, origin: Vec3, direction: Vec3) -> Option<RayHit> {
        let mut best: Option<RayHit> = None;
        for (t, tri) in self.indices.chunks_exact(3).enumerate() {
            let a = self.positions[tri[0] as usize];
            let e1 = self.positions[tri[1] as usize] - a;
            let e2 = self.positions[tri[2] as usize] - a;
            // Möller–Trumbore.
            let p = direction.cross(e2);
            let det = e1.dot(p);
            if det.abs() < 1e-12 {
                continue;
            }
            let inv = 1.0 / det;
            let s = origin - a;
            let u = s.dot(p) * inv;
            let q = s.cross(e1);
            let v = direction.dot(q) * inv;
            if u < 0.0 || v < 0.0 || u + v > 1.0 {
                continue;
            }
            let distance = e2.dot(q) * inv;
            if distance > 1e-5 && best.is_none_or(|b| distance < b.distance) {
                best = Some(RayHit {
                    triangle: t as u32,
                    barycentric: Vec3::new(1.0 - u - v, u, v),
                    distance,
                });
            }
        }
        best
    }
}

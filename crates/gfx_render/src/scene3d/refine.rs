//! More polygons for the studio render.
//!
//! The simulated garment has triangles of a centimetre or two: fine for
//! the cloth's motion, but its folds show facets and its outline corners.
//! For a still, every triangle is cut into many (Phong tessellation: the
//! new points are pulled onto the curved surface the vertex normals
//! describe), and where the fabric has relief of its own big enough to
//! see, corduroy's ribs, the surface is displaced by it: the ribs become
//! geometry, standing out of the outline and shadowing each other.

use std::ops::Range;

use glam::{Vec2, Vec3};

use super::mesh::MeshVertex;

/// How strongly new points are pulled onto the curved surface (Phong
/// tessellation's usual 3/4).
const PULL: f32 = 0.75;
/// The most triangles a refined garment may have.
pub const MAX_TRIANGLES: usize = 6_000_000;

/// Corduroy's ribs, as the shader draws them (`cord_profile` in pbr.wgsl):
/// 0 in the furrow, 1 on a rib's top.
pub fn cord_profile(x: f32) -> f32 {
    (std::f32::consts::PI * (x - x.floor())).sin().max(0.0).powf(0.7)
}

/// How one piece is refined.
#[derive(Debug, Clone, Copy)]
pub struct Refinement {
    /// Each triangle edge is cut into 2^levels.
    pub levels: u32,
    /// Corduroy to displace by: ribs per UV unit across U and their depth
    /// in metres.
    pub ribs: Option<(f32, f32)>,
}

/// Refines the triangles `indices[range]` of every piece; returns the new
/// vertices, indices and each piece's new index range.
pub fn refine(
    vertices: &[MeshVertex],
    indices: &[u32],
    pieces: &[(Range<u32>, Refinement)],
) -> (Vec<MeshVertex>, Vec<u32>, Vec<Range<u32>>) {
    let mut out_vertices = Vec::new();
    let mut out_indices = Vec::new();
    let mut ranges = Vec::with_capacity(pieces.len());
    for (range, refinement) in pieces {
        let first = out_indices.len() as u32;
        for t in indices[range.start as usize..range.end as usize].chunks_exact(3) {
            let corners = [t[0], t[1], t[2]].map(|i| vertices[i as usize]);
            // The outside face carries the relief; the inside and the cut
            // edge only need to be smooth.
            let side = corners.iter().map(|v| v.wear[3]).fold(f32::INFINITY, f32::min);
            let outside = side > 0.5;
            let levels = if outside {
                refinement.levels
            } else if side < -0.5 {
                refinement.levels.min(2)
            } else {
                0
            };
            let ribs = refinement.ribs.filter(|_| outside);
            triangle(&corners, levels, ribs, &mut out_vertices, &mut out_indices);
        }
        ranges.push(first..out_indices.len() as u32);
    }
    (out_vertices, out_indices, ranges)
}

/// Cuts one triangle into 4^levels.
fn triangle(
    corners: &[MeshVertex; 3],
    levels: u32,
    ribs: Option<(f32, f32)>,
    out_vertices: &mut Vec<MeshVertex>,
    out_indices: &mut Vec<u32>,
) {
    let n = 1u32 << levels;
    let base = out_vertices.len() as u32;
    let p = corners.map(|v| Vec3::from(v.position));
    let normal = corners.map(|v| Vec3::from(v.normal));
    let lerp = |w: [f32; 3], get: &dyn Fn(&MeshVertex) -> f32| {
        w[0] * get(&corners[0]) + w[1] * get(&corners[1]) + w[2] * get(&corners[2])
    };
    // Rows j = 0..=n along corner 2, i = 0..=n-j along corner 1.
    for j in 0..=n {
        for i in 0..=(n - j) {
            let (b, c) = (i as f32 / n as f32, j as f32 / n as f32);
            let w = [1.0 - b - c, b, c];
            let flat = p[0] * w[0] + p[1] * w[1] + p[2] * w[2];
            let curved: Vec3 = (0..3)
                .map(|k| (flat - normal[k] * (flat - p[k]).dot(normal[k])) * w[k])
                .sum();
            let mut position = flat.lerp(curved, PULL);
            let shading = (normal[0] * w[0] + normal[1] * w[1] + normal[2] * w[2]).normalize_or(normal[0]);
            let uv = Vec2::new(lerp(w, &|v| v.uv[0]), lerp(w, &|v| v.uv[1]));
            if let Some((count, depth)) = ribs {
                // Tops stay where the cloth is; the furrows sink.
                position -= shading * depth * (1.0 - cord_profile(uv.x * count));
            }
            let mut v = MeshVertex::new(position, shading, uv)
                .with_value(lerp(w, &|v| v.value))
                .with_color([0, 1, 2].map(|k| lerp(w, &|v| v.color[k])));
            v.wear = [0, 1, 2, 3].map(|k| lerp(w, &|v| v.wear[k]));
            out_vertices.push(v);
        }
    }
    // The index of point (i, j).
    let at = |i: u32, j: u32| base + j * (n + 1) - j * (j.saturating_sub(1)) / 2 + i;
    for j in 0..n {
        for i in 0..(n - j) {
            out_indices.extend([at(i, j), at(i + 1, j), at(i, j + 1)]);
            if i + 1 < n - j {
                out_indices.extend([at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
            }
        }
    }
}

/// Levels that give corduroy's ribs `per_rib` triangle edges each, for
/// triangles with edges of `edge` metres and ribs `rib` metres wide.
pub fn levels_for_ribs(edge: f32, rib: f32, per_rib: f32) -> u32 {
    if rib <= 0.0 || edge <= 0.0 {
        return 2;
    }
    ((edge * per_rib / rib).log2().ceil().max(2.0) as u32).min(6)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(p: Vec3, uv: Vec2) -> MeshVertex {
        let mut v = MeshVertex::new(p, Vec3::Z, uv);
        v.wear[3] = 1.0;
        v
    }

    #[test]
    fn a_flat_triangle_stays_flat_and_is_cut_into_four_to_the_level() {
        let vertices = [
            flat(Vec3::ZERO, Vec2::ZERO),
            flat(Vec3::X, Vec2::X),
            flat(Vec3::Y, Vec2::Y),
        ];
        let refinement = Refinement { levels: 3, ribs: None };
        let (v, i, ranges) = refine(&vertices, &[0, 1, 2], &[(0..3, refinement)]);
        assert_eq!(i.len() / 3, 64);
        assert_eq!(ranges, vec![0..192]);
        assert!(v.iter().all(|v| v.position[2].abs() < 1e-6));
        // Every index is a vertex of this triangle, every triangle faces +Z.
        for t in i.chunks_exact(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|k| Vec3::from(v[k as usize].position));
            assert!((b - a).cross(c - a).z > 0.0);
        }
    }

    #[test]
    fn corduroy_furrows_sink_into_the_cloth() {
        let vertices = [
            flat(Vec3::ZERO, Vec2::ZERO),
            flat(Vec3::X, Vec2::X),
            flat(Vec3::Y, Vec2::Y),
        ];
        let refinement = Refinement { levels: 5, ribs: Some((4.0, 0.001)) };
        let (v, _, _) = refine(&vertices, &[0, 1, 2], &[(0..3, refinement)]);
        let lowest = v.iter().map(|v| v.position[2]).fold(f32::INFINITY, f32::min);
        let highest = v.iter().map(|v| v.position[2]).fold(f32::NEG_INFINITY, f32::max);
        assert!(lowest < -0.0008, "{lowest}");
        assert!(highest.abs() < 1e-5, "{highest}");
    }

    #[test]
    fn levels_grow_with_finer_ribs() {
        assert_eq!(levels_for_ribs(0.02, 0.006, 5.0), 5);
        assert!(levels_for_ribs(0.02, 0.003, 5.0) > levels_for_ribs(0.02, 0.006, 5.0));
        assert_eq!(levels_for_ribs(0.02, 0.0005, 5.0), 6);
    }
}

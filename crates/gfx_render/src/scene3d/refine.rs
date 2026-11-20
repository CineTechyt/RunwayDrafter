//! More polygons for the studio render.
//!
//! The simulated garment has triangles of a centimetre or two: fine for
//! the cloth's motion, but its folds show facets and its outline corners.
//! For a still, every triangle is cut into many (Phong tessellation: the
//! new points are pulled onto the curved surface the vertex normals
//! describe), and where the fabric has relief of its own big enough to
//! see, corduroy's ribs or a scanned fabric's height map, the surface is
//! displaced by it: the ribs become geometry, standing out of the outline
//! and shadowing each other.

use std::ops::Range;
use std::sync::Arc;

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
    (std::f32::consts::PI * (x - x.floor()))
        .sin()
        .max(0.0)
        .powf(0.7)
}

/// A scan's height map (0 low … 1 high), repeating.
#[derive(Debug, Clone)]
pub struct HeightField {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
}

impl HeightField {
    /// The height map in the alpha of a surface texture, at most `max`
    /// samples across.
    pub fn from_surface(width: u32, height: u32, surface: &[[u8; 4]], max: u32) -> Self {
        let step = width.max(height).div_ceil(max.max(1)).max(1);
        let (w, h) = ((width / step).max(1), (height / step).max(1));
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0;
                for dy in 0..step {
                    for dx in 0..step {
                        let (sx, sy) = (
                            (x * step + dx).min(width - 1),
                            (y * step + dy).min(height - 1),
                        );
                        sum += f32::from(surface[(sy * width + sx) as usize][3]) / 255.0;
                    }
                }
                data.push(sum / (step * step) as f32);
            }
        }
        Self {
            width: w,
            height: h,
            data,
        }
    }

    /// Bilinear, wrapping, as the GPU samples it (texel centres at halves).
    pub fn sample(&self, uv: Vec2) -> f32 {
        let (w, h) = (self.width as i64, self.height as i64);
        let x = uv.x * w as f32 - 0.5;
        let y = uv.y * h as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let at = |i: i64, j: i64| self.data[(j.rem_euclid(h) * w + i.rem_euclid(w)) as usize];
        let (i, j) = (x0 as i64, y0 as i64);
        let top = at(i, j) * (1.0 - fx) + at(i + 1, j) * fx;
        let bottom = at(i, j + 1) * (1.0 - fx) + at(i + 1, j + 1) * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

/// The relief a surface is displaced by.
#[derive(Debug, Clone)]
pub enum Relief {
    /// Corduroy's ribs: per UV unit across U, and their depth in metres.
    Ribs { count: f32, depth: f32 },
    /// A scan's height map, `repeat` times per UV unit, `depth` metres
    /// deep, its patches shuffled as the shader does (`shuffled`).
    Map {
        field: Arc<HeightField>,
        repeat: [f32; 2],
        depth: f32,
        shuffled: bool,
    },
}

impl Relief {
    /// How far below the surface the cloth lies at `uv`, metres.
    fn sink(&self, uv: Vec2) -> f32 {
        match self {
            Relief::Ribs { count, depth } => depth * (1.0 - cord_profile(uv.x * count)),
            Relief::Map {
                field,
                repeat,
                depth,
                shuffled,
            } => {
                let at = uv * Vec2::from(*repeat);
                let h = if *shuffled {
                    let (offsets, weights) = stochastic_tiles(at);
                    (0..3)
                        .map(|k| weights[k] * field.sample(at + offsets[k]))
                        .sum()
                } else {
                    field.sample(at)
                };
                depth * (1.0 - h)
            }
        }
    }
}

/// The PCG hash of `hash_u` in pbr.wgsl.
fn hash(x: u32) -> u32 {
    let v = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let w = ((v >> ((v >> 28) + 4)) ^ v).wrapping_mul(277_803_737);
    (w >> 22) ^ w
}

fn hash22(cell: [i32; 2]) -> Vec2 {
    let a = hash(
        (cell[0] as u32).wrapping_mul(1_597_334_677) ^ (cell[1] as u32).wrapping_mul(3_812_015_801),
    );
    let b = hash(a);
    Vec2::new((a & 0xffff) as f32, (b & 0xffff) as f32) / 65536.0
}

/// `stochastic_tiles` of pbr.wgsl: the three patches around `uv`, their
/// offsets into the texture and weights.
pub fn stochastic_tiles(uv: Vec2) -> ([Vec2; 3], [f32; 3]) {
    let p = uv * 2.4;
    let skewed = Vec2::new(p.x - 0.577_350_27 * p.y, 1.154_700_5 * p.y);
    let cell = skewed.floor();
    let f = skewed - cell;
    let z = 1.0 - f.x - f.y;
    let c = [cell.x as i32, cell.y as i32];
    let (w, corners) = if z > 0.0 {
        ([z, f.y, f.x], [c, [c[0], c[1] + 1], [c[0] + 1, c[1]]])
    } else {
        (
            [-z, 1.0 - f.y, 1.0 - f.x],
            [[c[0] + 1, c[1] + 1], [c[0] + 1, c[1]], [c[0], c[1] + 1]],
        )
    };
    let sharp = w.map(|v| v * v * v);
    let sum = (sharp[0] + sharp[1] + sharp[2]).max(1e-5);
    (corners.map(hash22), sharp.map(|v| v / sum))
}

/// How one piece is refined.
#[derive(Debug, Clone)]
pub struct Refinement {
    /// Each triangle edge is cut into 2^levels.
    pub levels: u32,
    /// What the outside is displaced by.
    pub relief: Option<Relief>,
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
            let side = corners
                .iter()
                .map(|v| v.wear[3])
                .fold(f32::INFINITY, f32::min);
            let outside = side > 0.5;
            let levels = if outside {
                refinement.levels
            } else if side < -0.5 {
                refinement.levels.min(2)
            } else {
                0
            };
            let relief = refinement.relief.as_ref().filter(|_| outside);
            triangle(
                &corners,
                levels,
                relief,
                &mut out_vertices,
                &mut out_indices,
            );
        }
        ranges.push(first..out_indices.len() as u32);
    }
    (out_vertices, out_indices, ranges)
}

/// Cuts one triangle into 4^levels.
fn triangle(
    corners: &[MeshVertex; 3],
    levels: u32,
    relief: Option<&Relief>,
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
            let shading =
                (normal[0] * w[0] + normal[1] * w[1] + normal[2] * w[2]).normalize_or(normal[0]);
            let uv = Vec2::new(lerp(w, &|v| v.uv[0]), lerp(w, &|v| v.uv[1]));
            if let Some(relief) = relief {
                // Tops stay where the cloth is; the furrows sink.
                position -= shading * relief.sink(uv);
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
        let refinement = Refinement {
            levels: 3,
            relief: None,
        };
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
        let refinement = Refinement {
            levels: 5,
            relief: Some(Relief::Ribs {
                count: 4.0,
                depth: 0.001,
            }),
        };
        let (v, _, _) = refine(&vertices, &[0, 1, 2], &[(0..3, refinement)]);
        let lowest = v
            .iter()
            .map(|v| v.position[2])
            .fold(f32::INFINITY, f32::min);
        let highest = v
            .iter()
            .map(|v| v.position[2])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(lowest < -0.0008, "{lowest}");
        assert!(highest.abs() < 1e-5, "{highest}");
    }

    #[test]
    fn the_shuffled_patches_match_the_shader_s_hash() {
        // Values the WGSL hash gives (u32 arithmetic is the same).
        assert_eq!(hash(0), 129_708_002);
        let (offsets, weights) = stochastic_tiles(Vec2::new(0.3, 0.7));
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(
            offsets
                .iter()
                .all(|o| (0.0..1.0).contains(&o.x) && (0.0..1.0).contains(&o.y))
        );
        // Deterministic.
        assert_eq!(stochastic_tiles(Vec2::new(0.3, 0.7)).0, offsets);
    }

    #[test]
    fn a_height_field_samples_like_the_gpu() {
        let surface: Vec<[u8; 4]> = (0..16)
            .map(|i| [0, 0, 0, if i % 2 == 0 { 0 } else { 255 }])
            .collect();
        let field = HeightField::from_surface(4, 4, &surface, 64);
        // At a texel centre: that texel.
        assert!((field.sample(Vec2::new(0.125, 0.125)) - 0.0).abs() < 1e-6);
        assert!((field.sample(Vec2::new(0.375, 0.125)) - 1.0).abs() < 1e-6);
        // Between them: half.
        assert!((field.sample(Vec2::new(0.25, 0.125)) - 0.5).abs() < 1e-6);
        // Wraps.
        assert!((field.sample(Vec2::new(1.125, 0.125)) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn levels_grow_with_finer_ribs() {
        assert_eq!(levels_for_ribs(0.02, 0.006, 5.0), 5);
        assert!(levels_for_ribs(0.02, 0.003, 5.0) > levels_for_ribs(0.02, 0.006, 5.0));
        assert_eq!(levels_for_ribs(0.02, 0.0005, 5.0), 6);
    }
}

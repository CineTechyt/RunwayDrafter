//! Skinning with optimized centres of rotation (Le & Hodgins 2016).
//!
//! Linear blend skinning averages the bones' *matrices*: where two bones
//! turn far apart (the shoulder with the arm overhead, a twisted forearm)
//! the average is no longer a rotation and the skin collapses into a crease
//! or a "candy wrapper". Here each blended vertex is instead turned by the
//! blended *rotation* (a quaternion average, which stays rigid) about its
//! own centre of rotation, and that centre is carried along by linear blend
//! skinning. The centre is the area-weighted mean of all surface
//! triangles whose skin weights resemble the vertex's, so it lies inside
//! the limb, about where the joint really is for that patch of skin.
//!
//! Computing the centres directly compares every vertex with every
//! triangle. As the weights never change, triangles with nearly the same
//! weights are grouped once per model into bins; per body shape only the
//! bins' areas and centroids are summed, and a vertex visits only the bins
//! that share a pair of its bones.

use std::collections::HashMap;

use glam::{Affine3A, Mat3A, Quat, Vec3, Vec3A};
use rayon::prelude::*;

/// Width of the similarity kernel (the paper's σ).
const SIGMA: f32 = 0.1;
/// Weights are rounded to this step when binning triangles.
const QUANTUM: f32 = 0.05;
/// Rounds of neighbour averaging of the centres.
const SMOOTHING: usize = 6;

type Weights = ([u16; 4], [f32; 4]);

/// Triangles grouped by (rounded) skin weights.
#[derive(Debug, Clone, Default)]
pub(crate) struct Bins {
    /// The rounded weights of each bin, bones sorted, unused slots zero.
    weights: Vec<Weights>,
    /// Bin of every skin triangle.
    of_triangle: Vec<u32>,
    /// Bins containing both bones of a pair (lower bone first).
    by_pair: HashMap<(u16, u16), Vec<u32>>,
}

impl Bins {
    pub(crate) fn new(weights: &[Weights], triangles: &[[u32; 3]]) -> Self {
        let mut index: HashMap<Vec<(u16, u16)>, u32> = HashMap::new();
        let mut bins = Bins::default();
        for t in triangles {
            // The triangle's weights: the mean of its corners'.
            let mut sum: Vec<(u16, f32)> = Vec::with_capacity(8);
            for &v in t {
                let (bones, w) = weights[v as usize];
                for k in 0..4 {
                    if w[k] > 0.0 {
                        match sum.iter_mut().find(|(b, _)| *b == bones[k]) {
                            Some(entry) => entry.1 += w[k] / 3.0,
                            None => sum.push((bones[k], w[k] / 3.0)),
                        }
                    }
                }
            }
            sum.sort_by(|a, b| b.1.total_cmp(&a.1));
            sum.truncate(4);
            let total: f32 = sum.iter().map(|(_, w)| w).sum::<f32>().max(1e-9);
            let mut key: Vec<(u16, u16)> =
                sum.iter().map(|&(b, w)| (b, ((w / total) / QUANTUM).round() as u16)).filter(|&(_, q)| q > 0).collect();
            key.sort_unstable();
            let next = index.len() as u32;
            let bin = *index.entry(key.clone()).or_insert_with(|| {
                let mut bones = [0u16; 4];
                let mut w = [0.0f32; 4];
                for (k, &(b, q)) in key.iter().enumerate() {
                    bones[k] = b;
                    w[k] = q as f32 * QUANTUM;
                }
                bins.weights.push((bones, w));
                next
            });
            bins.of_triangle.push(bin);
        }
        for (bin, (bones, w)) in bins.weights.iter().enumerate() {
            for j in 0..4 {
                for k in j + 1..4 {
                    if w[j] > 0.0 && w[k] > 0.0 {
                        bins.by_pair.entry((bones[j].min(bones[k]), bones[j].max(bones[k]))).or_default().push(bin as u32);
                    }
                }
            }
        }
        bins
    }
}

/// Weight of bone `b` in sparse weights, zero if absent.
#[inline]
fn weight_of((bones, w): &Weights, b: u16) -> f32 {
    (0..4).find(|&k| bones[k] == b && w[k] > 0.0).map_or(0.0, |k| w[k])
}

/// The paper's similarity of two weight vectors: large when they blend the
/// same pairs of bones in the same proportion.
fn similarity(p: &Weights, v: &Weights) -> f32 {
    let mut s = 0.0;
    for j in 0..4 {
        for k in j + 1..4 {
            let (bj, bk) = (p.0[j], p.0[k]);
            let (pj, pk) = (p.1[j], p.1[k]);
            if pj <= 0.0 || pk <= 0.0 {
                continue;
            }
            let (vj, vk) = (weight_of(v, bj), weight_of(v, bk));
            if vj <= 0.0 || vk <= 0.0 {
                continue;
            }
            let d = pj * vk - pk * vj;
            s += pj * pk * vj * vk * (-(d * d) / (SIGMA * SIGMA)).exp();
        }
    }
    s
}

/// The centre of rotation of every vertex of a body in the modelled pose.
/// Vertices on a single bone keep their own position (any centre does).
pub(crate) fn centres(bins: &Bins, weights: &[Weights], positions: &[Vec3], triangles: &[[u32; 3]]) -> Vec<Vec3> {
    // Area and area-weighted centroid per bin, for this body shape.
    let mut area = vec![0.0f32; bins.weights.len()];
    let mut moment = vec![Vec3::ZERO; bins.weights.len()];
    for (t, &bin) in triangles.iter().zip(&bins.of_triangle) {
        let [a, b, c] = t.map(|i| positions[i as usize]);
        let w = 0.5 * (b - a).cross(c - a).length();
        area[bin as usize] += w;
        moment[bin as usize] += (a + b + c) * (w / 3.0);
    }
    weights
        .par_iter()
        .zip(positions)
        .with_min_len(256)
        .map(|(vertex, &p)| {
            let (bones, w) = vertex;
            let mut seen: Vec<u32> = Vec::new();
            let (mut total, mut sum) = (0.0f32, Vec3::ZERO);
            for j in 0..4 {
                for k in j + 1..4 {
                    if w[j] <= 0.0 || w[k] <= 0.0 {
                        continue;
                    }
                    let Some(list) = bins.by_pair.get(&(bones[j].min(bones[k]), bones[j].max(bones[k]))) else { continue };
                    for &bin in list {
                        if seen.contains(&bin) {
                            continue;
                        }
                        seen.push(bin);
                        let s = similarity(vertex, &bins.weights[bin as usize]);
                        total += s * area[bin as usize];
                        sum += moment[bin as usize] * s;
                    }
                }
            }
            if total > 1e-12 { sum / total } else { p }
        })
        .collect::<Vec<Vec3>>()
        .pipe(|centres| smooth(centres, weights, triangles))
}

/// Averages each blended vertex's centre with its blended neighbours'.
/// Skin weights limited to four bones swap a bone between neighbours now
/// and then; the centres, which depend on which bones blend, would jump
/// there and tear the skin.
fn smooth(mut centres: Vec<Vec3>, weights: &[Weights], triangles: &[[u32; 3]]) -> Vec<Vec3> {
    let blended: Vec<bool> = weights.iter().map(|(_, w)| w.iter().filter(|&&w| w > 0.0).count() > 1).collect();
    let mut neighbours: Vec<Vec<u32>> = vec![Vec::new(); weights.len()];
    for t in triangles {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            if blended[a as usize] && blended[b as usize] {
                neighbours[a as usize].push(b);
                neighbours[b as usize].push(a);
            }
        }
    }
    for _ in 0..SMOOTHING {
        let previous = centres.clone();
        centres.par_iter_mut().enumerate().with_min_len(512).for_each(|(v, centre)| {
            let around = &neighbours[v];
            if !around.is_empty() {
                let mean = around.iter().map(|&u| previous[u as usize]).sum::<Vec3>() / around.len() as f32;
                *centre = previous[v] * 0.5 + mean * 0.5;
            }
        });
    }
    centres
}

trait Pipe: Sized {
    fn pipe<T>(self, f: impl FnOnce(Self) -> T) -> T {
        f(self)
    }
}
impl<T> Pipe for T {}

/// Per-vertex transforms for the bone transforms `world`: rotation by the
/// blended quaternion about the vertex's centre, which is itself moved by
/// linear blend skinning.
pub(crate) fn transforms(weights: &[Weights], centres: &[Vec3], world: &[Affine3A]) -> Vec<Affine3A> {
    let rotations: Vec<Quat> = world.iter().map(|m| Quat::from_mat3a(&m.matrix3).normalize()).collect();
    weights
        .par_iter()
        .zip(centres)
        .with_min_len(512)
        .map(|((bones, w), &centre)| {
            // The heaviest bone sets the hemisphere of the average.
            let heaviest = (0..4).max_by(|&a, &b| w[a].total_cmp(&w[b])).unwrap_or(0);
            let reference = rotations[bones[heaviest] as usize];
            let mut q = glam::Vec4::ZERO;
            let mut moved = Vec3A::ZERO;
            let centre = Vec3A::from(centre);
            for k in 0..4 {
                if w[k] <= 0.0 {
                    continue;
                }
                let rotation = rotations[bones[k] as usize];
                let sign = if rotation.dot(reference) < 0.0 { -1.0 } else { 1.0 };
                q += glam::Vec4::from(rotation) * (w[k] * sign);
                moved += world[bones[k] as usize].transform_point3a(centre) * w[k];
            }
            let rotation = Mat3A::from_quat(Quat::from_vec4(q).normalize());
            Affine3A { matrix3: rotation, translation: moved - rotation * centre }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similar_weights_are_similar() {
        let a: Weights = ([1, 2, 0, 0], [0.5, 0.5, 0.0, 0.0]);
        let b: Weights = ([2, 1, 0, 0], [0.5, 0.5, 0.0, 0.0]);
        let c: Weights = ([1, 2, 0, 0], [0.9, 0.1, 0.0, 0.0]);
        let rigid: Weights = ([1, 0, 0, 0], [1.0, 0.0, 0.0, 0.0]);
        assert!((similarity(&a, &b) - 0.0625).abs() < 1e-6);
        assert!(similarity(&a, &c) < 0.01 * similarity(&a, &b));
        assert_eq!(similarity(&a, &rigid), 0.0);
    }

    #[test]
    fn a_single_bone_moves_its_skin_rigidly() {
        let world = [Affine3A::IDENTITY, Affine3A::from_rotation_translation(Quat::from_rotation_z(1.0), Vec3::new(0.1, 0.2, 0.3))];
        let weights = [([1u16, 0, 0, 0], [1.0, 0.0, 0.0, 0.0])];
        let p = Vec3::new(0.3, -0.2, 0.5);
        let t = transforms(&weights, &[Vec3::new(9.0, 9.0, 9.0)], &world);
        assert!((t[0].transform_point3(p) - world[1].transform_point3(p)).length() < 1e-5);
    }
}

//! Voxelises a body into the signed distance field cloth collides with.
//!
//! Cloth only collides at its vertices, so a body detail that is smaller
//! than the cloth's triangles (a nipple, a collar bone, a knuckle) can
//! poke through between them. The field can therefore be *closed*: the
//! surface is rolled over with a ball, which fills every dent and crease
//! the ball does not fit into and surrounds every small bump with a
//! fillet. The cloth then rests on a surface it is able to follow.

use glam::Vec3;
use rayon::prelude::*;
use topology_sync::DistanceField;

use crate::model::BodyMesh;

/// Distances are exact within this many voxels of the (closed) surface.
const MARGIN_VOXELS: i32 = 3;
/// Largest grid side; keeps the 3D texture within every GPU's limits.
const MAX_SIDE: f32 = 250.0;

/// Closest point to `p` on triangle `abc` (Ericson, Real-Time Collision
/// Detection, 5.1.5).
fn closest_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denominator = 1.0 / (va + vb + vc);
    a + ab * (vb * denominator) + ac * (vc * denominator)
}

/// Depth below the surface `{values = 0}` for the voxels inside it, by fast
/// sweeping (Zhao 2005). Outside voxels are taken as given, and so are the
/// inside voxels for which `known` is true; the rest are solved. Returns
/// positive depths for inside voxels.
fn depth_inside(values: &[f32], known: impl Fn(usize) -> bool, dims: [usize; 3], voxel: f32) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let at = |x: usize, y: usize, z: usize| x + nx * (y + ny * z);
    // Outside voxels are fixed at minus their distance, so the depth is
    // continuous across the surface and the crossing stays sub-voxel.
    let mut depth: Vec<f32> =
        values.iter().enumerate().map(|(i, &v)| if v >= 0.0 || known(i) { -v } else { f32::MAX }).collect();
    for sweep in 0..16 {
        let forward = |n: usize, flip: bool| -> Box<dyn Iterator<Item = usize>> { if flip { Box::new((0..n).rev()) } else { Box::new(0..n) } };
        let mut changed = false;
        for z in forward(nz, sweep & 4 != 0) {
            for y in forward(ny, sweep & 2 != 0) {
                for x in forward(nx, sweep & 1 != 0) {
                    let index = at(x, y, z);
                    if values[index] >= 0.0 || known(index) {
                        continue;
                    }
                    let lowest = |a: Option<usize>, b: Option<usize>| a.map_or(f32::MAX, |i| depth[i]).min(b.map_or(f32::MAX, |i| depth[i]));
                    let mut m = [
                        lowest((x > 0).then(|| at(x - 1, y, z)), (x + 1 < nx).then(|| at(x + 1, y, z))),
                        lowest((y > 0).then(|| at(x, y - 1, z)), (y + 1 < ny).then(|| at(x, y + 1, z))),
                        lowest((z > 0).then(|| at(x, y, z - 1)), (z + 1 < nz).then(|| at(x, y, z + 1))),
                    ];
                    m.sort_by(f32::total_cmp);
                    if m[0] == f32::MAX {
                        continue;
                    }
                    // Upwind solution of |grad u| = 1 using one, two or
                    // all three axes.
                    let mut u = m[0] + voxel;
                    if u > m[1] {
                        u = 0.5 * (m[0] + m[1] + (2.0 * voxel * voxel - (m[0] - m[1]).powi(2)).max(0.0).sqrt());
                        if u > m[2] {
                            let sum = m[0] + m[1] + m[2];
                            let squares = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
                            u = (sum + (sum * sum - 3.0 * (squares - voxel * voxel)).max(0.0).sqrt()) / 3.0;
                        }
                    }
                    if u < depth[index] - 1e-7 {
                        depth[index] = u;
                        changed = true;
                    }
                }
            }
        }
        if sweep >= 7 && !changed {
            break;
        }
    }
    depth
}

/// Builds the field with a grid spacing of about `voxel` metres (coarser if
/// the body would not fit the maximum grid size). `closing` is the radius
/// in metres of the ball the surface is closed with; 0 keeps the exact
/// body. Use roughly half the cloth's particle distance.
pub fn distance_field(mesh: &BodyMesh, voxel: f32, closing: f32) -> DistanceField {
    distance_field_within(mesh, voxel, closing, &FieldOptions::default())
}

/// Extras for [`distance_field_within`].
#[derive(Debug, Clone, Copy)]
pub struct FieldOptions<'a> {
    /// Only this part of the body (min, max corners).
    pub region: Option<(Vec3, Vec3)>,
    /// The body in motion: m/s per mesh vertex.
    pub velocities: Option<&'a [Vec3]>,
    /// Exact depth deep inside the body, so cloth pulled far in still knows
    /// the way out. Costly; not needed where cloth can only be a little
    /// inside (a body moving by small steps).
    pub deep: bool,
}

impl Default for FieldOptions<'_> {
    fn default() -> Self {
        Self { region: None, velocities: None, deep: true }
    }
}

/// Like [`distance_field`], covering only `region` (min, max corners) of
/// the body: much cheaper when only a garment's surroundings matter, such
/// as when the body moves every frame. Inside and out are decided near the
/// surface, so cutting through a limb at the region's edge does no harm
/// close to the skin.
///
/// With `velocities` (m/s per mesh vertex: the body in motion), the field
/// also records how fast the skin moves near each sample.
pub fn distance_field_within(mesh: &BodyMesh, voxel: f32, closing: f32, options: &FieldOptions<'_>) -> DistanceField {
    let FieldOptions { region, velocities, deep } = *options;
    let positions = &mesh.positions;
    let skin = mesh.skin();
    let (mut low, mut high) = skin
        .iter()
        .flatten()
        .map(|&i| positions[i as usize])
        .fold((Vec3::MAX, Vec3::MIN), |(lo, hi), p| (lo.min(p), hi.max(p)));
    if let Some((min, max)) = region {
        low = low.max(min);
        high = high.min(max);
        if low.cmpge(high).any() {
            return DistanceField { origin: Vec3::ZERO, voxel, dims: [2, 2, 2], values: vec![voxel; 8], velocity: Vec::new() };
        }
    }
    if skin.is_empty() {
        return DistanceField { origin: Vec3::ZERO, voxel, dims: [2, 2, 2], values: vec![voxel; 8], velocity: Vec::new() };
    }
    let closing = closing.max(0.0);
    let voxel = voxel.max((high - low).max_element() / (MAX_SIDE - 2.0 * (MARGIN_VOXELS as f32 + 2.0) - 2.0 * closing / voxel));
    // Closing first grows the body by `closing`, so that much more of the
    // field has to be exact.
    #[allow(non_snake_case)]
    let BAND = MARGIN_VOXELS + (closing / voxel).ceil() as i32;
    let padding = voxel * (BAND as f32 + 2.0);
    let origin = low - Vec3::splat(padding);
    let dims = ((high - low + Vec3::splat(2.0 * padding)) / voxel).ceil().as_uvec3() + 1;
    let (nx, ny, nz) = (dims.x as usize, dims.y as usize, dims.z as usize);
    let to_grid = |p: Vec3| (p - origin) / voxel;
    // Grid cells `lo..=hi` grown by the band, clipped to `0..n`; `None` when
    // they miss the grid (a triangle outside a region).
    let cells = |lo: f32, hi: f32, n: usize| -> Option<(usize, usize)> {
        let (from, to) = (lo.floor() as i32 - BAND, hi.ceil() as i32 + BAND);
        (to >= 0 && from < n as i32).then(|| (from.max(0) as usize, (to as usize).min(n - 1)))
    };

    // Triangles sorted into the z-slices they can influence, so slices can
    // be filled independently.
    let mut by_slice: Vec<Vec<u32>> = vec![Vec::new(); nz];
    for (index, t) in skin.iter().enumerate() {
        let z = t.map(|i| to_grid(positions[i as usize]).z);
        let Some((from, to)) = cells(z[0].min(z[1]).min(z[2]), z[0].max(z[1]).max(z[2]), nz) else { continue };
        for slice in &mut by_slice[from..=to] {
            slice.push(index as u32);
        }
    }

    let far = voxel * BAND as f32;
    let mut values = vec![f32::NAN; nx * ny * nz]; // NaN = not near the surface
    let mut velocity = vec![Vec3::ZERO; if velocities.is_some() { nx * ny * nz } else { 0 }];
    let mut velocity_slices: Vec<&mut [Vec3]> = if velocities.is_some() { velocity.chunks_mut(nx * ny).collect() } else { (0..nz).map(|_| &mut [][..]).collect() };
    values.par_chunks_mut(nx * ny).zip(velocity_slices.par_iter_mut()).zip(by_slice).enumerate().for_each(|(z, ((slice, velocity_slice), triangles))| {
        // Per voxel: squared distance and how squarely the closest
        // triangle faces the voxel (to break ties at shared edges), and the
        // skin's velocity there.
        let mut best = vec![(f32::MAX, 0.0f32); nx * ny];
        let mut moving = vec![Vec3::ZERO; if velocities.is_some() { nx * ny } else { 0 }];
        for index in triangles {
            let corners = skin[index as usize];
            let [a, b, c] = corners.map(|i| positions[i as usize]);
            let normal = (b - a).cross(c - a).normalize_or_zero();
            let (ga, gb, gc) = (to_grid(a), to_grid(b), to_grid(c));
            let (lo, hi) = (ga.min(gb).min(gc), ga.max(gb).max(gc));
            let (Some((x0, x1)), Some((y0, y1))) = (cells(lo.x, hi.x, nx), cells(lo.y, hi.y, ny)) else { continue };
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let p = origin + Vec3::new(x as f32, y as f32, z as f32) * voxel;
                    let closest = closest_on_triangle(p, a, b, c);
                    let offset = p - closest;
                    let d2 = offset.length_squared();
                    if d2 > far * far {
                        continue;
                    }
                    let facing = offset.dot(normal);
                    let entry = &mut best[x + nx * y];
                    // On an edge two triangles are equally close; the one
                    // facing the voxel more squarely knows the right side.
                    let tie = (d2 - entry.0).abs() <= 1e-10 && facing.abs() > entry.1.abs();
                    if d2 < entry.0 - 1e-10 || tie {
                        *entry = (d2.min(entry.0), facing);
                        if let Some(velocities) = velocities {
                            let [u, v, w] = barycentric(closest, a, b, c);
                            let [va, vb, vc] = corners.map(|i| velocities.get(i as usize).copied().unwrap_or(Vec3::ZERO));
                            moving[x + nx * y] = va * u + vb * v + vc * w;
                        }
                    }
                }
            }
        }
        for (value, (d2, facing)) in slice.iter_mut().zip(best) {
            if d2 < f32::MAX {
                *value = d2.sqrt().copysign(facing);
            }
        }
        velocity_slice.copy_from_slice(&moving);
    });
    drop(velocity_slices);

    // Everything further away only needs the right sign: walk each row
    // from outside the body and carry the sign of the last near-surface
    // sample.
    let near_surface: Vec<bool> = values.iter().map(|v| !v.is_nan()).collect();
    values.par_chunks_mut(nx).for_each(|row| {
        let mut sign = 1.0f32;
        for value in row {
            if value.is_nan() {
                *value = far * sign;
            } else {
                sign = 1.0f32.copysign(*value);
            }
        }
    });
    // Deep inside the body the field must still say which way is out, or
    // cloth that is pulled in further than the exact band (a garment much
    // too tight, a seam closing through a limb) would stay there. Outside,
    // a constant is fine: nothing needs pushing.
    if deep {
        let depth = depth_inside(&values, |i| near_surface[i], [nx, ny, nz], voxel);
        values.par_iter_mut().zip(depth).for_each(|(value, depth)| {
            if *value < 0.0 {
                *value = -depth;
            }
        });
    }

    if closing > 0.0 {
        // Closing = grow by the radius, then shrink by it. Growing is a
        // plain offset of the distances; shrinking needs the true depth
        // below the grown surface, which is where creases get filled.
        let grown: Vec<f32> = values.iter().map(|value| value - closing).collect();
        let depth = depth_inside(&grown, |_| false, [nx, ny, nz], voxel);
        let limit = voxel * MARGIN_VOXELS as f32;
        values.par_iter_mut().zip(grown).zip(depth).for_each(|((value, grown), depth)| {
            // The sweep is only first-order accurate; taking the smaller of
            // the two distances guarantees the closed body never ends up
            // inside the exact one.
            let closed = if grown < 0.0 { closing - depth } else { *value };
            *value = closed.min(*value).min(limit);
        });
    }

    DistanceField { origin, voxel, dims: dims.to_array(), values, velocity }
}

/// Weights of the corners `a`, `b`, `c` for a point `p` on their triangle.
fn barycentric(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> [f32; 3] {
    let (v0, v1, v2) = (b - a, c - a, p - a);
    let (d00, d01, d11, d20, d21) = (v0.dot(v0), v0.dot(v1), v1.dot(v1), v2.dot(v0), v2.dot(v1));
    let denominator = d00 * d11 - d01 * d01;
    if denominator.abs() < 1e-20 {
        return [1.0, 0.0, 0.0];
    }
    let v = (d11 * d20 - d01 * d21) / denominator;
    let w = (d00 * d21 - d01 * d20) / denominator;
    [1.0 - v - w, v, w]
}

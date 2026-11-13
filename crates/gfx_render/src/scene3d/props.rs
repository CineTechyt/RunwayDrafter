//! Buttons and buttonholes on the draped garment.
//!
//! Each button is a small lathe-turned model (rim, dished face, thread
//! holes with the thread crossing them, or a dome, or a riveted tack)
//! built in its own frame and then stood on the cloth: the frame follows
//! the triangle of the garment mesh under the button, so buttons ride
//! along with every fold of the drape. A done-up button sits on its
//! buttonhole, on top of the piece that laps over.

use std::f32::consts::TAU;

use cad_core::{Button, ButtonFinish, ButtonStyle, Buttonhole, PatternDocument, PieceId};
use glam::{Mat3, Vec2, Vec3};
use topology_sync::{PatternLocation, TopologySync};

use super::mesh::{MeshData, MeshVertex};

const MM: f32 = 1e-3;
/// Gap between the cloth (drawn with no thickness) and what sits on it.
const LIFT: f32 = 0.9 * MM;

/// A frame on the draped cloth: origin on the surface, `z` the outward
/// normal, `x` along the pattern's +X.
#[derive(Debug, Clone, Copy)]
struct ClothFrame {
    origin: Vec3,
    axes: Mat3,
}

impl ClothFrame {
    fn at(topology: &impl TopologySync, piece: PieceId, position: Vec2, angle: f32) -> Option<Self> {
        let location = topology.pattern_to_surface(PatternLocation { piece, position })?;
        let mesh = topology.mesh();
        let [a, b, c] = mesh.triangle(location.triangle).map(|v| v as usize);
        let w = location.barycentric;
        let origin = mesh.positions[a] * w.x + mesh.positions[b] * w.y + mesh.positions[c] * w.z;
        let normal = (mesh.normals[a] * w.x + mesh.normals[b] * w.y + mesh.normals[c] * w.z).try_normalize()?;
        // Where pattern +X (turned by `angle`) went: J = [e1 e2] [f1 f2]⁻¹.
        let (e1, e2) = (mesh.positions[b] - mesh.positions[a], mesh.positions[c] - mesh.positions[a]);
        let (f1, f2) = (
            mesh.pattern_positions[b] - mesh.pattern_positions[a],
            mesh.pattern_positions[c] - mesh.pattern_positions[a],
        );
        let det = f1.perp_dot(f2);
        let d = Vec2::from_angle(angle);
        let tangent = if det.abs() > 1e-9 {
            // Solve f1·s + f2·t = d.
            let (s, t) = (d.perp_dot(f2) / det, f1.perp_dot(d) / det);
            e1 * s + e2 * t
        } else {
            e1
        };
        let x = (tangent - normal * tangent.dot(normal)).try_normalize().unwrap_or_else(|| normal.any_orthonormal_vector());
        Some(Self { origin, axes: Mat3::from_cols(x, normal.cross(x), normal) })
    }

    fn place(&self, mesh: &mut MeshData, model: &MeshData) {
        let base = mesh.vertices.len() as u32;
        mesh.vertices.extend(model.vertices.iter().map(|v| MeshVertex {
            position: (self.origin + self.axes * Vec3::from(v.position)).to_array(),
            normal: (self.axes * Vec3::from(v.normal)).to_array(),
            ..*v
        }));
        mesh.indices.extend(model.indices.iter().map(|i| base + i));
    }
}

/// Every button and buttonhole of `document` standing on the draped
/// garment, in world space.
pub fn garment_props(document: &PatternDocument, topology: &impl TopologySync) -> MeshData {
    let mut out = MeshData::default();
    // Buttons that are done up sit on their hole instead of where they
    // are sewn: on top of the overlapping piece.
    for piece in document.pieces() {
        let fabric = document.fabric(piece.fabric).map_or([0.5; 3], |f| f.base_color);
        for hole in &piece.buttonholes {
            if let Some(frame) = ClothFrame::at(topology, piece.id, hole.position, hole.angle) {
                frame.place(&mut out, &buttonhole_model(hole, fabric));
            }
        }
        for button in &piece.buttons {
            let on_hole = button
                .fastened_to
                .and_then(|r| Some((r.piece, document.buttonhole(r)?)))
                .and_then(|(hole_piece, hole)| ClothFrame::at(topology, hole_piece, hole.position, 0.0));
            let frame = on_hole.or_else(|| ClothFrame::at(topology, piece.id, button.position, 0.0));
            if let Some(mut frame) = frame {
                // Above the buttonhole's stitching when done up.
                frame.origin += frame.axes.z_axis * if on_hole.is_some() { 0.6 * MM } else { 0.0 };
                frame.place(&mut out, &button_model(button));
            }
        }
    }
    out
}

/// A wooden box standing on the floor from `min` to `max`: the seat of
/// the sitting animation. The top is a slightly lighter, worn surface.
pub fn chair(mesh: &mut MeshData, min: Vec3, max: Vec3) {
    if max.y - min.y < 1e-4 {
        return;
    }
    let wood = [0.23, 0.12, 0.055];
    let top = [0.30, 0.17, 0.08];
    let material = (0.55, 0.0);
    let corner = |x: bool, y: bool, z: bool| Vec3::new(if x { max.x } else { min.x }, if y { max.y } else { min.y }, if z { max.z } else { min.z });
    // (normal, four corners counter-clockwise from outside, colour)
    let faces: [(Vec3, [Vec3; 4], [f32; 3]); 5] = [
        (Vec3::Y, [corner(false, true, false), corner(false, true, true), corner(true, true, true), corner(true, true, false)], top),
        (Vec3::Z, [corner(false, false, true), corner(true, false, true), corner(true, true, true), corner(false, true, true)], wood),
        (Vec3::NEG_Z, [corner(true, false, false), corner(false, false, false), corner(false, true, false), corner(true, true, false)], wood),
        (Vec3::X, [corner(true, false, true), corner(true, false, false), corner(true, true, false), corner(true, true, true)], wood),
        (Vec3::NEG_X, [corner(false, false, false), corner(false, false, true), corner(false, true, true), corner(false, true, false)], wood),
    ];
    for (normal, quad, color) in faces {
        let base = mesh.vertices.len() as u32;
        mesh.vertices.extend(quad.iter().map(|&p| vertex(p, normal, color, material)));
        mesh.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Vertex with a per-vertex material: `uv` = (roughness, metallic).
fn vertex(position: Vec3, normal: Vec3, color: [f32; 3], material: (f32, f32)) -> MeshVertex {
    MeshVertex::new(position, normal, Vec2::new(material.0, material.1)).with_color(color)
}

/// Turns a profile around the Z axis. `profile` runs from the axis on top
/// to the underside as `(radius, height)`; a repeated point makes a sharp
/// edge. `tint` may vary the colour by angle and radius (wood grain,
/// nacre).
fn lathe(
    mesh: &mut MeshData,
    profile: &[(f32, f32)],
    segments: u32,
    material: (f32, f32),
    tint: impl Fn(f32, f32) -> [f32; 3],
) {
    // Normals per profile point from its neighbours, except across a
    // repeated point (a crease), where each side keeps its own.
    let n = profile.len();
    let normal_2d = |i: usize| -> Vec2 {
        let p = Vec2::from(profile[i]);
        let prev = (i > 0).then(|| Vec2::from(profile[i - 1])).filter(|q| q.distance(p) > 1e-9);
        let next = (i + 1 < n).then(|| Vec2::from(profile[i + 1])).filter(|q| q.distance(p) > 1e-9);
        let tangent = match (prev, next) {
            (Some(a), Some(b)) => b - a,
            (Some(a), None) => p - a,
            (None, Some(b)) => b - p,
            (None, None) => Vec2::X,
        };
        // The profile runs outwards over the top and back in underneath,
        // so the outward normal is the tangent turned counter-clockwise.
        Vec2::new(-tangent.y, tangent.x).normalize_or(Vec2::Y)
    };
    for i in 0..n.saturating_sub(1) {
        let (p0, p1) = (Vec2::from(profile[i]), Vec2::from(profile[i + 1]));
        if p0.distance(p1) < 1e-9 {
            continue;
        }
        // Normals of this band: smooth unless the next/previous point is a crease.
        let (n0, n1) = {
            let crease_before = i > 0 && Vec2::from(profile[i - 1]).distance(p0) < 1e-9;
            let crease_after = i + 2 < n && Vec2::from(profile[i + 2]).distance(p1) < 1e-9;
            let flat = { let t = p1 - p0; Vec2::new(-t.y, t.x).normalize_or(Vec2::Y) };
            (if crease_before || i == 0 && p0.x > 1e-6 { flat } else { normal_2d(i) }, if crease_after { flat } else { normal_2d(i + 1) })
        };
        let base = mesh.vertices.len() as u32;
        for s in 0..=segments {
            let angle = s as f32 / segments as f32 * TAU;
            let (sin, cos) = angle.sin_cos();
            for (p, normal) in [(p0, n0), (p1, n1)] {
                let position = Vec3::new(p.x * cos, p.x * sin, p.y);
                let normal = Vec3::new(normal.x * cos, normal.x * sin, normal.y);
                mesh.vertices.push(vertex(position, normal, tint(angle, p.x), material));
            }
        }
        for s in 0..segments {
            let a = base + 2 * s;
            // Counter-clockwise seen from outside.
            mesh.indices.extend([a, a + 1, a + 2, a + 1, a + 3, a + 2]);
        }
    }
}

/// A small box from `from` to `to` (in the model frame), `width` wide
/// and `height` tall, standing on its lower face.
fn bar(mesh: &mut MeshData, from: Vec3, to: Vec3, width: f32, height: f32, color: [f32; 3], material: (f32, f32)) {
    let along = (to - from).normalize_or(Vec3::X);
    let up = Vec3::Z;
    let side = up.cross(along).normalize_or(Vec3::Y) * (width * 0.5);
    let lift = up * height;
    let corners = [from - side, from + side, to + side, to - side];
    let faces: [([Vec3; 4], Vec3); 5] = [
        ([corners[0] + lift, corners[1] + lift, corners[2] + lift, corners[3] + lift], up),
        ([corners[0], corners[3], corners[3] + lift, corners[0] + lift], -side.normalize()),
        ([corners[1], corners[1] + lift, corners[2] + lift, corners[2]], side.normalize()),
        ([corners[0], corners[0] + lift, corners[1] + lift, corners[1]], -along),
        ([corners[3], corners[2], corners[2] + lift, corners[3] + lift], along),
    ];
    for (quad, normal) in faces {
        let base = mesh.vertices.len() as u32;
        mesh.vertices.extend(quad.iter().map(|&p| vertex(p, normal, color, material)));
        // Counter-clockwise seen from outside, whichever way the corners came.
        if (quad[1] - quad[0]).cross(quad[2] - quad[0]).dot(normal) >= 0.0 {
            mesh.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        } else {
            mesh.indices.extend([base, base + 2, base + 1, base, base + 3, base + 2]);
        }
    }
}

/// A flat disc facing up at height `z`.
fn disc(mesh: &mut MeshData, centre: Vec3, radius: f32, segments: u32, color: [f32; 3], material: (f32, f32)) {
    let base = mesh.vertices.len() as u32;
    mesh.vertices.push(vertex(centre, Vec3::Z, color, material));
    for s in 0..segments {
        let (sin, cos) = (s as f32 / segments as f32 * TAU).sin_cos();
        mesh.vertices.push(vertex(centre + Vec3::new(cos, sin, 0.0) * radius, Vec3::Z, color, material));
    }
    for s in 0..segments {
        mesh.indices.extend([base, base + 1 + s, base + 1 + (s + 1) % segments]);
    }
}

/// One button in its own frame: metres, standing on z = 0, facing +Z.
pub fn button_model(button: &Button) -> MeshData {
    let mut mesh = MeshData::default();
    let r = button.diameter * 0.5 * MM;
    let t = button.thickness() * MM;
    let material = button.finish.shading();
    let base = button.color;
    let tint = |angle: f32, radius: f32| -> [f32; 3] {
        match button.finish {
            // Growth rings across the face.
            ButtonFinish::Wood => {
                let ring = 0.82 + 0.18 * ((radius / r * 7.0 + (angle * 3.0).sin() * 0.25) * TAU).sin();
                base.map(|c| c * ring)
            }
            // Nacre: a slow shimmer of pink and green.
            ButtonFinish::Pearl => {
                let shift = (angle * 3.0 + radius / r * 4.0).sin();
                [base[0] * (1.0 + 0.06 * shift), base[1] * (1.0 - 0.02 * shift), base[2] * (1.0 + 0.05 * -shift)]
            }
            ButtonFinish::Plastic | ButtonFinish::Metal => base,
        }
    };
    let segments = 40;
    match button.style {
        ButtonStyle::FourHole | ButtonStyle::TwoHole => {
            let dish = 0.62 * t;
            lathe(
                &mut mesh,
                &[
                    (0.0, dish),
                    (0.55 * r, dish + 0.05 * t),
                    (0.74 * r, 0.9 * t),
                    (0.84 * r, t),
                    (0.95 * r, 0.9 * t),
                    (r, 0.6 * t),
                    (0.97 * r, 0.12 * t),
                    (0.9 * r, 0.0),
                    (0.9 * r, 0.0),
                    (0.0, 0.0),
                ],
                segments,
                material,
                tint,
            );
            // Holes: dark wells in the dish, the thread crossing between them.
            let holes = button.style.holes();
            let spread = 0.24 * r;
            let positions: Vec<Vec3> = (0..holes)
                .map(|k| {
                    let angle = if holes == 4 { TAU / 8.0 + k as f32 * TAU / 4.0 } else { k as f32 * TAU / 2.0 };
                    Vec3::new(angle.cos() * spread, angle.sin() * spread, dish + 0.03 * t)
                })
                .collect();
            let hole_radius = 0.085 * r;
            for &p in &positions {
                disc(&mut mesh, p, hole_radius, 12, [0.01; 3], (0.9, 0.0));
            }
            let thread_material = (0.85, 0.0);
            let pairs: &[(usize, usize)] = if holes == 4 { &[(0, 2), (1, 3)] } else { &[(0, 1)] };
            for (pair, &(a, b)) in pairs.iter().enumerate() {
                // The second pass of thread lies over the first.
                let lift = Vec3::Z * (0.04 * t + pair as f32 * 0.05 * t);
                bar(&mut mesh, positions[a] + lift, positions[b] + lift, 0.9 * hole_radius, 0.07 * t, button.thread_color, thread_material);
            }
        }
        ButtonStyle::Shank => {
            let head = 0.32 * t;
            lathe(
                &mut mesh,
                &[
                    (0.0, t),
                    (0.45 * r, 0.93 * t),
                    (0.8 * r, 0.72 * t),
                    (r, 0.48 * t),
                    (0.96 * r, head),
                    (0.96 * r, head),
                    (0.0, head),
                ],
                segments,
                material,
                tint,
            );
            // The shank: a short post down to the cloth.
            lathe(&mut mesh, &[(0.0, head), (0.22 * r, head), (0.22 * r, head), (0.18 * r, 0.0), (0.0, 0.0)], 16, material, tint);
        }
        ButtonStyle::Tack => {
            lathe(
                &mut mesh,
                &[
                    (0.0, 0.9 * t),
                    (0.14 * r, 0.86 * t),
                    (0.22 * r, 0.97 * t),
                    (0.7 * r, t),
                    (0.9 * r, 0.93 * t),
                    (r, 0.72 * t),
                    (0.98 * r, 0.5 * t),
                    (0.86 * r, 0.42 * t),
                    (0.86 * r, 0.42 * t),
                    (0.5 * r, 0.4 * t),
                    (0.45 * r, 0.0),
                    (0.0, 0.0),
                ],
                segments,
                material,
                tint,
            );
        }
    }
    for v in &mut mesh.vertices {
        v.position[2] += LIFT;
    }
    mesh
}

/// A buttonhole in its own frame, along +X: two rows of satin stitch, a
/// bar tack across each end and the dark cut between them. `fabric` is
/// the colour the thread is matched to.
pub fn buttonhole_model(hole: &Buttonhole, fabric: [f32; 3]) -> MeshData {
    let mut mesh = MeshData::default();
    let half = hole.length * 0.5 * MM;
    let bead = 1.3 * MM;
    let height = 0.5 * MM;
    let thread = fabric.map(|c| c * 0.8);
    let material = (0.75, 0.0);
    for side in [-1.0, 1.0] {
        let y = Vec3::Y * side * (0.5 * bead + 0.25 * MM);
        bar(&mut mesh, Vec3::new(-half, 0.0, 0.0) + y, Vec3::new(half, 0.0, 0.0) + y, bead, height, thread, material);
    }
    for end in [-1.0, 1.0] {
        let x = Vec3::X * end * (half + 0.4 * MM);
        bar(&mut mesh, x - Vec3::Y * (bead + 0.4 * MM), x + Vec3::Y * (bead + 0.4 * MM), 0.9 * MM, height * 1.25, thread, material);
    }
    // The cut, just above the cloth.
    bar(&mut mesh, Vec3::new(-half, 0.0, 0.0), Vec3::new(half, 0.0, 0.0), 0.5 * MM, 0.1 * MM, [0.01; 3], (0.9, 0.0));
    for v in &mut mesh.vertices {
        v.position[2] += 0.2 * MM;
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_style_builds_a_closed_up_model_above_the_cloth() {
        for style in ButtonStyle::ALL {
            let model = button_model(&Button { style, ..Button::new(Vec2::ZERO, 20.0) });
            assert!(model.indices.len() > 100, "{style:?}");
            assert!(model.indices.iter().all(|&i| (i as usize) < model.vertices.len()));
            let (low, high) = model.vertices.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(v.position[2]), hi.max(v.position[2])));
            assert!(low >= 0.0 && high < 0.012, "{style:?}: {low}..{high}");
            let reach = model.vertices.iter().map(|v| Vec2::new(v.position[0], v.position[1]).length()).fold(0.0, f32::max);
            assert!((reach - 0.010).abs() < 1e-4, "{style:?}: radius {reach}");
        }
    }

    #[test]
    fn buttons_ride_on_the_draped_garment() {
        let mut doc = PatternDocument::demo_waistcoat();
        let mut topology = topology_sync::GarmentTopology::default();
        topology.rebuild(&doc);
        let props = garment_props(&doc, &topology);
        assert!(!props.indices.is_empty());
        // Move the whole garment: the props move with it.
        let before = props.vertices[0].position;
        for p in &mut topology.mesh_mut().positions {
            *p += Vec3::new(0.0, 0.1, 0.0);
        }
        let after = garment_props(&doc, &topology).vertices[0].position;
        assert!((after[1] - before[1] - 0.1).abs() < 1e-5);
        // Undone buttons sit where they are sewn on.
        doc.unfasten_buttons();
        assert!(!garment_props(&doc, &topology).indices.is_empty());
    }
}

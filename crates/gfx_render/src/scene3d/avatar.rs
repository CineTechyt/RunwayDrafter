//! Placeholder avatar.
//!
//! A real avatar is a skinned, scanned or parametric body mesh loaded from
//! disk (glTF/FBX/OBJ). Until an importer exists, this builds a mannequin
//! from ellipsoids. The same ellipsoids double as collision shapes for the
//! cloth solver, so what you see is exactly what the fabric rests on.

use glam::{Quat, Vec3};

use super::mesh::MeshData;

/// One ellipsoid of the mannequin, in world space (metres, Y up).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyPart {
    pub center: Vec3,
    pub radii: Vec3,
    pub rotation: Quat,
}

/// A roughly 1.72 m tall mannequin standing on the origin, facing +Z, in an
/// A-pose: the arms are held away from the torso so side seams can close.
pub fn mannequin_parts() -> Vec<BodyPart> {
    let mut parts = Vec::new();
    let mut part = |center: [f32; 3], radii: [f32; 3], roll: f32| {
        parts.push(BodyPart { center: center.into(), radii: radii.into(), rotation: Quat::from_rotation_z(roll) });
    };

    part([0.0, 1.60, 0.0], [0.095, 0.118, 0.105], 0.0); // head
    part([0.0, 1.47, 0.0], [0.050, 0.075, 0.050], 0.0); // neck
    part([0.0, 1.27, 0.0], [0.170, 0.185, 0.115], 0.0); // chest
    part([0.0, 1.07, 0.0], [0.145, 0.150, 0.105], 0.0); // abdomen
    part([0.0, 0.93, 0.0], [0.168, 0.130, 0.118], 0.0); // hips

    let abduction = 55f32.to_radians();
    for side in [-1.0f32, 1.0] {
        // Shoulders sit below the neck base so the shoulder line slopes
        // down and outwards, as garment patterns assume.
        part([side * 0.190, 1.345, 0.0], [0.066, 0.055, 0.060], 0.0); // shoulder
        // Arm segments hang from the shoulder joint along `down`.
        let joint = Vec3::new(side * 0.205, 1.335, 0.0);
        let down = Vec3::new(side * abduction.sin(), -abduction.cos(), 0.0);
        let roll = side * abduction;
        let mut limb = |distance: f32, radii: [f32; 3]| part((joint + down * distance).into(), radii, roll);
        limb(0.170, [0.046, 0.175, 0.048]); // upper arm
        limb(0.480, [0.038, 0.160, 0.040]); // forearm
        limb(0.680, [0.028, 0.062, 0.020]); // hand

        part([side * 0.090, 0.660, 0.0], [0.078, 0.245, 0.082], 0.0); // thigh
        part([side * 0.090, 0.270, 0.0], [0.052, 0.235, 0.056], 0.0); // shin
        part([side * 0.090, 0.035, 0.05], [0.045, 0.035, 0.110], 0.0); // foot
    }
    parts
}

/// Render mesh of [`mannequin_parts`].
pub fn mannequin() -> MeshData {
    let mut mesh = MeshData::default();
    for part in mannequin_parts() {
        mesh.add_ellipsoid(part.center, part.radii, part.rotation, 20);
    }
    mesh
}

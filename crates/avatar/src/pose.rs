//! Posing: a skeleton pose and linear blend skinning.
//!
//! A [`Pose`] is a rotation per bone, expressed in the axes of the modelled
//! (rest) pose and applied about the bone's head joint; children inherit
//! their parents' motion. Any pose can be built that way: [`PoseParams`]
//! covers the handful of angles garment work needs and is what the UI
//! edits, and [`Pose::set`] takes arbitrary bones for anything else.

use std::collections::HashMap;

use glam::{Affine3A, Quat, Vec3};

use crate::model::{BodyMesh, BodyModel};

/// Rotations by bone name. Bones not mentioned keep their rest rotation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Pose {
    rotations: HashMap<String, Quat>,
}

impl Pose {
    /// The pose the body was modelled in.
    pub fn rest() -> Self {
        Self::default()
    }

    /// Rotates `bone` (and everything attached to it) about its head joint,
    /// on top of any rotation already set for it.
    pub fn rotate(&mut self, bone: &str, rotation: Quat) {
        let entry = self.rotations.entry(bone.to_owned()).or_insert(Quat::IDENTITY);
        *entry = (rotation * *entry).normalize();
    }

    pub fn set(&mut self, bone: &str, rotation: Quat) {
        self.rotations.insert(bone.to_owned(), rotation);
    }

    pub fn get(&self, bone: &str) -> Quat {
        self.rotations.get(bone).copied().unwrap_or(Quat::IDENTITY)
    }

    pub fn is_rest(&self) -> bool {
        self.rotations.values().all(|q| q.abs_diff_eq(Quat::IDENTITY, 1e-6))
    }
}

/// The pose controls of the avatar editor, in degrees. Both sides move
/// symmetrically.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoseParams {
    /// Angle between each arm and the vertical: 0 hangs straight down,
    /// 90 is a T-pose. `None` keeps the angle the body was modelled with.
    pub arm_angle: Option<f32>,
    /// Swings the arms forward (positive) or back.
    pub arm_forward: f32,
    pub elbow_bend: f32,
    /// Angle between each leg and the vertical. `None` keeps the modelled
    /// stance.
    pub leg_angle: Option<f32>,
    pub knee_bend: f32,
}

impl PoseParams {
    /// Named starting points, in the order the UI lists them.
    pub const PRESETS: [(&'static str, PoseParams); 5] = [
        ("A-pose", PoseParams::a_pose()),
        ("T-pose", PoseParams { arm_angle: Some(90.0), ..PoseParams::modelled() }),
        ("Arms down", PoseParams { arm_angle: Some(14.0), ..PoseParams::modelled() }),
        ("Wide stance", PoseParams { leg_angle: Some(14.0), ..PoseParams::a_pose() }),
        ("As modelled", PoseParams::modelled()),
    ];

    /// The pose of the base mesh.
    pub const fn modelled() -> Self {
        Self { arm_angle: None, arm_forward: 0.0, elbow_bend: 0.0, leg_angle: None, knee_bend: 0.0 }
    }

    /// Arms at 45°: clear of the torso, so side seams can be sewn, without
    /// stressing the shoulders the way a T-pose does.
    pub const fn a_pose() -> Self {
        Self { arm_angle: Some(45.0), ..Self::modelled() }
    }

    /// The arm and leg angles (degrees from the vertical) the body was
    /// modelled with, i.e. what `None` stands for.
    pub fn modelled_angles(model: &BodyModel, rest: &BodyMesh) -> (f32, f32) {
        let angle = |upper: &str, lower: &str| {
            let (Some(a), Some(b)) = (model.bone(upper), model.bone(lower)) else { return 0.0 };
            let along = (rest.joints[b].head - rest.joints[a].head).normalize_or(Vec3::NEG_Y);
            along.x.abs().atan2(-along.y).to_degrees()
        };
        (angle("upperarm01.L", "lowerarm01.L"), angle("upperleg01.L", "lowerleg01.L"))
    }

    /// The bone rotations for these angles on a particular body (`rest` is
    /// that body in the modelled pose; its joints define the limb axes).
    pub fn to_pose(&self, model: &BodyModel, rest: &BodyMesh) -> Pose {
        let mut pose = Pose::rest();
        let joint = |name: &str| model.bone(name).map(|i| rest.joints[i]);
        for (suffix, side) in [("L", 1.0f32), ("R", -1.0)] {
            let bone = |stem: &str| format!("{stem}.{suffix}");

            // --- arm -------------------------------------------------------
            if let (Some(upper), Some(lower)) = (joint(&bone("upperarm01")), joint(&bone("lowerarm01"))) {
                let along = (lower.head - upper.head).normalize_or(Vec3::NEG_Y);
                // Raising happens about the front-back axis. The left arm
                // points towards +x, so it rises counter-clockwise.
                let modelled = along.x.abs().atan2(-along.y).to_degrees();
                let raise = (self.arm_angle.unwrap_or(modelled) - modelled).to_radians() * side;
                // A raise comes partly from the shoulder girdle; spreading
                // it over three bones keeps the skin from collapsing.
                for (stem, share) in [("clavicle", 0.12), ("shoulder01", 0.18), ("upperarm01", 0.70)] {
                    pose.rotate(&bone(stem), Quat::from_rotation_z(raise * share));
                }
                // Forward swing about the vertical through the shoulder.
                pose.rotate(&bone("upperarm01"), Quat::from_rotation_y(-self.arm_forward.to_radians() * side));
                // The elbow folds the forearm towards the front.
                if let Some(wrist) = joint(&bone("wrist")) {
                    let forearm = (wrist.head - lower.head).normalize_or(along);
                    let axis = forearm.cross(Vec3::Z).normalize_or(Vec3::X);
                    pose.rotate(&bone("lowerarm01"), Quat::from_axis_angle(axis, self.elbow_bend.to_radians()));
                }
            }

            // --- leg ---------------------------------------------------------
            if let (Some(upper), Some(lower)) = (joint(&bone("upperleg01")), joint(&bone("lowerleg01"))) {
                let along = (lower.head - upper.head).normalize_or(Vec3::NEG_Y);
                let modelled = (along.x * side).atan2(-along.y).to_degrees();
                let spread = (self.leg_angle.unwrap_or(modelled) - modelled).to_radians() * side;
                pose.rotate(&bone("upperleg01"), Quat::from_rotation_z(spread));
                // The knee folds the lower leg backwards.
                let axis = along.cross(Vec3::NEG_Z).normalize_or(Vec3::X);
                pose.rotate(&bone("lowerleg01"), Quat::from_axis_angle(axis, self.knee_bend.to_radians()));
            }
        }
        pose
    }
}

impl Default for PoseParams {
    fn default() -> Self {
        Self::a_pose()
    }
}

impl BodyModel {
    /// Applies `pose` to a body in the modelled pose (as returned by
    /// [`BodyModel::build`]) with linear blend skinning, and puts the
    /// result back on the floor.
    pub fn pose(&self, rest: &BodyMesh, pose: &Pose) -> BodyMesh {
        self.pose_with_transforms(rest, pose).0
    }

    /// Like [`BodyModel::pose`], and also returns the transform that took
    /// each vertex from the modelled pose to the posed one. Those are what
    /// lets something attached to the skin (a garment) follow it.
    pub fn pose_with_transforms(&self, rest: &BodyMesh, pose: &Pose) -> (BodyMesh, Vec<Affine3A>) {
        if pose.is_rest() || self.bones.is_empty() {
            return (rest.clone(), vec![Affine3A::IDENTITY; rest.positions.len()]);
        }
        // World transform of every bone: its parent's, then its own
        // rotation about its head joint.
        let mut world: Vec<Affine3A> = Vec::with_capacity(self.bones.len());
        for (index, bone) in self.bones.iter().enumerate() {
            let head = rest.joints[index].head;
            let local = Affine3A::from_translation(head) * Affine3A::from_quat(pose.get(&bone.name)) * Affine3A::from_translation(-head);
            world.push(match bone.parent {
                Some(parent) => world[parent] * local,
                None => local,
            });
        }

        // Linear blend skinning: each vertex moves by the weighted sum of
        // its bones' transforms.
        let mut transforms: Vec<Affine3A> = self
            .weights
            .iter()
            .map(|(bones, weights)| {
                let mut matrix = glam::Mat3A::ZERO;
                let mut translation = glam::Vec3A::ZERO;
                for k in 0..4 {
                    if weights[k] > 0.0 {
                        let bone = world[bones[k] as usize];
                        matrix += bone.matrix3 * weights[k];
                        translation += bone.translation * weights[k];
                    }
                }
                Affine3A { matrix3: matrix, translation }
            })
            .collect();

        let mut mesh = rest.clone();
        for ((position, &rest_position), transform) in mesh.positions.iter_mut().zip(&rest.positions).zip(&transforms) {
            *position = transform.transform_point3(rest_position);
        }
        for (joint, transform) in mesh.joints.iter_mut().zip(&world) {
            joint.head = transform.transform_point3(joint.head);
            joint.tail = transform.transform_point3(joint.tail);
        }

        // Spreading or bending the legs lifts the feet: stand again.
        let floor = mesh.positions.iter().map(|p| p.y).fold(f32::MAX, f32::min);
        for p in &mut mesh.positions {
            p.y -= floor;
        }
        for joint in &mut mesh.joints {
            joint.head.y -= floor;
            joint.tail.y -= floor;
        }
        for transform in &mut transforms {
            transform.translation.y -= floor;
        }
        mesh.recompute_normals();
        (mesh, transforms)
    }
}

/// A body as it stands: the modelled pose it was built in, the posed
/// result and the per-vertex transforms between the two.
#[derive(Debug, Clone, Copy)]
pub struct PosedBody<'a> {
    pub rest: &'a BodyMesh,
    pub posed: &'a BodyMesh,
    pub transforms: &'a [Affine3A],
}

/// Moves `points` (garment vertices, say) from where they are around the
/// body `from` to the corresponding place around `to`: the same body in
/// another pose, or reshaped.
///
/// Each point is attached to its nearest skin vertex and keeps its offset
/// from it, expressed in the modelled pose, so it follows that piece of
/// skin through both the pose change and any change of shape. Run the
/// cloth simulation afterwards to settle the result.
pub fn carry(points: &mut [Vec3], from: PosedBody<'_>, to: PosedBody<'_>) {
    use rayon::prelude::*;
    let count = from.posed.positions.len().min(to.posed.positions.len());
    if count == 0 {
        return;
    }
    points.par_iter_mut().for_each(|point| {
        let nearest = (0..count)
            .min_by(|&a, &b| from.posed.positions[a].distance_squared(*point).total_cmp(&from.posed.positions[b].distance_squared(*point)))
            .unwrap_or(0);
        let offset = from.transforms[nearest].inverse().transform_point3(*point) - from.rest.positions[nearest];
        *point = to.transforms[nearest].transform_point3(to.rest.positions[nearest] + offset);
    });
}

//! Animations of the avatar: walking in three styles, sitting down on a
//! chair, and a yoga flow.
//!
//! Motion is written as **joint angles** ([`Angles`]): how far the hips
//! flex, the knees bend, the arms rise, the spine bends and twists, the
//! way a physiotherapist would describe it. [`Angles::to_pose`] turns them
//! into bone rotations for a particular body.
//!
//! What makes it look natural:
//!
//! * **Smooth curves through keyframes.** Every angle follows a monotone
//!   cubic through its keys: motion flows on through a key instead of
//!   stopping at it, and never overshoots a held pose.
//! * **Follow-through.** Elbows, wrists and the head trail the movement
//!   that drives them by a fraction of a second.
//! * **Breathing.** The chest rises and falls all the time.
//! * **A real gait.** Walking is generated from the curves of human gait:
//!   hip, knee and ankle angles over the stride, the pelvis turning and
//!   dropping, shoulders turning against it, arms swinging opposite to
//!   the legs, the head held steady. The styles change those amounts.
//! * **Feet on the ground, weight over the feet.** A [`Performance`] plays
//!   the motion through on a body: feet stay where they were put down
//!   (the legs reach them with IK), the hips keep the centre of mass over
//!   the feet that carry it, and standing at ease the body sways and
//!   shifts its weight from leg to leg.

use glam::{Affine3A, Quat, Vec2, Vec3};

use crate::model::{BodyMesh, BodyModel};
use crate::pose::Pose;

/// The animations the avatar can play.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Motion {
    /// An everyday walk.
    Walk,
    /// Upright and rigid: short steps, straight knees, arms held still.
    StiffWalk,
    /// A catwalk: the hips swing and drop, feet land in a line.
    HipSwayWalk,
    /// A box appears behind the avatar and it sits down on it, looks
    /// around, and stands up again.
    SitOnChair,
    /// Mountain, upward salute, warrior II, reverse warrior, tree and
    /// side stretches.
    Yoga,
}

impl Motion {
    pub const ALL: [Motion; 5] = [Motion::Walk, Motion::StiffWalk, Motion::HipSwayWalk, Motion::SitOnChair, Motion::Yoga];

    pub fn label(self) -> &'static str {
        match self {
            Motion::Walk => "Walk",
            Motion::StiffWalk => "Stiff walk",
            Motion::HipSwayWalk => "Hip-sway walk",
            Motion::SitOnChair => "Sit on a chair",
            Motion::Yoga => "Yoga flow",
        }
    }

    /// Seconds until the motion repeats.
    pub fn duration(self) -> f32 {
        match self {
            Motion::Walk | Motion::StiffWalk | Motion::HipSwayWalk => WalkPlan::of(self).duration(),
            Motion::SitOnChair => sitting().duration(),
            Motion::Yoga => yoga().duration(),
        }
    }

    /// The joint angles at `time` seconds (wrapped into the loop).
    pub fn angles(self, time: f32) -> Angles {
        let t = time.rem_euclid(self.duration());
        let mut angles = match self {
            Motion::Walk | Motion::StiffWalk | Motion::HipSwayWalk => walk_angles(&WalkPlan::of(self), t),
            Motion::SitOnChair => sitting().sample(t),
            Motion::Yoga => yoga().sample(t),
        };
        let stiff = self == Motion::StiffWalk;
        breathe(&mut angles, time, if stiff { 0.4 } else { 1.0 });
        angles
    }

    /// The keyframes of a keyframed motion.
    fn keys(self) -> Option<Keys> {
        match self {
            Motion::SitOnChair => Some(sitting()),
            Motion::Yoga => Some(yoga()),
            _ => None,
        }
    }

    /// How far the chair has grown out of the floor, 0..=1, if this
    /// motion has one.
    pub fn chair(self, time: f32) -> Option<f32> {
        if self != Motion::SitOnChair {
            return None;
        }
        let t = time.rem_euclid(self.duration());
        // Rises with a little overshoot, sinks away at the end.
        let grow = ease_out_back(((t - 0.2) / 0.8).clamp(0.0, 1.0));
        let shrink = 1.0 - smooth(((t - 9.3) / 0.6).clamp(0.0, 1.0));
        Some(grow * shrink)
    }

    /// The seated pose, which decides where the chair goes.
    pub fn seated() -> Angles {
        seated_pose()
    }
}

// ---------------------------------------------------------------------------
// Joint angles
// ---------------------------------------------------------------------------

/// Which side of the body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// The avatar's left (+X).
    Left = 0,
    Right = 1,
}

impl Side {
    const BOTH: [Side; 2] = [Side::Left, Side::Right];

    /// +1 on the left (+X), −1 on the right.
    fn sign(self) -> f32 {
        if self == Side::Left { 1.0 } else { -1.0 }
    }

    fn suffix(self) -> &'static str {
        if self == Side::Left { "L" } else { "R" }
    }
}

/// One joint angle (degrees, or metres for the shifts). Body-wide ones
/// first, then one block per side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Joint {
    /// Pelvis tips forward (anterior tilt).
    PelvisTilt,
    /// Pelvis drops on the right (negative: on the left).
    PelvisList,
    /// Pelvis turns to face the avatar's left.
    PelvisTurn,
    /// The body moves towards the avatar's left, metres. Standing, it is
    /// the weight that moves: the hips shift over the feet.
    ShiftX,
    /// The body moves forwards, metres (standing: the weight moves).
    ShiftZ,
    /// The spine bends forwards (negative: arches back).
    SpineFlex,
    /// The spine bends towards the avatar's left.
    SpineSide,
    /// The chest turns to face the avatar's left.
    SpineTwist,
    /// The head nods forward (negative: looks up).
    NeckFlex,
    /// The head tilts towards the avatar's left.
    NeckSide,
    /// The head turns towards the avatar's left.
    NeckTurn,
    /// How much the body keeps its weight over its feet, 0..=1: 0 while
    /// something else carries it (a chair).
    Balance,
    /// How far the hips sink below where straight legs would hold them,
    /// metres: the legs bend to let them (a lunge, a squat).
    HipDrop,
    /// Which way the whole body faces, degrees from the front, turning
    /// towards its left.
    Heading,
    /// The shoulder lifts (shrug).
    Shrug(Side),
    /// The arm rises sideways from hanging (0) through horizontal (90) to
    /// overhead (180).
    ArmRaise(Side),
    /// The arm swings forwards (negative: back).
    ArmSwing(Side),
    /// The arm turns outwards about its own axis (negative: inwards).
    ArmTwist(Side),
    /// The elbow bends: 0 straight.
    Elbow(Side),
    /// The hand bends towards the palm.
    Wrist(Side),
    /// The thigh swings forwards (negative: back).
    HipFlex(Side),
    /// The leg spreads sideways, degrees from vertical.
    HipSpread(Side),
    /// The leg turns outwards (toes out).
    HipTurn(Side),
    /// The knee bends: 0 straight.
    Knee(Side),
    /// The toes come up (negative: point down).
    Ankle(Side),
    /// How much this foot stays where it stands, 0..=1.
    Plant(Side),
    /// How far from the hip a stepping foot is put down, metres; 0 puts
    /// it where the leg points. A long step bends the standing knee.
    Reach(Side),
}

const SHARED: usize = 14;
const PER_SIDE: usize = 13;
const COUNT: usize = SHARED + 2 * PER_SIDE;

impl Joint {
    fn index(self) -> usize {
        let side = |s: Side, k: usize| SHARED + s as usize * PER_SIDE + k;
        match self {
            Joint::PelvisTilt => 0,
            Joint::PelvisList => 1,
            Joint::PelvisTurn => 2,
            Joint::ShiftX => 3,
            Joint::ShiftZ => 4,
            Joint::SpineFlex => 5,
            Joint::SpineSide => 6,
            Joint::SpineTwist => 7,
            Joint::NeckFlex => 8,
            Joint::NeckSide => 9,
            Joint::NeckTurn => 10,
            Joint::Balance => 11,
            Joint::HipDrop => 12,
            Joint::Heading => 13,
            Joint::Shrug(s) => side(s, 0),
            Joint::ArmRaise(s) => side(s, 1),
            Joint::ArmSwing(s) => side(s, 2),
            Joint::ArmTwist(s) => side(s, 3),
            Joint::Elbow(s) => side(s, 4),
            Joint::Wrist(s) => side(s, 5),
            Joint::HipFlex(s) => side(s, 6),
            Joint::HipSpread(s) => side(s, 7),
            Joint::HipTurn(s) => side(s, 8),
            Joint::Knee(s) => side(s, 9),
            Joint::Ankle(s) => side(s, 10),
            Joint::Plant(s) => side(s, 11),
            Joint::Reach(s) => side(s, 12),
        }
    }

    /// Joints that trail the movement, and by how many seconds.
    fn lag(index: usize) -> f32 {
        if index < SHARED {
            return if (8..=10).contains(&index) { 0.12 } else { 0.0 };
        }
        match (index - SHARED) % PER_SIDE {
            4 => 0.1,  // elbow
            5 => 0.18, // wrist
            3 => 0.05, // arm twist
            _ => 0.0,
        }
    }
}

/// A full set of joint angles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Angles([f32; COUNT]);

impl Default for Angles {
    /// Standing at ease: arms down by the sides, palms in, elbows soft,
    /// feet under the hips and on the ground.
    fn default() -> Self {
        let mut a = Angles([0.0; COUNT]);
        for side in Side::BOTH {
            a.set(Joint::ArmRaise(side), 13.0);
            a.set(Joint::ArmSwing(side), 3.0);
            a.set(Joint::Elbow(side), 12.0);
            a.set(Joint::Wrist(side), 4.0);
            a.set(Joint::HipSpread(side), 4.0);
            a.set(Joint::HipTurn(side), 6.0);
            a.set(Joint::Plant(side), 1.0);
        }
        a.set(Joint::Balance, 1.0);
        a
    }
}

impl Angles {
    pub fn get(&self, joint: Joint) -> f32 {
        self.0[joint.index()]
    }

    pub fn set(&mut self, joint: Joint, value: f32) -> &mut Self {
        self.0[joint.index()] = value;
        self
    }

    /// Sets a per-side joint on both sides.
    fn both(&mut self, joint: fn(Side) -> Joint, value: f32) -> &mut Self {
        for side in Side::BOTH {
            self.set(joint(side), value);
        }
        self
    }

    fn add(&mut self, joint: Joint, value: f32) {
        self.0[joint.index()] += value;
    }

    /// The bone rotations for these angles on a body (`rest` is that body
    /// in the modelled pose).
    pub fn to_pose(&self, model: &BodyModel, rest: &BodyMesh) -> Pose {
        let rig = Rig::new(model, rest);
        let mut pose = Pose::rest();
        let a = |j: Joint| self.get(j).to_radians();

        // --- pelvis: moves everything; legs and spine are set against it,
        // so their angles are relative to the room, not the pelvis.
        let pelvis = Quat::from_rotation_y(a(Joint::PelvisTurn)) * Quat::from_rotation_z(a(Joint::PelvisList)) * Quat::from_rotation_x(a(Joint::PelvisTilt));
        pose.set("root", Quat::from_rotation_y(a(Joint::Heading)) * pelvis);
        let undo = pelvis.inverse();

        // --- spine: the bend shared over five vertebrae.
        let spine = |share: f32| {
            Quat::from_rotation_y(a(Joint::SpineTwist) * share)
                * Quat::from_rotation_z(-a(Joint::SpineSide) * share)
                * Quat::from_rotation_x(a(Joint::SpineFlex) * share)
        };
        for (bone, share) in [("spine05", 0.14), ("spine04", 0.2), ("spine03", 0.22), ("spine02", 0.24), ("spine01", 0.2)] {
            let q = spine(share);
            pose.set(bone, if bone == "spine05" { undo * q } else { q });
        }
        let neck = |share: f32| {
            Quat::from_rotation_y(a(Joint::NeckTurn) * share)
                * Quat::from_rotation_z(-a(Joint::NeckSide) * share)
                * Quat::from_rotation_x(a(Joint::NeckFlex) * share)
        };
        for (bone, share) in [("neck01", 0.3), ("neck02", 0.3), ("neck03", 0.15), ("head", 0.25)] {
            pose.set(bone, neck(share));
        }

        for side in Side::BOTH {
            let s = side.sign();
            let bone = |stem: &str| format!("{stem}.{}", side.suffix());

            // --- arm. A raise comes partly from the shoulder girdle: the
            // collarbone and shoulder lift the last part of the way up.
            let raise = self.get(Joint::ArmRaise(side));
            let swing = self.get(Joint::ArmSwing(side));
            let girdle_lift = 0.2 * (raise - 60.0).max(0.0) + 0.15 * (swing.abs() - 70.0).max(0.0);
            let delta = (raise - rig.arm_raise).to_radians();
            let shrug = (self.get(Joint::Shrug(side)) + girdle_lift * 0.5).to_radians();
            pose.set(&bone("clavicle"), Quat::from_rotation_z(s * (shrug + 0.08 * delta)));
            pose.set(&bone("shoulder01"), Quat::from_rotation_z(s * (girdle_lift * 0.5).to_radians() * 0.6 + s * 0.12 * delta));
            let twist = Quat::from_axis_angle(rig.arm_axis[side as usize], -s * a(Joint::ArmTwist(side)));
            let upper = Quat::from_rotation_x(-swing.to_radians()) * Quat::from_rotation_z(s * 0.8 * delta) * twist;
            pose.set(&bone("upperarm01"), upper);
            let elbow = (self.get(Joint::Elbow(side)) - rig.elbow).to_radians();
            let fold = rig.elbow_axis[side as usize];
            pose.set(&bone("lowerarm01"), Quat::from_axis_angle(fold, elbow));
            pose.set(&bone("wrist"), Quat::from_axis_angle(fold, a(Joint::Wrist(side))));

            // --- leg.
            let spread = (self.get(Joint::HipSpread(side)) - rig.leg_spread).to_radians();
            let leg = Quat::from_rotation_x(-a(Joint::HipFlex(side))) * Quat::from_rotation_z(s * spread) * Quat::from_rotation_y(s * a(Joint::HipTurn(side)));
            pose.set(&bone("upperleg01"), undo * leg);
            pose.set(&bone("lowerleg01"), Quat::from_rotation_x(a(Joint::Knee(side))));
            pose.set(&bone("foot"), Quat::from_rotation_x(-a(Joint::Ankle(side))));
        }

        pose.offset = Vec3::new(self.get(Joint::ShiftX), 0.0, self.get(Joint::ShiftZ));
        pose
    }
}

/// What the modelled pose already is, so angles can be absolute.
struct Rig {
    /// Arm raise from hanging, degrees.
    arm_raise: f32,
    /// Elbow bend, degrees.
    elbow: f32,
    /// Leg spread from vertical, degrees.
    leg_spread: f32,
    /// Upper arm directions, left and right.
    arm_axis: [Vec3; 2],
    /// Axes the forearms fold about, left and right.
    elbow_axis: [Vec3; 2],
}

impl Rig {
    fn new(model: &BodyModel, rest: &BodyMesh) -> Self {
        let joint = |name: &str| model.bone(name).map(|i| rest.joints[i].head);
        let mut rig = Rig { arm_raise: 40.0, elbow: 0.0, leg_spread: 4.0, arm_axis: [Vec3::NEG_Y; 2], elbow_axis: [Vec3::X; 2] };
        for side in Side::BOTH {
            let bone = |stem: &str| joint(&format!("{stem}.{}", side.suffix()));
            if let (Some(shoulder), Some(elbow), Some(wrist)) = (bone("upperarm01"), bone("lowerarm01"), bone("wrist")) {
                let upper = (elbow - shoulder).normalize_or(Vec3::NEG_Y);
                let fore = (wrist - elbow).normalize_or(upper);
                rig.arm_axis[side as usize] = upper;
                rig.elbow_axis[side as usize] = fore.cross(Vec3::Z).normalize_or(Vec3::X);
                if side == Side::Left {
                    rig.arm_raise = upper.x.abs().atan2(-upper.y).to_degrees();
                    rig.elbow = upper.angle_between(fore).to_degrees();
                }
            }
            if let (Some(hip), Some(knee)) = (bone("upperleg01"), bone("lowerleg01"))
                && side == Side::Left
            {
                let along = (knee - hip).normalize_or(Vec3::NEG_Y);
                rig.leg_spread = along.x.atan2(-along.y).to_degrees();
            }
        }
        rig
    }
}

// ---------------------------------------------------------------------------
// Keyframes
// ---------------------------------------------------------------------------

struct Keys {
    times: Vec<f32>,
    poses: Vec<Angles>,
}

impl Keys {
    fn new() -> Self {
        Self { times: Vec::new(), poses: Vec::new() }
    }

    /// Adds a key at `time`; `edit` changes a copy of the previous pose.
    fn key(&mut self, time: f32, edit: impl FnOnce(&mut Angles)) -> &mut Self {
        let mut pose = self.poses.last().copied().unwrap_or_default();
        edit(&mut pose);
        self.times.push(time);
        self.poses.push(pose);
        self
    }

    /// Holds the previous pose until `time`.
    fn hold(&mut self, time: f32) -> &mut Self {
        self.key(time, |_| {})
    }

    fn duration(&self) -> f32 {
        self.times.last().copied().unwrap_or(1.0)
    }

    /// Every joint at time `t`, each on its own monotone cubic, the
    /// trailing joints a little behind.
    fn sample(&self, t: f32) -> Angles {
        let mut out = Angles([0.0; COUNT]);
        for index in 0..COUNT {
            let at = (t - Joint::lag(index)).max(0.0);
            out.0[index] = self.channel(index, at);
        }
        out
    }

    fn channel(&self, index: usize, t: f32) -> f32 {
        let n = self.times.len();
        let value = |k: usize| self.poses[k].0[index];
        if t <= self.times[0] {
            return value(0);
        }
        if t >= self.times[n - 1] {
            return value(n - 1);
        }
        let k = self.times.windows(2).position(|w| t < w[1]).unwrap_or(n - 2);
        let (t0, t1) = (self.times[k], self.times[k + 1]);
        let h = (t1 - t0).max(1e-4);
        // Fritsch-Carlson slopes: zero at a turning point or a hold.
        let slope = |k: usize| -> f32 {
            if k == 0 || k == n - 1 {
                return 0.0;
            }
            let before = (value(k) - value(k - 1)) / (self.times[k] - self.times[k - 1]).max(1e-4);
            let after = (value(k + 1) - value(k)) / (self.times[k + 1] - self.times[k]).max(1e-4);
            if before * after <= 0.0 { 0.0 } else { 2.0 / (1.0 / before + 1.0 / after) }
        };
        let (p0, p1, m0, m1) = (value(k), value(k + 1), slope(k) * h, slope(k + 1) * h);
        let u = (t - t0) / h;
        let (u2, u3) = (u * u, u * u * u);
        (2.0 * u3 - 3.0 * u2 + 1.0) * p0 + (u3 - 2.0 * u2 + u) * m0 + (-2.0 * u3 + 3.0 * u2) * p1 + (u3 - u2) * m1
    }
}

/// The chest rises with each breath; `amount` 1 is a relaxed person.
fn breathe(angles: &mut Angles, time: f32, amount: f32) {
    use std::f32::consts::TAU;
    let breath = (time * TAU / 4.2).sin();
    angles.add(Joint::SpineFlex, -1.2 * breath * amount);
    angles.add(Joint::NeckFlex, 0.6 * breath * amount);
    for side in Side::BOTH {
        angles.add(Joint::Shrug(side), 1.2 * breath * amount);
        angles.add(Joint::ArmRaise(side), 0.6 * breath * amount);
    }
}

fn smooth(x: f32) -> f32 {
    x * x * (3.0 - 2.0 * x)
}

/// Like [`smooth`], and its acceleration also starts and ends at zero.
fn smoother(x: f32) -> f32 {
    x * x * x * (x * (6.0 * x - 15.0) + 10.0)
}

fn ease_out_back(x: f32) -> f32 {
    let c = 1.6;
    1.0 + (c + 1.0) * (x - 1.0).powi(3) + c * (x - 1.0).powi(2)
}

// ---------------------------------------------------------------------------
// Walking
// ---------------------------------------------------------------------------

/// How a person walks, in amounts.
struct Gait {
    /// Seconds per stride (two steps).
    period: f32,
    /// Hip swing either side of `hip_mean`, degrees.
    hip: f32,
    hip_mean: f32,
    /// Knee bend as the foot takes the weight, and in the swing.
    knee_load: f32,
    knee_swing: f32,
    /// How much the ankle works, 0..=1.
    ankle: f32,
    /// Pelvis turn and drop, degrees.
    pelvis_turn: f32,
    pelvis_drop: f32,
    /// Feet placed towards the centre line, degrees of hip adduction: at a
    /// few degrees they land on one line, and the hips swing over each.
    cross: f32,
    /// Shoulders turning against the pelvis, degrees.
    counter: f32,
    /// Arm swing either side, degrees, and the extra elbow bend with it.
    arm: f32,
    elbow: f32,
    elbow_swing: f32,
    /// Arms held out from the body, degrees.
    arm_out: f32,
    /// Forward lean of the trunk (negative: leaning back, chest up).
    lean: f32,
    /// Shoulders pulled up.
    shrug: f32,
    /// How much the shoulders rise and fall with the arm swing, degrees.
    shoulder_roll: f32,
    /// How loose the hands are: their lag behind the swing, 0..=1.
    loose: f32,
}

impl Gait {
    fn of(motion: Motion) -> Gait {
        let natural = Gait {
            period: 1.08,
            hip: 22.0,
            hip_mean: 8.0,
            knee_load: 17.0,
            knee_swing: 60.0,
            ankle: 1.0,
            pelvis_turn: 8.0,
            pelvis_drop: 6.0,
            cross: 2.5,
            counter: 9.0,
            arm: 24.0,
            elbow: 14.0,
            elbow_swing: 20.0,
            arm_out: 8.0,
            lean: 2.0,
            shrug: 0.0,
            shoulder_roll: 3.0,
            loose: 0.7,
        };
        match motion {
            Motion::StiffWalk => Gait {
                period: 0.95,
                hip: 13.0,
                hip_mean: 4.0,
                knee_load: 3.0,
                knee_swing: 26.0,
                ankle: 0.35,
                pelvis_turn: 1.0,
                pelvis_drop: 0.5,
                cross: 0.0,
                counter: 0.5,
                arm: 3.5,
                elbow: 6.0,
                elbow_swing: 2.0,
                arm_out: 5.0,
                lean: -1.5,
                shrug: 5.0,
                shoulder_roll: 0.0,
                loose: 0.0,
            },
            // The catwalk: long strides placed on one line, the hips
            // swinging and dropping over each, the chest up and the
            // shoulders answering the hips.
            Motion::HipSwayWalk => Gait {
                period: 1.16,
                hip: 24.0,
                knee_swing: 62.0,
                pelvis_turn: 13.0,
                pelvis_drop: 12.0,
                cross: 9.0,
                counter: 13.0,
                arm: 20.0,
                elbow: 18.0,
                elbow_swing: 14.0,
                arm_out: 11.0,
                lean: -3.0,
                shoulder_roll: 6.0,
                loose: 1.0,
                ..natural
            },
            _ => natural,
        }
    }

    /// The angles at `t` seconds into the cycle. Phase 0 is the left heel
    /// touching down; the right foot is half a stride behind. Each foot is
    /// on the floor (planted) from heel strike to just before toe-off.
    fn angles(&self, t: f32) -> Angles {
        use std::f32::consts::TAU;
        let mut a = Angles::default();
        let phase = (t / self.period).fract();
        let wave = |p: f32| (TAU * p).cos();
        // A bump centred at `c` (0..1, wrapping) of width `w`.
        let bump = |p: f32, c: f32, w: f32| {
            let d = (p - c + 0.5).rem_euclid(1.0) - 0.5;
            (-0.5 * (d / w).powi(2)).exp()
        };
        for side in Side::BOTH {
            let p = (phase + if side == Side::Left { 0.0 } else { 0.5 }).fract();
            a.set(Joint::HipFlex(side), self.hip_mean + self.hip * wave(p));
            a.set(
                Joint::Knee(side),
                4.0 + self.knee_load * bump(p, 0.13, 0.06) + self.knee_swing * bump(p, 0.72, 0.1),
            );
            a.set(
                Joint::Ankle(side),
                self.ankle * (-6.0 * bump(p, 0.06, 0.04) + 9.0 * bump(p, 0.42, 0.11) - 15.0 * bump(p, 0.62, 0.05) + 5.0 * bump(p, 0.84, 0.08)),
            );
            // Feet land towards the centre line, and come back out in the swing.
            a.set(Joint::HipSpread(side), 4.0 - self.cross * (1.0 - bump(p, 0.72, 0.14)));
            // Flat on the floor between rolling down off the heel (10 % of
            // the stride) and the heel rising (38 %); see `foot_roll`.
            let planted = smoothstep(0.04, 0.1, p) * (1.0 - smoothstep(0.38, 0.44, p));
            a.set(Joint::Plant(side), planted);
            // Arms swing opposite to the legs; the forearm and hand trail
            // the upper arm, and the elbow bends as the arm comes forward.
            let swing = |lag: f32| -self.arm * wave(p - lag) + 3.0;
            let arm = swing(0.0);
            a.set(Joint::ArmSwing(side), arm);
            let forearm = swing(0.05 * self.loose);
            a.set(Joint::Elbow(side), self.elbow + self.elbow_swing * (forearm / self.arm.max(1.0)).max(0.0));
            let hand = swing(0.1 * self.loose);
            a.set(Joint::Wrist(side), 4.0 + 10.0 * self.loose * (hand - arm) / self.arm.max(1.0));
            a.set(Joint::ArmRaise(side), self.arm_out + 2.0 * self.loose * bump(p, 0.25, 0.1));
            a.set(Joint::ArmTwist(side), 6.0);
            // The shoulder rises a little as its arm swings forward.
            a.set(Joint::Shrug(side), self.shrug + self.shoulder_roll * 0.5 * (1.0 + (arm - 3.0) / self.arm.max(1.0)));
        }
        let left = wave(phase);
        let stance = (TAU * phase).sin(); // + while on the left foot
        a.set(Joint::PelvisTurn, -self.pelvis_turn * left);
        a.set(Joint::PelvisList, self.pelvis_drop * stance);
        a.set(Joint::SpineTwist, self.counter * left);
        // The trunk bends against the dropping hip; the head stays level
        // and looks ahead.
        a.set(Joint::SpineSide, 0.55 * self.pelvis_drop * stance);
        a.set(Joint::NeckTurn, -0.8 * self.counter * left);
        a.set(Joint::NeckSide, -0.5 * self.pelvis_drop * stance);
        a.set(Joint::SpineFlex, self.lean);
        a.set(Joint::NeckFlex, -0.5 * self.lean);
        a.set(Joint::PelvisTilt, 0.5 * self.lean + 2.0 * (2.0 * TAU * phase).cos());
        // The hips follow the feet; nothing else balances them.
        a.set(Joint::Balance, 0.0);
        a
    }
}

/// Strides along each straight of the runway, and in each U-turn.
/// The runway: two straights joined by half circles, centred on the
/// origin. The avatar walks towards the front (+Z), turns to its left, walks
/// away and turns again.
struct Runway {
    straight: f32,
    radius: f32,
}

const RUNWAY: Runway = Runway { straight: 2.4, radius: 0.5 };

impl Runway {
    fn length(&self) -> f32 {
        2.0 * self.straight + 2.0 * std::f32::consts::PI * self.radius
    }

    /// The point `d` metres along the path, and the heading there (radians
    /// from +Z towards +X).
    fn at(&self, d: f32) -> (Vec3, f32) {
        use std::f32::consts::PI;
        let (l, r) = (self.straight, self.radius);
        let arc = PI * r;
        let d = d.rem_euclid(self.length());
        let (point, heading) = if d < l {
            (Vec3::new(0.0, 0.0, -0.5 * l + d), 0.0)
        } else if d < l + arc {
            let a = (d - l) / r;
            (Vec3::new(r - r * a.cos(), 0.0, 0.5 * l + r * a.sin()), a)
        } else if d < 2.0 * l + arc {
            (Vec3::new(2.0 * r, 0.0, 0.5 * l - (d - l - arc)), PI)
        } else {
            let a = (d - 2.0 * l - arc) / r;
            (Vec3::new(r + r * a.cos(), 0.0, -0.5 * l - r * a.sin()), PI + a)
        };
        (point - Vec3::new(r, 0.0, 0.0), heading)
    }
}

/// A walk around the runway: the gait, and where the feet go.
struct WalkPlan {
    gait: Gait,
    /// Metres per stride, chosen so the lap is a whole number of strides.
    stride: f32,
    strides: f32,
    /// How far each foot lands from the path's centre line, metres.
    track: f32,
    /// Sideways sway of the hips over the standing foot, metres.
    sway: f32,
    /// Up-and-down travel of the hips over a step, metres.
    bob: f32,
}

impl WalkPlan {
    fn of(motion: Motion) -> WalkPlan {
        let gait = Gait::of(motion);
        // (stride, track, sway, bob). A stride is about one and a half leg
        // lengths at an everyday pace (1 m/s for this body).
        let (stride, track, sway, bob) = match motion {
            Motion::StiffWalk => (0.85, 0.095, 0.006, 0.012),
            // Feet on one line, the hips swinging out over each.
            Motion::HipSwayWalk => (1.15, 0.012, 0.045, 0.02),
            _ => (1.12, 0.07, 0.022, 0.026),
        };
        let strides = (RUNWAY.length() / stride).round().max(1.0);
        WalkPlan { gait, stride: RUNWAY.length() / strides, strides, track, sway, bob }
    }

    fn duration(&self) -> f32 {
        self.strides * self.gait.period
    }

    fn speed(&self) -> f32 {
        self.stride / self.gait.period
    }

    /// Where a foot is put down for its `k`-th step: the point on the floor
    /// under its ankle and the way it points.
    fn footprint(&self, side: Side, k: f32) -> (Vec3, f32) {
        let strike = (k + if side == Side::Left { 0.0 } else { 0.5 }) * self.gait.period;
        // Landing a little ahead of the hips.
        let (point, heading) = RUNWAY.at(self.speed() * strike + 0.27 * self.stride);
        let left = Vec3::new(heading.cos(), 0.0, -heading.sin());
        // Toes turned out a little.
        (point + left * side.sign() * self.track, heading + side.sign() * 6f32.to_radians())
    }
}

/// The gait's angles with the body facing along the runway.
fn walk_angles(plan: &WalkPlan, t: f32) -> Angles {
    let mut a = plan.gait.angles(t);
    let (_, heading) = RUNWAY.at(plan.speed() * t);
    a.set(Joint::Heading, heading.to_degrees());
    a
}

/// How a foot is held at phase `p` of its stride: tilt (radians, toes
/// down positive) and whether it pivots on the heel, the ankle or the toes.
#[derive(Clone, Copy)]
enum Pivot {
    Heel,
    Ankle,
    Toe,
}

fn foot_roll(p: f32) -> (f32, Pivot) {
    const STRIKE: f32 = -12.0;
    // The foot's angle to the floor as the toes leave it.
    const PUSH: f32 = 50.0;
    if p < 0.1 {
        // Heel strike, rolling down onto the sole.
        ((STRIKE * (1.0 - smoother(p / 0.1))).to_radians(), Pivot::Heel)
    } else if p < 0.38 {
        (0.0, Pivot::Heel)
    } else if p < 0.62 {
        // The heel rises and the foot peels off over the toes.
        ((PUSH * smoother((p - 0.38) / 0.24)).to_radians(), Pivot::Toe)
    } else {
        // In the air: from the push-off, through level, to the toes up for
        // the next heel strike.
        let u = (p - 0.62) / 0.38;
        let tilt = if u < 0.45 { PUSH * (1.0 - smoother(u / 0.45)) } else { STRIKE * smoother((u - 0.45) / 0.55) };
        (tilt.to_radians(), Pivot::Ankle)
    }
}

// ---------------------------------------------------------------------------
// Sitting
// ---------------------------------------------------------------------------

fn seated_pose() -> Angles {
    let mut a = Angles::default();
    a.both(Joint::HipFlex, 88.0).both(Joint::Knee, 92.0).both(Joint::Ankle, 6.0).both(Joint::HipSpread, 7.0).both(Joint::HipTurn, 8.0);
    a.set(Joint::SpineFlex, 4.0).set(Joint::PelvisTilt, -6.0);
    // The chair carries the weight now.
    a.set(Joint::Balance, 0.0);
    // Hands resting on the thighs.
    a.both(Joint::ArmSwing, 16.0).both(Joint::Elbow, 70.0).both(Joint::ArmRaise, 12.0).both(Joint::ArmTwist, -12.0).both(Joint::Wrist, -8.0);
    a
}

fn sitting() -> Keys {
    let mut k = Keys::new();
    let seated = seated_pose();
    k.key(0.0, |_| {}).hold(1.0);
    // Glance down at the chair, a little weight onto the heels.
    k.key(1.6, |a| {
        a.set(Joint::NeckFlex, 16.0).set(Joint::NeckTurn, 12.0).both(Joint::Knee, 8.0).both(Joint::HipFlex, 6.0).both(Joint::ArmSwing, 10.0);
    });
    // Lean forward to keep the balance over the feet as the hips go back.
    k.key(2.4, |a| {
        a.set(Joint::NeckFlex, 4.0).set(Joint::NeckTurn, 0.0).set(Joint::SpineFlex, 30.0);
        a.both(Joint::HipFlex, 48.0).both(Joint::Knee, 52.0).both(Joint::Ankle, 16.0);
        a.both(Joint::ArmSwing, 28.0).both(Joint::Elbow, 30.0);
    });
    // Down onto the seat.
    k.key(3.3, |a| {
        *a = seated;
        a.set(Joint::SpineFlex, 22.0).set(Joint::NeckFlex, 0.0).both(Joint::Ankle, 10.0);
    });
    // Settle back and sit up.
    k.key(4.1, |a| *a = seated);
    k.hold(5.0);
    // Look around.
    k.key(5.8, |a| {
        a.set(Joint::NeckTurn, 35.0).set(Joint::SpineTwist, 8.0).set(Joint::NeckSide, 4.0);
    });
    k.hold(6.3);
    k.key(7.0, |a| {
        a.set(Joint::NeckTurn, -10.0).set(Joint::SpineTwist, -3.0).set(Joint::NeckSide, -2.0);
    });
    k.key(7.4, |a| {
        a.set(Joint::NeckTurn, 0.0).set(Joint::SpineTwist, 0.0).set(Joint::NeckSide, 0.0);
    });
    // Lean forward, push up from the thighs, stand.
    k.key(7.9, |a| {
        a.set(Joint::SpineFlex, 34.0).set(Joint::NeckFlex, 6.0);
        a.both(Joint::ArmSwing, 26.0).both(Joint::Elbow, 40.0).both(Joint::Ankle, 14.0);
    });
    k.key(8.7, |a| {
        a.set(Joint::SpineFlex, 16.0).both(Joint::HipFlex, 30.0).both(Joint::Knee, 30.0).both(Joint::Ankle, 8.0);
        // Off the seat: the feet carry the weight again.
        a.set(Joint::Balance, 1.0);
        a.both(Joint::ArmSwing, 12.0).both(Joint::Elbow, 20.0);
    });
    k.key(9.4, |a| *a = Angles::default());
    k.hold(10.4);
    k
}

// ---------------------------------------------------------------------------
// Yoga
// ---------------------------------------------------------------------------

fn prayer(a: &mut Angles) {
    // Palms together in front of the chest.
    a.both(Joint::ArmRaise, 30.0).both(Joint::ArmSwing, 46.0).both(Joint::ArmTwist, -60.0).both(Joint::Elbow, 128.0).both(Joint::Wrist, -30.0);
}

fn overhead(a: &mut Angles) {
    a.both(Joint::ArmRaise, 172.0).both(Joint::ArmSwing, 6.0).both(Joint::ArmTwist, 20.0).both(Joint::Elbow, 4.0).both(Joint::Wrist, 0.0);
}

fn mountain(a: &mut Angles) {
    *a = Angles::default();
    a.both(Joint::ArmRaise, 13.0).both(Joint::ArmTwist, 14.0).both(Joint::Elbow, 6.0).both(Joint::HipSpread, 3.0).both(Joint::HipTurn, 3.0);
}

/// Shifts the weight onto the left foot, lifts the right and puts it
/// down `spread` degrees out; returns the time it is down.
fn step_right(k: &mut Keys, at: f32, spread: f32, reach: f32) -> f32 {
    use Side::Right;
    // Off the right foot: the hips move over the left one.
    k.key(at + 0.7, |a| {
        a.set(Joint::Plant(Right), 0.0).set(Joint::Reach(Right), reach);
    });
    k.key(at + 1.3, |a| {
        a.set(Joint::HipFlex(Right), 18.0).set(Joint::Knee(Right), 34.0).set(Joint::Ankle(Right), -8.0);
        a.set(Joint::HipSpread(Right), 0.5 * (a.get(Joint::HipSpread(Right)) + spread));
        // Decided as the foot leaves the floor, held until it is down.
        a.set(Joint::Reach(Right), reach);
    });
    // The foot lands with the knee a little bent, as feet do.
    k.key(at + 1.95, |a| {
        a.set(Joint::HipFlex(Right), 2.0).set(Joint::Knee(Right), 5.0).set(Joint::Ankle(Right), 0.0).set(Joint::HipSpread(Right), spread);
        a.set(Joint::Reach(Right), reach);
    });
    // Down, and the weight comes back onto it, gradually.
    k.key(at + 2.05, |a| {
        a.set(Joint::Plant(Right), 0.02);
    });
    k.key(at + 2.9, |a| {
        a.set(Joint::Plant(Right), 1.0);
    });
    at + 2.9
}

fn yoga() -> Keys {
    use Side::{Left, Right};
    let mut k = Keys::new();
    k.key(0.0, mountain).hold(1.5);

    // Upward salute: arms sweep up the sides on the in-breath.
    k.key(3.0, |a| {
        a.both(Joint::ArmRaise, 95.0).both(Joint::ArmTwist, 45.0).both(Joint::Elbow, 4.0);
    });
    k.key(4.4, |a| {
        overhead(a);
        a.set(Joint::SpineFlex, -6.0).set(Joint::NeckFlex, -14.0);
    });
    k.hold(5.4);
    // Hands down to the heart.
    k.key(7.0, |a| {
        prayer(a);
        a.set(Joint::SpineFlex, 0.0).set(Joint::NeckFlex, 4.0);
    });
    k.hold(7.6);

    // Step the right foot out wide, opening the arms.
    let down = step_right(&mut k, 7.6, 40.0, 0.6);
    k.key(down + 0.8, |a| {
        a.both(Joint::ArmRaise, 88.0).both(Joint::ArmSwing, 0.0).both(Joint::ArmTwist, -5.0).both(Joint::Elbow, 4.0).both(Joint::Wrist, 0.0);
        a.set(Joint::NeckFlex, 0.0);
    });
    // Pivot the front foot out, then sink into Warrior II over the left
    // leg: the front knee over the ankle, the back leg straight, the gaze
    // over the front hand.
    k.key(down + 1.5, |a| {
        a.set(Joint::HipTurn(Left), 80.0).set(Joint::HipTurn(Right), -10.0);
    });
    k.key(down + 2.9, |a| {
        a.set(Joint::Knee(Left), 76.0).set(Joint::Ankle(Left), 8.0).set(Joint::HipSpread(Left), 40.0);
        // The hips sink, the weight towards the front foot.
        a.set(Joint::HipDrop, 0.03).set(Joint::ShiftX, 0.2);
        a.set(Joint::NeckTurn, 72.0).set(Joint::PelvisList, -3.0).set(Joint::SpineSide, 2.0);
    });
    let mut t = down + 4.9;
    k.hold(t);
    // Reverse warrior: the front arm sweeps up and back, the back hand
    // slides down the back leg.
    k.key(t + 1.6, |a| {
        a.set(Joint::ArmRaise(Left), 158.0).set(Joint::ArmTwist(Left), 30.0);
        a.set(Joint::ArmRaise(Right), 24.0).set(Joint::ArmSwing(Right), -6.0).set(Joint::Elbow(Right), 8.0);
        a.set(Joint::SpineSide, -16.0).set(Joint::NeckTurn, 25.0).set(Joint::NeckFlex, -18.0).set(Joint::NeckSide, 0.0);
    });
    k.hold(t + 2.8);
    k.key(t + 4.2, |a| {
        a.both(Joint::ArmRaise, 88.0).both(Joint::ArmTwist, -5.0).both(Joint::ArmSwing, 0.0).both(Joint::Elbow, 4.0);
        a.set(Joint::SpineSide, 2.0).set(Joint::NeckTurn, 72.0).set(Joint::NeckFlex, 0.0).set(Joint::NeckSide, 0.0);
    });
    // Straighten up, arms down, step back in.
    t += 5.6;
    k.key(t, |a| {
        a.set(Joint::Knee(Left), 0.0).set(Joint::Ankle(Left), 0.0).set(Joint::HipTurn(Left), 4.0).set(Joint::HipSpread(Left), 4.0);
        a.set(Joint::HipTurn(Right), 4.0).set(Joint::NeckTurn, 0.0).set(Joint::PelvisList, 0.0).set(Joint::SpineSide, 0.0);
        a.set(Joint::HipDrop, 0.0).set(Joint::ShiftX, 0.0);
        a.both(Joint::ArmRaise, 13.0).both(Joint::ArmTwist, 14.0).both(Joint::Elbow, 6.0).both(Joint::ArmSwing, 3.0);
    });
    let down = step_right(&mut k, t, 3.0, 0.0);
    k.key(down + 0.6, mountain);
    t = down + 1.2;
    k.hold(t);

    // Tree on the left leg: hands to the heart, weight onto the left foot,
    // the right sole against the left inner thigh.
    k.key(t + 1.4, |a| {
        prayer(a);
        a.set(Joint::Plant(Right), 0.0).set(Joint::PelvisList, 2.0);
    });
    k.key(t + 3.2, |a| {
        a.set(Joint::HipFlex(Right), 34.0).set(Joint::HipSpread(Right), 46.0).set(Joint::HipTurn(Right), 62.0).set(Joint::Knee(Right), 128.0);
        a.set(Joint::Ankle(Right), -18.0).set(Joint::PelvisList, 3.0).set(Joint::SpineSide, 2.0);
    });
    k.hold(t + 4.2);
    // Branches up.
    k.key(t + 5.8, |a| {
        overhead(a);
        a.both(Joint::ArmTwist, -10.0).set(Joint::NeckFlex, -4.0);
    });
    k.hold(t + 8.2);
    // Down from the tree.
    k.key(t + 9.8, prayer);
    k.key(t + 11.4, |a| {
        a.set(Joint::HipFlex(Right), 2.0).set(Joint::HipSpread(Right), 3.0).set(Joint::HipTurn(Right), 3.0).set(Joint::Knee(Right), 5.0);
        a.set(Joint::Ankle(Right), 0.0).set(Joint::PelvisList, 0.0).set(Joint::SpineSide, 0.0).set(Joint::NeckFlex, 0.0);
    });
    k.key(t + 11.6, |a| {
        a.set(Joint::Plant(Right), 0.02);
    });
    k.key(t + 12.4, |a| {
        a.set(Joint::Plant(Right), 1.0);
    });
    t += 12.4;

    // Side stretches: arms overhead, a long bend to each side.
    k.key(t + 1.6, |a| {
        overhead(a);
        a.both(Joint::Elbow, 2.0);
    });
    k.key(t + 3.4, |a| {
        a.set(Joint::SpineSide, 22.0).set(Joint::NeckSide, 6.0);
        a.set(Joint::ArmRaise(Right), 178.0).set(Joint::ArmRaise(Left), 160.0);
    });
    k.hold(t + 4.2);
    k.key(t + 6.2, |a| {
        a.set(Joint::SpineSide, -22.0).set(Joint::NeckSide, -6.0);
        a.set(Joint::ArmRaise(Left), 178.0).set(Joint::ArmRaise(Right), 160.0);
    });
    k.hold(t + 7.0);
    k.key(t + 8.4, |a| {
        overhead(a);
        a.set(Joint::SpineSide, 0.0).set(Joint::NeckSide, 0.0);
    });
    // Arms float down the sides.
    k.key(t + 10.0, |a| {
        a.both(Joint::ArmRaise, 90.0).both(Joint::ArmTwist, 30.0);
    });
    k.key(t + 11.6, mountain);
    k.hold(t + 12.6);
    k
}

// ---------------------------------------------------------------------------
// Performance: the whole sequence, physically
// ---------------------------------------------------------------------------

/// Mass shares of the body's segments and where along each its centre of
/// mass lies, from the proximal joint (Winter, *Biomechanics and Motor
/// Control of Human Movement*, table 4.1).
const SEGMENTS: [(&str, End, &str, End, f32, f32); 8] = [
    ("head", End::Head, "head", End::Tail, 0.081, 0.5),
    ("spine05", End::Head, "neck01", End::Head, 0.497, 0.5),
    ("upperarm01.?", End::Head, "lowerarm01.?", End::Head, 0.028, 0.436),
    ("lowerarm01.?", End::Head, "wrist.?", End::Head, 0.016, 0.43),
    ("wrist.?", End::Head, "wrist.?", End::Tail, 0.006, 0.5),
    ("upperleg01.?", End::Head, "lowerleg01.?", End::Head, 0.1, 0.433),
    ("lowerleg01.?", End::Head, "foot.?", End::Head, 0.0465, 0.433),
    ("foot.?", End::Head, "foot.?", End::Tail, 0.0145, 0.5),
];

#[derive(Debug, Clone, Copy)]
enum End {
    Head,
    Tail,
}

/// The bones the performance works with, looked up once.
struct Skeleton {
    /// `(from bone, end, to bone, end, mass, along)` with both sides spelled out.
    segments: Vec<(usize, End, usize, End, f32, f32)>,
    /// Per side: thigh, knee and foot bones, and the bones above them.
    legs: [[usize; 3]; 2],
    leg_parents: [[usize; 3]; 2],
    names: [[String; 3]; 2],
}

impl Skeleton {
    fn new(model: &BodyModel) -> Option<Self> {
        let mut segments = Vec::new();
        for &(from, from_end, to, to_end, mass, along) in &SEGMENTS {
            let sides: &[&str] = if from.contains('?') { &["L", "R"] } else { &[""] };
            for side in sides {
                let name = |n: &str| n.replace('?', side);
                segments.push((model.bone(&name(from))?, from_end, model.bone(&name(to))?, to_end, mass, along));
            }
        }
        let mut legs = [[0; 3]; 2];
        let mut leg_parents = [[0; 3]; 2];
        let mut names: [[String; 3]; 2] = Default::default();
        for side in Side::BOTH {
            for (k, stem) in ["upperleg01", "lowerleg01", "foot"].iter().enumerate() {
                let name = format!("{stem}.{}", side.suffix());
                let bone = model.bone(&name)?;
                legs[side as usize][k] = bone;
                leg_parents[side as usize][k] = model.bones()[bone].parent?;
                names[side as usize][k] = name;
            }
        }
        Some(Self { segments, legs, leg_parents, names })
    }

    fn point(rest: &BodyMesh, bones: &[Affine3A], bone: usize, end: End) -> Vec3 {
        let joint = rest.joints[bone];
        bones[bone].transform_point3(match end {
            End::Head => joint.head,
            End::Tail => joint.tail,
        })
    }

    fn centre_of_mass(&self, rest: &BodyMesh, bones: &[Affine3A]) -> Vec3 {
        let (mut sum, mut mass) = (Vec3::ZERO, 0.0);
        for &(from, from_end, to, to_end, m, along) in &self.segments {
            let (a, b) = (Self::point(rest, bones, from, from_end), Self::point(rest, bones, to, to_end));
            sum += (a + (b - a) * along) * m;
            mass += m;
        }
        sum / mass
    }

    /// Ankle joint of a side.
    fn ankle(&self, rest: &BodyMesh, bones: &[Affine3A], side: usize) -> Vec3 {
        Self::point(rest, bones, self.legs[side][2], End::Head)
    }
}

/// From rotation `a` towards `b` by `t`, the short way round. (`b` and
/// `-b` are the same rotation; blending towards the wrong one of the two
/// swings the long way, through everything in between.)
fn blend(a: Quat, b: Quat, t: f32) -> Quat {
    let b = if a.dot(b) < 0.0 { -b } else { b };
    a.slerp(b.normalize(), t.clamp(0.0, 1.0))
}

fn rotation(transform: &Affine3A) -> Quat {
    Quat::from_mat3a(&transform.matrix3).normalize()
}

/// Bends a leg so its ankle reaches `target` (world, before `offset`),
/// keeping the knee pointing the way the keyframes have it and the foot
/// turned as they have it. Writes the new rotations into `pose`, blended
/// with the keyframed ones by `weight`; the foot lies flat as far as
/// `flatten` says.
#[allow(clippy::too_many_arguments)]
fn reach(pose: &mut Pose, skeleton: &Skeleton, rest: &BodyMesh, bones: &[Affine3A], side: usize, target: Vec3, offset: Vec3, weight: f32, flatten: f32) {
    // A foot on the floor lies flat on it, turned the way the keyframes
    // turn it. The turn is read off the foot's sideways axis, which stays
    // level however far the toes point down (at toe-off the foot's length
    // points almost straight down, and its direction on the floor is
    // meaningless). Flat as far as `flatten` says; a foot leaving or
    // reaching the floor tilts between that and its keyframed angle.
    let foot = skeleton.legs[side][2];
    let across = bones[foot].transform_vector3(Vec3::X);
    let yaw = (-across.z).atan2(across.x);
    let foot_world = blend(rotation(&bones[foot]), Quat::from_rotation_y(yaw), flatten);
    reach_with(pose, skeleton, rest, bones, side, target, offset, weight, foot_world);
}

/// Like [`reach`], with the foot's orientation in the world given.
#[allow(clippy::too_many_arguments)]
fn reach_with(pose: &mut Pose, skeleton: &Skeleton, rest: &BodyMesh, bones: &[Affine3A], side: usize, target: Vec3, offset: Vec3, weight: f32, foot_world: Quat) {
    let [thigh, shin, foot] = skeleton.legs[side];
    let hip = Skeleton::point(rest, bones, thigh, End::Head) + offset;
    let knee = Skeleton::point(rest, bones, shin, End::Head) + offset;
    let ankle = Skeleton::point(rest, bones, foot, End::Head) + offset;
    let (upper, lower) = (hip.distance(knee), knee.distance(ankle));
    if upper < 1e-4 || lower < 1e-4 {
        return;
    }
    let reach = target - hip;
    // Feet land on soft knees and the hips sit a little low, so a leg is
    // never asked to reach past (or right up to) its full length, where
    // the knee would swing a long way for a tiny change of reach.
    // Near full stretch the knee angle changes wildly with the reach; ease
    // the reach in so a leg never snaps straight.
    let full = upper + lower;
    let (soft, most) = (0.97 * full, 0.99 * full);
    let wanted = reach.length();
    let eased = if wanted > soft { soft + (most - soft) * (1.0 - (-(wanted - soft) / (most - soft)).exp()) } else { wanted };
    let distance = eased.max((upper - lower).abs() + 1e-3);
    let along = reach.normalize_or(Vec3::NEG_Y);
    // Which way the knee points: along the thigh's own forward axis, so
    // it tracks over the toes however the hip is turned, and does not
    // flip as the leg straightens.
    let pole = bones[thigh].transform_vector3(Vec3::Z);
    let pole = (pole - along * pole.dot(along)).normalize_or(Vec3::Z);
    let cos = ((upper * upper + distance * distance - lower * lower) / (2.0 * upper * distance)).clamp(-1.0, 1.0);
    let new_knee = hip + (along * cos + pole * (1.0 - cos * cos).sqrt()) * upper;
    let new_ankle = hip + along * distance;

    let swing_thigh = Quat::from_rotation_arc((knee - hip) / upper, (new_knee - hip) / upper);
    let thigh_world = swing_thigh * rotation(&bones[thigh]);
    let shin_dir = swing_thigh * ((ankle - knee) / lower);
    let swing_shin = Quat::from_rotation_arc(shin_dir.normalize(), (new_ankle - new_knee).normalize_or(shin_dir));
    let shin_world = swing_shin * swing_thigh * rotation(&bones[shin]);

    // Back to rotations relative to each bone's parent. The bones between
    // (upperleg02, lowerleg02) are not rotated, so they share their
    // parent's rotation.
    let [thigh_parent, _, _] = skeleton.leg_parents[side];
    let locals = [
        rotation(&bones[thigh_parent]).inverse() * thigh_world,
        thigh_world.inverse() * shin_world,
        shin_world.inverse() * foot_world,
    ];
    for (name, local) in skeleton.names[side].iter().zip(locals) {
        pose.set(name, blend(pose.get(name), local, weight));
    }
}

/// Puts the feet where they belong: planted ones on their locks, a foot
/// coming down guided onto the floor, one lifting off eased from where it
/// stood onto its keyframed swing. Both blends are of positions, starting
/// and ending at rest, so a foot neither jumps nor whips.
#[allow(clippy::too_many_arguments)]
fn place_feet(
    pose: &mut Pose,
    skeleton: &Skeleton,
    rest: &BodyMesh,
    keyed: &[Affine3A],
    offset: Vec3,
    planted: &[(usize, Vec3, f32)],
    landing: &[(usize, Vec3, f32)],
    released: &[Option<(Vec3, f32)>; 2],
) {
    for &(side, lock, _) in planted {
        reach(pose, skeleton, rest, keyed, side, lock, offset, 1.0, 1.0);
    }
    for &(side, target, near) in landing {
        let swing = skeleton.ankle(rest, keyed, side) + offset;
        reach(pose, skeleton, rest, keyed, side, swing + (target - swing) * near, offset, near, near);
    }
    for (side, release) in released.iter().enumerate() {
        if let Some((lock, fade)) = release {
            // From where it stood, at rest, onto its keyframed swing.
            let swing = skeleton.ankle(rest, keyed, side) + offset;
            let left = smooth(*fade);
            reach(pose, skeleton, rest, keyed, side, swing + (*lock - swing) * left, offset, left, left);
        }
    }
}

/// Where a foot coming down will touch the floor: below where the leg
/// points, or `Reach` metres out from the hip in that direction.
fn landing_target(skeleton: &Skeleton, rest: &BodyMesh, bones: &[Affine3A], angles: &Angles, side: usize, offset: Vec3, ground: Option<f32>) -> Vec3 {
    let mut target = skeleton.ankle(rest, bones, side) + offset;
    let reach = angles.get(Joint::Reach(Side::BOTH[side]));
    if reach > 0.0 {
        let hip = Skeleton::point(rest, bones, skeleton.legs[side][0], End::Head) + offset;
        let out = Vec2::new(target.x - hip.x, target.z - hip.z).normalize_or(Vec2::X);
        target.x = hip.x + out.x * reach;
        target.z = hip.z + out.y * reach;
    }
    if let Some(floor) = ground {
        target.y = floor;
    }
    target
}

/// Which leg the weight rests on while standing at ease: +1 left, −1
/// right. People shift from one to the other every several seconds.
fn weight_side(t: f32) -> f32 {
    use std::f32::consts::TAU;
    (1.8 * ((t * TAU / 13.0 + 0.6).sin() + 0.25 * (t * TAU / 5.3).sin())).clamp(-1.0, 1.0)
}

/// Standing at ease (`idle` 0..=1): the weight on one leg, the other knee
/// soft, the hip dropping on the free side and the shoulders answering.
fn relax(a: &mut Angles, t: f32, idle: f32) {
    if idle < 1e-3 {
        return;
    }
    let side = weight_side(t);
    let free = if side > 0.0 { Side::Right } else { Side::Left };
    let amount = side.abs() * idle;
    a.add(Joint::Knee(free), 9.0 * amount);
    a.add(Joint::HipFlex(free), 4.0 * amount);
    a.add(Joint::HipTurn(free), 6.0 * amount);
    a.add(Joint::Ankle(free), -2.0 * amount);
    a.add(Joint::Elbow(free), 5.0 * amount);
    a.add(Joint::PelvisList, 3.5 * side * idle);
    a.add(Joint::SpineSide, 1.5 * side * idle);
    a.add(Joint::NeckSide, -1.2 * side * idle);
}

/// Postural sway: the centre of mass drifts a few millimetres over the
/// feet, slowly and irregularly; on one foot more, and faster.
fn sway(t: f32, single: f32) -> Vec3 {
    use std::f32::consts::TAU;
    let wave = |periods: [f32; 3], phase: f32| {
        (t * TAU / periods[0] + phase).sin() + 0.6 * (t * TAU / periods[1] + 2.0 * phase).sin() + 0.35 * (t * TAU / periods[2] + 3.0 * phase).sin()
    };
    let calm = Vec3::new(0.0035 * wave([6.7, 2.9, 1.3], 0.3), 0.0, 0.0055 * wave([8.3, 3.7, 1.7], 1.1));
    let wobble = Vec3::new(0.005 * wave([1.1, 0.71, 0.43], 2.0), 0.0, 0.004 * wave([1.3, 0.83, 0.52], 0.7));
    calm + wobble * single
}

/// One baked frame.
#[derive(Debug, Clone, Copy)]
struct Frame {
    /// Thigh, shin and foot rotations per side, after balancing.
    legs: [[Quat; 3]; 2],
    offset: Vec3,
    /// How much the body is standing at ease.
    idle: f32,
}

/// A motion played through once on a particular body, frame by frame,
/// with what cannot be decided one frame at a time: feet stay where they
/// were put down, and the body balances over them.
///
/// * A foot whose `Plant` weight is up is locked where it touched down,
///   and the leg reaches it (two-bone IK), so a stepping foot stays put.
/// * The hips move to keep the centre of mass over the feet that carry
///   the weight, so the weight shifts before a foot lifts.
/// * Standing still, the body sways and shifts its weight from leg to leg.
pub struct Performance {
    motion: Motion,
    frames: Vec<Frame>,
}

impl Performance {
    const STEP: f32 = 1.0 / 60.0;

    pub fn bake(motion: Motion, model: &BodyModel, rest: &BodyMesh) -> Self {
        let Some(skeleton) = Skeleton::new(model) else {
            return Self { motion, frames: Vec::new() };
        };
        if matches!(motion, Motion::Walk | Motion::StiffWalk | Motion::HipSwayWalk) {
            return Self::bake_walk(motion, &skeleton, model, rest);
        }
        let keys = motion.keys();
        let duration = motion.duration();
        let count = (duration / Self::STEP).ceil() as usize + 1;
        let mut frames = Vec::with_capacity(count);
        let mut locks: [Option<Vec3>; 2] = [None, None];
        // A foot that just let go: where it was, and how much the leg
        // still reaches for it (fading out as the swing takes over).
        let mut released: [Option<(Vec3, f32)>; 2] = [None, None];
        let mut previous = Vec3::ZERO;
        // The hips have mass: their path is smoothed by a critically damped
        // spring, so nothing can jerk them.
        let mut hips: Option<(Vec3, Vec3)> = None;
        // Height of a planted ankle: the floor, as the first foot down sets it.
        let mut ground: Option<f32> = None;
        let mut idle = keys.as_ref().map_or(0.0, |k| stillness(k, 0.0));
        for frame in 0..count {
            let t = frame as f32 * Self::STEP;
            if let Some(keys) = &keys {
                idle += (stillness(keys, t) - idle) * (1.0 - (-Self::STEP / 0.6).exp());
            }
            let mut angles = motion.angles(t);
            relax(&mut angles, t, idle);
            let mut pose = angles.to_pose(model, rest);
            let weights = Side::BOTH.map(|s| angles.get(Joint::Plant(s)).clamp(0.0, 1.0));
            let mut bones = model.bone_transforms(rest, &pose);

            // Feet touch down and lift off. A foot in contact stays put
            // completely; its weight only says how much of the body it
            // carries.
            for side in 0..2 {
                if weights[side] <= 0.01 {
                    if let Some(lock) = locks[side].take() {
                        released[side] = Some((lock, 1.0));
                    }
                    if let Some((_, fade)) = &mut released[side] {
                        // A walking foot swings away at once; a foot
                        // stepping in a pose lets go more slowly.
                        let time = if angles.get(Joint::Balance) < 0.5 { 0.25 } else { 0.6 };
                        *fade -= Self::STEP / time;
                        if *fade <= 0.0 {
                            released[side] = None;
                        }
                    }
                } else if locks[side].is_none() {
                    let mut lock = landing_target(&skeleton, rest, &bones, &angles, side, previous, None);
                    match ground {
                        Some(floor) => lock.y = floor,
                        None => ground = Some(lock.y),
                    }
                    locks[side] = Some(lock);
                    released[side] = None;
                }
            }
            // A foot about to touch down is brought onto the floor over the
            // last part of its swing: `(side, how close, 0..=1)`.
            let landing: Vec<(usize, f32)> = match ground {
                Some(_) => (0..2)
                    .filter(|&side| locks[side].is_none())
                    .filter_map(|side| {
                        let s = Side::BOTH[side];
                        let touches = |dt: f32| motion.angles(t + dt).get(Joint::Plant(s)) > 0.01;
                        let coarse = (1..=8).map(|k| k as f32 * 0.05).find(|&dt| touches(dt))?;
                        // The moment of touch-down, found exactly so the
                        // foot comes down smoothly.
                        let (mut early, mut late) = (coarse - 0.05, coarse);
                        for _ in 0..12 {
                            let mid = 0.5 * (early + late);
                            if touches(mid) { late = mid } else { early = mid }
                        }
                        // A walking foot is set down in a fifth of a second.
                        let window = if angles.get(Joint::Balance) < 0.5 { 0.3 } else { 0.4 };
                        Some((side, smooth((1.0 - late / window).clamp(0.0, 1.0))))
                    })
                    .collect(),
                None => Vec::new(),
            };
            let planted: Vec<(usize, Vec3, f32)> =
                (0..2).filter_map(|side| Some((side, locks[side]?, weights[side]))).collect();
            let mut offset = Vec3::new(angles.get(Joint::ShiftX), 0.0, angles.get(Joint::ShiftZ));
            if !planted.is_empty() {
                offset = previous;
                let total: f32 = planted.iter().map(|p| p.2).sum();
                let balance = angles.get(Joint::Balance).clamp(0.0, 1.0);
                // Where the weight goes: between the feet that carry it,
                // towards the standing leg when at ease, swaying.
                let mut support = planted.iter().map(|&(_, lock, w)| lock * w).sum::<Vec3>() / total;
                if planted.len() == 2 {
                    let side = weight_side(t);
                    let standing = if side > 0.0 { locks[0] } else { locks[1] };
                    if let Some(foot) = standing {
                        support += (foot - support) * (0.32 * side.abs() * idle * weights[0].min(weights[1]));
                    }
                }
                // Shifts move the weight, not the feet.
                support += Vec3::new(angles.get(Joint::ShiftX), 0.0, angles.get(Joint::ShiftZ));
                let single = (weights[0] - weights[1]).abs() * (1.0 - idle);
                let keyframed = model.bone_transforms(rest, &angles.to_pose(model, rest));
                support += sway(t, single) * balance.max(0.2);
                for _ in 0..3 {
                    // Feet where they were put down, the hips at the height
                    // the legs hold them...
                    let follow = planted.iter().map(|&(side, lock, w)| (lock - skeleton.ankle(rest, &bones, side)) * w).sum::<Vec3>() / total;
                    // ...and over the feet as far as the body balances.
                    let com = skeleton.centre_of_mass(rest, &bones);
                    let centred = support - com;
                    offset.x = follow.x + (centred.x - follow.x) * balance;
                    offset.z = follow.z + (centred.z - follow.z) * balance;
                    // The height: each planted leg wants to be as long as its
                    // keyframed knee makes it (a hair short of straight when
                    // standing: knees are never locked), reaching its foot
                    // where it really is. The straighter leg decides (a
                    // smooth maximum): in a lunge the back leg stays straight
                    // and the front knee bends as the hips sink.
                    let wanted: Vec<(f32, f32)> = planted
                        .iter()
                        .map(|&(side, lock, w)| {
                            let [thigh, _, foot] = skeleton.legs[side];
                            let hip = Skeleton::point(rest, &keyframed, thigh, End::Head);
                            let extension = (hip.distance(Skeleton::point(rest, &keyframed, foot, End::Head)) - 0.0008).max(0.05);
                            let across = Vec2::new(lock.x - hip.x - offset.x, lock.z - hip.z - offset.z).length().min(extension);
                            (lock.y + (extension * extension - across * across).sqrt() - hip.y, w)
                        })
                        .collect();
                    let sharpness = 80.0;
                    let top = wanted.iter().map(|h| h.0).fold(f32::MIN, f32::max);
                    offset.y = top + (wanted.iter().map(|&(y, w)| w * (sharpness * (y - top)).exp()).sum::<f32>() / total).ln() / sharpness
                        - angles.get(Joint::HipDrop);
                    let keyed = &keyframed;
                    // Low enough that every foot on (or reaching for) the
                    // floor can get there: a wide stance lowers the hips.
                    let landing_targets: Vec<(usize, Vec3, f32)> = landing
                        .iter()
                        .map(|&(side, near)| (side, landing_target(&skeleton, rest, keyed, &angles, side, offset, ground), near))
                        .collect();
                    // Feet on the floor hold the hips down fully; one just
                    // lifting off lets go of them gradually.
                    let holding: Vec<(usize, Vec3, f32)> = planted
                        .iter()
                        .map(|&(side, lock, _)| (side, lock, 1.0))
                        .chain(released.iter().enumerate().filter_map(|(side, r)| r.map(|(lock, fade)| (side, lock, smooth(fade)))))
                        .chain(landing_targets.iter().copied())
                        .collect();
                    for &(side, target, weight) in &holding {
                        let [thigh, shin, foot] = skeleton.legs[side];
                        let point = |bone: usize| Skeleton::point(rest, keyed, bone, End::Head);
                        let length = 0.99 * (point(thigh).distance(point(shin)) + point(shin).distance(point(foot)));
                        let hip = point(thigh) + offset;
                        // A foot further out than the leg reaches from here
                        // pulls the hips down until it does (a long step bends
                        // the standing knee).
                        let across = Vec3::new(target.x - hip.x, 0.0, target.z - hip.z).length().min(0.97 * length);
                        let highest = target.y + (length * length - across * across).sqrt() - point(thigh).y;
                        if highest < offset.y {
                            offset.y += (highest - offset.y) * weight;
                        }
                    }
                    pose = angles.to_pose(model, rest);
                    place_feet(&mut pose, &skeleton, rest, keyed, offset, &planted, &landing_targets, &released);
                    bones = model.bone_transforms(rest, &pose);
                }
                // Through the spring, then the legs reach the feet again from
                // where the hips actually are.
                let (position, velocity) = hips.get_or_insert((offset, Vec3::ZERO));
                let omega = std::f32::consts::TAU * 4.0;
                *velocity += (omega * omega * (offset - *position) - 2.0 * omega * *velocity) * Self::STEP;
                *position += *velocity * Self::STEP;
                offset = *position;
                let keyed = model.bone_transforms(rest, &angles.to_pose(model, rest));
                pose = angles.to_pose(model, rest);
                let landing_targets: Vec<(usize, Vec3, f32)> = landing
                    .iter()
                    .map(|&(side, near)| (side, landing_target(&skeleton, rest, &keyed, &angles, side, offset, ground), near))
                    .collect();
                place_feet(&mut pose, &skeleton, rest, &keyed, offset, &planted, &landing_targets, &released);
            } else {
                hips = None;
            }
            previous = offset;
            let legs = [0, 1].map(|side| skeleton.names[side].clone().map(|name| pose.get(&name)));
            frames.push(Frame { legs, offset, idle });
        }
        Self { motion, frames }
    }

    /// A walk around the runway. The feet are planned first, footprint by
    /// footprint along the path; the hips travel along it, swaying over the
    /// standing foot and rising and falling with each step; the legs reach
    /// the feet. A standing foot lands on its heel, lies flat and peels off
    /// over its toes; a swinging foot travels in an arc to its next print.
    fn bake_walk(motion: Motion, skeleton: &Skeleton, model: &BodyModel, rest: &BodyMesh) -> Self {
        use std::f32::consts::TAU;
        let plan = WalkPlan::of(motion);
        let period = plan.gait.period;
        let root = model.bone("root").map_or(Vec3::new(0.0, 0.9, 0.0), |i| rest.joints[i].head);
        // The foot's shape in the modelled pose: ankle, ball of the foot
        // and heel, relative to the ankle.
        let shape = Side::BOTH.map(|side| {
            let joint = rest.joints[skeleton.legs[side as usize][2]];
            let ankle = joint.head;
            let heel = Vec3::new(ankle.x, 0.0, ankle.z - 0.06);
            (ankle, Vec3::new(joint.tail.x, 0.0, joint.tail.z) - ankle, heel - ankle)
        });
        // Where the ankle is when the foot rests on `print` tilted by
        // `tilt` about `pivot`.
        let ankle_at = |side: Side, print: (Vec3, f32), tilt: f32, pivot: Pivot| -> Vec3 {
            let (ankle, toe, heel) = shape[side as usize];
            let turn = Quat::from_rotation_y(print.1);
            let flat_ankle = Vec3::new(print.0.x, ankle.y, print.0.z);
            let pivot_offset = match pivot {
                Pivot::Heel => heel,
                Pivot::Toe => toe,
                Pivot::Ankle => Vec3::ZERO,
            };
            let pivot_point = flat_ankle + turn * pivot_offset;
            pivot_point + turn * (Quat::from_rotation_x(tilt) * -pivot_offset)
        };
        let count = (plan.duration() / Self::STEP).ceil() as usize + 1;
        let mut frames = Vec::with_capacity(count);
        // The hips have mass: a spring smooths their path. It is wound up
        // over one lap first, so the loop starts as it ends.
        let mut spring: Option<(Vec3, Vec3)> = None;
        for frame in 0..2 * count {
            let t = (frame % count) as f32 * Self::STEP;
            let angles = motion.angles(t);
            let keyed = model.bone_transforms(rest, &angles.to_pose(model, rest));
            let phase = (t / period).fract();

            // The feet.
            let feet = Side::BOTH.map(|side| {
                let cycle = t / period - if side == Side::Left { 0.0 } else { 0.5 };
                let k = cycle.floor();
                let p = cycle - k;
                let (tilt, pivot) = foot_roll(p);
                if p < 0.62 {
                    let print = plan.footprint(side, k);
                    (ankle_at(side, print, tilt, pivot), print.1, tilt)
                } else {
                    // In the air, from where it pushed off to where it lands.
                    let u = (p - 0.62) / 0.38;
                    let (from, to) = (plan.footprint(side, k), plan.footprint(side, k + 1.0));
                    let start = ankle_at(side, from, 50f32.to_radians(), Pivot::Toe);
                    let end = ankle_at(side, to, -12f32.to_radians(), Pivot::Heel);
                    // From rest at toe-off (the push-off ends still) to rest on
                    // the next heel, every joint of the path smooth: forward
                    // travel done a little before the heel lands...
                    let travel = smoother((u / 0.85).min(1.0));
                    let mut ankle = start + (end - start) * travel;
                    // ...and the foot lifted early, as the knee folds straight
                    // after toe-off (highest about 40 % into the swing).
                    ankle.y += 0.08 * 0.82 / 0.109 * u.powf(1.3) * (1.0 - u).powi(2);
                    let mut turn = to.1 - from.1;
                    turn -= TAU * (turn / TAU).round();
                    (ankle, from.1 + turn * travel, tilt)
                }
            });

            // The hips: along the path, over the standing foot, up and down.
            let (centre, heading) = RUNWAY.at(plan.speed() * t);
            let left = Vec3::new(heading.cos(), 0.0, -heading.sin());
            let mut hips = centre + left * plan.sway * (TAU * phase).sin();
            hips.y = root.y - 0.012 - plan.bob * 0.5 * (1.0 + (2.0 * TAU * phase).cos());
            let mut offset = hips - root;
            // Never higher than the legs reach.
            for (side, &(ankle, _, _)) in feet.iter().enumerate() {
                let [thigh, shin, foot] = skeleton.legs[side];
                let point = |bone: usize| Skeleton::point(rest, &keyed, bone, End::Head);
                // A little short of straight: a leg pushing off is already
                // bending at the knee.
                let length = 0.955 * (point(thigh).distance(point(shin)) + point(shin).distance(point(foot)));
                let hip = point(thigh) + offset;
                let across = Vec2::new(ankle.x - hip.x, ankle.z - hip.z).length().min(0.97 * length);
                let highest = ankle.y + (length * length - across * across).sqrt() - point(thigh).y;
                // A soft minimum: no corner where the limit takes over.
                let k = 120.0;
                offset.y = -((-k * offset.y).exp() + (-k * highest).exp()).ln() / k;
            }
            // Along the path the target jumps by a lap at the wrap; follow it
            // relative to the path, which closes on itself.
            let (position, velocity) = spring.get_or_insert((offset, Vec3::ZERO));
            if position.distance(offset) > 1.0 {
                *position = offset;
            }
            let omega = TAU * 6.0;
            *velocity += (omega * omega * (offset - *position) - 2.0 * omega * *velocity) * Self::STEP;
            *position += *velocity * Self::STEP;
            let offset = *position;
            if frame < count {
                continue;
            }

            let mut pose = angles.to_pose(model, rest);
            for (side, &(ankle, yaw, tilt)) in feet.iter().enumerate() {
                let foot_world = Quat::from_rotation_y(yaw) * Quat::from_rotation_x(tilt);
                reach_with(&mut pose, skeleton, rest, &keyed, side, ankle, offset, 1.0, foot_world);
            }
            let legs = [0, 1].map(|side| skeleton.names[side].clone().map(|name| pose.get(&name)));
            frames.push(Frame { legs, offset, idle: 0.0 });
        }
        Self { motion, frames }
    }

    pub fn motion(&self) -> Motion {
        self.motion
    }

    /// The pose at `time` seconds (wrapped into the loop).
    pub fn pose(&self, time: f32, model: &BodyModel, rest: &BodyMesh) -> Pose {
        let t = time.rem_euclid(self.motion.duration());
        if self.frames.len() < 2 {
            return self.motion.angles(t).to_pose(model, rest);
        }
        let at = (t / Self::STEP).min((self.frames.len() - 1) as f32);
        let index = (at.floor() as usize).min(self.frames.len() - 2);
        let f = at - index as f32;
        let (a, b) = (&self.frames[index], &self.frames[index + 1]);
        let idle = a.idle + (b.idle - a.idle) * f;
        let mut angles = self.motion.angles(t);
        relax(&mut angles, t, idle);
        let mut pose = angles.to_pose(model, rest);
        for side in 0..2 {
            for (k, stem) in ["upperleg01", "lowerleg01", "foot"].iter().enumerate() {
                let name = format!("{stem}.{}", if side == 0 { "L" } else { "R" });
                pose.set(&name, blend(a.legs[side][k], b.legs[side][k], f));
            }
        }
        pose.offset = a.offset + (b.offset - a.offset) * f;
        pose
    }
}

/// How still and at ease the keyframed body is at `t`: standing on both
/// feet, legs straight, and barely moving.
fn stillness(keys: &Keys, t: f32) -> f32 {
    let (before, now, after) = (keys.sample((t - 0.3).max(0.0)), keys.sample(t), keys.sample(t + 0.3));
    let speed = (0..COUNT)
        .filter(|&i| !(3..=4).contains(&i))
        .map(|i| (after.0[i] - before.0[i]).abs() / 0.6)
        .fold(0.0, f32::max);
    let moving = 1.0 - smoothstep(4.0, 14.0, speed);
    let standing = Side::BOTH
        .iter()
        .map(|&s| {
            smoothstep(0.9, 1.0, now.get(Joint::Plant(s)))
                * (1.0 - smoothstep(10.0, 20.0, now.get(Joint::Knee(s))))
                * (1.0 - smoothstep(10.0, 18.0, now.get(Joint::HipSpread(s)).abs()))
                * (1.0 - smoothstep(10.0, 20.0, now.get(Joint::HipFlex(s)).abs()))
        })
        .product::<f32>();
    moving * standing * smoothstep(0.9, 1.0, now.get(Joint::Balance))
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    smooth(((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_pass_through_their_poses_without_overshoot() {
        let mut k = Keys::new();
        k.key(0.0, |a| {
            a.set(Joint::SpineFlex, 0.0);
        });
        k.key(1.0, |a| {
            a.set(Joint::SpineFlex, 30.0);
        });
        k.hold(2.0);
        k.key(3.0, |a| {
            a.set(Joint::SpineFlex, 10.0);
        });
        let flex = |t: f32| k.sample(t).get(Joint::SpineFlex);
        assert!((flex(1.0) - 30.0).abs() < 1e-3);
        assert!((flex(3.0) - 10.0).abs() < 1e-3);
        for step in 0..=300 {
            let v = flex(step as f32 * 0.01);
            assert!((-1e-3..=30.001).contains(&v), "{v}");
        }
    }

    #[test]
    fn every_motion_loops_seamlessly() {
        for motion in Motion::ALL {
            let d = motion.duration();
            let (start, end) = (motion.angles(0.0), motion.angles(d - 1e-3));
            for index in 0..COUNT {
                // Breathing is not tied to the loop; everything else is.
                let tolerance = 2.6;
                let mut difference = (start.0[index] - end.0[index]).abs();
                if index == Joint::Heading.index() {
                    // 0° and 360° face the same way.
                    difference = difference.rem_euclid(360.0).min(360.0 - difference.rem_euclid(360.0));
                }
                assert!(difference < tolerance, "{motion:?} joint {index}: {} vs {}", start.0[index], end.0[index]);
            }
        }
    }
}

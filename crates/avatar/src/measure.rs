//! Tape measurements taken off a body mesh.
//!
//! Girths are measured the way a tape would lie: the body is cut with a
//! plane, the cut is split into separate loops (torso, arms, legs), and the
//! girth is the perimeter of the convex hull of the relevant loop. Levels
//! such as "bust" or "waist" are not fixed heights; they are searched for
//! on the actual shape (fullest chest, narrowest torso, …), so they stay
//! meaningful for every body the model can produce.

use std::collections::HashMap;

use glam::{Vec2, Vec3};

use crate::model::{BodyMesh, BodyModel};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Measure {
    Height,
    Neck,
    Shoulders,
    Bust,
    Waist,
    Hips,
    Inseam,
    Thigh,
    Calf,
    Ankle,
    ArmLength,
    UpperArm,
    Wrist,
}

impl Measure {
    pub const ALL: [Measure; 13] = [
        Measure::Height,
        Measure::Neck,
        Measure::Shoulders,
        Measure::Bust,
        Measure::Waist,
        Measure::Hips,
        Measure::Inseam,
        Measure::Thigh,
        Measure::Calf,
        Measure::Ankle,
        Measure::ArmLength,
        Measure::UpperArm,
        Measure::Wrist,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Measure::Height => "Height",
            Measure::Neck => "Neck girth",
            Measure::Shoulders => "Shoulder width",
            Measure::Bust => "Bust / chest girth",
            Measure::Waist => "Waist girth",
            Measure::Hips => "Hip girth",
            Measure::Inseam => "Inseam (crotch height)",
            Measure::Thigh => "Thigh girth",
            Measure::Calf => "Calf girth",
            Measure::Ankle => "Ankle girth",
            Measure::ArmLength => "Arm length (shoulder to wrist)",
            Measure::UpperArm => "Upper arm girth",
            Measure::Wrist => "Wrist girth",
        }
    }

    /// How the measurement is taken, for tooltips.
    pub fn description(self) -> &'static str {
        match self {
            Measure::Height => "Floor to the top of the head.",
            Measure::Neck => "Smallest girth of the neck.",
            Measure::Shoulders => "Straight distance between the two shoulder points (the top of each shoulder joint).",
            Measure::Bust => "Largest horizontal girth of the chest below the armpits.",
            Measure::Waist => "Smallest horizontal girth of the torso between chest and hips.",
            Measure::Hips => "Largest horizontal girth around the seat.",
            Measure::Inseam => "Floor to the crotch.",
            Measure::Thigh => "Girth of one thigh just below the crotch.",
            Measure::Calf => "Largest girth of the lower leg.",
            Measure::Ankle => "Smallest girth of the lower leg above the foot.",
            Measure::ArmLength => "Shoulder point to the elbow joint, then on to the wrist joint.",
            Measure::UpperArm => "Largest girth around the middle of the upper arm.",
            Measure::Wrist => "Smallest girth of the forearm at the wrist joint.",
        }
    }
}

/// All measurements of one body, in metres (and kilograms).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Measurements {
    values: [f32; Measure::ALL.len()],
    /// Estimated from the body's volume at the density of water.
    pub weight_kg: f32,
    /// Heights at which the horizontal girths were taken.
    pub bust_level: f32,
    pub waist_level: f32,
    pub hip_level: f32,
    pub armpit_level: f32,
    /// Half the torso's width at its widest level below the armpits.
    pub torso_half_width: f32,
}

impl Measurements {
    /// What garment arrangement needs, estimated from proportions alone,
    /// for bodies without the landmarks real measuring relies on
    /// (imported avatars). The tape measurements stay zero.
    pub fn rough(mesh: &BodyMesh) -> Self {
        let height = mesh.height();
        // The torso's sides: the widest point near the centre line between
        // waist and chest, which leaves out arms held away from the body.
        let torso_half_width = mesh
            .positions
            .iter()
            .filter(|p| p.y > 0.55 * height && p.y < 0.7 * height && p.x.abs() < 0.13 * height)
            .map(|p| p.x.abs())
            .fold(0.08 * height, f32::max);
        let mut values = [0.0; Measure::ALL.len()];
        values[Measure::Height as usize] = height;
        Self {
            values,
            weight_kg: 0.0,
            bust_level: 0.72 * height,
            waist_level: 0.62 * height,
            hip_level: 0.52 * height,
            armpit_level: 0.75 * height,
            torso_half_width,
        }
    }

    pub fn get(&self, measure: Measure) -> f32 {
        self.values[measure as usize]
    }
}

/// One closed cut through the body.
#[derive(Debug, Clone, Copy)]
struct Loop {
    centroid: Vec3,
    /// Convex hull perimeter: what a tape pulled tight would read.
    girth: f32,
    /// Largest |x| of the loop.
    reach: f32,
}

struct Slicer<'a> {
    mesh: &'a BodyMesh,
    /// Vertical extent of every skin triangle, to reject most triangles of
    /// a horizontal cut with two comparisons.
    spans: Vec<(f32, f32)>,
}

impl<'a> Slicer<'a> {
    fn new(mesh: &'a BodyMesh) -> Self {
        let spans = mesh
            .skin()
            .iter()
            .map(|t| {
                let ys = t.map(|i| mesh.positions[i as usize].y);
                (ys[0].min(ys[1]).min(ys[2]), ys[0].max(ys[1]).max(ys[2]))
            })
            .collect();
        Self { mesh, spans }
    }

    fn horizontal(&self, y: f32) -> Vec<Loop> {
        self.cut(Vec3::new(0.0, y, 0.0), Vec3::Y, |t| y < self.spans[t].0 || y > self.spans[t].1)
    }

    fn plane(&self, origin: Vec3, normal: Vec3) -> Vec<Loop> {
        self.cut(origin, normal, |_| false)
    }

    fn cut(&self, origin: Vec3, normal: Vec3, skip: impl Fn(usize) -> bool) -> Vec<Loop> {
        let positions = &self.mesh.positions;
        // Every crossed mesh edge yields one point; the two crossed edges
        // of a triangle belong to the same loop.
        let mut edge_ids: HashMap<(u32, u32), usize> = HashMap::new();
        let mut points: Vec<Vec3> = Vec::new();
        let mut parent: Vec<usize> = Vec::new();
        fn find(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }

        for (index, triangle) in self.mesh.skin().iter().enumerate() {
            if skip(index) {
                continue;
            }
            let p = triangle.map(|i| positions[i as usize]);
            let d = p.map(|p| (p - origin).dot(normal));
            let above = d.map(|d| d >= 0.0);
            if above[0] == above[1] && above[1] == above[2] {
                continue;
            }
            let mut crossed = [usize::MAX; 2];
            let mut count = 0;
            for (a, b) in [(0, 1), (1, 2), (2, 0)] {
                if above[a] == above[b] {
                    continue;
                }
                let key = (triangle[a].min(triangle[b]), triangle[a].max(triangle[b]));
                let id = *edge_ids.entry(key).or_insert_with(|| {
                    points.push(p[a].lerp(p[b], d[a] / (d[a] - d[b])));
                    parent.push(parent.len());
                    parent.len() - 1
                });
                crossed[count] = id;
                count += 1;
            }
            let (a, b) = (find(&mut parent, crossed[0]), find(&mut parent, crossed[1]));
            parent[a] = b;
        }

        let (u, v) = normal.any_orthonormal_pair();
        let mut groups: HashMap<usize, Vec<Vec3>> = HashMap::new();
        for i in 0..points.len() {
            groups.entry(find(&mut parent, i)).or_default().push(points[i]);
        }
        groups
            .into_values()
            .filter(|group| group.len() >= 3)
            .map(|group| {
                let centroid = group.iter().sum::<Vec3>() / group.len() as f32;
                let flat: Vec<Vec2> = group.iter().map(|p| Vec2::new(p.dot(u), p.dot(v))).collect();
                Loop { centroid, girth: hull_perimeter(flat), reach: group.iter().map(|p| p.x.abs()).fold(0.0, f32::max) }
            })
            .collect()
    }

    /// The loop around the body's centre line at height `y`.
    fn torso(&self, y: f32) -> Option<Loop> {
        // Of the loops on the centre line the outermost is the skin; the
        // others are cavities inside it (mouth, nostrils).
        self.horizontal(y)
            .into_iter()
            .filter(|l| l.centroid.x.abs() < 0.05)
            .max_by(|a, b| a.girth.total_cmp(&b.girth))
    }

    /// The loop around the right leg at height `y`.
    fn leg(&self, y: f32) -> Option<Loop> {
        self.horizontal(y)
            .into_iter()
            .filter(|l| l.centroid.x > 0.02 && l.centroid.x < 0.35 && l.girth > 0.1)
            .min_by(|a, b| a.centroid.x.total_cmp(&b.centroid.x))
    }
}

/// Perimeter of the convex hull (Andrew's monotone chain).
fn hull_perimeter(mut points: Vec<Vec2>) -> f32 {
    points.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    let mut hull: Vec<Vec2> = Vec::with_capacity(points.len() * 2);
    for pass in 0..2 {
        let start = hull.len();
        let ordered: Box<dyn Iterator<Item = &Vec2>> = if pass == 0 { Box::new(points.iter()) } else { Box::new(points.iter().rev()) };
        for &p in ordered {
            while hull.len() >= start + 2 && (hull[hull.len() - 1] - hull[hull.len() - 2]).perp_dot(p - hull[hull.len() - 1]) <= 0.0 {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    (0..hull.len()).map(|i| hull[i].distance(hull[(i + 1) % hull.len()])).sum()
}

/// Evenly spaced samples of `from..=to`.
fn steps(from: f32, to: f32, step: f32) -> impl Iterator<Item = f32> {
    let count = ((to - from) / step).max(0.0) as usize;
    (0..=count).map(move |i| from + i as f32 * step)
}

/// The sample with the largest (`sign = 1`) or smallest (`sign = -1`) girth.
fn extreme(samples: impl Iterator<Item = (f32, Option<Loop>)>, sign: f32) -> Option<(f32, Loop)> {
    samples
        .filter_map(|(at, l)| Some((at, l?)))
        .max_by(|a, b| (a.1.girth * sign).total_cmp(&(b.1.girth * sign)))
}

/// Lowest height at which the arms are joined to the torso.
fn armpit_level(slicer: &Slicer<'_>, height: f32) -> f32 {
    steps(0.60 * height, 0.90 * height, 0.004)
        .find(|&y| !slicer.horizontal(y).iter().any(|l| l.centroid.x.abs() > 0.12 && l.girth > 0.1))
        .unwrap_or(0.75 * height)
}

/// Picks the shoulder points: the skin vertex on top of each shoulder
/// joint, slightly to the outside, where a tape would start. Index 0 is
/// the body's right (-x), index 1 its left (+x).
pub(crate) fn find_shoulder_points(model: &BodyModel, mesh: &BodyMesh) -> [u32; 2] {
    ["upperarm01.R", "upperarm01.L"].map(|bone| {
        let Some(joint) = model.bone(bone).map(|i| mesh.joints[i].head) else { return 0 };
        let above = Vec3::new(joint.x + 0.015 * joint.x.signum(), joint.y, joint.z);
        mesh.positions
            .iter()
            .enumerate()
            .filter(|(_, p)| p.y > joint.y && Vec2::new(p.x - above.x, p.z - above.z).length() < 0.02)
            .max_by(|a, b| a.1.y.total_cmp(&b.1.y))
            .map_or(0, |(i, _)| i as u32)
    })
}

/// Measures a body. `mesh` must be in the modelled pose (straight from
/// [`BodyModel::build`]): the levels are searched along the vertical.
pub fn measure(model: &BodyModel, mesh: &BodyMesh) -> Measurements {
    let mut m = Measurements::default();
    if mesh.positions.is_empty() {
        return m;
    }
    let slicer = Slicer::new(mesh);
    let positions = &mesh.positions;
    let height = mesh.height();
    let mut set = |measure: Measure, value: f32| m.values[measure as usize] = value;
    set(Measure::Height, height);

    // --- legs ---------------------------------------------------------------
    // The crotch is the lowest point on the body's centre line.
    let crotch = positions.iter().filter(|p| p.x.abs() < 0.004 && p.y > 0.2 * height).map(|p| p.y).fold(height, f32::min);
    set(Measure::Inseam, crotch);
    if let Some(l) = slicer.leg(crotch - 0.02) {
        set(Measure::Thigh, l.girth);
    }
    if let Some((_, l)) = extreme(steps(0.14 * height, 0.27 * height, 0.006).map(|y| (y, slicer.leg(y))), 1.0) {
        set(Measure::Calf, l.girth);
    }
    if let Some((_, l)) = extreme(steps(0.03 * height, 0.09 * height, 0.004).map(|y| (y, slicer.leg(y))), -1.0) {
        set(Measure::Ankle, l.girth);
    }

    // --- torso ----------------------------------------------------------------
    let armpit = armpit_level(&slicer, height);
    let torso = |from: f32, to: f32, sign: f32| extreme(steps(from, to, 0.006).map(|y| (y, slicer.torso(y))), sign);
    let hips = torso(crotch + 0.01, crotch + 0.14 * height, 1.0);
    let bust = torso(armpit - 0.11 * height, armpit - 0.006, 1.0);
    let hip_level = hips.map_or(crotch + 0.08 * height, |h| h.0);
    let bust_level = bust.map_or(armpit - 0.05 * height, |b| b.0);
    let waist = torso(hip_level + 0.05 * height, (bust_level - 0.04 * height).max(hip_level + 0.06 * height), -1.0);
    if let Some((_, l)) = hips {
        set(Measure::Hips, l.girth);
    }
    if let Some((_, l)) = bust {
        set(Measure::Bust, l.girth);
    }
    if let Some((_, l)) = waist {
        set(Measure::Waist, l.girth);
    }
    if let Some((_, l)) = torso(0.80 * height, 0.90 * height, -1.0) {
        set(Measure::Neck, l.girth);
    }

    // --- shoulders and arm ------------------------------------------------------
    // `right` is the +x side, the body's left.
    let [other, right] = mesh.shoulder_points.map(|i| positions[i as usize]);
    set(Measure::Shoulders, other.distance(right));

    // The arm is measured along the skeleton: girths are cut perpendicular
    // to the bone, lengths follow the joints.
    let joint = |name: &str| model.bone(name).map(|i| mesh.joints[i].head);
    if let (Some(shoulder), Some(elbow), Some(wrist)) = (joint("upperarm01.L"), joint("lowerarm01.L"), joint("wrist.L")) {
        set(Measure::ArmLength, right.distance(elbow) + elbow.distance(wrist));
        let around = |from: Vec3, to: Vec3, range: (f32, f32), sign: f32| {
            let (axis, length) = ((to - from).normalize_or(Vec3::NEG_Y), from.distance(to));
            let cut = |t: f32| {
                let on_axis = from + axis * t;
                slicer
                    .plane(on_axis, axis)
                    .into_iter()
                    // Skip fingers on one side and cuts that clip the torso
                    // on the other.
                    .filter(|l| l.girth > 0.08 && l.girth < 0.6)
                    .min_by(|a, b| a.centroid.distance(on_axis).total_cmp(&b.centroid.distance(on_axis)))
            };
            extreme(steps(range.0 * length, range.1 * length, length / 60.0).map(|t| (t, cut(t))), sign)
        };
        if let Some((_, l)) = around(shoulder, elbow, (0.35, 0.65), 1.0) {
            set(Measure::UpperArm, l.girth);
        }
        if let Some((_, l)) = around(elbow, wrist, (0.85, 1.0), -1.0) {
            set(Measure::Wrist, l.girth);
        }
    }

    // --- weight -------------------------------------------------------------------
    let volume: f32 = mesh
        .skin()
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|i| positions[i as usize]);
            a.dot(b.cross(c)) / 6.0
        })
        .sum();
    m.weight_kg = volume.abs() * 1000.0;
    (m.bust_level, m.waist_level, m.hip_level, m.armpit_level) = (bust_level, waist.map_or(0.0, |w| w.0), hip_level, armpit);
    m.torso_half_width = [hips, bust, waist].iter().flatten().map(|(_, l)| l.reach).fold(0.0, f32::max);
    m
}

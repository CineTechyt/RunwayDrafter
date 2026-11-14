//! Scripted scenes that put numbers on how worn cloth behaves while the
//! body moves: an arm raise, a side stretch, a walk and standing still.
//! Each prints one row per solver backend, so a change to a solver can be
//! judged against the others and against its own past.
//!
//! ```text
//! cargo test --release -p physics_xpbd --test scenes -- --ignored --nocapture
//! ```
//!
//! `SCENE_BACKENDS=cpu,contact,global` picks the backends, `SCENE_ONLY=arm`
//! the scenes (by a part of their name), `SCENE_DISTANCE=10` the particle
//! distance in millimetres.

use std::sync::Arc;
use std::time::Instant;

use avatar::motion::Joint;
use avatar::{
    Angles, BodyMesh, BodyModel, BodyParams, Motion, Performance, Pose, PoseParams, Side,
};
use cad_core::PatternDocument;
use glam::Vec3;
use physics_xpbd::XpbdSolver;
use topology_sync::{
    ClothSolver, DistanceField, GarmentMesh, GarmentTopology, SolverBackend, SolverContext,
    SolverSettings, TopologySync,
};

const FPS: f32 = 30.0;
const VOXEL: f32 = 0.01;
/// Frames of draping in the dressing pose before anything moves.
const DRAPE: usize = 75;
/// Frames of moving from the dressing pose into the scene's first pose.
const LEAD: usize = 45;

fn model() -> BodyModel {
    BodyModel::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/avatar/body.bin"
    ))
    .expect("run tools/bake_avatar.py")
}

fn smooth(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// 0 before `start`, 1 from `full` to `hold`, back to 0 at `end`.
fn pulse(t: f32, start: f32, full: f32, hold: f32, end: f32) -> f32 {
    smooth((t - start) / (full - start)) * (1.0 - smooth((t - hold) / (end - hold)))
}

fn both(a: &mut Angles, joint: fn(Side) -> Joint, value: f32) {
    a.set(joint(Side::Left), value);
    a.set(joint(Side::Right), value);
}

/// Both arms up the sides to overhead, held, and down again.
fn arm_raise(t: f32) -> Angles {
    let up = pulse(t, 0.5, 2.5, 4.0, 6.0);
    let mut a = Angles::default();
    both(&mut a, Joint::ArmRaise, 13.0 + (172.0 - 13.0) * up);
    both(&mut a, Joint::ArmTwist, 20.0 * up);
    both(&mut a, Joint::Elbow, 12.0 - 8.0 * up);
    a
}

/// The weight goes onto the left leg and the body bends to the left with
/// the right arm reaching over: the right side of the garment is the
/// stretched one.
fn side_stretch(t: f32) -> Angles {
    let lean = pulse(t, 0.5, 2.5, 4.0, 6.0);
    let mut a = Angles::default();
    a.set(Joint::ShiftX, 0.06 * lean);
    a.set(Joint::PelvisList, 6.0 * lean);
    a.set(Joint::SpineSide, 28.0 * lean);
    a.set(Joint::ArmRaise(Side::Right), 13.0 + (165.0 - 13.0) * lean);
    a.set(Joint::ArmTwist(Side::Right), 20.0 * lean);
    // The left hand comes away from the side the body bends over, as it
    // would: kept by it, it would sink into the thigh.
    a.set(Joint::ArmRaise(Side::Left), 13.0 + LEFT_ARM * lean);
    a
}

/// How far the left arm comes out in the side stretch, degrees.
const LEFT_ARM: f32 = 25.0;

fn standing(_: f32) -> Angles {
    Angles::default()
}

enum Script {
    Angles(fn(f32) -> Angles),
    Motion(Motion),
}

struct Scene {
    name: &'static str,
    seconds: f32,
    script: Script,
    /// When the pose is at its extreme, seconds: where the probes are read.
    extreme: f32,
}

const SCENES: [Scene; 4] = [
    Scene {
        name: "settle",
        seconds: 3.0,
        script: Script::Angles(standing),
        extreme: 3.0,
    },
    Scene {
        name: "arm raise",
        seconds: 7.0,
        script: Script::Angles(arm_raise),
        extreme: 4.0,
    },
    Scene {
        name: "side stretch",
        seconds: 7.0,
        script: Script::Angles(side_stretch),
        extreme: 4.0,
    },
    Scene {
        name: "walk",
        seconds: 4.0,
        script: Script::Motion(Motion::Walk),
        extreme: 4.0,
    },
];

/// Cloth edges (vertex pairs) with their length in the pattern, metres.
fn edges(mesh: &GarmentMesh) -> Vec<(u32, u32, f32)> {
    let mut edges: Vec<(u32, u32)> = mesh
        .indices
        .chunks_exact(3)
        .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
        .map(|(a, b)| (a.min(b), a.max(b)))
        .collect();
    edges.sort_unstable();
    edges.dedup();
    edges
        .into_iter()
        .map(|(a, b)| {
            let rest = mesh.pattern_positions[a as usize]
                .distance(mesh.pattern_positions[b as usize])
                * 1e-3;
            (a, b, rest)
        })
        .collect()
}

/// `(largest, 99th percentile)` of the edges' elongation.
fn stretch(positions: &[Vec3], edges: &[(u32, u32, f32)]) -> (f32, f32) {
    let mut strains: Vec<f32> = edges
        .iter()
        .map(|&(a, b, rest)| positions[a as usize].distance(positions[b as usize]) / rest - 1.0)
        .collect();
    strains.sort_by(f32::total_cmp);
    (
        *strains.last().unwrap(),
        strains[(strains.len() * 99 / 100).min(strains.len() - 1)],
    )
}

/// Segment `p`–`q` against triangle `a, b, c`.
fn crosses(p: Vec3, q: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    let (e1, e2, d) = (b - a, c - a, q - p);
    let h = d.cross(e2);
    let det = e1.dot(h);
    if det.abs() < 1e-12 {
        return false;
    }
    let s = p - a;
    let u = s.dot(h) / det;
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let qv = s.cross(e1);
    let v = d.dot(qv) / det;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    let t = e2.dot(qv) / det;
    (0.0..=1.0).contains(&t)
}

/// How many cloth edges pass through a cloth triangle they do not touch.
fn self_crossings(mesh: &GarmentMesh, edges: &[(u32, u32, f32)]) -> usize {
    use std::collections::HashMap;
    const CELL: f32 = 0.03;
    let cell = |p: Vec3| (p / CELL).floor().as_ivec3();
    let mut grid: HashMap<glam::IVec3, Vec<u32>> = HashMap::new();
    let x = &mesh.positions;
    for (index, t) in mesh.indices.chunks_exact(3).enumerate() {
        let [a, b, c] = [x[t[0] as usize], x[t[1] as usize], x[t[2] as usize]];
        let (low, high) = (cell(a.min(b).min(c)), cell(a.max(b).max(c)));
        for i in low.x..=high.x {
            for j in low.y..=high.y {
                for k in low.z..=high.z {
                    grid.entry(glam::IVec3::new(i, j, k))
                        .or_default()
                        .push(index as u32);
                }
            }
        }
    }
    // Sewn vertices lie on each other: an edge does not cross the
    // triangles of what it is sewn to.
    let mut sewn: Vec<Vec<u32>> = vec![Vec::new(); x.len()];
    for s in &mesh.stitches {
        sewn[s.a as usize].push(s.b);
        sewn[s.b as usize].push(s.a);
    }
    let touches =
        |tri: &[u32], v: u32| tri.contains(&v) || sewn[v as usize].iter().any(|s| tri.contains(s));
    let mut count = 0;
    let mut seen: Vec<u32> = Vec::new();
    for &(a, b, _) in edges {
        let (p, q) = (x[a as usize], x[b as usize]);
        let (low, high) = (cell(p.min(q)), cell(p.max(q)));
        seen.clear();
        for i in low.x..=high.x {
            for j in low.y..=high.y {
                for k in low.z..=high.z {
                    let Some(list) = grid.get(&glam::IVec3::new(i, j, k)) else {
                        continue;
                    };
                    for &t in list {
                        if seen.contains(&t) {
                            continue;
                        }
                        seen.push(t);
                        let tri = &mesh.indices[t as usize * 3..t as usize * 3 + 3];
                        if touches(tri, a) || touches(tri, b) {
                            continue;
                        }
                        let [u, v, w] =
                            [x[tri[0] as usize], x[tri[1] as usize], x[tri[2] as usize]];
                        count += usize::from(crosses(p, q, u, v, w));
                    }
                }
            }
        }
    }
    count
}

/// What one run of a scene measured.
#[derive(Default)]
struct Report {
    /// Milliseconds of solving per frame: mean and worst.
    ms: (f32, f32),
    /// Edge elongation over the whole scene: worst, and the worst frame's
    /// 99th percentile.
    stretch: (f32, f32),
    /// Cloth vertices inside the body, worst frame, and how deep (m).
    inside: (usize, f32),
    /// Cloth edges through cloth triangles, worst sampled frame.
    crossings: usize,
    /// Vertices held between two sides of the body, worst frame (global
    /// solver only).
    pinched: usize,
    /// Vertices inside the skin where it meets or overlaps itself, worst
    /// frame.
    trapped: usize,
    /// At the extreme pose: the hem's rise on the avatar's right and left,
    /// and the right underarm point's, metres.
    hem_rise: [f32; 2],
    underarm_rise: f32,
    /// At the extreme pose: how much further up the arm than the shoulder
    /// joint the cloth reaches, metres (cloth carried up by the arm).
    up_the_arm: f32,
    /// After the scene: how far the garment ended from where it started
    /// (mean over vertices, metres), and how much it still moves per frame
    /// (95th percentile, metres).
    returned: f32,
    still: f32,
}

struct Probes {
    hem: [Vec<u32>; 2],
    underarm: u32,
}

impl Probes {
    fn new(mesh: &GarmentMesh) -> Self {
        let x = &mesh.positions;
        let low = x.iter().map(|p| p.y).fold(f32::MAX, f32::min);
        let high = x.iter().map(|p| p.y).fold(f32::MIN, f32::max);
        let hem = |sign: f32| {
            (0..x.len() as u32)
                .filter(|&i| x[i as usize].y < low + 0.02 && x[i as usize].x * sign > 0.05)
                .collect::<Vec<u32>>()
        };
        // The top of the right side seam.
        let underarm = mesh
            .stitches
            .iter()
            .map(|s| s.a)
            .filter(|&i| x[i as usize].x < -0.08 && x[i as usize].y < high - 0.10)
            .max_by(|&a, &b| x[a as usize].y.total_cmp(&x[b as usize].y))
            .unwrap_or(0);
        Self {
            // The avatar's right is −X.
            hem: [hem(-1.0), hem(1.0)],
            underarm,
        }
    }

    fn heights(&self, x: &[Vec3]) -> ([f32; 2], f32) {
        let mean = |list: &Vec<u32>| {
            list.iter().map(|&i| x[i as usize].y).sum::<f32>() / list.len().max(1) as f32
        };
        (
            [mean(&self.hem[0]), mean(&self.hem[1])],
            x[self.underarm as usize].y,
        )
    }
}

fn run(scene: &Scene, backend: SolverBackend, distance: f32) -> Report {
    let model = model();
    let params = BodyParams::female();
    let rest = model.build(&params);
    let measurements = avatar::measure(&model, &rest);
    let dressing = PoseParams::default().to_pose(&model, &rest);
    let dressed = model.pose(&rest, &dressing);

    let mut document = PatternDocument::demo_top();
    avatar::arrange_around(&mut document, &dressed, &measurements);
    // `SCENE_STRETCH=0.9`: how stiff the fabric is against stretch (the
    // material panel's sliders, 0..1).
    if let Some(stiffness) = std::env::var("SCENE_STRETCH")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
    {
        let fabrics: Vec<_> = document.pieces().iter().map(|p| p.fabric).collect();
        for fabric in fabrics {
            if let Some(f) = document.fabric_mut(fabric) {
                f.physics.stretch_weft = stiffness;
                f.physics.stretch_warp = stiffness;
                f.physics.shear = stiffness * 0.7;
            }
        }
    }
    let mut topology = GarmentTopology::default();
    topology.set_params(topology_sync::MeshingParams {
        particle_distance: distance,
        ..topology.params()
    });
    topology.rebuild(&document);

    let performance = match scene.script {
        Script::Motion(motion) => Some(Performance::bake(motion, &model, &rest)),
        Script::Angles(_) => None,
    };
    let pose_at = |t: f32| -> Pose {
        match (&scene.script, &performance) {
            (Script::Angles(angles), _) => angles(t).to_pose(&model, &rest),
            (Script::Motion(_), Some(p)) => p.pose(t, &model, &rest),
            _ => unreachable!(),
        }
    };
    let first = pose_at(0.0);
    // The body at a frame counted from the start of the lead-in.
    let body_at = |frame: f32| -> BodyMesh {
        if frame <= LEAD as f32 {
            let t = smooth(frame / LEAD as f32);
            let mut pose = Pose::rest();
            for bone in model.bones() {
                let (a, b) = (dressing.get(&bone.name), first.get(&bone.name));
                let b = if a.dot(b) < 0.0 { -b } else { b };
                pose.set(&bone.name, a.slerp(b, t));
            }
            pose.offset = first.offset;
            model.pose(&rest, &pose)
        } else {
            model.pose(&rest, &pose_at((frame - LEAD as f32) / FPS))
        }
    };

    let settings = SolverSettings {
        backend,
        ..SolverSettings::default()
    };
    let region = |mesh: &GarmentMesh| {
        let (low, high) = mesh
            .positions
            .iter()
            .fold((Vec3::MAX, Vec3::MIN), |(lo, hi), &p| {
                (lo.min(p), hi.max(p))
            });
        (low - Vec3::splat(0.15), high + Vec3::splat(0.15))
    };
    let field_of = |body: &BodyMesh, mesh: &GarmentMesh, velocities: Option<&[Vec3]>| {
        Arc::new(avatar::distance_field_within(
            body,
            VOXEL,
            0.0,
            &avatar::FieldOptions {
                region: Some(region(mesh)),
                velocities,
                deep: false,
            },
        ))
    };

    // Dressed and draped where the scene starts.
    let start = Vec3::new(first.offset.x, 0.0, first.offset.z);
    for p in &mut topology.mesh_mut().positions {
        *p += start;
    }
    let mut body = body_at(0.0);
    let mut field: Arc<DistanceField> = Arc::new(avatar::distance_field(&body, VOXEL, 0.0));
    let mut collider = Arc::new(body.collider());
    let mut solver = XpbdSolver::new();
    let dt = 1.0 / FPS;
    macro_rules! ctx {
        ($dt:expr) => {
            SolverContext {
                document: &document,
                settings: &settings,
                colliders: &[],
                body: Some(&field),
                body_mesh: Some(&collider),
                ground: 0.0,
                dt: $dt,
            }
        };
    }
    solver.rebind(topology.mesh(), &ctx!(dt));
    for _ in 0..DRAPE {
        solver.step(topology.mesh_mut(), &ctx!(dt));
    }

    let edges = edges(topology.mesh());
    let probes = Probes::new(topology.mesh());
    let mut report = Report::default();
    let frames = LEAD + (scene.seconds * FPS) as usize;
    let extreme = LEAD + (scene.extreme * FPS) as usize;
    let mut before: Option<(Vec<Vec3>, [f32; 2], f32)> = None;
    let mut total_ms = 0.0;
    let mut last = topology.mesh().positions.clone();
    for frame in 1..=frames {
        let next = body_at(frame as f32);
        // Like the application's bake: a frame in which a body part moves
        // far against the rest is split.
        let (low, high) = region(topology.mesh());
        let moves: Vec<Vec3> = body
            .positions
            .iter()
            .zip(&next.positions)
            .filter(|(a, _)| a.cmpge(low).all() && a.cmple(high).all())
            .map(|(a, b)| *b - *a)
            .collect();
        let mean = moves.iter().copied().sum::<Vec3>() / moves.len().max(1) as f32;
        let moved = moves
            .iter()
            .map(|m| m.distance(mean))
            .fold(0.0f32, f32::max);
        let pieces = ((moved / 0.012).ceil() as usize).clamp(1, 6);
        let mut frame_ms = 0.0;
        for piece in 1..=pieces {
            let then = if piece == pieces {
                next.clone()
            } else {
                body_at(frame as f32 - 1.0 + piece as f32 / pieces as f32)
            };
            let step = dt / pieces as f32;
            // The cloth travels with the body as a whole.
            let travelled = avatar::travelled(&body, &then, (low, high));
            solver.travel(topology.mesh_mut(), travelled, step);
            let velocities: Vec<Vec3> = body
                .positions
                .iter()
                .zip(&then.positions)
                .map(|(a, b)| (*b - *a - travelled) / step)
                .collect();
            body = then;
            field = field_of(&body, topology.mesh(), Some(&velocities));
            collider = Arc::new(
                body.collider()
                    .with_velocities(velocities.iter().map(|&v| v.into()).collect()),
            );
            let clock = Instant::now();
            solver.step(topology.mesh_mut(), &ctx!(step));
            frame_ms += clock.elapsed().as_secs_f32() * 1e3;
        }
        total_ms += frame_ms;
        report.ms.1 = report.ms.1.max(frame_ms);

        if backend == SolverBackend::Global {
            report.pinched = report.pinched.max(solver.global_stats().pinched);
        }
        let x = &topology.mesh().positions;
        let (worst, typical) = stretch(x, &edges);
        report.stretch.0 = report.stretch.0.max(worst);
        report.stretch.1 = report.stretch.1.max(typical);
        // Against the skin itself: inside, unless squeezed where two
        // surfaces of the body meet or overlap (the armpit with the arm
        // down), where no place is outside both.
        let (mut inside, mut deepest, mut trapped) = (0, 0.0f32, 0);
        let margin = settings.collision_margin;
        for p in x {
            let [near, other] = collider.sides((*p).into(), 0.03);
            let Some(near) = near else { continue };
            if near.distance >= 0.0 {
                continue;
            }
            let squeezed = other.is_some_and(|o| o.distance + near.distance < 2.0 * margin);
            if squeezed {
                trapped += 1;
            } else {
                inside += 1;
                deepest = deepest.max(-near.distance);
            }
        }
        report.trapped = report.trapped.max(trapped);
        if inside > report.inside.0 {
            report.inside.0 = inside;
        }
        report.inside.1 = report.inside.1.max(deepest);
        if std::env::var("SCENE_TRACE").is_ok() {
            eprintln!(
                "    frame {frame:3} ({pieces} pieces, {frame_ms:6.1} ms): stretch {:6.1}% p99 {:5.1}%, {inside:3} inside {:5.1} mm",
                worst * 100.0,
                typical * 100.0,
                deepest * 1e3
            );
        }
        if std::env::var("SCENE_TRACE").is_ok() && backend == SolverBackend::Global {
            let s = solver.global_stats();
            eprintln!(
                "      {} factorisations, {} halvings, last moved {:.2} mm",
                s.factorisations,
                s.halvings,
                s.last_step * 1e3
            );
        }
        if frame % 5 == 0 {
            report.crossings = report
                .crossings
                .max(self_crossings(topology.mesh(), &edges));
        }
        if frame == LEAD {
            let (hem, underarm) = probes.heights(x);
            before = Some((x.clone(), hem, underarm));
        }
        if std::env::var("SCENE_DUMP")
            .ok()
            .and_then(|f| f.parse::<usize>().ok())
            == Some(frame)
        {
            let mut worst: Vec<(f32, u32, u32)> = edges
                .iter()
                .map(|&(a, b, rest)| (x[a as usize].distance(x[b as usize]) / rest - 1.0, a, b))
                .collect();
            worst.sort_by(|a, b| b.0.total_cmp(&a.0));
            for &(strain, a, b) in worst.iter().take(8) {
                let [near, other] = collider.sides(x[a as usize].into(), 0.03);
                eprintln!(
                    "    {:5.1}%  {a}-{b} at {:.3?} - {:.3?}, skin {:?} mm, other side {:?} mm",
                    strain * 100.0,
                    x[a as usize],
                    x[b as usize],
                    near.map(|s| (s.distance * 1e3).round()),
                    other.map(|s| (s.distance * 1e3).round()),
                );
            }
            for name in ["upperarm01.R", "upperarm01.L", "spine01"] {
                if let Some(bone) = model.bone(name) {
                    eprintln!("    joint {name} at {:.3?}", body.joints[bone].head);
                }
            }
        }
        if frame == extreme
            && let Some((_, hem, underarm)) = &before
        {
            let (now_hem, now_underarm) = probes.heights(x);
            if std::env::var("SCENE_WHERE").is_ok() {
                let mesh = topology.mesh();
                let mut worst: Vec<(f32, u32, u32)> = edges
                    .iter()
                    .map(|&(a, b, rest)| (x[a as usize].distance(x[b as usize]) / rest - 1.0, a, b))
                    .collect();
                worst.sort_by(|a, b| b.0.total_cmp(&a.0));
                for &(strain, a, b) in worst.iter().take(14) {
                    let stitched = |v: u32| mesh.stitches.iter().any(|s| s.a == v || s.b == v);
                    eprintln!(
                        "    {:5.1}%  {:?}{} - {:?}{}  pattern {:?} - {:?}  at {:.2?}",
                        strain * 100.0,
                        mesh.origins[a as usize],
                        if stitched(a) { " sewn" } else { "" },
                        mesh.origins[b as usize],
                        if stitched(b) { " sewn" } else { "" },
                        mesh.pattern_positions[a as usize].round(),
                        mesh.pattern_positions[b as usize].round(),
                        x[a as usize]
                    );
                }
                // Where the stretched edges are: height bands of 5 cm.
                let mut bands: std::collections::BTreeMap<i32, (usize, usize, f32)> =
                    Default::default();
                for &(a, b, rest) in &edges {
                    let (p, q) = (x[a as usize], x[b as usize]);
                    let strain = p.distance(q) / rest - 1.0;
                    let band = bands.entry(((p.y + q.y) * 10.0) as i32).or_default();
                    band.0 += 1;
                    if strain > 0.05 {
                        band.1 += 1;
                        band.2 += (p.x + q.x).abs() * 0.5;
                    }
                }
                for (band, (count, stretched, side)) in bands {
                    eprintln!(
                        "    y {:.2}: {stretched:4} of {count:4} edges over 5 %, mean |x| {:.2}",
                        band as f32 * 0.1,
                        side / stretched.max(1) as f32
                    );
                }
            }
            report.hem_rise = [now_hem[0] - hem[0], now_hem[1] - hem[1]];
            report.underarm_rise = now_underarm - underarm;
            // Along the right upper arm, from the shoulder joint.
            if let (Some(upper), Some(lower)) =
                (model.bone("upperarm01.R"), model.bone("lowerarm01.R"))
            {
                let (shoulder, elbow) = (body.joints[upper].head, body.joints[lower].head);
                let along = (elbow - shoulder).normalize_or_zero();
                report.up_the_arm = x
                    .iter()
                    .filter(|p| p.x < 0.0)
                    .filter(|&&p| (p - shoulder).reject_from(along).length() < 0.09)
                    .map(|&p| (p - shoulder).dot(along))
                    .fold(0.0f32, f32::max);
            }
        }
        if frame == frames {
            if let Some((start, _, _)) = &before {
                report.returned = x
                    .iter()
                    .zip(start)
                    .map(|(a, b)| a.distance(*b))
                    .sum::<f32>()
                    / x.len() as f32;
            }
            let mut motion: Vec<f32> = x.iter().zip(&last).map(|(a, b)| a.distance(*b)).collect();
            motion.sort_by(f32::total_cmp);
            report.still = motion[motion.len() * 95 / 100];
        }
        last.clone_from(x);
    }
    report.ms.0 = total_ms / frames as f32;
    if backend == SolverBackend::Global {
        let s = solver.global_stats();
        eprintln!(
            "  last step: {} factorisations {} us, solves {} us, assembly {} us (contact search {} us), {} contacts, {} halvings, moved {:.5}",
            s.factorisations,
            s.factor_us,
            s.solve_us,
            s.assemble_us,
            s.detect_us,
            s.contacts,
            s.halvings,
            s.last_step
        );
    }
    report
}

fn backends() -> Vec<(&'static str, SolverBackend)> {
    let wanted = std::env::var("SCENE_BACKENDS").unwrap_or_else(|_| "cpu,contact,global".into());
    [
        ("cpu", SolverBackend::Cpu),
        ("contact", SolverBackend::Contact),
        ("global", SolverBackend::Global),
    ]
    .into_iter()
    .filter(|(name, _)| wanted.split(',').any(|w| w.trim() == *name))
    .collect()
}

#[test]
#[ignore = "a benchmark: run it by name"]
fn scenes() {
    let only = std::env::var("SCENE_ONLY").ok();
    let distance = std::env::var("SCENE_DISTANCE")
        .ok()
        .and_then(|d| d.parse().ok())
        .unwrap_or(20.0);
    println!(
        "{:<13}{:<9}{:>8}{:>8}{:>9}{:>8}{:>8}{:>8}{:>8}{:>7}{:>7}{:>9}{:>9}{:>9}{:>8}{:>9}{:>9}",
        "scene",
        "solver",
        "ms/frm",
        "worst",
        "stretch",
        "p99",
        "inside",
        "deep mm",
        "trapped",
        "cross",
        "pinch",
        "hem R cm",
        "hem L cm",
        "arm cm",
        "up arm",
        "back cm",
        "still mm"
    );
    for scene in &SCENES {
        if only
            .as_ref()
            .is_some_and(|o| !scene.name.contains(o.as_str()))
        {
            continue;
        }
        for (name, backend) in backends() {
            let r = run(scene, backend, distance);
            println!(
                "{:<13}{:<9}{:>8.1}{:>8.1}{:>8.1}%{:>7.1}%{:>8}{:>8.1}{:>8}{:>7}{:>7}{:>9.1}{:>9.1}{:>9.1}{:>8.1}{:>9.1}{:>9.3}",
                scene.name,
                name,
                r.ms.0,
                r.ms.1,
                r.stretch.0 * 100.0,
                r.stretch.1 * 100.0,
                r.inside.0,
                r.inside.1 * 1e3,
                r.trapped,
                r.crossings,
                r.pinched,
                r.hem_rise[0] * 100.0,
                r.hem_rise[1] * 100.0,
                r.underarm_rise * 100.0,
                r.up_the_arm * 100.0,
                r.returned * 100.0,
                r.still * 1e3,
            );
        }
    }
}

/// How deep the left hand reaches into the rest of the body during the
/// side stretch: skin of the hand inside the torso or thigh.
#[test]
#[ignore = "a diagnostic"]
fn side_stretch_hand_against_hip() {
    let model = model();
    let rest = model.build(&BodyParams::female());
    let arm = model.arm_vertices();
    for t in [0.0f32, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0] {
        let body = model.pose(&rest, &side_stretch(t).to_pose(&model, &rest));
        // The rest of the body without the arms, as a collider.
        let others: Vec<[u32; 3]> = body
            .skin()
            .iter()
            .filter(|t| t.iter().all(|&v| !arm[v as usize]))
            .copied()
            .collect();
        let collider = physics_collision::TriangleMesh::new(
            body.positions.iter().map(|&p| p.into()).collect(),
            others,
        );
        let mut deepest = 0.0f32;
        let mut count = 0;
        for (v, p) in body.positions.iter().enumerate() {
            // The left hand and forearm (+X), below the elbow height.
            if !arm[v] || p.x < 0.0 || p.y > 1.05 {
                continue;
            }
            if let [Some(near), _] = collider.sides((*p).into(), 0.03)
                && near.distance < 0.0
            {
                count += 1;
                deepest = deepest.max(-near.distance);
            }
        }
        println!(
            "t {t:.1}: {count} hand vertices inside the body, deepest {:.1} mm",
            deepest * 1e3
        );
    }
}

//! Simulating a worn garment through an animation of the avatar.
//!
//! The garment is not pinned to the skin: it is simulated frame by frame
//! against the body as it is in that frame, on a worker thread, and the
//! result is played back. Per frame:
//!
//! * the body is posed and a collision field is built around the garment
//!   (only that region, which keeps it fast), plus the triangle mesh for
//!   the continuous test;
//! * the body moves through the cloth; collision and friction push fabric
//!   away from it while preserving sliding and momentum;
//! * the solver steps once at the animation's frame rate.
//!
//! First the body moves, over a second and a half, from the pose the
//! garment was draped on into the motion's first pose, and holds it half a
//! second while the garment settles: jumping there in one frame would put
//! arms straight through the cloth. The last frames blend into the first
//! so the loop is seamless.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use avatar::{BodyMesh, BodyModel, Motion, Performance, Pose};
use cad_core::PatternDocument;
use glam::Vec3;
use physics_xpbd::XpbdSolver;
use topology_sync::{ClothSolver, GarmentMesh, SolverContext, SolverSettings};

/// Frames per second of a simulated garment.
pub const FPS: f32 = 30.0;
/// Grid spacing of the per-frame collision field, metres.
const VOXEL: f32 = 0.01;
/// How far the per-frame collision field is smoothed over small gaps.
/// None: a hand by the hip or thighs passing each other leave small gaps
/// that smoothing would fill, and cloth in them would be pushed out round
/// the far side and caught.
const BAKE_CLOSING: f32 = 0.0;

/// A field around the garment, of a body moving by small steps (see the
/// sub-frames below), so no deep interior is needed.
fn field_options(region: (Vec3, Vec3), velocities: Option<&[Vec3]>) -> avatar::FieldOptions<'_> {
    avatar::FieldOptions {
        region: Some(region),
        velocities,
        deep: false,
    }
}

/// A garment simulated through a motion.
pub struct GarmentMotion {
    pub motion: Motion,
    /// What it was simulated for: the body's shape and the document.
    pub shape_version: u64,
    pub document_revision: u64,
    /// Garment vertex positions, `FPS` per second over the motion.
    pub frames: Vec<Vec<Vec3>>,
    /// How far the fabric was stretched beyond its cut length: the worst
    /// frame's 99th percentile, as a fraction.
    pub stretch: f32,
}

/// Beyond this stretch (see [`GarmentMotion::stretch`]) the simulation
/// went wrong somewhere (cloth caught between body parts) and is not shown.
pub const RELIABLE_STRETCH: f32 = 0.35;

impl GarmentMotion {
    /// The garment at `time` seconds (wrapped into the loop).
    pub fn positions(&self, time: f32, out: &mut [Vec3]) {
        let n = self.frames.len();
        if n == 0 {
            return;
        }
        let at = (time.rem_euclid(self.motion.duration()) * FPS).min(n as f32 - 1.0);
        let k = at.floor() as usize;
        let (a, b, f) = (&self.frames[k], &self.frames[(k + 1) % n], at.fract());
        for ((out, a), b) in out.iter_mut().zip(a).zip(b) {
            *out = *a + (*b - *a) * f;
        }
    }
}

pub enum Message {
    /// Fraction done.
    Progress(f32),
    Done(GarmentMotion),
}

/// Everything the worker needs, owned.
pub struct Job {
    pub motion: Motion,
    pub shape_version: u64,
    pub document: PatternDocument,
    pub mesh: GarmentMesh,
    pub model: BodyModel,
    pub rest: BodyMesh,
    /// The pose the garment is draped on: the body moves from it into the
    /// motion's first pose before the motion starts.
    pub start_pose: Pose,
    pub settings: SolverSettings,
    pub closing: f32,
    pub duration: Option<f32>,
}

/// Starts simulating on a worker thread.
pub fn start(job: Job) -> Receiver<Message> {
    let (sender, receiver) = channel();
    std::thread::spawn(move || run(job, &sender));
    receiver
}

fn run(job: Job, sender: &Sender<Message>) {
    let Job {
        motion,
        shape_version,
        document,
        mut mesh,
        model,
        rest,
        start_pose,
        settings,
        closing,
        duration,
    } = job;
    let performance = Performance::bake(motion, &model, &rest);
    let pose_at = |t: f32| model.pose_with_transforms(&rest, &performance.pose(t, &model, &rest));
    // Into the first pose, then a moment to settle.
    let (moving, settling) = ((1.5 * FPS) as usize, (0.5 * FPS) as usize);
    let lead = moving + settling;
    let first = performance.pose(0.0, &model, &rest);
    let lead_in = |frame: f32| -> BodyMesh {
        let t = (frame / moving as f32).clamp(0.0, 1.0);
        let t = t * t * (3.0 - 2.0 * t);
        let mut pose = Pose::rest();
        for bone in model.bones() {
            let (a, b) = (start_pose.get(&bone.name), first.get(&bone.name));
            let b = if a.dot(b) < 0.0 { -b } else { b };
            pose.set(&bone.name, a.slerp(b, t));
        }
        // Already standing where the motion starts (see below).
        pose.offset = first.offset;
        model.pose(&rest, &pose)
    };
    let count = (duration.unwrap_or_else(|| motion.duration()) * FPS)
        .ceil()
        .max(1.0) as usize;
    let dt = 1.0 / FPS;

    // The garment stays as draped; the body moves under it (never carried
    // along with the skin, which would glue the cloth to the avatar).
    // A motion may start somewhere else on the floor (a walk, at the start
    // of its runway). The worn garment goes there with the avatar, as if it
    // had been dressed there; from then on only the simulation moves it.
    let start = Vec3::new(
        first.offset.x - start_pose.offset.x,
        0.0,
        first.offset.z - start_pose.offset.z,
    );
    for p in &mut mesh.positions {
        *p += start;
    }
    let mut body = lead_in(0.0);
    mesh.recompute_normals();
    let (worst, typical) = stretch(&mesh, std::slice::from_ref(&mesh.positions));
    log::info!(
        "garment as draped: edges stretched at most {:.1} %, 99 % of them within {:.1} %",
        worst * 100.0,
        typical * 100.0
    );

    let region = |mesh: &GarmentMesh| {
        let (low, high) = mesh
            .positions
            .iter()
            .fold((Vec3::MAX, Vec3::MIN), |(lo, hi), &p| {
                (lo.min(p), hi.max(p))
            });
        (low - Vec3::splat(0.15), high + Vec3::splat(0.15))
    };
    let mut field = Arc::new(avatar::distance_field_within(
        &body,
        VOXEL,
        BAKE_CLOSING.min(closing),
        &field_options(region(&mesh), None),
    ));
    let mut collider = Arc::new(body.collider());
    let mut solver = XpbdSolver::new();
    let context = |field: &Arc<_>,
                   collider: &Arc<_>|
     -> (
        Arc<topology_sync::DistanceField>,
        Arc<physics_collision::TriangleMesh>,
    ) { (Arc::clone(field), Arc::clone(collider)) };
    {
        let (f, c) = context(&field, &collider);
        let ctx = SolverContext {
            document: &document,
            settings: &settings,
            colliders: &[],
            body: Some(&f),
            body_mesh: Some(&c),
            ground: 0.0,
            dt,
        };
        solver.rebind(&mesh, &ctx);
    }

    let mut frames = Vec::with_capacity(count);
    // The body at a (fractional) frame, or `None` while it holds still.
    let body_at = |frame: f32| -> Option<BodyMesh> {
        if frame <= 0.0 || (frame > moving as f32 && frame <= lead as f32 - 1.0) {
            None
        } else if frame <= moving as f32 {
            Some(lead_in(frame))
        } else {
            Some(pose_at((frame - lead as f32).max(0.0) / FPS).0)
        }
    };
    // Where the time goes: posing, collision field, collision mesh, solver.
    let mut timing = [std::time::Duration::ZERO; 4];
    for frame in 0..lead + count {
        let clock = std::time::Instant::now();
        // A body part that moves far in one frame (a leg swinging out) would
        // land deep inside cloth it swept through, and push it out on the
        // wrong side. Such frames are split so that nothing near the garment
        // moves more than about a centimetre at a time.
        let next = body_at(frame as f32);
        timing[0] += clock.elapsed();
        let pieces = match &next {
            Some(next) => {
                // How far parts move against the body as a whole: walking
                // carries everything along, which friction handles; a leg
                // swinging out is what can sweep through cloth.
                let (low, high) = region(&mesh);
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
                ((moved / 0.012).ceil() as usize).clamp(1, 6)
            }
            None => 1,
        };
        // Holding still: the skin is not moving any more.
        if next.is_none() && !field.velocity.is_empty() {
            field = Arc::new(avatar::distance_field_within(
                &body,
                VOXEL,
                BAKE_CLOSING.min(closing),
                &field_options(region(&mesh), None),
            ));
        }
        for piece in 1..=pieces {
            if let Some(next) = &next {
                let clock = std::time::Instant::now();
                let then = if piece == pieces {
                    next.clone()
                } else {
                    body_at(frame as f32 - 1.0 + piece as f32 / pieces as f32)
                        .unwrap_or_else(|| next.clone())
                };
                timing[0] += clock.elapsed();
                let clock = std::time::Instant::now();
                // The skin's velocity, for friction with the moving body.
                let step = dt / pieces as f32;
                // The garment travels with its wearer: only the body's
                // motion against itself is left to the collision.
                let travelled = avatar::travelled(&body, &then, region(&mesh));
                solver.travel(&mut mesh, travelled, step);
                let velocities: Vec<Vec3> = body
                    .positions
                    .iter()
                    .zip(&then.positions)
                    .map(|(a, b)| (*b - *a - travelled) / step)
                    .collect();
                body = then;
                field = Arc::new(avatar::distance_field_within(
                    &body,
                    VOXEL,
                    BAKE_CLOSING.min(closing),
                    &field_options(region(&mesh), Some(&velocities)),
                ));
                timing[1] += clock.elapsed();
                let clock = std::time::Instant::now();
                collider = Arc::new(
                    body.collider()
                        .with_velocities(velocities.iter().map(|&v| v.into()).collect()),
                );
                timing[2] += clock.elapsed();
            }
            let clock = std::time::Instant::now();
            let (f, c) = context(&field, &collider);
            let ctx = SolverContext {
                document: &document,
                settings: &settings,
                colliders: &[],
                body: Some(&f),
                body_mesh: Some(&c),
                ground: 0.0,
                dt: dt / pieces as f32,
            };
            solver.step(&mut mesh, &ctx);
            timing[3] += clock.elapsed();
        }
        mesh.recompute_normals();
        // CLO360_STRETCH_DEBUG=1: stretch and contacts after every frame.
        if std::env::var_os("CLO360_STRETCH_DEBUG").is_some() {
            let (worst, typical) = stretch(&mesh, std::slice::from_ref(&mesh.positions));
            let stats = solver.stats();
            eprintln!(
                "step {frame:3} ({pieces} parts): worst {:6.1} %, p99 {:5.1} %, body stops {}, cloth contacts {}, crossings {}",
                worst * 100.0,
                typical * 100.0,
                stats.body_stops,
                stats.cloth_contacts,
                stats.cloth_crossings
            );
            // Where the worst edge is.
            let p = &mesh.positions;
            if let Some((a, b, e)) = mesh
                .indices
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
                .map(|(a, b)| {
                    let r = mesh.pattern_positions[a as usize]
                        .distance(mesh.pattern_positions[b as usize])
                        * 1e-3;
                    (
                        a,
                        b,
                        if r > 1e-5 {
                            p[a as usize].distance(p[b as usize]) / r - 1.0
                        } else {
                            0.0
                        },
                    )
                })
                .max_by(|x, y| x.2.total_cmp(&y.2))
            {
                eprintln!(
                    "  worst edge {a}-{b} {:.0} % at {:.2?}",
                    e * 100.0,
                    p[a as usize]
                );
            }
        }
        if frame >= lead {
            frames.push(mesh.positions.clone());
        }
        if sender
            .send(Message::Progress(
                (frame + 1) as f32 / (lead + count) as f32,
            ))
            .is_err()
        {
            return; // nobody is waiting any more
        }
    }

    // How far the fabric was stretched beyond its cut length anywhere, at
    // any moment: the sign of cloth caught on (rather than sliding over)
    // the body.
    log::info!(
        "garment simulation time: posing {:.1} s, collision fields {:.1} s, collision meshes {:.1} s, cloth {:.1} s",
        timing[0].as_secs_f32(),
        timing[1].as_secs_f32(),
        timing[2].as_secs_f32(),
        timing[3].as_secs_f32()
    );
    let (worst, typical) = stretch(&mesh, &frames);
    log::info!(
        "garment through {motion:?}: edges stretched at most {:.1} %, 99 % of them within {:.1} %",
        worst * 100.0,
        typical * 100.0
    );

    // The loop's end blends into its start.
    let blend = (FPS * 0.4) as usize;
    if frames.len() > 2 * blend {
        let n = frames.len();
        let first = frames[0].clone();
        for k in 0..blend {
            let w = (k + 1) as f32 / (blend + 1) as f32;
            let w = w * w * (3.0 - 2.0 * w);
            for (p, q) in frames[n - blend + k].iter_mut().zip(&first) {
                *p += (*q - *p) * w;
            }
        }
    }
    let _ = sender.send(Message::Done(GarmentMotion {
        motion,
        shape_version,
        document_revision: document.revision(),
        frames,
        stretch: typical,
    }));
}

/// The worst stretch of any mesh edge over all frames, and the 99th
/// percentile of each frame's edges (the worst of those), as fractions of
/// the edge's length in the pattern.
fn stretch(mesh: &GarmentMesh, frames: &[Vec<Vec3>]) -> (f32, f32) {
    let mut edges: Vec<(u32, u32)> = mesh
        .indices
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
        .map(|(a, b)| (a.min(b), a.max(b)))
        .collect();
    edges.sort_unstable();
    edges.dedup();
    let rest: Vec<f32> = edges
        .iter()
        .map(|&(a, b)| {
            mesh.pattern_positions[a as usize].distance(mesh.pattern_positions[b as usize]) * 1e-3
        })
        .collect();
    let (mut worst, mut typical) = (0.0f32, 0.0f32);
    let mut strains = Vec::with_capacity(edges.len());
    for positions in frames {
        strains.clear();
        strains.extend(
            edges
                .iter()
                .zip(&rest)
                .filter(|(_, r)| **r > 1e-5)
                .map(|(&(a, b), r)| {
                    positions[a as usize].distance(positions[b as usize]) / r - 1.0
                }),
        );
        if strains.is_empty() {
            continue;
        }
        strains.sort_unstable_by(f32::total_cmp);
        worst = worst.max(*strains.last().unwrap());
        typical = typical.max(strains[(strains.len() as f32 * 0.99) as usize]);
    }
    (worst, typical)
}

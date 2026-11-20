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
//!
//! A walk is two steps that start over a stride further on. It is walked
//! several times in a row, the body carrying on forwards, until the cloth
//! swings the same way every stride; the last stride is kept, moved back
//! to where the loop starts. Played in a loop, it does not jump.

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

/// Strides walked before the one that is kept, so the cloth has settled
/// into the rhythm of the walk.
const WARM_UP_LOOPS: usize = 2;

/// `body` moved by `by`.
fn shifted(mut body: BodyMesh, by: Vec3) -> BodyMesh {
    if by != Vec3::ZERO {
        for p in &mut body.positions {
            *p += by;
        }
        for joint in &mut body.joints {
            joint.head += by;
            joint.tail += by;
        }
    }
    body
}

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
    /// Garment vertex positions over one loop of the motion, evenly spaced
    /// (about `FPS` per second).
    pub frames: Vec<Vec<Vec3>>,
    /// The time the frames cover, seconds: the whole motion, unless only
    /// its beginning was simulated.
    pub seconds: f32,
    /// How much further the fabric was stretched than it was as draped:
    /// the worst frame's 99th percentile less the draped garment's, as a
    /// fraction of the cut length. (A garment cut too tight for its wearer
    /// is stretched before it moves at all.)
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
        let duration = self.motion.duration();
        // Between the last frame and the next loop's first too: the loop
        // is `n` frames long, not `n - 1`.
        let at =
            (time.rem_euclid(duration) / self.seconds.max(1e-6) * n as f32).min(n as f32 - 1e-3);
        let k = at.floor() as usize;
        let (a, b, f) = (&self.frames[k], &self.frames[(k + 1) % n], at.fract());
        // Past the last frame the loop starts over, a walk a stride on.
        let ahead = if k + 1 == n {
            self.motion.travel()
        } else {
            Vec3::ZERO
        };
        for ((out, a), b) in out.iter_mut().zip(a).zip(b) {
            *out = *a + (*b + ahead - *a) * f;
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
        .round()
        .max(1.0) as usize;
    // A motion that moves on (a walk) is walked a few times over first.
    let travel = motion.travel();
    let warm_up = if duration.is_none() && travel != Vec3::ZERO {
        WARM_UP_LOOPS
    } else {
        0
    };
    let kept_from = lead + warm_up * count;
    let total = kept_from + count;
    // Whole loops take whole numbers of frames: a frame lasts a little more
    // or less than 1/FPS, so the kept frames are exactly one loop.
    let frame_time = match duration {
        Some(_) => 1.0 / FPS,
        None => motion.duration() / count as f32,
    };
    let dt = frame_time;

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
    let draped = typical;
    log::info!(
        "garment as draped: edges stretched at most {:.1} %, 99 % of them within {:.1} %",
        worst * 100.0,
        typical * 100.0
    );

    if std::env::var_os("CLO360_STRETCH_DEBUG").is_some() {
        for span in &mesh.pieces {
            let name = document
                .piece(span.piece)
                .map(|p| p.name.clone())
                .unwrap_or_default();
            let mut strains: Vec<f32> = mesh
                .indices
                .as_chunks::<3>()
                .0
                .iter()
                .filter(|t| span.vertices.contains(&t[0]))
                .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
                .filter_map(|(a, b)| {
                    let r = mesh.pattern_positions[a as usize]
                        .distance(mesh.pattern_positions[b as usize])
                        * 1e-3;
                    (r > 1e-5).then(|| {
                        mesh.positions[a as usize].distance(mesh.positions[b as usize]) / r - 1.0
                    })
                })
                .collect();
            strains.sort_unstable_by(f32::total_cmp);
            if !strains.is_empty() {
                eprintln!(
                    "  piece {name}: {} vertices, median {:.1} %, p90 {:.1} %, p99 {:.1} %, max {:.1} %",
                    span.vertices.len(),
                    strains[strains.len() / 2] * 100.0,
                    strains[strains.len() * 9 / 10] * 100.0,
                    strains[strains.len() * 99 / 100] * 100.0,
                    strains[strains.len() - 1] * 100.0
                );
            }
        }
    }
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
    // The global solver collides with the skin's triangles; it would only
    // use the distance field to split steps where the body moves fast,
    // which the bake does itself (see the parts below). Rebuilding the
    // field for every part was 40 % of the bake.
    let needs_field = settings.backend != topology_sync::SolverBackend::Global;
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
    // Frames faded over where the loop closes.
    let blend = ((FPS * if warm_up > 0 { 0.25 } else { 0.4 }) as usize).min(count / 3);
    let mut before: Vec<Vec<Vec3>> = Vec::with_capacity(blend);
    // The body at a (fractional) frame, or `None` while it holds still.
    let body_at = |frame: f32| -> Option<BodyMesh> {
        if frame <= 0.0 || (frame > moving as f32 && frame <= lead as f32 - 1.0) {
            None
        } else if frame <= moving as f32 {
            Some(lead_in(frame))
        } else {
            // Each time round, a stride further on.
            // Which time round and how far into it, both from the frame
            // count, so they always agree.
            let f = (frame - lead as f32).max(0.0);
            let loops = (f / count as f32).floor();
            let into = (f - loops * count as f32) * frame_time;
            Some(shifted(pose_at(into).0, travel * loops))
        }
    };
    // Where the time goes: posing, collision field, collision mesh, solver.
    let mut timing = [std::time::Duration::ZERO; 4];
    // CLO360_STRETCH_DEBUG: the waistband's vertices, to tell how much it
    // moves on its own.
    let band: Vec<usize> = mesh
        .pieces
        .iter()
        .filter(|span| {
            document
                .piece(span.piece)
                .is_some_and(|p| p.name.to_lowercase().contains("band"))
        })
        .flat_map(|span| span.vertices.start as usize..span.vertices.end as usize)
        .collect();
    for frame in 0..total {
        let clock = std::time::Instant::now();
        let frame_start = mesh.positions.clone();
        let mut carried = Vec3::ZERO;
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
        if needs_field && next.is_none() && !field.velocity.is_empty() {
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
                carried += travelled;
                solver.travel(&mut mesh, travelled, step);
                let velocities: Vec<Vec3> = body
                    .positions
                    .iter()
                    .zip(&then.positions)
                    .map(|(a, b)| (*b - *a - travelled) / step)
                    .collect();
                body = then;
                if needs_field {
                    field = Arc::new(avatar::distance_field_within(
                        &body,
                        VOXEL,
                        BAKE_CLOSING.min(closing),
                        &field_options(region(&mesh), Some(&velocities)),
                    ));
                }
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
            // How far the cloth moved this frame, apart from being carried
            // along with its wearer, mm.
            let moved = |indices: &mut dyn Iterator<Item = usize>| {
                let mut m: Vec<f32> = indices
                    .map(|i| (mesh.positions[i] - frame_start[i] - carried).length() * 1000.0)
                    .collect();
                m.sort_unstable_by(f32::total_cmp);
                if m.is_empty() {
                    return [0.0; 3];
                }
                [m[m.len() / 2], m[m.len() * 95 / 100], m[m.len() - 1]]
            };
            let all = moved(&mut (0..mesh.positions.len()));
            let waist = moved(&mut band.iter().copied());
            eprintln!(
                "  moved mm: all p50 {:.2} p95 {:.2} max {:.1}; band p50 {:.2} p95 {:.2} max {:.1}",
                all[0], all[1], all[2], waist[0], waist[1], waist[2]
            );
            eprintln!(
                "step {frame:3} ({pieces} parts): worst {:6.1} %, p99 {:5.1} %, body stops {}, cloth contacts {}, crossings {}",
                worst * 100.0,
                typical * 100.0,
                stats.body_stops,
                stats.cloth_contacts,
                stats.cloth_crossings
            );
            let g = solver.global_stats();
            let depths: Vec<f32> = mesh
                .positions
                .iter()
                .filter_map(|&p| collider.sides(p.into(), 0.08)[0].map(|side| side.distance))
                .collect();
            eprintln!(
                "  global: contacts {}, pinched {}, factorisations {}, halvings {}, last step {:.2} mm; inside skin: {} (> 5 mm: {}), deepest {:.1} mm, near skin {}",
                g.contacts,
                g.pinched,
                g.factorisations,
                g.halvings,
                g.last_step * 1000.0,
                depths.iter().filter(|d| **d < 0.0).count(),
                depths.iter().filter(|d| **d < -0.005).count(),
                depths.iter().copied().fold(0.0f32, f32::min) * -1000.0,
                depths.len()
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
        // The previous stride's end: what the kept stride's end fades
        // into (see below).
        if warm_up > 0 && frame + blend >= kept_from && frame < kept_from {
            let back = travel * (warm_up - 1) as f32;
            before.push(
                mesh.positions
                    .iter()
                    .map(|&p| p - back)
                    .collect::<Vec<Vec3>>(),
            );
        }
        if frame >= kept_from {
            // Back to where the loop starts.
            let back = travel * warm_up as f32;
            frames.push(mesh.positions.iter().map(|&p| p - back).collect());
        }
        if sender
            .send(Message::Progress((frame + 1) as f32 / total as f32))
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

    if warm_up > 0 && before.len() == blend && frames.len() > 2 * blend {
        // The cloth went from the previous stride's end straight into the
        // kept stride's start: fade the kept stride's end into the previous
        // one's, and the loop closes where the simulation itself went on.
        let n = frames.len();
        let gaps: Vec<f32> = frames[n - 1]
            .iter()
            .zip(&before[blend - 1])
            .map(|(a, b)| a.distance(*b))
            .collect();
        log::info!(
            "loop seam: one stride later the cloth is {:.1} mm (worst {:.1} mm) from where it was",
            gaps.iter().sum::<f32>() / gaps.len().max(1) as f32 * 1000.0,
            gaps.iter().copied().fold(0.0f32, f32::max) * 1000.0
        );
        for (k, earlier) in before.iter().enumerate() {
            let w = (k + 1) as f32 / blend as f32;
            let w = w * w * (3.0 - 2.0 * w);
            for (p, q) in frames[n - blend + k].iter_mut().zip(earlier) {
                *p += (*q - *p) * w;
            }
        }
    } else if frames.len() > 2 * blend {
        // A motion that ends where it began: its end blends into its start.
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
        seconds: count as f32 * frame_time,
        stretch: (typical - draped).max(0.0),
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

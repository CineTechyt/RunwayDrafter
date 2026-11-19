//! The simulation state manager: owns the model and the dynamic state,
//! picks the backend and implements `topology_sync::ClothSolver`.

use std::sync::Arc;
use std::time::Instant;

use glam::{Vec3, Vec3A};
use physics_collision::{ClothContacts, ContactParams};
use rayon::prelude::*;
use topology_sync::{
    ClothSolver, DistanceField, GarmentMesh, SolverBackend, SolverContext, SolverStats,
};

use crate::cpu::{self, Ellipsoid, State, SubstepParams};
use crate::global::Global;
use crate::model::Model;
use crate::ogc::{Ogc, StepInput};

/// Speed limit in m/s; far above anything cloth does under gravity.
const MAX_SPEED: f32 = 20.0;
const MAX_SUBSTEPS: u32 = 64;
/// Iterations per sub-step of the contact solver.
const CONTACT_ITERATIONS: usize = 2;
/// Newton steps per part of a step of the global solver, at most.
const GLOBAL_ITERATIONS: usize = 3;
/// The global solver splits a step so that the body moves no further
/// than this in a part, metres...
const GLOBAL_STRIDE: f32 = 0.004;
/// ...into this many parts at most.
const GLOBAL_MAX_SUBSTEPS: usize = 4;
/// The most of the body's acceleration the cloth is left to feel, m/s².
/// Beyond it (a body that starts walking from one frame to the next) the
/// cloth is simply taken along.
const MAX_FELT_ACCELERATION: f32 = 6.0;
/// Rounds of body points against cloth faces per step.
const PUSH_ROUNDS: usize = 3;
/// How far behind a cloth face a body point may be and still lift it,
/// in metres: less than the thinnest body part the cloth meets.
const PUSH_DEPTH: f32 = 0.01;
/// How far the global solver's cloth may always be moved back when it is
/// found on the wrong side of other cloth, metres (see `step_global`).
const MIN_PUT_BACK: f32 = 0.003;

/// XPBD cloth solver with a CPU backend and, when a device is supplied, a
/// compute-shader backend. The backend is chosen per step from
/// `SolverSettings::backend` and can be switched while the cloth is moving.
#[derive(Default)]
pub struct XpbdSolver {
    model: Model,
    state: State,
    colliders: Vec<Ellipsoid>,
    /// The avatar's body mesh as a distance field.
    body: Option<Arc<DistanceField>>,
    /// Seconds simulated since the seams started closing.
    seam_clock: f32,
    scratch: Vec<[Vec3A; 4]>,
    /// Cloth-against-cloth contacts, detected once per step.
    contacts: ClothContacts,
    /// The cloth triangles, for body points against cloth faces.
    triangles: Vec<[u32; 3]>,
    /// The contact solver's topology and contacts.
    ogc: Ogc,
    /// The global solver's matrix.
    global: Global,
    /// How fast the frame the cloth is simulated in moves, see
    /// [`XpbdSolver::travel`].
    frame_velocity: Vec3A,
    /// The GPU holds contact planes that are not all empty.
    planes_on_gpu: bool,
    /// Positions at the start of the step, for the continuous tests.
    step_start: Vec<Vec3A>,
    /// Where the solver left the cloth, before any was put back.
    contact_scratch: Vec<Vec3A>,
    #[cfg(feature = "gpu")]
    gpu: Option<crate::gpu::GpuBackend>,
    /// Which side holds the authoritative velocities.
    motion_on_gpu: bool,
    /// GPU particle buffers match `state` (false after CPU steps).
    gpu_current: bool,
    /// Soft area grab: vertex, target, influence radius and per-step strength.
    grab: Option<(u32, Vec3A, f32, f32)>,
    /// The cloth the grab moves, with how much of the move each vertex
    /// gets: found along the fabric from the grabbed vertex.
    grab_patch: Vec<(u32, f32)>,
    /// The vertices held in the hand (the middle of the patch) and their
    /// own inverse masses: while held they are immovable to the solver,
    /// so the cloth around follows them rather than pulling them back.
    grab_held: Vec<(u32, f32)>,
    /// Vertex → neighbours along the fabric (triangle edges, with their
    /// cut length in metres, and stitches, of length zero).
    fabric_graph: Vec<Vec<(u32, f32)>>,
    stats: SolverStats,
}

impl XpbdSolver {
    /// A CPU-only solver.
    pub fn new() -> Self {
        Self::default()
    }

    /// A solver that can also run on the GPU. The device must support
    /// compute shaders.
    #[cfg(feature = "gpu")]
    pub fn with_gpu(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            gpu: Some(crate::gpu::GpuBackend::new(device, queue)),
            ..Self::default()
        }
    }

    /// Moves the cloth to `mesh.positions` keeping its motion: where it
    /// was carried along (by the body it rests on), it carries on as it
    /// was moving. Unlike [`ClothSolver::set_positions`], which leaves it
    /// at rest.
    pub fn carry_along(&mut self, mesh: &GarmentMesh) {
        if mesh.vertex_count() != self.state.positions.len() {
            return;
        }
        self.pull_motion_from_gpu();
        for ((position, previous), &new) in self
            .state
            .positions
            .iter_mut()
            .zip(&mut self.state.previous)
            .zip(&mesh.positions)
        {
            let delta = Vec3A::from(new) - *position;
            *position += delta;
            *previous += delta;
        }
        self.gpu_current = false;
    }

    /// The body as a whole moved by `delta` over the next step of `dt`
    /// seconds (walking, a shift of weight): the cloth is simulated in a
    /// frame that travels with it. Otherwise a body walking at 1.3 m/s
    /// moves 4 cm a frame, and the garment has to be shoved along by its
    /// collision with the back; it is worn, and travels with the wearer.
    ///
    /// The cloth still feels the frame speeding up and slowing down (it
    /// swings back when the wearer sets off), up to a limit.
    ///
    /// The body's surface velocities given to the following step must be
    /// relative to this frame too, i.e. with `delta / dt` taken off.
    pub fn travel(&mut self, mesh: &mut GarmentMesh, delta: Vec3, dt: f32) {
        if mesh.vertex_count() != self.state.positions.len() || dt <= 0.0 {
            return;
        }
        self.pull_motion_from_gpu();
        let velocity = Vec3A::from(delta) / dt;
        let felt = (velocity - self.frame_velocity).clamp_length_max(MAX_FELT_ACCELERATION * dt);
        self.frame_velocity = velocity;
        let delta = Vec3A::from(delta);
        for i in 0..self.state.positions.len() {
            self.state.positions[i] += delta;
            self.state.previous[i] += delta;
            self.state.velocities[i] -= felt;
        }
        self.contacts.translate(delta);
        for (out, p) in mesh.positions.iter_mut().zip(&self.state.positions) {
            *out = (*p).into();
        }
        self.gpu_current = false;
    }

    pub fn has_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        return self.gpu.is_some();
        #[cfg(not(feature = "gpu"))]
        return false;
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    /// Where the global solver's last step spent its time.
    pub fn global_stats(&self) -> crate::global::GlobalStats {
        self.global.stats
    }

    /// Diagnostics of the current shape: `(largest stretch or compression
    /// along warp/weft, fraction of warp/weft rows strained beyond 10 %,
    /// widest stitch gap in metres)`. Useful for judging whether the
    /// sub-step count is high enough for the mesh resolution.
    pub fn diagnostics(&self) -> (f32, f32, f32) {
        use crate::model::{INVALID, ROW_SEAM};
        let x = &self.state.positions;
        let (mut worst, mut over, mut rows, mut gap) = (0.0f32, 0u32, 0u32, 0.0f32);
        for element in &self.model.elements {
            let combine = |row: &crate::model::Row| {
                (0..4)
                    .filter(|&k| element.particles[k] != INVALID)
                    .fold(Vec3A::ZERO, |sum, k| {
                        sum + x[element.particles[k] as usize] * row.weights[k]
                    })
            };
            if element.rows[0].flags & ROW_SEAM != 0 {
                gap = gap.max(combine(&element.rows[0]).length() - element.rows[0].seam_rest(0.0));
            } else if element.rows[0].rest == 1.0 {
                for row in &element.rows[..2] {
                    let strain = (combine(row).length() - 1.0).abs();
                    worst = worst.max(strain);
                    over += u32::from(strain > 0.1);
                    rows += 1;
                }
            }
        }
        (worst, over as f32 / rows.max(1) as f32, gap)
    }

    fn set_colliders(&mut self, ctx: &SolverContext<'_>) {
        self.colliders = ctx
            .colliders
            .iter()
            .map(|c| Ellipsoid {
                center: c.center.into(),
                radii: c.radii.into(),
                rotation: c.rotation,
            })
            .collect();
        self.body = ctx.body.cloned();
    }

    /// Makes `state.velocities` current if the GPU ran last.
    fn pull_motion_from_gpu(&mut self) {
        #[cfg(feature = "gpu")]
        if self.motion_on_gpu
            && let Some(gpu) = &mut self.gpu
        {
            gpu.download_motion(&mut self.state);
        }
        self.motion_on_gpu = false;
    }

    /// (Re)creates the GPU buffers; they start without contact planes.
    fn bind_gpu(&mut self) {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &mut self.gpu {
            gpu.bind(
                &self.model,
                &self.state,
                &self.colliders,
                self.body.as_deref(),
            );
        }
        self.gpu_current = true;
        self.motion_on_gpu = false;
        self.planes_on_gpu = false;
    }

    /// Rebuilds the cloth-against-cloth data for the current mesh: which
    /// layer every vertex is in and which vertices are sewn together.
    fn build_contacts(&mut self, mesh: &GarmentMesh, ctx: &SolverContext<'_>) {
        // Where one piece buttons onto another of the same layer, the one
        // with the buttonholes lies on top: it is half a layer further out,
        // so the lap is kept in order like two garments are, whatever the
        // cloth has been through.
        let piece_of = |v: u32| {
            mesh.pieces
                .iter()
                .find(|span| span.vertices.contains(&v))
                .map(|span| span.piece)
        };
        let piece = |id| ctx.document.piece(id);
        let mut on_top = Vec::new();
        for stitch in &mesh.stitches {
            if stitch.seam != topology_sync::Stitch::FASTENING {
                continue;
            }
            let (Some(a), Some(b)) = (piece_of(stitch.a), piece_of(stitch.b)) else {
                continue;
            };
            let (Some(pa), Some(pb)) = (piece(a), piece(b)) else {
                continue;
            };
            if a == b || pa.layer != pb.layer {
                continue;
            }
            match (pa.buttonholes.is_empty(), pb.buttonholes.is_empty()) {
                (false, true) => on_top.push(a),
                (true, false) => on_top.push(b),
                _ => {}
            }
        }
        let mut layers = vec![0u32; mesh.vertex_count()];
        for span in &mesh.pieces {
            let layer = 2 * piece(span.piece).map_or(0, |piece| piece.layer)
                + u32::from(on_top.contains(&span.piece));
            layers[span.vertices.start as usize..span.vertices.end as usize].fill(layer);
        }
        let triangles: Vec<[u32; 3]> = mesh.indices.as_chunks::<3>().0.to_vec();
        self.triangles.clone_from(&triangles);
        // Undistorted edge lengths come from the pattern, in millimetres.
        let rest: Vec<Vec3A> = mesh
            .pattern_positions
            .iter()
            .map(|p| Vec3A::new(p.x, p.y, 0.0) * 1e-3)
            .collect();
        // Buttons are not sewing: the layers they hold together still
        // collide right up to the button, which keeps the lap on top.
        let sewn: Vec<(u32, u32)> = mesh
            .stitches
            .iter()
            .filter(|s| s.seam != topology_sync::Stitch::FASTENING)
            .map(|s| (s.a, s.b))
            .collect();
        self.ogc = Ogc::new(&self.model, triangles.clone(), &sewn, &rest);
        self.global = Global::new(&self.model);
        // Fastened vertices are not one group either: grouped, the cloth
        // around every button and its hole was never tested against the
        // other front, and the fronts drifted through each other there.
        self.contacts = ClothContacts::new(triangles, layers, &sewn, &rest);
        self.clear_planes();
    }

    fn clear_planes(&mut self) {
        self.contacts.clear();
        #[cfg(feature = "gpu")]
        if self.planes_on_gpu
            && let Some(gpu) = &mut self.gpu
        {
            gpu.upload_planes(&[]);
        }
        self.planes_on_gpu = false;
    }

    /// The collision pass after the sub-steps of one step: continuous
    /// tests over the whole step and the contact planes for the next one.
    /// Returns `true` if it moved any particle.
    fn collide(&mut self, ctx: &SolverContext<'_>, ran_on_gpu: bool) -> bool {
        let settings = ctx.settings;
        let mut moved: Vec<u32> = Vec::new();
        let mut lifted: Vec<(u32, Vec3A)> = Vec::new();
        self.stats.body_stops = 0;
        if settings.continuous_body_collision
            && let Some(body) = ctx.body_mesh
        {
            let margin = settings.collision_margin.max(0.0);
            moved.extend(
                body.sweep_all(&self.step_start, &mut self.state.positions, margin)
                    .iter()
                    .map(|(i, _)| *i),
            );
            self.stats.body_stops = moved.len();
            // The other way round: skin features between cloth vertices
            // (collarbones, shoulder points, the bust) lift the faces they
            // poke through. A few rounds, as lifting one face can lower
            // its neighbour onto the next feature.
            let field = self.body.clone();
            let sample = |p: Vec3A| {
                let (distance, gradient) = field.as_ref()?.sample(p.into())?;
                Some((distance, Vec3A::from(gradient).try_normalize()?))
            };
            let sample = field
                .is_some()
                .then_some(&sample as &(dyn Fn(Vec3A) -> Option<(f32, Vec3A)> + Sync));
            for _ in 0..PUSH_ROUNDS {
                let round = body.push_out_cloth(
                    &self.triangles,
                    &mut self.state.positions,
                    margin,
                    PUSH_DEPTH,
                    sample,
                );
                if round.is_empty() {
                    break;
                }
                lifted.extend(round);
            }
            self.stats.body_stops += lifted.len();
        }

        if settings.self_collision {
            let params = ContactParams {
                thickness: settings.cloth_thickness.max(0.0),
                ..ContactParams::default()
            };
            let field = self.body.clone();
            let outward = |p: Vec3A| {
                field
                    .as_ref()?
                    .sample(p.into())
                    .and_then(|(_, gradient)| Vec3A::from(gradient).try_normalize())
            };
            let found = self.contacts.detect(
                &self.step_start,
                &mut self.state.positions,
                &params,
                field
                    .is_some()
                    .then_some(&outward as &(dyn Fn(Vec3A) -> Option<Vec3A> + Sync)),
            );
            self.stats.cloth_contacts = found.contacts;
            self.stats.cloth_crossings = found.crossings;
            moved.extend_from_slice(self.contacts.moved());
            #[cfg(feature = "gpu")]
            if let Some(gpu) = &mut self.gpu {
                // Also without contacts: nearby cloth has planes too.
                gpu.upload_planes(self.contacts.planes());
                self.planes_on_gpu = true;
            }
        } else {
            self.stats.cloth_contacts = 0;
            self.stats.cloth_crossings = 0;
            self.clear_planes();
        }

        if moved.is_empty() && lifted.is_empty() {
            return false;
        }
        if ran_on_gpu {
            self.pull_motion_from_gpu();
        }
        // A lifted face keeps sliding along the skin, only stops moving
        // into it: it is carried by the body, not stopped dead.
        for &(i, delta) in &lifted {
            if let Some(n) = delta.try_normalize() {
                let v = &mut self.state.velocities[i as usize];
                *v -= n * v.dot(n).min(0.0);
            }
            self.state.previous[i as usize] = self.state.positions[i as usize];
        }
        // A particle that was put back starts the next step at rest;
        // otherwise it would run into the same obstacle again.
        for &i in &moved {
            self.state.velocities[i as usize] = Vec3A::ZERO;
            self.state.previous[i as usize] = self.state.positions[i as usize];
        }
        #[cfg(feature = "gpu")]
        if ran_on_gpu && let Some(gpu) = &mut self.gpu {
            gpu.upload_state(&self.model, &self.state);
            self.motion_on_gpu = true;
        }
        true
    }

    fn refresh_stats(&mut self) {
        self.stats.particles = self.state.positions.len();
        self.stats.strain_triangles = self.model.strain_triangles;
        self.stats.bending_hinges = self.model.bending_hinges;
        self.stats.stitches = self.model.stitches;
        self.stats.batches = self.model.colours.len();
        self.stats.mass = self.model.mass;
    }

    /// Remaining fraction of the seam gaps at `seconds` into the closure.
    fn seam_gap(seconds: f32, closure_time: f32) -> f32 {
        if closure_time <= 0.0 {
            return 0.0;
        }
        let t = (seconds / closure_time).clamp(0.0, 1.0);
        // Ease in and out so the seams neither jerk at the start nor slam
        // shut at the end.
        1.0 - t * t * (3.0 - 2.0 * t)
    }

    /// Fraction of gravity applied at `seconds` into the seam closure.
    ///
    /// Until the seams hold the garment together nothing keeps the loose
    /// panels on the body, so they are sewn weightless and gravity fades in
    /// around the moment the seams shut (from 75 % to 125 % of the closure
    /// time).
    fn gravity_ramp(seconds: f32, closure_time: f32) -> f32 {
        if closure_time <= 0.0 {
            return 1.0;
        }
        let t = ((seconds / closure_time - 0.75) / 0.5).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }
}

impl ClothSolver for XpbdSolver {
    fn rebind(&mut self, mesh: &GarmentMesh, ctx: &SolverContext<'_>) {
        self.model = Model::build(mesh, ctx.document);
        self.fabric_graph = fabric_graph(mesh);
        self.grab_patch.clear();
        // The new model has everyone's own mass again.
        self.grab_held.clear();
        self.grab = None;
        self.state = State::at_rest(mesh.positions.iter().map(|&p| Vec3A::from(p)));
        self.set_colliders(ctx);
        self.seam_clock = 0.0;
        self.stats.seam_gap = 1.0;
        self.bind_gpu();
        self.build_contacts(mesh, ctx);
        self.refresh_stats();
    }

    fn refresh_materials(&mut self, mesh: &GarmentMesh, ctx: &SolverContext<'_>) {
        if mesh.vertex_count() != self.state.positions.len() {
            return self.rebind(mesh, ctx);
        }
        self.pull_motion_from_gpu();
        // Seam rest lengths are re-measured from the current gaps, so the
        // closure restarts from where the seams are now.
        self.model = Model::build(mesh, ctx.document);
        self.grab_held.clear();
        self.grab = None;
        self.set_colliders(ctx);
        self.seam_clock = 0.0;
        self.bind_gpu();
        // A piece may have changed its layer.
        self.build_contacts(mesh, ctx);
        self.refresh_stats();
    }

    fn set_positions(&mut self, mesh: &GarmentMesh) {
        if mesh.vertex_count() != self.state.positions.len() {
            return;
        }

        self.state = State::at_rest(mesh.positions.iter().map(|&p| Vec3A::from(p)));
        // Found for where the cloth was; they would hold it back there.
        self.clear_planes();
        self.global.reset();
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &mut self.gpu {
            gpu.upload_state(&self.model, &self.state);
        }
        self.gpu_current = true;
        self.motion_on_gpu = false;
    }

    fn set_grab(&mut self, vertex: u32, position: Vec3, radius: f32, strength: f32) {
        if self.grab.is_none_or(|(v, _, r, _)| v != vertex || r != radius) {
            self.release_held();
            self.grab_patch = grab_patch(&self.fabric_graph, vertex, radius.max(0.01));
            for &(index, weight) in &self.grab_patch {
                if weight >= 0.5
                    && let Some(w) = self.model.inverse_mass.get_mut(index as usize)
                {
                    self.grab_held.push((index, *w));
                    *w = 0.0;
                }
            }
        }
        self.grab = Some((
            vertex,
            position.into(),
            radius.max(0.01),
            strength.clamp(0.01, 1.0),
        ));
    }

    fn clear_grab(&mut self) {
        self.pull_motion_from_gpu();
        self.release_held();
        self.grab = None;
        self.grab_patch.clear();
    }

    fn step(&mut self, mesh: &mut GarmentMesh, ctx: &SolverContext<'_>) -> bool {
        if self.state.positions.is_empty() || mesh.vertex_count() != self.state.positions.len() {
            return false;
        }
        let started = Instant::now();
        let settings = ctx.settings;
        // The avatar was edited: collide with the new body from now on.
        let same_body = match (&self.body, ctx.body) {
            (Some(current), Some(new)) => Arc::ptr_eq(current, new),
            (None, None) => true,
            _ => false,
        };
        if !same_body {
            self.pull_motion_from_gpu();
            self.set_colliders(ctx);
            self.bind_gpu();
            // The new buffers are empty; the planes are still valid.
            #[cfg(feature = "gpu")]
            if let Some(gpu) = &mut self.gpu {
                gpu.upload_planes(self.contacts.planes());
                self.planes_on_gpu = true;
            }
        }
        self.step_start.clone_from(&self.state.positions);
        let count = settings.substeps.clamp(1, MAX_SUBSTEPS);
        let h = ctx.dt / count as f32;
        let substeps: Vec<SubstepParams> = (1..=count)
            .map(|i| {
                let seconds = self.seam_clock + i as f32 * h;
                let seam_gap = Self::seam_gap(seconds, settings.seam_closure_time);
                SubstepParams {
                    h,
                    gravity: Vec3A::from(settings.gravity)
                        * Self::gravity_ramp(seconds, settings.seam_closure_time),
                    damping: (-settings.damping.max(0.0) * h).exp(),
                    seam_gap,
                    // While the seams are being pulled shut the panels must be
                    // free to slide around the body into place; friction that
                    // early pins them where they first touch and leaves a tight
                    // garment bunched up. It fades in together with gravity.
                    friction: settings.friction.max(0.0)
                        * Self::gravity_ramp(seconds, settings.seam_closure_time),
                    margin: settings.collision_margin.max(0.0),
                    ground: ctx.ground,
                    max_speed: MAX_SPEED,
                    surface_collision: true,
                }
            })
            .collect();

        if settings.backend == SolverBackend::Global {
            self.step_global(ctx, &substeps);
            return self.finish_step(mesh, &substeps, started, "CPU · global");
        }
        if settings.backend == SolverBackend::Contact {
            self.step_contact(ctx, &substeps);
            return self.finish_step(mesh, &substeps, started, "CPU · contact (VBD + OGC)");
        }

        let mut ran_on_gpu = false;
        #[cfg(feature = "gpu")]
        if settings.backend == SolverBackend::Gpu
            && let Some(gpu) = &mut self.gpu
        {
            if !self.gpu_current {
                gpu.upload_state(&self.model, &self.state);
                // CPU steps may have changed the planes meanwhile.
                gpu.upload_planes(self.contacts.planes());
                self.planes_on_gpu = true;
                self.gpu_current = true;
            }
            ran_on_gpu = gpu.step(&substeps, &mut self.state.positions);
            self.motion_on_gpu = ran_on_gpu;
        }
        if !ran_on_gpu {
            self.pull_motion_from_gpu();
            for params in &substeps {
                cpu::substep(
                    &self.model,
                    &mut self.state,
                    &self.colliders,
                    self.body.as_deref(),
                    self.contacts.planes(),
                    params,
                    &mut self.scratch,
                );
            }
            self.gpu_current = false;
        }
        let _ = SolverBackend::Cpu; // referenced in both cfgs
        self.collide(ctx, ran_on_gpu);
        self.finish_step(
            mesh,
            &substeps,
            started,
            if ran_on_gpu {
                "GPU compute"
            } else {
                "CPU (rayon)"
            },
        )
    }

    fn stats(&self) -> SolverStats {
        self.stats.clone()
    }
}

impl XpbdSolver {
    /// Lets go of the held cloth: it has its own mass again.
    fn release_held(&mut self) {
        for (index, w) in self.grab_held.drain(..) {
            if let Some(slot) = self.model.inverse_mass.get_mut(index as usize) {
                *slot = w;
            }
        }
    }
}

impl XpbdSolver {
    /// One step with the global solver: a single implicit step for the
    /// whole frame.
    fn step_global(&mut self, ctx: &SolverContext<'_>, substeps: &[SubstepParams]) {
        self.pull_motion_from_gpu();
        self.gpu_current = false;
        let settings = ctx.settings;
        let Some(p) = substeps.last() else { return };
        // A step is one linear solve, which holds while the cloth only
        // turns a little: where the body pushes it far, the step is split
        // and the body swept through the parts.
        let moved = self.body.as_deref().map_or(0.0, |field| {
            self.state
                .positions
                .par_iter()
                .map(|&x| {
                    let near = field
                        .sample(x.into())
                        .is_some_and(|(distance, _)| distance < 4.0 * p.margin);
                    if near {
                        field.velocity_at(x.into()).length() * ctx.dt
                    } else {
                        0.0
                    }
                })
                .reduce(|| 0.0f32, f32::max)
        });
        let count = ((moved / GLOBAL_STRIDE).ceil() as usize).clamp(1, GLOBAL_MAX_SUBSTEPS);
        let h = ctx.dt / count as f32;
        let mut s = crate::global::GlobalStats::default();
        for index in 0..count {
            let input = crate::global::StepInput {
                h,
                gravity: p.gravity,
                damping: (-settings.damping.max(0.0) * h).exp(),
                seam_gap: p.seam_gap,
                friction: p.friction,
                margin: p.margin,
                ground: p.ground,
                max_speed: p.max_speed,
                iterations: GLOBAL_ITERATIONS,
                body: self.body.as_deref(),
                body_mesh: ctx.body_mesh.map(|m| &**m),
                body_lag: h * (count - 1 - index) as f32,
                cloth: if settings.self_collision {
                    self.contacts.pairs()
                } else {
                    &[]
                },
            };
            self.global.step(&mut self.state, &input);
            let step = self.global.stats;
            s.contacts = step.contacts;
            s.pinched = step.pinched;
            s.last_step = step.last_step;
            s.factorisations += step.factorisations;
            s.halvings += step.halvings;
            s.factor_us += step.factor_us;
            s.solve_us += step.solve_us;
            s.assemble_us += step.assemble_us;
            s.detect_us += step.detect_us;
        }
        self.global.stats = s;
        log::trace!(
            "global step: {count} parts, {} factorisations {} us, solves {} us, assembly {} us, last step {:.5}",
            s.factorisations,
            s.factor_us,
            s.solve_us,
            s.assemble_us,
            s.last_step
        );
        // No sweep of the paths here, unlike the other solvers: the step is
        // split so finely that cloth cannot pass through a limb within it,
        // and what ends a step inside the skin is put back on it by the
        // solver. Swept against the body where it ended up, paths of cloth
        // between two moving parts (a hand by the hip) were stopped short
        // by skin that had only swept past, tearing the cloth.
        let stopped = 0;
        // Cloth against cloth: what passed through other cloth during the
        // step is put back, and the planes for the next step are found.
        if settings.self_collision {
            let params = ContactParams {
                thickness: settings.cloth_thickness.max(0.0),
                // The planes bring a layer round within a few steps; a jump
                // would be undone by the seams every step (a fly shield
                // sewn under the front), shaking the cloth around it.
                move_layers: false,
                ..ContactParams::default()
            };
            // Which way is out, for garment layers: from the skin itself
            // (the distance field is not kept up to date for this solver).
            let skin = ctx.body_mesh.map(|m| &**m);
            let field = self.body.clone();
            let outward = |p: Vec3A| match skin {
                Some(skin) => skin.sides(p, 0.1)[0].map(|side| side.normal),
                None => field
                    .as_ref()?
                    .sample(p.into())
                    .and_then(|(_, gradient)| Vec3A::from(gradient).try_normalize()),
            };
            self.contact_scratch.clone_from(&self.state.positions);
            let found = self.contacts.detect(
                &self.step_start,
                &mut self.state.positions,
                &params,
                (skin.is_some() || field.is_some())
                    .then_some(&outward as &(dyn Fn(Vec3A) -> Option<Vec3A> + Sync)),
            );
            self.stats.cloth_contacts = found.contacts;
            self.stats.cloth_crossings = found.crossings;
            // Put back, it starts the next step at rest: otherwise it would
            // run into the same cloth again. Never further than it came
            // in this step (a hem swinging through the other leg goes all
            // the way back), but at least a fold's thickness. Cloth that
            // hardly moves (layers by a seam, held there by it) is only
            // nudged: jumped back, the seam pulls it through again, and it
            // jitters for ever. The planes ease it apart instead.
            for &i in self.contacts.moved() {
                let i = i as usize;
                let solved = self.contact_scratch[i];
                let came = (solved - self.step_start[i]).length();
                let shift = (self.state.positions[i] - solved)
                    .clamp_length_max(came.max(MIN_PUT_BACK));
                self.state.positions[i] = solved + shift;
                self.state.velocities[i] = Vec3A::ZERO;
                self.state.previous[i] = self.state.positions[i];
            }
        } else {
            self.stats.cloth_contacts = 0;
            self.stats.cloth_crossings = 0;
            self.clear_planes();
        }
        self.stats.body_stops = s.contacts + stopped;
        self.stats.batches = s.factorisations;
    }

    /// One step with the contact solver: fewer, implicit sub-steps, each
    /// iterated, with contact found inside them.
    fn step_contact(&mut self, ctx: &SolverContext<'_>, substeps: &[SubstepParams]) {
        self.pull_motion_from_gpu();
        self.gpu_current = false;
        let settings = ctx.settings;
        // As many sub-steps as the explicit solver, each iterated twice:
        // block descent, like XPBD, converges best with short steps.
        // Fewer, longer sub-steps with more (accelerated) iterations were
        // tried and stretch the cloth more for the same time.
        let count = substeps.len().max(1);
        // The body may have moved since the last step.
        self.ogc.invalidate();
        let h = ctx.dt / count as f32;
        let mut stats = crate::ogc::OgcStats::default();
        for index in 0..count {
            // The schedule (seams, gravity, friction) at this point of the step.
            let p = substeps[((index + 1) * substeps.len() / count)
                .saturating_sub(1)
                .min(substeps.len() - 1)];
            let input = StepInput {
                h,
                gravity: p.gravity,
                damping: (-settings.damping.max(0.0) * h).exp(),
                seam_gap: p.seam_gap,
                friction: p.friction,
                margin: p.margin,
                thickness: settings.cloth_thickness.max(0.0),
                ground: p.ground,
                max_speed: p.max_speed,
                iterations: CONTACT_ITERATIONS,
                acceleration: 0.0,
                body: self.body.as_deref(),
                body_mesh: ctx.body_mesh.map(|m| &**m),
                self_collision: settings.self_collision,
                body_lag: h * (count - 1 - index) as f32,
                colliders: &self.colliders,
            };
            self.ogc.substep(&self.model, &mut self.state, &input);
            let s = self.ogc.stats;
            stats.self_contacts = stats.self_contacts.max(s.self_contacts);
            stats.body_contacts = stats.body_contacts.max(s.body_contacts);
            stats.truncated += s.truncated;
            stats.detections += s.detections;
            stats.detect_us += s.detect_us;
            stats.solve_us += s.solve_us;
        }
        log::trace!(
            "contact step: {count} sub-steps, {} detections ({} ms), solve {} ms, {} truncated",
            stats.detections,
            stats.detect_us / 1000,
            stats.solve_us / 1000,
            stats.truncated
        );
        // Safety net for what the distance field cannot see (a body part
        // thinner than its voxels, or a body without one): no vertex ends
        // the step on the far side of a skin triangle it passed through.
        // Only its motion into the skin is taken away; it keeps sliding.
        let mut stopped = 0;
        if settings.continuous_body_collision
            && let Some(body) = ctx.body_mesh
        {
            let margin = settings.collision_margin.max(0.0);
            for (i, normal) in body.sweep_all(&self.step_start, &mut self.state.positions, margin) {
                let v = &mut self.state.velocities[i as usize];
                *v -= normal * v.dot(normal).min(0.0);
                stopped += 1;
            }
        }
        self.stats.cloth_contacts = stats.self_contacts;
        self.stats.cloth_crossings = stats.truncated;
        self.stats.body_stops = stats.body_contacts + stopped;
        self.stats.batches = self.ogc.colour_count();
    }

    /// Bookkeeping after either kind of step: the seam clock, the output
    /// positions and the interactive grab.
    fn finish_step(
        &mut self,
        mesh: &mut GarmentMesh,
        substeps: &[SubstepParams],
        started: Instant,
        backend: &'static str,
    ) -> bool {
        self.seam_clock += substeps.iter().map(|p| p.h).sum::<f32>();
        for (out, p) in mesh.positions.iter_mut().zip(&self.state.positions) {
            *out = (*p).into();
        }
        if let Some((vertex, target, _, strength)) = self.grab
            && let Some(center) = self.state.positions.get(vertex as usize).copied()
        {
            // The patch goes part of the way to the cursor, but never more
            // than a little per step: the fabric around it and the cloth
            // collision keep up, and a hard pull cannot drag the cloth
            // through itself.
            let _ = strength;
            // The cloth in the hand goes with the cursor, but never more
            // than a little per step: the fabric around it and the cloth
            // collision keep up, and a hard pull cannot drag the cloth
            // through itself or the body.
            let shift = (target - center).clamp_length_max(MAX_GRAB_SHIFT);
            if shift.length_squared() > 1e-12 {
                for &(index, weight) in &self.grab_patch {
                    let index = index as usize;
                    let Some(position) = self.state.positions.get_mut(index) else {
                        continue;
                    };
                    *position += shift * weight;
                    self.state.previous[index] = *position;
                    self.state.velocities[index] = Vec3A::ZERO;
                    if let Some(out) = mesh.positions.get_mut(index) {
                        *out = (*position).into();
                    }
                }
                mesh.recompute_normals();
            }
        }
        self.stats.backend = backend;
        self.stats.seam_gap = substeps.last().map_or(0.0, |p| p.seam_gap);
        self.stats.step_ms = started.elapsed().as_secs_f32() * 1000.0;
        true
    }
}

/// How far a grab moves the cloth in one step at most, metres.
const MAX_GRAB_SHIFT: f32 = 0.012;

/// Every vertex's neighbours along the fabric: the edges of its triangles,
/// with their length as cut, and the vertices it is stitched to.
fn fabric_graph(mesh: &GarmentMesh) -> Vec<Vec<(u32, f32)>> {
    let mut graph: Vec<Vec<(u32, f32)>> = vec![Vec::new(); mesh.vertex_count()];
    let mut link = |a: u32, b: u32, length: f32| {
        if let Some(list) = graph.get_mut(a as usize)
            && !list.iter().any(|&(n, _)| n == b)
        {
            list.push((b, length));
        }
    };
    for triangle in mesh.indices.chunks_exact(3) {
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let (a, b) = (triangle[a], triangle[b]);
            let length = mesh.pattern_positions[a as usize]
                .distance(mesh.pattern_positions[b as usize])
                * 1e-3;
            link(a, b, length);
            link(b, a, length);
        }
    }
    for stitch in &mesh.stitches {
        link(stitch.a, stitch.b, 0.0);
        link(stitch.b, stitch.a, 0.0);
    }
    graph
}

/// The cloth within `radius` of `vertex` along the fabric (not through the
/// air: a layer behind, or the other side of a fold, stays put), each with
/// its share of the move: the inner part moves as one, the rest less and
/// less towards the edge.
fn grab_patch(graph: &[Vec<(u32, f32)>], vertex: u32, radius: f32) -> Vec<(u32, f32)> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    if vertex as usize >= graph.len() {
        return Vec::new();
    }
    let mut distance = vec![f32::INFINITY; graph.len()];
    let mut queue = BinaryHeap::new();
    distance[vertex as usize] = 0.0;
    // Distances in micrometres, as integers, for the heap.
    queue.push(Reverse((0u64, vertex)));
    while let Some(Reverse((d, v))) = queue.pop() {
        let d = d as f32 * 1e-6;
        if d > distance[v as usize] {
            continue;
        }
        for &(n, length) in &graph[v as usize] {
            let next = d + length;
            if next < radius && next < distance[n as usize] {
                distance[n as usize] = next;
                queue.push(Reverse(((next * 1e6) as u64, n)));
            }
        }
    }
    let plateau = 0.35 * radius;
    distance
        .iter()
        .enumerate()
        .filter(|(_, d)| d.is_finite())
        .map(|(i, &d)| {
            let t = ((d - plateau) / (radius - plateau)).clamp(0.0, 1.0);
            (i as u32, 1.0 - t * t * (3.0 - 2.0 * t))
        })
        .filter(|&(_, w)| w > 0.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_grab_takes_the_connected_cloth_not_what_is_near_in_space() {
        // Two strips of 30 vertices 1 cm apart along the fabric; the second
        // strip lies right behind the first but is not connected to it.
        let mut graph: Vec<Vec<(u32, f32)>> = vec![Vec::new(); 60];
        for strip in [0u32, 30] {
            for i in 0..29 {
                graph[(strip + i) as usize].push((strip + i + 1, 0.01));
                graph[(strip + i + 1) as usize].push((strip + i, 0.01));
            }
        }
        let patch = grab_patch(&graph, 0, 0.2);
        assert!(patch.iter().all(|&(v, _)| v < 30), "the other strip stays put");
        let weight = |v: u32| patch.iter().find(|p| p.0 == v).map_or(0.0, |p| p.1);
        assert_eq!(weight(0), 1.0);
        assert_eq!(weight(5), 1.0, "the inner part moves as one");
        assert!(weight(12) > 0.0 && weight(12) < 1.0, "then less and less");
        assert_eq!(weight(25), 0.0, "beyond the reach");
    }
}

//! The simulation state manager: owns the model and the dynamic state,
//! picks the backend and implements `topology_sync::ClothSolver`.

use std::sync::Arc;
use std::time::Instant;

use glam::{Vec3, Vec3A};
use physics_collision::{ClothContacts, ContactParams};
use topology_sync::{ClothSolver, DistanceField, GarmentMesh, SolverBackend, SolverContext, SolverStats};

use crate::cpu::{self, Ellipsoid, State, SubstepParams};
use crate::model::Model;
use crate::ogc::{Ogc, StepInput};

/// Speed limit in m/s; far above anything cloth does under gravity.
const MAX_SPEED: f32 = 20.0;
const MAX_SUBSTEPS: u32 = 64;
/// Iterations per sub-step of the contact solver.
const CONTACT_ITERATIONS: usize = 2;
/// Rounds of body points against cloth faces per step.
const PUSH_ROUNDS: usize = 3;
/// How far behind a cloth face a body point may be and still lift it,
/// in metres: less than the thinnest body part the cloth meets.
const PUSH_DEPTH: f32 = 0.01;

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
    /// The GPU holds contact planes that are not all empty.
    planes_on_gpu: bool,
    /// Positions at the start of the step, for the continuous tests.
    step_start: Vec<Vec3A>,
    #[cfg(feature = "gpu")]
    gpu: Option<crate::gpu::GpuBackend>,
    /// Which side holds the authoritative velocities.
    motion_on_gpu: bool,
    /// GPU particle buffers match `state` (false after CPU steps).
    gpu_current: bool,
    /// Soft area grab: vertex, target, influence radius and per-step strength.
    grab: Option<(u32, Vec3A, f32, f32)>,
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
        Self { gpu: Some(crate::gpu::GpuBackend::new(device, queue)), ..Self::default() }
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
        for ((position, previous), &new) in self.state.positions.iter_mut().zip(&mut self.state.previous).zip(&mesh.positions) {
            let delta = Vec3A::from(new) - *position;
            *position += delta;
            *previous += delta;
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
                (0..4).filter(|&k| element.particles[k] != INVALID).fold(Vec3A::ZERO, |sum, k| {
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
            .map(|c| Ellipsoid { center: c.center.into(), radii: c.radii.into(), rotation: c.rotation })
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
            gpu.bind(&self.model, &self.state, &self.colliders, self.body.as_deref());
        }
        self.gpu_current = true;
        self.motion_on_gpu = false;
        self.planes_on_gpu = false;
    }

    /// Rebuilds the cloth-against-cloth data for the current mesh: which
    /// layer every vertex is in and which vertices are sewn together.
    fn build_contacts(&mut self, mesh: &GarmentMesh, ctx: &SolverContext<'_>) {
        let mut layers = vec![0u32; mesh.vertex_count()];
        for span in &mesh.pieces {
            let layer = ctx.document.piece(span.piece).map_or(0, |piece| piece.layer);
            layers[span.vertices.start as usize..span.vertices.end as usize].fill(layer);
        }
        let triangles: Vec<[u32; 3]> = mesh.indices.as_chunks::<3>().0.to_vec();
        self.triangles.clone_from(&triangles);
        let stitches: Vec<(u32, u32)> = mesh.stitches.iter().map(|s| (s.a, s.b)).collect();
        // Undistorted edge lengths come from the pattern, in millimetres.
        let rest: Vec<Vec3A> = mesh.pattern_positions.iter().map(|p| Vec3A::new(p.x, p.y, 0.0) * 1e-3).collect();
        // Buttons are not sewing: the layers they hold together still
        // collide right up to the button, which keeps the lap on top.
        let sewn: Vec<(u32, u32)> =
            mesh.stitches.iter().filter(|s| s.seam != topology_sync::Stitch::FASTENING).map(|s| (s.a, s.b)).collect();
        self.ogc = Ogc::new(&self.model, triangles.clone(), &sewn, &rest);
        self.contacts = ClothContacts::new(triangles, layers, &stitches, &rest);
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
            moved.extend(body.sweep_all(&self.step_start, &mut self.state.positions, margin).iter().map(|(i, _)| *i));
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
            let sample = field.is_some().then_some(&sample as &(dyn Fn(Vec3A) -> Option<(f32, Vec3A)> + Sync));
            for _ in 0..PUSH_ROUNDS {
                let round = body.push_out_cloth(&self.triangles, &mut self.state.positions, margin, PUSH_DEPTH, sample);
                if round.is_empty() {
                    break;
                }
                lifted.extend(round);
            }
            self.stats.body_stops += lifted.len();
        }

        if settings.self_collision {
            let params = ContactParams { thickness: settings.cloth_thickness.max(0.0), ..ContactParams::default() };
            let field = self.body.clone();
            let outward = |p: Vec3A| {
                field.as_ref()?.sample(p.into()).and_then(|(_, gradient)| Vec3A::from(gradient).try_normalize())
            };
            let found = self.contacts.detect(
                &self.step_start,
                &mut self.state.positions,
                &params,
                field.is_some().then_some(&outward as &(dyn Fn(Vec3A) -> Option<Vec3A> + Sync)),
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
        if ran_on_gpu
            && let Some(gpu) = &mut self.gpu
        {
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
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &mut self.gpu {
            gpu.upload_state(&self.model, &self.state);
        }
        self.gpu_current = true;
        self.motion_on_gpu = false;
    }

    fn set_grab(&mut self, vertex: u32, position: Vec3, radius: f32, strength: f32) {
        self.grab = Some((vertex, position.into(), radius.max(0.01), strength.clamp(0.01, 1.0)));
    }

    fn clear_grab(&mut self) {
        self.pull_motion_from_gpu();
        self.grab = None;
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
                    gravity: Vec3A::from(settings.gravity) * Self::gravity_ramp(seconds, settings.seam_closure_time),
                    damping: (-settings.damping.max(0.0) * h).exp(),
                    seam_gap,
                    // While the seams are being pulled shut the panels must be
                    // free to slide around the body into place; friction that
                    // early pins them where they first touch and leaves a tight
                    // garment bunched up. It fades in together with gravity.
                    friction: settings.friction.max(0.0) * Self::gravity_ramp(seconds, settings.seam_closure_time),
                    margin: settings.collision_margin.max(0.0),
                    ground: ctx.ground,
                    max_speed: MAX_SPEED,
                    surface_collision: true,
                }
            })
            .collect();

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
        self.finish_step(mesh, &substeps, started, if ran_on_gpu { "GPU compute" } else { "CPU (rayon)" })
    }

    fn stats(&self) -> SolverStats {
        self.stats.clone()
    }
}

impl XpbdSolver {
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
            let p = substeps[((index + 1) * substeps.len() / count).saturating_sub(1).min(substeps.len() - 1)];
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
    fn finish_step(&mut self, mesh: &mut GarmentMesh, substeps: &[SubstepParams], started: Instant, backend: &'static str) -> bool {
        self.seam_clock += substeps.iter().map(|p| p.h).sum::<f32>();
        for (out, p) in mesh.positions.iter_mut().zip(&self.state.positions) {
            *out = (*p).into();
        }
        if let Some((vertex, target, radius, strength)) = self.grab
            && let Some(center) = self.state.positions.get(vertex as usize).copied()
        {
            let delta = target - center;
            if delta.length_squared() > 1e-10 {
                for (index, position) in self.state.positions.iter_mut().enumerate() {
                    let distance = position.distance(center);
                    let falloff = (1.0 - distance / radius).clamp(0.0, 1.0);
                    let influence = falloff * falloff * strength;
                    if influence <= 0.0 {
                        continue;
                    }
                    *position += delta * influence;
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

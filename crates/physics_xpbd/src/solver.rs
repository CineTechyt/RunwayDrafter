//! The simulation state manager: owns the model and the dynamic state,
//! picks the backend and implements `topology_sync::ClothSolver`.

use std::sync::Arc;
use std::time::Instant;

use glam::Vec3A;
use physics_collision::{ClothContacts, ContactParams};
use topology_sync::{ClothSolver, DistanceField, GarmentMesh, SolverBackend, SolverContext, SolverStats};

use crate::cpu::{self, Ellipsoid, State, SubstepParams};
use crate::model::Model;

/// Speed limit in m/s; far above anything cloth does under gravity.
const MAX_SPEED: f32 = 20.0;
const MAX_SUBSTEPS: u32 = 64;

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
                gap = gap.max(combine(&element.rows[0]).length());
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
        let triangles = mesh.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
        let stitches: Vec<(u32, u32)> = mesh.stitches.iter().map(|s| (s.a, s.b)).collect();
        // Undistorted edge lengths come from the pattern, in millimetres.
        let rest: Vec<Vec3A> = mesh.pattern_positions.iter().map(|p| Vec3A::new(p.x, p.y, 0.0) * 1e-3).collect();
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
        self.stats.body_stops = 0;
        if settings.continuous_body_collision
            && let Some(body) = ctx.body_mesh
        {
            let margin = settings.collision_margin.max(0.0);
            moved.extend(body.sweep_all(&self.step_start, &mut self.state.positions, margin).iter().map(|(i, _)| *i));
            self.stats.body_stops = moved.len();
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

        if moved.is_empty() {
            return false;
        }
        // A particle that was put back starts the next step at rest;
        // otherwise it would run into the same obstacle again.
        if ran_on_gpu {
            self.pull_motion_from_gpu();
        }
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

        self.seam_clock += ctx.dt;
        for (out, p) in mesh.positions.iter_mut().zip(&self.state.positions) {
            *out = (*p).into();
        }
        self.stats.backend = if ran_on_gpu { "GPU compute" } else { "CPU (rayon)" };
        self.stats.seam_gap = substeps.last().map_or(0.0, |p| p.seam_gap);
        self.stats.step_ms = started.elapsed().as_secs_f32() * 1000.0;
        true
    }

    fn stats(&self) -> SolverStats {
        self.stats.clone()
    }
}

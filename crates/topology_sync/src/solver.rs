//! The cloth simulation interface.
//!
//! This crate only defines the contract; `physics_xpbd` implements it. The
//! application drives a solver from the timeline and re-uploads
//! `mesh.positions` whenever `step` reports motion.

use cad_core::PatternDocument;
use cad_core::glam::{Quat, Vec3};

use crate::mesh::GarmentMesh;

/// Where the constraint solve runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolverBackend {
    /// Multithreaded CPU solve.
    Cpu,
    /// Compute shaders. Falls back to the CPU when unavailable.
    Gpu,
    /// Multithreaded CPU solve built for contact: implicit (vertex block
    /// descent) with offset geometric contact. Cloth never passes through
    /// itself, and lies on the body rather than being held to it.
    #[default]
    Contact,
}

/// User-tunable simulation parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverSettings {
    pub backend: SolverBackend,
    /// XPBD sub-steps per simulation step. Each sub-step is one solver
    /// sweep, so this is what sets how closely the fabric reaches its real
    /// stiffness: too few and it sags like rubber under its own weight.
    /// The number needed grows with mesh resolution (roughly with the
    /// square of the element count from support to hem). Cost is linear.
    pub substeps: u32,
    /// Acceleration in m/s².
    pub gravity: Vec3,
    /// Velocity decay rate in 1/s (air drag and internal friction).
    pub damping: f32,
    /// Seconds over which seams contract from their initial gap to closed.
    /// The panels are sewn weightless: gravity fades in as the seams shut,
    /// so loose panels do not drop off the body first. Zero sews instantly.
    pub seam_closure_time: f32,
    /// Coulomb friction coefficient between cloth and colliders.
    pub friction: f32,
    /// Distance kept between cloth and colliders, in metres.
    pub collision_margin: f32,
    /// Cloth collides with cloth: folds do not pass through each other and
    /// garment layers stay in order.
    pub self_collision: bool,
    /// Distance kept between two cloth surfaces, in metres.
    pub cloth_thickness: f32,
    /// Sweep every vertex's path against the body mesh each step, so fast
    /// or thin parts (fingers, a raised arm) cannot be tunnelled through.
    pub continuous_body_collision: bool,
}

impl Default for SolverSettings {
    fn default() -> Self {
        Self {
            backend: SolverBackend::default(),
            substeps: 30,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 1.5,
            seam_closure_time: 0.8,
            friction: 0.4,
            collision_margin: 0.004,
            self_collision: true,
            cloth_thickness: 0.0025,
            continuous_body_collision: true,
        }
    }
}

/// A rigid ellipsoid the cloth collides with (one avatar body part).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Collider {
    pub center: Vec3,
    pub radii: Vec3,
    pub rotation: Quat,
}

/// A signed distance field of the avatar on a regular grid: negative inside
/// the body, positive outside. This is what lets cloth collide with an
/// arbitrary body mesh at a fixed cost per particle.
#[derive(Debug, Clone, PartialEq)]
pub struct DistanceField {
    /// World position of sample `(0, 0, 0)`.
    pub origin: Vec3,
    /// Grid spacing in metres.
    pub voxel: f32,
    /// Number of samples along x, y, z.
    pub dims: [u32; 3],
    /// Distances in metres, x fastest: `values[x + dims[0] * (y + dims[1] * z)]`.
    /// Exact near the surface, clamped to `±band` further away.
    pub values: Vec<f32>,
    /// How fast the body's surface moves, in m/s, per sample (same layout
    /// as `values`): the velocity of the closest point of the skin, near
    /// it, and zero further away. Empty for a body that stands still. Lets
    /// friction work relative to a moving body, so it carries cloth along.
    pub velocity: Vec<Vec3>,
}

impl DistanceField {
    /// The body's surface velocity near `p`, trilinearly interpolated; zero
    /// when unknown.
    pub fn velocity_at(&self, p: Vec3) -> Vec3 {
        if self.velocity.len() != self.values.len() {
            return Vec3::ZERO;
        }
        let g = (p - self.origin) / self.voxel;
        let max = Vec3::new(
            self.dims[0] as f32 - 1.0,
            self.dims[1] as f32 - 1.0,
            self.dims[2] as f32 - 1.0,
        );
        if g.cmplt(Vec3::ZERO).any() || g.cmpge(max).any() {
            return Vec3::ZERO;
        }
        let cell = g.floor();
        let f = g - cell;
        let (x, y, z) = (cell.x as u32, cell.y as u32, cell.z as u32);
        let at = |dx: u32, dy: u32, dz: u32| {
            self.velocity[(x + dx + self.dims[0] * (y + dy + self.dims[1] * (z + dz))) as usize]
        };
        let lerp = |a: Vec3, b: Vec3, t: f32| a + (b - a) * t;
        let y0 = lerp(
            lerp(at(0, 0, 0), at(1, 0, 0), f.x),
            lerp(at(0, 1, 0), at(1, 1, 0), f.x),
            f.y,
        );
        let y1 = lerp(
            lerp(at(0, 0, 1), at(1, 0, 1), f.x),
            lerp(at(0, 1, 1), at(1, 1, 1), f.x),
            f.y,
        );
        lerp(y0, y1, f.z)
    }

    #[inline]
    fn at(&self, x: u32, y: u32, z: u32) -> f32 {
        self.values[(x + self.dims[0] * (y + self.dims[1] * z)) as usize]
    }

    /// Trilinearly interpolated distance and its gradient (which points away
    /// from the body). `None` outside the grid.
    #[inline]
    pub fn sample(&self, p: Vec3) -> Option<(f32, Vec3)> {
        let g = (p - self.origin) / self.voxel;
        let max = Vec3::new(
            self.dims[0] as f32 - 1.0,
            self.dims[1] as f32 - 1.0,
            self.dims[2] as f32 - 1.0,
        );
        if g.cmplt(Vec3::ZERO).any() || g.cmpge(max).any() {
            return None;
        }
        let cell = g.floor();
        let f = g - cell;
        let (x, y, z) = (cell.x as u32, cell.y as u32, cell.z as u32);
        let c = |dx: u32, dy: u32, dz: u32| self.at(x + dx, y + dy, z + dz);
        let (c000, c100, c010, c110) = (c(0, 0, 0), c(1, 0, 0), c(0, 1, 0), c(1, 1, 0));
        let (c001, c101, c011, c111) = (c(0, 0, 1), c(1, 0, 1), c(0, 1, 1), c(1, 1, 1));
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        // Interpolate along x, then y, then z; the gradient is the exact
        // derivative of that interpolant.
        let (x00, x10, x01, x11) = (
            lerp(c000, c100, f.x),
            lerp(c010, c110, f.x),
            lerp(c001, c101, f.x),
            lerp(c011, c111, f.x),
        );
        let (y0, y1) = (lerp(x00, x10, f.y), lerp(x01, x11, f.y));
        let distance = lerp(y0, y1, f.z);
        let dx = lerp(
            lerp(c100 - c000, c110 - c010, f.y),
            lerp(c101 - c001, c111 - c011, f.y),
            f.z,
        );
        let dy = lerp(x10 - x00, x11 - x01, f.z);
        let dz = y1 - y0;
        Some((distance, Vec3::new(dx, dy, dz) / self.voxel))
    }
}

/// What the solver reports back for the UI.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SolverStats {
    /// Backend that ran the last step (may differ from the requested one).
    pub backend: &'static str,
    pub particles: usize,
    pub strain_triangles: usize,
    pub bending_hinges: usize,
    pub stitches: usize,
    /// Independent constraint batches (graph colours).
    pub batches: usize,
    /// Wall-clock cost of the last step in milliseconds.
    pub step_ms: f32,
    /// Total cloth mass in kilograms.
    pub mass: f32,
    /// 1.0 while seams are fully open, 0.0 once sewn shut.
    pub seam_gap: f32,
    /// Vertices held apart from other cloth in the last step.
    pub cloth_contacts: usize,
    /// Vertices that had passed through cloth in the last step and were
    /// put back.
    pub cloth_crossings: usize,
    /// Vertices stopped by the continuous test against the body mesh.
    pub body_stops: usize,
}

pub struct SolverContext<'a> {
    /// Fabric physics, seams and grainlines live in the document.
    pub document: &'a PatternDocument,
    pub settings: &'a SolverSettings,
    pub colliders: &'a [Collider],
    /// The avatar's body, if it is a mesh rather than ellipsoids.
    pub body: Option<&'a std::sync::Arc<DistanceField>>,
    /// The same body as triangles behind a BVH, for the continuous test.
    pub body_mesh: Option<&'a std::sync::Arc<physics_collision::TriangleMesh>>,
    /// Height of the floor plane.
    pub ground: f32,
    /// Fixed simulation time step in seconds.
    pub dt: f32,
}

pub trait ClothSolver {
    /// Called whenever the mesh topology was rebuilt or positions were
    /// reset: vertex indices are no longer valid. Rebuilds all constraints
    /// and restarts from rest at `mesh.positions`.
    fn rebind(&mut self, mesh: &GarmentMesh, ctx: &SolverContext<'_>);

    /// Fabric properties or grainlines changed but the topology did not:
    /// recompute masses and stiffnesses, keep the motion.
    fn refresh_materials(&mut self, mesh: &GarmentMesh, ctx: &SolverContext<'_>);

    /// The cloth was moved from outside (carried along with the avatar):
    /// continue from `mesh.positions`, at rest, keeping everything else.
    fn set_positions(&mut self, mesh: &GarmentMesh);

    /// Holds one cloth particle at a user-controlled world-space position
    /// while an interactive 3D grab is active.
    fn set_grab(&mut self, vertex: u32, position: Vec3, radius: f32, strength: f32);

    /// Releases the interactive grab constraint.
    fn clear_grab(&mut self);

    /// Advances the simulation by `ctx.dt`, writing `mesh.positions`.
    /// Returns `true` if any vertex moved.
    fn step(&mut self, mesh: &mut GarmentMesh, ctx: &SolverContext<'_>) -> bool;

    fn stats(&self) -> SolverStats;
}

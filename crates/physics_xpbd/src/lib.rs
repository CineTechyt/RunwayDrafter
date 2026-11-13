//! Extended Position-Based Dynamics (XPBD) cloth simulation.
//!
//! # Model
//!
//! Every constraint in this solver has the same shape:
//!
//! ```text
//! C(x) = | Σ wᵢ·xᵢ | − rest        (up to four particles)
//! ```
//!
//! which covers everything a garment needs:
//!
//! * **Anisotropic strain** — for a triangle, `Σ wᵢ·xᵢ` is the deformation
//!   gradient applied to a material direction, so `C` is the stretch along
//!   that direction. Four rows per triangle: weft, warp and the two 45° bias
//!   diagonals, each with its own stiffness.
//! * **Isometric bending** — for two triangles sharing an edge, the weights
//!   are the cotangent stencil of the quadratic bending energy (Bergou et
//!   al. 2006) and `rest = 0`.
//! * **Sewing** — two particles with weights `+1, −1`; `rest` is the initial
//!   gap, scaled down to zero while the seam closes.
//!
//! One layout ([`model::Element`]) therefore serves both backends, and the
//! WGSL shader is a line-by-line port of the CPU loop.
//!
//! # Solve
//!
//! Small-step XPBD: each step is split into sub-steps of one Gauss-Seidel
//! sweep. Elements are graph-coloured so that no two elements of a colour
//! share a particle; a colour is then solved in parallel, on CPU threads
//! ([`cpu`]) or as one compute dispatch ([`gpu`]).

pub mod cpu;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod material;
pub mod model;
pub mod ogc;
mod solver;

pub use model::{Element, Model, Row};
pub use solver::XpbdSolver;

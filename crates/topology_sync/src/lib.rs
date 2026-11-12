//! Keeps the 3D garment mesh in step with the 2D pattern document.
//!
//! * [`triangulate`] turns one outline into a triangle mesh whose boundary
//!   vertices are sampled per pattern edge.
//! * [`GarmentTopology`] caches those per-piece meshes, re-meshes only what a
//!   [`cad_core::DocumentChange`] invalidated and assembles the global
//!   [`GarmentMesh`], including seam stitches.
//! * The [`TopologySync`] trait is the interface the rest of the engine uses,
//!   in both directions: pattern edge → mesh vertices and mesh surface point →
//!   pattern position.
//! * [`ClothSolver`] is the extension point for draping.
//! * [`analysis`] measures the draped garment: strain and clearance to the
//!   body per vertex, for the technical views.

pub mod analysis;
pub mod mesh;
pub mod solver;
pub mod sync;
pub mod triangulate;

pub use analysis::{clearance_map, strain_map};
pub use mesh::{GarmentMesh, PieceSpan, RayHit, Stitch, VertexOrigin};
pub use solver::{
    ClothSolver, Collider, DistanceField, SolverBackend, SolverContext, SolverSettings, SolverStats,
};
pub use sync::{GarmentTopology, PatternLocation, SurfaceLocation, SyncReport, TopologySync};
pub use triangulate::{MeshingParams, PieceMesh};

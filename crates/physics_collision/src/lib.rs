//! Collision detection for cloth.
//!
//! * [`bvh`] — a bounding volume hierarchy over boxes: build, refit, box
//!   queries, ray casts and closest-primitive searches.
//! * [`ccd`] — the geometric tests: segment–triangle, closest point on a
//!   triangle and the continuous vertex–triangle test over a time step.
//! * [`TriangleMesh`] — a static triangle mesh (the avatar) behind a BVH:
//!   ray casts, clearance and the sweep that stops a cloth vertex from
//!   passing through the skin within a step.
//! * [`ClothContacts`] — cloth against cloth: finds vertices that are
//!   closer than the fabric thickness to a triangle, or passed through one
//!   during the step, and turns them into contact planes. Garment layers
//!   decide which side is the right one where two garments meet.
//!
//! The crate has no dependency on the solver: it works on plain position
//! slices, and the solver applies the planes it produces.

pub mod bvh;
pub mod ccd;
mod contacts;
mod mesh;

pub use bvh::{Aabb, Bvh};
pub use contacts::{
    ClothContacts, ContactPair, ContactParams, ContactPlane, ContactStats, PLANES_PER_VERTEX,
};
pub use mesh::{Clearance, SurfaceHit, SurfacePoint, TriangleMesh};

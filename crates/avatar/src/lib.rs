//! A parametric human body for dressing.
//!
//! The shape comes from the MakeHuman base mesh and morph targets (CC0),
//! baked into `assets/avatar/body.bin` by `tools/bake_avatar.py`. This crate
//! blends those targets ([`BodyModel::build`]), takes tape measurements off
//! the result ([`measure`]), solves the blend so that a measurement hits a
//! requested value ([`fit`]) and voxelises the body into the distance field
//! the cloth solver collides with ([`field`]).

pub mod arrange;
pub mod field;
pub mod fit;
pub mod measure;
pub mod model;
pub mod pose;

pub use arrange::arrange_around;
pub use field::distance_field;
pub use fit::{FitReport, fit, fit_all};
pub use measure::{Measure, Measurements, measure};
pub use model::{BodyMesh, BodyModel, BodyParams, Bone, Joint, LoadError, Morph};
pub use pose::{Pose, PoseParams, PosedBody, carry};

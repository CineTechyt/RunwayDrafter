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
pub mod look;
pub mod measure;
pub mod model;
pub mod motion;
pub mod pose;
mod skinning;

pub use arrange::arrange_around;
pub use field::{FieldOptions, distance_field, distance_field_within};
pub use fit::{FitReport, fit, fit_all};
pub use look::{Appearance, Hair, HairStyle, Head, LookMesh};
pub use measure::{Measure, Measurements, measure};
pub use motion::{Angles, Motion, Performance, Side};
pub use model::{BodyMesh, BodyModel, BodyParams, Bone, Joint, LoadError, Morph};
pub use pose::{Attachment, Pose, PoseParams, PosedBody, attach, attach_to, carry, follow};

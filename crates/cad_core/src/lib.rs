//! 2D vector CAD data model for garment patterns.
//!
//! This crate is the single source of truth for everything the pattern maker
//! edits. It has no GPU, windowing or UI dependencies so it can be used from
//! headless tools, tests and WASM alike.
//!
//! Conventions:
//! * Pattern space is 2D, **millimetres**, Y up.
//! * Every closed outline has as many segments as anchor points; segment `i`
//!   runs from anchor `i` to anchor `(i + 1) % n`. That index is the stable
//!   "edge index" that the rest of the engine tracks (see [`EdgeRef`]).

pub mod document;
pub mod edit;
pub mod fit;
pub mod geometry;
pub mod ids;
pub mod lines;
pub mod material;
pub mod pattern;
pub mod transform;
pub mod uv;

pub use document::{PatternDocument, Seam};
pub use edit::{DocumentChange, DocumentSnapshot, EditCommand, EditError, History, MoveHandle};
pub use geometry::{CubicBezier, Curve, CurveSegment, Line, QuadBezier};
pub use ids::{EdgeRef, FabricId, HandleKind, HandleRef, PieceId, SeamId};
pub use lines::EdgeSpan;
pub use material::{Fabric, FabricPattern, FabricPhysics};
pub use pattern::{
    Arrangement, Button, ButtonFinish, ButtonStyle, Buttonhole, ButtonholeRef, ControlPoint,
    FlatOutline, Grainline, Notch, NotchKind, Outline, PatternPiece, Placement, PointKind,
    SeamAllowance, Segment,
};
pub use transform::PieceTransform;
pub use uv::UvMapping;

pub use glam;

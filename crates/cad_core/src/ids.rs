//! Stable identifiers used to reference pattern entities across crates.

use serde::{Deserialize, Serialize};

macro_rules! id_type {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        pub struct $name(pub u32);
    };
}

id_type!(
    /// Identifies a pattern piece for the lifetime of a document.
    PieceId
);
id_type!(
    /// Identifies a fabric (material + physical properties).
    FabricId
);
id_type!(
    /// Identifies a sewing relationship between two pattern edges.
    SeamId
);

/// One edge (segment) of a piece outline. Edge `i` runs from anchor `i` to
/// anchor `i + 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EdgeRef {
    pub piece: PieceId,
    pub edge: u32,
}

impl EdgeRef {
    pub const fn new(piece: PieceId, edge: u32) -> Self {
        Self { piece, edge }
    }
}

/// A draggable handle inside one outline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HandleKind {
    /// Anchor point `i` (on the curve).
    Anchor(u32),
    /// Off-curve Bézier control of segment `segment`. `slot` is 0 for a
    /// quadratic control or the first cubic control, 1 for the second.
    Control { segment: u32, slot: u8 },
}

/// A draggable handle, addressed document-wide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HandleRef {
    pub piece: PieceId,
    pub kind: HandleKind,
}

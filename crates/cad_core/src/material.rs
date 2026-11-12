//! Fabric definitions: how a material looks (PBR) and how it behaves (physics).

use serde::{Deserialize, Serialize};

use crate::ids::FabricId;

/// Mechanical properties consumed by a cloth solver.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FabricPhysics {
    /// Areal density in g/m².
    pub density: f32,
    /// Stretch stiffness along the weft, 0..=1 (normalised).
    pub stretch_weft: f32,
    /// Stretch stiffness along the warp, 0..=1.
    pub stretch_warp: f32,
    /// Shear stiffness, 0..=1.
    pub shear: f32,
    /// Bending stiffness, 0..=1.
    pub bending: f32,
    /// Thickness in mm.
    pub thickness: f32,
}

impl Default for FabricPhysics {
    fn default() -> Self {
        // Roughly a medium-weight cotton jersey.
        Self { density: 180.0, stretch_weft: 0.35, stretch_warp: 0.45, shear: 0.3, bending: 0.2, thickness: 0.6 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fabric {
    pub id: FabricId,
    pub name: String,
    /// Linear RGB base colour.
    pub base_color: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    /// Strength of the procedural weave pattern in the viewport, 0..=1.
    pub weave_strength: f32,
    pub physics: FabricPhysics,
}

impl Fabric {
    pub fn new(id: FabricId, name: impl Into<String>, base_color: [f32; 3]) -> Self {
        Self {
            id,
            name: name.into(),
            base_color,
            roughness: 0.85,
            metallic: 0.0,
            weave_strength: 0.35,
            physics: FabricPhysics::default(),
        }
    }
}

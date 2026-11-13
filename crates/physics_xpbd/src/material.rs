//! Maps the document's fabric parameters to SI quantities.
//!
//! `cad_core::FabricPhysics` stores stiffnesses as normalised 0..=1 sliders
//! (what a pattern maker sees); the solver needs physical units. The ranges
//! below span typical apparel textiles on a logarithmic scale.

use cad_core::FabricPhysics;

/// Physical fabric properties as the solver uses them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FabricSi {
    /// Areal density in kg/m².
    pub density: f32,
    /// Tensile stiffness along the weft in N/m (force per unit width at
    /// unit strain).
    pub weft: f32,
    /// Tensile stiffness along the warp in N/m.
    pub warp: f32,
    /// Stiffness against stretch along the 45° bias, i.e. shear, in N/m.
    pub bias: f32,
    /// Bending rigidity in N·m.
    pub bending: f32,
}

/// Log-interpolates between `low` (slider 0) and `high` (slider 1).
fn log_scale(slider: f32, low: f32, high: f32) -> f32 {
    low * (high / low).powf(slider.clamp(0.0, 1.0))
}

impl FabricSi {
    pub fn from_physics(physics: &FabricPhysics) -> Self {
        Self {
            // g/m² → kg/m², with a floor so no particle becomes massless.
            density: physics.density.max(1.0) * 1e-3,
            // Stretchy knits sit near 30 N/m, canvas and denim near 10 kN/m.
            weft: log_scale(physics.stretch_weft, 30.0, 10_000.0),
            warp: log_scale(physics.stretch_warp, 30.0, 10_000.0),
            // Woven fabric shears far more easily than it stretches.
            bias: log_scale(physics.shear, 5.0, 2_000.0),
            // Silk is around 1e-6 N·m, heavy denim around 3e-4 N·m. The
            // slider describes a fabric 0.6 mm thick; thicker cloth is
            // harder to bend, by the 1.5th power of its thickness (a solid
            // plate goes by the cube; woven yarns slide on each other).
            bending: log_scale(physics.bending, 1e-6, 3e-4)
                * (physics.thickness.max(0.05) / 0.6).powf(1.5),
        }
    }
}

//! Pattern-space → texture-space mapping.
//!
//! A sewing pattern is already a distortion-free flattening of the garment, so
//! "UV unwrapping" reduces to an affine map from pattern millimetres to
//! texture repeats, rotated so that V follows the grainline.

use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::pattern::Grainline;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UvMapping {
    /// Pattern-space point that maps to UV `(0, 0)`.
    pub origin: Vec2,
    /// Physical size in mm of one texture repeat (weft, warp).
    pub tile_size: Vec2,
    /// Rotation of the texture's U axis from pattern +X, in radians.
    pub rotation: f32,
}

impl Default for UvMapping {
    fn default() -> Self {
        Self {
            origin: Vec2::ZERO,
            tile_size: Vec2::splat(100.0),
            rotation: 0.0,
        }
    }
}

impl UvMapping {
    /// V runs along the grainline (warp), U across it (weft).
    pub fn aligned_to_grain(grain: &Grainline) -> Self {
        Self {
            origin: grain.origin,
            rotation: grain.angle - std::f32::consts::FRAC_PI_2,
            ..Self::default()
        }
    }

    pub fn map(&self, p: Vec2) -> Vec2 {
        Vec2::from_angle(-self.rotation).rotate(p - self.origin) / self.tile_size
    }

    pub fn unmap(&self, uv: Vec2) -> Vec2 {
        Vec2::from_angle(self.rotation).rotate(uv * self.tile_size) + self.origin
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let m = UvMapping {
            origin: Vec2::new(10.0, -4.0),
            tile_size: Vec2::new(50.0, 80.0),
            rotation: 0.7,
        };
        let p = Vec2::new(123.0, 45.0);
        assert!(m.unmap(m.map(p)).distance(p) < 1e-3);
    }
}

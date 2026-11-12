//! The common result of all avatar importers.

use glam::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpAxis {
    /// Z up if the mesh is clearly taller along Z than along Y.
    #[default]
    Auto,
    Y,
    Z,
}

/// How an imported mesh is brought into the engine's conventions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImportOptions {
    /// Factor from file units to metres. `None` guesses from the height:
    /// a figure 0.5–3 units tall is taken to be in metres, 50–300 in
    /// centimetres, 500–3000 in millimetres, 20–50 in inches… and anything
    /// else is left alone.
    pub scale: Option<f32>,
    pub up: UpAxis,
    /// Move the mesh so that it stands on y = 0, centred on the y axis.
    pub ground: bool,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self { scale: None, up: UpAxis::Auto, ground: true }
    }
}

/// An indexed triangle mesh, counter-clockwise front faces.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImportedMesh {
    pub name: String,
    pub positions: Vec<Vec3>,
    /// One per position. Recomputed from the faces if the file has none.
    pub normals: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
}

impl ImportedMesh {
    /// Axis-aligned bounds `(min, max)`.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        self.positions.iter().fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(lo, hi), &p| (lo.min(p), hi.max(p)))
    }

    /// Area-weighted vertex normals.
    pub fn recompute_normals(&mut self) {
        self.normals.clear();
        self.normals.resize(self.positions.len(), Vec3::ZERO);
        for t in &self.triangles {
            let [a, b, c] = t.map(|i| self.positions[i as usize]);
            let n = (b - a).cross(c - a);
            for &i in t {
                self.normals[i as usize] += n;
            }
        }
        for n in &mut self.normals {
            *n = n.normalize_or(Vec3::Y);
        }
    }

    /// Drops triangles that refer to missing vertices or have no area, and
    /// makes sure there is a normal per vertex.
    pub(crate) fn finish(mut self) -> Self {
        let count = self.positions.len() as u32;
        self.triangles.retain(|t| t.iter().all(|&i| i < count) && t[0] != t[1] && t[1] != t[2] && t[0] != t[2]);
        if self.normals.len() != self.positions.len() || self.normals.iter().any(|n| !n.is_finite() || n.length_squared() < 1e-12) {
            self.recompute_normals();
        }
        self
    }

    /// Applies `options`: up axis, unit scale and grounding, in that order.
    pub fn normalise(&mut self, options: &ImportOptions) {
        if self.positions.is_empty() {
            return;
        }
        let (lo, hi) = self.bounds();
        let size = hi - lo;
        let z_up = match options.up {
            UpAxis::Y => false,
            UpAxis::Z => true,
            UpAxis::Auto => size.z > 1.5 * size.y,
        };
        if z_up {
            // Rotate −90° about x: (x, y, z) → (x, z, −y).
            for v in self.positions.iter_mut().chain(&mut self.normals) {
                *v = Vec3::new(v.x, v.z, -v.y);
            }
        }
        let height = if z_up { size.z } else { size.y };
        let scale = options.scale.unwrap_or(match height {
            h if (0.5..3.0).contains(&h) => 1.0,
            h if (20.0..50.0).contains(&h) => 0.0254,
            h if (50.0..300.0).contains(&h) => 0.01,
            h if (500.0..3000.0).contains(&h) => 0.001,
            _ => 1.0,
        });
        for p in &mut self.positions {
            *p *= scale;
        }
        if options.ground {
            let (lo, hi) = self.bounds();
            let shift = Vec3::new(-0.5 * (lo.x + hi.x), -lo.y, -0.5 * (lo.z + hi.z));
            for p in &mut self.positions {
                *p += shift;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_z_up_centimetre_figure_ends_up_standing_in_metres() {
        // A 40 × 20 × 170 box, corner at (100, 100, 5).
        let mut mesh = ImportedMesh {
            positions: (0..8)
                .map(|i| Vec3::new(100.0 + 40.0 * (i & 1) as f32, 100.0 + 20.0 * ((i >> 1) & 1) as f32, 5.0 + 170.0 * ((i >> 2) & 1) as f32))
                .collect(),
            triangles: vec![[0, 1, 2]],
            ..Default::default()
        }
        .finish();
        mesh.normalise(&ImportOptions::default());
        let (lo, hi) = mesh.bounds();
        assert!((lo - Vec3::new(-0.2, 0.0, -0.1)).length() < 1e-5, "{lo}");
        assert!((hi - Vec3::new(0.2, 1.7, 0.1)).length() < 1e-5, "{hi}");
    }
}

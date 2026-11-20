//! CPU and GPU mesh types of the 3D viewport.

use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec2, Vec3};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// A measured quantity for the technical views (strain, clearance);
    /// unused by the fabric view.
    pub value: f32,
    /// Linear RGB multiplied into the material's base colour.
    pub color: [f32; 3],
    /// Where a garment wears (zero elsewhere): x distance to the piece's
    /// edge (seam or hem) in metres, y how sharply it bulges out (a fold's
    /// ridge, rubbed), z how sharply it dips in (a crease), w the face:
    /// 1 outside, -1 inside, 0 the cut edge.
    pub wear: [f32; 4],
}

impl MeshVertex {
    // Vertex buffer layout for wgpu
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3, // position
            1 => Float32x3, // normal
            2 => Float32x2, // uv
            3 => Float32,   // value
            4 => Float32x3, // color
            5 => Float32x4, // wear
        ],
    };

    pub fn new(position: Vec3, normal: Vec3, uv: Vec2) -> Self {
        Self {
            position: position.to_array(),
            normal: normal.to_array(),
            uv: uv.to_array(),
            value: 0.0,
            color: [1.0; 3],
            wear: [0.0; 4],
        }
    }

    pub fn with_color(mut self, color: [f32; 3]) -> Self {
        self.color = color;
        self
    }

    pub fn with_value(mut self, value: f32) -> Self {
        self.value = value;
        self
    }
}

/// An indexed triangle mesh in world units (metres).
#[derive(Debug, Clone, Default)]
pub struct MeshData {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
}

impl MeshData {
    /// Appends an ellipsoid with the given orientation.
    pub fn add_ellipsoid(&mut self, center: Vec3, radii: Vec3, rotation: Quat, segments: u32) {
        let (rings, sectors) = (segments.max(3), segments.max(3) * 2);
        let base = self.vertices.len() as u32;
        for ring in 0..=rings {
            let v = ring as f32 / rings as f32;
            let (sin_phi, cos_phi) = (v * std::f32::consts::PI).sin_cos();
            for sector in 0..=sectors {
                let u = sector as f32 / sectors as f32;
                let (sin_theta, cos_theta) = (u * std::f32::consts::TAU).sin_cos();
                let unit = Vec3::new(sin_phi * sin_theta, cos_phi, sin_phi * cos_theta);
                // Gradient of the implicit ellipsoid gives the true normal.
                let normal = rotation * (unit / radii).normalize();
                self.vertices.push(MeshVertex::new(
                    center + rotation * (unit * radii),
                    normal,
                    Vec2::new(u, v),
                ));
            }
        }
        let stride = sectors + 1;
        for ring in 0..rings {
            for sector in 0..sectors {
                let a = base + ring * stride + sector;
                let b = a + stride;
                self.indices.extend([a, b, a + 1, a + 1, b, b + 1]);
            }
        }
    }

    /// Appends a horizontal square centred on `center`, facing up.
    pub fn add_ground(&mut self, center: Vec3, half_extent: f32) {
        let base = self.vertices.len() as u32;
        for (x, z) in [(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)] {
            let offset = Vec3::new(x, 0.0, z) * half_extent;
            self.vertices
                .push(MeshVertex::new(center + offset, Vec3::Y, Vec2::new(x, z)));
            self.indices.extend([0, 1, 2, 0, 2, 3].map(|i| base + i));
        }
    }
}

/// How a surface reflects light, beyond the metallic-roughness basics.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Shading {
    #[default]
    Standard,
    /// Light bleeds a little past the terminator and reddens there, as it
    /// does through real skin; a faint sheen at grazing angles.
    Skin,
    /// Strand-shaped highlights (Kajiya-Kay) along strands that run away
    /// from `root`, the crown, in world space; streaks follow the UV U.
    /// The vertex `value` is coverage: 1 solid, towards 0 the strands
    /// thin out (hairline, cut ends).
    Hair { root: [f32; 3] },
    /// Material per vertex, for small props such as buttons: the vertex
    /// colour is the base colour and the UV holds `(roughness, metallic)`.
    Prop,
}

impl Shading {
    /// `[kind, x, y, z]` as the shader reads it.
    pub(crate) fn uniform(&self) -> [f32; 4] {
        match *self {
            Self::Standard => [0.0; 4],
            Self::Skin => [1.0, 0.0, 0.0, 0.0],
            Self::Hair { root: [x, y, z] } => [2.0, x, y, z],
            Self::Prop => [3.0, 0.0, 0.0, 0.0],
        }
    }
}

/// Shading parameters of one draw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceMaterial {
    /// Linear RGB.
    pub base_color: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    /// 0 disables the procedural weave.
    pub weave_strength: f32,
    /// Procedural fabric pattern: 0 plain, 1 denim, 2 leather, 3 corduroy,
    /// 4 wool, 5 knit, 6 satin.
    pub pattern: f32,
    /// Procedural colour fading and variation.
    pub wash: f32,
    /// Procedural yarn/rib repeat scale.
    pub pattern_scale: f32,
    /// Procedural surface relief strength.
    pub relief: f32,
    /// Linear RGB added after lighting (selection highlight).
    pub emissive: [f32; 3],
    pub shading: Shading,
    /// Woven from a draft: the key its textures were added under (see
    /// [`super::SceneRenderer::add_weave`]); 0 for none.
    pub weave_key: u64,
    /// How many times the weave texture repeats per UV unit (U, V).
    pub weave_repeat: [f32; 2],
    /// The yarns of the weave, varied in the shader so the variation never
    /// repeats with the texture: threads across the texture (ends, picks),
    /// irregularity and slubs (0..=1).
    pub weave_yarn: [f32; 4],
    /// How worn a garment is, 0..=1 (see [`MeshVertex::wear`]).
    pub wear: f32,
    /// The textures under `weave_key` are image maps (colour with roughness
    /// in alpha), not a generated weave.
    pub weave_maps: bool,
}

impl Default for SurfaceMaterial {
    fn default() -> Self {
        Self {
            base_color: [0.6; 3],
            roughness: 0.6,
            metallic: 0.0,
            weave_strength: 0.0,
            pattern: 0.0,
            wash: 0.0,
            pattern_scale: 1.0,
            relief: 1.0,
            emissive: [0.0; 3],
            shading: Shading::Standard,
            weave_key: 0,
            weave_repeat: [1.0; 2],
            weave_yarn: [0.0; 4],
            wear: 0.0,
            weave_maps: false,
        }
    }
}

/// What the garment's surface shows.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SurfaceOverlay {
    /// The fabric, lit.
    #[default]
    Fabric,
    /// Strain heatmap: the per-vertex value is the relative elongation of
    /// the mesh edges (0.05 = stretched by 5 %). Blue is relaxed, red is
    /// stretched by `limit` or more.
    Strain { limit: f32 },
    /// Fit map: the per-vertex value is the clearance between fabric and
    /// body in metres. Red is `tight` or closer, blue is `loose` or
    /// further.
    Fit { tight: f32, loose: f32 },
}

impl SurfaceOverlay {
    /// `[mode, value at the blue end, value at the red end, unused]`.
    pub(crate) fn uniform(&self) -> [f32; 4] {
        match *self {
            Self::Fabric => [0.0; 4],
            Self::Strain { limit } => [1.0, 0.0, limit.max(1e-6), 0.0],
            Self::Fit { tight, loose } => [2.0, loose.max(tight + 1e-6), tight, 0.0],
        }
    }
}

/// The colour the heatmap shows for `t` in `0..=1` (0 = blue end,
/// 1 = red end), in sRGB. Mirrors `heat` in `pbr.wgsl`, for legends.
pub fn heat_srgb(t: f32) -> [u8; 3] {
    const STOPS: [[f32; 3]; 5] = [
        [0.02, 0.12, 0.85],
        [0.0, 0.62, 0.78],
        [0.05, 0.70, 0.12],
        [0.95, 0.80, 0.03],
        [0.90, 0.04, 0.02],
    ];
    let x = t.clamp(0.0, 1.0) * 4.0;
    let i = (x as usize).min(3);
    let f = x - i as f32;
    std::array::from_fn(|c| {
        let linear = STOPS[i][c] + (STOPS[i + 1][c] - STOPS[i][c]) * f;
        let srgb = if linear <= 0.0031308 {
            linear * 12.92
        } else {
            1.055 * linear.powf(1.0 / 2.4) - 0.055
        };
        (srgb * 255.0).round() as u8
    })
}

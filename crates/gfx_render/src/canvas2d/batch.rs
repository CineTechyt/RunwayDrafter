//! CPU-side geometry accumulation for the canvas: everything (lines, handles,
//! fills) becomes coloured triangles in one draw call.

use bytemuck::{Pod, Zeroable};
use glam::Vec2;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CanvasVertex {
    pub position: [f32; 2],
    /// Linear RGB, straight alpha.
    pub color: [f32; 4],
}

impl CanvasVertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4],
    };
}

/// Draw list for one canvas frame, in world (mm) coordinates.
#[derive(Debug, Default, Clone)]
pub struct CanvasBatch {
    pub vertices: Vec<CanvasVertex>,
    pub indices: Vec<u32>,
}

impl CanvasBatch {
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    fn push(&mut self, position: Vec2, color: [f32; 4]) -> u32 {
        self.vertices.push(CanvasVertex {
            position: position.to_array(),
            color,
        });
        self.vertices.len() as u32 - 1
    }

    /// A line of the given world-space width.
    pub fn segment(&mut self, a: Vec2, b: Vec2, width: f32, color: [f32; 4]) {
        let side = (b - a).normalize_or_zero().perp() * (width * 0.5);
        let base = self.push(a + side, color);
        self.push(a - side, color);
        self.push(b - side, color);
        self.push(b + side, color);
        self.indices.extend([0, 1, 2, 0, 2, 3].map(|i| base + i));
    }

    pub fn polyline(&mut self, points: &[Vec2], closed: bool, width: f32, color: [f32; 4]) {
        for pair in points.windows(2) {
            self.segment(pair[0], pair[1], width, color);
        }
        if let (true, [first, .., last]) = (closed, points) {
            self.segment(*last, *first, width, color);
        }
    }

    /// A dashed line; `dash` and `gap` are world-space lengths.
    pub fn dashed_polyline(
        &mut self,
        points: &[Vec2],
        closed: bool,
        width: f32,
        dash: f32,
        gap: f32,
        color: [f32; 4],
    ) {
        let period = (dash + gap).max(1e-3);
        let mut phase = 0.0;
        let count = if closed {
            points.len()
        } else {
            points.len().saturating_sub(1)
        };
        for i in 0..count {
            let (a, b) = (points[i], points[(i + 1) % points.len()]);
            let length = a.distance(b);
            let mut at = -phase;
            while at < length {
                let (start, end) = (at.max(0.0), (at + dash).min(length));
                if end > start {
                    self.segment(
                        a.lerp(b, start / length),
                        a.lerp(b, end / length),
                        width,
                        color,
                    );
                }
                at += period;
            }
            phase = (phase + length) % period;
        }
    }

    /// An axis-aligned square (anchor handles).
    pub fn square(&mut self, center: Vec2, half: f32, color: [f32; 4]) {
        let base = self.push(center + Vec2::new(-half, -half), color);
        self.push(center + Vec2::new(half, -half), color);
        self.push(center + Vec2::new(half, half), color);
        self.push(center + Vec2::new(-half, half), color);
        self.indices.extend([0, 1, 2, 0, 2, 3].map(|i| base + i));
    }

    /// A filled circle (Bézier control handles).
    pub fn disc(&mut self, center: Vec2, radius: f32, color: [f32; 4]) {
        const SIDES: u32 = 16;
        let base = self.push(center, color);
        for i in 0..SIDES {
            let angle = i as f32 / SIDES as f32 * std::f32::consts::TAU;
            self.push(center + Vec2::from_angle(angle) * radius, color);
        }
        for i in 0..SIDES {
            self.indices
                .extend([base, base + 1 + i, base + 1 + (i + 1) % SIDES]);
        }
    }

    /// Indexed triangles with one colour (piece fills).
    pub fn triangles(
        &mut self,
        positions: &[Vec2],
        indices: impl IntoIterator<Item = u32>,
        color: [f32; 4],
    ) {
        let base = self.vertices.len() as u32;
        self.vertices.extend(positions.iter().map(|p| CanvasVertex {
            position: p.to_array(),
            color,
        }));
        self.indices.extend(indices.into_iter().map(|i| base + i));
    }
}

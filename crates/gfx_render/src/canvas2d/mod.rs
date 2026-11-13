//! Viewport A: the 2D pattern editing canvas.

pub mod batch;
pub mod scene;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2};

pub use batch::{CanvasBatch, CanvasVertex};
pub use scene::{CanvasOverlay, LineHighlight};

use crate::camera::Camera2D;
use crate::context::GpuContext;
use crate::target::{GrowableBuffer, RenderTarget};
use crate::{ViewportRenderer, srgba};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
}

pub struct CanvasRenderer {
    target: RenderTarget,
    pipeline: wgpu::RenderPipeline,
    globals: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertices: GrowableBuffer,
    indices: GrowableBuffer,
    index_count: u32,
    pub clear_color: [f32; 4],
}

impl CanvasRenderer {
    pub fn new(gpu: &GpuContext, size: [u32; 2]) -> Self {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("canvas2d shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("canvas.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("canvas2d globals layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("canvas2d globals"),
            size: size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("canvas2d globals"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("canvas2d pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("canvas2d pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(CanvasVertex::LAYOUT)],
            },
            primitive: wgpu::PrimitiveState::default(), // triangle list, no culling
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: RenderTarget::SAMPLE_COUNT,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: RenderTarget::COLOR_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            target: RenderTarget::new(gpu, "canvas2d target", size, false),
            pipeline,
            globals,
            bind_group,
            vertices: GrowableBuffer::new(gpu, "canvas2d vertices", wgpu::BufferUsages::VERTEX),
            indices: GrowableBuffer::new(gpu, "canvas2d indices", wgpu::BufferUsages::INDEX),
            index_count: 0,
            clear_color: srgba(30, 32, 38, 1.0),
        }
    }

    /// Size of the target as a vector, for camera maths.
    pub fn viewport(&self) -> Vec2 {
        let [w, h] = self.target.size();
        Vec2::new(w as f32, h as f32)
    }

    /// Uploads this frame's geometry and camera.
    pub fn prepare(&mut self, gpu: &GpuContext, batch: &CanvasBatch, camera: &Camera2D) {
        let view_proj: Mat4 = camera.view_proj(self.viewport());
        gpu.queue.write_buffer(
            &self.globals,
            0,
            bytemuck::bytes_of(&Globals {
                view_proj: view_proj.to_cols_array_2d(),
            }),
        );
        self.vertices
            .write(gpu, bytemuck::cast_slice(&batch.vertices));
        self.indices
            .write(gpu, bytemuck::cast_slice(&batch.indices));
        self.index_count = batch.indices.len() as u32;
    }
}

impl ViewportRenderer for CanvasRenderer {
    fn target(&self) -> &RenderTarget {
        &self.target
    }

    fn resize(&mut self, gpu: &GpuContext, size: [u32; 2]) -> bool {
        self.target.resize(gpu, size)
    }

    fn render(&mut self, _gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder) {
        let [r, g, b, a] = self.clear_color.map(f64::from);
        let mut pass = self.target.begin_pass(encoder, wgpu::Color { r, g, b, a });
        if self.index_count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertices.buffer().slice(..));
        pass.set_index_buffer(self.indices.buffer().slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

//! Shadow maps: the depth of the scene as a light sees it, so surfaces
//! hidden from the light can be left out of its light.
//!
//! Lights are directional; each gets an orthographic view fitted around
//! the avatar and garment. Several maps live in the layers of one array
//! texture (the key light, and in the studio render the dome light).

use glam::camera::rh::{proj, view};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::context::GpuContext;
use crate::target::RenderTarget;

use super::mesh::MeshVertex;
use super::{Draw, Style};

pub(crate) const FORMAT: wgpu::TextureFormat = RenderTarget::DEPTH_FORMAT;

/// The layers of a shadow map array.
pub(crate) struct ShadowMaps {
    layers: Vec<wgpu::TextureView>,
    pub array: wgpu::TextureView,
}

impl ShadowMaps {
    pub fn new(gpu: &GpuContext, size: u32, layers: u32) -> Self {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow maps"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let layer_views = (0..layers)
            .map(|layer| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shadow map layer"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let array = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow map array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        Self {
            layers: layer_views,
            array,
        }
    }
}

/// The camera of a directional light coming from `direction` (pointing
/// towards the light), seeing everything within `radius` of `center`.
pub(crate) fn light_matrix(direction: Vec3, center: Vec3, radius: f32) -> Mat4 {
    let direction = direction.normalize_or(Vec3::Y);
    let up = if direction.y.abs() > 0.95 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let eye = center + direction * (radius + 1.0);
    proj::directx::orthographic(-radius, radius, -radius, radius, 0.05, 2.0 * radius + 2.0)
        * view::look_at_mat4(eye, center, up)
}

/// A light's camera on the GPU.
pub(crate) struct LightSlot {
    buffer: wgpu::Buffer,
    group: wgpu::BindGroup,
}

impl LightSlot {
    pub fn new(gpu: &GpuContext, kit: &ShadowKit) -> Self {
        let buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("shadow light"),
                contents: bytemuck::bytes_of(&Mat4::IDENTITY.to_cols_array_2d()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow light"),
            layout: &kit.light_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self { buffer, group }
    }

    pub fn write(&self, gpu: &GpuContext, matrix: Mat4) {
        gpu.queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::bytes_of(&matrix.to_cols_array_2d()),
        );
    }
}

/// The depth-only pipeline that draws shadow maps.
pub(crate) struct ShadowKit {
    light_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    pub sampler: wgpu::Sampler,
}

impl ShadowKit {
    pub fn new(device: &wgpu::Device, object_layout: &wgpu::BindGroupLayout) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shadow.wgsl").into()),
        });
        let light_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow light layout"),
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
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow pipeline layout"),
            bind_group_layouts: &[Some(&light_layout), Some(object_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_shadow"),
                compilation_options: Default::default(),
                buffers: &[Some(MeshVertex::LAYOUT)],
            },
            // Cloth is seen from both sides, so it shadows from both.
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        Self {
            light_layout,
            pipeline,
            sampler,
        }
    }

    /// Draws `draws` into layer `layer` of `maps` as `light` sees them.
    /// Hair is left out: its sheets would shadow as solid plates.
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        maps: &ShadowMaps,
        layer: usize,
        light: &LightSlot,
        draws: &[Draw],
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shadow map"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &maps.layers[layer],
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &light.group, &[]);
        for draw in draws.iter().filter(|d| d.style != Style::Hair) {
            pass.set_bind_group(1, &draw.object, &[]);
            pass.set_vertex_buffer(0, draw.vertices.slice(..));
            pass.set_index_buffer(draw.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(draw.range.clone(), 0, 0..1);
        }
    }
}

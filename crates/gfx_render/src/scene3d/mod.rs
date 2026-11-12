//! Viewport B: PBR view of the avatar and the draped garment.

pub mod avatar;
pub mod mesh;

use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use cad_core::PieceId;
use glam::{Mat4, Vec3};
use topology_sync::GarmentMesh;
use wgpu::util::DeviceExt;

pub use mesh::{MeshData, MeshVertex, SurfaceMaterial, SurfaceOverlay};

use crate::camera::OrbitCamera;
use crate::context::GpuContext;
use crate::target::{GrowableBuffer, RenderTarget};
use crate::{ViewportRenderer, srgba};

/// Weave repeats per UV unit (one UV unit is one texture tile of the fabric).
const WEAVE_THREADS: f32 = 30.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    key_direction: [f32; 4],
    key_color: [f32; 4],
    fill_direction: [f32; 4],
    fill_color: [f32; 4],
    sky: [f32; 4],
    ground: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ObjectUniform {
    model: [[f32; 4]; 4],
    base_color: [f32; 4],
    params: [f32; 4],
    emissive: [f32; 4],
    overlay: [f32; 4],
}

impl ObjectUniform {
    fn new(model: Mat4, m: &SurfaceMaterial) -> Self {
        let [r, g, b] = m.base_color;
        let [er, eg, eb] = m.emissive;
        Self {
            model: model.to_cols_array_2d(),
            base_color: [r, g, b, 1.0],
            params: [m.roughness, m.metallic, m.weave_strength, WEAVE_THREADS],
            emissive: [er, eg, eb, 0.0],
            overlay: [0.0; 4],
        }
    }
}

/// Per-draw uniform buffer and its bind group.
struct ObjectBinding {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl ObjectBinding {
    fn new(gpu: &GpuContext, layout: &wgpu::BindGroupLayout, uniform: &ObjectUniform) -> Self {
        let buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene3d object"),
            contents: bytemuck::bytes_of(uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene3d object"),
            layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });
        Self { buffer, bind_group }
    }
}

/// Immutable geometry such as the avatar or the floor.
struct StaticDraw {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    binding: ObjectBinding,
}

/// One pattern piece inside the shared garment buffers.
struct PieceDraw {
    piece: PieceId,
    indices: Range<u32>,
    binding: ObjectBinding,
}

pub struct SceneRenderer {
    target: RenderTarget,
    opaque_pipeline: wgpu::RenderPipeline,
    cloth_pipeline: wgpu::RenderPipeline,
    globals: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    object_layout: wgpu::BindGroupLayout,
    statics: Vec<StaticDraw>,
    avatar: Option<StaticDraw>,
    garment_vertices: GrowableBuffer,
    garment_indices: GrowableBuffer,
    garment_generation: Option<u64>,
    pieces: Vec<PieceDraw>,
    scratch: Vec<MeshVertex>,
    pub clear_color: [f32; 4],
    /// What the garment's surface shows; the avatar is always lit.
    pub overlay: SurfaceOverlay,
}

impl SceneRenderer {
    pub fn new(gpu: &GpuContext, size: [u32; 2]) -> Self {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene3d pbr shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("pbr.wgsl").into()),
        });
        let uniform_layout = |label, visibility| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            })
        };
        let stages = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let globals_layout = uniform_layout("scene3d globals layout", stages);
        let object_layout = uniform_layout("scene3d object layout", stages);

        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene3d globals"),
            size: size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene3d globals"),
            layout: &globals_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: globals.as_entire_binding() }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene3d pipeline layout"),
            bind_group_layouts: &[Some(&globals_layout), Some(&object_layout)],
            immediate_size: 0,
        });
        let pipeline = |label, cull_mode| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(MeshVertex::LAYOUT)],
                },
                primitive: wgpu::PrimitiveState { cull_mode, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: RenderTarget::DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState { count: RenderTarget::SAMPLE_COUNT, ..Default::default() },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: RenderTarget::COLOR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        Self {
            target: RenderTarget::new(gpu, "scene3d target", size, true),
            opaque_pipeline: pipeline("scene3d opaque pipeline", Some(wgpu::Face::Back)),
            // Cloth is a single-layer surface seen from both sides.
            cloth_pipeline: pipeline("scene3d cloth pipeline", None),
            globals,
            globals_bind_group,
            object_layout,
            statics: Vec::new(),
            avatar: None,
            garment_vertices: GrowableBuffer::new(gpu, "garment vertices", wgpu::BufferUsages::VERTEX),
            garment_indices: GrowableBuffer::new(gpu, "garment indices", wgpu::BufferUsages::INDEX),
            garment_generation: None,
            pieces: Vec::new(),
            scratch: Vec::new(),
            clear_color: srgba(38, 41, 48, 1.0),
            overlay: SurfaceOverlay::Fabric,
        }
    }

    fn static_draw(&self, gpu: &GpuContext, mesh: &MeshData, material: SurfaceMaterial) -> StaticDraw {
        let buffer = |label, contents, usage| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage })
        };
        StaticDraw {
            vertices: buffer("static vertices", bytemuck::cast_slice(&mesh.vertices), wgpu::BufferUsages::VERTEX),
            indices: buffer("static indices", bytemuck::cast_slice(&mesh.indices), wgpu::BufferUsages::INDEX),
            index_count: mesh.indices.len() as u32,
            binding: ObjectBinding::new(gpu, &self.object_layout, &ObjectUniform::new(Mat4::IDENTITY, &material)),
        }
    }

    /// Adds immutable scenery (floor, props).
    pub fn add_static(&mut self, gpu: &GpuContext, mesh: &MeshData, material: SurfaceMaterial) {
        let draw = self.static_draw(gpu, mesh, material);
        self.statics.push(draw);
    }

    /// Replaces the avatar. Called whenever the body is edited.
    pub fn set_avatar(&mut self, gpu: &GpuContext, mesh: &MeshData, material: SurfaceMaterial) {
        self.avatar = Some(self.static_draw(gpu, mesh, material));
    }

    /// Uploads the garment. Vertex attributes are always rewritten (draping
    /// moves them every step); indices and per-piece draws only when the mesh
    /// was reassembled. `values` are the per-vertex measurements of the
    /// technical views (see [`SurfaceOverlay`]); missing ones read as zero.
    pub fn sync_garment(&mut self, gpu: &GpuContext, mesh: &GarmentMesh, values: &[f32]) {
        self.scratch.clear();
        self.scratch.extend((0..mesh.vertex_count()).map(|i| {
            MeshVertex::new(mesh.positions[i], mesh.normals[i], mesh.uvs[i])
                .with_value(values.get(i).copied().unwrap_or(0.0))
        }));
        self.garment_vertices.write(gpu, bytemuck::cast_slice(&self.scratch));

        if self.garment_generation != Some(mesh.generation) {
            self.garment_generation = Some(mesh.generation);
            self.garment_indices.write(gpu, bytemuck::cast_slice(&mesh.indices));
            let placeholder = ObjectUniform::new(Mat4::IDENTITY, &SurfaceMaterial::default());
            self.pieces = mesh
                .pieces
                .iter()
                .filter(|span| !span.indices.is_empty())
                .map(|span| PieceDraw {
                    piece: span.piece,
                    indices: span.indices.clone(),
                    binding: ObjectBinding::new(gpu, &self.object_layout, &placeholder),
                })
                .collect();
        }
    }

    /// Uploads this frame's camera, lights and fabric materials.
    pub fn prepare(&mut self, gpu: &GpuContext, camera: &OrbitCamera, material_of: impl Fn(PieceId) -> SurfaceMaterial) {
        let eye = camera.eye();
        let direction = |v: Vec3| v.normalize().extend(0.0).to_array();
        let globals = Globals {
            view_proj: camera.view_proj(self.target.aspect()).to_cols_array_2d(),
            eye: eye.extend(1.0).to_array(),
            key_direction: direction(Vec3::new(0.45, 0.8, 0.55)),
            key_color: [3.0, 2.9, 2.75, 0.0],
            fill_direction: direction(Vec3::new(-0.6, 0.25, -0.5)),
            fill_color: [0.55, 0.62, 0.75, 0.0],
            sky: [0.34, 0.37, 0.43, 0.0],
            ground: [0.12, 0.11, 0.10, 0.0],
        };
        gpu.queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));
        for draw in &self.pieces {
            let mut uniform = ObjectUniform::new(Mat4::IDENTITY, &material_of(draw.piece));
            uniform.overlay = self.overlay.uniform();
            gpu.queue.write_buffer(&draw.binding.buffer, 0, bytemuck::bytes_of(&uniform));
        }
    }
}

impl ViewportRenderer for SceneRenderer {
    fn target(&self) -> &RenderTarget {
        &self.target
    }

    fn resize(&mut self, gpu: &GpuContext, size: [u32; 2]) -> bool {
        self.target.resize(gpu, size)
    }

    fn render(&mut self, _gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder) {
        let [r, g, b, a] = self.clear_color.map(f64::from);
        let mut pass = self.target.begin_pass(encoder, wgpu::Color { r, g, b, a });
        pass.set_bind_group(0, &self.globals_bind_group, &[]);

        pass.set_pipeline(&self.opaque_pipeline);
        for draw in self.statics.iter().chain(&self.avatar) {
            pass.set_bind_group(1, &draw.binding.bind_group, &[]);
            pass.set_vertex_buffer(0, draw.vertices.slice(..));
            pass.set_index_buffer(draw.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..draw.index_count, 0, 0..1);
        }

        if !self.pieces.is_empty() {
            pass.set_pipeline(&self.cloth_pipeline);
            pass.set_vertex_buffer(0, self.garment_vertices.buffer().slice(..));
            pass.set_index_buffer(self.garment_indices.buffer().slice(..), wgpu::IndexFormat::Uint32);
            for draw in &self.pieces {
                pass.set_bind_group(1, &draw.binding.bind_group, &[]);
                pass.draw_indexed(draw.indices.clone(), 0, 0..1);
            }
        }
    }
}

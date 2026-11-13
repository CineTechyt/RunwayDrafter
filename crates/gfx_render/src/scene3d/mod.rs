//! Viewport B: PBR view of the avatar and the draped garment.

pub mod avatar;
pub mod mesh;
pub mod props;

use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use cad_core::PieceId;
use glam::{Mat4, Vec3};
use topology_sync::GarmentMesh;
use wgpu::util::DeviceExt;

pub use mesh::{MeshData, MeshVertex, Shading, SurfaceMaterial, SurfaceOverlay};
pub use props::garment_props;

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
    shading: [f32; 4],
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
            shading: m.shading.uniform(),
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
    /// Hair: no back-face culling (open sheets) and alpha to coverage.
    double_sided: bool,
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
    hair_pipeline: wgpu::RenderPipeline,
    globals: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    object_layout: wgpu::BindGroupLayout,
    statics: Vec<StaticDraw>,
    /// Skin, eyes, hair: whatever makes up the avatar.
    avatar: Vec<StaticDraw>,
    /// Buttons and buttonholes on the garment; rebuilt as it drapes.
    props_vertices: GrowableBuffer,
    props_indices: GrowableBuffer,
    props_index_count: u32,
    props_binding: ObjectBinding,
    garment_vertices: GrowableBuffer,
    garment_indices: GrowableBuffer,
    garment_generation: Option<u64>,
    /// How the garment is drawn with thickness, for its current topology.
    shell: Shell,
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
        let pipeline = |label, cull_mode, alpha_to_coverage: bool| {
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
                multisample: wgpu::MultisampleState {
                    count: RenderTarget::SAMPLE_COUNT,
                    alpha_to_coverage_enabled: alpha_to_coverage,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: RenderTarget::COLOR_FORMAT,
                        blend: None,
                        // Alpha only steers coverage; the image stays opaque.
                        write_mask: if alpha_to_coverage { wgpu::ColorWrites::COLOR } else { wgpu::ColorWrites::ALL },
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        let props_binding = ObjectBinding::new(
            gpu,
            &object_layout,
            &ObjectUniform::new(Mat4::IDENTITY, &SurfaceMaterial { shading: Shading::Prop, ..SurfaceMaterial::default() }),
        );
        Self {
            target: RenderTarget::new(gpu, "scene3d target", size, true),
            opaque_pipeline: pipeline("scene3d opaque pipeline", Some(wgpu::Face::Back), false),
            // Cloth is a single-layer surface seen from both sides.
            cloth_pipeline: pipeline("scene3d cloth pipeline", None, false),
            // Hair: open sheets whose edges thin out strand by strand. The
            // multisampled coverage fades them without sorting.
            hair_pipeline: pipeline("scene3d hair pipeline", None, true),
            globals,
            globals_bind_group,
            object_layout,
            statics: Vec::new(),
            avatar: Vec::new(),
            props_vertices: GrowableBuffer::new(gpu, "prop vertices", wgpu::BufferUsages::VERTEX),
            props_indices: GrowableBuffer::new(gpu, "prop indices", wgpu::BufferUsages::INDEX),
            props_index_count: 0,
            props_binding,
            garment_vertices: GrowableBuffer::new(gpu, "garment vertices", wgpu::BufferUsages::VERTEX),
            garment_indices: GrowableBuffer::new(gpu, "garment indices", wgpu::BufferUsages::INDEX),
            garment_generation: None,
            shell: Shell::default(),
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
            double_sided: matches!(material.shading, Shading::Hair { .. }),
        }
    }

    /// Adds immutable scenery (floor, props).
    pub fn add_static(&mut self, gpu: &GpuContext, mesh: &MeshData, material: SurfaceMaterial) {
        let draw = self.static_draw(gpu, mesh, material);
        self.statics.push(draw);
    }

    /// Replaces the avatar. Called whenever the body is edited.
    pub fn set_avatar(&mut self, gpu: &GpuContext, mesh: &MeshData, material: SurfaceMaterial) {
        self.set_avatar_parts(gpu, &[(mesh, material)]);
    }

    /// Replaces the avatar with several meshes of their own material
    /// (skin, eyes, hair). Empty meshes are skipped.
    pub fn set_avatar_parts(&mut self, gpu: &GpuContext, parts: &[(&MeshData, SurfaceMaterial)]) {
        self.avatar = parts
            .iter()
            .filter(|(mesh, _)| !mesh.indices.is_empty())
            .map(|(mesh, material)| self.static_draw(gpu, mesh, *material))
            .collect();
    }

    /// Replaces the props on the garment (see [`garment_props`]); their
    /// material is per vertex ([`Shading::Prop`]).
    pub fn set_props(&mut self, gpu: &GpuContext, mesh: &MeshData) {
        self.props_index_count = mesh.indices.len() as u32;
        if !mesh.indices.is_empty() {
            self.props_vertices.write(gpu, bytemuck::cast_slice(&mesh.vertices));
            self.props_indices.write(gpu, bytemuck::cast_slice(&mesh.indices));
        }
    }

    /// Uploads the garment, drawn with the thickness of its fabric
    /// (`thickness`, metres, per piece; see [`Shell`]). Vertex attributes
    /// are always rewritten (draping moves them every step); indices and
    /// per-piece draws only when the mesh was reassembled. `values` are the per-vertex measurements of the
    /// technical views (see [`SurfaceOverlay`]); missing ones read as zero.
    pub fn sync_garment(&mut self, gpu: &GpuContext, mesh: &GarmentMesh, values: &[f32], thickness: impl Fn(PieceId) -> f32) {
        if self.garment_generation != Some(mesh.generation) {
            self.garment_generation = Some(mesh.generation);
            self.shell = Shell::new(mesh);
            self.garment_indices.write(gpu, bytemuck::cast_slice(&self.shell.indices));
            let placeholder = ObjectUniform::new(Mat4::IDENTITY, &SurfaceMaterial::default());
            self.pieces = self
                .shell
                .pieces
                .iter()
                .filter(|(_, indices)| !indices.is_empty())
                .map(|(piece, indices)| PieceDraw {
                    piece: *piece,
                    indices: indices.clone(),
                    binding: ObjectBinding::new(gpu, &self.object_layout, &placeholder),
                })
                .collect();
        }
        self.shell.vertices(mesh, values, &thickness, &mut self.scratch);
        self.garment_vertices.write(gpu, bytemuck::cast_slice(&self.scratch));
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

        for draw in self.statics.iter().chain(&self.avatar) {
            pass.set_pipeline(if draw.double_sided { &self.hair_pipeline } else { &self.opaque_pipeline });
            pass.set_bind_group(1, &draw.binding.bind_group, &[]);
            pass.set_vertex_buffer(0, draw.vertices.slice(..));
            pass.set_index_buffer(draw.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..draw.index_count, 0, 0..1);
        }

        if self.props_index_count > 0 {
            pass.set_pipeline(&self.cloth_pipeline);
            pass.set_bind_group(1, &self.props_binding.bind_group, &[]);
            pass.set_vertex_buffer(0, self.props_vertices.buffer().slice(..));
            pass.set_index_buffer(self.props_indices.buffer().slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.props_index_count, 0, 0..1);
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

/// The garment drawn as cloth with a body to it: each piece gets an outer
/// face half its thickness outside the simulated surface, an inner face
/// half inside, and a rim joining them along every free edge (hems,
/// armholes, openings), lit as the cut edge it is.
#[derive(Debug, Default)]
struct Shell {
    /// Per piece: its vertex range in the simulated mesh, its free edges as
    /// `(a, b, the third vertex of their triangle)`, and where its shell
    /// vertices start.
    layout: Vec<(PieceId, std::ops::Range<u32>, Vec<[u32; 3]>, u32)>,
    indices: Vec<u32>,
    /// Index range of each piece's draw.
    pieces: Vec<(PieceId, Range<u32>)>,
}

impl Shell {
    fn new(mesh: &GarmentMesh) -> Self {
        let mut shell = Shell::default();
        let mut base = 0u32;
        for span in &mesh.pieces {
            let triangles: Vec<[u32; 3]> = mesh.indices[span.indices.start as usize..span.indices.end as usize]
                .chunks_exact(3)
                .map(|t| [t[0], t[1], t[2]])
                .collect();
            // Free edges: used by one triangle only.
            let mut uses: std::collections::HashMap<(u32, u32), (u32, u32, u32, u32)> = std::collections::HashMap::new();
            for t in &triangles {
                for k in 0..3 {
                    let (a, b, c) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
                    let entry = uses.entry((a.min(b), a.max(b))).or_insert((a, b, c, 0));
                    entry.3 += 1;
                }
            }
            let mut edges: Vec<[u32; 3]> = uses.values().filter(|e| e.3 == 1).map(|e| [e.0, e.1, e.2]).collect();
            edges.sort_unstable();

            let count = span.vertices.len() as u32;
            let local = |v: u32| v - span.vertices.start;
            let first = shell.indices.len() as u32;
            // Outer face as simulated, inner face turned round.
            for t in &triangles {
                shell.indices.extend(t.map(|v| base + local(v)));
            }
            for t in &triangles {
                shell.indices.extend([t[0], t[2], t[1]].map(|v| base + count + local(v)));
            }
            // Rim: a quad of its own four vertices per free edge.
            let rim = base + 2 * count;
            for (k, _) in edges.iter().enumerate() {
                let q = rim + 4 * k as u32;
                shell.indices.extend([q, q + 1, q + 2, q + 1, q + 3, q + 2]);
            }
            shell.pieces.push((span.piece, first..shell.indices.len() as u32));
            shell.layout.push((span.piece, span.vertices.clone(), edges, base));
            base = rim + 4 * shell.layout.last().map_or(0, |l| l.2.len() as u32);
        }
        shell
    }

    fn vertices(&self, mesh: &GarmentMesh, values: &[f32], thickness: &impl Fn(PieceId) -> f32, out: &mut Vec<MeshVertex>) {
        out.clear();
        let vertex = |i: usize, offset: f32, normal: Vec3| {
            MeshVertex::new(mesh.positions[i] + mesh.normals[i] * offset, normal, mesh.uvs[i]).with_value(values.get(i).copied().unwrap_or(0.0))
        };
        for (piece, range, edges, _) in &self.layout {
            let half = 0.5 * thickness(*piece);
            let range = range.start as usize..range.end as usize;
            out.extend(range.clone().map(|i| vertex(i, half, mesh.normals[i])));
            out.extend(range.clone().map(|i| vertex(i, -half, -mesh.normals[i])));
            for &[a, b, c] in edges {
                let (a, b, c) = (a as usize, b as usize, c as usize);
                // Facing out of the piece, across its edge.
                let along = mesh.positions[b] - mesh.positions[a];
                let up = (mesh.normals[a] + mesh.normals[b]).normalize_or(Vec3::Y);
                let mut out_of = along.cross(up).normalize_or(Vec3::X);
                if out_of.dot(mesh.positions[a] - mesh.positions[c]) < 0.0 {
                    out_of = -out_of;
                }
                // Wound so the rim faces out whichever way the edge runs.
                let quad = [vertex(a, half, out_of), vertex(b, half, out_of), vertex(a, -half, out_of), vertex(b, -half, out_of)];
                let p = |v: &MeshVertex| Vec3::from(v.position);
                let facing = (p(&quad[1]) - p(&quad[0])).cross(p(&quad[2]) - p(&quad[0])).dot(out_of);
                if facing >= 0.0 {
                    out.extend(quad);
                } else {
                    out.extend([quad[1], quad[0], quad[3], quad[2]]);
                }
            }
        }
    }
}

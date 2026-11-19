//! Viewport B: PBR view of the avatar and the draped garment.
//!
//! The live view draws with one key light, which casts shadows, a fill
//! light and a flat ambient term: cheap enough for every frame while the
//! cloth drapes. [`StudioRender`] makes a still of the same scene with the
//! light of a photo studio (see [`studio`]).

pub mod avatar;
pub mod mesh;
pub mod props;
mod refine;
mod shadow;
pub mod stitches;
pub mod studio;
pub mod weave;

use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use cad_core::PieceId;
use glam::{Mat4, Vec3};
use topology_sync::GarmentMesh;
use wgpu::util::DeviceExt;

pub use mesh::{MeshData, MeshVertex, Shading, SurfaceMaterial, SurfaceOverlay};
pub use props::garment_props;
pub use stitches::Stitches;
pub use studio::StudioRender;

use crate::camera::OrbitCamera;
use crate::context::GpuContext;
use crate::target::{GrowableBuffer, RenderTarget};
use crate::{ViewportRenderer, srgba};

use shadow::{LightSlot, ShadowKit, ShadowMaps, light_matrix};
use studio::StudioKit;

/// How deep a scan's height map reaches at relief 1, metres (as
/// `MAP_RELIEF` in pbr.wgsl).
const MAP_RELIEF: f32 = 0.0015;

/// Resolution of the live view's shadow map.
const LIVE_SHADOW_SIZE: u32 = 2048;

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
    /// Where the key and the fill light see the scene from (shadow maps).
    shadow_key: [[f32; 4]; 4],
    shadow_fill: [[f32; 4]; 4],
    /// x: lights that cast shadows (0, 1: key, 2: key and fill), y: 1 for
    /// the studio render (linear output, no tone mapping), z: the studio
    /// sample, w: the studio's extras (corduroy depth) on.
    options: [f32; 4],
    /// x: a shadow map texel in metres, y: key filter radius, z: fill
    /// filter radius (texels).
    shadow: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct ObjectUniform {
    model: [[f32; 4]; 4],
    base_color: [f32; 4],
    params: [f32; 4],
    emissive: [f32; 4],
    overlay: [f32; 4],
    shading: [f32; 4],
    fabric: [f32; 4],
    weave: [f32; 4],
    yarn: [f32; 4],
}

impl ObjectUniform {
    fn new(model: Mat4, m: &SurfaceMaterial) -> Self {
        let [r, g, b] = m.base_color;
        let [er, eg, eb] = m.emissive;
        Self {
            model: model.to_cols_array_2d(),
            base_color: [r, g, b, 1.0],
            params: [
                m.roughness,
                m.metallic,
                m.weave_strength,
                m.pattern + m.wash.clamp(0.0, 1.0) * 0.1,
            ],
            fabric: [m.pattern_scale.max(0.1), m.relief.clamp(0.0, 2.0), 0.0, m.wear.clamp(0.0, 1.0)],
            emissive: [er, eg, eb, 0.0],
            overlay: [0.0; 4],
            shading: m.shading.uniform(),
            weave: [
                m.weave_repeat[0],
                m.weave_repeat[1],
                match (m.weave_key != 0, m.weave_maps) {
                    (false, _) => 0.0,
                    (true, false) => 1.0,
                    (true, true) => 2.0,
                },
                m.relief.clamp(0.0, 2.0),
            ],
            yarn: m.weave_yarn,
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
        let buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scene3d object"),
                contents: bytemuck::bytes_of(uniform),
                // COPY_SRC: the studio render keeps a copy.
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        let bind_group = object_group(&gpu.device, layout, &buffer);
        Self { buffer, bind_group }
    }
}

fn object_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene3d object"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    })
}

/// The globals with the shadow maps they refer to.
fn globals_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
    shadows: &ShadowMaps,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene3d globals"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&shadows.array),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
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
    /// The weave whose textures it shows (0: none).
    weave: u64,
    /// What was last written to its buffer.
    uniform: ObjectUniform,
}

/// How a draw is rasterised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Style {
    /// Closed, one-sided surfaces.
    Opaque,
    /// Seen from both sides.
    Cloth,
    /// Open sheets whose edges thin out strand by strand.
    Hair,
}

/// Everything one draw call needs: handles only, cheap to clone.
#[derive(Clone)]
pub(crate) struct Draw {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    range: Range<u32>,
    object: wgpu::BindGroup,
    weave: wgpu::BindGroup,
    style: Style,
}

/// The pipelines of one kind of target.
pub(crate) struct Pipelines {
    opaque: wgpu::RenderPipeline,
    cloth: wgpu::RenderPipeline,
    hair: wgpu::RenderPipeline,
}

impl Pipelines {
    fn new(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        format: wgpu::TextureFormat,
        samples: u32,
    ) -> Self {
        let pipeline = |label, cull_mode, alpha_to_coverage: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(MeshVertex::LAYOUT)],
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: RenderTarget::DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: samples,
                    alpha_to_coverage_enabled: alpha_to_coverage,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        // Alpha only steers coverage; the image stays opaque.
                        write_mask: if alpha_to_coverage {
                            wgpu::ColorWrites::COLOR
                        } else {
                            wgpu::ColorWrites::ALL
                        },
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        // Without multisampling (the studio render) hair thins out by
        // discarding: the samples average it.
        let coverage = samples > 1;
        Self {
            opaque: pipeline("scene3d opaque pipeline", Some(wgpu::Face::Back), false),
            // Cloth is a single-layer surface seen from both sides.
            cloth: pipeline("scene3d cloth pipeline", None, false),
            // Hair: open sheets whose edges thin out strand by strand. The
            // multisampled coverage fades them without sorting.
            hair: pipeline("scene3d hair pipeline", None, coverage),
        }
    }
}

/// Records `draws` into a pass that has its targets.
fn record(
    pass: &mut wgpu::RenderPass<'_>,
    pipelines: &Pipelines,
    globals: &wgpu::BindGroup,
    draws: &[Draw],
) {
    pass.set_bind_group(0, globals, &[]);
    for draw in draws {
        pass.set_pipeline(match draw.style {
            Style::Opaque => &pipelines.opaque,
            Style::Cloth => &pipelines.cloth,
            Style::Hair => &pipelines.hair,
        });
        pass.set_bind_group(1, &draw.object, &[]);
        pass.set_bind_group(2, &draw.weave, &[]);
        pass.set_vertex_buffer(0, draw.vertices.slice(..));
        pass.set_index_buffer(draw.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(draw.range.clone(), 0, 0..1);
    }
}

/// The smallest box holding `points`.
fn bounds(points: impl Iterator<Item = Vec3>) -> Option<(Vec3, Vec3)> {
    points.fold(None, |b, p| match b {
        None => Some((p, p)),
        Some((lo, hi)) => Some((lo.min(p), hi.max(p))),
    })
}

pub struct SceneRenderer {
    target: RenderTarget,
    pipelines: Pipelines,
    /// For the studio render's linear, single-sampled images.
    studio_pipelines: Pipelines,
    shadow_kit: ShadowKit,
    studio_kit: StudioKit,
    shadows: ShadowMaps,
    key_light: LightSlot,
    globals: wgpu::Buffer,
    globals_layout: wgpu::BindGroupLayout,
    globals_bind_group: wgpu::BindGroup,
    object_layout: wgpu::BindGroupLayout,
    weave_layout: wgpu::BindGroupLayout,
    weave_sampler: wgpu::Sampler,
    /// Textures of every weave in use, by key; and white for the rest.
    weaves: std::collections::HashMap<u64, wgpu::BindGroup>,
    /// The height maps of scanned fabrics (by the same key), on the CPU:
    /// what the studio render displaces the garment by.
    heights: std::collections::HashMap<u64, std::sync::Arc<refine::HeightField>>,
    neutral_weave: wgpu::BindGroup,
    statics: Vec<StaticDraw>,
    /// Skin, eyes, hair: whatever makes up the avatar.
    avatar: Vec<StaticDraw>,
    /// Where the avatar and the garment are: what the shadows must cover.
    avatar_bounds: Option<(Vec3, Vec3)>,
    garment_bounds: Option<(Vec3, Vec3)>,
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
    /// The still being made, if any.
    studio: Option<StudioRender>,
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
        let uniform_entry = |visibility| wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let stages = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene3d globals layout"),
            entries: &[
                uniform_entry(stages),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let object_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene3d object layout"),
            entries: &[uniform_entry(stages)],
        });
        let weave_layout = weave::layout(device);
        let weave_sampler = weave::sampler(device);
        let neutral_weave = weave::neutral(gpu, &weave_layout, &weave_sampler);
        let shadow_kit = ShadowKit::new(device, &object_layout);
        let shadows = ShadowMaps::new(gpu, LIVE_SHADOW_SIZE, 1);
        let key_light = LightSlot::new(gpu, &shadow_kit);

        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene3d globals"),
            size: size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals_bind_group =
            globals_group(device, &globals_layout, &globals, &shadows, &shadow_kit.sampler);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene3d pipeline layout"),
            bind_group_layouts: &[
                Some(&globals_layout),
                Some(&object_layout),
                Some(&weave_layout),
            ],
            immediate_size: 0,
        });
        let pipelines = Pipelines::new(
            device,
            &pipeline_layout,
            &shader,
            RenderTarget::COLOR_FORMAT,
            RenderTarget::SAMPLE_COUNT,
        );
        let studio_pipelines = Pipelines::new(device, &pipeline_layout, &shader, studio::HDR, 1);

        let props_binding = ObjectBinding::new(
            gpu,
            &object_layout,
            &ObjectUniform::new(
                Mat4::IDENTITY,
                &SurfaceMaterial {
                    shading: Shading::Prop,
                    ..SurfaceMaterial::default()
                },
            ),
        );
        Self {
            target: RenderTarget::new(gpu, "scene3d target", size, true),
            pipelines,
            studio_pipelines,
            studio_kit: StudioKit::new(device),
            shadow_kit,
            shadows,
            key_light,
            globals,
            globals_layout,
            globals_bind_group,
            object_layout,
            weave_layout,
            weave_sampler,
            weaves: std::collections::HashMap::new(),
            heights: std::collections::HashMap::new(),
            neutral_weave,
            statics: Vec::new(),
            avatar: Vec::new(),
            avatar_bounds: None,
            garment_bounds: None,
            props_vertices: GrowableBuffer::new(gpu, "prop vertices", wgpu::BufferUsages::VERTEX),
            props_indices: GrowableBuffer::new(gpu, "prop indices", wgpu::BufferUsages::INDEX),
            props_index_count: 0,
            props_binding,
            garment_vertices: GrowableBuffer::new(
                gpu,
                "garment vertices",
                wgpu::BufferUsages::VERTEX,
            ),
            garment_indices: GrowableBuffer::new(gpu, "garment indices", wgpu::BufferUsages::INDEX),
            garment_generation: None,
            shell: Shell::default(),
            pieces: Vec::new(),
            scratch: Vec::new(),
            studio: None,
            clear_color: srgba(38, 41, 48, 1.0),
            overlay: SurfaceOverlay::Fabric,
        }
    }

    fn static_draw(
        &self,
        gpu: &GpuContext,
        mesh: &MeshData,
        material: SurfaceMaterial,
    ) -> StaticDraw {
        let buffer = |label, contents, usage| {
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage,
                })
        };
        StaticDraw {
            vertices: buffer(
                "static vertices",
                bytemuck::cast_slice(&mesh.vertices),
                wgpu::BufferUsages::VERTEX,
            ),
            indices: buffer(
                "static indices",
                bytemuck::cast_slice(&mesh.indices),
                wgpu::BufferUsages::INDEX,
            ),
            index_count: mesh.indices.len() as u32,
            binding: ObjectBinding::new(
                gpu,
                &self.object_layout,
                &ObjectUniform::new(Mat4::IDENTITY, &material),
            ),
            double_sided: matches!(material.shading, Shading::Hair { .. }),
        }
    }

    /// Whether the textures of the weave with this key are loaded.
    pub fn has_weave(&self, key: u64) -> bool {
        self.weaves.contains_key(&key)
    }

    /// Loads a weave's textures under `key` (see [`SurfaceMaterial::weave_key`]).
    /// Weaves no longer in `keep` are dropped first.
    pub fn add_weave(
        &mut self,
        gpu: &GpuContext,
        key: u64,
        texture: &cad_core::WeaveTexture,
        keep: &[u64],
    ) {
        self.weaves.retain(|k, _| keep.contains(k));
        self.heights.retain(|k, _| keep.contains(k));
        let group = weave::bind_group(gpu, &self.weave_layout, &self.weave_sampler, texture);
        self.weaves.insert(key, group);
        // Scans (no threads): their height map, for displacement.
        if texture.ends == 0 {
            self.heights.insert(
                key,
                std::sync::Arc::new(refine::HeightField::from_surface(
                    texture.width,
                    texture.height,
                    &texture.surface,
                    1024,
                )),
            );
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
        self.avatar_bounds = bounds(
            parts
                .iter()
                .flat_map(|(mesh, _)| mesh.vertices.iter().map(|v| Vec3::from(v.position))),
        );
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
            self.props_vertices
                .write(gpu, bytemuck::cast_slice(&mesh.vertices));
            self.props_indices
                .write(gpu, bytemuck::cast_slice(&mesh.indices));
        }
    }

    /// Uploads the garment, drawn with the thickness of its fabric
    /// (`thickness`, metres, per piece; see [`Shell`]). Vertex attributes
    /// are always rewritten (draping moves them every step); indices and
    /// per-piece draws only when the mesh was reassembled. `values` are the per-vertex measurements of the
    /// technical views (see [`SurfaceOverlay`]); missing ones read as zero.
    pub fn sync_garment(
        &mut self,
        gpu: &GpuContext,
        mesh: &GarmentMesh,
        values: &[f32],
        thickness: impl Fn(PieceId) -> f32,
    ) {
        if self.garment_generation != Some(mesh.generation) {
            self.garment_generation = Some(mesh.generation);
            self.shell = Shell::new(mesh);
            self.garment_indices
                .write(gpu, bytemuck::cast_slice(&self.shell.indices));
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
                    weave: 0,
                    uniform: placeholder,
                })
                .collect();
        }
        self.garment_bounds = bounds(mesh.positions.iter().copied());
        self.shell
            .vertices(mesh, values, &thickness, &mut self.scratch);
        self.garment_vertices
            .write(gpu, bytemuck::cast_slice(&self.scratch));
    }

    /// The sphere the shadow maps cover: the avatar and the garment.
    fn focus(&self) -> (Vec3, f32) {
        let boxes = [self.avatar_bounds, self.garment_bounds];
        let Some((lo, hi)) = bounds(
            boxes
                .iter()
                .flatten()
                .flat_map(|&(lo, hi)| [lo, hi]),
        ) else {
            return (Vec3::new(0.0, 0.9, 0.0), 1.1);
        };
        // A little more, for the shadow on the floor around the feet.
        ((lo + hi) * 0.5, ((hi - lo).length() * 0.5 + 0.15).max(0.3))
    }

    /// Uploads this frame's camera, lights and fabric materials.
    pub fn prepare(
        &mut self,
        gpu: &GpuContext,
        camera: &OrbitCamera,
        material_of: impl Fn(PieceId) -> SurfaceMaterial,
    ) {
        let eye = camera.eye();
        let direction = |v: Vec3| v.normalize().extend(0.0).to_array();
        let key = Vec3::new(0.45, 0.8, 0.55).normalize();
        let (center, radius) = self.focus();
        let key_matrix = light_matrix(key, center, radius);
        self.key_light.write(gpu, key_matrix);
        let globals = Globals {
            view_proj: camera.view_proj(self.target.aspect()).to_cols_array_2d(),
            eye: eye.extend(1.0).to_array(),
            key_direction: key.extend(0.0).to_array(),
            key_color: [3.0, 2.9, 2.75, 0.0],
            fill_direction: direction(Vec3::new(-0.6, 0.25, -0.5)),
            fill_color: [0.55, 0.62, 0.75, 0.0],
            sky: [0.42, 0.45, 0.52, 0.0],
            ground: [0.12, 0.11, 0.10, 0.0],
            shadow_key: key_matrix.to_cols_array_2d(),
            shadow_fill: Mat4::IDENTITY.to_cols_array_2d(),
            options: [1.0, 0.0, 0.0, 0.0],
            shadow: [2.0 * radius / LIVE_SHADOW_SIZE as f32, 1.6, 1.0, 0.0],
        };
        gpu.queue
            .write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));
        for draw in &mut self.pieces {
            let material = material_of(draw.piece);
            draw.weave = material.weave_key;
            let mut uniform = ObjectUniform::new(Mat4::IDENTITY, &material);
            uniform.overlay = self.overlay.uniform();
            // Cloth: sheen, and light through it.
            uniform.fabric[2] = 1.0;
            // Textures still loading: the plain colour meanwhile.
            if draw.weave != 0 && !self.weaves.contains_key(&draw.weave) {
                uniform.weave[2] = 0.0;
            }
            gpu.queue
                .write_buffer(&draw.binding.buffer, 0, bytemuck::bytes_of(&uniform));
            draw.uniform = uniform;
        }
    }

    /// What is drawn this frame, in order.
    fn draws(&self) -> Vec<Draw> {
        let mut draws = Vec::new();
        for draw in self.statics.iter().chain(&self.avatar) {
            draws.push(Draw {
                vertices: draw.vertices.clone(),
                indices: draw.indices.clone(),
                range: 0..draw.index_count,
                object: draw.binding.bind_group.clone(),
                weave: self.neutral_weave.clone(),
                style: if draw.double_sided {
                    Style::Hair
                } else {
                    Style::Opaque
                },
            });
        }
        if self.props_index_count > 0 {
            draws.push(Draw {
                vertices: self.props_vertices.buffer().clone(),
                indices: self.props_indices.buffer().clone(),
                range: 0..self.props_index_count,
                object: self.props_binding.bind_group.clone(),
                weave: self.neutral_weave.clone(),
                style: Style::Cloth,
            });
        }
        for draw in &self.pieces {
            draws.push(Draw {
                vertices: self.garment_vertices.buffer().clone(),
                indices: self.garment_indices.buffer().clone(),
                range: draw.indices.clone(),
                object: draw.binding.bind_group.clone(),
                weave: self
                    .weaves
                    .get(&draw.weave)
                    .unwrap_or(&self.neutral_weave)
                    .clone(),
                style: Style::Cloth,
            });
        }
        draws
    }

    /// Starts a studio render of the scene as it is now (after
    /// [`Self::prepare`]), `size` pixels, through `camera`. The scene is
    /// copied: the live view can change meanwhile.
    /// `aperture`: radius of the lens in metres (depth of field; 0: all sharp).
    pub fn start_studio(
        &mut self,
        gpu: &GpuContext,
        size: [u32; 2],
        camera: &OrbitCamera,
        aperture: f32,
    ) {
        let device = &gpu.device;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("studio snapshot"),
        });
        let copy = |encoder: &mut wgpu::CommandEncoder, source: &wgpu::Buffer, usage| {
            let copy = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("studio snapshot"),
                size: source.size(),
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(source, 0, &copy, 0, source.size());
            copy
        };
        // Buffers rewritten while the cloth moves are copied; the rest is
        // never written again and can be shared.
        let mut draws = self.draws();
        let garment_vertices = copy(
            &mut encoder,
            self.garment_vertices.buffer(),
            wgpu::BufferUsages::VERTEX,
        );
        let props_vertices = copy(
            &mut encoder,
            self.props_vertices.buffer(),
            wgpu::BufferUsages::VERTEX,
        );
        let props_indices = copy(
            &mut encoder,
            self.props_indices.buffer(),
            wgpu::BufferUsages::INDEX,
        );
        for draw in &mut draws {
            if draw.vertices == *self.garment_vertices.buffer() {
                draw.vertices = garment_vertices.clone();
            } else if draw.vertices == *self.props_vertices.buffer() {
                draw.vertices = props_vertices.clone();
                draw.indices = props_indices.clone();
            }
        }
        // The garment itself with many more polygons, and corduroy's ribs
        // as geometry (see `refine`); its materials as they are now.
        let piece_draws = draws.len() - self.pieces.len();
        let (refined_vertices, refined_indices, refined) = self.refined_garment();
        if !refined_indices.is_empty() {
            let buffer = |contents: &[u8], usage| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("studio garment"),
                    contents,
                    usage,
                })
            };
            let vertices = buffer(bytemuck::cast_slice(&refined_vertices), wgpu::BufferUsages::VERTEX);
            let indices = buffer(bytemuck::cast_slice(&refined_indices), wgpu::BufferUsages::INDEX);
            for ((draw, piece), (range, displaced)) in
                draws[piece_draws..].iter_mut().zip(&self.pieces).zip(refined)
            {
                let mut uniform = piece.uniform;
                if displaced {
                    uniform.fabric[2] = 2.0;
                }
                let buffer = buffer(bytemuck::bytes_of(&uniform), wgpu::BufferUsages::UNIFORM);
                draw.vertices = vertices.clone();
                draw.indices = indices.clone();
                draw.range = range;
                draw.object = object_group(device, &self.object_layout, &buffer);
            }
        }
        gpu.queue.submit([encoder.finish()]);
        self.studio = Some(StudioRender::new(
            gpu,
            &self.globals_layout,
            &self.shadow_kit,
            &self.studio_kit,
            size,
            *camera,
            aperture,
            self.focus(),
            draws,
        ));
    }

    /// The garment refined for the studio render: vertices, indices, and
    /// per piece its index range and whether its ribs are geometry.
    fn refined_garment(&self) -> (Vec<MeshVertex>, Vec<u32>, Vec<(Range<u32>, bool)>) {
        let vertices = &self.scratch;
        let indices = &self.shell.indices;
        let mut plans = Vec::with_capacity(self.pieces.len());
        let mut counts = Vec::with_capacity(self.pieces.len());
        for piece in &self.pieces {
            let triangles = &indices[piece.indices.start as usize..piece.indices.end as usize];
            // The size of its triangles in metres and in UV.
            let (mut metres, mut uv, mut outside) = (0.0f32, 0.0f32, 0usize);
            for t in triangles.chunks_exact(3) {
                let [a, b, c] = [t[0], t[1], t[2]].map(|i| &vertices[i as usize]);
                if a.wear[3].min(b.wear[3]).min(c.wear[3]) < 0.5 {
                    continue;
                }
                outside += 1;
                metres += Vec3::from(a.position).distance(Vec3::from(b.position));
                uv += glam::Vec2::from(a.uv).distance(glam::Vec2::from(b.uv));
            }
            let edge = metres / outside.max(1) as f32;
            let metres_per_uv = metres / uv.max(1e-9);
            let u = &piece.uniform;
            let corduroy = u.weave[2] < 0.5
                && (u.params[3] + 0.001).floor() == 3.0
                && u.params[2] > 0.0;
            let mut refinement = refine::Refinement { levels: 2, relief: None };
            if corduroy {
                let ribs = 18.0 * u.fabric[0];
                let depth = 0.0012 * u.fabric[1] * u.params[2];
                let rib_width = metres_per_uv / ribs;
                if depth > 0.0 {
                    refinement = refine::Refinement {
                        levels: refine::levels_for_ribs(edge, rib_width, 5.0),
                        relief: Some(refine::Relief::Ribs { count: ribs, depth }),
                    };
                }
            }
            // A scan with a height map: displaced by it, as finely as its
            // features (a few samples of the map per triangle edge).
            if u.weave[2] > 1.5
                && let Some(field) = self.heights.get(&piece.weave)
            {
                let repeat = [u.weave[0].max(1e-3), u.weave[1].max(1e-3)];
                let swatch = metres_per_uv / repeat[0];
                let feature = swatch / field.width as f32 * 4.0;
                let depth = MAP_RELIEF * u.weave[3];
                if depth > 0.0 {
                    refinement = refine::Refinement {
                        levels: refine::levels_for_ribs(edge, feature, 1.0),
                        relief: Some(refine::Relief::Map {
                            field: field.clone(),
                            repeat,
                            depth,
                            shuffled: u.yarn[1] > 0.5,
                        }),
                    };
                }
            }
            counts.push((outside, triangles.len() / 3 - outside));
            plans.push((piece.indices.clone(), refinement));
        }
        // Within budget: take a level off the most refined until it fits.
        loop {
            let total: usize = plans
                .iter()
                .zip(&counts)
                .map(|((_, r), (outside, rest))| {
                    outside * (1 << (2 * r.levels)) + rest * (1 << (2 * r.levels.min(2)))
                })
                .sum();
            let finest = plans.iter_mut().map(|(_, r)| r).max_by_key(|r| r.levels);
            match finest {
                Some(r) if total > refine::MAX_TRIANGLES && r.levels > 1 => r.levels -= 1,
                _ => break,
            }
        }
        // Ribs coarser than a few edges each would only make noise: those
        // stay with the shader's parallax.
        for ((_, r), piece) in plans.iter_mut().zip(&self.pieces) {
            let _ = piece;
            if r.relief.is_some() && r.levels < 3 {
                r.relief = None;
            }
        }
        let displaced: Vec<bool> = plans.iter().map(|(_, r)| r.relief.is_some()).collect();
        let (v, i, ranges) = refine::refine(vertices, indices, &plans);
        (v, i, ranges.into_iter().zip(displaced).collect())
    }

    /// Takes `samples` more samples of the studio render, if one is going.
    pub fn step_studio(&mut self, gpu: &GpuContext, samples: u32) {
        if let Some(studio) = &mut self.studio {
            studio.step(
                gpu,
                &self.shadow_kit,
                &self.studio_kit,
                &self.studio_pipelines,
                samples,
            );
        }
    }

    /// The studio render, if one was started.
    pub fn studio(&self) -> Option<&StudioRender> {
        self.studio.as_ref()
    }

    pub fn stop_studio(&mut self) {
        self.studio = None;
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
        let draws = self.draws();
        self.shadow_kit
            .record(encoder, &self.shadows, 0, &self.key_light, &draws);
        let [r, g, b, a] = self.clear_color.map(f64::from);
        let mut pass = self.target.begin_pass(encoder, wgpu::Color { r, g, b, a });
        record(&mut pass, &self.pipelines, &self.globals_bind_group, &draws);
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
    /// Per vertex of the simulated mesh: its neighbours within the piece,
    /// and how far it lies from the piece's edge along the cloth (metres).
    neighbours: Vec<Vec<u32>>,
    edge_distance: Vec<f32>,
    /// Scratch: how the cloth bends at each vertex (see `bending`).
    bend: Vec<f32>,
    /// Vertices sewn together (seams, not buttons): their shading is one
    /// smooth surface across the seam rather than two creased ones.
    sewn: Vec<(u32, u32)>,
    /// Scratch: normals smoothed across the seams.
    normals: Vec<Vec3>,
}

impl Shell {
    fn new(mesh: &GarmentMesh) -> Self {
        let mut shell = Shell {
            neighbours: vec![Vec::new(); mesh.positions.len()],
            edge_distance: vec![f32::INFINITY; mesh.positions.len()],
            sewn: mesh
                .stitches
                .iter()
                .filter(|s| s.seam != topology_sync::Stitch::FASTENING && s.a != s.b)
                .map(|s| (s.a, s.b))
                .collect(),
            ..Shell::default()
        };
        let mut is_sewn = vec![false; mesh.positions.len()];
        for &(a, b) in &shell.sewn {
            is_sewn[a as usize] = true;
            is_sewn[b as usize] = true;
        }
        let mut base = 0u32;
        for span in &mesh.pieces {
            let triangles: Vec<[u32; 3]> = mesh.indices
                [span.indices.start as usize..span.indices.end as usize]
                .chunks_exact(3)
                .map(|t| [t[0], t[1], t[2]])
                .collect();
            // Free edges: used by one triangle only.
            let mut uses: std::collections::HashMap<(u32, u32), (u32, u32, u32, u32)> =
                std::collections::HashMap::new();
            for t in &triangles {
                for k in 0..3 {
                    let (a, b, c) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
                    let entry = uses.entry((a.min(b), a.max(b))).or_insert((a, b, c, 0));
                    entry.3 += 1;
                }
            }
            let mut edges: Vec<[u32; 3]> = uses
                .values()
                .filter(|e| e.3 == 1)
                .map(|e| [e.0, e.1, e.2])
                .collect();
            edges.sort_unstable();
            // Edges sewn to another piece get no cut edge of their own:
            // the seam closes them, and drawn they outline every seam.
            let rims: Vec<[u32; 3]> = edges
                .iter()
                .copied()
                .filter(|&[a, b, _]| !(is_sewn[a as usize] && is_sewn[b as usize]))
                .collect();
            for &(a, b) in uses.keys() {
                shell.neighbours[a as usize].push(b);
                shell.neighbours[b as usize].push(a);
            }
            // Distance along the cloth to the nearest edge (Dijkstra from
            // every edge vertex).
            let mut queue = std::collections::BinaryHeap::new();
            for &[a, b, _] in &edges {
                for v in [a, b] {
                    shell.edge_distance[v as usize] = 0.0;
                    queue.push((std::cmp::Reverse(0u32), v));
                }
            }
            while let Some((std::cmp::Reverse(d), v)) = queue.pop() {
                let d = f32::from_bits(d);
                if d > shell.edge_distance[v as usize] {
                    continue;
                }
                for &w in &shell.neighbours[v as usize] {
                    let nd = d + mesh.positions[v as usize].distance(mesh.positions[w as usize]);
                    if nd < shell.edge_distance[w as usize] {
                        shell.edge_distance[w as usize] = nd;
                        // Non-negative floats order like their bits.
                        queue.push((std::cmp::Reverse(nd.to_bits()), w));
                    }
                }
            }

            let count = span.vertices.len() as u32;
            let local = |v: u32| v - span.vertices.start;
            let first = shell.indices.len() as u32;
            // Outer face as simulated, inner face turned round.
            for t in &triangles {
                shell.indices.extend(t.map(|v| base + local(v)));
            }
            for t in &triangles {
                shell
                    .indices
                    .extend([t[0], t[2], t[1]].map(|v| base + count + local(v)));
            }
            // Rim: a quad of its own four vertices per free edge.
            let rim = base + 2 * count;
            for (k, _) in rims.iter().enumerate() {
                let q = rim + 4 * k as u32;
                shell.indices.extend([q, q + 1, q + 2, q + 1, q + 3, q + 2]);
            }
            shell
                .pieces
                .push((span.piece, first..shell.indices.len() as u32));
            shell
                .layout
                .push((span.piece, span.vertices.clone(), rims, base));
            base = rim + 4 * shell.layout.last().map_or(0, |l| l.2.len() as u32);
        }
        shell
    }

    /// How the cloth bends at every vertex, 1/metres: positive where it
    /// bulges outwards (the ridge of a fold), negative in a crease;
    /// smoothed once over the neighbours.
    fn bending(&mut self, mesh: &GarmentMesh) {
        let raw: Vec<f32> = (0..mesh.positions.len())
            .map(|i| {
                let (p, n) = (mesh.positions[i], mesh.normals[i]);
                let around = &self.neighbours[i];
                if around.is_empty() {
                    return 0.0;
                }
                let sum: f32 = around
                    .iter()
                    .map(|&j| {
                        let d = mesh.positions[j as usize] - p;
                        -2.0 * n.dot(d) / d.length_squared().max(1e-10)
                    })
                    .sum();
                sum / around.len() as f32
            })
            .collect();
        self.bend = (0..raw.len())
            .map(|i| {
                let around = &self.neighbours[i];
                let sum: f32 = around.iter().map(|&j| raw[j as usize]).sum();
                (raw[i] + sum) / (1 + around.len()) as f32
            })
            .collect();
    }

    fn vertices(
        &mut self,
        mesh: &GarmentMesh,
        values: &[f32],
        thickness: &impl Fn(PieceId) -> f32,
        out: &mut Vec<MeshVertex>,
    ) {
        out.clear();
        self.bending(mesh);
        // One surface across every seam: sewn vertices share their normal.
        self.normals.clear();
        self.normals.extend_from_slice(&mesh.normals);
        for &(a, b) in &self.sewn {
            let (a, b) = (a as usize, b as usize);
            if a < self.normals.len() && b < self.normals.len() {
                self.normals[a] += mesh.normals[b];
                self.normals[b] += mesh.normals[a];
            }
        }
        for n in &mut self.normals {
            *n = n.normalize_or(Vec3::Y);
        }
        let (bend, edge_distance, normals) = (&self.bend, &self.edge_distance, &self.normals);
        let vertex = |i: usize, offset: f32, normal: Vec3| {
            let mut v = MeshVertex::new(
                mesh.positions[i] + normals[i] * offset,
                normal,
                mesh.uvs[i],
            )
            .with_value(values.get(i).copied().unwrap_or(0.0));
            let side = if offset > 0.0 { 1.0 } else { -1.0 };
            let b = bend.get(i).copied().unwrap_or(0.0) * side;
            let edge = edge_distance.get(i).copied().filter(|d| d.is_finite()).unwrap_or(1.0);
            v.wear = [edge, b.max(0.0), (-b).max(0.0), side];
            v
        };
        for (piece, range, edges, _) in &self.layout {
            let half = 0.5 * thickness(*piece);
            let range = range.start as usize..range.end as usize;
            out.extend(range.clone().map(|i| vertex(i, half, normals[i])));
            out.extend(range.clone().map(|i| vertex(i, -half, -normals[i])));
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
                let rim = |i: usize, offset: f32| {
                    let mut v = vertex(i, offset, out_of);
                    v.wear = [0.0, 0.0, 0.0, 0.0];
                    v
                };
                let quad = [rim(a, half), rim(b, half), rim(a, -half), rim(b, -half)];
                let p = |v: &MeshVertex| Vec3::from(v.position);
                let facing = (p(&quad[1]) - p(&quad[0]))
                    .cross(p(&quad[2]) - p(&quad[0]))
                    .dot(out_of);
                if facing >= 0.0 {
                    out.extend(quad);
                } else {
                    out.extend([quad[1], quad[0], quad[3], quad[2]]);
                }
            }
        }
    }
}

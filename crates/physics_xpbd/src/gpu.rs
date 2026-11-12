//! GPU backend: runs the sub-steps of `xpbd.wgsl` as compute dispatches.
//!
//! All sub-steps of a simulation step are recorded into one compute pass
//! and submitted once. Per-dispatch parameters cannot be rewritten between
//! dispatches of one submission, so they live in uniform buffers with one
//! slot per sub-step / per colour, selected with dynamic offsets.
//!
//! Positions are read back after every step because the renderer and the
//! 2D ↔ 3D mapping consume them on the CPU. Rendering straight from the
//! particle buffer would remove that round trip.

use bytemuck::{Pod, Zeroable};
use glam::Vec3A;
use physics_collision::{ContactPlane, PLANES_PER_VERTEX};
use topology_sync::DistanceField;
use wgpu::util::DeviceExt;

use crate::cpu::{Ellipsoid, State, SubstepParams};
use crate::model::{Element, Model};

/// Most ellipsoids the shader's fixed-size collider array holds.
pub const MAX_COLLIDERS: usize = 32;
/// Most sub-steps one `step` call can run.
pub const MAX_SUBSTEPS: usize = 64;
const WORKGROUP_SIZE: u32 = 64;
/// Uniform slots are spaced by the largest alignment any backend requires.
const SLOT: u64 = 256;

/// Mirrors `Frame` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    gravity: [f32; 3],
    h: f32,
    damping: f32,
    inv_h2: f32,
    seam_gap: f32,
    friction: f32,
    margin: f32,
    ground: f32,
    max_speed: f32,
    _pad0: f32,
    particle_count: u32,
    collider_count: u32,
    surface_collision: u32,
    _pad1: u32,
}

/// Mirrors `Collider` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ColliderUniform {
    center: [f32; 4],
    radii: [f32; 4],
    rotation: [f32; 4],
}

/// Mirrors `BodyField` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BodyFieldUniform {
    origin: [f32; 3],
    voxel: f32,
    dims: [u32; 3],
    enabled: u32,
}

/// The body's distance field as a 3D texture.
struct FieldTexture {
    /// Address of the field it was made from, to skip redundant uploads.
    source: usize,
    view: wgpu::TextureView,
    uniform: wgpu::Buffer,
}

/// Buffers and bind groups sized for one model.
struct Bound {
    positions: wgpu::Buffer,
    /// Per particle: sub-step start position, then velocity.
    motion: wgpu::Buffer,
    /// Per particle: `PLANES_PER_VERTEX` contact planes.
    planes: wgpu::Buffer,
    /// Sized for `motion`; position readbacks use the first half.
    readback: wgpu::Buffer,
    uniforms: wgpu::BindGroup,
    particles: wgpu::BindGroup,
    particle_count: u32,
    collider_count: u32,
    /// Element count of every colour.
    colour_sizes: Vec<u32>,
}

pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    integrate: wgpu::ComputePipeline,
    solve: wgpu::ComputePipeline,
    finalize: wgpu::ComputePipeline,
    uniform_layout: wgpu::BindGroupLayout,
    particle_layout: wgpu::BindGroupLayout,
    frames: wgpu::Buffer,
    field: FieldTexture,
    bound: Option<Bound>,
}

impl GpuBackend {
    /// `device` must have been created with compute support (at least
    /// `wgpu::Limits::downlevel_defaults()`).
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("xpbd shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("xpbd.wgsl").into()),
        });
        let uniform = |binding, dynamic, size: u64| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: dynamic,
                min_binding_size: wgpu::BufferSize::new(size),
            },
            count: None,
        };
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("xpbd uniforms"),
            entries: &[
                uniform(0, true, size_of::<FrameUniform>() as u64),
                uniform(1, false, (size_of::<ColliderUniform>() * MAX_COLLIDERS) as u64),
                uniform(2, true, 16),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        // Read with textureLoad, so no filtering support needed.
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                uniform(4, false, size_of::<BodyFieldUniform>() as u64),
            ],
        });
        let particle_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("xpbd particles"),
            entries: &[storage(0, false), storage(1, false), storage(2, true), storage(3, true)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("xpbd layout"),
            bind_group_layouts: &[Some(&uniform_layout), Some(&particle_layout)],
            immediate_size: 0,
        });
        let pipeline = |entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let frames = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xpbd frame slots"),
            size: SLOT * MAX_SUBSTEPS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let field = Self::upload_field(&device, &queue, None);
        Self {
            field,
            integrate: pipeline("integrate"),
            solve: pipeline("solve"),
            finalize: pipeline("finalize"),
            uniform_layout,
            particle_layout,
            frames,
            bound: None,
            device,
            queue,
        }
    }

    /// Creates the field texture; without a body, a disabled 1-voxel dummy.
    fn upload_field(device: &wgpu::Device, queue: &wgpu::Queue, body: Option<&DistanceField>) -> FieldTexture {
        let dummy = [0.0f32];
        let (dims, values, uniform) = match body {
            Some(field) => (
                field.dims,
                field.values.as_slice(),
                BodyFieldUniform { origin: field.origin.to_array(), voxel: field.voxel, dims: field.dims, enabled: 1 },
            ),
            None => ([1; 3], dummy.as_slice(), BodyFieldUniform { origin: [0.0; 3], voxel: 1.0, dims: [1; 3], enabled: 0 }),
        };
        let size = wgpu::Extent3d { width: dims[0], height: dims[1], depth_or_array_layers: dims[2] };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("xpbd body field"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(values),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * dims[0]), rows_per_image: Some(dims[1]) },
            size,
        );
        FieldTexture {
            source: body.map_or(0, |field| std::ptr::from_ref(field) as usize),
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            uniform: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("xpbd body field"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            }),
        }
    }

    /// Uploads a model, its colliders and the state to start from. The
    /// body field is only re-uploaded when it is a different one.
    pub fn bind(&mut self, model: &Model, state: &State, colliders: &[Ellipsoid], body: Option<&DistanceField>) {
        if self.field.source != body.map_or(0, |field| std::ptr::from_ref(field) as usize) {
            self.field = Self::upload_field(&self.device, &self.queue, body);
        }
        let device = &self.device;
        let particle_count = state.positions.len();
        if particle_count == 0 || model.elements.is_empty() {
            self.bound = None;
            return;
        }
        let particle_bytes = (particle_count * 16) as u64;
        let particle_buffer = |label, vectors: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: particle_bytes * vectors,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let init = |label, contents: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage })
        };

        let elements = init("xpbd elements", bytemuck::cast_slice::<Element, u8>(&model.elements), wgpu::BufferUsages::STORAGE);

        // One 256-byte slot per colour: [first element, element count].
        let mut ranges = vec![0u8; SLOT as usize * model.colours.len().max(1)];
        for (slot, range) in model.colours.iter().enumerate() {
            let at = slot * SLOT as usize;
            ranges[at..at + 8].copy_from_slice(bytemuck::bytes_of(&[range.start, range.end - range.start]));
        }
        let ranges = init("xpbd colour ranges", &ranges, wgpu::BufferUsages::UNIFORM);

        if colliders.len() > MAX_COLLIDERS {
            log::warn!("{} colliders given, the GPU solver uses the first {MAX_COLLIDERS}", colliders.len());
        }
        let mut collider_data = [ColliderUniform::zeroed(); MAX_COLLIDERS];
        for (slot, e) in collider_data.iter_mut().zip(colliders) {
            *slot = ColliderUniform {
                center: e.center.extend(0.0).to_array(),
                radii: e.radii.extend(0.0).to_array(),
                rotation: e.rotation.to_array(),
            };
        }
        let collider_buffer = init("xpbd colliders", bytemuck::cast_slice(&collider_data), wgpu::BufferUsages::UNIFORM);

        let positions = particle_buffer("xpbd positions", 1);
        let motion = particle_buffer("xpbd motion", 2);
        // Zero-initialised: no contacts until the first detection.
        let planes = particle_buffer("xpbd contact planes", PLANES_PER_VERTEX as u64);
        fn sized(buffer: &wgpu::Buffer, size: u64) -> wgpu::BindingResource<'_> {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer, offset: 0, size: wgpu::BufferSize::new(size) })
        }
        let uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("xpbd uniforms"),
            layout: &self.uniform_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sized(&self.frames, size_of::<FrameUniform>() as u64) },
                wgpu::BindGroupEntry { binding: 1, resource: collider_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sized(&ranges, 16) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.field.view) },
                wgpu::BindGroupEntry { binding: 4, resource: self.field.uniform.as_entire_binding() },
            ],
        });
        let particles = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("xpbd particles"),
            layout: &self.particle_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: positions.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: motion.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: elements.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: planes.as_entire_binding() },
            ],
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xpbd readback"),
            size: particle_bytes * 2,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        self.bound = Some(Bound {
            positions,
            motion,
            planes,
            readback,
            uniforms,
            particles,
            particle_count: particle_count as u32,
            collider_count: colliders.len().min(MAX_COLLIDERS) as u32,
            colour_sizes: model.colours.iter().map(|r| r.end - r.start).collect(),
        });
        self.upload_state(model, state);
    }

    /// Overwrites the particle buffers, e.g. after the CPU backend ran.
    pub fn upload_state(&mut self, model: &Model, state: &State) {
        let Some(bound) = &self.bound else { return };
        let positions: Vec<[f32; 4]> =
            state.positions.iter().zip(&model.inverse_mass).map(|(p, &w)| [p.x, p.y, p.z, w]).collect();
        let motion: Vec<[[f32; 4]; 2]> = state
            .previous
            .iter()
            .zip(&state.velocities)
            .map(|(p, v)| [[p.x, p.y, p.z, 0.0], [v.x, v.y, v.z, 0.0]])
            .collect();
        self.queue.write_buffer(&bound.positions, 0, bytemuck::cast_slice(&positions));
        self.queue.write_buffer(&bound.motion, 0, bytemuck::cast_slice(&motion));
    }

    /// Replaces the cloth-against-cloth contact planes; an empty slice
    /// removes them all.
    pub fn upload_planes(&mut self, planes: &[[ContactPlane; PLANES_PER_VERTEX]]) {
        let Some(bound) = &self.bound else { return };
        let mut data = vec![[0.0f32; 4]; bound.particle_count as usize * PLANES_PER_VERTEX];
        for (slot, plane) in data.iter_mut().zip(planes.iter().flatten()) {
            *slot = [plane.normal[0], plane.normal[1], plane.normal[2], plane.offset];
        }
        self.queue.write_buffer(&bound.planes, 0, bytemuck::cast_slice(&data));
    }

    /// Runs one sub-step per entry of `substeps` and reads the positions
    /// back into `positions`. Returns `false` if nothing is bound.
    pub fn step(&mut self, substeps: &[SubstepParams], positions: &mut [Vec3A]) -> bool {
        let Some(bound) = &self.bound else { return false };
        let substeps = &substeps[..substeps.len().min(MAX_SUBSTEPS)];

        let mut slots = vec![0u8; SLOT as usize * substeps.len()];
        for (slot, p) in substeps.iter().enumerate() {
            let frame = FrameUniform {
                gravity: p.gravity.to_array(),
                h: p.h,
                damping: p.damping,
                inv_h2: 1.0 / (p.h * p.h),
                seam_gap: p.seam_gap,
                friction: p.friction,
                margin: p.margin,
                ground: p.ground,
                max_speed: p.max_speed,
                _pad0: 0.0,
                particle_count: bound.particle_count,
                collider_count: bound.collider_count,
                surface_collision: u32::from(p.surface_collision),
                _pad1: 0,
            };
            let at = slot * SLOT as usize;
            slots[at..at + size_of::<FrameUniform>()].copy_from_slice(bytemuck::bytes_of(&frame));
        }
        self.queue.write_buffer(&self.frames, 0, &slots);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("xpbd step") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("xpbd"), timestamp_writes: None });
            pass.set_bind_group(1, &bound.particles, &[]);
            let particle_groups = bound.particle_count.div_ceil(WORKGROUP_SIZE);
            for substep in 0..substeps.len() as u32 {
                let frame_offset = substep * SLOT as u32;
                pass.set_pipeline(&self.integrate);
                pass.set_bind_group(0, &bound.uniforms, &[frame_offset, 0]);
                pass.dispatch_workgroups(particle_groups, 1, 1);

                pass.set_pipeline(&self.solve);
                for (colour, &size) in bound.colour_sizes.iter().enumerate() {
                    pass.set_bind_group(0, &bound.uniforms, &[frame_offset, colour as u32 * SLOT as u32]);
                    pass.dispatch_workgroups(size.div_ceil(WORKGROUP_SIZE), 1, 1);
                }

                pass.set_pipeline(&self.finalize);
                pass.set_bind_group(0, &bound.uniforms, &[frame_offset, 0]);
                pass.dispatch_workgroups(particle_groups, 1, 1);
            }
        }
        encoder.copy_buffer_to_buffer(&bound.positions, 0, &bound.readback, 0, bound.positions.size());
        self.queue.submit([encoder.finish()]);
        self.read(&bound.readback, bound.positions.size(), |i, v| positions[i] = v);
        true
    }

    /// Reads velocities and sub-step start positions back, so the CPU
    /// backend can take over mid-simulation.
    pub fn download_motion(&mut self, state: &mut State) {
        let Some(bound) = &self.bound else { return };
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&bound.motion, 0, &bound.readback, 0, bound.motion.size());
        self.queue.submit([encoder.finish()]);
        let State { previous, velocities, .. } = state;
        self.read(&bound.readback, bound.motion.size(), |i, v| {
            if i % 2 == 0 { previous[i / 2] = v } else { velocities[i / 2] = v }
        });
    }

    /// Blocks until the first `bytes` of `buffer` (vec4 entries) are
    /// mapped and hands the xyz of each entry to `store`.
    fn read(&self, buffer: &wgpu::Buffer, bytes: u64, mut store: impl FnMut(usize, Vec3A)) {
        let slice = buffer.slice(..bytes);
        slice.map_async(wgpu::MapMode::Read, |result| {
            if let Err(error) = result {
                log::error!("xpbd readback failed: {error}");
            }
        });
        if let Err(error) = self.device.poll(wgpu::PollType::wait_indefinitely()) {
            log::error!("device poll failed during xpbd readback: {error}");
            return;
        }
        match slice.get_mapped_range() {
            Ok(view) => {
                let values: &[[f32; 4]] = bytemuck::cast_slice(&view);
                for (i, v) in values.iter().enumerate() {
                    store(i, Vec3A::new(v[0], v[1], v[2]));
                }
            }
            Err(error) => log::error!("xpbd readback buffer is not mapped: {error}"),
        }
        buffer.unmap();
    }
}

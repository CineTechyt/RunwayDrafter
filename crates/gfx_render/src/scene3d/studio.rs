//! The studio render: one still of the scene, made the slow way.
//!
//! Where the live view lights the scene with one key light and a flat
//! ambient term, the studio render lights it the way a photo studio does:
//! a large soft key light and the whole dome of the room around it. Each
//! sample takes the key from a slightly different point of its softbox and
//! one direction of the dome, both with shadows, and moves the camera by a
//! fraction of a pixel. Averaged over a few hundred samples that gives
//! soft shadows, light that does not reach into creases and armpits, and
//! edges and threads without any aliasing.
//!
//! The scene is copied when the render starts, so the editor can go on
//! draping and animating meanwhile.

use std::f32::consts::{PI, TAU};

use glam::{Mat4, Vec2, Vec3};
use wgpu::util::DeviceExt;

use crate::camera::OrbitCamera;
use crate::context::GpuContext;
use crate::target::RenderTarget;

use super::shadow::{LightSlot, ShadowKit, ShadowMaps, light_matrix};
use super::{Draw, Globals, Pipelines};

/// The linear HDR format of a sample and of the running average.
pub(crate) const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const SHADOW_SIZE: u32 = 4096;
pub const SAMPLES: u32 = 256;

/// What a studio render needs besides the scene's own pipelines: averaging
/// and finishing.
pub(crate) struct StudioKit {
    layout: wgpu::BindGroupLayout,
    accumulate: wgpu::RenderPipeline,
    resolve: wgpu::RenderPipeline,
}

impl StudioKit {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("studio shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("studio.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("studio layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("studio pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |label, entry, format, blend| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        // average += (sample − average) / n, as a blend:
        // sample · (1/n) + average · (1 − 1/n).
        let running_average = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Constant,
            dst_factor: wgpu::BlendFactor::OneMinusConstant,
            operation: wgpu::BlendOperation::Add,
        };
        Self {
            accumulate: pipeline(
                "studio accumulate",
                "fs_accumulate",
                HDR,
                Some(wgpu::BlendState {
                    color: running_average,
                    alpha: running_average,
                }),
            ),
            resolve: pipeline(
                "studio resolve",
                "fs_resolve",
                RenderTarget::COLOR_FORMAT,
                None,
            ),
            layout,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ResolveUniform {
    params: [f32; 4],
    backdrop_top: [f32; 4],
    backdrop_bottom: [f32; 4],
}

/// A studio render in progress, or done.
pub struct StudioRender {
    size: [u32; 2],
    done: u32,
    camera: OrbitCamera,
    /// Radius of the lens, metres: depth of field, sharp at the camera's
    /// target. 0 for a pinhole (all sharp).
    aperture: f32,
    /// The figure: where the shadow maps look.
    focus: (Vec3, f32),
    draws: Vec<Draw>,
    frame: wgpu::TextureView,
    depth: wgpu::TextureView,
    accumulated: wgpu::TextureView,
    output: wgpu::Texture,
    output_view: wgpu::TextureView,
    /// The picture for the UI (see [`RenderTarget::view`]).
    display: wgpu::TextureView,
    shadows: ShadowMaps,
    lights: [LightSlot; 2],
    globals: wgpu::Buffer,
    globals_group: wgpu::BindGroup,
    accumulate_group: wgpu::BindGroup,
    resolve_group: wgpu::BindGroup,
    _resolve_uniform: wgpu::Buffer,
}

/// Linear values of the studio's light.
mod light {
    /// The softbox key: front right, above.
    pub const KEY: [f32; 3] = [2.4, 2.3, 2.15];
    /// The room around: brighter overhead than at the horizon.
    pub const ZENITH: [f32; 3] = [0.50, 0.52, 0.56];
    pub const HORIZON: [f32; 3] = [0.30, 0.30, 0.31];
    /// Light thrown back up by the floor.
    pub const BOUNCE: [f32; 3] = [0.05, 0.047, 0.044];
    /// A strip light behind the figure that draws its outline.
    pub const RIM: f32 = 3.2;
    pub const EXPOSURE: f32 = 0.95;
}

impl StudioRender {
    /// Prepares a render of `draws` (a copy of the scene) through `camera`.
    pub(crate) fn new(
        gpu: &GpuContext,
        globals_layout: &wgpu::BindGroupLayout,
        shadow_kit: &ShadowKit,
        kit: &StudioKit,
        size: [u32; 2],
        camera: OrbitCamera,
        aperture: f32,
        focus: (Vec3, f32),
        draws: Vec<Draw>,
    ) -> Self {
        let device = &gpu.device;
        let size = [size[0].max(1), size[1].max(1)];
        let texture = |label, format, usage, view_formats: &[wgpu::TextureFormat]| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats,
            })
        };
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let target = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let frame = view(&texture("studio sample", HDR, target, &[]));
        let accumulated = view(&texture("studio average", HDR, target, &[]));
        let depth = view(&texture(
            "studio depth",
            RenderTarget::DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            &[],
        ));
        let output = texture(
            "studio image",
            RenderTarget::COLOR_FORMAT,
            target | wgpu::TextureUsages::COPY_SRC,
            &[RenderTarget::DISPLAY_FORMAT],
        );
        let output_view = view(&output);
        let display = crate::target::display_view(&output);
        let shadows = ShadowMaps::new(gpu, SHADOW_SIZE, 2);
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("studio globals"),
            size: size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals_group = super::globals_group(
            device,
            globals_layout,
            &globals,
            &shadows,
            &shadow_kit.sampler,
        );
        let resolve_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("studio resolve"),
            contents: bytemuck::bytes_of(&ResolveUniform {
                params: [light::EXPOSURE, 0.35, 0.0, 0.0],
                backdrop_top: [0.16, 0.165, 0.175, 0.0],
                backdrop_bottom: [0.42, 0.42, 0.425, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let group = |source: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("studio"),
                layout: &kit.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: resolve_uniform.as_entire_binding(),
                    },
                ],
            })
        };
        Self {
            size,
            done: 0,
            camera,
            // As a lens stopped down for a close-up: the same look at any
            // distance, rather than a blur that swamps a detail shot.
            aperture: aperture.max(0.0) * (camera.distance / 3.2).min(1.0),
            focus,
            draws,
            accumulate_group: group(&frame),
            resolve_group: group(&accumulated),
            frame,
            depth,
            accumulated,
            output,
            output_view,
            display,
            shadows,
            lights: [
                LightSlot::new(gpu, shadow_kit),
                LightSlot::new(gpu, shadow_kit),
            ],
            globals,
            globals_group,
            _resolve_uniform: resolve_uniform,
        }
    }

    pub fn size(&self) -> [u32; 2] {
        self.size
    }

    /// Samples done and to do.
    pub fn progress(&self) -> (u32, u32) {
        (self.done, SAMPLES)
    }

    pub fn finished(&self) -> bool {
        self.done >= SAMPLES
    }

    /// The picture so far, for the UI to show (see [`RenderTarget::view`]).
    pub fn view(&self) -> &wgpu::TextureView {
        &self.display
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.output
    }

    /// The lights and camera of sample `index`.
    fn sample_globals(&self, index: u32) -> (Globals, [Mat4; 2]) {
        let camera = &self.camera;
        let eye = camera.eye();
        // The studio turns with the camera: the key always comes from the
        // front right, the rim light from behind.
        let toward_camera = {
            let f = eye - camera.target;
            Vec3::new(f.x, 0.0, f.z).normalize_or(Vec3::Z)
        };
        let right = Vec3::Y.cross(toward_camera).normalize();
        let frame =
            |x: f32, y: f32, z: f32| (right * x + Vec3::Y * y + toward_camera * z).normalize();

        // The key, from a random point of a large softbox.
        let key_center = frame(0.55, 0.75, 0.62);
        let (u, v) = (halton(index + 1, 5), halton(index + 1, 7));
        let (a, b) = orthonormal(key_center);
        let disk = concentric_disk(u, v) * 0.16;
        let key = (key_center + a * disk.x + b * disk.y).normalize();

        // One direction of the dome, all of them in turn (spread evenly over
        // the upper hemisphere; scrambled so any prefix covers it evenly).
        let j = (index * 97) % SAMPLES;
        let y = 1.0 - (j as f32 + 0.5) / SAMPLES as f32;
        let phi = j as f32 * PI * (3.0 - 5f32.sqrt());
        let r = (1.0 - y * y).max(0.0).sqrt();
        let dome = Vec3::new(r * phi.cos(), y, r * phi.sin());
        let rim_direction = frame(-0.45, 0.55, -0.7);
        let rim = light::RIM * smoothstep(0.90, 0.95, dome.dot(rim_direction));
        let sky = Vec3::from(light::HORIZON).lerp(Vec3::from(light::ZENITH), y);
        // Uniform over the hemisphere: each direction stands for 2π / n of
        // it, and the average over samples divides by n.
        let dome_color = (sky + Vec3::splat(rim)) * TAU;

        let (center, radius) = self.focus;
        let matrices = [
            light_matrix(key, center, radius),
            light_matrix(dome, center, radius),
        ];

        // Sub-pixel jitter: the average over samples is the pixel's area.
        let jitter = Vec2::new(halton(index + 1, 2), halton(index + 1, 3)) - 0.5;
        let shift = jitter * 2.0 / Vec2::new(self.size[0] as f32, self.size[1] as f32);
        let aspect = self.size[0] as f32 / self.size[1] as f32;
        // A lens rather than a pinhole: each sample looks through another
        // point of the aperture at the same point in focus, so whatever is
        // nearer or farther than the target blurs.
        let forward = (camera.target - eye).normalize_or(Vec3::NEG_Z);
        let lens_right = forward.cross(Vec3::Y).normalize_or(Vec3::X);
        let lens_up = lens_right.cross(forward);
        let lens = concentric_disk(halton(index + 1, 11), halton(index + 1, 13)) * self.aperture;
        let eye = eye + lens_right * lens.x + lens_up * lens.y;
        let view = glam::camera::rh::view::look_at_mat4(eye, camera.target, Vec3::Y);
        let view_proj =
            Mat4::from_translation(shift.extend(0.0)) * camera.projection(aspect) * view;
        let texel = 2.0 * radius / SHADOW_SIZE as f32;
        let globals = Globals {
            view_proj: view_proj.to_cols_array_2d(),
            eye: eye.extend(1.0).to_array(),
            key_direction: key.extend(0.0).to_array(),
            key_color: Vec3::from(light::KEY).extend(0.0).to_array(),
            fill_direction: dome.extend(0.0).to_array(),
            fill_color: dome_color.extend(0.0).to_array(),
            // The dome is the light from above; only the floor's bounce
            // stays a flat ambient term.
            sky: [0.0; 4],
            ground: Vec3::from(light::BOUNCE).extend(0.0).to_array(),
            shadow_key: matrices[0].to_cols_array_2d(),
            shadow_fill: matrices[1].to_cols_array_2d(),
            options: [2.0, 1.0, index as f32, 1.0],
            shadow: [texel, 1.5, 1.0, camera.distance + 1.5],
        };
        (globals, matrices)
    }

    /// Renders up to `count` more samples; then the picture so far.
    pub(crate) fn step(
        &mut self,
        gpu: &GpuContext,
        shadow_kit: &ShadowKit,
        kit: &StudioKit,
        pipelines: &Pipelines,
        count: u32,
    ) {
        let last = (self.done + count).min(SAMPLES);
        while self.done < last {
            let index = self.done;
            let (globals, matrices) = self.sample_globals(index);
            gpu.queue
                .write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));
            for (slot, matrix) in self.lights.iter().zip(matrices) {
                slot.write(gpu, matrix);
            }
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("studio sample"),
                });
            for layer in 0..2 {
                shadow_kit.record(
                    &mut encoder,
                    &self.shadows,
                    layer,
                    &self.lights[layer],
                    &self.draws,
                );
            }
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("studio sample"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.frame,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                super::record(&mut pass, pipelines, &self.globals_group, &self.draws);
            }
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("studio accumulate"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.accumulated,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: if index == 0 {
                                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                            } else {
                                wgpu::LoadOp::Load
                            },
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                let weight = 1.0 / f64::from(index + 1);
                pass.set_blend_constant(wgpu::Color {
                    r: weight,
                    g: weight,
                    b: weight,
                    a: weight,
                });
                pass.set_pipeline(&kit.accumulate);
                pass.set_bind_group(0, &self.accumulate_group, &[]);
                pass.draw(0..3, 0..1);
            }
            self.done += 1;
            if self.finished() || index + 1 == last {
                self.resolve(&mut encoder, kit);
            }
            gpu.queue.submit([encoder.finish()]);
        }
    }

    fn resolve(&self, encoder: &mut wgpu::CommandEncoder, kit: &StudioKit) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("studio resolve"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.output_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&kit.resolve);
        pass.set_bind_group(0, &self.resolve_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// The finished picture as tightly packed sRGB RGBA8 rows, top first.
    pub fn read_pixels(&self, gpu: &GpuContext) -> Vec<u8> {
        crate::target::read_texture(gpu, &self.output, self.size)
    }
}

/// The radical inverse of `index` in `base`: a sequence that fills 0..1
/// evenly however far it has got.
fn halton(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut r = 0.0;
    while index > 0 {
        f /= base as f32;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

/// A point of the unit disk, evenly spread, from a point of the unit square.
fn concentric_disk(u: f32, v: f32) -> Vec2 {
    let (a, b) = (2.0 * u - 1.0, 2.0 * v - 1.0);
    if a == 0.0 && b == 0.0 {
        return Vec2::ZERO;
    }
    let (r, theta) = if a.abs() > b.abs() {
        (a, PI / 4.0 * (b / a))
    } else {
        (b, PI / 2.0 - PI / 4.0 * (a / b))
    };
    Vec2::new(theta.cos(), theta.sin()) * r
}

fn orthonormal(n: Vec3) -> (Vec3, Vec3) {
    let helper = if n.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let a = n.cross(helper).normalize();
    (a, n.cross(a))
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halton_fills_the_unit_interval_evenly() {
        let mut values: Vec<f32> = (1..=64).map(|i| halton(i, 2)).collect();
        values.sort_by(f32::total_cmp);
        for (k, v) in values.iter().enumerate() {
            assert!((v - k as f32 / 64.0).abs() < 1.0 / 64.0 + 1e-6, "{k}: {v}");
        }
    }

    #[test]
    fn every_dome_direction_is_used_once() {
        let mut seen: Vec<u32> = (0..SAMPLES).map(|i| (i * 97) % SAMPLES).collect();
        seen.sort_unstable();
        assert!(seen.iter().enumerate().all(|(k, &j)| k as u32 == j));
    }
}

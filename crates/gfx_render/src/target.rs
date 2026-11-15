//! Offscreen render targets.

use crate::context::GpuContext;

/// A multisampled colour target with a single-sample resolve texture that can
/// be sampled (by `egui`, for display) plus an optional depth buffer.
#[derive(Debug)]
pub struct RenderTarget {
    label: &'static str,
    size: [u32; 2],
    with_depth: bool,
    msaa_view: wgpu::TextureView,
    resolve_texture: wgpu::Texture,
    resolve_view: wgpu::TextureView,
    depth_view: Option<wgpu::TextureView>,
}

impl RenderTarget {
    /// Linear colours are written, sRGB-encoded on store.
    pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    pub const SAMPLE_COUNT: u32 = 4;

    pub fn new(gpu: &GpuContext, label: &'static str, size: [u32; 2], with_depth: bool) -> Self {
        let size = [size[0].max(1), size[1].max(1)];
        let texture = |format, sample_count, usage| {
            gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let view =
            |texture: &wgpu::Texture| texture.create_view(&wgpu::TextureViewDescriptor::default());
        let attachment = wgpu::TextureUsages::RENDER_ATTACHMENT;
        // Sampled by the UI; COPY_SRC allows read-back for snapshots/export.
        let resolve_usage =
            attachment | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC;
        let resolve_texture = texture(Self::COLOR_FORMAT, 1, resolve_usage);
        Self {
            label,
            size,
            with_depth,
            msaa_view: view(&texture(Self::COLOR_FORMAT, Self::SAMPLE_COUNT, attachment)),
            resolve_view: view(&resolve_texture),
            resolve_texture,
            depth_view: with_depth
                .then(|| view(&texture(Self::DEPTH_FORMAT, Self::SAMPLE_COUNT, attachment))),
        }
    }

    pub fn size(&self) -> [u32; 2] {
        self.size
    }

    pub fn aspect(&self) -> f32 {
        self.size[0] as f32 / self.size[1] as f32
    }

    /// The resolved image, for sampling.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.resolve_view
    }

    /// Reads the resolved image back as tightly packed sRGB RGBA8 rows, top
    /// row first. Blocks until the GPU is done, so it is meant for
    /// snapshots and export on native targets, not for per-frame use.
    pub fn read_pixels(&self, gpu: &GpuContext) -> Vec<u8> {
        read_texture(gpu, &self.resolve_texture, self.size)
    }

    /// Recreates the textures if `size` differs. Returns `true` if it did.
    pub fn resize(&mut self, gpu: &GpuContext, size: [u32; 2]) -> bool {
        let size = [size[0].max(1), size[1].max(1)];
        if size == self.size {
            return false;
        }
        *self = Self::new(gpu, self.label, size, self.with_depth);
        true
    }

    /// Begins a pass that clears the target and resolves MSAA on completion.
    pub fn begin_pass<'e>(
        &self,
        encoder: &'e mut wgpu::CommandEncoder,
        clear: wgpu::Color,
    ) -> wgpu::RenderPass<'e> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(self.label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.msaa_view,
                depth_slice: None,
                resolve_target: Some(&self.resolve_view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Discard,
                },
            })],
            depth_stencil_attachment: self.depth_view.as_ref().map(|view| {
                wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }
            }),
            ..Default::default()
        })
    }
}

/// Reads an RGBA8 texture (with `COPY_SRC`) back as tightly packed rows,
/// top row first. Blocks until the GPU is done.
pub fn read_texture(gpu: &GpuContext, texture: &wgpu::Texture, [width, height]: [u32; 2]) -> Vec<u8> {
    let row = width * 4;
    let padded = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read-back"),
        size: u64::from(padded) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read-back"),
        });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);

    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |result| {
        result.expect("failed to map the read-back buffer")
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device lost during read-back");
    let data = slice
        .get_mapped_range()
        .expect("read-back buffer is not mapped");
    data.chunks_exact(padded as usize)
        .flat_map(|line| &line[..row as usize])
        .copied()
        .collect()
}

/// A GPU buffer that is reallocated (with headroom) when the data outgrows it.
#[derive(Debug)]
pub(crate) struct GrowableBuffer {
    label: &'static str,
    usage: wgpu::BufferUsages,
    buffer: wgpu::Buffer,
}

impl GrowableBuffer {
    pub fn new(gpu: &GpuContext, label: &'static str, usage: wgpu::BufferUsages) -> Self {
        // COPY_SRC: the studio render takes a copy of what is drawn.
        let usage = usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        Self {
            label,
            usage,
            buffer: Self::allocate(gpu, label, usage, 4096),
        }
    }

    fn allocate(
        gpu: &GpuContext,
        label: &str,
        usage: wgpu::BufferUsages,
        size: u64,
    ) -> wgpu::Buffer {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    pub fn write(&mut self, gpu: &GpuContext, data: &[u8]) {
        let needed = data.len() as u64;
        if needed > self.buffer.size() {
            self.buffer = Self::allocate(gpu, self.label, self.usage, needed.next_power_of_two());
        }
        if !data.is_empty() {
            gpu.queue.write_buffer(&self.buffer, 0, data);
        }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

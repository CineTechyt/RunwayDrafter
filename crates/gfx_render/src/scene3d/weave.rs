//! Woven fabric textures on the GPU: the colour and surface images of a
//! [`cad_core::WeaveTexture`], with every mip level, sampled trilinearly
//! and anisotropically so threads do not shimmer or alias at any distance.

use cad_core::WeaveTexture;

use crate::context::GpuContext;

/// The bind group layout for a fabric's textures: colour, surface, sampler.
pub fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let texture = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("weave textures layout"),
        entries: &[
            texture(0),
            texture(1),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

pub fn sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("weave sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        anisotropy_clamp: 16,
        ..Default::default()
    })
}

/// The bind group of a weave's textures.
pub fn bind_group(
    gpu: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    texture: &WeaveTexture,
) -> wgpu::BindGroup {
    let size = [texture.width, texture.height];
    let albedo = upload(gpu, "weave colour", size, &texture.albedo, true);
    let surface = upload(gpu, "weave surface", size, &texture.surface, false);
    group(gpu, layout, sampler, &albedo, &surface)
}

/// For whatever is not woven: white, flat.
pub fn neutral(
    gpu: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    let albedo = upload(gpu, "neutral colour", [1, 1], &[[255; 4]], true);
    let surface = upload(gpu, "neutral surface", [1, 1], &[[128, 128, 255, 0]], false);
    group(gpu, layout, sampler, &albedo, &surface)
}

fn group(
    gpu: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    albedo: &wgpu::Texture,
    surface: &wgpu::Texture,
) -> wgpu::BindGroup {
    let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("weave textures"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view(albedo)),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view(surface)),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// A texture with its whole mip chain. Colour (`srgb`) is averaged in
/// linear light; surface normals are averaged and kept short (see
/// `mip_chain`), so a distant weave turns rough instead of noisy or smooth.
fn upload(
    gpu: &GpuContext,
    label: &str,
    [width, height]: [u32; 2],
    pixels: &[[u8; 4]],
    srgb: bool,
) -> wgpu::Texture {
    let levels = mip_chain([width, height], pixels, srgb);
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: levels.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: if srgb {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        },
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, ([w, h], data)) in levels.iter().enumerate() {
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(*h),
            },
            wgpu::Extent3d {
                width: *w,
                height: *h,
                depth_or_array_layers: 1,
            },
        );
    }
    texture
}

/// Every level down to 1 × 1, by averaging 2 × 2 blocks (the last row or
/// column of an odd size is folded into its neighbour).
pub fn mip_chain(size: [u32; 2], pixels: &[[u8; 4]], srgb: bool) -> Vec<([u32; 2], Vec<[u8; 4]>)> {
    let to_linear = |c: u8| {
        let s = f32::from(c) / 255.0;
        if !srgb {
            s
        } else if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    let from_linear = |v: f32| {
        let v = v.clamp(0.0, 1.0);
        let s = if !srgb {
            v
        } else if v <= 0.003_130_8 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0 + 0.5) as u8
    };
    let mut levels = vec![(size, pixels.to_vec())];
    let mut current: Vec<[f32; 4]> = pixels
        .iter()
        .map(|p| {
            [
                to_linear(p[0]),
                to_linear(p[1]),
                to_linear(p[2]),
                f32::from(p[3]) / 255.0,
            ]
        })
        .collect();
    let [mut w, mut h] = size;
    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut next = vec![[0.0f32; 4]; (nw * nh) as usize];
        for y in 0..h {
            for x in 0..w {
                let (tx, ty) = ((x / 2).min(nw - 1), (y / 2).min(nh - 1));
                let target = &mut next[(ty * nw + tx) as usize];
                let source = current[(y * w + x) as usize];
                for k in 0..4 {
                    target[k] += source[k];
                }
            }
        }
        for (i, p) in next.iter_mut().enumerate() {
            let (x, y) = (i as u32 % nw, i as u32 / nw);
            // How many source pixels went into this one.
            let span = |n: u32, big: u32, at: u32| {
                if at == n - 1 { big - 2 * (n - 1) } else { 2 }.min(big)
            };
            let count = (span(nw, w, x) * span(nh, h, y)).max(1) as f32;
            for k in 0..4 {
                p[k] /= count;
            }
            // Normals are left as averaged, not renormalised: how much
            // shorter they get says how bumpy the surface underneath is,
            // which the shader turns into roughness (Toksvig).
        }
        levels.push((
            [nw, nh],
            next.iter()
                .map(|p| {
                    [
                        from_linear(p[0]),
                        from_linear(p[1]),
                        from_linear(p[2]),
                        (p[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                    ]
                })
                .collect(),
        ));
        current = next;
        (w, h) = (nw, nh);
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `MIP_DUMP=dir cargo test -p gfx_render mip_dump -- --ignored`
    #[test]
    #[ignore]
    fn mip_dump() {
        let dir = std::env::var("MIP_DUMP").unwrap_or_else(|_| ".".into());
        let weave = cad_core::Weave::library()
            .into_iter()
            .find(|(n, _)| *n == "Denim")
            .unwrap()
            .1;
        let t = weave.texture(8);
        for (level, ([w, h], data)) in mip_chain([t.width, t.height], &t.albedo, true)
            .iter()
            .enumerate()
            .take(6)
        {
            let (w, h) = (*w as usize, *h as usize);
            let mut bytes = format!("P6\n{} {}\n255\n", 2 * w, 2 * h).into_bytes();
            for y in 0..2 * h {
                for x in 0..2 * w {
                    let p = data[(y % h) * w + x % w];
                    bytes.extend([p[0], p[1], p[2]]);
                }
            }
            std::fs::write(format!("{dir}/mip{level}.ppm"), bytes).unwrap();
        }
    }

    #[test]
    fn a_checkerboard_mips_to_its_average_in_linear_light() {
        // Black and white: the average in linear light is 0.5, which is
        // sRGB 188, not 128.
        let pixels: Vec<[u8; 4]> = (0..16)
            .map(|i| {
                if (i % 4 + i / 4) % 2 == 0 {
                    [0, 0, 0, 255]
                } else {
                    [255, 255, 255, 255]
                }
            })
            .collect();
        let levels = mip_chain([4, 4], &pixels, true);
        assert_eq!(levels.len(), 3);
        let ([w, h], last) = &levels[2];
        assert_eq!((*w, *h), (1, 1));
        assert!((i32::from(last[0][0]) - 188).abs() <= 1, "{:?}", last[0]);
    }
}

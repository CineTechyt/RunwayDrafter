//! Fabrics from image maps: downloaded PBR texture sets (ambientCG, Poly
//! Haven, ShareTextures…) and U3M, the garment industry's material format
//! (a `.u3m` JSON beside its images, or a `.u3ma` zip of both). Maps may
//! be PNG, JPEG or EXR, and lie in subfolders (Poly Haven's Blender
//! downloads keep them in `textures/` beside the `.blend`).
//!
//! [`open`] finds the maps; [`load`] decodes them into the two images the
//! renderer takes for any fabric surface (see [`cad_core::WeaveTexture`]):
//! colour with roughness in alpha, and the normal with height in alpha.

use std::path::{Path, PathBuf};

use cad_core::{FabricMaps, WeaveTexture};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MapsError {
    #[error("cannot read {0}: {1}")]
    Io(PathBuf, std::io::Error),
    #[error("cannot decode {0}: {1}")]
    Image(PathBuf, image::ImageError),
    #[error("cannot decode {0}: {1}")]
    Exr(PathBuf, String),
    #[error("no colour map found in {0}")]
    NoColour(PathBuf),
    #[error("not a U3M material: {0}")]
    U3m(String),
    #[error("cannot unpack {0}: {1}")]
    Zip(PathBuf, String),
}

/// A repeat of a downloaded texture set shows this much cloth unless told
/// otherwise; the size can be set in the material panel.
pub const DEFAULT_SIZE_MM: f32 = 250.0;

/// The maps of the material at `path`: a `.u3m` or `.u3ma` file, a zip
/// or folder of maps, any one map of a set (its siblings are found by
/// name), or a `.blend` with its textures beside it.
pub fn open(path: &Path) -> Result<FabricMaps, MapsError> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if path.is_dir() {
        return match find_file(path, "u3m") {
            Some(u3m) => read_u3m(&u3m),
            None => guess(path, None),
        };
    }
    match extension.as_str() {
        "u3m" => read_u3m(path),
        // A Blender material: its maps lie beside it.
        "blend" => guess(path.parent().unwrap_or(Path::new(".")), None),
        "u3ma" | "zip" => {
            let dir = unpack(path)?;
            match find_file(&dir, "u3m") {
                Some(u3m) => read_u3m(&u3m),
                None => guess(&dir, None),
            }
        }
        _ => guess(path.parent().unwrap_or(Path::new(".")), Some(path)),
    }
}

/// What a map file is, by the words in its name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Colour,
    NormalGl,
    NormalDx,
    Roughness,
    Height,
    Other,
}

fn role(path: &Path) -> Role {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let words: Vec<&str> = stem
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    let has = |names: &[&str]| words.iter().any(|w| names.contains(w));
    if has(&["normaldx", "nrmdx", "dx"]) && has(&["normaldx", "normal", "nrm", "nor", "nrmdx", "dx"]) {
        if has(&["normaldx", "nrmdx"]) || has(&["normal", "nrm", "nor"]) {
            return Role::NormalDx;
        }
    }
    if has(&["normalgl", "normal", "nrm", "nor", "norm", "normals"]) {
        return Role::NormalGl;
    }
    if has(&["roughness", "rough", "rgh"]) {
        return Role::Roughness;
    }
    if has(&["height", "displacement", "disp", "bump", "depth"]) {
        return Role::Height;
    }
    if has(&["ao", "ambientocclusion", "occlusion", "metal", "metalness", "metallic", "opacity", "alpha", "specular", "spec", "gloss", "mask", "emission", "preview", "thumb", "sphere"]) {
        return Role::Other;
    }
    if has(&["basecolor", "albedo", "diffuse", "color", "colour", "col", "diff", "base"]) {
        return Role::Colour;
    }
    Role::Other
}

fn is_image(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref(),
        Some("png" | "jpg" | "jpeg" | "exr")
    )
}

/// The maps among the images in `dir`; `hint`, a file the user picked,
/// is the colour map unless its name says otherwise.
fn guess(dir: &Path, hint: Option<&Path>) -> Result<FabricMaps, MapsError> {
    // The folder and its subfolders, a few levels down.
    let mut files: Vec<PathBuf> = Vec::new();
    let mut folders = vec![(dir.to_owned(), 0)];
    while let Some((folder, depth)) = folders.pop() {
        let entries = std::fs::read_dir(&folder).map_err(|e| MapsError::Io(folder.clone(), e))?;
        for path in entries.filter_map(|e| e.ok().map(|e| e.path())) {
            if path.is_dir() {
                if depth < 3 {
                    folders.push((path, depth + 1));
                }
            } else if is_image(&path) {
                files.push(path);
            }
        }
    }
    // Plain images before EXR where both exist; then by name.
    files.sort_by_key(|p| (p.extension().is_some_and(|e| e.eq_ignore_ascii_case("exr")), p.clone()));
    let pick = |wanted: Role| files.iter().find(|p| role(p) == wanted).cloned();
    let color = match hint {
        Some(h) if is_image(h) && matches!(role(h), Role::Colour | Role::Other) => Some(h.to_owned()),
        _ => pick(Role::Colour),
    }
    .or_else(|| (files.len() == 1).then(|| files[0].clone()))
    .ok_or_else(|| MapsError::NoColour(dir.to_owned()))?;
    let (normal, normal_y_up) = match (pick(Role::NormalGl), pick(Role::NormalDx)) {
        (Some(gl), _) => (Some(gl), true),
        (None, Some(dx)) => (Some(dx), false),
        (None, None) => (None, true),
    };
    Ok(FabricMaps {
        color,
        normal,
        normal_y_up,
        roughness: pick(Role::Roughness),
        height: pick(Role::Height),
        size_mm: [DEFAULT_SIZE_MM; 2],
        source: Some(dir.to_owned()),
        break_up_repeats: true,
    })
}

/// The maps of the front of a U3M material.
fn read_u3m(path: &Path) -> Result<FabricMaps, MapsError> {
    let text = std::fs::read_to_string(path).map_err(|e| MapsError::Io(path.to_owned(), e))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| MapsError::U3m(e.to_string()))?;
    let front = json
        .pointer("/material/front")
        .filter(|v| v.is_object())
        .ok_or_else(|| MapsError::U3m("no material.front".into()))?;
    let dir = path.parent().unwrap_or(Path::new("."));
    let image = |map: &str| front.get(map)?.get("texture")?.get("image");
    let file = |map: &str| {
        let relative = image(map)?.get("path")?.as_str()?;
        let full = dir.join(relative.replace('\\', "/"));
        full.exists().then_some(full)
    };
    let color = file("basecolor").ok_or_else(|| MapsError::NoColour(path.to_owned()))?;
    // The physical size: the image's pixels at its dpi, else width and
    // height (taken as centimetres).
    let number = |v: Option<&serde_json::Value>| v.and_then(serde_json::Value::as_f64).unwrap_or(0.0) as f32;
    let basecolor = image("basecolor");
    let dpi = number(basecolor.and_then(|i| i.get("dpi")));
    let mut size_mm = [DEFAULT_SIZE_MM; 2];
    if dpi > 0.0
        && let Ok((w, h)) = image::image_dimensions(&color)
    {
        size_mm = [w as f32 / dpi * 25.4, h as f32 / dpi * 25.4];
    } else {
        let (w, h) = (
            number(basecolor.and_then(|i| i.get("width"))),
            number(basecolor.and_then(|i| i.get("height"))),
        );
        if w > 0.0 && h > 0.0 {
            size_mm = [w * 10.0, h * 10.0];
        }
    }
    Ok(FabricMaps {
        color,
        normal: file("normal"),
        normal_y_up: true,
        roughness: file("roughness"),
        height: file("displacement"),
        size_mm,
        source: Some(path.to_owned()),
        break_up_repeats: true,
    })
}

fn find_file(dir: &Path, extension: &str) -> Option<PathBuf> {
    let mut stack = vec![dir.to_owned()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).ok()?.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case(extension)) {
                return Some(p);
            }
        }
    }
    None
}

/// Unpacks a zip (`.u3ma`) into a cache folder of its own; returns it.
fn unpack(path: &Path) -> Result<PathBuf, MapsError> {
    let zip_error = |m: &str| MapsError::Zip(path.to_owned(), m.to_owned());
    let bytes = std::fs::read(path).map_err(|e| MapsError::Io(path.to_owned(), e))?;
    let cache = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir)
        .join("clo360")
        .join("u3m");
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in path.to_string_lossy().as_bytes().iter().chain(&(bytes.len() as u64).to_le_bytes()) {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    let dir = cache.join(format!("{h:016x}"));
    let u16_at = |o: usize| bytes.get(o..o + 2).map(|b| u16::from_le_bytes([b[0], b[1]]) as usize);
    let u32_at = |o: usize| bytes.get(o..o + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize);
    // End of central directory: the last "PK\5\6".
    let end = (0..bytes.len().saturating_sub(21))
        .rev()
        .find(|&o| u32_at(o) == Some(0x0605_4b50))
        .ok_or_else(|| zip_error("no zip directory"))?;
    let count = u16_at(end + 10).ok_or_else(|| zip_error("truncated"))?;
    let mut at = u32_at(end + 16).ok_or_else(|| zip_error("truncated"))?;
    for _ in 0..count {
        if u32_at(at) != Some(0x0201_4b50) {
            return Err(zip_error("bad directory entry"));
        }
        let method = u16_at(at + 10).unwrap_or(0);
        let packed = u32_at(at + 20).unwrap_or(0);
        let (name_len, extra_len, comment_len) = (
            u16_at(at + 28).unwrap_or(0),
            u16_at(at + 30).unwrap_or(0),
            u16_at(at + 32).unwrap_or(0),
        );
        let local = u32_at(at + 42).unwrap_or(0);
        let name = String::from_utf8_lossy(bytes.get(at + 46..at + 46 + name_len).unwrap_or(&[])).into_owned();
        at += 46 + name_len + extra_len + comment_len;
        // Only plain relative names: nothing may land outside the folder.
        let relative = Path::new(&name);
        if name.ends_with('/')
            || relative.is_absolute()
            || relative.components().any(|c| !matches!(c, std::path::Component::Normal(_)))
        {
            continue;
        }
        let data_at = local + 30 + u16_at(local + 26).unwrap_or(0) + u16_at(local + 28).unwrap_or(0);
        let data = bytes.get(data_at..data_at + packed).ok_or_else(|| zip_error("truncated entry"))?;
        let content = match method {
            0 => data.to_vec(),
            8 => miniz_oxide::inflate::decompress_to_vec(data).map_err(|_| zip_error("bad deflate data"))?,
            _ => return Err(zip_error("unsupported compression")),
        };
        let target = dir.join(relative);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| MapsError::Io(parent.to_owned(), e))?;
        }
        std::fs::write(&target, content).map_err(|e| MapsError::Io(target.clone(), e))?;
    }
    Ok(dir)
}

/// An EXR map, whatever its channels: colour (R, G, B) or one grey
/// channel (Y, or any single one, as roughness maps often are).
fn decode_exr(path: &Path) -> Result<image::DynamicImage, MapsError> {
    let error = |message: String| MapsError::Exr(path.to_owned(), message);
    let file = exr::prelude::read_first_flat_layer_from_file(path).map_err(|e| error(e.to_string()))?;
    let layer = &file.layer_data;
    let (width, height) = (layer.size.0, layer.size.1);
    let channels = &layer.channel_data.list;
    let named = |names: &[&str]| {
        channels
            .iter()
            .find(|c| names.iter().any(|n| c.name.to_string().eq_ignore_ascii_case(n)))
    };
    let grey = named(&["Y", "L", "V"]).or_else(|| (channels.len() == 1).then(|| &channels[0]));
    let (r, g, b) = match (named(&["R"]), named(&["G"]), named(&["B"])) {
        (Some(r), Some(g), Some(b)) => (r, g, b),
        _ => {
            let y = grey.ok_or_else(|| error("no colour or grey channel".into()))?;
            (y, y, y)
        }
    };
    let pixels: Vec<f32> = (0..width * height)
        .flat_map(|i| {
            [r, g, b].map(|c| c.sample_data.value_by_flat_index(i).to_f32())
        })
        .collect();
    image::Rgb32FImage::from_raw(width as u32, height as u32, pixels)
        .map(image::DynamicImage::ImageRgb32F)
        .ok_or_else(|| error("unexpected size".into()))
}

/// A normal map as 8-bit colour. Float maps (EXR) come either encoded
/// like an image (0..1, flat = 0.5, 0.5, 1) or as the raw vectors (−1..1,
/// flat = 0, 0, 1): told apart by their average.
fn normal_rgb8(image: image::DynamicImage) -> image::RgbImage {
    let float = matches!(
        image,
        image::DynamicImage::ImageRgb32F(_) | image::DynamicImage::ImageRgba32F(_)
    );
    if !float {
        return image.to_rgb8();
    }
    let rgb = image.to_rgb32f();
    let count = (rgb.width() * rgb.height()).max(1) as f32;
    let mean = rgb.pixels().fold([0.0f32; 3], |m, p| [m[0] + p[0], m[1] + p[1], m[2] + p[2]]).map(|v| v / count);
    let raw = mean[0].abs() < 0.25 && mean[1].abs() < 0.25;
    let to_byte = |v: f32| ((if raw { v * 0.5 + 0.5 } else { v }).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    image::RgbImage::from_fn(rgb.width(), rgb.height(), |x, y| {
        let p = rgb.get_pixel(x, y);
        image::Rgb([to_byte(p[0]), to_byte(p[1]), to_byte(p[2])])
    })
}

/// Decodes the maps into the renderer's two images, at most `max_side`
/// pixels across: colour (sRGB) with roughness in alpha (255 when there is
/// no roughness map), and the tangent-space normal (green towards +V, down
/// the image) with height in alpha.
pub fn load(maps: &FabricMaps, max_side: u32) -> Result<WeaveTexture, MapsError> {
    let decode = |path: &Path| {
        if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("exr")) {
            decode_exr(path)
        } else {
            image::open(path).map_err(|e| MapsError::Image(path.to_owned(), e))
        }
    };
    let color = decode(&maps.color)?;
    let (mut width, mut height) = (color.width().max(1), color.height().max(1));
    let shrink = (max_side as f32 / width.max(height) as f32).min(1.0);
    width = ((width as f32 * shrink) as u32).max(1);
    height = ((height as f32 * shrink) as u32).max(1);
    let fit = |img: image::DynamicImage| {
        if img.width() == width && img.height() == height {
            img
        } else {
            img.resize_exact(width, height, image::imageops::FilterType::Triangle)
        }
    };
    let color = fit(color).to_rgba8();
    let roughness = maps.roughness.as_deref().map(decode).transpose()?.map(|i| fit(i).to_luma8());
    let normal = maps.normal.as_deref().map(decode).transpose()?.map(|i| normal_rgb8(fit(i)));
    let bump = maps.height.as_deref().map(decode).transpose()?.map(|i| fit(i).to_luma8());
    let count = (width * height) as usize;
    let mut albedo = Vec::with_capacity(count);
    let mut surface = Vec::with_capacity(count);
    for y in 0..height {
        for x in 0..width {
            let c = color.get_pixel(x, y).0;
            let r = roughness.as_ref().map_or(255, |m| m.get_pixel(x, y).0[0]);
            albedo.push([c[0], c[1], c[2], r]);
            let n = normal.as_ref().map_or([128, 128, 255], |m| m.get_pixel(x, y).0);
            // Green up in the image is −V here.
            let g = if maps.normal_y_up { 255 - n[1] } else { n[1] };
            let h = bump.as_ref().map_or(255, |m| m.get_pixel(x, y).0[0]);
            surface.push([n[0], g, n[2], h]);
        }
    }
    Ok(WeaveTexture {
        width,
        height,
        ends: 0,
        picks: 0,
        albedo,
        surface,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_files_are_told_apart_by_name() {
        let r = |n: &str| role(Path::new(n));
        assert_eq!(r("Fabric026_2K-JPG_Color.jpg"), Role::Colour);
        assert_eq!(r("Fabric026_2K-JPG_NormalGL.jpg"), Role::NormalGl);
        assert_eq!(r("Fabric026_2K-JPG_NormalDX.jpg"), Role::NormalDx);
        assert_eq!(r("Fabric026_2K-JPG_Roughness.jpg"), Role::Roughness);
        assert_eq!(r("Fabric026_2K-JPG_Displacement.jpg"), Role::Height);
        assert_eq!(r("Fabric026_2K-JPG_AmbientOcclusion.jpg"), Role::Other);
        assert_eq!(r("denim_diff_4k.jpg"), Role::Colour);
        assert_eq!(r("denim_nor_gl_4k.png"), Role::NormalGl);
        assert_eq!(r("denim_rough_4k.png"), Role::Roughness);
        // Poly Haven's Blender download.
        assert_eq!(r("ribbed_corduroy_diff_4k.jpg"), Role::Colour);
        assert_eq!(r("ribbed_corduroy_nor_gl_4k.exr"), Role::NormalGl);
        assert_eq!(r("ribbed_corduroy_rough_4k.exr"), Role::Roughness);
        assert_eq!(r("ribbed_corduroy_disp_4k.png"), Role::Height);
        for other in ["anisotropy_rotation", "anisotropy_strength", "spec_ior", "metal"] {
            assert_eq!(r(&format!("ribbed_corduroy_{other}_4k.png")), Role::Other, "{other}");
        }
    }

    #[test]
    fn a_folder_of_maps_and_a_u3m_zip_are_read() {
        let dir = std::env::temp_dir().join(format!("clo360_maps_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let png = |name: &str, value: u8| {
            image::RgbaImage::from_pixel(8, 4, image::Rgba([value, value, value, 255]))
                .save(dir.join(name))
                .unwrap();
        };
        png("Denim_Color.png", 60);
        png("Denim_NormalGL.png", 128);
        png("Denim_Roughness.png", 200);
        let maps = open(&dir.join("Denim_Roughness.png")).unwrap();
        assert!(maps.color.ends_with("Denim_Color.png"));
        assert!(maps.normal.is_some() && maps.normal_y_up);
        let t = load(&maps, 4).unwrap();
        assert_eq!((t.width, t.height), (4, 2));
        assert_eq!(t.albedo[0][3], 200);

        // The same as a U3M zip, 8 px at 25.4 dpi: 8 mm wide.
        let u3m = r#"{"schema":"1.0","material":{"front":{
            "basecolor":{"texture":{"image":{"path":"textures/Denim_Color.png","dpi":25.4,"width":0,"height":0}}},
            "normal":{"texture":{"image":{"path":"textures/Denim_NormalGL.png","dpi":25.4}}}}}}"#;
        let mut zip = Vec::new();
        let mut directory = Vec::new();
        let mut entries = 0u16;
        for (name, data) in [
            ("material.u3m".to_owned(), u3m.as_bytes().to_vec()),
            ("textures/Denim_Color.png".into(), std::fs::read(dir.join("Denim_Color.png")).unwrap()),
            ("textures/Denim_NormalGL.png".into(), std::fs::read(dir.join("Denim_NormalGL.png")).unwrap()),
        ] {
            let offset = zip.len() as u32;
            let header = |sig: u32| {
                let mut h = sig.to_le_bytes().to_vec();
                h.extend([0u8; 4]); // version, flags
                h.extend(0u16.to_le_bytes()); // stored
                h.extend([0u8; 8]); // time, date, crc
                h.extend((data.len() as u32).to_le_bytes());
                h.extend((data.len() as u32).to_le_bytes());
                h.extend((name.len() as u16).to_le_bytes());
                h.extend(0u16.to_le_bytes());
                h
            };
            zip.extend(header(0x0403_4b50));
            zip.extend(name.as_bytes());
            zip.extend(&data);
            let mut central = 0x0201_4b50u32.to_le_bytes().to_vec();
            central.extend([0u8; 6]);
            central.extend(0u16.to_le_bytes());
            central.extend([0u8; 8]);
            central.extend((data.len() as u32).to_le_bytes());
            central.extend((data.len() as u32).to_le_bytes());
            central.extend((name.len() as u16).to_le_bytes());
            central.extend([0u8; 12]);
            central.extend(offset.to_le_bytes());
            central.extend(name.as_bytes());
            directory.extend(central);
            entries += 1;
        }
        let directory_at = zip.len() as u32;
        zip.extend(&directory);
        zip.extend(0x0605_4b50u32.to_le_bytes());
        zip.extend([0u8; 4]);
        zip.extend(entries.to_le_bytes());
        zip.extend(entries.to_le_bytes());
        zip.extend((directory.len() as u32).to_le_bytes());
        zip.extend(directory_at.to_le_bytes());
        zip.extend([0u8; 2]);
        let archive = dir.join("denim.u3ma");
        std::fs::write(&archive, zip).unwrap();
        let maps = open(&archive).unwrap();
        assert!(maps.color.ends_with("textures/Denim_Color.png"));
        assert!((maps.size_mm[0] - 8.0).abs() < 0.01, "{:?}", maps.size_mm);
        std::fs::remove_dir_all(&dir).ok();
    }
}

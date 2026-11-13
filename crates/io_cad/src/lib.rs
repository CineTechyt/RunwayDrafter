//! File exchange with other tools.
//!
//! **In** — avatar geometry as a plain triangle mesh ([`ImportedMesh`]):
//! Wavefront OBJ ([`obj`]), glTF 2.0 / GLB ([`gltf`]) and FBX, binary or
//! ASCII ([`fbx`]). [`import_avatar`] picks the reader from the file name
//! and brings the result into the engine's conventions (metres, Y up,
//! standing on the floor).
//!
//! **Out** — the 2D pattern for manufacturing as DXF-AAMA or DXF-ASTM
//! ([`dxf`]), and the draped garment with its UV maps as OBJ or FBX.
//!
//! All readers and writers work on memory (`&[u8]` in, `String` out), so
//! they can be used on the web as well; only [`import_avatar`] and
//! external glTF buffers touch the file system.

pub mod drawn_text;
pub mod dxf;
pub mod fbx;
pub mod gltf;
mod mesh;
pub mod obj;
pub mod pattern_import;
pub mod pdf;
pub mod svg;

use std::path::Path;

pub use dxf::{DxfOptions, DxfStandard, DxfWriter, export_pattern_dxf};
pub use fbx::{FbxEncoding, export_garment_fbx};
pub use mesh::{ImportOptions, ImportedMesh, UpAxis};
pub use obj::{GarmentExport, UvLayout, export_garment_obj};
pub use pattern_import::{ImportedLoop, ImportedPattern, ImportedPieceCandidate, LineRole};
pub use pdf::{PT_TO_MM, PdfOptions, import_pattern_pdf};
pub use svg::{CSS_PX_TO_MM, SvgOptions, import_pattern_svg};

#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("unsupported file type: {0}")]
    UnsupportedFormat(String),
    #[error("line {line}: {message}")]
    Syntax { line: usize, message: String },
    #[error("malformed file: {0}")]
    Malformed(String),
    #[error("the file contains no triangles")]
    Empty,
}

impl IoError {
    pub(crate) fn malformed(message: impl Into<String>) -> Self {
        Self::Malformed(message.into())
    }
}

/// Reads an avatar from an `.obj`, `.gltf`, `.glb` or `.fbx` file and
/// normalises it with `options`.
pub fn import_avatar(path: &Path, options: &ImportOptions) -> Result<ImportedMesh, IoError> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let bytes = std::fs::read(path)?;
    let mut mesh = match extension.as_str() {
        "obj" => obj::import_obj(&bytes)?,
        "gltf" | "glb" => gltf::import_gltf(&bytes, path.parent())?,
        "fbx" => fbx::import_fbx(&bytes)?,
        other => return Err(IoError::UnsupportedFormat(other.to_owned())),
    };
    if mesh.triangles.is_empty() {
        return Err(IoError::Empty);
    }
    mesh.name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("avatar")
        .to_owned();
    mesh.normalise(options);
    Ok(mesh)
}

/// Reads a 2D pattern from a `.pdf` or `.svg` file.
pub fn import_pattern(path: &Path) -> Result<ImportedPattern, IoError> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let bytes = std::fs::read(path)?;
    match extension.as_str() {
        "pdf" => import_pattern_pdf(&bytes, &PdfOptions::default()),
        "svg" => import_pattern_svg(&bytes, &SvgOptions::default()),
        other => Err(IoError::UnsupportedFormat(other.to_owned())),
    }
}

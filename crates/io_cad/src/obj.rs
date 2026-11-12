//! Wavefront OBJ: avatar import and garment export.

use std::fmt::Write as _;

use cad_core::PatternDocument;
use glam::{Vec2, Vec3};
use topology_sync::GarmentMesh;

use crate::IoError;
use crate::mesh::ImportedMesh;

/// Reads the geometry of an OBJ file: `v`, `vn` and `f` (polygons are
/// fanned into triangles; negative, i.e. relative, indices are resolved).
/// Texture coordinates, groups and materials are ignored. A position used
/// with several normals keeps the first; avatars are smooth-shaded.
pub fn import_obj(bytes: &[u8]) -> Result<ImportedMesh, IoError> {
    let text = String::from_utf8_lossy(bytes);
    let mut mesh = ImportedMesh::default();
    let mut file_normals: Vec<Vec3> = Vec::new();
    let mut has_normal: Vec<bool> = Vec::new();
    let mut corners: Vec<u32> = Vec::new();

    for (number, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or_default();
        let mut words = line.split_whitespace();
        let Some(keyword) = words.next() else { continue };
        let syntax = |message: &str| IoError::Syntax { line: number + 1, message: message.to_owned() };
        let vector = |words: &mut std::str::SplitWhitespace<'_>| -> Result<Vec3, IoError> {
            let mut next = || words.next().and_then(|w| w.parse::<f32>().ok()).ok_or_else(|| syntax("expected three numbers"));
            Ok(Vec3::new(next()?, next()?, next()?))
        };
        match keyword {
            "v" => mesh.positions.push(vector(&mut words)?),
            "vn" => file_normals.push(vector(&mut words)?),
            "f" => {
                corners.clear();
                for word in words {
                    let mut parts = word.split('/');
                    let resolve = |part: Option<&str>, count: usize| -> Result<Option<u32>, IoError> {
                        let Some(part) = part.filter(|p| !p.is_empty()) else { return Ok(None) };
                        let index: i64 = part.parse().map_err(|_| syntax("bad face index"))?;
                        let resolved = if index < 0 { count as i64 + index } else { index - 1 };
                        if resolved < 0 || resolved >= count as i64 {
                            return Err(syntax("face index out of range"));
                        }
                        Ok(Some(resolved as u32))
                    };
                    let position = resolve(parts.next(), mesh.positions.len())?.ok_or_else(|| syntax("face without vertex"))?;
                    let _uv = parts.next();
                    if let Some(normal) = resolve(parts.next(), file_normals.len())? {
                        if mesh.normals.len() < mesh.positions.len() {
                            mesh.normals.resize(mesh.positions.len(), Vec3::ZERO);
                            has_normal.resize(mesh.positions.len(), false);
                        }
                        if !has_normal[position as usize] {
                            has_normal[position as usize] = true;
                            mesh.normals[position as usize] = file_normals[normal as usize];
                        }
                    }
                    corners.push(position);
                }
                if corners.len() < 3 {
                    return Err(syntax("a face needs at least three vertices"));
                }
                for k in 1..corners.len() - 1 {
                    mesh.triangles.push([corners[0], corners[k], corners[k + 1]]);
                }
            }
            _ => {}
        }
    }
    mesh.normals.resize(mesh.positions.len(), Vec3::ZERO);
    Ok(mesh.finish())
}

/// Which texture coordinates an exported garment carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UvLayout {
    /// The fabric mapping of each piece (texture tiles along the grain).
    /// Pieces overlap in UV space, as they should for a repeating fabric.
    #[default]
    Fabric,
    /// The 2D pattern layout scaled into the unit square: every piece has
    /// its own place, for baking or painting a texture atlas.
    Pattern,
}

/// A garment prepared for export: what OBJ and FBX share.
pub struct GarmentExport<'a> {
    pub mesh: &'a GarmentMesh,
    pub document: &'a PatternDocument,
    pub uv_layout: UvLayout,
}

impl GarmentExport<'_> {
    /// The texture coordinate of every vertex in the chosen layout.
    pub fn uvs(&self) -> Vec<Vec2> {
        match self.uv_layout {
            UvLayout::Fabric => self.mesh.uvs.clone(),
            UvLayout::Pattern => {
                let (lo, hi) = self
                    .mesh
                    .pattern_positions
                    .iter()
                    .fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(lo, hi), &p| (lo.min(p), hi.max(p)));
                // One scale for both axes keeps the pattern undistorted.
                let side = (hi - lo).max_element().max(1e-6);
                self.mesh.pattern_positions.iter().map(|&p| (p - lo) / side).collect()
            }
        }
    }

    /// `(piece name, fabric name, triangles)` per piece, with names made
    /// safe for use as identifiers.
    pub fn groups(&self) -> Vec<(String, String, &[u32])> {
        let safe = |name: &str| name.chars().map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' }).collect::<String>();
        self.mesh
            .pieces
            .iter()
            .filter(|span| !span.indices.is_empty())
            .map(|span| {
                let piece = self.document.piece(span.piece);
                let fabric = piece.and_then(|p| self.document.fabric(p.fabric));
                (
                    piece.map_or_else(|| format!("piece_{}", span.piece.0), |p| safe(&p.name)),
                    fabric.map_or_else(|| "fabric".to_owned(), |f| safe(&f.name)),
                    &self.mesh.indices[span.indices.start as usize..span.indices.end as usize],
                )
            })
            .collect()
    }
}

/// Writes the draped garment as OBJ with normals and UVs, one group per
/// pattern piece. Returns `(obj, mtl)`; the OBJ refers to the material
/// library as `mtl_name`. Units are metres, Y up.
pub fn export_garment_obj(garment: &GarmentExport<'_>, mtl_name: &str) -> (String, String) {
    let mesh = garment.mesh;
    let mut obj = String::with_capacity(mesh.vertex_count() * 96);
    let _ = writeln!(obj, "# clo360 garment: {} vertices, {} triangles", mesh.vertex_count(), mesh.triangle_count());
    let _ = writeln!(obj, "# units: metres, Y up\nmtllib {mtl_name}");
    for p in &mesh.positions {
        let _ = writeln!(obj, "v {:.6} {:.6} {:.6}", p.x, p.y, p.z);
    }
    for uv in garment.uvs() {
        let _ = writeln!(obj, "vt {:.6} {:.6}", uv.x, uv.y);
    }
    for n in &mesh.normals {
        let _ = writeln!(obj, "vn {:.5} {:.5} {:.5}", n.x, n.y, n.z);
    }
    for (piece, fabric, triangles) in garment.groups() {
        let _ = writeln!(obj, "g {piece}\nusemtl {fabric}");
        for t in triangles.chunks_exact(3) {
            // Position, UV and normal share one index (1-based).
            let [a, b, c] = [t[0] + 1, t[1] + 1, t[2] + 1];
            let _ = writeln!(obj, "f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}");
        }
    }

    let mut mtl = String::new();
    let mut written: Vec<String> = Vec::new();
    for fabric in garment.document.fabrics() {
        let name: String = fabric.name.chars().map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' }).collect();
        if written.contains(&name) {
            continue;
        }
        let [r, g, b] = fabric.base_color;
        let _ = writeln!(mtl, "newmtl {name}\nKd {r:.4} {g:.4} {b:.4}\nKs 0.04 0.04 0.04\nNs {:.1}\nillum 2\n", (1.0 - fabric.roughness) * 200.0);
        written.push(name);
    }
    (obj, mtl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quads_relative_indices_and_normals_are_read() {
        let source = "# a square\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvn 0 0 1\nvt 0 0\nf 1/1/1 2/1/1 3/1/1 4/1/1\nf -4//1 -3//1 -2//1\n";
        let mesh = import_obj(source.as_bytes()).unwrap();
        assert_eq!(mesh.positions.len(), 4);
        assert_eq!(mesh.triangles, vec![[0, 1, 2], [0, 2, 3], [0, 1, 2]]);
        assert!(mesh.normals.iter().all(|n| (*n - Vec3::Z).length() < 1e-6));
    }

    #[test]
    fn missing_normals_are_computed_and_errors_name_the_line() {
        let mesh = import_obj(b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
        assert!((mesh.normals[0] - Vec3::Z).length() < 1e-6);
        let error = import_obj(b"v 0 0 0\nf 1 2 9\n").unwrap_err();
        assert!(matches!(error, IoError::Syntax { line: 2, .. }), "{error}");
    }
}

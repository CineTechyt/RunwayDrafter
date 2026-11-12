//! glTF 2.0 import (`.gltf` with embedded or external buffers, and `.glb`).
//!
//! Reads the triangle geometry of the default scene with node transforms
//! applied. Not supported: sparse accessors, Draco/meshopt compression,
//! and skinning (skinned meshes are read in their bind pose).

use std::path::Path;

use glam::{Mat4, Quat, Vec3};
use serde_json::Value;

use crate::IoError;
use crate::mesh::ImportedMesh;

const GLB_MAGIC: u32 = 0x4654_6C67; // "glTF"
const CHUNK_JSON: u32 = 0x4E4F_534A;
const CHUNK_BIN: u32 = 0x004E_4942;

/// `base` is the directory external buffer files are looked up in.
pub fn import_gltf(bytes: &[u8], base: Option<&Path>) -> Result<ImportedMesh, IoError> {
    let (json, binary) = split_glb(bytes)?;
    let root: Value = serde_json::from_slice(json).map_err(|e| IoError::malformed(format!("glTF JSON: {e}")))?;

    // --- buffers ---------------------------------------------------------
    let mut buffers: Vec<Vec<u8>> = Vec::new();
    for (index, buffer) in array(&root["buffers"]).iter().enumerate() {
        let data = match buffer["uri"].as_str() {
            None if index == 0 => binary.ok_or_else(|| IoError::malformed("buffer 0 has no data"))?.to_vec(),
            None => return Err(IoError::malformed("buffer without uri")),
            Some(uri) if uri.starts_with("data:") => {
                let payload = uri.split_once(";base64,").ok_or_else(|| IoError::malformed("data uri is not base64"))?.1;
                decode_base64(payload)?
            }
            Some(uri) => {
                let base = base.ok_or_else(|| IoError::malformed("external buffer but no directory given"))?;
                std::fs::read(base.join(uri.replace("%20", " ")))?
            }
        };
        buffers.push(data);
    }
    let document = Document { root: &root, buffers };

    // --- scene graph -----------------------------------------------------
    let nodes = array(&root["nodes"]);
    let scene = root["scene"].as_u64().unwrap_or(0) as usize;
    let mut stack: Vec<(usize, Mat4)> = match array(&root["scenes"]).get(scene) {
        Some(scene) => array(&scene["nodes"]).iter().filter_map(|n| Some((n.as_u64()? as usize, Mat4::IDENTITY))).collect(),
        // No scene: every node that is nobody's child is a root.
        None => {
            let children: Vec<u64> = nodes.iter().flat_map(|n| array(&n["children"])).filter_map(Value::as_u64).collect();
            (0..nodes.len()).filter(|i| !children.contains(&(*i as u64))).map(|i| (i, Mat4::IDENTITY)).collect()
        }
    };
    let mut mesh = ImportedMesh::default();
    let mut all_normals = true;
    let mut visited = 0usize;
    while let Some((index, parent)) = stack.pop() {
        visited += 1;
        let Some(node) = nodes.get(index).filter(|_| visited <= 100_000) else {
            return Err(IoError::malformed("bad node hierarchy"));
        };
        let world = parent * node_transform(node);
        stack.extend(array(&node["children"]).iter().filter_map(|c| Some((c.as_u64()? as usize, world))));
        let Some(mesh_index) = node["mesh"].as_u64() else { continue };
        for primitive in array(&root["meshes"][mesh_index as usize]["primitives"]) {
            if primitive["mode"].as_u64().unwrap_or(4) != 4 {
                continue; // points and lines
            }
            let Some(position_accessor) = primitive["attributes"]["POSITION"].as_u64() else { continue };
            let positions = document.vectors(position_accessor as usize)?;
            let base_vertex = mesh.positions.len() as u32;
            mesh.positions.extend(positions.iter().map(|&p| world.transform_point3(p)));
            match primitive["attributes"]["NORMAL"].as_u64() {
                Some(accessor) => {
                    let normals = document.vectors(accessor as usize)?;
                    // Rigid or uniformly scaled transforms only; anything
                    // else gets its normals recomputed below.
                    mesh.normals.extend(normals.iter().map(|&n| world.transform_vector3(n).normalize_or_zero()));
                }
                None => all_normals = false,
            }
            mesh.normals.resize(mesh.positions.len(), Vec3::ZERO);
            let indices = match primitive["indices"].as_u64() {
                Some(accessor) => document.indices(accessor as usize)?,
                None => (0..positions.len() as u32).collect(),
            };
            // A mirroring transform turns the faces inside out.
            let flip = world.determinant() < 0.0;
            mesh.triangles.extend(indices.chunks_exact(3).map(|t| {
                let [a, b, c] = [t[0] + base_vertex, t[1] + base_vertex, t[2] + base_vertex];
                if flip { [a, c, b] } else { [a, b, c] }
            }));
        }
    }
    if !all_normals {
        mesh.normals.clear();
    }
    Ok(mesh.finish())
}

/// Splits a GLB container into its JSON and binary chunks; plain `.gltf`
/// text is returned as it is.
fn split_glb(bytes: &[u8]) -> Result<(&[u8], Option<&[u8]>), IoError> {
    let word = |at: usize| bytes.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    if word(0) != Some(GLB_MAGIC) {
        return Ok((bytes, None));
    }
    let (mut json, mut binary) = (None, None);
    let mut at = 12;
    while let (Some(length), Some(kind)) = (word(at), word(at + 4)) {
        let chunk = bytes.get(at + 8..at + 8 + length as usize).ok_or_else(|| IoError::malformed("truncated GLB chunk"))?;
        match kind {
            CHUNK_JSON => json = Some(chunk),
            CHUNK_BIN => binary = Some(chunk),
            _ => {}
        }
        at += 8 + length as usize;
    }
    Ok((json.ok_or_else(|| IoError::malformed("GLB without JSON chunk"))?, binary))
}

fn array(value: &Value) -> &[Value] {
    value.as_array().map_or(&[], Vec::as_slice)
}

fn node_transform(node: &Value) -> Mat4 {
    let numbers = |value: &Value, count: usize| -> Option<Vec<f32>> {
        let list = value.as_array().filter(|a| a.len() == count)?;
        list.iter().map(|v| v.as_f64().map(|f| f as f32)).collect()
    };
    if let Some(m) = numbers(&node["matrix"], 16) {
        return Mat4::from_cols_slice(&m); // column-major, like glam
    }
    let translation = numbers(&node["translation"], 3).map_or(Vec3::ZERO, |t| Vec3::from_slice(&t));
    let rotation = numbers(&node["rotation"], 4).map_or(Quat::IDENTITY, |r| Quat::from_xyzw(r[0], r[1], r[2], r[3]).normalize());
    let scale = numbers(&node["scale"], 3).map_or(Vec3::ONE, |s| Vec3::from_slice(&s));
    Mat4::from_scale_rotation_translation(scale, rotation, translation)
}

struct Document<'a> {
    root: &'a Value,
    buffers: Vec<Vec<u8>>,
}

impl Document<'_> {
    /// Calls `read` with the bytes of each element of an accessor.
    fn elements(&self, accessor: usize, element_size: usize, mut read: impl FnMut(&[u8])) -> Result<(), IoError> {
        let accessor = &self.root["accessors"][accessor];
        if !accessor["sparse"].is_null() {
            return Err(IoError::malformed("sparse accessors are not supported"));
        }
        let view = &self.root["bufferViews"][accessor["bufferView"].as_u64().ok_or_else(|| IoError::malformed("accessor without bufferView"))? as usize];
        let buffer = self
            .buffers
            .get(view["buffer"].as_u64().unwrap_or(u64::MAX) as usize)
            .ok_or_else(|| IoError::malformed("bufferView without buffer"))?;
        let offset = (view["byteOffset"].as_u64().unwrap_or(0) + accessor["byteOffset"].as_u64().unwrap_or(0)) as usize;
        let stride = view["byteStride"].as_u64().map_or(element_size, |s| s as usize).max(element_size);
        let count = accessor["count"].as_u64().unwrap_or(0) as usize;
        for i in 0..count {
            let at = offset + i * stride;
            read(buffer.get(at..at + element_size).ok_or_else(|| IoError::malformed("accessor runs past its buffer"))?);
        }
        Ok(())
    }

    fn vectors(&self, accessor: usize) -> Result<Vec<Vec3>, IoError> {
        let info = &self.root["accessors"][accessor];
        if info["componentType"].as_u64() != Some(5126) || info["type"].as_str() != Some("VEC3") {
            return Err(IoError::malformed("positions and normals must be float VEC3"));
        }
        let mut out = Vec::new();
        self.elements(accessor, 12, |b| {
            let f = |k: usize| f32::from_le_bytes([b[k], b[k + 1], b[k + 2], b[k + 3]]);
            out.push(Vec3::new(f(0), f(4), f(8)));
        })?;
        Ok(out)
    }

    fn indices(&self, accessor: usize) -> Result<Vec<u32>, IoError> {
        let mut out = Vec::new();
        match self.root["accessors"][accessor]["componentType"].as_u64() {
            Some(5121) => self.elements(accessor, 1, |b| out.push(b[0] as u32))?,
            Some(5123) => self.elements(accessor, 2, |b| out.push(u16::from_le_bytes([b[0], b[1]]) as u32))?,
            Some(5125) => self.elements(accessor, 4, |b| out.push(u32::from_le_bytes([b[0], b[1], b[2], b[3]])))?,
            _ => return Err(IoError::malformed("unsupported index type")),
        }
        Ok(out)
    }
}

fn decode_base64(text: &str) -> Result<Vec<u8>, IoError> {
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let (mut bits, mut count) = (0u32, 0u32);
    for c in text.bytes() {
        let value = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\n' | b'\r' | b' ' => continue,
            _ => return Err(IoError::malformed("bad base64 in data uri")),
        };
        bits = (bits << 6) | value as u32;
        count += 6;
        if count >= 8 {
            count -= 8;
            out.push((bits >> count) as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn encode_base64(data: &[u8]) -> String {
        const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in data.chunks(3) {
            let n = (chunk[0] as u32) << 16 | (*chunk.get(1).unwrap_or(&0) as u32) << 8 | *chunk.get(2).unwrap_or(&0) as u32;
            for k in 0..4 {
                if k <= chunk.len() {
                    out.push(TABLE[(n >> (18 - 6 * k)) as usize & 63] as char);
                } else {
                    out.push('=');
                }
            }
        }
        out
    }

    /// One triangle: three float positions followed by three u16 indices.
    fn triangle_buffer() -> Vec<u8> {
        let mut data = Vec::new();
        for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            data.extend(value.to_le_bytes());
        }
        for index in [0u16, 1, 2] {
            data.extend(index.to_le_bytes());
        }
        data
    }

    fn json(uri: &str) -> String {
        let uri = if uri.is_empty() { String::new() } else { format!(r#""uri":"{uri}","#) };
        format!(
            r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],
            "nodes":[{{"children":[1],"translation":[0,2,0]}},{{"mesh":0,"scale":[2,2,2]}}],
            "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1}}]}}],
            "buffers":[{{{uri}"byteLength":42}}],
            "bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":36}},{{"buffer":0,"byteOffset":36,"byteLength":6}}],
            "accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}},
                         {{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}}]}}"#
        )
    }

    fn check(mesh: &ImportedMesh) {
        // Scaled by 2, then lifted by 2.
        assert_eq!(mesh.positions, vec![Vec3::new(0.0, 2.0, 0.0), Vec3::new(2.0, 2.0, 0.0), Vec3::new(0.0, 4.0, 0.0)]);
        assert_eq!(mesh.triangles, vec![[0, 1, 2]]);
        assert!((mesh.normals[0] - Vec3::Z).length() < 1e-6);
    }

    #[test]
    fn embedded_buffers_and_node_transforms_are_applied() {
        let uri = format!("data:application/octet-stream;base64,{}", encode_base64(&triangle_buffer()));
        check(&import_gltf(json(&uri).as_bytes(), None).unwrap());
    }

    #[test]
    fn binary_containers_are_read() {
        let mut text = json("").into_bytes();
        while text.len() % 4 != 0 {
            text.push(b' ');
        }
        let mut binary = triangle_buffer();
        while binary.len() % 4 != 0 {
            binary.push(0);
        }
        let mut glb = Vec::new();
        glb.extend(GLB_MAGIC.to_le_bytes());
        glb.extend(2u32.to_le_bytes());
        glb.extend((12 + 8 + text.len() as u32 + 8 + binary.len() as u32).to_le_bytes());
        glb.extend((text.len() as u32).to_le_bytes());
        glb.extend(CHUNK_JSON.to_le_bytes());
        glb.extend(&text);
        glb.extend((binary.len() as u32).to_le_bytes());
        glb.extend(CHUNK_BIN.to_le_bytes());
        glb.extend(&binary);
        check(&import_gltf(&glb, None).unwrap());
    }
}

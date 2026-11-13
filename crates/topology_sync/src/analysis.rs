//! Per-vertex measurements of the draped garment for the technical views.

use cad_core::glam::Vec3A;
use physics_collision::TriangleMesh;

use crate::mesh::GarmentMesh;

/// Strain per vertex: the relative elongation of the mesh edges around it,
/// `(length − rest length) / rest length`, averaged with the rest lengths
/// as weights. The rest length of an edge is its length in the flat
/// pattern. 0.05 means stretched by 5 %; negative values are compression
/// (which cloth mostly answers by buckling instead).
pub fn strain_map(mesh: &GarmentMesh, out: &mut Vec<f32>) {
    const MM_TO_M: f32 = 1e-3;
    let count = mesh.vertex_count();
    let mut rest_sum = vec![0.0f32; count];
    out.clear();
    out.resize(count, 0.0);
    for triangle in mesh.indices.chunks_exact(3) {
        for k in 0..3 {
            let (a, b) = (triangle[k] as usize, triangle[(k + 1) % 3] as usize);
            let rest = mesh.pattern_positions[a].distance(mesh.pattern_positions[b]) * MM_TO_M;
            let elongation = mesh.positions[a].distance(mesh.positions[b]) - rest;
            // Interior edges are visited from both triangles, which
            // cancels out in the average.
            for v in [a, b] {
                out[v] += elongation;
                rest_sum[v] += rest;
            }
        }
    }
    for (strain, rest) in out.iter_mut().zip(&rest_sum) {
        *strain = if *rest > 0.0 { *strain / rest } else { 0.0 };
    }
}

/// Clearance per vertex: the distance in metres between the fabric and the
/// body, found by casting a ray from the vertex along the fabric's normal
/// towards the body (see [`TriangleMesh::clearance`]). Vertices further
/// than `max_distance` from the body get `max_distance`.
pub fn clearance_map(
    mesh: &GarmentMesh,
    body: &TriangleMesh,
    max_distance: f32,
    out: &mut Vec<f32>,
) {
    out.clear();
    out.extend(
        mesh.positions
            .iter()
            .zip(&mesh.normals)
            .map(|(&position, &normal)| {
                body.clearance(Vec3A::from(position), Vec3A::from(normal), max_distance)
                    .unwrap_or(max_distance)
            }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GarmentTopology, TopologySync};
    use cad_core::PatternDocument;
    use cad_core::glam::Vec3;

    #[test]
    fn flat_cloth_is_relaxed_and_stretched_cloth_reports_its_strain() {
        let document = PatternDocument::demo_top();
        let mut topology = GarmentTopology::default();
        topology.rebuild(&document);
        let mut strain = Vec::new();
        strain_map(topology.mesh(), &mut strain);
        assert!(
            strain.iter().all(|s| s.abs() < 1e-3),
            "as arranged, nothing is strained"
        );

        // Scale the whole garment by 8 % about the origin.
        for p in &mut topology.mesh_mut().positions {
            *p *= 1.08;
        }
        strain_map(topology.mesh(), &mut strain);
        assert!(strain.iter().all(|s| (s - 0.08).abs() < 2e-3));
    }

    #[test]
    fn clearance_is_the_gap_to_the_body() {
        // A large wall at z = 0 facing +z, and the demo top's front panel
        // moved to hang flat 3 cm in front of it.
        let wall = TriangleMesh::new(
            vec![
                Vec3A::new(-5.0, -5.0, 0.0),
                Vec3A::new(5.0, -5.0, 0.0),
                Vec3A::new(5.0, 5.0, 0.0),
                Vec3A::new(-5.0, 5.0, 0.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        );
        let document = PatternDocument::demo_top();
        let mut topology = GarmentTopology::default();
        topology.rebuild(&document);
        let mesh = topology.mesh_mut();
        for (p, pattern) in mesh.positions.iter_mut().zip(&mesh.pattern_positions) {
            *p = Vec3::new(pattern.x * 1e-3, pattern.y * 1e-3, 0.03);
        }
        mesh.recompute_normals();
        let mut clearance = Vec::new();
        clearance_map(mesh, &wall, 0.2, &mut clearance);
        assert!(clearance.iter().all(|c| (c - 0.03).abs() < 1e-4));
        clearance_map(mesh, &wall, 0.01, &mut clearance);
        assert!(
            clearance.iter().all(|&c| c == 0.01),
            "beyond the range: capped"
        );
    }
}

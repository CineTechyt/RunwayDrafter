//! Export of the draped garment, read back with the crate's own readers.

use cad_core::PatternDocument;
use cad_core::glam::{Vec2, Vec3};
use io_cad::{
    FbxEncoding, GarmentExport, ImportOptions, UvLayout, export_garment_fbx, export_garment_obj,
};
use topology_sync::{GarmentMesh, GarmentTopology, TopologySync};

fn garment() -> (PatternDocument, GarmentTopology) {
    let document = PatternDocument::demo_top();
    let mut topology = GarmentTopology::default();
    topology.rebuild(&document);
    topology.mesh_mut().recompute_normals();
    (document, topology)
}

fn surface_area(positions: &[Vec3], triangles: &[[u32; 3]]) -> f32 {
    triangles
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|i| positions[i as usize]);
            0.5 * (b - a).cross(c - a).length()
        })
        .sum()
}

fn mesh_area(mesh: &GarmentMesh) -> f32 {
    let triangles: Vec<[u32; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|t| [t[0], t[1], t[2]])
        .collect();
    surface_area(&mesh.positions, &triangles)
}

#[test]
fn obj_holds_every_vertex_with_uv_and_normal_grouped_by_piece() {
    let (document, topology) = garment();
    let mesh = topology.mesh();
    let export = GarmentExport {
        mesh,
        document: &document,
        uv_layout: UvLayout::Fabric,
    };
    let (obj, mtl) = export_garment_obj(&export, "top.mtl");

    let count = |prefix: &str| obj.lines().filter(|l| l.starts_with(prefix)).count();
    assert_eq!(count("v "), mesh.vertex_count());
    assert_eq!(count("vt "), mesh.vertex_count());
    assert_eq!(count("vn "), mesh.vertex_count());
    assert_eq!(count("f "), mesh.triangle_count());
    assert_eq!(count("g "), document.pieces().len());
    assert!(
        obj.contains("mtllib top.mtl")
            && obj.contains("g Front")
            && obj.contains("usemtl Cotton_jersey")
    );
    assert!(mtl.contains("newmtl Cotton_jersey") && mtl.contains("newmtl Linen"));

    // The UVs are the mesh's.
    let first_uv: Vec<f32> = obj.lines().find(|l| l.starts_with("vt ")).unwrap()[3..]
        .split(' ')
        .map(|v| v.parse().unwrap())
        .collect();
    assert!((Vec2::new(first_uv[0], first_uv[1]) - mesh.uvs[0]).length() < 1e-5);

    // Read back as geometry: the same surface.
    let read = io_cad::obj::import_obj(obj.as_bytes()).unwrap();
    assert_eq!(read.positions.len(), mesh.vertex_count());
    assert!((surface_area(&read.positions, &read.triangles) - mesh_area(mesh)).abs() < 1e-4);
}

#[test]
fn pattern_uvs_form_an_atlas_without_overlap_between_pieces() {
    let (document, topology) = garment();
    let mesh = topology.mesh();
    let uvs = GarmentExport {
        mesh,
        document: &document,
        uv_layout: UvLayout::Pattern,
    }
    .uvs();
    assert!(
        uvs.iter()
            .all(|uv| uv.min_element() >= 0.0 && uv.max_element() <= 1.0)
    );
    // The demo pieces lie side by side on the canvas: their UV boxes must
    // be disjoint.
    let boxes: Vec<(Vec2, Vec2)> = mesh
        .pieces
        .iter()
        .map(|span| {
            uvs[span.vertices.start as usize..span.vertices.end as usize]
                .iter()
                .fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(lo, hi), &uv| {
                    (lo.min(uv), hi.max(uv))
                })
        })
        .collect();
    assert_eq!(boxes.len(), 2);
    assert!(boxes[0].1.x <= boxes[1].0.x || boxes[1].1.x <= boxes[0].0.x);
}

#[test]
fn fbx_round_trips_through_the_importer() {
    let (document, topology) = garment();
    let mesh = topology.mesh();
    let export = GarmentExport {
        mesh,
        document: &document,
        uv_layout: UvLayout::Fabric,
    };
    let ascii = String::from_utf8(export_garment_fbx(&export, FbxEncoding::Ascii)).unwrap();
    assert!(
        ascii.contains("LayerElementUV")
            && ascii.contains("Geometry::Front")
            && ascii.contains("Material::Linen")
    );
    let binary = export_garment_fbx(&export, FbxEncoding::Binary);
    assert!(binary.starts_with(b"Kaydara FBX Binary"));

    let (mesh_lo, mesh_hi) = mesh
        .positions
        .iter()
        .fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(lo, hi), &p| {
            (lo.min(p), hi.max(p))
        });
    for file in [ascii.as_bytes(), &binary] {
        // The same node tree either way.
        let tree = io_cad::fbx::parse_fbx(file).unwrap();
        let objects = tree.child("Objects").unwrap();
        assert_eq!(
            objects.children_named("Geometry").count(),
            document.pieces().len()
        );
        assert_eq!(objects.children_named("Material").count(), 2);
        let uv = objects
            .child("Geometry")
            .unwrap()
            .child("LayerElementUV")
            .unwrap();
        assert_eq!(
            uv.child("UVIndex").unwrap().numbers().len(),
            3 * (mesh.pieces[0].indices.len() / 3)
        );

        // Centimetres in the file, metres after import.
        let read = io_cad::fbx::import_fbx(file).unwrap();
        assert_eq!(read.positions.len(), mesh.vertex_count());
        assert_eq!(read.triangles.len(), mesh.triangle_count());
        assert!((surface_area(&read.positions, &read.triangles) - mesh_area(mesh)).abs() < 1e-4);
        let (lo, hi) = read.bounds();
        assert!((lo - mesh_lo).length() < 1e-4 && (hi - mesh_hi).length() < 1e-4);
    }
}

#[test]
fn import_avatar_picks_the_reader_by_extension_and_normalises() {
    let directory = std::env::temp_dir().join(format!("clo360_io_test_{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    // A 170 cm tall, Z-up tetrahedron somewhere off the origin.
    let path = directory.join("figure.obj");
    std::fs::write(&path, "v 500 500 0\nv 540 500 0\nv 520 530 0\nv 520 510 170\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n").unwrap();
    let mesh = io_cad::import_avatar(&path, &ImportOptions::default()).unwrap();
    let (lo, hi) = mesh.bounds();
    assert!(
        (hi.y - 1.7).abs() < 1e-5 && lo.y.abs() < 1e-6,
        "stands 1.7 m tall on the floor"
    );
    assert!(
        (lo.x + hi.x).abs() < 1e-5 && (lo.z + hi.z).abs() < 1e-5,
        "centred"
    );
    assert_eq!(mesh.name, "figure");

    let unknown = directory.join("figure.stl");
    std::fs::write(&unknown, "solid").unwrap();
    assert!(matches!(
        io_cad::import_avatar(&unknown, &ImportOptions::default()),
        Err(io_cad::IoError::UnsupportedFormat(_))
    ));
    std::fs::remove_dir_all(&directory).unwrap();
}

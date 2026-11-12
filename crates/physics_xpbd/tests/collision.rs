//! Cloth against cloth, garment layers and the continuous body test, run
//! through the whole solver.

use std::sync::Arc;

use avatar::{BodyModel, BodyParams, PoseParams};
use cad_core::glam::{Quat, Vec2, Vec3, Vec3A};
use cad_core::{Outline, PatternDocument, PieceId};
use physics_collision::TriangleMesh;
use physics_xpbd::XpbdSolver;
use topology_sync::{ClothSolver, GarmentTopology, SolverContext, SolverSettings, TopologySync};

const DT: f32 = 1.0 / 30.0;

/// A 30 cm square lying flat at `height`, shifted sideways by `shift`.
fn add_sheet(document: &mut PatternDocument, name: &str, height: f32, shift: f32) -> PieceId {
    let fabric = document.fabrics()[0].id;
    let outline = Outline::polygon([Vec2::ZERO, Vec2::new(300.0, 0.0), Vec2::new(300.0, 300.0), Vec2::new(0.0, 300.0)]);
    let id = document.add_piece(name, outline, fabric);
    let piece = document.piece_mut(id).unwrap();
    // Pattern y → world −z: the sheet lies in the horizontal plane.
    piece.arrangement.rotation = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    piece.arrangement.translation = Vec3::new(shift, height, shift);
    id
}

fn sheets(upper_height: f32) -> (PatternDocument, GarmentTopology, [PieceId; 2]) {
    let mut document = PatternDocument::new();
    document.add_fabric("Test", [0.5; 3]);
    let lower = add_sheet(&mut document, "Lower", 0.02, 0.0);
    // Shifted so its vertices come down inside the lower sheet's triangles.
    let upper = add_sheet(&mut document, "Upper", upper_height, 0.007);
    let mut topology = GarmentTopology::default();
    topology.rebuild(&document);
    (document, topology, [lower, upper])
}

fn mean_height(topology: &GarmentTopology, piece: PieceId) -> f32 {
    let mesh = topology.mesh();
    let span = mesh.span(piece).unwrap();
    let points = &mesh.positions[span.vertices.start as usize..span.vertices.end as usize];
    points.iter().map(|p| p.y).sum::<f32>() / points.len() as f32
}

fn settings(self_collision: bool) -> SolverSettings {
    SolverSettings { seam_closure_time: 0.0, self_collision, ..SolverSettings::default() }
}

#[test]
fn a_falling_sheet_comes_to_rest_on_the_sheet_below() {
    let run = |self_collision: bool| {
        let (document, mut topology, [lower, upper]) = sheets(0.25);
        let settings = settings(self_collision);
        let ctx = SolverContext { document: &document, settings: &settings, colliders: &[], body: None, body_mesh: None, ground: 0.0, dt: DT };
        let mut solver = XpbdSolver::new();
        solver.rebind(topology.mesh(), &ctx);
        let mut most_contacts = 0;
        for _ in 0..90 {
            solver.step(topology.mesh_mut(), &ctx);
            most_contacts = most_contacts.max(solver.stats().cloth_contacts);
        }
        let fastest = solver.state().velocities.iter().map(|v| v.length()).fold(0.0, f32::max);
        (mean_height(&topology, upper) - mean_height(&topology, lower), most_contacts, fastest)
    };

    let (gap, contacts, fastest) = run(true);
    let thickness = SolverSettings::default().cloth_thickness;
    assert!(contacts > 20, "the sheets are in contact ({contacts} vertices)");
    assert!(gap > 0.4 * thickness, "the upper sheet lies on the lower one, gap {:.2} mm", gap * 1000.0);
    assert!(gap < 3.0 * thickness, "… and not above it, gap {:.2} mm", gap * 1000.0);
    assert!(fastest < 0.05, "… at rest, fastest particle {fastest} m/s");

    // The same without cloth collision: both end up on the floor.
    let (gap, contacts, _) = run(false);
    assert_eq!(contacts, 0);
    assert!(gap.abs() < 0.2 * thickness, "without it they share the floor, gap {:.2} mm", gap * 1000.0);
}

#[test]
fn a_sheet_thrown_through_another_within_one_step_is_caught() {
    // 20 m/s: 67 cm per step, far more than the 23 cm between the sheets,
    // so no sub-step ever sees them close together… but 30 sub-steps of
    // 2 cm do. What must hold is that the upper sheet never ends a step
    // below the lower one.
    let (document, mut topology, [lower, upper]) = sheets(0.25);
    let settings = SolverSettings { gravity: Vec3::new(0.0, -600.0, 0.0), damping: 0.0, ..settings(true) };
    let ctx = SolverContext { document: &document, settings: &settings, colliders: &[], body: None, body_mesh: None, ground: 0.0, dt: DT };
    let mut solver = XpbdSolver::new();
    solver.rebind(topology.mesh(), &ctx);
    let mut crossings = 0;
    for step in 0..20 {
        solver.step(topology.mesh_mut(), &ctx);
        crossings += solver.stats().cloth_crossings;
        let gap = mean_height(&topology, upper) - mean_height(&topology, lower);
        assert!(gap > 0.0, "step {step}: the upper sheet is below the lower one ({:.2} mm)", gap * 1000.0);
    }
    assert!(crossings > 0, "the continuous test had to step in");
}

#[test]
fn fast_cloth_cannot_tunnel_through_a_thin_body_part() {
    // A 2 mm thick plate, 10 cm above the floor: far thinner than a
    // distance field voxel, and here there is no field at all.
    let (w, y0, y1) = (0.5, 0.100, 0.102);
    let corners: Vec<Vec3A> =
        (0..8).map(|i| Vec3A::new(if i & 1 == 0 { -w } else { w }, if i & 2 == 0 { y0 } else { y1 }, if i & 4 == 0 { -w } else { w })).collect();
    let quads = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
    let plate = Arc::new(TriangleMesh::new(corners, quads.iter().flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]).collect()));

    let run = |continuous: bool| {
        let mut document = PatternDocument::new();
        document.add_fabric("Test", [0.5; 3]);
        let sheet = add_sheet(&mut document, "Sheet", 0.6, 0.0);
        let mut topology = GarmentTopology::default();
        topology.rebuild(&document);
        let settings = SolverSettings { continuous_body_collision: continuous, ..settings(false) };
        let ctx = SolverContext { document: &document, settings: &settings, colliders: &[], body: None, body_mesh: Some(&plate), ground: 0.0, dt: DT };
        let mut solver = XpbdSolver::new();
        solver.rebind(topology.mesh(), &ctx);
        let mut stops = 0;
        for _ in 0..45 {
            solver.step(topology.mesh_mut(), &ctx);
            stops += solver.stats().body_stops;
        }
        let lowest = topology.mesh().positions.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        (mean_height(&topology, sheet), lowest, stops)
    };

    let (height, lowest, stops) = run(true);
    assert!(stops > 0);
    assert!(lowest > y1, "no vertex got through the plate (lowest at {lowest})");
    assert!(height < y1 + 0.02, "the sheet lies on the plate (at {height})");

    let (height, _, stops) = run(false);
    assert_eq!(stops, 0);
    assert!(height < 0.02, "without the continuous test it falls through to the floor (at {height})");
}

#[test]
fn an_outer_layer_stays_outside_the_garment_under_it() {
    let Ok(model) = BodyModel::load(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/avatar/body.bin")) else {
        eprintln!("skipped: assets/avatar/body.bin not found");
        return;
    };
    let rest = model.build(&BodyParams::female());
    let measurements = avatar::measure(&model, &rest);
    let body = model.pose(&rest, &PoseParams::default().to_pose(&model, &rest));
    let field = Arc::new(avatar::distance_field(&body, 0.008, 0.01));
    let collider = Arc::new(body.collider());

    // The same top twice, the second as layer 2 and only 4 % larger: tight
    // enough over the first that the two are pressed together.
    let mut document = PatternDocument::demo_top();
    let outer = document.add_layer_copy(1.04, Vec2::new(0.0, 800.0));
    assert_eq!(document.top_layer(), 2);
    avatar::arrange_around(&mut document, &body, &measurements);
    let mut topology = GarmentTopology::default();
    topology.rebuild(&document);

    let settings = SolverSettings::default();
    let ctx = SolverContext { document: &document, settings: &settings, colliders: &[], body: Some(&field), body_mesh: Some(&collider), ground: 0.0, dt: DT };
    let mut solver = XpbdSolver::new();
    solver.rebind(topology.mesh(), &ctx);
    let mut most_contacts = 0;
    for _ in 0..150 {
        solver.step(topology.mesh_mut(), &ctx);
        most_contacts = most_contacts.max(solver.stats().cloth_contacts);
    }
    assert!(most_contacts > 50, "the layers touch ({most_contacts} vertices)");

    // Every vertex of the outer garment must be on the outer side of the
    // inner garment: cast from it towards the body and see what is hit.
    let mesh = topology.mesh();
    let inner_triangles: Vec<[u32; 3]> = mesh
        .pieces
        .iter()
        .filter(|span| !outer.contains(&span.piece))
        .flat_map(|span| mesh.indices[span.indices.start as usize..span.indices.end as usize].chunks_exact(3))
        .map(|t| [t[0], t[1], t[2]])
        .collect();
    let inner = TriangleMesh::new(mesh.positions.iter().map(|&p| p.into()).collect(), inner_triangles.clone());
    let (mut outside, mut inside) = (0, 0);
    for span in mesh.pieces.iter().filter(|span| outer.contains(&span.piece)) {
        for v in span.vertices.clone() {
            let p = Vec3A::from(mesh.positions[v as usize]);
            // The nearest point of the inner garment, and which of its
            // sides faces away from the body there.
            let Some(nearest) = inner.closest_point(p, 0.03) else { continue };
            let [a, b, c] = inner_triangles[nearest.triangle as usize].map(|i| Vec3A::from(mesh.positions[i as usize]));
            let normal = (b - a).cross(c - a).normalize_or_zero();
            let Some((_, gradient)) = field.sample(nearest.point.into()) else { continue };
            let facing = normal.dot(Vec3A::from(gradient).normalize_or_zero());
            if facing.abs() < 0.5 {
                continue; // far from the body, or cloth edge-on to it
            }
            if (p - nearest.point).dot(normal) * facing > 0.0 { outside += 1 } else { inside += 1 }
        }
    }
    eprintln!("outer vertices over the inner garment: {outside}, under it: {inside}; most contacts {most_contacts}; step {:.1} ms", solver.stats().step_ms);
    assert!(outside > 300, "most of the outer garment covers the inner one ({outside} vertices)");
    assert!(inside * 50 <= outside, "{inside} outer vertices are under the inner garment, {outside} over it");
    let (_, over_strained, gap) = solver.diagnostics();
    assert!(gap < 0.003 && over_strained < 0.05, "both garments are sewn and unstretched: gap {gap}, {over_strained}");
}

//! The global solver on the demo top worn by the avatar.

use std::sync::Arc;

use avatar::{BodyModel, BodyParams, PoseParams};
use cad_core::PatternDocument;
use glam::Vec3;
use physics_xpbd::XpbdSolver;
use topology_sync::{
    ClothSolver, GarmentMesh, GarmentTopology, SolverBackend, SolverContext, SolverSettings,
    TopologySync,
};

const DT: f32 = 1.0 / 30.0;

/// The 99th percentile of the mesh edges' elongation.
fn stretch(mesh: &GarmentMesh) -> f32 {
    let mut strains: Vec<f32> = mesh
        .indices
        .chunks_exact(3)
        .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
        .map(|(a, b)| {
            let rest = mesh.pattern_positions[a as usize]
                .distance(mesh.pattern_positions[b as usize])
                * 1e-3;
            mesh.positions[a as usize].distance(mesh.positions[b as usize]) / rest - 1.0
        })
        .collect();
    strains.sort_by(f32::total_cmp);
    strains[strains.len() * 99 / 100]
}

/// Drapes the top for three seconds; returns the mesh, how far its
/// vertices still move per frame (95th percentile) and the solver.
fn drape(backend: SolverBackend, stiffness: Option<f32>) -> (GarmentMesh, f32, usize) {
    let model = BodyModel::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/avatar/body.bin"
    ))
    .expect("run tools/bake_avatar.py");
    let rest = model.build(&BodyParams::female());
    let measurements = avatar::measure(&model, &rest);
    let body = model.pose(&rest, &PoseParams::default().to_pose(&model, &rest));
    let mut document = PatternDocument::demo_top();
    avatar::arrange_around(&mut document, &body, &measurements);
    if let Some(stiffness) = stiffness {
        let fabrics: Vec<_> = document.pieces().iter().map(|p| p.fabric).collect();
        for fabric in fabrics {
            let physics = &mut document.fabric_mut(fabric).unwrap().physics;
            (physics.stretch_weft, physics.stretch_warp) = (stiffness, stiffness);
        }
    }
    let mut topology = GarmentTopology::default();
    topology.rebuild(&document);
    let field = Arc::new(avatar::distance_field(&body, 0.008, 0.0));
    let settings = SolverSettings {
        backend,
        ..SolverSettings::default()
    };
    let ctx = SolverContext {
        document: &document,
        settings: &settings,
        colliders: &[],
        body: Some(&field),
        body_mesh: None,
        ground: 0.0,
        dt: DT,
    };
    let mut solver = XpbdSolver::new();
    solver.rebind(topology.mesh(), &ctx);
    let mut last: Vec<Vec3> = Vec::new();
    for _ in 0..90 {
        last.clone_from(&topology.mesh().positions);
        solver.step(topology.mesh_mut(), &ctx);
    }
    let mesh = topology.mesh().clone();
    let mut motion: Vec<f32> = mesh
        .positions
        .iter()
        .zip(&last)
        .map(|(a, b)| a.distance(*b))
        .collect();
    motion.sort_by(f32::total_cmp);
    let inside = mesh
        .positions
        .iter()
        .filter(|p| field.sample(**p).is_some_and(|(d, _)| d < 0.0))
        .count();
    (mesh, motion[motion.len() * 95 / 100], inside)
}

#[test]
fn the_worn_top_comes_to_rest_outside_the_body() {
    let (mesh, motion, inside) = drape(SolverBackend::Global, None);
    let stretch = stretch(&mesh);
    println!(
        "99 % of edges within {:.1} %, still moving {:.4} mm a frame, {inside} vertices inside",
        stretch * 100.0,
        motion * 1e3
    );
    assert_eq!(inside, 0);
    assert!(motion < 5e-5, "still moving {motion} m a frame");
    assert!(stretch < 0.07, "stretched by {stretch}");
    // It hangs from the shoulders down to the hips.
    let (low, high) = mesh
        .positions
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p.y), hi.max(p.y)));
    assert!(high > 1.3 && low > 0.6 && low < 1.0, "{low}..{high}");
}

#[test]
fn stiff_fabric_keeps_its_length() {
    // A fabric thirty times stiffer costs the global solve nothing, and
    // it holds its length; relaxing one constraint at a time cannot
    // reach that stiffness in a frame.
    let (global, ..) = drape(SolverBackend::Global, Some(0.95));
    let (explicit, ..) = drape(SolverBackend::Cpu, Some(0.95));
    let (a, b) = (stretch(&global), stretch(&explicit));
    println!("99 % of edges within {:.1} % (global) and {:.1} % (explicit)", a * 100.0, b * 100.0);
    assert!(a < 0.03, "stretched by {a}");
    assert!(a < 0.6 * b, "{a} against {b}");
}

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
    let collider = Arc::new(body.collider());
    let settings = SolverSettings {
        backend,
        ..SolverSettings::default()
    };
    let ctx = SolverContext {
        document: &document,
        settings: &settings,
        colliders: &[],
        body: Some(&field),
        body_mesh: Some(&collider),
        ground: 0.0,
        dt: DT,
    };
    let mut solver = XpbdSolver::new();
    solver.rebind(topology.mesh(), &ctx);
    let mut last: Vec<Vec3> = Vec::new();
    for frame in 0..90 {
        last.clone_from(&topology.mesh().positions);
        solver.step(topology.mesh_mut(), &ctx);
        if std::env::var("GLOBAL_TRACE").is_ok() {
            let s = solver.global_stats();
            eprintln!(
                "frame {frame:2}: p99 {:6.1} %, {} contacts, {} pinched, {} halvings, moved {:.2} mm",
                stretch(topology.mesh()) * 100.0,
                s.contacts,
                s.pinched,
                s.halvings,
                s.last_step * 1e3
            );
        }
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
        .fold((f32::MAX, f32::MIN), |(lo, hi), p| {
            (lo.min(p.y), hi.max(p.y))
        });
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
    println!(
        "99 % of edges within {:.1} % (global) and {:.1} % (explicit)",
        a * 100.0,
        b * 100.0
    );
    assert!(a < 0.03, "stretched by {a}");
    assert!(a < 0.6 * b, "{a} against {b}");
}

/// Two boxes 1 m wide either side of x = 0, `gap` apart; the one on +X is
/// raised by `lift` and both move at the given velocities.
fn boxes(gap: f32, lift: f32, left: Vec3, right: Vec3) -> physics_collision::TriangleMesh {
    use glam::Vec3A;
    let mut positions = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut velocities = Vec::new();
    for (centre, velocity) in [
        (Vec3::new(-0.5 - gap / 2.0, 1.0, 0.0), left),
        (Vec3::new(0.5 + gap / 2.0, 1.0 + lift, 0.0), right),
    ] {
        let base = positions.len() as u32;
        for i in 0..8 {
            let corner = Vec3::new(
                (i & 1) as f32 - 0.5,
                ((i >> 1) & 1) as f32 - 0.5,
                ((i >> 2) & 1) as f32 - 0.5,
            );
            positions.push(Vec3A::from(centre + corner));
            velocities.push(Vec3A::from(velocity));
        }
        for q in [
            [0, 2, 3, 1],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 4, 6, 2],
            [1, 3, 7, 5],
        ] {
            triangles.push([base + q[0], base + q[1], base + q[2]]);
            triangles.push([base + q[0], base + q[2], base + q[3]]);
        }
    }
    physics_collision::TriangleMesh::new(positions, triangles).with_velocities(velocities)
}

/// A 10 cm square of cloth hanging in the plane x = 0, squeezed between
/// two boxes; the boxes move up at the given speeds for a second. Returns
/// how far the cloth rose, metres.
fn squeezed(left: f32, right: f32) -> f32 {
    use cad_core::glam::Vec2;
    let mut document = PatternDocument::new();
    let fabric = document.add_fabric("Cotton", [0.8, 0.8, 0.8]);
    document.add_piece(
        "Patch",
        cad_core::Outline::polygon([
            Vec2::new(0.0, 0.0),
            Vec2::new(100.0, 0.0),
            Vec2::new(100.0, 100.0),
            Vec2::new(0.0, 100.0),
        ]),
        fabric,
    );
    let mut topology = GarmentTopology::default();
    topology.rebuild(&document);
    let mesh = topology.mesh_mut();
    for (p, q) in mesh.positions.iter_mut().zip(&mesh.pattern_positions) {
        *p = Vec3::new(0.0, 1.0 + q.y * 1e-3, -0.05 + q.x * 1e-3);
    }
    let settings = SolverSettings {
        backend: SolverBackend::Global,
        seam_closure_time: 0.0,
        continuous_body_collision: false,
        self_collision: false,
        ..SolverSettings::default()
    };
    // Each side reaches 1 mm into the cloth's margin.
    let gap = 2.0 * (settings.collision_margin - 0.001);
    let mean = |mesh: &GarmentMesh| {
        mesh.positions.iter().map(|p| p.y).sum::<f32>() / mesh.positions.len() as f32
    };
    let start = mean(topology.mesh());
    if std::env::var("GLOBAL_TRACE").is_ok() {
        let y: Vec<f32> = topology.mesh().positions.iter().map(|p| p.y).collect();
        eprintln!(
            "start: {} vertices, y {:.4}..{:.4}, mean {start:.4}",
            y.len(),
            y.iter().copied().fold(f32::MAX, f32::min),
            y.iter().copied().fold(f32::MIN, f32::max)
        );
    }
    let mut solver = XpbdSolver::new();
    let (mut lift_left, mut lift_right) = (0.0f32, 0.0f32);
    for frame in 0..30 {
        if frame > 0 {
            lift_left += left * DT;
            lift_right += right * DT;
        }
        // Both boxes, moved as a whole by the left one's lift.
        let mut mesh = boxes(gap, lift_right - lift_left, Vec3::Y * left, Vec3::Y * right);
        if lift_left != 0.0 {
            let moved: Vec<glam::Vec3A> = mesh
                .positions()
                .iter()
                .map(|p| *p + glam::Vec3A::Y * lift_left)
                .collect();
            let velocities = (0..16)
                .map(|i| glam::Vec3A::Y * if i < 8 { left } else { right })
                .collect();
            mesh = physics_collision::TriangleMesh::new(moved, {
                let mut t = Vec::new();
                for base in [0u32, 8] {
                    for q in [
                        [0, 2, 3, 1],
                        [4, 5, 7, 6],
                        [0, 1, 5, 4],
                        [2, 6, 7, 3],
                        [0, 4, 6, 2],
                        [1, 3, 7, 5],
                    ] {
                        t.push([base + q[0], base + q[1], base + q[2]]);
                        t.push([base + q[0], base + q[2], base + q[3]]);
                    }
                }
                t
            })
            .with_velocities(velocities);
        }
        let mesh = Arc::new(mesh);
        let ctx = SolverContext {
            document: &document,
            settings: &settings,
            colliders: &[],
            body: None,
            body_mesh: Some(&mesh),
            ground: -10.0,
            dt: DT,
        };
        if frame == 0 {
            solver.rebind(topology.mesh(), &ctx);
        }
        solver.step(topology.mesh_mut(), &ctx);
        if std::env::var("GLOBAL_TRACE").is_ok() {
            let s = solver.global_stats();
            let x: Vec<f32> = topology.mesh().positions.iter().map(|p| p.x).collect();
            eprintln!(
                "frame {frame}: rose {:.4}, y {:.4}..{:.4}, {} contacts, {} pinched, x {:.4}..{:.4}, {} halvings",
                mean(topology.mesh()) - start,
                topology
                    .mesh()
                    .positions
                    .iter()
                    .map(|p| p.y)
                    .fold(f32::MAX, f32::min),
                topology
                    .mesh()
                    .positions
                    .iter()
                    .map(|p| p.y)
                    .fold(f32::MIN, f32::max),
                s.contacts,
                s.pinched,
                x.iter().copied().fold(f32::MAX, f32::min),
                x.iter().copied().fold(f32::MIN, f32::max),
                s.halvings
            );
        }
    }
    mean(topology.mesh()) - start
}

#[test]
fn cloth_squeezed_between_two_surfaces_is_held_by_both() {
    // Squeezed between two still boxes, the cloth hangs where it is.
    let still = squeezed(0.0, 0.0);
    // Both rise at 0.3 m/s: the cloth rides up with them.
    let both = squeezed(0.3, 0.3);
    // Only one rises (an arm going up past the torso): the still one holds
    // the cloth as firmly as the moving one drags it.
    let one = squeezed(0.0, 0.3);
    println!("rose {still:.3} m held still, {both:.3} m both rising, {one:.3} m one rising");
    assert!(still.abs() < 0.005, "{still}");
    assert!(both > 0.2, "{both}");
    assert!(one.abs() < 0.05, "{one}");
}

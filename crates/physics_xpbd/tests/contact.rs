//! The contact solver (vertex block descent with offset geometric
//! contact) on a real garment and body.

use std::sync::Arc;

use avatar::{BodyModel, BodyParams, PoseParams};
use cad_core::PatternDocument;
use physics_collision::TriangleMesh;
use physics_xpbd::XpbdSolver;
use topology_sync::{
    ClothSolver, GarmentTopology, SolverBackend, SolverContext, SolverSettings, TopologySync,
};

const DT: f32 = 1.0 / 30.0;

struct Drape {
    stretch: (f32, f32, f32),
    through: usize,
    step_ms: f32,
    lowest_clearance: f32,
}

fn drape(backend: SolverBackend, steps: usize) -> Option<Drape> {
    let model = BodyModel::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/avatar/body.bin"
    ))
    .ok()?;
    let rest = model.build(&BodyParams::female());
    let measurements = avatar::measure(&model, &rest);
    let body = model.pose(&rest, &PoseParams::default().to_pose(&model, &rest));
    let field = Arc::new(avatar::distance_field(&body, 0.01, 0.01));
    let collider = Arc::new(body.collider());
    let mut document = PatternDocument::demo_top();
    avatar::arrange_around(&mut document, &body, &measurements);
    let mut topology = GarmentTopology::default();
    topology.rebuild(&document);
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
    let mut ms = 0.0;
    for _ in 0..steps {
        solver.step(topology.mesh_mut(), &ctx);
        ms += solver.stats().step_ms;
    }
    let mesh = topology.mesh();
    let cloth = TriangleMesh::new(
        mesh.positions.iter().map(|&p| p.into()).collect(),
        mesh.indices.as_chunks::<3>().0.to_vec(),
    );
    let through = collider
        .positions()
        .iter()
        .zip(collider.vertex_normals())
        .filter(|(p, n)| cloth.raycast(**p - **n * 0.008, **n, 0.008).is_some())
        .count();
    let lowest_clearance = mesh
        .positions
        .iter()
        .filter_map(|&p| field.sample(p).map(|(d, _)| d))
        .fold(f32::INFINITY, f32::min);
    eprintln!("{backend:?}: stats {:?}", solver.stats());
    Some(Drape {
        stretch: solver.diagnostics(),
        through,
        step_ms: ms / steps as f32,
        lowest_clearance,
    })
}

#[test]
fn the_contact_solver_drapes_a_top_without_clipping() {
    for backend in [SolverBackend::Cpu, SolverBackend::Contact] {
        let Some(d) = drape(backend, 150) else {
            eprintln!("skipped: assets/avatar/body.bin not found");
            return;
        };
        eprintln!(
            "{backend:?}: worst {:.1} %, over 10 %: {:.1} %, seam gap {:.1} mm, skin through {}, lowest {:.1} mm, {:.1} ms/step",
            d.stretch.0 * 100.0,
            d.stretch.1 * 100.0,
            d.stretch.2 * 1000.0,
            d.through,
            d.lowest_clearance * 1000.0,
            d.step_ms
        );
    }
}

/// A closed box, outward-facing, as a body.
fn block(centre: cad_core::glam::Vec3, half: cad_core::glam::Vec3) -> avatar::BodyMesh {
    use cad_core::glam::Vec3;
    let positions: Vec<Vec3> = (0..8)
        .map(|i| {
            centre
                + half
                    * Vec3::new(
                        if i & 1 == 0 { -1.0 } else { 1.0 },
                        if i & 2 == 0 { -1.0 } else { 1.0 },
                        if i & 4 == 0 { -1.0 } else { 1.0 },
                    )
        })
        .collect();
    let quads = [
        [0, 2, 3, 1],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 6, 7, 3],
        [0, 4, 6, 2],
        [1, 3, 7, 5],
    ];
    let triangles = quads
        .iter()
        .flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
        .collect();
    let normals = positions
        .iter()
        .map(|p| (*p - centre).normalize())
        .collect();
    avatar::BodyMesh::from_triangles(positions, normals, triangles)
}

/// A 30 cm sheet resting on a block that then moves with `velocity`
/// (m/s) for `seconds`. Returns how far the sheet's centre and the block
/// moved.
fn ride(
    backend: SolverBackend,
    velocity: cad_core::glam::Vec3,
    seconds: f32,
) -> (cad_core::glam::Vec3, cad_core::glam::Vec3) {
    use cad_core::Outline;
    use cad_core::glam::{Quat, Vec2, Vec3};
    let mut document = PatternDocument::new();
    document.add_fabric("Test", [0.5; 3]);
    let fabric = document.fabrics()[0].id;
    let outline = Outline::polygon([
        Vec2::ZERO,
        Vec2::new(300.0, 0.0),
        Vec2::new(300.0, 300.0),
        Vec2::new(0.0, 300.0),
    ]);
    let id = document.add_piece("Sheet", outline, fabric);
    let piece = document.piece_mut(id).unwrap();
    piece.arrangement.rotation = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    piece.arrangement.translation = Vec3::new(-0.15, 1.006, 0.15);
    let mut topology = GarmentTopology::default();
    topology.rebuild(&document);
    let settings = SolverSettings {
        backend,
        seam_closure_time: 0.0,
        ..SolverSettings::default()
    };
    let half = Vec3::new(0.3, 0.2, 0.3);
    let mut centre = Vec3::new(0.0, 0.8, 0.0);
    let body = |centre: Vec3, velocity: Vec3| {
        let mesh = block(centre, half);
        let velocities = vec![velocity; mesh.positions.len()];
        let options = avatar::FieldOptions {
            velocities: Some(&velocities),
            ..Default::default()
        };
        (
            Arc::new(avatar::distance_field_within(&mesh, 0.01, 0.0, &options)),
            Arc::new(mesh.collider()),
        )
    };
    let mut solver = XpbdSolver::new();
    let (field, collider) = body(centre, Vec3::ZERO);
    let ctx = SolverContext {
        document: &document,
        settings: &settings,
        colliders: &[],
        body: Some(&field),
        body_mesh: Some(&collider),
        ground: 0.0,
        dt: DT,
    };
    solver.rebind(topology.mesh(), &ctx);
    // Settle on the block first.
    for _ in 0..30 {
        solver.step(topology.mesh_mut(), &ctx);
    }
    let mean = |t: &GarmentTopology| {
        t.mesh().positions.iter().copied().sum::<Vec3>() / t.mesh().positions.len() as f32
    };
    let before = mean(&topology);
    let start = centre;
    for _ in 0..(seconds / DT).round() as usize {
        centre += velocity * DT;
        let (field, collider) = body(centre, velocity);
        let ctx = SolverContext {
            document: &document,
            settings: &settings,
            colliders: &[],
            body: Some(&field),
            body_mesh: Some(&collider),
            ground: 0.0,
            dt: DT,
        };
        solver.step(topology.mesh_mut(), &ctx);
    }
    (mean(&topology) - before, centre - start)
}

#[test]
fn cloth_lifts_off_a_body_that_drops_away_from_it() {
    use cad_core::glam::Vec3;
    // The block drops at 3 m/s, faster than anything falls in 0.2 s
    // (gravity reaches 2 m/s). Cloth lying on it must not be pulled down
    // with it: it falls freely and is left behind.
    for backend in [SolverBackend::Cpu, SolverBackend::Contact] {
        let (cloth, block) = ride(backend, Vec3::new(0.0, -3.0, 0.0), 0.2);
        eprintln!(
            "{backend:?}: block {:.3} m, cloth {:.3} m",
            block.y, cloth.y
        );
        if backend == SolverBackend::Contact {
            // Free fall for 0.2 s is 0.196 m.
            assert!(
                cloth.y > -0.25,
                "the cloth was dragged down with the block: {} m vs {} m",
                cloth.y,
                block.y
            );
            assert!(cloth.y < -0.12, "the cloth falls: {} m", cloth.y);
        }
    }
}

#[test]
fn cloth_rides_along_on_a_body_moving_under_it() {
    use cad_core::glam::Vec3;
    // The block jumps to 0.4 m/s. Only the cloth's weight presses on it, so
    // Coulomb friction (μ = 0.4) speeds the cloth up at μ·g = 3.9 m/s² and
    // it slips 2 cm before it keeps up: 18 of the block's 20 cm. (The
    // explicit solver carries it about 4 cm.)
    let (cloth, block) = ride(SolverBackend::Contact, Vec3::new(0.4, 0.0, 0.0), 0.5);
    eprintln!("block {:.3} m, cloth {:.3} m", block.x, cloth.x);
    assert!(
        cloth.x > 0.7 * block.x,
        "friction carries the cloth: {} m of {} m",
        cloth.x,
        block.x
    );
}

/// Cloth edges that cross a cloth triangle they are not part of.
fn self_intersections(mesh: &topology_sync::GarmentMesh) -> usize {
    use cad_core::glam::Vec3A;
    use physics_collision::ccd::segment_triangle;
    let x: Vec<Vec3A> = mesh.positions.iter().map(|&p| p.into()).collect();
    let triangles = mesh.indices.as_chunks::<3>().0;
    let mut edges: Vec<[u32; 2]> = triangles
        .iter()
        .flat_map(|t| [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]])
        .map(|[a, b]| [a.min(b), a.max(b)])
        .collect();
    edges.sort_unstable();
    edges.dedup();
    edges
        .iter()
        .filter(|[a, b]| {
            triangles.iter().any(|t| {
                !t.contains(a)
                    && !t.contains(b)
                    && segment_triangle(
                        x[*a as usize],
                        x[*b as usize],
                        x[t[0] as usize],
                        x[t[1] as usize],
                        x[t[2] as usize],
                    )
                    .is_some()
            })
        })
        .count()
}

#[test]
fn a_sheet_folding_over_a_block_never_passes_through_itself() {
    use cad_core::Outline;
    use cad_core::glam::{Quat, Vec2, Vec3};
    let mut document = PatternDocument::new();
    document.add_fabric("Test", [0.5; 3]);
    let fabric = document.fabrics()[0].id;
    let outline = Outline::polygon([
        Vec2::ZERO,
        Vec2::new(600.0, 0.0),
        Vec2::new(600.0, 600.0),
        Vec2::new(0.0, 600.0),
    ]);
    let id = document.add_piece("Sheet", outline, fabric);
    let piece = document.piece_mut(id).unwrap();
    piece.arrangement.rotation = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    piece.arrangement.translation = Vec3::new(-0.3, 0.45, 0.3);
    let mut topology = GarmentTopology::default();
    topology.rebuild(&document);
    // A narrow block: the sheet hangs down both sides and piles on the floor.
    let mesh = block(Vec3::new(0.0, 0.2, 0.0), Vec3::new(0.05, 0.2, 0.05));
    let field = Arc::new(avatar::distance_field(&mesh, 0.01, 0.0));
    let collider = Arc::new(mesh.collider());
    let settings = SolverSettings {
        seam_closure_time: 0.0,
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
    let mut most = 0;
    for step in 0..60 {
        solver.step(topology.mesh_mut(), &ctx);
        most = most.max(solver.stats().cloth_contacts);
        if step % 20 == 19 {
            let crossings = self_intersections(topology.mesh());
            eprintln!(
                "step {step}: {crossings} edges through the cloth, {} contacts",
                solver.stats().cloth_contacts
            );
            assert_eq!(
                crossings, 0,
                "the cloth passed through itself by step {step}"
            );
        }
    }
    assert!(most > 20, "the sheet folds onto itself ({most} contacts)");
    let lowest = topology
        .mesh()
        .positions
        .iter()
        .map(|p| p.y)
        .fold(f32::MAX, f32::min);
    assert!(lowest > 0.0, "it rests on the floor ({lowest})");
}

#[test]
fn buttons_done_up_from_scratch_close_the_right_sides_and_lap_on_top() {
    let Ok(model) = BodyModel::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/avatar/body.bin"
    )) else {
        eprintln!("skipped: assets/avatar/body.bin not found");
        return;
    };
    let rest = model.build(&BodyParams::female());
    let measurements = avatar::measure(&model, &rest);
    let body = model.pose(&rest, &PoseParams::default().to_pose(&model, &rest));
    let field = Arc::new(avatar::distance_field(&body, 0.01, 0.01));
    let collider = Arc::new(body.collider());

    // The waistcoat as someone would draw it: both fronts on the first
    // layer, buttons and holes placed but not done up.
    let mut document = PatternDocument::demo_waistcoat();
    let expected: Vec<_> = document
        .pieces()
        .iter()
        .flat_map(|p| p.buttons.iter().map(|b| b.fastened_to.unwrap()))
        .collect();
    for piece in document.pieces().iter().map(|p| p.id).collect::<Vec<_>>() {
        document.piece_mut(piece).unwrap().layer = 1;
    }
    document.unfasten_buttons();
    avatar::arrange_around(&mut document, &body, &measurements);
    assert_eq!(document.fasten_buttons(0.25), expected.len());
    let paired: Vec<_> = document
        .pieces()
        .iter()
        .flat_map(|p| p.buttons.iter().map(|b| b.fastened_to.unwrap()))
        .collect();
    assert_eq!(
        paired, expected,
        "every button goes through its own hole, top to bottom"
    );

    // Arranged again with the buttons done up, then sewn and worn.
    avatar::arrange_around(&mut document, &body, &measurements);
    let mut topology = GarmentTopology::default();
    topology.rebuild(&document);
    let settings = SolverSettings::default();
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
    for _ in 0..150 {
        solver.step(topology.mesh_mut(), &ctx);
    }

    // Where the fronts overlap, the buttonhole front is further from the
    // body than the button front under it.
    let mesh = topology.mesh();
    let piece = |name: &str| {
        document
            .pieces()
            .iter()
            .find(|p| p.name == name)
            .unwrap()
            .id
    };
    let (over, under) = (
        mesh.span(piece("Right front")).unwrap(),
        mesh.span(piece("Left front")).unwrap(),
    );
    let under_triangles: Vec<[u32; 3]> = mesh.indices
        [under.indices.start as usize..under.indices.end as usize]
        .as_chunks::<3>()
        .0
        .to_vec();
    let under_mesh = TriangleMesh::new(
        mesh.positions.iter().map(|&p| p.into()).collect(),
        under_triangles,
    );
    let (mut outside, mut inside) = (0, 0);
    for v in over.vertices.clone() {
        let p = cad_core::glam::Vec3A::from(mesh.positions[v as usize]);
        let Some((_, gradient)) = field.sample(p.into()) else {
            continue;
        };
        let towards_body = -cad_core::glam::Vec3A::from(gradient).normalize_or_zero();
        // Straight towards the body: is the button front in the way?
        if under_mesh.raycast(p, towards_body, 0.03).is_some() {
            outside += 1;
        } else if under_mesh.raycast(p, -towards_body, 0.03).is_some() {
            inside += 1;
        }
    }
    eprintln!("buttonhole front over the button front: {outside} vertices, under it: {inside}");
    assert!(outside > 10, "the fronts overlap ({outside})");
    assert!(
        inside * 20 <= outside,
        "{inside} vertices of the buttonhole front ended up under the button front"
    );
}

/// `cargo test --release -p physics_xpbd --test contact bench -- --ignored --nocapture`:
/// step time and how still the garment gets, per backend and garment.
#[test]
#[ignore]
fn bench_settling() {
    let Ok(model) = BodyModel::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/avatar/body.bin"
    )) else {
        return;
    };
    let rest = model.build(&BodyParams::female());
    let measurements = avatar::measure(&model, &rest);
    let body = model.pose(&rest, &PoseParams::default().to_pose(&model, &rest));
    let field = Arc::new(avatar::distance_field(&body, 0.008, 0.01));
    let collider = Arc::new(body.collider());
    let backends: Vec<SolverBackend> = match std::env::var("BENCH_BACKEND").as_deref() {
        Ok("cpu") => vec![SolverBackend::Cpu],
        Ok("contact") => vec![SolverBackend::Contact],
        _ => vec![SolverBackend::Cpu, SolverBackend::Contact],
    };
    for (name, mut document) in [
        ("top", PatternDocument::demo_top()),
        ("waistcoat", PatternDocument::demo_waistcoat()),
    ] {
        avatar::arrange_around(&mut document, &body, &measurements);
        for &backend in &backends {
            let mut topology = GarmentTopology::default();
            topology.rebuild(&document);
            let friction = std::env::var("BENCH_FRICTION")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(SolverSettings::default().friction);
            let settings = SolverSettings {
                backend,
                friction,
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
            let mut line = format!(
                "{name} {backend:?} ({} vertices):",
                topology.mesh().vertex_count()
            );
            let mut ms = 0.0;
            for step in 0..300 {
                let before = topology.mesh().positions.clone();
                solver.step(topology.mesh_mut(), &ctx);
                ms += solver.stats().step_ms;
                if step % 50 == 49 {
                    let mut moves: Vec<f32> = topology
                        .mesh()
                        .positions
                        .iter()
                        .zip(&before)
                        .map(|(a, b)| a.distance(*b) * 1000.0)
                        .collect();
                    moves.sort_by(f32::total_cmp);
                    let (fastest, _) = topology
                        .mesh()
                        .positions
                        .iter()
                        .zip(&before)
                        .enumerate()
                        .map(|(i, (a, b))| (i, a.distance(*b)))
                        .fold((0, 0.0f32), |m, (i, d)| if d > m.1 { (i, d) } else { m });
                    let piece = topology
                        .mesh()
                        .pieces
                        .iter()
                        .find(|s| s.vertices.contains(&(fastest as u32)))
                        .and_then(|s| document.piece(s.piece))
                        .map_or("?", |p| p.name.as_str());
                    line += &format!(
                        " | t={:.1}s {:.1} ms/step, moved max {:.2} p95 {:.3} mm (v{fastest} on {piece} at {:.2?})",
                        (step + 1) as f32 * DT,
                        ms / 50.0,
                        moves[moves.len() - 1],
                        moves[moves.len() * 95 / 100],
                        topology.mesh().positions[fastest]
                    );
                    ms = 0.0;
                }
            }
            eprintln!("{line}");
        }
    }
}

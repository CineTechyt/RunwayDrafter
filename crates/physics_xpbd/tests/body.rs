//! The demo top on the parametric avatar: arrangement, sewing, collision
//! with the body's distance field, on both backends.

use std::sync::Arc;

use avatar::{BodyMesh, BodyModel, BodyParams, Measure, PoseParams};
use cad_core::PatternDocument;
use physics_xpbd::XpbdSolver;
use topology_sync::{ClothSolver, DistanceField, GarmentTopology, SolverBackend, SolverContext, SolverSettings, TopologySync};

const DT: f32 = 1.0 / 30.0;

fn model() -> BodyModel {
    BodyModel::load(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/avatar/body.bin")).expect("run tools/bake_avatar.py")
}

/// The body for `params` in the default A-pose.
fn posed(model: &BodyModel, params: &BodyParams) -> (BodyMesh, avatar::Measurements) {
    let rest = model.build(params);
    let measurements = avatar::measure(model, &rest);
    (model.pose(&rest, &PoseParams::default().to_pose(model, &rest)), measurements)
}

fn dress_with(model: &BodyModel, params: &BodyParams, closing: f32) -> (PatternDocument, GarmentTopology, Arc<DistanceField>, BodyMesh) {
    let (body, measurements) = posed(model, params);
    let mut document = PatternDocument::demo_top();
    avatar::arrange_around(&mut document, &body, &measurements);
    let mut topology = GarmentTopology::default();
    topology.rebuild(&document);
    let field = Arc::new(avatar::distance_field(&body, 0.008, closing));
    (document, topology, field, body)
}

fn dress(model: &BodyModel, params: &BodyParams) -> (PatternDocument, GarmentTopology, Arc<DistanceField>) {
    let (document, topology, field, _) = dress_with(model, params, 0.012);
    (document, topology, field)
}

/// Skin vertices that show through the garment: the cloth is found a short
/// way *behind* them. `inset` moves the skin inwards first, as the
/// renderer does.
fn poking_vertices(body: &BodyMesh, garment: &topology_sync::GarmentMesh, inset: f32) -> usize {
    let skin: std::collections::BTreeSet<u32> = body.skin().iter().flatten().copied().collect();
    let (low, high) = garment.positions.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p.y), hi.max(p.y)));
    skin.iter()
        .map(|&i| (body.positions[i as usize], body.normals[i as usize]))
        // Only the middle of the torso, well inside the garment's outline:
        // at necklines and armholes skin is supposed to be visible.
        .filter(|(p, _)| p.x.abs() < 0.14 && p.y > low + 0.05 && p.y < high - 0.22)
        .filter(|(p, n)| garment.raycast(*p - *n * inset, -*n).is_some_and(|hit| hit.distance < 0.02))
        .count()
}

#[test]
fn no_body_detail_pokes_through_the_cloth() {
    let model = model();
    for params in [BodyParams::female(), BodyParams::male()] {
        let (document, mut topology, field, body) = dress_with(&model, &params, 0.012);
        let settings = SolverSettings::default();
        let ctx = SolverContext { document: &document, settings: &settings, colliders: &[], body: Some(&field), body_mesh: None, ground: 0.0, dt: DT };
        let mut solver = XpbdSolver::new();
        solver.rebind(topology.mesh(), &ctx);
        for _ in 0..90 {
            solver.step(topology.mesh_mut(), &ctx);
        }
        // Measured on the true skin, without the renderer's inset.
        let poking = poking_vertices(&body, topology.mesh(), 0.0);
        println!("gender {:.0}: {poking} skin vertices show through, {:.2} ms/step", params.gender, solver.stats().step_ms);
        assert_eq!(poking, 0);
    }
}

#[test]
fn arranged_seams_clear_every_body() {
    let model = model();
    let mut heavy = BodyParams::male();
    heavy.weight = 0.9;
    let mut tall = BodyParams::female();
    avatar::fit(&model, &mut tall, Measure::Height, 1.85);
    for (label, params) in [("woman", BodyParams::female()), ("man", BodyParams::male()), ("heavy man", heavy), ("tall woman", tall)] {
        // Sewing lines are checked against the true body, not the closed one.
        let (_, topology, field, _) = dress_with(&model, &params, 0.0);
        let mesh = topology.mesh();
        // No garment vertex starts inside the body...
        for p in &mesh.positions {
            assert!(field.sample(*p).is_none_or(|(d, _)| d > 0.0), "{label}: a piece starts inside the body at {p}");
        }
        // ...and no sewing line passes deep through it. On heavy bodies
        // the top of a side seam can graze the inside of the upper arm,
        // which the collision resolves (see the drape test below); a line
        // through the torso or a shoulder would not be recoverable.
        for stitch in &mesh.stitches {
            let (a, b) = (mesh.positions[stitch.a as usize], mesh.positions[stitch.b as usize]);
            for k in 0..=20 {
                let p = a.lerp(b, k as f32 / 20.0);
                assert!(field.sample(p).is_none_or(|(d, _)| d > -0.015), "{label}: a seam would be sewn through the body at {p}");
            }
        }
    }
}

#[test]
fn the_top_drapes_on_the_body_without_penetrating() {
    let model = model();
    let mut heavy = BodyParams::male();
    heavy.weight = 0.9;
    for params in [BodyParams::female(), BodyParams::male(), heavy] {
        let (document, mut topology, field) = dress(&model, &params);
        let settings = SolverSettings::default();
        let ctx = SolverContext { document: &document, settings: &settings, colliders: &[], body: Some(&field), body_mesh: None, ground: 0.0, dt: DT };
        let mut solver = XpbdSolver::new();
        solver.rebind(topology.mesh(), &ctx);
        for _ in 0..90 {
            assert!(solver.step(topology.mesh_mut(), &ctx));
        }
        let (_, over_ten_percent, widest_gap) = solver.diagnostics();
        println!("weight {:.1} gender {:.0}: gap {:.2} mm, over 10 %: {:.2} %", params.weight, params.gender, widest_gap * 1000.0, over_ten_percent * 100.0);
        assert!(widest_gap < 1e-3, "widest stitch gap {widest_gap} m");
        assert!(over_ten_percent < 0.02, "{:.1} % of the fabric is strained beyond 10 %", over_ten_percent * 100.0);

        let mesh = topology.mesh();
        let deepest = mesh.positions.iter().filter_map(|p| field.sample(*p)).map(|(d, _)| d).fold(f32::MAX, f32::min);
        assert!(deepest > settings.collision_margin - 0.002, "a particle is {deepest} m from the skin");
        // It hangs from the shoulders: the hem is well above the knees.
        let hem = mesh.positions.iter().map(|p| p.y).fold(f32::MAX, f32::min);
        assert!(hem > 0.35 * posed(&model, &params).0.height(), "hem at {hem}");
    }
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_and_cpu_agree_on_body_collision() {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::PRIMARY;
    let instance = wgpu::Instance::new(descriptor);
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) else {
        eprintln!("skipped: no GPU adapter");
        return;
    };
    let limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
    let request = wgpu::DeviceDescriptor { required_limits: limits, ..Default::default() };
    let Ok((device, queue)) = pollster::block_on(adapter.request_device(&request)) else {
        eprintln!("skipped: adapter cannot run compute");
        return;
    };

    let model = model();
    let (document, mut on_cpu, field) = dress(&model, &BodyParams::female());
    let (_, mut on_gpu, _) = dress(&model, &BodyParams::female());
    let cpu_settings = SolverSettings { backend: SolverBackend::Cpu, ..SolverSettings::default() };
    let gpu_settings = SolverSettings { backend: SolverBackend::Gpu, ..cpu_settings };
    let cpu_ctx = SolverContext { document: &document, settings: &cpu_settings, colliders: &[], body: Some(&field), body_mesh: None, ground: 0.0, dt: DT };
    let gpu_ctx = SolverContext { settings: &gpu_settings, ..SolverContext { document: &document, settings: &cpu_settings, colliders: &[], body: Some(&field), body_mesh: None, ground: 0.0, dt: DT } };
    let mut cpu = XpbdSolver::new();
    let mut gpu = XpbdSolver::with_gpu(device, queue);
    cpu.rebind(on_cpu.mesh(), &cpu_ctx);
    gpu.rebind(on_gpu.mesh(), &gpu_ctx);

    // Long enough for the panels to reach the body and press against it.
    let mut worst = 0.0f32;
    for _ in 0..16 {
        assert!(cpu.step(on_cpu.mesh_mut(), &cpu_ctx));
        assert!(gpu.step(on_gpu.mesh_mut(), &gpu_ctx));
        let difference = on_cpu.mesh().positions.iter().zip(&on_gpu.mesh().positions).map(|(a, b)| a.distance(*b)).fold(0.0, f32::max);
        worst = worst.max(difference);
    }
    assert_eq!(gpu.stats().backend, "GPU compute");
    let touching = on_gpu.mesh().positions.iter().filter_map(|p| field.sample(*p)).filter(|(d, _)| *d < gpu_settings.collision_margin + 1e-3).count();
    assert!(touching > 20, "only {touching} particles are in contact; the comparison would not exercise collisions");
    assert!(worst < 2e-3, "backends diverged by {worst} m");
}

#[test]
fn worn_cloth_comes_to_rest_and_follows_a_pose_change() {
    let model = model();
    let params = BodyParams::female();
    let rest = model.build(&params);
    let measurements = avatar::measure(&model, &rest);
    let pose = |angles: PoseParams| model.pose_with_transforms(&rest, &angles.to_pose(&model, &rest));
    let (body, transforms) = pose(PoseParams::default());

    let mut document = PatternDocument::demo_top();
    avatar::arrange_around(&mut document, &body, &measurements);
    let mut topology = GarmentTopology::default();
    topology.rebuild(&document);
    let field = Arc::new(avatar::distance_field(&body, 0.008, 0.01));
    let settings = SolverSettings::default();
    let ctx = SolverContext { document: &document, settings: &settings, colliders: &[], body: Some(&field), body_mesh: None, ground: 0.0, dt: DT };
    let mut solver = XpbdSolver::new();
    solver.rebind(topology.mesh(), &ctx);
    for _ in 0..150 {
        solver.step(topology.mesh_mut(), &ctx);
    }
    // Draped cloth lies still instead of buzzing on the skin: measured
    // from frame to frame, which is what shows (a sub-step velocity also
    // sees motion that cancels within the frame).
    let before = topology.mesh().positions.clone();
    solver.step(topology.mesh_mut(), &ctx);
    let fastest = topology.mesh().positions.iter().zip(&before).map(|(a, b)| a.distance(*b) / DT).fold(0.0, f32::max);
    assert!(fastest < 0.05, "cloth still moves at {fastest} m/s after 5 s");

    // Raise the arms to a T-pose with the garment on.
    let (raised, raised_transforms) = pose(PoseParams { arm_angle: Some(90.0), ..PoseParams::default() });
    avatar::carry(
        &mut topology.mesh_mut().positions,
        avatar::PosedBody { rest: &rest, posed: &body, transforms: &transforms },
        avatar::PosedBody { rest: &rest, posed: &raised, transforms: &raised_transforms },
    );
    solver.set_positions(topology.mesh());
    let raised_field = Arc::new(avatar::distance_field(&raised, 0.008, 0.01));
    let ctx = SolverContext { body: Some(&raised_field), ..ctx };
    for _ in 0..60 {
        solver.step(topology.mesh_mut(), &ctx);
    }
    let mesh = topology.mesh();
    assert!(mesh.positions.iter().all(|p| p.is_finite()));
    let (_, over_ten_percent, widest_gap) = solver.diagnostics();
    assert!(widest_gap < 2e-3 && over_ten_percent < 0.03, "after the pose change: gap {widest_gap} m, {:.1} % strained", over_ten_percent * 100.0);
    // Still worn, and outside the new body.
    let deepest = mesh.positions.iter().filter_map(|p| raised_field.sample(*p)).map(|(d, _)| d).fold(f32::MAX, f32::min);
    assert!(deepest > settings.collision_margin - 0.003, "a particle is {deepest} m from the skin");
    let top = mesh.positions.iter().map(|p| p.y).fold(f32::MIN, f32::max);
    assert!(top > 0.78 * raised.height(), "the garment slipped off: top at {top}");
}

//! End-to-end checks of the solver on the demo garment.

use cad_core::PatternDocument;
use cad_core::glam::{Quat, Vec3};
use physics_xpbd::XpbdSolver;
use physics_xpbd::model::{INVALID, ROW_ACTIVE, ROW_SEAM};
use topology_sync::{
    ClothSolver, Collider, GarmentTopology, SolverBackend, SolverContext, SolverSettings, TopologySync,
};

const DT: f32 = 1.0 / 30.0;

/// The torso of the application's mannequin, for the top to hang on.
fn torso() -> Vec<Collider> {
    let part = |center: [f32; 3], radii: [f32; 3]| Collider { center: center.into(), radii: radii.into(), rotation: Quat::IDENTITY };
    vec![
        part([0.0, 1.47, 0.0], [0.050, 0.075, 0.050]),
        part([0.0, 1.27, 0.0], [0.170, 0.185, 0.115]),
        part([0.0, 1.07, 0.0], [0.145, 0.150, 0.105]),
        part([0.0, 0.93, 0.0], [0.168, 0.130, 0.118]),
        part([-0.19, 1.345, 0.0], [0.066, 0.055, 0.060]),
        part([0.19, 1.345, 0.0], [0.066, 0.055, 0.060]),
    ]
}

fn setup() -> (PatternDocument, GarmentTopology) {
    let document = PatternDocument::demo_top();
    let mut topology = GarmentTopology::default();
    topology.rebuild(&document);
    (document, topology)
}

fn simulate(solver: &mut XpbdSolver, topology: &mut GarmentTopology, document: &PatternDocument, settings: &SolverSettings, colliders: &[Collider], steps: u32) {
    let ctx = SolverContext { document, settings, colliders, body: None, body_mesh: None, ground: 0.0, dt: DT };
    for _ in 0..steps {
        assert!(solver.step(topology.mesh_mut(), &ctx));
    }
}

/// Mean stretch along the warp over all triangles.
fn mean_warp_strain(solver: &XpbdSolver) -> f32 {
    let positions = &solver.state().positions;
    let (mut sum, mut count) = (0.0f32, 0u32);
    for element in &solver.model().elements {
        let row = &element.rows[1]; // weft, warp, bias, bias
        if element.particles[3] != INVALID || row.flags & ROW_ACTIVE == 0 || row.flags & ROW_SEAM != 0 {
            continue;
        }
        let stretched = (0..3).fold(cad_core::glam::Vec3A::ZERO, |acc, k| acc + positions[element.particles[k] as usize] * row.weights[k]);
        sum += stretched.length() - 1.0;
        count += 1;
    }
    sum / count.max(1) as f32
}

#[test]
fn free_fall_matches_gravity_and_keeps_the_cloth_rigid() {
    // A single unsewn square with no drag and nothing to hit.
    let mut document = PatternDocument::new();
    let fabric = document.add_fabric("Test", [0.5; 3]);
    let corners = [[0.0, 0.0], [300.0, 0.0], [300.0, 300.0], [0.0, 300.0]];
    document.add_piece("Square", cad_core::Outline::polygon(corners.map(cad_core::glam::Vec2::from)), fabric);
    let mut topology = GarmentTopology::default();
    topology.rebuild(&document);

    let settings = SolverSettings { damping: 0.0, seam_closure_time: 0.0, ..SolverSettings::default() };
    let start: Vec<Vec3> = topology.mesh().positions.clone();
    let ctx = SolverContext { document: &document, settings: &settings, colliders: &[], body: None, body_mesh: None, ground: -100.0, dt: DT };
    let mut solver = XpbdSolver::new();
    solver.rebind(topology.mesh(), &ctx);
    for _ in 0..15 {
        assert!(solver.step(topology.mesh_mut(), &ctx));
    }

    // Symplectic Euler over n sub-steps of h falls g·h²·n(n+1)/2, which is
    // within a centimetre of the analytic ½·g·t² here.
    let n = (15 * settings.substeps) as f32;
    let h = DT / settings.substeps as f32;
    let expected = 9.81 * h * h * n * (n + 1.0) / 2.0;
    assert!((expected - 0.5 * 9.81 * (15.0 * DT).powi(2)).abs() < 0.01);
    for (now, before) in topology.mesh().positions.iter().zip(&start) {
        let fallen = before.y - now.y;
        assert!((fallen - expected).abs() < 2e-3, "fell {fallen}, expected {expected}");
        assert!((now.x - before.x).abs() < 1e-3 && (now.z - before.z).abs() < 1e-3);
    }
    let (strain, _, _) = solver.diagnostics();
    assert!(strain < 1e-3, "free fall strained the cloth by {strain}");
}

#[test]
fn the_top_is_sewn_and_hangs_on_the_torso() {
    let (document, mut topology) = setup();
    let settings = SolverSettings::default();
    let colliders = torso();
    let mut solver = XpbdSolver::new();
    solver.rebind(topology.mesh(), &SolverContext { document: &document, settings: &settings, colliders: &colliders, body: None, body_mesh: None, ground: 0.0, dt: DT });
    assert_eq!(solver.stats().seam_gap, 1.0);

    simulate(&mut solver, &mut topology, &document, &settings, &colliders, 90); // 3 s
    let mesh = topology.mesh();
    assert!(mesh.positions.iter().all(|p| p.is_finite()), "the simulation blew up");

    // Seams are closed.
    assert_eq!(solver.stats().seam_gap, 0.0);
    let widest = mesh.stitches.iter().map(|s| mesh.positions[s.a as usize].distance(mesh.positions[s.b as usize])).fold(0.0, f32::max);
    assert!(widest < 1e-3, "widest stitch gap is {widest} m");

    // The garment stayed on the body: shoulders up, hem above the hips.
    let top = mesh.positions.iter().map(|p| p.y).fold(f32::MIN, f32::max);
    let bottom = mesh.positions.iter().map(|p| p.y).fold(f32::MAX, f32::min);
    assert!((1.40..1.50).contains(&top), "top of the garment is at {top}");
    assert!((0.70..0.85).contains(&bottom), "hem is at {bottom}");

    // Nothing is inside the torso (margin minus a small solver tolerance).
    for p in &mesh.positions {
        for c in &colliders {
            let q = (*p - c.center) / (c.radii + Vec3::splat(settings.collision_margin - 1.5e-3));
            assert!(q.length() >= 1.0, "{p} penetrates the collider at {}", c.center);
        }
    }
    // It was not stretched out of shape: only the few triangles that carry
    // the garment on the shoulders are visibly strained.
    let (_, over_ten_percent, _) = solver.diagnostics();
    assert!(over_ten_percent < 0.02, "{:.1} % of the fabric is strained beyond 10 %", over_ten_percent * 100.0);
}

#[test]
fn a_stiffer_warp_sags_less_under_its_own_weight() {
    let run = |stiffness: f32| {
        let (mut document, mut topology) = setup();
        for id in document.fabrics().iter().map(|f| f.id).collect::<Vec<_>>() {
            let physics = &mut document.fabric_mut(id).unwrap().physics;
            (physics.stretch_weft, physics.stretch_warp, physics.shear) = (stiffness, stiffness, stiffness);
        }
        let settings = SolverSettings::default();
        let colliders = torso();
        let mut solver = XpbdSolver::new();
        solver.rebind(topology.mesh(), &SolverContext { document: &document, settings: &settings, colliders: &colliders, body: None, body_mesh: None, ground: 0.0, dt: DT });
        simulate(&mut solver, &mut topology, &document, &settings, &colliders, 90);
        mean_warp_strain(&solver)
    };
    let (soft, stiff) = (run(0.0), run(1.0));
    assert!(soft > 0.0 && stiff < 0.7 * soft, "stiff {stiff} should sag clearly less than soft {soft}");
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_and_cpu_backends_agree() {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::PRIMARY;
    let instance = wgpu::Instance::new(descriptor);
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) else {
        eprintln!("skipped: no GPU adapter");
        return;
    };
    let request = wgpu::DeviceDescriptor { required_limits: wgpu::Limits::downlevel_defaults(), ..Default::default() };
    let Ok((device, queue)) = pollster::block_on(adapter.request_device(&request)) else {
        eprintln!("skipped: adapter cannot run compute");
        return;
    };

    let colliders = torso();
    let (document, mut on_cpu) = setup();
    let (_, mut on_gpu) = setup();
    let cpu_settings = SolverSettings::default();
    let gpu_settings = SolverSettings { backend: SolverBackend::Gpu, ..cpu_settings };
    let cpu_ctx = SolverContext { document: &document, settings: &cpu_settings, colliders: &colliders, body: None, body_mesh: None, ground: 0.0, dt: DT };
    let gpu_ctx = SolverContext { document: &document, settings: &gpu_settings, colliders: &colliders, body: None, body_mesh: None, ground: 0.0, dt: DT };
    let mut cpu = XpbdSolver::new();
    let mut gpu = XpbdSolver::with_gpu(device, queue);
    cpu.rebind(on_cpu.mesh(), &cpu_ctx);
    gpu.rebind(on_gpu.mesh(), &gpu_ctx);

    // Same algorithm and the same order of operations per particle; only
    // float rounding differs. Cloth in contact is chaotic, so rounding
    // differences grow over time: compare while they are still small, and
    // across the moment the panels first touch the body.
    let mut worst = 0.0f32;
    for step in 0..12 {
        assert!(cpu.step(on_cpu.mesh_mut(), &cpu_ctx));
        assert!(gpu.step(on_gpu.mesh_mut(), &gpu_ctx));
        let difference = on_cpu.mesh().positions.iter().zip(&on_gpu.mesh().positions).map(|(a, b)| a.distance(*b)).fold(0.0, f32::max);
        println!("step {step}: backends differ by {difference:e} m");
        worst = worst.max(difference);
    }
    assert_eq!(cpu.stats().backend, "CPU (rayon)");
    assert_eq!(gpu.stats().backend, "GPU compute");
    assert!(worst < 1e-3, "backends diverged by {worst} m within 12 steps");

    // Switching backends mid-flight carries the motion over.
    assert!(gpu.step(on_gpu.mesh_mut(), &cpu_ctx));
    assert_eq!(gpu.stats().backend, "CPU (rayon)");
    assert!(cpu.step(on_cpu.mesh_mut(), &cpu_ctx));
    let difference = on_cpu.mesh().positions.iter().zip(&on_gpu.mesh().positions).map(|(a, b)| a.distance(*b)).fold(0.0, f32::max);
    assert!(difference < 2e-3, "state was lost when switching backends: {difference} m");
}

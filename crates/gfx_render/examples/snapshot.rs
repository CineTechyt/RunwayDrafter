//! Renders both viewports without a window and writes them as PPM images.
//! Useful as a smoke test on machines or CI runners without a display:
//!
//! ```sh
//! cargo run -p gfx_render --example snapshot -- out_dir [seconds] [yaw] [cpu|gpu] [mm] [female|male]
//! ```
//!
//! With `seconds` > 0 the garment is draped with the XPBD solver first.

use std::path::{Path, PathBuf};

use cad_core::PatternDocument;
use avatar::{BodyModel, BodyParams};
use gfx_render::{
    Camera2D, CanvasBatch, CanvasOverlay, CanvasRenderer, GpuContext, MeshData, OrbitCamera, RenderTarget,
    SceneRenderer, SurfaceMaterial, ViewportRenderer, canvas2d, preferred_backends,
};
use glam::Vec3;
use physics_xpbd::XpbdSolver;
use topology_sync::{
    ClothSolver, Collider, GarmentTopology, MeshingParams, SolverBackend, SolverContext, SolverSettings, TopologySync,
};

const SIZE: [u32; 2] = [960, 720];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let out_dir = PathBuf::from(args.get(1).cloned().unwrap_or_else(|| ".".into()));
    let seconds: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let yaw: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.5);
    let use_gpu = args.get(4).is_some_and(|s| s == "gpu");
    let particle_distance: f32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(20.0);
    let mut params = if args.get(6).is_some_and(|s| s == "male") { BodyParams::male() } else { BodyParams::female() };
    let body_model = BodyModel::load(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/avatar/body.bin"))?;
    // Extra knobs for experiments: POSE=<preset name>, CLOSING=<mm>,
    // INSET=<mm>, ZOOM=<camera distance>, LOOK=<camera target height>.
    let env = |name: &str, default: f32| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
    let pose_name = std::env::var("POSE").unwrap_or_else(|_| "A-pose".into());
    let mut pose_params = avatar::PoseParams::PRESETS.iter().find(|(name, _)| *name == pose_name).map_or_else(Default::default, |(_, p)| *p);
    pose_params.elbow_bend = env("ELBOW", pose_params.elbow_bend);
    pose_params.arm_forward = env("ARM_FORWARD", pose_params.arm_forward);
    pose_params.knee_bend = env("KNEE", pose_params.knee_bend);
    // TARGETS="Height=1.72,Bust=0.98": tape measurements in metres, by `Measure` name.
    if let Ok(list) = std::env::var("TARGETS") {
        let targets: Vec<(avatar::Measure, f32)> = list
            .split(',')
            .filter_map(|item| {
                let (name, value) = item.split_once('=')?;
                let measure = avatar::Measure::ALL.into_iter().find(|m| format!("{m:?}") == name.trim())?;
                Some((measure, value.trim().parse().ok()?))
            })
            .collect();
        let report = avatar::fit_all(&body_model, &mut params, &targets);
        println!("fitted {} measurements, worst error {:.2} mm", targets.len(), report.worst_error * 1000.0);
    }
    let rest = body_model.build(&params);
    let measurements = avatar::measure(&body_model, &rest);
    let (mut body, transforms) = body_model.pose_with_transforms(&rest, &pose_params.to_pose(&body_model, &rest));
    let mut field = std::sync::Arc::new(avatar::distance_field(&body, 0.008, env("CLOSING", 0.5 * particle_distance) / 1000.0));
    let mut collider = std::sync::Arc::new(body.collider());
    std::fs::create_dir_all(&out_dir)?;
    // AVATAR_OBJ=<file> writes the body as OBJ in centimetres, Z up: a
    // file to try the avatar importer on.
    if let Ok(path) = std::env::var("AVATAR_OBJ") {
        let mut obj = String::new();
        for p in &body.positions {
            obj.push_str(&format!("v {} {} {}\n", p.x * 100.0, -p.z * 100.0, p.y * 100.0));
        }
        for t in body.skin() {
            obj.push_str(&format!("f {} {} {}\n", t[0] + 1, t[1] + 1, t[2] + 1));
        }
        std::fs::write(&path, obj)?;
        println!("wrote {path}");
    }

    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = preferred_backends();
    let instance = wgpu::Instance::new(descriptor.with_env());
    let (gpu, adapter) = pollster::block_on(GpuContext::request(&instance, None))?;
    println!("rendering on {}", adapter.get_info().name);

    let mut document = PatternDocument::demo_top();
    // LAYERS=2 wears a second, slightly larger copy over the first.
    for _ in 1..env("LAYERS", 1.0) as u32 {
        let copies = document.add_layer_copy(env("EASE", 1.06), glam::Vec2::new(0.0, 800.0));
        // A different colour, to tell the layers apart.
        let blue = document.add_fabric("Outer", [0.15, 0.3, 0.7]);
        for id in copies {
            document.piece_mut(id).unwrap().fabric = blue;
        }
    }
    avatar::arrange_around(&mut document, &body, &measurements);
    let mut topology = GarmentTopology::new(MeshingParams { particle_distance, ..Default::default() });
    topology.rebuild(&document);

    if seconds > 0.0 {
        let colliders: Vec<Collider> = Vec::new();
        let backend = if use_gpu { SolverBackend::Gpu } else { SolverBackend::Cpu };
        // FRICTION=<coefficient> and SUBSTEPS=<count> override the defaults.
        let defaults = SolverSettings::default();
        let settings = SolverSettings {
            backend,
            friction: env("FRICTION", defaults.friction),
            // SELF=0 and CCD=0 switch the cloth and body collision passes off.
            self_collision: env("SELF", 1.0) > 0.0,
            continuous_body_collision: env("CCD", 1.0) > 0.0,
            substeps: env("SUBSTEPS", defaults.substeps as f32) as u32,
            ..defaults
        };
        let ctx = SolverContext { document: &document, settings: &settings, colliders: &colliders, body: Some(&field), body_mesh: Some(&collider), ground: 0.0, dt: 1.0 / 30.0 };
        let mut solver = if use_gpu && gpu.supports_compute {
            XpbdSolver::with_gpu(gpu.device.clone(), gpu.queue.clone())
        } else {
            XpbdSolver::new()
        };
        solver.rebind(topology.mesh(), &ctx);
        let steps = (seconds * 30.0).round() as u32;
        let started = std::time::Instant::now();
        for _ in 0..steps {
            solver.step(topology.mesh_mut(), &ctx);
        }
        // POSE2=<preset>: change the pose with the garment on, as the app
        // does: carry the cloth with the skin, then let it settle.
        if let Ok(second) = std::env::var("POSE2") {
            let mut next = avatar::PoseParams::PRESETS.iter().find(|(name, _)| *name == second).map_or_else(Default::default, |(_, p)| *p);
            next.elbow_bend = env("ELBOW2", next.elbow_bend);
            next.arm_forward = env("ARM_FORWARD2", next.arm_forward);
            let (posed, next_transforms) = body_model.pose_with_transforms(&rest, &next.to_pose(&body_model, &rest));
            avatar::carry(
                &mut topology.mesh_mut().positions,
                avatar::PosedBody { rest: &rest, posed: &body, transforms: &transforms },
                avatar::PosedBody { rest: &rest, posed: &posed, transforms: &next_transforms },
            );
            body = posed;
            collider = std::sync::Arc::new(body.collider());
            field = std::sync::Arc::new(avatar::distance_field(&body, 0.008, env("CLOSING", 0.5 * particle_distance) / 1000.0));
            solver.set_positions(topology.mesh());
            let (strain, over, _) = solver.diagnostics();
            println!("right after carrying: max strain {:.1} %, rows over 10 %: {:.2} %", strain * 100.0, over * 100.0);
            let ctx = SolverContext { document: &document, settings: &settings, colliders: &colliders, body: Some(&field), body_mesh: Some(&collider), ground: 0.0, dt: 1.0 / 30.0 };
            for _ in 0..env("SETTLE", 60.0) as u32 {
                solver.step(topology.mesh_mut(), &ctx);
            }
        }
        topology.mesh_mut().recompute_normals();
        let (strain, over, gap) = solver.diagnostics();
        println!("max strain {:.1} %, rows over 10 %: {:.2} %, widest stitch gap {:.1} mm", strain * 100.0, over * 100.0, gap * 1000.0);
        let speeds: Vec<f32> = solver.state().velocities.iter().map(|v| v.length()).collect();
        println!("residual speed: mean {:.1} mm/s, max {:.1} mm/s", 1000.0 * speeds.iter().sum::<f32>() / speeds.len().max(1) as f32, 1000.0 * speeds.iter().cloned().fold(0.0, f32::max));
        if std::env::var("DEPTH").is_ok() {
            let m = topology.mesh();
            let exact = avatar::distance_field(&body, 0.008, 0.0);
            let mut inside: Vec<(f32, cad_core::glam::Vec3, f32)> = m.positions.iter().filter_map(|p| Some((exact.sample(*p)?.0, *p, field.sample(*p)?.0))).filter(|(d, ..)| *d < 0.0).collect();
            inside.sort_by(|a, b| a.0.total_cmp(&b.0));
            let ys: Vec<f32> = inside.iter().map(|(_, p, _)| p.y).collect();
            println!("{} cloth vertices inside the true body, y {:.2}..{:.2}; field dims {:?} voxel {:.4}", inside.len(), ys.iter().cloned().fold(9.0, f32::min), ys.iter().cloned().fold(-9.0, f32::max), field.dims, field.voxel);
            for (d, p, closed) in inside.iter().take(8) { println!("   exact {:.1} mm closed {:.1} mm at {:.3} grad {:?}", d * 1000.0, closed * 1000.0, p, field.sample(*p).map(|(_, g)| g)); }
        }
        if std::env::var("JITTER").is_ok() {
            let m = topology.mesh();
            let stitched: std::collections::HashSet<u32> = m.stitches.iter().flat_map(|s| [s.a, s.b]).collect();
            let mut order: Vec<usize> = (0..speeds.len()).collect();
            order.sort_by(|a, b| speeds[*b].total_cmp(&speeds[*a]));
            let fast: Vec<usize> = order.iter().copied().filter(|&i| speeds[i] > 0.05).collect();
            let near = |i: usize| field.sample(m.positions[i]).map_or(9.0, |(d, _)| d);
            println!("{} particles above 50 mm/s; stitched {}, boundary {}, within 8mm of body {}", fast.len(), fast.iter().filter(|i| stitched.contains(&(**i as u32))).count(), fast.iter().filter(|&&i| !matches!(m.origins[i], topology_sync::VertexOrigin::Interior { .. })).count(), fast.iter().filter(|&&i| near(i) < 0.008).count());
            for &i in order.iter().take(6) { println!("  {:.0} mm/s at {:.3} dist {:.1} mm origin {:?}", speeds[i] * 1000.0, m.positions[i], near(i) * 1000.0, m.origins[i]); }
        }
        let stats = solver.stats();
        println!("cloth contacts {}, crossings {}, body stops {} in the last step", stats.cloth_contacts, stats.cloth_crossings, stats.body_stops);
        println!(
            "simulated {seconds} s on {}: {} particles, {} batches, {:.2} ms/step",
            stats.backend,
            stats.particles,
            stats.batches,
            started.elapsed().as_secs_f32() * 1000.0 / steps.max(1) as f32
        );
    }
    let mesh = topology.mesh();

    // EXPORT=1 also writes the pattern (DXF-AAMA and DXF-ASTM) and the
    // garment as draped (OBJ + MTL, FBX) into the output directory.
    if std::env::var("EXPORT").is_ok() {
        for (name, standard) in [("pattern_aama.dxf", io_cad::DxfStandard::Aama), ("pattern_astm.dxf", io_cad::DxfStandard::Astm)] {
            let options = io_cad::DxfOptions { standard, style_name: "Demo top".into(), ..Default::default() };
            std::fs::write(out_dir.join(name), io_cad::export_pattern_dxf(&document, &options))?;
        }
        let garment = io_cad::GarmentExport { mesh, document: &document, uv_layout: io_cad::UvLayout::Fabric };
        let (obj, mtl) = io_cad::export_garment_obj(&garment, "garment.mtl");
        std::fs::write(out_dir.join("garment.obj"), obj)?;
        std::fs::write(out_dir.join("garment.mtl"), mtl)?;
        std::fs::write(out_dir.join("garment.fbx"), io_cad::export_garment_fbx(&garment, io_cad::FbxEncoding::Binary))?;
        println!("exported pattern_aama.dxf, pattern_astm.dxf, garment.obj, garment.mtl, garment.fbx");
    }

    // Viewport A.
    let mut canvas = CanvasRenderer::new(&gpu, SIZE);
    let mut camera = Camera2D::default();
    let bounds = document.pieces().iter().map(|p| p.outline.bounds()).reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)));
    if let Some((min, max)) = bounds {
        camera.fit(min, max, canvas.viewport());
    }
    let overlay =
        CanvasOverlay { selected_piece: document.pieces().first().map(|p| p.id), show_mesh: true, ..Default::default() };
    let mut batch = CanvasBatch::default();
    canvas2d::scene::build(&mut batch, &document, Some(mesh), &camera, canvas.viewport(), &overlay);
    canvas.prepare(&gpu, &batch, &camera);

    // Viewport B.
    let mut scene = SceneRenderer::new(&gpu, SIZE);
    let avatar_mesh = MeshData {
        vertices: body.positions.iter().zip(&body.normals).map(|(p, n)| gfx_render::MeshVertex::new(*p - *n * (env("INSET", 2.5) / 1000.0), *n, glam::Vec2::ZERO)).collect(),
        indices: body.triangles.iter().flatten().copied().collect(),
    };
    scene.set_avatar(&gpu, &avatar_mesh, SurfaceMaterial { base_color: [0.62, 0.45, 0.36], roughness: 0.65, ..Default::default() });
    let mut floor = MeshData::default();
    floor.add_ground(Vec3::ZERO, 4.0);
    scene.add_static(&gpu, &floor, SurfaceMaterial { base_color: [0.10, 0.105, 0.115], roughness: 0.9, ..Default::default() });
    // OVERLAY=strain|fit shows a technical view; RANGE=<percent or mm>.
    let mut values = Vec::new();
    match std::env::var("OVERLAY").as_deref() {
        Ok("strain") => {
            topology_sync::strain_map(mesh, &mut values);
            scene.overlay = gfx_render::SurfaceOverlay::Strain { limit: env("RANGE", 10.0) / 100.0 };
        }
        Ok("fit") => {
            let loose = env("RANGE", 40.0) / 1000.0;
            topology_sync::clearance_map(mesh, &collider, loose * 2.0, &mut values);
            scene.overlay = gfx_render::SurfaceOverlay::Fit { tight: 0.004, loose };
        }
        _ => {}
    }
    if !values.is_empty() {
        let mut sorted = values.clone();
        sorted.sort_by(f32::total_cmp);
        println!("overlay values: min {:.4}, median {:.4}, max {:.4}", sorted[0], sorted[sorted.len() / 2], sorted[sorted.len() - 1]);
    }
    scene.sync_garment(&gpu, mesh, &values, |_| 0.001);
    scene.prepare(&gpu, &OrbitCamera { yaw, distance: env("ZOOM", 2.2), target: Vec3::new(0.0, env("LOOK", 1.15), 0.0), ..Default::default() }, |piece| {
        let fabric = document.piece(piece).and_then(|p| document.fabric(p.fabric));
        fabric.map_or_else(SurfaceMaterial::default, |f| SurfaceMaterial {
            base_color: f.base_color,
            roughness: f.roughness,
            metallic: f.metallic,
            weave_strength: f.weave_strength,
            ..SurfaceMaterial::default()
        })
    });

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    canvas.render(&gpu, &mut encoder);
    scene.render(&gpu, &mut encoder);
    gpu.queue.submit([encoder.finish()]);

    write_ppm(&out_dir.join("canvas2d.ppm"), &gpu, canvas.target())?;
    write_ppm(&out_dir.join("scene3d.ppm"), &gpu, scene.target())?;
    Ok(())
}

fn write_ppm(path: &Path, gpu: &GpuContext, target: &RenderTarget) -> std::io::Result<()> {
    let [width, height] = target.size();
    let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
    bytes.extend(target.read_pixels(gpu).chunks_exact(4).flat_map(|rgba| &rgba[..3]));
    std::fs::write(path, bytes)?;
    println!("wrote {}", path.display());
    Ok(())
}

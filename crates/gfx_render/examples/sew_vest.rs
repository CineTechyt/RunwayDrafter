//! The whole pattern-to-garment workflow on the S-5017 vest PDF, headless,
//! with the same document operations the editor's toolbar and sewing tool
//! use: import, delete the facing, turn the pieces upright, add mirrored
//! copies, place them around the body, sew, drape, render.
//!
//! ```sh
//! cargo run -p gfx_render --example sew_vest -- S-5017_Vest.pdf out_dir [seconds]
//! ```
//!
//! Lines are found by their length, so the steps do not depend on how the
//! importer numbers them.

use std::path::PathBuf;
use std::sync::Arc;

use avatar::{BodyModel, BodyParams, PoseParams};
use cad_core::{EdgeSpan, PatternDocument, PieceId, PieceTransform};
use gfx_render::{
    Camera2D, CanvasBatch, CanvasOverlay, CanvasRenderer, GpuContext, MeshData, OrbitCamera,
    SceneRenderer, SurfaceMaterial, ViewportRenderer, canvas2d, preferred_backends,
};
use glam::{Vec2, Vec3};
use physics_xpbd::XpbdSolver;
use topology_sync::{
    ClothSolver, GarmentTopology, MeshingParams, SolverBackend, SolverContext, SolverSettings,
    TopologySync,
};

const SIZE: [u32; 2] = [900, 1000];

fn piece_named(document: &PatternDocument, name: &str) -> PieceId {
    document
        .pieces()
        .iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("no piece {name}"))
        .id
}

/// The line of `piece` whose length is closest to `length` (mm).
fn line(document: &PatternDocument, piece: PieceId, length: f32) -> EdgeSpan {
    let outline = &document.piece(piece).unwrap().outline;
    let best = outline
        .lines()
        .into_iter()
        .min_by(|a, b| {
            (outline.span_length(*a) - length)
                .abs()
                .total_cmp(&(outline.span_length(*b) - length).abs())
        })
        .unwrap();
    let found = outline.span_length(best);
    assert!(
        (found - length).abs() < 4.0,
        "no line of about {length} mm (closest {found:.0} mm)"
    );
    best
}

/// Two neighbouring lines as one side (what Shift+click does).
fn lines(document: &PatternDocument, piece: PieceId, lengths: &[f32]) -> EdgeSpan {
    let n = document.piece(piece).unwrap().outline.edge_count();
    lengths
        .iter()
        .map(|&l| line(document, piece, l))
        .reduce(|a, b| a.union(&b, n).expect("the lines are neighbours"))
        .unwrap()
}

/// Sews two sides, turned so that the ends lying closer together around
/// the body meet (what clicking matching ends does in the editor).
fn sew(document: &mut PatternDocument, a: (PieceId, EdgeSpan), b: (PieceId, EdgeSpan)) {
    let ends = |(piece, span): (PieceId, EdgeSpan)| {
        let p = document.piece(piece).unwrap();
        let points = p.outline.span_points(span, 1.0);
        (
            p.arrangement.place(points[0]),
            p.arrangement.place(points[points.len() - 1]),
        )
    };
    let ((a0, a1), (b0, b1)) = (ends(a), ends(b));
    let straight = a0.distance(b0) + a1.distance(b1);
    let crossed = a0.distance(b1) + a1.distance(b0);
    document.add_seam_spans(a, b, crossed < straight);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let pdf = args
        .get(1)
        .ok_or("usage: sew_vest <S-5017_Vest.pdf> [out_dir] [seconds]")?;
    let out_dir = PathBuf::from(args.get(2).cloned().unwrap_or_else(|| ".".into()));
    let seconds: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4.0);
    std::fs::create_dir_all(&out_dir)?;

    // --- 1. import, names from the drawn labels ------------------------------
    let mut pattern =
        io_cad::import_pattern_pdf(&std::fs::read(pdf)?, &io_cad::PdfOptions::default())?;
    let reader = io_cad::drawn_text::TesseractCli::find()
        .ok_or("needs tesseract (or CLO360_TESSERACT) to read the piece names")?;
    io_cad::drawn_text::name_pieces(&mut pattern, &reader);
    let mut document = pattern.to_document();
    println!(
        "imported: {:?}",
        document
            .pieces()
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>()
    );

    // --- 2. the facing is an inner piece: not part of the drape ------------
    document.remove_piece(piece_named(&document, "BACK NECKLINE FACING"));

    // --- 3. upright: the layout has the tops of all pieces on the left ------
    let (cf, sf, sb, cb) = (
        piece_named(&document, "CENTER FRONT PART"),
        piece_named(&document, "SIDE FRONT PART"),
        piece_named(&document, "SIDE BACK PART"),
        piece_named(&document, "CENTER BACK PART"),
    );
    for id in [cf, sf, sb, cb] {
        document.transform_piece(id, PieceTransform::RotateCw);
    }
    // As turned, they are the avatar's right half; mirrored copies make
    // the left half.
    for (id, angle) in [(cf, -22.0), (sf, -62.0), (sb, -118.0), (cb, -158.0)] {
        document.piece_mut(id).unwrap().placement.angle = angle;
    }
    let mirror = |document: &mut PatternDocument, id| document.duplicate_piece(id, true).unwrap();
    let (cf_l, sf_l, sb_l, cb_l) = (
        mirror(&mut document, cf),
        mirror(&mut document, sf),
        mirror(&mut document, sb),
        mirror(&mut document, cb),
    );

    document.tidy_layout(40.0);

    // --- 4. on the avatar ----------------------------------------------------
    let model = BodyModel::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/avatar/body.bin"
    ))?;
    let rest = model.build(&BodyParams::female());
    let measurements = avatar::measure(&model, &rest);
    let body = model.pose(&rest, &PoseParams::default().to_pose(&model, &rest));
    avatar::arrange_around(&mut document, &body, &measurements);

    // --- 5. sew ----------------------------------------------------------------
    for (cf, sf, sb, cb) in [(cf, sf, sb, cb), (cf_l, sf_l, sb_l, cb_l)] {
        let shoulder = (
            (cf, line(&document, cf, 96.0)),
            (cb, line(&document, cb, 99.0)),
        );
        let front_princess = (
            (cf, line(&document, cf, 374.0)),
            (sf, lines(&document, sf, &[265.0, 112.0])),
        );
        let side = (
            (sf, line(&document, sf, 260.0)),
            (sb, line(&document, sb, 259.0)),
        );
        let back_princess = (
            (sb, line(&document, sb, 375.0)),
            (cb, line(&document, cb, 382.0)),
        );
        for (a, b) in [shoulder, front_princess, side, back_princess] {
            sew(&mut document, a, b);
        }
    }
    let centre_back = (
        (cb, line(&document, cb, 494.0)),
        (cb_l, line(&document, cb_l, 494.0)),
    );
    sew(&mut document, centre_back.0, centre_back.1);
    println!(
        "{} pieces, {} seams",
        document.pieces().len(),
        document.seams().len()
    );
    for seam in document.seams() {
        let name = |id| document.piece(id).unwrap().name.clone();
        let length = |id, span| document.piece(id).unwrap().outline.span_length(span);
        println!(
            "  {} ({:.0} mm) ↔ {} ({:.0} mm){}",
            name(seam.a.piece),
            length(seam.a.piece, seam.a_span()),
            name(seam.b.piece),
            length(seam.b.piece, seam.b_span()),
            if seam.reversed { ", reversed" } else { "" }
        );
    }

    // --- 6. drape -----------------------------------------------------------------
    let field = Arc::new(avatar::distance_field(&body, 0.008, 0.01));
    let collider = Arc::new(body.collider());
    let mut topology = GarmentTopology::new(MeshingParams {
        particle_distance: 20.0,
        ..Default::default()
    });
    topology.rebuild(&document);
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = preferred_backends();
    let instance = wgpu::Instance::new(descriptor.with_env());
    let (gpu, _) = pollster::block_on(GpuContext::request(&instance, None))?;
    let settings = SolverSettings {
        backend: SolverBackend::Gpu,
        ..Default::default()
    };
    let ctx = SolverContext {
        document: &document,
        settings: &settings,
        colliders: &[],
        body: Some(&field),
        body_mesh: Some(&collider),
        ground: 0.0,
        dt: 1.0 / 30.0,
    };
    let mut solver = if gpu.supports_compute {
        XpbdSolver::with_gpu(gpu.device.clone(), gpu.queue.clone())
    } else {
        XpbdSolver::new()
    };
    solver.rebind(topology.mesh(), &ctx);
    let render =
        |topology: &GarmentTopology, name: &str| -> Result<(), Box<dyn std::error::Error>> {
            let mut topology_mesh = topology.mesh().clone();
            topology_mesh.recompute_normals();
            for (yaw, side) in [
                (0.0f32, "front"),
                (std::f32::consts::PI, "back"),
                (1.2, "side"),
            ] {
                let mut scene = SceneRenderer::new(&gpu, SIZE);
                let avatar_mesh = MeshData {
                    vertices: body
                        .positions
                        .iter()
                        .zip(&body.normals)
                        .map(|(p, n)| gfx_render::MeshVertex::new(*p - *n * 0.0025, *n, Vec2::ZERO))
                        .collect(),
                    indices: body.triangles.iter().flatten().copied().collect(),
                };
                scene.set_avatar(
                    &gpu,
                    &avatar_mesh,
                    SurfaceMaterial {
                        base_color: [0.62, 0.45, 0.36],
                        roughness: 0.65,
                        ..Default::default()
                    },
                );
                scene.sync_garment(&gpu, &topology_mesh, &[], |_| 0.001);
                let camera = OrbitCamera {
                    yaw,
                    distance: 1.7,
                    target: Vec3::new(0.0, 1.15, 0.0),
                    ..Default::default()
                };
                scene.prepare(&gpu, &camera, |piece| {
                    // Left and right halves in two shades, to tell them apart.
                    let left = document
                        .piece(piece)
                        .is_some_and(|p| p.placement.angle > 0.0);
                    SurfaceMaterial {
                        base_color: if left {
                            [0.20, 0.32, 0.55]
                        } else {
                            [0.30, 0.45, 0.72]
                        },
                        roughness: 0.8,
                        weave_strength: 0.3,
                        ..Default::default()
                    }
                });
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                scene.render(&gpu, &mut encoder);
                gpu.queue.submit([encoder.finish()]);
                write_ppm(
                    &out_dir.join(format!("{name}_{side}.ppm")),
                    &gpu,
                    scene.target(),
                )?;
            }
            Ok(())
        };
    render(&topology, "arranged")?;
    let steps = (seconds * 30.0) as u32;
    for _ in 0..steps {
        solver.step(topology.mesh_mut(), &ctx);
    }
    let (strain, over, gap) = solver.diagnostics();
    let stats = solver.stats();
    println!(
        "draped {seconds} s: max strain {:.0} %, rows over 10 % {:.2} %, widest stitch gap {:.1} mm, {} cloth contacts, {:.1} ms/step",
        strain * 100.0,
        over * 100.0,
        gap * 1000.0,
        stats.cloth_contacts,
        stats.step_ms
    );
    render(&topology, "draped")?;

    // --- 7. the 2D layout, as the editor shows it ---------------------------------
    let mut canvas = CanvasRenderer::new(&gpu, [1400, 900]);
    let mut camera = Camera2D::default();
    let bounds = document
        .pieces()
        .iter()
        .map(|p| p.outline.bounds())
        .reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)))
        .unwrap();
    camera.fit(bounds.0, bounds.1, canvas.viewport());
    let mut batch = CanvasBatch::default();
    let overlay = CanvasOverlay {
        selected_piece: Some(cf),
        ..Default::default()
    };
    canvas2d::scene::build(
        &mut batch,
        &document,
        Some(topology.mesh()),
        &camera,
        canvas.viewport(),
        &overlay,
    );
    canvas.prepare(&gpu, &batch, &camera);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    canvas.render(&gpu, &mut encoder);
    gpu.queue.submit([encoder.finish()]);
    write_ppm(&out_dir.join("layout.ppm"), &gpu, canvas.target())?;
    Ok(())
}

fn write_ppm(
    path: &std::path::Path,
    gpu: &GpuContext,
    target: &gfx_render::RenderTarget,
) -> std::io::Result<()> {
    let [width, height] = target.size();
    let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
    bytes.extend(
        target
            .read_pixels(gpu)
            .chunks_exact(4)
            .flat_map(|rgba| &rgba[..3]),
    );
    std::fs::write(path, bytes)
}

//! `clo360 --screenshot out.ppm`: runs the editor without a window and
//! saves what the window would show. For checking the interface on
//! machines without a display, and in tests.
//!
//! Options (all optional):
//!
//! * `--size 1600x1000` — window size in pixels;
//! * `--pattern file.pdf` — import a pattern (as a new project), name its
//!   pieces and tidy the layout first;
//! * `--tool select|points|sew` — the active 2D tool;
//! * `--piece N` — select the N-th piece (0-based);
//! * `--frames N` — frames to run before saving (default 12);
//! * `--open file.clo360` — open a project; `--new waistcoat|top` — start
//!   from a built-in garment;
//! * `--play` — run the simulation over those frames;
//! * `--hair STYLE[,LENGTH_CM]` — the avatar's haircut (bald, buzz, short,
//!   bob, long, ponytail, bun; add `+fringe` for a fringe), `--skin R,G,B`
//!   and `--hair-colour R,G,B` in sRGB 0–255;
//! * `--camera YAW,PITCH,DISTANCE,TARGET_Y` — the 3D view, angles in degrees;
//! * `--avatar-window` — leave the avatar window open;
//! * `--man` — the male body;
//! * `--animate walk|stiff|sway|sit|yoga` with `--at SECONDS` — a frame
//!   of an animation; add `--play-animation` to play it from there over
//!   the frames; `--simulate-garment` drapes first (over `--frames`), then
//!   simulates the garment through the animation and shows that.
//! * `--solver contact|global|cpu|gpu` — the live cloth solver (default: gpu
//!   where available, else cpu). Garment animation is baked with global if
//!   that is the live solver, otherwise with contact.
//! * `--bake-seconds N` — limit a garment bake to the first N seconds; useful
//!   for inspecting a particular early animation pose.
//! * `--fabric denim|corduroy` — apply a close-up material preset for visual
//!   shader checks.

use std::path::PathBuf;

use anyhow::Context as _;
use app_ui::{AppUi, CanvasTool, UiCommand, ViewportTextures};
use gfx_render::{
    CanvasBatch, CanvasRenderer, GpuContext, MeshData, SceneRenderer, SurfaceMaterial,
    ViewportRenderer,
};
use glam::Vec3;
use physics_xpbd::XpbdSolver;
use topology_sync::{ClothSolver, SolverBackend, TopologySync};

use crate::body::Body;
use crate::session::Session;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

struct Options {
    out: PathBuf,
    size: [u32; 2],
    pattern: Option<PathBuf>,
    tool: CanvasTool,
    piece: Option<usize>,
    frames: u32,
    commands: Vec<UiCommand>,
    play: bool,
    hair: Option<(avatar::HairStyle, Option<f32>, bool)>,
    skin: Option<[f32; 3]>,
    hair_colour: Option<[f32; 3]>,
    camera: Option<[f32; 4]>,
    avatar_window: bool,
    animation: Option<(avatar::Motion, f32)>,
    /// Every `--at` time (comma separated): with `--simulate-garment`, one
    /// image each, numbered.
    times: Vec<f32>,
    bake_seconds: Option<f32>,
    solver: Option<SolverBackend>,
    fabric: Option<cad_core::FabricPattern>,
}

/// `R,G,B` in sRGB 0–255 as linear RGB.
fn srgb(text: &str) -> anyhow::Result<[f32; 3]> {
    let values: Vec<u8> = text.split(',').map(str::parse).collect::<Result<_, _>>()?;
    anyhow::ensure!(values.len() == 3, "a colour is R,G,B");
    let [r, g, b, _] = gfx_render::srgba(values[0], values[1], values[2], 1.0);
    Ok([r, g, b])
}

fn options(args: &[String]) -> anyhow::Result<Options> {
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|at| args.get(at + 1))
            .cloned()
    };
    let size = value("--size").map_or(Ok([1600, 1000]), |s| {
        let (w, h) = s.split_once('x').context("--size takes WIDTHxHEIGHT")?;
        anyhow::Ok([w.parse()?, h.parse()?])
    })?;
    Ok(Options {
        out: value("--screenshot")
            .context("--screenshot takes an output file")?
            .into(),
        size,
        pattern: value("--pattern").map(PathBuf::from),
        tool: match value("--tool").as_deref() {
            Some("points") => CanvasTool::Points,
            Some("sew") => CanvasTool::Sew,
            _ => CanvasTool::Select,
        },
        piece: value("--piece").and_then(|p| p.parse().ok()),
        frames: value("--frames").and_then(|f| f.parse().ok()).unwrap_or(12),
        commands: [
            value("--open").map(|path| UiCommand::OpenProject(path.into())),
            value("--new").map(|name| {
                UiCommand::NewProject(if name == "waistcoat" {
                    app_ui::Sample::Waistcoat
                } else {
                    app_ui::Sample::SleevelessTop
                })
            }),
        ]
        .into_iter()
        .flatten()
        .collect(),
        play: args.iter().any(|a| a == "--play"),
        hair: value("--hair")
            .map(|text| {
                let (text, fringe) = text
                    .strip_suffix("+fringe")
                    .map_or((text.as_str(), false), |t| (t, true));
                let (name, length) = text
                    .split_once(',')
                    .map_or((text, None), |(n, l)| (n, Some(l)));
                let style = avatar::HairStyle::ALL
                    .into_iter()
                    .find(|s| format!("{s:?}").eq_ignore_ascii_case(name))
                    .with_context(|| format!("unknown hair style {name}"))?;
                anyhow::Ok((
                    style,
                    length
                        .map(|l| l.parse::<f32>())
                        .transpose()?
                        .map(|cm| cm / 100.0),
                    fringe,
                ))
            })
            .transpose()?,
        skin: value("--skin").map(|t| srgb(&t)).transpose()?,
        hair_colour: value("--hair-colour").map(|t| srgb(&t)).transpose()?,
        camera: value("--camera")
            .map(|text| {
                let v: Vec<f32> = text.split(',').map(str::parse).collect::<Result<_, _>>()?;
                anyhow::ensure!(v.len() == 4, "--camera takes YAW,PITCH,DISTANCE,TARGET_Y");
                Ok([v[0], v[1], v[2], v[3]])
            })
            .transpose()?,
        avatar_window: args.iter().any(|a| a == "--avatar-window"),
        times: value("--at")
            .map(|t| t.split(',').filter_map(|v| v.parse().ok()).collect())
            .unwrap_or_default(),
        animation: value("--animate")
            .map(|name| {
                let motion = match name.as_str() {
                    "walk" => avatar::Motion::Walk,
                    "stiff" => avatar::Motion::StiffWalk,
                    "sway" => avatar::Motion::HipSwayWalk,
                    "sit" => avatar::Motion::SitOnChair,
                    "yoga" => avatar::Motion::Yoga,
                    other => anyhow::bail!("unknown animation {other}"),
                };
                Ok((
                    motion,
                    value("--at")
                        .and_then(|t| t.split(',').next()?.parse().ok())
                        .unwrap_or(0.0),
                ))
            })
            .transpose()?,
        bake_seconds: value("--bake-seconds").map(|v| v.parse()).transpose()?,
        solver: match value("--solver").as_deref() {
            None => None,
            Some("contact") => Some(SolverBackend::Contact),
            Some("cpu") => Some(SolverBackend::Cpu),
            Some("gpu") => Some(SolverBackend::Gpu),
            Some("global") => Some(SolverBackend::Global),
            Some(other) => {
                anyhow::bail!("unknown solver {other:?} (contact, global, cpu or gpu)")
            }
        },
        fabric: match value("--fabric").as_deref() {
            None => None,
            Some("denim") => Some(cad_core::FabricPattern::Denim),
            Some("corduroy") => Some(cad_core::FabricPattern::Corduroy),
            Some(other) => anyhow::bail!("unknown fabric {other:?} (denim or corduroy)"),
        },
    })
}

pub fn run(args: &[String]) -> anyhow::Result<()> {
    let options = options(args)?;
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = gfx_render::preferred_backends();
    let instance = wgpu::Instance::new(descriptor.with_env());
    let (gpu, _) = pollster::block_on(GpuContext::request(&instance, None))?;

    // --- the same set-up as the window -------------------------------------
    let egui_ctx = egui::Context::default();
    egui_ctx.set_theme(egui::Theme::Dark);
    let mut egui_renderer =
        egui_wgpu::Renderer::new(&gpu.device, FORMAT, egui_wgpu::RendererOptions::default());
    let mut canvas = CanvasRenderer::new(&gpu, [16, 16]);
    let mut scene = SceneRenderer::new(&gpu, [16, 16]);
    let mut floor = MeshData::default();
    floor.add_ground(Vec3::ZERO, 4.0);
    scene.add_static(
        &gpu,
        &floor,
        SurfaceMaterial {
            base_color: [0.10, 0.105, 0.115],
            roughness: 0.9,
            ..Default::default()
        },
    );
    let filter = wgpu::FilterMode::Linear;
    let textures = ViewportTextures {
        canvas: egui_renderer.register_native_texture(&gpu.device, canvas.target().view(), filter),
        scene: egui_renderer.register_native_texture(&gpu.device, scene.target().view(), filter),
    };
    let body = crate::app::find_body_model()
        .and_then(|path| ::avatar::BodyModel::load(path).ok())
        .map(|model| {
            let closing = crate::session::closing_radius(app_ui::SimulationState::DRAFT_MM);
            let params = if args.iter().any(|a| a == "--man") {
                ::avatar::BodyParams::male()
            } else {
                ::avatar::BodyParams::female()
            };
            Body::new(model, params, ::avatar::PoseParams::default(), closing)
        });
    let solver: Box<dyn ClothSolver> = if gpu.supports_compute {
        Box::new(XpbdSolver::with_gpu(gpu.device.clone(), gpu.queue.clone()))
    } else {
        Box::new(XpbdSolver::new())
    };
    let mut session = Session::new(Some(solver), body, Vec::new());
    session.garment_bake_duration = options.bake_seconds;
    session.editor.simulation.gpu_available = gpu.supports_compute;
    // As in the app: a real-time solver for the live view.
    session.editor.simulation.settings.backend = if gpu.supports_compute {
        SolverBackend::Gpu
    } else {
        SolverBackend::Cpu
    };
    if let Some(solver) = options.solver {
        session.editor.simulation.settings.backend = solver;
    }
    session.editor.avatar.window_open = options.avatar_window;
    let looks = &mut session.editor.avatar.appearance;
    if let Some((style, length, fringe)) = options.hair {
        looks.hair.style = style;
        looks.hair.length = length.unwrap_or(style.default_length());
        looks.hair.fringe = fringe;
    }
    if let Some(skin) = options.skin {
        looks.skin = skin;
    }
    // `--garment-preview`: drape, then show the garment as previewed (carried
    // with the body), without simulating it.
    let preview_garment = args.iter().any(|a| a == "--garment-preview");
    let simulate_garment = preview_garment || args.iter().any(|a| a == "--simulate-garment");
    session.auto_simulate = !preview_garment;
    let mut garment_stage = 0; // 0 draping, 1 simulating, 2 done
    let mut garment_started = std::time::Instant::now();
    if let Some((motion, time)) = options.animation.filter(|_| !simulate_garment) {
        let playing = args.iter().any(|a| a == "--play-animation");
        session.editor.avatar.animation = app_ui::AnimationState {
            motion: Some(motion),
            playing,
            time,
            ..Default::default()
        };
    }
    if let Some(colour) = options.hair_colour {
        looks.hair.color = colour;
    }
    if let Some([yaw, pitch, distance, target_y]) = options.camera {
        session.scene_camera = gfx_render::OrbitCamera {
            yaw: yaw.to_radians(),
            pitch: pitch.to_radians(),
            distance,
            target: Vec3::new(0.0, target_y, 0.0),
            ..Default::default()
        };
    }
    let mut ui = AppUi::new();
    let mut canvas_batch = CanvasBatch::default();

    // --- optional pattern ------------------------------------------------------
    let mut pending: Vec<UiCommand> = options.commands.clone();
    if let Some(pattern) = &options.pattern {
        session.editor.pattern_import.file_path = pattern.display().to_string();
        pending.push(UiCommand::InspectPatternFile(pattern.clone()));
    }
    let mut pattern_applied = options.pattern.is_none();

    let [width, height] = options.size;
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("screenshot"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut frame = 0;
    let mut play_started = false;
    let mut shot_index = 0;
    let (mut jitter, mut jitter_previous): (Vec<f32>, Option<Vec<glam::Vec3>>) = (Vec::new(), None);
    let mut ready_since: Option<u32> = None;
    let mut fabric_applied = options.fabric.is_none();
    loop {
        // Once the import has run (and its piece names are in), make it the
        // project, then set up the requested state.
        if !pattern_applied
            && !session.busy()
            && session.editor.pattern_import.pattern.is_some()
            && frame > 1
        {
            let pattern = session.editor.pattern_import.pattern.clone().unwrap();
            pending.push(UiCommand::ApplyImportedPattern {
                pattern: Box::new(pattern),
                replace_project: true,
            });
            pending.push(UiCommand::TidyLayout);
            pattern_applied = true;
        }
        session.editor.canvas_tool = options.tool;
        if let Some(index) = options.piece {
            session.editor.selected_piece = session.document.pieces().get(index).map(|p| p.id);
        }

        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(width as f32, height as f32),
            )),
            ..Default::default()
        };
        let mut ui_output = None;
        let mut full_output = egui_ctx.run_ui(raw_input, |root| {
            ui_output = Some(ui.show(root, &mut session.document, &mut session.editor, textures));
            // CLO360_GLYPHS="…": shows those characters, to see which ones
            // the interface fonts can draw.
            if let Ok(glyphs) = std::env::var("CLO360_GLYPHS") {
                egui::Window::new("Glyphs")
                    .default_pos([300.0, 200.0])
                    .show(root.ctx(), |ui| {
                        for chunk in glyphs.chars().collect::<Vec<_>>().chunks(8) {
                            ui.horizontal(|ui| {
                                for c in chunk {
                                    ui.label(
                                        egui::RichText::new(format!("{c} {:04X}", *c as u32))
                                            .size(18.0),
                                    );
                                }
                            });
                        }
                    });
            }
        });
        for (id, deltas) in full_output.textures_delta.set.drain() {
            for delta in &deltas {
                egui_renderer.update_texture(&gpu.device, &gpu.queue, id, delta);
            }
        }
        let freed: Vec<egui::TextureId> = full_output.textures_delta.free.drain().collect();
        let mut ui_output = ui_output.unwrap_or_default();
        ui_output.commands.append(&mut pending);
        let inputs = (ui_output.canvas, ui_output.scene);
        session.update(ui_output, 1.0 / 30.0);
        if !fabric_applied && frame > 1 && !session.busy() {
            if let Some(pattern) = options.fabric {
                if let Some(id) = session.document.fabrics().first().map(|fabric| fabric.id) {
                    if let Some(fabric) = session.document.fabric_mut(id) {
                        fabric.pattern = pattern;
                        fabric.base_color = match pattern {
                            cad_core::FabricPattern::Denim => [0.035, 0.10, 0.28],
                            cad_core::FabricPattern::Corduroy => [0.33, 0.12, 0.045],
                            _ => fabric.base_color,
                        };
                        fabric.weave_strength = 0.7;
                        fabric.wash = if pattern == cad_core::FabricPattern::Denim {
                            0.35
                        } else {
                            0.0
                        };
                        fabric.pattern_scale = 1.0;
                        fabric.relief = if pattern == cad_core::FabricPattern::Corduroy {
                            1.0
                        } else {
                            0.35
                        };
                    }
                    session.document.touch();
                }
            }
            fabric_applied = true;
        }

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("screenshot frame"),
            });
        crate::app::draw_viewports(
            &gpu,
            &mut session,
            &mut scene,
            &mut canvas,
            &mut canvas_batch,
            &mut egui_renderer,
            textures,
            gpu.device.limits().max_texture_dimension_2d,
            inputs,
            &mut encoder,
        );
        let jobs = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: full_output.pixels_per_point,
        };
        let extra =
            egui_renderer.update_buffers(&gpu.device, &gpu.queue, &mut encoder, &jobs, &screen);
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            egui_renderer.render(&mut pass, &jobs, &screen);
        }
        for id in &freed {
            egui_renderer.free_texture(id);
        }
        gpu.queue
            .submit(extra.into_iter().chain([encoder.finish()]));

        // CLO360_JITTER=1: how much the cloth still moves once settled.
        if std::env::var_os("CLO360_JITTER").is_some() {
            let positions = session.topology.mesh().positions.clone();
            if let Some(previous) = &jitter_previous {
                let moved = positions
                    .iter()
                    .zip(previous)
                    .map(|(a, b): (&glam::Vec3, &glam::Vec3)| a.distance(*b))
                    .fold(0.0f32, f32::max);
                jitter.push(moved);
            }
            jitter_previous = Some(positions);
        }
        frame += 1;
        // Wait for the import and its background work, then let the layout
        // settle for the requested number of frames.
        let ready = pattern_applied && !session.busy();
        // Started once; when it reaches its end the drape is done.
        // Simulating a garment through an animation needs it sewn and worn
        // first: play the drape.
        if ready && (options.play || simulate_garment) && !play_started {
            session.editor.timeline.toggle();
            play_started = true;
        }
        if ready && ready_since.is_none() {
            ready_since = Some(frame);
        }
        let drape_done = !session.editor.timeline.playing && session.is_draped();
        if simulate_garment
            && garment_stage < 2
            && ready_since.is_some_and(|since| frame - since >= options.frames)
            && (garment_stage > 0 || drape_done)
        {
            if garment_stage == 0 {
                println!(
                    "garment draped after {} frames; simulating it through the animation",
                    frame
                );
                if let Some((motion, time)) = options.animation {
                    session.editor.avatar.animation = app_ui::AnimationState {
                        motion: Some(motion),
                        playing: false,
                        time,
                        ..Default::default()
                    };
                    if !preview_garment {
                        pending.push(UiCommand::SimulateGarmentMotion);
                    }
                    garment_started = std::time::Instant::now();
                }
                garment_stage = if preview_garment { 2 } else { 1 };
                ready_since = Some(frame);
                shot_index = 0;
            } else if matches!(
                session.editor.avatar.animation.garment,
                app_ui::GarmentStatus::Carried | app_ui::GarmentStatus::NotWorn
            ) && garment_started.elapsed().as_secs_f32() > 2.0
            {
                anyhow::bail!("the garment simulation did not start (is the garment draped?)");
            } else if session.editor.avatar.animation.garment == app_ui::GarmentStatus::Unreliable {
                println!(
                    "garment simulation rejected as unreliable after {:.1} s; showing it carried",
                    garment_started.elapsed().as_secs_f32()
                );
                garment_stage = 2;
                ready_since = Some(frame);
                shot_index = 0;
            } else if session.editor.avatar.animation.garment == app_ui::GarmentStatus::Simulated {
                println!(
                    "garment simulated in {:.1} s",
                    garment_started.elapsed().as_secs_f32()
                );
                garment_stage = 2;
                ready_since = Some(frame);
                shot_index = 0;
            } else {
                if frame % 30 == 0
                    && let app_ui::GarmentStatus::Simulating(progress) =
                        session.editor.avatar.animation.garment
                {
                    println!("garment simulation {:.0}%", progress * 100.0);
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            frame += 1;
            continue;
        }
        // One image per requested time, three frames after setting it.
        if garment_stage == 2
            && options.times.len() > 1
            && ready_since.is_some_and(|since| frame - since >= 3)
        {
            let path = options.out.with_file_name(format!(
                "{}_{shot_index}.{}",
                options
                    .out
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("shot"),
                options
                    .out
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("ppm")
            ));
            save(&gpu, &target, [width, height], &path)?;
            println!("wrote {}", path.display());
            shot_index += 1;
            match options.times.get(shot_index) {
                Some(&time) => {
                    session.editor.avatar.animation.time = time;
                    ready_since = Some(frame);
                }
                None => return Ok(()),
            }
        }
        let waiting_for_garment = simulate_garment && garment_stage < 2;
        if (!waiting_for_garment
            && ready_since.is_some_and(|since| {
                frame - since
                    >= options
                        .frames
                        .min(if garment_stage == 2 { 3 } else { u32::MAX })
            }))
            || frame > 200_000
        {
            break;
        }
        if !ready {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    if !jitter.is_empty() {
        let tail = &jitter[jitter.len().saturating_sub(30)..];
        println!(
            "cloth movement per frame over the last {} frames: max {:.2} mm, mean {:.2} mm",
            tail.len(),
            tail.iter().fold(0.0f32, |a, &b| a.max(b)) * 1000.0,
            tail.iter().sum::<f32>() / tail.len() as f32 * 1000.0
        );
    }
    save(&gpu, &target, [width, height], &options.out)?;
    println!("wrote {}", options.out.display());
    Ok(())
}

/// Reads the texture back and writes it as binary PPM.
fn save(
    gpu: &GpuContext,
    texture: &wgpu::Texture,
    [width, height]: [u32; 2],
    path: &std::path::Path,
) -> anyhow::Result<()> {
    let row = width * 4;
    let padded = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("screenshot read-back"),
        size: u64::from(padded) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
    let data = slice
        .get_mapped_range()
        .context("read-back buffer not mapped")?;
    let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
    for line in data.chunks_exact(padded as usize) {
        bytes.extend(
            line[..row as usize]
                .chunks_exact(4)
                .flat_map(|rgba| &rgba[..3]),
        );
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

//! Window loop: owns the window, the GPU surface and the `egui` integration,
//! and runs one frame per redraw request.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use app_ui::{AppUi, ViewportInput, ViewportTextures};
use gfx_render::scene3d::avatar;
use gfx_render::{
    CanvasBatch, CanvasRenderer, GpuContext, MeshData, MeshVertex, SceneRenderer, Shading,
    SurfaceMaterial, ViewportRenderer, canvas2d, preferred_backends,
};
use glam::Vec2;
use glam::Vec3;
use physics_xpbd::XpbdSolver;
use topology_sync::{ClothSolver, Collider, SolverBackend, TopologySync};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::window::{Window, WindowId};

use crate::body::Body;
use crate::session::Session;

/// The `winit` application. GPU state is created on `resumed`, as required
/// on Android and the web, and harmless everywhere else.
pub struct App {
    running: Option<Running>,
    error: Option<anyhow::Error>,
    autoplay: bool,
    /// An avatar file to import at start-up.
    avatar: Option<std::path::PathBuf>,
}

impl App {
    pub fn new(autoplay: bool, avatar: Option<std::path::PathBuf>) -> Self {
        Self {
            running: None,
            error: None,
            autoplay,
            avatar,
        }
    }

    /// The start-up error, if the event loop exited because of one.
    pub fn into_result(self) -> anyhow::Result<()> {
        self.error.map_or(Ok(()), Err)
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.running.is_some() {
            return;
        }
        // Native targets can block on device creation. A wasm32 build must
        // instead `spawn_local` this future and deliver the result through an
        // `EventLoopProxy`; everything below `Running::new` is target-agnostic.
        match pollster::block_on(Running::new(event_loop)) {
            Ok(mut running) => {
                if let Some(path) = self.avatar.take() {
                    let import = app_ui::UiOutput {
                        commands: vec![app_ui::UiCommand::ImportAvatar(path)],
                        ..Default::default()
                    };
                    running.session.update(import, 0.0);
                    match &running.session.editor.io.status {
                        Some(Ok(message)) => log::info!("{message}"),
                        Some(Err(message)) => log::error!("{message}"),
                        None => {}
                    }
                }
                running.session.editor.timeline.playing = self.autoplay;
                running.window.request_redraw();
                self.running = Some(running);
            }
            Err(error) => {
                self.error = Some(error);
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window: WindowId,
        event: WindowEvent,
    ) {
        let Some(running) = &mut self.running else {
            return;
        };

        // egui sees every event first; viewport input also flows through it,
        // so nothing here needs to know whether egui "consumed" the event.
        let response = running.egui_state.on_window_event(&running.window, &event);
        if response.repaint {
            running.window.request_redraw();
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                running.resize_surface(size.width, size.height);
                running.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                match running.frame() {
                    Ok(repaint_after) => {
                        if running.session.quit_requested {
                            event_loop.exit();
                        } else if repaint_after.is_zero() {
                            running.window.request_redraw();
                        } else if let Some(deadline) = Instant::now().checked_add(repaint_after) {
                            // e.g. a blinking text cursor or a tooltip delay.
                            event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
                        } else {
                            event_loop.set_control_flow(ControlFlow::Wait);
                        }
                    }
                    Err(error) => {
                        self.error = Some(error);
                        event_loop.exit();
                    }
                }
            }
            _ => {}
        }
    }

    fn new_events(&mut self, _event_loop: &ActiveEventLoop, cause: winit::event::StartCause) {
        // A `WaitUntil` deadline set by the last frame has passed.
        if let (winit::event::StartCause::ResumeTimeReached { .. }, Some(running)) =
            (cause, &self.running)
        {
            running.window.request_redraw();
        }
    }
}

/// How far the avatar is drawn inside its true surface, in metres.
const SKIN_INSET: f32 = 0.0025;

/// Looks for the baked body model: an explicit override, next to the
/// working directory, next to the executable, then in the source tree.
pub(crate) fn find_body_model() -> Option<std::path::PathBuf> {
    const RELATIVE: &str = "assets/avatar/body.bin";
    let mut candidates = Vec::new();
    if let Some(path) = std::env::var_os("CLO360_BODY") {
        candidates.push(std::path::PathBuf::from(path));
    }
    candidates.push(RELATIVE.into());
    if let Ok(exe) = std::env::current_exe() {
        candidates.extend(
            exe.ancestors()
                .skip(1)
                .take(4)
                .map(|dir| dir.join(RELATIVE)),
        );
    }
    candidates.push(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(RELATIVE),
    );
    candidates.into_iter().find(|path| path.is_file())
}

pub(crate) struct Running {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    gpu: GpuContext,
    max_texture_side: u32,

    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,

    canvas: CanvasRenderer,
    scene: SceneRenderer,
    textures: ViewportTextures,
    canvas_batch: CanvasBatch,

    ui: AppUi,
    session: Session,
    last_frame: Instant,
}

impl Running {
    async fn new(event_loop: &ActiveEventLoop) -> anyhow::Result<Self> {
        let attributes = Window::default_attributes()
            .with_title("clo360")
            .with_inner_size(LogicalSize::new(1440.0, 900.0));
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .context("failed to create the window")?,
        );

        // --- GPU ---------------------------------------------------------------
        // The display handle lets the GL fallback create a context on Wayland;
        // `with_env` honours WGPU_BACKEND and friends.
        let mut descriptor = wgpu::InstanceDescriptor::new_with_display_handle(Box::new(
            event_loop.owned_display_handle(),
        ));
        descriptor.backends = preferred_backends();
        let instance = wgpu::Instance::new(descriptor.with_env());
        let surface = instance
            .create_surface(window.clone())
            .context("failed to create the window surface")?;
        let (gpu, adapter) = GpuContext::request(&instance, Some(&surface)).await?;

        let size = window.inner_size();
        let mut surface_config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .context("the adapter cannot present to this window")?;
        // egui blends in gamma space and expects a non-sRGB framebuffer.
        let formats = surface.get_capabilities(&adapter).formats;
        if let Some(format) = formats.iter().copied().find(|f| !f.is_srgb()) {
            surface_config.format = format;
        }
        surface.configure(&gpu.device, &surface_config);
        let max_texture_side = gpu.device.limits().max_texture_dimension_2d;

        // --- egui --------------------------------------------------------------
        let egui_ctx = egui::Context::default();
        egui_ctx.set_theme(egui::Theme::Dark);
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(max_texture_side as usize),
        );
        let mut egui_renderer = egui_wgpu::Renderer::new(
            &gpu.device,
            surface_config.format,
            egui_wgpu::RendererOptions::default(),
        );

        // --- viewports ---------------------------------------------------------
        // Real sizes arrive with the first UI frame.
        let canvas = CanvasRenderer::new(&gpu, [16, 16]);
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
            canvas: egui_renderer.register_native_texture(
                &gpu.device,
                canvas.target().view(),
                filter,
            ),
            scene: egui_renderer.register_native_texture(
                &gpu.device,
                scene.target().view(),
                filter,
            ),
        };

        // --- avatar ------------------------------------------------------------
        let body = match find_body_model() {
            Some(path) => match ::avatar::BodyModel::load(&path) {
                Ok(model) => {
                    let closing = crate::session::closing_radius(app_ui::SimulationState::DRAFT_MM);
                    Some(Body::new(
                        model,
                        ::avatar::BodyParams::female(),
                        ::avatar::PoseParams::default(),
                        closing,
                    ))
                }
                Err(error) => {
                    log::error!("cannot load {}: {error}", path.display());
                    None
                }
            },
            None => {
                log::warn!(
                    "assets/avatar/body.bin not found; run tools/bake_avatar.py. Using the placeholder mannequin."
                );
                None
            }
        };
        // Without the body model, fall back to the ellipsoid mannequin,
        // whose ellipsoids are also what the cloth collides with.
        let mut colliders = Vec::new();
        if body.is_none() {
            scene.set_avatar(
                &gpu,
                &avatar::mannequin(),
                SurfaceMaterial {
                    base_color: [0.55, 0.56, 0.60],
                    roughness: 0.55,
                    ..Default::default()
                },
            );
            colliders = avatar::mannequin_parts()
                .iter()
                .map(|part| Collider {
                    center: part.center,
                    radii: part.radii,
                    rotation: part.rotation,
                })
                .collect();
        }

        // --- simulation --------------------------------------------------------
        let solver: Box<dyn ClothSolver> = if gpu.supports_compute {
            Box::new(XpbdSolver::with_gpu(gpu.device.clone(), gpu.queue.clone()))
        } else {
            log::warn!("no compute shader support: cloth simulation runs on the CPU only");
            Box::new(XpbdSolver::new())
        };
        let mut session = Session::new(Some(solver), body, colliders);
        session.editor.simulation.gpu_available = gpu.supports_compute;
        // As in the app: a real-time solver for the live view.
        session.editor.simulation.settings.backend = if gpu.supports_compute {
            SolverBackend::Gpu
        } else {
            SolverBackend::Cpu
        };
        // The live view must keep up with the frame rate: the explicit
        // solver on the GPU, or on the CPU without compute shaders. (The
        // contact solver bakes animations, and can be picked here too.)
        session.editor.simulation.settings.backend = if gpu.supports_compute {
            SolverBackend::Gpu
        } else {
            SolverBackend::Cpu
        };
        let info = adapter.get_info();
        session.editor.stats.backend = format!("{} · {:?}", info.name, info.backend);

        Ok(Self {
            window,
            surface,
            surface_config,
            gpu,
            max_texture_side,
            egui_ctx,
            egui_state,
            egui_renderer,
            canvas,
            scene,
            textures,
            canvas_batch: CanvasBatch::default(),
            ui: AppUi::new(),
            session,
            last_frame: Instant::now(),
        })
    }

    fn resize_surface(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return; // minimised
        }
        self.surface_config.width = width.min(self.max_texture_side);
        self.surface_config.height = height.min(self.max_texture_side);
        self.surface
            .configure(&self.gpu.device, &self.surface_config);
    }

    /// Runs one frame. Returns how long the UI is happy to wait before the
    /// next one (`Duration::ZERO` means "as soon as possible").
    fn frame(&mut self) -> anyhow::Result<Duration> {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.session.editor.stats.frame_ms = dt * 1000.0;

        // 1. Build the UI. Viewport tabs report their size and input.
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let mut ui_output = None;
        let mut full_output = self.egui_ctx.run_ui(raw_input, |ui| {
            ui_output = Some(self.ui.show(
                ui,
                &mut self.session.document,
                &mut self.session.editor,
                self.textures,
            ));
        });
        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);
        // egui's texture changes (font atlas etc.) must be applied even if
        // this frame ends up not being presented.
        for (id, deltas) in full_output.textures_delta.set.drain() {
            for delta in &deltas {
                self.egui_renderer
                    .update_texture(&self.gpu.device, &self.gpu.queue, id, delta);
            }
        }
        let freed_textures: Vec<egui::TextureId> =
            full_output.textures_delta.free.drain().collect();
        let ui_output = ui_output.unwrap_or_default();
        let (canvas_input, scene_input) = (ui_output.canvas, ui_output.scene);

        // 2. Apply input and edits; bring the 3D mesh up to date.
        self.session.update(ui_output, dt);

        // 3. Render the viewports that are on screen into their targets.
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        draw_viewports(
            &self.gpu,
            &mut self.session,
            &mut self.scene,
            &mut self.canvas,
            &mut self.canvas_batch,
            &mut self.egui_renderer,
            self.textures,
            self.max_texture_side,
            (canvas_input, scene_input),
            &mut encoder,
        );

        // 4. Composite the UI (which samples those targets) onto the window.
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface
                    .configure(&self.gpu.device, &self.surface_config);
                self.finish_without_presenting(encoder, &freed_textures);
                return Ok(Duration::ZERO); // try again next frame
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                self.finish_without_presenting(encoder, &freed_textures);
                return Ok(Duration::from_millis(100));
            }
            wgpu::CurrentSurfaceTexture::Validation => anyhow::bail!("surface validation error"),
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let pixels_per_point = full_output.pixels_per_point;
        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point,
        };
        let extra_commands = self.egui_renderer.update_buffers(
            &self.gpu.device,
            &self.gpu.queue,
            &mut encoder,
            &paint_jobs,
            &screen,
        );
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
            self.egui_renderer.render(&mut pass, &paint_jobs, &screen);
        }
        for id in &freed_textures {
            self.egui_renderer.free_texture(id);
        }

        self.gpu
            .queue
            .submit(extra_commands.into_iter().chain([encoder.finish()]));
        self.window.pre_present_notify();
        self.gpu.queue.present(frame);

        // 5. When is the next frame due?
        let egui_delay = full_output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .map_or(Duration::MAX, |v| v.repaint_delay);
        Ok(if self.session.animating() {
            Duration::ZERO
        } else {
            egui_delay
        })
    }

    /// Ends a frame whose surface texture could not be acquired: the viewport
    /// passes are still submitted so queued buffer writes are flushed.
    fn finish_without_presenting(
        &mut self,
        encoder: wgpu::CommandEncoder,
        freed_textures: &[egui::TextureId],
    ) {
        self.gpu.queue.submit([encoder.finish()]);
        for id in freed_textures {
            self.egui_renderer.free_texture(id);
        }
    }

    /// Matches a viewport's render target to the size its tab has on screen
    /// and re-points the `egui` texture at the new image if it was recreated.
    pub(crate) fn fit_target(
        gpu: &GpuContext,
        egui_renderer: &mut egui_wgpu::Renderer,
        renderer: &mut impl ViewportRenderer,
        texture: egui::TextureId,
        input: &ViewportInput,
        max_side: u32,
    ) {
        let size = input.size.map(|side| side.clamp(1, max_side));
        if renderer.resize(gpu, size) {
            egui_renderer.update_egui_texture_from_wgpu_texture(
                &gpu.device,
                renderer.target().view(),
                wgpu::FilterMode::Linear,
                texture,
            );
        }
    }
}

/// Draws the body with its looks: skin (with brows and lips painted on),
/// eyes and hair.
fn upload_avatar(gpu: &GpuContext, scene: &mut SceneRenderer, session: &mut Session) {
    let appearance = session.editor.avatar.appearance;
    let white = [1.0; 3];
    let skin_material = SurfaceMaterial {
        base_color: white,
        roughness: 0.5,
        shading: Shading::Skin,
        ..Default::default()
    };
    // Drawn slightly smaller than it is (CLO calls this the skin offset):
    // the cloth rests a few millimetres off the true skin, and this keeps
    // the last of it from shimmering through.
    let skin_mesh = |body: &::avatar::BodyMesh, colors: &dyn Fn(usize) -> [f32; 3]| MeshData {
        vertices: (0..body.positions.len())
            .map(|i| {
                MeshVertex::new(
                    body.positions[i] - body.normals[i] * SKIN_INSET,
                    body.normals[i],
                    Vec2::ZERO,
                )
                .with_color(colors(i))
            })
            .collect(),
        // The body's own eyes are replaced by coloured eyeballs.
        indices: body.skin().iter().flatten().copied().collect(),
    };

    // The parametric body: looks built once, carried along with the head.
    if session.imported.is_none()
        && let Some(body) = &mut session.body
        && body.looks(&appearance).is_some()
        && let Some(looks) = body.cached_looks()
    {
        let head = body.head_transform(looks);
        let moved = |mesh: &::avatar::LookMesh, offset: Vec3| MeshData {
            vertices: (0..mesh.positions.len())
                .map(|i| {
                    let position = head.transform_point3(mesh.positions[i] + offset);
                    let normal = head
                        .transform_vector3(mesh.normals[i])
                        .normalize_or(Vec3::Y);
                    MeshVertex::new(position, normal, mesh.uvs[i])
                        .with_color(mesh.colors[i])
                        .with_value(mesh.coverage[i])
                })
                .collect(),
            indices: mesh.triangles.iter().flatten().copied().collect(),
        };
        let skin = skin_mesh(&body.mesh, &|i| {
            looks.skin_colors.get(i).copied().unwrap_or(appearance.skin)
        });
        // Set back with the eyelids around them.
        let eyes = moved(&looks.eyes, Vec3::NEG_Z * SKIN_INSET);
        let hair = moved(&looks.hair, Vec3::ZERO);
        let root = head.transform_point3(looks.hair_root).to_array();
        scene.set_avatar_parts(
            gpu,
            &[
                (&skin, skin_material),
                (
                    &eyes,
                    SurfaceMaterial {
                        base_color: white,
                        roughness: 0.08,
                        ..Default::default()
                    },
                ),
                (
                    &hair,
                    SurfaceMaterial {
                        base_color: white,
                        roughness: 0.5,
                        shading: Shading::Hair { root },
                        ..Default::default()
                    },
                ),
            ],
        );
        return;
    }
    // An imported avatar (or a body without eyes): skin colour only.
    if let Some((body, _)) = session.avatar() {
        let skin = skin_mesh(body, &|_| appearance.skin);
        scene.set_avatar(gpu, &skin, skin_material);
    }
}

/// Brings the avatar and garment buffers up to date and draws the
/// viewports that are on screen into their targets.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_viewports(
    gpu: &GpuContext,
    session: &mut Session,
    scene: &mut SceneRenderer,
    canvas: &mut CanvasRenderer,
    canvas_batch: &mut CanvasBatch,
    egui_renderer: &mut egui_wgpu::Renderer,
    textures: ViewportTextures,
    max_side: u32,
    (canvas_input, scene_input): (Option<ViewportInput>, Option<ViewportInput>),
    encoder: &mut wgpu::CommandEncoder,
) {
    if session.take_avatar_dirty() {
        upload_avatar(gpu, scene, session);
    }
    scene.overlay = session.surface_overlay().0;
    if session.take_garment_dirty() {
        let (_, values) = session.surface_overlay();
        scene.sync_garment(gpu, session.topology.mesh(), values, |piece| {
            session.fabric_thickness(piece)
        });
    }
    if let Some(input) = canvas_input {
        Running::fit_target(
            gpu,
            egui_renderer,
            canvas,
            textures.canvas,
            &input,
            max_side,
        );
        canvas2d::scene::build(
            canvas_batch,
            &session.document,
            Some(session.topology.mesh()),
            &session.canvas_camera,
            canvas.viewport(),
            &session.canvas_overlay(),
        );
        canvas.prepare(gpu, &canvas_batch, &session.canvas_camera);
        canvas.render(gpu, encoder);
    }
    if let Some(input) = scene_input {
        Running::fit_target(gpu, egui_renderer, scene, textures.scene, &input, max_side);
        // Buttons ride on the cloth, so they follow it every frame; the
        // chair of the sitting animation rises out of the floor.
        let mut props = gfx_render::garment_props(&session.document, &session.topology);
        if let Some((min, max)) = session.chair() {
            gfx_render::scene3d::props::chair(&mut props, min, max);
        }
        scene.set_props(gpu, &props);
        let session = &*session;
        scene.prepare(gpu, &session.scene_camera, |piece| {
            session.material_of(piece)
        });
        scene.render(gpu, encoder);
    }
}

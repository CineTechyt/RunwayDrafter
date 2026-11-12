//! Window loop: owns the window, the GPU surface and the `egui` integration,
//! and runs one frame per redraw request.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use app_ui::{AppUi, ViewportInput, ViewportTextures};
use gfx_render::scene3d::avatar;
use gfx_render::{
    CanvasBatch, CanvasRenderer, GpuContext, MeshData, MeshVertex, SceneRenderer, SurfaceMaterial, ViewportRenderer,
    canvas2d,
    preferred_backends,
};
use glam::Vec3;
use glam::Vec2;
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
        Self { running: None, error: None, autoplay, avatar }
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
                    let import = app_ui::UiOutput { commands: vec![app_ui::UiCommand::ImportAvatar(path)], ..Default::default() };
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

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window: WindowId, event: WindowEvent) {
        let Some(running) = &mut self.running else { return };

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
        if let (winit::event::StartCause::ResumeTimeReached { .. }, Some(running)) = (cause, &self.running) {
            running.window.request_redraw();
        }
    }
}

/// How far the avatar is drawn inside its true surface, in metres.
const SKIN_INSET: f32 = 0.0025;

/// Looks for the baked body model: an explicit override, next to the
/// working directory, next to the executable, then in the source tree.
fn find_body_model() -> Option<std::path::PathBuf> {
    const RELATIVE: &str = "assets/avatar/body.bin";
    let mut candidates = Vec::new();
    if let Some(path) = std::env::var_os("CLO360_BODY") {
        candidates.push(std::path::PathBuf::from(path));
    }
    candidates.push(RELATIVE.into());
    if let Ok(exe) = std::env::current_exe() {
        candidates.extend(exe.ancestors().skip(1).take(4).map(|dir| dir.join(RELATIVE)));
    }
    candidates.push(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(RELATIVE));
    candidates.into_iter().find(|path| path.is_file())
}

struct Running {
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
        let attributes = Window::default_attributes().with_title("clo360").with_inner_size(LogicalSize::new(1440.0, 900.0));
        let window = Arc::new(event_loop.create_window(attributes).context("failed to create the window")?);

        // --- GPU ---------------------------------------------------------------
        // The display handle lets the GL fallback create a context on Wayland;
        // `with_env` honours WGPU_BACKEND and friends.
        let mut descriptor =
            wgpu::InstanceDescriptor::new_with_display_handle(Box::new(event_loop.owned_display_handle()));
        descriptor.backends = preferred_backends();
        let instance = wgpu::Instance::new(descriptor.with_env());
        let surface = instance.create_surface(window.clone()).context("failed to create the window surface")?;
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
        let mut egui_renderer =
            egui_wgpu::Renderer::new(&gpu.device, surface_config.format, egui_wgpu::RendererOptions::default());

        // --- viewports ---------------------------------------------------------
        // Real sizes arrive with the first UI frame.
        let canvas = CanvasRenderer::new(&gpu, [16, 16]);
        let mut scene = SceneRenderer::new(&gpu, [16, 16]);
        let mut floor = MeshData::default();
        floor.add_ground(Vec3::ZERO, 4.0);
        scene.add_static(&gpu, &floor, SurfaceMaterial { base_color: [0.10, 0.105, 0.115], roughness: 0.9, ..Default::default() });

        let filter = wgpu::FilterMode::Linear;
        let textures = ViewportTextures {
            canvas: egui_renderer.register_native_texture(&gpu.device, canvas.target().view(), filter),
            scene: egui_renderer.register_native_texture(&gpu.device, scene.target().view(), filter),
        };

        // --- avatar ------------------------------------------------------------
        let body = match find_body_model() {
            Some(path) => match ::avatar::BodyModel::load(&path) {
                Ok(model) => {
                    let closing = crate::session::closing_radius(app_ui::SimulationState::DRAFT_MM);
                    Some(Body::new(model, ::avatar::BodyParams::female(), ::avatar::PoseParams::default(), closing))
                }
                Err(error) => {
                    log::error!("cannot load {}: {error}", path.display());
                    None
                }
            },
            None => {
                log::warn!("assets/avatar/body.bin not found; run tools/bake_avatar.py. Using the placeholder mannequin.");
                None
            }
        };
        // Without the body model, fall back to the ellipsoid mannequin,
        // whose ellipsoids are also what the cloth collides with.
        let mut colliders = Vec::new();
        if body.is_none() {
            scene.set_avatar(&gpu, &avatar::mannequin(), SurfaceMaterial { base_color: [0.55, 0.56, 0.60], roughness: 0.55, ..Default::default() });
            colliders = avatar::mannequin_parts()
                .iter()
                .map(|part| Collider { center: part.center, radii: part.radii, rotation: part.rotation })
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
        if gpu.supports_compute {
            session.editor.simulation.settings.backend = SolverBackend::Gpu;
        }
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
        self.surface.configure(&self.gpu.device, &self.surface_config);
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
            ui_output = Some(self.ui.show(ui, &mut self.session.document, &mut self.session.editor, self.textures));
        });
        self.egui_state.handle_platform_output(&self.window, full_output.platform_output);
        // egui's texture changes (font atlas etc.) must be applied even if
        // this frame ends up not being presented.
        for (id, deltas) in full_output.textures_delta.set.drain() {
            for delta in &deltas {
                self.egui_renderer.update_texture(&self.gpu.device, &self.gpu.queue, id, delta);
            }
        }
        let freed_textures: Vec<egui::TextureId> = full_output.textures_delta.free.drain().collect();
        let ui_output = ui_output.unwrap_or_default();
        let (canvas_input, scene_input) = (ui_output.canvas, ui_output.scene);

        // 2. Apply input and edits; bring the 3D mesh up to date.
        self.session.update(ui_output, dt);

        // 3. Render the viewports that are on screen into their targets.
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        if self.session.take_avatar_dirty()
            && let Some((body, _)) = self.session.avatar()
        {
            let mesh = MeshData {
                // Drawn slightly smaller than it is (CLO calls this the skin
                // offset): the cloth rests a few millimetres off the true
                // skin, and this keeps the last of it from shimmering through.
                vertices: body
                    .positions
                    .iter()
                    .zip(&body.normals)
                    .map(|(p, n)| MeshVertex::new(*p - *n * SKIN_INSET, *n, Vec2::ZERO))
                    .collect(),
                indices: body.triangles.iter().flatten().copied().collect(),
            };
            self.scene.set_avatar(&self.gpu, &mesh, SurfaceMaterial { base_color: [0.62, 0.45, 0.36], roughness: 0.65, ..Default::default() });
        }
        self.scene.overlay = self.session.surface_overlay().0;
        if self.session.take_garment_dirty() {
            let (_, values) = self.session.surface_overlay();
            self.scene.sync_garment(&self.gpu, self.session.topology.mesh(), values);
        }
        if let Some(input) = canvas_input {
            Self::fit_target(&self.gpu, &mut self.egui_renderer, &mut self.canvas, self.textures.canvas, &input, self.max_texture_side);
            canvas2d::scene::build(
                &mut self.canvas_batch,
                &self.session.document,
                Some(self.session.topology.mesh()),
                &self.session.canvas_camera,
                self.canvas.viewport(),
                &self.session.canvas_overlay(),
            );
            self.canvas.prepare(&self.gpu, &self.canvas_batch, &self.session.canvas_camera);
            self.canvas.render(&self.gpu, &mut encoder);
        }
        if let Some(input) = scene_input {
            Self::fit_target(&self.gpu, &mut self.egui_renderer, &mut self.scene, self.textures.scene, &input, self.max_texture_side);
            let session = &self.session;
            self.scene.prepare(&self.gpu, &session.scene_camera, |piece| session.material_of(piece));
            self.scene.render(&self.gpu, &mut encoder);
        }

        // 4. Composite the UI (which samples those targets) onto the window.
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.gpu.device, &self.surface_config);
                self.finish_without_presenting(encoder, &freed_textures);
                return Ok(Duration::ZERO); // try again next frame
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                self.finish_without_presenting(encoder, &freed_textures);
                return Ok(Duration::from_millis(100));
            }
            wgpu::CurrentSurfaceTexture::Validation => anyhow::bail!("surface validation error"),
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let pixels_per_point = full_output.pixels_per_point;
        let paint_jobs = self.egui_ctx.tessellate(full_output.shapes, pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point,
        };
        let extra_commands =
            self.egui_renderer.update_buffers(&self.gpu.device, &self.gpu.queue, &mut encoder, &paint_jobs, &screen);
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut pass, &paint_jobs, &screen);
        }
        for id in &freed_textures {
            self.egui_renderer.free_texture(id);
        }

        self.gpu.queue.submit(extra_commands.into_iter().chain([encoder.finish()]));
        self.window.pre_present_notify();
        self.gpu.queue.present(frame);

        // 5. When is the next frame due?
        let egui_delay =
            full_output.viewport_output.get(&egui::ViewportId::ROOT).map_or(Duration::MAX, |v| v.repaint_delay);
        Ok(if self.session.animating() { Duration::ZERO } else { egui_delay })
    }

    /// Ends a frame whose surface texture could not be acquired: the viewport
    /// passes are still submitted so queued buffer writes are flushed.
    fn finish_without_presenting(&mut self, encoder: wgpu::CommandEncoder, freed_textures: &[egui::TextureId]) {
        self.gpu.queue.submit([encoder.finish()]);
        for id in freed_textures {
            self.egui_renderer.free_texture(id);
        }
    }

    /// Matches a viewport's render target to the size its tab has on screen
    /// and re-points the `egui` texture at the new image if it was recreated.
    fn fit_target(
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

//! The editing session: the document plus everything derived from it,
//! independent of windowing and GPU objects.

use app_ui::{EditorState, ExportFormat, SurfaceView, UiCommand, UiOutput};
use cad_core::{DocumentChange, History, PatternDocument, PieceId};
use gfx_render::{Camera2D, CanvasOverlay, OrbitCamera, SurfaceMaterial, SurfaceOverlay};
use glam::Vec2;
use topology_sync::{ClothSolver, Collider, GarmentTopology, MeshingParams, SolverContext, TopologySync};

use avatar::Measure;

use crate::body::{Body, ImportedAvatar};
use crate::controllers::{self, CanvasTool};

/// Radius the avatar's collision field is closed with for a cloth mesh of
/// the given particle distance (mm): half a triangle edge, within reason.
pub fn closing_radius(particle_distance_mm: f32) -> f32 {
    (0.5 * particle_distance_mm * 1e-3).clamp(0.002, 0.015)
}

/// Simulation steps run after the avatar changed under a worn garment.
const SETTLE_STEPS: u32 = 60;

pub struct Session {
    pub document: PatternDocument,
    pub topology: GarmentTopology,
    pub history: History,
    pub editor: EditorState,
    pub canvas_camera: Camera2D,
    pub scene_camera: OrbitCamera,
    canvas_tool: CanvasTool,
    /// Draping backend; `None` runs the editor without simulation.
    solver: Option<Box<dyn ClothSolver>>,
    /// The parametric avatar; `None` if its model file is missing.
    pub body: Option<Body>,
    /// An avatar loaded from a file; while set it stands in for `body`.
    pub imported: Option<ImportedAvatar>,
    /// Per-vertex values of the strain or fit map, as last computed.
    overlay_values: Vec<f32>,
    /// What `overlay_values` were computed for.
    overlay_shown: SurfaceView,
    /// Collision shapes of the placeholder mannequin used without `body`.
    colliders: Vec<Collider>,
    avatar_dirty: bool,
    /// The targets the body was last solved for.
    solved_targets: [Option<f32>; Measure::ALL.len()],
    /// Targets or build changed while dragging: solve on release.
    fit_due: bool,
    /// The garment has been simulated since it was last arranged, so it is
    /// worn rather than laid out.
    draped: bool,
    /// Simulation steps still owed after the avatar moved under the
    /// garment, so it settles even while the timeline is paused.
    settle_steps: u32,
    fit_canvas: bool,
    garment_dirty: bool,
    pub quit_requested: bool,
}

impl Session {
    pub fn new(solver: Option<Box<dyn ClothSolver>>, body: Option<Body>, colliders: Vec<Collider>) -> Self {
        let mut document = PatternDocument::demo_top();
        let mut editor = EditorState::default();
        if let Some(body) = &body {
            avatar::arrange_around(&mut document, &body.mesh, &body.measurements);
            editor.avatar.available = true;
            editor.avatar.params = body.params;
            editor.avatar.pose = body.pose;
            editor.avatar.measurements = body.measurements;
            editor.avatar.modelled_angles = body.modelled_angles();
        }
        let mut topology = GarmentTopology::default();
        topology.rebuild(&document);
        let mut solver = solver;
        if let Some(solver) = &mut solver {
            let ctx = SolverContext {
                document: &document,
                settings: &editor.simulation.settings,
                colliders: &colliders,
                body: body.as_ref().and_then(|b| b.field.as_ref()),
                body_mesh: body.as_ref().and_then(|b| b.collider.as_ref()),
                ground: 0.0,
                dt: editor.timeline.step_seconds(),
            };
            solver.rebind(topology.mesh(), &ctx);
        }
        Self {
            document,
            topology,
            history: History::default(),
            editor,
            canvas_camera: Camera2D::default(),
            scene_camera: OrbitCamera::default(),
            canvas_tool: CanvasTool::default(),
            solver,
            body,
            imported: None,
            overlay_values: Vec::new(),
            overlay_shown: SurfaceView::Fabric,
            colliders,
            avatar_dirty: true,
            solved_targets: [None; Measure::ALL.len()],
            fit_due: false,
            draped: false,
            settle_steps: 0,
            fit_canvas: true,
            garment_dirty: true,
            quit_requested: false,
        }
    }

    /// Applies one frame of UI output: commands, viewport tools, the
    /// 2D → 3D synchronisation and the simulation clock.
    pub fn update(&mut self, output: UiOutput, dt: f32) {
        let UiOutput { mut changes, commands, canvas, scene } = output;
        let mut extra_steps = 0;
        let mut arrange = false;

        // --- avatar ------------------------------------------------------------
        if let Some(imported) = &mut self.imported {
            imported.set_closing(closing_radius(self.editor.simulation.particle_distance));
        } else if let Some(body) = &mut self.body {
            let wanted = &mut self.editor.avatar;
            let mut rebuilt = false;
            let before = (self.draped && (wanted.params != body.params || wanted.pose != body.pose || self.fit_due))
                .then(|| body.snapshot());

            // Shape: the build sliders edit the parameters directly; held
            // tape measurements are then solved into them, all together.
            let targets: Vec<(Measure, f32)> =
                Measure::ALL.iter().filter_map(|&m| Some((m, wanted.targets[m as usize]?))).collect();
            let shape_edited = wanted.params != body.params || wanted.targets != self.solved_targets;
            self.fit_due |= shape_edited && !targets.is_empty();
            if targets.is_empty() {
                wanted.fit = None;
                self.fit_due = false;
            }
            if self.fit_due && !wanted.dragging {
                // Takes up to about a second, so it waits for the mouse
                // button to be released.
                wanted.fit = Some(avatar::fit_all(body.model(), &mut wanted.params, &targets));
                self.fit_due = false;
            }
            if shape_edited || wanted.params != body.params {
                self.solved_targets = wanted.targets;
                body.set_params(wanted.params);
                rebuilt = true;
            }
            if wanted.pose != body.pose {
                body.set_pose(wanted.pose);
                rebuilt = true;
            }
            if rebuilt {
                wanted.measurements = body.measurements;
                wanted.modelled_angles = body.modelled_angles();
                self.avatar_dirty = true;
                match &before {
                    // A worn garment stays on: it is carried along with the
                    // skin it rests on and then left to settle.
                    Some(before) => {
                        body.carry_from(before, &mut self.topology.mesh_mut().positions);
                        self.topology.mesh_mut().recompute_normals();
                        if let Some(solver) = &mut self.solver {
                            solver.set_positions(self.topology.mesh());
                        }
                        self.garment_dirty = true;
                        self.settle_steps = SETTLE_STEPS;
                        // Keep the pieces' start positions in step with the
                        // body for the next reset, without resetting now.
                        avatar::arrange_around(&mut self.document, &body.mesh, &body.measurements);
                    }
                    // Still laid out flat: start over around the new body.
                    None => {
                        arrange = true;
                        self.editor.timeline.stop();
                    }
                }
            }
            // The collision field smooths over details smaller than the
            // cloth's triangles, so it follows the mesh resolution.
            body.set_closing(closing_radius(self.editor.simulation.particle_distance));
            body.poll_field();
        }

        for command in commands {
            match command {
                UiCommand::Undo => self.apply_history(History::undo, &mut changes),
                UiCommand::Redo => self.apply_history(History::redo, &mut changes),
                UiCommand::ResetDrape => changes.push(DocumentChange::Everything),
                UiCommand::StepSimulation => extra_steps += 1,
                UiCommand::FitCanvas => self.fit_canvas = true,
                UiCommand::ArrangePieces => arrange = true,
                UiCommand::ResetCamera3d => self.scene_camera = OrbitCamera::default(),
                UiCommand::AddOuterLayer => {
                    // Next to the existing pieces on the canvas.
                    let width = self
                        .document
                        .pieces()
                        .iter()
                        .map(|p| p.outline.bounds())
                        .fold(0.0f32, |width, (lo, hi)| width.max(hi.y - lo.y));
                    self.document.add_layer_copy(1.06, Vec2::new(0.0, -(width + 120.0)));
                    arrange = true;
                    self.fit_canvas = true;
                }
                UiCommand::ImportAvatar(path) => {
                    let closing = closing_radius(self.editor.simulation.particle_distance);
                    self.editor.io.status = Some(match io_cad::import_avatar(&path, &io_cad::ImportOptions::default()) {
                        Ok(mesh) => {
                            let message = format!(
                                "Imported {}: {} vertices, {} triangles, {:.2} m tall",
                                path.display(),
                                mesh.positions.len(),
                                mesh.triangles.len(),
                                mesh.bounds().1.y
                            );
                            self.imported = Some(ImportedAvatar::new(mesh, closing));
                            self.editor.io.avatar_imported = true;
                            self.editor.avatar.available = false;
                            self.avatar_dirty = true;
                            self.editor.timeline.stop();
                            arrange = true;
                            Ok(message)
                        }
                        Err(error) => Err(format!("Cannot import {}: {error}", path.display())),
                    });
                }
                UiCommand::RestoreAvatar => {
                    self.imported = None;
                    self.editor.io.avatar_imported = false;
                    self.editor.avatar.available = self.body.is_some();
                    self.avatar_dirty = true;
                    self.editor.timeline.stop();
                    arrange = true;
                }
                UiCommand::Export(path) => {
                    self.editor.io.status = Some(self.export(&path).map_err(|error| format!("Cannot write {}: {error}", path.display())));
                }
                UiCommand::InspectPatternFile(path) => {
                    self.inspect_pattern(&path);
                }
                UiCommand::ApplyImportedPattern { pattern, replace_project } => {
                    if replace_project {
                        self.document = pattern.to_document();
                        self.editor.selected_piece = self.document.pieces().first().map(|p| p.id);
                        self.editor.selected_edge = None;
                        self.editor.sewing_edges_a.clear();
                        self.editor.sewing_edges_b.clear();
                        self.editor.sewing_select_b = false;
                        self.editor.timeline.stop();
                        self.history = History::default();
                        arrange = true;
                        self.fit_canvas = true;
                        changes.push(DocumentChange::Everything);
                        let count = self.document.pieces().len();
                        self.editor.pattern_import.status = Some(Ok(format!("Created project with {count} piece(s)")));
                        self.editor.pattern_import.window_open = false;
                    } else {
                        let fabric = self
                            .document
                            .fabrics()
                            .first()
                            .map(|f| f.id)
                            .unwrap_or_else(|| self.document.add_fabric("Imported Fabric", [0.72, 0.75, 0.82]));
                        let added = pattern.add_to_document(&mut self.document, fabric);
                        self.editor.selected_piece = added.first().copied();
                        self.editor.selected_edge = None;
                        arrange = true;
                        self.fit_canvas = true;
                        changes.push(DocumentChange::Everything);
                        self.editor.pattern_import.status = Some(Ok(format!("Added {} piece(s) to project", added.len())));
                        self.editor.pattern_import.window_open = false;
                    }
                }
                UiCommand::Quit => self.quit_requested = true,
            }
        }

        if arrange {
            if let Some((mesh, measurements)) = self.avatar() {
                let (mesh, measurements) = (mesh.clone(), *measurements);
                avatar::arrange_around(&mut self.document, &mesh, &measurements);
            }
            // Every piece goes back to its arrangement, moved or not.
            changes.push(DocumentChange::Everything);
        }

        if let Some(input) = &canvas {
            if std::mem::take(&mut self.fit_canvas) {
                self.fit_canvas_to_document(Vec2::new(input.size[0] as f32, input.size[1] as f32));
            }
            self.canvas_tool.update(
                input,
                &mut self.canvas_camera,
                &mut self.document,
                &mut self.history,
                &mut self.editor,
                &mut changes,
            );
        }
        if let Some(input) = &scene {
            controllers::update_scene(input, &mut self.scene_camera, &self.topology, &mut self.editor);
        }

        // Particle distance switcher: a new resolution re-meshes everything.
        let wanted = self.editor.simulation.particle_distance.clamp(3.0, 100.0);
        if (wanted - self.topology.params().particle_distance).abs() > 1e-3 {
            self.topology.set_params(MeshingParams { particle_distance: wanted, ..self.topology.params() });
            changes.push(DocumentChange::Everything);
        }

        // 2D → 3D: re-mesh only what the edits invalidated.
        let report = self.topology.apply(&self.document, &changes);
        self.garment_dirty |= report.changed();

        let step = self.editor.timeline.step_seconds();
        let mut steps = self.editor.timeline.advance(dt) + extra_steps;
        // After the avatar moved, let the garment settle even when paused.
        if steps == 0 && self.settle_steps > 0 {
            steps = self.settle_steps.min(2);
        }
        self.settle_steps = self.settle_steps.saturating_sub(steps);
        // Not against a body that is about to be replaced: the cloth would
        // be pushed back to where the avatar no longer is.
        if self.imported.is_none() && self.body.as_ref().is_some_and(Body::field_pending) {
            self.settle_steps += steps.min(2);
            steps = 0;
        }
        let (field, collider) = match (&self.imported, &self.body) {
            (Some(imported), _) => (Some(&imported.field), Some(&imported.collider)),
            (None, Some(body)) => (body.field.as_ref(), body.collider.as_ref()),
            (None, None) => (None, None),
        };
        if let Some(solver) = &mut self.solver {
            let ctx = SolverContext {
                document: &self.document,
                settings: &self.editor.simulation.settings,
                colliders: &self.colliders,
                body: field,
                body_mesh: collider,
                ground: 0.0,
                dt: step,
            };
            // Fabric physics and grainlines feed the constraints without
            // touching the topology.
            let materials_changed = changes.iter().any(|change| {
                matches!(
                    change,
                    DocumentChange::Fabric(_) | DocumentChange::PieceFabric(_) | DocumentChange::PieceAnnotations(_)
                )
            });
            let rearranged = changes.iter().any(|change| matches!(change, DocumentChange::PieceArrangement(_)));
            if report.topology_changed || rearranged {
                // New vertices, or pieces moved back to their arrangement.
                solver.rebind(self.topology.mesh(), &ctx);
                self.draped = false;
                self.settle_steps = 0;
            } else if materials_changed {
                solver.refresh_materials(self.topology.mesh(), &ctx);
            }
            let mut moved = false;
            for _ in 0..steps {
                moved |= solver.step(self.topology.mesh_mut(), &ctx);
            }
            if moved {
                self.topology.mesh_mut().recompute_normals();
                self.garment_dirty = true;
                self.draped = true;
            }
            self.editor.simulation.stats = solver.stats();
        }

        // Technical views: re-measure whenever the garment or the view changed.
        let surface = self.editor.view.surface;
        if surface != self.overlay_shown || (self.garment_dirty && surface != SurfaceView::Fabric) {
            self.overlay_shown = surface;
            self.garment_dirty = true;
            let mesh = self.topology.mesh();
            match (surface, collider) {
                (SurfaceView::Fabric, _) => self.overlay_values.clear(),
                (SurfaceView::Strain, _) => topology_sync::strain_map(mesh, &mut self.overlay_values),
                // Twice the displayed range is far enough to look.
                (SurfaceView::Fit, Some(body)) => {
                    let reach = 2.0 * self.editor.view.fit_range_mm.max(1.0) * 1e-3;
                    topology_sync::clearance_map(mesh, body, reach, &mut self.overlay_values);
                }
                // No body mesh to measure against: everything is "far".
                (SurfaceView::Fit, None) => {
                    self.overlay_values.clear();
                    self.overlay_values.resize(mesh.vertex_count(), f32::MAX);
                }
            }
        }

        let mesh = self.topology.mesh();
        self.editor.stats.vertices = mesh.vertex_count();
        self.editor.stats.triangles = mesh.triangle_count();
        self.editor.stats.stitches = mesh.stitches.len();
        self.editor.undo_label = self.history.undo_label().map(str::to_owned);
        self.editor.redo_label = self.history.redo_label().map(str::to_owned);
    }

    fn apply_history(
        &mut self,
        step: fn(&mut History, &mut PatternDocument) -> Option<Result<DocumentChange, cad_core::EditError>>,
        changes: &mut Vec<DocumentChange>,
    ) {
        match step(&mut self.history, &mut self.document) {
            Some(Ok(change)) => changes.push(change),
            Some(Err(error)) => log::warn!("history step failed: {error}"),
            None => {}
        }
    }

    fn fit_canvas_to_document(&mut self, viewport: Vec2) {
        let bounds = self.document.pieces().iter().map(|p| p.outline.bounds()).reduce(|(lo, hi), (min, max)| {
            (lo.min(min), hi.max(max))
        });
        if let Some((min, max)) = bounds {
            self.canvas_camera.fit(min, max, viewport);
        }
    }

    /// The avatar garments are arranged around and shown on: the imported
    /// one if there is one, else the parametric body.
    pub fn avatar(&self) -> Option<(&avatar::BodyMesh, &avatar::Measurements)> {
        match (&self.imported, &self.body) {
            (Some(imported), _) => Some((&imported.mesh, &imported.measurements)),
            (None, Some(body)) => Some((&body.mesh, &body.measurements)),
            (None, None) => None,
        }
    }

    /// Per-vertex values for the garment's technical view (empty for the
    /// fabric view) and how the renderer should colour them.
    pub fn surface_overlay(&self) -> (SurfaceOverlay, &[f32]) {
        let view = &self.editor.view;
        let overlay = match view.surface {
            SurfaceView::Fabric => SurfaceOverlay::Fabric,
            SurfaceView::Strain => SurfaceOverlay::Strain { limit: view.strain_limit_percent / 100.0 },
            SurfaceView::Fit => SurfaceOverlay::Fit {
                // Lying on the skin means at the collision gap.
                tight: self.editor.simulation.settings.collision_margin,
                loose: view.fit_range_mm * 1e-3,
            },
        };
        (overlay, &self.overlay_values)
    }

    /// Writes the pattern or the draped garment in the format chosen in
    /// the import/export window. Returns a message for the user.
    fn export(&self, path: &std::path::Path) -> std::io::Result<String> {
        let io = &self.editor.io;
        let garment = io_cad::GarmentExport {
            mesh: self.topology.mesh(),
            document: &self.document,
            uv_layout: if io.pattern_uvs { io_cad::UvLayout::Pattern } else { io_cad::UvLayout::Fabric },
        };
        let dxf = |standard| {
            let options = io_cad::DxfOptions {
                standard,
                style_name: path.file_stem().and_then(|s| s.to_str()).unwrap_or("Untitled").to_owned(),
                sample_size: io.sample_size.clone(),
                ..Default::default()
            };
            io_cad::export_pattern_dxf(&self.document, &options)
        };
        match io.export_format {
            ExportFormat::DxfAama => std::fs::write(path, dxf(io_cad::DxfStandard::Aama))?,
            ExportFormat::DxfAstm => std::fs::write(path, dxf(io_cad::DxfStandard::Astm))?,
            ExportFormat::Fbx => std::fs::write(path, io_cad::export_garment_fbx(&garment, io_cad::FbxEncoding::Binary))?,
            ExportFormat::Obj => {
                let library = path.with_extension("mtl");
                let name = library.file_name().and_then(|n| n.to_str()).unwrap_or("garment.mtl");
                let (obj, mtl) = io_cad::export_garment_obj(&garment, name);
                std::fs::write(path, obj)?;
                std::fs::write(&library, mtl)?;
            }
        }
        Ok(match io.export_format {
            ExportFormat::DxfAama | ExportFormat::DxfAstm => {
                format!("Wrote {} ({} pieces)", path.display(), self.document.pieces().len())
            }
            ExportFormat::Obj | ExportFormat::Fbx => {
                format!("Wrote {} ({} triangles)", path.display(), self.topology.mesh().triangle_count())
            }
        })
    }

    fn inspect_pattern(&mut self, path: &std::path::Path) {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                self.editor.pattern_import.status = Some(Err(format!("Cannot read {}: {e}", path.display())));
                return;
            }
        };

        let mode = self.editor.pattern_import.scale_mode;
        let allowance = self.editor.pattern_import.default_allowance_mm;
        let custom = self.editor.pattern_import.custom_scale;
        let flip_y = self.editor.pattern_import.flip_y;

        let result = match ext.as_str() {
            "pdf" => {
                let scale = match mode {
                    app_ui::ScaleMode::Auto | app_ui::ScaleMode::Points72Dpi => io_cad::PT_TO_MM,
                    app_ui::ScaleMode::Millimetres => 1.0,
                    app_ui::ScaleMode::Pixels96Dpi => io_cad::CSS_PX_TO_MM,
                    app_ui::ScaleMode::Custom => custom,
                };
                let opts = io_cad::PdfOptions {
                    scale,
                    default_allowance_mm: allowance,
                };
                io_cad::import_pattern_pdf(&bytes, &opts)
            }
            "svg" => {
                let scale = match mode {
                    app_ui::ScaleMode::Auto => None,
                    app_ui::ScaleMode::Millimetres => Some(1.0),
                    app_ui::ScaleMode::Points72Dpi => Some(io_cad::PT_TO_MM),
                    app_ui::ScaleMode::Pixels96Dpi => Some(io_cad::CSS_PX_TO_MM),
                    app_ui::ScaleMode::Custom => Some(custom),
                };
                let opts = io_cad::SvgOptions {
                    scale,
                    flip_y,
                    default_allowance_mm: allowance,
                };
                io_cad::import_pattern_svg(&bytes, &opts)
            }
            other => Err(io_cad::IoError::UnsupportedFormat(other.to_owned())),
        };

        match result {
            Ok(pattern) => {
                let count = pattern.pieces.len();
                self.editor.pattern_import.status = Some(Ok(format!(
                    "Loaded {}: {} piece candidate(s) detected",
                    path.display(),
                    count
                )));
                self.editor.pattern_import.pattern = Some(pattern);
            }
            Err(e) => {
                self.editor.pattern_import.status = Some(Err(format!("Import failed: {e}")));
                self.editor.pattern_import.pattern = None;
            }
        }
    }

    /// `true` once after the garment mesh changed and must be re-uploaded.
    pub fn take_garment_dirty(&mut self) -> bool {
        std::mem::take(&mut self.garment_dirty)
    }

    /// `true` once after the avatar's shape changed and must be re-uploaded.
    pub fn take_avatar_dirty(&mut self) -> bool {
        std::mem::take(&mut self.avatar_dirty)
    }

    /// Whether the session needs continuous redraws.
    pub fn animating(&self) -> bool {
        self.editor.timeline.playing
            || self.settle_steps > 0
            || (self.imported.is_none() && self.body.as_ref().is_some_and(Body::field_pending))
    }

    pub fn canvas_overlay(&self) -> CanvasOverlay {
        CanvasOverlay {
            hovered: self.canvas_tool.hovered,
            active: self.canvas_tool.active(),
            selected_piece: self.editor.selected_piece,
            selected_edge: self.editor.selected_edge,
            sewing_edges_a: self.editor.sewing_edges_a.clone(),
            sewing_edges_b: self.editor.sewing_edges_b.clone(),
            show_grid: self.editor.view.show_grid,
            show_mesh: self.editor.view.show_mesh,
            show_seam_allowance: self.editor.view.show_seam_allowance,
        }
    }

    /// Viewport material of a piece: its fabric, highlighted when selected.
    pub fn material_of(&self, piece: PieceId) -> SurfaceMaterial {
        let fabric = self.document.piece(piece).and_then(|p| self.document.fabric(p.fabric));
        let mut material = fabric.map_or_else(SurfaceMaterial::default, |f| SurfaceMaterial {
            base_color: f.base_color,
            roughness: f.roughness,
            metallic: f.metallic,
            weave_strength: f.weave_strength,
            emissive: [0.0; 3],
        });
        if self.editor.selected_piece == Some(piece) {
            material.emissive = [0.10, 0.08, 0.03];
        }
        material
    }
}

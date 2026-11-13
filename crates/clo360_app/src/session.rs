//! The editing session: the document plus everything derived from it,
//! independent of windowing and GPU objects.

use app_ui::{EditorState, ExportFormat, SurfaceView, UiCommand, UiOutput};
use cad_core::{DocumentChange, History, PatternDocument, PieceId};
use gfx_render::{Camera2D, CanvasOverlay, OrbitCamera, SurfaceMaterial, SurfaceOverlay};
use glam::Vec2;
use topology_sync::{
    ClothSolver, Collider, GarmentTopology, MeshingParams, SolverBackend, SolverContext,
    SolverSettings, TopologySync,
};

use avatar::Measure;

use crate::body::{Body, ImportedAvatar};
use crate::controllers::{self, CanvasTool};

/// Radius the avatar's collision field is closed with for a cloth mesh of
/// the given particle distance (mm): half a triangle edge, within reason.
pub fn closing_radius(particle_distance_mm: f32) -> f32 {
    (0.5 * particle_distance_mm * 1e-3).clamp(0.002, 0.015)
}

/// What a garment simulation is for: the motion, the body's shape version
/// and the document's revision.
type GarmentKey = (avatar::Motion, u64, u64);

/// A garment being simulated: its messages, how far it is, and what for.
type GarmentBake = (
    std::sync::mpsc::Receiver<crate::garment_motion::Message>,
    f32,
    GarmentKey,
);

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
    /// The avatar's looks as last drawn.
    shown_appearance: avatar::Appearance,
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
    /// Piece names being read from drawn lettering, for the pattern in the
    /// import window.
    pending_names: Option<std::sync::mpsc::Receiver<io_cad::ImportedPattern>>,
    /// The worn garment held to the skin while an animation plays.
    attachments: Option<Vec<avatar::Attachment>>,
    /// The animation frame the body shows.
    animation_shown: Option<(avatar::Motion, f32)>,
    /// The motion played through on this body, and the shape it was for.
    performance: Option<(u64, avatar::Performance)>,
    /// The garment simulated through an animation, and one being simulated
    /// (its messages and how far it is).
    garment_motion: Option<crate::garment_motion::GarmentMotion>,
    garment_bake: Option<GarmentBake>,
    pub(crate) garment_bake_duration: Option<f32>,
    /// What the garment was last simulated for automatically, so a failed
    /// or superseded simulation is not started over and over.
    garment_auto: Option<GarmentKey>,
    /// A simulation that went wrong (overstretched cloth) and is not shown.
    garment_rejected: Option<GarmentKey>,
    /// Whether a worn garment is simulated through an animation by itself.
    pub(crate) auto_simulate: bool,
    /// The garment as worn before the animation started, to go back to.
    garment_at_rest: Option<Vec<glam::Vec3>>,
    /// The chair for sitting, for a body shape: `(shape version, min, max)`.
    chair: Option<(u64, glam::Vec3, glam::Vec3)>,
    scene_grab: Option<SceneGrab>,
    pub quit_requested: bool,
}

struct SceneGrab {
    vertex: u32,
    origin: glam::Vec3,
    normal: glam::Vec3,
}

impl Session {
    pub fn new(
        solver: Option<Box<dyn ClothSolver>>,
        body: Option<Body>,
        colliders: Vec<Collider>,
    ) -> Self {
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
            shown_appearance: avatar::Appearance::default(),
            solved_targets: [None; Measure::ALL.len()],
            fit_due: false,
            draped: false,
            settle_steps: 0,
            fit_canvas: true,
            garment_dirty: true,
            pending_names: None,
            attachments: None,
            animation_shown: None,
            performance: None,
            garment_motion: None,
            garment_bake: None,
            garment_bake_duration: None,
            garment_at_rest: None,
            garment_auto: None,
            garment_rejected: None,
            auto_simulate: true,
            chair: None,
            scene_grab: None,
            quit_requested: false,
        }
    }

    /// Applies one frame of UI output: commands, viewport tools, the
    /// 2D → 3D synchronisation and the simulation clock.
    pub fn update(&mut self, output: UiOutput, dt: f32) {
        let UiOutput {
            mut changes,
            commands,
            canvas,
            scene,
        } = output;
        let mut extra_steps = 0;
        let mut arrange = false;

        // --- avatar ------------------------------------------------------------
        // Looks are display only: just redraw.
        if self.editor.avatar.appearance != self.shown_appearance {
            self.shown_appearance = self.editor.avatar.appearance;
            self.avatar_dirty = true;
        }
        if let Some(imported) = &mut self.imported {
            imported.set_closing(closing_radius(self.editor.simulation.particle_distance));
        } else if let Some(body) = &mut self.body {
            let wanted = &mut self.editor.avatar;
            let mut rebuilt = false;
            let before = (self.draped
                && (wanted.params != body.params || wanted.pose != body.pose || self.fit_due))
                .then(|| body.snapshot());

            // Shape: the build sliders edit the parameters directly; held
            // tape measurements are then solved into them, all together.
            let targets: Vec<(Measure, f32)> = Measure::ALL
                .iter()
                .filter_map(|&m| Some((m, wanted.targets[m as usize]?)))
                .collect();
            let shape_edited =
                wanted.params != body.params || wanted.targets != self.solved_targets;
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

        self.animate(dt);
        self.collect_piece_names();

        for command in commands {
            match command {
                UiCommand::Undo => self.apply_history(History::undo, &mut changes),
                UiCommand::Redo => self.apply_history(History::redo, &mut changes),
                UiCommand::ResetDrape => changes.push(DocumentChange::Everything),
                UiCommand::StepSimulation => extra_steps += 1,
                UiCommand::FitCanvas => self.fit_canvas = true,
                UiCommand::ArrangePieces => arrange = true,
                UiCommand::ResetCamera3d => self.scene_camera = OrbitCamera::default(),
                UiCommand::TransformPiece(transform) => {
                    if let Some(id) = self.editor.selected_piece {
                        self.edit(
                            transform.label(),
                            |document| document.transform_piece(id, transform),
                            &mut changes,
                        );
                        arrange = true;
                    }
                }
                UiCommand::MirrorCopy => {
                    if let Some(id) = self.editor.selected_piece {
                        let mut copy = None;
                        self.edit(
                            "Mirrored copy",
                            |document| {
                                copy = document.duplicate_piece(id, true);
                                copy.is_some()
                            },
                            &mut changes,
                        );
                        if copy.is_some() {
                            self.editor.selected_piece = copy;
                            self.editor.selected_edge = None;
                            arrange = true;
                        }
                    }
                }
                UiCommand::DeletePiece => {
                    if let Some(id) = self.editor.selected_piece {
                        self.edit(
                            "Delete piece",
                            |document| document.remove_piece(id).is_some(),
                            &mut changes,
                        );
                        self.editor.selected_piece = None;
                        self.editor.selected_edge = None;
                        self.editor.sew_a = None;
                        self.editor.sew_b = None;
                    }
                }
                UiCommand::TidyLayout => {
                    self.edit(
                        "Tidy layout",
                        |document| {
                            document.tidy_layout(40.0);
                            true
                        },
                        &mut changes,
                    );
                    self.fit_canvas = true;
                }
                UiCommand::DeleteSelection => {
                    let seam = (self.editor.canvas_tool == app_ui::CanvasTool::Sew)
                        .then_some(self.editor.selected_edge)
                        .flatten()
                        .and_then(|edge| self.document.seam_of(edge))
                        .map(|seam| seam.id);
                    match seam {
                        Some(seam) => self.edit(
                            "Remove seam",
                            |document| document.remove_seam(seam).is_some(),
                            &mut changes,
                        ),
                        None => {
                            if let Some(id) = self.editor.selected_piece {
                                self.edit(
                                    "Delete piece",
                                    |document| document.remove_piece(id).is_some(),
                                    &mut changes,
                                );
                                self.editor.selected_piece = None;
                                self.editor.selected_edge = None;
                            }
                        }
                    }
                }
                UiCommand::SetPointKind {
                    piece,
                    anchor,
                    kind,
                } => {
                    let label = if kind == cad_core::PointKind::Smooth {
                        "Join lines"
                    } else {
                        "Split line"
                    };
                    self.edit(
                        label,
                        |document| {
                            let changed = document
                                .piece_mut(piece)
                                .is_some_and(|p| p.outline.set_point_kind(anchor, kind));
                            if changed {
                                document.touch();
                            }
                            changed
                        },
                        &mut changes,
                    );
                }
                UiCommand::SaveProject(path) => {
                    let avatar = self.body.as_ref().map(|body| {
                        crate::project::AvatarFile::new(
                            &body.params,
                            &body.pose,
                            &self.editor.avatar.targets,
                            &self.editor.avatar.appearance,
                        )
                    });
                    let project = crate::project::Project::new(
                        self.document.clone(),
                        avatar,
                        self.editor.simulation.particle_distance,
                    );
                    self.editor.project_status = Some(match project.save(&path) {
                        Ok(()) => {
                            self.editor.project_path = Some(path.clone());
                            Ok(format!("Saved {}", path.display()))
                        }
                        Err(error) => Err(format!("{error:#}")),
                    });
                }
                UiCommand::OpenProject(path) => match crate::project::Project::load(&path) {
                    Ok(project) => {
                        if let Some(avatar) = &project.avatar {
                            // Applied with the next frame's avatar update.
                            self.editor.avatar.params = avatar.params();
                            self.editor.avatar.pose = avatar.pose();
                            self.editor.avatar.targets = avatar.targets();
                            self.editor.avatar.appearance = avatar.appearance();
                        }
                        self.editor.simulation.particle_distance = project.particle_distance;
                        self.replace_document(project.document, &mut changes);
                        arrange = true;
                        self.editor.project_path = Some(path.clone());
                        self.editor.project_status = Some(Ok(format!("Opened {}", path.display())));
                    }
                    Err(error) => {
                        self.editor.project_status =
                            Some(Err(format!("Cannot open {}: {error:#}", path.display())))
                    }
                },
                UiCommand::SimulateGarmentMotion => self.simulate_garment_motion(),
                UiCommand::NewProject(sample) => {
                    self.replace_document(sample.document(), &mut changes);
                    arrange = true;
                    self.editor.project_path = None;
                    self.editor.project_status =
                        Some(Ok(format!("New project: {}", sample.label())));
                }
                UiCommand::DeleteSeam(seam) => {
                    self.edit(
                        "Remove seam",
                        |document| document.remove_seam(seam).is_some(),
                        &mut changes,
                    );
                }
                UiCommand::FlipSeam(seam) => {
                    self.edit(
                        "Flip seam",
                        |document| document.flip_seam(seam),
                        &mut changes,
                    );
                }
                UiCommand::AddOuterLayer => {
                    // Next to the existing pieces on the canvas.
                    let width = self
                        .document
                        .pieces()
                        .iter()
                        .map(|p| p.outline.bounds())
                        .fold(0.0f32, |width, (lo, hi)| width.max(hi.y - lo.y));
                    self.document
                        .add_layer_copy(1.06, Vec2::new(0.0, -(width + 120.0)));
                    arrange = true;
                    self.fit_canvas = true;
                }
                UiCommand::ImportAvatar(path) => {
                    let closing = closing_radius(self.editor.simulation.particle_distance);
                    self.editor.io.status = Some(
                        match io_cad::import_avatar(&path, &io_cad::ImportOptions::default()) {
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
                        },
                    );
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
                    self.editor.io.status = Some(
                        self.export(&path)
                            .map_err(|error| format!("Cannot write {}: {error}", path.display())),
                    );
                }
                UiCommand::InspectPatternFile(path) => {
                    self.inspect_pattern(&path);
                }
                UiCommand::ApplyImportedPattern {
                    pattern,
                    replace_project,
                } => {
                    if replace_project {
                        self.document = pattern.to_document();
                        self.editor.selected_piece = self.document.pieces().first().map(|p| p.id);
                        self.editor.selected_edge = None;
                        self.editor.sew_a = None;
                        self.editor.sew_b = None;
                        self.editor.timeline.stop();
                        self.history = History::default();
                        arrange = true;
                        self.fit_canvas = true;
                        changes.push(DocumentChange::Everything);
                        let count = self.document.pieces().len();
                        self.editor.pattern_import.status =
                            Some(Ok(format!("Created project with {count} piece(s)")));
                        self.editor.pattern_import.window_open = false;
                    } else {
                        let fabric = self
                            .document
                            .fabrics()
                            .first()
                            .map(|f| f.id)
                            .unwrap_or_else(|| {
                                self.document
                                    .add_fabric("Imported Fabric", [0.72, 0.75, 0.82])
                            });
                        let added = pattern.add_to_document(&mut self.document, fabric);
                        self.editor.selected_piece = added.first().copied();
                        self.editor.selected_edge = None;
                        arrange = true;
                        self.fit_canvas = true;
                        changes.push(DocumentChange::Everything);
                        self.editor.pattern_import.status =
                            Some(Ok(format!("Added {} piece(s) to project", added.len())));
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
            if self.editor.canvas_tool == app_ui::CanvasTool::Pull {
                self.update_scene_pull(input);
            } else if matches!(
                self.editor.canvas_tool,
                app_ui::CanvasTool::Button | app_ui::CanvasTool::Buttonhole
            ) {
                self.update_scene_annotation(input, &mut changes);
            } else {
                controllers::update_scene(
                    input,
                    &mut self.scene_camera,
                    &self.topology,
                    &mut self.editor,
                );
            }
        }

        // Particle distance switcher: a new resolution re-meshes everything.
        let wanted = self.editor.simulation.particle_distance.clamp(3.0, 100.0);
        if (wanted - self.topology.params().particle_distance).abs() > 1e-3 {
            self.topology.set_params(MeshingParams {
                particle_distance: wanted,
                ..self.topology.params()
            });
            changes.push(DocumentChange::Everything);
        }

        // 2D → 3D: re-mesh only what the edits invalidated.
        let report = self.topology.apply(&self.document, &changes);
        self.garment_dirty |= report.changed();

        let step = self.editor.timeline.step_seconds();
        let mut steps = self.editor.timeline.advance(dt) + extra_steps;
        if self.scene_grab.is_some() {
            steps = steps.max(1);
        }
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
                    DocumentChange::Fabric(_)
                        | DocumentChange::PieceFabric(_)
                        | DocumentChange::PieceAnnotations(_)
                )
            });
            let rearranged = changes
                .iter()
                .any(|change| matches!(change, DocumentChange::PieceArrangement(_)));
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
                (SurfaceView::Strain, _) => {
                    topology_sync::strain_map(mesh, &mut self.overlay_values)
                }
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

    /// Plays the avatar's animation: poses the body for this frame and
    /// carries a worn garment along with it.
    fn animate(&mut self, dt: f32) {
        if !self.draped {
            self.attachments = None;
            self.garment_at_rest = None;
        }
        self.poll_garment_motion();
        if self.animate_body(dt) {
            self.simulate_garment_motion();
        }
    }

    /// Poses the body for this frame and moves the garment with it. Returns
    /// whether the garment should now be simulated through the motion.
    fn animate_body(&mut self, dt: f32) -> bool {
        let (Some(body), None) = (&mut self.body, &self.imported) else {
            return false;
        };
        let state = &mut self.editor.avatar.animation;
        let follows = state.garment_follows;
        let revision = self.document.revision();
        let simulated =
            |motion: avatar::Motion,
             body: &Body,
             garment: &Option<crate::garment_motion::GarmentMotion>| {
                garment.as_ref().is_some_and(|g| {
                    g.motion == motion
                        && g.shape_version == body.shape_version()
                        && g.document_revision == revision
                })
            };
        state.garment = match (&self.garment_bake, state.motion) {
            (Some((_, done, _)), _) => app_ui::GarmentStatus::Simulating(*done),
            (None, Some(motion)) if simulated(motion, body, &self.garment_motion) => {
                app_ui::GarmentStatus::Simulated
            }
            (None, Some(motion))
                if self.garment_rejected == Some((motion, body.shape_version(), revision)) =>
            {
                app_ui::GarmentStatus::Unreliable
            }
            _ if !self.draped && self.garment_at_rest.is_none() => app_ui::GarmentStatus::NotWorn,
            _ => app_ui::GarmentStatus::Carried,
        };
        match state.motion {
            None => {
                if body.is_animated() {
                    body.set_animated(None);
                    self.animation_shown = None;
                    self.avatar_dirty = true;
                    // Back to the edited pose, and the garment as it was worn.
                    self.attachments = None;
                    if let Some(worn) = self.garment_at_rest.take() {
                        self.topology.mesh_mut().positions = worn;
                        self.topology.mesh_mut().recompute_normals();
                        if let Some(solver) = &mut self.solver {
                            solver.set_positions(self.topology.mesh());
                        }
                        self.garment_dirty = true;
                    }
                }
            }
            Some(motion) => {
                if state.playing {
                    state.time = (state.time + dt * state.speed).rem_euclid(motion.duration());
                }
                let frame = (motion, state.time);
                if self.animation_shown == Some(frame) && body.is_animated() {
                    return false;
                }
                if self.draped && self.garment_at_rest.is_none() {
                    self.garment_at_rest = Some(self.topology.mesh().positions.clone());
                }
                if self.draped && follows && self.attachments.is_none() {
                    // To the torso and the legs, never the arms: an arm raised
                    // over the head would haul the armholes up with it.
                    let not_arms: Vec<bool> = body
                        .model()
                        .arm_vertices()
                        .into_iter()
                        .map(|arm| !arm)
                        .collect();
                    self.attachments = Some(avatar::attach_to(
                        &self.topology.mesh().positions,
                        body.posed(),
                        Some(&not_arms),
                    ));
                }
                let current = self.performance.as_ref().is_some_and(|(version, p)| {
                    *version == body.shape_version() && p.motion() == motion
                });
                if !current {
                    self.performance = Some((
                        body.shape_version(),
                        avatar::Performance::bake(motion, body.model(), body.rest()),
                    ));
                }
                let pose = match &self.performance {
                    Some((_, performance)) => {
                        performance.pose(state.time, body.model(), body.rest())
                    }
                    None => motion.angles(state.time).to_pose(body.model(), body.rest()),
                };
                body.set_animated(Some(pose));
                self.animation_shown = Some(frame);
                self.avatar_dirty = true;
                if !follows {
                    return false;
                }
                // The garment moves with the animation now, not the drape.
                self.editor.timeline.playing = false;
                let is_simulated = simulated(motion, body, &self.garment_motion);
                // A worn garment is simulated through the motion by itself,
                // once for each motion, body and version of the garment.
                let key = (motion, body.shape_version(), revision);
                let simulate = self.auto_simulate
                    && !is_simulated
                    && self.garment_bake.is_none()
                    && self.garment_at_rest.is_some()
                    && self.garment_auto != Some(key);
                if simulate {
                    self.garment_auto = Some(key);
                }
                if is_simulated && let Some(garment) = &self.garment_motion {
                    // Played back as simulated.
                    garment.positions(state.time, &mut self.topology.mesh_mut().positions);
                } else if let Some(attachments) = &self.attachments {
                    // Until then, carried with the skin it rests on.
                    avatar::follow(
                        attachments,
                        body.posed(),
                        &mut self.topology.mesh_mut().positions,
                    );
                } else {
                    return simulate;
                }
                self.topology.mesh_mut().recompute_normals();
                if let Some(solver) = &mut self.solver {
                    solver.set_positions(self.topology.mesh());
                }
                self.garment_dirty = true;
                return simulate;
            }
        }
        false
    }

    /// Whether the garment has been simulated since it was arranged: sewn
    /// and worn rather than laid out.
    pub fn is_draped(&self) -> bool {
        self.draped
    }

    /// Starts simulating the worn garment through the current animation.
    fn simulate_garment_motion(&mut self) {
        let Some(motion) = self.editor.avatar.animation.motion else {
            return;
        };
        let (Some(body), None) = (&self.body, &self.imported) else {
            return;
        };
        // The same simulation already under way: let it finish.
        let key = (motion, body.shape_version(), self.document.revision());
        if self
            .garment_bake
            .as_ref()
            .is_some_and(|(_, _, running)| *running == key)
        {
            return;
        }
        self.garment_auto = Some(key);
        if !self.draped && self.garment_at_rest.is_none() {
            return;
        }
        // From the garment as worn in the edited pose.
        let mut mesh = self.topology.mesh().clone();
        if let Some(worn) = &self.garment_at_rest {
            mesh.positions = worn.clone();
            mesh.recompute_normals();
        }
        let job = crate::garment_motion::Job {
            motion,
            shape_version: body.shape_version(),
            document: self.document.clone(),
            mesh,
            model: body.model().clone(),
            rest: body.rest().clone(),
            start_pose: body.pose.to_pose(body.model(), body.rest()),
            // Animation is baked offline, so it always gets the contact
            // solver: cloth that lifts off the body, follows it through
            // friction and never passes through itself. The live view
            // keeps whichever solver runs in real time.
            settings: SolverSettings {
                backend: SolverBackend::Contact,
                ..self.editor.simulation.settings
            },
            closing: closing_radius(self.editor.simulation.particle_distance),
            duration: self.garment_bake_duration.take(),
        };
        self.garment_bake = Some((crate::garment_motion::start(job), 0.0, key));
    }

    /// Collects the progress and result of a garment being simulated.
    fn poll_garment_motion(&mut self) {
        let Some((receiver, done, _)) = &mut self.garment_bake else {
            return;
        };
        loop {
            match receiver.try_recv() {
                Ok(crate::garment_motion::Message::Progress(fraction)) => *done = fraction,
                Ok(crate::garment_motion::Message::Done(garment)) => {
                    let key = self.garment_bake.as_ref().map(|(_, _, key)| *key);
                    if garment.stretch > crate::garment_motion::RELIABLE_STRETCH {
                        log::warn!(
                            "the garment simulation through {:?} overstretched the cloth ({:.0} %); showing it carried with the skin",
                            garment.motion,
                            garment.stretch * 100.0
                        );
                        self.garment_rejected = key;
                        self.garment_bake = None;
                        return;
                    }
                    self.garment_motion = Some(garment);
                    self.garment_bake = None;
                    self.animation_shown = None; // show it now
                    return;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => return,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.garment_bake = None;
                    return;
                }
            }
        }
    }

    /// The chair of the sitting animation as `(min, max)` corners, risen
    /// out of the floor as far as it is at this moment; `None` when there is
    /// none. Placed under where the hips actually come to rest in the played
    /// motion, and kept clear of the body as it stands before sitting, so it
    /// rises behind the legs rather than through them.
    pub fn chair(&mut self) -> Option<(glam::Vec3, glam::Vec3)> {
        let state = self.editor.avatar.animation;
        let grow = state.motion?.chair(state.time)?;
        let body = self.body.as_ref().filter(|_| self.imported.is_none())?;
        let (_, performance) = self.performance.as_ref().filter(|(v, p)| {
            *v == body.shape_version() && p.motion() == avatar::Motion::SitOnChair
        })?;
        if self
            .chair
            .is_none_or(|(version, ..)| version != body.shape_version())
        {
            let (model, rest) = (body.model(), body.rest());
            let seated = model.pose(rest, &performance.pose(4.6, model, rest));
            let standing = model.pose(rest, &performance.pose(0.0, model, rest));
            let hips = ["upperleg01.L", "upperleg01.R"]
                .iter()
                .filter_map(|name| model.bone(name).map(|i| seated.joints[i].head))
                .sum::<glam::Vec3>()
                * 0.5;
            // The seat is where the buttocks and the backs of the thighs are lowest.
            let seat = seated
                .positions
                .iter()
                .filter(|p| p.x.abs() < 0.2 && (p.z - hips.z).abs() < 0.14 && p.y < hips.y)
                .map(|p| p.y)
                .fold(f32::MAX, f32::min);
            let seat = if seat < f32::MAX { seat - 0.004 } else { 0.45 };
            // The back of the standing body (buttocks, calves, heels) at the
            // chair's height: its front edge stays behind that.
            let back = standing
                .positions
                .iter()
                .filter(|p| p.x.abs() < 0.3 && p.y < seat + 0.05)
                .map(|p| p.z)
                .fold(f32::MAX, f32::min);
            let front = (hips.z + 0.16).min(back - 0.03);
            let depth = 0.42;
            self.chair = Some((
                body.shape_version(),
                glam::Vec3::new(-0.23, 0.0, front - depth),
                glam::Vec3::new(0.23, seat, front),
            ));
        }
        let (_, min, max) = self.chair?;
        (grow > 0.0).then(|| (min, glam::Vec3::new(max.x, max.y * grow, max.z)))
    }

    /// Starts over with `document`: nothing selected, no history, not draped.
    fn replace_document(&mut self, document: PatternDocument, changes: &mut Vec<DocumentChange>) {
        self.document = document;
        self.editor.selected_piece = None;
        self.editor.selected_edge = None;
        self.editor.sew_a = None;
        self.editor.sew_b = None;
        self.editor.timeline.stop();
        self.history = History::default();
        self.draped = false;
        self.fit_canvas = true;
        changes.push(DocumentChange::Everything);
    }

    /// Runs an edit with one undo step for all of it. `edit` returns
    /// whether it changed anything.
    fn edit(
        &mut self,
        label: &str,
        edit: impl FnOnce(&mut PatternDocument) -> bool,
        changes: &mut Vec<DocumentChange>,
    ) {
        let before = self.document.clone();
        if edit(&mut self.document) {
            self.history
                .record(Box::new(cad_core::DocumentSnapshot::new(
                    label,
                    before,
                    self.document.clone(),
                )));
            self.editor.timeline.stop();
            changes.push(DocumentChange::Everything);
        }
    }

    fn apply_history(
        &mut self,
        step: fn(
            &mut History,
            &mut PatternDocument,
        ) -> Option<Result<DocumentChange, cad_core::EditError>>,
        changes: &mut Vec<DocumentChange>,
    ) {
        match step(&mut self.history, &mut self.document) {
            Some(Ok(change)) => changes.push(change),
            Some(Err(error)) => log::warn!("history step failed: {error}"),
            None => {}
        }
    }

    fn fit_canvas_to_document(&mut self, viewport: Vec2) {
        let bounds = self
            .document
            .pieces()
            .iter()
            .map(|p| p.outline.bounds())
            .reduce(|(lo, hi), (min, max)| (lo.min(min), hi.max(max)));
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

    fn update_scene_pull(&mut self, input: &app_ui::ViewportInput) {
        let viewport = Vec2::new(input.size[0] as f32, input.size[1] as f32);
        if input.secondary.dragging || input.middle.dragging {
            self.scene_camera.pan(input.drag_delta, viewport.y);
        }

        if input.zoom != 1.0 {
            self.scene_camera.dolly(input.zoom);
        }
        let Some(pointer) = input.pointer else { return };
        let (origin, direction) = self.scene_camera.ray(pointer, viewport);
        if input.primary.started
            && self.scene_grab.is_none()
            && let Some((hit, point)) = controllers::surface_hit(&self.topology, origin, direction)
        {
            let triangle = self.topology.mesh().triangle(hit.triangle);
            let weights = [hit.barycentric.x, hit.barycentric.y, hit.barycentric.z];
            let vertex = triangle
                .into_iter()
                .zip(weights)
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(vertex, _)| vertex);
            if let Some(vertex) = vertex {
                self.scene_grab = Some(SceneGrab {
                    vertex,
                    origin: point,
                    normal: direction.normalize_or_zero(),
                });
            }
        }
        if let Some(grab) = &self.scene_grab {
            let denominator = direction.dot(grab.normal);
            if denominator.abs() > 1e-5 {
                let distance = (grab.origin - origin).dot(grab.normal) / denominator;
                if distance.is_finite() && distance >= 0.0 {
                    let target = origin + direction * distance;
                    if let Some(solver) = &mut self.solver {
                        // A broad, soft influence rearranges a patch of
                        // cloth instead of pinching one particle.
                        // Keep the grab broad and low-force. This changes
                        // only the interaction, not the textile model or
                        // its spring parameters.
                        solver.set_grab(grab.vertex, target, 0.30, 0.08);
                        self.garment_dirty = true;
                    } else {
                        self.garment_dirty = true;
                    }
                }
            }
            if input.primary.stopped {
                if self.scene_grab.take().is_some() {
                    if let Some(solver) = &mut self.solver {
                        solver.clear_grab();
                        solver.set_positions(self.topology.mesh());
                    }
                    self.topology.mesh_mut().recompute_normals();
                    self.garment_dirty = true;
                }
            }
        }
    }

    fn update_scene_annotation(
        &mut self,
        input: &app_ui::ViewportInput,
        changes: &mut Vec<DocumentChange>,
    ) {
        if !input.clicked {
            return;
        }
        let viewport = Vec2::new(input.size[0] as f32, input.size[1] as f32);
        let Some(pointer) = input.pointer else { return };
        let (origin, direction) = self.scene_camera.ray(pointer, viewport);
        let Some((location, _)) = controllers::surface_hit(&self.topology, origin, direction)
        else {
            return;
        };
        let Some(pattern) = self.topology.surface_to_pattern(location) else {
            return;
        };
        let before = self.document.clone();
        let Some(piece) = self.document.piece_mut(pattern.piece) else {
            return;
        };
        match self.editor.canvas_tool {
            app_ui::CanvasTool::Button => piece
                .buttons
                .push(cad_core::Button::new(pattern.position, 18.0)),
            app_ui::CanvasTool::Buttonhole => {
                piece
                    .buttonholes
                    .push(cad_core::Buttonhole::for_button(pattern.position, 18.0));
            }
            _ => return,
        }
        self.document.touch();
        self.editor.selected_piece = Some(pattern.piece);
        self.history
            .record(Box::new(cad_core::DocumentSnapshot::new(
                if self.editor.canvas_tool == app_ui::CanvasTool::Button {
                    "Add button"
                } else {
                    "Add buttonhole"
                },
                before,
                self.document.clone(),
            )));
        changes.push(DocumentChange::PieceAnnotations(pattern.piece));
        self.garment_dirty = true;
    }

    /// Per-vertex values for the garment's technical view (empty for the
    /// fabric view) and how the renderer should colour them.
    pub fn surface_overlay(&self) -> (SurfaceOverlay, &[f32]) {
        let view = &self.editor.view;
        let overlay = match view.surface {
            SurfaceView::Fabric => SurfaceOverlay::Fabric,
            SurfaceView::Strain => SurfaceOverlay::Strain {
                limit: view.strain_limit_percent / 100.0,
            },
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
            uv_layout: if io.pattern_uvs {
                io_cad::UvLayout::Pattern
            } else {
                io_cad::UvLayout::Fabric
            },
        };
        let dxf = |standard| {
            let options = io_cad::DxfOptions {
                standard,
                style_name: path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("Untitled")
                    .to_owned(),
                sample_size: io.sample_size.clone(),
                ..Default::default()
            };
            io_cad::export_pattern_dxf(&self.document, &options)
        };
        match io.export_format {
            ExportFormat::DxfAama => std::fs::write(path, dxf(io_cad::DxfStandard::Aama))?,
            ExportFormat::DxfAstm => std::fs::write(path, dxf(io_cad::DxfStandard::Astm))?,
            ExportFormat::Fbx => std::fs::write(
                path,
                io_cad::export_garment_fbx(&garment, io_cad::FbxEncoding::Binary),
            )?,
            ExportFormat::Obj => {
                let library = path.with_extension("mtl");
                let name = library
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("garment.mtl");
                let (obj, mtl) = io_cad::export_garment_obj(&garment, name);
                std::fs::write(path, obj)?;
                std::fs::write(&library, mtl)?;
            }
        }
        Ok(match io.export_format {
            ExportFormat::DxfAama | ExportFormat::DxfAstm => {
                format!(
                    "Wrote {} ({} pieces)",
                    path.display(),
                    self.document.pieces().len()
                )
            }
            ExportFormat::Obj | ExportFormat::Fbx => {
                format!(
                    "Wrote {} ({} triangles)",
                    path.display(),
                    self.topology.mesh().triangle_count()
                )
            }
        })
    }

    /// Takes over piece names read in the background, for pieces the user
    /// has not renamed meanwhile.
    fn collect_piece_names(&mut self) {
        let Some(receiver) = &self.pending_names else {
            return;
        };
        let named = match receiver.try_recv() {
            Ok(named) => named,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.pending_names = None;
                return;
            }
        };
        self.pending_names = None;
        let import = &mut self.editor.pattern_import;
        let Some(pattern) = &mut import.pattern else {
            return;
        };
        let mut count = 0;
        for (piece, read) in pattern.pieces.iter_mut().zip(&named.pieces) {
            if piece.name.starts_with("Piece ") && !read.name.starts_with("Piece ") {
                piece.name = read.name.clone();
                count += 1;
            }
        }
        let note = format!("{count} piece name(s) read from the pattern");
        import.status = Some(Ok(match import.status.take() {
            Some(Ok(message)) => format!(
                "{} · {note}",
                message.split(" · reading").next().unwrap_or(&message)
            ),
            _ => note,
        }));
    }

    /// Whether background work needs the UI to keep redrawing.
    pub fn busy(&self) -> bool {
        self.pending_names.is_some()
    }

    fn inspect_pattern(&mut self, path: &std::path::Path) {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                self.editor.pattern_import.status =
                    Some(Err(format!("Cannot read {}: {e}", path.display())));
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
                let unnamed = pattern
                    .pieces
                    .iter()
                    .filter(|p| p.name.starts_with("Piece "))
                    .count();
                let mut message = format!("Loaded {}: {count} piece(s)", path.display());
                if unnamed > 0 {
                    // Names drawn as pen strokes need text recognition,
                    // which takes a moment: done in the background.
                    match io_cad::drawn_text::TesseractCli::find() {
                        Some(reader) => {
                            let (sender, receiver) = std::sync::mpsc::channel();
                            let mut named = pattern.clone();
                            std::thread::spawn(move || {
                                io_cad::drawn_text::name_pieces(&mut named, &reader);
                                let _ = sender.send(named);
                            });
                            self.pending_names = Some(receiver);
                            message.push_str(" · reading piece names drawn on the pattern…");
                        }
                        None => message.push_str(
                            " · piece names are drawn as lines in this file; install Tesseract OCR \
                             (e.g. `sudo apt install tesseract-ocr`) to read them",
                        ),
                    }
                }
                self.editor.pattern_import.status = Some(Ok(message));
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
            || (self.editor.avatar.animation.playing
                && self.editor.avatar.animation.motion.is_some())
            || self.garment_bake.is_some()
            || self.busy()
            || self.settle_steps > 0
            || (self.imported.is_none() && self.body.as_ref().is_some_and(Body::field_pending))
    }

    pub fn canvas_overlay(&self) -> CanvasOverlay {
        use app_ui::CanvasTool;
        use gfx_render::LineHighlight;
        let editor = &self.editor;
        let mut highlights = Vec::new();
        let line_of = |edge: cad_core::EdgeRef| {
            self.document
                .piece(edge.piece)
                .map(|piece| (edge.piece, piece.outline.line_span(edge.edge)))
        };
        if let Some((piece, span)) = self.canvas_tool.hovered_line
            && editor.canvas_tool != CanvasTool::Points
        {
            highlights.push(LineHighlight {
                piece,
                span,
                color: gfx_render::srgba(255, 255, 255, 0.35),
                start_at_end: None,
            });
        }
        if let Some((piece, span)) = editor.selected_edge.and_then(line_of) {
            highlights.push(LineHighlight {
                piece,
                span,
                color: gfx_render::srgba(255, 185, 40, 1.0),
                start_at_end: None,
            });
        }
        // The sides being sewn, with the end each seam starts at.
        for (pick, color) in [
            (&editor.sew_a, gfx_render::srgba(64, 210, 135, 1.0)),
            (&editor.sew_b, gfx_render::srgba(70, 180, 245, 1.0)),
        ] {
            if let Some(pick) = pick
                && let Some(piece) = self.document.piece(pick.piece)
            {
                let at_end = piece.outline.span_fraction(pick.span, pick.at) > 0.5;
                highlights.push(LineHighlight {
                    piece: pick.piece,
                    span: pick.span,
                    color,
                    start_at_end: Some(at_end),
                });
            }
        }
        CanvasOverlay {
            hovered: self.canvas_tool.hovered,
            active: self.canvas_tool.active(),
            selected_piece: editor.selected_piece,
            handles_for: editor
                .selected_piece
                .filter(|_| editor.canvas_tool == CanvasTool::Points),
            highlights,
            show_grid: editor.view.show_grid,
            show_mesh: editor.view.show_mesh,
            show_seam_allowance: editor.view.show_seam_allowance,
            sketch_points: editor.sketch_points.clone(),
        }
    }

    /// How thick a piece's fabric is drawn, in metres: its thickness, but
    /// at least enough to be seen at the edges.
    pub fn fabric_thickness(&self, piece: PieceId) -> f32 {
        let fabric = self
            .document
            .piece(piece)
            .and_then(|p| self.document.fabric(p.fabric));
        fabric.map_or(0.6, |f| f.physics.thickness).max(0.8) * 1e-3
    }

    /// Viewport material of a piece: its fabric, highlighted when selected.
    pub fn material_of(&self, piece: PieceId) -> SurfaceMaterial {
        let fabric = self
            .document
            .piece(piece)
            .and_then(|p| self.document.fabric(p.fabric));
        let mut material = fabric.map_or_else(SurfaceMaterial::default, |f| SurfaceMaterial {
            base_color: f.base_color,
            roughness: f.roughness,
            metallic: f.metallic,
            weave_strength: f.weave_strength,
            pattern: f.pattern as u8 as f32,
            wash: f.wash,
            pattern_scale: f.pattern_scale,
            relief: f.relief,
            ..SurfaceMaterial::default()
        });
        if self.editor.selected_piece == Some(piece) {
            material.emissive = [0.10, 0.08, 0.03];
        }
        material
    }
}

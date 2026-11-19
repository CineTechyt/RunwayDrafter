//! The editing session: the document plus everything derived from it,
//! independent of windowing and GPU objects.

use app_ui::{EditorState, ExportFormat, SurfaceView, UiCommand, UiOutput};
use cad_core::{DocumentChange, History, PatternDocument, PieceId};
use gfx_render::{Camera2D, CanvasOverlay, OrbitCamera, SurfaceMaterial, SurfaceOverlay};
use glam::{Vec2, Vec3};
use topology_sync::{
    ClothSolver, Collider, GarmentTopology, MeshingParams, SolverBackend,
    SolverSettings, TopologySync,
};

use avatar::Measure;

use crate::body::{Body, ImportedAvatar};
use crate::simulation::{Scene, Simulation};

/// What decides whether the simulation needs the scene again.
#[derive(Debug, Clone, Copy, PartialEq)]
struct SceneKey {
    revision: u64,
    settings: SolverSettings,
    body: usize,
    body_mesh: usize,
    colliders: usize,
    dt: f32,
}
use crate::controllers::{self, CanvasTool};

/// How long the machine takes to sew one row.
const SEW_TIME: std::time::Duration = std::time::Duration::from_millis(1600);

/// The garment's stitches and how far along each row is being sewn.
pub struct Sewing {
    stitches: gfx_render::Stitches,
    built: Option<(u64, u64)>,
    started: std::collections::HashMap<u64, std::time::Instant>,
    /// Rows found after the first build are new: sewn while you watch.
    primed: bool,
    epoch: std::time::Instant,
}

impl Default for Sewing {
    fn default() -> Self {
        Self {
            stitches: Default::default(),
            built: None,
            started: Default::default(),
            primed: false,
            epoch: std::time::Instant::now(),
        }
    }
}

impl Sewing {
    /// A row is still being sewn.
    pub fn busy(&self) -> bool {
        let now = std::time::Instant::now();
        self.started.values().any(|t| now.duration_since(*t) < SEW_TIME)
    }

    /// Everything there now counts as sewn already (an opened project).
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Image maps are loaded at most this many pixels across.
const MAP_PIXELS: u32 = 2048;
/// Pixels per thread of the woven fabric textures.
pub const WEAVE_PIXELS: u32 = 8;

/// How far round the grabbed point the pull tool moves the cloth, metres.
/// A handful: the middle of it is held and goes with the cursor, the
/// cloth around follows by itself.
const GRAB_RADIUS: f32 = 0.06;
/// How much of the way to the cursor the grabbed point goes per step.
const GRAB_STRENGTH: f32 = 1.0;

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
    /// The stitches pinned to the garment (for this topology and document
    /// revision), and when each row started being sewn.
    pub sewing: Sewing,
    /// Image maps that could not be loaded (by key): not tried again.
    failed_maps: std::cell::RefCell<std::collections::HashSet<u64>>,
    /// Image maps being decoded off the interface's thread (large scans
    /// take seconds), by key.
    loading_maps: std::cell::RefCell<
        std::collections::HashMap<u64, std::sync::mpsc::Receiver<Result<cad_core::WeaveTexture, String>>>,
    >,
    pub document: PatternDocument,
    pub topology: GarmentTopology,
    pub history: History,
    pub editor: EditorState,
    pub canvas_camera: Camera2D,
    pub scene_camera: OrbitCamera,
    /// How far the 3D view has been moved along with a walking avatar.
    camera_follow: Vec3,
    canvas_tool: CanvasTool,
    /// The live cloth simulation; `None` runs the editor without it.
    simulation: Option<Simulation>,
    /// What the simulation was last told about the scene.
    scene_sent: Option<SceneKey>,
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
    /// Reads the character library again for the Avatar window.
    pub fn refresh_characters(&mut self) {
        self.editor.avatar.characters = crate::characters::folder()
            .map(|folder| crate::characters::list(&folder).iter().map(|c| c.entry()).collect())
            .unwrap_or_default();
    }

    /// `threaded`: the simulation steps on its own thread (the editor);
    /// otherwise inline with the frames (headless runs, tests).
    pub fn new(
        solver: Option<Box<dyn ClothSolver + Send>>,
        body: Option<Body>,
        colliders: Vec<Collider>,
        threaded: bool,
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
        let simulation = solver.map(|solver| {
            let mut simulation = Simulation::new(solver, threaded);
            simulation.rebind(
                topology.mesh(),
                Scene {
                    document: document.clone(),
                    settings: editor.simulation.settings,
                    colliders: colliders.clone(),
                    body: body.as_ref().and_then(|b| b.field.clone()),
                    body_mesh: body.as_ref().and_then(|b| b.collider.clone()),
                    dt: editor.timeline.step_seconds(),
                },
            );
            simulation
        });
        Self {
            sewing: Sewing::default(),
            failed_maps: Default::default(),
            loading_maps: Default::default(),
            document,
            topology,
            history: History::default(),
            editor,
            canvas_camera: Camera2D::default(),
            scene_camera: OrbitCamera::default(),
            camera_follow: Vec3::ZERO,
            canvas_tool: CanvasTool::default(),
            simulation,
            scene_sent: None,
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
                        if let Some(simulation) = &mut self.simulation {
                            simulation.set_positions(self.topology.mesh());
                        }
                        self.garment_dirty = true;
                        self.settle_steps = SETTLE_STEPS;
                        // The body moved under the garment: it drapes on
                        // until it has settled on the new shape.
                        self.editor.timeline.playing = true;
                        // Keep the pieces' start positions in step with the
                        // body for the next reset, without resetting now.
                        avatar::arrange_around(&mut self.document, &body.mesh, &body.measurements);
                    }
                    // Still laid out flat: start over around the new body
                    // (a drape already playing starts again from there).
                    None => {
                        arrange = true;
                        self.editor.timeline.rewind();
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
                UiCommand::ResetCamera3d => {
                    self.scene_camera = OrbitCamera::default();
                    self.scene_camera.target += self.camera_follow;
                }
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
                    let seam = matches!(
                        self.editor.canvas_tool,
                        app_ui::CanvasTool::Sew | app_ui::CanvasTool::FlatFell | app_ui::CanvasTool::Zipper
                    )
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
                UiCommand::SplitLine { piece, edge, parts } => {
                    self.edit(
                        if parts == 2 { "Cut line in half" } else { "Split line" },
                        |document| controllers::split_line(document, piece, edge, parts),
                        &mut changes,
                    );
                }
                UiCommand::UnfoldPiece { piece, edge } => {
                    self.editor.selected_edge = None;
                    self.edit(
                        "Unfold piece",
                        |document| document.unfold_piece(piece, edge),
                        &mut changes,
                    );
                }
                UiCommand::SetLineLength {
                    piece,
                    edge,
                    length,
                } => {
                    self.edit(
                        "Line length",
                        |document| {
                            let changed = document.piece_mut(piece).is_some_and(|p| {
                                let span = p.outline.line_span(edge);
                                p.outline.set_span_length(span, length)
                            });
                            if changed {
                                document.touch();
                            }
                            changed
                        },
                        &mut changes,
                    );
                }
                UiCommand::SetLineAngle {
                    piece,
                    edge,
                    degrees,
                } => {
                    self.edit(
                        "Line angle",
                        |document| {
                            let changed = document.piece_mut(piece).is_some_and(|p| {
                                let span = p.outline.line_span(edge);
                                p.outline.set_span_angle(span, degrees)
                            });
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
                UiCommand::SaveCharacter(name) => {
                    let name = name.trim().to_owned();
                    let result = match (&self.body, crate::characters::folder()) {
                        (None, _) => Err(anyhow::anyhow!("only the built-in avatar can be saved")),
                        (_, None) => Err(anyhow::anyhow!("no home folder for the character library")),
                        (Some(body), Some(folder)) => {
                            let avatar = crate::project::AvatarFile::new(
                                &body.params,
                                &body.pose,
                                &self.editor.avatar.targets,
                                &self.editor.avatar.appearance,
                            );
                            let height = self.editor.avatar.measurements.get(avatar::Measure::Height);
                            crate::characters::save(&folder, &crate::characters::Character::new(&name, height, avatar))
                        }
                    };
                    self.editor.avatar.character_status = Some(match result {
                        Ok(()) => {
                            self.editor.avatar.character = Some(name.clone());
                            Ok(format!("Saved \u{201c}{name}\u{201d}"))
                        }
                        Err(error) => Err(format!("{error:#}")),
                    });
                    self.refresh_characters();
                }
                UiCommand::DeleteCharacter(name) => {
                    let result = crate::characters::folder()
                        .ok_or_else(|| anyhow::anyhow!("no character library"))
                        .and_then(|folder| crate::characters::delete(&folder, &name));
                    self.editor.avatar.character_status = Some(match result {
                        Ok(()) => Ok(format!("Deleted \u{201c}{name}\u{201d}")),
                        Err(error) => Err(format!("{error:#}")),
                    });
                    if self.editor.avatar.character.as_deref() == Some(name.as_str()) {
                        self.editor.avatar.character = None;
                    }
                    self.refresh_characters();
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
                    let document = match (sample, self.avatar()) {
                        // Drafted to the avatar's own measurements.
                        (app_ui::Sample::Jeans, Some((_, m))) => {
                            let mm = |v: f32, fallback: f32| if v > 0.0 { v * 1000.0 } else { fallback };
                            let base = cad_core::jeans::JeansSize::default();
                            let inseam = mm(m.get(avatar::Measure::Inseam), base.inseam);
                            cad_core::jeans::jeans(cad_core::jeans::JeansSize {
                                waist: mm(m.get(avatar::Measure::Waist), base.waist),
                                hips: mm(m.get(avatar::Measure::Hips), base.hips),
                                thigh: mm(m.get(avatar::Measure::Thigh), base.thigh),
                                inseam,
                                rise: (m.waist_level * 1000.0 - inseam).clamp(180.0, 380.0),
                            })
                        }
                        _ => sample.document(),
                    };
                    let mut document = document;
                    if sample == app_ui::Sample::Jeans {
                        // Drafted on top of each other: laid out side by side.
                        document.tidy_layout(40.0);
                    }
                    self.replace_document(document, &mut changes);
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
                UiCommand::AddDart {
                    piece,
                    edge,
                    t,
                    width,
                    length,
                } => {
                    self.edit(
                        "Add dart",
                        |document| document.add_dart(piece, edge, t, width, length).is_some(),
                        &mut changes,
                    );
                }
                UiCommand::AddPleat {
                    piece,
                    edge,
                    t,
                    intake,
                    length,
                } => {
                    self.edit(
                        "Add pleat",
                        |document| document.add_pleat(piece, edge, t, intake, length).is_some(),
                        &mut changes,
                    );
                }
                UiCommand::SetFolds { piece, folds } => {
                    self.edit(
                        "Fold lines",
                        |document| {
                            let Some(p) = document.piece_mut(piece) else {
                                return false;
                            };
                            p.folds = folds;
                            document.touch();
                            true
                        },
                        &mut changes,
                    );
                }
                UiCommand::SetSeamZipper(seam, zipper) => {
                    // Opening or closing it changes what holds the cloth.
                    self.edit(
                        "Zipper",
                        |document| document.set_seam_zipper(seam, zipper),
                        &mut changes,
                    );
                }
                UiCommand::SetSeamStitching(seam, stitching) => {
                    self.restitch("Stitch", |document| document.set_seam_stitching(seam, stitching));
                }
                UiCommand::SetEdgeStitching {
                    piece,
                    span,
                    stitching,
                } => {
                    self.restitch("Stitch edge", |document| {
                        document.set_edge_stitching(piece, span, stitching)
                    });
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
        self.editor.canvas_view = app_ui::CanvasView {
            center: self.canvas_camera.center,
            zoom: self.canvas_camera.zoom,
        };
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
        let threaded = self.simulation.as_ref().is_some_and(Simulation::threaded);
        // Inline, the clock decides how many steps run this frame; threaded,
        // the simulation counts its own steps (see below).
        let mut steps = 0;
        if !threaded {
            steps = self.editor.timeline.advance(dt) + extra_steps;
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
        }
        let (field, collider) = match (&self.imported, &self.body) {
            (Some(imported), _) => (Some(&imported.field), Some(&imported.collider)),
            (None, Some(body)) => (body.field.as_ref(), body.collider.as_ref()),
            (None, None) => (None, None),
        };
        let field_pending =
            self.imported.is_none() && self.body.as_ref().is_some_and(Body::field_pending);
        if self.simulation.is_some() {
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
            let key = SceneKey {
                revision: self.document.revision(),
                settings: self.editor.simulation.settings,
                body: field.map_or(0, |f| std::sync::Arc::as_ptr(f) as usize),
                body_mesh: collider.map_or(0, |c| std::sync::Arc::as_ptr(c) as usize),
                colliders: self.colliders.len(),
                dt: step,
            };
            let rebind = report.topology_changed || rearranged;
            let scene = (rebind || materials_changed || self.scene_sent != Some(key)).then(|| Scene {
                document: self.document.clone(),
                settings: self.editor.simulation.settings,
                colliders: self.colliders.clone(),
                body: field.cloned(),
                body_mesh: collider.cloned(),
                dt: step,
            });
            let simulation = self.simulation.as_mut().expect("checked");
            if let Some(scene) = scene {
                if rebind {
                    // New vertices, or pieces moved back to their arrangement.
                    simulation.rebind(self.topology.mesh(), scene);
                    self.draped = false;
                    self.settle_steps = 0;
                } else if materials_changed {
                    simulation.refresh_materials(scene);
                } else {
                    simulation.set_scene(scene);
                }
                self.scene_sent = Some(key);
            }

            let animating = self.editor.avatar.animation.motion.is_some();
            let progress = if simulation.threaded() {
                // It steps by itself while there is something to do; the
                // view goes on drawing meanwhile.
                self.settle_steps += extra_steps;
                let wanted = (self.editor.timeline.playing
                    || self.settle_steps > 0
                    || self.scene_grab.is_some())
                    && !field_pending
                    && !animating;
                simulation.run(wanted);
                simulation.poll(0).filter(|_| !animating)
            } else {
                simulation.poll(steps)
            };
            if let Some(progress) = progress {
                if progress.positions.len() == self.topology.mesh().vertex_count() {
                    self.topology.mesh_mut().positions = progress.positions;
                }
                if progress.moved {
                    self.topology.mesh_mut().recompute_normals();
                    self.garment_dirty = true;
                    self.draped = true;
                }
                if simulation.threaded() {
                    let timeline = &mut self.editor.timeline;
                    if timeline.playing {
                        timeline.frame += progress.steps;
                        if timeline.frame >= timeline.end_frame && !timeline.looping {
                            timeline.playing = false;
                        }
                    }
                    self.settle_steps = self.settle_steps.saturating_sub(progress.steps);
                }
                if progress.settled {
                    // At rest: nothing left to simulate until something changes.
                    self.editor.timeline.playing = false;
                    self.settle_steps = 0;
                }
                self.editor.simulation.stats = progress.stats;
            }
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
        self.follow_with_camera();
    }

    /// While a walk plays, the 3D view travels with the avatar: across the
    /// floor only, never up or down, and keeping whatever the user orbited,
    /// panned or zoomed to. When the walk starts over a stride back, the
    /// view goes back with it, so the walk looks endless.
    fn follow_with_camera(&mut self) {
        let walking = self
            .editor
            .avatar
            .animation
            .motion
            .is_some_and(|m| m.travel() != Vec3::ZERO);
        let wanted = match (&self.body, &self.imported, walking) {
            (Some(body), None, true) => body
                .model()
                .bone("root")
                .and_then(|i| body.mesh.joints.get(i))
                .map_or(Vec3::ZERO, |root| Vec3::new(root.head.x, 0.0, root.head.z)),
            _ => Vec3::ZERO,
        };
        self.scene_camera.target += wanted - self.camera_follow;
        self.camera_follow = wanted;
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
                        if let Some(simulation) = &mut self.simulation {
                            simulation.set_positions(self.topology.mesh());
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
                if let Some(simulation) = &mut self.simulation {
                    simulation.set_positions(self.topology.mesh());
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
        if let Some((_, m)) = self.avatar() {
            log::info!(
                "avatar: waist {:.0} mm at {:.0} mm, hips {:.0} mm at {:.0} mm, thigh {:.0} mm, inseam {:.0} mm",
                m.get(avatar::Measure::Waist) * 1000.0,
                m.waist_level * 1000.0,
                m.get(avatar::Measure::Hips) * 1000.0,
                m.hip_level * 1000.0,
                m.get(avatar::Measure::Thigh) * 1000.0,
                m.get(avatar::Measure::Inseam) * 1000.0
            );
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
            // Animation is baked offline with the global solver, whichever
            // solver the live view runs: on a walk it is about twenty times
            // faster than the contact solver (one big step per frame
            // instead of many small ones) and keeps the fabric's length
            // better. It has no cloth-on-cloth collision yet.
            settings: SolverSettings {
                backend: SolverBackend::Global,
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
                            "the garment simulation through {:?} stretched the cloth {:.0} % further than as draped; showing it carried with the skin",
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
    /// The stitches on the garment as it lies now, with the needles of rows
    /// being sewn, appended to `props`. True while something is being sewn.
    pub fn stitch_props(&mut self, props: &mut gfx_render::MeshData) -> bool {
        let mesh = self.topology.mesh();
        let key = (mesh.generation, self.document.revision());
        if self.sewing.built != Some(key) {
            self.sewing.built = Some(key);
            self.sewing.stitches = gfx_render::Stitches::build(&self.document, &self.topology);
            let now = std::time::Instant::now();
            let primed = self.sewing.primed;
            for row in self.sewing.stitches.rows() {
                // Rows there from the start (an opened project) are sewn
                // already; new ones are sewn now.
                self.sewing
                    .started
                    .entry(row)
                    .or_insert(if primed { now } else { now - SEW_TIME * 2 });
            }
            self.sewing.primed = true;
        }
        let now = std::time::Instant::now();
        let started = &self.sewing.started;
        let sewn = |row: u64| {
            started.get(&row).map_or(1.0, |t| {
                (now.duration_since(*t).as_secs_f32() / SEW_TIME.as_secs_f32()).clamp(0.0, 1.0)
            })
        };
        let needles = self.sewing.stitches.mesh(self.topology.mesh(), sewn, props);
        // The needle goes up and down about twelve times a second.
        let phase = (now.duration_since(self.sewing.epoch).as_secs_f32() * 12.0).fract();
        for needle in &needles {
            gfx_render::scene3d::stitches::needle_model(props, needle, phase);
        }
        !needles.is_empty()
    }

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
        self.sewing.reset();
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
    /// A change of how something is sewn: undoable, but only the look
    /// changes, so the drape goes on undisturbed.
    fn restitch(&mut self, label: &str, edit: impl FnOnce(&mut PatternDocument) -> bool) {
        let before = self.document.clone();
        if edit(&mut self.document) {
            self.history.record(Box::new(cad_core::DocumentSnapshot::new(
                label,
                before,
                self.document.clone(),
            )));
        }
    }

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
                    if let Some(simulation) = &mut self.simulation {
                        // A broad influence rearranges a patch of cloth
                        // instead of pinching one particle: the middle of
                        // it follows the cursor, the rest less and less.
                        simulation.set_grab(grab.vertex, target, GRAB_RADIUS, GRAB_STRENGTH);
                        self.garment_dirty = true;
                    } else {
                        self.garment_dirty = true;
                    }
                }
            }
            if input.primary.stopped {
                if self.scene_grab.take().is_some() {
                    if let Some(simulation) = &mut self.simulation {
                        simulation.clear_grab();
                    }
                    self.topology.mesh_mut().recompute_normals();
                    self.garment_dirty = true;
                    // Let go: the cloth falls into place and settles.
                    self.editor.timeline.playing = true;
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
            // A new body's collision field is still being worked out; the
            // cloth waits for it.
            || (self.imported.is_none() && self.body.as_ref().is_some_and(Body::field_pending))
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
        let join_pages = self.editor.pattern_import.join_pages;

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
                    join_pages,
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
                if let Some(layout) = &pattern.pages {
                    message.push_str(&format!(" · {}", layout.summary()));
                }
                let off = pattern.pieces.iter().filter(|p| !p.enabled).count();
                if off > 0 {
                    message.push_str(&format!(
                        " · {off} page mark(s) switched off"
                    ));
                }
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
        // Frames keep coming while textures load, to show them when done.
        !self.loading_maps.borrow().is_empty()
            || self.sewing.busy()
            || self.editor.timeline.playing
            || (self.editor.avatar.animation.playing
                && self.editor.avatar.animation.motion.is_some())
            || self.garment_bake.is_some()
            || self.busy()
            || self.settle_steps > 0
            || self.editor.render.rendering()
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
            sketch_cursor: editor.sketch_preview,
            marker: editor.point_marker.map(|m| m.position),
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

    /// The weaves the document's fabrics are woven from, by key.
    pub fn weaves(&self) -> Vec<(u64, &cad_core::Weave)> {
        self.document
            .fabrics()
            .iter()
            .filter(|f| f.maps.is_none())
            .filter_map(|f| f.weave.as_ref())
            .map(|w| (w.key(), w))
            .collect()
    }

    /// Loads the textures of every weave and set of image maps in use that
    /// the scene lacks. Maps that cannot be read are reported once.
    pub fn load_weaves(&self, gpu: &gfx_render::GpuContext, scene: &mut gfx_render::SceneRenderer) {
        let weaves = self.weaves();
        let maps: Vec<(u64, &cad_core::FabricMaps)> = self
            .document
            .fabrics()
            .iter()
            .filter_map(|f| f.maps.as_ref())
            .map(|m| (m.key(), m))
            .collect();
        let keys: Vec<u64> = weaves.iter().map(|(k, _)| *k).chain(maps.iter().map(|(k, _)| *k)).collect();
        for (key, weave) in weaves {
            if !scene.has_weave(key) {
                scene.add_weave(gpu, key, &weave.texture(WEAVE_PIXELS), &keys);
            }
        }
        for (key, set) in maps {
            if scene.has_weave(key) || self.failed_maps.borrow().contains(&key) {
                continue;
            }
            let mut loading = self.loading_maps.borrow_mut();
            let Some(receiver) = loading.get(&key) else {
                // Decoded on a thread of its own; shown when it is done.
                let (sender, receiver) = std::sync::mpsc::channel();
                let set = set.clone();
                std::thread::spawn(move || {
                    let _ = sender.send(io_cad::fabric_maps::load(&set, MAP_PIXELS).map_err(|e| e.to_string()));
                });
                loading.insert(key, receiver);
                continue;
            };
            match receiver.try_recv() {
                Ok(Ok(texture)) => {
                    scene.add_weave(gpu, key, &texture, &keys);
                    loading.remove(&key);
                }
                Ok(Err(error)) => {
                    log::error!("fabric maps: {error}");
                    self.failed_maps.borrow_mut().insert(key);
                    loading.remove(&key);
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    log::error!("fabric maps: the loader stopped");
                    self.failed_maps.borrow_mut().insert(key);
                    loading.remove(&key);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
    }

    /// Viewport material of a piece: its fabric, highlighted when selected.
    pub fn material_of(&self, piece: PieceId) -> SurfaceMaterial {
        let piece_data = self.document.piece(piece);
        let fabric = piece_data.and_then(|p| self.document.fabric(p.fabric));
        let mut material = fabric.map_or_else(SurfaceMaterial::default, |f| SurfaceMaterial {
            base_color: f.base_color,
            roughness: f.roughness,
            metallic: f.metallic,
            weave_strength: f.weave_strength,
            pattern: f.pattern as u8 as f32,
            wash: f.wash,
            pattern_scale: f.pattern_scale,
            relief: f.relief,
            weave_key: match (&f.maps, &f.weave) {
                (Some(maps), _) => maps.key(),
                (None, Some(weave)) => weave.key(),
                (None, None) => 0,
            },
            weave_maps: f.maps.is_some(),
            weave_repeat: match (&f.maps, &f.weave, piece_data) {
                (Some(maps), _, Some(p)) => {
                    let tile = p.uv.tile_size.to_array();
                    [tile[0] / maps.size_mm[0].max(1.0), tile[1] / maps.size_mm[1].max(1.0)]
                }
                (None, Some(weave), Some(p)) => {
                    weave.repeats(WEAVE_PIXELS, p.uv.tile_size.to_array())
                }
                _ => [1.0; 2],
            },
            wear: f.wear,
            weave_yarn: if let Some(maps) = &f.maps {
                // No threads to vary; w: the alpha holds a roughness map.
                // y: shuffle the patches.
                [
                    0.0,
                    if maps.break_up_repeats { 1.0 } else { 0.0 },
                    0.0,
                    if maps.roughness.is_some() { 1.0 } else { 0.0 },
                ]
            } else { f.weave.as_ref().map_or([0.0; 4], |weave| {
                let (ends, picks) = weave.texture_threads(WEAVE_PIXELS);
                [ends as f32, picks as f32, weave.irregularity, weave.slubs]
            })},
            ..SurfaceMaterial::default()
        });
        if self.editor.selected_piece == Some(piece) {
            material.emissive = [0.10, 0.08, 0.03];
        }
        material
    }
}

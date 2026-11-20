//! State shared between panels, and the data the UI hands back each frame.

use avatar::{Appearance, BodyParams, FitReport, Measure, Measurements, PoseParams};
use cad_core::{DocumentChange, FabricId, PieceId};
use topology_sync::{SolverSettings, SolverStats};

use crate::viewport::ViewportInput;

/// The `egui` textures the two viewports are rendered into.
#[derive(Debug, Clone, Copy)]
pub struct ViewportTextures {
    pub canvas: egui::TextureId,
    pub scene: egui::TextureId,
}

/// What the garment's surface shows in the 3D view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SurfaceView {
    #[default]
    Fabric,
    /// Heatmap of how far the fabric is stretched.
    Strain,
    /// Heatmap of the clearance between fabric and body.
    Fit,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewOptions {
    pub show_grid: bool,
    pub show_mesh: bool,
    pub show_seam_allowance: bool,
    /// Piece names written on the pieces in the 2D view.
    pub show_piece_names: bool,
    pub surface: SurfaceView,
    /// Stretch shown as fully red in the strain map, in percent.
    pub strain_limit_percent: f32,
    /// Clearance shown as fully blue in the fit map, in millimetres.
    pub fit_range_mm: f32,
}

impl Default for ViewOptions {
    fn default() -> Self {
        Self {
            show_grid: true,
            show_mesh: false,
            show_seam_allowance: true,
            show_piece_names: true,
            surface: SurfaceView::Fabric,
            strain_limit_percent: 10.0,
            fit_range_mm: 40.0,
        }
    }
}

/// File formats the application can write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// 2D pattern, DXF-AAMA.
    DxfAama,
    /// 2D pattern, DXF-ASTM.
    DxfAstm,
    /// Draped garment, Wavefront OBJ (+ MTL).
    Obj,
    /// Draped garment, binary FBX.
    Fbx,
}

impl ExportFormat {
    pub const ALL: [Self; 4] = [Self::DxfAama, Self::DxfAstm, Self::Obj, Self::Fbx];

    pub fn label(self) -> &'static str {
        match self {
            Self::DxfAama => "Pattern · DXF-AAMA",
            Self::DxfAstm => "Pattern · DXF-ASTM",
            Self::Obj => "Garment 3D · OBJ",
            Self::Fbx => "Garment 3D · FBX",
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::DxfAama | Self::DxfAstm => "dxf",
            Self::Obj => "obj",
            Self::Fbx => "fbx",
        }
    }
}

/// Units scale mode when importing pattern vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScaleMode {
    #[default]
    Auto,
    Millimetres,
    Points72Dpi,
    Pixels96Dpi,
    Custom,
}

impl ScaleMode {
    pub const ALL: [Self; 5] = [
        Self::Auto,
        Self::Millimetres,
        Self::Points72Dpi,
        Self::Pixels96Dpi,
        Self::Custom,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto (from file units)",
            Self::Millimetres => "1:1 Millimetres",
            Self::Points72Dpi => "72 DPI / Points (standard PDF)",
            Self::Pixels96Dpi => "96 DPI / Pixels (standard SVG)",
            Self::Custom => "Custom factor",
        }
    }
}

/// The pattern import inspector state (PDF & SVG).
#[derive(Debug, Clone, PartialEq)]
pub struct PatternImportState {
    pub window_open: bool,
    pub file_path: String,
    pub selected_piece_index: usize,
    pub scale_mode: ScaleMode,
    pub custom_scale: f32,
    pub flip_y: bool,
    pub default_allowance_mm: f32,
    /// Join a pattern printed on many sheets (A4, US Letter) into one.
    pub join_pages: bool,
    pub pattern: Option<io_cad::ImportedPattern>,
    pub status: Option<Result<String, String>>,
    pub replace_project: bool,
}

impl Default for PatternImportState {
    fn default() -> Self {
        Self {
            window_open: false,
            file_path: String::new(),
            selected_piece_index: 0,
            scale_mode: ScaleMode::Auto,
            custom_scale: 1.0,
            flip_y: true,
            default_allowance_mm: 10.0,
            join_pages: true,
            pattern: None,
            status: None,
            replace_project: true,
        }
    }
}

/// The import/export window.
#[derive(Debug, Clone, PartialEq)]
pub struct IoState {
    pub window_open: bool,
    /// Avatar file to import (.obj, .gltf, .glb, .fbx).
    pub avatar_path: String,
    pub export_path: String,
    pub export_format: ExportFormat,
    /// Size written into DXF files.
    pub sample_size: String,
    /// 3D export: lay the UVs out like the 2D pattern instead of per
    /// fabric tile.
    pub pattern_uvs: bool,
    /// An imported avatar replaces the built-in one.
    pub avatar_imported: bool,
    /// Outcome of the last import or export: `Ok(message)` or `Err(why)`.
    pub status: Option<Result<String, String>>,
}

impl Default for IoState {
    fn default() -> Self {
        Self {
            window_open: false,
            avatar_path: String::new(),
            export_path: "garment".to_owned(),
            export_format: ExportFormat::DxfAama,
            sample_size: "M".to_owned(),
            pattern_uvs: false,
            avatar_imported: false,
            status: None,
        }
    }
}

/// Figures shown in the status bar; filled in by the application.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SceneStats {
    pub vertices: usize,
    pub triangles: usize,
    pub stitches: usize,
    pub frame_ms: f32,
    pub backend: String,
}

/// Play/pause state and frame counter of the simulation timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct Timeline {
    pub playing: bool,
    pub frame: u32,
    pub end_frame: u32,
    /// Simulation steps per second.
    pub fps: f32,
    pub looping: bool,
    accumulator: f32,
}

impl Default for Timeline {
    fn default() -> Self {
        Self {
            playing: false,
            frame: 0,
            // No end: the drape runs until the cloth has settled.
            end_frame: u32::MAX,
            fps: 30.0,
            looping: false,
            accumulator: 0.0,
        }
    }
}

impl Timeline {
    /// Most steps taken in one rendered frame; avoids a spiral of death when
    /// rendering is slower than the simulation rate.
    const MAX_STEPS_PER_FRAME: u32 = 4;

    /// Fixed step length in seconds.
    pub fn step_seconds(&self) -> f32 {
        1.0 / self.fps.max(1.0)
    }

    pub fn seconds(&self) -> f32 {
        self.frame as f32 * self.step_seconds()
    }

    pub fn toggle(&mut self) {
        if !self.playing && self.frame >= self.end_frame {
            self.frame = 0;
        }
        self.playing = !self.playing;
        self.accumulator = 0.0;
    }

    pub fn rewind(&mut self) {
        self.frame = 0;
        self.accumulator = 0.0;
    }

    pub fn stop(&mut self) {
        self.playing = false;
        self.rewind();
    }

    /// Advances by wall-clock time `dt` and returns how many fixed simulation
    /// steps are due.
    pub fn advance(&mut self, dt: f32) -> u32 {
        if !self.playing {
            return 0;
        }
        let step = self.step_seconds();
        self.accumulator = (self.accumulator + dt).min(step * Self::MAX_STEPS_PER_FRAME as f32);
        let mut steps = 0;
        while self.accumulator >= step {
            self.accumulator -= step;
            if self.frame >= self.end_frame {
                if self.looping {
                    self.frame = 0;
                } else {
                    self.playing = false;
                    break;
                }
            }
            self.frame += 1;
            steps += 1;
        }
        steps
    }
}

/// Simulation controls shared between the UI and the application.
#[derive(Debug, Clone, PartialEq)]
pub struct SimulationState {
    pub settings: SolverSettings,
    /// Requested mesh resolution in mm. The application re-meshes (and
    /// resets the drape) when this differs from the current mesh.
    pub particle_distance: f32,
    /// Whether the GPU compute backend can be selected.
    pub gpu_available: bool,
    /// Reported by the solver after every step.
    pub stats: SolverStats,
}

impl SimulationState {
    /// Coarse grid for drafting: fast, soft folds.
    pub const DRAFT_MM: f32 = 20.0;
    pub const MEDIUM_MM: f32 = 10.0;
    /// Fine grid for final folds: 16× the particles of the draft grid.
    pub const FINE_MM: f32 = 5.0;
}

impl Default for SimulationState {
    fn default() -> Self {
        Self {
            settings: SolverSettings::default(),
            particle_distance: Self::DRAFT_MM,
            gpu_available: false,
            stats: SolverStats::default(),
        }
    }
}

/// Built-in garments a new project can start from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sample {
    SleevelessTop,
    /// A waistcoat that buttons down the front.
    Waistcoat,
    /// Straight-leg jeans, drafted to the avatar: pleats, darts,
    /// flat-felled seams, white topstitching on dark denim.
    Jeans,
}

impl Sample {
    pub const ALL: [Sample; 3] = [Sample::SleevelessTop, Sample::Waistcoat, Sample::Jeans];

    pub fn label(self) -> &'static str {
        match self {
            Sample::SleevelessTop => "Sleeveless top",
            Sample::Waistcoat => "Waistcoat with buttons",
            Sample::Jeans => "Straight-leg jeans",
        }
    }

    pub fn document(self) -> cad_core::PatternDocument {
        match self {
            Sample::SleevelessTop => cad_core::PatternDocument::demo_top(),
            Sample::Waistcoat => cad_core::PatternDocument::demo_waistcoat(),
            Sample::Jeans => cad_core::jeans::jeans(Default::default()),
        }
    }
}

/// An animation of the avatar and how it is being played.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnimationState {
    /// What is shown; `None` is the edited pose.
    pub motion: Option<avatar::Motion>,
    pub playing: bool,
    /// Seconds into the motion.
    pub time: f32,
    /// 1 is real time.
    pub speed: f32,
    /// A worn garment moves with the body.
    pub garment_follows: bool,
    /// Whether the garment has been simulated through this motion; set by
    /// the application.
    pub garment: GarmentStatus,
}

/// How a worn garment moves in an animation.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum GarmentStatus {
    /// Carried along with the skin (a preview).
    #[default]
    Carried,
    /// Being simulated: the fraction done.
    Simulating(f32),
    /// Simulated: played back as simulated.
    Simulated,
    /// Nothing to simulate: no garment is worn yet.
    NotWorn,
    /// Simulated, but the cloth got caught somewhere (overstretched): shown
    /// carried with the skin instead.
    Unreliable,
}

impl Default for AnimationState {
    fn default() -> Self {
        Self {
            motion: None,
            playing: false,
            time: 0.0,
            speed: 1.0,
            garment_follows: true,
            garment: GarmentStatus::default(),
        }
    }
}

/// The avatar as the UI sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct AvatarState {
    /// Edited directly by the build sliders; the application rebuilds the
    /// body whenever this changes.
    pub params: BodyParams,
    /// Apply these parameters directly instead of solving the targets again.
    /// Set when a complete saved character or project avatar is loaded.
    pub apply_saved_params: bool,
    pub pose: PoseParams,
    /// Tape measurements the body must have, in metres, indexed by
    /// `Measure as usize`. The application solves the shape so that all of
    /// them hold at once and keeps them through later edits.
    pub targets: [Option<f32>; Measure::ALL.len()],
    /// A slider or field is being dragged: preview only, solve on release.
    pub dragging: bool,
    /// Result of the last solve, if any targets are set.
    pub fit: Option<FitReport>,
    /// Arm and leg angles the body was modelled with, in degrees.
    pub modelled_angles: (f32, f32),
    /// Taken off the current body by the application.
    pub measurements: Measurements,
    pub window_open: bool,
    /// Skin, eyes and hair: display only.
    pub appearance: Appearance,
    pub animation: AnimationState,
    /// False if the body model could not be loaded (placeholder avatar).
    pub available: bool,
    /// Saved characters, by name (filled in by the application).
    pub characters: Vec<CharacterEntry>,
    /// The character last put on or saved, if any.
    pub character: Option<String>,
    /// The name typed for saving a character.
    pub character_name: String,
    /// What the last library action did (or why it failed).
    pub character_status: Option<Result<String, String>>,
    /// When true, render the avatar in a neutral colour (safe for public use).
    pub modest: bool,
}

/// One saved character: an avatar's build, measurements and looks.
#[derive(Debug, Clone, PartialEq)]
pub struct CharacterEntry {
    pub name: String,
    /// Measured height in metres when saved.
    pub height: f32,
    pub params: BodyParams,
    pub targets: [Option<f32>; Measure::ALL.len()],
    pub appearance: Appearance,
}

impl CharacterEntry {
    /// True if the avatar looks exactly like this character.
    pub fn worn_by(&self, avatar: &AvatarState) -> bool {
        self.params == avatar.params
            && self.targets == avatar.targets
            && self.appearance == avatar.appearance
    }

    /// Puts this character's build, measurements and looks on the avatar
    /// (the pose stays).
    pub fn put_on(&self, avatar: &mut AvatarState) {
        avatar.params = self.params;
        avatar.apply_saved_params = true;
        avatar.targets = self.targets;
        avatar.appearance = self.appearance;
        avatar.character = Some(self.name.clone());
        avatar.character_name = self.name.clone();
    }
}

impl Default for AvatarState {
    fn default() -> Self {
        Self {
            params: BodyParams::female(),
            pose: PoseParams::default(),
            targets: [None; Measure::ALL.len()],
            dragging: false,
            fit: None,
            modelled_angles: (30.0, 5.0),
            measurements: Measurements::default(),
            window_open: true,
            appearance: Appearance::default(),
            animation: AnimationState::default(),
            available: false,
            characters: Vec::new(),
            character: None,
            character_name: String::new(),
            character_status: None,
            modest: avatar::config::MODEST.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

/// The tools of the 2D pattern view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CanvasTool {
    /// Select pieces and lines, drag pieces around.
    #[default]
    Select,
    /// Edit the points and curve handles of the selected piece.
    Points,
    /// Sew a line of one piece to a line of another.
    Sew,
    /// Click points to draw a new straight-edged pattern piece.
    SketchLine,
    /// Click points to draw a new piece and fit its outline with curves.
    SketchSpline,
    /// Place a button on the selected piece.
    Button,
    /// Place a buttonhole on the selected piece.
    Buttonhole,
    /// Pull a selected piece around the flat layout.
    Pull,
    /// Click a line to add a turn point there, splitting it in two.
    AddPoint,
    /// Drag a line to bend it into a curve; double-click straightens it.
    Curve,
    /// Click a seam or a free edge to sew it with the chosen stitch.
    Stitch,
    /// Sew a line to another as a flat-felled seam.
    FlatFell,
    /// Put a zipper between two lines.
    Zipper,
    /// Draw guide lines (helplines) that formulas place.
    Guide,
}

impl CanvasTool {
    pub const ALL: [Self; 14] = [
        Self::Select,
        Self::Points,
        Self::AddPoint,
        Self::Curve,
        Self::Sew,
        Self::FlatFell,
        Self::Zipper,
        Self::Stitch,
        Self::SketchLine,
        Self::SketchSpline,
        Self::Button,
        Self::Buttonhole,
        Self::Pull,
        Self::Guide,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Select => "Select / move pieces (V)",
            Self::Points => "Edit points (P)",
            Self::Sew => "Sew lines (S)",
            Self::SketchLine => "Draw polygon (L)",
            Self::SketchSpline => "Draw spline (B)",
            Self::Button => "Place button",
            Self::Buttonhole => "Place buttonhole",
            Self::Pull => "Pull fabric (move piece)",
            Self::AddPoint => "Add point on a line (A)",
            Self::Curve => "Bend lines into curves (C)",
            Self::Stitch => "Stitch: click a seam or an edge to sew it with the chosen stitch (T)",
            Self::FlatFell => {
                "Flat-felled seam: sew two lines, allowances folded in, two rows outside (F)"
            }
            Self::Zipper => "Zipper: put a zipper between two lines, like sewing a seam (Z)",
            Self::Guide => {
                "Helpline (R): drag to draw a guide line, then give its distance as a formula (inseam, waist / 4 …). Shapes drawn on helplines stretch when they move"
            }
        }
    }
}

/// The 2D view's camera: what is drawn where.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CanvasView {
    /// Pattern point (mm) at the centre of the view.
    pub center: glam::Vec2,
    /// Physical pixels per millimetre.
    pub zoom: f32,
}

impl Default for CanvasView {
    fn default() -> Self {
        Self {
            center: glam::Vec2::ZERO,
            zoom: 1.0,
        }
    }
}

/// A point the add-point tool would put on a line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointMarker {
    pub piece: PieceId,
    pub edge: u32,
    pub t: f32,
    pub position: glam::Vec2,
    /// Distance along the line from its start and to its end, in mm.
    pub from_start: f32,
    pub to_end: f32,
    /// Snapped to the middle of the line.
    pub middle: bool,
}

/// One side of a seam being made: a run of lines on one piece and where
/// it was clicked, which says which end the seam starts at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SewPick {
    pub piece: PieceId,
    pub span: cad_core::EdgeSpan,
    /// Pattern-space point of the first click.
    pub at: glam::Vec2,
}

/// What a left click in the 3D view does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SceneTool {
    /// Select pieces, pull the cloth.
    #[default]
    Orbit,
    /// Directly move an arrangement piece around the avatar.
    Place,
    /// Draw lines on the avatar.
    Pen,
    /// Select drawn lines: make them straight, delete them.
    EditLines,
    /// Turn an area closed by drawn lines into a pattern piece.
    Flatten,
}

/// The 3D pen: drawing lines on the avatar and flattening the areas they
/// close.
#[derive(Debug, Clone)]
pub struct PenState {
    pub tool: SceneTool,
    /// The points of the line being drawn.
    pub drawing: Vec<cad_core::SkinPoint>,
    /// Draw every line on the other side of the body too.
    pub mirror: bool,
    pub selected_line: Option<u32>,
    /// The point of the selected line picked with Edit lines.
    pub selected_point: Option<usize>,
    /// Enter: finish the line being drawn (open).
    pub finish: bool,
    /// Escape: drop the line being drawn.
    pub cancel: bool,
    /// What the last flatten or drawing step did.
    pub status: Option<Result<String, String>>,
}

impl SceneTool {
    pub fn label(self) -> &'static str {
        match self {
            SceneTool::Orbit => "Select: click a piece, pull the cloth",
            SceneTool::Place => {
                "Place: drag the selected fabric around the avatar · left/right moves around it, up/down raises or lowers it"
            }
            SceneTool::Pen => {
                "3D pen: draw lines on the avatar (or just beside it). Click to add points; \
                 lines snap together where they meet"
            }
            SceneTool::EditLines => {
                "Edit lines: drag a line's points along the body · double-click a line to add a \
                 point, a point to remove it · Del deletes the selected point or line"
            }
            SceneTool::Flatten => {
                "Flatten: click inside an area closed by lines: it becomes a pattern piece in the \
                 2D view, sewn to the pieces next to it"
            }
        }
    }
}

impl Default for PenState {
    fn default() -> Self {
        Self {
            tool: SceneTool::Orbit,
            drawing: Vec::new(),
            mirror: true,
            selected_line: None,
            selected_point: None,
            finish: false,
            cancel: false,
            status: None,
        }
    }
}

/// Selection, view options and timeline: UI state that outlives a frame and
/// that the application reads as well.
#[derive(Debug, Clone, Default)]
pub struct EditorState {
    pub avatar: AvatarState,
    pub io: IoState,
    pub pattern_import: PatternImportState,
    pub file_browser: crate::file_browser::FileBrowserState,
    pub simulation: SimulationState,
    pub selected_piece: Option<PieceId>,
    /// The edge last clicked; the whole line it belongs to is selected.
    pub selected_edge: Option<cad_core::EdgeRef>,
    /// The active tool of the 2D pattern view.
    pub canvas_tool: CanvasTool,
    /// Sewing in progress: the first side, then the second.
    pub sew_a: Option<SewPick>,
    pub sew_b: Option<SewPick>,
    /// Points collected by one of the sketch tools before the new piece is closed.
    pub sketch_points: Vec<glam::Vec2>,
    /// What the 2D view's tool tells the user, shown inside the view.
    pub canvas_hint: String,
    /// Where the next sketch point would go (the cursor, snapped).
    pub sketch_preview: Option<glam::Vec2>,
    /// Typed length (and angle) for the next sketch point.
    pub sketch_input: String,
    /// Close the sketch into a piece on the next frame.
    pub sketch_finish: bool,
    /// Where the add-point tool would put a point.
    pub point_marker: Option<PointMarker>,
    /// The 2D view's camera as of the last frame, for drawing on it.
    pub canvas_view: CanvasView,
    /// Text of the length / angle boxes of the selected line.
    pub line_input: (String, String),
    /// Every name formulas can use, with its value: the avatar's
    /// measurements and the document's variables (filled in by the
    /// application).
    pub parameters: Vec<cad_core::Param>,
    /// Formulas without a value, with what they belong to.
    pub parameter_errors: Vec<(String, String)>,
    /// The guide line selected in the 2D view.
    pub selected_guide: Option<u32>,
    /// Drawing on the avatar in the 3D view.
    pub pen: PenState,
    /// The project file being edited, once saved or opened.
    pub project_path: Option<std::path::PathBuf>,
    /// Outcome of the last save or open, for the status bar.
    pub project_status: Option<Result<String, String>>,
    /// A save (`Some(true)`: save as) or open requested by shortcut, run
    /// once the frame's input is released.
    pub pending_save: Option<bool>,
    pub pending_open: bool,
    pub selected_fabric: Option<FabricId>,
    pub view: ViewOptions,
    pub timeline: Timeline,
    /// The studio render of the 3D view and its window.
    pub render: RenderState,
    /// The stitch the sewing tool sews new seams with, and the stitch tool
    /// puts on what it is clicked on.
    pub sew_stitching: cad_core::Stitching,
    /// How the flat-felled seam tool sews (its thread, stitch length…).
    pub felled_stitching: FelledStitching,
    /// The zipper the zipper tool puts in.
    pub zipper: cad_core::Zipper,
    /// Settings for the next dart or pleat cut into a line.
    pub shaping: Shaping,
    pub stats: SceneStats,
    /// Labels of the next undo/redo steps, for the Edit menu.
    pub undo_label: Option<String>,
    pub redo_label: Option<String>,
}

/// The studio render: a still of the 3D view in studio light, refined
/// sample by sample. The application makes it; the UI shows it.
#[derive(Debug, Clone)]
pub struct RenderState {
    /// The render window is shown.
    pub open: bool,
    /// Start a new render on the next frame.
    pub start: bool,
    /// The picture so far, once one has started.
    pub texture: Option<egui::TextureId>,
    /// Its size in pixels.
    pub size: [u32; 2],
    /// Samples done and to do.
    pub samples: (u32, u32),
    /// Write the finished picture to this PNG file.
    pub save_to: Option<std::path::PathBuf>,
    /// Outcome of the last save.
    pub status: Option<Result<String, String>>,
    /// Depth of field, 0..=1: how much what is nearer or farther than the
    /// figure blurs (0: everything sharp).
    pub depth_of_field: f32,
}

impl Default for RenderState {
    fn default() -> Self {
        Self {
            open: false,
            start: false,
            texture: None,
            size: [0; 2],
            samples: (0, 0),
            save_to: None,
            status: None,
            depth_of_field: 0.3,
        }
    }
}

impl RenderState {
    /// The lens radius in metres for [`Self::depth_of_field`].
    pub fn aperture(&self) -> f32 {
        self.depth_of_field.clamp(0.0, 1.0) * 0.025
    }

    pub fn rendering(&self) -> bool {
        self.start || (self.texture.is_some() && self.samples.0 < self.samples.1)
    }
}

/// The flat-felled seam tool's stitch.
#[derive(Debug, Clone, Copy)]
pub struct FelledStitching(pub cad_core::Stitching);

impl Default for FelledStitching {
    fn default() -> Self {
        Self(
            cad_core::Stitching::of(cad_core::StitchKind::FlatFelled)
                .with_thread([0.78, 0.76, 0.7]),
        )
    }
}

/// The next dart or pleat: where along the line, and how big (mm).
#[derive(Debug, Clone, Copy)]
pub struct Shaping {
    pub at: f32,
    pub dart_width: f32,
    pub dart_length: f32,
    pub pleat_intake: f32,
    pub pleat_length: f32,
}

impl Default for Shaping {
    fn default() -> Self {
        Self {
            at: 0.5,
            dart_width: 25.0,
            dart_length: 110.0,
            pleat_intake: 40.0,
            pleat_length: 140.0,
        }
    }
}

/// Requests the UI cannot fulfil itself.
#[derive(Debug, Clone, PartialEq)]
pub enum UiCommand {
    Undo,
    Redo,
    /// Put every piece back to its flat arrangement.
    ResetDrape,
    /// Advance the simulation by exactly one step.
    StepSimulation,
    FitCanvas,
    /// Put the pieces back around the avatar and reset the drape.
    ArrangePieces,
    ResetCamera3d,
    /// Turn or mirror the selected piece in the 2D layout.
    TransformPiece(cad_core::PieceTransform),
    /// Add a mirrored copy of the selected piece, for the other side of
    /// the body.
    MirrorCopy,
    /// Add a linked mirrored clone of the selected piece: changes to either piece
    /// are automatically mirrored to the other.
    MirrorClone,
    /// Break the mirrored clone link for a piece.
    UnlinkClone(PieceId),
    /// Delete the selected piece and its seams.
    DeletePiece,
    /// The Delete key: with the sewing tool, take the seam of the selected
    /// line out; otherwise delete the selected piece.
    DeleteSelection,
    /// Make an anchor a turn point (splits the line there) or a curve
    /// point (joins the two lines meeting there into one).
    SetPointKind {
        piece: PieceId,
        anchor: u32,
        kind: cad_core::PointKind,
    },
    /// Cut the line through `edge` into `parts` lines of equal length.
    SplitLine {
        piece: PieceId,
        edge: u32,
        parts: u32,
    },
    /// Mirror a piece cut on the fold about its straight line through
    /// `edge`, making the whole piece.
    UnfoldPiece {
        piece: PieceId,
        edge: u32,
    },
    /// Give the line through `edge` a new length in mm.
    SetLineLength {
        piece: PieceId,
        edge: u32,
        length: f32,
    },
    /// Turn the line through `edge` to a direction in degrees.
    SetLineAngle {
        piece: PieceId,
        edge: u32,
        degrees: f32,
    },
    /// Lay all pieces out side by side without overlaps.
    TidyLayout,
    DeleteSeam(cad_core::SeamId),
    /// Save the project to this file.
    SaveProject(std::path::PathBuf),
    /// Replace everything with the project in this file.
    OpenProject(std::path::PathBuf),
    /// Start a new project from one of the built-in garments.
    NewProject(Sample),
    /// Simulate the worn garment through the avatar's current animation.
    SimulateGarmentMotion,
    /// Turn a seam around (its start meets the other end).
    FlipSeam(cad_core::SeamId),
    /// Sew a seam with another stitch.
    SetSeamStitching(cad_core::SeamId, cad_core::Stitching),
    /// Make a seam a zipper, change it, or (`None`) a plain seam again.
    SetSeamZipper(cad_core::SeamId, Option<cad_core::Zipper>),
    /// Cut a dart into a piece at fraction `t` of `edge` (mm).
    AddDart {
        piece: PieceId,
        edge: u32,
        t: f32,
        width: f32,
        length: f32,
    },
    /// Lay a knife pleat into a piece at fraction `t` of `edge` (mm).
    AddPleat {
        piece: PieceId,
        edge: u32,
        t: f32,
        intake: f32,
        length: f32,
    },
    /// Replace a piece's fold lines.
    SetFolds {
        piece: PieceId,
        folds: Vec<cad_core::FoldLine>,
    },
    /// Stitch a free edge (hem, neckline) on its own; `None` unpicks it.
    SetEdgeStitching {
        piece: PieceId,
        span: cad_core::EdgeSpan,
        stitching: Option<cad_core::Stitching>,
    },
    /// Copy the outermost garment into a new layer worn over it.
    AddOuterLayer,
    /// Replace the avatar with the geometry in this file.
    ImportAvatar(std::path::PathBuf),
    /// Go back to the built-in parametric avatar.
    RestoreAvatar,
    /// Replace the document's variables.
    SetVariables(Vec<cad_core::Variable>),
    /// Scripted drawing (headless runs): a line on the avatar through
    /// points seen straight from the front (x, y in metres).
    PenLineFromFront {
        points: Vec<glam::Vec2>,
        closed: bool,
    },
    /// Scripted flattening: the area seen straight from the front at x, y.
    FlattenFromFront(glam::Vec2),
    /// Flatten every area the drawn lines close that is not a piece yet.
    FlattenAll,
    /// Flatten a drawn line as a straight line, or not.
    SetAvatarLineStraight {
        line: u32,
        straight: bool,
    },
    DeleteAvatarLine(u32),
    /// Replace the guide line with this one's id.
    SetGuide(cad_core::Guide),
    DeleteGuide(u32),
    /// Hold the line through `edge` at a formula's length; `None` lets it
    /// go. A plain number just sets the length.
    SetLineFormula {
        piece: PieceId,
        edge: u32,
        formula: Option<String>,
    },
    /// Save the avatar's build, measurements and looks to the character
    /// library under this name (replacing one of the same name).
    SaveCharacter(String),
    /// Remove the character with this name from the library.
    DeleteCharacter(String),
    /// Write the pattern or the draped garment to this file.
    Export(std::path::PathBuf),
    /// Inspect/load a pattern file (.pdf or .svg) for the pattern import inspector.
    InspectPatternFile(std::path::PathBuf),
    /// Apply the configured imported pattern pieces into the document.
    ApplyImportedPattern {
        pattern: Box<io_cad::ImportedPattern>,
        replace_project: bool,
    },
    Quit,
}

/// Everything one UI frame produced.
#[derive(Debug, Default)]
pub struct UiOutput {
    /// Edits made through property panels, already applied to the document.
    pub changes: Vec<DocumentChange>,
    pub commands: Vec<UiCommand>,
    /// `None` when the tab is hidden; the viewport then need not render.
    pub canvas: Option<ViewportInput>,
    pub scene: Option<ViewportInput>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeline_steps_at_a_fixed_rate_and_stops_at_the_end() {
        let mut t = Timeline {
            fps: 10.0,
            end_frame: 3,
            ..Timeline::default()
        };
        assert_eq!(t.advance(1.0), 0, "paused timelines do not advance");
        t.toggle();
        assert_eq!(t.advance(0.05), 0);
        assert_eq!(t.advance(0.05), 1);
        assert_eq!(t.advance(0.25), 2);
        assert_eq!(t.frame, 3);
        assert_eq!(t.advance(0.2), 0);
        assert!(!t.playing);
        t.toggle(); // restarts from the beginning
        assert_eq!(t.frame, 0);
        assert!(t.playing);
    }
}

//! The dock layout and the frame-level chrome (menu bar, status bar).

use cad_core::PatternDocument;
use egui_dock::{DockArea, DockState, NodeIndex, Style, TabViewer};

use crate::panels::material::MaterialPanel;
use crate::panels::pattern::PatternPropertiesPanel;
use crate::panels::simulation::SimulationPanel;
use crate::panels::timeline::TimelinePanel;
use crate::panels::{Panel, PanelContext};
use crate::state::{EditorState, SurfaceView, UiCommand, UiOutput, ViewportTextures};
use crate::viewport;

/// Identifies a dock tab. The dock state stores these, so the layout can be
/// serialised without touching panel state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tab {
    Canvas2d,
    Viewport3d,
    PatternProperties,
    Materials,
    Simulation,
    Timeline,
}

impl Tab {
    pub const ALL: [Tab; 6] = [
        Tab::Canvas2d,
        Tab::Viewport3d,
        Tab::PatternProperties,
        Tab::Materials,
        Tab::Simulation,
        Tab::Timeline,
    ];
}

/// Viewport A as a panel.
#[derive(Debug, Default)]
struct CanvasPanel;

impl Panel for CanvasPanel {
    fn title(&self) -> &'static str {
        "2D Pattern"
    }
    fn ui(&mut self, ui: &mut egui::Ui, cx: &mut PanelContext<'_>) {
        canvas_toolbar(ui, cx);
        // What the tool expects next is shown inside the view, not in the
        // toolbar.
        let hint = if cx.editor.canvas_hint.is_empty() {
            "Middle/right-drag to pan · scroll to zoom"
        } else {
            cx.editor.canvas_hint.as_str()
        };
        let hint = hint.to_owned();
        cx.output.canvas = Some(viewport::show(ui, cx.textures.canvas, &hint));
    }
    fn fills_tab(&self) -> bool {
        true
    }
}

/// The 2D view's toolbar: tools on the left, operations on the selected
/// piece next to them. Icons only; names and shortcuts are in the tooltips.
fn canvas_toolbar(ui: &mut egui::Ui, cx: &mut PanelContext<'_>) {
    use crate::icons::{self, Icon};
    use crate::state::CanvasTool;
    let (undo_label, redo_label) = (cx.editor.undo_label.clone(), cx.editor.redo_label.clone());
    let editor = &mut cx.editor;
    egui::Frame::new()
        .inner_margin(egui::Margin::symmetric(6, 3))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                for (tool, icon) in [
                    (CanvasTool::Select, Icon::Select),
                    (CanvasTool::Points, Icon::Points),
                    (CanvasTool::Sew, Icon::Sew),
                ] {
                    if icons::button(ui, icon, editor.canvas_tool == tool, true, tool.label())
                        .clicked()
                    {
                        editor.canvas_tool = tool;
                        editor.sew_a = None;
                        editor.sew_b = None;
                        editor.sketch_points.clear();
                    }
                }
                for (tool, icon) in [
                    (CanvasTool::SketchLine, Icon::SketchLine),
                    (CanvasTool::SketchSpline, Icon::SketchSpline),
                    (CanvasTool::Button, Icon::Button),
                    (CanvasTool::Buttonhole, Icon::Buttonhole),
                    (CanvasTool::Pull, Icon::Pull),
                ] {
                    if icons::button(ui, icon, editor.canvas_tool == tool, true, tool.label())
                        .clicked()
                    {
                        editor.canvas_tool = tool;
                        editor.sew_a = None;
                        editor.sew_b = None;
                        editor.sketch_points.clear();
                    }
                }
                icons::separator(ui);
                let piece = editor.selected_piece.is_some();
                let mut command = |ui: &mut egui::Ui, icon, tooltip, command: UiCommand| {
                    if icons::button(ui, icon, false, piece, tooltip).clicked() {
                        cx.output.commands.push(command);
                    }
                };
                use cad_core::PieceTransform as T;
                command(
                    ui,
                    Icon::RotateCcw,
                    "Rotate selected piece 90° counter-clockwise",
                    UiCommand::TransformPiece(T::RotateCcw),
                );
                command(
                    ui,
                    Icon::RotateCw,
                    "Rotate selected piece 90° clockwise",
                    UiCommand::TransformPiece(T::RotateCw),
                );
                command(
                    ui,
                    Icon::FlipHorizontal,
                    "Mirror selected piece left ↔ right",
                    UiCommand::TransformPiece(T::FlipHorizontal),
                );
                command(
                    ui,
                    Icon::FlipVertical,
                    "Mirror selected piece top ↔ bottom",
                    UiCommand::TransformPiece(T::FlipVertical),
                );
                icons::separator(ui);
                command(
                    ui,
                    Icon::MirrorCopy,
                    "Add a mirrored copy for the other side of the body",
                    UiCommand::MirrorCopy,
                );
                command(
                    ui,
                    Icon::Delete,
                    "Delete selected piece (Del)",
                    UiCommand::DeletePiece,
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if icons::button(ui, Icon::FitView, false, true, "Fit all pieces in view")
                        .clicked()
                    {
                        cx.output.commands.push(UiCommand::FitCanvas);
                    }
                    if icons::button(
                        ui,
                        Icon::Tidy,
                        false,
                        true,
                        "Tidy layout: pieces side by side, no overlaps",
                    )
                    .clicked()
                    {
                        cx.output.commands.push(UiCommand::TidyLayout);
                    }
                    icons::separator(ui);
                    let redo = redo_label
                        .as_deref()
                        .map_or("Redo (Ctrl+Shift+Z)".to_owned(), |l| {
                            format!("Redo {l} (Ctrl+Shift+Z)")
                        });
                    if icons::button(ui, Icon::Redo, false, redo_label.is_some(), &redo).clicked() {
                        cx.output.commands.push(UiCommand::Redo);
                    }
                    let undo = undo_label
                        .as_deref()
                        .map_or("Undo (Ctrl+Z)".to_owned(), |l| format!("Undo {l} (Ctrl+Z)"));
                    if icons::button(ui, Icon::Undo, false, undo_label.is_some(), &undo).clicked() {
                        cx.output.commands.push(UiCommand::Undo);
                    }
                });
            });
        });
}

/// Viewport B as a panel.
#[derive(Debug, Default)]
struct ScenePanel;

impl Panel for ScenePanel {
    fn title(&self) -> &'static str {
        "3D Garment"
    }
    fn ui(&mut self, ui: &mut egui::Ui, cx: &mut PanelContext<'_>) {
        surface_toolbar(ui, &mut cx.editor.view);
        let hint =
            "Drag to orbit · middle/right-drag to pan · scroll to zoom · click a piece to select";
        cx.output.scene = Some(viewport::show(ui, cx.textures.scene, hint));
    }
    fn fills_tab(&self) -> bool {
        true
    }
}

/// The colour of the heatmaps at `t` in `0..=1` (blue to red). The same
/// five stops as `heat` in the 3D shader, in sRGB.
fn heat_color(t: f32) -> egui::Color32 {
    const STOPS: [[f32; 3]; 5] = [
        [39.0, 97.0, 237.0],
        [0.0, 206.0, 228.0],
        [63.0, 217.0, 97.0],
        [249.0, 231.0, 47.0],
        [243.0, 56.0, 39.0],
    ];
    let x = t.clamp(0.0, 1.0) * 4.0;
    let i = (x as usize).min(3);
    let f = x - i as f32;
    let channel = |c: usize| (STOPS[i][c] + (STOPS[i + 1][c] - STOPS[i][c]) * f).round() as u8;
    egui::Color32::from_rgb(channel(0), channel(1), channel(2))
}

/// The row above the 3D view: what the garment's surface shows, the range
/// of the heatmap and its legend.
fn surface_toolbar(ui: &mut egui::Ui, view: &mut crate::state::ViewOptions) {
    egui::Frame::new()
        .inner_margin(egui::Margin::symmetric(8, 4))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut view.surface, SurfaceView::Fabric, "Fabric");
                ui.selectable_value(&mut view.surface, SurfaceView::Strain, "Strain map")
                    .on_hover_text(
                        "How far the fabric is stretched: blue is relaxed, red is over-stretched",
                    );
                ui.selectable_value(&mut view.surface, SurfaceView::Fit, "Fit map")
                    .on_hover_text(
                        "Distance between fabric and body: red lies on the skin, blue hangs free",
                    );
                let (low, high) = match view.surface {
                    SurfaceView::Fabric => return,
                    SurfaceView::Strain => {
                        ui.separator();
                        ui.label("Red at");
                        ui.add(
                            egui::DragValue::new(&mut view.strain_limit_percent)
                                .range(1.0..=100.0)
                                .speed(0.2)
                                .suffix(" %"),
                        );
                        (
                            "0 %".to_owned(),
                            format!("≥ {:.0} % stretch", view.strain_limit_percent),
                        )
                    }
                    SurfaceView::Fit => {
                        ui.separator();
                        ui.label("Blue at");
                        ui.add(
                            egui::DragValue::new(&mut view.fit_range_mm)
                                .range(5.0..=200.0)
                                .speed(0.5)
                                .suffix(" mm"),
                        );
                        (
                            format!("≥ {:.0} mm loose", view.fit_range_mm),
                            "on the body".to_owned(),
                        )
                    }
                };
                // Legend: both maps run from blue (left) to red (right).
                ui.separator();
                ui.weak(low);
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(120.0, 10.0), egui::Sense::hover());
                const STEPS: usize = 40;
                for k in 0..STEPS {
                    let x = |i: usize| rect.left() + rect.width() * i as f32 / STEPS as f32;
                    let cell = egui::Rect::from_min_max(
                        egui::pos2(x(k), rect.top()),
                        egui::pos2(x(k + 1) + 0.5, rect.bottom()),
                    );
                    ui.painter().rect_filled(
                        cell,
                        0.0,
                        heat_color((k as f32 + 0.5) / STEPS as f32),
                    );
                }
                ui.weak(high);
            });
        });
}

#[derive(Default)]
struct Panels {
    canvas: CanvasPanel,
    scene: ScenePanel,
    pattern: PatternPropertiesPanel,
    materials: MaterialPanel,
    simulation: SimulationPanel,
    timeline: TimelinePanel,
}

impl Panels {
    fn get(&mut self, tab: Tab) -> &mut dyn Panel {
        match tab {
            Tab::Canvas2d => &mut self.canvas,
            Tab::Viewport3d => &mut self.scene,
            Tab::PatternProperties => &mut self.pattern,
            Tab::Materials => &mut self.materials,
            Tab::Simulation => &mut self.simulation,
            Tab::Timeline => &mut self.timeline,
        }
    }
}

/// Bridges `egui_dock` to [`Panel`].
struct Viewer<'a, 'b> {
    panels: &'a mut Panels,
    cx: PanelContext<'b>,
}

impl TabViewer for Viewer<'_, '_> {
    type Tab = Tab;

    fn id(&mut self, tab: &mut Tab) -> egui::Id {
        egui::Id::new(*tab)
    }

    fn title(&mut self, tab: &mut Tab) -> egui::WidgetText {
        self.panels.get(*tab).title().into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Tab) {
        let panel = self.panels.get(*tab);
        if panel.fills_tab() {
            panel.ui(ui, &mut self.cx);
        } else {
            egui::Frame::new()
                .inner_margin(10.0)
                .show(ui, |ui| panel.ui(ui, &mut self.cx));
        }
    }

    fn scroll_bars(&self, tab: &Tab) -> [bool; 2] {
        match tab {
            Tab::Canvas2d | Tab::Viewport3d | Tab::Timeline => [false, false],
            _ => [false, true],
        }
    }
}

/// The whole user interface.
pub struct AppUi {
    dock: DockState<Tab>,
    panels: Panels,
}

impl Default for AppUi {
    fn default() -> Self {
        Self::new()
    }
}

impl AppUi {
    pub fn new() -> Self {
        Self {
            dock: Self::default_layout(),
            panels: Panels::default(),
        }
    }

    /// ```text
    /// ┌────────────┬────────────┬──────────┐
    /// │ 2D Pattern │ 3D Garment │ Pattern  │
    /// │            │            │ Materials│
    /// │            │            │ Simulat. │
    /// ├────────────┴────────────┤          │
    /// │ Timeline                │          │
    /// └─────────────────────────┴──────────┘
    /// ```
    fn default_layout() -> DockState<Tab> {
        let mut dock = DockState::new(vec![Tab::Canvas2d]);
        let surface = dock.main_surface_mut();
        let [work, _properties] = surface.split_right(
            NodeIndex::root(),
            0.76,
            vec![Tab::PatternProperties, Tab::Materials, Tab::Simulation],
        );
        let [views, _timeline] = surface.split_below(work, 0.86, vec![Tab::Timeline]);
        surface.split_right(views, 0.5, vec![Tab::Viewport3d]);
        dock
    }

    /// Builds one frame of UI into `ui` (the root `Ui` of the window).
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        document: &mut PatternDocument,
        editor: &mut EditorState,
        textures: ViewportTextures,
    ) -> UiOutput {
        let mut output = UiOutput::default();
        self.shortcuts(ui, editor, &mut output);
        self.menu_bar(ui, editor, &mut output);
        self.status_bar(ui, editor);

        let mut style = Style::from_egui(ui.style().as_ref());
        // Viewports go edge to edge; other panels add their own padding.
        style.tab.tab_body.inner_margin = egui::Margin::ZERO;
        let mut viewer = Viewer {
            panels: &mut self.panels,
            cx: PanelContext {
                document,
                editor,
                textures,
                output: &mut output,
            },
        };
        DockArea::new(&mut self.dock)
            .style(style)
            .show_leaf_close_all_buttons(false)
            .show_leaf_collapse_buttons(false)
            .show_inside(ui, &mut viewer);
        crate::avatar_window::show(ui.ctx(), editor, &mut output);
        crate::io_window::show(ui.ctx(), editor, &mut output);
        crate::pattern_window::show(ui.ctx(), editor, &mut output);
        crate::file_browser::show(ui.ctx(), &mut editor.file_browser);

        if editor.file_browser.accepted {
            editor.file_browser.accepted = false;
            if let Some(path) = editor.file_browser.result.take() {
                match editor.file_browser.target_id.as_str() {
                    "pattern_import" => {
                        editor.pattern_import.file_path = path.to_string_lossy().to_string();
                        output.commands.push(UiCommand::InspectPatternFile(path));
                    }
                    "avatar_import" => {
                        editor.io.avatar_path = path.to_string_lossy().to_string();
                    }
                    "export_save" => {
                        editor.io.export_path = path.to_string_lossy().to_string();
                    }
                    _ => {}
                }
            }
        }

        output
    }

    fn shortcuts(&mut self, ui: &mut egui::Ui, editor: &mut EditorState, output: &mut UiOutput) {
        use egui::{Key, KeyboardShortcut, Modifiers};
        let typing = ui.ctx().text_edit_focused();
        ui.input_mut(|input| {
            // Test the more specific shortcut first: Ctrl+Z also matches
            // Ctrl+Shift+Z.
            if input.consume_shortcut(&KeyboardShortcut::new(
                Modifiers::COMMAND | Modifiers::SHIFT,
                Key::Z,
            )) || input.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::Y))
            {
                output.commands.push(UiCommand::Redo);
            } else if input.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::Z)) {
                output.commands.push(UiCommand::Undo);
            }
            let save_as = input.consume_shortcut(&KeyboardShortcut::new(
                Modifiers::COMMAND | Modifiers::SHIFT,
                Key::S,
            ));
            let save = input.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::S));
            if save_as || save {
                // The dialog must not open while the input is borrowed.
                editor.pending_save = Some(save_as);
            }
            if input.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::O)) {
                editor.pending_open = true;
            }
            if !typing && input.consume_key(Modifiers::NONE, Key::Space) {
                editor.timeline.toggle();
            }
            if !typing {
                use crate::state::CanvasTool;
                for (key, tool) in [
                    (Key::V, CanvasTool::Select),
                    (Key::P, CanvasTool::Points),
                    (Key::S, CanvasTool::Sew),
                    (Key::L, CanvasTool::SketchLine),
                    (Key::B, CanvasTool::SketchSpline),
                    (Key::U, CanvasTool::Button),
                    (Key::H, CanvasTool::Buttonhole),
                    (Key::G, CanvasTool::Pull),
                ] {
                    if input.consume_key(Modifiers::NONE, key) {
                        editor.canvas_tool = tool;
                        editor.sew_a = None;
                        editor.sew_b = None;
                        editor.sketch_points.clear();
                    }
                }
                if input.consume_key(Modifiers::NONE, Key::Escape) {
                    editor.sew_a = None;
                    editor.sew_b = None;
                    editor.sketch_points.clear();
                }
                if editor.selected_piece.is_some()
                    && (input.consume_key(Modifiers::NONE, Key::Delete)
                        || input.consume_key(Modifiers::NONE, Key::Backspace))
                {
                    output.commands.push(UiCommand::DeleteSelection);
                }
            }
        });
    }

    fn menu_bar(&mut self, ui: &mut egui::Ui, editor: &mut EditorState, output: &mut UiOutput) {
        if let Some(save_as) = editor.pending_save.take() {
            save_project(editor, output, save_as);
        }
        if std::mem::take(&mut editor.pending_open) {
            open_project(output);
        }
        egui::Panel::top("menu bar").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    ui.menu_button("New project", |ui| {
                        for sample in crate::state::Sample::ALL {
                            if ui.button(sample.label()).clicked() {
                                output.commands.push(UiCommand::NewProject(sample));
                            }
                        }
                    });
                    if ui
                        .add(egui::Button::new("Open project…").shortcut_text("Ctrl+O"))
                        .clicked()
                    {
                        open_project(output);
                    }
                    if ui
                        .add(egui::Button::new("Save project").shortcut_text("Ctrl+S"))
                        .clicked()
                    {
                        save_project(editor, output, false);
                    }
                    if ui
                        .add(egui::Button::new("Save project as…").shortcut_text("Ctrl+Shift+S"))
                        .clicked()
                    {
                        save_project(editor, output, true);
                    }
                    ui.separator();
                    if ui.button("Open Pattern File…").clicked() {
                        editor.pattern_import.window_open = true;
                        crate::pattern_window::browse_for_pattern(
                            &mut editor.pattern_import,
                            output,
                        );
                    }
                    if ui.button("Import Pattern (PDF / SVG)…").clicked() {
                        editor.pattern_import.window_open = true;
                    }
                    if ui.button("Import / Export…").clicked() {
                        editor.io.window_open = true;
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        output.commands.push(UiCommand::Quit);
                    }
                });
                ui.menu_button("Edit", |ui| {
                    let label = |verb: &str, what: &Option<String>| match what {
                        Some(what) => format!("{verb} {what}"),
                        None => verb.to_owned(),
                    };
                    let undo = egui::Button::new(label("Undo", &editor.undo_label))
                        .shortcut_text("Ctrl+Z");
                    if ui.add_enabled(editor.undo_label.is_some(), undo).clicked() {
                        output.commands.push(UiCommand::Undo);
                    }
                    let redo = egui::Button::new(label("Redo", &editor.redo_label))
                        .shortcut_text("Ctrl+Shift+Z");
                    if ui.add_enabled(editor.redo_label.is_some(), redo).clicked() {
                        output.commands.push(UiCommand::Redo);
                    }
                });
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut editor.view.show_grid, "Grid");
                    ui.checkbox(&mut editor.view.show_seam_allowance, "Seam allowance");
                    ui.checkbox(&mut editor.view.show_mesh, "Mesh wireframe (2D)");
                    ui.separator();
                    ui.label("3D surface");
                    ui.radio_value(&mut editor.view.surface, SurfaceView::Fabric, "Fabric");
                    ui.radio_value(&mut editor.view.surface, SurfaceView::Strain, "Strain map");
                    ui.radio_value(&mut editor.view.surface, SurfaceView::Fit, "Fit map");
                    ui.separator();
                    if ui.button("Fit pattern in 2D view").clicked() {
                        output.commands.push(UiCommand::FitCanvas);
                    }
                    if ui.button("Reset 3D camera").clicked() {
                        output.commands.push(UiCommand::ResetCamera3d);
                    }
                });
                ui.menu_button("Avatar", |ui| {
                    ui.checkbox(&mut editor.avatar.window_open, "Avatar editor");
                    if ui.button("Arrange garment around avatar").clicked() {
                        editor.timeline.stop();
                        output.commands.push(UiCommand::ArrangePieces);
                    }
                });
                ui.menu_button("Window", |ui| {
                    for tab in Tab::ALL {
                        let open = self.dock.find_tab(&tab).is_some();
                        let title = self.panels.get(tab).title();
                        if ui.add_enabled(!open, egui::Button::new(title)).clicked() {
                            self.dock.push_to_focused_leaf(tab);
                        }
                    }
                    ui.separator();
                    if ui.button("Reset layout").clicked() {
                        self.dock = Self::default_layout();
                    }
                });
            });
        });
    }

    fn status_bar(&mut self, ui: &mut egui::Ui, editor: &EditorState) {
        egui::Panel::bottom("status bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                let stats = &editor.stats;
                if let Some(path) = &editor.project_path {
                    ui.label(path.file_name().map_or_else(
                        || path.display().to_string(),
                        |n| n.to_string_lossy().into_owned(),
                    ));
                    ui.separator();
                }
                ui.weak(format!(
                    "{} vertices · {} triangles · {} stitches",
                    stats.vertices, stats.triangles, stats.stitches
                ));
                match &editor.project_status {
                    Some(Ok(message)) => {
                        ui.separator();
                        ui.colored_label(egui::Color32::from_rgb(120, 200, 120), message);
                    }
                    Some(Err(message)) => {
                        ui.separator();
                        ui.colored_label(egui::Color32::from_rgb(240, 110, 100), message);
                    }
                    None => {}
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.weak(format!("{} · {:.1} ms", stats.backend, stats.frame_ms));
                });
            });
        });
    }
}

/// Saves to the current project file, or asks where (`ask` or first save).
fn save_project(editor: &EditorState, output: &mut UiOutput, ask: bool) {
    let path = match (&editor.project_path, ask) {
        (Some(path), false) => Some(path.clone()),
        _ => rfd::FileDialog::new()
            .set_title("Save project")
            .add_filter("clo360 project", &["clo360"])
            .set_file_name("garment.clo360")
            .save_file()
            .map(|path| {
                if path.extension().is_none() {
                    path.with_extension("clo360")
                } else {
                    path
                }
            }),
    };
    if let Some(path) = path {
        output.commands.push(UiCommand::SaveProject(path));
    }
}

fn open_project(output: &mut UiOutput) {
    if let Some(path) = rfd::FileDialog::new()
        .set_title("Open project")
        .add_filter("clo360 project", &["clo360"])
        .pick_file()
    {
        output.commands.push(UiCommand::OpenProject(path));
    }
}

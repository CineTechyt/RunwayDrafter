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
    pub const ALL: [Tab; 6] =
        [Tab::Canvas2d, Tab::Viewport3d, Tab::PatternProperties, Tab::Materials, Tab::Simulation, Tab::Timeline];
}

/// Viewport A as a panel.
#[derive(Debug, Default)]
struct CanvasPanel;

impl Panel for CanvasPanel {
    fn title(&self) -> &'static str {
        "2D Pattern"
    }
    fn ui(&mut self, ui: &mut egui::Ui, cx: &mut PanelContext<'_>) {
        sewing_toolbar(ui, cx);
        let hint = "Drag points to edit · click a piece to select · middle/right-drag to pan · scroll to zoom";
        cx.output.canvas = Some(viewport::show(ui, cx.textures.canvas, hint));
    }
    fn fills_tab(&self) -> bool {
        true
    }
}

fn sewing_toolbar(ui: &mut egui::Ui, cx: &mut PanelContext<'_>) {
    let editor = &mut cx.editor;
    ui.horizontal(|ui| {
        if ui.selectable_label(editor.sewing_mode, "⇄ Sew").clicked() {
            editor.sewing_mode = !editor.sewing_mode;
            editor.sewing_edges_a.clear();
            editor.sewing_edges_b.clear();
            editor.sewing_select_b = false;
        }
        if editor.sewing_mode {
            ui.colored_label(egui::Color32::from_rgb(64, 210, 135), "A");
            ui.selectable_value(
                &mut editor.sewing_select_b,
                false,
                format!("{} edges", editor.sewing_edges_a.len()),
            );
            ui.colored_label(egui::Color32::from_rgb(70, 180, 245), "B");
            ui.selectable_value(
                &mut editor.sewing_select_b,
                true,
                format!("{} edges", editor.sewing_edges_b.len()),
            );
            ui.checkbox(&mut editor.sewing_reversed, "Reverse match");
            let equal_counts = !editor.sewing_edges_a.is_empty()
                && editor.sewing_edges_a.len() == editor.sewing_edges_b.len();
            let valid_edges = editor
                .sewing_edges_a
                .iter()
                .zip(&editor.sewing_edges_b)
                .all(|(a, b)| {
                    a != b
                        && cx.document.seam_of(*a).is_none()
                        && cx.document.seam_of(*b).is_none()
                        && cx.document
                            .piece(a.piece)
                            .is_some_and(|piece| a.edge < piece.outline.edge_count())
                        && cx.document
                            .piece(b.piece)
                            .is_some_and(|piece| b.edge < piece.outline.edge_count())
                });
            ui.label("Click edges in order on A, then B.");
            if ui
                .add_enabled(equal_counts && valid_edges, egui::Button::new("Sew selected"))
                .clicked()
            {
                for (a, b) in editor.sewing_edges_a.iter().zip(&editor.sewing_edges_b) {
                    cx.document.add_seam(*a, *b, editor.sewing_reversed);
                }
                cx.output.changes.push(cad_core::DocumentChange::Seams);
                editor.sewing_edges_a.clear();
                editor.sewing_edges_b.clear();
                editor.sewing_mode = false;
            } else if equal_counts && !valid_edges {
                ui.colored_label(egui::Color32::LIGHT_RED, "A selected edge is invalid or already sewn.");
            } else if !editor.sewing_edges_a.is_empty()
                && !editor.sewing_edges_b.is_empty()
                && !equal_counts
            {
                ui.weak("Select the same number of edges on both sides.");
            }
            if ui.small_button("Clear").clicked() {
                editor.sewing_edges_a.clear();
                editor.sewing_edges_b.clear();
            }
            if ui.small_button("Done").clicked() {
                editor.sewing_edges_a.clear();
                editor.sewing_edges_b.clear();
                editor.sewing_mode = false;
            }
        }
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
        let hint = "Drag to orbit · middle/right-drag to pan · scroll to zoom · click a piece to select";
        cx.output.scene = Some(viewport::show(ui, cx.textures.scene, hint));
    }
    fn fills_tab(&self) -> bool {
        true
    }
}

/// The colour of the heatmaps at `t` in `0..=1` (blue to red). The same
/// five stops as `heat` in the 3D shader, in sRGB.
fn heat_color(t: f32) -> egui::Color32 {
    const STOPS: [[f32; 3]; 5] = [[39.0, 97.0, 237.0], [0.0, 206.0, 228.0], [63.0, 217.0, 97.0], [249.0, 231.0, 47.0], [243.0, 56.0, 39.0]];
    let x = t.clamp(0.0, 1.0) * 4.0;
    let i = (x as usize).min(3);
    let f = x - i as f32;
    let channel = |c: usize| (STOPS[i][c] + (STOPS[i + 1][c] - STOPS[i][c]) * f).round() as u8;
    egui::Color32::from_rgb(channel(0), channel(1), channel(2))
}

/// The row above the 3D view: what the garment's surface shows, the range
/// of the heatmap and its legend.
fn surface_toolbar(ui: &mut egui::Ui, view: &mut crate::state::ViewOptions) {
    egui::Frame::new().inner_margin(egui::Margin::symmetric(8, 4)).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut view.surface, SurfaceView::Fabric, "Fabric");
            ui.selectable_value(&mut view.surface, SurfaceView::Strain, "Strain map")
                .on_hover_text("How far the fabric is stretched: blue is relaxed, red is over-stretched");
            ui.selectable_value(&mut view.surface, SurfaceView::Fit, "Fit map")
                .on_hover_text("Distance between fabric and body: red lies on the skin, blue hangs free");
            let (low, high) = match view.surface {
                SurfaceView::Fabric => return,
                SurfaceView::Strain => {
                    ui.separator();
                    ui.label("Red at");
                    ui.add(egui::DragValue::new(&mut view.strain_limit_percent).range(1.0..=100.0).speed(0.2).suffix(" %"));
                    ("0 %".to_owned(), format!("≥ {:.0} % stretch", view.strain_limit_percent))
                }
                SurfaceView::Fit => {
                    ui.separator();
                    ui.label("Blue at");
                    ui.add(egui::DragValue::new(&mut view.fit_range_mm).range(5.0..=200.0).speed(0.5).suffix(" mm"));
                    (format!("≥ {:.0} mm loose", view.fit_range_mm), "on the body".to_owned())
                }
            };
            // Legend: both maps run from blue (left) to red (right).
            ui.separator();
            ui.weak(low);
            let (rect, _) = ui.allocate_exact_size(egui::vec2(120.0, 10.0), egui::Sense::hover());
            const STEPS: usize = 40;
            for k in 0..STEPS {
                let x = |i: usize| rect.left() + rect.width() * i as f32 / STEPS as f32;
                let cell = egui::Rect::from_min_max(egui::pos2(x(k), rect.top()), egui::pos2(x(k + 1) + 0.5, rect.bottom()));
                ui.painter().rect_filled(cell, 0.0, heat_color((k as f32 + 0.5) / STEPS as f32));
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
            egui::Frame::new().inner_margin(10.0).show(ui, |ui| panel.ui(ui, &mut self.cx));
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
        Self { dock: Self::default_layout(), panels: Panels::default() }
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
        let [work, _properties] =
            surface.split_right(NodeIndex::root(), 0.76, vec![Tab::PatternProperties, Tab::Materials, Tab::Simulation]);
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
        let mut viewer =
            Viewer { panels: &mut self.panels, cx: PanelContext { document, editor, textures, output: &mut output } };
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
            if input.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z))
                || input.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::Y))
            {
                output.commands.push(UiCommand::Redo);
            } else if input.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::Z)) {
                output.commands.push(UiCommand::Undo);
            }
            if !typing && input.consume_key(Modifiers::NONE, Key::Space) {
                editor.timeline.toggle();
            }
        });
    }

    fn menu_bar(&mut self, ui: &mut egui::Ui, editor: &mut EditorState, output: &mut UiOutput) {
        egui::Panel::top("menu bar").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
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
                    let undo = egui::Button::new(label("Undo", &editor.undo_label)).shortcut_text("Ctrl+Z");
                    if ui.add_enabled(editor.undo_label.is_some(), undo).clicked() {
                        output.commands.push(UiCommand::Undo);
                    }
                    let redo = egui::Button::new(label("Redo", &editor.redo_label)).shortcut_text("Ctrl+Shift+Z");
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
                ui.weak(format!(
                    "{} vertices · {} triangles · {} stitches",
                    stats.vertices, stats.triangles, stats.stitches
                ));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.weak(format!("{} · {:.1} ms", stats.backend, stats.frame_ms));
                });
            });
        });
    }
}

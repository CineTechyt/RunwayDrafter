//! Dock panels.

pub mod material;
pub mod pattern;
pub mod simulation;
pub mod timeline;
pub mod stitch_editor;
pub mod weave_editor;

use cad_core::PatternDocument;

use crate::state::{EditorState, UiOutput, ViewportTextures};

/// What a panel may read and write while drawing itself.
pub struct PanelContext<'a> {
    pub document: &'a mut PatternDocument,
    pub editor: &'a mut EditorState,
    pub textures: ViewportTextures,
    /// Panels push document changes and commands here.
    pub output: &'a mut UiOutput,
}

/// A dockable panel. Implement this to add a new tab; register it in
/// [`crate::dock`].
pub trait Panel {
    fn title(&self) -> &'static str;

    fn ui(&mut self, ui: &mut egui::Ui, cx: &mut PanelContext<'_>);

    /// `true` for panels that manage their whole rectangle themselves
    /// (viewports): no padding, no scroll bars.
    fn fills_tab(&self) -> bool {
        false
    }
}

/// A two-column "label: widget" grid used by the property panels.
pub(crate) fn property_grid(ui: &mut egui::Ui, id: &str, contents: impl FnOnce(&mut egui::Ui)) {
    egui::Grid::new(id)
        .num_columns(2)
        .spacing([12.0, 6.0])
        .show(ui, contents);
}

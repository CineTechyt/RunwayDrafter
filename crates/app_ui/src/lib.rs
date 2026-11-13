//! The `egui` user interface: a dockable layout hosting the two viewports,
//! the pattern and material property panels and the simulation timeline.
//!
//! This crate knows nothing about `wgpu`. Viewports are plain `egui`
//! textures supplied by the application, and everything the UI wants done is
//! returned as data in [`UiOutput`]: document changes, viewport input and
//! commands. That keeps the UI testable and the render backend swappable.

pub mod avatar_window;
pub mod dock;
pub mod file_browser;
pub mod icons;
pub mod io_window;
pub mod panels;
pub mod pattern_window;
pub mod state;
pub mod viewport;

pub use dock::{AppUi, Tab};
pub use file_browser::{FileBrowserMode, FileBrowserState, FileFilter};
pub use panels::{Panel, PanelContext};
pub use state::{
    AnimationState, AvatarState, CanvasTool, EditorState, ExportFormat, GarmentStatus, IoState,
    PatternImportState, Sample, ScaleMode, SceneStats, SewPick, SimulationState, SurfaceView,
    Timeline, UiCommand, UiOutput, ViewOptions, ViewportTextures,
};
pub use viewport::{ButtonDrag, ViewportInput};

pub use egui;

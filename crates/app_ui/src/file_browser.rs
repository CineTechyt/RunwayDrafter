//! Interactive, multi-platform file browser for egui.
//!
//! Provides a full-featured desktop-grade file dialog that works identically
//! across Linux, Windows, macOS and Web:
//! - Drive detection (Windows C:\, D:\, etc., Unix /, macOS /Volumes)
//! - Standard user bookmarks (Home, Documents, Downloads, Desktop, Project)
//! - Interactive breadcrumb navigation & directory tree
//! - Sorting by Name, Size, and Date
//! - File filtering by extension (*.pdf, *.svg, *.obj, etc.)
//! - Instant search / filter within the current directory
//! - Open and Save modes with double-click support

use std::cmp::Ordering;
use std::path::{Component, Path, PathBuf};

/// The operation mode of the file browser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileBrowserMode {
    /// Select an existing file to open.
    OpenFile,
    /// Specify a file path to save/export to.
    SaveFile,
}

/// A file extension filter (e.g. "PDF Patterns (*.pdf)").
#[derive(Debug, Clone, PartialEq)]
pub struct FileFilter {
    pub label: String,
    pub extensions: Vec<String>,
}

impl FileFilter {
    pub fn new(label: impl Into<String>, extensions: &[&str]) -> Self {
        Self {
            label: label.into(),
            extensions: extensions.iter().map(|s| s.to_ascii_lowercase()).collect(),
        }
    }

    pub fn matches(&self, path: &Path) -> bool {
        if self.extensions.is_empty() || self.extensions.iter().any(|e| e == "*") {
            return true;
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        self.extensions.iter().any(|e| e == &ext)
    }
}

/// A quick-access bookmark (e.g. Home, Documents).
#[derive(Debug, Clone, PartialEq)]
pub struct Bookmark {
    pub label: String,
    pub icon: &'static str,
    pub path: PathBuf,
}

/// Metadata for an entry in the current directory.
#[derive(Debug, Clone, PartialEq)]
pub struct DirEntryInfo {
    pub name: String,
    pub path: PathBuf,
    pub is_dir: bool,
    pub size_bytes: u64,
    pub modified_secs: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortColumn {
    Name,
    Size,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// Persistent state of the interactive file browser.
#[derive(Debug, Clone, PartialEq)]
pub struct FileBrowserState {
    pub open: bool,
    pub mode: FileBrowserMode,
    pub current_dir: PathBuf,
    pub history: Vec<PathBuf>,
    pub history_idx: usize,
    pub selected_file: Option<PathBuf>,
    pub file_name_input: String,
    pub search_query: String,
    pub filter_index: usize,
    pub filters: Vec<FileFilter>,
    pub show_hidden: bool,
    pub target_id: String,
    pub entries: Vec<DirEntryInfo>,
    pub error_message: Option<String>,
    pub bookmarks: Vec<Bookmark>,
    pub drives: Vec<Bookmark>,
    pub sort_column: SortColumn,
    pub sort_direction: SortDirection,
    /// Set to true for one frame when the user accepts a selection.
    pub accepted: bool,
    /// Path chosen by the user on accept.
    pub result: Option<PathBuf>,
}

impl Default for FileBrowserState {
    fn default() -> Self {
        let initial_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
        let bookmarks = detect_bookmarks();
        let drives = detect_drives();

        let mut state = Self {
            open: false,
            mode: FileBrowserMode::OpenFile,
            current_dir: initial_dir.clone(),
            history: vec![initial_dir],
            history_idx: 0,
            selected_file: None,
            file_name_input: String::new(),
            search_query: String::new(),
            filter_index: 0,
            filters: Vec::new(),
            show_hidden: false,
            target_id: String::new(),
            entries: Vec::new(),
            error_message: None,
            bookmarks,
            drives,
            sort_column: SortColumn::Name,
            sort_direction: SortDirection::Ascending,
            accepted: false,
            result: None,
        };
        state.refresh();
        state
    }
}

impl FileBrowserState {
    /// Opens the file browser configured for pattern import (*.pdf, *.svg).
    pub fn open_for_pattern(&mut self) {
        self.open = true;
        self.mode = FileBrowserMode::OpenFile;
        self.target_id = "pattern_import".to_owned();
        self.filters = vec![
            FileFilter::new("All Patterns (*.pdf, *.svg)", &["pdf", "svg"]),
            FileFilter::new("PDF Patterns (*.pdf)", &["pdf"]),
            FileFilter::new("SVG Patterns (*.svg)", &["svg"]),
            FileFilter::new("All Files (*.*)", &["*"]),
        ];
        self.filter_index = 0;
        self.search_query.clear();
        self.refresh();
    }

    /// Opens the file browser configured for avatar geometry (*.obj, *.gltf, *.glb, *.fbx).
    pub fn open_for_avatar(&mut self) {
        self.open = true;
        self.mode = FileBrowserMode::OpenFile;
        self.target_id = "avatar_import".to_owned();
        self.filters = vec![
            FileFilter::new("Avatar 3D (*.obj, *.gltf, *.glb, *.fbx)", &["obj", "gltf", "glb", "fbx"]),
            FileFilter::new("Wavefront OBJ (*.obj)", &["obj"]),
            FileFilter::new("glTF 2.0 (*.gltf, *.glb)", &["gltf", "glb"]),
            FileFilter::new("Autodesk FBX (*.fbx)", &["fbx"]),
            FileFilter::new("All Files (*.*)", &["*"]),
        ];
        self.filter_index = 0;
        self.search_query.clear();
        self.refresh();
    }

    /// Opens the file browser configured for garment/pattern export.
    pub fn open_for_export(&mut self, default_filename: &str, extension: &str) {
        self.open = true;
        self.mode = FileBrowserMode::SaveFile;
        self.target_id = "export_save".to_owned();
        self.file_name_input = if default_filename.ends_with(extension) {
            default_filename.to_owned()
        } else {
            format!("{default_filename}.{extension}")
        };
        self.filters = vec![
            FileFilter::new(format!("Target file (*.{extension})"), &[extension]),
            FileFilter::new("All Files (*.*)", &["*"]),
        ];
        self.filter_index = 0;
        self.search_query.clear();
        self.refresh();
    }

    /// Navigates to `dir`, recording the step in history.
    pub fn navigate_to(&mut self, dir: PathBuf) {
        if dir.is_dir() {
            if self.history_idx + 1 < self.history.len() {
                self.history.truncate(self.history_idx + 1);
            }
            self.history.push(dir.clone());
            self.history_idx = self.history.len() - 1;
            self.current_dir = dir;
            self.selected_file = None;
            self.error_message = None;
            self.refresh();
        }
    }

    pub fn can_go_back(&self) -> bool {
        self.history_idx > 0
    }

    pub fn go_back(&mut self) {
        if self.can_go_back() {
            self.history_idx -= 1;
            self.current_dir = self.history[self.history_idx].clone();
            self.selected_file = None;
            self.refresh();
        }
    }

    pub fn can_go_forward(&self) -> bool {
        self.history_idx + 1 < self.history.len()
    }

    pub fn go_forward(&mut self) {
        if self.can_go_forward() {
            self.history_idx += 1;
            self.current_dir = self.history[self.history_idx].clone();
            self.selected_file = None;
            self.refresh();
        }
    }

    pub fn go_up(&mut self) {
        if let Some(parent) = self.current_dir.parent() {
            self.navigate_to(parent.to_path_buf());
        }
    }

    /// Reloads the directory entries and sorts them.
    pub fn refresh(&mut self) {
        self.entries.clear();
        self.error_message = None;

        let read_res = std::fs::read_dir(&self.current_dir);
        let Ok(read_dir) = read_res else {
            self.error_message = Some(format!("Cannot open directory: {}", self.current_dir.display()));
            return;
        };

        for entry in read_dir.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();

            if !self.show_hidden && is_hidden(&name) {
                continue;
            }

            let is_dir = entry.file_type().map_or(false, |ft| ft.is_dir());
            let metadata = entry.metadata().ok();
            let size_bytes = metadata.as_ref().map_or(0, |m| m.len());
            let modified_secs = metadata
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs());

            // If it's a file, check if it matches current filter
            if !is_dir {
                if let Some(filter) = self.filters.get(self.filter_index) {
                    if !filter.matches(&path) {
                        continue;
                    }
                }
            }

            self.entries.push(DirEntryInfo {
                name,
                path,
                is_dir,
                size_bytes,
                modified_secs,
            });
        }

        self.sort_entries();
    }

    fn sort_entries(&mut self) {
        let col = self.sort_column;
        let dir = self.sort_direction;

        self.entries.sort_by(|a, b| {
            // Folders always precede files
            if a.is_dir != b.is_dir {
                return if a.is_dir { Ordering::Less } else { Ordering::Greater };
            }

            let cmp = match col {
                SortColumn::Name => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
                SortColumn::Size => a.size_bytes.cmp(&b.size_bytes),
            };

            if dir == SortDirection::Ascending { cmp } else { cmp.reverse() }
        });
    }

    pub fn accept_selection(&mut self) {
        let chosen = match self.mode {
            FileBrowserMode::OpenFile => self.selected_file.clone(),
            FileBrowserMode::SaveFile => {
                if !self.file_name_input.trim().is_empty() {
                    let mut path = self.current_dir.join(self.file_name_input.trim());
                    // Auto-append extension from filter if none provided
                    if path.extension().is_none() {
                        if let Some(filter) = self.filters.get(self.filter_index) {
                            if let Some(first_ext) = filter.extensions.first() {
                                if first_ext != "*" {
                                    path.set_extension(first_ext);
                                }
                            }
                        }
                    }
                    Some(path)
                } else {
                    None
                }
            }
        };

        if let Some(path) = chosen {
            self.result = Some(path);
            self.accepted = true;
            self.open = false;
        }
    }
}

/// Detects standard multi-platform bookmarks (Home, Documents, Downloads, Desktop, Current Dir).
fn detect_bookmarks() -> Vec<Bookmark> {
    let mut list = Vec::new();

    if let Ok(cur) = std::env::current_dir() {
        list.push(Bookmark { label: "Project".to_owned(), icon: "📂", path: cur });
    }

    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from);

    if let Some(home_path) = home {
        list.push(Bookmark { label: "Home".to_owned(), icon: "🏠", path: home_path.clone() });

        let docs = home_path.join("Documents");
        if docs.is_dir() {
            list.push(Bookmark { label: "Documents".to_owned(), icon: "📄", path: docs });
        }
        let downloads = home_path.join("Downloads");
        if downloads.is_dir() {
            list.push(Bookmark { label: "Downloads".to_owned(), icon: "📥", path: downloads });
        }
        let desktop = home_path.join("Desktop");
        if desktop.is_dir() {
            list.push(Bookmark { label: "Desktop".to_owned(), icon: "🖥", path: desktop });
        }
    }

    list
}

/// Detects available storage drives across platforms.
fn detect_drives() -> Vec<Bookmark> {
    let mut drives = Vec::new();

    // Windows drives detection (A:\ through Z:\)
    #[cfg(target_os = "windows")]
    {
        for letter in b'A'..=b'Z' {
            let drive_str = format!("{}:\\", letter as char);
            let path = PathBuf::from(&drive_str);
            if path.exists() {
                drives.push(Bookmark {
                    label: format!("Drive ({}:)", letter as char),
                    icon: "💽",
                    path,
                });
            }
        }
    }

    // Unix / Linux / macOS root and mounts
    #[cfg(not(target_os = "windows"))]
    {
        drives.push(Bookmark {
            label: "Root (/)".to_owned(),
            icon: "💽",
            path: PathBuf::from("/"),
        });
        let media = PathBuf::from("/media");
        if media.is_dir() {
            drives.push(Bookmark { label: "Media (/media)".to_owned(), icon: "💾", path: media });
        }
        let mnt = PathBuf::from("/mnt");
        if mnt.is_dir() {
            drives.push(Bookmark { label: "Mounts (/mnt)".to_owned(), icon: "💾", path: mnt });
        }
        let volumes = PathBuf::from("/Volumes");
        if volumes.is_dir() {
            drives.push(Bookmark { label: "Volumes".to_owned(), icon: "💾", path: volumes });
        }
    }

    drives
}

fn is_hidden(name: &str) -> bool {
    name.starts_with('.')
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;

    if bytes < KB {
        format!("{bytes} B")
    } else if bytes < MB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else if bytes < GB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    }
}

/// Renders the file browser dialog window.
pub fn show(ctx: &egui::Context, state: &mut FileBrowserState) {
    if !state.open {
        return;
    }

    let title = match state.mode {
        FileBrowserMode::OpenFile => "Open File",
        FileBrowserMode::SaveFile => "Save File",
    };

    let mut open = state.open;
    egui::Window::new(title)
        .open(&mut open)
        .default_size([720.0, 480.0])
        .min_size([540.0, 360.0])
        .resizable(true)
        .collapsible(false)
        .show(ctx, |ui| {
            // --- Top Navigation Toolbar ---
            ui.horizontal(|ui| {
                if ui.add_enabled(state.can_go_back(), egui::Button::new("⮜")).on_hover_text("Back").clicked() {
                    state.go_back();
                }
                if ui.add_enabled(state.can_go_forward(), egui::Button::new("⮞")).on_hover_text("Forward").clicked() {
                    state.go_forward();
                }
                if ui.button("⬆ Up").on_hover_text("Go to parent directory").clicked() {
                    state.go_up();
                }
                if ui.button("🔄").on_hover_text("Refresh").clicked() {
                    state.refresh();
                }

                ui.separator();

                // Interactive Breadcrumb path bar
                render_breadcrumbs(ui, state);

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.checkbox(&mut state.show_hidden, "Hidden");
                    let search = ui.add(egui::TextEdit::singleline(&mut state.search_query).hint_text("🔍 Search...").desired_width(120.0));
                    if search.changed() {
                        // filter is handled in render
                    }
                });
            });

            ui.separator();

            // --- Main Content: Sidebar + File List ---
            let total_height = ui.available_height() - 75.0;
            ui.horizontal(|ui| {
                // Sidebar: Places & Drives
                let mut navigate_sidebar = None;
                ui.allocate_ui_with_layout(
                    egui::Vec2::new(140.0, total_height),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        ui.weak("PLACES");
                        for bm in &state.bookmarks {
                            let is_current = state.current_dir == bm.path;
                            let text = format!("{} {}", bm.icon, bm.label);
                            if ui.selectable_label(is_current, text).clicked() {
                                navigate_sidebar = Some(bm.path.clone());
                            }
                        }

                        if !state.drives.is_empty() {
                            ui.add_space(8.0);
                            ui.weak("DRIVES");
                            for drv in &state.drives {
                                let is_current = state.current_dir.starts_with(&drv.path);
                                let text = format!("{} {}", drv.icon, drv.label);
                                if ui.selectable_label(is_current, text).clicked() {
                                    navigate_sidebar = Some(drv.path.clone());
                                }
                            }
                        }
                    },
                );

                if let Some(target) = navigate_sidebar {
                    state.navigate_to(target);
                }

                ui.separator();

                // File Table
                egui::ScrollArea::vertical().max_height(total_height).show(ui, |ui| {
                    if let Some(err) = &state.error_message {
                        ui.colored_label(egui::Color32::from_rgb(240, 110, 100), err);
                        return;
                    }

                    let query = state.search_query.trim().to_lowercase();
                    let mut navigate_to_folder = None;
                    let mut file_double_clicked = false;

                    egui::Grid::new("file_browser_grid")
                        .num_columns(3)
                        .striped(true)
                        .spacing([12.0, 4.0])
                        .min_col_width(80.0)
                        .show(ui, |ui| {
                            // Column Headers with Sorting
                            ui.horizontal(|ui| {
                                let sort_indicator = if state.sort_column == SortColumn::Name {
                                    if state.sort_direction == SortDirection::Ascending { " ▲" } else { " ▼" }
                                } else { "" };
                                if ui.selectable_label(state.sort_column == SortColumn::Name, format!("Name{sort_indicator}")).clicked() {
                                    if state.sort_column == SortColumn::Name {
                                        state.sort_direction = match state.sort_direction {
                                            SortDirection::Ascending => SortDirection::Descending,
                                            SortDirection::Descending => SortDirection::Ascending,
                                        };
                                    } else {
                                        state.sort_column = SortColumn::Name;
                                        state.sort_direction = SortDirection::Ascending;
                                    }
                                    state.sort_entries();
                                }
                            });

                            ui.horizontal(|ui| {
                                let sort_indicator = if state.sort_column == SortColumn::Size {
                                    if state.sort_direction == SortDirection::Ascending { " ▲" } else { " ▼" }
                                } else { "" };
                                if ui.selectable_label(state.sort_column == SortColumn::Size, format!("Size{sort_indicator}")).clicked() {
                                    if state.sort_column == SortColumn::Size {
                                        state.sort_direction = match state.sort_direction {
                                            SortDirection::Ascending => SortDirection::Descending,
                                            SortDirection::Descending => SortDirection::Ascending,
                                        };
                                    } else {
                                        state.sort_column = SortColumn::Size;
                                        state.sort_direction = SortDirection::Ascending;
                                    }
                                    state.sort_entries();
                                }
                            });

                            ui.weak("Type");
                            ui.end_row();

                            for entry in &state.entries {
                                if !query.is_empty() && !entry.name.to_lowercase().contains(&query) {
                                    continue;
                                }

                                let is_selected = state.selected_file.as_ref() == Some(&entry.path);
                                let icon = if entry.is_dir { "📁" } else { get_file_icon(&entry.name) };
                                let display_text = format!("{icon} {}", entry.name);

                                let label_resp = ui.selectable_label(is_selected, display_text);
                                if label_resp.clicked() {
                                    if entry.is_dir {
                                        // single click selects or prepares
                                    } else {
                                        state.selected_file = Some(entry.path.clone());
                                        state.file_name_input = entry.name.clone();
                                    }
                                }
                                if label_resp.double_clicked() {
                                    if entry.is_dir {
                                        navigate_to_folder = Some(entry.path.clone());
                                    } else {
                                        state.selected_file = Some(entry.path.clone());
                                        state.file_name_input = entry.name.clone();
                                        file_double_clicked = true;
                                    }
                                }

                                if entry.is_dir {
                                    ui.weak("—");
                                    ui.weak("Folder");
                                } else {
                                    ui.label(format_size(entry.size_bytes));
                                    let ext = entry.path.extension().and_then(|e| e.to_str()).unwrap_or("file");
                                    ui.weak(ext.to_ascii_uppercase());
                                }
                                ui.end_row();
                            }
                        });

                    if let Some(folder) = navigate_to_folder {
                        state.navigate_to(folder);
                    }
                    if file_double_clicked {
                        state.accept_selection();
                    }
                });
            });

            ui.separator();

            // --- Bottom Bar: Filename input, Filter combo, Open/Save & Cancel ---
            ui.horizontal(|ui| {
                ui.label("File:");
                ui.add(egui::TextEdit::singleline(&mut state.file_name_input).desired_width(ui.available_width() - 250.0));

                if !state.filters.is_empty() {
                    let mut filter_changed = false;
                    let current_filter = state.filters.get(state.filter_index).map_or("All files", |f| &f.label);
                    egui::ComboBox::from_id_salt("file_browser_filter")
                        .selected_text(current_filter)
                        .show_ui(ui, |ui| {
                            for (idx, filter) in state.filters.iter().enumerate() {
                                if ui.selectable_value(&mut state.filter_index, idx, &filter.label).clicked() {
                                    filter_changed = true;
                                }
                            }
                        });
                    if filter_changed {
                        state.refresh();
                    }
                }

                let btn_text = match state.mode {
                    FileBrowserMode::OpenFile => "Open",
                    FileBrowserMode::SaveFile => "Save",
                };

                let can_accept = match state.mode {
                    FileBrowserMode::OpenFile => state.selected_file.is_some(),
                    FileBrowserMode::SaveFile => !state.file_name_input.trim().is_empty(),
                };

                if ui.add_enabled(can_accept, egui::Button::new(btn_text)).clicked() {
                    state.accept_selection();
                }

                if ui.button("Cancel").clicked() {
                    state.open = false;
                }
            });
        });

    state.open = open;
}

/// Renders clickable breadcrumb segments for the current directory path.
fn render_breadcrumbs(ui: &mut egui::Ui, state: &mut FileBrowserState) {
    let mut jump_target = None;
    let mut accum = PathBuf::new();

    for comp in state.current_dir.components() {
        match comp {
            Component::RootDir => {
                accum.push("/");
                if ui.small_button("/").on_hover_text("Jump to /").clicked() {
                    jump_target = Some(accum.clone());
                }
            }
            Component::Prefix(prefix) => {
                accum.push(comp.as_os_str());
                let prefix_str = prefix.as_os_str().to_string_lossy();
                if ui.small_button(prefix_str.as_ref()).clicked() {
                    jump_target = Some(accum.clone());
                }
            }
            Component::Normal(seg) => {
                accum.push(seg);
                let seg_str = seg.to_string_lossy();
                ui.label("›");
                if ui.small_button(seg_str.as_ref()).clicked() {
                    jump_target = Some(accum.clone());
                }
            }
            _ => {}
        }
    }

    if let Some(target) = jump_target {
        state.navigate_to(target);
    }
}

fn get_file_icon(name: &str) -> &'static str {
    let ext = name.split('.').last().unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "pdf" => "📕",
        "svg" => "🎨",
        "obj" | "fbx" | "gltf" | "glb" => "🧊",
        "dxf" => "📐",
        "png" | "jpg" | "jpeg" => "🖼",
        "txt" | "md" | "json" => "📝",
        _ => "📄",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_filter_matching() {
        let filter = FileFilter::new("Patterns", &["pdf", "svg"]);
        assert!(filter.matches(Path::new("shirt.pdf")));
        assert!(filter.matches(Path::new("pant.SVG")));
        assert!(!filter.matches(Path::new("avatar.obj")));

        let all = FileFilter::new("All", &["*"]);
        assert!(all.matches(Path::new("anything.xyz")));
    }

    #[test]
    fn bookmarks_and_drives_are_detected() {
        let bms = detect_bookmarks();
        assert!(!bms.is_empty(), "Current project should be detected as a bookmark");

        let drives = detect_drives();
        assert!(!drives.is_empty(), "Root drive should be detected");
    }

    #[test]
    fn navigation_and_history() {
        let mut state = FileBrowserState::default();
        let cur = state.current_dir.clone();
        if let Some(parent) = cur.parent() {
            state.navigate_to(parent.to_path_buf());
            assert_eq!(state.current_dir, parent);
            assert!(state.can_go_back());
            state.go_back();
            assert_eq!(state.current_dir, cur);
        }
    }
}

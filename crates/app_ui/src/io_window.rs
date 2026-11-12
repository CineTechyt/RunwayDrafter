//! The import/export window: avatar geometry in, pattern and garment out.
//!
//! The UI only collects paths and options; reading and writing files is
//! the application's job (see [`UiCommand::ImportAvatar`] and
//! [`UiCommand::Export`]), and its outcome comes back in
//! [`crate::state::IoState::status`].

use std::path::PathBuf;

use crate::state::{EditorState, ExportFormat, UiCommand, UiOutput};

pub fn show(ctx: &egui::Context, editor: &mut EditorState, output: &mut UiOutput) {
    let io = &mut editor.io;
    let mut open = io.window_open;
    egui::Window::new("Import / Export").open(&mut open).default_width(420.0).resizable(true).show(ctx, |ui| {
        // --- avatar in -------------------------------------------------------
        ui.strong("Import avatar");
        ui.weak("OBJ, glTF (.gltf/.glb) or FBX. Units and up axis are detected; the figure is placed standing on the floor.");
        ui.horizontal(|ui| {
            ui.label("File");
            ui.add(egui::TextEdit::singleline(&mut io.avatar_path).hint_text("/path/to/avatar.obj").desired_width(ui.available_width() - 85.0));
            if ui.button("📂 Browse…").on_hover_text("Open interactive file browser").clicked() {
                editor.file_browser.open_for_avatar();
            }
        });
        ui.horizontal(|ui| {
            if ui.add_enabled(!io.avatar_path.trim().is_empty(), egui::Button::new("Import")).clicked() {
                output.commands.push(UiCommand::ImportAvatar(PathBuf::from(io.avatar_path.trim())));
            }
            if ui.add_enabled(io.avatar_imported, egui::Button::new("Back to built-in avatar")).clicked() {
                output.commands.push(UiCommand::RestoreAvatar);
            }
        });
        if io.avatar_imported {
            ui.weak("An imported avatar is fixed geometry: measurements and poses are not editable.");
        }

        // --- pattern in ------------------------------------------------------
        ui.add_space(10.0);
        ui.separator();
        ui.strong("Import Pattern (PDF / SVG)");
        ui.weak("Load vector patterns, inspect and select cutting lines vs seam allowances, and create a project.");
        if ui.button("Open Pattern Importer…").clicked() {
            editor.pattern_import.window_open = true;
        }

        // --- pattern and garment out --------------------------------------------
        ui.add_space(10.0);
        ui.separator();
        ui.strong("Export");
        egui::Grid::new("export options").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
            ui.label("Format");
            egui::ComboBox::from_id_salt("export format").selected_text(io.export_format.label()).show_ui(ui, |ui| {
                for format in ExportFormat::ALL {
                    ui.selectable_value(&mut io.export_format, format, format.label());
                }
            });
            ui.end_row();

            match io.export_format {
                ExportFormat::DxfAama | ExportFormat::DxfAstm => {
                    ui.label("Size");
                    ui.add(egui::TextEdit::singleline(&mut io.sample_size).desired_width(60.0));
                    ui.end_row();
                }
                ExportFormat::Obj | ExportFormat::Fbx => {
                    ui.label("UV map");
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut io.pattern_uvs, false, "Fabric tiles");
                        ui.radio_value(&mut io.pattern_uvs, true, "Pattern layout (atlas)");
                    });
                    ui.end_row();
                }
            }

            ui.label("File");
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut io.export_path).hint_text("/path/to/file").desired_width(ui.available_width() - 85.0));
                let default_name = io.export_path.trim();
                let name = if default_name.is_empty() { "garment" } else { default_name };
                if ui.button("📂 Browse…").on_hover_text("Choose export destination").clicked() {
                    editor.file_browser.open_for_export(name, io.export_format.extension());
                }
            });
            ui.end_row();
        });
        let extension = io.export_format.extension();
        ui.horizontal(|ui| {
            if ui.add_enabled(!io.export_path.trim().is_empty(), egui::Button::new("Export")).clicked() {
                // The format decides the extension.
                output.commands.push(UiCommand::Export(PathBuf::from(io.export_path.trim()).with_extension(extension)));
            }
            ui.weak(match io.export_format {
                ExportFormat::DxfAama | ExportFormat::DxfAstm => "Cut line, sew line, notches and grain line per piece, in mm.",
                ExportFormat::Obj => "The garment as draped, with normals and UVs; also writes a .mtl.",
                ExportFormat::Fbx => "The garment as draped, one mesh per piece, with normals and UVs.",
            });
        });

        // --- outcome ----------------------------------------------------------------
        if let Some(status) = &io.status {
            ui.add_space(6.0);
            match status {
                Ok(message) => ui.colored_label(egui::Color32::from_rgb(120, 200, 120), message),
                Err(message) => ui.colored_label(egui::Color32::from_rgb(240, 110, 100), message),
            };
        }
    });
    io.window_open = open;
}

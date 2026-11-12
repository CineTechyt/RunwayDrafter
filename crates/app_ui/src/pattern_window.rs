//! The Pattern Import window: inspects and configures imported PDF & SVG patterns,
//! lets the user classify lines as cutting lines or seam allowance (sew line),
//! and creates a new project or adds pieces to the active project.

use std::path::PathBuf;

use glam::Vec2;
use io_cad::LineRole;

use crate::state::{EditorState, ScaleMode, UiCommand, UiOutput};

pub fn browse_for_pattern(import: &mut crate::state::PatternImportState, output: &mut UiOutput) {
    if let Some(path) = rfd::FileDialog::new()
        .set_title("Open pattern file")
        .add_filter("Pattern files", &["pdf", "svg"])
        .add_filter("PDF", &["pdf"])
        .add_filter("SVG", &["svg"])
        .pick_file()
    {
        import.file_path = path.to_string_lossy().to_string();
        output.commands.push(UiCommand::InspectPatternFile(path));
    }
}

pub fn show(ctx: &egui::Context, editor: &mut EditorState, output: &mut UiOutput) {
    let import = &mut editor.pattern_import;
    let mut open = import.window_open;

    egui::Window::new("Import Pattern (PDF / SVG)")
        .open(&mut open)
        .default_width(760.0)
        .default_height(700.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.strong("Vector pattern import");

            // --- File path and Load button ---
            ui.horizontal(|ui| {
                ui.label("File");
                ui.add(
                    egui::TextEdit::singleline(&mut import.file_path)
                        .hint_text("/path/to/pattern.pdf or pattern.svg")
                        .desired_width(ui.available_width() - 170.0),
                );
                if ui
                    .button("📂 Browse…")
                    .on_hover_text("Open the system file picker")
                    .clicked()
                {
                    browse_for_pattern(import, output);
                }
                let can_load = !import.file_path.trim().is_empty();
                if ui
                    .add_enabled(can_load, egui::Button::new("Inspect"))
                    .clicked()
                {
                    output
                        .commands
                        .push(UiCommand::InspectPatternFile(PathBuf::from(
                            import.file_path.trim(),
                        )));
                }
            });

            egui::CollapsingHeader::new("Import settings")
                .default_open(false)
                .show(ui, |ui| {
                    egui::Grid::new("pattern_import_options")
                        .num_columns(2)
                        .spacing([12.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("Scale");
                            ui.horizontal(|ui| {
                                egui::ComboBox::from_id_salt("scale_mode")
                                    .selected_text(import.scale_mode.label())
                                    .show_ui(ui, |ui| {
                                        for mode in ScaleMode::ALL {
                                            ui.selectable_value(
                                                &mut import.scale_mode,
                                                mode,
                                                mode.label(),
                                            );
                                        }
                                    });
                                if import.scale_mode == ScaleMode::Custom {
                                    ui.add(
                                        egui::DragValue::new(&mut import.custom_scale)
                                            .speed(0.01)
                                            .range(0.001..=100.0)
                                            .prefix("×"),
                                    );
                                }
                            });
                            ui.end_row();

                            ui.label("Orientation");
                            ui.checkbox(&mut import.flip_y, "Flip Y");
                            ui.end_row();

                            ui.label("Default seam allowance");
                            ui.add(
                                egui::DragValue::new(&mut import.default_allowance_mm)
                                    .speed(0.25)
                                    .range(0.0..=50.0)
                                    .suffix(" mm"),
                            );
                            ui.end_row();
                        });
                });

            // --- Pattern pieces inspector ---
            if let Some(pattern) = &mut import.pattern {
                if import.selected_piece_index >= pattern.pieces.len() {
                    import.selected_piece_index = 0;
                }

                show_pattern_preview(ui, pattern, &mut import.selected_piece_index);
                ui.horizontal_wrapped(|ui| {
                    for (piece_idx, piece) in pattern.pieces.iter().enumerate() {
                        ui.selectable_value(
                            &mut import.selected_piece_index,
                            piece_idx,
                            &piece.name,
                        );
                    }
                });

                ui.horizontal(|ui| {
                    ui.strong(format!("Detected pieces: {}", pattern.pieces.len()));
                    if ui
                        .button("⇄ Swap all Cut & Sew lines")
                        .on_hover_text("Swaps the cutting line and sewing line on all pieces")
                        .clicked()
                    {
                        for p in &mut pattern.pieces {
                            p.swap_roles();
                        }
                    }
                });

                if let Some((piece_idx, piece)) = pattern
                    .pieces
                    .iter_mut()
                    .enumerate()
                    .find(|(index, _)| *index == import.selected_piece_index)
                {
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut piece.enabled, "Import");
                        ui.add(egui::TextEdit::singleline(&mut piece.name).desired_width(160.0));
                        ui.label("Allowance");
                        ui.add(
                            egui::DragValue::new(&mut piece.seam_allowance_mm)
                                .speed(0.25)
                                .range(0.0..=50.0)
                                .suffix(" mm"),
                        );
                        if ui
                            .small_button("⇄ Swap roles")
                            .on_hover_text("Swap cutting line and sew line for this piece")
                            .clicked()
                        {
                            piece.swap_roles();
                        }
                    });

                    egui::ScrollArea::vertical()
                        .max_height(150.0)
                        .show(ui, |ui| {
                            for (loop_idx, lp) in piece.loops.iter_mut().enumerate() {
                                ui.horizontal(|ui| {
                                    ui.label(format!(
                                        "Loop {} · {} pts",
                                        loop_idx + 1,
                                        lp.outline.edge_count()
                                    ));
                                    let (min, max) = lp.bounds;
                                    let size = max - min;
                                    ui.weak(format!(
                                        "{:.0}×{:.0} mm · {:.0} cm²",
                                        size.x,
                                        size.y,
                                        lp.area / 100.0
                                    ));
                                    egui::ComboBox::from_id_salt((
                                        "loop_role",
                                        piece_idx,
                                        loop_idx,
                                    ))
                                    .selected_text(lp.role.label())
                                    .show_ui(ui, |ui| {
                                        for role in LineRole::ALL {
                                            ui.selectable_value(&mut lp.role, role, role.label());
                                        }
                                    });
                                });
                            }
                        });
                }

                ui.separator();
                ui.horizontal(|ui| {
                    if ui
                        .button("✨ Create Project from Pattern")
                        .on_hover_text("Replaces current pattern with the imported pieces")
                        .clicked()
                    {
                        output.commands.push(UiCommand::ApplyImportedPattern {
                            pattern: Box::new(pattern.clone()),
                            replace_project: true,
                        });
                    }

                    if ui
                        .button("➕ Add Pieces to Project")
                        .on_hover_text("Appends the imported pieces to the active pattern document")
                        .clicked()
                    {
                        output.commands.push(UiCommand::ApplyImportedPattern {
                            pattern: Box::new(pattern.clone()),
                            replace_project: false,
                        });
                    }
                });
            } else {
                show_empty_preview(ui);
            }

            // --- Status outcome ---
            if let Some(status) = &import.status {
                ui.add_space(6.0);
                match status {
                    Ok(msg) => ui.colored_label(egui::Color32::from_rgb(120, 200, 120), msg),
                    Err(why) => ui.colored_label(egui::Color32::from_rgb(240, 110, 100), why),
                };
            }
        });

    import.window_open = open;
}

fn show_pattern_preview(
    ui: &mut egui::Ui,
    pattern: &io_cad::ImportedPattern,
    selected_piece: &mut usize,
) {
    let available = ui.available_width().max(120.0);
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(available, 260.0), egui::Sense::click());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);

    let bounds: Option<(Vec2, Vec2)> = pattern
        .pieces
        .iter()
        .flat_map(|piece| piece.loops.iter())
        .fold(None, |bounds, lp| {
            Some(match bounds {
                None => lp.bounds,
                Some((min, max)) => (min.min(lp.bounds.0), max.max(lp.bounds.1)),
            })
        });
    let Some((min, max)) = bounds else {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "No closed pattern pieces",
            egui::FontId::proportional(14.0),
            ui.visuals().weak_text_color(),
        );
        return;
    };

    let size = max - min;
    let scale =
        ((rect.width() - 28.0) / size.x.max(1.0)).min((rect.height() - 28.0) / size.y.max(1.0));
    let center = (min + max) * 0.5;
    let to_screen = |point: Vec2| {
        rect.center() + egui::vec2((point.x - center.x) * scale, -(point.y - center.y) * scale)
    };

    if let Some(pointer) = response
        .interact_pointer_pos()
        .filter(|_| response.clicked())
    {
        let world = center
            + Vec2::new(
                (pointer.x - rect.center().x) / scale,
                -(pointer.y - rect.center().y) / scale,
            );
        if let Some((index, _)) = pattern.pieces.iter().enumerate().rev().find(|(_, piece)| {
            piece.loops.iter().any(|lp| {
                let points = lp.outline.flatten(1.0).points;
                cad_core::geometry::point_in_polygon(world, &points)
            })
        }) {
            *selected_piece = index;
        }
    }

    for (piece_idx, piece) in pattern.pieces.iter().enumerate() {
        for lp in &piece.loops {
            let points = lp.outline.flatten(1.0).points;
            if points.len() < 2 {
                continue;
            }
            let color = if !piece.enabled {
                egui::Color32::DARK_GRAY
            } else {
                match lp.role {
                    LineRole::CuttingLine => egui::Color32::from_rgb(240, 150, 90),
                    LineRole::SeamAllowance => egui::Color32::from_rgb(80, 195, 220),
                    LineRole::Internal => egui::Color32::from_rgb(155, 165, 165),
                }
            };
            let stroke = egui::Stroke::new(
                if piece_idx == *selected_piece {
                    2.5
                } else {
                    1.2
                },
                color,
            );
            let screen_points: Vec<_> = points.iter().map(|point| to_screen(*point)).collect();
            for pair in screen_points.windows(2) {
                painter.line_segment([pair[0], pair[1]], stroke);
            }
            painter.line_segment([*screen_points.last().unwrap(), screen_points[0]], stroke);
        }

        if let Some(lp) = piece.loops.first() {
            let label_pos = to_screen((lp.bounds.0 + lp.bounds.1) * 0.5);
            painter.text(
                label_pos,
                egui::Align2::CENTER_CENTER,
                &piece.name,
                egui::FontId::proportional(12.0),
                ui.visuals().text_color(),
            );
        }
    }

    painter.text(
        rect.left_top() + egui::vec2(8.0, 8.0),
        egui::Align2::LEFT_TOP,
        "Cut line",
        egui::FontId::proportional(11.0),
        egui::Color32::from_rgb(240, 150, 90),
    );
    painter.text(
        rect.left_top() + egui::vec2(78.0, 8.0),
        egui::Align2::LEFT_TOP,
        "Sew line",
        egui::FontId::proportional(11.0),
        egui::Color32::from_rgb(80, 195, 220),
    );
}

fn show_empty_preview(ui: &mut egui::Ui) {
    let available = ui.available_width().max(120.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(available, 260.0), egui::Sense::hover());
    ui.painter()
        .rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        "Pattern preview",
        egui::FontId::proportional(14.0),
        ui.visuals().weak_text_color(),
    );
}

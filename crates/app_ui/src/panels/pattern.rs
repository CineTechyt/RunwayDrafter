//! Properties of the selected pattern piece.

use cad_core::geometry::{self, Curve};
use cad_core::{DocumentChange, EdgeRef, Notch, NotchKind, UvMapping};

use super::{Panel, PanelContext, property_grid};

#[derive(Debug, Default)]
pub struct PatternPropertiesPanel;

impl Panel for PatternPropertiesPanel {
    fn title(&self) -> &'static str {
        "Pattern"
    }

    fn ui(&mut self, ui: &mut egui::Ui, cx: &mut PanelContext<'_>) {
        ui.heading("Pieces");
        let selected = &mut cx.editor.selected_piece;
        if selected.is_some_and(|id| cx.document.piece(id).is_none()) {
            *selected = None;
        }
        for piece in cx.document.pieces() {
            ui.selectable_value(selected, Some(piece.id), &piece.name);
        }
        if ui
            .button("➕ Add outer layer")
            .on_hover_text("Copies the outermost garment, 6 % larger, into the next layer: a second garment worn over it.")
            .clicked()
        {
            cx.editor.timeline.stop();
            cx.output.commands.push(crate::state::UiCommand::AddOuterLayer);
        }
        ui.separator();

        let Some(id) = *selected else {
            ui.weak("Select a piece here, on the 2D canvas or in the 3D view.");
            return;
        };

        let mut duplicate_mirrored = None;
        ui.horizontal(|ui| {
            if ui.button("⧉ Duplicate").clicked() {
                duplicate_mirrored = Some(false);
            }
            if ui.button("↔ Mirror copy").clicked() {
                duplicate_mirrored = Some(true);
            }
        });
        if let Some(mirrored) = duplicate_mirrored {
            if let Some(new_id) = cx.document.duplicate_piece(id, mirrored) {
                *selected = Some(new_id);
                cx.editor.timeline.stop();
                cx.output.changes.push(DocumentChange::Everything);
                cx.output.commands.push(crate::state::UiCommand::ArrangePieces);
                cx.output.commands.push(crate::state::UiCommand::FitCanvas);
            }
            return;
        }

        // Snapshot what the widgets need from the rest of the document before
        // borrowing the piece mutably.
        let fabrics: Vec<_> = cx
            .document
            .fabrics()
            .iter()
            .map(|f| (f.id, f.name.clone()))
            .collect();
        let edge_count = cx.document.piece(id).map_or(0, |p| p.outline.edge_count());
        let seam_partner: Vec<Option<String>> = (0..edge_count)
            .map(|edge| {
                let this = EdgeRef::new(id, edge);
                let seam = cx.document.seam_of(this)?;
                let other = if seam.a == this { seam.b } else { seam.a };
                Some(format!(
                    "{} · edge {}",
                    cx.document.piece(other.piece)?.name,
                    other.edge
                ))
            })
            .collect();
        // let selected_edge = cx.editor.selected_edge; // Obsolete line removed

        let Some(piece) = cx.document.piece_mut(id) else {
            return;
        };
        let mut changes: Vec<DocumentChange> = Vec::new();
        let mut note = |changed: bool, change: DocumentChange| {
            if changed && !changes.contains(&change) {
                changes.push(change);
            }
        };

        // --- general ---------------------------------------------------------
        let flat = piece.outline.flatten(0.25);
        let area_cm2 = geometry::signed_area(&flat.points).abs() / 100.0;
        property_grid(ui, "piece general", |ui| {
            ui.label("Name");
            note(
                ui.text_edit_singleline(&mut piece.name).changed(),
                DocumentChange::PieceAnnotations(id),
            );
            ui.end_row();

            ui.label("Fabric");
            let current = fabrics
                .iter()
                .find(|(f, _)| *f == piece.fabric)
                .map_or("—", |(_, name)| name.as_str());
            egui::ComboBox::from_id_salt("piece fabric")
                .selected_text(current)
                .show_ui(ui, |ui| {
                    for (fabric, name) in &fabrics {
                        note(
                            ui.selectable_value(&mut piece.fabric, *fabric, name)
                                .changed(),
                            DocumentChange::PieceFabric(id),
                        );
                    }
                });
            ui.end_row();

            ui.label("Layer");
            note(
                ui.add(egui::DragValue::new(&mut piece.layer).range(1..=8))
                    .on_hover_text("Counted outwards from the body: 1 = shirt, 2 = jacket over it. Higher layers are kept outside lower ones.")
                    .changed(),
                DocumentChange::PieceAnnotations(id),
            );
            ui.end_row();

            ui.label("Area");
            ui.label(format!("{area_cm2:.0} cm²"));
            ui.end_row();
        });

        // --- seam allowance & edges -------------------------------------------
        ui.add_space(8.0);
        egui::CollapsingHeader::new("Seam allowance & line roles").default_open(true).show(ui, |ui| {
            let annotation = DocumentChange::PieceAnnotations(id);
            let geom_change = DocumentChange::PieceGeometry(id);

            let mut convert_to_sew = false;
            let mut convert_to_cut = false;

            ui.horizontal(|ui| {
                note(ui.checkbox(&mut piece.seam_allowance.enabled, "Enabled").changed(), annotation);
                ui.label("Default width");
                let drag = egui::DragValue::new(&mut piece.seam_allowance.default_width).range(0.0..=100.0).speed(0.25).suffix(" mm");
                note(ui.add(drag).changed(), annotation);
            });

            ui.add_space(3.0);
            ui.horizontal(|ui| {
                if ui
                    .button("Convert: Cutting Line → Sew Line")
                    .on_hover_text("If this piece was imported as a cutting line, shrink it inward by the seam allowance so the sewing line stitches correctly")
                    .clicked()
                {
                    convert_to_sew = true;
                }

                if ui
                    .button("Convert: Sew Line → Cut Line")
                    .on_hover_text("Expand the piece outline outwards by the seam allowance")
                    .clicked()
                {
                    convert_to_cut = true;
                }
            });

            if convert_to_sew {
                piece.convert_cutting_line_to_sew_line();
                note(true, geom_change);
            }
            if convert_to_cut {
                piece.convert_sew_line_to_cutting_line();
                note(true, geom_change);
            }

            let sa = &mut piece.seam_allowance;

            if let Some(selected_edge) = cx.editor.selected_edge {
                if selected_edge.piece == id && selected_edge.edge < edge_count {
                    ui.add_space(4.0);
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.strong(format!("Selected Edge {}", selected_edge.edge));
                            let edge_curve = piece.outline.curve(selected_edge.edge);
                            ui.label(format!("{:.1} mm", edge_curve.length(0.1)));

                            let mut width = sa.width(selected_edge.edge);
                            ui.label("Allowance:");
                            if ui.add(egui::DragValue::new(&mut width).range(0.0..=100.0).speed(0.25).suffix(" mm")).changed() {
                                sa.per_edge.insert(selected_edge.edge, width);
                                note(true, annotation);
                            }
                            if sa.per_edge.contains_key(&selected_edge.edge) && ui.small_button("⟲ Default").clicked() {
                                sa.per_edge.remove(&selected_edge.edge);
                                note(true, annotation);
                            }
                        });
                    });
                }
            }

            ui.add_space(4.0);
            egui::Grid::new("edges").num_columns(4).striped(true).spacing([12.0, 4.0]).show(ui, |ui| {
                for header in ["Edge", "Length", "Allowance", "Sewn to"] {
                    ui.weak(header);
                }
                ui.end_row();
                for edge in 0..edge_count {
                    let is_selected_edge = cx.editor.selected_edge == Some(EdgeRef::new(id, edge));
                    let edge_label = format!("Edge {edge}");
                    if ui.selectable_label(is_selected_edge, edge_label).clicked() {
                        cx.editor.selected_edge = Some(EdgeRef::new(id, edge));
                    }
                    ui.label(format!("{:.1} mm", piece.outline.curve(edge).length(0.1)));
                    let mut width = sa.width(edge);
                    let overridden = sa.per_edge.contains_key(&edge);
                    ui.horizontal(|ui| {
                        if ui.add(egui::DragValue::new(&mut width).range(0.0..=100.0).speed(0.25)).changed() {
                            sa.per_edge.insert(edge, width);
                            note(true, annotation);
                        }
                        if overridden && ui.small_button("⟲").on_hover_text("Use the default width").clicked() {
                            sa.per_edge.remove(&edge);
                            note(true, annotation);
                        }
                    });
                    ui.label(seam_partner[edge as usize].as_deref().unwrap_or("—"));
                    ui.end_row();
                }
            });

        });

        // --- grainline & UV ---------------------------------------------------
        egui::CollapsingHeader::new("Grainline & texture mapping")
            .default_open(true)
            .show(ui, |ui| {
                property_grid(ui, "grain", |ui| {
                    let annotation = DocumentChange::PieceAnnotations(id);
                    ui.label("Grain angle");
                    let mut degrees = piece.grainline.angle.to_degrees();
                    if ui
                        .add(egui::DragValue::new(&mut degrees).speed(0.5).suffix("°"))
                        .changed()
                    {
                        piece.grainline.angle = degrees.to_radians();
                        note(true, annotation);
                    }
                    ui.end_row();
                    ui.label("Arrow length");
                    let drag = egui::DragValue::new(&mut piece.grainline.length)
                        .range(10.0..=5000.0)
                        .suffix(" mm");
                    note(ui.add(drag).changed(), annotation);
                    ui.end_row();

                    let uv_change = DocumentChange::PieceUv(id);
                    ui.label("Texture tile");
                    ui.horizontal(|ui| {
                        for axis in [&mut piece.uv.tile_size.x, &mut piece.uv.tile_size.y] {
                            note(
                                ui.add(
                                    egui::DragValue::new(axis).range(1.0..=5000.0).suffix(" mm"),
                                )
                                .changed(),
                                uv_change,
                            );
                        }
                    });
                    ui.end_row();
                    ui.label("Texture rotation");
                    let mut degrees = piece.uv.rotation.to_degrees();
                    if ui
                        .add(egui::DragValue::new(&mut degrees).speed(0.5).suffix("°"))
                        .changed()
                    {
                        piece.uv.rotation = degrees.to_radians();
                        note(true, uv_change);
                    }
                    ui.end_row();
                    ui.label("");
                    if ui.button("Align texture to grain").clicked() {
                        piece.uv = UvMapping {
                            tile_size: piece.uv.tile_size,
                            ..UvMapping::aligned_to_grain(&piece.grainline)
                        };
                        note(true, uv_change);
                    }
                    ui.end_row();
                });
            });

        // --- notches ------------------------------------------------------------
        egui::CollapsingHeader::new("Notches")
            .default_open(false)
            .show(ui, |ui| {
                let annotation = DocumentChange::PieceAnnotations(id);
                let mut remove = None;
                egui::Grid::new("notches")
                    .num_columns(5)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        for (index, notch) in piece.notches.iter_mut().enumerate() {
                            let max_edge = edge_count.saturating_sub(1);
                            note(
                                ui.add(
                                    egui::DragValue::new(&mut notch.edge)
                                        .range(0..=max_edge)
                                        .prefix("edge "),
                                )
                                .changed(),
                                annotation,
                            );
                            note(
                                ui.add(
                                    egui::Slider::new(&mut notch.t, 0.0..=1.0).show_value(false),
                                )
                                .changed(),
                                annotation,
                            );
                            egui::ComboBox::from_id_salt(("notch kind", index))
                                .width(48.0)
                                .selected_text(format!("{:?}", notch.kind))
                                .show_ui(ui, |ui| {
                                    for kind in [NotchKind::Slit, NotchKind::V, NotchKind::T] {
                                        note(
                                            ui.selectable_value(
                                                &mut notch.kind,
                                                kind,
                                                format!("{kind:?}"),
                                            )
                                            .changed(),
                                            annotation,
                                        );
                                    }
                                });
                            note(
                                ui.add(
                                    egui::DragValue::new(&mut notch.depth)
                                        .range(1.0..=30.0)
                                        .speed(0.1)
                                        .suffix(" mm"),
                                )
                                .changed(),
                                annotation,
                            );
                            if ui.small_button("🗑").clicked() {
                                remove = Some(index);
                            }
                            ui.end_row();
                        }
                    });
                if let Some(index) = remove {
                    piece.notches.remove(index);
                    note(true, annotation);
                }
                if ui.button("＋ Add notch").clicked() {
                    piece.notches.push(Notch {
                        edge: 0,
                        t: 0.5,
                        kind: NotchKind::Slit,
                        depth: 6.0,
                    });
                    note(true, annotation);
                }
            });

        // --- arrangement ----------------------------------------------------------
        egui::CollapsingHeader::new("3D arrangement")
            .default_open(false)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Position");
                    let t = &mut piece.arrangement.translation;
                    for (axis, prefix) in [(&mut t.x, "x "), (&mut t.y, "y "), (&mut t.z, "z ")] {
                        let drag = egui::DragValue::new(axis)
                            .speed(0.005)
                            .prefix(prefix)
                            .suffix(" m");
                        note(ui.add(drag).changed(), DocumentChange::PieceArrangement(id));
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Rotation");
                    let (x, y, z) = piece.arrangement.rotation.to_euler(glam::EulerRot::XYZ);
                    let mut degrees = [x.to_degrees(), y.to_degrees(), z.to_degrees()];
                    let mut changed = false;
                    for (angle, axis) in degrees.iter_mut().zip(["x ", "y ", "z "]) {
                        changed |= ui
                            .add(
                                egui::DragValue::new(angle)
                                    .speed(0.5)
                                    .prefix(axis)
                                    .suffix("°"),
                            )
                            .changed();
                    }
                    if changed {
                        piece.arrangement.rotation = glam::Quat::from_euler(
                            glam::EulerRot::XYZ,
                            degrees[0].to_radians(),
                            degrees[1].to_radians(),
                            degrees[2].to_radians(),
                        );
                        note(true, DocumentChange::PieceArrangement(id));
                    }
                });
                ui.horizontal(|ui| {
                    if ui
                        .button("Auto-arrange around body")
                        .on_hover_text(
                            "Place pattern pieces around the current avatar before draping",
                        )
                        .clicked()
                    {
                        cx.output
                            .commands
                            .push(crate::state::UiCommand::ArrangePieces);
                    }
                    if ui.button("Reset piece transform").clicked() {
                        piece.arrangement.translation = glam::Vec3::ZERO;
                        piece.arrangement.rotation = glam::Quat::IDENTITY;
                        note(true, DocumentChange::PieceArrangement(id));
                    }
                });
            });

        if !changes.is_empty() {
            cx.document.touch();
            cx.output.changes.extend(changes);
        }
    }
}

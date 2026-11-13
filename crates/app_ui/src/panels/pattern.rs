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
            if ui.button("Duplicate").clicked() {
                duplicate_mirrored = Some(false);
            }
        });
        if let Some(mirrored) = duplicate_mirrored {
            if let Some(new_id) = cx.document.duplicate_piece(id, mirrored) {
                *selected = Some(new_id);
                cx.editor.timeline.stop();
                cx.output.changes.push(DocumentChange::Everything);
                cx.output
                    .commands
                    .push(crate::state::UiCommand::ArrangePieces);
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
        // Every seam on this piece: (id, partner name, length here, length there).
        let seams: Vec<(cad_core::SeamId, String, f32, f32)> = cx
            .document
            .seams()
            .iter()
            .filter(|s| s.a.piece == id || s.b.piece == id)
            .filter_map(|s| {
                let (mine, theirs) = if s.a.piece == id {
                    ((s.a.piece, s.a_span()), (s.b.piece, s.b_span()))
                } else {
                    ((s.b.piece, s.b_span()), (s.a.piece, s.a_span()))
                };
                let here = cx.document.piece(mine.0)?.outline.span_length(mine.1);
                let other = cx.document.piece(theirs.0)?;
                let name = if theirs.0 == id {
                    "itself".to_owned()
                } else {
                    other.name.clone()
                };
                Some((s.id, name, here, other.outline.span_length(theirs.1)))
            })
            .collect();
        // The piece's lines in order: (span, length, sewn?).
        let lines: Vec<(cad_core::EdgeSpan, f32, bool)> =
            cx.document.piece(id).map_or_else(Vec::new, |piece| {
                piece
                    .outline
                    .lines()
                    .into_iter()
                    .map(|span| {
                        let sewn = span
                            .edges(edge_count)
                            .any(|edge| cx.document.seam_of(EdgeRef::new(id, edge)).is_some());
                        (span, piece.outline.span_length(span), sewn)
                    })
                    .collect()
            });
        // Where the other pieces sit around the body, for the placement dial.
        let others: Vec<f32> = cx
            .document
            .pieces()
            .iter()
            .filter(|p| p.id != id)
            .map(|p| p.placement.angle)
            .collect();
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

        // Every buttonhole of the garment, for the "done up through" lists.
        let holes: Vec<(cad_core::ButtonholeRef, String)> = cx
            .document
            .pieces()
            .iter()
            .flat_map(|p| {
                (0..p.buttonholes.len()).map(move |index| {
                    (
                        cad_core::ButtonholeRef {
                            piece: p.id,
                            index: index as u32,
                        },
                        format!("{} · hole {}", p.name, index + 1),
                    )
                })
            })
            .collect();
        let taken: Vec<cad_core::ButtonholeRef> = cx
            .document
            .pieces()
            .iter()
            .flat_map(|p| &p.buttons)
            .filter_map(|b| b.fastened_to)
            .collect();
        let mut button_action = None;

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

        ui.add_space(8.0);
        egui::CollapsingHeader::new(format!("Buttons ({}) and buttonholes ({})", piece.buttons.len(), piece.buttonholes.len()))
            .default_open(true)
            .show(ui, |ui| {
                ui.weak("Place them with the toolbar tools on the 2D or 3D view. A button done up through a buttonhole holds the two together in the simulation.");
                let mut remove_button = None;
                for (index, button) in piece.buttons.iter_mut().enumerate() {
                    ui.push_id(("button", index), |ui| {
                        ui.horizontal(|ui| {
                            ui.strong(format!("Button {}", index + 1));
                            let mut moved = ui.add(egui::DragValue::new(&mut button.position.x).speed(0.5).prefix("x ").suffix(" mm")).changed();
                            moved |= ui.add(egui::DragValue::new(&mut button.position.y).speed(0.5).prefix("y ").suffix(" mm")).changed();
                            // The fastening stitch is re-picked where the button now is.
                            note(moved, if button.fastened_to.is_some() { DocumentChange::Seams } else { DocumentChange::PieceAnnotations(id) });
                            if ui.small_button("🗑").on_hover_text("Remove the button").clicked() {
                                remove_button = Some(index);
                            }
                        });
                        ui.horizontal(|ui| {
                            note(ui.add(egui::DragValue::new(&mut button.diameter).range(6.0..=60.0).speed(0.25).suffix(" mm")).on_hover_text("Diameter").changed(), DocumentChange::PieceAnnotations(id));
                            egui::ComboBox::from_id_salt("style").width(76.0).selected_text(button.style.label()).show_ui(ui, |ui| {
                                for style in cad_core::ButtonStyle::ALL {
                                    note(ui.selectable_value(&mut button.style, style, style.label()).changed(), DocumentChange::PieceAnnotations(id));
                                }
                            });
                            egui::ComboBox::from_id_salt("finish").width(110.0).selected_text(button.finish.label()).show_ui(ui, |ui| {
                                for finish in cad_core::ButtonFinish::ALL {
                                    if ui.selectable_value(&mut button.finish, finish, finish.label()).changed() {
                                        button.color = finish.default_color();
                                        note(true, DocumentChange::PieceAnnotations(id));
                                    }
                                }
                            });
                        });
                        ui.horizontal(|ui| {
                            ui.label("Colour");
                            note(ui.color_edit_button_rgb(&mut button.color).changed(), DocumentChange::PieceAnnotations(id));
                            if button.style.holes() > 0 {
                                ui.label("Thread");
                                note(ui.color_edit_button_rgb(&mut button.thread_color).changed(), DocumentChange::PieceAnnotations(id));
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Done up through").on_hover_text(
                                "The buttonhole this button closes through. The piece with the holes laps over the one with the buttons and is placed on top when the pieces are arranged.",
                            );
                            let current = button.fastened_to.and_then(|r| holes.iter().find(|(h, _)| *h == r)).map_or("— (open)", |(_, name)| name.as_str());
                            egui::ComboBox::from_id_salt(("fasten", id, index)).selected_text(current).show_ui(ui, |ui| {
                                note(ui.selectable_value(&mut button.fastened_to, None, "— (open)").changed(), DocumentChange::Seams);
                                for (hole, name) in &holes {
                                    let taken = taken.contains(hole) && button.fastened_to != Some(*hole);
                                    let entry = ui.add_enabled(!taken, egui::Button::selectable(button.fastened_to == Some(*hole), name));
                                    if entry.clicked() {
                                        button.fastened_to = Some(*hole);
                                        note(true, DocumentChange::Seams);
                                    }
                                }
                            });
                        });
                    });
                    ui.add_space(2.0);
                }
                if let Some(index) = remove_button {
                    piece.buttons.remove(index);
                    note(true, DocumentChange::Seams);
                }
                if !piece.buttonholes.is_empty() {
                    ui.separator();
                }
                for (index, hole) in piece.buttonholes.iter_mut().enumerate() {
                    ui.push_id(("hole", index), |ui| {
                        ui.horizontal(|ui| {
                            ui.strong(format!("Hole {}", index + 1));
                            let mut moved = ui.add(egui::DragValue::new(&mut hole.position.x).speed(0.5).prefix("x ").suffix(" mm")).changed();
                            moved |= ui.add(egui::DragValue::new(&mut hole.position.y).speed(0.5).prefix("y ").suffix(" mm")).changed();
                            note(moved, DocumentChange::Seams);
                            note(ui.add(egui::DragValue::new(&mut hole.length).range(4.0..=100.0).speed(0.25).suffix(" mm")).on_hover_text("Length").changed(), DocumentChange::PieceAnnotations(id));
                            let mut degrees = hole.angle.to_degrees();
                            if ui.add(egui::DragValue::new(&mut degrees).speed(1.0).suffix("°")).on_hover_text("Angle: 0° across the piece, 90° along it").changed() {
                                hole.angle = degrees.to_radians();
                                note(true, DocumentChange::PieceAnnotations(id));
                            }
                            if ui.small_button("🗑").on_hover_text("Remove the buttonhole").clicked() {
                                button_action = Some(ButtonAction::RemoveHole(index));
                            }
                        });
                    });
                }
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui.button("Fasten all").on_hover_text(
                        "Do up every open button: each piece's buttons go through the free holes of the piece placed nearest to them, in order down the row (top button through the top hole). Arrange the pieces around the avatar first; fix single buttons with “Done up through”.",
                    ).clicked() {
                        button_action = Some(ButtonAction::FastenAll);
                    }
                    if ui.button("Undo all").on_hover_text("Open every button of the garment").clicked() {
                        button_action = Some(ButtonAction::UnfastenAll);
                    }
                });
            });

        // --- placement on the avatar ----------------------------------------------
        ui.add_space(8.0);
        egui::CollapsingHeader::new("Placement on the avatar").default_open(true).show(ui, |ui| {
            let mut changed = false;
            ui.horizontal(|ui| {
                changed |= placement_dial(ui, &mut piece.placement.angle, &others);
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        for (label, angle) in [("Front", 0.0), ("Left", 90.0), ("Back", 180.0), ("Right", -90.0)] {
                            if ui.selectable_label((piece.placement.angle - angle).abs() < 0.5, label).clicked() {
                                piece.placement.angle = angle;
                                changed = true;
                            }
                        }
                    });
                    property_grid(ui, "placement", |ui| {
                        ui.label("Around body");
                        changed |= ui
                            .add(egui::DragValue::new(&mut piece.placement.angle).range(-180.0..=180.0).speed(1.0).suffix("°"))
                            .on_hover_text("0° centre front, 90° the avatar's left side, 180° centre back, -90° its right side")
                            .changed();
                        ui.end_row();
                        ui.label("Height");
                        changed |= ui
                            .add(egui::DragValue::new(&mut piece.placement.height).range(-800.0..=800.0).speed(1.0).suffix(" mm"))
                            .on_hover_text("Raise or lower the piece from where it is placed automatically")
                            .changed();
                        ui.end_row();
                    });
                    ui.weak("Pieces are placed on the avatar and sewn from there. Click the ring to choose a spot.");
                });
            });
            if changed {
                note(true, DocumentChange::PieceAnnotations(id));
                cx.editor.timeline.stop();
                cx.output.commands.push(crate::state::UiCommand::ArrangePieces);
            }
        });

        // --- seams ------------------------------------------------------------------
        egui::CollapsingHeader::new(format!("Seams ({})", seams.len())).default_open(true).show(ui, |ui| {
            if seams.is_empty() {
                ui.weak("Use the sewing tool (S) on the 2D view: click a line, then the line it is sewn to.");
            }
            egui::Grid::new("seams").num_columns(3).striped(true).spacing([10.0, 4.0]).show(ui, |ui| {
                for (seam, partner, here, there) in &seams {
                    ui.label(format!("to {partner}"));
                    let difference = here - there;
                    let text = format!("{here:.0} / {there:.0} mm");
                    if difference.abs() > 3.0 {
                        // Sewing unequal lengths gathers the longer side.
                        ui.colored_label(egui::Color32::from_rgb(240, 180, 90), text)
                            .on_hover_text(format!("The lines differ by {:.1} mm: the longer one will be gathered.", difference.abs()));
                    } else {
                        ui.label(text);
                    }
                    ui.horizontal(|ui| {
                        if ui.small_button("Flip").on_hover_text("Turn the seam around: its start meets the other end").clicked() {
                            cx.output.commands.push(crate::state::UiCommand::FlipSeam(*seam));
                        }
                        if ui.small_button("Remove").clicked() {
                            cx.output.commands.push(crate::state::UiCommand::DeleteSeam(*seam));
                        }
                    });
                    ui.end_row();
                }
            });
        });

        // --- lines -----------------------------------------------------------------
        egui::CollapsingHeader::new(format!("Lines ({})", lines.len())).default_open(false).show(ui, |ui| {
            ui.weak("A line runs from one turn point to the next. Join two neighbours to sew them as one; with the point tool, double-click a point to join or split there.");
            for (index, (span, length, sewn)) in lines.iter().enumerate() {
                ui.horizontal(|ui| {
                    let selected = cx.editor.selected_edge.is_some_and(|e| e.piece == id && span.contains(e.edge, edge_count));
                    if ui.selectable_label(selected, format!("Line {}", index + 1)).clicked() {
                        cx.editor.selected_edge = Some(EdgeRef::new(id, span.first));
                    }
                    ui.label(format!("{length:.0} mm"));
                    if *sewn {
                        ui.weak("sewn");
                    }
                });
                // The turn point between this line and the next.
                if lines.len() > 1 {
                    let next = (span.first + span.count) % edge_count;
                    let label = if index + 1 == lines.len() { "Join last and first line" } else { "Join with the line below" };
                    if ui.small_button(label).clicked() {
                        cx.output.commands.push(crate::state::UiCommand::SetPointKind { piece: id, anchor: next, kind: cad_core::PointKind::Smooth });
                    }
                }
            }
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
                    .button("Convert: Cutting Line ➡ Sew Line")
                    .on_hover_text("If this piece was imported as a cutting line, shrink it inward by the seam allowance so the sewing line stitches correctly")
                    .clicked()
                {
                    convert_to_sew = true;
                }

                if ui
                    .button("Convert: Sew Line ➡ Cut Line")
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
                if ui.button("➕ Add notch").clicked() {
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

        // Edits that reach beyond this piece, now that it is no longer borrowed.
        let structural = match button_action {
            Some(ButtonAction::RemoveHole(index)) => {
                cx.document.remove_buttonhole(id, index).is_some()
            }
            Some(ButtonAction::FastenAll) => cx.document.fasten_buttons(0.25) > 0,
            Some(ButtonAction::UnfastenAll) => cx.document.unfasten_buttons() > 0,
            None => false,
        };
        if structural && !changes.contains(&DocumentChange::Seams) {
            changes.push(DocumentChange::Seams);
        }

        if !changes.is_empty() {
            cx.document.touch();
            cx.output.changes.extend(changes);
        }
    }
}

/// Button edits that touch other pieces than the selected one.
enum ButtonAction {
    RemoveHole(usize),
    FastenAll,
    UnfastenAll,
}

/// A ring seen from above the avatar (front at the bottom): this piece's
/// spot, the other pieces' spots in grey. Click or drag to move it.
/// Returns `true` if the angle changed.
fn placement_dial(ui: &mut egui::Ui, angle: &mut f32, others: &[f32]) -> bool {
    let size = 104.0;
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    let centre = rect.center();
    let radius = size * 0.38;
    let visuals = ui.visuals();
    // Seen from above with the front at the bottom: +x (the avatar's left)
    // is on the right, +z (front) points down.
    let at = |degrees: f32| {
        let a = degrees.to_radians();
        centre + egui::vec2(a.sin(), a.cos()) * radius
    };
    painter.circle_stroke(
        centre,
        radius,
        egui::Stroke::new(1.0, visuals.weak_text_color()),
    );
    // The body's cross-section and the direction it faces.
    painter.add(egui::Shape::ellipse_filled(
        centre,
        egui::vec2(radius * 0.62, radius * 0.4),
        visuals.faint_bg_color.gamma_multiply(2.0),
    ));
    painter.add(egui::Shape::ellipse_stroke(
        centre,
        egui::vec2(radius * 0.62, radius * 0.4),
        egui::Stroke::new(1.0, visuals.weak_text_color()),
    ));
    let small = egui::FontId::proportional(9.0);
    painter.text(
        rect.center_bottom() + egui::vec2(0.0, -1.0),
        egui::Align2::CENTER_BOTTOM,
        "front",
        small.clone(),
        visuals.weak_text_color(),
    );
    painter.text(
        rect.center_top() + egui::vec2(0.0, 1.0),
        egui::Align2::CENTER_TOP,
        "back",
        small.clone(),
        visuals.weak_text_color(),
    );
    painter.text(
        rect.right_center(),
        egui::Align2::RIGHT_CENTER,
        "L",
        small.clone(),
        visuals.weak_text_color(),
    );
    painter.text(
        rect.left_center(),
        egui::Align2::LEFT_CENTER,
        "R",
        small,
        visuals.weak_text_color(),
    );
    for &other in others {
        painter.circle_filled(at(other), 3.0, visuals.weak_text_color());
    }
    let mut changed = false;
    if let Some(pointer) = response
        .interact_pointer_pos()
        .filter(|_| response.clicked() || response.dragged())
    {
        let d = pointer - centre;
        if d.length() > 4.0 {
            let new = d.x.atan2(d.y).to_degrees().round();
            changed = (new - *angle).abs() > 0.1;
            *angle = new;
        }
    }
    painter.circle_filled(at(*angle), 5.5, visuals.selection.bg_fill);
    painter.circle_stroke(
        at(*angle),
        5.5,
        egui::Stroke::new(1.5, visuals.strong_text_color()),
    );
    changed
}

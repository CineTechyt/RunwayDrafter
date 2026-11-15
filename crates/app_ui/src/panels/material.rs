//! Fabric appearance (PBR) and physical properties.

use cad_core::DocumentChange;

use super::{Panel, PanelContext, property_grid};

#[derive(Debug, Default)]
pub struct MaterialPanel {
    weave: super::weave_editor::WeaveEditorState,
}

impl Panel for MaterialPanel {
    fn title(&self) -> &'static str {
        "Materials"
    }

    fn ui(&mut self, ui: &mut egui::Ui, cx: &mut PanelContext<'_>) {
        ui.heading("Fabrics");
        let selected = &mut cx.editor.selected_fabric;
        if selected.is_none_or(|id| cx.document.fabric(id).is_none()) {
            *selected = cx.document.fabrics().first().map(|f| f.id);
        }
        for fabric in cx.document.fabrics() {
            ui.horizontal(|ui| {
                let [r, g, b] = fabric
                    .base_color
                    .map(|c| (c.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8);
                let (swatch, _) =
                    ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
                ui.painter()
                    .rect_filled(swatch, 3.0, egui::Color32::from_rgb(r, g, b));
                ui.selectable_value(selected, Some(fabric.id), &fabric.name);
            });
        }
        if ui.button("➕ New fabric").clicked() {
            let id = cx.document.add_fabric("New fabric", [0.5, 0.5, 0.5]);
            *selected = Some(id);
            cx.output.changes.push(DocumentChange::Fabric(id));
        }
        ui.separator();

        let Some(id) = *selected else { return };
        let Some(fabric) = cx.document.fabric_mut(id) else {
            return;
        };
        let mut changed = false;
        // Edits of the look only, which need no new drape.
        let mut look = false;

        ui.strong("Appearance");
        ui.horizontal(|ui| {
            ui.label("Preset");
            egui::ComboBox::from_id_salt("fabric-preset")
                .selected_text("Choose a fabric…")
                .height(480.0)
                .show_ui(ui, |ui| {
                    // Real cloths: woven from their drafts, or corduroy.
                    ui.weak("Woven and pile fabrics");
                    for preset in cad_core::fabric_library::presets() {
                        if ui.selectable_label(fabric.name == preset.name, &preset.name).clicked() {
                            *fabric = cad_core::Fabric { id: fabric.id, ..preset };
                            changed = true;
                        }
                    }
                    ui.separator();
                    ui.weak("Simple surfaces");
                    for pattern in cad_core::FabricPattern::ALL {
                        if ui
                            .selectable_label(
                                fabric.weave.is_none() && fabric.name == pattern.label(),
                                pattern.label(),
                            )
                            .clicked()
                        {
                            *fabric = preset(fabric.id, pattern);
                            changed = true;
                        }
                    }
                });
        });
        property_grid(ui, "fabric appearance", |ui| {
            ui.label("Name");
            changed |= ui.text_edit_singleline(&mut fabric.name).changed();
            ui.end_row();
            let woven = fabric.weave.is_some();
            if !woven {
                ui.label("Base colour");
                changed |= ui.color_edit_button_rgb(&mut fabric.base_color).changed();
                ui.end_row();
            }
            ui.label("Roughness");
            changed |= ui
                .add(egui::Slider::new(&mut fabric.roughness, 0.05..=1.0))
                .changed();
            ui.end_row();
            ui.label("Metallic");
            changed |= ui
                .add(egui::Slider::new(&mut fabric.metallic, 0.0..=1.0))
                .changed();
            ui.end_row();
            // A woven fabric's look comes from its draft (below).
            if !woven {
                ui.label("Weave");
                changed |= ui
                    .add(egui::Slider::new(&mut fabric.weave_strength, 0.0..=1.0))
                    .changed();
                ui.end_row();
                ui.label("Wash / variation");
                changed |= ui
                    .add(egui::Slider::new(&mut fabric.wash, 0.0..=1.0))
                    .changed();
                ui.end_row();
                ui.label("Pattern scale");
                changed |= ui
                    .add(egui::Slider::new(&mut fabric.pattern_scale, 0.25..=4.0))
                    .changed();
                ui.end_row();
            }
            ui.label("Surface relief");
            changed |= ui
                .add(egui::Slider::new(&mut fabric.relief, 0.0..=2.0))
                .changed();
            ui.end_row();
            ui.label("Wear");
            look |= ui
                .add(egui::Slider::new(&mut fabric.wear, 0.0..=1.0))
                .on_hover_text(
                    "Fading along seams, hems and the ridges of folds, darker creases, pilling",
                )
                .changed();
            ui.end_row();
        });

        // --- woven from a draft ------------------------------------------------------------
        ui.add_space(8.0);
        let mut woven = fabric.weave.is_some();
        if ui
            .checkbox(&mut woven, "Woven from a draft")
            .on_hover_text("Build the fabric's look from its weave: the interlacing, the yarns and their colours")
            .changed()
        {
            fabric.weave = woven.then(|| {
                // Start plain, in the fabric's own colour.
                let mut weave = cad_core::Weave::library().remove(0).1;
                weave.yarns[0].color = fabric.base_color;
                weave.yarns[1].color = fabric.base_color.map(|c| c * 0.75);
                weave.yarns[1].name = "Shade".into();
                weave
            });
            cx.document.touch();
            return;
        }
        let mut look_changed = look;
        if let Some(weave) = &mut fabric.weave {
            egui::CollapsingHeader::new("Weave")
                .default_open(true)
                .show(ui, |ui| {
                    look_changed |= super::weave_editor::show(ui, weave, &mut self.weave);
                });
        }
        ui.add_space(8.0);
        ui.strong("Physical properties");
        let physics = &mut fabric.physics;
        property_grid(ui, "fabric physics", |ui| {
            ui.label("Density");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut physics.density)
                        .range(10.0..=2000.0)
                        .suffix(" g/m²"),
                )
                .changed();
            ui.end_row();
            ui.label("Thickness");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut physics.thickness)
                        .range(0.05..=20.0)
                        .speed(0.01)
                        .suffix(" mm"),
                )
                .changed();
            ui.end_row();
            for (label, value) in [
                ("Stretch (weft)", &mut physics.stretch_weft),
                ("Stretch (warp)", &mut physics.stretch_warp),
                ("Shear", &mut physics.shear),
                ("Bending", &mut physics.bending),
            ] {
                ui.label(label);
                changed |= ui.add(egui::Slider::new(value, 0.0..=1.0)).changed();
                ui.end_row();
            }
        });

        if changed {
            cx.document.touch();
            cx.output.changes.push(DocumentChange::Fabric(id));
        } else if look_changed {
            // The weave is the look only: the drape is not disturbed.
            cx.document.touch();
        }
    }
}

fn preset(id: cad_core::FabricId, pattern: cad_core::FabricPattern) -> cad_core::Fabric {
    let mut fabric = cad_core::Fabric::new(
        id,
        pattern.label(),
        match pattern {
            cad_core::FabricPattern::Denim => [0.035, 0.10, 0.28],
            cad_core::FabricPattern::Leather => [0.16, 0.045, 0.025],
            cad_core::FabricPattern::Corduroy => [0.33, 0.12, 0.045],
            cad_core::FabricPattern::Wool => [0.28, 0.30, 0.34],
            cad_core::FabricPattern::Knit => [0.55, 0.12, 0.10],
            cad_core::FabricPattern::Satin => [0.40, 0.08, 0.18],
            cad_core::FabricPattern::Plain => [0.5; 3],
        },
    );
    fabric.pattern = pattern;
    fabric.weave_strength = if pattern == cad_core::FabricPattern::Leather {
        0.65
    } else {
        0.5
    };
    fabric.wash = if pattern == cad_core::FabricPattern::Denim {
        0.25
    } else {
        0.0
    };
    fabric.pattern_scale = match pattern {
        cad_core::FabricPattern::Denim => 1.0,
        cad_core::FabricPattern::Corduroy => 1.0,
        _ => 1.0,
    };
    fabric.relief = match pattern {
        cad_core::FabricPattern::Corduroy => 1.0,
        cad_core::FabricPattern::Denim => 0.35,
        _ => 0.0,
    };
    fabric.roughness = match pattern {
        cad_core::FabricPattern::Leather => 0.48,
        cad_core::FabricPattern::Satin => 0.24,
        _ => 0.82,
    };
    fabric.physics = match pattern {
        cad_core::FabricPattern::Leather => cad_core::FabricPhysics {
            density: 850.0,
            stretch_weft: 0.12,
            stretch_warp: 0.16,
            shear: 0.18,
            bending: 0.82,
            thickness: 1.2,
        },
        cad_core::FabricPattern::Denim => cad_core::FabricPhysics {
            density: 420.0,
            stretch_weft: 0.18,
            stretch_warp: 0.28,
            shear: 0.35,
            bending: 0.52,
            thickness: 0.8,
        },
        cad_core::FabricPattern::Corduroy => cad_core::FabricPhysics {
            density: 360.0,
            stretch_weft: 0.25,
            stretch_warp: 0.32,
            shear: 0.35,
            bending: 0.42,
            thickness: 0.9,
        },
        _ => fabric.physics,
    };
    fabric
}

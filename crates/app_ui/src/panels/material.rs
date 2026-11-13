//! Fabric appearance (PBR) and physical properties.

use cad_core::DocumentChange;

use super::{Panel, PanelContext, property_grid};

#[derive(Debug, Default)]
pub struct MaterialPanel;

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
                let [r, g, b] = fabric.base_color.map(|c| (c.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8);
                let (swatch, _) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
                ui.painter().rect_filled(swatch, 3.0, egui::Color32::from_rgb(r, g, b));
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
        let Some(fabric) = cx.document.fabric_mut(id) else { return };
        let mut changed = false;

        ui.strong("Appearance");
        property_grid(ui, "fabric appearance", |ui| {
            ui.label("Name");
            changed |= ui.text_edit_singleline(&mut fabric.name).changed();
            ui.end_row();
            ui.label("Base colour");
            changed |= ui.color_edit_button_rgb(&mut fabric.base_color).changed();
            ui.end_row();
            ui.label("Roughness");
            changed |= ui.add(egui::Slider::new(&mut fabric.roughness, 0.05..=1.0)).changed();
            ui.end_row();
            ui.label("Metallic");
            changed |= ui.add(egui::Slider::new(&mut fabric.metallic, 0.0..=1.0)).changed();
            ui.end_row();
            ui.label("Weave");
            changed |= ui.add(egui::Slider::new(&mut fabric.weave_strength, 0.0..=1.0)).changed();
            ui.end_row();
        });

        ui.add_space(8.0);
        ui.strong("Physical properties");
        let physics = &mut fabric.physics;
        property_grid(ui, "fabric physics", |ui| {
            ui.label("Density");
            changed |= ui.add(egui::DragValue::new(&mut physics.density).range(10.0..=2000.0).suffix(" g/m²")).changed();
            ui.end_row();
            ui.label("Thickness");
            changed |= ui
                .add(egui::DragValue::new(&mut physics.thickness).range(0.05..=20.0).speed(0.01).suffix(" mm"))
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
        }
    }
}

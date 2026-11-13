//! Play/pause transport and frame scrubber for the simulation.

use super::{Panel, PanelContext};
use crate::state::UiCommand;

#[derive(Debug, Default)]
pub struct TimelinePanel;

impl Panel for TimelinePanel {
    fn title(&self) -> &'static str {
        "Timeline"
    }

    fn ui(&mut self, ui: &mut egui::Ui, cx: &mut PanelContext<'_>) {
        let timeline = &mut cx.editor.timeline;
        ui.horizontal(|ui| {
            if ui
                .button("⏮")
                .on_hover_text("Rewind and reset the drape")
                .clicked()
            {
                timeline.stop();
                cx.output.commands.push(UiCommand::ResetDrape);
            }
            let (icon, tip) = if timeline.playing {
                ("⏸", "Pause (Space)")
            } else {
                ("▶", "Play (Space)")
            };
            if ui.button(icon).on_hover_text(tip).clicked() {
                timeline.toggle();
            }
            if ui
                .add_enabled(!timeline.playing, egui::Button::new("⏭"))
                .on_hover_text("Step one frame")
                .clicked()
            {
                cx.output.commands.push(UiCommand::StepSimulation);
            }
            ui.separator();

            ui.label("Frame");
            ui.add(egui::DragValue::new(&mut timeline.frame).range(0..=timeline.end_frame));
            ui.label("/");
            ui.add(egui::DragValue::new(&mut timeline.end_frame).range(1..=100_000));
            ui.separator();
            ui.label("Rate");
            ui.add(
                egui::DragValue::new(&mut timeline.fps)
                    .range(1.0..=240.0)
                    .speed(0.5)
                    .suffix(" fps"),
            );
            ui.checkbox(&mut timeline.looping, "Loop");
            ui.separator();
            ui.label("Sub-steps");
            ui.add(egui::DragValue::new(&mut cx.editor.simulation.settings.substeps).range(1..=64));
            ui.separator();
            ui.weak(format!("{:.2} s", timeline.seconds()));
        });

        ui.spacing_mut().slider_width = (ui.available_width() - 16.0).max(60.0);
        ui.add(egui::Slider::new(&mut timeline.frame, 0..=timeline.end_frame).show_value(false));
    }
}

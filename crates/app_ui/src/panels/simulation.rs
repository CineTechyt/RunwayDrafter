//! Solver controls: backend, sub-stepping, forces and mesh resolution.

use topology_sync::SolverBackend;

use super::{Panel, PanelContext, property_grid};
use crate::state::{SimulationState, UiCommand};

#[derive(Debug)]
pub struct SimulationPanel {
    /// Custom particle distance being typed, applied on request.
    custom_distance: f32,
}

impl Default for SimulationPanel {
    fn default() -> Self {
        Self {
            custom_distance: SimulationState::MEDIUM_MM,
        }
    }
}

impl Panel for SimulationPanel {
    fn title(&self) -> &'static str {
        "Simulation"
    }

    fn ui(&mut self, ui: &mut egui::Ui, cx: &mut PanelContext<'_>) {
        let timeline = &mut cx.editor.timeline;
        let sim = &mut cx.editor.simulation;

        // --- transport --------------------------------------------------------
        ui.horizontal(|ui| {
            let label = if timeline.playing {
                "⏸ Pause"
            } else {
                "▶ Simulate"
            };
            if ui.button(label).on_hover_text("Space").clicked() {
                timeline.toggle();
            }
            if ui
                .add_enabled(!timeline.playing, egui::Button::new("⏭ Step"))
                .clicked()
            {
                cx.output.commands.push(UiCommand::StepSimulation);
            }
            if ui
                .button("⟲ Reset")
                .on_hover_text("Unsew and put every piece back to its arrangement")
                .clicked()
            {
                timeline.stop();
                cx.output.commands.push(UiCommand::ResetDrape);
            }
        });
        ui.add_space(6.0);

        // --- solver -----------------------------------------------------------
        ui.strong("Solver");
        property_grid(ui, "solver", |ui| {
            ui.label("Backend");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut sim.settings.backend, SolverBackend::Global, "Global")
                    .on_hover_text("Solves the whole garment at once, one step per frame: the fabric keeps its length however stiff it is. No cloth-on-cloth collision yet");
                ui.selectable_value(&mut sim.settings.backend, SolverBackend::Contact, "Contact")
                    .on_hover_text("Implicit CPU solver with offset geometric contact: cloth never passes through itself and lies on the body instead of being held to it");
                ui.selectable_value(&mut sim.settings.backend, SolverBackend::Cpu, "CPU");
                ui.add_enabled_ui(sim.gpu_available, |ui| {
                    ui.selectable_value(&mut sim.settings.backend, SolverBackend::Gpu, "GPU")
                        .on_disabled_hover_text("This graphics adapter has no compute shader support");
                });
            });
            ui.end_row();

            ui.label("Sub-steps");
            ui.add(egui::Slider::new(&mut sim.settings.substeps, 1..=64))
                .on_hover_text(
                    "Solver sweeps per simulation step. Too few and the fabric sags like rubber; \
                 finer meshes need more.",
                );
            ui.end_row();

            ui.label("Gravity");
            ui.add(
                egui::DragValue::new(&mut sim.settings.gravity.y)
                    .range(-30.0..=0.0)
                    .speed(0.05)
                    .suffix(" m/s²"),
            );
            ui.end_row();

            ui.label("Damping");
            ui.add(egui::Slider::new(&mut sim.settings.damping, 0.0..=10.0).suffix(" /s"));
            ui.end_row();

            ui.label("Friction");
            ui.add(egui::Slider::new(&mut sim.settings.friction, 0.0..=1.5));
            ui.end_row();

            ui.label("Seam closure");
            ui.add(egui::Slider::new(&mut sim.settings.seam_closure_time, 0.0..=3.0).suffix(" s"))
                .on_hover_text("Time over which the sewing pulls matching edges together. Gravity fades in as the seams shut.");
            ui.end_row();

            ui.label("Collision gap");
            let mut millimetres = sim.settings.collision_margin * 1000.0;
            if ui
                .add(
                    egui::DragValue::new(&mut millimetres)
                        .range(0.5..=20.0)
                        .speed(0.1)
                        .suffix(" mm"),
                )
                .changed()
            {
                sim.settings.collision_margin = millimetres / 1000.0;
            }
            ui.end_row();
        });

        // --- collision -----------------------------------------------------------
        ui.add_space(8.0);
        ui.strong("Collision");
        property_grid(ui, "collision", |ui| {
            ui.label("Cloth on cloth");
            ui.checkbox(&mut sim.settings.self_collision, "").on_hover_text(
                "Folds do not pass through each other, and garments of different layers stay in order.",
            );
            ui.end_row();

            ui.label("Cloth thickness");
            let mut millimetres = sim.settings.cloth_thickness * 1000.0;
            let thickness = egui::DragValue::new(&mut millimetres)
                .range(0.5..=10.0)
                .speed(0.05)
                .suffix(" mm");
            if ui
                .add_enabled(sim.settings.self_collision, thickness)
                .changed()
            {
                sim.settings.cloth_thickness = millimetres / 1000.0;
            }
            ui.end_row();

            ui.label("Continuous (body)");
            ui.checkbox(&mut sim.settings.continuous_body_collision, "").on_hover_text(
                "Checks every particle's whole path against the body mesh, so fast cloth cannot \
                 jump through thin parts such as fingers.",
            );
            ui.end_row();
        });

        // --- resolution ---------------------------------------------------------
        ui.add_space(8.0);
        ui.strong("Particle distance");
        ui.horizontal(|ui| {
            for (label, distance) in [
                ("Draft 20 mm", SimulationState::DRAFT_MM),
                ("10 mm", SimulationState::MEDIUM_MM),
                ("Fine 5 mm", SimulationState::FINE_MM),
            ] {
                if ui
                    .selectable_label(sim.particle_distance == distance, label)
                    .clicked()
                {
                    sim.particle_distance = distance;
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Custom");
            ui.add(
                egui::DragValue::new(&mut self.custom_distance)
                    .range(3.0..=100.0)
                    .speed(0.25)
                    .suffix(" mm"),
            );
            if ui.button("Apply").clicked() {
                sim.particle_distance = self.custom_distance;
            }
        });
        ui.weak("Changing the resolution re-meshes every piece and resets the drape.");

        // --- statistics ---------------------------------------------------------
        ui.add_space(8.0);
        ui.strong("Statistics");
        let stats = &sim.stats;
        property_grid(ui, "solver stats", |ui| {
            let mut row = |label: &str, value: String| {
                ui.label(label);
                ui.label(value);
                ui.end_row();
            };
            row(
                "Running on",
                if stats.backend.is_empty() {
                    "—".to_owned()
                } else {
                    stats.backend.to_owned()
                },
            );
            row("Step time", format!("{:.2} ms", stats.step_ms));
            row("Particles", stats.particles.to_string());
            row("Strain triangles", stats.strain_triangles.to_string());
            row("Bending hinges", stats.bending_hinges.to_string());
            row("Stitches", stats.stitches.to_string());
            row("Parallel batches", stats.batches.to_string());
            row("Cloth mass", format!("{:.0} g", stats.mass * 1000.0));
            row("Cloth contacts", stats.cloth_contacts.to_string());
            row(
                "Pass-throughs caught",
                format!(
                    "{} cloth · {} body",
                    stats.cloth_crossings, stats.body_stops
                ),
            );
            row(
                "Seams",
                if stats.seam_gap <= 0.0 {
                    "closed".to_owned()
                } else {
                    format!("{:.0} % open", stats.seam_gap * 100.0)
                },
            );
        });
    }
}

//! The avatar editor: build sliders and tape measurements.

use avatar::{BodyParams, Measure, Morph};

use crate::state::{EditorState, UiCommand, UiOutput};

/// Shape controls that no tape measurement drives; shown as plain sliders.
const DETAIL_MORPHS: [Morph; 6] = [
    Morph::NeckHeight,
    Morph::FrontChestWidth,
    Morph::UnderbustGirth,
    Morph::NapeToWaist,
    Morph::WaistToHip,
    Morph::KneeGirth,
];

/// Allowed range of a measurement field, in centimetres.
fn range_cm(measure: Measure) -> std::ops::RangeInclusive<f32> {
    match measure {
        Measure::Height => 140.0..=210.0,
        Measure::Inseam => 55.0..=105.0,
        Measure::Bust | Measure::Waist | Measure::Hips => 50.0..=160.0,
        Measure::ArmLength => 35.0..=75.0,
        _ => 8.0..=90.0,
    }
}

pub fn show(ctx: &egui::Context, editor: &mut EditorState, output: &mut UiOutput) {
    let mut open = editor.avatar.window_open;
    // Set again below if a control is being dragged this frame.
    editor.avatar.dragging = false;
    egui::Window::new("Avatar").open(&mut open).default_width(340.0).default_pos([24.0, 64.0]).resizable(true).show(ctx, |ui| {
        if !editor.avatar.available {
            ui.colored_label(ui.visuals().warn_fg_color, "Body model not found (assets/avatar/body.bin).");
            ui.label("Run tools/bake_avatar.py to download it. A placeholder mannequin is shown instead.");
            return;
        }
        egui::ScrollArea::vertical().max_height(620.0).show(ui, |ui| contents(ui, editor, output));
    });
    editor.avatar.window_open = open;
}

fn contents(ui: &mut egui::Ui, editor: &mut EditorState, output: &mut UiOutput) {
    let avatar = &mut editor.avatar;
    let mut dragging = false;
    // Sliders report whether they are mid-drag, so the costly solve can
    // wait for the release.
    let mut track = |response: egui::Response| -> egui::Response {
        dragging |= response.dragged() || response.has_focus();
        response
    };

    // --- presets ---------------------------------------------------------------
    ui.horizontal(|ui| {
        let params = &mut avatar.params;
        if ui.selectable_label(params.gender <= 0.0, "♀ Woman").clicked() {
            *params = BodyParams::female();
        }
        if ui.selectable_label(params.gender >= 1.0, "♂ Man").clicked() {
            *params = BodyParams::male();
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("Reset shape").on_hover_text("Average body of the current gender; clears all measurements").clicked() {
                *params = BodyParams { gender: params.gender, ..BodyParams::female() };
                avatar.targets = [None; Measure::ALL.len()];
            }
        });
    });
    ui.add_space(4.0);

    // --- build -------------------------------------------------------------------
    ui.strong("Build");
    egui::Grid::new("avatar build").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
        let params = &mut avatar.params;
        ui.label("Gender");
        track(ui.add(egui::Slider::new(&mut params.gender, 0.0..=1.0).show_value(false)))
            .on_hover_text("Left: woman. Right: man. In between blends the two skeletons and fat distributions.");
        ui.end_row();
        ui.label("Body fat");
        track(ui.add(egui::Slider::new(&mut params.weight, 0.0..=1.0).show_value(false)))
            .on_hover_text("Slender to heavy. Adjusted automatically when typed girths are otherwise out of reach.");
        ui.end_row();
        ui.label("Muscle");
        track(ui.add(egui::Slider::new(&mut params.muscle, 0.0..=1.0).show_value(false))).on_hover_text("Untrained to muscular");
        ui.end_row();
        ui.label("Frame");
        track(ui.add(egui::Slider::new(&mut params.stature, -1.0..=1.0).show_value(false)))
            .on_hover_text("Short to tall build: changes proportions, not size.");
        ui.end_row();
        ui.label("Cup size");
        track(ui.add_enabled(params.gender < 1.0, egui::Slider::new(&mut params.cup, -1.0..=1.0).show_value(false)));
        ui.end_row();
    });

    // --- measurements ---------------------------------------------------------------
    ui.add_space(8.0);
    ui.strong("Tape measurements");
    ui.weak("Enter your measurements. Every value you set is held (📌) and the body is solved so that all of them are true at once.");
    egui::Grid::new("avatar measurements").num_columns(3).striped(true).spacing([10.0, 5.0]).show(ui, |ui| {
        for measure in Measure::ALL {
            let index = measure as usize;
            let measured = avatar.measurements.get(measure);
            ui.label(measure.label()).on_hover_text(measure.description());

            let mut centimetres = avatar.targets[index].unwrap_or(measured) * 100.0;
            let field = egui::DragValue::new(&mut centimetres).range(range_cm(measure)).speed(0.1).fixed_decimals(1).suffix(" cm");
            if track(ui.add(field)).on_hover_text(measure.description()).changed() {
                avatar.targets[index] = Some(centimetres / 100.0);
            }

            ui.horizontal(|ui| match avatar.targets[index] {
                Some(target) => {
                    if ui.small_button("📌").on_hover_text("Held. Click to release this measurement.").clicked() {
                        avatar.targets[index] = None;
                    }
                    // Say so when the body could not get there.
                    if !avatar.dragging && (measured - target).abs() >= avatar::FitReport::TOLERANCE {
                        ui.colored_label(ui.visuals().warn_fg_color, format!("is {:.1}", measured * 100.0))
                            .on_hover_text("The body model cannot reach this value together with the others. Try the build sliders.");
                    }
                }
                None => {
                    if ui.small_button("○").on_hover_text("Free: follows the body. Click to hold the current value.").clicked() {
                        avatar.targets[index] = Some(measured);
                    }
                }
            });
            ui.end_row();
        }
        ui.label("Weight (estimate)").on_hover_text("From the body's volume at the density of water.");
        ui.label(format!("{:.1} kg", avatar.measurements.weight_kg));
        ui.end_row();
    });
    ui.horizontal(|ui| {
        if ui.button("Hold all").on_hover_text("Hold every measurement at its current value").clicked() {
            for measure in Measure::ALL {
                avatar.targets[measure as usize].get_or_insert(avatar.measurements.get(measure));
            }
        }
        if ui.button("Release all").clicked() {
            avatar.targets = [None; Measure::ALL.len()];
        }
        match &avatar.fit {
            Some(fit) if avatar.targets.iter().any(Option::is_some) => {
                if fit.exact() {
                    ui.weak(format!("matched to {:.1} mm", fit.worst_error * 1000.0));
                } else {
                    ui.colored_label(ui.visuals().warn_fg_color, format!("{} out of reach", fit.unreachable.len()));
                }
            }
            _ => {}
        }
    });

    // --- pose -------------------------------------------------------------------------
    ui.add_space(8.0);
    ui.strong("Pose");
    ui.horizontal_wrapped(|ui| {
        for (name, preset) in avatar::PoseParams::PRESETS {
            if ui.selectable_label(avatar.pose == preset, name).clicked() {
                avatar.pose = preset;
            }
        }
    });
    egui::Grid::new("avatar pose").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
        let (modelled_arm, modelled_leg) = avatar.modelled_angles;
        let pose = &mut avatar.pose;
        let mut angle = |ui: &mut egui::Ui, label: &str, value: &mut f32, range: std::ops::RangeInclusive<f32>, hint: &str| {
            ui.label(label);
            let changed = track(ui.add(egui::Slider::new(value, range).suffix("°").fixed_decimals(0))).on_hover_text(hint).changed();
            ui.end_row();
            changed
        };
        let mut arm = pose.arm_angle.unwrap_or(modelled_arm);
        if angle(ui, "Arms raised", &mut arm, 5.0..=120.0, "Angle between arm and body: 90° is a T-pose.") {
            pose.arm_angle = Some(arm);
        }
        angle(ui, "Arms forward", &mut pose.arm_forward, -30.0..=90.0, "Swings the arms to the front (or back).");
        angle(ui, "Elbows bent", &mut pose.elbow_bend, 0.0..=130.0, "");
        let mut leg = pose.leg_angle.unwrap_or(modelled_leg);
        if angle(ui, "Legs apart", &mut leg, 0.0..=35.0, "Angle between each leg and the vertical.") {
            pose.leg_angle = Some(leg);
        }
        angle(ui, "Knees bent", &mut pose.knee_bend, 0.0..=120.0, "");
    });

    // --- details -----------------------------------------------------------------------
    ui.add_space(8.0);
    egui::CollapsingHeader::new("Shape details").default_open(false).show(ui, |ui| {
        egui::Grid::new("avatar details").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
            for morph in DETAIL_MORPHS {
                ui.label(morph.label());
                let mut value = avatar.params.morph(morph);
                if track(ui.add(egui::Slider::new(&mut value, -1.0..=1.0).show_value(false))).changed() {
                    avatar.params.set_morph(morph, value);
                }
                ui.end_row();
            }
        });
    });
    avatar.dragging = dragging;

    ui.add_space(8.0);
    ui.separator();
    if ui.button("Arrange garment around avatar").on_hover_text("Puts every piece back around the body and resets the drape").clicked() {
        editor.timeline.stop();
        output.commands.push(UiCommand::ArrangePieces);
    }
    ui.weak("A garment that has been simulated stays on when you change the pose or the body and settles again. Before that, it is re-arranged.");
}

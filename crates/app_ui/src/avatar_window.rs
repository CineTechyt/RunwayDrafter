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

/// Linear RGB of an sRGB colour.
fn linear(r: u8, g: u8, b: u8) -> [f32; 3] {
    let rgba = egui::Rgba::from(egui::Color32::from_rgb(r, g, b));
    [rgba.r(), rgba.g(), rgba.b()]
}

const SKIN_TONES: [(&str, [u8; 3]); 8] = [
    ("Porcelain", [242, 216, 198]),
    ("Fair", [232, 196, 170]),
    ("Light", [219, 172, 136]),
    ("Medium", [196, 144, 104]),
    ("Olive", [171, 128, 86]),
    ("Tan", [146, 99, 64]),
    ("Brown", [110, 72, 46]),
    ("Deep", [72, 46, 32]),
];

const EYE_COLOURS: [(&str, [u8; 3]); 6] = [
    ("Dark brown", [58, 34, 20]),
    ("Brown", [110, 68, 34]),
    ("Hazel", [128, 104, 52]),
    ("Green", [74, 118, 76]),
    ("Blue", [70, 116, 168]),
    ("Grey", [126, 136, 146]),
];

const HAIR_COLOURS: [(&str, [u8; 3]); 9] = [
    ("Black", [24, 20, 18]),
    ("Dark brown", [56, 36, 24]),
    ("Brown", [96, 62, 38]),
    ("Auburn", [128, 56, 30]),
    ("Copper", [176, 88, 42]),
    ("Dark blonde", [150, 116, 74]),
    ("Blonde", [212, 178, 120]),
    ("Platinum", [230, 222, 200]),
    ("Grey", [160, 158, 154]),
];

/// A row of colour swatches and a picker for any other colour. Returns
/// whether the colour changed.
fn colour_row(ui: &mut egui::Ui, colour: &mut [f32; 3], swatches: &[(&str, [u8; 3])]) -> bool {
    let mut changed = false;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 3.0;
        for &(name, [r, g, b]) in swatches {
            let value = linear(r, g, b);
            let selected = value
                .iter()
                .zip(colour.iter())
                .all(|(a, b)| (a - b).abs() < 1e-3);
            let (rect, response) =
                ui.allocate_exact_size(egui::vec2(20.0, 20.0), egui::Sense::click());
            let painter = ui.painter();
            painter.rect_filled(rect.shrink(2.0), 3.0, egui::Color32::from_rgb(r, g, b));
            let stroke = if selected {
                egui::Stroke::new(2.0, ui.visuals().selection.stroke.color)
            } else if response.hovered() {
                egui::Stroke::new(1.0, ui.visuals().strong_text_color())
            } else {
                egui::Stroke::new(1.0, ui.visuals().weak_text_color().gamma_multiply(0.5))
            };
            painter.rect_stroke(rect.shrink(1.0), 4.0, stroke, egui::StrokeKind::Inside);
            if response.on_hover_text(name).clicked() {
                *colour = value;
                changed = true;
            }
        }
        ui.add_space(4.0);
        changed |= ui
            .color_edit_button_rgb(colour)
            .on_hover_text("Any other colour")
            .changed();
    });
    changed
}

/// Choosing and playing an animation.
fn animation(ui: &mut egui::Ui, state: &mut crate::state::AnimationState, output: &mut UiOutput) {
    use avatar::Motion;
    ui.horizontal_wrapped(|ui| {
        for motion in Motion::ALL {
            if ui
                .selectable_label(state.motion == Some(motion), motion.label())
                .clicked()
            {
                if state.motion != Some(motion) {
                    state.time = 0.0;
                }
                state.motion = Some(motion);
                state.playing = true;
            }
        }
    });
    let Some(motion) = state.motion else {
        ui.weak("Pick an animation to play it on the avatar.");
        return;
    };
    ui.horizontal(|ui| {
        if ui
            .button(if state.playing {
                "⏸ Pause"
            } else {
                "▶ Play"
            })
            .clicked()
        {
            state.playing = !state.playing;
        }
        if ui
            .button("⏹ Stop")
            .on_hover_text("Back to the edited pose")
            .clicked()
        {
            *state = crate::state::AnimationState {
                garment_follows: state.garment_follows,
                speed: state.speed,
                ..Default::default()
            };
        }
        let duration = motion.duration();
        let mut time = state.time.rem_euclid(duration);
        let scrub = ui.add(
            egui::Slider::new(&mut time, 0.0..=duration)
                .suffix(" s")
                .fixed_decimals(1),
        );
        // Only a hand on the slider scrubs: the slider rounding the time it
        // shows must not stop playback.
        if scrub.changed() && (scrub.dragged() || scrub.clicked() || scrub.has_focus()) {
            state.time = time;
            state.playing = false;
        }
    });
    ui.horizontal(|ui| {
        ui.label("Speed");
        ui.add(
            egui::Slider::new(&mut state.speed, 0.25..=2.0)
                .suffix("×")
                .fixed_decimals(2),
        );
    });
    ui.checkbox(&mut state.garment_follows, "Garment moves with the body");
    if state.garment_follows {
        use crate::state::GarmentStatus;
        match state.garment {
            GarmentStatus::Simulating(done) => {
                ui.add(
                    egui::ProgressBar::new(done)
                        .show_percentage()
                        .text(format!("Simulating the garment… {:.0} %", done * 100.0)),
                );
            }
            GarmentStatus::Simulated => {
                ui.horizontal(|ui| {
                    ui.label("✔ Garment simulated");
                    if ui
                        .small_button("Again")
                        .on_hover_text(
                            "Simulate it once more, after changing the garment or the body",
                        )
                        .clicked()
                    {
                        output.commands.push(UiCommand::SimulateGarmentMotion);
                    }
                });
            }
            GarmentStatus::Carried => {
                ui.horizontal_wrapped(|ui| {
                    ui.weak("The garment is carried with the skin until it has been simulated.");
                    if ui
                        .button("Simulate garment")
                        .on_hover_text("Simulate the cloth through the whole animation, against the moving body, then play it back. Takes a little while.")
                        .clicked()
                    {
                        output.commands.push(UiCommand::SimulateGarmentMotion);
                    }
                });
            }
            GarmentStatus::Unreliable => {
                ui.horizontal_wrapped(|ui| {
                    ui.colored_label(ui.visuals().warn_fg_color, "⚠ The simulation got the cloth caught in this motion; it is carried with the skin instead.");
                    if ui.small_button("Try again").clicked() {
                        output.commands.push(UiCommand::SimulateGarmentMotion);
                    }
                });
            }
            GarmentStatus::NotWorn => {
                ui.weak("Drape the garment first (play the timeline) to simulate it through the animation.");
            }
        }
    }
}

/// Skin, eyes and hair.
fn appearance(ui: &mut egui::Ui, looks: &mut avatar::Appearance) {
    use avatar::HairStyle;

    egui::Grid::new("avatar looks")
        .num_columns(2)
        .spacing([12.0, 6.0])
        .show(ui, |ui| {
            ui.label("Skin");
            colour_row(ui, &mut looks.skin, &SKIN_TONES);
            ui.end_row();
            ui.label("Eyes");
            colour_row(ui, &mut looks.eyes, &EYE_COLOURS);
            ui.end_row();

            let hair = &mut looks.hair;
            ui.label("Hair");
            ui.horizontal_wrapped(|ui| {
                for style in HairStyle::ALL {
                    if ui
                        .selectable_label(hair.style == style, style.label())
                        .clicked()
                        && hair.style != style
                    {
                        hair.style = style;
                        if style.has_length() {
                            hair.length = style.default_length();
                        }
                    }
                }
            });
            ui.end_row();
            if hair.style != HairStyle::Bald {
                ui.label("Hair colour");
                colour_row(ui, &mut hair.color, &HAIR_COLOURS);
                ui.end_row();
                if hair.style.has_length() {
                    ui.label("Length");
                    let mut centimetres = hair.length * 100.0;
                    let range = if hair.style == HairStyle::Bob {
                        16.0..=34.0
                    } else {
                        20.0..=90.0
                    };
                    if ui
                        .add(
                            egui::Slider::new(&mut centimetres, range)
                                .suffix(" cm")
                                .fixed_decimals(0),
                        )
                        .on_hover_text("From the crown down")
                        .changed()
                    {
                        hair.length = centimetres / 100.0;
                    }
                    ui.end_row();
                }
                if hair.style != HairStyle::Buzz {
                    ui.label("Volume");
                    ui.add(egui::Slider::new(&mut hair.volume, 0.0..=1.0).show_value(false))
                        .on_hover_text("Sleek to full");
                    ui.end_row();
                }
                if hair.style.takes_fringe() {
                    ui.label("");
                    ui.checkbox(&mut hair.fringe, "Fringe");
                    ui.end_row();
                }
            }
        });
    ui.weak("Looks are for display: the hair does not touch the garment.");
}

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
        if ui
            .selectable_label(params.gender <= 0.0, "♀ Woman")
            .clicked()
        {
            *params = BodyParams::female();
        }
        if ui.selectable_label(params.gender >= 1.0, "♂ Man").clicked() {
            *params = BodyParams::male();
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .button("Reset shape")
                .on_hover_text("Average body of the current gender; clears all measurements")
                .clicked()
            {
                *params = BodyParams {
                    gender: params.gender,
                    ..BodyParams::female()
                };
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
        if ui
            .button("Hold all")
            .on_hover_text("Hold every measurement at its current value")
            .clicked()
        {
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
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        format!("{} out of reach", fit.unreachable.len()),
                    );
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
    egui::Grid::new("avatar pose")
        .num_columns(2)
        .spacing([12.0, 6.0])
        .show(ui, |ui| {
            let (modelled_arm, modelled_leg) = avatar.modelled_angles;
            let pose = &mut avatar.pose;
            let mut angle = |ui: &mut egui::Ui,
                             label: &str,
                             value: &mut f32,
                             range: std::ops::RangeInclusive<f32>,
                             hint: &str| {
                ui.label(label);
                let changed = track(
                    ui.add(
                        egui::Slider::new(value, range)
                            .suffix("°")
                            .fixed_decimals(0),
                    ),
                )
                .on_hover_text(hint)
                .changed();
                ui.end_row();
                changed
            };
            let mut arm = pose.arm_angle.unwrap_or(modelled_arm);
            if angle(
                ui,
                "Arms raised",
                &mut arm,
                5.0..=120.0,
                "Angle between arm and body: 90° is a T-pose.",
            ) {
                pose.arm_angle = Some(arm);
            }
            angle(
                ui,
                "Arms forward",
                &mut pose.arm_forward,
                -30.0..=90.0,
                "Swings the arms to the front (or back).",
            );
            angle(ui, "Elbows bent", &mut pose.elbow_bend, 0.0..=130.0, "");
            let mut leg = pose.leg_angle.unwrap_or(modelled_leg);
            if angle(
                ui,
                "Legs apart",
                &mut leg,
                0.0..=35.0,
                "Angle between each leg and the vertical.",
            ) {
                pose.leg_angle = Some(leg);
            }
            angle(ui, "Knees bent", &mut pose.knee_bend, 0.0..=120.0, "");
        });

    // --- animation -----------------------------------------------------------------------
    ui.add_space(8.0);
    egui::CollapsingHeader::new("Animation")
        .default_open(true)
        .show(ui, |ui| animation(ui, &mut avatar.animation, output));

    // --- looks ---------------------------------------------------------------------------
    ui.add_space(8.0);
    egui::CollapsingHeader::new("Appearance")
        .default_open(true)
        .show(ui, |ui| appearance(ui, &mut avatar.appearance));

    // --- details -----------------------------------------------------------------------
    ui.add_space(8.0);
    egui::CollapsingHeader::new("Shape details")
        .default_open(false)
        .show(ui, |ui| {
            egui::Grid::new("avatar details")
                .num_columns(2)
                .spacing([12.0, 6.0])
                .show(ui, |ui| {
                    for morph in DETAIL_MORPHS {
                        ui.label(morph.label());
                        let mut value = avatar.params.morph(morph);
                        if track(
                            ui.add(egui::Slider::new(&mut value, -1.0..=1.0).show_value(false)),
                        )
                        .changed()
                        {
                            avatar.params.set_morph(morph, value);
                        }
                        ui.end_row();
                    }
                });
        });
    avatar.dragging = dragging;

    ui.add_space(8.0);
    ui.separator();
    if ui
        .button("Arrange garment around avatar")
        .on_hover_text("Puts every piece back around the body and resets the drape")
        .clicked()
    {
        editor.timeline.stop();
        output.commands.push(UiCommand::ArrangePieces);
    }
    ui.weak("A garment that has been simulated stays on when you change the pose or the body and settles again. Before that, it is re-arranged.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn showing_the_window_does_not_stop_an_animation() {
        let ctx = egui::Context::default();
        let mut editor = EditorState::default();
        editor.avatar.available = true;
        editor.avatar.animation = crate::state::AnimationState {
            motion: Some(avatar::Motion::Walk),
            playing: true,
            time: 1.234_567,
            ..Default::default()
        };
        for frame in 0..5 {
            let mut output = UiOutput::default();
            let mut full = ctx.run_ui(egui::RawInput::default(), |ui| {
                show(ui.ctx(), &mut editor, &mut output)
            });
            full.textures_delta.clear();
            assert!(editor.avatar.animation.playing, "paused on frame {frame}");
            editor.avatar.animation.time += 0.033;
        }
    }
}

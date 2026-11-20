//! The Variables tab: the avatar's measurements, the document's own
//! variables, the helplines and the lines held at a formula's length.

use cad_core::params::{self, Axis, Guide, Param, ParamKind, Variable};
use egui::Color32;

use super::{Panel, PanelContext};
use crate::formula;
use crate::state::UiCommand;

const WARN: Color32 = Color32::from_rgb(230, 120, 60);

#[derive(Default)]
pub struct VariablesPanel {
    /// The variables being edited, and the document's when editing began
    /// (an undo or an opened project replaces the draft).
    draft: Vec<Variable>,
    base: Vec<Variable>,
    guides: Vec<Guide>,
    guides_base: Vec<Guide>,
}

impl Panel for VariablesPanel {
    fn title(&self) -> &'static str {
        "Variables"
    }

    fn ui(&mut self, ui: &mut egui::Ui, cx: &mut PanelContext<'_>) {
        let parameters = cx.document.parameters();
        if parameters.variables != self.base {
            self.base = parameters.variables.clone();
            self.draft = self.base.clone();
        }
        if parameters.guides != self.guides_base {
            self.guides_base = parameters.guides.clone();
            self.guides = self.guides_base.clone();
        }
        let params = cx.editor.parameters.clone();

        ui.weak(
            "Lengths in the pattern can be formulas over the avatar's measurements and your own \
             variables, in mm: type letters in any length box to pick a name. When the avatar \
             changes, everything built on them follows.",
        );
        ui.add_space(6.0);

        egui::CollapsingHeader::new("Body measurements (from the avatar)")
            .default_open(true)
            .show(ui, |ui| measurements(ui, &params));

        egui::CollapsingHeader::new("Your variables")
            .default_open(true)
            .show(ui, |ui| {
                if self.variables(ui, &params) {
                    self.base = self.draft.clone();
                    cx.output
                        .commands
                        .push(UiCommand::SetVariables(self.draft.clone()));
                }
            });

        egui::CollapsingHeader::new("Helplines")
            .default_open(true)
            .show(ui, |ui| {
                if self.guides.is_empty() {
                    ui.weak("None yet. Draw them in the 2D view with the helpline tool (R).");
                }
                let all = self.guides.clone();
                for k in 0..self.guides.len() {
                    let id = self.guides[k].id;
                    let selected = cx.editor.selected_guide == Some(id);
                    ui.push_id(id, |ui| {
                        egui::Frame::group(ui.style()).show(ui, |ui| {
                            ui.horizontal(|ui| {
                                if ui
                                    .selectable_label(selected, "◎")
                                    .on_hover_text("Select it in the 2D view")
                                    .clicked()
                                {
                                    cx.editor.selected_guide = Some(id);
                                }
                                if ui
                                    .small_button("🗑")
                                    .on_hover_text("Delete this helpline")
                                    .clicked()
                                {
                                    cx.output.commands.push(UiCommand::DeleteGuide(id));
                                }
                            });
                            if guide_editor(ui, &mut self.guides[k], &all, &params) {
                                cx.output
                                    .commands
                                    .push(UiCommand::SetGuide(self.guides[k].clone()));
                            }
                        });
                    });
                }
            });

        egui::CollapsingHeader::new("Lines held at a length")
            .default_open(true)
            .show(ui, |ui| {
                let rules = cx.document.parameters().lengths.clone();
                if rules.is_empty() {
                    ui.weak("None yet. Select a line and type a formula into its length box.");
                }
                egui::Grid::new("held lengths")
                    .num_columns(3)
                    .striped(true)
                    .show(ui, |ui| {
                        for rule in rules {
                            let name = cx
                                .document
                                .piece(rule.piece)
                                .map_or("?", |p| p.name.as_str());
                            ui.label(format!("{name} · line {}", rule.edge + 1));
                            let value = params::evaluate(&rule.formula, &params::lookup(&params));
                            ui.horizontal(|ui| {
                                ui.monospace(&rule.formula);
                                match value {
                                    Ok(v) => ui.weak(format!("= {}", params::number(v))),
                                    Err(e) => {
                                        ui.colored_label(WARN, "⚠").on_hover_text(e.to_string())
                                    }
                                };
                            });
                            if ui
                                .small_button("Let go")
                                .on_hover_text("Keep the line as it is, without the formula")
                                .clicked()
                            {
                                cx.output.commands.push(UiCommand::SetLineFormula {
                                    piece: rule.piece,
                                    edge: rule.edge,
                                    formula: None,
                                });
                            }
                            ui.end_row();
                        }
                    });
            });

        if !cx.editor.parameter_errors.is_empty() {
            ui.add_space(6.0);
            for (what, error) in &cx.editor.parameter_errors {
                ui.colored_label(WARN, format!("⚠ {what}: {error}"));
            }
        }
    }
}

impl VariablesPanel {
    /// The variables table. Returns whether a change is to be kept.
    fn variables(&mut self, ui: &mut egui::Ui, params: &[Param]) -> bool {
        let mut commit = false;
        let mut remove = None;
        // A variable may use measurements and the other variables.
        egui::Grid::new("variables")
            .num_columns(4)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.weak("Name");
                ui.weak("Formula (mm)");
                ui.weak("");
                ui.end_row();
                let names: Vec<String> = self.draft.iter().map(|v| v.name.clone()).collect();
                for (k, variable) in self.draft.iter_mut().enumerate() {
                    let taken = names.iter().filter(|n| **n == variable.name).count() > 1
                        || params
                            .iter()
                            .any(|p| p.kind == ParamKind::Measurement && p.name == variable.name);
                    let ok = params::valid_name(&variable.name) && !taken;
                    let name = ui.add_sized(
                        [120.0, 20.0],
                        egui::TextEdit::singleline(&mut variable.name)
                            .font(egui::TextStyle::Monospace)
                            .text_color_opt((!ok).then_some(WARN)),
                    );
                    let name = if ok {
                        name
                    } else {
                        name.on_hover_text(if taken {
                            "This name is already used"
                        } else {
                            "Letters, digits and _, starting with a letter"
                        })
                    };
                    commit |= name.lost_focus();
                    // Its own name is not offered to it.
                    let others: Vec<Param> = params
                        .iter()
                        .filter(|p| p.name != variable.name)
                        .cloned()
                        .collect();
                    ui.horizontal(|ui| {
                        let field = formula::field(
                            ui,
                            egui::Id::new(("variable formula", k)),
                            &mut variable.formula,
                            &others,
                            150.0,
                        );
                        commit |= field.submitted || field.response.lost_focus();
                    });
                    if ui
                        .small_button("🗑")
                        .on_hover_text("Delete this variable")
                        .clicked()
                    {
                        remove = Some(k);
                    }
                    ui.end_row();
                }
            });
        if let Some(k) = remove {
            self.draft.remove(k);
            commit = true;
        }
        if ui
            .button("+ Add variable")
            .on_hover_text("A name for a value or formula, usable in every length: ease, front_width = waist / 4 + ease …")
            .clicked()
        {
            let mut n = self.draft.len() + 1;
            while self.draft.iter().any(|v| v.name == format!("var{n}")) {
                n += 1;
            }
            self.draft.push(Variable {
                name: format!("var{n}"),
                formula: "0".into(),
                note: String::new(),
            });
            commit = true;
        }
        commit && self.draft != self.base
    }
}

/// The avatar's measurements under their formula names.
fn measurements(ui: &mut egui::Ui, params: &[Param]) {
    let measured: Vec<&Param> = params
        .iter()
        .filter(|p| p.kind == ParamKind::Measurement)
        .collect();
    if measured.is_empty() {
        ui.weak("This avatar has no tape measurements (an imported model): formulas using them have no value.");
        return;
    }
    egui::Grid::new("measurement names")
        .num_columns(3)
        .striped(true)
        .spacing([10.0, 3.0])
        .show(ui, |ui| {
            for param in measured {
                ui.monospace(&param.name)
                    .on_hover_text("Its name in formulas");
                if let Ok(v) = param.value {
                    ui.label(format!("{} mm", params::number(v)));
                } else {
                    ui.label("–");
                }
                ui.weak(&param.detail);
                ui.end_row();
            }
        });
}

/// Name, direction, origin and distance of a helpline. Returns whether a
/// change is to be kept.
pub fn guide_editor(
    ui: &mut egui::Ui,
    guide: &mut Guide,
    guides: &[Guide],
    params: &[Param],
) -> bool {
    let mut commit = false;
    ui.horizontal(|ui| {
        let name = ui.add_sized([110.0, 20.0], egui::TextEdit::singleline(&mut guide.name));
        commit |= name.lost_focus();
        ui.weak(match guide.axis {
            Axis::Horizontal => "horizontal",
            Axis::Vertical => "vertical",
        });
    });
    ui.horizontal(|ui| {
        let (forward, backward) = match guide.axis {
            Axis::Horizontal => ("above", "below"),
            Axis::Vertical => ("right of", "left of"),
        };
        if guide.from.is_some() {
            let label = if guide.reversed { backward } else { forward };
            egui::ComboBox::from_id_salt(("guide direction", guide.id))
                .selected_text(label)
                .width(80.0)
                .show_ui(ui, |ui| {
                    commit |= ui
                        .selectable_value(&mut guide.reversed, false, forward)
                        .changed();
                    commit |= ui
                        .selectable_value(&mut guide.reversed, true, backward)
                        .changed();
                });
        } else {
            ui.label("at");
        }
        let from_name = |from: Option<u32>| {
            from.and_then(|f| guides.iter().find(|g| g.id == f))
                .map_or("the sheet's zero".to_owned(), |g| g.name.clone())
        };
        egui::ComboBox::from_id_salt(("guide from", guide.id))
            .selected_text(from_name(guide.from))
            .width(110.0)
            .show_ui(ui, |ui| {
                let mut from = guide.from;
                ui.selectable_value(&mut from, None, "the sheet's zero");
                for other in guides
                    .iter()
                    .filter(|g| g.axis == guide.axis && g.id != guide.id)
                {
                    ui.selectable_value(&mut from, Some(other.id), &other.name);
                }
                if from != guide.from {
                    // Keep it where it is: the distance becomes the gap.
                    let base = from
                        .and_then(|f| guides.iter().find(|g| g.id == f))
                        .map_or(0.0, |g| g.at);
                    let distance = guide.at - base;
                    guide.from = from;
                    guide.reversed = from.is_some() && distance < 0.0;
                    guide.formula = params::number(if from.is_some() {
                        distance.abs()
                    } else {
                        distance
                    });
                    commit = true;
                }
            });
    });
    ui.horizontal(|ui| {
        ui.label(if guide.from.is_some() {
            "by"
        } else {
            "position"
        });
        let field = formula::field(
            ui,
            egui::Id::new(("guide formula", guide.id)),
            &mut guide.formula,
            params,
            170.0,
        );
        commit |= field.submitted || field.response.lost_focus();
        ui.weak("mm");
    });
    commit
}

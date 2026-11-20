//! A text field for formulas over the avatar's measurements and the
//! document's variables.
//!
//! * Typing letters lists the names that match (from the start of the
//!   name first, then anywhere in it or in its description), with their
//!   values: ↑/↓ choose, Enter or Tab puts the name in, a click too.
//! * A plain number lists the measurements and variables closest to it,
//!   also halves and quarters of them, with what is left over
//!   (`waist / 4 + 3`): ↓ and Enter, or a click, replace the number.
//! * The value of a formula is shown next to it, or what is wrong with it.

use cad_core::params::{self, Param, ParamKind};
use egui::text::{CCursor, CCursorRange};
use egui::{Color32, Id, Key, Modifiers, RichText};

/// What happened to the field this frame.
pub struct FormulaResponse {
    pub response: egui::Response,
    /// Enter was pressed with no suggestion chosen: take the formula.
    pub submitted: bool,
    /// The formula's value, if it has one.
    pub value: Option<f32>,
}

#[derive(Clone, Default)]
struct Popup {
    /// Formulas offered, with how they are shown.
    offers: Vec<Offer>,
    /// The one ↑/↓ point at; `None` until an arrow key is pressed for
    /// closest-value offers (Enter then takes the number as typed).
    chosen: Option<usize>,
    /// Name offers replace the word at the cursor; value offers replace
    /// the whole text.
    replaces_word: bool,
}

#[derive(Clone)]
struct Offer {
    insert: String,
    value: f32,
    note: String,
    kind: ParamKind,
}

/// The word (letters, digits, `_`) ending at char index `cursor`, as a
/// char range, if it starts with a letter.
fn word_before(text: &str, cursor: usize) -> Option<(usize, usize, String)> {
    let chars: Vec<char> = text.chars().collect();
    let cursor = cursor.min(chars.len());
    let mut start = cursor;
    while start > 0 && (chars[start - 1].is_alphanumeric() || chars[start - 1] == '_') {
        start -= 1;
    }
    let word: String = chars[start..cursor].iter().collect();
    word.chars()
        .next()
        .is_some_and(|c| c.is_alphabetic() || c == '_')
        .then_some((start, cursor, word))
}

/// Names matching `word`: the ones starting with it first, then the ones
/// with a part (between `_`) starting with it, then the ones containing
/// it, then the ones whose description does.
pub fn matching(params: &[Param], word: &str) -> Vec<usize> {
    let word = word.to_lowercase();
    let mut ranked: Vec<(u8, usize)> = params
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            let name = p.name.to_lowercase();
            let rank = if name.starts_with(&word) {
                0
            } else if name.split('_').any(|part| part.starts_with(&word)) {
                1
            } else if name.contains(&word) {
                2
            } else if p.detail.to_lowercase().contains(&word) {
                3
            } else {
                return None;
            };
            Some((rank, i))
        })
        .collect();
    ranked.sort_by_key(|&(rank, i)| (rank, params[i].name.len(), i));
    ranked.into_iter().map(|(_, i)| i).collect()
}

/// Formulas over the known names that come closest to `value`: each name,
/// half and quarter of it, plus what is left over. Closest first.
pub fn closest(params: &[Param], value: f32) -> Vec<(String, f32)> {
    let mut found: Vec<(f32, String, f32)> = Vec::new();
    for param in params {
        let Ok(v) = param.value else { continue };
        if v <= 0.0 {
            continue;
        }
        for (divisor, suffix) in [(1.0, ""), (2.0, " / 2"), (4.0, " / 4")] {
            let base = v / divisor;
            let rest = value - base;
            // Within a quarter of the value: a starting point, not a guess.
            if rest.abs() > 0.25 * value.abs().max(1.0) {
                continue;
            }
            let rest = (rest * 10.0).round() / 10.0;
            let formula = if rest.abs() < 0.05 {
                format!("{}{suffix}", param.name)
            } else if rest > 0.0 {
                format!("{}{suffix} + {}", param.name, params::number(rest))
            } else {
                format!("{}{suffix} - {}", param.name, params::number(-rest))
            };
            // Whole measurements before fractions of them at equal distance.
            found.push((rest.abs() + (divisor - 1.0) * 0.01, formula, base + rest));
        }
    }
    found.sort_by(|a, b| a.0.total_cmp(&b.0));
    found.into_iter().take(6).map(|(_, f, v)| (f, v)).collect()
}

/// A formula field `width` points wide.
pub fn field(
    ui: &mut egui::Ui,
    id: Id,
    text: &mut String,
    params: &[Param],
    width: f32,
) -> FormulaResponse {
    let popup_id = id.with("formula popup");
    let mut popup: Popup = ui.data(|d| d.get_temp(popup_id)).unwrap_or_default();
    let focused = ui.memory(|m| m.has_focus(id));

    // Keys for the list are taken before the text field sees them.
    let mut take: Option<usize> = None;
    if focused && !popup.offers.is_empty() {
        let n = popup.offers.len();
        ui.input_mut(|i| {
            if i.consume_key(Modifiers::NONE, Key::ArrowDown) {
                popup.chosen = Some(popup.chosen.map_or(0, |c| (c + 1) % n));
            }
            if i.consume_key(Modifiers::NONE, Key::ArrowUp) {
                popup.chosen = Some(popup.chosen.map_or(n - 1, |c| (c + n - 1) % n));
            }
            if let Some(chosen) = popup.chosen
                && (i.consume_key(Modifiers::NONE, Key::Enter)
                    || i.consume_key(Modifiers::NONE, Key::Tab))
            {
                take = Some(chosen);
            }
            if i.consume_key(Modifiers::NONE, Key::Escape) {
                popup.offers.clear();
            }
        });
    }

    let output = egui::TextEdit::singleline(text)
        .id(id)
        .desired_width(width)
        .font(egui::TextStyle::Monospace)
        .show(ui);
    let response = output.response.response.clone();
    let cursor = output
        .cursor_range
        .map(|r| r.primary.index.0)
        .unwrap_or(text.chars().count());

    // A click in the list.
    let mut clicked: Option<usize> = None;
    if focused || response.has_focus() {
        if !popup.offers.is_empty() {
            let area = egui::Area::new(popup_id.with("area"))
                .order(egui::Order::Foreground)
                .fixed_pos(response.rect.left_bottom() + egui::vec2(0.0, 2.0))
                .show(ui.ctx(), |ui| {
                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                        ui.set_min_width(width.max(180.0));
                        if !popup.replaces_word {
                            ui.weak("Use a measurement instead (arrow down, Enter):");
                        }
                        for (k, offer) in popup.offers.iter().enumerate() {
                            let mark = match offer.kind {
                                ParamKind::Measurement => "m",
                                ParamKind::Variable => "ƒ",
                                ParamKind::Guide => "|",
                            };
                            let label = RichText::new(format!(
                                "{mark} {}   = {}",
                                offer.insert,
                                params::number(offer.value)
                            ))
                            .monospace();
                            let row = ui
                                .selectable_label(popup.chosen == Some(k), label)
                                .on_hover_text(&offer.note);
                            if row.clicked() {
                                clicked = Some(k);
                            }
                        }
                    });
                });
            // Clicking the list takes the focus from the field: give it back.
            if area.response.contains_pointer() && ui.input(|i| i.pointer.any_pressed()) {
                ui.memory_mut(|m| m.request_focus(id));
            }
        }
    }

    // Put the chosen offer in.
    if let Some(k) = take.or(clicked)
        && let Some(offer) = popup.offers.get(k).cloned()
    {
        let new_cursor;
        if popup.replaces_word {
            let (start, end) = word_before(text, cursor)
                .map(|(s, e, _)| (s, e))
                .unwrap_or((cursor, cursor));
            let chars: Vec<char> = text.chars().collect();
            let mut replaced: String = chars[..start].iter().collect();
            replaced.push_str(&offer.insert);
            new_cursor = replaced.chars().count();
            replaced.extend(&chars[end.min(chars.len())..]);
            *text = replaced;
        } else {
            *text = offer.insert.clone();
            new_cursor = text.chars().count();
        }
        if let Some(mut state) = egui::TextEdit::load_state(ui.ctx(), id) {
            state
                .cursor
                .set_char_range(Some(CCursorRange::one(CCursor::new(new_cursor))));
            state.store(ui.ctx(), id);
        }
        ui.memory_mut(|m| m.request_focus(id));
        popup.offers.clear();
        popup.chosen = None;
    }

    // What to offer next frame.
    let still_focused = response.has_focus() || ui.memory(|m| m.has_focus(id));
    if !still_focused {
        popup = Popup::default();
    } else if take.is_none() && clicked.is_none() {
        let cursor = cursor.min(text.chars().count());
        let offers = if let Some((_, _, word)) = word_before(text, cursor) {
            let exact = params.iter().any(|p| p.name == word);
            let found: Vec<Offer> = matching(params, &word)
                .into_iter()
                .filter_map(|i| {
                    let p = &params[i];
                    Some(Offer {
                        insert: p.name.clone(),
                        value: *p.value.as_ref().ok()?,
                        note: p.detail.clone(),
                        kind: p.kind,
                    })
                })
                .take(8)
                .collect();
            // A name typed out in full needs no list (unless others start
            // with it too).
            if exact && found.len() == 1 {
                Vec::new()
            } else {
                found
            }
        } else if let Some(value) = params::as_number(text) {
            closest(params, value)
                .into_iter()
                .map(|(formula, value)| Offer {
                    note: format!("{formula} = {}", params::number(value)),
                    insert: formula,
                    value,
                    kind: ParamKind::Measurement,
                })
                .collect()
        } else {
            Vec::new()
        };
        let replaces_word = word_before(text, cursor).is_some();
        if replaces_word != popup.replaces_word
            || offers
                .iter()
                .map(|o| &o.insert)
                .ne(popup.offers.iter().map(|o| &o.insert))
        {
            popup.chosen = replaces_word.then_some(0).filter(|_| !offers.is_empty());
        }
        popup.offers = offers;
        popup.replaces_word = replaces_word;
    }
    ui.data_mut(|d| d.insert_temp(popup_id, popup));

    let submitted =
        response.lost_focus() && take.is_none() && ui.input(|i| i.key_pressed(Key::Enter));
    let value = params::evaluate(text, &params::lookup(params));
    // The value next to a formula (a plain number shows itself).
    if params::as_number(text).is_none() && !text.trim().is_empty() {
        match &value {
            Ok(v) => {
                ui.label(RichText::new(format!("= {}", params::number(*v))).weak());
            }
            Err(error) => {
                ui.label(RichText::new("⚠").color(Color32::from_rgb(230, 120, 60)))
                    .on_hover_text(error.to_string());
            }
        }
    }
    FormulaResponse {
        response,
        submitted,
        value: value.ok(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_core::params::FormulaError;

    fn param(name: &str, value: f32, detail: &str) -> Param {
        Param {
            name: name.into(),
            kind: ParamKind::Measurement,
            value: Ok(value),
            detail: detail.into(),
        }
    }

    #[test]
    fn names_starting_with_the_typed_letters_come_first() {
        let params = vec![
            param("hip_height", 900.0, "Floor to the seat"),
            param("hips", 960.0, "Hip girth"),
            param("waist_to_hip", 200.0, "Waist down to the seat"),
            param("thigh", 560.0, "Girth of one thigh"),
        ];
        let names: Vec<&str> = matching(&params, "hi")
            .iter()
            .map(|&i| params[i].name.as_str())
            .collect();
        assert_eq!(names, ["hips", "hip_height", "waist_to_hip", "thigh"]);
        let seat: Vec<&str> = matching(&params, "seat")
            .iter()
            .map(|&i| params[i].name.as_str())
            .collect();
        assert_eq!(seat, ["hip_height", "waist_to_hip"]);
    }

    #[test]
    fn a_number_is_matched_by_the_closest_measurements_and_their_parts() {
        let params = vec![
            param("waist", 700.0, ""),
            param("inseam", 780.0, ""),
            Param {
                name: "broken".into(),
                kind: ParamKind::Variable,
                value: Err(FormulaError("x".into())),
                detail: String::new(),
            },
        ];
        let offers = closest(&params, 778.0);
        assert_eq!(offers[0].0, "inseam - 2");
        let quarter = closest(&params, 175.0);
        assert_eq!(quarter[0].0, "waist / 4");
        assert!(closest(&params, 20.0).is_empty(), "nothing near 20 mm");
    }

    #[test]
    fn the_word_at_the_cursor_is_found() {
        assert_eq!(
            word_before("waist / 4 + hi", 14),
            Some((12, 14, "hi".into()))
        );
        assert_eq!(word_before("12", 2), None);
        assert_eq!(word_before("a2b", 3), Some((0, 3, "a2b".into())));
    }

    /// Runs the field for one frame with these input events.
    fn frame(
        ctx: &egui::Context,
        params: &[Param],
        text: &mut String,
        events: Vec<egui::Event>,
    ) -> bool {
        let input = egui::RawInput {
            events,
            ..Default::default()
        };
        let mut submitted = false;
        let mut full = ctx.run_ui(input, |ui| {
            submitted = field(ui, Id::new("formula"), text, params, 120.0).submitted;
        });
        full.textures_delta.clear();
        submitted
    }

    fn key(key: Key) -> egui::Event {
        egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: Modifiers::NONE,
        }
    }

    #[test]
    fn typing_letters_offers_names_and_enter_puts_the_first_in() {
        let ctx = egui::Context::default();
        let params = vec![
            param("waist", 700.0, ""),
            param("wrist", 150.0, ""),
            param("inseam", 780.0, ""),
        ];
        let mut text = String::new();
        ctx.memory_mut(|m| m.request_focus(Id::new("formula")));
        frame(&ctx, &params, &mut text, vec![]);
        frame(
            &ctx,
            &params,
            &mut text,
            vec![egui::Event::Text("wa".into())],
        );
        assert_eq!(text, "wa");
        frame(&ctx, &params, &mut text, vec![]);
        let submitted = frame(&ctx, &params, &mut text, vec![key(Key::Enter)]);
        assert_eq!(text, "waist");
        assert!(!submitted, "Enter took the name, it did not submit");
        // Go on typing: the formula is evaluated and Enter now submits it.
        frame(
            &ctx,
            &params,
            &mut text,
            vec![egui::Event::Text(" / 4".into())],
        );
        frame(&ctx, &params, &mut text, vec![]);
        let submitted = frame(&ctx, &params, &mut text, vec![key(Key::Enter)]);
        assert_eq!(text, "waist / 4");
        assert!(submitted);
    }

    #[test]
    fn a_number_is_replaced_by_the_closest_measurement_with_arrow_down_and_enter() {
        let ctx = egui::Context::default();
        let params = vec![param("waist", 700.0, ""), param("inseam", 780.0, "")];
        let mut text = "176".to_owned();
        ctx.memory_mut(|m| m.request_focus(Id::new("formula")));
        frame(&ctx, &params, &mut text, vec![]);
        frame(&ctx, &params, &mut text, vec![]);
        frame(&ctx, &params, &mut text, vec![key(Key::ArrowDown)]);
        let submitted = frame(&ctx, &params, &mut text, vec![key(Key::Enter)]);
        assert_eq!(text, "waist / 4 + 1");
        assert!(!submitted);
    }
}

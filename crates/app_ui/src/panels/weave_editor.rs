//! Editing a woven fabric's draft: the interlacing as a grid you click,
//! the yarns and the order they come in, the thread density and how
//! uneven the yarns are, with a preview of the cloth it makes.

use cad_core::{Weave, Yarn};
use egui::{Color32, Rect, Sense, Stroke, Ui, vec2};

/// Remembered between frames: the preview and how the draft is shown.
#[derive(Default)]
pub struct WeaveEditorState {
    preview: Option<(u64, egui::TextureHandle)>,
    weave_only: bool,
    /// While dragging over the draft: what the cells are painted to.
    painting: Option<bool>,
    warp_text: Option<String>,
    weft_text: Option<String>,
}

impl std::fmt::Debug for WeaveEditorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeaveEditorState").finish_non_exhaustive()
    }
}

/// Pixels per thread of the preview.
const PREVIEW_PIXELS: u32 = 6;

fn color32(c: [f32; 3]) -> Color32 {
    let [r, g, b] = c.map(|v| (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8);
    Color32::from_rgb(r, g, b)
}

/// The editor; returns whether the weave changed.
pub fn show(ui: &mut Ui, weave: &mut Weave, state: &mut WeaveEditorState) -> bool {
    let mut changed = false;

    // --- library ----------------------------------------------------------------
    ui.label("Weave library");
    ui.horizontal_wrapped(|ui| {
        for (name, entry) in Weave::library() {
            let response = ui.add(
                egui::Button::new("")
                    .min_size(vec2(56.0, 54.0))
                    .selected(weave.ends == entry.ends && weave.lift == entry.lift),
            );
            // A small draft on the button, and its name.
            let rect = response.rect.shrink(4.0);
            let icon = Rect::from_min_size(
                rect.min + vec2((rect.width() - 30.0) / 2.0, 0.0),
                vec2(30.0, 30.0),
            );
            draw_draft(ui, icon, &entry, true, 6);
            ui.painter().text(
                rect.center_bottom(),
                egui::Align2::CENTER_BOTTOM,
                name,
                egui::FontId::proportional(9.0),
                ui.visuals().text_color(),
            );
            if response
                .on_hover_text(if Weave::is_preset(name) {
                    "A fabric: its weave and its yarns"
                } else {
                    "Its interlacing; your yarns stay"
                })
                .clicked()
            {
                if Weave::is_preset(name) {
                    *weave = entry;
                } else {
                    weave.take_interlacing(&entry);
                }
                state.warp_text = None;
                state.weft_text = None;
                changed = true;
            }
        }
    });

    // --- the draft --------------------------------------------------------------
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        ui.label("Draft");
        ui.selectable_value(&mut state.weave_only, false, "In colour");
        ui.selectable_value(&mut state.weave_only, true, "Weave only");
    });
    let cell = ((ui.available_width() - 8.0) / weave.ends.max(1) as f32).clamp(4.0, 28.0);
    let size = vec2(cell * weave.ends as f32, cell * weave.picks as f32);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());
    draw_draft(
        ui,
        rect,
        weave,
        !state.weave_only,
        weave.ends.max(weave.picks),
    );
    // Click a crossing to swap which thread is on top; drag to paint.
    if let Some(pointer) = response.interact_pointer_pos() {
        let end = ((pointer.x - rect.left()) / cell).floor();
        let pick_from_top = ((pointer.y - rect.top()) / cell).floor();
        if end >= 0.0 && pick_from_top >= 0.0 {
            let (end, pick) = (
                end as u32,
                weave.picks.saturating_sub(1 + pick_from_top as u32),
            );
            if end < weave.ends && pick < weave.picks {
                let current = weave.up(end, pick);
                let target = *state.painting.get_or_insert(!current);
                if current != target {
                    weave.toggle(end, pick);
                    changed = true;
                }
            }
        }
    }
    if !response.is_pointer_button_down_on() {
        state.painting = None;
    }
    ui.weak("Ends run left to right, pick 1 is at the bottom. Click a crossing to swap which thread is on top, or drag to paint.");

    ui.horizontal_wrapped(|ui| {
        let (mut ends, mut picks) = (weave.ends, weave.picks);
        ui.label("Repeat");
        let a = ui.add(egui::DragValue::new(&mut ends).range(1..=32).speed(0.1));
        ui.label("×");
        let b = ui.add(egui::DragValue::new(&mut picks).range(1..=32).speed(0.1));
        if a.changed() || b.changed() {
            weave.resize(ends, picks);
            changed = true;
        }
        if ui
            .button("Invert")
            .on_hover_text("Warp and weft swap places")
            .clicked()
        {
            weave.lift.iter_mut().for_each(|b| *b = !*b);
            changed = true;
        }
        if ui
            .button("Shift right")
            .on_hover_text("Shift the draft one end right")
            .clicked()
        {
            let old = weave.clone();
            for p in 0..weave.picks {
                for e in 0..weave.ends {
                    let from = (e + weave.ends - 1) % weave.ends;
                    weave.lift[(p * weave.ends + e) as usize] = old.up(from, p);
                }
            }
            changed = true;
        }
        if ui
            .button("Shift up")
            .on_hover_text("Shift the draft one pick up")
            .clicked()
        {
            let old = weave.clone();
            for p in 0..weave.picks {
                for e in 0..weave.ends {
                    let from = (p + weave.picks - 1) % weave.picks;
                    weave.lift[(p * weave.ends + e) as usize] = old.up(e, from);
                }
            }
            changed = true;
        }
        if ui
            .button("Mirror")
            .on_hover_text("Mirror left to right")
            .clicked()
        {
            let old = weave.clone();
            for p in 0..weave.picks {
                for e in 0..weave.ends {
                    weave.lift[(p * weave.ends + e) as usize] = old.up(weave.ends - 1 - e, p);
                }
            }
            changed = true;
        }
    });

    // --- yarns and their order ---------------------------------------------------
    ui.add_space(6.0);
    ui.label("Yarns");
    let mut remove = None;
    for (index, yarn) in weave.yarns.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            changed |= ui.color_edit_button_rgb(&mut yarn.color).changed();
            let name = ui.add(egui::TextEdit::singleline(&mut yarn.name).desired_width(90.0));
            if name.changed() {
                yarn.name.retain(|c| !c.is_whitespace());
                state.warp_text = None;
                state.weft_text = None;
                changed = true;
            }
            if index >= 2
                && ui
                    .small_button("✖")
                    .on_hover_text("Remove this yarn")
                    .clicked()
            {
                remove = Some(index);
            }
        });
    }
    if let Some(index) = remove {
        weave.yarns.remove(index);
        for order in [&mut weave.warp_order, &mut weave.weft_order] {
            order.retain(|&i| usize::from(i) != index);
            order
                .iter_mut()
                .filter(|i| usize::from(**i) > index)
                .for_each(|i| *i -= 1);
            if order.is_empty() {
                order.push(0);
            }
        }
        state.warp_text = None;
        state.weft_text = None;
        changed = true;
    }
    if weave.yarns.len() < 8 && ui.button("➕ Yarn").clicked() {
        weave.yarns.push(Yarn {
            name: format!("Yarn{}", weave.yarns.len() + 1),
            color: [0.5; 3],
        });
        changed = true;
    }

    ui.horizontal(|ui| {
        ui.label("Colourway");
        for (label, warp, weft) in [
            ("Solid", vec![0u8], vec![1u8]),
            ("Stripes", [vec![0u8; 8], vec![1u8; 8]].concat(), vec![1u8]),
            (
                "Check",
                [vec![0u8; 6], vec![1u8; 6]].concat(),
                [vec![0u8; 6], vec![1u8; 6]].concat(),
            ),
            (
                "Pinstripe",
                [vec![0u8; 15], vec![1u8; 1]].concat(),
                vec![0u8],
            ),
        ] {
            if ui.button(label).clicked() {
                if weave.yarns.len() < 2 {
                    weave.yarns.push(Yarn {
                        name: "Cream".into(),
                        color: [0.8, 0.75, 0.6],
                    });
                }
                weave.warp_order = warp;
                weave.weft_order = weft;
                state.warp_text = None;
                state.weft_text = None;
                changed = true;
            }
        }
    });
    for (label, order, text) in [
        (
            "Warp colour order",
            &mut weave.warp_order,
            &mut state.warp_text,
        ),
        (
            "Weft colour order",
            &mut weave.weft_order,
            &mut state.weft_text,
        ),
    ] {
        ui.horizontal(|ui| {
            ui.label(label);
            let shown = text.get_or_insert_with(|| order_text(order, &weave.yarns));
            let response = ui
                .add(egui::TextEdit::singleline(shown).desired_width(150.0))
                .on_hover_text("Counts and yarn names, repeated: \"2 Blue 1 Cream\"");
            if response.changed()
                && let Some(parsed) = parse_order(shown, &weave.yarns)
            {
                *order = parsed;
                changed = true;
            }
            if response.lost_focus() {
                *text = None;
            }
        });
    }

    // --- density and yarn ------------------------------------------------------------
    ui.add_space(6.0);
    super::property_grid(ui, "weave density", |ui| {
        ui.label("Ends per cm");
        changed |= ui
            .add(
                egui::DragValue::new(&mut weave.ends_per_cm)
                    .range(2.0..=80.0)
                    .speed(0.2),
            )
            .changed();
        ui.end_row();
        ui.label("Picks per cm");
        changed |= ui
            .add(
                egui::DragValue::new(&mut weave.picks_per_cm)
                    .range(2.0..=80.0)
                    .speed(0.2),
            )
            .changed();
        ui.end_row();
        ui.label("Irregularity");
        changed |= ui
            .add(egui::Slider::new(&mut weave.irregularity, 0.0..=1.0))
            .on_hover_text("Threads vary in thickness and colour")
            .changed();
        ui.end_row();
        ui.label("Slubs");
        changed |= ui
            .add(egui::Slider::new(&mut weave.slubs, 0.0..=1.0))
            .on_hover_text("Thick stretches along the yarns")
            .changed();
        ui.end_row();
        ui.label("Warp fullness");
        changed |= ui
            .add(egui::Slider::new(&mut weave.warp_cover, 0.5..=1.3))
            .on_hover_text(
                "How wide the warp yarns are for their spacing: above 1 they crowd \
                 and hide the weft (denim)",
            )
            .changed();
        ui.end_row();
        ui.label("Weft fullness");
        changed |= ui
            .add(egui::Slider::new(&mut weave.weft_cover, 0.5..=1.3))
            .on_hover_text("How wide the weft yarns are for their spacing")
            .changed();
        ui.end_row();
        ui.label("Fuzz");
        changed |= ui
            .add(egui::Slider::new(&mut weave.fuzz, 0.0..=1.0))
            .on_hover_text(
                "Loose fibres and light scattered through the yarns: colours bleed \
                 into their neighbours (flannel, tweed)",
            )
            .changed();
        ui.end_row();
    });

    // --- preview ------------------------------------------------------------------------
    ui.add_space(6.0);
    let key = weave.key();
    if state.preview.as_ref().is_none_or(|(k, _)| *k != key) {
        let texture = weave.preview(PREVIEW_PIXELS);
        let image = egui::ColorImage::from_rgba_unmultiplied(
            [texture.width as usize, texture.height as usize],
            bytemuck_free(&texture.albedo),
        );
        let handle = ui
            .ctx()
            .load_texture("weave preview", image, egui::TextureOptions::LINEAR);
        state.preview = Some((key, handle));
    }
    if let Some((_, handle)) = &state.preview {
        let side = ui.available_width().min(220.0);
        let size = handle.size_vec2();
        let shown = vec2(side, side * size.y / size.x.max(1.0));
        ui.add(egui::Image::new((handle.id(), shown)));
        let cm = handle.size()[0] as f32 / PREVIEW_PIXELS as f32 / weave.ends_per_cm.max(0.1);
        ui.weak(format!("{cm:.1} cm of fabric"));
    }
    changed
}

/// RGBA bytes of the pixels.
fn bytemuck_free(pixels: &[[u8; 4]]) -> &[u8] {
    pixels.as_flattened()
}

/// The draft in `rect`: crossings with the warp on top in warp colour (or
/// black), the rest in weft colour (or white), `cells` across at most.
fn draw_draft(ui: &Ui, rect: Rect, weave: &Weave, in_colour: bool, cells: u32) {
    let painter = ui.painter_at(rect);
    let (ends, picks) = (weave.ends.min(cells), weave.picks.min(cells));
    let cell = (rect.width() / ends.max(1) as f32).min(rect.height() / picks.max(1) as f32);
    let yarn = |order: &[u8], thread: u32| {
        let index = order
            .get(thread as usize % order.len().max(1))
            .copied()
            .unwrap_or(0);
        weave
            .yarns
            .get(usize::from(index))
            .map_or([0.5; 3], |y| y.color)
    };
    for p in 0..picks {
        for e in 0..ends {
            let up = weave.up(e, p);
            let color = match (in_colour, up) {
                (true, true) => color32(yarn(&weave.warp_order, e)),
                (true, false) => color32(yarn(&weave.weft_order, p)),
                (false, true) => Color32::from_gray(30),
                (false, false) => Color32::from_gray(235),
            };
            // Pick 1 at the bottom.
            let min = rect.min + vec2(e as f32 * cell, (picks - 1 - p) as f32 * cell);
            let r = Rect::from_min_size(min, vec2(cell, cell));
            painter.rect_filled(r.shrink(0.5), 0.0, color);
        }
    }
    painter.rect_stroke(
        Rect::from_min_size(rect.min, vec2(cell * ends as f32, cell * picks as f32)),
        0.0,
        Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color),
        egui::StrokeKind::Outside,
    );
}

/// "2 Blue 1 Cream" for an order.
fn order_text(order: &[u8], yarns: &[Yarn]) -> String {
    let mut parts = Vec::new();
    let mut run: Option<(u8, usize)> = None;
    for &i in order {
        match &mut run {
            Some((yarn, count)) if *yarn == i => *count += 1,
            _ => {
                if let Some((yarn, count)) = run {
                    parts.push((yarn, count));
                }
                run = Some((i, 1));
            }
        }
    }
    parts.extend(run);
    parts
        .iter()
        .map(|&(yarn, count)| {
            let name = yarns
                .get(usize::from(yarn))
                .map_or("?", |y| y.name.as_str());
            format!("{count} {name}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Reads "2 Blue 1 Cream" (a count may be left out: "Blue Cream").
fn parse_order(text: &str, yarns: &[Yarn]) -> Option<Vec<u8>> {
    let mut order = Vec::new();
    let mut count = None;
    for word in text.split_whitespace() {
        if let Ok(n) = word.parse::<usize>() {
            count = Some(n.min(256));
            continue;
        }
        let index = yarns
            .iter()
            .position(|y| y.name.eq_ignore_ascii_case(word))?;
        order.extend(std::iter::repeat_n(index as u8, count.take().unwrap_or(1)));
    }
    (!order.is_empty() && order.len() <= 512).then_some(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colour_orders_read_and_write_alike() {
        let yarns = Weave::library().remove(0).1.yarns;
        let order = parse_order("2 Blue 1 cream", &yarns).unwrap();
        assert_eq!(order, vec![0, 0, 1]);
        assert_eq!(order_text(&order, &yarns), "2 Blue 1 Cream");
        assert_eq!(parse_order("Blue Cream", &yarns).unwrap(), vec![0, 1]);
        assert!(parse_order("2 Purple", &yarns).is_none());
    }
}

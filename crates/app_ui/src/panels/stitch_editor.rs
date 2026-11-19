//! Setting a stitch the way one sets a sewing machine: which stitch, the
//! thread, its length, how far from the seam, how wide.

use cad_core::{StitchKind, StitchSide, Stitching};
use egui::{Color32, Pos2, Rect, Shape, Stroke, pos2, vec2};

/// The stitch picker alone; returns whether it changed. Changing the kind
/// takes that stitch's usual settings, keeping the thread and side.
pub fn kind_picker(ui: &mut egui::Ui, id: impl std::hash::Hash + std::fmt::Debug, stitching: &mut Stitching) -> bool {
    let mut changed = false;
    egui::ComboBox::from_id_salt(id)
        .selected_text(stitching.kind.label())
        .width(130.0)
        .show_ui(ui, |ui| {
            for kind in StitchKind::ALL {
                let mut shown = Stitching::of(kind);
                shown.thread = stitching.thread;
                let clicked = ui
                    .horizontal(|ui| {
                        let (rect, _) = ui.allocate_exact_size(vec2(26.0, 26.0), egui::Sense::hover());
                        picture(ui.painter(), rect, &shown);
                        ui.selectable_label(stitching.kind == kind, kind.label())
                            .on_hover_text(describe(kind))
                            .clicked()
                    })
                    .inner;
                if clicked && stitching.kind != kind
                {
                    let (thread, side) = (stitching.thread, stitching.side);
                    *stitching = Stitching::of(kind);
                    stitching.thread = thread;
                    stitching.side = side;
                    changed = true;
                }
            }
        });
    changed
}

/// Everything about a stitch; `sides` names the two pieces of a seam (for
/// which one the topstitching lies on), `None` for a free edge.
pub fn show(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + std::fmt::Debug + Copy,
    stitching: &mut Stitching,
    sides: Option<(&str, &str)>,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        changed |= kind_picker(ui, (id, "kind"), stitching);
        if stitching.visible() {
            changed |= ui
                .color_edit_button_rgb(&mut stitching.thread)
                .on_hover_text("Thread colour")
                .changed();
        }
    });
    if !stitching.visible() {
        return changed;
    }
    egui::Grid::new((id, "stitch settings"))
        .num_columns(2)
        .spacing([8.0, 3.0])
        .show(ui, |ui| {
            let value = |ui: &mut egui::Ui, label: &str, v: &mut f32, range: std::ops::RangeInclusive<f32>, hint: &str| {
                ui.label(label);
                let r = ui
                    .add(egui::DragValue::new(v).range(range).speed(0.05).suffix(" mm"))
                    .on_hover_text(hint);
                ui.end_row();
                r.changed()
            };
            changed |= value(ui, "Stitch length", &mut stitching.length, 0.3..=12.0, "Length of one stitch");
            if stitching.kind != StitchKind::Overlock {
                changed |= value(ui, "From seam", &mut stitching.offset, 0.0..=40.0, "Distance of the first row from the seam or edge");
            }
            if matches!(
                stitching.kind,
                StitchKind::DoubleTopstitch
                    | StitchKind::FlatFelled
                    | StitchKind::Zigzag
                    | StitchKind::Overlock
                    | StitchKind::Coverstitch
                    | StitchKind::Bartack
            ) {
                changed |= value(ui, "Width", &mut stitching.width, 0.5..=20.0, "Zigzag width, or the gap between the rows");
            }
            changed |= value(ui, "Thread", &mut stitching.thread_size, 0.1..=1.5, "Thread thickness");
            if let Some((a, b)) = sides {
                ui.label("On");
                ui.horizontal(|ui| {
                    changed |= ui.selectable_value(&mut stitching.side, StitchSide::A, a).changed();
                    changed |= ui.selectable_value(&mut stitching.side, StitchSide::B, b).changed();
                });
                ui.end_row();
            }
        });
    changed
}

fn describe(kind: StitchKind) -> &'static str {
    match kind {
        StitchKind::Plain => "Closed, pressed open: nothing shows outside",
        StitchKind::Topstitch => "One row on the outside, a little off the seam",
        StitchKind::Edgestitch => "One row right at the seam or edge",
        StitchKind::DoubleTopstitch => "Two parallel rows, as on jeans",
        StitchKind::FlatFelled => "Allowances folded into each other: two rows outside, nothing raw inside (jeans inseams, shirts)",
        StitchKind::Zigzag => "Zigzag: stretchy, neatens raw edges",
        StitchKind::Overlock => "Serged edge inside: thread loops over the allowances",
        StitchKind::Coverstitch => "Hems of knits: two rows outside, a net of thread inside",
        StitchKind::Bartack => "Short, dense zigzag that reinforces a point",
        StitchKind::Blindhem => "Hems: hardly anything shows outside",
        StitchKind::FlyJ => "A fly's topstitching: a row beside the zipper that curves into the seam at the bottom",
    }
}

fn to_color(linear: [f32; 3]) -> Color32 {
    let [r, g, b] = linear.map(|c| {
        let c = c.clamp(0.0, 1.0);
        let s = if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
        (s * 255.0 + 0.5) as u8
    });
    Color32::from_rgb(r, g, b)
}

/// What a stitch looks like, drawn into `rect`: two pieces of cloth
/// meeting at a seam (left), the stitching on the right-hand piece, at
/// its real proportions (about 24 mm shown top to bottom).
pub fn picture(painter: &egui::Painter, rect: Rect, stitching: &Stitching) {
    let cloth = Color32::from_rgb(52, 66, 96);
    let lighter = Color32::from_rgb(64, 80, 114);
    let seam_x = rect.left() + 0.3 * rect.width();
    let mm = rect.height() / 24.0;
    painter.rect_filled(Rect::from_min_max(rect.min, pos2(seam_x, rect.bottom())), 2.0, cloth);
    painter.rect_filled(Rect::from_min_max(pos2(seam_x, rect.top()), rect.max), 2.0, lighter);
    painter.line_segment(
        [pos2(seam_x, rect.top()), pos2(seam_x, rect.bottom())],
        Stroke::new(1.5, Color32::from_rgb(25, 30, 44)),
    );
    let thread = to_color(stitching.thread);
    let width = (stitching.thread_size * mm).clamp(1.0, 3.0);
    let stroke = Stroke::new(width, thread);
    let x = |offset: f32| (seam_x + offset * mm).min(rect.right() - 2.0);
    let step = (stitching.length * mm).max(2.0);
    let rows = |painter: &egui::Painter, at: f32, stroke: Stroke| {
        let mut y = rect.top() + 1.0;
        while y + step * 0.8 < rect.bottom() {
            painter.line_segment([pos2(at, y), pos2(at, y + step * 0.78)], stroke);
            y += step;
        }
    };
    let zigzag = |painter: &egui::Painter, centre: f32, half: f32, from: f32, to: f32, stroke: Stroke| {
        let mut points: Vec<Pos2> = Vec::new();
        let mut y = from;
        let mut k = 0;
        while y <= to {
            points.push(pos2(centre + if k % 2 == 0 { -half } else { half }, y));
            y += step;
            k += 1;
        }
        painter.add(Shape::line(points, stroke));
    };
    let s = stitching;
    match s.kind {
        StitchKind::Plain => {}
        StitchKind::Topstitch | StitchKind::Edgestitch => rows(painter, x(s.offset), stroke),
        StitchKind::DoubleTopstitch => {
            rows(painter, x(s.offset), stroke);
            rows(painter, x(s.offset + s.width), stroke);
        }
        StitchKind::FlatFelled => {
            // The folded allowances: a raised, lighter band, then the rows.
            let band = Rect::from_min_max(pos2(seam_x, rect.top()), pos2(x(s.offset + s.width + 1.5), rect.bottom()));
            painter.rect_filled(band, 0.0, Color32::from_rgb(76, 94, 132));
            painter.line_segment(
                [pos2(band.right(), rect.top()), pos2(band.right(), rect.bottom())],
                Stroke::new(1.0, Color32::from_rgb(34, 42, 60)),
            );
            rows(painter, x(s.offset), stroke);
            rows(painter, x(s.offset + s.width), stroke);
        }
        StitchKind::Zigzag => {
            zigzag(painter, x(s.offset), 0.5 * s.width * mm, rect.top() + 2.0, rect.bottom() - 2.0, stroke);
        }
        StitchKind::Bartack => {
            let mid = rect.center().y;
            zigzag(painter, x(s.offset), 0.5 * s.width * mm, mid - 5.0 * mm, mid + 5.0 * mm, Stroke::new(width, thread));
        }
        StitchKind::Overlock => {
            // Seen from inside: loops over the raw edges of the allowances.
            let edge = x(s.width);
            let mut y = rect.top() + 1.0;
            while y + step < rect.bottom() {
                painter.line_segment([pos2(seam_x + 1.0, y), pos2(edge, y + step * 0.5)], stroke);
                painter.line_segment([pos2(edge, y + step * 0.5), pos2(seam_x + 1.0, y + step)], stroke);
                y += step;
            }
            painter.line_segment([pos2(edge, rect.top()), pos2(edge, rect.bottom())], stroke);
            painter.text(
                pos2(rect.right() - 3.0, rect.bottom() - 2.0),
                egui::Align2::RIGHT_BOTTOM,
                "inside",
                egui::FontId::proportional(9.0),
                Color32::from_gray(190),
            );
        }
        StitchKind::Coverstitch => {
            zigzag(
                painter,
                x(s.offset + 0.5 * s.width),
                0.5 * s.width * mm,
                rect.top() + 2.0,
                rect.bottom() - 2.0,
                Stroke::new(width * 0.7, thread.gamma_multiply(0.35)),
            );
            rows(painter, x(s.offset), stroke);
            rows(painter, x(s.offset + s.width), stroke);
        }
        StitchKind::FlyJ => {
            // Down beside the seam, then round into it.
            let at = x(s.offset.min(16.0));
            let bottom = rect.bottom() - 3.0;
            let radius = (at - seam_x).max(2.0);
            let mut points: Vec<Pos2> = Vec::new();
            let straight_end = bottom - radius;
            let mut y = rect.top() + 1.0;
            while y < straight_end {
                points.push(pos2(at, y));
                y += 1.0;
            }
            for k in 0..=12 {
                let a = std::f32::consts::FRAC_PI_2 * k as f32 / 12.0;
                points.push(pos2(seam_x + radius * a.cos(), straight_end + radius * a.sin()));
            }
            // Dashed: every other short piece.
            for (k, w) in points.windows(2).enumerate() {
                if (k / 3) % 2 == 0 {
                    painter.line_segment([w[0], w[1]], stroke);
                }
            }
        }
        StitchKind::Blindhem => {
            let mut y = rect.top() + 2.0;
            while y < rect.bottom() {
                painter.circle_filled(pos2(x(s.offset), y), width * 0.6, thread);
                y += step;
            }
        }
    }
}

/// Every stitch as a tile to click, like a sewing machine's stitch chart.
/// Picking one takes its usual settings, keeping the thread.
pub fn palette(ui: &mut egui::Ui, stitching: &mut Stitching) -> bool {
    let mut changed = false;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
        for kind in StitchKind::ALL {
            let mut shown = Stitching::of(kind);
            shown.thread = stitching.thread;
            let size = vec2(64.0, 78.0);
            let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
            let selected = stitching.kind == kind;
            let frame = if selected {
                ui.visuals().selection.stroke
            } else if response.hovered() {
                Stroke::new(1.0, ui.visuals().widgets.hovered.fg_stroke.color)
            } else {
                Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color)
            };
            let tile = Rect::from_min_size(rect.min + vec2(2.0, 2.0), vec2(size.x - 4.0, size.x - 4.0));
            picture(ui.painter(), tile, &shown);
            ui.painter().rect_stroke(rect.shrink(0.5), 3.0, frame, egui::StrokeKind::Inside);
            ui.painter().text(
                pos2(rect.center().x, rect.bottom() - 3.0),
                egui::Align2::CENTER_BOTTOM,
                kind.label(),
                egui::FontId::proportional(9.5),
                ui.visuals().text_color(),
            );
            if response.on_hover_text(describe(kind)).clicked() && !selected {
                let (thread, side) = (stitching.thread, stitching.side);
                *stitching = Stitching::of(kind);
                stitching.thread = thread;
                stitching.side = side;
                changed = true;
            }
        }
    });
    changed
}

/// A zipper's settings: teeth, colours, how far open, hidden or not.
pub fn zipper_settings(ui: &mut egui::Ui, id: impl std::hash::Hash + std::fmt::Debug + Copy, zipper: &mut cad_core::Zipper) -> bool {
    use cad_core::ZipperTeeth;
    let mut changed = false;
    egui::Grid::new((id, "zipper")).num_columns(2).spacing([8.0, 3.0]).show(ui, |ui| {
        ui.label("Teeth");
        ui.horizontal(|ui| {
            changed |= ui.selectable_value(&mut zipper.teeth, ZipperTeeth::Metal, "Metal").changed();
            changed |= ui.selectable_value(&mut zipper.teeth, ZipperTeeth::Coil, "Coil").changed();
        });
        ui.end_row();
        ui.label("Colours");
        ui.horizontal(|ui| {
            changed |= ui.color_edit_button_rgb(&mut zipper.tape).on_hover_text("Tape").changed();
            changed |= ui.color_edit_button_rgb(&mut zipper.metal).on_hover_text("Teeth and slider").changed();
        });
        ui.end_row();
        ui.label("Open");
        changed |= ui
            .add(egui::Slider::new(&mut zipper.open, 0.0..=1.0).show_value(false))
            .on_hover_text("Pull the slider down: the cloth comes apart above it")
            .drag_stopped();
        ui.end_row();
        ui.label("Concealed");
        changed |= ui
            .checkbox(&mut zipper.concealed, "under a fly")
            .on_hover_text("The teeth lie on the inside, behind a facing")
            .changed();
        ui.end_row();
    });
    changed
}

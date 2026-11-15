//! Text and input boxes drawn over the 2D pattern view: piece names, the
//! length and angle of the line being drawn, typed lengths, and the length
//! and angle of the selected line.

use cad_core::geometry::Curve;
use cad_core::{EdgeSpan, Outline, PatternDocument, PieceId};
use egui::{Align2, Color32, FontId, Pos2, Rect, Stroke, vec2};
use glam::Vec2;

use crate::state::{CanvasTool, EditorState, UiCommand, UiOutput};

/// Draws over the view occupying `rect` and handles its input boxes.
pub fn show(
    ui: &mut egui::Ui,
    rect: Rect,
    document: &PatternDocument,
    editor: &mut EditorState,
    output: &mut UiOutput,
) {
    let ppp = ui.ctx().pixels_per_point();
    let view = editor.canvas_view;
    let to_screen = |w: Vec2| {
        rect.center()
            + vec2(
                (w.x - view.center.x) * view.zoom / ppp,
                -(w.y - view.center.y) * view.zoom / ppp,
            )
    };
    let painter = ui.painter_at(rect);

    if editor.view.show_piece_names {
        piece_names(&painter, document, view.zoom / ppp, &to_screen);
    }

    match editor.canvas_tool {
        CanvasTool::SketchLine | CanvasTool::SketchSpline => {
            sketch(ui, &painter, rect, editor, &to_screen);
        }
        CanvasTool::AddPoint => {
            if let Some(marker) = editor.point_marker {
                let text = if marker.middle {
                    format!("middle · {:.1} | {:.1} mm", marker.from_start, marker.to_end)
                } else {
                    format!("{:.1} | {:.1} mm", marker.from_start, marker.to_end)
                };
                tag(&painter, to_screen(marker.position) + vec2(12.0, -14.0), &text);
            }
        }
        CanvasTool::Select | CanvasTool::Points => {
            selected_line(ui, rect, document, editor, output, &to_screen);
        }
        _ => {}
    }
}

/// Each piece's name in its middle, sized with the zoom like the piece.
fn piece_names(
    painter: &egui::Painter,
    document: &PatternDocument,
    points_per_mm: f32,
    to_screen: &impl Fn(Vec2) -> Pos2,
) {
    let size = (8.0 * points_per_mm).clamp(9.0, 28.0);
    for piece in document.pieces() {
        if piece.name.trim().is_empty() {
            continue;
        }
        let at = to_screen(label_point(&piece.outline));
        let galley = painter.layout_no_wrap(
            piece.name.clone(),
            FontId::proportional(size),
            Color32::from_white_alpha(225),
        );
        let text_rect = Rect::from_center_size(at, galley.size());
        painter.rect_filled(
            text_rect.expand2(vec2(5.0, 2.0)),
            4.0,
            Color32::from_black_alpha(120),
        );
        painter.galley(text_rect.min, galley, Color32::WHITE);
    }
}

/// Where a piece's name goes: the middle of its area, or of its bounds when
/// that falls outside the piece (an L-shape).
fn label_point(outline: &Outline) -> Vec2 {
    let points = outline.flatten(1.0).points;
    let (lo, hi) = outline.bounds();
    let fallback = (lo + hi) * 0.5;
    let mut area = 0.0;
    let mut centre = Vec2::ZERO;
    for i in 0..points.len() {
        let (a, b) = (points[i], points[(i + 1) % points.len()]);
        let cross = a.perp_dot(b);
        area += cross;
        centre += (a + b) * cross;
    }
    if area.abs() < 1e-3 {
        return fallback;
    }
    let centre = centre / (3.0 * area);
    if cad_core::geometry::point_in_polygon(centre, &points) {
        centre
    } else {
        fallback
    }
}

/// A small label with a dark background.
fn tag(painter: &egui::Painter, at: Pos2, text: &str) {
    let galley = painter.layout_no_wrap(
        text.to_owned(),
        FontId::proportional(12.0),
        Color32::from_rgb(170, 235, 255),
    );
    let text_rect = Rect::from_min_size(at, galley.size());
    painter.rect_filled(
        text_rect.expand2(vec2(4.0, 2.0)),
        3.0,
        Color32::from_black_alpha(170),
    );
    painter.galley(text_rect.min, galley, Color32::WHITE);
}

/// Direction of `d` in degrees, counter-clockwise from +x.
fn degrees(d: Vec2) -> f32 {
    d.y.atan2(d.x).to_degrees()
}

/// While sketching: every side's length, the rubber band's length and
/// angle, and a box to type the next side's length (and angle).
fn sketch(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    rect: Rect,
    editor: &mut EditorState,
    to_screen: &impl Fn(Vec2) -> Pos2,
) {
    for pair in editor.sketch_points.windows(2) {
        let d = pair[1] - pair[0];
        painter.text(
            to_screen((pair[0] + pair[1]) * 0.5),
            Align2::CENTER_BOTTOM,
            format!("{:.1}", d.length()),
            FontId::proportional(11.0),
            Color32::from_rgb(120, 220, 255),
        );
    }
    let (Some(&last), Some(preview)) = (editor.sketch_points.last(), editor.sketch_preview)
    else {
        return;
    };
    let d = preview - last;
    tag(
        painter,
        to_screen((last + preview) * 0.5) + vec2(10.0, 6.0),
        &format!("{:.1} mm  {:.1}°", d.length(), degrees(d)),
    );

    // The input box sits by the cursor. Typing a number starts it without
    // clicking into it first.
    let id = egui::Id::new("sketch_length_input");
    let focused = ui.memory(|m| m.has_focus(id));
    if !focused {
        let typed: String = ui.input(|input| {
            input
                .events
                .iter()
                .filter_map(|event| match event {
                    egui::Event::Text(text) => Some(text.clone()),
                    _ => None,
                })
                .collect()
        });
        if !typed.is_empty() && typed.chars().all(|c| c.is_ascii_digit() || ".,-<".contains(c)) {
            editor.sketch_input.push_str(&typed);
            ui.memory_mut(|m| m.request_focus(id));
        }
    }
    let at = to_screen(preview) + vec2(16.0, 14.0);
    if !rect.contains(at) && !focused {
        return;
    }
    let mut enter = false;
    egui::Area::new(egui::Id::new("sketch_length_area"))
        .fixed_pos(at)
        .order(egui::Order::Foreground)
        .show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut editor.sketch_input)
                            .id(id)
                            .desired_width(90.0)
                            .hint_text("length, angle"),
                    );
                    enter = response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    ui.label("mm");
                })
                .response
                .on_hover_text(
                    "Type the next side's length in mm and press Enter; add an angle in \
                     degrees after a comma (\"250, 90\" goes straight up). Without an \
                     angle the side points at the cursor. Enter on an empty box closes \
                     the piece.",
                );
            });
        });
    if enter {
        let input = std::mem::take(&mut editor.sketch_input);
        let parts: Vec<f32> = input
            .split([',', '<', ';', ' '])
            .filter(|p| !p.trim().is_empty())
            .filter_map(|p| p.trim().parse().ok())
            .collect();
        match parts.as_slice() {
            [] if input.trim().is_empty() => editor.sketch_finish = true,
            [length, rest @ ..] if *length > 0.0 => {
                let direction = match rest.first() {
                    Some(angle) => Vec2::from_angle(angle.to_radians()),
                    None => d.normalize_or(Vec2::X),
                };
                editor.sketch_points.push(last + direction * *length);
                ui.memory_mut(|m| m.request_focus(id));
            }
            _ => {}
        }
    }
}

/// Length and angle of the selected line, editable, and a button that cuts
/// it in half.
fn selected_line(
    ui: &mut egui::Ui,
    rect: Rect,
    document: &PatternDocument,
    editor: &mut EditorState,
    output: &mut UiOutput,
    to_screen: &impl Fn(Vec2) -> Pos2,
) {
    let Some(edge) = editor.selected_edge else {
        return;
    };
    let Some(piece) = document.piece(edge.piece) else {
        return;
    };
    let outline = &piece.outline;
    let n = outline.edge_count();
    if edge.edge >= n {
        return;
    }
    let span = outline.line_span(edge.edge);
    let length = outline.span_length(span);
    let start = outline.points()[(span.first % n) as usize].position;
    let end = outline.points()[((span.first + span.count) % n) as usize].position;
    let angle = degrees(end - start);
    let straight = cad_core::fit::is_straight(&outline.span_points(span, 0.25), 0.5);
    let middle = {
        let (e, t) = outline.span_location(span, 0.5);
        outline.curve(e).eval(t)
    };
    let at = to_screen(middle) + vec2(14.0, 10.0);
    if !rect.contains(at) {
        return;
    }

    let length_id = egui::Id::new("line_length_input");
    let angle_id = egui::Id::new("line_angle_input");
    let (length_focused, angle_focused) =
        ui.memory(|m| (m.has_focus(length_id), m.has_focus(angle_id)));
    if !length_focused {
        editor.line_input.0 = format!("{length:.1}");
    }
    if !angle_focused {
        editor.line_input.1 = format!("{angle:.1}");
    }

    let line = (edge.piece, span, edge.edge);
    egui::Area::new(egui::Id::new("line_measure_area"))
        .fixed_pos(at)
        .order(egui::Order::Foreground)
        .show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style())
                .stroke(Stroke::new(1.0, Color32::from_rgb(80, 180, 230)))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        let enter = |ui: &egui::Ui, r: &egui::Response| {
                            r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))
                        };
                        ui.label("↔");
                        let r = ui
                            .add(
                                egui::TextEdit::singleline(&mut editor.line_input.0)
                                    .id(length_id)
                                    .desired_width(56.0),
                            )
                            .on_hover_text("Length along the line in mm: type and press Enter");
                        if enter(ui, &r)
                            && let Ok(value) = editor.line_input.0.trim().parse::<f32>()
                        {
                            push_line_edit(output, line, |piece, edge| UiCommand::SetLineLength {
                                piece,
                                edge,
                                length: value,
                            });
                        }
                        ui.label("mm  ∠");
                        let r = ui
                            .add(
                                egui::TextEdit::singleline(&mut editor.line_input.1)
                                    .id(angle_id)
                                    .desired_width(46.0),
                            )
                            .on_hover_text(
                                "Direction from the line's start to its end, degrees \
                                 counter-clockwise from →; type and press Enter to turn it",
                            );
                        if enter(ui, &r)
                            && let Ok(value) = editor.line_input.1.trim().parse::<f32>()
                        {
                            push_line_edit(output, line, |piece, edge| UiCommand::SetLineAngle {
                                piece,
                                edge,
                                degrees: value,
                            });
                        }
                        ui.label("°");
                        if ui
                            .add_enabled(straight, egui::Button::new("⇋"))
                            .on_hover_text(
                                "Unfold: the piece is half of one cut on the fold along this \
                                 line; mirror it into the whole piece",
                            )
                            .clicked()
                        {
                            push_line_edit(output, line, |piece, edge| UiCommand::UnfoldPiece {
                                piece,
                                edge,
                            });
                        }
                        if ui
                            .button("½")
                            .on_hover_text("Cut the line in half: a turn point in its middle")
                            .clicked()
                        {
                            push_line_edit(output, line, |piece, edge| UiCommand::SplitLine {
                                piece,
                                edge,
                                parts: 2,
                            });
                        }
                    });
                });
        });
}

fn push_line_edit(
    output: &mut UiOutput,
    (piece, _span, edge): (PieceId, EdgeSpan, u32),
    command: impl FnOnce(PieceId, u32) -> UiCommand,
) {
    output.commands.push(command(piece, edge));
}

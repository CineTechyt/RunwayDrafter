//! Toolbar icons, drawn as vector strokes so they stay crisp at any UI
//! scale and need no icon font.

use egui::{Color32, Pos2, Rect, Response, Sense, Shape, Stroke, Ui, Vec2, pos2, vec2};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Icon {
    Select,
    Points,
    AddPoint,
    Curve,
    Sew,
    Stitch,
    FlatFell,
    Zipper,
    SketchLine,
    SketchSpline,
    Button,
    Buttonhole,
    Pull,
    Guide,
    Pen3d,
    EditLines3d,
    Flatten,
    RotateCcw,
    RotateCw,
    FlipHorizontal,
    FlipVertical,
    MirrorCopy,
    MirrorClone,
    Delete,
    FitView,
    Tidy,
    Undo,
    Redo,
}

/// Size of a toolbar button, in points.
pub const BUTTON: f32 = 30.0;

/// An icon button. `active` shows it pressed (the current tool).
pub fn button(ui: &mut Ui, icon: Icon, active: bool, enabled: bool, tooltip: &str) -> Response {
    let (rect, response) = ui.allocate_exact_size(
        Vec2::splat(BUTTON),
        if enabled {
            Sense::click()
        } else {
            Sense::hover()
        },
    );
    let visuals = ui.visuals();
    let hovered = enabled && response.hovered();
    let fill = if active {
        visuals.selection.bg_fill
    } else if hovered {
        visuals.widgets.hovered.weak_bg_fill
    } else {
        Color32::TRANSPARENT
    };
    ui.painter().rect_filled(rect, 5.0, fill);
    let color = if !enabled {
        visuals.weak_text_color().gamma_multiply(0.5)
    } else if active {
        visuals.selection.stroke.color
    } else if hovered {
        visuals.strong_text_color()
    } else {
        visuals.text_color()
    };
    paint(ui.painter(), rect.shrink(6.0), icon, color);
    response.on_hover_text(tooltip)
}

/// A thin vertical divider between groups of buttons.
pub fn separator(ui: &mut Ui) {
    let (rect, _) = ui.allocate_exact_size(vec2(9.0, BUTTON), Sense::hover());
    let x = rect.center().x;
    ui.painter().line_segment(
        [pos2(x, rect.top() + 6.0), pos2(x, rect.bottom() - 6.0)],
        Stroke::new(1.0, ui.visuals().weak_text_color().gamma_multiply(0.5)),
    );
}

/// Draws `icon` into `rect` (a square) with lines of `color`.
pub fn paint(painter: &egui::Painter, rect: Rect, icon: Icon, color: Color32) {
    // Icons are designed on a 16 × 16 grid.
    let s = rect.width() / 16.0;
    let p = |x: f32, y: f32| rect.min + vec2(x * s, y * s);
    let stroke = Stroke::new(1.5 * s.max(0.8), color);
    let thin = Stroke::new(1.0 * s.max(0.7), color);
    let line = |a: Pos2, b: Pos2| painter.line_segment([a, b], stroke);
    let polyline = |points: Vec<Pos2>| painter.add(Shape::line(points, stroke));
    let dashed =
        |a: Pos2, b: Pos2| painter.extend(Shape::dashed_line(&[a, b], thin, 1.6 * s, 1.4 * s));
    // An arc about `c`, from `from` to `to` degrees (0 = right, clockwise
    // on screen), with an arrowhead at its end.
    let arc_arrow = |c: Pos2, r: f32, from: f32, to: f32| {
        let steps = 24;
        let points: Vec<Pos2> = (0..=steps)
            .map(|i| {
                let a = (from + (to - from) * i as f32 / steps as f32).to_radians();
                c + vec2(a.cos(), a.sin()) * r * s
            })
            .collect();
        let end = points[steps];
        let before = points[steps - 2];
        polyline(points);
        let direction = (end - before).normalized();
        let side = vec2(-direction.y, direction.x);
        let back = end - direction * 3.2 * s;
        painter.add(Shape::convex_polygon(
            vec![
                end + direction * 1.2 * s,
                back + side * 2.6 * s,
                back - side * 2.6 * s,
            ],
            color,
            Stroke::NONE,
        ));
    };

    match icon {
        Icon::Select => {
            // A mouse pointer.
            let points = vec![
                p(3.0, 1.5),
                p(3.0, 13.5),
                p(6.2, 10.4),
                p(8.4, 15.0),
                p(10.2, 14.2),
                p(8.0, 9.6),
                p(12.5, 9.6),
            ];
            painter.add(Shape::convex_polygon(
                points.clone(),
                color.gamma_multiply(0.25),
                Stroke::NONE,
            ));
            painter.add(Shape::closed_line(points, stroke));
        }
        Icon::Points => {
            // A curve through two anchors, one with its handle.
            let curve: Vec<Pos2> = (0..=20)
                .map(|i| {
                    let t = i as f32 / 20.0;
                    let u = 1.0 - t;
                    let (a, b, c, d) = (p(2.0, 13.0), p(4.0, 3.0), p(11.0, 3.0), p(14.0, 12.0));
                    pos2(
                        u * u * u * a.x
                            + 3.0 * u * u * t * b.x
                            + 3.0 * u * t * t * c.x
                            + t * t * t * d.x,
                        u * u * u * a.y
                            + 3.0 * u * u * t * b.y
                            + 3.0 * u * t * t * c.y
                            + t * t * t * d.y,
                    )
                })
                .collect();
            polyline(curve);
            painter.line_segment([p(2.0, 13.0), p(5.0, 4.0)], thin);
            painter.circle_filled(p(5.0, 4.0), 1.5 * s, color);
            for anchor in [p(2.0, 13.0), p(14.0, 12.0)] {
                painter.rect_filled(
                    Rect::from_center_size(anchor, Vec2::splat(3.6 * s)),
                    0.0,
                    color,
                );
            }
        }
        Icon::Sew => {
            // A needle with its eye, and a stitched seam.
            line(p(3.5, 12.5), p(13.0, 3.0));
            painter.circle_stroke(p(12.0, 4.0), 1.1 * s, thin);
            for k in 0..3 {
                let x = 1.5 + 4.5 * k as f32;
                line(p(x, 15.0), p(x + 2.5, 15.0));
            }
            painter.add(Shape::line(
                vec![p(3.5, 12.5), p(5.0, 14.2), p(6.5, 13.5)],
                thin,
            ));
        }
        Icon::Stitch => {
            // A zigzag and a row of stitches: a stitch being chosen.
            painter.add(Shape::line(
                vec![
                    p(1.5, 5.0),
                    p(4.0, 2.0),
                    p(6.5, 5.0),
                    p(9.0, 2.0),
                    p(11.5, 5.0),
                    p(14.0, 2.0),
                ],
                thin,
            ));
            for k in 0..4 {
                let x = 1.0 + 3.8 * k as f32;
                line(p(x, 9.5), p(x + 2.4, 9.5));
            }
            for k in 0..3 {
                let x = 2.5 + 4.5 * k as f32;
                painter.add(Shape::line(
                    vec![p(x, 15.0), p(x + 2.0, 12.5), p(x + 4.0, 15.0)],
                    thin,
                ));
            }
        }
        Icon::FlatFell => {
            // Two pieces lapped over each other, two rows of stitches.
            painter.rect_stroke(
                Rect::from_min_max(p(5.5, 1.5), p(10.5, 14.5)),
                0.0,
                thin,
                egui::StrokeKind::Middle,
            );
            line(p(1.5, 1.5), p(1.5, 14.5));
            line(p(14.5, 1.5), p(14.5, 14.5));
            for y in [2.0, 6.0, 10.0] {
                line(p(6.8, y), p(6.8, y + 2.4));
                line(p(9.2, y), p(9.2, y + 2.4));
            }
        }
        Icon::Zipper => {
            // Teeth either side of a line, and the slider with its pull.
            line(p(8.0, 1.0), p(8.0, 15.0));
            for k in 0..5 {
                let y = 6.0 + 2.0 * k as f32;
                let x = if k % 2 == 0 { 5.5 } else { 8.0 };
                line(p(x, y), p(x + 2.5, y));
            }
            painter.rect_filled(Rect::from_min_max(p(5.5, 1.5), p(10.5, 4.5)), 1.0, color);
            line(p(8.0, 4.5), p(11.5, 8.0));
        }
        Icon::AddPoint => {
            // A line with a new point in its middle, and a plus.
            line(p(1.5, 13.0), p(14.5, 13.0));
            painter.rect_filled(
                Rect::from_center_size(p(8.0, 13.0), Vec2::splat(3.8 * s)),
                0.0,
                color,
            );
            line(p(8.0, 2.5), p(8.0, 8.5));
            line(p(5.0, 5.5), p(11.0, 5.5));
        }
        Icon::Curve => {
            // A line pulled into a curve, the pull as an arrow.
            let curve: Vec<Pos2> = (0..=20)
                .map(|i| {
                    let t = i as f32 / 20.0;
                    let (a, c, b) = (p(1.5, 13.5), p(8.0, 1.0), p(14.5, 13.5));
                    let u = 1.0 - t;
                    pos2(
                        u * u * a.x + 2.0 * u * t * c.x + t * t * b.x,
                        u * u * a.y + 2.0 * u * t * c.y + t * t * b.y,
                    )
                })
                .collect();
            polyline(curve);
            dashed(p(1.5, 13.5), p(14.5, 13.5));
            line(p(8.0, 12.5), p(8.0, 8.5));
            painter.add(Shape::convex_polygon(
                vec![p(8.0, 6.6), p(6.2, 9.4), p(9.8, 9.4)],
                color,
                Stroke::NONE,
            ));
        }
        Icon::SketchLine => {
            polyline(vec![p(2.0, 13.0), p(6.0, 3.0), p(14.0, 12.0), p(2.0, 13.0)]);
            for point in [p(2.0, 13.0), p(6.0, 3.0), p(14.0, 12.0)] {
                painter.circle_filled(point, 1.5 * s, color);
            }
        }
        Icon::SketchSpline => {
            let curve: Vec<Pos2> = (0..=20)
                .map(|i| {
                    let t = i as f32 / 20.0;
                    let u = 1.0 - t;
                    pos2(
                        u * u * u * p(2.0, 12.0).x
                            + 3.0 * u * u * t * p(5.0, 3.0).x
                            + 3.0 * u * t * t * p(11.0, 14.0).x
                            + t * t * t * p(14.0, 5.0).x,
                        u * u * u * p(2.0, 12.0).y
                            + 3.0 * u * u * t * p(5.0, 3.0).y
                            + 3.0 * u * t * t * p(11.0, 14.0).y
                            + t * t * t * p(14.0, 5.0).y,
                    )
                })
                .collect();
            polyline(curve);
            painter.circle_filled(p(2.0, 12.0), 1.5 * s, color);
            painter.circle_filled(p(14.0, 5.0), 1.5 * s, color);
        }
        Icon::Button => {
            painter.circle_stroke(p(8.0, 8.0), 5.0 * s, stroke);
            painter.circle_filled(p(6.2, 6.2), 0.9 * s, color);
            painter.circle_filled(p(9.8, 6.2), 0.9 * s, color);
            painter.circle_filled(p(6.2, 9.8), 0.9 * s, color);
            painter.circle_filled(p(9.8, 9.8), 0.9 * s, color);
        }
        Icon::Buttonhole => {
            painter.rect_stroke(
                Rect::from_center_size(p(8.0, 8.0), vec2(11.0 * s, 5.0 * s)),
                2.5 * s,
                stroke,
                egui::StrokeKind::Middle,
            );
            line(p(5.5, 8.0), p(10.5, 8.0));
        }
        Icon::Pen3d => {
            // A pen drawing a curve over a body's round.
            painter.add(Shape::line(
                (0..=12)
                    .map(|i| {
                        let t = i as f32 / 12.0;
                        p(1.5 + 9.0 * t, 14.0 - 5.0 * (t * std::f32::consts::PI).sin())
                    })
                    .collect(),
                thin,
            ));
            line(p(10.5, 9.5), p(14.5, 2.0));
            line(p(10.5, 9.5), p(11.5, 6.0));
            line(p(10.5, 9.5), p(13.2, 7.4));
            line(p(11.5, 6.0), p(13.2, 7.4));
        }
        Icon::EditLines3d => {
            let curve: Vec<Pos2> = (0..=12)
                .map(|i| {
                    let t = i as f32 / 12.0;
                    p(
                        2.0 + 12.0 * t,
                        12.0 - 8.0 * (t * std::f32::consts::PI).sin(),
                    )
                })
                .collect();
            polyline(curve);
            for (x, y) in [(2.0, 12.0), (8.0, 4.0), (14.0, 12.0)] {
                painter.rect_filled(
                    Rect::from_center_size(p(x, y), vec2(3.4 * s, 3.4 * s)),
                    0.5,
                    color,
                );
            }
        }
        Icon::Flatten => {
            // A rounded shape laid out flat.
            painter.add(Shape::line(
                (0..=12)
                    .map(|i| {
                        let t = i as f32 / 12.0;
                        p(2.0 + 12.0 * t, 6.0 - 4.0 * (t * std::f32::consts::PI).sin())
                    })
                    .collect(),
                thin,
            ));
            line(p(8.0, 6.5), p(8.0, 11.0));
            line(p(6.3, 9.3), p(8.0, 11.0));
            line(p(9.7, 9.3), p(8.0, 11.0));
            line(p(2.0, 14.0), p(14.0, 14.0));
        }
        Icon::Guide => {
            // Two helplines with a measured gap between them.
            dashed(p(1.5, 4.0), p(14.5, 4.0));
            dashed(p(1.5, 12.0), p(14.5, 12.0));
            line(p(8.0, 5.0), p(8.0, 11.0));
            line(p(6.5, 6.5), p(8.0, 5.0));
            line(p(9.5, 6.5), p(8.0, 5.0));
            line(p(6.5, 9.5), p(8.0, 11.0));
            line(p(9.5, 9.5), p(8.0, 11.0));
        }
        Icon::Pull => {
            line(p(2.0, 8.0), p(13.0, 8.0));
            painter.add(Shape::convex_polygon(
                vec![p(13.0, 8.0), p(9.0, 5.0), p(9.0, 11.0)],
                color,
                Stroke::NONE,
            ));
            painter.circle_stroke(p(3.0, 8.0), 2.0 * s, thin);
        }
        // Screen angles grow clockwise: an arc with falling angles turns
        // counter-clockwise.
        Icon::RotateCcw => arc_arrow(p(8.0, 8.5), 5.5, 30.0, -240.0),
        Icon::RotateCw => arc_arrow(p(8.0, 8.5), 5.5, 150.0, 420.0),
        Icon::FlipHorizontal => {
            dashed(p(8.0, 1.0), p(8.0, 15.0));
            painter.add(Shape::convex_polygon(
                vec![p(6.5, 3.0), p(6.5, 13.0), p(1.5, 13.0)],
                color,
                Stroke::NONE,
            ));
            painter.add(Shape::closed_line(
                vec![p(9.5, 3.0), p(9.5, 13.0), p(14.5, 13.0)],
                stroke,
            ));
        }
        Icon::FlipVertical => {
            dashed(p(1.0, 8.0), p(15.0, 8.0));
            painter.add(Shape::convex_polygon(
                vec![p(3.0, 6.5), p(13.0, 6.5), p(13.0, 1.5)],
                color,
                Stroke::NONE,
            ));
            painter.add(Shape::closed_line(
                vec![p(3.0, 9.5), p(13.0, 9.5), p(13.0, 14.5)],
                stroke,
            ));
        }
        Icon::MirrorCopy => {
            // A piece and its mirror image, with a plus.
            painter.add(Shape::closed_line(
                vec![p(1.5, 3.0), p(6.5, 3.0), p(6.5, 14.0), p(1.5, 11.0)],
                stroke,
            ));
            painter.extend(Shape::dashed_line(
                &[
                    p(9.5, 3.0),
                    p(14.5, 3.0),
                    p(14.5, 11.0),
                    p(9.5, 14.0),
                    p(9.5, 3.0),
                ],
                thin,
                1.6 * s,
                1.4 * s,
            ));
            line(p(12.0, 6.0), p(12.0, 10.0));
            line(p(10.0, 8.0), p(14.0, 8.0));
        }
        Icon::MirrorClone => {
            // Two mirrored pieces with a link / connector between them.
            painter.add(Shape::closed_line(
                vec![p(1.5, 3.0), p(5.5, 3.0), p(5.5, 14.0), p(1.5, 11.0)],
                stroke,
            ));
            painter.add(Shape::closed_line(
                vec![p(14.5, 3.0), p(10.5, 3.0), p(10.5, 14.0), p(14.5, 11.0)],
                stroke,
            ));
            line(p(5.5, 8.5), p(10.5, 8.5));
            line(p(7.0, 7.0), p(5.5, 8.5));
            line(p(7.0, 10.0), p(5.5, 8.5));
            line(p(9.0, 7.0), p(10.5, 8.5));
            line(p(9.0, 10.0), p(10.5, 8.5));
        }
        Icon::Delete => {
            // A bin.
            line(p(2.5, 4.0), p(13.5, 4.0));
            polyline(vec![p(6.0, 4.0), p(6.0, 2.0), p(10.0, 2.0), p(10.0, 4.0)]);
            painter.add(Shape::closed_line(
                vec![p(4.0, 4.0), p(5.0, 15.0), p(11.0, 15.0), p(12.0, 4.0)],
                stroke,
            ));
            painter.line_segment([p(7.0, 7.0), p(7.0, 12.5)], thin);
            painter.line_segment([p(9.0, 7.0), p(9.0, 12.5)], thin);
        }
        // A hooked arrow, pointing back (undo) or forward (redo).
        Icon::Undo | Icon::Redo => {
            let flip = if icon == Icon::Undo { 1.0 } else { -1.0 };
            let q = |x: f32, y: f32| p(8.0 + flip * (x - 8.0), y);
            let hook: Vec<Pos2> = (0..=16)
                .map(|i| {
                    let a = (-90.0 + 180.0 * i as f32 / 16.0f32).to_radians();
                    q(9.0 + 4.5 * a.cos(), 9.0 + 4.0 * a.sin())
                })
                .collect();
            polyline([vec![q(3.5, 5.0)], hook, vec![q(5.0, 13.0)]].concat());
            painter.add(Shape::convex_polygon(
                vec![q(1.0, 5.0), q(5.5, 1.8), q(5.5, 8.2)],
                color,
                Stroke::NONE,
            ));
        }
        Icon::Tidy => {
            // Pieces in tidy rows.
            for (x0, y0, x1, y1) in [
                (1.5, 1.5, 6.5, 8.0),
                (8.5, 1.5, 14.5, 6.0),
                (1.5, 10.0, 8.0, 14.5),
                (10.0, 8.0, 14.5, 14.5),
            ] {
                painter.rect_stroke(
                    Rect::from_min_max(p(x0, y0), p(x1, y1)),
                    1.0,
                    stroke,
                    egui::StrokeKind::Middle,
                );
            }
        }
        Icon::FitView => {
            // Four corners around a frame.
            for (corner, dx, dy) in [
                ((1.5, 1.5), 1.0, 1.0),
                ((14.5, 1.5), -1.0, 1.0),
                ((1.5, 14.5), 1.0, -1.0),
                ((14.5, 14.5), -1.0, -1.0),
            ] {
                let c = p(corner.0, corner.1);
                polyline(vec![
                    c + vec2(0.0, 4.0 * dy * s),
                    c,
                    c + vec2(4.0 * dx * s, 0.0),
                ]);
            }
            painter.rect_stroke(
                Rect::from_min_max(p(5.0, 5.0), p(11.0, 11.0)),
                1.0,
                thin,
                egui::StrokeKind::Middle,
            );
        }
    }
}

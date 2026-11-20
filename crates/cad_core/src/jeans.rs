//! A pair of straight-leg jeans drafted to a body's measurements: front and
//! back legs, a two-part waistband, a zip fly, a dart in each front and each
//! back, flat-felled inseams and back rise, overlocked outseams, and white
//! topstitching on dark denim.

use glam::Vec2;

use crate::document::PatternDocument;
use crate::ids::{FabricId, PieceId};
use crate::pattern::{BodyPart, ControlPoint, Outline, Segment};
use crate::stitch::{StitchKind, Stitching};

/// The body the jeans are drafted for, millimetres.
#[derive(Debug, Clone, Copy)]
pub struct JeansSize {
    pub waist: f32,
    pub hips: f32,
    pub thigh: f32,
    /// Floor to crotch.
    pub inseam: f32,
    /// Crotch to waist.
    pub rise: f32,
}

impl Default for JeansSize {
    fn default() -> Self {
        Self {
            waist: 900.0,
            hips: 1040.0,
            thigh: 600.0,
            inseam: 860.0,
            rise: 270.0,
        }
    }
}

/// Straight leg: knee and hem girth.
const KNEE: f32 = 470.0;
const HEM: f32 = 430.0;
/// The front darts: narrower and shorter than the back ones.
const FRONT_DART: f32 = 18.0;
const FRONT_DART_LENGTH: f32 = 85.0;
const DART_WIDTH: f32 = 22.0;
const BAND_HEIGHT: f32 = 40.0;

/// A piece's outline as drafted (for the left leg, inside edge at low x):
/// points counter-clockwise, and a curve control per edge where it bends.
struct Draft {
    points: Vec<Vec2>,
    controls: Vec<Option<Vec2>>,
}

impl Draft {
    /// The outline, mirrored left to right for the other side. Mirroring
    /// turns the winding round, so the points are taken backwards; edge
    /// `e` of the draft becomes edge `n − 2 − e`, running the other way.
    fn outline(&self, mirrored: bool) -> Outline {
        let flip = |p: Vec2| if mirrored { Vec2::new(-p.x, p.y) } else { p };
        let n = self.points.len();
        let (points, segments): (Vec<ControlPoint>, Vec<Segment>) = if mirrored {
            (0..n)
                .map(|j| {
                    let p = flip(self.points[n - 1 - j]);
                    let edge = (2 * n - 2 - j) % n;
                    let segment = self.controls[edge]
                        .map_or(Segment::Line, |c| Segment::Quadratic { ctrl: flip(c) });
                    (ControlPoint::corner(p.x, p.y), segment)
                })
                .unzip()
        } else {
            self.points
                .iter()
                .zip(&self.controls)
                .map(|(p, c)| {
                    (
                        ControlPoint::corner(p.x, p.y),
                        c.map_or(Segment::Line, |ctrl| Segment::Quadratic { ctrl }),
                    )
                })
                .unzip()
        };
        Outline::new(points, segments)
    }

    /// Where a run of the draft's edges ended up.
    #[cfg(test)]
    fn span(&self, first: u32, count: u32, mirrored: bool) -> crate::lines::EdgeSpan {
        let n = self.points.len() as u32;
        if mirrored {
            crate::lines::EdgeSpan {
                first: (2 * n - 1 - first - count) % n,
                count,
            }
        } else {
            crate::lines::EdgeSpan { first, count }
        }
    }
}

/// The hem of a leg draft (for the tests): from its point 2 to 3.
#[cfg(test)]
mod leg {
    pub const HEM: (u32, u32) = (2, 1);
}

fn leg_draft(size: &JeansSize, front: bool) -> Draft {
    let (h, w) = (size.hips, size.waist);
    let rise = size.rise;
    let hip_level = (rise - 190.0).max(0.3 * rise);
    let knee = -(0.5 * size.inseam - 50.0);
    let hem = -size.inseam;
    if front {
        let ext = h / 16.0;
        let width = h / 4.0 + 5.0;
        let crease = 0.5 * (ext + width);
        let waist = w / 4.0 + 8.0 + FRONT_DART;
        let (kw, hw) = (0.5 * KNEE - 20.0, 0.5 * HEM - 20.0);
        let cf_top = Vec2::new(ext + 10.0, rise);
        Draft {
            points: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(crease - 0.5 * kw, knee),
                Vec2::new(crease - 0.5 * hw, hem),
                Vec2::new(crease + 0.5 * hw, hem),
                Vec2::new(crease + 0.5 * kw, knee),
                Vec2::new(ext + width, hip_level),
                Vec2::new(cf_top.x + waist, rise - 6.0),
                cf_top,
                Vec2::new(ext, 0.35 * rise),
            ],
            controls: vec![
                Some(Vec2::new(crease - 0.5 * kw - 12.0, 0.45 * knee)),
                None,
                None,
                None,
                Some(Vec2::new(crease + 0.5 * kw + 22.0, 0.35 * knee)),
                Some(Vec2::new(ext + width + 4.0, 0.6 * rise + 0.4 * hip_level)),
                None,
                None,
                Some(Vec2::new(ext, 0.0)),
            ],
        }
    } else {
        let ext = h / 9.0;
        let width = h / 4.0 + 15.0;
        let crease = 0.5 * (ext + width) + 10.0;
        let waist = w / 4.0 + 12.0 + DART_WIDTH;
        let (kw, hw) = (0.5 * KNEE + 20.0, 0.5 * HEM + 20.0);
        let cb_top = Vec2::new(ext + 30.0, rise + 25.0);
        Draft {
            points: vec![
                Vec2::new(0.0, -10.0),
                Vec2::new(crease - 0.5 * kw, knee),
                Vec2::new(crease - 0.5 * hw, hem),
                Vec2::new(crease + 0.5 * hw, hem),
                Vec2::new(crease + 0.5 * kw, knee),
                Vec2::new(ext + width, hip_level),
                Vec2::new(cb_top.x + waist - 8.0, rise - 6.0),
                cb_top,
                Vec2::new(ext + 8.0, 0.4 * rise),
            ],
            controls: vec![
                Some(Vec2::new(crease - 0.5 * kw - 18.0, 0.4 * knee)),
                None,
                None,
                None,
                Some(Vec2::new(crease + 0.5 * kw + 26.0, 0.35 * knee)),
                Some(Vec2::new(ext + width + 6.0, 0.6 * rise + 0.4 * hip_level)),
                None,
                None,
                Some(Vec2::new(0.55 * ext, -8.0)),
            ],
        }
    }
}

/// Straight-leg jeans for `size`, in raw denim with white topstitching.
pub fn jeans(size: JeansSize) -> PatternDocument {
    let mut doc = PatternDocument::new();
    let fabric = doc.add_fabric("Raw denim", [0.016, 0.032, 0.115]);
    if let Some(preset) = crate::fabric_library::presets()
        .into_iter()
        .find(|f| f.name == "Raw denim")
        && let Some(f) = doc.fabric_mut(fabric)
    {
        *f = crate::material::Fabric {
            id: fabric,
            ..preset
        };
    }
    let thread = [0.78, 0.76, 0.70];
    let stitch = |kind: StitchKind| Stitching::of(kind).with_thread(thread);

    struct Placed {
        id: PieceId,
        mirrored: bool,
        draft: Draft,
    }
    let add = |doc: &mut PatternDocument,
               name: &str,
               draft: Draft,
               mirrored: bool,
               part: BodyPart,
               angle: f32| {
        let id = doc.add_piece(name, draft.outline(mirrored), fabric);
        if let Some(p) = doc.piece_mut(id) {
            p.placement.part = part;
            p.placement.angle = angle;
        }
        Placed {
            id,
            mirrored,
            draft,
        }
    };
    // The avatar's left leg is at +x: seen from the front its inside edge
    // is towards −x, as drafted; from behind the pattern is turned round.
    let front_left = add(
        &mut doc,
        "Front left",
        leg_draft(&size, true),
        false,
        BodyPart::LeftLeg,
        0.0,
    );
    let front_right = add(
        &mut doc,
        "Front right",
        leg_draft(&size, true),
        true,
        BodyPart::RightLeg,
        0.0,
    );
    let back_left = add(
        &mut doc,
        "Back left",
        leg_draft(&size, false),
        true,
        BodyPart::LeftLeg,
        180.0,
    );
    let back_right = add(
        &mut doc,
        "Back right",
        leg_draft(&size, false),
        false,
        BodyPart::RightLeg,
        180.0,
    );

    // The shaping first: a dart in each front over the crease, and in
    // each back. They cut points into the waist lines; the seams are then
    // sewn between points, which stay where they are.
    let flip = |placed: &Placed, p: Vec2| {
        if placed.mirrored {
            Vec2::new(-p.x, p.y)
        } else {
            p
        }
    };
    let corner = |placed: &Placed, k: usize| flip(placed, placed.draft.points[k]);
    let mut waist_parts = Vec::new();
    for placed in [&front_left, &front_right, &back_left, &back_right] {
        let front = placed.id == front_left.id || placed.id == front_right.id;
        let (outer, centre) = (corner(placed, 6), corner(placed, 7));
        let edge = doc
            .span_between(placed.id, outer, centre)
            .map_or(0, |(span, _)| span.first);
        // Along the edge as it now runs (mirrored pieces run centre to side).
        let along = |t: f32| if placed.mirrored { 1.0 - t } else { t };
        let points = if front {
            let d = &placed.draft;
            let crease = 0.5 * (d.points[3].x + d.points[2].x);
            let t = ((d.points[6].x - crease) / (d.points[6].x - d.points[7].x)).clamp(0.2, 0.8);
            // A dart: a triangle of cloth sewn away, its point towards the
            // crease, tapering to nothing.
            doc.add_dart_points(placed.id, edge, along(t), FRONT_DART, FRONT_DART_LENGTH)
                .map(|(_, p)| [p[0], p[1], p[2]])
        } else {
            doc.add_dart_points(placed.id, edge, 0.5, DART_WIDTH, 110.0)
                .map(|(_, p)| [p[0], p[1], p[2]])
        };
        let Some([first, _, last]) = points else {
            continue;
        };
        // The cloth either side of the dart, side to centre.
        let (near_side, near_centre) = if first.distance(outer) < last.distance(outer) {
            (first, last)
        } else {
            (last, first)
        };
        waist_parts.push([(outer, near_side), (near_centre, centre)]);
    }
    let [front_l, front_r, back_l, back_r] = [0, 1, 2, 3].map(|k| waist_parts.get(k).copied());
    let (Some(front_l), Some(_), Some(back_l), Some(_)) = (front_l, front_r, back_l, back_r) else {
        return doc;
    };

    // Waistband halves: from centre front along the bottom, the front waist
    // either side of its dart, then the back either side of its own.
    let lengths = [
        front_l[1].0.distance(front_l[1].1),
        front_l[0].0.distance(front_l[0].1),
        back_l[0].0.distance(back_l[0].1),
        back_l[1].0.distance(back_l[1].1),
    ];
    let mut x = 0.0;
    let mut bottom = vec![Vec2::ZERO];
    for l in lengths {
        x += l;
        bottom.push(Vec2::new(x, 0.0));
    }
    let band_draft = || {
        let mut points = bottom.clone();
        points.push(Vec2::new(x, BAND_HEIGHT));
        points.push(Vec2::new(0.0, BAND_HEIGHT));
        Draft {
            controls: vec![None; points.len()],
            points,
        }
    };
    let band_left = add(
        &mut doc,
        "Waistband left",
        band_draft(),
        false,
        BodyPart::Waist,
        90.0,
    );
    let band_right = add(
        &mut doc,
        "Waistband right",
        band_draft(),
        true,
        BodyPart::Waist,
        -90.0,
    );

    let felled = stitch(StitchKind::FlatFelled);
    let overlock = Stitching::of(StitchKind::Overlock).with_thread([0.05, 0.07, 0.15]);
    let line =
        |placed: &Placed, a: usize, b: usize| (placed.id, corner(placed, a), corner(placed, b));
    for (front, back) in [(&front_left, &back_left), (&front_right, &back_right)] {
        // Crotch to hem, and hem to waist.
        doc.sew_between(line(front, 0, 2), line(back, 0, 2), felled);
        doc.sew_between(line(front, 3, 6), line(back, 3, 6), overlock);
    }
    // The fly: a metal zipper down the straight centre front, hidden under
    // the left front with its J of topstitching; below it the crotch is
    // sewn with a double row. The back rise is flat-felled.
    if let Some(fly) = doc.sew_between(
        line(&front_left, 7, 8),
        line(&front_right, 7, 8),
        stitch(StitchKind::FlyJ),
    ) {
        doc.set_seam_zipper(
            fly,
            Some(crate::stitch::Zipper {
                teeth: crate::stitch::ZipperTeeth::Metal,
                tape: [0.012, 0.018, 0.05],
                metal: [0.50, 0.36, 0.16],
                open: 0.0,
                concealed: true,
            }),
        );
    }
    doc.sew_between(
        line(&front_left, 8, 0),
        line(&front_right, 8, 0),
        stitch(StitchKind::DoubleTopstitch),
    );
    // No fly shield: hidden inside real jeans, and in the simulation it
    // was squeezed between the tight fronts and the body, came through the
    // cloth on one side or the other and tore the fly open.
    doc.sew_between(line(&back_left, 7, 0), line(&back_right, 7, 0), felled);
    // The waistband onto the waist, edgestitched on the band.
    let on_band = stitch(StitchKind::Edgestitch);
    let band_points = |placed: &Placed| -> Vec<Vec2> {
        (0..placed.draft.points.len())
            .map(|k| corner(placed, k))
            .collect()
    };
    for (band, front, back, front_parts, back_parts) in [
        (
            &band_left,
            &front_left,
            &back_left,
            Some(waist_parts[0]),
            Some(waist_parts[2]),
        ),
        (
            &band_right,
            &front_right,
            &back_right,
            waist_parts.get(1).copied(),
            waist_parts.get(3).copied(),
        ),
    ] {
        let (Some(fp), Some(bp)) = (front_parts, back_parts) else {
            continue;
        };
        let b = band_points(band);
        // Band, centre front onwards; waist pieces from the centre out.
        doc.sew_between((band.id, b[0], b[1]), (front.id, fp[1].1, fp[1].0), on_band);
        doc.sew_between((band.id, b[1], b[2]), (front.id, fp[0].1, fp[0].0), on_band);
        doc.sew_between((band.id, b[2], b[3]), (back.id, bp[0].0, bp[0].1), on_band);
        doc.sew_between((band.id, b[3], b[4]), (back.id, bp[1].0, bp[1].1), on_band);
    }
    let (bl, br) = (band_points(&band_left), band_points(&band_right));
    doc.sew_between(
        (band_left.id, bl[4], bl[5]),
        (band_right.id, br[4], br[5]),
        Stitching::default(),
    );
    doc.sew_between(
        (band_left.id, bl[6], bl[0]),
        (band_right.id, br[6], br[0]),
        Stitching::default(),
    );

    // The waistband's riveted tack button, just beside centre front (the
    // band's halves are sewn shut there: a fastening pulling across that
    // seam would only crumple it).
    {
        let left = band_points(&band_left);
        if let Some(p) = doc.piece_mut(band_left.id) {
            let mut button =
                crate::pattern::Button::new(Vec2::new(left[0].x + 10.0, 0.5 * BAND_HEIGHT), 17.0);
            button.style = crate::pattern::ButtonStyle::Tack;
            button.finish = crate::pattern::ButtonFinish::Metal;
            button.color = [0.50, 0.36, 0.16];
            p.buttons.push(button);
        }
    }

    // Hems turned up and stitched; the band's top edgestitched.
    let mut hem = stitch(StitchKind::Topstitch);
    hem.offset = 12.0;
    for placed in [&front_left, &front_right, &back_left, &back_right] {
        if let Some((span, _)) = doc.span_between(placed.id, corner(placed, 2), corner(placed, 3)) {
            doc.set_edge_stitching(placed.id, span, Some(hem));
        }
    }
    for (placed, b) in [(&band_left, &bl), (&band_right, &br)] {
        if let Some((span, _)) = doc.span_between(placed.id, b[5], b[6]) {
            doc.set_edge_stitching(placed.id, span, Some(stitch(StitchKind::Edgestitch)));
        }
    }
    let _ = FabricId(0);
    doc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirrored_drafts_keep_their_edges_and_winding() {
        let draft = leg_draft(&JeansSize::default(), true);
        for mirrored in [false, true] {
            let outline = draft.outline(mirrored);
            assert!(outline.is_ccw(), "mirrored: {mirrored}");
            // The hem stays the hem: its two ends are the lowest points.
            let span = draft.span(leg::HEM.0, 1, mirrored);
            let p = outline.points();
            let n = p.len();
            let (a, b) = (
                p[span.first as usize].position,
                p[(span.first as usize + 1) % n].position,
            );
            assert!((a.y - b.y).abs() < 1e-3 && a.y < -800.0, "{a} {b}");
        }
    }

    #[test]
    fn tidying_the_layout_keeps_buttons_and_folds_on_their_pieces() {
        let mut doc = jeans(JeansSize::default());
        doc.tidy_layout(40.0);
        let (mut buttons, mut folds) = (0, 0);
        for p in doc.pieces() {
            for b in &p.buttons {
                assert!(
                    p.contains(b.position),
                    "{}: button at {}",
                    p.name,
                    b.position
                );
                buttons += 1;
            }
            for f in &p.folds {
                assert!(p.contains(f.a.lerp(f.b, 0.5)), "{}: fold", p.name);
                folds += 1;
            }
        }
        assert_eq!((buttons, folds), (1, 0));
    }

    #[test]
    #[ignore]
    fn print_seams() {
        let doc = jeans(JeansSize::default());
        for seam in doc.seams() {
            let name = |p| doc.piece(p).unwrap().name.clone();
            let length = |piece, span| doc.piece(piece).unwrap().outline.span_length(span);
            println!(
                "{:>16} {:?} {:6.1}  |  {:>16} {:?} {:6.1}  rev {}",
                name(seam.a.piece),
                seam.a_span(),
                length(seam.a.piece, seam.a_span()),
                name(seam.b.piece),
                seam.b_span(),
                length(seam.b.piece, seam.b_span()),
                seam.reversed
            );
        }
    }

    #[test]
    fn the_jeans_have_their_pieces_seams_and_shaping() {
        let doc = jeans(JeansSize::default());
        assert_eq!(doc.pieces().len(), 6);
        // 2 inseams, 2 outseams, 2 rises, 8 band seams, 2 band ends, plus a
        // dart per front and per back.
        assert_eq!(doc.seams().len(), 21);
        let folds: usize = doc.pieces().iter().map(|p| p.folds.len()).sum();
        assert_eq!(folds, 0);
        for seam in doc.seams() {
            let length = |piece, span| doc.piece(piece).unwrap().outline.span_length(span);
            let (a, b) = (
                length(seam.a.piece, seam.a_span()),
                length(seam.b.piece, seam.b_span()),
            );
            assert!(
                (a - b).abs() < 0.12 * a.max(b) + 6.0,
                "seam {:?}: {a:.0} vs {b:.0} mm",
                seam.id
            );
        }
    }
}

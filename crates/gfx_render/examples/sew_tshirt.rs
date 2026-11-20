//! Builds a ready-to-drape project from the Dr. Cos men's T-shirt, printed
//! on 11 A4 pages, with the editor's own document operations:
//!
//! 1. import the tiled PDF (the pages are joined into one sheet);
//! 2. turn front and back upright and unfold them, the neck band too
//!    (all three are cut on the fold);
//! 3. one sleeve per arm, front to the front;
//! 4. split the sleeve caps where the front armhole ends, the neck band
//!    where the front neckline ends;
//! 5. fit the avatar (male, height and weight as given), place the pieces
//!    around it and sew;
//! 6. write a `.clo360` project.
//!
//! ```sh
//! cargo run --release -p gfx_render --example sew_tshirt -- \
//!     ~/Downloads/tshirt-en-cm-ml.pdf ~/Downloads/tshirt.clo360 [height_m] [weight_kg]
//! ```
//!
//! Lines are found by their shape and place, not by how the importer
//! numbers them.

use avatar::{BodyModel, BodyParams, Measure, PoseParams};
use cad_core::{BodyPart, EdgeSpan, PatternDocument, PieceId, PieceTransform};
use glam::{Vec2, Vec3};

fn piece_named(document: &PatternDocument, name: &str) -> PieceId {
    document
        .pieces()
        .iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("no piece {name}"))
        .id
}

fn length(document: &PatternDocument, piece: PieceId, span: EdgeSpan) -> f32 {
    document.piece(piece).unwrap().outline.span_length(span)
}

/// The line of `piece` for which `score` is highest.
fn best_line(
    document: &PatternDocument,
    piece: PieceId,
    score: impl Fn(&[Vec2], f32) -> f32,
) -> EdgeSpan {
    let outline = &document.piece(piece).unwrap().outline;
    outline
        .lines()
        .into_iter()
        .max_by(|a, b| {
            let s = |span: &EdgeSpan| {
                score(&outline.span_points(*span, 1.0), outline.span_length(*span))
            };
            s(a).total_cmp(&s(b))
        })
        .unwrap()
}

fn is_straight(points: &[Vec2]) -> bool {
    cad_core::fit::is_straight(points, 0.5)
}

fn middle(points: &[Vec2]) -> Vec2 {
    points.iter().copied().sum::<Vec2>() / points.len() as f32
}

/// The line of `piece` whose length is closest to `mm`.
fn line_of_length(document: &PatternDocument, piece: PieceId, mm: f32) -> EdgeSpan {
    let span = best_line(document, piece, |_, l| -(l - mm).abs());
    let found = length(document, piece, span);
    assert!(
        (found - mm).abs() < 3.0,
        "no line of about {mm} mm (closest {found:.1})"
    );
    span
}

/// All lines of `piece` of about `mm`.
fn lines_of_length(document: &PatternDocument, piece: PieceId, mm: f32) -> Vec<EdgeSpan> {
    let outline = &document.piece(piece).unwrap().outline;
    outline
        .lines()
        .into_iter()
        .filter(|s| (outline.span_length(*s) - mm).abs() < 3.0)
        .collect()
}

/// Splits the line `span` of `piece` at `mm` from its end nearest to
/// `from` (a pattern point). Returns the new turn point's edge.
fn split_at(document: &mut PatternDocument, piece: PieceId, span: EdgeSpan, from: Vec2, mm: f32) {
    let outline = &document.piece(piece).unwrap().outline;
    let points = outline.span_points(span, 1.0);
    let total = outline.span_length(span);
    let from_start = points[0].distance(from) < points[points.len() - 1].distance(from);
    let fraction = if from_start {
        mm / total
    } else {
        1.0 - mm / total
    };
    let (edge, t) = outline.span_location(span, fraction);
    document
        .insert_point(piece, edge, t)
        .expect("split inside the line");
}

/// Where a line's middle is on the avatar, as placed.
fn placed_middle(document: &PatternDocument, piece: PieceId, span: EdgeSpan) -> Vec3 {
    let p = document.piece(piece).unwrap();
    let points = p.outline.span_points(span, 1.0);
    p.arrangement.place(points[points.len() / 2])
}

/// Sews two sides, turned so that the ends lying closer together on the
/// avatar meet (what clicking matching ends does in the editor).
fn sew(document: &mut PatternDocument, a: (PieceId, EdgeSpan), b: (PieceId, EdgeSpan)) {
    let ends = |(piece, span): (PieceId, EdgeSpan)| {
        let p = document.piece(piece).unwrap();
        let points = p.outline.span_points(span, 1.0);
        (
            p.arrangement.place(points[0]),
            p.arrangement.place(points[points.len() - 1]),
        )
    };
    let ((a0, a1), (b0, b1)) = (ends(a), ends(b));
    let straight = a0.distance(b0) + a1.distance(b1);
    let crossed = a0.distance(b1) + a1.distance(b0);
    document.add_seam_spans(a, b, crossed < straight);
}

/// Of `candidates` on `piece`, the line placed nearest to `to`.
fn nearest(
    document: &PatternDocument,
    piece: PieceId,
    candidates: &[EdgeSpan],
    to: Vec3,
) -> EdgeSpan {
    *candidates
        .iter()
        .min_by(|a, b| {
            placed_middle(document, piece, **a)
                .distance(to)
                .total_cmp(&placed_middle(document, piece, **b).distance(to))
        })
        .unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: sew_tshirt <tshirt.pdf> <out.clo360> [height_m] [weight_kg]";
    let pdf = args.get(1).ok_or(usage)?;
    let out = args.get(2).ok_or(usage)?;
    let height: f32 = args
        .get(3)
        .map_or(Ok(1.87), |s| s.replace(',', ".").parse())?;
    let weight_kg: f32 = args
        .get(4)
        .map_or(Ok(92.0), |s| s.replace(',', ".").parse())?;

    // --- 1. import ---------------------------------------------------------------
    let pattern = io_cad::import_pattern_pdf(
        &std::fs::read(pdf).map_err(|e| format!("cannot read {pdf}: {e}"))?,
        &io_cad::PdfOptions::default(),
    )?;
    if let Some(layout) = &pattern.pages {
        println!("{}", layout.summary());
    }
    let mut document = pattern.to_document();
    let (front, back, sleeve, neck) = (
        piece_named(&document, "Front Body"),
        piece_named(&document, "Back Body"),
        piece_named(&document, "Sleeve"),
        piece_named(&document, "Neck"),
    );
    if let Some(fabric) = document.fabrics().first().map(|f| f.id) {
        let fabric = document.fabric_mut(fabric).unwrap();
        fabric.name = "Cotton jersey".into();
        fabric.base_color = [0.05, 0.09, 0.20];
        fabric.pattern = cad_core::FabricPattern::Knit;
    }

    // --- 2. upright, unfolded -------------------------------------------------------
    // The bodies lie upside down on the sheet (hem at the top).
    for id in [front, back] {
        document.transform_piece(id, PieceTransform::RotateCw);
        document.transform_piece(id, PieceTransform::RotateCw);
    }
    // The fold: a body's longest straight line; the band's short end
    // marked "Symmetry" (the left one).
    for id in [front, back] {
        let fold = best_line(
            &document,
            id,
            |points, l| if is_straight(points) { l } else { 0.0 },
        );
        assert!(document.unfold_piece(id, fold.first), "unfold {id:?}");
    }
    let band_fold = best_line(&document, neck, |points, l| {
        if l < 100.0 {
            -middle(points).x
        } else {
            f32::MIN
        }
    });
    assert!(document.unfold_piece(neck, band_fold.first));
    for id in [front, back, neck] {
        let p = document.piece(id).unwrap();
        let (lo, hi) = p.outline.bounds();
        println!(
            "{}: unfolded to {:.0} × {:.0} mm",
            p.name,
            hi.x - lo.x,
            hi.y - lo.y
        );
    }

    // --- 3. sleeves -------------------------------------------------------------------
    // The sleeve is drawn with its front on the left. Wrapped round the
    // right arm, the piece's +x side faces forward: mirror it for that arm;
    // the mirrored copy, front on the left again, is the left sleeve.
    document.transform_piece(sleeve, PieceTransform::FlipHorizontal);
    let sleeve_left = document.duplicate_piece(sleeve, true).unwrap();
    document.piece_mut(sleeve).unwrap().name = "Sleeve right".into();
    document.piece_mut(sleeve_left).unwrap().name = "Sleeve left".into();
    document.piece_mut(sleeve).unwrap().placement.part = BodyPart::RightArm;
    document.piece_mut(sleeve_left).unwrap().placement.part = BodyPart::LeftArm;
    document.piece_mut(back).unwrap().placement.angle = 180.0;
    document.piece_mut(neck).unwrap().placement.part = BodyPart::Neck;

    // --- 4. split caps and band where the armholes and necklines meet ---------------
    let front_armhole = lines_of_length(&document, front, 268.8);
    let back_armhole = lines_of_length(&document, back, 259.7);
    assert_eq!(
        (front_armhole.len(), back_armhole.len()),
        (2, 2),
        "armholes"
    );
    let front_armhole_mm = length(&document, front, front_armhole[0]);
    for (id, front_side_x) in [(sleeve, f32::MAX), (sleeve_left, f32::MIN)] {
        let cap = line_of_length(&document, id, 527.8);
        // The cap's front end: on the +x side of the right sleeve.
        let points = document.piece(id).unwrap().outline.span_points(cap, 1.0);
        let (a, b) = (points[0], points[points.len() - 1]);
        let front_end = if (a.x - b.x) * front_side_x.signum() > 0.0 {
            a
        } else {
            b
        };
        split_at(&mut document, id, cap, front_end, front_armhole_mm);
    }
    let front_neckline = best_line(&document, front, |points, l| {
        if is_straight(points) {
            f32::MIN
        } else {
            middle(points).y - (l - 349.0).abs() * 10.0
        }
    });
    let back_neckline = best_line(&document, back, |points, l| {
        if is_straight(points) {
            f32::MIN
        } else {
            middle(points).y - (l - 299.0).abs() * 10.0
        }
    });
    let (front_neck_mm, back_neck_mm) = (
        length(&document, front, front_neckline),
        length(&document, back, back_neckline),
    );
    println!("necklines: front {front_neck_mm:.1} mm, back {back_neck_mm:.1} mm");
    // The back neckline is cut at centre back, where the band's ends meet.
    {
        let outline = &document.piece(back).unwrap().outline;
        let (edge, t) = outline.span_location(back_neckline, 0.5);
        document.insert_point(back, edge, t).unwrap();
    }
    // The band's lower edge: back half | front | back half.
    let band_edge = best_line(&document, neck, |points, l| {
        if l > 100.0 {
            -middle(points).y
        } else {
            f32::MIN
        }
    });
    let band_mm = length(&document, neck, band_edge);
    let band_start = document
        .piece(neck)
        .unwrap()
        .outline
        .span_points(band_edge, 1.0)[0];
    let back_share = band_mm * (back_neck_mm * 0.5) / (front_neck_mm + back_neck_mm);
    split_at(
        &mut document,
        neck,
        band_edge,
        band_start,
        band_mm - back_share,
    );
    let rest_of_edge = line_of_length(&document, neck, band_mm - back_share);
    split_at(&mut document, neck, rest_of_edge, band_start, back_share);

    document.tidy_layout(40.0);

    // --- 5. the avatar ------------------------------------------------------------------
    let model = BodyModel::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/avatar/body.bin"
    ))?;
    let mut params = BodyParams::male();
    // The weight is reached by the build, refitting the height each time.
    let weigh = |params: &mut BodyParams| {
        avatar::fit_all(&model, params, &[(Measure::Height, height)]);
        avatar::measure(&model, &model.build(params)).weight_kg
    };
    let bisect = |params: &mut BodyParams, set: fn(&mut BodyParams, f32), low: f32, high: f32| {
        let (mut low, mut high) = (low, high);
        for _ in 0..14 {
            let mid = 0.5 * (low + high);
            set(params, mid);
            if weigh(params) < weight_kg {
                low = mid;
            } else {
                high = mid;
            }
        }
        set(params, 0.5 * (low + high));
        weigh(params)
    };
    // Heavier builds carry both more fat and more muscle: one slider for
    // both, from slender (0.2) to heavy and strong (1.0).
    let reached = bisect(
        &mut params,
        |p, v| {
            p.weight = v;
            p.muscle = 0.5 + 0.5 * (v - 0.5).max(0.0) / 0.5;
        },
        0.2,
        1.0,
    );
    if (reached - weight_kg).abs() > 0.5 {
        println!("warning: the avatar reaches only {reached:.1} kg");
    }
    let rest = model.build(&params);
    let measurements = avatar::measure(&model, &rest);
    println!(
        "avatar: {:.2} m, {:.1} kg (fat {:.2}, muscle {:.2}, scale {:.3}), chest {:.0} cm, neck {:.0} cm, upper arm {:.0} cm",
        measurements.get(Measure::Height),
        measurements.weight_kg,
        params.weight,
        params.muscle,
        params.scale,
        measurements.get(Measure::Bust) * 100.0,
        measurements.get(Measure::Neck) * 100.0,
        measurements.get(Measure::UpperArm) * 100.0,
    );
    let pose = PoseParams::default();
    let body = model.pose(&rest, &pose.to_pose(&model, &rest));
    if let Some((shoulder, elbow)) = body.arm_axis(true) {
        println!("left arm: shoulder {shoulder:.2}, elbow {elbow:.2}");
    }
    avatar::arrange_around(&mut document, &body, &measurements);

    // --- 6. sew ---------------------------------------------------------------------------
    let shoulder = |document: &PatternDocument, id| lines_of_length(document, id, 94.3);
    let side = |document: &PatternDocument, id| lines_of_length(document, id, 451.5);
    for (fronts, backs) in [
        (shoulder(&document, front), shoulder(&document, back)),
        (side(&document, front), side(&document, back)),
    ] {
        assert_eq!((fronts.len(), backs.len()), (2, 2));
        for f in fronts {
            let b = nearest(&document, back, &backs, placed_middle(&document, front, f));
            sew(&mut document, (front, f), (back, b));
        }
    }
    for id in [sleeve, sleeve_left] {
        let cap_front = line_of_length(&document, id, front_armhole_mm);
        let cap_back = line_of_length(&document, id, 527.8 - front_armhole_mm);
        let at = placed_middle(&document, id, cap_front);
        let f = nearest(&document, front, &front_armhole, at);
        sew(&mut document, (id, cap_front), (front, f));
        let at = placed_middle(&document, id, cap_back);
        let b = nearest(&document, back, &back_armhole, at);
        sew(&mut document, (id, cap_back), (back, b));
        // The underarm seam closes the sleeve.
        let sides = lines_of_length(&document, id, 75.7);
        assert_eq!(sides.len(), 2, "sleeve sides");
        sew(&mut document, (id, sides[0]), (id, sides[1]));
    }
    // Neck band: middle to the front neckline, ends to the back halves,
    // and its two ends together at centre back.
    let band_front = line_of_length(&document, neck, band_mm - 2.0 * back_share);
    sew(&mut document, (neck, band_front), (front, front_neckline));
    let band_backs = lines_of_length(&document, neck, back_share);
    let back_halves = lines_of_length(&document, back, back_neck_mm * 0.5);
    assert_eq!((band_backs.len(), back_halves.len()), (2, 2), "neck band");
    for b in band_backs {
        let half = nearest(
            &document,
            back,
            &back_halves,
            placed_middle(&document, neck, b),
        );
        sew(&mut document, (neck, b), (back, half));
    }
    let ends = lines_of_length(&document, neck, 40.0);
    assert_eq!(ends.len(), 2, "band ends");
    sew(&mut document, (neck, ends[0]), (neck, ends[1]));

    println!(
        "{} pieces, {} seams:",
        document.pieces().len(),
        document.seams().len()
    );
    for seam in document.seams() {
        let name = |id| document.piece(id).unwrap().name.clone();
        println!(
            "  {} ({:.0} mm) ↔ {} ({:.0} mm){}",
            name(seam.a.piece),
            length(&document, seam.a.piece, seam.a_span()),
            name(seam.b.piece),
            length(&document, seam.b.piece, seam.b_span()),
            if seam.reversed { ", reversed" } else { "" }
        );
    }

    // --- 7. the project ---------------------------------------------------------------
    let project = serde_json::json!({
        "format": "clo360-project",
        "version": 1,
        "document": document,
        "avatar": {
            "gender": params.gender,
            "weight": params.weight,
            "muscle": params.muscle,
            "stature": params.stature,
            "cup": params.cup,
            "scale": params.scale,
            "morphs": params.morphs.to_vec(),
            "targets": [["Height", height]],
            "arm_angle": pose.arm_angle,
            "arm_forward": pose.arm_forward,
            "elbow_bend": pose.elbow_bend,
            "leg_angle": pose.leg_angle,
            "knee_bend": pose.knee_bend,
            "looks": null,
        },
        "particle_distance": 20.0,
    });
    std::fs::write(out, serde_json::to_string_pretty(&project)?)?;
    println!("wrote {out}");
    Ok(())
}

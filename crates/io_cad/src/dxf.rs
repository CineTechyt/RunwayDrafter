//! DXF pattern export for manufacturing: DXF-AAMA (ANSI/AAMA-292) and
//! DXF-ASTM (ASTM D6673).
//!
//! Both are plain DXF with a fixed vocabulary that cutters and markers
//! understand:
//!
//! * one `BLOCK` per pattern piece, placed with an `INSERT`;
//! * what an entity *is* follows from its numbered layer — 1 cut line,
//!   14 sew line, 4 notches, 7 grain line, 2/3 turn and curve points …
//!   (see [`layer`]);
//! * piece and style data travel as `TEXT` entities with fixed prefixes
//!   (`Piece Name: …`, `Size: …`).
//!
//! [`DxfWriter`] is the generic builder (group code / value pairs,
//! sections, entities); [`export_pattern_dxf`] maps a
//! [`PatternDocument`] onto the vocabulary. Coordinates are pattern space:
//! millimetres, Y up.

use std::fmt::Write as _;

use cad_core::geometry::{self, Curve};
use cad_core::{EdgeRef, NotchKind, PatternDocument, PatternPiece, PointKind};
use glam::Vec2;

/// The numbered layers of the AAMA/ASTM vocabulary.
pub mod layer {
    /// Piece boundary: the line the cutter follows. Also carries the piece
    /// and style system text.
    pub const BOUNDARY: u32 = 1;
    /// Turn points: corners of boundary, internal and sew lines.
    pub const TURN_POINTS: u32 = 2;
    /// Curve points: the remaining points of those lines.
    pub const CURVE_POINTS: u32 = 3;
    /// Notches (V and slit).
    pub const NOTCHES: u32 = 4;
    pub const GRADE_REFERENCE: u32 = 5;
    pub const MIRROR_LINE: u32 = 6;
    pub const GRAIN_LINE: u32 = 7;
    pub const INTERNAL_LINES: u32 = 8;
    pub const INTERNAL_CUTOUTS: u32 = 11;
    pub const DRILL_HOLES: u32 = 13;
    pub const SEW_LINE: u32 = 14;
    /// Annotation text that is not system text.
    pub const ANNOTATION: u32 = 15;
    /// ASTM only: T-notches.
    pub const T_NOTCHES: u32 = 80;
    /// ASTM only: a second copy of the boundary for quality validation.
    pub const BOUNDARY_VALIDATION: u32 = 84;
    /// ASTM only: a second copy of the sew lines for quality validation.
    pub const SEW_LINE_VALIDATION: u32 = 87;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DxfStandard {
    /// ANSI/AAMA-292, the older and most widely read dialect.
    #[default]
    Aama,
    /// ASTM D6673: adds notch-type layers and validation curves.
    Astm,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DxfOptions {
    pub standard: DxfStandard,
    pub style_name: String,
    /// The size the pieces are drawn in, e.g. "M" or "38".
    pub sample_size: String,
    pub author: String,
    /// Largest deviation of the exported polylines from the true curves.
    pub tolerance_mm: f32,
    /// `(date, time)` to stamp the file with; `None` uses the clock.
    pub timestamp: Option<(String, String)>,
}

impl Default for DxfOptions {
    fn default() -> Self {
        Self {
            standard: DxfStandard::default(),
            style_name: "Untitled".to_owned(),
            sample_size: "M".to_owned(),
            author: "clo360".to_owned(),
            tolerance_mm: 0.1,
            timestamp: None,
        }
    }
}

// ---------------------------------------------------------------------------
// The builder
// ---------------------------------------------------------------------------

/// Writes DXF (release 12 subset) as group code / value pairs.
///
/// ```text
///   0        ← group code: what the next line is (0 = entity type)
/// LINE       ← value
///   8        ← 8 = layer name
/// 7
///  10        ← 10/20 = first point x/y, 11/21 = second point
/// 0.0
/// ```
#[derive(Debug, Default)]
pub struct DxfWriter {
    out: String,
}

impl DxfWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// One group: the code right-aligned in three columns, then the value.
    pub fn pair(&mut self, code: u32, value: impl std::fmt::Display) -> &mut Self {
        let _ = write!(self.out, "{code:>3}\n{value}\n");
        self
    }

    /// A coordinate or length, with micrometre precision.
    pub fn number(&mut self, code: u32, value: f32) -> &mut Self {
        // No "-0.000": some readers choke on it.
        let value = if value.abs() < 5e-4 { 0.0 } else { value };
        self.pair(code, format_args!("{value:.3}"))
    }

    fn point(&mut self, first_code: u32, p: Vec2) -> &mut Self {
        self.number(first_code, p.x).number(first_code + 10, p.y)
    }

    pub fn begin_section(&mut self, name: &str) -> &mut Self {
        self.pair(0, "SECTION").pair(2, name)
    }

    pub fn end_section(&mut self) -> &mut Self {
        self.pair(0, "ENDSEC")
    }

    /// The HEADER section: DXF version and drawing extents.
    pub fn header(&mut self, min: Vec2, max: Vec2) -> &mut Self {
        self.begin_section("HEADER");
        self.pair(9, "$ACADVER").pair(1, "AC1009");
        self.pair(9, "$EXTMIN").point(10, min);
        self.pair(9, "$EXTMAX").point(10, max);
        self.end_section()
    }

    /// The TABLES section with one LAYER entry per layer used. Pattern
    /// readers go by the layer number alone, but general CAD programs want
    /// the table.
    pub fn layer_table(&mut self, layers: &[u32]) -> &mut Self {
        self.begin_section("TABLES");
        self.pair(0, "TABLE").pair(2, "LAYER").pair(70, layers.len());
        for layer in layers {
            // 62 = colour number, 6 = line type.
            self.pair(0, "LAYER").pair(2, layer).pair(70, 0).pair(62, 1 + layer % 7).pair(6, "CONTINUOUS");
        }
        self.pair(0, "ENDTAB");
        self.end_section()
    }

    pub fn begin_block(&mut self, name: &str, layer: u32) -> &mut Self {
        self.pair(0, "BLOCK").pair(8, layer).pair(2, name).pair(70, 0).point(10, Vec2::ZERO)
    }

    pub fn end_block(&mut self, layer: u32) -> &mut Self {
        self.pair(0, "ENDBLK").pair(8, layer)
    }

    /// Places a block.
    pub fn insert(&mut self, block: &str, layer: u32, at: Vec2) -> &mut Self {
        self.pair(0, "INSERT").pair(8, layer).pair(2, block).point(10, at)
    }

    pub fn line(&mut self, layer: u32, a: Vec2, b: Vec2) -> &mut Self {
        self.pair(0, "LINE").pair(8, layer).point(10, a).point(11, b)
    }

    /// A POLYLINE with one VERTEX per point. `closed` sets flag 70 = 1,
    /// which joins the last point to the first.
    pub fn polyline(&mut self, layer: u32, points: &[Vec2], closed: bool) -> &mut Self {
        // 66 = "vertices follow".
        self.pair(0, "POLYLINE").pair(8, layer).pair(66, 1).pair(70, u32::from(closed));
        for &p in points {
            self.pair(0, "VERTEX").pair(8, layer).point(10, p);
        }
        self.pair(0, "SEQEND").pair(8, layer)
    }

    pub fn point_entity(&mut self, layer: u32, p: Vec2) -> &mut Self {
        self.pair(0, "POINT").pair(8, layer).point(10, p)
    }

    /// A notch: a POINT on the cut line whose third coordinate carries the
    /// notch depth and whose angle (degrees from +X) points into the piece.
    pub fn notch(&mut self, layer: u32, p: Vec2, depth: f32, angle_degrees: f32) -> &mut Self {
        self.point_entity(layer, p).number(30, depth).number(50, angle_degrees)
    }

    /// Single-line text, `height` in drawing units.
    pub fn text(&mut self, layer: u32, at: Vec2, height: f32, text: &str) -> &mut Self {
        // DXF text is one line of plain characters.
        let clean: String = text.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
        self.pair(0, "TEXT").pair(8, layer).point(10, at).number(40, height).pair(1, clean)
    }

    pub fn finish(mut self) -> String {
        self.pair(0, "EOF");
        self.out
    }
}

// ---------------------------------------------------------------------------
// Pattern → DXF
// ---------------------------------------------------------------------------

/// Splits the vertices of a closed polygon into turn points (corners
/// sharper than 15°) and curve points.
fn classify_points(polygon: &[Vec2]) -> (Vec<Vec2>, Vec<Vec2>) {
    let count = polygon.len();
    let (mut turns, mut curves) = (Vec::new(), Vec::new());
    for i in 0..count {
        let incoming = (polygon[i] - polygon[(i + count - 1) % count]).normalize_or_zero();
        let outgoing = (polygon[(i + 1) % count] - polygon[i]).normalize_or_zero();
        if incoming.dot(outgoing) < 15f32.to_radians().cos() { turns.push(polygon[i]) } else { curves.push(polygon[i]) }
    }
    (turns, curves)
}

/// Removes consecutive duplicates (also across the closing edge).
fn dedup_closed(points: &mut Vec<Vec2>) {
    points.dedup_by(|a, b| a.distance_squared(*b) < 1e-8);
    while points.len() > 1 && points[0].distance_squared(points[points.len() - 1]) < 1e-8 {
        points.pop();
    }
}

/// A block name readers accept: letters, digits, `_` and `-`, and unique.
fn block_name(piece: &PatternPiece, used: &mut Vec<String>) -> String {
    let mut name: String =
        piece.name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c.to_ascii_uppercase() } else { '_' }).collect();
    if name.is_empty() {
        name = "PIECE".to_owned();
    }
    if used.contains(&name) {
        name = format!("{name}_{}", piece.id.0);
    }
    used.push(name.clone());
    name
}

fn write_piece(dxf: &mut DxfWriter, document: &PatternDocument, piece: &PatternPiece, name: &str, options: &DxfOptions) {
    let astm = options.standard == DxfStandard::Astm;
    dxf.begin_block(name, layer::BOUNDARY);

    // --- sew line and cut line ---------------------------------------------
    let flat = piece.outline.flatten(options.tolerance_mm);
    let mut sew_line = flat.points.clone();
    dedup_closed(&mut sew_line);
    let mut cut_line = piece.seam_allowance_outline(options.tolerance_mm);
    if let Some(line) = &mut cut_line {
        dedup_closed(line);
    }
    // Without seam allowance the piece is cut on the sew line.
    let boundary = cut_line.as_deref().unwrap_or(&sew_line);
    dxf.polyline(layer::BOUNDARY, boundary, true);
    if astm {
        dxf.polyline(layer::BOUNDARY_VALIDATION, boundary, true);
    }
    let (turns, curves) = classify_points(boundary);
    turns.iter().for_each(|&p| _ = dxf.point_entity(layer::TURN_POINTS, p));
    curves.iter().for_each(|&p| _ = dxf.point_entity(layer::CURVE_POINTS, p));

    if cut_line.is_some() {
        dxf.polyline(layer::SEW_LINE, &sew_line, true);
        if astm {
            dxf.polyline(layer::SEW_LINE_VALIDATION, &sew_line, true);
        }
        // The sew line's corners are the outline's corner anchors; that is
        // exact, where an angle test could misread a tight curve.
        let anchors = piece.outline.points();
        for &p in &sew_line {
            let corner = anchors.iter().any(|a| a.kind == PointKind::Corner && a.position.distance_squared(p) < 1e-6);
            dxf.point_entity(if corner { layer::TURN_POINTS } else { layer::CURVE_POINTS }, p);
        }
    }

    // --- notches ---------------------------------------------------------------
    for notch in &piece.notches {
        let Some((on_sew_line, inward)) = piece.notch_frame(notch) else { continue };
        // Notches are cut from the cutting edge inwards.
        let allowance = if piece.seam_allowance.enabled { piece.seam_allowance.width(notch.edge) } else { 0.0 };
        let position = on_sew_line - inward * allowance;
        let notch_layer = match (notch.kind, astm) {
            (NotchKind::T, true) => layer::T_NOTCHES,
            _ => layer::NOTCHES,
        };
        dxf.notch(notch_layer, position, notch.depth, inward.to_angle().to_degrees());
    }

    // --- grain line --------------------------------------------------------------
    let (from, to) = piece.grainline.end_points();
    dxf.line(layer::GRAIN_LINE, from, to);

    // --- system text ---------------------------------------------------------------
    let (lo, hi) = boundary.iter().fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(lo, hi), &p| (lo.min(p), hi.max(p)));
    let height = ((hi.y - lo.y) * 0.02).clamp(3.0, 10.0);
    let fabric = document.fabric(piece.fabric).map_or("", |f| f.name.as_str());
    let lines = [
        format!("Piece Name: {}", piece.name),
        format!("Size: {}", options.sample_size),
        "Quantity: 1".to_owned(),
        format!("Material: {fabric}"),
        format!("Category: Layer {}", piece.layer),
        "Annotation: ".to_owned(),
    ];
    let centre = (lo + hi) * 0.5;
    for (row, line) in lines.iter().enumerate() {
        let at = Vec2::new(lo.x + (hi.x - lo.x) * 0.25, centre.y - row as f32 * height * 1.6);
        dxf.text(layer::BOUNDARY, at, height, line);
    }

    // --- seam partners, for whoever sews it ----------------------------------------
    for edge in 0..piece.outline.edge_count() {
        let this = EdgeRef::new(piece.id, edge);
        let Some(seam) = document.seam_of(this) else { continue };
        let other = if seam.a == this { seam.b } else { seam.a };
        let Some(partner) = document.piece(other.piece) else { continue };
        let curve = piece.outline.curve(edge);
        let label = format!("Seam {}: to {} edge {}", seam.id.0, partner.name, other.edge);
        dxf.text(layer::ANNOTATION, curve.eval(0.5), height * 0.6, &label);
    }

    dxf.end_block(layer::BOUNDARY);
}

/// Days since 1970-01-01 → (year, month, day), proleptic Gregorian.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let mp = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (year_of_era + era * 400 + i64::from(month <= 2), month, day)
}

fn now() -> (String, String) {
    let seconds = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64);
    let (year, month, day) = civil_from_days(seconds.div_euclid(86_400));
    let time = seconds.rem_euclid(86_400);
    (format!("{day:02}-{month:02}-{year}"), format!("{:02}:{:02}:{:02}", time / 3600, time / 60 % 60, time % 60))
}

/// Writes every piece of `document` as DXF-AAMA or DXF-ASTM: cut line,
/// sew line, turn and curve points, notches, grain line and system text
/// per piece. Pieces keep their position on the 2D canvas.
pub fn export_pattern_dxf(document: &PatternDocument, options: &DxfOptions) -> String {
    let astm = options.standard == DxfStandard::Astm;
    let mut dxf = DxfWriter::new();

    let (mut lo, mut hi) = (Vec2::INFINITY, Vec2::NEG_INFINITY);
    for piece in document.pieces() {
        let outline = piece.seam_allowance_outline(1.0).unwrap_or_else(|| piece.outline.flatten(1.0).points);
        for p in outline {
            (lo, hi) = (lo.min(p), hi.max(p));
        }
    }
    if !lo.is_finite() {
        (lo, hi) = (Vec2::ZERO, Vec2::ZERO);
    }
    dxf.header(lo, hi);

    let mut layers = vec![
        layer::BOUNDARY,
        layer::TURN_POINTS,
        layer::CURVE_POINTS,
        layer::NOTCHES,
        layer::GRAIN_LINE,
        layer::SEW_LINE,
        layer::ANNOTATION,
    ];
    if astm {
        layers.extend([layer::T_NOTCHES, layer::BOUNDARY_VALIDATION, layer::SEW_LINE_VALIDATION]);
    }
    dxf.layer_table(&layers);

    let mut names = Vec::new();
    dxf.begin_section("BLOCKS");
    for piece in document.pieces() {
        let name = block_name(piece, &mut names);
        write_piece(&mut dxf, document, piece, &name, options);
    }
    dxf.end_section();

    dxf.begin_section("ENTITIES");
    // Style system text, once per file, above the pattern.
    let (date, time) = options.timestamp.clone().unwrap_or_else(now);
    let style = [
        match options.standard {
            DxfStandard::Aama => "ANSI/AAMA".to_owned(),
            DxfStandard::Astm => "ASTM/D13Proposal 1 Version: D 6673-04".to_owned(),
        },
        match options.standard {
            DxfStandard::Aama => "VERSION: 1.0.0".to_owned(),
            DxfStandard::Astm => format!("Style Name: {}", options.style_name),
        },
        format!("Creation Date: {date}"),
        format!("Creation Time: {time}"),
        format!("Author: {}", options.author),
        format!("Sample Size: {}", options.sample_size),
        "Grade Rule Table: ".to_owned(),
        "Units: METRIC".to_owned(),
    ];
    for (row, line) in style.iter().enumerate() {
        dxf.text(layer::BOUNDARY, Vec2::new(lo.x, hi.y + 20.0 + (style.len() - row) as f32 * 12.0), 8.0, line);
    }
    for name in &names {
        dxf.insert(name, layer::BOUNDARY, Vec2::ZERO);
    }
    dxf.end_section();
    dxf.finish()
}

/// Reads DXF text back into its `(group code, value)` pairs. For tests
/// and tools; not a general DXF reader.
pub fn read_pairs(dxf: &str) -> Option<Vec<(u32, &str)>> {
    let mut lines = dxf.lines();
    let mut pairs = Vec::new();
    while let Some(code) = lines.next() {
        pairs.push((code.trim().parse().ok()?, lines.next()?));
    }
    Some(pairs)
}

/// Area enclosed by a polygon; exposed for checks on exported outlines.
pub fn polygon_area(points: &[Vec2]) -> f32 {
    geometry::signed_area(points).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_core::Notch;

    /// The entities of block `name` as `(type, layer, pairs)`.
    fn block<'a>(pairs: &'a [(u32, &'a str)], name: &str) -> Vec<(&'a str, u32, &'a [(u32, &'a str)])> {
        let start = pairs.windows(3).position(|w| w[0] == (0, "BLOCK") && w[2] == (2, name)).expect("block exists");
        let end = start + pairs[start..].iter().position(|p| *p == (0, "ENDBLK")).unwrap();
        let body = &pairs[start + 1..end];
        let starts: Vec<usize> = body.iter().enumerate().filter(|(_, p)| p.0 == 0).map(|(i, _)| i).collect();
        starts
            .iter()
            .enumerate()
            .map(|(k, &at)| {
                let entity = &body[at..starts.get(k + 1).copied().unwrap_or(body.len())];
                (entity[0].1, entity.iter().find(|p| p.0 == 8).unwrap().1.parse().unwrap(), entity)
            })
            .collect()
    }

    fn polyline_points(entities: &[(&str, u32, &[(u32, &str)])], on_layer: u32) -> Vec<Vec2> {
        let start = entities.iter().position(|e| e.0 == "POLYLINE" && e.1 == on_layer).expect("polyline on layer");
        assert!(entities[start].2.contains(&(70, "1")), "closed");
        entities[start + 1..]
            .iter()
            .take_while(|e| e.0 == "VERTEX")
            .map(|e| {
                let value = |code| e.2.iter().find(|p| p.0 == code).unwrap().1.parse::<f32>().unwrap();
                Vec2::new(value(10), value(20))
            })
            .collect()
    }

    fn document_with_notch() -> PatternDocument {
        let mut document = PatternDocument::demo_top();
        let id = document.pieces()[0].id;
        let piece = document.piece_mut(id).unwrap();
        piece.notches.push(Notch { edge: 0, t: 0.5, kind: NotchKind::T, depth: 6.0 });
        document
    }

    #[test]
    fn the_file_is_well_formed_and_holds_one_block_per_piece() {
        let document = document_with_notch();
        let text = export_pattern_dxf(&document, &DxfOptions::default());
        let pairs = read_pairs(&text).expect("alternating code/value lines");
        assert_eq!(pairs.last(), Some(&(0, "EOF")));
        let count = |entity: &str| pairs.iter().filter(|p| **p == (0, entity)).count();
        assert_eq!(count("SECTION"), 4);
        assert_eq!(count("SECTION"), count("ENDSEC"));
        assert_eq!(count("BLOCK"), document.pieces().len());
        assert_eq!(count("BLOCK"), count("ENDBLK"));
        assert_eq!(count("INSERT"), document.pieces().len());
        assert_eq!(count("POLYLINE"), count("SEQEND"));
        assert!(text.contains("ANSI/AAMA\n") && text.contains("Units: METRIC\n"));
    }

    #[test]
    fn a_piece_keeps_its_sew_line_cut_line_notch_and_grain() {
        let document = document_with_notch();
        let piece = &document.pieces()[0];
        let text = export_pattern_dxf(&document, &DxfOptions::default());
        let pairs = read_pairs(&text).unwrap();
        let name: String = piece.name.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_uppercase() } else { '_' }).collect();
        let entities = block(&pairs, &name);

        // Sew line = the outline; cut line = outline + seam allowance.
        let sew = polyline_points(&entities, layer::SEW_LINE);
        let cut = polyline_points(&entities, layer::BOUNDARY);
        let outline = piece.outline.flatten(0.1).points;
        assert!((polygon_area(&sew) - polygon_area(&outline)).abs() < 1.0);
        assert!(polygon_area(&cut) > polygon_area(&sew) * 1.02, "the cut line lies outside the sew line");
        for anchor in piece.outline.points() {
            assert!(sew.iter().any(|p| p.distance(anchor.position) < 2e-3), "anchor {} is on the sew line", anchor.position);
        }

        // Every boundary vertex is announced as a turn or a curve point.
        let points = |on_layer| entities.iter().filter(|e| e.0 == "POINT" && e.1 == on_layer).count();
        assert_eq!(points(layer::TURN_POINTS) + points(layer::CURVE_POINTS), sew.len() + cut.len());
        assert!(points(layer::TURN_POINTS) >= 2 * piece.outline.points().iter().filter(|p| p.kind == PointKind::Corner).count() - 2);

        // The notch sits on the cut line, with its depth; AAMA has no
        // T-notch layer.
        let notch = entities.iter().find(|e| e.0 == "POINT" && e.1 == layer::NOTCHES).expect("notch");
        let value = |code| notch.2.iter().find(|p| p.0 == code).unwrap().1.parse::<f32>().unwrap();
        assert_eq!(value(30), 6.0);
        let at = Vec2::new(value(10), value(20));
        assert!(geometry::distance_to_polygon(at, &cut) < 0.05, "notch is on the cut line");

        // The grain line.
        let grain = entities.iter().find(|e| e.0 == "LINE" && e.1 == layer::GRAIN_LINE).expect("grain line");
        let value = |code| grain.2.iter().find(|p| p.0 == code).unwrap().1.parse::<f32>().unwrap();
        let (from, to) = piece.grainline.end_points();
        assert!(Vec2::new(value(10), value(20)).distance(from) < 2e-3 && Vec2::new(value(11), value(21)).distance(to) < 2e-3);

        assert!(entities.iter().any(|e| e.0 == "TEXT" && e.2.contains(&(1, format!("Piece Name: {}", piece.name).as_str()))));
        assert!(entities.iter().any(|e| e.0 == "TEXT" && e.1 == layer::ANNOTATION), "seam partners are annotated");
    }

    #[test]
    fn astm_adds_its_header_notch_layers_and_validation_curves() {
        let document = document_with_notch();
        let options = DxfOptions { standard: DxfStandard::Astm, style_name: "Top 1".to_owned(), ..DxfOptions::default() };
        let text = export_pattern_dxf(&document, &options);
        let pairs = read_pairs(&text).unwrap();
        assert!(text.contains("ASTM/D13Proposal 1 Version: D 6673-04\n") && text.contains("Style Name: Top 1\n"));
        let on_layer = |entity: &str, on: u32| {
            pairs.windows(2).filter(|w| w[0] == (0, entity) && w[1] == (8, on.to_string().as_str())).count()
        };
        assert_eq!(on_layer("POINT", layer::T_NOTCHES), 1);
        assert_eq!(on_layer("POLYLINE", layer::BOUNDARY_VALIDATION), document.pieces().len());
        assert_eq!(on_layer("POLYLINE", layer::SEW_LINE_VALIDATION), document.pieces().len());
    }

    #[test]
    fn dates_come_out_right() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
        assert_eq!(civil_from_days(20_769), (2026, 11, 12));
    }
}

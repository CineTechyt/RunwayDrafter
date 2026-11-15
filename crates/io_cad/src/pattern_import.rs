//! Pattern vector data assembly, piece recognition and role classification
//! (cutting line vs seam allowance).

use cad_core::geometry;
use cad_core::{
    ControlPoint, FabricId, Outline, PatternDocument, PatternPiece, PieceId, PointKind, Segment,
};
use glam::Vec2;
use serde::{Deserialize, Serialize};

/// How far a simplified outline may deviate from the drawn one, in mm.
const SIMPLIFY_TOLERANCE_MM: f32 = 0.25;
/// Outlines turning by more than this at a point keep a corner there.
const CORNER_DEGREES: f32 = 15.0;

/// The semantic role of a vector line or loop in a pattern piece.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LineRole {
    /// Sewing line (stitching outline): where parts are sewn together.
    /// This is the primary boundary used for cloth meshing and 3D simulation.
    #[default]
    SeamAllowance,
    /// Cutting line: the outer perimeter where the fabric is cut with scissors/cutter.
    CuttingLine,
    /// Internal markings, grainlines, notches or reference lines.
    Internal,
}

impl LineRole {
    pub const ALL: [Self; 3] = [Self::CuttingLine, Self::SeamAllowance, Self::Internal];

    pub fn label(self) -> &'static str {
        match self {
            Self::CuttingLine => "Cutting line",
            Self::SeamAllowance => "Seam allowance (Sew line)",
            Self::Internal => "Internal line",
        }
    }
}

/// A segment in a raw vector path before it is converted to a CAD outline.
#[derive(Debug, Clone, PartialEq)]
pub enum RawSegment {
    Line { to: Vec2 },
    Quadratic { ctrl: Vec2, to: Vec2 },
    Cubic { ctrl1: Vec2, ctrl2: Vec2, to: Vec2 },
}

impl RawSegment {
    pub fn end(&self) -> Vec2 {
        match *self {
            Self::Line { to } => to,
            Self::Quadratic { to, .. } => to,
            Self::Cubic { to, .. } => to,
        }
    }

    pub fn reversed(&self, from: Vec2) -> (RawSegment, Vec2) {
        match *self {
            Self::Line { to } => (RawSegment::Line { to: from }, to),
            Self::Quadratic { ctrl, to } => (RawSegment::Quadratic { ctrl, to: from }, to),
            Self::Cubic { ctrl1, ctrl2, to } => (
                RawSegment::Cubic {
                    ctrl1: ctrl2,
                    ctrl2: ctrl1,
                    to: from,
                },
                to,
            ),
        }
    }
}

/// A connected subpath of raw segments.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RawSubpath {
    pub start: Vec2,
    pub segments: Vec<RawSegment>,
    pub closed: bool,
}

impl RawSubpath {
    pub fn new(start: Vec2) -> Self {
        Self {
            start,
            segments: Vec::new(),
            closed: false,
        }
    }

    pub fn end(&self) -> Vec2 {
        self.segments.last().map_or(self.start, |s| s.end())
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// The path as a polyline, curves flattened to within `tolerance`.
    pub fn polyline(&self, tolerance: f32) -> Vec<Vec2> {
        use cad_core::geometry::{CubicBezier, Curve, QuadBezier};
        let mut points = vec![self.start];
        let mut current = self.start;
        for segment in &self.segments {
            match *segment {
                RawSegment::Line { to } => points.push(to),
                RawSegment::Quadratic { ctrl, to } => QuadBezier {
                    p0: current,
                    p1: ctrl,
                    p2: to,
                }
                .flatten_into(tolerance, &mut points),
                RawSegment::Cubic { ctrl1, ctrl2, to } => CubicBezier {
                    p0: current,
                    p1: ctrl1,
                    p2: ctrl2,
                    p3: to,
                }
                .flatten_into(tolerance, &mut points),
            }
            current = segment.end();
        }
        if self.closed && points.first() != points.last() {
            points.push(self.start);
        }
        points
    }

    pub fn reverse(&mut self) {
        let n = self.segments.len();
        if n == 0 {
            return;
        }
        let old_start = self.start;
        let old_end = self.end();
        let mut new_segs = Vec::with_capacity(n);
        for i in (0..n).rev() {
            let prev_pos = if i == 0 {
                old_start
            } else {
                self.segments[i - 1].end()
            };
            let (rev_seg, _) = self.segments[i].reversed(prev_pos);
            new_segs.push(rev_seg);
        }
        self.start = old_end;
        self.segments = new_segs;
    }

    /// Turns a closed subpath into an `Outline`.
    pub fn to_outline(&self) -> Option<Outline> {
        let n = self.segments.len();
        if n < 2 {
            return None;
        }
        let mut points = Vec::with_capacity(n);
        let mut segments = Vec::with_capacity(n);

        points.push(ControlPoint {
            position: self.start,
            kind: PointKind::Corner,
        });
        for (i, seg) in self.segments.iter().enumerate() {
            if i + 1 < n {
                points.push(ControlPoint {
                    position: seg.end(),
                    kind: PointKind::Corner,
                });
            }
            match *seg {
                RawSegment::Line { .. } => segments.push(Segment::Line),
                RawSegment::Quadratic { ctrl, .. } => segments.push(Segment::Quadratic { ctrl }),
                RawSegment::Cubic { ctrl1, ctrl2, .. } => {
                    segments.push(Segment::Cubic { ctrl1, ctrl2 })
                }
            }
        }

        let mut outline = Outline::new(points, segments);
        outline.ensure_ccw();
        Some(outline)
    }
}

/// A collection of raw vector subpaths extracted from a file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RawVectorDrawing {
    pub subpaths: Vec<RawSubpath>,
    pub labels: Vec<RawLabel>,
    /// Shapes only filled, never stroked: logos, lettering, arrow heads.
    /// They are kept as marks, never taken for outlines.
    pub fills: Vec<RawSubpath>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RawLabel {
    pub text: String,
    /// Where the text starts, on its baseline.
    pub position: Vec2,
    /// Letter height in mm; 0 when unknown.
    pub size: f32,
    /// Reading direction (unit vector).
    pub direction: Vec2,
    /// Each character and how far along `direction` from `position` it
    /// starts, in mm. Empty when the file does not say.
    pub glyphs: Vec<(char, f32)>,
    /// Length of the text along `direction`, in mm.
    pub width: f32,
}

impl RawLabel {
    /// A label whose letters' places are unknown.
    pub fn new(text: String, position: Vec2) -> Self {
        Self {
            text,
            position,
            size: 0.0,
            direction: Vec2::X,
            glyphs: Vec::new(),
            width: 0.0,
        }
    }

    /// Where character `index` ends along the text.
    fn glyph_end(&self, index: usize) -> f32 {
        self.glyphs.get(index + 1).map_or(self.width, |g| g.1)
    }

    /// The characters whose middle lies in `keep`, as a label of their own.
    pub(crate) fn keep_glyphs(&self, keep: impl Fn(Vec2) -> bool) -> Option<Self> {
        if self.glyphs.is_empty() {
            return keep(self.position).then(|| self.clone());
        }
        let kept: Vec<(char, f32, f32)> = (0..self.glyphs.len())
            .map(|i| (self.glyphs[i].0, self.glyphs[i].1, self.glyph_end(i)))
            .filter(|&(_, start, end)| keep(self.position + self.direction * (start + end) * 0.5))
            .collect();
        let (first, last) = (kept.first()?, kept.last()?);
        let shift = first.1;
        Some(Self {
            text: kept.iter().map(|g| g.0).collect(),
            position: self.position + self.direction * shift,
            size: self.size,
            direction: self.direction,
            glyphs: kept.iter().map(|g| (g.0, g.1 - shift)).collect(),
            width: last.2 - shift,
        })
    }
}

/// Puts text back together that was drawn in pieces: a word cut where two
/// pages meet, or set letter by letter. Characters on one baseline that
/// follow each other within a letter's height join; the same character
/// drawn twice at one place (the overlap of two pages) counts once.
pub(crate) fn merge_text(labels: Vec<RawLabel>) -> Vec<RawLabel> {
    let (placed, mut out): (Vec<RawLabel>, Vec<RawLabel>) = labels
        .into_iter()
        .partition(|l| !l.glyphs.is_empty() && l.size > 0.0);

    // Lines: same direction, size and baseline.
    let mut lines: Vec<Vec<RawLabel>> = Vec::new();
    for label in placed {
        let line = lines.iter_mut().find(|line| {
            let other = &line[0];
            let normal = other.direction.perp();
            other.direction.dot(label.direction) > 0.999
                && (other.size - label.size).abs() < 0.1 * other.size
                && (label.position - other.position).dot(normal).abs() < 0.2 * other.size
        });
        match line {
            Some(line) => line.push(label),
            None => lines.push(vec![label]),
        }
    }

    for line in lines {
        let origin = line[0].position;
        let direction = line[0].direction;
        let size = line[0].size;
        // (along the line, character, its end)
        let mut glyphs: Vec<(f32, char, f32)> = Vec::new();
        for label in &line {
            let base = (label.position - origin).dot(direction);
            for i in 0..label.glyphs.len() {
                glyphs.push((base + label.glyphs[i].1, label.glyphs[i].0, base + label.glyph_end(i)));
            }
        }
        glyphs.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut current: Option<RawLabel> = None;
        let mut end = f32::NEG_INFINITY;
        let mut last: Option<(f32, char)> = None;
        for (at, ch, stop) in glyphs {
            if let Some((was, prev)) = last
                && prev == ch
                && (at - was).abs() < 0.2 * size
            {
                continue; // drawn twice
            }
            if at - end > 1.5 * size {
                out.extend(current.take());
            }
            let label = current.get_or_insert_with(|| RawLabel {
                text: String::new(),
                position: origin + direction * at,
                size,
                direction,
                glyphs: Vec::new(),
                width: 0.0,
            });
            let start = (label.position - origin).dot(direction);
            if !label.text.is_empty() && at - end > 0.2 * size && !label.text.ends_with(' ') {
                label.glyphs.push((' ', end - start));
                label.text.push(' ');
            }
            label.glyphs.push((ch, at - start));
            label.text.push(ch);
            label.width = label.width.max(stop - start);
            end = end.max(stop);
            last = Some((at, ch));
        }
        out.extend(current);
    }
    for label in &mut out {
        label.text = label.text.trim().to_owned();
    }
    out.retain(|l| !l.text.is_empty());
    out
}

impl RawVectorDrawing {
    pub fn new() -> Self {
        Self::default()
    }

    /// Chains open subpaths whose endpoints meet within `tolerance_mm` and builds
    /// an `ImportedPattern`.
    pub fn assemble_pattern(&self, default_allowance_mm: f32) -> ImportedPattern {
        let mut chained = self.subpaths.clone();
        chain_subpaths(&mut chained, 1.5);
        for subpath in &mut chained {
            fit_polyline_segments(subpath, 2.0);
        }

        let mut candidate_loops = Vec::new();
        let mut open_lines = Vec::new();
        // Marks keep their drawn shape: chaining and simplifying, which
        // tidy up outlines, would wreck letters a few millimetres tall.
        let marks: Vec<RawSubpath> = self
            .subpaths
            .iter()
            .filter(|path| {
                let closed = path.closed || path.start.distance(path.end()) < 1.0;
                let outline_sized = path.to_outline().is_some_and(|o| {
                    let (min, max) = o.bounds();
                    o.signed_area().abs() > 100.0 && min.distance(max) > 15.0
                });
                !(closed && outline_sized) && !path.is_empty()
            })
            .cloned()
            .chain(self.fills.iter().filter(|p| !p.is_empty()).cloned())
            .collect();

        for subpath in chained {
            if subpath.is_empty() {
                continue;
            }
            let is_closed = subpath.closed || subpath.start.distance(subpath.end()) < 1.0;
            if is_closed {
                // Few segments with smooth curve points, instead of the
                // dozens of short pieces CAD exports draw curves with.
                if let Some(outline) = subpath
                    .to_outline()
                    .map(|o| o.simplified(SIMPLIFY_TOLERANCE_MM, CORNER_DEGREES))
                {
                    let (min, max) = outline.bounds();
                    let area = outline.signed_area().abs();
                    let diag = min.distance(max);
                    // Filter out microscopic dots or registration ticks.
                    if area > 100.0 && diag > 15.0 {
                        candidate_loops.push(ImportedLoop {
                            outline,
                            role: LineRole::SeamAllowance,
                            area,
                            bounds: (min, max),
                        });
                    }
                }
            } else {
                open_lines.push(subpath);
            }
        }

        deduplicate_loops(&mut candidate_loops, 0.75);

        // Sort loops by area descending so outer loops are examined first.
        candidate_loops.sort_by(|a, b| {
            b.area
                .partial_cmp(&a.area)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let polygons: Vec<Vec<Vec2>> = candidate_loops
            .iter()
            .map(|item| item.outline.flatten(1.0).points)
            .collect();
        let parents: Vec<Option<usize>> = (0..candidate_loops.len())
            .map(|child| {
                (0..candidate_loops.len())
                    .filter(|&outer| outer != child)
                    .filter(|&outer| {
                        candidate_loops[outer].area > candidate_loops[child].area
                            && loop_contains(
                                &candidate_loops[outer],
                                &polygons[outer],
                                &candidate_loops[child],
                                &polygons[child],
                            )
                    })
                    .min_by(|&a, &b| candidate_loops[a].area.total_cmp(&candidate_loops[b].area))
            })
            .collect();

        // Text inside each piece that may name it. Text found on most
        // pieces names the garment or the brand ("Tshirt", "MEN'S"), not
        // the piece.
        let naming = |root: usize| -> Vec<&RawLabel> {
            self.labels
                .iter()
                .filter(|label| {
                    !label.text.trim().is_empty()
                        && !is_pattern_annotation(&label.text)
                        && geometry::point_in_polygon(label.position, &polygons[root])
                })
                .collect()
        };
        let roots: Vec<usize> = (0..candidate_loops.len())
            .filter(|&root| parents[root].is_none())
            .collect();
        // Compared without case and punctuation, and a word cut short at
        // a page edge ("MEN" of "MEN'S") as the same text.
        let key = |text: &str| -> String {
            text.chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect()
        };
        let piece_texts: Vec<Vec<String>> = roots
            .iter()
            .map(|&root| naming(root).iter().map(|l| key(&l.text)).collect())
            .collect();
        let same = |a: &str, b: &str| {
            a == b || (a.len().min(b.len()) >= 3 && (a.starts_with(b) || b.starts_with(a)))
        };
        let shared = |text: &str| {
            let text = key(text);
            let on = piece_texts
                .iter()
                .filter(|texts| texts.iter().any(|t| same(t, &text)))
                .count();
            roots.len() >= 3 && on * 2 > roots.len()
        };

        let mut pieces = Vec::new();
        for root in 0..candidate_loops.len() {
            if parents[root].is_some() {
                continue;
            }

            let descendants: Vec<usize> = (0..candidate_loops.len())
                .filter(|&index| {
                    let mut ancestor = parents[index];
                    while let Some(parent) = ancestor {
                        if parent == root {
                            return true;
                        }
                        ancestor = parents[parent];
                    }
                    false
                })
                .collect();
            let sew_index = descendants
                .iter()
                .copied()
                .filter(|&index| {
                    parents[index] == Some(root)
                        && candidate_loops[index].area / candidate_loops[root].area >= 0.5
                })
                .max_by(|&a, &b| candidate_loops[a].area.total_cmp(&candidate_loops[b].area));

            let mut loops = Vec::with_capacity(descendants.len() + 1);
            let mut outer = candidate_loops[root].clone();
            outer.role = if sew_index.is_some() {
                LineRole::CuttingLine
            } else {
                LineRole::SeamAllowance
            };
            loops.push(outer);

            let allowance_mm = if let Some(sew_index) = sew_index {
                let mut sew_loop = candidate_loops[sew_index].clone();
                sew_loop.role = LineRole::SeamAllowance;
                let sew_points = &polygons[sew_index];
                let outer_points = &polygons[root];
                let mut distances = sew_points
                    .iter()
                    .step_by((sew_points.len() / 128).max(1))
                    .map(|point| geometry::distance_to_polygon(*point, outer_points));
                let mut sum = 0.0;
                let mut count = 0;
                for distance in &mut distances {
                    sum += distance;
                    count += 1;
                }
                let average = if count == 0 { 0.0 } else { sum / count as f32 };
                loops.push(sew_loop);

                for index in descendants
                    .iter()
                    .copied()
                    .filter(|&index| index != sew_index)
                {
                    let mut internal = candidate_loops[index].clone();
                    internal.role = LineRole::Internal;
                    loops.push(internal);
                }

                if average.is_finite() && average > 1.0 {
                    (average * 2.0).round() * 0.5
                } else {
                    default_allowance_mm
                }
            } else {
                for index in &descendants {
                    let mut internal = candidate_loops[*index].clone();
                    internal.role = LineRole::Internal;
                    loops.push(internal);
                }
                default_allowance_mm
            };

            let piece_num = pieces.len() + 1;
            // The biggest lettering names the piece; sentences are
            // instructions.
            let name = naming(root)
                .into_iter()
                .filter(|label| !shared(&label.text))
                .filter(|label| {
                    let text = label.text.trim();
                    !text.ends_with('.') && text.split_whitespace().count() <= 4
                })
                .max_by(|a, b| {
                    let size = |l: &RawLabel| (l.size * 2.0).round();
                    size(a)
                        .total_cmp(&size(b))
                        .then(a.text.trim().len().cmp(&b.text.trim().len()))
                })
                .map(|label| clean_piece_name(&label.text))
                .filter(|name| !name.is_empty())
                .unwrap_or_else(|| format!("Piece {piece_num}"));
            let grainline = find_grainline(&polygons[root], &marks);
            pieces.push(ImportedPieceCandidate {
                name,
                enabled: true,
                loops,
                seam_allowance_mm: allowance_mm,
                grainline,
            });
        }

        ImportedPattern {
            pieces,
            unassigned_open_lines: open_lines,
            marks,
            pages: None,
        }
    }
}

/// The grain line of a piece: the longest straight drawn line inside it
/// that spans at least a third of the piece.
fn find_grainline(polygon: &[Vec2], marks: &[RawSubpath]) -> Option<(Vec2, Vec2)> {
    let (lo, hi) = polygon
        .iter()
        .fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(lo, hi), &p| {
            (lo.min(p), hi.max(p))
        });
    let minimum = (hi - lo).max_element() / 3.0;
    marks
        .iter()
        .filter(|mark| !mark.closed)
        .map(|mark| mark.polyline(0.25))
        .filter(|points| points.len() >= 2 && cad_core::fit::is_straight(points, 0.75))
        .map(|points| (points[0], points[points.len() - 1]))
        .filter(|(a, b)| a.distance(*b) >= minimum)
        .filter(|(a, b)| {
            geometry::point_in_polygon(*a, polygon) && geometry::point_in_polygon(*b, polygon)
        })
        .max_by(|x, y| x.0.distance(x.1).total_cmp(&y.0.distance(y.1)))
}

/// Chains open subpaths whose ends meet within `tolerance` into longer
/// paths, closing those that come round to their own start.
///
/// Every end is paired with at most one other. Where more than two ends
/// meet (a grain line or a mark touching an outline), the pairs that
/// continue most smoothly, then most closely, win; the result does not
/// depend on the order the file draws its strokes in.
fn chain_subpaths(paths: &mut Vec<RawSubpath>, tolerance: f32) {
    // (path, at its end?, point, direction leaving the path there)
    let mut ends: Vec<(usize, bool, Vec2, Vec2)> = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        if path.closed || path.is_empty() {
            continue;
        }
        for at_end in [false, true] {
            let point = if at_end { path.end() } else { path.start };
            ends.push((i, at_end, point, outward(path, at_end)));
        }
    }
    ends.sort_by(|a, b| a.2.x.total_cmp(&b.2.x));

    let mut pairs = Vec::new();
    for a in 0..ends.len() {
        for b in a + 1..ends.len() {
            if ends[b].2.x - ends[a].2.x > tolerance {
                break;
            }
            let (ea, eb) = (&ends[a], &ends[b]);
            let distance = ea.2.distance(eb.2);
            if distance > tolerance {
                continue;
            }
            if ea.0 == eb.0 && rough_length(&paths[ea.0]) <= 4.0 * tolerance {
                // A stub's two ends lie close together; it is no loop.
                continue;
            }
            // Smooth: the two paths leave the joint in opposite directions.
            let turn = ea.3.angle_to(-eb.3).abs();
            let turn = if turn.is_finite() { turn } else { 0.0 };
            pairs.push((turn + 2.0 * distance / tolerance, a, b));
        }
    }
    pairs.sort_by(|x, y| x.0.total_cmp(&y.0));

    // link[path][0 = start, 1 = end] = (other path, its side)
    let mut link: Vec<[Option<(usize, usize)>; 2]> = vec![[None, None]; paths.len()];
    for (_, a, b) in pairs {
        let (pa, sa) = (ends[a].0, usize::from(ends[a].1));
        let (pb, sb) = (ends[b].0, usize::from(ends[b].1));
        if link[pa][sa].is_none() && link[pb][sb].is_none() {
            link[pa][sa] = Some((pb, sb));
            link[pb][sb] = Some((pa, sa));
        }
    }

    let mut used = vec![false; paths.len()];
    let mut out = Vec::with_capacity(paths.len());
    for first in 0..paths.len() {
        if used[first] {
            continue;
        }
        if link[first] == [None, None] {
            used[first] = true;
            out.push(std::mem::take(&mut paths[first]));
            continue;
        }
        // Walk back to the head of the chain, or once round a ring.
        // `reversed`: the path runs from its end to its start.
        let (mut head, mut reversed) = (first, false);
        loop {
            let entry = usize::from(reversed);
            match link[head][entry] {
                Some((j, side)) if j != first => {
                    head = j;
                    reversed = side == 0;
                }
                _ => break,
            }
        }
        let mut chain: Option<RawSubpath> = None;
        let (mut at, mut rev) = (head, reversed);
        loop {
            used[at] = true;
            let mut piece = std::mem::take(&mut paths[at]);
            if rev {
                piece.reverse();
            }
            match chain.as_mut() {
                None => chain = Some(piece),
                Some(chain) => {
                    // Close the gap by moving the piece onto the chain.
                    let gap = chain.end() - piece.start;
                    translate_subpath(&mut piece, gap);
                    chain.segments.extend(piece.segments);
                }
            }
            let exit = if rev { 0 } else { 1 };
            match link[at][exit] {
                Some((j, side)) if j == head && side == usize::from(reversed) => {
                    if let Some(chain) = chain.as_mut() {
                        let start = chain.start;
                        snap_subpath_end(chain, start);
                        chain.closed = true;
                    }
                    break;
                }
                Some((j, side)) if !used[j] => {
                    at = j;
                    rev = side == 1;
                }
                _ => break,
            }
        }
        out.extend(chain);
    }
    *paths = out;
}

/// Length of a path along its end points (curves as chords).
fn rough_length(path: &RawSubpath) -> f32 {
    let mut from = path.start;
    let mut length = 0.0;
    for segment in &path.segments {
        length += from.distance(segment.end());
        from = segment.end();
    }
    length
}

/// The direction a path leaves at its end (`at_end`) or its start, pointing
/// away from the path.
fn outward(path: &RawSubpath, at_end: bool) -> Vec2 {
    let Some(segment) = (if at_end { path.segments.last() } else { path.segments.first() }) else {
        return Vec2::ZERO;
    };
    let (tip, inner) = if at_end {
        let before = if path.segments.len() >= 2 {
            path.segments[path.segments.len() - 2].end()
        } else {
            path.start
        };
        let inner = match *segment {
            RawSegment::Line { .. } => before,
            RawSegment::Quadratic { ctrl, .. } => ctrl,
            RawSegment::Cubic { ctrl2, .. } => ctrl2,
        };
        (segment.end(), inner)
    } else {
        let inner = match *segment {
            RawSegment::Line { to } => to,
            RawSegment::Quadratic { ctrl, .. } => ctrl,
            RawSegment::Cubic { ctrl1, .. } => ctrl1,
        };
        (path.start, inner)
    };
    let direction = tip - inner;
    if direction.length() < 1e-6 {
        // Degenerate handle: fall back to the chord.
        let other = if at_end { path.start } else { path.end() };
        return (tip - other).normalize_or_zero();
    }
    direction.normalize()
}

fn translate_subpath(path: &mut RawSubpath, delta: Vec2) {
    path.start += delta;
    for segment in &mut path.segments {
        match segment {
            RawSegment::Line { to } | RawSegment::Quadratic { to, .. } => *to += delta,
            RawSegment::Cubic { ctrl1, ctrl2, to } => {
                *ctrl1 += delta;
                *ctrl2 += delta;
                *to += delta;
            }
        }
        if let RawSegment::Quadratic { ctrl, .. } = segment {
            *ctrl += delta;
        }
    }
}

fn snap_subpath_end(path: &mut RawSubpath, end: Vec2) {
    if let Some(segment) = path.segments.last_mut() {
        match segment {
            RawSegment::Line { to }
            | RawSegment::Quadratic { to, .. }
            | RawSegment::Cubic { to, .. } => *to = end,
        }
    }
}

fn deduplicate_loops(loops: &mut Vec<ImportedLoop>, tolerance: f32) {
    let mut unique: Vec<ImportedLoop> = Vec::with_capacity(loops.len());
    'candidate: for candidate in loops.drain(..) {
        let candidate_points = candidate.outline.flatten(1.0).points;
        for existing in &unique {
            if (candidate.bounds.0 - existing.bounds.0).abs().max_element() > tolerance
                || (candidate.bounds.1 - existing.bounds.1).abs().max_element() > tolerance
            {
                continue;
            }
            let existing_points = existing.outline.flatten(1.0).points;
            if contours_match(&candidate_points, &existing_points, tolerance) {
                continue 'candidate;
            }
        }
        unique.push(candidate);
    }
    *loops = unique;
}

fn contours_match(a: &[Vec2], b: &[Vec2], tolerance: f32) -> bool {
    fn points_match(source: &[Vec2], target: &[Vec2], tolerance: f32) -> bool {
        if source.is_empty() || target.is_empty() {
            return false;
        }
        let stride = (source.len() / 128).max(1);
        source
            .iter()
            .step_by(stride)
            .all(|point| geometry::distance_to_polygon(*point, target) <= tolerance)
    }

    points_match(a, b, tolerance) && points_match(b, a, tolerance)
}

fn loop_contains(
    outer: &ImportedLoop,
    outer_points: &[Vec2],
    inner: &ImportedLoop,
    inner_points: &[Vec2],
) -> bool {
    const BOUNDARY_TOLERANCE_MM: f32 = 1.5;
    if inner.bounds.0.x < outer.bounds.0.x - BOUNDARY_TOLERANCE_MM
        || inner.bounds.0.y < outer.bounds.0.y - BOUNDARY_TOLERANCE_MM
        || inner.bounds.1.x > outer.bounds.1.x + BOUNDARY_TOLERANCE_MM
        || inner.bounds.1.y > outer.bounds.1.y + BOUNDARY_TOLERANCE_MM
        || inner_points.is_empty()
    {
        return false;
    }

    let stride = (inner_points.len() / 16).max(1);
    inner_points.iter().step_by(stride).all(|point| {
        geometry::point_in_polygon(*point, outer_points)
            || geometry::distance_to_polygon(*point, outer_points) <= BOUNDARY_TOLERANCE_MM
    })
}

fn is_pattern_annotation(text: &str) -> bool {
    let text = text.to_ascii_lowercase();
    [
        "grain",
        "cut line",
        "cutting line",
        "sew line",
        "seam allowance",
        "piece qty",
        "fabric type",
        // Cutting instructions and marks.
        "cut on fold",
        "place on fold",
        "fold line",
        "symmetry",
        "sheets",
        "copyright",
        "©",
        // Addresses.
        "http",
        "www.",
        "@",
        ".com",
        ".info",
    ]
    .iter()
    .any(|marker| text.contains(marker))
        || text.split_whitespace().any(|word| word == "size")
}

fn clean_piece_name(text: &str) -> String {
    text.split_whitespace()
        .map(|word| word.replace(['_', '-'], " "))
        .collect::<Vec<_>>()
        .join(" ")
}

fn fit_polyline_segments(path: &mut RawSubpath, tolerance: f32) {
    let mut fitted = Vec::with_capacity(path.segments.len());
    let mut current = path.start;
    let mut index = 0;

    while index < path.segments.len() {
        if matches!(path.segments[index], RawSegment::Line { .. }) {
            let mut points = vec![current];
            while index < path.segments.len() {
                let RawSegment::Line { to } = path.segments[index] else {
                    break;
                };
                points.push(to);
                current = to;
                index += 1;
            }
            fit_polyline(&points, tolerance, &mut fitted);
        } else {
            let segment = path.segments[index].clone();
            current = segment.end();
            fitted.push(segment);
            index += 1;
        }
    }

    path.segments = fitted;
}

fn fit_polyline(points: &[Vec2], tolerance: f32, output: &mut Vec<RawSegment>) {
    if points.len() < 2 {
        return;
    }

    let start = points[0];
    let end = *points.last().unwrap();
    let line_error = points
        .iter()
        .map(|point| geometry::distance_to_segment(*point, start, end))
        .fold(0.0_f32, f32::max);
    if line_error <= tolerance {
        output.push(RawSegment::Line { to: end });
        return;
    }

    if points.len() >= 5 {
        if let Some((ctrl1, ctrl2, max_error, split_index)) = fit_cubic(points) {
            if max_error <= tolerance {
                output.push(RawSegment::Cubic {
                    ctrl1,
                    ctrl2,
                    to: end,
                });
                return;
            }
            let split_index = split_index.clamp(2, points.len() - 3);
            fit_polyline(&points[..=split_index], tolerance, output);
            fit_polyline(&points[split_index..], tolerance, output);
            return;
        }
    }

    if points.len() > 2 {
        let split_index = points.len() / 2;
        fit_polyline(&points[..=split_index], tolerance, output);
        fit_polyline(&points[split_index..], tolerance, output);
    } else {
        output.push(RawSegment::Line { to: end });
    }
}

fn fit_cubic(points: &[Vec2]) -> Option<(Vec2, Vec2, f32, usize)> {
    let start = points[0];
    let end = *points.last()?;
    let start_tangent = (points[1] - start).try_normalize()?;
    let end_tangent = (end - points[points.len() - 2]).try_normalize()?;

    let mut distances = Vec::with_capacity(points.len());
    distances.push(0.0_f32);
    for pair in points.windows(2) {
        distances.push(distances.last().copied()? + pair[0].distance(pair[1]));
    }
    let total = *distances.last()?;
    if total <= f32::EPSILON {
        return None;
    }

    let mut aa = 0.0;
    let mut ab = 0.0;
    let mut bb = 0.0;
    let mut ar = 0.0;
    let mut br = 0.0;
    for (index, point) in points.iter().enumerate().skip(1).take(points.len() - 2) {
        let t = distances[index] / total;
        let one_minus_t = 1.0 - t;
        let b0 = one_minus_t * one_minus_t * one_minus_t;
        let b1 = 3.0 * one_minus_t * one_minus_t * t;
        let b2 = 3.0 * one_minus_t * t * t;
        let b3 = t * t * t;
        let residual = *point - start * (b0 + b1) - end * (b2 + b3);
        let basis_a = start_tangent * b1;
        let basis_b = -end_tangent * b2;
        aa += basis_a.dot(basis_a);
        ab += basis_a.dot(basis_b);
        bb += basis_b.dot(basis_b);
        ar += basis_a.dot(residual);
        br += basis_b.dot(residual);
    }

    let determinant = aa * bb - ab * ab;
    if determinant.abs() <= 1e-8 {
        return None;
    }
    let alpha = (ar * bb - br * ab) / determinant;
    let beta = (aa * br - ab * ar) / determinant;
    if alpha <= 0.0 || beta <= 0.0 {
        return None;
    }

    let ctrl1 = start + start_tangent * alpha;
    let ctrl2 = end - end_tangent * beta;
    let mut max_error = 0.0_f32;
    let mut split_index = 1;
    for (index, point) in points.iter().enumerate().skip(1).take(points.len() - 2) {
        let t = distances[index] / total;
        let fitted = cubic_point(start, ctrl1, ctrl2, end, t);
        let error = point.distance(fitted);
        if error > max_error {
            max_error = error;
            split_index = index;
        }
    }
    Some((ctrl1, ctrl2, max_error, split_index))
}

fn cubic_point(start: Vec2, ctrl1: Vec2, ctrl2: Vec2, end: Vec2, t: f32) -> Vec2 {
    let one_minus_t = 1.0 - t;
    start * one_minus_t.powi(3)
        + ctrl1 * (3.0 * one_minus_t.powi(2) * t)
        + ctrl2 * (3.0 * one_minus_t * t.powi(2))
        + end * t.powi(3)
}

/// A closed loop imported from vector data, assigned a specific role.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedLoop {
    pub outline: Outline,
    pub role: LineRole,
    pub area: f32,
    pub bounds: (Vec2, Vec2),
}

/// A pattern piece candidate reconstructed from imported vector lines.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedPieceCandidate {
    pub name: String,
    pub enabled: bool,
    pub loops: Vec<ImportedLoop>,
    pub seam_allowance_mm: f32,
    /// End points of the grain line drawn on the piece, if one was found.
    pub grainline: Option<(Vec2, Vec2)>,
}

impl ImportedPieceCandidate {
    pub fn sew_loop(&self) -> Option<&ImportedLoop> {
        self.loops
            .iter()
            .find(|l| l.role == LineRole::SeamAllowance)
            .or_else(|| self.loops.first())
    }

    pub fn cut_loop(&self) -> Option<&ImportedLoop> {
        self.loops.iter().find(|l| l.role == LineRole::CuttingLine)
    }

    /// Swaps the roles of cutting line and seam allowance loops.
    pub fn swap_roles(&mut self) {
        if self.loops.len() >= 2 {
            for lp in &mut self.loops {
                lp.role = match lp.role {
                    LineRole::CuttingLine => LineRole::SeamAllowance,
                    LineRole::SeamAllowance => LineRole::CuttingLine,
                    LineRole::Internal => LineRole::Internal,
                };
            }
        } else if let Some(lp) = self.loops.first_mut() {
            lp.role = match lp.role {
                LineRole::CuttingLine => LineRole::SeamAllowance,
                LineRole::SeamAllowance => LineRole::CuttingLine,
                LineRole::Internal => LineRole::SeamAllowance,
            };
        }
    }

    /// Converts this candidate into a concrete `PatternPiece`.
    pub fn to_pattern_piece(&self, id: PieceId, fabric: FabricId) -> Option<PatternPiece> {
        let has_explicit_sew = self.loops.iter().any(|l| l.role == LineRole::SeamAllowance);
        let has_explicit_cut = self.loops.iter().any(|l| l.role == LineRole::CuttingLine);

        let outline = if has_explicit_sew {
            self.sew_loop()?.outline.clone()
        } else if has_explicit_cut {
            // Only a cutting line is selected: shrink inward by seam allowance so the
            // piece outline represents the sewing line!
            let cut = self.cut_loop()?;
            if self.seam_allowance_mm > 0.0 {
                cut.outline.offset(|_| -self.seam_allowance_mm)
            } else {
                cut.outline.clone()
            }
        } else {
            self.loops.first()?.outline.clone()
        };

        let mut piece = PatternPiece::new(id, self.name.clone(), outline, fabric);
        piece.seam_allowance.enabled = self.seam_allowance_mm > 0.0;
        piece.seam_allowance.default_width = self.seam_allowance_mm;
        self.apply_grainline(&mut piece);
        Some(piece)
    }

    /// Takes over the drawn grain line: the fabric's warp runs along it.
    fn apply_grainline(&self, piece: &mut PatternPiece) {
        let Some((a, b)) = self.grainline else { return };
        let direction = b - a;
        // Either end may come first: keep the angle in [0, π).
        let angle = direction
            .y
            .atan2(direction.x)
            .rem_euclid(std::f32::consts::PI);
        piece.grainline = cad_core::Grainline {
            origin: (a + b) * 0.5,
            angle,
            length: direction.length(),
        };
        piece.uv = cad_core::UvMapping::aligned_to_grain(&piece.grainline);
    }
}

/// The entire imported pattern document candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedPattern {
    pub pieces: Vec<ImportedPieceCandidate>,
    pub unassigned_open_lines: Vec<RawSubpath>,
    /// Every drawn path that is not a piece outline: lines, ticks, dots and
    /// the strokes of text drawn as lines (see [`crate::drawn_text`]).
    pub marks: Vec<RawSubpath>,
    /// How the pages were joined, for a pattern printed on many sheets.
    pub pages: Option<crate::tiles::PageLayout>,
}

impl ImportedPattern {
    /// Creates a complete new `PatternDocument` populated with the imported pieces.
    pub fn to_document(&self) -> PatternDocument {
        let mut doc = PatternDocument::new();
        let fabric = doc.add_fabric("Pattern Fabric", [0.72, 0.75, 0.82]);
        self.add_to_document(&mut doc, fabric);
        doc
    }

    /// Adds the enabled imported pieces to an existing `PatternDocument`.
    pub fn add_to_document(&self, doc: &mut PatternDocument, fabric: FabricId) -> Vec<PieceId> {
        let mut piece_ids = Vec::new();
        for candidate in &self.pieces {
            if !candidate.enabled {
                continue;
            }
            let (outline, allowance) = if let Some(sew) = candidate.sew_loop() {
                (sew.outline.clone(), candidate.seam_allowance_mm)
            } else if let Some(cut) = candidate.cut_loop() {
                if candidate.seam_allowance_mm > 0.0 {
                    (
                        cut.outline.offset(|_| -candidate.seam_allowance_mm),
                        candidate.seam_allowance_mm,
                    )
                } else {
                    (cut.outline.clone(), 0.0)
                }
            } else {
                continue;
            };

            let id = doc.add_piece(candidate.name.clone(), outline, fabric);
            if let Some(piece) = doc.piece_mut(id) {
                piece.seam_allowance.enabled = allowance > 0.0;
                piece.seam_allowance.default_width = allowance;
                candidate.apply_grainline(piece);
            }
            piece_ids.push(id);
        }
        piece_ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(min: f32, max: f32) -> RawSubpath {
        let mut path = RawSubpath::new(Vec2::splat(min));
        path.segments.push(RawSegment::Line {
            to: Vec2::new(max, min),
        });
        path.segments.push(RawSegment::Line {
            to: Vec2::splat(max),
        });
        path.segments.push(RawSegment::Line {
            to: Vec2::new(min, max),
        });
        path.segments.push(RawSegment::Line {
            to: Vec2::splat(min),
        });
        path.closed = true;
        path
    }

    fn rectangle(min: Vec2, max: Vec2) -> RawSubpath {
        let mut path = RawSubpath::new(min);
        path.segments.push(RawSegment::Line {
            to: Vec2::new(max.x, min.y),
        });
        path.segments.push(RawSegment::Line { to: max });
        path.segments.push(RawSegment::Line {
            to: Vec2::new(min.x, max.y),
        });
        path.segments.push(RawSegment::Line { to: min });
        path.closed = true;
        path
    }

    #[test]
    fn fragmented_outline_chains_from_both_ends_and_snaps_closed() {
        let mut paths = vec![
            RawSubpath {
                start: Vec2::new(10.0, 0.0),
                segments: vec![RawSegment::Line {
                    to: Vec2::new(10.0, 10.0),
                }],
                closed: false,
            },
            RawSubpath {
                start: Vec2::ZERO,
                segments: vec![RawSegment::Line {
                    to: Vec2::new(10.0, 0.1),
                }],
                closed: false,
            },
            RawSubpath {
                start: Vec2::new(10.0, 10.0),
                segments: vec![RawSegment::Line {
                    to: Vec2::new(0.0, 10.0),
                }],
                closed: false,
            },
            RawSubpath {
                start: Vec2::new(0.0, 10.0),
                segments: vec![RawSegment::Line { to: Vec2::ZERO }],
                closed: false,
            },
        ];

        chain_subpaths(&mut paths, 0.75);
        assert_eq!(paths.len(), 1);
        assert!(paths[0].closed);
        assert_eq!(paths[0].segments.len(), 4);
        assert_eq!(paths[0].start, paths[0].end());
    }

    #[test]
    fn duplicate_and_nested_contours_do_not_create_phantom_pieces() {
        let mut drawing = RawVectorDrawing::new();
        drawing.subpaths.extend([
            square(0.0, 200.0),
            square(10.0, 190.0),
            square(10.0, 190.0),
            square(80.0, 120.0),
        ]);

        let pattern = drawing.assemble_pattern(10.0);
        assert_eq!(pattern.pieces.len(), 1);
        assert_eq!(pattern.pieces[0].loops.len(), 3);
        assert_eq!(pattern.pieces[0].loops[0].role, LineRole::CuttingLine);
        assert_eq!(pattern.pieces[0].loops[1].role, LineRole::SeamAllowance);
        assert_eq!(pattern.pieces[0].loops[2].role, LineRole::Internal);
    }

    #[test]
    fn nested_sew_line_touching_cutline_bounds_stays_on_the_same_piece() {
        let mut drawing = RawVectorDrawing::new();
        drawing.subpaths.extend([
            square(0.0, 200.0),
            rectangle(Vec2::new(0.0, 10.0), Vec2::new(190.0, 190.0)),
        ]);

        let pattern = drawing.assemble_pattern(10.0);
        assert_eq!(pattern.pieces.len(), 1);
        assert_eq!(pattern.pieces[0].loops.len(), 2);
        assert_eq!(pattern.pieces[0].loops[0].role, LineRole::CuttingLine);
        assert_eq!(pattern.pieces[0].loops[1].role, LineRole::SeamAllowance);
    }

    #[test]
    fn raw_subpath_closure_and_outline() {
        let mut p = RawSubpath::new(Vec2::new(0.0, 0.0));
        p.segments.push(RawSegment::Line {
            to: Vec2::new(100.0, 0.0),
        });
        p.segments.push(RawSegment::Line {
            to: Vec2::new(100.0, 100.0),
        });
        p.segments.push(RawSegment::Line {
            to: Vec2::new(0.0, 100.0),
        });
        p.segments.push(RawSegment::Line {
            to: Vec2::new(0.0, 0.0),
        });
        p.closed = true;

        let outline = p.to_outline().unwrap();
        assert_eq!(outline.edge_count(), 4);
        assert!(outline.is_ccw());
    }

    #[test]
    fn polyline_fitting_merges_lines_and_recovers_smooth_curves() {
        let mut straight = RawSubpath::new(Vec2::ZERO);
        for index in 1..=128 {
            straight.segments.push(RawSegment::Line {
                to: Vec2::new(index as f32, 0.0),
            });
        }
        fit_polyline_segments(&mut straight, 0.5);
        assert_eq!(
            straight.segments,
            vec![RawSegment::Line {
                to: Vec2::new(128.0, 0.0)
            }]
        );

        let mut arc = RawSubpath::new(Vec2::new(100.0, 0.0));
        for index in 1..=64 {
            let angle = std::f32::consts::FRAC_PI_2 * index as f32 / 64.0;
            arc.segments.push(RawSegment::Line {
                to: Vec2::new(100.0 * angle.cos(), 100.0 * angle.sin()),
            });
        }
        fit_polyline_segments(&mut arc, 0.5);
        assert!(arc.segments.len() <= 2);
        assert!(
            arc.segments
                .iter()
                .all(|segment| matches!(segment, RawSegment::Cubic { .. }))
        );
    }

    #[test]
    fn concentric_loops_are_paired_as_cut_and_sew() {
        let mut drawing = RawVectorDrawing::new();
        // Outer loop (cutting line): 120 x 120
        let mut outer = RawSubpath::new(Vec2::new(-10.0, -10.0));
        outer.segments.push(RawSegment::Line {
            to: Vec2::new(110.0, -10.0),
        });
        outer.segments.push(RawSegment::Line {
            to: Vec2::new(110.0, 110.0),
        });
        outer.segments.push(RawSegment::Line {
            to: Vec2::new(-10.0, 110.0),
        });
        outer.segments.push(RawSegment::Line {
            to: Vec2::new(-10.0, -10.0),
        });
        outer.closed = true;

        // Inner loop (sew line): 100 x 100
        let mut inner = RawSubpath::new(Vec2::new(0.0, 0.0));
        inner.segments.push(RawSegment::Line {
            to: Vec2::new(100.0, 0.0),
        });
        inner.segments.push(RawSegment::Line {
            to: Vec2::new(100.0, 100.0),
        });
        inner.segments.push(RawSegment::Line {
            to: Vec2::new(0.0, 100.0),
        });
        inner.segments.push(RawSegment::Line {
            to: Vec2::new(0.0, 0.0),
        });
        inner.closed = true;

        drawing.subpaths.push(outer);
        drawing.subpaths.push(inner);

        let pattern = drawing.assemble_pattern(10.0);
        assert_eq!(pattern.pieces.len(), 1);
        let piece = &pattern.pieces[0];
        assert_eq!(piece.loops.len(), 2);
        assert_eq!(piece.cut_loop().unwrap().role, LineRole::CuttingLine);
        assert_eq!(piece.sew_loop().unwrap().role, LineRole::SeamAllowance);
        assert!((piece.seam_allowance_mm - 10.0).abs() < 1.0);
    }

    #[test]
    fn imported_pattern_creates_valid_document_and_swaps_roles() {
        let mut drawing = RawVectorDrawing::new();
        let mut outer = RawSubpath::new(Vec2::new(-10.0, -10.0));
        outer.segments.push(RawSegment::Line {
            to: Vec2::new(110.0, -10.0),
        });
        outer.segments.push(RawSegment::Line {
            to: Vec2::new(110.0, 110.0),
        });
        outer.segments.push(RawSegment::Line {
            to: Vec2::new(-10.0, 110.0),
        });
        outer.segments.push(RawSegment::Line {
            to: Vec2::new(-10.0, -10.0),
        });
        outer.closed = true;

        let mut inner = RawSubpath::new(Vec2::new(0.0, 0.0));
        inner.segments.push(RawSegment::Line {
            to: Vec2::new(100.0, 0.0),
        });
        inner.segments.push(RawSegment::Line {
            to: Vec2::new(100.0, 100.0),
        });
        inner.segments.push(RawSegment::Line {
            to: Vec2::new(0.0, 100.0),
        });
        inner.segments.push(RawSegment::Line {
            to: Vec2::new(0.0, 0.0),
        });
        inner.closed = true;

        drawing.subpaths.push(outer);
        drawing.subpaths.push(inner);

        let mut pattern = drawing.assemble_pattern(10.0);
        let doc = pattern.to_document();
        assert_eq!(doc.pieces().len(), 1);
        let p = &doc.pieces()[0];
        // The sew line is the 100x100 square
        assert_eq!(p.outline.edge_count(), 4);
        assert_eq!(p.outline.points()[0].position, Vec2::new(0.0, 0.0));
        assert!(p.seam_allowance.enabled);

        // Swap roles: now the outer loop becomes the sew line
        pattern.pieces[0].swap_roles();
        assert_eq!(
            pattern.pieces[0].sew_loop().unwrap().role,
            LineRole::SeamAllowance
        );
        assert_eq!(
            pattern.pieces[0].cut_loop().unwrap().role,
            LineRole::CuttingLine
        );

        let doc2 = pattern.to_document();
        let p2 = &doc2.pieces()[0];
        assert_eq!(p2.outline.points()[0].position, Vec2::new(-10.0, -10.0));
    }
}

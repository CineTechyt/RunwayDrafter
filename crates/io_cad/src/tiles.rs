//! Patterns printed on many small sheets ("print at home" A4 / US Letter
//! files): finds which page joins which and lays the pages out as the one
//! big sheet they were cut from.
//!
//! Nothing about the layout is taken from the file's page order: pattern
//! makers order pages by rows or by columns, leave out blank pages and let
//! neighbouring pages overlap by a glue margin. Instead, every pair of pages
//! is tested the way a person tapes them together: lines running off one
//! page's edge must carry on, at the same height, on the other page. Edge
//! labels printed on both pages ("Bonding line AB", "A1"…) count too.
//! The strongest joins are then placed first, like a jigsaw.

use std::collections::HashMap;

use glam::Vec2;

use crate::pattern_import::{ImportedPattern, RawLabel, RawSegment, RawSubpath, RawVectorDrawing};

/// One page of a PDF: its drawing in millimetres, the page's lower-left
/// corner at the origin. `size` is `None` when the page box is unknown.
#[derive(Debug, Clone, Default)]
pub(crate) struct Page {
    pub size: Option<Vec2>,
    pub drawing: RawVectorDrawing,
}

/// How the pages of a tiled pattern were put together.
#[derive(Debug, Clone, PartialEq)]
pub struct PageLayout {
    /// Pages in the file.
    pub pages: usize,
    /// Pages joined into the big sheet.
    pub joined: usize,
    /// Columns and rows of the joined sheet (including left-out blank pages).
    pub columns: usize,
    pub rows: usize,
    /// How far neighbouring pages overlap, beyond their trim frame, in mm.
    pub overlap_mm: Vec2,
    /// Paper of the joined pages, e.g. "A4".
    pub paper: String,
    /// Where joined pages meet, in pattern coordinates.
    pub seams: Vec<(Vec2, Vec2)>,
}

impl PageLayout {
    /// "11 A4 pages joined into a 5 × 3 sheet".
    pub fn summary(&self) -> String {
        let mut text = format!(
            "{} {} pages joined into a {} × {} sheet",
            self.joined, self.paper, self.columns, self.rows
        );
        if self.joined < self.pages {
            text.push_str(&format!(
                " ({} other page(s) placed beside it)",
                self.pages - self.joined
            ));
        }
        text
    }
}

/// An axis-aligned rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Rect {
    pub min: Vec2,
    pub max: Vec2,
}

impl Rect {
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self {
            min: min.min(max),
            max: min.max(max),
        }
    }

    pub fn from_points(points: impl IntoIterator<Item = Vec2>) -> Option<Self> {
        let mut iter = points.into_iter();
        let first = iter.next()?;
        let (min, max) = iter.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p)));
        Some(Self { min, max })
    }

    pub fn intersect(&self, other: &Self) -> Self {
        let min = self.min.max(other.min);
        Self {
            min,
            max: self.max.max(min).min(other.max).max(min),
        }
    }

    pub fn contains(&self, p: Vec2, slack: f32) -> bool {
        p.x >= self.min.x - slack
            && p.x <= self.max.x + slack
            && p.y >= self.min.y - slack
            && p.y <= self.max.y + slack
    }

    fn overlaps(&self, other: &Self) -> bool {
        self.min.x <= other.max.x
            && other.min.x <= self.max.x
            && self.min.y <= other.max.y
            && other.min.y <= self.max.y
    }

    fn size(&self) -> Vec2 {
        self.max - self.min
    }

    /// Whether `a` and `b` both lie on the same side of the rectangle.
    fn on_one_edge(&self, a: Vec2, b: Vec2) -> bool {
        const ON: f32 = 0.05;
        ((a.x - self.min.x).abs() < ON && (b.x - self.min.x).abs() < ON)
            || ((a.x - self.max.x).abs() < ON && (b.x - self.max.x).abs() < ON)
            || ((a.y - self.min.y).abs() < ON && (b.y - self.min.y).abs() < ON)
            || ((a.y - self.max.y).abs() < ON && (b.y - self.max.y).abs() < ON)
    }
}

/// The parts of `path` inside `rect`, appended to `out`. Curves crossing
/// the border are cut as polylines; curves wholly inside stay curves.
/// With `drop_edges`, pieces running along the border itself (a frame
/// line, a clip edge) are dropped.
pub(crate) fn clip_subpath(
    path: &RawSubpath,
    rect: Rect,
    drop_edges: bool,
    out: &mut Vec<RawSubpath>,
) {
    let points = || {
        std::iter::once(path.start).chain(path.segments.iter().flat_map(|s| match *s {
            RawSegment::Line { to } => [to, to, to],
            RawSegment::Quadratic { ctrl, to } => [ctrl, to, to],
            RawSegment::Cubic { ctrl1, ctrl2, to } => [ctrl1, ctrl2, to],
        }))
    };
    let Some(bounds) = Rect::from_points(points()) else {
        return;
    };
    if !bounds.overlaps(&rect) {
        return;
    }
    let inside = |p: Vec2| rect.contains(p, 1e-3);
    if bounds.min.x >= rect.min.x - 1e-3
        && bounds.min.y >= rect.min.y - 1e-3
        && bounds.max.x <= rect.max.x + 1e-3
        && bounds.max.y <= rect.max.y + 1e-3
        && !drop_edges
    {
        out.push(path.clone());
        return;
    }

    let first = out.len();
    let mut current: Option<RawSubpath> = None;
    let finish = |current: &mut Option<RawSubpath>, out: &mut Vec<RawSubpath>| {
        if let Some(done) = current.take()
            && !done.is_empty()
        {
            out.push(done);
        }
    };
    let line =
        |from: Vec2, to: Vec2, current: &mut Option<RawSubpath>, out: &mut Vec<RawSubpath>| {
            let Some((t0, t1)) = clip_line(from, to, rect) else {
                finish(current, out);
                return;
            };
            let a = from.lerp(to, t0);
            let b = from.lerp(to, t1);
            if a.distance(b) < 1e-4 || (drop_edges && rect.on_one_edge(a, b)) {
                finish(current, out);
                return;
            }
            if t0 > 1e-6 || current.is_none() {
                finish(current, out);
                *current = Some(RawSubpath::new(a));
            }
            if let Some(sub) = current.as_mut() {
                sub.segments.push(RawSegment::Line { to: b });
            }
            if t1 < 1.0 - 1e-6 {
                finish(current, out);
            }
        };

    let mut from = path.start;
    for segment in &path.segments {
        let to = segment.end();
        match segment {
            RawSegment::Line { .. } => line(from, to, &mut current, out),
            curve => {
                let control = match *curve {
                    RawSegment::Quadratic { ctrl, .. } => Rect::from_points([from, ctrl, to]),
                    RawSegment::Cubic { ctrl1, ctrl2, .. } => {
                        Rect::from_points([from, ctrl1, ctrl2, to])
                    }
                    RawSegment::Line { .. } => unreachable!(),
                }
                .expect("points");
                if inside(control.min) && inside(control.max) {
                    if current.is_none() {
                        current = Some(RawSubpath::new(from));
                    }
                    if let Some(sub) = current.as_mut() {
                        sub.segments.push(curve.clone());
                    }
                } else if !control.overlaps(&rect) {
                    finish(&mut current, out);
                } else {
                    let flat = RawSubpath {
                        start: from,
                        segments: vec![curve.clone()],
                        closed: false,
                    }
                    .polyline(0.05);
                    for pair in flat.windows(2) {
                        line(pair[0], pair[1], &mut current, out);
                    }
                }
            }
        }
        from = to;
    }
    let whole = out.len() == first;
    finish(&mut current, out);

    // A closed path cut open: its last piece runs on into its first.
    if path.closed && out.len() - first >= 2 {
        let last = out.len() - 1;
        if out[first].start.distance(path.start) < 1e-4
            && out[last].end().distance(path.start) < 1e-4
        {
            let head = out.remove(first);
            out.last_mut()
                .expect("piece")
                .segments
                .extend(head.segments);
        }
    } else if whole && out.len() == first + 1 && path.closed {
        let only = &mut out[first];
        if only.start.distance(path.start) < 1e-4 && only.end().distance(path.start) < 1e-4 {
            only.closed = true;
        }
    }
}

/// The part `t0..t1` of the line `a → b` inside `rect` (Liang–Barsky).
fn clip_line(a: Vec2, b: Vec2, rect: Rect) -> Option<(f32, f32)> {
    let d = b - a;
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for (p, q) in [
        (-d.x, a.x - rect.min.x),
        (d.x, rect.max.x - a.x),
        (-d.y, a.y - rect.min.y),
        (d.y, rect.max.y - a.y),
    ] {
        if p.abs() < 1e-9 {
            if q < -1e-4 {
                return None;
            }
        } else {
            let t = q / p;
            if p < 0.0 {
                t0 = t0.max(t);
            } else {
                t1 = t1.min(t);
            }
        }
    }
    (t0 <= t1).then_some((t0, t1))
}

/// Moves a path by `delta`.
fn translate(path: &mut RawSubpath, delta: Vec2) {
    path.start += delta;
    for segment in &mut path.segments {
        match segment {
            RawSegment::Line { to } => *to += delta,
            RawSegment::Quadratic { ctrl, to } => {
                *ctrl += delta;
                *to += delta;
            }
            RawSegment::Cubic { ctrl1, ctrl2, to } => {
                *ctrl1 += delta;
                *ctrl2 += delta;
                *to += delta;
            }
        }
    }
}

/// Gap between pages, or groups of pages, that are not joined.
const PAGE_GAP_MM: f32 = 20.0;

/// Pages put next to each other, left to right, without joining.
pub(crate) fn side_by_side(pages: Vec<Page>) -> RawVectorDrawing {
    let mut drawing = RawVectorDrawing::new();
    let mut x = 0.0;
    for page in pages {
        let delta = Vec2::new(x, 0.0);
        let width = page
            .size
            .map(|s| s.x)
            .unwrap_or_else(|| page_bounds(&page.drawing).map_or(0.0, |b| b.max.x));
        place(&mut drawing, page.drawing, delta);
        x += width + PAGE_GAP_MM;
    }
    drawing
}

fn place(into: &mut RawVectorDrawing, mut drawing: RawVectorDrawing, delta: Vec2) {
    for path in &mut drawing.subpaths {
        translate(path, delta);
    }
    for label in &mut drawing.labels {
        label.position += delta;
    }
    for path in &mut drawing.fills {
        translate(path, delta);
    }
    into.subpaths.append(&mut drawing.subpaths);
    into.labels.append(&mut drawing.labels);
    into.fills.append(&mut drawing.fills);
}

fn page_bounds(drawing: &RawVectorDrawing) -> Option<Rect> {
    Rect::from_points(
        drawing
            .subpaths
            .iter()
            .flat_map(|p| std::iter::once(p.start).chain(p.segments.iter().map(|s| s.end()))),
    )
}

/// A side of a page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Side {
    Left,
    Right,
    Bottom,
    Top,
}

const SIDES: [Side; 4] = [Side::Left, Side::Right, Side::Bottom, Side::Top];

/// Overlap candidates: 0 to 30 mm in half-millimetre steps.
const OVERLAP_STEP_MM: f32 = 0.5;
const OVERLAP_STEPS: usize = 61;
/// Lines continuing across a join must meet within this.
const MATCH_MM: f32 = 1.0;
/// Edge labels count within this distance of the frame.
const LABEL_BAND_MM: f32 = 12.0;

/// What one page shows at its edges.
struct Edges {
    /// The page's lines, flattened.
    lines: Vec<Vec<Vec2>>,
    /// Along-edge positions where lines run off each side.
    hits: HashMap<Side, Vec<f32>>,
    /// Along-edge positions where lines cross a line parallel to each side,
    /// `k · OVERLAP_STEP_MM` inside the frame; `k = 0` is `hits`.
    crossings: HashMap<Side, Vec<Vec<f32>>>,
    /// Labels printed along each side.
    labels: HashMap<Side, Vec<String>>,
}

/// Joins the pages of a tiled pattern. Returns the drawing on one big sheet
/// and how it was laid out, or the pages side by side and `None` when they
/// do not join.
pub(crate) fn join(pages: Vec<Page>) -> (RawVectorDrawing, Option<PageLayout>) {
    let count = pages.len();
    if count < 2 {
        return (side_by_side(pages), None);
    }

    // Tiles share one paper size; other pages (a cover, an A0 version) are not tiles.
    let size_of = |p: &Page| p.size.map(|s| (s.x.round() as i32, s.y.round() as i32));
    let mut sizes: HashMap<(i32, i32), usize> = HashMap::new();
    for size in pages.iter().filter_map(size_of) {
        *sizes.entry(size).or_default() += 1;
    }
    let Some((&paper, _)) = sizes.iter().max_by_key(|(size, n)| (**n, size.0 * size.1)) else {
        return (side_by_side(pages), None);
    };
    let tiles: Vec<usize> = (0..count)
        .filter(|&i| {
            size_of(&pages[i])
                .is_some_and(|s| (s.0 - paper.0).abs() <= 1 && (s.1 - paper.1).abs() <= 1)
        })
        .collect();
    if tiles.len() < 2 {
        return (side_by_side(pages), None);
    }
    let page_size = pages[tiles[0]].size.expect("sized");

    // The trim frame: guide lines across most of the page, at the same
    // place on most pages. They are furniture, not pattern: drop them.
    let mut pages = pages;
    let guides = find_guides(
        &tiles.iter().map(|&i| &pages[i]).collect::<Vec<_>>(),
        page_size,
    );
    for &i in &tiles {
        remove_guides(&mut pages[i].drawing, &guides, page_size);
    }
    let frame = find_frame(
        &tiles.iter().map(|&i| &pages[i]).collect::<Vec<_>>(),
        page_size,
        &guides,
    );

    // Each page's content inside its frame.
    let framed: Vec<RawVectorDrawing> = tiles
        .iter()
        .map(|&i| {
            let mut out = RawVectorDrawing::new();
            for path in &pages[i].drawing.subpaths {
                clip_subpath(path, frame, true, &mut out.subpaths);
            }
            out.labels = pages[i].drawing.labels.clone();
            out
        })
        .collect();
    let edges: Vec<Edges> = framed.iter().map(|d| page_edges(d, frame)).collect();

    // Edge labels that occur on exactly two pages pair those pages up.
    let mut label_pages: HashMap<&str, Vec<(usize, Side)>> = HashMap::new();
    for (t, e) in edges.iter().enumerate() {
        for (&side, texts) in &e.labels {
            for text in texts {
                label_pages
                    .entry(text.as_str())
                    .or_default()
                    .push((t, side));
            }
        }
    }
    let label_pair = |a: usize, sa: Side, b: usize, sb: Side| {
        label_pages.values().any(|places| {
            places.len() == 2 && places.contains(&(a, sa)) && places.contains(&(b, sb))
        })
    };

    // Score every ordered pair, B right of A and B below A, at every overlap:
    // (lines that carry on, lines that don't, how far apart they meet).
    type Fit = (i32, i32, f32);
    let mut fits: HashMap<(usize, usize, bool), Vec<Fit>> = HashMap::new();
    for a in 0..tiles.len() {
        for b in 0..tiles.len() {
            if a == b {
                continue;
            }
            for right in [true, false] {
                let (far, near) = sides(right);
                let per_overlap = (0..OVERLAP_STEPS)
                    .map(|k| {
                        let (m1, u1, e1) =
                            matches(&edges[a].hits[&far], &edges[b].crossings[&near][k]);
                        let (m2, u2, e2) =
                            matches(&edges[b].hits[&near], &edges[a].crossings[&far][k]);
                        (m1 + m2, u1 + u2, e1 + e2)
                    })
                    .collect();
                fits.insert((a, b, right), per_overlap);
            }
        }
    }
    let holds = |fit: &Fit| fit.0 >= 2 && fit.1 * 4 <= fit.0 + fit.1;
    let better = |x: &(usize, Fit), y: &(usize, Fit)| {
        // More lines carrying on, then a closer fit, then less overlap.
        (x.1.0 - x.1.1)
            .cmp(&(y.1.0 - y.1.1))
            .then(y.1.2.total_cmp(&x.1.2))
            .then(y.0.cmp(&x.0))
    };

    // One overlap for the whole sheet, so the pages line up on a grid: the
    // one that the pairs that join at all agree on best.
    let sheet_overlap = |right: bool| -> usize {
        let candidates: Vec<&Vec<Fit>> = fits
            .iter()
            .filter(|((_, _, r), list)| *r == right && list.iter().any(holds))
            .map(|(_, list)| list)
            .collect();
        (0..OVERLAP_STEPS)
            .map(|k| {
                let total = candidates.iter().fold((0, 0, 0.0), |acc, list| {
                    let f = list[k];
                    if holds(&f) {
                        (acc.0 + f.0, acc.1 + f.1, acc.2 + f.2)
                    } else {
                        acc
                    }
                });
                (k, total)
            })
            .max_by(better)
            .map_or(0, |(k, _)| k)
    };
    let overlap_steps = [sheet_overlap(true), sheet_overlap(false)];

    // The grid step is coarse for lines crossing a join at a shallow angle:
    // refine the overlap to 0.05 mm on the pairs that join.
    let refine = |right: bool, k: usize| -> f32 {
        let coarse = k as f32 * OVERLAP_STEP_MM;
        let (far, near) = sides(right);
        let pairs: Vec<(usize, usize)> = fits
            .iter()
            .filter(|((_, _, r), list)| *r == right && holds(&list[k]))
            .map(|(&(a, b, _), _)| (a, b))
            .collect();
        if pairs.is_empty() {
            return coarse;
        }
        let depth = |page: usize, side: Side, o: f32| {
            let e = &edges[page];
            if o < 1e-3 {
                return e.hits[&side].clone();
            }
            match side {
                Side::Left => crossings(&e.lines, true, frame.min.x + o),
                Side::Right => crossings(&e.lines, true, frame.max.x - o),
                Side::Bottom => crossings(&e.lines, false, frame.min.y + o),
                Side::Top => crossings(&e.lines, false, frame.max.y - o),
            }
        };
        (-10..=10)
            .map(|i| (coarse + i as f32 * 0.05).max(0.0))
            .map(|o| {
                let fit = pairs.iter().fold((0, 0, 0.0), |acc, &(a, b)| {
                    let (m1, u1, e1) = matches(&edges[a].hits[&far], &depth(b, near, o));
                    let (m2, u2, e2) = matches(&edges[b].hits[&near], &depth(a, far, o));
                    (acc.0 + m1 + m2, acc.1 + u1 + u2, acc.2 + e1 + e2)
                });
                (o, fit)
            })
            .max_by(|x, y| {
                (x.1.0 - x.1.1)
                    .cmp(&(y.1.0 - y.1.1))
                    .then(y.1.2.total_cmp(&x.1.2))
                    .then(y.0.total_cmp(&x.0))
            })
            .map_or(coarse, |(o, _)| o)
    };
    let overlap = Vec2::new(
        refine(true, overlap_steps[0]),
        refine(false, overlap_steps[1]),
    );

    // Joins that hold at that overlap (give or take a step), or that edge
    // labels name.
    struct Join {
        a: usize,
        b: usize,
        right: bool,
        score: f32,
    }
    let mut joins = Vec::new();
    for (&(a, b, right), list) in &fits {
        let k = overlap_steps[usize::from(!right)];
        let best = (k.saturating_sub(1)..=(k + 1).min(OVERLAP_STEPS - 1))
            .map(|k| (k, list[k]))
            .max_by(better)
            .expect("steps");
        let (far, near) = sides(right);
        let by_lines = holds(&best.1);
        let by_label = label_pair(a, far, b, near);
        if by_lines || by_label {
            joins.push(Join {
                a,
                b,
                right,
                score: if by_lines {
                    (best.1.0 - best.1.1) as f32
                } else {
                    0.0
                } + if by_label { 10.0 } else { 0.0 },
            });
        }
    }
    // Same order every run.
    joins.sort_by_key(|j| (j.a, j.b, j.right));

    if std::env::var_os("TILE_DEBUG").is_some() {
        eprintln!("frame {frame:?} guides {guides:?} overlap {overlap}");
        for (t, e) in edges.iter().enumerate() {
            let n = |s: Side| e.hits[&s].len();
            eprintln!(
                "page {}: hits L{} R{} B{} T{} labels {:?}",
                tiles[t] + 1,
                n(Side::Left),
                n(Side::Right),
                n(Side::Bottom),
                n(Side::Top),
                e.labels
            );
        }
        for j in &joins {
            eprintln!(
                "join {} {} {} score {}",
                tiles[j.a] + 1,
                if j.right { "→" } else { "↓" },
                tiles[j.b] + 1,
                j.score
            );
        }
    }
    if joins.is_empty() {
        return (side_by_side(pages), None);
    }

    let step = frame.size() - overlap;

    // Place the strongest joins first; a join that would put two pages on
    // one spot is a false match.
    joins.sort_by(|x, y| y.score.total_cmp(&x.score));
    let mut cell: Vec<Option<(i32, i32)>> = vec![None; tiles.len()];
    let mut group: Vec<usize> = (0..tiles.len()).collect();
    for join in &joins {
        let (ga, gb) = (group[join.a], group[join.b]);
        if ga == gb {
            continue;
        }
        let ca = cell[join.a].unwrap_or((0, 0));
        let cb = cell[join.b].unwrap_or((0, 0));
        let target = if join.right {
            (ca.0 + 1, ca.1)
        } else {
            (ca.0, ca.1 + 1)
        };
        let shift = (target.0 - cb.0, target.1 - cb.1);
        let moved: Vec<(usize, (i32, i32))> = (0..tiles.len())
            .filter(|&t| group[t] == gb)
            .map(|t| {
                let c = cell[t].unwrap_or((0, 0));
                (t, (c.0 + shift.0, c.1 + shift.1))
            })
            .collect();
        let clash = moved.iter().any(|(_, c)| {
            (0..tiles.len()).any(|t| group[t] == ga && cell[t].unwrap_or((0, 0)) == *c)
        });
        if clash {
            continue;
        }
        for t in 0..tiles.len() {
            if group[t] == ga && cell[t].is_none() {
                cell[t] = Some((0, 0));
            }
        }
        for (t, c) in moved {
            cell[t] = Some(c);
            group[t] = ga;
        }
    }

    // The largest group is the sheet; anything else goes beside it.
    let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
    for t in 0..tiles.len() {
        members.entry(group[t]).or_default().push(t);
    }
    let sheet: Vec<usize> = members
        .values()
        .max_by_key(|m| (m.len(), std::cmp::Reverse(m.iter().min().copied())))
        .cloned()
        .unwrap_or_default();
    if sheet.len() < 2 {
        return (side_by_side(pages), None);
    }
    let cells: HashMap<(i32, i32), usize> = sheet
        .iter()
        .map(|&t| (cell[t].unwrap_or((0, 0)), t))
        .collect();
    let (c0, c1) = cells.keys().fold((i32::MAX, i32::MIN), |(lo, hi), c| {
        (lo.min(c.0), hi.max(c.0))
    });
    let (r0, r1) = cells.keys().fold((i32::MAX, i32::MIN), |(lo, hi), c| {
        (lo.min(c.1), hi.max(c.1))
    });
    let rows = r1 - r0 + 1;

    // Furniture labels: edge labels that paired pages, and text printed on
    // most pages (a logo, a pattern title).
    let mut seen: HashMap<&str, usize> = HashMap::new();
    for d in &framed {
        let mut texts: Vec<&str> = d.labels.iter().map(|l| l.text.as_str()).collect();
        texts.sort_unstable();
        texts.dedup();
        for text in texts {
            *seen.entry(text).or_default() += 1;
        }
    }
    let furniture = |label: &RawLabel| {
        label_pages
            .get(label.text.as_str())
            .is_some_and(|p| p.len() == 2)
            || seen
                .get(label.text.as_str())
                .is_some_and(|&n| n * 2 > framed.len() && n > 2)
            || is_page_label(&label.text)
    };

    let mut drawing = RawVectorDrawing::new();
    let mut seams = Vec::new();
    let half = overlap * 0.5;
    for (&(c, r), &t) in &cells {
        // Pages are joined at the middle of their overlap.
        let has = |dc: i32, dr: i32| cells.contains_key(&(c + dc, r + dr));
        let cell_rect = Rect::new(
            Vec2::new(
                frame.min.x + if has(-1, 0) { half.x } else { 0.0 },
                frame.min.y + if has(0, 1) { half.y } else { 0.0 },
            ),
            Vec2::new(
                frame.max.x - if has(1, 0) { half.x } else { 0.0 },
                frame.max.y - if has(0, -1) { half.y } else { 0.0 },
            ),
        );
        // Rows run downwards; the bottom row sits at y = 0.
        let offset = Vec2::new((c - c0) as f32 * step.x, (r1 - r) as f32 * step.y) - frame.min;
        let mut piece = RawVectorDrawing::new();
        for path in &pages[tiles[t]].drawing.subpaths {
            clip_subpath(path, cell_rect, true, &mut piece.subpaths);
        }
        for path in &pages[tiles[t]].drawing.fills {
            clip_subpath(path, cell_rect, false, &mut piece.fills);
        }
        // Letters belong to the page whose part of the sheet they are on;
        // text cut at the join is put back together afterwards.
        piece.labels = pages[tiles[t]]
            .drawing
            .labels
            .iter()
            .filter(|l| !furniture(l))
            .filter_map(|l| l.keep_glyphs(|p| cell_rect.contains(p, 0.0)))
            .collect();
        place(&mut drawing, piece, offset);
        if has(1, 0) {
            seams.push((
                Vec2::new(cell_rect.max.x, cell_rect.min.y) + offset,
                cell_rect.max + offset,
            ));
        }
        if has(0, 1) {
            seams.push((
                cell_rect.min + offset,
                Vec2::new(cell_rect.max.x, cell_rect.min.y) + offset,
            ));
        }
    }

    drawing.subpaths = weld(std::mem::take(&mut drawing.subpaths), &seams);

    // Pages that did not join go to the right of the sheet.
    let joined: Vec<usize> = sheet.iter().map(|&t| tiles[t]).collect();
    let mut x = (c1 - c0 + 1) as f32 * step.x + overlap.x + PAGE_GAP_MM;
    for (index, page) in pages.into_iter().enumerate() {
        if joined.contains(&index) {
            continue;
        }
        let width = page.size.map_or(0.0, |s| s.x);
        place(&mut drawing, page.drawing, Vec2::new(x, 0.0));
        x += width + PAGE_GAP_MM;
    }

    let layout = PageLayout {
        pages: count,
        joined: joined.len(),
        columns: (c1 - c0 + 1) as usize,
        rows: rows as usize,
        overlap_mm: overlap,
        paper: paper_name(page_size),
        seams,
    };
    (drawing, Some(layout))
}

/// Matches of `values` in the sorted `targets`: (found, not found).
/// The facing sides of a join: B right of A, or B below A.
fn sides(right: bool) -> (Side, Side) {
    if right {
        (Side::Right, Side::Left)
    } else {
        (Side::Bottom, Side::Top)
    }
}

/// Rejoins the lines cut where pages meet: every cut end is welded to the
/// nearest cut end across the seam. Left to the general chaining, the stubs
/// a cut leaves (a line crossing the seam by half a millimetre) would be
/// taken for dots and break the outline.
fn weld(paths: Vec<RawSubpath>, seams: &[(Vec2, Vec2)]) -> Vec<RawSubpath> {
    const ON: f32 = 0.02;
    // Along the seam: a line crossing at a shallow angle lands a little
    // apart on the two pages when the overlap is off by a hair.
    const REACH: f32 = 3.0;
    const NEAR: f32 = 0.5;
    let on_seam = |p: Vec2| seams.iter().any(|&(a, b)| Rect::new(a, b).contains(p, ON));
    // (path, at its end?, point)
    let mut ends: Vec<(usize, bool, Vec2)> = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        if path.closed || path.is_empty() {
            continue;
        }
        for (at_end, point) in [(false, path.start), (true, path.end())] {
            if on_seam(point) {
                ends.push((i, at_end, point));
            }
        }
    }
    ends.sort_by(|a, b| a.2.x.total_cmp(&b.2.x));
    let mut pairs = Vec::new();
    for a in 0..ends.len() {
        for b in a + 1..ends.len() {
            if ends[b].2.x - ends[a].2.x > REACH {
                break;
            }
            if ends[a].0 == ends[b].0 && ends[a].1 == ends[b].1 {
                continue;
            }
            let d = ends[a].2.distance(ends[b].2);
            if d > REACH {
                continue;
            }
            // Beyond touching, the two must carry on in one direction.
            let leaving_a = outward_direction(&paths[ends[a].0], ends[a].1);
            let leaving_b = outward_direction(&paths[ends[b].0], ends[b].1);
            let turn = leaving_a.angle_to(-leaving_b).abs();
            let turn = if turn.is_finite() { turn } else { 0.0 };
            if d <= NEAR || turn < 0.2 {
                pairs.push((d + turn, a, b));
            }
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
            out.push(paths[first].clone());
            continue;
        }
        // Walk back to the head of the chain (or around a ring).
        // `reversed`: the path is run from its end to its start.
        let (mut head, mut reversed) = (first, false);
        loop {
            let entry = if reversed { 1 } else { 0 };
            match link[head][entry] {
                Some((j, side)) if j != first => {
                    head = j;
                    reversed = side == 0;
                }
                _ => break,
            }
        }
        let mut merged: Option<RawSubpath> = None;
        let (mut at, mut rev) = (head, reversed);
        loop {
            used[at] = true;
            let mut piece = paths[at].clone();
            if rev {
                piece.reverse();
            }
            match merged.as_mut() {
                None => merged = Some(piece),
                Some(chain) => chain.segments.extend(piece.segments),
            }
            let exit = if rev { 0 } else { 1 };
            match link[at][exit] {
                Some((j, _)) if j == head => {
                    if let Some(chain) = merged.as_mut() {
                        chain.closed = true;
                        let start = chain.start;
                        if let Some(last) = chain.segments.last_mut() {
                            match last {
                                RawSegment::Line { to }
                                | RawSegment::Quadratic { to, .. }
                                | RawSegment::Cubic { to, .. } => *to = start,
                            }
                        }
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
        out.extend(merged);
    }
    out
}

/// The direction a path leaves at its end (or start), away from the path.
fn outward_direction(path: &RawSubpath, at_end: bool) -> Vec2 {
    let points = path.polyline(0.1);
    let n = points.len();
    if n < 2 {
        return Vec2::ZERO;
    }
    let (tip, inner) = if at_end {
        (points[n - 1], points[n - 2])
    } else {
        (points[0], points[1])
    };
    (tip - inner).normalize_or_zero()
}

/// Matches of `values` in the sorted `targets`: (found, not found, summed
/// distance of the ones found).
fn matches(values: &[f32], targets: &[f32]) -> (i32, i32, f32) {
    let mut found = 0;
    let mut error = 0.0;
    for &v in values {
        let at = targets.partition_point(|&t| t < v - MATCH_MM);
        let nearest = targets[at..]
            .iter()
            .take_while(|&&t| t <= v + MATCH_MM)
            .map(|&t| (t - v).abs())
            .reduce(f32::min);
        if let Some(distance) = nearest {
            found += 1;
            error += distance;
        }
    }
    (found, values.len() as i32 - found, error)
}

/// Straight guide lines spanning most of the page, found at the same place
/// on at least half the pages: (vertical x positions, horizontal y positions).
fn find_guides(pages: &[&Page], size: Vec2) -> (Vec<f32>, Vec<f32>) {
    let mut vertical: HashMap<i32, usize> = HashMap::new();
    let mut horizontal: HashMap<i32, usize> = HashMap::new();
    for page in pages {
        let (mut v, mut h) = (Vec::new(), Vec::new());
        for path in &page.drawing.subpaths {
            let mut from = path.start;
            for segment in &path.segments {
                let to = segment.end();
                if let RawSegment::Line { .. } = segment {
                    let d = (to - from).abs();
                    if d.x < 0.1 && d.y > 0.9 * size.y {
                        v.push((from.x * 2.0).round() as i32);
                    } else if d.y < 0.1 && d.x > 0.9 * size.x {
                        h.push((from.y * 2.0).round() as i32);
                    }
                }
                from = to;
            }
        }
        for (list, counts) in [(&mut v, &mut vertical), (&mut h, &mut horizontal)] {
            list.sort_unstable();
            list.dedup();
            for &k in list.iter() {
                *counts.entry(k).or_default() += 1;
            }
        }
    }
    let keep = |counts: HashMap<i32, usize>| {
        let mut out: Vec<f32> = counts
            .into_iter()
            .filter(|&(_, n)| n * 4 >= pages.len() * 3 && n >= 2)
            .map(|(k, _)| k as f32 * 0.5)
            .collect();
        out.sort_by(f32::total_cmp);
        out
    };
    (keep(vertical), keep(horizontal))
}

fn is_guide(from: Vec2, to: Vec2, guides: &(Vec<f32>, Vec<f32>), size: Vec2) -> bool {
    let d = (to - from).abs();
    (d.x < 0.1 && d.y > 0.9 * size.y && guides.0.iter().any(|&x| (x - from.x).abs() <= 0.5))
        || (d.y < 0.1 && d.x > 0.9 * size.x && guides.1.iter().any(|&y| (y - from.y).abs() <= 0.5))
}

fn remove_guides(drawing: &mut RawVectorDrawing, guides: &(Vec<f32>, Vec<f32>), size: Vec2) {
    if guides.0.is_empty() && guides.1.is_empty() {
        return;
    }
    let mut out = Vec::with_capacity(drawing.subpaths.len());
    for path in drawing.subpaths.drain(..) {
        let mut current = RawSubpath::new(path.start);
        let mut from = path.start;
        let mut cut = false;
        for segment in path.segments.iter() {
            let to = segment.end();
            if matches!(segment, RawSegment::Line { .. }) && is_guide(from, to, guides, size) {
                cut = true;
                if !current.is_empty() {
                    out.push(std::mem::replace(&mut current, RawSubpath::new(to)));
                } else {
                    current = RawSubpath::new(to);
                }
            } else {
                current.segments.push(segment.clone());
            }
            from = to;
        }
        if !current.is_empty() {
            current.closed = path.closed && !cut;
            out.push(current);
        }
    }
    drawing.subpaths = out;
}

/// The trim frame of the pages: at the guide lines nearest the paper's
/// edges, else where lines most often run off, else the paper's edge.
fn find_frame(pages: &[&Page], size: Vec2, guides: &(Vec<f32>, Vec<f32>)) -> Rect {
    // Loose line ends per page. A cut edge leaves them on many pages at one
    // place; a pattern line drawn as many short strokes leaves them on one
    // or two pages only, so pages are counted, not ends.
    let free: Vec<(usize, (Vec2, Vec2))> = pages
        .iter()
        .enumerate()
        .flat_map(|(i, p)| free_ends(&p.drawing).into_iter().map(move |e| (i, e)))
        .collect();
    let cluster = |values: Vec<(usize, f32)>, outer_high: bool| -> Option<f32> {
        let mut bins: HashMap<i32, (Vec<usize>, Vec<f32>)> = HashMap::new();
        for (page, v) in values {
            let bin = bins.entry((v * 2.0).round() as i32).or_default();
            if !bin.0.contains(&page) {
                bin.0.push(page);
            }
            bin.1.push(v);
        }
        let (_, (on, best)) = bins
            .into_iter()
            .max_by_key(|(k, (on, _))| (on.len(), if outer_high { *k } else { -*k }))?;
        (on.len() >= 2 && on.len() * 4 >= pages.len())
            .then(|| best.iter().sum::<f32>() / best.len() as f32)
    };
    let side = |guide: Option<f32>, ends: Option<f32>, paper: f32| guide.or(ends).unwrap_or(paper);
    // A line cut by an edge heads into it: one cut at the right edge
    // says nothing about where the bottom edge is.
    let pick = |axis: usize, sign: f32, keep: &dyn Fn(f32) -> bool| -> Vec<(usize, f32)> {
        free.iter()
            .filter(|(_, (_, d))| d[axis] * sign > 0.3)
            .map(|&(i, (p, _))| (i, p[axis]))
            .filter(|&(_, v)| keep(v))
            .collect()
    };
    let left = side(
        guides
            .0
            .iter()
            .copied()
            .filter(|&x| x < size.x * 0.3)
            .reduce(f32::min),
        cluster(pick(0, -1.0, &|x| x < size.x * 0.25), false),
        0.0,
    );
    let right = side(
        guides
            .0
            .iter()
            .copied()
            .filter(|&x| x > size.x * 0.7)
            .reduce(f32::max),
        cluster(pick(0, 1.0, &|x| x > size.x * 0.75), true),
        size.x,
    );
    let bottom = side(
        guides
            .1
            .iter()
            .copied()
            .filter(|&y| y < size.y * 0.3)
            .reduce(f32::min),
        cluster(pick(1, -1.0, &|y| y < size.y * 0.25), false),
        0.0,
    );
    let top = side(
        guides
            .1
            .iter()
            .copied()
            .filter(|&y| y > size.y * 0.7)
            .reduce(f32::max),
        cluster(pick(1, 1.0, &|y| y > size.y * 0.75), true),
        size.y,
    );
    Rect::new(Vec2::new(left, bottom), Vec2::new(right, top))
}

/// Ends of open paths that no other path continues from: where drawn lines
/// were cut off. Each with the direction the line was heading there.
fn free_ends(drawing: &RawVectorDrawing) -> Vec<(Vec2, Vec2)> {
    let with_direction: Vec<(Vec2, Vec2)> = drawing
        .subpaths
        .iter()
        .filter(|p| !p.closed && !p.is_empty())
        .flat_map(|p| {
            [
                (p.start, outward_direction(p, false)),
                (p.end(), outward_direction(p, true)),
            ]
        })
        .collect();
    let ends: Vec<Vec2> = with_direction.iter().map(|e| e.0).collect();
    let mut sorted: Vec<(usize, Vec2)> = ends.iter().copied().enumerate().collect();
    sorted.sort_by(|a, b| a.1.x.total_cmp(&b.1.x));
    let mut joined = vec![false; ends.len()];
    for i in 0..sorted.len() {
        for j in i + 1..sorted.len() {
            if sorted[j].1.x - sorted[i].1.x > 0.05 {
                break;
            }
            if sorted[i].1.distance(sorted[j].1) <= 0.05 {
                joined[sorted[i].0] = true;
                joined[sorted[j].0] = true;
            }
        }
    }
    with_direction
        .into_iter()
        .zip(joined)
        .filter(|(_, j)| !j)
        .map(|(e, _)| e)
        .collect()
}

fn page_edges(drawing: &RawVectorDrawing, frame: Rect) -> Edges {
    let lines: Vec<Vec<Vec2>> = drawing.subpaths.iter().map(|p| p.polyline(0.2)).collect();
    let mut hits: HashMap<Side, Vec<f32>> = SIDES.iter().map(|&s| (s, Vec::new())).collect();
    for (p, _) in free_ends(drawing) {
        const ON: f32 = 0.3;
        if (p.x - frame.min.x).abs() < ON {
            hits.get_mut(&Side::Left).expect("side").push(p.y);
        }
        if (p.x - frame.max.x).abs() < ON {
            hits.get_mut(&Side::Right).expect("side").push(p.y);
        }
        if (p.y - frame.min.y).abs() < ON {
            hits.get_mut(&Side::Bottom).expect("side").push(p.x);
        }
        if (p.y - frame.max.y).abs() < ON {
            hits.get_mut(&Side::Top).expect("side").push(p.x);
        }
    }
    for list in hits.values_mut() {
        list.sort_by(f32::total_cmp);
    }

    let crossings_at = |vertical: bool, at: f32| crossings(&lines, vertical, at);
    let mut crossings = HashMap::new();
    for side in SIDES {
        let mut list = vec![hits[&side].clone()];
        for k in 1..OVERLAP_STEPS {
            let depth = k as f32 * OVERLAP_STEP_MM;
            list.push(match side {
                Side::Left => crossings_at(true, frame.min.x + depth),
                Side::Right => crossings_at(true, frame.max.x - depth),
                Side::Bottom => crossings_at(false, frame.min.y + depth),
                Side::Top => crossings_at(false, frame.max.y - depth),
            });
        }
        crossings.insert(side, list);
    }

    let mut labels: HashMap<Side, Vec<String>> = HashMap::new();
    for label in &drawing.labels {
        let p = label.position;
        let distances = [
            (Side::Left, (p.x - frame.min.x).abs()),
            (Side::Right, (p.x - frame.max.x).abs()),
            (Side::Bottom, (p.y - frame.min.y).abs()),
            (Side::Top, (p.y - frame.max.y).abs()),
        ];
        let (side, d) = distances
            .into_iter()
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("sides");
        if d < LABEL_BAND_MM {
            labels
                .entry(side)
                .or_default()
                .push(label.text.trim().to_owned());
        }
    }
    Edges {
        lines,
        hits,
        crossings,
        labels,
    }
}

/// Along-line positions where `lines` cross the vertical (or horizontal)
/// line at `at`, sorted.
fn crossings(lines: &[Vec<Vec2>], vertical: bool, at: f32) -> Vec<f32> {
    let mut out = Vec::new();
    for line in lines {
        for pair in line.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let (pa, pb, qa, qb) = if vertical {
                (a.x, b.x, a.y, b.y)
            } else {
                (a.y, b.y, a.x, b.x)
            };
            if (pa - at) * (pb - at) <= 0.0 && (pa - pb).abs() > 1e-6 {
                let t = (at - pa) / (pb - pa);
                out.push(qa + (qb - qa) * t);
            }
        }
    }
    out.sort_by(f32::total_cmp);
    out
}

/// Page numbers and tile names: "3", "A1", "Page 4", "2/12", "Seite 3 von 9".
fn is_page_label(text: &str) -> bool {
    let words: Vec<String> = text
        .split(|c: char| c.is_whitespace() || c == '/')
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect();
    let number = |w: &str| w.chars().all(|c| c.is_ascii_digit());
    let tile = |w: &str| {
        let (letters, digits): (String, String) = w.chars().partition(char::is_ascii_alphabetic);
        !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) && letters.len() <= 2
    };
    match words.as_slice() {
        [w] => number(w) || tile(w),
        [p, n] => {
            ["page", "seite", "p."].contains(&p.as_str()) && number(n) || number(p) && number(n)
        }
        [p, n, of, m] => {
            ["page", "seite"].contains(&p.as_str())
                && number(n)
                && ["of", "von"].contains(&of.as_str())
                && number(m)
        }
        _ => false,
    }
}

fn paper_name(size: Vec2) -> String {
    let (short, long) = (size.x.min(size.y), size.x.max(size.y));
    for (name, w, h) in [
        ("A4", 210.0, 297.0),
        ("US Letter", 215.9, 279.4),
        ("US Legal", 215.9, 355.6),
        ("A3", 297.0, 420.0),
        ("Tabloid", 279.4, 431.8),
    ] {
        if (short - w).abs() < 3.0 && (long - h).abs() < 3.0 {
            return name.to_owned();
        }
    }
    format!("{:.0} × {:.0} mm", size.x, size.y)
}

/// Closed shapes that only exist where pages meet (matching marks drawn
/// half on each page) are not pieces: switch them off.
pub(crate) fn disable_page_marks(pattern: &mut ImportedPattern, layout: &PageLayout) {
    for piece in &mut pattern.pieces {
        let Some(outer) = piece.loops.first() else {
            continue;
        };
        let bounds = Rect::new(outer.bounds.0, outer.bounds.1);
        if bounds.size().length() > 40.0 || piece.loops.len() > 1 {
            continue;
        }
        let on_seam = layout.seams.iter().any(|&(a, b)| {
            let seam = Rect::new(a, b);
            bounds.overlaps(&Rect::new(
                seam.min - Vec2::splat(0.5),
                seam.max + Vec2::splat(0.5),
            ))
        });
        if on_seam {
            piece.enabled = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(min: Vec2, size: f32) -> RawSubpath {
        let mut path = RawSubpath::new(min);
        for corner in [
            min + Vec2::new(size, 0.0),
            min + Vec2::splat(size),
            min + Vec2::new(0.0, size),
            min,
        ] {
            path.segments.push(RawSegment::Line { to: corner });
        }
        path.closed = true;
        path
    }

    #[test]
    fn a_closed_path_cut_by_the_edge_stays_one_piece() {
        // A square whose middle crosses x = 10: the part left of it is one
        // open path, start and end on the edge.
        let path = square(Vec2::new(0.0, 0.0), 20.0);
        let mut out = Vec::new();
        clip_subpath(
            &path,
            Rect::new(Vec2::new(-5.0, -5.0), Vec2::new(10.0, 30.0)),
            false,
            &mut out,
        );
        assert_eq!(out.len(), 1, "{out:?}");
        assert!((out[0].start.x - 10.0).abs() < 1e-4 && (out[0].end().x - 10.0).abs() < 1e-4);
    }

    #[test]
    fn a_curve_crossing_the_edge_is_cut_there() {
        let mut path = RawSubpath::new(Vec2::new(72.0, 151.0));
        path.segments.push(RawSegment::Line {
            to: Vec2::new(70.0, 140.0),
        });
        path.segments.push(RawSegment::Cubic {
            ctrl1: Vec2::new(90.0, 212.0),
            ctrl2: Vec2::new(96.0, 277.0),
            to: Vec2::new(95.0, 296.9),
        });
        let mut out = Vec::new();
        clip_subpath(
            &path,
            Rect::new(Vec2::splat(7.0), Vec2::new(203.0, 288.5)),
            true,
            &mut out,
        );
        assert_eq!(out.len(), 1, "{out:?}");
        assert!((out[0].end().y - 288.5).abs() < 0.01, "{:?}", out[0].end());
    }

    #[test]
    fn page_numbers_are_recognised() {
        for text in ["3", "A1", "b12", "Page 4", "2/12", "Seite 3 von 9"] {
            assert!(is_page_label(text), "{text}");
        }
        for text in ["Front", "Sleeve 2x", "Back Body"] {
            assert!(!is_page_label(text), "{text}");
        }
    }

    /// A 240 × 150 mm piece, its top slanted, with a sew line inside, printed on a
    /// 2 × 1 grid of 160 × 200 mm pages, each with a 5 mm margin and a
    /// frame drawn, the second page given first.
    fn tiled(overlap: f32) -> Vec<Page> {
        let size = Vec2::new(160.0, 200.0);
        let frame = Rect::new(Vec2::splat(5.0), size - 5.0);
        let step = frame.size().x - overlap;
        let cut = |mut p: RawSubpath, delta: Vec2, rect: Rect, out: &mut Vec<RawSubpath>| {
            translate(&mut p, delta);
            clip_subpath(&p, rect, false, out);
        };
        let mut pages = Vec::new();
        for column in [1, 0] {
            let delta = Vec2::new(30.0 - column as f32 * step, 20.0);
            let mut drawing = RawVectorDrawing::new();
            let visible = Rect::new(Vec2::ZERO, size);
            let mut outer = RawSubpath::new(Vec2::ZERO);
            for corner in [
                Vec2::new(240.0, 0.0),
                Vec2::new(240.0, 150.0),
                Vec2::new(0.0, 110.0),
                Vec2::ZERO,
            ] {
                outer.segments.push(RawSegment::Line { to: corner });
            }
            outer.closed = true;
            let mut inner = RawSubpath::new(Vec2::splat(10.0));
            for corner in [
                Vec2::new(230.0, 10.0),
                Vec2::new(230.0, 140.0),
                Vec2::new(10.0, 102.0),
                Vec2::splat(10.0),
            ] {
                inner.segments.push(RawSegment::Line { to: corner });
            }
            inner.closed = true;
            cut(outer, delta, visible, &mut drawing.subpaths);
            cut(inner, delta, visible, &mut drawing.subpaths);
            let mut border = RawSubpath::new(frame.min);
            for corner in [
                Vec2::new(frame.max.x, frame.min.y),
                frame.max,
                Vec2::new(frame.min.x, frame.max.y),
                frame.min,
            ] {
                border.segments.push(RawSegment::Line { to: corner });
            }
            drawing.subpaths.push(border);
            drawing.labels.push(RawLabel::new(
                format!("{}", column + 1),
                Vec2::new(80.0, 2.0),
            ));
            pages.push(Page {
                size: Some(size),
                drawing,
            });
        }
        pages
    }

    #[test]
    fn two_framed_pages_join_into_one_piece() {
        for overlap in [0.0, 12.0] {
            let (drawing, layout) = join(tiled(overlap));
            let layout = layout.expect("joined");
            assert_eq!((layout.columns, layout.rows, layout.joined), (2, 1, 2));
            assert!(
                (layout.overlap_mm.x - overlap).abs() < 0.6,
                "{overlap}: {layout:?}"
            );
            let pattern = drawing.assemble_pattern(10.0);
            let pieces: Vec<_> = pattern
                .pieces
                .iter()
                .filter(|p| p.loops.len() == 2)
                .collect();
            assert_eq!(pieces.len(), 1, "{overlap}: {pattern:#?}");
            let (lo, hi) = pieces[0].loops[0].bounds;
            assert!(
                ((hi - lo) - Vec2::new(240.0, 150.0)).abs().max_element() < 0.5,
                "{lo} {hi}"
            );
        }
    }
}

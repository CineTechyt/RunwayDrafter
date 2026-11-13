//! Pattern pieces: outlines plus the production annotations that travel with
//! them (seam allowance, notches, grainline, UV mapping, 3D arrangement).

use std::collections::BTreeMap;

use glam::{Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::geometry::{self, CubicBezier, Curve, CurveSegment, Line, QuadBezier};
use crate::ids::{FabricId, HandleKind, PieceId};
use crate::uv::UvMapping;

/// Default flattening tolerance in millimetres.
pub const DEFAULT_TOLERANCE_MM: f32 = 0.1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PointKind {
    /// Sharp corner; adjacent controls move independently.
    #[default]
    Corner,
    /// Tangent-continuous point; editors should keep adjacent controls
    /// collinear.
    Smooth,
}

/// An on-curve anchor point of an outline.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ControlPoint {
    pub position: Vec2,
    pub kind: PointKind,
}

impl ControlPoint {
    pub const fn corner(x: f32, y: f32) -> Self {
        Self { position: Vec2::new(x, y), kind: PointKind::Corner }
    }
}

/// How segment `i` travels from anchor `i` to anchor `i + 1`. Control points
/// are stored in absolute pattern coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Segment {
    Line,
    Quadratic { ctrl: Vec2 },
    Cubic { ctrl1: Vec2, ctrl2: Vec2 },
}

/// A closed outline. `points.len() == segments.len()` always holds.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Outline {
    points: Vec<ControlPoint>,
    segments: Vec<Segment>,
}

/// A flattened outline that remembers which pattern edge every vertex came
/// from: the polygon edge starting at `points[k]` belongs to edge `edge_of[k]`.
#[derive(Debug, Clone, Default)]
pub struct FlatOutline {
    pub points: Vec<Vec2>,
    pub edge_of: Vec<u32>,
}

impl Outline {
    pub fn new(points: Vec<ControlPoint>, segments: Vec<Segment>) -> Self {
        assert_eq!(points.len(), segments.len(), "a closed outline has one segment per anchor");
        Self { points, segments }
    }

    /// A closed polygon with straight edges.
    pub fn polygon(points: impl IntoIterator<Item = Vec2>) -> Self {
        let points: Vec<_> = points
            .into_iter()
            .map(|position| ControlPoint { position, kind: PointKind::Corner })
            .collect();
        let segments = vec![Segment::Line; points.len()];
        Self { points, segments }
    }

    pub fn edge_count(&self) -> u32 {
        self.points.len() as u32
    }

    pub fn points(&self) -> &[ControlPoint] {
        &self.points
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    pub fn set_segment(&mut self, edge: u32, segment: Segment) {
        self.segments[edge as usize] = segment;
    }

    fn next(&self, i: usize) -> usize {
        (i + 1) % self.points.len()
    }

    /// Resolved curve of edge `edge`.
    pub fn curve(&self, edge: u32) -> CurveSegment {
        let i = edge as usize;
        let a = self.points[i].position;
        let b = self.points[self.next(i)].position;
        match self.segments[i] {
            Segment::Line => CurveSegment::Line(Line { a, b }),
            Segment::Quadratic { ctrl } => CurveSegment::Quad(QuadBezier { p0: a, p1: ctrl, p2: b }),
            Segment::Cubic { ctrl1, ctrl2 } => {
                CurveSegment::Cubic(CubicBezier { p0: a, p1: ctrl1, p2: ctrl2, p3: b })
            }
        }
    }

    pub fn curves(&self) -> impl Iterator<Item = CurveSegment> + '_ {
        (0..self.edge_count()).map(|e| self.curve(e))
    }

    pub fn flatten(&self, tolerance: f32) -> FlatOutline {
        let mut flat = FlatOutline::default();
        for edge in 0..self.edge_count() {
            let curve = self.curve(edge);
            flat.points.push(curve.eval(0.0));
            curve.flatten_into(tolerance, &mut flat.points);
            flat.points.pop(); // the end point is the next edge's start
            flat.edge_of.resize(flat.points.len(), edge);
        }
        flat
    }

    /// Axis-aligned bounds `(min, max)` of the flattened outline.
    pub fn bounds(&self) -> (Vec2, Vec2) {
        let flat = self.flatten(1.0);
        flat.points.iter().fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(lo, hi), &p| (lo.min(p), hi.max(p)))
    }

    /// Signed area of the flattened outline; positive when counter-clockwise.
    pub fn signed_area(&self) -> f32 {
        let flat = self.flatten(1.0);
        geometry::signed_area(&flat.points)
    }

    /// Whether the outline runs counter-clockwise.
    pub fn is_ccw(&self) -> bool {
        self.signed_area() >= 0.0
    }

    /// Reverses the traversal direction of the outline.
    pub fn reverse(&self) -> Self {
        let n = self.points.len();
        if n < 2 {
            return self.clone();
        }
        let mut new_points = Vec::with_capacity(n);
        let mut new_segments = Vec::with_capacity(n);
        for k in 0..n {
            let orig_idx = (n - k) % n;
            new_points.push(self.points[orig_idx]);
        }
        for k in 0..n {
            let orig_edge = (n - k - 1) % n;
            let rev_seg = match self.segments[orig_edge] {
                Segment::Line => Segment::Line,
                Segment::Quadratic { ctrl } => Segment::Quadratic { ctrl },
                Segment::Cubic { ctrl1, ctrl2 } => Segment::Cubic { ctrl1: ctrl2, ctrl2: ctrl1 },
            };
            new_segments.push(rev_seg);
        }
        Self::new(new_points, new_segments)
    }

    /// Ensures the outline is oriented counter-clockwise.
    pub fn ensure_ccw(&mut self) {
        if !self.is_ccw() {
            *self = self.reverse();
        }
    }

    /// Offsets the outline. Positive width offsets outward; negative width offsets inward.
    pub fn offset(&self, width: impl Fn(u32) -> f32) -> Self {
        const MITRE_LIMIT: f32 = 4.0;
        let n = self.points.len();
        if n < 3 {
            return self.clone();
        }
        let sign = if self.is_ccw() { 1.0 } else { -1.0 };
        let normal = |i: usize| -> Vec2 {
            let next = (i + 1) % n;
            let d = (self.points[next].position - self.points[i].position).normalize_or_zero();
            Vec2::new(d.y, -d.x) * sign
        };

        let mut new_points = Vec::with_capacity(n);
        for i in 0..n {
            let prev = (i + n - 1) % n;
            let (n0, n1) = (normal(prev), normal(i));
            let (w0, w1) = (width(prev as u32), width(i as u32));
            let det = n0.perp_dot(n1);
            let offset = if det.abs() < 1e-4 {
                n1 * if w0.abs() > w1.abs() { w0 } else { w1 }
            } else {
                let q = Vec2::new(w0 * n1.y - w1 * n0.y, w1 * n0.x - w0 * n1.x) / det;
                let max_w = w0.abs().max(w1.abs());
                q.clamp_length_max(MITRE_LIMIT * max_w)
            };
            new_points.push(ControlPoint {
                position: self.points[i].position + offset,
                kind: self.points[i].kind,
            });
        }

        let mut new_segments = Vec::with_capacity(n);
        for i in 0..n {
            let w = width(i as u32);
            let n_i = normal(i);
            let seg = match self.segments[i] {
                Segment::Line => Segment::Line,
                Segment::Quadratic { ctrl } => Segment::Quadratic { ctrl: ctrl + n_i * w },
                Segment::Cubic { ctrl1, ctrl2 } => Segment::Cubic {
                    ctrl1: ctrl1 + n_i * w,
                    ctrl2: ctrl2 + n_i * w,
                },
            };
            new_segments.push(seg);
        }

        Self::new(new_points, new_segments)
    }

    /// Every draggable handle with its current position.
    pub fn handles(&self) -> impl Iterator<Item = (HandleKind, Vec2)> + '_ {
        let anchors = self
            .points
            .iter()
            .enumerate()
            .map(|(i, p)| (HandleKind::Anchor(i as u32), p.position));
        let controls = self.segments.iter().enumerate().flat_map(|(i, s)| {
            let segment = i as u32;
            let (a, b) = match *s {
                Segment::Line => (None, None),
                Segment::Quadratic { ctrl } => (Some(ctrl), None),
                Segment::Cubic { ctrl1, ctrl2 } => (Some(ctrl1), Some(ctrl2)),
            };
            a.map(|p| (HandleKind::Control { segment, slot: 0 }, p))
                .into_iter()
                .chain(b.map(|p| (HandleKind::Control { segment, slot: 1 }, p)))
        });
        anchors.chain(controls)
    }

    pub fn handle_position(&self, handle: HandleKind) -> Option<Vec2> {
        match handle {
            HandleKind::Anchor(i) => self.points.get(i as usize).map(|p| p.position),
            HandleKind::Control { segment, slot } => match (self.segments.get(segment as usize)?, slot) {
                (Segment::Quadratic { ctrl }, 0) => Some(*ctrl),
                (Segment::Cubic { ctrl1, .. }, 0) => Some(*ctrl1),
                (Segment::Cubic { ctrl2, .. }, 1) => Some(*ctrl2),
                _ => None,
            },
        }
    }

    /// The anchor a control handle is visually attached to.
    pub fn control_anchor(&self, segment: u32, slot: u8) -> Option<Vec2> {
        let i = segment as usize;
        match (self.segments.get(i)?, slot) {
            (Segment::Quadratic { .. } | Segment::Cubic { .. }, 0) => Some(self.points[i].position),
            (Segment::Cubic { .. }, 1) => Some(self.points[self.next(i)].position),
            _ => None,
        }
    }

    /// Moves a handle to `position`. Moving an anchor carries the Bézier
    /// controls attached to it along, so the local curve shape is preserved.
    /// Returns `false` if the handle does not exist.
    pub fn set_handle_position(&mut self, handle: HandleKind, position: Vec2) -> bool {
        match handle {
            HandleKind::Anchor(i) => {
                let i = i as usize;
                let Some(point) = self.points.get_mut(i) else { return false };
                let delta = position - point.position;
                point.position = position;
                let prev = (i + self.points.len() - 1) % self.points.len();
                // Outgoing segment: its first control hangs off this anchor.
                match &mut self.segments[i] {
                    Segment::Cubic { ctrl1, .. } => *ctrl1 += delta,
                    Segment::Quadratic { .. } | Segment::Line => {}
                }
                // Incoming segment: its last control hangs off this anchor.
                if let Segment::Cubic { ctrl2, .. } = &mut self.segments[prev] {
                    *ctrl2 += delta;
                }
                true
            }
            HandleKind::Control { segment, slot } => match (self.segments.get_mut(segment as usize), slot) {
                (Some(Segment::Quadratic { ctrl }), 0) => {
                    *ctrl = position;
                    true
                }
                (Some(Segment::Cubic { ctrl1, .. }), 0) => {
                    *ctrl1 = position;
                    true
                }
                (Some(Segment::Cubic { ctrl2, .. }), 1) => {
                    *ctrl2 = position;
                    true
                }
                _ => false,
            },
        }
    }

    /// Scales the outline by `factor` about `centre`.
    pub fn scale_about(&mut self, centre: Vec2, factor: f32) {
        let scale = |p: &mut Vec2| *p = centre + (*p - centre) * factor;
        for p in &mut self.points {
            scale(&mut p.position);
        }
        for s in &mut self.segments {
            match s {
                Segment::Line => {}
                Segment::Quadratic { ctrl } => scale(ctrl),
                Segment::Cubic { ctrl1, ctrl2 } => {
                    scale(ctrl1);
                    scale(ctrl2);
                }
            }
        }
    }

    pub fn translate(&mut self, delta: Vec2) {
        for p in &mut self.points {
            p.position += delta;
        }
        for s in &mut self.segments {
            match s {
                Segment::Line => {}
                Segment::Quadratic { ctrl } => *ctrl += delta,
                Segment::Cubic { ctrl1, ctrl2 } => {
                    *ctrl1 += delta;
                    *ctrl2 += delta;
                }
            }
        }
    }

    /// Mirrors the outline across a vertical axis and returns the old-to-new edge map.
    pub fn mirror_x_about(&mut self, axis_x: f32) -> Vec<u32> {
        let edge_count = self.edge_count();
        for point in &mut self.points {
            point.position.x = 2.0 * axis_x - point.position.x;
        }
        for segment in &mut self.segments {
            match segment {
                Segment::Line => {}
                Segment::Quadratic { ctrl } => ctrl.x = 2.0 * axis_x - ctrl.x,
                Segment::Cubic { ctrl1, ctrl2 } => {
                    ctrl1.x = 2.0 * axis_x - ctrl1.x;
                    ctrl2.x = 2.0 * axis_x - ctrl2.x;
                }
            }
        }
        if !self.is_ccw() {
            *self = self.reverse();
            (0..edge_count).map(|edge| edge_count - edge - 1).collect()
        } else {
            (0..edge_count).collect()
        }
    }
}

/// Seam allowance: extra fabric outside the sewing line, per edge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeamAllowance {
    pub enabled: bool,
    /// Width in mm for edges without an override.
    pub default_width: f32,
    /// Per-edge overrides in mm (e.g. a deeper hem).
    pub per_edge: BTreeMap<u32, f32>,
}

impl Default for SeamAllowance {
    fn default() -> Self {
        Self { enabled: true, default_width: 10.0, per_edge: BTreeMap::new() }
    }
}

impl SeamAllowance {
    pub fn width(&self, edge: u32) -> f32 {
        self.per_edge.get(&edge).copied().unwrap_or(self.default_width)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum NotchKind {
    /// A straight slit perpendicular to the edge.
    #[default]
    Slit,
    /// A V-shaped cut.
    V,
    /// A T-shaped mark.
    T,
}

/// An alignment mark on an edge.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Notch {
    pub edge: u32,
    /// Position along the edge's curve parameter, `0..=1`.
    pub t: f32,
    pub kind: NotchKind,
    /// Cut depth in mm, measured inwards from the cutting line.
    pub depth: f32,
}

/// The shape of a button.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ButtonStyle {
    /// Flat, slightly dished, sewn on through four holes (shirts, coats).
    #[default]
    FourHole,
    /// Like `FourHole` with two holes.
    TwoHole,
    /// A domed head on a loop underneath, no holes on top (blazers).
    Shank,
    /// A riveted metal tack button with a rolled rim (jeans).
    Tack,
}

impl ButtonStyle {
    pub const ALL: [ButtonStyle; 4] = [ButtonStyle::FourHole, ButtonStyle::TwoHole, ButtonStyle::Shank, ButtonStyle::Tack];

    pub fn label(self) -> &'static str {
        match self {
            ButtonStyle::FourHole => "4-hole",
            ButtonStyle::TwoHole => "2-hole",
            ButtonStyle::Shank => "Shank",
            ButtonStyle::Tack => "Jeans tack",
        }
    }

    /// Thread holes through the face of the button.
    pub fn holes(self) -> u32 {
        match self {
            ButtonStyle::FourHole => 4,
            ButtonStyle::TwoHole => 2,
            ButtonStyle::Shank | ButtonStyle::Tack => 0,
        }
    }
}

/// What a button is made of: how it reflects light.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ButtonFinish {
    #[default]
    Plastic,
    Metal,
    Wood,
    /// Mother of pearl: glossy, with a soft sheen.
    Pearl,
}

impl ButtonFinish {
    pub const ALL: [ButtonFinish; 4] = [ButtonFinish::Plastic, ButtonFinish::Metal, ButtonFinish::Wood, ButtonFinish::Pearl];

    pub fn label(self) -> &'static str {
        match self {
            ButtonFinish::Plastic => "Plastic",
            ButtonFinish::Metal => "Metal",
            ButtonFinish::Wood => "Wood",
            ButtonFinish::Pearl => "Mother of pearl",
        }
    }

    /// `(roughness, metallic)` of the surface.
    pub fn shading(self) -> (f32, f32) {
        match self {
            ButtonFinish::Plastic => (0.32, 0.0),
            ButtonFinish::Metal => (0.28, 1.0),
            ButtonFinish::Wood => (0.7, 0.0),
            ButtonFinish::Pearl => (0.18, 0.0),
        }
    }

    /// A typical colour of the material, linear RGB.
    pub fn default_color(self) -> [f32; 3] {
        match self {
            ButtonFinish::Plastic => [0.02, 0.02, 0.025],
            ButtonFinish::Metal => [0.80, 0.62, 0.32],
            ButtonFinish::Wood => [0.30, 0.15, 0.06],
            ButtonFinish::Pearl => [0.86, 0.84, 0.80],
        }
    }
}

/// A buttonhole, addressed document-wide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ButtonholeRef {
    pub piece: PieceId,
    /// Index into the piece's [`PatternPiece::buttonholes`].
    pub index: u32,
}

/// A button placed on a pattern piece, in pattern-space millimetres.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Button {
    pub position: Vec2,
    pub diameter: f32,
    #[serde(default)]
    pub style: ButtonStyle,
    #[serde(default)]
    pub finish: ButtonFinish,
    /// Linear RGB.
    #[serde(default = "Button::default_color")]
    pub color: [f32; 3],
    /// Linear RGB of the thread it is sewn on with.
    #[serde(default = "Button::default_thread")]
    pub thread_color: [f32; 3],
    /// The buttonhole it is done up through. A fastened button pulls its
    /// hole onto itself in the simulation, like a seam of one stitch.
    #[serde(default)]
    pub fastened_to: Option<ButtonholeRef>,
}

impl Button {
    pub fn new(position: Vec2, diameter: f32) -> Self {
        Self {
            position,
            diameter,
            style: ButtonStyle::default(),
            finish: ButtonFinish::default(),
            color: Self::default_color(),
            thread_color: Self::default_thread(),
            fastened_to: None,
        }
    }

    fn default_color() -> [f32; 3] {
        ButtonFinish::Plastic.default_color()
    }

    fn default_thread() -> [f32; 3] {
        [0.03, 0.03, 0.035]
    }

    /// Height of the button above the fabric in mm.
    pub fn thickness(&self) -> f32 {
        match self.style {
            ButtonStyle::FourHole | ButtonStyle::TwoHole => (self.diameter * 0.16).clamp(1.8, 5.0),
            ButtonStyle::Shank => (self.diameter * 0.35).clamp(3.0, 9.0),
            ButtonStyle::Tack => (self.diameter * 0.38).clamp(4.0, 9.0),
        }
    }
}

/// A buttonhole placed on a pattern piece, in pattern-space millimetres.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Buttonhole {
    pub position: Vec2,
    pub length: f32,
    /// Radians from the pattern's +X axis.
    pub angle: f32,
}

impl Buttonhole {
    /// A buttonhole sized for a button: the button's diameter plus its
    /// thickness, as tailors cut them.
    pub fn for_button(position: Vec2, button_diameter: f32) -> Self {
        Self { position, length: button_diameter + 3.0, angle: 0.0 }
    }
}

/// The direction of the warp threads, drawn as a double-headed arrow.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Grainline {
    /// Midpoint of the arrow in pattern space.
    pub origin: Vec2,
    /// Angle from +X in radians. π/2 means the warp runs along +Y.
    pub angle: f32,
    /// Drawn length in mm.
    pub length: f32,
}

impl Grainline {
    pub fn direction(&self) -> Vec2 {
        Vec2::from_angle(self.angle)
    }

    pub fn end_points(&self) -> (Vec2, Vec2) {
        let half = self.direction() * (self.length * 0.5);
        (self.origin - half, self.origin + half)
    }
}

/// Where a flat piece is placed around the avatar before draping.
///
/// A flat piece maps pattern point `p` to
/// `translation + rotation * ((p - anchor) / 1000, 0)` in world space
/// (metres, Y up). With `curvature` the piece is additionally wrapped
/// around a vertical cylinder behind it, so its side edges reach around the
/// body instead of sticking out past the arms.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Arrangement {
    /// The pattern-space point that lands on `translation`.
    pub anchor: Vec2,
    pub translation: Vec3,
    pub rotation: Quat,
    /// 1 / radius of the wrapping cylinder in 1/m; 0 keeps the piece flat.
    #[serde(default)]
    pub curvature: f32,
}

impl Arrangement {
    pub const MM_TO_M: f32 = 0.001;

    pub fn place(&self, p: Vec2) -> Vec3 {
        let d = (p - self.anchor) * Self::MM_TO_M;
        let local = if self.curvature.abs() > 1e-6 {
            // Arc length along the cylinder equals the pattern distance, so
            // wrapping does not stretch the piece.
            let angle = d.x * self.curvature;
            Vec3::new(angle.sin() / self.curvature, d.y, (angle.cos() - 1.0) / self.curvature)
        } else {
            d.extend(0.0)
        };
        self.translation + self.rotation * local
    }

    /// World-space normal at the centre of the piece.
    pub fn normal(&self) -> Vec3 {
        self.rotation * Vec3::Z
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatternPiece {
    pub id: PieceId,
    pub name: String,
    pub outline: Outline,
    pub seam_allowance: SeamAllowance,
    pub notches: Vec<Notch>,
    #[serde(default)]
    pub buttons: Vec<Button>,
    #[serde(default)]
    pub buttonholes: Vec<Buttonhole>,
    pub grainline: Grainline,
    pub uv: UvMapping,
    pub fabric: FabricId,
    pub arrangement: Arrangement,
    /// Which garment layer the piece belongs to, counted outwards from the
    /// body: 1 for a shirt, 2 for the jacket worn over it. Where pieces of
    /// different layers meet, the higher one is kept outside.
    #[serde(default = "PatternPiece::default_layer")]
    pub layer: u32,
    /// Where the piece goes around the body before sewing.
    #[serde(default)]
    pub placement: Placement,
}

/// Where a piece is put around the avatar before it is sewn, like the
/// arrangement points of other garment CAD tools.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Placement {
    /// Degrees around the body, seen from above: 0 = centre front,
    /// 90 = the avatar's left side, 180 = centre back, -90 = right side.
    pub angle: f32,
    /// Raises (positive) or lowers the piece from where it would go, in mm.
    pub height: f32,
}

impl PatternPiece {
    /// Creates a piece with sensible defaults derived from the outline: a
    /// vertical grainline through the centre and a UV mapping aligned to it.
    pub fn new(id: PieceId, name: impl Into<String>, outline: Outline, fabric: FabricId) -> Self {
        let (lo, hi) = outline.bounds();
        let centre = (lo + hi) * 0.5;
        let grainline = Grainline {
            origin: centre,
            angle: std::f32::consts::FRAC_PI_2,
            length: (hi.y - lo.y) * 0.5,
        };
        Self {
            id,
            name: name.into(),
            outline,
            seam_allowance: SeamAllowance::default(),
            notches: Vec::new(),
            buttons: Vec::new(),
            buttonholes: Vec::new(),
            grainline,
            uv: UvMapping::aligned_to_grain(&grainline),
            fabric,
            arrangement: Arrangement { anchor: centre, translation: Vec3::ZERO, rotation: Quat::IDENTITY, curvature: 0.0 },
            layer: Self::default_layer(),
            placement: Placement::default(),
        }
    }

    pub const fn default_layer() -> u32 {
        1
    }

    /// The cutting line: the outline offset outwards by the seam allowance.
    /// `None` when the allowance is disabled.
    pub fn seam_allowance_outline(&self, tolerance: f32) -> Option<Vec<Vec2>> {
        if !self.seam_allowance.enabled {
            return None;
        }
        let flat = self.outline.flatten(tolerance);
        Some(geometry::offset_polygon(&flat.points, |k| self.seam_allowance.width(flat.edge_of[k])))
    }

    /// Position and inward unit normal of a notch on the sewing line.
    pub fn notch_frame(&self, notch: &Notch) -> Option<(Vec2, Vec2)> {
        if notch.edge >= self.outline.edge_count() {
            return None;
        }
        let curve = self.outline.curve(notch.edge);
        let t = notch.t.clamp(0.0, 1.0);
        let tangent = curve.tangent(t).normalize_or_zero();
        // Inward is to the left of travel for a CCW outline.
        let ccw = geometry::signed_area(&self.outline.flatten(1.0).points) >= 0.0;
        let inward = if ccw { tangent.perp() } else { -tangent.perp() };
        Some((curve.eval(t), inward))
    }

    pub fn contains(&self, p: Vec2) -> bool {
        geometry::point_in_polygon(p, &self.outline.flatten(1.0).points)
    }

    /// Converts a piece whose outline was imported or drawn as a cutting line into a
    /// sewing line by shrinking the outline inwards by the seam allowance.
    pub fn convert_cutting_line_to_sew_line(&mut self) {
        if !self.seam_allowance.enabled {
            self.seam_allowance.enabled = true;
        }
        self.outline = self.outline.offset(|e| -self.seam_allowance.width(e));
    }

    /// Converts a piece whose outline is a sewing line into a cutting line
    /// by expanding the outline outwards by the seam allowance.
    pub fn convert_sew_line_to_cutting_line(&mut self) {
        self.outline = self.outline.offset(|e| self.seam_allowance.width(e));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rounded() -> Outline {
        let mut o = Outline::polygon([
            Vec2::new(0.0, 0.0),
            Vec2::new(100.0, 0.0),
            Vec2::new(100.0, 100.0),
            Vec2::new(0.0, 100.0),
        ]);
        o.set_segment(1, Segment::Cubic { ctrl1: Vec2::new(130.0, 30.0), ctrl2: Vec2::new(130.0, 70.0) });
        o
    }

    #[test]
    fn flatten_tracks_edges() {
        let flat = rounded().flatten(0.1);
        assert_eq!(flat.points.len(), flat.edge_of.len());
        assert_eq!(flat.edge_of[0], 0);
        assert_eq!(*flat.edge_of.last().unwrap(), 3);
        assert!(flat.edge_of.iter().filter(|&&e| e == 1).count() > 4);
        assert!(flat.edge_of.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn moving_an_anchor_carries_its_controls() {
        let mut o = rounded();
        let delta = Vec2::new(5.0, -3.0);
        assert!(o.set_handle_position(HandleKind::Anchor(1), Vec2::new(100.0, 0.0) + delta));
        assert_eq!(
            o.handle_position(HandleKind::Control { segment: 1, slot: 0 }),
            Some(Vec2::new(130.0, 30.0) + delta)
        );
        // The far control belongs to anchor 2 and stays put.
        assert_eq!(
            o.handle_position(HandleKind::Control { segment: 1, slot: 1 }),
            Some(Vec2::new(130.0, 70.0))
        );
        assert_eq!(o.handles().count(), 6);
    }

    #[test]
    fn outline_offset_and_reverse_work() {
        let sq = Outline::polygon([
            Vec2::new(0.0, 0.0),
            Vec2::new(100.0, 0.0),
            Vec2::new(100.0, 100.0),
            Vec2::new(0.0, 100.0),
        ]);
        assert!(sq.is_ccw());
        let inward = sq.offset(|_| -10.0);
        assert_eq!(inward.points()[0].position, Vec2::new(10.0, 10.0));
        assert_eq!(inward.points()[1].position, Vec2::new(90.0, 10.0));
        assert_eq!(inward.points()[2].position, Vec2::new(90.0, 90.0));
        assert_eq!(inward.points()[3].position, Vec2::new(10.0, 90.0));

        let outward = sq.offset(|_| 10.0);
        assert_eq!(outward.points()[0].position, Vec2::new(-10.0, -10.0));
        assert_eq!(outward.points()[1].position, Vec2::new(110.0, -10.0));

        let rev = sq.reverse();
        assert!(!rev.is_ccw());
        let mut fixed = rev.clone();
        fixed.ensure_ccw();
        assert!(fixed.is_ccw());
    }
}

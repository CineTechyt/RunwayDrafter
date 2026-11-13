//! SVG pattern reader: parses SVG vector paths, lines and splines.

use glam::Vec2;
use std::collections::HashMap;

use crate::IoError;
use crate::pattern_import::{ImportedPattern, RawLabel, RawSegment, RawSubpath, RawVectorDrawing};

/// Default CSS pixel to millimetre scale (96 DPI): 1 in = 96 px = 25.4 mm.
pub const CSS_PX_TO_MM: f32 = 25.4 / 96.0;

#[derive(Debug, Clone, PartialEq)]
pub struct SvgOptions {
    /// Scale factor to convert SVG coordinate units to millimetres.
    /// `None` auto-detects from SVG width/height and viewBox attributes (defaulting to 96 DPI).
    pub scale: Option<f32>,
    /// Whether to flip Y coordinates (convert SVG Y-down to CAD Y-up). Defaults to true.
    pub flip_y: bool,
    /// Default seam allowance width in mm.
    pub default_allowance_mm: f32,
}

impl Default for SvgOptions {
    fn default() -> Self {
        Self {
            scale: None,
            flip_y: true,
            default_allowance_mm: 10.0,
        }
    }
}

/// 2D Affine Transform for SVG groups and shapes.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Transform {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    pub const IDENTITY: Self = Self {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        e: 0.0,
        f: 0.0,
    };

    pub fn concat(&self, op: Self) -> Self {
        Self {
            a: self.a * op.a + self.c * op.b,
            b: self.b * op.a + self.d * op.b,
            c: self.a * op.c + self.c * op.d,
            d: self.b * op.c + self.d * op.d,
            e: self.a * op.e + self.c * op.f + self.e,
            f: self.b * op.e + self.d * op.f + self.f,
        }
    }

    pub fn transform(&self, p: Vec2) -> Vec2 {
        Vec2::new(
            self.a * p.x + self.c * p.y + self.e,
            self.b * p.x + self.d * p.y + self.f,
        )
    }
}

/// Reads vector paths from an SVG file and assembles pattern pieces.
pub fn import_pattern_svg(bytes: &[u8], options: &SvgOptions) -> Result<ImportedPattern, IoError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| IoError::malformed(format!("invalid UTF-8 in SVG: {e}")))?;

    let (doc_height, scale) = detect_dimensions_and_scale(text, options.scale);

    let mut drawing = RawVectorDrawing::new();
    let mut transform_stack = vec![Transform::IDENTITY];

    let mut parser = XmlScanner::new(text);
    while let Some(tag) = parser.next_tag() {
        match tag {
            Tag::Open { name, attrs } | Tag::SelfClosing { name, attrs } => {
                let elem_transform = attrs
                    .get("transform")
                    .map(|s| parse_transform(s))
                    .unwrap_or_default();
                let current_transform = transform_stack
                    .last()
                    .copied()
                    .unwrap_or_default()
                    .concat(elem_transform);
                let label = ["aria-label", "data-name", "title", "inkscape:label", "id"]
                    .iter()
                    .find_map(|key| attrs.get(*key))
                    .cloned();
                let first_subpath = drawing.subpaths.len();

                match name.as_str() {
                    "g" => {
                        transform_stack.push(current_transform);
                    }
                    "path" => {
                        if let Some(d) = attrs.get("d") {
                            parse_path_d(
                                d,
                                current_transform,
                                scale,
                                doc_height,
                                options.flip_y,
                                &mut drawing,
                            );
                        }
                    }
                    "line" => {
                        let x1 = parse_val(attrs.get("x1"));
                        let y1 = parse_val(attrs.get("y1"));
                        let x2 = parse_val(attrs.get("x2"));
                        let y2 = parse_val(attrs.get("y2"));
                        let p1 = map_point(
                            Vec2::new(x1, y1),
                            current_transform,
                            scale,
                            doc_height,
                            options.flip_y,
                        );
                        let p2 = map_point(
                            Vec2::new(x2, y2),
                            current_transform,
                            scale,
                            doc_height,
                            options.flip_y,
                        );
                        let mut sub = RawSubpath::new(p1);
                        sub.segments.push(RawSegment::Line { to: p2 });
                        drawing.subpaths.push(sub);
                    }
                    "polyline" | "polygon" => {
                        if let Some(points_str) = attrs.get("points") {
                            let pts = parse_points_list(points_str);
                            if pts.len() >= 2 {
                                let start = map_point(
                                    pts[0],
                                    current_transform,
                                    scale,
                                    doc_height,
                                    options.flip_y,
                                );
                                let mut sub = RawSubpath::new(start);
                                for pt in &pts[1..] {
                                    sub.segments.push(RawSegment::Line {
                                        to: map_point(
                                            *pt,
                                            current_transform,
                                            scale,
                                            doc_height,
                                            options.flip_y,
                                        ),
                                    });
                                }
                                if name == "polygon" {
                                    sub.segments.push(RawSegment::Line { to: start });
                                    sub.closed = true;
                                }
                                drawing.subpaths.push(sub);
                            }
                        }
                    }
                    "rect" => {
                        let x = parse_val(attrs.get("x"));
                        let y = parse_val(attrs.get("y"));
                        let w = parse_val(attrs.get("width"));
                        let h = parse_val(attrs.get("height"));
                        if w > 0.0 && h > 0.0 {
                            let p0 = map_point(
                                Vec2::new(x, y),
                                current_transform,
                                scale,
                                doc_height,
                                options.flip_y,
                            );
                            let p1 = map_point(
                                Vec2::new(x + w, y),
                                current_transform,
                                scale,
                                doc_height,
                                options.flip_y,
                            );
                            let p2 = map_point(
                                Vec2::new(x + w, y + h),
                                current_transform,
                                scale,
                                doc_height,
                                options.flip_y,
                            );
                            let p3 = map_point(
                                Vec2::new(x, y + h),
                                current_transform,
                                scale,
                                doc_height,
                                options.flip_y,
                            );
                            let mut sub = RawSubpath::new(p0);
                            sub.segments.push(RawSegment::Line { to: p1 });
                            sub.segments.push(RawSegment::Line { to: p2 });
                            sub.segments.push(RawSegment::Line { to: p3 });
                            sub.segments.push(RawSegment::Line { to: p0 });
                            sub.closed = true;
                            drawing.subpaths.push(sub);
                        }
                    }
                    "circle" | "ellipse" => {
                        let cx = parse_val(attrs.get("cx"));
                        let cy = parse_val(attrs.get("cy"));
                        let (rx, ry) = if name == "circle" {
                            let r = parse_val(attrs.get("r"));
                            (r, r)
                        } else {
                            (parse_val(attrs.get("rx")), parse_val(attrs.get("ry")))
                        };
                        if rx > 0.0 && ry > 0.0 {
                            add_ellipse(
                                cx,
                                cy,
                                rx,
                                ry,
                                current_transform,
                                scale,
                                doc_height,
                                options.flip_y,
                                &mut drawing,
                            );
                        }
                    }
                    _ => {}
                }

                if let Some(text) = label
                    && drawing.subpaths.len() > first_subpath
                    && let Some(position) = subpath_group_center(&drawing.subpaths[first_subpath..])
                {
                    drawing.labels.push(RawLabel { text, position });
                }
            }
            Tag::Close { name } => {
                if name == "g" && transform_stack.len() > 1 {
                    transform_stack.pop();
                }
            }
        }
    }

    if drawing.subpaths.is_empty() {
        return Err(IoError::malformed("no vector paths found in SVG"));
    }

    Ok(drawing.assemble_pattern(options.default_allowance_mm))
}

fn subpath_group_center(paths: &[RawSubpath]) -> Option<Vec2> {
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    for path in paths {
        min = min.min(path.start);
        max = max.max(path.start);
        for segment in &path.segments {
            let end = segment.end();
            min = min.min(end);
            max = max.max(end);
            match segment {
                RawSegment::Quadratic { ctrl, .. } => {
                    min = min.min(*ctrl);
                    max = max.max(*ctrl);
                }
                RawSegment::Cubic { ctrl1, ctrl2, .. } => {
                    min = min.min(*ctrl1).min(*ctrl2);
                    max = max.max(*ctrl1).max(*ctrl2);
                }
                RawSegment::Line { .. } => {}
            }
        }
    }
    min.is_finite().then_some((min + max) * 0.5)
}

fn map_point(p: Vec2, transform: Transform, scale: f32, doc_height: f32, flip_y: bool) -> Vec2 {
    let transformed = transform.transform(p);
    let y = if flip_y {
        doc_height - transformed.y
    } else {
        transformed.y
    };
    Vec2::new(transformed.x * scale, y * scale)
}

fn detect_dimensions_and_scale(xml: &str, user_scale: Option<f32>) -> (f32, f32) {
    if let Some(scale) = user_scale {
        return (0.0, scale);
    }
    // Scan root <svg> tag
    let mut scanner = XmlScanner::new(xml);
    while let Some(tag) = scanner.next_tag() {
        if let Tag::Open { name, attrs } | Tag::SelfClosing { name, attrs } = tag {
            if name == "svg" {
                let vb = attrs.get("viewBox").map(|v| parse_viewbox(v));
                let w_unit = attrs.get("width").map(|w| parse_length(w));
                let _h_unit = attrs.get("height").map(|h| parse_length(h));

                let vb_height = vb.map_or(0.0, |(_, _, _, h)| h);

                // If explicit unit is mm:
                if let Some((w_val, unit)) = w_unit {
                    match unit.as_str() {
                        "mm" => {
                            if let Some((_, _, vb_w, _)) = vb {
                                if vb_w > 0.0 {
                                    return (vb_height, w_val / vb_w);
                                }
                            }
                            return (vb_height, 1.0);
                        }
                        "cm" => {
                            let mm = w_val * 10.0;
                            if let Some((_, _, vb_w, _)) = vb {
                                if vb_w > 0.0 {
                                    return (vb_height, mm / vb_w);
                                }
                            }
                            return (vb_height, 10.0);
                        }
                        "in" => {
                            let mm = w_val * 25.4;
                            if let Some((_, _, vb_w, _)) = vb {
                                if vb_w > 0.0 {
                                    return (vb_height, mm / vb_w);
                                }
                            }
                            return (vb_height, 25.4);
                        }
                        "pt" => {
                            let mm = w_val * (25.4 / 72.0);
                            if let Some((_, _, vb_w, _)) = vb {
                                if vb_w > 0.0 {
                                    return (vb_height, mm / vb_w);
                                }
                            }
                            return (vb_height, 25.4 / 72.0);
                        }
                        _ => {}
                    }
                }

                return (vb_height, CSS_PX_TO_MM);
            }
        }
    }
    (0.0, CSS_PX_TO_MM)
}

fn parse_viewbox(s: &str) -> (f32, f32, f32, f32) {
    let nums = extract_floats(s);
    if nums.len() >= 4 {
        (nums[0], nums[1], nums[2], nums[3])
    } else {
        (0.0, 0.0, 0.0, 0.0)
    }
}

fn parse_length(s: &str) -> (f32, String) {
    let s = s.trim();
    let num_end = s
        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
        .unwrap_or(s.len());
    let val = s[..num_end].parse::<f32>().unwrap_or(0.0);
    let unit = s[num_end..].trim().to_ascii_lowercase();
    (val, unit)
}

fn parse_val(s: Option<&String>) -> f32 {
    s.and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0)
}

fn parse_points_list(s: &str) -> Vec<Vec2> {
    let nums = extract_floats(s);
    nums.chunks_exact(2)
        .map(|c| Vec2::new(c[0], c[1]))
        .collect()
}

fn parse_transform(s: &str) -> Transform {
    let mut cur = Transform::IDENTITY;
    let s = s.trim();
    let mut i = 0;
    let bytes = s.as_bytes();

    while i < bytes.len() {
        while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b',') {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let open_paren = match s[i..].find('(') {
            Some(p) => i + p,
            None => break,
        };
        let close_paren = match s[open_paren..].find(')') {
            Some(p) => open_paren + p,
            None => break,
        };
        let name = s[i..open_paren].trim();
        let args = extract_floats(&s[open_paren + 1..close_paren]);

        let op = match name {
            "matrix" if args.len() >= 6 => Transform {
                a: args[0],
                b: args[1],
                c: args[2],
                d: args[3],
                e: args[4],
                f: args[5],
            },
            "translate" => {
                let tx = args.first().copied().unwrap_or(0.0);
                let ty = args.get(1).copied().unwrap_or(0.0);
                Transform {
                    a: 1.0,
                    b: 0.0,
                    c: 0.0,
                    d: 1.0,
                    e: tx,
                    f: ty,
                }
            }
            "scale" => {
                let sx = args.first().copied().unwrap_or(1.0);
                let sy = args.get(1).copied().unwrap_or(sx);
                Transform {
                    a: sx,
                    b: 0.0,
                    c: 0.0,
                    d: sy,
                    e: 0.0,
                    f: 0.0,
                }
            }
            "rotate" => {
                let angle_rad = args.first().copied().unwrap_or(0.0).to_radians();
                let cos_a = angle_rad.cos();
                let sin_a = angle_rad.sin();
                if args.len() >= 3 {
                    let cx = args[1];
                    let cy = args[2];
                    let t1 = Transform {
                        a: 1.0,
                        b: 0.0,
                        c: 0.0,
                        d: 1.0,
                        e: cx,
                        f: cy,
                    };
                    let rot = Transform {
                        a: cos_a,
                        b: sin_a,
                        c: -sin_a,
                        d: cos_a,
                        e: 0.0,
                        f: 0.0,
                    };
                    let t2 = Transform {
                        a: 1.0,
                        b: 0.0,
                        c: 0.0,
                        d: 1.0,
                        e: -cx,
                        f: -cy,
                    };
                    t1.concat(rot).concat(t2)
                } else {
                    Transform {
                        a: cos_a,
                        b: sin_a,
                        c: -sin_a,
                        d: cos_a,
                        e: 0.0,
                        f: 0.0,
                    }
                }
            }
            _ => Transform::IDENTITY,
        };

        cur = cur.concat(op);
        i = close_paren + 1;
    }

    cur
}

fn add_ellipse(
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    transform: Transform,
    scale: f32,
    doc_height: f32,
    flip_y: bool,
    drawing: &mut RawVectorDrawing,
) {
    // 4 cubic beziers approximating an ellipse with kappa ~ 0.55228475
    const KAPPA: f32 = 0.55228475;
    let kx = rx * KAPPA;
    let ky = ry * KAPPA;

    let p = |x: f32, y: f32| map_point(Vec2::new(x, y), transform, scale, doc_height, flip_y);

    let mut sub = RawSubpath::new(p(cx + rx, cy));
    sub.segments.push(RawSegment::Cubic {
        ctrl1: p(cx + rx, cy + ky),
        ctrl2: p(cx + kx, cy + ry),
        to: p(cx, cy + ry),
    });
    sub.segments.push(RawSegment::Cubic {
        ctrl1: p(cx - kx, cy + ry),
        ctrl2: p(cx - rx, cy + ky),
        to: p(cx - rx, cy),
    });
    sub.segments.push(RawSegment::Cubic {
        ctrl1: p(cx - rx, cy - ky),
        ctrl2: p(cx - kx, cy - ry),
        to: p(cx, cy - ry),
    });
    sub.segments.push(RawSegment::Cubic {
        ctrl1: p(cx + kx, cy - ry),
        ctrl2: p(cx + rx, cy - ky),
        to: p(cx + rx, cy),
    });
    sub.closed = true;
    drawing.subpaths.push(sub);
}

/// Parses the SVG path `d` attribute.
fn parse_path_d(
    d: &str,
    transform: Transform,
    scale: f32,
    doc_height: f32,
    flip_y: bool,
    drawing: &mut RawVectorDrawing,
) {
    let mut current = Vec2::ZERO;
    let mut subpath_start = Vec2::ZERO;
    let mut last_ctrl: Option<Vec2> = None;
    let mut current_subpath: Option<RawSubpath> = None;

    let p = |v: Vec2| map_point(v, transform, scale, doc_height, flip_y);

    let mut lexer = PathLexer::new(d);
    while let Some(cmd) = lexer.next_command() {
        let is_rel = cmd.is_ascii_lowercase();
        let cmd_upper = cmd.to_ascii_uppercase();

        match cmd_upper {
            'M' => {
                let mut first = true;
                while let (Some(x), Some(y)) = (lexer.next_coord(), lexer.next_coord()) {
                    let target = if is_rel && !first {
                        current + Vec2::new(x, y)
                    } else if is_rel {
                        current + Vec2::new(x, y)
                    } else {
                        Vec2::new(x, y)
                    };
                    if first {
                        if let Some(sub) = current_subpath.take() {
                            if !sub.is_empty() {
                                drawing.subpaths.push(sub);
                            }
                        }
                        current = target;
                        subpath_start = current;
                        current_subpath = Some(RawSubpath::new(p(current)));
                        first = false;
                    } else {
                        // Subsequent coordinates after M/m are treated as Lineto
                        if let Some(sub) = &mut current_subpath {
                            sub.segments.push(RawSegment::Line { to: p(target) });
                        }
                        current = target;
                    }
                    last_ctrl = None;
                }
            }
            'L' => {
                while let (Some(x), Some(y)) = (lexer.next_coord(), lexer.next_coord()) {
                    let target = if is_rel {
                        current + Vec2::new(x, y)
                    } else {
                        Vec2::new(x, y)
                    };
                    if let Some(sub) = &mut current_subpath {
                        sub.segments.push(RawSegment::Line { to: p(target) });
                    }
                    current = target;
                    last_ctrl = None;
                }
            }
            'H' => {
                while let Some(x) = lexer.next_coord() {
                    let target = if is_rel {
                        Vec2::new(current.x + x, current.y)
                    } else {
                        Vec2::new(x, current.y)
                    };
                    if let Some(sub) = &mut current_subpath {
                        sub.segments.push(RawSegment::Line { to: p(target) });
                    }
                    current = target;
                    last_ctrl = None;
                }
            }
            'V' => {
                while let Some(y) = lexer.next_coord() {
                    let target = if is_rel {
                        Vec2::new(current.x, current.y + y)
                    } else {
                        Vec2::new(current.x, y)
                    };
                    if let Some(sub) = &mut current_subpath {
                        sub.segments.push(RawSegment::Line { to: p(target) });
                    }
                    current = target;
                    last_ctrl = None;
                }
            }
            'C' => {
                while let (Some(x1), Some(y1), Some(x2), Some(y2), Some(x), Some(y)) = (
                    lexer.next_coord(),
                    lexer.next_coord(),
                    lexer.next_coord(),
                    lexer.next_coord(),
                    lexer.next_coord(),
                    lexer.next_coord(),
                ) {
                    let c1 = if is_rel {
                        current + Vec2::new(x1, y1)
                    } else {
                        Vec2::new(x1, y1)
                    };
                    let c2 = if is_rel {
                        current + Vec2::new(x2, y2)
                    } else {
                        Vec2::new(x2, y2)
                    };
                    let to = if is_rel {
                        current + Vec2::new(x, y)
                    } else {
                        Vec2::new(x, y)
                    };
                    if let Some(sub) = &mut current_subpath {
                        sub.segments.push(RawSegment::Cubic {
                            ctrl1: p(c1),
                            ctrl2: p(c2),
                            to: p(to),
                        });
                    }
                    current = to;
                    last_ctrl = Some(c2);
                }
            }
            'S' => {
                while let (Some(x2), Some(y2), Some(x), Some(y)) = (
                    lexer.next_coord(),
                    lexer.next_coord(),
                    lexer.next_coord(),
                    lexer.next_coord(),
                ) {
                    let c1 = match last_ctrl {
                        Some(prev_c2) => 2.0 * current - prev_c2,
                        None => current,
                    };
                    let c2 = if is_rel {
                        current + Vec2::new(x2, y2)
                    } else {
                        Vec2::new(x2, y2)
                    };
                    let to = if is_rel {
                        current + Vec2::new(x, y)
                    } else {
                        Vec2::new(x, y)
                    };
                    if let Some(sub) = &mut current_subpath {
                        sub.segments.push(RawSegment::Cubic {
                            ctrl1: p(c1),
                            ctrl2: p(c2),
                            to: p(to),
                        });
                    }
                    current = to;
                    last_ctrl = Some(c2);
                }
            }
            'Q' => {
                while let (Some(x1), Some(y1), Some(x), Some(y)) = (
                    lexer.next_coord(),
                    lexer.next_coord(),
                    lexer.next_coord(),
                    lexer.next_coord(),
                ) {
                    let ctrl = if is_rel {
                        current + Vec2::new(x1, y1)
                    } else {
                        Vec2::new(x1, y1)
                    };
                    let to = if is_rel {
                        current + Vec2::new(x, y)
                    } else {
                        Vec2::new(x, y)
                    };
                    if let Some(sub) = &mut current_subpath {
                        sub.segments.push(RawSegment::Quadratic {
                            ctrl: p(ctrl),
                            to: p(to),
                        });
                    }
                    current = to;
                    last_ctrl = Some(ctrl);
                }
            }
            'T' => {
                while let (Some(x), Some(y)) = (lexer.next_coord(), lexer.next_coord()) {
                    let ctrl = match last_ctrl {
                        Some(prev_ctrl) => 2.0 * current - prev_ctrl,
                        None => current,
                    };
                    let to = if is_rel {
                        current + Vec2::new(x, y)
                    } else {
                        Vec2::new(x, y)
                    };
                    if let Some(sub) = &mut current_subpath {
                        sub.segments.push(RawSegment::Quadratic {
                            ctrl: p(ctrl),
                            to: p(to),
                        });
                    }
                    current = to;
                    last_ctrl = Some(ctrl);
                }
            }
            'A' => {
                while let (
                    Some(_rx),
                    Some(_ry),
                    Some(_rot),
                    Some(_large_arc),
                    Some(_sweep),
                    Some(x),
                    Some(y),
                ) = (
                    lexer.next_coord(),
                    lexer.next_coord(),
                    lexer.next_coord(),
                    lexer.next_coord(),
                    lexer.next_coord(),
                    lexer.next_coord(),
                    lexer.next_coord(),
                ) {
                    let to = if is_rel {
                        current + Vec2::new(x, y)
                    } else {
                        Vec2::new(x, y)
                    };
                    // Arc-to-lines approximation
                    let steps = 8;
                    for step in 1..=steps {
                        let t = step as f32 / steps as f32;
                        let mid = current.lerp(to, t);
                        if let Some(sub) = &mut current_subpath {
                            sub.segments.push(RawSegment::Line { to: p(mid) });
                        }
                    }
                    current = to;
                    last_ctrl = None;
                }
            }
            'Z' => {
                if let Some(sub) = &mut current_subpath {
                    if current.distance(subpath_start) > 1e-4 {
                        sub.segments.push(RawSegment::Line {
                            to: p(subpath_start),
                        });
                    }
                    sub.closed = true;
                }
                current = subpath_start;
                last_ctrl = None;
            }
            _ => {
                last_ctrl = None;
            }
        }
    }

    if let Some(sub) = current_subpath {
        if !sub.is_empty() {
            drawing.subpaths.push(sub);
        }
    }
}

/// Lexer for SVG path `d` strings.
struct PathLexer<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> PathLexer<'a> {
    pub fn new(s: &'a str) -> Self {
        Self {
            bytes: s.as_bytes(),
            pos: 0,
        }
    }

    fn skip_whitespace_and_commas(&mut self) {
        while self.pos < self.bytes.len()
            && (self.bytes[self.pos].is_ascii_whitespace() || self.bytes[self.pos] == b',')
        {
            self.pos += 1;
        }
    }

    pub fn next_command(&mut self) -> Option<char> {
        self.skip_whitespace_and_commas();
        if self.pos >= self.bytes.len() {
            return None;
        }
        let b = self.bytes[self.pos];
        if b.is_ascii_alphabetic() {
            self.pos += 1;
            Some(b as char)
        } else {
            None
        }
    }

    pub fn next_coord(&mut self) -> Option<f32> {
        self.skip_whitespace_and_commas();
        if self.pos >= self.bytes.len() {
            return None;
        }
        let b = self.bytes[self.pos];
        if b.is_ascii_alphabetic() {
            return None;
        }
        let start = self.pos;
        if self.bytes[self.pos] == b'-' || self.bytes[self.pos] == b'+' {
            self.pos += 1;
        }
        let mut has_digit = false;
        let mut has_dot = false;
        while self.pos < self.bytes.len() {
            let ch = self.bytes[self.pos];
            if ch.is_ascii_digit() {
                has_digit = true;
                self.pos += 1;
            } else if ch == b'.' && !has_dot {
                has_dot = true;
                self.pos += 1;
            } else if (ch == b'e' || ch == b'E') && has_digit {
                self.pos += 1;
                if self.pos < self.bytes.len()
                    && (self.bytes[self.pos] == b'-' || self.bytes[self.pos] == b'+')
                {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
        if !has_digit {
            self.pos = start;
            return None;
        }
        std::str::from_utf8(&self.bytes[start..self.pos])
            .ok()?
            .parse::<f32>()
            .ok()
    }
}

fn extract_floats(s: &str) -> Vec<f32> {
    let mut nums = Vec::new();
    let mut lexer = PathLexer::new(s);
    while let Some(n) = lexer.next_coord() {
        nums.push(n);
    }
    nums
}

enum Tag {
    Open {
        name: String,
        attrs: HashMap<String, String>,
    },
    Close {
        name: String,
    },
    SelfClosing {
        name: String,
        attrs: HashMap<String, String>,
    },
}

struct XmlScanner<'a> {
    s: &'a str,
    pos: usize,
}

impl<'a> XmlScanner<'a> {
    pub fn new(s: &'a str) -> Self {
        Self { s, pos: 0 }
    }

    pub fn next_tag(&mut self) -> Option<Tag> {
        while self.pos < self.s.len() {
            let start = self.s[self.pos..].find('<')? + self.pos;
            if start + 4 <= self.s.len() && &self.s[start..start + 4] == "<!--" {
                // Comment
                if let Some(end) = self.s[start..].find("-->") {
                    self.pos = start + end + 3;
                    continue;
                }
                return None;
            }
            let end = self.s[start..].find('>')? + start;
            self.pos = end + 1;
            let tag_content = self.s[start + 1..end].trim();
            if tag_content.starts_with('?') || tag_content.starts_with('!') {
                continue;
            }
            if let Some(close_name) = tag_content.strip_prefix('/') {
                return Some(Tag::Close {
                    name: close_name.trim().to_ascii_lowercase(),
                });
            }
            let self_closing = tag_content.ends_with('/');
            let clean = if self_closing {
                tag_content[..tag_content.len() - 1].trim()
            } else {
                tag_content
            };
            let mut parts = clean.splitn(2, |c: char| c.is_ascii_whitespace());
            let name = parts.next().unwrap_or("").to_ascii_lowercase();
            let attrs_str = parts.next().unwrap_or("");
            let attrs = parse_attributes(attrs_str);
            if self_closing {
                return Some(Tag::SelfClosing { name, attrs });
            }
            return Some(Tag::Open { name, attrs });
        }
        None
    }
}

fn parse_attributes(s: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let key_start = i;
        while i < bytes.len() && bytes[i] != b'=' && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let key = s[key_start..i].trim();
        while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b'=') {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let quote = bytes[i];
        if quote == b'"' || quote == b'\'' {
            i += 1;
            let val_start = i;
            while i < bytes.len() && bytes[i] != quote {
                i += 1;
            }
            let val = &s[val_start..i];
            if i < bytes.len() {
                i += 1;
            }
            map.insert(key.to_owned(), val.to_owned());
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_svg_path_with_cubic_spline() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
            <path d="M 10 10 L 90 10 C 100 30 100 70 90 90 L 10 90 Z" />
        </svg>"#;
        let options = SvgOptions {
            scale: Some(1.0),
            flip_y: false,
            default_allowance_mm: 10.0,
        };
        let pattern = import_pattern_svg(svg.as_bytes(), &options).unwrap();
        assert_eq!(pattern.pieces.len(), 1);
        let piece = &pattern.pieces[0];
        let outline = &piece.loops[0].outline;
        assert_eq!(outline.edge_count(), 4);
    }

    #[test]
    fn parse_svg_concentric_cut_and_sew() {
        let svg = r#"<svg viewBox="0 0 300 300">
            <!-- Cut line -->
            <path d="M 0 0 L 120 0 L 120 120 L 0 120 Z" />
            <!-- Sew line -->
            <path d="M 10 10 L 110 10 L 110 110 L 10 110 Z" />
        </svg>"#;
        let options = SvgOptions {
            scale: Some(1.0),
            flip_y: false,
            default_allowance_mm: 10.0,
        };
        let pattern = import_pattern_svg(svg.as_bytes(), &options).unwrap();
        assert_eq!(pattern.pieces.len(), 1);
        let piece = &pattern.pieces[0];
        assert_eq!(piece.loops.len(), 2);
        assert_eq!(
            piece.cut_loop().unwrap().role,
            crate::pattern_import::LineRole::CuttingLine
        );
        assert_eq!(
            piece.sew_loop().unwrap().role,
            crate::pattern_import::LineRole::SeamAllowance
        );
    }

    #[test]
    fn svg_piece_uses_its_source_label() {
        let svg = r#"<svg viewBox="0 0 200 200">
            <path id="Front_Panel" d="M 0 0 L 100 0 L 100 100 L 0 100 Z" />
        </svg>"#;
        let options = SvgOptions {
            scale: Some(1.0),
            flip_y: false,
            default_allowance_mm: 10.0,
        };
        let pattern = import_pattern_svg(svg.as_bytes(), &options).unwrap();
        assert_eq!(pattern.pieces[0].name, "Front Panel");
    }
}

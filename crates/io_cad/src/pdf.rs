//! PDF vector pattern reader: extracts lines, cubic splines and rectangles
//! from PDF graphics streams, regardless of authoring software (Inkscape,
//! Illustrator, Valentina, Seamly2D, Gerber, Lectra, etc.).

use glam::Vec2;

use crate::IoError;
use crate::pattern_import::{ImportedPattern, RawLabel, RawSegment, RawSubpath, RawVectorDrawing};

/// Points to millimetres conversion factor: 1 inch = 72 pt = 25.4 mm.
pub const PT_TO_MM: f32 = 25.4 / 72.0;

#[derive(Debug, Clone, PartialEq)]
pub struct PdfOptions {
    /// Scale factor from PDF user space to millimetres. Defaults to `PT_TO_MM`.
    pub scale: f32,
    /// Default seam allowance width in mm for pieces without an inner loop.
    pub default_allowance_mm: f32,
}

impl Default for PdfOptions {
    fn default() -> Self {
        Self {
            scale: PT_TO_MM,
            default_allowance_mm: 10.0,
        }
    }
}

/// 2D Affine Transformation: [a, b, c, d, e, f].
/// Point (x, y) maps to (a*x + c*y + e, b*x + d*y + f).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Affine2D {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
}

impl Default for Affine2D {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Affine2D {
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

    pub fn transform(&self, x: f32, y: f32) -> Vec2 {
        Vec2::new(
            self.a * x + self.c * y + self.e,
            self.b * x + self.d * y + self.f,
        )
    }
}

/// Reads vector paths from a PDF file and assembles pattern pieces.
pub fn import_pattern_pdf(bytes: &[u8], options: &PdfOptions) -> Result<ImportedPattern, IoError> {
    if !bytes.starts_with(b"%PDF-") && !bytes.windows(5).any(|w| w == b"%PDF-") {
        return Err(IoError::malformed("not a valid PDF document (missing %PDF- header)"));
    }

    let streams = extract_content_streams(bytes);
    if streams.is_empty() {
        return Err(IoError::malformed("no vector graphics streams found in PDF"));
    }

    let mut drawing = RawVectorDrawing::new();
    for stream in &streams {
        parse_content_stream(stream, options.scale, &mut drawing);
        drawing.labels.extend(extract_text_labels(stream, options.scale));
    }

    if drawing.subpaths.is_empty() {
        return Err(IoError::malformed("no vector paths or splines found in PDF content"));
    }

    Ok(drawing.assemble_pattern(options.default_allowance_mm))
}

fn extract_text_labels(stream: &[u8], scale: f32) -> Vec<RawLabel> {
    let text = String::from_utf8_lossy(stream);
    let mut tokens = Tokenizer::new(&text);
    let mut args = Vec::new();
    let mut pending_text = Vec::new();
    let mut position = Vec2::ZERO;
    let mut labels = Vec::new();

    while let Some(token) = tokens.next() {
        if let Ok(number) = token.parse::<f32>() {
            args.push(number);
            continue;
        }
        if token.starts_with('(') || (token.starts_with('<') && token != "<<") {
            pending_text.push(decode_pdf_text(token));
            continue;
        }

        match token {
            "Tm" if args.len() >= 6 => {
                position = Vec2::new(args[args.len() - 2], args[args.len() - 1]) * scale;
            }
            "Td" | "TD" if args.len() >= 2 => {
                position += Vec2::new(args[args.len() - 2], args[args.len() - 1]) * scale;
            }
            "Tj" | "'" | "\"" | "TJ" => {
                let value = pending_text.join("");
                let value = value.trim();
                if !value.is_empty() && value.chars().any(char::is_alphabetic) {
                    labels.push(RawLabel {
                        text: value.to_owned(),
                        position,
                    });
                }
                pending_text.clear();
            }
            _ => {}
        }
        args.clear();
    }
    labels
}

fn decode_pdf_text(token: &str) -> String {
    if token.starts_with('(') && token.ends_with(')') {
        let bytes = token.as_bytes();
        let mut decoded = Vec::with_capacity(bytes.len());
        let mut index = 1;
        while index + 1 < bytes.len() {
            if bytes[index] != b'\\' {
                decoded.push(bytes[index]);
                index += 1;
                continue;
            }
            index += 1;
            if index + 1 >= bytes.len() {
                break;
            }
            match bytes[index] {
                b'n' => decoded.push(b'\n'),
                b'r' => decoded.push(b'\r'),
                b't' => decoded.push(b'\t'),
                b'b' => decoded.push(8),
                b'f' => decoded.push(12),
                b'\n' => {}
                b'\r' => {
                    if bytes.get(index + 1) == Some(&b'\n') {
                        index += 1;
                    }
                }
                b'0'..=b'7' => {
                    let mut value = 0u16;
                    let mut digits = 0;
                    while digits < 3 && index + 1 < bytes.len() && (b'0'..=b'7').contains(&bytes[index]) {
                        value = value * 8 + u16::from(bytes[index] - b'0');
                        index += 1;
                        digits += 1;
                    }
                    decoded.push(value as u8);
                    continue;
                }
                escaped => decoded.push(escaped),
            }
            index += 1;
        }
        String::from_utf8_lossy(&decoded).into_owned()
    } else {
        let hex = token.trim_start_matches('<').trim_end_matches('>');
        let mut bytes = Vec::new();
        let mut chars = hex.chars().filter(|ch| !ch.is_whitespace());
        while let Some(high) = chars.next() {
            let low = chars.next().unwrap_or('0');
            if let (Some(high), Some(low)) = (high.to_digit(16), low.to_digit(16)) {
                bytes.push(((high << 4) | low) as u8);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Extracts all decompressed content streams from a PDF file.
fn extract_content_streams(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut streams = Vec::new();
    let mut cursor = 0;

    while cursor < bytes.len() {
        // Look for "stream" keyword.
        let Some(stream_rel) = find_subsequence(&bytes[cursor..], b"stream") else {
            break;
        };
        let stream_pos = cursor + stream_rel;
        let mut data_start = stream_pos + 6;
        if data_start < bytes.len() && bytes[data_start] == b'\r' {
            data_start += 1;
        }
        if data_start < bytes.len() && bytes[data_start] == b'\n' {
            data_start += 1;
        }

        // Look for preceding dictionary to check for /Filter /FlateDecode.
        let header_start = stream_pos.saturating_sub(400);
        let header_bytes = &bytes[header_start..stream_pos];
        let is_flate = contains_flate_filter(header_bytes);

        // Find "endstream"
        let Some(end_rel) = find_subsequence(&bytes[data_start..], b"endstream") else {
            cursor = data_start;
            continue;
        };
        let mut data_end = data_start + end_rel;
        // Trim trailing newline before endstream if present
        while data_end > data_start && (bytes[data_end - 1] == b'\r' || bytes[data_end - 1] == b'\n') {
            data_end -= 1;
        }

        let raw_stream = &bytes[data_start..data_end];
        cursor = data_start + end_rel + 9;

        if is_flate {
            if let Ok(decompressed) = miniz_oxide::inflate::decompress_to_vec_zlib(raw_stream) {
                streams.push(decompressed);
            } else if let Ok(decompressed) = miniz_oxide::inflate::decompress_to_vec(raw_stream) {
                streams.push(decompressed);
            }
        } else {
            // Uncompressed stream: check if it contains vector commands
            if contains_vector_commands(raw_stream) {
                streams.push(raw_stream.to_vec());
            }
        }
    }

    streams
}

fn contains_flate_filter(header: &[u8]) -> bool {
    let s = String::from_utf8_lossy(header);
    s.contains("/FlateDecode") || s.contains("/Fl")
}

fn contains_vector_commands(stream: &[u8]) -> bool {
    let s = String::from_utf8_lossy(stream);
    s.contains(" m\n") || s.contains(" m ") || s.contains(" l\n") || s.contains(" c\n") || s.contains(" re\n")
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Executes PDF content stream graphics commands and collects paths.
fn parse_content_stream(stream: &[u8], scale: f32, drawing: &mut RawVectorDrawing) {
    let text = String::from_utf8_lossy(stream);
    let mut tokens = Tokenizer::new(&text);

    let mut ctm = Affine2D::IDENTITY;
    let mut ctm_stack: Vec<Affine2D> = Vec::new();
    let mut args: Vec<f32> = Vec::with_capacity(8);

    let mut current_point: Option<Vec2> = None;
    let mut current_subpath: Option<RawSubpath> = None;
    let mut pending_subpaths: Vec<RawSubpath> = Vec::new();

    while let Some(tok) = tokens.next() {
        if let Ok(num) = tok.parse::<f32>() {
            args.push(num);
            continue;
        }

        match tok {
            // Concatenate matrix: a b c d e f cm
            "cm" => {
                if args.len() >= 6 {
                    let op = Affine2D {
                        a: args[args.len() - 6],
                        b: args[args.len() - 5],
                        c: args[args.len() - 4],
                        d: args[args.len() - 3],
                        e: args[args.len() - 2],
                        f: args[args.len() - 1],
                    };
                    ctm = ctm.concat(op);
                }
                args.clear();
            }
            // Push graphics state
            "q" => {
                ctm_stack.push(ctm);
                args.clear();
            }
            // Pop graphics state
            "Q" => {
                if let Some(saved) = ctm_stack.pop() {
                    ctm = saved;
                }
                args.clear();
            }
            // Moveto: x y m
            "m" => {
                if args.len() >= 2 {
                    if let Some(sub) = current_subpath.take() {
                        if !sub.is_empty() {
                            pending_subpaths.push(sub);
                        }
                    }
                    let x = args[args.len() - 2];
                    let y = args[args.len() - 1];
                    let pt = ctm.transform(x, y) * scale;
                    current_point = Some(pt);
                    current_subpath = Some(RawSubpath::new(pt));
                }
                args.clear();
            }
            // Lineto: x y l
            "l" => {
                if args.len() >= 2 {
                    let x = args[args.len() - 2];
                    let y = args[args.len() - 1];
                    let pt = ctm.transform(x, y) * scale;
                    if let Some(sub) = &mut current_subpath {
                        sub.segments.push(RawSegment::Line { to: pt });
                        current_point = Some(pt);
                    }
                }
                args.clear();
            }
            // Curveto (cubic Bézier spline): x1 y1 x2 y2 x3 y3 c
            "c" => {
                if args.len() >= 6 {
                    let c1 = ctm.transform(args[args.len() - 6], args[args.len() - 5]) * scale;
                    let c2 = ctm.transform(args[args.len() - 4], args[args.len() - 3]) * scale;
                    let p3 = ctm.transform(args[args.len() - 2], args[args.len() - 1]) * scale;
                    if let Some(sub) = &mut current_subpath {
                        sub.segments.push(RawSegment::Cubic { ctrl1: c1, ctrl2: c2, to: p3 });
                        current_point = Some(p3);
                    }
                }
                args.clear();
            }
            // Curveto (first control point coincident with current point): x2 y2 x3 y3 v
            "v" => {
                if args.len() >= 4 {
                    let c1 = current_point.unwrap_or(Vec2::ZERO);
                    let c2 = ctm.transform(args[args.len() - 4], args[args.len() - 3]) * scale;
                    let p3 = ctm.transform(args[args.len() - 2], args[args.len() - 1]) * scale;
                    if let Some(sub) = &mut current_subpath {
                        sub.segments.push(RawSegment::Cubic { ctrl1: c1, ctrl2: c2, to: p3 });
                        current_point = Some(p3);
                    }
                }
                args.clear();
            }
            // Curveto (second control point coincident with end point): x1 y1 x3 y3 y
            "y" => {
                if args.len() >= 4 {
                    let c1 = ctm.transform(args[args.len() - 4], args[args.len() - 3]) * scale;
                    let p3 = ctm.transform(args[args.len() - 2], args[args.len() - 1]) * scale;
                    let c2 = p3;
                    if let Some(sub) = &mut current_subpath {
                        sub.segments.push(RawSegment::Cubic { ctrl1: c1, ctrl2: c2, to: p3 });
                        current_point = Some(p3);
                    }
                }
                args.clear();
            }
            // Rectangle: x y w h re
            "re" => {
                if args.len() >= 4 {
                    if let Some(sub) = current_subpath.take() {
                        if !sub.is_empty() {
                            pending_subpaths.push(sub);
                        }
                    }
                    let x = args[args.len() - 4];
                    let y = args[args.len() - 3];
                    let w = args[args.len() - 2];
                    let h = args[args.len() - 1];

                    let p0 = ctm.transform(x, y) * scale;
                    let p1 = ctm.transform(x + w, y) * scale;
                    let p2 = ctm.transform(x + w, y + h) * scale;
                    let p3 = ctm.transform(x, y + h) * scale;

                    let mut rect_sub = RawSubpath::new(p0);
                    rect_sub.segments.push(RawSegment::Line { to: p1 });
                    rect_sub.segments.push(RawSegment::Line { to: p2 });
                    rect_sub.segments.push(RawSegment::Line { to: p3 });
                    rect_sub.segments.push(RawSegment::Line { to: p0 });
                    rect_sub.closed = true;
                    pending_subpaths.push(rect_sub);
                    current_point = Some(p0);
                }
                args.clear();
            }
            // Close subpath: h (appends a straight line back to start point if needed)
            "h" => {
                if let Some(sub) = &mut current_subpath {
                    let start = sub.start;
                    if let Some(cur) = current_point {
                        if cur.distance(start) > 1e-4 {
                            sub.segments.push(RawSegment::Line { to: start });
                            current_point = Some(start);
                        }
                    }
                    sub.closed = true;
                }
                args.clear();
            }
            // Stroke / Fill / Terminate path: S, s, f, F, f*, B, B*, b, b*, n
            "S" | "f" | "F" | "f*" | "B" | "B*" | "n" => {
                if let Some(sub) = current_subpath.take() {
                    if !sub.is_empty() {
                        pending_subpaths.push(sub);
                    }
                }
                drawing.subpaths.append(&mut pending_subpaths);
                args.clear();
            }
            "s" | "b" | "b*" => {
                if let Some(mut sub) = current_subpath.take() {
                    let start = sub.start;
                    if let Some(cur) = current_point {
                        if cur.distance(start) > 1e-4 {
                            sub.segments.push(RawSegment::Line { to: start });
                            current_point = Some(start);
                        }
                    }
                    sub.closed = true;
                    if !sub.is_empty() {
                        pending_subpaths.push(sub);
                    }
                }
                drawing.subpaths.append(&mut pending_subpaths);
                args.clear();
            }
            // Any other command (colors, text, linewidth)
            _ => {
                args.clear();
            }
        }
    }

    if let Some(sub) = current_subpath {
        if !sub.is_empty() {
            pending_subpaths.push(sub);
        }
    }
    drawing.subpaths.append(&mut pending_subpaths);
}

/// Simple whitespace-and-delimiter tokenizer for PDF content streams.
struct Tokenizer<'a> {
    chars: &'a str,
    pos: usize,
}

impl<'a> Tokenizer<'a> {
    pub fn new(chars: &'a str) -> Self {
        Self { chars, pos: 0 }
    }

    pub fn next(&mut self) -> Option<&'a str> {
        let bytes = self.chars.as_bytes();
        // Skip whitespace and comments
        while self.pos < bytes.len() {
            let b = bytes[self.pos];
            if b.is_ascii_whitespace() {
                self.pos += 1;
            } else if b == b'%' {
                // Comment until end of line
                while self.pos < bytes.len() && bytes[self.pos] != b'\n' && bytes[self.pos] != b'\r' {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }

        if self.pos >= bytes.len() {
            return None;
        }

        let start = self.pos;
        let b = bytes[start];
        if b == b'(' {
            // String literal: skip until matching ')'
            let mut depth = 1;
            self.pos += 1;
            while self.pos < bytes.len() && depth > 0 {
                if bytes[self.pos] == b'\\' {
                    self.pos += 2;
                } else if bytes[self.pos] == b'(' {
                    depth += 1;
                    self.pos += 1;
                } else if bytes[self.pos] == b')' {
                    depth -= 1;
                    self.pos += 1;
                } else {
                    self.pos += 1;
                }
            }
            return Some(&self.chars[start..self.pos]);
        }

        if b == b'<' {
            // Hex string or dictionary
            self.pos += 1;
            if self.pos < bytes.len() && bytes[self.pos] == b'<' {
                self.pos += 1;
                return Some("<<");
            }
            while self.pos < bytes.len() && bytes[self.pos] != b'>' {
                self.pos += 1;
            }
            if self.pos < bytes.len() {
                self.pos += 1;
            }
            return Some(&self.chars[start..self.pos]);
        }

        if b == b'>' {
            self.pos += 1;
            if self.pos < bytes.len() && bytes[self.pos] == b'>' {
                self.pos += 1;
                return Some(">>");
            }
            return Some(">");
        }

        if b == b'[' || b == b']' {
            self.pos += 1;
            return Some(&self.chars[start..self.pos]);
        }

        // Regular token
        while self.pos < bytes.len() {
            let ch = bytes[self.pos];
            if ch.is_ascii_whitespace() || ch == b'(' || ch == b')' || ch == b'<' || ch == b'>' || ch == b'[' || ch == b']' || ch == b'%' {
                break;
            }
            self.pos += 1;
        }

        Some(&self.chars[start..self.pos])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_stream_with_splines_and_lines() {
        // A PDF content stream with moveto, lineto, and cubic Bézier splines (c operator)
        let stream = b"
            100 100 m
            300 100 l
            300 150 250 250 200 300 c
            150 250 100 150 100 100 c
            h S
        ";
        let mut drawing = RawVectorDrawing::new();
        parse_content_stream(stream, 1.0, &mut drawing);
        assert_eq!(drawing.subpaths.len(), 1);
        let sub = &drawing.subpaths[0];
        assert!(sub.closed);
        assert_eq!(sub.segments.len(), 3);
        assert!(matches!(sub.segments[0], RawSegment::Line { .. }));
        assert!(matches!(sub.segments[1], RawSegment::Cubic { .. }));
        assert!(matches!(sub.segments[2], RawSegment::Cubic { .. }));
    }

    #[test]
    fn parse_minimal_pdf_document() {
        let pdf = b"%PDF-1.4\n1 0 obj\n<< /Length 50 >>\nstream\n0 0 m 100 0 l 100 100 l 0 100 l h S\nendstream\nendobj\ntrailer\n<<>>\n%%EOF";
        let options = PdfOptions { scale: 1.0, default_allowance_mm: 10.0 };
        let pattern = import_pattern_pdf(pdf, &options).unwrap();
        assert_eq!(pattern.pieces.len(), 1);
        assert_eq!(pattern.pieces[0].loops.len(), 1);
    }

    #[test]
    fn pdf_piece_uses_positioned_text_as_its_source_name() {
        let pdf = b"%PDF-1.4\n1 0 obj\n<< /Length 120 >>\nstream\n0 0 m 100 0 l 100 100 l 0 100 l h S\nBT /F1 12 Tf 1 0 0 1 25 25 Tm (Front Panel) Tj ET\nendstream\nendobj\ntrailer\n<<>>\n%%EOF";
        let options = PdfOptions { scale: 1.0, default_allowance_mm: 10.0 };
        let pattern = import_pattern_pdf(pdf, &options).unwrap();
        assert_eq!(pattern.pieces[0].name, "Front Panel");
    }

    #[test]
    fn parse_flatedecode_compressed_pdf() {
        // Stream data containing cubic splines and lines, compressed with zlib deflate
        let uncompressed = b"q 1 0 0 1 0 0 cm 0 0 m 200 0 l 200 50 150 150 100 200 c 50 150 0 50 0 0 c h S Q";
        let compressed = miniz_oxide::deflate::compress_to_vec_zlib(uncompressed, 6);
        let mut pdf = Vec::new();
        pdf.extend_from_slice(b"%PDF-1.4\n1 0 obj\n<< /Length ");
        pdf.extend_from_slice(compressed.len().to_string().as_bytes());
        pdf.extend_from_slice(b" /Filter /FlateDecode >>\nstream\n");
        pdf.extend_from_slice(&compressed);
        pdf.extend_from_slice(b"\nendstream\nendobj\ntrailer\n<<>>\n%%EOF");

        let options = PdfOptions { scale: 1.0, default_allowance_mm: 10.0 };
        let pattern = import_pattern_pdf(&pdf, &options).unwrap();
        assert_eq!(pattern.pieces.len(), 1);
        let piece = &pattern.pieces[0];
        let outline = &piece.loops[0].outline;
        assert_eq!(outline.edge_count(), 3);
        assert!(matches!(outline.segments()[0], cad_core::Segment::Line));
        assert!(matches!(outline.segments()[1], cad_core::Segment::Cubic { .. }));
        assert!(matches!(outline.segments()[2], cad_core::Segment::Cubic { .. }));
    }
}

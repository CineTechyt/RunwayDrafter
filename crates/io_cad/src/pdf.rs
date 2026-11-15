//! PDF vector pattern reader: extracts lines, cubic splines and rectangles
//! from PDF graphics streams, regardless of authoring software (Inkscape,
//! Illustrator, Valentina, Seamly2D, Gerber, Lectra, etc.).
//!
//! Every page is read on its own, as it would print: clipped to the page
//! and to its clip paths, with placed drawings (form XObjects) where they
//! are placed. Patterns printed on many small sheets are then joined into
//! one big sheet by [`crate::tiles`].

use std::collections::HashMap;

use glam::Vec2;

use crate::IoError;
use crate::pattern_import::{ImportedPattern, RawLabel, RawSegment, RawSubpath, RawVectorDrawing};
use crate::tiles::{self, Page, Rect};

/// Points to millimetres conversion factor: 1 inch = 72 pt = 25.4 mm.
pub const PT_TO_MM: f32 = 25.4 / 72.0;

#[derive(Debug, Clone, PartialEq)]
pub struct PdfOptions {
    /// Scale factor from PDF user space to millimetres. Defaults to `PT_TO_MM`.
    pub scale: f32,
    /// Default seam allowance width in mm for pieces without an inner loop.
    pub default_allowance_mm: f32,
    /// Join the pages of a pattern printed on many sheets (A4, US Letter)
    /// into one. Off, pages are placed side by side.
    pub join_pages: bool,
}

impl Default for PdfOptions {
    fn default() -> Self {
        Self {
            scale: PT_TO_MM,
            default_allowance_mm: 10.0,
            join_pages: true,
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

    pub fn scale(s: f32) -> Self {
        Self { a: s, b: 0.0, c: 0.0, d: s, e: 0.0, f: 0.0 }
    }

    pub fn translate(x: f32, y: f32) -> Self {
        Self { e: x, f: y, ..Self::IDENTITY }
    }

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
        return Err(IoError::malformed(
            "not a valid PDF document (missing %PDF- header)",
        ));
    }

    let mut pages = read_pages(bytes, options.scale);
    // A file whose lines were all turned into filled shapes ("outline
    // stroke"): the fills are all there is.
    if pages.iter().all(|page| page.drawing.subpaths.is_empty()) {
        for page in &mut pages {
            let fills = std::mem::take(&mut page.drawing.fills);
            page.drawing.subpaths.extend(fills);
        }
    }
    if pages.is_empty() {
        return Err(IoError::malformed(
            "no vector graphics streams found in PDF",
        ));
    }
    if pages.iter().all(|page| page.drawing.subpaths.is_empty()) {
        return Err(IoError::malformed(
            "no vector paths or splines found in PDF content",
        ));
    }

    let (drawing, layout) = if options.join_pages {
        tiles::join(pages)
    } else {
        (tiles::side_by_side(pages), None)
    };
    let mut drawing = drawing;
    drawing.labels = crate::pattern_import::merge_text(std::mem::take(&mut drawing.labels));
    let mut pattern = drawing.assemble_pattern(options.default_allowance_mm);
    if let Some(layout) = layout {
        tiles::disable_page_marks(&mut pattern, &layout);
        pattern.pages = Some(layout);
    }
    Ok(pattern)
}

/// Every page's drawing, in page order. A file whose page tree cannot be
/// followed (damaged, or a bare content stream) gives one page holding
/// every drawing stream in the file.
fn read_pages(bytes: &[u8], scale: f32) -> Vec<Page> {
    let objects = index_objects(bytes);
    let mut reader = Reader {
        bytes,
        objects: &objects,
        fonts: HashMap::new(),
    };
    let entries = page_entries(bytes, &objects);
    if !entries.is_empty() {
        let pages: Vec<Page> = entries.iter().map(|e| reader.read_page(e, scale)).collect();
        if pages.iter().any(|p| !p.drawing.subpaths.is_empty()) {
            return pages;
        }
    }

    let streams = extract_all_streams(bytes);
    if streams.is_empty() {
        return Vec::new();
    }
    let mut drawing = RawVectorDrawing::new();
    let state = State::new(Affine2D::scale(scale), None);
    for stream in &streams {
        reader.run(stream, "", state, 0, &mut drawing);
    }
    vec![Page {
        size: None,
        drawing,
    }]
}

/// A page and what it inherits from the page tree.
struct PageEntry {
    contents: Vec<u32>,
    resources: String,
    media: Option<[f32; 4]>,
    crop: Option<[f32; 4]>,
    rotate: i32,
}

/// The file's pages in reading order.
fn page_entries(bytes: &[u8], objects: &HashMap<u32, PdfObject>) -> Vec<PageEntry> {
    // The catalog the (last) trailer names, else any catalog.
    let root = find_last(bytes, b"/Root")
        .and_then(|at| references(&String::from_utf8_lossy(&bytes[at..(at + 40).min(bytes.len())])).first().copied())
        .filter(|n| objects.get(n).is_some_and(|o| has_type(&o.dictionary, "/Catalog")))
        .or_else(|| {
            let mut catalogs: Vec<u32> = objects
                .iter()
                .filter(|(_, o)| has_type(&o.dictionary, "/Catalog"))
                .map(|(n, _)| *n)
                .collect();
            catalogs.sort_unstable();
            catalogs.last().copied()
        });
    let mut leaves = Vec::new();
    if let Some(pages) = root
        .and_then(|n| objects.get(&n))
        .and_then(|catalog| dictionary_value(&catalog.dictionary, "/Pages"))
        .and_then(|v| references(v).first().copied())
    {
        walk_pages(objects, pages, &mut leaves, 0);
    }
    if leaves.is_empty() {
        leaves = objects
            .iter()
            .filter(|(_, o)| has_type(&o.dictionary, "/Page"))
            .map(|(n, _)| *n)
            .collect();
        leaves.sort_unstable();
    }
    leaves.into_iter().map(|n| page_entry(objects, n)).collect()
}

fn walk_pages(objects: &HashMap<u32, PdfObject>, node: u32, out: &mut Vec<u32>, depth: u32) {
    if depth > 32 || out.contains(&node) {
        return;
    }
    let Some(object) = objects.get(&node) else {
        return;
    };
    if has_type(&object.dictionary, "/Pages") {
        if let Some(kids) = dictionary_value(&object.dictionary, "/Kids") {
            for kid in references(&deref(objects, kids)) {
                walk_pages(objects, kid, out, depth + 1);
            }
        }
    } else if has_type(&object.dictionary, "/Page") {
        out.push(node);
    }
}

fn page_entry(objects: &HashMap<u32, PdfObject>, number: u32) -> PageEntry {
    // Inheritable attributes: from the page, else its nearest ancestor.
    let inherited = |key: &str| -> Option<String> {
        let mut node = number;
        for _ in 0..32 {
            let object = objects.get(&node)?;
            if let Some(value) = dictionary_value(&object.dictionary, key) {
                return Some(deref(objects, value));
            }
            node = *references(dictionary_value(&object.dictionary, "/Parent")?).first()?;
        }
        None
    };
    let rect = |key: &str| {
        inherited(key).and_then(|v| {
            let n = numbers(&v);
            (n.len() == 4).then(|| [n[0].min(n[2]), n[1].min(n[3]), n[0].max(n[2]), n[1].max(n[3])])
        })
    };
    let dictionary = objects.get(&number).map(|o| o.dictionary.as_str()).unwrap_or("");
    let contents = dictionary_value(dictionary, "/Contents")
        .map(|value| {
            references(value)
                .into_iter()
                .flat_map(|n| match objects.get(&n) {
                    // `/Contents 7 0 R` may point at an array of streams.
                    Some(object) if object.stream.is_none() => references(&object.dictionary),
                    _ => vec![n],
                })
                .collect()
        })
        .unwrap_or_default();
    PageEntry {
        contents,
        resources: inherited("/Resources").unwrap_or_default(),
        media: rect("/MediaBox"),
        crop: rect("/CropBox"),
        rotate: inherited("/Rotate")
            .and_then(|v| numbers(&v).first().copied())
            .map_or(0, |r| (r as i32).rem_euclid(360)),
    }
}

/// Whether a dictionary says `/Type /<name>` (`/Page` does not match `/Pages`).
fn has_type(dictionary: &str, name: &str) -> bool {
    let mut search = 0;
    while let Some(found) = dictionary[search..].find("/Type") {
        let after = dictionary[search + found + 5..].trim_start();
        if let Some(rest) = after.strip_prefix(name)
            && rest.chars().next().is_none_or(|c| !c.is_ascii_alphanumeric())
        {
            return true;
        }
        search += found + 5;
    }
    false
}

/// A value, or the dictionary of the object it refers to (`12 0 R`).
fn deref(objects: &HashMap<u32, PdfObject>, value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.ends_with('R') && !trimmed.contains('[') && !trimmed.contains("<<") {
        if let Some(object) = references(trimmed).first().and_then(|n| objects.get(n)) {
            return object.dictionary.clone();
        }
    }
    trimmed.to_owned()
}

fn numbers(text: &str) -> Vec<f32> {
    text.split(|c: char| c.is_whitespace() || c == '[' || c == ']')
        .filter_map(|w| w.parse().ok())
        .collect()
}

fn find_last(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).rposition(|w| w == needle)
}

/// What a font's codes mean as text (its `/ToUnicode` map).
#[derive(Debug, Default)]
struct ToUnicode {
    /// Bytes per code.
    width: usize,
    map: HashMap<u32, String>,
}

impl ToUnicode {
    fn parse(cmap: &[u8], default_width: usize) -> Self {
        let text = String::from_utf8_lossy(cmap);
        let mut tokens = Tokenizer::new(&text);
        let mut out = Self {
            width: default_width,
            map: HashMap::new(),
        };
        let hex = |t: &str| -> Option<Vec<u8>> {
            (t.starts_with('<') && t != "<<").then(|| pdf_string_bytes(t))
        };
        let code = |b: &[u8]| b.iter().fold(0u32, |v, &x| (v << 8) | u32::from(x));
        let utf16 = |b: &[u8]| {
            let units: Vec<u16> = b.chunks(2).map(|c| u16::from_be_bytes([c[0], *c.get(1).unwrap_or(&0)])).collect();
            String::from_utf16_lossy(&units)
        };
        let mut mode = "";
        let mut pending: Vec<Vec<u8>> = Vec::new();
        let mut array: Option<Vec<Vec<u8>>> = None;
        while let Some(token) = tokens.next() {
            match token {
                "begincodespacerange" | "beginbfchar" | "beginbfrange" => {
                    mode = token;
                    pending.clear();
                }
                "endcodespacerange" | "endbfchar" | "endbfrange" => mode = "",
                "[" => array = Some(Vec::new()),
                "]" => {
                    if let (Some(items), [lo, hi]) = (array.take(), pending.as_slice()) {
                        for (offset, item) in (code(lo)..=code(hi)).zip(items) {
                            out.map.insert(offset, utf16(&item));
                        }
                        pending.clear();
                    }
                }
                _ => {
                    let Some(bytes) = hex(token) else { continue };
                    if let Some(items) = array.as_mut() {
                        items.push(bytes);
                        continue;
                    }
                    pending.push(bytes);
                    match (mode, pending.len()) {
                        ("begincodespacerange", 2) => {
                            out.width = pending[0].len().max(1);
                            pending.clear();
                        }
                        ("beginbfchar", 2) => {
                            out.map.insert(code(&pending[0]), utf16(&pending[1]));
                            pending.clear();
                        }
                        ("beginbfrange", 3) => {
                            let (lo, hi) = (code(&pending[0]), code(&pending[1]));
                            let first = &pending[2];
                            for (k, c) in (lo..=hi.min(lo + 65535)).enumerate() {
                                let mut target = first.clone();
                                if let Some(last) = target.last_mut() {
                                    *last = last.wrapping_add(k as u8);
                                }
                                out.map.insert(c, utf16(&target));
                            }
                            pending.clear();
                        }
                        _ => {}
                    }
                }
            }
        }
        out
    }
}

/// Graphics state that `q`/`Q` save and restore.
#[derive(Debug, Clone, Copy)]
struct State {
    ctm: Affine2D,
    clip: Option<Rect>,
    /// Paper-coloured paint: masks and backgrounds, not drawn lines.
    white_fill: bool,
    white_stroke: bool,
}

impl State {
    fn new(ctm: Affine2D, clip: Option<Rect>) -> Self {
        Self {
            ctm,
            clip,
            white_fill: false,
            white_stroke: false,
        }
    }
}

/// Runs content streams, following placed drawings and fonts through the file.
struct Reader<'a> {
    bytes: &'a [u8],
    objects: &'a HashMap<u32, PdfObject>,
    fonts: HashMap<u32, std::rc::Rc<Font>>,
}

/// What text needs from a font: what its codes mean and how wide they are.
#[derive(Debug, Default)]
struct Font {
    /// Bytes per character code.
    code_bytes: usize,
    to_unicode: Option<ToUnicode>,
    /// Advance widths in thousandths of the font size, by code.
    widths: HashMap<u32, f32>,
    default_width: f32,
}

impl Font {
    /// The characters of `bytes`, each with its code's advance width.
    fn glyphs(&self, bytes: &[u8]) -> Vec<(String, u32, f32)> {
        bytes
            .chunks(self.code_bytes.max(1))
            .map(|chunk| {
                let code = chunk.iter().fold(0u32, |v, &x| (v << 8) | u32::from(x));
                let text = match &self.to_unicode {
                    Some(map) => map.map.get(&code).cloned().unwrap_or_default(),
                    None if self.code_bytes == 1 => char::from(chunk[0]).to_string(),
                    None => String::new(),
                };
                let width = self.widths.get(&code).copied().unwrap_or(self.default_width);
                (text, code, width / 1000.0)
            })
            .collect()
    }
}

/// Widths of a CID font's `/W` array: `c [w1 w2 …]` or `first last w`.
fn cid_widths(array: &str, out: &mut HashMap<u32, f32>) {
    let mut tokens = Tokenizer::new(array);
    let mut pending: Vec<f32> = Vec::new();
    let mut list: Option<(u32, u32)> = None; // (first code, index)
    let mut depth = 0;
    while let Some(token) = tokens.next() {
        match token {
            "[" => {
                depth += 1;
                if let Some(&first) = pending.last()
                    && depth >= 2
                {
                    list = Some((first as u32, 0));
                    pending.clear();
                }
            }
            "]" => {
                depth -= 1;
                list = None;
            }
            _ => {
                let Ok(value) = token.parse::<f32>() else { continue };
                if let Some((first, index)) = list.as_mut() {
                    out.insert(*first + *index, value);
                    *index += 1;
                } else {
                    pending.push(value);
                    if pending.len() == 3 {
                        let (a, b) = (pending[0] as u32, pending[1] as u32);
                        for code in a..=b.min(a + 65535) {
                            out.insert(code, pending[2]);
                        }
                        pending.clear();
                    }
                }
            }
        }
    }
}

/// What is to be shown by the next text operator.
enum Shown {
    Text(Vec<u8>),
    /// A TJ adjustment, in thousandths of the font size (positive: back).
    Kern(f32),
}

impl Reader<'_> {
    fn read_page(&mut self, entry: &PageEntry, scale: f32) -> Page {
        // The paper (MediaBox) places the drawing; the CropBox only hides
        // part of it. Tiles cut to the pattern's edge keep their place.
        let media = entry.media.unwrap_or([0.0, 0.0, 612.0, 792.0]);
        let visible = match entry.crop {
            Some(c) => [
                media[0].max(c[0]),
                media[1].max(c[1]),
                media[2].min(c[2]),
                media[3].min(c[3]),
            ],
            None => media,
        };
        let [x0, y0, x1, y1] = media;
        // As displayed: rotated clockwise by /Rotate, lower-left corner at 0.
        let view = match entry.rotate {
            90 => Affine2D { a: 0.0, b: -1.0, c: 1.0, d: 0.0, e: -y0, f: x1 },
            180 => Affine2D { a: -1.0, b: 0.0, c: 0.0, d: -1.0, e: x1, f: y1 },
            270 => Affine2D { a: 0.0, b: 1.0, c: -1.0, d: 0.0, e: y1, f: -x0 },
            _ => Affine2D { a: 1.0, b: 0.0, c: 0.0, d: 1.0, e: -x0, f: -y0 },
        };
        let base = Affine2D::scale(scale).concat(view);
        let (w, h) = (x1 - x0, y1 - y0);
        let size = if entry.rotate % 180 == 90 {
            Vec2::new(h, w)
        } else {
            Vec2::new(w, h)
        } * scale;
        let shown = Rect::from_points([
            base.transform(visible[0], visible[1]),
            base.transform(visible[2], visible[3]),
        ])
        .unwrap_or(Rect::new(Vec2::ZERO, size));
        let state = State::new(base, Some(shown));

        let mut content = Vec::new();
        for number in &entry.contents {
            if let Some(data) = self.objects.get(number).and_then(|o| decode_stream(self.bytes, o)) {
                // Streams of one page continue each other.
                content.extend_from_slice(&data);
                content.push(b'\n');
            }
        }
        let mut drawing = RawVectorDrawing::new();
        self.run(&content, &entry.resources, state, 0, &mut drawing);
        Page {
            size: Some(size),
            drawing,
        }
    }

    /// The font `/name` in `resources`.
    fn font(&mut self, resources: &str, name: &str) -> Option<std::rc::Rc<Font>> {
        let fonts = deref(self.objects, dictionary_value(resources, "/Font")?);
        let number = *references(dictionary_value(&fonts, name)?).first()?;
        if let Some(cached) = self.fonts.get(&number) {
            return Some(cached.clone());
        }
        let dictionary = self.objects.get(&number)?.dictionary.clone();
        let composite = dictionary.contains("/Type0");
        let code_bytes = if composite { 2 } else { 1 };
        let to_unicode = dictionary_value(&dictionary, "/ToUnicode")
            .and_then(|v| references(v).first().copied())
            .and_then(|n| self.objects.get(&n))
            .and_then(|o| decode_stream(self.bytes, o))
            .map(|data| ToUnicode::parse(&data, code_bytes));
        let mut widths = HashMap::new();
        let mut default_width = 500.0;
        if composite {
            let descendant = dictionary_value(&dictionary, "/DescendantFonts")
                .map(|v| deref(self.objects, v))
                .and_then(|v| references(&v).first().copied())
                .and_then(|n| self.objects.get(&n))
                .map(|o| o.dictionary.clone())
                .unwrap_or_default();
            default_width = dictionary_value(&descendant, "/DW")
                .and_then(|v| numbers(v).first().copied())
                .unwrap_or(1000.0);
            if let Some(w) = dictionary_value(&descendant, "/W") {
                cid_widths(&deref(self.objects, w), &mut widths);
            }
        } else {
            let first = dictionary_value(&dictionary, "/FirstChar")
                .and_then(|v| numbers(v).first().copied())
                .unwrap_or(0.0) as u32;
            if let Some(list) = dictionary_value(&dictionary, "/Widths") {
                for (i, w) in numbers(&deref(self.objects, list)).into_iter().enumerate() {
                    widths.insert(first + i as u32, w);
                }
            }
        }
        let font = std::rc::Rc::new(Font {
            code_bytes: to_unicode.as_ref().map_or(code_bytes, |m| m.width),
            to_unicode,
            widths,
            default_width,
        });
        self.fonts.insert(number, font.clone());
        Some(font)
    }

    /// Executes graphics commands, appending what they draw to `drawing`.
    fn run(
        &mut self,
        stream: &[u8],
        resources: &str,
        state: State,
        depth: u32,
        drawing: &mut RawVectorDrawing,
    ) {
        let text = String::from_utf8_lossy(stream);
        let mut tokens = Tokenizer::new(&text);

        let mut state = state;
        let mut stack: Vec<State> = Vec::new();
        let mut args: Vec<f32> = Vec::with_capacity(8);
        let mut name: Option<&str> = None;

        let mut current_point: Option<Vec2> = None;
        let mut current_subpath: Option<RawSubpath> = None;
        let mut pending_subpaths: Vec<RawSubpath> = Vec::new();
        let mut clip_next = false;

        // Text state.
        let mut font: Option<std::rc::Rc<Font>> = None;
        let mut font_size = 1.0f32;
        let mut char_spacing = 0.0f32;
        let mut word_spacing = 0.0f32;
        let mut horizontal_scale = 1.0f32;
        let mut line_matrix = Affine2D::IDENTITY;
        let mut text_matrix = Affine2D::IDENTITY;
        let mut leading = 0.0f32;
        let mut shown: Vec<Shown> = Vec::new();
        let mut in_array = false;

        let is_white = |args: &[f32]| match args.len() {
            1 | 3 => args.iter().all(|&v| v >= 0.999),
            4 => args.iter().all(|&v| v <= 0.001),
            _ => false,
        };

        while let Some(tok) = tokens.next() {
            if let Ok(num) = tok.parse::<f32>() {
                if in_array {
                    shown.push(Shown::Kern(num));
                }
                args.push(num);
                continue;
            }
            if tok.starts_with('(') || (tok.starts_with('<') && tok != "<<") {
                shown.push(Shown::Text(pdf_string_bytes(tok)));
                continue;
            }

            let mut paint: Option<(bool, bool, bool)> = None; // (fill, stroke, close)
            match tok {
                "cm" if args.len() >= 6 => {
                    let n = &args[args.len() - 6..];
                    state.ctm = state.ctm.concat(Affine2D {
                        a: n[0],
                        b: n[1],
                        c: n[2],
                        d: n[3],
                        e: n[4],
                        f: n[5],
                    });
                }
                "q" => stack.push(state),
                "Q" => {
                    if let Some(saved) = stack.pop() {
                        state = saved;
                    }
                }
                "g" | "rg" | "k" | "sc" | "scn" => state.white_fill = is_white(&args),
                "G" | "RG" | "K" | "SC" | "SCN" => state.white_stroke = is_white(&args),
                "cs" => state.white_fill = false,
                "CS" => state.white_stroke = false,
                "m" if args.len() >= 2 => {
                    if let Some(sub) = current_subpath.take()
                        && !sub.is_empty()
                    {
                        pending_subpaths.push(sub);
                    }
                    let pt = state.ctm.transform(args[args.len() - 2], args[args.len() - 1]);
                    current_point = Some(pt);
                    current_subpath = Some(RawSubpath::new(pt));
                }
                "l" if args.len() >= 2 => {
                    let pt = state.ctm.transform(args[args.len() - 2], args[args.len() - 1]);
                    if let Some(sub) = &mut current_subpath {
                        sub.segments.push(RawSegment::Line { to: pt });
                        current_point = Some(pt);
                    }
                }
                "c" | "v" | "y" => {
                    let needed = if tok == "c" { 6 } else { 4 };
                    if args.len() >= needed {
                        let n = &args[args.len() - needed..];
                        let p = |i: usize| state.ctm.transform(n[i], n[i + 1]);
                        let (c1, c2, p3) = match tok {
                            "c" => (p(0), p(2), p(4)),
                            "v" => (current_point.unwrap_or(p(0)), p(0), p(2)),
                            _ => (p(0), p(2), p(2)),
                        };
                        if let Some(sub) = &mut current_subpath {
                            sub.segments.push(RawSegment::Cubic {
                                ctrl1: c1,
                                ctrl2: c2,
                                to: p3,
                            });
                            current_point = Some(p3);
                        }
                    }
                }
                "re" if args.len() >= 4 => {
                    if let Some(sub) = current_subpath.take()
                        && !sub.is_empty()
                    {
                        pending_subpaths.push(sub);
                    }
                    let n = &args[args.len() - 4..];
                    let (x, y, w, h) = (n[0], n[1], n[2], n[3]);
                    let p0 = state.ctm.transform(x, y);
                    let mut rect = RawSubpath::new(p0);
                    for (cx, cy) in [(x + w, y), (x + w, y + h), (x, y + h), (x, y)] {
                        rect.segments.push(RawSegment::Line {
                            to: state.ctm.transform(cx, cy),
                        });
                    }
                    rect.closed = true;
                    pending_subpaths.push(rect);
                    current_point = Some(p0);
                }
                "h" => {
                    if let Some(sub) = &mut current_subpath {
                        close(sub, &mut current_point);
                    }
                }
                "W" | "W*" => clip_next = true,
                "S" => paint = Some((false, true, false)),
                "s" => paint = Some((false, true, true)),
                "f" | "F" | "f*" => paint = Some((true, false, false)),
                "B" | "B*" => paint = Some((true, true, false)),
                "b" | "b*" => paint = Some((true, true, true)),
                "n" => paint = Some((false, false, false)),

                // Text.
                "BT" => {
                    line_matrix = Affine2D::IDENTITY;
                    text_matrix = Affine2D::IDENTITY;
                }
                "Tf" => {
                    font = name.and_then(|n| self.font(resources, n));
                    if let Some(&size) = args.last() {
                        font_size = size;
                    }
                }
                "Tc" if !args.is_empty() => char_spacing = args[args.len() - 1],
                "Tw" if !args.is_empty() => word_spacing = args[args.len() - 1],
                "Tz" if !args.is_empty() => horizontal_scale = args[args.len() - 1] / 100.0,
                "TL" if !args.is_empty() => leading = args[args.len() - 1],
                "Tm" if args.len() >= 6 => {
                    let n = &args[args.len() - 6..];
                    line_matrix = Affine2D {
                        a: n[0],
                        b: n[1],
                        c: n[2],
                        d: n[3],
                        e: n[4],
                        f: n[5],
                    };
                    text_matrix = line_matrix;
                }
                "Td" | "TD" if args.len() >= 2 => {
                    let (tx, ty) = (args[args.len() - 2], args[args.len() - 1]);
                    if tok == "TD" {
                        leading = -ty;
                    }
                    line_matrix = line_matrix.concat(Affine2D::translate(tx, ty));
                    text_matrix = line_matrix;
                }
                "T*" => {
                    line_matrix = line_matrix.concat(Affine2D::translate(0.0, -leading));
                    text_matrix = line_matrix;
                }
                "[" => in_array = true,
                "]" => in_array = false,
                "Tj" | "TJ" | "'" | "\"" => {
                    if tok == "'" || tok == "\"" {
                        line_matrix = line_matrix.concat(Affine2D::translate(0.0, -leading));
                        text_matrix = line_matrix;
                    }
                    // Every character at its place along the baseline.
                    let m = state.ctm.concat(text_matrix);
                    let along = Vec2::new(m.a, m.b);
                    let up = Vec2::new(m.c, m.d);
                    let unit = along.length();
                    let mut x = 0.0f32;
                    let mut glyphs: Vec<(char, f32)> = Vec::new();
                    for item in shown.drain(..) {
                        match item {
                            Shown::Kern(k) => x -= k / 1000.0 * font_size * horizontal_scale,
                            Shown::Text(bytes) => {
                                let decoded = match &font {
                                    Some(f) => f.glyphs(&bytes),
                                    None => bytes
                                        .iter()
                                        .map(|&b| (char::from(b).to_string(), u32::from(b), 0.5))
                                        .collect(),
                                };
                                let single = font.as_ref().is_none_or(|f| f.code_bytes == 1);
                                for (text, code, width) in decoded {
                                    for ch in text.chars() {
                                        glyphs.push((ch, x * unit));
                                    }
                                    let space = if single && code == 32 { word_spacing } else { 0.0 };
                                    x += (width * font_size + char_spacing + space) * horizontal_scale;
                                }
                            }
                        }
                    }
                    let label = RawLabel {
                        text: glyphs.iter().map(|g| g.0).collect(),
                        position: m.transform(0.0, 0.0),
                        size: font_size.abs() * up.length(),
                        direction: along.normalize_or(Vec2::X),
                        glyphs,
                        width: x * unit,
                    };
                    text_matrix = text_matrix.concat(Affine2D::translate(x, 0.0));
                    let visible = match state.clip {
                        Some(clip) => label.keep_glyphs(|p| clip.contains(p, 0.5)),
                        None => Some(label),
                    };
                    if let Some(label) = visible
                        && label.text.chars().any(char::is_alphabetic)
                    {
                        drawing.labels.push(label);
                    }
                }

                // Placed drawings.
                "Do" => {
                    if let Some(n) = name {
                        self.draw_form(resources, n, state, depth, drawing);
                    }
                }
                _ => {
                    if tok.starts_with('/') {
                        name = Some(tok);
                        args.clear();
                        continue;
                    }
                }
            }
            args.clear();

            if let Some((fill, stroke, closing)) = paint {
                if let Some(mut sub) = current_subpath.take() {
                    if closing {
                        close(&mut sub, &mut current_point);
                    }
                    if !sub.is_empty() {
                        pending_subpaths.push(sub);
                    }
                }
                let shows = (fill && !state.white_fill) || (stroke && !state.white_stroke);
                let new_clip = clip_next.then(|| {
                    Rect::from_points(pending_subpaths.iter().flat_map(|p| {
                        std::iter::once(p.start).chain(p.segments.iter().map(RawSegment::end))
                    }))
                });
                if shows {
                    // Filled without a stroke: lettering and logos, not lines.
                    let target = if stroke && !state.white_stroke {
                        &mut drawing.subpaths
                    } else {
                        &mut drawing.fills
                    };
                    for sub in pending_subpaths.drain(..) {
                        match state.clip {
                            Some(clip) => tiles::clip_subpath(&sub, clip, false, target),
                            None => target.push(sub),
                        }
                    }
                }
                pending_subpaths.clear();
                if let Some(Some(region)) = new_clip {
                    state.clip = Some(match state.clip {
                        Some(clip) => clip.intersect(&region),
                        None => region,
                    });
                }
                clip_next = false;
            }
        }

        // A path left unpainted at the end: keep it, as earlier readers did.
        if let Some(sub) = current_subpath
            && !sub.is_empty()
        {
            pending_subpaths.push(sub);
        }
        drawing.subpaths.append(&mut pending_subpaths);
    }

    /// Draws the form XObject `/name` placed with the current state.
    fn draw_form(
        &mut self,
        resources: &str,
        name: &str,
        state: State,
        depth: u32,
        drawing: &mut RawVectorDrawing,
    ) {
        if depth >= 12 {
            return;
        }
        let Some(xobjects) = dictionary_value(resources, "/XObject").map(|v| deref(self.objects, v))
        else {
            return;
        };
        let Some(number) = dictionary_value(&xobjects, name).and_then(|v| references(v).first().copied())
        else {
            return;
        };
        let Some(form) = self.objects.get(&number) else {
            return;
        };
        if !form.dictionary.contains("/Form") {
            return; // an image
        }
        let Some(data) = decode_stream(self.bytes, form) else {
            return;
        };
        let matrix = dictionary_value(&form.dictionary, "/Matrix")
            .map(numbers)
            .filter(|n| n.len() == 6)
            .map_or(Affine2D::IDENTITY, |n| Affine2D {
                a: n[0],
                b: n[1],
                c: n[2],
                d: n[3],
                e: n[4],
                f: n[5],
            });
        let mut inner = state;
        inner.ctm = state.ctm.concat(matrix);
        if let Some(b) = dictionary_value(&form.dictionary, "/BBox").map(numbers)
            && b.len() == 4
            && let Some(region) = Rect::from_points([
                inner.ctm.transform(b[0], b[1]),
                inner.ctm.transform(b[2], b[1]),
                inner.ctm.transform(b[2], b[3]),
                inner.ctm.transform(b[0], b[3]),
            ])
        {
            inner.clip = Some(state.clip.map_or(region, |c| c.intersect(&region)));
        }
        let own = dictionary_value(&form.dictionary, "/Resources").map(|v| deref(self.objects, v));
        let form_resources = own.unwrap_or_else(|| resources.to_owned());
        self.run(&data, &form_resources, inner, depth + 1, drawing);
    }
}

/// Closes a subpath with a straight line back to its start.
fn close(sub: &mut RawSubpath, current_point: &mut Option<Vec2>) {
    let start = sub.start;
    if let Some(cur) = *current_point
        && cur.distance(start) > 1e-4
    {
        sub.segments.push(RawSegment::Line { to: start });
        *current_point = Some(start);
    }
    sub.closed = true;
}

/// Reads a content stream on its own, without a page: for tests.
#[cfg(test)]
fn parse_content_stream(stream: &[u8], scale: f32, drawing: &mut RawVectorDrawing) {
    let objects = HashMap::new();
    let mut reader = Reader {
        bytes: &[],
        objects: &objects,
        fonts: HashMap::new(),
    };
    reader.run(stream, "", State::new(Affine2D::scale(scale), None), 0, drawing);
}

/// The bytes of a PDF string token: `(literal)` or `<hex>`.
fn pdf_string_bytes(token: &str) -> Vec<u8> {
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
                    while digits < 3
                        && index + 1 < bytes.len()
                        && (b'0'..=b'7').contains(&bytes[index])
                    {
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
        decoded
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
        bytes
    }
}

/// One `N G obj … endobj` of the file: its dictionary text and, for a
/// stream object, where the raw stream data lies.
#[derive(Debug, Clone)]
struct PdfObject {
    dictionary: String,
    stream: Option<std::ops::Range<usize>>,
}

/// Every top-level object by number. A number defined twice keeps its last
/// definition, as an incremental update intends.
fn index_objects(bytes: &[u8]) -> HashMap<u32, PdfObject> {
    let mut objects = HashMap::new();
    let mut cursor = 0;
    while let Some(relative) = find_subsequence(&bytes[cursor..], b" obj") {
        let keyword = cursor + relative;
        cursor = keyword + 4;
        // "<number> <generation> obj", at the start of a line or after whitespace.
        let Some((number, header_start)) = object_number_before(bytes, keyword) else {
            continue;
        };
        if header_start > 0 && !bytes[header_start - 1].is_ascii_whitespace() {
            continue;
        }
        let body_start = keyword + 4;
        let Some(end_relative) = find_subsequence(&bytes[body_start..], b"endobj") else {
            break;
        };
        let body_end = body_start + end_relative;
        let body = &bytes[body_start..body_end];
        let (dictionary, stream) = match find_subsequence(body, b"stream") {
            Some(at) if !body[..at].ends_with(b"end") => {
                let mut data_start = body_start + at + 6;
                if bytes.get(data_start) == Some(&b'\r') {
                    data_start += 1;
                }
                if bytes.get(data_start) == Some(&b'\n') {
                    data_start += 1;
                }
                let data_end = find_subsequence(&bytes[data_start..body_end], b"endstream")
                    .map_or(body_end, |end| data_start + end);
                (
                    String::from_utf8_lossy(&body[..at]).into_owned(),
                    Some(data_start..data_end),
                )
            }
            _ => (String::from_utf8_lossy(body).into_owned(), None),
        };
        objects.insert(number, PdfObject { dictionary, stream });
        cursor = body_end + 6;
    }

    // PDF 1.5 packs small objects (pages, fonts, resources) into
    // compressed object streams. They hold no streams of their own.
    let packs: Vec<PdfObject> = objects
        .values()
        .filter(|o| o.stream.is_some() && has_type(&o.dictionary, "/ObjStm"))
        .cloned()
        .collect();
    for pack in packs {
        let Some(data) = decode_stream(bytes, &pack) else {
            continue;
        };
        let value = |key: &str| {
            dictionary_value(&pack.dictionary, key)
                .and_then(|v| v.trim().parse::<usize>().ok())
        };
        let (Some(count), Some(first)) = (value("/N"), value("/First")) else {
            continue;
        };
        if first > data.len() {
            continue;
        }
        let header: Vec<usize> = String::from_utf8_lossy(&data[..first])
            .split_whitespace()
            .filter_map(|w| w.parse().ok())
            .collect();
        let entries: Vec<(usize, usize)> = header
            .chunks(2)
            .take(count)
            .filter(|c| c.len() == 2)
            .map(|c| (c[0], c[1]))
            .collect();
        for (k, &(number, offset)) in entries.iter().enumerate() {
            let start = first + offset;
            let end = entries.get(k + 1).map_or(data.len(), |next| first + next.1);
            if start >= end || end > data.len() {
                continue;
            }
            // A top-level definition (an incremental update) wins.
            objects.entry(number as u32).or_insert_with(|| PdfObject {
                dictionary: String::from_utf8_lossy(&data[start..end]).into_owned(),
                stream: None,
            });
        }
    }
    objects
}

/// Parses "N G" right before byte `keyword`. Returns N and where "N" starts.
fn object_number_before(bytes: &[u8], keyword: usize) -> Option<(u32, usize)> {
    let digits_back = |end: usize| {
        let mut start = end;
        while start > 0 && bytes[start - 1].is_ascii_digit() {
            start -= 1;
        }
        (start < end).then_some(start)
    };
    let generation_start = digits_back(keyword)?;
    let mut number_end = generation_start;
    if number_end == 0 || !bytes[number_end - 1].is_ascii_whitespace() {
        return None;
    }
    while number_end > 0 && bytes[number_end - 1].is_ascii_whitespace() {
        number_end -= 1;
    }
    let number_start = digits_back(number_end)?;
    let number = std::str::from_utf8(&bytes[number_start..number_end])
        .ok()?
        .parse()
        .ok()?;
    Some((number, number_start))
}

/// Object numbers referenced as `N G R` in `text`.
fn references(text: &str) -> Vec<u32> {
    let words: Vec<&str> = text
        .split(|c: char| c.is_whitespace() || "[]<>/()".contains(c))
        .filter(|w| !w.is_empty())
        .collect();
    words
        .windows(3)
        .filter(|w| w[2] == "R" && w[1].bytes().all(|b| b.is_ascii_digit()))
        .filter_map(|w| w[0].parse().ok())
        .collect()
}

/// The text following `/Key` in a dictionary, up to the next key at the
/// same nesting depth: a value such as `12 0 R`, `[ 4 0 R 5 0 R ]` or
/// `<< /Fm0 9 0 R >>`.
fn dictionary_value<'a>(dictionary: &'a str, key: &str) -> Option<&'a str> {
    let mut search = 0;
    let at = loop {
        let found = search + dictionary[search..].find(key)?;
        let after = dictionary[found + key.len()..].chars().next();
        // "/Contents" must not match "/ContentsX".
        if after.is_none_or(|c| !c.is_ascii_alphanumeric()) {
            break found + key.len();
        }
        search = found + key.len();
    };
    let rest = &dictionary[at..];
    let (mut depth, mut end) = (0i32, rest.len());
    let bytes = rest.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'<' if bytes.get(i + 1) == Some(&b'<') => {
                depth += 1;
                i += 1;
            }
            b'>' if bytes.get(i + 1) == Some(&b'>') => {
                if depth == 0 {
                    end = i;
                    break;
                }
                depth -= 1;
                i += 1;
            }
            b'[' => depth += 1,
            b']' => depth -= 1,
            b'/' if depth == 0 && i > 0 => {
                end = i;
                break;
            }
            _ => {}
        }
        i += 1;
    }
    Some(&rest[..end])
}


/// The decoded data of a stream object (Flate or uncompressed).
fn decode_stream(bytes: &[u8], object: &PdfObject) -> Option<Vec<u8>> {
    let whole = &bytes[object.stream.clone()?];
    let mut end = whole.len();
    while end > 0 && (whole[end - 1] == b'\r' || whole[end - 1] == b'\n') {
        end -= 1;
    }
    let trimmed = &whole[..end];
    if object.dictionary.contains("/FlateDecode")
        || object.dictionary.contains("/Fl ")
        || object.dictionary.contains("/Fl/")
    {
        // Compressed data may itself end in a byte that reads as a line
        // break: inflate everything up to `endstream` first (the end of
        // line after the data does no harm), the trimmed data second.
        let inflate = |raw: &[u8]| {
            miniz_oxide::inflate::decompress_to_vec_zlib(raw)
                .ok()
                .or_else(|| miniz_oxide::inflate::decompress_to_vec(raw).ok())
        };
        inflate(whole)
            .or_else(|| inflate(&whole[..end.max(whole.len().saturating_sub(1))]))
            .or_else(|| inflate(trimmed))
    } else if object.dictionary.contains("/Filter") {
        None // other encodings are not supported
    } else {
        Some(trimmed.to_vec())
    }
}

/// Every decompressed stream in the file that looks like drawing commands.
fn extract_all_streams(bytes: &[u8]) -> Vec<Vec<u8>> {
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
        while data_end > data_start
            && (bytes[data_end - 1] == b'\r' || bytes[data_end - 1] == b'\n')
        {
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
    s.contains(" m\n")
        || s.contains(" m ")
        || s.contains(" l\n")
        || s.contains(" c\n")
        || s.contains(" re\n")
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
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
                while self.pos < bytes.len() && bytes[self.pos] != b'\n' && bytes[self.pos] != b'\r'
                {
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
            if ch.is_ascii_whitespace()
                || ch == b'('
                || ch == b')'
                || ch == b'<'
                || ch == b'>'
                || ch == b'['
                || ch == b']'
                || ch == b'%'
            {
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
    fn only_the_drawing_the_page_references_is_imported() {
        // Object 5 is the original drawing, left behind unreferenced by an
        // editor; the page draws object 11 (a different square).
        let pdf = b"%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [ 4 0 R ] /Count 1 >> endobj
4 0 obj << /Type /Page /Parent 2 0 R /Contents 11 0 R /MediaBox [0 0 500 500] >> endobj
5 0 obj << /Length 40 >>
stream
0 0 m 100 0 l 100 100 l 0 100 l h S
endstream
endobj
11 0 obj << /Length 44 >>
stream
200 0 m 300 0 l 300 100 l 200 100 l h S
endstream
endobj
trailer << /Root 1 0 R >>
%%EOF";
        let options = PdfOptions {
            scale: 1.0,
            default_allowance_mm: 10.0,
            ..Default::default()
        };
        let pattern = import_pattern_pdf(pdf, &options).unwrap();
        assert_eq!(pattern.pieces.len(), 1);
        assert!(
            pattern.pieces[0].loops[0].bounds.0.x > 150.0,
            "the referenced drawing, not the orphan"
        );
    }

    #[test]
    fn an_incremental_update_replaces_the_earlier_version_of_an_object() {
        let pdf = b"%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [ 4 0 R ] /Count 1 >> endobj
4 0 obj << /Type /Page /Parent 2 0 R /Contents [ 5 0 R ] >> endobj
5 0 obj << /Length 40 >>
stream
0 0 m 100 0 l 100 100 l 0 100 l h S
endstream
endobj
trailer << /Root 1 0 R >>
%%EOF
5 0 obj << /Length 44 >>
stream
200 0 m 300 0 l 300 100 l 200 100 l h S
endstream
endobj
trailer << /Root 1 0 R /Prev 0 >>
%%EOF";
        let options = PdfOptions {
            scale: 1.0,
            default_allowance_mm: 10.0,
            ..Default::default()
        };
        let pattern = import_pattern_pdf(pdf, &options).unwrap();
        assert_eq!(pattern.pieces.len(), 1);
        assert!(pattern.pieces[0].loops[0].bounds.0.x > 150.0);
    }

    #[test]
    fn a_piece_printed_across_two_pages_is_joined() {
        // Each 100 × 200 pt page draws the whole 140 × 100 piece, shifted
        // to the page's part of the sheet and clipped to the paper, the
        // way Illustrator exports tiles. The right page comes first.
        let pdf = b"%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [ 4 0 R 3 0 R ] /Count 2 /MediaBox [0 0 100 200] >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /Contents 5 0 R >> endobj
4 0 obj << /Type /Page /Parent 2 0 R /Contents 6 0 R >> endobj
5 0 obj << /Length 90 >>
stream
q 0 0 100 200 re W n 10 20 m 150 20 l 150 120 l 120 160 l 10 120 l h S Q
endstream
endobj
6 0 obj << /Length 90 >>
stream
q 0 0 100 200 re W n -90 20 m 50 20 l 50 120 l 20 160 l -90 120 l h S Q
endstream
endobj
trailer << /Root 1 0 R >>
%%EOF";
        let options = PdfOptions {
            scale: 1.0,
            default_allowance_mm: 10.0,
            ..Default::default()
        };
        let pattern = import_pattern_pdf(pdf, &options).unwrap();
        let layout = pattern.pages.as_ref().expect("pages joined");
        assert_eq!((layout.joined, layout.columns, layout.rows), (2, 2, 1));
        assert_eq!(pattern.pieces.len(), 1, "{pattern:#?}");
        let (lo, hi) = pattern.pieces[0].loops[0].bounds;
        assert!(((hi - lo) - Vec2::new(140.0, 140.0)).abs().max_element() < 0.5, "{lo} {hi}");

        // Not joined, the pages lie side by side: two pieces cut open.
        let apart = import_pattern_pdf(
            pdf,
            &PdfOptions {
                join_pages: false,
                ..options
            },
        )
        .unwrap();
        assert!(apart.pages.is_none());
        assert!(apart.pieces.is_empty());
    }

    #[test]
    fn text_cut_at_a_page_edge_is_put_back_together() {
        // "Sleeve" set in two runs, as Distiller splits text at page edges;
        // without a font every letter is half an em wide.
        let pdf = b"%PDF-1.4\n1 0 obj\n<< /Length 160 >>\nstream\n0 0 m 200 0 l 200 100 l 0 100 l h S\nBT 10 0 0 10 50 50 Tm (Sle) Tj ET BT 10 0 0 10 65 50 Tm (eve) Tj ET\nendstream\nendobj\ntrailer\n<<>>\n%%EOF";
        let options = PdfOptions {
            scale: 1.0,
            ..Default::default()
        };
        let pattern = import_pattern_pdf(pdf, &options).unwrap();
        assert_eq!(pattern.pieces[0].name, "Sleeve");
    }

    #[test]
    fn parse_minimal_pdf_document() {
        let pdf = b"%PDF-1.4\n1 0 obj\n<< /Length 50 >>\nstream\n0 0 m 100 0 l 100 100 l 0 100 l h S\nendstream\nendobj\ntrailer\n<<>>\n%%EOF";
        let options = PdfOptions {
            scale: 1.0,
            default_allowance_mm: 10.0,
            ..Default::default()
        };
        let pattern = import_pattern_pdf(pdf, &options).unwrap();
        assert_eq!(pattern.pieces.len(), 1);
        assert_eq!(pattern.pieces[0].loops.len(), 1);
    }

    #[test]
    fn pdf_piece_uses_positioned_text_as_its_source_name() {
        let pdf = b"%PDF-1.4\n1 0 obj\n<< /Length 120 >>\nstream\n0 0 m 100 0 l 100 100 l 0 100 l h S\nBT /F1 12 Tf 1 0 0 1 25 25 Tm (Front Panel) Tj ET\nendstream\nendobj\ntrailer\n<<>>\n%%EOF";
        let options = PdfOptions {
            scale: 1.0,
            default_allowance_mm: 10.0,
            ..Default::default()
        };
        let pattern = import_pattern_pdf(pdf, &options).unwrap();
        assert_eq!(pattern.pieces[0].name, "Front Panel");
    }

    #[test]
    fn parse_flatedecode_compressed_pdf() {
        // Stream data containing cubic splines and lines, compressed with zlib deflate
        let uncompressed =
            b"q 1 0 0 1 0 0 cm 0 0 m 200 0 l 200 50 150 150 100 200 c 50 150 0 50 0 0 c h S Q";
        let compressed = miniz_oxide::deflate::compress_to_vec_zlib(uncompressed, 6);
        let mut pdf = Vec::new();
        pdf.extend_from_slice(b"%PDF-1.4\n1 0 obj\n<< /Length ");
        pdf.extend_from_slice(compressed.len().to_string().as_bytes());
        pdf.extend_from_slice(b" /Filter /FlateDecode >>\nstream\n");
        pdf.extend_from_slice(&compressed);
        pdf.extend_from_slice(b"\nendstream\nendobj\ntrailer\n<<>>\n%%EOF");

        let options = PdfOptions {
            scale: 1.0,
            default_allowance_mm: 10.0,
            ..Default::default()
        };
        let pattern = import_pattern_pdf(&pdf, &options).unwrap();
        assert_eq!(pattern.pieces.len(), 1);
        let piece = &pattern.pieces[0];
        let outline = &piece.loops[0].outline;
        assert_eq!(outline.edge_count(), 3);
        assert!(matches!(outline.segments()[0], cad_core::Segment::Line));
        assert!(matches!(
            outline.segments()[1],
            cad_core::Segment::Cubic { .. }
        ));
        assert!(matches!(
            outline.segments()[2],
            cad_core::Segment::Cubic { .. }
        ));
    }
}

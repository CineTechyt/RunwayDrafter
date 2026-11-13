//! Reading text that a pattern file *draws* instead of writing.
//!
//! Many CAD and plotter exports render every label ("CENTER_FRONT_PART",
//! "Fabric type , Piece qty =(1,0)", "STRAIGHT GRAIN") as pen strokes, so a
//! PDF contains no text objects at all. To still name the pieces, the small
//! strokes inside each piece are drawn into an image, which an OCR engine
//! reads in all four orientations; the most confident reading wins.
//!
//! The engine is [Tesseract](https://github.com/tesseract-ocr/tesseract),
//! run as the `tesseract` command (Debian/Ubuntu: `apt install
//! tesseract-ocr`). Without it pieces keep their numbered names.

use std::io::Write as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use cad_core::geometry;
use glam::Vec2;

use crate::pattern_import::{ImportedPattern, RawSubpath};

/// Strokes larger than this (mm) are lines, not lettering.
const MAX_GLYPH_MM: f32 = 30.0;
/// Rendered letter height in pixels, and stroke thickness relative to it:
/// what Tesseract read best on plotter lettering (tried 20–56 px and
/// 3–20 %; thin strokes in large letters read worst).
const GLYPH_PX: f32 = 24.0;
/// Half the drawn stroke width, as a fraction of the letter height.
const STROKE_RADIUS: f32 = 0.15;
const MARGIN_PX: f32 = 24.0;

/// A grey-scale image, dark ink on white.
#[derive(Debug, Clone, PartialEq)]
pub struct GrayImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl GrayImage {
    /// Binary PGM, readable by every OCR engine and image viewer.
    pub fn to_pgm(&self) -> Vec<u8> {
        let mut out = format!("P5\n{} {}\n255\n", self.width, self.height).into_bytes();
        out.extend_from_slice(&self.pixels);
        out
    }

    fn line(&mut self, a: Vec2, b: Vec2, radius: f32) {
        let (lo, hi) = (
            a.min(b) - Vec2::splat(radius + 1.0),
            a.max(b) + Vec2::splat(radius + 1.0),
        );
        let x0 = lo.x.max(0.0) as u32;
        let y0 = lo.y.max(0.0) as u32;
        let x1 = (hi.x.ceil() as u32).min(self.width.saturating_sub(1));
        let y1 = (hi.y.ceil() as u32).min(self.height.saturating_sub(1));
        for y in y0..=y1 {
            for x in x0..=x1 {
                let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let distance = geometry::distance_to_segment(p, a, b);
                // One pixel of anti-aliasing.
                let ink = (radius + 0.5 - distance).clamp(0.0, 1.0);
                let pixel = &mut self.pixels[(y * self.width + x) as usize];
                *pixel = (*pixel).min((255.0 * (1.0 - ink)) as u8);
            }
        }
    }
}

/// One word an OCR engine found, with its confidence (0–100).
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub text: String,
    pub confidence: f32,
    /// Words with the same line number were read as one line.
    pub line: u32,
}

/// Something that can read text from an image.
pub trait TextReader: Sync {
    fn read(&self, image: &GrayImage) -> Option<Vec<Word>>;
}

/// The `tesseract` command-line program.
#[derive(Debug, Clone)]
pub struct TesseractCli {
    pub program: PathBuf,
}

impl TesseractCli {
    /// The program from `CLO360_TESSERACT`, else `tesseract` on the `PATH`,
    /// if it runs.
    pub fn find() -> Option<Self> {
        let program = std::env::var_os("CLO360_TESSERACT")
            .map_or_else(|| PathBuf::from("tesseract"), PathBuf::from);
        let works = Command::new(&program)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        works
            .is_ok_and(|status| status.success())
            .then_some(Self { program })
    }
}

impl TextReader for TesseractCli {
    fn read(&self, image: &GrayImage) -> Option<Vec<Word>> {
        // "Sparse text" finds words wherever they are; TSV gives each
        // word's confidence.
        let mut child = Command::new(&self.program)
            .args(["stdin", "stdout", "--psm", "11", "tsv"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        child.stdin.take()?.write_all(&image.to_pgm()).ok()?;
        let output = child.wait_with_output().ok()?;
        output
            .status
            .success()
            .then(|| parse_tsv(&String::from_utf8_lossy(&output.stdout)))
    }
}

/// Parses Tesseract's TSV output (level, page, block, par, line, word,
/// left, top, width, height, conf, text).
pub fn parse_tsv(tsv: &str) -> Vec<Word> {
    tsv.lines()
        .skip(1)
        .filter_map(|row| {
            let columns: Vec<&str> = row.split('\t').collect();
            let text = columns.get(11)?.trim();
            let confidence: f32 = columns.get(10)?.parse().ok()?;
            if text.is_empty() || confidence < 0.0 {
                return None;
            }
            // Block, paragraph and line together identify a line.
            let line = columns[2].parse::<u32>().ok()? * 10_000
                + columns[3].parse::<u32>().ok()? * 100
                + columns[4].parse::<u32>().ok()?;
            Some(Word {
                text: text.to_owned(),
                confidence,
                line,
            })
        })
        .collect()
}

/// The small strokes inside piece `index`: what drawn lettering consists of.
pub fn piece_strokes(pattern: &ImportedPattern, index: usize) -> Vec<Vec<Vec2>> {
    let Some(piece) = pattern.pieces.get(index) else {
        return Vec::new();
    };
    let Some(outer) = piece.loops.first() else {
        return Vec::new();
    };
    let polygon = outer.outline.flatten(1.0).points;
    // Some files draw the outlines themselves as many short strokes; those
    // are not lettering.
    let outlines: Vec<Vec<Vec2>> = piece
        .loops
        .iter()
        .map(|l| l.outline.flatten(0.25).points)
        .collect();
    let on_outline = |points: &[Vec2]| {
        outlines.iter().any(|outline| {
            points
                .iter()
                .all(|&p| geometry::distance_to_polygon(p, outline) < 1.0)
        })
    };
    pattern
        .marks
        .iter()
        .map(|mark: &RawSubpath| mark.polyline(0.1))
        .filter(|points| {
            let (lo, hi) = bounds(points);
            (hi - lo).max_element() <= MAX_GLYPH_MM
                && points
                    .iter()
                    .all(|&p| geometry::point_in_polygon(p, &polygon))
                && !on_outline(points)
        })
        .collect()
}

/// Groups strokes into blocks of text: strokes closer than about a letter
/// apart belong together, so every label becomes one block while marks
/// elsewhere on the piece stay out of it.
pub fn text_blocks(strokes: &[Vec<Vec2>]) -> Vec<Vec<Vec<Vec2>>> {
    if strokes.is_empty() {
        return Vec::new();
    }
    let boxes: Vec<(Vec2, Vec2)> = strokes.iter().map(|s| bounds(s)).collect();
    // Letters, words and the lines of one label are closer than this;
    // separate labels and other marks are further apart.
    let gap = 3.0 * letter_size(strokes);
    // Union-find over overlapping, grown boxes.
    let mut parent: Vec<usize> = (0..strokes.len()).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for i in 0..boxes.len() {
        for j in i + 1..boxes.len() {
            let (a, b) = (boxes[i], boxes[j]);
            let apart = (a.0 - b.1).max(b.0 - a.1).max(Vec2::ZERO);
            if apart.max_element() <= gap {
                let (ri, rj) = (root(&mut parent, i), root(&mut parent, j));
                parent[ri.max(rj)] = ri.min(rj);
            }
        }
    }
    let mut blocks: Vec<(usize, Vec<Vec<Vec2>>)> = Vec::new();
    for (i, stroke) in strokes.iter().enumerate() {
        let r = root(&mut parent, i);
        match blocks.iter_mut().find(|(id, _)| *id == r) {
            Some((_, block)) => block.push(stroke.clone()),
            None => blocks.push((r, vec![stroke.clone()])),
        }
    }
    // A word has a few letters at least.
    blocks
        .into_iter()
        .map(|(_, block)| block)
        .filter(|block| block.len() >= 3)
        .collect()
}

/// About the height of a letter: drawn letters consist of a few strokes,
/// each a good part of the letter's height.
fn letter_size(strokes: &[Vec<Vec2>]) -> f32 {
    let mut sizes: Vec<f32> = strokes
        .iter()
        .map(|s| {
            let (lo, hi) = bounds(s);
            (hi - lo).max_element()
        })
        .collect();
    sizes.sort_by(f32::total_cmp);
    (sizes[sizes.len() / 2] * 1.3).max(0.5)
}

fn bounds(points: &[Vec2]) -> (Vec2, Vec2) {
    points
        .iter()
        .fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(lo, hi), &p| {
            (lo.min(p), hi.max(p))
        })
}

/// Draws `strokes` turned by `quarter_turns` × 90° counter-clockwise, scaled
/// so that a typical stroke is about [`GLYPH_PX`] tall.
pub fn render_strokes(strokes: &[Vec<Vec2>], quarter_turns: u32) -> Option<GrayImage> {
    if strokes.is_empty() {
        return None;
    }
    render_strokes_at(strokes, quarter_turns, GLYPH_PX / letter_size(strokes))
}

/// Draws `strokes` turned by `quarter_turns` × 90° counter-clockwise at
/// `scale` pixels per millimetre.
pub fn render_strokes_at(
    strokes: &[Vec<Vec2>],
    quarter_turns: u32,
    scale: f32,
) -> Option<GrayImage> {
    if strokes.is_empty() {
        return None;
    }
    let turn = |p: Vec2| match quarter_turns % 4 {
        0 => p,
        1 => Vec2::new(-p.y, p.x),
        2 => -p,
        _ => Vec2::new(p.y, -p.x),
    };
    let turned: Vec<Vec<Vec2>> = strokes
        .iter()
        .map(|s| s.iter().map(|&p| turn(p) * scale).collect())
        .collect();
    let (lo, hi) = turned
        .iter()
        .fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(lo, hi), s| {
            let (a, b) = bounds(s);
            (lo.min(a), hi.max(b))
        });
    let size = hi - lo + Vec2::splat(2.0 * MARGIN_PX);
    if size.x * size.y > 16e6 {
        return None; // scattered marks, not a label
    }
    let mut image = GrayImage {
        width: size.x.ceil() as u32,
        height: size.y.ceil() as u32,
        pixels: Vec::new(),
    };
    image.pixels = vec![255; (image.width * image.height) as usize];
    // Half the stroke width, in pixels.
    let radius = (STROKE_RADIUS * letter_size(strokes) * scale).max(1.0);
    for stroke in &turned {
        // Image rows run downwards; pattern y runs up.
        let to_image = |p: Vec2| Vec2::new(p.x - lo.x + MARGIN_PX, hi.y - p.y + MARGIN_PX);
        for pair in stroke.windows(2) {
            image.line(to_image(pair[0]), to_image(pair[1]), radius);
        }
        if stroke.len() == 1 {
            image.line(to_image(stroke[0]), to_image(stroke[0]), radius);
        }
    }
    Some(image)
}

/// Text that is pattern bookkeeping rather than a name.
fn is_annotation(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    [
        "grain",
        "fabric",
        "piece qty",
        "qty",
        "cut ",
        "fold",
        "size",
    ]
    .iter()
    .any(|w| lower.contains(w))
}

/// Scores one reading: confident letters count, garbage does not.
fn score(words: &[Word]) -> f32 {
    words
        .iter()
        .filter(|w| w.text.chars().filter(|c| c.is_alphanumeric()).count() >= 2)
        .map(|w| w.confidence * w.text.chars().filter(|c| c.is_alphanumeric()).count() as f32)
        .sum()
}

/// Words that pattern piece names are made of. OCR of plotter lettering
/// gets a letter wrong now and then ("FRUNT", "PARI"); a word that nearly
/// matches one of these is taken to be it.
const VOCABULARY: &[&str] = &[
    "ARMHOLE",
    "BACK",
    "BAND",
    "BELT",
    "BIB",
    "BINDING",
    "BODICE",
    "BOTTOM",
    "CENTER",
    "CENTRE",
    "COAT",
    "COLLAR",
    "CUFF",
    "DART",
    "DRESS",
    "FACING",
    "FLAP",
    "FLY",
    "FRONT",
    "GODET",
    "GORE",
    "GUSSET",
    "HALF",
    "HEM",
    "HOOD",
    "INNER",
    "INSERT",
    "JACKET",
    "LAPEL",
    "LEFT",
    "LEG",
    "LINING",
    "LOOP",
    "LOWER",
    "MAIN",
    "NECK",
    "NECKLINE",
    "OUTER",
    "PANEL",
    "PANT",
    "PART",
    "PIECE",
    "PLACKET",
    "POCKET",
    "RIGHT",
    "RUFFLE",
    "SHIRT",
    "SHOULDER",
    "SIDE",
    "SKIRT",
    "SLEEVE",
    "STAND",
    "STRAP",
    "TAB",
    "TOP",
    "TROUSER",
    "UNDER",
    "UPPER",
    "VENT",
    "VEST",
    "WAIST",
    "WAISTBAND",
    "WELT",
    "YOKE",
];

/// Letters OCR mixes up in thin plotter lettering; swapping them counts
/// half.
const CONFUSIONS: &[(char, char)] = &[
    ('I', 'L'),
    ('I', '1'),
    ('L', '1'),
    ('I', 'T'),
    ('I', 'J'),
    ('U', 'O'),
    ('U', 'D'),
    ('O', 'D'),
    ('O', '0'),
    ('C', 'G'),
    ('C', 'K'),
    ('C', 'L'),
    ('B', '8'),
    ('S', '5'),
    ('Z', '2'),
    ('R', 'P'),
    ('F', 'P'),
    ('E', 'F'),
    ('N', 'H'),
    ('V', 'Y'),
];

/// Edit distance with cheap confusable substitutions.
fn edit_distance(a: &str, b: &str) -> f32 {
    let b: Vec<char> = b.chars().collect();
    let substitution = |x: char, y: char| {
        if x == y {
            0.0
        } else if CONFUSIONS
            .iter()
            .any(|&(p, q)| (p, q) == (x, y) || (q, p) == (x, y))
        {
            0.5
        } else {
            1.0
        }
    };
    let mut row: Vec<f32> = (0..=b.len()).map(|j| j as f32).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut previous = row[0];
        row[0] = i as f32 + 1.0;
        for (j, &cb) in b.iter().enumerate() {
            let substitute = previous + substitution(ca, cb);
            previous = row[j + 1];
            row[j + 1] = substitute.min(row[j] + 1.0).min(row[j + 1] + 1.0);
        }
    }
    row[b.len()]
}

/// The vocabulary word `token` most likely is, if any: a near match, or a
/// cut-off start of a long word ("NECKLI" for "NECKLINE").
fn known_word(token: &str) -> Option<&'static str> {
    let upper = token.to_ascii_uppercase();
    if upper.len() < 3 || !upper.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    let allowed = if upper.len() >= 6 { 2.0 } else { 1.0 };
    VOCABULARY
        .iter()
        .map(|word| {
            let cut_off = upper.len() >= 4
                && word.len() > upper.len()
                && word.starts_with(&upper[..upper.len() - 1]);
            (
                word,
                if cut_off {
                    1.0
                } else {
                    edit_distance(&upper, word)
                },
            )
        })
        .filter(|&(_, distance)| distance <= allowed)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(word, _)| *word)
}

/// Splits a line into tokens at spaces and underscores, corrects near
/// misses and drops OCR crumbs ("|", "1").
fn clean_line(line: &str) -> Vec<(String, bool)> {
    line.split(|c: char| c.is_whitespace() || c == '_')
        .filter(|token| token.chars().filter(|c| c.is_alphanumeric()).count() >= 2)
        .map(|token| match known_word(token) {
            Some(word) => (word.to_owned(), true),
            None => (token.to_owned(), false),
        })
        .collect()
}

/// The best name in a reading: the line with the most known words (then
/// the most letters) that is not bookkeeping ("Fabric type", "STRAIGHT
/// GRAIN"), a bare number or the drawing program's mark.
fn name_from(words: &[Word]) -> Option<String> {
    let mut lines: Vec<(u32, Vec<&Word>)> = Vec::new();
    for word in words.iter().filter(|w| w.confidence >= 50.0) {
        match lines.iter_mut().find(|(line, _)| *line == word.line) {
            Some((_, list)) => list.push(word),
            None => lines.push((word.line, vec![word])),
        }
    }
    lines.sort_by_key(|(line, _)| *line);
    let texts: Vec<(u32, String)> = lines
        .into_iter()
        .map(|(line, words)| {
            (
                line / BLOCK_LINES,
                words
                    .iter()
                    .map(|w| w.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
            )
        })
        .filter(|(_, line)| !is_annotation(line))
        .collect();
    // A name broken over two lines ("BACK_NECKLI" / "NE_FACING") is joined
    // again: the next line in the same block starts with the missing rest.
    let mut joined: Vec<Vec<(String, bool)>> = Vec::new();
    let mut previous: Option<(u32, String)> = None;
    for (block, text) in texts {
        let mut tokens = clean_line(&text);
        if let (Some((_, missing)), Some(last)) = (&previous, joined.last_mut())
            && let Some((first, _)) = tokens.first()
            && !missing.is_empty()
            && missing.starts_with(&first.to_ascii_uppercase())
        {
            tokens.remove(0);
            last.extend(tokens);
            previous = None;
            continue;
        }
        previous = text
            .split(|c: char| c.is_whitespace() || c == '_')
            .rev()
            .find(|t| t.len() >= 2)
            .and_then(|raw| {
                let word = known_word(raw)?;
                let raw = raw.to_ascii_uppercase();
                let stem = &raw[..raw.len().saturating_sub(1)];
                word.strip_prefix(stem).map(|rest| {
                    (
                        block,
                        rest[(raw.len() - stem.len()).min(rest.len())..].to_owned(),
                    )
                })
            });
        joined.push(tokens);
    }
    joined
        .into_iter()
        .filter(|tokens| {
            tokens
                .iter()
                .map(|(t, _)| t.chars().filter(|c| c.is_alphabetic()).count())
                .sum::<usize>()
                >= 4
        })
        .max_by_key(|tokens| {
            let known = tokens.iter().filter(|(_, known)| *known).count();
            let letters: usize = tokens
                .iter()
                .map(|(t, _)| t.chars().filter(|c| c.is_alphabetic()).count())
                .sum();
            (known, letters)
        })
        .filter(|tokens| tokens.iter().any(|(_, known)| *known) || tokens.len() == 1)
        .map(|tokens| {
            tokens
                .into_iter()
                .map(|(t, _)| t)
                .collect::<Vec<_>>()
                .join(" ")
        })
}

/// Blocks with fewer strokes than this are marks or single letters, not
/// a name.
const MIN_NAME_STROKES: usize = 6;
/// OCR runs at the same time. Each loads its language model, so many at
/// once can exhaust memory.
const MAX_CONCURRENT_READS: usize = 4;
/// Line numbers of different text blocks are kept this far apart.
const BLOCK_LINES: u32 = 1_000_000;

/// Reads the drawn label of every piece that still has a numbered name.
/// Returns how many pieces were named.
///
/// Each text block of a piece is turned so that its long side (the
/// direction of writing) is horizontal, and all of them are stacked into
/// one image. That leaves two possibilities, upright or upside down, so a
/// piece costs two OCR runs.
pub fn name_pieces(pattern: &mut ImportedPattern, reader: &dyn TextReader) -> usize {
    let unnamed: Vec<usize> = (0..pattern.pieces.len())
        .filter(|&i| {
            let name = &pattern.pieces[i].name;
            name.strip_prefix("Piece ")
                .is_some_and(|n| n.chars().all(|c| c.is_ascii_digit()))
        })
        .collect();
    let images: Vec<(usize, GrayImage)> = unnamed
        .iter()
        .flat_map(|&piece| {
            label_images(pattern, piece)
                .into_iter()
                .map(move |image| (piece, image))
        })
        .collect();

    // A small pool of workers runs the readings.
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    let workers = std::thread::available_parallelism()
        .map_or(2, |n| n.get())
        .clamp(1, MAX_CONCURRENT_READS);
    std::thread::scope(|scope| {
        for _ in 0..workers.min(images.len()) {
            scope.spawn(|| {
                loop {
                    let job = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((piece, image)) = images.get(job) else {
                        break;
                    };
                    if let Some(reading) = reader.read(image)
                        && let Ok(mut results) = results.lock()
                    {
                        results.push((*piece, reading));
                    }
                }
            });
        }
    });

    let mut best: Vec<(usize, Vec<Word>)> = Vec::new();
    for (piece, reading) in results.into_inner().unwrap_or_default() {
        match best.iter_mut().find(|(p, _)| *p == piece) {
            Some((_, current)) if score(&reading) > score(current) => *current = reading,
            Some(_) => {}
            None => best.push((piece, reading)),
        }
    }
    let mut named = 0;
    for (piece, words) in best {
        if let Some(name) = name_from(&words) {
            pattern.pieces[piece].name = name;
            named += 1;
        }
    }
    named
}

/// The lettering of piece `index` as two images, upright and upside down:
/// every text block turned so that its long side is horizontal, stacked.
pub fn label_images(pattern: &ImportedPattern, index: usize) -> Vec<GrayImage> {
    let blocks: Vec<Vec<Vec<Vec2>>> = text_blocks(&piece_strokes(pattern, index))
        .into_iter()
        .filter(|b| b.len() >= MIN_NAME_STROKES)
        .collect();
    (0..2u32)
        .filter_map(|flip| {
            let parts: Vec<GrayImage> = blocks
                .iter()
                .filter_map(|block| {
                    let (lo, hi) =
                        block
                            .iter()
                            .fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(lo, hi), s| {
                                let (a, b) = bounds(s);
                                (lo.min(a), hi.max(b))
                            });
                    let upright = u32::from(hi.y - lo.y > hi.x - lo.x);
                    render_strokes(block, upright + 2 * flip)
                })
                .collect();
            stack(&parts)
        })
        .collect()
}

/// Stacks images vertically, left-aligned, with white space between them.
fn stack(images: &[GrayImage]) -> Option<GrayImage> {
    let width = images.iter().map(|i| i.width).max()?;
    let gap = 32;
    let height = images.iter().map(|i| i.height + gap).sum::<u32>();
    if u64::from(width) * u64::from(height) > 16_000_000 {
        return None;
    }
    let mut out = GrayImage {
        width,
        height,
        pixels: vec![255; (width * height) as usize],
    };
    let mut top = 0;
    for image in images {
        for y in 0..image.height {
            let row = &image.pixels[(y * image.width) as usize..((y + 1) * image.width) as usize];
            let at = ((top + y) * width) as usize;
            out.pixels[at..at + image.width as usize].copy_from_slice(row);
        }
        top += image.height + gap;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern_import::{RawSegment, RawVectorDrawing};

    #[test]
    fn tsv_rows_become_words_by_line() {
        let tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n\
                   5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t91.5\tCENTER_FRONT_PART\n\
                   5\t1\t1\t1\t2\t1\t0\t0\t10\t10\t88\t5017\n\
                   5\t1\t1\t1\t2\t2\t0\t0\t10\t10\t90\tVest\n\
                   4\t1\t1\t1\t1\t0\t0\t0\t10\t10\t-1\t\n";
        let words = parse_tsv(tsv);
        assert_eq!(words.len(), 3);
        assert_eq!(name_from(&words).as_deref(), Some("CENTER FRONT PART"));
        // Underscores separate words like spaces.
    }

    #[test]
    fn bookkeeping_lines_are_not_names() {
        let word = |text: &str, line| Word {
            text: text.to_owned(),
            confidence: 90.0,
            line,
        };
        let words = [
            word("STRAIGHT", 1),
            word("GRAIN", 1),
            word("Fabric", 2),
            word("type", 2),
            word("Side", 3),
            word("Back", 3),
        ];
        assert_eq!(name_from(&words).as_deref(), Some("SIDE BACK"));
    }

    #[test]
    fn misread_letters_are_corrected_and_crumbs_dropped() {
        let word = |text: &str, line, confidence| Word {
            text: text.to_owned(),
            confidence,
            line,
        };
        // What OCR made of the vest's labels at a poor resolution.
        let words = [
            word("SIDE", 1, 90.0),
            word("_FRUNT", 1, 80.0),
            word("PARI", 1, 80.0),
            word("5017", 2, 90.0),
            word("Vest", 2, 90.0),
        ];
        assert_eq!(name_from(&words).as_deref(), Some("SIDE FRONT PART"));
        let words = [
            word("BACK_NECKLI", 1, 90.0),
            word("|", 1, 90.0),
            word("1", 1, 90.0),
            word("LEKO", 2, 90.0),
            word("CAD", 2, 90.0),
        ];
        assert_eq!(name_from(&words).as_deref(), Some("BACK NECKLINE"));
        // The rest of a word on the next line of the same block is joined.
        let words = [
            word("BACK_NECKLI", 1, 90.0),
            word("NE_FACING", 2, 90.0),
            word("Fabric", 3, 90.0),
        ];
        assert_eq!(name_from(&words).as_deref(), Some("BACK NECKLINE FACING"));
        assert_eq!(known_word("SLUE"), Some("SIDE"));
        assert_eq!(known_word("BAUK"), Some("BACK"));
        assert_eq!(known_word("LEKO"), None);
    }

    /// A reader that only understands upright text: it "reads" the name if
    /// the image is wider than tall, as a one-line label drawn upright is.
    struct Upright;
    impl TextReader for Upright {
        fn read(&self, image: &GrayImage) -> Option<Vec<Word>> {
            let confidence = if image.width > image.height {
                95.0
            } else {
                20.0
            };
            Some(vec![Word {
                text: "YOKE".into(),
                confidence,
                line: 1,
            }])
        }
    }

    #[test]
    fn pieces_are_named_from_the_best_orientation() {
        let mut drawing = RawVectorDrawing::new();
        let mut square = RawSubpath::new(Vec2::ZERO);
        for to in [
            Vec2::new(200.0, 0.0),
            Vec2::splat(200.0),
            Vec2::new(0.0, 200.0),
            Vec2::ZERO,
        ] {
            square.segments.push(RawSegment::Line { to });
        }
        square.closed = true;
        drawing.subpaths.push(square);
        // A "label": a row of short vertical strokes, written bottom to top.
        for k in 0..8 {
            let mut stroke = RawSubpath::new(Vec2::new(100.0, 40.0 + 6.0 * k as f32));
            stroke.segments.push(RawSegment::Line {
                to: Vec2::new(104.0, 40.0 + 6.0 * k as f32),
            });
            drawing.subpaths.push(stroke);
        }
        let mut pattern = drawing.assemble_pattern(10.0);
        assert_eq!(pattern.pieces[0].name, "Piece 1");
        assert_eq!(piece_strokes(&pattern, 0).len(), 8);
        assert_eq!(name_pieces(&mut pattern, &Upright), 1);
        assert_eq!(pattern.pieces[0].name, "YOKE");
    }
}

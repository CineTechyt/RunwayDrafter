//! Prints what the pattern importer makes of a PDF or SVG file:
//!
//! ```sh
//! cargo run -p io_cad --example inspect_pattern -- pattern.pdf
//! ```

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: inspect_pattern <file.pdf|file.svg>")?;
    let bytes = std::fs::read(&path)?;
    let mut pattern = if path.to_ascii_lowercase().ends_with(".svg") {
        io_cad::import_pattern_svg(&bytes, &io_cad::SvgOptions::default())?
    } else {
        io_cad::import_pattern_pdf(&bytes, &io_cad::PdfOptions::default())?
    };
    println!(
        "{} pieces, {} open lines",
        pattern.pieces.len(),
        pattern.unassigned_open_lines.len()
    );
    // SVG=<file>: draw the pieces (blue), open lines (red), seams (green).
    if let Ok(out) = std::env::var("SVG") {
        let mut body = String::new();
        let mut path = |points: Vec<glam::Vec2>, colour: &str, width: f32| {
            let d: Vec<String> = points
                .iter()
                .map(|p| format!("{:.2},{:.2}", p.x, -p.y))
                .collect();
            body.push_str(&format!(
                "<polyline points=\"{}\" fill=\"none\" stroke=\"{colour}\" stroke-width=\"{width}\"/>\n",
                d.join(" ")
            ));
        };
        for piece in &pattern.pieces {
            for l in &piece.loops {
                let mut points = l.outline.flatten(0.5).points;
                points.push(points[0]);
                path(points, "blue", 0.8);
            }
        }
        for line in &pattern.unassigned_open_lines {
            path(line.polyline(0.5), "red", 0.8);
        }
        if let Some(layout) = &pattern.pages {
            for &(a, b) in &layout.seams {
                path(vec![a, b], "green", 0.4);
            }
        }
        std::fs::write(
            &out,
            format!(
                "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"-20 -1000 1300 1050\" width=\"1300\" height=\"1050\"><rect x=\"-20\" y=\"-1000\" width=\"1300\" height=\"1050\" fill=\"white\"/>{body}</svg>"
            ),
        )?;
    }
    if let Some(layout) = &pattern.pages {
        println!("{} (overlap {:?} mm)", layout.summary(), layout.overlap_mm);
    }
    // LABELS=<dir>: write each piece's drawn lettering as images, one per
    // orientation, to check what text recognition gets to see.
    if let Ok(dir) = std::env::var("LABELS") {
        std::fs::create_dir_all(&dir)?;
        for index in 0..pattern.pieces.len() {
            let strokes = io_cad::drawn_text::piece_strokes(&pattern, index);
            let blocks = io_cad::drawn_text::text_blocks(&strokes);
            for (flip, image) in io_cad::drawn_text::label_images(&pattern, index)
                .iter()
                .enumerate()
            {
                std::fs::write(
                    format!("{dir}/piece{}_label{flip}.pgm", index + 1),
                    image.to_pgm(),
                )?;
            }
            if let Some(image) = io_cad::drawn_text::render_strokes_at(&strokes, 0, 6.0) {
                std::fs::write(format!("{dir}/piece{}_all.pgm", index + 1), image.to_pgm())?;
            }
            for (b, block) in blocks.iter().enumerate() {
                for turns in 0..4 {
                    if let Some(image) = io_cad::drawn_text::render_strokes(block, turns) {
                        std::fs::write(
                            format!("{dir}/piece{}_block{b}_{turns}.pgm", index + 1),
                            image.to_pgm(),
                        )?;
                    }
                }
            }
            println!(
                "piece {}: {} label strokes in {} blocks",
                index + 1,
                strokes.len(),
                blocks.len()
            );
            for block in blocks.iter().filter(|b| b.len() >= 6) {
                let (lo, hi) = block.iter().flatten().fold(
                    (glam::Vec2::INFINITY, glam::Vec2::NEG_INFINITY),
                    |(lo, hi), &p| (lo.min(p), hi.max(p)),
                );
                println!(
                    "    block of {:>3} strokes, {:>5.1} x {:<5.1} mm",
                    block.len(),
                    hi.x - lo.x,
                    hi.y - lo.y
                );
            }
        }
    }
    if let Some(reader) = io_cad::drawn_text::TesseractCli::find() {
        let named = io_cad::drawn_text::name_pieces(&mut pattern, &reader);
        println!("named {named} pieces from drawn labels");
    }
    for piece in &pattern.pieces {
        println!(
            "{:?} allowance {:.1} mm",
            piece.name, piece.seam_allowance_mm
        );
        for lp in &piece.loops {
            let curves = lp
                .outline
                .segments()
                .iter()
                .filter(|s| !matches!(s, cad_core::Segment::Line))
                .count();
            let size = lp.bounds.1 - lp.bounds.0;
            println!(
                "   {:<26} {:>4} edges ({curves} curved)  {:>5.0} x {:<5.0} mm at ({:.0}, {:.0})",
                lp.role.label(),
                lp.outline.edge_count(),
                size.x,
                size.y,
                lp.bounds.0.x,
                lp.bounds.0.y
            );
            if std::env::var_os("SEGMENTS").is_some() {
                let outline = &lp.outline;
                let n = outline.edge_count() as usize;
                for e in 0..n {
                    use cad_core::geometry::Curve;
                    let curve = outline.curve(e as u32);
                    let next = outline.curve(((e + 1) % n) as u32);
                    let turn = curve
                        .tangent(1.0)
                        .normalize_or_zero()
                        .angle_to(next.tangent(0.0).normalize_or_zero())
                        .to_degrees();
                    let kind = match outline.segments()[e] {
                        cad_core::Segment::Line => "line",
                        cad_core::Segment::Quadratic { .. } => "quad",
                        cad_core::Segment::Cubic { .. } => "cubic",
                    };
                    println!(
                        "      {e:>3} {kind:<5} {:>6.1} mm, turn to next {turn:>6.1}°",
                        curve.length(0.1)
                    );
                }
            }
        }
    }
    Ok(())
}

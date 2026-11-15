//! Writes a weave's colour and height as 2 × 2 tiles, to look at the joins.
fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let name = args.get(1).map_or("Denim", String::as_str);
    let out = args.get(2).map_or("tile", String::as_str);
    let weave = cad_core::Weave::library()
        .into_iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name))
        .expect("weave")
        .1;
    let t = weave.texture(8);
    let (w, h) = (t.width as usize, t.height as usize);
    for (suffix, pick) in [("color", 0usize), ("height", 1)] {
        let mut bytes = format!("P6\n{} {}\n255\n", 2 * w, 2 * h).into_bytes();
        for y in 0..2 * h {
            for x in 0..2 * w {
                let i = (y % h) * w + x % w;
                let p = if pick == 0 { t.albedo[i] } else { [t.surface[i][3]; 4] };
                bytes.extend([p[0], p[1], p[2]]);
            }
        }
        std::fs::write(format!("{out}_{suffix}.ppm"), bytes)?;
    }
    Ok(())
}

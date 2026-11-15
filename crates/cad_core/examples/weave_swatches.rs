//! Writes the library's weaves as images, to look at the generated cloth:
//!
//! ```sh
//! cargo run -p cad_core --example weave_swatches -- out_dir
//! ```

fn main() -> std::io::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    std::fs::create_dir_all(&dir)?;
    // The library's drafts, and the woven fabric presets.
    let presets = cad_core::fabric_library::presets()
        .into_iter()
        .filter_map(|f| Some((f.name.clone(), f.weave?)));
    let library = cad_core::Weave::library().into_iter().map(|(n, w)| (n.to_owned(), w));
    for (name, weave) in library.chain(presets) {
        let texture = weave.preview(10);
        let mut bytes = format!("P6\n{} {}\n255\n", texture.width, texture.height).into_bytes();
        bytes.extend(texture.albedo.iter().flat_map(|c| [c[0], c[1], c[2]]));
        let file = format!("{dir}/{}.ppm", name.replace(['/', ' '], "_"));
        std::fs::write(&file, bytes)?;
        println!("{file}: {} × {} threads", texture.ends, texture.picks);
    }
    Ok(())
}

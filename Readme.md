cargo run                                        # the app (binary: clo360)
cargo test --workspace
cargo run -p gfx_render --example snapshot -- out   # headless viewport images (PPM)
cargo run -- --screenshot out.ppm --pattern pattern.pdf --tool sew   # the editor rendered offscreen, saved as an image
cargo run -- --screenshot out.ppm --open projects/waistcoat_with_buttons.clo360 --play --frames 150 --hair bob+fringe   # a sample project, draped
cargo run -- --screenshot out.ppm --animate yoga --at 13.4 --hair taperfade --man   # one frame of an animation (walk, stiff, sway, sit, yoga)
cargo run --release -- --screenshot out.ppm --new top --animate yoga --at 4.6,12.8 --simulate-garment --bake-seconds 16 --solver contact   # simulated garment through an animation (always baked with the contact solver; --solver picks the live one)
cargo run -p io_cad --example inspect_pattern -- pattern.pdf         # what the importer makes of a PDF/SVG
cargo run -p gfx_render --example sew_vest -- S-5017_Vest.pdf out    # import, sew and drape the S-5017 vest headless
# Piece names that a PDF draws as lines (not text) are read with Tesseract: sudo apt install tesseract-ocr

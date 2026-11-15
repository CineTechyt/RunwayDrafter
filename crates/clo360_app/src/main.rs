//! clo360 desktop entry point.

mod app;
mod body;
mod controllers;
mod garment_motion;
mod project;
mod screenshot;
mod session;
mod simulation;

use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> anyhow::Result<()> {
    // `RUST_LOG` overrides; by default show our own messages and wgpu/naga
    // problems without their per-frame chatter.
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=warn,naga=warn"),
    )
    .init();

    let args: Vec<String> = std::env::args().collect();
    // `--screenshot out.ppm` renders the editor without a window.
    if args.iter().any(|arg| arg == "--screenshot") {
        return screenshot::run(&args);
    }

    let event_loop = EventLoop::new()?;
    // Sleep until there is input; the app requests redraws while animating.
    event_loop.set_control_flow(ControlFlow::Wait);

    // `--play` starts simulating immediately (handy for demos and smoke tests);
    // `--avatar <file>` imports an avatar (OBJ, glTF, FBX) instead of the built-in one.
    let avatar = args
        .iter()
        .position(|arg| arg == "--avatar")
        .and_then(|at| args.get(at + 1))
        .map(std::path::PathBuf::from);
    let mut app = app::App::new(args.iter().any(|arg| arg == "--play"), avatar);
    event_loop.run_app(&mut app)?;
    app.into_result()
}

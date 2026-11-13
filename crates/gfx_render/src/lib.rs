//! `wgpu` rendering for the two viewports.
//!
//! Both viewports render into offscreen [`RenderTarget`]s rather than the
//! window surface. The UI layer shows those textures inside dock tabs, which
//! keeps this crate free of any windowing or `egui` dependency and makes the
//! renderers usable headless (thumbnails, tests, export).
//!
//! * [`canvas2d`] — Viewport A: the pattern editing canvas.
//! * [`scene3d`] — Viewport B: PBR view of the avatar and the garment mesh.

pub mod camera;
pub mod canvas2d;
pub mod context;
pub mod scene3d;
pub mod target;

pub use camera::{Camera2D, OrbitCamera};
pub use canvas2d::{CanvasBatch, CanvasOverlay, CanvasRenderer, LineHighlight};
pub use context::{GpuContext, GpuError, preferred_backends};
pub use scene3d::{
    MeshData, MeshVertex, SceneRenderer, Shading, SurfaceMaterial, SurfaceOverlay, garment_props,
};
pub use target::RenderTarget;

pub use wgpu;

/// A renderer that owns an offscreen target the UI can display.
pub trait ViewportRenderer {
    fn target(&self) -> &RenderTarget;

    /// Resizes the target to `size` physical pixels. Returns `true` if the
    /// textures were recreated, in which case anything holding the old
    /// [`RenderTarget::view`] (e.g. an `egui` texture binding) must rebind.
    fn resize(&mut self, gpu: &GpuContext, size: [u32; 2]) -> bool;

    /// Records the draw commands for one frame. Per-frame data is supplied
    /// beforehand through the concrete renderer's `prepare` method.
    fn render(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder);
}

/// Converts an sRGB colour with straight alpha to the linear values the
/// renderers work in.
pub fn srgba(r: u8, g: u8, b: u8, a: f32) -> [f32; 4] {
    let lin = |c: u8| {
        let c = c as f32 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    [lin(r), lin(g), lin(b), a]
}

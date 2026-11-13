//! A dock tab that shows a render-target texture and reports raw input.

use glam::Vec2;

/// Drag state of one pointer button during this frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ButtonDrag {
    pub started: bool,
    pub dragging: bool,
    pub stopped: bool,
}

/// Input for one viewport, in **physical pixels** relative to the viewport's
/// top-left corner, which is the coordinate system of its render target.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ViewportInput {
    /// Size the viewport occupies on screen; the render target should match.
    pub size: [u32; 2],
    pub hovered: bool,
    /// Current pointer position while hovered or dragging.
    pub pointer: Option<Vec2>,
    /// Where the button went down for the drag in progress. Drags only start
    /// after a small movement threshold, so tools should pick here.
    pub press_origin: Option<Vec2>,
    /// Pointer movement since the last frame while dragging.
    pub drag_delta: Vec2,
    pub primary: ButtonDrag,
    pub secondary: ButtonDrag,
    pub middle: ButtonDrag,
    pub clicked: bool,
    pub double_clicked: bool,
    /// Multiplicative zoom from the wheel or a pinch; 1.0 means none.
    pub zoom: f32,
    /// Shift is held (extends selections).
    pub shift: bool,
}

/// Fills the available space with `texture` and collects the input on it.
pub fn show(ui: &mut egui::Ui, texture: egui::TextureId, hint: &str) -> ViewportInput {
    let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
    let scale = ui.pixels_per_point();
    let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
    ui.painter().image(texture, rect, uv, egui::Color32::WHITE);
    ui.painter().text(
        rect.left_bottom() + egui::vec2(8.0, -6.0),
        egui::Align2::LEFT_BOTTOM,
        hint,
        egui::FontId::proportional(11.0),
        egui::Color32::from_white_alpha(110),
    );

    let local = |p: egui::Pos2| Vec2::new((p.x - rect.left()) * scale, (p.y - rect.top()) * scale);
    let drag = |button| ButtonDrag {
        started: response.drag_started_by(button),
        dragging: response.dragged_by(button),
        stopped: response.drag_stopped_by(button),
    };
    let hovered = response.contains_pointer();
    let any_drag = response.dragged() || response.drag_stopped();
    let (zoom, press_origin, shift) = ui.input(|input| {
        let zoom = if hovered { input.zoom_delta() * (input.smooth_scroll_delta.y * 0.0025).exp() } else { 1.0 };
        (zoom, input.pointer.press_origin(), input.modifiers.shift)
    });

    ViewportInput {
        size: [(rect.width() * scale).round().max(1.0) as u32, (rect.height() * scale).round().max(1.0) as u32],
        hovered,
        pointer: response.hover_pos().or(response.interact_pointer_pos()).map(local),
        press_origin: press_origin.filter(|_| any_drag || response.is_pointer_button_down_on()).map(local),
        drag_delta: Vec2::new(response.drag_delta().x, response.drag_delta().y) * scale,
        primary: drag(egui::PointerButton::Primary),
        secondary: drag(egui::PointerButton::Secondary),
        middle: drag(egui::PointerButton::Middle),
        clicked: response.clicked(),
        double_clicked: response.double_clicked(),
        zoom,
        shift,
    }
}

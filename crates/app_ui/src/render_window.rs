//! The render window: the studio render of the 3D view as it refines,
//! and saving it.

use crate::state::EditorState;

pub fn show(ctx: &egui::Context, editor: &mut EditorState) {
    let render = &mut editor.render;
    if !render.open {
        return;
    }
    let mut open = true;
    egui::Window::new("Render")
        .open(&mut open)
        .default_size([900.0, 760.0])
        .resizable(true)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .button("Render again")
                    .on_hover_text("Render the 3D view as it is now")
                    .clicked()
                {
                    render.start = true;
                }
                let (done, total) = render.samples;
                let finished = render.texture.is_some() && done >= total;
                if ui
                    .add_enabled(finished, egui::Button::new("Save PNG…"))
                    .clicked()
                    && let Some(path) = rfd::FileDialog::new()
                        .add_filter("PNG image", &["png"])
                        .set_file_name("render.png")
                        .save_file()
                {
                    render.save_to = Some(path.with_extension("png"));
                }
                ui.separator();
                ui.label("Depth of field");
                if ui
                    .add(egui::Slider::new(&mut render.depth_of_field, 0.0..=1.0).show_value(false))
                    .on_hover_text("How much the floor and backdrop blur behind the sharp figure")
                    .drag_stopped()
                {
                    render.start = true;
                }
                ui.separator();
                if render.rendering() {
                    ui.add(
                        egui::ProgressBar::new(done as f32 / total.max(1) as f32)
                            .desired_width(160.0)
                            .text(format!("{done} / {total} samples")),
                    );
                } else if render.texture.is_some() {
                    ui.weak(format!(
                        "{} × {} px, {total} samples",
                        render.size[0], render.size[1]
                    ));
                }
            });
            match &render.status {
                Some(Ok(message)) => {
                    ui.weak(message);
                }
                Some(Err(message)) => {
                    ui.colored_label(ui.visuals().error_fg_color, message);
                }
                None => {}
            }
            if let Some(texture) = render.texture {
                let [w, h] = render.size.map(|s| s.max(1) as f32);
                let available = ui.available_size();
                let scale = (available.x / w).min(available.y / h).max(0.05);
                ui.centered_and_justified(|ui| {
                    ui.image((texture, egui::vec2(w, h) * scale));
                });
            }
        });
    render.open = open;
}

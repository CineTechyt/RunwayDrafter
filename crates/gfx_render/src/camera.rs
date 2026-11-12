//! Cameras for the two viewports. Screen coordinates are physical pixels of
//! the viewport's render target, origin top-left, Y down.

use glam::camera::rh::{proj, view};
use glam::{Mat4, Vec2, Vec3};

/// Orthographic camera of the pattern canvas (world = mm, Y up).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera2D {
    /// World point at the centre of the viewport.
    pub center: Vec2,
    /// Pixels per millimetre.
    pub zoom: f32,
}

impl Default for Camera2D {
    fn default() -> Self {
        Self { center: Vec2::ZERO, zoom: 1.0 }
    }
}

impl Camera2D {
    pub const MIN_ZOOM: f32 = 0.02;
    pub const MAX_ZOOM: f32 = 200.0;

    pub fn view_proj(&self, viewport: Vec2) -> Mat4 {
        let scale = 2.0 * self.zoom / viewport;
        Mat4::from_scale(scale.extend(1.0)) * Mat4::from_translation((-self.center).extend(0.0))
    }

    pub fn screen_to_world(&self, screen: Vec2, viewport: Vec2) -> Vec2 {
        self.center + (screen - viewport * 0.5) * Vec2::new(1.0, -1.0) / self.zoom
    }

    pub fn world_to_screen(&self, world: Vec2, viewport: Vec2) -> Vec2 {
        (world - self.center) * self.zoom * Vec2::new(1.0, -1.0) + viewport * 0.5
    }

    /// World length of one pixel.
    pub fn pixel_size(&self) -> f32 {
        1.0 / self.zoom
    }

    /// Visible world rectangle `(min, max)`.
    pub fn visible(&self, viewport: Vec2) -> (Vec2, Vec2) {
        let half = viewport * 0.5 / self.zoom;
        (self.center - half, self.center + half)
    }

    /// Drags the canvas by a screen-space delta.
    pub fn pan(&mut self, delta_px: Vec2) {
        self.center -= delta_px * Vec2::new(1.0, -1.0) / self.zoom;
    }

    /// Zooms by `factor`, keeping the world point under `screen` fixed.
    pub fn zoom_about(&mut self, screen: Vec2, viewport: Vec2, factor: f32) {
        let before = self.screen_to_world(screen, viewport);
        self.zoom = (self.zoom * factor).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
        self.center += before - self.screen_to_world(screen, viewport);
    }

    /// Frames the world rectangle with a margin.
    pub fn fit(&mut self, min: Vec2, max: Vec2, viewport: Vec2) {
        let size = (max - min).max(Vec2::splat(1.0));
        self.center = (min + max) * 0.5;
        self.zoom = ((viewport / size).min_element() * 0.85).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
    }
}

/// Perspective turntable camera of the 3D viewport (world = metres, Y up).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitCamera {
    pub target: Vec3,
    /// Rotation around the vertical axis, radians.
    pub yaw: f32,
    /// Elevation above the horizon, radians.
    pub pitch: f32,
    pub distance: f32,
    pub fov_y: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self { target: Vec3::new(0.0, 1.0, 0.0), yaw: 0.5, pitch: 0.12, distance: 3.2, fov_y: 40f32.to_radians() }
    }
}

impl OrbitCamera {
    const PITCH_LIMIT: f32 = 1.54;

    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + self.distance * Vec3::new(sy * cp, sp, cy * cp)
    }

    pub fn view(&self) -> Mat4 {
        view::look_at_mat4(self.eye(), self.target, Vec3::Y)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let near = (self.distance * 0.02).max(0.005);
        // wgpu clip space is Y up with depth 0..1 on every backend, which is
        // glam's "directx" convention (its "vulkan" one flips Y).
        proj::directx::perspective(self.fov_y, aspect.max(1e-3), near, near * 5000.0) * self.view()
    }

    pub fn orbit(&mut self, delta_px: Vec2) {
        self.yaw -= delta_px.x * 0.008;
        self.pitch = (self.pitch + delta_px.y * 0.008).clamp(-Self::PITCH_LIMIT, Self::PITCH_LIMIT);
    }

    /// Slides the target in the view plane so the scene follows the cursor.
    pub fn pan(&mut self, delta_px: Vec2, viewport_height: f32) {
        let world_per_px = 2.0 * self.distance * (self.fov_y * 0.5).tan() / viewport_height.max(1.0);
        let forward = (self.target - self.eye()).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward);
        self.target += (-right * delta_px.x + up * delta_px.y) * world_per_px;
    }

    pub fn dolly(&mut self, factor: f32) {
        self.distance = (self.distance / factor).clamp(0.2, 50.0);
    }

    /// World-space ray `(origin, direction)` through a screen position.
    pub fn ray(&self, screen: Vec2, viewport: Vec2) -> (Vec3, Vec3) {
        let ndc = Vec2::new(2.0 * screen.x / viewport.x - 1.0, 1.0 - 2.0 * screen.y / viewport.y);
        let inverse = self.view_proj(viewport.x / viewport.y).inverse();
        let near = inverse.project_point3(ndc.extend(0.0));
        let far = inverse.project_point3(ndc.extend(1.0));
        (near, (far - near).normalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera2d_round_trips_and_zooms_about_the_cursor() {
        let viewport = Vec2::new(800.0, 600.0);
        let mut cam = Camera2D { center: Vec2::new(40.0, -10.0), zoom: 2.5 };
        let screen = Vec2::new(123.0, 456.0);
        let world = cam.screen_to_world(screen, viewport);
        assert!(cam.world_to_screen(world, viewport).distance(screen) < 1e-3);
        cam.zoom_about(screen, viewport, 1.7);
        assert!(cam.screen_to_world(screen, viewport).distance(world) < 1e-3);
        // The projection agrees with the screen mapping.
        let ndc = cam.view_proj(viewport).project_point3(world.extend(0.0));
        let expected = Vec2::new(2.0 * screen.x / viewport.x - 1.0, 1.0 - 2.0 * screen.y / viewport.y);
        assert!(ndc.truncate().distance(expected) < 1e-4);
    }

    #[test]
    fn orbit_ray_through_the_centre_hits_the_target() {
        let cam = OrbitCamera::default();
        let viewport = Vec2::new(640.0, 480.0);
        let (origin, dir) = cam.ray(viewport * 0.5, viewport);
        let to_target = (cam.target - origin).normalize();
        assert!(dir.dot(to_target) > 0.9999);
    }
}

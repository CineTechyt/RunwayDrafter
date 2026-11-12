//! Viewport tools: translate raw viewport input into camera moves, selection
//! and document edits.

use app_ui::{EditorState, ViewportInput};
use cad_core::{DocumentChange, HandleRef, History, MoveHandle, PatternDocument};
use gfx_render::{Camera2D, OrbitCamera};
use glam::Vec2;
use topology_sync::{GarmentMesh, SurfaceLocation, TopologySync};

/// Pick radius around handles, in pixels.
const PICK_RADIUS_PX: f32 = 9.0;

struct HandleDrag {
    handle: HandleRef,
    /// Position when the drag began, for the undo record.
    from: Vec2,
    /// Handle position minus the grab point, so the handle does not jump to
    /// the cursor.
    grab_offset: Vec2,
}

/// The 2D canvas edit tool: pan/zoom, select, drag anchors and controls.
#[derive(Default)]
pub struct CanvasTool {
    pub hovered: Option<HandleRef>,
    drag: Option<HandleDrag>,
    panning: bool,
}

impl CanvasTool {
    /// The handle being dragged, if any.
    pub fn active(&self) -> Option<HandleRef> {
        self.drag.as_ref().map(|d| d.handle)
    }

    pub fn update(
        &mut self,
        input: &ViewportInput,
        camera: &mut Camera2D,
        document: &mut PatternDocument,
        history: &mut History,
        editor: &mut EditorState,
        changes: &mut Vec<DocumentChange>,
    ) {
        let viewport = Vec2::new(input.size[0] as f32, input.size[1] as f32);

        if let Some(pointer) = input.pointer
            && input.zoom != 1.0
        {
            camera.zoom_about(pointer, viewport, input.zoom);
        }
        if input.secondary.dragging || input.middle.dragging {
            camera.pan(input.drag_delta);
        }

        let radius = PICK_RADIUS_PX * camera.pixel_size();
        let world = |screen: Vec2, camera: &Camera2D| camera.screen_to_world(screen, viewport);

        if input.primary.started && !editor.sewing_mode {
            // Pick where the button went down, not where the drag threshold
            // was crossed.
            let origin = input
                .press_origin
                .or(input.pointer)
                .map(|p| world(p, camera));
            if let Some(at) = origin {
                if let Some(handle) = document.pick_handle(at, radius) {
                    let from = document.handle_position(handle).unwrap_or(at);
                    self.drag = Some(HandleDrag {
                        handle,
                        from,
                        grab_offset: from - at,
                    });
                    editor.selected_piece = Some(handle.piece);
                    editor.selected_edge = None;
                } else if let Some(edge) = document.pick_edge(at, radius * 1.5) {
                    editor.selected_piece = Some(edge.piece);
                    editor.selected_edge = Some(edge);
                } else if let Some(piece) = document.pick_piece(at) {
                    editor.selected_piece = Some(piece);
                    editor.selected_edge = None;
                } else {
                    self.panning = true;
                    editor.selected_edge = None;
                }
            } else {
                self.panning = true;
            }
        }

        if input.primary.dragging {
            if let (Some(drag), Some(pointer)) = (&self.drag, input.pointer) {
                let target = world(pointer, camera) + drag.grab_offset;
                let moved = document.handle_position(drag.handle) != Some(target)
                    && document.piece_mut(drag.handle.piece).is_some_and(|piece| {
                        piece.outline.set_handle_position(drag.handle.kind, target)
                    });
                if moved {
                    document.touch();
                    changes.push(DocumentChange::PieceGeometry(drag.handle.piece));
                }
            } else if self.panning {
                camera.pan(input.drag_delta);
            }
        }

        if input.primary.stopped {
            self.panning = false;
            // The document already holds the result; record one undo step
            // for the whole gesture.
            if let Some(drag) = self.drag.take()
                && let Some(to) = document.handle_position(drag.handle)
                && to != drag.from
            {
                history.record(Box::new(MoveHandle {
                    handle: drag.handle,
                    from: drag.from,
                    to,
                }));
            }
        }

        if input.clicked
            && let Some(at) = input.pointer.map(|p| world(p, camera))
        {
            if editor.sewing_mode {
                if let Some(edge) = document.pick_edge(at, radius * 1.5) {
                    editor.selected_piece = Some(edge.piece);
                    editor.selected_edge = Some(edge);
                    if document.seam_of(edge).is_none() {
                        let selection = if editor.sewing_select_b {
                            &mut editor.sewing_edges_b
                        } else {
                            &mut editor.sewing_edges_a
                        };
                        if let Some(index) = selection.iter().position(|selected| *selected == edge) {
                            selection.remove(index);
                        } else {
                            selection.push(edge);
                        }
                    }
                }
            } else if let Some(handle) = document.pick_handle(at, radius) {
                editor.selected_piece = Some(handle.piece);
                editor.selected_edge = None;
            } else if let Some(edge) = document.pick_edge(at, radius * 1.5) {
                editor.selected_piece = Some(edge.piece);
                editor.selected_edge = Some(edge);
            } else {
                editor.selected_piece = document.pick_piece(at);
                editor.selected_edge = None;
            }
        }

        self.hovered = match (&self.drag, input.pointer) {
            (Some(drag), _) => Some(drag.handle),
            (None, Some(pointer)) if input.hovered => {
                document.pick_handle(world(pointer, camera), radius)
            }
            _ => None,
        };
    }
}

/// The 3D viewport tool: orbit/pan/dolly and click-to-select through the
/// 3D → 2D mapping.
pub fn update_scene(
    input: &ViewportInput,
    camera: &mut OrbitCamera,
    topology: &impl TopologySync,
    editor: &mut EditorState,
) {
    let viewport = Vec2::new(input.size[0] as f32, input.size[1] as f32);
    if input.primary.dragging {
        camera.orbit(input.drag_delta);
    }
    if input.secondary.dragging || input.middle.dragging {
        camera.pan(input.drag_delta, viewport.y);
    }
    if input.zoom != 1.0 {
        camera.dolly(input.zoom);
    }
    if input.clicked
        && let Some(pointer) = input.pointer
    {
        let (origin, direction) = camera.ray(pointer, viewport);
        editor.selected_piece = pick_piece(topology, topology.mesh(), origin, direction);
    }
}

fn pick_piece(
    topology: &impl TopologySync,
    mesh: &GarmentMesh,
    origin: glam::Vec3,
    direction: glam::Vec3,
) -> Option<cad_core::PieceId> {
    let hit = mesh.raycast(origin, direction)?;
    let location = SurfaceLocation {
        triangle: hit.triangle,
        barycentric: hit.barycentric,
    };
    // `surface_to_pattern` also yields the exact pattern-space point under
    // the cursor: the hook for 3D tools such as drawing internal lines.
    topology.surface_to_pattern(location).map(|at| at.piece)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_core::geometry::Curve;

    fn click(point: Vec2) -> ViewportInput {
        ViewportInput {
            size: [1200, 1100],
            hovered: true,
            pointer: Some(point),
            clicked: true,
            zoom: 1.0,
            ..Default::default()
        }
    }

    fn click_edge(document: &PatternDocument, camera: &Camera2D, piece: cad_core::PieceId, edge: u32) -> ViewportInput {
        let viewport = Vec2::new(1200.0, 1100.0);
        let point = document.piece(piece).unwrap().outline.curve(edge).eval(0.5);
        click(camera.world_to_screen(point, viewport))
    }

    #[test]
    fn sewing_mode_groups_multiple_canvas_edges() {
        let mut document = PatternDocument::demo_top();
        let front = document.pieces()[0].id;
        let back = document.pieces()[1].id;
        let mut editor = EditorState {
            sewing_mode: true,
            sewing_reversed: true,
            ..Default::default()
        };
        let mut tool = CanvasTool::default();
        let mut camera = Camera2D::default();
        let mut history = History::default();
        let mut changes = Vec::new();

        for edge in [0, 4] {
            tool.update(
                &click_edge(&document, &camera, front, edge),
                &mut camera,
                &mut document,
                &mut history,
                &mut editor,
                &mut changes,
            );
        }
        assert_eq!(
            editor.sewing_edges_a,
            vec![cad_core::EdgeRef::new(front, 0), cad_core::EdgeRef::new(front, 4)]
        );
        editor.sewing_select_b = true;

        for edge in [0, 4] {
            tool.update(
                &click_edge(&document, &camera, back, edge),
                &mut camera,
                &mut document,
                &mut history,
                &mut editor,
                &mut changes,
            );
        }

        assert_eq!(
            editor.sewing_edges_b,
            vec![cad_core::EdgeRef::new(back, 0), cad_core::EdgeRef::new(back, 4)]
        );
        for (a, b) in editor.sewing_edges_a.iter().zip(&editor.sewing_edges_b) {
            document.add_seam(*a, *b, editor.sewing_reversed);
        }
        assert_eq!(document.seams().len(), 6);
        let seam = document.seam_of(cad_core::EdgeRef::new(front, 0)).unwrap();
        assert_eq!(seam.b, cad_core::EdgeRef::new(back, 0));
        assert!(seam.reversed);
        assert!(changes.is_empty());
    }
}

//! The session's avatar: the body model, the current body built from it
//! and the collision field, which is voxelised on a worker thread.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

use avatar::{BodyMesh, BodyModel, BodyParams, Measurements, PoseParams, PosedBody};
use glam::{Affine3A, Vec3};
use physics_collision::TriangleMesh;
use topology_sync::DistanceField;

/// Grid spacing of the collision field in metres.
const FIELD_VOXEL: f32 = 0.008;

/// A body frozen at one moment; see [`Body::snapshot`].
pub struct BodySnapshot {
    rest: BodyMesh,
    posed: BodyMesh,
    transforms: Vec<Affine3A>,
}

pub struct Body {
    model: BodyModel,
    /// Parameters the meshes were built with.
    pub params: BodyParams,
    pub pose: PoseParams,
    /// Radius the collision field is closed with, in metres.
    closing: f32,
    /// The body in the modelled pose: what measurements are taken from.
    rest: BodyMesh,
    /// The posed body: what is shown, dressed and collided with.
    pub mesh: BodyMesh,
    /// Per vertex, what took it from `rest` to `mesh`.
    transforms: Vec<Affine3A>,
    pub measurements: Measurements,
    /// The latest finished collision field. While the avatar is being
    /// edited this lags a few frames behind `mesh`.
    pub field: Option<Arc<DistanceField>>,
    /// The same body as triangles behind a BVH: continuous collision and
    /// the fit map. Always of the same pose as `field`.
    pub collider: Option<Arc<TriangleMesh>>,
    pending: Option<Receiver<(DistanceField, TriangleMesh)>>,
    /// `mesh` changed after the pending voxelisation started.
    field_outdated: bool,
}

impl Body {
    pub fn new(model: BodyModel, params: BodyParams, pose: PoseParams, closing: f32) -> Self {
        let rest = model.build(&params);
        let measurements = avatar::measure(&model, &rest);
        let (mesh, transforms) = model.pose_with_transforms(&rest, &pose.to_pose(&model, &rest));
        // The first field is built synchronously so the very first
        // simulation step already collides.
        let field = Some(Arc::new(avatar::distance_field(&mesh, FIELD_VOXEL, closing)));
        let collider = Some(Arc::new(mesh.collider()));
        Self { model, params, pose, closing, rest, mesh, transforms, measurements, field, collider, pending: None, field_outdated: false }
    }

    pub fn model(&self) -> &BodyModel {
        &self.model
    }

    /// Arm and leg angles of the modelled pose, in degrees.
    pub fn modelled_angles(&self) -> (f32, f32) {
        PoseParams::modelled_angles(&self.model, &self.rest)
    }

    /// Rebuilds the body for new shape parameters.
    pub fn set_params(&mut self, params: BodyParams) {
        self.params = params;
        self.rest = self.model.build(&params);
        self.measurements = avatar::measure(&self.model, &self.rest);
        self.repose();
    }

    pub fn set_pose(&mut self, pose: PoseParams) {
        self.pose = pose;
        self.repose();
    }

    /// Changes how much the collision field smooths over small details.
    pub fn set_closing(&mut self, closing: f32) {
        if (closing - self.closing).abs() > 1e-5 {
            self.closing = closing;
            self.field_outdated = true;
        }
    }

    fn repose(&mut self) {
        (self.mesh, self.transforms) = self.model.pose_with_transforms(&self.rest, &self.pose.to_pose(&self.model, &self.rest));
        self.field_outdated = true;
    }

    /// A copy of the body as it stands now, to carry things from later.
    pub fn snapshot(&self) -> BodySnapshot {
        BodySnapshot { rest: self.rest.clone(), posed: self.mesh.clone(), transforms: self.transforms.clone() }
    }

    /// Moves `points` from around the body as it was in `before` to the
    /// same place on the body as it is now.
    pub fn carry_from(&self, before: &BodySnapshot, points: &mut [Vec3]) {
        avatar::carry(
            points,
            PosedBody { rest: &before.rest, posed: &before.posed, transforms: &before.transforms },
            PosedBody { rest: &self.rest, posed: &self.mesh, transforms: &self.transforms },
        );
    }

    /// Starts and collects background voxelisations. Returns `true` when a
    /// new field became available.
    pub fn poll_field(&mut self) -> bool {
        let mut updated = false;
        if let Some(receiver) = &self.pending {
            match receiver.try_recv() {
                Ok((field, collider)) => {
                    self.field = Some(Arc::new(field));
                    self.collider = Some(Arc::new(collider));
                    self.pending = None;
                    updated = true;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.pending = None,
            }
        }
        if self.field_outdated && self.pending.is_none() {
            self.field_outdated = false;
            let (sender, receiver) = channel();
            let (mesh, closing) = (self.mesh.clone(), self.closing);
            std::thread::spawn(move || {
                // The receiver is gone if the session ended meanwhile.
                let _ = sender.send((avatar::distance_field(&mesh, FIELD_VOXEL, closing), mesh.collider()));
            });
            self.pending = Some(receiver);
        }
        updated
    }

    /// Whether a voxelisation is still running or due.
    pub fn field_pending(&self) -> bool {
        self.pending.is_some() || self.field_outdated
    }
}

/// An avatar imported from a file: fixed geometry, no skeleton, no
/// measurements to edit.
pub struct ImportedAvatar {
    pub mesh: BodyMesh,
    /// Estimated from proportions; enough to arrange a garment.
    pub measurements: Measurements,
    pub field: Arc<DistanceField>,
    pub collider: Arc<TriangleMesh>,
    closing: f32,
}

impl ImportedAvatar {
    pub fn new(imported: io_cad::ImportedMesh, closing: f32) -> Self {
        let mesh = BodyMesh::from_triangles(imported.positions, imported.normals, imported.triangles);
        Self {
            measurements: Measurements::rough(&mesh),
            field: Arc::new(avatar::distance_field(&mesh, FIELD_VOXEL, closing)),
            collider: Arc::new(mesh.collider()),
            mesh,
            closing,
        }
    }

    /// Rebuilds the collision field for another cloth resolution.
    pub fn set_closing(&mut self, closing: f32) {
        if (closing - self.closing).abs() > 1e-5 {
            self.closing = closing;
            self.field = Arc::new(avatar::distance_field(&self.mesh, FIELD_VOXEL, closing));
        }
    }
}

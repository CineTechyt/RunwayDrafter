//! Change notification and undoable editing.
//!
//! Every mutation of the document is described by a [`DocumentChange`]. The
//! topology layer consumes these to decide how much of the 3D mesh must be
//! rebuilt, which is what keeps 2D → 3D updates incremental.

use glam::Vec2;
use thiserror::Error;

use crate::document::PatternDocument;
use crate::ids::{FabricId, HandleRef, PieceId};

/// What changed, at the granularity downstream consumers care about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentChange {
    /// The outline moved: the piece must be re-meshed.
    PieceGeometry(PieceId),
    /// Notches, grainline, seam allowance, name: 2D drawing only.
    PieceAnnotations(PieceId),
    /// Texture mapping changed: UVs must be recomputed, topology is intact.
    PieceUv(PieceId),
    /// The 3D placement changed: positions reset, topology is intact.
    PieceArrangement(PieceId),
    /// The piece now uses another fabric.
    PieceFabric(PieceId),
    PieceAdded(PieceId),
    PieceRemoved(PieceId),
    /// Seams were added, removed or re-targeted.
    Seams,
    /// Fabric appearance or physics changed.
    Fabric(FabricId),
    /// Anything may have changed (e.g. a file was loaded).
    Everything,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EditError {
    #[error("piece {0:?} does not exist")]
    MissingPiece(PieceId),
    #[error("handle {0:?} does not exist")]
    MissingHandle(HandleRef),
}

/// An undoable document mutation.
pub trait EditCommand: std::fmt::Debug {
    fn label(&self) -> &str;
    fn apply(&mut self, doc: &mut PatternDocument) -> Result<DocumentChange, EditError>;
    fn undo(&mut self, doc: &mut PatternDocument) -> Result<DocumentChange, EditError>;
}

/// Moves an anchor or Bézier control from one position to another.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveHandle {
    pub handle: HandleRef,
    pub from: Vec2,
    pub to: Vec2,
}

impl MoveHandle {
    fn set(&self, doc: &mut PatternDocument, position: Vec2) -> Result<DocumentChange, EditError> {
        let piece = doc
            .piece_mut(self.handle.piece)
            .ok_or(EditError::MissingPiece(self.handle.piece))?;
        if !piece
            .outline
            .set_handle_position(self.handle.kind, position)
        {
            return Err(EditError::MissingHandle(self.handle));
        }
        doc.touch();
        Ok(DocumentChange::PieceGeometry(self.handle.piece))
    }
}

impl EditCommand for MoveHandle {
    fn label(&self) -> &str {
        "Move point"
    }
    fn apply(&mut self, doc: &mut PatternDocument) -> Result<DocumentChange, EditError> {
        self.set(doc, self.to)
    }
    fn undo(&mut self, doc: &mut PatternDocument) -> Result<DocumentChange, EditError> {
        self.set(doc, self.from)
    }
}

/// Any edit, recorded as the whole document before and after it. Patterns
/// are small, so this is the simplest correct undo for operations that
/// touch several things at once (turning a piece re-points its seams).
#[derive(Debug, Clone)]
pub struct DocumentSnapshot {
    label: String,
    before: PatternDocument,
    after: PatternDocument,
}

impl DocumentSnapshot {
    pub fn new(label: impl Into<String>, before: PatternDocument, after: PatternDocument) -> Self {
        Self {
            label: label.into(),
            before,
            after,
        }
    }
}

impl EditCommand for DocumentSnapshot {
    fn label(&self) -> &str {
        &self.label
    }
    fn apply(&mut self, doc: &mut PatternDocument) -> Result<DocumentChange, EditError> {
        *doc = self.after.clone();
        doc.touch();
        Ok(DocumentChange::Everything)
    }
    fn undo(&mut self, doc: &mut PatternDocument) -> Result<DocumentChange, EditError> {
        *doc = self.before.clone();
        doc.touch();
        Ok(DocumentChange::Everything)
    }
}

/// Linear undo/redo stack.
#[derive(Debug, Default)]
pub struct History {
    undo: Vec<Box<dyn EditCommand>>,
    redo: Vec<Box<dyn EditCommand>>,
}

impl History {
    /// Applies `command` and records it.
    pub fn execute(
        &mut self,
        doc: &mut PatternDocument,
        mut command: Box<dyn EditCommand>,
    ) -> Result<DocumentChange, EditError> {
        let change = command.apply(doc)?;
        self.record(command);
        Ok(change)
    }

    /// Records a command whose effect is already in the document. Interactive
    /// drags mutate live and commit a single command on release.
    pub fn record(&mut self, command: Box<dyn EditCommand>) {
        self.undo.push(command);
        self.redo.clear();
    }

    pub fn undo(&mut self, doc: &mut PatternDocument) -> Option<Result<DocumentChange, EditError>> {
        let mut command = self.undo.pop()?;
        let result = command.undo(doc);
        self.redo.push(command);
        Some(result)
    }

    pub fn redo(&mut self, doc: &mut PatternDocument) -> Option<Result<DocumentChange, EditError>> {
        let mut command = self.redo.pop()?;
        let result = command.apply(doc);
        self.undo.push(command);
        Some(result)
    }

    pub fn undo_label(&self) -> Option<&str> {
        self.undo.last().map(|c| c.label())
    }

    pub fn redo_label(&self) -> Option<&str> {
        self.redo.last().map(|c| c.label())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::HandleKind;

    #[test]
    fn move_handle_round_trips_through_history() {
        let mut doc = PatternDocument::demo_top();
        let piece = doc.pieces()[0].id;
        let handle = HandleRef {
            piece,
            kind: HandleKind::Anchor(0),
        };
        let from = doc.handle_position(handle).unwrap();
        let to = from + Vec2::new(12.0, 7.0);

        let mut history = History::default();
        let change = history
            .execute(&mut doc, Box::new(MoveHandle { handle, from, to }))
            .unwrap();
        assert_eq!(change, DocumentChange::PieceGeometry(piece));
        assert_eq!(doc.handle_position(handle), Some(to));

        history.undo(&mut doc).unwrap().unwrap();
        assert_eq!(doc.handle_position(handle), Some(from));
        history.redo(&mut doc).unwrap().unwrap();
        assert_eq!(doc.handle_position(handle), Some(to));
        assert!(history.redo(&mut doc).is_none());
    }
}

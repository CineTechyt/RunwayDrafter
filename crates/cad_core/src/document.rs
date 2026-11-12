//! The pattern document: pieces, fabrics and the seams that sew them together.

use glam::{Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::geometry;
use crate::ids::{EdgeRef, FabricId, HandleRef, PieceId, SeamId};
use crate::material::Fabric;
use crate::pattern::{ControlPoint, Notch, NotchKind, Outline, PatternPiece, Segment};

/// Sews edge `a` to edge `b`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Seam {
    pub id: SeamId,
    pub a: EdgeRef,
    pub b: EdgeRef,
    /// When `true`, the start of `a` meets the *end* of `b`.
    pub reversed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatternDocument {
    pieces: Vec<PatternPiece>,
    fabrics: Vec<Fabric>,
    seams: Vec<Seam>,
    next_id: u32,
    /// Bumped on every mutation; lets consumers detect staleness cheaply.
    revision: u64,
}

impl PatternDocument {
    pub fn new() -> Self {
        Self::default()
    }

    fn alloc_id(&mut self) -> u32 {
        self.next_id += 1;
        self.next_id
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Marks the document as changed. Call after mutating through
    /// [`Self::piece_mut`] or [`Self::fabric_mut`].
    pub fn touch(&mut self) {
        self.revision += 1;
    }

    // --- fabrics -----------------------------------------------------------

    pub fn add_fabric(&mut self, name: impl Into<String>, base_color: [f32; 3]) -> FabricId {
        let id = FabricId(self.alloc_id());
        self.fabrics.push(Fabric::new(id, name, base_color));
        self.touch();
        id
    }

    pub fn fabrics(&self) -> &[Fabric] {
        &self.fabrics
    }

    pub fn fabric(&self, id: FabricId) -> Option<&Fabric> {
        self.fabrics.iter().find(|f| f.id == id)
    }

    pub fn fabric_mut(&mut self, id: FabricId) -> Option<&mut Fabric> {
        self.fabrics.iter_mut().find(|f| f.id == id)
    }

    // --- pieces ------------------------------------------------------------

    pub fn add_piece(&mut self, name: impl Into<String>, outline: Outline, fabric: FabricId) -> PieceId {
        let id = PieceId(self.alloc_id());
        self.pieces.push(PatternPiece::new(id, name, outline, fabric));
        self.touch();
        id
    }

    /// Duplicates a piece beside its source, optionally mirrored across its vertical centreline.
    pub fn duplicate_piece(&mut self, source_id: PieceId, mirrored: bool) -> Option<PieceId> {
        let mut copy = self.piece(source_id)?.clone();
        let new_id = PieceId(self.alloc_id());
        let (min, max) = copy.outline.bounds();
        let axis_x = (min.x + max.x) * 0.5;
        let offset = Vec2::new((max.x - min.x).max(100.0) + 30.0, 0.0);

        if mirrored {
            let edge_map = copy.outline.mirror_x_about(axis_x);
            copy.seam_allowance.per_edge = copy
                .seam_allowance
                .per_edge
                .into_iter()
                .map(|(edge, width)| (edge_map[edge as usize], width))
                .collect();
            for notch in &mut copy.notches {
                notch.edge = edge_map[notch.edge as usize];
            }
            copy.grainline.origin.x = 2.0 * axis_x - copy.grainline.origin.x;
            copy.grainline.angle = std::f32::consts::PI - copy.grainline.angle;
            copy.uv.origin.x = 2.0 * axis_x - copy.uv.origin.x;
            copy.uv.rotation = std::f32::consts::PI - copy.uv.rotation;
            copy.arrangement.anchor.x = 2.0 * axis_x - copy.arrangement.anchor.x;
        }

        copy.id = new_id;
        copy.name = if mirrored {
            format!("{} mirrored", copy.name)
        } else {
            format!("{} copy", copy.name)
        };
        copy.outline.translate(offset);
        copy.grainline.origin += offset;
        copy.uv.origin += offset;
        copy.arrangement.anchor += offset;
        self.pieces.push(copy);
        self.touch();
        Some(new_id)
    }

    /// Removes a piece and every seam attached to it.
    pub fn remove_piece(&mut self, id: PieceId) -> Option<PatternPiece> {
        let index = self.pieces.iter().position(|p| p.id == id)?;
        self.seams.retain(|s| s.a.piece != id && s.b.piece != id);
        self.touch();
        Some(self.pieces.remove(index))
    }

    pub fn pieces(&self) -> &[PatternPiece] {
        &self.pieces
    }

    pub fn piece(&self, id: PieceId) -> Option<&PatternPiece> {
        self.pieces.iter().find(|p| p.id == id)
    }

    pub fn piece_mut(&mut self, id: PieceId) -> Option<&mut PatternPiece> {
        self.pieces.iter_mut().find(|p| p.id == id)
    }

    // --- seams -------------------------------------------------------------

    pub fn add_seam(&mut self, a: EdgeRef, b: EdgeRef, reversed: bool) -> SeamId {
        let id = SeamId(self.alloc_id());
        self.seams.push(Seam { id, a, b, reversed });
        self.touch();
        id
    }

    pub fn seams(&self) -> &[Seam] {
        &self.seams
    }

    /// The seam an edge takes part in, if any.
    pub fn seam_of(&self, edge: EdgeRef) -> Option<&Seam> {
        self.seams.iter().find(|s| s.a == edge || s.b == edge)
    }

    // --- layers ------------------------------------------------------------

    /// The highest garment layer in use (0 for an empty document).
    pub fn top_layer(&self) -> u32 {
        self.pieces.iter().map(|p| p.layer).max().unwrap_or(0)
    }

    /// Copies every piece of the top layer, with its seams, into a new
    /// layer above it: a garment worn over the first. The copies are
    /// scaled by `ease` about their centres (an outer garment needs room)
    /// and moved `offset` millimetres on the 2D canvas. Returns the new
    /// pieces.
    pub fn add_layer_copy(&mut self, ease: f32, offset: Vec2) -> Vec<PieceId> {
        let layer = self.top_layer();
        let sources: Vec<PatternPiece> = self.pieces.iter().filter(|p| p.layer == layer).cloned().collect();
        let mut copies: Vec<(PieceId, PieceId)> = Vec::new();
        for source in &sources {
            let mut piece = source.clone();
            piece.id = PieceId(self.alloc_id());
            piece.name = format!("{} (layer {})", source.name, layer + 1);
            piece.layer = layer + 1;
            let (lo, hi) = piece.outline.bounds();
            let centre = (lo + hi) * 0.5;
            piece.outline.scale_about(centre, ease);
            piece.grainline.length *= ease;
            piece.outline.translate(offset);
            piece.grainline.origin = centre + (piece.grainline.origin - centre) * ease + offset;
            piece.arrangement.anchor += offset;
            copies.push((source.id, piece.id));
            self.pieces.push(piece);
        }
        let copy_of = |id: PieceId| copies.iter().find(|(from, _)| *from == id).map(|(_, to)| *to);
        let seams: Vec<Seam> = self.seams.clone();
        for seam in seams {
            if let (Some(a), Some(b)) = (copy_of(seam.a.piece), copy_of(seam.b.piece)) {
                self.add_seam(EdgeRef::new(a, seam.a.edge), EdgeRef::new(b, seam.b.edge), seam.reversed);
            }
        }
        self.touch();
        copies.into_iter().map(|(_, to)| to).collect()
    }

    // --- picking -----------------------------------------------------------

    pub fn handle_position(&self, handle: HandleRef) -> Option<Vec2> {
        self.piece(handle.piece)?.outline.handle_position(handle.kind)
    }

    /// Closest handle within `radius` (mm) of `pos`. Later pieces draw on top,
    /// so they win ties.
    pub fn pick_handle(&self, pos: Vec2, radius: f32) -> Option<HandleRef> {
        let mut best: Option<(f32, HandleRef)> = None;
        for piece in &self.pieces {
            for (kind, p) in piece.outline.handles() {
                let d = p.distance(pos);
                if d <= radius && best.is_none_or(|(bd, _)| d <= bd) {
                    best = Some((d, HandleRef { piece: piece.id, kind }));
                }
            }
        }
        best.map(|(_, h)| h)
    }

    /// Closest edge within `radius` (mm) of `pos`.
    pub fn pick_edge(&self, pos: Vec2, radius: f32) -> Option<EdgeRef> {
        let mut best: Option<(f32, EdgeRef)> = None;
        for piece in &self.pieces {
            let flat = piece.outline.flatten(0.5);
            let n = flat.points.len();
            for k in 0..n {
                let d = geometry::distance_to_segment(pos, flat.points[k], flat.points[(k + 1) % n]);
                if d <= radius && best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, EdgeRef::new(piece.id, flat.edge_of[k])));
                }
            }
        }
        best.map(|(_, e)| e)
    }

    /// Topmost piece whose interior contains `pos`.
    pub fn pick_piece(&self, pos: Vec2) -> Option<PieceId> {
        self.pieces.iter().rev().find(|p| p.contains(pos)).map(|p| p.id)
    }

    // --- sample content ----------------------------------------------------

    /// A sleeveless top (front + back, four seams) used as start-up content.
    pub fn demo_top() -> Self {
        let mut doc = Self::new();
        let jersey = doc.add_fabric("Cotton jersey", [0.80, 0.22, 0.20]);
        let lining = doc.add_fabric("Linen", [0.78, 0.74, 0.64]);

        // Edge indices, counter-clockwise from the hem:
        // 0 hem, 1 right side, 2 right armhole, 3 right shoulder,
        // 4 neckline, 5 left shoulder, 6 left armhole, 7 left side.
        let bodice = |centre_x: f32, neck_drop: f32| -> Outline {
            let p = |x: f32, y: f32| ControlPoint::corner(centre_x + x, y);
            let v = |x: f32, y: f32| Vec2::new(centre_x + x, y);
            Outline::new(
                vec![
                    p(-250.0, 0.0),
                    p(250.0, 0.0),
                    p(250.0, 420.0),
                    p(190.0, 630.0),
                    p(90.0, 660.0),
                    p(-90.0, 660.0),
                    p(-190.0, 630.0),
                    p(-250.0, 420.0),
                ],
                vec![
                    Segment::Line,
                    Segment::Line,
                    Segment::Cubic { ctrl1: v(185.0, 440.0), ctrl2: v(170.0, 540.0) },
                    Segment::Line,
                    Segment::Cubic { ctrl1: v(70.0, 660.0 - neck_drop), ctrl2: v(-70.0, 660.0 - neck_drop) },
                    Segment::Line,
                    Segment::Cubic { ctrl1: v(-170.0, 540.0), ctrl2: v(-185.0, 440.0) },
                    Segment::Line,
                ],
            )
        };

        let front = doc.add_piece("Front", bodice(-320.0, 130.0), jersey);
        let back = doc.add_piece("Back", bodice(320.0, 40.0), lining);

        // Arrange the pieces around the avatar's torso. The back is turned
        // around the vertical axis so both pieces face outwards. They start
        // a little high: sewing pulls matching edges together in straight
        // lines, and the shoulder seams must clear the top of the shoulders
        // rather than pass through them. Gravity then settles the garment.
        let chest = Vec3::new(0.0, 1.15, 0.0);
        if let Some(p) = doc.piece_mut(front) {
            p.arrangement.translation = chest + Vec3::new(0.0, 0.0, 0.22);
            p.notches.push(Notch { edge: 0, t: 0.5, kind: NotchKind::Slit, depth: 6.0 });
            p.notches.push(Notch { edge: 2, t: 0.5, kind: NotchKind::V, depth: 6.0 });
            p.notches.push(Notch { edge: 6, t: 0.5, kind: NotchKind::V, depth: 6.0 });
            p.seam_allowance.per_edge.insert(0, 25.0); // hem
        }
        if let Some(p) = doc.piece_mut(back) {
            p.arrangement.translation = chest + Vec3::new(0.0, 0.0, -0.22);
            p.arrangement.rotation = Quat::from_rotation_y(std::f32::consts::PI);
            p.notches.push(Notch { edge: 0, t: 0.5, kind: NotchKind::Slit, depth: 6.0 });
            p.seam_allowance.per_edge.insert(0, 25.0);
        }

        // Turning the back around mirrors it, so its left edges meet the
        // front's right edges and run in the opposite direction.
        let e = EdgeRef::new;
        doc.add_seam(e(front, 1), e(back, 7), true); // right side
        doc.add_seam(e(front, 7), e(back, 1), true); // left side
        doc.add_seam(e(front, 3), e(back, 5), true); // right shoulder
        doc.add_seam(e(front, 5), e(back, 3), true); // left shoulder
        doc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::HandleKind;

    #[test]
    fn demo_document_is_consistent() {
        let doc = PatternDocument::demo_top();
        assert_eq!(doc.pieces().len(), 2);
        assert_eq!(doc.seams().len(), 4);
        for seam in doc.seams() {
            for edge in [seam.a, seam.b] {
                assert!(edge.edge < doc.piece(edge.piece).unwrap().outline.edge_count());
            }
        }
        for piece in doc.pieces() {
            assert!(doc.fabric(piece.fabric).is_some());
            assert!(piece.seam_allowance_outline(0.5).is_some());
        }
    }

    #[test]
    fn duplicate_and_mirror_preserve_piece_annotations_and_winding() {
        let mut doc = PatternDocument::demo_top();
        let source = doc.pieces()[0].clone();
        let mirrored_id = doc.duplicate_piece(source.id, true).unwrap();
        let mirrored = doc.piece(mirrored_id).unwrap();

        assert!(mirrored.name.ends_with("mirrored"));
        assert!(mirrored.outline.is_ccw());
        assert_eq!(mirrored.notches[0].edge, 7);
        assert_eq!(mirrored.seam_allowance.per_edge.get(&7), Some(&25.0));
        assert!(mirrored.outline.bounds().0.x > source.outline.bounds().1.x);

        let copy_id = doc.duplicate_piece(source.id, false).unwrap();
        let copy = doc.piece(copy_id).unwrap();
        assert!(copy.name.ends_with("copy"));
        assert_eq!(copy.outline.edge_count(), source.outline.edge_count());
        assert_eq!(copy.notches[0].edge, source.notches[0].edge);
    }

    #[test]
    fn picking_finds_the_nearest_handle() {
        let doc = PatternDocument::demo_top();
        let front = doc.pieces()[0].id;
        let hit = doc.pick_handle(Vec2::new(-568.0, 1.0), 5.0).unwrap();
        assert_eq!(hit, HandleRef { piece: front, kind: HandleKind::Anchor(0) });
        assert!(doc.pick_handle(Vec2::new(-320.0, 300.0), 5.0).is_none());
        assert_eq!(doc.pick_piece(Vec2::new(-320.0, 300.0)), Some(front));
    }
}

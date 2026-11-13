//! The pattern document: pieces, fabrics and the seams that sew them together.

use glam::{Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::geometry;
use crate::ids::{EdgeRef, FabricId, HandleRef, PieceId, SeamId};
use crate::lines::EdgeSpan;
use crate::material::Fabric;
use crate::pattern::{
    Button, Buttonhole, ButtonholeRef, ControlPoint, Notch, NotchKind, Outline, PatternPiece,
    Segment,
};

/// Sews one run of edges to another: `a_count` edges starting at `a` to
/// `b_count` edges starting at `b`. Usually each side is one *line* of its
/// piece (see [`crate::lines`]), which may consist of several segments.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Seam {
    pub id: SeamId,
    /// First edge of side A.
    pub a: EdgeRef,
    /// First edge of side B.
    pub b: EdgeRef,
    /// When `true`, the start of side A meets the *end* of side B.
    pub reversed: bool,
    #[serde(default = "Seam::one")]
    pub a_count: u32,
    #[serde(default = "Seam::one")]
    pub b_count: u32,
}

impl Seam {
    const fn one() -> u32 {
        1
    }

    pub fn a_span(&self) -> EdgeSpan {
        EdgeSpan {
            first: self.a.edge,
            count: self.a_count.max(1),
        }
    }

    pub fn b_span(&self) -> EdgeSpan {
        EdgeSpan {
            first: self.b.edge,
            count: self.b_count.max(1),
        }
    }

    /// Whether the seam sews `edge`, given the edge count of its piece.
    pub fn covers(&self, edge: EdgeRef, edge_count: u32) -> bool {
        (self.a.piece == edge.piece && self.a_span().contains(edge.edge, edge_count))
            || (self.b.piece == edge.piece && self.b_span().contains(edge.edge, edge_count))
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
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

    pub fn add_piece(
        &mut self,
        name: impl Into<String>,
        outline: Outline,
        fabric: FabricId,
    ) -> PieceId {
        let id = PieceId(self.alloc_id());
        self.pieces
            .push(PatternPiece::new(id, name, outline, fabric));
        self.touch();
        id
    }

    /// Duplicates a piece beside its source, optionally mirrored left ↔
    /// right: the matching piece for the other side of the body, which is
    /// also placed on the other side of the body in 3D.
    pub fn duplicate_piece(&mut self, source_id: PieceId, mirrored: bool) -> Option<PieceId> {
        let mut copy = self.piece(source_id)?.clone();
        let new_id = PieceId(self.alloc_id());
        let (min, max) = copy.outline.bounds();
        let offset = Vec2::new((max.x - min.x).max(100.0) + 30.0, 0.0);
        if mirrored {
            copy.apply_transform(crate::transform::PieceTransform::FlipHorizontal);
            copy.placement.angle = -copy.placement.angle;
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

    /// Moves a piece in the 2D layout; its 3D placement stays.
    pub fn move_piece(&mut self, id: PieceId, delta: Vec2) -> bool {
        let Some(piece) = self.piece_mut(id) else {
            return false;
        };
        piece.outline.translate(delta);
        piece.grainline.origin += delta;
        piece.uv.origin += delta;
        piece.arrangement.anchor += delta;
        self.touch();
        true
    }

    /// Lays the pieces out in rows without overlaps, `gap` mm apart, in
    /// their order around the body (right back, right front, left front,
    /// left back), so that matching pieces end up next to each other.
    pub fn tidy_layout(&mut self, gap: f32) {
        let mut order: Vec<(f32, usize)> = self
            .pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (p.placement.angle, i))
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let boxes: Vec<(Vec2, Vec2)> = self.pieces.iter().map(|p| p.outline.bounds()).collect();
        let area: f32 = boxes
            .iter()
            .map(|(lo, hi)| (hi.x - lo.x + gap) * (hi.y - lo.y + gap))
            .sum();
        let widest = boxes.iter().map(|(lo, hi)| hi.x - lo.x).fold(0.0, f32::max);
        // Rows about twice as wide as the layout is tall.
        let row_width = (area * 2.0).sqrt().max(widest);
        let (mut x, mut top, mut row_height) = (0.0f32, 0.0f32, 0.0f32);
        let mut moves = Vec::new();
        for &(_, index) in &order {
            let (lo, hi) = boxes[index];
            let size = hi - lo;
            if x > 0.0 && x + size.x > row_width {
                x = 0.0;
                top -= row_height + gap;
                row_height = 0.0;
            }
            // Top-left corner of the piece's bounds goes to (x, top).
            moves.push((self.pieces[index].id, Vec2::new(x - lo.x, top - hi.y)));
            x += size.x + gap;
            row_height = row_height.max(size.y);
        }
        for (id, delta) in moves {
            self.move_piece(id, delta);
        }
    }

    /// Removes a piece and every seam attached to it.
    pub fn remove_piece(&mut self, id: PieceId) -> Option<PatternPiece> {
        let index = self.pieces.iter().position(|p| p.id == id)?;
        self.seams.retain(|s| s.a.piece != id && s.b.piece != id);
        for button in self.pieces.iter_mut().flat_map(|p| &mut p.buttons) {
            if button.fastened_to.is_some_and(|hole| hole.piece == id) {
                button.fastened_to = None;
            }
        }
        self.touch();
        Some(self.pieces.remove(index))
    }

    // --- buttons -----------------------------------------------------------

    /// The buttonhole `hole` points at, if it still exists.
    pub fn buttonhole(&self, hole: ButtonholeRef) -> Option<&Buttonhole> {
        self.piece(hole.piece)?.buttonholes.get(hole.index as usize)
    }

    /// Removes a buttonhole. Buttons done up through it come undone, and
    /// references to the piece's later buttonholes move down by one.
    pub fn remove_buttonhole(&mut self, piece: PieceId, index: usize) -> Option<Buttonhole> {
        let hole = {
            let holes = &mut self.piece_mut(piece)?.buttonholes;
            (index < holes.len()).then(|| holes.remove(index))?
        };
        for button in self.pieces.iter_mut().flat_map(|p| &mut p.buttons) {
            match &mut button.fastened_to {
                Some(r) if r.piece == piece && r.index as usize == index => {
                    button.fastened_to = None
                }
                Some(r) if r.piece == piece && r.index as usize > index => r.index -= 1,
                _ => {}
            }
        }
        self.touch();
        Some(hole)
    }

    /// Every button that is done up, with its buttonhole:
    /// `(button piece, button, hole piece, hole)`.
    pub fn fastenings(
        &self,
    ) -> impl Iterator<Item = (PieceId, &Button, PieceId, &Buttonhole)> + '_ {
        self.pieces.iter().flat_map(move |piece| {
            piece.buttons.iter().filter_map(move |button| {
                let target = button.fastened_to?;
                Some((piece.id, button, target.piece, self.buttonhole(target)?))
            })
        })
    }

    /// Does up the open buttons, judged by where the pieces sit around the
    /// avatar. Each piece's open buttons go to the one other piece whose
    /// free buttonholes lie closest to them (at most `reach` metres away
    /// on average), and are paired with its holes in order along the
    /// button row, top button to top hole: nearest-first pairing would
    /// cross pairs or reach for a hole on the wrong side as soon as two
    /// rows are a little offset. Returns how many buttons were fastened.
    pub fn fasten_buttons(&mut self, reach: f32) -> usize {
        let mut taken: Vec<ButtonholeRef> = self
            .pieces
            .iter()
            .flat_map(|p| &p.buttons)
            .filter_map(|b| b.fastened_to)
            .collect();
        let mut fastened = 0;
        for piece_index in 0..self.pieces.len() {
            let piece = &self.pieces[piece_index];
            let open: Vec<(usize, Vec3)> = piece
                .buttons
                .iter()
                .enumerate()
                .filter(|(_, b)| !b.fastened_to.is_some_and(|r| self.buttonhole(r).is_some()))
                .map(|(i, b)| (i, piece.arrangement.place(b.position)))
                .collect();
            if open.is_empty() {
                continue;
            }
            // Free holes per other piece.
            let mut best: Option<(f32, Vec<(usize, ButtonholeRef)>)> = None;
            for other in self.pieces.iter().filter(|o| o.id != piece.id) {
                let holes: Vec<(ButtonholeRef, Vec3)> = other
                    .buttonholes
                    .iter()
                    .enumerate()
                    .map(|(index, hole)| {
                        (
                            ButtonholeRef {
                                piece: other.id,
                                index: index as u32,
                            },
                            other.arrangement.place(hole.position),
                        )
                    })
                    .filter(|(r, _)| !taken.contains(r))
                    .collect();
                if holes.is_empty() {
                    continue;
                }
                let pairs = pair_along_rows(&open, &holes);
                let mean = pairs
                    .iter()
                    .map(|&(b, h)| open[b].1.distance(holes[h].1))
                    .sum::<f32>()
                    / pairs.len().max(1) as f32;
                if mean <= reach && best.as_ref().is_none_or(|(d, _)| mean < *d) {
                    best = Some((
                        mean,
                        pairs
                            .into_iter()
                            .map(|(b, h)| (open[b].0, holes[h].0))
                            .collect(),
                    ));
                }
            }
            if let Some((_, pairs)) = best {
                for (button, hole) in pairs {
                    self.pieces[piece_index].buttons[button].fastened_to = Some(hole);
                    taken.push(hole);
                    fastened += 1;
                }
            }
        }
        if fastened > 0 {
            self.touch();
        }
        fastened
    }

    /// Undoes every button. Returns how many were done up.
    pub fn unfasten_buttons(&mut self) -> usize {
        let mut count = 0;
        for button in self.pieces.iter_mut().flat_map(|p| &mut p.buttons) {
            count += usize::from(button.fastened_to.take().is_some());
        }
        if count > 0 {
            self.touch();
        }
        count
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
        self.seams.push(Seam {
            id,
            a,
            b,
            reversed,
            a_count: 1,
            b_count: 1,
        });
        self.touch();
        id
    }

    pub(crate) fn seams_mut(&mut self) -> &mut [Seam] {
        &mut self.seams
    }

    pub fn seams(&self) -> &[Seam] {
        &self.seams
    }

    /// Sews a run of edges on one piece to a run on another (or the same)
    /// piece.
    pub fn add_seam_spans(
        &mut self,
        a: (PieceId, EdgeSpan),
        b: (PieceId, EdgeSpan),
        reversed: bool,
    ) -> SeamId {
        let id = SeamId(self.alloc_id());
        self.seams.push(Seam {
            id,
            a: EdgeRef::new(a.0, a.1.first),
            b: EdgeRef::new(b.0, b.1.first),
            reversed,
            a_count: a.1.count.max(1),
            b_count: b.1.count.max(1),
        });
        self.touch();
        id
    }

    pub fn remove_seam(&mut self, id: SeamId) -> Option<Seam> {
        let index = self.seams.iter().position(|s| s.id == id)?;
        self.touch();
        Some(self.seams.remove(index))
    }

    /// Turns a seam around: the start of side A then meets the other end
    /// of side B.
    pub fn flip_seam(&mut self, id: SeamId) -> bool {
        let Some(seam) = self.seams.iter_mut().find(|s| s.id == id) else {
            return false;
        };
        seam.reversed = !seam.reversed;
        self.touch();
        true
    }

    /// The seam an edge takes part in, if any.
    pub fn seam_of(&self, edge: EdgeRef) -> Option<&Seam> {
        let count = self.piece(edge.piece)?.outline.edge_count();
        self.seams.iter().find(|s| s.covers(edge, count))
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
        let sources: Vec<PatternPiece> = self
            .pieces
            .iter()
            .filter(|p| p.layer == layer)
            .cloned()
            .collect();
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
        let copy_of = |id: PieceId| {
            copies
                .iter()
                .find(|(from, _)| *from == id)
                .map(|(_, to)| *to)
        };
        let seams: Vec<Seam> = self.seams.clone();
        for seam in seams {
            if let (Some(a), Some(b)) = (copy_of(seam.a.piece), copy_of(seam.b.piece)) {
                self.add_seam(
                    EdgeRef::new(a, seam.a.edge),
                    EdgeRef::new(b, seam.b.edge),
                    seam.reversed,
                );
            }
        }
        self.touch();
        copies.into_iter().map(|(_, to)| to).collect()
    }

    // --- picking -----------------------------------------------------------

    pub fn handle_position(&self, handle: HandleRef) -> Option<Vec2> {
        self.piece(handle.piece)?
            .outline
            .handle_position(handle.kind)
    }

    /// Closest handle within `radius` (mm) of `pos`. Later pieces draw on top,
    /// so they win ties.
    pub fn pick_handle(&self, pos: Vec2, radius: f32) -> Option<HandleRef> {
        let mut best: Option<(f32, HandleRef)> = None;
        for piece in &self.pieces {
            for (kind, p) in piece.outline.handles() {
                let d = p.distance(pos);
                if d <= radius && best.is_none_or(|(bd, _)| d <= bd) {
                    best = Some((
                        d,
                        HandleRef {
                            piece: piece.id,
                            kind,
                        },
                    ));
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
                let d =
                    geometry::distance_to_segment(pos, flat.points[k], flat.points[(k + 1) % n]);
                if d <= radius && best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, EdgeRef::new(piece.id, flat.edge_of[k])));
                }
            }
        }
        best.map(|(_, e)| e)
    }

    /// Topmost piece whose interior contains `pos`.
    pub fn pick_piece(&self, pos: Vec2) -> Option<PieceId> {
        self.pieces
            .iter()
            .rev()
            .find(|p| p.contains(pos))
            .map(|p| p.id)
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
                    Segment::Cubic {
                        ctrl1: v(185.0, 440.0),
                        ctrl2: v(170.0, 540.0),
                    },
                    Segment::Line,
                    Segment::Cubic {
                        ctrl1: v(70.0, 660.0 - neck_drop),
                        ctrl2: v(-70.0, 660.0 - neck_drop),
                    },
                    Segment::Line,
                    Segment::Cubic {
                        ctrl1: v(-170.0, 540.0),
                        ctrl2: v(-185.0, 440.0),
                    },
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
            p.notches.push(Notch {
                edge: 0,
                t: 0.5,
                kind: NotchKind::Slit,
                depth: 6.0,
            });
            p.notches.push(Notch {
                edge: 2,
                t: 0.5,
                kind: NotchKind::V,
                depth: 6.0,
            });
            p.notches.push(Notch {
                edge: 6,
                t: 0.5,
                kind: NotchKind::V,
                depth: 6.0,
            });
            p.seam_allowance.per_edge.insert(0, 25.0); // hem
        }
        if let Some(p) = doc.piece_mut(back) {
            p.arrangement.translation = chest + Vec3::new(0.0, 0.0, -0.22);
            p.arrangement.rotation = Quat::from_rotation_y(std::f32::consts::PI);
            p.placement.angle = 180.0;
            p.notches.push(Notch {
                edge: 0,
                t: 0.5,
                kind: NotchKind::Slit,
                depth: 6.0,
            });
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

    /// A V-neck waistcoat that buttons down the centre front: two front
    /// halves, a back, five buttons done up through five buttonholes.
    pub fn demo_waistcoat() -> Self {
        use crate::pattern::{ButtonFinish, ButtonStyle};

        let mut doc = Self::new();
        let wool = doc.add_fabric("Wool suiting", [0.035, 0.045, 0.075]);
        let lining = doc.add_fabric("Satin lining", [0.30, 0.05, 0.06]);
        // A firm, fairly thick woven wool, and a thin, light, slippery lining.
        if let Some(fabric) = doc.fabric_mut(wool) {
            fabric.physics = crate::material::FabricPhysics {
                density: 290.0,
                stretch_weft: 0.72,
                stretch_warp: 0.78,
                shear: 0.45,
                bending: 0.42,
                thickness: 1.3,
            };
            fabric.roughness = 0.85;
        }
        if let Some(fabric) = doc.fabric_mut(lining) {
            fabric.physics = crate::material::FabricPhysics {
                density: 85.0,
                stretch_weft: 0.7,
                stretch_warp: 0.75,
                shear: 0.35,
                bending: 0.1,
                thickness: 0.25,
            };
            fabric.roughness = 0.3;
        }

        /// Overlap past the centre front that carries the buttons.
        const STAND: f32 = 20.0;
        // Left front (the wearer's left, world +X): centre front at x = 0
        // of its own coordinates, the side at x = 250. Edges: 0 hem, 1 side,
        // 2 armhole, 3 shoulder, 4 V-neck, 5 front edge.
        let left_front = |ox: f32| -> Outline {
            let p = |x: f32, y: f32| ControlPoint::corner(ox + x, y);
            let v = |x: f32, y: f32| Vec2::new(ox + x, y);
            Outline::new(
                vec![
                    p(-STAND, 0.0),
                    p(250.0, 0.0),
                    p(250.0, 420.0),
                    p(190.0, 630.0),
                    p(90.0, 660.0),
                    p(-STAND, 380.0),
                ],
                vec![
                    Segment::Line,
                    Segment::Line,
                    Segment::Cubic {
                        ctrl1: v(185.0, 440.0),
                        ctrl2: v(170.0, 540.0),
                    },
                    Segment::Line,
                    Segment::Cubic {
                        ctrl1: v(62.0, 570.0),
                        ctrl2: v(10.0, 440.0),
                    },
                    Segment::Line,
                ],
            )
        };
        // Right front, the mirror image. Edges: 0 hem, 1 front edge,
        // 2 V-neck, 3 shoulder, 4 armhole, 5 side.
        let right_front = |ox: f32| -> Outline {
            let p = |x: f32, y: f32| ControlPoint::corner(ox + x, y);
            let v = |x: f32, y: f32| Vec2::new(ox + x, y);
            Outline::new(
                vec![
                    p(-250.0, 0.0),
                    p(STAND, 0.0),
                    p(STAND, 380.0),
                    p(-90.0, 660.0),
                    p(-190.0, 630.0),
                    p(-250.0, 420.0),
                ],
                vec![
                    Segment::Line,
                    Segment::Line,
                    Segment::Cubic {
                        ctrl1: v(-10.0, 440.0),
                        ctrl2: v(-62.0, 570.0),
                    },
                    Segment::Line,
                    Segment::Cubic {
                        ctrl1: v(-170.0, 540.0),
                        ctrl2: v(-185.0, 440.0),
                    },
                    Segment::Line,
                ],
            )
        };
        let back_outline = {
            let ox = 420.0;
            let p = |x: f32, y: f32| ControlPoint::corner(ox + x, y);
            let v = |x: f32, y: f32| Vec2::new(ox + x, y);
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
                    Segment::Cubic {
                        ctrl1: v(185.0, 440.0),
                        ctrl2: v(170.0, 540.0),
                    },
                    Segment::Line,
                    Segment::Cubic {
                        ctrl1: v(70.0, 620.0),
                        ctrl2: v(-70.0, 620.0),
                    },
                    Segment::Line,
                    Segment::Cubic {
                        ctrl1: v(-170.0, 540.0),
                        ctrl2: v(-185.0, 440.0),
                    },
                    Segment::Line,
                ],
            )
        };

        let (right_x, left_x) = (-680.0, -330.0);
        let right = doc.add_piece("Right front", right_front(right_x), wool);
        let left = doc.add_piece("Left front", left_front(left_x), wool);
        let back = doc.add_piece("Back", back_outline, lining);

        // Buttons down the left front's centre line, buttonholes on the
        // right front, which laps over it (women's side). The right front
        // is the outer layer where they overlap.
        let rows = [345.0, 270.0, 195.0, 120.0, 45.0];
        let diameter = 18.0;
        if let Some(piece) = doc.piece_mut(right) {
            piece.layer = 2;
            // Horizontal holes reach from the centre line towards the edge;
            // the button sits at the hole's inner end.
            piece.buttonholes = rows
                .iter()
                .map(|&y| {
                    let hole = Buttonhole::for_button(Vec2::ZERO, diameter);
                    Buttonhole {
                        position: Vec2::new(right_x + 0.5 * hole.length - 2.0, y),
                        ..hole
                    }
                })
                .collect();
        }
        if let Some(piece) = doc.piece_mut(left) {
            piece.buttons = rows
                .iter()
                .enumerate()
                .map(|(index, &y)| Button {
                    style: ButtonStyle::FourHole,
                    finish: ButtonFinish::Pearl,
                    color: [0.62, 0.58, 0.50],
                    thread_color: [0.02, 0.025, 0.04],
                    fastened_to: Some(ButtonholeRef {
                        piece: right,
                        index: index as u32,
                    }),
                    ..Button::new(Vec2::new(left_x, y), diameter)
                })
                .collect();
        }

        // Fronts a third of the way round each side, the back behind. The
        // arrangement wraps them around the body; the buttons then pull
        // the centre fronts onto each other.
        for (id, angle) in [(right, -36.0), (left, 36.0), (back, 180.0)] {
            if let Some(piece) = doc.piece_mut(id) {
                piece.placement.angle = angle;
                piece.seam_allowance.per_edge.insert(0, 25.0); // hem
            }
        }
        if let Some(p) = doc.piece_mut(back) {
            p.arrangement.rotation = Quat::from_rotation_y(std::f32::consts::PI);
        }

        // As in `demo_top`: the back is turned round, so its pattern-left
        // edges meet the left front.
        let e = EdgeRef::new;
        doc.add_seam(e(left, 1), e(back, 7), true); // left side
        doc.add_seam(e(left, 3), e(back, 5), true); // left shoulder
        doc.add_seam(e(right, 5), e(back, 1), true); // right side
        doc.add_seam(e(right, 3), e(back, 3), true); // right shoulder
        doc
    }
}

/// Pairs buttons with holes in order along their rows: both are sorted
/// along the direction the button row runs (vertical for a single button)
/// and matched from the same end. Returns (button, hole) index pairs, as
/// many as the shorter list.
fn pair_along_rows(
    buttons: &[(usize, Vec3)],
    holes: &[(ButtonholeRef, Vec3)],
) -> Vec<(usize, usize)> {
    let along = |points: &[Vec3]| -> Vec3 {
        let (first, last) = points.iter().fold((points[0], points[0]), |(lo, hi), &p| {
            (
                if p.y > lo.y { p } else { lo },
                if p.y < hi.y { p } else { hi },
            )
        });
        (first - last)
            .try_normalize()
            .filter(|d| d.y.abs() > 0.3)
            .unwrap_or(Vec3::Y)
    };
    let row: Vec<Vec3> = buttons.iter().map(|b| b.1).collect();
    let direction = along(&row);
    let mut b: Vec<usize> = (0..buttons.len()).collect();
    let mut h: Vec<usize> = (0..holes.len()).collect();
    b.sort_by(|&i, &j| {
        buttons[j]
            .1
            .dot(direction)
            .total_cmp(&buttons[i].1.dot(direction))
    });
    h.sort_by(|&i, &j| {
        holes[j]
            .1
            .dot(direction)
            .total_cmp(&holes[i].1.dot(direction))
    });
    b.into_iter().zip(h).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::HandleKind;

    #[test]
    fn tidy_layout_removes_overlaps() {
        let mut doc = PatternDocument::demo_top();
        let id = doc.pieces()[0].id;
        doc.duplicate_piece(id, true);
        doc.duplicate_piece(id, false);
        doc.tidy_layout(30.0);
        let boxes: Vec<_> = doc.pieces().iter().map(|p| p.outline.bounds()).collect();
        for (i, a) in boxes.iter().enumerate() {
            for b in &boxes[i + 1..] {
                let apart = a.1.x + 29.0 < b.0.x
                    || b.1.x + 29.0 < a.0.x
                    || a.1.y + 29.0 < b.0.y
                    || b.1.y + 29.0 < a.0.y;
                assert!(apart, "{a:?} and {b:?} overlap");
            }
        }
    }

    #[test]
    fn buttons_are_done_up_in_order_down_the_row() {
        let mut doc = PatternDocument::new();
        let fabric = doc.add_fabric("Test", [0.5; 3]);
        let square = || {
            Outline::polygon([
                Vec2::ZERO,
                Vec2::new(200.0, 0.0),
                Vec2::new(200.0, 400.0),
                Vec2::new(0.0, 400.0),
            ])
        };
        let left = doc.add_piece("Left", square(), fabric);
        let right = doc.add_piece("Right", square(), fabric);
        // Buttons every 100 mm, added out of order; the holes sit 60 mm
        // lower. Nearest-first would give the lowest button the middle
        // hole and leave the middle button the bottom one.
        doc.piece_mut(left).unwrap().buttons = [100.0, 300.0, 200.0]
            .map(|y| Button::new(Vec2::new(190.0, y), 15.0))
            .to_vec();
        doc.piece_mut(right).unwrap().buttonholes = [240.0, 140.0, 40.0]
            .map(|y| Buttonhole::for_button(Vec2::new(10.0, y), 15.0))
            .to_vec();
        // Side by side, the right piece just past the left one's edge.
        doc.piece_mut(right).unwrap().arrangement.translation = Vec3::new(0.19, 0.0, 0.01);
        assert_eq!(doc.fasten_buttons(0.25), 3);
        let left_piece = doc.piece(left).unwrap();
        let right_piece = doc.piece(right).unwrap();
        for button in &left_piece.buttons {
            let hole = &right_piece.buttonholes[button.fastened_to.unwrap().index as usize];
            assert_eq!(
                button.position.y - hole.position.y,
                60.0,
                "button at {} done up through the hole at {}",
                button.position.y,
                hole.position.y
            );
        }
    }

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
        assert_eq!(
            hit,
            HandleRef {
                piece: front,
                kind: HandleKind::Anchor(0)
            }
        );
        assert!(doc.pick_handle(Vec2::new(-320.0, 300.0), 5.0).is_none());
        assert_eq!(doc.pick_piece(Vec2::new(-320.0, 300.0)), Some(front));
    }
}

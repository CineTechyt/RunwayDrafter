//! Incremental pattern → mesh synchronisation and the reverse lookups.

use std::collections::{HashMap, HashSet};

use cad_core::glam::{Vec2, Vec3};
use cad_core::{DocumentChange, EdgeRef, PatternDocument, PatternPiece, PieceId};

use crate::mesh::{GarmentMesh, PieceSpan, Stitch, VertexOrigin};
use crate::triangulate::{self, MeshingParams, PieceMesh};

/// What a synchronisation pass did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SyncReport {
    /// Vertex/index buffers were reassembled; GPU buffers and solver
    /// constraints must be rebuilt.
    pub topology_changed: bool,
    /// Vertex attributes changed without touching connectivity.
    pub attributes_changed: bool,
    /// Pieces that went through triangulation again.
    pub remeshed: Vec<PieceId>,
}

impl SyncReport {
    pub fn changed(&self) -> bool {
        self.topology_changed || self.attributes_changed
    }
}

/// A point on a pattern piece, in pattern space (mm).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatternLocation {
    pub piece: PieceId,
    pub position: Vec2,
}

/// A point on the garment surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceLocation {
    pub triangle: u32,
    /// Weights of the triangle's three vertices.
    pub barycentric: Vec3,
}

/// The 2D ↔ 3D mapping interface.
pub trait TopologySync {
    /// Discards all cached state and rebuilds the mesh from scratch.
    fn rebuild(&mut self, doc: &PatternDocument) -> SyncReport;

    /// Brings the mesh up to date after `changes`, doing as little work as
    /// the changes allow.
    fn apply(&mut self, doc: &PatternDocument, changes: &[DocumentChange]) -> SyncReport;

    fn mesh(&self) -> &GarmentMesh;

    /// Mutable access for the cloth solver (positions and normals only;
    /// connectivity is owned by the synchroniser).
    fn mesh_mut(&mut self) -> &mut GarmentMesh;

    // --- 2D → 3D -----------------------------------------------------------

    /// Ordered mesh vertices along a pattern edge, corner to corner.
    fn edge_vertices(&self, edge: EdgeRef) -> Option<&[u32]>;

    /// The surface point that a pattern-space point maps to.
    fn pattern_to_surface(&self, location: PatternLocation) -> Option<SurfaceLocation>;

    // --- 3D → 2D -----------------------------------------------------------

    fn vertex_origin(&self, vertex: u32) -> Option<VertexOrigin>;

    /// The pattern-space point under a surface point (e.g. a ray hit).
    fn surface_to_pattern(&self, location: SurfaceLocation) -> Option<PatternLocation>;
}

#[derive(Debug, Clone)]
struct CachedPiece {
    id: PieceId,
    /// Mesh edges per pattern edge that `mesh` was built with.
    plan: Vec<u32>,
    mesh: PieceMesh,
    positions: Vec<Vec3>,
    uvs: Vec<Vec2>,
}

impl CachedPiece {
    fn place(&mut self, piece: &PatternPiece) {
        self.positions = self.mesh.pattern_positions.iter().map(|&p| piece.arrangement.place(p)).collect();
    }

    fn map_uvs(&mut self, piece: &PatternPiece) {
        self.uvs = self.mesh.pattern_positions.iter().map(|&p| piece.uv.map(p)).collect();
    }
}

#[derive(Default)]
struct Dirty {
    everything: bool,
    geometry: HashSet<PieceId>,
    arrangement: HashSet<PieceId>,
    uv: HashSet<PieceId>,
    /// Piece list or seams changed: reassemble even if nothing was re-meshed.
    structure: bool,
}

/// Default [`TopologySync`] implementation.
#[derive(Debug, Default)]
pub struct GarmentTopology {
    params: MeshingParams,
    cache: Vec<CachedPiece>,
    mesh: GarmentMesh,
    edge_map: HashMap<EdgeRef, Vec<u32>>,
}

impl GarmentTopology {
    pub fn new(params: MeshingParams) -> Self {
        Self { params, ..Self::default() }
    }

    pub fn params(&self) -> MeshingParams {
        self.params
    }

    /// Changes the mesh resolution; takes effect on the next `rebuild`.
    pub fn set_params(&mut self, params: MeshingParams) {
        self.params = params;
    }

    /// Mesh edges per pattern edge for the whole document. Both sides of a
    /// seam get the same count so their vertices pair up one to one.
    fn plan(&self, doc: &PatternDocument) -> HashMap<PieceId, Vec<u32>> {
        let mut plan: HashMap<PieceId, Vec<u32>> = doc
            .pieces()
            .iter()
            .map(|piece| {
                let counts = triangulate::edge_lengths(piece, &self.params)
                    .into_iter()
                    .map(|len| self.params.subdivisions(len))
                    .collect();
                (piece.id, counts)
            })
            .collect();

        let get = |plan: &HashMap<PieceId, Vec<u32>>, e: EdgeRef| plan.get(&e.piece)?.get(e.edge as usize).copied();
        // An edge can only be in one seam today, but iterating to a fixed
        // point keeps this correct once 1:N seams exist.
        for _ in 0..doc.seams().len().max(1) {
            let mut stable = true;
            for seam in doc.seams() {
                let (Some(a), Some(b)) = (get(&plan, seam.a), get(&plan, seam.b)) else { continue };
                if a != b {
                    stable = false;
                    let n = a.max(b);
                    plan.get_mut(&seam.a.piece).unwrap()[seam.a.edge as usize] = n;
                    plan.get_mut(&seam.b.piece).unwrap()[seam.b.edge as usize] = n;
                }
            }
            if stable {
                break;
            }
        }
        plan
    }

    /// Copies solver-written positions back into the per-piece cache so that
    /// pieces which are not re-meshed keep their draped shape.
    fn save_positions(&mut self) {
        for span in &self.mesh.pieces {
            if let Some(cached) = self.cache.iter_mut().find(|c| c.id == span.piece) {
                let range = span.vertices.start as usize..span.vertices.end as usize;
                if range.len() == cached.positions.len() {
                    cached.positions.copy_from_slice(&self.mesh.positions[range]);
                }
            }
        }
    }

    fn sync(&mut self, doc: &PatternDocument, dirty: Dirty) -> SyncReport {
        self.save_positions();
        let mut plan = self.plan(doc);
        let mut report = SyncReport::default();

        let mut old: HashMap<PieceId, CachedPiece> = self.cache.drain(..).map(|c| (c.id, c)).collect();
        let had_removed_pieces = old.keys().any(|id| doc.piece(*id).is_none());
        for piece in doc.pieces() {
            let piece_plan = plan.remove(&piece.id).unwrap_or_default();
            let reusable = old
                .remove(&piece.id)
                .filter(|c| !dirty.everything && !dirty.geometry.contains(&piece.id) && c.plan == piece_plan);
            let cached = match reusable {
                Some(mut cached) => {
                    if dirty.arrangement.contains(&piece.id) {
                        cached.place(piece);
                        report.attributes_changed = true;
                    }
                    if dirty.uv.contains(&piece.id) {
                        cached.map_uvs(piece);
                        report.attributes_changed = true;
                    }
                    cached
                }
                None => {
                    let mesh = triangulate::mesh_piece(piece, &piece_plan, &self.params);
                    let mut cached =
                        CachedPiece { id: piece.id, plan: piece_plan, mesh, positions: Vec::new(), uvs: Vec::new() };
                    // A re-meshed piece restarts from its flat arrangement.
                    // Transferring the draped shape across topologies would
                    // go here (barycentric lookup in the old mesh).
                    cached.place(piece);
                    cached.map_uvs(piece);
                    report.remeshed.push(piece.id);
                    cached
                }
            };
            self.cache.push(cached);
        }

        report.topology_changed = !report.remeshed.is_empty() || dirty.structure || had_removed_pieces;
        if report.changed() {
            self.assemble(doc, report.topology_changed);
        }
        report
    }

    /// Concatenates the cached pieces into the global buffers.
    fn assemble(&mut self, doc: &PatternDocument, topology_changed: bool) {
        let generation = self.mesh.generation + 1;
        let mut mesh = GarmentMesh { generation, ..GarmentMesh::default() };
        if !topology_changed {
            // Connectivity is intact: keep it and only refresh attributes.
            mesh.indices = std::mem::take(&mut self.mesh.indices);
            mesh.pieces = std::mem::take(&mut self.mesh.pieces);
            mesh.stitches = std::mem::take(&mut self.mesh.stitches);
            mesh.origins = std::mem::take(&mut self.mesh.origins);
        } else {
            self.edge_map.clear();
        }

        for cached in &self.cache {
            let base = mesh.positions.len() as u32;
            mesh.positions.extend_from_slice(&cached.positions);
            mesh.uvs.extend_from_slice(&cached.uvs);
            mesh.pattern_positions.extend_from_slice(&cached.mesh.pattern_positions);
            if !topology_changed {
                continue;
            }
            let first_index = mesh.indices.len() as u32;
            mesh.origins.extend_from_slice(&cached.mesh.origins);
            mesh.indices.extend(cached.mesh.indices.iter().map(|i| base + i));
            mesh.pieces.push(PieceSpan {
                piece: cached.id,
                vertices: base..mesh.positions.len() as u32,
                indices: first_index..mesh.indices.len() as u32,
            });
            for (edge, chain) in cached.mesh.edge_chains.iter().enumerate() {
                self.edge_map
                    .insert(EdgeRef::new(cached.id, edge as u32), chain.iter().map(|i| base + i).collect());
            }
        }

        if topology_changed {
            for seam in doc.seams() {
                let (Some(a), Some(b)) = (self.edge_map.get(&seam.a), self.edge_map.get(&seam.b)) else {
                    continue;
                };
                // Equal by construction (see `plan`); `zip` guards the rest.
                debug_assert_eq!(a.len(), b.len());
                let pairs: Box<dyn Iterator<Item = (&u32, &u32)>> =
                    if seam.reversed { Box::new(a.iter().zip(b.iter().rev())) } else { Box::new(a.iter().zip(b.iter())) };
                mesh.stitches.extend(pairs.map(|(&a, &b)| Stitch { seam: seam.id, a, b }));
            }
        }

        mesh.recompute_normals();
        self.mesh = mesh;
    }

    fn triangles_of(&self, piece: PieceId) -> impl Iterator<Item = u32> + '_ {
        self.mesh.span(piece).into_iter().flat_map(|s| s.indices.start / 3..s.indices.end / 3)
    }
}

impl TopologySync for GarmentTopology {
    fn rebuild(&mut self, doc: &PatternDocument) -> SyncReport {
        self.cache.clear();
        self.mesh.pieces.clear();
        self.sync(doc, Dirty { everything: true, structure: true, ..Dirty::default() })
    }

    fn apply(&mut self, doc: &PatternDocument, changes: &[DocumentChange]) -> SyncReport {
        let mut dirty = Dirty::default();
        let mut relevant = false;
        for change in changes {
            match *change {
                DocumentChange::PieceGeometry(id) => {
                    dirty.geometry.insert(id);
                }
                DocumentChange::PieceArrangement(id) => {
                    dirty.arrangement.insert(id);
                }
                DocumentChange::PieceUv(id) => {
                    dirty.uv.insert(id);
                }
                DocumentChange::PieceAdded(_) | DocumentChange::PieceRemoved(_) | DocumentChange::Seams => {
                    dirty.structure = true;
                }
                DocumentChange::Everything => {
                    dirty.everything = true;
                    dirty.structure = true;
                }
                // Drawing- or shading-only changes never touch the mesh.
                DocumentChange::PieceAnnotations(_) | DocumentChange::PieceFabric(_) | DocumentChange::Fabric(_) => {
                    continue;
                }
            }
            relevant = true;
        }
        if relevant { self.sync(doc, dirty) } else { SyncReport::default() }
    }

    fn mesh(&self) -> &GarmentMesh {
        &self.mesh
    }

    fn mesh_mut(&mut self) -> &mut GarmentMesh {
        &mut self.mesh
    }

    fn edge_vertices(&self, edge: EdgeRef) -> Option<&[u32]> {
        self.edge_map.get(&edge).map(Vec::as_slice)
    }

    fn pattern_to_surface(&self, location: PatternLocation) -> Option<SurfaceLocation> {
        let p = location.position;
        self.triangles_of(location.piece).find_map(|triangle| {
            let [a, b, c] = self.mesh.triangle(triangle).map(|v| self.mesh.pattern_positions[v as usize]);
            let area = (b - a).perp_dot(c - a);
            if area.abs() < 1e-9 {
                return None;
            }
            let u = (b - p).perp_dot(c - p) / area;
            let v = (c - p).perp_dot(a - p) / area;
            let w = 1.0 - u - v;
            const EPS: f32 = -1e-4;
            (u >= EPS && v >= EPS && w >= EPS).then_some(SurfaceLocation { triangle, barycentric: Vec3::new(u, v, w) })
        })
    }

    fn vertex_origin(&self, vertex: u32) -> Option<VertexOrigin> {
        self.mesh.origins.get(vertex as usize).copied()
    }

    fn surface_to_pattern(&self, location: SurfaceLocation) -> Option<PatternLocation> {
        if location.triangle as usize >= self.mesh.triangle_count() {
            return None;
        }
        let piece = self.mesh.piece_of_triangle(location.triangle)?;
        let [a, b, c] = self.mesh.triangle(location.triangle).map(|v| self.mesh.pattern_positions[v as usize]);
        let w = location.barycentric;
        Some(PatternLocation { piece, position: a * w.x + b * w.y + c * w.z })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cad_core::{HandleKind, HandleRef, MoveHandle, EditCommand};

    fn built() -> (PatternDocument, GarmentTopology) {
        let doc = PatternDocument::demo_top();
        let mut topo = GarmentTopology::default();
        let report = topo.rebuild(&doc);
        assert!(report.topology_changed);
        assert_eq!(report.remeshed.len(), 2);
        (doc, topo)
    }

    #[test]
    fn rebuild_produces_a_consistent_manifold_mesh() {
        let (doc, topo) = built();
        let mesh = topo.mesh();
        assert!(mesh.is_manifold());
        let n = mesh.vertex_count();
        assert!(n > 500);
        assert_eq!(mesh.normals.len(), n);
        assert_eq!(mesh.uvs.len(), n);
        assert_eq!(mesh.pattern_positions.len(), n);
        assert_eq!(mesh.origins.len(), n);
        assert!(mesh.indices.iter().all(|&i| (i as usize) < n));
        // Spans tile the buffers.
        assert_eq!(mesh.pieces.len(), doc.pieces().len());
        assert_eq!(mesh.pieces.last().unwrap().vertices.end as usize, n);
        // Front faces +Z, the turned-around back faces -Z.
        let front = mesh.span(doc.pieces()[0].id).unwrap();
        let back = mesh.span(doc.pieces()[1].id).unwrap();
        assert!(mesh.normals[front.vertices.start as usize].z > 0.99);
        assert!(mesh.normals[back.vertices.start as usize].z < -0.99);
    }

    #[test]
    fn seams_pair_vertices_one_to_one_and_face_each_other() {
        let (doc, topo) = built();
        let mesh = topo.mesh();
        for seam in doc.seams() {
            let a = topo.edge_vertices(seam.a).unwrap();
            let b = topo.edge_vertices(seam.b).unwrap();
            assert_eq!(a.len(), b.len());
            let stitches: Vec<_> = mesh.stitches.iter().filter(|s| s.seam == seam.id).collect();
            assert_eq!(stitches.len(), a.len());
            for s in stitches {
                // Stitched vertices sit across the avatar from each other:
                // same x/y, opposite z.
                let (pa, pb) = (mesh.positions[s.a as usize], mesh.positions[s.b as usize]);
                assert!((pa.x - pb.x).abs() < 2e-3 && (pa.y - pb.y).abs() < 2e-3, "{pa} vs {pb}");
                assert!(pa.z > 0.0 && pb.z < 0.0);
            }
        }
    }

    #[test]
    fn edge_tracking_follows_the_outline() {
        let (doc, topo) = built();
        let piece = &doc.pieces()[0];
        let chain = topo.edge_vertices(EdgeRef::new(piece.id, 2)).unwrap();
        let mesh = topo.mesh();
        assert_eq!(mesh.origins[chain[0] as usize], VertexOrigin::Corner { piece: piece.id, anchor: 2 });
        assert_eq!(mesh.origins[*chain.last().unwrap() as usize], VertexOrigin::Corner { piece: piece.id, anchor: 3 });
        assert_eq!(mesh.pattern_positions[chain[0] as usize], piece.outline.points()[2].position);
        for &v in &chain[1..chain.len() - 1] {
            assert!(matches!(mesh.origins[v as usize], VertexOrigin::Edge { edge, .. } if edge.edge == 2));
        }
    }

    #[test]
    fn moving_a_point_only_remeshes_what_it_must() {
        let (mut doc, mut topo) = built();
        let (front, back) = (doc.pieces()[0].id, doc.pieces()[1].id);

        // Pretend the solver moved the back piece.
        let back_start = topo.mesh().span(back).unwrap().vertices.start as usize;
        let draped = Vec3::new(1.0, 2.0, 3.0);
        topo.mesh_mut().positions[back_start + 3] = draped;

        // Nudge the neckline control: no seam length changes.
        let handle = HandleRef { piece: front, kind: HandleKind::Control { segment: 4, slot: 0 } };
        let from = doc.handle_position(handle).unwrap();
        let change = MoveHandle { handle, from, to: from + Vec2::new(0.0, -15.0) }.apply(&mut doc).unwrap();
        let report = topo.apply(&doc, &[change]);
        assert_eq!(report.remeshed, vec![front]);
        assert!(topo.mesh().is_manifold());
        let back_start = topo.mesh().span(back).unwrap().vertices.start as usize;
        assert_eq!(topo.mesh().positions[back_start + 3], draped);

        // Lengthening a seamed edge re-plans the partner piece too.
        let handle = HandleRef { piece: front, kind: HandleKind::Anchor(0) };
        let from = doc.handle_position(handle).unwrap();
        let change = MoveHandle { handle, from, to: from + Vec2::new(0.0, -200.0) }.apply(&mut doc).unwrap();
        let report = topo.apply(&doc, &[change]);
        assert_eq!(report.remeshed, vec![front, back]);

        // Shading-only changes are free.
        assert_eq!(topo.apply(&doc, &[DocumentChange::Fabric(doc.fabrics()[0].id)]), SyncReport::default());
    }

    #[test]
    fn surface_and_pattern_locations_round_trip() {
        let (doc, topo) = built();
        let piece = &doc.pieces()[0];
        let location = PatternLocation { piece: piece.id, position: Vec2::new(-300.0, 250.0) };
        let surface = topo.pattern_to_surface(location).unwrap();
        let back = topo.surface_to_pattern(surface).unwrap();
        assert_eq!(back.piece, piece.id);
        assert!(back.position.distance(location.position) < 1e-2);

        // A ray through the same point finds the same pattern position.
        let world = piece.arrangement.place(location.position);
        let hit = topo.mesh().raycast(world + Vec3::Z, -Vec3::Z).unwrap();
        let picked = topo
            .surface_to_pattern(SurfaceLocation { triangle: hit.triangle, barycentric: hit.barycentric })
            .unwrap();
        assert_eq!(picked.piece, piece.id);
        assert!(picked.position.distance(location.position) < 0.5);
    }
}

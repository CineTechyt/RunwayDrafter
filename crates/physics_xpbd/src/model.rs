//! Builds the constraint model from the garment mesh and the document.

use std::collections::HashMap;
use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use cad_core::PatternDocument;
use cad_core::glam::Vec2;
use topology_sync::GarmentMesh;

use crate::material::FabricSi;

/// Marks an unused particle slot of an [`Element`].
pub const INVALID: u32 = u32::MAX;
/// The row holds a constraint.
pub const ROW_ACTIVE: u32 = 1;
/// The row is a seam stitch: its rest length is scaled by the seam gap.
pub const ROW_SEAM: u32 = 2;

/// Compliance of a closed stitch in m/N; effectively rigid.
const SEAM_COMPLIANCE: f32 = 1e-7;
const MM_TO_M: f32 = 1e-3;

/// One constraint `|Σ wᵢ·xᵢ| − rest` with XPBD compliance. Layout matches
/// `Row` in `xpbd.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct Row {
    pub weights: [f32; 4],
    pub rest: f32,
    /// Inverse stiffness, already including the element's area.
    pub compliance: f32,
    pub flags: u32,
    /// For bending rows: the value of `|Σ wᵢ·xᵢ|` at a 60° fold. Beyond
    /// it the row stiffens steeply, so cloth cannot collapse into a sharp
    /// crease and pass through itself. 0 = no limit.
    pub fold_limit: f32,
}

/// Up to four constraints over the same (up to four) particles, solved in
/// sequence as one unit. Layout matches `Element` in `xpbd.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct Element {
    pub particles: [u32; 4],
    pub rows: [Row; 4],
}

impl Element {
    fn new(particles: [u32; 4]) -> Self {
        Self { particles, rows: [Row::zeroed(); 4] }
    }

    fn with_row(mut self, slot: usize, weights: [f32; 4], rest: f32, compliance: f32, flags: u32) -> Self {
        self.rows[slot] = Row { weights, rest, compliance, flags: flags | ROW_ACTIVE, fold_limit: 0.0 };
        self
    }
}

/// Everything about the simulation that does not change while it runs.
#[derive(Debug, Clone, Default)]
pub struct Model {
    /// 1/mass per particle in 1/kg; zero pins the particle.
    pub inverse_mass: Vec<f32>,
    /// All elements, sorted by colour.
    pub elements: Vec<Element>,
    /// Element ranges that touch disjoint particle sets: each can be solved
    /// in parallel.
    pub colours: Vec<Range<u32>>,
    pub strain_triangles: usize,
    pub bending_hinges: usize,
    pub stitches: usize,
    /// Total cloth mass in kg.
    pub mass: f32,
}

impl Model {
    pub fn build(mesh: &GarmentMesh, document: &PatternDocument) -> Self {
        let particle_count = mesh.vertex_count();
        let mut masses = vec![0.0f32; particle_count];
        let mut elements = Vec::new();
        let mut model = Model::default();

        for span in &mesh.pieces {
            let Some(piece) = document.piece(span.piece) else { continue };
            let fabric = document.fabric(piece.fabric).map(|f| f.physics).unwrap_or_default();
            let fabric = FabricSi::from_physics(&fabric);
            // Material frame: V along the grainline (warp), U across (weft).
            let warp = Vec2::from_angle(piece.grainline.angle);
            let weft = Vec2::new(warp.y, -warp.x);
            let material = |vertex: u32| {
                let p = mesh.pattern_positions[vertex as usize] * MM_TO_M;
                Vec2::new(p.dot(weft), p.dot(warp))
            };

            let triangles = &mesh.indices[span.indices.start as usize..span.indices.end as usize];
            // Edge → the vertices opposite to it, for bending hinges.
            let mut wings: HashMap<(u32, u32), Vec<u32>> = HashMap::new();

            for tri in triangles.chunks_exact(3) {
                let [i0, i1, i2] = [tri[0], tri[1], tri[2]];
                let (m0, m1, m2) = (material(i0), material(i1), material(i2));
                let (e1, e2) = (m1 - m0, m2 - m0);
                let det = e1.perp_dot(e2);
                let area = 0.5 * det.abs();
                if area < 1e-12 {
                    continue;
                }
                // --- mass: a third of the triangle's weight per corner -----
                for i in [i0, i1, i2] {
                    masses[i as usize] += fabric.density * area / 3.0;
                }

                // --- anisotropic strain ---------------------------------------
                // With Dm = [e1 e2] (material) and Ds = [x1-x0 x2-x0] (world),
                // the deformation gradient is F = Ds·Dm⁻¹. For a material
                // direction d, F·d = q1·(x1-x0) + q2·(x2-x0) with q = Dm⁻¹·d,
                // a fixed linear combination of the three positions.
                let inverse = [[e2.y / det, -e2.x / det], [-e1.y / det, e1.x / det]];
                let diagonal = std::f32::consts::FRAC_1_SQRT_2;
                let directions = [
                    (Vec2::X, fabric.weft),
                    (Vec2::Y, fabric.warp),
                    (Vec2::new(diagonal, diagonal), fabric.bias),
                    (Vec2::new(diagonal, -diagonal), fabric.bias),
                ];
                let mut element = Element::new([i0, i1, i2, INVALID]);
                for (slot, (d, stiffness)) in directions.into_iter().enumerate() {
                    let q1 = inverse[0][0] * d.x + inverse[0][1] * d.y;
                    let q2 = inverse[1][0] * d.x + inverse[1][1] * d.y;
                    // Energy ½·k·A·C² ⇒ compliance 1/(k·A).
                    element = element.with_row(slot, [-q1 - q2, q1, q2, 0.0], 1.0, 1.0 / (stiffness * area), 0);
                }
                elements.push(element);
                model.strain_triangles += 1;

                for (a, b, opposite) in [(i0, i1, i2), (i1, i2, i0), (i2, i0, i1)] {
                    wings.entry((a.min(b), a.max(b))).or_default().push(opposite);
                }
            }

            // --- isometric bending ----------------------------------------------
            // Quadratic bending energy ½·k·(3/(A0+A1))·|Σ Kᵢ·xᵢ|² over the
            // four vertices of two adjacent triangles; K is built from the
            // cotangents of the flat rest shape and vanishes for any planar
            // configuration.
            let mut hinges: Vec<_> = wings.into_iter().filter(|(_, w)| w.len() == 2).collect();
            hinges.sort_unstable_by_key(|(edge, _)| *edge); // deterministic order
            for ((i0, i1), opposite) in hinges {
                let (i2, i3) = (opposite[0], opposite[1]);
                let (x0, x1, x2, x3) = (material(i0), material(i1), material(i2), material(i3));
                let cot = |a: Vec2, b: Vec2| a.dot(b) / a.perp_dot(b).abs().max(1e-12);
                let (e0, e1, e2, e3, e4) = (x1 - x0, x2 - x0, x3 - x0, x2 - x1, x3 - x1);
                let (c01, c02, c03, c04) = (cot(e0, e1), cot(e0, e2), cot(-e0, e3), cot(-e0, e4));
                let areas = 0.5 * (e0.perp_dot(e1).abs() + e0.perp_dot(e2).abs());
                if areas < 1e-12 {
                    continue;
                }
                let weights = [c03 + c04, c01 + c02, -c01 - c03, -c02 - c04];
                let compliance = areas / (3.0 * fabric.bending);
                let mut hinge = Element::new([i0, i1, i2, i3]).with_row(0, weights, 0.0, compliance, 0);
                // Folding one wing by an angle a about the shared edge moves
                // its far vertex by 2·h·sin(a/2), h being its distance from
                // the edge; at 60° that is h.
                let height = e0.perp_dot(e1).abs() / e0.length().max(1e-9);
                hinge.rows[0].fold_limit = weights[2].abs() * height;
                elements.push(hinge);
                model.bending_hinges += 1;
            }
        }

        // --- sewing -----------------------------------------------------------
        // The rest length starts at the current gap; the solver scales it to
        // zero over the seam closure time, reeling the edges together.
        let first_stitch = elements.len();
        for stitch in &mesh.stitches {
            if stitch.a == stitch.b {
                continue;
            }
            let gap = mesh.positions[stitch.a as usize].distance(mesh.positions[stitch.b as usize]);
            elements.push(
                Element::new([stitch.a, stitch.b, INVALID, INVALID])
                    .with_row(0, [1.0, -1.0, 0.0, 0.0], gap, SEAM_COMPLIANCE, ROW_SEAM),
            );
            model.stitches += 1;
        }

        model.mass = masses.iter().sum();
        model.inverse_mass = masses.iter().map(|&m| if m > 0.0 { 1.0 / m } else { 0.0 }).collect();
        // Stitches are coloured separately and solved last in every sweep:
        // whatever the fabric constraints did, the seams end up closed.
        let stitches = elements.split_off(first_stitch);
        (model.elements, model.colours) = colour(elements, particle_count);
        let (stitch_elements, stitch_colours) = colour(stitches, particle_count);
        let offset = model.elements.len() as u32;
        model.elements.extend(stitch_elements);
        model.colours.extend(stitch_colours.into_iter().map(|r| r.start + offset..r.end + offset));
        model
    }
}

/// Greedy graph colouring: gives every element the lowest colour none of its
/// particles has used yet, then sorts the elements by colour.
fn colour(elements: Vec<Element>, particle_count: usize) -> (Vec<Element>, Vec<Range<u32>>) {
    const MAX_SHARED: u32 = 128;
    let mut used = vec![0u128; particle_count];
    let mut colour_of = Vec::with_capacity(elements.len());
    let mut colour_count = 0u32;
    let mut overflow = Vec::new();

    for (index, element) in elements.iter().enumerate() {
        let particles = element.particles.iter().filter(|&&p| p != INVALID);
        let taken = particles.clone().fold(0u128, |mask, &p| mask | used[p as usize]);
        let colour = (!taken).trailing_zeros();
        if colour >= MAX_SHARED {
            // Pathological valence: solve this element on its own.
            overflow.push(index);
            colour_of.push(u32::MAX);
            continue;
        }
        for &p in particles {
            used[p as usize] |= 1 << colour;
        }
        colour_of.push(colour);
        colour_count = colour_count.max(colour + 1);
    }
    for index in overflow {
        colour_of[index] = colour_count;
        colour_count += 1;
    }

    // Counting sort by colour.
    let mut starts = vec![0u32; colour_count as usize + 1];
    for &c in &colour_of {
        starts[c as usize + 1] += 1;
    }
    for c in 0..colour_count as usize {
        starts[c + 1] += starts[c];
    }
    let colours = starts.windows(2).map(|w| w[0]..w[1]).collect();
    let mut cursor = starts;
    let mut sorted = vec![Element::zeroed(); elements.len()];
    for (element, &c) in elements.iter().zip(&colour_of) {
        sorted[cursor[c as usize] as usize] = *element;
        cursor[c as usize] += 1;
    }
    (sorted, colours)
}

#[cfg(test)]
mod tests {
    use super::*;
    use topology_sync::{GarmentTopology, TopologySync};

    fn demo() -> (PatternDocument, GarmentTopology) {
        let document = PatternDocument::demo_top();
        let mut topology = GarmentTopology::default();
        topology.rebuild(&document);
        (document, topology)
    }

    #[test]
    fn mass_is_density_times_area() {
        let (document, topology) = demo();
        let mesh = topology.mesh();
        let model = Model::build(mesh, &document);
        let mut expected = 0.0;
        for span in &mesh.pieces {
            let piece = document.piece(span.piece).unwrap();
            let density = document.fabric(piece.fabric).unwrap().physics.density * 1e-3;
            let boundary = piece.outline.flatten(0.05).points;
            expected += density * cad_core::geometry::signed_area(&boundary).abs() * 1e-6;
        }
        // Two ~0.3 m² panels at 180 g/m²: roughly 100 g.
        assert!((0.08..0.14).contains(&model.mass), "{}", model.mass);
        assert!((model.mass - expected).abs() < 0.01 * expected, "{} vs {expected}", model.mass);
        assert!(model.inverse_mass.iter().all(|&w| w > 0.0 && w.is_finite()));
    }

    #[test]
    fn colours_never_share_a_particle() {
        let (document, topology) = demo();
        let model = Model::build(topology.mesh(), &document);
        assert_eq!(model.colours.iter().map(|r| r.len()).sum::<usize>(), model.elements.len());
        assert_eq!(model.elements.len(), model.strain_triangles + model.bending_hinges + model.stitches);
        for range in &model.colours {
            let mut seen = std::collections::HashSet::new();
            for element in &model.elements[range.start as usize..range.end as usize] {
                for &p in element.particles.iter().filter(|&&p| p != INVALID) {
                    assert!(seen.insert(p), "particle {p} appears twice in one colour");
                }
            }
        }
        assert!(model.colours.len() < 64, "{} colours", model.colours.len());
    }

    #[test]
    fn rest_shape_satisfies_every_cloth_constraint() {
        let (document, topology) = demo();
        let mesh = topology.mesh();
        let model = Model::build(mesh, &document);
        for element in &model.elements {
            for row in element.rows.iter().filter(|r| r.flags & ROW_ACTIVE != 0 && r.flags & ROW_SEAM == 0) {
                let sum = (0..4).filter(|&k| element.particles[k] != INVALID).fold(cad_core::glam::Vec3::ZERO, |acc, k| {
                    acc + mesh.positions[element.particles[k] as usize] * row.weights[k]
                });
                // Strain rows measure 1 (no stretch), bending rows 0 (flat).
                assert!((sum.length() - row.rest).abs() < 2e-3, "C = {}", sum.length() - row.rest);
            }
        }
    }

    #[test]
    fn stiffness_follows_the_grain() {
        let (mut document, topology) = demo();
        let fabric = document.pieces()[0].fabric;
        let physics = &mut document.fabric_mut(fabric).unwrap().physics;
        (physics.stretch_weft, physics.stretch_warp, physics.shear) = (0.1, 0.9, 0.2);
        let model = Model::build(topology.mesh(), &document);
        let front = topology.mesh().span(document.pieces()[0].id).unwrap().vertices.clone();
        let strain = model
            .elements
            .iter()
            .find(|e| e.particles[3] == INVALID && e.particles[2] != INVALID && front.contains(&e.particles[0]))
            .unwrap();
        let [weft, warp, bias, _] = strain.rows.map(|r| r.compliance);
        assert!(warp < weft, "the warp must be stiffer than the weft");
        assert!(bias > warp, "the bias must be softer than the warp");
    }
}

//! CPU backend: the reference implementation of one XPBD sub-step.
//!
//! Positions are `glam::Vec3A`, which is a 128-bit SIMD register on x86-64
//! (SSE2), aarch64 (NEON) and wasm32 (simd128), so the vector arithmetic in
//! the loops below is vectorised. `rayon` spreads particles and the
//! elements of each colour across cores.

use glam::{Quat, Vec3A};
use physics_collision::{ContactPlane, PLANES_PER_VERTEX};
use topology_sync::DistanceField;
use rayon::prelude::*;

use crate::model::{Element, INVALID, Model, ROW_ACTIVE, ROW_SEAM};

/// Particles this close to the collision margin count as resting on the
/// body.
const CONTACT_BAND: f32 = 0.003;

/// Below this many items a loop runs on the calling thread: handing work to
/// the pool costs more than the work.
const PARALLEL_THRESHOLD: usize = 2048;

/// The dynamic state of the cloth.
#[derive(Debug, Clone, Default)]
pub struct State {
    pub positions: Vec<Vec3A>,
    /// Positions at the start of the current sub-step.
    pub previous: Vec<Vec3A>,
    pub velocities: Vec<Vec3A>,
}

impl State {
    pub fn at_rest(positions: impl IntoIterator<Item = Vec3A>) -> Self {
        let positions: Vec<Vec3A> = positions.into_iter().collect();
        Self { previous: positions.clone(), velocities: vec![Vec3A::ZERO; positions.len()], positions }
    }
}

/// An ellipsoid collider in solver form.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipsoid {
    pub center: Vec3A,
    pub radii: Vec3A,
    pub rotation: Quat,
}

/// Parameters of one sub-step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubstepParams {
    /// Sub-step length in seconds.
    pub h: f32,
    pub gravity: Vec3A,
    /// Velocity multiplier per sub-step, `exp(-damping·h)`.
    pub damping: f32,
    /// Remaining fraction of the initial seam gaps, 1 → 0.
    pub seam_gap: f32,
    pub friction: f32,
    pub margin: f32,
    pub ground: f32,
    /// Speed limit in m/s; keeps a bad configuration from exploding.
    pub max_speed: f32,
    /// Also collide triangle centres with the body. Must be the same in
    /// every sub-step: alternating it shakes cloth that rests on the body.
    pub surface_collision: bool,
}

/// The half-spaces every particle must stay in (cloth against cloth);
/// either one entry per particle or empty.
pub type ContactPlanes = [[ContactPlane; PLANES_PER_VERTEX]];

/// Advances `state` by one sub-step. `scratch` is reused between calls.
pub fn substep(
    model: &Model,
    state: &mut State,
    colliders: &[Ellipsoid],
    body: Option<&DistanceField>,
    planes: &ContactPlanes,
    params: &SubstepParams,
    scratch: &mut Vec<[Vec3A; 4]>,
) {
    integrate(model, state, params);
    for colour in &model.colours {
        solve_colour(&model.elements[colour.start as usize..colour.end as usize], model, state, body, params, scratch);
    }
    finalize(model, state, colliders, body, planes, params);
}

/// Predicts positions from velocities and external forces.
fn integrate(model: &Model, state: &mut State, p: &SubstepParams) {
    let step = |(((x, previous), v), &w): (((&mut Vec3A, &mut Vec3A), &mut Vec3A), &f32)| {
        *previous = *x;
        if w > 0.0 {
            *v = ((*v + p.gravity * p.h) * p.damping).clamp_length_max(p.max_speed);
            *x += *v * p.h;
        }
    };
    let State { positions, previous, velocities } = state;
    if positions.len() >= PARALLEL_THRESHOLD {
        positions.par_iter_mut().zip(previous).zip(velocities).zip(&model.inverse_mass).for_each(step);
    } else {
        positions.iter_mut().zip(previous).zip(velocities).zip(&model.inverse_mass).for_each(step);
    }
}

/// Projects all rows of one element and returns the corrected positions of
/// its particles. This is the XPBD update
///
/// ```text
/// Δλ = −C / (Σ wᵢ²/mᵢ + α/h²),   Δxᵢ = (wᵢ/mᵢ)·n·Δλ,   n = L/|L|,  L = Σ wᵢ·xᵢ
/// ```
///
/// with λ starting at zero each sub-step (one iteration per sub-step).
///
/// For cloth triangles the body is then also tested at the triangle's
/// centre (`collision` = field and margin). Vertices alone would let any
/// body detail smaller than a triangle poke through between them; with
/// the centre, no point of the cloth is further than 0.3 edge lengths
/// from a collision sample.
#[inline]
pub fn project(
    element: &Element,
    positions: &[Vec3A],
    inverse_mass: &[f32],
    inv_h2: f32,
    seam_gap: f32,
    collision: Option<(&DistanceField, f32)>,
) -> [Vec3A; 4] {
    let mut x = [Vec3A::ZERO; 4];
    let mut w = [0.0f32; 4];
    for k in 0..4 {
        let particle = element.particles[k];
        if particle != INVALID {
            x[k] = positions[particle as usize];
            w[k] = inverse_mass[particle as usize];
        }
    }
    for row in &element.rows {
        if row.flags & ROW_ACTIVE == 0 {
            continue;
        }
        let c = row.weights;
        let combined = x[0] * c[0] + x[1] * c[1] + x[2] * c[2] + x[3] * c[3];
        let length = combined.length();
        if length < 1e-9 {
            continue; // direction undefined; for rest = 0 it is also solved
        }
        let rest = if row.flags & ROW_SEAM != 0 { row.seam_rest(seam_gap) } else { row.rest };
        // Past its fold limit a bending row stiffens with the fourth power
        // of the overshoot: gentle curves stay soft, sharp creases cannot
        // form.
        let mut compliance = row.compliance;
        if row.fold_limit() > 0.0 && length > row.fold_limit() {
            let ratio = row.fold_limit() / length;
            compliance *= ratio * ratio * ratio * ratio;
        }
        let denominator = w[0] * c[0] * c[0] + w[1] * c[1] * c[1] + w[2] * c[2] * c[2] + w[3] * c[3] * c[3]
            + compliance * inv_h2;
        if denominator < 1e-12 {
            continue; // every particle is pinned
        }
        let correction = combined * (-(length - rest) / (denominator * length));
        for k in 0..4 {
            x[k] += correction * (w[k] * c[k]);
        }
    }

    let is_triangle = element.particles[2] != INVALID && element.particles[3] == INVALID;
    if let (true, Some((field, margin))) = (is_triangle, collision) {
        // Moving all three vertices by the same vector moves the centre
        // by that vector.
        let centre = (x[0] + x[1] + x[2]) / 3.0;
        if let Some((distance, gradient)) = field.sample(centre.into())
            && distance < margin
            && let Some(normal) = Vec3A::from(gradient).try_normalize()
        {
            for k in 0..3 {
                if w[k] > 0.0 {
                    x[k] += normal * (margin - distance);
                }
            }
        }
    }
    x
}

/// Solves one colour. No two elements of a colour share a particle, so the
/// projections are independent: they are computed in parallel into
/// `scratch` and then written back.
fn solve_colour(
    elements: &[Element],
    model: &Model,
    state: &mut State,
    body: Option<&DistanceField>,
    p: &SubstepParams,
    scratch: &mut Vec<[Vec3A; 4]>,
) {
    let inv_h2 = 1.0 / (p.h * p.h);
    let collision = body.filter(|_| p.surface_collision).map(|field| (field, p.margin));
    let positions = &mut state.positions;
    let write = |positions: &mut [Vec3A], element: &Element, solved: &[Vec3A; 4]| {
        for k in 0..4 {
            if element.particles[k] != INVALID {
                positions[element.particles[k] as usize] = solved[k];
            }
        }
    };

    if elements.len() < PARALLEL_THRESHOLD {
        for element in elements {
            let solved = project(element, positions, &model.inverse_mass, inv_h2, p.seam_gap, collision);
            write(positions, element, &solved);
        }
        return;
    }
    scratch.resize(elements.len(), [Vec3A::ZERO; 4]);
    {
        let positions = &*positions;
        scratch.par_iter_mut().zip(elements).with_min_len(512).for_each(|(out, element)| {
            *out = project(element, positions, &model.inverse_mass, inv_h2, p.seam_gap, collision);
        });
    }
    for (element, solved) in elements.iter().zip(scratch.iter()) {
        write(positions, element, solved);
    }
}

/// Pushes `x` out of the collider if it is inside. Returns the corrected
/// position and the contact normal.
#[inline]
fn collide_ellipsoid(x: Vec3A, e: &Ellipsoid, margin: f32) -> Option<(Vec3A, Vec3A)> {
    let radii = e.radii + Vec3A::splat(margin);
    let local = e.rotation.inverse() * (x - e.center);
    let scaled = local / radii;
    let d2 = scaled.length_squared();
    if d2 >= 1.0 || d2 < 1e-12 {
        return None;
    }
    // Project radially in the unit-sphere space: cheap, and exact enough
    // for the shallow penetrations of a sub-step.
    let on_sphere = scaled / d2.sqrt();
    let normal = e.rotation * (on_sphere / radii).normalize();
    Some((e.center + e.rotation * (on_sphere * radii), normal))
}

/// `x` moved out of the ellipsoid (by `margin`), if it was inside.
pub(crate) fn push_out_of_ellipsoid(x: Vec3A, e: &Ellipsoid, margin: f32) -> Option<Vec3A> {
    collide_ellipsoid(x, e, margin).map(|(corrected, _)| corrected)
}

/// Position-based Coulomb friction. `corrected` is `x` pushed out of the
/// collider; the push-out distance stands in for the normal force, so the
/// sliding of this sub-step is reduced by at most `friction` times that
/// distance (static friction when the sliding is smaller).
#[inline]
fn apply_friction(x: Vec3A, corrected: Vec3A, normal: Vec3A, start: Vec3A, friction: f32) -> Vec3A {
    apply_friction_moving(x, corrected, normal, start, friction, Vec3A::ZERO)
}

/// [`apply_friction`] against a surface that moved by `surface` during the
/// sub-step (a body in motion): friction acts on the cloth's motion
/// relative to the skin, so a moving body carries the cloth it presses on,
/// and cloth it moves past faster than friction can hold slides.
fn apply_friction_moving(x: Vec3A, corrected: Vec3A, normal: Vec3A, start: Vec3A, friction: f32, surface: Vec3A) -> Vec3A {
    // How hard the cloth pressed into the surface: by the two closing in
    // on each other this sub-step, at most the penetration. A body posed
    // anew each frame jumps into the cloth by much more than it really
    // pressed; that jump must not count as pressure, or cloth would stick
    // to every limb that moves.
    let pressed = (-(x - start - surface).dot(normal)).max(0.0);
    let depth = (corrected - x).length().min(pressed);
    let motion = corrected - start - surface;
    let sliding = motion - normal * motion.dot(normal);
    let distance = sliding.length();
    if distance < 1e-9 {
        return corrected;
    }
    corrected - sliding * (friction * depth / distance).min(1.0)
}

/// Resolves collisions with friction, then derives velocities from the
/// position change.
fn finalize(
    model: &Model,
    state: &mut State,
    colliders: &[Ellipsoid],
    body: Option<&DistanceField>,
    planes: &ContactPlanes,
    p: &SubstepParams,
) {
    let inv_h = 1.0 / p.h;
    const NO_PLANES: [ContactPlane; PLANES_PER_VERTEX] = [ContactPlane::NONE; PLANES_PER_VERTEX];
    let step = |(i, (((x, previous), v), &w)): (usize, (((&mut Vec3A, &Vec3A), &mut Vec3A), &f32))| {
        if w <= 0.0 {
            return;
        }
        // Other cloth first, so that the body has the last word.
        let mut resting_on_cloth = [None; PLANES_PER_VERTEX];
        for (slot, plane) in planes.get(i).unwrap_or(&NO_PLANES).iter().enumerate() {
            if !plane.is_active() {
                continue;
            }
            let normal = Vec3A::from(plane.normal);
            let distance = plane.distance(*x);
            if distance < 0.0 {
                *x = apply_friction(*x, *x - normal * distance, normal, *previous, p.friction);
            }
            if distance < CONTACT_BAND {
                resting_on_cloth[slot] = Some(normal);
            }
        }
        for collider in colliders {
            if let Some((corrected, normal)) = collide_ellipsoid(*x, collider, p.margin) {
                *x = apply_friction(*x, corrected, normal, *previous, p.friction);
            }
        }
        // The body mesh, through its distance field: move out along the
        // gradient until the particle is `margin` away from the skin.
        let mut resting_on = None;
        let mut skin_velocity = Vec3A::ZERO;
        if let Some(field) = body
            && let Some((distance, gradient)) = field.sample((*x).into())
            && distance < p.margin + CONTACT_BAND
            && let Some(normal) = Vec3A::from(gradient).try_normalize()
        {
            skin_velocity = Vec3A::from(field.velocity_at((*x).into()));
            if distance < p.margin {
                let corrected = *x + normal * (p.margin - distance);
                *x = apply_friction_moving(*x, corrected, normal, *previous, p.friction, skin_velocity * p.h);
            }
            resting_on = Some(normal);
        }
        let floor = p.ground + p.margin;
        if x.y < floor {
            *x = apply_friction(*x, Vec3A::new(x.x, floor, x.z), Vec3A::Y, *previous, p.friction);
        }
        // Clamped here too: a constraint fight must not turn into speed.
        *v = ((*x - *previous) * inv_h).clamp_length_max(p.max_speed);
        // Contact is inelastic: being pushed out of the body must not
        // become outward speed, or cloth under tension bounces on the skin
        // forever instead of coming to rest.
        for normal in resting_on_cloth.into_iter().flatten() {
            *v -= normal * v.dot(normal).max(0.0);
        }
        // Against the body, relative to its skin: cloth it pushes keeps its
        // pace, but does not fly off it.
        if let Some(normal) = resting_on {
            *v -= normal * (*v - skin_velocity).dot(normal).max(0.0);
        }
    };
    let State { positions, previous, velocities } = state;
    if positions.len() >= PARALLEL_THRESHOLD {
        positions.par_iter_mut().zip(&*previous).zip(velocities).zip(&model.inverse_mass).enumerate().for_each(step);
    } else {
        positions.iter_mut().zip(&*previous).zip(velocities).zip(&model.inverse_mass).enumerate().for_each(step);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UP: Vec3A = Vec3A::Y;

    #[test]
    fn a_body_moving_into_sliding_cloth_does_not_grip_it() {
        // Cloth sliding sideways over the skin, 1 mm this sub-step, while the
        // body (posed once per frame) has jumped 2 cm into it.
        let start = Vec3A::new(0.0, 0.0, 0.0);
        let x = Vec3A::new(0.001, 0.0, 0.0);
        let corrected = x + UP * 0.02;
        let out = apply_friction(x, corrected, UP, start, 0.4);
        // Pushed out, still sliding: the body did not grab it.
        assert!((out.y - 0.02).abs() < 1e-6);
        assert!((out.x - 0.001).abs() < 1e-6, "the slide was braked to {}", out.x);
    }

    #[test]
    fn a_moving_body_carries_the_cloth_it_presses_on() {
        // The skin moves 1 mm sideways and 0.5 mm up into still cloth: the
        // cloth comes along sideways rather than staying behind.
        let start = Vec3A::ZERO;
        let x = Vec3A::ZERO;
        let corrected = x + UP * 0.0005;
        let surface = Vec3A::new(0.001, 0.0005, 0.0);
        let out = apply_friction_moving(x, corrected, UP, start, 0.4, surface);
        assert!(out.x > 1e-5, "the cloth stayed behind: {}", out.x);
        assert!(out.x <= 0.001 + 1e-6, "the cloth overtook the body: {}", out.x);
    }

    #[test]
    fn cloth_pressing_into_the_body_still_feels_friction() {
        // Cloth moving down into a still body by 1 mm while slipping
        // sideways 1 mm: friction takes some of the slip.
        let start = Vec3A::new(0.0, 0.0005, 0.0);
        let x = Vec3A::new(0.001, -0.0005, 0.0);
        let corrected = Vec3A::new(0.001, 0.0, 0.0);
        let out = apply_friction(x, corrected, UP, start, 0.4);
        assert!(out.x < 0.001 - 1e-5, "no friction: {}", out.x);
        assert!(out.x > 0.0, "friction reversed the motion: {}", out.x);
    }
}

//! A cloth solver that couples the whole garment in every step.
//!
//! The other backends relax one constraint or one vertex at a time, so a
//! pull at the shoulder needs as many passes to reach the hem as there are
//! triangles in between; stopped earlier, the fabric behaves like rubber.
//! Here every step solves one linear system over all vertices at once
//! (Baraff and Witkin, "Large steps in cloth simulation", 1998): the
//! fabric holds its length at one step per frame.
//!
//! # Method
//!
//! One implicit Euler step minimises
//!
//! ```text
//! E(x) = Σ mᵢ/2h² |xᵢ − yᵢ|² + Σ_rows e(|Σ wᵢ·xᵢ|) + contact + friction
//! ```
//!
//! (`y` is where inertia and gravity alone would take the cloth) by a few
//! Newton steps `x ← x − H⁻¹·∇E`, each checked to lower the energy. `H` is
//! the exact second derivative, with the parts that would make it
//! indefinite left out: a row under compression keeps its stiffness along
//! itself but none across, so compressed fabric is free to buckle into
//! folds, as real fabric does.
//!
//! Contact with the body is a penalty along the body's normal only, so
//! cloth is free to slide; a contact that was stiff in every direction
//! would hold the cloth where it touches. Friction is a separate, physical
//! force: Coulomb's, with the pressure of the previous step and a small
//! stick zone (as in incremental potential contact, Li et al. 2020).
//!
//! The matrix keeps its pattern, so it is ordered and analysed once per
//! mesh ([`faer`]'s sparse Cholesky) and only refactorised in numbers.

use std::ops::Range;
use std::time::Instant;

use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{MatMut, Side};
use glam::{DVec3, Vec3, Vec3A};
use rayon::prelude::*;
use topology_sync::DistanceField;

use crate::cpu::State;
use crate::model::{INVALID, Model, ROW_ACTIVE, ROW_SEAM};

/// Stiffness of a stitch, N/m: far stiffer than fabric, so seams stay shut.
const SEAM_STIFFNESS: f32 = 1e5;
/// Stiffness of the body's surface, N/m per vertex.
const BODY_STIFFNESS: f32 = 4000.0;
/// Relative speed below which friction holds, m/s.
const STICK_SPEED: f32 = 0.01;
/// Largest pressure friction counts with, N per vertex.
const MAX_FRICTION_LOAD: f32 = 0.05;
/// A vertex this far inside the body after the solve is put back on the
/// skin outright, metres.
const TOO_DEEP: f32 = 0.002;
/// Real fabric gives easily for the first few percent (the yarns
/// straighten), then hardly at all (they would have to stretch): beyond
/// this strain along warp or weft, and this one on the bias...
const STRETCH_LIMIT: [f32; 2] = [0.03, 0.10];
/// ...a row is this many times stiffer.
const LOCKED: f32 = 30.0;
/// How much of its stiffness a row keeps when compressed. Fabric pushed
/// together in its plane does not shorten: it buckles into folds, which
/// triangles the size of a fold cannot do unless they are let.
const COMPRESSED: f32 = 0.25;
/// A Newton step is worked out for the contacts there are; a vertex it
/// would take further than this into the body's margin is stopped there
/// (and is a contact for the next step), metres.
const STOP_DEPTH: f32 = 0.00002;
/// How often a Newton step that raised the energy is halved.
const HALVINGS: usize = 5;
/// The steps of a frame stop once none moves a vertex further, metres.
const CONVERGED: f32 = 2e-5;

/// One energy term `e(|Σ wᵢ·xᵢ|)`.
#[derive(Debug, Clone, Copy)]
struct Term {
    particles: [u32; 4],
    weights: [f32; 4],
    /// Where the energy is zero.
    rest: f32,
    /// For seams: the gap they close to.
    closed: f32,
    stiffness: f32,
    seam: bool,
    /// For strain rows: the strain beyond which the fabric locks.
    limit: f32,
}

/// A term at the current shape.
#[derive(Debug, Clone, Copy, Default)]
struct Local {
    /// `e'(s)·d`: the term's part of the gradient, before the weights.
    force: Vec3A,
    /// Its second derivative is `across·I + (along − across)·d·dᵀ`.
    direction: Vec3A,
    along: f32,
    across: f32,
}

impl Term {
    #[inline]
    fn combine(&self, x: &[Vec3A]) -> Vec3A {
        let mut sum = Vec3A::ZERO;
        for k in 0..4 {
            if self.particles[k] != INVALID {
                sum += x[self.particles[k] as usize] * self.weights[k];
            }
        }
        sum
    }

    #[inline]
    fn rest(&self, seam_gap: f32) -> f32 {
        if self.seam {
            self.closed + (self.rest - self.closed) * seam_gap
        } else {
            self.rest
        }
    }

    /// The energy at length `s` with its first and second derivative.
    #[inline]
    fn curve(&self, s: f32, rest: f32) -> (f32, f32, f32) {
        let k = self.stiffness;
        let c = s - rest;
        if self.limit <= 0.0 {
            return (0.5 * k * c * c, k * c, k);
        }
        if c < 0.0 {
            let k = k * COMPRESSED;
            return (0.5 * k * c * c, k * c, k);
        }
        let over = c - self.limit;
        if over <= 0.0 {
            return (0.5 * k * c * c, k * c, k);
        }
        let locked = k * LOCKED;
        (
            0.5 * k * c * c + 0.5 * locked * over * over,
            k * c + locked * over,
            k + locked,
        )
    }

    #[inline]
    fn energy(&self, x: &[Vec3A], seam_gap: f32) -> f32 {
        self.curve(self.combine(x).length(), self.rest(seam_gap)).0
    }

    #[inline]
    fn local(&self, x: &[Vec3A], seam_gap: f32) -> Local {
        let u = self.combine(x);
        let rest = self.rest(seam_gap);
        let s = u.length();
        let (_, slope, curvature) = self.curve(s, rest);
        if rest <= 0.0 || s < 1e-9 {
            // Quadratic: the same stiffness in every direction.
            return Local {
                force: u * self.stiffness,
                direction: Vec3A::ZERO,
                along: self.stiffness,
                across: self.stiffness,
            };
        }
        let d = u / s;
        Local {
            force: d * slope,
            direction: d,
            along: curvature,
            // Stretched, the row resists turning (it would have to get
            // longer); compressed, turning relieves it, which is left out
            // so the matrix stays positive.
            across: (slope / s).max(0.0),
        }
    }
}

/// What a step is given.
pub struct StepInput<'a> {
    pub h: f32,
    pub gravity: Vec3A,
    /// Factor on the velocities per step.
    pub damping: f32,
    pub seam_gap: f32,
    pub friction: f32,
    pub margin: f32,
    pub ground: f32,
    pub max_speed: f32,
    /// Newton steps at most.
    pub iterations: usize,
    /// The body at the end of the frame, and how long after the end of
    /// this step that is: the body is where it will be then, less its
    /// velocity times that.
    pub body: Option<&'a DistanceField>,
    pub body_lag: f32,
}

impl StepInput<'_> {
    /// Distance to the body at the end of this step, the direction away
    /// from it and its surface velocity there.
    #[inline]
    fn body_at(&self, p: Vec3A) -> Option<(f32, Vec3A, Vec3A)> {
        let field = self.body?;
        let velocity = Vec3A::from(field.velocity_at(Vec3::from(p)));
        let (distance, slope) = field.sample((p + velocity * self.body_lag).into())?;
        Some((distance, Vec3A::from(slope).try_normalize()?, velocity))
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GlobalStats {
    pub contacts: usize,
    pub factorisations: usize,
    pub halvings: usize,
    pub assemble_us: u64,
    pub factor_us: u64,
    pub solve_us: u64,
    /// The last Newton step's largest vertex movement, metres.
    pub last_step: f32,
}

/// Two vertices that share a term (or one vertex with itself), and where
/// their 3×3 block sits in the matrix.
#[derive(Debug, Clone)]
struct Pair {
    /// `i >= j`: the block is in the lower triangle.
    i: u32,
    j: u32,
    /// Start of the block's three rows in each of its three columns.
    at: [usize; 3],
    /// Into `contributions`.
    terms: Range<u32>,
}

/// A vertex against the body (and the floor) at the current shape.
#[derive(Debug, Clone, Copy, Default)]
struct Contact {
    gradient: Vec3A,
    /// Second derivative, lower triangle: xx yx zx yy zy zz.
    hessian: [f32; 6],
    pressure: f32,
    energy: f32,
}

#[inline]
fn outer(a: Vec3A, scale: f32) -> [f32; 6] {
    [
        a.x * a.x * scale,
        a.y * a.x * scale,
        a.z * a.x * scale,
        a.y * a.y * scale,
        a.z * a.y * scale,
        a.z * a.z * scale,
    ]
}

/// Coulomb friction smoothed over the stick zone: the force (as a
/// fraction of µ·load) at slide `distance`, and the energy.
#[inline]
fn friction_curve(distance: f32, stick: f32) -> (f32, f32) {
    let z = distance / stick;
    if z < 1.0 {
        (2.0 * z - z * z, stick * (z * z - z * z * z / 3.0))
    } else {
        (1.0, distance - stick / 3.0)
    }
}

pub struct Global {
    terms: Vec<Term>,
    /// Per vertex: its terms and its slot in each.
    incident: Vec<Vec<(u32, u8)>>,
    mass: Vec<f32>,
    pairs: Vec<Pair>,
    /// Per pair entry: the term and the product of the two weights.
    contributions: Vec<(u32, f32)>,
    /// Per vertex: its own pair.
    own: Vec<u32>,
    // The matrix, lower triangle, compressed columns; three rows per vertex.
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    values: Vec<f64>,
    symbolic: Option<SymbolicLlt<usize>>,
    /// Pressure on each vertex at the end of the last step, N.
    load: Vec<f32>,
    // Scratch.
    target: Vec<Vec3A>,
    start: Vec<Vec3A>,
    trial: Vec<Vec3A>,
    locals: Vec<Local>,
    rhs: Vec<f64>,
    pub stats: GlobalStats,
}

impl Default for Global {
    fn default() -> Self {
        Self::new(&Model::default())
    }
}

impl Global {
    pub fn new(model: &Model) -> Self {
        // The systems are small: the library's threads cost more than
        // they save.
        faer::set_global_parallelism(faer::Par::Seq);
        let n = model.inverse_mass.len();
        let mut terms = Vec::new();
        for element in &model.elements {
            // A triangle's four strain rows: weft, warp and the two bias.
            let strain = element.particles[2] != INVALID
                && element.particles[3] == INVALID
                && element.rows[0].rest == 1.0;
            for (slot, row) in element.rows.iter().enumerate() {
                if row.flags & ROW_ACTIVE == 0 {
                    continue;
                }
                let seam = row.flags & ROW_SEAM != 0;
                terms.push(Term {
                    particles: element.particles,
                    weights: row.weights,
                    rest: row.rest,
                    closed: if seam { row.fold_limit } else { 0.0 },
                    stiffness: if seam {
                        SEAM_STIFFNESS
                    } else {
                        1.0 / row.compliance.max(1e-12)
                    },
                    seam,
                    limit: if strain && !seam {
                        STRETCH_LIMIT[slot / 2]
                    } else {
                        0.0
                    },
                });
            }
        }
        let mut incident: Vec<Vec<(u32, u8)>> = vec![Vec::new(); n];
        // Per column j: the rows i ≥ j it is coupled with, and through what.
        let mut coupling: Vec<std::collections::BTreeMap<u32, Vec<(u32, f32)>>> =
            vec![std::collections::BTreeMap::new(); n];
        for (j, column) in coupling.iter_mut().enumerate() {
            column.insert(j as u32, Vec::new());
        }
        for (index, term) in terms.iter().enumerate() {
            for a in 0..4 {
                let i = term.particles[a];
                if i == INVALID {
                    continue;
                }
                incident[i as usize].push((index as u32, a as u8));
                for b in 0..4 {
                    let j = term.particles[b];
                    if j == INVALID || j > i {
                        continue;
                    }
                    coupling[j as usize]
                        .entry(i)
                        .or_default()
                        .push((index as u32, term.weights[a] * term.weights[b]));
                }
            }
        }
        let (mut col_ptr, mut row_idx) = (vec![0usize], Vec::new());
        let (mut pairs, mut contributions, mut own) = (Vec::new(), Vec::new(), vec![0u32; n]);
        for (j, column) in coupling.iter().enumerate() {
            let first = pairs.len();
            for (&i, through) in column {
                let begin = contributions.len() as u32;
                contributions.extend_from_slice(through);
                pairs.push(Pair {
                    i,
                    j: j as u32,
                    at: [0; 3],
                    terms: begin..contributions.len() as u32,
                });
            }
            own[j] = first as u32;
            for a in 0..3 {
                for pair in &mut pairs[first..] {
                    pair.at[a] = row_idx.len();
                    if pair.i == pair.j {
                        // Its own block: the lower triangle only.
                        row_idx.extend((a..3).map(|b| 3 * j + b));
                    } else {
                        row_idx.extend((0..3).map(|b| 3 * pair.i as usize + b));
                    }
                }
                col_ptr.push(row_idx.len());
            }
        }
        let symbolic = (n > 0)
            .then(|| {
                SymbolicLlt::try_new(
                    SymbolicSparseColMatRef::new_checked(3 * n, 3 * n, &col_ptr, None, &row_idx),
                    Side::Lower,
                )
                .ok()
            })
            .flatten();
        Self {
            incident,
            mass: model
                .inverse_mass
                .iter()
                .map(|&w| if w > 0.0 { 1.0 / w } else { 0.0 })
                .collect(),
            values: vec![0.0; row_idx.len()],
            pairs,
            contributions,
            own,
            col_ptr,
            row_idx,
            symbolic,
            load: vec![0.0; n],
            target: Vec::new(),
            start: Vec::new(),
            trial: Vec::new(),
            locals: vec![Local::default(); terms.len()],
            terms,
            rhs: vec![0.0; 3 * n],
            stats: GlobalStats::default(),
        }
    }

    /// Forgets the pressures: the cloth was moved by hand.
    pub fn reset(&mut self) {
        self.load.fill(0.0);
    }

    /// One time step of length `input.h`.
    pub fn step(&mut self, state: &mut State, input: &StepInput<'_>) {
        let n = state.positions.len();
        if n == 0 || n != self.mass.len() {
            return;
        }
        let Some(symbolic) = self.symbolic.clone() else {
            return;
        };
        self.stats = GlobalStats::default();
        let h = input.h;
        let stick = STICK_SPEED * h;
        let inertia = 1.0 / (h * h);
        let (margin, seam_gap) = (input.margin, input.seam_gap);
        self.start.clone_from(&state.positions);
        // Where inertia and gravity alone take the cloth.
        self.target.clear();
        self.target.extend((0..n).map(|i| {
            if self.mass[i] > 0.0 {
                let v = (state.velocities[i] * input.damping).clamp_length_max(input.max_speed);
                self.start[i] + v * h + input.gravity * (h * h)
            } else {
                self.start[i]
            }
        }));

        // Per vertex: the body's normal and velocity where the step began,
        // which friction works against for the whole step.
        let frame: Vec<Option<(Vec3A, Vec3A)>> = if input.friction > 0.0 {
            self.start
                .par_iter()
                .zip(&self.load)
                .map(|(&p, &load)| {
                    if load <= 0.0 {
                        return None;
                    }
                    let (_, normal, velocity) = input.body_at(p)?;
                    Some((normal, velocity))
                })
                .collect()
        } else {
            vec![None; n]
        };
        let (start, load, friction) = (&self.start, &self.load, input.friction);
        // The body, the floor and friction on vertex `i` at `p`.
        let contact = |i: usize, p: Vec3A| -> Contact {
            let mut c = Contact::default();
            let mut touch = |normal: Vec3A, depth: f32| {
                c.pressure = BODY_STIFFNESS * depth;
                c.gradient -= normal * c.pressure;
                c.energy += 0.5 * BODY_STIFFNESS * depth * depth;
                c.hessian = outer(normal, BODY_STIFFNESS);
            };
            if let Some((distance, normal, _)) = input.body_at(p)
                && distance < margin
            {
                touch(normal, margin - distance);
            } else if p.y < input.ground + margin {
                touch(Vec3A::Y, input.ground + margin - p.y);
            }
            if let Some((normal, velocity)) = frame[i] {
                // How far the vertex has slid on the body since the step began.
                let moved = p - start[i] - velocity * h;
                let slid = moved - normal * moved.dot(normal);
                let distance = slid.length();
                let limit = friction * load[i].min(MAX_FRICTION_LOAD);
                let (force, energy) = friction_curve(distance, stick);
                c.energy += limit * energy;
                if distance > 1e-9 {
                    let along = slid / distance;
                    c.gradient += along * (limit * force);
                    // Across the slide: force / distance. Along it: the
                    // curve's slope.
                    let across = limit * force / distance;
                    let slope = if distance < stick {
                        limit * (2.0 - 2.0 * distance / stick) / stick
                    } else {
                        0.0
                    };
                    let (t, a, nn) = (
                        outer(along, slope - across),
                        outer(normal, -across),
                        [across, 0.0, 0.0, across, 0.0, across],
                    );
                    for k in 0..6 {
                        c.hessian[k] += t[k] + a[k] + nn[k];
                    }
                } else {
                    let across = limit * 2.0 / stick;
                    let (a, nn) = (
                        outer(normal, -across),
                        [across, 0.0, 0.0, across, 0.0, across],
                    );
                    for k in 0..6 {
                        c.hessian[k] += a[k] + nn[k];
                    }
                }
            }
            c
        };
        let (terms, mass, target) = (&self.terms, &self.mass, &self.target);
        let energy = |x: &[Vec3A]| -> f64 {
            let elastic: f64 = terms
                .par_iter()
                .with_min_len(1024)
                .map(|term| f64::from(term.energy(x, seam_gap)))
                .sum();
            let rest: f64 = (0..n)
                .into_par_iter()
                .with_min_len(512)
                .map(|i| {
                    f64::from(0.5 * mass[i] * inertia * (x[i] - target[i]).length_squared())
                        + f64::from(contact(i, x[i]).energy)
                })
                .sum();
            elastic + rest
        };

        // The first step is taken from where the cloth is: there its forces
        // are (nearly) in balance, which is the best place to linearise.
        let mut current = energy(&state.positions);
        let mut pressures = vec![0.0f32; n];
        for _ in 0..input.iterations.max(1) {
            let clock = Instant::now();
            let x = &state.positions;
            self.locals
                .par_iter_mut()
                .zip(terms)
                .with_min_len(512)
                .for_each(|(local, term)| *local = term.local(x, seam_gap));
            let locals = &self.locals;
            let contacts: Vec<Contact> = (0..n)
                .into_par_iter()
                .with_min_len(256)
                .map(|i| contact(i, x[i]))
                .collect();
            self.stats.contacts = contacts.iter().filter(|c| c.pressure > 0.0).count();
            for (pressure, c) in pressures.iter_mut().zip(&contacts) {
                *pressure = c.pressure;
            }
            let incident = &self.incident;
            let gradient: Vec<Vec3A> = (0..n)
                .into_par_iter()
                .with_min_len(256)
                .map(|i| {
                    if mass[i] <= 0.0 {
                        return Vec3A::ZERO;
                    }
                    let mut g = (x[i] - target[i]) * (mass[i] * inertia) + contacts[i].gradient;
                    for &(term, slot) in &incident[i] {
                        g += locals[term as usize].force * terms[term as usize].weights[slot as usize];
                    }
                    g
                })
                .collect();
            // The matrix: every pair's block, summed over its terms.
            let contributions = &self.contributions;
            let blocks: Vec<[f64; 6]> = self
                .pairs
                .par_iter()
                .with_min_len(256)
                .map(|pair| {
                    let (mut identity, mut dyad) = (0.0f64, [0.0f64; 6]);
                    for &(term, weight) in
                        &contributions[pair.terms.start as usize..pair.terms.end as usize]
                    {
                        let local = &locals[term as usize];
                        identity += f64::from(weight * local.across);
                        let d = local.direction;
                        let scale = weight * (local.along - local.across);
                        for (sum, value) in dyad.iter_mut().zip(outer(d, scale)) {
                            *sum += f64::from(value);
                        }
                    }
                    dyad[0] += identity;
                    dyad[3] += identity;
                    dyad[5] += identity;
                    dyad
                })
                .collect();
            for (pair, block) in self.pairs.iter().zip(&blocks) {
                // xx yx zx yy zy zz, symmetric.
                let full = [
                    [block[0], block[1], block[2]],
                    [block[1], block[3], block[4]],
                    [block[2], block[4], block[5]],
                ];
                for a in 0..3 {
                    if pair.i == pair.j {
                        for b in a..3 {
                            self.values[pair.at[a] + b - a] = full[b][a];
                        }
                    } else {
                        for b in 0..3 {
                            self.values[pair.at[a] + b] = full[b][a];
                        }
                    }
                }
            }
            for i in 0..n {
                let pair = &self.pairs[self.own[i] as usize];
                // A pinned vertex is held by a mass no force moves.
                let m = if mass[i] > 0.0 {
                    f64::from(mass[i] * inertia)
                } else {
                    1e12
                };
                let c = contacts[i].hessian.map(f64::from);
                let lower = [[c[0] + m, c[1], c[2]], [0.0, c[3] + m, c[4]], [0.0, 0.0, c[5] + m]];
                for a in 0..3 {
                    for b in a..3 {
                        self.values[pair.at[a] + b - a] += lower[a][b];
                    }
                }
            }
            self.stats.assemble_us += clock.elapsed().as_micros() as u64;

            let clock = Instant::now();
            let matrix = SparseColMatRef::new(
                SymbolicSparseColMatRef::new_checked(
                    3 * n,
                    3 * n,
                    &self.col_ptr,
                    None,
                    &self.row_idx,
                ),
                &self.values,
            );
            let Ok(factor) = Llt::try_new_with_symbolic(symbolic.clone(), matrix, Side::Lower)
            else {
                log::warn!("global cloth solve: the matrix could not be factorised");
                break;
            };
            self.stats.factorisations += 1;
            self.stats.factor_us += clock.elapsed().as_micros() as u64;

            let clock = Instant::now();
            for (i, g) in gradient.iter().enumerate() {
                self.rhs[3 * i] = -f64::from(g.x);
                self.rhs[3 * i + 1] = -f64::from(g.y);
                self.rhs[3 * i + 2] = -f64::from(g.z);
            }
            factor.solve_in_place(MatMut::from_column_major_slice_mut(&mut self.rhs, 3 * n, 1));
            let step = |i: usize| {
                DVec3::new(self.rhs[3 * i], self.rhs[3 * i + 1], self.rhs[3 * i + 2]).as_vec3a()
            };
            if (0..n).any(|i| !step(i).is_finite()) {
                log::warn!("global cloth solve: the step is not finite");
                break;
            }
            let size = (0..n).map(|i| step(i).length()).fold(0.0f32, f32::max);
            self.stats.solve_us += clock.elapsed().as_micros() as u64;

            // Take the step, or the largest part of it that lowers the energy.
            let mut scale = 1.0f32;
            let mut taken = false;
            for _ in 0..=HALVINGS {
                self.trial.clear();
                self.trial
                    .extend((0..n).map(|i| state.positions[i] + step(i) * scale));
                if input.body.is_some() {
                    self.trial.par_iter_mut().with_min_len(256).for_each(|p| {
                        if let Some((distance, normal, _)) = input.body_at(*p)
                            && distance < margin - STOP_DEPTH
                        {
                            *p += normal * (margin - STOP_DEPTH - distance);
                        }
                    });
                }
                let next = energy(&self.trial);
                if next <= current {
                    current = next;
                    taken = true;
                    break;
                }
                scale *= 0.5;
                self.stats.halvings += 1;
            }
            if !taken {
                break;
            }
            std::mem::swap(&mut state.positions, &mut self.trial);
            self.stats.last_step = size * scale;
            if size * scale < CONVERGED {
                break;
            }
        }
        self.load.clone_from(&pressures);

        // What the penalty left too deep is put on the skin.
        if input.body.is_some() {
            state.positions.par_iter_mut().for_each(|p| {
                if let Some((distance, normal, _)) = input.body_at(*p)
                    && distance < -TOO_DEEP
                {
                    *p -= normal * distance;
                }
            });
        }
        for i in 0..n {
            state.velocities[i] = if self.mass[i] > 0.0 {
                ((state.positions[i] - self.start[i]) / h).clamp_length_max(input.max_speed)
            } else {
                Vec3A::ZERO
            };
        }
        state.previous.clone_from(&self.start);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use topology_sync::{GarmentTopology, TopologySync};

    fn demo() -> (Model, State) {
        let document = cad_core::PatternDocument::demo_top();
        let mut topology = GarmentTopology::default();
        topology.rebuild(&document);
        let mesh = topology.mesh();
        (
            Model::build(mesh, &document),
            State::at_rest(mesh.positions.iter().map(|&p| Vec3A::from(p))),
        )
    }

    fn input(iterations: usize) -> StepInput<'static> {
        StepInput {
            h: 1.0 / 30.0,
            gravity: Vec3A::new(0.0, -9.81, 0.0),
            damping: 1.0,
            seam_gap: 1.0,
            friction: 0.4,
            margin: 0.004,
            ground: -100.0,
            max_speed: 20.0,
            iterations,
            body: None,
            body_lag: 0.0,
        }
    }

    #[test]
    fn free_cloth_falls_like_a_stone() {
        // Nothing holds it: every vertex must follow gravity exactly,
        // which only a solve that leaves rigid motion alone does.
        let (model, mut state) = demo();
        let mut global = Global::new(&model);
        let before = state.positions.clone();
        let steps = 15;
        for _ in 0..steps {
            global.step(&mut state, &input(2));
        }
        // Implicit Euler: h²·g·(1 + 2 + … + steps).
        let h = 1.0f32 / 30.0;
        let fall = 9.81 * h * h * (steps * (steps + 1) / 2) as f32;
        for (a, b) in state.positions.iter().zip(&before) {
            assert!(((b.y - a.y) - fall).abs() < 1e-3, "{} vs {fall}", b.y - a.y);
            assert!((a.x - b.x).abs() < 1e-3 && (a.z - b.z).abs() < 1e-3);
        }
    }

    #[test]
    #[ignore = "a timing"]
    fn factorisation_time() {
        let (model, mut state) = demo();
        let mut global = Global::new(&model);
        let clock = Instant::now();
        let steps = 20;
        let (mut factorisations, mut factor, mut assemble, mut solve) = (0, 0, 0, 0);
        for _ in 0..steps {
            global.step(&mut state, &input(2));
            factorisations += global.stats.factorisations;
            factor += global.stats.factor_us;
            assemble += global.stats.assemble_us;
            solve += global.stats.solve_us;
        }
        println!(
            "{} vertices, {} matrix entries: {:?} per step; per factorisation {} us, assembly {} us, solve {} us",
            model.inverse_mass.len(),
            global.row_idx.len(),
            clock.elapsed() / steps,
            factor / factorisations as u64,
            assemble / factorisations as u64,
            solve / factorisations as u64,
        );
    }
}

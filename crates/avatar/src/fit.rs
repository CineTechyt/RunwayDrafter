//! Solves the body parameters so that measurements read given values:
//! one at a time ([`fit`]) or a whole measurement chart at once
//! ([`fit_all`]).

use crate::measure::{Measure, measure};
use crate::model::{BodyModel, BodyParams, Morph};

/// Half a millimetre: finer than any tape.
const TOLERANCE: f32 = 0.0005;

/// The morphs that drive a measurement. `Height` is handled separately.
fn controls(which: Measure) -> &'static [Morph] {
    match which {
        Measure::Height => &[],
        Measure::Neck => &[Morph::NeckGirth],
        Measure::Shoulders => &[Morph::ShoulderWidth],
        Measure::Bust => &[Morph::BustGirth],
        Measure::Waist => &[Morph::WaistGirth],
        Measure::Hips => &[Morph::HipGirth],
        Measure::Inseam => &[Morph::UpperLegLength, Morph::LowerLegLength],
        Measure::Thigh => &[Morph::ThighGirth],
        Measure::Calf => &[Morph::CalfGirth],
        Measure::Ankle => &[Morph::AnkleGirth],
        Measure::ArmLength => &[Morph::UpperArmLength, Morph::LowerArmLength],
        Measure::UpperArm => &[Morph::UpperArmGirth],
        Measure::Wrist => &[Morph::WristGirth],
    }
}

/// Secant search for `value` in `low..=high` such that `read(value)` equals
/// `target`. Returns the best value found.
fn solve(mut read: impl FnMut(f32) -> f32, start: f32, low: f32, high: f32, target: f32) -> f32 {
    let mut x0 = start.clamp(low, high);
    let mut f0 = read(x0) - target;
    let mut best = (f0.abs(), x0);
    if f0.abs() < TOLERANCE {
        return x0;
    }
    // Every control increases its measurement, so step towards the target.
    let mut x1 = (x0 + if f0 < 0.0 { 0.25 } else { -0.25 }).clamp(low, high);
    for _ in 0..10 {
        if (x1 - x0).abs() < 1e-5 {
            break;
        }
        let f1 = read(x1) - target;
        if f1.abs() < best.0 {
            best = (f1.abs(), x1);
        }
        if f1.abs() < TOLERANCE || (f1 - f0).abs() < 1e-9 {
            break;
        }
        let next = (x1 - f1 * (x1 - x0) / (f1 - f0)).clamp(low, high);
        (x0, f0, x1) = (x1, f1, next);
    }
    best.1
}

/// Adjusts `params` so that `which` measures `target` metres, as far as the
/// model allows, and returns the value actually reached.
///
/// Each measurement has its own shape control, so fitting one leaves the
/// others largely (not exactly) unchanged. Height is reached by changing
/// the build first and, beyond the sculpted range, by scaling the whole
/// body.
pub fn fit(model: &BodyModel, params: &mut BodyParams, which: Measure, target: f32) -> f32 {
    let read = |params: &BodyParams| measure(model, &model.build(params)).get(which);

    if which == Measure::Height {
        params.scale = 1.0;
        let mut trial = *params;
        params.stature = solve(
            |stature| {
                trial.stature = stature;
                read(&trial)
            },
            params.stature,
            -1.0,
            1.0,
            target,
        );
        let reached = read(params);
        if (reached - target).abs() > TOLERANCE && reached > 0.0 {
            params.scale = (target / reached).clamp(0.5, 1.5);
        }
        return read(params);
    }

    let morphs = controls(which);
    let Some(&first) = morphs.first() else {
        return read(params);
    };
    let mut trial = *params;
    let value = solve(
        |value| {
            for &morph in morphs {
                trial.set_morph(morph, value);
            }
            read(&trial)
        },
        params.morph(first),
        -BodyParams::MORPH_LIMIT,
        BodyParams::MORPH_LIMIT,
        target,
    );
    for &morph in morphs {
        params.set_morph(morph, value);
    }
    read(params)
}

/// Result of [`fit_all`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FitReport {
    /// Largest difference between a target and the fitted body, in metres.
    pub worst_error: f32,
    /// Targets the model could not reach, with the value it got to.
    pub unreachable: Vec<(Measure, f32)>,
    /// How much the body fat slider was moved to bring girths into reach.
    pub weight_change: f32,
}

impl FitReport {
    /// Within a millimetre on every target.
    pub fn exact(&self) -> bool {
        self.worst_error < FitReport::TOLERANCE
    }

    pub const TOLERANCE: f32 = 0.001;
}

/// One degree of freedom of the solve.
#[derive(Clone, Copy)]
enum Control {
    /// Uniform scale; drives the height.
    Scale,
    Morphs(&'static [Morph]),
}

impl Control {
    fn of(which: Measure) -> Self {
        if which == Measure::Height {
            Control::Scale
        } else {
            Control::Morphs(controls(which))
        }
    }

    fn get(self, params: &BodyParams) -> f32 {
        match self {
            Control::Scale => params.scale,
            Control::Morphs(morphs) => params.morph(morphs[0]),
        }
    }

    fn set(self, params: &mut BodyParams, value: f32) {
        let (low, high) = self.bounds();
        match self {
            Control::Scale => params.scale = value.clamp(low, high),
            Control::Morphs(morphs) => morphs.iter().for_each(|&m| params.set_morph(m, value)),
        }
    }

    fn bounds(self) -> (f32, f32) {
        match self {
            Control::Scale => (0.7, 1.35),
            Control::Morphs(_) => (-BodyParams::MORPH_LIMIT, BodyParams::MORPH_LIMIT),
        }
    }

    /// Finite-difference step and the largest change per iteration.
    fn steps(self) -> (f32, f32) {
        match self {
            Control::Scale => (0.01, 0.2),
            Control::Morphs(_) => (0.1, 0.8),
        }
    }
}

/// Solves `a·x = b` for a small dense system by Gaussian elimination with
/// partial pivoting. Singular directions get a zero step.
fn solve_linear(mut a: Vec<Vec<f32>>, mut b: Vec<f32>) -> Vec<f32> {
    let n = b.len();
    for column in 0..n {
        let pivot = (column..n)
            .max_by(|&i, &j| a[i][column].abs().total_cmp(&a[j][column].abs()))
            .unwrap_or(column);
        a.swap(column, pivot);
        b.swap(column, pivot);
        if a[column][column].abs() < 1e-12 {
            continue;
        }
        for row in column + 1..n {
            let factor = a[row][column] / a[column][column];
            for k in column..n {
                a[row][k] -= factor * a[column][k];
            }
            b[row] -= factor * b[column];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let rest: f32 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = if a[row][row].abs() < 1e-12 {
            0.0
        } else {
            (b[row] - rest) / a[row][row]
        };
    }
    x
}

/// Newton iterations on all controls at once. Returns the measured values.
fn newton(model: &BodyModel, params: &mut BodyParams, targets: &[(Measure, f32)]) -> Vec<f32> {
    use rayon::prelude::*;
    let controls: Vec<Control> = targets
        .iter()
        .map(|(which, _)| Control::of(*which))
        .collect();
    let read = |params: &BodyParams| {
        let measured = measure(model, &model.build(params));
        targets
            .iter()
            .map(|(which, _)| measured.get(*which))
            .collect::<Vec<f32>>()
    };
    let n = targets.len();
    let mut measured = read(params);
    for _ in 0..10 {
        let residual: Vec<f32> = targets
            .iter()
            .zip(&measured)
            .map(|((_, target), m)| target - m)
            .collect();
        if residual
            .iter()
            .all(|r| r.abs() < 0.2 * FitReport::TOLERANCE)
        {
            break;
        }
        // Jacobian by finite differences, one column per control. Each
        // column is a full rebuild and re-measure, so they run in parallel.
        let columns: Vec<Vec<f32>> = controls
            .par_iter()
            .map(|control| {
                let (step, _) = control.steps();
                let value = control.get(params);
                // Step inwards when sitting on the upper bound.
                let step = if value + step > control.bounds().1 {
                    -step
                } else {
                    step
                };
                let mut trial = *params;
                control.set(&mut trial, value + step);
                read(&trial)
                    .iter()
                    .zip(&measured)
                    .map(|(moved, base)| (moved - base) / step)
                    .collect()
            })
            .collect();

        // Damped least squares: (JᵀJ + λI)·dx = Jᵀ·r. The damping keeps
        // the step finite when a control has run out of effect.
        let mut normal = vec![vec![0.0f32; n]; n];
        let mut rhs = vec![0.0f32; n];
        for i in 0..n {
            for j in 0..n {
                normal[i][j] = (0..n).map(|k| columns[i][k] * columns[j][k]).sum();
            }
            rhs[i] = (0..n).map(|k| columns[i][k] * residual[k]).sum();
        }
        let damping = 1e-6 * (0..n).map(|i| normal[i][i]).sum::<f32>().max(1e-9);
        for (i, row) in normal.iter_mut().enumerate() {
            row[i] += damping;
        }
        let step = solve_linear(normal, rhs);

        let before = *params;
        for (control, dx) in controls.iter().zip(step) {
            let (_, limit) = control.steps();
            control.set(params, control.get(params) + dx.clamp(-limit, limit));
        }
        if *params == before {
            break; // every control is pinned at a bound
        }
        measured = read(params);
    }
    measured
}

/// Reshapes the body so that every measurement in `targets` reads its
/// value at the same time.
///
/// All shape controls are solved together, so unlike repeated single
/// [`fit`]s the result does not depend on the order and earlier targets are
/// not disturbed by later ones. Height is reached by scaling the body;
/// `stature` (the build) is left to the user.
///
/// A girth can only move so far on its own. When girths end up out of
/// reach, the body fat is adjusted, as that is what distinguishes such a
/// body from the current one, and the solve is repeated.
pub fn fit_all(
    model: &BodyModel,
    params: &mut BodyParams,
    targets: &[(Measure, f32)],
) -> FitReport {
    if targets.is_empty() {
        return FitReport::default();
    }
    const GIRTHS: [Measure; 7] = [
        Measure::Bust,
        Measure::Waist,
        Measure::Hips,
        Measure::Thigh,
        Measure::Calf,
        Measure::UpperArm,
        Measure::Neck,
    ];
    let start_weight = params.weight;
    let mut best: Option<(f32, BodyParams, Vec<f32>)> = None;
    for _ in 0..7 {
        let measured = newton(model, params, targets);
        let worst = targets
            .iter()
            .zip(&measured)
            .map(|((_, t), m)| (t - m).abs())
            .fold(0.0, f32::max);
        if best.as_ref().is_none_or(|(error, ..)| worst < *error) {
            best = Some((worst, *params, measured.clone()));
        }
        if worst < FitReport::TOLERANCE {
            break;
        }
        // Do the girths that fell short want a heavier or a lighter body?
        let pull: f32 = targets
            .iter()
            .zip(&measured)
            .filter(|((which, target), m)| {
                GIRTHS.contains(which) && (target - **m).abs() >= FitReport::TOLERANCE
            })
            .map(|((_, target), m)| (target - m).signum())
            .sum();
        let weight = (params.weight + 0.12 * pull.signum()).clamp(0.0, 1.0);
        if pull == 0.0 || weight == params.weight {
            break;
        }
        params.weight = weight;
    }

    let (worst_error, fitted, measured) = best.expect("at least one solve ran");
    *params = fitted;
    FitReport {
        worst_error,
        unreachable: targets
            .iter()
            .zip(&measured)
            .filter(|((_, target), m)| (target - **m).abs() >= FitReport::TOLERANCE)
            .map(|((which, _), m)| (*which, *m))
            .collect(),
        weight_change: params.weight - start_weight,
    }
}

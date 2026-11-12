use avatar::{BodyModel, BodyParams, Measure, distance_field, fit, fit_all, measure};
use glam::Vec3;

fn model() -> BodyModel {
    BodyModel::load(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/avatar/body.bin")).expect("run tools/bake_avatar.py")
}

fn report(label: &str, m: &avatar::Measurements) {
    println!("{label}: {:.1} kg, levels bust {:.3} waist {:.3} hip {:.3} armpit {:.3}", m.weight_kg, m.bust_level, m.waist_level, m.hip_level, m.armpit_level);
    for which in Measure::ALL {
        println!("  {:<32} {:6.1} cm", which.label(), m.get(which) * 100.0);
    }
}

#[test]
fn default_bodies_have_human_proportions() {
    let model = model();
    let female = measure(&model, &model.build(&BodyParams::female()));
    let male = measure(&model, &model.build(&BodyParams::male()));
    report("female", &female);
    report("male", &male);

    for (m, height, bust, waist, hips) in [(&female, 1.55..1.75, 0.80..1.00, 0.60..0.80, 0.85..1.05), (&male, 1.68..1.88, 0.90..1.10, 0.72..0.92, 0.88..1.08)] {
        assert!(height.contains(&m.get(Measure::Height)));
        assert!(bust.contains(&m.get(Measure::Bust)));
        assert!(waist.contains(&m.get(Measure::Waist)));
        assert!(hips.contains(&m.get(Measure::Hips)));
        assert!((0.42..0.50).contains(&(m.get(Measure::Inseam) / m.get(Measure::Height))), "legs are about 45-48 % of the height");
        assert!((0.28..0.42).contains(&m.get(Measure::Neck)));
        assert!((0.30..0.48).contains(&m.get(Measure::Shoulders)));
        assert!((0.45..0.70).contains(&m.get(Measure::Thigh)));
        assert!((0.28..0.45).contains(&m.get(Measure::Calf)));
        assert!((0.17..0.28).contains(&m.get(Measure::Ankle)));
        assert!((0.20..0.40).contains(&m.get(Measure::UpperArm)));
        assert!((0.115..0.20).contains(&m.get(Measure::Wrist)));
        assert!((0.40..0.65).contains(&m.get(Measure::ArmLength)));
        assert!((40.0..110.0).contains(&m.weight_kg));
    }
    // The differences a pattern maker relies on.
    assert!(male.get(Measure::Height) > female.get(Measure::Height));
    assert!(male.get(Measure::Shoulders) > female.get(Measure::Shoulders));
    assert!(male.get(Measure::Waist) / male.get(Measure::Hips) > female.get(Measure::Waist) / female.get(Measure::Hips));
}

#[test]
fn fitting_reaches_the_requested_measurements() {
    let model = model();
    let mut params = BodyParams::female();
    let targets = [
        (Measure::Height, 1.72),
        (Measure::Bust, 0.96),
        (Measure::Waist, 0.74),
        (Measure::Hips, 1.02),
        (Measure::Inseam, 0.80),
        (Measure::Thigh, 0.58),
        (Measure::Neck, 0.34),
        (Measure::UpperArm, 0.29),
    ];
    // A few rounds: the controls overlap (leg length changes the height,
    // for instance), so later rounds remove what fits did to each other.
    for _ in 0..4 {
        for (which, target) in targets {
            fit(&model, &mut params, which, target);
        }
    }
    let result = measure(&model, &model.build(&params));
    report("fitted", &result);
    for (which, target) in targets {
        assert!((result.get(which) - target).abs() < 0.006, "{} is {:.1} cm, wanted {:.1} cm", which.label(), result.get(which) * 100.0, target * 100.0);
    }
}

#[test]
fn distance_field_matches_the_body() {
    let model = model();
    let mesh = model.build(&BodyParams::male());
    let started = std::time::Instant::now();
    let field = distance_field(&mesh, 0.008, 0.0);
    println!("field {:?} at {:.1} mm built in {:.0} ms", field.dims, field.voxel * 1000.0, started.elapsed().as_secs_f32() * 1000.0);
    assert!(field.dims.iter().all(|&d| d <= 256));

    // Surface points are at distance ~0 and a step along the normal changes
    // the distance by that step, in the right direction.
    let (mut worst_surface, mut worst_offset, mut wrong_gradient) = (0.0f32, 0.0f32, 0);
    let skin: std::collections::HashSet<u32> = mesh.skin().iter().flatten().copied().collect();
    for &i in skin.iter().step_by(7) {
        let (p, n) = (mesh.positions[i as usize], mesh.normals[i as usize]);
        let (on, _) = field.sample(p).unwrap();
        let (out, gradient) = field.sample(p + n * 0.012).unwrap();
        let (inside, _) = field.sample(p - n * 0.006).unwrap();
        worst_surface = worst_surface.max(on.abs());
        // Fingers and ears are thinner than the probe distance.
        if out > 0.011 {
            worst_offset = worst_offset.max((out - 0.012).abs());
            wrong_gradient += usize::from(gradient.normalize_or_zero().dot(n) < 0.5);
        }
        let _ = inside;
    }
    println!("surface error {:.2} mm, offset error {:.2} mm", worst_surface * 1000.0, worst_offset * 1000.0);
    assert!(worst_surface < 0.004);
    assert!(wrong_gradient < skin.len() / 7 / 50, "{wrong_gradient} gradients point the wrong way");

    // Inside the chest is inside; far in front of it is outside.
    let chest = Vec3::new(0.0, 0.72 * mesh.height(), 0.0);
    assert!(field.sample(chest).unwrap().0 < 0.0);
    assert!(field.sample(chest + Vec3::Z * 0.3).is_none_or(|(d, _)| d > 0.0));
}

#[test]
fn a_whole_measurement_chart_is_matched_at_once() {
    let model = model();
    // Two charts that differ a lot from the default bodies.
    let woman = [
        (Measure::Height, 1.72),
        (Measure::Neck, 0.335),
        (Measure::Shoulders, 0.36),
        (Measure::Bust, 0.98),
        (Measure::Waist, 0.78),
        (Measure::Hips, 1.06),
        (Measure::Inseam, 0.80),
        (Measure::Thigh, 0.61),
        (Measure::Calf, 0.38),
        (Measure::Ankle, 0.225),
        (Measure::ArmLength, 0.55),
        (Measure::UpperArm, 0.30),
        (Measure::Wrist, 0.16),
    ];
    let man = [
        (Measure::Height, 1.86),
        (Measure::Neck, 0.40),
        (Measure::Shoulders, 0.41),
        (Measure::Bust, 1.08),
        (Measure::Waist, 0.92),
        (Measure::Hips, 1.04),
        (Measure::Inseam, 0.87),
        (Measure::Thigh, 0.60),
        (Measure::Calf, 0.40),
        (Measure::Ankle, 0.245),
        (Measure::ArmLength, 0.62),
        (Measure::UpperArm, 0.34),
        (Measure::Wrist, 0.175),
    ];
    for (label, start, chart) in [("woman", BodyParams::female(), woman), ("man", BodyParams::male(), man)] {
        let mut params = start;
        let started = std::time::Instant::now();
        let report = fit_all(&model, &mut params, &chart);
        let result = measure(&model, &model.build(&params));
        println!("{label}: worst error {:.2} mm in {:.0} ms, body fat {:+.2}, scale {:.3}", report.worst_error * 1000.0, started.elapsed().as_secs_f32() * 1000.0, report.weight_change, params.scale);
        for (which, target) in chart {
            println!("  {:<32} {:6.2} cm (target {:6.2})", which.label(), result.get(which) * 100.0, target * 100.0);
            assert!((result.get(which) - target).abs() < 0.001, "{label}: {} is {:.2} cm, wanted {:.2} cm", which.label(), result.get(which) * 100.0, target * 100.0);
        }
        assert!(report.exact() && report.unreachable.is_empty());
    }
}

#[test]
fn closing_covers_small_details_without_moving_the_body() {
    let model = model();
    let mesh = model.build(&BodyParams::female());
    let exact = distance_field(&mesh, 0.008, 0.0);
    let started = std::time::Instant::now();
    let closed = distance_field(&mesh, 0.008, 0.012);
    println!("closed field {:?} built in {:.0} ms", closed.dims, started.elapsed().as_secs_f32() * 1000.0);
    assert_eq!(exact.dims.len(), closed.dims.len());

    // The closed body contains the exact one: no skin vertex is outside it.
    let skin: std::collections::HashSet<u32> = mesh.skin().iter().flatten().copied().collect();
    let (mut outside, mut raised, mut most) = (0, 0, 0.0f32);
    for &i in &skin {
        let p = mesh.positions[i as usize];
        let (Some((before, _)), Some((after, _))) = (exact.sample(p), closed.sample(p)) else { continue };
        outside += usize::from(after - before > 1e-4);
        raised += usize::from(before - after > 0.003);
        most = most.max(before - after);
    }
    println!("{raised} of {} vertices are covered by more than 3 mm (deepest {:.1} mm)", skin.len(), most * 1000.0);
    assert_eq!(outside, 0, "closing must never shrink the body");
    assert!(raised > 50, "creases should have been filled");

    // Open, convex areas are unchanged: the front of the thigh.
    let thigh = mesh.positions.iter().filter(|p| p.x > 0.05 && p.x < 0.13 && (p.y - 0.33 * mesh.height()).abs() < 0.03).max_by(|a, b| a.z.total_cmp(&b.z)).unwrap();
    let probe = *thigh + Vec3::Z * 0.006;
    let (before, after) = (exact.sample(probe).unwrap().0, closed.sample(probe).unwrap().0);
    assert!((before - after).abs() < 0.002, "{before} vs {after}");
}

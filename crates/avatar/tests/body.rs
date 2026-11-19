use avatar::{BodyModel, BodyParams, Measure, distance_field, fit, fit_all, measure};
use glam::{Vec2, Vec3};

fn model() -> BodyModel {
    BodyModel::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/avatar/body.bin"
    ))
    .expect("run tools/bake_avatar.py")
}

fn report(label: &str, m: &avatar::Measurements) {
    println!(
        "{label}: {:.1} kg, levels bust {:.3} waist {:.3} hip {:.3} armpit {:.3}",
        m.weight_kg, m.bust_level, m.waist_level, m.hip_level, m.armpit_level
    );
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

    for (m, height, bust, waist, hips) in [
        (&female, 1.55..1.75, 0.80..1.00, 0.60..0.80, 0.85..1.05),
        (&male, 1.68..1.88, 0.90..1.10, 0.72..0.92, 0.88..1.08),
    ] {
        assert!(height.contains(&m.get(Measure::Height)));
        assert!(bust.contains(&m.get(Measure::Bust)));
        assert!(waist.contains(&m.get(Measure::Waist)));
        assert!(hips.contains(&m.get(Measure::Hips)));
        assert!(
            (0.42..0.50).contains(&(m.get(Measure::Inseam) / m.get(Measure::Height))),
            "legs are about 45-48 % of the height"
        );
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
    assert!(
        male.get(Measure::Waist) / male.get(Measure::Hips)
            > female.get(Measure::Waist) / female.get(Measure::Hips)
    );
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
        assert!(
            (result.get(which) - target).abs() < 0.006,
            "{} is {:.1} cm, wanted {:.1} cm",
            which.label(),
            result.get(which) * 100.0,
            target * 100.0
        );
    }
}

#[test]
fn distance_field_matches_the_body() {
    let model = model();
    let mesh = model.build(&BodyParams::male());
    let started = std::time::Instant::now();
    let field = distance_field(&mesh, 0.008, 0.0);
    println!(
        "field {:?} at {:.1} mm built in {:.0} ms",
        field.dims,
        field.voxel * 1000.0,
        started.elapsed().as_secs_f32() * 1000.0
    );
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
    println!(
        "surface error {:.2} mm, offset error {:.2} mm",
        worst_surface * 1000.0,
        worst_offset * 1000.0
    );
    assert!(worst_surface < 0.004);
    assert!(
        wrong_gradient < skin.len() / 7 / 50,
        "{wrong_gradient} gradients point the wrong way"
    );

    // Inside the chest is inside; far in front of it is outside.
    let chest = Vec3::new(0.0, 0.72 * mesh.height(), 0.0);
    assert!(field.sample(chest).unwrap().0 < 0.0);
    assert!(
        field
            .sample(chest + Vec3::Z * 0.3)
            .is_none_or(|(d, _)| d > 0.0)
    );
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
    for (label, start, chart) in [
        ("woman", BodyParams::female(), woman),
        ("man", BodyParams::male(), man),
    ] {
        let mut params = start;
        let started = std::time::Instant::now();
        let report = fit_all(&model, &mut params, &chart);
        let result = measure(&model, &model.build(&params));
        println!(
            "{label}: worst error {:.2} mm in {:.0} ms, body fat {:+.2}, scale {:.3}",
            report.worst_error * 1000.0,
            started.elapsed().as_secs_f32() * 1000.0,
            report.weight_change,
            params.scale
        );
        for (which, target) in chart {
            println!(
                "  {:<32} {:6.2} cm (target {:6.2})",
                which.label(),
                result.get(which) * 100.0,
                target * 100.0
            );
            assert!(
                (result.get(which) - target).abs() < 0.001,
                "{label}: {} is {:.2} cm, wanted {:.2} cm",
                which.label(),
                result.get(which) * 100.0,
                target * 100.0
            );
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
    println!(
        "closed field {:?} built in {:.0} ms",
        closed.dims,
        started.elapsed().as_secs_f32() * 1000.0
    );
    assert_eq!(exact.dims.len(), closed.dims.len());

    // The closed body contains the exact one: no skin vertex is outside it.
    let skin: std::collections::HashSet<u32> = mesh.skin().iter().flatten().copied().collect();
    let (mut outside, mut raised, mut most) = (0, 0, 0.0f32);
    for &i in &skin {
        let p = mesh.positions[i as usize];
        let (Some((before, _)), Some((after, _))) = (exact.sample(p), closed.sample(p)) else {
            continue;
        };
        outside += usize::from(after - before > 1e-4);
        raised += usize::from(before - after > 0.003);
        most = most.max(before - after);
    }
    println!(
        "{raised} of {} vertices are covered by more than 3 mm (deepest {:.1} mm)",
        skin.len(),
        most * 1000.0
    );
    assert_eq!(outside, 0, "closing must never shrink the body");
    assert!(raised > 50, "creases should have been filled");

    // Open, convex areas are unchanged: the front of the thigh.
    let thigh = mesh
        .positions
        .iter()
        .filter(|p| p.x > 0.05 && p.x < 0.13 && (p.y - 0.33 * mesh.height()).abs() < 0.03)
        .max_by(|a, b| a.z.total_cmp(&b.z))
        .unwrap();
    let probe = *thigh + Vec3::Z * 0.006;
    let (before, after) = (
        exact.sample(probe).unwrap().0,
        closed.sample(probe).unwrap().0,
    );
    assert!((before - after).abs() < 0.002, "{before} vs {after}");
}

#[test]
fn hair_never_covers_the_face_or_ears() {
    use avatar::look::{Appearance, Hair, HairStyle, Head, hair};
    let model = model();
    for params in [
        BodyParams::female(),
        BodyParams::male(),
        BodyParams {
            weight: 1.0,
            ..BodyParams::male()
        },
    ] {
        let body = model.build(&params);
        let head = Head::find(&body).expect("the body has eyes");
        // The head's axis runs through the skull, well behind the eyes.
        assert!(
            head.axis_z < head.eye_line.z - 0.04,
            "axis at {} with eyes at {}",
            head.axis_z,
            head.eye_line.z
        );
        for style in HairStyle::ALL {
            let looks = Appearance {
                hair: Hair {
                    style,
                    length: style.default_length(),
                    fringe: true,
                    ..Hair::default()
                },
                ..Appearance::default()
            };
            let mesh = hair(&body, &head, &looks);
            // Nothing of the hair that shows is in front of the eyes or
            // over the cheeks and mouth.
            for (p, coverage) in mesh.positions.iter().zip(&mesh.coverage) {
                let in_face = p.z > head.eye_line.z - 0.01
                    && p.x.abs() < 0.06
                    && p.y < head.eye_line.y + 0.02
                    && p.y > head.eye_line.y - 0.14;
                assert!(
                    !(in_face && *coverage > 0.2),
                    "{style:?} covers the face at {p}"
                );
            }
        }
    }
}

#[test]
fn a_walk_is_two_steps_and_starts_over_a_stride_ahead() {
    use avatar::{Motion, Performance};
    let model = model();
    let rest = model.build(&BodyParams::male());
    for motion in [Motion::Walk, Motion::StiffWalk, Motion::HipSwayWalk] {
        let duration = motion.duration();
        assert!(duration < 1.5, "{motion:?}: one stride, not a lap ({duration} s)");
        let travel = motion.travel();
        assert!(travel.z > 0.7 && travel.x == 0.0, "{motion:?} walks ahead: {travel}");
        let performance = Performance::bake(motion, &model, &rest);
        let start = performance.pose(0.0, &model, &rest).offset;
        let end = performance.pose(duration - 1e-3, &model, &rest).offset;
        // Where the body is at the end of the loop is where the next loop
        // starts, a stride on.
        assert!(
            (end - (start + travel)).length() < 0.01,
            "{motion:?}: ends at {end}, starts over at {}",
            start + travel
        );
    }
}

#[test]
fn animations_keep_feet_planted_balance_and_move_smoothly() {
    use avatar::motion::Joint;
    use avatar::{Motion, Performance, Side};
    let model = model();
    let rest = model.build(&BodyParams::female());
    let foot = [model.bone("foot.L").unwrap(), model.bone("foot.R").unwrap()];
    let step = 1.0 / 60.0;
    for motion in Motion::ALL {
        let performance = Performance::bake(motion, &model, &rest);
        let frames = (motion.duration() / step) as usize;
        let mut previous: Option<Vec<Vec3>> = None;
        let mut velocity: Option<Vec<Vec3>> = None;
        let mut planted_at: [Option<Vec3>; 2] = [None, None];
        let (mut worst_slide, mut worst_speed, mut worst_accel) = (0.0f32, 0.0f32, 0.0f32);
        let mut worst_at = (0.0, String::new());
        for frame in 0..frames {
            let t = frame as f32 * step;
            let pose = performance.pose(t, &model, &rest);
            let bones = model.bone_transforms(&rest, &pose);
            let points: Vec<Vec3> = bones
                .iter()
                .zip(&rest.joints)
                .map(|(b, j)| b.transform_point3(j.head) + pose.offset)
                .collect();
            let angles = motion.angles(t);
            for (side, s) in [Side::Left, Side::Right].into_iter().enumerate() {
                let ankle = points[foot[side]];
                if angles.get(Joint::Plant(s)) >= 0.99 {
                    let start = *planted_at[side].get_or_insert(ankle);
                    worst_slide =
                        worst_slide.max(Vec2::new(ankle.x - start.x, ankle.z - start.z).length());
                } else {
                    planted_at[side] = None;
                }
            }
            if let Some(prev) = &previous {
                let v: Vec<Vec3> = points
                    .iter()
                    .zip(prev)
                    .map(|(a, b)| (*a - *b) / step)
                    .collect();
                worst_speed = worst_speed.max(v.iter().map(|v| v.length()).fold(0.0, f32::max));
                if let Some(pv) = &velocity {
                    for (bone, (a, b)) in v.iter().zip(pv).enumerate() {
                        let accel = ((*a - *b) / step).length();
                        if accel > worst_accel {
                            worst_accel = accel;
                            worst_at = (t, model.bones()[bone].name.clone());
                        }
                    }
                }
                velocity = Some(v);
            }
            previous = Some(points);
        }
        println!(
            "{motion:?}: slide {:.1} mm, speed {worst_speed:.2} m/s, accel {worst_accel:.1} m/s² at {:.2} s ({})",
            worst_slide * 1000.0,
            worst_at.0,
            worst_at.1
        );
        assert!(
            worst_slide < 0.005,
            "{motion:?}: a planted foot slid {:.1} mm",
            worst_slide * 1000.0
        );
        let walking = matches!(
            motion,
            Motion::Walk | Motion::StiffWalk | Motion::HipSwayWalk
        );
        // The tip of a swinging foot is the fastest thing in a walk.
        assert!(
            worst_speed < if walking { 7.0 } else { 4.0 },
            "{motion:?}: something moved at {worst_speed} m/s"
        );
        // A swinging foot whips through and strikes the ground; elsewhere the
        // most is a foot set down from a step, which real feet do at a few g.
        let limit = if walking { 80.0 } else { 50.0 };
        assert!(
            worst_accel < limit,
            "{motion:?}: a jolt of {worst_accel} m/s²"
        );
    }
}

/// The posed body for arms raised overhead, with both skinnings.
fn arms_overhead(model: &BodyModel) -> (avatar::BodyMesh, avatar::BodyMesh, avatar::BodyMesh) {
    let rest = model.build(&BodyParams::female());
    let overhead = avatar::PoseParams {
        arm_angle: Some(170.0),
        ..avatar::PoseParams::modelled()
    }
    .to_pose(model, &rest);
    let posed = model.pose(&rest, &overhead);
    let linear = model.pose_linear_blend(&rest, &overhead);
    (rest, posed, linear)
}

#[test]
fn raised_arms_keep_the_shoulders_full() {
    let model = model();
    let (rest, posed, linear) = arms_overhead(&model);
    // Skin over the top of each shoulder, against the shoulder joint: a
    // collapsing shoulder brings it down towards the joint.
    let clearance = |body: &avatar::BodyMesh| -> f32 {
        let joints: Vec<Vec3> = ["upperarm01.L", "upperarm01.R"]
            .iter()
            .map(|n| body.joints[model.bone(n).unwrap()].head)
            .collect();
        let top: Vec<usize> = (0..body.positions.len())
            .filter(|&i| {
                let p = rest.positions[i];
                p.x.abs() > 0.08 && p.x.abs() < 0.16 && p.y > 1.3 && p.y < 1.42 && p.z.abs() < 0.05
            })
            .collect();
        top.iter()
            .map(|&i| {
                joints
                    .iter()
                    .map(|j| body.positions[i].distance(*j))
                    .fold(f32::MAX, f32::min)
            })
            .sum::<f32>()
            / top.len() as f32
    };
    let (c0, c1, c2) = (clearance(&rest), clearance(&posed), clearance(&linear));
    eprintln!(
        "skin over the shoulder to the joint: rest {:.1} cm, centres of rotation {:.1} cm, linear blend {:.1} cm",
        c0 * 100.0,
        c1 * 100.0,
        c2 * 100.0
    );
    assert!(c1 > c2, "the shoulder keeps more of its volume");
}

#[test]
fn centres_of_rotation_do_not_tear_the_skin() {
    let model = model();
    let (rest, posed, linear) = arms_overhead(&model);
    let mut worst = 0.0f32;
    for t in rest.skin() {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let r = rest.positions[a as usize].distance(rest.positions[b as usize]);
            let c = posed.positions[a as usize].distance(posed.positions[b as usize]) / r;
            let l = linear.positions[a as usize].distance(linear.positions[b as usize]) / r;
            worst = worst.max(c / l);
        }
    }
    assert!(
        worst < 2.0,
        "an edge is stretched {worst:.1} times as far as linear blending stretches it"
    );
}

/// The buttonhole front of the sample waistcoat is on layer 2, the rest on
/// layer 1. In one sewn garment that only means "lapped on top": it must
/// start exactly as if it were on layer 1 (a whole layer further out, the
/// closure started turned round the body and the vest draped lopsided),
/// with its holes over the buttons.
#[test]
fn a_buttonhole_front_on_a_higher_layer_starts_lapped_not_a_layer_out() {
    let model = model();
    let body = model.build(&BodyParams::female());
    let measurements = measure(&model, &body);
    let mut layered = cad_core::PatternDocument::demo_waistcoat();
    let mut flat = layered.clone();
    let ids: Vec<_> = flat.pieces().iter().map(|p| p.id).collect();
    for id in ids {
        flat.piece_mut(id).unwrap().layer = 1;
    }
    assert!(layered.pieces().iter().any(|p| p.layer == 2));
    avatar::arrange_around(&mut layered, &body, &measurements);
    avatar::arrange_around(&mut flat, &body, &measurements);
    for (a, b) in layered.pieces().iter().zip(flat.pieces()) {
        assert_eq!(a.arrangement, b.arrangement, "{} starts elsewhere", a.name);
    }
    for (button_piece, button, hole_piece, hole) in layered.fastenings() {
        let b = layered.piece(button_piece).unwrap().arrangement.place(button.position);
        let holed = layered.piece(hole_piece).unwrap();
        let h = holed.arrangement.place(hole.button_end(&holed.outline));
        // Same direction round the body, the hole a little further out.
        let turn = (b.x.atan2(b.z) - h.x.atan2(h.z)).abs().to_degrees();
        assert!(turn < 0.5, "hole {turn:.1}° round from its button");
        let radius = |p: Vec3| Vec2::new(p.x, p.z).length();
        assert!(radius(h) > radius(b), "the hole side starts outside the buttons");
    }
}

//! Fabrics to start from: real cloths with their weave, yarn colours,
//! surface and how they hang.
//!
//! Each is an ordinary [`Fabric`], so everything about it stays editable.
//! Woven ones carry their draft ([`Weave`]); corduroy's ribs of pile are no
//! draft but a surface of their own ([`FabricPattern::Corduroy`]).

use crate::ids::FabricId;
use crate::material::{Fabric, FabricPattern, FabricPhysics};
use crate::weave::{Weave, Yarn};

/// How a cloth hangs, by kind.
fn physics(density: f32, bending: f32, thickness: f32, stretch: f32) -> FabricPhysics {
    FabricPhysics {
        density,
        stretch_weft: stretch,
        stretch_warp: stretch * 1.25,
        shear: 0.3,
        bending,
        thickness,
    }
}

/// Every preset, under its name (the fabric's id is a placeholder).
pub fn presets() -> Vec<Fabric> {
    let yarn = |name: &str, color: [f32; 3]| Yarn {
        name: name.into(),
        color,
    };
    let library = Weave::library();
    let draft = |name: &str| {
        library
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, w)| w.clone())
            .expect("a weave of the library")
    };
    // A woven fabric from a draft and its yarns.
    #[allow(clippy::too_many_arguments)]
    let woven = |name: &str,
                 mut weave: Weave,
                 yarns: Vec<Yarn>,
                 warp_order: Vec<u8>,
                 weft_order: Vec<u8>,
                 density: [f32; 2],
                 look: [f32; 5],
                 roughness: f32,
                 hang: FabricPhysics| {
        weave.yarns = yarns;
        weave.warp_order = warp_order;
        weave.weft_order = weft_order;
        [weave.ends_per_cm, weave.picks_per_cm] = density;
        [
            weave.irregularity,
            weave.slubs,
            weave.warp_cover,
            weave.weft_cover,
            weave.fuzz,
        ] = look;
        let mut fabric = Fabric::new(FabricId(0), name, weave.yarns[0].color);
        fabric.weave = Some(weave);
        fabric.roughness = roughness;
        fabric.relief = 1.0;
        fabric.physics = hang;
        fabric
    };
    let denim = |name: &str, warp: [f32; 3], weft: [f32; 3], wear: f32| {
        let mut weave = Weave::denim(warp, weft);
        weave.irregularity = (weave.irregularity + 0.25 * wear).min(1.0);
        weave.slubs = (weave.slubs + 0.3 * wear).min(1.0);
        weave.fuzz = (weave.fuzz + 0.2 * wear).min(1.0);
        let mut fabric = Fabric::new(FabricId(0), name, warp);
        fabric.weave = Some(weave);
        fabric.wear = 0.3 + 0.6 * wear;
        fabric.roughness = 0.88;
        fabric.relief = 1.2;
        // Denim hardly gives along the warp (down the leg) and a little
        // across, even with some elastane; its twill resists shear. (At
        // 0.2 it stretched like a knit: jeans came out wavy and limp.)
        fabric.physics = FabricPhysics {
            density: 420.0,
            stretch_weft: 0.62,
            stretch_warp: 0.92,
            shear: 0.5,
            bending: 0.58,
            thickness: 0.9,
        };
        fabric
    };
    let corduroy = |name: &str, ribs_scale: f32, relief: f32, color: [f32; 3]| {
        let mut fabric = Fabric::new(FabricId(0), name, color);
        fabric.pattern = FabricPattern::Corduroy;
        fabric.pattern_scale = ribs_scale;
        fabric.relief = relief;
        fabric.weave_strength = 1.0;
        fabric.roughness = 0.9;
        fabric.physics = physics(360.0, 0.42, 1.0 + 0.4 * relief, 0.28);
        fabric
    };
    let white = [0.74, 0.73, 0.69];
    let shirting = physics(115.0, 0.12, 0.3, 0.3);

    vec![
        denim("Raw denim", [0.016, 0.032, 0.115], [0.60, 0.58, 0.51], 0.0),
        denim(
            "Washed denim",
            [0.045, 0.095, 0.27],
            [0.66, 0.65, 0.60],
            0.6,
        ),
        denim(
            "Light wash denim",
            [0.13, 0.24, 0.48],
            [0.72, 0.72, 0.68],
            1.0,
        ),
        denim(
            "Black denim",
            [0.014, 0.014, 0.018],
            [0.20, 0.20, 0.20],
            0.3,
        ),
        woven(
            "Chambray",
            draft("Plain"),
            vec![yarn("Blue", [0.07, 0.17, 0.42]), yarn("White", white)],
            vec![0],
            vec![1],
            [34.0, 30.0],
            [0.6, 0.35, 0.95, 0.9, 0.4],
            0.8,
            shirting,
        ),
        woven(
            "Oxford shirting",
            draft("2/2 basket"),
            vec![yarn("Sky", [0.30, 0.47, 0.78]), yarn("White", white)],
            vec![0],
            vec![1],
            [36.0, 22.0],
            [0.4, 0.2, 0.92, 1.0, 0.3],
            0.8,
            physics(140.0, 0.16, 0.35, 0.3),
        ),
        woven(
            "Poplin",
            draft("Plain"),
            vec![yarn("White", [0.80, 0.80, 0.78])],
            vec![0],
            vec![0],
            [50.0, 28.0],
            [0.15, 0.0, 1.0, 0.85, 0.1],
            0.7,
            physics(100.0, 0.1, 0.25, 0.3),
        ),
        woven(
            "Chino twill",
            draft("2/1 twill"),
            vec![
                yarn("Khaki", [0.40, 0.31, 0.18]),
                yarn("Shade", [0.33, 0.25, 0.14]),
            ],
            vec![0],
            vec![1],
            [40.0, 24.0],
            [0.3, 0.1, 1.05, 0.85, 0.25],
            0.82,
            physics(260.0, 0.38, 0.55, 0.25),
        ),
        woven(
            "Gingham shirting",
            draft("Gingham"),
            vec![yarn("Red", [0.55, 0.05, 0.05]), yarn("White", white)],
            [vec![0u8; 8], vec![1u8; 8]].concat(),
            [vec![0u8; 8], vec![1u8; 8]].concat(),
            [32.0, 30.0],
            [0.3, 0.1, 0.92, 0.92, 0.25],
            0.8,
            shirting,
        ),
        woven(
            "Flannel check",
            draft("2/2 twill"),
            vec![
                yarn("Red", [0.40, 0.03, 0.03]),
                yarn("Black", [0.02, 0.02, 0.025]),
            ],
            [vec![0u8; 24], vec![1u8; 24]].concat(),
            [vec![0u8; 24], vec![1u8; 24]].concat(),
            [18.0, 17.0],
            [0.8, 0.4, 1.0, 1.0, 0.95],
            0.95,
            physics(200.0, 0.25, 0.8, 0.3),
        ),
        woven(
            "Herringbone tweed",
            draft("Herringbone"),
            vec![
                yarn("Grey", [0.20, 0.19, 0.17]),
                yarn("Charcoal", [0.05, 0.05, 0.05]),
            ],
            vec![0, 1],
            vec![1, 0],
            [12.0, 12.0],
            [1.0, 0.7, 1.05, 1.05, 0.85],
            0.95,
            physics(380.0, 0.55, 1.3, 0.2),
        ),
        woven(
            "Silk satin",
            draft("8-end satin"),
            vec![yarn("Champagne", [0.62, 0.50, 0.34])],
            vec![0],
            vec![0],
            [90.0, 45.0],
            [0.1, 0.0, 1.1, 0.8, 0.0],
            0.3,
            physics(90.0, 0.04, 0.2, 0.35),
        ),
        corduroy("Corduroy", 1.6, 1.0, [0.30, 0.11, 0.04]),
        corduroy("Wide wale corduroy", 0.85, 1.4, [0.16, 0.12, 0.06]),
        corduroy("Needlecord", 3.2, 0.7, [0.07, 0.10, 0.06]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_woven_preset_has_its_yarns_and_a_texture() {
        let presets = presets();
        assert!(presets.len() >= 10);
        for fabric in &presets {
            if let Some(weave) = &fabric.weave {
                for &i in weave.warp_order.iter().chain(&weave.weft_order) {
                    assert!((i as usize) < weave.yarns.len(), "{}", fabric.name);
                }
                let t = weave.texture(8);
                assert_eq!(
                    t.albedo.len(),
                    (t.width * t.height) as usize,
                    "{}",
                    fabric.name
                );
            }
        }
    }

    #[test]
    fn denim_shows_little_of_its_white_weft() {
        let fabric = presets()
            .into_iter()
            .find(|f| f.name == "Raw denim")
            .unwrap();
        let t = fabric.weave.unwrap().texture(8);
        let weft = t.albedo.iter().filter(|p| p[3] == 0).count();
        assert!(
            (weft as f32) < 0.2 * t.albedo.len() as f32,
            "weft on top in {:.0} % of the face",
            100.0 * weft as f32 / t.albedo.len() as f32
        );
    }
}

//! Fabric definitions: how a material looks (PBR) and how it behaves (physics).

use serde::{Deserialize, Serialize};

use crate::ids::FabricId;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FabricPattern {
    Plain,
    Denim,
    Leather,
    Corduroy,
    Wool,
    Knit,
    Satin,
}

impl Default for FabricPattern {
    fn default() -> Self {
        Self::Plain
    }
}

impl FabricPattern {
    pub const ALL: [Self; 7] = [
        Self::Plain,
        Self::Denim,
        Self::Leather,
        Self::Corduroy,
        Self::Wool,
        Self::Knit,
        Self::Satin,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Plain => "Plain weave",
            Self::Denim => "Denim",
            Self::Leather => "Leather",
            Self::Corduroy => "Corduroy",
            Self::Wool => "Wool",
            Self::Knit => "Knit",
            Self::Satin => "Satin",
        }
    }
}

/// Mechanical properties consumed by a cloth solver.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FabricPhysics {
    /// Areal density in g/m².
    pub density: f32,
    /// Stretch stiffness along the weft, 0..=1 (normalised).
    pub stretch_weft: f32,
    /// Stretch stiffness along the warp, 0..=1.
    pub stretch_warp: f32,
    /// Shear stiffness, 0..=1.
    pub shear: f32,
    /// Bending stiffness, 0..=1.
    pub bending: f32,
    /// Thickness in mm.
    pub thickness: f32,
}

impl Default for FabricPhysics {
    fn default() -> Self {
        // Roughly a medium-weight cotton jersey.
        Self {
            density: 180.0,
            stretch_weft: 0.35,
            stretch_warp: 0.45,
            shear: 0.3,
            bending: 0.2,
            thickness: 0.6,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fabric {
    pub id: FabricId,
    pub name: String,
    /// Linear RGB base colour.
    pub base_color: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    /// Strength of the procedural weave pattern in the viewport, 0..=1.
    pub weave_strength: f32,
    #[serde(default)]
    pub pattern: FabricPattern,
    /// Degree of fading and irregular colour variation, 0..=1.
    #[serde(default)]
    pub wash: f32,
    /// Repeat scale of the procedural yarn/rib pattern.
    #[serde(default = "default_pattern_scale")]
    pub pattern_scale: f32,
    /// Height of raised procedural surface detail, relative to the pattern.
    #[serde(default = "default_relief")]
    pub relief: f32,
    pub physics: FabricPhysics,
    /// Woven from this draft: its look comes from the weave (see
    /// [`crate::weave`]) instead of the colour and the preset pattern.
    #[serde(default)]
    pub weave: Option<crate::weave::Weave>,
    /// How worn the garment is, 0..=1: fading along seams and hems and on
    /// the ridges of folds, darker creases, the odd pill of fuzz.
    #[serde(default = "default_wear")]
    pub wear: f32,
    /// Its look from image maps (scanned or downloaded) instead of a
    /// generated weave; takes precedence over [`Self::weave`] for the look.
    #[serde(default)]
    pub maps: Option<FabricMaps>,
}

/// A fabric's look as image maps, one repeat of the cloth each: colour,
/// and optionally a normal map, roughness and height. Paths are absolute.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FabricMaps {
    pub color: std::path::PathBuf,
    #[serde(default)]
    pub normal: Option<std::path::PathBuf>,
    /// OpenGL-style normal map (green up), as most libraries ship; DirectX
    /// maps have green down.
    #[serde(default = "default_true")]
    pub normal_y_up: bool,
    #[serde(default)]
    pub roughness: Option<std::path::PathBuf>,
    #[serde(default)]
    pub height: Option<std::path::PathBuf>,
    /// How much cloth one repeat of the images shows, millimetres (width,
    /// height).
    pub size_mm: [f32; 2],
    /// Where the maps came from (a U3M file or a folder), for the user.
    #[serde(default)]
    pub source: Option<std::path::PathBuf>,
    /// Take every patch of the cloth from a different place of the images,
    /// so no repeat shows. Right for plain cloth; off for checks, stripes
    /// and prints, which must stay in register.
    #[serde(default = "default_true")]
    pub break_up_repeats: bool,
}

fn default_true() -> bool {
    true
}

impl FabricMaps {
    /// A number that changes whenever the maps change (which files, how
    /// read); loading them again after editing the files needs a new pick.
    pub fn key(&self) -> u64 {
        let mut h: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut eat = |bytes: &[u8]| {
            for &b in bytes {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        };
        for path in [
            Some(&self.color),
            self.normal.as_ref(),
            self.roughness.as_ref(),
            self.height.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            eat(path.to_string_lossy().as_bytes());
            eat(&[0]);
        }
        eat(&[u8::from(self.normal_y_up)]);
        h | 1
    }
}

fn default_wear() -> f32 {
    0.25
}

fn default_pattern_scale() -> f32 {
    1.0
}

fn default_relief() -> f32 {
    1.0
}

impl Fabric {
    pub fn new(id: FabricId, name: impl Into<String>, base_color: [f32; 3]) -> Self {
        Self {
            id,
            name: name.into(),
            base_color,
            roughness: 0.85,
            metallic: 0.0,
            weave_strength: 0.35,
            pattern: FabricPattern::Plain,
            wash: 0.0,
            pattern_scale: default_pattern_scale(),
            relief: default_relief(),
            physics: FabricPhysics::default(),
            weave: None,
            wear: default_wear(),
            maps: None,
        }
    }
}

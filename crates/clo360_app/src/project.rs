//! Project files (`.clo360`): the pattern and the avatar, as JSON.
//!
//! Saved: every piece with its outline, seam allowance, notches, grain,
//! fabric, layer and placement on the avatar; the seams; the fabrics; the
//! avatar's build, tape measurements and pose; the mesh resolution. Not
//! saved: the draped shape. After opening, the garment starts from its
//! arrangement and is simulated again.

use std::path::Path;

use anyhow::Context as _;
use avatar::{Appearance, BodyParams, Hair, HairStyle, Measure, PoseParams};
use cad_core::PatternDocument;
use serde::{Deserialize, Serialize};

const FORMAT: &str = "clo360-project";
const VERSION: u32 = 1;
pub const EXTENSION: &str = "clo360";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Project {
    format: String,
    version: u32,
    pub document: PatternDocument,
    pub avatar: Option<AvatarFile>,
    /// Mesh resolution in mm.
    pub particle_distance: f32,
}

/// The avatar's build, measurements and pose.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AvatarFile {
    gender: f32,
    weight: f32,
    muscle: f32,
    stature: f32,
    cup: f32,
    scale: f32,
    morphs: Vec<f32>,
    /// Tape measurements the body is fitted to, in metres, by measure name.
    targets: Vec<(String, f32)>,
    arm_angle: Option<f32>,
    arm_forward: f32,
    elbow_bend: f32,
    leg_angle: Option<f32>,
    knee_bend: f32,
    /// Skin, eyes and hair. Missing in files from before they existed.
    #[serde(default)]
    looks: Option<LooksFile>,
}

/// [`Appearance`] as saved. Colours are linear RGB.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct LooksFile {
    skin: [f32; 3],
    eyes: [f32; 3],
    /// A [`HairStyle`] by its `Debug` name.
    hair_style: String,
    hair_color: [f32; 3],
    hair_length: f32,
    hair_volume: f32,
    fringe: bool,
}

impl AvatarFile {
    pub fn new(params: &BodyParams, pose: &PoseParams, targets: &[Option<f32>], looks: &Appearance) -> Self {
        Self {
            gender: params.gender,
            weight: params.weight,
            muscle: params.muscle,
            stature: params.stature,
            cup: params.cup,
            scale: params.scale,
            morphs: params.morphs.to_vec(),
            targets: Measure::ALL
                .iter()
                .zip(targets)
                .filter_map(|(measure, target)| Some((format!("{measure:?}"), (*target)?)))
                .collect(),
            arm_angle: pose.arm_angle,
            arm_forward: pose.arm_forward,
            elbow_bend: pose.elbow_bend,
            leg_angle: pose.leg_angle,
            knee_bend: pose.knee_bend,
            looks: Some(LooksFile {
                skin: looks.skin,
                eyes: looks.eyes,
                hair_style: format!("{:?}", looks.hair.style),
                hair_color: looks.hair.color,
                hair_length: looks.hair.length,
                hair_volume: looks.hair.volume,
                fringe: looks.hair.fringe,
            }),
        }
    }

    /// The saved looks; defaults for what is missing or unknown.
    pub fn appearance(&self) -> Appearance {
        let Some(looks) = &self.looks else { return Appearance::default() };
        let style = HairStyle::ALL.into_iter().find(|s| format!("{s:?}") == looks.hair_style).unwrap_or_default();
        Appearance {
            skin: looks.skin,
            eyes: looks.eyes,
            hair: Hair { style, color: looks.hair_color, length: looks.hair_length, volume: looks.hair_volume, fringe: looks.fringe },
        }
    }

    pub fn params(&self) -> BodyParams {
        let mut params = BodyParams::female();
        params.gender = self.gender;
        params.weight = self.weight;
        params.muscle = self.muscle;
        params.stature = self.stature;
        params.cup = self.cup;
        params.scale = self.scale;
        // Files from versions with fewer morphs keep the defaults for the rest.
        for (slot, value) in params.morphs.iter_mut().zip(&self.morphs) {
            *slot = *value;
        }
        params
    }

    pub fn pose(&self) -> PoseParams {
        PoseParams {
            arm_angle: self.arm_angle,
            arm_forward: self.arm_forward,
            elbow_bend: self.elbow_bend,
            leg_angle: self.leg_angle,
            knee_bend: self.knee_bend,
        }
    }

    /// Targets by `Measure as usize`; unknown names are skipped.
    pub fn targets(&self) -> [Option<f32>; Measure::ALL.len()] {
        let mut targets = [None; Measure::ALL.len()];
        for (name, value) in &self.targets {
            if let Some(index) = Measure::ALL.iter().position(|m| format!("{m:?}") == *name) {
                targets[index] = Some(*value);
            }
        }
        targets
    }
}

impl Project {
    pub fn new(document: PatternDocument, avatar: Option<AvatarFile>, particle_distance: f32) -> Self {
        Self { format: FORMAT.to_owned(), version: VERSION, document, avatar, particle_distance }
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        // Write next to the target and rename: a crash mid-write never
        // leaves a half-written project behind.
        let temporary = path.with_extension(format!("{EXTENSION}.tmp"));
        std::fs::write(&temporary, json).with_context(|| format!("cannot write {}", temporary.display()))?;
        std::fs::rename(&temporary, path).with_context(|| format!("cannot write {}", path.display()))?;
        Ok(())
    }

    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
        let project: Self = serde_json::from_str(&text).context("not a clo360 project file")?;
        anyhow::ensure!(project.format == FORMAT, "not a clo360 project file");
        anyhow::ensure!(project.version <= VERSION, "made by a newer clo360 (format version {})", project.version);
        Ok(project)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_project_round_trips_through_a_file() {
        let mut document = PatternDocument::demo_top();
        let front = document.pieces()[0].id;
        document.piece_mut(front).unwrap().placement.angle = 30.0;
        let mut params = BodyParams::male();
        params.weight = 0.7;
        let pose = PoseParams { elbow_bend: 25.0, ..PoseParams::default() };
        let mut targets = [None; Measure::ALL.len()];
        targets[0] = Some(1.81);
        let looks = Appearance {
            skin: [0.3, 0.2, 0.15],
            hair: Hair { style: HairStyle::Ponytail, length: 0.4, fringe: true, ..Hair::default() },
            ..Appearance::default()
        };
        let project = Project::new(document, Some(AvatarFile::new(&params, &pose, &targets, &looks)), 10.0);

        let path = std::env::temp_dir().join(format!("clo360_project_test_{}.{EXTENSION}", std::process::id()));
        project.save(&path).unwrap();
        let loaded = Project::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(loaded, project);
        assert_eq!(loaded.document.seams(), project.document.seams());
        let avatar = loaded.avatar.unwrap();
        assert_eq!(avatar.params(), params);
        assert_eq!(avatar.pose(), pose);
        assert_eq!(avatar.targets(), targets);
        assert_eq!(avatar.appearance(), looks);
    }

    /// The sample project in `projects/`: the waistcoat with buttons, worn
    /// by the default avatar with long hair. Rewrite it after changing the
    /// sample with `CLO360_BLESS=1 cargo test -p clo360_app sample`.
    #[test]
    fn the_sample_project_file_is_current() {
        let looks = Appearance { hair: Hair { style: HairStyle::Long, length: 0.45, ..Hair::default() }, ..Appearance::default() };
        let avatar = AvatarFile::new(&BodyParams::female(), &PoseParams::default(), &[None; Measure::ALL.len()], &looks);
        let expected = Project::new(PatternDocument::demo_waistcoat(), Some(avatar), 20.0);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../projects/waistcoat_with_buttons.clo360");
        if std::env::var_os("CLO360_BLESS").is_some() {
            expected.save(&path).unwrap();
        }
        let shipped = Project::load(&path).expect("run with CLO360_BLESS=1 to write the sample project");
        assert_eq!(shipped, expected, "the sample project is out of date: run with CLO360_BLESS=1");
        let fastened = shipped.document.fastenings().count();
        assert_eq!(fastened, 5);
    }

    #[test]
    fn other_files_are_refused() {
        let path = std::env::temp_dir().join(format!("clo360_not_a_project_{}.json", std::process::id()));
        std::fs::write(&path, "{\"hello\": 1}").unwrap();
        assert!(Project::load(&path).is_err());
        std::fs::remove_file(&path).unwrap();
    }
}

//! The morphable body model.

use std::collections::HashMap;
use std::path::Path;

use glam::Vec3;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("cannot read the body file: {0}")]
    Io(#[from] std::io::Error),
    #[error("the body file is damaged or has an unknown format")]
    Format,
}

/// Local shape adjustments, each backed by a decrease/increase pair of
/// morph targets. Values run from -1 to 1 (0 leaves the body unchanged).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Morph {
    NeckGirth,
    NeckHeight,
    ShoulderWidth,
    FrontChestWidth,
    BustGirth,
    UnderbustGirth,
    WaistGirth,
    NapeToWaist,
    WaistToHip,
    HipGirth,
    ThighGirth,
    KneeGirth,
    CalfGirth,
    AnkleGirth,
    UpperLegLength,
    LowerLegLength,
    UpperArmGirth,
    WristGirth,
    UpperArmLength,
    LowerArmLength,
}

impl Morph {
    pub const ALL: [Morph; 20] = [
        Morph::NeckGirth,
        Morph::NeckHeight,
        Morph::ShoulderWidth,
        Morph::FrontChestWidth,
        Morph::BustGirth,
        Morph::UnderbustGirth,
        Morph::WaistGirth,
        Morph::NapeToWaist,
        Morph::WaistToHip,
        Morph::HipGirth,
        Morph::ThighGirth,
        Morph::KneeGirth,
        Morph::CalfGirth,
        Morph::AnkleGirth,
        Morph::UpperLegLength,
        Morph::LowerLegLength,
        Morph::UpperArmGirth,
        Morph::WristGirth,
        Morph::UpperArmLength,
        Morph::LowerArmLength,
    ];

    /// Name stem of the target pair in the body file.
    fn target(self) -> &'static str {
        match self {
            Morph::NeckGirth => "neck-circ",
            Morph::NeckHeight => "neck-height",
            Morph::ShoulderWidth => "shoulder-dist",
            Morph::FrontChestWidth => "frontchest-dist",
            Morph::BustGirth => "bust-circ",
            Morph::UnderbustGirth => "underbust-circ",
            Morph::WaistGirth => "waist-circ",
            Morph::NapeToWaist => "napetowaist-dist",
            Morph::WaistToHip => "waisttohip-dist",
            Morph::HipGirth => "hips-circ",
            Morph::ThighGirth => "thigh-circ",
            Morph::KneeGirth => "knee-circ",
            Morph::CalfGirth => "calf-circ",
            Morph::AnkleGirth => "ankle-circ",
            Morph::UpperLegLength => "upperleg-height",
            Morph::LowerLegLength => "lowerleg-height",
            Morph::UpperArmGirth => "upperarm-circ",
            Morph::WristGirth => "wrist-circ",
            Morph::UpperArmLength => "upperarm-length",
            Morph::LowerArmLength => "lowerarm-length",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Morph::NeckGirth => "Neck girth",
            Morph::NeckHeight => "Neck length",
            Morph::ShoulderWidth => "Shoulder width",
            Morph::FrontChestWidth => "Front chest width",
            Morph::BustGirth => "Bust / chest girth",
            Morph::UnderbustGirth => "Underbust girth",
            Morph::WaistGirth => "Waist girth",
            Morph::NapeToWaist => "Nape to waist",
            Morph::WaistToHip => "Waist to hip",
            Morph::HipGirth => "Hip girth",
            Morph::ThighGirth => "Thigh girth",
            Morph::KneeGirth => "Knee girth",
            Morph::CalfGirth => "Calf girth",
            Morph::AnkleGirth => "Ankle girth",
            Morph::UpperLegLength => "Upper leg length",
            Morph::LowerLegLength => "Lower leg length",
            Morph::UpperArmGirth => "Upper arm girth",
            Morph::WristGirth => "Wrist girth",
            Morph::UpperArmLength => "Upper arm length",
            Morph::LowerArmLength => "Forearm length",
        }
    }
}

/// Everything that defines one body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyParams {
    /// 0 = female, 1 = male; values in between blend the two.
    pub gender: f32,
    /// 0 = slender, 0.5 = average, 1 = heavy.
    pub weight: f32,
    /// 0 = untrained, 0.5 = average, 1 = muscular.
    pub muscle: f32,
    /// -1 = short build, 0 = average, 1 = tall build (changes proportions,
    /// not just size).
    pub stature: f32,
    /// -1 = small, 0 = average, 1 = large. Only affects the female share.
    pub cup: f32,
    /// Uniform scale applied last; used to reach heights beyond `stature`.
    pub scale: f32,
    /// Indexed by `Morph as usize`.
    pub morphs: [f32; Morph::ALL.len()],
}

impl BodyParams {
    /// Morph values may overshoot the sculpted range a little.
    pub const MORPH_LIMIT: f32 = 2.0;

    pub fn female() -> Self {
        Self { gender: 0.0, weight: 0.5, muscle: 0.5, stature: 0.0, cup: 0.0, scale: 1.0, morphs: [0.0; Morph::ALL.len()] }
    }

    pub fn male() -> Self {
        Self { gender: 1.0, ..Self::female() }
    }

    pub fn morph(&self, morph: Morph) -> f32 {
        self.morphs[morph as usize]
    }

    pub fn set_morph(&mut self, morph: Morph, value: f32) {
        self.morphs[morph as usize] = value.clamp(-Self::MORPH_LIMIT, Self::MORPH_LIMIT);
    }
}

impl Default for BodyParams {
    fn default() -> Self {
        Self::female()
    }
}

/// A bone's two end points in world space.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Joint {
    pub head: Vec3,
    pub tail: Vec3,
}

/// One bone of the skeleton. Its joints are defined by marker vertices of
/// the base mesh, so they follow every morph.
#[derive(Debug, Clone)]
pub struct Bone {
    pub name: String,
    pub parent: Option<usize>,
    head: Vec<u32>,
    tail: Vec<u32>,
}

/// A world-space body: metres, Y up, standing on y = 0, facing +Z.
#[derive(Debug, Clone, Default)]
pub struct BodyMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
    /// The first `skin_triangles` triangles are the closed skin surface;
    /// the rest (eyes) are for display only.
    pub skin_triangles: usize,
    /// Vertices on top of the left and right shoulder joints.
    pub shoulder_points: [u32; 2],
    /// One entry per bone of the model's skeleton, in the same order.
    pub joints: Vec<Joint>,
    /// Centres of rotation for skinning, worked out on first posing.
    pub(crate) centres: std::sync::OnceLock<std::sync::Arc<Vec<Vec3>>>,
}

impl BodyMesh {
    /// A body from plain geometry (an imported avatar): no skeleton, and
    /// every triangle counts as skin.
    pub fn from_triangles(positions: Vec<Vec3>, normals: Vec<Vec3>, triangles: Vec<[u32; 3]>) -> Self {
        Self { skin_triangles: triangles.len(), positions, normals, triangles, shoulder_points: [0; 2], joints: Vec::new(), centres: Default::default() }
    }

    /// The skin behind a BVH, for ray casts and continuous collision.
    pub fn collider(&self) -> physics_collision::TriangleMesh {
        physics_collision::TriangleMesh::new(self.positions.iter().map(|&p| p.into()).collect(), self.skin().to_vec())
    }

    pub fn skin(&self) -> &[[u32; 3]] {
        &self.triangles[..self.skin_triangles]
    }

    pub fn height(&self) -> f32 {
        self.positions.iter().map(|p| p.y).fold(0.0, f32::max)
    }

    /// Height of the body's upper surface (shoulders, not the head) above a
    /// point `x` to the side of the centre line, or `None` if the body does
    /// not reach that far out. Used to start garments above the shoulders.
    pub fn shoulder_surface(&self, x: f32) -> Option<f32> {
        let limit = 0.9 * self.height();
        let top = self
            .positions
            .iter()
            .filter(|p| (p.x - x).abs() < 0.012 && p.y < limit)
            .map(|p| p.y)
            .fold(f32::MIN, f32::max);
        (top > f32::MIN).then_some(top)
    }

    /// Largest distance of the torso surface in front of (`+1`) or behind
    /// (`-1`) the z = 0 plane, between hip and shoulder height.
    pub fn torso_depth(&self, side: f32) -> f32 {
        let height = self.height();
        self.positions
            .iter()
            .filter(|p| p.x.abs() < 0.2 && p.y > 0.45 * height && p.y < 0.85 * height)
            .map(|p| p.z * side)
            .fold(0.0, f32::max)
    }
}

type Target = Vec<(u32, Vec3)>;

/// The base mesh and its morph targets.
#[derive(Debug, Clone)]
pub struct BodyModel {
    /// Mesh vertices first, then the joint marker vertices.
    rest: Vec<Vec3>,
    mesh_vertices: usize,
    pub(crate) bones: Vec<Bone>,
    /// Per mesh vertex: up to four bones and their weights.
    pub(crate) weights: Vec<([u16; 4], [f32; 4])>,
    bone_names: HashMap<String, usize>,
    triangles: Vec<[u32; 3]>,
    skin_triangles: usize,
    targets: HashMap<String, Target>,
    shoulder_points: [u32; 2],
    /// Triangles grouped by skin weights, for the centres of rotation.
    pub(crate) bins: std::sync::Arc<crate::skinning::Bins>,
}

/// Weights of the low/average/high targets for a 0..=1 slider.
fn three_way(value: f32) -> [f32; 3] {
    let v = value.clamp(0.0, 1.0);
    [(1.0 - 2.0 * v).max(0.0), 1.0 - (2.0 * v - 1.0).abs(), (2.0 * v - 1.0).max(0.0)]
}

impl BodyModel {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let mut reader = Reader { bytes, at: 0 };
        if reader.take(8)? != b"C3BODY2\0" {
            return Err(LoadError::Format);
        }
        let vertex_count = reader.u32()? as usize;
        let mesh_vertices = reader.u32()? as usize;
        if mesh_vertices > vertex_count {
            return Err(LoadError::Format);
        }
        let rest = (0..vertex_count).map(|_| reader.vec3()).collect::<Result<Vec<_>, _>>()?;
        let triangle_count = reader.u32()? as usize;
        let skin_triangles = reader.u32()? as usize;
        let mut triangles = Vec::with_capacity(triangle_count);
        for _ in 0..triangle_count {
            let triangle = [reader.u32()?, reader.u32()?, reader.u32()?];
            if triangle.iter().any(|&i| i as usize >= mesh_vertices) {
                return Err(LoadError::Format);
            }
            triangles.push(triangle);
        }
        if skin_triangles > triangle_count {
            return Err(LoadError::Format);
        }
        let mut targets = HashMap::new();
        for _ in 0..reader.u32()? {
            let name_length = reader.u16()? as usize;
            let name = String::from_utf8(reader.take(name_length)?.to_vec()).map_err(|_| LoadError::Format)?;
            let entries = reader.u32()? as usize;
            let mut target = Vec::with_capacity(entries);
            for _ in 0..entries {
                let index = reader.u32()?;
                if index as usize >= vertex_count {
                    return Err(LoadError::Format);
                }
                target.push((index, reader.vec3()?));
            }
            targets.insert(name, target);
        }

        let mut bones: Vec<Bone> = Vec::new();
        for index in 0..reader.u32()? as usize {
            let name_length = reader.u16()? as usize;
            let name = String::from_utf8(reader.take(name_length)?.to_vec()).map_err(|_| LoadError::Format)?;
            let parent = reader.i32()?;
            // Parents come first, which the skinning loop relies on.
            let parent = if parent < 0 { None } else { Some(parent as usize).filter(|&p| p < index) };
            let mut markers = || -> Result<Vec<u32>, LoadError> {
                let count = reader.u32()? as usize;
                let list = (0..count).map(|_| reader.u32()).collect::<Result<Vec<_>, _>>()?;
                if list.is_empty() || list.iter().any(|&i| i as usize >= vertex_count) {
                    return Err(LoadError::Format);
                }
                Ok(list)
            };
            let (head, tail) = (markers()?, markers()?);
            bones.push(Bone { name, parent, head, tail });
        }
        let mut weights = Vec::with_capacity(mesh_vertices);
        for _ in 0..mesh_vertices {
            let influence = [reader.u16()?, reader.u16()?, reader.u16()?, reader.u16()?];
            let weight = [reader.f32()?, reader.f32()?, reader.f32()?, reader.f32()?];
            if influence.iter().any(|&b| b as usize >= bones.len().max(1)) {
                return Err(LoadError::Format);
            }
            weights.push((influence, weight));
        }
        let bone_names = bones.iter().enumerate().map(|(i, b)| (b.name.clone(), i)).collect();
        let bins = std::sync::Arc::new(crate::skinning::Bins::new(&weights, &triangles[..skin_triangles]));

        let mut model = Self {
            rest,
            mesh_vertices,
            bones,
            weights,
            bone_names,
            triangles,
            skin_triangles,
            targets,
            shoulder_points: [0, 0],
            bins,
        };
        let neutral = model.build(&BodyParams { gender: 0.5, ..BodyParams::female() });
        model.shoulder_points = crate::measure::find_shoulder_points(&model, &neutral);
        Ok(model)
    }

    /// Every target with a non-zero weight for `params`.
    fn weights(&self, params: &BodyParams) -> Vec<(&Target, f32)> {
        const MUSCLE: [&str; 3] = ["minmuscle", "averagemuscle", "maxmuscle"];
        const WEIGHT: [&str; 3] = ["minweight", "averageweight", "maxweight"];
        let gender = params.gender.clamp(0.0, 1.0);
        let genders = [("female", 1.0 - gender), ("male", gender)];
        let (muscle, weight) = (three_way(params.muscle), three_way(params.weight));
        let stature = params.stature.clamp(-1.0, 1.0);
        let cup = params.cup.clamp(-1.0, 1.0);

        let mut out = Vec::new();
        let mut add = |name: String, weight: f32| {
            if weight > 1e-4
                && let Some(target) = self.targets.get(&name)
            {
                out.push((target, weight));
            }
        };
        for (gender, g) in genders {
            // Facial and skeletal traits, averaged over the sculpted sets.
            for race in ["african", "asian", "caucasian"] {
                add(format!("{race}-{gender}"), g / 3.0);
            }
            for (m, muscle_name) in MUSCLE.iter().enumerate() {
                for (w, weight_name) in WEIGHT.iter().enumerate() {
                    let share = g * muscle[m] * weight[w];
                    let combo = format!("{gender}-{muscle_name}-{weight_name}");
                    add(format!("universal-{combo}"), share);
                    add(format!("minheight-{combo}"), share * (-stature).max(0.0));
                    add(format!("maxheight-{combo}"), share * stature.max(0.0));
                    add(format!("mincup-{combo}"), share * (-cup).max(0.0));
                    add(format!("maxcup-{combo}"), share * cup.max(0.0));
                }
            }
        }
        for morph in Morph::ALL {
            let value = params.morph(morph);
            add(format!("{}-decr", morph.target()), (-value).max(0.0));
            add(format!("{}-incr", morph.target()), value.max(0.0));
        }
        out
    }

    /// Per mesh vertex: whether it moves mostly with the arms (upper arm,
    /// forearm, hand), rather than with the torso or the legs.
    pub fn arm_vertices(&self) -> Vec<bool> {
        let arm: Vec<bool> = self
            .bones
            .iter()
            .map(|bone| ["upperarm", "lowerarm", "wrist", "finger", "metacarpal", "thumb"].iter().any(|stem| bone.name.starts_with(stem)))
            .collect();
        self.weights
            .iter()
            .map(|(bones, weights)| {
                let share: f32 = bones.iter().zip(weights).filter(|(b, _)| arm.get(**b as usize).copied().unwrap_or(false)).map(|(_, w)| w).sum();
                share > 0.5
            })
            .collect()
    }

    pub fn bones(&self) -> &[Bone] {
        &self.bones
    }

    pub fn bone(&self, name: &str) -> Option<usize> {
        self.bone_names.get(name).copied()
    }

    /// Blends the targets for `params` and places the result in the world,
    /// in the pose the body was modelled in.
    pub fn build(&self, params: &BodyParams) -> BodyMesh {
        let mut positions = self.rest.clone();
        for (target, weight) in self.weights(params) {
            for &(index, offset) in target {
                positions[index as usize] += offset * weight;
            }
        }

        // Stand on the floor, centre the torso on the z = 0 plane (garment
        // pieces are arranged in front of and behind it) and scale.
        let skin = &positions[..self.mesh_vertices];
        let (low, high) = skin.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p.y), hi.max(p.y)));
        let height = high - low;
        let torso = skin
            .iter()
            .filter(|p| p.x.abs() < 0.12 && p.y - low > 0.55 * height && p.y - low < 0.78 * height)
            .fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p.z), hi.max(p.z)));
        let centre_z = if torso.0 < torso.1 { 0.5 * (torso.0 + torso.1) } else { 0.0 };
        let scale = params.scale.clamp(0.5, 1.5);
        for p in &mut positions {
            *p = Vec3::new(p.x, p.y - low, p.z - centre_z) * scale;
        }

        // Joints sit at the centre of their marker vertices.
        let centre = |markers: &[u32]| markers.iter().map(|&i| positions[i as usize]).sum::<Vec3>() / markers.len() as f32;
        let joints = self.bones.iter().map(|bone| Joint { head: centre(&bone.head), tail: centre(&bone.tail) }).collect();
        positions.truncate(self.mesh_vertices);

        let mut mesh = BodyMesh {
            joints,
            positions,
            normals: Vec::new(),
            triangles: self.triangles.clone(),
            skin_triangles: self.skin_triangles,
            shoulder_points: self.shoulder_points,
            centres: Default::default(),
        };
        mesh.recompute_normals();
        mesh
    }
}

impl BodyMesh {
    pub(crate) fn recompute_normals(&mut self) {
        self.normals = vec![Vec3::ZERO; self.positions.len()];
        for &[a, b, c] in &self.triangles {
            let (pa, pb, pc) = (self.positions[a as usize], self.positions[b as usize], self.positions[c as usize]);
            let normal = (pb - pa).cross(pc - pa); // area weighted
            for i in [a, b, c] {
                self.normals[i as usize] += normal;
            }
        }
        for n in &mut self.normals {
            *n = n.normalize_or(Vec3::Y);
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], LoadError> {
        let end = self.at.checked_add(count).filter(|&end| end <= self.bytes.len()).ok_or(LoadError::Format)?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }
    fn u16(&mut self) -> Result<u16, LoadError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, LoadError> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, LoadError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, LoadError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn vec3(&mut self) -> Result<Vec3, LoadError> {
        Ok(Vec3::new(self.f32()?, self.f32()?, self.f32()?))
    }
}

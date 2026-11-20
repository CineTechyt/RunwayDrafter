//! What the avatar looks like: skin tone, eyes and hair. For display only:
//! none of it changes the body the cloth collides with.
//!
//! Hair is not simulated. It is modelled from the head it grows on:
//!
//! * a **shell** over the scalp, offset from the skin and feathered into
//!   it at the hairline;
//! * for longer styles a **curtain** that falls from the sides and back of
//!   the head, hanging straight but kept off the neck and shoulders by the
//!   body's outline around the head's axis;
//! * a **fringe** over the forehead, a **ponytail** (a tapering tube) or a
//!   **bun**.
//!
//! Each vertex also says how long its hair is (the UV's V): 0 shades as a
//! close crop, a fine grain over the scalp; 1 as strands. Fades are a
//! gradient of length, thickness and coverage up the sides of the head.
//!
//! The landmarks (eyes, crown, the head's axis) are found on the body, so
//! every style follows the body's shape and size.

use glam::{Vec2, Vec3};

use crate::model::BodyMesh;

/// Haircuts the avatar can wear.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HairStyle {
    Bald,
    /// Clipped to a few millimetres.
    Buzz,
    #[default]
    Short,
    /// Chin length all round.
    Bob,
    /// Falls past the shoulders.
    Long,
    Ponytail,
    Bun,
    /// Sides clipped short and fading to skin at the bottom, longer on top.
    TaperFade,
    /// Faded sides, textured top and a longer back.
    Mullet,
    /// Faded sides; the top swept up and back from the forehead.
    Quiff,
    /// Combed straight back, close to the head.
    SlickedBack,
}

impl HairStyle {
    pub const ALL: [HairStyle; 11] = [
        HairStyle::Bald,
        HairStyle::Buzz,
        HairStyle::TaperFade,
        HairStyle::Quiff,
        HairStyle::SlickedBack,
        HairStyle::Mullet,
        HairStyle::Short,
        HairStyle::Bob,
        HairStyle::Long,
        HairStyle::Ponytail,
        HairStyle::Bun,
    ];

    pub fn label(self) -> &'static str {
        match self {
            HairStyle::Bald => "Bald",
            HairStyle::Buzz => "Buzz cut",
            HairStyle::Short => "Short",
            HairStyle::Bob => "Bob",
            HairStyle::Long => "Long",
            HairStyle::Ponytail => "Ponytail",
            HairStyle::Bun => "Bun",
            HairStyle::TaperFade => "Taper fade",
            HairStyle::Mullet => "Modern mullet",
            HairStyle::Quiff => "Quiff",
            HairStyle::SlickedBack => "Slicked back",
        }
    }

    /// Whether [`Hair::length`] does anything for this style.
    pub fn has_length(self) -> bool {
        matches!(
            self,
            HairStyle::Bob | HairStyle::Long | HairStyle::Ponytail | HairStyle::Mullet
        )
    }

    /// Clipped sides that fade towards the skin.
    fn faded(self) -> bool {
        matches!(
            self,
            HairStyle::TaperFade | HairStyle::Mullet | HairStyle::Quiff
        )
    }

    /// Combed back from the forehead rather than falling from the crown.
    fn combed_back(self) -> bool {
        matches!(self, HairStyle::Quiff | HairStyle::SlickedBack)
    }

    /// Whether this style can have a fringe.
    pub fn takes_fringe(self) -> bool {
        matches!(self, HairStyle::Short | HairStyle::Bob | HairStyle::Long)
    }

    /// A length that suits the style, in metres from the crown.
    pub fn default_length(self) -> f32 {
        match self {
            HairStyle::Bob => 0.25,
            HairStyle::Long => 0.55,
            HairStyle::Ponytail => 0.38,
            HairStyle::Mullet => 0.25,
            _ => 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hair {
    pub style: HairStyle,
    /// Linear RGB.
    pub color: [f32; 3],
    /// How far the hair reaches down from the crown, in metres.
    pub length: f32,
    /// 0 = sleek and flat, 1 = full.
    pub volume: f32,
    pub fringe: bool,
}

impl Default for Hair {
    fn default() -> Self {
        Self {
            style: HairStyle::Short,
            color: [0.06, 0.035, 0.02],
            length: 0.0,
            volume: 0.5,
            fringe: false,
        }
    }
}

/// Everything about the avatar's looks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Appearance {
    /// Linear RGB.
    pub skin: [f32; 3],
    /// Iris colour, linear RGB.
    pub eyes: [f32; 3],
    pub hair: Hair,
}

impl Default for Appearance {
    fn default() -> Self {
        Self {
            skin: [0.62, 0.42, 0.31],
            eyes: [0.10, 0.17, 0.25],
            hair: Hair::default(),
        }
    }
}

/// A coloured triangle mesh in world space.
#[derive(Debug, Clone, Default)]
pub struct LookMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    /// For hair: U runs across the strands (metres), V is how long they
    /// are (0 a close crop, 1 strands).
    pub uvs: Vec<Vec2>,
    /// Linear RGB per vertex.
    pub colors: Vec<[f32; 3]>,
    /// For hair, how much of the surface the strands cover: 1 solid,
    /// towards 0 they thin out (hairline, cut ends).
    pub coverage: Vec<f32>,
    pub triangles: Vec<[u32; 3]>,
}

impl LookMesh {
    fn push(&mut self, position: Vec3, normal: Vec3, uv: Vec2, color: [f32; 3]) -> u32 {
        self.push_thin(position, normal, uv, color, 1.0)
    }

    fn push_thin(
        &mut self,
        position: Vec3,
        normal: Vec3,
        uv: Vec2,
        color: [f32; 3],
        coverage: f32,
    ) -> u32 {
        self.coverage.push(coverage);
        self.positions.push(position);
        self.normals.push(normal);
        self.uvs.push(uv);
        self.colors.push(color);
        self.positions.len() as u32 - 1
    }

    /// Connects a grid of `columns × rows` vertices starting at `base`,
    /// stored column by column.
    fn grid(&mut self, base: u32, columns: u32, rows: u32, flip: bool) {
        for c in 0..columns - 1 {
            for r in 0..rows - 1 {
                let a = base + c * rows + r;
                let (b, d, e) = (a + rows, a + 1, a + rows + 1);
                if flip {
                    self.triangles.extend([[a, d, b], [b, d, e]]);
                } else {
                    self.triangles.extend([[a, b, d], [b, e, d]]);
                }
            }
        }
    }
}

/// Landmarks of the head, found on the body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Head {
    /// Centres and radii of the eyeballs, right (−X) then left.
    pub eyes: [(Vec3, f32); 2],
    /// Midpoint between the eyes.
    pub eye_line: Vec3,
    /// Height of the top of the head.
    pub crown: f32,
    /// Depth (z) of the head's vertical axis.
    pub axis_z: f32,
    /// Half the skull's width above the ears; the ears stick out past it.
    pub skull: f32,
}

impl Head {
    /// Finds the head of a body that has eyes (the built-in body model).
    pub fn find(body: &BodyMesh) -> Option<Head> {
        let mut eye_vertices: Vec<u32> = body.triangles[body.skin_triangles..]
            .iter()
            .flatten()
            .copied()
            .collect();
        eye_vertices.sort_unstable();
        eye_vertices.dedup();
        let eye = |side: f32| -> Option<(Vec3, f32)> {
            let points: Vec<Vec3> = eye_vertices
                .iter()
                .map(|&i| body.positions[i as usize])
                .filter(|p| p.x * side > 0.0)
                .collect();
            if points.len() < 8 {
                return None;
            }
            let centre = points.iter().copied().sum::<Vec3>() / points.len() as f32;
            let radius =
                points.iter().map(|p| p.distance(centre)).sum::<f32>() / points.len() as f32;
            Some((centre, radius))
        };
        let eyes = [eye(-1.0)?, eye(1.0)?];
        let eye_line = 0.5 * (eyes[0].0 + eyes[1].0);
        let near_head =
            |p: &&Vec3| p.x.abs() < 0.13 && (p.z - eye_line.z).abs() < 0.2 && p.y > eye_line.y;
        let crown = body
            .positions
            .iter()
            .filter(near_head)
            .map(|p| p.y)
            .fold(f32::MIN, f32::max);
        if crown <= eye_line.y {
            return None;
        }
        // The head's front-to-back middle over the forehead band: the
        // front of the forehead and the back of the skull.
        let unit = crown - eye_line.y;
        let band = |p: &&Vec3| p.y > eye_line.y + 0.2 * unit && p.y < eye_line.y + 0.55 * unit;
        let (front, back) = body
            .positions
            .iter()
            .filter(band)
            .filter(|p| p.x.abs() < 0.04 && p.z < eye_line.z + 0.1 && p.z > eye_line.z - 0.35)
            .fold((f32::MIN, f32::MAX), |(front, back), p| {
                (front.max(p.z), back.min(p.z))
            });
        let axis_z = if front > back {
            0.5 * (front + back)
        } else {
            eye_line.z - 0.08
        };
        let skull = body
            .positions
            .iter()
            .filter(band)
            .filter(|p| (p.z - axis_z).abs() < 0.12 && p.x.abs() < 0.15)
            .map(|p| p.x.abs())
            .fold(0.0, f32::max);
        Some(Head {
            eyes,
            eye_line,
            crown,
            axis_z,
            skull: skull.max(0.05),
        })
    }

    /// Eyes to crown: the unit the head's features are measured in.
    pub fn unit(&self) -> f32 {
        self.crown - self.eye_line.y
    }

    /// Angle around the head's axis seen from above: 0 = front, ±π = back,
    /// positive towards the avatar's left (+X).
    fn azimuth(&self, p: Vec3) -> f32 {
        p.x.atan2(p.z - self.axis_z)
    }

    /// Height above the eyes in head units.
    fn level(&self, p: Vec3) -> f32 {
        (p.y - self.eye_line.y) / self.unit()
    }

    /// The point at `radius` from the axis, at `azimuth` and height `y`.
    fn around(&self, azimuth: f32, y: f32, radius: f32) -> Vec3 {
        Vec3::new(
            azimuth.sin() * radius,
            y,
            self.axis_z + azimuth.cos() * radius,
        )
    }

    /// How much `p` is part of an ear, 0..=1: what sticks out sideways past
    /// the skull, level with the eyes. The side of the skull itself, up to
    /// where the ear grows from it, is not.
    fn ear(&self, p: Vec3) -> f32 {
        let level = self.level(p);
        let a = self.azimuth(p).abs();
        let side = smoothstep(1.0 * self.skull, 1.06 * self.skull, p.x.abs());
        let height = smoothstep(-0.85, -0.7, level) * (1.0 - smoothstep(0.25, 0.4, level));
        let around = smoothstep(1.0, 1.2, a) * (1.0 - smoothstep(2.1, 2.3, a));
        // The bowl of the ear lies back near the skull: around the middle
        // of the ear, closer in counts too.
        let core = smoothstep(0.9 * self.skull, 0.95 * self.skull, p.x.abs())
            * smoothstep(-0.5, -0.4, level)
            * (1.0 - smoothstep(0.0, 0.08, level))
            * smoothstep(1.3, 1.4, a)
            * (1.0 - smoothstep(1.8, 1.9, a));
        (side * height * around).max(core)
    }

    /// Height of the hairline (head units) at an angle from the front:
    /// down the temples, a sideburn before the ear, round behind it to the
    /// nape.
    fn hairline(&self, azimuth: f32) -> f32 {
        const LINE: [(f32, f32); 10] = [
            (0.0, 0.6),
            (0.55, 0.58),
            (0.9, 0.4),
            (1.12, 0.3),
            (1.25, -0.12),
            (1.5, -0.25),
            (1.9, -0.3),
            (2.2, -0.5),
            (2.6, -0.7),
            (std::f32::consts::PI, -0.8),
        ];
        let a = azimuth.abs();
        let k = LINE
            .windows(2)
            .position(|w| a <= w[1].0)
            .unwrap_or(LINE.len() - 2);
        let ((a0, h0), (a1, h1)) = (LINE[k], LINE[k + 1]);
        h0 + (h1 - h0) * ((a - a0) / (a1 - a0)).clamp(0.0, 1.0)
    }

    /// Where hair grows, 0..=1: 1 on the scalp, 0 on the face, ears and
    /// neck, feathered across the hairline. The ears stay bare.
    fn scalp(&self, p: Vec3) -> f32 {
        let a = self.azimuth(p).abs();
        let line = self.hairline(a);
        // A soft, wide fade at the forehead; tighter at the sides, where
        // there is little room between the hairline and the ear.
        let (below, above) = if a < 0.9 { (0.1, 0.22) } else { (0.05, 0.1) };
        smoothstep(line - below, line + above, self.level(p)) * (1.0 - self.ear(p))
    }

    /// How the hair grows at skin point `p` for a cut.
    fn growth(&self, p: Vec3, hair: &Hair, thickness: f32) -> Growth {
        let m = self.scalp(p);
        if m <= 0.0 {
            return Growth::default();
        }
        let volume = hair.volume.clamp(0.0, 1.0);
        let level = self.level(p);
        let a = self.azimuth(p).abs();
        let cover = smoothstep(0.05, 0.9, m);
        // Fuller on top than at the sides and back.
        let profile = 0.6 + 0.4 * smoothstep(0.2, 0.8, level);
        let full = Growth {
            lift: 0.0015 + thickness * profile * m,
            coverage: cover,
            length: 1.0,
        };
        match hair.style {
            HairStyle::Bald => Growth::default(),
            // A close crop never covers the scalp completely.
            HairStyle::Buzz => Growth {
                lift: 0.0008 + 0.0012 * m,
                coverage: 0.72 * cover,
                length: 0.0,
            },
            style if style.faded() => {
                // Where the long top begins: a band around the head's
                // widest point; the front is all top. A mullet keeps its
                // back long too.
                let mut top = smoothstep(0.42, 0.6, level).max(1.0 - smoothstep(0.7, 1.0, a));
                if style == HairStyle::Mullet {
                    top = top.max(smoothstep(2.1, 2.5, a));
                }
                // Up the sides the clipped hair gets longer and denser.
                let above = level - self.hairline(a);
                let fade = smoothstep(0.0, 0.6, above);
                let side = Growth {
                    lift: 0.0006 + 0.004 * fade,
                    coverage: (0.2 + 0.65 * fade) * cover,
                    length: 0.15 * fade,
                };
                let mut top_lift = full.lift;
                if style == HairStyle::Quiff {
                    // Swept up off the forehead into a rounded crest.
                    let front = (-(a / 0.75).powi(2)).exp();
                    let band = smoothstep(0.4, 0.7, level) * (1.0 - smoothstep(0.95, 1.2, level));
                    top_lift += (0.022 + 0.02 * volume) * front * band;
                }
                Growth {
                    lift: side.lift + (top_lift - side.lift) * top,
                    coverage: side.coverage + (cover - side.coverage) * top,
                    length: side.length + (1.0 - side.length) * top,
                }
            }
            _ => full,
        }
    }
}

/// Hair at one point of the scalp.
#[derive(Debug, Clone, Copy, Default)]
struct Growth {
    /// Height of the hair's surface over the skin, metres.
    lift: f32,
    /// How much of the scalp it hides, 0..=1.
    coverage: f32,
    /// 0 a close crop, 1 long enough to show strands.
    length: f32,
}

/// Where the strands of a cut start from: the crown, or the forehead for
/// hair combed back.
pub fn hair_root(head: &Head, hair: &Hair) -> Vec3 {
    if hair.style.combed_back() {
        Vec3::new(
            0.0,
            head.eye_line.y + 0.35 * head.unit(),
            head.eye_line.z + 0.03,
        )
    } else {
        Vec3::new(0.0, head.crown, head.axis_z - 0.02)
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}

/// Repeatable noise in 0..1 from an integer.
fn hash(n: i32) -> f32 {
    let mut x = (n as u32).wrapping_mul(0x9E37_79B9) ^ 0x85EB_CA6B;
    x ^= x >> 15;
    x = x.wrapping_mul(0x2C1B_3C6D);
    x ^= x >> 12;
    (x & 0xFFFF) as f32 / 65535.0
}

/// Per-vertex skin colour: the skin tone with brows and lips painted on.
pub fn skin_colors(
    body: &BodyMesh,
    head: Option<&Head>,
    look: &Appearance,
    modest: bool,
) -> Vec<[f32; 3]> {
    // Modest mode: use a uniform neutral colour to hide skin details.
    if modest {
        return vec![[0.72, 0.72, 0.70]; body.positions.len()];
    }
    let skin = look.skin;
    let Some(head) = head else {
        return vec![skin; body.positions.len()];
    };
    let s = head.unit();
    // Brows follow the hair colour, a little darker than very fair hair.
    let brow = mix(look.hair.color, [0.03, 0.02, 0.015], 0.35);
    let lips = [skin[0] * 0.85, skin[1] * 0.5, skin[2] * 0.52];
    // Under the hair the skin is shadowed and tinted by the roots, which
    // is what makes a hairline fade instead of ending.
    let roots = mix(skin.map(|c| c * 0.4), look.hair.color, 0.6);
    let under = match look.hair.style {
        HairStyle::Bald => 0.0,
        HairStyle::Buzz => 0.85,
        style if style.faded() => 0.8,
        _ => 0.6,
    };
    body.positions
        .iter()
        .zip(&body.normals)
        .map(|(&p, &n)| {
            let density = head.growth(p, &look.hair, 0.0).coverage;
            let skin = if density > 0.0 {
                mix(skin, roots, under * density)
            } else {
                skin
            };
            if n.z < 0.1 || p.z < head.eye_line.z - 0.01 || (p.y - head.eye_line.y).abs() > 1.2 * s
            {
                return skin;
            }
            let mut color = skin;
            // Brows: an arc over each eye, thickest on the inner side.
            for (centre, radius) in head.eyes {
                let outward = (p.x - centre.x) * centre.x.signum() / (1.6 * radius);
                if outward.abs() < 1.15 {
                    let arch = centre.y + 0.2 * s + 0.045 * s * (1.0 - outward * outward);
                    let half = 0.034 * s * (1.0 - 0.45 * (outward + 1.0) * 0.5);
                    let along = 1.0 - smoothstep(0.85, 1.15, outward.abs());
                    let across = 1.0 - smoothstep(half * 0.5, half * 1.3, (p.y - arch).abs());
                    color = mix(color, brow, 0.7 * along * across);
                }
            }
            // Lips: an ellipse under the nose.
            let mouth = Vec2::new(
                p.x / (0.24 * s),
                (p.y - (head.eye_line.y - 0.62 * s)) / (0.1 * s),
            );
            let lip = 1.0 - smoothstep(0.7, 1.05, mouth.length());
            mix(color, lips, lip)
        })
        .collect()
}

/// Eyeballs that look straight ahead: sclera, iris and pupil, sized and
/// placed where the body's own eyes are.
pub fn eyeballs(head: &Head, look: &Appearance, modest: bool) -> LookMesh {
    const PUPIL: f32 = 0.14;
    const IRIS: f32 = 0.40;
    // Rings by angle from the line of sight, dense around the iris so its
    // edges stay round.
    let rings: Vec<f32> = [
        0.0,
        0.07,
        PUPIL,
        PUPIL + 0.025,
        0.23,
        0.3,
        0.36,
        IRIS,
        IRIS + 0.03,
        0.58,
        0.8,
        1.05,
        1.35,
        1.7,
        2.1,
        2.6,
        std::f32::consts::PI,
    ]
    .to_vec();
    let sectors = 36u32;
    let iris = if modest {
        [0.18, 0.18, 0.17]
    } else {
        look.eyes
    };
    let sclera = if modest {
        [0.55, 0.55, 0.54]
    } else {
        [0.80, 0.77, 0.72]
    };
    let mut mesh = LookMesh::default();
    for &(centre, radius) in &head.eyes {
        let base = mesh.positions.len() as u32;
        for s in 0..=sectors {
            let around = s as f32 / sectors as f32 * std::f32::consts::TAU;
            for &theta in &rings {
                let normal = Vec3::new(
                    theta.sin() * around.cos(),
                    theta.sin() * around.sin(),
                    theta.cos(),
                );
                let color = if theta < PUPIL {
                    [0.005; 3]
                } else if theta <= IRIS {
                    // Lighter towards the pupil, a dark ring at the edge.
                    let t = (theta - PUPIL) / (IRIS - PUPIL);
                    let fibres = 0.85 + 0.3 * hash(s as i32 * 7 + (t * 4.0) as i32);
                    let k = (1.25 - 0.55 * t) * fibres * if t > 0.85 { 0.45 } else { 1.0 };
                    iris.map(|c| c * k)
                } else {
                    // Veins and shadow towards the corners.
                    let shade = 1.0 - 0.25 * smoothstep(0.6, 1.4, theta);
                    sclera.map(|c| c * shade)
                };
                mesh.push(
                    centre + normal * radius,
                    normal,
                    Vec2::new(around, theta),
                    color,
                );
            }
        }
        let rows = rings.len() as u32;
        mesh.grid(base, sectors + 1, rows, true);
    }
    mesh
}

/// How far the body reaches from the head's axis, by angle and height:
/// what long hair has to fall around. Arms are left out.
struct Outline {
    azimuths: usize,
    step: f32,
    radius: Vec<f32>,
}

impl Outline {
    const AZIMUTHS: usize = 96;

    fn new(body: &BodyMesh, head: &Head) -> Self {
        let step = 0.005;
        let rows = (head.crown / step).ceil() as usize + 2;
        let mut radius = vec![0.0f32; rows * Self::AZIMUTHS];
        let shoulder = body.shoulder_points.map(|i| {
            body.positions
                .get(i as usize)
                .copied()
                .unwrap_or(Vec3::ZERO)
        });
        let torso_half_width = shoulder[0].x.abs().max(shoulder[1].x.abs()).max(0.12) + 0.03;
        let shoulder_top = shoulder[0].y.max(shoulder[1].y) + 0.06;
        for p in &body.positions {
            if p.y < shoulder_top && p.x.abs() > torso_half_width {
                continue; // arms
            }
            let d = Vec2::new(p.x, p.z - head.axis_z);
            let row = (p.y / step).floor() as isize;
            if row < 0 || row as usize >= rows {
                continue;
            }
            let bin = Self::bin(d.x.atan2(d.y));
            let slot = &mut radius[row as usize * Self::AZIMUTHS + bin];
            *slot = slot.max(d.length());
        }
        // Close the gaps between vertices: the largest of the neighbours.
        let mut grown = radius.clone();
        for row in 0..rows {
            for bin in 0..Self::AZIMUTHS {
                let mut best = 0.0f32;
                for dr in -1..=1isize {
                    for db in -2..=2isize {
                        let r = row as isize + dr;
                        if r < 0 || r as usize >= rows {
                            continue;
                        }
                        let b = (bin as isize + db).rem_euclid(Self::AZIMUTHS as isize) as usize;
                        best = best.max(radius[r as usize * Self::AZIMUTHS + b]);
                    }
                }
                grown[row * Self::AZIMUTHS + bin] = best;
            }
        }
        // Then round it off, so hair passes over an ear or a shoulder in
        // a curve rather than a step.
        let mut smooth = grown.clone();
        for row in 0..rows {
            for bin in 0..Self::AZIMUTHS {
                let (mut sum, mut count) = (0.0, 0.0);
                for dr in -3..=3isize {
                    for db in -3..=3isize {
                        let r = row as isize + dr;
                        if r < 0 || r as usize >= rows {
                            continue;
                        }
                        let b = (bin as isize + db).rem_euclid(Self::AZIMUTHS as isize) as usize;
                        sum += grown[r as usize * Self::AZIMUTHS + b];
                        count += 1.0;
                    }
                }
                // Never below the body itself.
                smooth[row * Self::AZIMUTHS + bin] =
                    (sum / count).max(radius[row * Self::AZIMUTHS + bin]);
            }
        }
        Self {
            azimuths: Self::AZIMUTHS,
            step,
            radius: smooth,
        }
    }

    fn bin(azimuth: f32) -> usize {
        let t = (azimuth / std::f32::consts::TAU).rem_euclid(1.0);
        ((t * Self::AZIMUTHS as f32) as usize).min(Self::AZIMUTHS - 1)
    }

    fn at(&self, azimuth: f32, y: f32) -> f32 {
        let rows = self.radius.len() / self.azimuths;
        let row = ((y / self.step).floor().max(0.0) as usize).min(rows - 1);
        self.radius[row * self.azimuths + Self::bin(azimuth)]
    }
}

/// The hair, or an empty mesh when bald.
pub fn hair(body: &BodyMesh, head: &Head, look: &Appearance, modest: bool) -> LookMesh {
    let hair = &look.hair;
    let mut mesh = LookMesh::default();
    if modest {
        return mesh;
    }
    if hair.style == HairStyle::Bald {
        return mesh;
    }
    let volume = hair.volume.clamp(0.0, 1.0);
    // Shell thickness on top of the head, in metres.
    let thickness = match hair.style {
        HairStyle::Bald => 0.0,
        HairStyle::Buzz => 0.0025,
        HairStyle::Short => 0.007 + 0.014 * volume,
        HairStyle::Bob => 0.009 + 0.014 * volume,
        HairStyle::Long => 0.006 + 0.012 * volume,
        HairStyle::Ponytail | HairStyle::Bun => 0.003 + 0.004 * volume,
        HairStyle::TaperFade => 0.008 + 0.012 * volume,
        HairStyle::Mullet => 0.01 + 0.012 * volume,
        HairStyle::Quiff => 0.007 + 0.006 * volume,
        HairStyle::SlickedBack => 0.004 + 0.005 * volume,
    };
    shell(&mut mesh, body, head, look, thickness);
    let outline = Outline::new(body, head);
    let length = hair.length.clamp(0.08, 1.2);
    match hair.style {
        HairStyle::Bob => curtain(
            &mut mesh, head, &outline, look, thickness, 1.05, length, false,
        ),
        HairStyle::Long => curtain(
            &mut mesh, head, &outline, look, thickness, 1.2, length, true,
        ),
        HairStyle::Ponytail => ponytail(&mut mesh, head, &outline, look, thickness, length),
        HairStyle::Mullet => curtain(
            &mut mesh, head, &outline, look, thickness, 2.15, length, false,
        ),
        HairStyle::Bun => bun(&mut mesh, head, &outline, look, thickness),
        _ => {}
    }
    if hair.fringe && hair.style.takes_fringe() {
        fringe(&mut mesh, head, &outline, look, thickness);
    }
    mesh
}

/// The scalp offset outwards, as the cut grows (see [`Head::growth`]):
/// thickest on top, thinning out over the hairline.
fn shell(mesh: &mut LookMesh, body: &BodyMesh, head: &Head, look: &Appearance, thickness: f32) {
    let skin = &body.triangles[..body.skin_triangles];
    let growth: Vec<Growth> = body
        .positions
        .iter()
        .map(|&p| head.growth(p, &look.hair, thickness))
        .collect();
    let mut remap = vec![u32::MAX; body.positions.len()];
    for triangle in skin {
        if triangle
            .iter()
            .all(|&i| growth[i as usize].coverage <= 0.02)
        {
            continue;
        }
        let corners = triangle.map(|i| {
            let i = i as usize;
            if remap[i] == u32::MAX {
                let (p, n, g) = (body.positions[i], body.normals[i], growth[i]);
                let radius = Vec2::new(p.x, p.z - head.axis_z).length();
                let uv = Vec2::new(head.azimuth(p) * radius.max(0.05), g.length);
                remap[i] = mesh.push_thin(p + n * g.lift, n, uv, look.hair.color, g.coverage);
            }
            remap[i]
        });
        mesh.triangles.push(corners);
    }
}

/// Hair that hangs from the sides and back of the head down to `length`
/// below the crown. It starts `front` radians from the front on either
/// side; `gather` draws it behind the shoulders as it falls. Two layers:
/// a darker one underneath gives the hair depth where the outer one thins.
#[allow(clippy::too_many_arguments)]
fn curtain(
    mesh: &mut LookMesh,
    head: &Head,
    outline: &Outline,
    look: &Appearance,
    thickness: f32,
    front: f32,
    length: f32,
    gather: bool,
) {
    let volume = look.hair.volume.clamp(0.0, 1.0);
    let inset = 0.004 + 0.006 * volume;
    curtain_layer(
        mesh,
        head,
        outline,
        look,
        thickness,
        front,
        length * 0.94,
        gather,
        -inset,
        0.45,
    );
    curtain_layer(
        mesh, head, outline, look, thickness, front, length, gather, 0.0, 1.0,
    );
}

#[allow(clippy::too_many_arguments)]
fn curtain_layer(
    mesh: &mut LookMesh,
    head: &Head,
    outline: &Outline,
    look: &Appearance,
    thickness: f32,
    front: f32,
    length: f32,
    gather: bool,
    offset: f32,
    shade: f32,
) {
    let s = head.unit();
    let volume = look.hair.volume.clamp(0.0, 1.0);
    let (columns, rows) = (96usize, 56usize);
    // Starts well up under the scalp layer, so it is solid where it
    // comes out (over the ears).
    let top = head.eye_line.y + 0.62 * s;
    let span = std::f32::consts::PI - front; // either side of centre back
    let neck = head.eye_line.y - 1.2 * s;
    let shoulders = head.eye_line.y - 2.6 * s;
    // Clumps: ripples across the hair, growing as it falls.
    let clump = 0.022;
    let ripple = 0.0012 + 0.0028 * volume;
    let mut grid: Vec<(Vec3, f32, f32)> = Vec::with_capacity(columns * rows); // point, U, coverage
    for c in 0..columns {
        let across = c as f32 / (columns - 1) as f32 * 2.0 - 1.0;
        // Shorter towards the face.
        let own_length = length * (1.0 - 0.12 * across * across);
        let bottom = (head.crown - own_length).min(top - 0.02);
        let mut radius: f32 = 0.0;
        let mut last_y = top;
        for r in 0..rows {
            let v = r as f32 / (rows - 1) as f32;
            let y = top + (bottom - top) * v;
            // Long hair is drawn in behind the shoulders as it falls.
            let squeeze = if gather {
                1.0 - 0.62 * smoothstep(head.eye_line.y - 0.3 * s, neck - 0.3 * s, y)
                    - 0.08 * smoothstep(neck, shoulders, y)
            } else {
                1.0
            };
            let azimuth = std::f32::consts::PI + across * span * squeeze;
            let body = outline.at(azimuth, y);
            // Lies on the head, then hangs: it may come in towards the
            // neck only slowly, and never into the body.
            let clearance = if y > neck {
                thickness * 0.85
            } else {
                0.008 + 0.01 * volume
            };
            radius = if r == 0 {
                body + clearance
            } else {
                (radius - 0.6 * (last_y - y)).max(body + clearance)
            };
            last_y = y;
            let u = across * span * 0.09;
            let phase = hash((u / clump).floor() as i32 + 31) * std::f32::consts::TAU;
            let wave = (u / clump * std::f32::consts::TAU + phase).sin()
                * ripple
                * smoothstep(0.05, 0.5, v);
            let placed = (radius + offset + wave).max(body + 0.003);
            // Strands end over the last part of the length, and thin out
            // at the sides by the face.
            // It also fades in at the top, where it comes out from under
            // the scalp layer.
            let coverage = (1.0 - smoothstep(0.8, 1.0, v))
                * (1.0 - smoothstep(0.88, 1.0, across.abs()))
                * smoothstep(0.0, 0.06, v);
            grid.push((head.around(azimuth, y, placed), u, coverage));
        }
    }
    // Normals from the surface itself, so the clumps catch the light.
    let at = |c: usize, r: usize| grid[c.min(columns - 1) * rows + r.min(rows - 1)].0;
    let base = mesh.positions.len() as u32;
    for c in 0..columns {
        for r in 0..rows {
            let (p, u, coverage) = grid[c * rows + r];
            let sideways = at(c + 1, r) - at(c.saturating_sub(1), r);
            let down = at(c, r + 1) - at(c, r.saturating_sub(1));
            let mut normal = sideways.cross(down).normalize_or(Vec3::Z);
            let outward = Vec3::new(p.x, 0.0, p.z - head.axis_z);
            if normal.dot(outward) < 0.0 {
                normal = -normal;
            }
            // Darker at the roots, under the head.
            let v = r as f32 / (rows - 1) as f32;
            let k = shade * (0.7 + 0.3 * smoothstep(0.0, 0.3, v));
            mesh.push_thin(
                p,
                normal,
                Vec2::new(u, 1.0),
                look.hair.color.map(|x| x * k),
                coverage,
            );
        }
    }
    mesh.grid(base, columns as u32, rows as u32, true);
}

/// A fringe over the forehead, cut straight above the brows.
fn fringe(mesh: &mut LookMesh, head: &Head, outline: &Outline, look: &Appearance, thickness: f32) {
    let s = head.unit();
    let columns = 40u32;
    let rows = 16u32;
    let base = mesh.positions.len() as u32;
    for c in 0..columns {
        let across = c as f32 / (columns - 1) as f32 * 2.0 - 1.0;
        let azimuth = across * 0.95;
        let top = head.eye_line.y + 0.95 * s;
        let bottom = head.eye_line.y + (0.3 + 0.12 * across * across) * s;
        for r in 0..rows {
            let v = r as f32 / (rows - 1) as f32;
            let y = top + (bottom - top) * v;
            let radius = outline.at(azimuth, y) + thickness * (1.0 - 0.7 * v) + 0.002;
            let p = head.around(azimuth, y, radius);
            let normal = Vec3::new(azimuth.sin(), 0.25, azimuth.cos()).normalize();
            let coverage = (1.0 - smoothstep(0.7, 1.0, v))
                * (1.0 - smoothstep(0.8, 1.0, across.abs()))
                * smoothstep(0.0, 0.25, v);
            mesh.push_thin(
                p,
                normal,
                Vec2::new(azimuth * radius, 1.0),
                look.hair.color,
                coverage,
            );
        }
    }
    mesh.grid(base, columns, rows, true);
}

/// A tube swept along `path` with `radius(t)` for t in 0..=1.
fn tube(
    mesh: &mut LookMesh,
    path: &[Vec3],
    radius: impl Fn(f32) -> f32,
    color: [f32; 3],
    coverage: impl Fn(f32) -> f32,
) {
    let sectors = 18u32;
    let base = mesh.positions.len() as u32;
    let n = path.len();
    for (k, &p) in path.iter().enumerate() {
        let t = k as f32 / (n - 1) as f32;
        let along =
            (path[(k + 1).min(n - 1)] - path[k.saturating_sub(1)]).normalize_or(Vec3::NEG_Y);
        let side = along.cross(Vec3::Z).try_normalize().unwrap_or(Vec3::X);
        let up = side.cross(along);
        let r = radius(t);
        for s in 0..=sectors {
            let a = s as f32 / sectors as f32 * std::f32::consts::TAU;
            let normal = side * a.cos() + up * a.sin();
            mesh.push_thin(
                p + normal * r,
                normal,
                Vec2::new(a * r, 1.0),
                color,
                coverage(t),
            );
        }
    }
    // Stored ring by ring: rings are "columns" of `sectors + 1`.
    mesh.grid(base, n as u32, sectors + 1, false);
}

fn ponytail(
    mesh: &mut LookMesh,
    head: &Head,
    outline: &Outline,
    look: &Appearance,
    thickness: f32,
    length: f32,
) {
    let s = head.unit();
    let volume = look.hair.volume.clamp(0.0, 1.0);
    let thick = 0.016 * (0.75 + 0.6 * volume) * (s / 0.1).clamp(0.7, 1.4);
    let back = std::f32::consts::PI;
    let tie_y = head.eye_line.y + 0.25 * s;
    let tie = head.around(
        back,
        tie_y,
        outline.at(back, tie_y) + thickness + 0.4 * thick,
    );
    // Out from the head, then down the back, never into it.
    let bottom = (head.crown - length).min(tie_y - 0.05);
    let samples = 40;
    let mut path = Vec::with_capacity(samples);
    let mut z = tie.z;
    for k in 0..samples {
        let t = k as f32 / (samples - 1) as f32;
        let y = tie_y + (bottom - tie_y) * t;
        let r = thick * taper(t);
        let wall = head.axis_z - outline.at(back, y) - r - 0.006;
        // Swings out behind the head before it falls.
        let swing = tie.z - 0.05 * (t * 6.0).min(1.0).sqrt();
        z = if k == 0 {
            tie.z
        } else {
            z.min(swing.max(z - 0.02)).min(wall)
        };
        path.push(Vec3::new(0.0, y, z));
    }
    tube(
        mesh,
        &path,
        |t| thick * taper(t),
        look.hair.color,
        |t| 1.0 - smoothstep(0.7, 1.0, t),
    );
    // The tie: a dark band.
    let band = [path[0], path[0] + (path[1] - path[0]) * 0.6];
    tube(mesh, &band, |_| thick * 0.95, [0.015, 0.015, 0.02], |_| 1.0);

    fn taper(t: f32) -> f32 {
        // Full just below the tie, tapering to a soft point.
        (0.8 + 0.5 * smoothstep(0.0, 0.12, t)) * (1.0 - 0.75 * smoothstep(0.35, 1.0, t))
    }
}

fn bun(mesh: &mut LookMesh, head: &Head, outline: &Outline, look: &Appearance, thickness: f32) {
    let s = head.unit();
    let volume = look.hair.volume.clamp(0.0, 1.0);
    let size = (0.034 + 0.016 * volume) * (s / 0.1).clamp(0.7, 1.4);
    let back = std::f32::consts::PI;
    let y = head.eye_line.y + 0.6 * s;
    let centre = head.around(back, y, outline.at(back, y) + thickness + 0.55 * size);
    let radii = Vec3::new(size, size * 0.85, size * 0.75);
    let (rings, sectors) = (16u32, 28u32);
    let base = mesh.positions.len() as u32;
    for s in 0..=sectors {
        let around = s as f32 / sectors as f32 * std::f32::consts::TAU;
        for r in 0..=rings {
            let phi = r as f32 / rings as f32 * std::f32::consts::PI;
            // The strands spiral round the bun.
            let unit = Vec3::new(
                phi.sin() * around.cos(),
                phi.sin() * around.sin(),
                phi.cos(),
            );
            // The pole faces away from the head (−Z).
            let p = centre + Vec3::new(unit.x * radii.x, unit.y * radii.y, -unit.z * radii.z);
            let normal =
                Vec3::new(unit.x / radii.x, unit.y / radii.y, -unit.z / radii.z).normalize();
            mesh.push(
                p,
                normal,
                Vec2::new((around + phi * 1.5) * size, 1.0),
                look.hair.color,
            );
        }
    }
    mesh.grid(base, sectors + 1, rings + 1, false);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashing_is_in_range() {
        for n in -50..50 {
            let h = hash(n);
            assert!((0.0..=1.0).contains(&h));
        }
    }
}

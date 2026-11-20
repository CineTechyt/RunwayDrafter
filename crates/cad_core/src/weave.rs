//! Woven fabric from its draft: which thread lies on top at every crossing
//! of warp (ends, along the grain) and weft (picks, across it), the order
//! the yarn colours come in, and how densely the threads are set.
//!
//! [`Weave::texture`] turns a draft into the fabric's look as images: every
//! crossing is a round yarn that rises where it floats on top and dives
//! where it goes under, threads vary a little in thickness and colour, and
//! the odd slub thickens a yarn for a stretch. It is the same weave a loom
//! would make, so stripes, checks, twill lines and diamonds come out right
//! without being drawn.

use serde::{Deserialize, Serialize};

/// A yarn colour of the weave.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Yarn {
    pub name: String,
    /// Linear RGB.
    pub color: [f32; 3],
}

/// A woven fabric's draft.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Weave {
    /// Threads in one repeat of the interlacing: warp ends across, weft picks along.
    pub ends: u32,
    pub picks: u32,
    /// `lift[pick * ends + end]`: the warp end lies on top of the pick there.
    pub lift: Vec<bool>,
    /// The yarns, and the order they come in across the warp and the weft
    /// (indices into `yarns`, repeated).
    pub yarns: Vec<Yarn>,
    pub warp_order: Vec<u8>,
    pub weft_order: Vec<u8>,
    /// Thread density.
    pub ends_per_cm: f32,
    pub picks_per_cm: f32,
    /// How unevenly the yarns are spun, 0..=1: thickness and colour vary
    /// from thread to thread.
    pub irregularity: f32,
    /// Slubs: thick stretches along a yarn, 0..=1.
    pub slubs: f32,
    /// How wide the yarns are relative to their spacing (about 0.5..=1.2):
    /// above 1 they flatten and crowd, hiding the threads beneath. Denim's
    /// heavy indigo warp nearly buries its white weft this way.
    #[serde(default = "default_cover")]
    pub warp_cover: f32,
    #[serde(default = "default_cover")]
    pub weft_cover: f32,
    /// Hairiness, 0..=1: loose fibres veil the yarns and light scatters
    /// through them, so neighbouring colours bleed into each other.
    #[serde(default = "default_fuzz")]
    pub fuzz: f32,
}

fn default_cover() -> f32 {
    0.86
}

fn default_fuzz() -> f32 {
    0.15
}

/// What a weave looks like, as two images of the same size that repeat
/// seamlessly: colour (sRGB, alpha 255) and surface (tangent-space normal
/// in RGB, height in alpha).
#[derive(Debug, Clone, PartialEq)]
pub struct WeaveTexture {
    pub width: u32,
    pub height: u32,
    /// Warp ends and weft picks the images span.
    pub ends: u32,
    pub picks: u32,
    pub albedo: Vec<[u8; 4]>,
    pub surface: Vec<[u8; 4]>,
}

impl Weave {
    /// A plain draft of `ends` × `picks` (warp on top where `up`).
    pub fn from_fn(ends: u32, picks: u32, up: impl Fn(u32, u32) -> bool) -> Self {
        let ends = ends.max(1);
        let picks = picks.max(1);
        let lift = (0..picks)
            .flat_map(|p| (0..ends).map(move |e| (e, p)))
            .map(|(e, p)| up(e, p))
            .collect();
        Self {
            ends,
            picks,
            lift,
            yarns: vec![
                Yarn {
                    name: "Blue".into(),
                    color: [0.045, 0.11, 0.36],
                },
                Yarn {
                    name: "Cream".into(),
                    color: [0.80, 0.74, 0.58],
                },
            ],
            warp_order: vec![0],
            weft_order: vec![1],
            ends_per_cm: 20.0,
            picks_per_cm: 20.0,
            irregularity: 0.35,
            slubs: 0.15,
            warp_cover: default_cover(),
            weft_cover: default_cover(),
            fuzz: default_fuzz(),
        }
    }

    /// Whether the warp end lies on top of the pick at this crossing.
    pub fn up(&self, end: u32, pick: u32) -> bool {
        let (e, p) = (end % self.ends.max(1), pick % self.picks.max(1));
        self.lift
            .get((p * self.ends + e) as usize)
            .copied()
            .unwrap_or(false)
    }

    /// Changes the size of the repeat, keeping what fits.
    pub fn resize(&mut self, ends: u32, picks: u32) {
        let (ends, picks) = (ends.clamp(1, 64), picks.clamp(1, 64));
        let old = self.clone();
        self.lift = (0..picks)
            .flat_map(|p| (0..ends).map(move |e| (e, p)))
            .map(|(e, p)| e < old.ends && p < old.picks && old.up(e, p))
            .collect();
        self.ends = ends;
        self.picks = picks;
    }

    /// Swaps which thread is on top at one crossing.
    pub fn toggle(&mut self, end: u32, pick: u32) {
        if end < self.ends && pick < self.picks {
            let i = (pick * self.ends + end) as usize;
            self.lift[i] = !self.lift[i];
        }
    }

    /// The colour of warp end `end` / weft pick `pick`.
    fn warp_yarn(&self, end: u32) -> [f32; 3] {
        self.yarn(&self.warp_order, end)
    }

    fn weft_yarn(&self, pick: u32) -> [f32; 3] {
        self.yarn(&self.weft_order, pick)
    }

    fn yarn(&self, order: &[u8], thread: u32) -> [f32; 3] {
        let index = if order.is_empty() {
            0
        } else {
            order[thread as usize % order.len()] as usize
        };
        self.yarns.get(index).map_or([0.5; 3], |y| y.color)
    }

    /// A number that changes whenever the look changes: for caching the
    /// textures made from it.
    pub fn key(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(0x0100_0000_01b3);
        };
        eat(u64::from(self.ends));
        eat(u64::from(self.picks));
        for &b in &self.lift {
            eat(u64::from(b));
        }
        for yarn in &self.yarns {
            for c in yarn.color {
                eat(u64::from(c.to_bits()));
            }
        }
        for &o in self
            .warp_order
            .iter()
            .chain([&255u8])
            .chain(&self.weft_order)
        {
            eat(u64::from(o));
        }
        for v in [
            self.ends_per_cm,
            self.picks_per_cm,
            self.irregularity,
            self.slubs,
            self.warp_cover,
            self.weft_cover,
            self.fuzz,
        ] {
            eat(u64::from(v.to_bits()));
        }
        h
    }

    /// How many warp ends and weft picks a texture of `pixels` per thread
    /// spans (see [`Weave::texture`]).
    pub fn texture_threads(&self, pixels: u32) -> (u32, u32) {
        let pixels = pixels.clamp(4, 32);
        let span = |repeat: u32, order: usize| {
            let base = lcm(repeat.max(1), order.max(1) as u32);
            let times = 96u32.div_ceil(base).max(1);
            (base * times)
                .min(1024 / pixels)
                .max(base.min(1024 / pixels))
        };
        (
            span(self.ends, self.warp_order.len()),
            span(self.picks, self.weft_order.len()),
        )
    }

    /// How many times the texture repeats over `size_mm` (across, along
    /// the grain) of fabric.
    pub fn repeats(&self, pixels: u32, size_mm: [f32; 2]) -> [f32; 2] {
        let (ends, picks) = self.texture_threads(pixels);
        [
            size_mm[0] * self.ends_per_cm.max(0.1) / (10.0 * ends as f32),
            size_mm[1] * self.picks_per_cm.max(0.1) / (10.0 * picks as f32),
        ]
    }

    /// The fabric's look, `pixels` per thread. The images span whole
    /// repeats of both the interlacing and the colour order, and enough
    /// threads (at least 96 each way) that the yarns' irregularities do not
    /// visibly repeat.
    pub fn texture(&self, pixels: u32) -> WeaveTexture {
        let pixels = pixels.clamp(4, 32);
        let (ends, picks) = self.texture_threads(pixels);
        let (width, height) = (ends * pixels, picks * pixels);

        // Per thread: thickness and colour, a little different each.
        let irregular = self.irregularity.clamp(0.0, 1.0);
        // Kept small: the texture repeats, and from afar every thread that
        // is darker than the rest turns into a line of a regular grid.
        // Shade and slubs are left to the renderer, which varies them per
        // thread of the whole garment (see `yarn_variation` in pbr.wgsl);
        // only a trace of thickness stays here.
        let thickness =
            |seed: u32, thread: u32| 1.0 + (hash(seed, thread) - 0.5) * 0.06 * irregular;
        let shade =
            |seed: u32, thread: u32| 1.0 + (hash(seed ^ 0x9e37, thread) - 0.5) * 0.02 * irregular;
        // Along a thread: slubs, smooth over a few crossings.
        let slubs = self.slubs.clamp(0.0, 1.0);
        // The texture repeats: noise along a thread must come round to
        // where it started after `length` crossings (about three per cell).
        let slub = |seed: u32, thread: u32, along: f32, length: u32| {
            let cells = (length / 3).max(1);
            let n = value_noise(seed, thread, along * cells as f32 / length as f32, cells);
            1.0 + slubs * 0.15 * ((n - 0.72) / 0.28).clamp(0.0, 1.0)
        };

        // How long each thread floats on top around a crossing: its height
        // arches over the float and dives at its ends.
        let warp_float = |end: u32, pick: u32| float_run(|p| self.up(end, p), pick, picks);
        let weft_float = |end: u32, pick: u32| float_run(|e| !self.up(e, pick), end, ends);

        let mut heights = vec![0.0f32; (width * height) as usize];
        let mut colors = vec![[0.0f32; 3]; (width * height) as usize];
        // Which way the yarn on top runs: 255 warp, 0 weft, 128 a gap.
        let mut directions = vec![128u8; (width * height) as usize];
        // Yarn width relative to the thread spacing. Yarns wider than their
        // spacing crowd: where a neighbour of the same set lies on top, it
        // spills over onto this crossing and covers what floats here.
        let warp_cover = self.warp_cover.clamp(0.4, 1.4);
        let weft_cover = self.weft_cover.clamp(0.4, 1.4);
        let warp_spill = ((warp_cover - 1.0) * 0.5).max(0.0);
        let weft_spill = ((weft_cover - 1.0) * 0.5).max(0.0);
        let fullest = warp_cover.max(weft_cover);
        let half = |warp: bool, thread: u32, along: f32| {
            let (cover, seed, length) = if warp {
                (warp_cover, 1, picks)
            } else {
                (weft_cover, 2, ends)
            };
            0.5 * cover * thickness(seed, thread) * slub(seed + 2, thread, along, length)
        };
        let profile = |a: f32| (1.0 - a * a).max(0.0).sqrt();
        // Fibres twist round the yarn: fine diagonal lines along it, a whole
        // number per crossing so they meet at the tile's edge.
        let twist = |along: f32, across: f32| {
            0.94 + 0.06 * ((along * 2.0 + across) * std::f32::consts::TAU).sin()
        };
        let wrap = |i: u32, d: i32, n: u32| (i as i32 + d).rem_euclid(n as i32) as u32;
        for y in 0..height {
            let pick = y / pixels;
            let fy = ((y % pixels) as f32 + 0.5) / pixels as f32;
            for x in 0..width {
                let end = x / pixels;
                let fx = ((x % pixels) as f32 + 0.5) / pixels as f32;
                let along_warp = pick as f32 + fy;
                let along_weft = end as f32 + fx;
                let warp_on_top = self.up(end, pick);

                // Candidates for what is seen here, nearest the eye first:
                // (warp?, thread, distance from its centre in spacings, arch).
                let mut candidates: Vec<(bool, u32, f32, f32)> = Vec::with_capacity(4);
                if warp_on_top {
                    if fy < weft_spill && !self.up(end, wrap(pick, -1, picks)) {
                        candidates.push((false, wrap(pick, -1, picks), fy + 0.5, 0.8));
                    }
                    if fy > 1.0 - weft_spill && !self.up(end, wrap(pick, 1, picks)) {
                        candidates.push((false, wrap(pick, 1, picks), 1.5 - fy, 0.8));
                    }
                } else {
                    if fx < warp_spill && self.up(wrap(end, -1, ends), pick) {
                        candidates.push((true, wrap(end, -1, ends), fx + 0.5, 0.8));
                    }
                    if fx > 1.0 - warp_spill && self.up(wrap(end, 1, ends), pick) {
                        candidates.push((true, wrap(end, 1, ends), 1.5 - fx, 0.8));
                    }
                }
                // This crossing's own float: it arches over its length.
                let (back, length) = if warp_on_top {
                    warp_float(end, pick)
                } else {
                    weft_float(end, pick)
                };
                let within = if warp_on_top { fy } else { fx };
                let position = (back as f32 + within) / length as f32;
                let arch = 0.55 + 0.45 * (std::f32::consts::PI * position.clamp(0.0, 1.0)).sin();
                if warp_on_top {
                    candidates.push((true, end, (fx - 0.5).abs(), arch));
                } else {
                    candidates.push((false, pick, (fy - 0.5).abs(), arch));
                }
                let top_count = candidates.len();
                // The crossing thread beneath shows between the floats.
                if warp_on_top {
                    candidates.push((false, pick, (fy - 0.5).abs(), 0.35));
                } else {
                    candidates.push((true, end, (fx - 0.5).abs(), 0.35));
                }

                let seen =
                    candidates
                        .iter()
                        .enumerate()
                        .find_map(|(k, &(warp, thread, d, arch))| {
                            let along = if warp { along_warp } else { along_weft };
                            let a = d / half(warp, thread, along).min(0.75);
                            (a < 1.0).then_some((k < top_count, warp, thread, a, arch))
                        });
                let yarn_color = |warp: bool, thread: u32| {
                    if warp {
                        scale(self.warp_yarn(thread), shade(1, thread))
                    } else {
                        scale(self.weft_yarn(thread), shade(2, thread))
                    }
                };
                let (h, color, direction) = match seen {
                    Some((true, warp, thread, a, arch)) => {
                        // A thinner yarn sits lower among fuller ones.
                        let cover = if warp { warp_cover } else { weft_cover };
                        let h = profile(a) * arch * (cover / fullest).sqrt();
                        let t = if warp {
                            twist(along_warp, fx)
                        } else {
                            twist(along_weft, fy)
                        };
                        // Sunk between fuller yarns it is also in their
                        // shadow, and their loose fibres lie over it.
                        let sunk = (cover / fullest).powf(1.5);
                        let mut c = scale(yarn_color(warp, thread), t * (0.7 + 0.3 * h) * sunk);
                        if cover < fullest {
                            let fuller = if warp {
                                yarn_color(false, pick)
                            } else {
                                yarn_color(true, end)
                            };
                            let veil = self.fuzz.clamp(0.0, 1.0) * 0.5;
                            for k in 0..3 {
                                c[k] += (fuller[k] - c[k]) * veil;
                            }
                        }
                        (h, c, warp)
                    }
                    Some((false, warp, thread, a, _)) => {
                        // Low, and in the shadow of the floats.
                        let h = profile(a) * 0.35;
                        (h, scale(yarn_color(warp, thread), 0.55 + 0.2 * h), warp)
                    }
                    None => (
                        0.0,
                        scale(
                            yarn_color(warp_on_top, if warp_on_top { end } else { pick }),
                            0.3,
                        ),
                        warp_on_top,
                    ),
                };
                let i = (y * width + x) as usize;
                heights[i] = h;
                colors[i] = color;
                directions[i] = if seen.is_none() {
                    128
                } else if direction {
                    255
                } else {
                    0
                };
            }
        }

        // Fuzz: loose fibres and light scattered through the yarns carry
        // each colour a little way into its neighbours (in linear light).
        let fuzz = self.fuzz.clamp(0.0, 1.0);
        if fuzz > 0.0 {
            let radius = ((0.7 * pixels as f32 * fuzz).round() as i64).max(1);
            let blurred = box_blur(
                &box_blur(&colors, width, height, radius),
                width,
                height,
                radius,
            );
            let amount = 0.75 * fuzz;
            for (c, b) in colors.iter_mut().zip(&blurred) {
                for k in 0..3 {
                    c[k] += (b[k] - c[k]) * amount;
                }
            }
        }

        // Normals from the heights; the yarns stand about a third of a
        // thread spacing proud.
        let at = |x: i64, y: i64| {
            let (w, h) = (i64::from(width), i64::from(height));
            heights[(y.rem_euclid(h) * w + x.rem_euclid(w)) as usize]
        };
        let bump = 0.35 * pixels as f32 / 2.0;
        let mut surface = Vec::with_capacity(heights.len());
        let mut albedo = Vec::with_capacity(heights.len());
        for y in 0..i64::from(height) {
            for x in 0..i64::from(width) {
                let dx = (at(x + 1, y) - at(x - 1, y)) * bump / pixels as f32 * 2.0;
                let dy = (at(x, y + 1) - at(x, y - 1)) * bump / pixels as f32 * 2.0;
                let n = normalize([-dx, -dy, 1.0]);
                let h = at(x, y);
                surface.push([
                    unorm(n[0] * 0.5 + 0.5),
                    unorm(n[1] * 0.5 + 0.5),
                    unorm(n[2] * 0.5 + 0.5),
                    unorm(h),
                ]);
                let c = colors[(y * i64::from(width) + x) as usize];
                // Alpha: the direction of the yarn on top, for highlights
                // that run along it (averaged over distance by the mips).
                let i = (y * i64::from(width) + x) as usize;
                albedo.push([srgb(c[0]), srgb(c[1]), srgb(c[2]), directions[i]]);
            }
        }
        WeaveTexture {
            width,
            height,
            ends,
            picks,
            albedo,
            surface,
        }
    }

    /// The colour of one repeat as the editor shows it: the texture with
    /// the yarns' irregularity and slubs on it (which the renderer adds
    /// per thread of the garment instead), opaque. sRGB RGBA8.
    pub fn preview(&self, pixels: u32) -> WeaveTexture {
        let mut texture = self.texture(pixels);
        let pixels = pixels.clamp(4, 32);
        let irregular = self.irregularity.clamp(0.0, 1.0);
        let slubs = self.slubs.clamp(0.0, 1.0);
        let smooth = |seed: u32, t: f32| {
            let i = t.floor();
            let f = t - i;
            let f = f * f * (3.0 - 2.0 * f);
            let (a, b) = (hash(seed, i as u32), hash(seed, i as u32 + 1));
            a + (b - a) * f
        };
        for y in 0..texture.height {
            for x in 0..texture.width {
                let (end, pick) = (x / pixels, y / pixels);
                let (along_warp, along_weft) = (
                    (y as f32 + 0.5) / pixels as f32,
                    (x as f32 + 0.5) / pixels as f32,
                );
                let mut shade = 1.0 + (hash(11, end) - 0.5) * 0.22 * irregular;
                shade *= 1.0 + (hash(12, pick) - 0.5) * 0.12 * irregular;
                let slub_warp = ((smooth(13 ^ end.wrapping_mul(7919), along_warp / 5.0) - 0.78)
                    / 0.2)
                    .clamp(0.0, 1.0);
                let slub_weft = ((smooth(14 ^ pick.wrapping_mul(7919), along_weft / 5.0) - 0.80)
                    / 0.18)
                    .clamp(0.0, 1.0);
                shade *= 1.0 + slubs * 0.35 * slub_warp.max(slub_weft * 0.6);
                shade *= 1.0 + (smooth(15, end as f32 * 0.21) - 0.5) * 0.14 * irregular;
                let p = &mut texture.albedo[(y * texture.width + x) as usize];
                for c in &mut p[..3] {
                    *c = srgb(linear(*c) * shade);
                }
                p[3] = 255;
            }
        }
        texture
    }

    /// Takes over `other`'s interlacing, keeping this weave's yarns.
    pub fn take_interlacing(&mut self, other: &Weave) {
        self.ends = other.ends;
        self.picks = other.picks;
        self.lift = other.lift.clone();
    }

    /// Whether a library entry brings its own yarns (a fabric, not just an
    /// interlacing).
    pub fn is_preset(name: &str) -> bool {
        matches!(name, "Denim" | "Gingham")
    }

    /// Denim: a 3/1 twill of a heavy, ring-dyed `warp` (the indigo) over a
    /// lighter `weft` (the white), which the crowding warp nearly hides.
    pub fn denim(warp: [f32; 3], weft: [f32; 3]) -> Self {
        let mut denim = Weave::from_fn(4, 4, |e, p| (e + 4 - p % 4) % 4 < 3);
        denim.yarns = vec![
            Yarn {
                name: "Indigo".into(),
                color: warp,
            },
            Yarn {
                name: "Ecru".into(),
                color: weft,
            },
        ];
        // About 62 ends and 40 picks per inch, as a 13–14 oz denim.
        denim.ends_per_cm = 24.0;
        denim.picks_per_cm = 16.0;
        denim.irregularity = 0.75;
        denim.slubs = 0.55;
        denim.warp_cover = 1.16;
        denim.weft_cover = 0.78;
        denim.fuzz = 0.45;
        denim
    }

    /// The weaves to start from.
    pub fn library() -> Vec<(&'static str, Weave)> {
        let twill = |up: u32, down: u32| {
            let n = up + down;
            Weave::from_fn(n, n, move |e, p| (e + n - p % n) % n < up)
        };
        let satin = |n: u32, step: u32| Weave::from_fn(n, n, move |e, p| e == (p * step) % n);
        let denim = Self::denim([0.018, 0.038, 0.13], [0.62, 0.60, 0.53]);
        let mut check = Weave::from_fn(2, 2, |e, p| (e + p) % 2 == 0);
        check.warp_order = [vec![0u8; 6], vec![1u8; 6]].concat();
        check.weft_order = check.warp_order.clone();
        vec![
            ("Plain", Weave::from_fn(2, 2, |e, p| (e + p) % 2 == 0)),
            ("2/1 twill", twill(2, 1)),
            ("2/2 twill", twill(2, 2)),
            ("3/1 twill", twill(3, 1)),
            ("Denim", denim),
            (
                "Herringbone",
                Weave::from_fn(8, 4, |e, p| {
                    let (e, p) = (e % 8, p % 4);
                    if e < 4 {
                        (e + 4 - p) % 4 < 2
                    } else {
                        (e + p) % 4 < 2
                    }
                }),
            ),
            ("5-end satin", satin(5, 2)),
            ("8-end satin", satin(8, 3)),
            (
                "2/2 basket",
                Weave::from_fn(4, 4, |e, p| (e / 2 + p / 2) % 2 == 0),
            ),
            ("Gingham", check),
            (
                "Diamond",
                Weave::from_fn(24, 24, |e, p| {
                    let d = (e as i32 - 12).abs() + (p as i32 - 12).abs();
                    d.rem_euclid(4) < 2
                }),
            ),
        ]
    }
}

/// How many crossings before `at` the float through it starts, and how
/// many crossings it lasts, `on_top(i)` saying whether the thread is on top
/// at crossing `i` of a repeat of `len`.
fn float_run(on_top: impl Fn(u32) -> bool, at: u32, len: u32) -> (u32, u32) {
    let len = len.max(1);
    let mut start = at;
    let mut count = 1;
    while count < len && on_top((start + len - 1) % len) {
        start = (start + len - 1) % len;
        count += 1;
    }
    let mut end = at;
    while count < len && on_top((end + 1) % len) {
        end = (end + 1) % len;
        count += 1;
    }
    ((at % len + len - start) % len, count)
}

fn lcm(a: u32, b: u32) -> u32 {
    let gcd = |mut a: u32, mut b: u32| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    a / gcd(a, b).max(1) * b
}

/// A repeatable random number in 0..1 for a thread.
fn hash(seed: u32, thread: u32) -> f32 {
    let mut x = thread.wrapping_mul(0x9E37_79B1) ^ seed.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 15;
    x = x.wrapping_mul(0x2C1B_3C6D);
    x ^= x >> 12;
    x = x.wrapping_mul(0x297A_2D39);
    x ^= x >> 15;
    (x & 0x00FF_FFFF) as f32 / 0x0100_0000 as f32
}

/// Smooth noise along a thread, repeating every `period` units of `t`.
fn value_noise(seed: u32, thread: u32, t: f32, period: u32) -> f32 {
    let i = t.floor();
    let f = t - i;
    let f = f * f * (3.0 - 2.0 * f);
    let period = period.max(1) as i64;
    let lattice = |k: i64| k.rem_euclid(period) as u32;
    let a = hash(seed ^ thread.wrapping_mul(7919), lattice(i as i64));
    let b = hash(seed ^ thread.wrapping_mul(7919), lattice(i as i64 + 1));
    a + (b - a) * f
}

/// A box blur of a wrapping image, across then down.
fn box_blur(image: &[[f32; 3]], width: u32, height: u32, radius: i64) -> Vec<[f32; 3]> {
    let (w, h) = (i64::from(width), i64::from(height));
    let n = (2 * radius + 1) as f32;
    let mut across = vec![[0.0f32; 3]; image.len()];
    for y in 0..h {
        for x in 0..w {
            let mut sum = [0.0f32; 3];
            for d in -radius..=radius {
                let p = image[(y * w + (x + d).rem_euclid(w)) as usize];
                for k in 0..3 {
                    sum[k] += p[k];
                }
            }
            across[(y * w + x) as usize] = sum.map(|v| v / n);
        }
    }
    let mut down = vec![[0.0f32; 3]; image.len()];
    for y in 0..h {
        for x in 0..w {
            let mut sum = [0.0f32; 3];
            for d in -radius..=radius {
                let p = across[((y + d).rem_euclid(h) * w + x) as usize];
                for k in 0..3 {
                    sum[k] += p[k];
                }
            }
            down[(y * w + x) as usize] = sum.map(|v| v / n);
        }
    }
    down
}

fn scale(c: [f32; 3], k: f32) -> [f32; 3] {
    c.map(|v| v * k)
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    v.map(|c| c / l)
}

fn unorm(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Linear → sRGB, 8 bit.
fn linear(c: u8) -> f32 {
    let s = f32::from(c) / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

fn srgb(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    unorm(s)
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_texture_says_which_yarn_is_on_top() {
        // 3/1 twill (denim): most of the face is warp (the weft shows in
        // the gaps between the floats too).
        let denim = Weave::library()
            .into_iter()
            .find(|(n, _)| *n == "Denim")
            .unwrap()
            .1;
        let t = denim.texture(8);
        let (warp, weft) = t.albedo.iter().fold((0, 0), |(w, f), p| match p[3] {
            255 => (w + 1, f),
            0 => (w, f + 1),
            _ => (w, f),
        });
        assert!(2 * warp > 3 * weft, "warp {warp}, weft {weft}");
        // The editor's preview is opaque.
        assert!(denim.preview(8).albedo.iter().all(|p| p[3] == 255));
    }
    use super::*;

    #[test]
    fn a_twill_climbs_one_end_per_pick() {
        let library = Weave::library();
        let twill = &library.iter().find(|(n, _)| *n == "2/2 twill").unwrap().1;
        for p in 0..4 {
            for e in 0..4 {
                assert_eq!(
                    twill.up(e, p),
                    twill.up(e + 1, p + 1),
                    "diagonal at {e},{p}"
                );
            }
        }
        // Half the crossings show warp.
        assert_eq!(twill.lift.iter().filter(|&&b| b).count(), 8);
    }

    #[test]
    fn the_texture_repeats_seamlessly_and_shows_both_yarns() {
        let weave = Weave::from_fn(2, 2, |e, p| (e + p) % 2 == 0);
        let texture = weave.texture(8);
        assert_eq!(texture.width, texture.ends * 8);
        assert!(texture.ends >= 96 && texture.ends % 2 == 0);
        assert_eq!(
            texture.albedo.len(),
            (texture.width * texture.height) as usize
        );
        // Blue warp and cream weft both show up.
        let blue = texture
            .albedo
            .iter()
            .filter(|c| u16::from(c[2]) > u16::from(c[0]) + 40)
            .count();
        let cream = texture
            .albedo
            .iter()
            .filter(|c| c[0] > 150 && c[1] > 140)
            .count();
        assert!(blue > texture.albedo.len() / 6, "{blue}");
        assert!(cream > texture.albedo.len() / 6, "{cream}");
        // Heights: the floats are proud, the gaps deep.
        let heights: Vec<u8> = texture.surface.iter().map(|s| s[3]).collect();
        assert!(heights.iter().any(|&h| h > 200) && heights.iter().any(|&h| h < 20));
    }

    #[test]
    fn the_texture_meets_itself_at_its_edges() {
        // The last column runs on into the first as any two neighbouring
        // columns do: no seam where the tile repeats.
        for (name, weave) in Weave::library() {
            let t = weave.texture(8);
            let (w, h) = (t.width as usize, t.height as usize);
            let step = |a: usize, b: usize| {
                (0..h)
                    .map(|y| {
                        let (p, q) = (t.surface[y * w + a], t.surface[y * w + b]);
                        (i32::from(p[3]) - i32::from(q[3])).abs()
                    })
                    .sum::<i32>() as f32
                    / h as f32
            };
            let seam = step(w - 1, 0);
            let inside = (1..w).map(|x| step(x - 1, x)).sum::<f32>() / (w - 1) as f32;
            assert!(
                seam < 3.0 * inside + 4.0,
                "{name}: seam {seam:.1} vs {inside:.1}"
            );
        }
    }

    #[test]
    fn floats_are_measured_round_the_repeat() {
        // On top at 3, 4, 0 of a repeat of 5: the float through 0 started at 3.
        let on = |i: u32| [true, false, false, true, true][i as usize];
        let (back, length) = float_run(on, 0, 5);
        assert_eq!((back, length), (2, 3));
    }

    #[test]
    fn the_key_follows_the_look() {
        let mut weave = Weave::library().remove(0).1;
        let key = weave.key();
        weave.toggle(0, 0);
        assert_ne!(weave.key(), key);
        weave.toggle(0, 0);
        assert_eq!(weave.key(), key);
    }
}

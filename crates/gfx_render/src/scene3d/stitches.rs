//! Stitches on the draped garment: every stitch a short piece of thread
//! lying on the cloth, the way the sewing machine laid it.
//!
//! The rows are worked out in the flat pattern, beside the seam or edge
//! at the distance the stitch is set to, and pinned to the cloth's mesh
//! once ([`Stitches::build`]); after that they follow every fold of the
//! drape ([`Stitches::mesh`]). Rows can be revealed stitch by stitch, with
//! a needle going up and down at the end, as if being sewn just then.

use cad_core::{Curve, PatternDocument, PieceId, StitchKind, StitchSide, Stitching, Zipper, ZipperTeeth};
use glam::{Vec2, Vec3};
use topology_sync::{GarmentMesh, PatternLocation, TopologySync};

use super::mesh::{MeshData, MeshVertex};

const MM: f32 = 1e-3;

/// A point pinned to the cloth.
#[derive(Debug, Clone, Copy)]
struct Pin {
    triangle: u32,
    barycentric: Vec3,
}

/// One stitch: a short thread on one face of the cloth.
#[derive(Debug, Clone)]
struct Thread {
    pins: Vec<Pin>,
    outside: bool,
    color: [f32; 3],
    radius: f32,
    /// Where the cloth's face is, off the simulated surface (half the
    /// fabric's thickness), metres.
    face: f32,
    /// Which row, and how far along it (0..1): for sewing it row by row.
    row: u64,
    along: f32,
}

/// One tooth of a zipper: where it sits on both sides of the seam (on the
/// edge, and a little in, for the direction across).
#[derive(Debug, Clone, Copy)]
struct Tooth {
    along: f32,
    a: Pin,
    a_in: Pin,
    b: Pin,
    b_in: Pin,
}

#[derive(Debug, Clone)]
struct ZipperRow {
    zipper: Zipper,
    face: f32,
    pitch: f32,
    teeth: Vec<Tooth>,
}

/// A row being sewn: where the needle is.
#[derive(Debug, Clone, Copy)]
pub struct Needle {
    pub position: Vec3,
    pub normal: Vec3,
}

/// The stitches of a garment, pinned to its mesh.
#[derive(Debug, Default)]
pub struct Stitches {
    threads: Vec<Thread>,
    zippers: Vec<ZipperRow>,
    /// Every row: its key, and pins along it for the needle.
    rows: Vec<(u64, Vec<(f32, Pin, f32)>)>,
}

/// The rows of a stitch, as offsets from the seam line (mm) and the
/// pattern of the thread along them.
fn rows_of(stitching: &Stitching) -> Vec<(f32, Pattern, bool)> {
    let s = stitching;
    match s.kind {
        StitchKind::Plain => Vec::new(),
        StitchKind::Topstitch | StitchKind::Edgestitch => vec![(s.offset, Pattern::Straight, true)],
        StitchKind::DoubleTopstitch | StitchKind::FlatFelled => vec![
            (s.offset, Pattern::Straight, true),
            (s.offset + s.width, Pattern::Straight, true),
        ],
        StitchKind::Zigzag | StitchKind::Bartack => vec![(s.offset, Pattern::Zigzag(s.width), true)],
        StitchKind::Overlock => vec![(0.6, Pattern::Overlock(s.width), false)],
        StitchKind::Coverstitch => vec![
            (s.offset, Pattern::Straight, true),
            (s.offset + s.width, Pattern::Straight, true),
            (s.offset, Pattern::Zigzag(s.width), false),
        ],
        StitchKind::Blindhem => vec![(s.offset, Pattern::Catch, true)],
        StitchKind::FlyJ => vec![(s.offset, Pattern::J, true)],
    }
}

#[derive(Debug, Clone, Copy)]
enum Pattern {
    Straight,
    Zigzag(f32),
    Overlock(f32),
    Catch,
    /// Straight, curving into the seam over its last few centimetres.
    J,
}

/// A polyline in pattern space with its inward normals (mm).
struct Line {
    points: Vec<Vec2>,
    inward: Vec<Vec2>,
    lengths: Vec<f32>,
}

impl Line {
    fn of_span(document: &PatternDocument, piece: PieceId, first: u32, count: u32) -> Option<Self> {
        let piece = document.piece(piece)?;
        let outline = &piece.outline;
        let edges = outline.edge_count();
        if edges == 0 {
            return None;
        }
        let mut points: Vec<Vec2> = Vec::new();
        for k in 0..count.max(1) {
            let curve = outline.curve((first + k) % edges);
            let flat = curve.flatten(0.3);
            let skip = usize::from(!points.is_empty());
            points.extend(flat.into_iter().skip(skip));
        }
        points.dedup_by(|a, b| a.distance(*b) < 1e-4);
        if points.len() < 2 {
            return None;
        }
        // Inside is to the left of a counter-clockwise outline.
        let sign = if outline.is_ccw() { 1.0 } else { -1.0 };
        let n = points.len();
        let inward = (0..n)
            .map(|i| {
                let a = points[i.saturating_sub(1)];
                let b = points[(i + 1).min(n - 1)];
                (b - a).perp().normalize_or_zero() * sign
            })
            .collect();
        let mut lengths = vec![0.0];
        for w in points.windows(2) {
            lengths.push(lengths.last().unwrap() + w[0].distance(w[1]));
        }
        Some(Self { points, inward, lengths })
    }

    fn length(&self) -> f32 {
        *self.lengths.last().unwrap_or(&0.0)
    }

    /// The point `s` mm along, `offset` mm inside.
    fn at(&self, s: f32, offset: f32) -> Vec2 {
        let s = s.clamp(0.0, self.length());
        let i = self.lengths.partition_point(|&l| l <= s).clamp(1, self.points.len() - 1);
        let (l0, l1) = (self.lengths[i - 1], self.lengths[i]);
        let t = if l1 > l0 { (s - l0) / (l1 - l0) } else { 0.0 };
        let p = self.points[i - 1].lerp(self.points[i], t);
        let n = self.inward[i - 1].lerp(self.inward[i], t).normalize_or_zero();
        p + n * offset
    }
}

impl Stitches {
    /// Pins every visible stitch of `document` to the garment.
    pub fn build(document: &PatternDocument, topology: &impl TopologySync) -> Self {
        let mut stitches = Self::default();
        let thickness = |piece: PieceId| {
            document
                .piece(piece)
                .and_then(|p| document.fabric(p.fabric))
                .map_or(0.6, |f| f.physics.thickness)
                // As the cloth is drawn (see `Session::fabric_thickness`).
                .max(0.8)
                * MM
        };
        let mut sew = |piece: PieceId, first: u32, count: u32, stitching: &Stitching, key: u64| {
            if !stitching.visible() {
                return;
            }
            let Some(line) = Line::of_span(document, piece, first, count) else {
                return;
            };
            let face = 0.5 * thickness(piece);
            for (r, (offset, pattern, outside)) in rows_of(stitching).into_iter().enumerate() {
                let row = key.wrapping_mul(31).wrapping_add(r as u64);
                stitches.row(topology, piece, &line, offset, pattern, outside, stitching, face, row);
            }
        };
        for seam in document.seams() {
            let (side, span) = match seam.stitching.side {
                StitchSide::A => (seam.a.piece, seam.a_span()),
                StitchSide::B => (seam.b.piece, seam.b_span()),
            };
            let key = (u64::from(seam.id.0) << 1) ^ seam.stitching.key();
            sew(side, span.first, span.count, &seam.stitching, key);
            // Inside, an overlock neatens both allowances together; on the
            // other side the seam shows only its own row.
        }
        // Zippers: teeth all along the seam, from both sides.
        let mut zipper_rows = Vec::new();
        for seam in document.seams() {
            let Some(zipper) = seam.zipper else { continue };
            let (a, b) = (seam.a_span(), seam.b_span());
            let (Some(la), Some(lb)) = (
                Line::of_span(document, seam.a.piece, a.first, a.count),
                Line::of_span(document, seam.b.piece, b.first, b.count),
            ) else {
                continue;
            };
            let pin = |piece: PieceId, p: Vec2| {
                topology
                    .pattern_to_surface(PatternLocation { piece, position: p })
                    .map(|l| Pin { triangle: l.triangle, barycentric: l.barycentric })
            };
            let pitch = match zipper.teeth {
                ZipperTeeth::Metal => 1.6,
                ZipperTeeth::Coil => 1.0,
            };
            let count = (la.length() / pitch).floor().max(1.0) as usize;
            let mut teeth = Vec::with_capacity(count);
            for k in 0..count {
                let along = (k as f32 + 0.5) / count as f32;
                let sa = along * la.length();
                let sb = if seam.reversed { (1.0 - along) * lb.length() } else { along * lb.length() };
                let pins = (
                    pin(seam.a.piece, la.at(sa, 0.4)),
                    pin(seam.a.piece, la.at(sa, 4.0)),
                    pin(seam.b.piece, lb.at(sb, 0.4)),
                    pin(seam.b.piece, lb.at(sb, 4.0)),
                );
                if let (Some(a), Some(a_in), Some(b), Some(b_in)) = pins {
                    teeth.push(Tooth { along, a, a_in, b, b_in });
                }
            }
            zipper_rows.push(ZipperRow {
                zipper,
                face: 0.5 * thickness(seam.a.piece),
                pitch: pitch * MM,
                teeth,
            });
        }
        for piece in document.pieces() {
            for (k, edge) in piece.edge_stitches.iter().enumerate() {
                let key = (u64::from(piece.id.0) << 33) ^ ((k as u64) << 20) ^ edge.stitching.key();
                sew(piece.id, edge.span.first, edge.span.count, &edge.stitching, key | 1);
            }
        }
        stitches.zippers = zipper_rows;
        log::debug!(
            "stitches: {} threads in {} rows",
            stitches.threads.len(),
            stitches.rows.len()
        );
        stitches
    }

    #[allow(clippy::too_many_arguments)]
    fn row(
        &mut self,
        topology: &impl TopologySync,
        piece: PieceId,
        line: &Line,
        offset: f32,
        pattern: Pattern,
        outside: bool,
        stitching: &Stitching,
        face: f32,
        row: u64,
    ) {
        let pin = |p: Vec2| {
            let location = topology.pattern_to_surface(PatternLocation { piece, position: p })?;
            Some(Pin {
                triangle: location.triangle,
                barycentric: location.barycentric,
            })
        };
        let length = line.length();
        let step = stitching.length.max(0.3);
        let count = (length / step).floor().max(1.0) as usize;
        let radius = 0.5 * stitching.thread_size.max(0.1) * MM;
        let mut needle = Vec::new();
        let add = |a: Vec2, b: Vec2, along: f32, threads: &mut Vec<Thread>| {
            // Long threads bend with the cloth: pinned every 2 mm.
            let pieces = ((a.distance(b) / 2.0).ceil() as usize).max(1);
            let pins: Option<Vec<Pin>> = (0..=pieces)
                .map(|k| pin(a.lerp(b, k as f32 / pieces as f32)))
                .collect();
            if let Some(pins) = pins {
                threads.push(Thread {
                    pins,
                    outside,
                    color: stitching.thread,
                    radius,
                    face,
                    row,
                    along,
                });
            }
        };
        let mut threads = Vec::new();
        // No machine sews perfectly: stitches a little longer or shorter,
        // the row wandering a fraction of a millimetre off its line.
        let jitter = |k: usize, salt: u64| {
            let mut x = row ^ (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt;
            x ^= x >> 33;
            x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
            x ^= x >> 33;
            (x & 0xffff) as f32 / 65535.0 - 0.5
        };
        let wander = |s: f32| {
            let t = s / 40.0;
            0.18 * ((t * 1.7 + (row % 97) as f32).sin() * 0.6 + (t * 4.3 + (row % 31) as f32).sin() * 0.4)
        };
        // A fly's J: the row swings round into the seam at the end.
        let curve = 35.0f32.min(0.4 * length);
        let taper = |s: f32| {
            if matches!(pattern, Pattern::J) && s > length - curve {
                let u = ((s - (length - curve)) / curve).clamp(0.0, 1.0);
                (1.0 - u * u).max(0.0).sqrt()
            } else {
                1.0
            }
        };
        let at = |s: f32, offset: f32| line.at(s, offset * taper(s) + wander(s));
        let mut ends: Vec<f32> = (0..=count).map(|k| k as f32 * step).collect();
        for (k, e) in ends.iter_mut().enumerate().take(count).skip(1) {
            *e += jitter(k, 1) * 0.16 * step;
        }
        for k in 0..count {
            let (s0, s1) = (ends[k], ends[k + 1]);
            let along = (k as f32 + 0.5) / count as f32;
            match pattern {
                Pattern::Straight | Pattern::J => {
                    // The thread dips into the needle holes at both ends.
                    let gap = 0.3;
                    add(at(s0 + gap * 0.5, offset), at(s1 - gap * 0.5, offset), along, &mut threads);
                }
                Pattern::Zigzag(width) => {
                    let side = |i: usize| if i % 2 == 0 { -0.5 } else { 0.5 } * width;
                    add(at(s0, offset + side(k)), at(s1, offset + side(k + 1)), along, &mut threads);
                }
                Pattern::Overlock(width) => {
                    // A loop over the allowances: across, then along the edge.
                    add(at(s0, offset), at(s0 + step * 0.5, offset + width), along, &mut threads);
                    add(at(s0 + step * 0.5, offset + width), at(s1, offset), along, &mut threads);
                    add(at(s0, offset), at(s1, offset), along, &mut threads);
                }
                Pattern::Catch => {
                    add(at(s0, offset), at(s0 + 0.6, offset), along, &mut threads);
                }
            }
            // Thread tension varies: some stitches lie tighter, some looser,
            // and catch the light a little differently.
            if let Some(last) = threads.last_mut().filter(|t: &&mut Thread| t.row == row) {
                let tension = jitter(k, 2);
                last.radius *= 1.0 + 0.25 * tension;
                last.color = last.color.map(|c| c * (1.0 + 0.12 * jitter(k, 3)));
            }
            if let Some(p) = pin(at(s1, offset)) {
                needle.push(((k as f32 + 1.0) / count as f32, p, face));
            }
        }
        self.threads.extend(threads);
        self.rows.push((row, needle));
    }

    /// Every row's key, for sewing them one after another.
    pub fn rows(&self) -> impl Iterator<Item = u64> + '_ {
        self.rows.iter().map(|(k, _)| *k)
    }

    pub fn is_empty(&self) -> bool {
        self.threads.is_empty() && self.zippers.is_empty()
    }

    /// The threads as they lie on `mesh` now, appended to `out` (props
    /// shading: colour per vertex, uv = roughness, metallic). `sewn(row)`
    /// says how much of each row is sewn yet (0..=1); where a row is being
    /// sewn its needle is returned.
    pub fn mesh(&self, mesh: &GarmentMesh, sewn: impl Fn(u64) -> f32, out: &mut MeshData) -> Vec<Needle> {
        let surface = |pin: &Pin| {
            let [a, b, c] = mesh.triangle(pin.triangle).map(|v| v as usize);
            let w = pin.barycentric;
            let flat = mesh.positions[a] * w.x + mesh.positions[b] * w.y + mesh.positions[c] * w.z;
            let n = (mesh.normals[a] * w.x + mesh.normals[b] * w.y + mesh.normals[c] * w.z)
                .normalize_or(Vec3::Z);
            // On the curved surface the studio render draws (Phong
            // tessellation, see `refine`), not the flat triangle under it:
            // over a fold the thread would otherwise sink into the cloth.
            let curved: Vec3 = [(a, w.x), (b, w.y), (c, w.z)]
                .into_iter()
                .map(|(i, wi)| (flat - mesh.normals[i] * (flat - mesh.positions[i]).dot(mesh.normals[i])) * wi)
                .sum();
            (flat.lerp(curved, 0.75), n)
        };
        for thread in &self.threads {
            if thread.along > sewn(thread.row) {
                continue;
            }
            let side = if thread.outside { 1.0 } else { -1.0 };
            let count = thread.pins.len();
            let centre: Vec<(Vec3, Vec3)> = thread
                .pins
                .iter()
                .enumerate()
                .map(|(k, pin)| {
                    let (p, n) = surface(pin);
                    let n = n * side;
                    // Lies on the face, arching a little between its holes,
                    // and sinks into them at the ends.
                    let t = k as f32 / (count - 1).max(1) as f32;
                    let arch = (std::f32::consts::PI * t).sin();
                    // Inside, the thread wraps the seam allowances, which
                    // stand off the cloth: it lies further in, out of
                    // sight from outside where the pieces meet at an angle.
                    let allowance = if thread.outside { 0.0 } else { 1.5 * MM };
                    let lift = thread.face + allowance + thread.radius * (0.2 + 0.9 * arch);
                    (p + n * lift, n)
                })
                .collect();
            tube(out, &centre, thread.radius, thread.color);
        }
        for row in &self.zippers {
            zipper_mesh(row, &surface, out);
        }
        let mut needles = Vec::new();
        for (row, pins) in &self.rows {
            let done = sewn(*row);
            if done <= 0.0 || done >= 1.0 {
                continue;
            }
            if let Some((_, pin, face)) = pins.iter().find(|(t, _, _)| *t >= done) {
                let (p, n) = surface(pin);
                needles.push(Needle {
                    position: p + n * face,
                    normal: n,
                });
            }
        }
        needles
    }
}

/// A box: `centre`, its axes (unit) and half sizes, in one colour.
fn add_box(out: &mut MeshData, centre: Vec3, axes: [Vec3; 3], half: Vec3, color: [f32; 3], material: Vec2) {
    for (axis, other) in [(0usize, [1usize, 2]), (1, [2, 0]), (2, [0, 1])] {
        for sign in [-1.0f32, 1.0] {
            let n = axes[axis] * sign;
            let (u, v) = (axes[other[0]] * half[other[0]], axes[other[1]] * half[other[1]]);
            let c = centre + n * half[axis];
            let base = out.vertices.len() as u32;
            for (du, dv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                out.vertices.push(MeshVertex::new(c + u * du + v * dv, n, material).with_color(color));
            }
            // Counter-clockwise from outside.
            if sign > 0.0 {
                out.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
            } else {
                out.indices.extend([base, base + 2, base + 1, base, base + 3, base + 2]);
            }
        }
    }
}

/// A zipper on the cloth now: tape either side, the teeth interlocked
/// where it is closed and apart on their own sides where open, and the
/// slider with its pull where the two meet.
fn zipper_mesh(row: &ZipperRow, surface: &impl Fn(&Pin) -> (Vec3, Vec3), out: &mut MeshData) {
    let z = &row.zipper;
    let open = z.open.clamp(0.0, 1.0);
    let side = if z.concealed { -1.0 } else { 1.0 };
    let metal = Vec2::new(0.3, 1.0);
    let cloth = Vec2::new(0.85, 0.0);
    let (tooth_color, tooth_material, tooth_half) = match z.teeth {
        ZipperTeeth::Metal => (z.metal, metal, Vec3::new(2.6 * MM, 0.55 * row.pitch, 0.7 * MM)),
        ZipperTeeth::Coil => (z.tape.map(|c| c * 0.8), cloth, Vec3::new(2.0 * MM, 0.4 * row.pitch, 0.45 * MM)),
    };
    let frame = |at: &Pin, inward: &Pin| {
        let (p, n) = surface(at);
        let n = n * side;
        let across = (surface(inward).0 - p).normalize_or(Vec3::X);
        let across = (across - n * across.dot(n)).normalize_or(Vec3::X);
        (p, n, across)
    };
    let mut slider = None;
    for (k, tooth) in row.teeth.iter().enumerate() {
        let (pa, na, across_a) = frame(&tooth.a, &tooth.a_in);
        let along = na.cross(across_a);
        let lift = row.face + 0.6 * MM;
        // Tape under the teeth, on both sides.
        for (p, n, across) in [(pa, na, across_a), frame(&tooth.b, &tooth.b_in)] {
            add_box(
                out,
                p + n * (row.face + 0.15 * MM) + across * 5.0 * MM,
                [across, n.cross(across), n],
                Vec3::new(4.5 * MM, 0.55 * row.pitch, 0.15 * MM),
                z.tape,
                cloth,
            );
        }
        if tooth.along >= open {
            // Closed: the teeth of both sides interlock across the seam.
            let (pb, _, _) = frame(&tooth.b, &tooth.b_in);
            let mid = 0.5 * (pa + pb);
            let shift = if k % 2 == 0 { 0.5 } else { -0.5 } * MM;
            add_box(out, mid + na * lift + across_a * shift, [across_a, along, na], tooth_half, tooth_color, tooth_material);
            if slider.is_none() {
                slider = Some((mid, na, across_a, along));
            }
        } else {
            // Open: each tooth on its own tape.
            let (p, n, across) = if k % 2 == 0 { (pa, na, across_a) } else { frame(&tooth.b, &tooth.b_in) };
            add_box(
                out,
                p + n * lift + across * 1.6 * MM,
                [across, n.cross(across), n],
                tooth_half,
                tooth_color,
                tooth_material,
            );
        }
    }
    // The slider sits where the teeth part (at the start when closed).
    let slider = slider.or_else(|| {
        row.teeth.last().map(|t| {
            let (p, n, across) = frame(&t.a, &t.a_in);
            (p, n, across, n.cross(across))
        })
    });
    if let Some((p, n, across, along)) = slider {
        let top = p + n * (row.face + 2.2 * MM);
        add_box(out, top, [across, along, n], Vec3::new(4.5 * MM, 5.0 * MM, 1.6 * MM), z.metal, metal);
        // The pull hangs down from it, towards the closed end.
        add_box(
            out,
            top + n * 1.8 * MM + along * 10.0 * MM,
            [across, along, n],
            Vec3::new(3.2 * MM, 9.0 * MM, 0.5 * MM),
            z.metal,
            metal,
        );
    }
}

/// A thread: a four-sided tube along `centre` (points and their up).
fn tube(out: &mut MeshData, centre: &[(Vec3, Vec3)], radius: f32, color: [f32; 3]) {
    if centre.len() < 2 {
        return;
    }
    let base = out.vertices.len() as u32;
    const SIDES: u32 = 4;
    let material = Vec2::new(0.55, 0.0);
    for (k, &(p, up)) in centre.iter().enumerate() {
        let next = centre[(k + 1).min(centre.len() - 1)].0;
        let previous = centre[k.saturating_sub(1)].0;
        let along = (next - previous).normalize_or(Vec3::X);
        let side = up.cross(along).normalize_or(Vec3::Y);
        let up = along.cross(side);
        // Thinner at the holes, where the thread is pulled tight.
        let end = k == 0 || k == centre.len() - 1;
        let r = if end { radius * 0.6 } else { radius };
        for s in 0..SIDES {
            let angle = (s as f32 + 0.5) / SIDES as f32 * std::f32::consts::TAU;
            let normal = side * angle.cos() + up * angle.sin();
            // Twisted ply: a little lighter on top.
            let shade = 0.85 + 0.15 * angle.sin().max(0.0);
            out.vertices.push(
                MeshVertex::new(p + normal * r, normal, material).with_color(color.map(|c| c * shade)),
            );
        }
    }
    for k in 0..(centre.len() as u32 - 1) {
        for s in 0..SIDES {
            let a = base + k * SIDES + s;
            let b = base + k * SIDES + (s + 1) % SIDES;
            let c = a + SIDES;
            let d = b + SIDES;
            // Counter-clockwise seen from outside: the two-sided shading
            // would otherwise turn the normals in, and the thread dark.
            out.indices.extend([a, b, c, b, d, c]);
        }
    }
}

/// A sewing machine needle at `needle`, `phase` (0..1) through a stitch:
/// down into the cloth and up again.
pub fn needle_model(out: &mut MeshData, needle: &Needle, phase: f32) {
    let n = needle.normal;
    let depth = (std::f32::consts::PI * phase).sin();
    let tip = needle.position + n * (0.5 * MM - 1.5 * MM * depth);
    let length = 40.0 * MM;
    let steel = [0.62, 0.63, 0.66];
    let material = Vec2::new(0.25, 1.0);
    let base = out.vertices.len() as u32;
    let side = n.any_orthonormal_vector();
    let up = n.cross(side);
    const SIDES: u32 = 8;
    for (k, (h, r)) in [(0.0, 0.05), (1.5, 0.35), (length / MM, 0.45)].iter().enumerate() {
        let _ = k;
        for s in 0..SIDES {
            let a = s as f32 / SIDES as f32 * std::f32::consts::TAU;
            let normal = side * a.cos() + up * a.sin();
            out.vertices.push(
                MeshVertex::new(tip + n * (h * MM) + normal * (r * MM), normal, material).with_color(steel),
            );
        }
    }
    for k in 0..2u32 {
        for s in 0..SIDES {
            let a = base + k * SIDES + s;
            let b = base + k * SIDES + (s + 1) % SIDES;
            out.indices.extend([a, b, a + SIDES, b, b + SIDES, a + SIDES]);
        }
    }
}

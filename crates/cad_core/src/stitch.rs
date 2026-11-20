//! How seams and edges are sewn: the stitch a sewing machine makes there,
//! with its thread, length and distance from the seam, as you would set
//! the machine.
//!
//! A seam is closed whatever its stitch; the stitch decides what shows:
//! nothing (a plain seam pressed open), rows of topstitching on the
//! outside, the two rows of a flat-felled seam, an overlocked edge inside.
//! Free edges (hems, necklines, pocket mouths) are stitched the same way,
//! see [`EdgeStitching`].

use serde::{Deserialize, Serialize};

use crate::lines::EdgeSpan;

/// What the machine sews.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum StitchKind {
    /// A plain seam: closed, nothing shows outside.
    #[default]
    Plain,
    /// One row of straight stitches on the outside, a little off the seam.
    Topstitch,
    /// One row right at the seam or edge (about 1.5 mm).
    Edgestitch,
    /// Two parallel rows (twin needle), as on jeans.
    DoubleTopstitch,
    /// The allowances folded into each other and stitched down: two rows
    /// on the outside, a flat ridge, nothing raw inside.
    FlatFelled,
    /// A zigzag, for stretch and for neatening.
    Zigzag,
    /// An overlock (serger) edge on the inside: loops of thread wrapped
    /// over the allowances.
    Overlock,
    /// A hem's coverstitch: two rows outside, the looper's net inside.
    Coverstitch,
    /// Short, dense zigzag that reinforces a point (pocket corners).
    Bartack,
    /// Hardly anything outside: a tiny catch every centimetre.
    Blindhem,
    /// The topstitching of a fly: a row at a distance from the seam that
    /// curves into it at the bottom, holding the zipper's facing (a J).
    FlyJ,
}

impl StitchKind {
    pub const ALL: [Self; 11] = [
        Self::Plain,
        Self::Topstitch,
        Self::Edgestitch,
        Self::DoubleTopstitch,
        Self::FlatFelled,
        Self::Zigzag,
        Self::Overlock,
        Self::Coverstitch,
        Self::Bartack,
        Self::Blindhem,
        Self::FlyJ,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Plain => "Plain seam",
            Self::Topstitch => "Topstitch",
            Self::Edgestitch => "Edgestitch",
            Self::DoubleTopstitch => "Double topstitch",
            Self::FlatFelled => "Flat-felled",
            Self::Zigzag => "Zigzag",
            Self::Overlock => "Overlock",
            Self::Coverstitch => "Coverstitch",
            Self::Bartack => "Bartack",
            Self::Blindhem => "Blind hem",
            Self::FlyJ => "Fly (J-stitch)",
        }
    }
}

/// Which piece of a seam the topstitching lies on (where the allowances
/// are pressed to).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum StitchSide {
    #[default]
    A,
    B,
}

/// How a seam or edge is sewn.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Stitching {
    pub kind: StitchKind,
    /// Thread colour, linear RGB.
    pub thread: [f32; 3],
    /// Length of one stitch, mm.
    pub length: f32,
    /// Distance of the (first) row from the seam or edge, mm.
    pub offset: f32,
    /// Zigzag and overlock width, or the gap between twin rows, mm.
    pub width: f32,
    /// Thread thickness, mm (0.25 for shirts, 0.6 for jeans topstitching).
    pub thread_size: f32,
    pub side: StitchSide,
}

impl Default for Stitching {
    fn default() -> Self {
        Self::of(StitchKind::Plain)
    }
}

impl Stitching {
    /// The machine's usual settings for a stitch.
    pub fn of(kind: StitchKind) -> Self {
        let (length, offset, width, thread_size) = match kind {
            StitchKind::Plain => (2.5, 0.0, 0.0, 0.25),
            StitchKind::Topstitch => (3.0, 6.0, 0.0, 0.5),
            StitchKind::Edgestitch => (2.5, 1.5, 0.0, 0.4),
            StitchKind::DoubleTopstitch => (3.2, 1.5, 6.4, 0.6),
            StitchKind::FlatFelled => (3.2, 1.5, 7.0, 0.6),
            StitchKind::Zigzag => (1.2, 3.0, 4.0, 0.25),
            StitchKind::Overlock => (1.0, 0.0, 4.0, 0.2),
            StitchKind::Coverstitch => (3.0, 8.0, 6.4, 0.3),
            StitchKind::Bartack => (0.4, 3.0, 2.5, 0.35),
            StitchKind::Blindhem => (10.0, 8.0, 0.0, 0.2),
            StitchKind::FlyJ => (3.2, 32.0, 0.0, 0.6),
        };
        Self {
            kind,
            thread: [0.02; 3],
            length,
            offset,
            width,
            thread_size,
            side: StitchSide::A,
        }
    }

    /// With this thread colour (linear RGB).
    pub fn with_thread(mut self, thread: [f32; 3]) -> Self {
        self.thread = thread;
        self
    }

    /// Whether any of it shows on the garment.
    pub fn visible(&self) -> bool {
        self.kind != StitchKind::Plain
    }

    /// A number that changes whenever how it is sewn changes.
    pub fn key(&self) -> u64 {
        let mut h = self.kind as u64 * 0x9e37_79b9;
        for v in self
            .thread
            .iter()
            .chain(&[self.length, self.offset, self.width, self.thread_size])
        {
            h = (h ^ u64::from(v.to_bits())).wrapping_mul(0x0100_0000_01b3);
        }
        h ^ (self.side as u64) << 40
    }
}

/// A free edge sewn on its own: a hem, a neckline, a pocket's mouth.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EdgeStitching {
    pub span: EdgeSpan,
    pub stitching: Stitching,
}

/// The teeth of a zipper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ZipperTeeth {
    /// Metal teeth, as on jeans.
    #[default]
    Metal,
    /// A nylon coil, as on skirts and jackets.
    Coil,
}

/// A seam that is a zipper: closed it holds like a seam; opened (from the
/// start of its first side) the cloth comes apart.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Zipper {
    pub teeth: ZipperTeeth,
    /// Linear RGB of the tape and of the teeth and slider.
    pub tape: [f32; 3],
    pub metal: [f32; 3],
    /// How far it is open, 0 (closed) ..= 1.
    pub open: f32,
    /// Hidden behind a facing (a fly): the teeth lie on the inside.
    pub concealed: bool,
}

impl Default for Zipper {
    fn default() -> Self {
        Self {
            teeth: ZipperTeeth::Metal,
            tape: [0.02, 0.025, 0.05],
            metal: [0.55, 0.45, 0.25],
            open: 0.0,
            concealed: false,
        }
    }
}

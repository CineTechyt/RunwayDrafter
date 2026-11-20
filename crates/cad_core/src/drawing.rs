//! Lines drawn on the avatar (a 3D pen), and which piece lines were
//! flattened from them.
//!
//! A line is a list of points on the avatar's skin, each a triangle of the
//! skin and barycentric weights in it. The avatar's shape and pose move the
//! triangles, and the lines go with them. Between its points a line runs
//! over the skin (see `avatar::drawing`).
//!
//! Lines meet where they share a point. The closed areas they enclose are
//! flattened into pattern pieces; every piece line remembers the stretch
//! of drawn line it came from, so a neighbouring area flattened later is
//! sewn to it.

use serde::{Deserialize, Serialize};

use crate::ids::PieceId;

/// A point on the avatar's skin.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SkinPoint {
    /// Index into the skin's triangles.
    pub triangle: u32,
    /// Barycentric weights of the triangle's corners.
    pub weights: [f32; 3],
}

impl SkinPoint {
    /// Whether two points are the same point of the skin (a shared corner
    /// of two lines).
    pub fn same(&self, other: &SkinPoint) -> bool {
        self.triangle == other.triangle
            && self
                .weights
                .iter()
                .zip(&other.weights)
                .all(|(a, b)| (a - b).abs() < 1e-4)
    }
}

/// A line drawn on the avatar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AvatarLine {
    pub id: u32,
    pub points: Vec<SkinPoint>,
    /// Runs on from the last point back to the first.
    #[serde(default)]
    pub closed: bool,
    /// Flattened as a straight line (a centre front or back).
    #[serde(default)]
    pub straight: bool,
    /// The line this one is the mirror image of (drawn with Mirror on):
    /// moving a point of either moves the other's too.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mirror_of: Option<u32>,
}

/// A piece line flattened from the stretch of drawn line between two of
/// its points.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlatEdge {
    pub line: u32,
    /// The stretch's ends, in the piece line's direction.
    pub from: SkinPoint,
    pub to: SkinPoint,
    pub piece: PieceId,
    /// Any edge of the piece line (see `Outline::line_span`).
    pub edge: u32,
}

/// Everything drawn on the avatar.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Drawing {
    #[serde(default)]
    pub lines: Vec<AvatarLine>,
    #[serde(default)]
    pub flat_edges: Vec<FlatEdge>,
    #[serde(default)]
    next_line: u32,
}

impl Drawing {
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty() && self.flat_edges.is_empty()
    }

    /// Adds a line; returns its id.
    pub fn add_line(&mut self, points: Vec<SkinPoint>, closed: bool) -> u32 {
        self.next_line = self
            .next_line
            .max(self.lines.iter().map(|l| l.id).max().unwrap_or(0))
            + 1;
        let id = self.next_line;
        self.lines.push(AvatarLine {
            id,
            points,
            closed,
            straight: false,
            mirror_of: None,
        });
        id
    }

    /// The mirror image of line `id`, either way round.
    pub fn twin(&self, id: u32) -> Option<u32> {
        let own = self.line(id)?;
        own.mirror_of.or_else(|| {
            self.lines
                .iter()
                .find(|l| l.mirror_of == Some(id))
                .map(|l| l.id)
        })
    }

    /// Every (line, point index) at the same skin point as `point`.
    pub fn at_point(&self, point: &SkinPoint) -> Vec<(u32, usize)> {
        self.lines
            .iter()
            .flat_map(|l| {
                l.points
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.same(point))
                    .map(move |(k, _)| (l.id, k))
            })
            .collect()
    }

    pub fn line(&self, id: u32) -> Option<&AvatarLine> {
        self.lines.iter().find(|l| l.id == id)
    }

    pub fn line_mut(&mut self, id: u32) -> Option<&mut AvatarLine> {
        self.lines.iter_mut().find(|l| l.id == id)
    }

    pub fn remove_line(&mut self, id: u32) {
        self.lines.retain(|l| l.id != id);
        for line in &mut self.lines {
            if line.mirror_of == Some(id) {
                line.mirror_of = None;
            }
        }
        self.flat_edges.retain(|e| e.line != id);
    }
}

//! A bounding volume hierarchy over axis-aligned boxes.
//!
//! The tree is built top-down with the binned surface area heuristic and
//! stored flat: the two children of a node are adjacent, and always come
//! after their parent, so a refit is one backwards sweep over the array.

use glam::Vec3A;

/// An axis-aligned box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3A,
    pub max: Vec3A,
}

impl Aabb {
    /// The box that contains nothing; the identity of [`Aabb::union`].
    pub const EMPTY: Self = Self {
        min: Vec3A::INFINITY,
        max: Vec3A::NEG_INFINITY,
    };

    pub fn from_points(points: impl IntoIterator<Item = Vec3A>) -> Self {
        points
            .into_iter()
            .fold(Self::EMPTY, |bounds, p| bounds.with_point(p))
    }

    #[inline]
    pub fn with_point(self, p: Vec3A) -> Self {
        Self {
            min: self.min.min(p),
            max: self.max.max(p),
        }
    }

    #[inline]
    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Grown by `margin` on every side.
    #[inline]
    pub fn expanded(self, margin: f32) -> Self {
        Self {
            min: self.min - Vec3A::splat(margin),
            max: self.max + Vec3A::splat(margin),
        }
    }

    #[inline]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.min.cmple(other.max).all() && other.min.cmple(self.max).all()
    }

    #[inline]
    pub fn centre(&self) -> Vec3A {
        (self.min + self.max) * 0.5
    }

    /// Half the surface area; all the heuristic needs.
    #[inline]
    fn half_area(&self) -> f32 {
        let d = (self.max - self.min).max(Vec3A::ZERO);
        d.x * d.y + d.y * d.z + d.z * d.x
    }

    /// Squared distance from `p` to the box (zero inside).
    #[inline]
    pub fn distance_squared(&self, p: Vec3A) -> f32 {
        (self.min - p)
            .max(p - self.max)
            .max(Vec3A::ZERO)
            .length_squared()
    }

    /// Where the ray `origin + t * direction` enters the box, if it does
    /// within `0..=max_t`. `inverse` is `1 / direction` per component.
    #[inline]
    pub fn ray_entry(&self, origin: Vec3A, inverse: Vec3A, max_t: f32) -> Option<f32> {
        let t0 = (self.min - origin) * inverse;
        let t1 = (self.max - origin) * inverse;
        let near = t0.min(t1).max_element().max(0.0);
        let far = t0.max(t1).min_element().min(max_t);
        (near <= far).then_some(near)
    }
}

#[derive(Debug, Clone, Copy)]
struct Node {
    bounds: Aabb,
    /// Leaf: first entry in `order`. Inner node: index of the left child;
    /// the right child follows it.
    first: u32,
    /// Number of primitives; zero for inner nodes.
    count: u32,
}

/// Primitives per leaf below which splitting stops.
const LEAF_SIZE: u32 = 4;
const BINS: usize = 12;

#[derive(Debug, Clone, Default)]
pub struct Bvh {
    nodes: Vec<Node>,
    /// Primitive indices, grouped by leaf.
    order: Vec<u32>,
}

impl Bvh {
    /// Builds the hierarchy over `boxes`; primitive `i` is `boxes[i]`.
    pub fn build(boxes: &[Aabb]) -> Self {
        let mut bvh = Self {
            nodes: Vec::with_capacity(boxes.len() / 2 + 1),
            order: (0..boxes.len() as u32).collect(),
        };
        if boxes.is_empty() {
            return bvh;
        }
        let centres: Vec<Vec3A> = boxes.iter().map(Aabb::centre).collect();
        bvh.nodes.push(Node {
            bounds: Aabb::EMPTY,
            first: 0,
            count: boxes.len() as u32,
        });
        let mut stack = vec![0usize];
        while let Some(index) = stack.pop() {
            let Node { first, count, .. } = bvh.nodes[index];
            let range = first as usize..(first + count) as usize;
            let bounds = bvh.order[range.clone()]
                .iter()
                .fold(Aabb::EMPTY, |b, &i| b.union(boxes[i as usize]));
            bvh.nodes[index].bounds = bounds;
            if count <= LEAF_SIZE {
                continue;
            }
            let Some(middle) = Self::split(&mut bvh.order[range], &centres, boxes, bounds) else {
                continue;
            };
            let left = bvh.nodes.len();
            bvh.nodes.push(Node {
                bounds: Aabb::EMPTY,
                first,
                count: middle,
            });
            bvh.nodes.push(Node {
                bounds: Aabb::EMPTY,
                first: first + middle,
                count: count - middle,
            });
            bvh.nodes[index].first = left as u32;
            bvh.nodes[index].count = 0;
            stack.extend([left, left + 1]);
        }
        bvh
    }

    /// Partitions `order` in place by the cheapest of `BINS - 1` planes
    /// along the longest axis of the centres. Returns the size of the left
    /// part, or `None` if a leaf is cheaper than any split.
    fn split(order: &mut [u32], centres: &[Vec3A], boxes: &[Aabb], bounds: Aabb) -> Option<u32> {
        let centre_bounds = Aabb::from_points(order.iter().map(|&i| centres[i as usize]));
        let extent = centre_bounds.max - centre_bounds.min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        if extent[axis] <= 1e-9 {
            // All centres coincide: nothing separates them, cut in half.
            return (order.len() as u32 > 4 * LEAF_SIZE).then_some(order.len() as u32 / 2);
        }
        let scale = BINS as f32 / extent[axis];
        let bin_of = |i: u32| {
            (((centres[i as usize][axis] - centre_bounds.min[axis]) * scale) as usize).min(BINS - 1)
        };

        let mut bins = [(Aabb::EMPTY, 0u32); BINS];
        for &i in order.iter() {
            let bin = &mut bins[bin_of(i)];
            bin.0 = bin.0.union(boxes[i as usize]);
            bin.1 += 1;
        }
        // Cost of cutting after bin k: area × count on either side.
        let mut right_cost = [0.0f32; BINS];
        let mut acc = (Aabb::EMPTY, 0u32);
        for k in (1..BINS).rev() {
            acc = (acc.0.union(bins[k].0), acc.1 + bins[k].1);
            right_cost[k] = acc.0.half_area() * acc.1 as f32;
        }
        let mut best = (f32::INFINITY, 0usize);
        let mut acc = (Aabb::EMPTY, 0u32);
        for k in 0..BINS - 1 {
            acc = (acc.0.union(bins[k].0), acc.1 + bins[k].1);
            let cost = acc.0.half_area() * acc.1 as f32 + right_cost[k + 1];
            if acc.1 > 0 && cost < best.0 {
                best = (cost, k);
            }
        }
        // A leaf costs area × count; do not split if that is cheaper,
        // unless the leaf would be unreasonably large.
        let leaf_cost = bounds.half_area() * order.len() as f32;
        if best.0 >= leaf_cost && order.len() as u32 <= 4 * LEAF_SIZE {
            return None;
        }
        let mut middle = 0;
        for k in 0..order.len() {
            if bin_of(order[k]) <= best.1 {
                order.swap(k, middle);
                middle += 1;
            }
        }
        (middle > 0 && middle < order.len()).then_some(middle as u32)
    }

    /// Updates the node boxes for moved primitives, keeping the tree. Much
    /// cheaper than a rebuild; the tree degrades only if primitives move
    /// far relative to each other.
    pub fn refit(&mut self, boxes: &[Aabb]) {
        for index in (0..self.nodes.len()).rev() {
            let node = self.nodes[index];
            self.nodes[index].bounds = if node.count > 0 {
                self.order[node.first as usize..(node.first + node.count) as usize]
                    .iter()
                    .fold(Aabb::EMPTY, |b, &i| b.union(boxes[i as usize]))
            } else {
                self.nodes[node.first as usize]
                    .bounds
                    .union(self.nodes[node.first as usize + 1].bounds)
            };
        }
    }

    pub fn primitive_count(&self) -> usize {
        self.order.len()
    }

    pub fn bounds(&self) -> Aabb {
        self.nodes.first().map_or(Aabb::EMPTY, |node| node.bounds)
    }

    /// Calls `visit` for every primitive whose leaf box overlaps `query`.
    /// The primitive's own box may still miss it.
    pub fn query(&self, query: &Aabb, mut visit: impl FnMut(u32)) {
        if self.nodes.is_empty() {
            return;
        }
        let mut stack = [0u32; 64];
        let mut top = 1;
        while top > 0 {
            top -= 1;
            let node = &self.nodes[stack[top] as usize];
            if !node.bounds.overlaps(query) {
                continue;
            }
            if node.count > 0 {
                self.order[node.first as usize..(node.first + node.count) as usize]
                    .iter()
                    .for_each(|&i| visit(i));
            } else if top + 2 <= stack.len() {
                stack[top] = node.first;
                stack[top + 1] = node.first + 1;
                top += 2;
            }
        }
    }

    /// The first primitive hit by the ray within `max_t`. `hit` tests one
    /// primitive against the ray limited to the given distance and returns
    /// where it is hit. Children are visited near to far, so most of the
    /// tree is skipped once something was hit.
    pub fn raycast(
        &self,
        origin: Vec3A,
        direction: Vec3A,
        max_t: f32,
        mut hit: impl FnMut(u32, f32) -> Option<f32>,
    ) -> Option<(u32, f32)> {
        if self.nodes.is_empty() {
            return None;
        }
        let inverse = direction.recip();
        let mut best: Option<(u32, f32)> = None;
        let mut limit = max_t;
        let mut stack = [0u32; 64];
        let mut top = 1;
        while top > 0 {
            top -= 1;
            let node = &self.nodes[stack[top] as usize];
            if node.bounds.ray_entry(origin, inverse, limit).is_none() {
                continue;
            }
            if node.count > 0 {
                for &i in &self.order[node.first as usize..(node.first + node.count) as usize] {
                    if let Some(t) = hit(i, limit)
                        && t <= limit
                    {
                        limit = t;
                        best = Some((i, t));
                    }
                }
            } else if top + 2 <= stack.len() {
                let entry = |child: u32| {
                    self.nodes[child as usize]
                        .bounds
                        .ray_entry(origin, inverse, limit)
                };
                let (left, right) = (node.first, node.first + 1);
                // Push the farther child first so the nearer is popped first.
                let (near, far) = if entry(left).unwrap_or(f32::INFINITY)
                    <= entry(right).unwrap_or(f32::INFINITY)
                {
                    (left, right)
                } else {
                    (right, left)
                };
                stack[top] = far;
                stack[top + 1] = near;
                top += 2;
            }
        }
        best
    }

    /// The primitive closest to `p` within `max_distance`.
    /// `distance_squared` measures one primitive.
    pub fn closest(
        &self,
        p: Vec3A,
        max_distance: f32,
        mut distance_squared: impl FnMut(u32) -> f32,
    ) -> Option<(u32, f32)> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best: Option<(u32, f32)> = None;
        let mut limit = max_distance * max_distance;
        let mut stack = [0u32; 64];
        let mut top = 1;
        while top > 0 {
            top -= 1;
            let node = &self.nodes[stack[top] as usize];
            if node.bounds.distance_squared(p) > limit {
                continue;
            }
            if node.count > 0 {
                for &i in &self.order[node.first as usize..(node.first + node.count) as usize] {
                    let d2 = distance_squared(i);
                    if d2 <= limit {
                        limit = d2;
                        best = Some((i, d2));
                    }
                }
            } else if top + 2 <= stack.len() {
                let (left, right) = (node.first, node.first + 1);
                let near_left = self.nodes[left as usize].bounds.distance_squared(p)
                    <= self.nodes[right as usize].bounds.distance_squared(p);
                let (near, far) = if near_left {
                    (left, right)
                } else {
                    (right, left)
                };
                stack[top] = far;
                stack[top + 1] = near;
                top += 2;
            }
        }
        best.map(|(i, d2)| (i, d2.sqrt()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random numbers in 0..1.
    fn noise(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (*seed >> 8) as f32 / (1u32 << 24) as f32
    }

    fn random_boxes(count: usize) -> Vec<Aabb> {
        let mut seed = 7;
        (0..count)
            .map(|_| {
                let centre = Vec3A::new(noise(&mut seed), noise(&mut seed), noise(&mut seed));
                Aabb {
                    min: centre,
                    max: centre,
                }
                .expanded(0.01 + 0.02 * noise(&mut seed))
            })
            .collect()
    }

    #[test]
    fn queries_find_exactly_the_overlapping_boxes() {
        let boxes = random_boxes(2000);
        let bvh = Bvh::build(&boxes);
        assert_eq!(bvh.primitive_count(), boxes.len());
        let mut seed = 99;
        for _ in 0..50 {
            let centre = Vec3A::new(noise(&mut seed), noise(&mut seed), noise(&mut seed));
            let query = Aabb {
                min: centre,
                max: centre,
            }
            .expanded(0.05);
            let mut found = Vec::new();
            bvh.query(&query, |i| {
                if boxes[i as usize].overlaps(&query) {
                    found.push(i);
                }
            });
            found.sort_unstable();
            let expected: Vec<u32> = (0..boxes.len() as u32)
                .filter(|&i| boxes[i as usize].overlaps(&query))
                .collect();
            assert_eq!(found, expected);
        }
    }

    #[test]
    fn the_tree_is_shallow_enough_to_prune() {
        let boxes = random_boxes(4000);
        let bvh = Bvh::build(&boxes);
        // A point query must touch a small fraction of the primitives.
        let query = Aabb {
            min: Vec3A::splat(0.5),
            max: Vec3A::splat(0.5),
        };
        let mut visited = 0;
        bvh.query(&query, |_| visited += 1);
        assert!(visited < 200, "visited {visited} of {}", boxes.len());
    }

    #[test]
    fn refit_follows_moved_boxes() {
        let mut boxes = random_boxes(500);
        let mut bvh = Bvh::build(&boxes);
        for b in &mut boxes {
            b.min += Vec3A::new(3.0, 0.0, 0.0);
            b.max += Vec3A::new(3.0, 0.0, 0.0);
        }
        bvh.refit(&boxes);
        let all = bvh.bounds();
        assert!(all.min.x > 2.9);
        let mut count = 0;
        bvh.query(&all, |_| count += 1);
        assert_eq!(count, boxes.len());
    }

    #[test]
    fn closest_matches_brute_force() {
        let boxes = random_boxes(1000);
        let bvh = Bvh::build(&boxes);
        let p = Vec3A::new(0.3, 0.6, 0.2);
        let (found, distance) = bvh
            .closest(p, 10.0, |i| boxes[i as usize].distance_squared(p))
            .unwrap();
        let brute = boxes
            .iter()
            .map(|b| b.distance_squared(p))
            .fold(f32::INFINITY, f32::min)
            .sqrt();
        assert!((distance - brute).abs() < 1e-6);
        assert!((boxes[found as usize].distance_squared(p).sqrt() - brute).abs() < 1e-6);
    }
}

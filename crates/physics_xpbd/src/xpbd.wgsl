// XPBD cloth solver, compute backend.
//
// One sub-step is: `integrate` over all particles, then `solve` once per
// colour (elements of a colour share no particle, so every invocation
// writes disjoint particles), then `finalize` over all particles.
// The maths mirrors `cpu.rs` line for line.

const INVALID: u32 = 0xffffffffu;
const ROW_ACTIVE: u32 = 1u;
const ROW_SEAM: u32 = 2u;
const MAX_COLLIDERS: u32 = 32u;
// Particles this close to the collision margin count as resting on the body.
const CONTACT_BAND: f32 = 0.003;

// Per sub-step parameters; one slot per sub-step, selected by dynamic offset.
struct Frame {
    gravity: vec3<f32>,
    h: f32,
    damping: f32,        // velocity multiplier, exp(-damping * h)
    inv_h2: f32,
    seam_gap: f32,       // remaining fraction of the initial seam gaps
    friction: f32,
    margin: f32,
    ground: f32,
    max_speed: f32,
    _pad0: f32,
    particle_count: u32,
    collider_count: u32,
    surface_collision: u32, // also collide triangle centres
    _pad1: u32,
};

struct Collider {
    center: vec4<f32>,
    radii: vec4<f32>,
    rotation: vec4<f32>, // quaternion xyzw
};

struct Colliders {
    items: array<Collider, MAX_COLLIDERS>,
};

struct BodyField {
    origin: vec3<f32>,
    voxel: f32,
    dims: vec3<u32>,
    enabled: u32,
};

// The element range of the colour being solved; one slot per colour.
struct ColourRange {
    first: u32,
    count: u32,
    _pad0: u32,
    _pad1: u32,
};

// C = |sum(weights[k] * x[k])| - rest
struct Row {
    weights: vec4<f32>,
    rest: f32,
    compliance: f32,
    flags: u32,
    fold_limit: f32, // bending rows stiffen beyond this length; 0 = never
};

struct Element {
    particles: vec4<u32>,
    rows: array<Row, 4>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> colliders: Colliders;
@group(0) @binding(2) var<uniform> colour: ColourRange;
// Signed distance to the avatar's skin on a regular grid (metres).
@group(0) @binding(3) var body_field: texture_3d<f32>;
@group(0) @binding(4) var<uniform> body: BodyField;

// xyz = position, w = inverse mass (0 pins the particle)
@group(1) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
// Kept in one buffer: the downlevel limit is four storage buffers per stage.
struct Motion {
    previous: vec4<f32>, // position at the start of the sub-step
    velocity: vec4<f32>,
};
@group(1) @binding(1) var<storage, read_write> motion: array<Motion>;
@group(1) @binding(2) var<storage, read> elements: array<Element>;
// Cloth-against-cloth contacts found on the CPU once per step: particle i
// must satisfy dot(n, x) + d >= 0 for planes[2i] and planes[2i + 1]
// (xyz = n, w = d; all zero = unused).
@group(1) @binding(3) var<storage, read> planes: array<vec4<f32>>;

// ---------------------------------------------------------------------------
// Pass 1: predict positions from velocities and gravity.
// ---------------------------------------------------------------------------
@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= frame.particle_count {
        return;
    }
    let p = positions[i];
    motion[i].previous = p;
    if p.w <= 0.0 {
        return;
    }
    var v = (motion[i].velocity.xyz + frame.gravity * frame.h) * frame.damping;
    let speed = length(v);
    if speed > frame.max_speed {
        v *= frame.max_speed / speed;
    }
    motion[i].velocity = vec4<f32>(v, 0.0);
    positions[i] = vec4<f32>(p.xyz + v * frame.h, p.w);
}

// ---------------------------------------------------------------------------
// Pass 2 (once per colour): constraint projection.
//
//   dl = -C / (sum(w_k^2 / m_k) + compliance / h^2)
//   dx_k = (w_k / m_k) * n * dl,   n = L / |L|,   L = sum(w_k * x_k)
// ---------------------------------------------------------------------------
@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= colour.count {
        return;
    }
    let element = elements[colour.first + id.x];

    var x: array<vec3<f32>, 4>;
    var w: array<f32, 4>;
    for (var k = 0u; k < 4u; k++) {
        let particle = element.particles[k];
        if particle != INVALID {
            let p = positions[particle];
            x[k] = p.xyz;
            w[k] = p.w;
        } else {
            x[k] = vec3<f32>(0.0);
            w[k] = 0.0;
        }
    }

    for (var r = 0u; r < 4u; r++) {
        let row = element.rows[r];
        if (row.flags & ROW_ACTIVE) == 0u {
            continue;
        }
        let c = row.weights;
        let combined = x[0] * c[0] + x[1] * c[1] + x[2] * c[2] + x[3] * c[3];
        let len = length(combined);
        if len < 1e-9 {
            continue;
        }
        var rest = row.rest;
        if (row.flags & ROW_SEAM) != 0u {
            rest *= frame.seam_gap;
        }
        var compliance = row.compliance;
        if row.fold_limit > 0.0 && len > row.fold_limit {
            let ratio = row.fold_limit / len;
            compliance *= ratio * ratio * ratio * ratio;
        }
        let denominator = w[0] * c[0] * c[0] + w[1] * c[1] * c[1] + w[2] * c[2] * c[2] + w[3] * c[3] * c[3]
            + compliance * frame.inv_h2;
        if denominator < 1e-12 {
            continue;
        }
        let correction = combined * (-(len - rest) / (denominator * len));
        for (var k = 0u; k < 4u; k++) {
            x[k] += correction * (w[k] * c[k]);
        }
    }

    // Cloth triangles also collide at their centre, or body details smaller
    // than a triangle would poke through between the vertices. Moving all
    // three vertices moves the centre equally.
    if frame.surface_collision != 0u && body.enabled != 0u && element.particles[2] != INVALID && element.particles[3] == INVALID {
        let sample = body_distance((x[0] + x[1] + x[2]) / 3.0);
        if sample.w < frame.margin && dot(sample.xyz, sample.xyz) > 1e-12 {
            let push = normalize(sample.xyz) * (frame.margin - sample.w);
            for (var k = 0u; k < 3u; k++) {
                if w[k] > 0.0 {
                    x[k] += push;
                }
            }
        }
    }

    for (var k = 0u; k < 4u; k++) {
        let particle = element.particles[k];
        if particle != INVALID {
            positions[particle] = vec4<f32>(x[k], w[k]);
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 3: collisions with friction, then velocity update.
// ---------------------------------------------------------------------------
fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

// Position-based Coulomb friction: the push-out distance stands in for the
// normal force, so sliding is reduced by at most friction * depth.
fn apply_friction(x: vec3<f32>, corrected: vec3<f32>, normal: vec3<f32>, start: vec3<f32>) -> vec3<f32> {
    let depth = length(corrected - x);
    let motion = corrected - start;
    let sliding = motion - normal * dot(motion, normal);
    let distance = length(sliding);
    if distance < 1e-9 {
        return corrected;
    }
    return corrected - sliding * min(frame.friction * depth / distance, 1.0);
}

fn field_at(cell: vec3<i32>) -> f32 {
    return textureLoad(body_field, cell, 0).r;
}

// Trilinear sample of the body field: xyz = gradient (points away from the
// body), w = distance. Outside the grid the distance is huge.
fn body_distance(p: vec3<f32>) -> vec4<f32> {
    let g = (p - body.origin) / body.voxel;
    let limit = vec3<f32>(body.dims) - vec3<f32>(1.0);
    if body.enabled == 0u || any(g < vec3<f32>(0.0)) || any(g >= limit) {
        return vec4<f32>(0.0, 0.0, 0.0, 1e9);
    }
    let cell = vec3<i32>(floor(g));
    let f = g - floor(g);
    let c000 = field_at(cell);
    let c100 = field_at(cell + vec3<i32>(1, 0, 0));
    let c010 = field_at(cell + vec3<i32>(0, 1, 0));
    let c110 = field_at(cell + vec3<i32>(1, 1, 0));
    let c001 = field_at(cell + vec3<i32>(0, 0, 1));
    let c101 = field_at(cell + vec3<i32>(1, 0, 1));
    let c011 = field_at(cell + vec3<i32>(0, 1, 1));
    let c111 = field_at(cell + vec3<i32>(1, 1, 1));
    let x00 = mix(c000, c100, f.x);
    let x10 = mix(c010, c110, f.x);
    let x01 = mix(c001, c101, f.x);
    let x11 = mix(c011, c111, f.x);
    let y0 = mix(x00, x10, f.y);
    let y1 = mix(x01, x11, f.y);
    let dx = mix(mix(c100 - c000, c110 - c010, f.y), mix(c101 - c001, c111 - c011, f.y), f.z);
    let dy = mix(x10 - x00, x11 - x01, f.z);
    let dz = y1 - y0;
    return vec4<f32>(vec3<f32>(dx, dy, dz) / body.voxel, mix(y0, y1, f.z));
}

@compute @workgroup_size(64)
fn finalize(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= frame.particle_count {
        return;
    }
    let p = positions[i];
    if p.w <= 0.0 {
        return;
    }
    let start = motion[i].previous.xyz;
    var x = p.xyz;

    // Other cloth first, so that the body has the last word.
    var cloth_normals: array<vec3<f32>, 2>;
    for (var k = 0u; k < 2u; k++) {
        cloth_normals[k] = vec3<f32>(0.0);
        let plane = planes[2u * i + k];
        if dot(plane.xyz, plane.xyz) == 0.0 {
            continue;
        }
        let distance = dot(plane.xyz, x) + plane.w;
        if distance < 0.0 {
            x = apply_friction(x, x - plane.xyz * distance, plane.xyz, start);
        }
        if distance < CONTACT_BAND {
            cloth_normals[k] = plane.xyz;
        }
    }

    for (var c = 0u; c < frame.collider_count; c++) {
        let collider = colliders.items[c];
        let radii = collider.radii.xyz + vec3<f32>(frame.margin);
        let inverse_rotation = vec4<f32>(-collider.rotation.xyz, collider.rotation.w);
        let scaled = rotate(inverse_rotation, x - collider.center.xyz) / radii;
        let d2 = dot(scaled, scaled);
        if d2 < 1.0 && d2 >= 1e-12 {
            let on_sphere = scaled / sqrt(d2);
            let normal = rotate(collider.rotation, normalize(on_sphere / radii));
            let corrected = collider.center.xyz + rotate(collider.rotation, on_sphere * radii);
            x = apply_friction(x, corrected, normal, start);
        }
    }

    let sample = body_distance(x);
    var resting = false;
    var contact_normal = vec3<f32>(0.0);
    if sample.w < frame.margin + CONTACT_BAND && dot(sample.xyz, sample.xyz) > 1e-12 {
        contact_normal = normalize(sample.xyz);
        resting = true;
        if sample.w < frame.margin {
            x = apply_friction(x, x + contact_normal * (frame.margin - sample.w), contact_normal, start);
        }
    }

    let floor_height = frame.ground + frame.margin;
    if x.y < floor_height {
        x = apply_friction(x, vec3<f32>(x.x, floor_height, x.z), vec3<f32>(0.0, 1.0, 0.0), start);
    }

    positions[i] = vec4<f32>(x, p.w);
    var v = (x - start) / frame.h;
    let speed = length(v);
    if speed > frame.max_speed {
        v *= frame.max_speed / speed;
    }
    // Contact is inelastic: being pushed out of the body must not become
    // outward speed, or cloth under tension never comes to rest.
    for (var k = 0u; k < 2u; k++) {
        v -= cloth_normals[k] * max(dot(v, cloth_normals[k]), 0.0);
    }
    if resting {
        v -= contact_normal * max(dot(v, contact_normal), 0.0);
    }
    motion[i].velocity = vec4<f32>(v, 0.0);
}

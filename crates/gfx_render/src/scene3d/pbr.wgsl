// Metallic-roughness PBR (Cook-Torrance, GGX) with one directional key light,
// a fill light and hemispherical ambient. Shared by the avatar and the fabric.

struct Globals {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    key_direction: vec4<f32>,   // points from the surface towards the light
    key_color: vec4<f32>,
    fill_direction: vec4<f32>,
    fill_color: vec4<f32>,
    sky: vec4<f32>,
    ground: vec4<f32>,
};

struct Object {
    model: mat4x4<f32>,
    base_color: vec4<f32>,
    // x: roughness, y: metallic, z: pattern strength, w: pattern id + wash/10
    params: vec4<f32>,
    emissive: vec4<f32>,
    // Technical views of the garment. x: 0 = fabric, 1 = strain heatmap,
    // 2 = fit map; y: the value shown blue; z: the value shown red.
    overlay: vec4<f32>,
    // x: 0 = standard, 1 = skin, 2 = hair (yzw: the crown, where strands
    // start), 3 = prop (material per vertex: colour, uv = roughness/metallic).
    shading: vec4<f32>,
    // x: procedural repeat scale, y: relief strength.
    fabric: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> object: Object;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // Strain (relative edge elongation) or clearance to the body (metres).
    @location(3) value: f32,
    @location(4) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) value: f32,
    @location(4) color: vec3<f32>,
};

fn pattern_height(uv: vec2<f32>, kind: f32, scale: f32, relief: f32) -> f32 {
    let s = max(scale, 0.1);
    if kind == 3.0 {
        let phase = uv.x * 18.0 * s * 2.0 * PI;
        let ridge = 0.5 + 0.5 * cos(phase);
        return ridge * ridge * (3.0 - 2.0 * ridge) * 0.006 * relief;
    }
    if kind == 1.0 {
        let phase = (uv.x + uv.y) * 36.0 * s * 2.0 * PI;
        return (0.5 + 0.5 * cos(phase)) * 0.001 * relief;
    }
    return 0.0;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // The relief of a weave is far finer than the mesh: it is shaded in the
    // fragment stage only. Moving vertices by it would only tear the edges.
    let world = object.model * vec4<f32>(in.position, 1.0);
    out.world_position = world.xyz;
    // Models are only translated/rotated/uniformly scaled, so the model
    // matrix is a valid normal matrix.
    out.world_normal = (object.model * vec4<f32>(in.normal, 0.0)).xyz;
    out.uv = in.uv;
    out.value = in.value;
    out.color = in.color;
    out.clip_position = globals.view_proj * world;
    return out;
}

const PI: f32 = 3.14159265359;

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = r * r / 8.0;
    let gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return gv * gl;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn shade_light(
    n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, radiance: vec3<f32>,
    albedo: vec3<f32>, roughness: f32, metallic: f32,
) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_h = max(dot(n, h), 0.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
    let specular = distribution_ggx(n_dot_h, roughness) * geometry_smith(n_dot_v, n_dot_l, roughness) * f
        / (4.0 * n_dot_v * n_dot_l + 1e-4);
    let diffuse = (1.0 - f) * (1.0 - metallic) * albedo / PI;
    return (diffuse + specular) * radiance * n_dot_l;
}

// Procedural plain weave driven by the pattern UVs: U runs along the weft,
// V along the warp (the grainline). Fades out before it can alias.
fn weave(uv: vec2<f32>, threads: f32) -> f32 {
    let p = uv * threads;
    let footprint = max(fwidth(p.x), fwidth(p.y));
    let fade = clamp(1.0 - footprint * 1.5, 0.0, 1.0);
    let pattern = sin(p.x * 2.0 * PI) * sin(p.y * 2.0 * PI);
    return 1.0 + (pattern * 0.5 - 0.5) * fade;
}

fn denim_fiber_mask(uv: vec2<f32>) -> f32 {
    let phase = (uv.x + uv.y) * 36.0 * 2.0 * PI;
    let footprint = fwidth(phase);
    let visibility = 1.0 - smoothstep(1.5, 7.0, footprint);
    let fiber = 0.5 + 0.5 * cos(phase);
    return smoothstep(0.78, 0.98, fiber) * visibility;
}

fn fabric_surface(uv: vec2<f32>, kind: f32, wash: f32, scale: f32) -> vec3<f32> {
    // Denim is a blue twill with pale warp fibres crossing the darker weft.
    // The narrow lines are deliberately irregular so it reads as yarn rather
    // than a regular checkerboard when the garment is viewed close-up.
    let pale_fiber = denim_fiber_mask(uv);
    let macro_twill = 0.80 + 0.40 * (0.5 + 0.5 * cos((uv.x + uv.y) * 5.5 * 2.0 * PI));
    let denim = vec3<f32>(macro_twill) * mix(vec3<f32>(0.88), vec3<f32>(1.35), pale_fiber);
    let leather = 0.82 + 0.28 * noise3(vec3<f32>(uv * 90.0, 0.0));
    let cord_phase = uv.x * 18.0 * max(scale, 0.1) * 2.0 * PI;
    let cord_visibility = 1.0 - smoothstep(1.5, 7.0, fwidth(cord_phase));
    let cord = 0.92 + 0.14 * (0.5 + 0.5 * cos(cord_phase)) * cord_visibility;
    let wool = 0.9 + 0.18 * noise3(vec3<f32>(uv * 180.0, 0.0));
    let knit = 0.88 + 0.12 * sin(uv.x * 150.0 * PI) * sin(uv.y * 150.0 * PI);
    let satin = 1.0 + 0.04 * sin(uv.y * 80.0 * PI);
    var value = vec3<f32>(1.0);
    if kind == 1.0 { value = denim; }
    if kind == 2.0 { value = vec3<f32>(leather); }
    if kind == 3.0 { value = vec3<f32>(cord); }
    if kind == 4.0 { value = vec3<f32>(wool); }
    if kind == 5.0 { value = vec3<f32>(knit); }
    if kind == 6.0 { value = vec3<f32>(satin); }
    let irregular = 0.88 + 0.24 * noise3(vec3<f32>(uv * 8.0, 0.0));
    return mix(value, value * irregular, clamp(wash, 0.0, 1.0));
}

// ---------------------------------------------------------------------------
// Skin and hair
// ---------------------------------------------------------------------------

// Wrapped diffuse: light reaches a little past the terminator, and what
// reaches there has travelled through the skin and comes out red.
fn shade_skin(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, radiance: vec3<f32>, albedo: vec3<f32>, roughness: f32) -> vec3<f32> {
    let n_dot_l = dot(n, l);
    let wrap = clamp((n_dot_l + 0.45) / 1.45, 0.0, 1.0);
    let lit = clamp(n_dot_l, 0.0, 1.0);
    let scatter = vec3<f32>(0.85, 0.28, 0.18) * (wrap - lit) * 0.7;
    let diffuse = albedo / PI * (wrap + scatter);
    // Specular only (no albedo): an oily sheen.
    let specular = shade_light(n, v, l, radiance, vec3<f32>(0.0), roughness, 0.0);
    return diffuse * radiance + specular;
}

fn hash11(x: f32) -> f32 {
    return fract(sin(x * 127.1) * 43758.5453);
}

// Smooth 1D value noise.
fn noise1(x: f32) -> f32 {
    let i = floor(x);
    let f = fract(x);
    return mix(hash11(i), hash11(i + 1.0), f * f * (3.0 - 2.0 * f));
}

fn hash31(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
}

// Smooth 3D value noise.
fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let x00 = mix(hash31(i), hash31(i + vec3<f32>(1.0, 0.0, 0.0)), u.x);
    let x10 = mix(hash31(i + vec3<f32>(0.0, 1.0, 0.0)), hash31(i + vec3<f32>(1.0, 1.0, 0.0)), u.x);
    let x01 = mix(hash31(i + vec3<f32>(0.0, 0.0, 1.0)), hash31(i + vec3<f32>(1.0, 0.0, 1.0)), u.x);
    let x11 = mix(hash31(i + vec3<f32>(0.0, 1.0, 1.0)), hash31(i + vec3<f32>(1.0, 1.0, 1.0)), u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

// A close crop: a fine, even grain of hair ends over the scalp, with no
// direction to it. `p` in metres; fades to the average before aliasing.
fn grain(p: vec3<f32>) -> f32 {
    var s = 0.0;
    var weight = 0.0;
    let scales = array<f32, 2>(900.0, 2400.0);
    let amounts = array<f32, 2>(0.4, 0.6);
    for (var k = 0; k < 2; k++) {
        let x = p * scales[k];
        let fade = clamp(1.5 - length(fwidth(x)) * 1.2, 0.0, 1.0);
        s += mix(0.5, noise3(x), fade) * amounts[k];
        weight += amounts[k];
    }
    return s / weight;
}

// Strands and clumps across the hair (U runs across, in metres): fades to
// the average before it would alias.
fn strands(u: f32) -> f32 {
    var s = 0.0;
    var weight = 0.0;
    let scales = array<f32, 3>(260.0, 900.0, 2600.0);
    let amounts = array<f32, 3>(0.45, 0.3, 0.25);
    for (var k = 0; k < 3; k++) {
        let x = u * scales[k];
        let fade = clamp(1.5 - fwidth(x) * 1.5, 0.0, 1.0);
        s += mix(0.5, noise1(x), fade) * amounts[k];
        weight += amounts[k];
    }
    return s / weight;
}

// Kajiya-Kay hair: highlights are bands across the strands, two lobes
// shifted along them (the white one off the cuticle, the coloured one
// through the fibre).
fn shade_hair(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, radiance: vec3<f32>, albedo: vec3<f32>, t: vec3<f32>, s: f32, len: f32) -> vec3<f32> {
    let h = normalize(l + v);
    let n_dot_l = dot(n, l);
    let shadow = smoothstep(-0.15, 0.25, n_dot_l);
    let diffuse = albedo / PI * clamp(n_dot_l * 0.75 + 0.25, 0.0, 1.0);
    let t1 = normalize(t - n * 0.08);
    let t2 = normalize(t + n * 0.12);
    let th1 = dot(t1, h);
    let th2 = dot(t2, h);
    // Strands catch the light one by one: the bands break up along them.
    let glint = s * s * 1.6;
    let spec1 = pow(sqrt(max(1.0 - th1 * th1, 0.0)), 120.0) * 0.035 * glint;
    let spec2 = pow(sqrt(max(1.0 - th2 * th2, 0.0)), 30.0) * 0.06 * glint;
    let tint = sqrt(albedo) * 1.5;
    // Short hair has no long fibres to line the highlight up along.
    let specular = (vec3<f32>(spec1) + tint * spec2) * shadow * (0.15 + 0.85 * len);
    return (diffuse + specular) * radiance;
}

// ---------------------------------------------------------------------------
// Technical heatmaps
// ---------------------------------------------------------------------------

// Blue → cyan → green → yellow → red for t in 0..1 (linear RGB). Five
// evenly spaced stops; mirrored by `SurfaceOverlay::heat_srgb` for legends.
fn heat(t: f32) -> vec3<f32> {
    let stops = array<vec3<f32>, 5>(
        vec3<f32>(0.02, 0.12, 0.85),
        vec3<f32>(0.0, 0.62, 0.78),
        vec3<f32>(0.05, 0.70, 0.12),
        vec3<f32>(0.95, 0.80, 0.03),
        vec3<f32>(0.90, 0.04, 0.02),
    );
    let x = clamp(t, 0.0, 1.0) * 4.0;
    let i = min(u32(x), 3u);
    return mix(stops[i], stops[i + 1u], x - f32(i));
}

// Colour of a measured value on the garment.
//
// Strain map (mode 1): `value` is the relative elongation of the mesh edges
// around the vertex, (length − rest length) / rest length. Relaxed and
// compressed fabric is blue; fabric stretched by `overlay.z` or more is red.
//
// Fit map (mode 2): `value` is the clearance between fabric and body in
// metres, measured on the CPU by casting a ray from the fabric against the
// body. Fabric lying on the skin (`overlay.z` or closer) is red; fabric
// hanging `overlay.y` or further away is blue.
fn technical_color(value: f32, n: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = (value - object.overlay.y) / (object.overlay.z - object.overlay.y);
    var color = heat(t);
    // Thin iso-lines every tenth of the range make gradients readable as
    // quantities, not just as hues. They fade out before they can alias.
    let scaled = clamp(t, 0.0, 1.0) * 10.0;
    let width = fwidth(scaled);
    let line = 1.0 - smoothstep(0.0, 1.5 * width, abs(fract(scaled + 0.5) - 0.5));
    color = mix(color, color * 0.55, line * clamp(1.0 - width * 2.0, 0.0, 1.0));
    // Headlight shading only: enough to read the folds, without highlights
    // or shadows that would be mistaken for values.
    return color * (0.45 + 0.55 * abs(dot(n, v)));
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    var n = normalize(in.world_normal);
    if !front_facing {
        n = -n; // cloth is double sided
    }
    let v = normalize(globals.eye.xyz - in.world_position);

    if object.overlay.x > 0.5 {
        // No tone mapping: the legend must match the surface.
        return vec4<f32>(technical_color(in.value, n, v) + object.emissive.rgb, 1.0);
    }

    let kind = u32(object.shading.x + 0.5);
    var roughness = clamp(object.params.x, 0.045, 1.0);
    var metallic = clamp(object.params.y, 0.0, 1.0);
    var albedo = object.base_color.rgb * in.color;
    let pattern_id = floor(object.params.w + 0.001);
    let wash = fract(object.params.w * 10.0);
    if kind == 3u {
        roughness = clamp(in.uv.x, 0.045, 1.0);
        metallic = clamp(in.uv.y, 0.0, 1.0);
    }
    if object.params.z > 0.0 {
        let pattern = fabric_surface(in.uv, pattern_id, wash, object.fabric.x);
        var threads = 36.0 * object.fabric.x;
        if pattern_id == 3.0 { threads = 18.0 * object.fabric.x; }
        albedo *= mix(vec3<f32>(1.0), pattern * weave(in.uv, threads), object.params.z);
        if pattern_id == 1.0 {
            albedo = mix(albedo, vec3<f32>(0.92, 0.94, 1.0), denim_fiber_mask(in.uv) * object.params.z * 0.72);
        }
        let height = pattern_height(in.uv, pattern_id, object.fabric.x, object.fabric.y) * object.params.z;
        // Use screen-space derivatives as a stable tangent-space bump. This
        // adds ridge highlights without changing the cloth mesh or physics.
        n = normalize(n - vec3<f32>(dpdx(height), dpdy(height), 0.0) * 18.0);
    }
    let key = normalize(globals.key_direction.xyz);
    let fill = normalize(globals.fill_direction.xyz);
    let ambient = mix(globals.ground.rgb, globals.sky.rgb, n.y * 0.5 + 0.5);

    var color = vec3<f32>(0.0);
    var alpha = 1.0;
    if kind == 1u {
        color = shade_skin(n, v, key, globals.key_color.rgb, albedo, roughness);
        color += shade_skin(n, v, fill, globals.fill_color.rgb, albedo, roughness);
        color += ambient * albedo;
        // A faint sheen where the skin turns away.
        color += globals.sky.rgb * 0.06 * pow(1.0 - max(dot(n, v), 0.0), 4.0);
    } else if kind == 2u {
        // Strands run away from the crown along the surface.
        let away = in.world_position - object.shading.yzw;
        var t = away - n * dot(away, n);
        if dot(t, t) < 1e-8 {
            t = vec3<f32>(0.0, -1.0, 0.0);
        }
        t = normalize(t);
        // V says how long the hair is: a crop is grain, long hair strands.
        let len = clamp(in.uv.y, 0.0, 1.0);
        let crop = grain(in.world_position);
        let s = mix(crop, strands(in.uv.x), len);
        albedo *= mix(0.75 + 0.5 * crop, 0.6 + 0.8 * s, len);
        color = shade_hair(n, v, key, globals.key_color.rgb, albedo, t, s, len);
        color += shade_hair(n, v, fill, globals.fill_color.rgb, albedo, t, s, len);
        color += ambient * albedo * 0.8;
        // Where coverage runs out, strands end one by one: each strand
        // has its own threshold, so the edge breaks up into tips.
        // Mixed with a per-pixel dither so thinning hair (a hairline, a
        // close crop) turns into a gradient rather than a comb.
        let strand_end = noise1(in.uv.x * 1400.0) * 0.6 + noise1(in.uv.x * 380.0 + 17.0) * 0.4;
        let strand = mix(noise3(in.world_position * 1800.0), strand_end, len);
        let pixel = fract(52.9829189 * fract(dot(in.clip_position.xy, vec2<f32>(0.06711056, 0.00583715))));
        let threshold = (0.55 * strand + 0.45 * pixel) * 0.92;
        alpha = smoothstep(threshold, threshold + 0.06, in.value);
    } else {
        color = shade_light(n, v, key, globals.key_color.rgb, albedo, roughness, metallic);
        color += shade_light(n, v, fill, globals.fill_color.rgb, albedo, roughness, metallic);
        color += ambient * albedo * (1.0 - metallic);
        if metallic > 0.0 {
            // Metals have no diffuse: give them something to reflect.
            let r = reflect(-v, n);
            let environment = mix(globals.ground.rgb, globals.sky.rgb * 1.6, smoothstep(-0.2, 0.4, r.y));
            let f0 = mix(vec3<f32>(0.04), albedo, metallic);
            color += environment * fresnel_schlick(max(dot(n, v), 0.0), f0) * (1.0 - roughness * 0.7) * metallic;
        }
    }
    color += object.emissive.rgb;

    // Reinhard tone mapping; the sRGB target applies the transfer function.
    color = color / (color + vec3<f32>(1.0));
    return vec4<f32>(color, alpha);
}

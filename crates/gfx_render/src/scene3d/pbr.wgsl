// Metallic-roughness PBR (Cook-Torrance, GGX) with one directional key light,
// a fill light and hemispherical ambient, with shadow maps. Shared by the
// avatar and the fabric, and by the live view and the studio render (which
// averages many samples of its lights; see studio.rs).

struct Globals {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    key_direction: vec4<f32>,   // points from the surface towards the light
    key_color: vec4<f32>,
    fill_direction: vec4<f32>,
    fill_color: vec4<f32>,
    sky: vec4<f32>,
    ground: vec4<f32>,
    // Where the key and fill light see the scene from (shadow map layers 0, 1).
    shadow_key: mat4x4<f32>,
    shadow_fill: mat4x4<f32>,
    // x: lights with shadows (0, 1: key, 2: key and fill), y: 1 in the studio
    // render (linear output, stochastic hair), z: studio sample, w: studio extras.
    options: vec4<f32>,
    // x: shadow texel in metres, y: key filter radius, z: fill filter radius,
    // w: studio render: how far from the camera the floor starts to fade
    // into the backdrop.
    shadow: vec4<f32>,
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
    // x: procedural repeat scale, y: relief strength, z: 1 for garment cloth
    // (sheen, light through it).
    fabric: vec4<f32>,
    // Woven from a draft. xy: texture repeats per UV unit, z: 1 if woven,
    // w: relief strength.
    weave: vec4<f32>,
    // The weave's yarns. xy: threads across the texture (ends, picks), z:
    // irregularity, w: slubs.
    yarn: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var shadow_maps: texture_depth_2d_array;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;
@group(1) @binding(0) var<uniform> object: Object;
// The weave's colour (sRGB) and surface (normal in RGB, height in alpha),
// with mip levels: sampled trilinearly and anisotropically.
@group(2) @binding(0) var weave_color: texture_2d<f32>;
@group(2) @binding(1) var weave_surface: texture_2d<f32>;
@group(2) @binding(2) var weave_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // Strain (relative edge elongation) or clearance to the body (metres).
    @location(3) value: f32,
    @location(4) color: vec3<f32>,
    // Garments: distance to the piece's edge, ridge, crease, face (see
    // MeshVertex::wear).
    @location(5) wear: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) value: f32,
    @location(4) color: vec3<f32>,
    @location(5) wear: vec4<f32>,
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
    out.wear = in.wear;
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

// Unevenly spun yarns: every thread a little lighter or darker than the
// next, and slubs (thick stretches) here and there. Computed per thread of
// the whole garment rather than baked into the texture, which repeats: a
// thread darker than the rest, repeated every few centimetres, turns into
// a grid. Fades out where threads get smaller than a pixel.
fn yarn_variation(weave_uv: vec2<f32>, grad_x: vec2<f32>, grad_y: vec2<f32>) -> f32 {
    let threads = weave_uv * object.yarn.xy; // x: ends, y: picks
    let footprint = max(
        abs(grad_x.x * object.yarn.x) + abs(grad_y.x * object.yarn.x),
        abs(grad_x.y * object.yarn.y) + abs(grad_y.y * object.yarn.y),
    );
    let visible = 1.0 - smoothstep(0.5, 1.5, footprint);
    let end = floor(threads.x);
    let pick = floor(threads.y);
    let irregular = object.yarn.z;
    var shade = 1.0 + (hash11(end * 0.618 + 3.1) - 0.5) * 0.22 * irregular;
    shade *= 1.0 + (hash11(pick * 0.733 + 7.9) - 0.5) * 0.12 * irregular;
    // Slubs: lighter stretches a few crossings long along some threads.
    let slub_warp = smoothstep(0.78, 0.98, noise1(threads.y / 5.0 + hash11(end) * 91.0));
    let slub_weft = smoothstep(0.80, 0.98, noise1(threads.x / 5.0 + hash11(pick + 0.5) * 53.0));
    shade *= 1.0 + object.yarn.w * 0.35 * max(slub_warp, slub_weft * 0.6);
    // Long streaks along the warp, a few threads wide, as ring-spun yarn
    // gives (denim); they stay visible a little further away.
    let streak_visible = 1.0 - smoothstep(2.0, 6.0, footprint);
    let streak = 1.0 + (noise1(threads.x * 0.21 + 13.0) - 0.5) * 0.14 * irregular * streak_visible;
    return streak * mix(1.0, shade, visible);
}

// How far, in threads, the yarns at `weave_uv` lie from where a perfect
// loom would put them (x: ends sideways, y: picks along). Smooth, so the
// threads stay whole: slow drifts over centimetres, and faster wobbles over
// a few threads, more for irregular yarns.
fn yarn_wander(weave_uv: vec2<f32>) -> vec2<f32> {
    let threads = weave_uv * object.yarn.xy;
    let irregular = object.yarn.z;
    // Slow drift: a thread or two over a few centimetres.
    let slow = vec2<f32>(
        fbm(vec3<f32>(threads * vec2<f32>(0.012, 0.02), 1.3)),
        fbm(vec3<f32>(threads * vec2<f32>(0.02, 0.012), 5.9)),
    ) - 0.5;
    // Each end meanders along its length; each pick is beaten in a little
    // closer or further than the last.
    let end_meander = noise3(vec3<f32>(threads.x * 0.35, threads.y * 0.14, 8.1)) - 0.5;
    let pick_beat = noise3(vec3<f32>(threads.x * 0.08, threads.y * 0.5, 2.7)) - 0.5;
    return slow * (1.5 + 2.5 * irregular) + vec2<f32>(end_meander, pick_beat) * (0.4 + 0.9 * irregular);
}

// Fractal noise, four octaves, 0..1.
fn fbm(p: vec3<f32>) -> f32 {
    var sum = 0.0;
    var amplitude = 0.5;
    var q = p;
    for (var k = 0; k < 4; k++) {
        sum += amplitude * noise3(q);
        q = q * 2.03 + vec3<f32>(17.1, 3.3, 9.7);
        amplitude *= 0.5;
    }
    return sum / 0.9375;
}

// What makes a real garment's colour uneven: dye and finish never take
// evenly (centimetre-scale clouds, domain-warped so they do not look like
// noise), and wear, where the garment is used: along seams and hems and on
// the ridges of folds the colour rubs off, patchily and along the threads
// (on denim the indigo warp first, showing its white core).
fn imperfections(albedo_in: vec3<f32>, in: VertexOutput, warpness: f32, woven: bool) -> vec3<f32> {
    var albedo = albedo_in;
    let irregular = select(0.4, object.yarn.z, woven);
    let p = vec3<f32>(in.uv * 12.0, 0.0);
    let warped = vec3<f32>(fbm(p * 0.5 + 1.7), fbm(p * 0.5 + 9.2), 0.0);
    let clouds = fbm(p + warped * 1.5);
    albedo *= 1.0 + (clouds - 0.5) * (0.08 + 0.2 * irregular);
    albedo *= 1.0 + (fbm(vec3<f32>(in.uv * 2.5, 4.0)) - 0.5) * 0.08;

    let amount = object.fabric.w * step(0.5, in.wear.w);
    if amount > 0.0 {
        let edge = exp(-in.wear.x / 0.009);
        let ridge = smoothstep(10.0, 90.0, in.wear.y);
        // Streaks along the warp, a thread or so wide, faded out before
        // they would shimmer.
        let streak_uv = in.uv * vec2<f32>(160.0, 40.0);
        let fine = 1.0 - smoothstep(0.5, 1.5, fwidth(streak_uv.x));
        let grain = mix(0.5, noise3(vec3<f32>(streak_uv, 2.0)), fine);
        let patches = smoothstep(0.3, 0.75, fbm(vec3<f32>(in.uv * 25.0, 7.0)) + 0.3 * (grain - 0.5));
        var rub = clamp((edge * 0.85 + ridge * 0.75) * (0.35 + 0.95 * patches), 0.0, 1.0) * amount;
        if woven {
            rub *= mix(0.7, 1.25, warpness);
        }
        let faded = pow(max(albedo, vec3<f32>(0.0)), vec3<f32>(0.6)) * 0.85 + vec3<f32>(0.025);
        albedo = mix(albedo, faded, rub);
        // And the cloth as a whole has been washed a few times.
        albedo = mix(albedo, faded, 0.2 * amount * fbm(vec3<f32>(in.uv * 4.0, 11.0)));
    }
    return albedo;
}

// A tangent frame from screen-space derivatives (the mesh carries no
// tangents): T along the texture's U, B along V, N the surface normal.
fn cotangent_frame(n: vec3<f32>, p: vec3<f32>, uv: vec2<f32>) -> mat3x3<f32> {
    let dp1 = dpdx(p);
    let dp2 = dpdy(p);
    let duv1 = dpdx(uv);
    let duv2 = dpdy(uv);
    let dp2perp = cross(dp2, n);
    let dp1perp = cross(n, dp1);
    let t = dp2perp * duv1.x + dp1perp * duv2.x;
    let b = dp2perp * duv1.y + dp1perp * duv2.y;
    let scale = inverseSqrt(max(max(dot(t, t), dot(b, b)), 1e-20));
    return mat3x3<f32>(t * scale, b * scale, n);
}

// Light off yarns, which are bundles of fibres: like hair (Kajiya-Kay) the
// highlight is a band across each fibre's direction `t`. `warpness` says
// how much of the pixel is warp on top (1) or weft (0).
fn fibre_lobe(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, t: vec3<f32>, exponent: f32) -> f32 {
    let h = normalize(v + l);
    let t_dot_h = dot(t, h);
    let along = sqrt(max(1.0 - t_dot_h * t_dot_h, 0.0));
    return pow(along, exponent) * (exponent + 2.0) / (2.0 * PI);
}

fn fibre_highlight(
    n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, radiance: vec3<f32>,
    warp: vec3<f32>, weft: vec3<f32>, warpness: f32, roughness: f32,
) -> vec3<f32> {
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 0.0);
    if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
        return vec3<f32>(0.0);
    }
    // Smooth yarns (silk, satin) shine in a narrow band, matt cotton in a
    // wide, faint one.
    let exponent = mix(160.0, 12.0, clamp((roughness - 0.3) / 0.7, 0.0, 1.0));
    let lobe = mix(fibre_lobe(n, v, l, weft, exponent), fibre_lobe(n, v, l, warp, exponent), warpness);
    let fresnel = fresnel_schlick(max(dot(normalize(v + l), v), 0.0), vec3<f32>(0.04));
    // Fibres shade each other: no highlight where the surface turns away.
    let shadowing = sqrt(n_dot_l * n_dot_v);
    // Matt, hairy yarns (cotton, wool) scatter it into a haze: only smooth
    // filament yarns (silk, satin) show a clear band.
    let smooth_yarn = 1.0 - smoothstep(0.3, 0.95, roughness);
    return fresnel * lobe * 0.25 * smooth_yarn * smooth_yarn * shadowing * radiance;
}

// The soft glow of fibres standing off cloth at grazing angles (Charlie
// sheen, Estevez & Kulla).
fn sheen(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, radiance: vec3<f32>, color: vec3<f32>) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 1e-3);
    let sin_h = sqrt(max(1.0 - pow(dot(n, h), 2.0), 0.0));
    let inv_r = 1.0 / 0.45;
    let d = (2.0 + inv_r) * pow(sin_h, inv_r) / (2.0 * PI);
    let visibility = 1.0 / (4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v) + 1e-3);
    return color * d * visibility * radiance * n_dot_l;
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

// Filmic tone mapping (ACES, Narkowicz's fit), as the studio render uses.
fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
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

// ---------------------------------------------------------------------------
// Shadows
// ---------------------------------------------------------------------------

// How much of the light from `l` reaches `p` (0..1): percentage-closer
// filtering over a small rotated disk of the shadow map, so edges are soft
// and free of stair steps.
fn shadow_visibility(p: vec3<f32>, n: vec3<f32>, l: vec3<f32>, layer: i32, radius: f32, pixel: vec2<f32>) -> f32 {
    var m = globals.shadow_key;
    if layer == 1 {
        m = globals.shadow_fill;
    }
    // Lift the point off its own surface, more where the light grazes it.
    let texel = globals.shadow.x;
    let grazing = 1.0 - abs(dot(n, l));
    let lifted = p + n * texel * (1.0 + 2.0 * grazing) + l * texel;
    let q = m * vec4<f32>(lifted, 1.0);
    let uv = q.xy * vec2<f32>(0.5, -0.5) + 0.5;
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || q.z > 1.0 {
        return 1.0;
    }
    let size = vec2<f32>(textureDimensions(shadow_maps).xy);
    let angle = 6.2831853 * fract(52.9829189 * fract(dot(pixel, vec2<f32>(0.06711056, 0.00583715))) + globals.options.z * 0.618034);
    let rotation = mat2x2<f32>(cos(angle), sin(angle), -sin(angle), cos(angle));
    let taps = array<vec2<f32>, 8>(
        vec2<f32>(-0.326, -0.406), vec2<f32>(-0.840, -0.074), vec2<f32>(-0.696, 0.457), vec2<f32>(-0.203, 0.621),
        vec2<f32>(0.962, -0.195), vec2<f32>(0.473, -0.480), vec2<f32>(0.519, 0.767), vec2<f32>(0.185, -0.893),
    );
    var lit = 0.0;
    for (var k = 0; k < 8; k++) {
        let offset = rotation * taps[k] * radius / size;
        lit += textureSampleCompareLevel(shadow_maps, shadow_sampler, uv + offset, layer, q.z);
    }
    return lit / 8.0;
}

// ---------------------------------------------------------------------------
// Corduroy: ribs of pile running along the warp (V), across U.
// ---------------------------------------------------------------------------

// Height of the pile at `x` ribs across (0 in the furrow, 1 on a rib top):
// full, rounded ribs that meet in a furrow. Mirrored by `cord_profile` in
// refine.rs, which turns them into geometry for the studio render.
fn cord_profile(x: f32) -> f32 {
    return pow(sin(PI * fract(x)), 0.7);
}

fn cord_slope(x: f32) -> f32 {
    let e = 0.01;
    return (cord_profile(x + e) - cord_profile(x - e)) / (2.0 * e);
}

// Metres of cloth per UV unit, from screen-space derivatives (the UV
// mapping of the pieces is uniform).
fn metres_per_uv(p: vec3<f32>, uv: vec2<f32>) -> f32 {
    let area = length(cross(dpdx(p), dpdy(p)));
    let uv_area = abs(dpdx(uv).x * dpdy(uv).y - dpdx(uv).y * dpdy(uv).x);
    return sqrt(area / max(uv_area, 1e-12));
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    var n = normalize(in.world_normal);
    if !front_facing {
        n = -n; // cloth is double sided
    }
    let v = normalize(globals.eye.xyz - in.world_position);
    // Derivatives first, before anything branches on the fragment.
    let weave_uv = in.uv * object.weave.xy;
    let studio = globals.options.y > 0.5;
    // The live view leans towards the blurrier mip: shading is sampled
    // once per pixel there, and a check or a twill a pixel or two wide
    // beats against the pixel grid (moiré). The studio render averages
    // hundreds of samples per pixel and takes the sharp one.
    let mip_bias = select(0.7, 0.0, studio);
    let grad_x = dpdx(weave_uv) * exp2(mip_bias);
    let grad_y = dpdy(weave_uv) * exp2(mip_bias);
    let texels = vec2<f32>(textureDimensions(weave_surface));
    let weave_lod = 0.5 * log2(max(max(dot(grad_x * texels, grad_x * texels), dot(grad_y * texels, grad_y * texels)), 1e-8));
    let m_per_uv = metres_per_uv(in.world_position, in.uv);
    let geometric_n = normalize(in.world_normal) * select(-1.0, 1.0, front_facing);
    let frame = cotangent_frame(geometric_n, in.world_position, in.uv);
    let rib_footprint = fwidth(in.uv.x * 18.0 * object.fabric.x);
    let key = normalize(globals.key_direction.xyz);
    // Tangent-space view and key light directions (T along U, B along V).
    let v_t = vec3<f32>(dot(v, frame[0]), dot(v, frame[1]), dot(v, geometric_n));
    let key_t = vec3<f32>(dot(key, frame[0]), dot(key, frame[1]), dot(key, geometric_n));

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
    let is_woven = object.weave.z > 0.5;
    var warp_dir = vec3<f32>(0.0, 1.0, 0.0);
    var weft_dir = vec3<f32>(1.0, 0.0, 0.0);
    var warpness = 0.5;
    var cavity = 1.0;
    // Light the relief itself keeps from the surface (its own shadows).
    var relief_shadow = 1.0;
    var woven_height = 1.0;
    if is_woven {
        // The yarns in 3D: where threads are a few pixels wide or more, the
        // view ray is followed into the weave (parallax occlusion), so
        // floats stand proud of the threads they cross and hide them at a
        // slant, and the key light is followed out again, so the floats
        // cast shadows into the gaps.
        // Real yarns do not run ruler-straight: ends drift sideways over a
        // few centimetres, picks are beaten in unevenly, and each crossing
        // sits a little off. So twill lines wobble and step instead of
        // being drawn with a ruler.
        let yarn_uv = weave_uv + yarn_wander(weave_uv) / max(object.yarn.xy, vec2<f32>(1.0));
        var uv = yarn_uv;
        let near = 1.0 - smoothstep(1.5, 3.0, weave_lod);
        if near > 0.0 {
            // Yarns stand about a third of their spacing proud.
            let spacing = m_per_uv / max(object.weave.x * object.yarn.x, 1e-3);
            let depth = 0.35 * object.weave.w * spacing * near;
            let m_per_weave_uv = m_per_uv / max(object.weave.xy, vec2<f32>(1e-3));
            let lod = max(weave_lod, 0.0);
            let steps = select(8, 16, studio);
            let run = -v_t.xy / max(v_t.z, 0.25) / m_per_weave_uv; // weave uv per metre down
            var previous_gap = 0.0;
            var previous_uv = uv;
            for (var k = 1; k <= steps; k++) {
                let descent = depth * f32(k) / f32(steps);
                let at = yarn_uv + run * descent;
                let h = textureSampleLevel(weave_surface, weave_sampler, at, lod).a;
                let gap = descent - depth * (1.0 - h);
                if gap >= 0.0 {
                    uv = mix(previous_uv, at, previous_gap / (previous_gap - gap + 1e-6));
                    break;
                }
                previous_gap = gap;
                previous_uv = at;
            }
            if studio && key_t.z > 0.0 {
                // Towards the key: does a float rise above the light ray?
                let h0 = textureSampleLevel(weave_surface, weave_sampler, uv, lod).a;
                let up = key_t.xy / max(key_t.z, 0.15) / m_per_weave_uv; // weave uv per metre up
                var blocked = 0.0;
                for (var k = 1; k <= 6; k++) {
                    let rise = depth * f32(k) / 6.0 * (1.0 - h0);
                    let at = uv + up * rise;
                    let h = textureSampleLevel(weave_surface, weave_sampler, at, lod).a;
                    blocked = max(blocked, (depth * (h - h0) - rise) / max(depth * 0.15, 1e-7));
                }
                relief_shadow = 1.0 - clamp(blocked, 0.0, 1.0) * 0.8 * near;
            }
        }
        let woven = textureSampleGrad(weave_color, weave_sampler, uv, grad_x, grad_y);
        let woven_surface = textureSampleGrad(weave_surface, weave_sampler, uv, grad_x, grad_y);
        woven_height = woven_surface.a;
        // From further away the eye no longer resolves the interlacing,
        // only the cloth's colour: twill lines a pixel wide would read as
        // printed stripes. Fade towards the colour averaged over a few
        // threads (which keeps stripes and checks of the colour order).
        let far = textureSampleLevel(weave_color, weave_sampler, uv, max(weave_lod, 5.0));
        let resolved = mix(woven.rgb, far.rgb, smoothstep(0.5, 3.0, weave_lod) * 0.75);
        albedo = resolved * in.color * yarn_variation(yarn_uv, grad_x, grad_y);
        // The normals of the threads average out with distance: what was
        // bumpy becomes rough, not smooth (Toksvig). The mips keep the
        // averaged, shortened normal for this.
        let ts = woven_surface.xyz * 2.0 - 1.0;
        let shortened = clamp(length(ts), 0.05, 1.0);
        let spread = (1.0 - shortened) / shortened * object.weave.w * object.weave.w;
        roughness = sqrt(roughness * roughness + 2.0 * spread);
        let bent = normalize(vec3<f32>(ts.xy * object.weave.w, max(ts.z, 0.05)));
        let yarn_frame = mat3x3<f32>(frame[0], frame[1], n);
        n = normalize(yarn_frame * bent);
        // Which way the yarns on top run here: along V (warp) or U (weft).
        warp_dir = frame[1];
        weft_dir = frame[0];
        warpness = woven.a;
        // Light catches the tops of the yarns; the gaps between are dull.
        roughness = clamp(roughness * (1.2 - 0.35 * woven_height), 0.3, 1.0);
        // Light reaches less into the gaps between yarns.
        cavity *= mix(0.78, 1.0, woven_height);
    }
    if object.params.z > 0.0 && !is_woven && pattern_id == 3.0 {
        // Corduroy: ribs of pile with real depth. The view ray is followed
        // into the furrows (parallax occlusion), so ribs hide the furrows
        // behind them at a slant, and the key light out of them, so ribs
        // shade the furrow beside them.
        let ribs = 18.0 * object.fabric.x;
        let depth = 0.0012 * object.fabric.y * object.params.z; // metres
        let rib_width = 1.0 / ribs * m_per_uv; // metres
        let visible = 1.0 - smoothstep(0.35, 0.9, rib_footprint);
        var x = in.uv.x * ribs;
        // In the studio the ribs may be real geometry (fabric.z = 2): then
        // they hide and shadow each other by themselves.
        let displaced = object.fabric.z > 1.5;
        if visible > 0.0 && !displaced {
            // Ribs travelled per metre of descent along the ray.
            let run = -v_t.x / max(v_t.z, 0.2) / rib_width;
            let steps = select(12, 24, studio);
            var previous_gap = 0.0;
            var previous_x = x;
            for (var k = 1; k <= steps; k++) {
                let descent = depth * f32(k) / f32(steps);
                let at = in.uv.x * ribs + run * descent;
                // How far below the rib tops the ray is, minus how deep the
                // pile surface lies there.
                let gap = descent - depth * (1.0 - cord_profile(at));
                if gap >= 0.0 {
                    x = mix(previous_x, at, previous_gap / (previous_gap - gap + 1e-6));
                    break;
                }
                previous_gap = gap;
                previous_x = at;
            }
            if key_t.z > 0.0 {
                let h0 = depth * cord_profile(x);
                let up = key_t.x / max(key_t.z, 0.1) / rib_width; // ribs per metre up
                var blocked = 0.0;
                for (var k = 1; k <= 8; k++) {
                    let rise = (depth - h0) * f32(k) / 8.0;
                    let h = depth * cord_profile(x + up * rise);
                    blocked = max(blocked, (h - h0 - rise) / (depth * 0.2));
                }
                relief_shadow = 1.0 - clamp(blocked, 0.0, 1.0) * 0.85 * visible;
            }
        }
        let h = cord_profile(x);
        let slope = cord_slope(x) * depth / rib_width * visible; // dh/dmetre across
        n = normalize(n - frame[0] * slope);
        // Furrows get less light and show the backing between the ribs;
        // the pile on the ribs is a fine, uneven velvet.
        cavity = mix(1.0, mix(0.5, 1.0, h), visible);
        let pile = 0.93 + 0.14 * noise3(vec3<f32>(x * 6.0, in.uv.y * 260.0, 0.0));
        // The pile leans one way (down the garment): seen against the nap
        // it looks deeper and richer, along it paler.
        let nap = 1.0 + 0.14 * clamp(dot(v, frame[1]), -1.0, 1.0);
        albedo *= mix(vec3<f32>(1.0), vec3<f32>((0.9 + 0.16 * h) * mix(1.0, pile, visible) * nap), object.params.z);
    } else if object.params.z > 0.0 && !is_woven {
        let pattern = fabric_surface(in.uv, pattern_id, wash, object.fabric.x);
        var threads = 36.0 * object.fabric.x;
        albedo *= mix(vec3<f32>(1.0), pattern * weave(in.uv, threads), object.params.z);
        if pattern_id == 1.0 {
            albedo = mix(albedo, vec3<f32>(0.92, 0.94, 1.0), denim_fiber_mask(in.uv) * object.params.z * 0.72);
        }
        let height = pattern_height(in.uv, pattern_id, object.fabric.x, object.fabric.y) * object.params.z;
        // Use screen-space derivatives as a stable tangent-space bump. This
        // adds ridge highlights without changing the cloth mesh or physics.
        n = normalize(n - vec3<f32>(dpdx(height), dpdy(height), 0.0) * 18.0);
    }
    let is_cloth = object.fabric.z > 0.5;
    if is_cloth {
        albedo = imperfections(albedo, in, warpness, is_woven);
        // Worn garments darken in creases a little: less light, more dust.
        cavity *= 1.0 - 0.18 * smoothstep(15.0, 120.0, in.wear.z) * object.fabric.w * step(0.5, in.wear.w);
    }
    let fill = normalize(globals.fill_direction.xyz);
    let ambient = mix(globals.ground.rgb, globals.sky.rgb, n.y * 0.5 + 0.5) * cavity;

    // Shadows, from the geometric surface (the bent normal would let the
    // relief cast shadows of its own on the mesh).
    let geometric = frame[2];
    var key_light = globals.key_color.rgb;
    var fill_light = globals.fill_color.rgb;
    key_light *= relief_shadow;
    if globals.options.x > 0.5 {
        key_light *= shadow_visibility(in.world_position, geometric, key, 0, globals.shadow.y, in.clip_position.xy);
    }
    if globals.options.x > 1.5 {
        fill_light *= shadow_visibility(in.world_position, geometric, fill, 1, globals.shadow.z, in.clip_position.xy);
    }

    var color = vec3<f32>(0.0);
    var alpha = 1.0;
    if kind == 1u {
        color = shade_skin(n, v, key, key_light, albedo, roughness);
        color += shade_skin(n, v, fill, fill_light, albedo, roughness);
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
        color = shade_hair(n, v, key, key_light, albedo, t, s, len);
        color += shade_hair(n, v, fill, fill_light, albedo, t, s, len);
        color += ambient * albedo * 0.8;
        // Where coverage runs out, strands end one by one: each strand
        // has its own threshold, so the edge breaks up into tips.
        // Mixed with a per-pixel dither so thinning hair (a hairline, a
        // close crop) turns into a gradient rather than a comb.
        let strand_end = noise1(in.uv.x * 1400.0) * 0.6 + noise1(in.uv.x * 380.0 + 17.0) * 0.4;
        let strand = mix(noise3(in.world_position * 1800.0), strand_end, len);
        let pixel = fract(52.9829189 * fract(dot(in.clip_position.xy, vec2<f32>(0.06711056, 0.00583715))));
        // The live view dithers the ends against the pixel grid (coverage
        // samples fade it). The studio thins them out per sample instead:
        // a fixed pattern per pixel would stay as a grain in the average.
        let end_dither = select(pixel, 0.5, studio);
        let threshold = (0.55 * strand + 0.45 * end_dither) * 0.92;
        alpha = smoothstep(threshold, threshold + 0.06, in.value);
        if studio {
            // No coverage samples here: keep or drop the fragment, the
            // average over samples thins the strands out instead.
            let dither = fract(pixel + globals.options.z * 0.618034);
            if alpha < dither {
                discard;
            }
            alpha = 1.0;
        }
    } else {
        color = shade_light(n, v, key, key_light, albedo, roughness, metallic) * cavity;
        color += shade_light(n, v, fill, fill_light, albedo, roughness, metallic) * cavity;
        color += ambient * albedo * (1.0 - metallic);
        if is_cloth {
            // Fibres standing off the cloth catch the light at grazing
            // angles; how much varies over the cloth as it does with wear.
            let fuzz = 0.75 + 0.5 * noise3(vec3<f32>(in.uv * 31.0, 1.3));
            var amount = 0.45;
            if pattern_id == 3.0 && !is_woven {
                amount = 0.9; // the pile of corduroy is velvety
            }
            let sheen_color = (albedo * amount + vec3<f32>(0.02)) * fuzz;
            color += sheen(n, v, key, key_light, sheen_color);
            color += sheen(n, v, fill, fill_light, sheen_color);
            if is_woven {
                // Each yarn shines along itself: a band of light across the
                // floats, running with the warp on warp floats and with the
                // weft on weft floats (the glint of satin, the diagonal
                // shine of twill).
                color += fibre_highlight(n, v, key, key_light, warp_dir, weft_dir, warpness, roughness);
                color += fibre_highlight(n, v, fill, fill_light, warp_dir, weft_dir, warpness, roughness);
            }
            // Thin cloth lets light through: lit from behind, it glows in
            // its own colour. Only where nothing else stands in the way.
            var through = globals.key_color.rgb * max(-dot(geometric, key), 0.0);
            if globals.options.x > 0.5 {
                through *= shadow_visibility(in.world_position + key * 0.012, geometric, key, 0, globals.shadow.y, in.clip_position.xy);
            }
            color += albedo * albedo * through * 0.35 / PI;
            // Light scattered among the fibres comes out a little past the
            // terminator, and having bounced, in a deeper colour: folds turn
            // saturated rather than grey.
            for (var k = 0; k < 2; k++) {
                let l = select(fill, key, k == 0);
                let radiance = select(fill_light, key_light, k == 0);
                let ndl = dot(n, l);
                let wrapped = clamp((ndl + 0.4) / 1.4, 0.0, 1.0) - clamp(ndl, 0.0, 1.0);
                color += albedo * albedo * radiance * wrapped * 0.5 / PI * cavity;
            }
        }
        if metallic > 0.0 {
            // Metals have no diffuse: give them something to reflect.
            let r = reflect(-v, n);
            let environment = mix(globals.ground.rgb, globals.sky.rgb * 1.6, smoothstep(-0.2, 0.4, r.y));
            let f0 = mix(vec3<f32>(0.04), albedo, metallic);
            color += environment * fresnel_schlick(max(dot(n, v), 0.0), f0) * (1.0 - roughness * 0.7) * metallic;
        }
    }
    color += object.emissive.rgb;

    if studio {
        // Linear light: the studio averages samples, then tone maps. Far
        // off, the floor gives way to the backdrop (alpha is coverage; the
        // colour is premultiplied by it), as a seamless studio wall would.
        let far = globals.shadow.w;
        let coverage = 1.0 - smoothstep(far, far * 3.0, distance(in.world_position, globals.eye.xyz));
        return vec4<f32>(color * coverage, coverage);
    }
    // Filmic tone mapping; the sRGB target applies the transfer function.
    color = aces(color * 0.9);
    return vec4<f32>(color, alpha);
}

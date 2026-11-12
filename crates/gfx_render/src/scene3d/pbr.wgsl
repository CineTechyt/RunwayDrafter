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
    // x: roughness, y: metallic, z: weave strength, w: weave threads per UV unit
    params: vec4<f32>,
    emissive: vec4<f32>,
    // Technical views of the garment. x: 0 = fabric, 1 = strain heatmap,
    // 2 = fit map; y: the value shown blue; z: the value shown red.
    overlay: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> object: Object;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // Strain (relative edge elongation) or clearance to the body (metres).
    @location(3) value: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) value: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world = object.model * vec4<f32>(in.position, 1.0);
    out.world_position = world.xyz;
    // Models are only translated/rotated/uniformly scaled, so the model
    // matrix is a valid normal matrix.
    out.world_normal = (object.model * vec4<f32>(in.normal, 0.0)).xyz;
    out.uv = in.uv;
    out.value = in.value;
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

    let roughness = clamp(object.params.x, 0.045, 1.0);
    let metallic = clamp(object.params.y, 0.0, 1.0);
    var albedo = object.base_color.rgb;
    if object.params.z > 0.0 {
        albedo *= mix(1.0, weave(in.uv, object.params.w), object.params.z);
    }

    var color = shade_light(n, v, normalize(globals.key_direction.xyz), globals.key_color.rgb, albedo, roughness, metallic);
    color += shade_light(n, v, normalize(globals.fill_direction.xyz), globals.fill_color.rgb, albedo, roughness, metallic);

    let ambient = mix(globals.ground.rgb, globals.sky.rgb, n.y * 0.5 + 0.5);
    color += ambient * albedo * (1.0 - metallic);
    color += object.emissive.rgb;

    // Reinhard tone mapping; the sRGB target applies the transfer function.
    color = color / (color + vec3<f32>(1.0));
    return vec4<f32>(color, 1.0);
}

// The studio render's own passes: averaging the samples, and turning the
// average into the final picture.

struct Resolve {
    // x: exposure, y: vignette strength, z: samples so far
    params: vec4<f32>,
    // The seamless backdrop behind the figure, linear RGB: top and bottom.
    backdrop_top: vec4<f32>,
    backdrop_bottom: vec4<f32>,
};

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<uniform> resolve: Resolve;

struct Fullscreen {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> Fullscreen {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: Fullscreen;
    out.position = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}

// One sample, blended into the running average by the pipeline (the blend
// constant is 1 / samples so far).
@fragment
fn fs_accumulate(in: Fullscreen) -> @location(0) vec4<f32> {
    return textureLoad(source, vec2<i32>(in.position.xy), 0);
}

// Filmic tone mapping (ACES, Narkowicz's fit): highlights roll off
// gently instead of clipping, shadows keep their depth.
fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_resolve(in: Fullscreen) -> @location(0) vec4<f32> {
    let sample = textureLoad(source, vec2<i32>(in.position.xy), 0);
    // Alpha is how much of the pixel the scene covers; the backdrop shows
    // through the rest.
    let backdrop = mix(resolve.backdrop_bottom.rgb, resolve.backdrop_top.rgb, smoothstep(0.0, 1.0, 1.0 - in.uv.y));
    var color = sample.rgb + backdrop * (1.0 - clamp(sample.a, 0.0, 1.0));
    color = aces(color * resolve.params.x);
    // A soft vignette, as a lens gives.
    let centred = in.uv * 2.0 - 1.0;
    color *= 1.0 - resolve.params.y * pow(dot(centred, centred) * 0.5, 1.6);
    // A trace of noise against banding in the smooth backdrop.
    let noise = fract(52.9829189 * fract(dot(in.position.xy, vec2<f32>(0.06711056, 0.00583715))));
    color += (noise - 0.5) / 255.0;
    return vec4<f32>(color, 1.0);
}

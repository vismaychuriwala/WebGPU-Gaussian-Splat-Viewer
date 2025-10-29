struct VertexOutput {
    @builtin(position) position: vec4<f32>, // clip-space position written to rasterizer
    //TODO: information passed from vertex shader to fragment shader
    @location(0) color: vec4<f32>, // linear colour
    @location(1) conic: vec3<f32>, // pixel-space conic coefficients
    @location(2) opacity: f32,
    @location(3) center_ndc: vec2<f32>,
};

struct Splat {
    //TODO: store information for 2D splat rendering
    centre_ndc: vec2<f32>,
    radius_ndc: vec2<f32>,
    color: vec4<f32>,
    conic: vec3<f32>, // pixel-space conic coefficients
    opacity: f32,
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@group(0) @binding(0)
var<storage, read> splats : array<Splat>;
@group(0) @binding(1)
var<storage, read> sorted_indices : array<u32>;
@group(0) @binding(2)
var<uniform> camera : CameraUniforms;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_instance_index: u32
) -> VertexOutput {
    // Reconstruct the quad for each splat in clip space.
    var out: VertexOutput;
    let splat_index = sorted_indices[in_instance_index];
    let splat = splats[splat_index];

    let centre_ndc = splat.centre_ndc; // clip space
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0)
    );

    let offset_ndc = corners[in_vertex_index] * splat.radius_ndc; // NDC delta
    
    out.position = vec4<f32>(centre_ndc + offset_ndc, 0.0, 1.0);
    out.color = splat.color;
    out.conic = splat.conic;
    out.opacity = splat.opacity;
    out.center_ndc = centre_ndc; 

    return out;
}

@fragment
fn fs_main(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    // return in.color;
    var frag_pos_ndc = (in.position.xy / camera.viewport) * 2.0 - 1.0;
    frag_pos_ndc.y *= -1.0;
    var d = frag_pos_ndc - in.center_ndc;
    d *= camera.viewport * 0.5;
    // return vec4<f32>(d.x, d.y, 0.0, 1.0);
    d.x *= -1;

    let power = -0.5 * (in.conic.x * d.x * d.x
          + 2.0 * in.conic.y * d.x * d.y
          + in.conic.z * d.y * d.y);

    let weight = exp(power) * in.opacity;
    if (weight < 1.0 / 255.0) {
        // return vec4<f32>(0.0, 0.0, 0.0, 0.0);
        discard;
    }
    // return in.color * min(in.opacity * exp(power), 0.99);
    let raw_alpha = weight  * in.color.a;
    let alpha = sigmoid(raw_alpha * 6.0 - 3.0); // remap to emphasise mid-range opacity
    return vec4<f32>(in.color.rgb, alpha);
}

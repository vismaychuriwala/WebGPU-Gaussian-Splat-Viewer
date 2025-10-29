struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) color: vec4<f32>,
    @location(1) conic: vec3<f32>,
    @location(2) opacity: f32,
    @location(3) center: vec2<f32>,
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    position: vec4<f32>,
    quad_size: vec2<f32>,
    color: vec4<f32>,
    conic: vec3<f32>,
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
    let pos = splat.position;
    let uv = vec2<f32>(0.5 * (pos.x + 1.0), 0.5 * (1.0 - pos.y));
    out.center = uv * camera.viewport;
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0)
    );

    let offset = corners[in_vertex_index] * splat.quad_size;
    out.position = vec4<f32>(pos.x + offset.x, pos.y + offset.y, pos.z, 1.0);
    out.color = splat.color;
    out.conic = splat.conic;
    out.opacity = splat.opacity;
    return out;
}

@fragment
fn fs_main(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    let frag_px = in.position.xy;
    let quad_center = in.center;
    let d = vec2<f32>(quad_center.x - frag_px.x, frag_px.y - quad_center.y);

    let q = in.conic.x * d.x * d.x
          + 2.0 * in.conic.y * d.x * d.y
          + in.conic.z * d.y * d.y;

    if (q > 9.0) {
        discard;
    }

    let weight = exp(-0.5 * q);
    let raw_alpha = weight * in.opacity * in.color.a;
    let alpha = sigmoid(raw_alpha * 6.0 - 3.0);
    return vec4<f32>(in.color.rgb, alpha);
}

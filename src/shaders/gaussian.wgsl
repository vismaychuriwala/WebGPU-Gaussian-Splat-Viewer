struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) color: vec4<f32>,
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    position: vec4<f32>,
    quad_size: vec2<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0)
var<storage, read> splats : array<Splat>;
@group(0) @binding(1)
var<storage, read> sorted_indices : array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_instance_index: u32
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;
    let splat_index = sorted_indices[in_instance_index];
    let splat = splats[splat_index];
    let pos = splat.position;
    var corners = array<vec2<f32>,4>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>(-0.5,  0.5),
        vec2<f32>( 0.5,  0.5)
    );

    let offset = corners[in_vertex_index] * splat.quad_size;
    out.position = vec4<f32>(pos.x + offset.x, pos.y + offset.y, pos.z, 1.);
    out.color = splat.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}

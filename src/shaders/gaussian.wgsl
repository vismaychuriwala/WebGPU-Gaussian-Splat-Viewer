struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    position: vec4<f32>,
};

@group(0) @binding(0)
var<storage,read> splats : array<Splat>;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_instance_index: u32
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;
    let pos = splats[in_instance_index].position;
    var corners = array<vec2<f32>,4>(
        vec2<f32>(-0.005, -0.005),
        vec2<f32>( 0.005, -0.005),
        vec2<f32>(-0.005,  0.005),
        vec2<f32>( 0.005,  0.005)
    );

    let offset = corners[in_vertex_index];
    out.position = vec4<f32>(pos.x + offset.x, pos.y + offset.y, pos.z, 1.);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.);
}
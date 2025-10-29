const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: u32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    //TODO: store information for 2D splat rendering
    position: vec4<f32>,
    quad_size: vec2<f32>,
    color: vec4<f32>,
    conic: vec3<f32>,
    opacity: f32,
};

//TODO: bind your data here

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> render_settings: RenderSettings;

@group(1) @binding(0)
var<storage,read> gaussians : array<Gaussian>;
@group(1) @binding(1)
var<storage, read_write> splats : array<Splat>;
@group(1) @binding(2)
var<storage, read> sh_buffer : array<u32>;


@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;


fn getR(rot: array<u32,2>) -> mat3x3<f32> {
    var x = unpack2x16float(rot[0]).x;
    var y = unpack2x16float(rot[0]).y;
    var z = unpack2x16float(rot[1]).x;
    var r = unpack2x16float(rot[1]).y;

    // Normalize
    let len = sqrt(x*x + y*y + z*z + r*r);
    x = x / len;
    y = y / len;
    z = z / len;
    r = r / len;

    let R = mat3x3<f32>(
        1. - 2. * (y * y + z * z), 2. * (x * y - r * z), 2. * (x * z + r * y),
        2. * (x * y + r * z), 1. - 2. * (x * x + z * z), 2. * (y * z - r * x),
        2. * (x * z - r * y), 2. * (y * z + r * x), 1. - 2. * (x * x + y * y)
    );
    return R;
}

fn getS(scale: array<u32,2>, gaussian_scaling: f32) -> mat3x3<f32> {
    var x = unpack2x16float(scale[0]).x;
    var y = unpack2x16float(scale[0]).y;
    var z = unpack2x16float(scale[1]).x;

    x = exp(x);
    y = exp(y);
    z = exp(z);

    let S = mat3x3<f32>(
        x * gaussian_scaling, 0., 0.,
        0., y * gaussian_scaling, 0.,
        0., 0., z * gaussian_scaling
    );
    return S;
}

fn computeCov3d(rot: array<u32,2>, scale: array<u32,2>, gaussian_scaling: f32) -> mat3x3<f32> {
    let R = getR(rot);
    let S = getS(scale, gaussian_scaling);
    return R * S * transpose(S) * transpose(R);
}

fn transformPoint4x3(p: vec3<f32>, m: mat4x4<f32>) -> vec3<f32> {
    let transformed = vec3<f32>(
        dot(m[0].xyz, p) + m[3].x,
        dot(m[1].xyz, p) + m[3].y,
        dot(m[2].xyz, p) + m[3].z
    );
    return transformed;
}

fn computeCov2D(mean: vec3<f32>, cov3D: mat3x3<f32>) -> vec3<f32> {
    let W = mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz
    );
    let t = transformPoint4x3(mean, camera.view);

    let J = mat3x3<f32>(
        camera.focal.x / t.z, 0., -(camera.focal.x * t.x) / (t.z * t.z),
		0., camera.focal.y / t.z, -(camera.focal.y * t.y) / (t.z * t.z),
		0., 0., 0.
    );
    let T = W * J;
    var cov = transpose(T) * cov3D * T;
    cov[0][0] += 0.3;
	cov[1][1] += 0.3;
    return vec3<f32>(cov[0][0], cov[0][1], cov[1][1]);
}

fn computeConic(cov: vec3<f32>) -> vec3<f32> {
    let det = (cov.x * cov.z - cov.y * cov.y);
    let det_inv = 1.0/ det;
    return vec3<f32> (cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv);
}

fn computeRadius(cov: vec3<f32>) -> f32 {
    let det = (cov.x * cov.z - cov.y * cov.y);
    let mid = 0.5 * (cov.x + cov.z);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    return ceil(3.0 * sqrt(max(lambda1, lambda2)));
}

fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let base = splat_idx * 24u;
    let word_idx = base + (c_idx * 3u) / 2u;
    let pair0 = unpack2x16float(sh_buffer[word_idx + 0u]);
    let pair1 = unpack2x16float(sh_buffer[word_idx + 1u]);
    // pick first 3 as RGB
    return vec3<f32>(pair0.x, pair0.y, pair1.x);
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    //TODO: set up pipeline as described in instruction
    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    let keys_per_dispatch = workgroupSize * sortKeyPerThread;
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys

    var splat_out: Splat;
    let vertex = gaussians[idx];

    let a = unpack2x16float(vertex.pos_opacity[0]);
    let b = unpack2x16float(vertex.pos_opacity[1]);

    let pos = vec4<f32>(a.x, a.y, b.x, 1.);
    let opacity = b.y;

    // // MVP calculations
    let clip_pos = camera.proj * camera.view * pos;
    let ndc = clip_pos.xyz / clip_pos.w;
    if (abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2 || ndc.z < -1.0 || ndc.z > 1.0 || clip_pos.w <= 0.0) {
        return;
    }

    let uv = vec2<f32>(
    0.5 * (ndc.x + 1.0),
    0.5 * (1.0 - ndc.y)
    );

    let gaussian_scaling = render_settings.gaussian_scaling;

    splat_out.position = vec4<f32>(ndc, 1.0);
    let out_idx = atomicAdd(&sort_infos.keys_size, 1u);

    if (out_idx >= arrayLength(&splats)) { return; }

    let processed = out_idx + 1u;
    if (keys_per_dispatch != 0u) {
        let required_dispatch = (processed + keys_per_dispatch - 1u) / keys_per_dispatch;
        atomicMax(&sort_dispatch.dispatch_x, required_dispatch);
    }

    let cov3D = computeCov3d(vertex.rot, vertex.scale, gaussian_scaling);
    let cov2D = computeCov2D(pos.xyz, cov3D);
    let conic = computeConic(cov2D);
    let radius = computeRadius(cov2D);
    
    splat_out.conic = conic;
    splat_out.opacity = opacity;

    let quad_size = vec2<f32>(
      radius * 2.0 / camera.viewport.x,
      radius * 2.0 / camera.viewport.y
    );

    splat_out.quad_size = quad_size;
    // splat_out.color = vec4<f32>(quad_size.x, quad_size.y, 0.0, 1.0);
    let cam_pos = camera.view_inv[3].xyz;
    let dir = normalize(pos.xyz - cam_pos);
    splat_out.color = vec4<f32>(computeColorFromSH(dir, idx, render_settings.sh_deg), 1.0);

    let depth = 1.0 - ndc.z;
    splats[out_idx] = splat_out;
    sort_indices[out_idx] = out_idx;
    sort_depths[out_idx] = bitcast<u32>(depth);
}

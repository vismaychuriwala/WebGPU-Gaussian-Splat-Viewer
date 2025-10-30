import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  setGaussianMultiplier: (value: number) => void;
}

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {

  const sorter = get_sorter(pc.num_points, device);
  
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);

  const quadVerts = new Float32Array([
    -0.5, -0.5,
    0.5, -0.5, 
    -0.5, 0.5, 
    0.5, 0.5,
  ]);
  const quadVBO = createBuffer(
    device,
    'quad vertices',
    quadVerts.byteLength,
    GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    quadVerts
  );

  const drawArgsData = new Uint32Array([4, pc.num_points, 0, 0]);
  const indirect_buffer = createBuffer(
    device,
    'indirect buffer',
    drawArgsData.byteLength,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    drawArgsData
  );

  const splatBufferSize = pc.num_points * 8 * Float32Array.BYTES_PER_ELEMENT;
  const splat_buffer = createBuffer(
    device,
    'gaussian splats',
    splatBufferSize,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  );

  const renderSettingsBlock = new ArrayBuffer(16);
  const renderSettingsF32 = new Float32Array(renderSettingsBlock);
  const renderSettingsU32 = new Uint32Array(renderSettingsBlock);
  renderSettingsF32[0] = 1.0;
  renderSettingsU32[1] = pc.sh_deg;
  const render_settings_buffer = createBuffer(
    device,
    'render settings',
    renderSettingsBlock.byteLength,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    renderSettingsBlock
  );

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  const preprocess_workgroup_count = Math.ceil(pc.num_points / C.histogram_wg_size);

  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });

  const preprocess_camera_bind_group = device.createBindGroup({
    label: 'preprocess camera',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: render_settings_buffer } },
    ],
  });

  const preprocess_data_bind_group = device.createBindGroup({
    label: 'preprocess data',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: splat_buffer } },
      { binding: 2, resource: { buffer: pc.sh_buffer } },
    ],
  });


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  
  const render_shader = device.createShaderModule({code: renderWGSL});
  const render_pipeline = device.createRenderPipeline({
    label: 'gaussian render',
    layout: 'auto',
    vertex: {
      module: render_shader,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{
        format: presentation_format,
        blend: {
          color: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          },
          alpha: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          },
        },
      }],
    },
    primitive: {
      topology: 'triangle-strip',
      cullMode: 'none'
    },
  });

  const splat_bind_group = device.createBindGroup({
    label: 'gaussian splats',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 2, resource: { buffer: camera_buffer } },
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
        }
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, splat_bind_group);

    pass.setVertexBuffer(0, quadVBO);

    pass.drawIndirect(indirect_buffer, 0);

    // pass.draw(pc.num_points);
    pass.end();
  };

  const preprocess = (encoder: GPUCommandEncoder) => {
    const pass = encoder.beginComputePass({
      label: 'gaussian preprocess',
    });
    pass.setPipeline(preprocess_pipeline);
    pass.setBindGroup(0, preprocess_camera_bind_group);
    pass.setBindGroup(1, preprocess_data_bind_group);
    pass.setBindGroup(2, sort_bind_group);
    pass.dispatchWorkgroups(preprocess_workgroup_count);
    pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      device.queue.writeBuffer(sorter.sort_info_buffer, 0, new Uint32Array([0]));
      device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, new Uint32Array([0, 1, 1]))
      preprocess(encoder);
      encoder.copyBufferToBuffer(
      sorter.sort_info_buffer,              // source: keys_size
        0,                                    // src offset (bytes)
        indirect_buffer,                      // dest: indirect draw args
        Uint32Array.BYTES_PER_ELEMENT,        // dst offset (skip vertexCount)
        Uint32Array.BYTES_PER_ELEMENT         // size (just the instanceCount)
      );
      sorter.sort(encoder);
      render(encoder, texture_view);
    },
    camera_buffer,
    setGaussianMultiplier: (value: number) => {
      renderSettingsF32[0] = value;
      device.queue.writeBuffer(render_settings_buffer, 0, renderSettingsBlock);
    },
  };
}

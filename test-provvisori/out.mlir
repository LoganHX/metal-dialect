void main() {
  size_t v1 = 1;
  size_t v2 = 0;
  size_t v3 = 12;
  size_t v4 = 10;
  float v5 = 1.021000000e+01f;
  float* v6;
  v6 = _MetalDeviceMakeBuffer(device, false, 10, 4);
  v6[(0 * gridDim.x * gridDim.y) + (0 * gridDim.x) + v3] = v5;
  size_t v7 = 1;
  size_t v8 = 10;
  size_t_MetalCommandBufferCommit(_MetalCommandQueueMakeCommandBufferWithDefaultLibrary(queue,"main_kernel",v8,v7,v7));
  _MetalRelease(v6);
  return;
}

  kernel void main_kernel(float* v1, uint3 id [[thread_position_in_grid]], uint3 gridDim [[threads_per_grid]]) {
    size_t v2;
    v2 = id.x;
    size_t v3;
    v3 = id.y;
    size_t v4;
    v4 = id.z;



    size_t v5;
    v5 = gridDim.x;
    size_t v6;
    v6 = gridDim.y;
    size_t v7;
    v7 = gridDim.z;



    size_t v8 = 1;
    size_t v9 = 0;
    float v10 = 1.021000000e+01f;
    v1[(0 * gridDim.x * gridDim.y) + (0 * gridDim.x) + v2] = v10;
    return;
  }





void main() {
  intptr_t v1;
  v1 = _MetalDeviceMakeDefault();
  intptr_t v2;
  v2 = _MetalDeviceMakeCommandQueue(v1);
  size_t v3 = 1;
  size_t v4 = 0;
  size_t v5 = 10;
  float v6 = 1.021000000e+01f;
  float v7 = 2.110000040e+01f;
  bool v8 = false;
  int64_t v9 = 10;
  int64_t v10 = 10;
  intptr_t v11;
  v11 = _MetalDeviceMakeBuffer(v1, v8, v9 * v10);
  size_t v12 = 1;
  size_t v13 = 10;
  size_t v14 = 10;
  intptr_t v15;
  v15 = _MetalCommandQueueMakeCommandBufferWithDefaultLibrary(v2, v13, v14, v12, (int8_t *)"main_kernel");
  int64_t v16 = 0;
  _MetalCommandBufferAddBuffer(v15, v11, v16);
  _MetalCommandBufferCommit(v15);
  _MetalCommandBufferWaitUntilCompleted(v15);
  _MetalRelease(v15);
  size_t v17 = 1;
  size_t v18 = 10;
  size_t v19 = 10;
  intptr_t v20;
  v20 = _MetalCommandQueueMakeCommandBufferWithDefaultLibrary(v2, v18, v19, v17, (int8_t *)"main_kernel_0");
  int64_t v21 = 0;
  _MetalCommandBufferAddBuffer(v20, v11, v21);
  _MetalCommandBufferCommit(v20);
  _MetalCommandBufferWaitUntilCompleted(v20);
  _MetalRelease(v20);
  _MetalRelease(v11);
  return;
}

  kernel void main_kernel(device float* v1, uint3 id [[thread_position_in_grid]], uint3 gridDim [[threads_per_grid]]) {
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
    float v11;
    v11 = v1[v2 * (1) + v3 * (1 * 10)];
    v1[v2 * (1) + v3 * (1 * 10)] = v10;
    return;
  }



  kernel void main_kernel_0(device float* v1, uint3 id [[thread_position_in_grid]], uint3 gridDim [[threads_per_grid]]) {
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
    float v10 = 2.110000040e+01f;
    float v11;
    v11 = v1[v2 * (1) + v3 * (1 * 10)];
    v1[v2 * (1) + v3 * (1 * 10)] = v10;
    return;
  }





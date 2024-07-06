void main(float* v1) {
  intptr_t v2;
  v2 = _MetalDeviceMakeDefault();
  intptr_t v3;
  v3 = _MetalDeviceMakeCommandQueue(v2);
  size_t v4 = 1;
  size_t v5 = 0;
  size_t v6 = 10;
  float v7 = 1.021000000e+01f;
  float v8;
  v8 = [v5 * (1) + v5 * (1 * 10)];
  bool v9 = false;
  int64_t v10 = 10;
  int64_t v11 = 10;
  intptr_t v12;
  v12 = _MetalDeviceMakeBuffer(v2, v9, v10 * v11);
  size_t v13 = 1;
  size_t v14 = 10;
  size_t v15 = 10;
  intptr_t v16;
  v16 = _MetalCommandQueueMakeCommandBufferWithDefaultLibrary(v3, v14, v15, v13, (int8_t *)"main_kernel");
  int64_t v17 = 0;
  _MetalCommandBufferAddBuffer(v16, v12, v17);
  _MetalCommandBufferCommit(v16);
  _MetalCommandBufferWaitUntilCompleted(v16);
  _MetalRelease(v16);
  _MetalRelease(v12);
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
    v11 = [v2 * (1) + v3 * (1 * 10)];
    v1[v2 * (1) + v3 * (1 * 10)] = v10;
    return;
  }





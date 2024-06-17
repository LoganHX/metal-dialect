void main() {
  size_t v1 = 1;
  size_t v2 = 0;
  size_t v3 = 10;
  float v4 = 1.021000000e+01f;
  bool v5 = false;
  int64_t v6 = 64;
  intptr_t v7;
  v7 = _MetalDeviceMakeDefault();
  int64_t v8 = 42;
  int64_t v9 = 1;
  int64_t v10 = 1;
  intptr_t v11;
  v11 = _MetalDeviceMakeBuffer(v7, v5, v8 * v9 * v10, sizeof(float));
  size_t v12 = 1;
  size_t v13 = 10;
  int64_t v14 = (int64_t) v13;
  int64_t v15 = (int64_t) v12;
  int64_t v16 = (int64_t) v12;
  intptr_t v17;
  v17 = _MetalDeviceMakeCommandQueue(v7);
  intptr_t v18;
  v18 = _MetalCommandQueueMakeCommandBufferWithDefaultLibrary(v17, v14, v15, v16, (int8_t *)"main_kernel");
  int64_t v19 = 0;
  _MetalCommandBufferAddBuffer(v18, v11, v19);
  _MetalCommandBufferCommit(v18);
  _MetalCommandBufferWaitUntilCompleted(v18);
  _MetalRelease(v18);
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
    v1[(0 * gridDim.x * gridDim.y) + (0 * gridDim.x) + v2] = v10;
    return;
  }





void main() {
  size_t v1 = 1;
  size_t v2 = 0;
  size_t v3 = 10;
  float v4 = 1.021000000e+01f;
  bool v5 = false;
  int64_t v6 = 10;
  intptr_t v7;
  v7 = _MetalDeviceMakeDefault();
  intptr_t v8;
  v8 = _MetalDeviceMakeBuffer(v7, v5, v6);
  size_t v9 = 1;
  size_t v10 = 10;
  intptr_t v11;
  v11 = _MetalDeviceMakeCommandQueue(v7);
  intptr_t v12;
  v12 = _MetalCommandQueueMakeCommandBufferWithDefaultLibrary(v11, v10, v9, v9, (int8_t *)"main_kernel");
  int64_t v13 = 0;
  _MetalCommandBufferAddBuffer(v12, v8, v13);
  _MetalCommandBufferCommit(v12);
  _MetalCommandBufferWaitUntilCompleted(v12);
  _MetalRelease(v12);
  _MetalRelease(v8);
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
    v1[] = v10;
    return;
  }





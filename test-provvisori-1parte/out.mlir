size_t main() {
  intptr_t v1;
  v1 = _MetalDeviceMakeDefault();
  intptr_t v2;
  v2 = _MetalDeviceMakeCommandQueue(v1);
  size_t v3 = 1;
  size_t v4 = 0;
  size_t v5 = 10;
  float v6 = 1.021000000e+01f;
  bool v7 = false;
  int64_t v8 = 10;
  int64_t v9 = 10;
  intptr_t v10;
  v10 = _MetalDeviceMakeBuffer(v1, v7, v8 * v9);
  size_t v11 = 1;
  size_t v12 = 10;
  size_t v13 = 10;
  intptr_t v14;
  v14 = _MetalCommandQueueMakeCommandBufferWithDefaultLibrary(v2, v12, v13, v11, (int8_t *)"main_kernel");
  int64_t v15 = 0;
  _MetalCommandBufferAddBuffer(v14, v10, v15);
  _MetalCommandBufferCommit(v14);
  _MetalCommandBufferWaitUntilCompleted(v14);
  _MetalRelease(v14);
  _MetalRelease(v10);
  return v10;
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





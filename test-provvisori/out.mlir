void main() {
  size_t v1 = 1;
  size_t v2 = 0;
  size_t v3 = 10;
  float v4 = 1.021000000e+01f;
  float* v5;
  v5 = _MetalDeviceMakeBuffer(device, false, 10, 4);
  size_t v6 = 1;
  size_t v7 = 10;
  _MetalCommandBufferCommit(_MetalCommandQueueMakeCommandBufferWithDefaultLibrary(queue,"main_kernel",v7,v6,v6));
  _MetalRelease(v5);
  return;
}

  kernel void main_kernel(float* v1) {
    size_t v2;
    v2 = id.x;
    size_t v3;
    v3 = id.y;
    size_t v4;
    v4 = id.z;
    size_t v5;
    v5 = id.x;
    size_t v6;
    v6 = id.y;
    size_t v7;
    v7 = id.z;
    size_t v8;
    v8 = id.x;
    size_t v9;
    v9 = id.y;
    size_t v10;
    v10 = id.z;
    size_t v11;
    v11 = id.x;
    size_t v12;
    v12 = id.y;
    size_t v13;
    v13 = id.z;
    size_t v14 = 1;
    size_t v15 = 0;
    float v16 = 1.021000000e+01f;
    v1[(0 * gridSize.x * gridSize.y) + (0 * gridSize.x) + v2] = v16;
    return;
  }





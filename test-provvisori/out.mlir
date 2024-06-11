void main() {
  float v1 = 0.0e+00f;
  size_t v2 = 1;
  size_t v3 = 2;
  size_t v4 = 8;
  size_t v5 = 0;
  size_t v6 = 40000;
  float* v7;
  v7 = _MetalDeviceMakeBuffer(device, false, 1600000000, 4);
  size_t v8 = 1;
  size_t v9 = 40000;
  size_t v10 = 40000;
  _MetalCommandBufferCommit(_MetalCommandQueueMakeCommandBufferWithDefaultLibrary(queue,"main_kernel",v9,v10,v8));
  size_t v11 = 1;
  size_t v12 = 40000;
  size_t v13 = 40000;
  _MetalCommandBufferCommit(_MetalCommandQueueMakeCommandBufferWithDefaultLibrary(queue,"main_kernel_0",v12,v13,v11));
  _MetalRelease(v7);
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
    float v16 = 0.0e+00f;
    v1[(v2 * 40000 * 1) + (v3 * 1) + 0] = v16;
    return;
  }



  kernel void main_kernel_0(float* v1) {
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
    size_t v16 = 8;
    size_t v17 = v2 * v16;
    size_t v18 = v3 + v17;
    int32_t v19 = 0;
    float v20 = 0.0e+00f;
    float v21;
    v21 = [(v2 * 40000 * 1) + (v3 * 1) + 0];
    float v22 = v20 + v21;
    v1[(v2 * 40000 * 1) + (v3 * 1) + 0] = v22;
    return;
  }



void printMemrefF32() {
}



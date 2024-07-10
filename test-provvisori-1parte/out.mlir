void main() {
  intptr_t v1;
  v1 = _MetalDeviceMakeDefault();
  intptr_t v2;
  v2 = _MetalDeviceMakeCommandQueue(v1);
  bool v3 = false;
  int64_t v4 = 4;
  int64_t v5 = 4;
  intptr_t v6;
  v6 = _MetalDeviceMakeBuffer(v1, v3, v4 * v5, sizeof(float));
  bool v7 = false;
  int64_t v8 = 4;
  int64_t v9 = 4;
  intptr_t v10;
  v10 = _MetalDeviceMakeBuffer(v1, v7, v8 * v9, sizeof(float));
  bool v11 = false;
  int64_t v12 = 4;
  int64_t v13 = 4;
  intptr_t v14;
  v14 = _MetalDeviceMakeBuffer(v1, v11, v12 * v13, sizeof(float));
  float v15 = 3.000000000e+00f;
  float v16 = 1.000000000e+00f;
  float v17 = 2.000000000e+00f;
  size_t v18 = 0;
  size_t v19 = 4;
  size_t v20 = 1;
  for (size_t v21 = v18; v21 < v19; v21 += v20) {
    for (size_t v22 = v18; v22 < v19; v22 += v20) {
      _MetalStore_float(v6, v21 * (1) + v22 * (1 * v5), v16);
      _MetalStore_float(v10, v21 * (1) + v22 * (1 * v9), v17);
    };
  }
  uint32_t v23 = 32;
  intptr_t v24;
  v24 = _MetalMatMul(v2, v6, v4, v5, v10, v8, v9, v14, v23);
  _MetalCommandBufferCommit(v24);
  _MetalCommandBufferWaitUntilCompleted(v24);
  return;
}



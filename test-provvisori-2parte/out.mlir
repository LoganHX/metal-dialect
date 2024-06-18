void main() {
  size_t v1 = 1;
  size_t v2 = 32;
  size_t v3 = 0;
  int32_t v4 = 0;
  bool v5 = false;
  intptr_t v6;
  v6 = _MetalDeviceMakeDefault();
  int64_t v7 = 32;
  int64_t v8 = 32;
  int64_t v9 = 1;
  intptr_t v10;
  v10 = _MetalDeviceMakeBuffer(v6, v5, v7 * v8 * v9, sizeof(int32_t));
  for (size_t v11 = v3; v11 < v2; v11 += v1) {
    for (size_t v12 = v3; v12 < v2; v12 += v1) {
      uint32_t v13 = 0;
      uint32_t v14 = 32;
      uint32_t v15 = 32;
      uint32_t v16 = 1;
      _MetalStore_int32_t(v10, (v13 * v14 * v15) + (v12 * v14) + v11, v4);
    };
  }
  bool v17 = false;
  int64_t v18 = 32;
  int64_t v19 = 32;
  int64_t v20 = 1;
  intptr_t v21;
  v21 = _MetalDeviceMakeBuffer(v6, v17, v18 * v19 * v20, sizeof(int32_t));
  for (size_t v22 = v3; v22 < v2; v22 += v1) {
    for (size_t v23 = v3; v23 < v2; v23 += v1) {
      uint32_t v24 = 0;
      uint32_t v25 = 32;
      uint32_t v26 = 32;
      uint32_t v27 = 1;
      _MetalStore_int32_t(v21, (v24 * v25 * v26) + (v23 * v25) + v22, v4);
    };
  }
  bool v28 = false;
  int64_t v29 = 32;
  int64_t v30 = 32;
  int64_t v31 = 1;
  intptr_t v32;
  v32 = _MetalDeviceMakeBuffer(v6, v28, v29 * v30 * v31, sizeof(int32_t));
  uint32_t v33 = 32;
  return;
}



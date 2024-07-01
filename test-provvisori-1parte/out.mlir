void main() {
  size_t v1 = 1;
  size_t v2 = 7;
  size_t v3 = 10;
  float v4 = 1.021000000e+01f;
  bool v5 = false;
  int64_t v6 = 40;
  int64_t v7 = 41;
  int64_t v8 = 42;
  intptr_t v9;
  v9 = _MetalDeviceMakeDefault();
  intptr_t v10;
  v10 = _MetalDeviceMakeBuffer(v9, v5, v6, v7, v8);
  _MetalStore_float(v10, v3 * (1) + v2 * (1 * v8) + v1 * (1 * v8 * v7), v4);
  float v11;
  v11 = _MetalLoad_float(v10, v3 * (1) + v1 * (1 * v8) + v2 * (1 * v8 * v7));
  _MetalRelease(v10);
  return;
}



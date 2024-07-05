void main() {
  intptr_t v1;
  v1 = _MetalDeviceMakeDefault();
  intptr_t v2;
  v2 = _MetalDeviceMakeCommandQueue(v1);
  size_t v3 = 1;
  float v4 = 1.021000000e+01f;
  bool v5 = false;
  int64_t v6 = 40;
  int64_t v7 = 41;
  int64_t v8 = 42;
  intptr_t v9;
  v9 = _MetalDeviceMakeBuffer(v1, v5, v6 * v7 * v8);
  _MetalStore_float(v9, v3 * (1) + v3 * (1 * v8) + v3 * (1 * v8 * v7), v4);
  float v10;
  v10 = _MetalLoad_float(v9, v3 * (1) + v3 * (1 * v8) + v3 * (1 * v8 * v7));
  _MetalRelease(v9);
  return;
}



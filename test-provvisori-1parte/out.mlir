void main() {
  size_t v1 = 1;
  size_t v2 = 7;
  size_t v3 = 10;
  float v4 = 1.021000000e+01f;
  bool v5 = false;
  int64_t v6 = 42;
  int64_t v7 = 42;
  intptr_t v8;
  v8 = _MetalDeviceMakeDefault();
  intptr_t v9;
  v9 = _MetalDeviceMakeBuffer(v8, v5, v6, v7);
  _MetalStore_float(v9, v1, v2, v6, v7, v4);
  float v10;
  v10 = _MetalLoad_float(v9, v2, v1, v6, v7);
  _MetalRelease(v9);
  return;
}



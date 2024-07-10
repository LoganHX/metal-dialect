void main() {
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
  intptr_t v9;
  v9 = _MetalDeviceMakeBuffer(v1, v7, v8);

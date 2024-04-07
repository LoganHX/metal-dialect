kernel void life(
  constant int32_t *v0 [[buffer(0)]],
  constant int32_t *v1 [[buffer(1)]],
  device int32_t *v2 [[buffer(2)]],
  uint3 id [[thread_position_in_grid]])
{  
  int32_t v3[1];
  v3[0] = int32_t(id.x);
  int32_t v4[1];
  v4[0] = int32_t(id.y);
  int32_t v5[1];
  int32_t v6[1];
  v5[0] = v1[0];
  v6[0] = v1[1];
  int32_t v7[1];
  v7[0] = 0;
  int32_t v8[1];
  v8[0] = (v3[0]) - (1);
  while ((v8[0]) <= ((v3[0]) + (1))) {    
    int32_t v9[1];
    v9[0] = (v4[0]) - (1);
    while ((v9[0]) <= ((v4[0]) + (1))) {      
      if (((((((v8[0]) >= (0)) && ((v8[0]) < (v5[0]))) && ((v9[0]) >= (0))) && ((v9[0]) < (v6[0]))) && (((v8[0]) != (v3[0])) || ((v9[0]) != (v4[0])))) && (bool(v0[uint32_t(((v8[0]) * (v6[0])) + (v9[0]))]))) {        
        v7[0] = (v7[0]) + (1);
      }
      v9[0] = (v9[0]) + (1);
    }
    v8[0] = (v8[0]) + (1);
  }
  int32_t v10[1];
  v10[0] = ((v3[0]) * (v6[0])) + (v4[0]);
  if (bool(v0[uint32_t(v10[0])])) {    
    if (((v7[0]) == (2)) || ((v7[0]) == (3))) {      
      v2[uint32_t(v10[0])] = 1;
    } else {      
      v2[uint32_t(v10[0])] = 0;
    }
  } else {    
    if ((v7[0]) == (3)) {      
      v2[uint32_t(v10[0])] = 1;
    } else {      
      v2[uint32_t(v10[0])] = 0;
    }
  }
  return;
}


module {
  func.func @main() {
    %memref = gpu.alloc  () : memref<4x4xf32>
    %memref_0 = gpu.alloc  () : memref<4x4xf32>
    %memref_1 = gpu.alloc  () : memref<4x4xf32>
    %0 = "emitc.constant"() <{value = 3.000000e+00 : f32}> : () -> f32
    %1 = "emitc.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
    %2 = "emitc.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
    %3 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %4 = "emitc.constant"() <{value = 4 : index}> : () -> index
    %5 = "emitc.constant"() <{value = 1 : index}> : () -> index
    emitc.for %arg0 = %3 to %4 step %5 {
      emitc.for %arg1 = %3 to %4 step %5 {
        memref.store %0, %memref[%arg0, %arg1] : memref<4x4xf32>
        memref.store %2, %memref_0[%arg0, %arg1] : memref<4x4xf32>
      }
    }
    %6 = "shader.matsum"(%memref, %memref_0, %memref_1) <{elementType = f32, operandSegmentSizes = array<i32: 0, 1, 0, 0, 1, 0, 0, 1>}> : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> index
    return
  }
}


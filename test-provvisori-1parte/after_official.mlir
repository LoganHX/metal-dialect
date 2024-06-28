module {
  func.func @main() {
    %0 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %1 = "emitc.constant"() <{value = 7 : index}> : () -> index
    %2 = "emitc.constant"() <{value = 10 : index}> : () -> index
    %3 = "emitc.constant"() <{value = 1.021000e+01 : f32}> : () -> f32
    %alloc = memref.alloc() : memref<42x42xf32>
    memref.store %3, %alloc[%0, %1] : memref<42x42xf32>
    %4 = memref.load %alloc[%1, %0] : memref<42x42xf32>
    memref.dealloc %alloc : memref<42x42xf32>
    return
  }
}


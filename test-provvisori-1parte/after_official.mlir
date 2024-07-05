module {
  func.func @main() {
    %0 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %1 = "emitc.constant"() <{value = 1.021000e+01 : f32}> : () -> f32
    %alloc = memref.alloc() : memref<40x41x42xf32>
    memref.store %1, %alloc[%0, %0, %0] : memref<40x41x42xf32>
    %2 = memref.load %alloc[%0, %0, %0] : memref<40x41x42xf32>
    memref.dealloc %alloc : memref<40x41x42xf32>
    return
  }
}


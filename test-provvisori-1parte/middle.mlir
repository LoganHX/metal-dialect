module {
  func.func @main() {
    %0 = metal.device_make_default : index
    %1 = metal.device_make_command_queue %0 : (index) -> index
    %2 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %3 = "emitc.constant"() <{value = 7 : index}> : () -> index
    %4 = "emitc.constant"() <{value = 10 : index}> : () -> index
    %5 = "emitc.constant"() <{value = 1.021000e+01 : f32}> : () -> f32
    %alloc = memref.alloc() : memref<40x41x42xf32>
    memref.store %5, %alloc[%2, %3, %4] : memref<40x41x42xf32>
    %6 = memref.load %alloc[%3, %2, %4] : memref<40x41x42xf32>
    memref.dealloc %alloc : memref<40x41x42xf32>
    return
  }
}


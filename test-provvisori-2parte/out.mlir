module {
  func.func @main() {
    %0 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %1 = "emitc.constant"() <{value = 32 : index}> : () -> index
    %2 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %3 = "emitc.constant"() <{value = 0 : si32}> : () -> si32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xsi32>
    emitc.for %arg0 = %2 to %1 step %0 {
      emitc.for %arg1 = %2 to %1 step %0 {
        memref.store %3, %alloc[%arg0, %arg1] : memref<32x32xsi32>
      }
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xsi32>
    emitc.for %arg0 = %2 to %1 step %0 {
      emitc.for %arg1 = %2 to %1 step %0 {
        memref.store %3, %alloc_0[%arg0, %arg1] : memref<32x32xsi32>
      }
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32x32xsi32>
    emitc.for %arg0 = %2 to %1 step %0 {
      emitc.for %arg1 = %2 to %1 step %0 {
        emitc.for %arg2 = %2 to %1 step %0 {
          %4 = memref.load %alloc[%arg0, %arg2] : memref<32x32xsi32>
          %5 = memref.load %alloc_0[%arg2, %arg1] : memref<32x32xsi32>
          %6 = memref.load %alloc_1[%arg0, %arg1] : memref<32x32xsi32>
          %7 = emitc.cast %4 : si32 to ui32
          %8 = emitc.cast %5 : si32 to ui32
          %9 = emitc.mul %7, %8 : (ui32, ui32) -> ui32
          %10 = emitc.cast %9 : ui32 to i32
          %11 = emitc.cast %6 : si32 to ui32
          %12 = emitc.cast %10 : i32 to ui32
          %13 = emitc.add %11, %12 : (ui32, ui32) -> ui32
          %14 = emitc.cast %13 : ui32 to si32
          memref.store %14, %alloc_1[%arg0, %arg1] : memref<32x32xsi32>
        }
      }
    }
    return
  }
}


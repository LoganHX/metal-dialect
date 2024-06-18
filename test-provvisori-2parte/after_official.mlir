  func.func @main() {
    %0 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %1 = "emitc.constant"() <{value = 32 : index}> : () -> index
    %2 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %3 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
    emitc.for %arg0 = %2 to %1 step %0 {
      emitc.for %arg1 = %2 to %1 step %0 {
        memref.store %3, %alloc[%arg0, %arg1] : memref<32x32xi32>
      }
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
    emitc.for %arg0 = %2 to %1 step %0 {
      emitc.for %arg1 = %2 to %1 step %0 {
        memref.store %3, %alloc_0[%arg0, %arg1] : memref<32x32xi32>
      }
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
    linalg.matmul ins(%alloc, %alloc_0 : memref<32x32xi32>, memref<32x32xi32>) outs(%alloc_1 : memref<32x32xi32>)

    return
  }



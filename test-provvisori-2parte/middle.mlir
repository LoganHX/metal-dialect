module {
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
    emitc.for %arg0 = %c0 to %c32 step %c1 {
      emitc.for %arg1 = %c0 to %c32 step %c1 {
        memref.store %c0_i32, %alloc[%arg0, %arg1] : memref<32x32xi32>
      }
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
    emitc.for %arg0 = %c0 to %c32 step %c1 {
      emitc.for %arg1 = %c0 to %c32 step %c1 {
        memref.store %c0_i32, %alloc_0[%arg0, %arg1] : memref<32x32xi32>
      }
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
    emitc.for %arg0 = %c0 to %c32 step %c1 {
      emitc.for %arg1 = %c0 to %c32 step %c1 {
        emitc.for %arg2 = %c0 to %c32 step %c1 {
          %0 = memref.load %alloc[%arg0, %arg2] : memref<32x32xi32>
          %1 = memref.load %alloc_0[%arg2, %arg1] : memref<32x32xi32>
          %2 = memref.load %alloc_1[%arg0, %arg1] : memref<32x32xi32>
          %3 = arith.muli %0, %1 : i32
          %4 = arith.addi %2, %3 : i32
          memref.store %4, %alloc_1[%arg0, %arg1] : memref<32x32xi32>
        }
      }
    }
    return
  }
}


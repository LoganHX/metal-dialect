
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 3 : i32

    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
    scf.for %arg0 = %c0 to %c32 step %c1 {
      scf.for %arg1 = %c0 to %c32 step %c1 {
        memref.store %c0_i32, %alloc[%arg0, %arg1] : memref<32x32xi32>
      }
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
    scf.for %arg0 = %c0 to %c32 step %c1 {
      scf.for %arg1 = %c0 to %c32 step %c1 {
        memref.store %c1_i32, %alloc_0[%arg0, %arg1] : memref<32x32xi32>
      }
    }
    
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
    linalg.matmul ins(%alloc, %alloc_0 : memref<32x32xi32>, memref<32x32xi32>) outs(%alloc_1 : memref<32x32xi32>)
    return
  
}


func.func @main() {
  %c0_i32 = arith.constant 0 : i32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x4xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<4x4xi32>)
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<4x4xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc_0 : memref<4x4xi32>)
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x4xi32>
  
  linalg.matmul ins(%alloc, %alloc_0 : memref<4x4xi32>, memref<4x4xi32>) outs(%alloc_1 : memref<4x4xi32>)

  return

}

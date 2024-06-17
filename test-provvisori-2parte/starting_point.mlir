// module attributes {torch.debug_module_name = "model"} {
//   memref.global "private" @global_seed : memref<i64> = dense<0>
//   func.func @forward(%arg0: memref<32x32xi32>, %arg1: memref<32x32xi32>, %arg2: memref<32x32xi32>) -> memref<32x32xi32> {
//     %c0_i32 = arith.constant 0 : i32
//     %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
//     linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<32x32xi32>)
//     %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
//     memref.copy %alloc, %alloc_0 : memref<32x32xi32> to memref<32x32xi32>
//     linalg.matmul ins(%arg0, %arg1 : memref<32x32xi32>, memref<32x32xi32>) outs(%alloc_0 : memref<32x32xi32>)
//     %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
//     memref.copy %alloc, %alloc_1 : memref<32x32xi32> to memref<32x32xi32>
//     linalg.matmul ins(%alloc_0, %arg2 : memref<32x32xi32>, memref<32x32xi32>) outs(%alloc_1 : memref<32x32xi32>)
//     return %alloc_1 : memref<32x32xi32>
//   }
// }

func.func @main() {
  %c0_i32 = arith.constant 0 : i32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<32x32xi32>)
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc_0 : memref<32x32xi32>)
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
  
  linalg.matmul ins(%alloc, %alloc_0 : memref<32x32xi32>, memref<32x32xi32>) outs(%alloc_1 : memref<32x32xi32>)

  return

}

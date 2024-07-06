
  func.func @main(%arg0: memref<10x10xf32>) -> () {
  %step = arith.constant 1: index
  %c1 = arith.constant 0 : index
  %c2 = arith.constant 10 : index
  %value = arith.constant 10.21 : f32

  %A = memref.alloc() : memref<10x10xf32>
  scf.parallel (%i, %j) = (%c1, %c1) to (%c2, %c2) step (%step, %step) {
    %t = memref.load %A[%j, %i] : memref<10x10xf32>
    memref.store %value, %A[%j, %i] : memref<10x10xf32>
  }
  memref.dealloc %A : memref<10x10xf32>

  func.return
}


// func.func @main() {
//   %c1 = arith.constant 1 : index
//   %value = arith.constant 10.21 : f32
  
//   %A = memref.alloc() : memref<40x41x42xf32>
//   memref.store %value, %A[%c1, %c1, %c1] : memref<40x41x42xf32>
//   %v = memref.load %A[%c1, %c1, %c1] : memref<40x41x42xf32>
//   memref.dealloc %A : memref<40x41x42xf32>

//   return
// }

// func.func @main() {
//    %a = memref.alloc() : memref<10x10xi32>
//    %b = memref.alloc() : memref<10x10xi32>
//    %c = memref.alloc() : memref<10x10xi32>

//   linalg.matmul 
//     ins(%a, %b: memref<10x10xi32>, memref<10x10xi32>)
//     outs(%c:memref<10x10xi32>)
//   return
// }
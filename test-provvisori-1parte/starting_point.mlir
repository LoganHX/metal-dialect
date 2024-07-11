func.func @main() {
  %A = memref.alloc() : memref<4x4xf32>
  %B = memref.alloc() : memref<4x4xf32>
  %C = memref.alloc() : memref<4x4xf32>

  %cst0 = arith.constant 3.0 : f32
  %cst1 = arith.constant 1.0 : f32
  %cst2 = arith.constant 2.0 : f32

  %start = arith.constant 0 : index
  %end = arith.constant 4 : index
  %step = arith.constant 1 : index

  scf.for %i = %start to %end step %step {
    scf.for %j = %start to %end step %step {
      memref.store %cst0, %A[%i, %j] : memref<4x4xf32>
      memref.store %cst2, %B[%i, %j] : memref<4x4xf32>
    }
  }

  linalg.add ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>) 
                outs(%C : memref<4x4xf32>)
  
  func.return
}

// func.func @main() {
//   %step = arith.constant 1: index
//   %c1 = arith.constant 0 : index
//   %c2 = arith.constant 10 : index
//   %value = arith.constant 10.21 : f32
  
//   %A = memref.alloc() : memref<10x10xf32>
//   %B = memref.alloc() : memref<10xf32>
//   scf.parallel (%i, %j) = (%c1, %c1) to (%c2, %c2) step (%step, %step) {
//     memref.store %value, %A[%i, %j] : memref<10x10xf32>
//   }

//   memref.store %value, %B[%c1] : memref<10xf32>

//   memref.dealloc %B : memref<10xf32>
//   memref.dealloc %A : memref<10x10xf32>
//   func.return
// }




// func.func @main() {
//   %step = arith.constant 1 : index
//   %c1 = arith.constant 0 : index
//   %c2 = arith.constant 10 : index
//   %valueGPU = arith.constant 10.21 : f32
//   %valueCPU = arith.constant 21.10 : f32
//   %CPU = memref.alloc() : memref<10xf32>
//   scf.for %j = %c1 to %c2 step %step {
//     memref.store %valueCPU, %CPU[%j] : memref<10xf32>
//   }
//   %GPU = memref.alloc() : memref<10xf32>

//   scf.parallel (%i) = (%c1) to (%c2) step (%step) {
//     memref.store %valueGPU, %GPU[%i] : memref<10xf32>
//   }
//   %t = memref.load %CPU[%c1]: memref<10xf32>
//   %s = memref.load %GPU[%c1]: memref<10xf32>

//   memref.dealloc %GPU : memref<10xf32>

//   func.return
// }






  func.func @main() {
    %step = arith.constant 1 : index
    %c1 = arith.constant 0 : index
    %c2 = arith.constant 10 : index
    %valueGPU = arith.constant 10.21 : f32
    %valueCPU = arith.constant 10.21 : f32
    %arg0 = memref.alloc() : memref<10xf32>
    scf.for %j = %c1 to %c2 step %step {
      memref.store %valueCPU, %arg0[%j] : memref<10xf32>
    }

    %GPU = memref.alloc() : memref<10xf32>

    scf.parallel (%i) = (%c1) to (%c2) step (%step) {
      memref.store %valueGPU, %GPU[%i] : memref<10xf32>
    }

    memref.dealloc %GPU : memref<10xf32>

    func.return
  }

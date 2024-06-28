module {
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c10 = arith.constant 10 : index
    %cst = arith.constant 1.021000e+01 : f32
    %alloc = memref.alloc() : memref<42x42xf32>
    memref.store %cst, %alloc[%c1, %c7] : memref<42x42xf32>
    %0 = memref.load %alloc[%c7, %c1] : memref<42x42xf32>
    memref.dealloc %alloc : memref<42x42xf32>
    return
  }
}


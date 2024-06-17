
func.func @main() {
  %step = arith.constant 1: index
  %c1 = arith.constant 0 : index
  %c2 = arith.constant 10 : index
  %value = arith.constant 10.21 : f32
  
  %A = memref.alloc() : memref<42xf32>

  scf.parallel (%i) = (%c1) to (%c2) step (%step) {
    memref.store %value, %A[%i] : memref<42xf32>
  }

  memref.dealloc %A : memref<42xf32>

  return
}

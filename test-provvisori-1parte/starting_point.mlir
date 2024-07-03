
func.func @main() {
  %step = arith.constant 1: index
  %c1 = arith.constant 7 : index
  %c2 = arith.constant 10 : index
  %value = arith.constant 10.21 : f32
  
  %A = memref.alloc() : memref<40x41x42xf32>
  memref.store %value, %A[%step, %c1, %c2] : memref<40x41x42xf32>
  %pollo = memref.load %A[%c1, %step, %c2] : memref<40x41x42xf32>
  

  memref.dealloc %A : memref<40x41x42xf32>

  return
}

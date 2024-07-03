module {
  func.func @main() {
    %alloc = memref.alloc() : memref<10x10xi32>
    %alloc_0 = memref.alloc() : memref<10x10xi32>
    %alloc_1 = memref.alloc() : memref<10x10xi32>
    linalg.matmul ins(%alloc, %alloc_0 : memref<10x10xi32>, memref<10x10xi32>) outs(%alloc_1 : memref<10x10xi32>)
    return
  }
}


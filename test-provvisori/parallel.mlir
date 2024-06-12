module attributes {gpu.container_module} {
  func.func @main() {
    %0 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %1 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %2 = "emitc.constant"() <{value = 12 : index}> : () -> index
    %3 = "emitc.constant"() <{value = 10 : index}> : () -> index
    %4 = "emitc.constant"() <{value = 1.021000e+01 : f32}> : () -> f32
    %alloc = memref.alloc() : memref<10xf32>
    memref.store %4, %alloc[%2] : memref<10xf32>
    %5 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %6 = "emitc.constant"() <{value = 10 : index}> : () -> index
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%6, %5, %5) threads in (%5, %5, %5)  args(%alloc : memref<10xf32>)
    memref.dealloc %alloc : memref<10xf32>
    return
  }
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<10xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = "emitc.constant"() <{value = 1 : index}> : () -> index
      %1 = "emitc.constant"() <{value = 0 : index}> : () -> index
      %2 = "emitc.constant"() <{value = 1.021000e+01 : f32}> : () -> f32
      memref.store %2, %arg0[%block_id_x] : memref<10xf32>
      gpu.return
    }
  }
}


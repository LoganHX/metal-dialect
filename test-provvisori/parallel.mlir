module attributes {gpu.container_module} {
  func.func @parallel_loop() {
    %0 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %1 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %2 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %3 = "emitc.constant"() <{value = 21 : index}> : () -> index
    %4 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %5 = "emitc.constant"() <{value = 21 : index}> : () -> index
    %6 = "emitc.constant"() <{value = 21 : index}> : () -> index
    gpu.launch_func  @parallel_loop_kernel::@parallel_loop_kernel blocks in (%5, %6, %4) threads in (%4, %4, %4)  args(%1 : index, %2 : index)
    return
  }
  gpu.module @parallel_loop_kernel {
    gpu.func @parallel_loop_kernel(%arg0: index, %arg1: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 21, 21, 1>} {
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
      %0 = emitc.mul %block_id_x, %arg0 : (index, index) -> index
      %1 = emitc.add %0, %arg1 : (index, index) -> index
      %2 = emitc.mul %block_id_y, %arg0 : (index, index) -> index
      %3 = emitc.add %2, %arg1 : (index, index) -> index
      %4 = emitc.add %arg0, %arg0 : (index, index) -> index
      gpu.return
    }
  }
}

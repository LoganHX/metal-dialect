module attributes {gpu.container_module} {
  func.func @parallel_loop() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c21 = arith.constant 21 : index
    %c1_1 = arith.constant 1 : index
    %c21_2 = arith.constant 21 : index
    %c21_3 = arith.constant 21 : index
    gpu.launch_func  @parallel_loop_kernel::@parallel_loop_kernel blocks in (%c21_2, %c21_3, %c1_1) threads in (%c1_1, %c1_1, %c1_1)  args(%c1 : index, %c0_0 : index)
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
      %0 = arith.muli %block_id_x, %arg0 : index
      %1 = arith.addi %0, %arg1 : index
      %2 = arith.muli %block_id_y, %arg0 : index
      %3 = arith.addi %2, %arg1 : index
      %4 = arith.addi %arg0, %arg0 : index
      gpu.return
    }
  }
}
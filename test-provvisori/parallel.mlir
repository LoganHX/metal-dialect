module attributes {gpu.container_module} {
  func.func @main() {
    %0 = "emitc.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %1 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %2 = "emitc.constant"() <{value = 2 : index}> : () -> index
    %3 = "emitc.constant"() <{value = 8 : index}> : () -> index
    %4 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %5 = "emitc.constant"() <{value = 40000 : index}> : () -> index
    %alloc = memref.alloc() : memref<40000x40000xf32>
    %6 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %7 = "emitc.constant"() <{value = 40000 : index}> : () -> index
    %8 = "emitc.constant"() <{value = 40000 : index}> : () -> index
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%7, %8, %6) threads in (%6, %6, %6)  
    %9 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %10 = "emitc.constant"() <{value = 40000 : index}> : () -> index
    %11 = "emitc.constant"() <{value = 40000 : index}> : () -> index
    gpu.launch_func  @main_kernel_0::@main_kernel blocks in (%10, %11, %9) threads in (%9, %9, %9)  
    memref.dealloc %alloc : memref<40000x40000xf32>
    return
  }
  gpu.module @main_kernel {
    gpu.func @main_kernel() kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
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
      gpu.return
    }
  }
  gpu.module @main_kernel_0 {
    gpu.func @main_kernel() kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
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
      %2 = "emitc.constant"() <{value = 8 : index}> : () -> index
      %3 = emitc.mul %block_id_x, %2 : (index, index) -> index
      %4 = emitc.add %block_id_y, %3 : (index, index) -> index
      %5 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %6 = "emitc.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      gpu.return
    }
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}


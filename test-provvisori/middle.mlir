module attributes {gpu.container_module} {
  func.func @main() {
    %0 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %1 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %2 = "emitc.constant"() <{value = 10 : index}> : () -> index
    %3 = "emitc.constant"() <{value = 1.021000e+01 : f32}> : () -> f32
    %4 = "emitc.constant"() <{value = false}> : () -> i1
    %5 = "emitc.constant"() <{value = 64 : i64}> : () -> i64
    %6 = metal.device_make_default : index
    %7 = "emitc.constant"() <{value = 42 : i64}> : () -> i64
    %8 = "emitc.constant"() <{value = 1 : i64}> : () -> i64
    %9 = "emitc.constant"() <{value = 1 : i64}> : () -> i64
    %10 = metal.device_make_buffer %6, %4, %7, %8, %9, "float" : (index, i1, i64, i64, i64) -> index
    %11 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %12 = "emitc.constant"() <{value = 10 : index}> : () -> index
    %13 = emitc.cast %12 : index to i64
    %14 = emitc.cast %11 : index to i64
    %15 = emitc.cast %11 : index to i64
    %16 = metal.device_make_command_queue %6 : (index) -> index
    %17 = metal.command_queue_make_command_buffer main_kernel %16, %13, %14, %15: (index, i64, i64, i64) -> index
    %18 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    metal.command_buffer_add_buffer %17, %10, %18 : (index, index, i64) -> ()
    metal.command_buffer_commit %17 : index
    metal.command_buffer_wait_until_completed %17 : index
    metal.release %17 : index
    metal.release %10 : index
    return
  }
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<42xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
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
      memref.store %2, %arg0[%block_id_x] : memref<42xf32>
      gpu.return
    }
  }
}


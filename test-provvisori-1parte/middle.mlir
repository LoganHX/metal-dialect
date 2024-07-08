module attributes {gpu.container_module} {
  func.func @main() {
    %0 = metal.device_make_default : index
    %1 = metal.device_make_command_queue %0 : (index) -> index
    %2 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %3 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %4 = "emitc.constant"() <{value = 10 : index}> : () -> index
    %5 = "emitc.constant"() <{value = 1.021000e+01 : f32}> : () -> f32
    %6 = "emitc.constant"() <{value = 2.110000e+01 : f32}> : () -> f32
    %7 = "emitc.constant"() <{value = false}> : () -> i1
    %8 = "emitc.constant"() <{value = 10 : i64}> : () -> i64
    %9 = "emitc.constant"() <{value = 10 : i64}> : () -> i64
    %10 = metal.device_make_buffer %0, %7, %8, %9 x f32 : (index, i1, i64, i64) -> index
    %11 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %12 = "emitc.constant"() <{value = 10 : index}> : () -> index
    %13 = "emitc.constant"() <{value = 10 : index}> : () -> index
    %14 = metal.command_queue_make_command_buffer main_kernel %1, %12, %13, %11: (index, index, index, index) -> index
    %15 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    metal.command_buffer_add_buffer %14, %10, %15 : (index, index, i64) -> ()
    metal.command_buffer_commit %14 : index
    metal.command_buffer_wait_until_completed %14 : index
    metal.release %14 : index
    %16 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %17 = "emitc.constant"() <{value = 10 : index}> : () -> index
    %18 = "emitc.constant"() <{value = 10 : index}> : () -> index
    %19 = metal.command_queue_make_command_buffer main_kernel_0 %1, %17, %18, %16: (index, index, index, index) -> index
    %20 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    metal.command_buffer_add_buffer %19, %10, %20 : (index, index, i64) -> ()
    metal.command_buffer_commit %19 : index
    metal.command_buffer_wait_until_completed %19 : index
    metal.release %19 : index
    metal.release %10 : index
    return
  }
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<10x10xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
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
      %3 = memref.load %arg0[%block_id_y, %block_id_x] : memref<10x10xf32>
      memref.store %2, %arg0[%block_id_y, %block_id_x] : memref<10x10xf32>
      gpu.return
    }
  }
  gpu.module @main_kernel_0 {
    gpu.func @main_kernel(%arg0: memref<10x10xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
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
      %2 = "emitc.constant"() <{value = 2.110000e+01 : f32}> : () -> f32
      %3 = memref.load %arg0[%block_id_y, %block_id_x] : memref<10x10xf32>
      memref.store %2, %arg0[%block_id_y, %block_id_x] : memref<10x10xf32>
      gpu.return
    }
  }
}


module attributes {gpu.container_module} {
  func.func @main() {
    %0 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %1 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %2 = "emitc.constant"() <{value = 10 : index}> : () -> index
    %3 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %4 = "emitc.constant"() <{value = 10 : index}> : () -> index
    %5 = emitc.cast %4 : index to i64
    %6 = emitc.cast %3 : index to i64
    %7 = emitc.cast %3 : index to i64
    %8 = metal.device_make_default : index
    %9 = metal.device_make_command_queue %8 : (index) -> index
    %10 = metal.command_queue_make_command_buffer main_kernel %9, %5, %6, %7: (index, i64, i64, i64) -> index
    metal.command_buffer_commit %10 : index
    metal.command_buffer_wait_until_completed %10 : index
    metal.release %10 : index
    return
  }
    }
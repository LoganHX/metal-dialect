module {
func.func @main() {
  %0 = metal.device_make_default : index
  %1 = metal.device_make_command_queue %0 : (index) -> index
  %2 = "emitc.constant"() <{value = false}> : () -> i1
  %3 = "emitc.constant"() <{value = 4 : i64}> : () -> i64
  %4 = "emitc.constant"() <{value = 4 : i64}> : () -> i64
  %5 = metal.device_make_buffer %0, %2, %3, %4 x f32 : (index, i1, i64, i64) -> index
  %6 = "emitc.constant"() <{value = false}> : () -> i1
  %7 = "emitc.constant"() <{value = 4 : i64}> : () -> i64
  %8 = "emitc.constant"() <{value = 4 : i64}> : () -> i64
  %9 = metal.device_make_buffer %0, %6, %7, %8 x f32 : (index, i1, i64, i64) -> index
  %10 = "emitc.constant"() <{value = false}> : () -> i1
  %11 = "emitc.constant"() <{value = 4 : i64}> : () -> i64
  %12 = "emitc.constant"() <{value = 4 : i64}> : () -> i64
  %13 = metal.device_make_buffer %0, %10, %11, %12 x f32 : (index, i1, i64, i64) -> index
  %14 = "emitc.constant"() <{value = 3.000000e+00 : f32}> : () -> f32
  %15 = "emitc.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
  %16 = "emitc.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
  %17 = "emitc.constant"() <{value = 0 : index}> : () -> index
  %18 = "emitc.constant"() <{value = 4 : index}> : () -> index
  %19 = "emitc.constant"() <{value = 1 : index}> : () -> index
  emitc.for %arg0 = %17 to %18 step %19 {
    emitc.for %arg1 = %17 to %18 step %19 {
      metal.store %14, %5, %arg0, %arg1[%3, %4] : f32, index, index, index, i64, i64
      metal.store %16, %9, %arg0, %arg1[%7, %8] : f32, index, index, index, i64, i64
    }
  }
  %20 = "metal.matmul"(%1, %5, %3, %4, %9, %7, %8, %13) <{elementType = f32, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1>}> : (index, index, i64, i64, index, i64, i64, index) -> index
  metal.command_buffer_commit %20 : index
  metal.command_buffer_wait_until_completed %20 : index
  metal.release %20 : index
  return
}
}


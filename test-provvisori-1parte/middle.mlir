module {
  func.func @main() {
    %0 = metal.device_make_default : index
    %1 = metal.device_make_command_queue %0 : (index) -> index
    %2 = "emitc.constant"() <{value = false}> : () -> i1
    %3 = "emitc.constant"() <{value = 10 : i64}> : () -> i64
    %4 = "emitc.constant"() <{value = 10 : i64}> : () -> i64
    %5 = metal.device_make_buffer %0, %2, %3, %4 x i32 : (index, i1, i64, i64) -> index
    %6 = "emitc.constant"() <{value = false}> : () -> i1
    %7 = "emitc.constant"() <{value = 10 : i64}> : () -> i64
    %8 = "emitc.constant"() <{value = 10 : i64}> : () -> i64
    %9 = metal.device_make_buffer %0, %6, %7, %8 x i32 : (index, i1, i64, i64) -> index
    %10 = "emitc.constant"() <{value = false}> : () -> i1
    %11 = "emitc.constant"() <{value = 10 : i64}> : () -> i64
    %12 = "emitc.constant"() <{value = 10 : i64}> : () -> i64
    %13 = metal.device_make_buffer %0, %10, %11, %12 x i32 : (index, i1, i64, i64) -> index
    %14 = "emitc.constant"() <{value = 32 : ui32}> : () -> ui32
    "metal.matmul"(%1, %5, %3, %4, %9, %7, %8, %13, %14) <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1>}> : (index, index, i64, i64, index, i64, i64, index, ui32) -> ()
    return
  }
}


module {
  func.func @main() {
    %0 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %1 = "emitc.constant"() <{value = 32 : index}> : () -> index
    %2 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %3 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %4 = "emitc.constant"() <{value = false}> : () -> i1
    %5 = metal.device_make_default : index
    %6 = "emitc.constant"() <{value = 32 : i64}> : () -> i64
    %7 = "emitc.constant"() <{value = 32 : i64}> : () -> i64
    %8 = "emitc.constant"() <{value = 1 : i64}> : () -> i64
    %9 = metal.device_make_buffer %5, %4, %6, %7, %8, "int32_t" : (index, i1, i64, i64, i64) -> index
    emitc.for %arg0 = %2 to %1 step %0 {
      emitc.for %arg1 = %2 to %1 step %0 {
        %22 = "emitc.constant"() <{value = 0 : ui32}> : () -> ui32
        %23 = "emitc.constant"() <{value = 32 : ui32}> : () -> ui32
        %24 = "emitc.constant"() <{value = 32 : ui32}> : () -> ui32
        %25 = "emitc.constant"() <{value = 1 : ui32}> : () -> ui32
        metal.store %3, %9[%arg0 : %23] [%arg1 : %24] [%22 : %25] : i32, index, index, index, ui32, ui32, ui32, ui32
      }
    }
    %10 = "emitc.constant"() <{value = false}> : () -> i1
    %11 = "emitc.constant"() <{value = 32 : i64}> : () -> i64
    %12 = "emitc.constant"() <{value = 32 : i64}> : () -> i64
    %13 = "emitc.constant"() <{value = 1 : i64}> : () -> i64
    %14 = metal.device_make_buffer %5, %10, %11, %12, %13, "int32_t" : (index, i1, i64, i64, i64) -> index
    emitc.for %arg0 = %2 to %1 step %0 {
      emitc.for %arg1 = %2 to %1 step %0 {
        %22 = "emitc.constant"() <{value = 0 : ui32}> : () -> ui32
        %23 = "emitc.constant"() <{value = 32 : ui32}> : () -> ui32
        %24 = "emitc.constant"() <{value = 32 : ui32}> : () -> ui32
        %25 = "emitc.constant"() <{value = 1 : ui32}> : () -> ui32
        metal.store %3, %14[%arg0 : %23] [%arg1 : %24] [%22 : %25] : i32, index, index, index, ui32, ui32, ui32, ui32
      }
    }
    %15 = "emitc.constant"() <{value = false}> : () -> i1
    %16 = "emitc.constant"() <{value = 32 : i64}> : () -> i64
    %17 = "emitc.constant"() <{value = 32 : i64}> : () -> i64
    %18 = "emitc.constant"() <{value = 1 : i64}> : () -> i64
    %19 = metal.device_make_buffer %5, %15, %16, %17, %18, "int32_t" : (index, i1, i64, i64, i64) -> index
    %20 = "emitc.constant"() <{value = 32 : ui32}> : () -> ui32
    %21 = metal.device_make_command_queue %5 : (index) -> index
    metal.matmul %21, %9<%6 : %7> %14<%11 : %12> %19, %20 : index, index, i64, i64, index, i64, i64, index, ui32
    return
  }
}


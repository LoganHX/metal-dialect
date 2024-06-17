module {
  func.func @main() {
    %0 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %1 = "emitc.constant"() <{value = 32 : index}> : () -> index
    %2 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %3 = "emitc.constant"() <{value = 0 : si32}> : () -> si32
    %4 = "emitc.constant"() <{value = false}> : () -> i1
    %5 = metal.device_make_default : index
    %6 = "emitc.constant"() <{value = 32 : i64}> : () -> i64
    %7 = "emitc.constant"() <{value = 32 : i64}> : () -> i64
    %8 = "emitc.constant"() <{value = 1 : i64}> : () -> i64
    %9 = metal.device_make_buffer %5, %4, %6, %7, %8, "int32_t" : (index, i1, i64, i64, i64) -> index
    emitc.for %arg0 = %2 to %1 step %0 {
      emitc.for %arg1 = %2 to %1 step %0 {
        %20 = "emitc.constant"() <{value = 0 : ui32}> : () -> ui32
        %21 = "emitc.constant"() <{value = 32 : ui32}> : () -> ui32
        %22 = "emitc.constant"() <{value = 32 : ui32}> : () -> ui32
        %23 = "emitc.constant"() <{value = 1 : ui32}> : () -> ui32
        metal.store %3, %9[%arg0 : %21] [%arg1 : %22] [%20 : %23] : si32, index, index, index, ui32, ui32, ui32, ui32
      }
    }
    %10 = "emitc.constant"() <{value = false}> : () -> i1
    %11 = "emitc.constant"() <{value = 32 : i64}> : () -> i64
    %12 = "emitc.constant"() <{value = 32 : i64}> : () -> i64
    %13 = "emitc.constant"() <{value = 1 : i64}> : () -> i64
    %14 = metal.device_make_buffer %5, %10, %11, %12, %13, "int32_t" : (index, i1, i64, i64, i64) -> index
    emitc.for %arg0 = %2 to %1 step %0 {
      emitc.for %arg1 = %2 to %1 step %0 {
        %20 = "emitc.constant"() <{value = 0 : ui32}> : () -> ui32
        %21 = "emitc.constant"() <{value = 32 : ui32}> : () -> ui32
        %22 = "emitc.constant"() <{value = 32 : ui32}> : () -> ui32
        %23 = "emitc.constant"() <{value = 1 : ui32}> : () -> ui32
        metal.store %3, %14[%arg0 : %21] [%arg1 : %22] [%20 : %23] : si32, index, index, index, ui32, ui32, ui32, ui32
      }
    }
    %15 = "emitc.constant"() <{value = false}> : () -> i1
    %16 = "emitc.constant"() <{value = 32 : i64}> : () -> i64
    %17 = "emitc.constant"() <{value = 32 : i64}> : () -> i64
    %18 = "emitc.constant"() <{value = 1 : i64}> : () -> i64
    %19 = metal.device_make_buffer %5, %15, %16, %17, %18, "int32_t" : (index, i1, i64, i64, i64) -> index
    emitc.for %arg0 = %2 to %1 step %0 {
      emitc.for %arg1 = %2 to %1 step %0 {
        emitc.for %arg2 = %2 to %1 step %0 {
          %20 = "emitc.constant"() <{value = 0 : ui32}> : () -> ui32
          %21 = "emitc.constant"() <{value = 32 : ui32}> : () -> ui32
          %22 = "emitc.constant"() <{value = 32 : ui32}> : () -> ui32
          %23 = "emitc.constant"() <{value = 1 : ui32}> : () -> ui32
          %24 = metal.get_element %9[%arg0 : %21] [%arg2 : %22] [%20 : %23] : (index, index, index, ui32, ui32, ui32, ui32) -> si32
          %25 = "emitc.constant"() <{value = 0 : ui32}> : () -> ui32
          %26 = "emitc.constant"() <{value = 32 : ui32}> : () -> ui32
          %27 = "emitc.constant"() <{value = 32 : ui32}> : () -> ui32
          %28 = "emitc.constant"() <{value = 1 : ui32}> : () -> ui32
          %29 = metal.get_element %14[%arg2 : %26] [%arg1 : %27] [%25 : %28] : (index, index, index, ui32, ui32, ui32, ui32) -> si32
          %30 = "emitc.constant"() <{value = 0 : ui32}> : () -> ui32
          %31 = "emitc.constant"() <{value = 32 : ui32}> : () -> ui32
          %32 = "emitc.constant"() <{value = 32 : ui32}> : () -> ui32
          %33 = "emitc.constant"() <{value = 1 : ui32}> : () -> ui32
          %34 = metal.get_element %19[%arg0 : %31] [%arg1 : %32] [%30 : %33] : (index, index, index, ui32, ui32, ui32, ui32) -> si32
          %35 = emitc.cast %24 : si32 to ui32
          %36 = emitc.cast %29 : si32 to ui32
          %37 = emitc.mul %35, %36 : (ui32, ui32) -> ui32
          %38 = emitc.cast %37 : ui32 to i32
          %39 = emitc.cast %34 : si32 to ui32
          %40 = emitc.cast %38 : i32 to ui32
          %41 = emitc.add %39, %40 : (ui32, ui32) -> ui32
          %42 = emitc.cast %41 : ui32 to si32
          %43 = "emitc.constant"() <{value = 0 : ui32}> : () -> ui32
          %44 = "emitc.constant"() <{value = 32 : ui32}> : () -> ui32
          %45 = "emitc.constant"() <{value = 32 : ui32}> : () -> ui32
          %46 = "emitc.constant"() <{value = 1 : ui32}> : () -> ui32
          metal.store %42, %19[%arg0 : %44] [%arg1 : %45] [%43 : %46] : si32, index, index, index, ui32, ui32, ui32, ui32
        }
      }
    }
    return
  }
}


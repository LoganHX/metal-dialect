module {
  func.func @main() {
    %0 = metal.device_make_default : index
    %1 = metal.device_make_command_queue %0 : (index) -> index
    %2 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %3 = "emitc.constant"() <{value = 1.021000e+01 : f32}> : () -> f32
    %4 = "emitc.constant"() <{value = false}> : () -> i1
    %5 = "emitc.constant"() <{value = 40 : i64}> : () -> i64
    %6 = "emitc.constant"() <{value = 41 : i64}> : () -> i64
    %7 = "emitc.constant"() <{value = 42 : i64}> : () -> i64
    %8 = metal.device_make_buffer %0, %4, %5, %6, %7 x f32 : (index, i1, i64, i64, i64) -> index
    metal.store %3, %8, %2, %2, %2[%5, %6, %7] : f32, index, index, index, index, i64, i64, i64
    %9 = metal.get_element %8, %2, %2, %2[%5, %6, %7] : (index, index, index, index, i64, i64, i64) -> f32
    metal.release %8 : index
    return
  }
}


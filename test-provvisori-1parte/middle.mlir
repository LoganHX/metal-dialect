module {
  func.func @main() {
    %0 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %1 = "emitc.constant"() <{value = 7 : index}> : () -> index
    %2 = "emitc.constant"() <{value = 10 : index}> : () -> index
    %3 = "emitc.constant"() <{value = 1.021000e+01 : f32}> : () -> f32
    %4 = "emitc.constant"() <{value = false}> : () -> i1
    %5 = "emitc.constant"() <{value = 42 : i64}> : () -> i64
    %6 = "emitc.constant"() <{value = 42 : i64}> : () -> i64
    %7 = metal.device_make_default : index
    %8 = metal.device_make_buffer %7, %4, %5, %6 x f32 : (index, i1, i64, i64) -> index
    metal.store %3, %8, %0, %1[%5, %6] : f32, index, index, index, i64, i64
    %9 = metal.get_element %8, %1, %0[%5, %6] : (index, index, index, i64, i64) -> f32
    metal.release %8 : index
    return
  }
}


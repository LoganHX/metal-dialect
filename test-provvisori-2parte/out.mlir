module {
  func.func private @expf(f32) -> f32 attributes {llvm.readnone}
  func.func private @roundevenf(f32) -> f32 attributes {llvm.readnone}
  func.func @main(%arg0: memref<1x28x28x1xf32, strided<[?, ?, ?, ?], offset: ?>> {ml_program.identifier = "serving_default_keras_tensor"}) -> (index {ml_program.identifier = "StatefulPartitionedCall_1"}) {
    %0 = "emitc.constant"() <{value = 10 : index}> : () -> index
    %1 = "emitc.constant"() <{value = 64 : index}> : () -> index
    %2 = "emitc.constant"() <{value = 784 : index}> : () -> index
    %3 = "emitc.constant"() <{value = 28 : index}> : () -> index
    %4 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %5 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %6 = "emitc.constant"() <{value = 274877906944 : i64}> : () -> i64
    %7 = "emitc.constant"() <{value = 39 : i64}> : () -> i64
    %8 = "emitc.constant"() <{value = 1089145654 : i64}> : () -> i64
    %9 = "emitc.constant"() <{value = 4398046511104 : i64}> : () -> i64
    %10 = "emitc.constant"() <{value = 43 : i64}> : () -> i64
    %11 = "emitc.constant"() <{value = 1852112935 : i64}> : () -> i64
    %12 = "emitc.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
    %13 = "emitc.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %14 = "emitc.constant"() <{value = -3.40282347E+38 : f32}> : () -> f32
    %15 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %16 = "emitc.constant"() <{value = 127 : i32}> : () -> i32
    %17 = "emitc.constant"() <{value = -1073741824 : i64}> : () -> i64
    %18 = "emitc.constant"() <{value = 1073741824 : i64}> : () -> i64
    %19 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %20 = "emitc.constant"() <{value = -128 : i32}> : () -> i32
    %21 = "emitc.constant"() <{value = 2147483647 : i32}> : () -> i32
    %22 = "emitc.constant"() <{value = 2.14748365E+9 : f32}> : () -> f32
    %23 = "emitc.constant"() <{value = -2.14748365E+9 : f32}> : () -> f32
    %24 = "emitc.constant"() <{value = false}> : () -> i1
    %25 = "emitc.constant"() <{value = 1 : i64}> : () -> i64
    %26 = "emitc.constant"() <{value = 28 : i64}> : () -> i64
    %27 = "emitc.constant"() <{value = 28 : i64}> : () -> i64
    %28 = "emitc.constant"() <{value = 1 : i64}> : () -> i64
    %29 = metal.device_make_default : index
    %30 = metal.device_make_buffer %29, %24, %25, %26, %27, %28 x f32 : (index, i1, i64, i64, i64, i64) -> index
    %31 = "emitc.constant"() <{value = false}> : () -> i1
    %32 = "emitc.constant"() <{value = 1 : i64}> : () -> i64
    %33 = "emitc.constant"() <{value = 10 : i64}> : () -> i64
    %34 = metal.device_make_buffer %29, %31, %32, %33 x f32 : (index, i1, i64, i64) -> index
    return %34 : index
  }
}


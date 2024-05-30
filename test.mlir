func.func @emitc_variable() {
  %c0 = "emitc.variable"(){value = #emitc.opaque<"">} : () -> i32
  %c1 = "emitc.variable"(){value = 42 : i32} : () -> i32
  %c2 = "emitc.variable"(){value = -1 : i32} : () -> i32
  %c3 = "emitc.variable"(){value = -1 : si8} : () -> si8
  %c4 = "emitc.variable"(){value = 255 : ui8} : () -> ui8
  %c5 = "emitc.variable"(){value = #emitc.opaque<"">} : () -> !emitc.ptr<i32>
  %c6 = "emitc.variable"(){value = #emitc.opaque<"NULL">} : () -> !emitc.ptr<i32>
  %c7 = "emitc.variable"(){value = #emitc.opaque<"">} : () -> !emitc.array<3x7xi32>
  %c8 = "emitc.variable"(){value = #emitc.opaque<"">} : () -> !emitc.array<5x!emitc.ptr<i8>>
  return
}
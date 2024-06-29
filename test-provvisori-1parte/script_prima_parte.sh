#!/bin/bash

#Scriptino
set -x

start="test-provvisori-1parte/starting_point.mlir"
input="test-provvisori-1parte/after_official.mlir"
middle="test-provvisori-1parte/middle.mlir"
output="test-provvisori-1parte/out.mlir"

mlir_opt=llvm-project/build/release/bin/mlir-opt
metal_opt=/build/debug/bin/metal-opt
pop_translate=build/debug/tools/mlir-translate/pop-translate

./$mlir_opt $start --gpu-map-parallel-loops                     1> $output 
./$mlir_opt $output --convert-parallel-loops-to-gpu             1> $input 
./$mlir_opt $input --gpu-launch-sink-index-computations        1> $output 
./$mlir_opt $output --gpu-kernel-outlining                      1> $input 
./$mlir_opt $input --lower-affine                              1> $output 
./$mlir_opt $output --convert-arith-to-emitc                    1> $input
./$metal_opt $input --convert-gpu-launch-func-to-metal -allow-unregistered-dialect 1> $middle
./$pop_translate $middle --mlir-to-metal 1> $output 

# Remove tmp files
# rm $assembly_file
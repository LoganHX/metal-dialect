#!/bin/bash

#Scriptino
set -x

start="test-provvisori-2parte/starting_point.mlir"

input="test-provvisori-2parte/after_official.mlir"
middle="test-provvisori-2parte/middle.mlir"
output="test-provvisori-2parte/out.mlir"

mlir_opt=llvm-project/build/release/bin/mlir-opt
metal_opt=build/debug/bin/metal-opt
pop_translate=build/debug/tools/mlir-translate/pop-translate

#  ./$mlir_opt $start -convert-linalg-to-loops                     1> $input 
# # ./$mlir_opt $middle --convert-parallel-loops-to-gpu             1> $output 
# # ./$mlir_opt $input --gpu-launch-sink-index-computations        1> $output 
# # ./$mlir_opt $output --gpu-kernel-outlining                      1> $input 
#  ./$mlir_opt $input --convert-scf-to-emitc                               1> $middle 
#  ./$mlir_opt $middle --convert-arith-to-emitc                    1> $input
 ./$metal_opt $input --convert-gpu-launch-func-to-metal -allow-unregistered-dialect 1> $middle
#./$pop_translate $middle --mlir-to-metal 1> $output 

# Remove tmp files
# rm $assembly_file
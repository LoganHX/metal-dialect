#!/bin/bash

#Scriptino
set -x

start="test-provvisori/starting_point.mlir"
input="test-provvisori/parallel.mlir"
output="test-provvisori/out.mlir"

mlir_opt=llvm-project/build/release/bin/mlir-opt
pop_translate=build/debug/tools/mlir-translate/pop-translate

./$mlir_opt $start --gpu-map-parallel-loops                     1> $output 
./$mlir_opt $output --convert-parallel-loops-to-gpu             1> $input 
./$mlir_opt $input --gpu-launch-sink-index-computations        1> $output 
./$mlir_opt $output --gpu-kernel-outlining                      1> $input 
./$mlir_opt $input --lower-affine                              1> $output 
./$mlir_opt $output --convert-arith-to-emitc                    1> $input
./$pop_translate $input --mlir-to-metal 1> $output 

# Remove tmp files
# rm $assembly_file
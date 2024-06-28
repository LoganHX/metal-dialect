#!/bin/bash

#Scriptino
set -x

start="/Users/c.stabile12/Downloads/anodo.mlir"

input="test-provvisori-2parte/after_official.mlir"
middle="test-provvisori-2parte/middle.mlir"
output="test-provvisori-2parte/out.mlir"

mlir_opt=llvm-project/build/release/bin/mlir-opt
metal_opt=build/debug/bin/metal-opt
pop_translate=build/debug/tools/mlir-translate/pop-translate


# ./$mlir_opt $start --canonicalize --cse 1> $output 
# ./$mlir_opt $output --pass-pipeline="builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-arith{include-apply-rescale=1}, tosa-to-tensor, tosa-to-scf))" 1> $middle 

#  ./$mlir_opt $middle --convert-tensor-to-linalg --empty-tensor-to-alloc-tensor \
#                     --eliminate-empty-tensors -one-shot-bufferize="bufferize-function-boundaries"  --finalizing-bufferize 1> $output 


# ./$mlir_opt $output --convert-linalg-to-loops     --convert-scf-to-emitc                 1> $input 
# ./$mlir_opt $input      --arith-expand    -arith-unsigned-when-equivalent            1> $middle 
./$metal_opt $middle     --fold-memref-alias-ops   --convert-arith-to-emitc            1> $input
./$metal_opt $input --convert-gpu-launch-func-to-metal -allow-unregistered-dialect 1> $output


# # ./$mlir_opt $middle --convert-parallel-loops-to-gpu             1> $output 
# # ./$mlir_opt $input --gpu-launch-sink-index-computations        1> $output 
# # ./$mlir_opt $output --gpu-kernel-outlining                      1> $input 
 #./$mlir_opt $input --convert-scf-to-emitc                               1> $middle 
 #./$mlir_opt $middle --convert-arith-to-emitc                    1> $input
 
#./$pop_translate $middle --mlir-to-metal 1> $output 

# Remove tmp files
# rm $assembly_file
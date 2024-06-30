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

# ./$mlir_opt $start --pass-pipeline="builtin.module(func.func(tosa-to-linalg-named, \
#                     tosa-to-linalg, tosa-to-arith{include-apply-rescale=1}, \
#                     tosa-to-tensor, tosa-to-scf))" 1> $output

./$mlir_opt $output \
                    --canonicalize \
                    --cse \
                    --convert-tensor-to-linalg \
                    --empty-tensor-to-alloc-tensor \
                    --eliminate-empty-tensors \
                    --one-shot-bufferize="bufferize-function-boundaries" \
                    --finalizing-bufferize \
                    --convert-linalg-to-loops \
                    --arith-expand  \
                    --arith-unsigned-when-equivalent \
                    --convert-scf-to-emitc \
                    --fold-memref-alias-ops \
                    --memref-expand \
                    --convert-math-to-libm \
                    --arith-expand \
                    --arith-unsigned-when-equivalent \
                    --convert-scf-to-emitc \
                    --gpu-launch-sink-index-computations \
                    --gpu-kernel-outlining  \
                    --convert-scf-to-emitc \
                    1> $input 


./$metal_opt $input   --arith-expand --convert-arith-to-emitc \
                      --convert-gpu-launch-func-to-metal --allow-unregistered-dialect  1> $middle

# ./$metal_opt $middle  --lower-affine \
#                       --memref-expand \
#                       --convert-arith-to-emitc \
#                       --convert-memref-to-emitc 1> $output

# ./$pop_translate $output --mlir-to-metal 1> $input 

# Remove tmp files
# rm $assembly_file
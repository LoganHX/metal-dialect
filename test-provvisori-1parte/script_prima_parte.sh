#!/bin/bash

#Scriptino
set -x

start="test-provvisori-1parte/starting_point.mlir"
input="test-provvisori-1parte/after_official.mlir"
middle="test-provvisori-1parte/middle.mlir"
output="test-provvisori-1parte/out.mlir"

mlir_opt=llvm-project/build/release/bin/mlir-opt
metal_opt=build/debug/bin/metal-opt
pop_translate=build/debug/tools/mlir-translate/pop-translate

./$mlir_opt $start  --gpu-map-parallel-loops \
                    --convert-parallel-loops-to-gpu \
                    --gpu-launch-sink-index-computations \
                    --gpu-kernel-outlining \
                    --lower-affine \
                    --convert-arith-to-emitc \
                    --convert-scf-to-emitc 1> $input 


./$metal_opt $input --convert-func-to-func  --convert-gpu-launch-func-to-metal 1> $middle
./$pop_translate $middle --mlir-to-metal 1> $output 
# Remove tmp files
# rm $assembly_file
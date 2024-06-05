func.func @parallel_loop() {
  %step = arith.constant 1 : index
  %start = arith.constant 0 : index
  %end = arith.constant 21 : index

  scf.parallel (%i1) = (%start) to (%end) step (%step)  {
   
  }
  return
}

module {
  func.func @parallel_loop() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c21 = arith.constant 21 : index
    scf.parallel (%arg0) = (%c0) to (%c21) step (%c1) {
      scf.reduce 
    } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    return
  }
}

#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
module {
  func.func @parallel_loop() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c21 = arith.constant 21 : index
    %c1_0 = arith.constant 1 : index
    %0 = affine.apply #map(%c21)[%c0, %c1]
    gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %0, %arg7 = %c1_0, %arg8 = %c1_0) threads(%arg3, %arg4, %arg5) in (%arg9 = %c1_0, %arg10 = %c1_0, %arg11 = %c1_0) {
      %1 = affine.apply #map1(%arg0)[%c1, %c0]
      gpu.terminator
    } {SCFToGPU_visited}
    return
  }
}


module {
  func.func @parallel_loop() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c21 = arith.constant 21 : index
    %c1_0 = arith.constant 1 : index
    %c21_1 = arith.constant 21 : index
    gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %c21_1, %arg7 = %c1_0, %arg8 = %c1_0) threads(%arg3, %arg4, %arg5) in (%arg9 = %c1_0, %arg10 = %c1_0, %arg11 = %c1_0) {
      %0 = arith.muli %arg0, %c1 : index
      %1 = arith.addi %0, %c0 : index
      gpu.terminator
    } {SCFToGPU_visited}
    return
  }
}

// Lower a `scf.parallel` operation into a corresponding `gpu.launch`
//  operation.

//  This essentially transforms a loop nest into a corresponding SIMT function.
//  The conversion is driven by mapping annotations on the `scf.parallel`
//  operations. The mapping is provided via a `DictionaryAttribute` named
//  `mapping`, which has three entries:
//   - processor: the hardware id to map to. 0-2 are block dimensions, 3-5 are
//                thread dimensions and 6 is sequential.
//   - map : An affine map that is used to pre-process hardware ids before
//           substitution.
//   - bound : An affine map that is used to compute the bound of the hardware
//             id based on an upper bound of the number of iterations.
//  If the `scf.parallel` contains nested `scf.parallel` operations, those
//  need to be annotated, as well. Structurally, the transformation works by
//  splicing all operations from nested `scf.parallel` operations into a single
//  sequence. Indices mapped to hardware ids are substituted with those ids,
//  wheras sequential mappings result in a sequential for-loop. To have more
//  flexibility when mapping code to hardware ids, the transform supports two
//  affine maps. The first `map` is used to compute the actual index for
//  substitution from the hardware id. The second `bound` is used to compute the
//  launch dimension for the hardware id from the number of iterations the
//  mapped loop is performing. Note that the number of iterations might be
//  imprecise if the corresponding loop-bounds are loop-dependent. In such case,
//  the hardware id might iterate over additional indices. The transformation
//  caters for this by predicating the created sequence of instructions on
//  the actual loop bound. This only works if an static upper bound for the
//  dynamic loop bound can be derived, currently via analyzing `affine.min`
//  operations.


// func.func @parallel_loop(%arg0 : index, %arg1 : index, %arg2 : index,
//                     %arg3 : index) {
//   %zero = arith.constant 0 : index
//   %one = arith.constant 1 : index
//   %four = arith.constant 4 : index
//   scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
//                                           step (%four, %four)  {
//     scf.parallel (%si0, %si1) = (%zero, %zero) to (%four, %four)
//                                             step (%one, %one)  {
//     }
//   }
//   return
// }


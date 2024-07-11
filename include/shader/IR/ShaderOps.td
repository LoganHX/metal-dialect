//===--- ShaderOps.td - Shader dialect ops -------------------*- tablegen -*-===//
//
// This source file is part of the shader-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef SHADER_OPS
#define SHADER_OPS

include "shader/IR/ShaderDialect.td"

include "shader/IR/ShaderTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
def Int_Type : AnyTypeOf<[I1, I8, I32, I64, UI8, UI16, UI32, UI64, SI8, SI16, SI32, SI64, Index]>; 

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

def MatmulOp : Shader_Op<"matmul", [AttrSizedOperandSegments]> {
  let summary = "Matrix multiplication operation";

  let arguments = (ins Optional<Index>:$queue, 
                      AnyType:$bufferA, Optional<Int_Type>:$rowsA, Optional<Int_Type>:$columnsA,
                      AnyType:$bufferB, Optional<Int_Type>:$rowsB, Optional<Int_Type>:$columnsB,
                      AnyType:$bufferC,
                      TypeAttr:$elementType);
  
  let results = (outs Index:$result);

  let hasVerifier = 0; //TODO scrivere i verifiers

}

#endif // SHADER_OPS

//===--- ShaderDialect.td - Shader dialect -------------------*- tablegen -*-===//
//
// This source file is part of the shader-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef SHADER_DIALECT
#define SHADER_DIALECT

include "mlir/IR/OpBase.td"


//===----------------------------------------------------------------------===//
// Shader dialect definition.
//===----------------------------------------------------------------------===//

def Shader_Dialect : Dialect {
    let name = "shader";
    let summary = "Shader MLIR dialect.";
    let cppNamespace = "::mlir::shader";
    
}

//===----------------------------------------------------------------------===//
// Base shader operation definition.
//===----------------------------------------------------------------------===//

class Shader_Op<string mnemonic, list<Trait> traits = []> :
        Op<Shader_Dialect, mnemonic, traits>;

#endif // SHADER_DIALECT

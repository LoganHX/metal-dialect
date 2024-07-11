//===--- MetalPasses.td - Metal passes ---------------------*- tablegen -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_PASSES_TD
#define METAL_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertFuncToFunc : Pass<"convert-func-to-func"> {
  let summary = "Converts Func to Func";
  let dependentDialects = ["mlir::metal::MetalDialect"];
}

def ConvertGpuLaunchToMetal : Pass<"convert-gpu-launch-func-to-metal"> {
  let summary = "Converts GPU Launch operations in operations of the metal dialect";
  let dependentDialects = ["mlir::metal::MetalDialect", "mlir::emitc::EmitCDialect", "mlir::shader::ShaderDialect"];
}

def ConvertLinalgToMetal : Pass<"convert-linalg-to-metal"> {
  let summary = "Converts GPU Launch operations in operations of the metal dialect";
  let dependentDialects = ["mlir::metal::MetalDialect", "mlir::emitc::EmitCDialect", "mlir::shader::ShaderDialect"];
}



#endif // METAL_PASSES_TD

//===--- metal-opt.cpp ------------------------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal/Conversion/MetalPasses.h"
#include "metal/IR/MetalDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"



int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::metal::registerMetalConversionPasses();

  mlir::DialectRegistry registry;
  registry.insert<
      mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect, mlir::func::FuncDialect, 
      mlir::LLVM::LLVMDialect, mlir::metal::MetalDialect, mlir::gpu::GPUDialect, mlir::memref::MemRefDialect, 
      mlir::emitc::EmitCDialect, mlir::linalg::LinalgDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Metal optimizer driver\n", registry));
}

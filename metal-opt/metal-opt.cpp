//===--- metal-opt.cpp ------------------------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal/Conversion/MetalPasses.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "metal/IR/MetalDialect.h"
#include "shader/IR/ShaderDialect.h"
#include "shader/IR/ShaderOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"


int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::metal::registerMetalConversionPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                  mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                  mlir::metal::MetalDialect, mlir::gpu::GPUDialect,
                  mlir::memref::MemRefDialect, mlir::emitc::EmitCDialect,
                  mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                  mlir::shader::ShaderDialect,
                  mlir::bufferization::BufferizationDialect, mlir::tosa::TosaDialect,
                  mlir::scf::SCFDialect, mlir::affine::AffineDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Metal optimizer driver\n", registry));
}

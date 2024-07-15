//===---
// ConvertMemrefToIndex.cpp--------------------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal/Conversion/MemrefToIndex.h"
#include "metal/Conversion/MetalPasses.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "shader/IR/ShaderDialect.h"
#include "shader/IR/ShaderOps.h"

#include "mlir/Transforms/DialectConversion.h"
#include <iostream>

namespace mlir::metal {

#define GEN_PASS_DEF_CONVERTMEMREFTOINDEX
#include "metal/Conversion/MetalPasses.h.inc"

bool isMemRefArgumentUsedInGPULaunchFunc(gpu::LaunchFuncOp launchOp) {
  // auto parentFunc = launchOp->getParentOfType<func::FuncOp>();
  // if (!parentFunc)
  //   return false;

  // for (auto arg : parentFunc.getArguments()) {
  //   if (arg.getType().isa<MemRefType>()) {
  //     for (auto operand : launchOp.getOperands()) {
  //       if (operand== arg) {
  //         return true;
  //       }
  //     }
  //   }
  // }
  launchOp.dump();
  for (auto operand : launchOp.getKernelOperands()) {

    if (isa<MemRefType>(operand.getType())) {
      operand.dump();
      return true;
    }
  }
  return false;
}

bool isOneArgumentBufferUsedByGPUFunc(Operation *op) {
  if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
    for (auto arg : funcOp.getArguments()) {
      if (arg.getType().isa<MemRefType>()) {
        for (auto &block : funcOp.getBody().getBlocks()) {
          for (auto &innerOp : block) {
            if (auto launchOp = dyn_cast<gpu::LaunchFuncOp>(&innerOp)) {
              for (auto operand : launchOp.getOperands()) {
                if (operand == arg) {
                  return true;
                }
              }
            }
          }
        }
      }
    }
  }
  return false;
}

func::FuncOp findFunctionByName(Operation *moduleOp, StringRef funcName) {
  for (auto &op : moduleOp->getRegion(0).getOps()) {
    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {

      if (funcOp.getName() == funcName) {
        return funcOp;
      }
    }
  }
  return nullptr;
}

mlir::ModuleOp getModuleOp(Operation *op) {
  while (!isa<mlir::ModuleOp>(op)) {
    op = op->getParentOp();
    if (isa<mlir::ModuleOp>(op))
      break;
  }
  return dyn_cast_or_null<mlir::ModuleOp>(op);
}

bool doesItCallAFaultyFunc(Operation *op) {
  if (auto callOp = dyn_cast<func::CallOp>(op)) {
    auto calledFunctionName = callOp.getCallee();
    if (auto moduleOp = getModuleOp(op)) {
      if (auto calledFunction =
              findFunctionByName(moduleOp, calledFunctionName)) {
        if (isOneArgumentBufferUsedByGPUFunc(calledFunction)) {
          return true;
        }
      }
    }
  }
  return false;
}

namespace mlir {
struct ConvertMemrefToIndex
    : public impl::ConvertMemrefToIndexBase<ConvertMemrefToIndex> {

  using impl::ConvertMemrefToIndexBase<
      ConvertMemrefToIndex>::ConvertMemrefToIndexBase;

  void runOnOperation() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<metal::MetalDialect>();
    target.addLegalDialect<shader::ShaderDialect>();
    target.addLegalDialect<gpu::GPUDialect>();
    target.addLegalDialect<tosa::TosaDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();

    target.addDynamicallyLegalOp<gpu::LaunchFuncOp>(
      [](gpu::LaunchFuncOp op) {
          return !isMemRefArgumentUsedInGPULaunchFunc(op);
       });
    target.addDynamicallyLegalOp<func::FuncOp>(
        [](func::FuncOp op) { return !isOneArgumentBufferUsedByGPUFunc(op); });
    target.addDynamicallyLegalOp<func::CallOp>(
        [](func::CallOp op) { return !doesItCallAFaultyFunc(op); });

    RewritePatternSet patterns(&getContext());
    metal::populateMemrefToIndexConversionPatterns(patterns, &getContext());

    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPartialConversion(getOperation(), target, patternSet)))
      signalPassFailure();
  }
};

} // namespace mlir
} // namespace mlir::metal
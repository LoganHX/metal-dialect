//===---
//ConvertGpuLaunchToMetal.cpp--------------------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal/Conversion/GpuLaunchToMetal.h"
#include "metal/Conversion/MetalPasses.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include "mlir/Transforms/DialectConversion.h"
#include <iostream>

namespace mlir::metal {

#define GEN_PASS_DEF_CONVERTGPULAUNCHTOMETAL
#include "metal/Conversion/MetalPasses.h.inc"

bool isAllocatedBufferUsedByGPU(memref::AllocOp allocOp) {
  for (auto *user : allocOp.getResult().getUsers()) {
    if (auto launchOp = dyn_cast<gpu::LaunchFuncOp>(user)) {
      for (Value arg : launchOp.getKernelOperands()) {
        if (arg == allocOp.getResult()) {
          return false;
        }
      }
    }
  }
  return true;
}

bool isDeallocatedBufferUsedByGPU(memref::DeallocOp deallocOp) {
  Value buffer = deallocOp.getMemref();

  for (auto *user : buffer.getUsers()) {
    if (auto launchOp = dyn_cast<gpu::LaunchFuncOp>(user)) {
      for (Value arg : launchOp.getKernelOperands()) {
        if (arg == buffer) {
          return false;
        }
      }
    }
  }
  return true;
}


bool doesReturnMemrefFunc(Operation *op) {
  if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
    if (funcOp.getResultTypes().size() == 0)
      return true;
    for (int i = 0; i < (int)funcOp.getResultTypes().size(); i++) {
      if (isa<MemRefType>(funcOp.getResultTypes()[0])) {
        return false;
      }
    }
    return true;
  }
  return true;
}

bool doesReturnMemrefReturn(Operation *op) {
  if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
    if (returnOp.getOperands().size() == 0)
      return true;
    for (int i = 0; i < (int)returnOp.getOperands().size(); i++) {
      if (isa<MemRefType>(returnOp.getOperand(i).getType())) {
        return false;
      }
    }
    return true;
  }
  return true;
}

// Check se il memref è definito tramite memref::AllocOp
bool isAllocatedByAllocOp(Value value) {
  if (auto definingOp = value.getDefiningOp()) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(definingOp)) {
      return isAllocatedBufferUsedByGPU(allocOp);
    }
  }
  return true;
};

// Funzione di utilità per controllare se l'operazione è valida
bool isLoadOrStoreOpValid(Operation *op) {
  if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    return isAllocatedByAllocOp(loadOp.getMemRef());
  }
  if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
    return isAllocatedByAllocOp(storeOp.getMemRef());
  }
  return false;
};

namespace {
struct ConvertGpuLaunchToMetal
    : public impl::ConvertGpuLaunchToMetalBase<ConvertGpuLaunchToMetal> {

  using impl::ConvertGpuLaunchToMetalBase<
      ConvertGpuLaunchToMetal>::ConvertGpuLaunchToMetalBase;

  void runOnOperation() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<MetalDialect>();
    target.addLegalDialect<gpu::GPUDialect>();
    target.addLegalDialect<tosa::TosaDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();

    target.addIllegalOp<gpu::LaunchFuncOp>();

    target.addIllegalOp<memref::StoreOp>();
    target.addIllegalOp<memref::LoadOp>();
    target.addIllegalOp<memref::AllocOp>();
    target.addIllegalOp<memref::DeallocOp>();

    target.addLegalDialect<bufferization::BufferizationDialect>();

    target.addDynamicallyLegalOp<memref::AllocOp>(
        [](memref::AllocOp op) { return isAllocatedBufferUsedByGPU(op); });
    target.addDynamicallyLegalOp<memref::DeallocOp>(
        [](memref::DeallocOp op) { return isDeallocatedBufferUsedByGPU(op); });

    target.addDynamicallyLegalOp<memref::LoadOp>(
        [](memref::LoadOp op) { return isLoadOrStoreOpValid(op); });
    target.addDynamicallyLegalOp<memref::StoreOp>(
        [](memref::StoreOp op) { return isLoadOrStoreOpValid(op); });

    target.addDynamicallyLegalOp<func::FuncOp>(
        [](func::FuncOp op) { return doesReturnMemrefFunc(op); });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [](func::ReturnOp op) { return doesReturnMemrefReturn(op); });
    target.addDynamicallyLegalOp<metal::MatmulOp>(
        [](metal::MatmulOp op) { return !(op.getQueue() == nullptr); });

    RewritePatternSet patterns(&getContext());
    mlir::metal::populateGpuLaunchToMetalConversionPatterns(patterns,
                                                            &getContext());

    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPartialConversion(getOperation(), target, patternSet)))
      signalPassFailure();
  }
};

} // end namespace
} // namespace mlir::metal
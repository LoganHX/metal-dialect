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
#include "shader/IR/ShaderOps.h"
#include "shader/IR/ShaderDialect.h"


#include "mlir/Transforms/DialectConversion.h"
#include <iostream>

namespace mlir::metal {

#define GEN_PASS_DEF_CONVERTGPULAUNCHTOMETAL
#include "metal/Conversion/MetalPasses.h.inc"

bool isAllocatedByGPU(Value value) {
  if (auto definingOp = value.getDefiningOp()) {
    if (auto allocOp = dyn_cast<gpu::AllocOp>(definingOp)) {
      return true;
    }
  }
  return false;
};


bool isLoadOrStoreOpValid(Operation *op) {
  if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    return isAllocatedByGPU(loadOp.getMemRef());
  }
  if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
    return isAllocatedByGPU(storeOp.getMemRef());
  }
  return false;
};

namespace mlir{
struct ConvertGpuLaunchToMetal
    : public impl::ConvertGpuLaunchToMetalBase<ConvertGpuLaunchToMetal> {

  using impl::ConvertGpuLaunchToMetalBase<
      ConvertGpuLaunchToMetal>::ConvertGpuLaunchToMetalBase;

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

    target.addIllegalOp<gpu::LaunchFuncOp>();

    target.addIllegalOp<memref::StoreOp>();
    target.addIllegalOp<memref::LoadOp>();
    target.addIllegalOp<gpu::AllocOp>();
    target.addIllegalOp<gpu::DeallocOp>();


    target.addLegalDialect<bufferization::BufferizationDialect>();



    target.addDynamicallyLegalOp<memref::LoadOp>(
        [](memref::LoadOp op) { return !isLoadOrStoreOpValid(op); });
    target.addDynamicallyLegalOp<memref::StoreOp>(
        [](memref::StoreOp op) { return !isLoadOrStoreOpValid(op); });
    
    target.addDynamicallyLegalOp<shader::MatmulOp>(
        [](shader::MatmulOp op) { return !(op.getQueue() == nullptr); });
    target.addDynamicallyLegalOp<shader::MatmulTransposeLeftOp>(
        [](shader::MatmulTransposeLeftOp op) { return !(op.getQueue() == nullptr); });
    target.addDynamicallyLegalOp<shader::MatmulTransposeRightOp>(
        [](shader::MatmulTransposeRightOp op) { return !(op.getQueue() == nullptr); });
    target.addDynamicallyLegalOp<shader::MatsumOp>(
        [](shader::MatsumOp op) { return !(op.getQueue() == nullptr); });

    RewritePatternSet patterns(&getContext());
    metal::populateGpuLaunchToMetalConversionPatterns(patterns,
                                                            &getContext());

    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPartialConversion(getOperation(), target, patternSet)))
      signalPassFailure();
  }
};

} // end namespace
} // namespace mlir::metal
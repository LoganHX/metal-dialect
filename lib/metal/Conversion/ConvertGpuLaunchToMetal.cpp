#include "metal/Conversion/GpuLaunchToMetal.h"
#include "metal/Conversion/MetalPasses.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"



#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Transforms/DialectConversion.h"
#include <iostream>

namespace mlir::metal {

#define GEN_PASS_DEF_CONVERTGPULAUNCHTOMETAL
#include "metal/Conversion/MetalPasses.h.inc"

bool isInsideGpuSpace(Operation *op) {
  auto parent = op->getParentOp();
  while (parent) {
    if (isa<gpu::GPUFuncOp>(parent) || isa<gpu::GPUModuleOp>(parent)) {
      return true;
    }
    parent = parent->getParentOp();
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

    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();

    target.addIllegalOp<gpu::LaunchFuncOp>();

    target.addIllegalOp<memref::StoreOp>();
    target.addIllegalOp<memref::LoadOp>();
    target.addIllegalOp<memref::AllocOp>();
    target.addIllegalOp<memref::DeallocOp>();

    target.addIllegalOp<linalg::MatmulOp>();

    target.addLegalDialect<bufferization::BufferizationDialect>();

    target.addDynamicallyLegalOp<memref::LoadOp>(
        [](memref::LoadOp op) { return isInsideGpuSpace(op); });
    target.addDynamicallyLegalOp<memref::StoreOp>(
        [](memref::StoreOp op) { return isInsideGpuSpace(op); });

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
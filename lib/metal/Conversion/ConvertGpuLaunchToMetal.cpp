#include "metal/Conversion/MetalPasses.h"
#include "metal/Conversion/GpuLaunchToMetal.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"


#include "mlir/Transforms/DialectConversion.h"
#include <iostream>

namespace mlir::metal {

#define GEN_PASS_DEF_CONVERTGPULAUNCHTOMETAL
#include "metal/Conversion/MetalPasses.h.inc"

bool isInsideGpuFunc(Operation *op) {
  // Verifica se l'operazione è all'interno di un gpu.funcOp
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
    
    target.addIllegalDialect<memref::MemRefDialect>();
    target.addIllegalOp<gpu::LaunchFuncOp>();

    target.addDynamicallyLegalOp<memref::LoadOp>([](memref::LoadOp op) {
        return isInsideGpuFunc(op);
    });
    target.addDynamicallyLegalOp<memref::StoreOp>([](memref::StoreOp op) {
        return isInsideGpuFunc(op);
    });

    RewritePatternSet patterns(&getContext());
    mlir::metal::populateGpuLaunchToMetalConversionPatterns(patterns, &getContext());

    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPartialConversion(getOperation(), target, patternSet)))
      signalPassFailure(); 
      
  }
  
};

} // end namespace
} // nam
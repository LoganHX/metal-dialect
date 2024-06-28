#include "metal/Conversion/GpuLaunchToMetal.h"
#include "metal/Conversion/MetalPasses.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"




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

// Funzione per controllare se il memref è definito tramite memref::AllocOp
bool isAllocatedByAllocOp(Value value) {
  if (auto definingOp = value.getDefiningOp()) {
   
    if(isa<memref::AllocOp>(definingOp)) {
      definingOp->dump();
      return false;
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
        [](memref::LoadOp op) { return isLoadOrStoreOpValid(op); });
    target.addDynamicallyLegalOp<memref::StoreOp>(
        [](memref::StoreOp op) { return isLoadOrStoreOpValid(op); });

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
#include "metal/Conversion/GpuLaunchToMetal.h"
#include "metal/IR/MetalOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"


#include <iostream>

#include "mlir/Transforms/DialectConversion.h"

namespace {

using namespace mlir;
using namespace LLVM;

class LaunchFuncOpLowering : public ConversionPattern {
public:
  explicit LaunchFuncOpLowering(MLIRContext *context)
      : ConversionPattern(gpu::LaunchFuncOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    // Posiziona il rewriter all'inizio dell'operazione
    rewriter.setInsertionPoint(op);

    // Crea due operazioni arith.constant
    rewriter.create<emitc::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getIntegerAttr(rewriter.getI64Type(), 21));
    rewriter.create<emitc::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getIntegerAttr(rewriter.getI64Type(), 42));
    rewriter.create<emitc::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getIntegerAttr(rewriter.getI64Type(), 21));
    rewriter.create<emitc::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getIntegerAttr(rewriter.getI64Type(), 42));


    // Elimina l'operazione originale
    rewriter.eraseOp(op);

    return success();
  }
};

} // end namespace

void mlir::metal::populateGpuLaunchToMetalConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<LaunchFuncOpLowering>(ctx);
}
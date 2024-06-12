#include "metal/Conversion/GpuLaunchToMetal.h"
#include "metal/IR/MetalOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"

#include "mlir/Transforms/DialectConversion.h"

namespace {

using namespace mlir;
using namespace LLVM;

class LaunchFuncOpLowering : public ConversionPattern {
public:
  explicit LaunchFuncOpLowering(MLIRContext *context)
      : ConversionPattern(gpu::LaunchFuncOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    
    rewriter.eraseOp(op);
    return success();
  }
};


} // end namespace

void mlir::metal::populateGpuLaunchToMetalConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<LaunchFuncOpLowering>(ctx);
}
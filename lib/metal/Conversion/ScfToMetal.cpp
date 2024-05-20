#include "metal/Conversion/ScfToMetal.h"
#include "metal/IR/MetalOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace {

using namespace mlir;
using namespace LLVM;



class ModuleOpLowering : public ConversionPattern {
public:
  explicit ModuleOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::metal::ModuleOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};


} // end namespace

void mlir::metal::populateScfToMetalConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ModuleOpLowering>(ctx);
}
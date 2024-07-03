//===--- ConvertFuncToFunc.cpp--------------------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal/Conversion/FuncToFunc.h"
#include "metal/Conversion/MetalPasses.h"
#include "metal/IR/MetalOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::metal {

#define GEN_PASS_DEF_CONVERTFUNCTOFUNC
#include "metal/Conversion/MetalPasses.h.inc"

bool isFirstOpDeviceOp(Operation *op, StringRef name) {
  auto funcOp = dyn_cast<func::FuncOp>(op);
  if (!funcOp || funcOp.getName() != name)
    return true;

  auto &body = funcOp.getBody();
  if (body.empty())
    return true;

  auto &firstBlock = body.front();
  if (firstBlock.empty())
    return true;

  auto *firstOp = &firstBlock.front();
  if (!isa<metal::DeviceMakeDefaultOp>(firstOp))
    return false;

  if (firstOp->getNextNode()) {
    auto *secondOp = firstOp->getNextNode();
    return isa<metal::DeviceMakeCommandQueueOp>(secondOp);
  }
  return false;
}

namespace {
struct ConvertFuncToFunc
    : public impl::ConvertFuncToFuncBase<ConvertFuncToFunc> {

  using impl::ConvertFuncToFuncBase<ConvertFuncToFunc>::ConvertFuncToFuncBase;

  void runOnOperation() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<metal::MetalDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>(
        [](func::FuncOp op) { return isFirstOpDeviceOp(op, "main"); });

    RewritePatternSet patterns(&getContext());
    mlir::metal::populateFuncToFuncConversionPatterns(patterns, &getContext());

    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPartialConversion(getOperation(), target, patternSet)))
      signalPassFailure();
  }
};

} // end namespace
} // namespace mlir::metal

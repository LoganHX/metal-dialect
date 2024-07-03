
//===--- GpuLaunchToMetal.cpp
//--------------------------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal/Conversion/LinalgToMetal.h"
#include "metal/IR/MetalOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"

#include <iostream>

namespace {

using namespace mlir;
mlir::metal::DeviceMakeDefaultOp device;
mlir::metal::DeviceMakeCommandQueueOp queue;

void retrieveDeviceAndQueue(Operation *op) {
  //TODO sarebbe meglio se restituisse qualcosa di informativo
  auto parent = op->getParentOp();
  while (parent) {
    if (auto funcOp = dyn_cast<func::FuncOp>(parent)) {
      auto &body = funcOp.getBody();
      if (body.empty())
        return;
      auto &firstBlock = body.front();
      if (firstBlock.empty())
        return;

      auto *firstOp = &firstBlock.front();
      if (isa<metal::DeviceMakeDefaultOp>(firstOp)) {
        device = dyn_cast<mlir::metal::DeviceMakeDefaultOp>(*firstOp);
      }
      if (firstOp->getNextNode()) {
        auto *secondOp = firstOp->getNextNode();
        if (isa<metal::DeviceMakeCommandQueueOp>(secondOp))
          queue = dyn_cast<mlir::metal::DeviceMakeCommandQueueOp>(*secondOp);
        return;
      }
    }
    parent = parent->getParentOp();
  }
  return;
};

mlir::metal::DeviceMakeCommandQueueOp getQueue(Operation *op) {
  if (queue)
    return queue;
  retrieveDeviceAndQueue(op);
  return queue;
}



struct ConvertMatmulOp : public OpConversionPattern<linalg::MatmulOp> {
  ConvertMatmulOp(mlir::MLIRContext *context)
      : OpConversionPattern<linalg::MatmulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto intValue = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getIntegerType(32, false),
        rewriter.getIntegerAttr(rewriter.getIntegerType(32, false), 32));

    auto rep = rewriter.create<mlir::metal::MatmulOp>(
        op.getLoc(), rewriter.getIndexType(), getQueue(op),
        adaptor.getOperands()[0],
        adaptor.getOperands()[0].getDefiningOp()->getOperand(2),
        adaptor.getOperands()[0].getDefiningOp()->getOperand(3),
        adaptor.getOperands()[1],
        adaptor.getOperands()[1].getDefiningOp()->getOperand(2),
        adaptor.getOperands()[1].getDefiningOp()->getOperand(3),
        adaptor.getOperands()[2], intValue);
    rewriter.replaceOp(op, rep);

    return success();
  }
};


} // end namespace

void mlir::metal::populateLinalgToMetalConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {

  patterns.insert<ConvertMatmulOp>(ctx);
}
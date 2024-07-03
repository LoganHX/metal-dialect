
//===--- FuncToFunc.cpp
//--------------------------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal/Conversion/FuncToFunc.h"
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
#include "metal/IR/MetalOps.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"

#include <iostream>

namespace {

using namespace mlir;

struct ConvertFuncOp : public OpConversionPattern<func::FuncOp> {
  ConvertFuncOp(mlir::MLIRContext *context)
      : OpConversionPattern<func::FuncOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();

      rewriter.modifyOpInPlace(op, [&] {
      rewriter.setInsertionPointToStart(&op.getBody().front());
      auto deviceOp = rewriter.create<mlir::metal::DeviceMakeDefaultOp>(op.getLoc());
      auto queueOp = rewriter.create<mlir::metal::DeviceMakeCommandQueueOp>(op.getLoc(), deviceOp);
    });
               


    return success();
  }
};

} // end namespace

void mlir::metal::populateFuncToFuncConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ConvertFuncOp>(ctx);
}

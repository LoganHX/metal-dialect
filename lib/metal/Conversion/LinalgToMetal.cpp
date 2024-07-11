
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

mlir::Type getElementType(mlir::Type type) {
  if (type.isa<mlir::MemRefType>()) {
    return type.cast<mlir::MemRefType>().getElementType();
  }
  // Per altri tipi semplici, restituisci il tipo stesso
  return type;
}

using namespace mlir;
mlir::metal::DeviceMakeDefaultOp device;
mlir::metal::DeviceMakeCommandQueueOp queue;

struct ConvertMatmulOp : public OpConversionPattern<linalg::MatmulOp> {
  ConvertMatmulOp(mlir::MLIRContext *context)
      : OpConversionPattern<linalg::MatmulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto rep = rewriter.create<mlir::metal::MatmulOp>(
        op.getLoc(), rewriter.getIndexType(), nullptr, 
        adaptor.getOperands()[0], nullptr, nullptr, 
        adaptor.getOperands()[1], nullptr, nullptr,
        adaptor.getOperands()[2],
        getElementType(adaptor.getOperands()[2].getType()));
    rewriter.eraseOp(op);

    return success();
  }
};

} // end namespace

void mlir::metal::populateLinalgToMetalConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {

  patterns.insert<ConvertMatmulOp>(ctx);
}
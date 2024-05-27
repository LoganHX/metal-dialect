//===--- MetalToLLVM.cpp --------------------------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal/Conversion/ScfToMetal.h"
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

#include <iostream>

namespace {

using namespace mlir;

// class ScfToMetalTypeConverter : public mlir::TypeConverter {
// public:
//   ScfToMetalTypeConverter() {
//     addConversion([](mlir::metal::MetalMemRefType type) {
//       return mlir::MemRefType::get(type.getSize(), type.getType());
//     });
//   }
// };


struct ConvertWhile : public OpConversionPattern<mlir::scf::WhileOp> {
  ConvertWhile(mlir::MLIRContext *context)
      : OpConversionPattern<mlir::scf::WhileOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOp(op, op);
    return success();
  }
};
} // end namespace

void mlir::metal::populateScfToMetalConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {

  patterns.insert<ConvertWhile>(ctx);
}

//===--- GpuLaunchToMetal.cpp
//--------------------------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal/Conversion/GpuLaunchToMetal.h"
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

mlir::metal::DeviceMakeDefaultOp
getDevice(ConversionPatternRewriter &rewriter, mlir::Location loc) {
  if (device)
    return device;
  device = rewriter.create<mlir::metal::DeviceMakeDefaultOp>(loc);
  return device;
}

mlir::metal::DeviceMakeCommandQueueOp
getQueue(ConversionPatternRewriter &rewriter, mlir::Location loc) {
  if (queue)
    return queue;
  queue = rewriter.create<mlir::metal::DeviceMakeCommandQueueOp>(
      loc, getDevice(rewriter, loc));
  return queue;
}

struct ConvertStoreOp : public OpConversionPattern<memref::StoreOp> {
  ConvertStoreOp(mlir::MLIRContext *context)
      : OpConversionPattern<memref::StoreOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto intValue = rewriter.create<emitc::ConstantOp>(
          op.getLoc(), rewriter.getIntegerType(32, false), rewriter.getIntegerAttr(rewriter.getIntegerType(32, false), 0));
          
    auto rep = rewriter.create<mlir::metal::StoreOp>(
      op.getLoc(),  adaptor.getValue(), adaptor.getMemref(), intValue );

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertAllocOp : public OpConversionPattern<memref::AllocOp> {
  ConvertAllocOp(mlir::MLIRContext *context)
      : OpConversionPattern<memref::AllocOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto boolValue = rewriter.create<emitc::ConstantOp>(
          op.getLoc(), rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
    auto intValue = rewriter.create<emitc::ConstantOp>(
          op.getLoc(), rewriter.getI64Type(), rewriter.getIntegerAttr(rewriter.getI64Type(), 64));
    auto rep = rewriter.create<mlir::metal::DeviceMakeBufferOp>(
            op.getLoc(), getDevice(rewriter, op.getLoc()), boolValue, intValue, intValue);
    rewriter.replaceOp(op, rep);
    return success();
  }
};

struct ConvertLaunchFuncOp : public OpConversionPattern<gpu::LaunchFuncOp> {
  ConvertLaunchFuncOp(mlir::MLIRContext *context)
      : OpConversionPattern<gpu::LaunchFuncOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::LaunchFuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto dimX = adaptor.getBlockSizeX();
    auto dimY = adaptor.getBlockSizeY();
    auto dimZ = adaptor.getBlockSizeZ();

    // TODO sarebbe meglio modificare metal::CommandQueueMakeCommandBufferOp
    //  in modo che accetti ConstantIndex come dimensioni X, Y e Z.


    if (dimX.getType() != rewriter.getI64Type()) {
      // servirebbe un check per vedere se dimX è un emit::ConstantOp
      dimX = rewriter.create<emitc::CastOp>(op.getLoc(), rewriter.getI64Type(),
                                            dimX);
    }
    if (dimY.getType() != rewriter.getI64Type()) {
      // servirebbe un check per vedere se dimY è un emit::ConstantOp
      dimY = rewriter.create<emitc::CastOp>(op.getLoc(), rewriter.getI64Type(),
                                            dimY);
    }
    if (dimZ.getType() != rewriter.getI64Type()) {
      // servirebbe un check per vedere se dimZ è un emit::ConstantOp
      dimZ = rewriter.create<emitc::CastOp>(op.getLoc(), rewriter.getI64Type(),
                                            dimZ);
    }

    auto commandBuffer =
        rewriter.create<mlir::metal::CommandQueueMakeCommandBufferOp>(
            op.getLoc(), getQueue(rewriter, op.getLoc()), op.getKernelModuleName(), dimX, dimY,
            dimZ);

    auto intValue = rewriter.create<emitc::ConstantOp>(
          op.getLoc(), rewriter.getI64Type(), rewriter.getIntegerAttr(rewriter.getI64Type(), 0));
    rewriter.create<mlir::metal::CommandBufferAddBufferOp>(
            op.getLoc(), commandBuffer, adaptor.getKernelOperands().back(), intValue);

    rewriter.eraseOp(op);
    return success();
  }
};

} // end namespace

void mlir::metal::populateGpuLaunchToMetalConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {

  patterns.insert<ConvertLaunchFuncOp, ConvertStoreOp, ConvertAllocOp>(ctx);
}

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
bool shouldMapToUnsigned(IntegerType::SignednessSemantics val) {
  switch (val) {
  case IntegerType::Signless:
    return false;
  case IntegerType::Signed:
    return false;
  case IntegerType::Unsigned:
    return true;
  }
}

SmallVector<Value, 4> getMemrefDims(Operation *op,
                                    ConversionPatternRewriter &rewriter,
                                    MemRefType mt, Type type) {
  SmallVector<Value, 4> dims;
  for (size_t i = 0; i < mt.getShape().size(); i++) {
    dims.push_back(rewriter.create<emitc::ConstantOp>(
        op->getLoc(), type, rewriter.getIntegerAttr(type, mt.getDimSize(i))));
  }

  return dims;
}

mlir::metal::DeviceMakeDefaultOp getDevice(ConversionPatternRewriter &rewriter,
                                           mlir::Location loc) {
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

struct ConvertDeallocOp : public OpConversionPattern<memref::DeallocOp> {
  ConvertDeallocOp(mlir::MLIRContext *context)
      : OpConversionPattern<memref::DeallocOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto rep = rewriter.create<mlir::metal::ReleaseOp>(op.getLoc(),
                                                       adaptor.getMemref());
    rewriter.replaceOp(op, rep);
    return success();
  }
};

struct ConvertLoadOp : public OpConversionPattern<memref::LoadOp> {
  ConvertLoadOp(mlir::MLIRContext *context)
      : OpConversionPattern<memref::LoadOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Operation *adaptedOp = adaptor.getMemref().getDefiningOp();

    auto deviceOp =
        llvm::dyn_cast_or_null<metal::DeviceMakeBufferOp>(adaptedOp);
    
    if (!deviceOp) {
      // auto rep = rewriter.create<mlir::metal::GetElementOp>(
      //     op.getLoc(), op.getMemRefType().getElementType(), op.getMemref(), op.getIndices(),
      //     getMemrefDims(op.getOperation(), rewriter, op.getMemref().getType(),
      //                   rewriter.getI64Type()));

      // rewriter.replaceOp(op, rep);

      // return success();
      return failure();
    }

    auto rep = rewriter.create<mlir::metal::GetElementOp>(
        op.getLoc(), deviceOp.getElementTypeAttr().getValue(),
        adaptor.getMemref(), adaptor.getIndices(), deviceOp.getDims());

    rewriter.replaceOp(op, rep.getResult());

    return success();
  }
};

struct ConvertStoreOp : public OpConversionPattern<memref::StoreOp> {
  ConvertStoreOp(mlir::MLIRContext *context)
      : OpConversionPattern<memref::StoreOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Operation *adaptedOp = adaptor.getMemref().getDefiningOp();

    auto deviceOp =
        llvm::dyn_cast_or_null<metal::DeviceMakeBufferOp>(adaptedOp);
    if (!deviceOp) {
      // auto rep = rewriter.create<mlir::metal::StoreOp>(
      //     op.getLoc(), adaptor.getValue(), op.getMemref(), op.getIndices(),
      //     getMemrefDims(op.getOperation(), rewriter, op.getMemref().getType(),
      //                   rewriter.getI64Type()));

      // rewriter.replaceOp(op, rep);

      return failure();
    }

    auto rep = rewriter.create<mlir::metal::StoreOp>(
        op.getLoc(), adaptor.getValue(), adaptor.getMemref(),
        adaptor.getIndices(), deviceOp.getDims());

    rewriter.replaceOp(op, rep); // TODO perché non getResult()?

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

    // isStorageModeManaged
    auto boolValue = rewriter.create<emitc::ConstantOp>(
        op.getLoc(), rewriter.getI1Type(),
        rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

    SmallVector<Value, 4> dims =
        getMemrefDims(op.getOperation(), rewriter, op.getMemref().getType(),
                      rewriter.getI64Type());
    auto rep = rewriter.create<mlir::metal::DeviceMakeBufferOp>(
        op.getLoc(), getDevice(rewriter, op.getLoc()), boolValue, dims,
        op.getMemref().getType().getElementType());

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

    auto dimX = adaptor.getGridSizeX();
    auto dimY = adaptor.getGridSizeY();
    auto dimZ = adaptor.getGridSizeZ();

    // TODO sarebbe meglio modificare metal::CommandQueueMakeCommandBufferOp
    //  in modo che accetti ConstantIndex come dimensioni X, Y e Z?

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
            op.getLoc(), getQueue(rewriter, op.getLoc()),
            op.getKernelModuleName(), dimX, dimY, dimZ);

    for (size_t i = 0; i < adaptor.getKernelOperands().size(); i++) {
      if (isa<MemRefType>(op.getKernelOperands()[i].getType())) {
        auto intValue = rewriter.create<emitc::ConstantOp>(
            op.getLoc(), rewriter.getI64Type(),
            rewriter.getIntegerAttr(rewriter.getI64Type(), (int64_t)i));
        rewriter.create<mlir::metal::CommandBufferAddBufferOp>(
            op.getLoc(), commandBuffer, adaptor.getKernelOperands()[i],
            intValue);
      }
    }

    rewriter.create<mlir::metal::CommandBufferCommitOp>(op.getLoc(),
                                                        commandBuffer);
    rewriter.create<mlir::metal::CommandBufferWaitUntilCompletedOp>(
        op.getLoc(), commandBuffer);
    rewriter.create<mlir::metal::ReleaseOp>(op.getLoc(), commandBuffer);

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertMatmulOp : public OpConversionPattern<linalg::MatmulOp> {
  ConvertMatmulOp(mlir::MLIRContext *context)
      : OpConversionPattern<linalg::MatmulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto intValue = rewriter.create<emitc::ConstantOp>(
        op.getLoc(), rewriter.getIntegerType(32, false),
        rewriter.getIntegerAttr(rewriter.getIntegerType(32, false), 32));

    auto rep = rewriter.create<mlir::metal::MatmulOp>(
        op.getLoc(), rewriter.getIndexType(), getQueue(rewriter, op.getLoc()),
        adaptor.getOperands()[0],
        adaptor.getOperands()[0].getDefiningOp()->getOperand(2),
        adaptor.getOperands()[0].getDefiningOp()->getOperand(3),
        adaptor.getOperands()[1],
        adaptor.getOperands()[1].getDefiningOp()->getOperand(2),
        adaptor.getOperands()[1].getDefiningOp()->getOperand(3),
        adaptor.getOperands()[2], intValue);
    // rewriter.replaceOp(op, rep);
    rewriter.eraseOp(op);

    return success();
  }
};

// struct ConvertReturnOp : public OpConversionPattern<func::ReturnOp> {
//   ConvertReturnOp(mlir::MLIRContext *context)
//       : OpConversionPattern<func::ReturnOp>(context) {}

//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
 
//     rewriter.eraseOp(op);

//     return success();
//   }
// };

} // end namespace

void mlir::metal::populateGpuLaunchToMetalConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {

  patterns.insert<ConvertLaunchFuncOp, ConvertStoreOp, ConvertAllocOp,
                  ConvertDeallocOp, ConvertLoadOp, ConvertMatmulOp>(ctx);
}
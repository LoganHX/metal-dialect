
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

// TODO genero troppe costanti. Constant Folding? Da approfondire.

emitc::ConstantOp getMemrefDim(Location loc,
                               ConversionPatternRewriter &rewriter,
                               MemRefType mt, size_t dim) {
  if (mt.getShape().size() > 3)
    return nullptr; // TODO dovrei emettere un errore

  if (dim > mt.getShape().size() - 1)
    return rewriter.create<emitc::ConstantOp>(
        loc, rewriter.getIntegerType(32, false),
        rewriter.getIntegerAttr(rewriter.getIntegerType(32, false), 1));

  return rewriter.create<emitc::ConstantOp>(
      loc, rewriter.getIntegerType(32, false),
      rewriter.getIntegerAttr(rewriter.getIntegerType(32, false),
                              mt.getDimSize(dim)));
}

mlir::Value getIndex(Location loc, ConversionPatternRewriter &rewriter,
                     ValueRange indices, size_t dim) {
  if (indices.size() > 3)
    return nullptr; // TODO dovrei emettere un errore

  if (dim > indices.size() - 1)
    return rewriter.create<emitc::ConstantOp>(
        loc, rewriter.getIntegerType(32, false),
        rewriter.getIntegerAttr(rewriter.getIntegerType(32, false), 0));

  return indices[dim];
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
    rewriter.create<mlir::metal::ReleaseOp>(op.getLoc(), adaptor.getMemref());
    rewriter.eraseOp(op);
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
      /*TODO è un workaround ->vedi ConvertStoreOp */

    if (isa<MemRefType>(adaptor.getMemref().getType()))
      return failure();
    rewriter.create<mlir::metal::GetElementOp>(
        op.getLoc(), op.getMemref().getType().getElementType(),
        adaptor.getMemref(),
        getIndex(op.getLoc(), rewriter, adaptor.getIndices(), 0),
        getIndex(op.getLoc(), rewriter, adaptor.getIndices(), 1),
        getIndex(op.getLoc(), rewriter, adaptor.getIndices(), 2),
        getMemrefDim(op.getLoc(), rewriter, op.getMemRefType(), 0),
        getMemrefDim(op.getLoc(), rewriter, op.getMemRefType(), 1),
        getMemrefDim(op.getLoc(), rewriter, op.getMemRefType(), 2));

    rewriter.eraseOp(op);
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
    /*TODO è un workaround
      mi serve per distinguere gli store che appartengono a un
      gpu.kernel (quelli che hanno tipo memref) da quelli di tipo index che
      stanno "lato CPU". Andrebbe fatto diversamente, attraversando
      "consciamente" l'alber:  getFunction().walk([&](memref::StoreOp storeOp)????

      Di questo workaround fa parte anche aver commentato l'istruzione signalPassFailure(); 
      nel file ConvertGpuLaunchToMetal.cpp, in modo da sopprimere l'errore e stampare a video
      il risultato a prescindere dall'esito.
    */

    if (isa<MemRefType>(adaptor.getMemref().getType()))
      return failure();
    rewriter.create<mlir::metal::StoreOp>(
        op.getLoc(), adaptor.getValue(), adaptor.getMemref(),
        getIndex(op.getLoc(), rewriter, adaptor.getIndices(), 0),
        getIndex(op.getLoc(), rewriter, adaptor.getIndices(), 1),
        getIndex(op.getLoc(), rewriter, adaptor.getIndices(), 2),
        getMemrefDim(op.getLoc(), rewriter, op.getMemRefType(), 0),
        getMemrefDim(op.getLoc(), rewriter, op.getMemRefType(), 1),
        getMemrefDim(op.getLoc(), rewriter, op.getMemRefType(), 2));

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
        op.getLoc(), rewriter.getI1Type(),
        rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
    auto intValue = rewriter.create<emitc::ConstantOp>(
        op.getLoc(), rewriter.getI64Type(),
        rewriter.getIntegerAttr(rewriter.getI64Type(), 64));
    auto rep = rewriter.create<mlir::metal::DeviceMakeBufferOp>(
        op.getLoc(), getDevice(rewriter, op.getLoc()), boolValue, intValue,
        intValue);
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

} // end namespace

void mlir::metal::populateGpuLaunchToMetalConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {

  patterns.insert<ConvertLaunchFuncOp, ConvertStoreOp, ConvertAllocOp,
                  ConvertDeallocOp, ConvertLoadOp>(ctx);
}
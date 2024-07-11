
//===--- GpuLaunchToMetal.cpp
//--------------------------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal/Conversion/GpuLaunchToMetal.h"
#include "metal/IR/MetalOps.h"
#include "shader/IR/ShaderOps.h"

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

#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>

namespace {

using namespace mlir;
mlir::metal::DeviceMakeDefaultOp device;
mlir::metal::DeviceMakeCommandQueueOp queue;

void retrieveDeviceAndQueue(Operation *op) {
  // TODO sarebbe meglio se restituisse qualcosa di informativo
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

mlir::metal::DeviceMakeDefaultOp getDevice(Operation *op) {
  if (device) {
    return device;
  }
  retrieveDeviceAndQueue(op);
  return device;
}

mlir::metal::DeviceMakeCommandQueueOp getQueue(Operation *op) {
  if (queue)
    return queue;
  retrieveDeviceAndQueue(op);
  return queue;
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
        op.getLoc(), getDevice(op), boolValue, dims,
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

    auto commandBuffer =
        rewriter.create<mlir::metal::CommandQueueMakeCommandBufferOp>(
            op.getLoc(), getQueue(op), op.getKernelModuleName(), dimX, dimY,
            dimZ);

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

struct LegalizeFuncOp : public OpConversionPattern<func::FuncOp> {
  LegalizeFuncOp(mlir::MLIRContext *context)
      : OpConversionPattern<func::FuncOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::vector<Type> argument_types;
    for (auto arg : op.getBody().front().getArguments()) {
      argument_types.push_back(arg.getType());
    }
    std::vector<Type> return_types;
    return_types.push_back(rewriter.getIndexType());
    auto newType =
        FunctionType::get(rewriter.getContext(), argument_types, return_types);
    rewriter.modifyOpInPlace(op, [&] { op.setFunctionType(newType); });

    return success();
  }
};

struct LegalizeCallOp : public OpConversionPattern<func::CallOp> {
  LegalizeCallOp(mlir::MLIRContext *context)
      : OpConversionPattern<func::CallOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<Value, 4> newOperands;

    for (size_t i = 0; i < op.getNumOperands(); i++) {
      Value operand = adaptor.getOperands()[i];

      if (isa<IndexType>(operand.getType()) &&
          isa<MemRefType>(op.getOperand(i).getType()) &&
          isa<metal::DeviceMakeBufferOp>(operand.getDefiningOp())) {
        auto intptrToPtrOp = rewriter.create<metal::IntptrToPtrOp>(
            op.getLoc(), op.getOperandTypes()[i], operand);

        newOperands.push_back(intptrToPtrOp);
      } else {
        newOperands.push_back(operand);
      }
    }

    rewriter.modifyOpInPlace(op, [&] {
      for (size_t i = 0; i < op.getNumOperands(); i++) {
        op.setOperand(i, newOperands[i]);
      }
    });
    return success();
  }
};

struct LegalizeReturnOp : public OpConversionPattern<func::ReturnOp> {
  LegalizeReturnOp(mlir::MLIRContext *context)
      : OpConversionPattern<func::ReturnOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });

    return success();
  }
};

struct LegalizeMatmulOp : public OpConversionPattern<shader::MatmulOp> {
  LegalizeMatmulOp(mlir::MLIRContext *context)
      : OpConversionPattern<shader::MatmulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(shader::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto rep = rewriter.create<mlir::shader::MatmulOp>(
        op.getLoc(), rewriter.getIndexType(), getQueue(op),
        adaptor.getOperands()[0],
        adaptor.getOperands()[0].getDefiningOp()->getOperand(2),
        adaptor.getOperands()[0].getDefiningOp()->getOperand(3),
        adaptor.getOperands()[1],
        adaptor.getOperands()[1].getDefiningOp()->getOperand(2),
        adaptor.getOperands()[1].getDefiningOp()->getOperand(3),
        adaptor.getOperands()[2],
        adaptor.getElementType());
    rewriter.create<mlir::metal::CommandBufferCommitOp>(op.getLoc(),
                                                        rep.getResult());
    rewriter.create<mlir::metal::CommandBufferWaitUntilCompletedOp>(
        op.getLoc(), rep.getResult());
    rewriter.create<mlir::metal::ReleaseOp>(op.getLoc(), rep.getResult());

    rewriter.eraseOp(op);

    return success();
  }
};

} // end namespace

void mlir::metal::populateGpuLaunchToMetalConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {

  patterns.insert<ConvertLaunchFuncOp, ConvertStoreOp, ConvertAllocOp,
                  ConvertDeallocOp, ConvertLoadOp, LegalizeReturnOp,
                  LegalizeFuncOp, LegalizeMatmulOp, LegalizeCallOp>(ctx);
}
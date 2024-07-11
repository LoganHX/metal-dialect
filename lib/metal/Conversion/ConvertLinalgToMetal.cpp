//===--- ConvertLinalgToMetal.cpp--------------------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "metal/Conversion/LinalgToMetal.h"
#include "metal/Conversion/MetalPasses.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "shader/IR/ShaderOps.h"
#include "shader/IR/ShaderDialect.h"

#include "mlir/Transforms/DialectConversion.h"
#include <iostream>

namespace mlir::metal {

#define GEN_PASS_DEF_CONVERTLINALGTOMETAL
#include "metal/Conversion/MetalPasses.h.inc"



namespace mlir{
struct ConvertLinalgToMetal
    : public impl::ConvertLinalgToMetalBase<ConvertLinalgToMetal> {

  using impl::ConvertLinalgToMetalBase<
      ConvertLinalgToMetal>::ConvertLinalgToMetalBase;

  void runOnOperation() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<metal::MetalDialect>();
    target.addLegalDialect<shader::ShaderDialect>();
    target.addLegalDialect<gpu::GPUDialect>();
    target.addLegalDialect<tosa::TosaDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();

    target.addIllegalOp<linalg::MatmulOp>();
    target.addIllegalOp<linalg::MatmulTransposeAOp>();
    target.addIllegalOp<linalg::MatmulTransposeBOp>();
    target.addIllegalOp<linalg::AddOp>();
   
    RewritePatternSet patterns(&getContext());
    metal::populateLinalgToMetalConversionPatterns(patterns,
                                                            &getContext());

    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPartialConversion(getOperation(), target, patternSet)))
      signalPassFailure();
  }
};

} // end namespace
} // namespace mlir::metal
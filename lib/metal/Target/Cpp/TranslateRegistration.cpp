//===- TranslateRegistration.cpp - Register translation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "metal/IR/MetalDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "metal/IR/MetalOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"



#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"


#include "metal/Target/Cpp/MetalEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "metal/Target/Cpp/TranslateRegistration.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace mlir {
  namespace metal {

//===----------------------------------------------------------------------===//
// Cpp registration
//===----------------------------------------------------------------------===//

void registerToMetalTranslation() {
  static llvm::cl::opt<bool> declareVariablesAtTop(
      "declare-variables-at-top",
      llvm::cl::desc("Declare variables at top when emitting C/C++"),
      llvm::cl::init(false));

  TranslateFromMLIRRegistration reg(
      "mlir-to-metal", "translate from mlir to metal",
      [](Operation *op, raw_ostream &output) {
        return translateToMetal(
            op, output,
            /*declareVariablesAtTop=*/declareVariablesAtTop);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<cf::ControlFlowDialect,
                        gpu::GPUDialect,
                        linalg::LinalgDialect,
                        metal::MetalDialect,
                        mlir::arith::ArithDialect,
                        math::MathDialect,
                        memref::MemRefDialect,
                        mlir::bufferization::BufferizationDialect,
                        emitc::EmitCDialect,
                        func::FuncDialect>();
        // clang-format on
      });
}
} // namespace metal
} // namespace mlir
//===--- FuncToFunc.h ------------------------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#pragma once


#include <memory>

namespace mlir {

class MLIRContext;
class RewritePatternSet;
class Pass;

namespace metal {
void populateFuncToFuncConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx);
} // end namespace metal
} // end namespace mlir


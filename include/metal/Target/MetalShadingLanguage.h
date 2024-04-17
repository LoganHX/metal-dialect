//===--- MetalShadingLanguage.h ---------------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#pragma once


#include "mlir/Support/LogicalResult.h"

namespace mlir {
class ModuleOp;

namespace metal {
mlir::LogicalResult translateModuleToMetalShadingLanguage(mlir::ModuleOp m,
                                                          raw_ostream &output);

} // end namespace metal
} // end namespace mlir


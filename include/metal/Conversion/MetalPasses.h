//===--- MetalPasses.h - Metal passes ---------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#pragma once

#include "metal/IR/MetalDialect.h"
#include "metal/IR/MetalOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace metal {
#define GEN_PASS_DECL
#include "metal/Conversion/MetalPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "metal/Conversion/MetalPasses.h.inc"
} // namespace metal
} // namespace mlir


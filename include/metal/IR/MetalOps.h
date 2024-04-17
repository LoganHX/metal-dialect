//===--- MetalOps.h - Metal dialect ops -------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#pragma once


#include "mlir/IR/BuiltinAttributes.h"

#include "metal/IR/MetalOpsEnums.h.inc"
#include "metal/IR/MetalTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "metal/IR/MetalOps.h.inc"

//===--- ShaderOps.h - Shader dialect ops -------------------------*- C++ -*-===//
//
// This source file is part of the shader-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#pragma once


#include "mlir/IR/BuiltinAttributes.h"

#include "shader/IR/ShaderOpsEnums.h.inc"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "shader/IR/ShaderTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "shader/IR/ShaderOps.h.inc"

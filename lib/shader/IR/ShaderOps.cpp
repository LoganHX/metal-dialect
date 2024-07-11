//===--- ShaderOps.cpp - Shader dialect ops ---------------------------------===//
//
// This source file is part of the shader-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "shader/IR/ShaderOps.h"
#include "shader/IR/ShaderDialect.h"
#include "shader/IR/ShaderOpsEnums.cpp.inc"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir::shader;

#define GET_OP_CLASSES
#include "shader/IR/ShaderOps.cpp.inc"

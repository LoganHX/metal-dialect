//===--- ShaderDialect.cpp - Shader dialect ---------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "shader/IR/ShaderDialect.h"
#include "shader/IR/ShaderOps.h"
#include "shader/IR/ShaderTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Dialect.h"

using namespace mlir;
using namespace mlir::shader;

#include "shader/IR/ShaderOpsDialect.cpp.inc"

void ShaderDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "shader/IR/ShaderOps.cpp.inc"
      >();
}


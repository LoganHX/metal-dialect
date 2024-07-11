//===- ShaderTypes.cpp - Shader dialect types ---------------------*- C++ -*-===//
//
// This source file is part of the Shader open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "shader/IR/ShaderTypes.h"
#include "shader/IR/ShaderDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::shader;

#define GET_TYPEDEF_CLASSES
#include "shader/IR/ShaderOpsTypes.cpp.inc"



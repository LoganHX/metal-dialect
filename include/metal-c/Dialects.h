//===- Dialects.h - CAPI for dialects -----------------------------*- C -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Metal, metal);

#ifdef __cplusplus
}
#endif

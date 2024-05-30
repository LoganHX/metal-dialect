//===- mlir-translate.cpp - MLIR Translate Driver -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "metal/Target/Cpp/TranslateRegistration.h"

#include <iostream>

using namespace mlir;

namespace mlir::metal {
// // Defined in the test directory, no public header.
// void registerTestRoundtripSPIRV();
// void registerTestRoundtripDebugSPIRV();
// #ifdef MLIR_INCLUDE_TESTS
// void registerTestToLLVMIR();
// void registerTestFromLLVMIR();
// #endif
// } // namespace mlir

// static void registerTestTranslations() {
//   registerTestRoundtripSPIRV();
//   registerTestRoundtripDebugSPIRV();
// #ifdef MLIR_INCLUDE_TESTS
//   registerTestToLLVMIR();
//   registerTestFromLLVMIR();
// #endif
}

int main(int argc, char **argv) {

  mlir::metal::registerMetalToCppTranslation();
  return failed(mlir::mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
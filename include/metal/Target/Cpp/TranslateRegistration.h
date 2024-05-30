//===- TranslateRegistration.cpp - Register translation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_TRANSLATE_REGISTRATION_H 
#define MLIR_TARGET_TRANSLATE_REGISTRATION_H


using namespace mlir;

namespace mlir {
  namespace metal {

//===----------------------------------------------------------------------===//
// Cpp registration
//===----------------------------------------------------------------------===//

void registerToMetalTranslation();

} // namespace metal
} // namespace mlir

#endif
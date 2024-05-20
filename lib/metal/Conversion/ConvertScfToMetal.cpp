#include "metal/Conversion/MetalPasses.h"
#include "metal/Conversion/ScfToMetal.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include <iostream>

namespace mlir::metal {

#define GEN_PASS_DEF_CONVERTSCFTOMETAL
#include "metal/Conversion/MetalPasses.h.inc"

namespace {
struct ConvertScfToMetal
    : public impl::ConvertScfToMetalBase<ConvertScfToMetal> {

  using impl::ConvertScfToMetalBase<
      ConvertScfToMetal>::ConvertScfToMetalBase;

  void runOnOperation() final {
    return;
  }
};

} // end namespace
} // namespace mlir::metal
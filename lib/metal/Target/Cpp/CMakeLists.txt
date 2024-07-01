add_mlir_translation_library(MLIRTargetMetal
  TranslateRegistration.cpp
  TranslateToMetal.cpp

  ADDITIONAL_HEADER_DIRS
  ${EMITC_MAIN_INCLUDE_DIR}/emitc/Target/Cpp

  LINK_LIBS PUBLIC
  MLIRControlFlowDialect
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  )
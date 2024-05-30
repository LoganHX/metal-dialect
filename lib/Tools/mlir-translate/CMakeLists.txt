  add_mlir_library(MetalTranslateLib
  MlirTranslateMain.cpp
  Translation.cpp

  ADDITIONAL_HEADER_DIRS
  ${METAL_SOURCE_DIR}/Tools/mlir-translate

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  )
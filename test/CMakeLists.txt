llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(METAL_TEST_DEPENDS
        FileCheck count not
        metal-capi-test
        metal-opt
        metal-translate
        )
if(MLIR_ENABLE_BINDINGS_PYTHON)
    list(APPEND METAL_TEST_DEPENDS MetalPythonModules)
endif()



add_subdirectory(CAPI)

set(LIB_HEADERS  ${PROJECT_SOURCE_DIR}/include/shader)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)


add_mlir_dialect_library(MLIRShader

		${LIB_HEADERS}/IR/ShaderDialect.h
		IR/ShaderDialect.cpp
		${LIB_HEADERS}/IR/ShaderOps.h
		IR/ShaderOps.cpp
		${LIB_HEADERS}/IR/ShaderTypes.h
		IR/ShaderTypes.cpp


		DEPENDS
		MLIRShaderOpsIncGen
		MLIRShaderOpsEnumsGen
		

		LINK_LIBS PUBLIC
		MLIRIR
		${conversion_libs}
)

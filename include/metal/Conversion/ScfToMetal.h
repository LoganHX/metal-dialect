#pragma once


#include <memory>

namespace mlir {

class MLIRContext;
class RewritePatternSet;
class Pass;

namespace metal {
void populateScfToMetalConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx);
} // end namespace metal
} // end namespace mlir
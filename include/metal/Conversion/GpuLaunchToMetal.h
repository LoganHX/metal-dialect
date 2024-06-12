#pragma once


#include <memory>

namespace mlir {

class MLIRContext;
class RewritePatternSet;
class Pass;

namespace metal {
void populateGpuLaunchToMetalConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx);
} // end namespace metal
} // end namespace mlir
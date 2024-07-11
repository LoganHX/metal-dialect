#include "MetalRuntime.h"
@import Foundation;
@import MetalRuntimeSwift;


// _____________________________________________________________________________
// Release

void _MetalRelease(intptr_t ref) {
  @autoreleasepool {
    [Wrappable release:ref];
  }
}

// _____________________________________________________________________________
// Device

intptr_t _MetalDeviceMakeDefault(void) {
  intptr_t value = 0;
  @autoreleasepool {
    value = [[Device makeDefault] wrap];
  }
  return value;
}

intptr_t _MetalDeviceMakeCommandQueue(intptr_t ref) {
  intptr_t value = 0;
  @autoreleasepool {
    Device *instance = [Device unwrap:ref];
    value = [[instance makeCommandQueue] wrap];
  }
  return value;
}

intptr_t _MetalDeviceMakeBuffer(intptr_t ref,
                                bool isStorageModeManaged,
                                int64_t count,
                                int64_t typeSize) {
  intptr_t value = 0;
  @autoreleasepool {
    Device *instance = [Device unwrap:ref];
    value = [[instance makeBufferWithIsStorageModeManaged:isStorageModeManaged
                                               bufferSize:count * typeSize
                                                    count:count] wrap];
  }
  return value;
}


// _____________________________________________________________________________
// Buffer

void _MetalBufferGetContents(intptr_t ref, void *memRef) {
  typedef struct {
    void *allocated;
    void *aligned;
    int64_t offset;
    int64_t sizes;
    int64_t strides;
  } MemRef;
  
  @autoreleasepool {
    Buffer *instance = [Buffer unwrap:ref];
    void *contents = [instance contents];
    MemRef *mem = memRef;
    mem->allocated = contents;
    mem->aligned = contents;
    mem->offset = 0;
    mem->sizes = [instance getCount];
    mem->strides = 1;
  }
}

void* _MetalBufferGetContents2(intptr_t ref) {
  void *value;
  @autoreleasepool {
    Buffer *instance = [Buffer unwrap:ref];
    void *contents = [instance contents];
    value = contents;
  }
  return value;
}

void _MetalStore_float(intptr_t ref, int64_t index, float value){
    float *arr = (float *) _MetalBufferGetContents2(ref);
    arr[index] = value;
    return;
}

float _MetalLoad_float(intptr_t ref, int64_t index){
    float *arr = (float *) _MetalBufferGetContents2(ref);
    return arr[index];
}

void _MetalStore_int32_t(intptr_t ref, int64_t index, int32_t value){
    int32_t *arr = (int32_t *) _MetalBufferGetContents2(ref);
    arr[index] = value;
    return;
}

int32_t _MetalLoad_int32_t(intptr_t ref, int64_t index){
    int32_t *arr = (int32_t *) _MetalBufferGetContents2(ref);
    return arr[index];
}

// _____________________________________________________________________________
// CommandQueue

intptr_t _MetalCommandQueueMakeCommandBuffer(intptr_t ref,
                                             const int8_t *libPath,
                                             const int8_t *functionName,
                                             int64_t width,
                                             int64_t height,
                                             int64_t depth) {
  intptr_t value = 0;
  @autoreleasepool {
    CommandQueue *instance = [CommandQueue unwrap:ref];
    NSString *nsLibPath = [NSString stringWithCString:(const char *)libPath
                                             encoding:NSUTF8StringEncoding];
    NSString *nsFunctionName = [NSString stringWithCString:(const char *)functionName
                                                  encoding:NSUTF8StringEncoding];
    value = [[instance makeCommandBufferWithLibPath:nsLibPath
                                       functionName:nsFunctionName
                                              width:width
                                             height:height
                                              depth:depth] wrap];
  }
  return value;
}

intptr_t _MetalCommandQueueMakeCommandBufferWithDefaultLibrary(intptr_t ref,
                                                               int64_t width,
                                                               int64_t height,
                                                               int64_t depth,
                                             const int8_t *functionName
                                             ) {
 
  
  return _MetalCommandQueueMakeCommandBuffer(ref, 
                                             (int8_t*)"/Users/c.stabile12/Downloads/default.metallib",
                                             functionName,
                                             width, height, depth);
}

// _____________________________________________________________________________
// CommandBuffer

void _MetalCommandBufferAddBuffer(intptr_t ref,
                                  intptr_t bufferRef,
                                  int64_t index) {
  @autoreleasepool {
    CommandBuffer *instance = [CommandBuffer unwrap:ref];
    Buffer *buffer = [Buffer unwrap:bufferRef];
    [instance addBufferWithBuffer:buffer index:index];
  }
}

void _MetalCommandBufferCommit(intptr_t ref) {
  @autoreleasepool {
    CommandBuffer *instance = [CommandBuffer unwrap:ref];
    [instance commit];
  }
}

void _MetalCommandBufferWaitUntilCompleted(intptr_t ref) {
  @autoreleasepool {
    CommandBuffer *instance = [CommandBuffer unwrap:ref];
    [instance waitUntilCompleted];
  }
}

// _____________________________________________________________________________
// MatrixSum

intptr_t _MetalMatSum(intptr_t ref, intptr_t matA, int rowsA, int columnsA,
                        intptr_t matB, int rowsB, int columnsB,
                        intptr_t matC,
                        const char *elementType){
    void* tmpA = _MetalBufferGetContents2(matA);
    void* tmpB = _MetalBufferGetContents2(matB);
    void* tmpC = _MetalBufferGetContents2(matC);
    

    NSString *tipo = [NSString stringWithCString:(const char *)elementType
                                             encoding:NSUTF8StringEncoding];
    
    intptr_t value = 0;
    @autoreleasepool {
        CommandQueue *instance = [CommandQueue unwrap:ref];
        value = [[instance matSumWithMatA:tmpA rowsA:rowsA columnsA:columnsA
                                     matB:tmpB rowsB:rowsB columnsB:columnsB
                                     matC:tmpC elementType:tipo] wrap];
    }
  return value;
    
}

// _____________________________________________________________________________
// MatrixMultiplication

intptr_t _Generic_MatMul(intptr_t ref, intptr_t matA, int rowsA, int columnsA,
                        intptr_t matB, int rowsB, int columnsB,
                        intptr_t matC,
                        const char *elementType,
                        bool transposeA, bool transposeB){
    void* tmpA = _MetalBufferGetContents2(matA);
    void* tmpB = _MetalBufferGetContents2(matB);
    void* tmpC = _MetalBufferGetContents2(matC);
    

    NSString *tipo = [NSString stringWithCString:(const char *)elementType
                                             encoding:NSUTF8StringEncoding];
    
    intptr_t value = 0;
    @autoreleasepool {
        CommandQueue *instance = [CommandQueue unwrap:ref];
        value = [[instance matMulWithMatA:tmpA rowsA:rowsA columnsA:columnsA
                                     matB:tmpB rowsB:rowsB columnsB:columnsB
                                     matC:tmpC
                                     elementType:tipo
                                     transposeA:transposeA transposeB:transposeB] wrap];
    }
  return value;
    
}

intptr_t _MetalMatMul(intptr_t ref,
                      intptr_t matA, int rowsA, int columnsA,
                      intptr_t matB, int rowsB, int columnsB,
                      intptr_t matC,
                      const char *elementType
) {
    return _Generic_MatMul(ref, matA, rowsA, columnsA, matB, rowsB, columnsB, matC, elementType, false, false);
    
}

intptr_t _MetalMatMul_TransposeLeft(intptr_t ref,
                                    intptr_t matA, int rowsA, int columnsA,
                                    intptr_t matB, int rowsB, int columnsB,
                                    intptr_t matC,
                                    const char *elementType
              ) {
    return _Generic_MatMul(ref, matA, rowsA, columnsA, matB, rowsB, columnsB, matC, elementType, true, false);
    
}
intptr_t _MetalMatMul_TransposeRight(intptr_t ref,
                                     intptr_t matA, int rowsA, int columnsA,
                                     intptr_t matB, int rowsB, int columnsB,
                                     intptr_t matC,
                                     const char *elementType
               ) {
    return _Generic_MatMul(ref, matA, rowsA, columnsA, matB, rowsB, columnsB, matC, elementType, false, true);
    
}


void _MetalPrintMat(intptr_t ref, intptr_t mat, int rows, int columns, const char* elementType) {
    void* tmp = _MetalBufferGetContents2(mat);
        NSString *elType = [NSString stringWithCString:elementType
                                             encoding:NSUTF8StringEncoding];
        @autoreleasepool {
        CommandQueue *instance = [CommandQueue unwrap:ref];
        [instance printMatWithMat:tmp rows:rows columns:columns elementType:elType];
    }
  return;
}

// _____________________________________________________________________________

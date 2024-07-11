#ifndef MetalRuntime_h
#define MetalRuntime_h

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// _____________________________________________________________________________
// Release

void _MetalRelease(intptr_t ref);

// _____________________________________________________________________________
// Device

intptr_t _MetalDeviceMakeDefault(void);

intptr_t _MetalDeviceMakeCommandQueue(intptr_t ref);

intptr_t _MetalDeviceMakeBuffer(intptr_t ref, bool isStorageModeManaged,
                                int64_t count, int64_t sizeType);

// _____________________________________________________________________________
// Buffer

void _MetalBufferGetContents(intptr_t ref, void *memRef);

void *_MetalBufferGetContents2(intptr_t ref);

void _MetalStore_float(intptr_t ref, int64_t index, float value);

float _MetalLoad_float(intptr_t ref, int64_t index);

void _MetalStore_int32_t(intptr_t ref, int64_t index, int32_t value);

int32_t _MetalLoad_int32_t(intptr_t ref, int64_t index);

// _____________________________________________________________________________
// CommandQueue

intptr_t _MetalCommandQueueMakeCommandBuffer(intptr_t ref,
                                             const int8_t *libPath,
                                             const int8_t *functionName,
                                             int64_t width, int64_t height,
                                             int64_t depth);

intptr_t _MetalCommandQueueMakeCommandBufferWithDefaultLibrary(intptr_t ref,
                                                               int64_t width,
                                                               int64_t height,
                                                               int64_t depth,
                                             const int8_t *functionName);

// _____________________________________________________________________________
// CommandBuffer

void _MetalCommandBufferAddBuffer(intptr_t ref, intptr_t bufferRef,
                                  int64_t index);

void _MetalCommandBufferCommit(intptr_t ref);

void _MetalCommandBufferWaitUntilCompleted(intptr_t ref);

// _____________________________________________________________________________
// MatrixMultiplication


void _MetalPrintMat(intptr_t ref, intptr_t mat, int rows, int columns, const char* elementType);

//TODO credo c'entri l'indexing che ho usato
//riguardo il fatto che devo invertire i booleani di //dx e sx

intptr_t _MetalMatMul(intptr_t ref,
                      intptr_t matA, int rowsA, int columnsA,
                      intptr_t matB, int rowsB, int columnsB,
                      intptr_t matC,
                      const char *elementType
                      );
intptr_t _MetalMatMul_TransposeLeft(intptr_t ref,
                                    intptr_t matA, int rowsA, int columnsA,
                                    intptr_t matB, int rowsB, int columnsB,
                                    intptr_t matC,
                                    const char *elementType
                                    );
intptr_t _MetalMatMul_TransposeRight(intptr_t ref,
                                     intptr_t matA, int rowsA, int columnsA,
                                     intptr_t matB, int rowsB, int columnsB,
                                     intptr_t matC,
                                     const char *elementType
                                     );

// _____________________________________________________________________________


#ifdef __cplusplus
}
#endif

#endif /* MetalRuntime_h */

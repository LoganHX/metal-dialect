//===--- MetalOps.td - Metal dialect ops -------------------*- tablegen -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_OPS
#define METAL_OPS

include "metal/IR/MetalTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def Metal_Type : AnyTypeOf<[I1, UI8, I8, UI16, UI32, UI64, SI8, SI16, I32, SI32, SI64,
                            F16, F32]>;

def Int_Type : AnyTypeOf<[I1, I8, I32, I64, UI8, UI16, UI32, UI64, SI8, SI16, SI32, SI64, Index]>; 
//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

def ModuleOp : Metal_Op<"module", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"ModuleEndOp">]> {
  let summary = "Module operation";

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<(ins)>];

  let hasCustomAssemblyFormat = 1;
}

def ModuleEndOp : Metal_Op<"module_end", [
    Terminator, HasParent<"ModuleOp">]> {
  let summary = "Module end operation";

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//

def KernelOp : Metal_Op<"kernel", [SingleBlockImplicitTerminator<"ReturnOp">,
    HasParent<"ModuleOp">]> {
  let summary = "Kernel operation";

  let arguments = (ins StrAttr:$name, ArrayAttr:$address_space_device);

  let regions = (region SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "llvm::SmallVectorImpl<Type>&":$buffers,
                      "llvm::SmallVectorImpl<bool>&":$isAddressSpaceDevice)>
  ];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    mlir::Block& getEntryBlock();
    mlir::Value getBuffer(uint32_t index);
    mlir::MutableArrayRef<mlir::BlockArgument> getBuffers();
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : Metal_Op<"return", [Pure, Terminator]> {
  let summary = "Return operation";

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

def MatmulOp : Metal_Op<"matmul", [AttrSizedOperandSegments]> {
  let summary = "Matrix multiplication operation";

  let arguments = (ins Optional<Index>:$queue, 
                      AnyType:$bufferA, Optional<Int_Type>:$rowsA, Optional<Int_Type>:$columnsA, 
                      AnyType:$bufferB, Optional<Int_Type>:$rowsB, Optional<Int_Type>:$columnsB,
                      AnyType:$bufferC,
                      Optional<Int_Type>:$elementSize);
  

  let hasVerifier = 0; //TODO scrivere i verifiers

  // Aggiungi type($result) per specificare il tipo del risultato
  //let assemblyFormat = "$queue `,` $bufferA `<` $rowsA `:`$columnsA `>` $bufferB `<` $rowsB `:`$columnsB `>` $bufferC `,` $elementSize attr-dict `:` type(operands) ";

}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def StoreOp : Metal_Op<"store", [SameVariadicOperandSize]> {
  let summary = "Store operation";

  let arguments = (ins Metal_Type:$value, Index:$buffer, Variadic<Int_Type>:$indexes, Variadic<Int_Type>:$sizes);

  let hasVerifier = 0; //TODO scrivere i verifiers

  let assemblyFormat = "$value `,` $buffer `,` $indexes `[` $sizes `]` attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// GetElementOp
//===----------------------------------------------------------------------===//

def GetElementOp : Metal_Op<"get_element", [Pure, SameVariadicOperandSize]> {
  let summary = "Get element operation";

  let arguments = (ins Index:$buffer, Variadic<Int_Type>:$indexes, Variadic<Int_Type>:$sizes);
  let results = (outs Metal_Type:$result);

  //let builders = [OpBuilder<(ins "Value":$buffer, "Value":$index)>];

  let hasVerifier = 0; //TODO scrivere i verifiers

  let assemblyFormat = "$buffer `,` $indexes `[` $sizes `]` attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// ThreadIdOp
//===----------------------------------------------------------------------===//

def ThreadIdOp : Metal_Op<"thread_id", [Pure]> {
  let summary = "Thread id operation";

  let arguments = (ins StrAttr:$dimension);
  let results = (outs UI32);

  let builders = [OpBuilder<(ins "StringRef":$dimension)>];

  let hasVerifier = 1;

  let assemblyFormat = "$dimension attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// ReleaseOp
//===----------------------------------------------------------------------===//

def ReleaseOp : Metal_Op<"release"> {
  let summary = "Release operation";

  let arguments = (ins Index:$ref);

  let assemblyFormat = "$ref attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// Runtime - Device
//===----------------------------------------------------------------------===//

def DeviceMakeDefaultOp : Metal_Op<"device_make_default"> {
  let summary = "Device make operation";

  let results = (outs Index:$device);

  let builders = [OpBuilder<(ins)>];

  let assemblyFormat = "attr-dict `:` type(results)";
}

def DeviceMakeCommandQueueOp : Metal_Op<"device_make_command_queue"> {
  let summary = "Device make command queue operation";

  let arguments = (ins Index:$device);
  let results = (outs Index:$commandQueue);

  let builders = [
    OpBuilder<(ins "Value":$device)>
  ];

  let assemblyFormat = "$device attr-dict `:` functional-type(operands, results)";
}

def DeviceMakeBufferOp : Metal_Op<"device_make_buffer"> {
  let summary = "Device make buffer operation";

  let arguments = (ins Index:$device, I1:$isStorageModeManaged,
                       Variadic<Int_Type>:$dims, TypeAttr:$elementType);

  let results = (outs Index:$bufferOp);

  let builders = [
    OpBuilder<(ins "Value":$device, "Value":$isStorageModeManaged, "ArrayRef<Value>":$dims, "Type":$elementType)>
  ];

  let assemblyFormat = "$device `,` $isStorageModeManaged `,` $dims `x` $elementType attr-dict `:` functional-type(operands, results)";
}



//===----------------------------------------------------------------------===//
// Runtime - Buffer
//===----------------------------------------------------------------------===//

def BufferGetContentsOp : Metal_Op<"buffer_get_contents"> {
  let summary = "Buffer get contents operation";

  let arguments = (ins Index:$device);

  let results = (outs AnyMemRef);

  let builders = [
    OpBuilder<(ins "Value":$device, "Type":$elementType)>
  ];

  let hasVerifier = 1;

  let assemblyFormat = "$device attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Runtime - CommandQueue
//===----------------------------------------------------------------------===//

def CommandQueueMakeCommandBufferOp : Metal_Op<"command_queue_make_command_buffer"> {
  let summary = "Command queue make command buffer operation";

  let arguments = (ins Index:$commandQueue, Int_Type:$dimX, Int_Type:$dimY, Int_Type:$dimZ,
                       StrAttr:$functionName);
  let results = (outs Index);

  let builders = [
    OpBuilder<(ins "Value":$commandQueue, "StringRef":$functionName,
                      "Value":$dimX, "Value":$dimY, "Value":$dimZ)>
  ];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Runtime - CommandBuffer
//===----------------------------------------------------------------------===//

def CommandBufferAddBufferOp : Metal_Op<"command_buffer_add_buffer"> {
  let summary = "Command buffer add buffer operation";

  let arguments = (ins Index:$commandBuffer, Index:$bufferRef, I64:$index);

  let assemblyFormat = "$commandBuffer `,`$bufferRef `,` $index "
                        "attr-dict `:` functional-type(operands, results)";
}

def CommandBufferCommitOp : Metal_Op<"command_buffer_commit"> {
  let summary = "Command buffer commit operation";

  let arguments = (ins Index:$commandBuffer);

  let assemblyFormat = "$commandBuffer attr-dict `:` type(operands)";
}

def CommandBufferWaitUntilCompletedOp : Metal_Op<"command_buffer_wait_until_completed"> {
  let summary = "Command buffer wait until completed operation";

  let arguments = (ins Index:$commandBuffer);

  let assemblyFormat = "$commandBuffer attr-dict `:` type(operands)";
}



#endif // METAL_OPS

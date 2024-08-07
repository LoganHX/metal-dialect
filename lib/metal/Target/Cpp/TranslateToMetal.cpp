//===- TranslateToMetal.cpp - Translating to C++ calls
//----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "metal/IR/MetalDialect.h"
#include "metal/IR/MetalOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"

#include "shader/IR/ShaderDialect.h"
#include "shader/IR/ShaderOps.h"

#include "metal/Target/Cpp/MetalEmitter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>
#include <iostream>
#include <stack>
#include <string>
#include <utility>

#define DEBUG_TYPE "translate-to-cpp"

using namespace mlir;
using namespace mlir::emitc;
using llvm::formatv;

/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STLExtras as functions used
/// on each element doesn't return a string.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline LogicalResult
interleaveWithError(ForwardIterator begin, ForwardIterator end,
                    UnaryFunctor eachFn, NullaryFunctor betweenFn) {
  if (begin == end)
    return success();
  if (failed(eachFn(*begin)))
    return failure();
  ++begin;
  for (; begin != end; ++begin) {
    betweenFn();
    if (failed(eachFn(*begin)))
      return failure();
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &c,
                                         UnaryFunctor eachFn,
                                         NullaryFunctor betweenFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container &c,
                                              raw_ostream &os,
                                              UnaryFunctor eachFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
}

/// Return the precedence of a operator as an integer, higher values
/// imply higher precedence.
static FailureOr<int> getOperatorPrecedence(Operation *operation) {
  return llvm::TypeSwitch<Operation *, FailureOr<int>>(operation)
      .Case<emitc::AddOp>([&](auto op) { return 12; })
      .Case<emitc::ApplyOp>([&](auto op) { return 15; })
      .Case<emitc::BitwiseAndOp>([&](auto op) { return 7; })
      .Case<emitc::BitwiseLeftShiftOp>([&](auto op) { return 11; })
      .Case<emitc::BitwiseNotOp>([&](auto op) { return 15; })
      .Case<emitc::BitwiseOrOp>([&](auto op) { return 5; })
      .Case<emitc::BitwiseRightShiftOp>([&](auto op) { return 11; })
      .Case<emitc::BitwiseXorOp>([&](auto op) { return 6; })
      .Case<emitc::CallOp>([&](auto op) { return 16; })
      .Case<emitc::CallOpaqueOp>([&](auto op) { return 16; })
      .Case<emitc::CastOp>([&](auto op) { return 15; })
      .Case<emitc::CmpOp>([&](auto op) -> FailureOr<int> {
        switch (op.getPredicate()) {
        case emitc::CmpPredicate::eq:
        case emitc::CmpPredicate::ne:
          return 8;
        case emitc::CmpPredicate::lt:
        case emitc::CmpPredicate::le:
        case emitc::CmpPredicate::gt:
        case emitc::CmpPredicate::ge:
          return 9;
        case emitc::CmpPredicate::three_way:
          return 10;
        }
        return op->emitError("unsupported cmp predicate");
      })
      .Case<emitc::ConditionalOp>([&](auto op) { return 2; })
      .Case<emitc::DivOp>([&](auto op) { return 13; })
      .Case<emitc::LogicalAndOp>([&](auto op) { return 4; })
      .Case<emitc::LogicalNotOp>([&](auto op) { return 15; })
      .Case<emitc::LogicalOrOp>([&](auto op) { return 3; })
      .Case<emitc::MulOp>([&](auto op) { return 13; })
      .Case<emitc::RemOp>([&](auto op) { return 13; })
      .Case<emitc::SubOp>([&](auto op) { return 12; })
      .Case<emitc::UnaryMinusOp>([&](auto op) { return 15; })
      .Case<emitc::UnaryPlusOp>([&](auto op) { return 15; })
      .Default([](auto op) { return op->emitError("unsupported operation"); });
}

namespace {
/// Emitter that uses dialect specific emitters to emit C++ code.
struct MetalEmitter {

  explicit MetalEmitter(raw_ostream &os, bool declareVariablesAtTop);

  /// Emits attribute or returns failure.
  LogicalResult emitAttribute(Location loc, Attribute attr);

  /// Emits operation 'op' with/without training semicolon or returns failure.
  LogicalResult emitOperation(Operation &op, bool trailingSemicolon);

  /// Emits type 'type' or returns failure.
  LogicalResult emitType(Location loc, Type type, bool isGPUSide = false);

  /// Emits array of types as a std::tuple of the emitted types.
  /// - emits void for an empty array;
  /// - emits the type of the only element for arrays of size one;
  /// - emits a std::tuple otherwise;
  LogicalResult emitTypes(Location loc, ArrayRef<Type> types);

  /// Emits array of types as a std::tuple of the emitted types independently of
  /// the array size.
  LogicalResult emitTupleType(Location loc, ArrayRef<Type> types);

  /// Emits an assignment for a variable which has been declared previously.
  LogicalResult emitVariableAssignment(OpResult result);

  /// Emits a variable declaration for a result of an operation.
  LogicalResult emitVariableDeclaration(OpResult result,
                                        bool trailingSemicolon);

  /// Emits a declaration of a variable with the given type and name.
  LogicalResult emitVariableDeclaration(Location loc, Type type, StringRef name,
                                        bool isGPUSide = false);
  /// Emits a declaration of a variable with the given type and name.
  LogicalResult emitKnownTypeVariableDeclaration(Location loc, StringRef type,
                                                 StringRef name,
                                                 bool trailingSemicolon);

  /// Emits a variable declaration for a result of an operation.
  LogicalResult emitVariableAssignmentAndDeclaration(OpResult result);

  LogicalResult emitKnownTypeVariableAssignmentAndDeclaration(OpResult result,
                                                              StringRef type);

  /// Emits the variable declaration and assignment prefix for 'op'.
  /// - emits separate variable followed by std::tie for multi-valued operation;
  /// - emits single type followed by variable for single result;
  /// - emits nothing if no value produced by op;
  /// Emits final '=' operator where a type is produced. Returns failure if
  /// any result type could not be converted.
  LogicalResult emitAssignPrefix(Operation &op);

  /// Emits a global variable declaration or definition.
  LogicalResult emitGlobalVariable(GlobalOp op);

  /// Emits a label for the block.
  LogicalResult emitLabel(Block &block);

  /// Emits the operands and atttributes of the operation. All operands are
  /// emitted first and then all attributes in alphabetical order.
  LogicalResult emitOperandsAndAttributes(Operation &op,
                                          ArrayRef<StringRef> exclude = {});

  /// Emits the operands of the operation. All operands are emitted in order.
  LogicalResult emitOperands(Operation &op);

  /// Emits value as an operands of an operation
  LogicalResult emitOperand(Value value);

  /// Emit an expression as a C expression.
  LogicalResult emitExpression(ExpressionOp expressionOp);

  LogicalResult emitMemRefSize(Location loc, MemRefType memRefType);

  LogicalResult emitTypeSize(Location loc, Type type);

  LogicalResult emitLinearIndexRefs(Location loc, SmallVector<StringRef> sizes,
                                    SmallVector<Value> indices, bool preload);

  LogicalResult emitLinearIndex(Location loc, SmallVector<Value> sizes,
                                SmallVector<Value> indices);
  /// Return the existing or a new name for a Value.
  StringRef getOrCreateName(Value val);

  // Returns the textual representation of a subscript operation.
  std::string getSubscriptName(emitc::SubscriptOp op);

  /// Return the existing or a new label of a Block.
  StringRef getOrCreateName(Block &block);

  /// Whether to map an mlir integer to a unsigned integer in C++.
  bool shouldMapToUnsigned(IntegerType::SignednessSemantics val);

  /// RAII helper function to manage entering/exiting C++ scopes.
  struct Scope {
    Scope(MetalEmitter &emitter)
        : valueMapperScope(emitter.valueMapper),
          blockMapperScope(emitter.blockMapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
      emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
    }
    ~Scope() {
      emitter.valueInScopeCount.pop();
      emitter.labelInScopeCount.pop();
    }

  private:
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    llvm::ScopedHashTableScope<Block *, std::string> blockMapperScope;
    MetalEmitter &emitter;
  };

  /// Returns wether the Value is assigned to a C++ variable in the scope.
  bool hasValueInScope(Value val);

  // Returns whether a label is assigned to the block.
  bool hasBlockLabel(Block &block);

  /// Returns the output stream.
  raw_indented_ostream &ostream() { return os; };

  /// Returns if all variables for op results and basic block arguments need to
  /// be declared at the beginning of a function.
  bool shouldDeclareVariablesAtTop() { return declareVariablesAtTop; };

  /// Get expression currently being emitted.
  ExpressionOp getEmittedExpression() { return emittedExpression; }

  /// Determine whether given value is part of the expression potentially being
  /// emitted.
  bool isPartOfCurrentExpression(Value value) {
    if (!emittedExpression)
      return false;
    Operation *def = value.getDefiningOp();
    if (!def)
      return false;
    auto operandExpression = dyn_cast<ExpressionOp>(def->getParentOp());
    return operandExpression == emittedExpression;
  };

private:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
  using BlockMapper = llvm::ScopedHashTable<Block *, std::string>;

  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Boolean to enforce that all variables for op results and block
  /// arguments are declared at the beginning of the function. This also
  /// includes results from ops located in nested regions.
  bool declareVariablesAtTop;

  /// Map from value to name of C++ variable that contain the name.
  ValueMapper valueMapper;

  /// Map from block to name of C++ label.
  BlockMapper blockMapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> valueInScopeCount;
  std::stack<int64_t> labelInScopeCount;

  /// State of the current expression being emitted.
  ExpressionOp emittedExpression;
  SmallVector<int> emittedExpressionPrecedence;

  void pushExpressionPrecedence(int precedence) {
    emittedExpressionPrecedence.push_back(precedence);
  }
  void popExpressionPrecedence() { emittedExpressionPrecedence.pop_back(); }
  static int lowestPrecedence() { return 0; }
  int getExpressionPrecedence() {
    if (emittedExpressionPrecedence.empty())
      return lowestPrecedence();
    return emittedExpressionPrecedence.back();
  }
};
} // namespace

/// Determine whether expression \p expressionOp should be emitted inline, i.e.
/// as part of its user. This function recommends inlining of any expressions
/// that can be inlined unless it is used by another expression, under the
/// assumption that  any expression fusion/re-materialization was taken care of
/// by transformations run by the backend.
static bool shouldBeInlined(ExpressionOp expressionOp) {
  // Do not inline if expression is marked as such.
  if (expressionOp.getDoNotInline())
    return false;

  // Do not inline expressions with side effects to prevent side-effect
  // reordering.
  if (expressionOp.hasSideEffects())
    return false;

  // Do not inline expressions with multiple uses.
  Value result = expressionOp.getResult();
  if (!result.hasOneUse())
    return false;

  Operation *user = *result.getUsers().begin();

  // Do not inline expressions used by subscript operations, since the
  // way the subscript operation translation is implemented requires that
  // variables be materialized.
  if (isa<emitc::SubscriptOp>(user))
    return false;

  // Do not inline expressions used by other expressions, as any desired
  // expression folding was taken care of by transformations.
  return !user->getParentOfType<ExpressionOp>();
}

static LogicalResult printConstantOp(MetalEmitter &emitter,
                                     Operation *operation, Attribute value) {
  OpResult result = operation->getResult(0);

  // Only emit an assignment as the variable was already declared when printing
  // the FuncOp.
  if (emitter.shouldDeclareVariablesAtTop()) {
    // Skip the assignment if the emitc.constant has no value.
    if (auto oAttr = dyn_cast<emitc::OpaqueAttr>(value)) {
      if (oAttr.getValue().empty())
        return success();
    }

    if (failed(emitter.emitVariableAssignment(result)))
      return failure();
    return emitter.emitAttribute(operation->getLoc(), value);
  }

  // Emit a variable declaration for an emitc.constant op without value.
  if (auto oAttr = dyn_cast<emitc::OpaqueAttr>(value)) {
    if (oAttr.getValue().empty())
      // The semicolon gets printed by the emitOperation function.
      return emitter.emitVariableDeclaration(result,
                                             /*trailingSemicolon=*/false);
  }

  // Emit a variable declaration.
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  return emitter.emitAttribute(operation->getLoc(), value);
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::ConstantOp constantOp) {
  Operation *operation = constantOp.getOperation();
  Attribute value = constantOp.getValue();

  return printConstantOp(emitter, operation, value);
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::VariableOp variableOp) {
  Operation *operation = variableOp.getOperation();
  Attribute value = variableOp.getValue();

  return printConstantOp(emitter, operation, value);
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::GlobalOp globalOp) {

  return emitter.emitGlobalVariable(globalOp);
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::AssignOp assignOp) {
  OpResult result = assignOp.getVar().getDefiningOp()->getResult(0);

  if (failed(emitter.emitVariableAssignment(result)))
    return failure();

  return emitter.emitOperand(assignOp.getValue());
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::GetGlobalOp op) {
  // Add name to cache so that `hasValueInScope` works.
  emitter.getOrCreateName(op.getResult());
  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::SubscriptOp subscriptOp) {
  // Add name to cache so that `hasValueInScope` works.
  emitter.getOrCreateName(subscriptOp.getResult());
  return success();
}

static LogicalResult printBinaryOperation(MetalEmitter &emitter,
                                          Operation *operation,
                                          StringRef binaryOperator) {
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  if (failed(emitter.emitOperand(operation->getOperand(0))))
    return failure();

  os << " " << binaryOperator << " ";

  if (failed(emitter.emitOperand(operation->getOperand(1))))
    return failure();

  return success();
}

static LogicalResult printUnaryOperation(MetalEmitter &emitter,
                                         Operation *operation,
                                         StringRef unaryOperator) {
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  os << unaryOperator;

  if (failed(emitter.emitOperand(operation->getOperand(0))))
    return failure();

  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter, emitc::AddOp addOp) {
  Operation *operation = addOp.getOperation();

  return printBinaryOperation(emitter, operation, "+");
}

static LogicalResult printOperation(MetalEmitter &emitter, emitc::DivOp divOp) {
  Operation *operation = divOp.getOperation();

  return printBinaryOperation(emitter, operation, "/");
}

static LogicalResult printOperation(MetalEmitter &emitter, emitc::MulOp mulOp) {
  Operation *operation = mulOp.getOperation();

  return printBinaryOperation(emitter, operation, "*");
}

static LogicalResult printOperation(MetalEmitter &emitter, emitc::RemOp remOp) {
  Operation *operation = remOp.getOperation();

  return printBinaryOperation(emitter, operation, "%");
}

static LogicalResult printOperation(MetalEmitter &emitter, emitc::SubOp subOp) {
  Operation *operation = subOp.getOperation();

  return printBinaryOperation(emitter, operation, "-");
}

static LogicalResult printOperation(MetalEmitter &emitter, emitc::CmpOp cmpOp) {
  Operation *operation = cmpOp.getOperation();

  StringRef binaryOperator;

  switch (cmpOp.getPredicate()) {
  case emitc::CmpPredicate::eq:
    binaryOperator = "==";
    break;
  case emitc::CmpPredicate::ne:
    binaryOperator = "!=";
    break;
  case emitc::CmpPredicate::lt:
    binaryOperator = "<";
    break;
  case emitc::CmpPredicate::le:
    binaryOperator = "<=";
    break;
  case emitc::CmpPredicate::gt:
    binaryOperator = ">";
    break;
  case emitc::CmpPredicate::ge:
    binaryOperator = ">=";
    break;
  case emitc::CmpPredicate::three_way:
    binaryOperator = "<=>";
    break;
  }

  return printBinaryOperation(emitter, operation, binaryOperator);
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::ConditionalOp conditionalOp) {
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*conditionalOp)))
    return failure();

  if (failed(emitter.emitOperand(conditionalOp.getCondition())))
    return failure();

  os << " ? ";

  if (failed(emitter.emitOperand(conditionalOp.getTrueValue())))
    return failure();

  os << " : ";

  if (failed(emitter.emitOperand(conditionalOp.getFalseValue())))
    return failure();

  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::VerbatimOp verbatimOp) {
  raw_ostream &os = emitter.ostream();

  os << verbatimOp.getValue();

  return success();
}

// static LogicalResult printOperation(MetalEmitter &emitter,
//                                     cf::BranchOp branchOp) {
//   raw_ostream &os = emitter.ostream();
//   Block &successor = *branchOp.getSuccessor();

//   for (auto pair :
//        llvm::zip(branchOp.getOperands(), successor.getArguments())) {
//     Value &operand = std::get<0>(pair);
//     BlockArgument &argument = std::get<1>(pair);
//     os << emitter.getOrCreateName(argument) << " = "
//        << emitter.getOrCreateName(operand) << ";\n";
//   }

//   os << "goto ";
//   if (!(emitter.hasBlockLabel(successor)))
//     return branchOp.emitOpError("unable to find label for successor block");
//   os << emitter.getOrCreateName(successor);
//   return success();
// }

// static LogicalResult printOperation(MetalEmitter &emitter,
//                                     cf::CondBranchOp condBranchOp) {
//   raw_indented_ostream &os = emitter.ostream();
//   Block &trueSuccessor = *condBranchOp.getTrueDest();
//   Block &falseSuccessor = *condBranchOp.getFalseDest();

//   os << "if (" << emitter.getOrCreateName(condBranchOp.getCondition())
//      << ") {\n";

//   os.indent();

//   // If condition is true.
//   for (auto pair : llvm::zip(condBranchOp.getTrueOperands(),
//                              trueSuccessor.getArguments())) {
//     Value &operand = std::get<0>(pair);
//     BlockArgument &argument = std::get<1>(pair);
//     os << emitter.getOrCreateName(argument) << " = "
//        << emitter.getOrCreateName(operand) << ";\n";
//   }

//   os << "goto ";
//   if (!(emitter.hasBlockLabel(trueSuccessor))) {
//     return condBranchOp.emitOpError("unable to find label for successor
//     block");
//   }
//   os << emitter.getOrCreateName(trueSuccessor) << ";\n";
//   os.unindent() << "} else {\n";
//   os.indent();
//   // If condition is false.
//   for (auto pair : llvm::zip(condBranchOp.getFalseOperands(),
//                              falseSuccessor.getArguments())) {
//     Value &operand = std::get<0>(pair);
//     BlockArgument &argument = std::get<1>(pair);
//     os << emitter.getOrCreateName(argument) << " = "
//        << emitter.getOrCreateName(operand) << ";\n";
//   }

//   os << "goto ";
//   if (!(emitter.hasBlockLabel(falseSuccessor))) {
//     return condBranchOp.emitOpError()
//            << "unable to find label for successor block";
//   }
//   os << emitter.getOrCreateName(falseSuccessor) << ";\n";
//   os.unindent() << "}";
//   return success();
// }

static LogicalResult printCallOperation(MetalEmitter &emitter,
                                        Operation *callOp, StringRef callee) {
  if (failed(emitter.emitAssignPrefix(*callOp)))
    return failure();

  raw_ostream &os = emitter.ostream();
  os << callee << "(";
  if (failed(emitter.emitOperands(*callOp)))
    return failure();
  os << ")";
  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    func::CallOp callOp) {
  Operation *operation = callOp.getOperation();
  StringRef callee = callOp.getCallee();

  return printCallOperation(emitter, operation, callee);
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::CallOp callOp) {
  Operation *operation = callOp.getOperation();
  StringRef callee = callOp.getCallee();

  return printCallOperation(emitter, operation, callee);
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::CallOpaqueOp callOpaqueOp) {
  raw_ostream &os = emitter.ostream();
  Operation &op = *callOpaqueOp.getOperation();

  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  os << callOpaqueOp.getCallee();

  auto emitArgs = [&](Attribute attr) -> LogicalResult {
    if (auto t = dyn_cast<IntegerAttr>(attr)) {
      // Index attributes are treated specially as operand index.
      if (t.getType().isIndex()) {
        int64_t idx = t.getInt();
        Value operand = op.getOperand(idx);
        auto literalDef =
            dyn_cast_if_present<LiteralOp>(operand.getDefiningOp());
        if (!literalDef && !emitter.hasValueInScope(operand))
          return op.emitOpError("operand ")
                 << idx << "'s value not defined in scope";
        os << emitter.getOrCreateName(operand);
        return success();
      }
    }
    if (failed(emitter.emitAttribute(op.getLoc(), attr)))
      return failure();

    return success();
  };

  if (callOpaqueOp.getTemplateArgs()) {
    os << "<";
    if (failed(interleaveCommaWithError(*callOpaqueOp.getTemplateArgs(), os,
                                        emitArgs)))
      return failure();
    os << ">";
  }

  os << "(";

  LogicalResult emittedArgs =
      callOpaqueOp.getArgs()
          ? interleaveCommaWithError(*callOpaqueOp.getArgs(), os, emitArgs)
          : emitter.emitOperands(op);
  if (failed(emittedArgs))
    return failure();
  os << ")";
  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::ApplyOp applyOp) {
  raw_ostream &os = emitter.ostream();
  Operation &op = *applyOp.getOperation();

  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  os << applyOp.getApplicableOperator();
  os << emitter.getOrCreateName(applyOp.getOperand());

  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::BitwiseAndOp bitwiseAndOp) {
  Operation *operation = bitwiseAndOp.getOperation();
  return printBinaryOperation(emitter, operation, "&");
}

static LogicalResult
printOperation(MetalEmitter &emitter,
               emitc::BitwiseLeftShiftOp bitwiseLeftShiftOp) {
  Operation *operation = bitwiseLeftShiftOp.getOperation();
  return printBinaryOperation(emitter, operation, "<<");
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::BitwiseNotOp bitwiseNotOp) {
  Operation *operation = bitwiseNotOp.getOperation();
  return printUnaryOperation(emitter, operation, "~");
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::BitwiseOrOp bitwiseOrOp) {
  Operation *operation = bitwiseOrOp.getOperation();
  return printBinaryOperation(emitter, operation, "|");
}

static LogicalResult
printOperation(MetalEmitter &emitter,
               emitc::BitwiseRightShiftOp bitwiseRightShiftOp) {
  Operation *operation = bitwiseRightShiftOp.getOperation();
  return printBinaryOperation(emitter, operation, ">>");
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::BitwiseXorOp bitwiseXorOp) {
  Operation *operation = bitwiseXorOp.getOperation();
  return printBinaryOperation(emitter, operation, "^");
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::UnaryPlusOp unaryPlusOp) {
  Operation *operation = unaryPlusOp.getOperation();
  return printUnaryOperation(emitter, operation, "+");
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::UnaryMinusOp unaryMinusOp) {
  Operation *operation = unaryMinusOp.getOperation();
  return printUnaryOperation(emitter, operation, "-");
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::CastOp castOp) {
  raw_ostream &os = emitter.ostream();
  Operation &op = *castOp.getOperation();

  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  os << "(";
  if (failed(emitter.emitType(op.getLoc(), op.getResult(0).getType())))
    return failure();
  os << ") ";
  return emitter.emitOperand(castOp.getOperand());
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::ExpressionOp expressionOp) {
  if (shouldBeInlined(expressionOp))
    return success();

  Operation &op = *expressionOp.getOperation();

  if (failed(emitter.emitAssignPrefix(op)))
    return failure();

  return emitter.emitExpression(expressionOp);
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::IncludeOp includeOp) {
  raw_ostream &os = emitter.ostream();

  os << "#include ";
  if (includeOp.getIsStandardInclude())
    os << "<" << includeOp.getInclude() << ">";
  else
    os << "\"" << includeOp.getInclude() << "\"";

  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::LogicalAndOp logicalAndOp) {
  Operation *operation = logicalAndOp.getOperation();
  return printBinaryOperation(emitter, operation, "&&");
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::LogicalNotOp logicalNotOp) {
  Operation *operation = logicalNotOp.getOperation();
  return printUnaryOperation(emitter, operation, "!");
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::LogicalOrOp logicalOrOp) {
  Operation *operation = logicalOrOp.getOperation();
  return printBinaryOperation(emitter, operation, "||");
}

static LogicalResult printOperation(MetalEmitter &emitter, emitc::ForOp forOp) {

  raw_indented_ostream &os = emitter.ostream();

  // Utility function to determine whether a value is an expression that will be
  // inlined, and as such should be wrapped in parentheses in order to guarantee
  // its precedence and associativity.
  auto requiresParentheses = [&](Value value) {
    auto expressionOp =
        dyn_cast_if_present<ExpressionOp>(value.getDefiningOp());
    if (!expressionOp)
      return false;
    return shouldBeInlined(expressionOp);
  };

  os << "for (";
  if (failed(
          emitter.emitType(forOp.getLoc(), forOp.getInductionVar().getType())))
    return failure();
  os << " ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " = ";
  if (failed(emitter.emitOperand(forOp.getLowerBound())))
    return failure();
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " < ";
  Value upperBound = forOp.getUpperBound();
  bool upperBoundRequiresParentheses = requiresParentheses(upperBound);
  if (upperBoundRequiresParentheses)
    os << "(";
  if (failed(emitter.emitOperand(upperBound)))
    return failure();
  if (upperBoundRequiresParentheses)
    os << ")";
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " += ";
  if (failed(emitter.emitOperand(forOp.getStep())))
    return failure();
  os << ") {\n";
  os.indent();

  Region &forRegion = forOp.getRegion();
  auto regionOps = forRegion.getOps();

  // We skip the trailing yield op.
  for (auto it = regionOps.begin(); std::next(it) != regionOps.end(); ++it) {
    if (failed(emitter.emitOperation(*it, /*trailingSemicolon=*/true)))
      return failure();
  }

  os.unindent() << "}";

  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter, emitc::IfOp ifOp) {
  raw_indented_ostream &os = emitter.ostream();
  // Helper function to emit all ops except the last one, expected to be
  // emitc::yield.
  auto emitAllExceptLast = [&emitter](Region &region) {
    Region::OpIterator it = region.op_begin(), end = region.op_end();
    for (; std::next(it) != end; ++it) {
      if (failed(emitter.emitOperation(*it, /*trailingSemicolon=*/true)))
        return failure();
    }
    assert(isa<emitc::YieldOp>(*it) &&
           "Expected last operation in the region to be emitc::yield");
    return success();
  };

  os << "if (";
  if (failed(emitter.emitOperand(ifOp.getCondition())))
    return failure();
  os << ") {\n";
  os.indent();
  if (failed(emitAllExceptLast(ifOp.getThenRegion())))
    return failure();
  os.unindent() << "}";

  Region &elseRegion = ifOp.getElseRegion();
  if (!elseRegion.empty()) {
    os << " else {\n";
    os.indent();
    if (failed(emitAllExceptLast(elseRegion)))
      return failure();
    os.unindent() << "}";
  }

  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    func::ReturnOp returnOp) {
  raw_ostream &os = emitter.ostream();
  os << "return";
  switch (returnOp.getNumOperands()) {
  case 0:
    return success();
  case 1:
    os << " ";
    if (failed(emitter.emitOperand(returnOp.getOperand(0))))
      return failure();
    return success();
  default:
    os << " std::make_tuple(";
    if (failed(emitter.emitOperandsAndAttributes(*returnOp.getOperation())))
      return failure();
    os << ")";
    return success();
  }
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::ReturnOp returnOp) {
  raw_ostream &os = emitter.ostream();
  os << "return";
  if (returnOp.getNumOperands() == 0)
    return success();

  os << " ";
  if (failed(emitter.emitOperand(returnOp.getOperand())))
    return failure();
  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter, ModuleOp moduleOp) {
  MetalEmitter::Scope scope(emitter);

  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false)))
      return failure();
  }
  return success();
}

static LogicalResult printFunctionArgs(MetalEmitter &emitter,
                                       Operation *functionOp,
                                       ArrayRef<Type> arguments) {
  raw_indented_ostream &os = emitter.ostream();

  return (
      interleaveCommaWithError(arguments, os, [&](Type arg) -> LogicalResult {
        return emitter.emitType(functionOp->getLoc(), arg);
      }));
}

static LogicalResult printFunctionArgs(MetalEmitter &emitter,
                                       Operation *functionOp,
                                       Region::BlockArgListType arguments,
                                       bool isGPUSide = false) {
  raw_indented_ostream &os = emitter.ostream();
  return (interleaveCommaWithError(
      arguments, os, [&](BlockArgument arg) -> LogicalResult {
        return emitter.emitVariableDeclaration(
            functionOp->getLoc(), arg.getType(), emitter.getOrCreateName(arg),
            isGPUSide);
      }));
}

static LogicalResult printFunctionBody(MetalEmitter &emitter,
                                       Operation *functionOp,
                                       Region::BlockListType &blocks) {
  raw_indented_ostream &os = emitter.ostream();
  os.indent();

  if (emitter.shouldDeclareVariablesAtTop()) {
    // Declare all variables that hold op results including those from nested
    // regions.
    WalkResult result =
        functionOp->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
          if (isa<emitc::LiteralOp>(op) ||
              isa<emitc::ExpressionOp>(op->getParentOp()) ||
              (isa<emitc::ExpressionOp>(op) &&
               shouldBeInlined(cast<emitc::ExpressionOp>(op))))
            return WalkResult::skip();
          for (OpResult result : op->getResults()) {
            if (failed(emitter.emitVariableDeclaration(
                    result, /*trailingSemicolon=*/true))) {
              return WalkResult(
                  op->emitError("unable to declare result variable for op"));
            }
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted())
      return failure();
  }

  // Create label names for basic blocks.
  for (Block &block : blocks) {
    emitter.getOrCreateName(block);
  }

  // Declare variables for basic block arguments.
  for (Block &block : llvm::drop_begin(blocks)) {
    for (BlockArgument &arg : block.getArguments()) {
      if (emitter.hasValueInScope(arg))
        return functionOp->emitOpError(" block argument #")
               << arg.getArgNumber() << " is out of scope";
      if (isa<ArrayType>(arg.getType()))
        return functionOp->emitOpError("cannot emit block argument #")
               << arg.getArgNumber() << " with array type";
      if (failed(
              emitter.emitType(block.getParentOp()->getLoc(), arg.getType()))) {
        return failure();
      }
      os << " " << emitter.getOrCreateName(arg) << ";\n";
    }
  }

  for (Block &block : blocks) {
    // Only print a label if the block has predecessors.
    if (!block.hasNoPredecessors()) {
      if (failed(emitter.emitLabel(block)))
        return failure();
    }
    for (Operation &op : block.getOperations()) {
      // When generating code for an emitc.if or cf.cond_br op no semicolon
      // needs to be printed after the closing brace.
      // When generating code for an emitc.for and emitc.verbatim op, printing a
      // trailing semicolon is handled within the printOperation function.
      bool trailingSemicolon =
          !isa<cf::CondBranchOp, emitc::DeclareFuncOp, emitc::ForOp,
               emitc::IfOp, emitc::LiteralOp, emitc::VerbatimOp, gpu::GPUFuncOp,
               gpu::ModuleEndOp, gpu::ThreadIdOp, gpu::BlockDimOp>(op);

      if (failed(emitter.emitOperation(
              op, /*trailingSemicolon=*/trailingSemicolon)))
        return failure();
    }
  }

  os.unindent();

  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    func::FuncOp functionOp) {
  // We need to declare variables at top if the function has multiple blocks.
  if (!emitter.shouldDeclareVariablesAtTop() &&
      functionOp.getBlocks().size() > 1) {
    return functionOp.emitOpError(
        "with multiple blocks needs variables declared at top");
  }

  if (llvm::any_of(functionOp.getResultTypes(), llvm::IsaPred<ArrayType>)) {
    return functionOp.emitOpError() << "cannot emit array type as result type";
  }

  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  if (failed(emitter.emitTypes(functionOp.getLoc(),
                               functionOp.getFunctionType().getResults())))
    return failure();
  os << " " << functionOp.getName();

  os << "(";
  Operation *operation = functionOp.getOperation();
  if (failed(printFunctionArgs(emitter, operation, functionOp.getArguments())))
    return failure();
  os << ") {\n";
  if (failed(printFunctionBody(emitter, operation, functionOp.getBlocks())))
    return failure();
  os << "}\n";

  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    memref::DeallocOp deallocOp) {

  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  os << "free(";
  os << emitter.getOrCreateName(
      deallocOp.getOperand().getDefiningOp()->getOpResult(0));
  os << ")";
  return success();
}

static LogicalResult getMemRefSize(MemRefType memrefType,
                                   SmallVector<std::string> &dimensions) {

  if (memrefType.getShape().size() > 3)
    return failure();
  if (memrefType.getShape().size() < 1)
    return failure();

  dimensions.resize(memrefType.getShape().size());

  if (memrefType.getShape().size() == 3) {
    dimensions[2] = std::to_string(memrefType.getDimSize(2));
  }
  if (memrefType.getShape().size() >= 2) {
    dimensions[1] = std::to_string(memrefType.getDimSize(1));
  }
  if (memrefType.getShape().size() >= 1) {
    dimensions[0] = std::to_string(memrefType.getDimSize(0));
  }

  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    memref::StoreOp storeOp) {

  MemRefType memrefType = cast<MemRefType>(storeOp.getMemref().getType());
  SmallVector<std::string> dimensions;

  if (failed(getMemRefSize(memrefType, dimensions)))
    return failure();

  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();

  os << emitter.getOrCreateName(storeOp.getOperand(1));
  os << "[";

  // Converti std::string a StringRef per emitLinearIndexRefs
  SmallVector<StringRef, 3> stringRefs;
  for (const auto &dim : dimensions) {
    stringRefs.push_back(StringRef(dim));
  }

  if (failed(emitter.emitLinearIndexRefs(storeOp.getLoc(), stringRefs,
                                         storeOp.getIndices(), true)))
    return failure();

  os << "]";
  os << " = ";
  os << emitter.getOrCreateName(storeOp.getOperand(0));

  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    memref::LoadOp loadOp) {

  MemRefType memrefType = cast<MemRefType>(loadOp.getMemref().getType());
  SmallVector<std::string> dimensions;

  dimensions.resize(memrefType.getShape().size());

  if (failed(getMemRefSize(memrefType, dimensions)))
    return failure();

  OpResult result = loadOp->getResult(0);
  if (failed(emitter.emitVariableAssignmentAndDeclaration(result)))
    return failure();

  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();

  os << emitter.getOrCreateName(loadOp.getOperand(0));
  os << "[";

  SmallVector<StringRef, 3> stringRefs;
  for (const auto &dim : dimensions) {
    stringRefs.push_back(StringRef(dim));
  }
  if (failed(emitter.emitLinearIndexRefs(loadOp.getLoc(), stringRefs,
                                         loadOp.getIndices(), true)))
    return failure();
  os << "]";
  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    memref::AllocOp allocOp) {

  MemRefType memrefType = cast<MemRefType>(allocOp.getMemref().getType());
  SmallVector<std::string> dimensions;

  dimensions.resize(memrefType.getShape().size());

  if (failed(getMemRefSize(memrefType, dimensions)))
    return failure();
  OpResult result = allocOp->getResult(0);
  if (failed(emitter.emitVariableAssignmentAndDeclaration(result)))
    return failure();
  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();

  os << "malloc(";
  for (size_t i = 0; i < dimensions.size(); i++) {
    os << dimensions[i];
    os << " * ";
  }

  os << "sizeof(";
  if (failed(emitter.emitType(allocOp.getLoc(), memrefType.getElementType())))
    return failure();
  os << ")";
  os << ")";
  return success();
}

static LogicalResult printMPS(MetalEmitter &emitter, OpResult result,
                              Value queue, Value bufferA, Value rowsA,
                              Value columnsA, Value bufferB, Value rowsB,
                              Value columnsB, Value bufferC, Type elementType,
                              const std::string &operationName) {
  if (failed(emitter.emitKnownTypeVariableAssignmentAndDeclaration(result,
                                                                   "intptr_t")))
    return failure();

  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();

  os << operationName << "(";
  os << emitter.getOrCreateName(queue);
  os << ", ";
  os << emitter.getOrCreateName(bufferA);
  os << ", ";
  os << emitter.getOrCreateName(rowsA);
  os << ", ";
  os << emitter.getOrCreateName(columnsA);
  os << ", ";
  os << emitter.getOrCreateName(bufferB);
  os << ", ";
  os << emitter.getOrCreateName(rowsB);
  os << ", ";
  os << emitter.getOrCreateName(columnsB);
  os << ", ";
  os << emitter.getOrCreateName(bufferC);
  os << ", ";
  os << "\"";
  if (failed(emitter.emitType(bufferC.getLoc(), elementType)))
    return failure();
  os << "\"";
  os << ")";

  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    shader::MatmulOp op) {
  if (failed(printMPS(emitter, op->getResult(0), op.getQueue(), op.getBufferA(),
                      op.getRowsA(), op.getColumnsA(), op.getBufferB(),
                      op.getRowsB(), op.getColumnsB(), op.getBufferC(),
                      op.getElementType(), "_MetalMatMul")))
    return failure();
  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    shader::MatsumOp op) {
  if (failed(printMPS(emitter, op->getResult(0), op.getQueue(), op.getBufferA(),
                      op.getRowsA(), op.getColumnsA(), op.getBufferB(),
                      op.getRowsB(), op.getColumnsB(), op.getBufferC(),
                      op.getElementType(), "_MetalMatSum")))
    return failure();
  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    metal::DeviceMakeDefaultOp op) {

  if (failed(emitter.emitKnownTypeVariableAssignmentAndDeclaration(
          op->getResult(0), "intptr_t")))
    return failure();

  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  os << "_MetalDeviceMakeDefault()";
  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    metal::DeviceMakeCommandQueueOp op) {
  if (failed(emitter.emitKnownTypeVariableAssignmentAndDeclaration(
          op->getResult(0), "intptr_t")))
    return failure();

  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  os << "_MetalDeviceMakeCommandQueue(";
  os << emitter.getOrCreateName(op.getDevice());
  os << ")";
  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    metal::CommandQueueMakeCommandBufferOp op) {
  if (failed(emitter.emitKnownTypeVariableAssignmentAndDeclaration(
          op->getResult(0), "intptr_t")))
    return failure();

  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  os << "_MetalCommandQueueMakeCommandBufferWithDefaultLibrary(";
  os << emitter.getOrCreateName(op.getCommandQueue());
  os << ", ";
  os << emitter.getOrCreateName(op.getDimX());
  os << ", ";
  os << emitter.getOrCreateName(op.getDimY());
  os << ", ";
  os << emitter.getOrCreateName(op.getDimZ());
  os << ", (int8_t *)\"";
  os << op.getFunctionName();
  os << "\")";
  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    metal::DeviceMakeBufferOp op) {

  if (failed(emitter.emitKnownTypeVariableAssignmentAndDeclaration(
          op->getResult(0), "intptr_t")))
    return failure();

  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();

  os << "_MetalDeviceMakeBuffer(";
  os << emitter.getOrCreateName(op.getDevice());
  os << ", ";
  os << emitter.getOrCreateName(op.getIsStorageModeManaged());
  os << ", ";

  for (int i = 0; i < (int)op.getDims().size(); i++) {

    mlir::Value dim = op.getDims()[i];
    os << emitter.getOrCreateName(dim);
    if (i != (int)op.getDims().size() - 1)
      os << " * ";
  }

  os << ", sizeof(";
  if (failed(emitter.emitType(op.getLoc(), op.getElementType())))
    return failure();
  os << ")";
  os << ")";
  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    metal::CommandBufferCommitOp op) {

  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  os << "_MetalCommandBufferCommit(";
  os << emitter.getOrCreateName(op.getCommandBuffer());
  os << ")";
  return success();
}

static LogicalResult
printOperation(MetalEmitter &emitter,
               metal::CommandBufferWaitUntilCompletedOp op) {

  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  os << "_MetalCommandBufferWaitUntilCompleted(";
  os << emitter.getOrCreateName(op.getCommandBuffer());
  os << ")";
  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    metal::ReleaseOp op) {

  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  os << "_MetalRelease(";
  os << emitter.getOrCreateName(op.getRef());
  os << ")";
  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    metal::CommandBufferAddBufferOp op) {

  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  os << "_MetalCommandBufferAddBuffer(";
  os << emitter.getOrCreateName(op.getCommandBuffer());
  os << ", ";
  os << emitter.getOrCreateName(op.getBufferRef());
  os << ", ";
  os << emitter.getOrCreateName(op.getIndex());
  os << ")";

  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter, metal::StoreOp op) {
  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  os << "_MetalStore_";
  if (failed(emitter.emitType(op.getLoc(), op.getValue().getType())))
    return failure();
  os << "(";
  os << emitter.getOrCreateName(op.getBuffer());
  os << ", ";

  if (failed(
          emitter.emitLinearIndex(op.getLoc(), op.getSizes(), op.getIndexes())))
    return failure();

  os << ", ";
  os << emitter.getOrCreateName(op.getValue());

  os << ")";
  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    metal::GetElementOp op) {

  if (failed(emitter.emitVariableAssignmentAndDeclaration(op->getResult(0))))
    return failure();

  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  os << "_MetalLoad_";
  if (failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return failure();
  os << "(";
  os << emitter.getOrCreateName(op.getBuffer());
  os << ", ";

  if (failed(
          emitter.emitLinearIndex(op.getLoc(), op.getSizes(), op.getIndexes())))
    return failure();

  os << ")";
  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    gpu::GPUModuleOp moduleOp) {

  Operation *operation = moduleOp.getOperation();

  if (failed(printFunctionBody(emitter, operation,
                               moduleOp.getBodyRegion().getBlocks())))
    return failure();

  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    gpu::ModuleEndOp moduleOp) {
  return success();
}

// static LogicalResult printOperation(MetalEmitter &emitter,
//                                     gpu::LaunchFuncOp functionOp) {

//   MetalEmitter::Scope scope(emitter);
//   raw_indented_ostream &os = emitter.ostream();
//   if(failed(emitter.emitType(functionOp.getLoc(),
//   functionOp.getGridSizeX().getType()))) return failure(); os <<
//   "_MetalCommandBufferCommit(_"
//         "MetalCommandQueueMakeCommandBufferWithDefaultLibrary(queue,"
//      << functionOp.getKernelModuleName();
//   os << ",";
//   if(failed(emitter.emitOperand(functionOp.getGridSizeX())))
//   return failure();
//   os << ",";
//   if(failed(emitter.emitOperand(functionOp.getGridSizeY())))
//   return failure();
//   os << ",";
//   if(failed(emitter.emitOperand(functionOp.getGridSizeZ())))
//     return failure();
//   os << "))";

//   return success();
// }

static LogicalResult printKernelSizeVariables(MetalEmitter &emitter) {
  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();

  os << ", "
     << "uint3 id [[thread_position_in_grid]], uint3 gridDim "
        "[[threads_per_grid]]";

  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    gpu::GPUFuncOp functionOp) {

  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  os << "kernel ";
  if (failed(emitter.emitTypes(functionOp.getLoc(),
                               functionOp.getFunctionType().getResults())))
    return failure();
  os << " " << functionOp.getParentOp().getName();

  os << "(";
  Operation *operation = functionOp.getOperation();
  if (failed(printFunctionArgs(emitter, operation, functionOp.getArguments(),
                               true)))
    return failure();
  if (failed(printKernelSizeVariables(emitter)))
    return failure();
  os << ") {\n";
  if (failed(printFunctionBody(emitter, operation, functionOp.getBlocks())))
    return failure();
  os << "}\n";

  return success();
}

static LogicalResult printAccessGPUIDDimensionOp(MetalEmitter &emitter,
                                                 OpResult result,
                                                 gpu::Dimension dim,
                                                 StringRef str) {

  if (failed(emitter.emitVariableAssignmentAndDeclaration(result)))
    return failure();

  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  os << str << "." << dim;

  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter, gpu::ThreadIdOp op) {
  // return printAccessGPUIDDimensionOp(emitter, op->getResult(0),
  //                                    op.getDimension(), "id");
  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter, gpu::BlockIdOp op) {
  return printAccessGPUIDDimensionOp(emitter, op->getResult(0),
                                     op.getDimension(), "id");
}

static LogicalResult printOperation(MetalEmitter &emitter, gpu::BlockDimOp op) {
  // return printAccessGPUIDDimensionOp(emitter, op->getResult(0),
  //                                    op.getDimension(), "blockDim");
  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter, gpu::GridDimOp op) {
  return printAccessGPUIDDimensionOp(emitter, op->getResult(0),
                                     op.getDimension(), "gridDim");
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    gpu::ReturnOp gridDimOp) {
  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  // i kernel meta sono tutti void
  os << "return";

  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    emitc::FuncOp functionOp) {
  // We need to declare variables at top if the function has multiple blocks.
  if (!emitter.shouldDeclareVariablesAtTop() &&
      functionOp.getBlocks().size() > 1) {
    return functionOp.emitOpError(
        "with multiple blocks needs variables declared at top");
  }

  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  if (functionOp.getSpecifiers()) {
    for (Attribute specifier : functionOp.getSpecifiersAttr()) {
      os << cast<StringAttr>(specifier).str() << " ";
    }
  }

  if (failed(emitter.emitTypes(functionOp.getLoc(),
                               functionOp.getFunctionType().getResults())))
    return failure();
  os << " " << functionOp.getName();

  os << "(";
  Operation *operation = functionOp.getOperation();
  if (functionOp.isExternal()) {
    if (failed(printFunctionArgs(emitter, operation,
                                 functionOp.getArgumentTypes())))
      return failure();
    os << ");";
    return success();
  }
  if (failed(printFunctionArgs(emitter, operation, functionOp.getArguments())))
    return failure();
  os << ") {\n";
  if (failed(printFunctionBody(emitter, operation, functionOp.getBlocks())))
    return failure();
  os << "}\n";

  return success();
}

static LogicalResult printOperation(MetalEmitter &emitter,
                                    DeclareFuncOp declareFuncOp) {
  MetalEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();

  auto functionOp = SymbolTable::lookupNearestSymbolFrom<emitc::FuncOp>(
      declareFuncOp, declareFuncOp.getSymNameAttr());

  if (!functionOp)
    return failure();

  if (functionOp.getSpecifiers()) {
    for (Attribute specifier : functionOp.getSpecifiersAttr()) {
      os << cast<StringAttr>(specifier).str() << " ";
    }
  }

  if (failed(emitter.emitTypes(functionOp.getLoc(),
                               functionOp.getFunctionType().getResults())))
    return failure();
  os << " " << functionOp.getName();

  os << "(";
  Operation *operation = functionOp.getOperation();
  if (failed(printFunctionArgs(emitter, operation, functionOp.getArguments())))
    return failure();
  os << ");";

  return success();
}

MetalEmitter::MetalEmitter(raw_ostream &os, bool declareVariablesAtTop)
    : os(os), declareVariablesAtTop(declareVariablesAtTop) {
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
}

std::string MetalEmitter::getSubscriptName(emitc::SubscriptOp op) {
  std::string out;
  llvm::raw_string_ostream ss(out);
  ss << getOrCreateName(op.getValue());
  for (auto index : op.getIndices()) {
    ss << "[" << getOrCreateName(index) << "]";
  }
  return out;
}

/// Return the existing or a new name for a Value.
StringRef MetalEmitter::getOrCreateName(Value val) {
  if (auto literal = dyn_cast_if_present<emitc::LiteralOp>(val.getDefiningOp()))
    return literal.getValue();
  if (!valueMapper.count(val)) {
    if (auto subscript =
            dyn_cast_if_present<emitc::SubscriptOp>(val.getDefiningOp())) {
      valueMapper.insert(val, getSubscriptName(subscript));
    } else if (auto getGlobal = dyn_cast_if_present<emitc::GetGlobalOp>(
                   val.getDefiningOp())) {
      valueMapper.insert(val, getGlobal.getName().str());
    } else {
      valueMapper.insert(val, formatv("v{0}", ++valueInScopeCount.top()));
    }
  }
  return *valueMapper.begin(val);
}

/// Return the existing or a new label for a Block.
StringRef MetalEmitter::getOrCreateName(Block &block) {
  if (!blockMapper.count(&block))
    blockMapper.insert(&block, formatv("label{0}", ++labelInScopeCount.top()));
  return *blockMapper.begin(&block);
}

bool MetalEmitter::shouldMapToUnsigned(IntegerType::SignednessSemantics val) {
  switch (val) {
  case IntegerType::Signless:
    return false;
  case IntegerType::Signed:
    return false;
  case IntegerType::Unsigned:
    return true;
  }
  llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
}

bool MetalEmitter::hasValueInScope(Value val) { return valueMapper.count(val); }

bool MetalEmitter::hasBlockLabel(Block &block) {
  return blockMapper.count(&block);
}

LogicalResult MetalEmitter::emitAttribute(Location loc, Attribute attr) {
  auto printInt = [&](const APInt &val, bool isUnsigned) {
    if (val.getBitWidth() == 1) {
      if (val.getBoolValue())
        os << "true";
      else
        os << "false";
    } else {
      SmallString<128> strValue;
      val.toString(strValue, 10, !isUnsigned, false);
      os << strValue;
    }
  };

  auto printFloat = [&](const APFloat &val) {
    if (val.isFinite()) {
      SmallString<128> strValue;
      // Use default values of toString except don't truncate zeros.
      val.toString(strValue, 0, 0, false);
      os << strValue;
      switch (llvm::APFloatBase::SemanticsToEnum(val.getSemantics())) {
      case llvm::APFloatBase::S_IEEEsingle:
        os << "f";
        break;
      case llvm::APFloatBase::S_IEEEdouble:
        break;
      default:
        llvm_unreachable("unsupported floating point type");
      };
    } else if (val.isNaN()) {
      os << "NAN";
    } else if (val.isInfinity()) {
      if (val.isNegative())
        os << "-";
      os << "INFINITY";
    }
  };

  // Print floating point attributes.
  if (auto fAttr = dyn_cast<FloatAttr>(attr)) {
    if (!isa<Float32Type, Float64Type>(fAttr.getType())) {
      return emitError(loc,
                       "expected floating point attribute to be f32 or f64");
    }
    printFloat(fAttr.getValue());
    return success();
  }
  if (auto dense = dyn_cast<DenseFPElementsAttr>(attr)) {
    if (!isa<Float32Type, Float64Type>(dense.getElementType())) {
      return emitError(loc,
                       "expected floating point attribute to be f32 or f64");
    }
    os << '{';
    interleaveComma(dense, os, [&](const APFloat &val) { printFloat(val); });
    os << '}';
    return success();
  }

  // Print integer attributes.
  if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(iAttr.getType())) {
      printInt(iAttr.getValue(), shouldMapToUnsigned(iType.getSignedness()));
      return success();
    }
    if (auto iType = dyn_cast<IndexType>(iAttr.getType())) {
      printInt(iAttr.getValue(), false);
      return success();
    }
  }
  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(
            cast<TensorType>(dense.getType()).getElementType())) {
      os << '{';
      interleaveComma(dense, os, [&](const APInt &val) {
        printInt(val, shouldMapToUnsigned(iType.getSignedness()));
      });
      os << '}';
      return success();
    }
    if (auto iType = dyn_cast<IndexType>(
            cast<TensorType>(dense.getType()).getElementType())) {
      os << '{';
      interleaveComma(dense, os,
                      [&](const APInt &val) { printInt(val, false); });
      os << '}';
      return success();
    }
  }

  // Print opaque attributes.
  if (auto oAttr = dyn_cast<emitc::OpaqueAttr>(attr)) {
    os << oAttr.getValue();
    return success();
  }

  // Print symbolic reference attributes.
  if (auto sAttr = dyn_cast<SymbolRefAttr>(attr)) {
    if (sAttr.getNestedReferences().size() > 1)
      return emitError(loc, "attribute has more than 1 nested reference");
    os << sAttr.getRootReference().getValue();
    return success();
  }

  // Print type attributes.
  if (auto type = dyn_cast<TypeAttr>(attr))
    return emitType(loc, type.getValue());

  return emitError(loc, "cannot emit attribute: ") << attr;
}

LogicalResult MetalEmitter::emitExpression(ExpressionOp expressionOp) {
  assert(emittedExpressionPrecedence.empty() &&
         "Expected precedence stack to be empty");
  Operation *rootOp = expressionOp.getRootOp();

  emittedExpression = expressionOp;
  FailureOr<int> precedence = getOperatorPrecedence(rootOp);
  if (failed(precedence))
    return failure();
  pushExpressionPrecedence(precedence.value());

  if (failed(emitOperation(*rootOp, /*trailingSemicolon=*/false)))
    return failure();

  popExpressionPrecedence();
  assert(emittedExpressionPrecedence.empty() &&
         "Expected precedence stack to be empty");
  emittedExpression = nullptr;

  return success();
}

LogicalResult MetalEmitter::emitOperand(Value value) {
  if (isPartOfCurrentExpression(value)) {
    Operation *def = value.getDefiningOp();
    assert(def && "Expected operand to be defined by an operation");
    FailureOr<int> precedence = getOperatorPrecedence(def);
    if (failed(precedence))
      return failure();
    bool encloseInParenthesis = precedence.value() < getExpressionPrecedence();
    if (encloseInParenthesis) {
      os << "(";
      pushExpressionPrecedence(lowestPrecedence());
    } else
      pushExpressionPrecedence(precedence.value());

    if (failed(emitOperation(*def, /*trailingSemicolon=*/false)))
      return failure();

    if (encloseInParenthesis)
      os << ")";

    popExpressionPrecedence();
    return success();
  }

  auto expressionOp = dyn_cast_if_present<ExpressionOp>(value.getDefiningOp());
  if (expressionOp && shouldBeInlined(expressionOp))
    return emitExpression(expressionOp);

  auto literalOp = dyn_cast_if_present<LiteralOp>(value.getDefiningOp());
  if (!literalOp && !hasValueInScope(value))
    return failure();
  os << getOrCreateName(value);
  return success();
}

LogicalResult MetalEmitter::emitOperands(Operation &op) {
  return interleaveCommaWithError(op.getOperands(), os, [&](Value operand) {
    // If an expression is being emitted, push lowest precedence as these
    // operands are either wrapped by parenthesis.
    if (getEmittedExpression())
      pushExpressionPrecedence(lowestPrecedence());
    if (failed(emitOperand(operand)))
      return failure();
    if (getEmittedExpression())
      popExpressionPrecedence();
    return success();
  });
}

LogicalResult
MetalEmitter::emitOperandsAndAttributes(Operation &op,
                                        ArrayRef<StringRef> exclude) {
  if (failed(emitOperands(op)))
    return failure();
  // Insert comma in between operands and non-filtered attributes if needed.
  if (op.getNumOperands() > 0) {
    for (NamedAttribute attr : op.getAttrs()) {
      if (!llvm::is_contained(exclude, attr.getName().strref())) {
        os << ", ";
        break;
      }
    }
  }
  // Emit attributes.
  auto emitNamedAttribute = [&](NamedAttribute attr) -> LogicalResult {
    if (llvm::is_contained(exclude, attr.getName().strref()))
      return success();
    os << "/* " << attr.getName().getValue() << " */";
    if (failed(emitAttribute(op.getLoc(), attr.getValue())))
      return failure();
    return success();
  };
  return interleaveCommaWithError(op.getAttrs(), os, emitNamedAttribute);
}

LogicalResult MetalEmitter::emitVariableAssignment(OpResult result) {

  if (!hasValueInScope(result)) {
    return result.getDefiningOp()->emitOpError(
        "result variable for the operation has not been declared");
  }
  os << getOrCreateName(result) << " = ";
  return success();
}

LogicalResult MetalEmitter::emitVariableDeclaration(OpResult result,
                                                    bool trailingSemicolon) {
  if (isa<emitc::SubscriptOp>(result.getDefiningOp()))
    return success();
  if (hasValueInScope(result)) {
    return result.getDefiningOp()->emitError(
        "result variable for the operation already declared");
  }
  if (failed(emitVariableDeclaration(result.getOwner()->getLoc(),
                                     result.getType(),
                                     getOrCreateName(result))))
    return failure();
  if (trailingSemicolon)
    os << ";\n";
  return success();
}

LogicalResult
MetalEmitter::emitVariableAssignmentAndDeclaration(OpResult result) {
  if (failed(emitVariableDeclaration(result, true)))
    return failure();
  if (failed(emitVariableAssignment(result)))
    return failure();

  return success();
}

LogicalResult
MetalEmitter::emitKnownTypeVariableAssignmentAndDeclaration(OpResult result,
                                                            StringRef type) {
  if (failed(emitKnownTypeVariableDeclaration(result.getOwner()->getLoc(), type,
                                              getOrCreateName(result), true)))
    return failure();
  if (failed(emitVariableAssignment(result)))
    return failure();

  return success();
}

LogicalResult MetalEmitter::emitGlobalVariable(GlobalOp op) {
  if (op.getExternSpecifier())
    os << "extern ";
  else if (op.getStaticSpecifier())
    os << "static ";
  if (op.getConstSpecifier())
    os << "const ";

  if (failed(emitVariableDeclaration(op->getLoc(), op.getType(),
                                     op.getSymName()))) {
    return failure();
  }

  std::optional<Attribute> initialValue = op.getInitialValue();
  if (initialValue) {
    os << " = ";
    if (failed(emitAttribute(op->getLoc(), *initialValue)))
      return failure();
  }

  os << ";";
  return success();
}

LogicalResult MetalEmitter::emitAssignPrefix(Operation &op) {
  // If op is being emitted as part of an expression, bail out.
  if (getEmittedExpression())
    return success();

  switch (op.getNumResults()) {
  case 0:
    break;
  case 1: {
    OpResult result = op.getResult(0);
    if (shouldDeclareVariablesAtTop()) {
      if (failed(emitVariableAssignment(result)))
        return failure();
    } else {
      if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/false)))
        return failure();
      os << " = ";
    }
    break;
  }
  default:
    if (!shouldDeclareVariablesAtTop()) {
      for (OpResult result : op.getResults()) {
        if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/true)))
          return failure();
      }
    }
    os << "std::tie(";
    interleaveComma(op.getResults(), os,
                    [&](Value result) { os << getOrCreateName(result); });
    os << ") = ";
  }
  return success();
}

LogicalResult MetalEmitter::emitLabel(Block &block) {
  if (!hasBlockLabel(block))
    return block.getParentOp()->emitError("label for block not found");
  // FIXME: Add feature in `raw_indented_ostream` to ignore indent for block
  // label instead of using `getOStream`.
  os.getOStream() << getOrCreateName(block) << ":\n";
  return success();
}

LogicalResult MetalEmitter::emitOperation(Operation &op,
                                          bool trailingSemicolon) {
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          // Builtin ops.
          .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
          .Case<emitc::AddOp, emitc::ApplyOp, emitc::AssignOp,
                emitc::BitwiseAndOp, emitc::BitwiseLeftShiftOp,
                emitc::BitwiseNotOp, emitc::BitwiseOrOp,
                emitc::BitwiseRightShiftOp, emitc::BitwiseXorOp, emitc::CallOp,
                emitc::CallOpaqueOp, emitc::CastOp, emitc::CmpOp,
                emitc::ConditionalOp, emitc::ConstantOp, emitc::DeclareFuncOp,
                emitc::DivOp, emitc::ExpressionOp, emitc::ForOp, emitc::FuncOp,
                emitc::GlobalOp, emitc::GetGlobalOp, emitc::IfOp,
                emitc::IncludeOp, emitc::LogicalAndOp, emitc::LogicalNotOp,
                emitc::LogicalOrOp, emitc::MulOp, emitc::RemOp, emitc::ReturnOp,
                emitc::SubOp, emitc::SubscriptOp, emitc::UnaryMinusOp,
                emitc::UnaryPlusOp, emitc::VariableOp, emitc::VerbatimOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Func ops.
          .Case<func::CallOp, func::FuncOp, func::ReturnOp>(
              [&](auto op) { return printOperation(*this, op); })
          // gpu ops.
          .Case<gpu::GPUModuleOp, gpu::GPUFuncOp, gpu::ModuleEndOp,
                gpu::ThreadIdOp, gpu::BlockIdOp, gpu::GridDimOp,
                gpu::BlockDimOp, gpu::ReturnOp>(
              [&](auto op) { return printOperation(*this, op); })
          // metal ops.
          .Case<metal::DeviceMakeDefaultOp, metal::DeviceMakeBufferOp,
                metal::DeviceMakeCommandQueueOp, metal::CommandBufferCommitOp,
                metal::CommandQueueMakeCommandBufferOp,
                metal::CommandBufferWaitUntilCompletedOp, metal::ReleaseOp,
                metal::StoreOp, metal::GetElementOp,
                metal::CommandBufferAddBufferOp, shader::MatmulOp,
                shader::MatsumOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<memref::DeallocOp, memref::StoreOp, memref::LoadOp,
                memref::AllocOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<emitc::LiteralOp>([&](auto op) { return success(); })
          .Default([&](Operation *) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status))
    return failure();

  if (isa<emitc::LiteralOp, emitc::SubscriptOp, emitc::GetGlobalOp>(op))
    return success();

  if (getEmittedExpression() ||
      (isa<emitc::ExpressionOp>(op) &&
       shouldBeInlined(cast<emitc::ExpressionOp>(op))))
    return success();

  os << (trailingSemicolon ? ";\n" : "\n");

  return success();
}

LogicalResult MetalEmitter::emitVariableDeclaration(Location loc, Type type,
                                                    StringRef name,
                                                    bool isGPUSide) {
  if (auto arrType = dyn_cast<emitc::ArrayType>(type)) {
    if (failed(emitType(loc, arrType.getElementType(), isGPUSide)))
      return failure();
    os << " " << name;
    for (auto dim : arrType.getShape()) {
      os << "[" << dim << "]";
    }
    return success();
  }
  if (failed(emitType(loc, type, isGPUSide)))
    return failure();
  os << " " << name;
  return success();
}

LogicalResult MetalEmitter::emitKnownTypeVariableDeclaration(
    Location loc, StringRef type, StringRef name, bool trailingSemicolon) {
  os << type << " " << name;
  if (trailingSemicolon)
    os << ";\n";
  return success();
}

LogicalResult MetalEmitter::emitType(Location loc, Type type, bool isGPUSide) {
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    if (isGPUSide)
      os << "device ";
    if (failed(emitType(loc, memrefType.getElementType())))
      return failure();
    os << "*";
    return success();
  }

  if (auto iType = dyn_cast<IntegerType>(type)) {
    switch (iType.getWidth()) {
    case 1:
      return (os << "bool"), success();
    case 8:
    case 16:
    case 32:
    case 64:
      if (shouldMapToUnsigned(iType.getSignedness()))
        return (os << "uint" << iType.getWidth() << "_t"), success();
      else
        return (os << "int" << iType.getWidth() << "_t"), success();
    default:
      return emitError(loc, "cannot emit integer type ") << type;
    }
  }
  if (auto fType = dyn_cast<FloatType>(type)) {
    switch (fType.getWidth()) {
    case 32:
      return (os << "float"), success();
    case 64:
      return (os << "double"), success();
    default:
      return emitError(loc, "cannot emit float type ") << type;
    }
  }
  if (auto iType = dyn_cast<IndexType>(type))
    return (os << "size_t"), success();
  if (auto tType = dyn_cast<TensorType>(type)) {
    if (!tType.hasRank())
      return emitError(loc, "cannot emit unranked tensor type");
    if (!tType.hasStaticShape())
      return emitError(loc, "cannot emit tensor type with non static shape");
    os << "Tensor<";
    if (isa<ArrayType>(tType.getElementType()))
      return emitError(loc, "cannot emit tensor of array type ") << type;
    if (failed(emitType(loc, tType.getElementType())))
      return failure();
    auto shape = tType.getShape();
    for (auto dimSize : shape) {
      os << ", ";
      os << dimSize;
    }
    os << ">";
    return success();
  }
  if (auto tType = dyn_cast<TupleType>(type))
    return emitTupleType(loc, tType.getTypes());
  if (auto oType = dyn_cast<emitc::OpaqueType>(type)) {
    os << oType.getValue();
    return success();
  }
  if (auto aType = dyn_cast<emitc::ArrayType>(type)) {
    if (failed(emitType(loc, aType.getElementType())))
      return failure();
    for (auto dim : aType.getShape())
      os << "[" << dim << "]";
    return success();
  }
  if (auto pType = dyn_cast<emitc::PointerType>(type)) {
    if (isa<ArrayType>(pType.getPointee()))
      return emitError(loc, "cannot emit pointer to array type ") << type;
    if (failed(emitType(loc, pType.getPointee())))
      return failure();
    os << "*";
    return success();
  }
  return emitError(loc, "cannot emit type ") << type;
}

LogicalResult MetalEmitter::emitTypes(Location loc, ArrayRef<Type> types) {
  switch (types.size()) {
  case 0:
    os << "void";
    return success();
  case 1:
    return emitType(loc, types.front());
  default:
    return emitTupleType(loc, types);
  }
}

LogicalResult MetalEmitter::emitTupleType(Location loc, ArrayRef<Type> types) {
  if (llvm::any_of(types, llvm::IsaPred<ArrayType>)) {
    return emitError(loc, "cannot emit tuple of array type");
  }
  os << "std::tuple<";
  if (failed(interleaveCommaWithError(
          types, os, [&](Type type) { return emitType(loc, type); })))
    return failure();
  os << ">";
  return success();
}

LogicalResult MetalEmitter::emitMemRefSize(Location loc,
                                           MemRefType memrefType) {

  if (memrefType.getShape().size() < 1) {
    return emitError(loc, "cannot emit such a size");
  }
  int size = 1;
  for (size_t i = 0; i < memrefType.getShape().size(); i++) {
    size = size * memrefType.getDimSize(i);
  }
  return (os << size), success();
}

LogicalResult MetalEmitter::emitTypeSize(Location loc, Type type) {

  if (auto iType = dyn_cast<IntegerType>(type)) {
    return (os << (int)ceil(iType.getWidth() / 8)), success();
  }
  if (auto fType = dyn_cast<FloatType>(type)) {
    return (os << (int)ceil(fType.getWidth() / 8)), success();
  }

  return emitError(loc, "cannot emit type size") << type;
}

LogicalResult MetalEmitter::emitLinearIndex(Location loc,
                                            SmallVector<Value> sizes,
                                            SmallVector<Value> indices) {
  if (sizes.size() != indices.size())
    return failure();

  SmallVector<StringRef> stringRefs;
  stringRefs.reserve(sizes.size());

  for (int i = 0; i < (int)sizes.size(); i++) {
    stringRefs.push_back(getOrCreateName(sizes[i]));
  }

  return emitLinearIndexRefs(loc, stringRefs, indices, true);
}

// TODO passaggi degli SmallVector per riferimento

LogicalResult
MetalEmitter::emitLinearIndexRefs(Location loc,
                                  SmallVector<StringRef> stringRefs,
                                  SmallVector<Value> indices, bool preload) {

  if (!preload) {
    stringRefs.clear();
    stringRefs.reserve(3);

    stringRefs.push_back(StringRef("1"));
    stringRefs.push_back(StringRef("gridDim.y"));
    stringRefs.push_back(StringRef("gridDim.z"));
  }

  if (indices.size() > stringRefs.size()) {
    return failure();
  }

  std::string line = "";
  std::string buffer = "1";

  int start = (int)indices.size() - 1;

  for (int i = start; i >= 0; i--) {
    if (i != start) {
      buffer += " * " + stringRefs[i + 1].str();
      line += " + ";
    }
    line += getOrCreateName(indices[i]).str() + " * (" + buffer + ")";
  }

  return (os << StringRef(line), success());
}

LogicalResult mlir::metal::translateToMetal(Operation *op, raw_ostream &os,
                                            bool declareVariablesAtTop) {
  MetalEmitter emitter(os, declareVariablesAtTop);
  return emitter.emitOperation(*op, /*trailingSemicolon=*/false);
}
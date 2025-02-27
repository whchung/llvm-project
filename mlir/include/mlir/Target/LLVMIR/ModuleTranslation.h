//===- ModuleTranslation.h - MLIR to LLVM conversion ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the translation between an MLIR LLVM dialect module and
// the corresponding LLVMIR module. It only handles core LLVM IR operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_MODULETRANSLATION_H
#define MLIR_TARGET_LLVMIR_MODULETRANSLATION_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Value.h"

#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MatrixBuilder.h"
#include "llvm/IR/Value.h"

namespace mlir {
class Attribute;
class Location;
class ModuleOp;
class Operation;

namespace LLVM {

namespace detail {
class DebugTranslation;
} // end namespace detail

class LLVMFuncOp;

/// Implementation class for module translation. Holds a reference to the module
/// being translated, and the mappings between the original and the translated
/// functions, basic blocks and values. It is practically easier to hold these
/// mappings in one class since the conversion of control flow operations
/// needs to look up block and function mappings.
class ModuleTranslation {
public:
  // TODO: Currently there is no way to specify target triple and data layout
  // in Std -> LLVM dialect conversion yet, this interface exposes a way to
  // inject custom target triple and data layout when translating from LLVM
  // dialect to LLVM IR.
  //
  // Once Std -> LLVM dialect conversion honors target triple and data
  // layout this interface shall be revised.
  template <typename T = ModuleTranslation>
  static std::unique_ptr<llvm::Module>
  translateModule(Operation *m, StringRef triple = "",
                  StringRef dataLayout = "") {
    if (!satisfiesLLVMModule(m))
      return nullptr;
    if (failed(checkSupportedModuleOps(m)))
      return nullptr;
    auto llvmModule = prepareLLVMModule(m, triple, dataLayout);
    if (!llvmModule)
      return nullptr;

    LLVM::ensureDistinctSuccessors(m);

    T translator(m, std::move(llvmModule));
    if (failed(translator.convertGlobals()))
      return nullptr;
    if (failed(translator.convertFunctions()))
      return nullptr;

    return std::move(translator.llvmModule);
  }

  /// A helper method to get the single Block in an operation honoring LLVM's
  /// module requirements.
  static Block &getModuleBody(Operation *m) { return m->getRegion(0).front(); }

protected:
  /// Translate the given MLIR module expressed in MLIR LLVM IR dialect into an
  /// LLVM IR module. The MLIR LLVM IR dialect holds a pointer to an
  /// LLVMContext, the LLVM IR module will be created in that context.
  ModuleTranslation(Operation *module,
                    std::unique_ptr<llvm::Module> llvmModule);
  virtual ~ModuleTranslation();

  virtual LogicalResult convertOperation(Operation &op,
                                         llvm::IRBuilder<> &builder);
  virtual LogicalResult convertOmpOperation(Operation &op,
                                            llvm::IRBuilder<> &builder);
  static std::unique_ptr<llvm::Module>
  prepareLLVMModule(Operation *m, StringRef triple = "",
                    StringRef dayaLayout = "");

  /// A helper to look up remapped operands in the value remapping table.
  SmallVector<llvm::Value *, 8> lookupValues(ValueRange values);

private:
  /// Check whether the module contains only supported ops directly in its body.
  static LogicalResult checkSupportedModuleOps(Operation *m);

  LogicalResult convertFunctions();
  LogicalResult convertGlobals();
  LogicalResult convertOneFunction(LLVMFuncOp func);
  void connectPHINodes(LLVMFuncOp func);
  LogicalResult convertBlock(Block &bb, bool ignoreArguments);

  llvm::Constant *getLLVMConstant(llvm::Type *llvmType, Attribute attr,
                                  Location loc);

  /// Original and translated module.
  Operation *mlirModule;
  std::unique_ptr<llvm::Module> llvmModule;
  /// A converter for translating debug information.
  std::unique_ptr<detail::DebugTranslation> debugTranslation;

  /// Builder for LLVM IR generation of OpenMP constructs.
  std::unique_ptr<llvm::OpenMPIRBuilder> ompBuilder;
  /// Precomputed pointer to OpenMP dialect.
  const Dialect *ompDialect;
  /// Pointer to the llvmDialect;
  LLVMDialect *llvmDialect;

  /// Mappings between llvm.mlir.global definitions and corresponding globals.
  DenseMap<Operation *, llvm::GlobalValue *> globalsMapping;

protected:
  /// Mappings between original and translated values, used for lookups.
  llvm::StringMap<llvm::Function *> functionMapping;
  DenseMap<Value, llvm::Value *> valueMapping;
  DenseMap<Block *, llvm::BasicBlock *> blockMapping;
};

} // namespace LLVM
} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_MODULETRANSLATION_H

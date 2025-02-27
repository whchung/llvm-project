//===- ConvertToROCDLIR.cpp - MLIR to LLVM IR conversion ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR LLVM + ROCDL dialects and
// LLVM IR with ROCDL intrinsics and metadata.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/ROCDLIR.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

// Create a call to llvm intrinsic
static llvm::Value *createIntrinsicCall(llvm::IRBuilder<> &builder,
                                        llvm::Intrinsic::ID intrinsic,
                                        ArrayRef<llvm::Value *> args = {},
                                        ArrayRef<llvm::Type *> tys = {}) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::Function *fn = llvm::Intrinsic::getDeclaration(module, intrinsic, tys);
  return builder.CreateCall(fn, args);
}

// Create a call to ROCm-Device-Library function
//   Currently this routine will work only for calling ROCDL functions that
// take a single int32 argument. It is likely that the interface of this
// function will change to make it more generic.
static llvm::Value *createDeviceFunctionCall(llvm::IRBuilder<> &builder,
                                             StringRef fn_name, int parameter) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::FunctionType *function_type = llvm::FunctionType::get(
      llvm::Type::getInt64Ty(module->getContext()), // return type.
      llvm::Type::getInt32Ty(module->getContext()), // parameter type.
      false);                                       // no variadic arguments.
  llvm::Function *fn = dyn_cast<llvm::Function>(
      module->getOrInsertFunction(fn_name, function_type).getCallee());
  llvm::Value *fn_op0 = llvm::ConstantInt::get(
      llvm::Type::getInt32Ty(module->getContext()), parameter);
  return builder.CreateCall(fn, ArrayRef<llvm::Value *>(fn_op0));
}

namespace {
class ModuleTranslation : public LLVM::ModuleTranslation {
public:
  using LLVM::ModuleTranslation::ModuleTranslation;

protected:
  LogicalResult convertOperation(Operation &opInst,
                                 llvm::IRBuilder<> &builder) override {

#include "mlir/Dialect/LLVMIR/ROCDLConversions.inc"

    return LLVM::ModuleTranslation::convertOperation(opInst, builder);
  }

  /// Allow access to the constructor.
  friend LLVM::ModuleTranslation;
};
} // namespace

std::unique_ptr<llvm::Module> mlir::translateModuleToROCDLIR(Operation *m) {
  // Locate a GPU module within a Module. Use it if we find one.
  if (auto module = dyn_cast<ModuleOp>(m)) {
    auto *block = module.getBody();
    for (auto op = block->begin(); op != block->end(); ++op)
      if (auto gpuModule = dyn_cast<gpu::GPUModuleOp>(op)) {
        m = gpuModule;
        break;
      }
  }

  // lower MLIR (with RODL Dialect) to LLVM IR (with ROCDL intrinsics)
  StringRef amdgcnTriple = "amdgcn-amd-amdhsa";
  StringRef amdgcnDataLayout =
      "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-"
      "v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:"
      "1024-v2048:2048-n32:64-S32-A5-ni:7";
  auto llvmModule = LLVM::ModuleTranslation::translateModule<ModuleTranslation>(
      m, amdgcnTriple, amdgcnDataLayout);

  // foreach GPU kernel
  // 1. Insert AMDGPU_KERNEL calling convention.
  // 2. Insert amdgpu-flat-workgroup-size(1, 1024) attribute.
  for (auto func :
       ModuleTranslation::getModuleBody(m).getOps<LLVM::LLVMFuncOp>()) {
    if (!func.getAttrOfType<UnitAttr>(gpu::GPUDialect::getKernelFuncAttrName()))
      continue;

    auto *llvmFunc = llvmModule->getFunction(func.getName());

    llvmFunc->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);

    llvmFunc->addFnAttr("amdgpu-flat-work-group-size", "1, 1024");
  }

  return llvmModule;
}

namespace mlir {
void registerToROCDLIRTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-rocdlir", [](ModuleOp module, raw_ostream &output) {
        auto llvmModule = mlir::translateModuleToROCDLIR(module);
        if (!llvmModule)
          return failure();

        llvmModule->print(output, nullptr);
        return success();
      });
}
} // namespace mlir

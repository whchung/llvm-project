//===- ConvertKernelFuncToHsaco.cpp - MLIR GPU lowering passes ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert gpu kernel functions into a
// corresponding binary blob that can be executed on a ROCm GPU. Currently
// only translates the function itself but no dependencies.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToROCm/GPUToROCmPass.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/ROCDLIR.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

using namespace mlir;

namespace {
static constexpr const char *kHsacoAnnotation = "rocdl.hsaco";

/// A pass converting tagged kernel modules to hsaco blobs.
///
/// If tagged as a kernel module, each contained function is translated to ROCDL
/// IR. A user provided HsacoGenerator compiles the IR to GPU binary code in HSA
/// code object format, which is then attached as an attribute to the function.
/// The function body is erased.
class GpuKernelToHsacoPass
    : public PassWrapper<GpuKernelToHsacoPass,
                         OperationPass<gpu::GPUModuleOp>> {
public:
  GpuKernelToHsacoPass(HsacoGenerator hsacoGenerator)
      : hsacoGenerator(hsacoGenerator) {}

  void runOnOperation() override {
    gpu::GPUModuleOp module = getOperation();

    // Lock access to the llvm context.
    llvm::sys::SmartScopedLock<true> scopedLock(
        module.getContext()
            ->getRegisteredDialect<LLVM::LLVMDialect>()
            ->getLLVMContextMutex());

    // Make sure the AMDGPU target is initialized.
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmPrinter();

    auto llvmModule = translateModuleToROCDLIR(module);
    if (!llvmModule)
      return signalPassFailure();

    // Translate the module to HSA code object and attach the result as
    // attribute to the module.
    if (auto hsacoAttr = translateGPUModuleToHsacoAnnotation(
            *llvmModule, module.getLoc(), module.getName()))
      module.setAttr(kHsacoAnnotation, hsacoAttr);
    else
      signalPassFailure();
  }

private:
  std::string translateModuleToISA(llvm::Module &module,
                                   llvm::TargetMachine &targetMachine);

  /// Converts llvmModule to hsaco using the user-provided generator. Location
  /// is used for error reporting and name is forwarded to the HSACO generator
  /// to use in its logging mechanisms.
  OwnedHsaco convertModuleToHsaco(llvm::Module &llvmModule, Location loc,
                                  StringRef name);

  /// Translates llvmModule to hsaco and returns the result as attribute.
  StringAttr translateGPUModuleToHsacoAnnotation(llvm::Module &llvmModule,
                                                 Location loc, StringRef name);

  HsacoGenerator hsacoGenerator;
};

} // anonymous namespace

std::string
GpuKernelToHsacoPass::translateModuleToISA(llvm::Module &module,
                                           llvm::TargetMachine &targetMachine) {
  std::string targetISA;
  {
    // Clone the llvm module into a new context to enable concurrent compilation
    // with multiple threads.
    llvm::LLVMContext llvmContext;
    auto clone = LLVM::cloneModuleIntoNewContext(&llvmContext, &module);

    llvm::raw_string_ostream stream(targetISA);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegenPasses;
    targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                      llvm::CGFT_AssemblyFile);
    codegenPasses.run(*clone);
  }

  return targetISA;
}

OwnedHsaco GpuKernelToHsacoPass::convertModuleToHsaco(llvm::Module &llvmModule,
                                                      Location loc,
                                                      StringRef name) {
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  {
    std::string error;
    constexpr const char *rocmTriple = "amdgcn-amd-amdhsa";
    llvm::Triple triple(rocmTriple);
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget("", triple, error);
    if (target == nullptr) {
      emitError(loc, "cannot initialize target triple");
      return {};
    }
    // TODO(whchung): be able to set target.
    targetMachine.reset(
        target->createTargetMachine(triple.str(), "gfx900", "", {}, {}));
  }

  llvmModule.setDataLayout(targetMachine->createDataLayout());

  auto targetISA = translateModuleToISA(llvmModule, *targetMachine);

  return hsacoGenerator(targetISA, loc, name);
}

StringAttr GpuKernelToHsacoPass::translateGPUModuleToHsacoAnnotation(
    llvm::Module &llvmModule, Location loc, StringRef name) {
  auto hsaco = convertModuleToHsaco(llvmModule, loc, name);
  if (!hsaco)
    return {};
  return StringAttr::get({hsaco->data(), hsaco->size()}, loc->getContext());
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createConvertGPUKernelToHsacoPass(HsacoGenerator hsacoGenerator) {
  return std::make_unique<GpuKernelToHsacoPass>(hsacoGenerator);
}

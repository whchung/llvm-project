//===- ConvertKernelFuncToCubin.cpp - MLIR GPU lowering passes ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert gpu kernel functions into a
// corresponding binary blob that can be executed on a CUDA GPU. Currently
// only translates the function itself but no dependencies.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/NVVMIR.h"

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
// TODO(herhut): Move to shared location.
static constexpr const char *kCubinAnnotation = "nvvm.cubin";

/// A pass converting tagged kernel modules to cubin blobs.
///
/// If tagged as a kernel module, each contained function is translated to NVVM
/// IR and further to PTX. A user provided CubinGenerator compiles the PTX to
/// GPU binary code, which is then attached as an attribute to the function. The
/// function body is erased.
class GpuKernelToCubinPass
    : public PassWrapper<GpuKernelToCubinPass,
                         OperationPass<gpu::GPUModuleOp>> {
public:
  GpuKernelToCubinPass(CubinGenerator cubinGenerator)
      : cubinGenerator(cubinGenerator) {}

  void runOnOperation() override {
    gpu::GPUModuleOp module = getOperation();

    // Lock access to the llvm context.
    llvm::sys::SmartScopedLock<true> scopedLock(
        module.getContext()
            ->getRegisteredDialect<LLVM::LLVMDialect>()
            ->getLLVMContextMutex());

    // Make sure the NVPTX target is initialized.
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();

    auto llvmModule = translateModuleToNVVMIR(module);
    if (!llvmModule)
      return signalPassFailure();

    // Translate the module to CUBIN and attach the result as attribute to the
    // module.
    if (auto cubinAttr = translateGPUModuleToCubinAnnotation(
            *llvmModule, module.getLoc(), module.getName()))
      module.setAttr(kCubinAnnotation, cubinAttr);
    else
      signalPassFailure();
  }

private:
  std::string translateModuleToPtx(llvm::Module &module,
                                   llvm::TargetMachine &target_machine);

  /// Converts llvmModule to cubin using the user-provided generator. Location
  /// is used for error reporting and name is forwarded to the CUBIN generator
  /// to use in its logging mechanisms.
  OwnedCubin convertModuleToCubin(llvm::Module &llvmModule, Location loc,
                                  StringRef name);

  /// Translates llvmModule to cubin and returns the result as attribute.
  StringAttr translateGPUModuleToCubinAnnotation(llvm::Module &llvmModule,
                                                 Location loc, StringRef name);

  CubinGenerator cubinGenerator;
};

} // anonymous namespace

std::string GpuKernelToCubinPass::translateModuleToPtx(
    llvm::Module &module, llvm::TargetMachine &target_machine) {
  std::string ptx;
  {
    // Clone the llvm module into a new context to enable concurrent compilation
    // with multiple threads.
    // TODO(zinenko): Reevaluate model of ownership of LLVMContext in
    //                LLVMDialect.
    llvm::LLVMContext llvmContext;
    auto clone = LLVM::cloneModuleIntoNewContext(&llvmContext, &module);

    llvm::raw_string_ostream stream(ptx);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegen_passes;
    target_machine.addPassesToEmitFile(codegen_passes, pstream, nullptr,
                                       llvm::CGFT_AssemblyFile);
    codegen_passes.run(*clone);
  }

  return ptx;
}

OwnedCubin GpuKernelToCubinPass::convertModuleToCubin(llvm::Module &llvmModule,
                                                      Location loc,
                                                      StringRef name) {
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  {
    std::string error;
    // TODO(herhut): Make triple configurable.
    constexpr const char *cudaTriple = "nvptx64-nvidia-cuda";
    llvm::Triple triple(cudaTriple);
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget("", triple, error);
    if (target == nullptr) {
      emitError(loc, "cannot initialize target triple");
      return {};
    }
    targetMachine.reset(
        target->createTargetMachine(triple.str(), "sm_35", "+ptx60", {}, {}));
  }

  // Set the data layout of the llvm module to match what the ptx target needs.
  llvmModule.setDataLayout(targetMachine->createDataLayout());

  auto ptx = translateModuleToPtx(llvmModule, *targetMachine);

  return cubinGenerator(ptx, loc, name);
}

StringAttr GpuKernelToCubinPass::translateGPUModuleToCubinAnnotation(
    llvm::Module &llvmModule, Location loc, StringRef name) {
  auto cubin = convertModuleToCubin(llvmModule, loc, name);
  if (!cubin)
    return {};
  return StringAttr::get({cubin->data(), cubin->size()}, loc->getContext());
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createConvertGPUKernelToCubinPass(CubinGenerator cubinGenerator) {
  return std::make_unique<GpuKernelToCubinPass>(cubinGenerator);
}
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

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"

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

/// A pass converting tagged kernel modules to a blob with target instructions.
///
/// If tagged as a kernel module, each contained function is translated to
/// user-specified IR. A user provided BlobGenerator then compiles the IR to
/// GPU binary code, which is then attached as an attribute to the function.
/// The function body is erased.
class GpuKernelToBlobPass
    : public PassWrapper<GpuKernelToBlobPass, OperationPass<gpu::GPUModuleOp>> {
public:
  GpuKernelToBlobPass(InitBackendCallback initBackendCallback,
                      LoweringCallback loweringCallback,
                      BlobGenerator blobGenerator, StringRef triple,
                      StringRef targetChip, StringRef features,
                      StringRef gpuBinaryAnnotation)
      : initBackendCallback(initBackendCallback),
        loweringCallback(loweringCallback), blobGenerator(blobGenerator),
        triple(triple), targetChip(targetChip), features(features),
        blobAnnotation(gpuBinaryAnnotation) {}

  void runOnOperation() override {
    gpu::GPUModuleOp module = getOperation();

    // Lock access to the llvm context.
    llvm::sys::SmartScopedLock<true> scopedLock(
        module.getContext()
            ->getRegisteredDialect<LLVM::LLVMDialect>()
            ->getLLVMContextMutex());

    // Initialize LLVM backend.
    if (!succeeded(initBackendCallback()))
      return signalPassFailure();

    // Lower the module to a llvm module.
    std::unique_ptr<llvm::Module> llvmModule = nullptr;
    if (!succeeded(loweringCallback(module, llvmModule)))
      return signalPassFailure();

    // Translate the llvm module to a target blob and attach the result as
    // attribute to the module.
    if (auto blobAttr = translateGPUModuleToBinaryAnnotation(
            *llvmModule, module.getLoc(), module.getName()))
      module.setAttr(blobAnnotation, blobAttr);
    else
      signalPassFailure();
  }

private:
  std::string translateModuleToISA(llvm::Module &module,
                                   llvm::TargetMachine &targetMachine);

  /// Converts llvmModule to a lob with target instructions using the
  /// user-provided generator. Location is used for error reporting and name is
  /// forwarded to the blob generator to use in its logging mechanisms.
  OwnedBlob convertModuleToBlob(llvm::Module &llvmModule, Location loc,
                                StringRef name);

  /// Translates llvmModule to a blob with target instructions and returns the
  /// result as attribute.
  StringAttr translateGPUModuleToBinaryAnnotation(llvm::Module &llvmModule,
                                                  Location loc, StringRef name);

  InitBackendCallback initBackendCallback;
  LoweringCallback loweringCallback;
  BlobGenerator blobGenerator;
  llvm::Triple triple;
  StringRef targetChip;
  StringRef features;
  StringRef blobAnnotation;
};

} // anonymous namespace

std::string
GpuKernelToBlobPass::translateModuleToISA(llvm::Module &module,
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

OwnedBlob GpuKernelToBlobPass::convertModuleToBlob(llvm::Module &llvmModule,
                                                   Location loc,
                                                   StringRef name) {
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  {
    std::string error;
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget("", triple, error);
    if (target == nullptr) {
      emitError(loc, "cannot initialize target triple");
      return {};
    }
    targetMachine.reset(target->createTargetMachine(triple.str(), targetChip,
                                                    features, {}, {}));
  }

  llvmModule.setDataLayout(targetMachine->createDataLayout());

  auto targetISA = translateModuleToISA(llvmModule, *targetMachine);

  return blobGenerator(targetISA, loc, name);
}

StringAttr GpuKernelToBlobPass::translateGPUModuleToBinaryAnnotation(
    llvm::Module &llvmModule, Location loc, StringRef name) {
  auto blob = convertModuleToBlob(llvmModule, loc, name);
  if (!blob)
    return {};
  return StringAttr::get({blob->data(), blob->size()}, loc->getContext());
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createConvertGPUKernelToBlobPass(InitBackendCallback initBackendCallback,
                                       LoweringCallback loweringCallback,
                                       BlobGenerator blobGenerator,
                                       StringRef triple, StringRef targetChip,
                                       StringRef features,
                                       StringRef gpuBinaryAnnotation) {
  return std::make_unique<GpuKernelToBlobPass>(
      initBackendCallback, loweringCallback, blobGenerator, triple, targetChip,
      features, gpuBinaryAnnotation);
}

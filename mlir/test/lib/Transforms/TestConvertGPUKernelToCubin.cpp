//===- TestConvertGPUKernelToCubin.cpp - Test gpu kernel cubin lowering ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/NVVMIR.h"
#include "llvm/Support/TargetSelect.h"
using namespace mlir;

#if MLIR_CUDA_CONVERSIONS_ENABLED
static LogicalResult initNVPTXBackendCallback() {
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
  return success();
}

static LogicalResult
compileModuleToNVVMIR(Operation *m, std::unique_ptr<llvm::Module> &llvmModule) {
  llvmModule = translateModuleToNVVMIR(m);
  if (llvmModule)
    return success();
  return failure();
}

static OwnedBlob compilePtxToCubinForTesting(const std::string &, Location,
                                             StringRef) {
  const char data[] = "CUBIN";
  return std::make_unique<std::vector<char>>(data, data + sizeof(data) - 1);
}

namespace mlir {
void registerTestConvertGPUKernelToCubinPass() {
  PassPipelineRegistration<>(
      "test-kernel-to-cubin",
      "Convert all kernel functions to CUDA cubin blobs",
      [](OpPassManager &pm) {
        pm.addPass(createConvertGPUKernelToBlobPass(
            initNVPTXBackendCallback, compileModuleToNVVMIR,
            compilePtxToCubinForTesting, "nvptx64-nvidia-cuda", "sm_35",
            "+ptx60", "nvvm.cubin"));
      });
}
} // namespace mlir
#endif

//===- GPUToROCmPass.h - MLIR ROCm runtime support --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUTOROCM_GPUTOROCMPASS_H_
#define MLIR_CONVERSION_GPUTOROCM_GPUTOROCMPASS_H_

#include "mlir/Support/LLVM.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mlir {

class Location;
class ModuleOp;

template <typename T>
class OperationPass;

namespace gpu {
class GPUModuleOp;
} // namespace gpu

namespace LLVM {
class LLVMDialect;
} // namespace LLVM

using OwnedHsaco = std::unique_ptr<std::vector<char>>;
using HsacoGenerator =
    std::function<OwnedHsaco(const std::string &, Location, StringRef)>;

/// Creates a pass to convert kernel functions into HSA code object blobs.
///
/// This transformation takes the body of each function that is annotated with
/// the 'gpu.kernel' attribute, copies it to a new LLVM module, compiles the
/// module with help of the AMDGPU backend to HSA code object and then invokes
/// the provided hsacoGenerator to produce a binary blob (the hsaco). Such blob
/// is then attached as a string attribute named 'rocdl.hsaco' to the kernel
/// function.
/// After the transformation, the body of the kernel function is removed (i.e.,
/// it is turned into a declaration).
std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
createConvertGPUKernelToHsacoPass(HsacoGenerator hsacoGenerator);

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOROCM_GPUTOROCMPASS_H_

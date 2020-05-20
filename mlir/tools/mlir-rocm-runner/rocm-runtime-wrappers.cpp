//===- rocm-runtime-wrappers.cpp - MLIR ROCM runner wrapper library -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements C wrappers around the ROCM library for easy linking in ORC jit.
// Also adds some debugging helpers that are helpful when writing MLIR code to
// run on GPUs.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <numeric>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "hip/hip_runtime.h"

namespace {
int32_t reportErrorIfAny(hipError_t result, const char *where) {
  if (result != hipSuccess) {
    llvm::errs() << "HIP failed with " << result << " in " << where << "\n";
  }
  return result;
}
} // anonymous namespace

extern "C" int32_t mgpuModuleLoad(void **module, void *data) {
  int32_t err = reportErrorIfAny(
      hipModuleLoadData(reinterpret_cast<hipModule_t *>(module), data),
      "ModuleLoad");
  return err;
}

extern "C" int32_t mgpuModuleGetFunction(void **function, void *module,
                                         const char *name) {
  return reportErrorIfAny(
      hipModuleGetFunction(reinterpret_cast<hipFunction_t *>(function),
                           reinterpret_cast<hipModule_t>(module), name),
      "GetFunction");
}

// The wrapper uses intptr_t instead of ROCM's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" int32_t mgpuLaunchKernel(void *function, intptr_t gridX,
                                    intptr_t gridY, intptr_t gridZ,
                                    intptr_t blockX, intptr_t blockY,
                                    intptr_t blockZ, int32_t smem, void *stream,
                                    void **params, void **extra) {
  return reportErrorIfAny(
      hipModuleLaunchKernel(reinterpret_cast<hipFunction_t>(function), gridX,
                            gridY, gridZ, blockX, blockY, blockZ, smem,
                            reinterpret_cast<hipStream_t>(stream), params,
                            extra),
      "LaunchKernel");
}

extern "C" void *mgpuGetStreamHelper() {
  hipStream_t stream;
  reportErrorIfAny(hipStreamCreate(&stream), "StreamCreate");
  return stream;
}

extern "C" int32_t mgpuStreamSynchronize(void *stream) {
  return reportErrorIfAny(
      hipStreamSynchronize(reinterpret_cast<hipStream_t>(stream)),
      "StreamSync");
}

/// Helper functions for writing mlir example code

// Allows to register byte array with the ROCM runtime. Helpful until we have
// transfer functions implemented.
extern "C" void mgpuMemHostRegister(void *ptr, uint64_t sizeBytes) {
  reportErrorIfAny(hipHostRegister(ptr, sizeBytes, /*flags=*/0),
                   "MemHostRegister");
}

// A struct that corresponds to how MLIR represents memrefs.
template <typename T, int N>
struct MemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

// Allows to register a MemRef with the ROCM runtime. Initializes array with
// value. Helpful until we have transfer functions implemented.
template <typename T>
void mgpuMemHostRegisterMemRef(T *pointer, llvm::ArrayRef<int64_t> sizes,
                               llvm::ArrayRef<int64_t> strides, T value) {
  assert(sizes.size() == strides.size());
  llvm::SmallVector<int64_t, 4> denseStrides(strides.size());

  std::partial_sum(sizes.rbegin(), sizes.rend(), denseStrides.rbegin(),
                   std::multiplies<int64_t>());
  auto count = denseStrides.front();

  // Only densely packed tensors are currently supported.
  std::rotate(denseStrides.begin(), denseStrides.begin() + 1,
              denseStrides.end());
  denseStrides.back() = 1;
  assert(strides == llvm::makeArrayRef(denseStrides));

  std::fill_n(pointer, count, value);
  mgpuMemHostRegister(pointer, count * sizeof(T));
}

extern "C" void mgpuMemHostRegisterMemRef1dFloat(float *allocated,
                                                 float *aligned, int64_t offset,
                                                 int64_t size, int64_t stride) {
  mgpuMemHostRegisterMemRef(aligned + offset, {size}, {stride}, 1.23f);
}

extern "C" void mgpuMemHostRegisterMemRef1dInt32(int32_t *allocated,
                                                 int32_t *aligned,
                                                 int64_t offset, int64_t size,
                                                 int64_t stride) {
  mgpuMemHostRegisterMemRef(aligned + offset, {size}, {stride}, 123);
}

template <typename T>
void mgpuMemHostGetDevicePointer(T *hostPtr, T **devicePtr) {
  reportErrorIfAny(hipSetDevice(0), "hipSetDevice");
  reportErrorIfAny(
      hipHostGetDevicePointer((void **)devicePtr, hostPtr, /*flags=*/0),
      "hipHostGetDevicePointer");
}

extern "C" MemRefType<float, 1>
mgpuMemHostGetDeviceMemRef1dFloat(float *allocated, float *aligned,
                                  int64_t offset, int64_t size,
                                  int64_t stride) {
  float *devicePtr = nullptr;
  mgpuMemHostGetDevicePointer(aligned, &devicePtr);
  return {devicePtr, devicePtr, offset, {size}, {stride}};
}

extern "C" MemRefType<int32_t, 1>
mgpuMemHostGetDeviceMemRef1dInt32(int32_t *allocated, int32_t *aligned,
                                  int64_t offset, int64_t size,
                                  int64_t stride) {
  int32_t *devicePtr = nullptr;
  mgpuMemHostGetDevicePointer(aligned, &devicePtr);
  return {devicePtr, devicePtr, offset, {size}, {stride}};
}

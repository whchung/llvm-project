//===- PassDetail.h - Loop Pass class details -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_MIOPEN_TRANSFORMS_PASSDETAIL_H_
#define DIALECT_MIOPEN_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_CLASSES
#include "mlir/Dialect/MIOpen/Passes.h.inc"

} // end namespace mlir

#endif // DIALECT_MIOPEN_TRANSFORMS_PASSDETAIL_H_

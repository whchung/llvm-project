//===--- AMDGPUDSReadClusting.cpp - AMDGPU DSRead Clustering  -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a DAG scheduling mutation to cluster LDS reads.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUDSReadClustering.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"

using namespace llvm;

namespace {

class DSReadClustering : public BaseMemOpClusterMutation {
public:
  DSReadClustering(const TargetInstrInfo *tii, const TargetRegisterInfo *tri)
      : BaseMemOpClusterMutation(tii, tri, true) {}
};

} // end namespace

namespace llvm {

std::unique_ptr<ScheduleDAGMutation>
createAMDGPUDSReadClusterDAGMutation(const TargetInstrInfo *TII,
                                     const TargetRegisterInfo *TRI) {
  return std::make_unique<DSReadClustering>(TII, TRI);
}

} // end namespace llvm

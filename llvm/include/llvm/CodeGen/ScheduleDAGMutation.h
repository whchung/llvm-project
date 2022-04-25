//===- ScheduleDAGMutation.h - MachineInstr Scheduling ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ScheduleDAGMutation class, which represents
// a target-specific mutation of the dependency graph for scheduling.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SCHEDULEDAGMUTATION_H
#define LLVM_CODEGEN_SCHEDULEDAGMUTATION_H

#include "llvm/CodeGen/ScheduleDAG.h"

namespace llvm {

class ScheduleDAGInstrs;

/// Mutate the DAG as a postpass after normal DAG building.
class ScheduleDAGMutation {
  virtual void anchor();

public:
  virtual ~ScheduleDAGMutation() = default;

  virtual void apply(ScheduleDAGInstrs *DAG) = 0;
};

//===----------------------------------------------------------------------===//
// BaseMemOpClusterMutation - DAG post-processing to cluster loads or stores.
//===----------------------------------------------------------------------===//

/// Post-process the DAG to create cluster edges between neighboring
/// loads or between neighboring stores.
class BaseMemOpClusterMutation : public ScheduleDAGMutation {
protected:
  struct MemOpInfo {
    SUnit *SU;
    SmallVector<const MachineOperand *, 4> BaseOps;
    int64_t Offset;
    unsigned Width;

    MemOpInfo(SUnit *SU, ArrayRef<const MachineOperand *> BaseOps,
              int64_t Offset, unsigned Width)
        : SU(SU), BaseOps(BaseOps.begin(), BaseOps.end()), Offset(Offset),
          Width(Width) {}

    static bool Compare(const MachineOperand *const &A,
                        const MachineOperand *const &B);

    bool operator<(const MemOpInfo &RHS) const {
      // FIXME: Don't compare everything twice. Maybe use C++20 three way
      // comparison instead when it's available.
      if (std::lexicographical_compare(BaseOps.begin(), BaseOps.end(),
                                       RHS.BaseOps.begin(), RHS.BaseOps.end(),
                                       Compare))
        return true;
      if (std::lexicographical_compare(RHS.BaseOps.begin(), RHS.BaseOps.end(),
                                       BaseOps.begin(), BaseOps.end(), Compare))
        return false;
      if (Offset != RHS.Offset)
        return Offset < RHS.Offset;
      return SU->NodeNum < RHS.SU->NodeNum;
    }
  };

  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  bool IsLoad;

public:
  BaseMemOpClusterMutation(const TargetInstrInfo *tii,
                           const TargetRegisterInfo *tri, bool IsLoad)
      : TII(tii), TRI(tri), IsLoad(IsLoad) {}

  void apply(ScheduleDAGInstrs *DAGInstrs) override;

protected:
  void clusterNeighboringMemOps(ArrayRef<MemOpInfo> MemOps, bool FastCluster,
                                ScheduleDAGInstrs *DAG);
  virtual void collectMemOpRecords(std::vector<SUnit> &SUnits,
                                   SmallVectorImpl<MemOpInfo> &MemOpRecords);
  bool groupMemOps(ArrayRef<MemOpInfo> MemOps, ScheduleDAGInstrs *DAG,
                   DenseMap<unsigned, SmallVector<MemOpInfo, 32>> &Groups);
};

} // end namespace llvm

#endif // LLVM_CODEGEN_SCHEDULEDAGMUTATION_H

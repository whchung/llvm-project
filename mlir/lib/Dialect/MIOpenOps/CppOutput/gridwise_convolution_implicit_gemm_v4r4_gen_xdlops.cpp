//===- ConvertToMIOpenCPP.cpp - MLIR to MIOpen C++ conversion -------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR MIOpen dialect and C++.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpenOps/MIOpenCPP.h"
#include "mlir/Dialect/MIOpenOps/MIOpenOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Translation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

namespace {
// result string to keep C++ source / header / flags emission.
std::string resultStr;

class TunableParameters : public TunableParametersBase {
public:

  // TBD: review logic here as they may be tied to NCHW layout.
  std::tuple<int, int, int, int, bool>
  calculateGemmABlockCopyPerformanceParameters(TunableParams &param) {
    int64_t clusterLengths_GemmK = 0;
    int64_t clusterLengths_GemmM = 0;
    int64_t srcDataPerRead_Gemm = 4;
    int64_t dstDataPerWrite_GemmM = 4;

    const auto waveSize = 64;
    const auto blockSize = param.gemmNPerBlock * param.gemmMPerBlock /
                           (param.gemmMPerWave * param.gemmNPerWave) * waveSize;

    // Determine vectorization dimensions and lengths.
    int64_t vectorizableLength = 0;

    // Find the fastest changing dimension.
    bool gemmKVectorizable = false;
    bool gemmMVectorizable = false;
    if (ctx.dimIndexVal["k"].first == 3) {
      // When K is the fastest changing dimension,
      // gemmM dimension is vectorizable.
      // vectorization width depending on length of K.
      vectorizableLength = ctx.dimIndexVal["k"].second;
      gemmMVectorizable = true;

      // gemmK dimension non-vectorizable.
    } else {
      // gemmK dimension vectorizable,
      // depending on which among C, Y, X be the fastest changing dimension.
      if (ctx.dimIndexVal["k"].first == 0) {
        // dimKF is the lowest changing dimension, which means dimC/dimY/dimX
        vectorizableLength = ctx.dimIndexVal["c"].second *
                             ctx.dimIndexVal["y"].second *
                             ctx.dimIndexVal["x"].second;
      } else {
        if (ctx.dimIndexVal["c"].first == 3) {
          vectorizableLength = ctx.dimIndexVal["c"].second;
        } else if (ctx.dimIndexVal["x"].first == 3 &&
                   ctx.dimIndexVal["y"].first == 2) {
          vectorizableLength =
              ctx.dimIndexVal["y"].second * ctx.dimIndexVal["x"].second;
        }
      }

      gemmKVectorizable = true;
      // gemmM dimension non-vectorizable.
    }

    if (gemmMVectorizable) {
      srcDataPerRead_Gemm = gcd(srcDataPerRead_Gemm, param.gemmMPerBlock);
    } else if (gemmKVectorizable) {
      srcDataPerRead_Gemm = gcd(srcDataPerRead_Gemm, param.gemmKPerBlock);
    } else {
      srcDataPerRead_Gemm = 1;
    }

    // calculate threadwise copy size
    const auto a_data_per_thread_copy =
        (param.gemmKPerBlock * param.gemmMPerBlock) / blockSize;

    if (!(a_data_per_thread_copy > 0))
      return std::make_tuple(-1, -1, -1, -1, false);

    // GemmABlockCopySrcDataPerRead_GemmK also bounded by size of threadwise
    // copy
    srcDataPerRead_Gemm = gcd(srcDataPerRead_Gemm, a_data_per_thread_copy);

    // decide threadwise copy lengths
    const auto a_data_per_thread_copy_gemm_vectorized = srcDataPerRead_Gemm;
    const auto a_data_per_thread_copy_gemm_nonvectorized =
        a_data_per_thread_copy / a_data_per_thread_copy_gemm_vectorized;

    int64_t a_data_per_thread_copy_gemmk = 0;
    int64_t a_data_per_thread_copy_gemmm = 0;
    if (gemmMVectorizable) {
      a_data_per_thread_copy_gemmk = a_data_per_thread_copy_gemm_nonvectorized;
      a_data_per_thread_copy_gemmm = a_data_per_thread_copy_gemm_vectorized;
    } else {
      a_data_per_thread_copy_gemmk = a_data_per_thread_copy_gemm_vectorized;
      a_data_per_thread_copy_gemmm = a_data_per_thread_copy_gemm_nonvectorized;
    }

    // GemmABlockCopyDstDataPerWrite_GemmM also bounded by size of threadwise
    // copy
    dstDataPerWrite_GemmM =
        gcd(dstDataPerWrite_GemmM, a_data_per_thread_copy_gemmm);

    // calculate blockwise copy thread cluster lengths
    clusterLengths_GemmK = param.gemmKPerBlock / a_data_per_thread_copy_gemmk;
    clusterLengths_GemmM = param.gemmMPerBlock / a_data_per_thread_copy_gemmm;

    if (!(clusterLengths_GemmK > 0 && clusterLengths_GemmM > 0))
      return std::make_tuple(-1, -1, -1, -1, false);

    // llvm::errs() << "======================\n";
    // llvm::errs() << "Matrix A\n";
    // llvm::errs() << "gemmK Vectorizable: " << gemmKVectorizable << "\n";
    // llvm::errs() << "gemmM Vectorizable: " << gemmMVectorizable << "\n";
    // llvm::errs() << "cluster lengths gemmK: " << clusterLengths_GemmK <<
    // "\n"; llvm::errs() << "cluster lengths gemmM: " << clusterLengths_GemmM <<
    // "\n"; llvm::errs() << "data per read: " << srcDataPerRead_Gemm << "\n";
    // llvm::errs() << "data per write: " << dstDataPerWrite_GemmM << "\n";
    // llvm::errs() << "======================\n";

    return std::make_tuple(clusterLengths_GemmK, clusterLengths_GemmM,
                           srcDataPerRead_Gemm, dstDataPerWrite_GemmM, true);
  }

  // TBD: review logic here as they may be tied to NCHW layout.
  std::tuple<int, int, int, int, bool>
  calculateGemmBBlockCopyPerformanceParameters(TunableParams &param) {
    int64_t clusterLengths_GemmK = 0;
    int64_t clusterLengths_GemmN = 0;
    int64_t srcDataPerRead_Gemm = 4;
    int64_t dstDataPerWrite_GemmN = 4;

    const int64_t waveSize = 64;
    const int64_t blockSize = param.gemmNPerBlock * param.gemmMPerBlock /
                              (param.gemmMPerWave * param.gemmNPerWave) *
                              waveSize;
    // Determine vectorization dimensions and lengths.
    int64_t vectorizableLength = 0;

    bool gemmKVectorizable = false;
    bool gemmNVectorizable = false;
    // Find the fastest changing dimension.
    if (ctx.dimIndexVal["ni"].first == 3) {
      // When N is the fastest changing dimension,
      // gemmN dimension is vectorizable.
      // vectorization width depending on length of N.
      vectorizableLength = ctx.dimIndexVal["ni"].second;
      gemmNVectorizable = true;

      // gemmK dimension non-vectorizable.
    } else if (ctx.dimIndexVal["ci"].first == 3) {
      // When C is the fastest changing dimension,
      // gemmK dimension vectorizable.
      // vectorization width depending on length of C.
      vectorizableLength = ctx.dimIndexVal["c"].second;
      gemmKVectorizable = true;
      // gemmN dimension non-vectorizable.
    } else if (ctx.dimIndexVal["ci"].first == 0) {
      if (ctx.dimIndexVal["y"].second == 1 &&
          ctx.dimIndexVal["x"].second == 1 && ctx.strideVal[0] == 1 &&
          ctx.strideVal[1] == 1 && ctx.paddingVal[0] == 0 &&
          ctx.paddingVal[1] == 0 && ctx.paddingVal[2] == 0 &&
          ctx.paddingVal[3] == 0) {
        // \todo there are more configs that can go through this if branch
        srcDataPerRead_Gemm =
            gcd(srcDataPerRead_Gemm, ctx.dimIndexVal["ni"].second *
                                         ctx.dimIndexVal["hi"].second *
                                         ctx.dimIndexVal["wi"].second);

        gemmNVectorizable = true;
      } else {
        srcDataPerRead_Gemm = 1;
      }
    } else if (ctx.dimIndexVal["hi"].first == 2 &&
               ctx.dimIndexVal["wi"].first == 3) {
      if (ctx.dimIndexVal["y"].second == 1 &&
          ctx.dimIndexVal["x"].second == 1 && ctx.strideVal[0] == 1 &&
          ctx.strideVal[1] == 1 && ctx.paddingVal[0] == 0 &&
          ctx.paddingVal[1] == 0 && ctx.paddingVal[2] == 0 &&
          ctx.paddingVal[3] == 0) {
        // \todo there are more configs that can go through this if branch
        srcDataPerRead_Gemm =
            gcd(srcDataPerRead_Gemm,
                ctx.dimIndexVal["hi"].second * ctx.dimIndexVal["wi"].second);

        gemmNVectorizable = true;
      } else if (ctx.strideVal[1] == 1) {
        srcDataPerRead_Gemm = gcd(srcDataPerRead_Gemm, ctx.paddingVal[2],
                                  ctx.dimIndexVal["wi"].second,
                                  ctx.paddingVal[3], ctx.dilationVal[1]);

        gemmNVectorizable = true;
      } else {
        srcDataPerRead_Gemm = 1;
      }
    } else {
      srcDataPerRead_Gemm = 1;
    }

    if (gemmNVectorizable) {
      srcDataPerRead_Gemm = gcd(srcDataPerRead_Gemm, param.gemmNPerBlock);
    } else if (gemmKVectorizable) {
      srcDataPerRead_Gemm = gcd(srcDataPerRead_Gemm, param.gemmKPerBlock);
    } else {
      srcDataPerRead_Gemm = 1;
    }

    // calculate threadwise copy size
    const int64_t b_data_per_thread_copy =
        (param.gemmKPerBlock * param.gemmNPerBlock) / blockSize;

    if (!(b_data_per_thread_copy > 0))
      return std::make_tuple(-1, -1, -1, -1, false);

    // GemmBBlockCopySrcDataPerRead_GemmN also bounded by size of threadwise
    // copy
    srcDataPerRead_Gemm = gcd(srcDataPerRead_Gemm, b_data_per_thread_copy);

    const int64_t b_data_per_thread_copy_gemm_vectorized = srcDataPerRead_Gemm;
    const int64_t b_data_per_thread_copy_gemm_nonvectorized =
        b_data_per_thread_copy / b_data_per_thread_copy_gemm_vectorized;

    int64_t b_data_per_thread_copy_gemmk = 0;
    int64_t b_data_per_thread_copy_gemmn = 0;
    if (gemmNVectorizable) {
      b_data_per_thread_copy_gemmk = b_data_per_thread_copy_gemm_nonvectorized;
      b_data_per_thread_copy_gemmn = b_data_per_thread_copy_gemm_vectorized;
    } else {
      b_data_per_thread_copy_gemmk = b_data_per_thread_copy_gemm_vectorized;
      b_data_per_thread_copy_gemmn = b_data_per_thread_copy_gemm_nonvectorized;
    }

    // GemmBBlockCopyDstDataPerWrite_GemmN also bounded by size of threadwise
    // copy
    dstDataPerWrite_GemmN =
        gcd(dstDataPerWrite_GemmN, b_data_per_thread_copy_gemmn);

    // calculate blockwise copy thread cluster lengths
    clusterLengths_GemmK = param.gemmKPerBlock / b_data_per_thread_copy_gemmk;
    clusterLengths_GemmN = param.gemmNPerBlock / b_data_per_thread_copy_gemmn;

    if (!(clusterLengths_GemmK > 0 && clusterLengths_GemmN > 0))
      return std::make_tuple(-1, -1, -1, -1, false);

    // llvm::errs() << "======================\n";
    // llvm::errs() << "Matrix B\n";
    // llvm::errs() << "gemmK Vectorizable: " << gemmKVectorizable << "\n";
    // llvm::errs() << "gemmN Vectorizable: " << gemmNVectorizable << "\n";
    // llvm::errs() << "cluster lengths gemmK: " << clusterLengths_GemmK <<
    // "\n"; llvm::errs() << "cluster lengths gemmN: " << clusterLengths_GemmN <<
    // "\n"; llvm::errs() << "data per read: " << srcDataPerRead_Gemm << "\n";
    // llvm::errs() << "data per write: " << dstDataPerWrite_GemmN << "\n";
    // llvm::errs() << "======================\n";

    return std::make_tuple(clusterLengths_GemmK, clusterLengths_GemmN,
                           srcDataPerRead_Gemm, dstDataPerWrite_GemmN, true);
  }

  // TBD: review logic here as they may be tied to NCHW layout.
  std::tuple<std::size_t, bool> calculateLdsNumberOfByte(TunableParams &param) {
    std::size_t lds_size = 0;

    bool valid = false;

    int64_t gemmABlockCopyDescDataPerWriteGemmM = 0;
    int64_t gemmABlockCopyClusterLengths_GemmM = 0;
    std::tie(std::ignore, gemmABlockCopyClusterLengths_GemmM, std::ignore,
             gemmABlockCopyDescDataPerWriteGemmM, valid) =
        calculateGemmABlockCopyPerformanceParameters(param);

    if (!valid)
      return std::make_tuple(0, false);

    int64_t gemmBBlockCopyDescDataPerWriteGemmN = 0;
    int64_t gemmBBlockCopyClusterLengths_GemmN = 0;
    std::tie(std::ignore, gemmBBlockCopyClusterLengths_GemmN, std::ignore,
             gemmBBlockCopyDescDataPerWriteGemmN, valid) =
        calculateGemmBBlockCopyPerformanceParameters(param);

    if (!valid)
      return std::make_tuple(0, false);

    int64_t threadGemmDataPerRead_GemmM =
        param.gemmMPerBlock / gemmABlockCopyClusterLengths_GemmM;
    int64_t threadGemmDataPerRead_GemmN =
        param.gemmNPerBlock / gemmBBlockCopyClusterLengths_GemmN;

    const auto max_lds_align =
        lcm(gemmABlockCopyDescDataPerWriteGemmM,
            gemmBBlockCopyDescDataPerWriteGemmN, threadGemmDataPerRead_GemmM,
            threadGemmDataPerRead_GemmN);

    const auto a_block_space =
        param.gemmKPerBlock *
        integer_least_multiple(param.gemmMPerBlock, max_lds_align);
    const auto b_block_space =
        param.gemmKPerBlock *
        integer_least_multiple(param.gemmNPerBlock, max_lds_align);

    lds_size = 2 * (a_block_space + b_block_space) * sizeof(float);

    return std::make_tuple(lds_size, true);
  }

  bool isValidXDLOPSGemm(TunableParams &param) {
    // TBD: support fp16/bf16
    const auto gemmKPackedPerBlock = param.gemmKPerBlock;

    // unsupported xdlops-gemm
    if (param.gemmMPerWave == 16 && param.gemmNPerWave == 32)
      return false;
    if (param.gemmMPerWave == 32 && param.gemmNPerWave == 16)
      return false;
    if (param.gemmMPerWave == 8 && param.gemmNPerWave != 64)
      return false;
    if (param.gemmMPerWave == 4 && param.gemmNPerWave != 64)
      return false;
    if (param.gemmMPerWave == 32 && param.gemmNPerWave == 32 &&
        gemmKPackedPerBlock % 2 != 0)
      return false;
    if (param.gemmMPerWave == 16 && param.gemmNPerWave == 16 &&
        gemmKPackedPerBlock % 4 != 0)
      return false;

    const auto waveSize  = 64;
    const auto blockSize = param.gemmNPerBlock * param.gemmMPerBlock /
                           (param.gemmMPerWave * param.gemmNPerWave) * waveSize;

    // fail with blockSize >= 512
    /// \todo fix the issue with blockSize >= 512
    if(blockSize < 64 || blockSize > 256)
        return false;

    return (param.gemmMPerBlock % param.gemmMPerWave) == 0 &&
           (param.gemmNPerBlock % param.gemmNPerWave) == 0;
  }

  // TBD review logic here for various layouts.
  bool isValidParameter(TunableParams &param) {
    int64_t gemmM = ctx.dimIndexVal["k"].second;
    int64_t gemmN = ctx.dimIndexVal["no"].second *
                    ctx.dimIndexVal["ho"].second * ctx.dimIndexVal["wo"].second;
    int64_t gemmK = ctx.dimIndexVal["c"].second * ctx.dimIndexVal["y"].second *
                    ctx.dimIndexVal["x"].second;

    // llvm::errs() << "gemmM: " << gemmM << " gemmN: " << gemmN << " gemmK: "
    // << gemmK << "\n"; llvm::errs() << "MPerBlock: " << param.gemmMPerBlock <<
    // "\n"; 
    // llvm::errs() << "NPerBlock: " << param.gemmNPerBlock << "\n";
    // llvm::errs() << "KPerBlock: " << param.gemmKPerBlock << "\n";
    // llvm::errs() << "MPerWave: " << param.gemmMPerWave << "\n";
    // llvm::errs() << "NPerWave: " << param.gemmNPerWave << "\n";

    if (!(gemmM % param.gemmMPerBlock == 0 &&
          gemmN % param.gemmNPerBlock == 0 &&
          gemmK % param.gemmKPerBlock == 0)) {
      //llvm::errs() << "NOT VALID\n";
      return false;
    }

    if (!isValidXDLOPSGemm(param)) {
      //llvm::errs() << "NOT VALID\n";
      return false;
    }

    bool valid = false;

    // check blockwise copy of A matrix
    std::tie(std::ignore, std::ignore, std::ignore, std::ignore, valid) =
        calculateGemmABlockCopyPerformanceParameters(param);

    if(!valid) {
      //llvm::errs() << "NOT VALID\n";
      return false;
    }

    // check blockwise copy of B matrix
    std::tie(std::ignore, std::ignore, std::ignore, std::ignore, valid) =
        calculateGemmBBlockCopyPerformanceParameters(param);

    if(!valid) {
      //llvm::errs() << "NOT VALID\n";
      return false;
    }

    std::size_t lds_size = 0;
    std::tie(lds_size, valid) = calculateLdsNumberOfByte(param);

    if (!valid || (lds_size > 64 * 1024)) {
      //llvm::errs() << "NOT VALID\n";
      return false;
    }

    //llvm::errs() << "VALID WITH LDS SIZE: " << lds_size << "\n";
    return (valid && lds_size <= 64 * 1024);
  }

  void customInit() override {
    // Check the following initial tuning parameters and find the valid one.
    llvm::SmallVector<TunableParams, 10> initParameters = {
        // M/block N/block K/block M/wave N/wave
        {128, 128, 16, 64, 64},
        {8, 64, 8, 8, 64},
        {4, 64, 16, 4, 64},
        {16, 16, 4, 16, 16},
    };

    bool foundValidParameters = false;
    TunableParams validParams;
    for (auto &param : initParameters) {
      if (isValidParameter(param)) {
        foundValidParameters = true;
        validParams = param;
        break;
      }
    }

    if (!foundValidParameters) {
      llvm::errs() << "FATAL ERROR! COULD NOT FIND VALID TUNING PARAMETERS!";
    }

    // parameters truly tunable.
    params["CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK"] = validParams.gemmMPerBlock;
    params["CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK"] = validParams.gemmNPerBlock;
    params["CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK"] = validParams.gemmKPerBlock;
    params["CK_PARAM_GEMM_M_PER_WAVE"] = validParams.gemmMPerWave;
    params["CK_PARAM_GEMM_N_PER_WAVE"] = validParams.gemmNPerWave;

    // parameters derivable from tunable parameters.
    const auto waveSize = 64;
    params["CK_PARAM_TUNABLE_BLOCK_SIZE"] =
        validParams.gemmMPerBlock * validParams.gemmNPerBlock /
        (validParams.gemmMPerWave * validParams.gemmNPerWave) * waveSize;

    int gemmABlockCopyClusterLengths_GemmK  = 0;
    int gemmABlockCopyClusterLengths_GemmM  = 0;
    int gemmABlockCopySrcDataPerRead_GemmK  = 0;
    int gemmABlockCopyDstDataPerWrite_GemmM = 0;
    int gemmBBlockCopyClusterLengths_GemmK  = 0;
    int gemmBBlockCopyClusterLengths_GemmN  = 0;
    int gemmBBlockCopySrcDataPerRead_GemmN  = 0;
    int gemmBBlockCopyDstDataPerWrite_GemmN = 0;

    std::tie(gemmABlockCopyClusterLengths_GemmK,
             gemmABlockCopyClusterLengths_GemmM,
             gemmABlockCopySrcDataPerRead_GemmK,
             gemmABlockCopyDstDataPerWrite_GemmM, std::ignore) =
        calculateGemmABlockCopyPerformanceParameters(validParams);

    std::tie(gemmBBlockCopyClusterLengths_GemmK,
             gemmBBlockCopyClusterLengths_GemmN,
             gemmBBlockCopySrcDataPerRead_GemmN,
             gemmBBlockCopyDstDataPerWrite_GemmN, std::ignore) =
        calculateGemmBBlockCopyPerformanceParameters(validParams);

    params["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K"] = gemmABlockCopyClusterLengths_GemmK;
    params["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M"] = gemmABlockCopyClusterLengths_GemmM;
    params["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM"] = gemmABlockCopySrcDataPerRead_GemmK;
    params["CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M"] = gemmABlockCopyDstDataPerWrite_GemmM;

    params["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K"] = gemmBBlockCopyClusterLengths_GemmK;
    params["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N"] = gemmBBlockCopyClusterLengths_GemmN;
    params["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM"] = gemmBBlockCopySrcDataPerRead_GemmN;
    params["CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N"] = gemmBBlockCopyDstDataPerWrite_GemmN;
  }
};

static constexpr StringLiteral kVarName[3] = {"weight", "input", "output"};

static constexpr int kConv2DTensorDimension = 4;

static constexpr StringLiteral kCppPreamblePart1 = R"(
#include "common_header.hpp"
#include "ConstantTensorDescriptor_deprecated.hpp"
)";

static constexpr StringLiteral kCppPreamblePart2 = R"(
#include "float_types.h"

extern "C" __global__
)";

static constexpr StringLiteral kCppPreamblePart3 = R"(
        (const FLOAT* const __restrict__ p_in_global,
        const FLOAT* const __restrict__ p_wei_global,
        FLOAT* const __restrict__ p_out_global)
{
    using namespace ck;

    constexpr index_t ConvStrideH = CK_PARAM_PROBLEM_CONV_STRIDE_H;
    constexpr index_t ConvStrideW = CK_PARAM_PROBLEM_CONV_STRIDE_W;

    constexpr index_t ConvDilationH = CK_PARAM_PROBLEM_CONV_DILATION_H;
    constexpr index_t ConvDilationW = CK_PARAM_PROBLEM_CONV_DILATION_W;

    // read params: tunable params
    constexpr index_t BlockSize = CK_PARAM_TUNABLE_BLOCK_SIZE;

    constexpr index_t GemmMPerBlock = CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK;
    constexpr index_t GemmNPerBlock = CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK;
    constexpr index_t GemmKPerBlock = CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK;

    // read params: dependent params
    constexpr index_t GridSize = CK_PARAM_DEPENDENT_GRID_SIZE;

    constexpr index_t LeftPadH = CK_PARAM_PROBLEM_LEFT_PAD_H;
    constexpr index_t LeftPadW = CK_PARAM_PROBLEM_LEFT_PAD_W;

    constexpr index_t RightPadH = CK_PARAM_PROBLEM_RIGHT_PAD_H;
    constexpr index_t RightPadW = CK_PARAM_PROBLEM_RIGHT_PAD_W;

    using InLeftPads  = Sequence<LeftPadH, LeftPadW>;
    using InRightPads = Sequence<RightPadH, RightPadW>;

)";

static constexpr StringLiteral kCppInterlude = R"(
    using ConvStrides   = Sequence<ConvStrideH, ConvStrideW>;
    using ConvDilations = Sequence<ConvDilationH, ConvDilationW>;

    constexpr index_t GemmBBlockCopyClusterLengths_GemmK =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K;
    constexpr index_t GemmBBlockCopyClusterLengths_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N;

    constexpr index_t GemmBBlockCopyThreadSliceLengths_GemmK =
        GemmKPerBlock / GemmBBlockCopyClusterLengths_GemmK;
    constexpr index_t GemmBBlockCopyThreadSliceLengths_GemmN =
        GemmNPerBlock / GemmBBlockCopyClusterLengths_GemmN;

    constexpr index_t GemmABlockCopyClusterLengths_GemmK =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K;
    constexpr index_t GemmABlockCopyClusterLengths_GemmM =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M;

    constexpr index_t GemmABlockCopyThreadSliceLengths_GemmK =
        GemmKPerBlock / GemmABlockCopyClusterLengths_GemmK;
    constexpr index_t GemmABlockCopyThreadSliceLengths_GemmM =
        GemmMPerBlock / GemmABlockCopyClusterLengths_GemmM;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN =
        Sequence<GemmBBlockCopyThreadSliceLengths_GemmK, GemmBBlockCopyThreadSliceLengths_GemmN>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN =
        Sequence<GemmBBlockCopyClusterLengths_GemmK, GemmBBlockCopyClusterLengths_GemmN>;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM =
        Sequence<GemmABlockCopyThreadSliceLengths_GemmK, GemmABlockCopyThreadSliceLengths_GemmM>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM =
        Sequence<GemmABlockCopyClusterLengths_GemmK, GemmABlockCopyClusterLengths_GemmM>;

    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N;
    constexpr index_t GemmABlockCopySrcDataPerRead_GemmM =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM;
    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM;

    constexpr auto GemmMPerWave                   = CK_PARAM_GEMM_M_PER_WAVE;
    constexpr auto GemmNPerWave                   = CK_PARAM_GEMM_N_PER_WAVE;
    constexpr index_t ThreadGemmDataPerRead_GemmM = 1;
    constexpr index_t ThreadGemmDataPerRead_GemmN = 1;
)";

static constexpr StringLiteral kCppEpiloguePart1 = R"(
            <GridSize,
            BlockSize,
            FLOAT,
            FLOAT_ACCUM,
)";

static constexpr StringLiteral kCppEpiloguePart2 =R"(
            ConvStrides,
            ConvDilations,
            InLeftPads,
            InRightPads,
            GemmMPerBlock,
            GemmNPerBlock,
            GemmKPerBlock,
            GemmMPerWave,
            GemmNPerWave,
            ThreadGemmDataPerRead_GemmM,
            ThreadGemmDataPerRead_GemmN,
            GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
            GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
            GemmABlockCopySrcDataPerRead_GemmK,
            GemmABlockCopySrcDataPerRead_GemmM,
            GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
            GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
            GemmBBlockCopySrcDataPerRead_GemmN,
            GemmBBlockCopyDstDataPerWrite_GemmN>{};

    gridwise_conv.Run(p_in_global, p_wei_global, p_out_global);
}
)";
 
void EmitCppPreamble(llvm::raw_ostream &output, llvm::StringRef layoutStr) {
  output << kCppPreamblePart1;
// Between Preamble Part 1 and Part 2:
// #include "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_nchw_kcyx_nkhw_lds_double_buffer.hpp"
  output << R"(#include "gridwise_convolution_implicit_gemm_v4r4_)";

  // Change to fixed "mlir".
  //output << layoutStr << R"(.hpp")";
  output << "mlir" << R"(.hpp")";

  output << kCppPreamblePart2;
// Between Preamble Part 2 and Par 3:
//    __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void gridwise_convolution_implicit_gemm_v4r4_nchw_kcyx_nkhw(
  output << R"(
    __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void gridwise_convolution_implicit_gemm_v4r4_)";
  // Change to fixed "mlir".
  //output << layoutStr;
  output << "mlir";

  output << kCppPreamblePart3;
}

void EmitCppInterlude(llvm::raw_ostream &output) {
  output << kCppInterlude;
}

void EmitCppEpilogue(llvm::raw_ostream &output, llvm::StringRef layoutStr, llvm::SmallVector<std::string, 3> tensorDescs) {
// Before Part1:
//    constexpr auto gridwise_conv = GridwiseConvolutionImplicitGemm_v4r4_nchw_kcyx_nkhw
  output << R"(
    constexpr auto gridwise_conv = GridwiseConvolutionImplicitGemm_v4r4_)";

  // Change to fixed "mlir".
  //output << layoutStr;
  output << "mlir";

  output << kCppEpiloguePart1;
// Between Part1 and Part2:
//        decltype(in_nchw_desc),
//        decltype(wei_kcyx_desc),
//        decltype(out_nkhw_desc),
  output << "            decltype(" << tensorDescs[1] << "),\n";
  output << "            decltype(" << tensorDescs[0] << "),\n";
  output << "            decltype(" << tensorDescs[2] << "),\n";
  output << kCppEpiloguePart2;
}

static constexpr StringLiteral kHeaderPreamblePart1 = R"(
#ifndef CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R4_HPP
#define CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R4_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "blockwise_generic_tensor_slice_copy.hpp"
#include "threadwise_generic_tensor_slice_copy.hpp"
#include "gridwise_gemm_xdlops.hpp"
#include "convolution_common.hpp"
#include "implicitgemm_params.hpp"

namespace ck {

// B = merge(N, Ho, Wo)
template <index_t GridSize,
          index_t BlockSize,
          class Float,
          class AccDataType,
          class InGlobalDesc,
          class WeiGlobalDesc,
          class OutGlobalDesc,
          class ConvStrides,
          class ConvDilations,
          class InLeftPads,
          class InRightPads,
          index_t GemmMPerBlock,
          index_t GemmNPerBlock,
          index_t GemmKPerBlock,
          index_t GemmMPerWave,
          index_t GemmNPerWave,
          index_t GemmThreadGemmDataPerReadM,
          index_t GemmThreadGemmDataPerReadN,
          class GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
          class GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
          index_t GemmABlockCopySrcDataPerRead_GemmK,
          index_t GemmABlockCopySrcDataPerRead_GemmM,
          class GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
          class GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
          index_t GemmBBlockCopySrcDataPerRead_GemmN,
          index_t GemmBBlockCopyDstDataPerWrite_GemmN>
)";

static constexpr StringLiteral kHeaderPreamblePart2 = R"(
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
    {
)";

static constexpr StringLiteral kHeaderPreamblePart3 = R"(
        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

)";

static constexpr StringLiteral kHeaderEpiloguePart1 = R"(
        // GEMM
        constexpr auto gridwise_gemm = GridwiseGemmTransposedANormalBNormalCXdlops_v1<
            GridSize,
            BlockSize,
            Float,
            AccDataType,
)";

static constexpr StringLiteral kHeaderEpiloguePart2 = R"(
            GemmMPerBlock,
            GemmNPerBlock,
            GemmKPerBlock,
            GemmMPerWave,
            GemmNPerWave,
            GemmThreadGemmDataPerReadM,
            GemmThreadGemmDataPerReadN,
            GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
            GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
            Sequence<1, 0>,
            Sequence<1, 0>,
            Sequence<0, 1>,
)";

static constexpr StringLiteral kHeaderEpiloguePart3 = R"(
            GemmABlockCopySrcDataPerRead_GemmK,
            GemmABlockCopySrcDataPerRead_GemmM,
            GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
            GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
            Sequence<0, 1>,
            Sequence<0, 1>,
            Sequence<0, 1>,
)";

static constexpr StringLiteral kHeaderEpiloguePart4 = R"(
            GemmBBlockCopySrcDataPerRead_GemmN,
            GemmBBlockCopyDstDataPerWrite_GemmN,
            InMemoryDataOperation::Set>{};

        gridwise_gemm.Run(p_wei_global, p_in_global, p_out_global);
    }
};

} // namespace ck
#endif
)";

void EmitHeaderPreamble(llvm::raw_ostream &output, llvm::StringRef layoutStr, llvm::SmallVector<std::string, 3> &tensorDescs) {
  output << kHeaderPreamblePart1;
  output << R"(
struct GridwiseConvolutionImplicitGemm_v4r4_)";

  // Change to fixed "mlir".
  //output << layoutStr;
  output << "mlir";

  output << kHeaderPreamblePart2;
  output << kHeaderPreamblePart3;
  output << '\n';

  output << R"(
        constexpr auto )" << tensorDescs[0] << " = WeiGlobalDesc{};";
  output << R"(
        constexpr auto )" << tensorDescs[1] << " = InGlobalDesc{};";
  output << R"(
        constexpr auto )" << tensorDescs[2] << " = OutGlobalDesc{};";
  output << '\n';
}

void EmitHeaderEpilogue(llvm::raw_ostream &output, llvm::SmallDenseMap<int64_t, std::string> &args, bool filterGemmKVectorizable, bool inputGemmKVectorizable) {
  output << kHeaderEpiloguePart1;
// Between Part1 and Part2 emit:
//                                                   decltype(wei_e_k_global_desc),
//                                                   decltype(in_e_b_global_desc),
//                                                   decltype(out_k_b_global_desc),
  for (unsigned i = 0; i < args.size(); ++i) {
    output << R"(
            decltype()" << args[i] << "),";
  }
  output << kHeaderEpiloguePart2;

// Between Part2 and Part3 emit which dimension the vectorization takes place for filter tensor.
// kcyx, kyxc, yxkc, ckyx: 0
// yxck, cyxk: 1
  if (filterGemmKVectorizable) {
    output << "            0,";
  } else {
    output << "            1,";
  }
  output << kHeaderEpiloguePart3;
// Between Part3 and Part4 emit which dimension the vectorization takes place for input tensor.
// nhwc, hwnc: 0
// chwn, hwcn: 1
// nchw, cnhw: non-vectorizable for now, set to 0, with vectorization width to 1.
  if (inputGemmKVectorizable) {
    output << "            0,";
  } else {
    output << "            1,";
  }
  output << kHeaderEpiloguePart4;
}

void EmitLayoutString(llvm::raw_ostream &output, llvm::ArrayRef<mlir::Attribute> &layoutArrayAttr, llvm::StringRef prefix, llvm::StringRef suffix, llvm::StringRef delimiter = "") {
  for (int i = 0; i < kConv2DTensorDimension; ++i) {
    auto attr = layoutArrayAttr[i];
    if (auto strAttr = attr.dyn_cast<StringAttr>()) {
      output << prefix << strAttr.getValue() << suffix;
    }
    if (i < kConv2DTensorDimension - 1) {
      output << delimiter;
    }
  }
}

void EmitHeaderDimensionLengths(llvm::raw_ostream &output, llvm::ArrayRef<mlir::Attribute> &layoutArrayAttr, llvm::StringRef tensorDesc) {
  for (int i = 0; i < kConv2DTensorDimension; ++i) {
    auto attr = layoutArrayAttr[i];
    if (auto strAttr = attr.dyn_cast<StringAttr>()) {
      output << "        constexpr index_t " << strAttr.getValue() << " = " << tensorDesc << ".GetLengths()[" << i << "];\n";
    }
  }
}

void EmitDimensionVariables(llvm::raw_ostream &output, llvm::ArrayRef<mlir::Attribute> &layoutArrayAttr) {
  for (int i = 0; i < kConv2DTensorDimension; ++i) {
    auto attr = layoutArrayAttr[i];
    if (auto strAttr = attr.dyn_cast<StringAttr>()) {
      output << "    constexpr index_t " << strAttr.getValue() << " = CK_PARAM_PROBLEM_";

      switch (llvm::toUpper(strAttr.getValue()[0])) {
          case 'H':
          case 'W':
            output << llvm::toUpper(strAttr.getValue()[0]);
            // XXX: fix this. 
            if (strAttr.getValue().size() > 1)
              output << llvm::toUpper(strAttr.getValue()[1]);
            break;
          default:
            output << llvm::toUpper(strAttr.getValue()[0]);
      }
      output << ";\n";
    }
  }
}

void EmitStrideVariables(llvm::raw_ostream &output, llvm::ArrayRef<mlir::Attribute> &layoutArrayAttr) {
  for (int i = kConv2DTensorDimension - 1; i >= 0; --i) {
    auto attr = layoutArrayAttr[i];
    if (auto strAttr = attr.dyn_cast<StringAttr>()) {
      output << "    constexpr index_t stride_" << strAttr.getValue() << " = ";

      if (i == kConv2DTensorDimension - 1) {
        output << "1;\n";
      } else {
        auto prevAttr = layoutArrayAttr[i + 1];
        if (auto strPrevAttr = prevAttr.dyn_cast<StringAttr>()) {
          output << strPrevAttr.getValue() << " * stride_" << strPrevAttr.getValue() << ";\n";
        }
      }
    }
  }
}

template<typename T>
void EmitInterleaveArrayAttrWithSeparator(llvm::raw_ostream &os, mlir::ArrayAttr &arrayAttr, const StringRef &separator) {
  if (arrayAttr) {
    interleave(arrayAttr, os, [&](Attribute attr) {
      if (auto typedAttr = attr.dyn_cast<T>())
        os << typedAttr.getValue();
    }, separator);
  }
}

template<typename T>
void EmitInterleaveCommaArrayAttr(llvm::raw_ostream &os, mlir::ArrayAttr &arrayAttr) {
  EmitInterleaveArrayAttrWithSeparator<T>(os, arrayAttr, ", ");
}

template<typename T>
void EmitInterleaveAsteriskArrayAttr(llvm::raw_ostream &os, mlir::ArrayAttr &arrayAttr) {
  EmitInterleaveArrayAttrWithSeparator<T>(os, arrayAttr, " * ");
}


void ObtainModuleInfo(ModuleOp &m, std::string &layoutStr, llvm::SmallVector<std::string, 3> &tensorDescs) {
  // (TBD verifiying logic) The Module could contain multiple FuncOp, and inside each FuncOp there
  // should be exactly:
  // - 3 input arguments
  // - 1 result.
  //
  // - 0 conv2d op.
  // - 5 transform ops (1 for filter, 3 for input, 1 for output).
  // - 1 gridwise gemm op.

  // Enumerate FuncOp instances inside the ModuleOp.
  for (auto f : m.getOps<FuncOp>()) {
    int srcLayoutAttrCtr = 0;
    llvm::raw_string_ostream los(layoutStr);

    // First iteration. Construct tensor descriptor names.
    f.walk([&srcLayoutAttrCtr, &tensorDescs, &los](miopen::TransformOp op) {
      // get source_layout attribute.
      auto srcLayoutAttr = op.getAttrOfType<ArrayAttr>("source_layout");
      if (srcLayoutAttr) {
        auto srcLayout = srcLayoutAttr.getValue();

        // Prepare tensor descriptor variable name.
        std::string desc{};
        llvm::raw_string_ostream os(desc);
        os << kVarName[srcLayoutAttrCtr++] << "_";
        EmitLayoutString(os, srcLayout, "", "", "_");
        os << "_desc";
        os.flush();
        tensorDescs.push_back(desc);

        // Prepare layout string.
        if (srcLayoutAttrCtr != 1)
          los << "_";
        EmitLayoutString(los, srcLayout, "", "");
      }
    });
    los.flush();
  }
}

void populateDimVal(const ArrayAttr &layoutAttr, const ArrayAttr &dimAttr,
                    llvm::StringMap<std::pair<size_t, int64_t>> &dimIndexVal) {
  assert(layoutAttr.size() == dimAttr.size());
  size_t dimValSize = layoutAttr.size();
  for (size_t i = 0; i < dimValSize; ++i) {
    auto key = layoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();
    auto value = dimAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt();
    dimIndexVal[key] = std::make_pair(i, value);
  }
}

void populateSeqVal(const ArrayAttr &seqAttr,
                    llvm::SmallVector<int64_t, 0> &seqVal) {
  size_t seqValSize = seqAttr.size();
  for (size_t i = 0; i < seqValSize; ++i) {
    // Not nested array, push back the value and be done
    if (seqAttr.getValue()[i].dyn_cast<ArrayAttr>() == nullptr) {
      seqVal.push_back(seqAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt());
      continue;
    }
    // There is nested values, continue to populate those
    for (size_t j = 0; j < seqAttr.getValue()[i].dyn_cast<ArrayAttr>().size();
         ++j) {
      seqVal.push_back(seqAttr.getValue()[i]
                           .dyn_cast<ArrayAttr>()
                           .getValue()[j]
                           .dyn_cast<IntegerAttr>()
                           .getInt());
    }
  }
}

} // anontmous namespace

std::unique_ptr<llvm::StringRef> mlir::translateModuleToMIOpenHeaderXDLOPS(ModuleOp m) {
  llvm::raw_string_ostream output(resultStr);

  // Enumerate FuncOp instances inside the ModuleOp.
  for (auto f : m.getOps<FuncOp>()) {
    std::string layoutStr;
    llvm::SmallVector<std::string, 3> tensorDescs;
    llvm::SmallDenseMap<int64_t, std::string> gridwiseGemmArguments;

    // Obtain critical information from ModuleOp.
    ObtainModuleInfo(m, layoutStr, tensorDescs);

    int srcLayoutAttrCtr = 0;

    // Start emitting.
    EmitHeaderPreamble(output, layoutStr, tensorDescs);

    // First iteration. Output source dimensions.
    f.walk([&output, &srcLayoutAttrCtr, &tensorDescs](miopen::TransformOp op) {
      // get source_layout attribute.
      auto srcLayoutAttr = op.getAttrOfType<ArrayAttr>("source_layout");
      if (srcLayoutAttr) {
        auto srcLayout = srcLayoutAttr.getValue();
        output << "\n        // ";
        EmitLayoutString(output, srcLayout, "", "", ", ");
        output << '\n';

        EmitHeaderDimensionLengths(output, srcLayout, tensorDescs[srcLayoutAttrCtr++]);
      }
    });
    output << '\n';
 
    srcLayoutAttrCtr = 0;
    // Second iteration. Output the rest.
    f.walk([&output, &srcLayoutAttrCtr, &tensorDescs, &gridwiseGemmArguments](miopen::TransformOp op) {
      // get source_layout attribute.
      auto srcLayoutAttr = op.getAttrOfType<ArrayAttr>("source_layout");

      // get layout attribute.
      auto layoutAttr = op.getAttrOfType<ArrayAttr>("layout");
      std::string inputTensorName;
      std::string transformedInputTensorName;
      std::string outputTensorName;
      std::string operationSpec;
      std::string srcDimSpec;
      std::string dstDimSpec;
      llvm::raw_string_ostream ins(inputTensorName);
      llvm::raw_string_ostream pins(transformedInputTensorName);
      llvm::raw_string_ostream outs(outputTensorName);
      llvm::raw_string_ostream ops(operationSpec);
      llvm::raw_string_ostream srcs(srcDimSpec);
      llvm::raw_string_ostream dsts(dstDimSpec);

      // determine input and output tensor name.
      auto immLayoutAttr = op.getAttrOfType<ArrayAttr>("intermediate_layout");
      auto outputLayoutAttr = op.getAttrOfType<ArrayAttr>("output_layout");
      if (srcLayoutAttr) {
        inputTensorName = tensorDescs[srcLayoutAttrCtr];
        outs << kVarName[srcLayoutAttrCtr] << "_";

        srcLayoutAttrCtr++;
      } else {
        // get intermediate_layout attribute.
        if (immLayoutAttr) {
          ins << kVarName[srcLayoutAttrCtr - 1] << "_";
          EmitInterleaveArrayAttrWithSeparator<StringAttr>(ins, immLayoutAttr, "_");
          ins << "_desc";
          ins.flush();

          outs << kVarName[srcLayoutAttrCtr - 1] << "_";
        }
      }
      EmitInterleaveArrayAttrWithSeparator<StringAttr>(outs, outputLayoutAttr, "_");
      outs << "_desc";
      outs.flush();

      // determine gridwise GEMM arguments.
      auto gridwiseGemmArgPosAttr = op.getAttrOfType<IntegerAttr>("gridwise_gemm_argument_position");
      if (gridwiseGemmArgPosAttr) {
        gridwiseGemmArguments[gridwiseGemmArgPosAttr.getInt()] = outputTensorName;
      }  

      ops << "            make_tuple(";
      srcs << "            make_tuple(";
      dsts << "            make_tuple(";

      // XXX see if we can get better than this.
      int convDilationCtr = 0;
      bool hasUnfoldTransform = false;

      for (auto layoutSpec = layoutAttr.begin(); layoutSpec != layoutAttr.end(); ) {
        if (auto layoutSpecDict = layoutSpec->dyn_cast<DictionaryAttr>()) {
          auto srcNames = layoutSpecDict.get("source_names").dyn_cast<ArrayAttr>();
          auto dstNames = layoutSpecDict.get("names").dyn_cast<ArrayAttr>();
          auto srcDims = layoutSpecDict.get("source_dimensions").dyn_cast<ArrayAttr>();
          auto dstDims = layoutSpecDict.get("dimensions").dyn_cast<ArrayAttr>();

          if (auto transform = layoutSpecDict.get("transformation").dyn_cast<StringAttr>()) {
            if (transform.getValue() == "PassThrough") {
              ops << transform.getValue() << "<";
              EmitInterleaveCommaArrayAttr<StringAttr>(ops, srcNames);
              ops << ">{}";
            } else if (transform.getValue() == "Merge") {
              ops << transform.getValue() << "<"
                  << "Sequence<";
              EmitInterleaveCommaArrayAttr<StringAttr>(ops, srcNames);
              ops << ">" << ">{}";
            } else if (transform.getValue() == "Pad") {
              ops << transform.getValue() << "<"
                  << "Sequence<";
              EmitInterleaveCommaArrayAttr<StringAttr>(ops, srcNames);
              ops << ">, InLeftPads, InRightPads" << ">{}";
            } else if (transform.getValue() == "Embed") {
              ops << transform.getValue() << "<"
                  << inputTensorName << ".GetLengths()[" << srcDims.getValue()[0].dyn_cast<IntegerAttr>().getInt() << "], "
                  << "Sequence<";
              EmitInterleaveCommaArrayAttr<StringAttr>(ops, dstNames);
              if (convDilationCtr == 0) {
                ops << ">, Sequence<ConvDilationH, ConvDilationH, 0>>{}";
                convDilationCtr++;
              } else {
                ops << ">, Sequence<ConvDilationW, ConvDilationW, 0>>{}";
              }
            } else if (transform.getValue() == "Unfold") {
              hasUnfoldTransform = true;
              ops << "PassThrough<";
              EmitInterleaveAsteriskArrayAttr<StringAttr>(ops, srcNames);
              ops << ">{}";
            }
            srcs << "Sequence<";
            if (transform.getValue() == "Unfold") {
              pins << "unfold_tensor_descriptor(" << inputTensorName << ", "
                   << "Number<" << srcDims.getValue()[0].dyn_cast<IntegerAttr>().getInt() << ">{}, "
                   << "Number<" << srcDims.getValue()[srcDims.size() - 1].dyn_cast<IntegerAttr>().getInt() << ">{})";
              pins.flush();
              srcs << srcDims.getValue()[0].dyn_cast<IntegerAttr>().getInt();
            } else {
              if (hasUnfoldTransform) {
                // XXX see if we can do better than this.
                if (srcDims.getValue()[0].dyn_cast<IntegerAttr>().getInt() == 0) {
                  srcs << "0";
                } else {
                  srcs << "1";
                }
              } else {
                EmitInterleaveCommaArrayAttr<IntegerAttr>(srcs, srcDims);
              }
            }
            srcs << ">{}";
            dsts << "Sequence<";
            EmitInterleaveCommaArrayAttr<IntegerAttr>(dsts, dstDims);
            dsts << ">{}";
          }
        }

        ++layoutSpec;
        if (layoutSpec != layoutAttr.end()) {
          ops << ", ";
          srcs << ", ";
          dsts << ", ";
        }
      }
      ops << "),\n";
      ops.flush();
      srcs << "),\n";
      srcs.flush();
      dsts << ")";
      dsts.flush();

      output << "        constexpr auto " << outputTensorName << " = transform_tensor_descriptor(\n";
      if (hasUnfoldTransform) {
        output << "            " << transformedInputTensorName << ",\n";
      } else {
        output << "            " << inputTensorName << ",\n";
      }
      output << operationSpec << srcDimSpec << dstDimSpec;
      output << ");\n\n";
    });

    bool filterGemmKVectorizable = false, inputGemmKVectorizable = false;
    f.walk([&filterGemmKVectorizable, &inputGemmKVectorizable](miopen::GridwiseGemmOp op) {
      auto filterLayoutAttr = op.getAttrOfType<ArrayAttr>("filter_layout");
      auto inputLayoutAttr = op.getAttrOfType<ArrayAttr>("input_layout");

      size_t dimKF, dimCF, dimYF, dimXF;
      size_t dimNI, dimCI, dimHI, dimWI;

      for (size_t i = 0; i < 4; ++i) {
        auto filterDim = filterLayoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();

        if (filterDim.str() == "k") {
          dimKF = i;
        } else if (filterDim.str() == "c") {
          dimCF = i;
        } else if (filterDim.str() == "y") {
          dimYF = i;
        } else if (filterDim.str() == "x") {
          dimXF = i;
        }

        auto inputDim = inputLayoutAttr.getValue()[i].dyn_cast<StringAttr>().getValue();
        if (inputDim.str() == "ni") {
          dimNI = i;
        } else if (inputDim.str() == "ci") {
          dimCI = i;
        } else if (inputDim.str() == "hi") {
          dimHI = i;
        } else if (inputDim.str() == "wi") {
          dimWI = i;
        }
      }

      // Filter tensor.
      // Find the fastest changing dimension.
      if (dimKF == 3) {
        // When K is the fastest changing dimension,
        // gemmM dimension is vectorizable.
        // vectorization width depending on length of K.

        // gemmK dimension non-vectorizable.
        filterGemmKVectorizable = false;
      } else {
        // gemmK dimension vectorizable,
        // depending on which among C, Y, X be the fastest changing dimension.
        filterGemmKVectorizable = true;
        // gemmM dimension non-vectorizable.
      }

      // Input tensor.
      // Find the fastest changing dimension.
      if (dimNI == 3) {
        // When N is the fastest changing dimension,
        // gemmN dimension is vectorizable.
        // vectorization width depending on length of N.

        // gemmK dimension non-vectorizable.
        inputGemmKVectorizable = false;
      } else if (dimCI == 3) {
        // When C is the fastest changing dimension,
        // gemmK dimension vectorizable.
        // vectorization width depending on length of C.
        inputGemmKVectorizable = true;

        // gemmN dimension non-vectorizable.
      }

    });

    EmitHeaderEpilogue(output, gridwiseGemmArguments, filterGemmKVectorizable, inputGemmKVectorizable);
  }

  output.flush();
  return std::make_unique<llvm::StringRef>(resultStr);
}

std::unique_ptr<llvm::StringRef> mlir::translateModuleToMIOpenCppXDLOPS(ModuleOp m) {
  llvm::raw_string_ostream output(resultStr);

  // Enumerate FuncOp instances inside the ModuleOp.
  for (auto f : m.getOps<FuncOp>()) {
    std::string layoutStr;
    llvm::SmallVector<std::string, 3> tensorDescs;

    // Obtain critical information from ModuleOp.
    ObtainModuleInfo(m, layoutStr, tensorDescs);

    int srcLayoutAttrCtr = 0;

    // Start emitting.
    EmitCppPreamble(output, layoutStr);

    f.walk([&output, &srcLayoutAttrCtr, &tensorDescs](miopen::TransformOp op) {
      // get source_layout attribute.
      auto srcLayoutAttr = op.getAttrOfType<ArrayAttr>("source_layout");
      if (srcLayoutAttr) {
        auto srcLayout = srcLayoutAttr.getValue();
        output << "    // ";
        EmitLayoutString(output, srcLayout, "", "", ",");
        output << '\n';

        EmitDimensionVariables(output, srcLayout);
        output << '\n';
        EmitStrideVariables(output, srcLayout);

        output << "    constexpr auto " << tensorDescs[srcLayoutAttrCtr++];
        output << " = make_native_tensor_descriptor(Sequence<";
        EmitLayoutString(output, srcLayout, "", "", ", ");
        output << ">{}, Sequence<";
        EmitLayoutString(output, srcLayout, "stride_", "", ", ");
        output << ">{});\n\n";
      }
    });

    EmitCppInterlude(output);

    EmitCppEpilogue(output, layoutStr, tensorDescs);
  }

  output.flush();
  return std::make_unique<llvm::StringRef>(resultStr);
}

std::unique_ptr<llvm::StringRef> mlir::translateModuleToMIOpenCFlagsXDLOPS(ModuleOp m) {
  llvm::raw_string_ostream output(resultStr);

  for (auto f : m.getOps<FuncOp>()) {
    f.walk([&output](miopen::GridwiseGemmOp op) {
      llvm::StringMap<std::pair<size_t, int64_t>> dimIndexVal;
      // Filter
      auto filterLayoutAttr = op.getAttrOfType<ArrayAttr>("filter_layout");
      auto filterDimensionAttr =
          op.getAttrOfType<ArrayAttr>("filter_dimension");
      populateDimVal(filterLayoutAttr, filterDimensionAttr, dimIndexVal);
      output << " -DCK_PARAM_PROBLEM_K=" << dimIndexVal["k"].second;
      output << " -DCK_PARAM_PROBLEM_C=" << dimIndexVal["c"].second;
      output << " -DCK_PARAM_PROBLEM_Y=" << dimIndexVal["y"].second;
      output << " -DCK_PARAM_PROBLEM_X=" << dimIndexVal["x"].second;
      // Input
      auto inputLayoutAttr = op.getAttrOfType<ArrayAttr>("input_layout");
      auto inputDimensionAttr = op.getAttrOfType<ArrayAttr>("input_dimension");
      populateDimVal(inputLayoutAttr, inputDimensionAttr, dimIndexVal);
      output << " -DCK_PARAM_PROBLEM_N=" << dimIndexVal["ni"].second;
      output << " -DCK_PARAM_PROBLEM_HI=" << dimIndexVal["hi"].second;
      output << " -DCK_PARAM_PROBLEM_WI=" << dimIndexVal["wi"].second;
      // Output
      auto outputLayoutAttr = op.getAttrOfType<ArrayAttr>("output_layout");
      auto outputDimensionAttr = op.getAttrOfType<ArrayAttr>("output_dimension");
      populateDimVal(outputLayoutAttr, outputDimensionAttr, dimIndexVal);
      output << " -DCK_PARAM_PROBLEM_HO=" << dimIndexVal["ho"].second;
      output << " -DCK_PARAM_PROBLEM_WO=" << dimIndexVal["wo"].second;

      // Stride
      auto strideAttr = op.getAttrOfType<ArrayAttr>("strides");
      llvm::SmallVector<int64_t, 0> strideVal;
      populateSeqVal(strideAttr, strideVal);
      output << " -DCK_PARAM_PROBLEM_CONV_STRIDE_H=" << strideVal[0];
      output << " -DCK_PARAM_PROBLEM_CONV_STRIDE_W=" << strideVal[1];

      // Dilation
      auto dilationAttr = op.getAttrOfType<ArrayAttr>("dilations");
      llvm::SmallVector<int64_t, 0> dilationVal;
      populateSeqVal(dilationAttr, dilationVal);
      output << " -DCK_PARAM_PROBLEM_CONV_DILATION_H=" << dilationVal[0];
      output << " -DCK_PARAM_PROBLEM_CONV_DILATION_W=" << dilationVal[1];

      // Padding
      auto paddingAttr = op.getAttrOfType<ArrayAttr>("padding");
      llvm::SmallVector<int64_t, 0> paddingVal;
      populateSeqVal(paddingAttr, paddingVal);
      output << " -DCK_PARAM_PROBLEM_LEFT_PAD_H=" << paddingVal[0];
      output << " -DCK_PARAM_PROBLEM_LEFT_PAD_W=" << paddingVal[1];
      output << " -DCK_PARAM_PROBLEM_RIGHT_PAD_H=" << paddingVal[2];
      output << " -DCK_PARAM_PROBLEM_RIGHT_PAD_W=" << paddingVal[3];

      // TBD: be able to set data type.
      output << " -DMIOPEN_USE_FP32=1 -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_BFP16=0";

      // TBD: be able to set convolution direction.
      output << " -DCK_PARAM_PROBLEM_CONV_DIRECTION_FORWARD=1";
      output << " -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_DATA=0";
      output << " -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_WEIGHT=0";

      // distinguish between:
      // - parameters truly need to be tuned.
      // - parameters deducible via transformations.
      // - parameters which have heuristic-based values.
      // - parameters which are related to code generation.

      ConvolutionContext convContext{dimIndexVal, strideVal, dilationVal,
                                     paddingVal};
      TunableParameters params;
      params.setContext(convContext);
      params.init();

      // XXX disable for now.
      //// Input tensor.
      // bool inputGemmKVectorizable = false;
      // vectorizableLength = 0;
      //// Find the fastest changing dimension.
      // if (ctx.dimIndexVal["ni"].first == 3) {
      //  // When N is the fastest changing dimension,
      //  // gemmN dimension is vectorizable.
      //  // vectorization width depending on length of N.
      //  vectorizableLength = ctx.dimIndexVal["ni"].second;

      //  // gemmK dimension non-vectorizable.
      //  inputGemmKVectorizable = false;
      //} else if (ctx.dimIndexVal["ci"].first == 3) {
      //  // When C is the fastest changing dimension,
      //  // gemmK dimension vectorizable.
      //  // vectorization width depending on length of C.
      //  vectorizableLength = ctx.dimIndexVal["c"].second;

      //  inputGemmKVectorizable = true;
      //  // gemmN dimension non-vectorizable.
      //}

      // Print out the tunable parameters.
      params.print(output);
      if (IsPopulateTunableParameters.getValue()) {
        // Populate YAML config file.
        params.dump();
      }

      // Emit parameters derived from tunable parameters.
      int64_t gemmMPerBlock = params["CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK"];
      int64_t gemmNPerBlock = params["CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK"];
      int64_t gemmM = dimIndexVal["k"].second;
      int64_t gemmN = dimIndexVal["no"].second * dimIndexVal["ho"].second *
                      dimIndexVal["wo"].second;
      int64_t gridSize = (gemmM / gemmMPerBlock) * (gemmN / gemmNPerBlock);
      output << " -DCK_PARAM_DEPENDENT_GRID_SIZE=" << gridSize;

      // Emit code-gen related parameters.
      output << " -DCK_USE_AMD_XDLOPS=1";
      output << " -DCK_USE_AMD_XDLOPS_INLINE_ASM=1";
      output << " -std=c++14";
      output << " -D__HIP_PLATFORM_HCC__=1";
    });
  }

  output.flush();
  return std::make_unique<llvm::StringRef>(resultStr);
}

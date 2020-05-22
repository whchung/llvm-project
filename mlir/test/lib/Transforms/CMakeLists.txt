if (MLIR_CUDA_CONVERSIONS_ENABLED)
  set(NVPTX_LIBS
    LLVMMC
    LLVMNVPTXCodeGen
    LLVMNVPTXDesc
    LLVMNVPTXInfo
    MLIRNVVMIR
    MLIRTargetNVVMIR
  )
endif()

if (MLIR_ROCM_CONVERSIONS_ENABLED)
  set(AMDGPU_LIBS
    LLVMMC
    LLVMAMDGPUCodeGen
    LLVMAMDGPUDesc
    LLVMAMDGPUInfo
    MLIRROCDLIR
    MLIRTargetROCDLIR
  )
endif()

# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestTransforms
  TestAllReduceLowering.cpp
  TestBufferPlacement.cpp
  TestCallGraph.cpp
  TestConstantFold.cpp
  TestConvertGPUKernelToCubin.cpp
  TestConvertGPUKernelToHsaco.cpp
  TestDominance.cpp
  TestLoopFusion.cpp
  TestGpuMemoryPromotion.cpp
  TestGpuParallelLoopMapping.cpp
  TestInlining.cpp
  TestLinalgTransforms.cpp
  TestLiveness.cpp
  TestLoopMapping.cpp
  TestLoopParametricTiling.cpp
  TestLoopUnrolling.cpp
  TestOpaqueLoc.cpp
  TestMemRefBoundCheck.cpp
  TestMemRefDependenceCheck.cpp
  TestMemRefStrideCalculation.cpp
  TestVectorToSCFConversion.cpp
  TestVectorTransforms.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRStandardOpsIncGen
  MLIRTestVectorTransformPatternsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineOps
  MLIRAnalysis
  MLIREDSC
  MLIRGPU
  MLIRGPUtoGPURuntimeTransforms
  MLIRLinalgOps
  MLIRLinalgTransforms
  MLIRSCF
  MLIRGPU
  MLIRPass
  MLIRStandardOpsTransforms
  MLIRTestDialect
  MLIRTransformUtils
  MLIRVectorToSCF
  MLIRVector
  ${AMDGPU_LIBS}
  ${NVPTX_LIBS}
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../DeclarativeTransforms)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../DeclarativeTransforms)

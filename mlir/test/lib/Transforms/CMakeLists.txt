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
  LLVMMC
  LLVMAMDGPUCodeGen
  LLVMAMDGPUDesc
  LLVMAMDGPUInfo
  MLIRAffineOps
  MLIRAnalysis
  MLIREDSC
  MLIRGPU
  MLIRGPUtoROCmTransforms
  MLIRGPUtoGPURuntimeTransforms
  MLIRLinalgOps
  MLIRLinalgTransforms
  MLIRSCF
  MLIRGPU
  MLIRPass
  MLIRROCDLIR
  MLIRStandardOpsTransforms
  MLIRTargetROCDLIR
  MLIRTestDialect
  MLIRTransformUtils
  MLIRVectorToSCF
  MLIRVector
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../DeclarativeTransforms)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../DeclarativeTransforms)

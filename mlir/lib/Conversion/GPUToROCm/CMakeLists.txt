set(LLVM_OPTIONAL_SOURCES
  ConvertKernelFuncToHsaco.cpp
)

if (MLIR_ROCM_CONVERSIONS_ENABLED)
 list(APPEND SOURCES "ConvertKernelFuncToHsaco.cpp")
  set(AMDGPU_LIBS
    MC
    AMDGPUCodeGen
    AMDGPUDesc
    AMDGPUInfo
  )

endif()

add_mlir_conversion_library(MLIRGPUtoROCmTransforms
  ${SOURCES}

  DEPENDS
  MLIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core
  ${AMDGPU_LIBS}

  LINK_LIBS PUBLIC
  MLIRGPU
  MLIRIR
  MLIRLLVMIR
  MLIRROCDLIR
  MLIRPass
  MLIRSupport
  MLIRTargetROCDLIR
)

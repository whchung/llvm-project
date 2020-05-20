set(LLVM_OPTIONAL_SOURCES
  ConvertKernelFuncToHsaco.cpp
)

if (MLIR_ROCM_CONVERSIONS_ENABLED)
  set(AMDGPU_LIBS
    MC
    AMDGPUCodeGen
    AMDGPUDesc
    AMDGPUInfo
  )

  add_mlir_conversion_library(MLIRGPUtoROCmTransforms
    ConvertKernelFuncToHsaco.cpp

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
else()
  add_library(MLIRGPUtoROCmTransforms INTERFACE IMPORTED GLOBAL)
endif()

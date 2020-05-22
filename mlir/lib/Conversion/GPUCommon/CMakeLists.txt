set(SOURCES
  ConvertLaunchFuncToRuntimeCalls.cpp
)

if (MLIR_CUDA_CONVERSIONS_ENABLED)
 list(APPEND SOURCES "ConvertKernelFuncToBlob.cpp")
  set(NVPTX_LIBS
    MC
    NVPTXCodeGen
    NVPTXDesc
    NVPTXInfo
  )

endif()

add_mlir_conversion_library(MLIRGPUtoGPURuntimeTransforms
  ${SOURCES}

  DEPENDS
  MLIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core
  ${NVPTX_LIBS}

  LINK_LIBS PUBLIC
  MLIRGPU
  MLIRIR
  MLIRLLVMIR
  MLIRNVVMIR
  MLIRPass
  MLIRSupport
  MLIRTargetNVVMIR
)

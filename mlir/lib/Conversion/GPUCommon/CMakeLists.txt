set(SOURCES
  ConvertLaunchFuncToRuntimeCalls.cpp
)

if (MLIR_CUDA_CONVERSIONS_ENABLED OR MLIR_ROCM_CONVERSIONS_ENABLED)
  list(APPEND SOURCES "ConvertKernelFuncToBlob.cpp")

  add_mlir_conversion_library(MLIRGPUtoGPURuntimeTransforms
    ${SOURCES}

    DEPENDS
    MLIRConversionPassIncGen
    intrinsics_gen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRGPU
    MLIRIR
    MLIRLLVMIR
    MLIRPass
    MLIRSupport
  )
else()
  add_library(MLIRGPUtoGPURuntimeTransforms INTERFACE IMPORTED GLOBAL)
endif()

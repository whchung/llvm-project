set(LLVM_OPTIONAL_SOURCES
  ConvertKernelFuncToHsaco.cpp
)

if (MLIR_ROCM_CONVERSIONS_ENABLED)
  add_mlir_conversion_library(MLIRGPUtoROCmTransforms
    ConvertKernelFuncToHsaco.cpp

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
  add_library(MLIRGPUtoROCmTransforms INTERFACE IMPORTED GLOBAL)
endif()

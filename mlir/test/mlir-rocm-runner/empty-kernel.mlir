// RUN: mlir-rocm-runner %s --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

func @other_func() {
  %cst = constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst, %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst, %block_y = %cst, %block_z = %cst) {
    gpu.terminator
  }
  return
}

// CHECK: [1.23, 1.23, 1.23, 1.23, 1.23]
func @main() {
  %arg0 = alloc() : memref<5xf32>
  %21 = constant 5 : i32
  %22 = memref_cast %arg0 : memref<5xf32> to memref<?xf32>
  %cast = memref_cast %22 : memref<?xf32> to memref<*xf32>
  call @mgpuMemHostRegisterFloat(%cast) : (memref<*xf32>) -> ()
  %23 = memref_cast %22 : memref<?xf32> to memref<*xf32>
  call @other_func() : () -> ()
  call @print_memref_f32(%23) : (memref<*xf32>) -> ()

  return
}

func @mgpuMemHostRegisterFloat(%ptr : memref<*xf32>)
func @print_memref_f32(%ptr : memref<*xf32>)

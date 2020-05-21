// RUN: mlir-rocm-runner %s --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

func @vecadd(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %arg2 : memref<?xf32>) {
  %cst = constant 1 : index
  %cst2 = dim %arg0, 0 : memref<?xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst, %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst2, %block_y = %cst, %block_z = %cst) {
    %a = load %arg0[%tx] : memref<?xf32>
    %b = load %arg1[%tx] : memref<?xf32>
    %c = addf %a, %b : f32
    store %c, %arg2[%tx] : memref<?xf32>
    gpu.terminator
  }
  return
}

// CHECK: [2.46, 2.46, 2.46, 2.46, 2.46]
func @main() {
  %cf1 = constant 1.0 : f32
  %arg0 = alloc() : memref<5xf32>
  %arg1 = alloc() : memref<5xf32>
  %arg2 = alloc() : memref<5xf32>
  %21 = constant 5 : i32
  %22 = memref_cast %arg0 : memref<5xf32> to memref<?xf32>
  %23 = memref_cast %arg1 : memref<5xf32> to memref<?xf32>
  %24 = memref_cast %arg2 : memref<5xf32> to memref<?xf32>
  %cast0 = memref_cast %22 : memref<?xf32> to memref<*xf32>
  %cast1 = memref_cast %23 : memref<?xf32> to memref<*xf32>
  %cast2 = memref_cast %24 : memref<?xf32> to memref<*xf32>
  call @mgpuMemHostRegisterFloat(%cast0) : (memref<*xf32>) -> ()
  call @mgpuMemHostRegisterFloat(%cast1) : (memref<*xf32>) -> ()
  call @mgpuMemHostRegisterFloat(%cast2) : (memref<*xf32>) -> ()
  %25 = call @mgpuMemGetDeviceMemRef1dFloat(%22) : (memref<?xf32>) -> (memref<?xf32>)
  %26 = call @mgpuMemGetDeviceMemRef1dFloat(%23) : (memref<?xf32>) -> (memref<?xf32>)
  %27 = call @mgpuMemGetDeviceMemRef1dFloat(%24) : (memref<?xf32>) -> (memref<?xf32>)

  call @vecadd(%25, %26, %27) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
  call @print_memref_f32(%cast2) : (memref<*xf32>) -> ()
  return
}

func @mgpuMemHostRegisterFloat(%ptr : memref<*xf32>)
func @mgpuMemGetDeviceMemRef1dFloat(%ptr : memref<?xf32>) -> (memref<?xf32>)
func @print_memref_f32(%ptr : memref<*xf32>)

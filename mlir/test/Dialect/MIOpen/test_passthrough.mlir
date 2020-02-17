#map0 = (d0, d1) -> (d0 * 32 + d1)
#map1 = (d0, d1) -> (d1 * 32 + d0)
func @test_passthrough_2d() {
  %0 = alloc() : memref<32x32xf32, #map0>

  %c1024 = constant 1024: index
  %c1 = constant 1: index
  %1 = subview %0[][][]
    : memref<32x32xf32, #map0> to
      memref<32x32xf32, #map1>
  return
}


#mapF0 = (d0, d1, d2, d3) -> (d0 * 32768 + d1 * 1024 + d2 * 32 + d3)
#mapF1 = (d0, d1, d2, d3)[s0, s1, s2, s3] -> (d2 * s0 + d3 * s1 + d0 * s2 + d1 * s3)
func @test_passthrough_4d() {
  %0 = alloc() : memref<32x32x32x32xf32, #mapF0>

  %c1 = constant 1: index
  %1 = subview %0[][][]
    : memref<32x32x32x32xf32, #mapF0> to
      memref<32x32x32x32xf32, #mapF1>
  return
}

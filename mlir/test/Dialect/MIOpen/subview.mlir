
#map0 = (d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)
#map1 = (d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + d1 * s2 + d2 * s3 + s0)

#map2 = (d0, d1) -> (d0 * 22 + d1)
#map3 = (d0, d1)[s0, s1, s2] -> (d0 * s1 + d1 * s2 + s0)

#map4 = (d0, d1)[s0] -> (d0 * 8 + d1 * 2 + s0)
func @subview_test(%arg0 : index, %arg1 : index, %arg2 : index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index

  %0 = alloc() : memref<8x16x4xf32, #map0>
  %1 = subview %0[%c0, %c0, %c0][%arg0, %arg1, %arg2][%c1, %c1, %c1]
    : memref<8x16x4xf32, #map0> to
      memref<?x?x?xf32, #map1>

  %2 = alloc() : memref<64x22xf32, #map2>
  %3 = subview %2[%c0, %c1][%arg0, %arg1][%c1, %c0]
    : memref<64x22xf32, #map2> to
      memref<?x?xf32, #map3>

  %4 = alloc() : memref<16x4xf32>
  %5 = subview %4[%arg1, %arg2][][]
    : memref<16x4xf32> to memref<1x1xf32, #map4>
  return
}

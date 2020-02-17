#map0 = (d0, d1, d2, d3) -> (d0)

func @blah(%0 : memref<32x32x32x32xf32, #map0>) {
  %c0 = constant 0 : index
  load %0[%c0, %c0, %c0, %c0] : memref<32x32x32x32xf32, #map0>
  return
}

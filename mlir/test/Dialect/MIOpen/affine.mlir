//func @affine_learn_1D() {
//  %0 = alloc() : memref<32xf32>
//
//  %c0 = constant 0 : index
//
//  %1 = affine.apply (d0) -> (d0 + 3) (%c0)
//  %2 = affine.apply (d0) -> (d0 + 2) (%1)
//  %3 = affine.apply (d0) -> (d0 floordiv 2) (%2)
//
//  load %0[%3] : memref<32xf32>
//  return
//}
//
//func @affine_learn_2D() {
//  %0 = alloc() : memref<32x32xf32>
//
//  %cW = constant 32 : index
//
//  %c0 = constant 0 : index
//  %c4 = constant 4 : index
//  %c5 = constant 5 : index
//
//  %1 = affine.apply (d0, d1)[s0] -> (d0 * s0 + d1) (%c4, %c5)[%cW]
//  %2 = affine.apply (d0) -> (d0 - 2) (%1)
//  load %0[%c0, %2] : memref<32x32xf32>
//  return
//}
//
//
//#map0 = (d0, d1)[s0] -> (d0 * s0 + d1)
//#map1 = (d0) -> (d0 - 2)
//
//func @affine_learn_2D_with_external_map() {
//  %0 = alloc() : memref<32x32xf32>
//
//  %cW = constant 32 : index
//
//  %c0 = constant 0 : index
//  %c1 = constant 1 : index
//  %c4 = constant 4 : index
//  %c5 = constant 5 : index
//
//  %1 = affine.apply #map0 (%c4, %c5)[%cW]
//  %2 = affine.apply #map1 (%1)
//
//  load %0[%c0, %2] : memref<32x32xf32>
//  return
//}
//
//
//#mapF = (x, y)[c0, c1, c2] -> (c0 * x + c1 * y + c2)
//#mapDF = (x, y, dx, dy)[c0, c1, c2] -> ((c0 * (x + dx) + c1 * (y + dy) + c2) - (c0 * x + c1 * y + c2))
//
//func @affine_learn_2d_test() {
//  %0 = alloc() : memref<32x32xf32>
//
//  %c0 = constant 0: index
//  %c1 = constant 1: index
//  %c10 = constant 10: index
//  %c100 = constant 100: index
//
//  %x = constant 3: index
//  %y = constant 4: index
//  %dx = constant 1: index
//  %dy = constant 1: index
//
//  //%1 = affine.apply #mapDF (%x, %y, %c1, %c1)[%c100, %c10, %c1]
//  //load %0[%c0, %1] : memref<32x32xf32>
//
//  %1 = affine.apply #mapF (%x, %y)[%c100, %c10, %c1]
//  %2 = addi %x, %dx : index
//  %3 = addi %y, %dy : index
//  %4 = affine.apply #mapF (%2, %3)[%c100, %c10, %c1]
//  %5 = subi %4, %1 : index
//  
//  load %0[%c0, %5] : memref<32x32xf32>
//  return
//}
//
//#padReverse = (x)[left_pad] -> (x - left_pad)
//#padReverseCond = (x)[left_pad, right_pad, dim] : (dim - (x - right_pad) >= 0, x - left_pad >= 0)
//
//#padReverseDelta = (x, dx)[left_pad, right_pad] -> (dx)
//
//func @pad_example() {
//  %0 = alloc() : memref<32xf32>
//
//  %left_pad = constant 1 : index
//  %right_pad = constant 1 : index
//
//  %coord = constant 5:index
//  %c32 = constant 32:index
//
//  affine.if #padReverseCond(%coord)[%left_pad, %right_pad, %c32] {
//    %1 = affine.apply #padReverse(%coord)[%left_pad]
//    load %0[%1] : memref<32xf32>
//  } 
//
//  return
//}
//
//#mapP0 = (x, y, z, w) -> (x + 3) 
//#mapP1 = (x, y, z, w) -> (y + 3 + (x + z + w) * 4)
//#mapP2 = (x, y, z, w) -> (z + 3 + (x + z + w) * 4 + (y * 3))
//#mapP3 = (x, y, z, w) -> (w + 3 + (x + z + w) * 4 + (y * 3))
//
//func @partial_map_example() {
//  %0 = alloc() : memref<32x32x32x32xf32>
//
//  %c5 = constant 5: index
//  %c6 = constant 6: index
//  %c7 = constant 7: index
//  %c8 = constant 8: index
//
//  %1 = affine.apply #mapP0(%c5, %c6, %c7, %c8)
//  %2 = affine.apply #mapP1(%1, %c6, %c7, %c8)
//  %3 = affine.apply #mapP2(%1, %2, %c7, %c8)
//  %4 = affine.apply #mapP3(%1, %2, %3, %c8)
//
//  // make sure the result is referenced by providing a dummy op.
//  "op.dummy"(%1, %2, %3, %4) : (index, index, index, index) -> ()
//
//  //%5 = affine.load %0[%1, %2, %3, %4] : memref<32x32x32x32xf32>
//  return
//}


#mapU0 = (y, x)[sy, sz] -> (y * sy + x + sz * 2)
#mapU1 = (y, x)[sy, sz] -> (y * sy + x + sz * 2)
#mapU2 = (y, x)[sy, sz] -> (y * sy + x + sz * 2 + 1)
#mapU3 = (y, x)[sy, sz] -> (y * sy + x + sz * 2 + 1)

func @test(%arg0:index) {
  %0 = alloc() : memref<32x32xf32>

  %c5 = constant 5: index
  %c32 = constant 32: index

  affine.for %i = 0 to 32 {
    %j = affine.apply (d0) -> (d0 + 1) (%i)

    %2 = affine.apply #mapU0(%i, %j)[%c32, %arg0]
    %3 = affine.apply #mapU1(%i, %j)[%c32, %arg0]

    %4 = affine.apply #mapU2(%i, %j)[%c32, %arg0]
    %5 = affine.apply #mapU3(%i, %j)[%c32, %arg0]

    "op.dummy"(%2, %3, %4, %5) : (index, index, index, index) -> ()
  }
  return
}


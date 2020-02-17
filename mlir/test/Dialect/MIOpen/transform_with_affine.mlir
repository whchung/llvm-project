func @test_passthrough(%memref: memref<?x?x?x?xf32>) {
  %transformed_memref = miopen.transform(%memref) {
    layout = [
      {
        dimensions = [3],
        names = ["n"],
        transformation = "passthrough",
        source_dimensions = [0],
        source_names = ["n"]
      },
      {
        dimensions = [3],
        names = ["c"],
        transformation = "passthrough",
        source_dimensions = [1],
        source_names = ["c"]
      },
      {
        dimensions = [0],
        names = ["hi"],
        transformation = "passthrough",
        source_dimensions = [2],
        source_names = ["hi"]
      },
      {
        dimensions = [2],
        names = ["wi"],
        transformation = "passthrough",
        source_dimensions = [3],
        source_names = ["wi"]
      }
    ],
    source_layout = ["n", "c", "hi", "wi"],
    output_layout = ["hi", "wi", "n", "c"]
  } : memref<?x?x?x?xf32> to memref<?x?x?x?xf32>
  return
}


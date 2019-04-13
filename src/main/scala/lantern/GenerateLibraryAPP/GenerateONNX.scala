// package lantern
// package GenerateLibraryAPP

// import java.io.PrintWriter
// import java.io.File

// object GenerateONNXLib {
//   def main(argv: Array[String]) = {
//     val driver = new LanternDriverLibC[Array[Float],Array[Float], Unit] {

//       val modelFile = argv(0) // this is the ONNX file that you want to read
//       override val libDir = argv(1) // this is the directory name of your library function
//       override val libFileName = argv(2) // this is the cpp filename of your library function (without the .cpp)
//       override val libInferenceFuncName = argv(3) // this is the name of the library function
//       System.out.println(s"reading ONNX model from $modelFile to generate library functions in $libDir/$libFileName.cpp")
//       System.out.println(s"it will also generate a header file: $libDir/$libFileName.h")

//       def snippet(x: Rep[Array[Float]], y: Rep[Array[Float]]): Rep[Unit] = {
//         // read the ONNX model
//         val model = readONNX(modelFile)
//         // get hold of the inference function and the input/output dimensions from the model
//         val inferenceFunc = model.inference_func(model.initializer_map_tensor)
//         val (xDims, yDims) = (model.x_dims, model.y_dims)
//         // cast input (x) into a Tensor with the input dimension read from the model
//         val input = Tensor(x, xDims: _*)
//         // comput the output from the input with the inferenceFunc
//         val output = inferenceFunc(input)
//         // copy result to y
//         val temp: Tensor = Tensor(y, yDims: _*)
//         temp.copy_data(output)
//       }
//     }
//     driver.generateLib
//   }
// }

// object GenerateONNXLibTest { // do not use directly (generated code need to be fixed manually)
//   def main(argv: Array[String]) = {
//     val driver = new LanternDriverC[String, Unit] {
//       def snippet(x: Rep[String]): Rep[Unit] = {
//         // get test data as TensorProto
//         val model_dir = "/home/fei/onnx_models/squeezenet/"
//         val input_file =  model_dir + "test_data_set_0/input_0.pb"
//         val output_file = model_dir + "test_data_set_0/output_0.pb"
//         val input = readTensor(input_file).tensor
//         val expect = readTensor(output_file).tensor
        
//         // collect output from generated function
//         // we know the function name is 'funcname'
//         val output = Tensor.zeros(1, 1000, 1, 1)
//         unchecked[Unit]("funcname(", input.data, ", ", output.data, ")")
//         Tensor.assertEqual(expect, output)
//       }
//     }
//     val root_dir = "/home/fei/bitbucket/Lantern/src/out/library/"
//     val test_file = new PrintWriter(new File(root_dir + "test_file.cpp"))
//     test_file.println(driver.code)
//     test_file.flush()
//   }
// }

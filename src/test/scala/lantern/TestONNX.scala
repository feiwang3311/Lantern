package lantern

import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

import org.scalatest.FunSuite

import java.io.PrintWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file._;

import org.bytedeco.javacpp._;
import org.bytedeco.javacpp.onnx._;
import scala.collection.mutable.Map;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

class ONNXTest extends LanternFunSuite {

  def getProtoProps[T](size: Int, propMethod: Int => T): Seq[T] = {
      (0 to (size - 1)).map(y => propMethod(y.toInt)).toSeq
  }

  def getInputs(node: NodeProto) = {
    getProtoProps(node.input_size, node.input(_).getString)
  }

  def getOutputs(node: NodeProto) = {
    getProtoProps(node.output_size, node.output(_).getString)
  }

  def getAttributes(node: NodeProto) = {
    getProtoProps(node.attribute_size, node.attribute(_))
  }

  val model_file_all = (name: String) => s"""${sys.env("HOME")}/onnx_models/$name/model.onnx"""
  val model_dir_all = (name: String) => s"""${sys.env("HOME")}/onnx_models/$name/"""
  val gene_dir = "/tmp/"

  test("onnx_reading_basic") {

    val model_file = model_file_all("squeezenet")
    System.out.println(s"testing reading onnx models from $model_file")

    val byteArray = Files.readAllBytes(Paths.get(model_file))

    val model = new ModelProto()
    ParseProtoFromBytes(model.asInstanceOf[MessageLite],
                      new BytePointer(byteArray: _*),
                      byteArray.length.toLong)



    // graph information
    val graph = model.graph

    def InitInfo(init: TensorProto) = {
      val dims: Seq[Long] = getProtoProps(init.dims_size , init.dims(_))
      println(dims)
      val name: String = init.name.getString
      println("name is " + name)
      val rawdata: org.bytedeco.javacpp.BytePointer = init.raw_data

      val bytes = new Array[Byte](rawdata.asByteBuffer.remaining())
      rawdata.asByteBuffer.get(bytes, 0, bytes.length);

      val bytebuffer: ByteBuffer = ByteBuffer.wrap(bytes)
      val floatbuffer: FloatBuffer = bytebuffer.asFloatBuffer()
      //val floatarray = floatbuffer.array()
      println(floatbuffer)
      val floatarray = new Array[Float](bytes.size / 4)
      floatbuffer.get(floatarray)
      floatarray.foreach { f =>
        print(f + ",")
      }
      println()
      println("length of array is " + floatarray.length)
      val data_type: Int = init.data_type
    }

    // initialize information: initialization values of all inputs except the data
    val init = getProtoProps(graph.initializer_size , graph.initializer(_))
    // input information: Type of data and dimension information
    val input = getProtoProps(graph.input_size , graph.input(_)) 

    def ValueInfo(in: ValueInfoProto) = {
      val name = in.name.getString
      println("name of this value info is " + name)
      val ty: TypeProto = in.`type`
      val tensor: TypeProto_Tensor = ty.tensor_type

      val elem_type: Int = tensor.elem_type
      println(elem_type)

      val shape: TensorShapeProto = tensor.`shape`
      val dim: Seq[Dimension] = getProtoProps(shape.dim_size , shape.dim(_)) 
      val dims: Seq[Long] = dim.map(x => x.dim_value)
      println(dims)
    }

    // node information
    val nodes = getProtoProps(graph.node_size , graph.node(_))

    // conv should have three inputs, one output, a op_type as Conv, and three attributes (strides, pads, and kernel_shape (type ints))
    def ConvNode(node: NodeProto) = {
      assert (node.op_type.getString == "Conv")
      val inputs: Seq[String] = getInputs(node)
      val outputs: Seq[String] =  getOutputs(node)
      val attributes: Seq[AttributeProto] = getAttributes(node)
      val atts: Map[String, Seq[Long]] = Map()
      attributes.foreach { att =>
        atts += (att.name.getString -> getProtoProps(att.ints_size , att.ints(_)))
      }
      println(inputs)
      println(outputs)
      println(atts)
    }

    // relu should have one input, one output, a op_type "Relu"
    def ReluNode(node: NodeProto) = {
      assert (node.op_type.getString == "Relu")
      val inputs: Seq[String] = getInputs(node)
      val outputs: Seq[String] = getOutputs(node)
      println(inputs)
      println(outputs)
    }

    // maxpool have one input, one output, a op_type "MaxPool", and three attributes (strides, pads, and kernel_shape)
    def MaxPoolNode(node: NodeProto) = {
      assert (node.op_type.getString == "MaxPool")
      val inputs: Seq[String] = getInputs(node)
      val outputs: Seq[String] = getOutputs(node)
      val atts: Map[String, Seq[Long]] = Map()
      getAttributes(node).foreach { att =>
        atts += (att.name.getString -> getProtoProps(att.ints_size , att.ints(_)))
      }
      println(inputs)
      println(outputs)
      println(atts)
    }

    // concat have one more than one inputs, one outputs, op_type "Concat", one attribute(axis,(type int))
    def ConcatNode(node: NodeProto) = {
      assert (node.op_type == "Concat")
      val inputs: Seq[String] = getInputs(node)
      val outputs: Seq[String] = getOutputs(node)
      val axis = getAttributes(node).head.i
      println(inputs)
      println(outputs)
      println(axis)
    }

    // dropout should have one input, two outputs (one for real, one for mask), op_type "Dropout", two attributes (ratio (f), is_test (i))
    def DropoutNode(node: NodeProto) = {
      assert (node.op_type == "Dropout")
      val inputs: Seq[String] = getInputs(node)
      val outputs: Seq[String] = getInputs(node)
      val atts = getAttributes(node)
      val ratio: Float = if (atts.head.name == "ratio") atts.head.f
                         else (atts.last.f)
      val is_test: Long = if (atts.last.name == "is_test") atts.last.i
                          else (atts.head.i)
      println(inputs)
      println(outputs)
      println(ratio)
      println(is_test)
    }

    // globalaveragepool should have one input and one output
    def GlobalAveragePoolNode(node: NodeProto) = {
      assert (node.op_type == "GlobalAveragePool")
      val inputs: Seq[String] = getInputs(node)
      val outputs: Seq[String] = getOutputs(node)
      println(inputs)
      println(outputs)
    }

    // softmax should have one input and one output
    def SoftmaxNode(node: NodeProto) = {
      assert (node.op_type == "Softmax")
      val inputs: Seq[String] = getInputs(node)
      val outputs: Seq[String] = getOutputs(node)
      println(inputs)
      println(outputs)
    }
  }

  test("build_from_onnx") {

    val squeezenet = new LanternDriverC[String, Unit] {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        val model_file = model_file_all("squeezenet")
        val byteArray = Files.readAllBytes(Paths.get(model_file))

        val model = new ModelProto()
        ParseProtoFromBytes(model.asInstanceOf[MessageLite],
                      new BytePointer(byteArray: _*),
                      byteArray.length.toLong)
 
 
        val graph = model.graph

        def graphBuilder(graph: GraphProto) = {
          val initializer: Seq[TensorProto] = getProtoProps(graph.initializer_size , graph.initializer(_))
          val inputs: Seq[ValueInfoProto] = getProtoProps(graph.input_size , graph.input(_))
          System.out.println("Inputs: + " +  inputs)
          val outputs: Seq[ValueInfoProto] = getProtoProps(graph.output_size , graph.output(_)) 
          val nodes: Seq[NodeProto] = getProtoProps(graph.node_size , graph.node(_))

          // extract the initialization tensor values (as array[Float]) and dims (as Vector[Long]) and types for each initialization
          // TODO (Fei Wang): problem: this function is assuming that the data type is Float, will break if not!!!
          val initializer_map: Map[String, (Seq[Int], Int, Array[Float])] = extract_inits(initializer)
          // collect all tensor and tensorR from the initialzer_map
          val initializer_map_tensor: Map[String, Tensor] = initializer_map.map { case (name, (dims, _, value)) => (name -> Tensor(Array((value.map(x=>unit(x)).toSeq: _*)), dims: _*)) }
          val initializer_map_tensorR: Map[String, TensorR] = initializer_map_tensor.map { case (name, tensor) => (name -> TensorR(tensor))}

          // extract the inputs values (as dims and types)
          val input_map: Map[String, (Seq[Int], Int)] = extract_values(inputs)
          val output_map: Map[String, (Seq[Int], Int)] = extract_values(outputs)

          // extract the nodes (as Functors)
          val (func, x_dims) = map_function(nodes, initializer_map_tensor, input_map, output_map)
          (func, x_dims)
        }

        def extract_inits(inits: Seq[TensorProto]): Map[String, (Seq[Int], Int, Array[Float])] = {
          val map: Map[String, (Seq[Int], Int, Array[Float])] = Map()
          // TODO: (Fei Wang) use immutable map instead!!
          inits.foreach { init =>
            val (name, dims, datatype, floatarray) = extract_init(init)
            map += (name -> (dims, datatype, floatarray))
          }
          map
        }

        def extract_init(init: TensorProto): (String, Seq[Int], Int, Array[Float]) = {
          val dims: Seq[Int] = getProtoProps(init.dims_size , init.dims(_).toInt)
          val name: String = init.name.getString
          val datatype: Int = init.data_type
          if (datatype != TensorProto.FLOAT) throw new RuntimeException("data type not Float, Not handling yet: " + datatype)
          val rawdata: org.bytedeco.javacpp.BytePointer = init.raw_data

          val bytes = new Array[Byte](rawdata.asByteBuffer.remaining())
          rawdata.asByteBuffer.get(bytes, 0, bytes.length);

          val bytebuffer: ByteBuffer = ByteBuffer.wrap(bytes)
          val floatbuffer: FloatBuffer = bytebuffer.asFloatBuffer()
          val floatarray: Array[Float] = new Array[Float](bytes.size / 4)
          floatbuffer.get(floatarray)
          // make sure that the initialization values correspond with the dims
          assert (floatarray.length == dims.fold(1)(_ * _))
          (name, dims, datatype, floatarray)
        }

        def extract_values(puts: Seq[ValueInfoProto]): Map[String, (Seq[Int], Int)] = {
          val map: Map[String, (Seq[Int], Int)] = Map()
          // TODO: (Fei Wang) use immutable map instead!!
          puts.foreach { put =>
            val (name, dims, elem_type) = extract_value(put)
            map += (name -> (dims, elem_type))
          }
          map
        }

        def extract_value(put: ValueInfoProto): (String, Seq[Int], Int) = {
          val name: String = put.name.getString
          val ty: TypeProto = put.`type`
          val tensor: TypeProto_Tensor = ty.tensor_type
          val elem_type: Int = tensor.elem_type
          val shape: TensorShapeProto = tensor.shape
          val dim: Seq[Dimension] = getProtoProps(shape.dim_size , shape.dim(_))
          val dims: Seq[Int] = (dim.map(x => x.dim_value)).map(x => x.toInt)
          (name, dims, elem_type)
        }

        def map_function(nodes: Seq[NodeProto], initializer_map_tensor: Map[String, Tensor],
          input_map: Map[String, (Seq[Int], Int)],
          output_map: Map[String, (Seq[Int], Int)]): ((Rep[Array[Float]] => Tensor), Seq[Int]) = {

          // collect the parameter tensor in initializer_map_tensor
          val all_inputs = input_map.keys
          val non_initialized_inputs = all_inputs.filter(k => !initializer_map_tensor.contains(k))
          assert(non_initialized_inputs.size == 1, "there should be one uninitialized input")
          val x_name: String = non_initialized_inputs.head
          val x_dims: Seq[Int] = input_map(x_name)._1.map(x => x.toInt)

          val ret: (Rep[Array[Float]] => Tensor) = { x: Rep[Array[Float]] =>

            val x_tensor: Tensor = Tensor(x, x_dims: _*)
            initializer_map_tensor += (x_name -> x_tensor)

            // generate Tensors (or TensorRs) of intermediate steps
            val intermediate_map_tensor: Map[String, Tensor] = Map()

            def get_from_two_maps(key: String) = {
              initializer_map_tensor.get(key) match {
                case Some(v) => v
                case None => intermediate_map_tensor.get(key) match {
                  case Some(v) => v
                  case None => throw new RuntimeException(key + " is not found in either maps")
                }
              }
            }

            def handle_conv(node: NodeProto) = {

              val inputs: Seq[String] = getInputs(node)
              assert (inputs.size == 3, "number of inputs of a conv node should always be 3")
              val input1 = get_from_two_maps(inputs.head)
              val input2 = get_from_two_maps(inputs.tail.head)
              val input3 = get_from_two_maps(inputs.last)

              val outputs: Seq[String] = getOutputs(node)
              assert (outputs.size == 1, "number of output of a conv node should always be 1")

              val attributes: Seq[AttributeProto] = getAttributes(node)
              assert (attributes.size == 3, "number of attributes of a conv node should always be 3")
              val atts: Map[String, Seq[Int]] = Map()
              attributes.foreach { att => atts += (att.name.getString -> getProtoProps(att.ints_size , att.ints(_)).map(x => x.toInt)) }
              assert (atts.contains("strides"), "attributes of a conv node should have strides")
              assert (atts.contains("pads"), "attributes of a conv node should have pads")
              assert (atts.contains("kernel_shape"), "attributes of a conv node should have kernel_shape")
              val strides = atts("strides")
              val pads = atts("pads")
              val kernel_shape = atts("kernel_shape")  // this attribute is actually not used

              val (out, finputOption) = input1.conv2D_batch(input2, Some(input3), strides, pads)
              intermediate_map_tensor += (outputs.head -> out)
            }

            def handle_relu(node: NodeProto) = {

              val inputs: Seq[String] = getInputs(node)
              assert (inputs.size == 1, "number of inputs of a relu node should always be 1")
              val input1 = get_from_two_maps(inputs.head)

              val outputs: Seq[String] = getOutputs(node)
              assert (outputs.size == 1, "number of outputs of a relu node should always be 1")

              val out = input1.relu()
              intermediate_map_tensor += (outputs.head -> out)
            }

            def handle_maxpool(node: NodeProto) = {

              val inputs: Seq[String] = getInputs(node)
              assert (inputs.size == 1, "number of inputs of a maxpool node should always be 1")
              val input1 = get_from_two_maps(inputs.head)

              val outputs: Seq[String] = getOutputs(node)
              assert (outputs.size == 1, "number of outputs of a maxpool node should always be 1")

              val attributes: Seq[AttributeProto] = getAttributes(node)
              assert (attributes.size == 3, "number of attributes of a conv node should always be 3")
              val atts: Map[String, Seq[Int]] = Map()
              attributes.foreach { att => atts += (att.name.getString -> getProtoProps(att.ints_size , att.ints(_)).map(x => x.toInt)) }
              assert (atts.contains("strides"), "attributes of a conv node should have strides")
              assert (atts.contains("pads"), "attributes of a conv node should have pads")
              assert (atts.contains("kernel_shape"), "attributes of a conv node should have kernel_shape")
              val strides = atts("strides")
              val pads = atts("pads")
              val kernel_shape = atts("kernel_shape")
              assert(strides.size == 2, "strides should be length 2")
              assert(kernel_shape.size == 2, "kernel_shape should be length 2")

              // TODO: (Fei Wang) erroneous code, the implementation assumes that pads are all 0
              val (out, dummy) = input1.maxPool2D_batch(kernel_shape, strides, None)
              intermediate_map_tensor += (outputs.head -> out)
            }

            def handle_concat(node: NodeProto) = {

              val inputs: Seq[String] = getInputs(node)
              assert (inputs.size > 1, "number of inputs for concat node should be larger than 1")
              val input_s = inputs.map(x => get_from_two_maps(x))

              val outputs: Seq[String] = getOutputs(node)
              assert (outputs.size == 1, "number of outputs for concat node should be 1")

              val attributes: Seq[AttributeProto] = getAttributes(node)
              assert (attributes.size == 1, "number of attributes of a concat node should be 1")
              val axis: Int = attributes.head.i.toInt

              val out = input_s.head.concat(axis, input_s.tail: _*)
              intermediate_map_tensor += (outputs.head -> out)
            }

            def handle_dropout(node: NodeProto) = {

              val inputs: Seq[String] = getInputs(node)
              assert (inputs.size == 1, "number of inputs for dropout node should always be 1")
              val input1 = get_from_two_maps(inputs.head)

              val outputs: Seq[String] = getOutputs(node)
              assert (outputs.size == 2, "number of outputs for dropout node should always be 2")

              val attributes: Seq[AttributeProto] = getAttributes(node)
              assert (attributes.size == 1, "number of attributes for dropout node should be 1")
              val ratio: Float = attributes.head.f

              val (out, helper, size) = input1.dropout(ratio)
              // val (out1, out2) = dropout_fun(input1, ratio, is_test)
              intermediate_map_tensor += (outputs.head -> out)
              // intermediate_map_tensor += (outputs.last -> Tensor(helper, input1.shape: _*))
            }

            def handle_global_average_pool(node: NodeProto) = {

              val inputs: Seq[String] = getInputs(node)
              assert (inputs.size == 1, "number of inputs for global_average_pool should be 1")
              val input1 = get_from_two_maps(inputs.head)

              val outputs: Seq[String] = getOutputs(node)
              assert (outputs.size == 1, "number of outputs for global_average_pool should be 1")

              val out = input1.global_ave_batch()
              intermediate_map_tensor += (outputs.head -> out)
            }

            def handle_softmax(node: NodeProto) = {

              val inputs: Seq[String] = getInputs(node)
              assert (inputs.size == 1, "number of inputs for softmax node should 1")
              val input1 = get_from_two_maps(inputs.head)

              val outputs: Seq[String] = getOutputs(node)
              assert (outputs.size == 1, "number of outputs for softmax node should be 1")

              val out = input1.softmax_batch()
              intermediate_map_tensor += (outputs.head -> out)
            }

            nodes.foreach { node =>
              node.op_type.getString match {
                case "Conv" => {handle_conv(node)}
                case "Relu" => {handle_relu(node)}
                case "MaxPool" => {handle_maxpool(node)}
                case "Concat" => {handle_concat(node)}
                case "Dropout" => {handle_dropout(node)}
                case "GlobalAveragePool" => {handle_global_average_pool(node)}
                case "Softmax" => {handle_softmax(node)}
              }
            }

            val out_keys = output_map.keys
            assert (out_keys.size == 1, "we hope that there is only one output for the model")
            val out_key: String = out_keys.head
            // return this tensor / tensorR / tensorR@diff for this function
            intermediate_map_tensor(out_key)
          }

          (ret, x_dims)
        }

        val (func, x_dims) = graphBuilder(graph)
        val input = Tensor.zeros(x_dims: _*)
        val output = func(input.data)
        println(output.data(0))
      }
    }

    val squeezenet_file = new PrintWriter(new File(gene_dir + "squeezenet.cpp"))
    squeezenet_file.println(squeezenet.code)
    squeezenet_file.flush()

  }

  test("squeezenet_inference") {

    val model_file = model_file_all("squeezenet")
    val model_dir = model_dir_all("squeezenet")
    System.out.println(s"testing reading ONNX model using library from $model_file")

    val inference_func = new LanternDriverC[String, Unit] with ONNXLib {
      override val fileName = currentTestName

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        val model = readONNX(model_file)
        val (func, x_dims) = (model.inference_func(model.initializer_map_tensor), model.x_dims)

        // get test data as TensorProto
        val input_file =  model_dir + "test_data_set_0/input_0.pb"
        val output_file = model_dir + "test_data_set_0/output_0.pb"
        val input = readTensor(input_file).tensor
        val output = readTensor(output_file).tensor
        val output1 = func(input)
        Tensor.assertEqual(output, output1.resize(1, 1000, 1, 1))
      }
    }

    val squeezenet_file = new PrintWriter(new File(gene_dir + "squeezenet.cpp"))
    squeezenet_file.println(inference_func.code)
    squeezenet_file.flush()
    runTest(inference_func)
  }

  test("squeezenet_training") {

    val model_file = model_file_all("squeezenet")
    val model_dir = model_dir_all("squeezenet")
    System.out.println(s"testing reading ONNX model using library from $model_file for training")

    val training_func = new LanternDriverC[String, Unit] with ONNXLib {
      override val fileName = currentTestName

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        // reading ONNX model
        val model = readONNX(model_file)
        val ((func, _), x_dims, y_dims) = (model.training_func(model.initializer_map_tensor), model.x_dims, model.y_dims)

        // fake input and target
        val input_file = model_dir + "test_data_set_0/input_0.pb"
        val output_file = model_dir + "test_data_set_0/output_0.pb"
        val input = readTensor(input_file).tensor
        val output = readTensor(output_file).tensor

        val target = NewArray[Int](x_dims(0))
        for (i <- DataLoop(x_dims(0))) target(i) = 1
        def lossFun(dummy: TensorR) = func(TensorR(input)).nllLossB(target).sum()

        val loss = gradR_loss(lossFun)(Tensor.zeros(1))
        println(loss.data(0))
      }
    }

    val squeezenet_file = new PrintWriter(new File(gene_dir + "squeezenetTraining.cpp"))
    squeezenet_file.println(training_func.code)
    squeezenet_file.flush()
    runTest(training_func)
  }

  // test("resnet_inference") {

  //   val model_file = model_file_all("resnet50")
  //   val model_dir = model_dir_all("resnet50")
  //   System.out.println(s"testing reading ONNX model using library from $model_file")

  //   val inference_func = new LanternDriverC[String, Unit] with ONNXLib {

  //     @virtualize
  //     def snippet(a: Rep[String]): Rep[Unit] = {
  //       val model = readONNX(model_file)
  //       val (func, x_dims) = (model.inference_func, model.x_dims)

  //       // get test data as TensorProto
  //       val input_file = model_dir + "test_data_set_0/input_0.pb"
  //       val output_file = model_dir + "test_data_set_0/output_0.pb"
  //       val input = readTensor(input_file).tensor
  //       val output = readTensor(output_file).tensor
  //       val output1 = func(input)
  //       Tensor.assertEqual(output, output1)
  //     }
  //   }
  //   val resnet_file = new PrintWriter(new File(gene_dir + "resnet.cpp"))
  //   resnet_file.println(inference_func.code)
  //   resnet_file.flush()
  //   runTest(inference_func)
  // }

  // test("resnet_training") {

  //   val model_file = model_file_all("resnet50")
  //   val model_dir = model_dir_all("resnet50")
  //   System.out.println(s"testing reading ONNX model using library from $model_file for training")

  //   val training_func = new LanternDriverC[String, Unit] with ONNXLib {

  //     @virtualize
  //     def snippet(a: Rep[String]): Rep[Unit] = {
  //       val model = readONNX(model_file)
  //       val (func, x_dims, y_dims) = (model.training_func, model.x_dims, model.y_dims)

  //       // fake input and target
  //       val input_file = model_dir + "test_data_set_0/input_0.pb"
  //       val output_file = model_dir + "test_data_set_0/output_0.pb"
  //       val input = readTensor(input_file).tensor
  //       val output = readTensor(output_file).tensor

  //       val target = NewArray[Int](x_dims(0))
  //       for (i <- DataLoop(x_dims(0))) target(i) = 1
  //       def lossFun(dummy: TensorR) = func(TensorR(input)).nllLossB(target).sum()

  //       val loss = gradR_loss(lossFun)(Tensor.zeros(1))
  //       println(loss.data(0))
  //     }
  //   }

  //   val resnet_file = new PrintWriter(new File(gene_dir + "resnetTraining.cpp"))
  //   resnet_file.println(training_func.code)
  //   resnet_file.flush()
  //   // runTest(training_func)
  // }
}

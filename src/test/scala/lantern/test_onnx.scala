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

import onnx.onnx_ml;
import scala.collection.mutable.Map;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

class ONNXTest extends FunSuite {

  test("onnx_reading_basic") {
    
    val filename = "src/test/onnxModel/squeezenet/model.onnx"
    println(s"testing reading onnx models from $filename")

    // model information
    val model = onnx_ml.ModelProto.parseFrom(new FileInputStream(filename))
    // println("irversion is " + model.getIrVersion)
    // println("producer name is " + model.getProducerName)
    // println("producer version is " + model.getProducerVersion)
    // println("domain is " + model.getDomain)
    // println("model version is " + model.getModelVersion)
    // println("doc string is " + model.getDocString)


    // graph information
    val graph = model.getGraph
    // println("name of graph is " + graph.getName)


    // initialize information: initialization values of all inputs except the data
    // val init = graph.initializer
    // println("number of initializer is " + init.length)
    // val inithead: onnx_ml.TensorProto = init.head
    // println(inithead.toProtoString)
    // InitInfo(inithead)

    def InitInfo(init: onnx_ml.TensorProto) = {
      val dims: Seq[Long] = init.dims
      println(dims)
      val name: String = init.getName
      println("name is " + name)
      val rawdata: com.google.protobuf.ByteString = init.getRawData
      val bytearray = rawdata.toByteArray
      println(rawdata.size)

      val bytebuffer: ByteBuffer = ByteBuffer.wrap(bytearray)
      val floatbuffer: FloatBuffer = bytebuffer.asFloatBuffer()
      //val floatarray = floatbuffer.array()
      println(floatbuffer)
      val floatarray = new Array[Float](rawdata.size / 4)
      floatbuffer.get(floatarray)
      floatarray.foreach { f => 
        print(f + ",")
      }
      println()
      println("length of array is " + floatarray.length)
      val data_type: onnx_ml.TensorProto.DataType = init.getDataType
      val data_type_name: String = data_type.name
      println(data_type.name)
    }
    

    // input information: Type of data and dimension information
    //val input = graph.input
    //println("number of input is " + input.length)

    def ValueInfo(in: onnx_ml.ValueInfoProto) = {
      val name = in.getName
      println("name of this value info is " + name)
      val ty: onnx_ml.TypeProto = in.getType
      val tensor: onnx_ml.TypeProto.Tensor = ty.getTensorType
      
      val elem_type: onnx_ml.TensorProto.DataType = tensor.getElemType
      println(elem_type)
      
      val shape: onnx_ml.TensorShapeProto = tensor.getShape      
      val dim: Seq[onnx_ml.TensorShapeProto.Dimension] = shape.dim
      val dims: Seq[Long] = dim.map(x => x.getDimValue)
      println(dims)
    }

    // output information: Type of data and dimension information
    // val output = graph.output
    // println("number of output is " + output.length)
    // output.foreach { out =>
    //  ValueInfo(out)
    // }


    // node information
    val nodes = graph.node
    println("number of node is " + nodes.length)
    
    nodes.foreach { node =>
      println(node.toProtoString)
    }
    
    /*
    nodes.foreach { node => 
      node.getOpType match {
        case "Conv" => ConvNode(node)
        case "Relu" => ReluNode(node)
        case "MaxPool" => MaxPoolNode(node)
        case "Concat" => ConcatNode(node)
        case "Dropout" => DropoutNode(node)
        case "GlobalAveragePool" => GlobalAveragePoolNode(node)
        case "Softmax" => SoftmaxNode(node)
        case _ => println("ERROR: node getOpType is " + node.getOpType)
      }
    } */

    // conv should have three inputs, one output, a op_type as Conv, and three attributes (strides, pads, and kernel_shape (type ints))
    def ConvNode(node: onnx_ml.NodeProto) = {
      assert (node.getOpType == "Conv")
      val inputs: Seq[String] = node.input
      val outputs: Seq[String] = node.output
      val attributes: Seq[onnx_ml.AttributeProto] = node.attribute
      val atts: Map[String, Seq[Long]] = Map()
      attributes.foreach { att =>
        atts += (att.getName -> att.ints)
      }
      println(inputs)
      println(outputs)
      println(atts)
    }

    // relu should have one input, one output, a op_type "Relu"
    def ReluNode(node: onnx_ml.NodeProto) = {
      assert (node.getOpType == "Relu")
      val inputs: Seq[String] = node.input
      val outputs: Seq[String] = node.output
      println(inputs)
      println(outputs)
    }

    // maxpool have one input, one output, a op_type "MaxPool", and three attributes (strides, pads, and kernel_shape)
    def MaxPoolNode(node: onnx_ml.NodeProto) = {
      assert (node.getOpType == "MaxPool")
      val inputs: Seq[String] = node.input
      val outputs: Seq[String] = node.output
      val atts: Map[String, Seq[Long]] = Map()
      node.attribute.foreach { att =>
        atts += (att.getName -> att.ints)
      }
      println(inputs)
      println(outputs)
      println(atts)
    }

    // concat have one more than one inputs, one outputs, op_type "Concat", one attribute(axis,(type int))
    def ConcatNode(node: onnx_ml.NodeProto) = {
      assert (node.getOpType == "Concat")
      val inputs: Seq[String] = node.input
      val outputs: Seq[String] = node.output
      val axis = node.attribute.head.getI
      println(inputs)
      println(outputs)
      println(axis)
    }

    // dropout should have one input, two outputs (one for real, one for mask), op_type "Dropout", two attributes (ratio (f), is_test (i))
    def DropoutNode(node: onnx_ml.NodeProto) = {
      assert (node.getOpType == "Dropout")
      val inputs: Seq[String] = node.input
      val outputs: Seq[String] = node.output
      val ratio: Float = if (node.attribute.head.name == "ratio") node.attribute.head.getF 
                         else (node.attribute.last.getF)
      val is_test: Long = if (node.attribute.last.name == "is_test") node.attribute.last.getI
                          else (node.attribute.head.getI)
      println(inputs)
      println(outputs)
      println(ratio)
      println(is_test)
    }

    // globalaveragepool should have one input and one output
    def GlobalAveragePoolNode(node: onnx_ml.NodeProto) = {
      assert (node.getOpType == "GlobalAveragePool")
      val inputs: Seq[String] = node.input
      val outputs: Seq[String] = node.output
      println(inputs)
      println(outputs)
    }

    // softmax should have one input and one output
    def SoftmaxNode(node: onnx_ml.NodeProto) = {
      assert (node.getOpType == "Softmax")
      val inputs: Seq[String] = node.input
      val outputs: Seq[String] = node.output
      println(inputs)
      println(outputs)
    }
  }

  val model_file = "src/test/onnxModel/squeezenet/model.onnx"
  val gene_dir = "src/out/untested/"
  
  val squeezenet = new DslDriverC[String, Unit] with TensorExp {

    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {

      val model = onnx_ml.ModelProto.parseFrom(new FileInputStream(model_file))
      val graph = model.getGraph

      def graphBuilder(graph: onnx_ml.GraphProto) = {
        val initializer: Seq[onnx_ml.TensorProto] = graph.initializer
        val inputs: Seq[onnx_ml.ValueInfoProto] = graph.input
        val outputs: Seq[onnx_ml.ValueInfoProto] = graph.output
        val nodes: Seq[onnx_ml.NodeProto] = graph.node

        // extract the initialization tensor values (as array[Float]) and dims (as Vector[Long]) and types for each initialization
        // TODO (Fei Wang): problem: this function is assuming that the data type is Float, will break if not!!!
        val initializer_map: Map[String, (Seq[Int], onnx_ml.TensorProto.DataType, Array[Float])] = extract_inits(initializer)
        // collect all tensor and tensorR from the initialzer_map
        val initializer_map_tensor: Map[String, Tensor] = initializer_map.map { case (name, (dims, _, value)) => (name -> Tensor(Array((value.map(x=>unit(x)).toSeq: _*)), dims: _*)) }
        val initializer_map_tensorR: Map[String, TensorR] = initializer_map_tensor.map { case (name, tensor) => (name -> TensorR(tensor))}

        // extract the inputs values (as dims and types)
        val input_map: Map[String, (Seq[Int], onnx_ml.TensorProto.DataType)] = extract_values(inputs)
        val output_map: Map[String, (Seq[Int], onnx_ml.TensorProto.DataType)] = extract_values(outputs)

        // extract the nodes (as Functors)
        val (func, x_dims) = map_function(nodes, initializer_map_tensor, input_map, output_map)
        (func, x_dims)
      }
        
      def extract_inits(inits: Seq[onnx_ml.TensorProto]): Map[String, (Seq[Int], onnx_ml.TensorProto.DataType, Array[Float])] = {
        val map: Map[String, (Seq[Int], onnx_ml.TensorProto.DataType, Array[Float])] = Map()
        // TODO: (Fei Wang) use immutable map instead!!
        inits.foreach { init => 
          val (name, dims, datatype, floatarray) = extract_init(init)
          map += (name -> (dims, datatype, floatarray))
        }
        map
      }

      def extract_init(init: onnx_ml.TensorProto): (String, Seq[Int], onnx_ml.TensorProto.DataType, Array[Float]) = {
        val dims: Seq[Int] = init.dims.map(x => x.toInt)
        val name: String = init.getName
        val datatype: onnx_ml.TensorProto.DataType = init.getDataType
        if (datatype.name != "FLOAT") throw new RuntimeException("data type not Float, Not handling yet: " + datatype.name)
        val rawdata: com.google.protobuf.ByteString = init.getRawData
        val bytearray: Array[Byte] = rawdata.toByteArray
        val bytebuffer: ByteBuffer = ByteBuffer.wrap(bytearray)
        val floatbuffer: FloatBuffer = bytebuffer.asFloatBuffer()
        val floatarray: Array[Float] = new Array[Float](rawdata.size / 4)
        floatbuffer.get(floatarray)
        // make sure that the initialization values correspond with the dims
        assert (floatarray.length == dims.fold(1)(_ * _))
        (name, dims, datatype, floatarray)
      }

      def extract_values(puts: Seq[onnx_ml.ValueInfoProto]): Map[String, (Seq[Int], onnx_ml.TensorProto.DataType)] = {
        val map: Map[String, (Seq[Int], onnx_ml.TensorProto.DataType)] = Map()
        // TODO: (Fei Wang) use immutable map instead!!
        puts.foreach { put =>
          val (name, dims, elem_type) = extract_value(put)
          map += (name -> (dims, elem_type))
        }
        map
      }

      def extract_value(put: onnx_ml.ValueInfoProto): (String, Seq[Int], onnx_ml.TensorProto.DataType) = {
        val name: String = put.getName
        val ty: onnx_ml.TypeProto = put.getType
        val tensor: onnx_ml.TypeProto.Tensor = ty.getTensorType
        val elem_type: onnx_ml.TensorProto.DataType = tensor.getElemType
        val shape: onnx_ml.TensorShapeProto = tensor.getShape
        val dim: Seq[onnx_ml.TensorShapeProto.Dimension] = shape.dim
        val dims: Seq[Int] = (dim.map(x => x.getDimValue)).map(x => x.toInt)
        (name, dims, elem_type)
      }

      def map_function(nodes: Seq[onnx_ml.NodeProto], initializer_map_tensor: Map[String, Tensor],
        input_map: Map[String, (Seq[Int], onnx_ml.TensorProto.DataType)],
        output_map: Map[String, (Seq[Int], onnx_ml.TensorProto.DataType)]): ((Rep[Array[Float]] => Tensor), Seq[Int]) = {

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

          def handle_conv(node: onnx_ml.NodeProto) = {
            
            val inputs: Seq[String] = node.input
            assert (inputs.size == 3, "number of inputs of a conv node should always be 3")
            val input1 = get_from_two_maps(inputs.head)
            val input2 = get_from_two_maps(inputs.tail.head)
            val input3 = get_from_two_maps(inputs.last)
            
            val outputs: Seq[String] = node.output
            assert (outputs.size == 1, "number of output of a conv node should always be 1")
            
            val attributes: Seq[onnx_ml.AttributeProto] = node.attribute
            assert (attributes.size == 3, "number of attributes of a conv node should always be 3")
            val atts: Map[String, Seq[Int]] = Map()
            attributes.foreach { att => atts += (att.getName -> att.ints.map(x => x.toInt)) }
            assert (atts.contains("strides"), "attributes of a conv node should have strides")
            assert (atts.contains("pads"), "attributes of a conv node should have pads")
            assert (atts.contains("kernel_shape"), "attributes of a conv node should have kernel_shape")
            val strides = atts("strides")
            val pads = atts("pads")
            val kernel_shape = atts("kernel_shape")  // this attribute is actually not used
            
            val out = input1.conv2D_batch(input2, input3, strides, pads)
            intermediate_map_tensor += (outputs.head -> out)
          }

          def handle_relu(node: onnx_ml.NodeProto) = {
            
            val inputs: Seq[String] = node.input
            assert (inputs.size == 1, "number of inputs of a relu node should always be 1")
            val input1 = get_from_two_maps(inputs.head)

            val outputs: Seq[String] = node.output
            assert (outputs.size == 1, "number of outputs of a relu node should always be 1")
            
            val out = input1.relu()
            intermediate_map_tensor += (outputs.head -> out)
          }

          def handle_maxpool(node: onnx_ml.NodeProto) = {

            val inputs: Seq[String] = node.input
            assert (inputs.size == 1, "number of inputs of a maxpool node should always be 1")
            val input1 = get_from_two_maps(inputs.head)

            val outputs: Seq[String] = node.output
            assert (outputs.size == 1, "number of outputs of a maxpool node should always be 1")
            
            val attributes: Seq[onnx_ml.AttributeProto] = node.attribute
            assert (attributes.size == 3, "number of attributes of a conv node should always be 3")
            val atts: Map[String, Seq[Int]] = Map()
            attributes.foreach { att => atts += (att.getName -> att.ints.map(x => x.toInt)) }
            assert (atts.contains("strides"), "attributes of a conv node should have strides")
            assert (atts.contains("pads"), "attributes of a conv node should have pads")
            assert (atts.contains("kernel_shape"), "attributes of a conv node should have kernel_shape")
            val strides = atts("strides")
            val pads = atts("pads")
            val kernel_shape = atts("kernel_shape")
            assert(strides.size == 2, "strides should be length 2")
            assert(kernel_shape.size == 2, "kernel_shape should be length 2")
            
            // TODO: (Fei Wang) erroneous code, the implementation assumes that pads are all 0
            val (out, dummy) = input1.maxPool_k_batch(kernel_shape, strides)
            intermediate_map_tensor += (outputs.head -> out)
          }

          def handle_concat(node: onnx_ml.NodeProto) = {

            val inputs: Seq[String] = node.input
            assert (inputs.size > 1, "number of inputs for concat node should be larger than 1")
            val input_s = inputs.map(x => get_from_two_maps(x))

            val outputs: Seq[String] = node.output
            assert (outputs.size == 1, "number of outputs for concat node should be 1")

            val attributes: Seq[onnx_ml.AttributeProto] = node.attribute
            assert (attributes.size == 1, "number of attributes of a concat node should be 1")
            val axis: Int = attributes.head.getI.toInt

            val out = input_s.head.concat(axis, input_s.tail: _*)
            intermediate_map_tensor += (outputs.head -> out)
          }

          def handle_dropout(node: onnx_ml.NodeProto) = {

            val inputs: Seq[String] = node.input
            assert (inputs.size == 1, "number of inputs for dropout node should always be 1")
            val input1 = get_from_two_maps(inputs.head)

            val outputs: Seq[String] = node.output
            assert (outputs.size == 2, "number of outputs for dropout node should always be 2")

            val attributes: Seq[onnx_ml.AttributeProto] = node.attribute
            assert (attributes.size == 2, "number of attributes for dropout node should be 2")
            val ratio: Float = if (attributes.head.name == "ratio") attributes.head.getF else attributes.last.getF
            val is_test: Int = (if (attributes.last.name == "is_test") attributes.last.getI else attributes.head.getI).toInt

            // TODO: (Fei Wang) warning - the is_test is not considered by the implementation
            val (out1, out2) = input1.dropout(ratio)
            // val (out1, out2) = dropout_fun(input1, ratio, is_test)
            intermediate_map_tensor += (outputs.head -> out1)
            intermediate_map_tensor += (outputs.last -> out2)
          }

          def handle_global_average_pool(node: onnx_ml.NodeProto) = {

            val inputs: Seq[String] = node.input
            assert (inputs.size == 1, "number of inputs for global_average_pool should be 1")
            val input1 = get_from_two_maps(inputs.head)

            val outputs: Seq[String] = node.output
            assert (outputs.size == 1, "number of outputs for global_average_pool should be 1")

            val out = input1.global_ave_batch()
            intermediate_map_tensor += (outputs.head -> out)
          }

          def handle_softmax(node: onnx_ml.NodeProto) = {

            val inputs: Seq[String] = node.input
            assert (inputs.size == 1, "number of inputs for softmax node should 1")
            val input1 = get_from_two_maps(inputs.head)

            val outputs: Seq[String] = node.output
            assert (outputs.size == 1, "number of outputs for softmax node should be 1")

            val out = input1.softmax_batch()
            intermediate_map_tensor += (outputs.head -> out)
          }

          // System.out.println("handle each node respectively")
          nodes.foreach { node =>
            node.getOpType match {
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

      def func_dummy(b: Rep[Boolean]) = {
        if (b) println("fooled you")
        else {
          val input = Tensor.zeros(x_dims: _*)
          val output = func(input.data)
          println(output.data(0))
        }
      }

      func_dummy(a > "a")
    }
  }

  test("build_from_onnx") {
    
    println(s"testing reading ONNX models from $model_file")

    val squeezenet_file = new PrintWriter(new File(gene_dir + "squeezenet.cpp"))
    squeezenet_file.println(squeezenet.code)
    squeezenet_file.flush()

  }

  test("inference") {

    System.out.println(s"testing reading ONNX model using library from $model_file")

    val inference_func = new DslDriverC[String, Unit] with ONNXLib {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        
        val model = readONNX(model_file)

        val (func, x_dims) = (model.inference_func, model.x_dims)
        
        // must use the function in order to generate it
        if (a == "run") {
          val input = Tensor.zeros(x_dims: _*)
          val output = func(input)
          println(output.data(0))
        }
      }
    }

    val squeezenet_file = new PrintWriter(new File(gene_dir + "squeezenet.cpp"))
    squeezenet_file.println(inference_func.code)
    squeezenet_file.flush()

  }

  test("training") {

    System.out.println(s"testing reading ONNX model using library from $model_file for training")

    val training_func = new DslDriverC[String, Unit] with ONNXLib {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        // reading ONNX model
        val model = readONNX(model_file)
        val func = model.training_func
        val x_dims = model.x_dims
        val y_dims = model.y_dims


        // fake input and target
        val input = TensorR(Tensor.zeros(x_dims: _*))
        val target = NewArray[Int](x_dims(0))
        for (i <- DataLoop(x_dims(0))) target(i) = 1
        def lossFun(dummy: TensorR) = func(input).nllLossB(target).sum()

        val loss = gradR_loss(lossFun)(Tensor.zeros(1))
        println(loss.data(0))
      }
    }

    val squeezenet_file = new PrintWriter(new File(gene_dir + "squeezenetTraining.cpp"))
    squeezenet_file.println(training_func.code)
    squeezenet_file.flush()
  }

}
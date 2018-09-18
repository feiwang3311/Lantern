package lantern

import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

import scala.collection.mutable.ArrayBuffer
import scala.collection.{Seq => NSeq}
import scala.math._
import scala.collection.mutable.{Map => MMap};

import java.io.PrintWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

import onnx.onnx_ml;

trait ONNXLib extends TensorExp {

<<<<<<< HEAD
  case class readONNX(val model_file: String) {
=======
  def readONNX(model_file: String) = {
>>>>>>> roadblock -- cannot construct training function from ONNX model

    val model = onnx_ml.ModelProto.parseFrom(new FileInputStream(model_file))
    val graph = model.getGraph
    val initializer: Seq[onnx_ml.TensorProto] = graph.initializer
    val inputs: Seq[onnx_ml.ValueInfoProto] = graph.input
    val outputs: Seq[onnx_ml.ValueInfoProto] = graph.output
    val nodes: Seq[onnx_ml.NodeProto] = graph.node

    // TODO (Fei Wang): problem: this function is assuming that the data type is Float, will break if not!!!
    // extract information from TensorProto
    def extract_init(init: onnx_ml.TensorProto): (String, (Seq[Int], onnx_ml.TensorProto.DataType, Array[Float])) = {
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
      (name -> (dims, datatype, floatarray))
    }

    val initializer_map: Map[String, (Seq[Int], onnx_ml.TensorProto.DataType, Array[Float])] =
      initializer.map(init => extract_init(init)).toMap

    val initializer_map_tensor: Map[String, Tensor] =
      initializer_map.map { case (name, (dims, _, value)) => (name -> Tensor(Array((value.map(x=>unit(x)).toSeq: _*)), dims: _*)) }
    val initializer_map_tensorR: Map[String, TensorR] =
      initializer_map_tensor.map { case (name, tensor) => (name -> TensorR(tensor))}

    // extract information from ValueInfoProto
    def extract_value(put: onnx_ml.ValueInfoProto): (String, (Seq[Int], onnx_ml.TensorProto.DataType)) = {
      val name: String = put.getName
      val ty: onnx_ml.TypeProto = put.getType
      val tensor: onnx_ml.TypeProto.Tensor = ty.getTensorType
      val elem_type: onnx_ml.TensorProto.DataType = tensor.getElemType
      val shape: onnx_ml.TensorShapeProto = tensor.getShape
      val dim: Seq[onnx_ml.TensorShapeProto.Dimension] = shape.dim
      val dims: Seq[Int] = (dim.map(x => x.getDimValue)).map(x => x.toInt)
      (name -> (dims, elem_type))
    }

    val input_map: Map[String, (Seq[Int], onnx_ml.TensorProto.DataType)] = inputs.map(i => extract_value(i)).toMap
    val output_map: Map[String, (Seq[Int], onnx_ml.TensorProto.DataType)] = outputs.map(o => extract_value(o)).toMap

    // find out the real input (a entry of input_map that is not in initializer)
    def real_input(): (String, Seq[Int]) = {
      val all_inputs = input_map.keys
      val non_initialized_inputs = all_inputs.filter(k => !initializer_map_tensor.contains(k))
      assert(non_initialized_inputs.size == 1, "there should be one uninitialized input")
      val x_name: String = non_initialized_inputs.head
      val x_dims: Seq[Int] = input_map(x_name)._1.map(x => x.toInt)
      (x_name, x_dims)
    }

    // find out the output
<<<<<<< HEAD
    def real_output(): (String, Seq[Int]) = {
      val out_keys = output_map.keys
      assert (out_keys.size == 1, "we hope that there is only one output for the model")
      val out_key: String = out_keys.head
      val out_dims = output_map(out_key)._1.map(x => x.toInt)
      (out_key, out_dims)
    }

    val (x_name, x_dims) = real_input()
    val (y_name, y_dims) = real_output()
=======
    def real_output(): String = {
      val out_keys = output_map.keys
      assert (out_keys.size == 1, "we hope that there is only one output for the model")
      val out_key: String = out_keys.head
      out_key
    }

    val (x_name, x_dims) = real_input()
    val y_name = real_output()

    // collect basic info of the model, can be used for pretty printing
    val modelMap: MMap[String, Any] = MMap()
    modelMap += ("irversion:" -> model.getIrVersion)
    modelMap += ("producer name:" -> model.getProducerName)
    modelMap += ("producer version:" -> model.getProducerVersion)
    modelMap += ("domain:" -> model.getDomain)
    modelMap += ("model version:" -> model.getModelVersion)
    modelMap += ("doc string:" -> model.getDocString)
    modelMap += ("name of graph:" -> graph.getName)
    modelMap += ("number of initializer:" -> initializer.size)
    modelMap += ("number of inputs:" -> inputs.size)
    modelMap += ("number of outputs:" -> outputs.size)
    modelMap += ("number of nodes:" -> nodes.size)
>>>>>>> roadblock -- cannot construct training function from ONNX model

    abstract class Node
    case class convNode(inputs: Seq[String], output: String, attributes: Map[String, Seq[Int]]) extends Node
    case class reluNode(input: String, output: String) extends Node
    case class maxpoolNode(input: String, output: String, attributes: Map[String, Seq[Int]]) extends Node
    case class concatNode(inputs: Seq[String], output: String, axis: Int) extends Node
    case class dropoutNode(input: String, outputs: Seq[String], ratio: Float, is_test: Int) extends Node
    case class globalAveragePoolNode(input: String, output: String) extends Node
    case class softmaxNode(input: String, output: String) extends Node

    val allNodes: Seq[Node] = nodes.map { node =>
      node.getOpType match {

        case "Conv" => {

          val inputs: Seq[String] = node.input
          assert (inputs.size == 3, "number of inputs of a conv node should always be 3")

          val outputs: Seq[String] = node.output
          assert (outputs.size == 1, "number of output of a conv node should always be 1")

          val attributes: Seq[onnx_ml.AttributeProto] = node.attribute
          assert (attributes.size == 3, "number of attributes of a conv node should always be 3")
          val atts: Map[String, Seq[Int]] = attributes.map(att => att.getName -> att.ints.map(x => x.toInt)).toMap
          assert (atts.contains("strides"), "attributes of a conv node should have strides")
          assert (atts.contains("pads"), "attributes of a conv node should have pads")
          assert (atts.contains("kernel_shape"), "attributes of a conv node should have kernel_shape")
          assert(atts("strides").size == 2, "strides should be length 2")
          assert(atts("kernel_shape").size == 2, "kernel_shape should be length 2")

          convNode(inputs, outputs.head, atts)
        }

        case "Relu" => {

          val inputs: Seq[String] = node.input
          assert (inputs.size == 1, "number of inputs of a relu node should always be 1")

          val outputs: Seq[String] = node.output
          assert (outputs.size == 1, "number of outputs of a relu node should always be 1")

          reluNode(inputs.head, outputs.head)
        }

        case "MaxPool" => {

          val inputs: Seq[String] = node.input
          assert (inputs.size == 1, "number of inputs of a maxpool node should always be 1")

          val outputs: Seq[String] = node.output
          assert (outputs.size == 1, "number of outputs of a maxpool node should always be 1")

          val attributes: Seq[onnx_ml.AttributeProto] = node.attribute
          assert (attributes.size == 3, "number of attributes of a conv node should always be 3")
          val atts: Map[String, Seq[Int]] = attributes.map(att => att.getName -> att.ints.map(x => x.toInt)).toMap
          assert (atts.contains("strides"), "attributes of a conv node should have strides")
          assert (atts.contains("pads"), "attributes of a conv node should have pads")
          assert (atts.contains("kernel_shape"), "attributes of a conv node should have kernel_shape")
          assert(atts("strides").size == 2, "strides should be length 2")
          assert(atts("kernel_shape").size == 2, "kernel_shape should be length 2")
          // TODO: (Fei Wang) erroneous code, the implementation assumes that pads are all 0

          maxpoolNode(inputs.head, outputs.head, atts)
        }

        case "Concat" => {

          val inputs: Seq[String] = node.input
          assert (inputs.size > 1, "number of inputs for concat node should be larger than 1")

          val outputs: Seq[String] = node.output
          assert (outputs.size == 1, "number of outputs for concat node should be 1")

          val attributes: Seq[onnx_ml.AttributeProto] = node.attribute
          assert (attributes.size == 1, "number of attributes of a concat node should be 1")
          val axis: Int = attributes.head.getI.toInt

          concatNode(inputs, outputs.head, axis)
        }

        case "Dropout" => {

          val inputs: Seq[String] = node.input
          assert (inputs.size == 1, "number of inputs for dropout node should always be 1")

          val outputs: Seq[String] = node.output
          assert (outputs.size == 2, "number of outputs for dropout node should always be 2")

          val attributes: Seq[onnx_ml.AttributeProto] = node.attribute
          assert (attributes.size == 2, "number of attributes for dropout node should be 2")
          val ratio: Float = if (attributes.head.name == "ratio") attributes.head.getF else attributes.last.getF
          val is_test: Int = (if (attributes.last.name == "is_test") attributes.last.getI else attributes.head.getI).toInt
          // TODO: (Fei Wang) warning - the is_test is not considered by the implementation (Don't know what it means!!)
          // TODO: (Fei Wang) for inference, should dropout be ignored??

          dropoutNode(inputs.head, outputs, ratio, is_test)
        }

        case "GlobalAveragePool" => {

          val inputs: Seq[String] = node.input
          assert (inputs.size == 1, "number of inputs for global_average_pool should be 1")

          val outputs: Seq[String] = node.output
          assert (outputs.size == 1, "number of outputs for global_average_pool should be 1")

          globalAveragePoolNode(inputs.head, outputs.head)
        }

        case "Softmax" => {

          val inputs: Seq[String] = node.input
          assert (inputs.size == 1, "number of inputs for softmax node should 1")

          val outputs: Seq[String] = node.output
          assert (outputs.size == 1, "number of outputs for softmax node should be 1")

          softmaxNode(inputs.head, outputs.head)
        }

        case _ => throw new RuntimeException("Node not yet implemented")
      }
    }

<<<<<<< HEAD
    // collect basic info of the model, can be used for pretty printing
    val modelMap: Map[String, Any] = Map(
      "irversion:" -> model.getIrVersion,
      "producer name:" -> model.getProducerName,
      "producer version:" -> model.getProducerVersion,
      "domain:" -> model.getDomain,
      "model version:" -> model.getModelVersion,
      "doc string:" -> model.getDocString,
      "name of graph:" -> graph.getName,
      "number of initializer:" -> initializer.size,
      "number of inputs:" -> inputs.size,
      "number of outputs:" -> outputs.size,
      "number of nodes:" -> nodes.size,
      "all nodes:" -> allNodes,
    )

    // read the nodes and build the function for inference
    lazy val inference_func: (Tensor => Tensor) = { x: Tensor =>
=======
    // read the nodes and build the function for inference
    val inference_func: (Tensor => Tensor) = { x: Tensor =>
>>>>>>> roadblock -- cannot construct training function from ONNX model

      assert(x.dimsSeq == x_dims, "input tensor is not of the correct dimensions")

      // generate Tensors (or TensorRs) of intermediate steps and inputs
      val intermediate_map_tensor: MMap[String, Tensor] = MMap()
      intermediate_map_tensor += (x_name -> x)

      // TODO (Fei Wang): ask Greg, is there a better way to do this?
      def get_from_two_maps(key: String) = {
        initializer_map_tensor.get(key) match {
          case Some(v) => v
          case None => intermediate_map_tensor.get(key) match {
            case Some(v) => v
            case None => throw new RuntimeException(key + " is not found in either maps")
          }
        }
      }

      allNodes.foreach { node =>

        node match {

          case convNode(inputs, output, atts) => {

            val input1 = get_from_two_maps(inputs.head)
            val input2 = get_from_two_maps(inputs.tail.head)
            val input3 = get_from_two_maps(inputs.last)

            val strides = atts("strides")
            val pads = atts("pads")
            val kernel_shape = atts("kernel_shape")  // this attribute is actually not used

            val out = input1.conv2D_batch(input2, input3, strides, pads)
            intermediate_map_tensor += (output -> out)
          }

          case reluNode(input, output) => {

            val in = get_from_two_maps(input)
            val out = in.relu()
            intermediate_map_tensor += (output -> out)
          }

          case maxpoolNode(input, output, atts) => {

            val in = get_from_two_maps(input)

            val strides = atts("strides")
            val pads = atts("pads")
            val kernel_shape = atts("kernel_shape")

            // TODO: (Fei Wang) erroneous code, the implementation assumes that pads are all 0
            val (out, _) = in.maxPool_k_batch(kernel_shape, strides)
            intermediate_map_tensor += (output -> out)
          }

          case concatNode(inputs, output, axis) => {

            val input_s = inputs.map(x => get_from_two_maps(x))
            val out = input_s.head.concat(axis, input_s.tail: _*)
            intermediate_map_tensor += (output -> out)
          }

          case dropoutNode(input, outputs, ratio, is_test) => {

            val in = get_from_two_maps(input)
            val (out1, out2) = in.dropout(ratio)
            intermediate_map_tensor += (outputs.head -> out1)
            intermediate_map_tensor += (outputs.last -> out2)
          }

          case globalAveragePoolNode(input, output) => {

            val in = get_from_two_maps(input)
            val out = in.global_ave_batch()
            intermediate_map_tensor += (output -> out)
          }

          case softmaxNode(input, output) => {

            val in = get_from_two_maps(input)
            val out = in.softmax_batch()
            intermediate_map_tensor += (output -> out)
          }

          case _ => throw new RuntimeException("not yet implemented")
        }
      }

      intermediate_map_tensor(y_name)
    }

    // read the nodes and build the function for training

    lazy val training_func: (TensorR => TensorR @diff) = { x: TensorR =>

      assert(x.x.dimsSeq == x_dims, "input tensor is not of the correct dimensions")

      // generate Tensors (or TensorRs) of intermediate steps and inputs
      val intermediate_map_tensorR: MMap[String, TensorR] = MMap()
      intermediate_map_tensorR.update(x_name, x)

      // TODO (Fei Wang): ask Greg, is there a better way to do this?
      def get_from_two_maps(key: String): TensorR = {
        initializer_map_tensorR.get(key) match {
          case Some(v) => v
          case None => intermediate_map_tensorR.get(key) match {
            case Some(v) => v
            case None => throw new RuntimeException(key + " is not found in either maps")
          }
        }
      }

      val iter = allNodes.iterator

      while (iter.hasNext) {

        val node = iter.next

        if (node.isInstanceOf[convNode]) {
          val convNode(inputs, output, atts) = node
          val input1 = get_from_two_maps(inputs.head)
          val input2 = get_from_two_maps(inputs.tail.head)
          val input3 = get_from_two_maps(inputs.last)

          val strides = atts("strides")
          val pads = atts("pads")
          val kernel_shape = atts("kernel_shape")  // this attribute is actually not used

          val out = input1.convBBP(input2, input3, strides, pads)
          intermediate_map_tensorR.update(output, out)
        } else if (node.isInstanceOf[reluNode]) {
          val reluNode(input, output) = node
          val in = get_from_two_maps(input)
          val out = in.relu()
          intermediate_map_tensorR.update(output, out)
        } else if (node.isInstanceOf[maxpoolNode]) {
          val maxpoolNode(input, output, atts) = node
          val in = get_from_two_maps(input)
          val strides = atts("strides")
          val pads = atts("pads")
          val kernel_shape = atts("kernel_shape")
          // TODO: (Fei Wang) erroneous code, the implementation assumes that pads are all 0
          val out = in.maxPoolBK(kernel_shape, strides)
          intermediate_map_tensorR.update(output, out)
        } else if (node.isInstanceOf[concatNode]) {
          val concatNode(inputs, output, axis) = node
          val input_s = inputs.map(x => get_from_two_maps(x))
          val out = input_s.head.concat(axis, input_s.tail: _*)
          intermediate_map_tensorR.update(output, out)
        } else if (node.isInstanceOf[dropoutNode]) {

          val dropoutNode(input, outputs, ratio, is_test) = node
          val in = get_from_two_maps(input)
          val out = in.dropout(ratio)
          intermediate_map_tensorR.update(outputs.head, out)
          // intermediate_map_tensor += (outputs.last -> out2)

        } else if (node.isInstanceOf[globalAveragePoolNode]) {

          val globalAveragePoolNode(input, output) = node
          val in = get_from_two_maps(input)
          val out = in.global_ave_batch()
          intermediate_map_tensorR.update(output, out)

        } else if (node.isInstanceOf[softmaxNode]) {

          val softmaxNode(input, output) = node
          val in = get_from_two_maps(input)
          val out = in.logSoftmaxB()
          intermediate_map_tensorR.update(output, out)

        } else {

          shift{ (k: Tensor => Unit) => ???}
        }
      }

      intermediate_map_tensorR(y_name)
    }

    // TODO: (Fei Wang) define nicer API for inferencing and training
  }

}
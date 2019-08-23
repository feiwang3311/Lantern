package lantern

import scala.util.continuations._
import scala.util.continuations

import scala.collection.mutable.ArrayBuffer
import scala.math._
import scala.collection.mutable.{Map => MMap};
import scala.io.Source
import scala.annotation.tailrec
import scala.util.{Try, Success, Failure}

import java.nio._
import java.nio.file._
import java.io._
import java.util.Scanner;

import org.bytedeco.javacpp._;
import org.bytedeco.javacpp.onnx._;

import lms.core.stub._
import lms.macros.SourceContext
import lms.core.virtualize

trait ONNXLib extends TensorDsl with ScannerOps {

  object ParseHelper {

    def getProtoProps[T](size: Int, propMethod: Int => T): Seq[T] = {
      ((0 until size): Range).map(y => propMethod(y.toInt)).toSeq
    }

    def toInts(ba: Array[Byte]): Seq[Int] = {
      val bs = new ByteArrayInputStream(ba)
      val ds = new DataInputStream(bs)
      val ints = toBigEndiansInts(ds)
      bs.close()
      ds.close()
      ints
    }

    def toFloats(ba: Array[Byte]): Seq[Float] = {
      val bs = new ByteArrayInputStream(ba)
      val ds = new DataInputStream(bs)
      val floats = toBigEndians(ds)
      bs.close()
      ds.close()
      floats
    }

    def toBigEndiansInts(stream: DataInputStream): Seq[Int] = {
      val bf = streamToByteBuffer(stream); bf.rewind()
      val intBuffer = bf.order(ByteOrder.LITTLE_ENDIAN).asLongBuffer();
      val n = intBuffer.remaining

      @tailrec
      def intBufferToArray_(idx: Int, ints: Array[Int]): Array[Int] = {
        if (intBuffer.hasRemaining) {
          ints(idx) = intBuffer.get.toInt
          intBufferToArray_(idx + 1, ints)
        } else ints
      }
      val intArray = scala.Array.ofDim[Int](n)
      intBufferToArray_(0, intArray)
    }

    def toBigEndians(stream: DataInputStream): Seq[Float] = {

      val bf = streamToByteBuffer(stream); bf.rewind()

      // when we read, we want to get it in BIG_ENDIAN
      val floatBuffer = bf.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
      val n = floatBuffer.remaining

      @tailrec
      def floatBufferToArray_(idx: Int, floats: Array[Float]):  Array[Float] = {
        if (floatBuffer.hasRemaining) {
          // floatBuffer.get returns current an increments position
          floats(idx) = floatBuffer.get
          floatBufferToArray_(idx + 1, floats)
        }
        else floats
      }
      // allocate result float array
      val floatArray = scala.Array.ofDim[Float](n)
      floatBufferToArray_(0, floatArray)
    }

    def streamToByteBuffer(stream: DataInputStream): ByteBuffer = {
      @tailrec
      def streamToByteBuffer_(stream: DataInputStream,
                             bf: ByteBuffer): ByteBuffer = {
          Try(bf.put(stream.readByte())) match {
              case Success(_) => streamToByteBuffer_(stream, bf)
              case Failure(ex) if ex.isInstanceOf[EOFException] => bf
              case Failure(ex) => throw ex
          }
      }
      // pre-allocate with the size of buffer
      val bf = ByteBuffer.allocateDirect(stream.available)
        streamToByteBuffer_(stream, bf)
    }

    // TODO (Fei Wang): problem: this function is assuming that the data type is Float, will break if not!!!
    def extract_init(init: TensorProto): (String, (Seq[Int], Int, Array[Float])) = {
      val dims: Seq[Int] = ParseHelper.getProtoProps(init.dims_size , init.dims(_)).map(x => x.toInt)
      val name: String = if(init.name != null) init.name.getString else ""
      val datatype: Int = init.data_type
      if (datatype == TensorProto.FLOAT) {
        val rawdata: org.bytedeco.javacpp.BytePointer = init.raw_data
        val bytes = new Array[Byte](rawdata.asByteBuffer.remaining())
        rawdata.asByteBuffer.get(bytes, 0, bytes.length);
        val floatarray: Array[Float] = toFloats(bytes).toArray
        // make sure that the initialization values correspond with the dims
        assert(floatarray.length == dims.fold(1)(_ * _), s"${floatarray.length} != $dims.fold(1)")
        (name, (dims, datatype, floatarray))
      } else if (datatype == TensorProto.INT64) {
        val rawdata: org.bytedeco.javacpp.BytePointer = init.raw_data
        val bytes = new Array[Byte](rawdata.asByteBuffer.remaining())
        rawdata.asByteBuffer.get(bytes, 0, bytes.length);
        val intarray: Array[Int] = toInts(bytes).toArray
        // val intarray: Array[Int] = toInts(rawdata.asByteBuffer.array).toArray
        assert(intarray.length == dims.product, s"${intarray.length} != $dims.product")
        (name, (dims, datatype, intarray.map(_.toFloat)))
      } else {
        System.out.println(init.toString)
        throw new RuntimeException("data type not Float, Not handling yet: " + datatype)
      }
    }

    // extract information from ValueInfoProto
    def extract_value(put: ValueInfoProto): (String, (Seq[Int], Int)) = {
      val name: String = put.name.getString
      val ty: TypeProto = put.`type`()
      val tensor: TypeProto_Tensor = ty.tensor_type
      val elem_type: Int = tensor.elem_type
      val shape: TensorShapeProto = tensor.shape
      val dim: Seq[Dimension] = ParseHelper.getProtoProps(shape.dim_size , shape.dim(_))
      val dims: Seq[Int] = (dim.map(x => x.dim_value)).map(x => x.toInt)
      (name -> (dims, elem_type))
    }

  }

  case class readTensor(val tensor_file: String) {
    val byteArray = Files.readAllBytes(Paths.get(tensor_file))

    val tensorInput = new TensorProto()
    ParseProtoFromBytes(tensorInput.asInstanceOf[MessageLite],
                      new BytePointer(byteArray: _*),
                      byteArray.length.toLong)

    val (_, (dims, _, floatarray)) = ParseHelper.extract_init(tensorInput)
    val tensor: Tensor = Tensor(Array((floatarray.map(x=>unit(x)).toSeq: _*)), dims: _*)
  }

  case class ParameterWriter(val filename: String) {
    val output = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filename)))
    var off = 0
    def addParams(arr: Array[Byte]) = {
      output.write(arr, 0, arr.length)
      val offset = off
      off += arr.length
      offset / 4
    }
    def record_init(init: TensorProto): (String, (Seq[Int], Int, Int)) = {
      val dims: Seq[Int] = ParseHelper.getProtoProps(init.dims_size , init.dims(_)).map(x => x.toInt)
      val name: String = if (init.name == null) "" else init.name.getString
      val datatype: Int = init.data_type
      if (datatype == TensorProto.FLOAT) {
        val rawdata: org.bytedeco.javacpp.BytePointer = init.raw_data
        val bytes = new Array[Byte](rawdata.asByteBuffer.remaining())
        rawdata.asByteBuffer.get(bytes, 0, bytes.length);
        val offset: Int = addParams(bytes)
        (name -> (dims, datatype, offset))
      } else throw new RuntimeException("data type not Float, Not handling yet: " + datatype)
    }
    def close() = output.close()
  }

  case class ParameterReader(val filename: String) {
    val fd = open(filename)
    val parameters: Rep[Array[Float]] = mmap[Float](fd, filelen(fd))
    def getOffset(offset: Int) = slice(parameters, offset)
  }

  case class readONNX(val model_file: String) {
    val byteArray = Files.readAllBytes(Paths.get(model_file))

    val model = new ModelProto()
    ParseProtoFromBytes(model.asInstanceOf[MessageLite],
                        new BytePointer(byteArray: _*),
                        byteArray.length.toLong)

    val graph = model.graph

    val initializer: Seq[TensorProto] = ParseHelper.getProtoProps(graph.initializer_size, graph.initializer(_))
    val inputs: Seq[ValueInfoProto] = ParseHelper.getProtoProps(graph.input_size, graph.input(_))
    val outputs: Seq[ValueInfoProto] = ParseHelper.getProtoProps(graph.output_size, graph.output(_))
    val nodes: Seq[NodeProto] = ParseHelper.getProtoProps(graph.node_size, graph.node(_))

    // partition initializer by data type
    val (float_init, int_init) = initializer.partition(_.data_type == TensorProto.FLOAT)

    // intMap are tensors that hold shape informations.
    // The Seq[Int] is the rank, and will always be size 1 or 0 (for scalar).
    // The Seq[Rep[Int]] is the shape, which is unknown at staging time
    val intMap: MMap[String, (Seq[Int], Seq[Rep[Int]])] =
      MMap(int_init.map(ParseHelper.extract_init(_)).map{
        case (name, (dims, dt, arr: Array[Float])) =>
          val resAfter: Seq[Rep[Int]] = arr.map(_.toInt).toSeq.map(unit(_))
          ( name -> (dims, resAfter) )
      }.toMap.toSeq: _*)

    // record all float parameters
    val parameterFileName = model_file + ".bin"
    val writer = ParameterWriter(parameterFileName)
    val byteMap: Map[String, (Seq[Int], Int, Int)] =
      float_init.map(init => writer.record_init(init)).toMap
    writer.close()

    // set up reading from parameters
    // because the parameterFileName is also used in the generated file for reading in parameters, we must make sure (for robustness) that the path is absolute
    val whereami = System.getProperty("user.dir")
    val absoluteParameterFileName = if (parameterFileName startsWith "/") parameterFileName
                                    else new File(System.getProperty("user.dir"), parameterFileName).getPath
    val reader = ParameterReader(absoluteParameterFileName)

    val input_map: Map[String, (Seq[Int], Int)] = inputs.map(i => ParseHelper.extract_value(i)).toMap
    val output_map: Map[String, (Seq[Int], Int)] = outputs.map(o => ParseHelper.extract_value(o)).toMap

    // find out the real input (a entry of input_map that is not in initializer)
    def real_input(): (String, Seq[Int]) = {
      val all_inputs = input_map.keys
      val init_names = byteMap.map{ case (name, _) => name}.toSet
      val non_initialized_inputs = all_inputs.filter(k => !init_names.contains(k) && !intMap.contains(k))
      assert(non_initialized_inputs.size == 1, s"there should be one uninitialized input, got ${non_initialized_inputs}")
      val x_name: String = non_initialized_inputs.head
      val x_dims: Seq[Int] = input_map(x_name)._1.map(x => x.toInt)
      (x_name, x_dims)
    }

    // find out the output
    def real_output(): (String, Seq[Int]) = {
      val out_keys = output_map.keys
      assert (out_keys.size == 1, "we hope that there is only one output for the model")
      val out_key: String = out_keys.head
      val out_dims = output_map(out_key)._1.map(x => x.toInt)
      (out_key, out_dims)
    }

    val (x_name, x_dims) = real_input()
    val (y_name, y_dims) = real_output()

    abstract class Node
    case class convNode(inputs: Seq[String], output: String, attributes: Map[String, Seq[Int]]) extends Node
    case class bnNode(inputs: Seq[String], output: String, attributeMap: Map[String, Float]) extends Node
    case class sumNode(inputs: Seq[String], output: String) extends Node
    case class reluNode(input: String, output: String) extends Node
    case class maxpoolNode(input: String, output: String, attributes: Map[String, Seq[Int]]) extends Node
    case class averagePoolNode(input: String, output: String, attributes: Map[String, Seq[Int]]) extends Node
    case class concatNode(inputs: Seq[String], output: String, axis: Int) extends Node
    case class dropoutNode(input: String, outputs: Seq[String], ratio: Float) extends Node
    case class globalAveragePoolNode(input: String, output: String) extends Node
    case class softmaxNode(input: String, output: String) extends Node
    case class reshapeNode(inputs: Seq[String], output: String) extends Node
    case class gemmNode(inputs: Seq[String], output: String, attInts: Map[String, Int], attFloats: Map[String, Float]) extends Node
    case class flattenNode(input: String, output: String, axis: Int) extends Node
    case class addNode(inputs: Seq[String], output: String) extends Node
    case class padNode(input: String, output: String, mode: String, pads: List[Int], value: Float) extends Node
    case class shapeNode(input: String, output: String) extends Node
    case class sliceNode(input: String, output: String, attMap: Map[String, List[Int]]) extends Node
    case class squeezeNode(input: String, output: String, axes: List[Int]) extends Node
    case class unsqueezeNode(input: String, output: String, axes: List[Int]) extends Node
    case class constantNode(output: String, data: Float) extends Node
    case class gatherNode(input: String, output: String, axis: Int) extends Node

    def getAttributeProtoInts(attribute: AttributeProto): Seq[Long] = {
      ((0 until attribute.ints_size): Range).map(y => attribute.ints(y.toInt)).toSeq
    }


    def getConvMaxPAvPAttr(attributes: Seq[AttributeProto]): Map[String, Seq[Int]] = {
      val atts: Map[String, Seq[Int]] = attributes.map{att =>
        if (att.name.getString == "group") att.name.getString -> Seq(att.i.toInt)
        else att.name.getString -> getAttributeProtoInts(att).map(x => x.toInt)
      }.toMap
      assert (atts.contains("strides"), "attributes of a conv/maxP/avP node should have strides")
      assert (atts.contains("kernel_shape"), "attributes of a conv/maxP/avP node should have kernel_shape")
      assert(atts("strides").size == 2, "strides should be length 2")
      assert(atts("kernel_shape").size == 2, "kernel_shape should be length 2")
      if (atts.contains("pads")) atts else atts + (("pads", Seq(0,0,0,0)))
    }

    def getInputs(node: NodeProto) = {
      ParseHelper.getProtoProps(node.input_size, node.input(_).getString)
    }

    def getOutputs(node: NodeProto) = {
      ParseHelper.getProtoProps(node.output_size, node.output(_).getString)
    }

    def getAttributes(node: NodeProto) = {
      ParseHelper.getProtoProps(node.attribute_size, node.attribute(_))
    }


    val allNodes: Seq[Node] = nodes.map { node =>
      node.op_type.getString match {

        case "Conv" => {
          val inputs: Seq[String] = getInputs(node)
          assert (inputs.size == 2 || inputs.size == 3, s"number of inputs of a conv node should always be 2 or 3, got ${inputs.size}")

          val outputs: Seq[String] = getOutputs(node)
          assert (outputs.size == 1, "number of output of a conv node should always be 1")

          val attributes: Seq[AttributeProto] = getAttributes(node)
          convNode(inputs, outputs.head, getConvMaxPAvPAttr(attributes))
        }

        case "BatchNormalization" => {
          val inputs: Seq[String] = getInputs(node)

          assert (inputs.size == 5, s"Number of inputs for a batch normalization layer should be 5, got ${inputs.size}")

          val outputs: Seq[String] = getOutputs(node)
          assert (outputs.size == 1, s"number of outputs of a batch normalization layer should be 1, got ${outputs.size}")

          val attributes: Seq[AttributeProto] = getAttributes(node)

          val attributeMap: Map[String, Float] = attributes.map { att =>
            att.name.getString match {
              case "epsilon" => ("epsilon" -> att.f.toFloat)
              case "is_test" => ("is_test" -> att.i.toFloat)
              case "momentum" => ("momentum" -> att.f.toFloat)
              case other => System.out.println(s"not yet handling BatchNormalization attributes $other"); ???
            }
          }.toMap
          bnNode(inputs, outputs.head, attributeMap)
        }

        case "Sum" => {
          val inputs: Seq[String] = getInputs(node)
          assert (inputs.size == 2, s"number of inputs for Sum layer should be 2, got ${inputs.size}")

          val outputs: Seq[String] = getOutputs(node)
          assert (outputs.size == 1, s"number of outputs for Sum layer should be 1, got ${outputs.size}")

          sumNode(inputs, outputs.head)
        }

        case "Relu" => {

          val inputs: Seq[String] = getInputs(node)
          assert (inputs.size == 1, "number of inputs of a relu node should always be 1")

          val outputs: Seq[String] = getOutputs(node)
          assert (outputs.size == 1, "number of outputs of a relu node should always be 1")

          reluNode(inputs.head, outputs.head)
        }

        case "MaxPool" => {

          val inputs: Seq[String] = getInputs(node)
          assert (inputs.size == 1, "number of inputs of a maxpool node should always be 1")

          val outputs: Seq[String] = getOutputs(node)
          assert (outputs.size == 1, "number of outputs of a maxpool node should always be 1")

          val attributes: Seq[AttributeProto] = getAttributes(node)

          maxpoolNode(inputs.head, outputs.head, getConvMaxPAvPAttr(attributes))
        }

        case "Concat" => {

          val inputs: Seq[String] = getInputs(node)
          assert (inputs.size > 1, "number of inputs for concat node should be larger than 1")

          val outputs: Seq[String] = getOutputs(node)
          assert (outputs.size == 1, "number of outputs for concat node should be 1")

          val attributes: Seq[AttributeProto] = getAttributes(node)
          assert (attributes.size == 1, "number of attributes of a concat node should be 1")
          val axis: Int = attributes.head.i.toInt

          concatNode(inputs, outputs.head, axis)
        }

        case "Dropout" => {

          val inputs: Seq[String] = getInputs(node)
          assert (inputs.size == 1, "number of inputs for dropout node should always be 1")

          val outputs: Seq[String] = getOutputs(node)
          assert (outputs.size == 2, "number of outputs for dropout node should always be 2")

          val attributes: Seq[AttributeProto] = getAttributes(node)
          assert (attributes.size == 1, "number of attributes for drop out node should be 1")
          val ratio: Float = attributes.head.f

          dropoutNode(inputs.head, outputs, ratio)
        }

        case "GlobalAveragePool" => {

          val inputs: Seq[String] = getInputs(node)
          assert (inputs.size == 1, "number of inputs for global_average_pool should be 1")

          val outputs: Seq[String] = getOutputs(node)
          assert (outputs.size == 1, "number of outputs for global_average_pool should be 1")

          globalAveragePoolNode(inputs.head, outputs.head)
        }

        case "AveragePool" => {
          val inputs: Seq[String] = getInputs(node)
          assert (inputs.size == 1, s"number of inputs for AveragePool layer should be 1, got ${inputs.size}")

          val outputs: Seq[String] = getOutputs(node)
          assert (outputs.size == 1, s"number of outputs for AveragePool layer should be 1, got ${outputs.size}")

          val attributes: Seq[AttributeProto] = getAttributes(node)
          averagePoolNode(inputs.head, outputs.head, getConvMaxPAvPAttr(attributes))
        }

        case "Softmax" => {

          val inputs: Seq[String] = getInputs(node)
          assert (inputs.size == 1, "number of inputs for softmax node should 1")

          val outputs: Seq[String] = getOutputs(node)
          assert (outputs.size == 1, "number of outputs for softmax node should be 1")

          softmaxNode(inputs.head, outputs.head)
        }

        case "Reshape" => {
          val inputs: Seq[String] = getInputs(node)
          assert (inputs.size == 2, s"number of inputs for Reshape layer should be 2")

          val outputs: Seq[String] = getOutputs(node)
          assert (outputs.size == 1, s"number of outputs for Reshape layer should be 1")

          reshapeNode(inputs, outputs.head)
        }

        case "Gemm" => {
          val inputs: Seq[String] = getInputs(node)
          assert (inputs.size == 3, s"number of inputs for Gemm layer should be 3, got ${inputs.size}")

          val outputs: Seq[String] = getOutputs(node)
          assert (outputs.size == 1, s"number of outputs for Gemm layer should be 1, got ${outputs.size}")

          val attributes: Seq[AttributeProto] = getAttributes(node)

          val attsInts = attributes.filter(att => att.name.getString == "transA" || att.name.getString == "transB").map(att => att.name.getString -> att.i.toInt).toMap
          val attsFloats = attributes.filter(att => att.name.getString == "alpha" || att.name.getString == "beta").map(att => att.name.getString -> att.f.toFloat).toMap
          // val transB: Int = attributes.head.getI.toInt

          gemmNode(inputs, outputs.head, attsInts, attsFloats)
        }

        case "Flatten" => {
          val inputs: Seq[String] = getInputs(node)
          assert (inputs.size == 1, s"number of inputs for Flatten layer should be 1, got ${inputs.size}")

          val outputs: Seq[String] = getOutputs(node)
          assert (outputs.size == 1, s"number of outputs for Flatten layer should be 1, got ${outputs.size}")

          val attributes: Seq[AttributeProto] = getAttributes(node)
          assert (attributes.size == 1, "number of attributes of a Flatten node should be 1")
          val axis: Int = attributes.head.i.toInt

          flattenNode(inputs.head, outputs.head, axis)
        }

        case "Add" => {
          val inputs : Seq[String] = getInputs(node)
          assert(inputs.size == 2, s"number of inputs for Add node should be 2, got ${inputs.size}")
          val outputs: Seq[String] = getOutputs(node)
          assert(outputs.size == 1, s"number of outputs for Add node should be 1, got ${outputs.size}")
          addNode(inputs, outputs.head)
        }

        case "Pad" => {
          val inputs: Seq[String] = getInputs(node)
          assert(inputs.size == 1, s"number of inputs for Pad node should be 1, got ${inputs.size}")
          val outputs: Seq[String] = getOutputs(node)
          assert(outputs.size == 1, s"number of outputs for Pad node should be 1, got ${outputs.size}")
          val attributes: Seq[AttributeProto] = getAttributes(node)
          assert(attributes.size == 3, s"number of attributes of a Pad node should be 3, got ${attributes.size}")
          val sortedAttr = attributes.sortBy { att => att.name.getString match {
            case "mode" => 1
            case "pads" => 2
            case "value" => 3
          }}
          val mode: String = sortedAttr(0).s.toString()
          val pads: List[Int] = getAttributeProtoInts(sortedAttr(1)).toList.map(l => l.toInt)
          val value: Float = sortedAttr(2).f.toFloat
          padNode(inputs.head, outputs.head, mode, pads, value)
        }

        case "Shape" => {
          val inputs: Seq[String] = getInputs(node)
          val outputs: Seq[String] = getOutputs(node)
          assert(inputs.size == 1, s"inputs should be size 1 for shapeNode, got ${inputs.size}")
          assert(outputs.size == 1, s"outputs should be size 1 for shapeNode, got ${outputs.size}")
          shapeNode(inputs.head, outputs.head)
        }

        case "Slice" => {
          val inputs: Seq[String] = getInputs(node)
          val outputs: Seq[String] = getOutputs(node)
          assert(inputs.size == 1, s"inputs should be size 1 for sliceNode, got ${inputs.size}")
          assert(outputs.size == 1, s"outputs should be size 1 for sliceNode, got ${outputs.size}")
          val attributes: Seq[AttributeProto] = getAttributes(node)
          val attMap: Map[String, List[Int]] = attributes.map {att =>
            att.name.getString -> getAttributeProtoInts(att).toList.map(_.toInt)
          }.toMap
          sliceNode(inputs.head, outputs.head, attMap)
        }

        case "Squeeze" => {
          val inputs: Seq[String] = getInputs(node)
          val outputs: Seq[String] = getOutputs(node)
          assert(inputs.size == 1, s"inputs should be size 1 for squeezeNode, got ${inputs.size}")
          assert(outputs.size == 1, s"outputs should be size 1 for squeezeNode, got ${outputs.size}")
          val attributes: Seq[AttributeProto] = getAttributes(node)
          assert(attributes.size ==1, s"size of attributes should be 1 for squeezeNode, got ${attributes.size}")
          val axes: List[Int] = getAttributeProtoInts(attributes.head).toList.map(_.toInt)
          squeezeNode(inputs.head, outputs.head, axes)
        }

        case "Unsqueeze" => {
          val inputs: Seq[String] = getInputs(node)
          val outputs: Seq[String] = getOutputs(node)
          assert(inputs.size == 1, s"inputs should be size 1 for unsqueezeNode, got ${inputs.size}")
          assert(outputs.size == 1, s"outputs should be size 1 for unsqueezeNode, got ${outputs.size}")
          val attributes: Seq[AttributeProto] = getAttributes(node)
          assert(attributes.size ==1, s"size of attributes should be 1 for unsqueezeNode, got ${attributes.size}")
          val axes: List[Int] = getAttributeProtoInts(attributes.head).toList.map(_.toInt)
          unsqueezeNode(inputs.head, outputs.head, axes)
        }

        case "Constant" => {
          val outputs : Seq[String] = getOutputs(node)
          assert(outputs.size == 1, s"outputs should be size 1 for constantNode, got ${outputs.size}")
          val attributes: Seq[AttributeProto] = getAttributes(node)
          assert(attributes.size ==1, s"size of attributes should be 1 for constantNode, got ${attributes.size}")
          val value: TensorProto = attributes.head.t
          val (_, (dims, _, data)): (String, (Seq[Int], Int, Array[Float])) = ParseHelper.extract_init(value)
          assert(dims.product == 1, s"constant node should have tensor with single value, got ${dims}")
          constantNode(outputs.head, data(0))
        }

      case "Gather" => {
        val inputs: Seq[String] = getInputs(node)
        val outputs: Seq[String] = getOutputs(node)
        assert(inputs.size == 1, s"inputs should be size 1 for GatherNode, got ${inputs.size}")
        assert(outputs.size == 1, s"outputs should be size 1 for GatherNode, got ${outputs.size}")
        val attributes: Seq[AttributeProto] = getAttributes(node)
        assert(attributes.size == 1, s"attributes size should be 1 for GatherNode, got ${attributes.size}")
        gatherNode(inputs.head, outputs.head, attributes.head.i().toInt) // the last parameter is the axis
      }

        case n =>
          System.out.println(node.toString)
          throw new RuntimeException(s"Node $n not yet implemented")
      }
    }

    // collect basic info of the model, can be used for pretty printing
    val modelMap: Map[String, Any] = Map(
      "irversion:" -> model.ir_version,
      "producer name:" -> model.producer_name,
      "producer version:" -> model.producer_version,
      "domain:" -> model.domain,
      "model version:" -> model.model_version,
      "doc string:" -> model.doc_string,
      "name of graph:" -> graph.name.getString,
      "number of initializer:" -> initializer.size,
      "number of inputs:" -> inputs.size,
      "number of outputs:" -> outputs.size,
      "number of nodes:" -> nodes.size,
      "all nodes:" -> allNodes,
    )

    lazy val initializer_map_tensor: Map[String, Tensor] =
      byteMap.map{ case (name, (dims, dt, offset)) => (name -> Tensor(reader.getOffset(offset), dims: _*)) }
    // lazy val initializer_map_tensorR: Map[String, TensorR] =
    //   initializer_map_tensor.map { case (name, tensor) => (name -> TensorR(tensor))}

    // read the nodes and build the function for inference
    def inference_func(initializer_map_tensor: Map[String, Tensor], allNodesPassed: Seq[Node] = allNodes): (Tensor => Tensor) = { x: Tensor =>
    // lazy val inference_func: (Tensor => Tensor) = { x: Tensor =>
      Tensor.assertShapeEqual(x.shape, Dimensions(x_dims))

      // generate Tensors (or TensorRs) of intermediate steps and inputs
      val intermediate_map_tensor: MMap[String, Tensor] = MMap()
      intermediate_map_tensor += (x_name -> x)

      // TODO (Fei Wang): ask Greg, is there a better way to do this?
      def twoMaps = initializer_map_tensor orElse intermediate_map_tensor
      def inTwoMaps = (k: String) => initializer_map_tensor.contains(k) || intermediate_map_tensor.contains(k)

      allNodesPassed.foreach { node =>

        node match {

          case convNode(inputs, output, atts) => {
            // TODO: not yet handling dilations and groups

            val input1 = twoMaps(inputs.head)
            val input2 = twoMaps(inputs.tail.head)
            // bias tensor may not exist
            val input3 = if (inputs.size == 2) None else Some(twoMaps(inputs.last))
            val strides = atts("strides")
            val pads = atts("pads")                  // pads may be zero
            val kernel_shape = atts("kernel_shape")  // this attribute is actually not used

            val (out, finputOption, _) = input1.conv2D_batch(input2, input3, strides, pads)
            intermediate_map_tensor += (output -> out)
          }

          case reluNode(input, output) => {

            val in = twoMaps(input)
            val out = in.relu()
            intermediate_map_tensor += (output -> out)
          }

          case maxpoolNode(input, output, atts) => {

            val in = twoMaps(input)

            val strides = atts("strides")
            val pads = atts("pads")                     // pads may be zero
            val kernel_shape = atts("kernel_shape")

            val (out, _) = in.maxPool2D_batch(kernel_shape, strides, Some(pads))
            intermediate_map_tensor += (output -> out)
          }

          case averagePoolNode(input, output, atts) => {

            val input1 = twoMaps(input)

            val strides = atts("strides")
            val kernel = atts("kernel_shape")
            val pads = atts("pads")           // pads may be zero

            val out = input1.averagePool2D_batch(kernel, strides, Some(pads))
            intermediate_map_tensor += (output -> out)
          }

          case concatNode(inputs, output, axis) => {
            if (inTwoMaps(inputs.head)) {
              val input_s = inputs.map(x => twoMaps(x))
              val out = input_s.head.concat(axis, input_s.tail: _*)
              intermediate_map_tensor += (output -> out)
            } else {
              // For now this means that the inputs are in intMaps
              val input_s = inputs.map(x => intMap.get(x).get)
              val dims: Seq[Seq[Int]] = input_s map {case (d, a) => d}
              val datas: Seq[Seq[Rep[Int]]] = input_s map {case (d, a) => a}
              dims foreach { d => assert(d.size == 1, s"shape tensors should have rank 1, got ${d}")}
              assert(axis == 0, s"concatenating shape tensors should only be on dim 0, got ${axis}")
              val conDims = dims.foldLeft(Seq(0))((a, b) => Seq(a(0) + b(0)))
              val conDatas = datas.foldLeft(scala.Seq[Rep[Int]]())((a, b) => a ++ b)
              intMap += (output -> (conDims, conDatas))
            }
          }

          case dropoutNode(input, outputs, ratio) => {

            // dropoutNode in inference function should act as identity function
            val in = twoMaps(input)
            intermediate_map_tensor += (outputs.head -> in)
          }

          case globalAveragePoolNode(input, output) => {

            val in = twoMaps(input)
            val out = in.global_ave_batch()
            intermediate_map_tensor += (output -> out)
          }

          case softmaxNode(input, output) => {

            val in = twoMaps(input)
            val out = in.softmax_batch()
            intermediate_map_tensor += (output -> out)
          }

          case bnNode(inputs, output, attMap) => {

            assert(inputs.size == 5, "For inference mode, BatchNormalization should have 5 inputs")
            val (in::scale::bias::runningMean::runningVariance::Nil) = inputs.toList.map(twoMaps(_))
            val out = in.batchNormInference(scale, bias, runningMean, runningVariance)
            intermediate_map_tensor += (output -> out)
          }

          case sumNode(inputs, output) => {

            val input1 = twoMaps(inputs.head)
            val input2 = twoMaps(inputs.last)
            val out = input1 + input2
            intermediate_map_tensor += (output -> out)
          }

          case reshapeNode(inputs, output) => {

            val input1 = twoMaps(inputs.head)
            // input2 should be Int64 tensor (we can find it in intMap)
            val (dim2, input2: Seq[Rep[Int]]) = intMap(inputs.last)
            assert (dim2.size == 1, s"reshape parameter (if presented as a tensor) should be dim 1, got ${dim2.size}")
            val out = input1.resize(input2: _*)

            intermediate_map_tensor += (output -> out)
          }

          case gemmNode(inputs, output, attInts, attFloats) => {

            val input1 = twoMaps(inputs.head)
            val input2 = twoMaps(inputs.tail.head)
            val input3 = twoMaps(inputs.last)

            val alpha = attFloats.getOrElse("alpha", 1.0f)
            val beta  = attFloats.getOrElse("beta", 1.0f)
            val transA = attInts.getOrElse("transA", 0)
            val transB = attInts.getOrElse("transB", 0)

            val out = input1.gemm(input2, transA == 1, transB == 1, alpha) + (if (beta == 1.0f) input3 else input3 * beta)
            intermediate_map_tensor += (output -> out)
          }

          case flattenNode(input, output, axis) => {
            val input1 = twoMaps(input)
            val shape = (input1.shape.take(axis) :+ unit(-1))
            intermediate_map_tensor += (output -> input1.resize(shape: _*))
          }

          case addNode(inputs, output) => {
            assert(inputs.size == 2, s"inputs for addNode should be size 2, got ${inputs.size}")
            val (input1 :: input2 :: Nil)  = inputs.map(a => twoMaps(a)).take(2).toList
            intermediate_map_tensor += (output -> (input1 + input2))
          }

          case shapeNode(input, output) => {
            val input1 = twoMaps(input)
            val out: Seq[Rep[Int]] = input1.shape
            // For now, saving Int typed Tensor to intMap (mutable map of type String -> (Seq[Int], Array[Int]))
            intMap += (output -> ( Seq(out.size), out) )
          }

          case sliceNode(input, output, attMap: Map[String, List[Int]]) => {
            if (inTwoMaps(input)) {
              ??? // TODO (Fei Wang): implemented general slice node
            } else {
              val (dim1, input1: Seq[Rep[Int]]) = intMap.get(input).get
              val axes: List[Int] = attMap.get("axes").get
              val starts: List[Int] = attMap.get("starts").get
              val ends: List[Int] = attMap.get("ends").get
              assert(axes == List(0), s"slice on shape tensor must be on dim 0, got ${axes}")
              assert(starts.size == 1, s"index on shape tensor must be of size 1, got ${starts}")
              assert(ends.size == 1, s"index on shape tensor must be of size 1, got ${ends}")
              val start = starts(0); val end = ends(0)
              intMap += (output -> (Seq(end - start), input1.take(end).drop(start)))
            }
          }

          case squeezeNode(input, output, axes) => {
            if (inTwoMaps(input)) {
              ??? // TODO (Fei Wang): implement general squeeze node function
            } else {
              val (dim1: Seq[Int], input1) = intMap.get(input).get
              assert(axes == List(0),  s"squeeze on shape tensor must be on dim 0, axes got ${axes}")
              assert(dim1 == Seq(1), s"the dimension to squeeze must be dim 1, got ${dim1}")
              assert(input1.size == 1, s"to squeeze on the shape tensor, it must have 1 element, got ${input1}")
              intMap += (output -> (Seq[Int](), input1))
            }
          }

          case constantNode(output, data) => {
            // For now we assume that constantNode are Int (for tensor shape)
            intMap += (output -> (Seq[Int](), Seq(unit(data.toInt))))
          }

          case unsqueezeNode(input, output, axes) => {
            if (inTwoMaps(input)) {
              ??? // TODO (Fei Wang): implement general unsqueeze function
            } else {
              val (dim1: Seq[Int], input1) = intMap.get(input).get
              assert(axes == List(0),  s"unsqueeze on shape tensor must be on dim 0, axes got ${axes}")
              assert(dim1 == Seq[Int](),  s"unsqueeze on shape tensor must be empty seq, dims got ${dim1}")
              assert(input1.size == 1, s"unsqueeze on shape tensor must have 1 element, got ${input1}")
              intMap += (output -> (Seq(1), input1))
            }
          }

          case padNode(input, output, mode, pads, value) => {
            if (inTwoMaps(input)) {
              val input1 = twoMaps(input)
              assert(pads.sum == 0 , s"TODO: only supporting no padding so far, got ${pads}")
              intermediate_map_tensor += (output -> input1)
            } else {
              ???
            }
          }

          case gatherNode(input, output, axis) => {
            if (inTwoMaps(input)) {
              ???
            } else {
              val (dim1: Seq[Int], input1) = intMap.get(input).get
              assert(axis < input1.size, s"axis out of bound, got ${axis}, but array size is ${input1.size}")
              intMap += (output -> (Seq(), Seq(input1(axis))))
            }
          }

          case x =>
            throw new RuntimeException(s"not yet implemented, $x")
        }
      }

      intermediate_map_tensor(y_name)
    }

    // read the nodes and build the function for training
    def training_func(initializer_map_tensor: Map[String, Tensor]): ((TensorR => TensorR @diff), Map[String, TensorR]) = {
      val initializer_map_tensorR: Map[String, TensorR] = initializer_map_tensor.map { case(name, tensor) => (name -> TensorR(tensor))}
      val func = { x: TensorR =>
        // lazy val training_func: (TensorR => TensorR @diff) = { x: TensorR =>
        Tensor.assertShapeEqual(x.x.shape, Dimensions(x_dims), "input tensor is not of the correct dimensions")

        // generate Tensors (or TensorRs) of intermediate steps and inputs
        val intermediate_map_tensorR: MMap[String, TensorR] = MMap()
        intermediate_map_tensorR.update(x_name, x)

        // TODO (Fei Wang): ask Greg, is there a better way to do this?
        def twoMaps = initializer_map_tensorR orElse intermediate_map_tensorR
        def inTwoMaps(key: String) = initializer_map_tensorR.contains(key) || intermediate_map_tensorR.contains(key)

        val iter = allNodes.iterator

        while (iter.hasNext) {

          val node = iter.next

          if (node.isInstanceOf[convNode]) {
            val convNode(inputs, output, atts) = node
            val input1 = twoMaps(inputs.head)
            val input2 = twoMaps(inputs.tail.head)
            val input3 = if (inputs.size == 2) None else Some(twoMaps(inputs.last))

            val strides = atts("strides")
            val pads = atts("pads")
            val kernel_shape = atts("kernel_shape")  // this attribute is actually not used

            val out = input1.convBBP(input2, input3, strides, pads)
            intermediate_map_tensorR.update(output, out)
          } else if (node.isInstanceOf[reluNode]) {
            val reluNode(input, output) = node
            val in = twoMaps(input)
            val out = in.relu(true)
            intermediate_map_tensorR.update(output, out)
          } else if (node.isInstanceOf[maxpoolNode]) {
            val maxpoolNode(input, output, atts) = node
            val in = twoMaps(input)
            val strides = atts("strides")
            val pads = atts("pads")
            val kernel_shape = atts("kernel_shape")
            // TODO: (Fei Wang) erroneous code, the implementation assumes that pads are all 0
            val out = in.maxPoolBK(kernel_shape, strides, None)
            intermediate_map_tensorR.update(output, out)
          } else if (node.isInstanceOf[concatNode]) {
            val concatNode(inputs, output, axis) = node
            if (inTwoMaps(inputs.head)) {
              val input_s = inputs.map(x => twoMaps(x))
              val out = input_s.head.concat(axis, input_s.tail: _*)
              intermediate_map_tensorR.update(output, out)
            } else {
              val input_s = inputs.map(x => intMap.get(x).get)
              val dims = input_s.map{case (d, a) => d}
              val datas = input_s.map{case (d, a) => a}
              dims foreach (d => assert(d.size == 1, s"intMaps (for shapeTensors) should all be rank 1, got ${d}"))
              val con_dims = dims.foldLeft(Seq(0))((a, b) => Seq(a(0) + b(0)))
              val con_datas = datas.foldLeft(scala.Seq[Rep[Int]]())((a, b) => a ++ b)
              intMap += (output -> (con_dims, con_datas))
            }

          } else if (node.isInstanceOf[dropoutNode]) {

            val dropoutNode(input, outputs, ratio) = node
            val in = twoMaps(input)
            val out = in.dropout(ratio)
            intermediate_map_tensorR.update(outputs.head, out)
            // intermediate_map_tensor += (outputs.last -> out2)

          } else if (node.isInstanceOf[globalAveragePoolNode]) {

            val globalAveragePoolNode(input, output) = node
            val in = twoMaps(input)
            val out = in.global_ave_batch()
            intermediate_map_tensorR.update(output, out)

          } else if (node.isInstanceOf[softmaxNode]) {

            val softmaxNode(input, output) = node
            val in = twoMaps(input)
            val out = in.logSoftmaxB()
            intermediate_map_tensorR.update(output, out)

          } else if (node.isInstanceOf[averagePoolNode]) {

            val averagePoolNode(input, output, atts) = node
            val input1 = twoMaps(input)
            val strides = atts("strides")
            val kernel = atts("kernel_shape")
            val pads = atts("pads")           // pads may be zero
            val out = input1.averagePoolBK(kernel, strides, Some(pads))
            intermediate_map_tensorR.update(output, out)

          } else if (node.isInstanceOf[bnNode]) {

            val bnNode(inputs, output, attMap) = node
            val (in::scale::bias::runningMean::runningVariance::Nil) = inputs.toList.map(twoMaps(_))
            val out = in.batchNorm(scale, bias, runningMean.x, runningVariance.x)

            intermediate_map_tensorR.update(output, out)

          } else if (node.isInstanceOf[sumNode]) {

            val sumNode(inputs, output) = node

            val input1 = twoMaps(inputs.head)
            val input2 = twoMaps(inputs.last)
            val out = input1 + input2
            intermediate_map_tensorR.update(output, out)

          } else if (node.isInstanceOf[reshapeNode]) {

            val reshapeNode(inputs, output) = node

            val input1 = twoMaps(inputs.head)
            // input2 should be Int64 tensor (we can find it in intMap)
            val (dim2, input2) = intMap(inputs.last)
            assert (dim2.size == 1, s"reshape parameter (if presented as a tensor) should be dim 1, got ${dim2.size}")
            val out = input1.resize(input2.toSeq: _*)

            intermediate_map_tensorR.update(output, out)

          } else if (node.isInstanceOf[gemmNode]) {

            val gemmNode(inputs, output, attInts, attFloats) = node

            val input1 = twoMaps(inputs.head)
            val input2 = twoMaps(inputs.tail.head)
            val input3 = twoMaps(inputs.last)

            val alpha = attFloats.getOrElse("alpha", 1.0f)
            val beta  = attFloats.getOrElse("beta", 1.0f)
            val transA = attInts.getOrElse("transA", 0)
            val transB = attInts.getOrElse("transB", 0)
            if (beta == 1.0f) {
              val out = input1.gemm(input2, transA == 1, transB == 1, alpha) + input3
              intermediate_map_tensorR.update(output, out)
            } else {
              val out = input1.gemm(input2, transA == 1, transB == 1, alpha) + (input3 * beta)
              intermediate_map_tensorR.update(output, out)
            }

          } else if (node.isInstanceOf[flattenNode]) {
            val flattenNode(input, output, axis) = node
            val input1 = twoMaps(input)
            val shape = input1.x.shape.take(axis) :+ unit(-1)
            val out = input1.resize(shape: _*)
            intermediate_map_tensorR.update(output, out)

          } else if (node.isInstanceOf[addNode]) {
            val addNode(inputs, output) = node
            assert(inputs.size == 2)
            val (input1 :: input2::Nil) = inputs.map(twoMaps).take(2).toList
            intermediate_map_tensorR.update(output, input1 + input2)

          } else if (node.isInstanceOf[shapeNode]) {
            val shapeNode(input, output) = node
            val input1 = twoMaps(input)
            val shape = input1.x.shape
            // Note: shape information, which is non Rep type, should be saved in intMap
            intMap += (output -> (Seq(shape.size), shape))

          } else if (node.isInstanceOf[constantNode]) {
            val constantNode(output, data) = node
            // For now we assume constantNode are Int (for shape)
            intMap += (output -> (Seq[Int](), Seq(unit(data.toInt))))

          } else if (node.isInstanceOf[sliceNode]) {
            val sliceNode(input, output, attMap) = node
            if (inTwoMaps(input)) {
              ???  // TODO (Fei Wang) handle general slice case
            } else {
              val (dim1: Seq[Int], input1) = intMap.get(input).get
              val axes: List[Int] = attMap.getOrElse("axes", List[Int]())
              val starts: List[Int] = attMap.getOrElse("starts", List[Int]())
              val ends: List[Int] = attMap.getOrElse("ends", List[Int]())
              assert(axes == List(0), s"for slice shape tensor, axes must be 0, axes got ${axes}")
              assert(starts.size == 1, s"for slice shape tensor, starts must be size 1, starts got ${starts}")
              assert(ends.size == 1, s"for slice shape tensor, ends must be size 1, ends got ${ends}")
              val start = starts(0); val end = ends(0)
              intMap += (output -> (Seq(end - start), input1.take(end).drop(start)))
            }

          } else if (node.isInstanceOf[squeezeNode]) {
            val squeezeNode(input, output, axes) = node
            if (inTwoMaps(input)) {
              ???  // TODO (Fei Wang) handle general slice case
            } else {
              val (dim1: Seq[Int], input1) = intMap.get(input).get
              assert(axes == List(0),  s"squeeze on shape tensor must be on dim 0, axes got ${axes}")
              assert(dim1 == Seq(1), s"the dimension to squeeze must be dim 1, got ${dim1}")
              assert(input1.size == 1, s"to squeeze on the shape tensor, it must have 1 element, got ${input1}")
              intMap += (output -> (Seq[Int](), input1))
            }

          } else if (node.isInstanceOf[unsqueezeNode]) {
            val unsqueezeNode(input, output, axes) = node
            if (inTwoMaps(input)) {
              ??? // TODO: (Fei Wang) handle the general unsqueeze case
            } else {
              val (dim1: Seq[Int], input1) = intMap.get(input).get
              assert(axes == List(0),  s"unsqueeze on shape tensor must be on dim 0, axes got ${axes}")
              assert(dim1 == Seq[Int](),  s"unsqueeze on shape tensor must be empty seq, dims got ${dim1}")
              assert(input1.size == 1, s"unsqueeze on shape tensor must have 1 element, got ${input1}")
              intMap += (output -> (Seq(1), input1))
            }

          } else if (node.isInstanceOf[padNode]) {
            val padNode(input, output, mode, pads, value) = node
            if (inTwoMaps(input)) {
              val input1 = twoMaps(input)
              assert(pads.sum == 0 , s"TODO: only supporting no padding so far, got ${pads}")
              intermediate_map_tensorR += (output -> input1)
            } else {
              ???
            }

          } else if (node.isInstanceOf[gatherNode]) {

            val gatherNode(input, output, axis) = node
            if (inTwoMaps(input)) {
              ???
            } else {
              val (dim1: Seq[Int], input1) = intMap.get(input).get
              assert(axis < input1.size, s"axis out of bound, got ${axis}, but array size is ${input1.size}")
              intMap += (output -> (Seq(), Seq(input1(axis))))
            }

          } else {
            System.out.println(s"node $node is not implemented")
            shift{ (k: Tensor => Unit) => ???}
          }
        }

        intermediate_map_tensorR(y_name)
      }
      (func, initializer_map_tensorR)
    }

  }

  def readOnnxData(filename: String): Rep[Array[Float]] =
    if (filename.endsWith(".csv"))
      readCsv(filename)
    else if (filename.endsWith(".pb"))
      ???
    else
      ???

  def readCsv(filename: String) = {
    Array(
      Source.fromFile(filename).getLines.flatMap { (line: String) =>
        if (line != "" && line != "\n")
          line.split(",") map { x => System.out.println(x); unit(x.toFloat) }
        else
          Nil
      }.toSeq : _*
    )
  }

}

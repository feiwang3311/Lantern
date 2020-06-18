package lantern

import scala.util.continuations._

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.{Map => MutableMap}
import scala.math._

import lms.core.stub._
import lms.macros.SourceContext
import lms.core.virtualize

trait TensorDslCudnn extends TensorDslCublas with GPUOps {

  val elementWiseWithBroadCastKernelMap = new scala.collection.mutable.HashMap[(Int, String), (String, String)]()
  val elementWiseUpdateWithBroadCastKernelMap = new scala.collection.mutable.HashMap[(Int, String), (String, String)]()
  val convOpIndexSet = scala.collection.mutable.ListBuffer.empty[Int]
  val attributesMap = new scala.collection.mutable.HashMap[Int, String]()
  def printAttributes() = unchecked[Unit](attributesMap.values.mkString("\n"))

  // A map from tensor shapes to cuDNN tensor descriptors.
  private var tensorDescriptorCache = MutableMap[Dimensions, String]()
  private var tensorDescriptorCount = 0
  def freshDescriptorId: Int = { val tmp = tensorDescriptorCount; tensorDescriptorCount += 1; tmp }

  class TensorDescriptorOps(x: Tensor) {
    def descriptor: Rep[String] = {
      if (tensorDescriptorCache.contains(x.shape)) {
        tensorDescriptorCache(x.shape)
      } else {
        val id = freshDescriptorId
        val descName = s"desc$id"
        if (x.rank == 4) {
          unchecked[Unit](
            Seq(s"""
               |cudnnTensorDescriptor_t $descName;
               |CUDNN_CALL(cudnnCreateTensorDescriptor(&$descName));
               |CUDNN_CALL(cudnnSetTensor4dDescriptor(
               |    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
               |    """.stripMargin, x.shape(0), ", ", x.shape(1), ", ", x.shape(2), ", ", x.shape(3), "))"): _*)
        } else {
          assert(x.rank >= 3, "'cudnnCreateTensorDescriptor' only supports descriptors for tensors with rank at least 3")
          val dims: Seq[Any] = x.shape.flatMap(dim => Seq[Any](dim, ", "))
          val strides: Seq[Any] = x.shape.strides.flatMap(stride => Seq[Any](stride, ", "))
          val dimsName = s"dims$id"
          val stridesName = s"strides$id"
          unchecked[Unit](
            Seq(
               s"cudnnTensorDescriptor_t $descName;\n" +
               s"CUDNN_CALL(cudnnCreateTensorDescriptor(&$descName));\n" +
               s"int $dimsName[] = {") ++ dims ++ Seq("};\n" +
               s"int $stridesName[] = {") ++ strides ++ Seq("};\n" +
               "CUDNN_CALL(cudnnSetTensorNdDescriptor(\n" +
               s"    $descName, CUDNN_DATA_FLOAT, /*nbDims*/ ${x.rank}, $dimsName, $stridesName))"): _*)
        }
        // Update descriptor cache.
        tensorDescriptorCache(x.shape) = descName
        // Return descriptor name.
        descName
      }
    }
  }
  implicit def tensorToDescriptorOps(x: Tensor) = new TensorDescriptorOps(x)

  // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnActivationMode_t
  object Activation extends Enumeration {
    val Sigmoid = Value("CUDNN_ACTIVATION_SIGMOID")
    val Relu = Value("CUDNN_ACTIVATION_RELU")
    val Tanh = Value("CUDNN_ACTIVATION_TANH")
    val ClippedRelu = Value("CUDNN_ACTIVATION_CLIPPED_RELU")
    val Elu = Value("CUDNN_ACTIVATION_ELU")
  }

  // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnPoolingMode_t
  object PoolModes extends Enumeration {
    val Max = Value("CUDNN_POOLING_MAX")
    val AverageIP = Value("CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING")
    val AverageEP = Value("CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING")
    val MaxD = Value("CUDNN_POOLING_MAX_DETERMINISTIC")
  }

  // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnNanPropagation_t
  object NanOpt extends Enumeration {
    val NotProp = Value("CUDNN_NOT_PROPAGATE_NAN")
    val Prop = Value("CUDNN_PROPAGATE_NAN")
  }

  // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnSoftmaxMode_t
  object SoftmaxMode extends Enumeration {
    val Fast = Value("CUDNN_SOFTMAX_FAST")
    val Accurate = Value("CUDNN_SOFTMAX_ACCURATE")
    val Log = Value("CUDNN_SOFTMAX_LOG")
  }

  // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnReduceTensorOp_t
  object ReductionOp extends Enumeration {
    val Add = Value("CUDNN_REDUCE_TENSOR_ADD")
    val Mul = Value("CUDNN_REDUCE_TENSOR_MUL")
    val Min = Value("CUDNN_REDUCE_TENSOR_MIN")
    val Max = Value("CUDNN_REDUCE_TENSOR_MAX")
    val Avg = Value("CUDNN_REDUCE_TENSOR_AVG")
    // Maximum of absolute values.
    val Amax = Value("CUDNN_REDUCE_TENSOR_AMAX")
    // Addition of absolute values.
    val Norm1 = Value("CUDNN_REDUCE_TENSOR_NORM1")
    // Square root of sum of squares.
    val Norm2 = Value("CUDNN_REDUCE_TENSOR_NORM2")
    // Multiplication, ignoring zero elements.
    val MulNoZeros = Value("CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS")
  }

  // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
  sealed trait RnnMode {
    val numGates: Int
  }
  case object RnnReluMode extends RnnMode {
    override def toString: String = "CUDNN_RNN_RELU"
    override val numGates: Int = 1
  }
  case object RnnTanhMode extends RnnMode {
    override def toString: String = "CUDNN_RNN_TANH"
    override val numGates: Int = 1
  }
  case object LstmMode extends RnnMode {
    override def toString: String = "CUDNN_LSTM"
    override val numGates: Int = 4
  }
  case object GruMode extends RnnMode {
    override def toString: String = "CUDNN_GRU"
    override val numGates: Int = 3
  }

  // val cudnnMathType = None
  // val cudnnMathType = Some("CUDNN_DEFAULT_MATH")
  val cudnnMathType = Some("CUDNN_TENSOR_OP_MATH")
  // val cudnnMathType = Some("CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION")

  /**
    * cuDNN tensor operation backend. WIP.
    * Extends `BackendCublas` to leverage cuBLAS primitives.
    */
  class BackendCudnn protected() extends BackendCublas {
    override def setup(): Unit = {
      super.setup()
      unchecked("cudnnHandle_t cudnnHandle;\nCUDNN_CALL(cudnnCreate(&cudnnHandle));")
    }

    override def cleanup(): Unit = {
      super.cleanup()
      unchecked("CUDNN_CALL(cudnnDestroy(cudnnHandle));")
    }

    def elementWiseWithBroadCastKernel(rank: Int, op: String): String = {
      if (!elementWiseWithBroadCastKernelMap.contains((rank, op))) {
        val in1Stride = ((0 until rank): Range).map(x => s"int in1Stride$x").mkString(", ")
        val in2Stride = ((0 until rank): Range).map(x => s"int in2Stride$x").mkString(", ")
        val outStride = ((0 until rank): Range).map(x => s"int outStride$x").mkString(", ")
        val linearToStep = ((0 until rank): Range).map(x => s"int outIndex$x = linearIdx / outStride$x; linearIdx = linearIdx - outIndex$x * outStride$x;").mkString("\n    ")
        val in1Index = ((0 until rank): Range).map(x => s"in1Stride$x * outIndex$x").mkString(" + ")
        val in2Index = ((0 until rank): Range).map(x => s"in2Stride$x * outIndex$x").mkString(" + ")
        val kernel = s"""
        |__global__ void elementWiseWithBroadCast${nextKernel}(float* in1, float* in2, float* out, int size,
        |                ${in1Stride}, ${in2Stride}, ${outStride}) {
        |  int tid = threadIdx.x + blockIdx.x * blockDim.x;
        |  int stride = gridDim.x * blockDim.x;
        |  for (; tid < size; tid += stride) {
        |    int linearIdx = tid;
        |    ${linearToStep}
        |    int in1Index = ${in1Index};
        |    int in2Index = ${in2Index};
        |    out[tid] = in1[in1Index] ${op} in2[in2Index];
        |  }
        |}
        """.stripMargin
        val kernelName = s"elementWiseWithBroadCast${nextKernel}"
        elementWiseWithBroadCastKernelMap((rank, op)) = (kernel, kernelName)
        // don't forget to increment counter!!
        nextKernel += 1
      }
      val (kernel, kernelName) = elementWiseWithBroadCastKernelMap((rank, op))
      kernelName
    }

    def elementWiseWithBroadCast(in1: Tensor, in2: Tensor, op: String): (Tensor, Dimensions, Dimensions) = {
      Tensor.dimBroadcast(in1.shape, in2.shape) match {
        case Some((xShape, yShape, resShape)) => {
          val resData = mallocArray[Float](resShape.scalarCount)
          val res = new Tensor(resData, resShape)
          val xStridesShadow = (xShape.strides zip xShape.dims) map {case (a, b) => if (b == unit(1)) 0 else a}
          val yStridesShadow = (yShape.strides zip yShape.dims) map {case (a, b) => if (b == unit(1)) 0 else a}
          val kernelName = elementWiseWithBroadCastKernel(resShape.dims.size, op)
          val nGrid = 28
          if (resShape.dims.size == 1) {
            unchecked[Unit](s"${kernelName}<<<${nGrid}, 512>>>(", in1.data, ", ", in2.data, ", ", resData, ", ", res.scalarCount, ", ",
              xStridesShadow(0), ", ", yStridesShadow(0), ", ", resShape.strides(0), ")")
          } else if (resShape.dims.size == 2) {
            unchecked[Unit](s"${kernelName}<<<${nGrid}, 512>>>(", in1.data, ", ", in2.data, ", ", resData, ", ", res.scalarCount, ", ",
              xStridesShadow(0), ", ", xStridesShadow(1), ", ", yStridesShadow(0), ", ", yStridesShadow(1), ", ", resShape.strides(0), ", ", resShape.strides(1), ")")
          } else if (resShape.dims.size == 3) {
            unchecked[Unit](s"${kernelName}<<<${nGrid}, 512>>>(", in1.data, ", ", in2.data, ", ", resData, ", ", res.scalarCount, ", ",
              xStridesShadow(0), ", ", xStridesShadow(1), ", ", xStridesShadow(2), ", ",
              yStridesShadow(0), ", ", yStridesShadow(1), ", ", yStridesShadow(2), ", ",
              resShape.strides(0), ", ", resShape.strides(1), ", ", resShape.strides(2), ")")
          } else if (resShape.dims.size == 4) {
            unchecked[Unit](s"${kernelName}<<<${nGrid}, 512>>>(", in1.data, ", ", in2.data, ", ", resData, ", ", res.scalarCount, ", ",
              xStridesShadow(0), ", ", xStridesShadow(1), ", ", xStridesShadow(2), ", ", xStridesShadow(3), ", ",
              yStridesShadow(0), ", ", yStridesShadow(1), ", ", yStridesShadow(2), ", ", yStridesShadow(3), ", ",
              resShape.strides(0), ", ", resShape.strides(1), ", ", resShape.strides(2), ", ", resShape.strides(3), ")")
          } else {
            assert(false, s"elementWiseWithBroadCast only handle tensors with rank no larger than 4, got ${resShape.dims.size}")
          }
          (res, xShape, yShape)
        }
        case _ => ???
      }
    }

    def elementWiseUpdateWithBroadCastKernel(rank: Int, op: String): String = {
      if (!elementWiseUpdateWithBroadCastKernelMap.contains((rank, op))) {
        val in1Stride = ((0 until rank): Range).map(x => s"int in1Stride$x").mkString(", ")
        val in2Stride = ((0 until rank): Range).map(x => s"int in2Stride$x").mkString(", ")
        val linearToStep = ((0 until rank): Range).map(x => s"    int in1Index$x = linearIdx / in1Stride$x; linearIdx = linearIdx - in1Index$x * in1Stride$x;").mkString("\n")
        val in1Index = ((0 until rank): Range).map(x => s"in1Stride$x * in1Index$x").mkString(" + ")
        val in2Index = ((0 until rank): Range).map(x => s"in2Stride$x * in1Index$x").mkString(" + ")
        val kernel = s"""
        |__global__ void elementWiseUpdateWithBroadCast${nextKernel}(float* in1, float* in2, int size,
        |                ${in1Stride}, ${in2Stride}) {
        |  int tid = threadIdx.x + blockIdx.x * blockDim.x;
        |  int stride = gridDim.x * blockDim.x;
        |  for (int i = tid; i < size; i += stride) {
        |    int linearIdx = tid;
        |    ${linearToStep}
        |    int in1Index = ${in1Index};
        |    int in2Index = ${in2Index};
        |    in1[in1Index] ${op} in2[in2Index];
        |  }
        |}
        """
        val kernelName = s"elementWiseUpdateWithBroadCast${nextKernel}"
        elementWiseUpdateWithBroadCastKernelMap((rank, op)) = (kernel, kernelName)
        // don't forget to increment counter!!
        nextKernel += 1
      }
      val (kernel, kernelName) = elementWiseUpdateWithBroadCastKernelMap((rank, op))
      kernelName
    }

    @virtualize
    def elementWiseUpdateWithBroadCast(in1: Tensor, in2: Tensor, op: String) = {
        Tensor.dimBroadcast(in1.shape, in2.shape) match {
          case Some((xShape, yShape, resShape)) => {
            if (xShape.broadcasted) {
              assertC(false, "TODO: so far only handles in1 += in2 where in1 is large!!")
            } // TODO: so far only handles in1 += in2 where in1 is larger
            else {
              val xStridesShadow = (xShape.strides zip xShape.dims) map {case (a, b) => if (b == unit(1)) 0 else a}
              val yStridesShadow = (yShape.strides zip yShape.dims) map {case (a, b) => if (b == unit(1)) 0 else a}
              val kernelName = elementWiseUpdateWithBroadCastKernel(resShape.dims.size, op)
              val nGrid = 28
              if (resShape.dims.size == 1) {
                unchecked[Unit](s"${kernelName}<<<${nGrid}, 512>>>(", in1.data, ", ", in2.data, ", ", in1.scalarCount, ", ",
                  xStridesShadow(0), ", ", yStridesShadow(0), ")")
              } else if (resShape.dims.size == 2) {
                unchecked[Unit](s"${kernelName}<<<${nGrid}, 512>>>(", in1.data, ", ", in2.data, ", ", in1.scalarCount, ", ",
                  xStridesShadow(0), ", ", xStridesShadow(1), ", ", yStridesShadow(0), ", ", yStridesShadow(1), ")")
              } else if (resShape.dims.size == 3) {
                unchecked[Unit](s"${kernelName}<<<${nGrid}, 512>>>(", in1.data, ", ", in2.data, ", ", in1.scalarCount, ", ",
                  xStridesShadow(0), ", ", xStridesShadow(1), ", ", xStridesShadow(2), ", ",
                  yStridesShadow(0), ", ", yStridesShadow(1), ", ", yStridesShadow(2), ")")
              } else if (resShape.dims.size == 4) {
                unchecked[Unit](s"${kernelName}<<<${nGrid}, 512>>>(", in1.data, ", ", in2.data, ", ", in1.scalarCount, ", ",
                  xStridesShadow(0), ", ", xStridesShadow(1), ", ", xStridesShadow(2), ", ", xStridesShadow(3), ", ",
                  yStridesShadow(0), ", ", yStridesShadow(1), ", ", yStridesShadow(2), ", ", yStridesShadow(3), ")")
              } else {
                assert(false, s"elementWiseUpdateWithBroadCast only handle tensors with rank no larger than 4, got ${resShape.dims.size}")
              }
              ()
            }
          }
          case _ => ???
        }
    }

    override def +(x: Tensor, y: Rep[Float]): Tensor = {
      val res = mallocArray[Float](x.scalarCount)
      unchecked[Unit](s"addScalar<<<28, 512>>>(", x.data, ", ", res, ", ", y, ", ", x.scalarCount, ")")
      Tensor(res, x.shape: _*)
    }
    override def +(x: Tensor, y: Tensor): (Tensor, Dimensions, Dimensions) = elementWiseWithBroadCast(x, y, "+")
    @virtualize
    override def add_grad(x: TensorR, y: TensorR, output: TensorR, xShape: Dimensions, yShape: Dimensions): Unit = {
      val one = NewArray[Float](1); one(0) = 1
      if (!x.isInput) {
        if (xShape.broadcasted) cudnnReduceUpdateTensor(x.d, xShape, output.d, output.d.shape, one, one)
        else geam(x.d, false, 1.0f, output.d, false, 1.0f, x.d)
      }
      if (!y.isInput) {
        if (yShape.broadcasted) cudnnReduceUpdateTensor(y.d, yShape, output.d, output.d.shape, one, one)
        else geam(y.d, false, 1.0f, output.d, false, 1.0f, y.d)
      }
    }

    override def +=(x: Tensor, y: Rep[Float]): Unit = unchecked[Unit](s"addScalar<<<28, 512>>>(", x.data, ", ", x.data, ", ", y, ", ", x.scalarCount, ")")
    override def +=(x: Tensor, y: Tensor): Unit = elementWiseUpdateWithBroadCast(x, y, "+=")

    override def -(x: Tensor, y: Rep[Float]): Tensor = {
      val res = mallocArray[Float](x.scalarCount)
      unchecked[Unit](s"minusScalar<<<28, 512>>>(", x.data, ", ", res, ", ", y, ", ", x.scalarCount, ")")
      Tensor(res, x.shape: _*)
    }
    override def -(x: Tensor, y: Tensor): (Tensor, Dimensions, Dimensions) = elementWiseWithBroadCast(x, y, "-")
    @virtualize
    override def minus_grad(x: TensorR, y: TensorR, output: TensorR, xShape: Dimensions, yShape: Dimensions): Unit = {
      val one = NewArray[Float](1); one(0) = 1
      val minus_one = NewArray[Float](1); minus_one(0) = -1
      if (!x.isInput) {
        if (xShape.broadcasted) cudnnReduceUpdateTensor(x.d, xShape, output.d, output.d.shape, one, one)
        else geam(x.d, false, 1.0f, output.d, false, 1.0f, x.d)
      }
      if (!y.isInput) {
        if (yShape.broadcasted) cudnnReduceUpdateTensor(y.d, yShape, output.d, output.d.shape, minus_one, one)
        else geam(y.d, false, 1.0f, output.d, false, -1.0f, x.d)
      }
    }

    override def -=(x: Tensor, y: Rep[Float]): Unit = unchecked[Unit](s"minusScalar<<<28, 512>>>(", x.data, ", ", x.data, ", ", y, ", ", x.scalarCount, ")")
    override def -=(x: Tensor, y: Tensor): Unit = elementWiseUpdateWithBroadCast(x, y, "-=")

    override def *(x: Tensor, y: Rep[Float]): Tensor = {
      val res = mallocArray[Float](x.scalarCount)
      unchecked[Unit](s"multScalar<<<28, 512>>>(", x.data, ", ", res, ", ", y, ", ", x.scalarCount, ")")
      Tensor(res, x.shape: _*)
    }
    override def *(x: Tensor, y: Tensor): (Tensor, Dimensions, Dimensions) = elementWiseWithBroadCast(x, y, "*")
    @virtualize
    override def mul_grad(x: TensorR, y: TensorR, output: TensorR, xShape: Dimensions, yShape: Dimensions): Unit = {
      val one = NewArray[Float](1); one(0) = 1
      if (!x.isInput) {
        val scaledXD = y.x * output.d
        if (xShape.broadcasted) cudnnReduceUpdateTensor(x.d, xShape, scaledXD, scaledXD.shape, one, one)
        else geam(x.d, false, 1.0f, scaledXD, false, 1.0f, x.d)
      }
      if (!y.isInput) {
        val scaledYD = x.x * output.d
        if (yShape.broadcasted) cudnnReduceUpdateTensor(y.d, yShape, scaledYD, scaledYD.shape, one, one)
        else geam(y.d, false, 1.0f, scaledYD, false, 1.0f, y.d)
      }
    }

    override def *=(x: Tensor, y: Rep[Float]): Unit = unchecked[Unit](s"multScalar<<<28, 512>>>(", x.data, ", ", x.data, ", ", y, ", ", x.scalarCount, ")")
    override def *=(x: Tensor, y: Tensor): Unit = ???

    override def /(x: Tensor, y: Rep[Float]): Tensor = {
      val res = mallocArray[Float](x.scalarCount)
      unchecked[Unit](s"divScalar<<<28, 512>>>(", x.data, ", ", res, ", ", y, ", ", x.scalarCount, ")")
      Tensor(res, x.shape: _*)
    }
    override def /(x: Tensor, y: Tensor): (Tensor, Dimensions, Dimensions) = elementWiseWithBroadCast(x, y, "/")
    @virtualize
    override def div_grad(x: TensorR, y: TensorR, output: TensorR, xShape: Dimensions, yShape: Dimensions): Unit = {
      val one = NewArray[Float](1); one(0) = 1
      val minus_one = NewArray[Float](1); minus_one(0) = -1
      if (!x.isInput) {
        val scaledXD = output.d / y.x
        if (xShape.broadcasted) cudnnReduceUpdateTensor(x.d, xShape, scaledXD, scaledXD.shape, one, one)
        else geam(x.d, false, 1.0f, scaledXD, false, 1.0f, x.d)
      }
      if (!y.isInput) {
        val scaledYD = x.x * output.d / (y.x * y.x) // TODO (fuse kernel)
        if (yShape.broadcasted) cudnnReduceUpdateTensor(y.d, yShape, scaledYD, scaledYD.shape, minus_one, one)
        else geam(y.d, false, 1.0f, scaledYD, false, -1.0f, y.d)
      }
    }

    override def /=(x: Tensor, y: Rep[Float]): Unit = unchecked[Unit](s"divScalar<<<28, 512>>>(", x.data, ", ", x.data, ", ", y, ", ", x.scalarCount, ")")
    override def /=(x: Tensor, y: Tensor): Unit = ???

    override def repeat0(in: Tensor, context: Int): Tensor = {
      assert(in.rank <= 3, s"only support input with no more than 3D, got ${in.rank}")
      val resShape = Seq(in.shape(0) - context, unit(context+1)) ++ in.shape.drop(1)
      val resTensor = Tensor(mallocArray[Float](resShape.product1), resShape: _*)
      // call user-defined kernel (which is similar to concat)
      val nGrid = 28
      unchecked[Unit](s"repeat0<<<${nGrid}, 512>>>(", in.data, ", ", resTensor.data, ", ", resTensor.shape.strides(0), ", ", resTensor.shape.strides(1), ", ", resTensor.scalarCount, ")")
      resTensor
    }

    override def repeat0_grad(in: TensorR, out: TensorR, context: Int): Unit = {
      // use shift and reduce (TODO (Fei Wang) may need to improve with a user-kernel?)
      val temp = Tensor(mallocArray[Float](out.x.scalarCount), out.x.shape: _*)
      val nGrid = 28
      unchecked[Unit](s"shift0<<<${nGrid}, 512>>>(", out.d.data, ", ", temp.data, ", ", out.x.shape(0), ", ", out.x.shape.strides(0), ", ", out.x.shape.strides(1), ", ", out.x.scalarCount, ")")
      // then reduce temp and add into in.d
      // TODO (Fei Wang): should not use smallerInD
      val smallerInD: Tensor = in.d(0, in.x.shape(0) - context)
      cudnnReduceTensor(temp, ReductionOp.Add, Seq(1), true, Some(smallerInD.data), false)
      ()
    }


    // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnAddTensor
    // Note: this function performs in-place addition for `res`.
    @virtualize
    def cudnnAddBiasTensor(bias: Tensor, res: Tensor, scale: Rep[Float] = 1.0f): Unit = {
      val (biasShape, resShape): (Seq[Rep[Int]], Seq[Rep[Int]]) = if (bias.shape == res.shape) {
        (bias.shape.padTo(4, unit(1)), res.shape.padTo(4, unit(1)))
      } else {
        if (bias.rank == 4 && res.rank == 4) {
          assertC((bias.shape zip res.shape).forallR{case (a, b) => a == 1 || a == b}, s"bias shape should be equal to res or be 1, got bias: ${bias.shape}, res: ${res.shape}")
          (bias.shape, res.shape)
        } else {
          assert(bias.rank == 1 && res.rank >= 2, "if not equal shape, bias must be rank 1, and res must be rank 2 or more")
          // TODO (Fei Wang): Need more thinking. Is it safe to assume that bias is on dim 1??
          (Seq(1, bias.shape(0), 1, 1), res.shape.padTo(4, unit(1)))
        }
      }
      val scaled = NewArray[Float](1); scaled(0) = scale
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq("""
          |{
          |cudnnTensorDescriptor_t bias_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, biasShape(0), ", ", biasShape(1), ", ", biasShape(2), ", ", biasShape(3), """));
          |
          |cudnnTensorDescriptor_t out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, resShape(0), ", ", resShape(1), ", ", resShape(2), ", ", resShape(3), """));
          |
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnAddTensor(\n" +
          "    cudnnHandle, ", scaled, ", bias_desc, ", bias.data, ", ", one, ", out_desc, ", res.data, "));\n" +
          "}"): _*
      )
    }

    // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionForward
    def cudnnConvolutionForward(input: Tensor, filter: Tensor, res: Tensor, bias: Option[Tensor] = None,
                                padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int)): Int = {
      assert(input.rank == 4, s"Convolution input must have rank 4, but got ${input.rank}")
      assert(res.rank == 4, s"Convolution result must have rank 4, but got ${res.rank}")
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1

      val counter = nextKernel
      convOpIndexSet += counter

      nextKernel += 1

      attributesMap(counter) = "// Attributes;"

      unchecked[Unit](
        Seq(s"""
          |cudnnTensorDescriptor_t in_desc_$counter;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_$counter));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    in_desc_$counter, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, input.shape(0), ", ", input.shape(1), ", ", input.shape(2), ", ",  input.shape(3), s"""));
          |
          |cudnnFilterDescriptor_t filt_desc_$counter;
          |CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc_$counter));
          |CUDNN_CALL(cudnnSetFilter4dDescriptor(
          |    filt_desc_$counter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
          |    """.stripMargin, filter.shape(0), ", ", filter.shape(1), ", ", filter.shape(2), ", ", filter.shape(3), s"""));
          |
          |cudnnTensorDescriptor_t out_desc_$counter;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_$counter));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    out_desc_$counter, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, res.shape(0), ", ", res.shape(1), ", ", res.shape(2), ", ", res.shape(3), s"""));
          |
          |cudnnConvolutionDescriptor_t conv_desc_$counter;
          |CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_$counter));
          |CUDNN_CALL(cudnnSetConvolution2dDescriptor(
          |    conv_desc_$counter,
          |    ${padding._1}, ${padding._2}, ${strides._1}, ${strides._2}, ${dilations._1}, ${dilations._2},
          |    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
          |""".stripMargin) ++
        cudnnMathType.map(mathType => Seq(s"CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc_$counter, $mathType));\n")).getOrElse(Seq()):_*
      )

      if (false)
      unchecked[Unit](
        Seq(s"""
          |if (!init_algo_$counter) {
          |cudnnConvolutionFwdAlgo_t algos_$counter[] = {
          |      CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
          |      CUDNN_CONVOLUTION_FWD_ALGO_FFT,
          |      CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
          |      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
          |      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
          |      CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
          |      CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
          |      CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
          |};
          |size_t max_sz_$counter = 0;
          |for (int c = 0; c < 8; c++) {
          |     size_t sz = 0;
          |     if (CUDNN_STATUS_SUCCESS == cudnnGetConvolutionForwardWorkspaceSize(
          |         cudnnHandle, in_desc_$counter, filt_desc_$counter, conv_desc_$counter, out_desc_$counter, algos_$counter[c], &sz)
          |         && max_sz_$counter < sz)
          |         max_sz_$counter = sz;
          |}
          |max_sz_$counter = CAP_AVAIL(max_sz_$counter);
          |cudnnConvolutionFwdAlgoPerf_t perfResults_$counter[8];
          |int perf_count_$counter;
          |void* maxSpace_$counter = myGpuMalloc(max_sz_$counter);
         """.stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnFindConvolutionForwardAlgorithmEx(\n" +
         s"    cudnnHandle, in_desc_$counter, ", input.data, s", filt_desc_$counter, ", filter.data, ",\n" +
         s"    conv_desc_$counter, out_desc_$counter, ", res.data, s", CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &perf_count_$counter,\n" +
         s"    perfResults_$counter, maxSpace_$counter, max_sz_$counter));\n" +
         s"myGpuFree(max_sz_$counter);\n" +
         s"algo_$counter = perfResults_$counter[0].algo;\n" +
         s"init_algo_$counter = true;\n}\n"): _*)
      else
      unchecked[Unit](
        s"""
          |int returned_algo_count_$counter;
          |cudnnConvolutionFwdAlgoPerf_t perfResults_$counter[8];
          |CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(
          |    cudnnHandle,
          |    in_desc_$counter, filt_desc_$counter, conv_desc_$counter, out_desc_$counter, 8, &returned_algo_count_$counter,
          |    perfResults_$counter));
          |algo_$counter = perfResults_$counter[0].algo;
         """.stripMargin)

      unchecked[Unit](
        Seq(
        s"""
          |size_t ws_size_$counter;
          |CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
          |    cudnnHandle, in_desc_$counter, filt_desc_$counter, conv_desc_$counter, out_desc_$counter, algo_$counter, &ws_size_$counter));
          |void *ws_data_$counter = myGpuMalloc(ws_size_$counter);
          |""".stripMargin) ++
        Seq(
          "// Execute convolution.\n" +
          "CUDNN_CALL(cudnnConvolutionForward(\n" +
          "    cudnnHandle,\n" +
          "    ", one, s", in_desc_$counter, ", input.data, s", filt_desc_$counter, ", filter.data, ",\n" +
         s"    conv_desc_$counter, algo_$counter, ws_data_$counter, ws_size_$counter,\n" +
          "    ", zero, s", out_desc_$counter, ", res.data, "));\n") ++
        Seq(s"myGpuFree(ws_size_$counter);\n"): _*)

      counter
    }

    override def plusBias(main: Tensor, bias: Tensor): Tensor = {
      // use cudnnAddTensor (bias is the first parameter, main tensor is the second parameter, addition is in-place on main tensor)
      cudnnAddBiasTensor(bias, main)
      main
    }

    @virtualize
    override def plusBias_grad(main: TensorR, bias: TensorR): Unit = if (!bias.isInput) {
      val shapeBias = Seq.fill(main.x.rank - bias.x.rank)(unit(1)) ++ bias.x.shape.dims
      val sameBias = (shapeBias zip main.x.shape.dims).forallR{case (x, y) => x == y}
      if (sameBias) geam(bias.d, false, 1.0f, main.d, false, 1.0f, bias.d)
      else {
        val one = NewArray[Float](1); one(0) = 1
        cudnnReduceUpdateTensor(bias.d, shapeBias, main.d, main.d.shape, one, one)
      }
    }

    override def plusEqual(base: Tensor, adder: Tensor): Tensor = {
      cudnnAddBiasTensor(adder, base)
      base
    }
    override def plusEqual_grad(base: TensorR, adder: TensorR): Unit = {
      if (!adder.isInput) cudnnAddBiasTensor(base.d, adder.d)
    }

    // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardBias
    // This is effectively the gradient of `cudnnAddBiasTensor`.
    def cudnnConvolutionBackwardBias(biasGrad: Tensor, resGrad: Tensor): Unit = {
      val biasShape: Seq[Rep[Int]] =
        if (biasGrad.rank == 1) Seq(1, biasGrad.shape(0), 1, 1)
        else if (biasGrad.rank == 4) biasGrad.shape
        else { assert(false, s"Bias gradient is neither rank 1 or rank 4, got ${biasGrad.shape}"); ???}
      assert(resGrad.rank >= 2, "Convolution result gradient must have rank no less than 2")
      if (biasGrad.rank == 1) assert(resGrad.shape(1) == biasGrad.shape(0), "Convolution result gradient shape(1) must equal to Bias gradient shape(0)")
      val resGradShape = resGrad.shape.padTo(4, 1)
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq("""
          |{
          |cudnnTensorDescriptor_t grad_bias_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, biasShape(0), ", ", biasShape(1), ", ", biasShape(2), ", ", biasShape(3), """));
          |
          |cudnnTensorDescriptor_t grad_out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, resGradShape(0), ", ", resGradShape(1), ", ", resGradShape(2), ", ", resGradShape(3), """));
          |
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnConvolutionBackwardBias(\n" +
          "    cudnnHandle, ", one, ", grad_out_desc, ", resGrad.data, ",\n",
          "    ", one, ", grad_bias_desc, ", biasGrad.data, "));\n" +
          "}"): _*
      )
    }

    // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardData
    def cudnnConvolutionBackwardData(inputGrad: Tensor, filter: Tensor, resGrad: Tensor,
                                     padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int), counter: Int): Unit = {
      assert(resGrad.rank == 4, s"Convolution result gradient must have rank 4, but got ${resGrad.rank}")
      assert(inputGrad.rank == 4, s"Convolution input gradient must have rank 4, but got ${inputGrad.rank}")
      val one = NewArray[Float](1); one(0) = 1

      if (false)
      unchecked[Unit](
        Seq(
        s"""
          |if (!init_algo_bwd_$counter) {
          |  int input_size_$counter = """.stripMargin, inputGrad.scalarCount, s""";
          |  if (input_size_$counter * sizeof(float) > AVAIL_GPU_MEM - 1000) {
          |    init_algo_bwd_$counter = true;
          |  } else {
          |void* input_pointer_$counter = myGpuMalloc(input_size_$counter * sizeof(float));
          |cudnnConvolutionBwdDataAlgo_t algos_bwd_$counter[] = {
          |       CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
          |       CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
          |       CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
          |       CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
          |       CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
          |       CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED,
          |};
          |size_t max_sz_bwd_$counter = 0;
          |for (int c = 0; c < 6; c++) {
          |    size_t sz = 0;
          |    if (CUDNN_STATUS_SUCCESS == cudnnGetConvolutionBackwardDataWorkspaceSize(
          |        cudnnHandle, filt_desc_$counter, out_desc_$counter, conv_desc_$counter, in_desc_$counter,
          |        algos_bwd_$counter[c], &sz) && max_sz_bwd_$counter < sz)
          |        max_sz_bwd_$counter = sz;
          |}
          |max_sz_bwd_$counter = CAP_AVAIL(max_sz_bwd_$counter);
          |cudnnConvolutionBwdDataAlgoPerf_t perfResults_bwd_$counter[6];
          |int perf_count_bwd_$counter;
          |void* maxSpace_bwd_$counter = myGpuMalloc(max_sz_bwd_$counter);
         """.stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnFindConvolutionBackwardDataAlgorithmEx(\n" +
         s"    cudnnHandle, filt_desc_$counter, ", filter.data, s", out_desc_$counter, ", resGrad.data, ",\n" +
         s"    conv_desc_$counter, in_desc_$counter, input_pointer_$counter, 6, &perf_count_bwd_$counter, \n" +
         s"    perfResults_bwd_$counter, maxSpace_bwd_$counter, max_sz_bwd_$counter));\n" +
         s"myGpuFree(max_sz_bwd_$counter);\n" +
         s"myGpuFree(input_size_$counter * sizeof(float));\n" +
         s"algo_bwd_$counter = perfResults_bwd_$counter[0].algo;\n" +
         s"init_algo_bwd_$counter = true;\n}\n}\n"): _*)
      else
      unchecked[Unit](
        s"""
          |int returned_algo_count_bwd_$counter;
          |cudnnConvolutionBwdDataAlgoPerf_t perfResults_bwd_$counter[6];
          |CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm_v7(
          |    cudnnHandle,
          |    filt_desc_$counter, out_desc_$counter, conv_desc_$counter, in_desc_$counter, 6,
          |    &returned_algo_count_bwd_$counter, perfResults_bwd_$counter));
          |algo_bwd_$counter = perfResults_bwd_$counter[0].algo;
          |// algo_bwd_$counter = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
         """.stripMargin)

      unchecked[Unit](
        Seq(s"""
          |size_t ws_size_bwd_$counter;
          |CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
          |    cudnnHandle, filt_desc_$counter, out_desc_$counter, conv_desc_$counter, in_desc_$counter,
          |    algo_bwd_$counter, &ws_size_bwd_$counter));
          |void *ws_data_bwd_$counter = myGpuMalloc(ws_size_bwd_$counter);
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnConvolutionBackwardData(\n" +
          "    cudnnHandle,\n" +
          "    ", one, s", filt_desc_$counter, ", filter.data, s", out_desc_$counter, ", resGrad.data, ",\n" +
         s"    conv_desc_$counter, algo_bwd_$counter, ws_data_bwd_$counter, ws_size_bwd_$counter,\n" +
          "    ", one, s", in_desc_$counter, ", inputGrad.data, "));\n") ++
        Seq(s"myGpuFree(ws_size_bwd_$counter);\n"): _*
      )
    }

    // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardFilter
    def cudnnConvolutionBackwardFilter(filterGrad: Tensor, input: Tensor, resGrad: Tensor,
                                       padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int), counter: Int): Unit = {
      assert(resGrad.rank == 4, s"Convolution result gradient must have rank 4, got ${resGrad.rank}")
      val one = NewArray[Float](1); one(0) = 1

      if (false)
      unchecked[Unit](
        Seq(s"""
          |if (!init_algo_bwf_$counter) {
          |  int filter_size_$counter = """.stripMargin, filterGrad.scalarCount, s""";
          |  if (filter_size_$counter * sizeof(float) > AVAIL_GPU_MEM - 1000) {
          |    init_algo_bwf_$counter = true;
          |  } else {
          |void* filter_pointer_$counter = myGpuMalloc(filter_size_$counter * sizeof(float)); // filter can be overwritten by FindAlgo call!
          |cudnnConvolutionBwdFilterAlgo_t algos_bwf_$counter[] = {
          |       CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
          |       CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
          |       CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
          |       CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
          |       CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
          |       CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
          |};
          |size_t max_sz_bwf_$counter = 0;
          |for (int c = 0; c < 6; c++) {
          |    size_t sz = 0;
          |    if (CUDNN_STATUS_SUCCESS == cudnnGetConvolutionBackwardFilterWorkspaceSize(
          |        cudnnHandle, in_desc_$counter, out_desc_$counter, conv_desc_$counter, filt_desc_$counter,
          |        algos_bwf_$counter[c], &sz) && max_sz_bwf_$counter < sz)
          |        max_sz_bwf_$counter = sz;
          |}
          |max_sz_bwf_$counter = CAP_AVAIL(max_sz_bwf_$counter);
          |cudnnConvolutionBwdFilterAlgoPerf_t perfResults_bwf_$counter[6];
          |int perf_count_bwf_$counter;
          |void* maxSpace_bwf_$counter = myGpuMalloc(max_sz_bwf_$counter);
          """.stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnFindConvolutionBackwardFilterAlgorithmEx(\n" +
         s"    cudnnHandle, in_desc_$counter, ", input.data, s", out_desc_$counter, ", resGrad.data, ",\n" +
         s"    conv_desc_$counter, filt_desc_$counter, filter_pointer_$counter, 6, &perf_count_bwf_$counter,\n" +
         s"    perfResults_bwf_$counter, maxSpace_bwf_$counter, max_sz_bwf_$counter));\n" +
         s"myGpuFree(max_sz_bwf_$counter);\n" +
         s"myGpuFree(filter_size_$counter * sizeof(float));\n" +
         s"algo_bwf_$counter = perfResults_bwf_$counter[0].algo;\n" +
         s"init_algo_bwf_$counter = true;\n}\n}\n"): _*)
    else
      unchecked[Unit](
        s"""
          |int returned_algo_counter_bwf_$counter;
          |cudnnConvolutionBwdFilterAlgoPerf_t perfResults_bwf_$counter[6];
          |CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
          |    cudnnHandle,
          |    in_desc_$counter, out_desc_$counter, conv_desc_$counter, filt_desc_$counter, 6,
          |    &returned_algo_counter_bwf_$counter, perfResults_bwf_$counter));
          |algo_bwf_$counter = perfResults_bwf_$counter[0].algo;
          |// algo_bwf_$counter = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1; // should have for ResNet
          """.stripMargin)

      unchecked[Unit](
        Seq(s"""
          |size_t ws_size_bwf_$counter;
          |CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
          |    cudnnHandle, in_desc_$counter, out_desc_$counter, conv_desc_$counter, filt_desc_$counter,
          |    algo_bwf_$counter, &ws_size_bwf_$counter));
          |void *ws_data_bwf_$counter = myGpuMalloc(ws_size_bwf_$counter);
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnConvolutionBackwardFilter(\n" +
          "    cudnnHandle,\n" +
          "    ", one, s", in_desc_$counter, ", input.data, s", out_desc_$counter, ", resGrad.data, ",\n" +
         s"    conv_desc_$counter, algo_bwf_$counter, ws_data_bwf_$counter, ws_size_bwf_$counter,\n" +
          "    ", one, s", filt_desc_$counter, ", filterGrad.data, "));\n"): _*
      )
    }

    override def conv2D_batch(input: Tensor, kernel: Tensor, bias: Option[Tensor], strides: Seq[Int], pads: Seq[Int]): (Tensor, Option[Tensor], Int) ={
      // TODO: Dedupe assertions/shape calculations with CPU implementation.
      assert(input.rank == 4, s"Input must be 4-D (first dimension is batch size) but got ${input.rank}")
      assert(kernel.rank == 4, s"Kernel must be 4-D, but got ${kernel.rank}")
      bias match {
        case Some(bias) =>
          assert(bias.rank == 1, s"Bias should be 1-D, got ${bias.shape}")
          assert(bias.shape(0) == kernel.shape(0), "Bias length must equal number of out-channels")
        case None => ()
      }

      assert(strides.size == 2, "Strides should have length 2: [strideRow, strideColumn]")
      val (padH, padW) = if (pads.size == 1) (pads(0), pads(0)) else {if (pads.size == 2) (pads(0), pads(1)) else {if (pads.size == 4) (pads(0), pads(2)) else ???}}
      val ((strideRow:Int) :: (strideCol:Int) :: Nil) = strides.take(2).toList
      assert(strideRow >= 1, "Row stride must be at least 1")
      assert(strideCol >= 1, "Column stride must be at least 1")

      assert(kernel.shape(1) == input.shape(1), s"In-channel count mismatch: input.shape[1] ${input.shape(1)} should match kernel.shape[1] ${kernel.shape(1)}")
      assertC(input.shape(2) + 2 * padH >= kernel.shape(2) && input.shape(3) + 2 * padW >= kernel.shape(3), "Error")

      // Execute `cudnnConvolutionForward`.
      val resWidth = convSize(input.shape(2) + padH * 2, kernel.shape(2), strideRow)
      val resHeight = convSize(input.shape(3) + padW * 2, kernel.shape(3), strideCol)
      val resShape = Seq(input.shape(0), kernel.shape(0), resWidth, resHeight)
      val resData = mallocArray[Float](resShape.product1)
      val res = Tensor(resData, resShape: _*)
      val counterId = cudnnConvolutionForward(input, kernel, res, padding = (padH, padW), strides = (strideRow, strideCol), dilations = (1, 1))

      // If bias is defined, execute `cudnnAddBiasTensor`.
      bias match {
        case None =>
        case Some(bias) => cudnnAddBiasTensor(bias, res)
      }
      (res, None, counterId)
    }

    override def conv2D_batch_grad(input: TensorR, finput: Option[TensorR], filter: TensorR, res: TensorR, bias: Option[TensorR] = None,
                                   padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int), counterId: Int): Unit = {
      assert(input.x.rank == 4, s"convolution input values should be 4D, but got ${input.x.rank}")
      assert(input.isInput || input.d.rank == 4, s"convolution input gradients is either ignored (for training data) or should be 4D, but got ${input.d.rank}")
      if (!input.isInput) cudnnConvolutionBackwardData(input.d, filter.x, res.d, padding, strides, dilations, counterId)
      cudnnConvolutionBackwardFilter(filter.d, input.x, res.d, padding, strides, dilations, counterId)
      bias match {
        case None =>
        case Some(bias) =>
          cudnnConvolutionBackwardBias(bias.d, res.d)
      }
    }

    def Pool2D_batch(input: Tensor, kernel: Seq[Int], strides: Seq[Int], pads: Option[Seq[Int]], mode: PoolModes.Value, nanOpt: NanOpt.Value): Tensor = {
      val (windowHeight :: windowWidth :: Nil) = kernel.take(2).toList
      val (verticalPadding, horizontalPadding) = pads match {
        case None => (0, 0)
        case Some(pads) => (pads(0), pads(2))
      }
      val (verticalStride :: horizontalStride :: Nil) = strides.take(2).toList
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      val (outputHeight, outputWidth) = pads match {
        case None => (convSize(input.shape(2), kernel(0), strides(0)), convSize(input.shape(3), kernel(1), strides(1)))
        case Some(pads) => (convSize(input.shape(2), kernel(0), strides(0), pads(0)), convSize(input.shape(3), kernel(1), strides(1), pads(2)))
      }
      val output = Tensor.zeros(input.shape(0), input.shape(1), outputHeight, outputWidth)
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t in_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, input.shape(0), ", ", input.shape(1), ", ", input.shape(2), ", ", input.shape(3), """) );
          |
          |cudnnTensorDescriptor_t out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, output.shape(0), ", ", output.shape(1), ", ", output.shape(2), ", ", output.shape(3), s"""));
          |
          |cudnnPoolingDescriptor_t poolingDesc;
          |CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));
          |CUDNN_CALL(cudnnSetPooling2dDescriptor(
          |    poolingDesc, ${mode.toString}, ${nanOpt.toString},
          |    ${windowHeight}, ${windowWidth}, ${verticalPadding},
          |    ${horizontalPadding}, ${verticalStride}, ${horizontalStride}
          |));
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnPoolingForward(\n" +
          "    cudnnHandle, \n" +
          "    poolingDesc, \n" +
          "    ", one, ", in_desc, ", input.data, ", ", zero, ", out_desc, ", output.data, "));\n" +
          "}"): _*)
      output
    }

    override def maxPool2D_batch(input: Tensor, kernel: Seq[Int], strides: Seq[Int], pads: Option[Seq[Int]]): (Tensor, Option[Rep[Array[Int]]]) = {
      assert(input.rank == 4, "Currently, maxpool2D only supports inputs of rank 4")
      (Pool2D_batch(input, kernel, strides, pads, PoolModes.Max, NanOpt.NotProp), None)
    }

    def Pool2D_batch_grad(input: TensorR, output: TensorR, kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int], mode: PoolModes.Value, nanOpt: NanOpt.Value): Unit = {
      val (windowHeight :: windowWidth :: Nil) = kernel.take(2).toList
      val (verticalPadding, horizontalPadding) = (pads(0), pads(2))
      val (verticalStride :: horizontalStride :: Nil) = strides.take(2).toList
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t in_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, input.x.shape(0), ", ", input.x.shape(1), ", ", input.x.shape(2), ", ", input.x.shape(3), """));
          |
          |cudnnTensorDescriptor_t out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, output.x.shape(0), ", ", output.x.shape(1), ", ", output.x.shape(2), ", ", output.x.shape(3), s"""));
          |
          |cudnnPoolingDescriptor_t poolingDesc;
          |CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));
          |CUDNN_CALL(cudnnSetPooling2dDescriptor(
          |    poolingDesc, ${mode.toString}, ${nanOpt.toString},
          |    ${windowHeight}, ${windowWidth}, ${verticalPadding},
          |    ${horizontalPadding}, ${verticalStride}, ${horizontalStride}
          |));
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnPoolingBackward(\n" +
          "    cudnnHandle, \n" +
          "    poolingDesc, \n" +
          "    ", one, ", out_desc, ", output.x.data, ", out_desc, ", output.d.data, ", in_desc, ", input.x.data,
          "  , ", zero, ", in_desc, ", input.d.data, "));\n" +
          "}"): _*)
    }

    override def maxPool2D_batch_grad(input: TensorR, output: TensorR, sidx: Option[Rep[Array[Int]]], kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Unit = {
      Pool2D_batch_grad(input, output, kernel, strides, pads, PoolModes.Max, NanOpt.NotProp)
    }

    override def averagePool2D_batch(input: Tensor, kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Tensor = {
      assert(input.rank == 4, "Current, averagePool2D_batch only supports inputs of rank 4")
      Pool2D_batch(input, kernel, strides, Some(pads), PoolModes.AverageEP, NanOpt.NotProp)
    }

    override def averagePool2D_batch_grad(input: TensorR, output: TensorR, kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Unit = {
      Pool2D_batch_grad(input, output, kernel, strides, pads, PoolModes.AverageEP, NanOpt.NotProp)
    }

    def cudnnBatchNormalizationForwardInference(x: Tensor, res: Tensor, scale: Tensor, bias: Tensor,
                                                runningMean: Tensor, runningVar: Tensor,
                                                momentum: Double = 1.0, epsilon: Double = 1e-5): Unit = {
      val biasShape: Seq[Rep[Int]] =
        if (bias.rank == 1) Seq(1, bias.shape(0), 1, 1)
        else if (bias.rank == 4) bias.shape.dims
        else {System.out.println(s"bias.rank is not 1 or 4 but ${bias.rank}"); ???}
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t in_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, x.shape(0), ", ", x.shape(1), ", ", x.shape(2), ", ", x.shape(3), """));
          |
          |cudnnTensorDescriptor_t out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, res.shape(0), ", ", res.shape(1), ", ", res.shape(2), ", ", res.shape(3), """));
          |
          |cudnnTensorDescriptor_t sbmv_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, biasShape(0), ", ", biasShape(1), ", ", biasShape(2), ", ", biasShape(3), """));
          |
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnBatchNormalizationForwardInference(\n" +
          // "    cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,\n" +
          "    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,\n" +
          "    ", one, ", ", one, ", in_desc, ", x.data, ", out_desc, ", res.data, ", sbmv_desc, ", scale.data, ",\n" +
          "    ", bias.data, ", ", runningMean.data, ", ", runningVar.data, ", ", epsilon, "));\n" +
          "}"): _*)
    }

    // TODO (Fei Wang): What is proper value for momentum (or should be called exponentialAverageFactor) here?
    def cudnnBatchNormalizationForwardTraining(x: Tensor, res: Tensor, scale: Tensor, bias: Tensor,
                                               runningMean: Tensor, runningVar: Tensor, saveMean: Tensor, saveInvVariance: Tensor,
                                               momentum: Double = 0.1, epsilon: Double = 1e-5): Int = {
      val biasShape =
        if (bias.rank == 1) Seq(1, bias.shape(0), 1, 1)
        else if (bias.rank == 4) bias.shape.dims
        else {System.out.println(s"bias rank is not 1 or 4, but ${bias.rank}"); ???}
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1

      val counter = nextKernel
      nextKernel += 1

      unchecked[Unit](
        Seq(s"""
          |cudnnTensorDescriptor_t in_desc_$counter;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_$counter));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    in_desc_$counter, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, x.shape(0), ", ", x.shape(1), ", ", x.shape(2), ", ", x.shape(3), s"""));
          |
          |cudnnTensorDescriptor_t out_desc_$counter;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_$counter));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    out_desc_$counter, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, res.shape(0), ", ", res.shape(1), ", ", res.shape(2), ", ", res.shape(3), s"""));
          |
          |cudnnTensorDescriptor_t sbmv_desc_$counter;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc_$counter));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    sbmv_desc_$counter, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, biasShape(0), ", ", biasShape(1), ", ", biasShape(2), ", ", biasShape(3), """));
          |
          |""".stripMargin):_*)
      unchecked[Unit](
        Seq(
          "CUDNN_CALL(cudnnBatchNormalizationForwardTraining(\n" +
          // "    cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,\n" +
          "    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,\n" +
          "    ", one, ", ", zero, s", in_desc_$counter, ", x.data, s", out_desc_$counter, ", res.data, s", sbmv_desc_$counter, ", scale.data, ",\n" +
          "    ", bias.data, ", ", momentum, ", ", runningMean.data, ", ", runningVar.data, ", ", epsilon, ",\n" +
          "    ", saveMean.data, ", ", saveInvVariance.data, "));\n"): _*)
       counter
    }

    def cudnnBatchNormalizationBackward(input: TensorR, res: TensorR, scale: TensorR, bias: TensorR,
                                        saveMean: Tensor, saveInvVariance: Tensor, counter: Int,
                                        momentum: Double = 1.0, epsilon: Double = 1e-5): Unit = {
      val biasShape =
        if (bias.x.rank == 1) Seq(1, bias.x.shape(0), 1, 1)
        else if (bias.x.rank == 4) bias.x.shape.dims
        else {System.out.println(s"bias rank is not 1 or 4, but ${bias.x.rank}"); ???}
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1

      unchecked[Unit](
        Seq(
          "CUDNN_CALL(cudnnBatchNormalizationBackward(\n" +
          // "    cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,\n" +
          "    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,\n" +
          "    ", one, ", ", one, ", ", one, ", ", one, s", in_desc_$counter, ", input.x.data, ",\n" +
          s"    out_desc_$counter, ", res.d.data, s", in_desc_$counter, ", input.d.data, s", sbmv_desc_$counter, ", scale.x.data, ",\n" +
          "    ", scale.d.data, ",", bias.d.data, ", ", epsilon, ", ", saveMean.data, ", ", saveInvVariance.data, "));\n"): _*)
    }

    override def batchNormInference(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): Tensor = {
      val res = Tensor(mallocArray[Float](x.scalarCount), x.shape: _*)
      cudnnBatchNormalizationForwardInference(x, res, scale, bias, runningMean, runningVar)
      res
    }

    override def batchNormTraining(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): (Tensor, Option[Tensor], Option[Tensor], Int) = {
      val res = Tensor(mallocArray[Float](x.scalarCount), x.shape: _*)
      val saveMean = Tensor(mallocArray[Float](bias.scalarCount), bias.shape: _*)
      val saveInvVariance = Tensor(mallocArray[Float](bias.scalarCount), bias.shape: _*)
      val counterId = cudnnBatchNormalizationForwardTraining(x, res, scale, bias, runningMean, runningVar, saveMean, saveInvVariance)
      (res, Some(saveMean), Some(saveInvVariance), counterId)
    }

    override def batchNorm_grad(input: TensorR, res: TensorR, scale: TensorR, bias: TensorR,
                                saveMean: Option[Tensor], saveInvVariance: Option[Tensor], counterId: Int): Unit = {
      (saveMean, saveInvVariance) match {
        case (Some(saveMean), Some(saveInvVariance)) => cudnnBatchNormalizationBackward(input, res, scale, bias, saveMean, saveInvVariance, counterId)
        case _ => ???
      }
    }

    def cudnnBatchNormalization1DForwardInference(x: Tensor, res: Tensor, scale: Tensor, bias: Tensor,
                                                  runningMean: Tensor, runningVar: Tensor,
                                                  momentum: Double = 0.1, epsilon: Double = 1e-5): Unit = {
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t in_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, x.shape(0), ", ", x.shape(1), """, 1, 1));
          |
          |cudnnTensorDescriptor_t sbmv_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    1, """.stripMargin, bias.shape(0), """, 1, 1));
          |
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnBatchNormalizationForwardInference(\n" +
          "    cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,\n" +
         // "    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,\n" +
          "    ", one, ", ", one, ", in_desc, ", x.data, ", in_desc, ", res.data, ", sbmv_desc, ", scale.data, ",\n" +
          "    ", bias.data, ", ", runningMean.data, ", ", runningVar.data, ", ", epsilon, "));\n" +
          "}"): _*)
    }

    def cudnnBatchNormalization1DForwardTraining(x: Tensor, res: Tensor, scale: Tensor, bias: Tensor,
                                               runningMean: Tensor, runningVar: Tensor, saveMean: Tensor, saveInvVariance: Tensor,
                                               momentum: Double = 0.1, epsilon: Double = 1e-5): Int = {
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1

      val counter = nextKernel
      nextKernel += 1

      unchecked[Unit](
        Seq(s"""
          |cudnnTensorDescriptor_t in_desc_$counter;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_$counter));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    in_desc_$counter, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, x.shape(0), ", ", x.shape(1), s""", 1, 1));
          |
          |cudnnTensorDescriptor_t sbmv_desc_$counter;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc_$counter));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    sbmv_desc_$counter, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    1, """.stripMargin, bias.shape(0), """, 1, 1));
          |
          |""".stripMargin):_*)

      unchecked[Unit](
        Seq(
          "CUDNN_CALL(cudnnBatchNormalizationForwardTraining(\n" +
          "    cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,\n" +
          // "    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,\n" +
          "    ", one, ", ", zero, s", in_desc_$counter, ", x.data, s", in_desc_$counter, ", res.data, s", sbmv_desc_$counter, ", scale.data, ",\n" +
          "    ", bias.data, ", ", momentum, ", ", runningMean.data, ", ", runningVar.data, ", ", epsilon, ",\n" +
          "    ", saveMean.data, ", ", saveInvVariance.data, "));\n"): _*)
      counter
    }

    def cudnnBatchNormalization1DBackward(input: TensorR, res: TensorR, scale: TensorR, bias: TensorR,
                                        saveMean: Tensor, saveInvVariance: Tensor, counter: Int,
                                        momentum: Double = 1.0, epsilon: Double = 1e-5): Unit = {
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1

      unchecked[Unit](
        Seq(
          "CUDNN_CALL(cudnnBatchNormalizationBackward(\n" +
          "    cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,\n" +
          // "    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,\n" +
          "    ", one, ", ", one, ", ", one, ", ", one, s", in_desc_$counter, ", input.x.data, ",\n" +
         s"    in_desc_$counter, ", res.d.data, s", in_desc_$counter, ", input.d.data, s", sbmv_desc_$counter, ", scale.x.data, ",\n" +
          "    ", scale.d.data, ",", bias.d.data, ", ", epsilon, ", ", saveMean.data, ", ", saveInvVariance.data, "));\n"): _*)
    }

    @virtualize
    override def batchNorm1DInference(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): Tensor = {
      assert(x.rank == 2, s"batchNorm1D only applies to inputs of 2D matrix, got ${x.shape}")
      assert(scale.rank == 1, s"scale should be rank 1, got ${scale.rank}")
      assert(scale.shape(0) == x.shape(1), s"scale should have the same size as input dim 1, got ${scale.shape(0)} and ${x.shape(1)}")
      assert(bias.rank == 1 && bias.shape(0) == x.shape(1), s"bias should be rank 1 and have the same size as input dim 1, got ${bias.shape} and ${x.shape}")
      assert(runningMean.rank == 1 && runningMean.shape(0) == x.shape(1), s"runningMean should be rank 1 and have the same size as input dim 1, got ${runningMean.shape} and ${x.shape}")
      assert(runningVar.rank == 1 && runningVar.shape(0) == x.shape(1), s"runningVar should be rank 1 and have the same size as input dim 1, got ${runningVar.shape} and ${x.shape}")
      val res = Tensor(mallocArray[Float](x.scalarCount), x.shape: _*)
      cudnnBatchNormalization1DForwardInference(x, res, scale, bias, runningMean, runningVar)
      res
    }

    @virtualize
    override def batchNorm1DTraining(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): (Tensor, Option[Tensor], Option[Tensor], Int) = {
      assert(x.rank == 2, s"batchNorm1D only applies to inputs of 2D matrix, got ${x.shape}")
      assert(scale.rank == 1)
      assert(scale.shape(0) == x.shape(1), s"scale should be rank 1 and have the same size as input dim 1, got ${scale.shape} and ${x.shape}")
      assert(bias.rank == 1)
      assert(bias.shape(0) == x.shape(1), s"bias should be rank 1 and have the same size as input dim 1, got ${bias.shape} and ${x.shape}")
      assert(runningMean.rank == 1)
      assert(runningMean.shape(0) == x.shape(1), s"runningMean should be rank 1 and have the same size as input dim 1, got ${runningMean.shape} and ${x.shape}")
      assert(runningVar.rank == 1)
      assert(runningVar.shape(0) == x.shape(1), s"runningVar should be rank 1 and have the same size as input dim 1, got ${runningVar.shape} and ${x.shape}")
      val res = Tensor(mallocArray[Float](x.scalarCount), x.shape: _*)
      val saveMean = Tensor(mallocArray[Float](bias.scalarCount), bias.shape: _*)
      val saveInvVariance = Tensor(mallocArray[Float](bias.scalarCount), bias.shape: _*)
      val counterId = cudnnBatchNormalization1DForwardTraining(x, res, scale, bias, runningMean, runningVar, saveMean, saveInvVariance)
      (res, Some(saveMean), Some(saveInvVariance), counterId)
    }

    override def batchNorm1D_grad(input: TensorR, res: TensorR, scale: TensorR, bias: TensorR, saveMean: Option[Tensor], saveInvVariance: Option[Tensor], counterId: Int): Unit = {
      (saveMean, saveInvVariance) match {
        case (Some(saveMean), Some(saveInvVariance)) => cudnnBatchNormalization1DBackward(input, res, scale, bias, saveMean, saveInvVariance, counterId)
        case _ => ???
      }
    }

    override def dropout(input: Tensor, prob: Float = 0.5f): (Tensor, Rep[Array[Float]], Rep[Int]) = {
      val output = Tensor.zeros_like(input)
      val reserveSpace = var_new(unchecked[Array[Float]]("(float*)NULL"))
      val sizeInBytes = var_new(unchecked[Int]("0"))
      val padShape = input.shape.padTo(4, unit(1)) // pad the dimension to 4D
      unchecked[Unit](
        s"""
          |{
          |cudnnTensorDescriptor_t in_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, padShape(0), ", ", padShape(1), ", ", padShape(2), ", ", padShape(3), s"""));
          |
          |size_t stateSizeInBytes;
          |CUDNN_CALL(cudnnDropoutGetStatesSize(
          |    cudnnHandle, &stateSizeInBytes
          |));
          |void* state = myGpuMalloc(stateSizeInBytes);
          |
          |size_t sizeInBytes;
          |CUDNN_CALL(cudnnDropoutGetReserveSpaceSize(
          |    in_desc, &sizeInBytes
          |));
          |void* reserveSpace = myGpuMalloc(sizeInBytes);
          |
          |""".stripMargin,
          reserveSpace, " = (float*)reserveSpace;\n",
          sizeInBytes, " = (int)sizeInBytes;\n",
        s"""
          |cudnnDropoutDescriptor_t dropoutDesc;
          |CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropoutDesc));
          |CUDNN_CALL(cudnnSetDropoutDescriptor(
          |    dropoutDesc, cudnnHandle, ${prob}, state, stateSizeInBytes, time(NULL)
          |));
          |""".stripMargin,

          "CUDNN_CALL(cudnnDropoutForward(\n" +
          "    cudnnHandle,\n" +
          "    dropoutDesc,\n" +
          "    in_desc, ", input.data, ", in_desc, ", output.data, ", ", "reserveSpace, sizeInBytes));\n" +
          "}")
      (output, reserveSpace, sizeInBytes)
    }

    override def dropout_grad(input: TensorR, output: TensorR, prob: Float, helper: Rep[Array[Float]], size: Rep[Int]): Unit = {
      val padShape = input.x.shape.padTo(4, unit(1)) // pad the dimension to 4D
      unchecked[Unit](
        s"""
          |{
          |cudnnTensorDescriptor_t in_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, padShape(0), ", ", padShape(1), ", ", padShape(2), ", ", padShape(3), s"""));
          |
          |size_t stateSizeInBytes;
          |CUDNN_CALL(cudnnDropoutGetStatesSize(
          |    cudnnHandle, &stateSizeInBytes
          |));
          |void* state = myGpuMalloc(stateSizeInBytes);
          |
          |size_t sizeInBytes;
          |CUDNN_CALL(cudnnDropoutGetReserveSpaceSize(
          |    in_desc, &sizeInBytes
          |));
          |void* reserveSpace = myGpuMalloc(sizeInBytes);
          |
          |cudnnDropoutDescriptor_t dropoutDesc;
          |CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropoutDesc));
          |CUDNN_CALL(cudnnSetDropoutDescriptor(
          |    dropoutDesc, cudnnHandle, ${prob}, state, stateSizeInBytes, time(NULL)
          |));
          |""".stripMargin,
          "CUDNN_CALL(cudnnDropoutBackward(\n" +
          "    cudnnHandle,\n" +
          "    dropoutDesc,\n" +
          "    in_desc, ", output.d.data, ", in_desc, ", input.d.data, ", (void*)", helper, ", (size_t)", size, "));\n" +
          "}")
    }

    // multihead attention
    override def multiheadAttention(query: TensorR, key: TensorR, value: TensorR, weights: TensorR, numHeads: Int, embedDim:Int, 
      qSeqArray: Rep[Array[Int]], kSeqArray: Rep[Array[Int]], loWinIdx: Rep[Array[Int]], hiWinIdx: Rep[Array[Int]], bias: Boolean,
      dropoutRate :Float = 0.0f, smScaler: Float = 1.0f): (Tensor, Rep[Array[Float]], Rep[Int], Rep[Array[Float]], Rep[Int], Rep[Int], Rep[Array[Int]], Rep[Array[Int]]) = 
      {
        cudnnMultiheadAttnForward(query, key, value, weights, numHeads, embedDim, qSeqArray, kSeqArray, loWinIdx, hiWinIdx, bias, dropoutRate, smScaler)
      }

    override def multiheadAttention_grad(output: TensorR, query: TensorR, key: TensorR, value: TensorR, weights: TensorR, numHeads: Int, embedDim:Int, 
      qSeqArray: Rep[Array[Int]], kSeqArray: Rep[Array[Int]], devQSeqArray: Rep[Array[Int]], devKSeqArray: Rep[Array[Int]], loWinIdx: Rep[Array[Int]],
       hiWinIdx: Rep[Array[Int]], bias: Boolean, dropoutRate :Float = 0.0f, smScaler: Float = 1.0f, devWkSpace: Rep[Array[Float]], sizeWkSpace: Rep[Int], 
       devReserve: Rep[Array[Float]], sizeReserve: Rep[Int], sizeWeights: Rep[Int]): Unit = 
      {
        cudnnMultiHeadAttnBackward(output, query, key, value, weights, numHeads, embedDim, qSeqArray, kSeqArray, 
        devQSeqArray, devKSeqArray, loWinIdx, hiWinIdx, bias, dropoutRate, smScaler, devWkSpace, sizeWkSpace, devReserve, sizeReserve, sizeWeights)
      }


    def cudnnActivationForward(x: Tensor, activation: Activation.Value, inPlace: Boolean = false): Tensor = {
      val xShape = x.shape.padTo(4, unit(1)) //activation functions only support tensors of rank 4
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      val res = if (inPlace) x else Tensor(mallocArray[Float](x.scalarCount), x.shape: _*)
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t x_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, xShape(0), ", ", xShape(1), ", ", xShape(2), ", ", xShape(3), s"""));
          |
          |cudnnActivationDescriptor_t act_desc;
          |CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
          |CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
          |                                        /*mode=*/ ${activation.toString},
          |                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
          |                                        /*relu_coef=*/ 0));
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnActivationForward(\n" +
          "    cudnnHandle, act_desc,\n" +
          "    ", one, ", x_desc, ", x.data, ", ", zero, ", x_desc, ", res.data, "));\n" +
          "}"): _*
      )
      res
    }

    def cudnnActivationBackward(input: TensorR, res: TensorR, activation: Activation.Value, inPlace: Boolean = false): Unit = {
      val inputXShape = input.x.shape.padTo(4, unit(1)) // activation functions only support tensors of rank 4
      Tensor.assertShapeEqual(input.x.shape, res.x.shape)
      Tensor.assertShapeEqual(input.d.shape, res.d.shape)
      val one = NewArray[Float](1); one(0) = 1
      val zero = NewArray[Float](1); zero(0) = 0
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t x_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, inputXShape(0), ", ", inputXShape(1), ", ", inputXShape(2), ", ", inputXShape(3), s"""));
          |
          |cudnnActivationDescriptor_t act_desc;
          |CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
          |CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
          |                                        /*mode=*/ ${activation.toString},
          |                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
          |                                        /*relu_coef=*/ 0));
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnActivationBackward(\n" +
          "    cudnnHandle, act_desc,\n" +
          "    ", one, ", x_desc, ", res.x.data, ", x_desc, ", res.d.data, ", x_desc, ", input.x.data, ",\n",
          "    ", (if (inPlace) zero else one), ", x_desc, ", input.d.data, "));\n" +
          "}"): _*
      )
    }

    override def relu(x: Tensor, inPlace: Boolean = false): Tensor = {
      cudnnActivationForward(x, Activation.Relu, inPlace)
    }
    override def relu_grad(input: TensorR, res: TensorR, inPlace: Boolean = false): Unit = {
      cudnnActivationBackward(input, res, Activation.Relu, inPlace)
    }

    override def tanh(x: Tensor): Tensor = {
      cudnnActivationForward(x, Activation.Tanh)
    }
    override def tanh_grad(input: TensorR, res: TensorR): Unit = {
      cudnnActivationBackward(input, res, Activation.Tanh)
    }

    override def sigmoid(x: Tensor): Tensor = {
      cudnnActivationForward(x, Activation.Sigmoid)
    }
    override def sigmoid_grad(input: TensorR, res: TensorR): Unit = {
      cudnnActivationBackward(input, res, Activation.Sigmoid)
    }

    def cudnnSoftmaxForward(x: Tensor, mode: SoftmaxMode.Value): Tensor = {
      assert(x.rank == 4, s"Softmax kernel only takes tensors of rank 4, and reduce on dim 1. Reshape your tensor accordingly before using this function. Got ${x.shape}")
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      val res = Tensor(mallocArray[Float](x.scalarCount), x.shape: _*)
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t x_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, x.shape(0), ", ", x.shape(1), ", ", x.shape(2), ", ", x.shape(3), """));
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnSoftmaxForward(\n" +
          s"    cudnnHandle, ${mode.toString}, CUDNN_SOFTMAX_MODE_CHANNEL,\n" +
          "    ", one, ", x_desc, ", x.data, ", ", zero, ", x_desc, ", res.data, "));\n" +
          "}"): _*
      )
      res
    }

    def cudnnSoftmaxBackward(input: TensorR, res: TensorR, mode: SoftmaxMode.Value): Unit = {
      assert(input.x.rank == 4, s"SoftmaxBackward kernel only takes tensors of rank 4, and reduce on dim 1. Reshape your tensor accordingly before using this function. Got ${input.x.shape}")
      // NOTE: shape assertions are relaxed.
      // Assume that {input/result * forward/backward} values all have the same shape.
      // The shape of the input forward value is used in the generated code.
      Tensor.assertShapeEqual(input.x.shape, res.x.shape)
      Tensor.assertShapeEqual(input.d.shape, res.d.shape)
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t x_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, input.x.shape(0), ", ", input.x.shape(1), ", ", input.x.shape(2), ", ", input.x.shape(3), """));
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnSoftmaxBackward(\n" +
          s"    cudnnHandle, ${mode.toString}, CUDNN_SOFTMAX_MODE_CHANNEL,\n" +
          "    ", one, ", x_desc, ", res.x.data, ", x_desc, ", res.d.data, ",\n" +
          "    ", one, ", x_desc, ", input.d.data, "));\n" +
          "}"): _*
      )
    }

    def softmaxHelper(x: Tensor, dim: Int, mode: SoftmaxMode.Value): Tensor = {
      assert(dim >= 0 && dim < x.rank, s"dim should be in range of input rank, got ${x.shape}, ${dim}")
      val tmpIn = x.resizeNoCheck(x.shape.take(dim).product1, x.shape(dim), x.shape.drop(dim+1).product1, 1)
      val tmpOut = cudnnSoftmaxForward(tmpIn, mode)
      val res = tmpOut.resizeNoCheck(x.shape: _*)
      res
    }

    override def softmax(x: Tensor, dim: Int = 1): Tensor = softmaxHelper(x, dim, SoftmaxMode.Accurate)
    override def logSoftmax(x: Tensor, dim: Int = 1): Tensor = softmaxHelper(x, dim, SoftmaxMode.Log)

    def softmaxBackwardHelper(input: TensorR, res: TensorR, dim: Int, mode: SoftmaxMode.Value): Unit = {
      assert(dim >= 0 && dim < input.x.rank, s"dim should be in range of input rank, got ${input.x.shape}, ${dim}")
      val tmpIn = new TensorR(input.x.resizeNoCheck(input.x.shape.take(dim).product1, input.x.shape(dim), input.x.shape.drop(dim+1).product1, 1),
                              input.d.resizeNoCheck(input.x.shape.take(dim).product1, input.x.shape(dim), input.x.shape.drop(dim+1).product1, 1))
      val tmpOut = new TensorR(res.x.resizeNoCheck(res.x.shape.take(dim).product1, res.x.shape(dim), res.x.shape.drop(dim+1).product1, 1),
                               res.d.resizeNoCheck(res.x.shape.take(dim).product1, res.x.shape(dim), res.x.shape.drop(dim+1).product1, 1))
      cudnnSoftmaxBackward(tmpIn, tmpOut, mode)
    }

    override def softmax_grad(input: TensorR, res: TensorR, dim: Int = 1): Unit =
      softmaxBackwardHelper(input, res, dim, SoftmaxMode.Accurate)
    override def logSoftmax_grad(input: TensorR, res: TensorR, dim: Int = 1): Unit = {
      softmaxBackwardHelper(input, res, dim, SoftmaxMode.Log)
    }

    def cudnnReduceUpdateTensor(receiver: Tensor, rDim: Dimensions, provider: Tensor, pDim: Dimensions, alpha: Rep[Array[Float]], beta: Rep[Array[Float]], op: ReductionOp.Value = ReductionOp.Add): Unit = {
      val rShape: Seq[Rep[Int]] = rDim.dims.padTo(4, unit(1))
      val pShape: Seq[Rep[Int]] = pDim.dims.padTo(4, unit(1))
      cudnnReduceTensorUnchecked(pShape, provider.data, rShape, receiver.data, op, alpha, beta)
    }

    def cudnnReduceTensorUnchecked(xShape: Seq[Rep[Int]], xData: Rep[Array[Float]], resShape: Seq[Rep[Int]], resData: Rep[Array[Float]], op: ReductionOp.Value, alpha: Rep[Array[Float]], beta: Rep[Array[Float]]) = {
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t x_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, xShape(0), ", ", xShape(1), ", ", xShape(2), ", ", xShape(3), """));
          |
          |cudnnTensorDescriptor_t out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    """.stripMargin, resShape(0), ", ", resShape(1), ", ", resShape(2), ", ", resShape(3), s"""));
          |
          |cudnnReduceTensorDescriptor_t reduce_desc;
          |CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&reduce_desc));
          |CUDNN_CALL(cudnnSetReduceTensorDescriptor(
          |    reduce_desc, ${op.toString}, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
          |    CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));
          |
          |void *indices = nullptr; // Don't store indices.
          |
          |// Workspace.
          |size_t ws_size;
          |CUDNN_CALL(cudnnGetReductionWorkspaceSize(
          |    cudnnHandle, reduce_desc, x_desc, out_desc, &ws_size));
          |void *ws_data = myGpuMalloc(ws_size);
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnReduceTensor(\n" +
          s"    cudnnHandle, reduce_desc, indices, 0, ws_data, ws_size,\n" +
          "    ", alpha, ", x_desc, ", xData, ", ", beta, ", out_desc, ", resData, "));\n" +
          "}"): _*
      )
    }

    // TODO: Relax rank 4 requirement after implementing tensor descriptor helper functions.
    // `cudnnReduceTensor` supports tensors up to dimension 8.
    @virtualize
    def cudnnReduceTensor(x: Tensor, op: ReductionOp.Value, indices: Seq[Int], flatten: Boolean = true, toTensor: Option[Rep[Array[Float]]] = None, clear: Boolean = true): Tensor = {
      assert(indices.forall(i => x.shape.indices.contains(i)), s"Indices out of bounds: $indices, tensor shape is ${x.shape}")
      val xShape: Seq[Rep[Int]] = x.shape.padTo(4, unit(1))
      val unflatShape: Seq[Rep[Int]] = xShape.zipWithIndex.map { case (dim, i) =>
        if (indices.contains(i)) unit(1) else dim
      }
      // TODO: if shape is the same, return x??
      if ((xShape zip unflatShape).forallR{case (a, b) => a == b})
        assertC(false, "shape of reduction kernels should not be equal")
      val res = toTensor match {
        case None => Tensor(mallocArray[Float](unflatShape.product1), unflatShape: _*)
        case Some(array) => Tensor(array, unflatShape: _*)
      }
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      cudnnReduceTensorUnchecked(xShape, x.data, res.shape, res.data, op, one, (if (clear) zero else one))
      val resShape: Seq[Rep[Int]] = x.shape.zipWithIndex.flatMap { case (dim, i) =>
        if (indices.contains(i)) if (flatten) None else Some(unit(1)) else Some(dim)
      }

      // TODO: Remove if expression when rank-0 tensor support is fixed.
      if (resShape.isEmpty) Tensor(res.data, 1)
      else Tensor(res.data, resShape: _*)
    }

    @virtualize
    override def sum(x: Tensor): Tensor = {
      val xx = x.resizeNoCheck(x.shape.padTo(4, unit(1)): _*)
      if (x.isScalar) x.resizeNoCheck(1) // TODO: double check if returning x (not a copy) is OK
      else {
        val res = cudnnReduceTensor(xx, ReductionOp.Add, xx.shape.indices)
        res.resizeNoCheck(1)
      }
    }

    override def sum_grad(input: TensorR, res: TensorR): Unit = {
      generate_comment("backprop for sum op")
      assert(res.d.shape.dims == Seq(unit(1)), s"result of sum reduce should be scalar, got ${res.d.shape}")
      unchecked[Unit](s"addScalarInArrayInPlace<<<28, 512>>>(", input.d.data, ", ", res.d.data, ", ", 1.0f, ", ", input.d.scalarCount, ")")
    }

    override def mean(x: Tensor): Tensor = {
      val xx = x.resize(x.shape.padTo(4, unit(1)): _*)
      val res = cudnnReduceTensor(xx, ReductionOp.Avg, xx.shape.indices)
      res.resize(1)
    }

    override def mean_grad(input: TensorR, res: TensorR): Unit = {
      generate_comment("backprop for mean op")
      assert(res.d.shape.dims == Seq(unit(1)), s"result of mean reduce should be scalar, got ${res.d.shape}")
      unchecked[Unit](s"addScalarInArrayInPlace<<<28, 512>>>(", input.d.data, ", ", res.d.data, ", ", 1.0f/input.x.scalarCount, ", ", input.d.scalarCount, ")")
    }

    @virtualize
    override def sum(x: Tensor, dim: Int): Tensor = {
      assert(dim >= 0 && dim < x.rank, s"dim should be in range, got ${dim} from ${x.shape}")
      val resShape: Seq[Rep[Int]] = x.shape.take(dim) ++ x.shape.drop(dim+1)
      val resData = if (x.shape(dim) < 5) {
        val resData = mallocArray[Float](resShape.scalarCount)
        val inputStride = x.shape.strides.padTo(4, 1)
        val outputStride = resShape.strides.padTo(3, 1)
        generate_comment("optimization for dimension sum if size is small")
        unchecked[Unit](
          "sum_optimization<<<28, 512>>>(", x.data, ", ", inputStride(0), ", ", inputStride(1), ", ", inputStride(2), ", ", inputStride(3), ", ", resData, ", ", outputStride(0), ", ", outputStride(1), ", ", outputStride(2), ", ", dim, ", ", resShape.product1, ", ", x.shape(dim), ");\n")
        resData
      } else {
        val xx = x.resizeNoCheck(x.shape.padTo(4, unit(1)): _*)
        val indices = dim +: ((x.rank until xx.rank): Range).toSeq
        val ret = cudnnReduceTensor(xx, ReductionOp.Add, indices)
        ret.data
      }
      new Tensor(resData, resShape)
    }

    override def sum_grad(input: TensorR, output: TensorR, dim: Int): Unit = {
      // TODO: (Fei Wang) there are limitations in cudnnAddBiasTensor (dim 0 must be 1). So we need user-defined kernel for this!!
      assert(input.x.rank == output.x.rank + 1, s"input should be 1 rank higher than the output, got ${input.x.shape}, ${output.x.shape}")
      val inputShape = input.x.shape.padTo(4, 1)
      val outputStride = output.x.shape.strides.padTo(3, 1)
      generate_comment("backprop for sum on dim op")
      unchecked[Unit](
        "sum_grad<<<28, 512>>>(", input.d.data, ", ", inputShape(0), ", ", inputShape(1), ", ", inputShape(2), ", ", inputShape(3), ", ", input.x.scalarCount, ", ",
        output.d.data, ", ", outputStride(0), ", ", outputStride(1), ", ", outputStride(2), ", ", dim, ");\n")
    }

    def cudnnRNNForwardHelper(mode: RnnMode,
                              training: Boolean,
                              x: Tensor, hx: Option[Tensor], cx: Option[Tensor], w: Tensor,
                              numLayers: Int, hiddenSize: Int,
                              dropout: Float = 0f,
                              bidirectional: Boolean = false): (Tensor, Option[Tensor], Option[(Rep[Array[Float]], Rep[Int])], Int) = {
      assert(x.rank == 3, "RNN input should have rank 3: [seqLength x batchSize x inputSize]")
      hx match {
        case None =>
        case Some(hx) =>
          assert(hx.rank == 3, "RNN hidden state should have rank 3: [numLayers * numDirections x batchSize x hiddenSize]")
          assert(x.shape(1) == hx.shape(1), "RNN hidden state second dimension should equal input second dimension (batch size)")
      }
      cx match {
        case None =>
        case Some(cx) =>
          assert(cx.rank == 3, "RNN hidden state should have rank 3: [numLayers * numDirections x batchSize x hiddenSize]")
          assert(x.shape(1) == cx.shape(1), "RNN hidden state second dimension should equal input second dimension (batch size)")
      }
      val hxData = hx.map(_.data).getOrElse(unchecked[Array[Float]]("(float*)NULL"))
      val cxData = cx.map(_.data).getOrElse(unchecked[Array[Float]]("(float*)NULL"))
      // TODO: Optionally calculate final hidden state `hy` based on flag.
      val hy = hx.map(hx => Tensor(mallocArray[Float](hx.scalarCount), hx.shape: _*))
      val hyData = hy.map(_.data).getOrElse(unchecked[Array[Float]]("(float*)NULL"))

      val seqLength = x.shape(0)
      val batchSize = x.shape(1)
      val inputSize = x.shape(2)
      val numDirections = if (bidirectional) 2 else 1

      val resShape: Seq[Rep[Int]] = Seq(seqLength, batchSize, hiddenSize * numDirections)
      val res = Tensor(mallocArray[Float](resShape.product1), resShape: _*)

      val counter = nextKernel
      nextKernel += 1

      // set up descriptors (need a counter to avoid name clasion)
      unchecked[Unit](
        Seq(s"""
         |size_t dropoutStateSize_$counter;
         |CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize_$counter));
         |void* dropoutStates_$counter = ${if (dropout == 0f) "NULL;" else s"myGpuMalloc(dropoutStateSize_$counter);"}
         |
         |cudnnDropoutDescriptor_t dropout_desc_$counter;
         |CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc_$counter));
         |CUDNN_CALL(cudnnSetDropoutDescriptor(
         |    dropout_desc_$counter, cudnnHandle, $dropout, dropoutStates_$counter, dropoutStateSize_$counter, time(NULL)));
         |
         |cudnnRNNDescriptor_t rnn_desc_$counter;
         |CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc_$counter));
         |CUDNN_CALL(cudnnSetRNNDescriptor(
         |    cudnnHandle, rnn_desc_$counter,
         |    /*hiddenSize*/ $hiddenSize, /*numLayers*/ $numLayers,
         |    dropout_desc_$counter, CUDNN_LINEAR_INPUT, ${if(bidirectional) "CUDNN_BIDIRECTIONAL" else "CUDNN_UNIDIRECTIONAL"},
         |    ${mode.toString}, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
         """.stripMargin) ++
        cudnnMathType.map(mathType => Seq(s"CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc_$counter, $mathType));")).getOrElse(Seq()) ++
        Seq(s"""
         |int32_t seqLength_$counter = """.stripMargin, seqLength, s""";
         |int32_t batchSize_$counter = """.stripMargin, batchSize, s""";
         |int32_t inputSize_$counter = """.stripMargin, inputSize, s""";
         |
         |cudnnTensorDescriptor_t x_descs_$counter[seqLength_$counter];
         |cudnnTensorDescriptor_t x_desc_$counter;
         |CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc_$counter));
         |int x_dims_$counter[] = {batchSize_$counter, inputSize_$counter, 1};
         |int x_strides_$counter[] = {x_dims_$counter[1] * x_dims_$counter[2], x_dims_$counter[2], 1};
         |CUDNN_CALL(cudnnSetTensorNdDescriptor(
         |    x_desc_$counter, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims_$counter, x_strides_$counter));
         |for (int i = 0; i < seqLength_$counter; i++) {
         |  x_descs_$counter[i] = x_desc_$counter;
         |}
         |cudnnTensorDescriptor_t hx_desc_$counter;
         |CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc_$counter));
         |int hx_dims_$counter[] = {${numLayers * numDirections}, batchSize_$counter, $hiddenSize};
         |int hx_strides_$counter[] = {hx_dims_$counter[1] * hx_dims_$counter[2], hx_dims_$counter[2], 1};
         |CUDNN_CALL(cudnnSetTensorNdDescriptor(
         |    hx_desc_$counter, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims_$counter, hx_strides_$counter));
         |
         |size_t paramsSize_$counter;
         |CUDNN_CALL(cudnnGetRNNParamsSize(
         |    cudnnHandle, rnn_desc_$counter, x_descs_$counter[0], &paramsSize_$counter, CUDNN_DATA_FLOAT));
         |#ifdef DEBUG
         |assert(paramsSize_$counter / sizeof(float) == """.stripMargin, w.scalarCount, s""" && "Expected parameter size mismatch");
         |#endif
         |
         |cudnnFilterDescriptor_t w_desc_$counter;
         |CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc_$counter));
         |int w_dims_$counter[] = {int(paramsSize_$counter / sizeof(float)), 1, 1};
         |CUDNN_CALL(cudnnSetFilterNdDescriptor(
         |    w_desc_$counter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims_$counter));
         |
         |cudnnTensorDescriptor_t y_descs_$counter[seqLength_$counter];
         |cudnnTensorDescriptor_t y_desc_$counter;
         |CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc_$counter));
         |int y_dims_$counter[] = {batchSize_$counter, ${hiddenSize * numDirections}, 1};
         |int y_strides_$counter[] = {y_dims_$counter[1] * y_dims_$counter[2], y_dims_$counter[2], 1};
         |CUDNN_CALL(cudnnSetTensorNdDescriptor(
         |    y_desc_$counter, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims_$counter, y_strides_$counter));
         |for (int i = 0; i < seqLength_$counter; i++) {
         |  y_descs_$counter[i] = y_desc_$counter;
         |}
         |
         |size_t workspaceSize_$counter;
         |CUDNN_CALL(cudnnGetRNNWorkspaceSize(
         |    cudnnHandle, rnn_desc_$counter, seqLength_$counter, x_descs_$counter, &workspaceSize_$counter));
         |void* workspace_$counter = myGpuMalloc(workspaceSize_$counter);
         """.stripMargin): _*
      )

      // If training, create reserve space and call `ForwardTraining` function.
      if (training) {
        val reserveSpace = var_new(unchecked[Array[Float]]("(float*)NULL"))
        val reserveSpaceSize = var_new(unchecked[Int]("0"))
        unchecked[Unit](
          Seq(s"""
            |{// Reserve space used by `ForwardTraining` function.
            |size_t reserveSize;
            |CUDNN_CALL(cudnnGetRNNTrainingReserveSize(
            |    cudnnHandle, rnn_desc_$counter, seqLength_$counter, x_descs_$counter, &reserveSize));
            |void* reserveSpace = myGpuMalloc(reserveSize);
            |""".stripMargin,
            reserveSpace, " = (float*)reserveSpace;\n",
            reserveSpaceSize, " = (int)reserveSize;\n") ++
          Seq(
            "CUDNN_CALL(cudnnRNNForwardTraining(\n" +
            s"    cudnnHandle, rnn_desc_$counter, seqLength_$counter, x_descs_$counter, ", x.data, ",\n" +
            s"    hx_desc_$counter,", hxData, s", hx_desc_$counter,", cxData, s", w_desc_$counter, ", w.data, s", y_descs_$counter, ", res.data, ",\n" +
            s"    hx_desc_$counter, ", hyData, s", hx_desc_$counter, NULL, workspace_$counter, workspaceSize_$counter, reserveSpace, reserveSize));\n") ++
          Seq(s"}"): _*)
        (res, hy, Some((reserveSpace, reserveSpaceSize)), counter)
      } else { // If inference, call `ForwardInference` function.
        unchecked[Unit](
          Seq(
             "CUDNN_CALL(cudnnRNNForwardInference(\n" +
            s"    cudnnHandle, rnn_desc_$counter, seqLength_$counter, x_descs_$counter, ", x.data, ",\n" +
            s"    hx_desc_$counter,", hxData, s", hx_desc_$counter,", cxData, s", w_desc_$counter, ", w.data, s", y_descs_$counter, ", res.data, ",\n" +
            s"    hx_desc_$counter,", hyData, s", hx_desc_$counter, NULL, workspace_$counter, workspaceSize_$counter));\n"
          ) ++
          Seq(s"myGpuFree(workspaceSize_$counter);\n"): _*)
        (res, hy, None, counter)
      }
    }

    def cudnnRNNForwardInference(mode: RnnMode,
                                 x: Tensor, hx: Option[Tensor] = None, cx: Option[Tensor] = None, w: Tensor,
                                 numLayers: Int, hiddenSize: Int,
                                 dropout: Float = 0f,
                                 bidirectional: Boolean = false): Tensor = {
      cudnnRNNForwardHelper(mode, training = false, x, hx, cx, w, numLayers, hiddenSize, dropout, bidirectional)._1
    }

    def cudnnRNNForwardTraining(mode: RnnMode,
                                x: Tensor, hx: Option[Tensor] = None, cx: Option[Tensor] = None, w: Tensor,
                                numLayers: Int, hiddenSize: Int,
                                dropout: Float = 0f,
                                bidirectional: Boolean = false): (Tensor, Option[Tensor], Rep[Array[Float]], Rep[Int], Int) = {
      val (output, hy, reserve, counter) =
        cudnnRNNForwardHelper(mode, training = true, x, hx, cx, w, numLayers, hiddenSize, dropout, bidirectional)
      reserve match {
        case None => throw new IllegalArgumentException("Expected RNN reserve space to be defined")
        case Some((reserveSpace, reserveSpaceSize)) => (output, hy, reserveSpace, reserveSpaceSize, counter)
      }
    }

    // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNBackwardData
    def cudnnRNNBackwardData(mode: RnnMode,
                             input: TensorR, hx: Option[Tensor], cx: Option[Tensor], w: TensorR, output: TensorR,
                             numLayers: Int, hiddenSize: Int,
                             dropout: Float = 0f,
                             bidirectional: Boolean = false,
                             reserve: Rep[Array[Float]],
                             reserveSize: Rep[Int],
                             counter: Int): Unit = {
      // TODO: Calculate hidden state gradients?
      assert(input.x.rank == 3, "RNN input should have rank 3: [seqLength x batchSize x inputSize]")
      assert(output.x.rank == 3, "RNN output should have rank 3: [seqLength x batchSize x hiddenSize * numDirections]")
      hx match {
        case None =>
        case Some(hx) =>
          assert(hx.rank == 3, "RNN hidden state should have rank 3: [numLayers * numDirections x batchSize x hiddenSize]")
          assert(input.x.shape(1) == hx.shape(1), "RNN hidden state second dimension should equal input second dimension (batch size)")
      }
      cx match {
        case None =>
        case Some(cx) =>
          assert(cx.rank == 3, "RNN hidden state should have rank 3: [numLayers * numDirections x batchSize x hiddenSize]")
          assert(input.x.shape(1) == cx.shape(1), "RNN hidden state second dimension should equal input second dimension (batch size)")
      }
      val hxData = hx.map(_.data).getOrElse(unchecked[Array[Float]]("(float*)NULL"))
      val cxData = cx.map(_.data).getOrElse(unchecked[Array[Float]]("(float*)NULL"))

      val seqLength = input.x.shape(0)
      val batchSize = input.x.shape(1)
      val inputSize = input.x.shape(2)
      val numDirections = if (bidirectional) 2 else 1

      unchecked[Unit](
         s"CUDNN_CALL(cudnnRNNBackwardData(\n" +
         s"    cudnnHandle, rnn_desc_$counter, seqLength_$counter, y_descs_$counter, ", output.x.data, s", y_descs_$counter, ", output.d.data, ",\n" +
         s"    hx_desc_$counter, NULL, hx_desc_$counter, NULL, w_desc_$counter, ", w.x.data, s", hx_desc_$counter, ", hxData, ",\n" +
         s"    hx_desc_$counter, ", cxData, s", x_descs_$counter, ", input.d.data, s", hx_desc_$counter, NULL, hx_desc_$counter, NULL,\n" +
         s"    workspace_$counter, workspaceSize_$counter, ", reserve, ", ", reserveSize, "));\n"
      )
    }

    // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNBackwardWeights
    def cudnnRNNBackwardWeights(mode: RnnMode,
                                input: TensorR, hx: Option[Tensor], w: TensorR, output: TensorR,
                                numLayers: Int, hiddenSize: Int,
                                dropout: Float = 0f,
                                bidirectional: Boolean = false,
                                reserve: Rep[Array[Float]],
                                reserveSize: Rep[Int],
                                counter: Int): Unit = {
      assert(input.x.rank == 3, "RNN input should have rank 3: [seqLength x batchSize x inputSize]")
      assert(output.x.rank == 3, "RNN output should have rank 3: [seqLength x batchSize x hiddenSize * numDirections]")
      hx match {
        case None =>
        case Some(hx) =>
          assert(hx.rank == 3, "RNN hidden state should have rank 3: [numLayers * numDirections x batchSize x hiddenSize]")
          assert(input.x.shape(1) == hx.shape(1), "RNN hidden state second dimension should equal input second dimension (batch size)")
      }
      val hxData = hx.map(_.data).getOrElse(unchecked[Array[Float]]("(float*)NULL"))

      val seqLength = input.x.shape(0)
      val batchSize = input.x.shape(1)
      val inputSize = input.x.shape(2)
      val numDirections = if (bidirectional) 2 else 1

      unchecked[Unit](
          s"CUDNN_CALL(cudnnRNNBackwardWeights(\n" +
          s"    cudnnHandle, rnn_desc_$counter, seqLength_$counter, x_descs_$counter, ", input.x.data, s", hx_desc_$counter, ", hxData, ",\n" +
          s"    y_descs_$counter, ", output.x.data, s", workspace_$counter, workspaceSize_$counter,\n" +
          s"    w_desc_$counter, ", w.d.data, ", ", reserve, ", ", reserveSize, "));\n"
      )
    }

    def cudnnRNNBackward(mode: RnnMode,
                         input: TensorR, hx: Option[Tensor], cx: Option[Tensor],
                         w: TensorR, output: TensorR,
                         numLayers: Int, hiddenSize: Int,
                         dropout: Float = 0f,
                         bidirectional: Boolean = false,
                         reserve: Rep[Array[Float]],
                         reserveSize: Rep[Int],
                         counter: Int): Unit = {
      if (!input.isInput)
        cudnnRNNBackwardData(mode, input, hx, cx, w, output, numLayers, hiddenSize, dropout, bidirectional, reserve, reserveSize, counter)
      cudnnRNNBackwardWeights(mode, input, hx, w, output, numLayers, hiddenSize, dropout, bidirectional, reserve, reserveSize, counter)
    }

    override def ctcLoss(prob: TensorR, inputLengths: Rep[Array[Int]], labels: Rep[Array[Int]], labelLengths: Rep[Array[Int]]): Tensor = {
      cudnnCTCLoss(prob, labels, inputLengths, labelLengths)
    }

    // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnCTCLoss
    def cudnnCTCLoss(probs: TensorR, labels: Rep[Array[Int]], inputLengths: Rep[Array[Int]], targetLengths: Rep[Array[Int]]): Tensor = {
      assert(probs.x.rank == 3, "Probability tensor should have rank 3: [inputLength, batchSize, alphabetSize]")
      val inputLength = probs.x.shape(0)
      val batchSize = probs.x.shape(1)
      val alphabetSize = probs.x.shape(2)
      // Note: `inputLengths` and `targetLengths` should have length equal to `batchSize`.
      // Note: `cudnnGetCTCLossWorkspaceSize` requires that the batchSize (i.e. size of targetLengths) is NO greater than 256.
      assertC(batchSize <= 256, "'cudnnGetCTCLossWorkspaceSize' requires batch size less than 256, got %d\\n", batchSize)

      val costs = Tensor(mallocArray[Float](batchSize), batchSize)
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t probs_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&probs_desc));
          |int probs_dims[] = {""".stripMargin, inputLength, ", ", batchSize, ", ", alphabetSize, s"""};
          |int probs_strides[] = {probs_dims[1] * probs_dims[2], probs_dims[2], 1};
          |CUDNN_CALL(cudnnSetTensorNdDescriptor(
          |    probs_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, probs_dims, probs_strides));
          |
          |cudnnTensorDescriptor_t grad_desc = probs_desc;
          |
          |cudnnCTCLossDescriptor_t ctc_desc;
          |CUDNN_CALL(cudnnCreateCTCLossDescriptor(&ctc_desc));
          |CUDNN_CALL(cudnnSetCTCLossDescriptor(ctc_desc, CUDNN_DATA_FLOAT));
          |""".stripMargin) ++
        Seq(
          "size_t wsSize;\n" +
          "CUDNN_CALL(cudnnGetCTCLossWorkspaceSize(\n" +
          "    cudnnHandle, probs_desc, grad_desc, ", labels, ", ", targetLengths, ", ", inputLengths, ",\n" +
          "    CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, &wsSize));\n" +
          "void *ws = myGpuMalloc(wsSize);\n\n" +
          "CUDNN_CALL(cudnnCTCLoss(\n" +
          "    cudnnHandle, probs_desc, ", probs.x.data, ", ", labels, ", ", targetLengths, ", ", inputLengths, ",\n" +
          "    ", costs.data, ", grad_desc, ", probs.d.data, ", CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, ws, wsSize));\n" +
          "}"): _*)
      // reduce costs to scalar value
      cudnnReduceTensor(costs, ReductionOp.Avg, Seq(0), false)
    }

    def seqDataDescriptorHelper(name: String, batchSize: Rep[Int], beamSize: Rep[Int], seqLen: Rep[Int], embSize: Rep[Int], seqLenArray: Rep[Array[Int]], first: Boolean = false) = 
        Seq(if (first) "int dimA[CUDNN_SEQDATA_DIM_COUNT];\n" else "",
          s"""|dimA[CUDNN_SEQDATA_BEAM_DIM] = ${Unwrap(beamSize)};
          |dimA[CUDNN_SEQDATA_BATCH_DIM] = ${Unwrap(batchSize)};
          |dimA[CUDNN_SEQDATA_TIME_DIM] = ${Unwrap(seqLen)};
          |dimA[CUDNN_SEQDATA_VECT_DIM] = ${Unwrap(embSize)};
          |cudnnSeqDataDescriptor_t ${name};
          |CUDNN_CALL(cudnnCreateSeqDataDescriptor(&${name}));
          |CUDNN_CALL(cudnnSetSeqDataDescriptor(${name}, CUDNN_DATA_FLOAT, CUDNN_SEQDATA_DIM_COUNT, dimA, dataAxes,""".stripMargin, batchSize * beamSize ,",", seqLenArray, ", NULL))\n"
          )

    def attnDescriptorHelper(name: String, dropoutDescName: String = "NULL", numHeads: Int, qSize: Rep[Int], kSize: Rep[Int], vSize: Rep[Int], embedDim: Int, bias: Boolean = false, 
    smScaler: Float = 1.0f, seqLenQ: Rep[Int], seqLenK: Rep[Int], batchSize: Rep[Int], beamSize: Rep[Int]) = 
        // seqLenQ and seqLenK is the maximum sentence length in Q and K/V
        Seq(s"""
        |cudnnAttnDescriptor_t ${name};
        |CUDNN_CALL(cudnnCreateAttnDescriptor(&attn_desc));
        |CUDNN_CALL(cudnnSetAttnDescriptor(${name}, CUDNN_ATTN_QUERYMAP_ALL_TO_ONE ${if (bias) "| CUDNN_ATTN_ENABLE_PROJ_BIASES" else ""}, ${numHeads}, ${smScaler}, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH,
        | ${dropoutDescName}, NULL,""".stripMargin, qSize , "," , kSize, ",", vSize , "," , embedDim / numHeads, "," , embedDim / numHeads, "," , embedDim / numHeads, ", 0, ", seqLenQ, "," , seqLenK, ",", batchSize, "," , beamSize, "));\n")

    def cudnnMultiheadAttnForward(query: TensorR, key: TensorR, value: TensorR, weights: TensorR, numHeads: Int, embedDim:Int, qSeqArray: Rep[Array[Int]], 
    kSeqArray: Rep[Array[Int]], loWinIdx: Rep[Array[Int]], hiWinIdx: Rep[Array[Int]], bias: Boolean, dropoutRate: Float = 0.0f, smScaler: Float = 1.0f): 
    (Tensor, Rep[Array[Float]], Rep[Int], Rep[Array[Float]], Rep[Int], Rep[Int], Rep[Array[Int]], Rep[Array[Int]]) = {
      // Assumes tensors in [T(time) N(batch) B(beamsize) V(vector-embed)]
      // qSeqArray, kSeqArray (input)
      // loWinIdx, hiWinIdx - attention window (input)
      // dim definition
      // q, v, k, o desc
      // dimA stuff
      // order of stuff
      // define dataAxes

      val output = Tensor.zeros(query.x.shape(0), query.x.shape(1), query.x.shape(2), embedDim).toGPU()
      val devWkSpace = var_new(unchecked[Array[Float]]("(float*)NULL"))
      val sizeWkspace = var_new(unchecked[Int]("0"))
      val devReserve = var_new(unchecked[Array[Float]]("(float*)NULL"))
      val sizeReserve = var_new(unchecked[Int]("0"))
      val sizeWeights = var_new(unchecked[Int]("0"))
      val devQSeqArray = qSeqArray.toGPU(query.x.shape(1) * query.x.shape(2))
      val devKSeqArray = kSeqArray.toGPU(query.x.shape(1) * query.x.shape(2))

      // create dropout desc
      // create attn_desc
      // get attn buffer sizes
      // forward call
      unchecked[Unit](
      Seq(s"""{
      |cudnnSeqDataAxis_t dataAxes[CUDNN_SEQDATA_DIM_COUNT];
      |dataAxes[0] = CUDNN_SEQDATA_TIME_DIM;
      |dataAxes[1] = CUDNN_SEQDATA_BATCH_DIM;
      |dataAxes[2] = CUDNN_SEQDATA_BEAM_DIM;
      |dataAxes[3] = CUDNN_SEQDATA_VECT_DIM;
      |""".stripMargin)
      ++
      Seq(s"""cudnnDropoutDescriptor_t drop_desc;
      |CUDNN_CALL(cudnnCreateDropoutDescriptor(&drop_desc));
      |size_t dropoutBufSize;
      |void *dropoutBuf;
      |CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutBufSize));
      |CUDA_CALL(cudaMalloc((void **)&dropoutBuf, dropoutBufSize));
      |CUDNN_CALL(cudnnSetDropoutDescriptor(drop_desc, cudnnHandle, ${dropoutRate}, dropoutBuf, dropoutBufSize, 0));
      |""".stripMargin)
      ++
      attnDescriptorHelper("attn_desc", "drop_desc", numHeads, query.x.shape(3), key.x.shape(3), value.x.shape(3), embedDim, bias, smScaler, query.x.shape(0), key.x.shape(0), query.x.shape(1), query.x.shape(2))
      ++
      Seq(s"""
      |size_t sizeWeights;
      |size_t sizeWkspace;
      |size_t sizeReserve;
      |CUDNN_CALL(cudnnGetMultiHeadAttnBuffers(cudnnHandle, attn_desc, &sizeWeights, &sizeWkspace, &sizeReserve));
      |void *devWkspace = myGpuMalloc(sizeWkspace);
      |void *devReserve = myGpuMalloc(sizeReserve);
      """.stripMargin,
      devReserve, " = (float *)devReserve;\n",
      sizeReserve, " = (int) sizeReserve;\n",
      devWkSpace, " = (float *)devWkspace;\n",
      sizeWkspace, " = (int) sizeWkspace;\n",
      sizeWeights, " = (int) sizeWeights;\n"
      )
      ++
      seqDataDescriptorHelper("q_desc", query.x.shape(1), query.x.shape(2), query.x.shape(0), query.x.shape(3), qSeqArray, first=true)
      ++
      seqDataDescriptorHelper("o_desc", query.x.shape(1), query.x.shape(2), query.x.shape(0), embedDim, qSeqArray, first=false)
      ++
      seqDataDescriptorHelper("k_desc", key.x.shape(1), key.x.shape(2), key.x.shape(0), key.x.shape(3), kSeqArray, first=false)
      ++
      seqDataDescriptorHelper("v_desc", value.x.shape(1), value.x.shape(2), value.x.shape(0), value.x.shape(3), kSeqArray, first=false)
      ++
      Seq("CUDNN_CALL(cudnnMultiHeadAttnForward(cudnnHandle, attn_desc, -1,", loWinIdx, "," , hiWinIdx, ",", devQSeqArray, "," , devKSeqArray, ", ",  
      "q_desc, ", query.x.data, ", NULL, k_desc, ", key.x.data , ",v_desc, ", value.x.data, ", o_desc,", output.data, ", sizeWeights,", weights.x.data ,"," 
      , sizeWkspace, ",", devWkSpace, ",", sizeReserve,",", devReserve, "));\n}") : _*
      )
      (output, devWkSpace, sizeWkspace, devReserve, sizeReserve, sizeWeights, devQSeqArray, devKSeqArray)
    }

    def cudnnMultiHeadAttnBackward(output: TensorR, query: TensorR, key: TensorR, value: TensorR, weights: TensorR, numHeads: Int, embedDim:Int, 
      qSeqArray: Rep[Array[Int]], kSeqArray: Rep[Array[Int]], devQSeqArray: Rep[Array[Int]], devKSeqArray: Rep[Array[Int]], loWinIdx: Rep[Array[Int]], 
      hiWinIdx: Rep[Array[Int]], bias: Boolean, dropoutRate :Float = 0.0f,smScaler: Float = 1.0f, devWkSpace: Rep[Array[Float]], sizeWkSpace: Rep[Int],
       devReserve: Rep[Array[Float]], sizeReserve: Rep[Int], sizeWeights: Rep[Int]) = {
        unchecked[Unit](
      Seq(s"""{
      |cudnnSeqDataAxis_t dataAxes[CUDNN_SEQDATA_DIM_COUNT];
      |dataAxes[0] = CUDNN_SEQDATA_TIME_DIM;
      |dataAxes[1] = CUDNN_SEQDATA_BATCH_DIM;
      |dataAxes[2] = CUDNN_SEQDATA_BEAM_DIM;
      |dataAxes[3] = CUDNN_SEQDATA_VECT_DIM;
      |""".stripMargin)
      ++
      Seq(s"""cudnnDropoutDescriptor_t drop_desc;
      |CUDNN_CALL(cudnnCreateDropoutDescriptor(&drop_desc));
      |size_t dropoutBufSize;
      |void *dropoutBuf;
      |CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutBufSize));
      |CUDA_CALL(cudaMalloc((void **)&dropoutBuf, dropoutBufSize));
      |CUDNN_CALL(cudnnSetDropoutDescriptor(drop_desc, cudnnHandle, ${dropoutRate}, dropoutBuf, dropoutBufSize, 0));
      |""".stripMargin)
      ++
      attnDescriptorHelper("attn_desc", "drop_desc", numHeads, query.x.shape(3), key.x.shape(3), value.x.shape(3), embedDim, bias, smScaler, query.x.shape(0), key.x.shape(0), query.x.shape(1), query.x.shape(2))
      ++
      seqDataDescriptorHelper("q_desc", query.x.shape(1), query.x.shape(2), query.x.shape(0), query.x.shape(3), qSeqArray, first=true)
      ++
      seqDataDescriptorHelper("o_desc", query.x.shape(1), query.x.shape(2), query.x.shape(0), embedDim, qSeqArray, first=false)
      ++
      seqDataDescriptorHelper("k_desc", key.x.shape(1), key.x.shape(2), key.x.shape(0), key.x.shape(3), kSeqArray, first=false)
      ++
      seqDataDescriptorHelper("v_desc", value.x.shape(1), value.x.shape(2), value.x.shape(0), value.x.shape(3), kSeqArray, first=false)
      ++
      Seq("CUDNN_CALL(cudnnMultiHeadAttnBackwardData(cudnnHandle, attn_desc,", loWinIdx, "," , hiWinIdx, ",", devQSeqArray, "," , devKSeqArray, ", ",
      "o_desc, ", output.d.data, ",q_desc, ", query.d.data, "," , query.x.data , ",k_desc,", key.d.data, ",", key.x.data, ",v_desc,", value.d.data, ", ",
      value.x.data, ",", sizeWeights, ",", weights.d.data, ",", sizeWkSpace, ",", devWkSpace, ",", sizeReserve, ",", devReserve, "));\n")
      ++
      Seq("CUDNN_CALL(cudnnMultiHeadAttnBackwardWeights(cudnnHandle, attn_desc, CUDNN_WGRAD_MODE_ADD, q_desc,", query.x.data, ",k_desc, ", key.x.data, ",v_desc,",
      value.x.data, ",o_desc,", output.d.data, ",", sizeWeights, ",", weights.x.data, ",", weights.d.data, ",", sizeWkSpace, ",", devWkSpace, ",", sizeReserve, ",", devReserve, 
      "));\n}")
       : _*
      )
    }

  }

  object BackendCudnn {
    def apply() = new BackendCudnn
  }

  // Define default GPU backend.
  override def BackendGPU: Backend = BackendCudnn()
  backend = BackendGPU
}

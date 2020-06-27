package lantern.thirdparty

import lms.core._
import lms.util._
import lms.core.stub._
import lms.core.Backend._
import lms.core.virtualize
import lms.core.utils.time
import lms.macros.{SourceContext, RefinedManifest}
import lms.thirdparty.{CLibs}

import lantern.collection.mutable.{StackArrayOps}

trait CuDNNOps extends CuBLASOps with CLibs with StackArrayOps { b: Base  =>
  /* LMS support for CuDNN library */

  // macros for data layout
  abstract class TensorFormat
  def knchw = cmacro[TensorFormat]("CUDNN_TENSOR_NCHW")

  // macros for data type
  abstract class CuDNNDataType
  def kdouble = cmacro[CuDNNDataType]("CUDNN_DATA_DOUBLE")
  def kfloat = cmacro[CuDNNDataType]("CUDNN_DATA_FLOAT")
  def khalf = cmacro[CuDNNDataType]("CUDNN_DATA_HALF")
  def kint8 = cmacro[CuDNNDataType]("CUDNN_DATA_INT8")
  def kuint8 = cmacro[CuDNNDataType]("CUDNN_DATA_UINT8")
  def kint32 = cmacro[CuDNNDataType]("CUDNN_DATA_INT32")
  def kint8x4 = cmacro[CuDNNDataType]("CUDNN_DATA_INT8x4")
  def kint8x32 = cmacro[CuDNNDataType]("CUDNN_DATA_INT8x32")
  def kuint8x4 = cmacro[CuDNNDataType]("CUDNN_DATA_UINT8x4")

  // macros for activation
  abstract class ActivationType
  def ksigmoid = cmacro[ActivationType]("CUDNN_ACTIVATION_SIGMOID")
  def krelu = cmacro[ActivationType]("CUDNN_ACTIVATION_RELU")
  def ktanh = cmacro[ActivationType]("CUDNN_ACTIVATION_TANH")
  def kclipped_relu = cmacro[ActivationType]("CUDNN_ACTIVATION_CLIPPED_RELU")
  def kelu = cmacro[ActivationType]("CUDNN_ACTIVATION_ELU")

  // macros for pool mode
  abstract class PoolModes
  def kmax = cmacro[PoolModes]("CUDNN_POOLING_MAX")
  def kaverage_ip = cmacro[PoolModes]("CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING")
  def kaverage_ep = cmacro[PoolModes]("CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING")
  def kmax_d = cmacro[PoolModes]("CUDNN_POOLING_MAX_DETERMINISTIC")

  // macros for nan opt
  abstract class NanOpt
  def knot_prop = cmacro[NanOpt]("CUDNN_NOT_PROPAGATE_NAN")
  def kprop = cmacro[NanOpt]("CUDNN_PROPAGATE_NAN")

  // macros for softmax mode
  abstract class SoftmaxMode
  def kinstance = cmacro[SoftmaxMode]("CUDNN_SOFTMAX_MODE_INSTANCE")
  def kchannel = cmacro[SoftmaxMode]("CUDNN_SOFTMAX_MODE_CHANNEL")

  // macros for reduction ops
  abstract class ReduceTensorOp
  def kradd = cmacro[ReduceTensorOp]("CUDNN_REDUCE_TENSOR_ADD")
  def krmul = cmacro[ReduceTensorOp]("CUDNN_REDUCE_TENSOR_MUL")
  def krmin = cmacro[ReduceTensorOp]("CUDNN_REDUCE_TENSOR_MIN")
  def krmax = cmacro[ReduceTensorOp]("CUDNN_REDUCE_TENSOR_MAX")
  def kramax = cmacro[ReduceTensorOp]("CUDNN_REDUCE_TENSOR_AMAX")
  def kravg = cmacro[ReduceTensorOp]("CUDNN_REDUCE_TENSOR_AVG")
  def krnorm1 = cmacro[ReduceTensorOp]("CUDNN_REDUCE_TENSOR_NORM1")
  def krnorm2 = cmacro[ReduceTensorOp]("CUDNN_REDUCE_TENSOR_NORM2")
  def krmul_no_zeros = cmacro[ReduceTensorOp]("CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS")

  // macros for math type
  abstract class MathType
  def kdefault = cmacro[MathType]("CUDNN_DEFAULT_MATH")
  def ktensor_op = cmacro[MathType]("CUDNN_TENSOR_OP_MATH")
  def ktensor_allow_conversion = cmacro[MathType]("CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION")
  def kfma = cmacro[MathType]("CUDNN_FMA_MATH")

  // macros for conv types
  abstract class CudnnConvolutionMode
  def kconvolution = cmacro[CudnnConvolutionMode]("CUDNN_CONVOLUTION")
  def kcross_correlation = cmacro[CudnnConvolutionMode]("CUDNN_CROSS_CORRELATION")

  // cudnnStatus_t and CUDNN_CALL
  abstract class CudnnStatusT
  def cudnn_call(status: Rep[CudnnStatusT]): Rep[Unit] = {
    Wrap[Unit](Adapter.g.reflectWrite("cudnn-call", Unwrap(status))(Adapter.CTRL)) // ??? need read key?
  }

  // cudnnHandle_t struct
  abstract class CudnnHandleT
  def getCudnnHandleT = newStruct[CudnnHandleT]
  def cudnnCreate(handle: Rep[CudnnHandleT]): Rep[CudnnStatusT] = {
    Wrap[CudnnStatusT](Adapter.g.reflectWrite("cudnnCreate-f", Unwrap(handle))(Unwrap(handle)))
  }
  def cudnnDestroy(handle: Rep[CudnnHandleT]): Rep[CudnnStatusT] = {
    Wrap[CudnnStatusT](Adapter.g.reflectWrite("cudnnDestroy-f", Unwrap(handle))(Unwrap(handle)))
  }
  // macros for working with hard coded cudnnHandle
  def khandle = cmacro[CudnnHandleT]("cudnnHandle")


  // cudnnTensorDescriptor_t struct
  abstract class CudnnTensorDescriptorT
  def getCudnnTensorDescriptorT = newStruct[CudnnTensorDescriptorT]
  def cudnnCreateTensorDescriptor(desc: Rep[CudnnTensorDescriptorT]): Rep[CudnnStatusT] = {
    Wrap[CudnnStatusT](Adapter.g.reflectWrite("cudnnCreateTensorDescriptor-f", Unwrap(desc))(Unwrap(desc)))
  }
  def cudnnSetTensor4dDescriptor(desc: Rep[CudnnTensorDescriptorT], layout: Rep[TensorFormat], dtype: Rep[CuDNNDataType], n: Rep[Int],
    h: Rep[Int], c: Rep[Int], w: Rep[Int]): Rep[CudnnStatusT] = {
    Wrap[CudnnStatusT](Adapter.g.reflectEffect("cudnnSetTensor4dDescriptor-f", Unwrap(desc), Unwrap(layout), Unwrap(dtype),
      Unwrap(n), Unwrap(h), Unwrap(c), Unwrap(w))(Unwrap(desc))(Unwrap(desc)))
  }
  def cudnnSetTensorNdDescriptor(desc: Rep[CudnnTensorDescriptorT], dtype: Rep[CuDNNDataType],  nbDims: Rep[Int],
    dimA: Rep[Array[Int]], strideA: Rep[Array[Int]]): Rep[CudnnStatusT] = {
    Wrap[CudnnStatusT](Adapter.g.reflectEffect("cudnnSetTensorNdDescriptor-f",
      Unwrap(desc), Unwrap(dtype), Unwrap(nbDims), Unwrap(dimA), Unwrap(strideA))(Unwrap(desc))(Unwrap(desc)))
  }

  def cudnnGetTensor4dDescriptor(layout: Rep[TensorFormat], dtype: Rep[CuDNNDataType], shape: Seq[Rep[Int]]) = {
    val desc = getCudnnTensorDescriptorT
    cudnn_call(cudnnCreateTensorDescriptor(desc))
    cudnn_call(cudnnSetTensor4dDescriptor(desc, layout, dtype, shape(0), shape(1), shape(2), shape(3)))
    desc
  }

  // cudnnFilterDescriptor_t struct
  abstract class CudnnFilterDescriptorT
  def getCudnnFilterDescriptorT = newStruct[CudnnFilterDescriptorT]
  def cudnnCreateFilterDescriptor(desc: Rep[CudnnFilterDescriptorT]): Rep[CudnnStatusT] = {
    Wrap[CudnnStatusT](Adapter.g.reflectWrite("cudnnCreateFilterDescriptor-f", Unwrap(desc))(Unwrap(desc)))
  }
  def cudnnSetFilter4dDescriptor(desc: Rep[CudnnFilterDescriptorT], dtype: Rep[CuDNNDataType], layout: Rep[TensorFormat], n: Rep[Int],
    h: Rep[Int], c: Rep[Int], w: Rep[Int]): Rep[CudnnStatusT] = {
    Wrap[CudnnStatusT](Adapter.g.reflectEffect("cudnnSetFilter4dDescriptor-f", Unwrap(desc), Unwrap(dtype), Unwrap(layout),
      Unwrap(n), Unwrap(h), Unwrap(c), Unwrap(w))(Unwrap(desc))(Unwrap(desc)))
  }

  def cudnnGetFilter4dDescriptor(layout: Rep[TensorFormat], dtype: Rep[CuDNNDataType], shape: Seq[Rep[Int]]) = {
    val desc = getCudnnFilterDescriptorT
    cudnn_call(cudnnCreateFilterDescriptor(desc))
    cudnn_call(cudnnSetFilter4dDescriptor(desc, dtype, layout, shape(0), shape(1), shape(2), shape(3)))
    desc
  }

  // cudnnConvolutionDescriptor_t struct
  abstract class CudnnConvolutionDescriptorT
  def getCudnnConvolutionDescriptorT = newStruct[CudnnConvolutionDescriptorT]
  def cudnnCreateConvolutionDescriptor(desc: Rep[CudnnConvolutionDescriptorT]): Rep[CudnnStatusT] = {
    Wrap[CudnnStatusT](Adapter.g.reflectWrite("cudnnCreateConvolutionDescriptor-f", Unwrap(desc))(Unwrap(desc)))
  }
  def cudnnSetConvolution2dDescriptor(desc: Rep[CudnnConvolutionDescriptorT], padding1: Rep[Int], padding2: Rep[Int],
    strides1: Rep[Int], strides2: Rep[Int], dilation1: Rep[Int], dilation2: Rep[Int], conv_mode: Rep[CudnnConvolutionMode],
    dtype: Rep[CuDNNDataType]) = {
      Wrap[CudnnStatusT](Adapter.g.reflectEffect("cudnnSetConvolution2dDescriptor-f", Unwrap(desc), Unwrap(padding1),
        Unwrap(padding2), Unwrap(strides1), Unwrap(strides2), Unwrap(dilation1), Unwrap(dilation2), Unwrap(conv_mode),
        Unwrap(dtype))(Unwrap(desc))(Unwrap(desc)))
  }
  def cudnnSetConvolutionMathType(desc: Rep[CudnnConvolutionDescriptorT], mathType: Rep[MathType]) = {
    Wrap[CudnnStatusT](Adapter.g.reflectWrite("cudnnSetConvolutionMathType-f", Unwrap(desc), Unwrap(mathType))(Unwrap(desc)))
  }

  def cudnnGetConvolution2dDescriptor(paddings: (Int, Int), strides: (Int, Int), dilations: (Int, Int),
    convMode: Rep[CudnnConvolutionMode], dtype: Rep[CuDNNDataType], mathType: Option[Rep[MathType]]) = {
      val desc = getCudnnConvolutionDescriptorT
      cudnn_call(cudnnCreateConvolutionDescriptor(desc))
      cudnn_call(cudnnSetConvolution2dDescriptor(desc, paddings._1, paddings._2, strides._1, strides._2, dilations._1, dilations._2,
        convMode, dtype))
      mathType match {
        case Some(mt: Rep[MathType]) => cudnn_call(cudnnSetConvolutionMathType(desc, mt))
        case _ => ()
      }
      desc
  }

  // macro for cudnnConvolutionFwdAlgo_t
  abstract class CudnnConvolutionFwdAlgoT
  def kconvFwdAlgoImplicitGemm = cmacro[CudnnConvolutionFwdAlgoT]("CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM")
  def kconvFwdAlgoGemm = cmacro[CudnnConvolutionFwdAlgoT]("CUDNN_CONVOLUTION_FWD_ALGO_GEMM")
  def kconvFwdAlgoDirect = cmacro[CudnnConvolutionFwdAlgoT]("CUDNN_CONVOLUTION_FWD_ALGO_DIRECT")
  def kconvFwdAlgoFFT = cmacro[CudnnConvolutionFwdAlgoT]("CUDNN_CONVOLUTION_FWD_ALGO_FFT")
  def kconvFwdAlgoFFTTiling = cmacro[CudnnConvolutionFwdAlgoT]("CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING")
  def kconvFwdAlgoWinograd = cmacro[CudnnConvolutionFwdAlgoT]("CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD")
  def kconvFwdAlgoWinogradNonfused = cmacro[CudnnConvolutionFwdAlgoT]("CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED")

  // macro for cudnnConvolutionBwdDataAlgo_t
  abstract class CudnnConvolutionBwdDataAlgoT

  // cudnnConvolutionBwdDataAlgoPerf_t struct
  abstract class CudnnConvolutionBwdDataAlgoPerfT
  implicit class CudnnConvolutionBwdDataAlgoPerfTOps(x: Rep[CudnnConvolutionBwdDataAlgoPerfT]) {
    val algo = readField[CudnnConvolutionBwdDataAlgoPerfT, CudnnConvolutionBwdDataAlgoT](x, "algo")
  }
  def getCudnnConvolutionBwdDataAlgoPerfT = newStruct[CudnnConvolutionBwdDataAlgoPerfT]
  def cudnnGetConvolutionBackwardDataAlgorithm_v7(handle: Rep[CudnnHandleT], wDesc: Rep[CudnnFilterDescriptorT],
    yDesc: Rep[CudnnTensorDescriptorT], convDesc: Rep[CudnnConvolutionDescriptorT], xDesc: Rep[CudnnTensorDescriptorT],
    requestedAlgoCount: Rep[Int], returnedAlgoCountBwd: Var[Int], perfResultsBwd: Rep[Array[CudnnConvolutionBwdDataAlgoPerfT]]) = {
      libFunction[CudnnStatusT]("cudnnGetConvolutionBackwardDataAlgorithm_v7", Unwrap(handle), Unwrap(wDesc), Unwrap(yDesc),
        Unwrap(convDesc), Unwrap(xDesc), Unwrap(requestedAlgoCount), UnwrapV(returnedAlgoCountBwd), Unwrap(perfResultsBwd))(Seq[Int](),
        Seq(6, 7), Set(6))
  }

  // cudnnConvolutionFwdAlgoPerf_t struct
  abstract class CudnnConvolutionFwdAlgoPerfT
  implicit class CudnnConvolutionFwdAlgoPerfTOps(x: Rep[CudnnConvolutionFwdAlgoPerfT]) {
    val algo = readField[CudnnConvolutionFwdAlgoPerfT, CudnnConvolutionFwdAlgoT](x, "algo")
  }
  def getCudnnConvolutionFwdAlgoPerfT = newStruct[CudnnConvolutionFwdAlgoPerfT]
  def cudnnGetConvolutionForwardAlgorithm_v7(handle: Rep[CudnnHandleT], xDesc: Rep[CudnnTensorDescriptorT], wDesc: Rep[CudnnFilterDescriptorT],
    convDesc: Rep[CudnnConvolutionDescriptorT], yDesc: Rep[CudnnTensorDescriptorT], requestedAlgoCount: Rep[Int],
    returnedAlgoCount: Var[Int], perfResult: Rep[Array[CudnnConvolutionFwdAlgoPerfT]]) = {
      Wrap[CudnnStatusT](Adapter.g.reflectWrite("cudnnGetConvolutionForwardAlgorithm_v7-f", Unwrap(handle),
      Unwrap(xDesc), Unwrap(wDesc), Unwrap(convDesc), Unwrap(yDesc), Unwrap(requestedAlgoCount), UnwrapV(returnedAlgoCount), Unwrap(perfResult))
      (UnwrapV(returnedAlgoCount), Unwrap(perfResult)))
  }

  // conv work space
  def cudnnGetConvolutionForwardWorkspaceSize(handle: Rep[CudnnHandleT], xDesc: Rep[CudnnTensorDescriptorT], wDesc: Rep[CudnnFilterDescriptorT],
    convDesc: Rep[CudnnConvolutionDescriptorT], yDesc: Rep[CudnnTensorDescriptorT], algo: Rep[CudnnConvolutionFwdAlgoT],
    sizeInBytes: Var[SizeT]) = {
      Wrap[CudnnStatusT](Adapter.g.reflectWrite("cudnnGetConvolutionForwardWorkspaceSize-f", Unwrap(handle), Unwrap(xDesc),
      Unwrap(wDesc), Unwrap(convDesc), Unwrap(yDesc), Unwrap(algo), UnwrapV(sizeInBytes))(UnwrapV(sizeInBytes)))
  }
  // conv forward
  def cudnnConvolutionForward_a[T:Manifest](handle: Rep[CudnnHandleT], alpha: Var[T], xDesc: Rep[CudnnTensorDescriptorT], input: Rep[Array[T]],
    wDesc: Rep[CudnnFilterDescriptorT], filter: Rep[Array[T]], convDesc: Rep[CudnnConvolutionDescriptorT], algo: Rep[CudnnConvolutionFwdAlgoT],
    wsArray: Rep[Array[T]], wsSize: Rep[SizeT], beta: Var[T], yDesc: Rep[CudnnTensorDescriptorT], output: Rep[Array[T]]) = {
      libFunction[CudnnStatusT]("cudnnConvolutionForward", Unwrap(handle), UnwrapV(alpha), Unwrap(xDesc),
      Unwrap(input), Unwrap(wDesc), Unwrap(filter), Unwrap(convDesc), Unwrap(algo), Unwrap(wsArray), Unwrap(wsSize), UnwrapV(beta), Unwrap(yDesc),
      Unwrap(output))(Seq(1, 3, 5, 10, 12), Seq(9, 12), Set(1, 10))
      // Wrap[CudnnStatusT](Adapter.g.reflectEffect("cudnnConvolutionForward-f", Unwrap(handle), UnwrapV(alpha), Unwrap(xDesc),
      // Unwrap(input), Unwrap(wDesc), Unwrap(filter), Unwrap(convDesc), Unwrap(algo), Unwrap(wsArray), Unwrap(wsSize), UnwrapV(beta), Unwrap(yDesc),
      // Unwrap(output))(Unwrap(input), Unwrap(filter), Unwrap(output), UnwrapV(alpha), UnwrapV(beta))(Unwrap(wsArray), Unwrap(output)))
  }

  // cudnnAddTensor
  // def cudnnAddTensor[T:Manifest](handle: Rep[CudnnHandleT], alpha: Rep[Array[T]], aDesc: Rep[CudnnTensorDescriptorT],
  //   A: Rep[Array[T]], beta: Rep[Array[T]], cDesc: Rep[CudnnTensorDescriptorT], C: Rep[Array[T]]) = {
  //     Wrap[CudnnStatusT](Adapter.g.reflectEffect("cudnnAddTensor-f", Unwrap(handle), Unwrap(alpha), Unwrap(aDesc),
  //       Unwrap(A), Unwrap(beta), Unwrap(cDesc), Unwrap(C))(Unwrap(A), Unwrap(C), Unwrap(alpha), Unwrap(beta))(Unwrap(C)))
  // }

  def cudnnAddTensor[T:Manifest](handle: Rep[CudnnHandleT], alpha: Var[T], aDesc: Rep[CudnnTensorDescriptorT],
    A: Rep[Array[T]], beta: Var[T], cDesc: Rep[CudnnTensorDescriptorT], C: Rep[Array[T]]) = {
      libFunction[CudnnStatusT]("cudnnAddTensor", Unwrap(handle), UnwrapV(alpha), Unwrap(aDesc), Unwrap(A),
        UnwrapV(beta), Unwrap(cDesc), Unwrap(C))(Seq(1, 3, 4, 6), Seq(6), Set(1, 4))
    }
}


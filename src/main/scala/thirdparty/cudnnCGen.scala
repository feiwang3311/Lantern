package lantern.thirdparty

import lms.core._
import lms.util._
import lms.core.stub._
import lms.core.Backend._
import lms.core.virtualize
import lms.core.utils.time
import lms.macros.SourceContext

trait CCodeGenCuDNN extends ExtendedCCodeGen {

  val typeNameMapCuDNN = Map( // FIXME: tutorial specific
    "lantern.thirdparty.CuDNNOps$CudnnConvolutionFwdAlgoT" -> "cudnnConvolutionFwdAlgo_t",
    "lantern.thirdparty.CuDNNOps$CudnnConvolutionFwdAlgoPerfT" -> "cudnnConvolutionFwdAlgoPerf_t",
  )
  override def remap(m: Manifest[_]): String =
    typeNameMapCuDNN.getOrElse(m.runtimeClass.getName, super.remap(m))

  override def shallow(n: Node): Unit = n match {
    case Node(s, "cudnn-call", List(x: Sym), _) =>
      emit("CUDNN_CALL("); shallow(x); emit(")")
    case Node(s, "cudnnCreate-f", List(x: Sym), _) =>
      emit("cudnnCreate(&"); shallow(x); emit(")")
    case Node(s, "cudnnDestroy-f", List(x: Sym), _) =>
      emit("cudnnDestroy("); shallow(x); emit(")")
    case Node(s, "cudnnCreateTensorDescriptor-f", List(x: Sym), _) =>
      emit("cudnnCreateTensorDescriptor(&"); shallow(x); emit(")")
    case Node(s, "cudnnSetTensor4dDescriptor-f", args@List(a, b, c, d, e, f, g), _) =>
      emit("cudnnSetTensor4dDescriptor("); shallow(args.head);
      args.tail.foreach(a => {emit(", "); shallow(a)}); emit(")")
    case Node(s, "cudnnSetTensorNdDescriptor-f", args@List(a, b, c, d, e), _) =>
      emit("cudnnSetTensorNdDescriptor("); shallow(args.head);
      args.tail.foreach(a => {emit(", "); shallow(a)}); emit(")")
    case Node(s, "cudnnCreateFilterDescriptor-f", List(x: Sym), _) =>
      emit("cudnnCreateFilterDescriptor(&"); shallow(x); emit(")")
    case Node(s, "cudnnSetFilter4dDescriptor-f", args@List(a, b, c, d, e, f, g), _) =>
      emit("cudnnSetFilter4dDescriptor("); shallow(args.head)
      args.tail.foreach(a => {emit(", "); shallow(a)}); emit(")")
    case Node(s, "cudnnCreateConvolutionDescriptor-f", List(x: Sym), _) =>
      emit("cudnnCreateConvolutionDescriptor(&"); shallow(x); emit(")")
    case Node(s, "cudnnSetConvolution2dDescriptor-f", args, _) =>
      emit("cudnnSetConvolution2dDescriptor("); shallow(args.head)
      args.tail.foreach(a => {emit(", "); shallow(a)}); emit(")")
    case Node(s, "cudnnSetConvolutionMathType-f", List(a, b), _) =>
      emit("cudnnSetConvolutionMathType("); shallow(a); emit(", "); shallow(b); emit(")")
    case Node(s, "cudnnGetConvolutionForwardAlgorithm_v7-f", List(a, b, c, d, e, f, g, h), _) =>
      emit("cudnnGetConvolutionForwardAlgorithm_v7(");
      shallow(a); emit(", "); shallow(b); emit(", ")
      shallow(c); emit(", "); shallow(d); emit(", ")
      shallow(e); emit(", "); shallow(f); emit(", &")
      shallow(g); emit(", "); shallow(h); emit(")")
    case Node(s, "cudnnGetConvolutionForwardWorkspaceSize-f", List(a, b, c, d, e, f, g), _) =>
      emit("cudnnGetConvolutionForwardWorkspaceSize(");
      shallow(a); emit(", "); shallow(b); emit(", ")
      shallow(c); emit(", "); shallow(d); emit(", ")
      shallow(e); emit(", "); shallow(f); emit(", &")
      shallow(g); emit(")")
    case Node(s, "cudnnConvolutionForward-f", args, _) =>
      emit("cudnnConvolutionForward("); shallow(args.head)
      args.tail.foreach(a => {emit(", "); shallow(a)}); emit(")")
    case Node(s, "cudnnAddTensor-f", args, _) =>
      emit("cudnnAddTensor("); shallow(args.head);
      args.tail.foreach(a => {emit(", "); shallow(a)}); emit(")")


    case Node(s, "cudnn-finalize", _, _) =>
      emit("CUDNN_FINALIZE()")
    case _ => super.shallow(n)
  }

}

package lantern

import scala.util.continuations._

import lms.core.stub._
import lms.macros.SourceContext
import lms.core.virtualize
import lms.core.Backend
import lms.core.Backend._
import lms.core.Graph

import java.io.File
import java.io.PrintWriter

import lms.thirdparty._
import lantern.thirdparty._
import lantern.collection.mutable._

trait LanternGenC extends DslGenCPP with CCodeGenLibs {

  class Unknown
  def isInt(d: Backend.Def): Boolean = d match {
    case Backend.Const(x: Int) => true
    case s : Backend.Sym => typeMap.get(s).fold(false)(_ == manifest[Int])
    case _ => false
  }
  override def shallow(n: Node): Unit = n match {
    case n @ Node(s, "NewArray", List(x), _) if isInt(x) =>
      val ctype = remap(typeMap.get(s).map(_.typeArguments.head).getOrElse(manifest[Unknown]))
      emit(s"($ctype*)myMalloc("); shallow(x); emit(s" * sizeof($ctype))")
    case _ => super.shallow(n)
  }

  def templateRawCode: String = ""

  override def emitAll(ng: Graph, name: String)(m1:Manifest[_],m2:Manifest[_]): Unit = {
    registerHeader("<assert.h>", "<err.h>", "<errno.h>", "<fcntl.h>", "<functional>",
      "<math.h>", "<memory>", "<string.h>", "<sys/mman.h>", "<sys/stat.h>",
      "<sys/time.h>", "<time.h>", "<unistd.h>", "<algorithm>", "<numeric>")

    // -I /opt/OpenBLAS/include -L /opt/OpenBLAS/lib -lopenblas -lpthread
    registerHeader("/opt/OpenBLAS/include", "<cblas.h>")
    registerLibrary("-lopenblas")

    // add cpu_header.h
    val curPath = System.getProperty("user.dir") // /u/ml00_s/wang603/lms-umbrella/lantern
    val tailPath = "src/main/cpp/headers/"
    val headerFile = "\"cpu_header.h\""
    registerHeader(s"$curPath/$tailPath", headerFile)

    super.emitAll(ng, name)(m1, m2)
  }
}

trait LanternGenCublas extends LanternGenC with CCodeGenCuBLAS {

  override def remap(m: Manifest[_]) = m.runtimeClass.getName match {
    case s: String if s.endsWith("CublasHandleT") => "cublasHandle_t"
    case _ => super.remap(m)
  }

  override def shallow(n: Node): Unit = n match {
    case n @ Node(s, "NewGpuArray", List(x), _) =>
      val ctype = remap(typeMap.get(s).map(_.typeArguments.head).getOrElse(manifest[Unknown]))
      emit(s"($ctype*)myGpuMalloc("); shallow(x); emit(s" * sizeof($ctype))")
      // s"($ctype*)myGpuMalloc(${shallow(x)} * sizeof($ctype))"
    case n @ Node(s, op, List(x,y,size),_) if op.startsWith("h2dCopy[") =>
      // TODO: this is a bad fix for shallow(x), which doesn't work if x is static float array like {1.0, 2.0}
      // The issue is in c++ -std=c++11, we can use `(const float[]){1.0, 2.0}` to initalize a static float array namelessly
      // But we cannot do so in cuda code, which is compiled via nvcc!
      // I found no way to initialize a nameless float array in cuda, thus we cannot use `shallow(x)`!
      // We must to `traverse(x)' first, and use the result of it in the cudaMemcpy
      val ty = op.substring(8, op.length - 1).toLowerCase
      x match {
        case nx @ InlineSym(nnx @ Node(sx, "Array" , xs, _)) =>
          traverse(nnx)
          emit(s"CUDA_CALL(cudaMemcpy("); shallow(y); emit(s", ${quote(sx)}, "); shallow(size); emit(s" * sizeof($ty), cudaMemcpyHostToDevice))")
        case _ =>
          emit(s"CUDA_CALL(cudaMemcpy("); shallow(y); emit(", "); shallow(x); emit(", "); shallow(size); emit(s" * sizeof($ty), cudaMemcpyHostToDevice))")
      }
      // s"CUDA_CALL(cudaMemcpy(${shallow(y)}, ${shallow(x)}, ${shallow(size)}*sizeof(${ty}), cudaMemcpyHostToDevice))"
    case n @ Node(s, op, List(x,y,size),_) if op.startsWith("d2hCopy[") =>
      val ty = op.substring(8, op.length - 1).toLowerCase // op.drop(8).dropRight(1).toLowerCase
      emit(s"CUDA_CALL(cudaMemcpy("); shallow(y); emit(", "); shallow(x); emit(", "); shallow(size); emit(s" * sizeof($ty), cudaMemcpyDeviceToHost))")
      // s"CUDA_CALL(cudaMemcpy(${shallow(y)}, ${shallow(x)}, ${shallow(size)}*sizeof(${ty}), cudaMemcpyDeviceToHost))"
    case n @ Node(s, op, List(x,y,size),_) if op.startsWith("d2dCopy[") =>
      val ty = op.substring(8, op.length - 1).toLowerCase // op.drop(8).dropRight(1).toLowerCase
      emit(s"CUDA_CALL(cudaMemcpy("); shallow(y); emit(", "); shallow(x); emit(", "); shallow(size); emit(s" * sizeof($ty), cudaMemcpyDeviceToDevice))")
      // s"CUDA_CALL(cudaMemcpy(${shallow(y)}, ${shallow(x)}, ${shallow(size)}*sizeof(${ty}), cudaMemcpyDeviceToDevice))"
    case _ => super.shallow(n)
  }

  override def emitAll(ng: Graph, name: String)(m1: Manifest[_], m2: Manifest[_]): Unit = {
    registerHeader("<cuda.h>", "<cuda_runtime.h>", "<cublas_v2.h>")

    // -I /opt/OpenBLAS/include -L /opt/OpenBLAS/lib -lopenblas -lstdc++ -lcublas
    registerLibrary("-lstdc++", "-lcublas")

    // add cublas_header.h
    val curPath = System.getProperty("user.dir") // /u/ml00_s/wang603/lms-umbrella/lantern
    val tailPath = "src/main/cpp/headers/"
    val headerFile = "\"cublas_header.h\""
    registerHeader(s"$curPath/$tailPath", headerFile)

    super.emitAll(ng, name)(m1, m2)
  }
}

trait LanternGenCudnn extends LanternGenCublas with CCodeGenCuDNN with CCodeGenStackArray with CCodeGenLibs {

  override def remap(m: Manifest[_]) = m.runtimeClass.getName match {
    case s: String if s.endsWith("CudnnHandleT") => "cudnnHandle_t"
    case s: String if s.endsWith("CudnnTensorDescriptorT") => "cudnnTensorDescriptor_t"
    case s: String if s.endsWith("CudnnActivationDescriptorT") => "cudnnActivationDescriptor_t"
    case s: String if s.endsWith("CudnnReduceTensorDescriptorT") => "cudnnReduceTensorDescriptor_t"
    case s: String if s.endsWith("CudnnCTCLossDescriptorT") => "cudnnCTCLossDescriptor_t"
    case s: String if s.endsWith("CudnnFilterDescriptorT") => "cudnnFilterDescriptor_t"
    case s: String if s.endsWith("CudnnConvolutionDescriptorT") => "cudnnConvolutionDescriptor_t"
    case s: String if s.endsWith("CudnnConvolutionBwdDataAlgoPerfT") => "cudnnConvolutionBwdDataAlgoPerf_t"
    case s: String if s.endsWith("CudnnConvolutionBwdDataAlgoT") => "cudnnConvolutionBwdDataAlgo_t"
    case s: String if s.endsWith("CudnnConvolutionBwdFilterAlgoPerfT") => "cudnnConvolutionBwdFilterAlgoPerf_t"
    case s: String if s.endsWith("CudnnConvolutionBwdFilterAlgoT") => "cudnnConvolutionBwdFilterAlgo_t"
    case s: String if s.endsWith("CudnnPoolingDescriptorT") => "cudnnPoolingDescriptor_t"
    case _ => super.remap(m)
  }

  override def emitAll(ng: Graph, name: String)(m1: Manifest[_], m2: Manifest[_]): Unit = {
    registerHeader("<cudnn.h>")

    //--expt-extended-lambda -Wno-deprecated-gpu-targets -I /opt/OpenBLAS/include -L /opt/OpenBLAS/lib -lopenblas -lstdc++ -lcublas -lcudnn
    registerLibrary("--expt-extended-lambda", "-Wno-deprecated-gpu-targets", "-lcudnn")

    // add cublas_header.h
    val curPath = System.getProperty("user.dir") // /u/ml00_s/wang603/lms-umbrella/lantern
    val tailPath = "src/main/cpp/headers/"
    val headerFile = "\"cudnn_header.h\""
    registerHeader(s"$curPath/$tailPath", headerFile)

    super.emitAll(ng, name)(m1, m2)
  }
}

// TODO: bad design!! NNModule should not depend on backend!
abstract class LanternDriverBase[A: Manifest, B: Manifest] extends DslDriverCPP[A, B]
  with TensorDsl with NNModule with Dataset with ONNXLib with ScannerOpsExp with TimerOpsExp {

  // For saving the generated code somewhere
  val dir = "/tmp/"
  val fileName = s"lantern-snippet-${scala.util.Random.alphanumeric.take(4).mkString}"
  val filetype = ".cpp"
  def codeToFile(name: Option[String] = None) = {
    val outFileName = name match {
      case Some(s) => s
      case None => dir + fileName + filetype
    }
    System.out.println(s"code => $outFileName")
    val outFile = new PrintWriter(new File(outFileName))
    outFile.println(this.code)
    outFile.flush()
  }

}

abstract class LanternDriverC[A: Manifest, B: Manifest] extends LanternDriverBase[A, B] with TensorDslCPU { q =>
  override val codegen = new LanternGenC {
    val IR: q.type = q
  }
  backend = BackendCPU()
  compilerCommand = "g++ -std=c++11 -O3"
}

abstract class LanternDriverCublas[A: Manifest, B: Manifest] extends LanternDriverBase[A, B] with TensorDslCublas { q =>
  override val codegen = new LanternGenCublas {
    val IR: q.type = q
  }

  backend = BackendCublas()
  override val filetype = ".cu"

  compilerCommand = "nvcc -std=c++11 -O3"
  override val fileForRun = "/tmp/snippet.cu"

  override def wrapper(x: Rep[A]): Rep[B] = {
    generate_comment("Backend setup.")
    BackendCublas().setup()
    val result = snippet(x)

    generate_comment("Backend cleanup.")
    BackendCublas().cleanup()
    result
  }
}

abstract class LanternDriverCudnn[A: Manifest, B: Manifest] extends LanternDriverCublas[A, B] with NNModuleCudnn with TensorDslCudnn { q =>
  override val codegen = new LanternGenCudnn {
    val IR: q.type = q
  }

  override def wrapper(x: Rep[A]): Rep[B] = {
    generate_comment("Backend setup.")
    backend = BackendCudnn()
    backend.setup()
    val result = snippet(x)

    generate_comment("Backend cleanup.")
    BackendCudnn().cleanup()
    result
  }
}

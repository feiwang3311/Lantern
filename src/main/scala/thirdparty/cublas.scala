package lantern.thirdparty

import lms.core._
import lms.util._
import lms.core.stub._
import lms.core.Backend._
import lms.core.virtualize
import lms.core.utils.time
import lms.macros.{SourceContext, RefinedManifest}
import lms.thirdparty.{CLibs, CudaFunction}

import lantern.collection.mutable.{StackArrayOps}

trait CuBLASOps extends CBLASOps with CLibs with CudaFunction with StackArrayOps { b: Base  =>
  /* LMS support for CuDNN library */

  // GPU Memory Management FIXME(feiw) put this in the library??
  def startArena() = {
      unchecked[Unit]("CUDA_CALL(cudaMalloc(&gpuMallocBase, HEAP_SIZE))")
      unchecked[Unit]("CUDA_CALL(cudaMemset(gpuMallocBase, 0, HEAP_SIZE))")
      unchecked[Unit]("gpuMallocAddr = gpuMallocBase")
  }

  // alloc and free memory
  // case class SizeT(x: Int) { override def toString() = x.toString }
  // implicit def sizeTRepToOps(x: Rep[SizeT])(implicit __pos: SourceContext): SizeTOps = new SizeTOps(x)(__pos)
  // implicit def sizeTVarToOps(x: Var[SizeT])(implicit __pos: SourceContext): SizeTOps = new SizeTOps(readVar(x))(__pos)
  // class SizeTOps(x: Rep[SizeT])(implicit __pos: SourceContext) {
  //   def toInt: Rep[Int] = Wrap[Int](Unwrap(x))
  // }
  // implicit def sizeTFromInt(x: Int) = SizeT(x)

  def gpuArenaMalloc[T:Manifest](size: Rep[SizeT]): Rep[Array[T]] = {
    // libFunction[Array[T]]("myGpuMalloc", Unwrap(size))(Seq[Int](), Seq[Int](), Set[Int](), Adapter.CTRL)
    Wrap[Array[T]](Adapter.g.reflectWrite("myGpuMalloc-f", Unwrap(size))(Adapter.CTRL)) // FIXME(feiw) fix write effect to arena??
  }
  def gpuArenaFree(size: Rep[SizeT]): Rep[Unit] = {
    // libFunction[Unit]("myGpuFree", Unwrap(size))(Seq[Int](), Seq[Int](), Set[Int](), Adapter.CTRL)
    Wrap[Unit](Adapter.g.reflectWrite("myGpuFree-f", Unwrap(size))(Adapter.CTRL)) // FIXME(feiw) fix write effect to arena ??
  }

  // More Principled Cublas binding approach
  abstract class CublasHandleT
  // lazy val here so that we only ever create one handle
  lazy val cublasHandle = newStruct[CublasHandleT]
  lazy val zero = var_new(0.0f)
  lazy val one = var_new(1.0f)
  lazy val minus_one = var_new(-1.0f)

  // cuRAND - random generator global seed
  var seed = 0L
  // global per thread (GPU) offset for random num generator
  lazy val offset = var_new[Long](0L)

  def resetSeed(value: Long = 0L) = {
    seed = value
    offset -= offset
  }

  abstract class CublasStatusT
  def cublasCreate(handle: Rep[CublasHandleT]) =
    libFunction[CublasStatusT]("cublasCreate", Unwrap(handle))(Seq[Int](), Seq(0), Set(0))
  def cublasDestroy(handle: Rep[CublasHandleT]) =
    libFunction[CublasStatusT]("cublasDestroy", Unwrap(handle))(Seq[Int](), Seq(0), Set[Int]())
  def cublasCall(status: Rep[CublasStatusT]) =
    libFunction[Unit]("CUBLAS_CALL", Unwrap(status))(Seq[Int](), Seq[Int](), Set[Int](), Adapter.CTRL)

  abstract class CublasOperationT
  def cublasOpN = cmacro[CublasOperationT]("CUBLAS_OP_N")
  def cublasOpT = cmacro[CublasOperationT]("CUBLAS_OP_T")
  def INFINITY = cmacro[Float]("INFINITY")

  /*
    cublasStatus_t cublasSdot (cublasHandle_t handle, int n,
                           const float           *x, int incx,
                           const float           *y, int incy,
                           float           *result)
   */
  def cublasSdot_(handle: Rep[CublasHandleT], n: Rep[Int], x: Rep[Array[Float]], incx: Rep[Int],
                  y: Rep[Array[Float]], incy: Rep[Int], result: Rep[Array[Float]]) =
    libFunction[CublasStatusT]("cublasSdot",
      Unwrap(handle), Unwrap(n), Unwrap(x), Unwrap(incx), Unwrap(y), Unwrap(incy), Unwrap(result))(Seq(0, 2, 4), Seq(6), Set[Int]())

  /*
    matching syntax of this library function
    cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *x, int incx,
                           const float           *beta,
                           float           *y, int incy)
  */
  def cublasSgemv_(handle: Rep[CublasHandleT], trans: Rep[CublasOperationT], m: Rep[Int], n: Rep[Int], alpha: Var[Float],
                   A: Rep[Array[Float]], lda: Rep[Int], x: Rep[Array[Float]], incx: Rep[Int], beta: Var[Float],
                   y: Rep[Array[Float]], incy: Rep[Int]): Rep[CublasStatusT] =
    libFunction[CublasStatusT]("cublasSgemv",
      Unwrap(handle), Unwrap(trans), Unwrap(m), Unwrap(n), UnwrapV(alpha), Unwrap(A), Unwrap(lda), Unwrap(x), Unwrap(incx),
      UnwrapV(beta), Unwrap(y), Unwrap(incy))(Seq(0,5,7,10), Seq(10), Set(4, 9))

  /*
    matching syntax of this library function
    cublasStatus_t cublasSgeam(cublasHandle_t handle,
                          cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n,
                          const float           *alpha,
                          const float           *A, int lda,
                          const float           *beta,
                          const float           *B, int ldb,
                          float           *C, int ldc)
  */
  def cublasSgeam_(handle: Rep[CublasHandleT], transa: Rep[CublasOperationT], transb: Rep[CublasOperationT],
                    m: Rep[Int], n: Rep[Int], alpha: Var[Float], A: Rep[Array[Float]], lda: Rep[Int],
                    beta: Var[Float], B: Rep[Array[Float]], ldb: Rep[Int],
                    C: Rep[Array[Float]], ldc: Rep[Int]): Rep[CublasStatusT] =
    libFunction[CublasStatusT]("cublasSgeam",
      Unwrap(cublasHandle), Unwrap(transa), Unwrap(transb), Unwrap(m), Unwrap(n), UnwrapV(alpha),
      Unwrap(A), Unwrap(lda), UnwrapV(beta), Unwrap(B), Unwrap(ldb), Unwrap(C), Unwrap(ldc))(Seq(0, 6, 9, 11), Seq(11), Set(5, 8))

  /*
    cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc)
   */
  def cublasSgemm_(handle: Rep[CublasHandleT], transa: Rep[CublasOperationT], transb: Rep[CublasOperationT],
                   m: Rep[Int], n: Rep[Int], k: Rep[Int], alpha: Var[Float], A: Rep[Array[Float]], lda: Rep[Int],
                   B: Rep[Array[Float]], ldb: Rep[Int], beta: Var[Float], C: Rep[Array[Float]], ldc: Rep[Int]) =
    libFunction[CublasStatusT]("cublasSgemm",
      Unwrap(cublasHandle), Unwrap(transa), Unwrap(transb), Unwrap(m), Unwrap(n), Unwrap(k), UnwrapV(alpha),
      Unwrap(A), Unwrap(lda), Unwrap(B), Unwrap(ldb), UnwrapV(beta), Unwrap(C), Unwrap(ldc))(Seq(0,7,9,12), Seq(12), Set(6, 11))

  /*
  cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle,
                                        cublasOperation_t transa,
                                        cublasOperation_t transb,
                                        int m, int n, int k,
                                        const float *alpha,
                                        const float *A, int lda,
                                        long long int strideA,
                                        const float *B, int ldb,
                                        long long int strideB,
                                        const float *beta,
                                        float *C, int ldc,
                                        long long int strideC,
                                        int batchCount)
   */
  def cublasSgemmStridedBatched_(handle: Rep[CublasHandleT], transa: Rep[CublasOperationT], transb: Rep[CublasOperationT],
                                 m: Rep[Int], n: Rep[Int], k: Rep[Int], alpha: Var[Float], A: Rep[Array[Float]], lda: Rep[Int],
                                 strideA: Rep[Long], B: Rep[Array[Float]], ldb: Rep[Int], strideB: Rep[Long], beta: Var[Float],
                                 C: Rep[Array[Float]], ldc: Rep[Int], strideC: Rep[Long], batchCount: Rep[Int]) =
    libFunction[CublasStatusT]("cublasSgemmStridedBatched", Unwrap(cublasHandle), Unwrap(transa), Unwrap(transb), Unwrap(m),
      Unwrap(n), Unwrap(k), UnwrapV(alpha), Unwrap(A), Unwrap(lda), Unwrap(strideA), Unwrap(B), Unwrap(ldb), Unwrap(strideB),
      UnwrapV(beta), Unwrap(C), Unwrap(ldc), Unwrap(strideC), Unwrap(batchCount))((0 to 13) ++ (15 to 17), Seq(14), Set(6, 13))

  // other gpu kernel function bindings
  def cudaArrayFill_(res: Rep[Array[Float]], value: Rep[Float], size: Rep[Int]): Rep[Unit] =
    libFunction[Unit]("arrayFill<<<28,512>>>", Unwrap(res), Unwrap(value), Unwrap(size))(Seq[Int](), Seq(0), Set[Int]())

  def cudaArrayClipAt_(res: Rep[Array[Float]], bound: Rep[Float], size: Rep[Int]): Rep[Unit] =
    libFunction[Unit]("clipAt<<<28,512>>>", Unwrap(res), Unwrap(bound), Unwrap(size))(Seq(0), Seq(0), Set[Int]())

  abstract class Dim3
  def dim3(a: Rep[Int], b: Rep[Int], c: Rep[Int]): Rep[Dim3] =
    Wrap[Dim3](Adapter.g.reflectUnsafe("lib-struct", lms.core.Backend.Const(manifest[Dim3]), Unwrap(a), Unwrap(b), Unwrap(c)))
  def dim3(a: Rep[Int], b: Rep[Int]): Rep[Dim3] =
    Wrap[Dim3](Adapter.g.reflectUnsafe("lib-struct", lms.core.Backend.Const(manifest[Dim3]), Unwrap(a), Unwrap(b)))

  def permute2D_(dimGrid: Rep[Dim3], dimBlock: Rep[Dim3], res: Rep[Array[Float]], input: Rep[Array[Float]], shape0: Rep[Int], shape1: Rep[Int]) =
    cudaFunction[Unit]("permute2D", Seq(Unwrap(dimGrid), Unwrap(dimBlock)), Unwrap(res), Unwrap(input),
      Unwrap(shape0), Unwrap(shape1))(Seq(0, 1), Seq(0), Set[Int]())

  def permute3DSim_(dimGrid: Rep[Dim3], dimBlock: Rep[Dim3], res: Rep[Array[Float]], input: Rep[Array[Float]], shape0: Rep[Int],
      shape1: Rep[Int], shape2: Rep[Int]) =
    cudaFunction[Unit]("permuteSim3D", Seq(Unwrap(dimGrid), Unwrap(dimBlock)), Unwrap(res), Unwrap(input),
      Unwrap(shape0), Unwrap(shape1), Unwrap(shape2))(Seq(0, 1), Seq(0), Set[Int]())

  def permute3D210_(dimGrid: Rep[Dim3], dimBlock: Rep[Dim3], res: Rep[Array[Float]], input: Rep[Array[Float]], shape0: Rep[Int],
      shape1: Rep[Int], shape2: Rep[Int], in_strides0: Rep[Int], in_strides1: Rep[Int], res_strides0: Rep[Int], res_strides1: Rep[Int]) =
    cudaFunction[Unit]("permute3D_dim2to0_dim0to2", Seq(Unwrap(dimGrid), Unwrap(dimBlock)), Unwrap(res), Unwrap(input),
      Unwrap(shape0), Unwrap(shape1), Unwrap(shape2), Unwrap(in_strides0), Unwrap(in_strides1), Unwrap(res_strides0), Unwrap(res_strides1))(Seq(0, 1), Seq(0), Set[Int]())

  def permute3D120_(dimGrid: Rep[Dim3], dimBlock: Rep[Dim3], res: Rep[Array[Float]], input: Rep[Array[Float]], shape0: Rep[Int],
        shape1: Rep[Int], shape2: Rep[Int], in_strides0: Rep[Int], in_strides1: Rep[Int], res_strides0: Rep[Int], res_strides1: Rep[Int]) =
      cudaFunction[Unit]("permute3D_dim2to1_dim0to2", Seq(Unwrap(dimGrid), Unwrap(dimBlock)), Unwrap(res), Unwrap(input),
        Unwrap(shape0), Unwrap(shape1), Unwrap(shape2), Unwrap(in_strides0), Unwrap(in_strides1), Unwrap(res_strides0), Unwrap(res_strides1))(Seq(0, 1), Seq(0), Set[Int]())

  def permute3D201_(dimGrid: Rep[Dim3], dimBlock: Rep[Dim3], res: Rep[Array[Float]], input: Rep[Array[Float]], shape0: Rep[Int],
        shape1: Rep[Int], shape2: Rep[Int], in_strides0: Rep[Int], in_strides1: Rep[Int], res_strides0: Rep[Int], res_strides1: Rep[Int]) =
      cudaFunction[Unit]("permute3D_dim2to0_dim0to1", Seq(Unwrap(dimGrid), Unwrap(dimBlock)), Unwrap(res), Unwrap(input),
        Unwrap(shape0), Unwrap(shape1), Unwrap(shape2), Unwrap(in_strides0), Unwrap(in_strides1), Unwrap(res_strides0), Unwrap(res_strides1))(Seq(0, 1), Seq(0), Set[Int]())

  def permute3D021_(dimGrid: Rep[Dim3], dimBlock: Rep[Dim3], res: Rep[Array[Float]], input: Rep[Array[Float]], shape0: Rep[Int],
        shape1: Rep[Int], shape2: Rep[Int], in_strides0: Rep[Int], in_strides1: Rep[Int], res_strides0: Rep[Int], res_strides1: Rep[Int]) =
      cudaFunction[Unit]("permute3D_dim2to1_dim0to0", Seq(Unwrap(dimGrid), Unwrap(dimBlock)), Unwrap(res), Unwrap(input),
        Unwrap(shape0), Unwrap(shape1), Unwrap(shape2), Unwrap(in_strides0), Unwrap(in_strides1), Unwrap(res_strides0), Unwrap(res_strides1))(Seq(0, 1), Seq(0), Set[Int]())

  def permute4D0123_(dimGrid: Rep[Dim3], dimBlock: Rep[Dim3], res: Rep[Array[Float]], input: Rep[Array[Float]],
      in_strides0: Rep[Int], in_strides1: Rep[Int], in_strides2: Rep[Int],
      res_strides0: Rep[Int], res_strides1: Rep[Int], res_strides2: Rep[Int]) =
    cudaFunction[Unit]("permuteSim4DSim012", Seq(Unwrap(dimGrid), Unwrap(dimBlock)), Unwrap(res), Unwrap(input),
      Unwrap(in_strides0), Unwrap(in_strides1), Unwrap(in_strides2),
      Unwrap(res_strides0), Unwrap(res_strides1), Unwrap(res_strides2))(Seq(0, 1), Seq(0), Set[Int]())

  def permute4D0213_(dimGrid: Rep[Dim3], dimBlock: Rep[Dim3], res: Rep[Array[Float]], input: Rep[Array[Float]],
      in_strides0: Rep[Int], in_strides1: Rep[Int], in_strides2: Rep[Int],
      res_strides0: Rep[Int], res_strides1: Rep[Int], res_strides2: Rep[Int]) =
    cudaFunction[Unit]("permuteSim4DSim021", Seq(Unwrap(dimGrid), Unwrap(dimBlock)), Unwrap(res), Unwrap(input),
      Unwrap(in_strides0), Unwrap(in_strides1), Unwrap(in_strides2),
      Unwrap(res_strides0), Unwrap(res_strides1), Unwrap(res_strides2))(Seq(0, 1), Seq(0), Set[Int]())

  def permute4D1023_(dimGrid: Rep[Dim3], dimBlock: Rep[Dim3], res: Rep[Array[Float]], input: Rep[Array[Float]],
      in_strides0: Rep[Int], in_strides1: Rep[Int], in_strides2: Rep[Int],
      res_strides0: Rep[Int], res_strides1: Rep[Int], res_strides2: Rep[Int]) =
    cudaFunction[Unit]("permuteSim4DSim102", Seq(Unwrap(dimGrid), Unwrap(dimBlock)), Unwrap(res), Unwrap(input),
      Unwrap(in_strides0), Unwrap(in_strides1), Unwrap(in_strides2),
      Unwrap(res_strides0), Unwrap(res_strides1), Unwrap(res_strides2))(Seq(0, 1), Seq(0), Set[Int]())

  def permute4D1203_(dimGrid: Rep[Dim3], dimBlock: Rep[Dim3], res: Rep[Array[Float]], input: Rep[Array[Float]],
      in_strides0: Rep[Int], in_strides1: Rep[Int], in_strides2: Rep[Int],
      res_strides0: Rep[Int], res_strides1: Rep[Int], res_strides2: Rep[Int]) =
    cudaFunction[Unit]("permuteSim4DSim120", Seq(Unwrap(dimGrid), Unwrap(dimBlock)), Unwrap(res), Unwrap(input),
      Unwrap(in_strides0), Unwrap(in_strides1), Unwrap(in_strides2),
      Unwrap(res_strides0), Unwrap(res_strides1), Unwrap(res_strides2))(Seq(0, 1), Seq(0), Set[Int]())

  def permute4D2013_(dimGrid: Rep[Dim3], dimBlock: Rep[Dim3], res: Rep[Array[Float]], input: Rep[Array[Float]],
      in_strides0: Rep[Int], in_strides1: Rep[Int], in_strides2: Rep[Int],
      res_strides0: Rep[Int], res_strides1: Rep[Int], res_strides2: Rep[Int]) =
    cudaFunction[Unit]("permuteSim4DSim201", Seq(Unwrap(dimGrid), Unwrap(dimBlock)), Unwrap(res), Unwrap(input),
      Unwrap(in_strides0), Unwrap(in_strides1), Unwrap(in_strides2),
      Unwrap(res_strides0), Unwrap(res_strides1), Unwrap(res_strides2))(Seq(0, 1), Seq(0), Set[Int]())

  def permute4D2103_(dimGrid: Rep[Dim3], dimBlock: Rep[Dim3], res: Rep[Array[Float]], input: Rep[Array[Float]],
      in_strides0: Rep[Int], in_strides1: Rep[Int], in_strides2: Rep[Int],
      res_strides0: Rep[Int], res_strides1: Rep[Int], res_strides2: Rep[Int]) =
    cudaFunction[Unit]("permuteSim4DSim210", Seq(Unwrap(dimGrid), Unwrap(dimBlock)), Unwrap(res), Unwrap(input),
      Unwrap(in_strides0), Unwrap(in_strides1), Unwrap(in_strides2),
      Unwrap(res_strides0), Unwrap(res_strides1), Unwrap(res_strides2))(Seq(0, 1), Seq(0), Set[Int]())

  def mask4D_(input: Rep[Array[Float]], lengths: Rep[Array[Int]], in_strides0: Rep[Int], in_strides1: Rep[Int],
      in_strides2: Rep[Int], in_strides3: Rep[Int], total: Rep[Int]) =
    libFunction[Unit]("mask4D<<<28, 512>>>", Unwrap(input), Unwrap(lengths), Unwrap(in_strides0), Unwrap(in_strides1),
    Unwrap(in_strides2), Unwrap(in_strides3), Unwrap(total))(Seq(0, 1), Seq(0), Set[Int]())

  def maskedFill_(input: Rep[Array[Float]], output: Rep[Array[Float]], mask: Rep[Array[Int]], value: Rep[Float],
                  dim0Shape: Rep[Int], dim0Stride: Rep[Int], dim1Shape: Rep[Int], dim1Stride: Rep[Int],
                  offsetSize: Rep[Int], inputSize: Rep[Int], ijSwapped: Boolean) =
  libFunction[Unit](s"maskedFill<${Unwrap{ijSwapped}}><<<28, 512>>>", Unwrap(input), Unwrap(output), Unwrap(mask), Unwrap(value),
    Unwrap(dim0Shape), Unwrap(dim0Stride), Unwrap(dim1Shape), Unwrap(dim1Stride), Unwrap(offsetSize),
    Unwrap(inputSize))(Seq(0, 2), Seq(1), Set())

  // update x gradients based on y values (y comes from backprop)
  def maskedFillGrad_(y_d: Rep[Array[Float]], x_d: Rep[Array[Float]], mask: Rep[Array[Int]], dim0Shape: Rep[Int],
                      dim0Stride: Rep[Int], dim1Shape: Rep[Int], dim1Stride: Rep[Int], offsetSize: Rep[Int],
                      inputSize: Rep[Int], ijSwapped: Boolean) =
    libFunction[Unit](s"maskedFillGrad<${Unwrap{ijSwapped}}><<<28, 512>>>", Unwrap(y_d), Unwrap(x_d), Unwrap(mask), Unwrap(dim0Shape),
      Unwrap(dim0Stride), Unwrap(dim1Shape), Unwrap(dim1Stride), Unwrap(offsetSize), Unwrap(inputSize))(Seq(0, 1, 2), Seq(1), Set())

  def hardTanh_(input: Rep[Array[Float]], res: Rep[Array[Float]], min_val: Rep[Float], max_val: Rep[Float], total: Rep[Int]) =
    libFunction[Unit]("hardTanh<<<28, 512>>>", Unwrap(input), Unwrap(res), Unwrap(min_val), Unwrap(max_val),
      Unwrap(total))(Seq(0,1), Seq(0,1), Set[Int]())

  def hardTanhGrad_(inputX: Rep[Array[Float]], inputD: Rep[Array[Float]], resD: Rep[Array[Float]], min_val: Rep[Float],
      max_val: Rep[Float], total: Rep[Int], inPlace: Rep[Boolean]) =
    libFunction[Unit]("hardTanh_grad<<<28, 512>>>", Unwrap(inputX), Unwrap(inputD), Unwrap(resD), Unwrap(min_val),
      Unwrap(max_val), Unwrap(total), Unwrap(inPlace))(Seq(0,1,2), Seq(1), Set[Int]())

  def log_(input: Rep[Array[Float]], res: Rep[Array[Float]], total: Rep[Int]) =
    libFunction[Unit]("elementwise_1D_1D_log<<<28, 512>>>", Unwrap(input), Unwrap(res), Unwrap(total))(Seq(0), Seq(1), Set[Int]())
  def exp_(input: Rep[Array[Float]], res: Rep[Array[Float]], total: Rep[Int]) =
    libFunction[Unit]("elementwise_1D_1D_exp<<<28, 512>>>", Unwrap(input), Unwrap(res), Unwrap(total))(Seq(0), Seq(1), Set[Int]())
  def sqrt_(input: Rep[Array[Float]], res: Rep[Array[Float]], total: Rep[Int]) =
    libFunction[Unit]("elementwise_1D_1D_sqrt<<<28, 512>>>", Unwrap(input), Unwrap(res), Unwrap(total))(Seq(0), Seq(1), Set[Int]())
  def square_(input: Rep[Array[Float]], res: Rep[Array[Float]], total: Rep[Int]) =
    libFunction[Unit]("elementwise_1D_1D_square<<<28, 512>>>", Unwrap(input), Unwrap(res), Unwrap(total))(Seq(0), Seq(1), Set[Int]())
  def log_grad_(inputX: Rep[Array[Float]], inputD: Rep[Array[Float]], outputX: Rep[Array[Float]], outputD: Rep[Array[Float]],
    total: Rep[Int]) = libFunction[Unit]("elementwise_1D_1D_log_grad<<<28, 512>>>", Unwrap(inputX), Unwrap(inputD),
      Unwrap(outputX), Unwrap(outputD), Unwrap(total))(Seq(0,1,2,3), Seq(1), Set[Int]())
  def exp_grad_(inputX: Rep[Array[Float]], inputD: Rep[Array[Float]], outputX: Rep[Array[Float]], outputD: Rep[Array[Float]],
    total: Rep[Int]) = libFunction[Unit]("elementwise_1D_1D_exp_grad<<<28, 512>>>", Unwrap(inputX), Unwrap(inputD),
      Unwrap(outputX), Unwrap(outputD), Unwrap(total))(Seq(0,1,2,3), Seq(1), Set[Int]())
  def sqrt_grad_(inputX: Rep[Array[Float]], inputD: Rep[Array[Float]], outputX: Rep[Array[Float]], outputD: Rep[Array[Float]],
    total: Rep[Int]) = libFunction[Unit]("elementwise_1D_1D_sqrt_grad<<<28, 512>>>", Unwrap(inputX), Unwrap(inputD),
      Unwrap(outputX), Unwrap(outputD), Unwrap(total))(Seq(0,1,2,3), Seq(1), Set[Int]())
  def square_grad_(inputX: Rep[Array[Float]], inputD: Rep[Array[Float]], outputX: Rep[Array[Float]], outputD: Rep[Array[Float]],
    total: Rep[Int]) = libFunction[Unit]("elementwise_1D_1D_square_grad<<<28, 512>>>", Unwrap(inputX), Unwrap(inputD),
      Unwrap(outputX), Unwrap(outputD), Unwrap(total))(Seq(0,1,2,3), Seq(1), Set[Int]())

  def nllLoss_(input: Rep[Array[Float]], target: Rep[Array[Int]], res: Rep[Array[Float]], batchSize: Rep[Int], in_strides: Rep[Int]) =
    cudaFunction[Unit]("nllLoss", Seq(Unwrap(batchSize), lms.core.Backend.Const(1)), Unwrap(input), Unwrap(in_strides),
      Unwrap(res), Unwrap(target))(Seq(0, 1, 2), Seq(2), Set[Int]())
  def nllLossGrad_(inputD: Rep[Array[Float]], target: Rep[Array[Int]], resD: Rep[Array[Float]], batchSize: Rep[Int], in_strides: Rep[Int]) =
    cudaFunction[Unit]("nllLoss_grad", Seq(Unwrap(batchSize), lms.core.Backend.Const(1)), Unwrap(in_strides), Unwrap(resD),
      Unwrap(target), Unwrap(inputD))(Seq(0,1,2), Seq(0), Set[Int]())

  def concat4D_(grid: Rep[Dim3],
      in1: Rep[Array[Float]], dimSize1: Rep[Int], nElement1: Rep[Int],
      in2: Rep[Array[Float]], dimSize2: Rep[Int], nElement2: Rep[Int],
      out: Rep[Array[Float]], concatDim: Rep[Int],
      outSize0: Rep[Int], outSize1: Rep[Int], outSize2: Rep[Int], outSize3: Rep[Int],
      outStride0: Rep[Int], outStride1: Rep[Int], outStride2: Rep[Int], outStride3: Rep[Int]) =
    cudaFunction[Unit]("concat2D_1D_greg", Seq(Unwrap(grid), lms.core.Backend.Const(512)),
      Unwrap(in1), Unwrap(dimSize1), Unwrap(nElement1),
      Unwrap(in2), Unwrap(dimSize2), Unwrap(nElement2),
      Unwrap(out), Unwrap(concatDim),
      Unwrap(outSize0), Unwrap(outSize1), Unwrap(outSize2), Unwrap(outSize3),
      Unwrap(outStride0), Unwrap(outStride1), Unwrap(outStride2), Unwrap(outStride3))(Seq(0, 3), Seq(6), Set[Int]())

  def concat4D_grad_(grid: Rep[Dim3],
      in1: Rep[Array[Float]], dimSize1: Rep[Int], nElement1: Rep[Int],
      in2: Rep[Array[Float]], dimSize2: Rep[Int], nElement2: Rep[Int],
      out: Rep[Array[Float]], concatDim: Rep[Int],
      outSize0: Rep[Int], outSize1: Rep[Int], outSize2: Rep[Int], outSize3: Rep[Int],
      outStride0: Rep[Int], outStride1: Rep[Int], outStride2: Rep[Int], outStride3: Rep[Int]) =
    cudaFunction[Unit]("concat2D_1D_greg_grad", Seq(Unwrap(grid), lms.core.Backend.Const(512)),
      Unwrap(in1), Unwrap(dimSize1), Unwrap(nElement1),
      Unwrap(in2), Unwrap(dimSize2), Unwrap(nElement2),
      Unwrap(out), Unwrap(concatDim),
      Unwrap(outSize0), Unwrap(outSize1), Unwrap(outSize2), Unwrap(outSize3),
      Unwrap(outStride0), Unwrap(outStride1), Unwrap(outStride2), Unwrap(outStride3))(Seq(6), Seq(0, 3), Set[Int]())

  def adagrad_(x: Rep[Array[Float]], d: Rep[Array[Float]], m: Rep[Array[Float]], clip: Rep[Float], lr: Rep[Float], size: Rep[Int]) =
    libFunction[Unit]("adagrad_update_1D_1D<<<28, 512>>>", Unwrap(x), Unwrap(d), Unwrap(m), Unwrap(clip),
      Unwrap(lr), Unwrap(size))(Seq(0, 1, 2), Seq(0, 1, 2), Set[Int]())

  def momentum_(x: Rep[Array[Float]], d: Rep[Array[Float]], m: Rep[Array[Float]], lr: Rep[Float], momentum: Rep[Float],
      clip: Rep[Float], nesterov: Rep[Boolean], size: Rep[Int]) =
    libFunction[Unit]("momentum_update_1D_1D<<<28, 512>>>", Unwrap(x), Unwrap(d), Unwrap(m), Unwrap(lr),
      Unwrap(momentum), Unwrap(clip), Unwrap(nesterov), Unwrap(size))(Seq(0, 1, 2), Seq(0, 1, 2), Set[Int]())

  def arithScalar_(op: String, in: Rep[Array[Float]], out: Rep[Array[Float]], mult: Rep[Float], size: Rep[Int]) =
    libFunction[Unit]( op match {
      case "+" => "addScalar<<<28, 512>>>"
      case "-" => "minusScalar<<<28, 512>>>"
      case "*" => "multScalar<<<28, 512>>>"
      case "/" => "divScalar<<<28, 512>>>"
    }, Unwrap(in), Unwrap(out), Unwrap(mult), Unwrap(size))(Seq(0), Seq(1), Set[Int]())

  def elementWiseWithBroadCastRank1_(
      op: String, size: Rep[Int],
      in1: Rep[Array[Float]], in1Stride0: Rep[Int],
      in2: Rep[Array[Float]], in2Stride0: Rep[Int],
      out: Rep[Array[Float]], outStride0: Rep[Int]) =
    libFunction[Unit](
      op match {
        case "+" => "elementWiseWithBroadCastRank1Add<<<28, 512>>>"
        case "-" => "elementWiseWithBroadCastRank1Minus<<<28, 512>>>"
        case "*" => "elementWiseWithBroadCastRank1Mult<<<28, 512>>>"
        case "/" => "elementWiseWithBroadCastRank1Div<<<28, 512>>>"
      },
      Unwrap(in1), Unwrap(in2), Unwrap(out), Unwrap(size), Unwrap(in1Stride0), Unwrap(in2Stride0),
      Unwrap(outStride0))(Seq(0, 1), Seq(2), Set[Int]())

  def elementWiseWithBroadCastRank2_(
      op: String, size: Rep[Int],
      in1: Rep[Array[Float]], in1Stride0: Rep[Int], in1Stride1: Rep[Int],
      in2: Rep[Array[Float]], in2Stride0: Rep[Int], in2Stride1: Rep[Int],
      out: Rep[Array[Float]], outStride0: Rep[Int], outStride1: Rep[Int]) =
    libFunction[Unit](
      op match {
        case "+" => "elementWiseWithBroadCastRank2Add<<<28, 512>>>"
        case "-" => "elementWiseWithBroadCastRank2Minus<<<28, 512>>>"
        case "*" => "elementWiseWithBroadCastRank2Mult<<<28, 512>>>"
        case "/" => "elementWiseWithBroadCastRank2Div<<<28, 512>>>"
      },
      Unwrap(in1), Unwrap(in2), Unwrap(out), Unwrap(size), Unwrap(in1Stride0), Unwrap(in1Stride1), Unwrap(in2Stride0),
      Unwrap(in2Stride1), Unwrap(outStride0), Unwrap(outStride1))(Seq(0, 1), Seq(2), Set[Int]())

  def elementWiseWithBroadCastRank3_(
      op: String, size: Rep[Int],
      in1: Rep[Array[Float]], in1Stride0: Rep[Int], in1Stride1: Rep[Int], in1Stride2: Rep[Int],
      in2: Rep[Array[Float]], in2Stride0: Rep[Int], in2Stride1: Rep[Int], in2Stride2: Rep[Int],
      out: Rep[Array[Float]], outStride0: Rep[Int], outStride1: Rep[Int], outStride2: Rep[Int]) =
    libFunction[Unit](
      op match {
        case "+" => "elementWiseWithBroadCastRank3Add<<<28, 512>>>"
        case "-" => "elementWiseWithBroadCastRank3Minus<<<28, 512>>>"
        case "*" => "elementWiseWithBroadCastRank3Mult<<<28, 512>>>"
        case "/" => "elementWiseWithBroadCastRank3Div<<<28, 512>>>"
      },
      Unwrap(in1), Unwrap(in2), Unwrap(out), Unwrap(size), Unwrap(in1Stride0), Unwrap(in1Stride1), Unwrap(in1Stride2),
      Unwrap(in2Stride0), Unwrap(in2Stride1), Unwrap(in2Stride2),
      Unwrap(outStride0), Unwrap(outStride1), Unwrap(outStride2))(Seq(0, 1), Seq(2), Set[Int]())

  def elementWiseWithBroadCastRank4_(
      op: String, size: Rep[Int],
      in1: Rep[Array[Float]], in1Stride0: Rep[Int], in1Stride1: Rep[Int], in1Stride2: Rep[Int], in1Stride3: Rep[Int],
      in2: Rep[Array[Float]], in2Stride0: Rep[Int], in2Stride1: Rep[Int], in2Stride2: Rep[Int], in2Stride3: Rep[Int],
      out: Rep[Array[Float]], outStride0: Rep[Int], outStride1: Rep[Int], outStride2: Rep[Int], outStride3: Rep[Int]) =
    libFunction[Unit](
      op match {
        case "+" => "elementWiseWithBroadCastRank4Add<<<28, 512>>>"
        case "-" => "elementWiseWithBroadCastRank4Minus<<<28, 512>>>"
        case "*" => "elementWiseWithBroadCastRank4Mult<<<28, 512>>>"
        case "/" => "elementWiseWithBroadCastRank4Div<<<28, 512>>>"
      },
      Unwrap(in1), Unwrap(in2), Unwrap(out), Unwrap(size),
      Unwrap(in1Stride0), Unwrap(in1Stride1), Unwrap(in1Stride2), Unwrap(in1Stride3),
      Unwrap(in2Stride0), Unwrap(in2Stride1), Unwrap(in2Stride2), Unwrap(in2Stride3),
      Unwrap(outStride0), Unwrap(outStride1), Unwrap(outStride2), Unwrap(outStride3))(Seq(0, 1), Seq(2), Set[Int]())


  // addScalarInArrayInPlace(float* in, float* add, float scale, int size)
  def addScalarInArrayInPlace_(in: Rep[Array[Float]], add: Rep[Array[Float]], scale: Rep[Float], size: Rep[Int]) =
    libFunction[Unit]("addScalarInArrayInPlace<<<28, 512>>>", Unwrap(in), Unwrap(add), Unwrap(scale), Unwrap(size))(Seq(0, 1), Seq(0, 1), Set[Int]())

  // sum_grad(float* in, int inSize0, int inSize1, int inSize2, int inSize3, int nElement,
  //                        float* out, int outStride0, int outStride1, int outStride2, int dim)
  def sumGrad_(in: Rep[Array[Float]], inSize0: Rep[Int], inSize1: Rep[Int], inSize2: Rep[Int], inSize3: Rep[Int],
      nElement: Rep[Int], out: Rep[Array[Float]], outStride0: Rep[Int], outStride1: Rep[Int], outStride2: Rep[Int], dim: Rep[Int])=
    libFunction[Unit]("sum_grad<<<28, 512>>>", Unwrap(in), Unwrap(inSize0), Unwrap(inSize1), Unwrap(inSize2), Unwrap(inSize3),
      Unwrap(nElement), Unwrap(out), Unwrap(outStride0), Unwrap(outStride1), Unwrap(outStride2),
      Unwrap(dim))(Seq(0, 6), Seq(0, 6), Set[Int]())

  // repeat0(float* in, float* out, int outStride0, int outStride1, int outScalarCount)
  def repeat_(in: Rep[Array[Float]], out: Rep[Array[Float]], outStride0: Rep[Int], outStride1: Rep[Int], outScalarCount: Rep[Int]) =
    libFunction[Unit]("repeat0<<<28, 512>>>", Unwrap(in), Unwrap(out), Unwrap(outStride0), Unwrap(outStride1),
      Unwrap(outScalarCount))(Seq(0), Seq(1), Set[Int]())

  // void shift0(float* in, float* out, int inDim0, int inStride0, int inStride1, int inScalarCount)
  def shift_(in: Rep[Array[Float]], out: Rep[Array[Float]], inDim0: Rep[Int], inStride0: Rep[Int], inStride1: Rep[Int], inScalarCount: Rep[Int]) =
    libFunction[Unit]("shift0<<<28, 512>>>", Unwrap(in), Unwrap(out), Unwrap(inDim0), Unwrap(inStride0), Unwrap(inStride1),
      Unwrap(inScalarCount))(Seq(0), Seq(1), Set[Int]())

  // void dropout(float* input, float *result, float p, bool *mask, int inputSize, long seed, long offset)
  def dropout_(input: Rep[Array[Float]], result: Rep[Array[Float]], p: Float, mask: Rep[Array[Boolean]], inputSize: Rep[Int],
               seed: Rep[Long], offset: Rep[Long]) =
    libFunction[Unit]("dropout<<<28, 512>>>", Unwrap(input), Unwrap(result), Unwrap(p), Unwrap(mask), Unwrap(inputSize),
      Unwrap(seed), Unwrap(offset))(Seq(0), Seq(1, 3), Set())

  // void dropoutGrad(float *y_d, float *x_d, bool *mask, float pinv)
  def dropoutGrad_(output: Rep[Array[Float]], input: Rep[Array[Float]], mask: Rep[Array[Boolean]], inputSize: Rep[Int], pInv: Rep[Float]) =
    libFunction[Unit]("dropoutGrad<<<28, 512>>>", Unwrap(output), Unwrap(input), Unwrap(mask), Unwrap(inputSize), Unwrap(pInv))(Seq(0, 1, 2),
      Seq(1), Set())

  // eg. outerSize for a 4d tensor scalaCount / lastDimSize (or shape(0) * shape(1) * shape(2)
  def softmax_(input: Rep[Array[Float]], output: Rep[Array[Float]], size: Rep[Int], outerSize: Rep[Int]) =
    libFunction[Unit](s"softmax<<<${Unwrap(outerSize)},64>>>", Unwrap(input), Unwrap(output), Unwrap(size))(Seq(0), Seq(1), Set())

  def softmaxGrad_(inputGrad: Rep[Array[Float]], outputGrad: Rep[Array[Float]], output: Rep[Array[Float]], size: Rep[Int],
                   outerSize: Rep[Int]) =
    libFunction[Unit](s"softmaxGrad<<<${Unwrap(outerSize)},64>>>", Unwrap(inputGrad), Unwrap(outputGrad), Unwrap(output),
      Unwrap(size))(Seq(1, 2), Seq(0), Set())

  // This is to compute softmax in the last dim. Last dim should have a stride of 1. This only works if dimSize <= 1024
  // softmaxElementsStride is the stride from one batch to another (usually, equals to softmaxElements)
  def dispatch_softmax_forward_(output: Rep[Array[Float]], input: Rep[Array[Float]], softmaxElements: Rep[Int],
                                softmaxElementsStride: Rep[Int], batchCount: Rep[Int]) =
    libFunction[Unit]("dispatch_softmax_forward<false>", Unwrap(output), Unwrap(input), Unwrap(softmaxElements),
      Unwrap(softmaxElementsStride), Unwrap(batchCount))(Seq(1), Seq(0), Set())

  def dispatch_softmax_backward(gradInput: Rep[Array[Float]], grad: Rep[Array[Float]], output: Rep[Array[Float]],
                                softmaxElements: Rep[Int], softmaxElementsStride: Rep[Int], batchCount: Rep[Int]) =
    libFunction[Unit]("dispatch_softmax_backward<false>", Unwrap(gradInput), Unwrap(grad), Unwrap(output), Unwrap(softmaxElements),
      Unwrap(softmaxElementsStride), Unwrap(batchCount))(Seq(1, 2), Seq(0), Set())

  def layer_norm_forward(x: Rep[Array[Float]], mean: Rep[Array[Float]], rstd: Rep[Array[Float]], gamma: Rep[Array[Float]],
                         beta: Rep[Array[Float]], res: Rep[Array[Float]], eps: Rep[Float], vectSize: Rep[Int],
                         outerSize: Rep[Int]) =
    libFunction[Unit](s"layer_norm_forward<<<${Unwrap(outerSize)}, 512>>>", Unwrap(x), Unwrap(mean), Unwrap(rstd),
      Unwrap(gamma), Unwrap(beta), Unwrap(res), Unwrap(eps), Unwrap(vectSize))(Seq(0, 3, 4), Seq(1, 2, 5), Set())

//  void layer_norm_grad(float* y_grad, float* x, float* mean, float* rstd, float* gamma, int outerSize, int vect_size,
//    float* x_grad, float* gamma_grad, float* beta_grad, float* scale, float* bias, float* s_grad, float* b_grad)
  // TODO - layer_norm_grad is not a cuda kernel, it launches multiple kernels. Write the logic in Scala and just do the kernel launches using this
  def layer_norm_grad(y_grad: Rep[Array[Float]], x: Rep[Array[Float]], mean: Rep[Array[Float]], rstd: Rep[Array[Float]],
                      gamma: Rep[Array[Float]], outerSize: Rep[Int], vectSize: Rep[Int], x_grad: Rep[Array[Float]],
                      gamma_grad: Rep[Array[Float]], beta_grad: Rep[Array[Float]], scale: Rep[Array[Float]],
                      bias: Rep[Array[Float]], s_grad: Rep[Array[Float]], b_grad: Rep[Array[Float]]) =
  libFunction[Unit]("layer_norm_grad", Unwrap(y_grad), Unwrap(x), Unwrap(mean), Unwrap(rstd), Unwrap(gamma), Unwrap(outerSize),
    Unwrap(vectSize), Unwrap(x_grad), Unwrap(gamma_grad), Unwrap(beta_grad), Unwrap(scale), Unwrap(bias), Unwrap(s_grad),
    Unwrap(b_grad))(Seq(1, 2, 3, 4), Seq(0, 7, 8, 9, 10, 11, 12, 13), Set())

  def plus_bias_kernel(input: Rep[Array[Float]], bias: Rep[Array[Float]], output: Rep[Array[Float]], inputSize: Rep[Int],
                       biasSize: Rep[Int], gridSize: Rep[Int]) =
    libFunction[Unit](s"plus_bias_kernel<<<${Unwrap(gridSize)}, 64>>>", Unwrap(input), Unwrap(bias), Unwrap(output),
      Unwrap(inputSize), Unwrap(biasSize))(Seq(0, 1), Seq(2), Set())

  def plus_bias_grad(y_grad: Rep[Array[Float]], bias_grad: Rep[Array[Float]], outer_size: Rep[Int],
                     bias_size: Rep[Int], blockSize: Rep[Int]) =
    libFunction[Unit](s"plus_bias_grad<<<${Unwrap(bias_size)}, ${Unwrap(blockSize)}>>>", Unwrap(y_grad),
      Unwrap(bias_grad), Unwrap(outer_size), Unwrap(bias_size))(Seq(0, 1), Seq(1), Set())

  def relu_kernel(input: Rep[Array[Float]], output: Rep[Array[Float]], inputSize: Rep[Int]) =
    libFunction[Unit]("relu_kernel<<<28, 512>>>", Unwrap(input), Unwrap(output), Unwrap(inputSize))(Seq(0, 1), Seq(0, 1), Set())

  def relu_grad_kernel(y_grad: Rep[Array[Float]], x_grad: Rep[Array[Float]], x: Rep[Array[Float]], inputSize: Rep[Int]) =
    libFunction[Unit]("relu_grad<<<28, 512>>>", Unwrap(y_grad), Unwrap(x_grad), Unwrap(x), Unwrap(inputSize))(Seq(0, 2), Seq(1), Set())

  def embedding_forward(weights: Rep[Array[Float]], indices: Rep[Array[Int]], output: Rep[Array[Float]],
                        embedSize: Rep[Int], outerSize: Rep[Int], blockSize: Rep[Int]) =
    libFunction[Unit](s"embedding_forward<<<${Unwrap(outerSize)}, ${Unwrap(blockSize)}>>>", Unwrap(weights),
      Unwrap(indices), Unwrap(output), Unwrap(embedSize))(Seq(0, 1), Seq(2), Set())

  def embedding_backward(indices: Rep[Array[Int]], y_grad: Rep[Array[Float]], weight_grad: Rep[Array[Float]],
                         indices_length: Rep[Int], embedSize: Rep[Int], paddingIdx: Rep[Int], gridSize: Rep[Int]) =
    libFunction[Unit](s"embedding_backward_feature_kernel<<<${Unwrap(gridSize)}, (32, 32), 32*32*sizeof(int) + 32*32*sizeof(float)>>>",
      Unwrap(indices), Unwrap(y_grad), Unwrap(weight_grad), Unwrap(indices_length), Unwrap(embedSize),
      Unwrap(paddingIdx))(Seq(0, 1, 2), Seq(2), Set())
}


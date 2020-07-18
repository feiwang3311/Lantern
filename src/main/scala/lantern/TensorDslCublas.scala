package lantern

import scala.util.continuations._

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.{Map => MutableMap}
import scala.math._

import lms.core.stub._
import lms.macros.SourceContext
import lms.core.virtualize
import lms.collection.mutable.{ArrayOps}
import lms.thirdparty._
import lantern.thirdparty._

trait GPUOps extends Base with ArrayOps {
  object NewGPUArray {
    // Allocate an array of the specified size on the GPU.
    def apply[T: Manifest](scalarCount: Rep[Int]): Rep[Array[T]] = gpu_array_new(scalarCount)
  }
  object GPUArray {
    // Initialize an array with the specified elements on the GPU.
    def apply[T: Manifest](xs: Rep[T]*) = gpu_array_fromseq(xs)
  }
  def gpu_array_new[T: Manifest](scalarCount: Rep[Int]): Rep[Array[T]]
  def gpu_array_fromseq[T: Manifest](xs: Seq[Rep[T]]): Rep[Array[T]]
  // Copy an array from device to host.
  def gpu_array_copy_device_to_host[T: Manifest](src: Rep[Array[T]], dest: Rep[Array[T]], len: Rep[Int])(implicit pos: SourceContext): Rep[Unit]
  // Copy an array from host to device.
  def gpu_array_copy_host_to_device[T: Manifest](src: Rep[Array[T]], dest: Rep[Array[T]], len: Rep[Int])(implicit pos: SourceContext): Rep[Unit]
  // Copy an array from device to device.
  def gpu_array_copy_device_to_device[T: Manifest](src: Rep[Array[T]], dest: Rep[Array[T]], len: Rep[Int])(implicit pos: SourceContext): Rep[Unit]
}

trait GPUOpsExp extends Base with ArrayOpsExpOpt {
  object CopyDirection extends Enumeration {
    val HostToDevice = Value("cudaMemcpyHostToDevice")
    val DeviceToHost = Value("cudaMemcpyDeviceToHost")
    val DeviceToDevice = Value("cudaMemcpyDeviceToDevice")
  }
  def gpu_array_new[T: Manifest](x: Exp[Int]) = Wrap[Array[T]](Adapter.g.reflectMutable("NewGpuArray", Unwrap(x)))
  def gpu_array_fromseq[T: Manifest](xs: Seq[Rep[T]]): Rep[Array[T]] = Wrap[Array[T]](Adapter.g.reflectMutable("new GPUArrayFromSeq["+manifest[T]+"]", Unwrap(xs)))
  def gpu_array_copy_device_to_host[T: Manifest](src: Rep[Array[T]], dest: Rep[Array[T]], len: Rep[Int])(implicit pos: SourceContext): Rep[Unit] =
    Wrap[Unit](Adapter.g.reflectEffect("d2hCopy["+manifest[T]+"]", Unwrap(src), Unwrap(dest), Unwrap(len))(Unwrap(src))(Unwrap(dest)))
    // reflectEffect(GPUArrayCopy(src, dest, len, CopyDirection.DeviceToHost))
  def gpu_array_copy_host_to_device[T: Manifest](src: Rep[Array[T]], dest: Rep[Array[T]], len: Rep[Int])(implicit pos: SourceContext): Rep[Unit] =
    // Wrap[Unit](Adapter.g.reflectMutable("h2dCopy["+manifest[T]+"]", Unwrap(src), Unwrap(dest), Unwrap(len)))
    Wrap[Unit](Adapter.g.reflectEffect("h2dCopy["+manifest[T]+"]", Unwrap(src), Unwrap(dest), Unwrap(len))(Unwrap(src))(Unwrap(dest)))
  def gpu_array_copy_device_to_device[T: Manifest](src: Rep[Array[T]], dest: Rep[Array[T]], len: Rep[Int])(implicit pos: SourceContext): Rep[Unit] =
    Wrap[Unit](Adapter.g.reflectEffect("d2dCopy["+manifest[T]+"]", Unwrap(src), Unwrap(dest), Unwrap(len))(Unwrap(src))(Unwrap(dest)))
    // reflectEffect(GPUArrayCopy(src, dest, len, CopyDirection.DeviceToDevice))
}

trait CudaGenGPUOps extends CGenBase {
  val IR: GPUOpsExp
  import IR._

  // Allocate GPU memory.
  def getCudaMallocString(buffer: String, count: String, dataType: String): String =
    "CUDA_CALL(cudaMalloc((void **)&" + buffer + ", " + count + " * sizeof(" + dataType + ")))"

  // Allocate GPU memory from memory arena.
  def getCudaMallocArenaString(count: String, dataType: String): String =
    "(" + dataType + "*)myGpuMalloc(" + count + " * sizeof(" + dataType + "))"

  // Copy an array in the specified direction.
  def cudaMemcpy(dest: String, src: String, count: String, dataType: String, direction: CopyDirection.Value): String =
    s"CUDA_CALL(cudaMemcpy($dest, $src, $count * sizeof($dataType), ${direction.toString}));"

  // Allocate unified memory, accessible by CPU and GPU.

  // Allocate unified memory, accessible by CPU and GPU.
  // FIXME: I encountered "bus error" when performing CPU ops on managed memory:
  //     Thread 1 "snippet" received signal SIGBUS, Bus error.
  //     Snippet (x0=<optimized out>) at snippet.cpp:144
  //     144  float x32 = x30 - x31;
  // I wonder if others can replicate this issue.
  def getCudaMallocManagedString(buffer: String, count: String, dataType: String): String =
  "CUDA_CALL(cudaMallocManaged((void **)&" + buffer + ", " + count + " * sizeof(" + dataType + ")));"

}

trait TensorDslCublas extends TensorDslCPU with GPUOpsExp with CLibs with CuBLASOps {

  def getCudaMallocAddr(): Rep[Long] = {
    unchecked[Long]("(long)gpuMallocAddr")
  }

  def resetCudaMallocAddr(addr: Rep[Long]) = {
    unchecked[Unit]("cudaMemset((void*)", addr, ", 0, ", getCudaMallocAddr() - addr, ")")
    unchecked[Unit]("gpuMallocAddr = (void*)", addr)
  }

  // NOTE: `cudaMemset` is not very useful because it only works with an integer array/value.
  protected def cudaMemset(array: Rep[Array[Int]], value: Rep[Int], n: Int): Rep[Unit] =
    unchecked[Unit]("CUDA_CALL(cudaMemset((void **)&", array, ", ", value, ", ", n, " * sizeof(int)))")

  protected def cublasSetPointerModeDevice(): Rep[Unit] =
    unchecked[Unit]("cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE)")

  protected def cublasSetPointerModeHost(): Rep[Unit] =
    unchecked[Unit]("cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST)")

  class ArrayTransferOps[T: Manifest](array: Rep[Array[T]]) {
    // Get a CPU-allocated copy of this array.
    def toCPU(length: Rep[Int]): Rep[Array[T]] = {
      val res = BackendCPU().mallocArray[T](length)
      gpu_array_copy_device_to_host(array, res, length)
      res
    }

    // Get a GPU-allocated copy of this array.
    def toGPU(length: Rep[Int]): Rep[Array[T]] = {
      val res = BackendGPU.mallocArray[T](length)
      gpu_array_copy_host_to_device(array, res, length)
      res
    }

    // Move the underlying data of this array to the CPU.
    def moveToCPU(length: Rep[Int]): Unit = {
      val res = BackendCPU().mallocArray[T](length)
      gpu_array_copy_device_to_host(array, res, length)
      unchecked[Unit](array, " = ", res)
    }

    // Move the underlying data of this array to the GPU.
    def moveToGPU(length: Rep[Int]): Unit = {
      val res = BackendGPU.mallocArray[T](length)
      gpu_array_copy_host_to_device(array, res, length)
      unchecked[Unit](array, " = ", res)
    }
  }
  implicit def arrayToTransferOps[T: Manifest](array: Rep[Array[T]]) = new ArrayTransferOps(array)

  // Tensor backend transfer operations.
  class TensorTransferOps(t: Tensor) {
    // Get a CPU-allocated copy of this tensor.
    def toCPU(): Tensor = {
      generate_comment("Tensor 'toCPU' invocation.")
      new Tensor(t.data.toCPU(t.scalarCount), t.shape)
    }

    // Get a GPU-allocated copy of this tensor.
    def toGPU(): Tensor = {
      generate_comment("Tensor 'toGPU' invocation.")
      // val res = BackendGPU.mallocArray[Float](t.scalarCount)
      new Tensor(t.data.toGPU(t.scalarCount), t.shape)
    }

    // Move the underlying data of this tensor to the CPU.
    def moveToCPU(): Unit = {
      generate_comment("Tensor 'moveToCPU' invocation.")
      t.data.moveToCPU(t.scalarCount)
    }

    // Move the underlying data of this tensor to the GPU.
    def moveToGPU(): Unit = {
      generate_comment("Tensor 'moveToGPU' invocation.")
      t.data.moveToGPU(t.scalarCount)
    }
  }
  implicit def tensorToTransferOps(t: Tensor) = new TensorTransferOps(t)

  class TensorRTransferOps(t: TensorR) {
    def toCPU(): TensorR = new TensorR(t.x.toCPU(), t.d.toCPU())
    def toGPU(): TensorR = {
      val temp = new TensorR(t.x.toGPU(), t.d.toGPU())
      temp.isInput = t.isInput
      temp
    }
    def moveToCPU(): Unit = { t.x.moveToCPU(); t.d.moveToCPU() }
    def moveToGPU(): Unit = { t.x.moveToGPU(); t.d.moveToGPU() }
  }
  implicit def tensorRToTransferOps(t: TensorR) = new TensorRTransferOps(t)


  /**
    * cuBLAS tensor operation backend. WIP.
    */
  class BackendCublas protected() extends Backend {
    override def setup(): Unit = {
      cublasCall(cublasCreate(cublasHandle))
      unchecked(
        """
        |CUDA_CALL(cudaMalloc(&gpuMallocBase, HEAP_SIZE));
        |CUDA_CALL(cudaMemset(gpuMallocBase, 0, HEAP_SIZE));
        |gpuMallocAddr = gpuMallocBase
        """.stripMargin)
    }

    override def cleanup(): Unit = {
      cublasCall(cublasDestroy(cublasHandle))
      unchecked("CUDA_CALL(cudaFree(gpuMallocBase))")
    }

    override def mallocArray[T: Manifest](length: Rep[Int]): Rep[Array[T]] = gpu_array_new[T](length)

    override def copyFloatArray(dest: Rep[Array[Float]], src: Rep[Array[Float]], length: Rep[Int]): Unit =
      gpu_array_copy_device_to_device(src, dest, length)

    override def arrayToTensor(array: Rep[Array[Float]], dims: Rep[Int]*): Tensor = new Tensor(array, dims)

    override def makeTensor(dims: Seq[Rep[Int]], scalars: Float*): Tensor =
      BackendCPU().makeTensor(dims, scalars: _*).toGPU()

    override def fill(dims: Seq[Rep[Int]], value: Rep[Float]): Tensor = {
      val size: Rep[Int] = dims.foldLeft(unit(1)){case (a, b) => a * b}
      val resArray = mallocArray[Float](size)
      cudaArrayFill_(resArray, value, size)
      Tensor(resArray, dims: _*)
    }

    override def fillWithBias(dims: Seq[Rep[Int]], bias: Tensor, dim: Int): Tensor =
      BackendCPU().fillWithBias(dims, bias.toCPU(), dim).toGPU()

    override def fillInPlace(x: Tensor, value: Rep[Float]): Unit = {
      val size = x.scalarCount
      cudaArrayFill_(x.data, value, size)
    }

    // TODO: Implement random initialization using cuRAND API.
    override def randinit(dims: Seq[Int], scale: Float = 1.0f, seed: Option[Int] = None): Tensor =
      BackendCPU().randinit(dims, scale, seed).toGPU()

    override def clipAt(x: Tensor, bound: Float) = cudaArrayClipAt_(x.data, bound, x.scalarCount)

    // Cannot implement (Need kernel functions!)
    override def mutate(x: Tensor, delta: Rep[Int] => Rep[Float]): Unit = ???
    override def mapInPlace(x: Tensor, op: Rep[Float] => Rep[Float]): Unit = ???
    override def changeTo(x: Tensor, gen: Rep[Int] => Rep[Float]): Unit = ???
    override def map(x: Tensor, op: Rep[Float] => Rep[Float]): Tensor = ???
    override def fold(init: Rep[Float])(x: Tensor, op: (Rep[Float], Rep[Float]) => Rep[Float]): Rep[Float] = ???

    // Reference: https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-dot
    // NOTE: `sdot` fails when the cuBLAS pointer mode is host (as opposed to device).
    // Investigate performance impact.
    def sdot(n: Rep[Int], a: Rep[Array[Float]], b: Rep[Array[Float]], result: Rep[Array[Float]]) = {
      cublasCall(cublasSdot_(cublasHandle, n, a, 1, b, 1, result))
    }

    override def vectorVectorDot(x: Tensor, y: Tensor): Tensor = {
      val res = BackendCPU().mallocArray[Float](1)
      sdot(x.scalarCount, x.data, y.data, res)
      Tensor(res, 1).toGPU()  // TODO (Fei Wang): if use GPU memory for result, there is segfault
    }

    // Reference: https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
    def sgemv(m: Rep[Int], n: Rep[Int], matrix: Rep[Array[Float]], vector: Rep[Array[Float]], result: Rep[Array[Float]]) = {
      cublasCall(cublasSgemv_(cublasHandle, cublasOpT, n, m, one, matrix, n, vector, 1, zero, result, 1))
    }

    override def matrixVectorDot(x: Tensor, y: Tensor): Tensor = {
      val m = x.shape(0)
      val n = x.shape(1)
      val res = mallocArray[Float](m)
      sgemv(m, n, x.data, y.data, res)
      Tensor(res, m)
    }

    // Reference: https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
    @virtualize
    def sgemm(m: Rep[Int], n: Rep[Int], k: Rep[Int], a: Rep[Array[Float]], b: Rep[Array[Float]], result: Rep[Array[Float]]) = {
      cublasCall(cublasSgemm_(cublasHandle, cublasOpN, cublasOpN, n, m, k, one, b, n, a, k, zero, result, n))
    }

    override def matrixMatrixDot(x: Tensor, y: Tensor): Tensor = {
      val m = x.shape(0)
      val n = y.shape(1)
      val k = y.shape(0)
      val res = mallocArray[Float](m * n)
      sgemm(m, n, k, x.data, y.data, res)
      Tensor(res, m, n)
    }

    private def stride(t: Tensor) = t.shape(1) * t.shape(2)

    private def add_dotTrans2Batched(res: Tensor, a: Tensor, b: Tensor) = {
      // handle column-major (passed in are row-major)
      val m = a.shape(1) // dim1
      val k = a.shape(2) // dim3
      val n = b.shape(1) // dim2
      val batchCount = a.shape(0)
      cublasCall(cublasSgemmStridedBatched_(cublasHandle, cublasOpT, cublasOpN, n, m, k, one, b.data, k, stride(b), a.data, k,
        stride(a), one, res.data, n, stride(res), batchCount))
    }
    private def add_dotTrans1Batched(res: Tensor, a: Tensor, b: Tensor) = {
      val m = a.shape(2) // dim2
      val k = a.shape(1) // dim1
      val n = b.shape(2) // dim3
      val batchCount = res.shape(0)
      cublasCall(cublasSgemmStridedBatched_(cublasHandle, cublasOpN, cublasOpT, n, m, k, one, b.data, n, stride(b),
        a.data, m, stride(a), one, res.data, n, stride(res), batchCount))
    }

    private def bmm_sgemmbatched(x: Tensor, y: Tensor): Tensor = {
      val m = x.shape(1)
      val k = x.shape(2)
      val n = y.shape(2)
      val batchCount = x.shape(0)

      val res = mallocArray[Float](batchCount * m * n)
      cublasCall(cublasSgemmStridedBatched_(cublasHandle, cublasOpN, cublasOpN, n, m, k, one, y.data, n, stride(y), x.data,
        k, stride(x), zero, res, n, m * n, batchCount))
      Tensor(res, batchCount, m, n)
    }

    private def bmm_grad_sgemmbatched(x: TensorR, y: TensorR, output: TensorR) = {
      generate_comment("backprop of batched matrix multiplication")
      if (!x.isInput) add_dotTrans2Batched(x.d, output.d, y.x)
      if (!y.isInput) add_dotTrans1Batched(y.d, x.x, output.d)
    }

    override def bmm(x: Tensor, y: Tensor): Tensor = {
      bmm_sgemmbatched(x, y)
//      val batchSize = x.shape(0)
//      val m = x.shape(1)
//      val k = x.shape(2)
//      val n = y.shape(2)
//      val res = mallocArray[Float](batchSize * m * n)
//
//      for(i <- 0 until batchSize) {
//        sgemm(m, n, k, x(i).data, y(i).data, res.slice(i * m * n, (i + 1) * m * n))
//      }
//
//      Tensor(res, batchSize, m, n)
    }

    override def bmm_grad(x: TensorR, y: TensorR, output: TensorR) = {
      bmm_grad_sgemmbatched(x, y, output)
//      val batchSize = x.x.shape(0)
//      for(i <- 0 until batchSize) {
//        if (!x.isInput) add_dotTrans2(x.d(i), output.d(i), y.x(i))
//        if (!y.isInput) add_dotTrans1(y.d(i), x.x(i), output.d(i))
//      }
    }

    @virtualize
    override def dot_grad(x: TensorR, y: TensorR, output: TensorR): Unit = {
      (x.x.rank, y.x.rank) match {
        case (1, 1) =>
          val dim = x.x.shape(0)
          // val scale = output.d.toCPU()  // TODO (Fei Wang) fix this for optimization
          var scale = output.d.toCPU().data(0)
          if (!x.isInput) cublasCall(cublasSgeam_(cublasHandle, cublasOpN, cublasOpN, dim, 1, one, x.d.data, dim, scale,
                                     y.x.data, dim, x.d.data, dim))
          if (!y.isInput) cublasCall(cublasSgeam_(cublasHandle, cublasOpN, cublasOpN, dim, 1, one, y.d.data, dim, scale,
                                     x.x.data, dim, y.d.data, dim))
        case (2, 1) =>
          if (!x.isInput) add_cartesian(x.d, y.x, output.d)
          if (!y.isInput) add_composition(y.d, x.x, output.d)
        case (2, 2) =>
          generate_comment("backprop of matrix-matrix-dot")
          if (!x.isInput) add_dotTrans2(x.d, output.d, y.x)
          if (!y.isInput) add_dotTrans1(y.d, x.x, output.d)
      }
    }
    @virtualize
    override def add_cartesian(x: Tensor, y: Tensor, output: Tensor): Unit = {
      val dim1 = x.shape(0); val dim2 = x.shape(1)
      cublasCall(cublasSgemm_(cublasHandle, cublasOpN, cublasOpN, dim2, dim1, 1, one, y.data, dim2, output.data, 1, one, x.data, dim2))
    }

    @virtualize
    override def add_composition(x: Tensor, y: Tensor, output: Tensor): Unit = {
      val dim1 = y.shape(0); val dim2 = y.shape(1)
      cublasCall(cublasSgemv_(cublasHandle, cublasOpN, dim2, dim1, one, y.data, dim2, output.data, 1, one, x.data, 1))
    }
    // more complication because cublas requires column-major
    @virtualize
    override def add_dotTrans1(x: Tensor, y: Tensor, output: Tensor): Unit = {
      val dim1 = y.shape(0); val dim2 = y.shape(1); val dim3 = output.shape(1)
      cublasCall(cublasSgemm_(cublasHandle, cublasOpN, cublasOpT, dim3, dim2, dim1, one, output.data, dim3, y.data, dim2,
                 one, x.data, dim3))
    }
    // more complication because cublas requires column-major
    @virtualize
    override def add_dotTrans2(x: Tensor, y: Tensor, output: Tensor): Unit = {
      val dim1 = x.shape(0); val dim2 = x.shape(1); val dim3 = output.shape(1)
      cublasCall(cublasSgemm_(cublasHandle, cublasOpT, cublasOpN, dim2, dim1, dim3, one, output.data, dim3, y.data, dim3,
                 one, x.data, dim2))
    }

    def elementWiseWithBroadCast(in1: Tensor, in2: Tensor, op: String): (Tensor, Dimensions, Dimensions) = {
      Tensor.dimBroadcast(in1.shape, in2.shape) match {
        case Some((xShape, yShape, resShape)) => {
          val resData = mallocArray[Float](resShape.scalarCount)
          val res = new Tensor(resData, resShape)
          val xStridesShadow = (xShape.strides zip xShape.dims) map {case (a, b) => if (b == unit(1)) unit(0) else a}
          val yStridesShadow = (yShape.strides zip yShape.dims) map {case (a, b) => if (b == unit(1)) unit(0) else a}
          resShape.dims.size match {
            case 1 => elementWiseWithBroadCastRank1_(
              op, res.scalarCount,
              in1.data, xStridesShadow(0),
              in2.data, yStridesShadow(0),
              resData, resShape.strides(0))
            case 2 => elementWiseWithBroadCastRank2_(
              op, res.scalarCount,
              in1.data, xStridesShadow(0), xStridesShadow(1),
              in2.data, yStridesShadow(0), yStridesShadow(1),
              resData, resShape.strides(0), resShape.strides(1))
            case 3 => elementWiseWithBroadCastRank3_(
              op, res.scalarCount,
              in1.data, xStridesShadow(0), xStridesShadow(1), xStridesShadow(2),
              in2.data, yStridesShadow(0), yStridesShadow(1), yStridesShadow(2),
              resData, resShape.strides(0), resShape.strides(1), resShape.strides(2))
            case 4 => elementWiseWithBroadCastRank4_(
              op, res.scalarCount,
              in1.data, xStridesShadow(0), xStridesShadow(1), xStridesShadow(2), xStridesShadow(3),
              in2.data, yStridesShadow(0), yStridesShadow(1), yStridesShadow(2), yStridesShadow(3),
              resData, resShape.strides(0), resShape.strides(1), resShape.strides(2), resShape.strides(3))
            case _ => assert(false, s"elementWiseWithBroadCast only handle tensors with rank no larger than 4, got ${resShape.dims.size}")
          }
          (res, xShape, yShape)
        }
        case _ => ???
      }
    }

    def elementWiseWithBroadCastUpdate(in1: Tensor, in2: Tensor, op: String): Unit = {
      Tensor.dimBroadcast(in1.shape, in2.shape) match {
        case Some((xShape, yShape, resShape)) => {
          assertC(!xShape.broadcasted, "in elementwise update, the xShape cannot be broadcasted!")
          val xStridesShadow = (xShape.strides zip xShape.dims) map {case (a, b) => if (b == unit(1)) unit(0) else a}
          val yStridesShadow = (yShape.strides zip yShape.dims) map {case (a, b) => if (b == unit(1)) unit(0) else a}
          resShape.dims.size match {
            case 1 => elementWiseWithBroadCastRank1_(
              op, in1.scalarCount,
              in1.data, xStridesShadow(0),
              in2.data, yStridesShadow(0),
              in1.data, xStridesShadow(0))
            case 2 => elementWiseWithBroadCastRank2_(
              op, in1.scalarCount,
              in1.data, xStridesShadow(0), xStridesShadow(1),
              in2.data, yStridesShadow(0), yStridesShadow(1),
              in1.data, xStridesShadow(0), xStridesShadow(1))
            case 3 => elementWiseWithBroadCastRank3_(
              op, in1.scalarCount,
              in1.data, xStridesShadow(0), xStridesShadow(1), xStridesShadow(2),
              in2.data, yStridesShadow(0), yStridesShadow(1), yStridesShadow(2),
              in1.data, xStridesShadow(0), xStridesShadow(1), xStridesShadow(2))
            case 4 => elementWiseWithBroadCastRank4_(
              op, in1.scalarCount,
              in1.data, xStridesShadow(0), xStridesShadow(1), xStridesShadow(2), xStridesShadow(3),
              in2.data, yStridesShadow(0), yStridesShadow(1), yStridesShadow(2), yStridesShadow(3),
              in1.data, xStridesShadow(0), xStridesShadow(1), xStridesShadow(2), xStridesShadow(3))
            case _ => assert(false, s"elementWiseWithBroadCastUpdate only handle tensors with rank no larger than 4, got ${resShape.dims.size}")
          }
        }
        case _ => ???
      }
    }

    def arithWithScalar(op: String, x: Tensor, y: Rep[Float]): Tensor = {
      val res = mallocArray[Float](x.scalarCount)
      arithScalar_(op, x.data, res, y, x.scalarCount)
      Tensor(res, x.shape: _*)
    }

    def arithWithScalarUpdate(op: String, x: Tensor, y: Rep[Float]): Unit = arithScalar_(op, x.data, x.data, y, x.scalarCount)

    override def +(x: Tensor, y: Rep[Float]): Tensor = arithWithScalar("+", x, y)
    override def +(x: Tensor, y: Tensor): (Tensor, Dimensions, Dimensions) = elementWiseWithBroadCast(x, y, "+")
    override def add_grad(x: TensorR, y: TensorR, output: TensorR, xShape: Dimensions, yShape: Dimensions): Unit = ???

    override def +=(x: Tensor, y: Rep[Float]): Unit = arithWithScalarUpdate("+", x, y)
    override def +=(x: Tensor, y: Tensor): Unit = elementWiseWithBroadCastUpdate(x, y, "+")

    override def -(x: Tensor, y: Rep[Float]): Tensor = arithWithScalar("-", x, y)
    override def -(x: Tensor, y: Tensor): (Tensor, Dimensions, Dimensions) = elementWiseWithBroadCast(x, y, "-")
    override def minus_grad(x: TensorR, y: TensorR, output: TensorR, xShape: Dimensions, yShape: Dimensions): Unit = ???

    override def -=(x: Tensor, y: Rep[Float]): Unit = arithWithScalarUpdate("-", x, y)
    override def -=(x: Tensor, y: Tensor): Unit = elementWiseWithBroadCastUpdate(x, y, "-")

    override def *(x: Tensor, y: Rep[Float]): Tensor = arithWithScalar("*", x, y)
    override def *(x: Tensor, y: Tensor): (Tensor, Dimensions, Dimensions) = elementWiseWithBroadCast(x, y, "*")
    override def mul_grad(x: TensorR, y: TensorR, output: TensorR, xShape: Dimensions, yShape: Dimensions): Unit = ???

    override def *=(x: Tensor, y: Rep[Float]): Unit = arithWithScalarUpdate("*", x, y)
    override def *=(x: Tensor, y: Tensor): Unit = elementWiseWithBroadCastUpdate(x, y, "*")

    override def /(x: Tensor, y: Rep[Float]): Tensor = arithWithScalar("/", x, y)
    override def /(x: Tensor, y: Tensor): (Tensor, Dimensions, Dimensions) = elementWiseWithBroadCast(x, y, "/")
    override def div_grad(x: TensorR, y: TensorR, output: TensorR, xShape: Dimensions, yShape: Dimensions): Unit = ???

    override def /=(x: Tensor, y: Rep[Float]): Unit = arithWithScalarUpdate("/", x, y)
    override def /=(x: Tensor, y: Tensor): Unit = elementWiseWithBroadCastUpdate(x, y, "/")

    override def plusBias(main: Tensor, bias: Tensor): Tensor = ???
    override def plusBias_grad(main: TensorR, bias: TensorR): Unit = ???

    override def plusEqual(base: Tensor, adder: Tensor): Tensor = ???
    override def plusEqual_grad(base: TensorR, adder: TensorR): Unit = ???

    override def geam(x: Tensor, transX: Boolean, alpha: Rep[Float], y: Tensor, transY: Boolean, beta: Rep[Float], output: Tensor): Unit = {
      val alpha1 = var_new(alpha)
      val beta1 = var_new(beta)
      (transX, transY) match {
        case (false, false) =>
          Tensor.assertShapeEqual(x.shape, y.shape)
          Tensor.assertShapeEqual(x.shape, output.shape)
          val m = x.shape(0)
          val n = x.shape.drop(1).product1
          cublasCall(cublasSgeam_(cublasHandle, cublasOpN, cublasOpN, n, m, alpha1, x.data, n, beta1, y.data, n, output.data, n))
          // unchecked[Unit](
          //   "CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ",
          //   n, ",", m, ",", alpha1, ",",
          //   x.data, ",", n, ",", beta1, ", ", y.data, ", ", n, ", ", output.data, ",", n, "))")
        case (false, true) =>
          assert(x.rank == 2 && y.rank == 2)
          assert(x.shape(0) == y.shape(1) && x.shape(1) == y.shape(0), "is this assertion correct in terms of types?")
          val m = x.shape(0)
          val n = x.shape(1)
          generate_comment("is error here?")
          cublasCall(cublasSgeam_(cublasHandle, cublasOpN, cublasOpT, n, m, alpha1, x.data, n, beta1, y.data, m, output.data, n))
          // unchecked[Unit](
          //   "CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ",
          //   n, ",", m, ",", alpha1, ",",
          //   x.data, ",", n, ",", beta1, ", ", y.data, ", ", m, ", ", output.data, ",", n, "))")
        case (true, false) =>
          assert(x.rank == 2 && y.rank == 2)
          assert(x.shape(0) == y.shape(1) && x.shape(1) == y.shape(0))
          val m = x.shape(1)
          val n = x.shape(0)
          cublasCall(cublasSgeam_(cublasHandle, cublasOpT, cublasOpN, n, m, alpha1, x.data, m, beta1, y.data, n, output.data, n))
          // unchecked[Unit](
          //   "CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, ",
          //   n, ",", m, ",", alpha1, ",",
          //   x.data, ",", m, ",", beta1, ", ", y.data, ", ", n, ", ", output.data, ",", n, "))")
        case (true, true) =>
          assert(x.rank == 2 && y.rank == 2)
          Tensor.assertShapeEqual(x.shape, y.shape)
          val m = x.shape(1)
          val n = x.shape(0)
          cublasCall(cublasSgeam_(cublasHandle, cublasOpT, cublasOpT, n, m, alpha1, x.data, m, beta1, y.data, m, output.data, n))
          // unchecked[Unit](
          //   "CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, ",
          //   n, ",", m, ",", alpha1, ",",
          //   x.data, ",", m, ",", beta1, ", ", y.data, ", ", m, ", ", output.data, ",", n, "))")
      }
    }

    override def trans(x: Tensor): Tensor = {
      assert(x.rank == 2, s"trans only supported for 2D matrix, got ${x.shape.seq}")
      val res = Tensor(mallocArray[Float](x.scalarCount), x.shape.reverse: _*)
      generate_comment("trans casted as geam call")
      this.geam(x, true, 1.0f, x, true, 0.0f, res)
      res
    }

    override def trans_grad(x: TensorR, y: TensorR): Unit = {
      assert(x.x.rank == 2 && y.x.rank == 2, s"rank has to be 2 for trans, got ${x.x.rank} ${y.x.rank}")
      Tensor.assertShapeEqual(x.x.shape.reverse, y.x.shape)
      this.geam(x.d, false, 1.0f, y.d, true, 1.0f, x.d)
    }

    // this helper function add x to resTensor after permutation via dims
    def permuteHelper(x: Tensor, resTensor: Tensor, dims: Int*): Unit = {

      assert(x.rank <= 4, s"TODO, only handle tensor with rank at most 4D for now")
      val order = ((0 until x.rank): Range)
      assert(dims != order && dims.sorted == order, s"dimensions should be permutation of ranks, got rank: ${x.rank}, dims: ${dims}")

      // generate specialized kernel functions if the kernel function is not in the Map already
      val TILE_DIM = 32;
      val BLOCK_ROWS = 8;
      /*
      if (!permutationKernelMap.contains(dims.toSeq)) {
        if (x.rank == 2) { // this is transpose
          // val kernel = s"""
          //   |__global__ void permute2D${nextKernel}(float *odata, const float *idata, int dimy, int dimx) {
          //   |
          //   |  __shared__ float tile[$TILE_DIM][$TILE_DIM+1];
          //   |  int x = blockIdx.x * $TILE_DIM + threadIdx.x;
          //   |  int y = blockIdx.y * $TILE_DIM + threadIdx.y;
          //   |  if (x < dimx)
          //   |    for (int j = 0; j < $TILE_DIM && j < dimy - y; j += $BLOCK_ROWS)
          //   |      tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*dimx + x];
          //   |  __syncthreads();
          //   |  x = blockIdx.y * $TILE_DIM + threadIdx.x;  // transpose block offset
          //   |  y = blockIdx.x * $TILE_DIM + threadIdx.y;
          //   |  if (x < dimy)
          //   |    for (int j = 0; j < $TILE_DIM && j < dimx-y; j += $BLOCK_ROWS)
          //   |      odata[(y+j)*dimy + x] += tile[threadIdx.x][threadIdx.y + j];
          //   |}
          //  """.stripMargin
          // val kernelName = s"permute2D${nextKernel}"
          // permutationKernelMap(dims.toSeq) = (kernel, kernelName)
          // nextKernel += 1
        } else if (x.rank == 3) { // this is permutation for 3D Tensor
          if (dims(2) == 2) { // this is the simple case, where the inner most dimension is not permutated
			      // val kernel = s"""
            //   |__global__ void permuteSim3D${nextKernel}(float* odata, const float* idata, int dim0, int dim1, int dim2) {
            //   |  int ioffset = blockIdx.y * dim1 * dim2 + blockIdx.x * dim2;
            //   |  int ooffset = blockIdx.x * dim0 * dim2 + blockIdx.y * dim2;
            //   |  for (int i = threadIdx.x; i < dim2; i += blockDim.x)
            //   |    odata[ooffset + i] += idata[ioffset + i];
            //   |}
            //  """.stripMargin
            // val kernelName = s"permuteSim3D${nextKernel}"
            // permutationKernelMap(dims.toSeq) = (kernel, kernelName)
            // nextKernel += 1
          } else { // this is the complicate case for 3D permutation (the inner most dimension is also permutated)
            // val kernel = s"""
            //  |__global__ void permute3D${nextKernel}(float *odata, const float *idata,
            //  |         int dim0, int dim1, int dim2,
            //  |         int istr0, int istr1, int ostr0, int ostr1) {
            //  |
            //  |  __shared__ float tile[$TILE_DIM][$TILE_DIM+1];
            //  |
            //  |  int x = blockIdx.x * $TILE_DIM + threadIdx.x;
            //  |  int y = blockIdx.y * $TILE_DIM + threadIdx.y;
            //  |  int z = blockIdx.z;
            //  |
            //  |  if (x < dim2)
            //  |    for (int j = 0; j < $TILE_DIM && j < ${if (dims(2) == 0) "dim0" else "dim1"} - y; j += $BLOCK_ROWS)
            //  |      tile[threadIdx.y+j][threadIdx.x] = idata[z*${if (dims(2) == 0) "istr1" else "istr0"} + (y+j)*${if (dims(2) == 0) "istr0" else "istr1"} + x];
            //  |
            //  |  __syncthreads();
            //  |
            //  |  x = blockIdx.y * $TILE_DIM + threadIdx.x;  // transpose block offset
            //  |  y = blockIdx.x * $TILE_DIM + threadIdx.y;
            //  |
            //  |  if (x < ${if (dims(2) == 0) "dim0" else "dim1"})
            //  |    for (int j = 0; j < $TILE_DIM && j < dim2-y; j += $BLOCK_ROWS)
            //  |      odata[(y+j)*${if (dims(0) == 2) "ostr0" else "ostr1"} + z*${if (dims(0) == 2) "ostr1" else "ostr0"} + x] += tile[threadIdx.x][threadIdx.y + j];
            //  |}
            //  """.stripMargin
            // val kernelName = s"permute3D${nextKernel}"
            // permutationKernelMap(dims.toSeq) = (kernel, kernelName)
            // nextKernel += 1
          }
        } else { // this is for 4D permutation
          if (dims(3) == 3) { // this is the simple case, where the last dimension is not permutated
            // val idxes = Seq("blockIdx.z", "blockIdx.y", "blockIdx.x")
            // val kernel = s"""
            //  |__global__ void permuteSim4D${nextKernel}(float* odata, const float* idata,
            //  |      int istr0, int istr1, int istr2,   // elide istr3/ostr3 because that is '1'
            //  |      int ostr0, int ostr1, int ostr2) { // actually ostr2 should be the same as istr2 (can remove)
            //  |
            //  |  int ioffset = ${idxes(0)} * istr0 + ${idxes(1)} * istr1 + ${idxes(2)} * istr2;
            //  |  int ooffset = ${idxes(dims(0))} * ostr0 + ${idxes(dims(1))} * ostr1 + ${idxes(dims(2))} * ostr2;
            //  |  for (int i = threadIdx.x; i < istr2; i += blockDim.x)
            //  |    odata[ooffset + i] += idata[ioffset + i];
            //  |}
            //  """.stripMargin
            // val kernelName = s"permuteSim4D${nextKernel}"
            // permutationKernelMap(dims.toSeq) = (kernel, kernelName)
            // nextKernel += 1
          } else { // this is the complicated case, where the last dimension is permutated

            // val setIOffsetBase = if (dims(3) == 0) {
            //   "int ioffsetBase = x + y * istr0 + blockIdx.z * istr1 + blockIdx.y * istr2;"
            // } else if (dims(3) == 1) {
            //   "int ioffsetBase = x + y * istr1 + blockIdx.z * istr0 + blockIdx.y * istr2;"
            // } else { // dims(3) must be 2 now
            //   "int ioffsetBase = x + y * istr2 + blockIdx.z * istr0 + blockIdx.y * istr1;"
            // }

            // val pos: Seq[Int] = Seq(0,1,2,3).map(dims.indexOf(_))
            // val ostrPos: Seq[String] = pos.map(Seq("ostr0", "ostr1", "ostr2", "1")(_))
            // val setOOffsetBase = if (dims(3) == 0) {
            //   s"int ooffsetBase = x + y * ${ostrPos(3)} + blockIdx.z * ${ostrPos(1)} + blockIdx.y * ${ostrPos(2)};"
            // } else if (dims(3) == 1) {
            //   s"int ooffsetBase = x + y * ${ostrPos(3)} + blockIdx.z * ${ostrPos(0)} + blockIdx.y * ${ostrPos(2)};"
            // } else { // dim(3) must be 2 now
            //   s"int ooffsetBase = x + y * ${ostrPos(3)} + blockIdx.z * ${ostrPos(0)} + blockIdx.y * ${ostrPos(1)};"
            // }

            // val kernel = s"""
            //   |__global__ void permute4D${nextKernel}(float *odata, const float *idata,
            //   |    int dim0, int dim1, int dim2, int dim_3,
            //   |    int istr0, int istr1, int istr2,
            //   |    int ostr0, int ostr1, int ostr2,
            //   |    int strideBIdxX) {
            //   |
            //   |  __shared__ float tile[$TILE_DIM][$TILE_DIM+1];
            //   |
            //   |  int blockIdxY = blockIdx.x / strideBIdxX;
            //   |  int blockIdxX = blockIdx.x - blockIdxY * strideBIdxX;
            //   |  int x = blockIdxX * $TILE_DIM + threadIdx.x;
            //   |  int y = blockIdxY * $TILE_DIM + threadIdx.y;
            //   |
            //   |  if (x < dim_3) {
            //   |    ${setIOffsetBase}
            //   |    for (int j = 0; j < $TILE_DIM && j < ${Seq("dim0", "dim1", "dim2")(dims(3))} - y; j += $BLOCK_ROWS)
            //   |      tile[threadIdx.y+j][threadIdx.x] = idata[ioffsetBase + j * ${Seq("istr0", "istr1", "istr2")(dims(3))}];
            //   |  }
            //   |  __syncthreads();
            //   |
            //   |  x = blockIdxY * $TILE_DIM + threadIdx.x;  // transpose block offset
            //   |  y = blockIdxX * $TILE_DIM + threadIdx.y;
            //   |
            //   |  if (x < ${Seq("dim0", "dim1", "dim2", "dim_3")(dims(3))}) {
            //   |    ${setOOffsetBase}
            //   |    for (int j = 0; j < $TILE_DIM && j < dim_3 - y; j += $BLOCK_ROWS)
            //   |      odata[ooffsetBase + j * ${ostrPos(3)}] += tile[threadIdx.x][threadIdx.y + j];
            //   |  }
            //   |}
            //  """.stripMargin
            // val kernelName = s"permute4D${nextKernel}"
            // permutationKernelMap(dims.toSeq) = (kernel, kernelName)
            // nextKernel += 1
          }
        }
      }
      */
      // end of if (permutationKernelMap.contains()), the following code should call the kernel function
      // val (_, kernelName) = permutationKernelMap(dims.toSeq)

      if (x.rank == 2) { // this is transpose
        val dimGrid = dim3( (x.shape(1) + TILE_DIM - 1) / TILE_DIM, (x.shape(0) + TILE_DIM - 1) / TILE_DIM, 1 )
        val dimBlock = dim3( TILE_DIM, BLOCK_ROWS, 1 )
        permute2D_(dimGrid, dimBlock, resTensor.data, x.data, x.shape(0), x.shape(1))
        // unchecked[Unit](
        //  "{\n",
        //  s"dim3 dimGrid((", x.shape(1), s"+$TILE_DIM-1)/$TILE_DIM, (", x.shape(0), s"+$TILE_DIM-1)/$TILE_DIM, 1);\n",
        //  s"dim3 dimBlock($TILE_DIM, $BLOCK_ROWS, 1);\n",
        //  s"permute2D<<<dimGrid, dimBlock>>>(", resTensor.data, ", ", x.data, ", ", x.shape(0), ", ", x.shape(1), ");\n",
        //  "}\n"
        // )
      } else if (x.rank == 3) { // this is permutation for 3D Tensor
        if (dims(2) == 2) { // this is the simple case (inner most dimension doesn't permute)
          val dimGrid = dim3(x.shape(1), x.shape(0), 1)
          val dimBlock = dim3(256, 1, 1)
          permute3DSim_(dimGrid, dimBlock, resTensor.data, x.data, x.shape(0), x.shape(1), x.shape(2))
          // unchecked[Unit](
          //   "{\n",
          //  s"dim3 dimGrid(", x.shape(1), ", ", x.shape(0), ", 1);\n",
          //  s"dim3 dimBlock(256, 1, 1);\n",
          //  s"permuteSim3D<<<dimGrid, dimBlock>>>(", resTensor.data, ", ", x.data, ", ", x.shape(0), ", ", x.shape(1), ", ", x.shape(2), ");\n",
          //   "}\n"
          // )
        } else { // this is the complicated case for 3D permutation
          val dimGrid = dim3( (x.shape(2) + TILE_DIM - 1) / TILE_DIM , (x.shape(dims(2)) + TILE_DIM - 1) / TILE_DIM, if (dims(2) == 0) x.shape(1) else x.shape(0) )
          val dimBlock = dim3( TILE_DIM, BLOCK_ROWS, 1 )
          (dims(2), dims(0)) match {
            case (0, 2) => permute3D210_(dimGrid, dimBlock, resTensor.data, x.data, x.shape(0), x.shape(1), x.shape(2),
              x.shape.strides(0), x.shape.strides(1), resTensor.shape.strides(0), resTensor.shape.strides(1))
            case (1, 2) => permute3D120_(dimGrid, dimBlock, resTensor.data, x.data, x.shape(0), x.shape(1), x.shape(2),
              x.shape.strides(0), x.shape.strides(1), resTensor.shape.strides(0), resTensor.shape.strides(1))
            case (0, 1) => permute3D201_(dimGrid, dimBlock, resTensor.data, x.data, x.shape(0), x.shape(1), x.shape(2),
              x.shape.strides(0), x.shape.strides(1), resTensor.shape.strides(0), resTensor.shape.strides(1))
            case (1, 0) => permute3D021_(dimGrid, dimBlock, resTensor.data, x.data, x.shape(0), x.shape(1), x.shape(2),
              x.shape.strides(0), x.shape.strides(1), resTensor.shape.strides(0), resTensor.shape.strides(1))
            case _ => ???
          }
          // unchecked[Unit](
          //   "{\n",
          //  s"dim3 dimGrid((", x.shape(2), s"+$TILE_DIM-1)/$TILE_DIM,(", x.shape(dims(2)), s"+$TILE_DIM-1)/$TILE_DIM,", if(dims(2)==0) x.shape(1) else x.shape(0), ");\n",
          //  s"dim3 dimBlock($TILE_DIM, $BLOCK_ROWS, 1);\n",
          //  s"$kernelName<<<dimGrid, dimBlock>>>(", resTensor.data, ", ", x.data, ", ", x.shape(0), ", ", x.shape(1), ", ", x.shape(2), ", ", x.shape.strides(0),
          //        ", ", x.shape.strides(1), ", ", resTensor.shape.strides(0), ", ", resTensor.shape.strides(1), ");\n",
          //   "}\n"
          // )
        }
      } else { // this is the permutation for 4D Tensor
        if (dims(3) == 3) { // this is the simple case for 4D Tensor (inner most dimension doesn't permute)
          val dimGrid = dim3(x.shape(2), x.shape(1), x.shape(0))
          val dimBlock = dim3(256, 1, 1)

          (dims(0), dims(1), dims(2)) match {
            case (0, 1, 2) => permute4D0123_(dimGrid, dimBlock, resTensor.data, x.data,
              x.shape.strides(0), x.shape.strides(1), x.shape.strides(2),
              resTensor.shape.strides(0), resTensor.shape.strides(1), resTensor.shape.strides(2))
            case (0, 2, 1) => permute4D0213_(dimGrid, dimBlock, resTensor.data, x.data,
              x.shape.strides(0), x.shape.strides(1), x.shape.strides(2),
              resTensor.shape.strides(0), resTensor.shape.strides(1), resTensor.shape.strides(2))
            case (1, 0, 2) => permute4D1023_(dimGrid, dimBlock, resTensor.data, x.data,
              x.shape.strides(0), x.shape.strides(1), x.shape.strides(2),
              resTensor.shape.strides(0), resTensor.shape.strides(1), resTensor.shape.strides(2))
            case (1, 2, 0) => permute4D1203_(dimGrid, dimBlock, resTensor.data, x.data,
              x.shape.strides(0), x.shape.strides(1), x.shape.strides(2),
              resTensor.shape.strides(0), resTensor.shape.strides(1), resTensor.shape.strides(2))
            case (2, 0, 1) => permute4D2013_(dimGrid, dimBlock, resTensor.data, x.data,
              x.shape.strides(0), x.shape.strides(1), x.shape.strides(2),
              resTensor.shape.strides(0), resTensor.shape.strides(1), resTensor.shape.strides(2))
            case (2, 1, 0) => permute4D2103_(dimGrid, dimBlock, resTensor.data, x.data,
              x.shape.strides(0), x.shape.strides(1), x.shape.strides(2),
              resTensor.shape.strides(0), resTensor.shape.strides(1), resTensor.shape.strides(2))
            case _ => ???
          }
          // val kernelName = "permuteSim4DSim021"
          // unchecked[Unit](
          //   "{\n",
          //  s"dim3 dimGrid(", x.shape(2), ", ", x.shape(1), ", ", x.shape(0), ");\n",
          //  s"dim3 dimBlock(256, 1, 1);\n",
          //  s"$kernelName<<<dimGrid, dimBlock>>>(", resTensor.data, ", ", x.data, ", ", x.shape.strides(0), ", ", x.shape.strides(1), ", ",
          //       x.shape.strides(2), ", ", resTensor.shape.strides(0), ", ", resTensor.shape.strides(1), ", ", resTensor.shape.strides(2), ");\n",
          //   "}\n"
          // )
        } else { // this is the complicated case for 4D Tensor
          System.out.println("Permutation kernel is too complex"); ???
          // unchecked[Unit](
          //   "{\n",
          //  s"int strideBIdxX = (", x.shape(3), s" + $TILE_DIM - 1)/$TILE_DIM;\n",
          //  s"int strideBIdxY = (", x.shape(dims(3)), s" + $TILE_DIM -1)/$TILE_DIM;\n",
          //  s"dim3 dimGrid(strideBIdxX * strideBIdxY, ", if (dims(3)==2) x.shape(1) else x.shape(2), ", ", if (dims(3)==0) x.shape(1) else x.shape(0), ");\n",
          //  s"dim3 dimBlock($TILE_DIM, $BLOCK_ROWS, 1);\n",
          //  s"$kernelName<<<dimGrid, dimBlock>>>(", resTensor.data, ", ", x.data, ", ", x.shape(0), ", ", x.shape(1), ", ", x.shape(2), ", ", x.shape(3), ", ",
          //        x.shape.strides(0), ", ", x.shape.strides(1), ", ", x.shape.strides(2), ", ",
          //        resTensor.shape.strides(0), ", ", resTensor.shape.strides(1), ", ", resTensor.shape.strides(2), ", strideBIdxX);\n",
          //   "}\n"
          // )
        }
      }
    }

    override def permute(x: Tensor, dims: Int*): Tensor = {
      val resTensor = Tensor(mallocArray[Float](x.scalarCount), dims.map(x.shape(_)): _*)
      permuteHelper(x, resTensor, dims: _*)
      resTensor
    }

    override def permute_grad(x: TensorR, y: TensorR, dims: Int*): Unit = {
      val revDims = ((0 until dims.length): Range).toSeq.map(dims.indexOf(_))
      permuteHelper(y.d, x.d, revDims:_*)
    }

    @virtualize
    override def gemm(x: Tensor, transX: Boolean, y: Tensor, transY: Boolean, alpha: Float): Tensor = {
      var alpha1 = alpha
      (transX, transY) match {
        case (false, false) =>
          val m = x.shape(0)
          val n = y.shape(1)
          val k = y.shape(0)
          val res = mallocArray[Float](m * n)
          cublasCall(cublasSgemm_(cublasHandle, cublasOpN, cublasOpN, n, m, k, alpha1, y.data, n, x.data, k, zero, res, n))
          Tensor(res, m, n)
        case (false, true) =>
          val m = x.shape(0)
          val n = y.shape(0)
          val k = y.shape(1)
          val res = mallocArray[Float](m * n)
          cublasCall(cublasSgemm_(cublasHandle, cublasOpT, cublasOpN, n, m, k, alpha1, y.data, k, x.data, k, zero, res, n))
          Tensor(res, m, n)
        case (true, false) =>
          val m = x.shape(1)
          val n = y.shape(1)
          val k = y.shape(0)
          val res = mallocArray[Float](m * n)
          cublasCall(cublasSgemm_(cublasHandle, cublasOpN, cublasOpT, n, m, k, alpha1, y.data, n, x.data, m, zero, res, n))
          Tensor(res, m, n)
        case (true, true) =>
          val m = x.shape(1)
          val n = y.shape(0)
          val k = y.shape(1)
          val res = mallocArray[Float](m * n)
          cublasCall(cublasSgemm_(cublasHandle, cublasOpT, cublasOpT, n, m, k, alpha1, y.data, k, x.data, m, zero, res, n))
          Tensor(res, m, n)
      }
    }

    @virtualize
    override def gemm_grad(x: TensorR, transX: Boolean, y: TensorR, transY: Boolean, alpha: Float, output: TensorR): Unit = {
      var alpha1 = alpha
      generate_comment("backprop of gemm")
      (transX, transY) match {
        case (false, false) =>
          val dim1 = x.x.shape(0); val dim2 = x.x.shape(1); val dim3 = y.x.shape(1)
          if (!x.isInput)
            cublasCall(cublasSgemm_(cublasHandle, cublasOpT, cublasOpN, dim2, dim1, dim3, alpha1, y.x.data, dim3, output.d.data,
                       dim3, one, x.d.data, dim2))
          if (!y.isInput)
            cublasCall(cublasSgemm_(cublasHandle, cublasOpN, cublasOpT, dim3, dim2, dim1, alpha1, output.d.data, dim3, x.x.data,
                       dim2, one, y.d.data, dim3))
        case (false, true) =>
          val dim1 = x.x.shape(0); val dim2 = x.x.shape(1); val dim3 = y.x.shape(0)
          if (!x.isInput)
            cublasCall(cublasSgemm_(cublasHandle, cublasOpN, cublasOpN, dim2, dim1, dim3, alpha1, y.x.data, dim2, output.d.data,
                       dim3, one, x.d.data, dim2))
          if (!y.isInput)
            cublasCall(cublasSgemm_(cublasHandle, cublasOpN, cublasOpT, dim2, dim3, dim1, alpha1, x.x.data, dim2, output.d.data,
                       dim3, one, y.d.data, dim2))
        case (true, false) =>
          val dim1 = x.x.shape(1); val dim2 = x.x.shape(0); val dim3 = y.x.shape(1)
          if (!x.isInput)
            cublasCall(cublasSgemm_(cublasHandle, cublasOpT, cublasOpN, dim1, dim2, dim3, alpha1, output.d.data,
                       dim3, y.x.data, dim3, one, x.d.data, dim1))
          if (!y.isInput)
            cublasCall(cublasSgemm_(cublasHandle, cublasOpN, cublasOpN, dim3, dim2, dim1, alpha1, output.d.data,
                       dim3, x.x.data, dim1, one, y.d.data, dim3))
        case (true, true) =>
          val dim1 = x.x.shape(1); val dim2 = x.x.shape(0); val dim3 = y.x.shape(0)
          if (!x.isInput)
            cublasCall(cublasSgemm_(cublasHandle, cublasOpT, cublasOpT, dim1, dim2, dim3, alpha1, output.d.data,
                       dim3, y.x.data, dim2, one, x.d.data, dim1))
          if (!y.isInput)
            cublasCall(cublasSgemm_(cublasHandle, cublasOpT, cublasOpT, dim2, dim3, dim1, alpha1, x.x.data, dim1,
                       output.d.data, dim3, one, y.d.data, dim2))
      }
    }

    override def conv2D_batch(input: Tensor, kernel: Tensor, bias: Option[Tensor], strides: Seq[Int], pads: Seq[Int]): (Tensor, Option[Tensor], Int) = ???
    override def conv2D_batch_grad(input: TensorR, finput: Option[TensorR], filter: TensorR, res: TensorR, bias: Option[TensorR] = None,
                                   padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int), counter: Int): Unit = ???
    override def conv2DTraining(input: TensorR, kernel: TensorR, bias: Option[TensorR], resShape: Seq[Rep[Int]], strides: Seq[Int], pads: Seq[Int]): TensorR@diff = ???

    override def maxPool2D_batch(input: Tensor, kernel: Seq[Int], strides: Seq[Int], pads: Option[Seq[Int]]): (Tensor, Option[Rep[Array[Int]]]) = ???
    override def maxPool2D_batch_grad(input: TensorR, output: TensorR, sidx: Option[Rep[Array[Int]]], kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Unit = ???

    override def averagePool2D_batch(input: Tensor, kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Tensor = ???
    override def averagePool2D_batch_grad(input: TensorR, output: TensorR, kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Unit = ???

    override def batchNormInference(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): Tensor = ???
    override def batchNormTraining(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): (Tensor, Option[Tensor], Option[Tensor], Int) = ???
    override def batchNorm_grad(input: TensorR, res: TensorR, scale: TensorR, bias: TensorR, saveMean: Option[Tensor], saveInvVariance: Option[Tensor], counterId: Int): Unit = ???

    override def batchNorm1DInference(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): Tensor = ???
    override def batchNorm1DTraining(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): (Tensor, Option[Tensor], Option[Tensor], Int) = ???
    override def batchNorm1D_grad(input: TensorR, res: TensorR, scale: TensorR, bias: TensorR, saveMean: Option[Tensor], saveInvVariance: Option[Tensor], counterId: Int): Unit = ???

    override def dropout(input: Tensor, prob: Float = 0.5f): (Tensor, Rep[Array[Float]], Rep[Int]) = ???
    override def dropout_grad(input: TensorR, output: TensorR, prob: Float, helper: Rep[Array[Float]], size: Rep[Int]): Unit = ???

    override def dropout_v2(input: Tensor, prob: Float = 0.5f): (Tensor, Rep[Array[Boolean]]) = {
      val res = mallocArray[Float](input.scalarCount)
      val mask = mallocArray[Boolean](input.scalarCount)
      dropout_(input.data, res, 1 - prob, mask, input.scalarCount, seed, offset)

      // update the offset based on how many random numbers are used by a thread
      // grid and block sizes are hardcoded in cublas.scala (<<<28, 512>>>)
      val gridSize = 28
      val blockSize = 512
      offset += (input.scalarCount / (blockSize * gridSize)) + 1

      (Tensor(res, input.shape :_*), mask)
    }

    override def dropout_v2_grad(input: TensorR, output: TensorR, prob: Float, mask: Rep[Array[Boolean]]): Unit = {
      dropoutGrad_(output.d.data, input.d.data, mask, input.d.scalarCount, 1/(1 - prob))
    }

    override def mask4D(input: Tensor, lengths: Rep[Array[Int]]): Tensor = {
      // inplace mask (input is of size Batch * c * d * Time, lengths are the actual length of each sequence in batch)
      // Note: We assume that lengths is passed to GPU already, at the beginning of each epoch
      assert(input.rank == 4, s"mask4D only deals with inputs of 4D, got ${input.rank}")
      mask4D_(input.data, lengths, input.shape.strides(0), input.shape.strides(1), input.shape.strides(2),
              input.shape.strides(3), input.scalarCount)
      // val nGrid = 28
      // // unchecked[Unit]("{\n__device__ int dims[4] = {", input.shape.strides(0), ", ", input.shape.strides(1), ", ", input.shape.strides(2), ", ", input.shape.strides(3), "}")
      // unchecked[Unit](s"mask4D<<<${nGrid}, 512>>>(", input.data, ", ", lengths, ", ", input.shape.strides(0), ", ", input.shape.strides(1), ", ",
      //                                                input.shape.strides(2), ", ", input.shape.strides(3), ", ", input.scalarCount, ")")
      input
    }

    override def maskedFill3D(input: Tensor, mask: Rep[Array[Int]], value: Rep[Float]): Tensor = {
      // Note - assumes mask is already in GPU
      assert(input.rank == 3, s"maskedFill3D only accepts 3D tensors, got ${input.rank}")
      assertC((input.shape.strides(0) > input.shape.strides(1)) && (input.shape.strides(1) > input.shape.strides(2)),
        s"maskedFill3D only accepts contiguous tensors")
      val res = mallocArray[Float](input.scalarCount)
      maskedFill3D_(input.data, res, mask, value, input.shape(0) * input.shape(1), input.scalarCount)
      Tensor(res, input.shape :_*)
    }

    override def maskedFill3D_grad(output: TensorR, x: TensorR, mask: Rep[Array[Int]], value: Rep[Float]): Unit = {
      maskedFill3DGrad_(output.d.data, x.d.data, mask, output.x.shape(0) * output.x.shape(1), output.x.scalarCount)
    }


    override def relu(x: Tensor, inPlace: Boolean = false): Tensor = ???
    override def tanh(x: Tensor): Tensor = ???
    override def sigmoid(x: Tensor): Tensor = ???
    override def relu_grad(input: TensorR, res: TensorR, inPlace: Boolean = false): Unit = ???
    override def tanh_grad(input: TensorR, res: TensorR): Unit = ???
    override def sigmoid_grad(input: TensorR, res: TensorR): Unit = ???

    override def softmax(x: Tensor, dim: Int = 1): Tensor = ???
    override def logSoftmax(x: Tensor, dim: Int = 1): Tensor = ???
    override def softmax_grad(input: TensorR, res: TensorR, dim: Int = 1): Unit = ???
    override def logSoftmax_grad(input: TensorR, res: TensorR, dim: Int = 1): Unit = ???

    @virtualize
    override def softmax_v2(x: Tensor, dim: Int = -1): Tensor = {
      assert(dim != -1 || dim != x.rank - 1, s"softmax_v2 is only implemented for last dim. Use softmax() instead.")
      val res = mallocArray[Float](x.scalarCount)
      val lastDimSize = x.shape.last
      val outerSize = x.scalarCount / lastDimSize

      if (lastDimSize <= 1024) {
        // launch the optimized kernel for <= 1024
        dispatch_softmax_forward_(res, x.data, lastDimSize, lastDimSize, outerSize)
      } else {
        softmax_(x.data, res, lastDimSize, outerSize)
      }

      Tensor(res, x.shape :_*)
    }

    @virtualize
    def softmax_v2_grad(input: TensorR, res: TensorR, dim: Int = -1): Unit = {
      val lastDimSize = input.x.shape.last
      val outerSize = input.x.scalarCount / lastDimSize

      if (lastDimSize <= 1024) {
        // launch the optimized kernel for <= 1024
        dispatch_softmax_backward(input.d.data, res.d.data, res.x.data, lastDimSize, lastDimSize, outerSize)
      } else {
        softmaxGrad_(input.d.data, res.d.data, res.x.data, lastDimSize, outerSize)
      }
    }


    override def hardTanh(x: Tensor, min_val: Float = -1.0f, max_val: Float = 1.0f, inPlace: Boolean = false): Tensor = {
      val size = x.scalarCount
      val res = if (inPlace) x.data else mallocArray[Float](size)
      hardTanh_(x.data, res, min_val, max_val, x.scalarCount)
      // val nGrid = 28
      // unchecked[Unit](s"hardTanh<<<${nGrid}, 512>>>(", x.data, ", ", res, ", ", min_val, ", ", max_val, ", ", x.scalarCount, ")")
      Tensor(res, x.shape.seq: _*)
    }

    override def hardTanh_grad(input: TensorR, res: TensorR, min_val: Float = -1.0f, max_val: Float = 1.0f, inPlace: Boolean = false): Unit =
      hardTanhGrad_(input.x.data, input.d.data, res.d.data, min_val, max_val, input.x.scalarCount, inPlace)
      // val size = input.x.scalarCount
      // val nGrid = 28
      // unchecked[Unit](s"hardTanh_grad<<<${nGrid}, 512>>>(", input.x.data, ", ", input.d.data, ", ", res.d.data, ", ", min_val, ", ", max_val, ", ", size, ", ", inPlace, ")")
    // }

    override def exp(x: Tensor) = elementwiseOpNoBroadcast(x, ElementWiseNoBroadCastOpt.Exp)
    override def exp_grad(x: TensorR, y: TensorR): Unit = elementwiseOpNoBroadcastGrad(x, y, ElementWiseNoBroadCastOpt.ExpGrad)

    override def log(x: Tensor) = elementwiseOpNoBroadcast(x, ElementWiseNoBroadCastOpt.Log)
    override def log_grad(x: TensorR, y: TensorR): Unit = elementwiseOpNoBroadcastGrad(x, y, ElementWiseNoBroadCastOpt.LogGrad)

    override def sqrt(x: Tensor) = elementwiseOpNoBroadcast(x, ElementWiseNoBroadCastOpt.Sqrt)
    override def sqrt_grad(x: TensorR, y: TensorR): Unit = elementwiseOpNoBroadcastGrad(x, y, ElementWiseNoBroadCastOpt.SqrtGrad)

    override def square(x: Tensor) = elementwiseOpNoBroadcast(x, ElementWiseNoBroadCastOpt.Square)
    override def square_grad(x: TensorR, y: TensorR): Unit = elementwiseOpNoBroadcastGrad(x, y, ElementWiseNoBroadCastOpt.SquareGrad)

    object ElementWiseNoBroadCastOpt extends Enumeration {
      val Log = Value("LOG")
      val LogGrad = Value("LOG_GRAD")
      val Exp = Value("EXP")
      val ExpGrad = Value("EXP_GRAD")
      val Sqrt = Value("SQRT")
      val SqrtGrad = Value("SQRT_GRAD")
      val Square = Value("SQUARE")
      val SquareGrad = Value("SQUARE_GRAD")
    }

    def elementwiseOpNoBroadcast(input: Tensor, op: ElementWiseNoBroadCastOpt.Value, inplace: Boolean = false): Tensor = {
      val numBlocks = 28 // (input.scalarCount + 511) / 512
      val res = if (inplace) input.data else mallocArray[Float](input.scalarCount)
      op match {
        case ElementWiseNoBroadCastOpt.Log => log_(input.data, res, input.scalarCount)
          // unchecked[Unit](s"elementwise_1D_1D_log<<<${numBlocks},", "512>>>(", input.data, ",", res, ", ", input.scalarCount, ")")
        case ElementWiseNoBroadCastOpt.Exp => exp_(input.data, res, input.scalarCount)
          // unchecked[Unit](s"elementwise_1D_1D_exp<<<${numBlocks},", "512>>>(", input.data, ",", res, ", ", input.scalarCount, ")")
        case ElementWiseNoBroadCastOpt.Sqrt => sqrt_(input.data, res, input.scalarCount)
          // unchecked[Unit](s"elementwise_1D_1D_sqrt<<<${numBlocks},", "512>>>(", input.data, ",", res, ", ", input.scalarCount, ")")
        case ElementWiseNoBroadCastOpt.Square => square_(input.data, res, input.scalarCount)
          // unchecked[Unit](s"elementwise_1D_1D_square<<<${numBlocks},", "512>>>(", input.data, ",", res, ", ", input.scalarCount, ")")
        case _ => ???
      }
      Tensor(res, input.shape: _*)
    }

    @virtualize
    def elementwiseOpNoBroadcastGrad(input: TensorR, output: TensorR, op: ElementWiseNoBroadCastOpt.Value): Unit = {
      val numBlocks = 28 // (input.x.scalarCount + 511) / 512
      op match {
        case ElementWiseNoBroadCastOpt.LogGrad => log_grad_(input.x.data, input.d.data, output.x.data, output.d.data, input.x.scalarCount)
          // unchecked[Unit](s"elementwise_1D_1D_log_grad<<<${numBlocks},", "512>>>(", input.x.data, ", ", input.d.data, ", ", output.x.data, ", ", output.d.data, ", ", input.x.scalarCount, ")")
        case ElementWiseNoBroadCastOpt.ExpGrad => exp_grad_(input.x.data, input.d.data, output.x.data, output.d.data, input.x.scalarCount)
          // unchecked[Unit](s"elementwise_1D_1D_exp_grad<<<${numBlocks},", "512>>>(", input.x.data, ", ", input.d.data, ", ", output.x.data, ", ", output.d.data, ", ", input.x.scalarCount, ")")
        case ElementWiseNoBroadCastOpt.SqrtGrad => sqrt_grad_(input.x.data, input.d.data, output.x.data, output.d.data, input.x.scalarCount)
          // unchecked[Unit](s"elementwise_1D_1D_sqrt_grad<<<${numBlocks},", "512>>>(", input.x.data, ", ", input.d.data, ", ", output.x.data, ", ", output.d.data, ", ", input.x.scalarCount, ")")
        case ElementWiseNoBroadCastOpt.SquareGrad => square_grad_(input.x.data, input.d.data, output.x.data, output.d.data, input.x.scalarCount)
          // unchecked[Unit](s"elementwise_1D_1D_square_grad<<<${numBlocks},", "512>>>(", input.x.data, ", ", input.d.data, ", ", output.x.data, ", ", output.d.data, ", ", input.x.scalarCount, ")")
        case _ => ???
      }
    }

    override def nllLoss(x: Tensor, target: Rep[Array[Int]]): Tensor = {
      assert(x.rank == 2, "Input must be a 2-D tensor")
      val batchSize = x.shape(0)
      val res = Tensor(mallocArray[Float](batchSize), batchSize)
      nllLoss_(x.data, target, res.data, batchSize, x.shape.strides(0))
      // unchecked[Unit]("nllLoss<<<", batchSize, ", 1>>>(", x.data, ", ", x.shape.strides(0), ", ", res.data, ", ", target, ")")
      res
    }

    override def nllLoss_grad(input: TensorR, res: TensorR, target: Rep[Array[Int]]): Unit =
      nllLossGrad_(input.d.data, target, res.d.data, input.d.shape(0), input.d.shape.strides(0))
      // unchecked[Unit]("nllLoss_grad<<<", input.d.shape(0), ", 1>>>(", input.d.shape.strides(0), ", ", res.d.data, ", ", target, ", ", input.d.data, ")")

    override def multiheadAttentionInit(embedDim: Int, numHeads: Int, kDim: Int, vDim: Int, bias: Boolean = false, dropOut: Float,
                                        residualConnection: Boolean, maxSeqLenQ: Rep[Int], maxSeqLenK: Rep[Int], maxBatchSize: Rep[Int],
                                        maxBeamSize: Rep[Int]): MultiheadAttnConfig = ???
    // multihead attention
    override def multiheadAttention(query: TensorR, key: TensorR, value: TensorR, weights: TensorR, attnMask: Boolean,
                                    config: MultiheadAttnConfig): (Tensor, MHAData) = ???

    override def multiheadAttention_grad(output: TensorR, query: TensorR, key: TensorR, value: TensorR, weights: TensorR,
                                         attnMask: Boolean, config: MultiheadAttnConfig, data: MHAData): Unit = ???


    // TODO - Supun - Haven't implemented mse for GPU
    override def mseLoss(x: Tensor, target: Rep[Array[Float]]): Tensor = ???
    override def mseLoss_grad(input: TensorR, res: TensorR, target: Rep[Array[Float]]): Unit = ???

    override def ctcLoss(prob: TensorR, inputLengths: Rep[Array[Int]], labels: Rep[Array[Int]], labelLengths: Rep[Array[Int]]): Tensor = ???

    override def sum(x: Tensor): Tensor = ??? // implemented using cudnn functions
    override def sum_grad(input: TensorR, res: TensorR): Unit = {
      generate_comment("backprop for sum op")
      assert(res.d.shape.dims == Seq(unit(1)), s"result of sum reduce should be scalar, got ${res.d.shape}")
      addScalarInArrayInPlace_(input.d.data, res.d.data, 1.0f, input.d.scalarCount)
      // unchecked[Unit](s"addScalarInArrayInPlace<<<28, 512>>>(", input.d.data, ", ", res.d.data, ", ", 1.0f, ", ", input.d.scalarCount, ")")
    }
    override def mean(x: Tensor): Tensor = ??? // implemented using cudnn functions
    override def mean_grad(input: TensorR, res: TensorR): Unit = {
      generate_comment("backprop for mean op")
      assert(res.d.shape.dims == Seq(unit(1)), s"result of mean reduce should be scalar, got ${res.d.shape}")
      addScalarInArrayInPlace_(input.d.data, res.d.data, 1.0f/input.x.scalarCount, input.d.scalarCount)
      // unchecked[Unit](s"addScalarInArrayInPlace<<<28, 512>>>(", input.d.data, ", ", res.d.data, ", ", 1.0f/input.x.scalarCount, ", ", input.d.scalarCount, ")")
    }
    override def sum(x: Tensor, dim: Int): Tensor = ???
    override def sum_grad(input: TensorR, res: TensorR, dim: Int): Unit = ???

    // TODO (Fei Wang): extend this to support 3D 2D 1D
    override def concat(dim: Int, tensors: Seq[Tensor]): Tensor = {
      assert(dim == 1, "TODO (Fei Wang): only support dim = 1 so far")
      assert(tensors.size == 2, "TODO: (Fei Wang): only support two tensor concatenation so far")
      assert(tensors(0).rank == 4 && tensors(1).rank == 4, "TODO: (Fei Wang): only support 4D concat so far")

      val dim0 = tensors(0).shape(0)
      val dim1 = tensors(0).shape(1) + tensors(1).shape(1)
      val dim2 = tensors(0).shape(2)
      val dim3_ = tensors(0).shape(3)
      val resShape = Seq(dim0, dim1, dim2, dim3_)
      val res = this.mallocArray[Float](resShape.product1)
      val resTensor = Tensor(res, dim0, dim1, dim2, dim3_)
      val sizeLow = dim2 * dim3_
      val sizeHigh = dim0
      val sizeDim1 = tensors(0).shape(1)
      val sizeDim2 = tensors(1).shape(1)

      val grid = dim3(28, 2)
      concat4D_(grid, tensors(0).data, sizeDim1, tensors(0).scalarCount,
                      tensors(1).data, sizeDim2, tensors(1).scalarCount,
                      res, 1, dim0, dim1, dim2, dim3_,
                      resTensor.shape.strides(0), resTensor.shape.strides(1), resTensor.shape.strides(2), resTensor.shape.strides(3))

      // val nGrid = 28 // tensors(0).scalarCount / 512 / 5 + 1
      // unchecked[Unit](
      //   "{\n",
      //   s"dim3 grid(${nGrid}, 2);\n",
      //   "concat2D_1D_greg<<<grid, 512>>>(", tensors(0).data, ", ", sizeDim1, ", ", tensors(0).scalarCount, ", ",
      //   tensors(1).data, ", ", sizeDim2, ", ", tensors(1).scalarCount, ", ",
      //   res, ", ", 1, ", ",
      //   dim0, ", ", dim1, ", ", dim2, ", ", dim3, ", ",
      //   resTensor.shape.strides(0), ", ", resTensor.shape.strides(1), ", ",resTensor.shape.strides(2), ", ",resTensor.shape.strides(3), ");\n",
      //   "}")
      resTensor
    }

    override def concat_grad(dim: Int, tensorRs: Seq[TensorR], output: TensorR): Unit = {
      assert(dim == 1, "TODO (Fei Wang): only support dim = 1 so far")
      assert(tensorRs.size == 2, "TODO: (Fei Wang): only support two tensor concatenation so far")
      assert(tensorRs(0).x.rank == 4 && tensorRs(1).x.rank == 4, "TODO: (Fei Wang): only support 4D concat so far")

      val dim0 = tensorRs(0).x.shape(0)
      val dim1 = tensorRs(0).x.shape(1) + tensorRs(1).x.shape(1)
      val dim2 = tensorRs(0).x.shape(2)
      val dim3_ = tensorRs(0).x.shape(3)
      val sizeLow = dim2 * dim3_
      val sizeHigh = dim0
      val sizeDim1 = tensorRs(0).x.shape(1)
      val sizeDim2 = tensorRs(1).x.shape(1)

      val grid = dim3(28, 2)
      concat4D_grad_(grid, tensorRs(0).d.data, sizeDim1, tensorRs(0).d.scalarCount,
                           tensorRs(1).d.data, sizeDim2, tensorRs(1).d.scalarCount,
                           output.d.data, 1,
                           dim0, dim1, dim2, dim3_,
                           output.d.shape.strides(0), output.d.shape.strides(1), output.d.shape.strides(2), output.d.shape.strides(3))

      // val nGrid = 28 //tensorRs(0).x.scalarCount / 512 / 5 + 1
      // unchecked[Unit](
      //   "{\n",
      //   s"dim3 grid(${nGrid}, 2);\n",
      //   "concat2D_1D_greg_grad<<<grid, 512>>>(", tensorRs(0).d.data, ", ", sizeDim1, ", ", tensorRs(0).d.scalarCount, ", ",
      //   tensorRs(1).d.data, ", ", sizeDim2, ", ", tensorRs(1).d.scalarCount, ", ",
      //   output.d.data, ", ", 1, ", ",
      //   dim0, ", ", dim1, ", ", dim2, ", ", dim3, ", ",
      //   output.d.shape.strides(0), ", ", output.d.shape.strides(1), ", ", output.d.shape.strides(2), ", ", output.d.shape.strides(3), ");\n",
      //   "}")
    }

    override def repeat0(in: Tensor, context: Int): Tensor = ???
    override def repeat0_grad(in: TensorR, out: TensorR, context: Int): Unit = ???

    override def adagrad_update(tr: TensorR, t: Tensor, learning_rate: Float, gradClip: Float, descent: Boolean): Unit = {
      assert(descent, s"TODO: only handle gradient descent (not ascent) so far")
      adagrad_(tr.x.data, tr.d.data, t.data, gradClip, learning_rate, t.scalarCount)
      // assert(tr.x.shape == t.shape, s"tensor and momentum should have the same shape, got ${tr.x.shape} and ${t.shape}")
      // val gridDimX = 28 // (t.scalarCount + 511) / 512
      // assert(gridDimX < 65535, s"gridDimX should not breach the limit, got ${gridDimX}")
      // unchecked[Unit](s"adagrad_update_1D_1D<<<${gridDimX}, 512>>>(", tr.x.data, ", ", tr.d.data, ", ", t.data, ", ", gradClip, ", ", learning_rate, ", ", t.scalarCount, ")")
    }
    override def momentum_update(tr: TensorR, t: Tensor, learning_rate: Float, momentum: Float, gradClip: Float, nesterov: Boolean, descent: Boolean) = {
      assert(descent, s"TODO: only handle gradient descent (not ascent) so far")
      momentum_(tr.x.data, tr.d.data, t.data, learning_rate, momentum, gradClip, nesterov, t.scalarCount)
      // val gridDimX = 28
      // unchecked[Unit](s"momentum_update_1D_1D<<<${gridDimX}, 512>>>(", tr.x.data, ", ", tr.d.data, ", ", t.data, ", ",
      // learning_rate, ", ", momentum, ", ", gradClip, ", ", nesterov, ", ", t.scalarCount, ")")
    }
  }

  object BackendCublas {
    def apply() = new BackendCublas
  }

  // Define default GPU backend.
  def BackendGPU: Backend = BackendCublas()
}

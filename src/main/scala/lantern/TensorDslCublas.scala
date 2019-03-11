package lantern

import scala.util.continuations._
import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._
import scala.virtualization.lms.common._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.{Map => MutableMap}
import scala.math._

trait TensorDslCublas extends TensorDslCPU with GPUOps {

  val permutationKernelMap = new scala.collection.mutable.HashMap[Seq[Int], (String, String)]()
  var nextKernel: Int = 0

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
      generateRawComment("Tensor 'toCPU' invocation.")
      new Tensor(t.data.toCPU(t.scalarCount), t.shape)
    }

    // Get a GPU-allocated copy of this tensor.
    def toGPU(): Tensor = {
      generateRawComment("Tensor 'toGPU' invocation.")
      // val res = BackendGPU.mallocArray[Float](t.scalarCount)
      new Tensor(t.data.toGPU(t.scalarCount), t.shape)
    }

    // Move the underlying data of this tensor to the CPU.
    def moveToCPU(): Unit = {
      generateRawComment("Tensor 'moveToCPU' invocation.")
      t.data.moveToCPU(t.scalarCount)
    }

    // Move the underlying data of this tensor to the GPU.
    def moveToGPU(): Unit = {
      generateRawComment("Tensor 'moveToGPU' invocation.")
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
    override def setup(): Unit = generateRawCode(
      """cublasHandle_t cublasHandle;
        |CUBLAS_CALL(cublasCreate(&cublasHandle));
        |CUDA_CALL(cudaMalloc(&gpuMallocBase, HEAP_SIZE));
        |CUDA_CALL(cudaMemset(gpuMallocBase, 0, HEAP_SIZE));
        |gpuMallocAddr = gpuMallocBase;
      """.stripMargin)

    override def cleanup(): Unit = generateRawCode(
      """CUBLAS_CALL(cublasDestroy(cublasHandle));
        |CUDA_CALL(cudaFree(gpuMallocBase));
      """.stripMargin)

    override def mallocArray[T: Manifest](length: Rep[Int]): Rep[Array[T]] = NewGPUArray[T](length)

    override def copyFloatArray(dest: Rep[Array[Float]], src: Rep[Array[Float]], length: Rep[Int]): Unit =
      gpu_array_copy_device_to_device(src, dest, length)

    override def arrayToTensor(array: Rep[Array[Float]], dims: Rep[Int]*): Tensor = new Tensor(array, dims)

    override def makeTensor(dims: Seq[Rep[Int]], scalars: Float*): Tensor =
      BackendCPU().makeTensor(dims, scalars: _*).toGPU()

    override def fill(dims: Seq[Rep[Int]], value: Rep[Float]): Tensor = {
      val size: Rep[Int] = dims.foldLeft(unit(1)){case (a, b) => a * b}
      val resArray = mallocArray[Float](size)
      val nGrid = 28
      unchecked[Unit](s"arrayFill<<<${nGrid}, 512>>>(", resArray, ", ", value, ", ", size, ")")
      Tensor(resArray, dims: _*)
    }

    override def fillWithBias(dims: Seq[Rep[Int]], bias: Tensor, dim: Int): Tensor =
      BackendCPU().fillWithBias(dims, bias.toCPU(), dim).toGPU()

    override def fillInPlace(x: Tensor, value: Rep[Float]): Unit = {
      val size = x.scalarCount
      val nGrid = 28
      unchecked[Unit](s"arrayFill<<<${nGrid}, 512>>>(", x.data, ", ", value, ", ", size, ")")
    }

    // TODO: Implement random initialization using cuRAND API.
    override def randinit(dims: Seq[Int], scale: Float = 1.0f, seed: Option[Int] = None): Tensor =
      BackendCPU().randinit(dims, scale, seed).toGPU()

    override def clipAt(x: Tensor, bound: Float) = {
      val size = x.scalarCount
      val nGrid = 28
      unchecked[Unit](s"clipAt<<<${nGrid}, 512>>>(", x.data, ", ", bound, ", ", size, ")")
    }

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
      generateRawComment("calling Sdot API function")
      unchecked[Unit]("CUBLAS_CALL(cublasSdot(cublasHandle, ", n, ",", a, ",", 1, ",", b, ",", 1, ",", result, "))")
    }

    override def vectorVectorDot(x: Tensor, y: Tensor): Tensor = {
      val res = BackendCPU().mallocArray[Float](1)
      generateRawComment("calling sdot from vectorVectorDot function")
      sdot(x.scalarCount, x.data, y.data, res)
      Tensor(res, 1).toGPU()  // TODO (Fei Wang): if use GPU memory for result, there is segfault
    }

    // Reference: https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
    def sgemv(m: Rep[Int], n: Rep[Int], matrix: Rep[Array[Float]], vector: Rep[Array[Float]], result: Rep[Array[Float]]) = {
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        "CUBLAS_CALL(cublasSgemv(cublasHandle, CUBLAS_OP_T, ",
        n, ",", m, ",", one, ",",
        matrix, ",", n, ",", vector, ",", 1, ",", zero, ",", result, ",", 1, "))")
    }

    override def matrixVectorDot(x: Tensor, y: Tensor): Tensor = {
      val m = x.shape(0)
      val n = x.shape(1)
      val res = mallocArray[Float](m)
      sgemv(m, n, x.data, y.data, res)
      Tensor(res, m)
    }

    // Reference: https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
    def sgemm(m: Rep[Int], n: Rep[Int], k: Rep[Int], a: Rep[Array[Float]], b: Rep[Array[Float]], result: Rep[Array[Float]]) = {
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ",
        n, ",", m, ",", k, ",", one, ",",
        b, ",", n, ",", a, ",", k, ",", zero, ",", result, ",", n, "))")
    }

    override def matrixMatrixDot(x: Tensor, y: Tensor): Tensor = {
      val m = x.shape(0)
      val n = y.shape(1)
      val k = y.shape(0)
      val res = mallocArray[Float](m * n)
      sgemm(m, n, k, x.data, y.data, res)
      Tensor(res, m, n)
    }

    override def dot_grad(x: TensorR, y: TensorR, output: TensorR): Unit = {
      // use CuBLAS instead
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      (x.x.rank, y.x.rank) match {
        case (1, 1) =>
          val dim = x.x.shape(0)
          val scale = output.d.toCPU()  // TODO (Fei Wang) fix this for optimization
          // x.d.addMul(output.d.data(0), y.x)
          if (!x.isInput) unchecked[Unit](
            "CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ",
            dim, ",", 1, ",", one, ",",
            x.d.data, ",", dim, ",", scale.data, ", ", y.x.data, ", ", dim, ", ", x.d.data, ",", dim, "))")
          // y.d.addMul(output.d.data(0), x.x)
          if (!y.isInput) unchecked[Unit](
            "CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ",
            dim, ",", 1, ",", one, ",",
            y.d.data, ",", dim, ",", scale.data, ", ", x.x.data, ", ", dim, ", ", y.d.data, ",", dim, "))")
        case (2, 1) =>
          if (!x.isInput) add_cartesian(x.d, y.x, output.d)
          if (!y.isInput) add_composition(y.d, x.x, output.d)
        case (2, 2) =>
          generateRawComment("backprop of matrix-matrix-dot")
          if (!x.isInput) add_dotTrans2(x.d, output.d, y.x)
          if (!y.isInput) add_dotTrans1(y.d, x.x, output.d)
      }
    }
    override def add_cartesian(x: Tensor, y: Tensor, output: Tensor): Unit = {
      val dim1 = x.shape(0); val dim2 = x.shape(1)
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ",
        dim2, ", ", dim1, ", ", 1, ", ", one, ", ",
        y.data, ", ", dim2, ", ", output.data, ", ", 1, ", ", one, ", ", x.data, ", ", dim2, "))")
    } 
    override def add_composition(x: Tensor, y: Tensor, output: Tensor): Unit = {
      val dim1 = y.shape(0); val dim2 = y.shape(1)
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        "CUBLAS_CALL(cublasSgemv(cublasHandle, CUBLAS_OP_N, ",
         dim2, ",", dim1, ",", one, ",",
         y.data, ",", dim2, ",", output.data, ",", 1, ",", one, ",", x.data, ",", 1, "))")
    }
    // more complication because cublas requires column-major
    override def add_dotTrans1(x: Tensor, y: Tensor, output: Tensor): Unit = {
      val dim1 = y.shape(0); val dim2 = y.shape(1); val dim3 = output.shape(1)
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ",
        dim3, ",", dim2, ",", dim1, ",", one, ",",
        output.data, ",", dim3, ",", y.data, ",", dim2, ",", one, ",", x.data, ",", dim3, "))")
    }
    // more complication because cublas requires column-major
    override def add_dotTrans2(x: Tensor, y: Tensor, output: Tensor): Unit = {
      val dim1 = x.shape(0); val dim2 = x.shape(1); val dim3 = output.shape(1)
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, ",
         dim2, ",", dim1, ",", dim3, ",", one, ",",
         output.data, ",", dim3, ",", y.data, ",", dim3, ",", one, ",", x.data, ",", dim2, "))")
    }

    override def +(x: Tensor, y: Rep[Float]): Tensor = ??? //elementwiseUnaryOp(x)(s => Seq(s + " + ", y))
    override def +(x: Tensor, y: Tensor): (Tensor, Dimensions, Dimensions) = ??? //elementwiseBinaryOp(x, y) { _ + " + " + _ }
    override def add_grad(x: TensorR, y: TensorR, output: TensorR, xShape: Dimensions, yShape: Dimensions): Unit = ???

    override def +=(x: Tensor, y: Rep[Float]): Unit = ??? //elementwiseInplaceUnaryOp(x)(s => Seq(s + " + ", y))
    override def +=(x: Tensor, y: Tensor): Unit = ??? //elementwiseInplaceBinaryOp(x, y) { _ + " + " + _ }

    override def -(x: Tensor, y: Rep[Float]): Tensor = ??? //elementwiseUnaryOp(x)(s => Seq(s + " - ", y))
    override def -(x: Tensor, y: Tensor): (Tensor, Dimensions, Dimensions) = ??? //elementwiseBinaryOp(x, y) { _ + " - " + _ }
    override def minus_grad(x: TensorR, y: TensorR, output: TensorR, xShape: Dimensions, yShape: Dimensions): Unit = ???

    override def -=(x: Tensor, y: Rep[Float]): Unit = ??? //elementwiseInplaceUnaryOp(x)(s => Seq(s + " - ", y))
    override def -=(x: Tensor, y: Tensor): Unit = ??? //elementwiseInplaceBinaryOp(x, y) { _ + " - " + _ }

    override def *(x: Tensor, y: Rep[Float]): Tensor = ??? //elementwiseUnaryOp(x)(s => Seq(s + " * ", y))
    override def *(x: Tensor, y: Tensor): (Tensor, Dimensions, Dimensions) = ??? //elementwiseBinaryOp(x, y) { _ + " * " + _ }
    override def mul_grad(x: TensorR, y: TensorR, output: TensorR, xShape: Dimensions, yShape: Dimensions): Unit = ???

    override def *=(x: Tensor, y: Rep[Float]): Unit = ??? //elementwiseInplaceUnaryOp(x)(s => Seq(s + " * ", y))
    override def *=(x: Tensor, y: Tensor): Unit = ??? //elementwiseInplaceBinaryOp(x, y) { _ + " * " + _ }

    override def /(x: Tensor, y: Rep[Float]): Tensor = ??? //elementwiseUnaryOp(x)(s => Seq(s + " / ", y))
    override def /(x: Tensor, y: Tensor): (Tensor, Dimensions, Dimensions) = ??? //elementwiseBinaryOp(x, y) { _ + " / " + _ }
    override def div_grad(x: TensorR, y: TensorR, output: TensorR, xShape: Dimensions, yShape: Dimensions): Unit = ???

    override def /=(x: Tensor, y: Rep[Float]): Unit = ??? //elementwiseInplaceUnaryOp(x)(s => Seq(s + " / ", y))
    override def /=(x: Tensor, y: Tensor): Unit = ??? //elementwiseInplaceBinaryOp(x, y) { _ + " / " + _ }

    override def plusBias(main: Tensor, bias: Tensor): Tensor = ???
    override def plusBias_grad(main: TensorR, bias: TensorR): Unit = ???

    override def plusEqual(base: Tensor, adder: Tensor): Tensor = ???
    override def plusEqual_grad(base: TensorR, adder: TensorR): Unit = ???

    override def geam(x: Tensor, transX: Boolean, alpha: Rep[Float], y: Tensor, transY: Boolean, beta: Rep[Float], output: Tensor): Unit = {
      val alpha1 = NewArray[Float](1); alpha1(0) = alpha
      val beta1 = NewArray[Float](1); beta1(0) = beta
      (transX, transY) match {
        case (false, false) =>
          Tensor.assertShapeEqual(x.shape, y.shape)
          Tensor.assertShapeEqual(x.shape, output.shape)
          val m = x.shape(0)
          val n = x.shape.drop(1).product1
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ",
            n, ",", m, ",", alpha1, ",",
            x.data, ",", n, ",", beta1, ", ", y.data, ", ", n, ", ", output.data, ",", n, "))")
        case (false, true) =>
          assert(x.rank == 2 && y.rank == 2)
          assert(x.shape(0) == y.shape(1) && x.shape(1) == y.shape(0), "is this assertion correct in terms of types?")
          val m = x.shape(0)
          val n = x.shape(1)
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ",
            n, ",", m, ",", alpha1, ",",
            x.data, ",", n, ",", beta1, ", ", y.data, ", ", m, ", ", output.data, ",", n, "))")
        case (true, false) =>
          assert(x.rank == 2 && y.rank == 2)
          assert(x.shape(0) == y.shape(1) && x.shape(1) == y.shape(0))
          val m = x.shape(1)
          val n = x.shape(0)
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, ",
            n, ",", m, ",", alpha1, ",",
            x.data, ",", m, ",", beta1, ", ", y.data, ", ", n, ", ", output.data, ",", n, "))")
        case (true, true) =>
          assert(x.rank == 2 && y.rank == 2) 
          Tensor.assertShapeEqual(x.shape, y.shape)
          val m = x.shape(1)
          val n = x.shape(0)
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, ",
            n, ",", m, ",", alpha1, ",",
            x.data, ",", m, ",", beta1, ", ", y.data, ", ", m, ", ", output.data, ",", n, "))")
      }
    }

    override def trans(x: Tensor): Tensor = {
      assert(x.rank == 2, s"trans only supported for 2D matrix, got ${x.shape.seq}")
      val res = Tensor(mallocArray[Float](x.scalarCount), x.shape.reverse: _*)
      generateRawComment("trans casted as geam call")
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
      if (!permutationKernelMap.contains(dims.toSeq)) {
        if (x.rank == 2) { // this is transpose
          val kernel = s"""
            |__global__ void permute2D${nextKernel}(float *odata, const float *idata, int dimy, int dimx) {
            |
            |  __shared__ float tile[$TILE_DIM][$TILE_DIM+1];  
            |  int x = blockIdx.x * $TILE_DIM + threadIdx.x;
            |  int y = blockIdx.y * $TILE_DIM + threadIdx.y;
            |  if (x < dimx)
            |    for (int j = 0; j < $TILE_DIM && j < dimy - y; j += $BLOCK_ROWS)
            |      tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*dimx + x];
            |  __syncthreads();
            |  x = blockIdx.y * $TILE_DIM + threadIdx.x;  // transpose block offset
            |  y = blockIdx.x * $TILE_DIM + threadIdx.y;
            |  if (x < dimy)
            |    for (int j = 0; j < $TILE_DIM && j < dimx-y; j += $BLOCK_ROWS)
            |      odata[(y+j)*dimy + x] += tile[threadIdx.x][threadIdx.y + j];
            |}
           """.stripMargin
          val kernelName = s"permute2D${nextKernel}"
          permutationKernelMap(dims.toSeq) = (kernel, kernelName)
          nextKernel += 1
        } else if (x.rank == 3) { // this is permutation for 3D Tensor
          if (dims(2) == 2) { // this is the simple case, where the inner most dimension is not permutated
			val kernel = s"""
              |__global__ void permuteSim3D${nextKernel}(float* odata, const float* idata, int dim0, int dim1, int dim2) {
              |  int ioffset = blockIdx.y * dim1 * dim2 + blockIdx.x * dim2;
              |  int ooffset = blockIdx.x * dim0 * dim2 + blockIdx.y * dim2;
              |  for (int i = threadIdx.x; i < dim2; i += blockDim.x)
              |    odata[ooffset + i] += idata[ioffset + i];
              |}
             """.stripMargin
            val kernelName = s"permuteSim3D${nextKernel}"
            permutationKernelMap(dims.toSeq) = (kernel, kernelName)
            nextKernel += 1
          } else { // this is the complicate case for 3D permutation (the inner most dimension is also permutated)
            val kernel = s"""
             |__global__ void permute3D${nextKernel}(float *odata, const float *idata,
             |         int dim0, int dim1, int dim2,
             |         int istr0, int istr1, int ostr0, int ostr1) {
             | 
             |  __shared__ float tile[$TILE_DIM][$TILE_DIM+1];
             | 
             |  int x = blockIdx.x * $TILE_DIM + threadIdx.x;
             |  int y = blockIdx.y * $TILE_DIM + threadIdx.y;
             |  int z = blockIdx.z;
             | 
             |  if (x < dim2)
             |    for (int j = 0; j < $TILE_DIM && j < ${if (dims(2) == 0) "dim0" else "dim1"} - y; j += $BLOCK_ROWS)
             |      tile[threadIdx.y+j][threadIdx.x] = idata[z*${if (dims(2) == 0) "istr1" else "istr0"} + (y+j)*${if (dims(2) == 0) "istr0" else "istr1"} + x];
             | 
             |  __syncthreads();
             | 
             |  x = blockIdx.y * $TILE_DIM + threadIdx.x;  // transpose block offset
             |  y = blockIdx.x * $TILE_DIM + threadIdx.y;
             | 
             |  if (x < ${if (dims(2) == 0) "dim0" else "dim1"})
             |    for (int j = 0; j < $TILE_DIM && j < dim2-y; j += $BLOCK_ROWS)
             |      odata[(y+j)*${if (dims(0) == 2) "ostr0" else "ostr1"} + z*${if (dims(0) == 2) "ostr1" else "ostr0"} + x] += tile[threadIdx.x][threadIdx.y + j];
             |}
             """.stripMargin
            val kernelName = s"permute3D${nextKernel}"
            permutationKernelMap(dims.toSeq) = (kernel, kernelName)
            nextKernel += 1
          }
        } else { // this is for 4D permutation
          if (dims(3) == 3) { // this is the simple case, where the last dimension is not permutated
            val idxes = Seq("blockIdx.z", "blockIdx.y", "blockIdx.x")
            val kernel = s""" 
             |__global__ void permuteSim4D${nextKernel}(float* odata, const float* idata,
             |      int istr0, int istr1, int istr2,   // elide istr3/ostr3 because that is '1'
             |      int ostr0, int ostr1, int ostr2) { // actually ostr2 should be the same as istr2 (can remove)
             |
             |  int ioffset = ${idxes(0)} * istr0 + ${idxes(1)} * istr1 + ${idxes(2)} * istr2;
             |  int ooffset = ${idxes(dims(0))} * ostr0 + ${idxes(dims(1))} * ostr1 + ${idxes(dims(2))} * ostr2;
             |  for (int i = threadIdx.x; i < istr2; i += blockDim.x)
             |    odata[ooffset + i] += idata[ioffset + i];
             |}
             """.stripMargin
            val kernelName = s"permuteSim4D${nextKernel}"
            permutationKernelMap(dims.toSeq) = (kernel, kernelName)
            nextKernel += 1
          } else { // this is the complicated case, where the last dimension is permutated
            
            val setIOffsetBase = if (dims(3) == 0) {
              "int ioffsetBase = x + y * istr0 + blockIdx.z * istr1 + blockIdx.y * istr2;"
            } else if (dims(3) == 1) {
              "int ioffsetBase = x + y * istr1 + blockIdx.z * istr0 + blockIdx.y * istr2;"
            } else { // dims(3) must be 2 now
              "int ioffsetBase = x + y * istr2 + blockIdx.z * istr0 + blockIdx.y * istr1;"
            }
            val pos: Seq[Int] = Seq(0,1,2,3).map(dims.indexOf(_))
            val ostrPos: Seq[String] = pos.map(Seq("ostr0", "ostr1", "ostr2", "1")(_))
            val setOOffsetBase = if (dims(3) == 0) {
              s"int ooffsetBase = x + y * ${ostrPos(3)} + blockIdx.z * ${ostrPos(1)} + blockIdx.y * ${ostrPos(2)};" 
            } else if (dims(3) == 1) {
              s"int ooffsetBase = x + y * ${ostrPos(3)} + blockIdx.z * ${ostrPos(0)} + blockIdx.y * ${ostrPos(2)};"
            } else { // dim(3) must be 2 now
              s"int ooffsetBase = x + y * ${ostrPos(3)} + blockIdx.z * ${ostrPos(0)} + blockIdx.y * ${ostrPos(1)};"
            }

            val kernel = s"""
              |__global__ void permute4D${nextKernel}(float *odata, const float *idata,
              |    int dim0, int dim1, int dim2, int dim_3,
              |    int istr0, int istr1, int istr2,
              |    int ostr0, int ostr1, int ostr2,
              |    int strideBIdxX) {
              |
              |  __shared__ float tile[$TILE_DIM][$TILE_DIM+1];
              |
              |  int blockIdxY = blockIdx.x / strideBIdxX;
              |  int blockIdxX = blockIdx.x - blockIdxY * strideBIdxX;
              |  int x = blockIdxX * $TILE_DIM + threadIdx.x;
              |  int y = blockIdxY * $TILE_DIM + threadIdx.y;
              |
              |  if (x < dim_3) {
              |    ${setIOffsetBase}
              |    for (int j = 0; j < $TILE_DIM && j < ${Seq("dim0", "dim1", "dim2")(dims(3))} - y; j += $BLOCK_ROWS)
              |      tile[threadIdx.y+j][threadIdx.x] = idata[ioffsetBase + j * ${Seq("istr0", "istr1", "istr2")(dims(3))}];
              |  }  
              |  __syncthreads();
              |
              |  x = blockIdxY * $TILE_DIM + threadIdx.x;  // transpose block offset
              |  y = blockIdxX * $TILE_DIM + threadIdx.y;
              |
              |  if (x < ${Seq("dim0", "dim1", "dim2", "dim_3")(dims(3))}) {
              |    ${setOOffsetBase}
              |    for (int j = 0; j < $TILE_DIM && j < dim_3 - y; j += $BLOCK_ROWS)
              |      odata[ooffsetBase + j * ${ostrPos(3)}] += tile[threadIdx.x][threadIdx.y + j];
              |  }
              |}
             """.stripMargin
            val kernelName = s"permute4D${nextKernel}"
            permutationKernelMap(dims.toSeq) = (kernel, kernelName)
            nextKernel += 1
          }
        }
      }
      // end of if (permutationKernelMap.contains()), the following code should call the kernel function
      val (_, kernelName) = permutationKernelMap(dims.toSeq)
      if (x.rank == 2) { // this is transpose
        unchecked[Unit](
         "{\n",
         s"dim3 dimGrid((", x.shape(1), s"+$TILE_DIM-1)/$TILE_DIM, (", x.shape(0), s"+$TILE_DIM-1)/$TILE_DIM, 1);\n",
         s"dim3 dimBlock($TILE_DIM, $BLOCK_ROWS, 1);\n",
         s"$kernelName<<<dimGrid, dimBlock>>>(", resTensor.data, ", ", x.data, ", ", x.shape(0), ", ", x.shape(1), ");\n", 
         "}\n" 
        )
      } else if (x.rank == 3) { // this is permutation for 3D Tensor
        if (dims(2) == 2) { // this is the simple case (inner most dimension doesn't permute)
          unchecked[Unit](
            "{\n",
           s"dim3 dimGrid(", x.shape(1), ", ", x.shape(0), ", 1);\n",
           s"dim3 dimBlock(256, 1, 1);\n",
           s"$kernelName<<<dimGrid, dimBlock>>>(", resTensor.data, ", ", x.data, ", ", x.shape(0), ", ", x.shape(1), ", ", x.shape(2), ");\n",
            "}\n"
          )
        } else { // this is the complicated case for 3D permutation
          unchecked[Unit](
            "{\n",
           s"dim3 dimGrid((", x.shape(2), s"+$TILE_DIM-1)/$TILE_DIM,(", x.shape(dims(2)), s"+$TILE_DIM-1)/$TILE_DIM,", if(dims(2)==0) x.shape(1) else x.shape(0), ");\n",
           s"dim3 dimBlock($TILE_DIM, $BLOCK_ROWS, 1);\n", 
           s"$kernelName<<<dimGrid, dimBlock>>>(", resTensor.data, ", ", x.data, ", ", x.shape(0), ", ", x.shape(1), ", ", x.shape(2), ", ", x.shape.strides(0),
                 ", ", x.shape.strides(1), ", ", resTensor.shape.strides(0), ", ", resTensor.shape.strides(1), ");\n",
            "}\n"
          )
        } 
      } else { // this is the permutation for 4D Tensor
        if (dims(3) == 3) { // this is the simple case for 4D Tensor (inner most dimension doesn't permute)
          unchecked[Unit](
            "{\n",
           s"dim3 dimGrid(", x.shape(2), ", ", x.shape(1), ", ", x.shape(0), ");\n",
           s"dim3 dimBlock(256, 1, 1);\n",
           s"$kernelName<<<dimGrid, dimBlock>>>(", resTensor.data, ", ", x.data, ", ", x.shape.strides(0), ", ", x.shape.strides(1), ", ",
                x.shape.strides(2), ", ", resTensor.shape.strides(0), ", ", resTensor.shape.strides(1), ", ", resTensor.shape.strides(2), ");\n",
            "}\n"
          )
        } else { // this is the complicated case for 4D Tensor
          unchecked[Unit](
            "{\n",
           s"int strideBIdxX = (", x.shape(3), s" + $TILE_DIM - 1)/$TILE_DIM;\n",
           s"int strideBIdxY = (", x.shape(dims(3)), s" + $TILE_DIM -1)/$TILE_DIM;\n",
           s"dim3 dimGrid(strideBIdxX * strideBIdxY, ", if (dims(3)==2) x.shape(1) else x.shape(2), ", ", if (dims(3)==0) x.shape(1) else x.shape(0), ");\n",
           s"dim3 dimBlock($TILE_DIM, $BLOCK_ROWS, 1);\n",
           s"$kernelName<<<dimGrid, dimBlock>>>(", resTensor.data, ", ", x.data, ", ", x.shape(0), ", ", x.shape(1), ", ", x.shape(2), ", ", x.shape(3), ", ",
                 x.shape.strides(0), ", ", x.shape.strides(1), ", ", x.shape.strides(2), ", ",
                 resTensor.shape.strides(0), ", ", resTensor.shape.strides(1), ", ", resTensor.shape.strides(2), ", strideBIdxX);\n",
            "}\n"
          )
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

    override def gemm(x: Tensor, transX: Boolean, y: Tensor, transY: Boolean, alpha: Float): Tensor = {
      (transX, transY) match {
        case (false, false) =>
          val m = x.shape(0)
          val n = y.shape(1)
          val k = y.shape(0)
          val res = mallocArray[Float](m * n)
          val zero = NewArray[Float](1); zero(0) = 0
          val Alpha = NewArray[Float](1); Alpha(0) = alpha
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ",
            n, ",", m, ",", k, ",", Alpha, ",",
            y.data, ",", n, ",", x.data, ",", k, ",", zero, ",", res, ",", n, "))")
          Tensor(res, m, n)
        case (false, true) =>
          val m = x.shape(0)
          val n = y.shape(0)
          val k = y.shape(1)
          val res = mallocArray[Float](m * n)
          val zero = NewArray[Float](1); zero(0) = 0
          val Alpha = NewArray[Float](1); Alpha(0) = alpha
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, ",
            n, ",", m, ",", k, ",", Alpha, ",",
            y.data, ",", k, ",", x.data, ",", k, ",", zero, ",", res, ",", n, "))")
          Tensor(res, m, n)
        case (true, false) =>
          val m = x.shape(1)
          val n = y.shape(1)
          val k = y.shape(0)
          val res = mallocArray[Float](m * n)
          val zero = NewArray[Float](1); zero(0) = 0
          val Alpha = NewArray[Float](1); Alpha(0) = alpha
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ",
            n, ",", m, ",", k, ",", Alpha, ",",
            y.data, ",", n, ",", x.data, ",", m, ",", zero, ",", res, ",", n, "))")
          Tensor(res, m, n)
        case (true, true) =>
          val m = x.shape(1)
          val n = y.shape(0)
          val k = y.shape(1)
          val res = mallocArray[Float](m * n)
          val zero = NewArray[Float](1); zero(0) = 0
          val Alpha = NewArray[Float](1); Alpha(0) = alpha
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, ",
            n, ",", m, ",", k, ",", Alpha, ",",
            y.data, ",", k, ",", x.data, ",", m, ",", zero, ",", res, ",", n, "))")
          Tensor(res, m, n)
      }
    }

    override def gemm_grad(x: TensorR, transX: Boolean, y: TensorR, transY: Boolean, alpha: Float, output: TensorR): Unit = {
      val alpha1 = NewArray[Float](1); alpha1(0) = alpha;
      val one = NewArray[Float](1); one(0) = 1.0f;
      generateRawComment("backprop of gemm")
      (transX, transY) match {
        case (false, false) =>
          val dim1 = x.x.shape(0); val dim2 = x.x.shape(1); val dim3 = y.x.shape(1)
          if (!x.isInput) unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, ",
            dim2, ",", dim1, ",", dim3, ",", alpha1, ",",
            y.x.data, ",", dim3, ",", output.d.data, ",", dim3, ",", one, ",", x.d.data, ",", dim2, "))")
          if (!y.isInput) unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ",
            dim3, ",", dim2, ",", dim1, ",", alpha1, ",",
            output.d.data, ",", dim3, ",", x.x.data, ",", dim2, ",", one, ",", y.d.data, ",", dim3, "))")
        case (false, true) =>
          val dim1 = x.x.shape(0); val dim2 = x.x.shape(1); val dim3 = y.x.shape(0)
          if (!x.isInput) unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ",
            dim2, ",", dim1, ",", dim3, ",", alpha1, ",",
            y.x.data, ",", dim2, ",", output.d.data, ",", dim3, ",", one, ",", x.d.data, ",", dim2, "))")
          if (!y.isInput) unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ",
            dim2, ",", dim3, ",", dim1, ",", alpha1, ",",
            x.x.data, ",", dim2, ",", output.d.data, ",", dim3, ",", one, ",", y.d.data, ",", dim2, "))")
        case (true, false) =>
          val dim1 = x.x.shape(1); val dim2 = x.x.shape(0); val dim3 = y.x.shape(1)
          if (!x.isInput) unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, ",
            dim1, ",", dim2, ",", dim3, ",", alpha1, ",",
            output.d.data, ",", dim3, ",", y.x.data, ",", dim3, ",", one, ",", x.d.data, ",", dim1, "))")
          if (!y.isInput) unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ",
            dim3, ",", dim2, ",", dim1, ",", alpha1, ",",
            output.d.data, ",", dim3, ",", x.x.data, ",", dim1, ",", one, ",", y.d.data, ",", dim3, "))")
        case (true, true) =>
          val dim1 = x.x.shape(1); val dim2 = x.x.shape(0); val dim3 = y.x.shape(0)
          if (!x.isInput) unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, ",
            dim1, ",", dim2, ",", dim3, ",", alpha1, ",",
            output.d.data, ",", dim3, ",", y.x.data, ",", dim2, ",", one, ",", x.d.data, ",", dim1, "))")
          if (!y.isInput) unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, ",
            dim2, ",", dim3, ",", dim1, ",", alpha1, ",",
            x.x.data, ",", dim1, ",", output.d.data, ",", dim3, ",", one, ",", y.d.data, ",", dim2, "))")
      }
    }

    override def conv2D_batch(input: Tensor, kernel: Tensor, bias: Option[Tensor], strides: Seq[Int], pads: Seq[Int]): (Tensor, Option[Tensor], Int) = ???
    override def conv2D_batch_grad(input: TensorR, finput: Option[TensorR], filter: TensorR, res: TensorR, bias: Option[TensorR] = None,
                                   padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int), counter: Int): Unit = ???
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

    override def mask4D(input: Tensor, lengths: Rep[Array[Int]]): Tensor = {
      // inplace mask (input is of size Batch * c * d * Time, lengths are the actual length of each sequence in batch)
      // Note: We assume that lengths is passed to GPU already, at the beginning of each epoch
      assert(input.rank == 4, s"mask4D only deals with inputs of 4D, got ${input.rank}")
      val nGrid = 28
      // unchecked[Unit]("{\n__device__ int dims[4] = {", input.shape.strides(0), ", ", input.shape.strides(1), ", ", input.shape.strides(2), ", ", input.shape.strides(3), "}")
      unchecked[Unit](s"mask4D<<<${nGrid}, 512>>>(", input.data, ", ", lengths, ", ", input.shape.strides(0), ", ", input.shape.strides(1), ", ",
                                                     input.shape.strides(2), ", ", input.shape.strides(3), ", ", input.scalarCount, ")")
      input
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

    override def hardTanh(x: Tensor, min_val: Float = -1.0f, max_val: Float = 1.0f, inPlace: Boolean = false): Tensor = {
      val size = x.scalarCount
      val res = if (inPlace) x.data else mallocArray[Float](size)
      val nGrid = 28
      unchecked[Unit](s"hardTanh<<<${nGrid}, 512>>>(", x.data, ", ", res, ", ", min_val, ", ", max_val, ", ", inPlace, ")")
      Tensor(res, x.shape.seq: _*)
    }
    override def hardTanh_grad(input: TensorR, res: TensorR, min_val: Float = -1.0f, max_val: Float = 1.0f, inPlace: Boolean = false): Unit = {
      val size = input.x.scalarCount
      val nGrid = 28
      unchecked[Unit](s"hardTanh_grad<<<${nGrid}, 512>>>(", input.x.data, ", ", input.d.data, ", ", res.d.data, ", ", min_val, ", ", max_val, ", ", size, ", ", inPlace, ")")
    }

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
        case ElementWiseNoBroadCastOpt.Log =>
          unchecked[Unit](s"elementwise_1D_1D_log<<<${numBlocks},", "512>>>(", input.data, ",", res, ", ", input.scalarCount, ")")
        case ElementWiseNoBroadCastOpt.Exp =>
          unchecked[Unit](s"elementwise_1D_1D_exp<<<${numBlocks},", "512>>>(", input.data, ",", res, ", ", input.scalarCount, ")")
        case ElementWiseNoBroadCastOpt.Sqrt =>
          unchecked[Unit](s"elementwise_1D_1D_sqrt<<<${numBlocks},", "512>>>(", input.data, ",", res, ", ", input.scalarCount, ")")
        case ElementWiseNoBroadCastOpt.Square =>
          unchecked[Unit](s"elementwise_1D_1D_square<<<${numBlocks},", "512>>>(", input.data, ",", res, ", ", input.scalarCount, ")")
        case _ => ???
      }
      Tensor(res, input.shape: _*)
    }

    @virtualize
    def elementwiseOpNoBroadcastGrad(input: TensorR, output: TensorR, op: ElementWiseNoBroadCastOpt.Value): Unit = {
      val numBlocks = 28 // (input.x.scalarCount + 511) / 512
      op match {
        case ElementWiseNoBroadCastOpt.LogGrad =>
          unchecked[Unit](s"elementwise_1D_1D_log_grad<<<${numBlocks},", "512>>>(", input.x.data, ", ", input.d.data, ", ", output.x.data, ", ", output.d.data, ", ", input.x.scalarCount, ")")
        case ElementWiseNoBroadCastOpt.ExpGrad =>
          unchecked[Unit](s"elementwise_1D_1D_exp_grad<<<${numBlocks},", "512>>>(", input.x.data, ", ", input.d.data, ", ", output.x.data, ", ", output.d.data, ", ", input.x.scalarCount, ")")
        case ElementWiseNoBroadCastOpt.SqrtGrad =>
          unchecked[Unit](s"elementwise_1D_1D_sqrt_grad<<<${numBlocks},", "512>>>(", input.x.data, ", ", input.d.data, ", ", output.x.data, ", ", output.d.data, ", ", input.x.scalarCount, ")")
        case ElementWiseNoBroadCastOpt.SquareGrad =>
          unchecked[Unit](s"elementwise_1D_1D_square_grad<<<${numBlocks},", "512>>>(", input.x.data, ", ", input.d.data, ", ", output.x.data, ", ", output.d.data, ", ", input.x.scalarCount, ")")
        case _ => ???
      }
    }

    override def nllLoss(x: Tensor, target: Rep[Array[Int]]): Tensor = {
      assert(x.rank == 2, "Input must be a 2-D tensor")

      val batchSize = x.shape(0)
      val res = Tensor(mallocArray[Float](batchSize), batchSize)
      unchecked[Unit]("nllLoss<<<", batchSize, ", 1>>>(", x.data, ", ", x.shape.strides(0), ", ", res.data, ", ", target, ")")
      res
    }

    override def nllLoss_grad(input: TensorR, res: TensorR, target: Rep[Array[Int]]): Unit = {
      unchecked[Unit]("nllLoss_grad<<<", input.d.shape(0), ", 1>>>(", input.d.shape.strides(0), ", ", res.d.data, ", ", target, ", ", input.d.data, ")")
    }

    override def ctcLoss(prob: TensorR, inputLengths: Rep[Array[Int]], labels: Rep[Array[Int]], labelLengths: Rep[Array[Int]]): Tensor = ???

    override def sum(x: Tensor): Tensor = ???
    override def sum_grad(input: TensorR, res: TensorR): Unit = ???
    override def mean(x: Tensor): Tensor = ???
    override def mean_grad(input: TensorR, res: TensorR): Unit = ???
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
      val dim3 = tensors(0).shape(3)
      val resShape = Seq(dim0, dim1, dim2, dim3)
      val res = this.mallocArray[Float](resShape.product1)
      val resTensor = Tensor(res, dim0, dim1, dim2, dim3)
      val sizeLow = dim2 * dim3
      val sizeHigh = dim0
      val sizeDim1 = tensors(0).shape(1)
      val sizeDim2 = tensors(1).shape(1)

      val nGrid = 28 // tensors(0).scalarCount / 512 / 5 + 1
      unchecked[Unit](
        "{\n",
        s"dim3 grid(${nGrid}, 2);\n",
        "concat2D_1D_greg<<<grid, 512>>>(", tensors(0).data, ", ", sizeDim1, ", ", tensors(0).scalarCount, ", ",
        tensors(1).data, ", ", sizeDim2, ", ", tensors(1).scalarCount, ", ",
        res, ", ", 1, ", ",
        dim0, ", ", dim1, ", ", dim2, ", ", dim3, ", ",
        resTensor.shape.strides(0), ", ", resTensor.shape.strides(1), ", ",resTensor.shape.strides(2), ", ",resTensor.shape.strides(3), ");\n",
        "}")
      resTensor
    }

    override def concat_grad(dim: Int, tensorRs: Seq[TensorR], output: TensorR): Unit = {
      assert(dim == 1, "TODO (Fei Wang): only support dim = 1 so far")
      assert(tensorRs.size == 2, "TODO: (Fei Wang): only support two tensor concatenation so far")
      assert(tensorRs(0).x.rank == 4 && tensorRs(1).x.rank == 4, "TODO: (Fei Wang): only support 4D concat so far")

      val dim0 = tensorRs(0).x.shape(0)
      val dim1 = tensorRs(0).x.shape(1) + tensorRs(1).x.shape(1)
      val dim2 = tensorRs(0).x.shape(2)
      val dim3 = tensorRs(0).x.shape(3)
      val sizeLow = dim2 * dim3
      val sizeHigh = dim0
      val sizeDim1 = tensorRs(0).x.shape(1)
      val sizeDim2 = tensorRs(1).x.shape(1)

      val nGrid = 28 //tensorRs(0).x.scalarCount / 512 / 5 + 1
      unchecked[Unit](
        "{\n",
        s"dim3 grid(${nGrid}, 2);\n",
        "concat2D_1D_greg_grad<<<grid, 512>>>(", tensorRs(0).d.data, ", ", sizeDim1, ", ", tensorRs(0).d.scalarCount, ", ",
        tensorRs(1).d.data, ", ", sizeDim2, ", ", tensorRs(1).d.scalarCount, ", ",
        output.d.data, ", ", 1, ", ",
        dim0, ", ", dim1, ", ", dim2, ", ", dim3, ", ",
        output.d.shape.strides(0), ", ", output.d.shape.strides(1), ", ", output.d.shape.strides(2), ", ", output.d.shape.strides(3), ");\n",
        "}")
    }

    override def repeat0(in: Tensor, context: Int): Tensor = ???
    override def repeat0_grad(in: TensorR, out: TensorR, context: Int): Unit = ???

    override def adagrad_update(tr: TensorR, t: Tensor, learning_rate: Float, gradClip: Float, descent: Boolean): Unit = {
      assert(descent, s"TODO: only handle gradient descent (not ascent) so far")
      // assert(tr.x.shape == t.shape, s"tensor and momentum should have the same shape, got ${tr.x.shape} and ${t.shape}")
      val gridDimX = 28 // (t.scalarCount + 511) / 512
      // assert(gridDimX < 65535, s"gridDimX should not breach the limit, got ${gridDimX}")
      unchecked[Unit](s"adagrad_update_1D_1D<<<${gridDimX}, 512>>>(", tr.x.data, ", ", tr.d.data, ", ", t.data, ", ", gradClip, ", ", learning_rate, ", ", t.scalarCount, ")")
    }
    override def momentum_update(tr: TensorR, t: Tensor, learning_rate: Float, momentum: Float, gradClip: Float, nesterov: Boolean, descent: Boolean) = {
      assert(descent, s"TODO: only handle gradient descent (not ascent) so far")
      val gridDimX = 28
      unchecked[Unit](s"momentum_update_1D_1D<<<${gridDimX}, 512>>>(", tr.x.data, ", ", tr.d.data, ", ", t.data, ", ", learning_rate, ", ", momentum, ", ", gradClip, ", ", nesterov, ", ", t.scalarCount, ")")
    }
  }

  object BackendCublas {
    def apply() = new BackendCublas
  }

  // Define default GPU backend.
  def BackendGPU: Backend = BackendCublas()
  backend = BackendGPU
}

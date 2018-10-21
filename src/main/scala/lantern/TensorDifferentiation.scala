package lantern

import scala.util.continuations._
import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._
import scala.collection.mutable.ArrayBuffer
import scala.math._

trait TensorDsl extends DslOps with Diff {

  /**
    Memory Management:
      finally we used a temperate solution called "memory arena". The base code will claim a large piece of code for the whole program.
      internally, every malloc will borrow memory from this arena.
      By using getAllocMem and setAllocMem, we can selectively return a big trunk of memory after one iteration of training.
   **/

  class Timer (val index: Int){
    unchecked[Unit](s"clock_t begin_$index, end_$index; double time_spent_$index")
    def startTimer = { unchecked[Unit](s"begin_$index = clock()") }
    def stopTimer = { unchecked[Unit](s"end_$index = clock()") }
    def printElapsedTime = {
      unchecked[Unit](
        s"end_$index = clock(); printf(",
        "\"Time elapsed: %f\\n\", ",
        s"(double)(end_$index - begin_$index) / CLOCKS_PER_SEC)")
    }
  }

  object Timer {
    var index: Int = 0
    def apply(): Timer = {
      val timer = new Timer(index)
      index += 1
      timer
    }
  }

  def get_time() = unchecked[Double]("((double)clock() / CLOCKS_PER_SEC)")

  class Timer2 (index: Int) {
    unchecked[Unit](s"struct timeval begin_$index, end_$index, diff_$index")
    def startTimer = { unchecked[Unit](s"gettimeofday(&begin_$index, NULL)") }
    def getElapsedTime: Rep[Long] = {
      unchecked[Unit](s"gettimeofday(&end_$index, NULL)")
      unchecked[Unit](s"timeval_subtract(&diff_$index, &end_$index, &begin_$index);")
      unchecked[Long](s"((diff_$index.tv_sec * 1000000L) + (diff_$index.tv_usec))")
    }
  }

  object Timer2 {
    var index: Int = 0
    def apply(): Timer2 = {
      val timer = new Timer2(index)
      index += 1
      timer
    }
  }

  object Dataset {
    class DataLoader(name: String, train: Boolean, mean: Float, std: Float, dims: Seq[Int]) {

      val fd = open(s"../data/bin/${name}_${if (train) "train" else "test"}.bin")
      val len = filelen(fd)
      val data = mmap[Float](fd, len)
      val dLength = (len/4L).toInt

      val tfd = open(s"../data/bin/${name}_${if (train) "train" else "test"}_target.bin")
      val tlen = filelen(tfd)
      val target = mmap[Int](tfd, tlen)
      val length = tlen/4

      def dataset = new Tensor(data, Seq(60000, dims(1), dims(2)))

      @virtualize
      def normalize() = {
        this.foreach { (i, t, d) =>
          t.normalize(mean, std, inPlace = true)
        }
      }

      @virtualize
      def foreach(f: (Rep[Int], Tensor, Rep[Int]) => Unit) = {
        var off = var_new(0)
        for (img <- 0 until length: Rep[Range]) {
          val dataPtr = slice(data, off)
          val t = Tensor(dataPtr, dims : _*)
          f(img, t, target(img))
          off += t.scalarCount
        }
        assertC(off == dLength, "Data length doesn't match\\n")
      }

      @virtualize
      def foreachBatch(batchSize: Int)(f: (Rep[Int], Tensor, Rep[Array[Int]]) => Unit) = {
        var off = var_new(0)
        for (img <- 0 until (length / batchSize): Rep[Range]) {
          val dataPtr = slice(data, off)
          val t = Tensor(dataPtr, (batchSize +: dims.toSeq): _*)
          val targets = slice(target, img * batchSize)
          f(img, t, targets)
          off += t.scalarCount
        }
      }
    }
  }

  def convSize(size: Int, kernelSize: Int, strideSize: Int) = (size - kernelSize)/strideSize + 1
  def mmax(a: Int, b: Int) = if (a >= b) a else b

  @virtualize
  def assertC(cond: Rep[Boolean], msg: String, args: Rep[Any]*): Unit = {
    if(!cond) { printf(msg, args : _*); exit() }
  }

  def slice[T: Manifest](arr: Rep[Array[T]], off: Rep[Int]) = uncheckedPure[Array[T]](arr, "+", off)

  object Encoding {
    val ix_a = 96  // index starts from 1

    def char_to_ix(ch: Rep[Char]): Rep[Int] = ch.AsInstanceOf[Int] - ix_a
    def ix_to_char(ix: Rep[Int]): Rep[Char] = (ix + ix_a).AsInstanceOf[Char]
  }

  case class Dimensions(val dims: Seq[Int]) {
    def apply(idx: Int) = {
      if (idx >= dims.length) ???
      else dims(idx)
    }
    def last = dims.last
    def reverse = Dimensions(dims.reverse)

    val (scalarCount +: strides) = (dims :\ Seq[Int](1)) {
      case (dim, seq@(t +: q)) => (dim * t) +: seq
    }

    override def toString = dims mkString " x "
  }

  implicit def Dimensions2Seq(x: Dimensions) = x.dims

  object Random {
    def rand() = unchecked[Float]("(float)rand()/RAND_MAX")
    def srand(seed: Option[Int] = None) = unchecked[Unit]("srand(",seed.map(_.toString).getOrElse("time(NULL)"),")")
  }

  def exit() = unchecked[Unit]("exit(0)")

  abstract class DataLoop {
    def foreach(f: Rep[Int] => Unit): Unit
  }

  @virtualize
  object DataLoop {
    def apply(size: Int) = if (size <= 3) {
      new DataLoop {
        def foreach(f: Rep[Int] => Unit) = {
          for (i <- 0 until size: Range) f(unit(i))
        }
      }
    } else {
      new DataLoop {
        def foreach(f: Rep[Int] => Unit) = {
          for (i <- 0 until size: Rep[Range]) f(i)
        }
      }
    }
  }

  /**
    * A code generation backend for tensor operations.
    *
    * Note: Eventually, a tensor operation IR may be introduced to enable analyses and
    * transformations such as operator fusion and matrix chain multiplication optimization.
    */
  trait Backend extends Serializable {
    // Setup the backend. This is the first function that will be called in DSL programs.
    def setup(): Unit

    // Cleanup the backend. This is the last function that will be called in DSL programs.
    def cleanup(): Unit

    // Allocate an array with the specified length.
    def mallocArray[T: Manifest](length: Int): Rep[Array[T]]

    // Copy data from one array to another.
    // NOTE: This function is intentionally not defined generically to simplify the codegen implementation.
    // The only user of this function is currently `copyTensorData`.
    def copyFloatArray(dest: Rep[Array[Float]], src: Rep[Array[Float]], length: Int): Unit

    // Copy data from one tensor to another.
    def copyTensorData(dest: Tensor, src: Tensor): Unit = {
      assert(dest.scalarCount == src.scalarCount,
        s"Tensors do not have same scalar count: ${dest.scalarCount}, ${src.scalarCount}")
      copyFloatArray(dest.data, src.data, dest.scalarCount)
    }

    // Initialize a tensor with the specified dimensions and scalar values.
    def makeTensor(dims: Seq[Int], scalars: Float*): Tensor

    // Initialize a tensor with the specified dimensions and repeated value.
    def fill(dims: Seq[Int], value: Rep[Float]): Tensor

    // Initialize a tensor with the specified bias tensor at the specified dimension.
    def fillWithBias(dims: Seq[Int], bias: Tensor, dim: Int): Tensor

    // Fill a tensor in-place with the specified value.
    def fillInPlace(x: Tensor, value: Rep[Float])

    // Compute vector-vector dot product, i.e. inner product.
    // [V] dot [V] => [1] (scalar)
    def vectorVectorDot(x: Tensor, y: Tensor): Tensor

    // Compute matrix-vector dot product.
    // [M1 x M2] dot [M2] => [M1]
    def matrixVectorDot(x: Tensor, y: Tensor): Tensor

    // Compute matrix-matrix dot product.
    // [M1 x M2] dot [M2 x M3] => [M1 x M3]
    def matrixMatrixDot(x: Tensor, y: Tensor): Tensor

    def dot(x: Tensor, y: Tensor): Tensor =
      (x.rank, y.rank) match {
        case (1, 1) => vectorVectorDot(x, y)
        case (2, 1) => matrixVectorDot(x, y)
        case (2, 2) => matrixMatrixDot(x, y)
        case _ => throw new IllegalArgumentException(s"Incompatible shapes: ${x.shape}, ${y.shape}")
      }

    // Elementwise addition.
    def +(x: Tensor, y: Rep[Float]): Tensor
    def +(x: Tensor, y: Tensor): Tensor

    // In-place elementwise addition.
    def +=(x: Tensor, y: Rep[Float]): Unit
    def +=(x: Tensor, y: Tensor): Unit

    // Elementwise subtraction.
    def -(x: Tensor, y: Rep[Float]): Tensor
    def -(x: Tensor, y: Tensor): Tensor

    // In-place elementwise subtraction.
    def -=(x: Tensor, y: Rep[Float]): Unit
    def -=(x: Tensor, y: Tensor): Unit

    // Elementwise multiplication.
    def *(x: Tensor, y: Rep[Float]): Tensor
    def *(x: Tensor, y: Tensor): Tensor

    // In-place elementwise multiplication.
    def *=(x: Tensor, y: Rep[Float]): Unit
    def *=(x: Tensor, y: Tensor): Unit

    // Elementwise division.
    def /(x: Tensor, y: Rep[Float]): Tensor
    def /(x: Tensor, y: Tensor): Tensor

    // In-place elementwise division.
    def /=(x: Tensor, y: Rep[Float]): Unit
    def /=(x: Tensor, y: Tensor): Unit

    /**
      * 2D convolution.
      * @param input Input with shape [batchSize, inChannels, iW, iH].
      * @param kernel Kernel with shape [outChannels, inChannels, kW, kH].
      * @param bias Optional bias with shape [outChannels].
      * @param strides Kernel strides of length two: [strideWidth, strideHeight].
      * @param pads Padding of length four: [padTop, padBottom, padLeft, padRight].
      * @return Result of 2D convolution.
      */
    // NOTE: cuDNN accepts only two padding arguments: [padVertical, padHorizontal].
    def conv2D_batch(input: Tensor, kernel: Tensor, bias: Option[Tensor], strides: Seq[Int], pads: Seq[Int]): (Tensor, Option[Tensor])

    @virtualize
    def conv2D_batch_grad(input: TensorR, finput: Option[TensorR], filter: TensorR, res: TensorR, bias: Option[TensorR] = None,
                          padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int)): Unit

    // TODO: Add more ops:
    // - Reduction operators (e.g. sum).
    //   - Reduction op GPU implementations are non-trivial.
    //   - Roll out own reduction op kernels? There may be significant boilerplate.
    //   - Use thrust library reduction ops? Need to consider device_vector initialization overhead.
    // - Pooling, dropout.
    // - Activation functions (e.g. relu).
    // - Fused multiply add operations?
  }

  /**
    * CPU tensor operation backend. WIP.
    * Tensor ops are defined in terms of primitive operations.
    */
  class BackendCPU protected() extends Backend {
    override def setup() {}
    override def cleanup() {}
    override def mallocArray[T: Manifest](length: Int): Rep[Array[T]] = NewArray[T](length)

    override def copyFloatArray(dest: Rep[Array[Float]], src: Rep[Array[Float]], length: Int): Unit = {
      for (i <- DataLoop(length)) dest(i) = src(i)
    }

    override def makeTensor(dims: Seq[Int], scalars: Float*): Tensor = {
      Tensor(Array(scalars.map(unit(_)): _*), dims: _*)
    }

    override def fill(dims: Seq[Int], value: Rep[Float]): Tensor = {
      val scalarCount = dims.product
      val array = mallocArray[Float](scalarCount)
      for (i <- (0 until scalarCount): Rep[Range]) array(i) = value
      Tensor(array, dims: _*)
    }

    override def fillWithBias(dims: Seq[Int], bias: Tensor, dim: Int): Tensor = {
      assert(dim >= 0 && dim < dims.size, s"Target dimension $dim is out of range $dims")
      assert(bias.rank == 1 && bias.scalarCount == dims(dim),
        "Bias must be 1D and have length equal to the target dimension")
      val scalarCount = dims.product
      val res = mallocArray[Float](scalarCount)

      // iterate for higherDims
      val offset = var_new(0)
      for (hd <- DataLoop(dims.take(dim).product)) {
        // iterate for current dim
        for (cd <- DataLoop(dims.drop(dim).head)) {
          // iterate for lowerDims
          for (ld <- DataLoop(dims.drop(dim+1).product)) {
            res(offset) = bias.data(cd)
            offset += 1
          }
        }
      }
      Tensor(res, dims: _*)
    }

    override def fillInPlace(x: Tensor, value: Rep[Float]): Unit = {
      for (i <- DataLoop(x.scalarCount)) x.data(i) = value
    }

    override def vectorVectorDot(x: Tensor, y: Tensor): Tensor = {
      assert(x.shape(0) == y.shape(0))
      val value = var_new(0.0f)
      for (i <- DataLoop(x.shape.last)) {
        value += x.data(i) * y.data(i)
      }
      val res = mallocArray[Float](1)
      res(0) = readVar(value)
      Tensor(res, 1)
    }

    override def matrixVectorDot(x: Tensor, y: Tensor): Tensor = {
      assert(x.shape(1) == y.shape(0))
      val dim1 = x.shape(0)
      val dim2 = x.shape(1)
      val res = mallocArray[Float](dim1)
      unchecked[Unit] (
        "cblas_sgemv(CblasRowMajor, CblasNoTrans, ",
        dim1, ",", dim2, ",", 1, ",",
        x.data, ",", dim2, ",", y.data, ",", 1, ",", 0, ",", res, ",", 1, ")")
      Tensor(res, dim1)
    }

    override def matrixMatrixDot(x: Tensor, y: Tensor): Tensor = {
      assert(x.shape(1) == y.shape(0))
      val dim1 = x.shape(0)
      val dim2 = x.shape(1)
      val dim3 = y.shape(1)
      val res = mallocArray[Float](dim1 * dim3)
      unchecked[Unit](
        "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ",
        dim1, ",", dim3, ",", dim2, ",", 1, ",",
        x.data, ",", dim2, ",", y.data, ",", dim3, ",", 0, ",", res, ",", dim3, ")")
      // for (i <- DataLoop(dim1)) {
      //   for (j <- DataLoop(dim3)) {
      //     val value = var_new(0.0f)
      //     for (k <- DataLoop(dim2)) {
      //       value += x.data(i * dim2 + k) * y.data(k * dim3 + j)
      //     }
      //     res(i * dim3 + j) = readVar(value)
      //   }
      // }
      Tensor(res, dim1, dim3)
    }

    def elementWiseOpWithBroadCast(x: Tensor, y: Tensor, op: ((Rep[Float], Rep[Float]) => Rep[Float])) = {
      Tensor.dimBroadcast(x.shape, y.shape) match {
        case None => throw new IllegalArgumentException(s"dimensions of vector do not match! ${x.shape.seq} != ${y.shape.seq}")
        case Some((xShape, yShape, resShape)) => {
          val resData = mallocArray[Float](resShape.scalarCount)
          val res = new Tensor(resData, resShape)

          def inplace(offX: Rep[Int], offY: Rep[Int], offRes: Rep[Int], dim: Int): Unit = {
            val offres = var_new[Int](offRes)
            val offx = var_new[Int](offX)
            val offy = var_new[Int](offY)
            for (i <- DataLoop(resShape(dim))) {
              if (dim == resShape.size - 1) {
                resData(offres) = op(x.data(offx), y.data(offy))
              } else {
                inplace(offx, offy, offres, dim + 1)
              }
              offres += resShape.strides(dim)
              if (xShape(dim) > 1) offx += xShape.strides(dim)
              if (yShape(dim) > 1) offy += yShape.strides(dim)
            }
          }
          inplace(0, 0, 0, 0)
          res
        }
      }
    }

    override def +(x: Tensor, y: Rep[Float]): Tensor = x.map(s => s + y)
    override def +(x: Tensor, y: Tensor): Tensor = elementWiseOpWithBroadCast(x, y, _ + _)

    override def +=(x: Tensor, y: Rep[Float]): Unit = x.mapInPlace(s => s + y)
    override def +=(x: Tensor, y: Tensor): Unit = {
      if (y.scalarCount == 1) {
        generateRawComment("+= tensor of dim 0")
        x += y.data(0) // broadcast
      }
      else if (x.scalarCount == 1) ??? // x.data(0) = y.fold(x.data(0))((agg, s) => agg + s)
      else if (x.shape == y.shape)
        for (i <- DataLoop(x.scalarCount)) x.data(i) += y.data(i)
      else throw new IllegalArgumentException(s"dimensions of vector do not match +=! ${x.shape.seq} != ${y.shape.seq}")
    }

    override def -(x: Tensor, y: Rep[Float]): Tensor = x.map(s => s - y)
    override def -(x: Tensor, y: Tensor): Tensor = elementWiseOpWithBroadCast(x, y, _ - _)

    override def -=(x: Tensor, y: Rep[Float]): Unit = x.mapInPlace(s => s - y)
    override def -=(x: Tensor, y: Tensor): Unit = {
      if (y.scalarCount == 1) x -= y.data(0) // broadcast
      else if (x.scalarCount == 1) {
        ???
        // x.data(0) = y.fold(x.data(0))((agg, s) => agg - s)
      }
      else if (x.shape == y.shape)
        for (i <- DataLoop(x.scalarCount)) x.data(i) -= y.data(i)
      else throw new IllegalArgumentException("dimensions of vector do not match -=!")
    }

    override def *(x: Tensor, y: Rep[Float]): Tensor = x.map(s => s * y)
    override def *(x: Tensor, y: Tensor): Tensor = elementWiseOpWithBroadCast(x, y, _ * _)

    override def *=(x: Tensor, y: Rep[Float]): Unit = x.mapInPlace(s => s * y)
    override def *=(x: Tensor, y: Tensor): Unit = {
      if (y.scalarCount == 1) x *= y.data(0) // broadcast
      else if (x.scalarCount == 1) {
        ???
        // x.data(0) = y.fold(x.data(0))((agg, s) => agg * s)
      }
      else if (x.shape == y.shape)
        for (i <- DataLoop(x.scalarCount)) x.data(i) *= y.data(i)
      else throw new IllegalArgumentException("dimensions of vector do not match *=!")
    }

    override def /(x: Tensor, y: Rep[Float]): Tensor = x.map(s => s / y)
    override def /(x: Tensor, y: Tensor): Tensor = elementWiseOpWithBroadCast(x, y, _ / _)

    override def /=(x: Tensor, y: Rep[Float]): Unit = x.mapInPlace(s => s / y)
    override def /=(x: Tensor, y: Tensor): Unit = {
      if (y.scalarCount == 1) x /= y.data(0) // broadcast
      else if (x.scalarCount == 1) ??? // x.data(0) = y.fold(x.data(0))((agg, s) => agg / s)
      else if (x.shape == y.shape)
        for (i <- DataLoop(x.scalarCount)) x.data(i) /= y.data(i)
      else throw new IllegalArgumentException("dimensions of vector do not match /=!")
    }

    type RAF = Rep[Array[Float]]
    def memsetFloatZero(where: RAF, howmany: Rep[Int]) = {
      unchecked[Unit]("memset(", where, ", 0, 4 * ", howmany, ");")
    }
    def memcpyFloat(dst: RAF, src: RAF, howmany: Rep[Int]) = {
      unchecked[Unit]("memcpy(", dst, ", ", src, ", 4 * ", howmany, ");")
    }

    // @virtualize
    def unfoldedCopy(finput: RAF, input: RAF, kW: Int, kH: Int, dW: Int, dH: Int, padW: Int, padH: Int,
    nInputPlane: Int, inputWidth: Int, inputHeight: Int, outputWidth: Int, outputHeight: Int) {
      for (k <- (0 until nInputPlane * kH * kW): Rep[Range]) {
        val nip = k / (kH * kW)
        val rest = k % (kH * kW)
        val kh = rest / kW
        val kw = rest % kW
        val dst = slice(finput, nip*kH*kW*outputHeight*outputWidth + kh*kW*outputHeight*outputWidth + kw*outputWidth*outputWidth)
        val src = slice(input,  nip*inputHeight*inputWidth)
        if (padW > 0 || padH > 0) {
          for (y <- (0 until outputHeight): Rep[Range]) {
            val iy = y * dH - padH + kh
            __ifThenElse ((iy < 0 || iy >= inputHeight), {
              memsetFloatZero(slice(dst, y*outputWidth), outputWidth); ()
            }, {
              if (dW == 1) {
                val ix = 0 - padW + kw;
                val lpad = __ifThenElse ((padW-kw > 0), padW-kw, 0)
                val rpad = __ifThenElse ((padW-(kW-kw-1) > 0), padW-(kW-kw-1), 0)
                __ifThenElse ((outputWidth-rpad-lpad <= 0), {
                  memsetFloatZero(slice(dst, y*outputWidth), outputWidth)
                }, {
                  __ifThenElse ((lpad > 0), memsetFloatZero(slice(dst, y*outputWidth), lpad), ())
                  memcpyFloat(slice(dst, y*outputWidth+lpad), slice(src, iy*inputWidth+ix+lpad), outputWidth-rpad-lpad)
                  __ifThenElse ((rpad > 0), memsetFloatZero(slice(dst, y*outputWidth+outputWidth-rpad), rpad), ())
                })
              } else {
                for (x <- (0 until outputWidth): Rep[Range]) {
                  val ix = x * dW - padW + kw
                  __ifThenElse ((ix < 0 || ix >= inputWidth), memsetFloatZero(slice(dst, y*outputWidth+x), 1),
                    memcpyFloat(slice(dst, y*outputWidth+x), slice(src, iy*inputWidth+ix), 1))
                }
              }
            })
          }
        } else {
          for (y <- (0 until outputHeight): Rep[Range]) {
            val iy = y * dH + kh
            val ix = kw
            if (dW == 1) memcpyFloat(slice(dst, y*outputWidth), slice(src, iy*inputWidth+ix), outputWidth)
            else for (x <- (0 until outputWidth): Rep[Range])
              memcpyFloat(slice(dst, y*outputWidth+x), slice(src, iy*inputWidth+ix+x*dW), 1)
          }
        }
      }
    }

    override def conv2D_batch(input: Tensor, kernel: Tensor, bias: Option[Tensor], strides: Seq[Int], pads: Seq[Int]): (Tensor, Option[Tensor]) = {
      val ((dH:Int) :: (dW:Int) :: Nil) = strides.take(2).toList
      val ((padH:Int) :: (_:Int) :: (padW:Int) :: (_:Int) :: Nil) = pads.take(4).toList
      val nOutputPlane = kernel.shape(0)
      val kH = kernel.shape(2)
      val kW = kernel.shape(3)
      val batchSize = input.shape(0)
      val nInputPlane = input.shape(1)
      val inputHeight = input.shape(2)
      val inputWidth = input.shape(3)
      val outputHeight = (inputHeight + 2*padH - kH) / dH + 1
      val outputWidth  = (inputWidth + 2*padW - kW) / dW + 1
      val output = bias match {
          case Some(bias) => Tensor.fillWithBias(Seq(input.shape(0), kernel.shape(0), outputHeight, outputWidth), bias, 1)
          case None => Tensor.zeros(input.shape(0), kernel.shape(0), outputHeight, outputWidth)
        }
      val finput = Tensor.zeros(batchSize, kW * kH * nInputPlane, outputHeight * outputWidth)
      for (t <- (0 until batchSize): Rep[Range]) {
        val input_t = input(t).data
        val output_t = output(t).data
        val finput_t = finput(t).data
        ConvOutputFrame(input_t, output_t, kernel.data, finput_t, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, nOutputPlane, outputWidth, outputHeight)
      }
      (output, Some(finput))
    }

    def ConvOutputFrame(input: RAF, output: RAF, weight: RAF, finput: RAF, kW: Int, kH: Int, dW: Int, dH: Int, padW: Int, padH: Int,
      nInputPlane: Int, inputWidth: Int, inputHeight: Int, nOutputPlane: Int, outputWidth: Int, outputHeight: Int) {

      unfoldedCopy(finput, input, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight)
      // addmm(output, 1, output, 1, weight, finput)
      // finput viewed as: kW*kH*nInputPlane, outputHeight * outputWidth
      // input  viewed as: nInputPlane, inputWidth, inputHeight
      val dim1 = nOutputPlane
      val dim2 = kW * kH *nInputPlane
      val dim3 = outputHeight * outputWidth
      unchecked[Unit](
        "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ",
        dim1, ",", dim3, ",", dim2, ",", 1, ",",
        weight, ",", dim2, ",", finput, ",", dim3, ",", 1, ",", output, ",", dim3, ")")
    }

    // 2-D convolution that supports batches. Uses `conv2D_inplace` as subroutine.
    def conv2D_batch_deprecate(input: Tensor, kernel: Tensor, bias: Option[Tensor], strides: Seq[Int], pads: Seq[Int]): Tensor = {
      // TODO: Dedupe assertions/shape calculations with cuDNN implementation.
      assert(input.rank == 4, "Input must be 4-D (first dimension is batch size)")
      assert(kernel.rank == 4, "Kernel must be 4-D")
      bias match {
        case Some(bias) =>
          assert(bias.rank == 1, s"Bias should be 1-D, got ${bias.shape}")
          assert(bias.shape(0) == kernel.shape(0), "Bias length must equal number of out-channels")
        case None => ()
      }
      assert(kernel.shape(1) == input.shape(1), "In-channel count mismatch: input.shape[1] should match kernel.shape[1]")
      assert(input.shape(2) >= kernel.shape(2) && input.shape(3) >= kernel.shape(3), "Image too small for conv")

      assert(strides.size == 2, "Strides should have length 2: [strideRow, strideColumn]")
      assert(pads.size == 4, "Pads should have length 4: [padTop, padBottom, padLeft, padRight]")
      val ((strideRow:Int) :: (strideCol:Int) :: Nil) = strides.take(2).toList
      val ((padUp:Int) :: (padDown:Int) :: (padLeft:Int) :: (padRight:Int) :: Nil) = pads.take(4).toList
      assert(strideRow >= 1, "Row stride must be at least 1")
      assert(strideCol >= 1, "Column stride must be at least 1")
      assert(padUp == padDown && padUp == padLeft && padUp == padRight, "All paddings must be equal (for now)")

      val resWidth = convSize(input.shape(2) + padLeft + padRight, kernel.shape(2), strideRow)
      val resHeight = convSize(input.shape(3) + padUp + padDown, kernel.shape(3), strideCol)
      val res = bias match {
        case Some(bias) => Tensor.fillWithBias(Seq(input.shape(0), kernel.shape(0), resWidth, resHeight), bias, 1)
        case None => Tensor.zeros(input.shape(0), kernel.shape(0), resWidth, resHeight)
      }

      for (i <- DataLoop(input.shape(0))) {
        val ptrInput = slice(input.data, i * input.shape.strides(0))
        val ptrOutput = slice(res.data, i * res.shape.strides(0))
        conv2D_inplace(
          Tensor(ptrOutput, res.shape.drop(1): _*),
          Tensor(ptrInput, input.shape.drop(1): _*),
          kernel, strides, pads)
      }
      res
    }

    @virtualize
    def conv2D_inplace(res: Tensor, input: Tensor, kernel: Tensor, strides: Seq[Int], pads: Seq[Int]): Unit = {
      val totalPads = pads.sum
      val ((strideRow:Int) :: (strideCol:Int) :: Nil) = strides.take(2).toList
      val ((padUp:Int) :: (padDown:Int) :: (padLeft:Int) :: (padRight:Int) :: Nil) = pads.take(4).toList
      val resWidth = res.shape(1)
      val resHeight = res.shape(2)

      val offOut = var_new(0)                         // offset for the res by channel
      val offWeight1 = var_new(0)                     // offset for the kernel by channel (dim_0)
      for (outPane <- DataLoop(kernel.shape(0))) {
        val offWeight2 = var_new(offWeight1)          // offset for the kernel for each z-dim of a given channel
        val offInput = var_new(0)                     // offset for this for each channel of input
        val ptrOutput = slice(res.data, offOut)           // res, restarting from the start of this output channel (2D)
        for (inPane <- DataLoop(input.shape(0))) {
          val ptrInput = slice(input.data, offInput)       // input, restarting from the start of this input channel (2D)
          val ptrWeight = slice(kernel.data, offWeight2)  // kernel, restarting from the start of this input channel (2D)

          if (totalPads == 0) conv2D1(
            Tensor(ptrOutput, resHeight, resWidth),
            Tensor(ptrInput, input.shape(1), input.shape(2)),
            Tensor(ptrWeight, kernel.shape(2), kernel.shape(3)),
            strideRow, strideCol)
          else conv2D2(
            Tensor(ptrOutput, resHeight, resWidth),
            Tensor(ptrInput, input.shape(1), input.shape(2)),
            Tensor(ptrWeight, kernel.shape(2), kernel.shape(3)),
            strideRow, strideCol, padUp, padDown, padLeft, padRight)

          offWeight2 += kernel.shape.strides(1)
          offInput += input.shape.strides(0)
        }
        offWeight1 += kernel.shape.strides(0)
        offOut += res.shape.strides(0)
      }
    }

    // Taken from Torch: THTensorConv.cpp#198L
    // https://github.com/pytorch/pytorch/blob/master/aten/src/TH/generic/THTensorConv.cpp
    @virtualize  // subroutine of conv ops (don't handle padding)
    def conv2D1(res: Tensor, input: Tensor, kernel: Tensor, strideRow: Int, strideCol: Int): Unit = {
      assert(res.rank == 2 && input.rank == 2 && kernel.rank == 2)

      // looping for the output
      val offOutput = var_new(0)                 // offset of the output, move one by one in second loop
      val offInputR = var_new(0)                 // offset of the input, move by each row * strideRow
      for (outRow <- DataLoop(res.shape(0))) {
        val offInputC = var_new(offInputR)       // offset of the input, built on offInputR, move by each strideCol
        for (outCol <- DataLoop(res.shape(1))) {

          // looping for the kernel
          val sum = var_new(0.0f)
          val offKernel = var_new(0)             // offset of the kernel, move by kernel.strides(1) i.e. by row of kernel
          val offInput = var_new(offInputC)      // offset of the input, built on offInputC, move by row of input
          for (kernelRow <- DataLoop(kernel.shape(0))) {
            val ptrInput = slice(input.data, offInput)
            val ptrKernel = slice(kernel.data, offKernel)
            for (kernelCol <- DataLoop(kernel.shape(1))) {
              sum +=  ptrInput(kernelCol) * ptrKernel(kernelCol)
            }
            offKernel += kernel.shape.strides(0)
            offInput += input.shape.strides(0)
          }
          res.data(offOutput) = res.data(offOutput) + sum
          offOutput += 1
          offInputC += strideCol
        }
        offInputR += strideRow * input.shape.strides(0)
      }
    }

    @virtualize  // subroutine of conv ops (handle padding)
    def conv2D2(res: Tensor, input: Tensor, kernel: Tensor, strideRow: Int, strideCol: Int, padUp: Int, padDown: Int, padLeft: Int, padRight: Int): Unit = {
      assert(res.rank == 2 && input.rank == 2 && kernel.rank == 2)

      // looping for the output
      val offOutput = var_new(0)                    // offset of the output, move one by one in second loop
      val InputR = var_new(-padLeft)
      for (outRow <- DataLoop(res.shape(0))) {
        val InputC = var_new(-padUp)
        for (outCol <- DataLoop(res.shape(1))) {

          // looping for the kernel
          val sum = var_new(0.0f)
          val offKernel = var_new(0)                // offset of the kernel, move by row of kernel
          for (kernelRow <- DataLoop(kernel.shape(0))) {
            for (kernelCol <- DataLoop(kernel.shape(1))) {
              val iR = InputR + kernelRow
              val iC = InputC + kernelCol
              if (iR < 0 || iC < 0 || iR >= input.shape(0) || iC >= input.shape(1)) ()
              else {
                sum += kernel.data(offKernel) * input.data(iR * input.shape.strides(0) + iC)
              }
              offKernel += 1
            }
          }
          res.data(offOutput) = res.data(offOutput) + sum

          // stepping of the offsets of the looping for the output
          offOutput += 1
          InputC += strideCol
        }
        InputR += strideRow
      }
    }

    // Gradient of `conv2D_batch`.
    @virtualize
    override def conv2D_batch_grad(input: TensorR, finput: Option[TensorR], filter: TensorR, res: TensorR, bias: Option[TensorR] = None,
                                   padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int)): Unit = {
      // NOTE: Strides/paddings may be in the wrong order.
      assert(dilations._1 == 1 && dilations._2 == 1, "Currently, only dilations of 1 are supported")
      val finputR: TensorR = finput match {
        case None => assert(false, "BackendCPU needs finput to be Some[TensorR], found None"); TensorR(Tensor.zeros(1))
        case Some(finputr) => finputr
      }

      // back-propagate to inputs
      if (!input.isInput) ConvGradInput(res.d, input.d, finputR.d, filter.x, strides._1, strides._2, padding._1, padding._2)
      // back-propagate to weights
      bias match {
        case None => ConvGradParam(finputR.x, res.d, filter.d, None, strides._1, strides._2, padding._1, padding._2)
        case Some(bias) => ConvGradParam(finputR.x, res.d, filter.d, Some(bias.d), strides._1, strides._2, padding._1, padding._2)
      }

      // // back propagate deprecated
      // val strideRow = strides._1
      // val strideCol = strides._2

      // if (padding._1 + padding._2 == 0) {
      //   for (batch <- DataLoop(res.d.shape(0))) {
      //     val offOutputD = var_new(batch * res.d.shape.strides(0)) // offset for the output, based on batch, step by 1
      //     val offFilter = var_new(0) // offset for filter, step by filter strides(1) -- which filter
      //     // looping for the output
      //     for (kOut <- DataLoop(res.d.shape(1))) {
      //       val sum = var_new(0.0f) // collector of bias gradient
      //       val offInputR = var_new(batch * input.x.shape.strides(0)) // offset of input, based on batch, step by input.strides(2) * strideRow
      //       for (row <- DataLoop(res.d.shape(2))) {
      //         val offInputC = var_new(offInputR) // offset of input, based on offInputR, step by strideCol
      //         for (col <- DataLoop(res.d.shape(3))) {
      //           val dCurr: Rep[Float] = res.d.data(offOutputD)
      //           sum += dCurr // collect bias gradient

      //           // looping for the filter
      //           val offInputP = var_new(offInputC) // offset of input, based on offInputC, step by input.strides(2)
      //           val offFilterR = var_new(offFilter) // offset of filter, based on offFilter, step by 1
      //           for (pane <- DataLoop(filter.x.shape(1))) {
      //             val offInputKR = var_new(offInputP) // offset of input, step by input.strides(3) -- row
      //             for (kR <- DataLoop(filter.x.shape(2))) {
      //               for (kC <- DataLoop(filter.x.shape(3))) {
      //                 if (!input.isInput) input.d.data(offInputKR + kC) = input.d.data(offInputKR + kC) + dCurr * filter.x.data(offFilterR)
      //                 filter.d.data(offFilterR) = filter.d.data(offFilterR) + dCurr * input.x.data(offInputKR + kC)
      //                 offFilterR += 1
      //               }
      //               offInputKR += input.x.shape.strides(2)
      //             }
      //             offInputP += input.x.shape.strides(1)
      //           }

      //           offInputC += strideCol
      //           offOutputD += 1
      //         }
      //         offInputR += strideRow * input.x.shape.strides(2)
      //       }
      //       bias match {
      //         case Some(bias) => bias.d.data(kOut) = bias.d.data(kOut) + sum // give value of collector to the bias gradient
      //         case None => ()
      //       }
      //       offFilter += filter.x.shape.strides(0)
      //     }
      //   }
      // } else {
      //   for (batch <- DataLoop(input.x.shape(0))) {
      //     val offOutputD = var_new(batch * res.d.shape.strides(0)) // offset for the output, based on batch, step by 1
      //     val offFilter = var_new(0) // offset for the filter, step by filter strides(1) -- which filter
      //     val offInputD = batch * input.x.shape.strides(0) // fixed offset for the input, based on batch
      //     // looping for the output
      //     for (kOut <- DataLoop(res.d.shape(1))) {
      //       val InputR = var_new(-padding._1) // Row of input, starting from -pads
      //       val sum = var_new(0.0f) // collector of bias gradient
      //       for (row <- DataLoop(res.d.shape(2))) {
      //         val InputC = var_new(-padding._1) // Col of input, starting from -pads
      //         for (col <- DataLoop(res.d.shape(3))) {
      //           val dCurr: Rep[Float] = res.d.data(offOutputD)
      //           sum += dCurr // collect the bias gradient

      //           // looping for the filter
      //           val offFilterR = var_new(offFilter) // offset if filter, based on offFilter, step by 1
      //           // offset of input based on batch, pane, and index of output
      //           val InputI_pane = var_new[Int](offInputD + InputR * input.x.shape.strides(2) + InputC)
      //           for (pane <- DataLoop(filter.d.shape(1))) {
      //             val InputI_kR = var_new[Int](InputI_pane) // offset of input based on InputI_pane and row
      //             for (kR <- DataLoop(filter.d.shape(2))) {
      //               for (kC <- DataLoop(filter.d.shape(3))) {
      //                 if (InputR + kR < 0 || InputR + kR >= input.x.shape(2) || InputC + kC < 0 || InputC + kC >= input.x.shape(3)) ()
      //                 else {
      //                   val InputI = InputI_kR + kC
      //                   if (!input.isInput) input.d.data(InputI) = input.d.data(InputI) + dCurr * filter.x.data(offFilterR)
      //                   filter.d.data(offFilterR) = filter.d.data(offFilterR) + dCurr * input.x.data(InputI)
      //                 }
      //                 offFilterR += 1
      //               }
      //               InputI_kR += input.x.shape.strides(2)
      //             }
      //             InputI_pane += input.x.shape.strides(1)
      //           }

      //           InputC += strideCol
      //           offOutputD += 1
      //         }
      //         InputR += strideRow
      //       }
      //       bias match {
      //         case Some(bias) => bias.d.data(kOut) = bias.d.data(kOut) + sum
      //         case None => ()
      //       }
      //       offFilter += filter.x.shape.strides(0)
      //     }
      //   }
      // }
      // ()
    }

    def ConvGradParam(finput: Tensor, gradOutput: Tensor, gradWeight: Tensor, gradBias: Option[Tensor], dH: Int, dW: Int, padH: Int, padW: Int, scale: Float = 1.0f) = {
      val nInputPlane = gradWeight.shape(1)
      val kH = gradWeight.shape(2)
      val kW = gradWeight.shape(3)
      val batchSize = gradOutput.shape(0)
      val nOutputPlane = gradOutput.shape(1)
      val outputHeight = gradOutput.shape(2)
      val outputWidth = gradOutput.shape(3)
      for (t <- (0 until batchSize): Rep[Range]) {
        val gradOutput_t = gradOutput(t).data
        val finput_t = finput(t).data
        val tfinput = Tensor(finput_t, kW * kH * nInputPlane, outputHeight * outputWidth).trans()  // TODO (Fei Wang): can be optimized by doing trans in gemm
        val dim1 = nOutputPlane
        val dim2 = outputWidth * outputHeight
        val dim3 = kW * kH * nInputPlane
        unchecked[Unit](
          "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ",
          dim1, ",", dim3, ",", dim2, ",", scale, ",",
          gradOutput_t, ",", dim2, ",", tfinput.data, ",", dim3, ",", 1, ",", gradWeight.data, ",", dim3, ")")
        gradBias match {
          case None => ()
          case Some(gradBias) =>
            for (i <- (0 until nOutputPlane): Rep[Range]) {
              val sum = var_new(0.0f)
              val data = slice(gradOutput_t, i * outputWidth * outputHeight)
              for (k <- (0 until outputWidth * outputHeight): Rep[Range]) {
                sum += data(k)
              }
              gradBias.data(i) += scale * sum
            }
        }
      }
    }

    def ConvGradInput(gradOutput: Tensor, gradInput: Tensor, fgradInput: Tensor, weight: Tensor, dH: Int, dW: Int, padH: Int, padW: Int) = {
      val batchSize = gradInput.shape(0)
      val inputHeight = gradInput.shape(2)
      val inputWidth = gradInput.shape(3)
      val nOutputPlane = weight.shape(0)
      val nInputPlane = weight.shape(1)
      val kH = weight.shape(2)
      val kW = weight.shape(3)
      val outputHeight = gradOutput.shape(2)
      val outputWidth = gradOutput.shape(3)
      val tweight = weight.resize(nOutputPlane, nInputPlane * kH * kW).trans()
      for (t <- DataLoop(batchSize)) {
        val gradInput_t = gradInput(t).data
        val gradOutput_t = gradOutput(t).data
        val fgradInput_t = fgradInput(t).data
        val dim1 = kW * kH * nInputPlane
        val dim2 = nOutputPlane
        val dim3 = outputHeight * outputWidth
        unchecked[Unit](
          "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ",
          dim1, ",", dim3, ",", dim2, ",", 1, ",",
          tweight.data, ",", dim2, ",", gradOutput_t, ",", dim3, ",", 0, ",", fgradInput_t, ",", dim3, ")")
        unfoldedAcc(fgradInput_t, gradInput_t, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight)
      }
    }

    def unfoldedAcc(finput: RAF, input: RAF, kW: Int, kH: Int, dW: Int, dH: Int, padW: Int, padH: Int, nInputPlane: Int, inputWidth: Int, inputHeight: Int, outputWidth: Int, outputHeight: Int) {
      for (nip <- (0 until nInputPlane): Rep[Range]) {
        for (kh <- (0 until kH): Rep[Range]) {
          for (kw <- (0 until kW): Rep[Range]) {
            val src = slice(finput, nip*kH*kW*outputHeight*outputWidth + kh*kW*outputHeight*outputWidth + kw*outputHeight*outputWidth)
            val dst = slice(input, nip*inputHeight*inputWidth)
            if (padW > 0 || padH > 0) {
              for (y <- (0 until outputHeight): Rep[Range]) {
                val iy: Rep[Int] = y * dH - padH + kh
                __ifThenElse ((iy < 0 || iy >= inputHeight), (), {
                  if (dW == 1) {
                    val ix: Rep[Int] = 0 - padW + kw
                    val lpad: Rep[Int] = __ifThenElse((padW-kw > 0), padW-kw, 0)
                    val rpad: Rep[Int] = __ifThenElse((padW-(kW-kw-1) > 0), padW-(kW-kw-1), 0)
                    val dst_slice = slice(dst, iy*inputWidth+ix+lpad)
                    val src_slice = slice(src, y*outputWidth+lpad)
                    for (i <- 0 until (outputWidth - lpad - rpad)) dst_slice(i) += src_slice(i)
                  } else {
                    for (x <- (0 until outputWidth): Rep[Range]) {
                      val ix = x*dW - padW + kw
                      __ifThenElse ((ix < 0 || ix >= inputWidth), (), dst(iy*inputWidth+ix) += src(y*outputWidth+x))
                    }
                  }
                  ()
                })
              }
            } else {
              for (y <- (0 until outputHeight): Rep[Range]) {
                val iy = y*dH + kh
                val ix = kw
                if (dW == 1) {
                  val dst_slice = slice(dst, iy*inputWidth+ix)
                  val src_slice = slice(src, y*outputWidth)
                  for (i <- (0 until outputWidth): Rep[Range]) dst_slice(i) += src_slice(i)
                } else {
                  for (x <- (0 until outputWidth): Rep[Range]) {
                    dst(iy*inputWidth+ix+x*dW) += src(y*outputWidth+x)
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  object BackendCPU {
    def apply() = new BackendCPU
  }

  // The current backend for code generation.
  // To switch code generation to a different backend, simply change this value
  // in your DSL program.
  var backend: Backend = BackendCPU()

  class Tensor(val data: Rep[Array[Float]], val dimensions: Seq[Int]) extends Serializable {

    def shape = Dimensions(dimensions)
    val rank = dimensions.length
    val scalarCount = shape.scalarCount
    val isScalar = scalarCount == 1

    assert(shape.strides.length >= 1)
    assert(scalarCount != 0, "Empty Tensor!!!")

    def apply(i: Rep[Int]): Tensor = new Tensor(slice(data, i * shape.strides(0)), shape.tail)
    // def apply(i: Rep[Int], j: Rep[Int]): Tensor = new Tensor(slice(data, i * shape.strides(0)), (j - i + 1) +: shape.tail)

    @virtualize
    def clipAt(bound: Float) = {
      for (i <- DataLoop(scalarCount)) {
        val temp = data(i)
        if (temp > bound) data(i) = bound
        if (temp < -1.0f * bound) data(i) = -1.0f * bound
      }
    }

    def mutate(delta: Rep[Int] => Rep[Float]) = {
      for (i <- DataLoop(scalarCount)) this.data(i) += delta(i)
    }

    def mapInPlace(op: Rep[Float] => Rep[Float]) = {
      for (i <- DataLoop(scalarCount)) this.data(i) = op(this.data(i))
    }

    def changeTo(gen: Rep[Int] => Rep[Float]) = {
      for (i <- DataLoop(scalarCount)) this.data(i) = gen(i)
    }

    def map(op: Rep[Float] => Rep[Float]) = {
      val res = backend.mallocArray[Float](scalarCount)
      for (i <- DataLoop(scalarCount)) res(i) = op(this.data(i))
      new Tensor(res, shape)
    }

    def fold(init: Rep[Float])(op: (Rep[Float], Rep[Float]) => Rep[Float]) = {
      val res = var_new[Float](init)
      for (i <- DataLoop(scalarCount)) var_assign(res, op(res, this.data(i)))
      res
    }

    // Elementwise addition.
    def +(that: Rep[Float]): Tensor = backend.+(this, that)
    def +(that: Tensor): Tensor = backend.+(this, that)

    // In-place elementwise addition.
    def +=(that: Rep[Float]): Unit = backend.+=(this, that)
    def += (that: Tensor): Unit = backend.+=(this, that)

    // Elementwise subtraction.
    def -(that: Rep[Float]): Tensor = backend.-(this, that)
    def -(that: Tensor): Tensor = backend.-(this, that)

    // In-place elementwise subtraction.
    def -=(that: Rep[Float]): Unit = backend.-=(this, that)
    def -= (that: Tensor): Unit = backend.-=(this, that)

    // Elementwise multiplication.
    def *(that: Rep[Float]): Tensor = backend.*(this, that)
    def *(that: Tensor): Tensor = backend.*(this, that)

    // In-place elementwise multiplication.
    def *=(that: Rep[Float]): Unit = backend.*=(this, that)
    def *= (that: Tensor): Unit = backend.*=(this, that)

    // Elementwise division.
    def /(that: Rep[Float]): Tensor = backend./(this, that)
    def /(that: Tensor): Tensor = backend./(this, that)

    // In-place elementwise division.
    def /=(that: Rep[Float]): Unit = backend./=(this, that)
    def /= (that: Tensor): Unit = backend./=(this, that)

    def fillInPlace(value: Rep[Float]): Unit = backend.fillInPlace(this, value)
    def setAsOne() = fillInPlace(1)
    def clear() = fillInPlace(0)

    // Copy data from another tensor, in-place.
    def copy_data(src: Tensor) = backend.copyTensorData(this, src)

    // `dot` represents the following:
    // - vector-vector dot product.
    //   [V] dot [V] => [1] (scalar)
    // - matrix-vector multiplication.
    //   [M1 x M2] dot [M2] => [M1]
    // - matrix-matrix multiplication.
    //   [M1 x M2] dot [M2 x M3] => [M1 x M3]
    def dot(that: Tensor) = {
      generateRawComment(s"dot: ${this.shape.seq}, ${that.shape.seq}")
      (this.rank, that.rank) match {
        case (1, 1) => assert(this.shape(0) == that.shape(0), s"Incompatible shapes: ${this.shape}, ${that.shape}")
        case (2, 1) | (2, 2) => assert(this.shape(1) == that.shape(0), s"Incompatible shapes: ${this.shape}, ${that.shape}")
        case _ => throw new IllegalArgumentException(
          s"Only vector-vector, matrix-vector, and matrix-matrix multiplication are allowed (actual shapes: ${this.shape}, ${that.shape})")
      }
      backend.dot(this, that)
    }

    def dot_trans(that: Tensor) = {
      (this.rank, that.rank) match {
        case (1, 1) | (2, 1) => this.dot(that)
        case (2, 2) =>
          assert (this.shape(1) == that.shape(1), s"Incompatible shapes for dot_trans ${this.shape}. ${that.shape}")
          val dim1 = this.shape(0)
          val dim2 = that.shape(0)
          val dim3 = this.shape(1)
          val res = backend.mallocArray[Float](dim1 * dim2)
          for (i <- DataLoop(dim1)) {
            for (j <- DataLoop(dim2)) {
              val value = var_new(0.0f)
              for (k <- DataLoop(dim3)) {
                value += this.data(i * dim3 + k) * that.data(j * dim3 + k)
              }
              res(i * dim2 + j) = readVar(value)
            }
          }
          Tensor(res, dim1, dim2)
      }
    }

    // NOTE: only handles (Vector Cartesian Vector)
    def cart(that: Tensor) = {
      assert(this.rank == 1 && that.rank == 1, "cartesian product is only for 1d vectors")
      val res = backend.mallocArray[Float](this.shape(0) * that.shape(0))
      val off = var_new(0)
      for (i <- DataLoop(this.shape(0))) {
        for (j <- DataLoop(that.shape(0))) {
          res(off) = data(i) * that.data(j)
          off += 1
        }
      }
      Tensor(res, this.shape(0), that.shape(0))
    }

    def trans() = {
      assert(this.rank == 2, "transpose is only for matrix. Tensor transpose is not supported here")
      val res = backend.mallocArray[Float](this.scalarCount)
      val offT = var_new(0)
      for (i <- DataLoop(this.shape(1))) {
        val off = var_new(0)
        for (j <- DataLoop(this.shape(0))) {
          res(offT + j) = data(off + i)
          off += this.shape(1)
        }
        offT += this.shape(0)
      }
      new Tensor(res, this.shape.reverse)
    }

    // def transpose(dims: Int*) = {
    //   assert(dims.size == this.rank, "transpose parameters should be the same as rank")

    //   val res = Tensor.zeros(this)
    //   for ()
    // }

    def tanh() = this.map(x => Math.tanh(x).toFloat)
    def exp() = this.map(x => Math.exp(x).toFloat)
    def log() = this.map(x => Math.log(x).toFloat)
    def sqrt() = this.map(x => Math.sqrt(x).toFloat)
    def sigmoid() = this.map(x => 1.0f / (Math.exp(-1.0f * x).toFloat + 1.0f))
    def square() = this.map(x => x * x)

    // NOTE: sum all elements
    def sum() = Tensor.scalar(this.fold(0.0f)(_ + _))

    @virtualize
    def sum(dim: Int) = {
      assert(dim >= 0 && dim < this.rank, "dim should be within range of this.nbDims")
      val higherDims = this.shape.take(dim)
      val higherDimsSquashed = higherDims.product
      val resDims = higherDims ++ this.shape.drop(dim + 1)
      val res = Tensor.zeros(resDims: _*)

      // looping over the dims higher than dim, squashed
      for (high <- DataLoop(higherDimsSquashed)) {
        // looping over the dimension to be summed
        val offres = var_new(high * (if (dim == 0) res.scalarCount else res.shape.strides(dim - 1)))
        val offthis = var_new(high * (if (dim == 0) this.scalarCount else this.shape.strides(dim - 1)))
        for (sum <- DataLoop(this.shape(dim))) {
          // looping over the dims lower than dim
          for (low <- DataLoop(this.shape.strides(dim))) {
            res.data(offres + low) += this.data(offthis + low)
          }
          offthis += this.shape.strides(dim)
        }
      }
      res
    }

    @virtualize
    def batchNormAv() = {
      assert(this.rank == 4, "tensor for batch normal averaging should have 4 dimensions")
      val base: Rep[Float] = this.shape(0) * this.shape(2) * this.shape(3) * 1.0f
      val res = Tensor.zeros(this.shape(1), 1, 1)

      for (batch <- DataLoop(this.shape(0))) {
        val offsetBatch = batch * this.shape.strides(0)
        for (channel <- DataLoop(this.shape(1))) {
          val offset = offsetBatch + channel * this.shape.strides(1)
          for (lower <- DataLoop(this.shape.strides(1))) {
            res.data(channel) = res.data(channel) + this.data(offset + lower)
          }
        }
      }
      res / base
    }

    @virtualize
    def check(limit: Float) = {
      val idx = var_new(0)
      while (idx < this.scalarCount && -limit < this.data(idx) && this.data(idx) < limit) {
        idx += 1
      }
      idx != this.scalarCount
    }

    @virtualize
    def max() = this.fold(scala.Float.MinValue)((agg, x) => if (x > agg) x else agg)

    @virtualize
    def max2D(dim: Int) = {
      assert (this.rank == 2, "Only deal with 2D tensor")
      assert (dim == 0 || dim == 1, "dim must be in range of this.nbDims")

      if (dim == 0) ???
      else {
        val res = backend.mallocArray[Float](this.shape(0))
        val offset = var_new(0)
        for (i <- DataLoop(this.shape(0))) {
          val max = var_new(scala.Float.MinValue)
          for (j <- DataLoop(this.shape(1))) {
            if (this.data(offset) > max) max = this.data(offset)
            offset += 1
          }
          res(i) = max
        }
        Tensor(res, this.shape(0))
      }
    }

    // FIXME: Proper tensor
    @virtualize
    def maxIndex() = {
      assert(this.rank == 1)
      val vMax = var_new(this.data(0))
      val iMax = var_new(0)
      for (idx <- 1 until this.scalarCount: Rep[Range]) {
        if (this.data(idx) > vMax) {
          iMax = idx
          vMax = this.data(idx)
        }
      }
      iMax
    }

    @virtualize  // batched log softmax
    def logSoftmaxB() = {
      assert(this.rank == 2, "logSoftmaxB should handle 2D tensors: batch * 1D")

      val max = this.max2D(dim = 1)
      val res = Tensor.zeros_like(this)
      // fill res with exp(x_i - max)
      val offset = var_new(0)
      for (batch <- DataLoop(this.shape(0))) {
        for (i <- DataLoop(this.shape(1))) {
          res.data(offset) = Math.exp(this.data(offset) - max.data(batch)).toFloat
          offset += 1
        }
      }
      val sum = res.sum(dim = 1)
      offset = 0
      for (batch <- DataLoop(res.shape(0))) {
        val logsum = max.data(batch) + Math.log(sum.data(batch)).toFloat
        for (i <- DataLoop(res.shape(1))) {
          res.data(offset) = this.data(offset) - logsum
          offset += 1
        }
      }
      res
    }

    @virtualize
    def logSoftmax() = {
      assert(this.rank == 1, "TODO: logSoftmax only handles 1d vectors so far")

      val m = this.max
      val logsum = m + Math.log(this.fold(0.0f)((agg, x) => agg + Math.exp(x - m).toFloat)).toFloat
      this.map(x => x - logsum)
    }

    @virtualize
    def softmax_batch() = {
      assert(this.rank == 2, "softmax input should be 2-D (batch * 1D logits)")
      val max = this.max2D(dim = 1)
      val res = Tensor.zeros_like(this)
      val offset = var_new(0)
      for (batch <- DataLoop(this.shape(0))) {
        for (i <- DataLoop(this.shape(1))) {
          res.data(offset) = Math.exp(this.data(offset) - max.data(batch)).toFloat
          offset += 1
        }
      }
      val sum = res.sum(dim = 1)
      offset = 0
      for (batch <- DataLoop(res.shape(0))) {
        for (i <- DataLoop(res.shape(1))) {
          res.data(offset) = res.data(offset) / sum.data(batch)
          offset += 1
        }
      }
      res
    }

    @virtualize
    def softmax() = {
      assert(this.rank == 1, "TODO: softmax only handles 1d vectors so far: " + this.rank)

      val m = this.max
      val normalized = this.map(x => x - m)
      val nor_exp = normalized.exp()
      nor_exp / nor_exp.sum()
    }

    @virtualize
    def nllLossB(target: Rep[Array[Int]]) = {
      assert(this.rank == 2, "For nllLossB, input should be 2D and target should be 1D")

      val res = backend.mallocArray[Float](this.shape(0))
      val offset = var_new(0)
      for (batch <- DataLoop(this.shape(0))) {
        res(batch) = -1.0f * this.data(offset + target(batch))
        offset += this.shape.strides(0)
      }
      Tensor(res, this.shape(0))
    }

    @virtualize
    def nllLoss(target: Rep[Int]) = {
      assert(this.rank == 1, "input for nllLoss has to be 1d")

      // assertC(0 <= target && target < this.nbElem, "Incorrect target")
      Tensor.scalar(-1.0f * this.data(target))
    }

    def resize(dims: Int*) = {
      val new_dims = if (dims.forall(_ > 0)) dims else {
        assert(dims.filter(_ < 0) == Seq(-1), s"there should be at most one -1 in the resize dims, got $dims")
        dims.updated(dims.indexOf(-1, 0), this.scalarCount / dims.filter(_ > 0).product)
      }
      assert(new_dims.product == this.scalarCount, s"dims: $new_dims != scalarCount: $scalarCount")

      Tensor(this.data, new_dims : _*)
    }

    def printHead(count: Int = 10, msg: String = ""): Unit = {
      if (msg != "")
        printf(s"$msg (size ${this.shape.seq mkString " x "})\\n")
      for (i <- 0 until count: Rep[Range]) {
        printf(format, this.data(i))
      }
      printf("\\n")
    }

    def print(msg: String = ""): Unit = {
      if (msg != "")
        printf(s"$msg (size ${this.shape.seq mkString " x "})\\n")
      if (this.rank == 4) this.print4D
      else if (this.rank == 3) this.print3D
      else this.printRaw(this.shape.last)
    }

    val format = "%.10f "
    def print4D = {
      val idx = var_new(1)
      for (i <- 0 until this.shape(0): Rep[Range]) {
        val idx1 = var_new(1)
        for (j <- 0 until this.shape(1): Rep[Range]) {
          printf(s"Pane #(%d, %d) - ${this.shape(2)} x ${this.shape(3)}\\n", idx, idx1)
          for (k <- 0 until this.shape(2): Rep[Range]) {
            for (l <- 0 until this.shape(3): Rep[Range]) {
              printf(format, this.data(i * this.shape.strides(0) + j * this.shape.strides(1) + k * this.shape.strides(2) + l))
            }
            printf("\\n")
          }
          printf("\\n\\n")
          idx1 += 1
        }
        idx += 1
      }
    }

    def print3D = {
      val idx = var_new(1)
      for (i <- 0 until this.shape(0): Rep[Range]) {
        printf(s"Pane #%d - ${this.shape(1)} x ${this.shape(2)}\\n", idx)
        for (k <- 0 until this.shape(1): Rep[Range]) {
          for (l <- 0 until this.shape(2): Rep[Range]) {
            printf(format, this.data(i * this.shape.strides(0) + k * this.shape.strides(1) + l))
          }
          printf("\\n")
        }
        printf("\\n\\n")
        idx += 1
      }
    }

    @virtualize
    def printRaw(row: Int = 10) = {
      for (i <- 0 until this.scalarCount: Rep[Range]) {
        printf(format, data(i))
        val imod = i % row
        if (imod == row - 1)
          printf("\\n")
      }
      printf("\\n")
    }

    // setting: this is matrix, that is dims(0)-sized vector, y is dims(1)-sized vector
    // the result is to update this so that this += that * y, where * is Cartesian product
    def add_cartesian(that: Tensor, y: Tensor) = {
      generateRawComment("add_cartesian")
      assert(this.rank == 2 && that.shape == Dimensions(Seq(this.shape(1))) && y.shape == Dimensions(Seq(this.shape(0))) ||
        this.rank == 1 && that.shape == this.shape && y.isScalar, s"${shape} - ${that.shape} - ${y.shape}")
      val off = var_new(0)
      val up = if (this.rank > 1) this.shape(0) else 1
      for (i <- DataLoop(up)) {
        for (j <- DataLoop(shape(1))) {
          this.data(off + j) = this.data(off + j) + that.data(j) * y.data(i)
        }
        off += this.shape(1)
      }
    }

    // setting: this is dims(0)-sized vector, that is matrix (dims(0) * dims(1)), y is dims(1)-sized vector
    // the result is to update this so that this accumulate every matrix col * y
    def add_composion(that: Tensor, y: Tensor) = {
      assert(that.rank == 2 && this.shape.seq == Seq(that.shape(1)) && y.shape.seq == Seq(that.shape(0))
        || that.rank == 1 && this.shape == that.shape && y.isScalar, s"${shape} - ${that.shape} - ${y.shape}")
      val off = var_new(0)
      val up = if (that.rank > 1) that.shape(0) else 1
      for (i <- DataLoop(up)) {
        for (j <- DataLoop(that.shape(1))) {
          data(j) += that.data(off + j) * y.data(i)
        }
        off += that.shape(1)
      }
    }

    @virtualize
    def addMul(that: Tensor, y: Tensor) = {
      assert(this.rank == 2 && that.rank == 2 && y.rank == 2, s"Dimensions: ${this.shape.seq} - ${that.shape.seq} - ${y.shape.seq}")
      assert(this.shape(0) == that.shape(0) && this.shape(1) == y.shape(1) && that.shape(1) == y.shape(0), s"Dimensions: ${this.shape.seq} + ${that.shape.seq} * ${y.shape.seq}")

      var offThis = var_new(0)
      var offThatR = var_new(0)
      var offYC = var_new(0)
      for (i <- DataLoop(this.shape(0))) {
        val offYR = var_new(offYC)
        for (j <- DataLoop(this.shape(1))) {
          val offY = var_new(offYR)
          val offThat = var_new(offThatR)
          for (k <- DataLoop(that.shape(1))) {
            this.data(offThis) = this.data(offThis) + that.data(offThat) * y.data(offY)
            offThat += 1
            offY += y.shape.strides(0)
          }
          offThis += 1
          offYR += 1
        }
        offThatR += that.shape.strides(0)
        offYC *= 0
      }
    }

    // private function to get data with default to the only element
    def getAt(i: Rep[Int]) = {
      if (this.isScalar) data(0)
      else data(i)
    }

    def square(t: Rep[Float]) = t * t
    def add_mult(a: Tensor, b: Tensor) = {
      assert(Tensor.dimCompatible(a, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, b), "dim not Compatible in add_mult")

      val dims0M = mmax(shape(0), mmax(a.shape(0), b.shape(0)))
      val dims1M = mmax(if (this.rank > 1) shape(1) else 1, mmax(if (a.rank > 1) a.shape(1) else 1, if (b.rank > 1) b.shape(1) else 1))
      for (i <- DataLoop(dims0M * dims1M)) {
        if (this.isScalar) { data(0) = data(0) + a.getAt(i) * b.getAt(i) }
        else { data(i) = data(i) + a.getAt(i) * b.getAt(i) }
      }
    }

    def addMul(a: Rep[Float], b: Tensor) = {
      assert(this.shape == b.shape)

      generateRawComment("Generate code for addMul")
      for (i <- DataLoop(this.scalarCount)) {
        this.data(i) = this.data(i) + a * b.data(i)
      }
    }

    def cmulAdd(a: Float, b: Tensor) = {
      assert(this.shape == b.shape)
      for (i <- DataLoop(this.scalarCount))
        this.data(i) = a * this.data(i) + b.data(i)

      this // FIXME ??
    }

    def add_div(a: Tensor, b: Tensor) = {
      assert(Tensor.dimCompatible(a, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, b), "dim not Compatible in add_div")
      val dims0M = mmax(shape(0), mmax(a.shape(0), b.shape(0)))
      val dims1M = mmax(if (rank > 1) shape(1) else 1, mmax(if (a.rank > 1) a.shape(1) else 1, if (b.rank > 1) b.shape(1) else 1))
      for (i <- DataLoop(dims0M * dims1M)) {
        if (this.isScalar) { data(0) = data(0) + a.getAt(i) / b.getAt(i) }
        else { data(i) = data(i) + a.getAt(i) / b.getAt(i) }
      }
    }

    def minus_mult_div_square(a: Tensor, b: Tensor, c: Tensor) = {
      assert(Tensor.dimCompatible(a, b)    && Tensor.dimCompatible(a, c)    && Tensor.dimCompatible(c, b)    &&
        Tensor.dimCompatible(this, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, c),
        "dim not competible in minus_mult_div_square")
      val dims0M = mmax(shape(0), mmax(a.shape(0), mmax(b.shape(0), c.shape(0))))
      val dims1M = mmax(if (rank > 1) shape(1) else 1, mmax(if (a.rank > 1) a.shape(1) else 1, if (b.rank > 1) b.shape(1) else 1))
      for (i <- DataLoop(dims0M * dims1M)) {
        if (this.isScalar) { data(0) = data(0) - a.getAt(i) * b.getAt(i) / square(c.getAt(i)) }
        else { data(i) = data(i) - a.getAt(i) * b.getAt(i) / square(c.getAt(i)) }
      }
    }

    def add_oneMinusSquare_mult(a: Tensor, b: Tensor) = {
      assert(Tensor.dimCompatible(a, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, b), "dim not Compatible in add_oneMinusSquare_mult")
      val dims0M = mmax(shape(0), mmax(a.shape(0), b.shape(0)))
      val dims1M = mmax(if (rank > 1) shape(1) else 1, mmax(if (a.rank > 1) a.shape(1) else 1, if (b.rank > 1) b.shape(1) else 1))
      for (i <- DataLoop(dims0M * dims1M)) {
        if (this.isScalar) { data(0) = data(0) + (1.0f - square(a.getAt(i))) * b.getAt(i) }
        else { data(i) = data(i) + (1.0f - square(a.getAt(i))) * b.getAt(i) }
      }
    }

    def oneMinusThenMult(t: Rep[Float]) = (1.0f - t) * t

    def add_oneMinusThenMult_mult(a: Tensor, b: Tensor) = {
      assert(Tensor.dimCompatible(a, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, b), "dim not Compatible in add_oneMinusThenMult_mult")
      val dims0M = mmax(shape(0), mmax(a.shape(0), b.shape(0)))
      val dims1M = mmax(if (rank > 1) shape(1) else 1, mmax(if (a.rank > 1) a.shape(1) else 1, if (b.rank > 1) b.shape(1) else 1))
      for (i <- DataLoop(dims0M * dims1M)) {
        if (this.isScalar) { data(0) = data(0) + oneMinusThenMult(a.getAt(i)) * b.getAt(i) }
        else { data(i) = data(i) + oneMinusThenMult(a.getAt(i)) * b.getAt(i) }
      }
    }

    // @virtualize
    // def batchNorm(epsilon: Rep[Float], gamma: Tensor, beta: Tensor): Tensor = {
    //   assert(this.rank == 4, "For batchNormalization, the input needs to be 4D tensor")
    //   assert(gamma.rank == 1 && gamma.shape(0) == this.shape(1), "gamma should be 1D, the same as number of feature maps")
    //   assert(beta.rank == 1 && beta.shape(0) == this.shape(1), "beta should be 1D, the same as number of feature maps")
    //   val shift = this - this.global_ave_batch().sum(dim = 0) / this.shape(0)
    //   val vi = shift.square().global_ave_batch().sum(dim = 0) / this.shape(0)
    //   val xhat = shift / (vi + epsilon).sqrt()
    //   val y = xhat * gamma.resize(-1, 1, 1) + beta.resize(-1, 1, 1)
    //   y
    // }

    @virtualize
    def conv2D_batch(kernel: Tensor, bias: Option[Tensor], strides: Seq[Int], pads: Seq[Int]): (Tensor, Option[Tensor]) =
      backend.conv2D_batch(this, kernel, bias, strides, pads)

    @virtualize  // conv op, with bias, with padding, use either conv2D1 or conv2D2 as subroutine, depending on whether padding is zero
    def conv2D(kernel: Tensor, bias: Tensor, strides: Seq[Int], pads: Seq[Int]) = {
      assert(this.rank == 3 && kernel.rank == 4, "For Conv, input should be 3-D and kernel should be 4-D: " + this.rank + "|" + kernel.rank)
      assert(kernel.shape(1) == this.shape(0), "For Conv, input dim_0 should be the same as kernel dim_1")
      assert(this.shape(1) >= kernel.shape(2) && this.shape(2) >= kernel.shape(3), "Image too small for Conv")

      val totalPads = pads.sum
      // TODO: (Fei Wang) not sure if the order is correct!!!
      assert(pads.size == 4, "pads should have 4 values, up, down, left, right")
      assert(strides.size == 2, "strides should have a strideRow and a strideCol")
      val ((strideRow:Int) :: (strideCol:Int) :: Nil) = strides.take(2).toList
      val ((padUp:Int) :: (padDown:Int) :: (padLeft:Int) :: (padRight:Int) :: Nil) = pads.take(4).toList
      assert(strideRow >= 1, "stride of row should be at least 1")
      assert(strideCol >= 1, "stride of col should be at least 1")
      assert(padUp == padDown && padUp == padLeft && padUp == padRight, "For now, assume all values in pads are the same")

      val resWidth = convSize(this.shape(1) + padLeft + padRight, kernel.shape(2), strideRow)
      val resHeight = convSize(this.shape(2) + padUp + padDown, kernel.shape(3), strideCol)
      val res = Tensor.fillWithBias(Seq(kernel.shape(0), resWidth, resHeight), bias, 0)

      val offOut = var_new(0)                         // offset for the res by channel
      val offWeight1 = var_new(0)                     // offset for the kernel by channel (dim_0)
      for (outPane <- DataLoop(kernel.shape(0))) {
        val offWeight2 = var_new(offWeight1)          // offset for the kernel for each z-dim of a given channel
        val offInput = var_new(0)                     // offset for this for each channel of input
        val ptrOutput = slice(res.data, offOut)           // res, restarting from the start of this output channel (2D)
        for (inPane <- DataLoop(this.shape(0))) {
          val ptrInput = slice(this.data, offInput)      // input, restarting from the start of this input channel (2D)
          val ptrWeight = slice(kernel.data, offWeight2)  // kernel, restarting from the start of this input channel (2D)

          if (totalPads == 0) BackendCPU().conv2D1(
            Tensor(ptrOutput, resHeight, resWidth),
            Tensor(ptrInput, this.shape(1), this.shape(2)),
            Tensor(ptrWeight, kernel.shape(2), kernel.shape(3)),
            strideRow, strideCol)
          else BackendCPU().conv2D2(
            Tensor(ptrOutput, resHeight, resWidth),
            Tensor(ptrInput, this.shape(1), this.shape(2)),
            Tensor(ptrWeight, kernel.shape(2), kernel.shape(3)),
            strideRow, strideCol, padUp, padDown, padLeft, padRight)

          offWeight2 += kernel.shape.strides(1)
          offInput += this.shape.strides(0)
        }
        offWeight1 += kernel.shape.strides(0)
        offOut += res.shape.strides(0)
      }
      res
    }

    @virtualize  // conv op, no bias, no padding, use conv2D1 as subroutine
    def conv2D(kernel: Tensor, strideRow: Int, strideCol: Int): Tensor = {
      assert(this.rank == 3 && kernel.rank == 4, "input should be 3-D and kernel should be 4-D for Conv")
      assert(strideRow >= 1, "stride of row should be at least 1")
      assert(strideCol >= 1, "stride of col should be at least 1")
      assert(kernel.shape(1) == this.shape(0), "input dim_0 should be the same as kernel dim_1")
      assert(this.shape(1) >= kernel.shape(2) && this.shape(2) >= kernel.shape(3), "Image too small for Conv")

      val resHeight = convSize(this.shape(1), kernel.shape(2), strideRow)
      val resWidth = convSize(this.shape(2), kernel.shape(3), strideCol)
      val res = Tensor.zeros(kernel.shape(0), resHeight, resWidth)

      val offOut = var_new(0)                      // offset for the res for each channel of the output
      val offWeight1 = var_new(0)                  // offset for the kernel for each channel of the output
      for (outPane <- DataLoop(kernel.shape(0))) {
        val offWeight2 = var_new(offWeight1)       // offset for the kernel for each z-dim of a given channel
        val offInput = var_new(0)                  // offset for this for each channel of input
        val ptrOutput = slice(res.data, offOut)          // res, restarting from the start of this output channel (2D)
        for (inPane <- DataLoop(this.shape(0))) {
          val ptrInput = slice(this.data, offInput)     // input, restarting from the start of this input channel (2D)
          val ptrWeight = slice(kernel.data, offWeight2) // kernel, restarting from the start of this input channel (2D)

          BackendCPU().conv2D1(
            Tensor(ptrOutput, resHeight, resWidth),
            Tensor(ptrInput, this.shape(1), this.shape(2)),
            Tensor(ptrWeight, kernel.shape(2), kernel.shape(3)), strideRow, strideCol)

          offWeight2 += kernel.shape.strides(1)
          offInput += this.shape.strides(0)
        }
        offWeight1 += kernel.shape.strides(0)
        offOut += res.shape.strides(0)
      }
      res
    }

    @virtualize
    def averagePool_batch(kernels: Seq[Int], strides: Seq[Int], paddings: Option[Seq[Int]]): Tensor = {
      assert(this.rank == 4, "the input for averagePool_batch should have 4 dimensions")
      assert(kernels.size == 2 && strides.size == 2, "kernels and strides should be size 2")
      paddings match {
        case None => ()
        case Some(paddings) => assert(paddings.size == 4, "paddings should be size 4 for averagePool_batch")
      }
      val (strideRow :: strideCol :: Nil) = strides.toList
      val (kernelRow :: kernelCol :: Nil) = kernels.toList
      val (padUp :: padDown :: padLeft :: padRight :: Nil) = paddings match {
        case None => List(0, 0, 0, 0)
        case Some(paddings) => paddings.toList
      }
      assert(strideRow >= 1 && kernelRow >= 1, "kernel width and stride width should be at least 1")
      assert(strideCol >= 1 && kernelCol >= 1, "kernel height and stride height should be at least 1")
      assert(this.shape(2) >= kernelRow && this.shape(3) >= kernelCol, "Image too small for averagePool_batch: " + this.shape + "|" + (kernelRow, kernelCol))
      assert(padUp == padDown && padUp == padLeft && padUp == padRight && padUp >= 0, "pad should be the same")

      val resWidth = convSize(this.shape(2) + padUp + padDown, kernelRow, strideRow)
      val resHeight = convSize(this.shape(3) + padLeft + padRight, kernelCol, strideCol)
      val res = Tensor.zeros(this.shape(0), this.shape(1), resWidth, resHeight)

      for (i <- DataLoop(this.shape(0))) {
        val ptrInput = slice(this.data, i * this.shape.strides(0))
        val ptrOutput = slice(res.data, i * res.shape.strides(0))
        Tensor(ptrInput, this.shape.drop(1): _*).averagePool_inplace(
          kernelRow, kernelCol, strideRow, strideCol, padUp, padDown, padLeft, padRight, Tensor(ptrOutput, res.shape.drop(1): _*))
      }
      res
    }

    @virtualize
    def averagePool_inplace(kernelRow: Int, kernelCol: Int, strideRow: Int, strideCol: Int, padUp: Int, padDown: Int, padLeft: Int, padRight: Int, res: Tensor): Unit = {
      val resWidth = res.shape(1)
      val resHeight = res.shape(2)
      val kernelSize = kernelRow * kernelCol * 1.0f

      if (padUp == 0) {
        // looping for the output
        for (resPane <- DataLoop(res.shape(0))) {
          for (resRow <- DataLoop(res.shape(1))) {
            for (resCol <- DataLoop(res.shape(2))) {
              val resOff = resPane * res.shape.strides(0) + resRow * res.shape.strides(1) + resCol
              val inOff = resPane * this.shape.strides(0) + resRow * strideRow * this.shape.strides(1) + resCol * strideCol
              // looping for the kernel
              val sum = var_new[Float](0.0f)
              for (kRow <- DataLoop(kernelRow)) {
                for (kCol <- DataLoop(kernelCol)) {
                  sum += this.data(inOff + kRow * this.shape.strides(1) + kCol)
                }
              }
              res.data(resOff) = sum / kernelSize
            }
          }
        }
      } else {
        ???
      }
    }

    @virtualize
    def maxPool(strideRow: Int, strideCol: Int) = {
      assert(this.rank == 3)

      val resHeight = this.shape(1) / strideRow
      val resWidth = this.shape(2) / strideCol
      val res = Tensor.fill(Seq(this.shape(0), resHeight, resWidth), scala.Float.MinValue)

      // FIXME adhoc transform tensor to be using generic type!
      val savedIdx = NewArray[Int](res.scalarCount)

      val oidxW = var_new(0)  // walks by channel in res
      val iidx = var_new(0)   // walks by 1 in input (this.data)
      for (ichan <- DataLoop(this.shape(0))) {
        val oidx = var_new(oidxW)  // walks by row in res
        for (ox <- DataLoop(res.shape(1))) {
          for (sx <- DataLoop(strideRow)) {
            val oidx2 = var_new(oidx)  // walks by 1 in res
            for (oy <- DataLoop(res.shape(2))) {
              for (sy <- DataLoop(strideCol)) {
                if (this.data(iidx) > res.data(oidx2)) {
                  res.data(oidx2) = this.data(iidx)
                  savedIdx(oidx2) = iidx
                }
                iidx += 1
              }
              oidx2 += 1
            }
          }
          oidx += res.shape.strides(1)
        }
        oidxW += res.shape.strides(0)
      }

      (res, savedIdx)
    }

    @virtualize
    def maxPool_k_batch(kernels: Seq[Int], strides: Seq[Int], paddings: Option[Seq[Int]]): (Tensor, Rep[Array[Int]]) = {
      assert(this.rank == 4, "the input for maxPool (with batch) should have 4 dimensions")
      assert(kernels.size == 2 && strides.size == 2, "kernels and strides should be size 2")
      paddings match {
        case None => ()
        case Some(paddings) => assert(paddings.size == 4, "paddings should be size 4 for maxPool_k_batch")
      }
      val (strideRow :: strideCol :: _) = strides.toList
      val (kernelRow :: kernelCol :: _) = kernels.toList
      val (padUp :: padDown :: padLeft :: padRight :: Nil) = paddings match {
        case None => List(0, 0, 0, 0)
        case Some(paddings) => paddings.toList
      }
      assert(strideRow >= 1 && kernelRow >= 1, "kernel width and stride width should be at least 1")
      assert(strideCol >= 1 && kernelCol >= 1, "kernel height and stride height should be at least 1")
      assert(this.shape(2) >= kernelRow && this.shape(3) >= kernelCol, "Image too small for maxPool_k: " + this.shape + "|" + (kernelRow, kernelCol))
      assert(padUp == padDown && padUp == padLeft && padUp == padRight && padUp >= 0, "pad should be the same")

      val resWidth = convSize(this.shape(2) + padUp + padDown, kernelRow, strideRow)
      val resHeight = convSize(this.shape(3) + padLeft + padRight, kernelCol, strideCol)
      val res = Tensor.fill(Seq(this.shape(0), this.shape(1), resWidth, resHeight), scala.Float.MinValue)
      val savedIdx = NewArray[Int](res.scalarCount)

      for (i <- DataLoop(this.shape(0))) {
        val ptrInput  = slice(this.data, i * this.shape.strides(0))
        val ptrOutput = slice(res.data, i * res.shape.strides(0))
        val ptrIdx    = slice(savedIdx, i * res.shape.strides(0))
        val saveIdxBase = i * this.shape.strides(0)
        Tensor(ptrInput, this.shape.drop(1): _*).maxPool_k_inplace(
          kernelRow, kernelCol, strideRow, strideCol, padUp, padDown, padLeft, padRight, Tensor(ptrOutput, res.shape.drop(1): _*), ptrIdx, saveIdxBase)
      }
      (res, savedIdx)
    }

    @virtualize
    def maxPool_k_inplace(kernelRow: Int, kernelCol: Int, strideRow: Int, strideCol: Int, padUp: Int, padDown: Int, padLeft: Int, padRight: Int, res: Tensor, savedIdx: Rep[Array[Int]], saveIdxBase: Rep[Int]): Unit = {
      val resWidth = res.shape(1)
      val resHeight = res.shape(2)

      if (padUp == 0) {
        // looping for the output
        val offout = var_new[Int](0)                              // offset of res, by channel
        val offin  = var_new[Int](0)                              // offset of input, by channel
        for (outPane <- DataLoop(res.shape(0))) {
          val offout_1 = var_new[Int](offout)                     // offset of res, built on offout, by row
          val offin_1  = var_new[Int](offin)                      // offset of input, built on offin, by row
          for (outRow <- DataLoop(res.shape(1))) {
            val offout_2 = var_new[Int](offout_1)                 // offset of res, built on offout_1, by col
            val offin_2  = var_new[Int](offin_1)                  // offset of input, built on offin_1, by col
            for (outCol <- DataLoop(res.shape(2))) {

              // looping for the kernel
              val this_index_1 = var_new[Int](offin_2)            // offset of this (input) by row of kernel size
              for (dummy1 <- DataLoop(kernelRow)) {
                val this_index_2 = var_new[Int](this_index_1)     // offset of this (input), built on this_index_1, by col of kernel size
                for (dummy <- DataLoop(kernelCol)) {
                  if (this.data(this_index_2) > res.data(offout_2)) {
                    res.data(offout_2) = this.data(this_index_2)
                    savedIdx(offout_2) = this_index_2 + saveIdxBase
                  } else ()
                  this_index_2 += 1
                }
                this_index_1 += this.shape.strides(1)
              }

              offout_2 += 1
              offin_2  += strideCol
            }
            offout_1 += res.shape.strides(1)
            offin_1  += strideRow * this.shape.strides(1)
          }
          offout += res.shape.strides(0)
          offin  += this.shape.strides(0)
        }
      } else {
        // looping for the output
        for (resPane <- DataLoop(res.shape(0))) {
          for (resRow <- DataLoop(res.shape(1))) {
            for (resCol <- DataLoop(res.shape(2))) {
              val resOff = resPane * res.shape.strides(0) + resRow * res.shape.strides(1) + resCol
              // looping for the kernel
              for (kRow <- DataLoop(kernelRow)) {
                for (kCol <- DataLoop(kernelCol)) {
                  val inRow = resRow * strideRow - padUp + kRow
                  val inCol = resCol * strideCol - padUp + kCol
                  if (inRow < 0 || inRow >= this.shape(1) || inCol < 0 || inCol >= this.shape(2)) ()
                  else {
                    val inOff = resPane * this.shape.strides(0) + inRow * this.shape.strides(1) + inCol
                    if (this.data(inOff) > res.data(resOff)) {
                      res.data(resOff) = this.data(inOff)
                      savedIdx(resOff) = inOff
                    } else ()
                  }
                }
              }
            }
          }
        }
      }
    }

    @virtualize
    def maxPool_k(kernels: Seq[Int], strides: Seq[Int], paddings: Option[Seq[Int]]) = {
      assert(this.rank == 3, "the input for maxPool should have 3 dimensions")
      assert(kernels.size == 2 && strides.size == 2, "kernels and strides should be size 2 for maxPool_k")
      paddings match {
        case None => ()
        case Some(paddings) => assert(paddings.size == 4, "paddings should be size 4 for maxPool_k")
      }
      val (strideRow :: strideCol :: Nil) = strides.toList
      val (kernelRow :: kernelCol :: Nil) = kernels.toList
      val (padUp :: padDown :: padLeft :: padRight :: Nil) = paddings match {
        case None => List(0, 0, 0, 0)
        case Some(paddings) => paddings.toList
      }
      assert (strideRow >= 1 && kernelRow >= 1, "kernel width and stride width should be at least 1")
      assert (strideCol >= 1 && kernelCol >= 1, "kernel height and stride height should be at least 1")
      assert (this.shape(1) >= kernelRow && this.shape(2) >= kernelCol, "Image too small for maxpool_k")
      assert (padUp == padDown && padUp == padLeft && padUp == padRight && padUp >= 0, "pad should be the same")

      val resWidth = convSize(this.shape(1) + padUp + padDown, kernelRow, strideRow)
      val resHeight = convSize(this.shape(2) + padLeft + padRight, kernelCol, strideCol)
      val res = Tensor.fill(Seq(this.shape(0), resWidth, resHeight), scala.Float.MinValue)
      val savedIdx = NewArray[Int](res.scalarCount)

      this.maxPool_k_inplace(kernelRow, kernelCol, strideRow, strideCol, padUp, padDown, padLeft, padRight, res, savedIdx, 0)
      (res, savedIdx)
    }

    @virtualize
    def dropout(prob: Float = 0.5f) = {
      assert(0.0f <= prob && prob < 1.0f, s"dropout rate should be [0.0, 1), got $prob")

      val res = backend.mallocArray[Float](this.scalarCount)
      val mask = backend.mallocArray[Float](this.scalarCount)

      val scale = 1.0f / (1.0f - prob)

      for (i <- DataLoop(this.scalarCount)) {
        if (Random.rand() > prob) {
          res(i) = this.data(i) * scale
          mask(i) = scale
        } else {
          res(i) = 0.0f
          mask(i) = 0.0f
        }
      }

      (Tensor(res, this.shape.seq : _*), Tensor(mask, this.shape.seq : _*))
    }

    @virtualize
    def relu(inPlace: Boolean = false) = {
      assert(!inPlace)

      val res = backend.mallocArray[Float](this.scalarCount)
      for (i <- 0 until this.scalarCount: Rep[Range]) {
        if (this.data(i) < 0.0f)
          res(i) = 0.0f
        else
          res(i) = this.data(i)
      }

      Tensor(res, this.shape.seq : _*)
    }

    @virtualize
    def concat(dim: Int, others: Tensor*) = {
      assert(others.size >= 1, "there should be at least one tensor in others")
      assert(dim >= 0 && dim < this.rank, "dim should be within range of this.nbDims")
      assert(others.forall(x => x.rank == this.rank), "all tensors should have the same number of dimensions")
      assert(others.forall(x => (0 until this.rank: Range).forall(i =>  i == dim || x.shape(i) == this.shape(i))),
        "all dimensions except the concatenation dimension should be the same")

      // prepare result tensor
      val higherDims = this.shape.take(dim)
      val higherDimsSquashed = higherDims.product
      val resDims    = (0 until this.rank: Range).map{ i =>
        if (i != dim) this.shape(i)
        else others.map(x => x.shape(dim)).sum + this.shape(dim)}
      val totalnbElem = resDims.product
      val res = backend.mallocArray[Float](totalnbElem)

      // prepare for looping/copying
      val totalFrom = this +: others        // put all tensors in one Seq for easy of handling
      val targetId = var_new(0)             // this is the index of res to write to
      // looping over dims higher than dim, squashed
      for (high <- DataLoop(higherDimsSquashed)) {
        // looping over the concatenation dim
        for (whichTensor <- totalFrom) {
          // looping over the dimensions lower than or equal to dim, in the current tensor
          val stride = if (dim == 0) whichTensor.shape.scalarCount else whichTensor.shape.strides(dim-1)
          val ptrIntput = slice(whichTensor.data, high * stride)
          for (lowOrEqual <- DataLoop(stride)) {
            res(targetId) = ptrIntput(lowOrEqual)
            targetId += 1
          }
        }
      }
      Tensor(res, resDims: _*)
    }

    // @virtualize
    // def split(dim: Int, howmany: Int) = {
    //   // TODO (Fei Wang): only support split at given dimensions into "howmany" equal size
    //   assert(dim >= 0 && dim < this.rank, "dim should be within range of this.rank")
    //   assert(this.shape(dim) % howmany == 0, "dim should be able to divide howmany")

    //   val resDim = this.shape.updated(dim, this.shape(dim) / howmany)
    //   val results = ArrayBuffer[Tensor]()
    //   for (i <- 0 until howmany: Range) results.append(Tensor.zeros(resDim: _*))
    // }

    @virtualize
    def global_ave_batch() = {
      assert(this.rank == 4, "assume this is Tensor with 4D (batch * channel * width * height")
      val resTensor = Tensor.zeros(this.shape.take(2): _*)

      val scale = this.shape.strides(1)
      val offset = var_new(0)                      // offset of this, should step by this.strides(2)
      val res_offset = var_new(0)                  // offset of res, should step by 1
      // looping over each batch
      for (batch <- DataLoop(this.shape(0))) {
        // looping over each channel
        for (channel <- DataLoop(this.shape(1))) {
          // get average of a channel
          val sum = var_new(0.0f)
          val offset_a = var_new(offset)           // offset of this for averaging, should step by this.strides(3)
          for (i <- DataLoop(this.shape(2))) {
            for (j <- DataLoop(this.shape(3))) {
              sum += this.data(offset_a + j)
            }
            offset_a += this.shape.strides(2)
          }
          resTensor.data(res_offset) = sum / scale
          offset += this.shape.strides(1)
          res_offset += 1
        }
      }
      resTensor
    }

    @virtualize
    def global_ave() = {
      // the result should be 1D (channel), by averaging the numbers in (width * height)
      assert(this.rank == 3, "assume this is Tensor with 3D (channel * width * height)")

      val res = backend.mallocArray[Float](this.shape(0))
      // looping over each channel
      for (channel <- DataLoop(this.shape(0))) {
        val offset = var_new(this.shape.strides(0) * channel)
        val sum = var_new(0.0f)
        for (i <- DataLoop(this.shape(1))) {
          for (j <- DataLoop(this.shape(2))) {
            sum += this.data(offset + j)
          }
          offset += this.shape.strides(1)
        }
        res(channel) = sum / (this.shape.strides(0))
      }
      Tensor(res, this.shape(0))
    }

    // FIXME: the MNIST example precomput the mean and std
    // I thought that normalize would need to compute it first and then
    // modify the data to match the one requested.
    // SO here what is expected is to have mean = 0 and std = 1 knowing that
    // the current mean is m and the current std is s
    @virtualize
    def normalize(m: Float, s: Float, inPlace: Boolean = false) = {
      if (inPlace) {
        this.mapInPlace(x => (x - m)/s)
        this
      } else {
        this.map(x => (x - m)/s)
      }
    }

    // fused ops
    def linearTanh(x: Tensor, b: Tensor) = {
      // this is W. We want (W.dot(x)+b).tanh()
      assert(this.rank == 2 && x.rank == 1 && b.rank == 1, "limited generalization")
      generateRawComment("forward for linearTanh")
      // val res_dot = backend.mallocArray[Float](this.shape(0))
      // val res_add = backend.mallocArray[Float](this.shape(0))
      val res_tanh = backend.mallocArray[Float](this.shape(0))
      val offSet = var_new(0)
      for (i <- DataLoop(this.shape(0))) {
        val sum = var_new(0.0f)
        for (j <- DataLoop(this.shape(1))) {
          sum += this.data(offSet + j) * x.data(j)
        }
        // res_dot(i) = sum
        // res_add(i) = b.data(i) + sum
        res_tanh(i) = Math.tanh(b.data(i) + sum).toFloat
        offSet += this.shape(1)
      }
      (Tensor(res_tanh, this.shape(0)))
    }

    def linear2Tanh(x: Tensor, W2: Tensor, x2: Tensor, b: Tensor) = {
      // this is W. We want (W.dot(x) + W2.dot(x2) + b).tanh()
      assert(this.rank == 2 && x.rank == 1 && W2.rank == 2 && x2.rank == 1 && b.rank == 1, "limited generalization")
      generateRawComment("forward for linear2Tanh")
      val res_tanh = backend.mallocArray[Float](this.shape(0))
      val offSet = var_new(0)
      for (i <- DataLoop(this.shape(0))) {
        val sum = var_new(0.0f)
        for (j <- DataLoop(this.shape(1))) {
          sum += this.data(offSet + j) * x.data(j)
          sum += W2.data(offSet + j) * x2.data(j)
        }
        res_tanh(i) = Math.tanh(b.data(i) + sum).toFloat
        offSet += this.shape(1)
      }
      Tensor(res_tanh, this.shape(0))
    }
  }

  object Tensor {
    def apply(data: Rep[Array[Float]], dims: Int*) = new Tensor(data, dims)

    def dimCompatible(a: Tensor, b: Tensor) = {
      (a.shape == b.shape) || a.isScalar || b.isScalar
    }

    def dimBroadcast(a: Seq[Int], b: Seq[Int]): Option[(Dimensions, Dimensions, Dimensions)] = {
      def bc(a: Seq[Int], b: Seq[Int], trail: List[Int]): List[Int] = {
        if (a.size == 0) b.toList ++ trail
        else if (b.size == 0) a.toList ++ trail
        else if (a.last == 1) bc(a.init, b.init, b.last :: trail)
        else if (b.last == 1) bc(a.init, b.init, a.last :: trail)
        else if (a.last == b.last) bc(a.init, b.init, a.last :: trail)
        else List(-1) // indicate dim not Compatible by broadcast
      }
      val res = bc(a, b, List())
      if (res == List(-1)) None
      else {
        // add dimensions of 1 to tensors with smaller rank
        if (a.size > b.size) Some((Dimensions(a), Dimensions(Seq.fill(a.size - b.size)(1) ++ b), Dimensions(res)))
        else if (a.size < b.size) Some((Dimensions(Seq.fill(b.size - a.size)(1) ++ a), Dimensions(b), Dimensions(res)))
        else Some((Dimensions(a), Dimensions(b), Dimensions(res)))
      }
    }

    def randseed(seed: Int) = unchecked[Unit]("srand(", seed, ")")
    def randseed() = unchecked[Unit]("srand(time(NULL))")
    def rand(dims: Int*) = randinit(dims.toSeq, 1.0f, None)
    def rand(dims: Seq[Int], scale: Float) = randinit(dims.toSeq, scale, None)
    def randinit(dim0: Int): Tensor = randinit(Seq(dim0), 1.0f, None)
    def randinit(dim0: Int, seed: Option[Int]): Tensor = randinit(Seq(dim0), 1.0f, seed)
    def randinit(dim0: Int, dim1: Int, scale: Float): Tensor = randinit(Seq(dim0, dim1), scale, None)
    def randinit(dims: Seq[Int], scale: Float = 1.0f, seed: Option[Int] = None): Tensor = {
      val scalarCount = dims.product
      val res = backend.mallocArray[Float](scalarCount)
      for (i <- (0 until scalarCount): Rep[Range]) res(i) = (Random.rand() - 0.5f) * scale
      new Tensor(res, dims)
    }

    def randn(dim0: Int, dim1: Int = 1, scale: Float = 1.0f, offset: Int = 0) = {
      val res = backend.mallocArray[Float](dim0 * dim1)
      for (i <- (0 until dim0 * dim1): Rep[Range]) res(i) = unchecked[Float]("d(gen)") * scale
      Tensor(res, dim0, dim1)
    }

    def randPositive(dims: Int*) = {
      val scalarCount = dims.product
      val res = backend.mallocArray[Float](scalarCount)
      for (i <- (0 until scalarCount): Rep[Range]) res(i) = Random.rand()
      new Tensor(res, dims)
    }

    def fill(dims: Seq[Int], value: Rep[Float]): Tensor = backend.fill(dims, value)

    def fill(dims: Seq[Int], fFill: Seq[Rep[Int]] => Rep[Float]) = {
      val scalarCount = dims.product
      val res = backend.mallocArray[Float](scalarCount)

      var idx = var_new(0)
      def innerFill(args: Seq[Rep[Int]]) = {
        res(idx) = fFill(args)
        idx += 1
      }

      val dum = (dims :\ innerFill _) {
        case (up, f) =>
          (args: Seq[Rep[Int]]) => {
            for (i <- 0 until up: Rep[Range]) {
              f(args :+ i)
            }
          }
      }
      dum(Seq[Rep[Int]]())
      new Tensor(res, dims)
    }

    def fillWithBias(dims: Seq[Int], bias: Tensor, dim: Int) = backend.fillWithBias(dims, bias, dim)

    def scalar(value: Rep[Float]) = {
      val res = backend.mallocArray[Float](1)
      res(0) = value
      Tensor(res, 1)
    }

    def zeros(dims: Int*): Tensor = // fill(dims, 0.0f)
      new Tensor(backend.mallocArray[Float](dims.product), dims)
    def zeros_like(that: Tensor) = zeros(that.shape: _*)
    def ones(dims: Int*) = fill(dims, 1.0f)
    def ones_like(that: Tensor) = ones(that.shape: _*)
    def halves(dims: Int*) = fill(dims, 0.5f)

    def expand(vector: Tensor, dim1: Int) = {
      assert(vector.rank == 1)
      val res = backend.mallocArray[Float](vector.shape(0) * dim1)
      val off = var_new(0)
      for (j <- (0 until dim1): Rep[Range]){
        for (i <- (0 until vector.shape(0)): Rep[Range]) {
          res(off) = vector.data(i)
          off += 1
        }
      }
      new Tensor(res, dim1 +: vector.shape)
    }

    def copy(tensor: Tensor) = {
      val res = backend.mallocArray[Float](tensor.scalarCount)
      for (i <- (0 until tensor.scalarCount): Rep[Range]) res(i) = tensor.data(i)
      new Tensor(res, tensor.shape)
    }

    def fromData(scalars: Float*): Tensor = backend.makeTensor(Seq(scalars.length), scalars: _*)

    def fromData(dims: Seq[Int], scalars: Float*): Tensor = backend.makeTensor(dims, scalars: _*)

    @virtualize
    def assertEqual(a: Tensor, b: Tensor, mark: String = "", tal: Float = 0.000001f) = {
      val errorPrefix = if (mark.nonEmpty) s"ERROR ($mark)" else "ERROR"
      assert(a.shape == b.shape, s"$errorPrefix: tensor shapes are not equal, ${a.shape.seq} != ${b.shape.seq}\\n")

      val i = var_new(0)
      while (i < a.scalarCount && { val diff = a.data(i) - b.data(i); diff > -tal && diff < tal }) {
        i += 1
      }
      if (i < a.scalarCount) {
        printf("%s: tensor data are not equal at index %d, %.4f != %.4f\\n", errorPrefix, i, a.data(i), b.data(i))
        error("")
      }
    }
  }


  // Tensor type is the similar to NumR, just replace RFloat with Tensor
  // also Tensor internally use array, which is mutable by default
  // so both field are val (not var) and can be updated by += -= *= /= setAsOne()
  // all instances of vectors will be shepherded by c++ smart pointers, alleviating the memory leak problem
  // type diff = cps[Unit]

  class TensorR(val x: Tensor, val d: Tensor) extends Serializable {
    var isInput: Boolean = false // true if it is an input (no need to compute gradient)

    def apply(i: Rep[Int]) = new TensorR(x(i), d(i))

    def clip_grad(bound: Float) = {
      d.clipAt(bound)
    }

    // fused ops (but slower!)
    def linearTanh(x: TensorR, b: TensorR) = shift { k: (TensorR => Unit) =>
      val y = TensorR(this.x.linearTanh(x.x, b.x)); k(y)
      generateRawComment("back prop for linearTanh")
      val offSet = var_new(0)
      for (i <- DataLoop(this.x.shape(0))) {
        val d_res_add = (1.0f - y.x.data(i) * y.x.data(i)) * y.d.data(i)
        b.d.data(i) += d_res_add
        for (j <- DataLoop(this.x.shape(1))) {
          x.d.data(j) += d_res_add * this.x.data(offSet + j)
          this.d.data(offSet + j) += d_res_add * x.x.data(j)
        }
        offSet += this.x.shape(1)
      }
    }

    // fused ops (but slower!)
    def linear2Tanh(x: TensorR, W2: TensorR, x2: TensorR, b: TensorR) = shift { k: (TensorR => Unit) =>
      val y = TensorR(this.x.linear2Tanh(x.x, W2.x, x2.x, b.x)); k(y)
      generateRawComment("back prop for linear2Tanh")
      val offSet = var_new(0)
      for (i <- DataLoop(this.x.shape(0))) {
        val d_res_add = (1.0f - y.x.data(i) * y.x.data(i)) * y.d.data(i)
        b.d.data(i) += d_res_add
        for (j <- DataLoop(this.x.shape(1))) {
          val idx = offSet + j
          x.d.data(j) += d_res_add * this.x.data(idx)
          this.d.data(idx) += d_res_add * x.x.data(j)
          x2.d.data(j) += d_res_add * W2.x.data(idx)
          W2.d.data(idx) += d_res_add * x2.x.data(j)
        }
        offSet += this.x.shape(1)
      }
    }

    def backpropElementWiseOpWithBroadCast(that: TensorR, y: TensorR, opThis: ((Rep[Float], Rep[Float], Rep[Float]) => Rep[Float]), opThat: ((Rep[Float], Rep[Float], Rep[Float]) => Rep[Float])): Unit = {
      // assume y.x = elementWiseOpWithBroadCast(this.x, that.x, someOp)
      // assume that opThis returns the increment of this.d; opThat returns the increment of that.d
      // both opThis and opThat takes this.x, that.x, and y.d as parameters
      // TODO (Fei Wang): in some cases (if this, or that are inputs (x, y), there gradients are not initialized/useless)
      Tensor.dimBroadcast(this.x.shape, that.x.shape) match {
        case None => throw new IllegalArgumentException(s"dimensions of tensors do not match! ${this.x.shape.seq} != ${that.x.shape.seq}")
        case Some((thisShape, thatShape, yShape)) => {
          def inplace(offThis: Rep[Int], offThat: Rep[Int], offY: Rep[Int], dim: Int): Unit = {
            val offthis = var_new[Int](offThis)
            val offthat = var_new[Int](offThat)
            val offy = var_new[Int](offY)
            for (i <- DataLoop(yShape(dim))) {
              if (dim == yShape.size - 1) {
                this.d.data(offthis) += opThis(this.x.data(offthis), that.x.data(offthat), y.d.data(offy))
                that.d.data(offthat) += opThat(this.x.data(offthis), that.x.data(offthat), y.d.data(offy))
              } else {
                inplace(offthis, offthat, offy, dim + 1)
              }
              offy += yShape.strides(dim)
              if (thisShape(dim) > 1) offthis += thisShape.strides(dim)
              if (thatShape(dim) > 1) offthat += thatShape.strides(dim)
            }
          }
          inplace(0, 0, 0, 0)
        }
      }
    }

    def + (that: Rep[Float]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x + that); k(y)
      this.d += y.d
    }
    def + (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x + that.x); k(y)
      // this.d += y.d; that.d += y.d
      val opThis = (_: Rep[Float], _: Rep[Float], c: Rep[Float]) => c
      val opThat = (_: Rep[Float], _: Rep[Float], c: Rep[Float]) => c
      backpropElementWiseOpWithBroadCast(that, y, opThis, opThat)
    }

    def - (that: Rep[Float]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x - that); k(y)
      this.d += y.d
    }
    def - (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x - that.x); k(y)
      // this.d += y.d; that.d -= y.d
      val opThis = (_: Rep[Float], _: Rep[Float], c: Rep[Float]) => c
      val opThat = (_: Rep[Float], _: Rep[Float], c: Rep[Float]) => -1.0f * c
      backpropElementWiseOpWithBroadCast(that, y, opThis, opThat)
    }

    // this is element wise multiplication
    def * (that: Rep[Float]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x * that); k(y)
      // this.d += y.d * that  // TODO (Fei Wang) can be optimized to save space
      this.d.addMul(that, y.d)
    }
    def * (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x * that.x); k(y)
      // this.d.add_mult(that.x, y.d); that.d.add_mult(this.x, y.d)
      val opThis = (_: Rep[Float], b: Rep[Float], c: Rep[Float]) => c * b
      val opThat = (a: Rep[Float], _: Rep[Float], c: Rep[Float]) => c * a
      backpropElementWiseOpWithBroadCast(that, y, opThis, opThat)
    }

    // element wise division
    def / (that: Rep[Float]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x / that); k(y)
      // this.d += y.d / that  // TODO (Fei Wang) can be optimized to save space
      this.d.addMul(1.0f / that, y.d)
    }
    def / (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x / that.x); k(y)
      // this.d.add_div(y.d, that.x); that.d.minus_mult_div_square(this.x, y.d, that.x)
      val opThis = (_: Rep[Float], b: Rep[Float], c: Rep[Float]) => c / b
      val opThat = (a: Rep[Float], b: Rep[Float], c: Rep[Float]) => -1.0f * a * c / (b * b)
      backpropElementWiseOpWithBroadCast(that, y, opThis, opThat)
    }

    // `dot` represents the following:
    // - vector-vector dot product.
    //   [V] dot [V] => [1] (scalar)
    // - matrix-vector multiplication.
    //   [M1 x M2] dot [M2] => [M1]
    // - matrix-matrix multiplication.
    //   [M1 x M2] dot [M2 x M3] => [M1 x M3]
    def dot(that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val res = x dot that.x
      val y = TensorR(res); k(y)

      // back-propagate
      (this.x.rank, that.x.rank) match {
        case (1, 1) => this.d.addMul(y.d.data(0), that.x); that.d.addMul(y.d.data(0), this.x)
        case (2, 1) => this.d.add_cartesian(that.x, y.d); // that.d.add_composion(this.x, y.d)
          val dim1 = this.x.shape(0); val dim2 = this.x.shape(1)
          unchecked[Unit](
            "cblas_sgemv(CblasRowMajor, CblasTrans, ",
            dim1, ",", dim2, ",", 1, ",",
            this.x.data, ",", dim2, ",", y.d.data, ",", 1, ",", 1, ",", that.d.data, ",", 1, ")")
        case (2, 2) => val dim1 = this.x.shape(0); val dim2 = this.x.shape(1); val dim3 = that.x.shape(1)
          // use cblas_sgemm
          // this.d += y.d dot that.x.trans()
          unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ",
            dim1, ",", dim2, ",", dim3, ",", 1, ",",
            y.d.data, ",", dim3, ",", that.x.data, ",", dim3, ",", 1, ",", this.d.data, ",", dim2, ")")
          // that.d += this.x.trans() dot y.d
          unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ",
            dim2, ",", dim3, ",", dim1, ",", 1, ",",
            this.x.data, ",", dim2, ",", y.d.data, ",", dim3, ",", 1, ",", that.d.data, ",", dim3, ")")
          // for (i <- DataLoop(dim1)) {
          //   for (j <- DataLoop(dim3)) {
          //     for (k <- DataLoop(dim2)) {
          //       this.d.data(i * dim2 + k) += that.x.data(k * dim3 + j) * y.d.data(i * dim3 + j)
          //       that.d.data(k * dim3 + j) += this.x.data(i * dim2 + k) * y.d.data(i * dim3 + j)
          //     }
          //   }
          // }
      }
    }

    def dot_trans(that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      assert(this.x.rank == 2 && that.x.rank == 2 && this.x.shape(1) == that.x.shape(1), s"shape must match for dot_trans, got ${this.x.shape}, ${that.x.shape}")
      val y = TensorR(x dot_trans that.x); k(y)
      // back-propagate
      val dim1 = this.x.shape(0)
      val dim2 = that.x.shape(0)
      val dim3 = this.x.shape(1)
      for (i <- DataLoop(dim1)) {
        for (j <- DataLoop(dim2)) {
          val curr = y.d.data(i * dim2 + j)
          for (k <- DataLoop(dim3)) {
            this.d.data(i * dim3 + k) += that.x.data(j * dim3 + k) * curr
            that.d.data(j * dim3 + k) += this.x.data(i * dim3 + k) * curr
          }
        }
      }
    }

    def trans(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(this.x.trans()); k(y)
      // back-propagate
      val offT = var_new(0)
      for (i <- DataLoop(this.x.shape(1))) {
        val off = var_new(0)
        for (j <- DataLoop(this.x.shape(0))) {
          this.d.data(off + i) = y.d.data(offT + j)
          off += this.x.shape(1)
        }
        offT += this.x.shape(0)
      }
    }

    def tanh(): TensorR @diff = shift { (k : TensorR => Unit) =>
      val y = TensorR(x.tanh()); k(y)
      this.d.add_oneMinusSquare_mult(y.x, y.d)
    }

    def exp(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.exp()); k(y)
      this.d.add_mult(y.x, y.d)
    }

    def log(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.log()); k(y)
      this.d.add_div(y.d, x)
    }

    def sigmoid(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.sigmoid()); k(y)
      this.d.add_oneMinusThenMult_mult(y.x, y.d)
    }

    def sqrt(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.sqrt()); k(y)
      // this.d += y.d / y.x / 2
      this.d.mutate { (i: Rep[Int]) => y.d.data(i) / y.x.data(i) / 2.0f }
    }

    def square(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.square()); k(y)
      this.d.mutate { (i: Rep[Int]) => y.d.data(i) * this.x.data(i) * 2.0f }
    }

    def sum(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = new TensorR(x.sum(), Tensor.zeros(1)); k(y)
      this.d += y.d
    }

    def sum(dim: Int): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.sum(dim)); k(y)

      // back-propagate
      val higherDims = this.x.shape.take(dim)
      val higherDimsSquashed = higherDims.product
      val resDims = higherDims ++ this.x.shape.drop(dim + 1)
      // looping over the dims higher than dim, squashed
      for (high <- DataLoop(higherDimsSquashed)) {
        // looping over the dimension to be summed
        val offres = var_new(high * (if (dim == 0) y.x.scalarCount else y.x.shape.strides(dim - 1)))
        val offthis = var_new(high * (if (dim == 0) this.x.scalarCount else this.x.shape.strides(dim - 1)))
        for (sum <- DataLoop(this.x.shape(dim))) {
          // looping over the dims lower than dim
          for (low <- DataLoop(this.x.shape.strides(dim))) {
            this.d.data(offthis + low) += y.d.data(offres + low)
          }
          offthis += this.x.shape.strides(dim)
        }
      }
    }

    def batchNormAv(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.batchNormAv()); k(y)

      // back-propagate
      val base: Rep[Float] = this.x.shape(0) * this.x.shape(2) * this.x.shape(3) * 1.0f

      for (batch <- DataLoop(this.x.shape(0))) {
        val offsetBatch = batch * this.x.shape.strides(0)
        for (channel <- DataLoop(this.x.shape(1))) {
          val offset = offsetBatch + channel * this.x.shape.strides(1)
          for (lower <- DataLoop(this.x.shape.strides(1))) {
            this.d.data(offset + lower) += y.d.data(channel) / base
          }
        }
      }
    }

    def logSoftmax(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.logSoftmax()); k(y)

      val s = y.d.sum().data(0)
      for (i <- 0 until y.x.scalarCount: Rep[Range]) {
        this.d.data(i) += y.d.data(i) - Math.exp(y.x.data(i)).toFloat * s
      }
    }

    def logSoftmaxB(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.logSoftmaxB()); k(y)

      // back propagate
      val sum = y.d.sum(dim = 1)
      val offset = var_new(0)
      for (batch <- DataLoop(this.x.shape(0))) {
        for (i <- DataLoop(this.x.shape(1))) {
          this.d.data(offset) += y.d.data(offset) - Math.exp(y.x.data(offset)).toFloat * sum.data(batch)
          offset += 1
        }
      }
    }

    def resize(dims: Int*): TensorR @diff = shift { (k: TensorR => Unit) =>
      k(new TensorR(this.x.resize(dims : _*), this.d.resize(dims : _*)))
    }

    def nllLossB(target: Rep[Array[Int]]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.nllLossB(target)); k(y)

      // back propagate
      val offset = var_new(0)
      for (batch <- DataLoop(this.x.shape(0))) {
        this.d.data(offset + target(batch)) += -1.0f * y.d.data(batch)
        offset += this.x.shape.strides(0)
      }
    }

    def nllLoss(target: Rep[Int]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.nllLoss(target)); k(y)
      assert(y.x.isScalar, "y need to be a scalar")
      this.d.data(target) += -1.0f * y.d.data(0)
    }

    @virtualize
    def averagePoolBK(kernels: Seq[Int], strides: Seq[Int], pads: Option[Seq[Int]]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(this.x.averagePool_batch(kernels, strides, pads))
      k(y)

      // back prop
      val strideRow = strides.head
      val strideCol = strides.last
      val kernelRow = kernels.head
      val kernelCol = kernels.last
      val kernelSize = kernelRow * kernelCol
      val pad = pads match {
        case None => 0
        case Some(Seq(padUp, padDown, padLeft, padRight)) => padUp
      }

      if (pad == 0) {
        for (batch <- DataLoop(this.x.shape(0))) {
          // looping for the output
          for (yPane <- DataLoop(y.x.shape(1))) {
            for (yRow <- DataLoop(y.x.shape(2))) {
              for (yCol <- DataLoop(y.x.shape(3))) {
                val indexCurr = batch * y.x.shape.strides(0) + yPane * y.x.shape.strides(1) + yRow * y.x.shape.strides(2) + yCol
                val dCurr = y.d.data(indexCurr) / kernelSize
                val indexThis = batch * this.x.shape.strides(0) + yPane * this.x.shape.strides(1) + yRow * strideRow * this.x.shape.strides(2) + yCol * strideCol
                // looping for the kernel
                for (kRow <- DataLoop(kernelRow)) {
                  for (kCol <- DataLoop(kernelCol)) {
                    this.d.data(indexThis + kRow * this.x.shape.strides(2) + kCol) += dCurr
                  }
                }
              }
            }
          }
        }
      } else {
        ???
      }
    }

    @virtualize  // conv with batch, bias, and pads
    def convBBP(kernel: TensorR, bias: Option[TensorR], strides: Seq[Int], pads: Seq[Int]): TensorR@diff = shift { (k: TensorR => Unit) =>
      assert(this.isInput || this.d.scalarCount == this.x.scalarCount, "For convBBP, THIS is either input or intermediate stage")
      assert(this.x.rank == 4, "For convBBP, THIS is dim 4: batch, channel, row, col")
      assert(pads.tail.forall(x => x == pads.head), "pads should be the same in all directions")
      val (output, finputOption) = bias match {
        case Some(bias) => x.conv2D_batch(kernel.x, Some(bias.x), strides, pads)
        case None =>       x.conv2D_batch(kernel.x, None, strides, pads)
      }
      val y = TensorR(output); k(y)

      generateRawComment("conv2D back-propagate")
      finputOption match {
        case None => backend.conv2D_batch_grad(this, None, kernel, y, bias, (pads.head, pads.head), (strides.head, strides.last), dilations = (1, 1))
        case Some(finput) => backend.conv2D_batch_grad(this, Some(TensorR(finput)), kernel, y, bias, (pads.head, pads.head), (strides.head, strides.last), dilations = (1, 1))
      }
    }

    @virtualize  // conv with bias and pads
    def convBP(kernel: TensorR, bias: TensorR, strides: Seq[Int], pads: Seq[Int]): TensorR@diff = shift { (k: TensorR => Unit) =>

      assert(this.isInput || this.d.scalarCount == this.x.scalarCount)
      assert(pads.tail.forall(x => x == pads.head), "pads should be the same in all directions")
      val y = TensorR(x conv2D(kernel.x, bias.x, strides, pads))
      k(y)

      // back propagate
      val strideRow = strides.head
      val strideCol = strides.last

      if (pads.sum == 0) {
        val offOutputD = var_new(0)                          // offset for the output, step by 1
        val offKernel = var_new(0)                           // offset for kernel, step by kernel strides(1) -- which kernel
        // looping for the output
        for (kOut <- 0 until y.d.shape(0): Rep[Range]) {
          val offInputR = var_new(0)                         // offset of input, step by input.strides(2) * strideRow
          val sum = var_new(0.0f)                            // collector of bias gradient
          for (row <- 0 until y.d.shape(1): Rep[Range]) {
            val offInputC = var_new(offInputR)               // offset of input, step by strideCol, based on offInputR
            for (col <- 0 until y.d.shape(2): Rep[Range]) {
              val dCurr: Rep[Float] = y.d.data(offOutputD)
              sum += dCurr                                   // collect bias gradient

              // looping for the kernel
              val offInputP = var_new(offInputC)             // offset of input, step by input.strides(1), based on offInputC
              val offKernelR = var_new(offKernel)            // offset of kernel, step by 1, based on offKernel
              for (pane <- 0 until kernel.d.shape(1): Rep[Range]) {
                val offInputKR = var_new(offInputP)                  // offset of input, step by input.strides(2) -- row
                for (kR <- 0 until kernel.d.shape(2): Rep[Range]) {
                  for (kC <- 0 until kernel.d.shape(3): Rep[Range]) {
                    if (!this.isInput) this.d.data(offInputKR + kC) = this.d.data(offInputKR + kC) + dCurr * kernel.x.data(offKernelR)
                    kernel.d.data(offKernelR) = kernel.d.data(offKernelR) + dCurr * this.x.data(offInputKR + kC)
                    offKernelR += 1
                  }
                  offInputKR += this.x.shape.strides(1)
                }
                offInputP += this.x.shape.strides(0)
              }

              offInputC += strideCol
              offOutputD += 1
            }
            offInputR += strideRow * this.x.shape.strides(1)
          }
          bias.d.data(kOut) = bias.d.data(kOut) + sum                            // give value of collector to the bias gradient
          offKernel += kernel.x.shape.strides(0)
        }
      } else {
        val offOutputD = var_new(0)                          // offset for the output, step by 1
        val offKernel  = var_new(0)                          // offset for the kernel, step by kernel strides(1) -- which kernel
        // looping for the output
        for (kOut <- DataLoop(y.d.shape(0))) {
          val InputR = var_new(-pads.head)                   // Row of input, starting from -pads
          val sum = var_new(0.0f)                            // collector of bias gradient
          for (row <- DataLoop(y.d.shape(1))) {
            val InputC = var_new(-pads.head)                 // Col of input, starting from -pads
            for (col <- DataLoop(y.d.shape(2))) {
              val dCurr: Rep[Float] = y.d.data(offOutputD)
              sum += dCurr                                   // collect the bias gradient

              // looping for the kernel
              val offKernelR = var_new(offKernel)            // offset of kernel, step by 1, based on offKernel
              val InputI_pane = var_new[Int](InputR * this.x.shape.strides(1) + InputC)  // offset of input based on pane and index of output
              for (pane <- DataLoop(kernel.d.shape(1))) {
                val InputI_kR = var_new[Int](InputI_pane)                          // offset of input based on InputI_pane and row
                for (kR <- DataLoop(kernel.d.shape(2))) {
                  for (kC <- DataLoop(kernel.d.shape(3))) {
                    if (InputR+kR < 0 || InputR+kR >= this.x.shape(1) || InputC+kC < 0 || InputC+kC >= this.x.shape(2)) ()
                    else {
                      val InputI = InputI_kR + kC                                  // offset of input based on pane and row and col
                      if (!this.isInput) this.d.data(InputI) = this.d.data(InputI) + dCurr * kernel.x.data(offKernelR)
                      kernel.d.data(offKernelR) = kernel.d.data(offKernelR) + dCurr * this.x.data(InputI)
                      offKernelR += 1
                    }
                  }
                  InputI_kR += this.x.shape.strides(1)
                }
                InputI_pane += this.x.shape.strides(0)
              }

              InputC += strideCol
              offOutputD += 1
            }
            InputR += strideRow
          }
          bias.d.data(kOut) = bias.d.data(kOut) + sum
          offKernel += kernel.x.shape.strides(0)
        }
      }

      ()
    }

    @virtualize  // conv, no bias, no padding
    def conv(kernel: TensorR, strideRow: Int, strideCol: Int, tot: Rep[Array[Long]]): TensorR @diff = shift { (k: TensorR => Unit) =>
      assert(this.isInput || this.d.scalarCount == this.x.scalarCount)
      // val timer = Timer2()
      // timer.startTimer
      val y = TensorR(x conv2D(kernel.x, strideRow, strideCol))
      // tot(0) += timer.getElapsedTime
      k(y)
      //y.d.print("conv")

      // val timerBwd = Timer2()
      // TODO think about the loop order
      val offOutputD = var_new(0)                          // offset for the output, step by 1
      val offKernel = var_new(0)                           // offset for kernel, step by kernel strides(1) -- which kernel
      assert(y.d.shape(0) == kernel.x.shape(0))
      // timerBwd.startTimer

      // looping for the output
      for (kOut <- 0 until y.d.shape(0): Rep[Range]) {
        val offInputR = var_new(0)                         // offset of input, step by input.strides(2) * strideRow
        for (row <- 0 until y.d.shape(1): Rep[Range]) {
          val offInputC = var_new(offInputR)               // offset of input, step by strideCol, based on offInputR
          for (col <- 0 until y.d.shape(2): Rep[Range]) {
            val dCurr: Rep[Float] = y.d.data(offOutputD)
            val offInputP = var_new(offInputC)             // offset of input, step by input.strides(1), based on offInputC
            val offKernelR = var_new(offKernel)            // offset of kernel, step by 1, based on offKernel

            // looping for the kernel
            for (pane <- 0 until kernel.d.shape(1): Rep[Range]) {
              val offInputKR = var_new(offInputP)                  // offset of input, step by input.strides(2) -- row
              for (kR <- 0 until kernel.d.shape(2): Rep[Range]) {
                for (kC <- 0 until kernel.d.shape(3): Rep[Range]) {
                  if (!this.isInput) this.d.data(offInputKR + kC) = this.d.data(offInputKR + kC) + dCurr * kernel.x.data(offKernelR)
                  kernel.d.data(offKernelR) = kernel.d.data(offKernelR) + dCurr * this.x.data(offInputKR + kC)
                  offKernelR += 1
                }
                offInputKR += this.x.shape.strides(1)
              }
              offInputP += this.x.shape.strides(0)
            }

            offInputC += strideCol
            offOutputD += 1
          }
          offInputR += strideRow * this.x.shape.strides(1)
        }
        offKernel += kernel.x.shape.strides(0)
      }
      // tot(1) += timerBwd.getElapsedTime
      ()
    }

    @virtualize  // maxpool with kernel size potentially different from strides, and works with batch dimension! can have optional paddings
    def maxPoolBK(kernels: Seq[Int], strides: Seq[Int], pads: Option[Seq[Int]]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val (y, sidx) = this.x.maxPool_k_batch(kernels, strides, pads)
      val ty = TensorR(y)
      k(ty)

      // back propagate
      for (i <- DataLoop(y.scalarCount)) {
        this.d.data(sidx(i)) = ty.d.data(i)
      }
    }

    @virtualize  // maxpool with kernel size potentially different from strides
    def maxPoolK(kernels: Seq[Int], strides: Seq[Int]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val (y, sidx) = this.x.maxPool_k(kernels, strides, None)
      val ty = TensorR(y)
      k(ty)

      // back propagate
      for (i <- DataLoop(y.scalarCount)) {
        this.d.data(sidx(i)) += ty.d.data(i)
      }
    }

    @virtualize
    def maxPool(strideRow: Int, strideCol: Int): TensorR @diff = shift { (k: TensorR => Unit) =>
      val (y, sidx) = this.x.maxPool(strideRow, strideCol)

      val ty = TensorR(y)
      k(ty)

      for (i <- 0 until y.scalarCount: Rep[Range]) {
        this.d.data(sidx(i)) += ty.d.data(i)
      }
    }

    @virtualize
    def concat(dim: Int, others: TensorR*): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = this.x.concat(dim, others.map(t => t.x): _*)
      val ty = TensorR(y)
      k(ty)

      // back propagate
      val higherDims = this.x.shape.take(dim)
      val higherDimsSquashed = higherDims.product

      val totalFrom = this +: others   // put all tensorRs in one Seq for easy handling
      val targetId = var_new(0)        // this is the index of res to read gradient from
      // looping over dims higher than dim, squashed
      for (high <- DataLoop(higherDimsSquashed)) {
        // looping over the concatenation dim
        for (whichTensorR <- totalFrom) {
          // looping over the dimensions lower than or equal to dim (but within an input tensor)
          val stride = if (dim == 0) whichTensorR.x.shape.scalarCount else whichTensorR.x.shape.strides(dim-1)
          val ptrInput = slice(whichTensorR.d.data, high * stride)
          for (lowOrEqual <- DataLoop(stride)) {
            ptrInput(lowOrEqual) += ty.d.data(targetId)
            targetId += 1
          }
        }
      }
    }

    @virtualize
    def global_ave_batch(): TensorR @diff = shift { k: (TensorR => Unit) =>
      val y = this.x.global_ave_batch()
      val ty = TensorR(y)
      k(ty)

      // back propagate
      val scale = 1.0f / this.x.shape.strides(1)
      val offset = var_new(0)                      // offset of this, should step by this.x.strides(2)
      val res_offset = var_new(0)                  // offset of res, should step by 1
      // looping over each batch
      for (batch <- DataLoop(this.x.shape(0))) {
        // looping over each channel
        for (channel <- DataLoop(this.x.shape(1))) {
          // reflect gradient of ty to this, by scale
          val offset_a = var_new(offset)           // offset of this, should step by this.x.strides(3)
          for (i <- DataLoop(this.x.shape(2))) {
            for (j <- DataLoop(this.x.shape(3))) {
              this.d.data(offset_a + j) += ty.d.data(res_offset) * scale
            }
            offset_a += this.x.shape.strides(2)
          }
          offset += this.x.shape.strides(1)
          res_offset += 1
        }
      }
    }

    @virtualize
    def dropout(prob: Float): TensorR @diff = shift { (k: TensorR => Unit) =>
      val (y, noise) = this.x.dropout(prob)
      val ty = TensorR(y)

      k(ty)

      this.d += noise * ty.d
    }

    @virtualize
    def relu(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(this.x.relu(false))
      k(y)

      for (i <- 0 until this.x.scalarCount: Rep[Range]) {
        this.d.data(i) = if (this.x.data(i) < 0.0f) 0.0f else y.d.data(i)
      }
    }

    def print(msg: String = "", derivative: Boolean = false): Unit = {
      this.x.print(msg)
      if (derivative) {
        if (msg == "")
          printf("=========\\n")
        this.d.print(s"derivative $msg")
      }
    }

    def clear_all() = {
      x.clear()
      d.clear()
    }

    def clear_grad() = {
      d.clear()
    }
  }

  object TensorR {
    def apply(a: Tensor, isInput: Boolean = false): TensorR = {
      val d = if (isInput) Tensor.scalar(0.0f) else Tensor.zeros_like(a)
      val res = new TensorR(a, d)
      res.isInput = isInput
      res
    }
    def apply(a: Rep[Array[Float]], dim0: Int, dim1: Int): TensorR = {
      new TensorR(Tensor(a, dim0, dim1), Tensor.zeros(dim0, dim1))
    }

    // def apply(dim0: Int, dim1: Int): TensorR = {
    //   new TensorR(Tensor.zeros(dim0, dim1), Tensor.zeros(dim0, dim1))
    // }

  }

  // change fun signature for memory leak issue (no more returning of array, just update the array provided by the caller)
  // this is in accordance of the destination-passing style
  // the fun take array[array[double]] of size 2, with the first array to be the x, and the second array to be the d
  def FUNc(f: TensorR => Unit): (TensorR => Unit) = { (x:TensorR) =>
    val dims = x.x.shape.toSeq
    val f1 = fun { (x: Rep[Array[Array[Float]]]) =>
      val tensor = new TensorR(Tensor(x(0), dims: _*), Tensor(x(1), dims: _*))
      f(tensor)
    }
    val in = NewArray[Array[Float]](2)
    in(0) = x.x.data; in(1) = x.d.data
    f1(in) // f1 should take Array[Array[Float]] and update the gradient of x
  }

  def FUNm(f: ArrayBuffer[TensorR] => Unit): (ArrayBuffer[TensorR] => Unit) = { (x: ArrayBuffer[TensorR]) =>
    val dims = x.map(_.x.shape.toSeq)
    val f1 = fun { (x: Rep[Array[Array[Float]]]) =>
      val tensors = ArrayBuffer[TensorR]()
      for (u <- (0 until dims.length): Range) {
        tensors.append(new TensorR(Tensor(x(u * 2), dims(u) : _*), Tensor(x(u*2+1), dims(u) : _*)))
      }
      f(tensors)
    }
    val in = NewArray[Array[Float]](2 * dims.length)
    for (u <- (0 until dims.length): Range) {
      in(u*2) = x(u).x.data; in(u*2+1) = x(u).d.data
    }
    f1(in)
  }

  @virtualize
  def IF(c: Rep[Boolean])(a: =>TensorR @diff)(b: =>TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>
    val k1 = FUNc(k)

    if (c) RST(k1(a)) else RST(k1(b))
  }

  @virtualize
  def IFm(c: Rep[Boolean])(a: => ArrayBuffer[TensorR] @diff)(b: => ArrayBuffer[TensorR] @diff): ArrayBuffer[TensorR] @diff = shift { k: (ArrayBuffer[TensorR] => Unit) =>
    val k1 = FUNm(k)
    if (c) RST(k1(a)) else RST(k1(b))
  }

  @virtualize
  def LOOP(init: TensorR)(c: TensorR => Rep[Boolean])(b: TensorR => TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>

    lazy val loop: TensorR => Unit = FUNc { (x: TensorR) =>
      if (c(x)) RST(loop(b(x))) else RST(k(x))
    }
    loop(init)
  }

  def FUNs(f: Rep[Int] => TensorR => Unit): (Rep[Int] => TensorR => Unit) = { (i: Rep[Int]) => (x:TensorR) =>
    val dims = x.x.shape.toSeq
    val f1 = fun { (i: Rep[Int], x: Rep[Array[Array[Float]]]) =>
      val tensor = new TensorR(Tensor(x(0), dims: _*), Tensor(x(1), dims: _*))
      f(i)(tensor)
    }
    val in = NewArray[Array[Float]](2)
    in(0) = x.x.data; in(1) = x.d.data
    f1(i, in)
  }

  @virtualize
  def LOOPS(init: TensorR)(c: Rep[Int])(b: Rep[Int] => TensorR => TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>
    lazy val loop: Rep[Int] => TensorR => Unit = FUNs { (i: Rep[Int]) => (x: TensorR) =>
      if (i < c) { RST(loop(i+1)(b(i)(x))) } else RST(k(x))
    }
    loop(0)(init)
  }

  def FUNsm(f: Rep[Int] => ArrayBuffer[TensorR] => Unit): (Rep[Int] => ArrayBuffer[TensorR] => Unit) = { (i: Rep[Int]) => (x:ArrayBuffer[TensorR]) =>
    val dims = x.map(_.x.shape.seq)
    val f1 = fun { (i: Rep[Int], x: Rep[Array[Array[Float]]]) =>
      val tensors = ArrayBuffer[TensorR]()
      for (u <- (0 until dims.length): Range) {
        tensors.append(new TensorR(Tensor(x(u*2), dims(u) : _*), Tensor(x(u*2+1), dims(u) : _*)))
      }
      f(i)(tensors)
    }
    val in = NewArray[Array[Float]](2 * dims.length)
    for (u <- (0 until dims.length): Range) {
      in(u*2) = x(u).x.data; in(u*2+1) = x(u).d.data
    }
    f1(i, in)
  }

  @virtualize
  def LOOPSM(init: ArrayBuffer[TensorR])(c: Rep[Int])(b: Rep[Int] => ArrayBuffer[TensorR] => ArrayBuffer[TensorR] @diff):
  ArrayBuffer[TensorR] @diff = shift { k: (ArrayBuffer[TensorR] => Unit) =>
    lazy val loop: Rep[Int] => ArrayBuffer[TensorR] => Unit = FUNsm { (i: Rep[Int]) => (x: ArrayBuffer[TensorR]) =>
      if (i < c) {
        RST(loop(i+1)(b(i)(x)))
      } else {
        RST(k(x))
      }
    }
    loop(0)(init)
  }

  def FUN0(f: ((TensorR => Unit) => TensorR => Unit)): ((TensorR => Unit) => TensorR => Unit) = { k1: (TensorR => Unit) => (x: TensorR) =>
    val dims = x.x.shape.toSeq
    val f1 = fun { (t1: Rep[Array[Array[Float]] => Unit], xx: Rep[Array[Array[Float]]]) =>
      val t2: (TensorR => Unit) = { (x: TensorR) =>
        val temp = NewArray[Array[Float]](2)
        temp(0) = x.x.data; temp(1) = x.d.data
        t1(temp)
      }
      val t3: (TensorR => Unit) = f(t2)
      t3(new TensorR(Tensor(xx(0), dims: _*), Tensor(xx(1), dims: _*)))
    }
    val k2: Rep[Array[Array[Float]] => Unit] = fun { (x: Rep[Array[Array[Float]]]) =>
      k1(new TensorR(Tensor(x(0), dims: _*), Tensor(x(1), dims: _*)))
    }
    val temp = NewArray[Array[Float]](2)
    temp(0) = x.x.data; temp(1) = x.d.data
    f1(k2, temp)
  }

  def FUNll(f: (Rep[Int] => (TensorR => Unit) => TensorR => Unit)): (Rep[Int] => (TensorR => Unit) => TensorR => Unit) = {i: Rep[Int] => k1: (TensorR => Unit) => (x: TensorR) =>

    val dims = x.x.shape.toSeq

    val f1 = fun { (i: Rep[Int], t1: Rep[Array[Array[Float]] => Unit], xx: Rep[Array[Array[Float]]]) =>
      val t2: (TensorR => Unit) = { (x: TensorR) =>
        val temp = NewArray[Array[Float]](2)
        temp(0) = x.x.data; temp(1) = x.d.data
        t1(temp)
      }
      val t3: (TensorR => Unit) = f(i)(t2)
      t3(new TensorR(Tensor(xx(0), dims: _*), Tensor(xx(1), dims: _*)))
    }

    val k2: Rep[Array[Array[Float]] => Unit] = fun { (x: Rep[Array[Array[Float]]]) =>
      k1(new TensorR(Tensor(x(0), dims: _*), Tensor(x(1), dims: _*)))
    }
    val temp = NewArray[Array[Float]](2)
    temp(0) = x.x.data; temp(1) = x.d.data
    f1(i, k2, temp)
  }

  def FUNl(f: (Rep[Int] => (TensorR => Unit) => TensorR => Unit)): (Rep[Int] => (TensorR => Unit) => TensorR => Unit) = {i: Rep[Int] => k1: (TensorR => Unit) => (x: TensorR) =>

    val dims = x.x.shape.toSeq

    val f1 = fun { (i: Rep[Int], t1: Rep[((Array[Float], Array[Float])) => Unit], x0: Rep[Array[Float]], x1: Rep[Array[Float]]) =>
      val t2: (TensorR => Unit) = { (x: TensorR) =>
        t1(x.x.data, x.d.data)
      }
      val t3: (TensorR => Unit) = f(i)(t2)
      t3(new TensorR(Tensor(x0, dims: _*), Tensor(x1, dims: _*)))
    }

    val k2: Rep[((Array[Float], Array[Float])) => Unit] = fun { (x1: Rep[Array[Float]], x2: Rep[Array[Float]]) =>
      k1(new TensorR(Tensor(x1, dims: _*), Tensor(x2, dims: _*)))
    }
    f1(i, k2, x.x.data, x.d.data)
  }

  @virtualize
  def LOOPL(init: TensorR)(c: Rep[Int])(b: Rep[Int] => TensorR => TensorR @diff): TensorR @diff = shift { k: (TensorR => Unit) =>
    lazy val loop: Rep[Int] => (TensorR => Unit) => TensorR => Unit = FUNl{ (gc: Rep[Int]) => (k: TensorR => Unit) => (x: TensorR) =>
      def sh_loop: (Rep[Int] => TensorR @diff) = (i: Rep[Int]) => shift{(k: TensorR => Unit) => loop(i)(k)(x)}
      RST(k (IF(gc < c) { b(gc)(sh_loop(gc+1)) } { init }) )
    }
    loop(0)(k)(init)
  }

  @virtualize
  def LOOPT(start: Rep[Int])(init: TensorR)(lch: Rep[Array[Int]], rch: Rep[Array[Int]])(b: (TensorR, TensorR, Rep[Int]) => TensorR @diff): TensorR @diff = shift {
    k: (TensorR => Unit) =>

      lazy val tree: Rep[Int] => (TensorR => Unit) => TensorR => Unit = FUNl{ (i: Rep[Int]) => (k: TensorR => Unit) => (x: TensorR) =>
        def sh_tree: (Rep[Int] => TensorR @diff) = (i: Rep[Int]) => shift{(k: TensorR => Unit) => tree(i)(k)(x)}
        RST(k( IF(i >= 0) { b(sh_tree(lch(i)), sh_tree(rch(i)), i) } { init } ))
      }
      tree(start)(k)(init)
  }

  def FUNlm(f: (Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit)):
  (Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit) = {i: Rep[Int] => k1: (ArrayBuffer[TensorR] => Unit) => (x: ArrayBuffer[TensorR]) =>

    val length = x.length
    val dims = x.map(_.x.shape.toSeq)
    val f1 = fun { (i: Rep[Int], t1: Rep[Array[Array[Float]] => Unit], xx: Rep[Array[Array[Float]]]) =>
      val t2: (ArrayBuffer[TensorR] => Unit) = { (x: ArrayBuffer[TensorR]) =>
        val aa = NewArray[Array[Float]](2*length)
        for (u <- (0 until length): Range) {
          aa(u*2) = x(u).x.data; aa(u*2+1) = x(u).d.data
        }
        t1(aa)
      }
      val t3: (ArrayBuffer[TensorR] => Unit) = f(i)(t2)
      val tensors = ArrayBuffer[TensorR]()
      for (u <- (0 until length): Range) {
        tensors.append(new TensorR(Tensor(xx(u*2), dims(u): _*), Tensor(xx(u*2+1), dims(u): _*)))
      }
      t3(tensors)
    }
    val k2: Rep[Array[Array[Float]] => Unit] = fun { (x: Rep[Array[Array[Float]]]) =>
      val tensors = ArrayBuffer[TensorR]()
      for (u <- (0 until length): Range) {
        tensors.append(new TensorR(Tensor(x(u*2), dims(u): _*), Tensor(x(u*2+1), dims(u): _*)))
      }
      k1(tensors)
    }
    val arrays = NewArray[Array[Float]](2*length)
    for (u <- (0 until length): Range) {
      arrays(u*2) = x(u).x.data; arrays(u*2+1) = x(u).d.data
    }
    f1(i, k2, arrays)
  }

  // type Fun6Array = ((Array[Float], Array[Float], Array[Float], Array[Float], Array[Float], Array[Float])) => Unit
  // type RAF = Rep[Array[Float]]

  // def FUNlx(f: (Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit)):
  // (Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit) = {i: Rep[Int] => k1: (ArrayBuffer[TensorR] => Unit) => (x: ArrayBuffer[TensorR]) =>

  //   val length = x.length
  //   val dims = x.map(_.x.shape.toSeq)
  //   x.length match {
  //     case 3 =>
  //       val f1 = fun { (i: Rep[Int], t1: Rep[Fun6Array], x0: RAF, x1: RAF, x2: RAF, x3: RAF, x4: RAF, x5: RAF) =>
  //         val t2 = { x: ArrayBuffer[TensorR] =>
  //           t1((x(0).x.data, x(0).d.data, x(1).x.data, x(1).d.data, x(2).x.data, x(2).d.data))
  //         }
  //         val t3: (ArrayBuffer[TensorR] => Unit) = f(i)(t2)
  //         val tensors = ArrayBuffer(
  //           new TensorR(Tensor(x0, dims(0): _*), Tensor(x1, dims(0): _*)),
  //           new TensorR(Tensor(x2, dims(1): _*), Tensor(x3, dims(1): _*)),
  //           new TensorR(Tensor(x4, dims(2): _*), Tensor(x5, dims(2): _*)),
  //         )
  //         t3(tensors)
  //       }
  //       val k2: Rep[Fun6Array] = fun { (x0: RAF, x1: RAF, x2: RAF, x3: RAF, x4: RAF, x5: RAF) =>
  //         val tensors = ArrayBuffer(
  //           new TensorR(Tensor(x0, dims(0): _*), Tensor(x1, dims(0): _*)),
  //           new TensorR(Tensor(x2, dims(1): _*), Tensor(x3, dims(1): _*)),
  //           new TensorR(Tensor(x4, dims(2): _*), Tensor(x5, dims(2): _*)),
  //         )
  //         k1(tensors)
  //       }
  //       f1(i, k2, x(0).x.data, x(0).d.data, x(1).x.data, x(1).d.data, x(2).x.data, x(2).d.data)
  //     case 2 => ???
  //   }
  // }

  @virtualize
  def LOOPLM(start: Rep[Int])(init: ArrayBuffer[TensorR])(c: Rep[Int])(b: Rep[Int] => ArrayBuffer[TensorR] => ArrayBuffer[TensorR] @diff):
  ArrayBuffer[TensorR] @diff = shift { k: (ArrayBuffer[TensorR] => Unit) =>

    lazy val loop: Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit = FUNlm { (i: Rep[Int]) => (k: ArrayBuffer[TensorR] => Unit) => (x: ArrayBuffer[TensorR]) =>

      def sh_loop: (Rep[Int] => ArrayBuffer[TensorR] @diff) = (i: Rep[Int]) => shift{ (k: ArrayBuffer[TensorR] => Unit) => loop(i)(k)(x) }

      RST(k( IFm(i < c) { b(i)(sh_loop(i+1)) } { init } ))
    }
    loop(start)(k)(init)
  }

  @virtualize
  def LOOPTM(start: Rep[Int])(init: ArrayBuffer[TensorR])(lch: Rep[Array[Int]], rch: Rep[Array[Int]])
  (b: (ArrayBuffer[TensorR], ArrayBuffer[TensorR], Rep[Int]) => ArrayBuffer[TensorR] @diff): ArrayBuffer[TensorR] @diff = shift { k: (ArrayBuffer[TensorR] => Unit) =>

      lazy val tree: Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit = FUNlm { (i: Rep[Int]) => (k: ArrayBuffer[TensorR] => Unit) => (x: ArrayBuffer[TensorR]) =>

        def sh_tree: (Rep[Int] => ArrayBuffer[TensorR] => ArrayBuffer[TensorR] @diff) = (i: Rep[Int]) => (x: ArrayBuffer[TensorR]) => shift{(k: ArrayBuffer[TensorR] => Unit) => tree(i)(k)(x)}

        RST(k( IFm (i >= 0) { b(sh_tree(lch(i))(init), sh_tree(rch(i))(init), i) } { init } ))
      }
      tree(start)(k)(init)
  }

  def gradR(f: TensorR => TensorR @diff)(x: Tensor): Tensor = {
    val x1 = new TensorR(x, Tensor.zeros(x.shape(0)))
    reset {
      val y = f(x1)
      y.d.setAsOne()
      ()
    }
    x1.d
  }

  // same as gradR function, except that we return the final result of f, not the gradient of input
  // gradient of input is supposed to be dummy value here
  // gradient of useful tensors are in closure, and can be accessed directly from outside of this function
  def gradR_loss(f: TensorR => TensorR @diff)(x: Tensor): Tensor = {
    val x1 = TensorR(x) // this should be a dummy tensor
    val result = Tensor.zeros(1)                  // this should be the loss
    reset {
      val y = f(x1)
      assert(y.x.scalarCount == 1, s"Loss function must return a Tensor of size 1, got ${y.x.scalarCount}")
      y.d.setAsOne()
      result.copy_data(y.x)
      ()
    }
    result
  }

  def getMallocAddr(): Rep[Long] = {
    unchecked[Long]("(long)mallocAddr")
  }

  def resetMallocAddr(addr: Rep[Long]) = {
    unchecked[Unit]("memset((void*)", addr, ", 0, ", getMallocAddr() - addr, ")")
    unchecked[Unit]("mallocAddr = (void*)", addr)
  }
}

trait TensorDslCublas extends TensorDsl with GPUOps {

  protected def cudaMemcpyHostToDevice(dest: Rep[Array[Float]], src: Rep[Array[Float]], n: Int): Rep[Unit] =
    unchecked[Unit]("CUDA_CALL(cudaMemcpy(", dest, ", ", src, ", ", n, " * sizeof(float), cudaMemcpyHostToDevice))")

  protected def cudaMemcpyDeviceToHost(dest: Rep[Array[Float]], src: Rep[Array[Float]], n: Int): Rep[Unit] =
    unchecked[Unit]("CUDA_CALL(cudaMemcpy(", dest, ", ", src, ", ", n, " * sizeof(float), cudaMemcpyDeviceToHost))")

  protected def cudaMemcpyDeviceToDevice(dest: Rep[Array[Float]], src: Rep[Array[Float]], n: Int): Rep[Unit] =
    unchecked[Unit]("CUDA_CALL(cudaMemcpy(", dest, ", ", src, ", ", n, " * sizeof(float), cudaMemcpyDeviceToDevice))")

  // NOTE: `cudaMemset` is not very useful because it only works with an integer array/value.
  protected def cudaMemset(array: Rep[Array[Int]], value: Rep[Int], n: Int): Rep[Unit] =
    unchecked[Unit]("CUDA_CALL(cudaMemset((void **)&", array, ", ", value, ", ", n, " * sizeof(int)))")

  protected def cublasSetPointerModeDevice(): Rep[Unit] =
    unchecked[Unit]("cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE)")

  protected def cublasSetPointerModeHost(): Rep[Unit] =
    unchecked[Unit]("cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST)")

  /**
    * Defines tensor backend transfer operations.
    */
  class TensorTransferOps(t: Tensor) {
    // Get a CPU-allocated copy of this tensor.
    def toCPU(): Tensor = {
      generateRawComment("'toCPU' invocation.")
      val res = BackendCPU().mallocArray[Float](t.scalarCount)
      cudaMemcpyDeviceToHost(res, t.data, t.scalarCount)
      Tensor(res, t.shape: _*)
    }

    // Get a GPU-allocated copy of this tensor.
    def toGPU(): Tensor = {
      generateRawComment("'toGPU' invocation.")
      val res = BackendGPU.mallocArray[Float](t.scalarCount)
      cudaMemcpyHostToDevice(res, t.data, t.scalarCount)
      Tensor(res, t.shape: _*)
    }
  }
  implicit def tensorToTransferOps(t: Tensor) = new TensorTransferOps(t)

  class TensorRTransferOps(t: TensorR) {
    def toCPU(): TensorR = new TensorR(t.x.toCPU(), t.d.toCPU())
    def toGPU(): TensorR = new TensorR(t.x.toGPU(), t.d.toGPU())
  }
  implicit def tensorRToTransferOps(t: TensorR) = new TensorRTransferOps(t)

  /**
    * cuBLAS tensor operation backend. WIP.
    */
  class BackendCublas protected() extends Backend {
    override def setup(): Unit = generateRawCode(
      """cublasHandle_t cublasHandle;
        |CUBLAS_CALL(cublasCreate(&cublasHandle));
        |CUDA_CALL(cudaMalloc(&gpuMallocAddr, HEAP_SIZE));
      """.stripMargin)

    override def cleanup(): Unit = generateRawCode(
      """CUBLAS_CALL(cublasDestroy(cublasHandle));
        |CUDA_CALL(cudaFree(gpuMallocAddr));
      """.stripMargin)

    override def mallocArray[T: Manifest](length: Int): Rep[Array[T]] = NewGPUArray[T](length)

    override def copyFloatArray(dest: Rep[Array[Float]], src: Rep[Array[Float]], length: Int): Unit =
      cudaMemcpyDeviceToDevice(dest, src, length)

    override def makeTensor(dims: Seq[Int], scalars: Float*): Tensor = {
      BackendCPU().makeTensor(dims, scalars: _*).toGPU()
    }

    override def fill(dims: Seq[Int], value: Rep[Float]): Tensor = {
      // TOOO: Compare performance with GPU allocation + `fillInPlace`.
      BackendCPU().fill(dims, value).toGPU()
    }

    override def fillWithBias(dims: Seq[Int], bias: Tensor, dim: Int): Tensor = {
      BackendCPU().fillWithBias(dims, bias.toCPU(), dim).toGPU()
    }

    override def fillInPlace(x: Tensor, value: Rep[Float]): Unit = {
      // TODO: Consider different grid/block parameters.
      unchecked[Unit](s"arrayFill<<<${x.scalarCount}, 1>>>(", x.data, ", ", value, ")")
    }

    override def copyTensorData(dest: Tensor, src: Tensor): Unit = {
      assert(dest.scalarCount == src.scalarCount,
        s"Tensors do not have same shape: ${dest.shape.seq}, ${src.shape.seq}")
      cudaMemcpyDeviceToDevice(dest.data, src.data, dest.scalarCount)
    }

    // Reference: https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-dot
    // NOTE: `sdot` fails when the cuBLAS pointer mode is host (as opposed to device).
    // Investigate performance impact.
    def sdot(n: Int, a: Rep[Array[Float]], b: Rep[Array[Float]], result: Rep[Array[Float]]) = {
      cublasSetPointerModeDevice()
      unchecked[Unit]("CUBLAS_CALL(cublasSdot(cublasHandle, ", n, ",", a, ",", 1, ",", b, ",", 1, ",", result, "))")
      cublasSetPointerModeHost()
    }

    override def vectorVectorDot(x: Tensor, y: Tensor): Tensor = {
      val res = mallocArray[Float](1)
      sdot(x.scalarCount, x.data, y.data, res)
      Tensor(res, 1)
    }

    // Reference: https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
    def sgemv(m: Int, n: Int, matrix: Rep[Array[Float]], vector: Rep[Array[Float]], result: Rep[Array[Float]]) = {
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
    def sgemm(m: Int, n: Int, k: Int, a: Rep[Array[Float]], b: Rep[Array[Float]], result: Rep[Array[Float]]) = {
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, ",
        m, ",", n, ",", k, ",", one, ",",
        a, ",", k, ",", b, ",", n, ",", zero, ",", result, ",", m, "))")
    }

    override def matrixMatrixDot(x: Tensor, y: Tensor): Tensor = {
      val m = x.shape(0)
      val n = y.shape(1)
      val k = y.shape(0)
      val res = mallocArray[Float](m * n)
      sgemm(m, n, k, x.data, y.data, res)
      Tensor(res, m, n)
    }

    // Compute broadcasting strides.
    // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorIterator.cpp#L396
    // TODO: Generalize for different-ranked tensors. Currently, broadcasting works only for same-rank tensors.
    def getBroadcastingStrides(shape: Dimensions): Seq[Int] = {
      shape.strides.zipWithIndex.map { case (s, i) =>
        if (shape(i) == 1) 0 else s
      }
    }

    def launchUnaryKernel(res: Tensor, x: Tensor)(op: String => Seq[Any]): Unit = {
      assert(res.shape == x.shape, s"Unary kernel incompatible shapes: ${res.shape.seq}, ${x.shape.seq}")

      // Store shapes as local variables.
      val resShape = res.shape
      // Convert shapes to Rep[Array[Int]].
      val resDims = Array(resShape.map(unit(_)): _*)
      // Compute strides.
      val strides = NewArray[Array[Int]](2)
      val tmp = Array(getBroadcastingStrides(resShape).map(unit(_)): _*)
      strides(0) = tmp
      strides(1) = tmp
      // Launch kernel.
      // NOTE: Hacky way to propagate `Rep[Float]` as an argument to `unchecked`.
      unchecked[Unit](
        Seq("{\n" +
        "OffsetCalculator<2> calc(", resShape.length, ",", resDims, ",", strides, "); \n" +
        "launch_kernel<128, 4>(", resShape.scalarCount, ", [=]__device__(int idx) {\n" +
        "  auto offsets = calc.get(idx);\n" +
        "  float* out = (float*)&", res.data, "[offsets[0]];\n" +
        "  float* in = (float*)&", x.data, "[offsets[1]];\n" +
        "  *out = ") ++ op("(*in)") ++ Seq(";\n" +
        "});\n" +
        "}"): _*)
    }

    def elementwiseUnaryOp(x: Tensor)(op: String => Seq[Any]): Tensor = {
      val resData = mallocArray[Float](x.scalarCount)
      val res = Tensor(resData, x.shape: _*)
      launchUnaryKernel(res, x)(op)
      res
    }

    def elementwiseInplaceUnaryOp(x: Tensor)(op: String => Seq[Any]): Unit = {
      launchUnaryKernel(x, x)(op)
    }

    def launchBinaryKernel(res: Tensor, x: Tensor, y: Tensor)(op: (String, String) => String): Unit = {
      // Store shapes as local variables.
      val resShape = res.shape
      val xShape = x.shape
      val yShape = y.shape
      // Convert shapes to Rep[Array[Int]].
      val resDims = Array(resShape.map(unit(_)): _*)
      val xDims = Array(xShape.map(unit(_)): _*)
      val yDims = Array(yShape.map(unit(_)): _*)
      // Compute strides.
      val strides = NewArray[Array[Int]](3)
      strides(0) = Array(getBroadcastingStrides(resShape).map(unit(_)): _*)
      strides(1) = Array(getBroadcastingStrides(xShape).map(unit(_)): _*)
      strides(2) = Array(getBroadcastingStrides(yShape).map(unit(_)): _*)
      // Launch kernel.
      unchecked[Unit](
        "{\n" +
        "OffsetCalculator<3> calc(", resShape.length, ",", resDims, ",", strides, "); \n" +
        "launch_kernel<128, 4>(", resShape.scalarCount, ", [=]__device__(int idx) {\n" +
        "  auto offsets = calc.get(idx);\n" +
        "  float* out = (float*)&", res.data, "[offsets[0]];\n" +
        "  float* in1 = (float*)&", x.data, "[offsets[1]];\n" +
        "  float* in2 = (float*)&", y.data, "[offsets[2]];\n" +
        s"  *out = ${op("(*in1)", "(*in2)")};\n" +
        "});\n" +
        "}")
    }

    def elementwiseBinaryOp(x: Tensor, y: Tensor)(op: (String, String) => String): Tensor = {
      Tensor.dimBroadcast(x.shape, y.shape) match {
        case None => throw new IllegalArgumentException(s"Shapes cannot be broadcasted: ${x.shape.seq}, ${y.shape.seq}")
        case Some((xShape, yShape, resShape)) =>
          val resData = mallocArray[Float](resShape.scalarCount)
          val res = Tensor(resData, resShape: _*)
          launchBinaryKernel(res, x, y)(op)
          res
      }
    }

    def elementwiseInplaceBinaryOp(x: Tensor, y: Tensor)(op: (String, String) => String): Unit = {
      Tensor.dimBroadcast(x.shape, y.shape) match {
        case None => throw new IllegalArgumentException(s"Shapes cannot be broadcasted: ${x.shape.seq}, ${y.shape.seq}")
        case Some((xShape, yShape, resShape)) =>
          assert(x.shape == resShape, s"Output shape ${xShape.seq} does not match broadcast shape: ${resShape.seq}")
          launchBinaryKernel(x, x, y)(op)
      }
    }

    override def +(x: Tensor, y: Rep[Float]): Tensor = elementwiseUnaryOp(x)(s => Seq(s + " + ", y))
    override def +(x: Tensor, y: Tensor): Tensor = elementwiseBinaryOp(x, y) { _ + " + " + _ }

    override def +=(x: Tensor, y: Rep[Float]): Unit = elementwiseInplaceUnaryOp(x)(s => Seq(s + " + ", y))
    override def +=(x: Tensor, y: Tensor): Unit = elementwiseInplaceBinaryOp(x, y) { _ + " + " + _ }

    override def -(x: Tensor, y: Rep[Float]): Tensor = elementwiseUnaryOp(x)(s => Seq(s + " - ", y))
    override def -(x: Tensor, y: Tensor): Tensor = elementwiseBinaryOp(x, y) { _ + " - " + _ }

    override def -=(x: Tensor, y: Rep[Float]): Unit = elementwiseInplaceUnaryOp(x)(s => Seq(s + " - ", y))
    override def -=(x: Tensor, y: Tensor): Unit = elementwiseInplaceBinaryOp(x, y) { _ + " - " + _ }

    override def *(x: Tensor, y: Rep[Float]): Tensor = elementwiseUnaryOp(x)(s => Seq(s + " * ", y))
    override def *(x: Tensor, y: Tensor): Tensor = elementwiseBinaryOp(x, y) { _ + " * " + _ }

    override def *=(x: Tensor, y: Rep[Float]): Unit = elementwiseInplaceUnaryOp(x)(s => Seq(s + " * ", y))
    override def *=(x: Tensor, y: Tensor): Unit = elementwiseInplaceBinaryOp(x, y) { _ + " * " + _ }

    override def /(x: Tensor, y: Rep[Float]): Tensor = elementwiseUnaryOp(x)(s => Seq(s + " / ", y))
    override def /(x: Tensor, y: Tensor): Tensor = elementwiseBinaryOp(x, y) { _ + " / " + _ }

    override def /=(x: Tensor, y: Rep[Float]): Unit = elementwiseInplaceUnaryOp(x)(s => Seq(s + " / ", y))
    override def /=(x: Tensor, y: Tensor): Unit = elementwiseInplaceBinaryOp(x, y) { _ + " / " + _ }

    // TODO: Implement cuBLAS `conv2D_batch` in terms of primitive ops.
    // Reuse CPU implementation?
    override def conv2D_batch(input: Tensor, kernel: Tensor, bias: Option[Tensor], strides: Seq[Int], pads: Seq[Int]): (Tensor, Option[Tensor]) = ???
    override def conv2D_batch_grad(input: TensorR, finput: Option[TensorR], filter: TensorR, res: TensorR, bias: Option[TensorR] = None,
                                   padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int)): Unit = ???
  }

  object BackendCublas {
    def apply() = new BackendCublas
  }

  // Define default GPU backend.
  def BackendGPU: Backend = BackendCublas()
  backend = BackendGPU
}

trait TensorDslCudnn extends TensorDslCublas {
  /**
    * cuDNN tensor operation backend. WIP.
    * Extends `BackendCublas` to leverage cuBLAS primitives.
    */
  class BackendCudnn protected() extends BackendCublas {
    override def setup(): Unit = {
      super.setup()
      generateRawCode("cudnnHandle_t cudnnHandle;\nCUDNN_CALL(cudnnCreate(&cudnnHandle));")
    }

    override def cleanup(): Unit = {
      super.cleanup()
      generateRawCode("CUDNN_CALL(cudnnDestroy(cudnnHandle));")
    }

    // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnAddTensor
    // Note: this function performs in-place addition for `res`.
    def cudnnAddBiasTensor(bias: Tensor, res: Tensor): Unit = {
      assert(bias.rank == 1, "Bias must have rank 1")
      assert(res.rank == 4, "Currently, only rank 4 tensors are supported")
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t bias_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    1, ${bias.shape(0)}, 1, 1));
          |
          |cudnnTensorDescriptor_t out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${res.shape(0)}, ${res.shape(1)}, ${res.shape(2)}, ${res.shape(3)}));
          |
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnAddTensor(\n" +
          "    cudnnHandle, ", one, ", bias_desc, ", bias.data, ", ", one, ", out_desc, ", res.data, "));\n" +
          "}"): _*
      )
    }

    // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionForward
    def cudnnConvolutionForward(input: Tensor, filter: Tensor, res: Tensor, bias: Option[Tensor] = None,
                                padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int)): Unit = {
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t in_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${input.shape(0)}, ${input.shape(1)}, ${input.shape(2)}, ${input.shape(3)}));
          |
          |cudnnFilterDescriptor_t filt_desc;
          |CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
          |CUDNN_CALL(cudnnSetFilter4dDescriptor(
          |    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
          |    ${filter.shape(0)}, ${filter.shape(1)}, ${filter.shape(2)}, ${filter.shape(3)}));
          |
          |cudnnTensorDescriptor_t out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${res.shape(0)}, ${res.shape(1)}, ${res.shape(2)}, ${res.shape(3)}));
          |
          |cudnnConvolutionDescriptor_t conv_desc;
          |CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
          |CUDNN_CALL(cudnnSetConvolution2dDescriptor_v5(
          |    conv_desc,
          |    ${padding._1}, ${padding._2}, ${strides._1}, ${strides._2}, ${dilations._1}, ${dilations._2},
          |    CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
          |
          |// Algorithm.
          |cudnnConvolutionFwdAlgo_t algo;
          |CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
          |    cudnnHandle,
          |    in_desc, filt_desc, conv_desc, out_desc,
          |    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
          |
          |// Workspace.
          |size_t ws_size;
          |CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
          |    cudnnHandle, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));
          |float *ws_data;
          |CUDA_CALL(cudaMalloc(&ws_data, ws_size));
          |""".stripMargin) ++
        Seq(
          "// Execute convolution.\n" +
          "CUDNN_CALL(cudnnConvolutionForward(\n" +
          "    cudnnHandle,\n" +
          "    ", one, ", in_desc, ", input.data, ", filt_desc, ", filter.data, ",\n" +
          "    conv_desc, algo, ws_data, ws_size,\n" +
          "    ", zero, ", out_desc, ", res.data, "));\n" +
          "}"): _*
      )
    }

    // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardBias
    // This is effectively the gradient of `cudnnAddBiasTensor`.
    def cudnnConvolutionBackwardBias(biasGrad: Tensor, resGrad: Tensor): Unit = {
      assert(biasGrad.rank == 1, "Bias gradient must have rank 1")
      assert(resGrad.rank == 4, "Convolution result gradient must have rank 4")
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t grad_bias_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    1, ${biasGrad.shape(0)}, 1, 1));
          |
          |cudnnTensorDescriptor_t grad_out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${resGrad.shape(0)}, ${resGrad.shape(1)}, ${resGrad.shape(2)}, ${resGrad.shape(3)}));
          |
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnConvolutionBackwardBias(\n" +
          "    cudnnHandle, ", one, ", grad_out_desc, ", resGrad.data, ",\n",
          "    ", zero, ", grad_bias_desc, ", biasGrad.data, "));\n" +
          "}"): _*
      )
    }

    // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardData
    def cudnnConvolutionBackwardData(inputGrad: Tensor, filter: Tensor, resGrad: Tensor,
                                     padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int)): Unit = {
      assert(resGrad.rank == 4, "Convolution result gradient must have rank 4")
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnFilterDescriptor_t filt_desc;
          |CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
          |CUDNN_CALL(cudnnSetFilter4dDescriptor(
          |    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
          |    ${filter.shape(0)}, ${filter.shape(1)}, ${filter.shape(2)}, ${filter.shape(3)}));
          |
          |cudnnTensorDescriptor_t grad_in_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${inputGrad.shape(0)}, ${inputGrad.shape(1)}, ${inputGrad.shape(2)}, ${inputGrad.shape(3)}));
          |
          |cudnnTensorDescriptor_t grad_out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${resGrad.shape(0)}, ${resGrad.shape(1)}, ${resGrad.shape(2)}, ${resGrad.shape(3)}));
          |
          |cudnnConvolutionDescriptor_t conv_desc;
          |CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
          |CUDNN_CALL(cudnnSetConvolution2dDescriptor_v5(
          |    conv_desc,
          |    ${padding._1}, ${padding._2}, ${strides._1}, ${strides._2}, ${dilations._1}, ${dilations._2},
          |    CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
          |
          |// Algorithm.
          |cudnnConvolutionBwdDataAlgo_t algo;
          |CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(
          |    cudnnHandle,
          |    filt_desc, grad_out_desc, conv_desc, grad_in_desc,
          |    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algo));
          |
          |// Workspace.
          |size_t ws_size;
          |CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
          |    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
          |float *ws_data;
          |CUDA_CALL(cudaMalloc(&ws_data, ws_size));
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnConvolutionBackwardData(\n" +
          "    cudnnHandle,\n" +
          "    ", one, ", filt_desc, ", filter.data, ", grad_out_desc, ", resGrad.data, ",\n" +
          "    conv_desc, algo, ws_data, ws_size,\n" +
          "    ", zero, ", grad_in_desc, ", inputGrad.data, "));\n" +
          "}"): _*
      )
    }

    // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardFilter
    def cudnnConvolutionBackwardFilter(filterGrad: Tensor, input: Tensor, resGrad: Tensor,
                                       padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int)): Unit = {
      assert(resGrad.rank == 4, "Convolution result gradient must have rank 4")
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnFilterDescriptor_t grad_filt_desc;
          |CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
          |CUDNN_CALL(cudnnSetFilter4dDescriptor(
          |    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
          |    ${filterGrad.shape(0)}, ${filterGrad.shape(1)}, ${filterGrad.shape(2)}, ${filterGrad.shape(3)}));
          |
          |cudnnTensorDescriptor_t grad_out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${resGrad.shape(0)}, ${resGrad.shape(1)}, ${resGrad.shape(2)}, ${resGrad.shape(3)}));
          |
          |cudnnTensorDescriptor_t in_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${input.shape(0)}, ${input.shape(1)}, ${input.shape(2)}, ${input.shape(3)}));
          |
          |cudnnConvolutionDescriptor_t conv_desc;
          |CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
          |CUDNN_CALL(cudnnSetConvolution2dDescriptor_v5(
          |    conv_desc,
          |    ${padding._1}, ${padding._2}, ${strides._1}, ${strides._2}, ${dilations._1}, ${dilations._2},
          |    CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
          |
          |// Algorithm.
          |cudnnConvolutionBwdFilterAlgo_t algo;
          |CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(
          |    cudnnHandle,
          |    in_desc, grad_out_desc, conv_desc, grad_filt_desc,
          |    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algo));
          |
          |// Workspace.
          |size_t ws_size;
          |CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
          |    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
          |float *ws_data;
          |CUDA_CALL(cudaMalloc(&ws_data, ws_size));
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnConvolutionBackwardFilter(\n" +
          "    cudnnHandle,\n" +
          "    ", one, ", in_desc, ", input.data, ", grad_out_desc, ", resGrad.data, ",\n" +
          "    conv_desc, algo, ws_data, ws_size,\n" +
          "    ", zero, ", grad_filt_desc, ", filterGrad.data, "));\n" +
          "}"): _*
      )
    }

    override def conv2D_batch(input: Tensor, kernel: Tensor, bias: Option[Tensor], strides: Seq[Int], pads: Seq[Int]): (Tensor, Option[Tensor]) = {
      // TODO: Dedupe assertions/shape calculations with CPU implementation.
      assert(input.rank == 4, "Input must be 4-D (first dimension is batch size)")
      assert(kernel.rank == 4, "Kernel must be 4-D")
      bias match {
        case Some(bias) =>
          assert(bias.rank == 1, s"Bias should be 1-D, got ${bias.shape}")
          assert(bias.shape(0) == kernel.shape(0), "Bias length must equal number of out-channels")
        case None => ()
      }
      assert(kernel.shape(1) == input.shape(1), "In-channel count mismatch: input.shape[1] should match kernel.shape[1]")
      assert(input.shape(2) >= kernel.shape(2) && input.shape(3) >= kernel.shape(3), "Image too small for conv")

      assert(strides.size == 2, "Strides should have length 2: [strideRow, strideColumn]")
      assert(pads.size == 4, "Pads should have length 4: [padTop, padBottom, padLeft, padRight]")
      val ((strideRow:Int) :: (strideCol:Int) :: Nil) = strides.take(2).toList
      val ((padUp:Int) :: (padDown:Int) :: (padLeft:Int) :: (padRight:Int) :: Nil) = pads.take(4).toList
      assert(strideRow >= 1, "Row stride must be at least 1")
      assert(strideCol >= 1, "Column stride must be at least 1")
      assert(padUp == padDown && padUp == padLeft && padUp == padRight, "All paddings must be equal (for now)")

      // Execute `cudnnConvolutionForward`.
      val resWidth = convSize(input.shape(2) + padLeft + padRight, kernel.shape(2), strideRow)
      val resHeight = convSize(input.shape(3) + padUp + padDown, kernel.shape(3), strideCol)
      val resShape = Seq(input.shape(0), kernel.shape(0), resWidth, resHeight)
      val resData = mallocArray[Float](resShape.product)
      val res = Tensor(resData, resShape: _*)
      cudnnConvolutionForward(input, kernel, res, padding = (padUp, padLeft), strides = (strideCol, strideRow), dilations = (1, 1))

      // If bias is defined, execute `cudnnAddBiasTensor`.
      bias match {
        case None =>
        case Some(bias) => cudnnAddBiasTensor(bias, res)
      }
      (res, None)
    }

    @virtualize
    override def conv2D_batch_grad(input: TensorR, finput: Option[TensorR], filter: TensorR, res: TensorR, bias: Option[TensorR] = None,
                                   padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int)): Unit = {
      // Peephole-optimization: use uninitialized arrays rather than allocating to zero.
      val inputGrad = Tensor(mallocArray[Float](input.d.scalarCount), input.d.shape: _*)
      val filterGrad = Tensor(mallocArray[Float](filter.d.scalarCount), filter.d.shape: _*)
      cudnnConvolutionBackwardData(inputGrad, filter.x, res.d, padding, strides, dilations)
      cudnnConvolutionBackwardFilter(filterGrad, input.x, res.d, padding, strides, dilations)
      input.d += inputGrad
      filter.d += filterGrad
      bias match {
        case None =>
        case Some(bias) =>
          val biasGrad = Tensor(mallocArray[Float](bias.d.scalarCount), bias.d.shape: _*)
          cudnnConvolutionBackwardBias(biasGrad, res.d)
          bias.d += biasGrad
      }
    }
  }

  object BackendCudnn {
    def apply() = new BackendCudnn
  }

  // Define default GPU backend.
  override def BackendGPU: Backend = BackendCudnn()
  backend = BackendGPU
}

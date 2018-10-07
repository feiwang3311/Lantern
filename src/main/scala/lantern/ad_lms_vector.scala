package lantern

import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

import scala.collection.mutable.ArrayBuffer
import scala.collection.{Seq => NSeq}
import scala.math._

trait TensorExp extends Dsl with Diff {

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
    class DataLoader(name: String, train: Boolean, mean: Float, std: Float, dims: Int*) {
      def remap[T:Typ] = if (typ[T] == typ[Float]) "float"
        else if (typ[T] == typ[Int]) "int"
        else ???
      def open(path: Rep[String]) = uncheckedPure[Int]("open(",path,",0)")
      def filelen(fd: Rep[Int]) = uncheckedPure[Long]("fsize(",fd,")") // FIXME: fresh name
      def mmap[T:Typ](fd: Rep[Int], len: Rep[Long]) = uncheckedPure[Array[T]]("(",remap(typ[T]),"*)mmap(0, ",len,", PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, ",fd,", 0)")

      val fd = open(s"../data/bin/${name}_${if (train) "train" else "test"}.bin")
      val len = filelen(fd)
      val data = mmap[Float](fd, len)
      val dLength = (len/4L).toInt

      val tfd = open(s"../data/bin/${name}_${if (train) "train" else "test"}_target.bin")
      val tlen = filelen(tfd)
      val target = mmap[Int](tfd, tlen)
      val length = (tlen/4L).toInt

      def dataset = new Tensor(data, NSeq(60000, dims(1), dims(2)))

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

  class Dimensions(val dims: NSeq[Int]) {
    def apply(idx: Int) = {
      if (idx >= dims.length) ???
      else dims(idx)
    }
    def last = dims.last
    def reverse = Dimensions(dims.reverse: _*)

    val (nbElem +: strides) = (dims :\ NSeq[Int](1)) {
      case (dim, seq@(t +: q)) => (dim * t) +: seq
    }

    override def toString = dims mkString " x "
    override def equals(o: Any) = o match {
      case t: Dimensions => this.dims == t.dims
      case _ => false
    }
  }

  implicit def Dimensions2Seq(x: Dimensions) = x.dims

  object Dimensions {
    def apply(x: Int*) = new Dimensions(x)
  }

  /*
  case class TTT(seq: NSeq[Int]) {
    def apply(x: Int) = {
      if (x >= seq.length) ???
      seq(x)
    }

    def last = seq.last
    def reverse = TTT(seq.reverse)

    def equal(that: TTT) = {
      that.seq == seq
    }
  }

  implicit def ttttoSeq(x: TTT) = x.seq
  */

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
    def apply(size: Int) = if (size <= 1) {
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

  /* Not supported in LMS??
  abstract class ForLoop {
    def foreach(f: Rep[Int] => Unit): Unit
  }

  @virtualize
  object ForLoop {
    def apply(start: Int, step: Int, step_size: Int) = if (step <= 5) {
      new ForLoop {
        def foreach(f: Rep[Int] => Unit) = {
          for (i <- (start until (start + step_size * step) by step_size): Range) f(unit(i))
        }
      }
    } else {
      new ForLoop {
        def foreach(f: Rep[Int] => Unit) = {
          for (i <- (start until (start + step * step_size) by step_size): Rep[Range]) f(i)
        }
      }
    }
  } */

  /**
    * A code generation backend for tensor operations.
    *
    * Note: Eventually, a tensor operation IR may be introduced to enable analyses and
    * transformations such as operator fusion and matrix chain multiplication optimization.
    */
  trait Backend {
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

    // TODO: Add more ops:
    // - Elementwise binary ops (+, -, *, /).
    //   - GPU backends need to address broadcasting.
    //   - `BackendCublas` can define addition using `cublasSaxpy`.
    // - Conv2d.
    // - Activation functions (e.g. relu).
    // - Fused multiply add operations?
  }

  /**
    * Native tensor operation backend. WIP.
    * Tensor ops are defined in terms of primitive operations.
    */
  class BackendNative extends Backend {
    override def vectorVectorDot(x: Tensor, y: Tensor): Tensor = {
      assert(x.shape(0) == y.shape(0))
      val value = var_new(0.0f)
      for (i <- DataLoop(x.shape.last)) {
        value += x.data(i) * y.data(i)
      }
      val res = NewArray[Float](1)
      res(0) = readVar(value)
      Tensor(res, 1)
    }

    override def matrixVectorDot(x: Tensor, y: Tensor): Tensor = {
      assert(x.shape(1) == y.shape(0))
      val dim1 = x.shape(0)
      val dim2 = x.shape(1)
      val res = NewArray[Float](dim1)
      for (i <- DataLoop(dim1)) {
        val value = var_new(0.0f)
        for (j <- DataLoop(dim2)) {
          value += x.data(i * dim2 + j) * y.data(j)
        }
        res(i) = readVar(value)
      }
      Tensor(res, dim1)
    }

    override def matrixMatrixDot(x: Tensor, y: Tensor): Tensor = {
      assert(x.shape(1) == y.shape(0))
      val dim1 = x.shape(0)
      val dim2 = x.shape(1)
      val dim3 = y.shape(1)
      val res = NewArray[Float](dim1 * dim3)
      for (i <- DataLoop(dim1)) {
        for (j <- DataLoop(dim3)) {
          val value = var_new(0.0f)
          for (k <- DataLoop(dim2)) {
            value += x.data(i * dim2 + k) * y.data(k * dim3 + j)
          }
          res(i * dim3 + j) = readVar(value)
        }
      }
      Tensor(res, dim1, dim3)
    }
  }

  /**
    * cuBLAS tensor operation backend. WIP.
    */
  class BackendCublas extends Backend {
    // Reference:
    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-dot
    def sdot(a: Rep[Array[Float]], b: Rep[Array[Float]], result: Rep[Array[Float]]) =
      unchecked[Unit]("CUBLAS_CALL(cublasSdot(handle, ", a.length, ",", a, ",1,", b, ",1,", result, "))")

    override def vectorVectorDot(x: Tensor, y: Tensor): Tensor = {
      val res = NewArray[Float](1)
      sdot(x.data, y.data, res)
      Tensor(res, 1)
    }

    // Reference:
    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
    def sgemv(m: Int, n: Int, a: Rep[Array[Float]], b: Rep[Array[Float]], result: Rep[Array[Float]]) = {
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        "CUBLAS_CALL(cublasSgemv(handle, CUBLAS_OP_N, ",
        m, ",", n, ",", one, ",",
        a, ",", m, ",", b, ",", zero, ",", result, ",", one, "))")
    }

    override def matrixVectorDot(x: Tensor, y: Tensor): Tensor = {
      val m = x.shape(0)
      val n = x.shape(1)
      val res = NewArray[Float](m)
      sgemv(m, n, x.data, y.data, res)
      Tensor(res, m)
    }

    // Reference:
    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
    def sgemm(m: Int, n: Int, k: Int, a: Rep[Array[Float]], b: Rep[Array[Float]], result: Rep[Array[Float]]) = {
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        "CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ",
        m, ",", n, ",", k, ",", one, ",",
        a, ",", m, ",", b, ",", k, ",", zero, ",", result, ",", m, "))")
    }

    override def matrixMatrixDot(x: Tensor, y: Tensor): Tensor = {
      val m = x.shape(0)
      val n = y.shape(1)
      val k = y.shape(0)
      val res = NewArray[Float](m * n)
      sgemm(m, n, k, x.data, y.data, res)
      Tensor(res, m, n)
    }
  }

  /**
    * cuDNN tensor operation backend. WIP.
    */
  class BackendCudnn extends Backend {
    override def vectorVectorDot(x: Tensor, y: Tensor): Tensor = ???
    override def matrixVectorDot(x: Tensor, y: Tensor): Tensor = ???
    override def matrixMatrixDot(x: Tensor, y: Tensor): Tensor = ???
  }

  // The current backend for code generation.
  // To switch code generation to a different backend, simply change this value
  // in your DSL program.
  var backend: Backend = new BackendNative

  class Tensor(val data: Rep[Array[Float]], val dimensions: NSeq[Int]) extends Serializable {

    def shape = Dimensions(dimensions: _*)
    val rank = dimensions.length
    val scalarCount = shape.nbElem
    val isScalar = scalarCount == 1

    assert(shape.strides.length >= 1)
    assert(scalarCount != 0, "Empty Tensor!!!")

    def apply(i: Rep[Int]) = data(i)
    def apply(i: Rep[Int], j: Rep[Int]) = data(i * shape(1) + j) // FIXME the index of matrix is not the normal way

    @virtualize
    def clipAt(bound: Float) = {
      for (i <- DataLoop(scalarCount)) {
        if (data(i) > bound) data(i) = bound
        if (data(i) < -1.0f * bound) data(i) = -1.0f * bound
      }
    }

    def mapInPlace(op: Rep[Float] => Rep[Float]) = {
      for (i <- DataLoop(scalarCount)) this.data(i) = op(this.data(i))
    }

    def map(op: Rep[Float] => Rep[Float]) = {
      val res = NewArray[Float](scalarCount)
      for (i <- DataLoop(scalarCount)) res(i) = op(this.data(i))
      new Tensor(res, shape)
    }

    def fold(init: Rep[Float])(op: (Rep[Float], Rep[Float]) => Rep[Float]) = {
      val res = var_new[Float](init)
      for (i <- DataLoop(scalarCount)) var_assign(res, op(res, this.data(i)))
      res
    }

    def elementWiseOpWithBroadCast(that: Tensor, op: ((Rep[Float], Rep[Float]) => Rep[Float])) = {
      Tensor.dimBroadcast(shape, that.shape) match {
        case None => throw new IllegalArgumentException(s"dimensions of vector do not match! ${this.shape.seq} != ${that.shape.seq}")
        case Some((thisShape, thatShape, resShape)) => {
          val resData = NewArray[Float](resShape.nbElem)
          val res = new Tensor(resData, resShape)

          def inplace(offThis: Rep[Int], offThat: Rep[Int], offRes: Rep[Int], dim: Int): Unit = {
            val offres = var_new[Int](offRes)
            val offthis = var_new[Int](offThis)
            val offthat = var_new[Int](offThat)
            for (i <- DataLoop(resShape(dim))) {
              if (dim == resShape.size - 1) {
                resData(offres) = op(this.data(offthis), that.data(offthat))
              } else {
                inplace(offthis, offthat, offres, dim + 1)
              }
              offres += resShape.strides(dim)
              if (thisShape(dim) > 1) offthis += thisShape.strides(dim)
              if (thatShape(dim) > 1) offthat += thatShape.strides(dim)
            }
          }
          inplace(0, 0, 0, 0)
          res
        }
      }
    }

    def +(that: Rep[Float]): Tensor = this.map(x => x + that)
    def +(that: Tensor): Tensor = this.elementWiseOpWithBroadCast(that, _ + _)

    // this operator updates the values of this, unlike the + operator
    def +=(that: Rep[Float]): Unit = this.mapInPlace(x => x + that)
    def += (that: Tensor): Unit = {
      if (that.scalarCount == 1) {
        generate_comment("+= tensor of dim 0")
        this += that.data(0) // broadcast
      }
      else if (this.scalarCount == 1) ??? // this.data(0) = that.fold(this.data(0))((agg, x) => agg + x)
      else if (this.shape == that.shape)
        for (i <- DataLoop(scalarCount)) this.data(i) += that.data(i)
      else throw new IllegalArgumentException(s"dimensions of vector do not match +=! ${this.shape.seq} != ${that.shape.seq}")
    }

    def -(that: Rep[Float]): Tensor = this.map(x => x - that)
    def -(that: Tensor): Tensor = this.elementWiseOpWithBroadCast(that, _ - _)

    // this operator updates the values of this, unlike the - operator
    def -=(that: Rep[Float]): Unit = this.mapInPlace(x => x - that)
    def -= (that: Tensor): Unit = {
      if (that.scalarCount == 1) this -= that.data(0) // broadcast
      else if (this.scalarCount == 1) {
        ???
        // this.data(0) = that.fold(this.data(0))((agg, x) => agg - x)
      }
      else if (this.shape == that.shape)
        for (i <- DataLoop(scalarCount)) this.data(i) -= that.data(i)
      else throw new IllegalArgumentException("dimensions of vector do not match +=!")
    }

    // Element wise multiplication
    def *(that: Rep[Float]): Tensor = this.map(x => x * that)
    def *(that: Tensor): Tensor = this.elementWiseOpWithBroadCast(that, _ * _)

    // this operator updates the values of this, unlike the * operator
    def *=(that: Rep[Float]): Unit = this.mapInPlace(x => x * that)
    def *= (that: Tensor): Unit = {
      if (that.scalarCount == 1) this *= that.data(0) // broadcast
      else if (this.scalarCount == 1) {
        ???
        // this.data(0) = that.fold(this.data(0))((agg, x) => agg * x)
      }
      else if (this.shape == that.shape)
        for (i <- DataLoop(scalarCount)) this.data(i) *= that.data(i)
      else throw new IllegalArgumentException("dimensions of vector do not match +=!")
    }

    // element wise division
    def /(that: Rep[Float]): Tensor = this.map(x => x / that)
    def /(that: Tensor): Tensor = this.elementWiseOpWithBroadCast(that, _ / _)

    // this operator updates the values of this, unlike the / operator
    def /=(that: Rep[Float]): Unit = this.mapInPlace(x => x / that)
    def /= (that: Tensor): Unit = {
      if (that.scalarCount == 1) this /= that.data(0) // broadcast
      else if (this.scalarCount == 1) ??? // this.data(0) = that.fold(this.data(0))((agg, x) => agg / x)
      else if (this.shape == that.shape)
        for (i <- DataLoop(scalarCount)) this.data(i) /= that.data(i)
      else throw new IllegalArgumentException("dimensions of vector do not match +=!")
    }

    def setAsOne() = { this.mapInPlace(x => 1.0f); () }
    def clear() = { this.mapInPlace(x => 0.0f); () }

    def copy_data(that: Tensor) = {
      assert(this.scalarCount == that.scalarCount, "dimensions of vector do not match copy_data!")
      for (i <- DataLoop(scalarCount)) this.data(i) = that.data(i)
    }

    // `dot` represents the following:
    // - vector-vector dot product.
    //   [V] dot [V] => [1] (scalar)
    // - matrix-vector multiplication.
    //   [M1 x M2] dot [M2] => [M1]
    // - matrix-matrix multiplication.
    //   [M1 x M2] dot [M2 x M3] => [M1 x M3]
    def dot(that: Tensor) = {
      generate_comment(s"dot: ${this.shape.seq}, ${that.shape.seq}")
      (this.rank, that.rank) match {
        case (1, 1) => assert(this.shape(0) == that.shape(0), s"Incompatible shapes: ${this.shape}, ${that.shape}")
        case (2, 1) => assert(this.shape(1) == that.shape(0), s"Incompatible shapes: ${this.shape}, ${that.shape}")
        case (2, 2) => assert(this.shape(0) == that.shape(1), s"Incompatible shapes: ${this.shape}, ${that.shape}")
        case _ => throw new IllegalArgumentException(
          s"Only vector-vector, matrix-vector, and matrix-matrix multiplication are allowed (actual shapes: ${this.shape}, ${that.shape})")
      }
      backend.dot(this, that)
    }

    // NOTE: only handles (Vector cart Vector)
    def cart(that: Tensor) = {
      assert(this.rank == 1 && that.rank == 1, "cartesian product is only for 1d vectors")
      val res = NewArray[Float](this.shape(0) * that.shape(0))
      val off = var_new(0)
      for (i <- DataLoop(this.shape(0))) {
      //for (i <- (0 until this.dims(0)): Rep[Range]) {
        for (j <- DataLoop(that.shape(0))) {
        //for (j <- (0 until that.dims(0)): Rep[Range]) {
          res(off) = data(i) * that.data(j)
          off += 1
        }
      }
      Tensor(res, this.shape(0), that.shape(0))
    }

    def trans() = {
      assert(this.rank == 2, "transpose is only for matrix. Tensor transpose is not supported here")
      val res = NewArray[Float](this.scalarCount)
      val offT = var_new(0)
      for (i <- DataLoop(this.shape(1))) {
      //for (i <- (0 until this.dims(1)): Rep[Range]) {
        val off = var_new(0)
        for (j <- DataLoop(this.shape(0))) {
        //for (j <- (0 until this.dims(0)): Rep[Range]) {
          res(offT + j) = data(off + i)
          off += this.shape(1)
        }
        offT += this.shape(0)
      }
      new Tensor(res, this.shape.reverse)
    }

    def tanh() = this.map(x => Math.tanh(x).toFloat)
    def exp() = this.map(x => Math.exp(x).toFloat)
    def log() = this.map(x => Math.log(x).toFloat)
    def sqrt() = this.map(x => Math.sqrt(x).toFloat)
    def sigmoid() = this.map(x => 1.0f / (Math.exp(-1.0f * x).toFloat + 1.0f))

    // NOTE: sum all elements
    def sum() = Tensor.scalar(this.fold(0.0f)(_ + _))

    @virtualize
    def sum2D(dim: Int) = {
      assert(this.rank == 2, "Only deal with 2D tensor")
      assert(dim == 0 || dim == 1, "dim must be in range of this.nbDims")

      if (dim == 0) ???
      else {
        val res = NewArray[Float](this.shape(0))
        val offset = var_new(0)
        for (i <- DataLoop(this.shape(0))) {
          val sum = var_new(0.0f)
          for (j <- DataLoop(this.shape(1))) {
            sum += this.data(offset)
            offset += 1
          }
          res(i) = sum
        }
        Tensor(res, this.shape(0))
      }
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
        val res = NewArray[Float](this.shape(0))
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
      val sum = res.sum2D(dim = 1)
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
      val sum = res.sum2D(dim = 1)
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

      val res = NewArray[Float](this.shape(0))
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
      assert(dims.product == this.scalarCount, s"dims: $dims != scalarCount: $scalarCount")

      Tensor(this.data, dims : _*)
    }


    // NOTE: sum matrix to vector, condense on the dims(1) dimension
    def sumOnDim1() = {
      assert(this.rank <= 2)
      if (this.rank == 1) this
      else {
        val res = NewArray[Float](this.shape(1))
        val off = var_new(0)
        for (j <- DataLoop(this.shape(1))) {
        //for (j <- (0 until this.dims(1)): Rep[Range]) {
          res(off) = this.data(off)
          off += 1
        }

        for (i <- (1 until this.shape(0)): Rep[Range]) {
          val offR = var_new(0)
          for (j <- DataLoop(this.shape(1))) {
          //for (j <- (0 until this.dims(1)): Rep[Range]) {
            res(offR) += data(off)
            off += 1
            offR += 1
          }
        }
        Tensor(res, this.shape(1))
      }
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
    // the result is to update this so that this += that * y, where * is cartesian product
    def add_cartesian(that: Tensor, y: Tensor) = {
      generate_comment("add_cartesian")
      assert(this.rank == 2 && that.shape == Dimensions(this.shape(1)) && y.shape == Dimensions(this.shape(0)) ||
        this.rank == 1 && that.shape == this.shape && y.isScalar, s"${shape} - ${that.shape} - ${y.shape}")
      val off = var_new(0)
      // TODO remove loop if not used
      val up = if (this.rank > 1) this.shape(0) else 1
      for (i <- DataLoop(up)) {
      //for (i <- (0 until up): Rep[Range]) {
        for (j <- DataLoop(shape(1))) {
        //for (j <- (0 until dims(1)): Rep[Range]) {
          this.data(off + j) = this.data(off + j) + that.data(j) * y.data(i)
        }
        off += this.shape(1)
      }
    }
    // FIXME: Maybe try to support slicing??
    // FIXME: Maybe add support for reshaping??
    // FIXME: Maybe support transposing??


    // setting: this is dims(0)-sized vector, that is matrix (dims(0) * dims(1)), y is dims(1)-sized vector
    // the result is to update this so that this accumulate every matrix col * y
    def add_composion(that: Tensor, y: Tensor) = {
      assert(that.rank == 2 && this.shape.seq == NSeq(that.shape(1)) && y.shape.seq == NSeq(that.shape(0))
        || that.rank == 1 && this.shape == that.shape && y.isScalar, s"${shape} - ${that.shape} - ${y.shape}")
      val off = var_new(0)
      // FIXME!!
      val up = if (that.rank > 1) that.shape(0) else 1
      for (i <- DataLoop(up)) {
      //for (i <- (0 until up): Rep[Range]) {
        for (j <- DataLoop(that.shape(1))) {
        //for (j <- (0 until that.dims(1)): Rep[Range]) {
          data(j) += that.data(off + j) * y.data(i)
        }
        off += that.shape(1)
      }
    }
    // def add_composion(that: Tensor, y: Tensor) = {
    //   if (this.nbDims == 1)
    //     this.resize(that.dims(0), )
    // }

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

      // FIXME!!!
      val dims0M = mmax(shape(0), mmax(a.shape(0), b.shape(0)))
      val dims1M = mmax(if (this.rank > 1) shape(1) else 1, mmax(if (a.rank > 1) a.shape(1) else 1, if (b.rank > 1) b.shape(1) else 1))
      //if (this.isScalar) {
      //  for (i <- 0 until (dims0M * dims1M): Rep[Range]) data(0) = data(0) + a.getAt(i) * b.getAt(i)
      //} else {
      //  for (i <- (0 until dims0M * dims1M): Rep[Range]) data(i) = data(i) + a.getAt(i) * b.getAt(i)
      //}
      for (i <- DataLoop(dims0M * dims1M)) {
        if (this.isScalar) { data(0) = data(0) + a.getAt(i) * b.getAt(i) }
        else { data(i) = data(i) + a.getAt(i) * b.getAt(i) }
      }
    }

    def addMul(a: Rep[Float], b: Tensor) = {
      assert(this.shape == b.shape)

      generate_comment("Generate code for addMul")
      for (i <- DataLoop(this.scalarCount)) {
      //for (i <- 0 until this.nbElem: Rep[Range]) {
        this.data(i) = this.data(i) + a * b.data(i)
      }
    }

    def cmulAdd(a: Float, b: Tensor) = {
      assert(this.shape == b.shape)
      for (i <- DataLoop(this.scalarCount))
      //for (i <- 0 until this.nbElem: Rep[Range])
        this.data(i) = a * this.data(i) + b.data(i)

      this // FIXME ??
    }

    def add_div(a: Tensor, b: Tensor) = {
      assert(Tensor.dimCompatible(a, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, b), "dim not Compatible in add_div")
      val dims0M = mmax(shape(0), mmax(a.shape(0), b.shape(0)))
      // FIXME
      val dims1M = mmax(if (rank > 1) shape(1) else 1, mmax(if (a.rank > 1) a.shape(1) else 1, if (b.rank > 1) b.shape(1) else 1))
      //if (this.isScalar) {
      //  for (i <- (0 until dims0M * dims1M): Rep[Range]) data(0) = data(0) + a.getAt(i) / b.getAt(i)
      //} else {
      //  for (i <- (0 until dims0M * dims1M): Rep[Range]) data(i) = data(i) + a.getAt(i) / b.getAt(i)
      //}
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
      // FIXME
      val dims1M = mmax(if (rank > 1) shape(1) else 1, mmax(if (a.rank > 1) a.shape(1) else 1, if (b.rank > 1) b.shape(1) else 1))
      //if (this.isScalar) {
      //  for (i <- (0 until dims0M * dims1M): Rep[Range]) data(0) = data(0) - a.getAt(i) * b.getAt(i) / square(c.getAt(i))
      //} else {
      //  for (i <- (0 until dims0M * dims1M): Rep[Range]) data(i) = data(i) - a.getAt(i) * b.getAt(i) / square(c.getAt(i))
      //}
      for (i <- DataLoop(dims0M * dims1M)) {
        if (this.isScalar) { data(0) = data(0) - a.getAt(i) * b.getAt(i) / square(c.getAt(i)) }
        else { data(i) = data(i) - a.getAt(i) * b.getAt(i) / square(c.getAt(i)) }
      }
    }

    def add_oneMinusSquare_mult(a: Tensor, b: Tensor) = {
      assert(Tensor.dimCompatible(a, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, b), "dim not Compatible in add_oneMinusSquare_mult")
      val dims0M = mmax(shape(0), mmax(a.shape(0), b.shape(0)))
      // FIXME
      val dims1M = mmax(if (rank > 1) shape(1) else 1, mmax(if (a.rank > 1) a.shape(1) else 1, if (b.rank > 1) b.shape(1) else 1))
      //if (this.isScalar) {
      //  for (i <- (0 until dims0M * dims1M): Rep[Range]) data(0) = data(0) + (1.0f - square(a.getAt(i))) * b.getAt(i)
      //} else {
      //  for (i <- (0 until dims0M * dims1M): Rep[Range]) data(i) = data(i) + (1.0f - square(a.getAt(i))) * b.getAt(i)
      //}
      for (i <- DataLoop(dims0M * dims1M)) {
        if (this.isScalar) { data(0) = data(0) + (1.0f - square(a.getAt(i))) * b.getAt(i) }
        else { data(i) = data(i) + (1.0f - square(a.getAt(i))) * b.getAt(i) }
      }
    }

    def oneMinusThenMult(t: Rep[Float]) = (1.0f - t) * t

    def add_oneMinusThenMult_mult(a: Tensor, b: Tensor) = {
      assert(Tensor.dimCompatible(a, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, b), "dim not Compatible in add_oneMinusThenMult_mult")
      val dims0M = mmax(shape(0), mmax(a.shape(0), b.shape(0)))
      // FIXME
      val dims1M = mmax(if (rank > 1) shape(1) else 1, mmax(if (a.rank > 1) a.shape(1) else 1, if (b.rank > 1) b.shape(1) else 1))
      //if (this.isScalar) {
      //  for (i <- (0 until dims0M * dims1M): Rep[Range]) data(0) = data(0) + oneMinusThenMult(a.getAt(i)) * b.getAt(i)
      //} else {
      //  for (i <- (0 until dims0M * dims1M): Rep[Range]) data(i) = data(i) + oneMinusThenMult(a.getAt(i)) * b.getAt(i)
      //}
      for (i <- DataLoop(dims0M * dims1M)) {
        if (this.isScalar) { data(0) = data(0) + oneMinusThenMult(a.getAt(i)) * b.getAt(i) }
        else { data(i) = data(i) + oneMinusThenMult(a.getAt(i)) * b.getAt(i) }
      }
    }

    @virtualize
    def conv2D_batch(kernel: Tensor, bias: Tensor, strides: NSeq[Int], pads: NSeq[Int]): Tensor = {
      assert (this.rank == 4, "For conv_batch , input should be 4-D, with the first dim to be batch")
      assert(kernel.rank == 4, "For Conv, kernel should be 4-D")
      assert(bias.rank == 1, "For Conv, bias should be 1-D")
      assert(bias.shape(0) == kernel.shape(0), "For Conv, bias length should be the same as number of kernels")
      assert(kernel.shape(1) == this.shape(1), "For Conv, input dim_0 should be the same as kernel dim_1")
      assert(this.shape(2) >= kernel.shape(2) && this.shape(3) >= kernel.shape(3), "Image too small for Conv")

      val totalPads = pads.sum
      // TODO: (Fei Wang) not sure if the order is correct!!!
      assert(pads.size == 4, "pads should have 4 values, up, down, left, right")
      assert(strides.size == 2, "strides should have a strideRow and a strideCol")
      val ((strideRow:Int) :: (strideCol:Int) :: Nil) = strides.take(2).toList
      val ((padUp:Int) :: (padDown:Int) :: (padLeft:Int) :: (padRight:Int) :: Nil) = pads.take(4).toList
      assert(strideRow >= 1, "stride of row should be at least 1")
      assert(strideCol >= 1, "stride of col should be at least 1")
      assert(padUp == padDown && padUp == padLeft && padUp == padRight, "For now, assume all values in pads are the same")

      val resWidth = convSize(this.shape(2) + padLeft + padRight, kernel.shape(2), strideRow)
      val resHeight = convSize(this.shape(3) + padUp + padDown, kernel.shape(3), strideCol)
      val res = Tensor.fillWithBias(bias, 1, this.shape(0), kernel.shape(0), resWidth, resHeight)

      for (i <- DataLoop(this.shape(0))) {
        val ptrInput = slice(this.data, i * this.shape.strides(0))
        val ptrOutput = slice(res.data, i * res.shape.strides(0))
        Tensor(ptrInput, this.shape.drop(1): _*).conv2D_inplace(kernel, strides, pads, Tensor(ptrOutput, res.shape.drop(1): _*))
      }
      res
    }

    @virtualize
    def conv2D_inplace(kernel: Tensor, strides: NSeq[Int], pads: NSeq[Int], res: Tensor): Unit = {
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
        for (inPane <- DataLoop(this.shape(0))) {
          val ptrInput = slice(this.data, offInput)       // input, restarting from the start of this input channel (2D)
          val ptrWeight = slice(kernel.data, offWeight2)  // kernel, restarting from the start of this input channel (2D)

          if (totalPads == 0) Tensor(ptrOutput, resHeight, resWidth).conv2D1(
            Tensor(ptrInput, this.shape(1), this.shape(2)),
            Tensor(ptrWeight, kernel.shape(2), kernel.shape(3)),
            strideRow, strideCol)
          else Tensor(ptrOutput, resHeight, resWidth).conv2D2(
            Tensor(ptrInput, this.shape(1), this.shape(2)),
            Tensor(ptrWeight, kernel.shape(2), kernel.shape(3)),
            strideRow, strideCol, padUp, padDown, padLeft, padRight)

          offWeight2 += kernel.shape.strides(1)
          offInput += this.shape.strides(0)
        }
        offWeight1 += kernel.shape.strides(0)
        offOut += res.shape.strides(0)
      }
    }

    @virtualize
    def conv2D(kernel: Tensor, bias: Tensor, strides: NSeq[Int], pads: NSeq[Int]) = {

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
      val res = Tensor.fillWithBias(bias, 0, kernel.shape(0), resWidth, resHeight)

      val offOut = var_new(0)                         // offset for the res by channel
      val offWeight1 = var_new(0)                     // offset for the kernel by channel (dim_0)
      for (outPane <- DataLoop(kernel.shape(0))) {
        val offWeight2 = var_new(offWeight1)          // offset for the kernel for each z-dim of a given channel
        val offInput = var_new(0)                     // offset for this for each channel of input
        val ptrOutput = slice(res.data, offOut)           // res, restarting from the start of this output channel (2D)
        for (inPane <- DataLoop(this.shape(0))) {
          val ptrInput = slice(this.data, offInput)      // input, restarting from the start of this input channel (2D)
          val ptrWeight = slice(kernel.data, offWeight2)  // kernel, restarting from the start of this input channel (2D)

          if (totalPads == 0) Tensor(ptrOutput, resHeight, resWidth).conv2D1(
            Tensor(ptrInput, this.shape(1), this.shape(2)),
            Tensor(ptrWeight, kernel.shape(2), kernel.shape(3)),
            strideRow, strideCol)
          else Tensor(ptrOutput, resHeight, resWidth).conv2D2(
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

    @virtualize
    def conv2D2(input: Tensor, kernel: Tensor, strideRow: Int, strideCol: Int, padUp: Int, padDown: Int, padLeft: Int, padRight: Int): Unit = {
      assert(this.rank == 2 && input.rank == 2 && kernel.rank == 2)

      // looping for the output
      val offOutput = var_new(0)                    // offset of the output, move one by one in second loop
      // val offInputR = var_new(0)                 // offset of the input, move by each row * strideRow
      val InputR = var_new(-padLeft)
      for (outRow <- DataLoop(this.shape(0))) {
        // val offInputC = var_new(offInputR)       // offset of the input, build on offInputR, move by each strideCol
        val InputC = var_new(-padUp)
        for (outCol <- DataLoop(this.shape(1))) {

          // looping for the kernel
          val sum = var_new(0.0f)
          val offKernel = var_new(0)                // offset of the kernel, move by row of kernel
          // val offInput  = var_new(offInputC)     // offset of the input, built on offInputC, move by row of input
          for (kernelRow <- DataLoop(kernel.shape(0))) {
            // val ptrInput = slice(input.data, offInput)
            for (kernelCol <- DataLoop(kernel.shape(1))) {
              val iR = InputR + kernelRow
              val iC = InputC + kernelCol
              if (iR < 0 || iC < 0 || iR >= input.shape(0) || iC >= input.shape(1)) ()
              else {
                sum += kernel.data(offKernel) * input.data(iR * input.shape.strides(0) + iC)
              }
              offKernel += 1
            }
            // offInput  += input.strides(1)
          }
          this.data(offOutput) = this.data(offOutput) + sum

          // stepping of the offsets of the looping for the output
          // offInputC += strideCol
          offOutput += 1
          InputC += strideCol
        }
        // offInputR += strideRow * input.strides(1)
        InputR += strideRow
      }
    }

    @virtualize
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

          Tensor(ptrOutput, resHeight, resWidth).conv2D1(
            Tensor(ptrInput, this.shape(1), this.shape(2)), Tensor(ptrWeight, kernel.shape(2), kernel.shape(3)), strideRow, strideCol)

          offWeight2 += kernel.shape.strides(1)
          offInput += this.shape.strides(0)
        }
        offWeight1 += kernel.shape.strides(0)
        offOut += res.shape.strides(0)
      }
      res
    }

    // Taken from Torch: THTensorConv.cpp#198L
    // https://github.com/pytorch/pytorch/blob/master/aten/src/TH/generic/THTensorConv.cpp
    @virtualize
    def conv2D1(input: Tensor, kernel: Tensor, strideRow: Int, strideCol: Int): Unit = {
      assert(this.rank == 2 && input.rank == 2 && kernel.rank == 2)

      // looping for the output
      val offOutput = var_new(0)                 // offset of the output, move one by one in second loop
      val offInputR = var_new(0)                 // offset of the input, move by each row * strideRow
      for (outRow <- DataLoop(this.shape(0))) {
        val offInputC = var_new(offInputR)       // offset of the input, built on offInputR, move by each strideCol
        for (outCol <- DataLoop(this.shape(1))) {

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
          this.data(offOutput) = this.data(offOutput) + sum
          offOutput += 1
          offInputC += strideCol
        }
        offInputR += strideRow * input.shape.strides(0)
      }
    }

    @virtualize
    def maxPool(strideRow: Int, strideCol: Int) = {
      assert(this.rank == 3)

      val resHeight = this.shape(1) / strideRow
      val resWidth = this.shape(2) / strideCol
      val res = Tensor.fill(scala.Float.MinValue, this.shape(0), resHeight, resWidth)

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
    def maxPool_k_batch(kernels: Seq[Int], strides: Seq[Int]): (Tensor, Rep[Array[Int]]) = {
      assert(this.rank == 4, "the input for maxPool (with batch) should have 4 dimensions")
      assert(kernels.size == 2 && strides.size == 2, "kernels and strides should be size 2")
      val (strideRow :: strideCol :: _) = strides.toList
      val (kernelRow :: kernelCol :: _) = kernels.toList
      assert(strideRow >= 1 && kernelRow >= 1, "kernel width and stride width should be at least 1")
      assert(strideCol >= 1 && kernelCol >= 1, "kernel height and stride height should be at least 1")
      assert(this.shape(2) >= kernelRow && this.shape(3) >= kernelCol, "Image too small for maxPool_k: " + this.shape + "|" + (kernelRow, kernelCol))

      val resWidth = convSize(this.shape(2), kernelRow, strideRow)
      val resHeight = convSize(this.shape(3), kernelCol, strideCol)
      val res = Tensor.fill(scala.Float.MinValue, this.shape(0), this.shape(1), resWidth, resHeight)
      val savedIdx = NewArray[Int](res.scalarCount)

      for (i <- DataLoop(this.shape(0))) {
        val ptrInput  = slice(this.data, i * this.shape.strides(0))
        val ptrOutput = slice(res.data, i * res.shape.strides(0))
        val ptrIdx    = slice(savedIdx, i * res.shape.strides(0))
        Tensor(ptrInput, this.shape.drop(1): _*).maxPool_k_inplace(
          kernelRow, kernelCol, strideRow, strideCol, Tensor(ptrOutput, res.shape.drop(1): _*), ptrIdx)
      }
      (res, savedIdx)
    }

    @virtualize
    def maxPool_k_inplace(kernelRow: Int, kernelCol: Int, strideRow: Int, strideCol: Int, res: Tensor, savedIdx: Rep[Array[Int]]): Unit = {
      val resWidth = res.shape(1)
      val resHeight = res.shape(2)

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
                if (this.data(this_index_2) > res(offout_2)) {
                  res.data(offout_2) = this.data(this_index_2)
                  savedIdx(offout_2) = this_index_2
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
    }

    @virtualize
    def maxPool_k(kernels: Seq[Int], strides: Seq[Int]) = {
      assert(this.rank == 3, "the input for maxPool should have 3 dimensions")
      assert(kernels.size == 2 && strides.size == 2, "kernels and strides should be size 2 for maxpool_k")
      val (strideRow :: strideCol :: _) = strides.toList
      val (kernelRow :: kernelCol :: _) = kernels.toList
      assert(strideRow >= 1 && kernelRow >= 1, "kernel width and stride width should be at least 1")
      assert(strideCol >= 1 && kernelCol >= 1, "kernel height and stride height should be at least 1")
      assert(this.shape(1) >= kernelRow && this.shape(2) >= kernelCol, "Image too small for maxPool_k")

      val resWidth = convSize(this.shape(1), kernelRow, strideRow)
      val resHeight = convSize(this.shape(2), kernelCol, strideCol)
      val res = Tensor.fill(scala.Float.MinValue, this.shape(0), resWidth, resHeight)
      val savedIdx = NewArray[Int](res.scalarCount)

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
                if (this.data(this_index_2) > res(offout_2)) {
                  res.data(offout_2) = this.data(this_index_2)
                  savedIdx(offout_2) = this_index_2
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
      (res, savedIdx)
    }

    @virtualize
    def dropout(prob: Float = 0.5f) = {
      assert(0.0f <= prob && prob <= 1.0f)

      val res = NewArray[Float](this.scalarCount)
      val mask = NewArray[Float](this.scalarCount)

      val scale = if (prob < 1.0f) 1.0f / (1.0f - prob) else 0.0f

      val guard: Rep[Boolean] = prob < 1.0f
      for (i <- DataLoop(this.scalarCount)) {
      // for (i <- 0 until this.nbElem: Rep[Range]) {
        if (guard && Random.rand() > prob) {
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

      val res = NewArray[Float](this.scalarCount)
      for (i <- 0 until this.scalarCount: Rep[Range]) {
        if (this(i) < 0.0f)
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
      val res = NewArray[Float](totalnbElem)

      // prepare for looping/copying
      val totalFrom = this +: others        // put all tensors in one Seq for easy of handling
      val targetId = var_new(0)             // this is the index of res to write to
      // looping over dims higher than dim, squashed
      for (high <- DataLoop(higherDimsSquashed)) {
        // looping over the concatenation dim
        for (whichTensor <- totalFrom) {
          // looping over the dimensions lower than or equal to dim, in the current tensor
          val ptrIntput = slice(whichTensor.data, high * whichTensor.shape.strides(dim-1))
          for (lowOrEqual <- DataLoop(whichTensor.shape.strides(dim-1))) {
            res(targetId) = ptrIntput(lowOrEqual)
            targetId += 1
          }
        }
      }
      Tensor(res, resDims: _*)
    }

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

      val res = NewArray[Float](this.shape(0))
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
  }

  object Tensor {
    def apply(dims: Int*) = {
      val size = dims.product
      new Tensor(NewArray[Float](size), dims)
    }
    def apply(data: Rep[Array[Float]], dims: Int*) = new Tensor(data, dims)

    def dimCompatible(a: Tensor, b: Tensor) = {
      (a.shape == b.shape) || a.isScalar || b.isScalar
    }

    def dimBroadcast(a: NSeq[Int], b: NSeq[Int]): Option[(Dimensions, Dimensions, Dimensions)] = {
      def bc(a: NSeq[Int], b: NSeq[Int], trail: List[Int]): List[Int] = {
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
        if (a.size > b.size) Some((new Dimensions(a), new Dimensions(NSeq.fill(a.size - b.size)(1) ++ b), new Dimensions(res.toSeq)))
        else if (a.size < b.size) Some((new Dimensions(NSeq.fill(b.size - a.size)(1) ++ a), new Dimensions(b), new Dimensions(res.toSeq)))
        else Some((new Dimensions(a), new Dimensions(b), new Dimensions(res.toSeq)))
      }
    }

    def randseed(seed: Int) = unchecked[Unit]("srand(", seed, ")")
    def randseed() = unchecked[Unit]("srand(time(NULL))")
    def rand(dims: Int*) = randinit(dims.toSeq, 1.0f, None)
    def rand(scale: Float, dims: Int*) = randinit(dims.toSeq, scale, None)
    def randinit(dim0: Int): Tensor = randinit(NSeq(dim0), 1.0f, None)
    def randinit(dim0: Int, seed: Option[Int]): Tensor = randinit(NSeq(dim0), 1.0f, seed)
    def randinit(dim0: Int, dim1: Int, scale: Float): Tensor = randinit(NSeq(dim0, dim1), scale, None)
    def randinit(dims: NSeq[Int], scale: Float = 1.0f, seed: Option[Int] = None): Tensor = {
      val size = dims.product
      val res = NewArray[Float](size)
      for (i <- (0 until size): Rep[Range]) res(i) = (Random.rand() - 0.5f) * scale
      new Tensor(res, dims)
    }

    def randn(dim0: Int, dim1: Int = 1, scale: Float = 1.0f, offset: Int = 0) = {
      val res = NewArray[Float](dim0 * dim1)
      for (i <- (0 until dim0 * dim1): Rep[Range]) res(i) = unchecked[Float]("d(gen)") * scale
      Tensor(res, dim0, dim1)
    }

    def randPositive(dims: Int*) = {
      val size = dims.product
      val res = NewArray[Float](size)
      for (i <- (0 until size): Rep[Range]) res(i) = Random.rand()
      new Tensor(res, dims)
    }

    def fill(value: Rep[Float], dims: Int*) = {
      val size = dims.product
      val res = NewArray[Float](size)
      for (i <- (0 until size): Rep[Range]) res(i) = value
      new Tensor(res, dims)
    }

    def fill(fFill: NSeq[Rep[Int]] => Rep[Float], dims: Int*) = {
      val size = dims.product
      val res = NewArray[Float](size)

      var idx = var_new(0)
      def innerFill(args: NSeq[Rep[Int]]) = {
        res(idx) = fFill(args)
        idx += 1
      }

      val dum = (dims :\ innerFill _) {
        case (up, f) =>
          (args: NSeq[Rep[Int]]) => {
            for (i <- 0 until up: Rep[Range]) {
              f(args :+ i)
            }
          }
      }
      dum(NSeq[Rep[Int]]())
      new Tensor(res, dims)
    }

    def fillWithBias(bias: Tensor, dim: Int, dims: Int*) = {
      assert(dim < dims.size && dim >= 0, s"target dimension ${dim} is out of range ${dims}")
      assert(bias.rank == 1 && bias.scalarCount == dims.drop(dim).head, s"bias should be 1D and have the same length as given dim")
      val size = dims.product
      val res = NewArray[Float](size)

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
      new Tensor(res, dims)
    }

    def zeros(dims: Int*): Tensor = {
      fill(0.0f, dims: _*)
    }

    def zeros(that: Tensor): Tensor = {
      zeros(that.shape : _*)
    }

    def zeros_like(that: Tensor) = {
      zeros(that.shape : _*)
    }

    def scalar(value: Rep[Float]) = {
      val res = NewArray[Float](1)
      res(0) = value
      Tensor(res, 1)
    }

    def ones(dims: Int*) = fill(1.0f, dims: _*)
    def ones(that: Tensor) = fill(1.0f, that.shape: _*)
    def halves(dims: Int*) = fill(0.5f, dims: _*)

    def expand(vector: Tensor, dim1: Int) = {
      assert(vector.rank == 1)
      val res = NewArray[Float](vector.shape(0) * dim1)
      val off = var_new(0)
      for (j <- (0 until dim1): Rep[Range]){
        for (i <- (0 until vector.shape(0)): Rep[Range]) {
          res(off) = vector.data(i)
          off += 1
        }
      }
      new Tensor(res, dim1 +: vector.shape)
    }

    def copy(vector: Tensor) = {
      val res = NewArray[Float](vector.scalarCount)
      for (i <- (0 until vector.scalarCount): Rep[Range]) res(i) = vector.data(i)
      new Tensor(res, vector.shape)
    }

    def fromData(x: Float*) = {
      val y = x.toArray
      val res = NewArray[Float](y.length)
      for (i <- 0 until y.length: Range) res(i) = y(i)
      Tensor(res, y.length)
    }

    def fromData(dims: Seq[Int], x: Float*) = {
      val y = x.toArray
      val res = NewArray[Float](y.length)
      for (i <- 0 until y.length: Range) res(i) = y(i)
      Tensor(res, dims: _*)
    }

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

    def clip_grad(bound: Float) = {
      d.clipAt(bound)
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

    def + (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x + that.x); k(y)
      // this.d += y.d; that.d += y.d
      val opThis = (_: Rep[Float], _: Rep[Float], c: Rep[Float]) => c
      val opThat = (_: Rep[Float], _: Rep[Float], c: Rep[Float]) => c
      backpropElementWiseOpWithBroadCast(that, y, opThis, opThat)
    }

    def - (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x - that.x); k(y)
      // this.d += y.d; that.d -= y.d
      val opThis = (_: Rep[Float], _: Rep[Float], c: Rep[Float]) => c
      val opThat = (_: Rep[Float], _: Rep[Float], c: Rep[Float]) => -1.0f * c
      backpropElementWiseOpWithBroadCast(that, y, opThis, opThat)
    }

    // this is element wise multiplication
    def * (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x * that.x); k(y)
      // this.d.add_mult(that.x, y.d); that.d.add_mult(this.x, y.d)
      val opThis = (_: Rep[Float], b: Rep[Float], c: Rep[Float]) => c * b
      val opThat = (a: Rep[Float], _: Rep[Float], c: Rep[Float]) => c * a
      backpropElementWiseOpWithBroadCast(that, y, opThis, opThat)
    }

    // element wise division
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
      // FIXME: intermediate Tensors donot need to be substatiated, can optimize!
      //y.d.print("dot")
      if (this.d.rank == 1) {
        assert(y.d.isScalar)
        this.d.addMul(y.d.data(0), that.x)
        that.d.addMul(y.d.data(0), this.x)
      } else {
        // FIXME: need optimization using addMul and dataloop!!
        this.d.add_cartesian(that.x, y.d)
        that.d.add_composion(this.x, y.d)
        //this.d.addMul(y.d.resize(y.d.dims(0), 1), that.x.resize(1, that.x.dims(0)))
        //that.d.resize(1, that.d.dims(0)).addMul(y.d.resize(1, y.d.dims(0)), this.x)
      }
      // this.d += that.x * y.d // broadcasting
      // that.d += this.x * y.d // broadcasting
    }

    def tanh(): TensorR @diff = shift { (k : TensorR => Unit) =>
      val y = TensorR(x.tanh()); k(y)
      // FIXME: intermediate Tensors donot need to be substatiated, can optimize!
      //this.d += (Tensor.ones(1) - y.x * y.x) * y.d // broadcasting
      generate_comment("backpropagate tanh")
      this.d.add_oneMinusSquare_mult(y.x, y.d)
    }

    def exp(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.exp()); k(y)
      // Fix
      //this.d += y.x * y.d
      generate_comment("backpropage exp")
      this.d.add_mult(y.x, y.d)
    }

    def log(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.log()); k(y)
      // Fix
      //this.d += y.d / x
      this.d.add_div(y.d, x)
    }

    def sigmoid(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.sigmoid()); k(y)
      //this.d += (Tensor.ones(1) - y.x) * y.x * y.d
      this.d.add_oneMinusThenMult_mult(y.x, y.d)
    }

    def sum(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = new TensorR(x.sum(), Tensor.zeros(1)); k(y)
      this.d += y.d
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
      val sum = y.d.sum2D(dim = 1)
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

    def update(lr: Float, mom: Float) = {
    }

    @virtualize
    // conv with batch, bias, and pads
    def convBBP(kernel: TensorR, bias: TensorR, stride: NSeq[Int], pads: Seq[Int]): TensorR@diff = shift { (k: TensorR => Unit) =>
      assert(this.isInput || this.d.scalarCount == this.x.scalarCount, "For convBBP, THIS is either input or intermediate stage")
      assert(this.x.rank == 4, "For convBBP, THIS is dim 4: batch, channel, row, col")
      assert(pads.tail.forall(x => x == pads.head), "pads should be the same in all directions")
      val y = TensorR(x conv2D_batch(kernel.x, bias.x, stride, pads))
      k(y)

      // back propagate
      val strideRow = stride.head
      val strideCol = stride.last

      if (pads.sum == 0) {
        for (batch <- DataLoop(this.x.shape(0))) {
          val offOutputD = var_new(batch * y.d.shape.strides(0))     // offset for the output, based on batch, step by 1
          val offKernel = var_new(0)                           // offset for kernel, step by kernel strides(1) -- which kernel
          // looping for the output
          for (kOut <- DataLoop(y.d.shape(1))) {
            val offInputR = var_new(batch * this.x.shape.strides(0)) // offset of input, based on batch, step by input.strides(3) * strideRow
            val sum = var_new(0.0f)                            // collector of bias gradient
            for (row <- DataLoop(y.d.shape(2))) {
              val offInputC = var_new(offInputR)               // offset of input, based on offInputR, step by strideCol
              for (col <- DataLoop(y.d.shape(3))) {
                val dCurr: Rep[Float] = y.d.data(offOutputD)
                sum += dCurr                                   // collect bias gradient

                // looping for the kernel
                val offInputP = var_new(offInputC)             // offset of input, based on offInputC, step by input.strides(2)
                val offKernelR = var_new(offKernel)            // offset of kernel, based on offKernel, step by 1
                for (pane <- DataLoop(kernel.d.shape(1))) {
                  val offInputKR = var_new(offInputP)          // offset of input, step by input.strides(3) -- row
                  for (kR <- DataLoop(kernel.d.shape(2))) {
                    for (kC <- DataLoop(kernel.d.shape(3))) {
                      if (!this.isInput) this.d.data(offInputKR + kC) = this.d.data(offInputKR + kC) + dCurr * kernel.x.data(offKernelR)
                      kernel.d.data(offKernelR) = kernel.d.data(offKernelR) + dCurr * this.x.data(offInputKR + kC)
                      offKernelR += 1
                    }
                    offInputKR += this.x.shape.strides(2)
                  }
                  offInputP += this.x.shape.strides(1)
                }

                offInputC += strideCol
                offOutputD += 1
              }
              offInputR += strideRow * this.x.shape.strides(2)
            }
            bias.d.data(kOut) = bias.d.data(kOut) + sum        // give value of collector to the bias gradient
            offKernel += kernel.x.shape.strides(0)
          }
        }
      } else {
        for (batch <- DataLoop(this.x.shape(0))) {
          val offOutputD = var_new(batch * y.d.shape.strides(0))     // offset for the output, based on batch, step by 1
          val offKernel  = var_new(0)                          // offset for the kernel, step by kernel strides(1) -- which kernel
          val offInputD  = batch * this.x.shape.strides(0)           // fixed offset for the input, based on batch
          // looping for the output
          for (kOut <- DataLoop(y.d.shape(1))) {
            val InputR = var_new(-pads.head)                   // Row of input, starting from -pads
            val sum = var_new(0.0f)                            // collector of bias gradient
            for (row <- DataLoop(y.d.shape(2))) {
              val InputC = var_new(-pads.head)                 // Col of input, starting from -pads
              for (col <- DataLoop(y.d.shape(3))) {
                val dCurr: Rep[Float] = y.d.data(offOutputD)
                sum += dCurr                                   // collect the bias gradient

                // looping for the kernel
                val offKernelR = var_new(offKernel)            // offset if kernel, based on offKernel, step by 1
                // offset of input based on batch, pane, and index of output
                val InputI_pane = var_new[Int](offInputD + InputR * this.x.shape.strides(2) + InputC)
                for (pane <- DataLoop(kernel.d.shape(1))) {
                  val InputI_kR = var_new[Int](InputI_pane)    // offset of input based on InputI_pane and row
                  for (kR <- DataLoop(kernel.d.shape(2))) {
                    for (kC <- DataLoop(kernel.d.shape(3))) {
                      if (InputR+kR < 0 || InputR+kR >= this.x.shape(2) || InputC+kC < 0 || InputC+kC >= this.x.shape(3)) ()
                      else {
                        val InputI = InputI_kR + kC
                        if (!this.isInput) this.d.data(InputI) = this.d.data(InputI) + dCurr * kernel.x.data(offKernelR)
                        kernel.d.data(offKernelR) = kernel.d.data(offKernelR) + dCurr * this.x.data(InputI)
                      }
                      offKernelR += 1
                    }
                    InputI_kR += this.x.shape.strides(2)
                  }
                  InputI_pane += this.x.shape.strides(1)
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
      }

      ()
    }

    @virtualize
    // conv with bias and pads
    def convBP(kernel: TensorR, bias: TensorR, strides: NSeq[Int], pads: NSeq[Int]): TensorR@diff = shift { (k: TensorR => Unit) =>

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

    @virtualize
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

    @virtualize  // maxpool with kernel size potentially different from strides, and works with batch dimension!
    def maxPoolBK(kernels: Seq[Int], strides: Seq[Int]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val (y, sidx) = this.x.maxPool_k_batch(kernels, strides)
      val ty = TensorR(y)
      k(ty)

      // back propagate
      for (i <- DataLoop(y.scalarCount)) {
        this.d.data(sidx(i)) += ty.d.data(i)
      }
    }

    @virtualize  // maxpool with kernel size potentially different from strides
    def maxPoolK(kernels: Seq[Int], strides: Seq[Int]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val (y, sidx) = this.x.maxPool_k(kernels, strides)
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
          val ptrInput = slice(whichTensorR.d.data, high * whichTensorR.x.shape.strides(dim-1))
          for (lowOrEqual <- DataLoop(whichTensorR.x.shape.strides(dim-1))) {
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
              this.d.data(offset_a + j) += ty.d(res_offset) * scale
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

      //y.d.print("relu")

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

    def apply(dim0: Int, dim1: Int): TensorR = {
      new TensorR(Tensor.zeros(dim0, dim1), Tensor.zeros(dim0, dim1))
    }

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
    // val k1 = FUN(init.x.dims(0))(k)

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
/*
  def FUNl(dim0: Int)(f: (Rep[Int] => (TensorR => Unit) => (TensorR => Unit))): (Rep[Int] => (TensorR => Unit) => (TensorR => Unit)) = {

    val f1 = fun { (i:  Rep[Int], t1: Rep[Array[Array[Float]] => Unit], xx: Rep[Array[Array[Float]]]) =>
      val t2: (TensorR => Unit) = { (x:TensorR) =>
        val temp = NewArray[Array[Float]](2)
        temp(0) = x.x.data; temp(1) = x.d.data
        t1(temp)
      }
      val t3: (TensorR => Unit) = f(i)(t2)
      t3(new TensorR(Tensor(xx(0), dim0), Tensor(xx(1), dim0)))
    }

    {i: Rep[Int] => k1: (TensorR => Unit) =>
      {
        val k2: Rep[Array[Array[Float]] => Unit] = fun { (x: Rep[Array[Array[Float]]]) =>
          k1(new TensorR(Tensor(x(0), dim0), Tensor(x(1), dim0)))
        }
        val k4: (TensorR => Unit) = {(x: TensorR) =>
          val temp = NewArray[Array[Float]](2)
          temp(0) = x.x.data; temp(1) = x.d.data
          f1(i, k2, temp)
        }
        k4
      }
    }
  }*/

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

  def FUNl(f: (Rep[Int] => (TensorR => Unit) => TensorR => Unit)): (Rep[Int] => (TensorR => Unit) => TensorR => Unit) = {i: Rep[Int] => k1: (TensorR => Unit) => (x: TensorR) =>

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

  @virtualize
  def LOOPL(init: TensorR)(c: Rep[Int])(b: Rep[Int] => TensorR => TensorR @diff): TensorR @diff = shift { k: (TensorR => Unit) =>
    lazy val loop: Rep[Int] => (TensorR => Unit) => TensorR => Unit = FUNl{ (gc: Rep[Int]) => (k: TensorR => Unit) => (x: TensorR) =>
      def sh_loop: (Rep[Int] => TensorR @diff) = (i: Rep[Int]) => shift{(k: TensorR => Unit) => loop(i)(k)(x)}
      RST(k (IF(gc < c) { b(gc)(sh_loop(gc+1)) } { init }) )
      // if (gc < c) {RST(k(b(gc)(sh_loop(gc + 1))))} else {RST(k(x))}
      // if (gc < c) { loop(gc+1)((x: TensorR) => RST(k(b(gc)(x))))(x) } else { RST(k(x)) }
    }
    loop(0)(k)(init)
  }

  @virtualize
  def LOOPT(start: Rep[Int])(init: TensorR)(lch: Rep[Array[Int]], rch: Rep[Array[Int]])(b: (TensorR, TensorR, Rep[Int]) => TensorR @diff): TensorR @diff = shift {
    k: (TensorR => Unit) =>

      lazy val tree: Rep[Int] => (TensorR => Unit) => TensorR => Unit = FUNl{ (i: Rep[Int]) => (k: TensorR => Unit) => (x: TensorR) =>
        def sh_tree: (Rep[Int] => TensorR @diff) = (i: Rep[Int]) => shift{(k: TensorR => Unit) => tree(i)(k)(x)}
        RST(k( IF(i >= 0) { b(sh_tree(lch(i)), sh_tree(rch(i)), i) } { init } ))
        // if (i >= 0) { RST(k(b(sh_tree(lch(i)), sh_tree(rch(i)), i))) } else { RST(k(x)) }
        // if (i >= 0) { tree(lch(i))((l: TensorR) => tree(rch(i))((r: TensorR) => RST(k(b(l, r, i))))(x))(x) } else { RST(k(x)) }
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

  @virtualize
  def LOOPLM(start: Rep[Int])(init: ArrayBuffer[TensorR])(c: Rep[Int])(b: Rep[Int] => ArrayBuffer[TensorR] => ArrayBuffer[TensorR] @diff):
  ArrayBuffer[TensorR] @diff = shift { k: (ArrayBuffer[TensorR] => Unit) =>

    lazy val loop: Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit = FUNlm { (i: Rep[Int]) => (k: ArrayBuffer[TensorR] => Unit) => (x: ArrayBuffer[TensorR]) =>

      def sh_loop: (Rep[Int] => ArrayBuffer[TensorR] @diff) = (i: Rep[Int]) => shift{ (k: ArrayBuffer[TensorR] => Unit) => loop(i)(k)(x) }

      RST(k( IFm(i < c) { b(i)(sh_loop(i+1)) } { init } ))
      //if (i < c) ( RST(k(b(i)(sh_loop(i+1)) ) else {RST(k(x))}
      //if (i < c) { loop(i+1)((x: ArrayBuffer[TensorR]) => RST(k(b(i)(x))))(x) } else { RST(k(x)) }
    }
    loop(start)(k)(init)
  }

  @virtualize
  def LOOPTM(start: Rep[Int])(init: ArrayBuffer[TensorR])(lch: Rep[Array[Int]], rch: Rep[Array[Int]])
  (b: (ArrayBuffer[TensorR], ArrayBuffer[TensorR], Rep[Int]) => ArrayBuffer[TensorR] @diff): ArrayBuffer[TensorR] @diff = shift { k: (ArrayBuffer[TensorR] => Unit) =>

      lazy val tree: Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit = FUNlm { (i: Rep[Int]) => (k: ArrayBuffer[TensorR] => Unit) => (x: ArrayBuffer[TensorR]) =>

        def sh_tree: (Rep[Int] => ArrayBuffer[TensorR] => ArrayBuffer[TensorR] @diff) = (i: Rep[Int]) => (x: ArrayBuffer[TensorR]) => shift{(k: ArrayBuffer[TensorR] => Unit) => tree(i)(k)(x)}

        RST(k( IFm (i >= 0) { b(sh_tree(lch(i))(init), sh_tree(rch(i))(init), i) } { init } ))
        //if (i >= 0) { tree(lch(i))((l: ArrayBuffer[TensorR]) => tree(rch(i))((r: ArrayBuffer[TensorR]) => RST(k(b(l, r, i))))(x))(x) }
        //else { RST(k(x)) }
      }
      tree(start)(k)(init)
  }

  def gradR(f: TensorR => TensorR @diff)(x: Tensor): Tensor = {
    val x1 = new TensorR(x, Tensor.zeros(x.shape(0)))
    reset { val y = f(x1)
      y.d.setAsOne()
      // y.x.print() // this is the result of forward propagation (likely the loss)
    () }
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
      y.d.setAsOne()
      result.copy_data(y.x)
      //y.x.print()
    () }
    result
  }

  def getMallocAddr(): Rep[Long] = {
    unchecked[Long]("(long)mallocAddr")
  }

  def resetMallocAddr(addr: Rep[Long]) = {
    unchecked[Unit]("mallocAddr = (void*)", addr)
  }

}

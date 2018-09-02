package lantern

import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

import scala.collection.mutable.ArrayBuffer
import scala.collection.{Seq => NSeq}
import scala.math._

trait TensorExp extends Dsl {

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

      val fd = open(s"data/bin/${name}_${if (train) "train" else "test"}.bin")
      val len = filelen(fd)
      val data = mmap[Float](fd, len)
      val dLength = (len/4L).toInt

      val tfd = open(s"data/bin/${name}_${if (train) "train" else "test"}_target.bin")
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
          off += t.nbElem
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

  def slice(arr: Rep[Array[Float]], off: Rep[Int]) = uncheckedPure[Array[Float]](arr, "+", off)

  object Encoding {
    val ix_a = 96  // index starts from 1

    def char_to_ix(ch: Rep[Char]): Rep[Int] = ch.AsInstanceOf[Int] - ix_a
    def ix_to_char(ix: Rep[Int]): Rep[Char] = (ix + ix_a).AsInstanceOf[Char]
  }

  case class TTT(seq: NSeq[Int]) {
    def apply(x: Int) = {
      if (x >= seq.length) ???
      seq(x)
    }

    def last = seq.last
    def reverse = TTT(seq.reverse)
  }

  implicit def ttttoSeq(x: TTT) = x.seq

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

  /**
    * Defines tensor-specific operations.
    * Eventually, a tensor operation IR may be introduced to enable analyses/transformations.
    */
  trait Backend {
    def dot(x: Tensor, y: Tensor): Tensor
    // TODO: Add more ops.
  }

  /**
    * Native tensor op backend.
    * Tensor ops are defined in terms of primitive operations.
    */
  trait BackendNative extends Backend {
    override def dot(x: Tensor, y: Tensor): Tensor = {
      // TODO: remove loop if not needed
      val off = var_new(0)
      val up = if (x.nbDims > 1) x.dims(0) else 1
      val res = NewArray[Float](up)
      for (j <- DataLoop(up)) {
        //for (j <- (0 until up): Rep[Range]) {
        val value = var_new(0.0f)
        for (i <- DataLoop(x.dims.last)) {
          //for (i <- (0 until x.dims.last): Rep[Range]) {
          value += x.data(off) * y.data(i)
          off += 1
        }
        res(j) = readVar(value)
      }
      val dim = if (x.nbDims == 1) 1 else x.dims(0)
      Tensor(res, dim)
    }
  }

  /**
    * cuBLAS tensor op backend. WIP.
    */
  trait BackendCUBLAS extends Backend {
    // GEMM reference:
    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
    //
    // cublasStatus_t cublasSgemm(cublasHandle_t handle,
    //                            cublasOperation_t transa, cublasOperation_t transb,
    //                            int m, int n, int k,
    //                            const float           *alpha,
    //                            const float           *A, int lda,
    //                            const float           *B, int ldb,
    //                            const float           *beta,
    //                            float           *C, int ldc)
    def sgemm(a: Array[Float], b: Array[Float], c: Array[Float]) = unchecked[Array[Float]]("cublasSgemm(...)")

    override def dot(x: Tensor, y: Tensor): Tensor = ???
  }

  /**
    * Default tensor op backend, extending `BackendNative`.
    */
  class BackendDefault extends BackendNative
  val backend: Backend = new BackendDefault

  class Tensor(val data: Rep[Array[Float]], val dimsSeq: NSeq[Int]) extends Serializable {

    val MAX_DOUBLE = 1e10f // FIXME

    val strides = (dimsSeq :\ NSeq[Int]()) {
      case (dimX, seq@(h +: t)) => (dimX * h) +: seq
      case (dimX, _) => NSeq(dimX)
    }

    def dims = TTT(dimsSeq)

    assert(strides.length >= 1)
    assert(strides(0) != 0, "Empty Tensor!!!")

    val nbElem = strides(0)

    val nbDims = dimsSeq.length
    val isScalar = nbElem == 1

    def apply(i: Rep[Int]) = data(i)
    def apply(i: Rep[Int], j: Rep[Int]) = data(i * dims(1) + j) // FIXME the index of matrix is not the normal way

    @virtualize
    def clipAt(bound: Float) = {
      for (i <- DataLoop(nbElem)) {
        if (data(i) > bound) data(i) = bound
        if (data(i) < -1.0f * bound) data(i) = -1.0f * bound
      }
    }

    def mapInPlace(op: Rep[Float] => Rep[Float]) = {
      for (i <- DataLoop(nbElem)) this.data(i) = op(this.data(i))
    }

    def map(op: Rep[Float] => Rep[Float]) = {
      val res = NewArray[Float](nbElem)
      for (i <- DataLoop(nbElem)) res(i) = op(this.data(i))
      new Tensor(res, dims)
    }

    def fold(init: Rep[Float])(op: (Rep[Float], Rep[Float]) => Rep[Float]) = {
      val res = var_new[Float](init)
      for (i <- DataLoop(nbElem)) var_assign(res, op(res, this.data(i)))
      res
    }

    def +(that: Rep[Float]): Tensor = this.map(x => x + that)
    def +(that: Tensor): Tensor = {
      if (nbElem == 1) that + this.data(0)
      else if (that.nbElem == 1) this + that.data(0)
      else if (that.dims == this.dims) {
        val res = NewArray[Float](nbElem)
        for (i <- DataLoop(nbElem)) res(i) = this.data(i) + that.data(i)
        new Tensor(res, dims)
      }
      else throw new IllegalArgumentException(s"dimensions of vector do not match +! ${this.dims.seq} != ${that.dims.seq}")
    }

    // this operator updates the values of this, unlike the + operator
    def +=(that: Rep[Float]): Unit = this.mapInPlace(x => x + that)
    def += (that: Tensor): Unit = {
      if (that.nbElem == 1) {
        generate_comment("+= tensor of dim 0")
        this += that.data(0) // broadcast
      }
      else if (this.nbElem == 1) ??? // this.data(0) = that.fold(this.data(0))((agg, x) => agg + x)
      else if (this.dims == that.dims)
        for (i <- DataLoop(nbElem)) this.data(i) += that.data(i)
      else throw new IllegalArgumentException(s"dimensions of vector do not match +=! ${this.dims.seq} != ${that.dims.seq}")
    }

    def -(that: Rep[Float]): Tensor = this.map(x => x - that)
    def -(that: Tensor): Tensor = {
      if (nbElem == 1) that.map(x => this.data(0) - x)
      else if (that.nbElem == 1) this - that.data(0)
      else if (that.dims == this.dims) {
        val res = NewArray[Float](nbElem)
        for (i <- DataLoop(nbElem)) res(i) = this.data(i) - that.data(i)
        new Tensor(res, dims)
      }
      else throw new IllegalArgumentException("dimensions of vector do not match +!")
    }

    // this operator updates the values of this, unlike the - operator
    def -=(that: Rep[Float]): Unit = this.mapInPlace(x => x - that)
    def -= (that: Tensor): Unit = {
      if (that.nbElem == 1) this -= that.data(0) // broadcast
      else if (this.nbElem == 1) {
        ???
        // this.data(0) = that.fold(this.data(0))((agg, x) => agg - x)
      }
      else if (this.dims == that.dims)
        for (i <- DataLoop(nbElem)) this.data(i) -= that.data(i)
      else throw new IllegalArgumentException("dimensions of vector do not match +=!")
    }

    // Element wise multiplication
    def *(that: Rep[Float]): Tensor = this.map(x => x * that)
    def *(that: Tensor): Tensor = {
      if (nbElem == 1) that * this.data(0)
      else if (that.nbElem == 1) this * that.data(0)
      else if (that.dims == this.dims) {
        val res = NewArray[Float](nbElem)
        for (i <- DataLoop(nbElem)) res(i) = this.data(i) * that.data(i)
        new Tensor(res, dims)
      }
      else throw new IllegalArgumentException(s"dimensions of vector do not match * ${this.dims.seq} != ${that.dims.seq}")
    }

    // this operator updates the values of this, unlike the * operator
    def *=(that: Rep[Float]): Unit = this.mapInPlace(x => x * that)
    def *= (that: Tensor): Unit = {
      if (that.nbElem == 1) this *= that.data(0) // broadcast
      else if (this.nbElem == 1) {
        ???
        // this.data(0) = that.fold(this.data(0))((agg, x) => agg * x)
      }
      else if (this.dims == that.dims)
        for (i <- DataLoop(nbElem)) this.data(i) *= that.data(i)
      else throw new IllegalArgumentException("dimensions of vector do not match +=!")
    }

    // element wise division
    def /(that: Rep[Float]): Tensor = this.map(x => x / that)
    def /(that: Tensor): Tensor = {
      if (nbElem == 1) that.map(x => this.data(0) / x)
      else if (that.nbElem == 1) this / that.data(0)
      else if (that.dims == this.dims) {
        val res = NewArray[Float](nbElem)
        for (i <- DataLoop(nbElem)) res(i) = this.data(i) / that.data(i)
        new Tensor(res, dims)
      }
      else throw new IllegalArgumentException("dimensions of vector do not match +!")
    }

    // this operator updates the values of this, unlike the / operator
    def /=(that: Rep[Float]): Unit = this.mapInPlace(x => x / that)
    def /= (that: Tensor): Unit = {
      if (that.nbElem == 1) this /= that.data(0) // broadcast
      else if (this.nbElem == 1) ??? // this.data(0) = that.fold(this.data(0))((agg, x) => agg / x)
      else if (this.dims == that.dims)
        for (i <- DataLoop(nbElem)) this.data(i) /= that.data(i)
      else throw new IllegalArgumentException("dimensions of vector do not match +=!")
    }

    def setAsOne() = { this.mapInPlace(x => 1.0f); () }
    def clear() = { this.mapInPlace(x => 0.0f); () }

    def copy_data(that: Tensor) = {
      assert(this.nbElem == that.nbElem, "dimensions of vector do not match copy_data!")
      for (i <- DataLoop(nbElem)) this.data(i) = that.data(i)
    }

    // NOTE: only handles (Matrix dot Vector) and (Vector dot Vector)
    def dot(that: Tensor) = {
      // assert that and this have the same dimension
      generate_comment(s"dot ${this.dims.seq} - ${that.dims.seq}")
      assert(this.nbDims <= 2 && that.nbDims == 1, s"Only M x V or V x V allowed ${this.dims} - ${that.dims}")
      assert(this.dims.last == that.dims(0), s"dimensions of vector do not match dot! ${this.dims.seq} - ${that.dims.seq}")
      backend.dot(this, that)
    }

    // NOTE: only handles (Vector cart Vector)
    def cart(that: Tensor) = {
      assert(this.nbDims == 1 && that.nbDims == 1, "cartesian product is only for 1d vectors")
      val res = NewArray[Float](this.dims(0) * that.dims(0))
      val off = var_new(0)
      for (i <- DataLoop(this.dims(0))) {
      //for (i <- (0 until this.dims(0)): Rep[Range]) {
        for (j <- DataLoop(that.dims(0))) {
        //for (j <- (0 until that.dims(0)): Rep[Range]) {
          res(off) = data(i) * that.data(j)
          off += 1
        }
      }
      Tensor(res, this.dims(0), that.dims(0))
    }

    def trans() = {
      assert(this.nbDims == 2, "transpose is only for matrix. Tensor transpose is not supported here")
      val res = NewArray[Float](this.nbElem)
      val offT = var_new(0)
      for (i <- DataLoop(this.dims(1))) {
      //for (i <- (0 until this.dims(1)): Rep[Range]) {
        val off = var_new(0)
        for (j <- DataLoop(this.dims(0))) {
        //for (j <- (0 until this.dims(0)): Rep[Range]) {
          res(offT + j) = data(off + i)
          off += this.dims(1)
        }
        offT += this.dims(0)
      }
      new Tensor(res, this.dims.reverse)
    }

    def tanh() = this.map(x => Math.tanh(x).toFloat)
    def exp() = this.map(x => Math.exp(x).toFloat)
    def log() = this.map(x => Math.log(x).toFloat)
    def sqrt() = this.map(x => Math.sqrt(x).toFloat)
    def sigmoid() = this.map(x => 1.0f / (Math.exp(-1.0f * x).toFloat + 1.0f))

    // NOTE: sum all elements
    def sum() = Tensor.scalar(this.fold(0.0f)(_ + _))

    @virtualize
    def check(limit: Float) = {
      val idx = var_new(0)
      while (idx < this.nbElem && -limit < this.data(idx) && this.data(idx) < limit) {
        idx += 1
      }

      idx != this.nbElem
    }

    @virtualize
    def max() = this.fold(-MAX_DOUBLE)((agg, x) => if (x > agg) x else agg)

    // FIXME: Proper tensor
    @virtualize
    def maxIndex() = {
      assert(this.nbDims == 1)
      val vMax = var_new(this.data(0))
      val iMax = var_new(0)
      for (idx <- 1 until this.nbElem: Rep[Range]) {
        if (this.data(idx) > vMax) {
          iMax = idx
          vMax = this.data(idx)
        }
      }

      iMax
    }

    @virtualize
    def logSoftmax() = {
      assert(this.nbDims == 1, "TODO")

      val m = this.max
      val logsum = m + Math.log(this.fold(0.0f)((agg, x) => agg + Math.exp(x - m).toFloat)).toFloat
      this.map(x => x - logsum)
    }

    @virtualize
    def nllLoss(target: Rep[Int]) = {
      assert(this.nbDims == 1)

      // assertC(0 <= target && target < this.nbElem, "Incorrect target")
      Tensor.scalar(-1.0f * this.data(target))
    }

    def resize(dims: Int*) = {
      assert(dims.product == this.nbElem)

      Tensor(this.data, dims : _*)
    }


    // NOTE: sum matrix to vector, condense on the dims(1) dimension
    def sumOnDim1() = {
      assert(this.nbDims <= 2)
      if (this.nbDims == 1) this
      else {
        val res = NewArray[Float](this.dims(1))
        val off = var_new(0)
        for (j <- DataLoop(this.dims(1))) {
        //for (j <- (0 until this.dims(1)): Rep[Range]) {
          res(off) = this.data(off)
          off += 1
        }

        for (i <- (1 until this.dims(0)): Rep[Range]) {
          val offR = var_new(0)
          for (j <- DataLoop(this.dims(1))) {
          //for (j <- (0 until this.dims(1)): Rep[Range]) {
            res(offR) += data(off)
            off += 1
            offR += 1
          }
        }
        Tensor(res, this.dims(1))
      }
    }

    def print(msg: String = ""): Unit = {
      if (msg != "")
        printf(s"$msg (size ${this.dims.seq mkString " x "})\\n")
      if (this.nbDims == 4) this.print4D
      else if (this.nbDims == 3) this.print3D
      else this.printRaw(this.dims.last)
    }

    val format = "%.10f "
    def print4D = {
      val idx = var_new(1)
      for (i <- 0 until this.dims(0): Rep[Range]) {
        val idx1 = var_new(1)
        for (j <- 0 until this.dims(1): Rep[Range]) {
          printf(s"Pane #(%d, %d) - ${this.dims(2)} x ${this.dims(3)}\\n", idx, idx1)
          for (k <- 0 until this.dims(2): Rep[Range]) {
            for (l <- 0 until this.dims(3): Rep[Range]) {
              printf(format, this.data(i * this.strides(1) + j * this.strides(2) + k * this.strides(3) + l))
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
      for (i <- 0 until this.dims(0): Rep[Range]) {
        printf(s"Pane #%d - ${this.dims(1)} x ${this.dims(2)}\\n", idx)
        for (k <- 0 until this.dims(1): Rep[Range]) {
          for (l <- 0 until this.dims(2): Rep[Range]) {
            printf(format, this.data(i * this.strides(1) + k * this.strides(2) + l))
          }
          printf("\\n")
        }
        printf("\\n\\n")
        idx += 1
      }
    }

    @virtualize
    def printRaw(row: Int = 10) = {
      for (i <- 0 until this.nbElem: Rep[Range]) {
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
      assert(this.nbDims == 2 && that.dims == TTT(NSeq(this.dims(1))) && y.dims == TTT(NSeq(this.dims(0))) ||
        this.nbDims == 1 && that.dims == this.dims && y.isScalar, s"${dims} - ${that.dims} - ${y.dims}")
      val off = var_new(0)
      // TODO remove loop if not used
      val up = if (this.nbDims > 1) this.dims(0) else 1
      for (i <- DataLoop(up)) {
      //for (i <- (0 until up): Rep[Range]) {
        for (j <- DataLoop(dims(1))) {
        //for (j <- (0 until dims(1)): Rep[Range]) {
          this.data(off + j) = this.data(off + j) + that.data(j) * y.data(i)
        }
        off += this.dims(1)
      }
    }
    // FIXME: Maybe try to support slicing??
    // FIXME: Maybe add support for reshaping??
    // FIXME: Maybe support transposing??


    // setting: this is dims(0)-sized vector, that is matrix (dims(0) * dims(1)), y is dims(1)-sized vector
    // the result is to update this so that this accumulate every matrix col * y
    def add_composion(that: Tensor, y: Tensor) = {
      assert(that.nbDims == 2 && this.dims.seq == NSeq(that.dims(1)) && y.dims.seq == NSeq(that.dims(0))
        || that.nbDims == 1 && this.dims == that.dims && y.isScalar, s"${dims} - ${that.dims} - ${y.dims}")
      val off = var_new(0)
      // FIXME!!
      val up = if (that.nbDims > 1) that.dims(0) else 1
      for (i <- DataLoop(up)) {
      //for (i <- (0 until up): Rep[Range]) {
        for (j <- DataLoop(that.dims(1))) {
        //for (j <- (0 until that.dims(1)): Rep[Range]) {
          data(j) += that.data(off + j) * y.data(i)
        }
        off += that.dims(1)
      }
    }
    // def add_composion(that: Tensor, y: Tensor) = {
    //   if (this.nbDims == 1)
    //     this.resize(that.dims(0), )
    // }

    @virtualize
    def addMul(that: Tensor, y: Tensor) = {
      assert(this.nbDims == 2 && that.nbDims == 2 && y.nbDims == 2, s"Dimensions: ${this.dims.seq} - ${that.dims.seq} - ${y.dims.seq}")
      assert(this.dims(0) == that.dims(0) && this.dims(1) == y.dims(1) && that.dims(1) == y.dims(0), s"Dimensions: ${this.dims.seq} + ${that.dims.seq} * ${y.dims.seq}")

      var offThis = var_new(0)
      var offThatR = var_new(0)
      var offYC = var_new(0)
      for (i <- DataLoop(this.dims(0))) {
      //for (i <- 0 until this.dims(0): Rep[Range]) {
        val offYR = var_new(offYC)
        for (j <- DataLoop(this.dims(1))) {
        //for (j <- 0 until this.dims(1): Rep[Range]) {
          val offY = var_new(offYR)
          val offThat = var_new(offThatR)
          for (k <- DataLoop(that.dims(1))) {
          //for (k <- 0 until that.dims(1): Rep[Range]) {
            // assertC(unit(0) <= offThis && offThis < this.nbElem, s"Index error this %d > ${this.nbElem}", offThis)
            // assertC(unit(0) <= offThat && offThat < that.nbElem, s"Index error that %d > ${that.nbElem}", offThat)
            // assertC(unit(0) <= offY && offY < y.nbElem, s"Index error this %d > ${y.nbElem}", offY)
            this.data(offThis) = this.data(offThis) + that.data(offThat) * y.data(offY)
            offThat += 1
            offY += y.strides(1)
          }
          offThis += 1
          offYR += 1
        }
        offThatR += that.strides(1)
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
      val dims0M = mmax(dims(0), mmax(a.dims(0), b.dims(0)))
      val dims1M = mmax(if (this.nbDims > 1) dims(1) else 1, mmax(if (a.nbDims > 1) a.dims(1) else 1, if (b.nbDims > 1) b.dims(1) else 1))
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
      assert(this.dims == b.dims)

      generate_comment("Generate code for addMul")
      for (i <- DataLoop(this.nbElem)) {
      //for (i <- 0 until this.nbElem: Rep[Range]) {
        this.data(i) = this.data(i) + a * b.data(i)
      }
    }

    def cmulAdd(a: Float, b: Tensor) = {
      assert(this.dims == b.dims)
      for (i <- DataLoop(this.nbElem))
      //for (i <- 0 until this.nbElem: Rep[Range])
        this.data(i) = a * this.data(i) + b.data(i)

      this // FIXME ??
    }

    def add_div(a: Tensor, b: Tensor) = {
      assert(Tensor.dimCompatible(a, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, b), "dim not Compatible in add_div")
      val dims0M = mmax(dims(0), mmax(a.dims(0), b.dims(0)))
      // FIXME
      val dims1M = mmax(if (nbDims > 1) dims(1) else 1, mmax(if (a.nbDims > 1) a.dims(1) else 1, if (b.nbDims > 1) b.dims(1) else 1))
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
      val dims0M = mmax(dims(0), mmax(a.dims(0), mmax(b.dims(0), c.dims(0))))
      // FIXME
      val dims1M = mmax(if (nbDims > 1) dims(1) else 1, mmax(if (a.nbDims > 1) a.dims(1) else 1, if (b.nbDims > 1) b.dims(1) else 1))
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
      val dims0M = mmax(dims(0), mmax(a.dims(0), b.dims(0)))
      // FIXME
      val dims1M = mmax(if (nbDims > 1) dims(1) else 1, mmax(if (a.nbDims > 1) a.dims(1) else 1, if (b.nbDims > 1) b.dims(1) else 1))
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
      val dims0M = mmax(dims(0), mmax(a.dims(0), b.dims(0)))
      // FIXME
      val dims1M = mmax(if (nbDims > 1) dims(1) else 1, mmax(if (a.nbDims > 1) a.dims(1) else 1, if (b.nbDims > 1) b.dims(1) else 1))
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
    def conv2D(kernel: Tensor, strideRow: Int, strideCol: Int): Tensor = {

      assert(this.nbDims == 3 && kernel.nbDims == 4)

      assert(strideRow >= 1)
      assert(strideCol >= 1)

      assert(kernel.dims(1) == this.dims(0))
      assert(this.dims(1) >= kernel.dims(2) && this.dims(2) >= kernel.dims(3), "Image too small")

      val resHeight = convSize(this.dims(1), kernel.dims(2), strideRow)
      val resWidth = convSize(this.dims(2), kernel.dims(3), strideCol)
      val res = Tensor.zeros(kernel.dims(0), resHeight, resWidth)

      val offOut = var_new(0)
      val offWeight1 = var_new(0)
      for (outPane <- DataLoop(kernel.dims(0))) {
      //for (outPane <- 0 until kernel.dims(0): Rep[Range]) {
        // assertC(offOut == outPane * res.strides(1), "Invalid Output Idx %d != %d (%d)", offOut, outPane * res.strides(1), outPane)
        // assertC(offWeight1 == outPane * kernel.strides(1), "Invalid Kernel Idx")
        val offWeight2 = var_new(offWeight1)
        val offInput = var_new(0)
        val ptrOutput = slice(res.data, offOut)
        for (inPane <- DataLoop(this.dims(0))) {
        //for (inPane <- 0 until this.dims(0): Rep[Range]) {
          // assertC(offInput == inPane * this.strides(1), "Invalid Input Idx")
          // assertC(offWeight2 == outPane * kernel.strides(1) + inPane * kernel.strides(2), "Invalid kernel Idx (2) %d != %d (%d - %d)", offWeight1, outPane * kernel.strides(1) + inPane * kernel.strides(2), outPane, inPane)
          val ptrIntput = slice(this.data, offInput)
          val ptrWeight = slice(kernel.data, offWeight2)

          Tensor(ptrOutput, resHeight, resWidth).conv2D(Tensor(ptrIntput, this.dims(1), this.dims(2)), Tensor(ptrWeight, kernel.dims(2), kernel.dims(3)), strideRow, strideCol)

          offWeight2 += kernel.strides(2)
          offInput += this.strides(1)
        }
        offWeight1 += kernel.strides(1)
        offOut += res.strides(1)
      }
      res
    }

    // Taken from Torch: THTensorConv.cpp#198L
    @virtualize
    def conv2D(input: Tensor, kernel: Tensor, strideRow: Int, strideCol: Int): Unit = {
      assert(this.nbDims == 2 && input.nbDims == 2 && kernel.nbDims == 2)
      assert(strideRow >= 1)
      assert(strideCol >= 1)

      val offOuput = var_new(0)
      val offInputR = var_new(0)
      for (outRow <- DataLoop(this.dims(0))) {
      //for (outRow <- 0 until this.dims(0): Rep[Range]) {
        // assertC(offInputR == outRow * input.strides(1), "intputR invalid")
        val offInputC = var_new(offInputR)
        for (outCol <- DataLoop(this.dims(1))) {
        //for (outCol <- 0 until this.dims(1): Rep[Range]) {
          // assertC(offInputC == outRow * strideRow * input.strides(1) + outCol * strideCol, "intputC invalid")
          val offKernel = var_new(0)
          val offInput = var_new(offInputC)
          val sum = var_new(0.0f)
          for (kernelRow <- DataLoop(kernel.dims(0))) {
          //for (kernelRow <- 0 until kernel.dims(0): Rep[Range]) {
            // assertC(offInput == (outRow * strideRow + kernelRow) * input.strides(1) + outCol * strideCol, "input invalid")
            // assertC(offKernel == kernelRow * kernel.strides(1), "kernel invalid")
            val ptrIntput = slice(input.data, offInput)
            val ptrKernel = slice(kernel.data, offKernel)
            for (kernelCol <- DataLoop(kernel.dims(1))) {
            //for (kernelCol <- 0 until kernel.dims(1): Rep[Range]) {
              sum +=  ptrIntput(kernelCol) * ptrKernel(kernelCol)
            }
            offKernel += kernel.strides(1)
            offInput += input.strides(1)
          }
          this.data(offOuput) = this.data(offOuput) + sum
          offOuput += 1
          offInputC += strideCol
        }
        offInputR += strideRow * input.strides(1)
      }
    }

    // TO HERE DataLoop

    @virtualize
    def maxPool(strideRow: Int, strideCol: Int) = {
      assert(this.nbDims == 3)


      val resHeight = this.dims(1) / strideRow
      val resWidth = this.dims(2) / strideCol
      val res = Tensor.fill(-MAX_DOUBLE, this.dims(0), resHeight, resWidth)

      // FIXME adhoc transform tensor to be using generic type!
      val savedIdx = NewArray[Int](res.nbElem)


      val oidxW = var_new(0)
      val iidx = var_new(0)
      for (ichan <- 0 until this.dims(0): Rep[Range]) {
        val oidx = var_new(oidxW)
        for (ox <- 0 until res.dims(1): Rep[Range]) {
          for (sx <- 0 until strideRow: Rep[Range]) {
            // assertC(oidx == ichan * res.strides(1) + ox * res.strides(2), "MAXPOOL output row idx error %d != %d (%d - %d - %d)\\n", oidx, ichan * this.strides(1) + ox * res.strides(2), ichan, ox, sx)
            val oidx2 = var_new(oidx)
            for (oy <- 0 until res.dims(2): Rep[Range]) {
              for (sy <- 0 until strideCol: Range) { // FIXME unrol by default
                // assertC(iidx == ichan * this.strides(1) + (ox * strideRow + sx) * this.strides(2) + oy * strideCol + sy, "MAXPOOL input idx error %d != %d (%d - %d (%d), %d (%d))\\n", iidx, ichan * this.strides(1) + (ox * strideRow + sx) * this.strides(2) + oy * strideCol + sy, ichan, ox, sx, oy, sy)
                // assertC(oidx2 == ichan * res.strides(1) + ox * res.strides(2) + oy, "MAXPOOL output idx error %d != %d (%d - %d (%d), %d (%d))\\n", oidx2, ichan * res.strides(1) + ox * res.strides(2) + oy, ichan, ox, sx, oy, sy)
                if (this.data(iidx) > res.data(oidx2)) {
                  res.data(oidx2) = this.data(iidx)
                  savedIdx(oidx2) = iidx
                }
                iidx += 1
              }
              oidx2 += 1
            }
          }
          oidx += res.strides(2)
        }
        oidxW += res.strides(1)
      }

      (res, savedIdx)
    }

    @virtualize
    def dropout(prob: Float = 0.5f) = {
      assert(0.0f <= prob && prob <= 1.0f)

      val res = NewArray[Float](this.nbElem)
      val mask = NewArray[Float](this.nbElem)

      val scale = if (prob < 1.0f) 1.0f / (1.0f - prob) else 0.0f

      val guard: Rep[Boolean] = prob < 1.0f
      for (i <- 0 until this.nbElem: Rep[Range]) {
        if (guard && Random.rand() > prob) {
          res(i) = this.data(i) * scale
          mask(i) = scale
        } else {
          res(i) = 0.0f
          mask(i) = 0.0f
        }
      }

      (Tensor(res, this.dims.seq : _*), Tensor(mask, this.dims.seq : _*))
    }

    @virtualize
    def relu(inPlace: Boolean = false) = {
      assert(!inPlace)

      val res = NewArray[Float](this.nbElem)
      for (i <- 0 until this.nbElem: Rep[Range]) {
        if (this(i) < 0.0f)
          res(i) = 0.0f
        else
          res(i) = this.data(i)
      }

      Tensor(res, this.dims.seq : _*)
    }

    // FIXME: the MNIST example precomput the mean and std
    // I thought that normalize would need to compute it first and then
    // modify the data to match the one requested.
    // SO here what is expected is to have mean = 0 and std = 1 knowing that
    // the current mean is m and the current std is s
    @virtualize
    def normalize(m: Float, s: Float, inPlace: Boolean = false) = {
      assert(this.nbDims == 3 && this.dims(0) == 1) // FIXME
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
      ???
      val size = dims.product
      new Tensor(NewArray[Float](size), dims)
    }
    def apply(data: Rep[Array[Float]], dims: Int*) = new Tensor(data, dims)

    def dimCompatible(a: Tensor, b: Tensor) = {
      (a.dims == b.dims) || a.isScalar || b.isScalar
    }

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

    def zeros(dims: Int*): Tensor = {
      fill(0.0f, dims: _*)
    }

    def zeros(that: Tensor): Tensor = {
      zeros(that.dims : _*)
    }

    def zeros_like(that: Tensor) = {
      zeros(that.dims : _*)
    }

    def scalar(value: Rep[Float]) = {
      val res = NewArray[Float](1)
      res(0) = value
      Tensor(res, 1)
    }

    def ones(dims: Int*) = fill(1.0f, dims: _*)
    def ones(that: Tensor) = fill(1.0f, that.dims: _*)
    def halves(dims: Int*) = fill(0.5f, dims: _*)

    def expand(vector: Tensor, dim1: Int) = {
      assert(vector.nbDims == 1)
      val res = NewArray[Float](vector.dims(0) * dim1)
      val off = var_new(0)
      for (j <- (0 until dim1): Rep[Range]){
        for (i <- (0 until vector.dims(0)): Rep[Range]) {
          res(off) = vector.data(i)
          off += 1
        }
      }
      new Tensor(res, dim1 +: vector.dims)
    }

    def copy(vector: Tensor) = {
      val res = NewArray[Float](vector.nbElem)
      for (i <- (0 until vector.nbElem): Rep[Range]) res(i) = vector.data(i)
      new Tensor(res, vector.dims)
    }

    def fromData(x: Float*) = {
      val y = x.toArray
      val res = NewArray[Float](y.length)
      for (i <- 0 until y.length: Range) res(i) = y(i)
      Tensor(res, y.length)
    }


    // def conv(that: Tensor, stride: (Int, Int) = (1, 1))

    @virtualize
    def assertEqual(a: Tensor, b: Tensor, mark: String = "", tal: Float = 0.000001f) = {
      assert(a.dims == b.dims, s"ERROR: $mark not equal in dimensionsi ${a.dims.seq} != ${b.dims.seq}\\n")

      val i = var_new(0)
      while (i < a.nbElem && { val diff = a.data(i) - b.data(i); diff > -tal && diff < tal }) {
        i += 1
      }
      if (i < a.nbElem)
        printf("ERROR: %s not equal in some data - %.4f != %.4f (%d)\\n", mark, a.data(i), b.data(i), i)
    }
  }


  // Tensor type is the similar to NumR, just replace RFloat with Tensor
  // also Tensor internally use array, which is mutable by default
  // so both field are val (not var) and can be updated by += -= *= /= setAsOne()
  // all instances of vectors will be shepherded by c++ smart pointers, alleviating the memory leak problem
  type diff = cps[Unit]

  class TensorR(val x: Tensor, val d: Tensor) extends Serializable {
    var isInput: Boolean = false // true if it is an input (no need to compute gradient)

    def clip_grad(bound: Float) = {
      d.clipAt(bound)
    }

    def + (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x + that.x);
      k(y)
      generate_comment("backpropagate +")
      this.d += y.d; that.d += y.d
    }

    def - (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x - that.x); k(y)
      //y.d.print("dot")
      this.d += y.d; that.d -= y.d
    }

    // this is element wise multiplication
    def * (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x * that.x); k(y)
      // intermediate Tensors donot need to be substatiated, can optimize!
      //this.d += that.x * y.d; that.d += this.x * y.d;
      this.d.add_mult(that.x, y.d); that.d.add_mult(this.x, y.d)
    }

    // element wise division
    def / (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x / that.x); k(y)
      // intermediate Tensors donot need to be substatiated, can optimize!
      //this.d += y.d / that.x
      this.d.add_div(y.d, that.x)
      //that.d -= this.x * y.d / (that.x * that.x)
      that.d.minus_mult_div_square(this.x, y.d, that.x)
    }

    // vector dot product or Matrix vector dot (viewed as multiple vector dot product) (not the common view)
    def dot(that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val res = x dot that.x
      val y = TensorR(res); k(y)
      // FIXME: intermediate Tensors donot need to be substatiated, can optimize!
      //y.d.print("dot")
      if (this.d.nbDims == 1) {
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

      //y.d.print("log")
      val s = y.d.sum().data(0)
      for (i <- 0 until y.x.nbElem: Rep[Range]) {
        this.d.data(i) = y.d.data(i) - Math.exp(y.x.data(i)).toFloat * s
      }
    }

    def resize(dims: Int*): TensorR @diff = shift { (k: TensorR => Unit) =>
      k(new TensorR(this.x.resize(dims : _*), this.d.resize(dims : _*)))
    }

    def nllLoss(target: Rep[Int]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.nllLoss(target)); k(y)

      assert(y.x.isScalar)
      //y.d.print("nll")

      this.d.data(target) = -1.0f * y.d.data(0)
    }

    def update(lr: Float, mom: Float) = {
    }


    @virtualize
    def conv(kernel: TensorR, strideRow: Int, strideCol: Int, tot: Rep[Array[Long]]): TensorR @diff = shift { (k: TensorR => Unit) =>
      // val timer = Timer2()
      // timer.startTimer
      val y = TensorR(x conv2D(kernel.x, strideRow, strideCol))
      // tot(0) += timer.getElapsedTime
      k(y)
      //y.d.print("conv")

      // val timerBwd = Timer2()
      // TODO think about the loop order
      val offOutputD = var_new(0)
      val offKernel = var_new(0)
      assert(y.d.dims(0) == kernel.x.dims(0))
      // timerBwd.startTimer
      for (kOut <- 0 until y.d.dims(0): Rep[Range]) { // forall output pane
        val offInputR = var_new(0)
        for (row <- 0 until y.d.dims(1): Rep[Range]) {
          // assertC(offInputR == row * strideRow * this.x.strides(2), s"ERROR: input offsetR %d != %d (%d, %d)\\n", offInputR, row * strideRow * this.x.strides(2), kOut, row)
          val offInputC = var_new(offInputR)
          for (col <- 0 until y.d.dims(2): Rep[Range]) {
            val dCurr: Rep[Float] = y.d.data(offOutputD)

            val offInputP = var_new(offInputC)
            val offKernelR = var_new(offKernel)
            assert(this.d.dims(0) == kernel.d.dims(1))
            for (pane <- 0 until this.d.dims(0): Rep[Range]) {
              // assertC(offInputP == pane * this.x.strides(1) + row * strideRow * this.x.strides(2) + col * strideCol, s"ERROR: input offsetC %d != %d (%d, %d, %d, %d)\\n", offInputP, pane * this.x.strides(1) + row * strideRow * this.x.strides(2) + col * strideCol, kOut, row, col, pane)
              val offInputKR = var_new(offInputP)
              for (kR <- 0 until kernel.d.dims(2): Rep[Range]) {
                // assertC(offInputKR == pane * this.x.strides(1) + (row * strideRow + kR) * this.x.strides(2) + col * strideCol, s"ERROR: input offset %d != %d (%d, %d)\\n", offInputKR, pane * this.x.strides(1) + (row + kR) * strideRow * this.x.strides(2) + col * strideCol, pane, kR)
                for (kC <- 0 until kernel.d.dims(3): Rep[Range]) {
                  assert(this.isInput || this.d.nbElem == this.x.nbElem)
                  // assertC(unit[Int](0) <= offInputKR + kC && offInputKR + kC <= this.d.nbElem, "Bounds check error\\n")
                  if (!this.isInput)
                    this.d.data(offInputKR + kC) = this.d.data(offInputKR + kC) + dCurr * kernel.x.data(offKernelR)
                  // assertC(unit[Int](0) <= offKernelR && offKernelR <= kernel.d.nbElem, "Bounds check error\\n")
                  kernel.d.data(offKernelR) = kernel.d.data(offKernelR) + dCurr * this.x.data(offInputKR + kC)
                  offKernelR += 1
                }
                offInputKR += this.x.strides(2)
              }
              offInputP += this.x.strides(1)
            }
            offInputC += strideCol
            // assertC(offKernelR/(kOut + 1) == kernel.x.strides(1), s"ERROR: kernel size %d != (%d + 1) * ${kernel.x.strides(1)}\\n", offKernelR, kOut)
            offOutputD += 1
          }
          offInputR += strideRow * this.x.strides(2)
        }
        offKernel += kernel.x.strides(1)
      }
      // tot(1) += timerBwd.getElapsedTime

      ()
    }

    @virtualize
    def maxPool(strideRow: Int, strideCol: Int): TensorR @diff = shift { (k: TensorR => Unit) =>
      val (y, sidx) = this.x.maxPool(strideRow, strideCol)

      val ty = TensorR(y)
      k(ty)

      //t//y.d.print("Maxpool")

      for (i <- 0 until y.nbElem: Rep[Range]) {
        this.d.data(sidx(i)) = ty.d.data(i)
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

      for (i <- 0 until this.x.nbElem: Rep[Range]) {
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

    def add(x: TensorR, y: TensorR): TensorR @diff = x + y
    def dot(x: TensorR, y: TensorR): TensorR @diff = x.dot(y)
  }

  // change fun signature for memory leak issue (no more returning of array, just update the array provided by the caller)
  // this is in accordance of the destination-programming style
  // the fun take array[array[double]] of size 2, with the first array to be the x, and the second array to be the d
  def FUNc(dim0: Int)(f: TensorR => Unit): (TensorR => Unit) = {
    val f1 = fun { (x: Rep[Array[Array[Float]]]) =>
      val tensor = new TensorR(Tensor(x(0), dim0), Tensor(x(1), dim0))
      f(tensor)
    };
    {
      (x:TensorR) => {
        val in = NewArray[Array[Float]](2)
        in(0) = x.x.data; in(1) = x.d.data
        f1(in) // f1 should take Array[Array[Float]] and update the gradient of x
      }
    }
  }

  def RST(a: => Unit @diff) = continuations.reset {
    a;
    ()
  }

  @virtualize
  def IF(dim0: Int)(c: Rep[Boolean])(a: =>TensorR @diff)(b: =>TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>
    val k1 = FUNc(dim0)(k)

    if (c) RST(k1(a)) else RST(k1(b))
  }

  @virtualize
  def LOOP(init: TensorR)(c: TensorR => Rep[Boolean])(b: TensorR => TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>
    // val k1 = FUN(init.x.dims(0))(k)

    lazy val loop: TensorR => Unit = FUNc (init.x.dims(0)) { (x: TensorR) =>
      if (c(x)) RST(loop(b(x))) else RST(k(x))
    }
    loop(init)
  }

  def FUNs(dim0: Int)(f: Rep[Int] => TensorR => Unit): (Rep[Int] => TensorR => Unit) = {
    val f1 = fun { (xx: Rep[(Int, Array[Array[Float]])]) =>
      val i: Rep[Int]                  = tuple2_get1(xx)
      val x: Rep[Array[Array[Float]]] = tuple2_get2(xx)
      val tensor = new TensorR(Tensor(x(0), dim0), Tensor(x(1), dim0))
      f(i)(tensor)
    };
    {
      (i: Rep[Int]) => (x:TensorR) => {
        val in = NewArray[Array[Float]](2)
        in(0) = x.x.data; in(1) = x.d.data
        f1((i, in))
      }
    }
  }

  @virtualize
  def LOOPS(init: TensorR)(c: Rep[Int])(b: Rep[Int] => TensorR => TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>
    lazy val loop: Rep[Int] => TensorR => Unit = FUNs (init.x.dims(0)) { (i: Rep[Int]) => (x: TensorR) =>
      if (i < c) { RST(loop(i+1)(b(i)(x))) } else RST(k(x))
    }
    loop(0)(init)
  }

  def FUNsm(dim0s: ArrayBuffer[Seq[Int]])(f: Rep[Int] => ArrayBuffer[TensorR] => Unit): (Rep[Int] => ArrayBuffer[TensorR] => Unit) = {
    val f1 = fun { (xx: Rep[(Int, Array[Array[Float]])]) =>
      val i: Rep[Int]                  = tuple2_get1(xx)
      val x: Rep[Array[Array[Float]]] = tuple2_get2(xx)
      val tensors = ArrayBuffer[TensorR]()
      for (u <- (0 until dim0s.length): Range) {
        tensors.append(new TensorR(Tensor(x(u*2), dim0s(u) : _*), Tensor(x(u*2+1), dim0s(u) : _*)))
      }
      f(i)(tensors)
    };
    (i: Rep[Int]) => (x:ArrayBuffer[TensorR]) => {
      val in = NewArray[Array[Float]](2 * dim0s.length)
      for (u <- (0 until dim0s.length): Range) {
        in(u*2) = x(u).x.data; in(u*2+1) = x(u).d.data
      }
      f1((i, in))
    }
  }

  @virtualize
  def LOOPSM(init: ArrayBuffer[TensorR])(c: Rep[Int])(b: Rep[Int] => ArrayBuffer[TensorR] => ArrayBuffer[TensorR] @diff):
  ArrayBuffer[TensorR] @diff = shift { k: (ArrayBuffer[TensorR] => Unit) =>
    lazy val loop: Rep[Int] => ArrayBuffer[TensorR] => Unit = FUNsm (init map (_.x.dims.seq)) { (i: Rep[Int]) => (x: ArrayBuffer[TensorR]) =>
      if (i < c) {
        RST(loop(i+1)(b(i)(x)))
      } else {
        RST(k(x))
      }
    }
    loop(0)(init)
  }

  def FUNl(dim0: Int)(f: (Rep[Int] => (TensorR => Unit) => (TensorR => Unit))): (Rep[Int] => (TensorR => Unit) => (TensorR => Unit)) = {

    val f1 = fun { (yy: Rep[(Int, (Array[Array[Float]] => Unit), Array[Array[Float]])]) =>
      val i:  Rep[Int] = tuple3_get1(yy)
      val t1: Rep[Array[Array[Float]] => Unit] = tuple3_get2(yy)
      val xx: Rep[Array[Array[Float]]] = tuple3_get3(yy)
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
          f1((i, k2, temp))
        }
        k4
      }
    }
  }

  @virtualize
  def LOOPL(init: TensorR)(c: Rep[Int])(b: Rep[Int] => TensorR => TensorR @diff): TensorR @diff = shift { k: (TensorR => Unit) =>
    lazy val loop: Rep[Int] => (TensorR => Unit) => TensorR => Unit = FUNl(init.x.dims(0)){ (gc: Rep[Int]) => (k: TensorR => Unit) => (x: TensorR) =>
      if (gc < c) { loop(gc+1)((x: TensorR) => RST(k(b(gc)(x))))(x) } else { RST(k(x)) }
    }
    loop(0)(k)(init)
  }

  @virtualize
  def LOOPT(init: TensorR)(lch: Rep[Array[Int]], rch: Rep[Array[Int]])(b: (TensorR, TensorR, Rep[Int]) => TensorR @diff): TensorR @diff = shift {
    k: (TensorR => Unit) =>

      lazy val tree: Rep[Int] => (TensorR => Unit) => TensorR => Unit = FUNl(init.x.dims(0)){ (i: Rep[Int]) => (k: TensorR => Unit) => (x: TensorR) =>
        if (i >= 0) { tree(lch(i))((l: TensorR) => tree(rch(i))((r: TensorR) => RST(k(b(l, r, i))))(x))(x) } else { RST(k(x)) }
      }
      tree(0)(k)(init)
  }

  def FUNlm(dim0s: ArrayBuffer[Int])(f: (Rep[Int] => (ArrayBuffer[TensorR] => Unit) => (ArrayBuffer[TensorR] => Unit))):
  (Rep[Int] => (ArrayBuffer[TensorR] => Unit) => (ArrayBuffer[TensorR] => Unit)) = {
    val length = dim0s.length
    val f1 = fun { (yy: Rep[(Int, (Array[Array[Float]] => Unit), Array[Array[Float]])]) =>
      val i: Rep[Int] = tuple3_get1(yy)
      val t1: Rep[Array[Array[Float]] => Unit] = tuple3_get2(yy)
      val xx: Rep[Array[Array[Float]]] = tuple3_get3(yy)

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
        tensors.append(new TensorR(Tensor(xx(u*2), dim0s(u)), Tensor(xx(u*2+1), dim0s(u))))
      }
      t3(tensors)
    };

    {i: Rep[Int] => k1: (ArrayBuffer[TensorR] => Unit) =>
      {
        val k2: Rep[Array[Array[Float]] => Unit] = fun { (x: Rep[Array[Array[Float]]]) =>
          val tensors = ArrayBuffer[TensorR]()
          for (u <- (0 until length): Range) {
            tensors.append(new TensorR(Tensor(x(u*2), dim0s(u)), Tensor(x(u*2+1), dim0s(u))))
          }
          k1(tensors)
        }
        val k4: (ArrayBuffer[TensorR] => Unit) = {(x: ArrayBuffer[TensorR]) =>
          val arrays = NewArray[Array[Float]](2*length)
          for (u <- (0 until length): Range) {
            arrays(u*2) = x(u).x.data; arrays(u*2+1) = x(u).d.data
          }
          f1((i, k2, arrays))
        }
        k4
      }
    }
  }

  @virtualize
  def LOOPLM(init: ArrayBuffer[TensorR])(c: Rep[Int])(b: Rep[Int] => ArrayBuffer[TensorR] => ArrayBuffer[TensorR] @diff):
  ArrayBuffer[TensorR] @diff = shift { k: (ArrayBuffer[TensorR] => Unit) =>
    lazy val loop: Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit = FUNlm(init map (_.x.dims(0))) {
      (i: Rep[Int]) => (k: ArrayBuffer[TensorR] => Unit) => (x: ArrayBuffer[TensorR]) =>
        if (i < c) { loop(i+1)((x: ArrayBuffer[TensorR]) => RST(k(b(i)(x))))(x) } else { RST(k(x)) }
    }
    loop(0)(k)(init)
  }

  @virtualize
  def LOOPTM(init: ArrayBuffer[TensorR])(lch: Rep[Array[Int]], rch: Rep[Array[Int]])
  (b: (ArrayBuffer[TensorR], ArrayBuffer[TensorR], Rep[Int]) => ArrayBuffer[TensorR] @diff): ArrayBuffer[TensorR] @diff = shift {
    k: (ArrayBuffer[TensorR] => Unit) =>

      lazy val tree: Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit = FUNlm(init.map(_.x.dims(0))) {
        (i: Rep[Int]) => (k: ArrayBuffer[TensorR] => Unit) => (x: ArrayBuffer[TensorR]) =>
          if (i >= 0) { tree(lch(i))((l: ArrayBuffer[TensorR]) => tree(rch(i))((r: ArrayBuffer[TensorR]) => RST(k(b(l, r, i))))(x))(x) }
          else { RST(k(x)) }
      }
      tree(0)(k)(init)
  }

  def gradR(f: TensorR => TensorR @diff)(x: Tensor): Tensor = {
    val x1 = new TensorR(x, Tensor.zeros(x.dims(0)))
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

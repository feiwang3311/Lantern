import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

import scala.collection.mutable.ArrayBuffer
import scala.collection.{Seq => NSeq}

object LMS_vector {

  trait TensorExp extends Dsl {

    /**
      Note: Need to see how to manage memory because everytime the NewArray is used,
        C code will use malloc without thinking about where to call free.
        The memory will leak unless we explicitly use unchecked("free(",x,")")

        This is a deep problem because statically determine the free sites is unsolvable
        but stronger type systems or regulations is possible, like Rust
        Either we manually maintain the memory, or build some sort of system to handle it in a stronger way.
        Leo's escape paper maybe of interest too.

        However Greg just gave me a very good idea. Basically by using delimited continuations, our intermediate
        values are only used within the lexical scope of that operation, and the continuation it calls.
        Just like a withFile("filename") {} or withArray("array") {} construct,
        where the file and array are only used in the scope of with, can can be implicitly closed or deleted at escape
        Our intermediate Tensors are created by an overloaded operation, used in delimited continuations and updating
        the gradients of @this@ and @that@, and then never used outside
        So we can add unchecked("free") to reclaim their memory right at the end of each overloaded operator def.

      Note:

        There is a bug using the above method. Using free at the end of scope with LOOP together had an issue of
        use-after-free. Not sure why.

        Tiark suggested to use smart pointer in c++, which is a good idea because it takes away the burden of manually managing them.
        All we should have changed is the c code generation for NewArray[Double], so that the malloc is in the hand of smart pointers

        However, using smart pointers has problems too. we cannot define a function that takes a smartpointer as argument and return a smartpointer
        the returned smartpointer out-lives the data, which is not OK for smart pointer.

      Note:

        finally we used a temperate solution called "memory arena". The base code will claim a large piece of code for the whole program.
        internally, every malloc will borrow memory from this arena.

        By using getAllocMem and setAllocMem, we can selectively return a big trunk of memory after one iteration of training.

      Note:

        We are currently only very narrowly supporting matrix (2d vectors)
        We only support Matrix vector multiplication, which is like several vector_vector dot product
        Matrix has >1 dims(1) field and number of values dims(0) * dims(1)
        but the current implementation silently ignore the 2:end columns unless it is dot product
        The idea of thinking Matrix row as dims(0) and colume as dims(1) is not the common way, but we are going by it for now because
        we want to simplify the implementation and just borrow the logic of dot

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
        unchecked[Long](s"((diff_$index.tv_sec * 1000L) + (diff_$index.tv_usec/1000L))")
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
    def convSize(size: Int, kernelSize: Int, strideSize: Int) = (size - kernelSize)/strideSize + 1
    def mmax(a: Int, b: Int) = if (a >= b) a else b

    @virtualize
    def assertC(cond: Rep[Boolean], msg: String, args: Rep[Any]*): Unit = {
      if(!cond) { printf(msg, args : _*); exit() }
    }

    def slice(arr: Rep[Array[Double]], off: Rep[Int]) = uncheckedPure[Array[Double]](arr, "+", off)

    /**
     Add: Scanner class for Input
     Copied from lms-query-tutorial
     **/

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
      def rand() = unchecked[Double]("(double)rand()/RAND_MAX")
      def srand(seed: Option[Int] = None) = unchecked[Unit]("srand(",seed.map(_.toString).getOrElse("time(NULL)"),")")
    }

    def exit() = unchecked[Unit]("exit(0)")

    class Tensor(val data: Rep[Array[Double]], val dimsSeq: NSeq[Int]) extends Serializable {

      val MAX_DOUBLE = 1e10 // FIXME

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
      def clipAt(bound: Double) = {
        for (i <- (0 until nbElem): Rep[Range]) {
          if (data(i) > bound) data(i) = bound
          if (data(i) < -1.0 * bound) data(i) = -1.0 * bound
        }
      }

      def mapInPlace(op: Rep[Double] => Rep[Double]) = {
        for (i <- (0 until nbElem): Rep[Range]) this.data(i) = op(this.data(i))
      }

      def map(op: Rep[Double] => Rep[Double]) = {
        val res = NewArray[Double](nbElem)
        for (i <- (0 until nbElem): Rep[Range]) res(i) = op(this.data(i))
        new Tensor(res, dims)
      }

      def fold(init: Rep[Double])(op: (Rep[Double], Rep[Double]) => Rep[Double]) = {
        //generate_comment("Here")
        val res = var_new[Double](init)
        for (i <- (0 until nbElem): Rep[Range]) var_assign(res, op(res, this.data(i)))
        res
      }

      def +(that: Rep[Double]): Tensor = this.map(x => x + that)
      def +(that: Tensor): Tensor = {
        if (nbElem == 1) that + this.data(0)
        else if (that.nbElem == 1) this + that.data(0)
        else if (that.dims == this.dims) {
          val res = NewArray[Double](nbElem)
          for (i <- (0 until nbElem): Rep[Range]) res(i) = this.data(i) + that.data(i)
          new Tensor(res, dims)
        }
        else throw new IllegalArgumentException(s"dimensions of vector do not match +! ${this.dims.seq} != ${that.dims.seq}")
      }

      // this operator updates the values of this, unlike the + operator
      def +=(that: Rep[Double]): Unit = this.mapInPlace(x => x + that)
      def += (that: Tensor): Unit = {
        if (that.nbElem == 1) {
          generate_comment("+= tensor of dim 0")
          this += that.data(0) // broadcast
        }
        else if (this.nbElem == 1) ??? // this.data(0) = that.fold(this.data(0))((agg, x) => agg + x)
        else if (this.dims == that.dims)
          for (i <- (0 until nbElem): Rep[Range]) this.data(i) += that.data(i)
        else throw new IllegalArgumentException(s"dimensions of vector do not match +=! ${this.dims.seq} != ${that.dims.seq}")
      }

      def -(that: Rep[Double]): Tensor = this.map(x => x - that)
      def -(that: Tensor): Tensor = {
        if (nbElem == 1) that.map(x => this.data(0) - x)
        else if (that.nbElem == 1) this - that.data(0)
        else if (that.dims == this.dims) {
          val res = NewArray[Double](nbElem)
          for (i <- (0 until nbElem): Rep[Range]) res(i) = this.data(i) - that.data(i)
          new Tensor(res, dims)
        }
        else throw new IllegalArgumentException("dimensions of vector do not match +!")
      }

      // this operator updates the values of this, unlike the - operator
      def -=(that: Rep[Double]): Unit = this.mapInPlace(x => x - that)
      def -= (that: Tensor): Unit = {
        if (that.nbElem == 1) this -= that.data(0) // broadcast
        else if (this.nbElem == 1) {
          ???
          // this.data(0) = that.fold(this.data(0))((agg, x) => agg - x)
        }
        else if (this.dims == that.dims)
          for (i <- (0 until nbElem): Rep[Range]) this.data(i) -= that.data(i)
        else throw new IllegalArgumentException("dimensions of vector do not match +=!")
      }

      // Element wise multiplication
      def *(that: Rep[Double]): Tensor = this.map(x => x * that)
      def *(that: Tensor): Tensor = {
        if (nbElem == 1) that * this.data(0)
        else if (that.nbElem == 1) this * that.data(0)
        else if (that.dims == this.dims) {
          val res = NewArray[Double](nbElem)
          for (i <- (0 until nbElem): Rep[Range]) res(i) = this.data(i) * that.data(i)
          new Tensor(res, dims)
        }
        else throw new IllegalArgumentException(s"dimensions of vector do not match * ${this.dims.seq} != ${that.dims.seq}")
      }

      // this operator updates the values of this, unlike the * operator
      def *=(that: Rep[Double]): Unit = this.mapInPlace(x => x * that)
      def *= (that: Tensor): Unit = {
        if (that.nbElem == 1) this *= that.data(0) // broadcast
        else if (this.nbElem == 1) {
          ???
          // this.data(0) = that.fold(this.data(0))((agg, x) => agg * x)
        }
        else if (this.dims == that.dims)
          for (i <- (0 until nbElem): Rep[Range]) this.data(i) *= that.data(i)
        else throw new IllegalArgumentException("dimensions of vector do not match +=!")
      }

      // element wise division
      def /(that: Rep[Double]): Tensor = this.map(x => x / that)
      def /(that: Tensor): Tensor = {
        if (nbElem == 1) that.map(x => this.data(0) / x)
        else if (that.nbElem == 1) this / that.data(0)
        else if (that.dims == this.dims) {
          val res = NewArray[Double](nbElem)
          for (i <- (0 until nbElem): Rep[Range]) res(i) = this.data(i) / that.data(i)
          new Tensor(res, dims)
        }
        else throw new IllegalArgumentException("dimensions of vector do not match +!")
      }

      // this operator updates the values of this, unlike the / operator
      def /=(that: Rep[Double]): Unit = this.mapInPlace(x => x / that)
      def /= (that: Tensor): Unit = {
        if (that.nbElem == 1) this /= that.data(0) // broadcast
        else if (this.nbElem == 1) ??? // this.data(0) = that.fold(this.data(0))((agg, x) => agg / x)
        else if (this.dims == that.dims)
          for (i <- (0 until nbElem): Rep[Range]) this.data(i) /= that.data(i)
        else throw new IllegalArgumentException("dimensions of vector do not match +=!")
      }

      def setAsOne() = { this.mapInPlace(x => 1.0); () }
      def clear() = { this.mapInPlace(x => 0.0); () }

      def copy_data(that: Tensor) = {
        assert(this.nbElem == that.nbElem, "dimensions of vector do not match copy_data!")
        for (i <- (0 until nbElem): Rep[Range]) this.data(i) = that.data(i)
      }

      // NOTE: only handles (Matrix dot Vector) and (Vector dot Vector)
      def dot(that: Tensor) = {
        // assert that and this have the same dimension
        generate_comment(s"dot ${this.dims.seq} - ${that.dims.seq}")
        assert(this.nbDims <= 2 && that.nbDims == 1, s"Only M x V or V x V allowed ${this.dims} - ${that.dims}")
        assert(this.dims.last == that.dims(0), s"dimensions of vector do not match dot! ${this.dims.seq} - ${that.dims.seq}")
        // TODO: remove loop if not needed
        val off = var_new(0)
        val up = if (this.nbDims > 1) this.dims(0) else 1
        val res = NewArray[Double](up)
        for (j <- (0 until up): Rep[Range]) {
          val value = var_new(0.0)
          for (i <- (0 until this.dims.last): Rep[Range]) {
            value += data(off) * that.data(i)
            off += 1
          }
          res(j) = readVar(value)
        }
        val dim = if (this.nbDims == 1) 1 else this.dims(0)
        Tensor(res, dim)
      }

      // NOTE: only handles (Vector cart Vector)
      def cart(that: Tensor) = {
        assert(this.nbDims == 1 && that.nbDims == 1, "cartesian product is only for 1d vectors")
        val res = NewArray[Double](this.dims(0) * that.dims(0))
        val off = var_new(0)
        for (i <- (0 until this.dims(0)): Rep[Range]) {
          for (j <- (0 until that.dims(0)): Rep[Range]) {
            res(off) = data(i) * that.data(j)
            off += 1
          }
        }
        Tensor(res, this.dims(0), that.dims(0))
      }

      def trans() = {
        assert(this.nbDims == 2, "transpose is only for matrix. Tensor transpose is not supported here")
        val res = NewArray[Double](this.nbElem)
        val offT = var_new(0)
        for (i <- (0 until this.dims(1)): Rep[Range]) {
          val off = var_new(0)
          for (j <- (0 until this.dims(0)): Rep[Range]) {
            res(offT + j) = data(off + i)
            off += this.dims(1)
          }
          offT += this.dims(0)
        }
        new Tensor(res, this.dims.reverse)
      }

      def tanh() = this.map(x => Math.tanh(x))
      def exp() = this.map(x => Math.exp(x))
      def log() = this.map(x => Math.log(x))
      def sqrt() = this.map(x => Math.sqrt(x))
      def sigmoid() = this.map(x => 1.0 / (Math.exp(-1.0 * x) + 1.0))

      // NOTE: sum all elements
      def sum() = Tensor.scalar(this.fold(0.0)(_ + _))

      @virtualize
      def check(limit: Double) = {
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
        val logsum = m + Math.log(this.fold(0.0)((agg, x) => agg + Math.exp(x - m)))
        this.map(x => x - logsum)
      }

      @virtualize
      def nllLoss(target: Rep[Int]) = {
        assert(this.nbDims == 1)

        // assertC(0 <= target && target < this.nbElem, "Incorrect target")
        Tensor.scalar(-1.0 * this.data(target))
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
          val res = NewArray[Double](this.dims(1))
          val off = var_new(0)
          for (j <- (0 until this.dims(1)): Rep[Range]) {
            res(off) = this.data(off)
            off += 1
          }
          for (i <- (1 until this.dims(0)): Rep[Range]) {
            val offR = var_new(0)
            for (j <- (0 until this.dims(1)): Rep[Range]) {
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
        for (i <- (0 until up): Rep[Range]) {
          for (j <- (0 until dims(1)): Rep[Range]) {
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
        for (i <- (0 until up): Rep[Range]) {
          for (j <- (0 until that.dims(1)): Rep[Range]) {
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
        for (i <- 0 until this.dims(0): Rep[Range]) {
          val offYR = var_new(offYC)
          for (j <- 0 until this.dims(1): Rep[Range]) {
            val offY = var_new(offYR)
            val offThat = var_new(offThatR)
            for (k <- 0 until that.dims(1): Rep[Range]) {
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
      def square(t: Rep[Double]) = t * t
      def add_mult(a: Tensor, b: Tensor) = {
        assert(Tensor.dimCompatible(a, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, b), "dim not Compatible in add_mult")

        // FIXME!!!
        val dims0M = mmax(dims(0), mmax(a.dims(0), b.dims(0)))
        val dims1M = mmax(if (this.nbDims > 1) dims(1) else 1, mmax(if (a.nbDims > 1) a.dims(1) else 1, if (b.nbDims > 1) b.dims(1) else 1))
        if (this.isScalar) {
          for (i <- 0 until (dims0M * dims1M): Rep[Range]) data(0) = data(0) + a.getAt(i) * b.getAt(i)
        } else {
          for (i <- (0 until dims0M * dims1M): Rep[Range]) data(i) = data(i) + a.getAt(i) * b.getAt(i)
        }
      }

      def addMul(a: Rep[Double], b: Tensor) = {
        assert(this.dims == b.dims)

        generate_comment("Generate code for addMul")
        for (i <- 0 until this.nbElem: Rep[Range]) {
          this.data(i) = this.data(i) + a * b.data(i)
        }
      }
      def cmulAdd(a: Double, b: Tensor) = {
        assert(this.dims == b.dims)

        for (i <- 0 until this.nbElem: Rep[Range])
          this.data(i) = a * this.data(i) + b.data(i)

        this // FIXME ??
      }

      def add_div(a: Tensor, b: Tensor) = {
        assert(Tensor.dimCompatible(a, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, b), "dim not Compatible in add_div")
        val dims0M = mmax(dims(0), mmax(a.dims(0), b.dims(0)))
        // FIXME
        val dims1M = mmax(if (nbDims > 1) dims(1) else 1, mmax(if (a.nbDims > 1) a.dims(1) else 1, if (b.nbDims > 1) b.dims(1) else 1))
        if (this.isScalar) {
          for (i <- (0 until dims0M * dims1M): Rep[Range]) data(0) = data(0) + a.getAt(i) / b.getAt(i)
        } else {
          for (i <- (0 until dims0M * dims1M): Rep[Range]) data(i) = data(i) + a.getAt(i) / b.getAt(i)
        }
      }

      def minus_mult_div_square(a: Tensor, b: Tensor, c: Tensor) = {
        assert(Tensor.dimCompatible(a, b)    && Tensor.dimCompatible(a, c)    && Tensor.dimCompatible(c, b)    &&
          Tensor.dimCompatible(this, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, c),
          "dim not competible in minus_mult_div_square")
        val dims0M = mmax(dims(0), mmax(a.dims(0), mmax(b.dims(0), c.dims(0))))
        // FIXME
        val dims1M = mmax(if (nbDims > 1) dims(1) else 1, mmax(if (a.nbDims > 1) a.dims(1) else 1, if (b.nbDims > 1) b.dims(1) else 1))
        if (this.isScalar) {
          for (i <- (0 until dims0M * dims1M): Rep[Range]) data(0) = data(0) - a.getAt(i) * b.getAt(i) / square(c.getAt(i))
        } else {
          for (i <- (0 until dims0M * dims1M): Rep[Range]) data(i) = data(i) - a.getAt(i) * b.getAt(i) / square(c.getAt(i))
        }
      }

      def add_oneMinusSquare_mult(a: Tensor, b: Tensor) = {
        assert(Tensor.dimCompatible(a, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, b), "dim not Compatible in add_oneMinusSquare_mult")
        val dims0M = mmax(dims(0), mmax(a.dims(0), b.dims(0)))
        // FIXME
        val dims1M = mmax(if (nbDims > 1) dims(1) else 1, mmax(if (a.nbDims > 1) a.dims(1) else 1, if (b.nbDims > 1) b.dims(1) else 1))
        if (this.isScalar) {
          for (i <- (0 until dims0M * dims1M): Rep[Range]) data(0) = data(0) + (1.0 - square(a.getAt(i))) * b.getAt(i)
        } else {
          for (i <- (0 until dims0M * dims1M): Rep[Range]) data(i) = data(i) + (1.0 - square(a.getAt(i))) * b.getAt(i)
        }
      }
      def oneMinusThenMult(t: Rep[Double]) = (1.0 - t) * t
      def add_oneMinusThenMult_mult(a: Tensor, b: Tensor) = {
        assert(Tensor.dimCompatible(a, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, b), "dim not Compatible in add_oneMinusThenMult_mult")
        val dims0M = mmax(dims(0), mmax(a.dims(0), b.dims(0)))
        // FIXME
        val dims1M = mmax(if (nbDims > 1) dims(1) else 1, mmax(if (a.nbDims > 1) a.dims(1) else 1, if (b.nbDims > 1) b.dims(1) else 1))
        if (this.isScalar) {
          for (i <- (0 until dims0M * dims1M): Rep[Range]) data(0) = data(0) + oneMinusThenMult(a.getAt(i)) * b.getAt(i)
        } else {
          for (i <- (0 until dims0M * dims1M): Rep[Range]) data(i) = data(i) + oneMinusThenMult(a.getAt(i)) * b.getAt(i)
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
        for (outPane <- 0 until kernel.dims(0): Rep[Range]) {
          // assertC(offOut == outPane * res.strides(1), "Invalid Output Idx %d != %d (%d)", offOut, outPane * res.strides(1), outPane)
          // assertC(offWeight1 == outPane * kernel.strides(1), "Invalid Kernel Idx")
          val offWeight2 = var_new(offWeight1)
          val offInput = var_new(0)
          val ptrOutput = slice(res.data, offOut)
          for (inPane <- 0 until this.dims(0): Rep[Range]) {
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
        for (outRow <- 0 until this.dims(0): Rep[Range]) {
          // assertC(offInputR == outRow * input.strides(1), "intputR invalid")
          val offInputC = var_new(offInputR)
          for (outCol <- 0 until this.dims(1): Rep[Range]) {
            // assertC(offInputC == outRow * strideRow * input.strides(1) + outCol * strideCol, "intputC invalid")
            val offKernel = var_new(0)
            val offInput = var_new(offInputC)
            val sum = var_new(0.0)
            for (kernelRow <- 0 until kernel.dims(0): Rep[Range]) {
              // assertC(offInput == (outRow * strideRow + kernelRow) * input.strides(1) + outCol * strideCol, "input invalid")
              // assertC(offKernel == kernelRow * kernel.strides(1), "kernel invalid")
              val ptrIntput = slice(input.data, offInput)
              val ptrKernel = slice(kernel.data, offKernel)
              for (kernelCol <- 0 until kernel.dims(1): Rep[Range]) {
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
      def dropout(prob: Double = 0.5) = {
        assert(0.0 <= prob && prob <= 1.0)

        val res = NewArray[Double](this.nbElem)
        val mask = NewArray[Double](this.nbElem)

        val scale = if (prob < 1.0) 1.0 / (1.0 - prob) else 0.0

        val guard: Rep[Boolean] = prob < 1.0
        for (i <- 0 until this.nbElem: Rep[Range]) {
          if (guard && Random.rand() > prob) {
            res(i) = this.data(i) * scale
            mask(i) = scale
          } else {
            res(i) = 0.0
            mask(i) = 0.0
          }
        }

        (Tensor(res, this.dims.seq : _*), Tensor(mask, this.dims.seq : _*))
      }

      @virtualize
      def relu(inPlace: Boolean = false) = {
        assert(!inPlace)

        val res = NewArray[Double](this.nbElem)
        for (i <- 0 until this.nbElem: Rep[Range]) {
          if (this.data(i) < 0.0)
            res(i) = 0.0
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
      def normalize(m: Double, s: Double, inPlace: Boolean = false) = {
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
        new Tensor(NewArray[Double](size), dims)
      }
      def apply(data: Rep[Array[Double]], dims: Int*) = new Tensor(data, dims)

      def dimCompatible(a: Tensor, b: Tensor) = {
        (a.dims == b.dims) || a.isScalar || b.isScalar
      }

      def rand(dims: Int*) = randinit(dims.toSeq, 1.0, None)
      def rand(scale: Double, dims: Int*) = randinit(dims.toSeq, scale, None)
      def randinit(dim0: Int): Tensor = randinit(NSeq(dim0), 1.0, None)
      def randinit(dim0: Int, seed: Option[Int]): Tensor = randinit(NSeq(dim0), 1.0, seed)
      def randinit(dim0: Int, dim1: Int, scale: Double): Tensor = randinit(NSeq(dim0, dim1), scale, None)
      def randinit(dims: NSeq[Int], scale: Double = 1.0, seed: Option[Int] = None): Tensor = {
        val size = dims.product
        val res = NewArray[Double](size)
        for (i <- (0 until size): Rep[Range]) res(i) = (Random.rand() - 0.5) * scale
        new Tensor(res, dims)
      }

      def randn(dim0: Int, dim1: Int = 1, scale: Double = 1.0, offset: Int = 0) = {
        val res = NewArray[Double](dim0 * dim1)
        for (i <- (0 until dim0 * dim1): Rep[Range]) res(i) = unchecked[Double]("d(gen)") * scale
        Tensor(res, dim0, dim1)
      }

      def randPositive(dims: Int*) = {
        val size = dims.product
        val res = NewArray[Double](size)
        for (i <- (0 until size): Rep[Range]) res(i) = Random.rand()
        new Tensor(res, dims)
      }

      def fill(value: Rep[Double], dims: Int*) = {
        val size = dims.product
        val res = NewArray[Double](size)
        for (i <- (0 until size): Rep[Range]) res(i) = value
        new Tensor(res, dims)
      }

      def fill(fFill: NSeq[Rep[Int]] => Rep[Double], dims: Int*) = {
        val size = dims.product
        val res = NewArray[Double](size)

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
        fill(0.0, dims: _*)
      }

      def zeros(that: Tensor): Tensor = {
        zeros(that.dims : _*)
      }

      def zeros_like(that: Tensor) = {
        zeros(that.dims : _*)
      }

      def scalar(value: Rep[Double]) = {
        val res = NewArray[Double](1)
        res(0) = value
        Tensor(res, 1)
      }

      def ones(dims: Int*) = fill(1.0, dims: _*)
      def ones(that: Tensor) = fill(1.0, that.dims: _*)
      def halves(dims: Int*) = fill(0.5, dims: _*)

      def expand(vector: Tensor, dim1: Int) = {
        assert(vector.nbDims == 1)
        val res = NewArray[Double](vector.dims(0) * dim1)
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
        val res = NewArray[Double](vector.nbElem)
        for (i <- (0 until vector.nbElem): Rep[Range]) res(i) = vector.data(i)
        new Tensor(res, vector.dims)
      }

      def fromData(x: Double*) = {
        val y = x.toArray
        val res = NewArray[Double](y.length)
        for (i <- 0 until y.length: Range) res(i) = y(i)
        Tensor(res, y.length)
      }


      // def conv(that: Tensor, stride: (Int, Int) = (1, 1))

      @virtualize
      def assertEqual(a: Tensor, b: Tensor, mark: String = "", tal: Double = 0.000001) = {
        assert(a.dims == b.dims, s"ERROR: $mark not equal in dimensionsi ${a.dims.seq} != ${b.dims.seq}\\n")

        val i = var_new(0)
        while (i < a.nbElem && { val diff = a.data(i) - b.data(i); diff > -tal && diff < tal }) {
          i += 1
        }
        if (i < a.nbElem)
          printf("ERROR: %s not equal in some data - %.4f != %.4f (%d)\\n", mark, a.data(i), b.data(i), i)
      }
    }


    // Tensor type is the similar to NumR, just replace RDouble with Tensor
    // also Tensor internally use array, which is mutable by default
    // so both field are val (not var) and can be updated by += -= *= /= setAsOne()
    // all instances of vectors will be shepherded by c++ smart pointers, alleviating the memory leak problem
    type diff = cps[Unit]

    class TensorR(val x: Tensor, val d: Tensor) extends Serializable {
      var isInput: Boolean = false // true if it is an input (no need to compute gradient)

      def clip_grad(bound: Double) = {
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
          this.d.data(i) = y.d.data(i) - Math.exp(y.x.data(i)) * s
        }
      }

      def resize(dims: Int*): TensorR @diff = shift { (k: TensorR => Unit) =>
        k(new TensorR(this.x.resize(dims : _*), this.d.resize(dims : _*)))
      }

      def nllLoss(target: Rep[Int]): TensorR @diff = shift { (k: TensorR => Unit) =>
        val y = TensorR(x.nllLoss(target)); k(y)

        assert(y.x.isScalar)
        //y.d.print("nll")

        this.d.data(target) = -1.0 * y.d.data(0)
      }

      def update(lr: Double, mom: Double) = {
      }

      @virtualize
      def conv(kernel: TensorR, strideRow: Int, strideCol: Int): TensorR @diff = shift { (k: TensorR => Unit) =>

        val y = TensorR(x conv2D(kernel.x, strideRow, strideCol))
        k(y)
        //y.d.print("conv")

        // TODO think about the loop order
        val offOutputD = var_new(0)
        val offKernel = var_new(0)
        assert(y.d.dims(0) == kernel.x.dims(0))
        for (kOut <- 0 until y.d.dims(0): Rep[Range]) { // forall output pane
          val offInputR = var_new(0)
          for (row <- 0 until y.d.dims(1): Rep[Range]) {
            // assertC(offInputR == row * strideRow * this.x.strides(2), s"ERROR: input offsetR %d != %d (%d, %d)\\n", offInputR, row * strideRow * this.x.strides(2), kOut, row)
            val offInputC = var_new(offInputR)
            for (col <- 0 until y.d.dims(2): Rep[Range]) {
              val dCurr: Rep[Double] = y.d.data(offOutputD)

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
      def dropout(prob: Double): TensorR @diff = shift { (k: TensorR => Unit) =>
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
          this.d.data(i) = if (this.x.data(i) < 0.0) 0.0 else y.d.data(i)
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
        val d = if (isInput) Tensor.scalar(0.0) else Tensor.zeros_like(a)
        val res = new TensorR(a, d)
        res.isInput = isInput
        res
      }
      def apply(a: Rep[Array[Double]], dim0: Int, dim1: Int): TensorR = {
        new TensorR(Tensor(a, dim0, dim1), Tensor.zeros(dim0, dim1))
      }

      def apply(dim0: Int, dim1: Int): TensorR = {
        new TensorR(Tensor.zeros(dim0, dim1), Tensor.zeros(dim0, dim1))
      }
    }

    // change fun signature for memory leak issue (no more returning of array, just update the array provided by the caller)
    // this is in accordance of the destination-programming style
    // the fun take array[array[double]] of size 2, with the first array to be the x, and the second array to be the d
    def FUNc(dim0: Int)(f: TensorR => Unit): (TensorR => Unit) = {
      val f1 = fun { (x: Rep[Array[Array[Double]]]) =>
        val tensor = new TensorR(Tensor(x(0), dim0), Tensor(x(1), dim0))
        f(tensor)
      };
      {
        (x:TensorR) => {
          val in = NewArray[Array[Double]](2)
          in(0) = x.x.data; in(1) = x.d.data
          f1(in) // f1 should take Array[Array[Double]] and update the gradient of x
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
      val f1 = fun { (xx: Rep[(Int, Array[Array[Double]])]) =>
        val i: Rep[Int]                  = tuple2_get1(xx)
        val x: Rep[Array[Array[Double]]] = tuple2_get2(xx)
        val tensor = new TensorR(Tensor(x(0), dim0), Tensor(x(1), dim0))
        f(i)(tensor)
      };
      {
        (i: Rep[Int]) => (x:TensorR) => {
          val in = NewArray[Array[Double]](2)
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
      val f1 = fun { (xx: Rep[(Int, Array[Array[Double]])]) =>
        val i: Rep[Int]                  = tuple2_get1(xx)
        val x: Rep[Array[Array[Double]]] = tuple2_get2(xx)
        val tensors = ArrayBuffer[TensorR]()
        for (u <- (0 until dim0s.length): Range) {
          tensors.append(new TensorR(Tensor(x(u*2), dim0s(u) : _*), Tensor(x(u*2+1), dim0s(u) : _*)))
        }
        f(i)(tensors)
      };
      (i: Rep[Int]) => (x:ArrayBuffer[TensorR]) => {
        val in = NewArray[Array[Double]](2 * dim0s.length)
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

      val f1 = fun { (yy: Rep[(Int, (Array[Array[Double]] => Unit), Array[Array[Double]])]) =>
        val i:  Rep[Int] = tuple3_get1(yy)
        val t1: Rep[Array[Array[Double]] => Unit] = tuple3_get2(yy)
        val xx: Rep[Array[Array[Double]]] = tuple3_get3(yy)
        val t2: (TensorR => Unit) = { (x:TensorR) =>
          val temp = NewArray[Array[Double]](2)
          temp(0) = x.x.data; temp(1) = x.d.data
          t1(temp)
        }
        val t3: (TensorR => Unit) = f(i)(t2)
        t3(new TensorR(Tensor(xx(0), dim0), Tensor(xx(1), dim0)))
      }

      {i: Rep[Int] => k1: (TensorR => Unit) =>
        {
          val k2: Rep[Array[Array[Double]] => Unit] = fun { (x: Rep[Array[Array[Double]]]) =>
            k1(new TensorR(Tensor(x(0), dim0), Tensor(x(1), dim0)))
          }
          val k4: (TensorR => Unit) = {(x: TensorR) =>
            val temp = NewArray[Array[Double]](2)
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
      val f1 = fun { (yy: Rep[(Int, (Array[Array[Double]] => Unit), Array[Array[Double]])]) =>
        val i: Rep[Int] = tuple3_get1(yy)
        val t1: Rep[Array[Array[Double]] => Unit] = tuple3_get2(yy)
        val xx: Rep[Array[Array[Double]]] = tuple3_get3(yy)

        val t2: (ArrayBuffer[TensorR] => Unit) = { (x: ArrayBuffer[TensorR]) =>
          val aa = NewArray[Array[Double]](2*length)
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
          val k2: Rep[Array[Array[Double]] => Unit] = fun { (x: Rep[Array[Array[Double]]]) =>
            val tensors = ArrayBuffer[TensorR]()
            for (u <- (0 until length): Range) {
              tensors.append(new TensorR(Tensor(x(u*2), dim0s(u)), Tensor(x(u*2+1), dim0s(u))))
            }
            k1(tensors)
          }
          val k4: (ArrayBuffer[TensorR] => Unit) = {(x: ArrayBuffer[TensorR]) =>
            val arrays = NewArray[Array[Double]](2*length)
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


  def main(args: Array[String]): Unit = {
    import java.io.PrintWriter;
    import java.io.File;

    if (false) {
      val array0 = new DslDriverC[String, Unit] with TensorExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
          val addr = getMallocAddr()
          //printf("address is at %ld \\n", addr)
          resetMallocAddr(addr)
          //printf("now lets use some memory\\n")
          val mem = Tensor.zeros(100)
          val addr1 = getMallocAddr()
          //printf("Now address is at %ld \\n", addr1)
          resetMallocAddr(addr)
          val addr2 = getMallocAddr()
          //printf("after reset, the address is back to %ld\\n", addr2)

          //assertions
          if (addr + 800 != addr1) printf("ERROR: addr did not increase by 800")
          if (addr != addr2) printf("ERROR: addr did not reset to the give value")
          // unchecked[Unit](s"assert($addr1 == $addr + 800)")
        //assert (addr1 == addr + 800l, "addr did not increase by 800")
      //assert (addr == addr2, "addr did not reset to the given value")
        }
      }

      val array0_file = new PrintWriter(new File("array0.cpp"))
      array0_file.println(array0.code)
      array0_file.flush()
      println("run test case array0")
      array0.eval("abc")

      val array1 = new DslDriverC[String, Unit]  with TensorExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
          val length = 2
          val res = Tensor.randinit(length)
          val res2 = Tensor.randinit(length, seed = Some(5))
          //res.print()
          //res2.print()

          val result = res dot res2
          //result.print()

          // assertions
          if (res.data(0) * res2.data(0) + res.data(1) * res2.data(1) != result.data(0))
            println("ERROR: the dot product of two vectors is not correct")

        }
      }

      //println("test dot")
      //val array1_file = new PrintWriter(new File("array1(2).cpp"))
      //array1_file.println(array1.code)
      //array1_file.flush()
      //println(array1.code)
      println("run test case array1")
      array1.eval("abc")

      val array1_1 = new DslDriverC[String, Unit] with TensorExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
          val dim0 = 2
          val dim1 = 3
          val matrix = Tensor.rand(dim0, dim1)
          val vector = Tensor.randinit(dim1, seed = Some(4))
          //matrix.print()
          //vector.print()

          //println("the result is:")
          val result = matrix dot vector
          //result.print()

          if (matrix(0, 0) * vector(0) + matrix(0, 1) * vector(1) + matrix(0, 2) * vector(2) != result(0))
            printf("ERROR: the matrix vector dot product is wrong on the first element of result, %.3f != %.3f\\n", matrix(0, 0) * vector(0) + matrix(0, 1) * vector(1) + matrix(0, 2) * vector(2), result(0))
          if (matrix(1, 0) * vector(0) + matrix(1, 1) * vector(1) + matrix(1, 2) * vector(2) != result(1))
            printf("ERROR: the matrix vector dot product is wrong on the second element of result, %.3f != %.3f\\n", matrix(1, 0) * vector(0) + matrix(1, 1) * vector(1) + matrix(1, 2) * vector(2), result(1))
        }
      }

      //println(array1_1.code)
      println("run test case array1_1")
      array1_1.eval("abc")

      val array2 = new DslDriverC[String, Unit] with TensorExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
          // read training data from file (for now just use random)
          val length = 2
          val v = Tensor.randinit(length)
          //v.print()

          // calculate gradient
          val grad = gradR(t => t dot t)(v)
          // show gradient
          //println("show gradient in the traditional way")
          //grad.print()

          // assertions
          Tensor.assertEqual(v * 2.0, grad)

          // construct TensorR for closure
          val tv = TensorR(v)
          val loss = gradR_loss(dummy => tv dot tv)(Tensor.zeros(1))
          //println("gradient:")
          //tv.d.print()
          //println("loss")
          //loss.print()
          // assertions
          Tensor.assertEqual((v dot v), loss)
          Tensor.assertEqual(tv.d, grad)
          ()
        }
      }

      //println("test dot gradient")
      //println(array2.code)
      println("run test case array2")
      array2.eval("2.0")

      val array2_1 = new DslDriverC[String, Unit] with TensorExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {

          val dim0 = 2
          val vector = Tensor.randinit(dim0, seed = Some(4))

          // initialize tensors for closure
          val ve = new TensorR(vector, Tensor.zeros(dim0))
          val half = new TensorR(Tensor.halves(dim0), Tensor.zeros(dim0))

          // define function of model
          def model(dummy: TensorR): TensorR @diff = {
            ((ve dot ve) * half).sum()
          }
          val loss = gradR_loss(model)(Tensor.zeros(1))
          Tensor.assertEqual(loss, ((vector dot vector) * Tensor.halves(dim0)).sum(), "1")
          Tensor.assertEqual(ve.d, vector * 2.0 ,"2")
          Tensor.assertEqual(half.d, Tensor.fill((vector dot vector).data(0), 2), "3")
          ()
        }
      }

      println("run test case array2_1")
      array2_1.eval("abc")

      val array2_2 = new DslDriverC[String, Unit] with TensorExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {

          val dim0 = 2
          val dim1 = 3
          val matrix = Tensor.rand(dim0, dim1)
          val vector = Tensor.randinit(dim1, seed = Some(4))

          // initialize tensors for closure
          val ma = TensorR(matrix)
          val ve = TensorR(vector)

          // define function of model
          def model(dummy: TensorR): TensorR @diff = {
            (ma dot ve).sum()
          }
          val loss = gradR_loss(model)(Tensor.zeros(1))
          Tensor.assertEqual(loss, (matrix dot vector).sum(), "11")
          Tensor.assertEqual(ma.d, Tensor.expand(vector, dim0), "12")
          val sol = matrix.sumOnDim1()
          Tensor.assertEqual(ve.d, sol, "13")
          ()
        }
      }

      // println("test matrix vector dot gradient as side effect")
      //println(array2_2.code)
      println("run test case array2_2")
      array2_2.eval("abc")


      val testTrans = new DslDriverC[String, Unit] with TensorExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
          val idx = var_new(0)
          val t = Tensor.fill(seq => { idx += 1; idx }, 2, 3)

          Tensor.assertEqual(t.trans(), Tensor.fromData(1.0, 4.0, 2.0, 5.0, 3.0, 6.0).resize(3, 2), "Transpose invalid")
        }
      }
      println("run test trans")
      testTrans.eval("abs")


      val array2_3 = new DslDriverC[String, Unit] with TensorExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {

          val vocab_size = 3
          val hidden_size = 10
          val Wxh = Tensor.randinit(hidden_size, vocab_size, 0.1)  // input to hidden
          val Whh = Tensor.randinit(hidden_size, hidden_size, 0.1) // hidden to hidden
          val Why = Tensor.randinit(vocab_size, hidden_size, 0.1)  // hidden to output
          val bh  = Tensor.randinit(hidden_size)
          val by  = Tensor.randinit(vocab_size)
          val hprev = Tensor.randinit(hidden_size)

          val hprev_next = Tensor.zeros_like(hprev) // this vector catches the new hidden value, see the NOTE below
          /*
          NOTE: initially I simply updated hprev with new hidden value. That turned out to be a bug.
          Syntactically I updated hprev after the LOOPCCM cycle, but because we are constructing a static computation graph with continuations,
          symantically the update happens before the end of the forward propagation.

          So instead of updating hprev after autodifferentiation, I updated it before autodifferentiation.
          That is a easily fallen pitfall.

          NEED to think about how to avoid it or send WARNING for code like this!!

          The solution is to copy it to an independent vector. MAYBE need better solutions?
          */

         // wrap as tensors
         val Wxh1 = TensorR(Wxh)
         val Whh1 = TensorR(Whh)
         val Why1 = TensorR(Why)
         val bh1  = TensorR(bh)
         val by1  = TensorR(by)
         val hprev1 = TensorR(hprev)

         // encode input and output
         val x_data = NewArray[Int](3); x_data(0) = 0; x_data(1) = 1; x_data(2) = 2
         val y_data = NewArray[Int](3); y_data(0) = 2; y_data(1) = 0; y_data(2) = 1
         //val x_data = mutableStaticData(scala.Array(0, 1, 2))
         //val y_data = mutableStaticData(scala.Array(2, 0, 1))

         // our method of loss and gradient calculation
         def lossFun: (TensorR => TensorR @diff) = { (dummy: TensorR) =>
           val loss = TensorR(Tensor.zeros(1))
           val in = ArrayBuffer[TensorR]()
           in.append(loss)
           in.append(hprev1)
           val outputs = LOOPSM(in)(1) { i => t =>

             // get input as one-hot tensor
             val x = Tensor.zeros(vocab_size)
             x.data(x_data(i)) = 1
             val x1 = TensorR(x)
             // get output as one-hot tensor
             val y = Tensor.zeros(vocab_size)
             y.data(y_data(i)) = 1
             val y1 = TensorR(y)

             val tmp = (Wxh1 dot x1)
             val h1 = (tmp + (Whh1 dot t(1)) + bh1).tanh() // use hidden state and x1 to compute hidden state
             val e1 = (Why1.dot(h1) + by1).exp()                       // use new hidden state to compute unnormalized prob
             val p1 = e1 / e1.sum()                            // use unnormalized prob to compute normalize prob
             generate_comment("Compute new loss")
             val newloss = t(0) - (p1 dot y1).log()            // loss is updated by original loss t(0) and additional loss
             generate_comment("Done computing loss")
             val out = ArrayBuffer[TensorR]()

             out.append(newloss)
             out.append(h1)
             out
           }
           hprev_next.copy_data(outputs(1).x)  // update the hidden state with the result from LOOP
           outputs(0)                          // return the final loss
         }
         val loss1 = gradR_loss(lossFun)(Tensor.zeros(1))
         printf("bh1\\n")
         bh1.d.printRaw(hidden_size)

         generate_comment("Compute real value")


         // correct method of loss and gradient calculation, adapting from Numpy
         // preset space for gradients
         val dWxh = Tensor.zeros_like(Wxh)
         val dWhh = Tensor.zeros_like(Whh)
         val dWhy = Tensor.zeros_like(Why)
         val dbh  = Tensor.zeros_like(bh)
         val dby  = Tensor.zeros_like(by)
         val dhnext = Tensor.zeros_like(hprev)
         val sum_loss = Tensor.zeros(1)
         val hprev_new = Tensor.zeros_like(hprev)

         def lossOneCycle(i: Int, hprev: Tensor): Unit = {

           // get input as one-hot tensor
           val x = Tensor.zeros(vocab_size)
           x.data(x_data(i)) = 1
           // get output as one-hot tensor
           val y = Tensor.zeros(vocab_size)
           y.data(y_data(i)) = 1

           // forward pass
           val tmp = (Wxh dot x)
           val hs = (tmp + (Whh dot hprev) + bh).tanh()
           val ys = (Why dot hs) + by
           val ye = ys.exp()
           val ps = ye / ye.sum()
           sum_loss -= (ps dot y).log()

           if (i < 0) lossOneCycle(i + 1, hs)
           else hprev_new.copy_data(hs)

           // backward pass
           val dy = Tensor.copy(ps)
           dy.data(y_data(i)) -= 1
           dWhy += (dy cart hs)
           dby += dy
           val dh = (Why.trans() dot dy) + dhnext
           val dhraw = (Tensor.ones(1) - hs * hs) * dh
           dbh += dhraw
           dWxh += (dhraw cart x)
           dWhh += (dhraw cart hprev)
           dhnext.copy_data(Whh.trans() dot dhraw)
           ()
         }

         lossOneCycle(0, hprev)

         // assertions
         Tensor.assertEqual(loss1, sum_loss, "loss")
         Tensor.assertEqual(hprev_next, hprev_new, "hidden")
         Tensor.assertEqual(Wxh1.d, dWxh, "dWxh")
         Tensor.assertEqual(Whh1.d, dWhh, "dWhh")
         Tensor.assertEqual(Why1.d, dWhy, "dWhy")
         Tensor.assertEqual(bh1.d, dbh, "dbh")
         Tensor.assertEqual(by1.d, dby, "dby")
        }
      }

      println("try array2_3")
      val array2_3file = new PrintWriter(new File("array2_3.cpp"))
      array2_3file.println(array2_3.code)
      array2_3file.flush()
      println("run test case array2_3")
      array2_3.eval("abc")

      val array2_4 = new DslDriverC[String, Unit] with TensorExp {

        @virtualize
        def snippet (a: Rep[String]): Rep[Unit] = {
          val vocab_size = 3
          val by   = Tensor.zeros(vocab_size)
          val by1  = TensorR(by)
          val y = Tensor.zeros(vocab_size)
          y.data(1) = 1
          val y1 = TensorR(y)

          def lossFun = { (dummy: TensorR) =>

            val e1 = (by1).exp()
            val p1 = e1 / e1.sum()
            (p1 dot y1).log()
          }
          val dummy = gradR(lossFun)(Tensor.zeros(1))
          // by1.d.print()


          // FIXME: need a correct implementation of gradient to check with
        }
      }

      //println("try array2_2_4")
      println("run test case array2_4")
      array2_4.eval("abc")

      val array2_5 = new DslDriverC[String, Unit] with TensorExp {

        @virtualize
        def snippet (a: Rep[String]): Rep[Unit] = {
          val vocab_size = 3
          val e   = Tensor.ones(vocab_size)
          val e1  = TensorR(e)
          val a   = Tensor.ones(vocab_size)
          val a1  = TensorR(a)
          val y = Tensor.zeros(vocab_size)
          y.data(1) = 1
          val y1 = TensorR(y)

          def lossFun = { (dummy: TensorR) =>
            //e1.sum()
            val p1 = a1 / e1.sum()
            (p1 dot y1).log()
          }
          val dummy = gradR(lossFun)(Tensor.zeros(1))
          //e1.d.print()
          //a1.d.print()

          // FIXME: need a correct implementation of gradient to check with
        }
      }
      //println("try array2_2_5")
      println("run test case array2_5")
      array2_5.eval("abc")

      val array3 = new DslDriverC[String, Unit] with TensorExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
          // use random array as input
          val length = 2
          val v = Tensor.randinit(length)
          //v.print()

          // calcuate gradient
          val grad = gradR(t => {val y = IF (length)(t.x.data(0) > 0.0) {t + t}{t * t}
          y.sum() })(v)
          // show gradient
          //grad.print()

          // another way of implementing it
          val grad1 = gradR(t => (t + t).sum())(v)
          val grad2 = gradR(t => (t * t).sum())(v)
          if (v(0) > 0) Tensor.assertEqual(grad, grad1)
          else Tensor.assertEqual(grad, grad2)
        }
      }

      //println("test IF gradient")
      val array3_file = new PrintWriter(new File("array3.cpp"))
      array3_file.println(array3.code)
      array3_file.flush()
      println("run test case array3")
      array3.eval("abc")

      val array4 = new DslDriverC[String, Unit] with TensorExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
          // use random array as input
          val length = 2
          val v = Tensor.randinit(length)
          // v.print()

          val halfv = Tensor.halves(length)
          val half = (new TensorR(halfv, Tensor.zeros(length)))
          // calculate gradient
          val grad = gradR(t => {val y = LOOP(t)(t => t.x.data(0) > 0.1)(t => t * half)
          y.sum() })(v)
          // show gradient
          grad.print()
          //println("Tensor in closure can also accumulate gradient, which is important")
          half.d.print()

          // FIXME: Implement the correct gradient and assertEqual
        }
      }

      // println("test LOOP gradient")
      //println(array4.code)
      val parray4 = new PrintWriter(new File("array4.cpp"))
      parray4.println(array4.code)
      parray4.flush()
      println("run test case array4")
      array4.eval("abc")

      val array4_1 = new DslDriverC[String, Unit] with TensorExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
          val length = 2
          val v = Tensor.randinit(length)
          // v.print()

          val half = new TensorR(Tensor.halves(length), Tensor.zeros(length))
          val grad = gradR(t => {
            val y = LOOPS(t)(3)(i => t => t * half )
            y.sum()
          })(v)
          // show gradient
          //grad.print()
          //println("Tensor in closure can also accumulate gradient, which is important")
          //half.d.print()

          val save_half_grad = Tensor.zeros(length)
          save_half_grad.copy_data(half.d)

          // alternative implementation
          half.d.clear()
          val grad2 = gradR( t => {
            (t * half * half * half).sum()
          })(v)

          // assertion
          Tensor.assertEqual(grad, grad2)
          Tensor.assertEqual(save_half_grad, half.d)
        }
      }

      // println("test LOOP gradient")
      println("run test case array4_1")
      array4_1.eval("abc")

      // test using array data by closure
      val array4_2 = new DslDriverC[String, Unit] with TensorExp {

        def snippet(a: Rep[String]): Rep[Unit] = {

          // random initialization
          val length = 3
          val v = Tensor.randinit(length)
          // v.print()

          // get data from "file" (more like generate static data and lift it to Rep type)
          val ddim0 = 2
          val ddim1 = 3
          val data1 = NewArray[Double](ddim1)
          val data2 = NewArray[Double](ddim1)
          for (i <- (0 until ddim1): Rep[Range]) {
            data1(i) = (i + 1)
            data2(i) = (i + 1) * 2
          }
          val data = NewArray[Array[Double]](ddim0)
          data(0) = data1; data(1) = data2

          val model: TensorR => TensorR @diff = { (x: TensorR) =>
            val y = LOOPS(x)(ddim0)(i => x1 => {
              val data_point = TensorR(Tensor(data(i), ddim1))
              x1 * data_point
            })
            y.sum()
          }

          val grad = gradR(model)(v)
          // show gradient
          //grad.print()

          // alternative implememetation
          val grad1 = gradR(t =>
              (t * TensorR(Tensor(data(0), ddim1)) * TensorR(Tensor(data(1), ddim1))).sum()
              )(v)
          // assertion
          Tensor.assertEqual(grad, grad1)
        }
      }

      //println(array4_2_1.code)
      //val array4_2_1_file = new PrintWriter(new File("array4_2_1.cpp"))
      //array4_2_1_file.println(array4_2_1.code)
      //array4_2_1_file.flush()
      println("run test case of array4_2")
      array4_2.eval("abc")

      val array4_4 = new DslDriverC[String, Unit] with TensorExp {

        def snippet(a: Rep[String]): Rep[Unit] = {
          val length = 2
          val v = Tensor.randinit(length)
          //v.print()
          val u = Tensor.randinit(length, seed = Some(5))
          //u.print()

          val half = new TensorR(Tensor.halves(length), Tensor.zeros(length))
          val vv = TensorR(v)
          val uu = TensorR(u)

          val dummy = gradR(dum => {
            val in = ArrayBuffer[TensorR](vv, uu)
            val y = LOOPSM(in)(3)(i => ins => {
              val vvv = ins(0) * half
              val uuu = ins(1) * half
              ArrayBuffer[TensorR](vvv, uuu)
            })
          y(1).sum() + y(0).sum()})(Tensor.zeros(1))
          // show gradient
          //println("Tensor in closure can also accumulate gradient, which is important")
          //half.d.print()
          //vv.d.print()
          //uu.d.print()

          // save gradients
          val save_vv_grad = Tensor.zeros(length); save_vv_grad.copy_data(vv.d);   vv.clear_grad()
          val save_uu_grad = Tensor.zeros(length); save_uu_grad.copy_data(uu.d);   uu.clear_grad()
          val save_ha_grad = Tensor.zeros(length); save_ha_grad.copy_data(half.d); half.clear_grad()

          // alternative implementation
          val dummy1 = gradR(dum => {
            (vv * half * half * half + uu * half * half * half).sum()
          })(Tensor.zeros(1))

          // assertions
          Tensor.assertEqual(save_ha_grad, half.d)
          Tensor.assertEqual(save_vv_grad, vv.d)
          Tensor.assertEqual(save_uu_grad, uu.d)
        }
      }

      //println("support 2 tensors in loop using LOOPCCM")
      //println(array4_4.code)
      //val array4_4_file = new PrintWriter(new File("array4_4.cpp"))
      //array4_4_file.println(array4_4.code)
      //array4_4_file.flush()
      println("run test case in array4_4")
      array4_4.eval("abc")

      val array5 = new DslDriverC[String, Unit] with TensorExp {

        def snippet(a: Rep[String]): Rep[Unit] = {
          val length = 2
          val v = Tensor.randinit(length)
          //v.print()

          val grad = gradR(t => (t * t).sum())(v)
          //grad.print()

          Tensor.assertEqual(grad, v * 2.0)
        }
      }

      //println("test elementwise multiplication")
      //println(array5.code)
      println("run test case in array5")
      array5.eval("abc")

      val array6 = new DslDriverC[String, Unit] with TensorExp {

        def snippet(a: Rep[String]): Rep[Unit] = {
          val length = 2
          val v = Tensor.randinit(length)
          //v.print()

          val grad = gradR(t => (t / t).sum())(v)
          //grad.print()

          Tensor.assertEqual(grad, Tensor.zeros(length))
        }
      }

      // println("test elementwise division")
      //println(array6.code)
      println("run test case in array6")
      array6.eval("abc")

      val array7 = new DslDriverC[String, Unit] with TensorExp {

        def snippet(a: Rep[String]): Rep[Unit] = {
          val length = 2
          val v = Tensor.randinit(length)
          //v.print()

          val grad = gradR(t => (t.tanh()).sum())(v)
          //grad.print()

          val e1 = v.tanh();
          val ee = Tensor.ones(length) - e1 * e1
          Tensor.assertEqual(grad, ee)
        }
      }

      // println("test tanh")
      //println(array7.code)
      println("run test case array7")
      array7.eval("abc")

      val array7_1 = new DslDriverC[String, Unit] with TensorExp {

        def snippet(a: Rep[String]): Rep[Unit] = {
          val length = 2
          val v = Tensor.randinit(length)

          val grad = gradR(t => (t.sigmoid()).sum())(v)

          val e1 = v.sigmoid()
          val ee = (Tensor.ones(1) - e1) * e1
          Tensor.assertEqual(grad, ee)
        }
      }

      println("run test case array7_1")
      array7_1.eval("abc")

      val array8 = new DslDriverC[String, Unit] with TensorExp {

        def snippet(a: Rep[String]): Rep[Unit] = {
          val length = 2
          val v = Tensor.randinit(length)
          // v.print()

          val grad = gradR(t => (t.exp()).sum())(v)
          //grad.print()

          Tensor.assertEqual(grad, v.exp())
        }
      }

      // println("test exp")
      //println(array8.code)
      println("run test case in array8")
      array8.eval("abc")

      val array9 = new DslDriverC[String, Unit] with TensorExp {

        def snippet(a: Rep[String]): Rep[Unit] = {
          val length = 2
          val v = Tensor.randPositive(length)
          //v.print()

          val grad = gradR(t => (t.log()).sum())(v)
          //grad.print()

          Tensor.assertEqual(grad, Tensor.ones(length) / v)
        }
      }

      //println("test log")
      // println(array9.code)
      println("run test case array9")
      array9.eval("abc")

      val array10 = new DslDriverC[String, Unit] with TensorExp {

        def snippet(a: Rep[String]): Rep[Unit] = {
          val length = 2
          val v = Tensor.randinit(length)
          //v.print()

          val arra = NewArray[Array[Double]](2)
          arra(0) = NewArray[Double](2)
          arra(0)(0) = 4.0
          arra(0)(1) = 2.0
          arra(1) = NewArray[Double](2)
          arra(1)(0) = 1.5
          arra(1)(1) = 2.0

          // create a model that recursively use the data in arr (originated from list)
          def model: TensorR => TensorR @diff = { (x: TensorR) =>
            LOOPL(x)(arra.length)(i => x1 => new TensorR(Tensor(arra(i), length), Tensor.zeros(length)) * x1)
          }
          val grad = gradR(t => (model(t)).sum())(v)
          //grad.print()

          val grad1 = gradR(t =>
              (t * TensorR(Tensor(arra(0), length)) * TensorR(Tensor(arra(1), length))).sum()
              )(v)

          Tensor.assertEqual(grad, grad1)
        }
      }

      //println(array10.code)
      println("run test case in array10")
      array10.eval("abc")

      val array11 = new DslDriverC[String, Unit] with TensorExp {

        def snippet(a: Rep[String]): Rep[Unit] = {
          val length = 2
          val v = Tensor.randinit(length)
          //v.print()

          /*
          5.0, 4.0
          /       \
          /         \
          3.0, 2.0   1.5, 1.4
          */

         val arra = NewArray[Array[Double]](3)
         arra(0) = NewArray[Double](2)
         arra(0)(0) = 5.0; arra(0)(1) = 4.0
         arra(1) = NewArray[Double](2)
         arra(1)(0) = 3.0; arra(1)(1) = 2.0
         arra(2) = NewArray[Double](2)
         arra(2)(0) = 1.5; arra(2)(1) = 1.4
         val lch1 = NewArray[Int](3)
         lch1(0) = 1; lch1(1) = -1; lch1(2) = -1
         val rch1 = NewArray[Int](3)
         rch1(0) = 2; rch1(1) = -1; rch1(2) = -1

         // create a model that recursively use the data (originated from tree)
         def model: TensorR => TensorR @diff = { (x: TensorR) =>
           LOOPT(x)(lch1, rch1){ (l: TensorR, r: TensorR, i: Rep[Int]) =>
             l * r * new TensorR(Tensor(arra(i), length), Tensor.zeros(length))
           }
         }

         val grad = gradR(t => model(t).sum())(v)
         //grad.print()

         def model1: TensorR => TensorR @diff = { (x: TensorR) =>
           val leftchild  = x * TensorR(Tensor(arra(1), length)) * x
           val rightchild = x * TensorR(Tensor(arra(2), length)) * x
           val root = leftchild * TensorR(Tensor(arra(0), length)) * rightchild
           root.sum()
         }

         val grad1 = gradR(model1)(v)
         // assertion
         Tensor.assertEqual(grad, grad1)
        }
      }

      //println(array11.code)
      println("run test case array11")
      array11.eval("abc")

      val array11_1 = new DslDriverC[String, Unit] with TensorExp {

        def snippet(a: Rep[String]): Rep[Unit] = {
          val length = 2
          val v = Tensor.randinit(length)
          //v.print()

          /*
          5.0, 4.0
          /       \
          /         \
          3.0, 2.0   1.5, 1.4
          */

         val arra = NewArray[Array[Double]](3)
         arra(0) = NewArray[Double](2)
         arra(0)(0) = 5.0; arra(0)(1) = 4.0
         arra(1) = NewArray[Double](2)
         arra(1)(0) = 3.0; arra(1)(1) = 2.0
         arra(2) = NewArray[Double](2)
         arra(2)(0) = 1.5; arra(2)(1) = 1.4
         val lch1 = NewArray[Int](3)
         lch1(0) = 1; lch1(1) = -1; lch1(2) = -1
         val rch1 = NewArray[Int](3)
         rch1(0) = 2; rch1(1) = -1; rch1(2) = -1

         val add: TensorR = TensorR(Tensor.ones(length))

         // create a model that recursively use the data (originated from tree)
         def model: TensorR => TensorR @diff = { (x: TensorR) =>
           val in = new ArrayBuffer[TensorR](); in.append(x); in.append(add)
           val tmp = LOOPTM(in)(lch1, rch1){ (l: ArrayBuffer[TensorR], r: ArrayBuffer[TensorR], i: Rep[Int]) =>
             val curr = TensorR(Tensor(arra(i), length))
             val new_x = l(0) * r(0) * curr; val new_add = l(1) + r(1) + curr
             val out = new ArrayBuffer[TensorR](); out.append(new_x); out.append(new_add)
             out
           }
           tmp(0).sum() + tmp(1).sum()
         }

         val grad = gradR(t => model(t))(v)
         //grad.print()
         // save gradient of add
         val save_grad_add = Tensor.zeros(length); save_grad_add.copy_data(add.d); add.clear_grad()

         def model1: TensorR => TensorR @diff = { (x: TensorR) =>
           val val1 = TensorR(Tensor(arra(1), length))
           val leftchild  = x * val1 * x; val leftch = add + val1 + add
           val val2 = TensorR(Tensor(arra(2), length))
           val rightchild = x * val2 * x; val rightch = add + val2 + add
           val val0 = TensorR(Tensor(arra(0), length))
           val root = leftchild * val0 * rightchild; val root2 = leftch + val0 + rightch
           root.sum() + root2.sum()
         }

         val grad1 = gradR(model1)(v)
         // assertion
         Tensor.assertEqual(grad, grad1)
         Tensor.assertEqual(save_grad_add, add.d)
        }
      }

      //println(array11.code)
      println("run test case array11_1")
      array11_1.eval("abc")

    } // if (false) closing

      val root_dir = "/home/fei/bitbucket/privategitrepoforshare/ICFP18evaluation/"

      val min_char_rnn = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        class Scanner(name: Rep[String]) {
          val fd = open(name)
          val fl = filelen(fd)
          val data = mmap[Char](fd,fl)
          var pos = 0

          def nextChar: Rep[Char] = {
            val ch = data(pos)
            pos += 1
            ch
          }

          def hasNextChar = pos < fl
          def done = close(fd)
        }


        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
          /**
           add scanner
           **/

          val startTime = get_time()

          val scanner = new Scanner("graham.txt")
          val training_data = scanner.data
          val data_size = scanner.fl
          // val chars = training_data.distinct  /** this can be done in second stage **/
          // val vocab_size = chars.length
          printf("data has %d chars\\n", data_size)

          //val translated_data = NewArray[Int](data_size)
          //for (i <- (0 until data_size)) translated_data(i) = char_to_ix(unit(training_data).charAt(i))
          val translated_data = NewArray[Int](data_size)
          for (i <- (0 until data_size)) { translated_data(i) = Encoding.char_to_ix(training_data(i)) }

          val vocab_size = 26                 // Do we have to get this size?
          val hidden_size = 50
          val learning_rate = 1e-1
          val seq_length = 20
          //val Wxh = Tensor.randinit(vocab_size, hidden_size, 0.01)  // input to hidden
          val Wxh = Tensor.randn(hidden_size, vocab_size, 0.01)  // input to hidden
          val Whh = Tensor.randn(hidden_size, hidden_size, 0.01) // hidden to hidden
          val Why = Tensor.randn(vocab_size, hidden_size, 0.01)  // hidden to output
          val bh  = Tensor.zeros(hidden_size)
          val by  = Tensor.zeros(vocab_size)
          val hprev = Tensor.zeros(hidden_size)

          val hnext = Tensor.zeros_like(hprev)

          // wrap as tensors
          val Wxh1 = TensorR(Wxh)
          val Whh1 = TensorR(Whh)
          val Why1 = TensorR(Why)
          val bh1  = TensorR(bh)
          val by1  = TensorR(by)
          val hprev1 = TensorR(hprev)

          def lossFun(inputs: Rep[Array[Int]], targets: Rep[Array[Int]]) = { (dummy: TensorR) =>
            val loss = TensorR(Tensor.zeros(1))
            val in = ArrayBuffer[TensorR]()
            in.append(loss)
            in.append(hprev1)
            val outputs = LOOPSM(in)(inputs.length){i => t =>

              // printf("at iteration %d ", i)
              // get input as one-hot tensor
              val x = Tensor.zeros(vocab_size)
              x.data(inputs(i)) = 1
              val x1 = TensorR(x)
              // get output as one-hot tensor
              val y = Tensor.zeros(vocab_size)
              y.data(targets(i)) = 1
              val y1 = TensorR(y)

              val h1 = ((Wxh1 dot x1) + (Whh1 dot t(1)) + bh1).tanh() // use hidden state and x1 to compute hidden state
              val e1 = (Why1.dot(h1) + by1).exp()                       // use new hidden state to compute unnormalized prob
              val p1 = e1 / e1.sum()                            // use unnormalized prob to compute normalize prob
              val newloss = t(0) - (p1 dot y1).log()            // loss is updated by original loss t(0) and additional loss
              val out = ArrayBuffer[TensorR]()
              out.append(newloss)
              out.append(h1)
              out
            }
            hnext.copy_data(outputs(1).x)     // update the hidden state with the result from LOOP
            outputs(0)                        // return the final loss
          }


          val lr = learning_rate
          val hp = 1e-8

          val mWxh = Tensor.zeros_like(Wxh)
          val mWhh = Tensor.zeros_like(Whh)
          val mWhy = Tensor.zeros_like(Why)
          val mbh  = Tensor.zeros_like(bh)
          val mby  = Tensor.zeros_like(by)

          val loss_save = NewArray[Double](51)
          val loopStartTime = get_time()

          val addr = getMallocAddr() // remember current allocation pointer here

          val startAt = var_new[Int](0)
          startAt -= seq_length

          var smooth_loss = 60.0
          for (n <- (0 until 5001): Rep[Range]) {

            startAt += seq_length
            if (startAt + seq_length + 1 >= data_size) {
              startAt = 0
              hprev.clear()
            }

            val inputs = NewArray[Int](seq_length)
            val targets = NewArray[Int](seq_length)
            for (i <- (0 until seq_length): Rep[Range]) {
              inputs(i) = translated_data(startAt+i)
              targets(i) = translated_data(startAt+i+1)
            }

            val loss = gradR_loss(lossFun(inputs, targets))(Tensor.zeros(1))
            val loss_value = loss.data(0) // we suppose the loss is scala (Tensor of size 1)
            smooth_loss = smooth_loss * 0.9 + loss_value * 0.1
            if (n % 100 == 0) {
              printf("iter %d, loss %f\\n", n, smooth_loss)
              loss_save(n / 100) = smooth_loss
            }

            val pars = ArrayBuffer(Wxh1, Whh1, Why1, bh1, by1)
            val mems = ArrayBuffer(mWxh, mWhh, mWhy, mbh, mby)
            for ((par, mem) <- pars.zip(mems)) {
              par.clip_grad(5.0)
              mem += par.d * par.d
              par.x -= par.d * lr / (mem + hp).sqrt()
              par.clear_grad()
            }
            hprev1.clear_grad()          // clear gradient of all Tensors for next cycle
            hprev1.x.copy_data(hnext)

            resetMallocAddr(addr)  // reset malloc_addr to the value when we remember allocation pointer
          }

          val loopEndTime = get_time()
          val prepareTime = loopStartTime - startTime
          val loopTime    = loopEndTime - loopStartTime

          val fp = openf(a, "w")
          fprintf(fp, "unit: %s\\n", "100 iteration")
          for (i <- (0 until loss_save.length): Rep[Range]) {
            fprintf(fp, "%lf\\n", loss_save(i))
          }
          fprintf(fp, "run time: %lf %lf\\n", prepareTime, loopTime)
          closef(fp)

        }
      }

      //println("run min_char_rnn")
      //val min_char_rnn_file = new PrintWriter(new File(root_dir + "evaluationRNN/Lantern.cpp"))
      //min_char_rnn_file.println(min_char_rnn.code)
      //min_char_rnn_file.flush()
      //min_char_rnn.eval("abc")
      //println("verified that in this small example the values of gradients are about right (up to precision)")

      val min_char_list = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        class Scanner(name: Rep[String]) {
          val fd = open(name)
          val fl = filelen(fd)
          val data = mmap[Char](fd,fl)
          var pos = 0

          def nextChar: Rep[Char] = {
            val ch = data(pos)
            pos += 1
            ch
          }

          def hasNextChar = pos < fl
          def done = close(fd)
        }

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
          /**
           add scanner
           **/
          val scanner = new Scanner("test_data")
          val training_data = scanner.data
          val data_size = scanner.fl
          // val chars = training_data.distinct  /** this can be done in second stage **/
          // val vocab_size = chars.length
          printf("data has %d chars\\n", data_size)

          //val translated_data = NewArray[Int](data_size)
          //for (i <- (0 until data_size)) translated_data(i) = char_to_ix(unit(training_data).charAt(i))
          val translated_data = NewArray[Int](data_size)
          for (i <- (0 until data_size)) { translated_data(i) = Encoding.char_to_ix(training_data(i)) }

          val vocab_size = 26                 // Do we have to get this size?
          val hidden_size = 50
          val learning_rate = 1e-1
          val seq_length = 20
          //val Wxh = Tensor.randinit(vocab_size, hidden_size, 0.01)  // input to hidden
          val Wxh = Tensor.randn(hidden_size, vocab_size, 0.01)  // input to hidden
          val Whh = Tensor.randn(hidden_size, hidden_size, 0.01) // hidden to hidden
          val Why = Tensor.randn(vocab_size, hidden_size, 0.01)  // hidden to output
          val bh  = Tensor.zeros(hidden_size)
          val by  = Tensor.zeros(vocab_size)
          val hprev = Tensor.zeros(hidden_size)

          val hnext = Tensor.zeros_like(hprev)

          // wrap as tensors
          val Wxh1 = TensorR(Wxh)
          val Whh1 = TensorR(Whh)
          val Why1 = TensorR(Why)
          val bh1  = TensorR(bh)
          val by1  = TensorR(by)
          val hprev1 = TensorR(hprev)

          def lossFun(inputs: Rep[Array[Int]], targets: Rep[Array[Int]]) = { (dummy: TensorR) =>
            val loss = TensorR(Tensor.zeros(1))
            val in = ArrayBuffer[TensorR]()
            in.append(loss)
            in.append(hprev1)
            val outputs = LOOPLM(in)(inputs.length){i => t =>

              // printf("at iteration %d ", i)
              // get input as one-hot tensor
              val x = Tensor.zeros(vocab_size)
              x.data(inputs(i)) = 1
              val x1 = TensorR(x)
              // get output as one-hot tensor
              val y = Tensor.zeros(vocab_size)
              y.data(targets(i)) = 1
              val y1 = TensorR(y)

              val h1 = ((Wxh1 dot x1) + (Whh1 dot t(1)) + bh1).tanh() // use hidden state and x1 to compute hidden state
              val e1 = (Why1.dot(h1) + by1).exp()                       // use new hidden state to compute unnormalized prob
              val p1 = e1 / e1.sum()                            // use unnormalized prob to compute normalize prob
              val newloss = t(0) - (p1 dot y1).log()            // loss is updated by original loss t(0) and additional loss
              val out = ArrayBuffer[TensorR]()
              out.append(newloss)
              out.append(h1)
              out
            }
            hnext.copy_data(outputs(1).x)     // update the hidden state with the result from LOOP
            outputs(0)                        // return the final loss
          }


          val lr = learning_rate
          val hp = 1e-8

          val mWxh = Tensor.zeros_like(Wxh)
          val mWhh = Tensor.zeros_like(Whh)
          val mWhy = Tensor.zeros_like(Why)
          val mbh  = Tensor.zeros_like(bh)
          val mby  = Tensor.zeros_like(by)

          val addr = getMallocAddr() // remember current allocation pointer here

          val startAt = var_new[Int](0)
          startAt -= seq_length

          val timer = Timer()
          timer.startTimer

          for (n <- (0 until 2001): Rep[Range]) {

            startAt += seq_length
            if (startAt + seq_length + 1 >= data_size) {
              startAt = 0
              hprev.clear()
            }

            val inputs = NewArray[Int](seq_length)
            val targets = NewArray[Int](seq_length)
            for (i <- (0 until seq_length): Rep[Range]) {
              inputs(seq_length-1-i) = translated_data(startAt+i)
              targets(seq_length-1-i) = translated_data(startAt+i+1)
            }

            val loss = gradR_loss(lossFun(inputs, targets))(Tensor.zeros(1))
            val loss_value = loss.data(0) // we suppose the loss is scala (Tensor of size 1)
            if (n % 100 == 0) {
              printf("iter %d, loss %f\\n", n, loss_value)
              timer.printElapsedTime
            }

            val pars = ArrayBuffer(Wxh1, Whh1, Why1, bh1, by1)
            val mems = ArrayBuffer(mWxh, mWhh, mWhy, mbh, mby)
            for ((par, mem) <- pars.zip(mems)) {
              par.clip_grad(5.0)
              mem += par.d * par.d
              par.x -= par.d * lr / (mem + hp).sqrt()
              par.clear_grad()
            }
            hprev1.clear_grad()          // clear gradient of all Tensors for next cycle
            hprev1.x.copy_data(hnext)

            resetMallocAddr(addr)  // reset malloc_addr to the value when we remember allocation pointer
          }

        }
      }



      //println("run min_char_list")
      //val min_char_list_file = new PrintWriter(new File(root_dir + "evaluationRNNlist/Lantern.cpp"))
      //min_char_list_file.println(min_char_list.code)
      //min_char_list_file.flush()
      //min_char_list.eval("abc")
      //println("verified that in this small example the values of gradients are about right (up to precision)")


      val min_char_lstm = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        class Scanner(name: Rep[String]) {
          val fd = open(name)
          val fl = filelen(fd)
          val data = mmap[Char](fd,fl)
          var pos = 0

          def nextChar: Rep[Char] = {
            val ch = data(pos)
            pos += 1
            ch
          }

          def hasNextChar = pos < fl
          def done = close(fd)
        }

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
          /**
           add scanner
           **/
          val startTime = get_time()
          val scanner = new Scanner("graham.txt")
          val training_data = scanner.data
          val data_size = scanner.fl
          // val chars = training_data.distinct  /** this can be done in second stage **/
          // val vocab_size = chars.length
          printf("data has %d chars\\n", data_size)

          //val translated_data = NewArray[Int](data_size)
          //for (i <- (0 until data_size)) translated_data(i) = char_to_ix(unit(training_data).charAt(i))
          val translated_data = NewArray[Int](data_size)
          for (i <- (0 until data_size)) { translated_data(i) = Encoding.char_to_ix(training_data(i)) }

          val vocab_size = 26
          val hidden_size = 50
          val learning_rate = 1e-1
          val seq_length = 20

          // initialize all parameters:
          val Wfh = Tensor.randn(hidden_size, hidden_size, 0.01)
          val Wfx = Tensor.randn(hidden_size, vocab_size, 0.01)
          val bf  = Tensor.zeros(hidden_size)
          val Wih = Tensor.randn(hidden_size, hidden_size, 0.01)
          val Wix = Tensor.randn(hidden_size, vocab_size, 0.01)
          val bi  = Tensor.zeros(hidden_size)
          val Wch = Tensor.randn(hidden_size, hidden_size, 0.01)
          val Wcx = Tensor.randn(hidden_size, vocab_size, 0.01)
          val bc  = Tensor.zeros(hidden_size)
          val Woh = Tensor.randn(hidden_size, hidden_size, 0.01)
          val Wox = Tensor.randn(hidden_size, vocab_size, 0.01)
          val bo  = Tensor.zeros(hidden_size)
          val Why = Tensor.randn(vocab_size, hidden_size, 0.01)  // hidden to output
          val by  = Tensor.zeros(vocab_size)

          val hprev = Tensor.zeros(hidden_size)
          val cprev = Tensor.zeros(hidden_size)
          val hsave = Tensor.zeros_like(hprev)
          val csave = Tensor.zeros_like(cprev)

          // wrap as Tensors
          val tWfh = TensorR(Wfh)
          val tWfx = TensorR(Wfx)
          val tbf = TensorR(bf)
          val tWih = TensorR(Wih)
          val tWix = TensorR(Wix)
          val tbi = TensorR(bi)
          val tWch = TensorR(Wch)
          val tWcx = TensorR(Wcx)
          val tbc = TensorR(bc)
          val tWoh = TensorR(Woh)
          val tWox = TensorR(Wox)
          val tbo = TensorR(bo)
          val tWhy = TensorR(Why)
          val tby = TensorR(by)
          val thprev = TensorR(hprev)
          val tcprev = TensorR(cprev)


          // lossFun
          def lossFun(inputs: Rep[Array[Int]], targets: Rep[Array[Int]]) = { (dummy: TensorR) =>

            val loss = TensorR(Tensor.zeros(1))
            val in = ArrayBuffer[TensorR]()

            in.append(loss)
            in.append(thprev)
            in.append(tcprev)

            val outputs = LOOPSM(in)(inputs.length){i => t =>

              // get input as one-hot tensor
              val x = Tensor.zeros(vocab_size)
              x.data(inputs(i)) = 1
              val x1 = TensorR(x)
              // get output as one-hot tensor
              val y = Tensor.zeros(vocab_size)
              y.data(targets(i)) = 1
              val y1 = TensorR(y)

              val ft = (tWfh.dot(t(1)) + tWfx.dot(x1) + tbf).sigmoid()
              val it = (tWih.dot(t(1)) + tWix.dot(x1) + tbi).sigmoid()
              val ot = (tWoh.dot(t(1)) + tWox.dot(x1) + tbo).sigmoid()
              val Ct = (tWch.dot(t(1)) + tWcx.dot(x1) + tbc).tanh()
              val ct = ft * t(2) + it * Ct
              val ht = ot * ct.tanh()
              val et = (tWhy.dot(ht) + tby).exp()
              val pt = et / et.sum()
              val loss = t(0) - (pt dot y1).log()

              val out = ArrayBuffer[TensorR]()
              out.append(loss)
              out.append(ht)
              out.append(ct)
              out
            }
            hsave.copy_data(outputs(1).x)     // save the hidden state with the result from LOOP
            csave.copy_data(outputs(2).x)     // save the cell state with the result from LOOP
            outputs(0)                        // return the final loss
          }


          val lr = learning_rate
          val hp = 1e-8

          val mWfh = Tensor.zeros_like(Wfh)
          val mWfx = Tensor.zeros_like(Wfx)
          val mbf = Tensor.zeros_like(bf)
          val mWih = Tensor.zeros_like(Wih)
          val mWix = Tensor.zeros_like(Wix)
          val mbi = Tensor.zeros_like(bi)
          val mWch = Tensor.zeros_like(Wch)
          val mWcx = Tensor.zeros_like(Wcx)
          val mbc = Tensor.zeros_like(bc)
          val mWoh = Tensor.zeros_like(Woh)
          val mWox = Tensor.zeros_like(Wox)
          val mbo = Tensor.zeros_like(bo)
          val mWhy = Tensor.zeros_like(Why)
          val mby = Tensor.zeros_like(by)

          val loopStart = get_time()
          val loss_save = NewArray[Double](51)

          val addr = getMallocAddr() // remember current allocation pointer here

          val startAt = var_new[Int](0)
          startAt -= seq_length

          var smooth_loss = 70.0
          for (n <- (0 until 5001): Rep[Range]) {

            startAt += seq_length
            if (startAt + seq_length + 1 >= data_size) {
              startAt = 0
              hprev.clear()
            }

            val inputs = NewArray[Int](seq_length)
            val targets = NewArray[Int](seq_length)
            for (i <- (0 until seq_length): Rep[Range]) {
              inputs(i) = translated_data(startAt+i)
              targets(i) = translated_data(startAt+i+1)
            }

            val loss = gradR_loss(lossFun(inputs, targets))(Tensor.zeros(1))
            val loss_value = loss.data(0) // we suppose the loss is scala (Tensor of size 1)
            smooth_loss = smooth_loss * 0.9 + loss_value * 0.1
            if (n % 100 == 0) {
              printf("iter %d, loss %f\\n", n, smooth_loss)
              loss_save(n / 100) = smooth_loss
            }

            val pars = ArrayBuffer(tWfh, tWfx, tbf, tWih, tWix, tbi, tWch, tWcx, tbc, tWoh, tWox, tbo, tWhy, tby)
            val mems = ArrayBuffer(mWfh, mWfx, mbf, mWih, mWix, mbi, mWch, mWcx, mbc, mWoh, mWox, mbo, mWhy, mby)
            for ((par, mem) <- pars.zip(mems)) {
              par.clip_grad(5.0)
              mem += par.d * par.d
              par.x -= par.d * lr / (mem + hp).sqrt()
              par.clear_grad()
            }
            thprev.clear_grad()          // clear gradient of all Tensors for next cycle
            tcprev.clear_grad()          // clear gradient of all Tensors for next cycle
            thprev.x.copy_data(hsave)
            tcprev.x.copy_data(csave)

            resetMallocAddr(addr)  // reset malloc_addr to the value when we remember allocation pointer
          }

          val loopEndTime = get_time()
          val prepareTime = loopStart - startTime
          val loopTime    = loopEndTime - loopStart

          val fp = openf(a, "w")
          fprintf(fp, "unit: %s\\n", "100 iteration")
          for (i <- (0 until loss_save.length): Rep[Range]) {
            //printf("loss_saver is %lf \\n", loss_save(i))
            fprintf(fp, "%lf\\n", loss_save(i))
          }
          fprintf(fp, "run time: %lf %lf\\n", prepareTime, loopTime)
          closef(fp)

        }
      }

      //println("run min_char_lstm")
      //val min_char_lstm_file = new PrintWriter(new File(root_dir + "evaluationLSTM/Lantern.cpp"))
      //min_char_lstm_file.println(min_char_lstm.code)
      //min_char_lstm_file.flush()
      //min_char_lstm.eval("abc")
      //println("verified that in this small example the values of gradients are about right (up to precision)")

      val senti_seq_lstm = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {

          // read in the data for word embedding
          val word_embedding_size   = 300
          val word_embedding_length = 5265 // need to know the size of file first, need fix
          val fp = openf("senti/small_glove.txt", "r")
          val word_embedding_data = NewArray[Array[Double]](word_embedding_length)

          for (i <- (0 until word_embedding_length): Rep[Range]) {
            word_embedding_data(i) = NewArray[Double](word_embedding_size)
            for (j <- (0 until word_embedding_size): Rep[Range]) getFloat(fp, word_embedding_data(i), j)
          }
          closef(fp)

          // read in the data for sequences (assume size, label, newline, word_indexes)
          val seq_number = 1101 // need to know the size of training data, need fix
          val fp1 = openf("senti/array_seq.txt", "r")
          val seq_data  = NewArray[Array[Int]](seq_number)
          val seq_label = NewArray[Int](seq_number)

          val size = NewArray[Int](1)
          for (i <- (0 until seq_number): Rep[Range]) {
            getInt(fp1, size, 0)
            seq_data(i) = NewArray[Int](size(0))
            getInt(fp1, seq_label, i)
            for (k <- (0 until size(0)): Rep[Range]) getInt(fp1, seq_data(i), k)
          }

          val hidden_size = 150
          val output_size = 5
          val learning_rate = 1e-1

          // initialize all parameters:
          val Wfh = Tensor.randn(hidden_size, hidden_size, 0.01)
          val Wfx = Tensor.randn(hidden_size, word_embedding_size, 0.01)
          val bf  = Tensor.zeros(hidden_size)
          val Wih = Tensor.randn(hidden_size, hidden_size, 0.01)
          val Wix = Tensor.randn(hidden_size, word_embedding_size, 0.01)
          val bi  = Tensor.zeros(hidden_size)
          val Wch = Tensor.randn(hidden_size, hidden_size, 0.01)
          val Wcx = Tensor.randn(hidden_size, word_embedding_size, 0.01)
          val bc  = Tensor.zeros(hidden_size)
          val Woh = Tensor.randn(hidden_size, hidden_size, 0.01)
          val Wox = Tensor.randn(hidden_size, word_embedding_size, 0.01)
          val bo  = Tensor.zeros(hidden_size)
          val Why = Tensor.randn(output_size, hidden_size, 0.01)  // hidden to output
          val by  = Tensor.zeros(output_size)

          val hprev = Tensor.zeros(hidden_size)
          val cprev = Tensor.zeros(hidden_size)

          // wrap as Tensors
          val tWfh = TensorR(Wfh)
          val tWfx = TensorR(Wfx)
          val tbf = TensorR(bf)
          val tWih = TensorR(Wih)
          val tWix = TensorR(Wix)
          val tbi = TensorR(bi)
          val tWch = TensorR(Wch)
          val tWcx = TensorR(Wcx)
          val tbc = TensorR(bc)
          val tWoh = TensorR(Woh)
          val tWox = TensorR(Wox)
          val tbo = TensorR(bo)
          val tWhy = TensorR(Why)
          val tby = TensorR(by)
          val thprev = TensorR(hprev)
          val tcprev = TensorR(cprev)

          // lossFun
          def lossFun(inputs: Rep[Array[Int]], label: Rep[Int]) = { (dummy: TensorR) =>

            val in = ArrayBuffer[TensorR]()
            in.append(thprev)
            in.append(tcprev)

            val outputs = LOOPSM(in)(inputs.length){i => t =>

              // get word embedding
              val x    = word_embedding_data(inputs(i))
              val x1   = TensorR(Tensor(x, word_embedding_size))

              val ft = (tWfh.dot(t(0)) + tWfx.dot(x1) + tbf).sigmoid()
              val it = (tWih.dot(t(0)) + tWix.dot(x1) + tbi).sigmoid()
              val ot = (tWoh.dot(t(0)) + tWox.dot(x1) + tbo).sigmoid()
              val Ct = (tWch.dot(t(0)) + tWcx.dot(x1) + tbc).tanh()
              val ct = ft * t(1) + it * Ct
              val ht = ot * ct.tanh()

              val out = ArrayBuffer[TensorR]()
              out.append(ht)
              out.append(ct)
              out
            }
            val et = (tWhy.dot(outputs(0)) + tby).exp()
            val pt = et / et.sum()

            val y = Tensor.zeros(output_size)
            y.data(label) = 1
            val y1 = TensorR(y)

            val loss = TensorR(Tensor.zeros(1)) - (pt dot y1).log()
            loss
          }


          val lr = learning_rate
          val hp = 1e-8

          val mWfh = Tensor.zeros_like(Wfh)
          val mWfx = Tensor.zeros_like(Wfx)
          val mbf = Tensor.zeros_like(bf)
          val mWih = Tensor.zeros_like(Wih)
          val mWix = Tensor.zeros_like(Wix)
          val mbi = Tensor.zeros_like(bi)
          val mWch = Tensor.zeros_like(Wch)
          val mWcx = Tensor.zeros_like(Wcx)
          val mbc = Tensor.zeros_like(bc)
          val mWoh = Tensor.zeros_like(Woh)
          val mWox = Tensor.zeros_like(Wox)
          val mbo = Tensor.zeros_like(bo)
          val mWhy = Tensor.zeros_like(Why)
          val mby = Tensor.zeros_like(by)

          val addr = getMallocAddr() // remember current allocation pointer here

          for (n <- (0 until 2001): Rep[Range]) {

            val index  = n % seq_number
            val inputs = seq_data(index)
            val label  = seq_label(index)

            val loss = gradR_loss(lossFun(inputs, label))(Tensor.zeros(1))
            val loss_value = loss.data(0) // we suppose the loss is scala (Tensor of size 1)
            if (n % 100 == 0) {
              printf("iter %d, loss %f\\n", n, loss_value)
              //timer.printElapsedTime
            }

            val pars = ArrayBuffer(tWfh, tWfx, tbf, tWih, tWix, tbi, tWch, tWcx, tbc, tWoh, tWox, tbo, tWhy, tby)
            val mems = ArrayBuffer(mWfh, mWfx, mbf, mWih, mWix, mbi, mWch, mWcx, mbc, mWoh, mWox, mbo, mWhy, mby)
            for ((par, mem) <- pars.zip(mems)) {
              par.clip_grad(5.0)
              mem += par.d * par.d
              par.x -= par.d * lr / (mem + hp).sqrt()
              par.clear_grad()
            }
            thprev.clear_grad()          // clear gradient of all Tensors for next cycle
            tcprev.clear_grad()          // clear gradient of all Tensors for next cycle

            resetMallocAddr(addr)  // reset malloc_addr to the value when we remember allocation pointer
          }
        }
      }


      //println("try senti_seq_lstm")
      //val min_char_rnn_file = new PrintWriter(new File("senti_seq_lstm.cpp"))
      //min_char_rnn_file.println(senti_seq_lstm.code)
      //min_char_rnn_file.flush()
      //senti_seq_lstm.eval("abc")
      //println("verified that in this small example the values of gradients are about right (up to precision)")


      val sentimental_rnn = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {

          // read in the data for word embedding
          val word_embedding_size   = 300
          val word_embedding_length = 5265 // need to know the size of file first, need fix
          val fp = openf("senti/small_glove.txt", "r")
          val word_embedding_data = NewArray[Array[Double]](word_embedding_length)

          for (i <- (0 until word_embedding_length): Rep[Range]) {
            word_embedding_data(i) = NewArray[Double](word_embedding_size)
            for (j <- (0 until word_embedding_size): Rep[Range]) getFloat(fp, word_embedding_data(i), j)
          }
          closef(fp)

          // read in the data for trees
          val tree_number = 1101 // need to know the size of training data, need fix
          val fp1 = openf("senti/array_tree.txt", "r")
          val tree_data = NewArray[Array[Int]](tree_number * 4) // each tree data has 4 lines (score, word, lch, rch)

          val size = NewArray[Int](1)
          for (i <- (0 until tree_number): Rep[Range]) {
            getInt(fp1, size, 0)
            for (j <- (0 until 4): Rep[Range]) {
              tree_data(i * 4 + j) = NewArray[Int](size(0))
              for (k <- (0 until size(0)): Rep[Range]) getInt(fp1, tree_data(i * 4 + j), k)
            }
          }

          /* // this piece of code proves that the data reading is correct
          for (j <- (0 until 4): Rep[Range]) {
            val barray = tree_data(j)
            for (k <- (0 until size(0)): Rep[Range]) printf("%d ", barray(k))
            printf("\\n")
          }

          val carray = tree_data(1)
          for (j <- (0 until size(0)):Rep[Range]) {
            if (carray(j) > 0) {
              val darray = word_embedding_data(carray(j))
              for (t <- (0 until word_embedding_size): Rep[Range]) printf("%lf ", darray(t))
              printf("\\n")
            }
          }*/


         // set up hyperparameters and parameters
         val hidden_size = 100
         val output_size = 5
         val learning_rate = 0.05
         val Wxh = Tensor.randinit(hidden_size, word_embedding_size, 0.01) // from word embedding to hidden vector
         val bx  = Tensor.zeros(hidden_size)                               // bias word embedding to hidden vector
         val Wlh = Tensor.randinit(hidden_size, hidden_size, 0.01)         // from hidden vector of left child to hidden
         val Wrh = Tensor.randinit(hidden_size, hidden_size, 0.01)         // from hidden vector of right child to hidden
         val bh  = Tensor.zeros(hidden_size)                               // bias from children hidden vector to hidden
         val Why = Tensor.randinit(output_size, hidden_size, 0.01)         // from hidden vector to output
         val by  = Tensor.zeros(output_size)                               // bias hidden vector to output

         // Cast Tensors as Tensors
         val Wxh1 = TensorR(Wxh)
         val bx1  = TensorR(bx)
         val Wlh1 = TensorR(Wlh)
         val Wrh1 = TensorR(Wrh)
         val bh1  = TensorR(bh)
         val Why1 = TensorR(Why)
         val by1  = TensorR(by)

         def lossFun(scores: Rep[Array[Int]], words: Rep[Array[Int]], lchs: Rep[Array[Int]], rchs: Rep[Array[Int]]) = { (dummy: TensorR) =>

           val initial_loss = TensorR(Tensor.zeros(1))
           val initial_hidd = TensorR(Tensor.zeros(hidden_size))
           val inBuffer     = new ArrayBuffer[TensorR]()
           inBuffer.append(initial_loss); inBuffer.append(initial_hidd) // construct the input to LOOPTM

           val outBuffer = LOOPTM(inBuffer)(lchs, rchs) { (l: ArrayBuffer[TensorR], r: ArrayBuffer[TensorR], i: Rep[Int]) =>

             val targ = Tensor.zeros(output_size); targ.data(scores(i)) = 1; val targ1 = TensorR(targ)
             val lossl = l(0); val hiddenl = l(1)
             val lossr = r(0); val hiddenr = r(1)

             val hidden = IF (hidden_size) (lchs(i) < 0) { // leaf node
               val embedding_array = word_embedding_data(words(i))
               val embedding_tensor = TensorR(Tensor(embedding_array, word_embedding_size))
               (Wxh1.dot(embedding_tensor) + bx1).tanh()
             } { (Wlh1.dot(hiddenl) + Wrh1.dot(hiddenr) + bh1).tanh() } // non-leaf node
             val pred1 = (Why1.dot(hidden) + by1).exp()
             val pred2 = pred1 / pred1.sum()
             val loss = lossl + lossr - (pred2 dot targ1).log()
             val out = ArrayBuffer[TensorR]()
             out.append(loss)
             out.append(hidden)
             out
           }
           outBuffer(0)
         }

         val lr = learning_rate
         val hp = 1e-8

         val mWxh = Tensor.zeros_like(Wxh)
         val mbx  = Tensor.zeros_like(bx)
         val mWlh = Tensor.zeros_like(Wlh)
         val mWrh = Tensor.zeros_like(Wrh)
         val mbh  = Tensor.zeros_like(bh)
         val mWhy = Tensor.zeros_like(Why)
         val mby  = Tensor.zeros_like(by)

         val addr = getMallocAddr() // remember current allocation pointer here

         for (epoc <- (0 until 10): Rep[Range]) {

           var ave_loss = 0.0
           for (n <- (0 until tree_number): Rep[Range]) {

             val index = n % tree_number
             val scores   = tree_data(index * 4)
             val words    = tree_data(index * 4 + 1)
             val leftchs  = tree_data(index * 4 + 2)
             val rightchs = tree_data(index * 4 + 3)
             val loss = gradR_loss(lossFun(scores, words, leftchs, rightchs))(Tensor.zeros(1))
             val loss_value = loss.data(0)  // we suppose the loss is scala (Tensor of size 1)
             ave_loss = ave_loss * n / (n + 1) + loss_value / (n + 1)

             val pars = ArrayBuffer(Wxh1, bx1, Wlh1, Wrh1, bh1, Why1, by1)
             val mems = ArrayBuffer(mWxh, mbx, mWlh, mWrh, mbh, mWhy, mby)
             for ((par, mem) <- pars.zip(mems)) {
               par.clip_grad(1.0)
               mem += par.d * par.d
               par.x -= par.d * lr / (mem + hp).sqrt()
               par.clear_grad()
             }

             resetMallocAddr(addr)  // reset malloc_addr to the value when we remember allocation pointer */
           }

           printf("epoc %d, ave_loss %f\\n", epoc, ave_loss)
         }

        }
      }


      //val senti_file = new PrintWriter(new File("senti.cpp"))
      //senti_file.println(sentimental_rnn.code)
      //senti_file.flush()
      //sentimental_rnn.eval("abc")

      val sentimental_lstm = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {

          val startTime = get_time()

          // read in the data for word embedding
          val word_embedding_size   = 300

          val readingSlot1 = NewArray[Int](1) // this is a slot of memory used for reading from file
          val fp = openf("small_glove.txt", "r")
          getInt(fp, readingSlot1, 0) // read the first number in the file, which is "How many rows"
          val word_embedding_length = readingSlot1(0)

          val word_embedding_data = NewArray[Array[Double]](word_embedding_length)
          
          for (i <- (0 until word_embedding_length): Rep[Range]) {
            word_embedding_data(i) = NewArray[Double](word_embedding_size)
            for (j <- (0 until word_embedding_size): Rep[Range]) getFloat(fp, word_embedding_data(i), j)
          }
          closef(fp)

          // read in the data for trees
          val readingSlot2 = NewArray[Int](1) // need a new readingSlot, other wise have error
          val fp1 = openf("array_tree.txt", "r")
          getInt(fp1, readingSlot2, 0)
          val tree_number = readingSlot2(0) 
          val tree_data = NewArray[Array[Int]](tree_number * 4) // each tree data has 4 lines (score, word, lch, rch)

          val readingSlot3 = NewArray[Int](1) // yet another readingSlot, not sure if this one can be reused
          for (i <- (0 until tree_number): Rep[Range]) {
            getInt(fp1, readingSlot3, 0)
            for (j <- (0 until 4): Rep[Range]) {
              tree_data(i * 4 + j) = NewArray[Int](readingSlot3(0))
              for (k <- (0 until readingSlot3(0)): Rep[Range]) getInt(fp1, tree_data(i * 4 + j), k)
            }
          }
          closef(fp1)

          // set up hyperparameters and parameters
          val hidden_size = 150
          val output_size = 5
          val learning_rate = 0.05

          // parameters for leaf node
          val Wi = Tensor.randinit(hidden_size, word_embedding_size, 0.01)  // from word embedding to hidden vector, input gate
          val bi = Tensor.zeros(hidden_size)                                // bias word embedding to hidden vector, input gate
          val Wo = Tensor.randinit(hidden_size, word_embedding_size, 0.01)  // from word embedding to hidden vector, outout gate
          val bo = Tensor.zeros(hidden_size)                                // bias word embedding to hidden vector, outout gate
          val Wu = Tensor.randinit(hidden_size, word_embedding_size, 0.01)  // from word embedding to hidden vector, cell state
          val bu = Tensor.zeros(hidden_size)                                // bias word embedding to hidden vector, cell state
          // parameters for non-leaf node
          val U0i  = Tensor.randinit(hidden_size, hidden_size, 0.01) // left child, input gate
          val U1i  = Tensor.randinit(hidden_size, hidden_size, 0.01) // right child, input gate
          val bbi  = Tensor.zeros(hidden_size)                       // bias, input gate
          val U00f = Tensor.randinit(hidden_size, hidden_size, 0.01) // left-left forget gate
          val U01f = Tensor.randinit(hidden_size, hidden_size, 0.01) // left-right forget gate
          val U10f = Tensor.randinit(hidden_size, hidden_size, 0.01) // right-left forget gate
          val U11f = Tensor.randinit(hidden_size, hidden_size, 0.01) // right-right forget gate
          val bbf  = Tensor.zeros(hidden_size)                       // bias, forget gate
          val U0o  = Tensor.randinit(hidden_size, hidden_size, 0.01) // left child, output gate
          val U1o  = Tensor.randinit(hidden_size, hidden_size, 0.01) // right child, output gate
          val bbo  = Tensor.zeros(hidden_size)                       // bias, output gate
          val U0u  = Tensor.randinit(hidden_size, hidden_size, 0.01) // left child, cell state
          val U1u  = Tensor.randinit(hidden_size, hidden_size, 0.01) // right child, cell state
          val bbu  = Tensor.zeros(hidden_size)                       // bias, cell state
          // parameters for softmax
          val Why = Tensor.randinit(output_size, hidden_size, 0.01)         // from hidden vector to output
          val by  = Tensor.zeros(output_size)                               // bias hidden vector to output

          // Cast Tensors as Tensors
          val tWi = TensorR(Wi)
          val tbi = TensorR(bi)
          val tWo = TensorR(Wo)
          val tbo = TensorR(bo)
          val tWu = TensorR(Wu)
          val tbu = TensorR(bu)
          // Cast Tensors as Tensors
          val tU0i  = TensorR(U0i)
          val tU1i  = TensorR(U1i)
          val tbbi  = TensorR(bbi)
          val tU00f = TensorR(U00f)
          val tU01f = TensorR(U01f)
          val tU10f = TensorR(U10f)
          val tU11f = TensorR(U11f)
          val tbbf = TensorR(bbf)
          val tU0o = TensorR(U0o)
          val tU1o = TensorR(U1o)
          val tbbo = TensorR(bbo)
          val tU0u = TensorR(U0u)
          val tU1u = TensorR(U1u)
          val tbbu = TensorR(bbu)
          // Cast Tensors as Tensors
          val tWhy = TensorR(Why)
          val tby  = TensorR(by)

          val dummy_word_embedding = TensorR(Tensor.zeros(word_embedding_size))
          val dummy_forget_gate    = TensorR(Tensor.zeros(hidden_size))

          def lossFun(scores: Rep[Array[Int]], words: Rep[Array[Int]], lchs: Rep[Array[Int]], rchs: Rep[Array[Int]]) = { (dummy: TensorR) =>

            val initial_loss = TensorR(Tensor.zeros(1))
            val initial_hidd = TensorR(Tensor.zeros(hidden_size))
            val initial_cell = TensorR(Tensor.zeros(hidden_size))
            val inBuffer     = new ArrayBuffer[TensorR]()
            inBuffer.append(initial_loss); inBuffer.append(initial_hidd); inBuffer.append(initial_cell)

            val outBuffer = LOOPTM(inBuffer)(lchs, rchs) { (l: ArrayBuffer[TensorR], r: ArrayBuffer[TensorR], i: Rep[Int]) =>

              val lossl = l(0); val hiddenl = l(1); val celll = l(2)
              val lossr = r(0); val hiddenr = r(1); val cellr = r(2)

              val targ = Tensor.zeros(output_size); targ.data(scores(i)) = 1; val targ1 = TensorR(targ)

              val embedding_tensor = IF (word_embedding_size) (lchs(i) < 0) {
                TensorR(Tensor(word_embedding_data(words(i)), word_embedding_size))
              } {dummy_word_embedding}

              val i_gate = IF (hidden_size) (lchs(i) < 0) {
              (tWi.dot(embedding_tensor) + tbi).sigmoid()
              } {
                (tU0i.dot(hiddenl) + tU1i.dot(hiddenr) + tbbi).sigmoid()
              }

              val fl_gate = IF (hidden_size) (lchs(i) < 0) {
                dummy_forget_gate
              } {
                (tU00f.dot(hiddenl) + tU01f.dot(hiddenr) + tbbf).sigmoid()
              }

              val fr_gate = IF (hidden_size) (lchs(i) < 0) {
                dummy_forget_gate
              } {
                (tU10f.dot(hiddenl) + tU11f.dot(hiddenr) + tbbf).sigmoid()
              }

              val o_gate = IF (hidden_size) (lchs(i) < 0) {
                (tWo.dot(embedding_tensor) + tbo).sigmoid()
              } {
                (tU0o.dot(hiddenl) + tU1o.dot(hiddenr) + tbbo).sigmoid()
              }

              val u_value = IF (hidden_size) (lchs(i) < 0) {
                (tWu.dot(embedding_tensor) + tbu).tanh()
              } {
                (tU0u.dot(hiddenl) + tU1u.dot(hiddenr) + tbbu).tanh()
              }

              val cell = IF (hidden_size) (lchs(i) < 0) {
                i_gate * u_value
              } {
                i_gate * u_value + fl_gate * celll + fr_gate * cellr
              }

              val hidden = o_gate * cell.tanh()

              val pred1 = (tWhy.dot(hidden) + tby).exp()
              val pred2 = pred1 / pred1.sum()
              val loss = lossl + lossr - (pred2 dot targ1).log()

              val out = ArrayBuffer[TensorR]()
              out.append(loss)
              out.append(hidden)
              out.append(cell)
              out
            }
            outBuffer(0)
          }

          val lr = learning_rate
          val hp = 1e-8

          // parameters for leaf node
          val mWi = Tensor.zeros_like(Wi)
          val mbi = Tensor.zeros_like(bi)
          val mWo = Tensor.zeros_like(Wo)
          val mbo = Tensor.zeros_like(bo)
          val mWu = Tensor.zeros_like(Wu)
          val mbu = Tensor.zeros_like(bu)
          // parameters for non-leaf node
          val mU0i  = Tensor.zeros_like(U0i)
          val mU1i  = Tensor.zeros_like(U1i)
          val mbbi  = Tensor.zeros_like(bbi)
          val mU00f = Tensor.zeros_like(U00f)
          val mU01f = Tensor.zeros_like(U01f)
          val mU10f = Tensor.zeros_like(U10f)
          val mU11f = Tensor.zeros_like(U11f)
          val mbbf  = Tensor.zeros_like(bbf)
          val mU0o  = Tensor.zeros_like(U0o)
          val mU1o  = Tensor.zeros_like(U1o)
          val mbbo  = Tensor.zeros_like(bbo)
          val mU0u  = Tensor.zeros_like(U0u)
          val mU1u  = Tensor.zeros_like(U1u)
          val mbbu  = Tensor.zeros_like(bbu)
          // parameters for softmax
          val mWhy = Tensor.zeros_like(Why)
          val mby  = Tensor.zeros_like(by)

          val loss_save = NewArray[Double](30)

          val addr = getMallocAddr() // remember current allocation pointer here

           val loopStart = get_time()

          for (epoc <- (0 until 30): Rep[Range]) {

            var average_loss = 0.0
            for (n <- (0 until tree_number): Rep[Range]) {

              val index = n % tree_number
              val scores   = tree_data(index * 4)
              val words    = tree_data(index * 4 + 1)
              val leftchs  = tree_data(index * 4 + 2)
              val rightchs = tree_data(index * 4 + 3)
              val loss = gradR_loss(lossFun(scores, words, leftchs, rightchs))(Tensor.zeros(1))
              val loss_value = loss.data(0)  // we suppose the loss is scala (Tensor of size 1)
              average_loss = average_loss * (n) / (n+1) + loss_value / (n+1)

              val pars = ArrayBuffer(tWi, tbi, tWo, tbo, tWu, tbu, tU0i, tU1i, tbbi, tU00f, tU01f, tU10f, tU11f, tbbf, tU0o, tU1o, tbbo, tU0u, tU1u, tbbu, tWhy, tby)
              val mems = ArrayBuffer(mWi, mbi, mWo, mbo, mWu, mbu, mU0i, mU1i, mbbi, mU00f, mU01f, mU10f, mU11f, mbbf, mU0o, mU1o, mbbo, mU0u, mU1u, mbbu, mWhy, mby)
              for ((par, mem) <- pars.zip(mems)) {
                par.clip_grad(5.0)
                mem += par.d * par.d
                par.x -= par.d * lr / (mem + hp).sqrt()
                par.clear_grad()
              }

              resetMallocAddr(addr)  // reset malloc_addr to the value when we remember allocation pointer */
            }

            loss_save(epoc) = average_loss
            val tempTime = get_time()
            printf("epoc %d, average_loss %f, time %lf\\n", epoc, average_loss, (tempTime - loopStart))
            
          }

          val loopEnd = get_time()
          val prepareTime = loopStart - startTime
          val loopTime = loopEnd - loopStart
          val timePerEpoc = loopTime / 30

          val fp2 = openf(a, "w")
          fprintf(fp2, "unit: %s\\n", "1 epoch")
          for (i <- (0 until loss_save.length): Rep[Range]) {
            //printf("loss_saver is %lf \\n", loss_save(i))
            fprintf(fp2, "%lf\\n", loss_save(i))
          }
          fprintf(fp2, "run time: %lf %lf\\n", prepareTime, timePerEpoc)
          closef(fp2)

        }
      }

      //println("run sentiment analysis tree LSTM")
      //val sentit_file = new PrintWriter(new File(root_dir + "evaluationTreeLSTM/Lantern/Lantern.cpp"))
      //sentit_file.println(sentimental_lstm.code)
      //sentit_file.flush()

    if (false) {
      val cnn_test1 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {

          val iPane = 1
          val iRow = 16
          val iCol = 20
          val input = Tensor.ones(iPane, iRow, iCol)
          val kOut = 1
          val kIn = iPane
          val kRow = 3
          val kCol = 3
          val kernel = Tensor.ones(kOut, kIn, kRow, kCol)

          val res = input.conv2D(kernel, 1, 1)
          Tensor.assertEqual(res, Tensor.fill((kRow * kCol * kIn) * 1.0, kOut, iRow - kRow + 1, iCol - kCol + 1), "CNN 1")

          // printf("Result:\\n")
          // res.print3D
        }
      }
      println("start cnn_test1")
      cnn_test1.eval("abc")

      val cnn_test2 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {

          val iPane = 1
          val iRow = 16
          val iCol = 20
          val input = Tensor.ones(iPane, iRow, iCol)
          val kOut = 1
          val kIn = iPane
          val kRow = 3
          val kCol = 3
          val kernel = Tensor.fill((i: NSeq[Rep[Int]]) => if (i(2) == kRow/2 && i(3) == kCol/2) 1.0 else 0.0, kOut, kIn, kRow, kCol)

          // printf("Kernel\\n")
          // kernel.print4D

          val res = input.conv2D(kernel, 1, 1)
          Tensor.assertEqual(res, Tensor.fill(1.0, kOut, iRow - kRow + 1, iCol - kCol + 1), "CNN 2")
          // printf("Result:\\n")
          // res.print3D
        }
      }


      println("start cnn_test2")
      cnn_test2.eval("abc")

      val cnn_test3 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {

          val iPane = 1
          val iRow = 16
          val iCol = 20
          val input = Tensor.ones(iPane, iRow, iCol)
          val kOut = 1
          val kIn = iPane
          val kRow = 3
          val kCol = 3
          val kernel = Tensor.fill((i: NSeq[Rep[Int]]) => if (i(2) == kRow/2 && i(3) == kCol/2) 1.0 else 0.0 ,kOut, kIn, kRow, kCol)

          val res = input.conv2D(kernel, 2, 2)
          Tensor.assertEqual(res, Tensor.fill(1.0, kOut, (iRow - kRow)/2 + 1, (iCol - kCol)/2 + 1), "CNN 3")
          // printf("Result:\\n")
          // res.print3D
        }
      }

      println("start cnn_test3")
      cnn_test3.eval("abc")

      val cnn_back_test1 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {

          val iPane = 1
          val iRow = 16
          val iCol = 20
          val input = Tensor.ones(iPane, iRow, iCol)
          val kOut = 1
          val kIn = iPane
          val kRow = 3
          val kCol = 3
          val kernel = Tensor.ones(kOut, kIn, kRow, kCol)

          val varInput = TensorR(input)
          val varKernel = TensorR(kernel)

          val rS = 1
          val cS = 1

          def lossFun = { (dummy: TensorR) =>
            val res = varInput.conv(varKernel, rS, cS)
            res.sum()
          }

          val loss = gradR_loss(lossFun)(Tensor.zeros(1))

          val resR = (iRow - kRow)/rS + 1
          val resC = (iCol - kCol)/cS + 1

          Tensor.assertEqual(loss, Tensor.scalar(resR * resC * 9.0), "BACK - LOSS")
          // printf("Loss:\\n")
          // loss.printRaw()

          // FIXME complete correct result
          // Tensor.assertEqual(varInput.d, Tensor.fill(
          //   (p: Rep[Int], x: Rep[Int], y: Rep[Int]) =>
          //     if (x >= 2 && x < iRow - 2 && 2 <= y && y < iCol - 2)
          //       9.0
          //     else if (x > ), iPane, iRow, iCol), "BACK 1 - INPUT D")
          // printf("Input gradient:\\n")
          // varInput.d.print3D

          Tensor.assertEqual(varKernel.d, Tensor.fill(resR * resC * 1.0, kIn, kOut, kRow, kCol), "BACK 1 - KERNEL D")
          // printf("Kernel gradient:\\n")
          // varKernel.d.print4D
        }
      }

      println("start cnn_back_test1")
      cnn_back_test1.eval("abc")

      val cnn_back_test2 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {

          val iPane = 1
          val iRow = 16
          val iCol = 20
          val input = Tensor.ones(iPane, iRow, iCol)
          val kOut = 1
          val kIn = iPane
          val kRow = 3
          val kCol = 3
          val kernel = Tensor.fill((i: NSeq[Rep[Int]]) => if (i(2) == kRow/2 && i(3) == kCol/2) 1.0 else 0.0 ,kOut, kIn, kRow, kCol)

          val varInput = TensorR(input)
          val varKernel = TensorR(kernel)

          val rS = 1
          val cS = 1

          def lossFun = { (dummy: TensorR) =>
            val res = varInput.conv(varKernel, rS, cS)
            res.sum()
          }

          val loss = gradR_loss(lossFun)(Tensor.zeros(1))
          // printf("Loss:\\n")
          // loss.printRaw()

          val resR = (iRow - kRow)/rS + 1
          val resC = (iCol - kCol)/cS + 1
          Tensor.assertEqual(loss, Tensor.scalar(resR * resC * 1.0), "BACK 2 - LOSS")

          // FIXME complete correct result
          // Tensor.assertEqual(varInput.d, Tensor.fill(
          //   (p: Rep[Int], x: Rep[Int], y: Rep[Int]) =>
          //     if (x >= 2 && x < iRow - 2 && 2 <= y && y < iCol - 2)
          //       9.0
          //     else if (x > ), iPane, iRow, iCol), "BACK 1 - INPUT D")
          // varInput.d.print3D
          // printf("Input gradient:\\n")
          // varInput.d.print3D

          Tensor.assertEqual(varKernel.d, Tensor.fill(resR * resC * 1.0, kIn, kOut, kRow, kCol), "BACK 2 - KERNEL D")
          // printf("Kernel gradient:\\n")
          // varKernel.d.print4D
        }
      }

      println("start cnn_back_test2")
      cnn_back_test2.eval("abc")

      val cnn_back_test3 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {

          val iPane = 1
          val iRow = 16
          val iCol = 20
          val input = Tensor.ones(iPane, iRow, iCol)
          val kOut = 1
          val kIn = iPane
          val kRow = 3
          val kCol = 3
          val kernel = Tensor.fill((i: NSeq[Rep[Int]]) => if (i(2) == kRow/2 && i(3) == kCol/2) 1.0 else 0.0 ,kOut, kIn, kRow, kCol)

          val varInput = TensorR(input)
          val varKernel = TensorR(kernel)

          val rS = 2
          val cS = 2

          def lossFun = { (dummy: TensorR) =>
            val res = varInput.conv(varKernel, rS, cS)
            res.sum()
          }

          val loss = gradR_loss(lossFun)(Tensor.zeros(1))
          // printf("Loss:\\n")
          // loss.printRaw()

          val resR = (iRow - kRow)/rS + 1
          val resC = (iCol - kCol)/cS + 1
          Tensor.assertEqual(loss, Tensor.scalar(resR * resC * 1.0), "BACK 2 - LOSS")

          // FIXME complete correct result
          // Tensor.assertEqual(varInput.d, Tensor.fill(
          //   (p: Rep[Int], x: Rep[Int], y: Rep[Int]) =>
          //     if (x >= 2 && x < iRow - 2 && 2 <= y && y < iCol - 2)
          //       9.0
          //     else if (x > ), iPane, iRow, iCol), "BACK 1 - INPUT D")
          // varInput.d.print3D
          // printf("Input gradient:\\n")
          // varInput.d.print3D

          Tensor.assertEqual(varKernel.d, Tensor.fill(resR * resC * 1.0, kIn, kOut, kRow, kCol), "BACK 2 - KERNEL D")
          // printf("Kernel gradient:\\n")
          // varKernel.d.print4D
        }
      }

      println("start cnn_back_test3")
      cnn_back_test3.eval("abc")

      val cnn_test4 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {

          val iPane = 3
          val iRow = 16
          val iCol = 20
          val input = Tensor.ones(iPane, iRow, iCol)
          val kOut = 1
          val kIn = iPane
          val kRow = 3
          val kCol = 3
          val kernel = Tensor.ones(kOut, kIn, kRow, kCol)

          val rS = 2
          val cS = 2
          val res = input.conv2D(kernel, rS, cS)
          Tensor.assertEqual(res, Tensor.fill(iPane * kRow * kCol * 1.0, kOut, (iRow - kRow)/rS + 1, (iCol - kCol)/cS + 1), "CNN 4")
          // printf("Result:\\n")
          // res.print3D
        }
      }

      println("start cnn_test4")
      cnn_test4.eval("abc")

      val cnn_back_test4 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {

          val iPane = 3
          val iRow = 16
          val iCol = 20
          val input = Tensor.ones(iPane, iRow, iCol)
          val kOut = 1
          val kIn = iPane
          val kRow = 3
          val kCol = 3
          val kernel = Tensor.ones(kOut, kIn, kRow, kCol)

          val varInput = TensorR(input)
          val varKernel = TensorR(kernel)

          val rS = 2
          val cS = 2

          def lossFun = { (dummy: TensorR) =>
            val res = varInput.conv(varKernel, rS, cS)
            res.sum()
          }

          val loss = gradR_loss(lossFun)(Tensor.zeros(1))
          // printf("Loss:\\n")
          // loss.printRaw()

          val resR = (iRow - kRow)/rS + 1
          val resC = (iCol - kCol)/cS + 1
          Tensor.assertEqual(loss, Tensor.scalar(kOut * resR * resC * 27.0), "BACK 4 - LOSS")

          // FIXME complete correct result
          // Tensor.assertEqual(varInput.d, Tensor.fill(
          //   (p: Rep[Int], x: Rep[Int], y: Rep[Int]) =>
          //     if (x >= 2 && x < iRow - 2 && 2 <= y && y < iCol - 2)
          //       9.0
          //     else if (x > ), iPane, iRow, iCol), "BACK 1 - INPUT D")
          // varInput.d.print3D
          // printf("Input gradient:\\n")
          // varInput.d.print3D

          Tensor.assertEqual(varKernel.d, Tensor.fill(resR * resC * 1.0, kOut, kIn, kRow, kCol), "BACK 4 - KERNEL D")
          // printf("Kernel gradient:\\n")
          // varKernel.d.print4D
        }
      }

      println("start cnn_back_test4")
      cnn_back_test4.eval("abc")


      val cnn_test5 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {

          val iPane = 3
          val iRow = 16
          val iCol = 20
          val input = Tensor.ones(iPane, iRow, iCol)
          val kOut = 4
          val kIn = iPane
          val kRow = 3
          val kCol = 3
          val kernel = Tensor.ones(kOut, kIn, kRow, kCol)

          val rS = 2
          val cS = 2
          val res = input.conv2D(kernel, rS, cS)
          Tensor.assertEqual(res, Tensor.fill(iPane * kRow * kCol * 1.0, kOut, (iRow - kRow)/rS + 1, (iCol - kCol)/cS + 1), "CNN 4")
          // printf("Result:\\n")
          // res.print3D
        }
      }

      println("start cnn_test5")
      cnn_test5.eval("abc")

      val cnn_back_test5 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {

          val iPane = 3
          val iRow = 16
          val iCol = 20
          val input = Tensor.ones(iPane, iRow, iCol)
          val kOut = 4
          val kIn = iPane
          val kRow = 3
          val kCol = 3
          val kernel = Tensor.fill((i: NSeq[Rep[Int]]) => if (i(2) == kRow/2 && i(3) == kCol/2) 1.0 else 0.0 ,kOut, kIn, kRow, kCol)

          val varInput = TensorR(input)
          val varKernel = TensorR(kernel)

          val rS = 2
          val cS = 2

          def lossFun = { (dummy: TensorR) =>
            val res = varInput.conv(varKernel, rS, cS)
            res.sum()
          }

          val loss = gradR_loss(lossFun)(Tensor.zeros(1))
          // printf("Loss:\\n")
          // loss.printRaw()

          val resR = (iRow - kRow)/rS + 1
          val resC = (iCol - kCol)/cS + 1
          Tensor.assertEqual(loss, Tensor.scalar(kOut * resR * resC * kIn * 1.0), "BACK 5 - LOSS")

          // FIXME complete correct result
          // Tensor.assertEqual(varInput.d, Tensor.fill(
          //   (p: Rep[Int], x: Rep[Int], y: Rep[Int]) =>
          //     if (x >= 2 && x < iRow - 2 && 2 <= y && y < iCol - 2)
          //       9.0
          //     else if (x > ), iPane, iRow, iCol), "BACK 1 - INPUT D")
          // varInput.d.print3D
          // printf("Input gradient:\\n")
          // varInput.d.print3D

          Tensor.assertEqual(varKernel.d, Tensor.fill(resR * resC * 1.0, kOut, kIn, kRow, kCol), "BACK 5 - KERNEL D")
          // printf("Kernel gradient:\\n")
          // varKernel.d.print4D
        }
      }

      println("start cnn_back_test5")
      cnn_back_test5.eval("abc")

      val maxpool_test1 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
          val iPane = 2
          val iRow = 8
          val iCol = 10
          val input = Tensor.ones(iPane, iRow, iCol)

          val sR = 2
          val sC = 2
          val (res, idx) = input.maxPool(sR, sC)

          Tensor.assertEqual(res, Tensor.ones(iPane, iRow/sR, iCol/sC), "MAXPOOL 1")
          for (i <- 0 until res.nbElem: Rep[Range]) {
            // assertC(idx(i) ==  (i / res.strides(2)) * sR * input.strides(2) + sC * (i % res.strides(2)), s"Maxpool index invalid %d != %d (%d - %d)\\n", idx(i), (i / res.strides(2)) * sR * input.strides(2) + sC * (i % res.strides(2)), i / res.strides(2), i % res.strides(2))
          }

        }

      }

      println("start maxpool test1")
      maxpool_test1.eval("abc")

      val maxpool_back_test1 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
          val iPane = 2
          val iRow = 8
          val iCol = 10
          val input = Tensor.ones(iPane, iRow, iCol)

          val sR = 2
          val sC = 2

          val varInput = TensorR(input)

          def lossFun = { (dummy: TensorR) =>
            val res = varInput.maxPool(sR, sC)
            res.sum()
          }

          val loss = gradR_loss(lossFun)(Tensor.zeros(1))

          Tensor.assertEqual(loss, Tensor.scalar(iPane * (iRow/sR) * (iCol/sC) * 1.0), "MAXPOOL BACK 1 - LOSS")
          Tensor.assertEqual(varInput.d, Tensor.fill((i: NSeq[Rep[Int]]) => if (i(1) % sR == 0 && i(2) % sC == 0) 1.0 else 0.0, iPane, iRow, iCol), "MAXPOOL BACK 1 - D")
          // printf("Input derivative:\\n")
          // varInput.d.print3D
        }

      }

      println("start maxpool back test1")
      maxpool_back_test1.eval("abc")

      val dropout_test1 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
          val iPane = 2
          val iRow = 16
          val iCol = 20
          val input = Tensor.rand(iPane, iRow, iCol)

          val (resAll, idxAll) = input.dropout(0.0)
          val (resNone, idxNone) = input.dropout(1.0)

          Tensor.assertEqual(resAll, input, "DROPOUT 1")
          Tensor.assertEqual(resNone, Tensor.zeros(input), "DROPOUT 2")

          for (i <- 0 until input.nbElem: Rep[Range]) {
            assertC(idxAll(i) == 1.0, "idxAll incorrect %.3f != 1\\n", idxAll(i))
            assertC(idxNone(i) == 0.0, "idxNone incorrect %.3f != 0\\n", idxNone(i))
          }
        }
      }

      println("start dropout test1")
      dropout_test1.eval("abc")

      val dropout_back_test1 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
          val iPane = 2
          val iRow = 16
          val iCol = 20
          val input = Tensor.rand(iPane, iRow, iCol)

          val varInput = TensorR(input)

          def lossFun = { (dummy: TensorR) =>
            val res = varInput.dropout(0.0)
            res.sum()
          }

          val loss = gradR_loss(lossFun)(Tensor.zeros(1))
          Tensor.assertEqual(varInput.d, Tensor.ones(input), "DROPOUT BACK 1 - D")
          // printf("Input derivative:\\n")
          // varInput.d.print3D

        }
      }

      println("start dropout back test1")
      dropout_back_test1.eval("abc")

      val dropout_back_test2 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
          val iPane = 2
          val iRow = 16
          val iCol = 20
          val input = Tensor.rand(iPane, iRow, iCol)

          val varInput = TensorR(input)

          def lossFun = { (dummy: TensorR) =>
            val res = varInput.dropout(1.0)
            res.sum()
          }

          val loss = gradR_loss(lossFun)(Tensor.zeros(1))
          Tensor.assertEqual(varInput.d, Tensor.zeros(input), "DROPOUT BACK 1 - D")
          // printf("Input derivative:\\n")
          // varInput.d.print3D

        }
      }


      println("start dropout back test2")
      dropout_back_test2.eval("abc")

      val test_cnn_full1 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

        // FIXME: add proper check for result. see adworkplace/pytorch/cnn_test.py
        def snippet(a: Rep[String]): Rep[Unit] = {

          Random.srand(Some(1000))

          val iChan1 = 2
          val iRow1 = 10
          val iCol1 = 10

          val input = Tensor.rand(1.0, iChan1, iRow1, iCol1)

          // Layer 1
          val inChan1 = iChan1
          val outChan1 = 2
          val kRow1 = 3
          val kCol1 = 3

          // stride conv
          val sRow1 = 1
          val sCol1 = 1

          // stride maxpool
          val smRow1 = 2
          val smCol1 = 2

          val conv1 = Tensor.rand(1.0, outChan1, inChan1, kRow1, kCol1)
          val oRow1 = convSize(iRow1, kRow1, sRow1)/smRow1
          val oCol1 = convSize(iCol1, kCol1, sCol1)/smCol1

          val inChan2 = outChan1
          val outChan2 = 3

          val conv2 = Tensor.rand(1.0, outChan2, inChan2, kRow1, kCol1)

          val oRow2 = convSize(oRow1, kRow1, sRow1)
          val oCol2 = convSize(oCol1, kCol1, sCol1)
          val out3 = 5
          val in3 = outChan2 * oRow2 * oCol2

          val a1 = Tensor.rand(1.0, out3, in3)
          val b1 = Tensor.rand(1.0, out3)


          val varInput = TensorR(input)
          val varConv1 = TensorR(conv1)
          val varConv2 = TensorR(conv2)
          val varA1 = TensorR(a1)
          val varB1 = TensorR(b1)

          def lossFun = { (dummy: TensorR) =>
            varInput.print("Input")
            val resConv = varInput.conv(varConv1, sRow1, sCol1)
            resConv.print("First conv")
            val resMax = resConv.maxPool(smRow1, smCol1)
            resMax.print("MaxPool")
            val resRL = resMax.relu()
            resRL.print("ReLu 2")
            val resConv2 = resRL.conv(varConv2, sRow1, sCol1)
            resConv2.print("Second conv")
            val resRL2 = resConv2.relu()
            resRL2.print("ReLu 2")
            val resMMul = varA1 dot resRL2.resize(in3)
            resMMul.print("Matrix Multiplication")
            val resVAdd = resMMul + varB1
            resVAdd.print("Vector Addition")
            val resLSM = resVAdd.logSoftmax()
            resLSM.print("LogSoftMax")
            resLSM.nllLoss(2)
          }

          for (x <- 0 until 1000: Rep[Range]) {
            val loss = gradR_loss(lossFun)(Tensor.scalar(0.0))
            loss.print("Loss")

            // Update weight
            for ((weight, idx) <- NSeq(varConv1, varConv2, varA1, varB1).zipWithIndex) {
              weight.print(s"Before ${idx + 1}", derivative = true)
              weight.x.addMul(-0.5, weight.d)
              weight.print(s"After ${idx + 1}")
              weight.clear_grad()
              printf("\\n")
            }
          }
        }
      }
      println("start full CNN test")
      test_cnn_full1.eval("abc")
    
    } // if false 2 closing

    val mnist  = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

      // From the MNIST pytorch example
      val mean = 0.1307
      val std = 0.3081


      class DataLoader(name: String, train: Boolean, dims: Int*) {

        def open(path: Rep[String]) = uncheckedPure[Int]("open(",path,",0)")
        def filelen(fd: Rep[Int]) = uncheckedPure[Long]("fsize(",fd,")") // FIXME: fresh name
        def mmap[T:Typ](fd: Rep[Int], len: Rep[Long]) = uncheckedPure[Array[T]]("(",codegen.remap(typ[T]),"*)mmap(0, ",len,", PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, ",fd,", 0)")

        val fd = open(s"../data/bin/${name}_${if (train) "train" else "test"}.bin")
        val len = filelen(fd)
        val data = mmap[Double](fd, len)
        val dLength = (len/8L).toInt

        val tfd = open(s"../data/bin/${name}_${if (train) "train" else "test"}_target.bin")
        val tlen = filelen(tfd)
        val target = mmap[Int](tfd, tlen)
        val length = (tlen/4L).toInt

        @virtualize
        def normalize() = {
          this.foreach { (t, d) =>
            t.normalize(mean, std, inPlace = true)
          }
        }


        @virtualize
        def foreach(f: (Tensor, Rep[Int]) => Unit) = {
          var off = var_new(0)
          for (img <- 0 until length: Rep[Range]) {
            val dataPtr = slice(data, off)
            val t = Tensor(dataPtr, dims : _*)
            f(t, target(img))
            off += t.nbElem
          }
          assertC(off == dLength, "Data length doesn't match\\n")
        }
      }

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        printf("Here we go!! Go MNIST!!!!\\n")
        Random.srand(Some(42))

        val variables = ArrayBuffer[TensorR]()

        // input size
        val iChan1 = 1
        val iRow1 = 28
        val iCol1 = 28

        System.out.println(s"Input size: $iChan1 x $iRow1 x $iCol1")

        // TODO create modules
        // Layer 1
        val inChan1 = iChan1
        val outChan1 = 10
        val kRow1 = 5
        val kCol1 = 5

        // stride conv
        val sRow1 = 1
        val sCol1 = 1

        // stride maxpool
        val smRow1 = 2
        val smCol1 = 2

        // FIXME scale based on PyTorch
        val conv1 = Tensor.rand(0.2, outChan1, inChan1, kRow1, kCol1)
        val varConv1 = TensorR(conv1)
        variables += varConv1

        // input size
        val iChan2 = outChan1
        val iRow2 = convSize(iRow1, kRow1, sRow1)/smRow1
        val iCol2 = convSize(iCol1, kCol1, sCol1)/smCol1

        System.out.println(s"Layer 1 output size: $iChan2 x $iRow2 x $iCol2")

        // Layer 2
        val inChan2 = outChan1
        val outChan2 = 20
        val kRow2 = 5
        val kCol2 = 5

        // stride conv
        val sRow2 = 1
        val sCol2 = 1

        // stride maxpool
        val smRow2 = 2
        val smCol2 = 2

        val conv2 = Tensor.rand(0.063, outChan2, inChan2, kRow2, kCol2)
        val varConv2 = TensorR(conv2)
        variables += varConv2

        // Layer 3
        val oRow2 = convSize(iRow2, kRow2, sRow2)/smRow2
        val oCol2 = convSize(iCol2, kCol2, sCol2)/smCol2
        val in3 = 320
        val out3 = 50

        System.out.println(s"Layer 2 output size: $outChan2 x $oRow2 x $oCol2")

        assert(in3 == outChan2 * oRow2 * oCol2, s"The input of the first Linear layer should be $in3, got ${outChan2 * oRow2 * oCol2}")

        val a1 = Tensor.rand(0.055, out3, in3)
        val b1 = Tensor.rand(0.055, out3)
        val varA1 = TensorR(a1)
        val varB1 = TensorR(b1)
        variables += varA1
        variables += varB1

        // Layer 4
        val in4 = out3
        val out4 = 10

        val a2 = Tensor.rand(0.15, out4, in4)
        val b2 = Tensor.rand(0.05, out4)
        val varA2 = TensorR(a2)
        val varB2 = TensorR(b2)
        variables += varA2
        variables += varB2

        // Training
        val nbEpoch = 10
        val lr = 0.0005
        val mom = 0.0

        val momentum = if (mom > 0.0) variables map(tR => Tensor.zeros(tR.d)) else ArrayBuffer[Tensor]()

        val dataTimer = Timer2()
        dataTimer.startTimer
        val train = new DataLoader("mnist", true, iChan1, iRow1, iCol1)
        printf("Start normalize\\n")
        train.normalize()
        def trainFun(input: TensorR, target: Rep[Int]) = { (dummy: TensorR) =>
          val resL1 = input.conv(varConv1, sRow1, sCol1).maxPool(smRow1, smCol1).relu()
          val resL2 = resL1.conv(varConv2, sRow2, sCol2).maxPool(smRow2, smCol2).relu().dropout(0.5)
          val resL3 = ((varA1 dot resL2.resize(in3)) + varB1).relu()
          val resL4 = (varA2 dot resL3) + varB2
          val res = resL4.logSoftmax()
          res.nllLoss(target)
        }

        val test = new DataLoader("mnist", false, iChan1, iRow1, iCol1)
        test.normalize()
        printf("Data normalized in %ldms\\n", dataTimer.getElapsedTime)


        val addr = getMallocAddr() // remember current allocation pointer here
        for (epoch <- 0 until nbEpoch: Rep[Range]) {

          val trainTimer = Timer2()
          var imgIdx = var_new(0)
          var trainLoss = var_new(0.0)
          printf("Start training epoch %d\\n", epoch + 1)
          trainTimer.startTimer
          train foreach { (input: Tensor, target: Rep[Int]) =>
            imgIdx += 1
            // assertC(0 <= target && target <= 9, "Target should be a number between 0 and 9, got %d\\n", target)

            val inputR = TensorR(input , isInput=true)
            val loss = gradR_loss(trainFun(inputR, target))(Tensor.scalar(0.0))
            trainLoss += loss.data(0)

            // for ((weight, idx) <- variables.zipWithIndex) {
            //   weight.print(s"Variable ${idx + 1}", derivative = true)
            // }

            // Update weights
            for ((weight, idx) <- variables.zipWithIndex) {
              val d = if (mom > 0.0) {
                printf("TBT\\n")
                exit()
                val sMom = momentum(idx)
                sMom.cmulAdd(mom, weight.d)
              } else {
                weight.d
              }

              // printf("Weight before %.10f -", weight.x.data(0))
              weight.x.addMul(-lr, d)
              // if (weight.x.check(5.0)) {
              //   printf("Iteration %d\\n", imgIdx)
              //   weight.print(s"Weight of variable ${idx + 1} diverged!!!", derivative = true)
              //   exit()
              // }
              // printf("%.10f weigth after (%.10f - %.5f)\\n", weight.x.data(0), weight.d.data(0), lr)
              weight.clear_grad()
            }

            // for ((weight, idx) <- variables.zipWithIndex) {
            //   weight.print(s"Variable ${idx + 1}")
            // }

            if (imgIdx %  (train.length / 10) == 0) {
              printf(s"Train epoch %d: [%d/%d (%.0f%%)]\\tAverage Loss: %.6f\\n", epoch, imgIdx, train.length, 100.0 * imgIdx /train.length, trainLoss/imgIdx)
              unchecked[Unit]("fflush(stdout)")
            }
            resetMallocAddr(addr)
          }
          val delta = trainTimer.getElapsedTime
          printf("Training completed in %ldms (%ld ms/images)\\n", delta, delta/train.length)

          def testFun(input: Tensor) = {
            val (resL1, _) = input.conv2D(conv1, sRow1, sCol1).maxPool(smRow1, smCol1)
            val (resL2, _) = resL1.relu().conv2D(conv2, sRow2, sCol2).maxPool(smRow2, smCol2)
            val resL3 = ((a1 dot resL2.relu().resize(in3)) + b1).relu()
            val resL4 = (a2 dot resL3) + b2
            resL4.logSoftmax()
          }

          printf("\\nStart testing:\\n")
          val testTimer = Timer2()
          testTimer.startTimer
          imgIdx = var_new(0)
          var testLoss = var_new(0.0)
          val correct = var_new(0)
          test foreach { (input: Tensor, target: Rep[Int]) =>
            imgIdx += 1
            val res = testFun(input)

            testLoss += res.nllLoss(target).data(0)
            if (res.maxIndex() == target)
              correct += 1
          }
          printf("Test set: Average loss: %.4f, Acurracy: %d/%d (%.0f) in %ldms\\n", testLoss / test.length, correct, test.length, 100.0 * correct / test.length, testTimer.getElapsedTime)
          printf("\\n\\n")
        }
      }

    }
    
    println("run simple CNN test case")
    val cnn_file = new PrintWriter(new File(root_dir + "evaluationCNN/Lantern/Lantern.cpp"))
    cnn_file.println(mnist.code)
    cnn_file.flush()
    
  }
}

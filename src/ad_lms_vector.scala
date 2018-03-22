import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

import scala.collection.mutable.ArrayBuffer

object LMS_vector {

  trait VectorExp extends Dsl {

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
        Matrix has >1 dim1 field and number of values dim0 * dim1
        but the current implementation silently ignore the 2:end columns unless it is dot product
        The idea of thinking Matrix row as dim0 and colume as dim1 is not the common way, but we are going by it for now because
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

    /**
     Add: Scanner class for Input
     Copied from lms-query-tutorial
     **/

    object Encoding {
      val ix_a = 96  // index starts from 1

      def char_to_ix(ch: Rep[Char]): Rep[Int] = ch.AsInstanceOf[Int] - ix_a
      def ix_to_char(ix: Rep[Int]): Rep[Char] = (ix + ix_a).AsInstanceOf[Char]
    }

    class Vector(val data: Rep[Array[Double]], val dim0: Int, val dim1:Int = 1 /*, val dim2: Int*/) extends Serializable {

      def apply(i: Rep[Int]) = data(i)
      def apply(i: Rep[Int], j: Rep[Int]) = data(i + j * dim0) // FIXME the index of matrix is not the normal way

      def max(a: Int, b: Int) = if (a >= b) a else b

      @virtualize
      def clipAt(bound: Double) = {
        for (i <- (0 until dim0 * dim1): Rep[Range]) {
          if (data(i) > bound) data(i) = bound
          if (data(i) < -1.0 * bound) data(i) = -1.0 * bound
        }
      }

      def + (that: Vector) = {
        val dim0M = max(dim0, that.dim0); val dim1M = max(dim1, that.dim1)
        val res = NewArray[Double](dim0M * dim1M)
        if (dim0 == that.dim0 && dim1 == that.dim1) for (i <- (0 until dim0M * dim1M): Rep[Range]) res(i) = data(i) + that.data(i)
        else if (dim0 == 1 && dim1 == 1)            for (i <- (0 until dim0M * dim1M): Rep[Range]) res(i) = data(0) + that.data(i)
        else if (that.dim0 == 1 && that.dim1 == 1)  for (i <- (0 until dim0M * dim1M): Rep[Range]) res(i) = data(i) + that.data(0)
        else throw new IllegalArgumentException("dimensions of vector do not match +!")
        new Vector(res, dim0M, dim1M)
      }

      // this operator updates the values of this, unlike the + operator
      def += (that: Vector) = {
        if (dim0 == that.dim0 && dim1 == that.dim1) for (i <- (0 until dim0 * dim1): Rep[Range]) data(i) += that.data(i)
        else if (that.dim0 == 1 && that.dim1 == 1) for (i <- (0 until dim0 * dim1): Rep[Range]) data(i) += that.data(0) // broadcast
        else if (dim0 == 1 && dim1 == 1) for (i <- (0 until that.dim0 * that.dim1): Rep[Range]) data(0) += that.data(i) // shrink (not sure)
        else throw new IllegalArgumentException("dimensions of vector do not match +=!")
      }

      def - (that: Vector) = {
        val dim0M = max(dim0, that.dim0); val dim1M = max(dim1, that.dim1)
        val res = NewArray[Double](dim0M * dim1M)
        if (dim0 == that.dim0 && dim1 == that.dim1) for (i <- (0 until dim0M * dim1M): Rep[Range]) res(i) = data(i) - that.data(i)
        else if (dim0 == 1 && dim1 == 1)            for (i <- (0 until dim0M * dim1M): Rep[Range]) res(i) = data(0) - that.data(i)
        else if (that.dim0 == 1 && that.dim1 == 1)  for (i <- (0 until dim0M * dim1M): Rep[Range]) res(i) = data(i) - that.data(0)
        else throw new IllegalArgumentException("dimensions of vector do not match -!")
        new Vector(res, dim0M, dim1M)
      }

      // this operator updates the values of this, unlike the - operator
      def -= (that: Vector) = {
        if (dim0 == that.dim0 && dim1 == that.dim1) for (i <- (0 until dim0 * dim1): Rep[Range]) data(i) -= that.data(i)
        else if (that.dim0 == 1 && that.dim1 == 1) for (i <- (0 until dim0 * dim1): Rep[Range]) data(i) -= that.data(0) // broadcast
        else if (dim0 == 1 && dim1 == 1) for (i <- (0 until that.dim0 * that.dim1): Rep[Range]) data(0) -= that.data(i) // shrink (not sure)
        else throw new IllegalArgumentException("dimensions of vector do not match -=!")
      }

      // element wise multiplication
      def * (that: Vector) = {
        val dim0M = max(dim0, that.dim0); val dim1M = max(dim1, that.dim1)
        val res = NewArray[Double](dim0M * dim1M)
        if (dim0 == that.dim0 && dim1 == that.dim1) for (i <- (0 until dim0M * dim1M): Rep[Range]) res(i) = data(i) * that.data(i)
        else if (dim0 == 1 && dim1 == 1)            for (i <- (0 until dim0M * dim1M): Rep[Range]) res(i) = data(0) * that.data(i)
        else if (that.dim0 == 1 && that.dim1 == 1)  for (i <- (0 until dim0M * dim1M): Rep[Range]) res(i) = data(i) * that.data(0)
        else throw new IllegalArgumentException("dimensions of vector do not match *!")
        new Vector(res, dim0M, dim1M)
      }

      // this operator updates the values of this, unlike * operator
      def *= (that: Vector) = {
        if (dim0 == that.dim0 && dim1 == that.dim1) for (i <- (0 until dim0 * dim1): Rep[Range]) data(i) *= that.data(i)
        else if (that.dim0 == 1 && that.dim1 == 1) for (i <- (0 until dim0 * dim1): Rep[Range]) data(i) *= that.data(0) // broadcast
        else if (dim0 == 1 && dim1 == 1) for (i <- (0 until that.dim0 * that.dim1): Rep[Range]) data(0) *= that.data(i) // shrink (not sure)
        else throw new IllegalArgumentException("dimensions of vector do not match -=!")
      }

      // element wise division
      def / (that: Vector) = {
        val dim0M = max(dim0, that.dim0); val dim1M = max(dim1, that.dim1)
        val res = NewArray[Double](dim0M * dim1M)
        if (dim0 == that.dim0 && dim1 == that.dim1) for (i <- (0 until dim0M * dim1M): Rep[Range]) res(i) = data(i) / that.data(i)
        else if (dim0 == 1 && dim1 == 1)            for (i <- (0 until dim0M * dim1M): Rep[Range]) res(i) = data(0) / that.data(i)
        else if (that.dim0 == 1 && that.dim1 == 1)  for (i <- (0 until dim0M * dim1M): Rep[Range]) res(i) = data(i) / that.data(0)
        else throw new IllegalArgumentException("dimensions of vector do not match /!")
        new Vector(res, dim0M, dim1M)
      }

      // this operator updates the values of this, unlike / operator
      def /= (that: Vector) = {
        if (dim0 == that.dim0 && dim1 == that.dim1) for (i <- (0 until dim0 * dim1): Rep[Range]) data(i) /= that.data(i)
        else if (that.dim0 == 1 && that.dim1 == 1) for (i <- (0 until dim0 * dim1): Rep[Range]) data(i) /= that.data(0) // broadcast
        else if (dim0 == 1 && dim1 == 1) for (i <- (0 until that.dim0 * that.dim1): Rep[Range]) data(0) /= that.data(i) // shrink (not sure)
        else throw new IllegalArgumentException("dimensions of vector do not match -=!")
      }

      def setAsOne() = {
        for (i <- (0 until dim0 * dim1): Rep[Range]) data(i) = 1.0
      }

      def clear() = {
        for (i <- (0 until dim0 * dim1): Rep[Range]) data(i) = 0.0
      }

      def copy_data(that: Vector) = {
        if (dim0 == that.dim0 && dim1 == that.dim1) for (i <- (0 until dim0 * dim1): Rep[Range]) data(i) = that.data(i)
        else throw new IllegalArgumentException("dimensions of vector do not match copy_data!")
      }

      // NOTE: only handles (Matrix dot Vector) and (Vector dot Vector)
      def dot(that: Vector) = {
        // assert that and this have the same dimension
        if (dim0 != that.dim0) throw new IllegalArgumentException("dimensions of vector do not match dot!")
        val res = NewArray[Double](dim1)
        for (j <- (0 until dim1): Rep[Range]) {
          val value = var_new(0.0)
          for (i <- (0 until dim0): Rep[Range]) value += data(i + dim0 * j) * that.data(i)
          res(j) = readVar(value)
        }
        new Vector(res, dim1)
      }

      // NOTE: only handles (Vector cart Vector)
      def cart(that: Vector) = {
        if (dim1 != 1 || that.dim1 != 1) throw new IllegalArgumentException("cartesian product is only for 1d vectors")
        val res = NewArray[Double](dim0 * that.dim0)
        for (i <- (0 until dim0): Rep[Range]) {
          for (j <- (0 until that.dim0): Rep[Range]) {
            res(i * that.dim0 + j) = data(i) * that.data(j)
          }
        }
        new Vector(res, that.dim0, dim0)
      }

      def trans() = {
        if (dim1 == 1) throw new IllegalArgumentException("transpose is only for matrix. Vector transpose is not supported here")
        val res = NewArray[Double](dim0 * dim1)
        for (i <- (0 until dim0): Rep[Range]) {
          for (j <- (0 until dim1): Rep[Range]) {
            res(i * dim1 + j) = data(j * dim0 + i)
          }
        }
        new Vector(res, dim1, dim0)
      }

      def tanh() = {
        val res = NewArray[Double](dim0 * dim1)
        for (i <- (0 until dim0 * dim1): Rep[Range]) res(i) = Math.tanh(data(i))
        new Vector(res, dim0, dim1)
      }

      def exp() = {
        val res = NewArray[Double](dim0 * dim1)
        for (i <- (0 until dim0 * dim1): Rep[Range]) res(i) = Math.exp(data(i))
        new Vector(res, dim0, dim1)
      }

      def log() = {
        val res = NewArray[Double](dim0 * dim1)
        for (i <- (0 until dim0 * dim1): Rep[Range]) res(i) = Math.log(data(i))
        new Vector(res, dim0, dim1)
      }

      def sqrt() = {
        val res = NewArray[Double](dim0 * dim1)
        for (i <- (0 until dim0 * dim1): Rep[Range]) res(i) = Math.sqrt(data(i))
        new Vector(res, dim0, dim1)
      }

      def sigmoid() = {
        val res = NewArray[Double](dim0 * dim1)
        for (i <- (0 until dim0 * dim1): Rep[Range]) res(i) = 1.0 / (Math.exp(-1.0 * data(i)) + 1.0)
        new Vector(res, dim0, dim1)
      }

      // NOTE: sum all elements
      def sum() = {
        val value = var_new(0.0)
        for (i <- (0 until dim0 * dim1): Rep[Range]) value += data(i)
        val res = NewArray[Double](1)
        res(0) = readVar(value)
        new Vector(res, 1)
      }

      // NOTE: sum matrix to vector, condense on the dim1 dimension
      def sumOnDim1() = {
        if (dim1 == 1) this
        else {
          val res = NewArray[Double](dim0)
          for (i <- (0 until dim0): Rep[Range]) {
            val temp = var_new(0.0)
            for (j <- (0 until dim1): Rep[Range]) {
              temp += data(i + j * dim0)
            }
            res(i) = readVar(temp)
          }
          new Vector(res, dim0)
        }
      }

      def print() = {
        for (j <- (0 until dim1): Rep[Range]) {
          for (i <- (0 until dim0): Rep[Range]) {
            println(data(i + j * dim0))
          }
          println(" ")
        }
      }

      // setting: this is matrix, that is dim0-sized vector, y is dim1-sized vector
      // the result is to update this so that this += that * y, where * is cartesian product
      def add_cartesian(that: Vector, y: Vector) = {
        for (i <- (0 until dim1): Rep[Range]) {
          for (j <- (0 until dim0): Rep[Range]) {
            val ind = dim0 * i + j
            data(ind) += that.data(j) * y.data(i)
          }
        }
      }
      // FIXME: Maybe try to support slicing??
      // FIXME: Maybe add support for reshaping??
      // FIXME: Maybe support transposing??


      // setting: this is dim0-sized vector, that is matrix (dim0 * dim1), y is dim1-sized vector
      // the result is to update this so that this accumulate every matrix col * y
      def add_composion(that: Vector, y: Vector) = {
        for (i <- (0 until that.dim1): Rep[Range]) {
          for (j <- (0 until dim0): Rep[Range]) {
            data(j) += that.data(dim0 * i + j) * y.data(i)
          }
        }
      }

      // private function to get data with default to the only element
      def getAt(i: Rep[Int]) = {
        if (dim0 == 1 && dim1 == 1) data(0)
        else data(i)
      }
      def square(t: Rep[Double]) = t * t
      def add_mult(a: Vector, b: Vector) = {
        if (Vector.dimCompetible(a, b) && Vector.dimCompetible(a, this) && Vector.dimCompetible(this, b)) {
          val dim0M = max(dim0, max(a.dim0, b.dim0))
          val dim1M = max(dim1, max(a.dim1, b.dim1))
          if (dim0 == 1 && dim1 == 1) {
            for (i <- (0 until dim0M * dim1M): Rep[Range]) data(0) = data(0) + a.getAt(i) * b.getAt(i)
          } else {
            for (i <- (0 until dim0M * dim1M): Rep[Range]) data(i) = data(i) + a.getAt(i) * b.getAt(i)
          }
        } else throw new IllegalArgumentException("dim not Competible in add_mult")
      }

      def add_div(a: Vector, b: Vector) = {
        if (Vector.dimCompetible(a, b) && Vector.dimCompetible(a, this) && Vector.dimCompetible(this, b)) {
          val dim0M = max(dim0, max(a.dim0, b.dim0))
          val dim1M = max(dim1, max(a.dim1, b.dim1))
          if (dim0 == 1 && dim1 == 1) {
            for (i <- (0 until dim0M * dim1M): Rep[Range]) data(0) = data(0) + a.getAt(i) / b.getAt(i)
          } else {
            for (i <- (0 until dim0M * dim1M): Rep[Range]) data(i) = data(i) + a.getAt(i) / b.getAt(i)
          }
        } else throw new IllegalArgumentException("dim not Competible in add_div")
      }

      def minus_mult_div_square(a: Vector, b: Vector, c: Vector) = {
        if (Vector.dimCompetible(a, b)    && Vector.dimCompetible(a, c)    && Vector.dimCompetible(c, b)    &&
          Vector.dimCompetible(this, b) && Vector.dimCompetible(a, this) && Vector.dimCompetible(this, c)) {
            val dim0M = max(dim0, max(a.dim0, max(b.dim0, c.dim0)))
            val dim1M = max(dim1, max(a.dim1, max(b.dim1, c.dim1)))
            if (dim0 == 1 && dim1 == 1) {
              for (i <- (0 until dim0M * dim1M): Rep[Range]) data(0) = data(0) - a.getAt(i) * b.getAt(i) / square(c.getAt(i))
            } else {
              for (i <- (0 until dim0M * dim1M): Rep[Range]) data(i) = data(i) - a.getAt(i) * b.getAt(i) / square(c.getAt(i))
            }
          } else throw new IllegalArgumentException("dim not competible in minus_mult_div_square")
      }

      def add_oneMinusSquare_mult(a: Vector, b: Vector) = {
        if (Vector.dimCompetible(a, b) && Vector.dimCompetible(a, this) && Vector.dimCompetible(this, b)) {
          val dim0M = max(dim0, max(a.dim0, b.dim0))
          val dim1M = max(dim1, max(a.dim1, b.dim1))
          if (dim0 == 1 && dim1 == 1) {
            for (i <- (0 until dim0M * dim1M): Rep[Range]) data(0) = data(0) + (1.0 - square(a.getAt(i))) * b.getAt(i)
          } else {
            for (i <- (0 until dim0M * dim1M): Rep[Range]) data(i) = data(i) + (1.0 - square(a.getAt(i))) * b.getAt(i)
          }
        } else throw new IllegalArgumentException("dim not Competible in add_oneMinusSquare_mult")
      }
      def oneMinusThenMult(t: Rep[Double]) = (1.0 - t) * t
      def add_oneMinusThenMult_mult(a: Vector, b: Vector) = {
        if (Vector.dimCompetible(a, b) && Vector.dimCompetible(a, this) && Vector.dimCompetible(this, b)) {
          val dim0M = max(dim0, max(a.dim0, b.dim0))
          val dim1M = max(dim1, max(a.dim1, b.dim1))
          if (dim0 == 1 && dim1 == 1) {
            for (i <- (0 until dim0M * dim1M): Rep[Range]) data(0) = data(0) + oneMinusThenMult(a.getAt(i)) * b.getAt(i)
          } else {
            for (i <- (0 until dim0M * dim1M): Rep[Range]) data(i) = data(i) + oneMinusThenMult(a.getAt(i)) * b.getAt(i)
          }
        } else throw new IllegalArgumentException("dim not Competible in add_oneMinusThenMult_mult")
      }
    }

    object Vector {

      def dimCompetible(a: Vector, b: Vector) = {
        (a.dim0 == b.dim0 && a.dim1 == b.dim1) ||
        (a.dim0 == 1 && a.dim1 == 1) ||
        (b.dim0 == 1 && b.dim1 == 1)
      }

      def randinit(dim0: Int, dim1: Int = 1, scale: Double = 1.0, offset: Int = 0) = {
        unchecked[Unit]("srand(time(NULL)" + "+" + offset.toString + ")")
        val res = NewArray[Double](dim0 * dim1)
        for (i <- (0 until dim0 * dim1): Rep[Range]) res(i) = unchecked[Double]("(double)rand()/RAND_MAX*2.0-1.0") * scale
        new Vector(res, dim0, dim1)
      }

      def randn(dim0: Int, dim1: Int = 1, scale: Double = 1.0, offset: Int = 0) = {
        val res = NewArray[Double](dim0 * dim1)
        for (i <- (0 until dim0 * dim1): Rep[Range]) res(i) = unchecked[Double]("d(gen)") * scale
        new Vector(res, dim0, dim1)
      }

      def randPositive(dim0: Int) = {
        val res = NewArray[Double](dim0)
        unchecked[Unit]("srand(time(NULL))")
        for (i <- (0 until dim0): Rep[Range]) res(i) = unchecked[Double]("(double)rand()/RAND_MAX*2.0")
        new Vector(res, dim0)
      }

      def zeros(dim0: Int, dim1: Int = 1) = {
        val res = NewArray[Double](dim0 * dim1)
        for (i <- (0 until dim0 * dim1): Rep[Range]) res(i) = 0.0
        new Vector(res, dim0, dim1)
      }

      def zeros_like(that: Vector) = {
        val res = NewArray[Double](that.dim0 * that.dim1)
        for (i <- (0 until that.dim0 * that.dim1): Rep[Range]) res(i) = 0.0
        new Vector(res, that.dim0, that.dim1)
      }

      def ones(dim0: Int) = {
        val res = NewArray[Double](dim0)
        for (i <- (0 until dim0): Rep[Range]) res(i) = 1.0
        new Vector(res, dim0)
      }

      def halves(dim0: Int) = {
        val res = NewArray[Double](dim0)
        for (i <- (0 until dim0): Rep[Range]) res(i) = 0.5
        new Vector(res, dim0)
      }

      def consts(dim0: Int, dim1: Int = 1, value: Double = 0.001) = {
        val res = NewArray[Double](dim0 * dim1)
        for (i <- (0 until dim0 * dim1): Rep[Range]) res(i) = value
        new Vector(res, dim0, dim1)
      }

      def expand(value: Rep[Double], dim0: Int, dim1: Int = 1) = {
        val res = NewArray[Double](dim0 * dim1)
        for (i <- (0 until dim0 * dim1): Rep[Range]) res(i) = value
        new Vector(res, dim0, dim1)
      }

      def expand(vector: Vector, dim1: Int) = {
        assert (vector.dim1 == 1)
        val res = NewArray[Double](vector.dim0 * dim1)
        for (j <- (0 until dim1): Rep[Range]){
          for (i <- (0 until vector.dim0): Rep[Range]) {
            res(i + j * vector.dim0) = vector.data(i)
          }
        }
        new Vector(res, vector.dim0, dim1)
      }

      def copy(vector: Vector) = {
        val res = NewArray[Double](vector.dim0 * vector.dim1)
        for (i <- (0 until vector.dim0 * vector.dim1): Rep[Range]) res(i) = vector.data(i)
        new Vector(res, vector.dim0, vector.dim1)
      }

      def fromData(x: Double*) = {
        val y = x.toArray
        val res = NewArray[Double](y.length)
        for (i <- (0 until y.length): Rep[Range]) res(i) = y(i)
        new Vector(res, y.length)
      }

      @virtualize
      def assertEqual(a: Vector, b: Vector, mark: String = "", tal: Double = 0.000001) = {
        if (a.dim0 != b.dim0 || a.dim1 != b.dim1) printf("ERROR: %s not equal in dimensions\\n", mark)
        else {
          val mismatch = var_new(0.0)
          for (i <- (0 until a.dim0 * a.dim1): Rep[Range]) {
            val diff = a.data(i) - b.data(i)
            if (diff < -1.0 * tal || diff > tal) mismatch += 1.0
          }
          if (readVar(mismatch) != 0.0) printf("ERROR: %s not equal in some data\\n", mark)
        }
      }
    }


    // Tensor type is the similar to NumR, just replace RDouble with Vector
    // also Vector internally use array, which is mutable by default
    // so both field are val (not var) and can be updated by += -= *= /= setAsOne()
    // all instances of vectors will be shepherded by c++ smart pointers, alleviating the memory leak problem
    type diff = cps[Unit]

    class TensorR(val x: Vector, val d: Vector) extends Serializable {

      def clip_grad(bound: Double) = {
        d.clipAt(bound)
      }

      def + (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
        val y = TensorR.Tensor(x + that.x); k(y)
        this.d += y.d; that.d += y.d
      }

      def - (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
        val y = TensorR.Tensor(x - that.x); k(y)
        this.d += y.d; that.d -= y.d
      }

      // this is element wise multiplication
      def * (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
        val y = TensorR.Tensor(x * that.x); k(y)
        // intermediate Tensors donot need to be substatiated, can optimize!
        //this.d += that.x * y.d; that.d += this.x * y.d;
        this.d.add_mult(that.x, y.d); that.d.add_mult(this.x, y.d)
      }

      // element wise division
      def / (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
        val y = TensorR.Tensor(x / that.x); k(y)
        // intermediate Tensors donot need to be substatiated, can optimize!
        //this.d += y.d / that.x
        this.d.add_div(y.d, that.x)
        //that.d -= this.x * y.d / (that.x * that.x)
        that.d.minus_mult_div_square(this.x, y.d, that.x)
      }

      // vector dot product or Matrix vector dot (viewed as multiple vector dot product) (not the common view)
      def dot(that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
        val y = TensorR.Tensor(x dot that.x); k(y)
        // FIXME: intermediate Tensors donot need to be substatiated, can optimize!
        this.d.add_cartesian(that.x, y.d)
        that.d.add_composion(this.x, y.d)
        // this.d += that.x * y.d // broadcasting
        // that.d += this.x * y.d // broadcasting
      }

      def tanh(): TensorR @diff = shift { (k : TensorR => Unit) =>
        val y = TensorR.Tensor(x.tanh()); k(y)
        // FIXME: intermediate Tensors donot need to be substatiated, can optimize!
        //this.d += (Vector.ones(1) - y.x * y.x) * y.d // broadcasting
        this.d.add_oneMinusSquare_mult(y.x, y.d)
      }

      def exp(): TensorR @diff = shift { (k: TensorR => Unit) =>
        val y = TensorR.Tensor(x.exp()); k(y)
        // Fix
        //this.d += y.x * y.d
        this.d.add_mult(y.x, y.d)
      }

      def log(): TensorR @diff = shift { (k: TensorR => Unit) =>
        val y = TensorR.Tensor(x.log()); k(y)
        // Fix
        //this.d += y.d / x
        this.d.add_div(y.d, x)
      }

      def sigmoid(): TensorR @diff = shift { (k: TensorR => Unit) =>
        val y = TensorR.Tensor(x.sigmoid()); k(y)
        //this.d += (Vector.ones(1) - y.x) * y.x * y.d
        this.d.add_oneMinusThenMult_mult(y.x, y.d)
      }

      def sum(): TensorR @diff = shift { (k: TensorR => Unit) =>
        val y = new TensorR(x.sum(), Vector.zeros(1)); k(y)
        this.d += y.d
      }
      /*
      def free() = {
        unchecked[Unit]("free(",x.data,")")
        unchecked[Unit]("free(",d.data,")")
      } */

     def clear_all() = {
       x.clear()
       d.clear()
     }

     def clear_grad() = {
       d.clear()
     }

    }

    object TensorR {
      def Tensor(a: Vector) = {
        new TensorR(a, Vector.zeros(a.dim0, a.dim1))
      }
    }

    // change fun signature for memory leak issue (no more returning of array, just update the array provided by the caller)
    // this is in accordance of the destination-programming style
    // the fun take array[array[double]] of size 2, with the first array to be the x, and the second array to be the d
    def FUNc(dim0: Int)(f: TensorR => Unit): (TensorR => Unit) = {
      val f1 = fun { (x: Rep[Array[Array[Double]]]) =>
        val tensor = new TensorR(new Vector(x(0), dim0), new Vector(x(1), dim0))
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

    def RST(a: => Unit @diff) = continuations.reset { a; () }

    @virtualize
    def IF(dim0: Int)(c: Rep[Boolean])(a: =>TensorR @diff)(b: =>TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>
      val k1 = FUNc(dim0)(k)

      if (c) RST(k1(a)) else RST(k1(b))
    }

    @virtualize
    def LOOP(init: TensorR)(c: TensorR => Rep[Boolean])(b: TensorR => TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>
      // val k1 = FUN(init.x.dim0)(k)

      lazy val loop: TensorR => Unit = FUNc (init.x.dim0) { (x: TensorR) =>
        if (c(x)) RST(loop(b(x))) else RST(k(x))
      }
      loop(init)
    }

    def FUNs(dim0: Int)(f: Rep[Int] => TensorR => Unit): (Rep[Int] => TensorR => Unit) = {
      val f1 = fun { (xx: Rep[(Int, Array[Array[Double]])]) =>
        val i: Rep[Int]                  = tuple2_get1(xx)
        val x: Rep[Array[Array[Double]]] = tuple2_get2(xx)
        val tensor = new TensorR(new Vector(x(0), dim0), new Vector(x(1), dim0))
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
      lazy val loop: Rep[Int] => TensorR => Unit = FUNs (init.x.dim0) { (i: Rep[Int]) => (x: TensorR) =>
        if (i < c) { RST(loop(i+1)(b(i)(x))) } else RST(k(x))
      }
      loop(0)(init)
    }

    def FUNsm(dim0s: ArrayBuffer[Int])(f: Rep[Int] => ArrayBuffer[TensorR] => Unit): (Rep[Int] => ArrayBuffer[TensorR] => Unit) = {
      val f1 = fun { (xx: Rep[(Int, Array[Array[Double]])]) =>
        val i: Rep[Int]                  = tuple2_get1(xx)
        val x: Rep[Array[Array[Double]]] = tuple2_get2(xx)
        val tensors = ArrayBuffer[TensorR]()
        for (u <- (0 until dim0s.length): Range) {
          tensors.append(new TensorR(new Vector(x(u*2), dim0s(u)), new Vector(x(u*2+1), dim0s(u))))
        }
        f(i)(tensors)
      };
      {
        (i: Rep[Int]) => (x:ArrayBuffer[TensorR]) => {
          val in = NewArray[Array[Double]](2 * dim0s.length)
          for (u <- (0 until dim0s.length): Range) {
            in(u*2) = x(u).x.data; in(u*2+1) = x(u).d.data
          }
          f1((i, in))
        }
      }
    }

    @virtualize
    def LOOPSM(init: ArrayBuffer[TensorR])(c: Rep[Int])(b: Rep[Int] => ArrayBuffer[TensorR] => ArrayBuffer[TensorR] @diff):
    ArrayBuffer[TensorR] @diff = shift { k: (ArrayBuffer[TensorR] => Unit) =>
      lazy val loop: Rep[Int] => ArrayBuffer[TensorR] => Unit = FUNsm (init map (_.x.dim0)) { (i: Rep[Int]) => (x: ArrayBuffer[TensorR]) =>
        if (i < c) { RST(loop(i+1)(b(i)(x))) } else RST(k(x))
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
        t3(new TensorR(new Vector(xx(0), dim0), new Vector(xx(1), dim0)))
      }

      {i: Rep[Int] => k1: (TensorR => Unit) =>
        {
          val k2: Rep[Array[Array[Double]] => Unit] = fun { (x: Rep[Array[Array[Double]]]) =>
            k1(new TensorR(new Vector(x(0), dim0), new Vector(x(1), dim0)))
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
      lazy val loop: Rep[Int] => (TensorR => Unit) => TensorR => Unit = FUNl(init.x.dim0){ (gc: Rep[Int]) => (k: TensorR => Unit) => (x: TensorR) =>
        if (gc < c) { loop(gc+1)((x: TensorR) => RST(k(b(gc)(x))))(x) } else { RST(k(x)) }
      }
      loop(0)(k)(init)
    }

    @virtualize
    def LOOPT(init: TensorR)(lch: Rep[Array[Int]], rch: Rep[Array[Int]])(b: (TensorR, TensorR, Rep[Int]) => TensorR @diff): TensorR @diff = shift {
      k: (TensorR => Unit) =>

        lazy val tree: Rep[Int] => (TensorR => Unit) => TensorR => Unit = FUNl(init.x.dim0){ (i: Rep[Int]) => (k: TensorR => Unit) => (x: TensorR) =>
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
          tensors.append(new TensorR(new Vector(xx(u*2), dim0s(u)), new Vector(xx(u*2+1), dim0s(u))))
        }
        t3(tensors)
      };

      {i: Rep[Int] => k1: (ArrayBuffer[TensorR] => Unit) =>
        {
          val k2: Rep[Array[Array[Double]] => Unit] = fun { (x: Rep[Array[Array[Double]]]) =>
            val tensors = ArrayBuffer[TensorR]()
            for (u <- (0 until length): Range) {
              tensors.append(new TensorR(new Vector(x(u*2), dim0s(u)), new Vector(x(u*2+1), dim0s(u))))
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
      lazy val loop: Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit = FUNlm(init map (_.x.dim0)) {
        (i: Rep[Int]) => (k: ArrayBuffer[TensorR] => Unit) => (x: ArrayBuffer[TensorR]) =>
          if (i < c) { loop(i+1)((x: ArrayBuffer[TensorR]) => RST(k(b(i)(x))))(x) } else { RST(k(x)) }
      }
      loop(0)(k)(init)
    }

    @virtualize
    def LOOPTM(init: ArrayBuffer[TensorR])(lch: Rep[Array[Int]], rch: Rep[Array[Int]])
    (b: (ArrayBuffer[TensorR], ArrayBuffer[TensorR], Rep[Int]) => ArrayBuffer[TensorR] @diff): ArrayBuffer[TensorR] @diff = shift {
      k: (ArrayBuffer[TensorR] => Unit) =>

        lazy val tree: Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit = FUNlm(init.map(_.x.dim0)) {
          (i: Rep[Int]) => (k: ArrayBuffer[TensorR] => Unit) => (x: ArrayBuffer[TensorR]) =>
            if (i >= 0) { tree(lch(i))((l: ArrayBuffer[TensorR]) => tree(rch(i))((r: ArrayBuffer[TensorR]) => RST(k(b(l, r, i))))(x))(x) }
            else { RST(k(x)) }
        }
        tree(0)(k)(init)
    }

    def gradR(f: TensorR => TensorR @diff)(x: Vector): Vector = {
      val x1 = new TensorR(x, Vector.zeros(x.dim0))
      reset { val y = f(x1)
        y.d.setAsOne()
        // y.x.print() // this is the result of forward propagation (likely the loss)
      () }
      x1.d
    }

    // same as gradR function, except that we return the final result of f, not the gradient of input
    // gradient of input is supposed to be dummy value here
    // gradient of useful tensors are in closure, and can be accessed directly from outside of this function
    def gradR_loss(f: TensorR => TensorR @diff)(x: Vector): Vector = {
      val x1 = new TensorR(x, Vector.zeros(x.dim0)) // this should be a dummy tensor
      val result = Vector.zeros(1)                  // this should be the loss
      reset { val y = f(x1)
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
      val array0 = new DslDriverC[String, Unit] with VectorExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
          val addr = getMallocAddr()
          //printf("address is at %ld \\n", addr)
          resetMallocAddr(addr)
          //printf("now lets use some memory\\n")
          val mem = Vector.zeros(100)
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

      val array1 = new DslDriverC[String, Unit]  with VectorExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
          val length = 2
          val res = Vector.randinit(length)
          val res2 = Vector.randinit(length, offset = 5)
          //res.print()
          //res2.print()

          val result = res dot res2
          //result.print()

          // assertions
          if (res(0) * res2(0) + res(1) * res2(1) != result(0))
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

      val array1_1 = new DslDriverC[String, Unit] with VectorExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
          val dim0 = 2
          val dim1 = 3
          val matrix = Vector.randinit(dim0, dim1)
          val vector = Vector.randinit(dim0, offset = 4)
          //matrix.print()
          //vector.print()

          //println("the result is:")
          val result = matrix dot vector
          //result.print()

          if (matrix(0, 0) * vector(0) + matrix(1, 0) * vector(1) != result(0))
            println("ERROR: the matrix vector dot product is wrong on the first element of result")
          if (matrix(0, 1) * vector(0) + matrix(1, 1) * vector(1) != result(1))
            println("ERROR: the matrix vector dot product is wrong on the second element of result")
          if (matrix(0, 2) * vector(0) + matrix(1, 2) * vector(1) != result(2))
            println("ERROR: the matrix vector dot product is wrong on the third element of result")
        }
      }

      //println(array1_1.code)
      println("run test case array1_1")
      array1_1.eval("abc")

      val array2 = new DslDriverC[String, Unit] with VectorExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
          // read training data from file (for now just use random)
          val length = 2
          val v = Vector.randinit(length)
          //v.print()

          // calculate gradient
          val grad = gradR(t => t dot t)(v)
          // show gradient
          //println("show gradient in the traditional way")
          //grad.print()

          // assertions
          Vector.assertEqual(v * Vector.consts(1, value = 2.0), grad)

          // construct TensorR for closure
          val tv = TensorR.Tensor(v)
          val loss = gradR_loss(dummy => tv dot tv)(Vector.zeros(1))
          //println("gradient:")
          //tv.d.print()
          //println("loss")
          //loss.print()
          // assertions
          Vector.assertEqual((v dot v), loss)
          Vector.assertEqual(tv.d, grad)
          ()
        }
      }

      //println("test dot gradient")
      //println(array2.code)
      println("run test case array2")
      array2.eval("2.0")

      val array2_1 = new DslDriverC[String, Unit] with VectorExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {

          val dim0 = 2
          val vector = Vector.randinit(dim0, offset = 4)

          // initialize tensors for closure
          val ve = new TensorR(vector, Vector.zeros(dim0))
          val half = new TensorR(Vector.halves(dim0), Vector.zeros(dim0))

          // define function of model
          def model(dummy: TensorR): TensorR @diff = {
            ((ve dot ve) * half).sum()
          }
          val loss = gradR_loss(model)(Vector.zeros(1))
          Vector.assertEqual(loss, ((vector dot vector) * Vector.halves(dim0)).sum(), "1")
          Vector.assertEqual(ve.d, vector * Vector.consts(1, value = 2.0),"2")
          Vector.assertEqual(half.d, Vector.expand((vector dot vector).data(0), 2), "3")
          ()
        }
      }

      println("run test case array2_1")
      array2_1.eval("abc")


      val array2_2 = new DslDriverC[String, Unit] with VectorExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {

          val dim0 = 2
          val dim1 = 3
          val matrix = Vector.randinit(dim0, dim1)
          val vector = Vector.randinit(dim0, offset = 4)

          // initialize tensors for closure
          val ma = new TensorR(matrix, Vector.zeros(dim0, dim1))
          val ve = new TensorR(vector, Vector.zeros(dim0))

          // define function of model
          def model(dummy: TensorR): TensorR @diff = {
            (ma dot ve).sum()
          }
          val loss = gradR_loss(model)(Vector.zeros(1))
          Vector.assertEqual(loss, (matrix dot vector).sum(), "11")
          Vector.assertEqual(ma.d, Vector.expand(vector, dim1), "12")
          Vector.assertEqual(ve.d, matrix.sumOnDim1(), "13")
          ()
        }
      }

      // println("test matrix vector dot gradient as side effect")
      //println(array2_2.code)
      println("run test case array2_2")
      array2_2.eval("abc")


      val array2_3 = new DslDriverC[String, Unit] with VectorExp {

        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {

          val vocab_size = 3
          val hidden_size = 10
          val Wxh = Vector.randinit(vocab_size, hidden_size, 0.1)  // input to hidden
          val Whh = Vector.randinit(hidden_size, hidden_size, 0.1) // hidden to hidden
          val Why = Vector.randinit(hidden_size, vocab_size, 0.1)  // hidden to output
          val bh  = Vector.randinit(hidden_size)
          val by  = Vector.randinit(vocab_size)
          val hprev = Vector.randinit(hidden_size)

          val hprev_next = Vector.zeros_like(hprev) // this vector catches the new hidden value, see the NOTE below
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
         val Wxh1 = TensorR.Tensor(Wxh)
         val Whh1 = TensorR.Tensor(Whh)
         val Why1 = TensorR.Tensor(Why)
         val bh1  = TensorR.Tensor(bh)
         val by1  = TensorR.Tensor(by)
         val hprev1 = TensorR.Tensor(hprev)

         // encode input and output
         val x_data = NewArray[Int](3); x_data(0) = 0; x_data(1) = 1; x_data(2) = 2
         val y_data = NewArray[Int](3); y_data(0) = 2; y_data(1) = 0; y_data(2) = 1
         //val x_data = mutableStaticData(scala.Array(0, 1, 2))
         //val y_data = mutableStaticData(scala.Array(2, 0, 1))

         // our method of loss and gradient calculation
         def lossFun: (TensorR => TensorR @diff) = { (dummy: TensorR) =>
           val loss = TensorR.Tensor(Vector.zeros(1))
           val in = ArrayBuffer[TensorR]()
           in.append(loss)
           in.append(hprev1)
           val outputs = LOOPSM(in)(3){i => t =>

             // get input as one-hot tensor
             val x = Vector.zeros(vocab_size)
             x.data(x_data(i)) = 1
             val x1 = TensorR.Tensor(x)
             // get output as one-hot tensor
             val y = Vector.zeros(vocab_size)
             y.data(y_data(i)) = 1
             val y1 = TensorR.Tensor(y)

             val h1 = ((Wxh1 dot x1) + (Whh1 dot t(1)) + bh1).tanh() // use hidden state and x1 to compute hidden state
             val e1 = (Why1.dot(h1) + by1).exp()                       // use new hidden state to compute unnormalized prob
             val p1 = e1 / e1.sum()                            // use unnormalized prob to compute normalize prob
             val newloss = t(0) - (p1 dot y1).log()            // loss is updated by original loss t(0) and additional loss
             val out = ArrayBuffer[TensorR]()
             out.append(newloss)
             out.append(h1)
             out
           }
           hprev_next.copy_data(outputs(1).x)  // update the hidden state with the result from LOOP
           outputs(0)                          // return the final loss
         }
         val loss1 = gradR_loss(lossFun)(Vector.zeros(1))


         // correct method of loss and gradient calculation, adapting from Numpy
         // preset space for gradients
         val dWxh = Vector.zeros_like(Wxh)
         val dWhh = Vector.zeros_like(Whh)
         val dWhy = Vector.zeros_like(Why)
         val dbh  = Vector.zeros_like(bh)
         val dby  = Vector.zeros_like(by)
         val dhnext = Vector.zeros_like(hprev)
         val sum_loss = Vector.zeros(1)
         val hprev_new = Vector.zeros_like(hprev)

         def lossOneCycle(i: Int, hprev: Vector): Unit = {

           // get input as one-hot tensor
           val x = Vector.zeros(vocab_size)
           x.data(x_data(i)) = 1
           // get output as one-hot tensor
           val y = Vector.zeros(vocab_size)
           y.data(y_data(i)) = 1

           // forward pass
           val hs = ((Wxh dot x) + (Whh dot hprev) + bh).tanh()
           val ys = (Why dot hs) + by
           val ye = ys.exp()
           val ps = ye / ye.sum()
           sum_loss -= (ps dot y).log()

           if (i < 2) lossOneCycle(i + 1, hs)
           else hprev_new.copy_data(hs)

           // backward pass
           val dy = Vector.copy(ps)
           dy.data(y_data(i)) -= 1
           dWhy += (dy cart hs)
           dby += dy
           val dh = (Why.trans() dot dy) + dhnext
           val dhraw = (Vector.ones(1) - hs * hs) * dh
           dbh += dhraw
           dWxh += (dhraw cart x)
           dWhh += (dhraw cart hprev)
           dhnext.copy_data(Whh.trans() dot dhraw)
           ()
         }

         lossOneCycle(0, hprev)

         // assertions
         Vector.assertEqual(loss1, sum_loss, "loss")
         Vector.assertEqual(hprev_next, hprev_new, "hidden")
         Vector.assertEqual(Wxh1.d, dWxh, "dWxh")
         Vector.assertEqual(Whh1.d, dWhh, "dWhh")
         Vector.assertEqual(Why1.d, dWhy, "dWhy")
         Vector.assertEqual(bh1.d, dbh, "dbh")
         Vector.assertEqual(by1.d, dby, "dby")

        }
      }

      /*
      println("try array2_3")
      val array2_3file = new PrintWriter(new File("array2_3.cpp"))
      array2_3file.println(array2_3.code)
      array2_3file.flush()
      */
     println("run test case array2_3")
     array2_3.eval("abc")

     val array2_4 = new DslDriverC[String, Unit] with VectorExp {

       @virtualize
       def snippet (a: Rep[String]): Rep[Unit] = {
         val vocab_size = 3
         val by   = Vector.zeros(vocab_size)
         val by1  = TensorR.Tensor(by)
         val y = Vector.zeros(vocab_size)
         y.data(1) = 1
         val y1 = TensorR.Tensor(y)

         def lossFun = { (dummy: TensorR) =>

           val e1 = (by1).exp()
           val p1 = e1 / e1.sum()
           (p1 dot y1).log()
         }
         val dummy = gradR(lossFun)(Vector.zeros(1))
         // by1.d.print()


         // FIXME: need a correct implementation of gradient to check with
       }
     }

     //println("try array2_2_4")
     println("run test case array2_4")
     array2_4.eval("abc")

     val array2_5 = new DslDriverC[String, Unit] with VectorExp {

       @virtualize
       def snippet (a: Rep[String]): Rep[Unit] = {
         val vocab_size = 3
         val e   = Vector.ones(vocab_size)
         val e1  = TensorR.Tensor(e)
         val a   = Vector.ones(vocab_size)
         val a1  = TensorR.Tensor(a)
         val y = Vector.zeros(vocab_size)
         y.data(1) = 1
         val y1 = TensorR.Tensor(y)

         def lossFun = { (dummy: TensorR) =>
           //e1.sum()
           val p1 = a1 / e1.sum()
           (p1 dot y1).log()
         }
         val dummy = gradR(lossFun)(Vector.zeros(1))
         //e1.d.print()
         //a1.d.print()

         // FIXME: need a correct implementation of gradient to check with
       }
     }
     //println("try array2_2_5")
     println("run test case array2_5")
     array2_5.eval("abc")

     val array3 = new DslDriverC[String, Unit] with VectorExp {

       @virtualize
       def snippet(a: Rep[String]): Rep[Unit] = {
         // use random array as input
         val length = 2
         val v = Vector.randinit(length)
         //v.print()

         // calcuate gradient
         val grad = gradR(t => {val y = IF (length)(t.x.data(0) > 0.0) {t + t}{t * t}
         y.sum() })(v)
         // show gradient
         //grad.print()

         // another way of implementing it
         val grad1 = gradR(t => (t + t).sum())(v)
         val grad2 = gradR(t => (t * t).sum())(v)
         if (v(0) > 0) Vector.assertEqual(grad, grad1)
         else Vector.assertEqual(grad, grad2)
       }
     }

     //println("test IF gradient")
     val array3_file = new PrintWriter(new File("array3.cpp"))
     array3_file.println(array3.code)
     array3_file.flush()
     println("run test case array3")
     array3.eval("abc")

     val array4 = new DslDriverC[String, Unit] with VectorExp {

       @virtualize
       def snippet(a: Rep[String]): Rep[Unit] = {
         // use random array as input
         val length = 2
         val v = Vector.randinit(length)
         // v.print()

         val halfv = Vector.halves(length)
         val half = (new TensorR(halfv, Vector.zeros(length)))
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

     val array4_1 = new DslDriverC[String, Unit] with VectorExp {

       @virtualize
       def snippet(a: Rep[String]): Rep[Unit] = {
         val length = 2
         val v = Vector.randinit(length)
         // v.print()

         val half = new TensorR(Vector.halves(length), Vector.zeros(length))
         val grad = gradR(t => {
           val y = LOOPS(t)(3)(i => t => t * half )
           y.sum()
         })(v)
         // show gradient
         //grad.print()
         //println("Tensor in closure can also accumulate gradient, which is important")
         //half.d.print()

         val save_half_grad = Vector.zeros(length)
         save_half_grad.copy_data(half.d)

         // alternative implementation
         half.d.clear()
         val grad2 = gradR( t => {
           (t * half * half * half).sum()
         })(v)

         // assertion
         Vector.assertEqual(grad, grad2)
         Vector.assertEqual(save_half_grad, half.d)
       }
     }

     // println("test LOOP gradient")
     println("run test case array4_1")
     array4_1.eval("abc")

     // test using array data by closure
     val array4_2 = new DslDriverC[String, Unit] with VectorExp {

       def snippet(a: Rep[String]): Rep[Unit] = {

         // random initialization
         val length = 3
         val v = Vector.randinit(length)
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
             val data_point = TensorR.Tensor(new Vector(data(i), ddim1))
             x1 * data_point
           })
           y.sum()
         }

         val grad = gradR(model)(v)
         // show gradient
         //grad.print()

         // alternative implememetation
         val grad1 = gradR(t =>
             (t * TensorR.Tensor(new Vector(data(0), ddim1)) * TensorR.Tensor(new Vector(data(1), ddim1))).sum()
             )(v)
         // assertion
         Vector.assertEqual(grad, grad1)
       }
     }

     //println(array4_2_1.code)
     //val array4_2_1_file = new PrintWriter(new File("array4_2_1.cpp"))
     //array4_2_1_file.println(array4_2_1.code)
     //array4_2_1_file.flush()
     println("run test case of array4_2")
     array4_2.eval("abc")

     val array4_4 = new DslDriverC[String, Unit] with VectorExp {

       def snippet(a: Rep[String]): Rep[Unit] = {
         val length = 2
         val v = Vector.randinit(length)
         //v.print()
         val u = Vector.randinit(length, offset = 5)
         //u.print()

         val half = new TensorR(Vector.halves(length), Vector.zeros(length))
         val vv = TensorR.Tensor(v)
         val uu = TensorR.Tensor(u)

         val dummy = gradR(dum => {
           val in = ArrayBuffer[TensorR](vv, uu)
           val y = LOOPSM(in)(3)(i => ins => {
             val vvv = ins(0) * half
             val uuu = ins(1) * half
             ArrayBuffer[TensorR](vvv, uuu)
           })
         y(1).sum() + y(0).sum()})(Vector.zeros(1))
         // show gradient
         //println("Tensor in closure can also accumulate gradient, which is important")
         //half.d.print()
         //vv.d.print()
         //uu.d.print()

         // save gradients
         val save_vv_grad = Vector.zeros(length); save_vv_grad.copy_data(vv.d);   vv.clear_grad()
         val save_uu_grad = Vector.zeros(length); save_uu_grad.copy_data(uu.d);   uu.clear_grad()
         val save_ha_grad = Vector.zeros(length); save_ha_grad.copy_data(half.d); half.clear_grad()

         // alternative implementation
         val dummy1 = gradR(dum => {
           (vv * half * half * half + uu * half * half * half).sum()
         })(Vector.zeros(1))

         // assertions
         Vector.assertEqual(save_ha_grad, half.d)
         Vector.assertEqual(save_vv_grad, vv.d)
         Vector.assertEqual(save_uu_grad, uu.d)
       }
     }

     //println("support 2 tensors in loop using LOOPCCM")
     //println(array4_4.code)
     //val array4_4_file = new PrintWriter(new File("array4_4.cpp"))
     //array4_4_file.println(array4_4.code)
     //array4_4_file.flush()
     println("run test case in array4_4")
     array4_4.eval("abc")

     val array5 = new DslDriverC[String, Unit] with VectorExp {

       def snippet(a: Rep[String]): Rep[Unit] = {
         val length = 2
         val v = Vector.randinit(length)
         //v.print()

         val grad = gradR(t => (t * t).sum())(v)
         //grad.print()

         Vector.assertEqual(grad, v * Vector.consts(1, value = 2.0))
       }
     }

     //println("test elementwise multiplication")
     //println(array5.code)
     println("run test case in array5")
     array5.eval("abc")

     val array6 = new DslDriverC[String, Unit] with VectorExp {

       def snippet(a: Rep[String]): Rep[Unit] = {
         val length = 2
         val v = Vector.randinit(length)
         //v.print()

         val grad = gradR(t => (t / t).sum())(v)
         //grad.print()

         Vector.assertEqual(grad, Vector.zeros(length))
       }
     }

     // println("test elementwise division")
     //println(array6.code)
     println("run test case in array6")
     array6.eval("abc")

     val array7 = new DslDriverC[String, Unit] with VectorExp {

       def snippet(a: Rep[String]): Rep[Unit] = {
         val length = 2
         val v = Vector.randinit(length)
         //v.print()

         val grad = gradR(t => (t.tanh()).sum())(v)
         //grad.print()

         val e1 = v.tanh();
         val ee = Vector.ones(length) - e1 * e1
         Vector.assertEqual(grad, ee)
       }
     }

     // println("test tanh")
     //println(array7.code)
     println("run test case array7")
     array7.eval("abc")

     val array7_1 = new DslDriverC[String, Unit] with VectorExp {

       def snippet(a: Rep[String]): Rep[Unit] = {
         val length = 2
         val v = Vector.randinit(length)

         val grad = gradR(t => (t.sigmoid()).sum())(v)

         val e1 = v.sigmoid()
         val ee = (Vector.ones(1) - e1) * e1
         Vector.assertEqual(grad, ee)
       }
     }

     println("run test case array7_1")
     array7_1.eval("abc")

     val array8 = new DslDriverC[String, Unit] with VectorExp {

       def snippet(a: Rep[String]): Rep[Unit] = {
         val length = 2
         val v = Vector.randinit(length)
         // v.print()

         val grad = gradR(t => (t.exp()).sum())(v)
         //grad.print()

         Vector.assertEqual(grad, v.exp())
       }
     }

     // println("test exp")
     //println(array8.code)
     println("run test case in array8")
     array8.eval("abc")

     val array9 = new DslDriverC[String, Unit] with VectorExp {

       def snippet(a: Rep[String]): Rep[Unit] = {
         val length = 2
         val v = Vector.randPositive(length)
         //v.print()

         val grad = gradR(t => (t.log()).sum())(v)
         //grad.print()

         Vector.assertEqual(grad, Vector.ones(length) / v)
       }
     }

     //println("test log")
     // println(array9.code)
     println("run test case array9")
     array9.eval("abc")

     val array10 = new DslDriverC[String, Unit] with VectorExp {

       def snippet(a: Rep[String]): Rep[Unit] = {
         val length = 2
         val v = Vector.randinit(length)
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
           LOOPL(x)(arra.length)(i => x1 => new TensorR(new Vector(arra(i), length), Vector.zeros(length)) * x1)
         }
         val grad = gradR(t => (model(t)).sum())(v)
         //grad.print()

         val grad1 = gradR(t =>
             (t * TensorR.Tensor(new Vector(arra(0), length)) * TensorR.Tensor(new Vector(arra(1), length))).sum()
             )(v)

         Vector.assertEqual(grad, grad1)
       }
     }

     //println(array10.code)
     println("run test case in array10")
     array10.eval("abc")

     val array11 = new DslDriverC[String, Unit] with VectorExp {

       def snippet(a: Rep[String]): Rep[Unit] = {
         val length = 2
         val v = Vector.randinit(length)
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
            l * r * new TensorR(new Vector(arra(i), length), Vector.zeros(length))
          }
        }

        val grad = gradR(t => model(t).sum())(v)
        //grad.print()

        def model1: TensorR => TensorR @diff = { (x: TensorR) =>
          val leftchild  = x * TensorR.Tensor(new Vector(arra(1), length)) * x
          val rightchild = x * TensorR.Tensor(new Vector(arra(2), length)) * x
          val root = leftchild * TensorR.Tensor(new Vector(arra(0), length)) * rightchild
          root.sum()
        }

        val grad1 = gradR(model1)(v)
        // assertion
        Vector.assertEqual(grad, grad1)
       }
     }

     //println(array11.code)
     println("run test case array11")
     array11.eval("abc")

     val array11_1 = new DslDriverC[String, Unit] with VectorExp {

       def snippet(a: Rep[String]): Rep[Unit] = {
         val length = 2
         val v = Vector.randinit(length)
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

        val add: TensorR = TensorR.Tensor(Vector.ones(length))

        // create a model that recursively use the data (originated from tree)
        def model: TensorR => TensorR @diff = { (x: TensorR) =>
          val in = new ArrayBuffer[TensorR](); in.append(x); in.append(add)
          val tmp = LOOPTM(in)(lch1, rch1){ (l: ArrayBuffer[TensorR], r: ArrayBuffer[TensorR], i: Rep[Int]) =>
            val curr = TensorR.Tensor(new Vector(arra(i), length))
            val new_x = l(0) * r(0) * curr; val new_add = l(1) + r(1) + curr
            val out = new ArrayBuffer[TensorR](); out.append(new_x); out.append(new_add)
            out
          }
          tmp(0).sum() + tmp(1).sum()
        }

        val grad = gradR(t => model(t))(v)
        //grad.print()
        // save gradient of add
        val save_grad_add = Vector.zeros(length); save_grad_add.copy_data(add.d); add.clear_grad()

        def model1: TensorR => TensorR @diff = { (x: TensorR) =>
          val val1 = TensorR.Tensor(new Vector(arra(1), length))
          val leftchild  = x * val1 * x; val leftch = add + val1 + add
          val val2 = TensorR.Tensor(new Vector(arra(2), length))
          val rightchild = x * val2 * x; val rightch = add + val2 + add
          val val0 = TensorR.Tensor(new Vector(arra(0), length))
          val root = leftchild * val0 * rightchild; val root2 = leftch + val0 + rightch
          root.sum() + root2.sum()
        }

        val grad1 = gradR(model1)(v)
        // assertion
        Vector.assertEqual(grad, grad1)
        Vector.assertEqual(save_grad_add, add.d)
       }
     }

     //println(array11.code)
     println("run test case array11_1")
     array11_1.eval("abc")

    }

    val root_dir = "/home/fei/bitbucket/privategitrepoforshare/ICFP18evaluation/"
    val min_char_rnn = new DslDriverC[String, Unit] with VectorExp with ScannerLowerExp {

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
        //val Wxh = Vector.randinit(vocab_size, hidden_size, 0.01)  // input to hidden
        val Wxh = Vector.randn(vocab_size, hidden_size, 0.01)  // input to hidden
        val Whh = Vector.randn(hidden_size, hidden_size, 0.01) // hidden to hidden
        val Why = Vector.randn(hidden_size, vocab_size, 0.01)  // hidden to output
        val bh  = Vector.zeros(hidden_size)
        val by  = Vector.zeros(vocab_size)
        val hprev = Vector.zeros(hidden_size)

        val hnext = Vector.zeros_like(hprev)

        // wrap as tensors
        val Wxh1 = TensorR.Tensor(Wxh)
        val Whh1 = TensorR.Tensor(Whh)
        val Why1 = TensorR.Tensor(Why)
        val bh1  = TensorR.Tensor(bh)
        val by1  = TensorR.Tensor(by)
        val hprev1 = TensorR.Tensor(hprev)

        def lossFun(inputs: Rep[Array[Int]], targets: Rep[Array[Int]]) = { (dummy: TensorR) =>
          val loss = TensorR.Tensor(Vector.zeros(1))
          val in = ArrayBuffer[TensorR]()
          in.append(loss)
          in.append(hprev1)
          val outputs = LOOPSM(in)(inputs.length){i => t =>

            // printf("at iteration %d ", i)
            // get input as one-hot tensor
            val x = Vector.zeros(vocab_size)
            x.data(inputs(i)) = 1
            val x1 = TensorR.Tensor(x)
            // get output as one-hot tensor
            val y = Vector.zeros(vocab_size)
            y.data(targets(i)) = 1
            val y1 = TensorR.Tensor(y)

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


        val lr = Vector.consts(1, value = learning_rate)
        val hp = Vector.consts(1, value = 1e-8)

        val mWxh = Vector.zeros_like(Wxh)
        val mWhh = Vector.zeros_like(Whh)
        val mWhy = Vector.zeros_like(Why)
        val mbh  = Vector.zeros_like(bh)
        val mby  = Vector.zeros_like(by)

        val loss_save = NewArray[Double](51) // this array collects all loss
        val loopStartTime = get_time()

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

          val loss = gradR_loss(lossFun(inputs, targets))(Vector.zeros(1))
          val loss_value = loss.data(0) // we suppose the loss is scala (Vector of size 1)
          smooth_loss = smooth_loss * 0.9 + loss_value * 0.1
          if (n % 100 == 0) {
            printf("iter %d, loss %f\\n", n, smooth_loss)
            loss_save(n/100) = smooth_loss
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
          //printf("loss_saver is %lf \\n", loss_save(i))
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


    val min_char_list = new DslDriverC[String, Unit] with VectorExp with ScannerLowerExp {

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
        //val Wxh = Vector.randinit(vocab_size, hidden_size, 0.01)  // input to hidden
        val Wxh = Vector.randn(vocab_size, hidden_size, 0.01)  // input to hidden
        val Whh = Vector.randn(hidden_size, hidden_size, 0.01) // hidden to hidden
        val Why = Vector.randn(hidden_size, vocab_size, 0.01)  // hidden to output
        val bh  = Vector.zeros(hidden_size)
        val by  = Vector.zeros(vocab_size)
        val hprev = Vector.zeros(hidden_size)

        val hnext = Vector.zeros_like(hprev)

        // wrap as tensors
        val Wxh1 = TensorR.Tensor(Wxh)
        val Whh1 = TensorR.Tensor(Whh)
        val Why1 = TensorR.Tensor(Why)
        val bh1  = TensorR.Tensor(bh)
        val by1  = TensorR.Tensor(by)
        val hprev1 = TensorR.Tensor(hprev)

        def lossFun(inputs: Rep[Array[Int]], targets: Rep[Array[Int]]) = { (dummy: TensorR) =>
          val loss = TensorR.Tensor(Vector.zeros(1))
          val in = ArrayBuffer[TensorR]()
          in.append(loss)
          in.append(hprev1)
          val outputs = LOOPLM(in)(inputs.length){i => t =>

            // printf("at iteration %d ", i)
            // get input as one-hot tensor
            val x = Vector.zeros(vocab_size)
            x.data(inputs(i)) = 1
            val x1 = TensorR.Tensor(x)
            // get output as one-hot tensor
            val y = Vector.zeros(vocab_size)
            y.data(targets(i)) = 1
            val y1 = TensorR.Tensor(y)

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


        val lr = Vector.consts(1, value = learning_rate)
        val hp = Vector.consts(1, value = 1e-8)

        val mWxh = Vector.zeros_like(Wxh)
        val mWhh = Vector.zeros_like(Whh)
        val mWhy = Vector.zeros_like(Why)
        val mbh  = Vector.zeros_like(bh)
        val mby  = Vector.zeros_like(by)

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

          val loss = gradR_loss(lossFun(inputs, targets))(Vector.zeros(1))
          val loss_value = loss.data(0) // we suppose the loss is scala (Vector of size 1)
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


    //println("try array2_2_3")
    //val min_char_rnn_file = new PrintWriter(new File("minchar.cpp"))
    //min_char_rnn_file.println(min_char_rnn.code)
    //min_char_rnn_file.flush()
    //min_char_list.eval("abc")
    //println("verified that in this small example the values of gradients are about right (up to precision)")

    val min_char_lstm = new DslDriverC[String, Unit] with VectorExp with ScannerLowerExp {

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
        val Wfh = Vector.randn(hidden_size, hidden_size, 0.01)
        val Wfx = Vector.randn(vocab_size, hidden_size, 0.01)
        val bf  = Vector.zeros(hidden_size)
        val Wih = Vector.randn(hidden_size, hidden_size, 0.01)
        val Wix = Vector.randn(vocab_size, hidden_size, 0.01)
        val bi  = Vector.zeros(hidden_size)
        val Wch = Vector.randn(hidden_size, hidden_size, 0.01)
        val Wcx = Vector.randn(vocab_size, hidden_size, 0.01)
        val bc  = Vector.zeros(hidden_size)
        val Woh = Vector.randn(hidden_size, hidden_size, 0.01)
        val Wox = Vector.randn(vocab_size, hidden_size, 0.01)
        val bo  = Vector.zeros(hidden_size)
        val Why = Vector.randn(hidden_size, vocab_size, 0.01)  // hidden to output
        val by  = Vector.zeros(vocab_size)

        val hprev = Vector.zeros(hidden_size)
        val cprev = Vector.zeros(hidden_size)
        val hsave = Vector.zeros_like(hprev)
        val csave = Vector.zeros_like(cprev)

        // wrap as Tensors
        val tWfh = TensorR.Tensor(Wfh)
        val tWfx = TensorR.Tensor(Wfx)
        val tbf = TensorR.Tensor(bf)
        val tWih = TensorR.Tensor(Wih)
        val tWix = TensorR.Tensor(Wix)
        val tbi = TensorR.Tensor(bi)
        val tWch = TensorR.Tensor(Wch)
        val tWcx = TensorR.Tensor(Wcx)
        val tbc = TensorR.Tensor(bc)
        val tWoh = TensorR.Tensor(Woh)
        val tWox = TensorR.Tensor(Wox)
        val tbo = TensorR.Tensor(bo)
        val tWhy = TensorR.Tensor(Why)
        val tby = TensorR.Tensor(by)
        val thprev = TensorR.Tensor(hprev)
        val tcprev = TensorR.Tensor(cprev)


        // lossFun
        def lossFun(inputs: Rep[Array[Int]], targets: Rep[Array[Int]]) = { (dummy: TensorR) =>

          val loss = TensorR.Tensor(Vector.zeros(1))
          val in = ArrayBuffer[TensorR]()

          in.append(loss)
          in.append(thprev)
          in.append(tcprev)

          val outputs = LOOPSM(in)(inputs.length){i => t =>

            // get input as one-hot tensor
            val x = Vector.zeros(vocab_size)
            x.data(inputs(i)) = 1
            val x1 = TensorR.Tensor(x)
            // get output as one-hot tensor
            val y = Vector.zeros(vocab_size)
            y.data(targets(i)) = 1
            val y1 = TensorR.Tensor(y)

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


        val lr = Vector.consts(1, value = learning_rate)
        val hp = Vector.consts(1, value = 1e-8)

        val mWfh = Vector.zeros_like(Wfh)
        val mWfx = Vector.zeros_like(Wfx)
        val mbf = Vector.zeros_like(bf)
        val mWih = Vector.zeros_like(Wih)
        val mWix = Vector.zeros_like(Wix)
        val mbi = Vector.zeros_like(bi)
        val mWch = Vector.zeros_like(Wch)
        val mWcx = Vector.zeros_like(Wcx)
        val mbc = Vector.zeros_like(bc)
        val mWoh = Vector.zeros_like(Woh)
        val mWox = Vector.zeros_like(Wox)
        val mbo = Vector.zeros_like(bo)
        val mWhy = Vector.zeros_like(Why)
        val mby = Vector.zeros_like(by)

        val loopStart = get_time()
        val loss_save = NewArray[Double](51)

        val addr = getMallocAddr() // remember current allocation pointer here

        val startAt = var_new[Int](0)
        startAt -= seq_length

        //val timer = Timer()
        //timer.startTimer

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

          val loss = gradR_loss(lossFun(inputs, targets))(Vector.zeros(1))
          val loss_value = loss.data(0) // we suppose the loss is scala (Vector of size 1)
          smooth_loss = smooth_loss * 0.9 + loss_value * 0.1
          if (n % 100 == 0) {
            printf("iter %d, loss %f\\n", n, smooth_loss)
            //timer.printElapsedTime
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

    val senti_seq_lstm = new DslDriverC[String, Unit] with VectorExp with ScannerLowerExp {

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
        val Wfh = Vector.randn(hidden_size, hidden_size, 0.01)
        val Wfx = Vector.randn(word_embedding_size, hidden_size, 0.01)
        val bf  = Vector.zeros(hidden_size)
        val Wih = Vector.randn(hidden_size, hidden_size, 0.01)
        val Wix = Vector.randn(word_embedding_size, hidden_size, 0.01)
        val bi  = Vector.zeros(hidden_size)
        val Wch = Vector.randn(hidden_size, hidden_size, 0.01)
        val Wcx = Vector.randn(word_embedding_size, hidden_size, 0.01)
        val bc  = Vector.zeros(hidden_size)
        val Woh = Vector.randn(hidden_size, hidden_size, 0.01)
        val Wox = Vector.randn(word_embedding_size, hidden_size, 0.01)
        val bo  = Vector.zeros(hidden_size)
        val Why = Vector.randn(hidden_size, output_size, 0.01)  // hidden to output
        val by  = Vector.zeros(output_size)

        val hprev = Vector.zeros(hidden_size)
        val cprev = Vector.zeros(hidden_size)

        // wrap as Tensors
        val tWfh = TensorR.Tensor(Wfh)
        val tWfx = TensorR.Tensor(Wfx)
        val tbf = TensorR.Tensor(bf)
        val tWih = TensorR.Tensor(Wih)
        val tWix = TensorR.Tensor(Wix)
        val tbi = TensorR.Tensor(bi)
        val tWch = TensorR.Tensor(Wch)
        val tWcx = TensorR.Tensor(Wcx)
        val tbc = TensorR.Tensor(bc)
        val tWoh = TensorR.Tensor(Woh)
        val tWox = TensorR.Tensor(Wox)
        val tbo = TensorR.Tensor(bo)
        val tWhy = TensorR.Tensor(Why)
        val tby = TensorR.Tensor(by)
        val thprev = TensorR.Tensor(hprev)
        val tcprev = TensorR.Tensor(cprev)

        // lossFun
        def lossFun(inputs: Rep[Array[Int]], label: Rep[Int]) = { (dummy: TensorR) =>

          val in = ArrayBuffer[TensorR]()
          in.append(thprev)
          in.append(tcprev)

          val outputs = LOOPSM(in)(inputs.length){i => t =>

            // get word embedding
            val x    = word_embedding_data(inputs(i))
            val x1   = TensorR.Tensor(new Vector(x, word_embedding_size))

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

          val y = Vector.zeros(output_size)
          y.data(label) = 1
          val y1 = TensorR.Tensor(y)

          val loss = TensorR.Tensor(Vector.zeros(1)) - (pt dot y1).log()
          loss
        }


        val lr = Vector.consts(1, value = learning_rate)
        val hp = Vector.consts(1, value = 1e-8)

        val mWfh = Vector.zeros_like(Wfh)
        val mWfx = Vector.zeros_like(Wfx)
        val mbf = Vector.zeros_like(bf)
        val mWih = Vector.zeros_like(Wih)
        val mWix = Vector.zeros_like(Wix)
        val mbi = Vector.zeros_like(bi)
        val mWch = Vector.zeros_like(Wch)
        val mWcx = Vector.zeros_like(Wcx)
        val mbc = Vector.zeros_like(bc)
        val mWoh = Vector.zeros_like(Woh)
        val mWox = Vector.zeros_like(Wox)
        val mbo = Vector.zeros_like(bo)
        val mWhy = Vector.zeros_like(Why)
        val mby = Vector.zeros_like(by)

        val addr = getMallocAddr() // remember current allocation pointer here

        for (n <- (0 until 2001): Rep[Range]) {

          val index  = n % seq_number
          val inputs = seq_data(index)
          val label  = seq_label(index)

          val loss = gradR_loss(lossFun(inputs, label))(Vector.zeros(1))
          val loss_value = loss.data(0) // we suppose the loss is scala (Vector of size 1)
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


    val sentimental_rnn = new DslDriverC[String, Unit] with VectorExp with ScannerLowerExp {

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
       val Wxh = Vector.randinit(word_embedding_size, hidden_size, 0.01) // from word embedding to hidden vector
       val bx  = Vector.zeros(hidden_size)                               // bias word embedding to hidden vector
       val Wlh = Vector.randinit(hidden_size, hidden_size, 0.01)         // from hidden vector of left child to hidden
       val Wrh = Vector.randinit(hidden_size, hidden_size, 0.01)         // from hidden vector of right child to hidden
       val bh  = Vector.zeros(hidden_size)                               // bias from children hidden vector to hidden
       val Why = Vector.randinit(hidden_size, output_size, 0.01)         // from hidden vector to output
       val by  = Vector.zeros(output_size)                               // bias hidden vector to output

       // Cast Vectors as Tensors
       val Wxh1 = TensorR.Tensor(Wxh)
       val bx1  = TensorR.Tensor(bx)
       val Wlh1 = TensorR.Tensor(Wlh)
       val Wrh1 = TensorR.Tensor(Wrh)
       val bh1  = TensorR.Tensor(bh)
       val Why1 = TensorR.Tensor(Why)
       val by1  = TensorR.Tensor(by)

       def lossFun(scores: Rep[Array[Int]], words: Rep[Array[Int]], lchs: Rep[Array[Int]], rchs: Rep[Array[Int]]) = { (dummy: TensorR) =>

         val initial_loss = TensorR.Tensor(Vector.zeros(1))
         val initial_hidd = TensorR.Tensor(Vector.zeros(hidden_size))
         val inBuffer     = new ArrayBuffer[TensorR]()
         inBuffer.append(initial_loss); inBuffer.append(initial_hidd) // construct the input to LOOPTM

         val outBuffer = LOOPTM(inBuffer)(lchs, rchs) { (l: ArrayBuffer[TensorR], r: ArrayBuffer[TensorR], i: Rep[Int]) =>

           val targ = Vector.zeros(output_size); targ.data(scores(i)) = 1; val targ1 = TensorR.Tensor(targ)
           val lossl = l(0); val hiddenl = l(1)
           val lossr = r(0); val hiddenr = r(1)

           val hidden = IF (hidden_size) (lchs(i) < 0) { // leaf node
             val embedding_array = word_embedding_data(words(i))
             val embedding_tensor = TensorR.Tensor(new Vector(embedding_array, word_embedding_size))
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

       val lr = Vector.consts(1, value = learning_rate)
       val hp = Vector.consts(1, value = 1e-8)

       val mWxh = Vector.zeros_like(Wxh)
       val mbx  = Vector.zeros_like(bx)
       val mWlh = Vector.zeros_like(Wlh)
       val mWrh = Vector.zeros_like(Wrh)
       val mbh  = Vector.zeros_like(bh)
       val mWhy = Vector.zeros_like(Why)
       val mby  = Vector.zeros_like(by)

       val addr = getMallocAddr() // remember current allocation pointer here

       for (epoc <- (0 until 10): Rep[Range]) {

         var ave_loss = 0.0
         for (n <- (0 until tree_number): Rep[Range]) {

           val index = n % tree_number
           val scores   = tree_data(index * 4)
           val words    = tree_data(index * 4 + 1)
           val leftchs  = tree_data(index * 4 + 2)
           val rightchs = tree_data(index * 4 + 3)
           val loss = gradR_loss(lossFun(scores, words, leftchs, rightchs))(Vector.zeros(1))
           val loss_value = loss.data(0)  // we suppose the loss is scala (Vector of size 1)
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

    val sentimental_lstm = new DslDriverC[String, Unit] with VectorExp with ScannerLowerExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        val startTime = get_time()

        // read in the data for word embedding
        val word_embedding_size   = 300 // this is statically known
        
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

        // read in the data for trees (in array format)
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
        val Wi = Vector.randinit(word_embedding_size, hidden_size, 0.01)  // from word embedding to hidden vector, input gate
        val bi = Vector.zeros(hidden_size)                                // bias word embedding to hidden vector, input gate
        val Wo = Vector.randinit(word_embedding_size, hidden_size, 0.01)  // from word embedding to hidden vector, outout gate
        val bo = Vector.zeros(hidden_size)                                // bias word embedding to hidden vector, outout gate
        val Wu = Vector.randinit(word_embedding_size, hidden_size, 0.01)  // from word embedding to hidden vector, cell state
        val bu = Vector.zeros(hidden_size)                                // bias word embedding to hidden vector, cell state
        // parameters for non-leaf node
        val U0i  = Vector.randinit(hidden_size, hidden_size, 0.01) // left child, input gate
        val U1i  = Vector.randinit(hidden_size, hidden_size, 0.01) // right child, input gate
        val bbi  = Vector.zeros(hidden_size)                       // bias, input gate
        val U00f = Vector.randinit(hidden_size, hidden_size, 0.01) // left-left forget gate
        val U01f = Vector.randinit(hidden_size, hidden_size, 0.01) // left-right forget gate
        val U10f = Vector.randinit(hidden_size, hidden_size, 0.01) // right-left forget gate
        val U11f = Vector.randinit(hidden_size, hidden_size, 0.01) // right-right forget gate
        val bbf  = Vector.zeros(hidden_size)                       // bias, forget gate
        val U0o  = Vector.randinit(hidden_size, hidden_size, 0.01) // left child, output gate
        val U1o  = Vector.randinit(hidden_size, hidden_size, 0.01) // right child, output gate
        val bbo  = Vector.zeros(hidden_size)                       // bias, output gate
        val U0u  = Vector.randinit(hidden_size, hidden_size, 0.01) // left child, cell state
        val U1u  = Vector.randinit(hidden_size, hidden_size, 0.01) // right child, cell state
        val bbu  = Vector.zeros(hidden_size)                       // bias, cell state
        // parameters for softmax
        val Why = Vector.randinit(hidden_size, output_size, 0.01)         // from hidden vector to output
        val by  = Vector.zeros(output_size)                               // bias hidden vector to output

        // Cast Vectors as Tensors
        val tWi = TensorR.Tensor(Wi)
        val tbi = TensorR.Tensor(bi)
        val tWo = TensorR.Tensor(Wo)
        val tbo = TensorR.Tensor(bo)
        val tWu = TensorR.Tensor(Wu)
        val tbu = TensorR.Tensor(bu)
        // Cast Vectors as Tensors
        val tU0i  = TensorR.Tensor(U0i)
        val tU1i  = TensorR.Tensor(U1i)
        val tbbi  = TensorR.Tensor(bbi)
        val tU00f = TensorR.Tensor(U00f)
        val tU01f = TensorR.Tensor(U01f)
        val tU10f = TensorR.Tensor(U10f)
        val tU11f = TensorR.Tensor(U11f)
        val tbbf = TensorR.Tensor(bbf)
        val tU0o = TensorR.Tensor(U0o)
        val tU1o = TensorR.Tensor(U1o)
        val tbbo = TensorR.Tensor(bbo)
        val tU0u = TensorR.Tensor(U0u)
        val tU1u = TensorR.Tensor(U1u)
        val tbbu = TensorR.Tensor(bbu)
        // Cast Vectors as Tensors
        val tWhy = TensorR.Tensor(Why)
        val tby  = TensorR.Tensor(by)

        val dummy_word_embedding = TensorR.Tensor(Vector.zeros(word_embedding_size))
        val dummy_forget_gate    = TensorR.Tensor(Vector.zeros(hidden_size))

        def lossFun(scores: Rep[Array[Int]], words: Rep[Array[Int]], lchs: Rep[Array[Int]], rchs: Rep[Array[Int]]) = { (dummy: TensorR) =>

          val initial_loss = TensorR.Tensor(Vector.zeros(1))
          val initial_hidd = TensorR.Tensor(Vector.zeros(hidden_size))
          val initial_cell = TensorR.Tensor(Vector.zeros(hidden_size))
          val inBuffer     = new ArrayBuffer[TensorR]()
          inBuffer.append(initial_loss); inBuffer.append(initial_hidd); inBuffer.append(initial_cell)

          val outBuffer = LOOPTM(inBuffer)(lchs, rchs) { (l: ArrayBuffer[TensorR], r: ArrayBuffer[TensorR], i: Rep[Int]) =>

            val lossl = l(0); val hiddenl = l(1); val celll = l(2)
            val lossr = r(0); val hiddenr = r(1); val cellr = r(2)

            val targ = Vector.zeros(output_size); targ.data(scores(i)) = 1; val targ1 = TensorR.Tensor(targ)

            val embedding_tensor = IF (word_embedding_size) (lchs(i) < 0) {
              TensorR.Tensor(new Vector(word_embedding_data(words(i)), word_embedding_size))
            } {dummy_word_embedding}

            val i_gate = IF (hidden_size) (lchs(i) < 0) {
              (tWi.dot(embedding_tensor) + tbi).sigmoid()
            } {
              (tU0i.dot(hiddenl) + tU1i.dot(hiddenr) + tbbi).sigmoid()
            }

            //val i_gate = (tWi.dot(embedding_tensor) + tU0i.dot(hiddenl) + tU1i.dot(hiddenr) + tbi).sigmoid()

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

        val lr = Vector.consts(1, value = learning_rate)
        val hp = Vector.consts(1, value = 1e-8)

        // parameters for leaf node
        val mWi = Vector.zeros_like(Wi)
        val mbi = Vector.zeros_like(bi)
        val mWo = Vector.zeros_like(Wo)
        val mbo = Vector.zeros_like(bo)
        val mWu = Vector.zeros_like(Wu)
        val mbu = Vector.zeros_like(bu)
        // parameters for non-leaf node
        val mU0i  = Vector.zeros_like(U0i)
        val mU1i  = Vector.zeros_like(U1i)
        val mbbi  = Vector.zeros_like(bbi)
        val mU00f = Vector.zeros_like(U00f)
        val mU01f = Vector.zeros_like(U01f)
        val mU10f = Vector.zeros_like(U10f)
        val mU11f = Vector.zeros_like(U11f)
        val mbbf  = Vector.zeros_like(bbf)
        val mU0o  = Vector.zeros_like(U0o)
        val mU1o  = Vector.zeros_like(U1o)
        val mbbo  = Vector.zeros_like(bbo)
        val mU0u  = Vector.zeros_like(U0u)
        val mU1u  = Vector.zeros_like(U1u)
        val mbbu  = Vector.zeros_like(bbu)
        // parameters for softmax
        val mWhy = Vector.zeros_like(Why)
        val mby  = Vector.zeros_like(by)

        // for saving loss array
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
            val loss = gradR_loss(lossFun(scores, words, leftchs, rightchs))(Vector.zeros(1))
            val loss_value = loss.data(0)  // we suppose the loss is scala (Vector of size 1)
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
          
          //timer.printElapsedTime

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
    
    println("run sentiment analysis tree LSTM")
    val sentit_file = new PrintWriter(new File(root_dir + "evaluationTreeLSTM/Lantern/Lantern.cpp"))
    sentit_file.println(sentimental_lstm.code)
    sentit_file.flush()
    //sentimental_lstm.eval("abc")
  }
}
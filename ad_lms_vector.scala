import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

import scala.collection.mutable.ArrayBuffer

object TEST1 {

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

    /**
      Add: Scanner class for Input
      Copied from lms-query-tutorial
      **/
    
    object Encoding {
      val ix_a = 97
      def char_to_ix(ch: Rep[Char]): Rep[Int] = ch.AsInstanceOf[Int] - ix_a
      def ix_to_char(ix: Rep[Int]): Rep[Char] = (ix +ix_a).AsInstanceOf[Char]
    }

    class Vector(val data: Rep[Array[Double]], val dim0: Int, val dim1:Int = 1 /*, val dim2: Int*/) extends Serializable {

      def apply(i: Rep[Int]) = data(i)
      def apply(i: Rep[Int], j: Rep[Int]) = data(i + j * dim0) // FIXME the index of matrix is not the normal way

      def max(a: Int, b: Int) = if (a >= b) a else b

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
        else throw new IllegalArgumentException("dimensions of vector do not match +!")
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
        else throw new IllegalArgumentException("dimensions of vector do not match +!")
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
        else throw new IllegalArgumentException("dimensions of vector do not match +!")
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

    }

    object Vector {

      def randinit(dim0: Int, dim1: Int = 1, scale: Double = 1.0, offset: Int = 0) = {
        unchecked[Unit]("srand(time(NULL)" + "+" + offset.toString + ")")
        val res = NewArray[Double](dim0 * dim1)
        for (i <- (0 until dim0 * dim1): Rep[Range]) res(i) = unchecked[Double]("(double)rand()/RAND_MAX*2.0-1.0") * scale
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

      def + (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) => 
        val y = new TensorR(x + that.x, Vector.zeros(x.dim0)); k(y)
        this.d += y.d; that.d += y.d
      }

      def - (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
        val y = new TensorR(x - that.x, Vector.zeros(x.dim0)); k(y)
        this.d += y.d; that.d -= y.d
      }

      // this is element wise multiplication
      def * (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
        val y = new TensorR(x * that.x, Vector.zeros(x.dim0)); k(y)
        // FIXME: intermediate Tensors donot need to be substatiated, can optimize!
        this.d += that.x * y.d; 
        that.d += this.x * y.d; 
      }

      // element wise division
      def / (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
        val y = new TensorR(x / that.x, Vector.zeros(x.dim0)); k(y)
        // FIXME: intermediate Tensors donot need to be substatiated, can optimize!
        this.d += y.d / that.x
        that.d -= this.x * y.d / (that.x * that.x) 
      }

      // vector dot product or Matrix vector dot (viewed as multiple vector dot product) (not the common view)
      def dot(that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
        val y = new TensorR(x dot that.x, Vector.zeros(x.dim1)); k(y)
        // FIXME: intermediate Tensors donot need to be substatiated, can optimize!
        this.d.add_cartesian(that.x, y.d) 
        that.d.add_composion(this.x, y.d)
        // this.d += that.x * y.d // broadcasting
        // that.d += this.x * y.d // broadcasting 
      }

      def tanh(): TensorR @diff = shift { (k : TensorR => Unit) =>
        val y = new TensorR(x.tanh(), Vector.zeros(x.dim0)); k(y)
        // FIXME: intermediate Tensors donot need to be substatiated, can optimize!
        this.d += (Vector.ones(x.dim0) - y.x * y.x) * y.d // broadcasting
      }

      def exp(): TensorR @diff = shift { (k: TensorR => Unit) =>
        val y = new TensorR(x.exp(), Vector.zeros(x.dim0)); k(y)
        // Fix
        this.d += y.x * y.d
      }

      def log(): TensorR @diff = shift { (k: TensorR => Unit) =>
        val y = new TensorR(x.log(), Vector.zeros(x.dim0)); k(y)
        // Fix
        this.d += y.d / x
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


    def FUN(dim0: Int)(f: TensorR => Unit): (TensorR => Unit) = {
      // val dim0: Int = 1 // FIXME: what is the best way to carry this known dimensional information?
      val f1 = fun { (x: Rep[Array[Double]]) => 
        val deltaVar: Vector = Vector.zeros(dim0)
        f(new TensorR(new Vector(x, dim0), deltaVar))
        deltaVar.data
      };
      { (x:TensorR) => x.d += new Vector(f1(x.x.data), dim0) }
    }

/*
    // "clever" way of changing fun signature for memory leak issue, not working
    def FUNc(dim0: Int)(f: TensorR => Unit): (TensorR => Unit) = {
      val f1 = fun { (x: Rep[Array[Array[Double]]]) =>
        val length = 2
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
*/

    def RST(a: => Unit @diff) = continuations.reset { a; () }

    @virtualize
    def IF(dim0: Int)(c: Rep[Boolean])(a: =>TensorR @diff)(b: =>TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>
      val k1 = FUN(dim0)(k)

      if (c) RST(k1(a)) else RST(k1(b))
    }

    @virtualize
    def LOOP(init: TensorR)(c: TensorR => Rep[Boolean])(b: TensorR => TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>
      // val k1 = FUN(init.x.dim0)(k)

      lazy val loop: TensorR => Unit = FUN (init.x.dim0) { (x: TensorR) =>
      if (c(x)) RST(loop(b(x))) else RST(k(x))
      }
      loop(init)
    }

    @virtualize
    def LOOPC(init: TensorR)(c: Rep[Int])(b: TensorR => TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>
      
      var gc = 0

      lazy val loop: TensorR => Unit = FUN (init.x.dim0){ (x: TensorR) =>
      if (gc < c) { gc += 1; RST(loop(b(x))) } else RST(k(x))
      }
      loop(init)
    }

    @virtualize
    def LOOPCC(init: TensorR)(c: Rep[Int])(b: Rep[Int] => TensorR => TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>

      var gc = 0

      lazy val loop: TensorR => Unit = FUN (init.x.dim0){ (x: TensorR) =>
        if (gc < c) { gc += 1; RST(loop(b(gc - 1)(x))) } else RST(k(x))
      }
      loop(init)
    }
    
    def FUN2(dim0: Int)(f: ArrayBuffer[TensorR] => Unit): (ArrayBuffer[TensorR] => Unit) = {
      // val dim0: Int = 1 // FIXME: what is the best way to carry this known dimensional information?
      val f1 = fun { (x: Rep[Array[Array[Double]]]) =>
        val length = 2
        val deltas = ArrayBuffer[Vector]() 
        for (i <- (0 until length): Range) deltas.append(Vector.zeros(dim0))
        val inputs = ArrayBuffer[TensorR]()
        for (i <- (0 until length): Range) inputs.append(new TensorR(new Vector(x(i), dim0), deltas(i)))
        f(inputs)
        val ret = NewArray[Array[Double]](length)
        for (i <- (0 until length): Range) ret(i) = deltas(i).data
        ret
      };
      { (x:ArrayBuffer[TensorR]) => {
        val length = 2
        val in = NewArray[Array[Double]](length)
        for (i <- (0 until length): Range) in(i) = x(i).x.data
        val out = f1(in)
        for (i <- (0 until length): Range) x(i).d += new Vector(out(i), dim0)
        } 
      }
    }

    @virtualize
    def LOOPCC2(init: ArrayBuffer[TensorR])(c: Rep[Int])(b: Rep[Int] => ArrayBuffer[TensorR] => ArrayBuffer[TensorR] @diff): 
    ArrayBuffer[TensorR] @diff = shift { k:(ArrayBuffer[TensorR] => Unit) =>

      var gc = 0

      lazy val loop: ArrayBuffer[TensorR] => Unit = FUN2 (init(0).x.dim0){ (x: ArrayBuffer[TensorR]) =>
      if (gc < c) { gc += 1; RST(loop(b(gc - 1)(x))) } else RST(k(x))
      }
      loop(init)
    }      

    def FUNM(dim0s: ArrayBuffer[Int])(f: ArrayBuffer[TensorR] => Unit): (ArrayBuffer[TensorR] => Unit) = {
      val f1 = fun { (x: Rep[Array[Array[Double]]]) =>
        val length = dim0s.length
        val deltas = ArrayBuffer[Vector]()
        for (i <- (0 until length): Range) deltas.append(Vector.zeros(dim0s(i)))
        val inputs = ArrayBuffer[TensorR]()
        for (i <- (0 until length): Range) inputs.append(new TensorR(new Vector(x(i), dim0s(i)), deltas(i)))
        f(inputs)
        val rets = NewArray[Array[Double]](length)
        for (i <- (0 until length): Range) rets(i) = deltas(i).data
        rets
      };
      {
        (x: ArrayBuffer[TensorR]) => {
          val length = dim0s.length
          val in = NewArray[Array[Double]](length)
          for (i <- (0 until length): Range) in(i) = x(i).x.data
          val out = f1(in)
          for (i <- (0 until length): Range) x(i).d += new Vector(out(i), dim0s(i))
        }
      }
    }

    @virtualize
    def LOOPCCM(init: ArrayBuffer[TensorR])(c: Rep[Int])(b: Rep[Int] => ArrayBuffer[TensorR] => ArrayBuffer[TensorR] @diff):
    ArrayBuffer[TensorR] @diff = shift { k: (ArrayBuffer[TensorR] => Unit) =>

      var gc = 0
      lazy val loop: ArrayBuffer[TensorR] => Unit = FUNM (init map (_.x.dim0)) { (x: ArrayBuffer[TensorR]) =>
        if (gc < c){gc += 1; RST(loop(b(gc-1)(x)))} else RST(k(x))
      }
      loop(init)
    } 


    def FUNL(dim0: Int)(f: (Rep[Int] => (TensorR => Unit) => (TensorR => Unit))): (Rep[Int] => (TensorR => Unit) => (TensorR => Unit)) = {      

      val f1 = fun { (yy: Rep[(Int, (Array[Double] => Array[Double]), Array[Double])]) => // Problem! no support for tuple type code generation in LMS!
        val i: Rep[Int] = tuple3_get1(yy)
        val t1: Rep[Array[Double] => Array[Double]] = tuple3_get2(yy)
        val xx: Rep[Array[Double]] = tuple3_get3(yy)
        //case (i: Rep[Int], t1: Rep[Double => Double]) =>
        val t2: (TensorR => Unit) = { (x: TensorR) => x.d += new Vector(t1(x.x.data), dim0) }
        val t3: (TensorR => Unit) = f(i)(t2)

        val deltas = Vector.zeros(dim0)
        t3(new TensorR(new Vector(xx, dim0), deltas))
        deltas.data
      };

      {i: Rep[Int] => k1: (TensorR => Unit) => 
        {
          val k2: Rep[Array[Double] => Array[Double]] = fun { (x: Rep[Array[Double]]) =>
            val deltas = Vector.zeros(dim0)
            k1(new TensorR(new Vector(x, dim0), deltas))
            deltas.data
          }
          val k4: (TensorR => Unit) = {(x: TensorR) => 
            x.d += new Vector(f1((i, k2, x.x.data)), dim0)
          }
          k4
        } 
      }
    }



    @virtualize
    def LOOPL(init: TensorR)(c: Rep[Int])(b: Rep[Int] => TensorR => TensorR @diff): TensorR @diff = shift { k: (TensorR => Unit) =>
      lazy val loop: Rep[Int] => (TensorR => Unit) => TensorR => Unit = FUNL(init.x.dim0){ (gc: Rep[Int]) => (k: TensorR => Unit) => (x: TensorR) =>
        if (gc < c) { loop(gc+1)((x: TensorR) => RST(k(b(gc)(x))))(x) } else { RST(k(x)) }
      }
      loop(0)(k)(init)
    }

    @virtualize
    def LOOPT(init: TensorR)(bound: Rep[Int], lch: Rep[Array[Int]], rch: Rep[Array[Int]])(b: (TensorR, TensorR, Rep[Int]) => TensorR @diff): TensorR @diff = shift {
      k: (TensorR => Unit) =>

      lazy val tree: Rep[Int] => (TensorR => Unit) => TensorR => Unit = FUNL(init.x.dim0){ (i: Rep[Int]) => (k: TensorR => Unit) => (x: TensorR) =>
        if (i < bound) { tree(lch(i))((l: TensorR) => tree(rch(i))((r: TensorR) => RST(k(b(l, r, i))))(x))(x) } else { RST(k(x)) }
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

    //println(array0.code)
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
    array2.eval("2.0")

    val array2_1 = new DslDriverC[String, Unit] with VectorExp {
      // update gradient as side effect
      
      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Vector.randinit(length)
        // v.print()

        // initialize tensor for closure
        var t = new TensorR(v, Vector.zeros(length))            
        val half = new TensorR(Vector.halves(length), Vector.zeros(length))
        // call grad_side_effect_using_closure
        val dummy = gradR(dummy => {
          ((t dot t) * half).sum()
          })(Vector.zeros(1))
        // print the gradient of t
        //t.d.print()
        //half.d.print()
        Vector.assertEqual(t.d, v * Vector.consts(1, value = 2.0))
        Vector.assertEqual(half.d, Vector.expand((v dot v).data(0), 2))
        ()
      }
    }

    //println("test dot gradient as side effect with var update") 
    //println("proving that I can use var update without creating cycles in static computation graph")
    //println(array2_1.code)
    array2_1.eval("2.0")

    val array2_2 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {

        val dim0 = 2
        val dim1 = 3
        val matrix = Vector.randinit(dim0, dim1)
        val vector = Vector.randinit(dim0, offset = 4)
        //matrix.print()
        //vector.print()

        // initialize tensors for closure
        val ma = new TensorR(matrix, Vector.zeros(dim0, dim1))
        val ve = new TensorR(vector, Vector.zeros(dim0))
        // define function of model
        def model(dummy: TensorR): TensorR @diff = {
          (ma dot ve).sum()
        }
        val loss = gradR_loss(model)(Vector.zeros(1))
        // print the gradient of ma and ve
        //ma.d.print()
        //ve.d.print()
        Vector.assertEqual(loss, (matrix dot vector).sum())
        Vector.assertEqual(ma.d, Vector.expand(vector, dim1))
        Vector.assertEqual(ve.d, matrix.sumOnDim1())
        ()
      }
    }

    // println("test matrix vector dot gradient as side effect")
    //println(array2_2.code)
    array2_2.eval("abc")


    val array2_2_1 = new DslDriverC[String, Unit] with VectorExp {

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
          val outputs = LOOPCCM(in)(3){i => t => 
            
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
    println("try array2_2_1")
    val p = new PrintWriter(new File("array2_2_1.cpp"))
    p.println(array2_2_1.code)
    p.flush()
    */
    array2_2_1.eval("abc")

    val array2_2_3 = new DslDriverC[String, Unit] with VectorExp with ScannerLowerExp {
      
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
        val scanner = new Scanner("input.txt")        
        val training_data = scanner.data
        val data_size = scanner.fl
        // val chars = training_data.distinct  /** this can be done in second stage **/
        // val vocab_size = chars.length
        println(s"data has $data_size chars")

        //val translated_data = NewArray[Int](data_size)
        //for (i <- (0 until data_size)) translated_data(i) = char_to_ix(unit(training_data).charAt(i))
        val translated_data = NewArray[Int](data_size)
        for (i <- (0 until data_size)) { translated_data(i) = Encoding.char_to_ix(training_data(i)) }

        val vocab_size = 26                 // Do we have to get this size?
        val hidden_size = 10
        val learning_rate = 1e-1
        val seq_length = 10
        //val Wxh = Vector.randinit(vocab_size, hidden_size, 0.01)  // input to hidden
        val Wxh = Vector.randinit(vocab_size, hidden_size, 0.01)  // input to hidden
        val Whh = Vector.randinit(hidden_size, hidden_size, 0.01) // hidden to hidden
        val Why = Vector.randinit(hidden_size, vocab_size, 0.01)  // hidden to output
        val bh  = Vector.zeros(hidden_size)
        val by  = Vector.zeros(vocab_size)
        val hprev = Vector.zeros(hidden_size) 

        //val hnext = Vector.zeros_like(hprev)

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
          val outputs = LOOPCCM(in)(inputs.length){i => t => 
            
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
          hprev1.x.copy_data(outputs(1).x)     // update the hidden state with the result from LOOP
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

        var startAt = -seq_length

        for (n <- (0 until 2001): Rep[Range]) {

          if (startAt + seq_length + 1 >= data_size) {
            startAt = 0
            hprev.clear()
          } else startAt += seq_length

          val inputs = NewArray[Int](seq_length)
          val targets = NewArray[Int](seq_length)
          for (i <- (0 until seq_length): Rep[Range]) {
            inputs(i) = translated_data(startAt+i)
            targets(i) = translated_data(startAt+i+1)
          }

          val loss = gradR_loss(lossFun(inputs, targets))(Vector.zeros(1)) 
          val loss_value = loss.data(0) // we suppose the loss is scala (Vector of size 1)
          if (n % 100 == 0) {
          //  loss.print()
            println(s"iter $n, loss $loss_value") // FIXME loss need to be fixed
          }

          /*
          Wxh1.d.print()  
          Whh1.d.print()
          Why1.d.print()  
          bh1.d.print()
          by1.d.print()
          hprev1.x.print()    
          */
          val pars = ArrayBuffer(Wxh1, Whh1, Why1, bh1, by1)
          val mems = ArrayBuffer(mWxh, mWhh, mWhy, mbh, mby)
          for ((par, mem) <- pars.zip(mems)) {
            mem += par.d * par.d
            par.x -= par.d * lr / (mem + hp).sqrt()
            par.clear_grad()
          }
          hprev1.clear_grad()          // clear gradient of all Tensors for next cycle
          //hprev1.x.copy_data(hnext)
          
          resetMallocAddr(addr)  // reset malloc_addr to the value when we remember allocation pointer
        }

      }
    }

    
    //println("try array2_2_3")
    val array2_2_3_file = new PrintWriter(new File("array2_2_3.cpp"))
    array2_2_3_file.println(array2_2_3.code)
    array2_2_3_file.flush()
    array2_2_3.eval("abc")
    //println("verified that in this small example the values of gradients are about right (up to precision)")

    val array2_2_4Debug = new DslDriverC[String, Unit] with VectorExp {
      def snippet (a: Rep[String]): Rep[Unit] = {
        val vocab_size = 3
        val by  = Vector.zeros(vocab_size)
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
        by1.d.print()
      }
    }
    //println("try array2_2_4")
    //array2_2_4Debug.eval("abc")

    val array2_2_5Debug = new DslDriverC[String, Unit] with VectorExp {
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
        e1.d.print()
        a1.d.print()
      }
    }
    //println("try array2_2_5")
    //array2_2_5Debug.eval("abc")


    val array2_3 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {

        val training_data = "abcdefghijklmn"
        val data_size = training_data.length
        val chars = training_data.distinct
        val vocab_size = chars.length
        println(s"data has $data_size chars and $vocab_size unique chars")

        import scala.collection.immutable.Map
        val char_to_ix = (chars zip (0 until vocab_size)).foldLeft(Map.empty[Char, Int]) {
          (m, c_i) => m.updated(c_i._1, c_i._2) 
        }
        val ix_to_char = (chars zip (0 until vocab_size)).foldLeft(Map.empty[Int, Char]) {
          (m, c_i) => m.updated(c_i._2, c_i._1)
        }

        //val translated_data = NewArray[Int](data_size)
        //for (i <- (0 until data_size)) translated_data(i) = char_to_ix(unit(training_data).charAt(i))
        val translated_data = training_data.map(char_to_ix).toArray 

        val hidden_size = 100 // size of hidden layer of neurons
        val seq_length = 25   // number of steps to unroll the RNN for
        val learning_rate = 1e-1

        // model parameters
        val Wxh = Vector.randinit(vocab_size, hidden_size, 0.01)  // input to hidden
        val Whh = Vector.randinit(hidden_size, hidden_size, 0.01) // hidden to hidden
        val Why = Vector.randinit(hidden_size, vocab_size, 0.01)  // hidder to output
        val bh  = Vector.zeros(hidden_size)                       // hidden bias
        val by  = Vector.zeros(vocab_size)                        // output bias
        val hprev = Vector.zeros(hidden_size)                     // initial hidden state

        // wrap the parameters as Tensors, use them in model
        val Wxh1 = TensorR.Tensor(Wxh)
        val Whh1 = TensorR.Tensor(Whh)
        val Why1 = TensorR.Tensor(Why)
        val bh1  = TensorR.Tensor(bh)
        val by1  = TensorR.Tensor(by)
        var hprev1 = TensorR.Tensor(hprev) // this is not paramters but need to be carried on in iterations

        // define model (loss function)
        def lossFun(inputs: Rep[Array[Int]], targets: Rep[Array[Int]]) = { (dummy: TensorR) => 
          
          var loss = TensorR.Tensor(Vector.zeros(1))
          val in = ArrayBuffer[TensorR]()
          in.append(loss)
          in.append(hprev1)
          val outputs = LOOPCCM(in)(inputs.length){i => t => 
          
            // get input as one-hot tensor
            val x = Vector.zeros(vocab_size)
            x.data(inputs(i)) = 1
            val x1 = TensorR.Tensor(x) 
            // get output as one-hot tensor
            val y = Vector.zeros(vocab_size)
            y.data(targets(i)) = 1
            val y1 = TensorR.Tensor(y)

            val h1 = ((Wxh1 dot x1) + (Whh1 dot hprev1) + bh1).tanh() // hidden state
            val e1 = ((Why1 dot h1) + by1).exp()                      // unnormalized prob
            val p1 = e1 / e1.sum()                                    // normalized prob
            
            val newloss = t(0) - (p1 dot y1).log()                    // loss is updated by original loss t(0) and additional loss
            val out = ArrayBuffer[TensorR]()
            out.append(newloss)
            out.append(h1)
            out
          }

          hprev1.x.copy_data(outputs(1).x)  // update the hidden state with the result from LOOP
          outputs(0)                        // return the final loss
        }

        // the learning cycle starts here
        // var n = 0
        var p = 0
        val mWxh = Vector.zeros_like(Wxh)
        val mWhh = Vector.zeros_like(Whh)
        val mWhy = Vector.zeros_like(Why)
        val mbh  = Vector.zeros_like(bh)
        val mby  = Vector.zeros_like(by)
        var smooth_loss = - scala.math.log(1.0 / vocab_size) * seq_length

        val lr = Vector.consts(1, value = learning_rate)
        val hp = Vector.consts(1, value = 1e-8)

        for (n <- (0 until 3): Rep[Range]) {
          if (p + seq_length + 1 >= data_size) {
            hprev1.clear_all() // clear the value and the gradient to reset RNN memory
            p = 0              // go to the start of training data
          }

          val inputs = NewArray[Int](seq_length)
          val targets = NewArray[Int](seq_length)
          for (i <- (0 until seq_length): Rep[Range]) {
            inputs(i) = i
            targets(i) = (i+1) % seq_length
          }

          // no sample so far

          val dummy = gradR(lossFun(inputs, targets))(Vector.zeros(1)) 
          // need to track loss but the current gradR function discard loss. Need helper function to save it
          // update and show smooth_loss

          if (n % 100 == unit(0)) {
            println(s"iter $n, loss 0.99") // FIXME loss need to be fixed
          }

          // update parameters based on gradient
          val pars = ArrayBuffer(Wxh1, Whh1, Why1, bh1, by1)
          val mems = ArrayBuffer(mWxh, mWhh, mWhy, mbh, mby)
          for ((par, mem) <- pars.zip(mems)) {
            mem += par.d * par.d
            par.x -= par.d * lr / (mem + hp).sqrt()
            par.clear_grad()
          }
          hprev1.clear_grad()          // clear gradient of all Tensors for next cycle
          
          p += seq_length
          //n += 1
        }

      }
    }

    //println(array2_3.code)
    //array2_3.eval("abc")

    val array3 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        // use random array as input
        val length = 2
        val v = Vector.randinit(length)
        v.print()

        // calcuate gradient
        val grad = gradR(t => {val y = IF (length)(t.x.data(0) > 0.0) {t + t}{t * t}
                     y.sum() })(v)
        // show gradient
        grad.print()
      }
    }

    //println("test IF gradient")
    //val array3_file = new PrintWriter(new File("array3.cpp"))
    //array3_file.println(array3.code)
    //array3_file.flush()
    //array3.eval("abc")

    val array4 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        // use random array as input
        val length = 2
        val v = Vector.randinit(length)
        v.print()

        val half = (new TensorR(Vector.halves(length), Vector.zeros(length)))
        // calculate gradient
        val grad = gradR(t => {val y = LOOP(t)(t => t.x.data(0) > 0.1)(t => t * half)
                     y.sum() })(v)
        // show gradient
        grad.print()
        println("Tensor in closure can also accumulate gradient, which is important")
        half.d.print()
      }
    }

    // println("test LOOP gradient")
    //println(array4.code)
    //val p = new PrintWriter(new File("fei_needs_help_for_basic_java_thing.cpp"))
    //p.println(array4.code)
    //p.flush()
    //array4.eval("abc" )

    val array4_1 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Vector.randinit(length)
        v.print()

        val half = new TensorR(Vector.halves(length), Vector.zeros(length))
        val grad = gradR(t => {
            val y = LOOPC(t)(3)(t => t * half )
            y.sum()
          })(v)
        // show gradient
        grad.print()
        println("Tensor in closure can also accumulate gradient, which is important")
        half.d.print()
      }
    }

    // println("test LOOP gradient")
    //array4_1.eval("abc")

    val array4_2 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Vector.randinit(length)
        v.print()

        val half = new TensorR(Vector.halves(length), Vector.zeros(length))
        val grad = gradR(t => {
          val y = LOOPCC(t)(3)(i => t => {
            println(i)
            t * half})
          y.sum()})(v)
        // show gradient
        grad.print()
        println("Tensor in closure can also accumulate gradient, which is important")
        half.d.print()
      }
    }

    //array4_2.eval("abc")

    // test using array data by closure
    val array4_2_1 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        
        // random initialization
        val length = 3
        val v = Vector.randinit(length)
        v.print()

        // get data from "file" (more like generate static data and lift it to Rep type)
        val A = scala.Array
        val dat1 = A(0.9, 0.8, 0.7)
        val dat2 = A(0.1, 0.2, 0.3)
        val ddim0 = 2
        val ddim1 = dat1.length 
        // lift it to RepArray (staticData not working for c code generation in 2d array)
        // val dat = staticData(da)
        val data1 = NewArray[Double](ddim1)
        val data2 = NewArray[Double](ddim1)
        for (i <- (0 until ddim1): Rep[Range]) {
          data1(i) = (i + 1)
          data2(i) = (i + 1) * 2
        }
        val data = NewArray[Array[Double]](ddim0)
        data(0) = data1; data(1) = data2

        val model: TensorR => TensorR @diff = { (x: TensorR) =>
          val y = LOOPCC(x)(ddim0)(i => x1 => {
            val data_point = TensorR.Tensor(new Vector(data(i), ddim1))
            x1 * data_point
            })
          y.sum()
        }

        val grad = gradR(model)(v)
        // show gradient
        grad.print() 
      }
    }

    //println(array4_2_1.code)
    //val array4_2_1_file = new PrintWriter(new File("array4_2_1.cpp"))
    //array4_2_1_file.println(array4_2_1.code)
    //array4_2_1_file.flush()
    //array4_2_1.eval("abc")

/*
    val array4_2_2 = new DslDriverC[String, Unit] with VectorExp {
      def snippet(a: Rep[String]): Rep[Unit] = {
        
        val hidden_size = 3

        // tree that is flatted to array (should be rep type in practise)
        /*   node4
             /   \
            node3 \
            / \    \
           0  1    2 
        */
        val inputs  = staticData(scala.Array(0, 1, 2))          // this is the leaves (data) in tree
        val leftch  = staticData(scala.Array(-1, -1, -1, 0, 3)) // this is mapping from node to left child
        val rightch = staticData(scala.Array(-1, -1, -1, 1, 2)) // this is mapping from node to right child

        val Whhl = Vector.randinit(hidden_size, hidden_size, 0.01) // hidden of left child to hidden of node
        val Whhr = Vector.randinit(hidden_size, hidden_size, 0.01)  // hidden of right child to hidden of node
        
        // wrap as tensors
        val Whhl1 = TensorR.Tensor(Whhl)
        val Whhr1 = TensorR.Tensor(Whhr)
        
        def model: TensorR => TensorR @diff = (dum => {
          // initialize the ArrayBuffer of tensors for iterating the tree data
          val in = ArrayBuffer[TensorR]()
          for (i <- (0 until inputs.length): Rep[Range]) { 
            val v_temp = Vector.zeros(hidden_size)
            v_temp.data(inputs(i)) = 1 // one-hot
            in.append(TensorR.Tensor(v_temp))
          }
          for (i <- (0 until (leftch.length - inputs.length)): Rep[Range]) {
            in.append(TensorR.Tensor(Vector.zeros(hidden_size))) 
          }
          // put the "in" in the loop
          val y = LOOPCCM(in)(leftch.length - inputs.length)(i => ins => {
            ins(i) = (Whhl1 dot ins(leftch(i))) + (Whhr1 dot ins(rightch(i)))
            ins
          })
          y(leftch.length - 1).sum()
        })

        val dummy = gradR(model)(Vector.zeros(1))

        // show gradient
        Whhl1.d.print()
        Whhr1.d.print() 
      }
    }

    array4_2_2.eval("abc")
*/

/*
    val array4_2_3 = new DslDriverC[String, Unit] with VectorExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        val hidden_size = 3
        val inputs  = staticData(scala.Array(0, 1, 2))
        val leftch  = staticData(scala.Array(-1, -1, -1, 0, 3))
        val rightch = staticData(scala.Array(-1, -1, -1, 1, 2))

        val Whhl = TensorR.Tensor(Vector.randinit(hidden_size, hidden_size))
        val Whhr = TensorR.Tensor(Vector.randinit(hidden_size, hidden_size))

        def model_r(n: Rep[Int]): TensorR @diff = {
          if (leftch(n) == -1) {
            val v = Vector.zeros(hidden_size)
            v.data(inputs(n)) = 1.0
            TensorR.Tensor(v)
          } else {
            val l = model_r(leftch(n))
            val r = model_r(rightch(n))
            (Whhl dot l) + (Whhr dot r)
          }
        }
        val model: TensorR => TensorR @diff = {(dum: TensorR) => model_r(0)}
        val dummy = gradR(model)(Vector.zeros(1))

        // show gradient
        Whhl.d.print()
        Whhr.d.print() 
      }
    }

    array4_2_3.eval("abc")
*/

    val array4_3 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Vector.randinit(length)
        v.print()
        val u = Vector.randinit(length, offset = 5)
        u.print()

        val half = new TensorR(Vector.halves(length), Vector.zeros(length))
        val vv = TensorR.Tensor(v)
        val uu = TensorR.Tensor(u)
        
        val dummy = gradR(dum => {
          val in = ArrayBuffer[TensorR](vv, uu)
          val y = LOOPCC2(in)(3)(i => ins => {
            val vvv = ins(0) * half
            val uuu = ins(1) * half
            ArrayBuffer[TensorR](vvv, uuu)})
          y(1).sum() + y(0).sum()})(Vector.zeros(1))
        // show gradient
        println("Tensor in closure can also accumulate gradient, which is important")
        half.d.print()
        vv.d.print()
        uu.d.print()
      }
    }

    // println("support 2 tensors in loop")
    //println(array4_3.code)
    //array4_3.eval("abc")

    val array4_4 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Vector.randinit(length)
        v.print()
        val u = Vector.randinit(length, offset = 5)
        u.print()

        val half = new TensorR(Vector.halves(length), Vector.zeros(length))
        val vv = TensorR.Tensor(v)
        val uu = TensorR.Tensor(u)
        
        val dummy = gradR(dum => {
          val in = ArrayBuffer[TensorR](vv, uu)
          val y = LOOPCCM(in)(3)(i => ins => {
            val vvv = ins(0) * half
            val uuu = ins(1) * half
            ArrayBuffer[TensorR](vvv, uuu)
            })
          y(1).sum() + y(0).sum()})(Vector.zeros(1))
        // show gradient
        println("Tensor in closure can also accumulate gradient, which is important")
        half.d.print()
        vv.d.print()
        uu.d.print()
      }
    }

    //println("support 2 tensors in loop using LOOPCCM")
    //println(array4_4.code)
    //val array4_4_file = new PrintWriter(new File("array4_4.cpp"))
    //array4_4_file.println(array4_4.code)
    //array4_4_file.flush()
    //array4_4.eval("abc")

    val array5 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Vector.randinit(length)
        v.print()

        val grad = gradR(t => (t * t).sum())(v)
        grad.print()
      }
    }

    //println("test elementwise multiplication")
    //println(array5.code)
    //array5.eval("abc")

    val array6 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Vector.randinit(length)
        v.print()

        val grad = gradR(t => (t / t).sum())(v)
        grad.print()
      }
    }

    // println("test elementwise division")
    //println(array6.code)
    //array6.eval("abc")

    val array7 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Vector.randinit(length)
        v.print()

        val grad = gradR(t => (t.tanh()).sum())(v)
        grad.print()
      }
    }

    // println("test tanh")
    //println(array7.code)
    //array7.eval("abc")

    val array8 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Vector.randinit(length)
        v.print()

        val grad = gradR(t => (t.exp()).sum())(v)
        grad.print()
      }
    }

    // println("test exp")
    //println(array8.code)
    //array8.eval("abc")

    val array9 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Vector.randPositive(length)
        v.print()

        val grad = gradR(t => (t.log()).sum())(v)
        grad.print()
      }
    }

    //println("test log")
    // println(array9.code)
    //array9.eval("abc")

    val array10 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Vector.randinit(length)
        v.print()

        val A = scala.Array
        //val arr1 = A(4.0, 3.0)
        //val arr2 = A(1.5, 2.0)
        // /val arra = mutableStaticData(A(mutableStaticData(arr1), mutableStaticData(arr2)))
        //val arr = A(A(4.0, 3.0), A(1.5, 2.0))
        //val arra = mutableStaticData(arr)

        val arra = NewArray[Array[Double]](2)
        //arra(0) = mutableStaticData(A(4.0, 3.0))
        //arra(1) = mutableStaticData(A(1.5, 2.0))
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
        grad.print()
      }
    }

    //println(array10.code)
    //array10.eval("abc")

    val array11 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Vector.randinit(length)
        v.print()

        val A = scala.Array
        val arra = NewArray[Array[Double]](3)
        arra(0) = NewArray[Double](2)
        arra(0)(0) = 5.0; arra(0)(1) = 4.0
        arra(1) = NewArray[Double](2)
        arra(1)(0) = 3.0; arra(1)(1) = 2.0
        arra(2) = NewArray[Double](2)
        arra(2)(0) = 1.5; arra(2)(1) = 1.4
        val lch1 = NewArray[Int](3)
        lch1(0) = 1; lch1(1) = 100; lch1(2) = 100
        val rch1 = NewArray[Int](3)
        rch1(0) = 2; rch1(1) = 100; rch1(2) = 100
        
        // create a model that recursively use the data (originated from tree)
        def model: TensorR => TensorR @diff = { (x: TensorR) =>
          LOOPT(x)(arra.length, lch1, rch1){ (l: TensorR, r: TensorR, i: Rep[Int]) =>
            l * r * new TensorR(new Vector(arra(i), length), Vector.zeros(length))
          }
        }

        val grad = gradR(t => model(t).sum())(v)
        grad.print()
      }
    }

    //println(array11.code)
    //array11.eval("abc")


  }
}

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

      Note:

        We are currently only very narrowly supporting matrix (2d vectors)
        We only support Matrix vector multiplication, which is like several vector_vector dot product
        Matrix has >1 dim1 field and number of values dim0 * dim1
        but the current implementation silently ignore the 2:end columns unless it is dot product
        The idea of thinking Matrix row as dim0 and colume as dim1 is not the common way, but we are going by it for now because
        we want to simplify the implementation and just borrow the logic of dot
  
    **/

    class Vector(val data: Rep[Array[Double]], val dim0: Int, val dim1:Int = 1 /*, val dim2: Int*/) extends Serializable {

      def foreach(f: Rep[Double] => Rep[Unit]): Rep[Unit] =
        for (i <- (0 until dim0):Rep[Range]) f(data(i))

      @virtualize
      def sumIf(f: Rep[Double] => Rep[Boolean]) = { 
        val n = var_new(0.0); 
        foreach(x => if (f(x)) n += x); 
        readVar(n) 
      }

      def + (that: Vector) = {
        val dimM = if (dim0 > that.dim0) dim0 else that.dim0
        val res = NewArray[Double](dimM)
        if (that.dim0 == 1) for (i <- (0 until dimM): Rep[Range]) res(i) = data(i) + that.data(0)
        else if (dim0 == 1) for (i <- (0 until dimM): Rep[Range]) res(i) = data(0) + that.data(i)
        else if (dim0 == that.dim0) for (i <- (0 until dimM): Rep[Range]) res(i) = data(i) + that.data(i)
        else throw new IllegalArgumentException("dimensions of vector do not match +!")
        new Vector(res, dimM)
      }

      // this operator updates the values of this, unlike the + operator
      def += (that: Vector) = {
        if (dim0 == that.dim0) for (i <- (0 until dim0): Rep[Range]) data(i) += that.data(i)
        else if (that.dim0 == 1) for (i <- (0 until dim0): Rep[Range]) data(i) += that.data(0)
        else if (dim0 == 1) for (i <- (0 until that.dim0): Rep[Range]) data(0) += that.data(i)
        //throw new IllegalArgumentException("dimensions needs to be expanded!")
        else throw new IllegalArgumentException("dimensions of vector do not match +=!")
      }

      def - (that: Vector) = {
        val dimM = if (dim0 > that.dim0) dim0 else that.dim0
        val res = NewArray[Double](dimM)
        if (that.dim0 == 1) for (i <- (0 until dimM): Rep[Range]) res(i) = data(i) - that.data(0)
        else if (dim0 == 1) for (i <- (0 until dimM): Rep[Range]) res(i) = data(0) - that.data(i)
        else if (dim0 == that.dim0) for (i <- (0 until dimM): Rep[Range]) res(i) = data(i) - that.data(i)
        else throw new IllegalArgumentException("dimensions of vector do not match -!")
        new Vector(res, dimM)
      }

      // this operator updates the values of this, unlike the - operator
      def -= (that: Vector) = {
        if (dim0 == that.dim0) for (i <- (0 until dim0): Rep[Range]) data(i) -= that.data(i)
        else if (that.dim0 == 1) for (i <- (0 until dim0): Rep[Range]) data(i) -= that.data(0)
        else if (dim0 == 1) for (i <- (0 until that.dim0): Rep[Range]) data(0) -= that.data(i)
        //throw new IllegalArgumentException("dimensions needs to be expanded!")
        else throw new IllegalArgumentException("dimensions of vector do not match -=!")
      }

      // element wise multiplication
      def * (that: Vector) = {
        val dimM = if (dim0 > that.dim0) dim0 else that.dim0
        val res = NewArray[Double](dimM)
        if (that.dim0 == 1) for (i <- (0 until dimM): Rep[Range]) res(i) = data(i) * that.data(0)
        else if (dim0 == 1) for (i <- (0 until dimM): Rep[Range]) res(i) = data(0) * that.data(i)
        else if (dim0 == that.dim0) for (i <- (0 until dimM): Rep[Range]) res(i) = data(i) * that.data(i)
        else throw new IllegalArgumentException("dimensions of vector do not match *!")
        new Vector(res, dimM)
      }

      // this operator updates the values of this, unlike * operator
      def *= (that: Vector) = {
        if (dim0 == that.dim0) for (i <- (0 until dim0): Rep[Range]) data(i) *= that.data(i)
        else if (that.dim0 == 1) for (i <- (0 until dim0): Rep[Range]) data(i) *= that.data(0)
        else if (dim0 == 1) throw new IllegalArgumentException("dimensions needs to be expanded *=!")
        else throw new IllegalArgumentException("dimensions of vector do not match *=!")
      }

      // element wise division
      def / (that: Vector) = {
        val dimM = if (dim0 > that.dim0) dim0 else that.dim0
        val res = NewArray[Double](dimM)
        if (that.dim0 == 1) for (i <- (0 until dimM): Rep[Range]) res(i) = data(i) / that.data(0)
        else if (dim0 == 1) for (i <- (0 until dimM): Rep[Range]) res(i) = data(0) / that.data(i)
        else if (dim0 == that.dim0) for (i <- (0 until dimM): Rep[Range]) res(i) = data(i) / that.data(i)
        else throw new IllegalArgumentException("dimensions of vector do not match /!")
        new Vector(res, dimM)
      }

      // this operator updates the values of this, unlike / operator
      def /= (that: Vector) = {
        if (dim0 == that.dim0) for (i <- (0 until dim0): Rep[Range]) data(i) /= that.data(i)
        else if (that.dim0 == 1) for (i <- (0 until dim0): Rep[Range]) data(i) /= that.data(0)
        else if (dim0 == 1) throw new IllegalArgumentException("dimensions needs to be expanded /=!")
        else throw new IllegalArgumentException("dimensions of vector do not match /=!")
      }

      def setAsOne() = {
        for (i <- (0 until dim0 * dim1): Rep[Range]) data(i) = 1.0
      }

      def clear() = {
        for (i <- (0 until dim0 * dim1): Rep[Range]) data(i) = 0.0
      }

      def copy_data(that: Vector) = {
        if (dim0 * dim1 != that.dim0 * that.dim1) throw new IllegalArgumentException("dimensions of vector do not match copy_data!")
        for (i <- (0 until dim0 * dim1): Rep[Range]) data(i) = that.data(i)
      }

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

      def tanh() = {
        val res = NewArray[Double](dim0)
        for (i <- (0 until dim0): Rep[Range]) {
          res(i) = Math.tanh(data(i)) // need fix, MathOps C code gene is not supporting tanh
        }
        new Vector(res, dim0)
      }

      def exp() = {
        val res = NewArray[Double](dim0)
        for (i <- (0 until dim0): Rep[Range]) {
          res(i) = Math.exp(data(i))
        }
        new Vector(res, dim0)
      }

      def log() = {
        val res = NewArray[Double](dim0)
        for (i <- (0 until dim0): Rep[Range]) {
          res(i) = Math.log(data(i))
        }
        new Vector(res, dim0)
      }

      def sqrt() = {
        val res = NewArray[Double](dim0 * dim1)
        for (i <- (0 until dim0 * dim1): Rep[Range]) res(i) = Math.sqrt(data(i))
        new Vector(res, dim0, dim1)
      }

      def sum() = {
        val value = var_new(0.0)
        for (i <- (0 until dim0): Rep[Range]) {
          value += data(i)
        }
        val res = NewArray[Double](1)
        res(0) = readVar(value)
        new Vector(res, 1)
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

      def fromData(x: Double*) = {
        val y = x.toArray
        val res = NewArray[Double](y.length)
        for (i <- (0 until y.length): Rep[Range]) res(i) = y(i)
        new Vector(res, y.length)
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

/*
    @virtualize
    def LOOPA(init: TensorR)(a: Rep[Array[Array[Double]]])(b: Rep[Int] => TensorR => TensorR @diff): TensorR @diff = shift { k: (TensorR => Unit) =>
      var gc = 0
      val bound = a.length
      lazy val loop: TensorR => Unit = FUN (init.x.dim0){ (x : TensorR) =>
        if (gc < bound) {gc += 1; RST(loop(b(gc-1)(x)))} else RST(k(x))
      }
      loop(init)
    }
*/

    def gradR(f: TensorR => TensorR @diff)(x: Vector): Vector = {
      val x1 = new TensorR(x, Vector.zeros(x.dim0))
      reset { val y = f(x1)
          y.d.setAsOne()
          y.x.print() // this is the result of forward propagation (likely the loss)
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

    def getMallocAddr() = {
      unchecked[Unit]("waterMark = mallocAddr")
    }

    def resetMallocAddr() = {
      unchecked[Unit]("mallocAddr = waterMark")
    }

    @virtualize
    def doIf(b: Rep[Boolean])(a: => Rep[Unit]) = {
      if(b) a
    }
  }


  def main(args: Array[String]): Unit = {
    import java.io.PrintWriter;
    import java.io.File;    

    val array1 = new DslDriverC[String, Unit]  with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val res = Vector.randinit(length)
        getMallocAddr()
        val res2 = Vector.randPositive(length)
        res.print()
        res2.print()
        
        val result = res dot res2
        result.print()
        resetMallocAddr()

        // still not working
        for (i <- (0 until 10): Rep[Range]) {
          doIf(i == unit(2)){println("found")}
          //if (equals(i, unit(2))) println("found")
        }
      }
    }

    //println("test dot")
    //val array1_file = new PrintWriter(new File("array1(2).cpp"))
    //array1_file.println(array1.code)
    //array1_file.flush()
    //println(array1.code)
    //array1.eval("abc")

    val array1_1 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val dim0 = 2
        val dim1 = 3
        val matrix = Vector.randinit(dim0, dim1)
        val vector = Vector.randPositive(dim0)
        matrix.print()
        vector.print()

        println("the result is:")
        val result = matrix dot vector
        result.print()
      }
    }

    //println(array1_1.code)
    //array1_1.eval("abc")

    val array2 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        // read training data from file (for now just use random)
        val length = 2
        val v = Vector.randinit(length)
        v.print()   

        // calculate gradient
        val grad = gradR(t => t dot t)(v)
        // show gradient
        println("show gradient in the traditional way")
        grad.print()

        // construct TensorR for closure
        val tv = TensorR.Tensor(v)
        val loss = gradR_loss(dummy => tv dot tv)(Vector.zeros(1))
        println("gradient:")
        tv.d.print()
        println("loss")
        loss.print()
      }
    }

    //println("test dot gradient")
    //println(array2.code)
    //array2.eval("2.0")

    val array2_1 = new DslDriverC[String, Unit] with VectorExp {
      // update gradient as side effect
      
      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Vector.randinit(length)
        v.print()

        // initialize tensor for closure
        val t = new TensorR(v, Vector.zeros(length))            
        // call grad_side_effect_using_closure
        val dummy = gradR(dummy => t dot t)(Vector.zeros(1))
        // print the gradient of t
        t.x.print()
        t.d.print()
      }
    }

    //println("test dot gradient as side effect")
    //println(array2_1.code)
    //array2_1.eval("2.0")

    val array2_1_1 = new DslDriverC[String, Unit] with VectorExp {
      // update gradient as side effect
      
      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 3
        val v = Vector.randinit(length)
        v.print()

        // initialize tensor for closure
        var t = new TensorR(v, Vector.zeros(length))            
        val half = new TensorR(Vector.halves(length), Vector.zeros(length))
        // call grad_side_effect_using_closure
        val dummy = gradR(dummy => {
          t = (t dot t)
          t = (half * t).sum()
          t
          })(Vector.zeros(1))
        // print the gradient of t
        t.d.print()
        half.d.print()
      }
    }

    //println("test dot gradient as side effect with var update") 
    //println("proving that I can use var update without creating cycles in static computation graph")
    //println(array2_1.code)
    //array2_1_1.eval("2.0")

    val array2_2 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {

        val dim0 = 2
        val dim1 = 3
        val matrix = Vector.randinit(dim0, dim1)
        val vector = Vector.randPositive(dim0)
        matrix.print()
        vector.print()

        // initialize tensors for closure
        val ma = new TensorR(matrix, Vector.zeros(dim0, dim1))
        val ve = new TensorR(vector, Vector.zeros(dim0))
        // define function of model
        def model(dummy: TensorR): TensorR @diff = {
          (ma dot ve).sum()
        }
        val dummy = gradR(model)(Vector.zeros(1))
        // print the gradient of ma and ve
        ma.d.print()
        ve.d.print()
      }
    }

    // println("test matrix vector dot gradient as side effect")
    //println(array2_2.code)
    //array2_2.eval("abc")

    val array2_2_1 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val vocab_size = 3
        val hidden_size = 2
        val Wxh = Vector.randinit(vocab_size, hidden_size, 0.01)  // input to hidden
        val Whh = Vector.randinit(hidden_size, hidden_size, 0.01) // hidden to hidden
        val Why = Vector.randinit(hidden_size, vocab_size, 0.01)  // hidden to output
        val hprev = Vector.zeros(hidden_size) 

        // wrap as tensors
        val Wxh1 = TensorR.Tensor(Wxh)
        val Whh1 = TensorR.Tensor(Whh)
        val Why1 = TensorR.Tensor(Why)
        val hprev1 = TensorR.Tensor(hprev)

        def lossFun = { (dummy: TensorR) =>
          var loss = TensorR.Tensor(Vector.zeros(1))
          LOOPCC(loss)(3){i => t => 
            
            // printf("at iteration %d ", i)
            // get input as one-hot tensor
            val x = Vector.zeros(vocab_size)
            x.data(i) = 1
            val x1 = TensorR.Tensor(x)
            // get output as one-hot tensor
            val y = Vector.zeros(vocab_size)
            y.data((i+1)%vocab_size) = 1
            val y1 = TensorR.Tensor(y)

            val h1 = ((Wxh1 dot x1) + (Whh1 dot hprev1)).tanh() // hidden state
            // carefully update hprev1 with h1 in the last cycle
            /*
              I have evidence that in this case, the hprev1.x.data has been updated, and Whh1 carried some gradient
              However, this seems wrong, because in this case, the gradient is not flowing through the hidden vector
            */
            hprev1.x.copy_data(h1.x)
            
            val e1 = (Why1 dot h1).exp()
            val p1 = e1 / e1.sum()
            t - (p1 dot y1).log() // loss is updated by this value (t is loss from last cycle)
          }
        }
        val dummy = gradR(lossFun)(Vector.zeros(1)) 
        
        Wxh1.d.print()  
        Whh1.d.print()
        Why1.d.print()  
        hprev1.x.print()    

      }
    }

    /*
    println("try array2_2_1")
    val p = new PrintWriter(new File("array2_2_1.cpp"))
    p.println(array2_2_1.code)
    p.flush()
    array2_2_1.eval("abc")
    */

    val array2_2_2 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val vocab_size = 3
        val hidden_size = 2
        val Wxh = Vector.randinit(vocab_size, hidden_size, 0.01)  // input to hidden
        val Whh = Vector.randinit(hidden_size, hidden_size, 0.01) // hidden to hidden
        val Why = Vector.randinit(hidden_size, vocab_size, 0.01)  // hidden to output
        val hprev = Vector.zeros(hidden_size) 

        // wrap as tensors
        val Wxh1 = TensorR.Tensor(Wxh)
        val Whh1 = TensorR.Tensor(Whh)
        val Why1 = TensorR.Tensor(Why)
        val hprev1 = TensorR.Tensor(hprev)

        def lossFun = { (dummy: TensorR) =>
          // what about the idea of accumulating loss as a var closure
          // NO: Stack overflow!!
          // what about saving the loss of each step, then add up at the end
          import scala.collection.mutable.MutableList
          val losses = new MutableList[TensorR]()


          var loss = TensorR.Tensor(Vector.zeros(1))
          val h_new = LOOPCC(hprev1)(3){i => t => 
            
            // printf("at iteration %d ", i)
            // get input as one-hot tensor
            val x = Vector.zeros(vocab_size)
            x.data(i) = 1
            val x1 = TensorR.Tensor(x)
            // get output as one-hot tensor
            val y = Vector.zeros(vocab_size)
            y.data((i+1)%vocab_size) = 1
            val y1 = TensorR.Tensor(y)

            val h1 = ((Wxh1 dot x1) + (Whh1 dot t)).tanh() // hidden state
            // carefully update hprev1 with h1 in the last cycle
            /*
              I have evidence that in this case, the hprev1.x.data has been updated, and Whh1 carried some gradient
              However, this seems wrong, because in this case, the gradient is not flowing through the hidden vector
            */
            // hprev1.x.copy_data(h1.x)
            
            val e1 = (Why1 dot h1).exp()
            val p1 = e1 / e1.sum()
            // t - (p1 dot y1).log() // loss is updated by this value (t is loss from last cycle)
            // loss -= (p1 dot y1).log()
            losses += (p1 dot y1).log() // FIXME, need -
            h1
          }
          hprev1.x.copy_data(h_new.x)
          
          // another loop to collect losses
          LOOPCC(loss)(3){i => t =>
            // t + losses(i) // Problem!! :: the losses(i) became Rep[TensorR]
            t + t // changed this just to compile
          }
        }
        val dummy = gradR(lossFun)(Vector.zeros(1)) 
        
        Wxh1.d.print()  
        Whh1.d.print()
        Why1.d.print()  
        hprev1.x.print()    

      }
    }

    // println("try array2_2_2")
    //val p = new PrintWriter(new File("array2_2_2.cpp"))
    //p.println(array2_2_2.code)
    //p.flush()
    // array2_2_2.eval("abc")

    val array2_2_3 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val vocab_size = 30
        val hidden_size = 100
        val learning_rate = 1e-1
        val seq_length = 20
        //val Wxh = Vector.randinit(vocab_size, hidden_size, 0.01)  // input to hidden
        val Wxh = Vector.randinit(vocab_size, hidden_size, 0.01)  // input to hidden
        val Whh = Vector.randinit(hidden_size, hidden_size, 0.01) // hidden to hidden
        val Why = Vector.randinit(hidden_size, vocab_size, 0.01)  // hidden to output
        val bh  = Vector.zeros(hidden_size)
        val by  = Vector.zeros(vocab_size)
        val hprev = Vector.zeros(hidden_size) 

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
          hprev1.x.copy_data(outputs(1).x)  // update the hidden state with the result from LOOP
          outputs(0)                        // return the final loss
        }


        val lr = Vector.consts(1, value = learning_rate)
        val hp = Vector.consts(1, value = 1e-8)

        val mWxh = Vector.zeros_like(Wxh)
        val mWhh = Vector.zeros_like(Whh)
        val mWhy = Vector.zeros_like(Why)
        val mbh  = Vector.zeros_like(bh)
        val mby  = Vector.zeros_like(by)

        getMallocAddr() // remember current allocation pointer here

        for (n <- (0 until 100): Rep[Range]) {

          val inputs = NewArray[Int](seq_length)
          val targets = NewArray[Int](seq_length)
          for (i <- (0 until seq_length): Rep[Range]) {
            inputs(i) = (i + n) % vocab_size 
            targets(i) = (i + 1 + n) % vocab_size
          }

          val loss = gradR_loss(lossFun(inputs, targets))(Vector.zeros(1)) 
          val loss_value = loss.data(0) // we suppose the loss is scala (Vector of size 1)
          //if (n % 100 == unit(0)) {
            loss.print()
          //  println(s"iter $n, loss $loss_value") // FIXME loss need to be fixed
          //}

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
          
          resetMallocAddr()  // reset malloc_addr to the value when we remember allocation pointer
        }

      }
    }

    
    //println("try array2_2_3")
    //val array2_2_3_file = new PrintWriter(new File("array2_2_3.cpp"))
    //array2_2_3_file.println(array2_2_3.code)
    //array2_2_3_file.flush()
    //array2_2_3.eval("abc")
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

    val array4_2_1 = new DslDriverC[String, Unit] with VectorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        
        // random initialization
        val length = 3
        val v = Vector.randinit(length)
        v.print()

        // get data from "file" (more like generate static data and lift it to Rep type)
        val A = scala.Array
        val dat = A(A(0.9, 0.8, 0.7),
                    A(0.1, 0.2, 0.3))
        val ddim0 = dat.length
        val ddim1 = dat(0).length 
        // lift it to RepArray (not working for c code generation)
        //val data = staticData(dat)
        val data1 = NewArray[Double](ddim1)
        val data2 = NewArray[Double](ddim1)
        for (i <- (0 until ddim1): Rep[Range]) {
          data1(i) = dat(0)(i)
          data2(i) = dat(1)(i)
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
    val array4_2_1_file = new PrintWriter(new File("array4_2_1.cpp"))
    array4_2_1_file.println(array4_2_1.code)
    array4_2_1_file.flush()
    array4_2_1.eval("abc")

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
            ArrayBuffer[TensorR](vvv, uuu)})
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

  }
}

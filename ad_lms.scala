import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

object LMS {

  trait DiffApi extends Dsl {

    type diff = cps[Unit]

    type RDouble = Rep[Double]

    class NumF(val x: RDouble, val d: RDouble = 1.0) {
      def +(that: NumF) = 
        new NumF(this.x + that.x, this.d + that.d)
      def *(that: NumF) =
        new NumF(this.x * that.x, this.d * that.x + that.d * this.x)
      override def toString = (x,d).toString
    }
    class NumFF(val x: RDouble, val d: NumF) {
      def +(that: NumFF) = 
        new NumFF(this.x + that.x, this.d + that.d)
      def *(that: NumFF) =
        new NumFF(this.x * that.x, 
          this.d * new NumF(that.x, that.d.x) + that.d * new NumF(this.x, this.d.x))
      override def toString = (x,d).toString
    }

    class NumR(val x: RDouble, val d: Var[Double]) extends Serializable {
      def +(that: NumR): NumR @diff = shift { (k: NumR => Unit) => 
        val y = new NumR(x + that.x, var_new(0.0)); k(y)
        this.d += y.d; that.d += y.d }
      def *(that: NumR): NumR @diff = shift { (k: NumR => Unit) => 
        // is it worth optimizing x*x --> 2*x (this == that)?
        val y = new NumR(x * that.x, var_new(0.0)); k(y)
        this.d += that.x * y.d; that.d += this.x * y.d }
    }

    // difference between static var and staged var:
    // static var won't work for nested scopes!!!
    class NumRV(val x: RDouble, var d: RDouble) {
      def +(that: NumRV): NumRV @diff = shift { (k: NumRV => Unit) => 
        val y = new NumRV(x + that.x, 0.0); k(y)
        this.d += y.d; that.d += y.d }
      def *(that: NumRV): NumRV @diff = shift { (k: NumRV => Unit) => 
        val y = new NumRV(x * that.x, 0.0); k(y)
        this.d += that.x * y.d; that.d += this.x * y.d }
    }


    // Note: we make the generated function return the accumulated deltaVar
    // and add it to the var after calling the continuation. Slightly different
    // than in the unstaged version. The main reason is that we don't (want to)
    // have NumR objects in the generated code and that we can't (easily) pass
    // a mutable var to a function with reference semantics (we could with 
    // explicit boxing, and in C/C++ we could just pass the address)
    def FUN(f: NumR => Unit): (NumR => Unit) = {
      val f1 = fun { (x:Rep[Double]) => 
        val deltaVar = var_new(0.0)
        f(new NumR(x, deltaVar))
        readVar(deltaVar)
      };
      { (x:NumR) => x.d += f1(x.x) }
    }

    def RST(a: =>Unit @diff) = continuations.reset { a; () }

    @virtualize
    def IF(c: Rep[Boolean])(a: =>NumR @diff)(b: =>NumR @diff): NumR @diff = shift { k:(NumR => Unit) =>
      val k1 = FUN(k)

      if (c) RST(k1(a)) else RST(k1(b))
    }

    @virtualize
    def LOOP(init: NumR)(c: NumR => Rep[Boolean])(b: NumR => NumR @diff): NumR @diff = shift { k:(NumR => Unit) =>
      val k1 = FUN(k)

      lazy val loop: NumR => Unit = FUN { (x: NumR) =>
        if (c(x)) RST(loop(b(x))) else RST(k1(x))
      }
      loop(init)
    }

    @virtualize
    def LOOPC(init: NumR)(c: Rep[Int])(b: NumR => NumR @diff): NumR @diff = shift { k:(NumR => Unit) =>

      var gc = 0

      lazy val loop: NumR => Unit = FUN { (x: NumR) =>
        if (gc < c) { gc += 1; RST(loop(b(x))) } else RST(k(x))
      }
      loop(init)

    }

    @virtualize
    def LOOPCC(init: NumR)(c: Rep[Int])(b: Rep[Int] => NumR => NumR @diff): NumR @diff = shift { k:(NumR => Unit) =>
      var gc = 0
      lazy val loop: NumR => Unit = FUN { (x: NumR) =>
        if (gc < c) { gc += 1; RST(loop(b(gc-1)(x))) } else RST(k(x))
      }
      loop(init)
    }

    @virtualize
    def LOOPA(init: NumR)(a: Rep[Array[Double]])(b: Rep[Int] => NumR => NumR @diff): NumR @diff = shift { k: (NumR => Unit) =>
      var gc = 0
      val bound = a.length
      lazy val loop: NumR => Unit = FUN { (x : NumR) =>
        if (gc < bound) {gc += 1; RST(loop(b(gc-1)(x)))} else RST(k(x))
      }
      loop(init)
    }
/*
    @virtualize
    def LOOPL(init: NumR)(c: Rep[Int])(b: Rep[Int] => NumR => NumR @diff): NumR @diff = shift { k:(NumR => Unit) =>
      var gc = 0
      lazy val loop: NumR => Unit = FUN { (x: NumR) =>
        if (gc < c) { gc += 1; RST(b(gc-1)(loop(x))) } else RST(k(x))
      }                                   // type error, loop(x) is unit type
      loop(init)
    }

    @virtualize
    def LOOPL1(init: NumR)(c: Rep[Int])(b: Rep[Int] => NumR => NumR @diff): NumR @diff = shift { k: (NumR => Unit) =>
      var gc = 0
      lazy val loop: NumR => NumR @diff = FUNL { (x: NumR) => // How to design the FUNL, when the type is NumR => NumR @diff?
        if (gc < c) { gc += 1; b(gc-1)(loop(x)) } else x
      }
      RST(k(loop(init)))
    }
*/
/*
    @virtualize
    def LOOPL2(init: NumR)(c: Rep[Int])(b: Rep[Int] => NumR => NumR @diff): NumR @diff = shift { k: (NumR => Unit) =>
      var gc = 0
      var k_g = k
      lazy val loop: NumR => Unit = FUN { (x: NumR) =>
        if (gc < c) { gc += 1; k_g = { (x: NumR) => k_g(b(gc-1)(x)) }; loop(x) } else RST(k_g(x))
      }
      loop(init)
    }
*/
    def FUNL(f: ((NumR => Unit) => (NumR => Unit))): ((NumR => Unit) => (NumR => Unit)) = {
      /* we have to have the continuation to be dynamic here:
         meaning that we almost have to have Rep[NumR => Unit] type as the fun parameter
         but we do extra trick to equvilently transfrom between Rep[NumR => Unit] and Rep[Double => Double]
      */
      val f1 = fun { (t1: Rep[Double => Double]) =>
        val t2: (NumR => Unit) = { (x: NumR) => x.d += t1(x.x) }
        val t3: (NumR => Unit) = f(t2)
        fun {(x: Rep[Double]) => 
          val deltaVar = var_new(0.0)
          t3(new NumR(x, deltaVar))
          readVar(deltaVar)
        }
      };

      {k1: (NumR => Unit) => 
        {
          val k2: Rep[Double => Double] = fun { (x: Rep[Double]) =>
            val deltaVar = var_new(0.0)
            k1(new NumR(x, deltaVar))
            readVar(deltaVar)
          }
          val k3: Rep[Double => Double] = f1(k2)
          val k4: (NumR => Unit) = {(x: NumR) => x.d += k3(x.x)}
          k4
        } 
      }
    }
/*
    @virtualize
    def LOOPL3(init: NumR)(c: Rep[Int])(b: Rep[Int] => NumR => NumR): NumR @diff = shift { k: (NumR => Unit) =>
      var gc = 0
      lazy val loop: (NumR => Unit) => (NumR => Unit) = FUNL { (k: NumR => Unit) =>
        //if (gc < c) { gc += 1; loop ((x: NumR) => k(b(gc-1)(x))) } else { k }
        { z => if (gc < c) { gc += 1; loop ((x: NumR) => k(b(gc-1)(x)))(z) } else k(z) }
      }
      loop(k)(init)
    } 
*/

    @virtualize
    def LOOPL4(init: NumR)(c: Rep[Int])(b: Rep[Int] => NumR => NumR @diff): NumR @diff = shift { k: (NumR => Unit) =>
      var gc = 0
      lazy val loop: (NumR => Unit) => NumR => Unit = FUNL { (k: NumR => Unit) => (x: NumR) =>
        if (gc < c) { gc += 1; loop((x: NumR) => RST(k(b(gc-1)(x))))(x)  } else { RST(k(x)) } 
      }                                               // Problem! gc is a var, so it changes all the time
      loop(k)(init)                                   // so all recursive call will use the same value of gc in the last recursion
    }                                                 // Trying to multiply all elements in a list will endup multiplying the last element many times

    def FUNL1(f: (Rep[Int] => (NumR => Unit) => (NumR => Unit))): (Rep[Int] => (NumR => Unit) => (NumR => Unit)) = {      

      val f1 = fun { (yy: Rep[(Int, Double => Double)]) => // Problem! no support for tuple type code generation in LMS!
        val i: Rep[Int] = tuple2_get1(yy)
        val t1: Rep[Double => Double] = tuple2_get2(yy)
        //case (i: Rep[Int], t1: Rep[Double => Double]) =>
        val t2: (NumR => Unit) = { (x: NumR) => x.d += t1(x.x) }
        val t3: (NumR => Unit) = f(i)(t2)
        fun {(x: Rep[Double]) => 
          val deltaVar = var_new(0.0)
          t3(new NumR(x, deltaVar))
          readVar(deltaVar)
        }
      };

      {i: Rep[Int] => k1: (NumR => Unit) => 
        {
          val k2: Rep[Double => Double] = fun { (x: Rep[Double]) =>
            val deltaVar = var_new(0.0)
            k1(new NumR(x, deltaVar))
            readVar(deltaVar)
          }
          val k3: Rep[Double => Double] = f1((i, k2))
          val k4: (NumR => Unit) = {(x: NumR) => x.d += k3(x.x)}
          k4
        } 
      }
    }

    @virtualize
    def LOOPL5(init: NumR)(c: Rep[Int])(b: Rep[Int] => NumR => NumR @diff): NumR @diff = shift { k: (NumR => Unit) =>
      lazy val loop: (Rep[Int]) => (NumR => Unit) => NumR => Unit = FUNL1 { (gc: Rep[Int]) => (k: NumR => Unit) => (x: NumR) =>
        if (gc < 0) { loop(gc+1)((x: NumR) => RST(k(b(gc)(x))))(x)  } else { RST(k(x)) }
        //{ z => if (gc < c) { gc += 1; loop ((x: NumR) => k(b(gc-1)(x)))(z) } else k(z) }
      }
      loop(0)(k)(init)
    }    

/*
    @virtualize
    def LOOPL4(init: NumR)(c: Rep[Int])(b: Rep[Int] => NumR => NumR @diff): NumR @diff = shift { k: (NumR => Unit) =>
      var gc = 0
      lazy val loop: NumR => Unit = FUN { (x: NumR) =>
        if (gc < c) { gc += 1; RST(k(b(gc-1)(shift{kk: (NumR => Unit) => kk(x)}))) } else RST(k(x))
      }                                     
      loop(init)
    } 
*/

/*
    // How to support more entities in LOOP???
    import scala.collection.mutable.ArrayBuffer
    def FUNM(f: Array[NumR] => Unit): (Array[NumR] => Unit) = {
      val f1 = fun { (x:Rep[Array[Double]]) => 
        
        val length = 2
        val deltas = NewArray[Double](length)
        val nums = new NewArray[NumR](length)
        for (i <- (0 until length)) nums(i) = new NumR(x(i), deltas(i))
        
        //val nums = (x zip deltas) map (t => new NumR(t._1, t._2))
        // val nums = x map (t => new NumR(t, var_new(0.0)))
        f(nums)
        //nums map (t => readVar(t.d))
        deltas
        //deltas map (t => readVar(t))

        //val deltaVar = var_new(0.0)
        //f(new NumR(x, deltaVar))
        //readVar(deltaVar)
      };
      { (x:Array[NumR]) => {
        val in = x map (t => t.x)
        val out = f1(in)
        (x zip out) foreach (t => t._1.d += t._2)

        //x.d += f1(x.x)
        } 
      }
    }


    @virtualize
    def LOOPCCM(init: Array[NumR])(c: Rep[Int])(b: Rep[Int] => Array[NumR] => Array[NumR] @diff): Array[NumR] @diff = shift { k:(Array[NumR] => Unit) =>
      var gc = 0
      lazy val loop: Array[NumR] => Unit = FUNM { (x: Array[NumR]) =>
        if (gc < c) { gc += 1; RST(loop(b(gc-1)(x))) } else RST(k(x))
      }
      loop(init)
    }
*/

/*
    def FUN2(f: (NumR, NumR) => Unit): ((NumR, NumR) => Unit) = {
      val f1 = fun { (x:Rep[(Double, Double)]) => 
        val deltas = (var_new(0.0), var_new(0.0))
        f(new NumR(x._1, deltas._1), new NumR(x._2, deltas._2))
        (readVar(deltas._1), readVar(deltas._2))

        //val deltaVar = var_new(0.0)
        //f(new NumR(x, deltaVar))
        //readVar(deltaVar)
      };
      { (x: (NumR, NumR)) => {
        val in = (x._1.x, x._2.x)
        val out = f1(in)
        x._1.d += out._1
        x._2.d += out._2

        //x.d += f1(x.x)
        } 
      }
    }

    @virtualize
    def LOOPCC2(init: NumR, init1: NumR)(c: Rep[Int])(b: Rep[Int] => (NumR, NumR) => (NumR, NumR) @diff): (NumR, NumR) @diff = shift { k:((NumR, NumR) => Unit) =>
      var gc = 0
      lazy val loop: ((NumR, NumR)) => Unit = FUN2 { (x: (NumR, NumR)) =>
        if (gc < c) { gc += 1; RST(loop(b(gc-1)(x))) } else RST(k(x))
      }
      loop(init)
    }
*/

    def gradRV(f: NumRV => NumRV @diff)(x: Rep[Double]): Rep[Double] = {
      val x1 = new NumRV(x, 0.0)
      reset { f(x1).d = 1.0 }
      x1.d
    }
    def gradR(f: NumR => NumR @diff)(x: RDouble): Rep[Double] = {
      val x1 = new NumR(x, var_new(0.0))
      reset { var_assign(f(x1).d, 1.0); () }
      x1.d
    }

    def gradF(f: NumF => NumF)(x: RDouble) = {
      val x1 = new NumF(x, 1.0)
      f(x1).d
    }

    def gradFF(f: NumFF => NumFF)(x: RDouble) = {
      val x1 = new NumFF(x, new NumF(1.0, 0.0))
      f(x1).d.d
    }
  }

  def main(args: Array[String]): Unit = {

    val gr1 = new DslDriver[Double,Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        gradR(x => x + x*x*x)(x)
      }
    }

    val grv1 = new DslDriver[Double,Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        gradRV(x => x + x*x*x)(x)
      }
    }

    val gf1 = new DslDriver[Double,Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        gradF(x => x + x*x*x)(x)
      }
    }

    val gff1 = new DslDriver[Double,Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        gradFF(x => x + x*x*x)(x)
      }
    }
/*
    println("---- reverse mode with dynamic vars ---- \n")
    println(gr1.code)
    println("---- reverse mode with static vars ---- \n")
    println(grv1.code)
    println("---- forward mode ---- \n")
    println(gf1.code)
    println("---- forward mode second order ---- \n")
    println(gff1.code)

    // reverse mode
    for (x <- 0 until 10) {
      assert(gr1.eval(x) == 1 + 3*x*x)
    }

    // reverse mode (with static accumulation of reverse 
    // computation -- only for straightline code)
    for (x <- 0 until 10) {
      assert(grv1.eval(x) == 1 + 3*x*x)
    }

    // forward mode
    for (x <- 0 until 10) {
      assert(gf1.eval(x) == 1 + 3*x*x)
    }

    // forward of forward mode
    for (x <- 0 until 10) {
      assert(gff1.eval(x) == 6*x)
    }
*/
    // test 2 -- conditional
    val gr2 = new DslDriver[Double,Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        val minus_1 = (new NumR(-1.0,var_new(0.0)))
        gradR(x => IF (x.x > 0.0) { minus_1*x*x } { x*x })(x)
      }
    }
/*
    println("---- reverse mode with dynamic vars ---- \n")
    println(gr2.code)
    // NOTE: in the generated code, code motion has pushed
    // x3 = {x4: (Double) => ... } into both if branches.
    // (suboptimal in terms of code size)

    // reverse mode
    for (x <- -10 until 10) {
      assert(gr2.eval(x) == (if (x > 0.0) -2*x else 2*x))
    }
*/
    // test 3 -- loop using fold
    def fr(x: Double): Double = {
      // Divide by 2.0 until less than 1.0
      if (x > 1.0) fr(0.5 * x) else x 
    }
    // Hand-coded correct derivative
    def gfr(x: Double): Double = {
      if (x > 1.0) 0.5 * gfr(0.5 * x) else 1.0
    }


    val gr3 = new DslDriver[Double,Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        val half = (new NumR(0.5,var_new(0.0)))
        val res = gradR(x => LOOP(x)(x1 => x1.x > 1.0)(x1 => half * x1))(x)
        println(readVar(half.d))
        res
      }
    }
/*
    println("---- reverse mode with dynamic vars ---- \n")
    println(gr3.code)
    println(gr3.eval(10))


    // NOTE: in the generated code, code motion has pushed
    // x3 = {x4: (Double) => ... } into both if branches.
    // (suboptimal in terms of code size)

    // reverse mode
    for (x <- 0 until 10) {
      assert(gr3.eval(x) == gfr(x))
    }

    println("done")
*/    
    val gr4 = new DslDriver[Double, Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        val half = new NumR(0.5, var_new(0.0))
        val res = gradR(x => LOOPC(x)(3)(x1 =>{half * x1}))(x)
        println(readVar(half.d))
        res
      }
    }

    //println(gr4.code)
    //println(gr4.eval(10))

    val gr7 = new DslDriver[Double, Double] with DiffApi {

      def snippet(x: Rep[Double]): Rep[Double] = {
        val half = new NumR(0.5, var_new(0.0))
        val res = gradR(x => LOOPCC(x)(3)(i => x1 => {
          println(i)
          half * x1 }))(x)
        println(readVar(half.d))
        res
      }
    }

    //println(gr7.code)
    //println(gr7.eval(10))

    val gr8 = new DslDriver[Double, Double] with DiffApi {

      def snippet(x: Rep[Double]): Rep[Double] = {
        val array = NewArray[Double](3)
        for (i <- (0 until 3): Rep[Range]) array(i) = i + 2
        val model: NumR => NumR @diff = { (x: NumR) =>
          LOOPA(x)(array)(i => x1 => {
            val t = new NumR(array(i), var_new(0.0))
            t * x1
          })
        }
        val res = gradR(model)(x)
        res
      }
    }

    //println(gr8.code)
    //println(gr8.eval(10))

    val gr9 = new DslDriver[Double, Double] with DiffApi {

      def snippet(x: Rep[Double]): Rep[Double] = {
        // preprocess data (wrap Array as RepArray)
        val arr = scala.Array(1.5,3.0,4.0)
        val arra = staticData(arr)
        
        val model: NumR => NumR @diff = { (x: NumR) =>
          LOOPA(x)(arra)(i => x1 => {
            val t = new NumR(arra(i), var_new(0.0))
            t * x1
            })
        }
        val res = gradR(model)(x)
        res
      }
    }

    //println(gr9.code)
    //println(gr9.eval(10))

    val gr10 = new DslDriver[Double, Double] with DiffApi {

      def snippet(x: Rep[Double]): Rep[Double] = {
        // preprocess data
        val arr = scala.Array(1.5, 2.0, 3.0)
        val arra = staticData(arr)
        val length = arr.length

        // maybe we can use loopcc, just use arra by closure
        val model: NumR => NumR @diff = { (x: NumR) =>
          LOOPCC(x)(length)(i => x1 => {
            val t = new NumR(arra(i), var_new(0.0))
            t * x1
            })
        }
        val res = gradR(model)(x)
        res
        /*
          Note: It is interesting to note that the recursive function body can make use of closured array data
                but recursive guard (the if condition of LOOP) cannot, because the def of LOOP is before the presence of data
                So the length has to be passed in as a parameter explicitly
          Is there a better way to do it??
        */
      }
    }

    //println(gr10.code)
    //println(gr10.eval(10))

    val gr11 = new DslDriver[Double, Double] with DiffApi {

      @virtualize
      def snippet(x: Rep[Double]): Rep[Double] = {
        // represent list as array
        val arr = scala.Array(4.0, 3.0, 2.5, 2.0)
        val arra = staticData(arr)
        val length = arr.length

        // create a model that recursively use the data in arr (originated from list)
        def model: NumR => NumR @diff = { (x: NumR) =>
          LOOPL5(x)(arra.length)(i => x1 => {
            val t = new NumR(arra(i), var_new(0.0))
            t * x1
            })
        }
        val res = gradR(model)(x)
        res
      }
    }
    
    import java.io.PrintWriter;
    import java.io.File;    
    println(gr11.code)
    val p = new PrintWriter(new File("gr11.scala"))
    p.println(gr11.code)
    p.flush()
    println(gr11.eval(2))

  }
}
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

    // test 2 -- conditional
    val gr2 = new DslDriver[Double,Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        val minus_1 = (new NumR(-1.0,var_new(0.0)))
        gradR(x => IF (x.x > 0.0) { minus_1*x*x } { x*x })(x)
      }
    }

    println("---- reverse mode with dynamic vars ---- \n")
    println(gr2.code)
    // NOTE: in the generated code, code motion has pushed
    // x3 = {x4: (Double) => ... } into both if branches.
    // (suboptimal in terms of code size)

    // reverse mode
    for (x <- -10 until 10) {
      assert(gr2.eval(x) == (if (x > 0.0) -2*x else 2*x))
    }

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
        gradR(x => LOOP(x)(x1 => x1.x > 1.0)(x1 => half * x1))(x)
      }
    }

    println("---- reverse mode with dynamic vars ---- \n")
    println(gr3.code)
    // NOTE: in the generated code, code motion has pushed
    // x3 = {x4: (Double) => ... } into both if branches.
    // (suboptimal in terms of code size)

    // reverse mode
    for (x <- 0 until 10) {
      assert(gr3.eval(x) == gfr(x))
    }

    println("done")
  }
}
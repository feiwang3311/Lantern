/*
package lantern

import lms.core.stub._
import lms.core.virtualize
import lms.macros.SourceContext
import org.scalatest.FunSuite

import scala.util.continuations._
import scala.util.continuations


object Hessian_MuWang_1 {

  def main(args: Array[String]): Unit = {

    import scala.util.continuations._
    type diff = cps[Unit]

    import scala.collection.mutable.{Map, HashMap, Set, HashSet}

    // use external Map data structure to track hessian matrix
    var hessian: Map[(Num, Num), Double] = new HashMap[(Num, Num), Double]()
    var liveSet: Set[Num] = new HashSet[Num]()

    def getHessian(a: Num, b: Num): Double = {
      if (liveSet.contains(a) && liveSet.contains(b)) {
        hessian.get((a, b)) match {
          case Some(d) => d
          case None => hessian.get((b, a)) match {
            case Some(d) => d
            case None => 0.0
          }
        }
      } else 0.0
    }

    def incrementHessian(a: Num, b: Num, d: Double) = {
      hessian.get((a, b)) match {
        case Some(d0) => hessian += (((a, b), d0 + d))
        case None     => hessian.get((b, a)) match {
          case Some(d0) => hessian += (((b, a), d0 + d))
          case None     => hessian += (((b, a), d))
        }
      }
    }

    var debug = false
    def dprintln(s: String) = if (debug) println(s)

    class Num(val x: Double, var d: Double) {

      def +(that: Num): Num @diff = shift { (k: Num => Unit) =>
        val y = new Num(x + that.x, 0.0); k(y)
        this.d += y.d; that.d += y.d // update adjoints
        liveSet.add(this); liveSet.add(that) // update liveSet step 1
        // update hessian
        for (v <- liveSet if (v != y)) {
          val t = getHessian(v, y)
          if (t != 0.0) {
            incrementHessian(v, this, (if (v == this) 2 else 1) * t)
            incrementHessian(v, that, (if (v == that) 2 else 1) * t)
          }
        }
        val t = getHessian(y, y)
        if (t != 0.0) incrementHessian(this, that, t)
        liveSet.remove(y) // update liveSet step 2
      }

      def *(that: Num): Num @diff = shift { (k: Num => Unit) =>
        val y = new Num(x * that.x, 0.0); k(y)
        this.d += that.x * y.d; that.d += this.x * y.d // update adjoints
        liveSet.add(this); liveSet.add(that) // update liveSet step 1
        // update hessian
        for (v <- liveSet if (v != y)) {
          val t = getHessian(v, y)
          if (t != 0.0) {
            incrementHessian(v, this, (if (v == this) 2 else 1) * that.x * t)
            incrementHessian(v, that, (if (v == that) 2 else 1) * this.x * t)
          }
        }
        val t = getHessian(y, y)
        if (t != 0.0) incrementHessian(this, that, 2 * this.x * that.x * t)
        if (y.d != 0) incrementHessian(this, that, y.d * (if (this == that) 2 else 1))
        liveSet.remove(y); // update liveSet step 2
      }
    }

    def setFinalClosure(t : Num) = {
      val result = t.x
      // println(s"result of function is $result")     // print result of the function
      t.d = 1.0                                     // set gradient
      liveSet = Set(t)                              // set liveSet with t
      hessian = new HashMap[(Num, Num), Double]()   // set hessian as empty
    }

    implicit def toNum(x: Double): Num = new Num(x, 0.0)

    def hessian_1(f: Num => Num @diff)(x: Double) = {
      val x1 = new Num(x, 0.0)
      reset {
        setFinalClosure(f(x1))
      }
      val grad = x1.d
      // println(s"result of gradient is $grad")
      val hess = getHessian(x1, x1)
      // println(s"result of hessian is $hess")
      (grad, hess)
    }

    /* tests for hessian_1 */
    println("Step 1")
    for (x <- -10 until 10) {
      assert(hessian_1(x => x + x*x*x)(x) == (1 + 3*x*x, 6*x))
    }

    for (x <- -10 until 10) {
      assert(hessian_1(x => 3.0 * x * x + x * x * x)(x) == (6 * x + 3 * x * x, 6 + 6 * x))
    }

    for (x <- -10 until 10) {
      assert(hessian_1(x => (x + x) * x)(x) == (4 * x, 4))
    }

    for (x <- -10 until 10) {
      assert(hessian_1(x => (x + x) * x * x + (2 * x + 2 * x) * x)(x) == (6 * x * x + 8 * x, 12 * x + 8))
    }

    def hessian_2(f: (Num, Num) => Num @diff)(x1: Double, x2: Double) = {
      val x11 = toNum(x1)
      val x22 = toNum(x2)
      reset {
        setFinalClosure(f(x11, x22))
      }
      val grads = (x11.d, x22.d)
      // println(s"result of gradient is $grads")
      val hess = ((getHessian(x11, x11), getHessian(x11, x22)), (getHessian(x22, x11), getHessian(x22, x22)))
      // println(s"result of hessian is $hess")
      (grads, hess)
    }

    /* tests for hessian_2 */
    println("Step 2")
    for (x <- -10 until 10) {
      for (y <- -10 until 10) {
        assert (hessian_2((x1,x2) => x1*x1 + 2*x1*x2 + x2*x2)(x,y) == ((2*(x+y),2*(x+y)), ((2,2),(2,2))))
      }
    }

    for (x <- -10 until 10) {
      for (y <- -10 until 10) {
        assert (hessian_2((x1,x2) => 3*x1*x1*(x1+x1) + 2*x1*x2*(x1+x2) + x2*x2*(x2+x2))(x,y) ==
          ((18*x*x+4*x*y+2*y*y, 2*x*x+4*x*y+6*y*y),  // gradients
           ((36*x+4*y, 4*x+4*y), (4*x+4*y, 4*x+12*y)) // hessians
          ))
      }
    }

    println("done")
  }
}

object Hessian_MuWang_LMS {

  trait DiffApi extends Dsl {

    type diff = cps[Unit]

    type RDouble = Rep[Double]

    import scala.collection.mutable.{Map, HashMap, Set, HashSet}

    // use external Map data structure to track hessian matrix
    var hessian: Map[(Num, Num), RDouble] = new HashMap[(Num, Num), RDouble]()
    var liveSet: Set[Num] = new HashSet[Num]()

    def getHessian(a: Num, b: Num): RDouble = {
      if (liveSet.contains(a) && liveSet.contains(b)) {
        hessian.get((a, b)) match {
          case Some(d) => d
          case None => hessian.get((b, a)) match {
            case Some(d) => d
            case None => 0.0
          }
        }
      } else 0.0
    }

    def incrementHessian(a: Num, b: Num, d: RDouble) = {
      hessian.get((a, b)) match {
        case Some(d0) => hessian += (((a, b), d0 + d))
        case None     => hessian.get((b, a)) match {
          case Some(d0) => hessian += (((b, a), d0 + d))
          case None     => hessian += (((b, a), d))
        }
      }
    }

    class Num(val x: RDouble, var d: RDouble) {

      def +(that: Num): Num @diff = shift { (k: Num => Unit) =>
        val y = new Num(x + that.x, 0.0); k(y)
        this.d += y.d; that.d += y.d // update adjoints
        liveSet.add(this); liveSet.add(that) // update liveSet step 1
        // update hessian
        for (v <- liveSet if (v != y)) {
          val t = getHessian(v, y)
          if (t != 0.0) {
            incrementHessian(v, this, (if (v == this) 2 else 1) * t)
            incrementHessian(v, that, (if (v == that) 2 else 1) * t)
          }
        }
        val t = getHessian(y, y)
        if (t != 0.0) incrementHessian(this, that, t)
        liveSet.remove(y) // update liveSet step 2
      }

      def *(that: Num): Num @diff = shift { (k: Num => Unit) =>
        val y = new Num(x * that.x, 0.0); k(y)
        this.d += that.x * y.d; that.d += this.x * y.d // update adjoints
        liveSet.add(this); liveSet.add(that) // update liveSet step 1
        // update hessian
        for (v <- liveSet if (v != y)) {
          val t = getHessian(v, y)
          if (t != 0.0) {
            incrementHessian(v, this, (if (v == this) 2 else 1) * that.x * t)
            incrementHessian(v, that, (if (v == that) 2 else 1) * this.x * t)
          }
        }
        val t = getHessian(y, y)
        if (t != 0.0) incrementHessian(this, that, 2 * this.x * that.x * t)
        if (y.d != 0) incrementHessian(this, that, y.d * (if (this == that) 2 else 1))
        liveSet.remove(y); // update liveSet step 2
      }
    }

    def setFinalClosure(t : Num) = {
      val result = t.x
      // println(s"result of function is $result")     // print result of the function
      t.d = 1.0                                     // set gradient
      liveSet = Set(t)                              // set liveSet with t
      hessian = new HashMap[(Num, Num), RDouble]()   // set hessian as empty
    }

    implicit def toNum(x: Double): Num = new Num(x, 0.0)

    def hessian_1(f: Num => Num @diff)(x: RDouble): Rep[Unit] = {
      val x1 = new Num(x, 0.0)
      reset {
        setFinalClosure(f(x1))
      }
      val grad = x1.d
      printf("the result of gradient is %f\n", grad)
      val hess = getHessian(x1, x1)
      printf("the result of hessian is %f\n", hess)
      // println(s"result of hessian is $hess")
      // (grad, hess)
    }

    def hessian_2(f: (Num, Num) => Num @diff)(x1: RDouble, x2: RDouble) = {
      val x11 = new Num(x1, 0.0)
      val x22 = new Num(x2, 0.0)
      reset {
        setFinalClosure(f(x11, x22))
      }
      val grads = (x11.d, x22.d)
      printf("the result of gradient is (%f, %f)\n", grads._1, grads._2)
      // println(s"result of gradient is $grads")
      val hess = ((getHessian(x11, x11), getHessian(x11, x22)), (getHessian(x22, x11), getHessian(x22, x22)))
      printf("the result of hessian is ((%f, %f),(%f, %f))\n", hess._1._1, hess._1._2, hess._2._1, hess._2._2)
      // println(s"result of hessian is $hess")
      // (grads, hess)
    }
  }

  def main(args: Array[String]): Unit = {

    val gr1 = new DslDriver[Double, Unit] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Unit] = {
        hessian_1(x => x + x*x*x)(x)
      }
    }

    val gr2 = new DslDriver[Double, Unit] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Unit] = {
        hessian_1(x => toNum(3)*x*x + x*x*x)(x)
      }
    }

    val gr3 = new DslDriver[Double, Unit] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Unit] = {
        hessian_1(x => (x+x)*x)(x)
      }
    }

    val gr4 = new DslDriver[Double, Unit] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Unit] = {
        hessian_1(x => (x+x)*x*x + (toNum(2)*x+toNum(2)*x)*x)(x)
      }
    }

    println("---- test hessian_1 reverse mode with dynamic vars ---- \n")
    println(gr1.code)
    println(gr2.code)
    println(gr3.code)
    println(gr4.code)

    for (x <- -5 until 5) {
      gr1.eval(x)
      gr2.eval(x)
      gr3.eval(x)
      gr4.eval(x)
    }

    val gr5 = new DslDriver[(Double, Double), Unit] with DiffApi { q =>
      override val codegen = new DslGen {
        val IR: q.type = q
      }

      def snippet(x: Rep[(Double, Double)]): Rep[Unit] = {
        hessian_2((x1,x2) => x1*x1 + toNum(2)*x1*x2 + x2*x2)(x._1, x._2)
      }
    }

    val gr6 = new DslDriver[(Double, Double), Unit] with DiffApi { q =>
      override val codegen = new DslGen {
        val IR: q.type = q
      }

      def snippet(x: Rep[(Double, Double)]): Rep[Unit] = {
        hessian_2((x1,x2) => toNum(3)*x1*x1*(x1+x1) + toNum(2)*x1*x2*(x1+x2) + x2*x2*(x2+x2))(x._1, x._2)
      }
    }

    println("---- test hessian_2 reverse mode with dynamic vars ---- \n")
    println(gr5.code)
    println(gr6.code)

    for (x <- -3 until 3) {
      for (y <- -3 until 3) {
        // gr5.eval((x, y)); gr6.eval((x, y))
        // Question: cannot run the code:
        // error: not found: type Tuple2DoubleDouble
      }
    }

  }
}



object Hessian_MuWang_LMS_IFLOOP {

  trait DiffApi extends Dsl {

    type diff = cps[Unit]

    type RDouble = Rep[Double]

    import scala.collection.mutable.{Map, HashMap, Set, HashSet}

    // use external Map data structure to track hessian matrix
    var hessian: Map[(Num, Num), Var[Double]] = new HashMap[(Num, Num), Var[Double]]()
    var liveSet: Set[Num] = new HashSet[Num]()

    def getHessian(a: Num, b: Num): RDouble = {
      if (liveSet.contains(a) && liveSet.contains(b)) {
        hessian.get((a, b)) match {
          case Some(d) => readVar(d)
          case None => hessian.get((b, a)) match {
            case Some(d) => readVar(d)
            case None => 0.0
          }
        }
      } else 0.0
    }

    def incrementHessian(a: Num, b: Num, d: RDouble) = {
      hessian.get((a, b)) match {
        case Some(d0) => d0 += d
        case None     => hessian.get((b, a)) match {
          case Some(d0) => d0 += d
          case None     => hessian += (((b, a), var_new(d)))
          /*
          Note: the last case in incrementHessian is the tricky part of the design
          Reason: the last case create a new var, which translates to "var x?? : Double = 0.0" in generated code
                  there is no guarentee that this var will not be used in a higher scope, thus giving "effect in the wrong order error"
                  This error showed up when I added IF function, and can be resolved by making sure that all hessian between parameters are
                      pre-initialized before running the hessian function
                      (so that the returned hessian pair is not created in an inner scope such as if branch)
          Qeustion: Why LMS cannot automatically lift the var to the outer scope if necessary?
                    All Syms are of different name, so simply lifting the var creation out should fix any potential
                    "effect in the wrong order" error
          */
        }
      }
    }

    // can be optimized to remove order of pairs TODO
    def initHessian(a: Num*) = {
      val b = a.toSet
      for (v1 <- b; v2 <- b) {
          initIfNotInHessian(v1, v2)
      }
    }
    def initIfNotInHessian(a: Num, b: Num) = {
      hessian.get((a, b)) match {
        case Some(_) => ()
        case None => hessian.get((b, a)) match {
          case Some(_) => ()
          case None => hessian.update((a, b), var_new(0.0))
        }
      }
    }

    class Num(val x: RDouble, val d: Var[Double]) extends Serializable {

      def +(that: Num): Num @diff = shift { (k: Num => Unit) =>
        val y = new Num(x + that.x, var_new(0.0)); k(y)
        this.d += y.d; that.d += y.d          // update adjoints
        liveSet.add(this); liveSet.add(that)  // update liveSet step 1
        // update hessian
        for (v <- liveSet if (v != y)) {
          val t = getHessian(v, y)
          if (t != 0.0) {
            incrementHessian(v, this, (if (v == this) 2 else 1) * t)
            incrementHessian(v, that, (if (v == that) 2 else 1) * t)
          }
        }
        val t = getHessian(y, y)
        if (t != 0.0) incrementHessian(this, that, t)
        liveSet.remove(y)                      // update liveSet step 2
      }

      def *(that: Num): Num @diff = shift { (k: Num => Unit) =>
        val y = new Num(x * that.x, var_new(0.0)); k(y)
        this.d += that.x * y.d; that.d += this.x * y.d // update adjoints
        liveSet.add(this); liveSet.add(that)           // update liveSet step 1
        // update hessian
        for (v <- liveSet if (v != y)) {
          val t = getHessian(v, y)
          if (t != 0.0) {
            incrementHessian(v, this, (if (v == this) 2 else 1) * that.x * t)
            incrementHessian(v, that, (if (v == that) 2 else 1) * this.x * t)
          }
        }
        val t = getHessian(y, y)
        if (t != 0.0) incrementHessian(this, that, 2 * this.x * that.x * t)
        if (readVar(y.d) != 0.0) incrementHessian(this, that, readVar(y.d) * (if (this == that) 2 else 1))
        liveSet.remove(y);                              // update liveSet step 2
      }
    }

    def RST(a: =>Unit @diff) = continuations.reset { a; () }

    def FUN(f: Num => Unit): (Num => Unit) = {
      val f1 = fun { (x:Rep[Double]) =>
        val deltaVar = var_new(0.0)
        f(new Num(x, deltaVar))
        readVar(deltaVar)
      };
      { (x:Num) => x.d += f1(x.x) }
    }

    @virtualize
    def IF(c: Rep[Boolean])(a: =>Num @diff)(b: =>Num @diff): Num @diff = shift { k:(Num => Unit) =>
      val k1 = FUN(k)

      if (c) RST(k(a)) else RST(k(b)) // need to use k !
    }

    def setFinalClosure(t : Num) = {
      val result = t.x
      // println(s"result of function is $result")            // print result of the function
      __assign(t.d, 1.0)                                    // set gradient
      liveSet = Set(t)                                        // reset liveSet with t only
    }

    implicit def toNum(x: Double): Num = new Num(x, var_new(0.0))

    def hessian_1(f: Num => Num @diff)(x: RDouble): Rep[Unit] = {
      val x1 = new Num(x, var_new(0.0))
      hessian.clear(); initHessian(x1)                        // pre-initialize hessian with all parameter pairs
      reset {
        setFinalClosure(f(x1))
      }
      val grad = x1.d
      printf("the result of gradient is %f\n", grad)
      val hess = getHessian(x1, x1)
      // hess
      printf("the result of hessian is %f\n", hess)
      // println(s"result of hessian is $hess")
      // (grad, hess)
    }

    def hessian_2(f: (Num, Num) => Num @diff)(x1: RDouble, x2: RDouble) = {
      val x11 = new Num(x1, var_new(0.0))
      val x22 = new Num(x2, var_new(0.0))
      hessian.clear(); initHessian(x11, x22)                   // pre-initialize hessian with all parameter pairs
      reset {
        setFinalClosure(f(x11, x22))
      }
      val grads = (x11.d, x22.d)
      printf("the result of gradient is (%f, %f)\n", grads._1, grads._2)
      // println(s"result of gradient is $grads")
      val hess = ((getHessian(x11, x11), getHessian(x11, x22)), (getHessian(x22, x11), getHessian(x22, x22)))
      printf("the result of hessian is ((%f, %f),(%f, %f))\n", hess._1._1, hess._1._2, hess._2._1, hess._2._2)
      // println(s"result of hessian is $hess")
      // (grads, hess)
    }
  }

  def main(args: Array[String]): Unit = {

    val gr1 = new DslDriver[Double, Unit] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Unit] = {
        hessian_1(x => x*x*x)(x)
      }
    }

    val gr2 = new DslDriver[Double, Unit] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Unit] = {
        hessian_1(x => toNum(3)*x*x + x*x*x)(x)
      }
    }

    val gr3 = new DslDriver[Double, Unit] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Unit] = {
        hessian_1(x => (x+x)*x)(x)
      }
    }

    val gr4 = new DslDriver[Double, Unit] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Unit] = {
        hessian_1(x => (x+x)*x*x + (toNum(2)*x+toNum(2)*x)*x)(x)
      }
    }

    val grif1 = new DslDriver[Double, Unit] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Unit] = {
        hessian_1(x => IF (x.x > 0.0) {x*x*x} {toNum(-1)*x*x})(x)
      }
    }

    val grif2 = new DslDriver[Double, Unit] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Unit] = {
        hessian_1( x => {
            val y = IF(x.x > 0.0){x*x}{toNum(-1)*x}
            y + x*x
            } )(x)
      }
    }

    println("---- test hessian_1 reverse mode with dynamic vars ---- \n")
    println(gr1.code)
    println(gr2.code)
    println(gr3.code)
    println(gr4.code)
    println(grif1.code)
    println(grif2.code)

    for (x <- -3 until 3) {
      gr1.eval(x)
      gr2.eval(x)
      gr3.eval(x)
      gr4.eval(x)
      grif1.eval(x)
      grif2.eval(x)
    }

    val gr5 = new DslDriver[(Double, Double), Unit] with DiffApi with TupleOpsExp { q =>
      override val codegen = new DslGen with ScalaGenTupleOps {
        val IR: q.type = q
      }

      def snippet(x: Rep[(Double, Double)]): Rep[Unit] = {
        hessian_2((x1,x2) => x1*x1 + toNum(2)*x1*x2 + x2*x2)(x._1, x._2)
      }
    }

    val gr6 = new DslDriver[(Double, Double), Unit] with DiffApi with TupleOpsExp { q =>
      override val codegen = new DslGen with ScalaGenTupleOps {
        val IR: q.type = q
      }

      def snippet(x: Rep[(Double, Double)]): Rep[Unit] = {
        hessian_2((x1,x2) => toNum(3)*x1*x1*(x1+x1) + toNum(2)*x1*x2*(x1+x2) + x2*x2*(x2+x2))(x._1, x._2)
      }
    }

    val gr7 = new DslDriver[(Double, Double), Unit] with DiffApi with TupleOpsExp { q =>
      override val codegen = new DslGenScala with ScalaGenTupleOps {
        val IR: q.type = q
      }

      def snippet(x: Rep[(Double, Double)]): Rep[Unit] = {
        hessian_2((x1, x2) => IF(x1.x > 1.0){x1*x1}{x1*x2} + x2*x2)(x._1, x._2)
      }
    }

    println("---- test hessian_2 reverse mode with dynamic vars ---- \n")
    println(gr5.code)
    println(gr6.code)
    println(gr7.code)

    for (x <- -3 until 3) {
      for (y <- -3 until 3) {
        // gr5.eval((x, y)); gr6.eval((x, y))
        // Question: cannot run the code:
        // error: not found: type Tuple2DoubleDouble
      }
    }

  }
}
*/

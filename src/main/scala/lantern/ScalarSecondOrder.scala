package lantern

import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

trait SecOrderApi extends DslOps with Diff {

  import scala.collection.mutable.ArrayBuffer
  import scala.util.continuations._

  /* because we are calculating Hessian_vector product, the tangent is not a Map, but a number
     By the paper "Fast Exact Multiplication by the Hessian", Hv = diff_{r=0} {G(w + rv)} = J(G(w))*v
     So if we are looking at a function R^n -> R,
     basically we are doing gradient of G(w) at position w, but in regard to a fix direction v, and only one variable r
     In terms of implementation, we need to change the tangent in the forward pass (from a Map to just a number),
     this change should reflect the norm and direction of v
   */

  object NumF {
    var counter = 0
    // def apply(tag: Int, x: Double, d: Double) = new NumF(tag, x, d)
    // def apply(x: Double, d: Double = 0.0) = {
    //   val temp = new NumF(counter, x, d)
    //   printf(s"$temp\n")
    //   counter += 1
    //   temp
    // }
    def apply(x: Double = 0.0, d: Double = 0.0, negtag: Boolean = false) = 
      if (negtag) new NumF(-1, x, d)
      else {
        val temp = new NumF(counter, x, d)
        // printf(s"$temp\n")
        counter += 1
        temp
      }
  }

  class NumF(tag: Int, var x: Double, var d: Double) {
    def +(that: NumF, negtag: Boolean = false) = NumF(x + that.x, this.d + that.d, negtag)
    def *(that: NumF, negtag: Boolean = false) = NumF(x * that.x, this.d * that.x + that.d * this.x, negtag)
    def sin(negtag: Boolean = false) = NumF(scala.math.sin(x), d * scala.math.cos(x), negtag)
    def cos(negtag: Boolean = false) = NumF(scala.math.cos(x), - d * scala.math.sin(x), negtag)
    override def toString() = s"tag: $tag, val: $x, grad: $d"
    
    def +=(that: NumF) = {x += that.x; d += that.d}
    def +=(that: NumF, scale: Double) = {x += that.x * scale; d += that.d * scale}
    def update(that0: NumF) = this += that0
    def update(that0: NumF, that1: NumF, f: (NumF, NumF) => NumF) = this += f(that0, that1)
  }

  implicit def toNumF(x: Double) = NumF(x)
  implicit def toNumR(x: Double) = new NumR(NumF(x), NumF())

  class NumR(val x: NumF, val d: NumF) {
    def +(that: NumR) = shift { (k: NumR => Unit) => 
      val y = new NumR(x + that.x, NumF()); k(y)
      this.d update y.d; that.d update y.d
    }
    def *(that: NumR) = shift { (k: NumR => Unit) => 
      val y = new NumR(x * that.x, NumF()); k(y)
      this.d update (that.x, y.d, (a, b) => a.*(b, true))
      that.d update (this.x, y.d, (a, b) => a.*(b, true))
    }
    def sin() = shift { (k: NumR => Unit) =>
      val y = new NumR(x.sin(), NumF()); k(y)
      this.d update (y.d, x, (a, b) => a.*(x.cos(true), true)) 
    }
  }

  def finalClosure(t: NumR) = {
    val result = t.x.x // this is the value of the function
    println(s"value of the function is $result")
    t.d.x = 1.0      // set the gradient value to be 1.0, the gradient tangent remains empty
  }

  /* tests: R^2 -> R */
  // println("test for R^2 -> R")
  def grad_two_inputs(f: (NumR, NumR) => NumR @diff)(v0: Double, v1: Double)(v: (Double, Double)) = {
    val x1 = new NumR(NumF(v0, v._1), 0.0)
    val x2 = new NumR(NumF(v1, v._2), 0.0)
    reset {
      finalClosure(f(x1, x2))
    }
    val gradient = (x1.d.x, x2.d.x)
    val hessian_vector = (x1.d.d, x2.d.d)
    (gradient, hessian_vector)
  }

  object NumFS {
    // def apply(x: Rep[Array[Double]], d: Rep[Array[Double]]) = new NumFS(x, d)
    def apply(x: Rep[Array[Double]]) = {
      val d = NewArray[Double](1); d(0) = 0.0
      new NumFS(x, d)
    }
    def apply() = {
      val x = NewArray[Double](1); x(0) = 0.0
      val d = NewArray[Double](1); d(0) = 0.0
      new NumFS(x, d)
    }
    def apply(x: Rep[Double], d: Rep[Double] = 0.0) = {
      val xx = NewArray[Double](1); xx(0) = x
      val dd = NewArray[Double](1); dd(0) = d
      new NumFS(xx, dd)
    }
  }

  // Array here is always size 1
  class NumFS(val x: Rep[Array[Double]], val d: Rep[Array[Double]]) extends Serializable {
    def + (that: NumFS) = NumFS(x(0) + that.x(0), d(0) + that.d(0))
    def * (that: NumFS) = NumFS(x(0) * that.x(0), d(0) * that.x(0) + that.d(0) * x(0))
    def sin() = NumFS(Math.sin(x(0)), d(0) * Math.cos(x(0)))
    def cos() = NumFS(Math.cos(x(0)), 0 - d(0) * Math.sin(x(0)))

    def += (that: NumFS) = {
      x(0) = x(0) + that.x(0)
      d(0) = d(0) + that.d(0)
    }
    def update(that: NumFS): Unit = this += that
    def update(that0: NumFS, that1: NumFS, f: (NumFS, NumFS) => NumFS): Unit = this += f(that0, that1)

    def < (that: NumFS): Rep[Boolean] = x(0) < that.x(0)
    def > (that: NumFS): Rep[Boolean] = x(0) > that.x(0)
  }

  class NumRS(val x: NumFS, val d: NumFS) extends Serializable {
    def + (that: NumRS) = shift { (k: NumRS => Unit) =>
      val y = new NumRS(x + that.x, NumFS()); k(y)
      this.d update y.d; that.d update y.d
    }
    def * (that: NumRS) = shift { (k: NumRS => Unit) =>
      val y = new NumRS(x * that.x, NumFS()); k(y)
      this.d update (that.x, y.d, (a, b) => a * b)
      that.d update (this.x, y.d, (a, b) => a * b)
    }
    def sin() = shift { (k: NumRS => Unit) =>
      val y = new NumRS(x.sin(), NumFS()); k(y)
      this.d update (y.d, x, (a, b) => a * x.cos()) 
    }

    def < (that: NumRS): Rep[Boolean] = x < that.x
    def > (that: NumRS): Rep[Boolean] = x > that.x
  }
  
  def toNumRS(x: Rep[Double]) = new NumRS(NumFS(x), NumFS())

  /* tests: R^2 -> R */
  // println("test for R^2 -> R")
  def grad_two_inputsS(f: (NumRS, NumRS) => NumRS @diff)(v0: Double, v1: Double)(v: (Double, Double)) = {
    val x1 = new NumRS(NumFS(v0, v._1), NumFS())
    val x2 = new NumRS(NumFS(v1, v._2), NumFS())
    reset {
      val temp = f(x1, x2)
      temp.d.x(0) = 1.0
      ()
    }
    val gradient = (x1.d.x(0), x2.d.x(0))
    val hessian_vector = (x1.d.d(0), x2.d.d(0))
    (gradient, hessian_vector)
  }

  @virtualize
  def assertVectorEqual(result: (Rep[Double], Rep[Double]), expected: (Rep[Double], Rep[Double]), eps: Double = 0.00001) =
    if (result._1 < expected._1 - eps || result._1 > expected._1 + eps ||
        result._2 < expected._2 - eps || result._2 > expected._2 + eps) {
      printf("(%f, %f) is not as expected (%f, %f)", result._1, result._2, expected._1, expected._2)
      error("")
    }

  def helperArray(x: NumRS): Rep[Array[Array[Double]]] = {
    val temp = NewArray[Array[Double]](4)
    temp(0) = x.x.x; temp(1) = x.x.d; temp(2) = x.d.x; temp(3) = x.d.d
    temp
  }

  def FUN(f: NumRS => Unit): (NumRS => Unit) = { (x: NumRS) =>
    val f1 = fun { (in: Rep[Array[Array[Double]]]) =>
      f(new NumRS(new NumFS(in(0), in(1)), new NumFS(in(2), in(3))))
    }
    f1(helperArray(x))
  }

  @virtualize
  def IF(c: Rep[Boolean])(a: =>NumRS @diff)(b: =>NumRS @diff): NumRS @diff = shift { k:(NumRS => Unit) =>
    val k1 = FUN(k)
    if (c) RST(k1(a)) else RST(k1(b))
  }

  @virtualize
  def WHILE(init: NumRS)(c: NumRS => Rep[Boolean])(b: NumRS => NumRS @diff): NumRS @diff = shift { k:(NumRS => Unit) =>
    lazy val loop: NumRS => Unit = FUN { (x: NumRS) =>
      if (c(x)) RST(loop(b(x))) else RST(k(x))
    }
    loop(init)
  }

  def FUN(f: (Rep[Int], NumRS) => Unit): (Rep[Int], NumRS) => Unit = { (i: Rep[Int], x: NumRS) =>
    val f1 = fun { (i: Rep[Int], in: Rep[Array[Array[Double]]]) =>
      f(i, new NumRS(new NumFS(in(0), in(1)), new NumFS(in(2), in(3))))
    }
    f1(i, helperArray(x))
  }

  @virtualize
  def FOR(init: NumRS)(c: Rep[Int])(b: (Rep[Int], NumRS) => NumRS @diff): NumRS @diff = shift { k:(NumRS => Unit) =>
    lazy val loop: (Rep[Int], NumRS) => Unit = FUN { (i: Rep[Int], x: NumRS) =>
      if (i < c) { RST(loop(i+1, b(i, x))) } else RST(k(x))
    }
    loop(0, init)
  }

  // stack continuations for recursive models
  def FUN(f: (Rep[Int], NumRS => Unit, NumRS) => Unit): (Rep[Int], NumRS => Unit, NumRS) => Unit = { (i: Rep[Int], k: NumRS => Unit, x: NumRS) =>
    val ks: Rep[Array[Array[Double]] => Unit] = fun { (in: Rep[Array[Array[Double]]]) =>
      k(new NumRS(new NumFS(in(0), in(1)), new NumFS(in(2), in(3))))
    }
    val f1 = fun { (i: Rep[Int], ks: Rep[Array[Array[Double]] => Unit], in: Rep[Array[Array[Double]]]) =>
      val k: (NumRS => Unit) = (x: NumRS) => ks(helperArray(x))
      f(i, k, new NumRS(new NumFS(in(0), in(1)), new NumFS(in(2), in(3))))
    }
    f1(i, ks, helperArray(x))
  }

  @virtualize // NOTE: this version cannot handle empty trees // assume that children array use -1 for leaf nodes
  def TREE(init: NumRS)(lch: Rep[Array[Int]], rch: Rep[Array[Int]])(b: (NumRS, NumRS, Rep[Int]) => NumRS @diff): NumRS @diff = shift {
    (k: NumRS => Unit) =>
    lazy val tree: (Rep[Int], NumRS => Unit, NumRS) => Unit = FUN { (i: Rep[Int], k: NumRS => Unit, x: NumRS) =>
      def shift_tree = (i: Rep[Int]) => shift { k: (NumRS => Unit) => tree(i, k, x) }
      RST(k( IF(i >= 0){ b(shift_tree(lch(i)), shift_tree(rch(i)), i) } { x } ))
    }
    tree(0, k, init)
  }

}
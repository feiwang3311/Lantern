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
  class NumFS(val x: Rep[Array[Double]], val d: Rep[Array[Double]]) {
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
  }

  class NumRS(val x: NumFS, val d: NumFS) {
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
  def assertVectorEqual(result: (Rep[Double], Rep[Double]), expected: (Rep[Double], Rep[Double])) =
    if (result._1 != expected._1 || result._2 != expected._2) {
      printf("(%f, %f) is not as expected (%f, %f)", result._1, result._2, expected._1, expected._2)
      error("")
    }

}
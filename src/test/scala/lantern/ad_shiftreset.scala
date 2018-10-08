/*
package lantern

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._
import org.scalatest.FunSuite

object ShiftReset {
  
  def main(args: Array[String]): Unit = {

    import scala.util.continuations._
    type diff = cps[Unit]

    class Num(val x: Double, var d: Double) {
      def +(that: Num) = shift { (k: Num => Unit) => 
        val y = new Num(x + that.x, 0.0); k(y)
        this.d += y.d; that.d += y.d 
      }
      def -(that: Num) = shift { (k: Num => Unit) =>
        val y = new Num(x - that.x, 0.0); k(y)
        this.d += y.d; that.d -= y.d 
      }
      def *(that: Num) = shift { (k: Num => Unit) => 
        val y = new Num(x * that.x, 0.0); k(y)
        this.d += that.x * y.d; that.d += this.x * y.d 
      }
      def sin() = shift { (k: Num => Unit) =>
        val y = new Num(math.sin(x), 0.0); k(y)
        this.d += math.cos(x) * y.d
      }
      def >(that: Num) = this.x > that.x
    }

    object NumMath {
      def sin(x: Num) = x.sin()
    }

    implicit def toNum(x: Double): Num = new Num(x, 0.0)

    def grad(f: Num => Num @diff)(x: Double) = {
      val x1 = new Num(x, 0.0)
      reset { f(x1).d = 1.0 }
      x1.d
    }

    def print_result_and_set_gradient(t: Num) = {
        t.d = 1.0; 
        val temp = t.x; 
        println(s"result of function is $temp")
    }

    def grad_with_value(f: Num => Num @diff)(x: Double) = {
      val x1 = new Num(x, 0.0)
      reset { 
        print_result_and_set_gradient(f(x1))
      }
      x1.d
    }

    def grad_two_inputs(f: (Num, Num) => Num @diff)(v0: Double, v1: Double) = {
      val x1 = new Num(v0, 0.0)
      val x2 = new Num(v1, 0.0)
      reset {
        print_result_and_set_gradient(f(x1, x2))
      }
      (x1.d, x2.d)
    }

    def grad_array(f: (Array[Num]) => Num @diff)(v0: Array[Double]) = {
      val x0 = v0.map(new Num(_, 0.0))
      reset {
        print_result_and_set_gradient(f(x0))
      }
      x0.map(_.d)
    }

    for (x <- 0 until 10) {
      assert(grad(x => x + x*x*x)(x) == 1 + 3*x*x)
    }

    println("done")

    /*** transfer the test examples from ad.scala to here ***/
    /** Step 1: simple example **/
    println("Step 1")
    // Simple function 
    def f_simple(v0: Num): Num @diff = {
      val v1 = v0 * v0
      val v2 = v0 + v1
      v2
    }

    // Hand-coded correct derivative
    def gf_simple(v0: Double) = 2 * v0 + 1

    // assert the derivative to be correct
    val gradient = grad_with_value(f_simple)(3.0)
    println(s"gradient is $gradient")
    assert(gradient == gf_simple(3.0))

    /** Step 2: simple control flow (if-statement) **/
    println("Step 2")
    // Function with if-statement
    def fif(x1: Num, x2: Num): Num @diff = {
      // Max operation
      if (x1 > x2) x1 else x2
    }

    // Hand-coded correct derivative
    def gfif(x1: Double, x2: Double): (Double, Double) = {
      if (x1 > x2) (1.0, 0.0) else (0.0, 1.0) 
    } 

    val gradient_if = grad_two_inputs(fif)(2.4, 3.9)
    println(s"gradient is $gradient_if")
    assert (gradient_if == gfif(2.4, 3.9))   

    println("Step 2b")
    // Function with if-statement (like the one in ad_lms.scala)
    def fif2(x: Num): Num @diff = {
      if (x > 0.0) x * x else -1.0 * x * x
    }

    // Hand-coded correct derivative
    def gfif2(x: Double): Double = {
      if (x > 0) 2.0 * x else -2.0 * x
    }

    val grad_if2 = grad_with_value(fif2)(-2.0)
    println(s"gradient is $grad_if2")
    assert (grad_if2 == gfif2(-2.0))

    /** Step 3: linear direct recursive function **/
    println("Step 3")
    // Linear recursive function
    def fr(x: Num): Num @diff = {
      // Divide by 2.0 until less than 1.0
      if (x > 1.0) fr(0.5 * x) else x 
      /* 
        Question: what is a better way to handle constant Double in model?
        Discussion: 1. avoid 0.5 * x, always do x * 0.5, and support * operation in Num that takes Double
                    2. define implicit function to lift Double to Num, discard gradient of it
                    3. add stop_gradient flag in Num so that gradient propagation is conditional,
                       define implicit function to lift Double to Num with stop_gradient flag 
        Fei's opinion: Use 3. The feature of "stop_gradient" is useful for normal Num too, if users don't need 
                              the gradient of some weights.
                       current implementation use 2.
      */
    }

    // Hand-coded correct derivative
    def gfr(x: Double): Double = {
      if (x > 1.0) 0.5 * gfr(0.5 * x) else 1.0
    }
 
    val gradient_ldr = grad_with_value(fr)(9.0)
    println(s"gradient is $gradient_ldr")
    assert (gradient_ldr == gfr(9.0))

    /** Step 4: non-linear direct recursion **/
    println("Step 4")
    // Non-linear recursive function
    def frs(x: Num): Num @diff = {
      // Take sin(x) until the value is smaller than 0.5
      if (x > 0.5) frs(NumMath.sin(x)) else x
      /* Question: is this NumMath object a good way to handle sin, cos, log, exp? 
                   other suggestions?
      */
    }

    // Hand-coded correct derivative
    def gfrs(x: Double): Double = {
      if (x > 0.5) math.cos(x) * gfrs(math.sin(x)) else 1.0
    }

    val gradient_nldr = grad_with_value(frs)(0.6)
    println(s"gradient is $gradient_nldr")
    assert (gradient_nldr == gfrs(0.6))

    /** Step 5: mutual recursion **/
    println("Step 5")
    // Mutually recursive functions
    def fn2(x: Num): Num @diff = {
      if (x > 1.0) fn3(x * 0.5) else x
    }
    def fn3(x: Num): Num @diff = {
      if (x > 1.0) fn2(x * (1.0/3.0)) else x
    }

    // Hand-code correct derivatives
    def gfn2(x: Double): Double = {
      if (x > 1.0) gfn3(x * 0.5) * 0.5 else 1.0
    }
    def gfn3(x: Double): Double = {
      if (x > 1.0) gfn2(x * (1.0/3.0)) * (1.0/3.0) else 1.0
    }

    val gradient_mr = grad_with_value(fn2)(17.0)
    println(s"gradient is $gradient_mr")
    assert (gradient_mr == gfn2(17.0))

    /** Step 6: loops **/
    /** SubStep 6.1: handle step 3 in while construct **/
    println("Step 6.1")
    // While-loop: handle step 3
    def f_while_1(a: Num): Num @diff = {
      var temp = a
      while (temp > 1.0) {
        temp = temp * 0.5 
      }
      temp
    }

    val gradient_w1 = grad_with_value(f_while_1)(9.0)
    println(s"gradient is $gradient_w1")
    assert (gradient_w1 == gfr(9.0))

    println("Step 6.2")
    // While-loop, imperative implementation with mutable vars
    def f_while_2(a: Num, b: Num): Num @diff = {
      var r = toNum(1.0)
      var b1 = b
      while (b1 > 0) {
        r = a * r
        b1 = b1 - 1
      }
      r
    }

    // Hand-coded correct derivation with stack (translating from tangent)
    import scala.collection.mutable.Stack
    def gf_while_2(a: Double, b: Double): (Double, Double) = {
      // Initialize stack and loop count
      var stack = Stack.empty[Double]
      var count = 0

      // Copy forward pass, incrementing loop count and pushing operands onto stack
      var r = 1.0
      var b1 = b
      while (b1 > 0) {
        count += 1
        stack.push(r)
        r = a * r
        stack.push(b1)
        b1 = b1 -1
      }

      var dydr = 1.0 
      var dyda = 0.0
      var dydb1 = 0.0
      // Perform backward pass, popping operands off the stack for gradient calculations
      for (_ <- 1 to count) {
        // grad of: b1 = b1 - 1
        val _b1 = b1
        b1 = stack.pop()
        dydb1 = dydb1 

        // grad of: r = a * r
        val _r = r
        r = stack.pop()
        dyda += dydr * r
        dydr = dydr * a
      }
      (dyda, dydb1)
    }

    val gradient_w2 = grad_two_inputs(f_while_2)(2.0, 4.0) 
    /* Question: undifferentiable points "b" for now has gradient of 0.0.
                 is this a good default?
                 */
    println(s"gradient is $gradient_w2")
    assert (gradient_w2 == gf_while_2(2.0, 4.0))


    /** Step 7: Sum of array **/
    println("Step 7.1")
    // test a sum function in while loop
    def sum(v: Array[Num]): Num @diff = {
      var r = toNum(0.0)
      var i = 0
      while (i < v.length) {
        r = v(i) + r // this compiles
        i += 1
      }
      r
    } 

    val gradient_sum = grad_array(sum)(Array(12, 6, 15))
    // println(gradient_sum.getClass.getName)
    print_array(gradient_sum)
    def print_array[T](v: Array[T]) = {
      print("(")
      v.foreach(t => print(s"$t,"))
      println(")")
    }
    assert(gradient_sum.sameElements(Array(1.0, 1.0, 1.0)))

    println("Step 7.2")
    // generalize sum to a reduce_left function
    // It is important to re-implement the reduce_left function and add @diff notation to f parameter!!
    def reduce_left(v: Array[Num])(f: (Num, Num) => Num @diff): Num @diff = { 
      var r = v(0)
      var i = 1
      while (i < v.length) {
        r = f(v(i), r)
        i += 1
      } 
      r
    }

    def sum2(v: Array[Num]) = {
      reduce_left(v)(_ + _)
    }

    val gradient_sum2 = grad_array(sum2)(Array(12, 6, 15))
    print_array(gradient_sum2)
    assert (gradient_sum2.sameElements(Array(1.0, 1.0, 1.0)))

    println("Step 8, fib")
    def fib(n: Int)(seed: Num): Num @diff = {
      if (n == 1) seed
      else if (n == 2) seed + seed + 1
      else {
        val x1 = fib(n-2)(seed)
        val x2 = fib(n-1)(seed)
        x1 + x2
      } 
    }

    val ggg = grad_with_value(fib(6))(2.0)
    println(s"grad is $ggg")
    println("done")


  }
}
*/
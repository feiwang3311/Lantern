/*
package lantern

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._
import org.scalatest.FunSuite

/** in this file I try to produce Jacobian **/
object Jacobian {
  
  def main(args: Array[String]): Unit = {

    import scala.util.continuations._
    type diff = cps[Unit]

    /** Step 1, R^2 -> R^2 function **/
    class Num2(val x: Double, var d1: Double, var d2: Double) {
      def +(that: Num2) = shift { (k: Num2 => Unit) => 
        val y = new Num2(x + that.x, 0.0, 0.0); k(y)
        this.d1 += y.d1; that.d1 += y.d1;
        this.d2 += y.d2; that.d2 += y.d2 
      }
      def *(that: Num2) = shift { (k: Num2 => Unit) => 
        val y = new Num2(x * that.x, 0.0, 0.0); k(y)
        this.d1 += that.x * y.d1; that.d1 += this.x * y.d1;
        this.d2 += that.x * y.d2; that.d2 += this.x * y.d2; 
      }
    }

    def print_result_and_set_gradient(t: (Num2, Num2)) = {
        t._1.d1 = 1.0; t._1.d2 = 0.0;
        t._2.d1 = 0.0; t._2.d2 = 1.0;
        val temp = (t._1.x, t._2.x); 
        println(s"result of function is $temp")
    }

    def grad_with_value(f: (Num2, Num2) => (Num2, Num2) @diff)(x1: Double, x2: Double) = {
      val xx1 = new Num2(x1, 0.0, 0.0)
      val xx2 = new Num2(x2, 0.0, 0.0)
      reset { 
        print_result_and_set_gradient(f(xx1, xx2))
      }
      (xx1.d1, xx1.d2, xx2.d1, xx2.d2)
    }

    /* test 1 */
    println("Test 1: R^2 -> R^2")
    def f(v0: Num2, v1: Num2) = {
    	(v0 * v1, v0 + v1)
    }

    val jacob = grad_with_value(f)(3.0, 2.0)
    println(jacob)
    assert (jacob == (2.0, 1.0, 3.0, 1.0))

    /** Step 2, R^2 -> R^n function **/
    import scala.collection.mutable.ArrayBuffer
    
    class Num(val x : Double, val d: ArrayBuffer[Double]) {
      def +(that: Num) = shift { (k: Num => Unit) =>
      	val y = new Num(x + that.x, new ArrayBuffer[Double]()); k(y)
      	bufferFun(this.d, y.d)(_ + _)
      	bufferFun(that.d, y.d)(_ + _)
      }
      def *(that: Num) = shift { (k: Num => Unit) => 
      	val y = new Num(x * that.x, new ArrayBuffer[Double]()); k(y)
      	bufferFun(this.d, y.d)(_ + _ * that.x)
      	bufferFun(that.d, y.d)(_ + _ * this.x)
      }
    } 

  	def bufferFun(a: => ArrayBuffer[Double], b: => ArrayBuffer[Double])(f: (Double, Double) => Double) = {
    	val n = b.length
    	val status = a.length == 0
    	var i = 0
    	while (i < n) {
    		if (status) a.append(f(0.0, b(i))) else a(i) = f(a(i), b(i))
    		i += 1
    	}    	
    }

    def print_result_and_set_gradient_array(t: ArrayBuffer[Num]) = {
    	val n = t.length
    	var i = 0
    	while (i < n) {
 			var j = 0
 			while (j < n) {
 				t(i).d.append(if (j == i) 1.0 else 0.0)
 				j += 1
 			}
 			i += 1
 		}
        val temp = t.map(_.x)
        println(s"result of function is $temp")
    }

    def grad_2(f: (Num, Num) => ArrayBuffer[Num] @diff)(x1: Double, x2: Double) = {
      val xx1 = new Num(x1, new ArrayBuffer[Double]())
      val xx2 = new Num(x2, new ArrayBuffer[Double]())
      reset { 
        print_result_and_set_gradient_array(f(xx1, xx2))
      }
      (xx1.d, xx2.d)
    }

    /* test 2 */
    println("Test 2: R^2 -> [R]")
    def f_2_array(v0: Num, v1: Num) = {
    	val res = new ArrayBuffer[Num]()
    	res.append(v0 + v1 * new Num(1, new ArrayBuffer[Double]()))
    	res.append(v0 + v1 * new Num(2, new ArrayBuffer[Double]()))
    	res.append(v0 + v1 * new Num(3, new ArrayBuffer[Double]()))
    	res.append(v0 + v1 * new Num(4, new ArrayBuffer[Double]()))
    	res
    }

    val jacob2 = grad_2(f_2_array)(2.0, 3.0)
    println(jacob2)
    assert (jacob2 == (ArrayBuffer(1.0, 1.0, 1.0, 1.0),ArrayBuffer(1.0, 2.0, 3.0, 4.0)))

    /** Step 3:  R^n -> R^n **/
    println("Test 3: [R] -> [R]")

    def grad_array(f: ArrayBuffer[Num] => ArrayBuffer[Num] @diff)(x: ArrayBuffer[Double]) = {
    	val v = x.map(new Num(_, new ArrayBuffer[Double]()))
    	reset {
    		print_result_and_set_gradient_array(f(v))
    	}
    	v.map(_.d)
    }

    def f_array_array(v: ArrayBuffer[Num]) = {
      val n = v.length
      val res = new ArrayBuffer[Num]()
      var r = 0
      while (r < n) {
        if (r == 0) res.append(v(0) + new Num(0.0, new ArrayBuffer[Double]()))
        else res.append(res(r-1) + v(r))
        r += 1
      }
      res
    }

    val jacob3 = grad_array(f_array_array)(ArrayBuffer(1.0, 2.0, 3.0, 4.0))
    println(jacob3)
    assert (jacob3 == ArrayBuffer(ArrayBuffer(1.0, 1.0, 1.0, 1.0), ArrayBuffer(0.0, 1.0, 1.0, 1.0), 
    	ArrayBuffer(0.0, 0.0, 1.0, 1.0), ArrayBuffer(0.0, 0.0, 0.0, 1.0)))

    /* Step 4: See if I can get hessian from Jacobian 
    type diff2 = cps[Unit] @scala.util.continuations.cpsParam[Unit,Unit] @scala.util.continuations.cpsSynth

    class NumRF(val x: Double, var d: Num) {
      def *(that: NumRF): diff2 = shift { (k: NumRF => Unit) =>
        val y = new NumRF(x * that.x, new Num(0.0, new ArrayBuffer[Double]())); k(y)
        this.d = this.d + new Num(that.x, new ArrayBuffer[Double]()) * y.d
        that.d = that.d + new Num(this.x, new ArrayBuffer[Double]()) * y.d
      }
    }

    def gradRF(f: NumRF => NumRF @diff)(v0: Num) = {
      val x1 = new NumRF(v0.x, new Num(0.0, new ArrayBuffer[Double]()))
      reset {
        f(x1).d = new Num(1.0, new ArrayBuffer[Double]())
      }
      new Array(x1.d)
    }

    def f_simple(v0: NumRF) = v0 * v0
    val v1: (Num => Num @diff) = 
      (t: Num) => gradRF(f_simple)(t) // this should be a function that can be fed into Jacobian
      */
  }
}

/** Use Map instead of Array[Double] in Jacobian **/
object Jacobian_Map {
  def main(args: Array[String]): Unit = {

    import scala.util.continuations._
    type diff = cps[Unit]
    import scala.collection.mutable.{Map, HashMap, ArrayBuffer}
    
    class Num(val x : Double, val d: Map[Num, Double]) {
      def +(that: Num) = shift { (k: Num => Unit) =>
        val y = new Num(x + that.x, new HashMap[Num, Double]()); k(y)
        mapFun(this.d, y.d)(_ + _)
        mapFun(that.d, y.d)(_ + _)
      }
      def *(that: Num) = shift { (k: Num => Unit) => 
        val y = new Num(x * that.x, new HashMap[Num, Double]()); k(y)
        mapFun(this.d, y.d)(_ + _ * that.x)
        mapFun(that.d, y.d)(_ + _ * this.x)
      }
    }     

    def mapFun(a: => Map[Num, Double], b: => Map[Num, Double])(f: (Double, Double) => Double) = {
      // update values in a by values in b with function in f
      val b_iter = b.iterator
      while (b_iter.hasNext) {
        val (b_num, b_double) = b_iter.next()
        val a_double = a.getOrElse(b_num, 0.0)
        a.update(b_num, f(a_double, b_double))
      }
    }

    var targetArray = new ArrayBuffer[Num]() // keep track of all targets

    def print_result_and_set_gradient_array(t: ArrayBuffer[Num]) = {
      val n = t.length
      var i = 0
      while (i < n) {
        t(i).d.update(t(i), 1.0)
        i += 1
      }
      val temp = t.map(_.x)
      println(s"result of function is $temp")
      targetArray = t                        // update the target array for use in grad_array function
    }

    /** test:  R^n -> R^n **/
    println("Test (Map implementation): [R] -> [R]")

    def grad_array(f: ArrayBuffer[Num] => ArrayBuffer[Num] @diff)(x: ArrayBuffer[Double]) = {
      val v = x.map(new Num(_, new HashMap[Num, Double]()))
      reset {
        print_result_and_set_gradient_array(f(v))
      }
      v.map(vv => targetArray.map(n => vv.d.getOrElse(n, 0.0)))
    }

    def f_array_array(v: ArrayBuffer[Num]) = {
      val n = v.length
      val res = new ArrayBuffer[Num]()
      var r = 0
      while (r < n) {
        if (r == 0) res.append(v(0) + new Num(0.0, new HashMap[Num, Double]()))
        else res.append(res(r-1) + v(r))
        r += 1
      }
      res
    }

    val jacob3 = grad_array(f_array_array)(ArrayBuffer(1.0, 2.0, 3.0, 4.0))
    println(jacob3)
    assert (jacob3 == ArrayBuffer(ArrayBuffer(1.0, 1.0, 1.0, 1.0), ArrayBuffer(0.0, 1.0, 1.0, 1.0), 
      ArrayBuffer(0.0, 0.0, 1.0, 1.0), ArrayBuffer(0.0, 0.0, 0.0, 1.0)))
  }
}

/** Use Forward propagation in Jacobian **/
object Jacobian_Forward {
  def main(args: Array[String]): Unit = {

    import scala.collection.mutable.{Map, HashMap, ArrayBuffer}

    class Num(val x : Double, val d: Map[Num, Double]) {
      def +(that: Num) = {
        new Num(x + that.x, mapFun(this.d, that.d)(_ + _))
      }
      def *(that: Num) = {
        new Num(x * that.x, mapFun(this.d, that.d)(_ * that.x + _ * this.x))
      }
    }

    def mapFun(a: => Map[Num, Double], b: => Map[Num, Double])(f: (Double, Double) => Double) = {
      // return a new Map with Union of Keys from a and b, 
      // and values compounded by f (if key not exists in one map, use 0.0 instead
      val res = new HashMap[Num, Double]()
      for (key <- a.keySet ++ b.keySet) {
        res.update(key, f(a.getOrElse(key, 0.0), b.getOrElse(key, 0.0)))
      }
      res
    }

    def grad_array(f: ArrayBuffer[Num] => ArrayBuffer[Num])(x: ArrayBuffer[Double]) = {
      val v = x.map(new Num(_, new HashMap[Num, Double]()))
      v.foreach(vv => vv.d.update(vv, 1.0)) // initialize the tangent of input parameters to be 1.0
      val y = f(v)
      val results = y.map(_.x)
      val jacobs = y.map(yy => v.map(yy.d.getOrElse(_, 0.0)))
      (results, jacobs)
    }

    def f_array_array(v: ArrayBuffer[Num]) = {
      val n = v.length
      val res = new ArrayBuffer[Num]()
      var r = 0
      while (r < n) {
        if (r == 0) res.append(v(0) + new Num(0.0, new HashMap[Num, Double]()))
        else res.append(res(r-1) + v(r))
        r += 1
      }
      res
    }

    val (result, jacob) = grad_array(f_array_array)(ArrayBuffer(1.0, 2.0, 3.0, 4.0))
    println(s"the result of function is $result")
    println(s"the jacobians are $jacob")
    assert (result == ArrayBuffer(1.0, 3.0, 6.0, 10.0))
    assert (jacob == ArrayBuffer(ArrayBuffer(1.0, 0.0, 0.0, 0.0), ArrayBuffer(1.0, 1.0, 0.0, 0.0), 
      ArrayBuffer(1.0, 1.0, 1.0, 0.0), ArrayBuffer(1.0, 1.0, 1.0, 1.0))) // Note this result is different from reverse mode: FIXME
  }
}

/* with Jacobian_Forward, I try Jacobian of Gradient as Hessian */
object Hessian_JacobianOfGradient {
  def main(args: Array[String]): Unit = {

    import scala.collection.mutable.{Map, HashMap, ArrayBuffer}    
    import scala.util.continuations._
    type diff = cps[Unit]

    class NumF(val x : Double, val d: Map[NumF, Double]) {
      def +(that: NumF) = {
        new NumF(x + that.x, mapFun(this.d, that.d)(_ + _))
      }
      def *(that: NumF) = {
        new NumF(x * that.x, mapFun(this.d, that.d)(_ * that.x + _ * this.x))
      }
    }

    def mapFun(a: => Map[NumF, Double], b: => Map[NumF, Double])(f: (Double, Double) => Double) = {
      // return a new Map with Union of Keys from a and b, 
      // and values compounded by f (if key not exists in one map, use 0.0 instead
      val res = new HashMap[NumF, Double]()
      for (key <- a.keySet ++ b.keySet) {
        res.update(key, f(a.getOrElse(key, 0.0), b.getOrElse(key, 0.0)))
      }
      res
    }

    implicit def toNumF(x: Double) = new NumF(x, new HashMap[NumF, Double]())
    implicit def toNumR(x: Double) = new NumR(x, 0.0)

    class NumR(val x: NumF, var d: NumF) {
      def +(that: NumR) = shift { (k: NumR => Unit) => 
        val y = new NumR(x + that.x, 0.0); k(y)
        this.d = this.d + y.d; that.d = that.d + y.d
      }
      def *(that: NumR) = shift { (k: NumR => Unit) => 
        val y = new NumR(x * that.x, 0.0); k(y)
        this.d = this.d + that.x * y.d; that.d = that.d + this.x * y.d
      }
    }

    def finalClosure(t: NumR) = {
      val result = t.x.x // this is the value of the function
      println(s"value of the function is $result")
      t.d = 1.0          // set the gradient value to be 1.0, the gradient tangent remains empty
    }

    /* tests: R^2 -> R */
    println("test for R^2 -> R")
    def grad_two_inputs(f: (NumR, NumR) => NumR @diff)(v0: Double, v1: Double) = {
      val x1 = new NumR(v0, 0.0); x1.x.d.update(x1.x, 1.0)
      val x2 = new NumR(v1, 0.0); x2.x.d.update(x2.x, 1.0)
      reset {
        finalClosure(f(x1, x2))
      }
      val gradient = (x1.d.x, x2.d.x)
      val hessian = ((x1.d.d.getOrElse(x1.x, 0.0), x1.d.d.getOrElse(x2.x, 0.0)),
                     (x2.d.d.getOrElse(x1.x, 0.0), x2.d.d.getOrElse(x2.x, 0.0)))
      (gradient, hessian)
    }

    val (grad, hess) = grad_two_inputs((x1, x2) => x1 * x2)(2.0, 3.0)
    println(grad)
    println(hess)
    assert (grad == (3.0, 2.0))
    assert (hess == ((0.0, 1.0),(1.0, 0.0)))

    /* tests: R^n -> R */
    println("tests: R^n -> R")
    def grad_array(f: ArrayBuffer[NumR] => NumR @diff)(x: ArrayBuffer[Double]) = {
      val v = x.map(new NumR(_, 0.0))
      v.foreach(vv => vv.x.d.update(vv.x, 1.0)) // initialize the tangent of input parameters to be 1.0
      reset{ 
        finalClosure(f(v))
      }
      val gradient = v.map(_.d.x)
      val hessian = v.map(vv => v.map(vi => vv.d.d.getOrElse(vi.x, 0.0)))
      (gradient, hessian)
    }

    def f_array(v: ArrayBuffer[NumR]): NumR @diff = {
      val n = v.length; assert (n > 0)
      var temp = v(0)
      var r = 1
      while (r < n) {
        temp = temp + (r+1) * v(r)
        r += 1
      }
      temp
    }

    val (grad2, hess2) = grad_array(f_array)(ArrayBuffer(1.0, 2.0, 3.0, 4.0))
    println(s"the gradient of function is $grad2")
    println(s"the hessian are $hess2")
    assert (grad2 == ArrayBuffer(1.0, 2.0, 3.0, 4.0))
    assert (hess2 == ArrayBuffer(ArrayBuffer(0.0, 0.0, 0.0, 0.0), ArrayBuffer(0.0, 0.0, 0.0, 0.0), 
      ArrayBuffer(0.0, 0.0, 0.0, 0.0), ArrayBuffer(0.0, 0.0, 0.0, 0.0))) 

    def f_array2(v: ArrayBuffer[NumR]): NumR @diff = {
      val n = v.length; assert (n > 0)
      var temp = v(0)
      var r = 1
      while (r < n) {
        temp = temp * v(r)
        r += 1
      }
      temp
    }

    println("more tests: R^n -> R")
    val (grad3, hess3) = grad_array(f_array2)(ArrayBuffer(1.0, 2.0, 3.0, 4.0))
    println(s"the gradient of function is $grad3")
    println(s"the hessian are $hess3")
    assert (grad3 == ArrayBuffer(24.0, 12.0, 8.0, 6.0))
    assert (hess3 == ArrayBuffer(ArrayBuffer(0.0, 12.0, 8.0, 6.0), ArrayBuffer(12.0, 0.0, 4.0, 3.0), 
      ArrayBuffer(8.0, 4.0, 0.0, 2.0), ArrayBuffer(6.0, 3.0, 2.0, 0.0)))
  }
}


/* with Jacobian_Forward, I try Hessian_vector product as in "Fast Exact Multiplication by the Hessian" */
object Hessian_vector_product {
  def main(args: Array[String]): Unit = {

    import scala.collection.mutable.ArrayBuffer
    import scala.util.continuations._
    type diff = cps[Unit]
    
    /* because we are calculating Hessian_vector product, the tangent is not a Map, but a number
       By the paper "Fast Exact Multiplication by the Hessian", Hv = diff_{r=0} {G(w + rv)} = J(G(w))*v
       So if we are looking at a function R^n -> R,
       basically we are doing gradient of G(w) at position w, but in regard to a fix direction v, and only one variable r
       In terms of implementation, we need to change the tangent in the forward pass (from a Map to just a number),
       this change should reflect the norm and direction of v
     */
    class NumF(val x : Double, val d: Double) {
      def +(that: NumF) = {
        new NumF(x + that.x, this.d + that.d)
      }
      def *(that: NumF) = {
        new NumF(x * that.x, this.d * that.x + that.d * this.x)
      }
    }

    implicit def toNumF(x: Double) = new NumF(x, 0.0)
    implicit def toNumR(x: Double) = new NumR(x, 0.0)

    class NumR(val x: NumF, var d: NumF) {
      def +(that: NumR) = shift { (k: NumR => Unit) => 
        val y = new NumR(x + that.x, 0.0); k(y)
        this.d = this.d + y.d; that.d = that.d + y.d
      }
      def *(that: NumR) = shift { (k: NumR => Unit) => 
        val y = new NumR(x * that.x, 0.0); k(y)
        this.d = this.d + that.x * y.d; that.d = that.d + this.x * y.d
      }
    }

    def finalClosure(t: NumR) = {
      val result = t.x.x // this is the value of the function
      println(s"value of the function is $result")
      t.d = 1.0          // set the gradient value to be 1.0, the gradient tangent remains empty
    }

    /* tests: R^2 -> R */
    println("test for R^2 -> R")
    def grad_two_inputs(f: (NumR, NumR) => NumR @diff)(v0: Double, v1: Double)(v: (Double, Double)) = {
      val x1 = new NumR(new NumF(v0, v._1), 0.0)
      val x2 = new NumR(new NumF(v1, v._2), 0.0)
      reset {
        finalClosure(f(x1, x2))
      }
      val gradient = (x1.d.x, x2.d.x)
      val hessian_vector = (x1.d.d, x2.d.d)
      (gradient, hessian_vector)
    }

    val (grad, hess) = grad_two_inputs((x1, x2) => x1 * x2)(2.0, 3.0)((4.0, 5.0))
    println(s"the gradient is $grad")
    println(s"the hessian vector product is $hess")
    assert (grad == (3.0, 2.0))
    assert (hess == (5.0, 4.0)) 

    /* tests: R^n -> R */
    println("tests: R^n -> R")
    def grad_array(f: ArrayBuffer[NumR] => NumR @diff)(x: ArrayBuffer[Double])(vd: ArrayBuffer[Double]) = {
      // basic assertion check
      val n = x.length
      assert (n == vd.length)
      assert (n > 0)
      // initialize the input
      val v = (x zip vd).map(z => new NumR(new NumF(z._1, z._2), 0.0))
      reset{ 
        finalClosure(f(v))
      }
      val gradient = v.map(_.d.x)
      val hessian_vector = v.map(_.d.d)
      (gradient, hessian_vector)
    }

    def f_array(v: ArrayBuffer[NumR]): NumR @diff = {
      val n = v.length; assert (n > 0)
      var temp = v(0)
      var r = 1
      while (r < n) {
        temp = temp + (r+1) * v(r)
        r += 1
      }
      temp
    }

    val (grad2, hess2) = grad_array(f_array)(ArrayBuffer(1.0, 2.0, 3.0, 4.0))(ArrayBuffer(5.0, 6.0, 7.0, 8.0))
    println(s"the gradient of function is $grad2")
    println(s"the hessian_vector product is $hess2")
    assert (grad2 == ArrayBuffer(1.0, 2.0, 3.0, 4.0))
    assert (hess2 == ArrayBuffer(0.0, 0.0, 0.0, 0.0)) 

    def f_array2(v: ArrayBuffer[NumR]): NumR @diff = {
      val n = v.length; assert (n > 0)
      var temp = v(0)
      var r = 1
      while (r < n) {
        temp = temp * v(r)
        r += 1
      }
      temp
    } 

    println("more tests: R^n -> R")
    val (grad3, hess3) = grad_array(f_array2)(ArrayBuffer(1.0, 2.0, 3.0, 4.0))(ArrayBuffer(5.0, 6.0, 7.0, 8.0))
    println(s"the gradient of function is $grad3")
    println(s"the hessian vector product is $hess3")
    assert (grad3 == ArrayBuffer(24.0, 12.0, 8.0, 6.0))
    assert (hess3 == ArrayBuffer(176.0, 112.0, 80.0, 62.0))
  }
}
*/
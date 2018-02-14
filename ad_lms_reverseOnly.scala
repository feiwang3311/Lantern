import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

object LMS_Reverse {

  trait DiffApi extends Dsl {

 	type diff = cps[Unit]

    type RDouble = Rep[Double]

    class NumR(val x: RDouble, val d: Var[Double]) extends Serializable {
      def +(that: NumR): NumR @diff = shift { (k: NumR => Unit) => 
        val y = new NumR(x + that.x, var_new(0.0)); k(y)
        this.d += y.d; that.d += y.d }
      def -(that: NumR): NumR @diff = shift { (k: NumR => Unit) =>
      	val y = new NumR(x - that.x, var_new(0.0)); k(y)
      	this.d += y.d; that.d -= y.d
      }
      def *(that: NumR): NumR @diff = shift { (k: NumR => Unit) => 
        // is it worth optimizing x*x --> 2*x (this == that)?
        val y = new NumR(x * that.x, var_new(0.0)); k(y)
        this.d += that.x * y.d; that.d += this.x * y.d }
    }

    implicit def toNumR(x: Double) = new NumR(x, var_new(0.0))

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
      // val k1 = FUN(k)

      lazy val loop: NumR => Unit = FUN { (x: NumR) =>
        if (c(x)) RST(loop(b(x))) else RST(k(x))
      }
      loop(init)
    }

    def gradR(f: NumR => NumR @diff)(x: RDouble): Rep[Double] = {
      val x1 = new NumR(x, var_new(0.0))
      reset { var_assign(f(x1).d, 1.0); () }
      x1.d
    }

    def gradR_array(f: Array[NumR] => NumR @diff)(x: Array[RDouble]) = {
    	val x1 = x.map(new NumR(_, var_new(0.0)))
    	reset { var_assign(f(x1).d, 1.0); () }
    	x1.map((t: NumR) => readVar(t.d))
    }
  }

  def main(args: Array[String]): Unit = {

    val gr1 = new DslDriverC[Double,Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        gradR(x => x + x*x*x)(x)
      }
    }    

    println("---- reverse mode with dynamic vars ---- \n")
    println(gr1.code)

    println("---- optimize for a goal ------ \n")
    
    val grad = new DslDriver[Double, Double] with DiffApi {

	    def model(x : NumR) = x + x * x *x
	    val goal = 5.0
	    def loss(x: NumR) = (model(x) + (0-goal)) * (model(x) + (0-goal))

    	def snippet(x: RDouble): RDouble = {
    		gradR(loss)(x)
    	}
    }

    def grad_desc(init: Double, lr: Double, maxiter: Int) = {
    	var temp = init
    	for (i <- 0 until maxiter) {
    		val gg = grad.eval(temp)
    		temp = temp - gg * lr
    	}
    	temp
    }
    val x_opt = grad_desc(0.0, 0.005, 10000)
    println(s"x_opt is $x_opt")
    println("---- the code generated ----")
    println(grad.code)

    println("---- optimze for a goal, with all logic generated ---- \n")
    val grad_desc_all_generate = new DslDriver[Double, Double] with DiffApi {
    	
    	def model(x: NumR) = x + x*x*x
    	val goal = 5.0
    	def loss(x: NumR) = (model(x) + (0-goal)) * (model(x) + (0-goal))
    	val lr = 0.005
    	val eps = 0.001 // this is not used
    	val maxIter = 10000
    	def snippet(x: RDouble): Rep[Double] = {
    		val temp = var_new(x) // using var temp = x doesn't work
    		for (i <- (0 until maxIter): Rep[Range]) {
    			val gg = gradR(loss)(readVar(temp))
    			temp -= gg * lr
    		}
    		readVar(temp)
    	}
    }
    println(grad_desc_all_generate.code)
    println(grad_desc_all_generate.eval(0.0))

    println("---- optimize for a goal, dealing with array of parameters ---- \n")
    //import scala.collection.mutable.ArrayBuffer
    val grad_array = new DslDriver[Array[Double], Array[Double]] with DiffApi {
    	
    	def reduce_left(v: Array[NumR])(f: (NumR, NumR) => NumR @diff): NumR @diff = { 
	      var r = v(0)
	      var i = 1
	      while (i < v.length) {
	        r = f(v(i), r)
	        i += 1
	      } 
	      r
	    }

	    def model(x: Array[NumR]) = reduce_left(x)(_ + _)
	    val goal = 10.0
	    def loss(x: Array[NumR]) = (model(x) + (0-goal)) * (model(x) + (0-goal))
	    def lr = 0.005
	    val maxIter = 1000
     	def snippet(v: Rep[Array[Double]]): Rep[Array[Double]] = {
     		val temp = new Array[Var[Double]](3)
     		for (i <- (0 until 3): Range)
     		  temp(i) = var_new(v(i))
     		// val temp = v.map( (t: Rep[Double]) => var_new(t) )
     		for (i <- (0 until maxIter): Rep[Range]) {
     			val gg = gradR_array(loss)(temp.map(readVar(_)))
     			(temp zip gg).foreach({case (t, g) => t -= g * lr})
     		}

            val res = NewArray[Double](3)
            for (i <- (0 until 3): Range)
            	res(i) = readVar(temp(i))
            res
     		// temp.map(readVar(_))
		}
    }
    println(grad_array.code)
    val res = grad_array.eval(scala.Array(0.0, 1.0, 2.0))
    res.foreach(r => print(s"$r "))

    println("---- optimize a matrix with linear goal ---- \n")
    /*
    val grad_matrix = new DslDriver[Array[Double], Array[Double]] with DiffApi {

    	val n = 3 // 3*3 M 

    	def dot(v: Array[NumR], w: Array[NumR]): NumR @diff = {    		
    		var r = v(0) * w(0)
    		var i = 1
    		while (i < n) {
    			r = r + v(i) * w(i)
    			i += 1
    		}
    		r
    	}
    	def norm_diff(v: Array[NumR], w: Array[NumR]): NumR @diff = {    		
    		var r = (v(0) - w(0)) * (v(0) - w(0))
    		var i = 1
    		while (i < n) {
    			r = r + (v(i) - w(i)) * (v(i) - w(i))
    			i += 1
    		}
    		r 
    	}
      
    	def linear(M: Array[Array[NumR]], v: Array[NumR]): Array[NumR] @diff = {    		
    		val res = new Array[NumR](n)
    		var i = 0
    		while (i < n) {
    			res(i) = dot(M(i), v)
    			i += 1
    		}
    		res
    	} 

    	val goal = scala.Array(9.0, 17.0, 30.0).map(toNumR)
    	val v = scala.Array(1.0, 2.0, 3.0).map(toNumR)
    	def loss(x: Array[Array[NumR]]) = norm_diff(linear(x, v), goal)
    	def lr = 0.005
    	val maxIter = 1000
    	def snippet(v: Rep[Array[Double]]): Rep[Array[Double]] = {
    		val temp = new Array[Array[Var[Double]]](n)
    		for (i <- (0 until n): Range) {
    			temp(i) = new Array[Var[Double]](n)
    			for (j <- (0 until n): Range) {
    				temp(i)(j) = var_new(v(i * n + j))
    			}
    		}
    		for (i <- (0 until maxIter): Rep[Range]) {
    			val gg = 
    		}
    	}

    }*/
    
  }
}
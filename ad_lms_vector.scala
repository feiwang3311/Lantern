import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

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
	
		**/

		class Vector(val data: Rep[Array[Double]], val dim0: Int /*, val dim1: Int, val dim2: Int*/) {

			def foreach(f: Rep[Double] => Rep[Unit]): Rep[Unit] =
		    	for (i <- 0 until data.length) f(data(i))

		    @virtualize
			def sumIf(f: Rep[Double] => Rep[Boolean]) = { 
		    	val n = var_new(0.0); 
		    	foreach(x => if (f(x)) n += x); 
		    	readVar(n) 
		    }

		    def dot(that: Vector) = {
		    	// assert that and this have the same dimension
		    	// fixME: the result should also be a vector of size 1, to keep the type consistent
		    	val value = var_new(0.0)
		    	for (i <- 0 until data.length) {
		    		value += data(i) * that.data(i)
		    	}
		    	// readVar(res)
		    	val res = NewArray[Double](1)
		    	res(0) = readVar(value)
		    	new Vector(res, 1)
		    }

		    def + (that: Vector) = {
		    	val res = NewArray[Double](dim0)
		    	if (that.dim0 == 1) for (i <- (0 until dim0): Rep[Range]) res(i) = data(i) + that.data(0)
		    	else if (dim0 == 1) for (i <- (0 until dim0): Rep[Range]) res(i) = data(0) + that.data(i)
		    	else if (dim0 == that.dim0) for (i <- (0 until dim0): Rep[Range]) res(i) = data(i) + that.data(i)
		    	else throw new IllegalArgumentException("dimensions of vector do not match!")
		    	new Vector(res, dim0)
		    }

		    def - (that: Vector) = {
		    	val res = NewArray[Double](dim0)
		    	if (that.dim0 == 1) for (i <- (0 until dim0): Rep[Range]) res(i) = data(i) - that.data(0)
		    	else if (dim0 == 1) for (i <- (0 until dim0): Rep[Range]) res(i) = data(0) - that.data(i)
		    	else if (dim0 == that.dim0) for (i <- (0 until dim0): Rep[Range]) res(i) = data(i) - that.data(i)
		    	else throw new IllegalArgumentException("dimensions of vector do not match!")
		    	new Vector(res, dim0)
		    }

		    def * (that: Vector) = {
		    	// element wise multiplication
		    	val res = NewArray[Double](dim0)
		    	if (that.dim0 == 1) for (i <- (0 until dim0): Rep[Range]) res(i) = data(i) * that.data(0)
		    	else if (dim0 == 1) for (i <- (0 until dim0): Rep[Range]) res(i) = data(0) * that.data(i)
		    	else if (dim0 == that.dim0) for (i <- (0 until dim0): Rep[Range]) res(i) = data(i) * that.data(i)
		    	else throw new IllegalArgumentException("dimensions of vector do not match!")
		    	new Vector(res, dim0)
		    }

		    def print() = {
		    	for (i <- (0 until dim0): Rep[Range]) println(data(i))
		    }

		}

		object Vector {

			def randDouble() = unchecked[Double]("(double)rand()") // Fixme, add srand(time)

			def randinit(dim0: Int) = {
				val res = NewArray[Double](dim0)
				for (i <- (0 until dim0): Rep[Range]) res(i) = randDouble()
				new Vector(res, dim0)
			}

			def zeros(dim0: Int) = {
				val res = NewArray[Double](dim0)
				for (i <- (0 until dim0): Rep[Range]) res(i) = 0.0
				new Vector(res, dim0)
			}

			def ones(dim0: Int) = {
				val res = NewArray[Double](dim0)
				for (i <- (0 until dim0): Rep[Range]) res(i) = 1.0
				new Vector(res, dim0)
			}
		}


		// Tensor type is the same as NumR, just replace RDouble with Vector
		type diff = cps[Unit]

		class TensorR(val x: Vector, var d: Vector) {
			def + (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) => 
				val y = new TensorR(x + that.x, Vector.zeros(x.dim0)); k(y)
				this.d += y.d; that.d += y.d
			}
			def - (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
				val y = new TensorR(x - that.x, Vector.zeros(x.dim0)); k(y)
				this.d += y.d; that.d -= y.d
			}
			def * (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
				val y = new TensorR(x * that.x, Vector.zeros(x.dim0)); k(y)
				this.d += that.x * y.d; that.d += this.x * y.d
			}
			def dot(that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
				// assert both this and that are 1d vectors with the same size
				val y = new TensorR(x dot that.x, Vector.zeros(1)); k(y)
				this.d += that.x * y.d // broadcasting
				that.d += this.x * y.d // broadcasting
			}
		}

		def gradR(f: TensorR => TensorR @diff)(x: Vector): Vector = {
	    	val x1 = new TensorR(x, Vector.zeros(x.dim0))
	    	reset { f(x1).d = Vector.ones(1) } 
	    	x1.d
	    }

	}


	def main(args: Array[String]): Unit = {

		val array1 = new DslDriverC[String, Unit]  with VectorExp {

			def snippet(a: Rep[String]): Rep[Unit] = {
				
				// randomly generate an array of Double of size 5 in C code
				val res = Vector.randinit(5)
				val res2 = Vector.randinit(5)

				// val res3 = res map (t => 1.0) ERROR: map is not supported for C code generation
				val result = res dot res2
				result.print()
				//val value = result.data(0)
				//printf("the result is %f", value)
			}

		}

		println(array1.code)
		array1.eval("abc")

		val array2 = new DslDriverC[String, Unit] with VectorExp {

			def snippet(a: Rep[String]): Rep[Unit] = {
				// read training data from file
				val length = 2
				val vector = NewArray[Double](length)
				vector(0) = 2.0; vector(1) = 3.0			
				// wrap as Vector type
				val v = new Vector(vector, length)
				// calculate gradient
				val grad = gradR(t => t dot t)(v)
				// show gradient
				grad.print()
			}

		}

		println(array2.code)
		array2.eval("abc")

	}
}


/*
object LMS_Reverse {

  trait DiffApi extends Dsl {

    type diff = cps[Unit]

    class NumR(val x: Rep[Array[Double]], val d: Var[Array[Double]], val u: Int) extends Serializable {

    	def dot(that: NumR): NumR @diff = shift { (k: NumR => Unit) => 
    		val r = (x zip that.x) map (t => t._1 * t._2) foldLeft (_ + _)


	        val y = new NumR(x + that.x, var_new(0.0)); k(y)
	        this.d += y.d; that.d += y.d 
	    }


    }






  }
}
*/
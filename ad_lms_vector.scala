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

				However Greg just gave me a very good idea. Basically by using delimited continuations, our intermediate
				values are only used within the lexical scope of that operation, and the continuation it calls.
				Just like a withFile("filename") {} or withArray("array") {} construct,
				where the file and array are only used in the scope of with, can can be implicitly closed or deleted at escape
				Our intermediate Tensors are created by an overloaded operation, used in delimited continuations and updating
				the gradients of @this@ and @that@, and then never used outside
				So we can add unchecked("free") to reclaim their memory right at the end of each overloaded operator def.
	
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

		    def + (that: Vector) = {
		    	val res = NewArray[Double](dim0)
		    	if (that.dim0 == 1) for (i <- (0 until dim0): Rep[Range]) res(i) = data(i) + that.data(0)
		    	else if (dim0 == 1) for (i <- (0 until dim0): Rep[Range]) res(i) = data(0) + that.data(i)
		    	else if (dim0 == that.dim0) for (i <- (0 until dim0): Rep[Range]) res(i) = data(i) + that.data(i)
		    	else throw new IllegalArgumentException("dimensions of vector do not match!")
		    	new Vector(res, dim0)
		    }

		    def += (that: Vector) = {
		    	if (dim0 == that.dim0) for (i <- (0 until dim0): Rep[Range]) data(i) += that.data(i)
		    	else if (that.dim0 == 1) for (i <- (0 until dim0): Rep[Range]) data(i) += that.data(0)
		    	else throw new IllegalArgumentException("dimensions of vector do not match!")
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

		    def setAsOne() = {
		    	for (i <- (0 until dim0): Rep[Range]) data(i) = 1.0
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

		    def sum() = {
		    	val value = var_new(0.0)
		    	for (i <- (0 until data.length)) {
		    		value += data(i)
		    	}
		    	val res = NewArray[Double](1)
		    	res(0) = readVar(value)
		    	new Vector(res, 1)
		    }
/*
		    def squaredDistance()
*/


		    def print() = {
		    	for (i <- (0 until dim0): Rep[Range]) println(data(i))
		    }

		}

		object Vector {

			def randDouble() = unchecked[Double]("(double)rand()") // Fixme, add srand(time)

			def randinit(dim0: Int) = {
				val res = NewArray[Double](dim0)
				unchecked[Unit]("srand(time(NULL))")
				for (i <- (0 until dim0): Rep[Range]) res(i) = unchecked[Double]("(double)rand()/RAND_MAX*2.0-1.0")
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
				// delete y to prevent memory leak
				y.free()
			}
			def - (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
				val y = new TensorR(x - that.x, Vector.zeros(x.dim0)); k(y)
				this.d += y.d; that.d -= y.d
				// delete y to prevent memory leak
				y.free()
			}
			def * (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
				val y = new TensorR(x * that.x, Vector.zeros(x.dim0)); k(y)
				this.d += that.x * y.d 
				that.d += this.x * y.d // FIXme: intermediate Tensors donot need to be substatiated
				// delete y to prevent memory leak
				y.free()
			}
			def dot(that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
				// assert both this and that are 1d vectors with the same size
				val y = new TensorR(x dot that.x, Vector.zeros(1)); k(y)
				this.d += that.x * y.d // broadcasting
				that.d += this.x * y.d // broadcasting // Fixme: intermediate Tensors donot need to be substatiated
				// delete y to prevent memory leak
				y.free()
			}

			def sum(): TensorR @diff = shift { (k: TensorR => Unit) =>
				val y = new TensorR(x.sum(), Vector.zeros(1)); k(y)
				this.d += y.d
				y.free()
			}

			def free() = {
				unchecked[Unit]("free(",x.data,")")
				unchecked[Unit]("free(",d.data,")")
			}
		}

/*
		def FUN(f: TensorR => Unit): (TensorR => Unit) = {
			/*
	    	val f1 = fun { (x:Rep[Array[Double]]) =>
	    		val dim0 = x.length 
	    		val deltaVar: Vector = Vector.zeros(dim0)
	    		f(new TensorR(new Vector(x, dim0), deltaVar))
	    		deltaVar.data
	    		// Fixme: when to free the memories of x and deltaVar
	    	};*/
	    	{ 
	    		(x:TensorR) => {
	    			val dim0: Int = x.d.dim0
	    			val f2 = fun { (x: Rep[Array[Double]]) =>
	    				val delta: Vector = Vector.zeros(dim0)
	    				f(new TensorR(new Vector(x, dim0), delta))
	    				delta.data
	    			}
	    			x.d += new Vector(f2(x.x.data), dim0) 
	    		}
	    	}
	    }
*/

	    def FUN(f: TensorR => Unit): (TensorR => Unit) = {
	    	val dim0: Int = 1 // FIXME: what is the best way to carry this known dimensional information?
	    	val f1 = fun { (x: Rep[Array[Double]]) => 
	    		val deltaVar: Vector = Vector.zeros(dim0)
	    		f(new TensorR(new Vector(x, dim0), deltaVar))
	    		deltaVar.data
	    	};
	    	{ (x:TensorR) => x.d += new Vector(f1(x.x.data), dim0) }
	    }

	    def RST(a: => Unit @diff) = continuations.reset { a; () }

    	@virtualize
	    def IF(c: Rep[Boolean])(a: =>TensorR @diff)(b: =>TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>
	      val k1 = FUN(k)

	      if (c) RST(k1(a)) else RST(k1(b))
	    }

		def gradR(f: TensorR => TensorR @diff)(x: Vector): Vector = {
	    	val x1 = new TensorR(x, Vector.zeros(x.dim0))
	    	reset { f(x1).d.setAsOne(); () } 
	    	x1.d
	    }

	    

	}


	def main(args: Array[String]): Unit = {

		val array1 = new DslDriverC[String, Unit]  with VectorExp {

			def snippet(a: Rep[String]): Rep[Unit] = {
				println(a)
				// val temp = string_split(a, " ")
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

		//println(array1.code)
		//array1.eval("abc")

		val array2 = new DslDriverC[String, Unit] with VectorExp {

			def snippet(a: Rep[String]): Rep[Unit] = {
				// read training data from file (for now just use random)
				
				//val input = string_split(a, ",")
				// println(input.getClass.getName)
				///println(input(0)(0))
				//implicit val pos = implicitly[SourceContext]
				//printf("%s\n", input(0)(pos)(0))
				// vector(0) = string_todouble(input(0))
				// vector(1) = string_todouble(input(1))
				
				val length = 2
				val v = Vector.randinit(length)
				v.print()				

				// calculate gradient
				val grad = gradR(t => t dot t)(v)
				// show gradient
				grad.print()
			}

		}

		//println(array2.code)
		//array2.eval("2.0")

		val array3 = new DslDriverC[String, Unit] with VectorExp {

			def snippet(a: Rep[String]): Rep[Unit] = {
				// use random array as input
				val length = 2
				val v = Vector.randinit(length)
				v.print()

				// calcuate gradient
				val grad = gradR(t => IF (t.x.data(0) > 0.0) {t dot t}{t.sum()})(v)
				// show gradient
				grad.print()
			}
		}
		println(array3.code)
		array3.eval("abc")

	}
}

/*
object TEST2 {

	trait VectorExp extends Dsl {



	}

}
*/
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

			Note:

				There is a bug using the above method. Using free at the end of scope with LOOP together had an issue of
				use-after-free. Not sure why.

				Tiark suggested to use smart pointer in c++, which is a good idea because it takes away the burden of manually managing them.
				All we should have changed is the c code generation for NewArray[Double], so that the malloc is in the hand of smart pointers

			Note:

				We are currently only very narrowly supporting matrix (2d vectors)
				We only support Matrix vector multiplication, which is like several vector_vector dot product
				Matrix has >1 dim1 field and number of values dim0 * dim1
				but the current implementation silently ignore the 2:end columns unless it is dot product
				The idea of thinking Matrix row as dim0 and colume as dim1 is not the common way, but we are going by it for now because
				we want to simplify the implementation and just borrow the logic of dot
	
		**/

		class Vector(val data: Rep[Array[Double]], val dim0: Int, val dim1:Int = 1 /*, val dim2: Int*/) extends Serializable {

			def foreach(f: Rep[Double] => Rep[Unit]): Rep[Unit] =
		    	for (i <- (0 until dim0):Rep[Range]) f(data(i))

		    @virtualize
			def sumIf(f: Rep[Double] => Rep[Boolean]) = { 
		    	val n = var_new(0.0); 
		    	foreach(x => if (f(x)) n += x); 
		    	readVar(n) 
		    }

		    def + (that: Vector) = {
		    	val dimM = if (dim0 > that.dim0) dim0 else that.dim0
		    	val res = NewArray[Double](dimM)
		    	if (that.dim0 == 1) for (i <- (0 until dimM): Rep[Range]) res(i) = data(i) + that.data(0)
		    	else if (dim0 == 1) for (i <- (0 until dimM): Rep[Range]) res(i) = data(0) + that.data(i)
		    	else if (dim0 == that.dim0) for (i <- (0 until dimM): Rep[Range]) res(i) = data(i) + that.data(i)
		    	else throw new IllegalArgumentException("dimensions of vector do not match +!")
		    	new Vector(res, dimM)
		    }

		    // this operator updates the values of this, unlike the + operator
		    def += (that: Vector) = {
		    	if (dim0 == that.dim0) for (i <- (0 until dim0): Rep[Range]) data(i) += that.data(i)
		    	else if (that.dim0 == 1) for (i <- (0 until dim0): Rep[Range]) data(i) += that.data(0)
		    	else if (dim0 == 1) throw new IllegalArgumentException("dimensions needs to be expanded!")
		    	else throw new IllegalArgumentException("dimensions of vector do not match +=!")
		    }

		    def - (that: Vector) = {
		    	val dimM = if (dim0 > that.dim0) dim0 else that.dim0
		    	val res = NewArray[Double](dimM)
		    	if (that.dim0 == 1) for (i <- (0 until dimM): Rep[Range]) res(i) = data(i) - that.data(0)
		    	else if (dim0 == 1) for (i <- (0 until dimM): Rep[Range]) res(i) = data(0) - that.data(i)
		    	else if (dim0 == that.dim0) for (i <- (0 until dimM): Rep[Range]) res(i) = data(i) - that.data(i)
		    	else throw new IllegalArgumentException("dimensions of vector do not match -!")
		    	new Vector(res, dimM)
		    }

		    // this operator updates the values of this, unlike the - operator
		    def -= (that: Vector) = {
		    	if (dim0 == that.dim0) for (i <- (0 until dim0): Rep[Range]) data(i) -= that.data(i)
		    	else if (that.dim0 == 1) for (i <- (0 until dim0): Rep[Range]) data(i) -= that.data(0)
		    	else if (dim0 == 1) throw new IllegalArgumentException("dimensions needs to be expanded!")
		    	else throw new IllegalArgumentException("dimensions of vector do not match -=!")
		    }

		    // element wise multiplication
		    def * (that: Vector) = {
		    	val dimM = if (dim0 > that.dim0) dim0 else that.dim0
		    	val res = NewArray[Double](dimM)
		    	if (that.dim0 == 1) for (i <- (0 until dimM): Rep[Range]) res(i) = data(i) * that.data(0)
		    	else if (dim0 == 1) for (i <- (0 until dimM): Rep[Range]) res(i) = data(0) * that.data(i)
		    	else if (dim0 == that.dim0) for (i <- (0 until dimM): Rep[Range]) res(i) = data(i) * that.data(i)
		    	else throw new IllegalArgumentException("dimensions of vector do not match *!")
		    	new Vector(res, dimM)
		    }

		    // this operator updates the values of this, unlike * operator
		    def *= (that: Vector) = {
		    	if (dim0 == that.dim0) for (i <- (0 until dim0): Rep[Range]) data(i) *= that.data(i)
		    	else if (that.dim0 == 1) for (i <- (0 until dim0): Rep[Range]) data(i) *= that.data(0)
		    	else if (dim0 == 1) throw new IllegalArgumentException("dimensions needs to be expanded *=!")
		    	else throw new IllegalArgumentException("dimensions of vector do not match *=!")
		    }

		    // element wise division
		    def / (that: Vector) = {
		    	val dimM = if (dim0 > that.dim0) dim0 else that.dim0
		    	val res = NewArray[Double](dimM)
		    	if (that.dim0 == 1) for (i <- (0 until dimM): Rep[Range]) res(i) = data(i) / that.data(0)
		    	else if (dim0 == 1) for (i <- (0 until dimM): Rep[Range]) res(i) = data(0) / that.data(i)
		    	else if (dim0 == that.dim0) for (i <- (0 until dimM): Rep[Range]) res(i) = data(i) / that.data(i)
		    	else throw new IllegalArgumentException("dimensions of vector do not match /!")
		    	new Vector(res, dimM)
		    }

		    // this operator updates the values of this, unlike / operator
		    def /= (that: Vector) = {
		    	if (dim0 == that.dim0) for (i <- (0 until dim0): Rep[Range]) data(i) /= that.data(i)
		    	else if (that.dim0 == 1) for (i <- (0 until dim0): Rep[Range]) data(i) /= that.data(0)
		    	else if (dim0 == 1) throw new IllegalArgumentException("dimensions needs to be expanded /=!")
		    	else throw new IllegalArgumentException("dimensions of vector do not match /=!")
		    }

		    def setAsOne() = {
		    	for (i <- (0 until dim0): Rep[Range]) data(i) = 1.0
		    }

		    def dot(that: Vector) = {
		    	// assert that and this have the same dimension
		    	if (dim0 != that.dim0) throw new IllegalArgumentException("dimensions of vector do not match dot!")
		    	val res = NewArray[Double](dim1)
		    	for (j <- (0 until dim1): Rep[Range]) {
		    		val value = var_new(0.0)
		    		for (i <- (0 until dim0): Rep[Range]) value += data(i + dim0 * j) * that.data(i)
		    		res(j) = readVar(value)
		    	}
		    	new Vector(res, dim1)
		    }

		    def tanh() = {
		    	val res = NewArray[Double](dim0)
		    	for (i <- (0 until dim0): Rep[Range]) {
		    		res(i) = Math.tanh(data(i)) // need fix, MathOps C code gene is not supporting tanh
		    	}
		    	new Vector(res, dim0)
		    }

		    def exp() = {
		    	val res = NewArray[Double](dim0)
		    	for (i <- (0 until dim0): Rep[Range]) {
		    		res(i) = Math.exp(data(i))
		    	}
		    	new Vector(res, dim0)
		    }

		    def log() = {
		    	val res = NewArray[Double](dim0)
		    	for (i <- (0 until dim0): Rep[Range]) {
		    		res(i) = Math.log(data(i))
		    	}
		    	new Vector(res, dim0)
		    }

		    def sum() = {
		    	val value = var_new(0.0)
		    	for (i <- (0 until dim0): Rep[Range]) {
		    		value += data(i)
		    	}
		    	val res = NewArray[Double](1)
		    	res(0) = readVar(value)
		    	new Vector(res, 1)
		    }

		    def print() = {

		    	for (j <- (0 until dim1): Rep[Range]) {
		    		for (i <- (0 until dim0): Rep[Range]) {
		    			println(data(i + j * dim0))
		    		}
		    		println(" ")
		    	}
		    }

		    // setting: this is matrix, that is dim0-sized vector, y is dim1-sized vector
		    // the result is to update this so that this += that * y, where * is cartesian product
		    def add_cartesian(that: Vector, y: Vector) = {
		    	for (i <- (0 until dim1): Rep[Range]) {
					for (j <- (0 until dim0): Rep[Range]) {
						val ind = dim0 * i + j
						data(ind) += that.data(j) * y.data(i)
					}
				}
		    } 
			
			// setting: this is dim0-sized vector, that is matrix (dim0 * dim1), y is dim1-sized vector
			// the result is to update this so that this accumulate every matrix col * y
			def add_composion(that: Vector, y: Vector) = {	
				for (i <- (0 until that.dim1): Rep[Range]) {
					for (j <- (0 until dim0): Rep[Range]) {
						data(j) += that.data(dim0 * i + j) * y.data(i)
					}
				}
			}

		}

		object Vector {

			def randinit(dim0: Int, dim1: Int = 1) = {
				unchecked[Unit]("srand(time(NULL))")
				val res = NewArray[Double](dim0 * dim1)
				for (i <- (0 until dim0 * dim1): Rep[Range]) res(i) = unchecked[Double]("(double)rand()/RAND_MAX*2.0-1.0")
				new Vector(res, dim0, dim1)
			}

			def randPositive(dim0: Int) = {
				val res = NewArray[Double](dim0)
				unchecked[Unit]("srand(time(NULL))")
				for (i <- (0 until dim0): Rep[Range]) res(i) = unchecked[Double]("(double)rand()/RAND_MAX*2.0")
				new Vector(res, dim0)
			}

			def zeros(dim0: Int, dim1: Int = 1) = {
				val res = NewArray[Double](dim0 * dim1)
				for (i <- (0 until dim0 * dim1): Rep[Range]) res(i) = 0.0
				new Vector(res, dim0, dim1)
			}

			def ones(dim0: Int) = {
				val res = NewArray[Double](dim0)
				for (i <- (0 until dim0): Rep[Range]) res(i) = 1.0
				new Vector(res, dim0)
			}

			def halves(dim0: Int) = {
				val res = NewArray[Double](dim0)
				for (i <- (0 until dim0): Rep[Range]) res(i) = 0.5
				new Vector(res, dim0)
			}

			def consts(dim0: Int, value: Double) = {
				val res = NewArray[Double](dim0)
				for (i <- (0 until dim0): Rep[Range]) res(i) = value
				new Vector(res, dim0)
			}
		}


		// Tensor type is the similar to NumR, just replace RDouble with Vector
		// also Vector internally use array, which is mutable by default
		// so both field are val (not var) and can be updated by += -= *= /= setAsOne() 
		// all instances of vectors will be shepherded by c++ smart pointers, alleviating the memory leak problem
		type diff = cps[Unit]

		class TensorR(val x: Vector, val d: Vector) extends Serializable {

			def + (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) => 
				val y = new TensorR(x + that.x, Vector.zeros(x.dim0)); k(y)
				this.d += y.d; that.d += y.d
			}

			def - (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
				val y = new TensorR(x - that.x, Vector.zeros(x.dim0)); k(y)
				this.d += y.d; that.d -= y.d
			}

			// this is element wise multiplication
			def * (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
				val y = new TensorR(x * that.x, Vector.zeros(x.dim0)); k(y)
				// FIXME: intermediate Tensors donot need to be substatiated, can optimize!
				this.d += that.x * y.d; 
				that.d += this.x * y.d;	
			}

			// element wise division
			def / (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
				val y = new TensorR(x / that.x, Vector.zeros(x.dim0)); k(y)
				// FIXME: intermediate Tensors donot need to be substatiated, can optimize!
				this.d += y.d / that.x
				that.d -= this.x * y.d / (that.x * that.x) 
			}

			// vector dot product or Matrix vector dot (viewed as multiple vector dot product) (not the common view)
			def dot(that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
				val y = new TensorR(x dot that.x, Vector.zeros(x.dim1)); k(y)
				// FIXME: intermediate Tensors donot need to be substatiated, can optimize!
				this.d.add_cartesian(that.x, y.d) 
				that.d.add_composion(this.x, y.d)
				// this.d += that.x * y.d // broadcasting
				// that.d += this.x * y.d // broadcasting 
			}

			def tanh(): TensorR @diff = shift { (k : TensorR => Unit) =>
				val y = new TensorR(x.tanh(), Vector.zeros(x.dim0)); k(y)
				// FIXME: intermediate Tensors donot need to be substatiated, can optimize!
				this.d += (Vector.ones(x.dim0) - y.x * y.x) * y.d // broadcasting
			}

			def exp(): TensorR @diff = shift { (k: TensorR => Unit) =>
				val y = new TensorR(x.exp(), Vector.zeros(x.dim0)); k(y)
				// Fix
				this.d += y.x * y.d
			}

			def log(): TensorR @diff = shift { (k: TensorR => Unit) =>
				val y = new TensorR(x.log(), Vector.zeros(x.dim0)); k(y)
				// Fix
				this.d += y.d / x
			}

			def sum(): TensorR @diff = shift { (k: TensorR => Unit) =>
				val y = new TensorR(x.sum(), Vector.zeros(1)); k(y)
				this.d += y.d
			}
			/*
			def free() = {
				unchecked[Unit]("free(",x.data,")")
				unchecked[Unit]("free(",d.data,")")
			} */
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

	    def FUN(dim0: Int)(f: TensorR => Unit): (TensorR => Unit) = {
	    	// val dim0: Int = 1 // FIXME: what is the best way to carry this known dimensional information?
	    	val f1 = fun { (x: Rep[Array[Double]]) => 
	    		val deltaVar: Vector = Vector.zeros(dim0)
	    		f(new TensorR(new Vector(x, dim0), deltaVar))
	    		deltaVar.data
	    	};
	    	{ (x:TensorR) => x.d += new Vector(f1(x.x.data), dim0) }
	    }

	    def RST(a: => Unit @diff) = continuations.reset { a; () }

    	@virtualize
	    def IF(dim0: Int)(c: Rep[Boolean])(a: =>TensorR @diff)(b: =>TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>
	      val k1 = FUN(dim0)(k)

	      if (c) RST(k1(a)) else RST(k1(b))
	    }

	    @virtualize
	    def LOOP(init: TensorR)(c: TensorR => Rep[Boolean])(b: TensorR => TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>
	      // val k1 = FUN(init.x.dim0)(k)

	      lazy val loop: TensorR => Unit = FUN (init.x.dim0) { (x: TensorR) =>
	        if (c(x)) RST(loop(b(x))) else RST(k(x))
	      }
	      loop(init)
	    }

		def gradR(f: TensorR => TensorR @diff)(x: Vector): Vector = {
	    	val x1 = new TensorR(x, Vector.zeros(x.dim0))
	    	reset { val y = f(x1)
	    			y.d.setAsOne()
	    			y.x.print()
	    			() } 
	    	x1.d
	    }
/*
	    def grad_side_effect_using_closure(f: () => TensorR @diff): Rep[Unit] = {
	    	reset {
	    		val y = f()
	    		y.d.setAsOne()
	    		y.x.print()
	    		()
	    	}
	    } */
	}


	def main(args: Array[String]): Unit = {

		val array1 = new DslDriverC[String, Unit]  with VectorExp {

			def snippet(a: Rep[String]): Rep[Unit] = {
				val length = 2
				val res = Vector.randinit(length)
				val res2 = Vector.randPositive(length)
				res.print()
				res2.print()
				
				val result = res dot res2
				result.print()
			}
		}

		//println("test dot")
		//println(array1.code)
		//array1.eval("abc")

		val array1_1 = new DslDriverC[String, Unit] with VectorExp {

			def snippet(a: Rep[String]): Rep[Unit] = {
				val dim0 = 2
				val dim1 = 3
				val matrix = Vector.randinit(dim0, dim1)
				val vector = Vector.randPositive(dim0)
				matrix.print()
				vector.print()

				println("the result is:")
				val result = matrix dot vector
				result.print()
			}
		}

		//println(array1_1.code)
		//array1_1.eval("abc")

		val array2 = new DslDriverC[String, Unit] with VectorExp {

			def snippet(a: Rep[String]): Rep[Unit] = {
				// read training data from file (for now just use random)
				val length = 2
				val v = Vector.randinit(length)
				v.print()	

				// calculate gradient
				val grad = gradR(t => t dot t)(v)
				// show gradient
				grad.print()
			}
		}

		//println("test dot gradient")
		//println(array2.code)
		//array2.eval("2.0")

		val array2_1 = new DslDriverC[String, Unit] with VectorExp {
			// update gradient as side effect
			
			def snippet(a: Rep[String]): Rep[Unit] = {
				val length = 2
				val v = Vector.randinit(length)
				v.print()

				// initialize tensor for closure
				val t = new TensorR(v, Vector.zeros(length))			
				// call grad_side_effect_using_closure
				val dummy = gradR(dummy => t dot t)(Vector.zeros(1))
				// print the gradient of t
				t.x.print()
				t.d.print()
			}
		}

		//println("test dot gradient as side effect")
		//println(array2_1.code)
		//array2_1.eval("2.0")

		val array2_2 = new DslDriverC[String, Unit] with VectorExp {

			def snippet(a: Rep[String]): Rep[Unit] = {

				val dim0 = 2
				val dim1 = 3
				val matrix = Vector.randinit(dim0, dim1)
				val vector = Vector.randPositive(dim0)
				matrix.print()
				vector.print()

				// initialize tensors for closure
				val ma = new TensorR(matrix, Vector.zeros(dim0, dim1))
				val ve = new TensorR(vector, Vector.zeros(dim0))
				// define function of model
				def model(dummy: TensorR): TensorR @diff = {
					(ma dot ve).sum()
				}
				val dummy = gradR(model)(Vector.zeros(1))
				// print the gradient of ma and ve
				ma.d.print()
				ve.d.print()
			}
		}

		// println("test matrix vector dot gradient as side effect")
		println(array2_2.code)
		array2_2.eval("abc")

		val array3 = new DslDriverC[String, Unit] with VectorExp {

			def snippet(a: Rep[String]): Rep[Unit] = {
				// use random array as input
				val length = 2
				val v = Vector.randinit(length)
				v.print()

				// calcuate gradient
				val grad = gradR(t => {val y = IF (length)(t.x.data(0) > 0.0) {t + t}{t * t}
									   y.sum() })(v)
				// show gradient
				grad.print()
			}
		}

		//println("test IF gradient")
		//println(array3.code)
		//array3.eval("abc")

		val array4 = new DslDriverC[String, Unit] with VectorExp {

	    	def snippet(a: Rep[String]): Rep[Unit] = {
	    		// use random array as input
	    		val length = 2
	    		val v = Vector.randinit(length)
	    		v.print()

	    		val half = (new TensorR(Vector.halves(length), Vector.zeros(length)))
	    		// calculate gradient
	    		val grad = gradR(t => {val y = LOOP(t)(t => t.x.data(0) > 0.1)(t => t * half)
	    							   y.sum() })(v)
	    		// show gradient
	    		grad.print()
	    	}
	    }

		// println("test LOOP gradient")
		//import java.io.PrintWriter;
		//import java.io.File;	
	    //println(array4.code)
	    //val p = new PrintWriter(new File("fei_needs_help_for_basic_java_thing.cpp"))
		//p.println(array4.code)
		//p.flush()
	    // array4.eval("abc" )

	    val array5 = new DslDriverC[String, Unit] with VectorExp {

	    	def snippet(a: Rep[String]): Rep[Unit] = {
	    		val length = 2
	    		val v = Vector.randinit(length)
	    		v.print()

	    		val grad = gradR(t => (t * t).sum())(v)
	    		grad.print()
	    	}
	    }

	    //println("test elementwise multiplication")
	    //println(array5.code)
	    //array5.eval("abc")

	    val array6 = new DslDriverC[String, Unit] with VectorExp {

	    	def snippet(a: Rep[String]): Rep[Unit] = {
	    		val length = 2
	    		val v = Vector.randinit(length)
	    		v.print()

	    		val grad = gradR(t => (t / t).sum())(v)
	    		grad.print()
	    	}
	    }

	    // println("test elementwise division")
	    //println(array6.code)
	    //array6.eval("abc")

	    val array7 = new DslDriverC[String, Unit] with VectorExp {

	    	def snippet(a: Rep[String]): Rep[Unit] = {
	    		val length = 2
	    		val v = Vector.randinit(length)
	    		v.print()

	    		val grad = gradR(t => (t.tanh()).sum())(v)
	    		grad.print()
	    	}
	    }

	    // println("test tanh")
	    //println(array7.code)
	    //array7.eval("abc")

	    val array8 = new DslDriverC[String, Unit] with VectorExp {

	    	def snippet(a: Rep[String]): Rep[Unit] = {
	    		val length = 2
	    		val v = Vector.randinit(length)
	    		v.print()

	    		val grad = gradR(t => (t.exp()).sum())(v)
	    		grad.print()
	    	}
	    }

	    // println("test exp")
	    //println(array8.code)
	    //array8.eval("abc")

	    val array9 = new DslDriverC[String, Unit] with VectorExp {

	    	def snippet(a: Rep[String]): Rep[Unit] = {
	    		val length = 2
	    		val v = Vector.randPositive(length)
	    		v.print()

	    		val grad = gradR(t => (t.log()).sum())(v)
	    		grad.print()
	    	}
	    }

	    //println("test log")
	    // println(array9.code)
	    //array9.eval("abc")

	}
}
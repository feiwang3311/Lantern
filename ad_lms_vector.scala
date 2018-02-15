import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

object TEST1 {

	trait VectorExp extends Dsl {

		class Vector(val data: Rep[Array[Double]], val dim0: Rep[Int]) {

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
		    	val res = var_new(0.0)
		    	for (i <- 0 until data.length) {
		    		res += data(i) * that.data(i)
		    	}
		    	readVar(res)
		    }
		}

		object Vector {

			def randDouble() = unchecked[Double]("(double)rand()") // Fixme

			def randinit(dim0: Int) = {
				val res = NewArray[Double](dim0)
				for (i <- (0 until dim0): Rep[Range]) {
					res(i) = randDouble()
				}
				new Vector(res, dim0)
			}
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
				printf("the result is %f", result)
			}

		}

		println(array1.code)
		array1.eval("abc")

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
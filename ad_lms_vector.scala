import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

object TEST1 {

	def main(args: Array[String]): Unit = {

		val array1 = new DslDriverC[String, Unit] {

			def randDouble() = unchecked[Double]("(double)rand()")

			def snippet(a: Rep[String]): Rep[Unit] = {
				
				// randomly generate an array of Double of size 5 in C code
				val res = NewArray[Double](5)
				val res2 = NewArray[Double](5)
				for (i <- (0 until 5): Rep[Range]) {
					res(i) = randDouble()
					res2(i) = randDouble()
				}

				// val res3 = res map (t => 1.0) ERROR: map is not supported for C code generation
				val res3 = var_new(0.0)
				for (i <- (0 until 5): Rep[Range]) {
					res3 += res(i) * res2(i)
				}
				val result = readVar(res3)
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
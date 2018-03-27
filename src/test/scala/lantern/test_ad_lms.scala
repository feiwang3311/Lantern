package lantern

import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

import org.scalatest.FunSuite

import java.io.PrintWriter;
import java.io.File;  

class AdLMSTest extends FunSuite {
  
  val gr1 = new DslDriver[Double,Double] with DiffApi {
    def snippet(x: Rep[Double]): Rep[Double] = {
      gradR(x => x + x*x*x)(x)
    }
  }

  val grv1 = new DslDriver[Double,Double] with DiffApi {
    def snippet(x: Rep[Double]): Rep[Double] = {
      gradRV(x => x + x*x*x)(x)
    }
  }

  val gf1 = new DslDriver[Double,Double] with DiffApi {
    def snippet(x: Rep[Double]): Rep[Double] = {
      gradF(x => x + x*x*x)(x)
    }
  }

  val gff1 = new DslDriver[Double,Double] with DiffApi {
    def snippet(x: Rep[Double]): Rep[Double] = {
      gradFF(x => x + x*x*x)(x)
    }
  }


  //println("demonstrate the problem of purturbation confusion")
  val grr = new DslDriver[Double, Double] with DiffApi {
    def snippet(x: Rep[Double]): Rep[Double] = {
      gradR{ (x: NumR) =>
        val temp = new NumR(gradR(y => x + y)(1), var_new(0.0))
        x * temp
        } (x)
    }
  }
  println(grr.eval(1))

  // test 2 -- conditional
  val gr2 = new DslDriver[Double,Double] with DiffApi {
    def snippet(x: Rep[Double]): Rep[Double] = {
      val minus_1 = (new NumR(-1.0,var_new(0.0)))
      gradR(x => IF (x.x > 0.0) { minus_1*x*x } { x*x })(x)
    }
  }

  // test 3 -- loop using fold
  def fr(x: Double): Double = {
    // Divide by 2.0 until less than 1.0
    if (x > 1.0) fr(0.5 * x) else x 
  }
  // Hand-coded correct derivative
  def gfr(x: Double): Double = {
    if (x > 1.0) 0.5 * gfr(0.5 * x) else 1.0
  }


  val gr3 = new DslDriver[Double,Double] with DiffApi {
    def snippet(x: Rep[Double]): Rep[Double] = {
      val half = (new NumR(0.5,var_new(0.0)))
      val res = gradR(x => LOOP(x)(x1 => x1.x > 1.0)(x1 => half * x1))(x)
      println(readVar(half.d))
      res
    }
  }
  
  val gr4 = new DslDriver[Double, Double] with DiffApi {
    def snippet(x: Rep[Double]): Rep[Double] = {
      val half = new NumR(0.5, var_new(0.0))
      val res = gradR(x => LOOPC(x)(3)(x1 =>{half * x1}))(x)
      println(readVar(half.d))
      res
    }
  }

  //println(gr4.code)
  //println(gr4.eval(10))

  val gr7 = new DslDriver[Double, Double] with DiffApi {

    def snippet(x: Rep[Double]): Rep[Double] = {
      val half = new NumR(0.5, var_new(0.0))
      val res = gradR(x => LOOPCC(x)(3)(i => x1 => {
        println(i)
        half * x1 }))(x)
      println(readVar(half.d))
      res
    }
  }

  //println(gr7.code)
  //println(gr7.eval(10))

  val gr8 = new DslDriver[Double, Double] with DiffApi {

    def snippet(x: Rep[Double]): Rep[Double] = {
      val array = NewArray[Double](3)
      for (i <- (0 until 3): Rep[Range]) array(i) = i + 2
      val model: NumR => NumR @diff = { (x: NumR) =>
        LOOPA(x)(array)(i => x1 => {
          val t = new NumR(array(i), var_new(0.0))
          t * x1
        })
      }
      val res = gradR(model)(x)
      res
    }
  }

  //println(gr8.code)
  //println(gr8.eval(10))

  val gr9 = new DslDriver[Double, Double] with DiffApi {

    def snippet(x: Rep[Double]): Rep[Double] = {
      // preprocess data (wrap Array as RepArray)
      val arr = scala.Array(1.5,3.0,4.0)
      val arra = staticData(arr)
      
      val model: NumR => NumR @diff = { (x: NumR) =>
        LOOPA(x)(arra)(i => x1 => {
          val t = new NumR(arra(i), var_new(0.0))
          t * x1
          })
      }
      val res = gradR(model)(x)
      res
    }
  }

  //println(gr9.code)
  //println(gr9.eval(10))

  val gr10 = new DslDriver[Double, Double] with DiffApi {

    def snippet(x: Rep[Double]): Rep[Double] = {
      // preprocess data
      val arr = scala.Array(1.5, 2.0, 3.0)
      val arra = staticData(arr)
      val length = arr.length

      // maybe we can use loopcc, just use arra by closure
      val model: NumR => NumR @diff = { (x: NumR) =>
        LOOPCC(x)(length)(i => x1 => {
          val t = new NumR(arra(i), var_new(0.0))
          t * x1
          })
      }
      val res = gradR(model)(x)
      res
      /*
        Note: It is interesting to note that the recursive function body can make use of closured array data
              but recursive guard (the if condition of LOOP) cannot, because the def of LOOP is before the presence of data
              So the length has to be passed in as a parameter explicitly
        Is there a better way to do it??
      */
    }
  }

  //println(gr10.code)
  //println(gr10.eval(10))

  val gr11 = new DslDriver[Double, Double] with DiffApi {

    def snippet(x: Rep[Double]): Rep[Double] = {
      // represent list as array
      val arr = scala.Array(4.0, 3.0, 1.5, 2.0)
      //val arr = scala.Array[Double]()
      val arra = mutableStaticData(arr)

      // create a model that recursively use the data in arr (originated from list)
      def model: NumR => NumR @diff = { (x: NumR) =>
        LOOPL5(x)(arra.length)(i => x1 => new NumR(arra(i), var_new(0.0)) * x1)
      }
      val res = gradR(model)(x)
      res
    }
  }
  
  //println(gr11.code)
  /*val p = new PrintWriter(new File("gr11.scala"))
  p.println(gr11.code)
  p.flush()*/
  //println(gr11.eval(2))

  val gr12 = new DslDriver[Double, Double] with DiffApi {

    def snippet(x: Rep[Double]): Rep[Double] = {
      // represent tree as arrays
      /*    5
           / \
          3   4
      */
      val data = scala.Array[Double](5.0, 3.0, 4.0)
      val lch  = scala.Array[Int](1, 100, 100) // use very large number to mean non nodes
      val rch  = scala.Array[Int](2, 100, 100)
      val data1 = mutableStaticData(data)
      val lch1  = mutableStaticData(lch)
      val rch1  = mutableStaticData(rch)

      // create a model that recursively use the data (originated from tree)
      def model: NumR => NumR @diff = { (x: NumR) =>
        TREE1(x)(data1.length, lch1, rch1){ (l: NumR, r: NumR, i: Rep[Int]) =>
          l * r * new NumR(data1(i), var_new(0.0))
        }
      }

      val res = gradR(model)(x)
      res
    }
  }

  //println(gr12.code)
  //println(gr12.eval(2))

  val gr12_2 = new DslDriver[Double, Double] with DiffApi {

    def snippet(x: Rep[Double]): Rep[Double] = {
      val A = scala.Array
      val data = A[Double](5.0, 3.0, 4.0)
      val lch  = A[Int](1, 100, 100)
      val rch  = A[Int](2, 100, 100)
      val data1 = mutableStaticData(data)
      val lch1  = mutableStaticData(lch)
      val rch1  = mutableStaticData(rch)

      val para = new NumR(2.0, var_new(0.0))
      def model: NumR => NumR @diff = { (x: NumR) =>
        TREE1(x)(data1.length, lch1, rch1){ (l: NumR, r: NumR, i: Rep[Int]) =>
          l * r * new NumR(data1(i), var_new(0.0)) * para
        }
      }

      val res = gradR(model)(x)
      printf("the grad of para is %f\n", readVar(para.d))
      res
    }
  }

  //println(gr12_2.code)
  //printf("grad of x is %f\n", gr12_2.eval(1))

  val gr12_3 = new DslDriver[Double, Double] with DiffApi {

    @virtualize
    def snippet(x: Rep[Double]): Rep[Double] = {
      val A = scala.Array
      val data = A[Double](0.0, 3.0, 4.0)
      val lch  = A[Int](1, 100, 100)
      val rch  = A[Int](2, 100, 100)
      val data1 = mutableStaticData(data)
      val lch1  = mutableStaticData(lch)
      val rch1  = mutableStaticData(rch)

      val para  = new NumR(2.0, var_new(0.0))
      val paral = new NumR(1.5, var_new(0.0))
      val parar = new NumR(1.6, var_new(0.0))
      val bias  = new NumR(2.5, var_new(0.0))
      def model: NumR => NumR @diff = { (dummy: NumR) =>
        TREE1(dummy)(data1.length, lch1, rch1){ (l: NumR, r: NumR, i: Rep[Int]) =>
          IF (lch1(i) > data1.length) {para * new NumR(data1(i), var_new(0.0)) + bias}
          {paral * l + parar * r + bias}
        }
      }

      val res = gradR(model)(x)
      printf("the grad of para is %f\n",  readVar(para.d))
      printf("the grad of paral is %f\n", readVar(paral.d))
      printf("the grad of parar is %f\n", readVar(parar.d))
      printf("the grad of bias is %f\n",  readVar(bias.d))
      0.0
    }
  }
  /*
  val p1 = new PrintWriter(new File("temp/gr12_3.scala"))
  p1.println(gr12_3.code)
  p1.flush()*/
  gr12_3.eval(0.0)
  //printf("grad of x is %f\n", gr12_3.eval(1))

  val gr12_4 = new DslDriver[Double, Double] with DiffApi {

    @virtualize
    def snippet(x: Rep[Double]): Rep[Double] = {
      val A = scala.Array
      val data = A[Double](0.0, 3.0, 4.0)
      val lch  = A[Int](1, -1, -1)
      val rch  = A[Int](2, -1, -1)
      val data1 = mutableStaticData(data)
      val lch1  = mutableStaticData(lch)
      val rch1  = mutableStaticData(rch)

      val para  = new NumR(2.0, var_new(0.0))
      val paral = new NumR(1.5, var_new(0.0))
      val parar = new NumR(1.6, var_new(0.0))
      val bias  = new NumR(2.5, var_new(0.0))
      def model: NumR => NumR @diff = { (dummy: NumR) =>
        TREE2(dummy)(lch1, rch1){ (l: NumR, r: NumR, i: Rep[Int]) =>
          IF (lch1(i) < 0) {para * new NumR(data1(i), var_new(0.0)) + bias}
          {paral * l + parar * r + bias}
        }
      }

      val res = gradR(model)(x)
      printf("the grad of para is %f\n",  readVar(para.d))
      printf("the grad of paral is %f\n", readVar(paral.d))
      printf("the grad of parar is %f\n", readVar(parar.d))
      printf("the grad of bias is %f\n",  readVar(bias.d))
      0.0
    }
  }

  /*
  val p2 = new PrintWriter(new File("temp/gr12_4.scala"))
  p2.println(gr12_4.code)
  p2.flush()
  gr12_4.eval(0.0)()
  //printf("grad of x is %f\n", gr12_4.eval(1))
  */

}
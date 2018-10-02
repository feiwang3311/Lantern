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

  test("function_composition") {
    val g1 = new DslDriver[Double, Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        def f1(x: NumR) = x * x * x
        def f2(x: NumR) = x * x
        def f3(x: NumR) = f2(f1(x))
        gradR(f3)(x)
      }
    }
    def grad(x: Double) = 6 * x * x * x * x * x
    for (x <- (-5 until 5)) {
      assert (g1.eval(x) == grad(x))
    }
  }

  test("simple") {
    val g1 = new DslDriver[Double, Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        def f(x: NumR) = x * x
        gradR(f)(x)
      }
    }
    System.out.println(g1.code)
  }

  test("reccc") {
    val g1 = new DslDriver[Double, Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        def f(x: NumR) = shift { (k: NumR => Unit) =>
          lazy val rec = FUNL { (k: NumR => Unit) => (x: NumR) =>
            RST{k( x * x )}
          }
          rec(k)(x)
        }
        gradR(f)(x)
      }
    }
    System.out.println(g1.code)
  }

  test("rec") {
    val g1 = new DslDriver[Double, Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        def f(x: NumR) = Rec(x)(y => y * y)
        gradR(f)(x)
      }
    }
    System.out.println(g1.code)
  }
  /*
  class Snippet extends ((Double)=>(Double)) {
    def apply(x0:Double): Double = {
      val x2 = {x3: (scala.Function1[Double,Double]) =>
        val x4 = {x5: (Double) =>
          var x6: Double = 0.0
          val x7 = x5 * x5
          val x9 = x3(x7)
          val x12 = x5 * x9
          x6 += x12
          x6 += x12
          x6
        }
        x4: scala.Function1[Double,Double]
      }
      val x23 = x2({x18: (Double) => 1.0})
      x23(x0)
    }
  }
  */

  test("leaky") {
    val g1 = new DslDriver[Double, Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        def f1(x: NumR) = x * x * x
        val v1: NumR = getNumR(1.0)

        gradR(x => f1(x) * (v1 + v1))(x)
      }
    }
    def grad(x: Double) = 6 * x * x
    for (x <- (-5 until 5)) {
      assert (g1.eval(x) == grad(x))
    }
  }

  test("side_effect") {
    val g1 = new DslDriver[Double, Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        val v1: NumR = getNumR(1.0)
        val v2: NumR = getNumR(2.0)
        gradR(x => {
          val v3 = v1 + v2
          v3.print()
          x * v3
        })(x)
      }
    }
    def grad(x: Double) = 3
    for (x <- (-5 until 5)) {
      assert (g1.eval(x) == grad(x))
    }
  }

  test("reverse_mode") {
    val gr1 = new DslDriver[Double,Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        gradR(x => x + x*x*x)(x)
      }
    }
    def grad(x: Double) = 1 + 3 * x * x
    for (x <- (-5 until 5)) {
      assert (gr1.eval(x) == grad(x))
    }
  }

  test("reverseRV") {
    val grv1 = new DslDriver[Double,Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        gradRV(x => x + x*x*x)(x)
      }
    }
    def grad(x: Double) = 1 + 3 * x * x
    for (x <- (-5 until 5)) {
      assert (grv1.eval(x) == grad(x))
    }
  }

  test("forward_mode") {
    val gf1 = new DslDriver[Double,Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        gradF(x => x + x*x*x)(x)
      }
    }
    def grad(x: Double) = 1 + 3 * x * x
    for (x <- (-5 until 5)) {
      assert (gf1.eval(x) == grad(x))
    }
  }


  test("foward_forward") {
    val gff1 = new DslDriver[Double,Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        gradFF(x => x + x*x*x)(x)
      }
    }
    def gradff(x: Double) = 6 * x
    for (x <- (-5 until 5)) {
      assert (gff1.eval(x) == gradff(x))
    }
  }

  //println("demonstrate the problem of purturbation confusion")
  test("purturbation confusion") {
    val grr = new DslDriver[Double, Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        gradR{ (x: NumR) =>
          val temp = new NumR(gradR(y => x + y)(1), var_new(0.0))
          x * temp
          } (x)
      }
    }
    def grad_confusion = 2
    assert(grr.eval(1) == grad_confusion)
  }


  test("condition") {
    val gr2 = new DslDriver[Double,Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        val minus_1 = (new NumR(-1.0,var_new(0.0)))
        gradR(x => IF (x.x > 0.0) { minus_1*x*x } { x*x })(x)
      }
    }
    def grad(x: Double) = if (x > 0) -2 * x else 2 * x
    for (x <- (-5 until 5)) {
      assert(gr2.eval(x) == grad(x))
    }
  }

  test("while") {
    val gr3 = new DslDriver[Double,Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        val half = (new NumR(0.5,var_new(0.0)))
        val res = gradR(x => LOOP(x)(x1 => x1.x > 1.0)(x1 => half * x1))(x)
        // println(readVar(half.d))
        res
      }
    }

    System.out.println(gr3.code)
    /*****************************************
      Emitting Generated Code
    *******************************************
    class Snippet extends ((Double)=>(Double)) {
      def apply(x0:Double): Double = {
        val k = { x: Double => 1.0 }
        var loop: [scala.Function1[Double,Double]] = {x10: (Double) =>
          var x11: Double = 0.0
          if (x10 > 1.0) {
            val x13 = 0.5 * x10
            x11 += 0.5 * loop(x13)
          } else {
            x11 += k(x10)
          }
          x11
        }
        loop(x0)
      }
    }
    *****************************************
      End of Generated Code
    *******************************************/
    // Hand-coded correct derivative
    def gfr(x: Double): Double = {
      if (x > 1.0) 0.5 * gfr(0.5 * x) else 1.0
    }
    for (x <- (-5 until 5)) {
      assert(gr3.eval(x) == gfr(x))
    }
  }

  test("loop3times") {
    val gr4 = new DslDriver[Double, Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        val half = new NumR(0.5, var_new(0.0))
        val res = gradR(x => LOOPC(x)(3)(x1 =>{half * x1}))(x)
        println(readVar(half.d))
        res
      }
    }
    def grad(x: Double) = 0.125
    for (x <- (-5 until 5)) {
      assert(gr4.eval(x) == grad(x))
    }
  }


  test("loop_with_index") {
    val gr7 = new DslDriver[Double, Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        val half = new NumR(0.5, var_new(0.0))
        val res = gradR(x => LOOPCC(x)(3)(i => x1 => {
          half * x1 }))(x)
        //assert(readVar(half.d) == grad_half(x))
        readVar(half.d)
      }
    }
    def grad_half(x: Double) = {
      0.75 * x
    }
    for (x <- (-5 until 5)) {
      assert(gr7.eval(x) == grad_half(x))
    }
  }

  test("traverse_array") {
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
    def grad(x: Double): Double = 2 * 3 * 4
    for (x <- (-5 until 5)) {
      assert (gr8.eval(x) == grad(x))
    }
  }

  test("traverse_array2") {
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
    def grad(x: Double): Double = 1.5 * 3.0 * 4.0
    for (x <- (-5 until 5)) {
      assert (gr9.eval(x) == grad(x))
    }
  }


  test("tranverse_array3") {
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
      }
    }
    def grad(x: Double): Double = 1.5 * 2.0 * 3.0
    for (x <- (-5 until 5)) {
      assert (gr10.eval(x) == grad(x))
    }
  }

  test("tranverse_array4") {
    val gr11 = new DslDriver[Double, Double] with DiffApi {

      def snippet(x: Rep[Double]): Rep[Double] = {
        // represent list as array
        val arr = scala.Array(4.0, 3.0, 1.5, 2.0)
        //val arr = scala.Array[Double]()
        val arra = mutableStaticData(arr)

        // create a model that recursively use the data in arr (originated from list)
        def model: NumR => NumR @diff = { (x: NumR) =>
          LOOPL5(0)(x)(arra.length)(i => x1 => new NumR(arra(i), var_new(0.0)) * x1)
        }
        val res = gradR(model)(x)
        res
      }
    }
    def grad(x: Double): Double = 1.5 * 2.0 * 3.0 * 4.0
    for (x <- (-5 until 5)) {
      assert (gr11.eval(x) == grad(x))
    }
  }

  /*
  test("recursive") {
    val gr112 = new DslDriver[Double, Double] with DiffApi {

      def snippet(x: Rep[Double]): Rep[Double] = {

        def model: NumR => NumR @diff = { (y: NumR) =>
          val fun = FUN(if (y.x == 0.0) new NumR(1.0, 0.0) else y * model(y - 1))
          fun(y)
        }

        val res = gradR(fun(model(5)))(x)
        res
      }
    }

    System.out.println(gr112.code)
  }*/

  test("tranverse_array5") {
    val gr11 = new DslDriver[Double, Double] with DiffApi {

      def snippet(x: Rep[Double]): Rep[Double] = {
        val array = mutableStaticData(scala.Array(4.0, 3.0, 1.5, 2.0))

        // create a model that recursively use the data in arr (originated from list)
        def model: NumR => NumR @diff = { (x: NumR) =>
          LOOPL6(0)(x)(array)(i => x1 => new NumR(array(i), var_new(0.0)) * x1)
        }
        val res = gradR(model)(x)
        res
      }
    }
    System.out.println(gr11.code)
    def grad(x: Double): Double = 1.5 * 2.0 * 3.0 * 4.0
    for (x <- (-5 until 5)) {
      assert (gr11.eval(x) == grad(x))
    }
  }
  /** code is correct!
  double Snippet(double  x0) {
    double x1[4] = {4.0,3.0,1.5,2.0};
    function<double(int32_t,function<double(double)>,double)> x3 = [&](int32_t x7,function<double(double)> x8,double x9) {
      double x10 = 0.0;
      if (x7 < 4) {
        function<double(double)> x13 = [&](double x14) {
          double x16 = x1[x7];
          return x16 * x8(x16 * x14);
        };
        x10 += x3(x7 + 1, x13, x9);
      } else {
        x10 += x8(x9);
      }
      return x10;
    };
    return x3(0, [&](double x47) { return 1.0; }, x0);
  }
  **/

  test("tranverse_array6") {
    val gr11 = new DslDriverC[Double, Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        val array = Array(4.0, 3.0, 1.5, 2.0)
        def model: Rep[Array[Double]] => NumR => NumR @diff = { (array: Rep[Array[Double]]) => (x: NumR) =>

          val F = (start: Rep[Int]) => (y: NumR) => shift { k: (NumR => Unit) =>
            lazy val loop: (Rep[Int] => (NumR => Unit) => NumR => Unit) = FUNL1 { (i: Rep[Int]) => (k: NumR => Unit) => (x: NumR) =>
              def sh_loop = (i: Rep[Int]) => (x: NumR) => shift { (k: NumR => Unit) => loop(i)(k)(x) }
              RST(k( IF(i < array.length) { new NumR(array(i), var_new(0.0)) * sh_loop(i+1)(x) } {x} ))
            }
            loop(start)(k)(y)
          }
          F(0)(x)
        }
        gradR(model(array))(x)
      }
    }
    System.out.println(gr11.code)
  }

  /* still correct
  double Snippet(double  x0) {
    double x1[4] = {4.0,3.0,1.5,2.0};
    function<double(int32_t,function<double(double)>,double)> x3 = [&](int32_t x7,function<double(double)> x5,double x6) {
      double x10 = 0.0;
      if (x7 < 4) {
        function<double(double)> x13 = [&](double x14) {
          double x16 = x1[x7];
          return x16 * x5(x16 * x14);
        };
        x10 += x3(x7 + 1,x13, x6);
      } else {
        x10 += x5(x6);
      }
      return x10;
    };
    return x3(0, [&](double x47) {return 1.0;}, x0);
  }
  */

  test("tree") {
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

    System.out.println(gr12.code)
/*
// generating Scala code
class Snippet() extends (Double=>Double) {

  def apply(tree: Tree, base: Double): Double = {

    val rec: scala.Function3[Tree, scala.Function1[Double, Double], Double, Double] = {
      (tree: Tree, cont: scala.Function1[Double, Double], x9: Double) =>

      var x13: Double = 0.0

      if (!tree.isEmpty) {
        val new_cont_l = { x17: (Double) =>
          var x18: Double = 0.0
          val new_cont_r = { x21: (Double) =>
            val x25: Double = tree.value
            var x24: Double = x25 * cont(x17 * x21 * x25)
            x18 += x21 * x24
            x17 * x24
          }
          x13 += rec(tree.right, new_cont_r, x9)
          x18
        }
        x13 += rec(tree.left, new_cont_l, x9)
      } else {
        x13 += cont(x9)
      }
      x13
    }
    rec(tree, {x: (Double) => 1.0}, base)
  }
}

// generating C++ code
double Snippet(double base, Tree tree) {
  function<double(Tree, function<double(double)>, double)> rec = [&](Tree tree, function<double(double)> cont, double base) {
    double x13 = 0.0;
    if (!tree.isEmpty) {
      function<double(double)> cont_l = [&](double x17) {
        double x18 = 0.0;
        function<double(double)> cont_r = [&](double x21) {
          double x25 = tree.value;
          double x24 = x25 * cont(x17 * x21 * x25);;
          x18 += x21 * x24;
          return x17 * x24;
        };
        x13 += rec(tree.right, cont_r, base);
        return x18;
      };
      x13 += rec(tree.left, cont_l, base);
    } else {
      x13 += cont(base);
    }
    return x13;
  };
  return rec(tree, [&](double x){return 1.0;}, base);
}
*/
    def grad(x: Double) = 3 * 4 * 5 * 4 * x * x * x
    for (x <- (-5 until 5)) {
      assert (gr12.eval(x) == grad(x))
    }
  }

  test("tree_closure") {
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
        //printf("the grad of para is %f\n", readVar(para.d))
        readVar(para.d)
      }
    }
    def grad_para(x: Double) = 5.0 * 3.0 * 4.0 * x * x * x * x * 3 * 2 * 2
    for (x <- (-5 until 5)) {
      assert (gr12_2.eval(x) == grad_para(x))
    }
  }

  test("tree_if") {
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
        //printf("the grad of para is %f\n",  readVar(para.d))
        //printf("the grad of paral is %f\n", readVar(paral.d))
        //printf("the grad of parar is %f\n", readVar(parar.d))
        //printf("the grad of bias is %f\n",  readVar(bias.d))
        0.0
      }
    }
    // Fixme
  }

  test("tree_if2") {
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
    // Fix me
    gr12_4.eval(2.0)
  }

}
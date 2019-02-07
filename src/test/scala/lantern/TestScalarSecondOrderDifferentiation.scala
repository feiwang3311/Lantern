package lantern

import scala.util.continuations._
import scala.util.continuations

import scala.virtualization.lms._
import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import org.scalatest.FunSuite

import java.io.PrintWriter
import java.io.File

class ScalarSecondOrderTest extends FunSuite {
  
  test("1") {
    val g1 = new DslDriverScala[String, Unit] with SecOrderApi {
      def snippet(a: Rep[String]): Rep[Unit] = {
        val (grad, hessV) = grad_two_inputs((x1, x2) => x1 * x2)(2.0, 3.0)((4.0, 5.0))
        assert (grad == (3.0, 2.0))
        assert (hessV == (5.0, 4.0)) 
        ()
      }
    }
    g1.eval("a")
  }

  test("1.1") {
    val g1 = new DslDriverScala[String, Unit] with SecOrderApi {
      def snippet(a: Rep[String]): Rep[Unit] = {
        val (grad, hessV) = grad_two_inputs((x1, x2) => x1 * x1 * x2 * x2)(2.0, 3.0)((4.0, 5.0))
        assert (grad == (36.0, 24.0))
        assert (hessV == (192, 136))
        ()
      }
    }
    g1.eval("a")
  }

  test("1.2") {
    val g1 = new DslDriverScala[String, Unit] with SecOrderApi {
      def snippet(a: Rep[String]): Rep[Unit] = {
        val (grad, hessV) = grad_two_inputs((x1, x2) => toNumR(2) * x1 * x1 * x2 + toNumR(3) * x2)(2, 3)((4, 5))
        assert (grad == (24, 11))
        assert (hessV == (88, 32))
        ()
      }
    }
    g1.eval("a")
  }

  test("1.3s") {
    @virtualize
    val g1 = new DslDriverScala[String, Unit] with SecOrderApi {
      def snippet(a: Rep[String]): Rep[Unit] = {
        val (grad, hessV) = grad_two_inputsS((x1, x2) => x1 * x2)(2.0, 3.0)((4.0, 5.0))
        assertVectorEqual(grad, (3.0, 2.0))
        assertVectorEqual(hessV, (5.0, 4.0))
        ()
      }
    }
    g1.eval("a")
  }

  test("1.4s") {
    val g1 = new DslDriverScala[String, Unit] with SecOrderApi {
      def snippet(a: Rep[String]): Rep[Unit] = {
        val (grad, hessV) = grad_two_inputsS((x1, x2) => x1 * x1 * x2 * x2)(2.0, 3.0)((4.0, 5.0))
        assertVectorEqual(grad, (36.0, 24.0))
        assertVectorEqual(hessV, (192, 136))
        ()
      }
    }
    g1.eval("a")
  }

  test("1.5s") {
    val g1 = new DslDriverScala[String, Unit] with SecOrderApi {
      def snippet(a: Rep[String]): Rep[Unit] = {
        val (grad, hessV) = grad_two_inputsS((x1, x2) => toNumRS(2) * x1 * x1 * x2 + toNumRS(3) * x2)(2, 3)((4, 5))
        assertVectorEqual(grad, (24, 11))
        assertVectorEqual(hessV, (88, 32))
        ()
      }
    }
    g1.eval("a")
  }

  test("2") {
    val g1 = new DslDriverScala[String, Unit] with SecOrderApi {
      def snippet(a: Rep[String]): Rep[Unit] = {
        val (grad, hessV) = grad_two_inputs((x1, x2) => (x1 + x2).sin())(2.0, 3.0)((4.0, 5.0))
        assert (grad == (scala.math.cos(5.0), scala.math.cos(5.0)))
        assert (hessV == (-9 * scala.math.sin(5.0), -9 * scala.math.sin(5.0))) 
        ()
      }
    }
    g1.eval("a")
  }

  test("2s") {
    val g1 = new DslDriverScala[String, Unit] with SecOrderApi {
      def snippet(a: Rep[String]): Rep[Unit] = {
        val (grad, hessV) = grad_two_inputsS((x1, x2) => (x1 + x2).sin())(2.0, 3.0)((4.0, 5.0))
        assertVectorEqual(grad, (scala.math.cos(5.0), scala.math.cos(5.0)))
        assertVectorEqual(hessV, (-9 * scala.math.sin(5.0), -9 * scala.math.sin(5.0))) 
        ()
      }
    }
    g1.eval("a")
  }

  test("if_0") {
    val g1 = new DslDriverScala[String, Unit] with SecOrderApi {
      def snippet(a: Rep[String]): Rep[Unit] = {
        val (grad, hessV) = grad_two_inputsS((x1, x2) => IF(x1 < x2){x1 * x2}{x1 * x1 * x2 * x2})(2.0, 3.0)((4.0, 5.0))
        assertVectorEqual(grad, (3.0, 2.0))
        assertVectorEqual(hessV, (5.0, 4.0))
        ()
      }
    }
    g1.eval("a")
  }

  test("if_1") {
    val g1 = new DslDriverScala[String, Unit] with SecOrderApi {
      def snippet(a: Rep[String]): Rep[Unit] = {
        val (grad, hessV) = grad_two_inputsS((x1, x2) => IF(x1 > x2){x1 * x2}{x1 * x1 * x2 * x2})(2.0, 3.0)((4.0, 5.0))
        assertVectorEqual(grad, (36.0, 24.0))
        assertVectorEqual(hessV, (192, 136))
        ()
      }
    }
    println(g1.code)
    g1.eval("a")
  }

  test("while_0") {
    val g1 = new DslDriverScala[String, Unit] with SecOrderApi {
      def snippet(a: Rep[String]): Rep[Unit] = {
        val (grad, hessV) = grad_two_inputsS((x1, x2) => WHILE(x1)(_ < toNumRS(10)){_ * x2} * x1)(2.0, 3.0)((4.0, 5.0))
        assertVectorEqual(grad, (36.0, 24.0))
        assertVectorEqual(hessV, (192, 136))
        ()
      }
    }
    g1.eval("a")
  }

  test("for_0") {
    val g1 = new DslDriverScala[String, Unit] with SecOrderApi {
      def snippet(a: Rep[String]): Rep[Unit] = {
        val (grad, hessV) = grad_two_inputsS((x1, x2) => FOR(x1)(2){(i: Rep[Int], n: NumRS) => n * x2} * x1)(2.0, 3.0)((4.0, 5.0))
        assertVectorEqual(grad, (36.0, 24.0))
        assertVectorEqual(hessV, (192, 136))
        ()
      }
    }
    g1.eval("a")
  }

  test("tree_0") {
    val gr1 = new DslDriverScala[String, Unit] with SecOrderApi {
      def snippet(x: Rep[String]): Rep[Unit] = {
        // represent tree as arrays
        /*    5
             / \
            3   4
        */
        val data = mutableStaticData(scala.Array[Double](5.0, 3.0, 4.0))
        val lch  = mutableStaticData(scala.Array[Int](1, -1, -1)) // -1 means leaf nodes
        val rch  = mutableStaticData(scala.Array[Int](2, -1, -1))
        val x1 = 2.0
        val x2 = 3.0

        val (grad, hessV) = grad_two_inputsS((x1, x2) =>
          TREE(x1)(lch, rch){ (l: NumRS, r: NumRS, i: Rep[Int]) =>
            l * r * new NumRS(NumFS(data(i)), NumFS())
          } * x2 * x2
        )(x1, x2)((0.4, 0.6))

        def dy_x1(x1: Double, x2: Double) = 240 * x1 * x1 * x1 * x2 * x2
        def dy_x2(x1: Double, x2: Double) = 120 * x1 * x1 * x1 * x1 * x2
        def dy_x1x1(x1: Double, x2: Double) = 720 * x1 * x1 * x2 * x2
        def dy_x1x2(x1: Double, x2: Double) = 480 * x1 * x1 * x1 * x2
        def dy_x2x2(x1: Double, x2: Double) = 120 * x1 * x1 * x1 * x1

        assertVectorEqual(grad, (dy_x1(x1, x2), dy_x2(x1, x2)))
        assertVectorEqual(hessV, (dy_x1x1(x1, x2) * 0.4 + dy_x1x2(x1, x2) * 0.6, 
                                  dy_x1x2(x1, x2) * 0.4 + dy_x2x2(x1, x2) * 0.6))
        ()
      }
    }
    println(gr1.code)
    gr1.eval("a")
  }
}
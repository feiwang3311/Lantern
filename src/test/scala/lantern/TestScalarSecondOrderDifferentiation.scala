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

}
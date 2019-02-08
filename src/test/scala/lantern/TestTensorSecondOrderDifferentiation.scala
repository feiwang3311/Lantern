package lantern

import scala.util.continuations._
import scala.util.continuations

import scala.virtualization.lms._
import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import java.io.PrintWriter
import java.io.File

class TensorSecondOrderTest extends LanternFunSuite {  
  
  test("basic") {
    val g1 = new LanternDriverC[String, Unit] with TensorSecOrderApi {
      
      override val fileName = currentTestName
      def snippet(a: Rep[String]): Rep[Unit] = {

        // set input and vector for Hessen
        val x = Tensor.fromData(Seq(4), 1, 2, 3, 4)
        val d = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start = new TensorF(x, d)

        // compute gradient and hessV
        val (grad, hessV) = gradHessV(x => x.sum())(start)
        Tensor.assertEqual(grad, Tensor.fromData(Seq(4), 1, 1, 1, 1))
        Tensor.assertEqual(hessV, Tensor.fromData(Seq(4), 0, 0, 0, 0)) 
        ()
      }
    }
    g1.eval("a")
  }

  test("basic1") {
    val g1 = new LanternDriverC[String, Unit] with TensorSecOrderApi {
      override val fileName = currentTestName
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set input and vector for Hessen
        val x = Tensor.fromData(Seq(4), 1,2,3,4)
        val d = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start = new TensorF(x, d)

        // compute gradient and hessV
        val (grad, hessV) = gradHessV(x => (x + x).sum())(start)

        // correctness assertion
        Tensor.assertEqual(grad, Tensor.fromData(Seq(4), 2,2,2,2))
        Tensor.assertEqual(hessV, Tensor.fromData(Seq(4), 0,0,0,0))
        ()
      }
    }
    g1.eval("a")
  }

  test("basic2") {
    val g1 = new LanternDriverC[String, Unit] with TensorSecOrderApi {
      override val fileName = currentTestName
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set input and vector for Hessen
        val x = Tensor.fromData(Seq(4), 1,2,3,4)
        val d = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start = new TensorF(x, d)

        // compute gradient and hessV
        val (grad, hessV) = gradHessV(x => (x * x).sum())(start)

        // correctness assertion
        Tensor.assertEqual(grad, Tensor.fromData(Seq(4), 2,4,6,8))
        Tensor.assertEqual(hessV, Tensor.fromData(Seq(4), 0.8f,1.0f,1.2f,1.4f))
        ()
      }
    }
    g1.eval("a")
  }

  test("basic2.1") {
    val g1 = new LanternDriverC[String, Unit] with TensorSecOrderApi {
      override val fileName = currentTestName
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set input and vector for Hessen
        val x = Tensor.fromData(Seq(4), 1,2,3,4)
        val d = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start = new TensorF(x, d)

        // compute gradient and hessV
        val (grad, hessV) = gradHessV(x => (x * x * x).sum())(start)

        // correctness assertion
        Tensor.assertEqual(grad, Tensor.fromData(Seq(4), 3,12,27,48))
        Tensor.assertEqual(hessV, Tensor.fromData(Seq(4), 2.4f,6.0f,10.8f,16.8f))
        ()
      }
    }
    g1.eval("a")
  }  

  test("basic3") {
    val g1 = new LanternDriverC[String, Unit] with TensorSecOrderApi {
      override val fileName = currentTestName
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessen
        val x1 = Tensor.fromData(Seq(4), 1, 2, 3, 4)
        val d1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        val x2 = Tensor.fromData(Seq(4), 4, 3, 2, 1)
        val d2 = Tensor.fromData(Seq(4), 0.2f, 0.3f, 0.4f, 0.5f)
        val start2 = TensorFR(new TensorF(x2, d2))

        // compute gradient and hessV
        gradHessV{ () => 
          (start1 * start2).sum
        }

        // correctness assertion
        Tensor.assertEqual(getGradient(start1), x2)
        Tensor.assertEqual(getGradient(start2), x1)
        Tensor.assertEqual(getHessV(start1), d2)
        Tensor.assertEqual(getHessV(start2), d1)
        ()
      }
    }
    g1.eval("a")
  }

  test("basic3.1") {
    val g1 = new LanternDriverC[String, Unit] with TensorSecOrderApi {
      override val fileName = currentTestName
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessen
        val x1 = Tensor.fromData(Seq(4), 1, 2, 3, 4)
        val d1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        val x2 = Tensor.fromData(Seq(4), 4, 3, 2, 1)
        val d2 = Tensor.fromData(Seq(4), 0.2f, 0.3f, 0.4f, 0.5f)
        val start2 = TensorFR(new TensorF(x2, d2))

        // compute gradient and hessV
        gradHessV{ () => 
          (start1 * start2 * start1).sum
        }

        // correctness assertion
        Tensor.assertEqual(getGradient(start1), Tensor.fromData(Seq(4), 8,12,12,8))
        Tensor.assertEqual(getGradient(start2), Tensor.fromData(Seq(4), 1,4,9,16))
        Tensor.assertEqual(getHessV(start1), Tensor.fromData(Seq(4), 3.6f,4.2f,4.8f,5.4f))
        Tensor.assertEqual(getHessV(start2), Tensor.fromData(Seq(4), 0.8f,2f,3.6f,5.6f))
        ()
      }
    }
    g1.eval("a")
  }

  test("basic4") {
    val g1 = new LanternDriverC[String, Unit] with TensorSecOrderApi {
      override val fileName = currentTestName
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessen
        val x1 = Tensor.fromData(Seq(4), 1, 2, 3, 4)
        val d1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        val x2 = Tensor.fromData(Seq(4), 4, 3, 2, 1)
        val d2 = Tensor.fromData(Seq(4), 0.2f, 0.3f, 0.4f, 0.5f)
        val start2 = TensorFR(new TensorF(x2, d2))

        // compute gradient and hessV
        gradHessV{ () => 
          ((start1 * start2) * (start1 + start2)).sum
        }

        // correctness assertion
        Tensor.assertEqual(getGradient(start1), x1 * x2 * 2 + x2 * x2)
        Tensor.assertEqual(getGradient(start2), x1 * x1 + x1 * x2 * 2)
        Tensor.assertEqual(getHessV(start1), Tensor.fromData(Seq(4), 5.2f,6f,6.4f,6.4f))
        Tensor.assertEqual(getHessV(start2), Tensor.fromData(Seq(4), 4.4f,6.2f,8.4f,11f))
        ()
      }
    }
    g1.eval("a")
  }

  test("vv_dot1") {
    val g1 = new LanternDriverC[String, Unit] with TensorSecOrderApi {
      override val fileName = currentTestName
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val x1 = Tensor.fromData(Seq(4), 1,2,3,4)
        val d1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        // compute gradient and hessV
        gradHessV { () =>
          start1 dot start1
        }

        // correctness assertion
        Tensor.assertEqual(getGradient(start1), x1 * 2)
        Tensor.assertEqual(getHessV(start1), d1 * 2)
      }
    }
    g1.eval("a")
  }
  
  test("vv_dot2") {
    val g1 = new LanternDriverC[String, Unit] with TensorSecOrderApi {
      override val fileName = currentTestName
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val x1 = Tensor.fromData(Seq(4), 1,2,3,4)
        val d1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        val x2 = Tensor.fromData(Seq(4), 4, 3, 2, 1)
        val d2 = Tensor.fromData(Seq(4), 0.2f, 0.3f, 0.4f, 0.5f)
        val start2 = TensorFR(new TensorF(x2, d2))

        // compute gradient and hessV
        val res = gradHessV { () =>
          start1 dot start2
        }

        // correctness assertion
        Tensor.assertEqual(res, Tensor.scalar(20))
        Tensor.assertEqual(getGradient(start1), x2)
        Tensor.assertEqual(getGradient(start2), x1)
        Tensor.assertEqual(getHessV(start1), d2)
        Tensor.assertEqual(getHessV(start2), d1)
      }
    }
    g1.eval("a")
  }

  test("mv_dot1") {
    val g1 = new LanternDriverC[String, Unit] with TensorSecOrderApi {
      override val fileName = currentTestName
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val x1 = Tensor.fromData(Seq(2,2), 1,2,3,4)
        val d1 = Tensor.fromData(Seq(2,2), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        val x2 = Tensor.fromData(Seq(2), 4, 3)
        val d2 = Tensor.fromData(Seq(2), 0.2f, 0.3f)
        val start2 = TensorFR(new TensorF(x2, d2))

        // compute gradient and hessV
        val res: Tensor = gradHessV { () =>
          (start1 dot start2).sum()
        }

        // correctness assertion
        Tensor.assertEqual(res, Tensor.scalar(34))
        Tensor.assertEqual(getGradient(start1), Tensor.fromData(Seq(2,2), 4,3,4,3))
        Tensor.assertEqual(getGradient(start2), Tensor.fromData(Seq(2), 4,6))
        Tensor.assertEqual(getHessV(start1), Tensor.fromData(Seq(2,2), 0.2f, 0.3f, 0.2f, 0.3f))
        Tensor.assertEqual(getHessV(start2), Tensor.fromData(Seq(2), 1.0f, 1.2f))
      }
    }
    g1.eval("a")
  }

  test("mm_dot1") {
    val g1 = new LanternDriverC[String, Unit] with TensorSecOrderApi {
      override val fileName = currentTestName
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val x1 = Tensor.fromData(Seq(2,2), 1,2,3,4)
        val d1 = Tensor.fromData(Seq(2,2), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        val x2 = Tensor.fromData(Seq(2,2), 4,3,2,1)
        val d2 = Tensor.fromData(Seq(2,2), 0.2f, 0.3f, 0.4f, 0.5f)
        val start2 = TensorFR(new TensorF(x2, d2))

        // compute gradient and hessV
        val res: Tensor = gradHessV { () =>
          (start1 dot start2).sum()
        }

        // correctness assertion
        Tensor.assertEqual(res, Tensor.scalar(46))
        Tensor.assertEqual(getGradient(start1), Tensor.fromData(Seq(2,2), 7,3,7,3))
        Tensor.assertEqual(getGradient(start2), Tensor.fromData(Seq(2,2), 4,4,6,6))
        Tensor.assertEqual(getHessV(start1), Tensor.fromData(Seq(2,2), 0.5f, 0.9f, 0.5f, 0.9f))
        Tensor.assertEqual(getHessV(start2), Tensor.fromData(Seq(2,2), 1.0f, 1.0f, 1.2f, 1.2f))
      }
    }
    g1.eval("a")
  }

}
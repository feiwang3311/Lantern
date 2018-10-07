package lantern

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import org.scalactic.source
import org.scalatest.{FunSuite, Tag}

import scala.collection.mutable.ArrayBuffer
import scala.collection.{Seq => NSeq}
import java.io.{File, PrintWriter}

class CublasTest extends LanternFunSuite {
  // TODO: Edit this function to actually detect whether GPU codegen is possible.
  // One idea: check for:
  // - The existence of cuBLAS header files (<cuda_runtime.h>, "cublas_v2.h").
  // - The existence of a GPU (perhaps run `nvidia-smi`).
  def isGPUAvailable = false

  testGPU("vector-vector-dot") {
    val vvdot = new LanternDriverCublas[String, Unit] {
      backend = new BackendCublas

      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val length = 2
        val v1 = Tensor.fromData(NSeq(4), 1, 2, 3, 4)
        val v2 = Tensor.fromData(NSeq(4), -1, -2, -3, -4)
        val expected = Tensor.fromData(NSeq(1), -30)
        Tensor.assertEqual(v1.dot(v2), expected)
      }
    }
    runTest(vvdot)
  }

  testGPU("matrix-vector-dot") {
    val mvdot = new LanternDriverCublas[String, Unit] {
      backend = new BackendCublas

      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val m = Tensor.fromData(NSeq(2, 4), 1, 2, 3, 4, 5, 6, 7, 8)
        val v = Tensor.fromData(NSeq(4), -1, -2, -3, -4)
        val expected = Tensor.fromData(NSeq(2), -30, -70)
        Tensor.assertEqual(m.dot(v), expected)
      }
    }
    runTest(mvdot)
  }

  testGPU("matrix-matrix-dot") {
    val mmdot = new LanternDriverCublas[String, Unit] {
      backend = new BackendCublas

      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        // Note: it's better to test with non-square matrices.
        val m1 = Tensor.fromData(NSeq(2, 3), 1, 2, 3, 4, 5, 6)
        val m2 = Tensor.fromData(NSeq(3, 2), 2, 3, 4, 5, 6, 7)
        val expected = Tensor.fromData(NSeq(2, 2), 28, 34, 64, 79)
        Tensor.assertEqual(m1.dot(m2), expected)
      }
    }
    runTest(mmdot)
  }

  // Utility function wrapping `test` that checks whether GPU is available.
  def testGPU(testName: String, testTags: Tag*)(testFun: => Any /* Assertion */)(implicit pos: source.Position) {
    if (isGPUAvailable)
      test(testName, testTags: _*)(testFun)(pos)
    else
      ignore(testName, testTags: _*)(testFun)(pos)
  }
}
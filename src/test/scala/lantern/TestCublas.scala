package lantern

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.collection.{Seq => NSeq}

class TestCublas extends LanternFunSuite {
  testGPU("vector-vector-dot") {
    val vvdot = new LanternDriverCublas[String, Unit] {
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
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
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        // Note: it's better to test with matrices [M1 x M2] and [M2 x M3] where M1 != M3.
        val m1 = Tensor.fromData(NSeq(2, 3), 1, 2, 3, 4, 5, 6)
        val m2 = Tensor.fromData(NSeq(3, 1), 2, 3, 4)
        val expected = Tensor.fromData(NSeq(2, 1), 20, 47)
        Tensor.assertEqual(m1.dot(m2), expected)
      }
    }
    runTest(mmdot)
  }

  // TODO: Fix the other `dot` tests.
  testGPU("matrix-matrix-dot-transfer") {
    val test = new LanternDriverCublas[String, Unit] {
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        backend = BackendCPU()
        // NOTE: `sgemm` seems to act differently than
        val c1 = Tensor.ones(4, 4)
        val c2 = Tensor.ones(4, 4)
        val g1 = c1.toGPU()
        val g2 = c2.toGPU()

        backend = BackendCublas()
        val g3 = g1.dot(g2)
        val c3 = g3.toCPU()

        backend = BackendCPU()
        val expected = Tensor.fill(4, 4, 4)
        Tensor.assertEqual(c3, expected)
        c3.print()
      }
    }
    runTest(test)
  }

  // TODO: Simplify when Tensor initialization on GPU is supported, e.g. `fill` and `rand`.
  testGPU("matrix-matrix-dot-with-backend") {
    val test = new LanternDriverCublas[String, Unit] {
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        backend = BackendCPU()
        val c1 = Tensor.ones(4, 4)
        val c2 = Tensor.ones(4, 4)
        val g1 = c1.toGPU()
        val g2 = c2.toGPU()

        backend = BackendCublas()
        val g3 = g1.dot(g2)

        withCPU(g3) { c3 =>
          val expected = Tensor.fill(4, 4, 4)
          Tensor.assertEqual(c3, expected)
          c3.print()
        }
      }
    }
    runTest(test)
  }
}
package lantern

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

class TestCublas extends LanternFunSuite {
  testGPU("vector-vector-dot") {
    val vvdot = new LanternDriverCublas[String, Unit] {
      override val fileName = "lantern-gpu-vvdot"

      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val v1 = Tensor.fromData(Seq(4), 1, 2, 3, 4)
        val v2 = Tensor.fromData(Seq(4), -1, -2, -3, -4)
        val result = v1.dot(v2).toCPU()

        backend = BackendCPU()
        val expected = Tensor.fromData(Seq(1), -30)
        Tensor.assertEqual(result, expected)
      }
    }
    runTest(vvdot)
  }

  testGPU("matrix-vector-dot") {
    val mvdot = new LanternDriverCublas[String, Unit] {
      override val fileName = "lantern-gpu-mvdot"

      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val m = Tensor.fromData(Seq(2, 4), 1, 2, 3, 4, 5, 6, 7, 8)
        val v = Tensor.fromData(Seq(4), -1, -2, -3, -4)
        val result = m.dot(v).toCPU()

        backend = BackendCPU()
        val expected = Tensor.fromData(Seq(2), -30, -70)
        result.print()
        Tensor.assertEqual(result, expected)
      }
    }
    runTest(mvdot)
  }

  testGPU("matrix-matrix-dot") {
    val mmdot = new LanternDriverCublas[String, Unit] {
      override val fileName = "lantern-gpu-mmdot"

      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val m1 = Tensor.fromData(Seq(2, 3), 1, 2, 3, 4, 5, 6)
        val m2 = Tensor.fromData(Seq(3, 1), 2, 3, 4)
        val result = m1.dot(m2).toCPU()

        backend = BackendCPU()
        val expected = Tensor.fromData(Seq(2, 1), 20, 47)
        result.print()
        Tensor.assertEqual(result, expected)
      }
    }
    runTest(mmdot)
  }

  testGPU("binary-ops") {
    val binops = new LanternDriverCublas[String, Unit] {
      override val fileName = "lantern-gpu-binops"

      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fromData(Seq(2, 2), 1, 2, 3, 4)
        val y1 = Tensor.fromData(Seq(2, 2), 4, 5, 6, 7)
        val y2 = Tensor.fromData(Seq(2, 2), 5, 7, 9, 11)
        val y3 = Tensor.fromData(Seq(2, 2), 2, 3, 4, 5)
        val result = (((x + y1) / y2) * y3).toCPU()

        backend = BackendCPU()
        val expected = Tensor.fromData(Seq(2, 2), 2, 3, 4, 5)
        result.print()
        Tensor.assertEqual(result, expected)
      }
    }
    runTest(binops)
  }

  // TODO: Implement broadcasting.
  testGPU("binary-ops-broadcast") {
    val binops = new LanternDriverCublas[String, Unit] {
      override val fileName = "lantern-gpu-binops-broadcast"

      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fromData(Seq(2, 2), 1, 2, 3, 4)
        val y = Tensor.fromData(Seq(2, 1), 4, 5)
        val result = (x + y).toCPU()

        backend = BackendCPU()
        val expected = Tensor.fromData(Seq(2, 2), 5, 7, 7, 9)
        result.print()
        Tensor.assertEqual(result, expected)
      }
    }
    runTest(binops)
  }
}

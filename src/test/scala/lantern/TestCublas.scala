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
}

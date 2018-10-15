package lantern

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

class TestCublas extends LanternFunSuite {
  testGPU("vector-vector-dot") {
    val vvdot = new LanternDriverCublas[String, Unit] {
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
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        // NOTE: `cublasSgemv` behaves differently than CPU gemv implementation.
        // This test fails for tensors with different scalar values.

        // TODO: Fix this test for the following values:
        val m = Tensor.fromData(Seq(2, 4), 1, 2, 3, 4, 5, 6, 7, 8)
        val v = Tensor.fromData(Seq(4), -1, -2, -3, -4)

        // val m = Tensor.ones(2, 4)
        // val v = Tensor.ones(4)
        val result = m.dot(v).toCPU()

        backend = BackendCPU()
        // val expected = Tensor.fill(Seq(2), 4)
        val expected = Tensor.fromData(Seq(2), -30, -70)
        result.print()
        Tensor.assertEqual(result, expected)
      }
    }
    val file = "src/out/untested/vvd.cu"
    val out = new java.io.PrintWriter(file)
    out.println(mvdot.code)
    out.close()
    runTest(mvdot)
  }

  // TODO: Fix the other `dot` tests.
  testGPU("matrix-matrix-dot") {
    val test = new LanternDriverCublas[String, Unit] {
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        // NOTE: `cublasSgemm` behaves differently than CPU gemm implementation.
        // This test fails for tensors with different scalar values.

        // TODO: Fix this test for the following values:
        val m1 = Tensor.fromData(Seq(2, 3), 1, 2, 3, 4, 5, 6)
        val m2 = Tensor.fromData(Seq(3, 1), 2, 3, 4)

        // val m1 = Tensor.ones(4, 4)
        // val m2 = Tensor.ones(4, 4)
        val result = m1.dot(m2).toCPU()

        backend = BackendCPU()
        // val expected = Tensor.fill(Seq(4, 4), 4)
        val expected = Tensor.fromData(Seq(2, 1), 20, 47)
        Tensor.assertEqual(result, expected)
      }
    }
    runTest(test)
  }
}

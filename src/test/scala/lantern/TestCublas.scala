package lantern

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

class TestCublas extends LanternFunSuite {
  testGPU("vector-vector-dot") {
    val vvdot = new LanternDriverCublas[String, Unit] {
      override val fileName = "lantern-cublas-vvdot"

      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val v1 = Tensor.fromData(Seq(4), 1, 2, 3, 4)
        val v2 = Tensor.fromData(Seq(4), -1, -2, -3, -4)
        val result = v1.dot(v2)
        val v1r = TensorR(v1)
        val v2r = TensorR(v2)
        gradR(dummy => v1r dot v2r)(Tensor.zeros(1))

        backend = BackendCPU()
        val expected = Tensor.scalar(-30)
        Tensor.assertEqual(result.toCPU(), expected)
        Tensor.assertEqual(v1r.d.toCPU(), v2.toCPU())
        Tensor.assertEqual(v2r.d.toCPU(), v1.toCPU())
      }
    }
    runTest(vvdot)
  }

  testGPU("matrix-vector-dot") {
    val mvdot = new LanternDriverCublas[String, Unit] {
      override val fileName = "lantern-cublas-mvdot"

      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val m = Tensor.fromData(Seq(2, 4), 1, 2, 3, 4, 5, 6, 7, 8)
        val v = Tensor.fromData(Seq(4), -1, -2, -3, -4)
        val result = m.dot(v)
        val mm = TensorR(m)
        val vv = TensorR(v)
        gradR(dummy => mm dot vv)(Tensor.zeros(1))

        backend = BackendCPU()
        val expected = Tensor.fromData(Seq(2), -30, -70)
        val expected1 = Tensor.fromData(Seq(2, 4), -1,-2,-3,-4,-1,-2,-3,-4)
        val expected2 = Tensor.fromData(Seq(4), 6, 8, 10, 12)
        Tensor.assertEqual(result.toCPU(), expected)
        Tensor.assertEqual(mm.d.toCPU(), expected1)
        Tensor.assertEqual(vv.d.toCPU(), expected2)
      }
    }
    runTest(mvdot)
  }

  testGPU("matrix-matrix-dot") {
    val mmdot = new LanternDriverCublas[String, Unit] {
      override val fileName = "lantern-cublas-mmdot"

      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val m1 = Tensor.fromData(Seq(2, 3), 1, 2, 3, 4, 5, 6)
        val m2 = Tensor.fromData(Seq(3, 2), 2, 3, 4, 2, 3, 4)
        val result = m1.dot(m2)
        val mm1 = TensorR(m1)
        val mm2 = TensorR(m2)
        gradR(dummy => mm1 dot mm2)(Tensor.zeros(1))

        backend = BackendCPU()
        val expected = Tensor.fromData(Seq(2, 2), 19, 19, 46, 46)
        val expected1 = Tensor.fromData(Seq(2, 3), 5, 6, 7, 5, 6, 7)
        val expected2 = Tensor.fromData(Seq(3, 2), 5, 5, 7, 7, 9, 9)
        Tensor.assertEqual(result.toCPU(), expected)
        Tensor.assertEqual(mm1.d.toCPU(), expected1)
        Tensor.assertEqual(mm2.d.toCPU(), expected2)
      }
    }
    runTest(mmdot)
  }

  testGPU("binary-ops") {
    val binops = new LanternDriverCublas[String, Unit] {
      override val fileName = "lantern-cublas-binops"

      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fromData(Seq(2, 2), 1, 2, 3, 4)
        val y1 = Tensor.fromData(Seq(2, 2), 4, 5, 6, 7)
        val y2 = Tensor.fromData(Seq(2, 2), 5, 7, 9, 11)
        val y3 = Tensor.fromData(Seq(2, 2), 2, 3, 4, 5)
        val result = ((((x + y1) / y2) * y3) - x).toCPU()

        backend = BackendCPU()
        val expected = Tensor.ones(2, 2)
        Tensor.assertEqual(result, expected)
      }
    }
    runTest(binops)
  }

  testGPU("binary-ops-broadcast1") {
    val binops = new LanternDriverCublas[String, Unit] {
      override val fileName = "lantern-cublas-binops-broadcast1"

      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fromData(Seq(2, 1, 4), 1, 2, 3, 4, 5, 6, 7, 8)
        val y = Tensor.fromData(Seq(1, 3, 1), 1, 2, 3)
        val result = (x + y).toCPU()

        backend = BackendCPU()
        val expected = Tensor.fromData(Seq(2, 3, 4),
          2, 3, 4, 5,
          3, 4, 5, 6,
          4, 5, 6, 7,
          6, 7, 8, 9,
          7, 8, 9, 10,
          8, 9, 10, 11)
        Tensor.assertEqual(result, expected)
      }
    }
    runTest(binops)
  }

  testGPU("binary-ops-broadcast2") {
    val binops = new LanternDriverCublas[String, Unit] {
      override val fileName = "lantern-cublas-binops-broadcast2"

      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fromData(Seq(3, 1, 2), 1, 2, 3, 4, 5, 6)
        val y = Tensor.fromData(Seq(3, 1, 1), 1, 2, 3)
        x += y
        x -= y
        x *= y
        x /= y
        val result = x.toCPU()

        backend = BackendCPU()
        val expected = Tensor.fromData(Seq(3, 1, 2), 1, 2, 3, 4, 5, 6)
        Tensor.assertEqual(result, expected)
      }
    }
    runTest(binops)
  }

  testGPU("binary-ops-tensor-scalar") {
    val binops = new LanternDriverCublas[String, Unit] {
      override val fileName = "lantern-cublas-binops-tensor-scalar"

      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fromData(Seq(3, 1, 2), 1, 2, 3, 4, 5, 6)
        x += 4
        x -= 4
        x *= -8
        x /= -8
        val result = x.toCPU()

        backend = BackendCPU()
        val expected = Tensor.fromData(Seq(3, 1, 2), 1, 2, 3, 4, 5, 6)
        Tensor.assertEqual(result, expected)
      }
    }
    runTest(binops)
  }
}

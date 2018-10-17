package lantern

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

class TestCudnn extends LanternFunSuite {
  testGPU("vector-vector-dot") {
    val vvdot = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-vvdot"

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
    val mvdot = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-mvdot"

      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val m = Tensor.fromData(Seq(2, 4), 1, 2, 3, 4, 5, 6, 7, 8)
        val v = Tensor.fromData(Seq(4), -1, -2, -3, -4)
        val result = m.dot(v).toCPU()

        backend = BackendCPU()
        val expected = Tensor.fromData(Seq(2), -30, -70)
        Tensor.assertEqual(result, expected)
      }
    }
    runTest(mvdot)
  }

  testGPU("matrix-matrix-dot") {
    val mmdot = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-mmdot"

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

  testGPU("conv2D") {
    val conv2D = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-conv2d"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.ones(1, 3, 8, 8)
        val kernel = Tensor.ones(1, 3, 3, 3)
        val bias = Tensor.ones(1)
        val strides = Seq(2, 2)
        val pads = Seq(0,0,0,0)
        val result = input.conv2D_batch(kernel, None, strides, pads).toCPU()

        backend = BackendCPU()
        val expected = Tensor.fill(Seq(1,1,3,3), 27.0f)
        result.print()
        Tensor.assertEqual(expected, result)
      }
    }
    runTest(conv2D)
  }

  testGPU("conv2D-bias") {
    val conv2D = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-conv2d-bias"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.ones(1, 3, 8, 8)
        val kernel = Tensor.ones(1, 3, 3, 3)
        val bias = Tensor.ones(1)
        val strides = Seq(2, 2)
        val pads = Seq(0,0,0,0)
        val result = input.conv2D_batch(kernel, Some(bias), strides, pads).toCPU()

        backend = BackendCPU()
        val expected = Tensor.fill(Seq(1,1,3,3), 28.0f)
        result.print()
        Tensor.assertEqual(expected, result)
      }
    }
    runTest(conv2D)
  }
}

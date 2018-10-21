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
        val result = input.conv2D_batch(kernel, None, strides, pads)._1.toCPU()

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
        val result = input.conv2D_batch(kernel, Some(bias), strides, pads)._1.toCPU()

        backend = BackendCPU()
        val expected = Tensor.fill(Seq(1,1,3,3), 28.0f)
        result.print()
        Tensor.assertEqual(expected, result)
      }
    }
    runTest(conv2D)
  }

  testGPU("conv2D-bias-grad") {
    val conv2D = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-conv2d-bias-grad"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = TensorR(Tensor.ones(1,1,4,4))
        val kernel = TensorR(Tensor.ones(1,1,3,3))
        val bias = TensorR(Tensor.zeros(1))
        val strides = Seq(1,1)
        val pads = Seq(0,0,0,0)

        def loss(x: TensorR) = {
          input.convBBP(kernel, Some(bias), strides, pads)
        }
        gradR(loss)(Tensor.zeros(1))
        gradR(loss)(Tensor.zeros(1))

        backend = BackendCPU()
        val expect_input_grad = Tensor.fromData(Seq(1,1,4,4),
          1.0f, 2.0f, 2.0f, 1.0f, 2.0f, 4.0f, 4.0f, 2.0f, 2.0f, 4.0f, 4.0f, 2.0f, 1.0f, 2.0f, 2.0f, 1.0f)
        val expect_kernel_grad = Tensor.fill(Seq(1, 1, 3, 3), 4.0f)
        val expect_bias_grad = Tensor.fromData(Seq(1), 4.0f)

        Tensor.assertEqual(expect_input_grad * 2.0f, input.d.toCPU(), "expect and input.gradient are")
        Tensor.assertEqual(expect_kernel_grad * 2.0f, kernel.d.toCPU(), "expect and kernel.gradient are")
        Tensor.assertEqual(expect_bias_grad * 2.0f, bias.d.toCPU(), "expect and bias.gradient are")
      }
    }
    runTest(conv2D)
  }

  // TODO: Use `gradR_loss` and merge with "relu_grad" test when `sum` is supported on GPU.
  testGPU("relu") {
    val relu = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-relu"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        // TODO: Test NaN values.
        val input = Tensor.fromData(Seq(1,1,2,3), -1, 2, -3, 4, -5, 6)
        val result = input.relu().toCPU()

        backend = BackendCPU()
        val expected = Tensor.fromData(Seq(1,1,2,3), 0, 2, 0, 4, 0, 6)
        Tensor.assertEqual(expected, result)
      }
    }
    runTest(relu)
  }

  testGPU("relu-grad") {
    val relu = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-relu-grad"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        // TODO: Test NaN values.
        val input = TensorR(Tensor.fromData(Seq(1,1,2,3), -1, 2, -3, 4, -5, 6))

        def relu(x: TensorR) = {
          input.relu()
        }
        gradR(relu)(Tensor.zeros(1))

        backend = BackendCPU()
        val expected = Tensor.fromData(Seq(1,1,2,3), 0, 1, 0, 1, 0, 1)
        Tensor.assertEqual(expected, input.d.toCPU())
      }
    }
    runTest(relu)
  }
}

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
        val expected = Tensor.scalar(-30)
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
        Tensor.assertEqual(result, expected)
      }
    }
    runTest(mmdot)
  }

  testGPU("conv2D_forward") {
    val conv2D = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-conv2d"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(1, 1, 4, 4), 1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8)
        val kernel = Tensor.fromData(Seq(1, 1, 2, 2), 1,2,3,4)
        val bias = Tensor.zeros(1)
        val strides = Seq(2, 2)
        val pads = Seq(0,0,0,0)
        val (output, finputOption) = input.conv2D_batch(kernel, Some(bias), strides, pads)
        // output.print("output")
        // assert equal
        backend = BackendCPU()
        val expect = Tensor.fromData(Seq(1,1,2,2), 44, 64, 44, 64)
        Tensor.assertEqual(expect, output.toCPU(), "expect and output are")
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

        def conv(x: TensorR) = {
          input.convBBP(kernel, Some(bias), strides, pads)
        }
        gradR(conv)(Tensor.zeros(1))
        gradR(conv)(Tensor.zeros(1))

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

  testGPU("relu") {
    val relu = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-relu"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        // TODO: Test NaN values.
        val input = Tensor.fromData(Seq(1,1,2,3), -1, 2, -3, 4, -5, 6)
        val result = input.relu()
        val grad = gradR(x => x.relu())(input)

        backend = BackendCPU()
        val expectedRes = Tensor.fromData(Seq(1,1,2,3), 0, 2, 0, 4, 0, 6)
        val expectedGrad = Tensor.fromData(Seq(1,1,2,3), 0, 1, 0, 1, 0, 1)
        Tensor.assertEqual(expectedRes, result.toCPU())
        Tensor.assertEqual(expectedGrad, grad.toCPU())
      }
    }
    runTest(relu)
  }

  testGPU("tanh") {
    val tanh = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-tanh"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.randinit(Seq(1,1,2,3))
        val result = input.tanh()
        val grad = gradR(x => x.tanh())(input)
        val expected = Tensor.ones(1) - result * result
        backend = BackendCPU()
        Tensor.assertEqual(expected.toCPU(), grad.toCPU())
      }
    }
    runTest(tanh)
  }

  testGPU("sigmoid") {
    val sigmoid = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-sigmoid"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.randinit(Seq(1,1,2,3))
        val result = input.sigmoid()
        val grad = gradR(x => x.sigmoid())(input)
        val expected = (Tensor.ones(1) - result) * result
        backend = BackendCPU()
        Tensor.assertEqual(expected.toCPU(), grad.toCPU())
      }
    }
    runTest(sigmoid)
  }

  testGPU("softmax") {
    val softmax = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-softmax"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(2, 3), 1, 2, 3, 4, 5, 6)
        val result = input.softmax_batch()
        val grad = gradR(x => x.softmax_batch())(input)
        backend = BackendCPU()
        val expectedResult = Tensor.fromData(Seq(2, 3),
          0.0900305733f, 0.2447284758f, 0.6652409434f,
          0.0900305733f, 0.2447284758f, 0.6652409434f)
        val expectedGrad = Tensor.fromData(Seq(2, 3),
          0.0000000107f, 0.0000000292f, 0.0000000793f,
          0.0000000107f, 0.0000000292f, 0.0000000793f)
        Tensor.assertEqual(expectedResult, result.toCPU())
        Tensor.assertEqual(expectedGrad, grad.toCPU())
      }
    }
    runTest(softmax)
  }

  testGPU("log-softmax") {
    val logSoftmax = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-log-softmax"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(2, 3), 1, 2, 3, 4, 5, 6)
        val result = input.logSoftmaxB()
        val grad = gradR(x => x.logSoftmaxB())(input)
        backend = BackendCPU()
        val expectedResult = Tensor.fromData(Seq(2, 3),
          -2.4076058865f, -1.4076058865f, -0.4076058865f,
          -2.4076061249f, -1.4076061249f, -0.4076061249f)
        val expectedGrad = Tensor.fromData(Seq(2, 3),
          0.7299082279f, 0.2658145428f, -0.9957230091f,
          0.7299083471f, 0.2658147216f, -0.9957225323f)
        Tensor.assertEqual(expectedResult, result.toCPU())
        Tensor.assertEqual(expectedGrad, grad.toCPU())
      }
    }
    runTest(logSoftmax)
  }

  testGPU("maxPool2D_batch") {
    val maxPool2D = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-maxpool"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(1,1,3,3),1,2,3,4,5,6,7,8,9)
        val kernel = Seq(2,2)
        val strides = Seq(1,1)
        val (output, _) = input.maxPool2D_batch(kernel, strides, None)

        backend = BackendCPU()
        val expect_output = Tensor.fromData(Seq(1,1,2,2), 5, 6, 8, 9)
        Tensor.assertEqual(expect_output, output.toCPU(), "expect and output are")
      }
    }
    runTest(maxPool2D)
  }

  testGPU("maxPool2D_batch_grad") {
    val maxPool2D = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-maxpool-grad"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = TensorR(Tensor.fromData(Seq(1,1,3,3),1,2,3,4,5,6,7,8,9))
        val kernel = Seq(2,2)
        val strides = Seq(1,1)
        def lossFun(dummy: TensorR) = {
          input.maxPoolBK(kernel, strides, None)
        }
        gradR(lossFun)(Tensor.zeros(1))

        backend = BackendCPU()
        val expect_input_grad = Tensor.fromData(Seq(1,1,3,3), 0,0,0,0,1,1,0,1,1)
        Tensor.assertEqual(expect_input_grad, input.d.toCPU(), "expect and output are")
      }
    }
    runTest(maxPool2D)
  }

  testGPU("dropout_batch") {
    val dropout = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-dropout"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(3,3,1,1),1,2,3,4,5,6,7,8,9)
        val prob = 0.0f
        val (output, _, _) = input.dropout(prob)

        backend = BackendCPU()
        Tensor.assertEqual(input.toCPU(), output.toCPU(), "expect and output are")
      }
    }
    runTest(dropout)
  }

  testGPU("dropout_batch_grad") {
    val dropout = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-dropout-grad"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = TensorR(Tensor.fromData(Seq(1,1,3,3),1,2,3,4,5,6,7,8,9))
        val prob = 0.0f
        def lossFun(dummy: TensorR) = {
          input.dropout(prob)
        }
        gradR(lossFun)(Tensor.zeros(1))

        backend = BackendCPU()
        val expect_input_grad = Tensor.fromData(Seq(1,1,3,3), 1,1,1,1,1,1,1,1,1)
        Tensor.assertEqual(expect_input_grad, input.d.toCPU(), "expect and output are")
      }
    }
    runTest(dropout)
  }

  testGPU("nll-loss") {
    val nllLoss = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-nll-loss"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(2, 3), 1, 2, 3, 4, 5, 6)
        val target: Rep[Array[Int]] = Array(1, 0)
        val result = input.logSoftmaxB().nllLossB(target)
        val grad = gradR(x => x.logSoftmaxB().nllLossB(target))(input)

        backend = BackendCPU()
        val expectedResult = Tensor.fromData(Seq(2), 1.4076058865f, 2.4076061249f)
        val expectedGrad = Tensor.fromData(Seq(2, 3),
          0.0900305808f, -0.7552714944f, 0.6652410030f,
          -0.9099694490f, 0.2447284311f, 0.6652408242f)
        Tensor.assertEqual(expectedResult, result.toCPU())
        Tensor.assertEqual(expectedGrad, grad.toCPU())
      }
    }
    runTest(nllLoss)
  }

  testGPU("sum") {
    val sum = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-sum"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(1, 1, 2, 3), 1, 2, 3, 4, 5, 6)
        val result = input.sum()
        val grad = gradR(x => x.sum())(input)

        backend = BackendCPU()
        val expectedResult = Tensor.scalar(10)
        val expectedGrad = Tensor.fromData(Seq(1, 1, 2, 3), 1, 1, 1, 1, 1, 1)
        Tensor.assertEqual(expectedResult, result.toCPU())
        Tensor.assertEqual(expectedGrad, grad.toCPU())
      }
    }
    runTest(sum)
  }

  testGPU("batch-norm") {
    val batchNorm = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-batch-norm"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.ones(1, 2, 1, 1)
        val scale = TensorR(Tensor.ones(1, 2, 1, 1))
        val bias = TensorR(Tensor.zeros(1, 2, 1, 1))
        val runningMean = Tensor.zeros(1, 2, 1, 1)
        val runningVar = Tensor.zeros(1, 2, 1, 1)
        val result = input.batchNorm(scale.x, bias.x, runningMean, runningVar)
        val grad = gradR(x => x.batchNorm(scale, bias, runningMean, runningVar))(input)

        backend = BackendCPU()
        result.toCPU().print()
        grad.toCPU().print()
        val expectedResult = Tensor.fromData(Seq(1, 2, 1, 1), 316.22775f, 316.22775f)
        val expectedGrad = Tensor.fromData(Seq(1, 2, 1, 1), 316.22775f, 316.22775f)
        Tensor.assertEqual(expectedResult, result.toCPU())
        Tensor.assertEqual(expectedGrad, grad.toCPU())
      }
    }
    runTest(batchNorm)
  }
}

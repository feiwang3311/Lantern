package lantern

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

class TestCudnn extends LanternFunSuite {

  testGPU("conv2D-forward") {
    val conv2D = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-conv2d"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(1, 1, 4, 4), 1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8)
        val kernel = Tensor.fromData(Seq(1, 1, 2, 2), 1,2,3,4)
        val bias = Tensor.ones(1)
        val strides = Seq(2, 2)
        val pads = Seq(0,0,0,0)
        val (output, finputOption) = input.conv2D_batch(kernel, Some(bias), strides, pads)

        generateRawComment("check")
        backend = BackendCPU()
        val expect = Tensor.fromData(Seq(1,1,2,2), 45, 65, 45, 65)
        Tensor.assertEqual(expect, output.toCPU(), "expect and output are")
      }
    }
    runTest(conv2D)
  }

  testGPU("conv2D-bias-grad") {
    val conv2D = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-conv2d-bias-grad"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        generateRawComment("input")
        val input = TensorR(Tensor.ones(1,1,4,4))
        val kernel = TensorR(Tensor.ones(1,1,3,3))
        val bias = TensorR(Tensor.zeros(1))
        val strides = Seq(1,1)
        val pads = Seq(0,0,0,0)
        def conv(x: TensorR) = {
          input.convBBP(kernel, Some(bias), strides, pads)
        }
        generateRawComment("grad")
        gradR(conv)(Tensor.zeros(1))

        generateRawComment("check")
        backend = BackendCPU()
        val expect_input_grad = Tensor.fromData(Seq(1,1,4,4),
          1.0f, 2.0f, 2.0f, 1.0f, 2.0f, 4.0f, 4.0f, 2.0f, 2.0f, 4.0f, 4.0f, 2.0f, 1.0f, 2.0f, 2.0f, 1.0f)
        val expect_kernel_grad = Tensor.fill(Seq(1, 1, 3, 3), 4.0f)
        val expect_bias_grad = Tensor.fromData(Seq(1), 4.0f)

        Tensor.assertEqual(expect_input_grad, input.d.toCPU(), "expect and input.gradient are")
        Tensor.assertEqual(expect_kernel_grad, kernel.d.toCPU(), "expect and kernel.gradient are")
        Tensor.assertEqual(expect_bias_grad, bias.d.toCPU(), "expect and bias.gradient are")
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

  testGPU("averagePool2D_batch") {
    val averagePool2D = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-averagePool"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(1,1,3,3),1,2,3,4,5,6,7,8,9)
        val kernel = Seq(2,2)
        val strides = Seq(1,1)
        val output = input.averagePool_batch(kernel, strides, None)

        backend = BackendCPU()
        val expect_output = Tensor.fromData(Seq(1,1,2,2), 3, 4, 6, 7)
        Tensor.assertEqual(expect_output, output.toCPU(), "expect and output are")
      }
    }
    runTest(averagePool2D)
  }

  testGPU("averagePool2D_batch_grad") {
    val averagePool2D = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-averagePool-grad"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = TensorR(Tensor.fromData(Seq(1,1,3,3),1,2,3,4,5,6,7,8,9))
        val kernel = Seq(2,2)
        val strides = Seq(1,1)
        def lossFun(dummy: TensorR) = {
          input.averagePoolBK(kernel, strides, None)
        }
        gradR(lossFun)(Tensor.zeros(1))

        backend = BackendCPU()
        val expect_input_grad = Tensor.fromData(Seq(1,1,3,3), 0.25f,0.5f,0.25f,0.5f,1,0.5f,0.25f,0.5f,0.25f)
        Tensor.assertEqual(expect_input_grad, input.d.toCPU(), "expect and output are")
      }
    }
    runTest(averagePool2D)
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
        val target: Rep[Array[Int]] = GPUArray[Int](1, 0)
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

  testGPU("concat_grad") {
    val concat = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-concat-grad"
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input1 = Tensor.fromData(Seq(2,3,2,2), 1,2,3,4,5,6,7,8,1,2,3,4, 5,6,7,8,1,2,3,4,5,6,7,8)
        val input2 = Tensor.fromData(Seq(2,4,2,2), 3,4,5,6,3,4,5,6,3,4,5,6,3,4,5,6, 6,7,8,9,6,7,8,9,6,7,8,9,6,7,8,9)
        val result = input1.concat(1, input2)
        val input1R = TensorR(input1)
        val input2R = TensorR(input2)
        val grad = gradR(x => input1R.concat(1, input2R))(Tensor.zeros(1))

        backend = BackendCPU()
        val expectedResult = Tensor.fromData(Seq(2,7,2,2), 1,2,3,4,5,6,7,8,1,2,3,4,3,4,5,6,3,4,5,6,3,4,5,6,3,4,5,6,
          5,6,7,8,1,2,3,4,5,6,7,8,6,7,8,9,6,7,8,9,6,7,8,9,6,7,8,9)
        val expectedGrad1 = Tensor.fill(Seq(2,3,2,2), 1.0f)
        val expectedGrad2 = Tensor.fill(Seq(2,4,2,2), 1.0f)
        Tensor.assertEqual(expectedResult, result.toCPU())
        Tensor.assertEqual(expectedGrad1, input1R.d.toCPU())
        Tensor.assertEqual(expectedGrad2, input2R.d.toCPU())
      }
    }
    runTest(concat)
  }

  testGPU("concat_big") {
    val concat = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-concat-grad"
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input1 = Tensor.fill(Seq(100,3,32,32), 1.0f)
        val input2 = Tensor.fill(Seq(100,3,32,32), 1.0f)
        val result = input1.concat(1, input2)
        val input1R = TensorR(input1)
        val input2R = TensorR(input2)
        val grad = gradR(x => input1R.concat(1, input2R))(Tensor.zeros(1))

        backend = BackendCPU()
        val expectedResult = Tensor.fill(Seq(100,6,32,32), 1.0f)
        val expectedGrad1 = Tensor.fill(Seq(100, 3, 32, 32), 1.0f)
        val expectedGrad2 = Tensor.fill(Seq(100, 3, 32, 32), 1.0f)
        Tensor.assertEqual(expectedResult, result.toCPU())
        Tensor.assertEqual(expectedGrad1, input1R.d.toCPU())
        Tensor.assertEqual(expectedGrad2, input2R.d.toCPU())
      }
    }
    runTest(concat)
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
        val expectedResult = Tensor.scalar(21)
        val expectedGrad = Tensor.fromData(Seq(1, 1, 2, 3), 1, 1, 1, 1, 1, 1)
        Tensor.assertEqual(expectedResult, result.toCPU())
        Tensor.assertEqual(expectedGrad, grad.toCPU())
      }
    }
    runTest(sum)
  }

  testGPU("batch-norm-inference") {
    val batchNorm = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-batch-norm"
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.ones(3,2,3,3)
        val scale = Tensor.fromData(Seq(1, 2, 1, 1), 3.0f, 3.0f)
        val bias = Tensor.fromData(Seq(1, 2, 1, 1), 2.0f, 2.0f)
        val runningMean = Tensor.fromData(Seq(1, 2, 1, 1), 3.0f, 3.0f)
        val runningVar = Tensor.fromData(Seq(1, 2, 1, 1), 4.0f, 4.0f)
        val result = input.batchNormInference(scale, bias, runningMean, runningVar)
        backend = BackendCPU()
        val expectedResult = Tensor.fill(Seq(3,2,3,3), -1.0f)
        Tensor.assertEqual(expectedResult, result.toCPU(), tal = 0.0001f)
      }
    }
    runTest(batchNorm)
  }

  testGPU("batch-norm-training") {
    val batchNorm = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-batch-norm-training"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        Tensor.randseed(42)
        val input = Tensor.rand(3, 2, 3, 3)
        val scale = TensorR(Tensor.fromData(Seq(1, 2, 1, 1), 2, 2))
        val bias = TensorR(Tensor.fromData(Seq(1, 2, 1, 1), -1, -1))
        val runningMean = Tensor.ones(1,2,1,1)
        val runningVar = Tensor.ones(1, 2, 1, 1)
        val result = Tensor.zeros(3, 2, 3, 3)
        def lossFun(x: TensorR) = {
          val y = x.batchNorm(scale, bias, runningMean, runningVar)
          result.copy_data(y.x)
          y.relu()
        }
        val grad = gradR(lossFun)(input)

        backend = BackendCPU()
        // result.toCPU().print("result")
        // grad.toCPU().print("grad")
        // TODO (Fei Wang) need to be confirmed by PyTorch
        val expectedResult = Tensor.fromData(Seq(3, 2, 3, 3),
          -4.70055f, -2.42800f, 0.33646f, 
          -1.71884f, -3.37612f, -3.03992f, 
          -0.07803f, 1.66236f, -2.64498f, 
          -4.40594f, -2.17456f, 0.45260f, 
          -2.48382f, -3.67889f, -3.86906f, 
          0.42170f, -4.03358f, -0.95436f, 
          -0.62918f, 0.73989f, 2.56785f, 
          -3.27966f, -1.47426f, -0.98174f, 
          -0.55874f, -0.70138f, 0.62703f, 
          -0.44310f, 0.65435f, -0.73941f, 
          -2.76659f, 0.87396f, 1.42567f, 
          1.76504f, -2.91541f, -3.78245f, 
          -3.30510f, 1.85414f, 2.58171f, 
          -0.99299f, 2.04517f, -2.28546f, 
          -2.79128f, -3.18326f, -1.24514f, 
          -2.01499f, 1.94021f, -0.85590f, 
          1.60013f, -0.91634f, -2.54044f, 
          1.48045f, 0.51965f, 0.44109f)
        val expectedGrad = Tensor.fromData(Seq(3, 2, 3, 3),
          3.09040f, -0.20212f, 3.45742f,
          -1.22957f, 1.17154f, 0.68444f, 
          -3.60681f, 1.53643f, 0.11224f, 
          1.99813f, -1.06226f, 1.89603f, 
          -0.63810f, 1.00096f, 1.26179f, 
          1.93841f, 1.48743f, -2.73582f, 
          -2.80830f, 2.87292f, 0.22452f, 
          1.03179f, -1.58392f, -2.29750f, 
          -2.91035f, -2.70368f, 3.03643f, 
          -3.43702f, 1.61933f, -3.03063f, 
          -0.25029f, 1.31812f, 0.56144f, 
          0.09597f, -0.04617f, 1.14300f, 
          1.06865f, 1.25857f, 0.20445f, 
          -2.28120f, 0.98180f, -0.40863f, 
          0.32422f, 0.89211f, -1.91587f, 
          -1.28113f, -0.14428f, -2.87085f, 
          0.32216f, -2.78796f, -0.56045f, 
          0.48630f, 1.80406f, 1.91181f)
        Tensor.assertEqual(expectedResult, result.toCPU(), tal = 0.0001f)
        Tensor.assertEqual(expectedGrad, grad.toCPU(), tal = 0.0001f)
      }
    }
    runTest(batchNorm)
  }

  testGPU("rnn-inference") {
    val rnnInference = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-rnn-inference"
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val inputSize = 10
        val hiddenSize = 40
        val numLayers = 2
        val seqLength = 5
        val batchSize = 3
        val bidirectional = false
        val numDirections = if (bidirectional) 2 else 1

        def getParameterSize(): Int = {
          val gateSize = hiddenSize
          val w_ih_size = gateSize * inputSize + (numLayers - 1) * gateSize * hiddenSize * numDirections
          val w_hh_size = numLayers * gateSize * hiddenSize
          val b_ih_size = numLayers * gateSize
          val b_hh_size = numLayers * gateSize
          w_ih_size + w_hh_size + b_ih_size + b_hh_size
        }

        val x = Tensor.ones(seqLength, batchSize, inputSize)
        val hx = Tensor.ones(numLayers, batchSize, hiddenSize)
        val w = Tensor.fill(Seq(getParameterSize), 0.01f)
        val res1 = BackendCudnn().cudnnRNNForwardInference(x, hx, w, numLayers = numLayers, hiddenSize = hiddenSize)
        val res2 = BackendCudnn().cudnnRNNForwardTraining(x, hx, w, numLayers = numLayers, hiddenSize = hiddenSize)
        backend = BackendCPU()
        res2.toCPU().print()
        Tensor.assertEqual(res1.toCPU(), res2.toCPU())
      }
    }
    runTest(rnnInference)
  }
}

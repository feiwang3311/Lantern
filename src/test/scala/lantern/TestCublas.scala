package lantern

import lms.core.stub._
import lms.core.virtualize
import lms.macros.SourceContext

class TestCublas extends LanternFunSuite {
  testGPU("vector-vector-dot") {
    val vvdot = new LanternDriverCublas[String, Unit] {
      override val fileName = "lantern-cublas-vvdot"

      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        generate_comment("tensor computation")
        val v1 = Tensor.fromData(Seq(4), 1, 2, 3, 4)
        val v2 = Tensor.fromData(Seq(4), -1, -2, -3, -4)
        val result = v1.dot(v2)

        generate_comment("tensor computation with gradient")
        val v1r = TensorR(v1)
        val v2r = TensorR(v2)
        gradR(dummy => v1r dot v2r)(Tensor.zeros(1))

        backend = BackendCPU()
        generate_comment("checking")
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

  testGPU("gemm_grad") {
    val gemm = new LanternDriverCublas[String, Unit] {
      override val fileName = "lantern-cublas-gemm-grad"
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val m1 = Tensor.rand(2,3)
        val m2 = Tensor.rand(3,4)
        val tr1 = TensorR(m1); val tr2 = TensorR(m2)
        val tr3 = TensorR(m1); val tr4 = TensorR(m2)
        gradR(x => (tr1 dot tr2))(Tensor.zeros(1))
        generate_comment("in between a dot call and a gemm call")
        gradR(x => tr3.gemm(tr4, false, false, 0.5f))(Tensor.zeros(1))

        generate_comment("second test")
        val m3 = Tensor.rand(4,3)
        val tr5 = TensorR(m1); val tr6 = TensorR(m3)
        val tr7 = TensorR(m1); val tr8 = TensorR(m3)
        gradR(x => (tr5 dot tr6.trans()))(Tensor.zeros(1))
        gradR(x => tr7.gemm(tr8, false, true, 0.5f))(Tensor.zeros(1))

        generate_comment("third test")
        val m4 = Tensor.rand(3,2)
        val tr9 = TensorR(m4); val tr10 = TensorR(m2)
        val tr11 = TensorR(m4); val tr12 = TensorR(m2)
        gradR(x => (tr9.trans() dot tr10))(Tensor.zeros(1))
        gradR(x => tr11.gemm(tr12, true, false, 0.5f))(Tensor.zeros(1))

        generate_comment("fourth test")
        val tr13 = TensorR(m4); val tr14 = TensorR(m3)
        val tr15 = TensorR(m4); val tr16 = TensorR(m3)
        gradR(x => (tr13.trans() dot tr14.trans()))(Tensor.zeros(1))
        gradR(x => tr15.gemm(tr16, true, true, 0.5f))(Tensor.zeros(1))

        backend = BackendCPU()
        generate_comment("check for correctness")
        Tensor.assertEqual(tr1.d.toCPU() * 0.5f, tr3.d.toCPU())
        Tensor.assertEqual(tr2.d.toCPU() * 0.5f, tr4.d.toCPU())
        Tensor.assertEqual(tr5.d.toCPU() * 0.5f, tr7.d.toCPU())
        Tensor.assertEqual(tr6.d.toCPU() * 0.5f, tr8.d.toCPU())
        Tensor.assertEqual(tr9.d.toCPU() * 0.5f, tr11.d.toCPU())
        Tensor.assertEqual(tr10.d.toCPU() * 0.5f, tr12.d.toCPU())
        Tensor.assertEqual(tr13.d.toCPU() * 0.5f, tr15.d.toCPU())
        Tensor.assertEqual(tr14.d.toCPU() * 0.5f, tr16.d.toCPU())
      }
    }
    runTest(gemm)
  }

  testGPU("elementwiseOpNoBroadCastSqrt") {
    val sqrt = new LanternDriverCublas[String, Unit] {
      override val fileName = "lantern-cublas-sqrt"
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fill(Seq(3, 2, 3, 3), 16)
        val result = x.sqrt()
        val grad = gradR(x => x.sqrt())(x)

        backend = BackendCPU()
        val expected = Tensor.fill(Seq(3, 2, 3, 3), 4.0f)
        val expectedGrad = Tensor.fill(Seq(3, 2, 3, 3), 0.125f)
        Tensor.assertEqual(expected, result.toCPU())
        Tensor.assertEqual(expectedGrad, grad.toCPU())
      }
    }
    runTest(sqrt)
  }

  testGPU("elementwiseOpNoBroadCastSquare") {
    val square = new LanternDriverCublas[String, Unit] {
      override val fileName = "lantern-cublas-square"
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fill(Seq(3, 2, 3, 3), 3)
        val result = x.square()
        val grad = gradR(x => x.square())(x)

        backend = BackendCPU()
        val expected = Tensor.fill(Seq(3, 2, 3, 3), 9.0f)
        val expectedGrad = Tensor.fill(Seq(3, 2, 3, 3), 6.0f)
        Tensor.assertEqual(expected, result.toCPU())
        Tensor.assertEqual(expectedGrad, grad.toCPU())
      }
    }
    runTest(square)
  }

  testGPU("elementwiseOpNoBroadCastExp") {
    val exp = new LanternDriverCublas[String, Unit] {
      override val fileName = "lantern-cublas-exp"
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fill(Seq(3, 2, 3, 3), 0.5f)
        val result = x.exp()
        val grad = gradR(x => x.exp())(x)

        backend = BackendCPU()
        val expected = Tensor.fill(Seq(3, 2, 3, 3), 1.64872127f)
        val expectedGrad = Tensor.fill(Seq(3, 2, 3, 3), 1.64872127f)
        Tensor.assertEqual(expected, result.toCPU())
        Tensor.assertEqual(expectedGrad, grad.toCPU())
      }
    }
    runTest(exp)
  }

  testGPU("elementwiseOpNoBroadCastLog") {
    val exp = new LanternDriverCublas[String, Unit] {
      override val fileName = "lantern-cublas-log"
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fill(Seq(3, 2, 3, 3), 2)
        val result = x.log()
        val grad = gradR(x => x.log())(x)

        backend = BackendCPU()
        val expected = Tensor.fill(Seq(3, 2, 3, 3), 0.6931471f)
        val expectedGrad = Tensor.fill(Seq(3, 2, 3, 3), 0.5f)
        Tensor.assertEqual(expected, result.toCPU())
        Tensor.assertEqual(expectedGrad, grad.toCPU())
      }
    }
    runTest(exp)
  }

  testGPU("mask4D") {
    val exp = new LanternDriverCublas[String, Unit] {
      override val fileName = "lantern-cublas-mask4D"
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fill(Seq(2, 3, 2, 3), 2)
        val lengths: Rep[Array[Int]] = Array(1, 2)
        val lengthsGPU = lengths.toGPU(2)
        val result = x.mask4D(lengthsGPU)
        backend = BackendCPU()
        val expected = Tensor.fromData(Seq(2, 3, 2, 3), 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0,
                                                        2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0)
        Tensor.assertEqual(expected, result.toCPU())
      }
    }
    runTest(exp)
  }

  testGPU("permute-2D") {
    val exp = new LanternDriverCublas[String, Unit] {
      override val fileName = currentTestName
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fromData(Seq(2, 4), 1, 2, 3, 4, 5, 6, 7, 8)
        val result = x.permute(1,0)

        generate_comment("checking result")
        backend = BackendCPU()
        val expected = Tensor.fromData(Seq(4, 2), 1,5, 2,6, 3, 7, 4, 8)
        Tensor.assertEqual(expected, result.toCPU())
      }
    }
    runTest(exp)
  }

  testGPU("permute-3D") {
    val exp = new LanternDriverCublas[String, Unit] {
      override val fileName = currentTestName
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fromData(Seq(2,3,4),
          0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)
        val result1 = x.permute(1,0,2)
        val result2 = x.permute(2,1,0)
        val result3 = x.permute(1,2,0)
        val result4 = x.permute(2,0,1)
        val result5 = x.permute(0,2,1)

        generate_comment("checking result")
        backend = BackendCPU()
        val expected1 = Tensor.fromData(Seq(3,2,4),
          0,  1,  2,  3, 12, 13, 14, 15,  4,  5,  6,  7, 16, 17, 18, 19,  8,  9,
          10, 11, 20, 21, 22, 23)
        Tensor.assertEqual(expected1, result1.toCPU())
        val expected2 = Tensor.fromData(Seq(4,3,2),
         0, 12,  4, 16,  8, 20,  1, 13,  5, 17,  9, 21,  2, 14,  6, 18, 10, 22,
         3, 15,  7, 19, 11, 23)
        Tensor.assertEqual(expected2, result2.toCPU())
        val expected3 = Tensor.fromData(Seq(3,4,2),
          0, 12,  1, 13,  2, 14,  3, 15,  4, 16,  5, 17,  6, 18,  7, 19,  8, 20,
          9, 21, 10, 22, 11, 23)
        Tensor.assertEqual(expected3, result3.toCPU())
        val expected4 = Tensor.fromData(Seq(4,2,3),
          0,  4,  8, 12, 16, 20,  1,  5,  9, 13, 17, 21,  2,  6, 10, 14, 18, 22,
          3,  7, 11, 15, 19, 23)
        Tensor.assertEqual(expected4, result4.toCPU())
        val expected5 = Tensor.fromData(Seq(2,4,3),
          0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11, 12, 16, 20, 13, 17, 21,
          14, 18, 22, 15, 19, 23)
        Tensor.assertEqual(expected5, result5.toCPU())
      }
    }
    runTest(exp)
  }

  testGPU("permute-4D-sim") {
    val exp = new LanternDriverCublas[String, Unit] {
      override val fileName = currentTestName
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fromData(Seq(2,3,2,3),
          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
          19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35)
        val result1 = x.permute(0,2,1,3)
        val result2 = x.permute(1,0,2,3)
        val result3 = x.permute(1,2,0,3)
        val result4 = x.permute(2,0,1,3)
        val result5 = x.permute(2,1,0,3)

        generate_comment("checking result")
        backend = BackendCPU()
        val expected1 = Tensor.fromData(Seq(2,2,3,3),
          0,  1,  2,  6,  7,  8, 12, 13, 14,  3,  4,  5,  9, 10, 11, 15, 16, 17,
          18, 19, 20, 24, 25, 26, 30, 31, 32, 21, 22, 23, 27, 28, 29, 33, 34, 35)
        Tensor.assertEqual(expected1, result1.toCPU())
        val expected2 = Tensor.fromData(Seq(3,2,2,3),
          0,  1,  2,  3,  4,  5, 18, 19, 20, 21, 22, 23,  6,  7,  8,  9, 10, 11,
          24, 25, 26, 27, 28, 29, 12, 13, 14, 15, 16, 17, 30, 31, 32, 33, 34, 35)
        Tensor.assertEqual(expected2, result2.toCPU())
        val expected3 = Tensor.fromData(Seq(3,2,2,3),
          0,  1,  2, 18, 19, 20,  3,  4,  5, 21, 22, 23,  6,  7,  8, 24, 25, 26,
          9, 10, 11, 27, 28, 29, 12, 13, 14, 30, 31, 32, 15, 16, 17, 33, 34, 35)
        Tensor.assertEqual(expected3, result3.toCPU())
        val expected4 = Tensor.fromData(Seq(2,2,3,3),
          0,  1,  2,  6,  7,  8, 12, 13, 14, 18, 19, 20, 24, 25, 26, 30, 31, 32,
          3,  4,  5,  9, 10, 11, 15, 16, 17, 21, 22, 23, 27, 28, 29, 33, 34, 35)
        Tensor.assertEqual(expected4, result4.toCPU())
        val expected5 = Tensor.fromData(Seq(2,3,2,3),
          0,  1,  2, 18, 19, 20,  6,  7,  8, 24, 25, 26, 12, 13, 14, 30, 31, 32,
          3,  4,  5, 21, 22, 23,  9, 10, 11, 27, 28, 29, 15, 16, 17, 33, 34, 35)
        Tensor.assertEqual(expected5, result5.toCPU())
      }
    }
    runTest(exp)
  }
}

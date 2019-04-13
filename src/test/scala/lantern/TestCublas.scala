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

  testGPU("gemm") {
    val gemm = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cublas-gemm"
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val m1 = Tensor.rand(2,3)
        val m2 = Tensor.rand(3,4)
        val viaDot = (m1 dot m2) * 0.5f
        val viaGemm = m1.gemm(m2, false, false, 0.5f)
        Tensor.assertEqual(viaDot.toCPU(), viaGemm.toCPU())

        val m3 = Tensor.rand(4,3)
        val temp = m3.trans()
        val viaDot01 = (m1 dot temp) * 0.5f
        val viaGemm01 = m1.gemm(m3, false, true, 0.5f)
        Tensor.assertEqual(viaDot01.toCPU(), viaGemm01.toCPU())

        val m4 = Tensor.rand(3,2)
        val viaDot10 = (m4.trans() dot m2) * 0.5f
        val viaGemm10 = m4.gemm(m2, true, false, 0.5f)
        Tensor.assertEqual(viaDot10.toCPU(), viaGemm10.toCPU())

        val viaDot11 = (m4.trans() dot m3.trans()) * 0.5f
        val viaGemm11 = m4.gemm(m3, true, true, 0.5f)
        Tensor.assertEqual(viaDot11.toCPU(), viaGemm11.toCPU())
      }
    }
    runTest(gemm)
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

  testGPU("binary-ops") {
    val binops = new LanternDriverCudnn[String, Unit] {
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
    val binops = new LanternDriverCudnn[String, Unit] {
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

//  testGPU("binary-ops-broadcast2") {
//    val binops = new LanternDriverCublas[String, Unit] {
//      override val fileName = "lantern-cublas-binops-broadcast2"

//      @virtualize
//      def snippet(x: Rep[String]): Rep[Unit] = {
//        val x = Tensor.fromData(Seq(3, 1, 2), 1, 2, 3, 4, 5, 6)
//        val y = Tensor.fromData(Seq(3, 1, 1), 1, 2, 3)
//        x += y
//        x -= y
//        x *= y
//        x /= y
//        val result = x.toCPU()

//        backend = BackendCPU()
//        val expected = Tensor.fromData(Seq(3, 1, 2), 1, 2, 3, 4, 5, 6)
//        Tensor.assertEqual(result, expected)
//      }
//    }
//    runTest(binops)
//  }

  testGPU("binary-ops-tensor-scalar") {
    val binops = new LanternDriverCudnn[String, Unit] {
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
}

package lantern

import scala.util.continuations._
import scala.util.continuations

import scala.virtualization.lms._
import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import java.io.PrintWriter
import java.io.File

class TensorSecondOrderCudnnTest extends LanternFunSuite {

  testGPU("sum0") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {

      override val fileName = "secOrder-gpu-sum0"
      def snippet(a: Rep[String]): Rep[Unit] = {

        // set input and vector for Hessen
        val x = Tensor.fromData(Seq(4), 1, 2, 3, 4)
        val d = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start = new TensorF(x, d)

        // compute gradient and hessV
        val (grad, hessV) = gradHessV(x => x.sum())(start)
        backend = BackendCPU()
        Tensor.assertEqual(grad.toCPU(), Tensor.fromData(Seq(4), 1, 1, 1, 1))
        Tensor.assertEqual(hessV.toCPU(), Tensor.fromData(Seq(4), 0, 0, 0, 0))
        ()
      }
    }
    g1.eval("a")
  }

  testGPU("add0") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-add0"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set input and vector for Hessen
        val x = Tensor.fromData(Seq(4), 1,2,3,4)
        val d = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start = new TensorF(x, d)

        // compute gradient and hessV
        val (grad, hessV) = gradHessV(x => (x + x).sum())(start)

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(grad.toCPU(), Tensor.fromData(Seq(4), 2,2,2,2))
        Tensor.assertEqual(hessV.toCPU(), Tensor.fromData(Seq(4), 0,0,0,0))
        ()
      }
    }
    g1.eval("a")
  }

  testGPU("tanhff") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-tanhff"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set input and vector for Hessian
        val x1 = Tensor.fromData(Seq(4), 1.0f, 2.0f, 0.5f, 0.2f)
        val d1 = Tensor.fromData(Seq(4), 1.0f, 1.0f, 1.0f, 1.0f)
        val start = TensorFR(new TensorF(x1, d1))

        val res = gradHessV{() => start.tanh.sum}

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(getGradient(start).toCPU(), Tensor.fromData(Seq(4), 0.419974f, 0.070651f, 0.786448f, 0.961043f))
        Tensor.assertEqual(getHessV(start).toCPU(), Tensor.fromData(Seq(4), -0.639700f, -0.136219f, -0.726862f, -0.379372f))
        // PyTorch equvilent code
        // start1 = torch.tensor([1.0, 2.0, 0.5, 0.2], requires_grad=True)
        // out = (start1).tanh().sum()
        // grads = torch.autograd.grad(out, [start1], create_graph=True, retain_graph = True)
        // flatten = torch.cat([g.reshape(-1) for g in grads if g is not None])
        // v = torch.tensor([1.0, 1.0, 1.0, 1.0])
        // hvps = torch.autograd.grad([flatten @ v], [start1], allow_unused=True)
        // torch.set_printoptions(precision=6)
        // print(out.data) # tensor(3.891992)
        // print(grads[0].data) # tensor([0.070651, 0.000182, 0.070651, 0.070651])
        // print(hvps[0].data)  # tensor([-0.081731, -0.000290, -0.136219, -0.163462])
      }
    }
    g1.eval("a")
  }

  testGPU("add_tanh") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-add_tanh"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set input and vector for Hessen
        val x1 = Tensor.fromData(Seq(4), 5.0f, 7.0f, 3.0f, 2.0f)
        val d1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        val x2 = Tensor.fromData(Seq(4), -4, -3, -2, -1)
        val d2 = Tensor.fromData(Seq(4), 0.2f, 0.3f, 0.4f, 0.5f)
        val start2 = TensorFR(new TensorF(x2, d2))

        // compute gradient and hessV
        val res = gradHessV{ () =>
          (start1 + start2 + 1).tanh.sum
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(3.891992f))
        Tensor.assertEqual(getGradient(start1).toCPU(), Tensor.fromData(Seq(4), 0.070651f, 0.000182f, 0.070651f, 0.070651f))
        Tensor.assertEqual(getGradient(start2).toCPU(), Tensor.fromData(Seq(4), 0.070651f, 0.000182f, 0.070651f, 0.070651f))
        Tensor.assertEqual(getHessV(start1).toCPU(), Tensor.fromData(Seq(4), -0.081731f, -0.000290f, -0.136219f, -0.163462f))
        Tensor.assertEqual(getHessV(start2).toCPU(), Tensor.fromData(Seq(4), -0.081731f, -0.000290f, -0.136219f, -0.163462f))
        // PyTorch equvilent code
        // start1 = torch.tensor([5.0, 7.0, 3.0, 2.0], requires_grad=True)
        // start2 = torch.tensor([-4.0, -3.0, -2.0, -1.0], requires_grad=True)
        // out = (start1 + start2 + 1).tanh().sum()
        // grads = torch.autograd.grad(out, [start1, start2], create_graph=True, retain_graph = True)
        // flatten = torch.cat([g.reshape(-1) for g in grads if g is not None])
        // v = torch.tensor([0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5])
        // hvps = torch.autograd.grad([flatten @ v], [start1, start2], allow_unused=True)
        // torch.set_printoptions(precision=6)
        // print(out.data) # tensor(3.891992)
        // print(grads[0].data) # tensor([0.070651, 0.000182, 0.070651, 0.070651])
        // print(grads[1].data) # tensor([0.070651, 0.000182, 0.070651, 0.070651])
        // print(hvps[0].data)  # tensor([-0.081731, -0.000290, -0.136219, -0.163462])
        // print(hvps[1].data)  # tensor([-0.081731, -0.000290, -0.136219, -0.163462])
      }
    }
    g1.eval("a")
  }


  testGPU("minus_tanh") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-minus_tanh"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set input and vector for Hessen
        val x1 = Tensor.fromData(Seq(4), 6.0f, 7.0f, 3.0f, 2.0f)
        val d1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        val x2 = Tensor.fromData(Seq(4), 4, 3, 2, 1)
        val d2 = Tensor.fromData(Seq(4), 0.2f, 0.3f, 0.4f, 0.5f)
        val start2 = TensorFR(new TensorF(x2, d2))

        // compute gradient and hessV
        val res = gradHessV{ () =>
          (start1 - start2 - 2).tanh.sum
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(-0.559161f))
        Tensor.assertEqual(getGradient(start1).toCPU(), Tensor.fromData(Seq(4), 1.000000f, 0.070651f, 0.419974f, 0.419974f))
        Tensor.assertEqual(getGradient(start2).toCPU(), Tensor.fromData(Seq(4), -1.000000f, -0.070651f, -0.419974f, -0.419974f))
        Tensor.assertEqual(getHessV(start1).toCPU(), Tensor.fromData(Seq(4), -0.000000f, -0.027244f, 0.127940f, 0.127940f))
        Tensor.assertEqual(getHessV(start2).toCPU(), Tensor.fromData(Seq(4), 0.000000f,  0.027244f, -0.127940f, -0.127940f))
        // PyTorch equvilent code
        // start1 = torch.tensor([6.0, 7.0, 3.0, 2.0], requires_grad=True)
        // start2 = torch.tensor([4.0, 3.0, 2.0, 1.0], requires_grad=True)
        // out = (start1 - start2 - 2).tanh().sum()
        // grads = torch.autograd.grad(out, [start1, start2], create_graph=True, retain_graph = True)
        // flatten = torch.cat([g.reshape(-1) for g in grads if g is not None])
        // v = torch.tensor([0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5])
        // hvps = torch.autograd.grad([flatten @ v], [start1, start2], allow_unused=True)
        // torch.set_printoptions(precision=6)
        // print(out.data) # tensor(-0.559161)
        // print(grads[0].data) # tensor([1.000000, 0.070651, 0.419974, 0.419974])
        // print(grads[1].data) # tensor([-1.000000, -0.070651, -0.419974, -0.419974])
        // print(hvps[0].data)  # tensor([-0.000000, -0.027244,  0.127940,  0.127940])
        // print(hvps[1].data)  # tensor([ 0.000000,  0.027244, -0.127940, -0.127940])
      }
    }
    g1.eval("a")
  }

  testGPU("mult_tanh") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-mult_tanh"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set input and vector for Hessen
        val x1 = Tensor.fromData(Seq(4), -0.6f, 0.7f, -0.3f, 0.2f)
        val d1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        val x2 = Tensor.fromData(Seq(4), 0.4f, 0.3f, 0.2f, 1)
        val d2 = Tensor.fromData(Seq(4), 0.2f, 0.3f, 0.4f, 0.5f)
        val start2 = TensorFR(new TensorF(x2, d2))

        // compute gradient and hessV
        val res = gradHessV{ () =>
          (start1 * start2 * 2).tanh.sum
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(0.211209f))
        Tensor.assertEqual(getGradient(start1).toCPU(), Tensor.fromData(Seq(4), 0.640693f, 0.505468f, 0.394295f, 1.711277f))
        Tensor.assertEqual(getGradient(start2).toCPU(), Tensor.fromData(Seq(4), -0.961040f,  1.179425f, -0.591442f,  0.342256f))
        Tensor.assertEqual(getHessV(start1).toCPU(), Tensor.fromData(Seq(4), 0.366092f,  0.216553f,  0.788590f, -1.224995f))
        Tensor.assertEqual(getHessV(start2).toCPU(), Tensor.fromData(Seq(4), 0.572076f, 0.168311f, 1.182885f, 0.781767f))
        // PyTorch equvilent code
        // start1 = torch.tensor([-0.6, 0.7, -0.3, 0.2], requires_grad=True)
        // start2 = torch.tensor([0.4, 0.3, 0.2, 1.0], requires_grad=True)
        // out = (start1 * start2 * 2).tanh().sum()
        // grads = torch.autograd.grad(out, [start1, start2], create_graph=True, retain_graph = True)
        // flatten = torch.cat([g.reshape(-1) for g in grads if g is not None])
        // v = torch.tensor([0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5])
        // hvps = torch.autograd.grad([flatten @ v], [start1, start2], allow_unused=True)
        // torch.set_printoptions(precision=6)
        // print(out.data) # tensor(0.211209)
        // print(grads[0].data) # tensor([0.640693, 0.505468, 0.394295, 1.711277])
        // print(grads[1].data) # tensor([-0.961040,  1.179425, -0.591442,  0.342256])
        // print(hvps[0].data)  # tensor([ 0.366092,  0.216553,  0.788590, -1.224995])
        // print(hvps[1].data)  # tensor([0.572076, 0.168311, 1.182885, 0.781767])
      }
    }
    g1.eval("a")
  }

  testGPU("div_tanh") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-div_tanh"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set input and vector for Hessen
        val x1 = Tensor.fromData(Seq(4), 5.0f, 7.0f, 3.0f, 2.0f)
        val d1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        val x2 = Tensor.fromData(Seq(4), 4, 3, 2, 1)
        val d2 = Tensor.fromData(Seq(4), 0.2f, 0.3f, 0.4f, 0.5f)
        val start2 = TensorFR(new TensorF(x2, d2))

        // compute gradient and hessV
        val res = gradHessV{ () =>
          (start1 / start2 / 2).tanh.sum
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(2.774543f))
        Tensor.assertEqual(getGradient(start1).toCPU(), Tensor.fromData(Seq(4), 0.086552f, 0.053723f, 0.149146f, 0.209987f))
        Tensor.assertEqual(getGradient(start2).toCPU(), Tensor.fromData(Seq(4), -0.108190f, -0.125355f, -0.223720f, -0.419974f))
        Tensor.assertEqual(getHessV(start1).toCPU(), Tensor.fromData(Seq(4), -0.006128f, -0.002424f, -0.029829f, -0.057016f))
        Tensor.assertEqual(getHessV(start2).toCPU(), Tensor.fromData(Seq(4), 0.004414f, 0.009238f, 0.044744f, 0.177028f))
        // PyTorch equvilent code
        // start1 = torch.tensor([5.0, 7.0, 3.0, 2.0], requires_grad=True)
        // start2 = torch.tensor([4.0, 3.0, 2.0, 1.0], requires_grad=True)
        // out = (start1 / start2 / 2).tanh().sum()
        // grads = torch.autograd.grad(out, [start1, start2], create_graph=True, retain_graph = True)
        // flatten = torch.cat([g.reshape(-1) for g in grads if g is not None])
        // v = torch.tensor([0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5])
        // hvps = torch.autograd.grad([flatten @ v], [start1, start2], allow_unused=True)
        // torch.set_printoptions(precision=6)
        // print(out.data) # tensor(2.774543)
        // print(grads[0].data) # tensor([0.086552, 0.053723, 0.149146, 0.209987])
        // print(grads[1].data) # tensor([-0.108190, -0.125355, -0.223720, -0.419974])
        // print(hvps[0].data)  # tensor([-0.006128, -0.002424, -0.029829, -0.057016])
        // print(hvps[1].data)  # tensor([0.004414, 0.009238, 0.044744, 0.177028])
      }
    }
    g1.eval("a")
  }

  testGPU("basic3") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-basic3"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessen
        val x1 = Tensor.fromData(Seq(4), 1, 2, 3, 4)
        val d1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        val x2 = Tensor.fromData(Seq(4), 4, 3, 2, 1)
        val d2 = Tensor.fromData(Seq(4), 0.2f, 0.3f, 0.4f, 0.5f)
        val start2 = TensorFR(new TensorF(x2, d2))

        // compute gradient and hessV
        gradHessV{ () =>
          (start1 * start2).sum
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(getGradient(start1).toCPU(), x2.toCPU())
        Tensor.assertEqual(getGradient(start2).toCPU(), x1.toCPU())
        Tensor.assertEqual(getHessV(start1).toCPU(), d2.toCPU())
        Tensor.assertEqual(getHessV(start2).toCPU(), d1.toCPU())
        ()
      }
    }
    g1.eval("a")
  }

  testGPU("basic3.1") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-basic3.1"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessen
        val x1 = Tensor.fromData(Seq(4), 1, 2, 3, 4)
        val d1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        val x2 = Tensor.fromData(Seq(4), 4, 3, 2, 1)
        val d2 = Tensor.fromData(Seq(4), 0.2f, 0.3f, 0.4f, 0.5f)
        val start2 = TensorFR(new TensorF(x2, d2))

        // compute gradient and hessV
        gradHessV{ () =>
          (start1 * start2 * start1).sum
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(getGradient(start1).toCPU(), Tensor.fromData(Seq(4), 8,12,12,8))
        Tensor.assertEqual(getGradient(start2).toCPU(), Tensor.fromData(Seq(4), 1,4,9,16))
        Tensor.assertEqual(getHessV(start1).toCPU(), Tensor.fromData(Seq(4), 3.6f,4.2f,4.8f,5.4f))
        Tensor.assertEqual(getHessV(start2).toCPU(), Tensor.fromData(Seq(4), 0.8f,2f,3.6f,5.6f))
        ()
      }
    }
    g1.eval("a")
  }

  testGPU("basic4") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-basic4"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessen
        val x1 = Tensor.fromData(Seq(4), 1, 2, 3, 4)
        val d1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        val x2 = Tensor.fromData(Seq(4), 4, 3, 2, 1)
        val d2 = Tensor.fromData(Seq(4), 0.2f, 0.3f, 0.4f, 0.5f)
        val start2 = TensorFR(new TensorF(x2, d2))

        // compute gradient and hessV
        gradHessV{ () =>
          ((start1 * start2) * (start1 + start2)).sum
        }

        // correctness assertion
        val exp1 = x1 * x2 * 2 + x2 * x2
        val exp2 = x1 * x1 + x1 * x2 * 2
        backend = BackendCPU()
        Tensor.assertEqual(getGradient(start1).toCPU(), exp1.toCPU())
        Tensor.assertEqual(getGradient(start2).toCPU(), exp2.toCPU())
        Tensor.assertEqual(getHessV(start1).toCPU(), Tensor.fromData(Seq(4), 5.2f,6f,6.4f,6.4f))
        Tensor.assertEqual(getHessV(start2).toCPU(), Tensor.fromData(Seq(4), 4.4f,6.2f,8.4f,11f))
        ()
      }
    }
    g1.eval("a")
  }

  testGPU("vv_dot1") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-vv_dot1"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val x1 = Tensor.fromData(Seq(4), 1,2,3,4)
        val d1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        // compute gradient and hessV
        gradHessV { () =>
          start1 dot start1
        }

        // correctness assertion
        val exp1 = x1 * 2
        val exp2 = d1 * 2
        backend = BackendCPU()
        Tensor.assertEqual(getGradient(start1).toCPU(), exp1.toCPU())
        Tensor.assertEqual(getHessV(start1).toCPU(), exp2.toCPU())
      }
    }
    g1.eval("a")
  }

  testGPU("vv_dot2") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-vv_dot2"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val x1 = Tensor.fromData(Seq(4), 1,2,3,4)
        val d1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        val x2 = Tensor.fromData(Seq(4), 4, 3, 2, 1)
        val d2 = Tensor.fromData(Seq(4), 0.2f, 0.3f, 0.4f, 0.5f)
        val start2 = TensorFR(new TensorF(x2, d2))

        // compute gradient and hessV
        val res = gradHessV { () =>
          start1 dot start2
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(20))
        Tensor.assertEqual(getGradient(start1).toCPU(), x2.toCPU())
        Tensor.assertEqual(getGradient(start2).toCPU(), x1.toCPU())
        Tensor.assertEqual(getHessV(start1).toCPU(), d2.toCPU())
        Tensor.assertEqual(getHessV(start2).toCPU(), d1.toCPU())
      }
    }
    g1.eval("a")
  }

  testGPU("mv_dot1") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-mv_dot1"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val x1 = Tensor.fromData(Seq(2,2), 1,2,3,4)
        val d1 = Tensor.fromData(Seq(2,2), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        val x2 = Tensor.fromData(Seq(2), 4, 3)
        val d2 = Tensor.fromData(Seq(2), 0.2f, 0.3f)
        val start2 = TensorFR(new TensorF(x2, d2))

        // compute gradient and hessV
        val res: Tensor = gradHessV { () =>
          (start1 dot start2).sum()
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(34))
        Tensor.assertEqual(getGradient(start1).toCPU(), Tensor.fromData(Seq(2,2), 4,3,4,3))
        Tensor.assertEqual(getGradient(start2).toCPU(), Tensor.fromData(Seq(2), 4,6))
        Tensor.assertEqual(getHessV(start1).toCPU(), Tensor.fromData(Seq(2,2), 0.2f, 0.3f, 0.2f, 0.3f))
        Tensor.assertEqual(getHessV(start2).toCPU(), Tensor.fromData(Seq(2), 1.0f, 1.2f))
      }
    }
    g1.eval("a")
  }

  testGPU("mm_dot1") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-mm_dot1"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val x1 = Tensor.fromData(Seq(2,2), 1,2,3,4)
        val d1 = Tensor.fromData(Seq(2,2), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        val x2 = Tensor.fromData(Seq(2,2), 4,3,2,1)
        val d2 = Tensor.fromData(Seq(2,2), 0.2f, 0.3f, 0.4f, 0.5f)
        val start2 = TensorFR(new TensorF(x2, d2))

        // compute gradient and hessV
        val res: Tensor = gradHessV { () =>
          (start1 dot start2).sum()
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(46))
        Tensor.assertEqual(getGradient(start1).toCPU(), Tensor.fromData(Seq(2,2), 7,3,7,3))
        Tensor.assertEqual(getGradient(start2).toCPU(), Tensor.fromData(Seq(2,2), 4,4,6,6))
        Tensor.assertEqual(getHessV(start1).toCPU(), Tensor.fromData(Seq(2,2), 0.5f, 0.9f, 0.5f, 0.9f))
        Tensor.assertEqual(getHessV(start2).toCPU(), Tensor.fromData(Seq(2,2), 1.0f, 1.0f, 1.2f, 1.2f))
      }
    }
    g1.eval("a")
  }

  testGPU("tanh0") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-tanh0"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val x1 = Tensor.fromData(Seq(4), 1,2,3,4)
        val d1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        // compute gradient and hessV
        val res: Tensor = gradHessV { () =>
          start1.tanh.sum
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(3.720006f))
        Tensor.assertEqual(getGradient(start1).toCPU(), Tensor.fromData(Seq(4), 0.419974f, 0.070651f, 0.009866f, 0.001341f))
        Tensor.assertEqual(getHessV(start1).toCPU(), Tensor.fromData(Seq(4), -0.255880f, -0.068109f, -0.011781f, -0.001876f))
        // PyTorch equvilent code
        // v = torch.tensor([0.4, 0.5, 0.6, 0.7])
        // x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        // # f = 3 * x[0] ** 2 + 4 * x[0] * x[1] + x[1] **2
        // f = x.tanh().sum()
        // grad_f, = torch.autograd.grad(f, x, create_graph=True)
        // z = grad_f @ v
        // z.backward()
        // torch.set_printoptions(precision=6)
        // f.data
        // >> tensor(3.720006)
        // grad_f.data
        // >> tensor([0.419974, 0.070651, 0.009866, 0.001341])
        // x.grad
        // >> tensor([-0.255880, -0.068109, -0.011781, -0.001876])
      }
    }
    g1.eval("a")
  }

  testGPU("exp0") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-exp0"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val x1 = Tensor.fromData(Seq(4), 1,2,3,4)
        val d1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        // compute gradient and hessV
        val res: Tensor = gradHessV { () =>
          start1.exp.sum
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(84.791023f))
        Tensor.assertEqual(getGradient(start1).toCPU(), Tensor.fromData(Seq(4), 2.718282f,  7.389056f, 20.085537f, 54.598148f))
        Tensor.assertEqual(getHessV(start1).toCPU(), Tensor.fromData(Seq(4), 1.087313f,  3.694528f, 12.051323f, 38.218704f))
        // PyTorch equvilent code
        // v = torch.tensor([0.4, 0.5, 0.6, 0.7])
        // x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        // f = x.exp().sum()
        // grad_f, = torch.autograd.grad(f, x, create_graph=True)
        // z = grad_f @ v
        // z.backward()
        // torch.set_printoptions(precision=6)
        // f.data # tensor(84.791023)
        // grad_f.data # tensor([ 2.718282,  7.389056, 20.085537, 54.598148])
        // x.grad # tensor([ 1.087313,  3.694528, 12.051323, 38.218704])
      }
    }
    g1.eval("a")
  }

  testGPU("log0") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-log0"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val x1 = Tensor.fromData(Seq(4), 1,2,3,4)
        val d1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        // compute gradient and hessV
        val res: Tensor = gradHessV { () =>
          start1.log.sum
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(3.178054f))
        Tensor.assertEqual(getGradient(start1).toCPU(), Tensor.fromData(Seq(4),1.000000f, 0.500000f, 0.333333f, 0.250000f))
        Tensor.assertEqual(getHessV(start1).toCPU(), Tensor.fromData(Seq(4), -0.400000f, -0.125000f, -0.066667f, -0.043750f))
        // PyTorch equvilent code
        // v = torch.tensor([0.4, 0.5, 0.6, 0.7])
        // x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        // f = x.log().sum()
        // grad_f, = torch.autograd.grad(f, x, create_graph=True)
        // z = grad_f @ v
        // z.backward()
        // torch.set_printoptions(precision=6)
        // f.data # tensor(3.178054)
        // grad_f.data # tensor([1.000000, 0.500000, 0.333333, 0.250000])
        // x.grad # tensor([-0.400000, -0.125000, -0.066667, -0.043750])
      }
    }
    g1.eval("a")
  }

  testGPU("sqrt0") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-sqrt0"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val x1 = Tensor.fromData(Seq(4), 1,2,3,4)
        val d1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        // compute gradient and hessV
        val res: Tensor = gradHessV { () =>
          start1.sqrt.sum
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(6.146265f))
        Tensor.assertEqual(getGradient(start1).toCPU(), Tensor.fromData(Seq(4), 0.500000f, 0.353553f, 0.288675f, 0.250000f))
        Tensor.assertEqual(getHessV(start1).toCPU(), Tensor.fromData(Seq(4), -0.100000f, -0.044194f, -0.028868f, -0.021875f))
        // PyTorch equvilent code
        // v = torch.tensor([0.4, 0.5, 0.6, 0.7])
        // x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        // f = x.sqrt().sum()
        // grad_f, = torch.autograd.grad(f, x, create_graph=True)
        // z = grad_f @ v
        // z.backward()
        // torch.set_printoptions(precision=6)
        // f.data # tensor(6.146265)
        // grad_f.data # tensor([0.500000, 0.353553, 0.288675, 0.250000])
        // x.grad # tensor([-0.100000, -0.044194, -0.028868, -0.021875])
      }
    }
    g1.eval("a")
  }

  testGPU("square0") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-square0"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val x1 = Tensor.fromData(Seq(4), 1,2,3,4)
        val d1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        // compute gradient and hessV
        val res: Tensor = gradHessV { () =>
          start1.square.sum
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(30))
        Tensor.assertEqual(getGradient(start1).toCPU(), Tensor.fromData(Seq(4), 2, 4, 6, 8))
        Tensor.assertEqual(getHessV(start1).toCPU(), Tensor.fromData(Seq(4), 0.800000f, 1.000000f, 1.200000f, 1.400000f))
        // PyTorch equvilent code
        // v = torch.tensor([0.4, 0.5, 0.6, 0.7])
        // x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        // f = (x * x).sum()
        // grad_f, = torch.autograd.grad(f, x, create_graph=True)
        // z = grad_f @ v
        // z.backward()
        // torch.set_printoptions(precision=6)
        // f.data # tensor(30.)
        // grad_f.data # tensor([2., 4., 6., 8.])
        // x.grad # tensor([0.800000, 1.000000, 1.200000, 1.400000])
      }
    }
    g1.eval("a")
  }

  testGPU("relu0") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-relu0"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set input and vector for Hessen
        val x1 = Tensor.fromData(Seq(4), -0.6f, 0.7f, -0.3f, 0.2f)
        val d1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        // compute gradient and hessV
        val res = gradHessV{ () =>
          start1.relu(inPlace=false).sum
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(0.9f))
        Tensor.assertEqual(getGradient(start1).toCPU(), Tensor.fromData(Seq(4), 0.000000f, 1, 0.000000f, 1))
        Tensor.assertEqual(getHessV(start1).toCPU(), Tensor.fromData(Seq(4), 0.000000f, 0, 0.000000f, 0))
        // PyTorch equvilent code
        // start1 = torch.tensor([-0.6, 0.7, -0.3, 0.2], requires_grad=True)
        // out = start1.relu().sum()
        // grads = torch.autograd.grad(out, [start1], create_graph=True, retain_graph = True)
        // flatten = torch.cat([g.reshape(-1) for g in grads if g is not None])
        // v = torch.tensor([0.4, 0.5, 0.6, 0.7])
        // hvps = torch.autograd.grad([flatten @ v], [start1], allow_unused=True)
        // torch.set_printoptions(precision=6)
        // print(out.data) # tensor(0.900000)
        // print(grads[0].data) # tensor([0., 1., 0., 1.])
        // print(hvps[0].data)  # tensor([0., 0., 0., 0.])
      }
    }
    g1.eval("a")
  }

  testGPU("relu1") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-relu1"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set input and vector for Hessen
        val x1 = Tensor.fromData(Seq(4), -0.6f, 0.7f, -0.3f, 0.2f)
        val d1 = Tensor.fromData(Seq(4), -0.4f, -0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        val x2 = Tensor.fromData(Seq(4), 0.4f, 0.3f, 0.2f, 1)
        val d2 = Tensor.fromData(Seq(4), -0.2f, -0.3f, -0.4f, 0.5f)
        val start2 = TensorFR(new TensorF(x2, d2))

        // compute gradient and hessV
        val res = gradHessV{ () =>
          (start1 * start2).tanh().relu(inPlace=false).sum
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(0.404342f))
        Tensor.assertEqual(getGradient(start1).toCPU(), Tensor.fromData(Seq(4), 0.000000f, 0.287149f, 0.000000f, 0.961043f))
        Tensor.assertEqual(getGradient(start2).toCPU(), Tensor.fromData(Seq(4), -0.000000f, 0.670015f, -0.000000f, 0.192209f))
        Tensor.assertEqual(getHessV(start1).toCPU(), Tensor.fromData(Seq(4), 0.000000f, -0.244360f, 0.000000f, 0.177024f))
        Tensor.assertEqual(getHessV(start2).toCPU(), Tensor.fromData(Seq(4), 0.000000f, -0.378740f, 0.000000f, 0.612031f))
        // PyTorch equvilent code
        // start1 = torch.tensor([-0.6, 0.7, -0.3, 0.2], requires_grad=True)
        // start2 = torch.tensor([0.4, 0.3, 0.2, 1.0], requires_grad=True)
        // out = (start1 * start2).tanh().relu().sum()
        // grads = torch.autograd.grad(out, [start1, start2], create_graph=True, retain_graph = True)
        // flatten = torch.cat([g.reshape(-1) for g in grads if g is not None])
        // v = torch.tensor([-0.4, -0.5, 0.6, 0.7, -0.2, -0.3, -0.4, 0.5])
        // hvps = torch.autograd.grad([flatten @ v], [start1, start2], allow_unused=True)
        // torch.set_printoptions(precision=6)
        // print(out.data) # tensor(0.404342)
        // print(grads[0].data) # tensor([0.000000, 0.287149, 0.000000, 0.961043])
        // print(grads[1].data) # tensor([-0.000000, 0.670015, -0.000000, 0.192209])
        // print(hvps[0].data)  # tensor([ 0.000000, -0.244360,  0.000000,  0.177024])
        // print(hvps[1].data)  # tensor([-0.000000, -0.378740,  0.000000,  0.612031])
      }
    }
    g1.eval("a")
  }

  testGPU("hardTanh1") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-hardTanh1"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set input and vector for Hessen
        val x1 = Tensor.fromData(Seq(4), -6f, 7f, -3f, 2f)
        val d1 = Tensor.fromData(Seq(4), -0.4f, -0.5f, 0.6f, 0.7f)
        val start1 = TensorFR(new TensorF(x1, d1))

        val x2 = Tensor.fromData(Seq(4), 0.4f, 0.3f, 0.2f, 0.1f)
        val d2 = Tensor.fromData(Seq(4), -0.2f, -0.3f, -0.4f, 0.5f)
        val start2 = TensorFR(new TensorF(x2, d2))

        // compute gradient and hessV
        val res = gradHessV{ () =>
          val temp = (start1 * start2).square()
          temp.hardTanh(inPlace=false).sum
        }

        // correctness assertion
        backend = BackendCPU()
        generateRawComment("test result")
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(2.4f))
        generateRawComment("test gradient")
        Tensor.assertEqual(getGradient(start1).toCPU(), Tensor.fromData(Seq(4), -0.000000f,  0.000000f, -0.240000f,  0.040000f))
        Tensor.assertEqual(getGradient(start2).toCPU(), Tensor.fromData(Seq(4), 0.000000f, 0.000000f, 3.600000f, 0.800000f))
        Tensor.assertEqual(getHessV(start1).toCPU(), Tensor.fromData(Seq(4), 0.000000f, 0.000000f, 1.008000f, 0.414000f))
        Tensor.assertEqual(getHessV(start2).toCPU(), Tensor.fromData(Seq(4), 0.000000f,  0.000000f, -8.640000f,  4.560000f))
        // PyTorch equvilent code
        // start1 = torch.tensor([-6.0, 7.0, -3.0, 2.0], requires_grad=True)
        // start2 = torch.tensor([0.4, 0.3, 0.2, 0.1], requires_grad=True)
        // hardTanh = torch.nn.Hardtanh()
        // out = hardTanh((start1 * start2) * (start1 * start2)).sum()
        // grads = torch.autograd.grad(out, [start1, start2], create_graph=True, retain_graph = True)
        // flatten = torch.cat([g.reshape(-1) for g in grads if g is not None])
        // v = torch.tensor([-0.4, -0.5, 0.6, 0.7, -0.2, -0.3, -0.4, 0.5])
        // hvps = torch.autograd.grad([flatten @ v], [start1, start2], allow_unused=True)
        // torch.set_printoptions(precision=6)
        // print(out.data) # tensor(2.400000)
        // print(grads[0].data) # tensor([-0.000000,  0.000000, -0.240000,  0.040000])
        // print(grads[1].data) # tensor([0.000000, 0.000000, 3.600000, 0.800000])
        // print(hvps[0].data)  # tensor([0.000000, 0.000000, 1.008000, 0.414000])
        // print(hvps[1].data)  # tensor([0.000000,  0.000000, -8.640000,  4.560000])
      }
    }
    g1.eval("a")
  }

  testGPU("conv0") { // for kernel
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-conv0"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val input1 = Tensor.fromData(Seq(1,1,2,2), 1,2,3,4)
        val inputd1 = Tensor.fromData(Seq(1,1,2,2), 0,0,0,0)
        val input = TensorFR(new TensorF(input1, inputd1))

        val kernel1 = Tensor.fromData(Seq(1,1,2,2), 0.177578f, 0.153097f, -0.454294f, 0.442411f)
        val kerneld1 = Tensor.fromData(Seq(1,1,2,2), 0.4f, 0.5f, 0.6f, 0.7f)
        val kernel = TensorFR(new TensorF(kernel1, kerneld1))

        // compute gradient and hessV
        val res: Tensor = gradHessV { () =>
          input.conv2D_batch(kernel).tanh()
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(0.711658f))
        Tensor.assertEqual(getGradient(kernel).toCPU(),
          Tensor.fromData(Seq(1,1,2,2), 0.493542f, 0.987085f, 1.480627f, 1.974169f))
        Tensor.assertEqual(getHessV(kernel).toCPU(),
          Tensor.fromData(Seq(1,1,2,2), -4.214802f,  -8.429605f, -12.644407f, -16.859209f))
        // PyTorch equvilent code
        // torch.manual_seed(999)
        // conv = nn.Sequential(nn.Conv2d(1, 1, 2, bias=False), nn.Tanh())  # if just conv, it fails to work
        // input = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
        // out = conv(input).sum()
        // grads = torch.autograd.grad([out], conv.parameters(), create_graph=True)
        // flatten = torch.cat([g.reshape(-1) for g in grads if g is not None])
        // x = torch.tensor([0.4, 0.5, 0.6, 0.7])
        // hvps, = torch.autograd.grad([flatten @ x], conv.parameters(), allow_unused=True)
        // for i in conv.parameters():
        //   print(i) # tensor([[[[ 0.177578,  0.153097], [-0.454294,  0.442411]]]])
        //
        // print(out.data) # tensor(0.711658)
        // print(grads[0].data) # tensor([[[[0.493542, 0.987085], [1.480627, 1.974169]]]])
        // print(hvps.data) # tensor([[[[ -4.214802,  -8.429605], [-12.644407, -16.859209]]]])
      }
    }
    g1.eval("a")
  }

  testGPU("conv1") { // for input
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-conv1"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val input1 = Tensor.fromData(Seq(1,1,2,2), 1,2,3,4)
        val inputd1 = Tensor.fromData(Seq(1,1,2,2), 0.4f, 0.5f, 0.6f, 0.7f)
        val input = TensorFR(new TensorF(input1, inputd1))

        val kernel1 = Tensor.fromData(Seq(1,1,2,2), 0.177578f, 0.153097f, -0.454294f, 0.442411f)
        val kerneld1 = Tensor.fromData(Seq(1,1,2,2), 0,0,0,0)
        val kernel = TensorFR(new TensorF(kernel1, kerneld1))

        // compute gradient and hessV
        val res: Tensor = gradHessV { () =>
          input.conv2D_batch(kernel).tanh()
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(0.711658f))
        Tensor.assertEqual(getGradient(input).toCPU(),
          Tensor.fromData(Seq(1,1,2,2), 0.087642f, 0.075560f, -0.224213f, 0.218349f))
        Tensor.assertEqual(getHessV(input).toCPU(),
          Tensor.fromData(Seq(1,1,2,2), -0.023039f, -0.019863f, 0.058940f, -0.057398f))
        // PyTorch equvilent code
        // torch.manual_seed(999)
        // conv = nn.Sequential(nn.Conv2d(1, 1, 2, bias=False), nn.Tanh())  # if just conv, it fails to work
        // input = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
        // out = conv(input).sum()
        // grads = torch.autograd.grad([out], input, create_graph=True)
        // flatten = torch.cat([g.reshape(-1) for g in grads if g is not None])
        // x = torch.tensor([0.4, 0.5, 0.6, 0.7])
        // hvps, = torch.autograd.grad([flatten @ x], input, allow_unused=True)
        // for i in conv.parameters():
        //     print(i.data) # tensor([[[[ 0.177578,  0.153097], [-0.454294,  0.442411]]]])

        // print(out.data) # tensor(0.711658)
        // print(grads[0].data)  # tensor([[[[ 0.087642,  0.075560], [-0.224213,  0.218349]]]])
        // print(hvps.data) # tensor([[[[-0.023039, -0.019863], [ 0.058940, -0.057398]]]])
      }
    }
    g1.eval("a")
  }

  testGPU("conv2") { // for both kernel and input
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-conv2"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val input1 = Tensor.fromData(Seq(1,1,2,2), 1,2,3,4)
        val inputd1 = Tensor.fromData(Seq(1,1,2,2), 0.2f, 0.3f, 0.4f, 0.5f)
        val input = TensorFR(new TensorF(input1, inputd1))

        val kernel1 = Tensor.fromData(Seq(1,1,2,2), 0.177578f, 0.153097f, -0.454294f, 0.442411f)
        val kerneld1 = Tensor.fromData(Seq(1,1,2,2), 0.4f, 0.5f, 0.6f, 0.7f)
        val kernel = TensorFR(new TensorF(kernel1, kerneld1))

        // compute gradient and hessV
        val res: Tensor = gradHessV { () =>
          input.conv2D_batch(kernel).tanh()
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(0.711658f))
        Tensor.assertEqual(getGradient(input).toCPU(),
          Tensor.fromData(Seq(1,1,2,2), 0.087642f, 0.075560f, -0.224213f, 0.218349f))
        Tensor.assertEqual(getGradient(kernel).toCPU(),
          Tensor.fromData(Seq(1,1,2,2), 0.493542f, 0.987085f, 1.480627f, 1.974169f))
        Tensor.assertEqual(getHessV(input).toCPU(),
          Tensor.fromData(Seq(1,1,2,2), -0.566125f, -0.411508f, 2.249478f, -1.556781f))
        Tensor.assertEqual(getHessV(kernel).toCPU(),
          Tensor.fromData(Seq(1,1,2,2), -4.201046f, -8.451446f, -12.701845f, -16.952246f))
        // PyTorch equvilent code
        // torch.manual_seed(999)
        // conv = nn.Sequential(nn.Conv2d(1, 1, 2, bias=False), nn.Tanh())  # if just conv, it fails to work
        // input = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
        // out = conv(input).sum()
        // grads = torch.autograd.grad([out], [input] + list(conv.parameters()), create_graph=True, retain_graph = True)
        // flatten = torch.cat([g.reshape(-1) for g in grads if g is not None])
        // x = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.7])
        // hvps = torch.autograd.grad([flatten @ x], [input] + list(conv.parameters()), allow_unused=True)
        // for i in conv.parameters():
        //     print(i.data) # tensor([[[[ 0.177578,  0.153097], [-0.454294,  0.442411]]]])

        // print(out.data) # tensor(0.711658)
        // print(grads[0].data) # tensor([[[[ 0.087642,  0.075560], [-0.224213,  0.218349]]]])
        // print(grads[1].data) # tensor([[[[0.493542, 0.987085],[1.480627, 1.974169]]]])
        // print(hvps[0].data) # tensor([[[[-0.566125, -0.411508], [ 2.249478, -1.556781]]]])
        // print(hvps[1].data) # tensor([[[[ -4.201046,  -8.451446], [-12.701845, -16.952246]]]])
      }
    }
    g1.eval("a")
  }

  testGPU("vv_dot3") {
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-vv_dot3"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val input1 = Tensor.fromData(Seq(4), 1,2,3,4)
        val inputd1 = Tensor.fromData(Seq(4), 0.2f, 0.3f, 0.4f, 0.5f)
        val input = TensorFR(new TensorF(input1, inputd1))

        val kernel1 = Tensor.fromData(Seq(4), 0.177578f, 0.153097f, -0.454294f, 0.442411f)
        val kerneld1 = Tensor.fromData(Seq(4), 0.4f, 0.5f, 0.6f, 0.7f)
        val kernel = TensorFR(new TensorF(kernel1, kerneld1))

        val bias1 = Tensor.scalar(-0.007465f)
        val biasd1 = Tensor.scalar(0.8f)
        val bias = TensorFR(new TensorF(bias1, biasd1))

        val res: Tensor = gradHessV { () =>
          (input.dot(kernel) + bias).tanh()
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(0.707955f))
        Tensor.assertEqual(getGradient(input).toCPU(),
          Tensor.fromData(Seq(4), 0.088576f,  0.076365f, -0.226602f,  0.220675f))
        Tensor.assertEqual(getGradient(kernel).toCPU(),
          Tensor.fromData(Seq(4), 0.498800f, 0.997601f, 1.496401f, 1.995202f))
        Tensor.assertEqual(getGradient(bias).toCPU(), Tensor.scalar(0.498800f))
        Tensor.assertEqual(getHessV(input).toCPU(),
          Tensor.fromData(Seq(4), -0.668472f, -0.498930f, 2.519846f, -1.813325f))
        Tensor.assertEqual(getHessV(kernel).toCPU(),
          Tensor.fromData(Seq(4), -4.788190f, -9.626261f, -14.464331f, -19.302401f))
        Tensor.assertEqual(getHessV(bias).toCPU(), Tensor.scalar(-4.887950f))
      }
    }
    g1.eval("a")
  }

  testGPU("conv2.1") { // for both kernel and input (with bias)
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-conv2.1"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val input1 = Tensor.fromData(Seq(1,1,2,2), 1,2,3,4)
        val inputd1 = Tensor.fromData(Seq(1,1,2,2), 0.2f, 0.3f, 0.4f, 0.5f)
        val input = TensorFR(new TensorF(input1, inputd1))

        val kernel1 = Tensor.fromData(Seq(1,1,2,2), 0.177578f, 0.153097f, -0.454294f, 0.442411f)
        val kerneld1 = Tensor.fromData(Seq(1,1,2,2), 0.4f, 0.5f, 0.6f, 0.7f)
        val kernel = TensorFR(new TensorF(kernel1, kerneld1))

        val bias1 = Tensor.scalar(-0.007465f)
        val biasd1 = Tensor.scalar(0.8f)
        val bias = TensorFR(new TensorF(bias1, biasd1))

        // compute gradient and hessV
        val res: Tensor = gradHessV { () =>
          input.conv2D_batch(kernel, Some(bias)).tanh()
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(0.707955f))
        Tensor.assertEqual(getGradient(input).toCPU(),
          Tensor.fromData(Seq(1,1,2,2), 0.088576f,  0.076365f, -0.226602f,  0.220675f))
        Tensor.assertEqual(getGradient(kernel).toCPU(),
          Tensor.fromData(Seq(1,1,2,2), 0.498800f, 0.997601f, 1.496401f, 1.995202f))
        Tensor.assertEqual(getGradient(bias).toCPU(), Tensor.scalar(0.498800f))
        Tensor.assertEqual(getHessV(input).toCPU(),
          Tensor.fromData(Seq(1,1,2,2), -0.668472f, -0.498930f, 2.519846f, -1.813325f))
        Tensor.assertEqual(getHessV(kernel).toCPU(),
          Tensor.fromData(Seq(1,1,2,2), -4.788190f, -9.626261f, -14.464331f, -19.302401f))
        Tensor.assertEqual(getHessV(bias).toCPU(), Tensor.scalar(-4.887950f))
        // PyTorch equvilent code
        // torch.manual_seed(999)
        // torch.set_printoptions(precision=6)
        // conv = nn.Sequential(nn.Conv2d(1, 1, 2), nn.Tanh())
        // input = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
        // out = conv(input).sum()
        // grads = torch.autograd.grad([out], [input] + list(conv.parameters()), create_graph=True, retain_graph = True)
        // flatten = torch.cat([g.reshape(-1) for g in grads if g is not None])
        // x = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.7, 0.8])
        // hvps = torch.autograd.grad([flatten @ x], [input] + list(conv.parameters()), allow_unused=True)
        // for i in conv.parameters():
        //     print(i.data)
        // tensor([[[[ 0.177578,  0.153097],
        //   [-0.454294,  0.442411]]]])
        // tensor([-0.007465])

        // print(out.data) # tensor(0.707955)
        // print(grads[0].data) # tensor([[[[ 0.088576,  0.076365], [-0.226602,  0.220675]]]])
        // print(grads[1].data) # tensor([[[[0.498800, 0.997601], [1.496401, 1.995202]]]])
        // print(grads[2].data) # tensor([0.498800])
        // print(hvps[0].data); print(hvps[1].data); print(hvps[2].data)
        // tensor([[[[-0.668472, -0.498930],
        //           [ 2.519846, -1.813325]]]])
        // tensor([[[[ -4.788190,  -9.626261],
        //           [-14.464331, -19.302401]]]])
        // tensor([-4.887950])
      }
    }
    g1.eval("a")
  }


  testGPU("conv3") { // for both kernel and input, larger pane of input
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-conv3"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val input1 = Tensor.fromData(Seq(1,1,4,4),
          -0.981558f, -3.421909f, 1.491033f, 0.242209f,
          0.250686f, -1.227035f, -0.312874f, 0.167297f,
          0.627056f, -1.166596f, -0.786248f, 0.075932f,
          -0.765170f, 2.210146f, -0.064985f, 0.604543f)
        val inputd1 = Tensor.fromData(Seq(1,1,4,4),
          0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f,
          0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f)
        val input = TensorFR(new TensorF(input1, inputd1))

        val kernel1 = Tensor.fromData(Seq(1,1,2,2), 0.177578f, 0.153097f, -0.454294f, 0.442411f)
        val kerneld1 = Tensor.fromData(Seq(1,1,2,2), 0.2f, 0.3f, 0.4f, 0.5f)
        val kernel = TensorFR(new TensorF(kernel1, kerneld1))

        // compute gradient and hessV
        val res: Tensor = gradHessV { () =>
          input.conv2D_batch(kernel).tanh().sum()
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(-0.692684f))
        Tensor.assertEqual(getGradient(input).toCPU(),
          Tensor.fromData(Seq(1,1,4,4),
            0.041554f, 0.213125f,   0.290173f, 0.118385f,
           -0.025237f, -0.103820f,  0.398781f, 0.476878f,
           -0.158290f, -0.165651f,  0.247495f, 0.538271f,
           -0.125638f,  0.017048f, -0.339015f, 0.430015f))
        Tensor.assertEqual(getGradient(kernel).toCPU(),
          Tensor.fromData(Seq(1,1,2,2), -4.733940f, -0.279437f, -2.735202f, -0.532896f))
        Tensor.assertEqual(getHessV(input).toCPU(),
          Tensor.fromData(Seq(1,1,4,4),
            -0.070684f, 0.178874f,  0.395819f,  0.174026f,
            0.430365f,  0.456669f,  1.459121f,  0.486585f,
            0.320804f,  0.679685f,  1.045418f,  0.725112f,
            0.259240f, -0.012524f,  0.649163f,  0.438986f))
        Tensor.assertEqual(getHessV(kernel).toCPU(),
          Tensor.fromData(Seq(1,1,2,2), 2.002554f, 5.854410f, 3.204655f, 3.609499f))
        // PyTorch equvilent code
        // torch.manual_seed(999)
        // conv = nn.Sequential(nn.Conv2d(1, 1, 2, bias=False), nn.Tanh())
        // input = torch.randn(1, 1, 4, 4, requires_grad=True)
        // out = conv(input).sum()
        // grads = torch.autograd.grad([out], [input] + list(conv.parameters()), create_graph=True, retain_graph = True)
        // flatten = torch.cat([g.reshape(-1) for g in grads if g is not None])
        // x = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5])
        // hvps = torch.autograd.grad([flatten @ x], [input] + list(conv.parameters()), allow_unused=True)
        // for i in conv.parameters():
        //     print(i.data)

        // print(input.data)
        // tensor([[[[-0.981558, -3.421909,  1.491033,  0.242209],
        //   [ 0.250686, -1.227035, -0.312874,  0.167297],
        //   [ 0.627056, -1.166596, -0.786248,  0.075932],
        //   [-0.765170,  2.210146, -0.064985,  0.604543]]]])
        // print(out.data) # tensor(-0.692684)
        // print(grads[0].data)
        // tensor([[[[ 0.041554,  0.213125,  0.290173,  0.118385],
        //   [-0.025237, -0.103820,  0.398781,  0.476878],
        //   [-0.158290, -0.165651,  0.247495,  0.538271],
        //   [-0.125638,  0.017048, -0.339015,  0.430015]]]])
        // print(grads[1].data)
        // tensor([[[[-4.733940, -0.279437],
        //   [-2.735202, -0.532896]]]])
        // print(hvps[0].data)
        // tensor([[[[-0.070684,  0.178874,  0.395819,  0.174026],
        //   [ 0.430365,  0.456669,  1.459121,  0.486585],
        //   [ 0.320804,  0.679685,  1.045418,  0.725112],
        //   [ 0.259240, -0.012524,  0.649163,  0.438986]]]])
        // print(hvps[1].data)
        // tensor([[[[2.002554, 5.854410],
        //   [3.204655, 3.609499]]]])
      }
    }
    g1.eval("a")
  }

  testGPU("conv4") { // for both kernel and input, multiple channels
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-conv4"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val input1 = Tensor.fromData(Seq(1,3,2,2),
          -0.534282f,  0.555920f,  0.443033f,  1.373440f,
           0.640149f, -1.005894f,  0.238307f, -0.790597f,
           0.974699f,  0.196461f, -0.578166f, -0.726573f)
        val inputd1 = Tensor.fromData(Seq(1,3,2,2),
          0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f, 0.2f, 0.3f, 0.4f, 0.5f)
        val input = TensorFR(new TensorF(input1, inputd1))

        val kernel1 = Tensor.fromData(Seq(3,3,2,2),
          0.102525f,   0.088391f, -0.262287f,  0.255426f,
          -0.004310f,  0.287837f,  0.149217f, -0.270372f,
          -0.128825f,  0.265027f, -0.259937f, -0.185567f,
          -0.052237f,  0.030191f,  0.244085f, -0.260760f,
          0.173429f,  -0.097767f,  0.018818f, -0.169145f,
          -0.157547f,  0.200471f,  0.152380f,  0.128154f,
          0.143896f,  -0.068740f,  0.057309f, -0.186119f,
          -0.245372f,  0.089714f,  0.028990f, -0.268995f,
          0.154737f,  -0.019933f,  0.127669f, -0.077612f)
        val kerneld1 = Tensor.fromData(Seq(3,3,2,2),
          0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f, 0.2f, 0.3f, 0.4f, 0.5f,
          0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f, 0.2f, 0.3f, 0.4f, 0.5f,
          0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f, 0.2f, 0.3f, 0.4f, 0.5f)
        val kernel = TensorFR(new TensorF(kernel1, kerneld1))

        // compute gradient and hessV
        val res: Tensor = gradHessV { () =>
          input.conv2D_batch(kernel).tanh().sum()
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(-0.012916f))
        Tensor.assertEqual(getGradient(input).toCPU(),
          Tensor.fromData(Seq(1,3,2,2),
             0.172536f,  0.040444f, 0.067660f, -0.211290f,
            -0.065639f,  0.235815f, 0.173626f, -0.650645f,
            -0.118430f,  0.404227f, 0.046431f, -0.107054f))
        Tensor.assertEqual(getGradient(kernel).toCPU(),
          Tensor.fromData(Seq(3,3,2,2),
          -0.457980f,  0.476529f,  0.379763f,  1.177298f,
           0.548728f, -0.862241f,  0.204274f, -0.677691f,
           0.835501f,  0.168404f, -0.495597f, -0.622810f,
          -0.521948f,  0.543087f,  0.432806f,  1.341735f,
           0.625371f, -0.982673f,  0.232805f, -0.772347f,
           0.952198f,  0.191926f, -0.564819f, -0.709800f,
          -0.503793f,  0.524197f,  0.417751f,  1.295065f,
           0.603619f, -0.948493f,  0.224708f, -0.745482f,
           0.919078f,  0.185250f, -0.545173f, -0.685112f))
        Tensor.assertEqual(getHessV(input).toCPU(),
          Tensor.fromData(Seq(1,3,2,2),
            0.542069f, 0.848753f, 1.081665f, 1.436157f,
            1.144114f, 1.396216f, 1.671912f, 1.966781f,
            0.526062f, 0.851566f, 1.072706f, 1.385639f))
        Tensor.assertEqual(getHessV(kernel).toCPU(),
          Tensor.fromData(Seq(3,3,2,2),
           0.135136f, 0.294928f, 0.372977f, 0.521912f,
           0.386370f, 0.360249f, 0.530505f, 0.546315f,
           0.237663f, 0.270505f, 0.303592f, 0.379228f,
           0.201924f, 0.286268f, 0.385342f, 0.471643f,
           0.382929f, 0.500773f, 0.583231f, 0.693520f,
           0.183450f, 0.290669f, 0.397844f, 0.497353f,
           0.266352f, 0.201966f, 0.312690f, 0.271562f,
           0.284000f, 0.617877f, 0.531075f, 0.775127f,
           0.046719f, 0.254285f, 0.461327f, 0.577221f))
        // PyTorch equvilent code
        // torch.manual_seed(999)
        // conv = nn.Sequential(nn.Conv2d(3, 3, 2, bias=False), nn.Tanh())
        // input = torch.randn(1, 3, 2, 2, requires_grad=True)
        // out = conv(input).sum()
        // grads = torch.autograd.grad([out], [input] + list(conv.parameters()), create_graph=True, retain_graph = True)
        // flatten = torch.cat([g.reshape(-1) for g in grads if g is not None])
        // x = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5,
        // 0.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5,
        // 0.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5,
        // 0.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5])
        // hvps = torch.autograd.grad([flatten @ x], [input] + list(conv.parameters()), allow_unused=True)
        // for i in conv.parameters():
        //     print(i.data)
        // tensor([[[[ 0.102525,  0.088391],
        //   [-0.262287,  0.255426]],
        //  [[-0.004310,  0.287837],
        //   [ 0.149217, -0.270372]],
        //  [[-0.128825,  0.265027],
        //   [-0.259937, -0.185567]]],
        // [[[-0.052237,  0.030191],
        //   [ 0.244085, -0.260760]],
        //  [[ 0.173429, -0.097767],
        //   [ 0.018818, -0.169145]],
        //  [[-0.157547,  0.200471],
        //   [ 0.152380,  0.128154]]],
        // [[[ 0.143896, -0.068740],
        //   [ 0.057309, -0.186119]],
        //  [[-0.245372,  0.089714],
        //   [ 0.028990, -0.268995]],
        //  [[ 0.154737, -0.019933],
        //   [ 0.127669, -0.077612]]]])
        // print(input.data)
        // tensor([[[[-0.534282,  0.555920],
        //   [ 0.443033,  1.373440]],
        //  [[ 0.640149, -1.005894],
        //   [ 0.238307, -0.790597]],
        //  [[ 0.974699,  0.196461],
        //   [-0.578166, -0.726573]]]])
        // print(out.data) # tensor(-0.012916)
        // print(grads[0].data)
        // tensor([[[[ 0.172536,  0.040444],
        //   [ 0.067660, -0.211290]],
        //  [[-0.065639,  0.235815],
        //   [ 0.173626, -0.650645]],
        //  [[-0.118430,  0.404227],
        //   [ 0.046431, -0.107054]]]])
        // print(grads[1].data)
        // tensor([[[[-0.457980,  0.476529],
        //   [ 0.379763,  1.177298]],
        //  [[ 0.548728, -0.862241],
        //   [ 0.204274, -0.677691]],
        //  [[ 0.835501,  0.168404],
        //   [-0.495597, -0.622810]]],
        // [[[-0.521948,  0.543087],
        //   [ 0.432806,  1.341735]],
        //  [[ 0.625371, -0.982673],
        //   [ 0.232805, -0.772347]],
        //  [[ 0.952198,  0.191926],
        //   [-0.564819, -0.709800]]],
        // [[[-0.503793,  0.524197],
        //   [ 0.417751,  1.295065]],
        //  [[ 0.603619, -0.948493],
        //   [ 0.224708, -0.745482]],
        //  [[ 0.919078,  0.185250],
        //   [-0.545173, -0.685112]]]])
        // print(hvps[0].data)
        // tensor([[[[0.542069, 0.848753],
        //   [1.081665, 1.436157]],
        //  [[1.144114, 1.396216],
        //   [1.671912, 1.966781]],
        //  [[0.526062, 0.851566],
        //   [1.072706, 1.385639]]]])
        // print(hvps[1].data)
        // tensor([[[[0.135136, 0.294928],
        //   [0.372977, 0.521912]],
        //  [[0.386370, 0.360249],
        //   [0.530505, 0.546315]],
        //  [[0.237663, 0.270505],
        //   [0.303592, 0.379228]]],
        // [[[0.201924, 0.286268],
        //   [0.385342, 0.471643]],
        //  [[0.382929, 0.500773],
        //   [0.583231, 0.693520]],
        //  [[0.183450, 0.290669],
        //   [0.397844, 0.497353]]],
        // [[[0.266352, 0.201966],
        //   [0.312690, 0.271562]],
        //  [[0.284000, 0.617877],
        //   [0.531075, 0.775127]],
        //  [[0.046719, 0.254285],
        //   [0.461327, 0.577221]]]])
      }
    }
    g1.eval("a")
  }

  testGPU("conv5") { // for both kernel and input, multiple channels and batches
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-conv5"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val input1 = Tensor.fromData(Seq(2,3,2,2),
          0.914696f, -1.189630f, -0.750090f, -1.546543f,
          0.819371f, 0.611651f, 0.760206f, 1.478829f,
          0.212609f, -0.109771f, -0.134923f, 2.003842f,
          0.000396f, -0.688232f, 0.476999f, -0.416337f,
          -0.211574f, 0.527553f, 0.568272f, -0.678634f,
          0.821981f, 0.191211f, -0.119856f, -1.937328f)
        val inputd1 = Tensor.fromData(Seq(2,3,2,2),
          0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f, 0.2f, 0.3f, 0.4f, 0.5f,
          0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f, 0.2f, 0.3f, 0.4f, 0.5f)
        val input = TensorFR(new TensorF(input1, inputd1))

        val kernel1 = Tensor.fromData(Seq(3,3,2,2),
          0.102525f, 0.088391f, -0.262287f, 0.255426f,
          -0.004310f, 0.287837f, 0.149217f, -0.270372f,
          -0.128825f, 0.265027f, -0.259937f, -0.185567f,
          -0.052237f, 0.030191f, 0.244085f, -0.260760f,
          0.173429f, -0.097767f, 0.018818f, -0.169145f,
          -0.157547f, 0.200471f, 0.152380f, 0.128154f,
          0.143896f, -0.068740f, 0.057309f, -0.186119f,
          -0.245372f, 0.089714f, 0.028990f, -0.268995f,
          0.154737f, -0.019933f, 0.127669f, -0.077612f)
        val kerneld1 = Tensor.fromData(Seq(3,3,2,2),
          0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f, 0.2f, 0.3f, 0.4f, 0.5f,
          0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f, 0.2f, 0.3f, 0.4f, 0.5f,
          0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f, 0.2f, 0.3f, 0.4f, 0.5f)
        val kernel = TensorFR(new TensorF(kernel1, kerneld1))

        // compute gradient and hessV
        val res: Tensor = gradHessV { () =>
          input.conv2D_batch(kernel).tanh().sum()
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(0.276402f))
        Tensor.assertEqual(getGradient(input).toCPU(),
          Tensor.fromData(Seq(2,3,2,2),
             0.151112f, 0.018337f, 0.129597f, -0.273825f,
             -0.069504f, 0.169978f, 0.138963f, -0.591207f,
             -0.084882f, 0.340861f, 0.109350f, -0.065168f,
             0.122073f, 0.058400f, 0.063809f, -0.166714f,
             0.013531f, 0.193614f, 0.157941f, -0.555341f,
             -0.162885f, 0.400483f, 0.019421f, -0.072988f))
        Tensor.assertEqual(getGradient(kernel).toCPU(),
          Tensor.fromData(Seq(3,3,2,2),
          0.569146f, -1.298678f, -0.079115f, -1.299841f,
          0.337736f, 0.808769f, 0.934217f, 0.368560f,
          0.799706f, 0.087009f, -0.181234f, -0.327070f,
          0.891004f, -1.837317f, -0.259738f, -1.916583f,
          0.589061f, 1.116029f, 1.300845f, 0.770359f,
          1.017974f, 0.081767f, -0.249620f, 0.039729f,
          0.878842f, -1.574832f, -0.420968f, -1.746929f,
          0.654179f, 0.918772f, 1.087031f, 0.994333f,
          0.720356f, 0.014627f, -0.204857f, 0.708264f))
        Tensor.assertEqual(getHessV(input).toCPU(),
          Tensor.fromData(Seq(2,3,2,2),
            0.770650f, 0.829371f, 0.551087f, 1.671777f,
            0.765570f, 1.787284f, 1.746014f, 1.353123f,
            0.520100f, 1.002175f, 0.651053f, 0.899320f,
            0.703790f, 0.719610f, 0.786376f, 1.275637f,
            0.700524f, 1.521839f, 1.584556f, 1.291855f,
            0.572984f, 0.853624f, 0.873635f, 0.981845f))
        Tensor.assertEqual(getHessV(kernel).toCPU(),
          Tensor.fromData(Seq(3,3,2,2),
            1.549379f, -1.698212f, -0.124295f, -1.711723f,
            1.554727f, 1.934100f, 2.311296f, 2.564720f,
            1.161422f, 0.413895f, 0.302620f, 2.112442f,
            -0.147936f, 1.438166f, 1.124345f, 1.982459f,
            0.345936f, 0.505732f, 0.605323f, 0.645138f,
            0.089916f, 0.611782f, 0.889502f, 0.213686f,
            0.871806f, -0.888163f, 0.627982f, -0.531921f,
            0.933283f, 1.658551f, 1.945461f, 1.371656f,
            1.216165f, 0.589146f, 0.441458f, 0.193068f))
        // PyTorch equvilent code
        // torch.manual_seed(999)
        // conv = nn.Sequential(nn.Conv2d(3, 3, 2, bias=False), nn.Tanh())
        // input = torch.randn(2, 3, 2, 2, requires_grad=True)
        // out = conv(input).sum()
        // grads = torch.autograd.grad([out], [input] + list(conv.parameters()), create_graph=True, retain_graph = True)
        // flatten = torch.cat([g.reshape(-1) for g in grads if g is not None])
        // x = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5,
        // 0.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5,
        // 0.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5,
        // 0.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5,
        // 0.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5])
        // hvps = torch.autograd.grad([flatten @ x], [input] + list(conv.parameters()), allow_unused=True)
        // def better_print(a):
        //     for (x, y) in enumerate(a.view(a.numel()).tolist()):
        //         print('%.6f'%y, end='f, \n' if (x+1) %4 == 0 else 'f, ')
        //     print()

        // for i in conv.parameters():
        //     better_print(i.data)

        // better_print(input.data)
        // better_print(out.data)
        // better_print(grads[0].data)
        // better_print(grads[1].data)
        // better_print(hvps[0].data)
        // better_print(hvps[1].data)
      }
    }
    g1.eval("a")
  }

  testGPU("conv6") { // for both kernel and input, multiple channels and batches and bias
    val g1 = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
      override val fileName = "secOrder-gpu-conv6"
      def snippet(a: Rep[String]): Rep[Unit] = {
        // set inputs and vectors for Hessian
        val input1 = Tensor.fromData(Seq(2,3,2,2),
          -1.546543f, 0.819371f, 0.611651f, 0.760206f,
          1.478829f, 1.964689f, 0.941443f, 0.388325f,
          2.003842f, 0.000396f, -0.688232f, 0.476999f,
          -0.416337f, -1.209718f, 0.251931f, -0.564288f,
          -0.678634f, 0.821981f, 0.191211f, -0.119856f,
          -1.937328f, 1.643681f, 0.775187f, 0.598673f)
        val inputd1 = Tensor.fromData(Seq(2,3,2,2),
          0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f, 0.2f, 0.3f, 0.4f, 0.5f,
          0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f, 0.2f, 0.3f, 0.4f, 0.5f)
        val input = TensorFR(new TensorF(input1, inputd1))

        val kernel1 = Tensor.fromData(Seq(3,3,2,2),
          0.102525f, 0.088391f, -0.262287f, 0.255426f,
          -0.004310f, 0.287837f, 0.149217f, -0.270372f,
          -0.128825f, 0.265027f, -0.259937f, -0.185567f,
          -0.052237f, 0.030191f, 0.244085f, -0.260760f,
          0.173429f, -0.097767f, 0.018818f, -0.169145f,
          -0.157547f, 0.200471f, 0.152380f, 0.128154f,
          0.143896f, -0.068740f, 0.057309f, -0.186119f,
          -0.245372f, 0.089714f, 0.028990f, -0.268995f,
          0.154737f, -0.019933f, 0.127669f, -0.077612f)
        val kerneld1 = Tensor.fromData(Seq(3,3,2,2),
          0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f, 0.2f, 0.3f, 0.4f, 0.5f,
          0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f, 0.2f, 0.3f, 0.4f, 0.5f,
          0.2f, 0.3f, 0.4f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f, 0.2f, 0.3f, 0.4f, 0.5f)
        val kernel = TensorFR(new TensorF(kernel1, kerneld1))

        val bias1 = Tensor.fromData(Seq(3), 0.099004f, 0.061055f, -0.125780f)
        val biasd1 = Tensor.fromData(Seq(3), 0.1f, 0.2f, 0.3f)
        val bias = TensorFR(new TensorF(bias1, biasd1))

        // compute gradient and hessV
        val res: Tensor = gradHessV { () =>
          input.conv2D_batch(kernel, Some(bias)).tanh().sum()
        }

        // correctness assertion
        backend = BackendCPU()
        Tensor.assertEqual(res.toCPU(), Tensor.scalar(0.816457f))
        Tensor.assertEqual(getGradient(input).toCPU(),
          Tensor.fromData(Seq(2,3,2,2),
          0.136432f, 0.050447f, 0.062079f, -0.176134f,
          -0.015090f, 0.203413f, 0.158936f, -0.572141f,
          -0.142247f, 0.389753f, 0.027388f, -0.083439f,
          0.205704f, 0.020680f, -0.047919f, -0.094678f,
          -0.165767f, 0.286762f, 0.164489f, -0.579253f,
          -0.030075f, 0.300888f, -0.019579f, -0.173400f))
        Tensor.assertEqual(getGradient(kernel).toCPU(),
          Tensor.fromData(Seq(3,3,2,2),
          -1.598537f, -0.365414f, 0.706187f, 0.134005f,
          0.615891f, 2.279156f, 0.920347f, 0.211182f,
          -0.028123f, 1.393516f, 0.102725f, 0.891633f,
          -1.670365f, 0.198778f, 0.702491f, 0.452167f,
          1.080655f, 2.262594f, 0.986963f, 0.311761f,
          0.976029f, 0.788943f, -0.282547f, 0.740800f,
          -1.528917f, -0.619946f, 0.691932f, -0.017226f,
          0.385472f, 2.235308f, 0.868497f, 0.159554f,
          -0.495217f, 1.643599f, 0.279860f, 0.941720f))
        Tensor.assertEqual(getGradient(bias).toCPU(), Tensor.fromData(Seq(3), 1.653047f, 1.430667f, 1.719234f))
        Tensor.assertEqual(getHessV(input).toCPU(),
          Tensor.fromData(Seq(2,3,2,2),
            0.539991f, 0.396836f, 2.139926f, -0.252682f,
            0.628161f, 0.626031f, 1.227581f, 1.438508f,
            0.969371f, 0.351513f, 2.177961f, 1.679266f,
            0.459648f, 0.669387f, 0.904206f, 1.198558f,
            0.875149f, 1.133637f, 1.357323f, 1.747462f,
            0.545399f, 0.572863f, 0.934196f, 1.160932f))
        Tensor.assertEqual(getHessV(kernel).toCPU(),
          Tensor.fromData(Seq(3,3,2,2),
            4.162665f, -1.224702f, -0.873236f, -0.890667f,
            -2.769955f, -4.105068f, -1.327425f, 0.243511f,
            -4.098367f, 0.139125f, 2.158610f, -0.457197f,
            -1.898151f, 2.072245f, 1.406501f, 2.050189f,
            3.026625f, 3.390650f, 2.209198f, 1.626139f,
            3.957515f, -0.128473f, -0.725986f, 1.229307f,
            -3.644490f, 2.642019f, 2.264258f, 2.827186f,
            4.511339f, 5.923594f, 3.459974f, 2.206976f,
            5.534351f, 0.501626f, -1.096032f, 2.085396f))
        Tensor.assertEqual(getHessV(bias).toCPU(), Tensor.fromData(Seq(3), -2.636025f, 1.164155f, 2.572128f))
        // PyTorch equvilent code
        // torch.manual_seed(999)
        // conv = nn.Sequential(nn.Conv2d(3, 3, 2), nn.Tanh())
        // input = torch.randn(2, 3, 2, 2, requires_grad=True)
        // out = conv(input).sum()
        // grads = torch.autograd.grad([out], [input] + list(conv.parameters()), create_graph=True, retain_graph = True)
        // flatten = torch.cat([g.reshape(-1) for g in grads if g is not None])
        // x = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5,
        // 0.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5,
        // 0.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5,
        // 0.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5,
        // 0.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3])
        // hvps = torch.autograd.grad([flatten @ x], [input] + list(conv.parameters()), allow_unused=True)
        // def better_print(a):
        //     for (x, y) in enumerate(a.view(a.numel()).tolist()):
        //         print('%.6f'%y, end='f, \n' if (x+1) %4 == 0 else 'f, ')
        //     print()

        // for i in conv.parameters():
        //     better_print(i.data)

        // better_print(input.data)
        // better_print(out.data)
        // better_print(grads[0].data)
        // better_print(grads[1].data)
        // better_print(grads[2].data)
        // better_print(hvps[0].data)
        // better_print(hvps[1].data)
        // better_print(hvps[2].data)
      }
    }
    g1.eval("a")
  }
}

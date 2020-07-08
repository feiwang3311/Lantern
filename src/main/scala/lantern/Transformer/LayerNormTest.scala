package lantern
package Transformer

import lms.core.stub._
import lms.macros.SourceContext
import lms.core.virtualize

import scala.sys.process._
import java.io.PrintWriter
import java.io.File

object LayerNormTest {
  val driver = new LanternDriverCudnn[String, Unit] with ScannerOpsExp with TimerOpsExp {
//    override def snippet(x: Rep[String]): Rep[Unit] = {
//      val model = LayerNorm(10)
//
//       def lossFun(input: TensorR) = { (batchIndex:TensorR) =>
//           val res = model(input)
//           res.sum()
//       }
//
//      val in = TensorR(Tensor.rand(10,10,10,10))
//      val loss = gradR_loss(lossFun(in))(Tensor.zeros(4)).toCPU()
//      printf("%f\n", loss.data(0))
//    }

    override def snippet(x: Rep[String]): Rep[Unit] = {
      val m1 = Tensor.fromData(Seq(2, 2, 3), 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6)
      val m2 = Tensor.fromData(Seq(2, 3, 2), 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4)
      val result = m1.bmm(m2)
      val mm1 = TensorR(m1)
      val mm2 = TensorR(m2)

      val loss = gradR_loss(dummy => (mm1 bmm mm2).sum())(Tensor.zeros(1))
      loss.toCPU().print()
      mm1.d.toCPU().print()
      mm2.d.toCPU().print()

//      backend = BackendCPU()
//      result.toCPU().print()
//      mm1.d.print()
//      mm2.d.print()
//      val expected = Tensor.fromData(Seq(2, 2, 2), 19, 19, 46, 46, 19, 19, 46, 46)
//      val expected1 = Tensor.fromData(Seq(2, 2, 3), 5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7)
//      val expected2 = Tensor.fromData(Seq(2, 3, 2), 5, 5, 7, 7, 9, 9, 5, 5, 7, 7, 9, 9)
//      Tensor.assertEqual(result.toCPU(), expected)
//      Tensor.assertEqual(mm1.d.toCPU(), expected1)
//      Tensor.assertEqual(mm2.d.toCPU(), expected2)
    }
  }

  def main(args :Array[String]) = {
    val code_file = new PrintWriter(new File("src/out/Transformers/Lantern/layer_norm_test.cu"))
    code_file.println(driver.code)
    code_file.flush()
  }
}

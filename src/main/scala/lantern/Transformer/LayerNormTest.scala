package lantern
package Transformer

import lms.core.stub._
import lms.thirdparty.{ScannerOps}
import lms.macros.SourceContext
import lms.core.virtualize

import scala.sys.process._
import java.io.PrintWriter
import java.io.File

object LayerNormTest {
  val driver = new LanternDriverCudnn[String, Unit] with ScannerOps with TimerOpsExp {
    override def snippet(x: Rep[String]): Rep[Unit] = {
      val model = LayerNorm(10)

       def lossFun(input: TensorR) = { (batchIndex:TensorR) =>
           val res = model(input)
           res.sum()
       }

      val in = TensorR(Tensor.rand(10,10,10,10))
      val loss = gradR_loss(lossFun(in))(Tensor.zeros(4)).toCPU()
      printf("%f\n", loss.data(0))
    }
  }

  def main(args :Array[String]) = {
    val code_file = new PrintWriter(new File("src/out/Transformers/Lantern/layer_norm_test.cu"))
    code_file.println(driver.code)
    code_file.flush()
  }
}

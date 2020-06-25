package lantern

import lms.core.stub._
import lms.macros.SourceContext
import lms.core.virtualize

import scala.sys.process._
import java.io.PrintWriter
import java.io.File

object LayerNormTest {
  val driver = new LanternDriverCudnn[String, Unit] with ScannerOpsExp with TimerOpsExp {
    override def snippet(x: Rep[String]): Rep[Unit] = {

      case class LayerNorm(dim_size: Int, epsilon: Float = 0.00005, featureDim: Int = 3, name: String = "Layer Norm") extends Module {
        // performs layer norm on the last dimension
        val weights = TensorR(Tensor.ones(dim_size))
        val bias = TensorR(Tensor.zeros(dim_size))

        def apply(input: TensorR) = {
          val mean = input.sum(featureDim) / dim_size
          val mean_squared = mean * mean
          val squared = input * input
          val squared_mean = squared.sum(featureDim) / dim_size

          val variance = (squared_mean - mean_squared + epsilon).sqrt()
          val normalized = (input - mean) / variance

          normalized * weights + bias
        }
      }

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

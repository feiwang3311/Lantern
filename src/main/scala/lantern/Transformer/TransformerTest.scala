package lantern
package Transformer

import lms.core.stub._
import lms.macros.SourceContext
import lms.core.virtualize

import scala.sys.process._

import java.io.PrintWriter;
import java.io.File;

object TransformerTest {

  val driver = new LanternDriverCudnn[String, Unit] with ScannerOpsExp with TimerOpsExp {
    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {
      val embedDim = 500
      val seqLen = 50
      val batchSize = 500
      val beamSize = 1
      val nheads = 10
      val numBlocks = 4
      val dimFeedForward = 200
      val dropOut = 0.1f

      val model = Transformer(embedDim, seqLen, nheads, 4, 4, dimFeedForward, dropOut, seqLen, batchSize, beamSize)

      val src = TensorR(Tensor.rand(Seq(seqLen, batchSize, 1, embedDim): _*))
      val tgt = TensorR(Tensor.rand(Seq(seqLen, batchSize, 1, embedDim): _*))

      def lossFun(src: TensorR, tgt: TensorR) = { (batchIndex: TensorR) =>
        val res = model(src, tgt)
        res.sum()
      }

      val opt = SGD(model, learning_rate = 0.0005f, gradClip = 1.0f)

      val num_iter = 5
      for (i <- 0 until num_iter: Rep[Range]) {
        val trainTimer = Timer2()
        trainTimer.startTimer

        val loss = gradR_loss(lossFun(src, tgt))(Tensor.zeros(4))
        opt.step()
        val delta = trainTimer.getElapsedTime
        printf("Training iter done in %ldms \\n", delta / 1000L)

        printf("loss = %f\n", loss.toCPU().data(0))
      }


    }
  }

  def main(args: Array[String]) = {
    val code_file = new PrintWriter(new File("src/out/Transformers/Lantern/transformer.cu"))
    code_file.println(driver.code)
    code_file.flush()
  }
}
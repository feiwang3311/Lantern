package lantern
package Transformer

import lms.core.stub._
import lms.thirdparty.{ScannerOps}
import lms.macros.SourceContext
import lms.core.virtualize

import scala.sys.process._

import java.io.PrintWriter;
import java.io.File;

object TransformerTest {

  val driver = new LanternDriverCudnn[String, Unit] with ScannerOps with TimerOpsExp {
    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {
      val embedDim = 256
      val seqLen = 32
      val batchSize = 16
      val nheads = 8
      val numEncoderLayers = 6
      val numDecoderLayers = 6
      val dimFeedForward = 512
      val dropOut = 0.1f

      val model = Transformer(embedDim, seqLen, nheads, numEncoderLayers, numDecoderLayers, dimFeedForward, dropOut)

      val src = TensorR(Tensor.rand(Seq(seqLen, batchSize, embedDim): _*))
      val tgt = TensorR(Tensor.rand(Seq(seqLen, batchSize, embedDim): _*))

      def lossFun(src: TensorR, tgt: TensorR) = { (batchIndex: TensorR) =>
        val res = model(src, tgt)
        res.sum()
      }

      val opt = SGD(model, learning_rate = 0.000000005f, gradClip = 0.001f)

      val num_iter = 1
      val mini_batch_count = 100
      for (i <- 0 until num_iter: Rep[Range]) {
        val trainTimer = Timer2()
        trainTimer.startTimer
        // mimic multiple mini batches
        var totalLoss = var_new[Float](0)

        for(j <- 0 until mini_batch_count: Rep[Range]) {
          val addr = getMallocAddr()
          val addrCuda = getCudaMallocAddr()

          val loss = gradR_loss(lossFun(src, tgt))(Tensor.zeros(4))
          totalLoss += loss.toCPU().data(0)
          opt.step()
          resetMallocAddr(addr)
          resetCudaMallocAddr(addrCuda)
//          printf("Mini batch complete = %f\n", loss.toCPU().data(0))
        }

        val delta = trainTimer.getElapsedTime
        printf("Training iter done in %ldms \\n", delta / 1000L)
        printf("loss = %f\n", totalLoss)
      }


    }
  }

  def main(args: Array[String]): Unit = {
    import sys.process._
    val code_file = new PrintWriter(new File("src/out/Transformers/Lantern/transformer.cu"))
    code_file.println(driver.code)
    code_file.flush()

    val logger = ProcessLogger(
      (o: String) => println("out " + o),
      (e: String) => println("err " + e))

    "nvcc src/out/Transformers/Lantern/transformer.cu -o  src/out/Transformers/Lantern/transformer -lcuda -lcublas -lcudnn" ! logger;
//    "./src/out/Transformers/Lantern/transformer q" ! logger;
  }
}
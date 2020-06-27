package lantern
package Transformer

import lms.core.stub._
import lms.macros.SourceContext
import lms.core.virtualize

import scala.sys.process._
import java.io.PrintWriter
import java.io.File

object MultiheadAttentionTest {
  val driver = new LanternDriverCudnn[String, Unit] with ScannerOps with TimerOpsExp {
    override def snippet(x: Rep[String]): Rep[Unit] = {
      // model
      // requirements: qsize = ksize and vsize * numHeads = embedDim
      val qsize = 500
      val ksize = 50
      val vsize = 50
      val embedDim = 500
      val numHeads = 5
      val batchSize = 10
      val beamSize = 1
      val seqLen = 500 // both klen and qlen
      val dropOut = 0.1f


      case class Model(val name: String = "test_model") extends Module {
        val mha = MultiheadAttention(embedDim, numHeads, ksize, vsize, bias = true, dropOut)
        val linear = Linear1D(inSize = embedDim * seqLen, outSize = 1)

        def apply(q: TensorR, k: TensorR, v: TensorR) = {
          val step1 = mha(q, k, v)
          linear(step1.permute(1, 2, 0, 3).resize(-1, q.x.shape(0) * q.x.shape(3)))
        }
      }

      val model = Model()

      val q = TensorR(Tensor.rand(Seq(seqLen, batchSize, beamSize, qsize): _*)).toGPU()
      val k = TensorR(Tensor.rand(Seq(seqLen, batchSize, beamSize, ksize): _*)).toGPU()
      val v = TensorR(Tensor.rand(Seq(seqLen, batchSize, beamSize, vsize): _*)).toGPU()

      val opt = SGD(model, learning_rate = 0.0005f, gradClip = 1000.0f)


      def lossFun(query: TensorR, key: TensorR, value: TensorR) = { (batchIndex: TensorR) =>
        //     trainTimer.startTimer
        val res = model(query, key, value)
        res.sum()
      }


      val num_iter = 5
      for (i <- 0 until num_iter: Rep[Range]) {
        val trainTimer = Timer2()
        trainTimer.startTimer
        val loss = gradR_loss(lossFun(q, k, v))(Tensor.zeros(4))
        opt.step()
        val delta = trainTimer.getElapsedTime
        printf("Training iter done in %ldms \\n", delta / 1000L)
        printf("loss = %f\n", loss.toCPU().data(0))
      }
    }
  }

  def main(args: Array[String]) = {
    val code_file = new PrintWriter(new File("src/out/Transformers/Lantern/mha_test.cu"))
    code_file.println(driver.code)
    code_file.flush()
  }
}

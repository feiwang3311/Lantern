package lantern
package Transformer

import lms.core.stub._
import lms.thirdparty.{ScannerOps}
import lms.macros.SourceContext
import lms.core.virtualize

import scala.sys.process._
import java.io.PrintWriter
import java.io.File

object MultiheadAttentionTest {
  val driver = new LanternDriverCudnn[String, Unit] with TimerOpsExp {
    override def snippet(x: Rep[String]): Rep[Unit] = {
      // model
      // requirements: qsize = ksize and vsize * numHeads = embedDim
      val qsize = 512
      val ksize = 512
      val vsize = 515
      val embedDim = 512
      val numHeads = 8
      val batchSize = 2048
      val beamSize = 1
      val seqLen = 50 // both klen and qlen
      val dropOut = 0.1f

      case class Model(val name: String = "test_model") extends Module {
        val mha = MultiheadAttention(embedDim, numHeads, ksize, vsize, bias = true, dropOut, residualConnection = true,
          seqLen, seqLen, batchSize, beamSize)

        def apply(q: TensorR, k: TensorR, v: TensorR) = {
          val step1 = mha(q, k, v)
          step1
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


      // let's mimic mini batches
      val num_iter = 2
      val mini_batch_count = 1

      for (i <- 0 until num_iter: Rep[Range]) {
        val trainTimer = Timer2()
        var lossTotal = var_new[Float](0.0f)
        val addr = getMallocAddr()
        val addrCuda = getCudaMallocAddr()
        trainTimer.startTimer
        for (j <- 0 until mini_batch_count: Rep[Range]) {
          val loss = gradR_loss(lossFun(q, k, v))(Tensor.zeros(4))
          lossTotal += loss.toCPU().data(0)
          opt.step()
          resetMallocAddr(addr)
          resetCudaMallocAddr(addrCuda)
        }
        unchecked[Unit]("cudaDeviceSynchronize()")
        val delta = trainTimer.getElapsedTime
        printf("Training iter done in %ldms \\n", delta / 1000L)
        printf("loss = %f\n", lossTotal)
      }
    }
  }

  def main(args: Array[String]): Unit = {
    val code_file = new PrintWriter(new File("src/out/Transformers/Lantern/mha_test.cu"))
    code_file.println(driver.code)
    code_file.flush()

    "nvcc src/out/Transformers/Lantern/mha_test.cu -o  src/out/Transformers/Lantern/mha_test -lcuda -lcublas -lcudnn" !;
  }
}

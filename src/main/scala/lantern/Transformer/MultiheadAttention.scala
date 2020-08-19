package lantern
package Transformer

import lms.core.stub._
import lms.macros.SourceContext
import lms.core.virtualize

import scala.sys.process._
import java.io.PrintWriter
import java.io.File

object MultiheadAttention {
  val driver = new LanternDriverCudnn[String, Unit] with TimerOpsExp {
    override def snippet(x: Rep[String]): Rep[Unit] = {
      // model
      // requirements: qsize = ksize and vsize * numHeads = embedDim
      val qsize = 256
      val ksize = 256
      val vsize = 256
      val embedDim = 256
      val numHeads = 8
      val batchSize = 256
      val seqLen = 32 // both klen and qlen
      val dropOut = 0.1f

      val attnMask = Array(((0 until seqLen * seqLen :Range) map (_ => 1)) :_*).toGPU(seqLen * seqLen)
      val model = MultiheadAttention_v2(embedDim, numHeads, dropOut, bias = false, 0, Some(qsize), Some(ksize), Some(vsize))

      val q = TensorR(Tensor.rand(Seq(seqLen, batchSize, qsize): _*))
      val k = TensorR(Tensor.rand(Seq(seqLen, batchSize, ksize): _*))
      val v = TensorR(Tensor.rand(Seq(seqLen, batchSize, vsize): _*))

      val opt = SGD(model, learning_rate = 0.0005f) // SGD grad clip doesn't work
//      val opt = Adagrad(model, learning_rate = 0.0005f, gradClip = 1.0f)

      def lossFun(query: TensorR, key: TensorR, value: TensorR) = { (dummy: TensorR) =>
        model(query, key, value, None).sum()
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
//          val temp  = loss.toCPU().data(0)
          lossTotal += loss.toCPU().data(0)
//          printf("mini batch loss = %f\n", temp)
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
    val code_file = new PrintWriter(new File("src/out/Transformers/Lantern/mha_test2.cu"))
    code_file.println(driver.code)
    code_file.flush()

    "nvcc src/out/Transformers/Lantern/mha_test2.cu -o src/out/Transformers/Lantern/mha_test2 -Isrc/main/cpp/headers/ -I../lms-clean/src/main/scala/lms/thirdparty/thirdpartyAdaptor/ -lcuda -lcublas -lcudnn" !;
  }
}

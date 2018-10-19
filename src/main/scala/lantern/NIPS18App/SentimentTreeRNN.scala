package lantern
package NIPS18App

import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

import scala.collection.mutable.ArrayBuffer
import scala.math._

import java.io.PrintWriter;
import java.io.File;

object SentimentTreeRNN {

  val root_dir = "src/out/ICFP18evaluation/"
  val file_dir = "evaluationTreeLSTM/Lantern/LanternRNN.cpp"
  val root_dir2 = "src/out/NIPS18evaluation/"

  val sentimental_lstm = new LanternDriverC[String, Unit] {

    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {

      val startTime = get_time()

      // read in the data for word embedding
      val word_embedding_size   = 300

      val readingSlot1 = NewArray[Int](1) // this is a slot of memory used for reading from file
      val fp = openf("small_glove.txt", "r")
      getInt(fp, readingSlot1, 0) // read the first number in the file, which is "How many rows"
      val word_embedding_length = readingSlot1(0)

      val word_embedding_data = NewArray[Array[Float]](word_embedding_length)

      for (i <- (0 until word_embedding_length): Rep[Range]) {
        word_embedding_data(i) = NewArray[Float](word_embedding_size)
        for (j <- (0 until word_embedding_size): Rep[Range]) getFloat(fp, word_embedding_data(i), j)
      }
      closef(fp)

      // read in the data for trees
      val readingSlot2 = NewArray[Int](1) // need a new readingSlot, other wise have error
      val fp1 = openf("array_tree.txt", "r")
      getInt(fp1, readingSlot2, 0)
      val tree_number = readingSlot2(0)
      val tree_data = NewArray[Array[Int]](tree_number * 4) // each tree data has 4 lines (score, word, lch, rch)

      val readingSlot3 = NewArray[Int](1) // yet another readingSlot, not sure if this one can be reused
      for (i <- (0 until tree_number): Rep[Range]) {
        getInt(fp1, readingSlot3, 0)
        for (j <- (0 until 4): Rep[Range]) {
          tree_data(i * 4 + j) = NewArray[Int](readingSlot3(0))
          for (k <- (0 until readingSlot3(0)): Rep[Range]) getInt(fp1, tree_data(i * 4 + j), k)
        }
      }
      closef(fp1)

      // set up hyperparameters and parameters
      val hidden_size = 150
      val output_size = 5
      val learning_rate = 0.05f

      val Wu = Tensor.randinit(hidden_size, word_embedding_size, 0.01f)  // from word embedding to hidden vector, cell state
      val bu = Tensor.zeros(hidden_size)                                // bias word embedding to hidden vector, cell state
      val U0u  = Tensor.randinit(hidden_size, hidden_size, 0.01f) // left child, cell state
      val U1u  = Tensor.randinit(hidden_size, hidden_size, 0.01f) // right child, cell state
      val bbu  = Tensor.zeros(hidden_size)                       // bias, cell state
      // parameters for softmax
      val Why = Tensor.randinit(output_size, hidden_size, 0.01f)         // from hidden vector to output
      val by  = Tensor.zeros(output_size)                               // bias hidden vector to output

      val tWu = TensorR(Wu)
      val tbu = TensorR(bu)
      val tU0u = TensorR(U0u)
      val tU1u = TensorR(U1u)
      val tbbu = TensorR(bbu)
      val tWhy = TensorR(Why)
      val tby  = TensorR(by)

      def lossFun(scores: Rep[Array[Int]], words: Rep[Array[Int]], lchs: Rep[Array[Int]], rchs: Rep[Array[Int]]) = { (dummy: TensorR) =>

        val inBuffer = ArrayBuffer(TensorR(Tensor.zeros(1)), TensorR(Tensor.zeros(hidden_size)))
        val outBuffer = LOOPTM(0)(inBuffer)(lchs, rchs) { (l: ArrayBuffer[TensorR], r: ArrayBuffer[TensorR], i: Rep[Int]) =>

          IFm (lchs(i) < 0) {
            val embedding_tensor = TensorR(Tensor(word_embedding_data(words(i)), word_embedding_size))
            val hidden = (tWu.dot(embedding_tensor) + tbu).tanh()
            // val hidden = tWu.linearTanh(embedding_tensor, tbu)
            val pred1 = (tWhy.dot(hidden) + tby)
            val loss = pred1.logSoftmax().nllLoss(scores(i))
            ArrayBuffer(loss, hidden)
          } {
            val lossl = l(0); val hiddenl = l(1);
            val lossr = r(0); val hiddenr = r(1);
            val hidden = (tU0u.dot(hiddenl) + tU1u.dot(hiddenr) + tbbu).tanh()
            // val hidden = tU0u.linear2Tanh(hiddenl, tU1u, hiddenr, tbbu)
            val pred1 = (tWhy.dot(hidden) + tby)
            val loss = lossl + lossr + pred1.logSoftmax().nllLoss(scores(i))
            ArrayBuffer(loss, hidden)
          }
        }
        outBuffer(0)
      }

      val lr = learning_rate
      val hp = 1e-8f

      val mWu = Tensor.zeros_like(Wu)
      val mbu = Tensor.zeros_like(bu)
      val mU0u  = Tensor.zeros_like(U0u)
      val mU1u  = Tensor.zeros_like(U1u)
      val mbbu  = Tensor.zeros_like(bbu)
      // parameters for softmax
      val mWhy = Tensor.zeros_like(Why)
      val mby  = Tensor.zeros_like(by)

      val epocN = 6

      val loss_save = NewArray[Double](epocN)

      val addr = getMallocAddr() // remember current allocation pointer here

      val loopStart = get_time()

      for (epoc <- (0 until epocN): Rep[Range]) {

        var average_loss = 0.0f
        for (n <- (0 until tree_number): Rep[Range]) {

          val scores   = tree_data(n * 4)
          val words    = tree_data(n * 4 + 1)
          val leftchs  = tree_data(n * 4 + 2)
          val rightchs = tree_data(n * 4 + 3)
          val loss = gradR_loss(lossFun(scores, words, leftchs, rightchs))(Tensor.zeros(1))
          val loss_value = loss.data(0)  // we suppose the loss is scala (Tensor of size 1)
          average_loss = average_loss * (n) / (n+1) + loss_value / (n+1)

          val pars = ArrayBuffer(tWu, tbu, tU0u, tU1u, tbbu, tWhy, tby)
          val mems = ArrayBuffer(mWu, mbu, mU0u, mU1u, mbbu, mWhy, mby)
          for ((par, mem) <- pars.zip(mems)) {
            par.d.changeTo { i =>
              val temp = var_new(par.d.data(i))
              // if (temp > 5.0f) temp = 5.0f
              // if (temp < -5.0f) temp = -5.0f
              mem.data(i) += temp * temp
              par.x.data(i) -= lr * temp / Math.sqrt(mem.data(i) + hp).toFloat
              0.0f
            }
            // par.clip_grad(5.0f)
            // mem.mutate{i => val temp = par.d.data(i); temp * temp}
            // par.x.mutate(i => -lr * par.d.data(i) / Math.sqrt(mem.data(i) + hp).toFloat)
            // // par.x -= par.d * lr / (mem + hp).sqrt()
            // par.clear_grad()
          }

          resetMallocAddr(addr)  // reset malloc_addr to the value when we remember allocation pointer */
        }

        loss_save(epoc) = average_loss
        val tempTime = get_time()
        printf("epoc %d, average_loss %f, time %lf\\n", epoc, average_loss, (tempTime - loopStart))

      }

      val loopEnd = get_time()
      val prepareTime = loopStart - startTime
      val loopTime = loopEnd - loopStart
      val timePerEpoc = loopTime / epocN

      val fp2 = openf(a, "w")
      fprintf(fp2, "unit: %s\\n", "1 epoch")
      for (i <- (0 until loss_save.length): Rep[Range]) {
        //printf("loss_saver is %lf \\n", loss_save(i))
        fprintf(fp2, "%lf\\n", loss_save(i))
      }
      fprintf(fp2, "run time: %lf %lf\\n", prepareTime, timePerEpoc)
      closef(fp2)

    }
  }

  def main(args: Array[String]) {
    val sentit_file = new PrintWriter(new File(root_dir2 + file_dir))
    sentit_file.println(sentimental_lstm.code)
    sentit_file.flush()
  }

}
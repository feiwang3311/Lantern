package lantern

import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

import scala.collection.mutable.ArrayBuffer
import scala.collection.{Seq => NSeq}
import scala.math._

import org.scalatest.FunSuite

import java.io.PrintWriter;
import java.io.File;

class SentimentLSTM extends FunSuite {

  val file_dir = "/tmp/sentiment_lstm.cpp"

  val senti_seq_lstm = new LanternDriverC[String, Unit] with ScannerLowerExp {

    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {

      // read in the data for word embedding
      val word_embedding_size   = 300
      val word_embedding_length = 5265 // need to know the size of file first, need fix
      val fp = openf("senti/small_glove.txt", "r")
      val word_embedding_data = NewArray[Array[Float]](word_embedding_length)

      for (i <- (0 until word_embedding_length): Rep[Range]) {
        word_embedding_data(i) = NewArray[Float](word_embedding_size)
        for (j <- (0 until word_embedding_size): Rep[Range]) getFloat(fp, word_embedding_data(i), j)
      }
      closef(fp)

      // read in the data for sequences (assume size, label, newline, word_indexes)
      val seq_number = 1101 // need to know the size of training data, need fix
      val fp1 = openf("senti/array_seq.txt", "r")
      val seq_data  = NewArray[Array[Int]](seq_number)
      val seq_label = NewArray[Int](seq_number)

      val size = NewArray[Int](1)
      for (i <- (0 until seq_number): Rep[Range]) {
        getInt(fp1, size, 0)
        seq_data(i) = NewArray[Int](size(0))
        getInt(fp1, seq_label, i)
        for (k <- (0 until size(0)): Rep[Range]) getInt(fp1, seq_data(i), k)
      }

      val hidden_size = 150
      val output_size = 5
      val learning_rate = 1e-1f

      // initialize all parameters:
      val Wfh = Tensor.randn(hidden_size, hidden_size, 0.01f)
      val Wfx = Tensor.randn(hidden_size, word_embedding_size, 0.01f)
      val bf  = Tensor.zeros(hidden_size)
      val Wih = Tensor.randn(hidden_size, hidden_size, 0.01f)
      val Wix = Tensor.randn(hidden_size, word_embedding_size, 0.01f)
      val bi  = Tensor.zeros(hidden_size)
      val Wch = Tensor.randn(hidden_size, hidden_size, 0.01f)
      val Wcx = Tensor.randn(hidden_size, word_embedding_size, 0.01f)
      val bc  = Tensor.zeros(hidden_size)
      val Woh = Tensor.randn(hidden_size, hidden_size, 0.01f)
      val Wox = Tensor.randn(hidden_size, word_embedding_size, 0.01f)
      val bo  = Tensor.zeros(hidden_size)
      val Why = Tensor.randn(output_size, hidden_size, 0.01f)  // hidden to output
      val by  = Tensor.zeros(output_size)

      val hprev = Tensor.zeros(hidden_size)
      val cprev = Tensor.zeros(hidden_size)

      // wrap as Tensors
      val tWfh = TensorR(Wfh)
      val tWfx = TensorR(Wfx)
      val tbf = TensorR(bf)
      val tWih = TensorR(Wih)
      val tWix = TensorR(Wix)
      val tbi = TensorR(bi)
      val tWch = TensorR(Wch)
      val tWcx = TensorR(Wcx)
      val tbc = TensorR(bc)
      val tWoh = TensorR(Woh)
      val tWox = TensorR(Wox)
      val tbo = TensorR(bo)
      val tWhy = TensorR(Why)
      val tby = TensorR(by)
      val thprev = TensorR(hprev)
      val tcprev = TensorR(cprev)

      // lossFun
      def lossFun(inputs: Rep[Array[Int]], label: Rep[Int]) = { (dummy: TensorR) =>

        val in = ArrayBuffer[TensorR]()
        in.append(thprev)
        in.append(tcprev)

        val outputs = LOOPSM(in)(inputs.length){i => t =>

          // get word embedding
          val x    = word_embedding_data(inputs(i))
          val x1   = TensorR(Tensor(x, word_embedding_size))

          val ft = (tWfh.dot(t(0)) + tWfx.dot(x1) + tbf).sigmoid()
          val it = (tWih.dot(t(0)) + tWix.dot(x1) + tbi).sigmoid()
          val ot = (tWoh.dot(t(0)) + tWox.dot(x1) + tbo).sigmoid()
          val Ct = (tWch.dot(t(0)) + tWcx.dot(x1) + tbc).tanh()
          val ct = ft * t(1) + it * Ct
          val ht = ot * ct.tanh()

          val out = ArrayBuffer[TensorR]()
          out.append(ht)
          out.append(ct)
          out
        }
        val et = (tWhy.dot(outputs(0)) + tby).exp()
        val pt = et / et.sum()

        val y = Tensor.zeros(output_size)
        y.data(label) = 1
        val y1 = TensorR(y)

        val loss = TensorR(Tensor.zeros(1)) - (pt dot y1).log()
        loss
      }


      val lr = learning_rate
      val hp = 1e-8f

      val mWfh = Tensor.zeros_like(Wfh)
      val mWfx = Tensor.zeros_like(Wfx)
      val mbf = Tensor.zeros_like(bf)
      val mWih = Tensor.zeros_like(Wih)
      val mWix = Tensor.zeros_like(Wix)
      val mbi = Tensor.zeros_like(bi)
      val mWch = Tensor.zeros_like(Wch)
      val mWcx = Tensor.zeros_like(Wcx)
      val mbc = Tensor.zeros_like(bc)
      val mWoh = Tensor.zeros_like(Woh)
      val mWox = Tensor.zeros_like(Wox)
      val mbo = Tensor.zeros_like(bo)
      val mWhy = Tensor.zeros_like(Why)
      val mby = Tensor.zeros_like(by)

      val addr = getMallocAddr() // remember current allocation pointer here

      for (n <- (0 until 2001): Rep[Range]) {

        val index  = n % seq_number
        val inputs = seq_data(index)
        val label  = seq_label(index)

        val loss = gradR_loss(lossFun(inputs, label))(Tensor.zeros(1))
        val loss_value = loss.data(0) // we suppose the loss is scala (Tensor of size 1)
        if (n % 100 == 0) {
          printf("iter %d, loss %f\\n", n, loss_value)
          //timer.printElapsedTime
        }

        val pars = ArrayBuffer(tWfh, tWfx, tbf, tWih, tWix, tbi, tWch, tWcx, tbc, tWoh, tWox, tbo, tWhy, tby)
        val mems = ArrayBuffer(mWfh, mWfx, mbf, mWih, mWix, mbi, mWch, mWcx, mbc, mWoh, mWox, mbo, mWhy, mby)
        for ((par, mem) <- pars.zip(mems)) {
          par.clip_grad(5.0f)
          mem += par.d * par.d
          par.x -= par.d * lr / (mem + hp).sqrt()
          par.clear_grad()
        }
        thprev.clear_grad()          // clear gradient of all Tensors for next cycle
        tcprev.clear_grad()          // clear gradient of all Tensors for next cycle

        resetMallocAddr(addr)  // reset malloc_addr to the value when we remember allocation pointer
      }
    }
  }

  test("generate_code_for_sentiment_lstm") {
    val min_char_rnn_file = new PrintWriter(new File(file_dir))
    min_char_rnn_file.println(senti_seq_lstm.code)
    min_char_rnn_file.flush()
  }

}
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

class SentimentTreeLSTM extends FunSuite {

  val root_dir = "src/out/ICFP18evaluation/"
  val file_dir = "evaluationTreeLSTM/Lantern/Lantern.cpp"

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

      // parameters for leaf node
      val Wi = Tensor.randinit(hidden_size, word_embedding_size, 0.01f)  // from word embedding to hidden vector, input gate
      val bi = Tensor.zeros(hidden_size)                                // bias word embedding to hidden vector, input gate
      val Wo = Tensor.randinit(hidden_size, word_embedding_size, 0.01f)  // from word embedding to hidden vector, outout gate
      val bo = Tensor.zeros(hidden_size)                                // bias word embedding to hidden vector, outout gate
      val Wu = Tensor.randinit(hidden_size, word_embedding_size, 0.01f)  // from word embedding to hidden vector, cell state
      val bu = Tensor.zeros(hidden_size)                                // bias word embedding to hidden vector, cell state
      // parameters for non-leaf node
      val U0i  = Tensor.randinit(hidden_size, hidden_size, 0.01f) // left child, input gate
      val U1i  = Tensor.randinit(hidden_size, hidden_size, 0.01f) // right child, input gate
      val bbi  = Tensor.zeros(hidden_size)                       // bias, input gate
      val U00f = Tensor.randinit(hidden_size, hidden_size, 0.01f) // left-left forget gate
      val U01f = Tensor.randinit(hidden_size, hidden_size, 0.01f) // left-right forget gate
      val U10f = Tensor.randinit(hidden_size, hidden_size, 0.01f) // right-left forget gate
      val U11f = Tensor.randinit(hidden_size, hidden_size, 0.01f) // right-right forget gate
      val bbf  = Tensor.zeros(hidden_size)                       // bias, forget gate
      val U0o  = Tensor.randinit(hidden_size, hidden_size, 0.01f) // left child, output gate
      val U1o  = Tensor.randinit(hidden_size, hidden_size, 0.01f) // right child, output gate
      val bbo  = Tensor.zeros(hidden_size)                       // bias, output gate
      val U0u  = Tensor.randinit(hidden_size, hidden_size, 0.01f) // left child, cell state
      val U1u  = Tensor.randinit(hidden_size, hidden_size, 0.01f) // right child, cell state
      val bbu  = Tensor.zeros(hidden_size)                       // bias, cell state
      // parameters for softmax
      val Why = Tensor.randinit(output_size, hidden_size, 0.01f)         // from hidden vector to output
      val by  = Tensor.zeros(output_size)                               // bias hidden vector to output

      // Cast Tensors as Tensors
      val tWi = TensorR(Wi)
      val tbi = TensorR(bi)
      val tWo = TensorR(Wo)
      val tbo = TensorR(bo)
      val tWu = TensorR(Wu)
      val tbu = TensorR(bu)
      // Cast Tensors as Tensors
      val tU0i  = TensorR(U0i)
      val tU1i  = TensorR(U1i)
      val tbbi  = TensorR(bbi)
      val tU00f = TensorR(U00f)
      val tU01f = TensorR(U01f)
      val tU10f = TensorR(U10f)
      val tU11f = TensorR(U11f)
      val tbbf = TensorR(bbf)
      val tU0o = TensorR(U0o)
      val tU1o = TensorR(U1o)
      val tbbo = TensorR(bbo)
      val tU0u = TensorR(U0u)
      val tU1u = TensorR(U1u)
      val tbbu = TensorR(bbu)
      // Cast Tensors as Tensors
      val tWhy = TensorR(Why)
      val tby  = TensorR(by)

      val dummy_word_embedding = TensorR(Tensor.zeros(word_embedding_size))
      val dummy_forget_gate    = TensorR(Tensor.zeros(hidden_size))

      def lossFun(scores: Rep[Array[Int]], words: Rep[Array[Int]], lchs: Rep[Array[Int]], rchs: Rep[Array[Int]]) = { (dummy: TensorR) =>

        val initial_loss = TensorR(Tensor.zeros(1))
        val initial_hidd = TensorR(Tensor.zeros(hidden_size))
        val initial_cell = TensorR(Tensor.zeros(hidden_size))
        val inBuffer     = new ArrayBuffer[TensorR]()
        inBuffer.append(initial_loss); inBuffer.append(initial_hidd); inBuffer.append(initial_cell)

        val outBuffer = LOOPTM(0)(inBuffer)(lchs, rchs) { (l: ArrayBuffer[TensorR], r: ArrayBuffer[TensorR], i: Rep[Int]) =>

          val lossl = l(0); val hiddenl = l(1); val celll = l(2)
          val lossr = r(0); val hiddenr = r(1); val cellr = r(2)

          val targ = Tensor.zeros(output_size); targ.data(scores(i)) = 1; val targ1 = TensorR(targ)

          IFm (lchs(i) < 0) {
            val embedding_tensor = TensorR(Tensor(word_embedding_data(words(i)), word_embedding_size))
            val i_gate = (tWi.dot(embedding_tensor) + tbi).sigmoid()
            val o_gate = (tWo.dot(embedding_tensor) + tbo).sigmoid()
            val u_value = (tWu.dot(embedding_tensor) + tbu).tanh()
            val cell = i_gate * u_value

            val hidden = o_gate * cell.tanh()
            val pred1 = (tWhy.dot(hidden) + tby).exp()
            val pred2 = pred1 / pred1.sum()
            val loss = lossl + lossr - (pred2 dot targ1).log()

            val out = ArrayBuffer[TensorR]()
            out.append(loss)
            out.append(hidden)
            out.append(cell)
            out
          } {
            val i_gate = (tU0i.dot(hiddenl) + tU1i.dot(hiddenr) + tbbi).sigmoid()
            val fl_gate = (tU00f.dot(hiddenl) + tU01f.dot(hiddenr) + tbbf).sigmoid()
            val fr_gate = (tU10f.dot(hiddenl) + tU11f.dot(hiddenr) + tbbf).sigmoid()
            val o_gate = (tU0o.dot(hiddenl) + tU1o.dot(hiddenr) + tbbo).sigmoid()
            val u_value = (tU0u.dot(hiddenl) + tU1u.dot(hiddenr) + tbbu).tanh()
            val cell = i_gate * u_value + fl_gate * celll + fr_gate * cellr

            val hidden = o_gate * cell.tanh()
            val pred1 = (tWhy.dot(hidden) + tby).exp()
            val pred2 = pred1 / pred1.sum()
            val loss = lossl + lossr - (pred2 dot targ1).log()

            val out = ArrayBuffer[TensorR]()
            out.append(loss)
            out.append(hidden)
            out.append(cell)
            out
          }
        }
        outBuffer(0)
      }

      val lr = learning_rate
      val hp = 1e-8f

      // parameters for leaf node
      val mWi = Tensor.zeros_like(Wi)
      val mbi = Tensor.zeros_like(bi)
      val mWo = Tensor.zeros_like(Wo)
      val mbo = Tensor.zeros_like(bo)
      val mWu = Tensor.zeros_like(Wu)
      val mbu = Tensor.zeros_like(bu)
      // parameters for non-leaf node
      val mU0i  = Tensor.zeros_like(U0i)
      val mU1i  = Tensor.zeros_like(U1i)
      val mbbi  = Tensor.zeros_like(bbi)
      val mU00f = Tensor.zeros_like(U00f)
      val mU01f = Tensor.zeros_like(U01f)
      val mU10f = Tensor.zeros_like(U10f)
      val mU11f = Tensor.zeros_like(U11f)
      val mbbf  = Tensor.zeros_like(bbf)
      val mU0o  = Tensor.zeros_like(U0o)
      val mU1o  = Tensor.zeros_like(U1o)
      val mbbo  = Tensor.zeros_like(bbo)
      val mU0u  = Tensor.zeros_like(U0u)
      val mU1u  = Tensor.zeros_like(U1u)
      val mbbu  = Tensor.zeros_like(bbu)
      // parameters for softmax
      val mWhy = Tensor.zeros_like(Why)
      val mby  = Tensor.zeros_like(by)

      val loss_save = NewArray[Double](30)

      val addr = getMallocAddr() // remember current allocation pointer here

       val loopStart = get_time()

      for (epoc <- (0 until 30): Rep[Range]) {

        var average_loss = 0.0f
        for (n <- (0 until tree_number): Rep[Range]) {

          val index = n % tree_number
          val scores   = tree_data(index * 4)
          val words    = tree_data(index * 4 + 1)
          val leftchs  = tree_data(index * 4 + 2)
          val rightchs = tree_data(index * 4 + 3)
          val loss = gradR_loss(lossFun(scores, words, leftchs, rightchs))(Tensor.zeros(1))
          val loss_value = loss.data(0)  // we suppose the loss is scala (Tensor of size 1)
          average_loss = average_loss * (n) / (n+1) + loss_value / (n+1)

          val pars = ArrayBuffer(tWi, tbi, tWo, tbo, tWu, tbu, tU0i, tU1i, tbbi, tU00f, tU01f, tU10f, tU11f, tbbf, tU0o, tU1o, tbbo, tU0u, tU1u, tbbu, tWhy, tby)
          val mems = ArrayBuffer(mWi, mbi, mWo, mbo, mWu, mbu, mU0i, mU1i, mbbi, mU00f, mU01f, mU10f, mU11f, mbbf, mU0o, mU1o, mbbo, mU0u, mU1u, mbbu, mWhy, mby)
          for ((par, mem) <- pars.zip(mems)) {
            par.clip_grad(5.0f)
            mem += par.d * par.d
            par.x -= par.d * lr / (mem + hp).sqrt()
            par.clear_grad()
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
      val timePerEpoc = loopTime / 30

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

  test("generate_code_for_sentiment_tree_lstm"){
    //println("generate code for Sentiment Tree LSTM")
    val sentit_file = new PrintWriter(new File(root_dir + file_dir))
    sentit_file.println(sentimental_lstm.code)
    sentit_file.flush()
    //println(s"now your code at $root_dir/$file_dir is generated.")
  }

}

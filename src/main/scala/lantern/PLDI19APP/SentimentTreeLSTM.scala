package lantern
package PLDI19App

import lms.core.stub._
import lms.macros.SourceContext
import lms.core.virtualize

import scala.collection.mutable.ArrayBuffer
import scala.math._

import java.io.PrintWriter
import java.io.File

object SentimentTreeLSTM {

  val file_dir_cpu = "treelstm/lantern/LanternTraining.cpp"
  val file_dir_gpu = "treelstm/lantern/LanternTraining.cu"
  val root_dir = "src/out/PLDI19evaluation/"

  val sentimental_lstm_cpu = new LanternDriverC[String, Unit] {

    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {

      debug = false

      val startTime = get_time()

      // read in the data for word embedding
      val word_embedding_size   = 300
      val readingSlot1 = NewArray[Int](1) // this is a slot of memory used for reading from file
      val fp = fopen("small_glove.txt", "r")
      getInt(fp, readingSlot1, 0) // read the first number in the file, which is "How many rows"
      val word_embedding_length = readingSlot1(0)
      val word_embedding_data = NewArray[Array[Float]](word_embedding_length)
      for (i <- (0 until word_embedding_length): Rep[Range]) {
        word_embedding_data(i) = NewArray[Float](word_embedding_size)
        for (j <- (0 until word_embedding_size): Rep[Range]) getFloat(fp, word_embedding_data(i), j)
      }
      fclose(fp)

      // read in the data for trees
      val readingSlot2 = NewArray[Int](1) // need a new readingSlot, other wise have error
      val fp1 = fopen("array_tree.txt", "r")
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
      fclose(fp1)

      // set up hyperparameters and parameters
      val hidden_size = 150
      val output_size = 5
      val learning_rate = 0.05f

      case class Leaf(val name: String = "leaf", val inSize: Int, val hSize: Int) extends Module {
        val Wi = TensorR(Tensor.randinit(hSize, inSize, 0.01f))
        val bi = TensorR(Tensor.zeros(hSize))
        val Wo = TensorR(Tensor.randinit(hSize, inSize, 0.01f))
        val bo = TensorR(Tensor.zeros(hSize))
        val Wu = TensorR(Tensor.randinit(hSize, inSize, 0.01f))
        val bu = TensorR(Tensor.zeros(hSize))
        def apply(in: TensorR) = {
          val i_gate = (Wi.dot(in) plusBias bi).sigmoid()
          val o_gate = (Wo.dot(in) plusBias bo).sigmoid()
          val u_value = (Wu.dot(in) plusBias bu).tanh()
          val cell = i_gate * u_value
          val hidden = o_gate * cell.tanh()
          ArrayBuffer(hidden, cell)
        }
      }

      case class Node(val name: String = "node", val hSize: Int) extends Module {
        val U0i  = TensorR(Tensor.randinit(hSize, hSize, 0.01f))
        val U1i  = TensorR(Tensor.randinit(hSize, hSize, 0.01f))
        val bbi  = TensorR(Tensor.zeros(hSize))
        val U00f = TensorR(Tensor.randinit(hSize, hSize, 0.01f))
        val U01f = TensorR(Tensor.randinit(hSize, hSize, 0.01f))
        val U10f = TensorR(Tensor.randinit(hSize, hSize, 0.01f))
        val U11f = TensorR(Tensor.randinit(hSize, hSize, 0.01f))
        val bbf  = TensorR(Tensor.zeros(hSize))
        val U0o  = TensorR(Tensor.randinit(hSize, hSize, 0.01f))
        val U1o  = TensorR(Tensor.randinit(hSize, hSize, 0.01f))
        val bbo  = TensorR(Tensor.zeros(hSize))
        val U0u  = TensorR(Tensor.randinit(hSize, hSize, 0.01f))
        val U1u  = TensorR(Tensor.randinit(hSize, hSize, 0.01f))
        val bbu  = TensorR(Tensor.zeros(hSize))
        def apply(l: ArrayBuffer[TensorR], r: ArrayBuffer[TensorR]) = {
          val ArrayBuffer(lossl, hiddenl, celll) = l
          val ArrayBuffer(lossr, hiddenr, cellr) = r
          val i_gate = (U0i.dot(hiddenl) plusBias U1i.dot(hiddenr) plusBias bbi).sigmoid()
          val fl_gate = (U00f.dot(hiddenl) plusBias U01f.dot(hiddenr) plusBias bbf).sigmoid()
          val fr_gate = (U10f.dot(hiddenl) plusBias U11f.dot(hiddenr) plusBias bbf).sigmoid()
          val o_gate = (U0o.dot(hiddenl) plusBias U1o.dot(hiddenr) plusBias bbo).sigmoid()
          val u_value = (U0u.dot(hiddenl) plusBias U1u.dot(hiddenr) plusBias bbu).tanh()
          val cell = i_gate * u_value plusBias fl_gate * celll plusBias fr_gate * cellr
          val hidden = o_gate * cell.tanh()
          ArrayBuffer(hidden, cell)
        }
      }

      case class Out(val name: String = "output", val hSize: Int, outSize: Int) extends Module {
        val Why = TensorR(Tensor.randinit(outSize, hSize, 0.01f))
        val by  = TensorR(Tensor.zeros(outSize))
        def apply(in: TensorR, score: Rep[Array[Int]]) = {
          val pred1 = Why.dot(in) plusBias by
          val loss = pred1.resize(1, pred1.x.shape(0)).logSoftmaxB(1).nllLossB(score)
          loss
        }
      }

      case class Tree(val name: String = "tree", val inSize: Int, val hSize: Int, val outSize: Int) extends Module {
        val leaf = Leaf(inSize = inSize, hSize = hSize)
        val node = Node(hSize = hSize)
        val out = Out(hSize = hSize, outSize = outSize)
        val inBuffer = ArrayBuffer(TensorR(Tensor.zeros(1)), TensorR(Tensor.zeros(hSize)), TensorR(Tensor.zeros(hSize)))
        def apply(scores: Rep[Array[Int]], words: Rep[Array[Int]], lchs: Rep[Array[Int]], rchs: Rep[Array[Int]])(dummy: TensorR) = {
          val outBuffer = LOOPTM(0)(inBuffer)(lchs, rchs) { (l: ArrayBuffer[TensorR], r: ArrayBuffer[TensorR], i: Rep[Int]) =>
            IFm (lchs(i) < 0) {
              val embedding_tensor = TensorR(Tensor(word_embedding_data(words(i)), word_embedding_size))
              val ArrayBuffer(cell, hidden) = leaf(embedding_tensor)
              val loss = out(hidden, slice(scores, i))
              ArrayBuffer(loss, hidden, cell)
            } {
              val ArrayBuffer(cell, hidden) = node(l, r)
              val loss = out(hidden, slice(scores, i)) plusBias l(0) plusBias r(0)
              ArrayBuffer(loss, hidden, cell)
            }
          }
          outBuffer(0)
        }
      }

      val net = Tree(inSize = word_embedding_size, hSize = hidden_size, outSize = output_size)
      val opt = Adagrad(net, learning_rate = learning_rate)

      val epocN = 6
      val loss_save = NewArray[Double](epocN)

      val addr = getMallocAddr() // remember current allocation pointer here
      val loopStart = get_time()

      for (epoc <- (0 until epocN): Rep[Range]) {
        var average_loss = 0.0f
        for (n <- (0 until tree_number): Rep[Range]) {
          val index = n % tree_number
          val scores   = tree_data(index * 4)
          val words    = tree_data(index * 4 + 1)
          val leftchs  = tree_data(index * 4 + 2)
          val rightchs = tree_data(index * 4 + 3)
          val loss = gradR_loss(net(scores, words, leftchs, rightchs))(Tensor.zeros(1))
          val loss_value = loss.data(0)
          average_loss = average_loss * (n) / (n+1) + loss_value / (n+1)
          opt.step()
          resetMallocAddr(addr)
        }

        loss_save(epoc) = average_loss
        val tempTime = get_time()
        printf("epoc %d, average_loss %f, time %lf\\n", epoc, average_loss, (tempTime - loopStart))
      }

      val loopEnd = get_time()
      val prepareTime = loopStart - startTime
      val loopTime = loopEnd - loopStart
      val timePerEpoc = loopTime / epocN

      val fp2 = fopen(a, "w")
      fprintf(fp2, "unit: %s\\n", "1 epoch")
      for (i <- (0 until loss_save.length): Rep[Range]) {
        //printf("loss_saver is %lf \\n", loss_save(i))
        fprintf(fp2, "%lf\\n", loss_save(i))
      }
      fprintf(fp2, "run time: %lf %lf\\n", prepareTime, timePerEpoc)
      fclose(fp2)

    }
  }

  val sentimental_lstm_gpu = new LanternDriverCudnn[String, Unit] {

    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {

      debug = false

      val startTime = get_time()

      // read in the data for word embedding
      val word_embedding_size   = 300
      val readingSlot1 = NewArray[Int](1) // this is a slot of memory used for reading from file
      val fp = fopen("small_glove.txt", "r")
      getInt(fp, readingSlot1, 0) // read the first number in the file, which is "How many rows"
      val word_embedding_length = readingSlot1(0)
      val word_embedding_data = NewArray[Array[Float]](word_embedding_length)
      for (i <- (0 until word_embedding_length): Rep[Range]) {
        word_embedding_data(i) = NewArray[Float](word_embedding_size)
        for (j <- (0 until word_embedding_size): Rep[Range]) getFloat(fp, word_embedding_data(i), j)
      }
      fclose(fp)

      // explicitly send word embedding to GPU (as input, so no gradient needed)
      val word_embedding_data_gpu = NewArray[Array[Float]](word_embedding_length)
      for (i <- (0 until word_embedding_length): Rep[Range]) {
        word_embedding_data_gpu(i) = word_embedding_data(i).toGPU(word_embedding_size)
      }

      // read in the data for trees
      val readingSlot2 = NewArray[Int](1) // need a new readingSlot, other wise have error
      val fp1 = fopen("array_tree.txt", "r")
      getInt(fp1, readingSlot2, 0)
      val tree_number = readingSlot2(0)
      val tree_data = NewArray[Array[Int]](tree_number * 4) // each tree data has 4 lines (score, word, lch, rch)
      val tree_size = NewArray[Int](tree_number)
      val readingSlot3 = NewArray[Int](1) // yet another readingSlot, not sure if this one can be reused
      for (i <- (0 until tree_number): Rep[Range]) {
        getInt(fp1, readingSlot3, 0)
        tree_size(i) = readingSlot3(0)
        for (j <- (0 until 4): Rep[Range]) {
          tree_data(i * 4 + j) = NewArray[Int](readingSlot3(0))
          for (k <- (0 until readingSlot3(0)): Rep[Range]) getInt(fp1, tree_data(i * 4 + j), k)
        }
      }
      fclose(fp1)

      // set up hyperparameters and parameters
      val hidden_size = 150
      val output_size = 5
      val learning_rate = 0.05f

      case class Leaf(val name: String = "leaf", val inSize: Int, val hSize: Int) extends Module {
        val Wi = TensorR(Tensor.randinit(hSize, inSize, 0.01f))
        val bi = TensorR(Tensor.zeros(hSize))
        val Wo = TensorR(Tensor.randinit(hSize, inSize, 0.01f))
        val bo = TensorR(Tensor.zeros(hSize))
        val Wu = TensorR(Tensor.randinit(hSize, inSize, 0.01f))
        val bu = TensorR(Tensor.zeros(hSize))
        def apply(in: TensorR) = {
          val i_gate = (Wi.dot(in) plusBias bi).sigmoid()
          val o_gate = (Wo.dot(in) plusBias bo).sigmoid()
          val u_value = (Wu.dot(in) plusBias bu).tanh()
          val cell = i_gate * u_value
          val hidden = o_gate * cell.tanh()
          ArrayBuffer(hidden, cell)
        }
      }

      case class Node(val name: String = "node", val hSize: Int) extends Module {
        val U0i  = TensorR(Tensor.randinit(hSize, hSize, 0.01f))
        val U1i  = TensorR(Tensor.randinit(hSize, hSize, 0.01f))
        val bbi  = TensorR(Tensor.zeros(hSize))
        val U00f = TensorR(Tensor.randinit(hSize, hSize, 0.01f))
        val U01f = TensorR(Tensor.randinit(hSize, hSize, 0.01f))
        val U10f = TensorR(Tensor.randinit(hSize, hSize, 0.01f))
        val U11f = TensorR(Tensor.randinit(hSize, hSize, 0.01f))
        val bbf  = TensorR(Tensor.zeros(hSize))
        val U0o  = TensorR(Tensor.randinit(hSize, hSize, 0.01f))
        val U1o  = TensorR(Tensor.randinit(hSize, hSize, 0.01f))
        val bbo  = TensorR(Tensor.zeros(hSize))
        val U0u  = TensorR(Tensor.randinit(hSize, hSize, 0.01f))
        val U1u  = TensorR(Tensor.randinit(hSize, hSize, 0.01f))
        val bbu  = TensorR(Tensor.zeros(hSize))
        def apply(l: ArrayBuffer[TensorR], r: ArrayBuffer[TensorR]) = {
          val ArrayBuffer(_, hiddenl, celll) = l  // this tensors are in GPU as well
          val ArrayBuffer(_, hiddenr, cellr) = r  // this tensors are in GPU as well
          val i_gate = (U0i.dot(hiddenl) plusEqual U1i.dot(hiddenr) plusBias bbi).sigmoid()
          val fl_gate = (U00f.dot(hiddenl) plusEqual U01f.dot(hiddenr) plusBias bbf).sigmoid()
          val fr_gate = (U10f.dot(hiddenl) plusEqual U11f.dot(hiddenr) plusBias bbf).sigmoid()
          val o_gate = (U0o.dot(hiddenl) plusEqual U1o.dot(hiddenr) plusBias bbo).sigmoid()
          val u_value = (U0u.dot(hiddenl) plusEqual U1u.dot(hiddenr) plusBias bbu).tanh()
          val cell = i_gate * u_value plusEqual fl_gate * celll plusEqual fr_gate * cellr
          val hidden = o_gate * cell.tanh()
          ArrayBuffer(hidden, cell)
        }
      }

      case class Out(val name: String = "output", val hSize: Int, outSize: Int) extends Module {
        val Why = TensorR(Tensor.randinit(outSize, hSize, 0.01f))
        val by  = TensorR(Tensor.zeros(outSize))
        def apply(in: TensorR, score: Rep[Array[Int]]) = {
          val pred1 = Why.dot(in) plusBias by
          val loss = pred1.resize(1, pred1.x.shape(0)).logSoftmaxB(1).nllLossB(score)
          loss
        }
      }

      case class Tree(val name: String = "tree", val inSize: Int, val hSize: Int, val outSize: Int) extends Module {
        val leaf = Leaf(inSize = inSize, hSize = hSize)
        val node = Node(hSize = hSize)
        val out = Out(hSize = hSize, outSize = outSize)
        val inBuffer = ArrayBuffer(TensorR(Tensor.zeros(1), isInput = true), TensorR(Tensor.zeros(hSize), isInput = true), TensorR(Tensor.zeros(hSize), isInput = true))
        def apply(scores: Rep[Array[Int]], words: Rep[Array[Int]], lchs: Rep[Array[Int]], rchs: Rep[Array[Int]])(dummy: TensorR) = {
          val outBuffer = LOOPTM(0)(inBuffer)(lchs, rchs) { (l: ArrayBuffer[TensorR], r: ArrayBuffer[TensorR], i: Rep[Int]) =>
            IFm (lchs(i) < 0) {
              val embedding_tensor = TensorR(Tensor(word_embedding_data_gpu(words(i)), word_embedding_size), isInput=true)
              val ArrayBuffer(cell, hidden) = leaf(embedding_tensor)
              val loss = out(hidden, slice(scores, i))
              ArrayBuffer(loss, hidden, cell)
            } {
              val ArrayBuffer(cell, hidden) = node(l, r)
              val loss = out(hidden, slice(scores, i)) plusEqual l(0) plusEqual r(0)
              ArrayBuffer(loss, hidden, cell)
            }
          }
          outBuffer(0)
        }
      }

      val net = Tree(inSize = word_embedding_size, hSize = hidden_size, outSize = output_size)
      val opt = Adagrad(net, learning_rate = learning_rate)

      val epocN = 6
      val loss_save = NewArray[Double](epocN)
      val time_save = NewArray[Double](epocN)

      val addr = getMallocAddr() // remember current allocation pointer here
      val cudaAddr = getCudaMallocAddr()
      val loopStart = get_time()

      var epoc_start_time = loopStart
      for (epoc <- (0 until epocN): Rep[Range]) {
        var average_loss = 0.0f
        for (n <- (0 until tree_number): Rep[Range]) {
          val index = n
          val size = tree_size(n)
          val scores   = tree_data(index * 4)
          val words    = tree_data(index * 4 + 1)
          val leftchs  = tree_data(index * 4 + 2)
          val rightchs = tree_data(index * 4 + 3)
          val loss = gradR_loss(net(scores.toGPU(size), words, leftchs, rightchs))(Tensor.zeros(1))
          val loss_value = loss.toCPU().data(0)
          average_loss = average_loss * (n) / (n+1) + loss_value / (n+1)
          opt.step()
          resetMallocAddr(addr)
          resetCudaMallocAddr(cudaAddr)
        }

        loss_save(epoc) = average_loss
        val epoc_end_time = get_time()
        time_save(epoc) = epoc_end_time - epoc_start_time
        epoc_start_time = epoc_end_time
        printf("epoc %d, average_loss %f, time %lf\\n", epoc, average_loss, time_save(epoc))
      }

      val loopEnd = get_time()
      val prepareTime = loopStart - startTime
      val loopTime = loopEnd - loopStart
      val timePerEpoc = loopTime / epocN

      // get median time of epochs
      unchecked[Unit]("sort(", time_save, ", ", time_save, " + ", epocN, ")")
      val median_time =  time_save(epocN / 2)

      val fp2 = fopen(a, "w")
      fprintf(fp2, "unit: %s\\n", "1 epoch")
      for (i <- (0 until loss_save.length): Rep[Range]) {
        //printf("loss_saver is %lf \\n", loss_save(i))
        fprintf(fp2, "%lf\\n", loss_save(i))
      }
      fprintf(fp2, "run time: %lf %lf\\n", prepareTime, median_time)
      fclose(fp2)

    }
  }

  def main(args: Array[String]) {
    val sentit_file_cpu = new PrintWriter(new File(root_dir + file_dir_cpu))
    sentit_file_cpu.println(sentimental_lstm_cpu.code)
    sentit_file_cpu.flush()
    val sentit_file_gpu = new PrintWriter(new File(root_dir + file_dir_gpu))
    sentit_file_gpu.println(sentimental_lstm_gpu.code)
    sentit_file_gpu.flush()
  }
}

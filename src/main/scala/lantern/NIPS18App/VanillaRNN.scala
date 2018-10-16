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

object VanillaRNN {

  val root_dir = "src/out/ICFP18evaluation/"
  val file_dir = "evaluationRNN/Lantern.cpp"
  val root_dir2 = "src/out/NIPS18evaluation/"

  val min_char_rnn = new LanternDriverC[String, Unit] {

    class Scanner(name: Rep[String]) {
      val fd = open(name)
      val fl = filelen(fd)
      val data = mmap[Char](fd,fl)
      var pos = 0

      def nextChar: Rep[Char] = {
        val ch = data(pos)
        pos += 1
        ch
      }

      def hasNextChar = pos < fl
      def done = close(fd)
    }


    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {
      /**
       add scanner
       **/

      val startTime = get_time()

      val scanner = new Scanner("graham.txt")
      val training_data = scanner.data
      val data_size = scanner.fl
      // val chars = training_data.distinct  /** this can be done in second stage **/
      // val vocab_size = chars.length
      printf("data has %d chars\\n", data_size)

      //val translated_data = NewArray[Int](data_size)
      //for (i <- (0 until data_size)) translated_data(i) = char_to_ix(unit(training_data).charAt(i))
      val translated_data = NewArray[Int](data_size)
      for (i <- (0 until data_size)) { translated_data(i) = Encoding.char_to_ix(training_data(i)) }

      val vocab_size = 26                 // Do we have to get this size?
      val hidden_size = 50
      val learning_rate = 1e-1f
      val seq_length = 20
      //val Wxh = Tensor.randinit(vocab_size, hidden_size, 0.01f)  // input to hidden
      val Wxh = Tensor.randn(hidden_size, vocab_size, 0.01f)  // input to hidden
      val Whh = Tensor.randn(hidden_size, hidden_size, 0.01f) // hidden to hidden
      val Why = Tensor.randn(vocab_size, hidden_size, 0.01f)  // hidden to output
      val bh  = Tensor.zeros(hidden_size)
      val by  = Tensor.zeros(vocab_size)
      val hprev = Tensor.zeros(hidden_size)

      val hnext = Tensor.zeros_like(hprev)

      // wrap as tensors
      val Wxh1 = TensorR(Wxh)
      val Whh1 = TensorR(Whh)
      val Why1 = TensorR(Why)
      val bh1  = TensorR(bh)
      val by1  = TensorR(by)
      val hprev1 = TensorR(hprev)

      def lossFun(inputs: Rep[Array[Int]], targets: Rep[Array[Int]]) = { (dummy: TensorR) =>
        val loss = TensorR(Tensor.zeros(1))
        val in = ArrayBuffer[TensorR]()
        in.append(loss)
        in.append(hprev1)
        val outputs = LOOPSM(in)(inputs.length){i => t =>

          // printf("at iteration %d ", i)
          // get input as one-hot tensor
          val x = Tensor.zeros(vocab_size)
          x.data(inputs(i)) = 1
          val x1 = TensorR(x)
          // get output as one-hot tensor
          val y = Tensor.zeros(vocab_size)
          y.data(targets(i)) = 1
          val y1 = TensorR(y)

          val h1 = ((Wxh1 dot x1) + (Whh1 dot t(1)) + bh1).tanh() // use hidden state and x1 to compute hidden state
          val e1 = (Why1.dot(h1) + by1).exp()                       // use new hidden state to compute unnormalized prob
          val p1 = e1 / e1.sum()                            // use unnormalized prob to compute normalize prob
          val newloss = t(0) - (p1 dot y1).log()            // loss is updated by original loss t(0) and additional loss
          val out = ArrayBuffer[TensorR]()
          out.append(newloss)
          out.append(h1)
          out
        }
        hnext.copy_data(outputs(1).x)     // update the hidden state with the result from LOOP
        outputs(0)                        // return the final loss
      }


      val lr = learning_rate
      val hp = 1e-8f

      val mWxh = Tensor.zeros_like(Wxh)
      val mWhh = Tensor.zeros_like(Whh)
      val mWhy = Tensor.zeros_like(Why)
      val mbh  = Tensor.zeros_like(bh)
      val mby  = Tensor.zeros_like(by)

      val loss_save = NewArray[Double](51)
      val loopStartTime = get_time()

      val addr = getMallocAddr() // remember current allocation pointer here

      val startAt = var_new[Int](0)
      startAt -= seq_length

      var smooth_loss = 60.0f
      for (n <- (0 until 5001): Rep[Range]) {

        startAt += seq_length
        if (startAt + seq_length + 1 >= data_size) {
          startAt = 0
          hprev.clear()
        }

        val inputs = NewArray[Int](seq_length)
        val targets = NewArray[Int](seq_length)
        for (i <- (0 until seq_length): Rep[Range]) {
          inputs(i) = translated_data(startAt+i)
          targets(i) = translated_data(startAt+i+1)
        }

        val loss = gradR_loss(lossFun(inputs, targets))(Tensor.zeros(1))
        val loss_value = loss.data(0) // we suppose the loss is scala (Tensor of size 1)
        smooth_loss = smooth_loss * 0.9f + loss_value * 0.1f
        if (n % 100 == 0) {
          printf("iter %d, loss %f\\n", n, smooth_loss)
          loss_save(n / 100) = smooth_loss
        }

        val pars = ArrayBuffer(Wxh1, Whh1, Why1, bh1, by1)
        val mems = ArrayBuffer(mWxh, mWhh, mWhy, mbh, mby)
        for ((par, mem) <- pars.zip(mems)) {
          par.clip_grad(5.0f)
          mem += par.d * par.d
          par.x -= par.d * lr / (mem + hp).sqrt()
          par.clear_grad()
        }
        hprev1.clear_grad()          // clear gradient of all Tensors for next cycle
        hprev1.x.copy_data(hnext)

        resetMallocAddr(addr)  // reset malloc_addr to the value when we remember allocation pointer
      }

      val loopEndTime = get_time()
      val prepareTime = loopStartTime - startTime
      val loopTime    = loopEndTime - loopStartTime

      val fp = openf(a, "w")
      fprintf(fp, "unit: %s\\n", "100 iteration")
      for (i <- (0 until loss_save.length): Rep[Range]) {
        fprintf(fp, "%lf\\n", loss_save(i))
      }
      fprintf(fp, "run time: %lf %lf\\n", prepareTime, loopTime)
      closef(fp)

    }
  }

  val min_char_rnn_module = new DslDriverC[String, Unit] with NNModule {

    class Scanner(name: Rep[String]) {
      val fd = open(name)
      val fl = filelen(fd)
      val data = mmap[Char](fd,fl)
      var pos = 0

      def nextChar: Rep[Char] = {
        val ch = data(pos)
        pos += 1
        ch
      }

      def hasNextChar = pos < fl
      def done = close(fd)
    }

    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {

      val startTime = get_time()

      val scanner = new Scanner("graham.txt")
      val training_data = scanner.data
      val data_size = scanner.fl

      val translated_data = NewArray[Int](data_size)
      for (i <- (0 until data_size)) { translated_data(i) = Encoding.char_to_ix(training_data(i)) }
      val seq_length = 20
      val vocab_size = 26
      val hiddenSize = 50
      val batchSize = 20

      val RNN = DynamicRNNFix(VanillaRNNCell(inputSize = 26, hiddenSize = 50, outputSize = 26))
      // val RNN = DynamicRNN(VanillaRNNCell(inputSize = 26, hiddenSize = 50, outputSize = 26))
      val opt = Adagrad(RNN, learning_rate = 1e-1f, gradClip = 5.0f)

      def oneHot(input: Rep[Array[Int]]): TensorR = {
        val res = Tensor.zeros(seq_length, batchSize, vocab_size)
        for (i <- 0 until seq_length: Rep[Range]) {
          for (j <- 0 until batchSize: Rep[Range])
            res.data(i * vocab_size * batchSize + j * vocab_size + input(j * seq_length + i)) = 1.0f
        }
        TensorR(res)
      }

      def lossFun(input: Rep[Array[Int]], target: Rep[Array[Int]]) = { (dummy: TensorR) =>
        val res: ArrayBuffer[TensorR] = RNN(oneHot(input), target, lengths = None)  // returns an ArrayBuffer[TensorR]
        res.head.sum()
        // val resCon = res.head.concat(0, res.tail.toSeq: _*)
        // //val resCon = res.init.head.concat(0, res.init.tail.toSeq: _*)
        // resCon.logSoftmaxB().nllLossB(target).sum()
      }

      val loss_save = NewArray[Double](51)
      val loopStartTime = get_time()

      val addr = getMallocAddr() // remember current allocation pointer here

      val startAt = var_new[Int](0)
      startAt -= seq_length * batchSize

      for (n <- (0 until 5001): Rep[Range]) {

        startAt += seq_length * batchSize
        if (startAt + seq_length * batchSize + 1 >= data_size) {
          startAt = 0
        }

        val inputs = NewArray[Int](seq_length * batchSize)
        val targets = NewArray[Int](seq_length * batchSize)
        for (i <- (0 until seq_length * batchSize): Rep[Range]) {
          inputs(i) = translated_data(startAt+i)
          targets(i) = translated_data(startAt+i+1)
        }

        val loss = gradR_loss(lossFun(inputs, targets))(Tensor.zeros(1))
        val loss_value = loss.data(0) // we suppose the loss is scala (Tensor of size 1)
        if (n % 100 == 0) {
          printf("iter %d, loss %f\\n", n, loss_value)
          loss_save(n / 100) = loss_value
        }

        opt.step()

        resetMallocAddr(addr)  // reset malloc_addr to the value when we remember allocation pointer
      }

      val loopEndTime = get_time()
      val prepareTime = loopStartTime - startTime
      val loopTime    = loopEndTime - loopStartTime

      val fp = openf(a, "w")
      fprintf(fp, "unit: %s\\n", "100 iteration")
      for (i <- (0 until loss_save.length): Rep[Range]) {
        fprintf(fp, "%lf\\n", loss_save(i))
      }
      fprintf(fp, "run time: %lf %lf\\n", prepareTime, loopTime)
      closef(fp)

    }
  }

  def main(args: Array[String]) {
    val min_char_rnn_file = new PrintWriter(new File(root_dir2 + file_dir))
    min_char_rnn_file.println(min_char_rnn_module.code)
    min_char_rnn_file.flush()
  }
}
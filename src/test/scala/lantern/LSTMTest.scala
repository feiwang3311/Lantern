package lantern

import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

import scala.collection.mutable.ArrayBuffer
import scala.math._

import org.scalatest.FunSuite

import java.io.PrintWriter;
import java.io.File;


class LSTMTest extends FunSuite {

  val root_dir = "src/out/ICFP18evaluation/"
  val file_dir = "evaluationLSTM/Lantern.cpp"
  val root_dir2 = "src/out/NIPS18evaluation/"

  val min_char_lstm = new LanternDriverC[String, Unit] {

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

      val vocab_size = 26
      val hidden_size = 50
      val learning_rate = 1e-1f
      val seq_length = 20

      // initialize all parameters:
      val Wfh = Tensor.randn(hidden_size, hidden_size, 0.01f)
      val Wfx = Tensor.randn(hidden_size, vocab_size, 0.01f)
      val bf  = Tensor.zeros(hidden_size)
      val Wih = Tensor.randn(hidden_size, hidden_size, 0.01f)
      val Wix = Tensor.randn(hidden_size, vocab_size, 0.01f)
      val bi  = Tensor.zeros(hidden_size)
      val Wch = Tensor.randn(hidden_size, hidden_size, 0.01f)
      val Wcx = Tensor.randn(hidden_size, vocab_size, 0.01f)
      val bc  = Tensor.zeros(hidden_size)
      val Woh = Tensor.randn(hidden_size, hidden_size, 0.01f)
      val Wox = Tensor.randn(hidden_size, vocab_size, 0.01f)
      val bo  = Tensor.zeros(hidden_size)
      val Why = Tensor.randn(vocab_size, hidden_size, 0.01f)  // hidden to output
      val by  = Tensor.zeros(vocab_size)

      val hprev = Tensor.zeros(hidden_size)
      val cprev = Tensor.zeros(hidden_size)
      val hsave = Tensor.zeros_like(hprev)
      val csave = Tensor.zeros_like(cprev)

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
      def lossFun(inputs: Rep[Array[Int]], targets: Rep[Array[Int]]) = { (dummy: TensorR) =>

        val loss = TensorR(Tensor.zeros(1))
        val in = ArrayBuffer[TensorR]()

        in.append(loss)
        in.append(thprev)
        in.append(tcprev)

        val outputs = LOOPSM(in)(inputs.length){i => t =>

          // get input as one-hot tensor
          val x = Tensor.zeros(vocab_size)
          x.data(inputs(i)) = 1
          val x1 = TensorR(x)
          // get output as one-hot tensor
          val y = Tensor.zeros(vocab_size)
          y.data(targets(i)) = 1
          val y1 = TensorR(y)

          val ft = (tWfh.dot(t(1)) + tWfx.dot(x1) + tbf).sigmoid()
          val it = (tWih.dot(t(1)) + tWix.dot(x1) + tbi).sigmoid()
          val ot = (tWoh.dot(t(1)) + tWox.dot(x1) + tbo).sigmoid()
          val Ct = (tWch.dot(t(1)) + tWcx.dot(x1) + tbc).tanh()
          val ct = ft * t(2) + it * Ct
          val ht = ot * ct.tanh()
          val et = (tWhy.dot(ht) + tby).exp()
          val pt = et / et.sum()
          val loss = t(0) - (pt dot y1).log()

          val out = ArrayBuffer[TensorR]()
          out.append(loss)
          out.append(ht)
          out.append(ct)
          out
        }
        hsave.copy_data(outputs(1).x)     // save the hidden state with the result from LOOP
        csave.copy_data(outputs(2).x)     // save the cell state with the result from LOOP
        outputs(0)                        // return the final loss
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

      val loopStart = get_time()
      val loss_save = NewArray[Double](51)

      val addr = getMallocAddr() // remember current allocation pointer here

      val startAt = var_new[Int](0)
      startAt -= seq_length

      var smooth_loss = 70.0
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
        smooth_loss = smooth_loss * 0.9 + loss_value * 0.1
        if (n % 100 == 0) {
          printf("iter %d, loss %f\\n", n, smooth_loss)
          loss_save(n / 100) = smooth_loss
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
        thprev.x.copy_data(hsave)
        tcprev.x.copy_data(csave)

        resetMallocAddr(addr)  // reset malloc_addr to the value when we remember allocation pointer
      }

      val loopEndTime = get_time()
      val prepareTime = loopStart - startTime
      val loopTime    = loopEndTime - loopStart

      val fp = openf(a, "w")
      fprintf(fp, "unit: %s\\n", "100 iteration")
      for (i <- (0 until loss_save.length): Rep[Range]) {
        //printf("loss_saver is %lf \\n", loss_save(i))
        fprintf(fp, "%lf\\n", loss_save(i))
      }
      fprintf(fp, "run time: %lf %lf\\n", prepareTime, loopTime)
      closef(fp)

    }
  }

  test("generate_code_for_lstm") {
    val min_char_lstm_file = new PrintWriter(new File(root_dir + file_dir))
    min_char_lstm_file.println(min_char_lstm.code)
    min_char_lstm_file.flush()
  }

  val min_char_lstm_module = new LanternDriverC[String, Unit] with NNModule {

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
      printf("LSTM Test: >> data has %d chars\\n", data_size)

      val translated_data = NewArray[Int](data_size)
      for (i <- (0 until data_size)) { translated_data(i) = Encoding.char_to_ix(training_data(i)) }

      val vocab_size = 26
      val hidden_size = 50
      val learning_rate = 1e-1f
      val seq_length = 20
      val batchSize = 20

      val RNN = DynamicRNNFix(LSTMCell(inputSize = 26, hiddenSize = 50, outputSize = 26))
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
      }

      val loopStart = get_time()
      val loss_save = NewArray[Double](51)

      val addr = getMallocAddr() // remember current allocation pointer here

      val startAt = var_new[Int](0)
      startAt -= seq_length * batchSize

      var smooth_loss = 70.0
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
      val prepareTime = loopStart - startTime
      val loopTime    = loopEndTime - loopStart

      val fp = openf(a, "w")
      fprintf(fp, "unit: %s\\n", "100 iteration")
      for (i <- (0 until loss_save.length): Rep[Range]) {
        fprintf(fp, "%lf\\n", loss_save(i))
      }
      fprintf(fp, "run time: %lf %lf\\n", prepareTime, loopTime)
      closef(fp)
    }
  }

  test("generate_code_for_lstm_module") {
    val min_char_lstm_file = new PrintWriter(new File(root_dir2 + file_dir))
    min_char_lstm_file.println(min_char_lstm_module.code)
    min_char_lstm_file.flush()
  }
}
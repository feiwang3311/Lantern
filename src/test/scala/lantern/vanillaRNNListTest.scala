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

class VanillaRNNList extends FunSuite {

  val file_dir = "/tmp/vanilla_rnn_list.cpp/"

  val min_char_list = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

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
      val scanner = new Scanner("test_data")
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
        val outputs = LOOPLM(0)(in)(inputs.length){i => t =>

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

      val addr = getMallocAddr() // remember current allocation pointer here

      val startAt = var_new[Int](0)
      startAt -= seq_length

      val timer = Timer()
      timer.startTimer

      for (n <- (0 until 2001): Rep[Range]) {

        startAt += seq_length
        if (startAt + seq_length + 1 >= data_size) {
          startAt = 0
          hprev.clear()
        }

        val inputs = NewArray[Int](seq_length)
        val targets = NewArray[Int](seq_length)
        for (i <- (0 until seq_length): Rep[Range]) {
          inputs(seq_length-1-i) = translated_data(startAt+i)
          targets(seq_length-1-i) = translated_data(startAt+i+1)
        }

        val loss = gradR_loss(lossFun(inputs, targets))(Tensor.zeros(1))
        val loss_value = loss.data(0) // we suppose the loss is scala (Tensor of size 1)
        if (n % 100 == 0) {
          printf("iter %d, loss %f\\n", n, loss_value)
          timer.printElapsedTime
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

    }
  }

  test("generate_code_for_vanilla_rnn_list(traverse_data_in_reverse_order)") {
    //println("generate code for VanillaRNNList")
    val min_char_list_file = new PrintWriter(new File(file_dir))
    min_char_list_file.println(min_char_list.code)
    min_char_list_file.flush()  
    //println("now your code at $file_dir is generated.")
  }
  
}
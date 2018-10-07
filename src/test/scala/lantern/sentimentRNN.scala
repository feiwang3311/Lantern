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

class SentimentTreeRNN extends FunSuite {

  val file_dir = "/tmp/sentiment_tree_rnn.cpp"

	val sentimental_rnn = new LanternDriverC[String, Unit] with ScannerLowerExp {

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

      // read in the data for trees
      val tree_number = 1101 // need to know the size of training data, need fix
      val fp1 = openf("senti/array_tree.txt", "r")
      val tree_data = NewArray[Array[Int]](tree_number * 4) // each tree data has 4 lines (score, word, lch, rch)

      val size = NewArray[Int](1)
      for (i <- (0 until tree_number): Rep[Range]) {
        getInt(fp1, size, 0)
        for (j <- (0 until 4): Rep[Range]) {
          tree_data(i * 4 + j) = NewArray[Int](size(0))
          for (k <- (0 until size(0)): Rep[Range]) getInt(fp1, tree_data(i * 4 + j), k)
        }
      }

      /* // this piece of code proves that the data reading is correct
      for (j <- (0 until 4): Rep[Range]) {
        val barray = tree_data(j)
        for (k <- (0 until size(0)): Rep[Range]) printf("%d ", barray(k))
        printf("\\n")
      }

      val carray = tree_data(1)
      for (j <- (0 until size(0)):Rep[Range]) {
        if (carray(j) > 0) {
          val darray = word_embedding_data(carray(j))
          for (t <- (0 until word_embedding_size): Rep[Range]) printf("%lf ", darray(t))
          printf("\\n")
        }
      }*/


     // set up hyperparameters and parameters
     val hidden_size = 100
     val output_size = 5
     val learning_rate = 0.05f
     val Wxh = Tensor.randinit(hidden_size, word_embedding_size, 0.01f) // from word embedding to hidden vector
     val bx  = Tensor.zeros(hidden_size)                               // bias word embedding to hidden vector
     val Wlh = Tensor.randinit(hidden_size, hidden_size, 0.01f)         // from hidden vector of left child to hidden
     val Wrh = Tensor.randinit(hidden_size, hidden_size, 0.01f)         // from hidden vector of right child to hidden
     val bh  = Tensor.zeros(hidden_size)                               // bias from children hidden vector to hidden
     val Why = Tensor.randinit(output_size, hidden_size, 0.01f)         // from hidden vector to output
     val by  = Tensor.zeros(output_size)                               // bias hidden vector to output

     // Cast Tensors as Tensors
     val Wxh1 = TensorR(Wxh)
     val bx1  = TensorR(bx)
     val Wlh1 = TensorR(Wlh)
     val Wrh1 = TensorR(Wrh)
     val bh1  = TensorR(bh)
     val Why1 = TensorR(Why)
     val by1  = TensorR(by)

     def lossFun(scores: Rep[Array[Int]], words: Rep[Array[Int]], lchs: Rep[Array[Int]], rchs: Rep[Array[Int]]) = { (dummy: TensorR) =>

       val initial_loss = TensorR(Tensor.zeros(1))
       val initial_hidd = TensorR(Tensor.zeros(hidden_size))
       val inBuffer     = new ArrayBuffer[TensorR]()
       inBuffer.append(initial_loss); inBuffer.append(initial_hidd) // construct the input to LOOPTM

       val outBuffer = LOOPTM(0)(inBuffer)(lchs, rchs) { (l: ArrayBuffer[TensorR], r: ArrayBuffer[TensorR], i: Rep[Int]) =>

         val targ = Tensor.zeros(output_size); targ.data(scores(i)) = 1; val targ1 = TensorR(targ)
         val lossl = l(0); val hiddenl = l(1)
         val lossr = r(0); val hiddenr = r(1)

         val hidden = IF (lchs(i) < 0) { // leaf node
           val embedding_array = word_embedding_data(words(i))
           val embedding_tensor = TensorR(Tensor(embedding_array, word_embedding_size))
           (Wxh1.dot(embedding_tensor) + bx1).tanh()
         } { (Wlh1.dot(hiddenl) + Wrh1.dot(hiddenr) + bh1).tanh() } // non-leaf node
         val pred1 = (Why1.dot(hidden) + by1).exp()
         val pred2 = pred1 / pred1.sum()
         val loss = lossl + lossr - (pred2 dot targ1).log()
         val out = ArrayBuffer[TensorR]()
         out.append(loss)
         out.append(hidden)
         out
       }
       outBuffer(0)
     }

     val lr = learning_rate
     val hp = 1e-8f

     val mWxh = Tensor.zeros_like(Wxh)
     val mbx  = Tensor.zeros_like(bx)
     val mWlh = Tensor.zeros_like(Wlh)
     val mWrh = Tensor.zeros_like(Wrh)
     val mbh  = Tensor.zeros_like(bh)
     val mWhy = Tensor.zeros_like(Why)
     val mby  = Tensor.zeros_like(by)

     val addr = getMallocAddr() // remember current allocation pointer here

     for (epoc <- (0 until 10): Rep[Range]) {

       var ave_loss = 0.0
       for (n <- (0 until tree_number): Rep[Range]) {

         val index = n % tree_number
         val scores   = tree_data(index * 4)
         val words    = tree_data(index * 4 + 1)
         val leftchs  = tree_data(index * 4 + 2)
         val rightchs = tree_data(index * 4 + 3)
         val loss = gradR_loss(lossFun(scores, words, leftchs, rightchs))(Tensor.zeros(1))
         val loss_value = loss.data(0)  // we suppose the loss is scala (Tensor of size 1)
         ave_loss = ave_loss * n / (n + 1) + loss_value / (n + 1)

         val pars = ArrayBuffer(Wxh1, bx1, Wlh1, Wrh1, bh1, Why1, by1)
         val mems = ArrayBuffer(mWxh, mbx, mWlh, mWrh, mbh, mWhy, mby)
         for ((par, mem) <- pars.zip(mems)) {
           par.clip_grad(1.0f)
           mem += par.d * par.d
           par.x -= par.d * lr / (mem + hp).sqrt()
           par.clear_grad()
         }

         resetMallocAddr(addr)  // reset malloc_addr to the value when we remember allocation pointer */
       }

       printf("epoc %d, ave_loss %f\\n", epoc, ave_loss)
     }

    }
  }

  test("generate_code_for_sentiment_tree_rnn"){
  	//println("generate code for sentimental_tree_rnn")
  	val senti_file = new PrintWriter(new File(file_dir))
    senti_file.println(sentimental_rnn.code)
    senti_file.flush()
    //println("now your code at $file_dir is generated.")
  }
}
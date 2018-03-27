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

class AdLMSVectorTest extends FunSuite {

  if (false) {
    val array0 = new DslDriverC[String, Unit] with TensorExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val addr = getMallocAddr()
        //printf("address is at %ld \\n", addr)
        resetMallocAddr(addr)
        //printf("now lets use some memory\\n")
        val mem = Tensor.zeros(100)
        val addr1 = getMallocAddr()
        //printf("Now address is at %ld \\n", addr1)
        resetMallocAddr(addr)
        val addr2 = getMallocAddr()
        //printf("after reset, the address is back to %ld\\n", addr2)

        //assertions
        if (addr + 800 != addr1) printf("ERROR: addr did not increase by 800")
        if (addr != addr2) printf("ERROR: addr did not reset to the give value")
        // unchecked[Unit](s"assert($addr1 == $addr + 800)")
      //assert (addr1 == addr + 800l, "addr did not increase by 800")
    //assert (addr == addr2, "addr did not reset to the given value")
      }
    }

    val array0_file = new PrintWriter(new File("array0.cpp"))
    array0_file.println(array0.code)
    array0_file.flush()
    println("run test case array0")
    array0.eval("abc")

    val array1 = new DslDriverC[String, Unit]  with TensorExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val res = Tensor.randinit(length)
        val res2 = Tensor.randinit(length, seed = Some(5))
        //res.print()
        //res2.print()

        val result = res dot res2
        //result.print()

        // assertions
        if (res.data(0) * res2.data(0) + res.data(1) * res2.data(1) != result.data(0))
          println("ERROR: the dot product of two vectors is not correct")

      }
    }

    //println("test dot")
    //val array1_file = new PrintWriter(new File("array1(2).cpp"))
    //array1_file.println(array1.code)
    //array1_file.flush()
    //println(array1.code)
    println("run test case array1")
    array1.eval("abc")

    val array1_1 = new DslDriverC[String, Unit] with TensorExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val dim0 = 2
        val dim1 = 3
        val matrix = Tensor.rand(dim0, dim1)
        val vector = Tensor.randinit(dim1, seed = Some(4))
        //matrix.print()
        //vector.print()

        //println("the result is:")
        val result = matrix dot vector
        //result.print()

        if (matrix(0, 0) * vector(0) + matrix(0, 1) * vector(1) + matrix(0, 2) * vector(2) != result(0))
          printf("ERROR: the matrix vector dot product is wrong on the first element of result, %.3f != %.3f\\n", matrix(0, 0) * vector(0) + matrix(0, 1) * vector(1) + matrix(0, 2) * vector(2), result(0))
        if (matrix(1, 0) * vector(0) + matrix(1, 1) * vector(1) + matrix(1, 2) * vector(2) != result(1))
          printf("ERROR: the matrix vector dot product is wrong on the second element of result, %.3f != %.3f\\n", matrix(1, 0) * vector(0) + matrix(1, 1) * vector(1) + matrix(1, 2) * vector(2), result(1))
      }
    }

    //println(array1_1.code)
    println("run test case array1_1")
    array1_1.eval("abc")

    val array2 = new DslDriverC[String, Unit] with TensorExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        // read training data from file (for now just use random)
        val length = 2
        val v = Tensor.randinit(length)
        //v.print()

        // calculate gradient
        val grad = gradR(t => t dot t)(v)
        // show gradient
        //println("show gradient in the traditional way")
        //grad.print()

        // assertions
        Tensor.assertEqual(v * 2.0f, grad)

        // construct TensorR for closure
        val tv = TensorR(v)
        val loss = gradR_loss(dummy => tv dot tv)(Tensor.zeros(1))
        //println("gradient:")
        //tv.d.print()
        //println("loss")
        //loss.print()
        // assertions
        Tensor.assertEqual((v dot v), loss)
        Tensor.assertEqual(tv.d, grad)
        ()
      }
    }

    //println("test dot gradient")
    //println(array2.code)
    println("run test case array2")
    array2.eval("2.0f")

    val array2_1 = new DslDriverC[String, Unit] with TensorExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        val dim0 = 2
        val vector = Tensor.randinit(dim0, seed = Some(4))

        // initialize tensors for closure
        val ve = new TensorR(vector, Tensor.zeros(dim0))
        val half = new TensorR(Tensor.halves(dim0), Tensor.zeros(dim0))

        // define function of model
        def model(dummy: TensorR): TensorR @diff = {
          ((ve dot ve) * half).sum()
        }
        val loss = gradR_loss(model)(Tensor.zeros(1))
        Tensor.assertEqual(loss, ((vector dot vector) * Tensor.halves(dim0)).sum(), "1")
        Tensor.assertEqual(ve.d, vector * 2.0f ,"2")
        Tensor.assertEqual(half.d, Tensor.fill((vector dot vector).data(0), 2), "3")
        ()
      }
    }

    println("run test case array2_1")
    array2_1.eval("abc")

    val array2_2 = new DslDriverC[String, Unit] with TensorExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        val dim0 = 2
        val dim1 = 3
        val matrix = Tensor.rand(dim0, dim1)
        val vector = Tensor.randinit(dim1, seed = Some(4))

        // initialize tensors for closure
        val ma = TensorR(matrix)
        val ve = TensorR(vector)

        // define function of model
        def model(dummy: TensorR): TensorR @diff = {
          (ma dot ve).sum()
        }
        val loss = gradR_loss(model)(Tensor.zeros(1))
        Tensor.assertEqual(loss, (matrix dot vector).sum(), "11")
        Tensor.assertEqual(ma.d, Tensor.expand(vector, dim0), "12")
        val sol = matrix.sumOnDim1()
        Tensor.assertEqual(ve.d, sol, "13")
        ()
      }
    }

    // println("test matrix vector dot gradient as side effect")
    //println(array2_2.code)
    println("run test case array2_2")
    array2_2.eval("abc")


    val testTrans = new DslDriverC[String, Unit] with TensorExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val idx = var_new(0)
        val t = Tensor.fill(seq => { idx += 1; idx }, 2, 3)

        Tensor.assertEqual(t.trans(), Tensor.fromData(1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f).resize(3, 2), "Transpose invalid")
      }
    }
    println("run test trans")
    testTrans.eval("abs")


    val array2_3 = new DslDriverC[String, Unit] with TensorExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        val vocab_size = 3
        val hidden_size = 10
        val Wxh = Tensor.randinit(hidden_size, vocab_size, 0.1f)  // input to hidden
        val Whh = Tensor.randinit(hidden_size, hidden_size, 0.1f) // hidden to hidden
        val Why = Tensor.randinit(vocab_size, hidden_size, 0.1f)  // hidden to output
        val bh  = Tensor.randinit(hidden_size)
        val by  = Tensor.randinit(vocab_size)
        val hprev = Tensor.randinit(hidden_size)

        val hprev_next = Tensor.zeros_like(hprev) // this vector catches the new hidden value, see the NOTE below
        /*
        NOTE: initially I simply updated hprev with new hidden value. That turned out to be a bug.
        Syntactically I updated hprev after the LOOPCCM cycle, but because we are constructing a static computation graph with continuations,
        symantically the update happens before the end of the forward propagation.

        So instead of updating hprev after autodifferentiation, I updated it before autodifferentiation.
        That is a easily fallen pitfall.

        NEED to think about how to avoid it or send WARNING for code like this!!

        The solution is to copy it to an independent vector. MAYBE need better solutions?
        */

       // wrap as tensors
       val Wxh1 = TensorR(Wxh)
       val Whh1 = TensorR(Whh)
       val Why1 = TensorR(Why)
       val bh1  = TensorR(bh)
       val by1  = TensorR(by)
       val hprev1 = TensorR(hprev)

       // encode input and output
       val x_data = NewArray[Int](3); x_data(0) = 0; x_data(1) = 1; x_data(2) = 2
       val y_data = NewArray[Int](3); y_data(0) = 2; y_data(1) = 0; y_data(2) = 1
       //val x_data = mutableStaticData(scala.Array(0, 1, 2))
       //val y_data = mutableStaticData(scala.Array(2, 0, 1))

       // our method of loss and gradient calculation
       def lossFun: (TensorR => TensorR @diff) = { (dummy: TensorR) =>
         val loss = TensorR(Tensor.zeros(1))
         val in = ArrayBuffer[TensorR]()
         in.append(loss)
         in.append(hprev1)
         val outputs = LOOPSM(in)(1) { i => t =>

           // get input as one-hot tensor
           val x = Tensor.zeros(vocab_size)
           x.data(x_data(i)) = 1
           val x1 = TensorR(x)
           // get output as one-hot tensor
           val y = Tensor.zeros(vocab_size)
           y.data(y_data(i)) = 1
           val y1 = TensorR(y)

           val tmp = (Wxh1 dot x1)
           val h1 = (tmp + (Whh1 dot t(1)) + bh1).tanh() // use hidden state and x1 to compute hidden state
           val e1 = (Why1.dot(h1) + by1).exp()                       // use new hidden state to compute unnormalized prob
           val p1 = e1 / e1.sum()                            // use unnormalized prob to compute normalize prob
           generate_comment("Compute new loss")
           val newloss = t(0) - (p1 dot y1).log()            // loss is updated by original loss t(0) and additional loss
           generate_comment("Done computing loss")
           val out = ArrayBuffer[TensorR]()

           out.append(newloss)
           out.append(h1)
           out
         }
         hprev_next.copy_data(outputs(1).x)  // update the hidden state with the result from LOOP
         outputs(0)                          // return the final loss
       }
       val loss1 = gradR_loss(lossFun)(Tensor.zeros(1))
       printf("bh1\\n")
       bh1.d.printRaw(hidden_size)

       generate_comment("Compute real value")


       // correct method of loss and gradient calculation, adapting from Numpy
       // preset space for gradients
       val dWxh = Tensor.zeros_like(Wxh)
       val dWhh = Tensor.zeros_like(Whh)
       val dWhy = Tensor.zeros_like(Why)
       val dbh  = Tensor.zeros_like(bh)
       val dby  = Tensor.zeros_like(by)
       val dhnext = Tensor.zeros_like(hprev)
       val sum_loss = Tensor.zeros(1)
       val hprev_new = Tensor.zeros_like(hprev)

       def lossOneCycle(i: Int, hprev: Tensor): Unit = {

         // get input as one-hot tensor
         val x = Tensor.zeros(vocab_size)
         x.data(x_data(i)) = 1
         // get output as one-hot tensor
         val y = Tensor.zeros(vocab_size)
         y.data(y_data(i)) = 1

         // forward pass
         val tmp = (Wxh dot x)
         val hs = (tmp + (Whh dot hprev) + bh).tanh()
         val ys = (Why dot hs) + by
         val ye = ys.exp()
         val ps = ye / ye.sum()
         sum_loss -= (ps dot y).log()

         if (i < 0) lossOneCycle(i + 1, hs)
         else hprev_new.copy_data(hs)

         // backward pass
         val dy = Tensor.copy(ps)
         dy.data(y_data(i)) -= 1
         dWhy += (dy cart hs)
         dby += dy
         val dh = (Why.trans() dot dy) + dhnext
         val dhraw = (Tensor.ones(1) - hs * hs) * dh
         dbh += dhraw
         dWxh += (dhraw cart x)
         dWhh += (dhraw cart hprev)
         dhnext.copy_data(Whh.trans() dot dhraw)
         ()
       }

       lossOneCycle(0, hprev)

       // assertions
       Tensor.assertEqual(loss1, sum_loss, "loss")
       Tensor.assertEqual(hprev_next, hprev_new, "hidden")
       Tensor.assertEqual(Wxh1.d, dWxh, "dWxh")
       Tensor.assertEqual(Whh1.d, dWhh, "dWhh")
       Tensor.assertEqual(Why1.d, dWhy, "dWhy")
       Tensor.assertEqual(bh1.d, dbh, "dbh")
       Tensor.assertEqual(by1.d, dby, "dby")
      }
    }

    println("try array2_3")
    val array2_3file = new PrintWriter(new File("array2_3.cpp"))
    array2_3file.println(array2_3.code)
    array2_3file.flush()
    println("run test case array2_3")
    array2_3.eval("abc")

    val array2_4 = new DslDriverC[String, Unit] with TensorExp {

      @virtualize
      def snippet (a: Rep[String]): Rep[Unit] = {
        val vocab_size = 3
        val by   = Tensor.zeros(vocab_size)
        val by1  = TensorR(by)
        val y = Tensor.zeros(vocab_size)
        y.data(1) = 1
        val y1 = TensorR(y)

        def lossFun = { (dummy: TensorR) =>

          val e1 = (by1).exp()
          val p1 = e1 / e1.sum()
          (p1 dot y1).log()
        }
        val dummy = gradR(lossFun)(Tensor.zeros(1))
        // by1.d.print()


        // FIXME: need a correct implementation of gradient to check with
      }
    }

    //println("try array2_2_4")
    println("run test case array2_4")
    array2_4.eval("abc")

    val array2_5 = new DslDriverC[String, Unit] with TensorExp {

      @virtualize
      def snippet (a: Rep[String]): Rep[Unit] = {
        val vocab_size = 3
        val e   = Tensor.ones(vocab_size)
        val e1  = TensorR(e)
        val a   = Tensor.ones(vocab_size)
        val a1  = TensorR(a)
        val y = Tensor.zeros(vocab_size)
        y.data(1) = 1
        val y1 = TensorR(y)

        def lossFun = { (dummy: TensorR) =>
          //e1.sum()
          val p1 = a1 / e1.sum()
          (p1 dot y1).log()
        }
        val dummy = gradR(lossFun)(Tensor.zeros(1))
        //e1.d.print()
        //a1.d.print()

        // FIXME: need a correct implementation of gradient to check with
      }
    }
    //println("try array2_2_5")
    println("run test case array2_5")
    array2_5.eval("abc")

    val array3 = new DslDriverC[String, Unit] with TensorExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        // use random array as input
        val length = 2
        val v = Tensor.randinit(length)
        //v.print()

        // calcuate gradient
        val grad = gradR(t => {val y = IF (length)(t.x.data(0) > 0.0f) {t + t}{t * t}
        y.sum() })(v)
        // show gradient
        //grad.print()

        // another way of implementing it
        val grad1 = gradR(t => (t + t).sum())(v)
        val grad2 = gradR(t => (t * t).sum())(v)
        if (v(0) > 0) Tensor.assertEqual(grad, grad1)
        else Tensor.assertEqual(grad, grad2)
      }
    }

    //println("test IF gradient")
    val array3_file = new PrintWriter(new File("array3.cpp"))
    array3_file.println(array3.code)
    array3_file.flush()
    println("run test case array3")
    array3.eval("abc")

    val array4 = new DslDriverC[String, Unit] with TensorExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        // use random array as input
        val length = 2
        val v = Tensor.randinit(length)
        // v.print()

        val halfv = Tensor.halves(length)
        val half = (new TensorR(halfv, Tensor.zeros(length)))
        // calculate gradient
        val grad = gradR(t => {val y = LOOP(t)(t => t.x.data(0) > 0.1f)(t => t * half)
        y.sum() })(v)
        // show gradient
        grad.print()
        //println("Tensor in closure can also accumulate gradient, which is important")
        half.d.print()

        // FIXME: Implement the correct gradient and assertEqual
      }
    }

    // println("test LOOP gradient")
    //println(array4.code)
    val parray4 = new PrintWriter(new File("array4.cpp"))
    parray4.println(array4.code)
    parray4.flush()
    println("run test case array4")
    array4.eval("abc")

    val array4_1 = new DslDriverC[String, Unit] with TensorExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randinit(length)
        // v.print()

        val half = new TensorR(Tensor.halves(length), Tensor.zeros(length))
        val grad = gradR(t => {
          val y = LOOPS(t)(3)(i => t => t * half )
          y.sum()
        })(v)
        // show gradient
        //grad.print()
        //println("Tensor in closure can also accumulate gradient, which is important")
        //half.d.print()

        val save_half_grad = Tensor.zeros(length)
        save_half_grad.copy_data(half.d)

        // alternative implementation
        half.d.clear()
        val grad2 = gradR( t => {
          (t * half * half * half).sum()
        })(v)

        // assertion
        Tensor.assertEqual(grad, grad2)
        Tensor.assertEqual(save_half_grad, half.d)
      }
    }

    // println("test LOOP gradient")
    println("run test case array4_1")
    array4_1.eval("abc")

    // test using array data by closure
    val array4_2 = new DslDriverC[String, Unit] with TensorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {

        // random initialization
        val length = 3
        val v = Tensor.randinit(length)
        // v.print()

        // get data from "file" (more like generate static data and lift it to Rep type)
        val ddim0 = 2
        val ddim1 = 3
        val data1 = NewArray[Float](ddim1)
        val data2 = NewArray[Float](ddim1)
        for (i <- (0 until ddim1): Rep[Range]) {
          data1(i) = (i + 1)
          data2(i) = (i + 1) * 2
        }
        val data = NewArray[Array[Float]](ddim0)
        data(0) = data1; data(1) = data2

        val model: TensorR => TensorR @diff = { (x: TensorR) =>
          val y = LOOPS(x)(ddim0)(i => x1 => {
            val data_point = TensorR(Tensor(data(i), ddim1))
            x1 * data_point
          })
          y.sum()
        }

        val grad = gradR(model)(v)
        // show gradient
        //grad.print()

        // alternative implememetation
        val grad1 = gradR(t =>
            (t * TensorR(Tensor(data(0), ddim1)) * TensorR(Tensor(data(1), ddim1))).sum()
            )(v)
        // assertion
        Tensor.assertEqual(grad, grad1)
      }
    }

    //println(array4_2_1.code)
    //val array4_2_1_file = new PrintWriter(new File("array4_2_1.cpp"))
    //array4_2_1_file.println(array4_2_1.code)
    //array4_2_1_file.flush()
    println("run test case of array4_2")
    array4_2.eval("abc")

    val array4_4 = new DslDriverC[String, Unit] with TensorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randinit(length)
        //v.print()
        val u = Tensor.randinit(length, seed = Some(5))
        //u.print()

        val half = new TensorR(Tensor.halves(length), Tensor.zeros(length))
        val vv = TensorR(v)
        val uu = TensorR(u)

        val dummy = gradR(dum => {
          val in = ArrayBuffer[TensorR](vv, uu)
          val y = LOOPSM(in)(3)(i => ins => {
            val vvv = ins(0) * half
            val uuu = ins(1) * half
            ArrayBuffer[TensorR](vvv, uuu)
          })
        y(1).sum() + y(0).sum()})(Tensor.zeros(1))
        // show gradient
        //println("Tensor in closure can also accumulate gradient, which is important")
        //half.d.print()
        //vv.d.print()
        //uu.d.print()

        // save gradients
        val save_vv_grad = Tensor.zeros(length); save_vv_grad.copy_data(vv.d);   vv.clear_grad()
        val save_uu_grad = Tensor.zeros(length); save_uu_grad.copy_data(uu.d);   uu.clear_grad()
        val save_ha_grad = Tensor.zeros(length); save_ha_grad.copy_data(half.d); half.clear_grad()

        // alternative implementation
        val dummy1 = gradR(dum => {
          (vv * half * half * half + uu * half * half * half).sum()
        })(Tensor.zeros(1))

        // assertions
        Tensor.assertEqual(save_ha_grad, half.d)
        Tensor.assertEqual(save_vv_grad, vv.d)
        Tensor.assertEqual(save_uu_grad, uu.d)
      }
    }

    //println("support 2 tensors in loop using LOOPCCM")
    //println(array4_4.code)
    //val array4_4_file = new PrintWriter(new File("array4_4.cpp"))
    //array4_4_file.println(array4_4.code)
    //array4_4_file.flush()
    println("run test case in array4_4")
    array4_4.eval("abc")

    val array5 = new DslDriverC[String, Unit] with TensorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randinit(length)
        //v.print()

        val grad = gradR(t => (t * t).sum())(v)
        //grad.print()

        Tensor.assertEqual(grad, v * 2.0f)
      }
    }

    //println("test elementwise multiplication")
    //println(array5.code)
    println("run test case in array5")
    array5.eval("abc")

    val array6 = new DslDriverC[String, Unit] with TensorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randinit(length)
        //v.print()

        val grad = gradR(t => (t / t).sum())(v)
        //grad.print()

        Tensor.assertEqual(grad, Tensor.zeros(length))
      }
    }

    // println("test elementwise division")
    //println(array6.code)
    println("run test case in array6")
    array6.eval("abc")

    val array7 = new DslDriverC[String, Unit] with TensorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randinit(length)
        //v.print()

        val grad = gradR(t => (t.tanh()).sum())(v)
        //grad.print()

        val e1 = v.tanh();
        val ee = Tensor.ones(length) - e1 * e1
        Tensor.assertEqual(grad, ee)
      }
    }

    // println("test tanh")
    //println(array7.code)
    println("run test case array7")
    array7.eval("abc")

    val array7_1 = new DslDriverC[String, Unit] with TensorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randinit(length)

        val grad = gradR(t => (t.sigmoid()).sum())(v)

        val e1 = v.sigmoid()
        val ee = (Tensor.ones(1) - e1) * e1
        Tensor.assertEqual(grad, ee)
      }
    }

    println("run test case array7_1")
    array7_1.eval("abc")

    val array8 = new DslDriverC[String, Unit] with TensorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randinit(length)
        // v.print()

        val grad = gradR(t => (t.exp()).sum())(v)
        //grad.print()

        Tensor.assertEqual(grad, v.exp())
      }
    }

    // println("test exp")
    //println(array8.code)
    println("run test case in array8")
    array8.eval("abc")

    val array9 = new DslDriverC[String, Unit] with TensorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randPositive(length)
        //v.print()

        val grad = gradR(t => (t.log()).sum())(v)
        //grad.print()

        Tensor.assertEqual(grad, Tensor.ones(length) / v)
      }
    }

    //println("test log")
    // println(array9.code)
    println("run test case array9")
    array9.eval("abc")

    val array10 = new DslDriverC[String, Unit] with TensorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randinit(length)
        //v.print()

        val arra = NewArray[Array[Float]](2)
        arra(0) = NewArray[Float](2)
        arra(0)(0) = 4.0f
        arra(0)(1) = 2.0f
        arra(1) = NewArray[Float](2)
        arra(1)(0) = 1.5f
        arra(1)(1) = 2.0f

        // create a model that recursively use the data in arr (originated from list)
        def model: TensorR => TensorR @diff = { (x: TensorR) =>
          LOOPL(x)(arra.length)(i => x1 => new TensorR(Tensor(arra(i), length), Tensor.zeros(length)) * x1)
        }
        val grad = gradR(t => (model(t)).sum())(v)
        //grad.print()

        val grad1 = gradR(t =>
            (t * TensorR(Tensor(arra(0), length)) * TensorR(Tensor(arra(1), length))).sum()
            )(v)

        Tensor.assertEqual(grad, grad1)
      }
    }

    //println(array10.code)
    println("run test case in array10")
    array10.eval("abc")

    val array11 = new DslDriverC[String, Unit] with TensorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randinit(length)
        //v.print()

        /*
        5.0f, 4.0f
        /       \
        /         \
        3.0f, 2.0f   1.5f, 1.4f
        */

       val arra = NewArray[Array[Float]](3)
       arra(0) = NewArray[Float](2)
       arra(0)(0) = 5.0f; arra(0)(1) = 4.0f
       arra(1) = NewArray[Float](2)
       arra(1)(0) = 3.0f; arra(1)(1) = 2.0f
       arra(2) = NewArray[Float](2)
       arra(2)(0) = 1.5f; arra(2)(1) = 1.4f
       val lch1 = NewArray[Int](3)
       lch1(0) = 1; lch1(1) = -1; lch1(2) = -1
       val rch1 = NewArray[Int](3)
       rch1(0) = 2; rch1(1) = -1; rch1(2) = -1

       // create a model that recursively use the data (originated from tree)
       def model: TensorR => TensorR @diff = { (x: TensorR) =>
         LOOPT(x)(lch1, rch1){ (l: TensorR, r: TensorR, i: Rep[Int]) =>
           l * r * new TensorR(Tensor(arra(i), length), Tensor.zeros(length))
         }
       }

       val grad = gradR(t => model(t).sum())(v)
       //grad.print()

       def model1: TensorR => TensorR @diff = { (x: TensorR) =>
         val leftchild  = x * TensorR(Tensor(arra(1), length)) * x
         val rightchild = x * TensorR(Tensor(arra(2), length)) * x
         val root = leftchild * TensorR(Tensor(arra(0), length)) * rightchild
         root.sum()
       }

       val grad1 = gradR(model1)(v)
       // assertion
       Tensor.assertEqual(grad, grad1)
      }
    }

    //println(array11.code)
    println("run test case array11")
    array11.eval("abc")

    val array11_1 = new DslDriverC[String, Unit] with TensorExp {

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randinit(length)
        //v.print()

        /*
        5.0f, 4.0f
        /       \
        /         \
        3.0f, 2.0f   1.5f, 1.4f
        */

       val arra = NewArray[Array[Float]](3)
       arra(0) = NewArray[Float](2)
       arra(0)(0) = 5.0f; arra(0)(1) = 4.0f
       arra(1) = NewArray[Float](2)
       arra(1)(0) = 3.0f; arra(1)(1) = 2.0f
       arra(2) = NewArray[Float](2)
       arra(2)(0) = 1.5f; arra(2)(1) = 1.4f
       val lch1 = NewArray[Int](3)
       lch1(0) = 1; lch1(1) = -1; lch1(2) = -1
       val rch1 = NewArray[Int](3)
       rch1(0) = 2; rch1(1) = -1; rch1(2) = -1

       val add: TensorR = TensorR(Tensor.ones(length))

       // create a model that recursively use the data (originated from tree)
       def model: TensorR => TensorR @diff = { (x: TensorR) =>
         val in = new ArrayBuffer[TensorR](); in.append(x); in.append(add)
         val tmp = LOOPTM(in)(lch1, rch1){ (l: ArrayBuffer[TensorR], r: ArrayBuffer[TensorR], i: Rep[Int]) =>
           val curr = TensorR(Tensor(arra(i), length))
           val new_x = l(0) * r(0) * curr; val new_add = l(1) + r(1) + curr
           val out = new ArrayBuffer[TensorR](); out.append(new_x); out.append(new_add)
           out
         }
         tmp(0).sum() + tmp(1).sum()
       }

       val grad = gradR(t => model(t))(v)
       //grad.print()
       // save gradient of add
       val save_grad_add = Tensor.zeros(length); save_grad_add.copy_data(add.d); add.clear_grad()

       def model1: TensorR => TensorR @diff = { (x: TensorR) =>
         val val1 = TensorR(Tensor(arra(1), length))
         val leftchild  = x * val1 * x; val leftch = add + val1 + add
         val val2 = TensorR(Tensor(arra(2), length))
         val rightchild = x * val2 * x; val rightch = add + val2 + add
         val val0 = TensorR(Tensor(arra(0), length))
         val root = leftchild * val0 * rightchild; val root2 = leftch + val0 + rightch
         root.sum() + root2.sum()
       }

       val grad1 = gradR(model1)(v)
       // assertion
       Tensor.assertEqual(grad, grad1)
       Tensor.assertEqual(save_grad_add, add.d)
      }
    }

    //println(array11.code)
    println("run test case array11_1")
    array11_1.eval("abc")

  } // if (false) closing

  val root_dir = "src/out/ICFP18evaluation/"

  val min_char_rnn = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

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

  /*
  println("run min_char_rnn")
  val min_char_rnn_file = new PrintWriter(new File(root_dir + "evaluationRNN/Lantern.cpp"))
  min_char_rnn_file.println(min_char_rnn.code)
  min_char_rnn_file.flush()
  */

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
        val outputs = LOOPLM(in)(inputs.length){i => t =>

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

  /*
  println("run min_char_list")
  val min_char_list_file = new PrintWriter(new File(root_dir + "evaluationRNNlist/Lantern.cpp"))
  min_char_list_file.println(min_char_list.code)
  min_char_list_file.flush()
  */

  val min_char_lstm = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

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

  /*
  println("run min_char_lstm")
  val min_char_lstm_file = new PrintWriter(new File(root_dir + "evaluationLSTM/Lantern.cpp"))
  min_char_lstm_file.println(min_char_lstm.code)
  min_char_lstm_file.flush()
  */

  val senti_seq_lstm = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

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


  //println("try senti_seq_lstm")
  //val min_char_rnn_file = new PrintWriter(new File("senti_seq_lstm.cpp"))
  //min_char_rnn_file.println(senti_seq_lstm.code)
  //min_char_rnn_file.flush()
  //senti_seq_lstm.eval("abc")
  //println("verified that in this small example the values of gradients are about right (up to precision)")


  val sentimental_rnn = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

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

       val outBuffer = LOOPTM(inBuffer)(lchs, rchs) { (l: ArrayBuffer[TensorR], r: ArrayBuffer[TensorR], i: Rep[Int]) =>

         val targ = Tensor.zeros(output_size); targ.data(scores(i)) = 1; val targ1 = TensorR(targ)
         val lossl = l(0); val hiddenl = l(1)
         val lossr = r(0); val hiddenr = r(1)

         val hidden = IF (hidden_size) (lchs(i) < 0) { // leaf node
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


  //val senti_file = new PrintWriter(new File("senti.cpp"))
  //senti_file.println(sentimental_rnn.code)
  //senti_file.flush()
  //sentimental_rnn.eval("abc")

  val sentimental_lstm = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

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

        val outBuffer = LOOPTM(inBuffer)(lchs, rchs) { (l: ArrayBuffer[TensorR], r: ArrayBuffer[TensorR], i: Rep[Int]) =>

          val lossl = l(0); val hiddenl = l(1); val celll = l(2)
          val lossr = r(0); val hiddenr = r(1); val cellr = r(2)

          val targ = Tensor.zeros(output_size); targ.data(scores(i)) = 1; val targ1 = TensorR(targ)

          val embedding_tensor = IF (word_embedding_size) (lchs(i) < 0) {
            TensorR(Tensor(word_embedding_data(words(i)), word_embedding_size))
          } {dummy_word_embedding}

          val i_gate = IF (hidden_size) (lchs(i) < 0) {
          (tWi.dot(embedding_tensor) + tbi).sigmoid()
          } {
            (tU0i.dot(hiddenl) + tU1i.dot(hiddenr) + tbbi).sigmoid()
          }

          val fl_gate = IF (hidden_size) (lchs(i) < 0) {
            dummy_forget_gate
          } {
            (tU00f.dot(hiddenl) + tU01f.dot(hiddenr) + tbbf).sigmoid()
          }

          val fr_gate = IF (hidden_size) (lchs(i) < 0) {
            dummy_forget_gate
          } {
            (tU10f.dot(hiddenl) + tU11f.dot(hiddenr) + tbbf).sigmoid()
          }

          val o_gate = IF (hidden_size) (lchs(i) < 0) {
            (tWo.dot(embedding_tensor) + tbo).sigmoid()
          } {
            (tU0o.dot(hiddenl) + tU1o.dot(hiddenr) + tbbo).sigmoid()
          }

          val u_value = IF (hidden_size) (lchs(i) < 0) {
            (tWu.dot(embedding_tensor) + tbu).tanh()
          } {
            (tU0u.dot(hiddenl) + tU1u.dot(hiddenr) + tbbu).tanh()
          }

          val cell = IF (hidden_size) (lchs(i) < 0) {
            i_gate * u_value
          } {
            i_gate * u_value + fl_gate * celll + fr_gate * cellr
          }

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
  
  //println("run sentiment analysis tree LSTM")
  //val sentit_file = new PrintWriter(new File(root_dir + "evaluationTreeLSTM/Lantern/Lantern.cpp"))
  //sentit_file.println(sentimental_lstm.code)
  //sentit_file.flush()
  
  if (false) {
    val cnn_test1 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        val iPane = 1
        val iRow = 16
        val iCol = 20
        val input = Tensor.ones(iPane, iRow, iCol)
        val kOut = 1
        val kIn = iPane
        val kRow = 3
        val kCol = 3
        val kernel = Tensor.ones(kOut, kIn, kRow, kCol)

        val res = input.conv2D(kernel, 1, 1)
        Tensor.assertEqual(res, Tensor.fill((kRow * kCol * kIn) * 1.0f, kOut, iRow - kRow + 1, iCol - kCol + 1), "CNN 1")

        // printf("Result:\\n")
        // res.print3D
      }
    }
    println("start cnn_test1")
    cnn_test1.eval("abc")

    val cnn_test2 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        val iPane = 1
        val iRow = 16
        val iCol = 20
        val input = Tensor.ones(iPane, iRow, iCol)
        val kOut = 1
        val kIn = iPane
        val kRow = 3
        val kCol = 3
        val kernel = Tensor.fill((i: NSeq[Rep[Int]]) => if (i(2) == kRow/2 && i(3) == kCol/2) 1.0f else 0.0f, kOut, kIn, kRow, kCol)

        // printf("Kernel\\n")
        // kernel.print4D

        val res = input.conv2D(kernel, 1, 1)
        Tensor.assertEqual(res, Tensor.fill(1.0f, kOut, iRow - kRow + 1, iCol - kCol + 1), "CNN 2")
        // printf("Result:\\n")
        // res.print3D
      }
    }


    println("start cnn_test2")
    cnn_test2.eval("abc")

    val cnn_test3 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        val iPane = 1
        val iRow = 16
        val iCol = 20
        val input = Tensor.ones(iPane, iRow, iCol)
        val kOut = 1
        val kIn = iPane
        val kRow = 3
        val kCol = 3
        val kernel = Tensor.fill((i: NSeq[Rep[Int]]) => if (i(2) == kRow/2 && i(3) == kCol/2) 1.0f else 0.0f ,kOut, kIn, kRow, kCol)

        val res = input.conv2D(kernel, 2, 2)
        Tensor.assertEqual(res, Tensor.fill(1.0f, kOut, (iRow - kRow)/2 + 1, (iCol - kCol)/2 + 1), "CNN 3")
        // printf("Result:\\n")
        // res.print3D
      }
    }

    println("start cnn_test3")
    cnn_test3.eval("abc")

    val cnn_back_test1 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        val iPane = 1
        val iRow = 16
        val iCol = 20
        val input = Tensor.ones(iPane, iRow, iCol)
        val kOut = 1
        val kIn = iPane
        val kRow = 3
        val kCol = 3
        val kernel = Tensor.ones(kOut, kIn, kRow, kCol)

        val varInput = TensorR(input)
        val varKernel = TensorR(kernel)

        val rS = 1
        val cS = 1

        val tot = NewArray[Long](2)
        def lossFun = { (dummy: TensorR) =>
          val res = varInput.conv(varKernel, rS, cS, tot)
          res.sum()
        }

        val loss = gradR_loss(lossFun)(Tensor.zeros(1))

        val resR = (iRow - kRow)/rS + 1
        val resC = (iCol - kCol)/cS + 1

        Tensor.assertEqual(loss, Tensor.scalar(resR * resC * 9.0f), "BACK - LOSS")
        // printf("Loss:\\n")
        // loss.printRaw()

        // FIXME complete correct result
        // Tensor.assertEqual(varInput.d, Tensor.fill(
        //   (p: Rep[Int], x: Rep[Int], y: Rep[Int]) =>
        //     if (x >= 2 && x < iRow - 2 && 2 <= y && y < iCol - 2)
        //       9.0f
        //     else if (x > ), iPane, iRow, iCol), "BACK 1 - INPUT D")
        // printf("Input gradient:\\n")
        // varInput.d.print3D

        Tensor.assertEqual(varKernel.d, Tensor.fill(resR * resC * 1.0f, kIn, kOut, kRow, kCol), "BACK 1 - KERNEL D")
        // printf("Kernel gradient:\\n")
        // varKernel.d.print4D
      }
    }

    println("start cnn_back_test1")
    cnn_back_test1.eval("abc")

    val cnn_back_test2 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        val iPane = 1
        val iRow = 16
        val iCol = 20
        val input = Tensor.ones(iPane, iRow, iCol)
        val kOut = 1
        val kIn = iPane
        val kRow = 3
        val kCol = 3
        val kernel = Tensor.fill((i: NSeq[Rep[Int]]) => if (i(2) == kRow/2 && i(3) == kCol/2) 1.0f else 0.0f ,kOut, kIn, kRow, kCol)

        val varInput = TensorR(input)
        val varKernel = TensorR(kernel)

        val rS = 1
        val cS = 1


        val tot = NewArray[Long](2)
        def lossFun = { (dummy: TensorR) =>
          val res = varInput.conv(varKernel, rS, cS, tot)
          res.sum()
        }

        val loss = gradR_loss(lossFun)(Tensor.zeros(1))
        // printf("Loss:\\n")
        // loss.printRaw()

        val resR = (iRow - kRow)/rS + 1
        val resC = (iCol - kCol)/cS + 1
        Tensor.assertEqual(loss, Tensor.scalar(resR * resC * 1.0f), "BACK 2 - LOSS")

        // FIXME complete correct result
        // Tensor.assertEqual(varInput.d, Tensor.fill(
        //   (p: Rep[Int], x: Rep[Int], y: Rep[Int]) =>
        //     if (x >= 2 && x < iRow - 2 && 2 <= y && y < iCol - 2)
        //       9.0f
        //     else if (x > ), iPane, iRow, iCol), "BACK 1 - INPUT D")
        // varInput.d.print3D
        // printf("Input gradient:\\n")
        // varInput.d.print3D

        Tensor.assertEqual(varKernel.d, Tensor.fill(resR * resC * 1.0f, kIn, kOut, kRow, kCol), "BACK 2 - KERNEL D")
        // printf("Kernel gradient:\\n")
        // varKernel.d.print4D
      }
    }

    println("start cnn_back_test2")
    cnn_back_test2.eval("abc")

    val cnn_back_test3 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        val iPane = 1
        val iRow = 16
        val iCol = 20
        val input = Tensor.ones(iPane, iRow, iCol)
        val kOut = 1
        val kIn = iPane
        val kRow = 3
        val kCol = 3
        val kernel = Tensor.fill((i: NSeq[Rep[Int]]) => if (i(2) == kRow/2 && i(3) == kCol/2) 1.0f else 0.0f, kOut, kIn, kRow, kCol)

        val varInput = TensorR(input)
        val varKernel = TensorR(kernel)

        val rS = 2
        val cS = 2

        val tot = NewArray[Long](2)
        def lossFun = { (dummy: TensorR) =>
          val res = varInput.conv(varKernel, rS, cS, tot)
          res.sum()
        }

        val loss = gradR_loss(lossFun)(Tensor.zeros(1))
        // printf("Loss:\\n")
        // loss.printRaw()

        val resR = (iRow - kRow)/rS + 1
        val resC = (iCol - kCol)/cS + 1
        Tensor.assertEqual(loss, Tensor.scalar(resR * resC * 1.0f), "BACK 2 - LOSS")

        // FIXME complete correct result
        // Tensor.assertEqual(varInput.d, Tensor.fill(
        //   (p: Rep[Int], x: Rep[Int], y: Rep[Int]) =>
        //     if (x >= 2 && x < iRow - 2 && 2 <= y && y < iCol - 2)
        //       9.0f
        //     else if (x > ), iPane, iRow, iCol), "BACK 1 - INPUT D")
        // varInput.d.print3D
        // printf("Input gradient:\\n")
        // varInput.d.print3D

        Tensor.assertEqual(varKernel.d, Tensor.fill(resR * resC * 1.0f, kIn, kOut, kRow, kCol), "BACK 2 - KERNEL D")
        // printf("Kernel gradient:\\n")
        // varKernel.d.print4D
      }
    }

    println("start cnn_back_test3")
    cnn_back_test3.eval("abc")

    val cnn_test4 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        val iPane = 3
        val iRow = 16
        val iCol = 20
        val input = Tensor.ones(iPane, iRow, iCol)
        val kOut = 1
        val kIn = iPane
        val kRow = 3
        val kCol = 3
        val kernel = Tensor.ones(kOut, kIn, kRow, kCol)

        val rS = 2
        val cS = 2
        val res = input.conv2D(kernel, rS, cS)
        Tensor.assertEqual(res, Tensor.fill(iPane * kRow * kCol * 1.0f, kOut, (iRow - kRow)/rS + 1, (iCol - kCol)/cS + 1), "CNN 4")
        // printf("Result:\\n")
        // res.print3D
      }
    }

    println("start cnn_test4")
    cnn_test4.eval("abc")

    val cnn_back_test4 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        val iPane = 3
        val iRow = 16
        val iCol = 20
        val input = Tensor.ones(iPane, iRow, iCol)
        val kOut = 1
        val kIn = iPane
        val kRow = 3
        val kCol = 3
        val kernel = Tensor.ones(kOut, kIn, kRow, kCol)

        val varInput = TensorR(input)
        val varKernel = TensorR(kernel)

        val rS = 2
        val cS = 2

        val tot = NewArray[Long](2)
        def lossFun = { (dummy: TensorR) =>
          val res = varInput.conv(varKernel, rS, cS, tot)
          res.sum()
        }

        val loss = gradR_loss(lossFun)(Tensor.zeros(1))
        // printf("Loss:\\n")
        // loss.printRaw()

        val resR = (iRow - kRow)/rS + 1
        val resC = (iCol - kCol)/cS + 1
        Tensor.assertEqual(loss, Tensor.scalar(kOut * resR * resC * 27.0f), "BACK 4 - LOSS")

        // FIXME complete correct result
        // Tensor.assertEqual(varInput.d, Tensor.fill(
        //   (p: Rep[Int], x: Rep[Int], y: Rep[Int]) =>
        //     if (x >= 2 && x < iRow - 2 && 2 <= y && y < iCol - 2)
        //       9.0f
        //     else if (x > ), iPane, iRow, iCol), "BACK 1 - INPUT D")
        // varInput.d.print3D
        // printf("Input gradient:\\n")
        // varInput.d.print3D

        Tensor.assertEqual(varKernel.d, Tensor.fill(resR * resC * 1.0f, kOut, kIn, kRow, kCol), "BACK 4 - KERNEL D")
        // printf("Kernel gradient:\\n")
        // varKernel.d.print4D
      }
    }

    println("start cnn_back_test4")
    cnn_back_test4.eval("abc")


    val cnn_test5 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        val iPane = 3
        val iRow = 16
        val iCol = 20
        val input = Tensor.ones(iPane, iRow, iCol)
        val kOut = 4
        val kIn = iPane
        val kRow = 3
        val kCol = 3
        val kernel = Tensor.ones(kOut, kIn, kRow, kCol)

        val rS = 2
        val cS = 2
        val res = input.conv2D(kernel, rS, cS)
        Tensor.assertEqual(res, Tensor.fill(iPane * kRow * kCol * 1.0f, kOut, (iRow - kRow)/rS + 1, (iCol - kCol)/cS + 1), "CNN 4")
        // printf("Result:\\n")
        // res.print3D
      }
    }

    println("start cnn_test5")
    cnn_test5.eval("abc")

    val cnn_back_test5 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        val iPane = 3
        val iRow = 16
        val iCol = 20
        val input = Tensor.ones(iPane, iRow, iCol)
        val kOut = 4
        val kIn = iPane
        val kRow = 3
        val kCol = 3
        val kernel = Tensor.fill((i: NSeq[Rep[Int]]) => if (i(2) == kRow/2 && i(3) == kCol/2) 1.0f else 0.0f ,kOut, kIn, kRow, kCol)

        val varInput = TensorR(input)
        val varKernel = TensorR(kernel)

        val rS = 2
        val cS = 2

        val tot = NewArray[Long](2)
        def lossFun = { (dummy: TensorR) =>
          val res = varInput.conv(varKernel, rS, cS, tot)
          res.sum()
        }

        val loss = gradR_loss(lossFun)(Tensor.zeros(1))
        // printf("Loss:\\n")
        // loss.printRaw()

        val resR = (iRow - kRow)/rS + 1
        val resC = (iCol - kCol)/cS + 1
        Tensor.assertEqual(loss, Tensor.scalar(kOut * resR * resC * kIn * 1.0f), "BACK 5 - LOSS")

        // FIXME complete correct result
        // Tensor.assertEqual(varInput.d, Tensor.fill(
        //   (p: Rep[Int], x: Rep[Int], y: Rep[Int]) =>
        //     if (x >= 2 && x < iRow - 2 && 2 <= y && y < iCol - 2)
        //       9.0f
        //     else if (x > ), iPane, iRow, iCol), "BACK 1 - INPUT D")
        // varInput.d.print3D
        // printf("Input gradient:\\n")
        // varInput.d.print3D

        Tensor.assertEqual(varKernel.d, Tensor.fill(resR * resC * 1.0f, kOut, kIn, kRow, kCol), "BACK 5 - KERNEL D")
        // printf("Kernel gradient:\\n")
        // varKernel.d.print4D
      }
    }

    println("start cnn_back_test5")
    cnn_back_test5.eval("abc")

    val maxpool_test1 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val iPane = 2
        val iRow = 8
        val iCol = 10
        val input = Tensor.ones(iPane, iRow, iCol)

        val sR = 2
        val sC = 2
        val (res, idx) = input.maxPool(sR, sC)

        Tensor.assertEqual(res, Tensor.ones(iPane, iRow/sR, iCol/sC), "MAXPOOL 1")
        for (i <- 0 until res.nbElem: Rep[Range]) {
          // assertC(idx(i) ==  (i / res.strides(2)) * sR * input.strides(2) + sC * (i % res.strides(2)), s"Maxpool index invalid %d != %d (%d - %d)\\n", idx(i), (i / res.strides(2)) * sR * input.strides(2) + sC * (i % res.strides(2)), i / res.strides(2), i % res.strides(2))
        }

      }

    }

    println("start maxpool test1")
    maxpool_test1.eval("abc")

    val maxpool_back_test1 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val iPane = 2
        val iRow = 8
        val iCol = 10
        val input = Tensor.ones(iPane, iRow, iCol)

        val sR = 2
        val sC = 2

        val varInput = TensorR(input)

        def lossFun = { (dummy: TensorR) =>
          val res = varInput.maxPool(sR, sC)
          res.sum()
        }

        val loss = gradR_loss(lossFun)(Tensor.zeros(1))

        Tensor.assertEqual(loss, Tensor.scalar(iPane * (iRow/sR) * (iCol/sC) * 1.0f), "MAXPOOL BACK 1 - LOSS")
        Tensor.assertEqual(varInput.d, Tensor.fill((i: NSeq[Rep[Int]]) => if (i(1) % sR == 0 && i(2) % sC == 0) 1.0f else 0.0f, iPane, iRow, iCol), "MAXPOOL BACK 1 - D")
        // printf("Input derivative:\\n")
        // varInput.d.print3D
      }

    }

    println("start maxpool back test1")
    maxpool_back_test1.eval("abc")

    val dropout_test1 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val iPane = 2
        val iRow = 16
        val iCol = 20
        val input = Tensor.rand(iPane, iRow, iCol)

        val (resAll, idxAll) = input.dropout(0.0f)
        val (resNone, idxNone) = input.dropout(1.0f)

        Tensor.assertEqual(resAll, input, "DROPOUT 1")
        Tensor.assertEqual(resNone, Tensor.zeros(input), "DROPOUT 2")

        for (i <- 0 until input.nbElem: Rep[Range]) {
          assertC(idxAll(i) == 1.0f, "idxAll incorrect %.3f != 1\\n", idxAll(i))
          assertC(idxNone(i) == 0.0f, "idxNone incorrect %.3f != 0\\n", idxNone(i))
        }
      }
    }

    println("start dropout test1")
    dropout_test1.eval("abc")

    val dropout_back_test1 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val iPane = 2
        val iRow = 16
        val iCol = 20
        val input = Tensor.rand(iPane, iRow, iCol)

        val varInput = TensorR(input)

        def lossFun = { (dummy: TensorR) =>
          val res = varInput.dropout(0.0f)
          res.sum()
        }

        val loss = gradR_loss(lossFun)(Tensor.zeros(1))
        Tensor.assertEqual(varInput.d, Tensor.ones(input), "DROPOUT BACK 1 - D")
        // printf("Input derivative:\\n")
        // varInput.d.print3D

      }
    }

    println("start dropout back test1")
    dropout_back_test1.eval("abc")

    val dropout_back_test2 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val iPane = 2
        val iRow = 16
        val iCol = 20
        val input = Tensor.rand(iPane, iRow, iCol)

        val varInput = TensorR(input)

        def lossFun = { (dummy: TensorR) =>
          val res = varInput.dropout(1.0f)
          res.sum()
        }

        val loss = gradR_loss(lossFun)(Tensor.zeros(1))
        Tensor.assertEqual(varInput.d, Tensor.zeros(input), "DROPOUT BACK 1 - D")
        // printf("Input derivative:\\n")
        // varInput.d.print3D

      }
    }


    println("start dropout back test2")
    dropout_back_test2.eval("abc")

    val test_cnn_full1 = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

      // FIXME: add proper check for result. see adworkplace/pytorch/cnn_test.py
      def snippet(a: Rep[String]): Rep[Unit] = {

        Random.srand(Some(1000))

        val iChan1 = 2
        val iRow1 = 10
        val iCol1 = 10

        val input = Tensor.rand(1.0f, iChan1, iRow1, iCol1)

        // Layer 1
        val inChan1 = iChan1
        val outChan1 = 2
        val kRow1 = 3
        val kCol1 = 3

        // stride conv
        val sRow1 = 1
        val sCol1 = 1

        // stride maxpool
        val smRow1 = 2
        val smCol1 = 2

        val conv1 = Tensor.rand(1.0f, outChan1, inChan1, kRow1, kCol1)
        val oRow1 = convSize(iRow1, kRow1, sRow1)/smRow1
        val oCol1 = convSize(iCol1, kCol1, sCol1)/smCol1

        val inChan2 = outChan1
        val outChan2 = 3

        val conv2 = Tensor.rand(1.0f, outChan2, inChan2, kRow1, kCol1)

        val oRow2 = convSize(oRow1, kRow1, sRow1)
        val oCol2 = convSize(oCol1, kCol1, sCol1)
        val out3 = 5
        val in3 = outChan2 * oRow2 * oCol2

        val a1 = Tensor.rand(1.0f, out3, in3)
        val b1 = Tensor.rand(1.0f, out3)


        val varInput = TensorR(input)
        val varConv1 = TensorR(conv1)
        val varConv2 = TensorR(conv2)
        val varA1 = TensorR(a1)
        val varB1 = TensorR(b1)

        val tot = NewArray[Long](2)
        def lossFun = { (dummy: TensorR) =>
          varInput.print("Input")
          val resConv = varInput.conv(varConv1, sRow1, sCol1, tot)
          resConv.print("First conv")
          val resMax = resConv.maxPool(smRow1, smCol1)
          resMax.print("MaxPool")
          val resRL = resMax.relu()
          resRL.print("ReLu 2")
          val resConv2 = resRL.conv(varConv2, sRow1, sCol1, tot)
          resConv2.print("Second conv")
          val resRL2 = resConv2.relu()
          resRL2.print("ReLu 2")
          val resMMul = varA1 dot resRL2.resize(in3)
          resMMul.print("Matrix Multiplication")
          val resVAdd = resMMul + varB1
          resVAdd.print("Vector Addition")
          val resLSM = resVAdd.logSoftmax()
          resLSM.print("LogSoftMax")
          resLSM.nllLoss(2)
        }

        for (x <- 0 until 1000: Rep[Range]) {
          val loss = gradR_loss(lossFun)(Tensor.scalar(0.0f))
          loss.print("Loss")

          // Update weight
          for ((weight, idx) <- NSeq(varConv1, varConv2, varA1, varB1).zipWithIndex) {
            weight.print(s"Before ${idx + 1}", derivative = true)
            weight.x.addMul(-0.5f, weight.d)
            weight.print(s"After ${idx + 1}")
            weight.clear_grad()
            printf("\\n")
          }
        }
      }
    }
    println("start full CNN test")
    test_cnn_full1.eval("abc")
  
  } // if false 2 closing

  val mnist  = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

    // From the MNIST pytorch example
    val mean = 0.1307f
    val std = 0.3081f


    class DataLoader(name: String, train: Boolean, dims: Int*) {

      def open(path: Rep[String]) = uncheckedPure[Int]("open(",path,",0)")
      def filelen(fd: Rep[Int]) = uncheckedPure[Long]("fsize(",fd,")") // FIXME: fresh name
      def mmap[T:Typ](fd: Rep[Int], len: Rep[Long]) = uncheckedPure[Array[T]]("(",codegen.remap(typ[T]),"*)mmap(0, ",len,", PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, ",fd,", 0)")

      val fd = open(s"../data/bin/${name}_${if (train) "train" else "test"}.bin")
      val len = filelen(fd)
      val data = mmap[Float](fd, len)
      val dLength = (len/4L).toInt

      val tfd = open(s"../data/bin/${name}_${if (train) "train" else "test"}_target.bin")
      val tlen = filelen(tfd)
      val target = mmap[Int](tfd, tlen)
      val length = (tlen/4L).toInt

      @virtualize
      def normalize() = {
        this.foreach { (t, d) =>
          t.normalize(mean, std, inPlace = true)
        }
      }


      @virtualize
      def foreach(f: (Tensor, Rep[Int]) => Unit) = {
        var off = var_new(0)
        for (img <- 0 until length: Rep[Range]) {
          val dataPtr = slice(data, off)
          val t = Tensor(dataPtr, dims : _*)
          f(t, target(img))
          off += t.nbElem
        }
        assertC(off == dLength, "Data length doesn't match\\n")
      }
    }

    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {
      printf("Here we go!! Go MNIST!!!!\\n")
      Random.srand(Some(42))

      //move timer here to track all of the prepare time
      val dataTimer = Timer2()
      dataTimer.startTimer
      
      val variables = ArrayBuffer[TensorR]()

      // input size
      val iChan1 = 1
      val iRow1 = 28
      val iCol1 = 28

      System.out.println(s"Input size: $iChan1 x $iRow1 x $iCol1")

      // TODO create modules
      // Layer 1
      val inChan1 = iChan1
      val outChan1 = 10
      val kRow1 = 5
      val kCol1 = 5

      // stride conv
      val sRow1 = 1
      val sCol1 = 1

      // stride maxpool
      val smRow1 = 2
      val smCol1 = 2

      // FIXME scale based on PyTorch
      val conv1 = Tensor.rand(1.0f / sqrt(inChan1 * kRow1 * kCol1).toFloat, outChan1, inChan1, kRow1, kCol1)
      val varConv1 = TensorR(conv1)
      variables += varConv1

      // input size
      val iChan2 = outChan1
      val iRow2 = convSize(iRow1, kRow1, sRow1)/smRow1
      val iCol2 = convSize(iCol1, kCol1, sCol1)/smCol1

      System.out.println(s"Layer 1 output size: $iChan2 x $iRow2 x $iCol2")

      // Layer 2
      val inChan2 = outChan1
      val outChan2 = 20
      val kRow2 = 5
      val kCol2 = 5

      // stride conv
      val sRow2 = 1
      val sCol2 = 1

      // stride maxpool
      val smRow2 = 2
      val smCol2 = 2

      val conv2 = Tensor.rand(1.0f / sqrt(inChan2 * kRow2 * kCol2).toFloat, outChan2, inChan2, kRow2, kCol2)
      val varConv2 = TensorR(conv2)
      variables += varConv2

      // Layer 3
      val oRow2 = convSize(iRow2, kRow2, sRow2)/smRow2
      val oCol2 = convSize(iCol2, kCol2, sCol2)/smCol2
      val in3 = 320
      val out3 = 50

      System.out.println(s"Layer 2 output size: $outChan2 x $oRow2 x $oCol2")

      assert(in3 == outChan2 * oRow2 * oCol2, s"The input of the first Linear layer should be $in3, got ${outChan2 * oRow2 * oCol2}")

      val a1 = Tensor.rand(1.0f / sqrt(in3).toFloat, out3, in3)
      val b1 = Tensor.rand(1.0f / sqrt(in3).toFloat, out3)
      val varA1 = TensorR(a1)
      val varB1 = TensorR(b1)
      variables += varA1
      variables += varB1

      // Layer 4
      val in4 = out3
      val out4 = 10

      val a2 = Tensor.rand(1.0f / sqrt(in4).toFloat, out4, in4)
      val b2 = Tensor.rand(1.0f / sqrt(in4).toFloat, out4)
      val varA2 = TensorR(a2)
      val varB2 = TensorR(b2)
      variables += varA2
      variables += varB2

      // Training
      val nbEpoch = 10
      val lr = 0.0005f
      val mom = 0.0f

      val momentum = if (mom > 0.0f) variables map(tR => Tensor.zeros(tR.d)) else ArrayBuffer[Tensor]()

      val tot1 = NewArray[Long](2)
      val tot2 = NewArray[Long](2)

      //val dataTimer = Timer2()
      //dataTimer.startTimer
      
      val train = new DataLoader("mnist", true, iChan1, iRow1, iCol1)
      printf("Start normalize\\n")
      train.normalize()

      def trainFun(input: TensorR, target: Rep[Int]) = { (dummy: TensorR) =>
        val resL1 = input.conv(varConv1, sRow1, sCol1, tot1).maxPool(smRow1, smCol1).relu()
        val resL2 = resL1.conv(varConv2, sRow2, sCol2, tot2).maxPool(smRow2, smCol2).relu()
        val resL3 = ((varA1 dot resL2.resize(in3)) + varB1).relu().dropout(0.5f)
        val resL4 = (varA2 dot resL3) + varB2
        val res = resL4.logSoftmax()
        res.nllLoss(target)
      }
      
      // we skip tests for the experiments
      //val test = new DataLoader("mnist", false, iChan1, iRow1, iCol1)
      //test.normalize()
      
      val prepareTime = dataTimer.getElapsedTime / 1e6f
      printf("Data normalized (all prepare time) in %lf sec\\n", prepareTime)

      val loss_save = NewArray[Double](nbEpoch)

      val addr = getMallocAddr() // remember current allocation pointer here
      for (epoch <- 0 until nbEpoch: Rep[Range]) {
    
        val trainTimer = Timer2()
        var imgIdx = var_new(0)
        var trainLoss = var_new(0.0f)
        printf("Start training epoch %d\\n", epoch + 1)
        trainTimer.startTimer
        train foreach { (input: Tensor, target: Rep[Int]) =>
          imgIdx += 1
          // assertC(0 <= target && target <= 9, "Target should be a number between 0 and 9, got %d\\n", target)

          val inputR = TensorR(input , isInput=true)
          val loss = gradR_loss(trainFun(inputR, target))(Tensor.scalar(0.0f))
          trainLoss += loss.data(0)

          // for ((weight, idx) <- variables.zipWithIndex) {
          //   weight.print(s"Variable ${idx + 1}", derivative = true)
          // }

          // Update weights
          for ((weight, idx) <- variables.zipWithIndex) {
            val d = if (mom > 0.0f) {
              printf("TBT\\n")
              exit()
              val sMom = momentum(idx)
              sMom.cmulAdd(mom, weight.d)
            } else {
              weight.d
            }

            // printf("Weight before %.10f -", weight.x.data(0))
            weight.x.addMul(-lr, d)
            // if (weight.x.check(5.0f)) {
            //   printf("Iteration %d\\n", imgIdx)
            //   weight.print(s"Weight of variable ${idx + 1} diverged!!!", derivative = true)
            //   exit()
            // }
            // printf("%.10f weigth after (%.10f - %.5f)\\n", weight.x.data(0), weight.d.data(0), lr)
            weight.clear_grad()
          }

          // for ((weight, idx) <- variables.zipWithIndex) {
          //   weight.print(s"Variable ${idx + 1}")
          // }

          if (imgIdx %  (train.length / 10) == 0) {
            printf(s"Train epoch %d: [%d/%d (%.0f%%)]\\tAverage Loss: %.6f\\n", epoch, imgIdx, train.length, 100.0 * imgIdx /train.length, trainLoss/imgIdx)
            // printf("Conv1 fwd %ld us/image - bwd %ld us/image\\n", tot1(0)/imgIdx, tot1(1)/imgIdx)
            // printf("Conv2 fwd %ld us/image - bwd %ld us/image\\n", tot2(0)/imgIdx, tot2(1)/imgIdx)
            unchecked[Unit]("fflush(stdout)")
          }
          resetMallocAddr(addr)
        }
        val delta = trainTimer.getElapsedTime
        printf("Training completed in %ldms (%ld us/images)\\n", delta/1000L, delta/train.length)
        
        // save trainLoss / train.length to loss_save
        loss_save(epoch) = trainLoss / train.length

        /* skip tests
        def testFun(input: Tensor) = {
          val (resL1, _) = input.conv2D(conv1, sRow1, sCol1).maxPool(smRow1, smCol1)
          val (resL2, _) = resL1.relu().conv2D(conv2, sRow2, sCol2).maxPool(smRow2, smCol2)
          val resL3 = ((a1 dot resL2.relu().resize(in3)) + b1).relu()
          val resL4 = (a2 dot resL3) + b2
          resL4.logSoftmax()
        }

        printf("\\nStart testing:\\n")
        val testTimer = Timer2()
        testTimer.startTimer
        imgIdx = var_new(0)
        var testLoss = var_new(0.0)
        val correct = var_new(0)
        test foreach { (input: Tensor, target: Rep[Int]) =>
          imgIdx += 1
          val res = testFun(input)

          testLoss += res.nllLoss(target).data(0)
          if (res.maxIndex() == target)
            correct += 1
        }
        printf("Test set: Average loss: %.4f, Acurracy: %d/%d (%.0f) in %ldms\\n", testLoss / test.length, correct, test.length, 100.0 * correct / test.length, testTimer.getElapsedTime/1000L)
        printf("\\n\\n")
        */

      }

      val totalTime = dataTimer.getElapsedTime / 1e6f 
      val loopTime = totalTime - prepareTime
      val timePerEpoc = loopTime / nbEpoch

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
  
  /*
  println("run simple CNN test case")
  val cnn_file = new PrintWriter(new File(root_dir + "evaluationCNN/Lantern/Lantern.cpp"))
  cnn_file.println(mnist.code)
  cnn_file.flush()
  */

  val becky = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        for (i <- 0 until 10: Range) {
          val j: Rep[Int] = i
          if (j % 3 == 0) printf(s"Found number %d\\n", j)
        }

        printf("Done\\n")

      }
  }

  //becky.eval("111")
}
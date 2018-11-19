package lantern

import scala.util.continuations._
import scala.util.continuations

import scala.virtualization.lms._
import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.collection.mutable.ArrayBuffer
import scala.math._

import org.scalatest.FunSuite

import java.io.PrintWriter
import java.io.File

class AdLMSVectorTest extends LanternFunSuite {

  test("IF") {
    val IF_Test = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.zeros(3, 4)
        val grad = gradR(x => IF(1 < 0){
          generateRawComment("true branch")
          x + x
        }{
          generateRawComment("false branch")
          x * x
        })(input)
        generateRawComment("show gradient")
      }
    }
    // System.out.println(IF_Test.code)
    IF_Test.eval("abc")
  }

  test("array0") {
    val array0 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val addr = getMallocAddr()
        resetMallocAddr(addr)
        val mem = Tensor.zeros(100)
        val addr1 = getMallocAddr()
        resetMallocAddr(addr)
        val addr2 = getMallocAddr()

        //assertions
        if (addr + 400 != addr1) printf("ERROR: addr did not increase by 800")
        if (addr != addr2) printf("ERROR: addr did not reset to the give value")
      }
    }
    array0.eval("abc")
  }

  test("vector-vector-dot") {
    val vvdot = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val length = 2
        val v1 = Tensor.fromData(Seq(4), 1, 2, 3, 4)
        val v2 = Tensor.fromData(Seq(4), -1, -2, -3, -4)
        val expected = Tensor.scalar(-30)
        Tensor.assertEqual(v1.dot(v2), expected)
      }
    }
    runTest(vvdot)
  }

  test("matrix-vector-dot") {
    val mvdot = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val m = Tensor.fromData(Seq(2, 4), 1, 2, 3, 4, 5, 6, 7, 8)
        val v = Tensor.fromData(Seq(4), -1, -2, -3, -4)
        val expected = Tensor.fromData(Seq(2), -30, -70)
        Tensor.assertEqual(m.dot(v), expected)

        val mm = TensorR(m)
        val vv = TensorR(v)
        gradR_loss(dummy => (mm dot vv).sum())(Tensor.zeros(1))
        // mm.d.print("mm grad")
        // vv.d.print("vv grad")
        val expected1 = Tensor.fromData(Seq(2, 4), -1,-2,-3,-4,-1,-2,-3,-4)
        val expected2 = Tensor.fromData(Seq(4), 6, 8, 10, 12)
        Tensor.assertEqual(mm.d, expected1)
        Tensor.assertEqual(vv.d, expected2)
      }
    }
    runTest(mvdot)
  }

  test("matrix-matrix-dot") {
    val mmdot = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        // Note: it's better to test with matrices [M1 x M2] and [M2 x M3] where M1 != M3.
        val m1 = Tensor.fromData(Seq(2, 3), 1, 2, 3, 4, 5, 6)
        val m2 = Tensor.fromData(Seq(3, 2), 2, 3, 4, 2, 3, 4)
        val expected = Tensor.fromData(Seq(2, 2), 19, 19, 46, 46)
        Tensor.assertEqual(m1.dot(m2), expected)

        val mm1 = TensorR(m1)
        val mm2 = TensorR(m2)
        gradR_loss(dummy => (mm1 dot mm2).sum())(Tensor.zeros(1))
        val expected1 = Tensor.fromData(Seq(2, 3), 5, 6, 7, 5, 6, 7)
        val expected2 = Tensor.fromData(Seq(3, 2), 5, 5, 7, 7, 9, 9)
        Tensor.assertEqual(mm1.d, expected1)
        Tensor.assertEqual(mm2.d, expected2)
      }
    }
    runTest(mmdot)
  }

  test("gemm") {
    val gemm = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val m1 = Tensor.rand(2,3)
        val m2 = Tensor.rand(3,4)
        val viaDot = (m1 dot m2) * 0.5f
        val viaGemm = m1.gemm(m2, false, false, 0.5f)
        Tensor.assertEqual(viaDot, viaGemm)

        val m3 = Tensor.rand(4,3)
        val viaDot01 = (m1 dot m3.trans()) * 0.5f
        val viaGemm01 = m1.gemm(m3, false, true, 0.5f)
        Tensor.assertEqual(viaDot01, viaGemm01)

        val m4 = Tensor.rand(3,2)
        val viaDot10 = (m4.trans() dot m2) * 0.5f
        val viaGemm10 = m4.gemm(m2, true, false, 0.5f)
        Tensor.assertEqual(viaDot10, viaGemm10)

        val viaDot11 = (m4.trans() dot m3.trans()) * 0.5f
        val viaGemm11 = m4.gemm(m3, true, true, 0.5f)
        Tensor.assertEqual(viaDot11, viaGemm11)
      }
    }
    runTest(gemm)
  }

  test("gemm_grad") {
    val gemm = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val m1 = Tensor.rand(2,3)
        val m2 = Tensor.rand(3,4)
        val tr1 = TensorR(m1); val tr2 = TensorR(m2)
        val tr3 = TensorR(m1); val tr4 = TensorR(m2)
        gradR(x => (tr1 dot tr2) * 0.5f)(Tensor.zeros(1))
        gradR(x => tr3.gemm(tr4, false, false, 0.5f))(Tensor.zeros(1))
        Tensor.assertEqual(tr1.d, tr3.d)
        Tensor.assertEqual(tr2.d, tr4.d)

        val m3 = Tensor.rand(4,3)
        val tr5 = TensorR(m1); val tr6 = TensorR(m3)
        val tr7 = TensorR(m1); val tr8 = TensorR(m3)
        gradR(x => (tr5 dot tr6.trans()) * 0.5f)(Tensor.zeros(1))
        gradR(x => tr7.gemm(tr8, false, true, 0.5f))(Tensor.zeros(1))
        Tensor.assertEqual(tr5.d, tr7.d)
        Tensor.assertEqual(tr6.d, tr8.d)

        val m4 = Tensor.rand(3,2)
        val tr9 = TensorR(m4); val tr10 = TensorR(m2)
        val tr11 = TensorR(m4); val tr12 = TensorR(m2)
        gradR(x => (tr9.trans() dot tr10) * 0.5f)(Tensor.zeros(1))
        gradR(x => tr11.gemm(tr12, true, false, 0.5f))(Tensor.zeros(1))
        Tensor.assertEqual(tr9.d, tr11.d)
        Tensor.assertEqual(tr10.d, tr12.d)

        val tr13 = TensorR(m4); val tr14 = TensorR(m3)
        val tr15 = TensorR(m4); val tr16 = TensorR(m3)
        gradR(x => (tr13.trans() dot tr14.trans()) * 0.5f)(Tensor.zeros(1))
        gradR(x => tr15.gemm(tr16, true, true, 0.5f))(Tensor.zeros(1))
        Tensor.assertEqual(tr13.d, tr15.d)
        Tensor.assertEqual(tr14.d, tr16.d)
      }
    }
    runTest(gemm)
  }

  test("softmax") {
    val softmax = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(2, 3), 1, 2, 3, 4, 5, 6)
        val result = input.softmax_batch()
        val expectedResult = Tensor.fromData(Seq(2, 3),
          0.0900305733f, 0.2447284758f, 0.6652409434f,
          0.0900305733f, 0.2447284758f, 0.6652409434f)
        Tensor.assertEqual(expectedResult, result)

        // TODO: Implement CPU `softmax_grad`.
        /*
        val grad = gradR(x => x.softmax_batch())(input)
        val expectedGrad = Tensor.fromData(Seq(2, 3),
          0.0000000107f, 0.0000000292f, 0.0000000793f,
          0.0000000107f, 0.0000000292f, 0.0000000793f)
        Tensor.assertEqual(expectedGrad, grad)
        */
      }
    }
    runTest(softmax)
  }

  test("log-softmax") {
    val logSoftmax = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(2, 3), 1, 2, 3, 4, 5, 6)
        val result = input.logSoftmaxB()
        val grad = gradR(x => x.logSoftmaxB())(input)
        val expectedResult = Tensor.fromData(Seq(2, 3),
          -2.4076058865f, -1.4076058865f, -0.4076058865f,
          -2.4076061249f, -1.4076061249f, -0.4076061249f)
        val expectedGrad = Tensor.fromData(Seq(2, 3),
          0.7299082279f, 0.2658145428f, -0.9957230091f,
          0.7299083471f, 0.2658147216f, -0.9957225323f)
        Tensor.assertEqual(expectedResult, result)
        Tensor.assertEqual(expectedGrad, grad)
      }
    }
    runTest(logSoftmax)
  }

  test("nll-loss") {
    val nllLoss = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(2, 3), 1, 2, 3, 4, 5, 6)
        val target: Rep[Array[Int]] = Array(1, 0)
        val result = input.logSoftmaxB().nllLossB(target)
        val grad = gradR(x => x.logSoftmaxB().nllLossB(target))(input)

        result.print()
        grad.print()
        val expectedResult = Tensor.fromData(Seq(2), 1.4076058865f, 2.4076061249f)
        val expectedGrad = Tensor.fromData(Seq(2, 3),
          0.0900305808f, -0.7552714944f, 0.6652410030f,
          -0.9099694490f, 0.2447284311f, 0.6652408242f)
        Tensor.assertEqual(expectedResult, result)
        Tensor.assertEqual(expectedGrad, grad)
      }
    }
    runTest(nllLoss)
  }

  test("array2") {
    val array2 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        // read training data from file (for now just use random)
        val length = 2
        val v = Tensor.randinit(length)
        // calculate gradient
        val grad = gradR(t => t dot t)(v)
        // assertions
        Tensor.assertEqual(v * 2.0f, grad)

        // construct TensorR for closure
        val tv = TensorR(v)
        val loss = gradR_loss(dummy => tv dot tv)(Tensor.scalar(0))
        Tensor.assertEqual((v dot v), loss)
        Tensor.assertEqual(tv.d, grad)
      }
    }
    array2.eval("2.0f")
  }

  test("array2_1"){
    val array2_1 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

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
        val loss = gradR_loss(model)(Tensor.scalar(0))
        Tensor.assertEqual(loss, ((vector dot vector) * Tensor.halves(dim0)).sum(), "1")
        Tensor.assertEqual(ve.d, vector * 2.0f ,"2")
        Tensor.assertEqual(half.d, Tensor.fill(Seq(2), (vector dot vector).data(0)), "3")
        ()
      }
    }
    array2_1.eval("abc")
  }

  test("array2_2") {
    val array2_2 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

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
        val loss = gradR_loss(model)(Tensor.scalar(0))
        Tensor.assertEqual(loss, (matrix dot vector).sum(), "11")
        Tensor.assertEqual(ma.d, Tensor.expand(vector, dim0), "12")
        val sol = matrix.sum(dim = 0)
        Tensor.assertEqual(ve.d, sol, "13")
        ()
      }
    }
    array2_2.eval("abc")
  }

  test("testTrans") {
    val testTrans = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val idx = var_new(0)
        val t = Tensor.fill(Seq(2, 3), (seq => { idx += 1; idx }))

        Tensor.assertEqual(t.trans(), Tensor.fromData(1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f).resize(3, 2), "Transpose invalid")
      }
    }
    testTrans.eval("abs")
  }

  test("array2_3") {
    val array2_3 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

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
           val temp1 = e1.sum()
           val p1 = e1 / temp1                            // use unnormalized prob to compute normalize prob
           generateRawComment("Compute new loss")
           val newloss = t(0) - (p1 dot y1).log()            // loss is updated by original loss t(0) and additional loss
           generateRawComment("Done computing loss")
           val out = ArrayBuffer[TensorR]()

           out.append(newloss)
           out.append(h1)
           out
         }
         hprev_next.copy_data(outputs(1).x)  // update the hidden state with the result from LOOP
         outputs(0)                          // return the final loss
       }
       val loss1 = gradR_loss(lossFun)(Tensor.scalar(0))

       generateRawComment("Compute real value")


       // correct method of loss and gradient calculation, adapting from Numpy
       // preset space for gradients
       val dWxh = Tensor.zeros_like(Wxh)
       val dWhh = Tensor.zeros_like(Whh)
       val dWhy = Tensor.zeros_like(Why)
       val dbh  = Tensor.zeros_like(bh)
       val dby  = Tensor.zeros_like(by)
       val dhnext = Tensor.zeros_like(hprev)
       val sum_loss = Tensor.scalar(0)
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
    array2_3.eval("abc")
  }

  test("array2_4"){
    val array2_4 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

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
        val dummy = gradR(lossFun)(Tensor.scalar(0))
        // FIXME: need a correct implementation of gradient to check with
      }
    }
    array2_4.eval("abc")
  }

  test("array2_5") {
    val array2_5 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

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

        // FIXME: need a correct implementation of gradient to check with
      }
    }
    array2_5.eval("abc")
  }

  test("array3") {
    val array3 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        // use random array as input
        val length = 2
        val v = Tensor.randinit(length)

        // calcuate gradient
        val grad = gradR(t => {val y = IF(t.x.data(0) > 0.0f) {t + t}{t * t}
        y.sum() })(v)

        // another way of implementing it
        val grad1 = gradR(t => (t + t).sum())(v)
        val grad2 = gradR(t => (t * t).sum())(v)
        if (v.data(0) > 0) Tensor.assertEqual(grad, grad1)
        else Tensor.assertEqual(grad, grad2)
      }
    }
    array3.eval("abc")
  }

  test("array4") {
    val array4 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        // use random array as input
        val length = 2
        Tensor.randseed()
        val v = Tensor.randinit(length)

        val halfv = Tensor.halves(length)
        val half = (new TensorR(halfv, Tensor.zeros(length)))
        // calculate gradient
        val grad = gradR(t => {val y = LOOP(t)(t => t.x.data(0) > 0.1f)(t => t * half)
        y.sum() })(v)
        // show gradient

        // FIXME: Implement the correct gradient and assertEqual
      }
    }
    array4.eval("abc")
  }

  test("array4_1") {
    val array4_1 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randinit(length)

        val half = new TensorR(Tensor.halves(length), Tensor.zeros(length))
        val grad = gradR(t => {
          val y = LOOPS(t)(3)(i => t => t * half )
          y.sum()
        })(v)

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
    array4_1.eval("abc")
  }

  test("array4_2") {
    // test using array data by closure
    val array4_2 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      def snippet(a: Rep[String]): Rep[Unit] = {

        // random initialization
        val length = 3
        val v = Tensor.randinit(length)

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

        // alternative implememetation
        val grad1 = gradR(t =>
            (t * TensorR(Tensor(data(0), ddim1)) * TensorR(Tensor(data(1), ddim1))).sum()
            )(v)
        // assertion
        Tensor.assertEqual(grad, grad1)
      }
    }

    array4_2.eval("abc")
  }


  test("array4_4") {
    val array4_4 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randinit(length)
        val u = Tensor.randinit(length, seed = Some(5))

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
        y(1).sum() + y(0).sum()})(Tensor.scalar(0))

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

    array4_4.eval("abc")
  }

  test("array5") {
    val array5 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randinit(length)

        val grad = gradR(t => (t * t).sum())(v)

        Tensor.assertEqual(grad, v * 2.0f)
      }
    }

    array5.eval("abc")
  }

  test("array6") {
    val array6 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randinit(length)

        val grad = gradR(t => (t / t).sum())(v)

        Tensor.assertEqual(grad, Tensor.zeros(length))
      }
    }
    array6.eval("abc")
  }

  test("array7") {
    val array7 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randinit(length)

        val grad = gradR(t => (t.tanh()).sum())(v)

        val e1 = v.tanh();
        val ee = Tensor.ones(length) - e1 * e1
        Tensor.assertEqual(grad, ee)
      }
    }
    array7.eval("abc")
  }

  test("array7_1") {
    val array7_1 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randinit(length)

        val grad = gradR(t => (t.sigmoid()).sum())(v)

        val e1 = v.sigmoid()
        val ee = (Tensor.ones(1) - e1) * e1
        Tensor.assertEqual(grad, ee)
      }
    }

    array7_1.eval("abc")
  }

  test("array8"){
    val array8 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randinit(length)

        val grad = gradR(t => (t.exp()).sum())(v)

        Tensor.assertEqual(grad, v.exp())
      }
    }

    array8.eval("abc")

  }

  test("array9") {
    val array9 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randPositive(length)

        val grad = gradR(t => (t.log()).sum())(v)

        Tensor.assertEqual(grad, Tensor.ones(length) / v)
      }
    }

    array9.eval("abc")
  }

  test("array10") {
    val array10 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randinit(length)

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

        val grad1 = gradR(t =>
            (t * TensorR(Tensor(arra(0), length)) * TensorR(Tensor(arra(1), length))).sum()
            )(v)

        Tensor.assertEqual(grad, grad1)
      }
    }

    array10.eval("abc")
  }

  test("array11") {
    val array11 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randinit(length)

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
         LOOPT(0)(x)(lch1, rch1){ (l: TensorR, r: TensorR, i: Rep[Int]) =>
           l * r * new TensorR(Tensor(arra(i), length), Tensor.zeros(length))
         }
       }

       val grad = gradR(t => model(t).sum())(v)

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

    array11.eval("abc")
  }

  test("array11_1") {
    val array11_1 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val v = Tensor.randinit(length)

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
         val tmp = LOOPTM(0)(in)(lch1, rch1){ (l: ArrayBuffer[TensorR], r: ArrayBuffer[TensorR], i: Rep[Int]) =>
           val curr = TensorR(Tensor(arra(i), length))
           val new_x = l(0) * r(0) * curr; val new_add = l(1) + r(1) + curr
           val out = new ArrayBuffer[TensorR](); out.append(new_x); out.append(new_add)
           out
         }
         tmp(0).sum() + tmp(1).sum()
       }

       val grad = gradR(t => model(t))(v)
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
    array11_1.eval("abc")
  }

  test("maxpool_test1") {
    val maxpool_test1 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

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
        for (i <- 0 until res.scalarCount: Rep[Range]) {
          // assertC(idx(i) ==  (i / res.strides(2)) * sR * input.strides(2) + sC * (i % res.strides(2)), s"Maxpool index invalid %d != %d (%d - %d)\\n", idx(i), (i / res.strides(2)) * sR * input.strides(2) + sC * (i % res.strides(2)), i / res.strides(2), i % res.strides(2))
        }

      }

    }
    maxpool_test1.eval("abc")
  }

  test("maxpool_back_test1") {
    val maxpool_back_test1 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

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

        val loss = gradR_loss(lossFun)(Tensor.scalar(0))

        Tensor.assertEqual(loss, Tensor.scalar(iPane * (iRow/sR) * (iCol/sC) * 1.0f), "MAXPOOL BACK 1 - LOSS")
        Tensor.assertEqual(varInput.d, Tensor.fill(Seq(iPane, iRow, iCol), (i: Seq[Rep[Int]]) => if (i(1) % sR == 0 && i(2) % sC == 0) 1.0f else 0.0f), "MAXPOOL BACK 1 - D")
      }
    }
    maxpool_back_test1.eval("abc")
  }

  test("dropout_test1") {
    val dropout_test1 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val iPane = 2
        val iRow = 16
        val iCol = 20
        val input = Tensor.rand(iPane, iRow, iCol)

        val (resAll, helper, size) = input.dropout(0.0f)
        // val (resNone, idxNone) = input.dropout(1.0f)

        Tensor.assertEqual(resAll, input, "DROPOUT 1")
        // Tensor.assertEqual(resNone, Tensor.zeros_like(input), "DROPOUT 2")
        val idxAll = Tensor(helper, input.shape: _*)
        for (i <- 0 until input.scalarCount: Rep[Range]) {
          assertC(idxAll.data(i) == 1.0f, "idxAll incorrect %.3f != 1\\n", idxAll.data(i))
          // assertC(idxNone.data(i) == 0.0f, "idxNone incorrect %.3f != 0\\n", idxNone.data(i))
        }
      }
    }

    dropout_test1.eval("abc")
  }

  test("dropout_back_test1") {
    val dropout_back_test1 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

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
        Tensor.assertEqual(varInput.d, Tensor.ones_like(input), "DROPOUT BACK 1 - D")
      }
    }
    dropout_back_test1.eval("abc")
  }

  val gen_dir = "/tmp/"

  test("op_conv_forward") {
    val deb = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(1, 1, 4, 4), 1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8)
        val kernel = Tensor.fromData(Seq(1, 1, 2, 2), 1,2,3,4)
        val bias = Tensor.zeros(1)
        val strides = Seq(2, 2)
        val pads = Seq(0,0,0,0)
        val (output, finputOption) = input.conv2D_batch(kernel, Some(bias), strides, pads)
        // output.print("output")
        // assert equal
        val expect = Tensor.fromData(Seq(1,1,2,2), 44, 64, 44, 64)
        Tensor.assertEqual(expect, output, "expect and output are")
      }
    }
    runTest(deb)
  }

  test("op_conv_pad") {
    val deb = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.ones(1, 1, 4, 4)
        val kernel = Tensor.ones(1, 1, 3, 3)
        val bias = Tensor.zeros(1)
        val strides = Seq(3, 3)
        val pads = Seq(1, 1, 1, 1)
        val (output, finputOption) = input.conv2D_batch(kernel, Some(bias), strides, pads)

        // assert equal
        val expect = Tensor.fromData(Seq(1,1,2,2), 4.0f, 4.0f, 4.0f, 4.0f)
        Tensor.assertEqual(expect, output, "expect and output are")
      }
    }
    runTest(deb)
  }

  test("op_conv_pad2") {
    val deb = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(1,1,4,4),1,2,3,4,5,6,7,8,2,3,4,5,6,7,8,9)
        val kernel = Tensor.ones(1, 1, 3, 3)
        val bias = Tensor.zeros(1)
        val strides = Seq(3, 3)
        val pads = Seq(1, 1, 1, 1)
        val (output, finputOption) = input.conv2D_batch(kernel, Some(bias), strides, pads)

        // assert equal
        val expect = Tensor.fromData(Seq(1,1,2,2), 14.0f, 22.0f, 18.0f, 26.0f)
        Tensor.assertEqual(expect, output, "expect and output are")
      }
    }
    runTest(deb)
  }

  test("op_conv_pad_nobias") {
    val deb = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.ones(1, 1, 4, 4)
        val kernel = Tensor.ones(1, 1, 3, 3)
        val strides = Seq(3, 3)
        val pads = Seq(1, 1, 1, 1)
        val (output, finputOption) = input.conv2D_batch(kernel, None, strides, pads)

        val expect = Tensor.fromData(Seq(1,1,2,2), 4.0f, 4.0f, 4.0f, 4.0f)
        Tensor.assertEqual(expect, output, "expect and output are")
      }
    }
    runTest(deb)
  }

  test("backprop_op_conv") {

    val deb = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = TensorR(Tensor.ones(1,1,3,3))
        val kernel = TensorR(Tensor.ones(1,1,2,2))
        val bias = TensorR(Tensor.zeros(1))
        val strides = Seq(1,1)
        val pads = Seq(0,0,0,0)

        def lossFun(x: TensorR) = {
          val output = input.convBBP(kernel, Some(bias), strides, pads)
          output.sum()
        }
        gradR_loss(lossFun)(Tensor.zeros(1))

        // assert equal
        val expect_input_grad = Tensor.fromData(Seq(1,1,3,3),
          1.0f, 2.0f, 1.0f, 2.0f, 4.0f, 2.0f, 1.0f, 2.0f, 1.0f)
        val expect_kernel_grad = Tensor.fill(Seq(1, 1, 2, 2), 4.0f)
        val expect_bias_grad = Tensor.fromData(Seq(1), 4.0f)
        Tensor.assertEqual(expect_input_grad, input.d, "expect and input.gradient are")
        Tensor.assertEqual(expect_kernel_grad, kernel.d, "expect and kernel.gradient are")
        Tensor.assertEqual(expect_bias_grad, bias.d, "expect and bias.gradient are")
      }
    }
    runTest(deb)
  }

  test("backprop_op_conv_pad") {

    val deb = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = TensorR(Tensor.ones(1,1,4,4))
        val kernel = TensorR(Tensor.ones(1,1,3,3))
        val bias = TensorR(Tensor.zeros(1))
        val strides = Seq(3,3)
        val pads = Seq(1,1,1,1)

        def lossFun(x: TensorR) = {
          val output = input.convBBP(kernel, Some(bias), strides, pads)
          output.sum()
        }
        val loss = gradR_loss(lossFun)(Tensor.zeros(1))

        // assert equal
        val expect_input_grad = Tensor.fromData(Seq(1,1,4,4),
          1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f)
        val expect_kernel_grad = Tensor.fromData(Seq(1,1,3,3),
          1.0f, 2.0f, 1.0f, 2.0f, 4.0f, 2.0f, 1.0f, 2.0f, 1.0f)
        val expect_bias_grad = Tensor.fromData(Seq(1), 4.0f)
        Tensor.assertEqual(expect_input_grad, input.d, "expect and input.gradient are")
        Tensor.assertEqual(expect_kernel_grad, kernel.d, "expect and kernel.gradient are")
        Tensor.assertEqual(expect_bias_grad, bias.d, "expect and bias.gradient are")
      }
    }
    val debug_file = new PrintWriter(new File(gen_dir + "backprop_conv_pad.cpp"))
    debug_file.println(deb.code)
    debug_file.flush()

    runTest(deb)
  }

  test("averagePool_backprop") {
    val deb = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = TensorR(Tensor.ones(1,1,4,4))
        def lossFun(x: TensorR) = {
          input.averagePoolBK(Seq(2, 2), Seq(2, 2), None).sum()
        }
        gradR_loss(lossFun)(Tensor.zeros(1))
        // assert equal
        val expected_grad = Tensor.fill(Seq(1, 1, 4, 4), 0.25f)
        Tensor.assertEqual(expected_grad, input.d, "expect and input.gradient are")

        input.clear_grad()
        def lossFun2(x: TensorR) = {
          input.averagePoolBK(Seq(2, 2), Seq(1, 1), None).sum()
        }
        gradR_loss(lossFun2)(Tensor.zeros(1))
        // assert equal
        val expected_grad2 = Tensor(Array[Float](0.25f, 0.5f, 0.5f, 0.25f, 0.5f, 1, 1, 0.5f, 0.5f, 1, 1, 0.5f, 0.25f, 0.5f, 0.5f, 0.25f), 1, 1, 4, 4)
        Tensor.assertEqual(expected_grad2, input.d, "")
      }
    }
    runTest(deb)
  }

  test("sum_on_any_dimension") {
    val deb = new LanternDriverC[String, Unit] {
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input1 = Tensor.ones(3,4,5)
        Tensor.assertEqual(input1.sum(dim = 0), Tensor.fill(Seq(4, 5), 3.0f))
        Tensor.assertEqual(input1.sum(dim = 1), Tensor.fill(Seq(3, 5), 4.0f))
        Tensor.assertEqual(input1.sum(dim = 2), Tensor.fill(Seq(3, 4), 5.0f))

        val input2 = TensorR(input1)
        gradR_loss(dummy => input2.sum(dim = 0).sum())(Tensor.zeros(1))
        Tensor.assertEqual(input2.d, Tensor.fill(Seq(3, 4, 5), 1.0f))
        gradR_loss(dummy => input2.sum(dim = 1).sum())(Tensor.zeros(1))
        Tensor.assertEqual(input2.d, Tensor.fill(Seq(3, 4, 5), 2.0f))
        gradR_loss(dummy => input2.sum(dim = 2).sum())(Tensor.zeros(1))
        Tensor.assertEqual(input2.d, Tensor.fill(Seq(3, 4, 5), 3.0f))

        val input3 = Tensor.ones(2,4,5,5)
        Tensor.assertEqual(input3.sum(3).sum(2).sum(0).resize(4, 1, 1), input3.batchNormAv() * 2 * 5 * 5)

        val input4 = TensorR(input3)
        gradR_loss(dummy => input4.batchNormAv().sum())(Tensor.zeros(1))
        Tensor.assertEqual(input4.d, Tensor.fill(Seq(2, 4, 5, 5), 1.0f / 50))

        val input5 = TensorR(new Tensor(Array(2.0f, 3.0f), Seq(2)))
        val input6 = TensorR(new Tensor(Array(2.0f, 3.0f), Seq(2)))
        gradR_loss(dummy => (input5 * input5).sum())(Tensor.zeros(1))
        gradR_loss(dummy => input6.square().sum())(Tensor.zeros(1))
        Tensor.assertEqual(input5.d, input6.d)
      }
    }
    runTest(deb)
  }

  test("elementwiseOpNoBroadCastSqrt") {

    val sqrt = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fill(Seq(3, 2, 3, 3), 16)
        val result = x.sqrt()
        val grad = gradR(x => x.sqrt())(x)

        val expected = Tensor.fill(Seq(3, 2, 3, 3), 4.0f)
        val expectedGrad = Tensor.fill(Seq(3, 2, 3, 3), 0.125f)
        Tensor.assertEqual(expected, result)
        Tensor.assertEqual(expectedGrad, grad)
      }
    }
    runTest(sqrt)
  }

  test("elementwiseOpNoBroadCastSquare") {
    val square = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fill(Seq(3, 2, 3, 3), 3)
        val result = x.square()
        val grad = gradR(x => x.square())(x)

        val expected = Tensor.fill(Seq(3, 2, 3, 3), 9.0f)
        val expectedGrad = Tensor.fill(Seq(3, 2, 3, 3), 6.0f)
        Tensor.assertEqual(expected, result)
        Tensor.assertEqual(expectedGrad, grad)
      }
    }
    runTest(square)
  }

  test("elementwiseOpNoBroadCastExp") {
    val exp = new LanternDriverC[String, Unit] {
      override val fileName = "lantern-cublas-exp"
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fill(Seq(3, 2, 3, 3), 0.5f)
        val result = x.exp()
        val grad = gradR(x => x.exp())(x)

        val expected = Tensor.fill(Seq(3, 2, 3, 3), 1.64872127f)
        val expectedGrad = Tensor.fill(Seq(3, 2, 3, 3), 1.64872127f)
        Tensor.assertEqual(expected, result)
        Tensor.assertEqual(expectedGrad, grad)
      }
    }
    runTest(exp)
  }

  test("elementwiseOpNoBroadCastLog") {
    val exp = new LanternDriverC[String, Unit] {
      override val fileName = "lantern-cublas-log"
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fill(Seq(3, 2, 3, 3), 2)
        val result = x.log()
        val grad = gradR(x => x.log())(x)

        val expected = Tensor.fill(Seq(3, 2, 3, 3), 0.6931471f)
        val expectedGrad = Tensor.fill(Seq(3, 2, 3, 3), 0.5f)
        Tensor.assertEqual(expected, result)
        Tensor.assertEqual(expectedGrad, grad)
      }
    }
    runTest(exp)
  }
}

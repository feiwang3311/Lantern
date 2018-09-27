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

  test("array0") {

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
        if (addr + 400 != addr1) printf("ERROR: addr did not increase by 800")
        if (addr != addr2) printf("ERROR: addr did not reset to the give value")
      }
    }
    array0.eval("abc")
  }

  test("array1") {

    val array1 = new DslDriverC[String, Unit] with TensorExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val length = 2
        val res = Tensor.randinit(length)
        val res2 = Tensor.randinit(length, seed = Some(5))

        val result = res dot res2

        // assertions
        if (res.data(0) * res2.data(0) + res.data(1) * res2.data(1) != result.data(0))
          println("ERROR: the dot product of two vectors is not correct")
      }
    }
    array1.eval("abc")
  }

  test("array1_1") {
    val array1_1 = new DslDriverC[String, Unit] with TensorExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val dim0 = 2
        val dim1 = 3
        val matrix = Tensor.rand(dim0, dim1)
        val vector = Tensor.randinit(dim1, seed = Some(4))

        //println("the result is:")
        val result = matrix dot vector
        //result.print()

        if (matrix(0, 0) * vector(0) + matrix(0, 1) * vector(1) + matrix(0, 2) * vector(2) != result(0))
          printf("ERROR: the matrix vector dot product is wrong on the first element of result, %.3f != %.3f\\n", matrix(0, 0) * vector(0) + matrix(0, 1) * vector(1) + matrix(0, 2) * vector(2), result(0))
        if (matrix(1, 0) * vector(0) + matrix(1, 1) * vector(1) + matrix(1, 2) * vector(2) != result(1))
          printf("ERROR: the matrix vector dot product is wrong on the second element of result, %.3f != %.3f\\n", matrix(1, 0) * vector(0) + matrix(1, 1) * vector(1) + matrix(1, 2) * vector(2), result(1))
      }
    }
    array1_1.eval("abc")
  }

  test("array2") {

    val array2 = new DslDriverC[String, Unit] with TensorExp {

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
        val loss = gradR_loss(dummy => tv dot tv)(Tensor.zeros(1))
        Tensor.assertEqual((v dot v), loss)
        Tensor.assertEqual(tv.d, grad)

      }
    }
    array2.eval("2.0f")
  }

  test("array2_1"){
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
    array2_1.eval("abc")
  }

  test("array2_2") {
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
    array2_2.eval("abc")
  }

  test("testTrans") {
    val testTrans = new DslDriverC[String, Unit] with TensorExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val idx = var_new(0)
        val t = Tensor.fill(seq => { idx += 1; idx }, 2, 3)

        Tensor.assertEqual(t.trans(), Tensor.fromData(1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f).resize(3, 2), "Transpose invalid")
      }
    }
    testTrans.eval("abs")
  }

  test("array2_3") {
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
       //printf("bh1\\n")
       //bh1.d.printRaw(hidden_size)

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
    array2_3.eval("abc")

  }

  test("array2_4"){

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
        // FIXME: need a correct implementation of gradient to check with
      }
    }

    array2_4.eval("abc")
  }

  test("array2_5") {
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

        // FIXME: need a correct implementation of gradient to check with
      }
    }
    array2_5.eval("abc")
  }

  test("array3") {
    val array3 = new DslDriverC[String, Unit] with TensorExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        // use random array as input
        val length = 2
        val v = Tensor.randinit(length)
        //v.print()

        // calcuate gradient
        val grad = gradR(t => {val y = IF(t.x.data(0) > 0.0f) {t + t}{t * t}
        y.sum() })(v)

        // another way of implementing it
        val grad1 = gradR(t => (t + t).sum())(v)
        val grad2 = gradR(t => (t * t).sum())(v)
        if (v(0) > 0) Tensor.assertEqual(grad, grad1)
        else Tensor.assertEqual(grad, grad2)
      }
    }
    //println(array3.code)
    array3.eval("abc")
  }

  test("array4") {
    val array4 = new DslDriverC[String, Unit] with TensorExp {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        // use random array as input
        val length = 2
        Tensor.randseed()
        val v = Tensor.randinit(length)
        v.print()

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
    array4.eval("abc")
  }

  test("array4_1") {
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

    println("run test case in array5")
    array5.eval("abc")
  }

  test("array6") {
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
    array6.eval("abc")
  }

  test("array7") {
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
    array7.eval("abc")
  }

  test("array7_1") {
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

    array7_1.eval("abc")
  }

  test("array8"){
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

    array8.eval("abc")

  }

  test("array9") {

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

    array9.eval("abc")
  }

  test("array10") {
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

    array10.eval("abc")
  }

  test("array11") {
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

    array11.eval("abc")
  }

  test("array11_1") {

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
    array11_1.eval("abc")
  }

  test("cnn_test1") {
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
      }
    }
    cnn_test1.eval("abc")
  }

  test("cnn_test2") {
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

        val res = input.conv2D(kernel, 1, 1)
        Tensor.assertEqual(res, Tensor.fill(1.0f, kOut, iRow - kRow + 1, iCol - kCol + 1), "CNN 2")
      }
    }

    cnn_test2.eval("abc")

  }

  test("cnn_test3") {
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
      }
    }

    cnn_test3.eval("abc")
  }

  test("cnn_back_test1") {

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

        Tensor.assertEqual(varKernel.d, Tensor.fill(resR * resC * 1.0f, kIn, kOut, kRow, kCol), "BACK 1 - KERNEL D")
      }
    }

    cnn_back_test1.eval("abc")
  }

  test("cnn_back_test2") {
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

        val resR = (iRow - kRow)/rS + 1
        val resC = (iCol - kCol)/cS + 1
        Tensor.assertEqual(loss, Tensor.scalar(resR * resC * 1.0f), "BACK 2 - LOSS")

        Tensor.assertEqual(varKernel.d, Tensor.fill(resR * resC * 1.0f, kIn, kOut, kRow, kCol), "BACK 2 - KERNEL D")
      }
    }

    cnn_back_test2.eval("abc")
  }

  test("cnn_back_test3") {
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

        val resR = (iRow - kRow)/rS + 1
        val resC = (iCol - kCol)/cS + 1
        Tensor.assertEqual(loss, Tensor.scalar(resR * resC * 1.0f), "BACK 2 - LOSS")

        Tensor.assertEqual(varKernel.d, Tensor.fill(resR * resC * 1.0f, kIn, kOut, kRow, kCol), "BACK 2 - KERNEL D")
      }
    }

    cnn_back_test3.eval("abc")
  }

  test("cnn_test4") {
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
      }
    }

    cnn_test4.eval("abc")
  }

  test("cnn_back_test4") {
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

        val resR = (iRow - kRow)/rS + 1
        val resC = (iCol - kCol)/cS + 1
        Tensor.assertEqual(loss, Tensor.scalar(kOut * resR * resC * 27.0f), "BACK 4 - LOSS")

        Tensor.assertEqual(varKernel.d, Tensor.fill(resR * resC * 1.0f, kOut, kIn, kRow, kCol), "BACK 4 - KERNEL D")
      }
    }

    cnn_back_test4.eval("abc")

  }

  test("cnn_test5") {
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
        Tensor.assertEqual(res, Tensor.fill(iPane * kRow * kCol * 1.0f, kOut, (iRow - kRow)/rS + 1, (iCol - kCol)/cS + 1), "CNN 5")
      }
    }

    cnn_test5.eval("abc")
  }

  test("cnn_back_test5") {
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

        val resR = (iRow - kRow)/rS + 1
        val resC = (iCol - kCol)/cS + 1
        Tensor.assertEqual(loss, Tensor.scalar(kOut * resR * resC * kIn * 1.0f), "BACK 5 - LOSS")

        Tensor.assertEqual(varKernel.d, Tensor.fill(resR * resC * 1.0f, kOut, kIn, kRow, kCol), "BACK 5 - KERNEL D")
      }
    }

    cnn_back_test5.eval("abc")
  }

  test("maxpool_test1") {
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
    maxpool_test1.eval("abc")
  }

  test("maxpool_back_test1") {
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

      }

    }

    maxpool_back_test1.eval("abc")

  }

  test("dropout_test1") {
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

    dropout_test1.eval("abc")
  }

  test("dropout_back_test1") {
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

      }
    }

    dropout_back_test1.eval("abc")
  }

  test("dropout_back_test2") {
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

      }
    }

    dropout_back_test2.eval("abc")
  }

  test("test_cnn_full1") {
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
            //weight.print(s"Before ${idx + 1}", derivative = true)
            weight.x.addMul(-0.5f, weight.d)
            //weight.print(s"After ${idx + 1}")
            weight.clear_grad()
            //printf("\\n")
          }
        }
      }
    }
    //test_cnn_full1.eval("abc")
  }


}
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

class BroadCastingTest extends LanternFunSuite {
  test("broadcasting") {
    val test1 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      def snippet(a: Rep[String]) = {test()}
      def test() = {
        def testDim(in1: Seq[Rep[Int]], in2: Seq[Rep[Int]], out1: Seq[Rep[Int]], out2: Seq[Rep[Int]], out3: Seq[Rep[Int]]) = {
          Tensor.dimBroadcast(in1, in2) match {
            case Some((a, b, c)) =>
              Tensor.assertShapeEqual(a, new Dimensions(out1))
              Tensor.assertShapeEqual(b, new Dimensions(out2))
              Tensor.assertShapeEqual(c, new Dimensions(out3))
          }
        }
        testDim(Seq(2, 3), Seq(2, 3), Seq(2, 3), Seq(2, 3), Seq(2, 3))
        testDim(Seq(2, 3), Seq(2, 3), Seq(2, 3), Seq(2, 3), Seq(2, 3))
        testDim(Seq(3), Seq(2, 3), Seq(1, 3), Seq(2, 3), Seq(2, 3))
        testDim(Seq(2, 3), Seq(3), Seq(2, 3), Seq(1, 3), Seq(2, 3))
        testDim(Seq(1, 3), Seq(2, 3), Seq(1, 3), Seq(2, 3), Seq(2, 3))
        testDim(Seq(2, 3), Seq(1, 3), Seq(2, 3), Seq(1, 3), Seq(2, 3))
        testDim(Seq(2, 3), Seq(2, 1), Seq(2, 3), Seq(2, 1), Seq(2, 3))
        testDim(Seq(2, 1), Seq(2, 3), Seq(2, 1), Seq(2, 3), Seq(2, 3))
        testDim(Seq(), Seq(2, 3), Seq(1, 1), Seq(2, 3), Seq(2, 3))
        testDim(Seq(2, 3), Seq(), Seq(2, 3), Seq(1, 1), Seq(2, 3))
        // assert(Tensor.dimBroadcast(Seq(2, 3), Seq(3, 3)) == dimfy(None))
        // assert(Tensor.dimBroadcast(Seq(2, 3), Seq(2, 3, 3)) == dimfy(None))
      }
    }
    runTest(test1)
  }

  test("add_broadcast1") {
    val test1 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      def snippet(a: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor(Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f), 2, 3)
        val tensor2 = Tensor(Array[Float](6,5,4,3,2,1), 2, 3)
        Tensor.assertEqual(tensor1 + tensor2, Tensor(Array[Float](7,7,7,7,7,7), 2, 3))
        Tensor.assertEqual(tensor2 + tensor1, Tensor(Array[Float](7,7,7,7,7,7), 2, 3))

        // test backprop
        val a = TensorR(tensor1)
        val b = TensorR(tensor2)
        def loss(dummy: TensorR) = (a + b).sum()
        gradR_loss(loss)(Tensor.zeros(1))
        Tensor.assertEqual(a.d, Tensor(Array[Float](1,1,1,1,1,1), 2, 3))
        Tensor.assertEqual(b.d, Tensor(Array[Float](1,1,1,1,1,1), 2, 3))
      }
    }
    runTest(test1)
  }

  test("add_broadcast2") {
    val test2 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      def snippet(a: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor(Array[Float](1,2,3,4,5,6), 2, 3)
        val tensor2 = Tensor(Array[Float](1,2), 2, 1)
        Tensor.assertEqual(tensor1 + tensor2, Tensor(Array[Float](2,3,4, 6,7,8), 2, 3))
        Tensor.assertEqual(tensor2 + tensor1, Tensor(Array[Float](2,3,4, 6,7,8), 2, 3))

        // test backprop
        val a = TensorR(tensor1)
        val b = TensorR(tensor2)
        def loss(dummy: TensorR) = (a + b).sum()
        gradR_loss(loss)(Tensor.zeros(1))
        Tensor.assertEqual(a.d, Tensor(Array[Float](1,1,1,1,1,1), 2, 3))
        Tensor.assertEqual(b.d, Tensor(Array[Float](3, 3), 2, 1))
      }
    }
    test2.eval("a")
  }

  test("add_broadcast3") {
    val test3 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      def snippet(a: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor(Array[Float](1,2,3,4,5,6), 2, 3)
        val tensor2 = Tensor(Array[Float](3,4,5), 1, 3)
        Tensor.assertEqual(tensor1 + tensor2, Tensor(Array[Float](4,6,8,7,9,11), 2,3))
        Tensor.assertEqual(tensor2 + tensor1, Tensor(Array[Float](4,6,8,7,9,11), 2,3))

        // test backprop
        val a = TensorR(tensor1)
        val b = TensorR(tensor2)
        def loss(dummy: TensorR) = (a + b).sum()
        gradR_loss(loss)(Tensor.zeros(1))
        Tensor.assertEqual(a.d, Tensor(Array[Float](1,1,1,1,1,1), 2, 3))
        Tensor.assertEqual(b.d, Tensor(Array[Float](2, 2, 2), 1, 3))
      }
    }
    test3.eval("a")
  }

  test("add_broadcast4") {
    val test4 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      def snippet(a: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor(Array[Float](1,2,3,4,5,6,7,8), 2,2,2)
        val tensor2 = Tensor(Array[Float](1,2), 2)
        Tensor.assertEqual(tensor1 + tensor2, Tensor(Array[Float](2,4,4,6,6,8,8,10), 2,2,2))
        Tensor.assertEqual(tensor2 + tensor1, Tensor(Array[Float](2,4,4,6,6,8,8,10), 2,2,2))

        // test backprop
        val a = TensorR(tensor1)
        val b = TensorR(tensor2)
        def loss(dummy: TensorR) = (a + b).sum()
        gradR_loss(loss)(Tensor.zeros(1))
        Tensor.assertEqual(a.d, Tensor(Array[Float](1,1,1,1,1,1,1,1), 2, 2, 2))
        Tensor.assertEqual(b.d, Tensor(Array[Float](4, 4), 2))
      }
    }
    test4.eval("a")
  }

  test("add_broadcast5") {
    val test5 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      def snippet(a: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor(Array[Float](1,2,3,4,5,6,7,8), 2, 2, 2)
        val tensor2 = Tensor(Array[Float](1,2,3,4), 2, 1, 2)
        val res = tensor1 + tensor2
        generateRawComment("ignore the rest for code inspection")
        Tensor.assertEqual(res, Tensor(Array[Float](2,4,4,6,8,10,10,12), 2,2,2))
        Tensor.assertEqual(tensor2 + tensor1, Tensor(Array[Float](2,4,4,6,8,10,10,12), 2,2,2))

        // test backprop
        val a = TensorR(tensor1)
        val b = TensorR(tensor2)
        def loss(dummy: TensorR) = (a + b).sum()
        gradR_loss(loss)(Tensor.zeros(1))
        Tensor.assertEqual(a.d, Tensor(Array[Float](1,1,1,1,1,1,1,1), 2, 2, 2))
        Tensor.assertEqual(b.d, Tensor(Array[Float](2,2,2,2), 2, 1, 2))
      }
    }
    test5.eval("a")
    // System.out.println(test5.code)
  }

  test("minus_broadcast5") {
    val test5 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName

      def snippet(a: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor(Array[Float](1,2,3,4,5,6,7,8), 2, 2, 2)
        val tensor2 = Tensor(Array[Float](1,2,3,4), 2, 1, 2)
        Tensor.assertEqual(tensor1 - tensor2, Tensor(Array[Float](0,0,2,2,2,2,4,4), 2,2,2))

        // test backprop
        val a = TensorR(tensor1)
        val b = TensorR(tensor2)
        def loss(dummy: TensorR) = (a - b).sum()
        gradR_loss(loss)(Tensor.zeros(1))
        Tensor.assertEqual(a.d, Tensor(Array[Float](1,1,1,1,1,1,1,1), 2, 2, 2))
        Tensor.assertEqual(b.d, Tensor(Array[Float](-2,-2,-2,-2), 2, 1, 2))
      }
    }
    test5.eval("a")
  }

  test("mult_broadcast5") {
    val test5 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName
      def snippet(a: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor(Array[Float](1,2,3,4,5,6,7,8), 2, 2, 2)
        val tensor2 = Tensor(Array[Float](1,2,3,4), 2, 1, 2)
        Tensor.assertEqual(tensor1 * tensor2, Tensor(Array[Float](1,4,3,8,15,24,21,32), 2,2,2))

        // test backprop
        val a = TensorR(tensor1)
        val b = TensorR(tensor2)
        def loss(dummy: TensorR) = (a * b).sum()
        gradR_loss(loss)(Tensor.zeros(1))

        Tensor.assertEqual(a.d, Tensor(Array[Float](1,2,1,2,3,4,3,4), 2, 2, 2))
        Tensor.assertEqual(b.d, Tensor(Array[Float](4,6,12,14), 2, 1, 2))
      }
    }
    test5.eval("a")
  }

  test("div_broadcast5") {
    val test5 = new LanternDriverC[String, Unit] {
      override val fileName = currentTestName
      def snippet(a: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor(Array[Float](1,2,3,4,5,6,7,8), 2, 2, 2)
        val tensor2 = Tensor(Array[Float](1,2,2,4), 2, 1, 2)
        Tensor.assertEqual(tensor1 / tensor2, Tensor(Array[Float](1,1,3,2,2.5f,1.5f,3.5f,2), 2,2,2))

        // test backprop
        val a = TensorR(tensor1)
        val b = TensorR(tensor2)
        def loss(dummy: TensorR) = (a / b).sum()
        gradR_loss(loss)(Tensor.zeros(1))
        Tensor.assertEqual(a.d, Tensor(Array[Float](1,0.5f,1,0.5f,0.5f,0.25f,0.5f,0.25f), 2, 2, 2))
        Tensor.assertEqual(b.d, Tensor(Array[Float](-4, -1.5f, -3, -0.875f), 2, 1, 2))
      }
    }
    test5.eval("a")
  }
}

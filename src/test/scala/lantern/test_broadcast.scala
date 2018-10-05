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

class BroadCastingTest extends FunSuite {
  test("broadcasting") {
    val test1 = new DslDriverC[String, Unit] with TensorExp {
      def snippet(a: Rep[String]) = {}
      def test() = {
        assert(Tensor.dimBroadcast(NSeq(2, 3), NSeq(2, 3)) == Some((NSeq(2, 3), NSeq(2, 3), NSeq(2, 3))))
        assert(Tensor.dimBroadcast(NSeq(3), NSeq(2, 3)) == Some((NSeq(1, 3), NSeq(2, 3), NSeq(2, 3))))
        assert(Tensor.dimBroadcast(NSeq(2, 3), NSeq(3)) == Some((NSeq(2, 3), NSeq(1, 3), NSeq(2, 3))))
        assert(Tensor.dimBroadcast(NSeq(1, 3), NSeq(2, 3)) == Some((NSeq(1, 3), NSeq(2, 3), NSeq(2, 3))))
        assert(Tensor.dimBroadcast(NSeq(2, 3), NSeq(1, 3)) == Some((NSeq(2, 3), NSeq(1, 3), NSeq(2, 3))))
        assert(Tensor.dimBroadcast(NSeq(2, 3), NSeq(2, 1)) == Some((NSeq(2, 3), NSeq(2, 1), NSeq(2, 3))))
        assert(Tensor.dimBroadcast(NSeq(2, 1), NSeq(2, 3)) == Some((NSeq(2, 1), NSeq(2, 3), NSeq(2, 3))))
        assert(Tensor.dimBroadcast(NSeq(), NSeq(2, 3)) == Some((NSeq(1, 1), NSeq(2, 3), NSeq(2, 3))))
        assert(Tensor.dimBroadcast(NSeq(2, 3), NSeq()) == Some((NSeq(2, 3), NSeq(1, 1), NSeq(2, 3))))
        assert(Tensor.dimBroadcast(NSeq(2, 3), NSeq(3, 3)) == None)
        assert(Tensor.dimBroadcast(NSeq(2, 3), NSeq(2, 3, 3)) == None)
      }
    }
    test1.test()
  }

  test("add_broadcast1") {
    val test1 = new DslDriverC[String, Unit] with TensorExp {
      def snippet(a: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor(Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f), 2, 3)
        val tensor2 = Tensor(Array[Float](6,5,4,3,2,1), 2, 3)
        Tensor.assertEqual(tensor1 + tensor2, Tensor(Array[Float](7,7,7,7,7,7), 2, 3))
      }
    }
    test1.eval("a")
  }

  test("add_broadcast2") {
    val test2 = new DslDriverC[String, Unit] with TensorExp {
      def snippet(a: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor(Array[Float](1,2,3,4,5,6), 2, 3)
        val tensor2 = Tensor(Array[Float](1,2), 2, 1)
        Tensor.assertEqual(tensor1 + tensor2, Tensor(Array[Float](2,3,4, 6,7,8), 2, 3))
        Tensor.assertEqual(tensor2 + tensor1, Tensor(Array[Float](2,3,4, 6,7,8), 2, 3))
      }
    }
    test2.eval("a")
  }

  test("add_broadcast3") {
    val test3 = new DslDriverC[String, Unit] with TensorExp {
      def snippet(a: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor(Array[Float](1,2,3,4,5,6), 2, 3)
        val tensor2 = Tensor(Array[Float](3,4,5), 1, 3)
        Tensor.assertEqual(tensor1 + tensor2, Tensor(Array[Float](4,6,8,7,9,11), 2,3))
        Tensor.assertEqual(tensor2 + tensor1, Tensor(Array[Float](4,6,8,7,9,11), 2,3))
      }
    }
    test3.eval("a")
  }

  test("add_broadcast4") {
    val test4 = new DslDriverC[String, Unit] with TensorExp {
      def snippet(a: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor(Array[Float](1,2,3,4,5,6,7,8), 2,2,2)
        val tensor2 = Tensor(Array[Float](1,2), 2)
        Tensor.assertEqual(tensor1 + tensor2, Tensor(Array[Float](2,4,4,6,6,8,8,10), 2,2,2))
        Tensor.assertEqual(tensor2 + tensor1, Tensor(Array[Float](2,4,4,6,6,8,8,10), 2,2,2))
      }
    }
    test4.eval("a")
  }

  test("add_broadcast5") {
    val test5 = new DslDriverC[String, Unit] with TensorExp {
      def snippet(a: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor(Array[Float](1,2,3,4,5,6,7,8), 2, 2, 2)
        val tensor2 = Tensor(Array[Float](1,2,3,4), 2, 1, 2)
        val res = tensor1 + tensor2
        generate_comment("ignore the rest for code inspection")
        Tensor.assertEqual(res, Tensor(Array[Float](2,4,4,6,8,10,10,12), 2,2,2))
        Tensor.assertEqual(tensor2 + tensor1, Tensor(Array[Float](2,4,4,6,8,10,10,12), 2,2,2))
      }
    }
    test5.eval("a")
    System.out.println(test5.code)
  }
}
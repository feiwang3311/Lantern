package lantern

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._
import org.scalatest.FunSuite

class GregTest extends FunSuite {

  test("1") {
    val test1 = new DslDriverC[String, Unit] with TestExp {
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val arr1 = NewArray[Int](1)
        val arr2 = NewArray[Int](1)

        arr1(0) = 2
        arr2(0) = 3

        val t1 = new Test(arr1, Dimensions(1))
        val t2 = new Test(arr2, Dimensions(1))

        val t3 = t1 + t2

        t3.printRaw()
      }
    }
    //println("Test 1")
    //test1.eval("1")
  }

  test("2") {
    val test2 = new DslDriverC[String, Unit] with TestExp {
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val arr1 = NewArray[Float](1)
        val arr2 = NewArray[Float](1)

        arr1(0) = 42.0f
        arr2(0) = 89.0f

        val t1 = new Test(arr1, Dimensions(1))
        val t2 = new Test(arr2, Dimensions(1))

        val t3 = t1 + t2

        t3.printRaw()
      }
    }
    //println("Test 2")
    //test2.eval("2")
  }

  test("3") {
    val test3 = new DslDriverC[String, Unit] with TestExp {
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val arr1 = NewArray[Int](10)
        val arr2 = NewArray[Int](10)

        for (i <- DataLoop(10)) {
          arr1(i) = i
          arr2(i) = 25 - i
        }

        val t1 = new Test(arr1, Dimensions(2, 5))
        val t2 = new Test(arr2, Dimensions(2, 5))

        val t3 = t1 + t2

        t3.printRaw()
      }
    }
    //println("Test 3")
    //test3.eval("2")
  }


  test("4") {
    val test4 = new DslDriverC[String, Unit] with TestExp {
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val arr1 = NewArray[Int](10)

        for (i <- DataLoop(10)) {
          arr1(i) = 5 - i
        }

        val t1 = new Test(arr1, Dimensions(10))
        t1.clipAt(1)

        t1.printRaw()
      }
    }
    //println("Test 4")
    //test4.eval("2")
  }

}

package lantern

import scala.virtualization.lms._
import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.collection.mutable.ArrayBuffer
import scala.collection.{Seq => NSeq}
import scala.math._

import org.scalatest.FunSuite

import java.io.PrintWriter
import java.io.File


class LanternFunSuite extends FunSuite {

  def runTest(snippet: DslDriverC[String, Unit]) = {
    import scala.util.Random
    object OneTimeCode {
      def apply(length: Int = 6) = {
        Random.alphanumeric.take(length).mkString("")
      }
    }
    val name = OneTimeCode()
    val test = new PrintWriter(new File(s"/tmp/$name.cpp"))
    test.println(snippet.code)
    test.flush()
    new java.io.File(s"/tmp/$name").delete
    import scala.sys.process._
    System.out.println("Compile C++ code")
    (s"g++ -std=c++11 -O1 /tmp/$name.cpp -o /tmp/$name": ProcessBuilder).lines.foreach(System.out.println)
    System.out.println("Run C++ code")
    (s"/tmp/$name a": ProcessBuilder).lines.foreach(System.out.println)
  }

}
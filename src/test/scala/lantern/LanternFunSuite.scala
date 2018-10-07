package lantern

import scala.virtualization.lms._
import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.collection.mutable.ArrayBuffer
import scala.collection.{Seq => NSeq}
import scala.math._
import scala.sys.process._
import scala.util.Random

import org.scalatest.FunSuite

import java.io.PrintWriter
import java.io.File

class LanternFunSuite extends FunSuite {
  def runTest(snippet: DslDriverC[String, Unit]) {
    val cppFileName = s"/tmp/lantern-snippet.cpp"
    val binaryFileName = s"/tmp/lantern-snippet"
    val test = new PrintWriter(new File(cppFileName))
    test.println(snippet.code)
    test.flush()
    new java.io.File(binaryFileName).delete

    System.out.println("Compile C++ code")
    (s"g++ -std=c++11 -O1 $cppFileName -o $binaryFileName": ProcessBuilder).lines.foreach(System.out.println)
    System.out.println("Run C++ code")
    (s"$binaryFileName a": ProcessBuilder).lines.foreach(System.out.println)
  }
}
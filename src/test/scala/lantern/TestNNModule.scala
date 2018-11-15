package lantern

import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

import org.scalatest.FunSuite

import java.io.PrintWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintStream;

import scala.collection.mutable.Map;
import scala.collection.mutable.ArrayBuffer;

class ModuleTest extends FunSuite {

  test("reflection") {

    val test1 = new DslDriverC[String, Unit] with NNModule {

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        case class Linear(val inSize: Int, val outSize: Int, val name: String = "linear1d") extends Module {
          val weight = TensorR(Tensor.zeros(inSize, outSize))
          val bias = TensorR(Tensor.zeros(outSize))
          def apply(in: TensorR): TensorR @diff = in.dot(weight) + bias
        }
        case class MyModule(val inSize: Int, val outSize: Int, val name: String = "MyModule") extends Module {
          val weight = TensorR(Tensor.zeros(inSize, outSize))
          val bias = TensorR(Tensor.zeros(outSize))
          val other = Linear(5, 6)
        }

        val li = MyModule(3, 4)
        li.registerParameters("")
        li.forEachNamedParameter { case (name, (tensorR, _)) => System.out.println(s"$name: $tensorR") }
      }
    }
    test1.eval("a")
  }

  test("option") {
    val test = new DslDriverC[String, Unit] with NNModule {
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        case class Linear1(val inSize: Int, val outSize: Int, val name: String = "linear1d") extends Module {
          val weight = TensorR(Tensor.zeros(inSize, outSize))
          val bias = TensorR(Tensor.zeros(outSize))
          def apply(in: TensorR): TensorR @diff = in.dot(weight) + bias
        }

        case class Linear(val inSize: Int, val outSize: Int, bias: Option[TensorR], val name: String = "linear") extends Module {
          val weight = TensorR(Tensor.zeros(inSize, outSize))
          val other = Some(1)
          val other2 = Some(Linear1(47,48))
          def apply(in:TensorR):TensorR@diff = bias match {
            case Some(b) => in.dot(weight) + b
            case None => in.dot(weight)
          }
        }
        val testModule = Linear(3, 4, Some(TensorR(Tensor.zeros(4))))
        testModule.registerParameters("")
        testModule.forEachNamedParameter {case (name, (tr, _)) => System.out.println((s"$name: $tr"))}
      }
    }
    test.eval("a")
  }

  test("seq") {
    val test = new DslDriverC[String, Unit] with NNModule {
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        case class Linear1(val inSize: Int, val outSize: Int, val name: String = "linear1d") extends Module {
          val weight = TensorR(Tensor.zeros(inSize, outSize))
          val bias = TensorR(Tensor.zeros(outSize))
          def apply(in: TensorR): TensorR @diff = in.dot(weight) + bias
        }

        case class Linear(val inSize: Int, val outSize: Int, val name: String = "linear") extends Module {
          val weight = TensorR(Tensor.zeros(inSize, outSize))
          val other = Seq(1, 2, 3)
          val others = Seq(TensorR(Tensor.zeros(3, 4)), TensorR(Tensor.zeros(5,6)))
          val others1 = Seq(Linear1(3, 4), Linear1(5,6))
          def apply(in:TensorR):TensorR@diff = in.dot(weight)
        }
        val testModule = Linear(3, 4)
        testModule.registerParameters("")
        testModule.forEachNamedParameter {case (name, (tr, _)) => System.out.println((s"$name: $tr"))}
      }
    }
    test.eval("a")
  }

  test("module") {
    val test = new DslDriverC[String, Unit] with NNModule {
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        case class Linear(val inSize: Int, val outSize: Int, val name: String = "linear1d") extends Module {
          val weight = TensorR(Tensor.zeros(outSize, inSize))
          val bias = TensorR(Tensor.zeros(outSize))
          def apply(in: TensorR): TensorR @diff = weight.dot(in) + bias
        }

        val testModule = new Module {
          val name = "test"
          val bias = TensorR(Tensor.zeros(4))
          val module = Linear(6, 4)
          def apply(in: TensorR): TensorR@diff = {
            val temp = module(in)
            temp plusBias bias
          }
        }

        testModule.registerParameters(testModule.name + "/")
        testModule.forEachNamedParameter{ case(name, (tr, _)) => System.out.println(s"$name: $tr") }
        val x = Tensor.zeros(6)
        gradR(x => testModule(x))(x)
        ()
      }
    }
    test.eval("a")
  }

  test("stackOverFlow") {
    val test = new DslDriverC[String, Unit] with NNModule {
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        case class TestNested(val name: String = "TestNested") extends Module {
          val randomName = new Module {
            val name: String = "fully_connected"
            val dummy = TensorR(Tensor.zeros(1))
            def apply(in: TensorR) = {
              in.resize(in.x.shape(0) * in.x.shape(1), in.x.shape(2))
              // in.resize(20, 10)
            }
          }
        }

        val module = TestNested()
        module.registerParameters(module.name + "/")
        module.forEachNamedParameter{case(name, (tr, _)) => System.out.println(s"$name: $tr")}
      }
    }
    test.eval("a")
  }

}

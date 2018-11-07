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

import onnx.onnx_ml;
import scala.collection.mutable.Map;

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
}

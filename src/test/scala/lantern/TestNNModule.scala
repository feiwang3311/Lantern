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
import scala.collection.mutable.ArrayBuffer;

class ModuleTest extends FunSuite {

  test("reflection") {

    val test1 = new LanternDriverC[String, Unit] with NNModule {

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

        // check registration
        val nameSets = scala.collection.mutable.Set[String]()
        li.forEachNamedParameter{case (name: String, _) => nameSets.add(name); ()}
        assert(nameSets == Set("bias", "weight", "other/weight", "other/bias"))
      }
    }
    test1.eval("a")
  }

  test("option") {
    val test = new LanternDriverC[String, Unit] with NNModule {
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

        // check registration
        val nameSets = scala.collection.mutable.Set[String]()
        testModule.forEachNamedParameter{case (name: String, _) => nameSets.add(name); ()}
        assert(nameSets == Set("other2/bias", "bias", "weight", "other2/weight"))
      }
    }
    test.eval("a")
  }

  test("seq") {
    val test = new LanternDriverC[String, Unit] with NNModule {
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

        // check registration
        val nameSets = scala.collection.mutable.Set[String]()
        testModule.forEachNamedParameter{case (name: String, _) => nameSets.add(name); ()}
        assert(nameSets == Set("others_index_0", "weight", "others_index_1", "others1_index_1/weight", "others1_index_1/bias", "others1_index_0/bias", "others1_index_0/weight"))
      }
    }
    test.eval("a")
  }

  test("module") {
    val test = new LanternDriverC[String, Unit] with NNModule {
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

        // check registration
        val nameSets = scala.collection.mutable.Set[String]()
        testModule.forEachNamedParameter{case (name: String, _) => nameSets.add(name); ()}
        assert(nameSets == Set("test/bias", "test/module/bias", "test/module/weight"))

        val x = Tensor.zeros(6)
        gradR(x => testModule(x))(x)
        ()
      }
    }
    test.eval("a")
  }

  test("newModule") {
    val test = new LanternDriverC[String, Unit] with NNModule {
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        val rnnHiddenSize = 50
        val numClasses = 10

        val fc = new Module {
          val name: String = "fully_connected"
          val bn = BatchNorm1D(rnnHiddenSize)
          val linear = Linear1D(rnnHiddenSize, numClasses, bias=false)
          def apply(in: TensorR): TensorR @diff = {
            val shape0 = in.x.shape(0)
            val shape1 = in.x.shape(1)
            val shape2 = in.x.shape(2)
            val in2D = in.resize(shape0 * shape1, shape2)
            val out2D = linear(bn(in2D))
            out2D.resize(shape0, shape1, shape2)
          }
        }
        fc.registerParameters(fc.name + "/")

        // check registration
        val nameSets = scala.collection.mutable.Set[String]()
        fc.forEachNamedParameter{case (name: String, _) => nameSets.add(name); ()}
        assert(nameSets == Set("fully_connected/bn/bias", "fully_connected/bn/scale", "fully_connected/linear/weight"))
      }
    }
    test.eval("a")
  }

  test("stackOverFlow") {
    val test = new LanternDriverC[String, Unit] with NNModule {
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        case class TestNested(val name: String = "TestNested") extends Module {
          val randomName = new Module {
            val name: String = "fully_connected"
            val dummy = TensorR(Tensor.zeros(1))
            def apply(in: TensorR) = {
              in.resize(in.x.shape(0) * in.x.shape(1), in.x.shape(2))
            }
          }
        }

        val module = TestNested()
        module.registerParameters(module.name + "/")

        // check registration
        val nameSets = scala.collection.mutable.Set[String]()
        module.forEachNamedParameter{case (name: String, _) => nameSets.add(name); ()}
        assert(nameSets == Set("TestNested/randomName/dummy"))
      }
    }
    test.eval("a")
  }

  test("newModuleNested2") {
    val test = new LanternDriverC[String, Unit] with NNModule {
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {

        val rnnHiddenSize = 50
        val numClasses = 10

        case class TestNested(val name: String = "TestNested") extends Module {
          val conv = new Module {
            val name = "conv"
            val conv1 = Conv2D(1, 32, Seq(41, 11), stride = Seq(2, 2), pad = Seq(20, 5))
            val bn1 = BatchNorm2D(32)
            val conv2 = Conv2D(32, 32, Seq(21, 11), stride = Seq(2, 1), pad = Seq(10, 5))
            val bn2 = BatchNorm2D(32)
            def apply(in: TensorR, lengths: Rep[Array[Int]]): TensorR @diff = {
              // NOTE: This function assume that the lengths array is already on GPU
              val step1 = conv1(in).mask4D(lengths)
              val step2 = bn1(step1).mask4D(lengths).hardTanh(0, 20, inPlace = true)
              val step3 = conv2(step2).mask4D(lengths)
              bn2(step3).hardTanh(0, 20, inPlace = true)
            }
          }
        }

        val module = TestNested()

        module.registerParameters(module.name + "/")
        module.forEachNamedParameter{case(name, (tr, _)) => System.out.println(s"$name: $tr")}

        // check registration
        val nameSets = scala.collection.mutable.Set[String]()
        module.forEachNamedParameter{case (name: String, _) => nameSets.add(name); ()}
        assert(nameSets == Set("TestNested/conv/bn1/scale", "TestNested/conv/bn1/bias",
          "TestNested/conv/bn2/bias", "TestNested/conv/bn2/scale", "TestNested/conv/conv1/kernel",
          "TestNested/conv/conv1/bias", "TestNested/conv/conv2/bias", "TestNested/conv/conv2/kernel"))
      }
    }
    test.eval("a")
  }
}

package lantern

import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Map
import scala.collection.{Seq => NSeq}
import scala.math._

trait NNModule extends TensorExp {

  abstract class Module {
    val name: String
    val parameters = Map[String, (TensorR, Option[Tensor])]() // option of tensor is for axillary info needed in gradient descent
    val modules = Map[String, Module]()
    def apply(in: TensorR): TensorR @diff
    def regTensorWithName(name: String)(tensor: TensorR) = {
      parameters += (name -> (tensor, None))
      tensor
    }
    def regModuleWithName(nameScope: String)(module: Module) = {
      modules += (nameScope + "/" + module.name -> module)
      module
    }
    def forEachNamedParameter(f: (String, (TensorR, Option[Tensor])) => Unit): Unit = {
      parameters.foreach{ case (k, v) => f(k, v) }
      for ((_, module) <- modules) module.forEachNamedParameter(f)
    }
    def enrichParameter(): Unit = {
      for ((k, (tensorR, _)) <- parameters) parameters(k) = (tensorR, Some(Tensor.zeros(tensorR.x)))
      for ((_, module) <- modules) module.enrichParameter()
    }
    def forEachParameter(f: TensorR => Unit) = forEachNamedParameter{case (_, (tensorR, _)) => f(tensorR)}
    def forEachPairParameter(f: (TensorR, Option[Tensor]) => Unit) = forEachNamedParameter{case (_, (tr, t)) => f(tr, t)}
  }

  case class Linear1D(val inSize: Int, val outSize: Int, val name: String = "linear1d") extends Module {
    val scale: Float = 1.0f / inSize
    val weight = regTensorWithName("w")(TensorR(Tensor.rand(scale, outSize, inSize)))
    val bias = regTensorWithName("b")(TensorR(Tensor.zeros(outSize)))
    def apply(in: TensorR): TensorR @diff = weight.dot(in) + bias
  }

  case class Conv2D(val inChannel: Int, val outChannel: Int, val kernelSize: NSeq[Int], val stride: NSeq[Int] = NSeq(1, 1), val pad: Int = 0, val name: String = "conv2d") extends Module {
    assert(kernelSize.size == 2, "kernel_size should be Seq[Int] of size 2")
    assert(stride.size == 2, "stride should be Seq[Int] of size 2")
    val scale: Float = 1.0f / (inChannel * kernelSize.head * kernelSize.last)
    val kernel = regTensorWithName("k")(TensorR(Tensor.rand(scale, outChannel, inChannel, kernelSize.head, kernelSize.last)))
    val bias = regTensorWithName("b")(TensorR(Tensor.zeros(outChannel)))
    def apply(in: TensorR): TensorR @diff = in.convBBP(kernel, bias, stride, NSeq(pad, pad, pad, pad))
  }

  abstract class Optim {
    val module: Module
    def step_func: (TensorR, Option[Tensor]) => Unit
    def zero_grad() = module.forEachParameter(_.clear_grad())
    def step() = module.forEachPairParameter(step_func)
  }
  case class SGD(val module: Module, val learning_rate: Float, val gradClip: Float = 1.0f, val descent: Boolean = true) extends Optim {
    def step_func = { case (tr, _) =>
      tr.clip_grad(gradClip)
      if (descent)
        tr.x -= tr.d * learning_rate
      else
        tr.x -= tr.d * learning_rate
      tr.clear_grad()
    }
  }
  case class Adagrad(val module: Module, val learning_rate: Float, val gradClip: Float = 1.0f, val descent: Boolean = true) extends Optim {
    module.enrichParameter()
    def step_func = { case (tr, Some(t)) =>
      tr.clip_grad(gradClip)
      t += tr.d * tr.d
      if (descent)
        tr.x -= tr.d * learning_rate / (t + 1e-8f).sqrt()
      else
        tr.x += tr.d * learning_rate / (t + 1e-8f).sqrt()
      tr.clear_grad()
    }
  }

}

/*
import scala.reflect.runtime.universe._
import java.lang.reflect.Field

abstract class Module {
  def getParamters: Seq[Int] = {
    val allParams = this.getClass.getDeclaredFields

    // Get only the ints
    val onlyInt = allParams.filter { _.getType == classOf[Int] }
    val subModules = allParams.filter { _.getType.isAssignableFrom(classOf[Module]) }

    def oops[T](field: Field)(read: Field => T) = {
      val acc = field.isAccessible
      field.setAccessible(true);
      val res = read(field)
      field.setAccessible(acc)
      res
    }

    val values = onlyInt.map { field => oops[Int](field) { _.getInt(this) } }
    val subValues = subModules.flatMap( field => oops[Seq[Int]](field) { _.get(this).asInstanceOf[Module].getParamters })

    values ++ subValues
  }
}

object Empty extends Module

class Add3(x: Int, y: Module) extends Module {

  val three = 3
  val four = 4.0

  def print() = { println(y); println(x) }
}

val x = new Add3(4, Empty)
x.getParamters
*/
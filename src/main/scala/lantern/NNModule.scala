package lantern

import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Map
import scala.math._

import scala.reflect.runtime.universe._
import java.lang.reflect.Field

trait NNModule extends TensorDsl {

  abstract class Module {
    val name: String
    val parameters = Map[String, (TensorR, Option[Tensor])]() // option of tensor is for axillary info needed in gradient descent
    val modules = Map[String, Module]()
    def forEachNamedParameter(f: (String, (TensorR, Option[Tensor])) => Unit): Unit = {
      parameters.foreach{ case (k, v) => f(k, v) }
      for ((_, module) <- modules) module.forEachNamedParameter(f)
    }
    def enrichParameter(): Unit = {
      for ((k, (tensorR, _)) <- parameters) parameters(k) = (tensorR, Some(Tensor.zeros_like(tensorR.x)))
      for ((_, module) <- modules) module.enrichParameter()
    }
    def forEachParameter(f: TensorR => Unit) = forEachNamedParameter{case (_, (tensorR, _)) => f(tensorR)}
    def forEachPairParameter(f: (TensorR, Option[Tensor]) => Unit) = forEachNamedParameter{case (_, (tr, t)) => f(tr, t)}

    def registerParamters(nameScope: String): Unit = {
      def oops[T](field: Field)(read: Field => T) = {
        val acc = field.isAccessible
        field.setAccessible(true);
        val res = read(field)
        field.setAccessible(acc)
        res
      }
      val allParams = this.getClass.getDeclaredFields
      val subParameters = allParams.filter { _.getType.isAssignableFrom(classOf[TensorR]) }
      val subModules = allParams.filter { t => classOf[Module].isAssignableFrom(t.getType) && oops[Boolean](t) { _.get(this) != this }}

      subParameters.map{ field => oops[Unit](field) {x => parameters.update(s"$nameScope${x.getName()}", (x.get(this).asInstanceOf[TensorR], None))} }
      subModules.map{ field => oops[Unit](field) {x => {val a = x.get(this).asInstanceOf[Module]; modules.update(s"$nameScope${x.getName()}", a); a.registerParamters(s"$nameScope${x.getName()}/")}}}
    }
  }

  case class Linear1D(val inSize: Int, val outSize: Int, val name: String = "linear1d") extends Module {
    val scale: Float = 1.0f / sqrt(inSize).toFloat
    val weight = TensorR(Tensor.rand(Seq(inSize, outSize), scale))
    val bias = TensorR(Tensor.zeros(outSize))
    def apply(in: TensorR): TensorR @diff = in.dot(weight) + bias
  }

  case class Linear1DTrans(val inSize: Int, val outSize: Int, val name: String = "linear1dtrans") extends Module {
    val scale: Float = 1.0f / sqrt(inSize).toFloat
    val weight = TensorR(Tensor.rand(Seq(outSize, inSize), scale))
    val bias = TensorR(Tensor.zeros(outSize))
    def apply(in: TensorR): TensorR @diff = in.dot_trans(weight) + bias
  }

  case class Linear1D2(val inSize1: Int, val inSize2: Int, val outSize: Int, val name: String = "Linear1d2") extends Module {
    val scale1: Float = 1.0f / sqrt(inSize1).toFloat
    val scale2: Float = 1.0f / sqrt(inSize2).toFloat
    val weight1 = TensorR(Tensor.rand(Seq(inSize1, outSize), scale1))
    val weight2 = TensorR(Tensor.rand(Seq(inSize2, outSize), scale2))
    val bias    = TensorR(Tensor.zeros(outSize))
    def apply(in1: TensorR, in2: TensorR): TensorR @diff = in1.dot(weight1) + in2.dot(weight2) + bias
  }

  case class Linear1D2Trans(val inSize1: Int, val inSize2: Int, val outSize: Int, val name: String = "Linear1d2trans") extends Module {
    val scale1: Float = 1.0f / sqrt(inSize1).toFloat
    val scale2: Float = 1.0f / sqrt(inSize2).toFloat
    val weight1 = TensorR(Tensor.rand(Seq(outSize, inSize1), scale1))
    val weight2 = TensorR(Tensor.rand(Seq(outSize, inSize2), scale2))
    val bias    = TensorR(Tensor.zeros(outSize))
    def apply(in1: TensorR, in2: TensorR): TensorR @diff = in1.dot_trans(weight1) + in2.dot_trans(weight2) + bias
  }

  case class Conv2D(val inChannel: Int, val outChannel: Int, val kernelSize: Seq[Int], val stride: Seq[Int] = Seq(1, 1), val useBias: Boolean = true, val pad: Int = 0, val name: String = "conv2d") extends Module {
    assert(kernelSize.size == 2, "kernel_size should be Seq[Int] of size 2")
    assert(stride.size == 2, "stride should be Seq[Int] of size 2")
    val scale: Float = 1.0f / sqrt(inChannel * kernelSize.head * kernelSize.last).toFloat
    val kernel = TensorR(Tensor.rand(Seq(outChannel, inChannel, kernelSize.head, kernelSize.last), scale))
    val bias = if (useBias) Some(TensorR(Tensor.zeros(outChannel))) else None
    def apply(in: TensorR): TensorR @diff = in.convBBP(kernel, bias, stride, Seq(pad, pad, pad, pad))
  }

  abstract class RnnCell extends Module {
    def init(batchSize: Int): ArrayBuffer[TensorR]
    def apply(ins: ArrayBuffer[TensorR]): ArrayBuffer[TensorR] @diff
  }

  case class VanillaRNNCell(val inputSize: Int, val hiddenSize: Int, val outputSize: Int, val name: String = "vanilla_rnn_cell") extends RnnCell {
    val inLinear = Linear1D2(inputSize, hiddenSize, hiddenSize)
    val outLinear = Linear1D(hiddenSize, outputSize)
    def apply(ins: ArrayBuffer[TensorR]): ArrayBuffer[TensorR] @diff = {
      assert(ins.size == 2, "vanilla rnn cell should take a input of two tensors, the next element, and the last hidden layer")
      val in = ins(0)
      val lastHidden = ins(1)
      val hidden = inLinear(in, lastHidden).tanh()
      ArrayBuffer(outLinear(hidden), hidden)
    }
    def init(batchSize: Int) = ArrayBuffer(TensorR(Tensor.zeros(batchSize, hiddenSize)))
  }

  case class VanillaRNNCellTrans(val inputSize: Int, val hiddenSize: Int, val outputSize: Int, val name: String = "vanilla_rnn_cell_trans") extends RnnCell {
    val inLinear = Linear1D2Trans(inputSize, hiddenSize, hiddenSize)
    val outLinear = Linear1DTrans(hiddenSize, outputSize)
    def apply(ins: ArrayBuffer[TensorR]): ArrayBuffer[TensorR] @diff = {
      assert(ins.size == 2, "vanilla rnn cell trans should take a input of two tensors, the next element, and the last hidden layer")
      val in = ins(0)
      val lastHidden = ins(1)
      val hidden = inLinear(in, lastHidden).tanh()
      ArrayBuffer(outLinear(hidden), hidden)
    }
    def init(batchSize: Int) = ArrayBuffer(TensorR(Tensor.zeros(batchSize, hiddenSize)))
  }

  case class LSTMCell(val inputSize: Int, val hiddenSize: Int, val outputSize: Int, val name: String = "lstm_cell") extends RnnCell {
    val scale1: Float = 1.0f / sqrt(inputSize).toFloat
    val scale2: Float = 1.0f / sqrt(hiddenSize).toFloat

    // initialize all parameters
    val fGate = Linear1D2(inputSize, hiddenSize, hiddenSize)
    val iGate = Linear1D2(inputSize, hiddenSize, hiddenSize)
    val cGate = Linear1D2(inputSize, hiddenSize, hiddenSize)
    val oGate = Linear1D2(inputSize, hiddenSize, hiddenSize)
    val outLinear = Linear1D(hiddenSize, outputSize)
    def apply(ins: ArrayBuffer[TensorR]): ArrayBuffer[TensorR] @diff = {
      assert(ins.size == 3, "LSTM cell should take a input of three tensors, the next element, the last hidden layer, and the last cell layer")
      val in = ins(0)
      val lastHidden = ins(1)
      val lastCell = ins(2)
      val f = fGate(in, lastHidden).sigmoid()
      val i = iGate(in, lastHidden).sigmoid()
      val o = oGate(in, lastHidden).sigmoid()
      val C = cGate(in, lastHidden).tanh()
      val c = f * lastCell + i * C
      val h = o * c.tanh()
      ArrayBuffer(outLinear(h), h, c)
    }
    def init(batchSize: Int) = ArrayBuffer(TensorR(Tensor.zeros(batchSize, hiddenSize)), TensorR(Tensor.zeros(batchSize, hiddenSize)))
  }

  case class LSTMCellTrans(val inputSize: Int, val hiddenSize: Int, val outputSize: Int, val name: String = "lstm_cell_trans") extends RnnCell {
    val scale1: Float = 1.0f / sqrt(inputSize).toFloat
    val scale2: Float = 1.0f / sqrt(hiddenSize).toFloat

    // initialize all parameters
    val fGate = Linear1D2Trans(inputSize, hiddenSize, hiddenSize)
    val iGate = Linear1D2Trans(inputSize, hiddenSize, hiddenSize)
    val cGate = Linear1D2Trans(inputSize, hiddenSize, hiddenSize)
    val oGate = Linear1D2Trans(inputSize, hiddenSize, hiddenSize)
    val outLinear = Linear1DTrans(hiddenSize, outputSize)
    def apply(ins: ArrayBuffer[TensorR]): ArrayBuffer[TensorR] @diff = {
      assert(ins.size == 3, "LSTM cell should take a input of three tensors, the next element, the last hidden layer, and the last cell layer")
      val in = ins(0)
      val lastHidden = ins(1)
      val lastCell = ins(2)
      val f = fGate(in, lastHidden).sigmoid()
      val i = iGate(in, lastHidden).sigmoid()
      val o = oGate(in, lastHidden).sigmoid()
      val C = cGate(in, lastHidden).tanh()
      val c = f * lastCell + i * C
      val h = o * c.tanh()
      ArrayBuffer(outLinear(h), h, c)
    }
    def init(batchSize: Int) = ArrayBuffer(TensorR(Tensor.zeros(batchSize, hiddenSize)), TensorR(Tensor.zeros(batchSize, hiddenSize)))
  }

  // case class DynamicRNN(val cell: RnnCell, val name: String = "dynamic_rnn_unroll") extends Module {
  //   @virtualize
  //   override def apply(input: TensorR, target: Rep[Array[Int]], lengths: Option[Rep[Array[Int]]] = None, batchFirst: Boolean = false): ArrayBuffer[TensorR] @diff = {
  //     // TODO (Fei Wang): assuming for now that batchFirst is false and lengths is None
  //     LOOPSM(ArrayBuffer(cell.init(input.x.shape(1))))(input.x.shape(0)){ (i: Rep[Int]) => (x: ArrayBuffer[TensorR]) =>
  //       val ArrayBuffer(new_y, new_hidden) = cell(ArrayBuffer(input(i), x(0)))
  //       val x1 = if (x.size == 1) new_y else x(1).concat(dim = 0, new_y)
  //       ArrayBuffer(new_hidden, x1)
  //     }
  //   }
  // }

  case class DynamicRNNFix(val cell: RnnCell, val name: String = "dynamic_rnn_unroll_fix") extends Module {
    def apply(input: TensorR, target: Rep[Array[Int]], lengths: Option[Rep[Array[Int]]] = None, batchFirst: Boolean = false): ArrayBuffer[TensorR] @diff = {
      // TODO (Fei Wang): assuming for now that batchFirst is false and lengths is None
      val init = TensorR(Tensor.zeros(1)) +=: cell.init(input.x.shape(1))
      LOOPSM(init)(input.x.shape(0)) { (i: Rep[Int]) => (x: ArrayBuffer[TensorR]) =>
        val res = cell( input(i) +=: x.tail )
        val y = NewArray[Int](input.x.shape(1))
        for (j <- DataLoop(input.x.shape(1)))
          y(j) = target(i + j * input.x.shape(0))
        val loss = x(0) + res(0).logSoftmaxB().nllLossB(y).sum()
        loss +=: res.tail
      }
    }
  }

  abstract class Optim {
    val module: Module
    module.registerParamters(s"${module.name}/")
    def step_func: (TensorR, Option[Tensor]) => Unit
    def zero_grad() = module.forEachParameter(_.clear_grad())
    def step() = module.forEachPairParameter(step_func)
    def show() = module.forEachNamedParameter{case (name, (tr, ot)) => tr.d.printHead(5, name)}
  }

  case class SGD(val module: Module, val learning_rate: Float, val gradClip: Float = 1.0f, val descent: Boolean = true) extends Optim {
    @virtualize
    def step_func = { case (tr, _) =>
      tr.d.changeTo { i =>
        val temp = var_new(tr.d.data(i))
        if (temp > gradClip) temp = gradClip
        if (temp < -gradClip) temp = -gradClip
        if (descent)
          tr.x.data(i) -= learning_rate * temp
        else
          tr.x.data(i) += learning_rate * temp
        0.0f
      }
      // tr.clip_grad(gradClip)
      // if (descent)
      //   tr.x -= tr.d * learning_rate
      // else
      //   tr.x += tr.d * learning_rate
      // tr.clear_grad()
    }
  }

  case class Adagrad(val module: Module, val learning_rate: Float, val gradClip: Float = 1.0f, val descent: Boolean = true) extends Optim {
    module.enrichParameter()
    @virtualize
    def step_func = { case (tr, Some(t)) =>
      tr.d.changeTo { i =>
        val temp = var_new(tr.d.data(i))
        if (temp > gradClip) temp = gradClip
        if (temp < -gradClip) temp = -gradClip
        t.data(i) += temp * temp
        if (descent)
          tr.x.data(i) -= learning_rate * temp / Math.sqrt(t.data(i) + 1e-8f).toFloat
        else
          tr.x.data(i) += learning_rate * temp / Math.sqrt(t.data(i) + 1e-8f).toFloat
        0.0f
      }
      // tr.clip_grad(gradClip)
      // t += tr.d * tr.d
      // if (descent)
      //   tr.x -= tr.d * learning_rate / (t + 1e-8f).sqrt()
      // else
      //   tr.x += tr.d * learning_rate / (t + 1e-8f).sqrt()
      // tr.clear_grad()
    }
  }
}

// FIXME: Eliminate explicit `backend` definition if possible.
// `TensorDsl(Cublas|Cudnn)` already explicitly define `backend`.
// Without the definitions here, however, `LanternDriver(Cublas|Cudnn).backend` is null.
trait NNModuleCublas extends NNModule with TensorDslCublas {
  backend = BackendCublas()
}
trait NNModuleCudnn extends NNModule with TensorDslCudnn {
  backend = BackendCudnn()
}

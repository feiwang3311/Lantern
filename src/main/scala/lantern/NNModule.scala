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
      val subParameters = allParams.filter { t => classOf[Option[TensorR]].isAssignableFrom(t.getType) || classOf[TensorR].isAssignableFrom(t.getType) }
      val subModules = allParams.filter { t => classOf[Module].isAssignableFrom(t.getType) && oops[Boolean](t) { _.get(this) != this }}

      subParameters.map{ field => oops[Unit](field) { x =>
        val field = x.get(this)
        val name = x.getName()
        val fullName = s"$nameScope${name}"
        if (field.isInstanceOf[TensorR]) parameters.update(fullName, (field.asInstanceOf[TensorR], None))
        else field match {
          case Some(field) => parameters.update(fullName, (field.asInstanceOf[TensorR], None))
          case None => ()
        }
      }}
      subModules.map{ field => oops[Unit](field) { x =>
        val a = x.get(this).asInstanceOf[Module]
        modules.update(s"$nameScope${x.getName()}", a)
        a.registerParamters(s"$nameScope${x.getName()}/")
      }}
    }
  }

  case class Linear1D(val inSize: Int, val outSize: Int, val name: String = "linear1d") extends Module {
    val scale: Float = 1.0f / sqrt(inSize).toFloat
    val weight = TensorR(Tensor.rand(Seq(inSize, outSize), scale))
    val bias = TensorR(Tensor.zeros(outSize))
    def apply(in: TensorR): TensorR @diff = in.dot(weight) plusBias bias
  }

  case class Linear1D2(val inSize1: Int, val inSize2: Int, val outSize: Int, val name: String = "Linear1d2") extends Module {
    val scale1: Float = 1.0f / sqrt(inSize1).toFloat
    val scale2: Float = 1.0f / sqrt(inSize2).toFloat
    val weight1 = TensorR(Tensor.rand(Seq(inSize1, outSize), scale1))
    val weight2 = TensorR(Tensor.rand(Seq(inSize2, outSize), scale2))
    val bias    = TensorR(Tensor.zeros(outSize))
    def apply(in1: TensorR, in2: TensorR): TensorR @diff = in1.dot(weight1) + in2.dot(weight2) + bias
  }

  // fan_in = kernel.shape(1) * kernel.shape(2) * kernel.shape(3)
  // fan_out = kernel.shape(0) * kernel.shape(2) * kernel.shape(3)
  // kaiming_uniform [-bound, bound] where bound = sqrt(6/fan_in)
  // xaiver_uniform [-bound, bound] where bound = sqrt(6/(fan_in + fan_out))

  case class Conv2D(val inChannel: Int, val outChannel: Int, val kernelSize: Seq[Int], val stride: Seq[Int] = Seq(1, 1), val useBias: Boolean = true, val pad: Int = 0, val name: String = "conv2d") extends Module {
    assert(kernelSize.size == 2, "kernel_size should be Seq[Int] of size 2")
    assert(stride.size == 2, "stride should be Seq[Int] of size 2")
    // xaiver_uniform initialization
    // val fan_in = inChannel * kernelSize.head * kernelSize.last
    // val fan_out = outChannel * kernelSize.head * kernelSize.last
    // val scale: Float = 2.0f * sqrt(6.0f / (fan_in + fan_out)).toFloat
    // kaiming_uniform initialization
    val scale: Float = 2.0f * sqrt(6.0f / (inChannel * kernelSize.head * kernelSize.last)).toFloat
    val kernel = TensorR(Tensor.rand(Seq(outChannel, inChannel, kernelSize.head, kernelSize.last), scale))
    val bias = if (useBias) Some(TensorR(Tensor.zeros(outChannel))) else None
    def apply(in: TensorR): TensorR @diff = in.convBBP(kernel, bias, stride, Seq(pad, pad, pad, pad))
  }

  case class Conv2Dn(val inChannel: Int, val outChannel: Int, val kernelSize: Seq[Int], val stride: Seq[Int] = Seq(1, 1), val useBias: Boolean = true, val pad: Int = 0, val name: String = "conv2d") extends Module {
    assert(kernelSize.size == 2, "kernel_size should be Seq[Int] of size 2")
    assert(stride.size == 2, "stride should be Seq[Int] of size 2")
    // normal initialization with mean 0.0 and std 0.01
    val kernel = TensorR(Tensor.randnorm(outChannel, inChannel, kernelSize.head, kernelSize.last))
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
    def perform(f: (String, (TensorR, Option[Tensor])) => Unit) = module.forEachNamedParameter(f)
  }

  case class SGD(val module: Module, val learning_rate: Float, val gradClip: Float = 1.0f, val descent: Boolean = true) extends Optim {
    @virtualize
    def step_func = { case (tr, _) =>
      // tr.d.changeTo { i =>
      //   val temp = var_new(tr.d.data(i))
      //   if (temp > gradClip) temp = gradClip
      //   if (temp < -gradClip) temp = -gradClip
      //   if (descent)
      //     tr.x.data(i) -= learning_rate * temp
      //   else
      //     tr.x.data(i) += learning_rate * temp
      //   0.0f
      // }
      // tr.clip_grad(gradClip)
      if (descent)
        backend.geam(tr.x, false, 1.0f, tr.d, false, -1.0f * learning_rate, tr.x)
        // tr.x -= tr.d * learning_rate
      else
        backend.geam(tr.x, false, 1.0f, tr.d, false, learning_rate, tr.x)
        // tr.x += tr.d * learning_rate
      tr.clear_grad()
    }
  }

  case class Adagrad(val module: Module, val learning_rate: Float, val gradClip: Float = 1.0f, val descent: Boolean = true) extends Optim {
    module.enrichParameter()
    @virtualize
    def step_func = { case (tr, Some(t)) =>
      backend.adagrad_update(tr, t, learning_rate, gradClip, descent)
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

  trait RnnBase extends Module {
    val inputSize: Int
    val hiddenSize: Int
    val numLayers: Int = 1
    val dropout: Float = 0f
    val bidirectional: Boolean = false

    // def apply(input: TensorR, hidden: Option[TensorR] = None): TensorR @diff
    def apply(input: TensorR, hidden: Option[TensorR] = None): TensorR
  }

  case class Rnn(inputSize: Int, hiddenSize: Int,
                 override val numLayers: Int = 1,
                 override val dropout: Float = 0f,
                 override val bidirectional: Boolean = false,
                 val name: String = "rnn") extends RnnBase {

    val w_ih = ArrayBuffer[Tensor]()
    val w_hh = ArrayBuffer[Tensor]()
    val b_ih = ArrayBuffer[Tensor]()
    val b_hh = ArrayBuffer[Tensor]()

    val numDirections = if (bidirectional) 2 else 1
    val gateSize = hiddenSize

    // NOTE: Choose different initialization strategy?
    // parameterBuffer = Tensor(backend.mallocArray[Float](getParameterSize()), getParameterSize())
    val parameterBuffer = Tensor.fill(Seq(getParameterSize()), 0.1f)

    def getParameterSize(): Int = {
      val w_ih_size = gateSize * inputSize + (numLayers - 1) * gateSize * hiddenSize * numDirections
      val w_hh_size = numLayers * gateSize * hiddenSize
      val b_ih_size = numLayers * gateSize
      val b_hh_size = numLayers * gateSize
      w_ih_size + w_hh_size + b_ih_size + b_hh_size
    }

    def setupParameters(): Unit = {
      val address = parameterBuffer.data
      var offset = var_new(0)

      def getParameter(dims: Int*): Tensor = {
        val x = Tensor(slice(parameterBuffer.data, offset), dims: _*)
        offset += dims.product
        x
      }

      for (layer <- (0 until numLayers): Range) {
        val layerInputSize = if (layer == 0) inputSize else hiddenSize * numDirections
        w_ih += getParameter(gateSize, layerInputSize)
        w_hh += getParameter(gateSize, hiddenSize)
        b_ih += getParameter(gateSize)
        b_hh += getParameter(gateSize)
      }
    }

    setupParameters()

    // def apply(input: TensorR, hidden: Option[TensorR] = None): TensorR @diff = {
    def apply(input: TensorR, hidden: Option[TensorR] = None): TensorR = {
      assert(input.x.rank == 3, "RNN input should have rank 3: [seqLength x batchSize x inputSize]")
      val seqLength = input.x.shape(0)
      val batchSize = input.x.shape(1)
      val inputSize = input.x.shape(2)

      val hx = hidden.getOrElse(TensorR(Tensor.zeros(numLayers * numDirections, batchSize, hiddenSize)))
      assert(hx.x.rank == 3, "RNN hidden state should have rank 3: [numLayers x batchSize x hiddenSize]")
      assert(batchSize == hx.x.shape(1), "RNN hidden state second dimension should equal input second dimension (batch size)")

      val (y, _, _) = BackendCudnn().cudnnRNNForwardTraining(
        input.x, hx.x, parameterBuffer, numLayers, hiddenSize, dropout, bidirectional = bidirectional)
      TensorR(y)

      // TODO: Implement backward pass.
    }
  }
}

package lantern

import scala.util.continuations._
import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._
import scala.virtualization.lms.common._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.{Map => MutableMap}
import scala.math._

trait TensorSecOrderApi extends TensorDsl with Diff {

  object TensorF {
    def zerosLike(that: TensorF) = new TensorF(Tensor.zeros_like(that.x), Tensor.zeros_like(that.d))
  }

  class TensorF(val x: Tensor, val d: Tensor) extends Serializable {
    var isInput: Boolean = false // true if it is an input (no need to compute gradient)

    def apply(i: Rep[Int]) = new TensorF(x(i), d(i))
    def apply(i: Int, j: Int) = new TensorF(x(i, j), d(i, j))

    def clip_grad(bound: Float) = {
      d.clipAt(bound)
    }

    // TODO: optimization to fuse loops are needed here!
    def + (that: TensorF): TensorF = new TensorF(x + that.x, d + that.d)
    def + (that: Rep[Float]): TensorF = new TensorF(x + that, d)
    def - (that: TensorF): TensorF = new TensorF(x - that.x, d - that.d)
    def - (that: Rep[Float]): TensorF = new TensorF(x - that, d)
    def * (that: TensorF): TensorF = new TensorF(x * that.x, d * that.x + x * that.d)
    def * (that: Rep[Float]): TensorF = new TensorF(x * that, d * that)
    def / (that: TensorF): TensorF = new TensorF(x / that.x, d / that.x - x * that.d / that.x / that.x)
    def / (that: Rep[Float]): TensorF = new TensorF(x / that, d / that)
    def dot(that: TensorF): TensorF = new TensorF(x dot that.x, x.dot(that.d) + d.dot(that.x))
    def sum(): TensorF = new TensorF(x.sum(), d.sum())
    def tanh(): TensorF = {
      val value = x.tanh()
      new TensorF(value, d - value * value * d)
    }
    def exp(): TensorF = {
      val value = x.exp()
      new TensorF(value, d * value)
    }
    def log(): TensorF = new TensorF(x.log(), d / x)
    def sqrt(): TensorF = {
      val value = x.sqrt()
      new TensorF(value, d / value / 2)
    }
    def square(): TensorF = new TensorF(x * x, x * d * 2)
    def relu(inPlace: Boolean = false): TensorF = if (inPlace) {
      val temp = new TensorR(x, d)
      backend.relu_grad(temp, temp, inPlace)
      new TensorF(x.relu(inPlace), d)
    } else {
      val temp = TensorR(x)
      backend.relu_grad(temp, new TensorR(x, d), inPlace)
      new TensorF(x.relu(inPlace), temp.d)
    }
    def hardTanh(min_val: Float = -1.0f, max_val: Float = 1.0f, inPlace: Boolean = false) = if (inPlace) {
      val temp = new TensorR(x, d)
      backend.hardTanh_grad(temp, temp, min_val, max_val, inPlace)
      new TensorF(x.hardTanh(min_val, max_val, inPlace), d)
    } else {
      val temp = TensorR(x)
      backend.hardTanh_grad(temp, new TensorR(x, d), min_val, max_val, inPlace)
      new TensorF(x.hardTanh(min_val, max_val, inPlace), temp.d)
    }
    def conv2D_batch(kernel: TensorF, bias: Option[TensorF], strides: Seq[Int], pads: Seq[Int]): (TensorF, Option[TensorF], Int) = {
      val (value, opValue, counterId) = x.conv2D_batch(kernel.x, bias.map(_.x), strides, pads)
      val (tangent1, opTangent1, _) = x.conv2D_batch(kernel.d, bias.map(_.d), strides, pads)
      val (tangent2, opTangent2, _) = d.conv2D_batch(kernel.x, None, strides, pads)
      (new TensorF(value, tangent1 + tangent2), opValue.map(new TensorF(_, opTangent2.get)), counterId)
    }

    // inplace mutations
    def += (that: TensorF) = {x += that.x; d += that.d}
    def += (that: Rep[Float]) = x += that
    def -= (that: TensorF) = {x -= that.x; d -= that.d}
    def -= (that: Rep[Float]) = x -= that
    def *= (that: TensorF) = {x *= that.x; d *= that.x; d += x * that.d}
    def *= (that: Rep[Float]) = {x *= that; d *= that}
    def add_cartesian(y: TensorF, output: TensorF) = {
      x.add_cartesian(y.x, output.x)
      d.add_cartesian(y.x, output.d); d.add_cartesian(y.d, output.x)
    }
    def add_composition(y: TensorF, output: TensorF) = {
      x.add_composition(y.x, output.x)
      d.add_composition(y.x, output.d); d.add_composition(y.d, output.x)
    }
    // this += y^T dot output
    def add_dotTrans1(y: TensorF, output: TensorF) = {
      x.add_dotTrans1(y.x, output.x)
      d.add_dotTrans1(y.x, output.d); d.add_dotTrans1(y.d, output.x)
    }
    // this += y dot output^T
    def add_dotTrans2(y: TensorF, output: TensorF) = {
      x.add_dotTrans2(y.x, output.x)
      d.add_dotTrans2(y.x, output.d); d.add_dotTrans2(y.d, output.x)
    }
  }

  object TensorFR {
    def apply(x: TensorF) = new TensorFR(x, TensorF.zerosLike(x))
  }

  class TensorFR(val x: TensorF, val d: TensorF) extends Serializable {
    var isInput: Boolean = false

    def apply(i: Rep[Int]) = new TensorFR(x(i), d(i))
    def apply(i: Int, j: Int) = new TensorFR(x(i, j), d(i, j))

    def clip_grad(bound: Float) = {
      d.clip_grad(bound)
    }

    def + (that: TensorFR): TensorFR @diff = shift { (k: TensorFR => Unit) =>
      val y = TensorFR(x + that.x); k(y)
      d += y.d; that.d += y.d
    }
    def + (that: Rep[Float]): TensorFR @diff = shift { (k: TensorFR => Unit) =>
      val y = TensorFR(x + that); k(y)
      d += y.d
    }
    def - (that: TensorFR): TensorFR @diff = shift { (k: TensorFR => Unit) =>
      val y = TensorFR(x - that.x); k(y)
      d += y.d; that.d -= y.d
    }
    def - (that: Rep[Float]): TensorFR @diff = shift { (k: TensorFR => Unit) =>
      val y = TensorFR(x - that); k(y)
      d += y.d
    }
    def * (that: TensorFR): TensorFR @diff = shift { (k: TensorFR => Unit) =>
      val y = TensorFR(x * that.x); k(y)
      d += y.d * that.x; that.d += y.d * x
    }
    def * (that: Rep[Float]): TensorFR @diff = shift { (k: TensorFR => Unit) =>
      val y = TensorFR(x * that); k(y)
      d += y.d * that
    }
    def / (that: TensorFR): TensorFR @diff = shift { (k: TensorFR => Unit) =>
      val y = TensorFR(x / that.x); k(y)
      d += y.d / that.x; that.d -= y.d * x / that.x / that.x
    }
    def / (that: Rep[Float]): TensorFR @diff = shift { (k: TensorFR => Unit) =>
      val y = TensorFR(x / that); k(y)
      d += y.d / that
    }
    def dot (that: TensorFR): TensorFR @diff = shift { (k: TensorFR => Unit) =>
      val y = TensorFR(x dot that.x); k(y)
      (x.x.rank, that.x.x.rank) match {
        case (1, 1) => d += that.x * y.d; that.d += x * y.d
        case (2, 1) => d.add_cartesian(that.x, y.d); that.d.add_composition(x, y.d)
        case (2, 2) => d.add_dotTrans2(y.d, that.x); that.d.add_dotTrans1(x, y.d)
      }
    }
    def sum(): TensorFR @diff = shift { (k: TensorFR => Unit) =>
      val y = TensorFR(x.sum()); k(y)
      d += y.d
    }
    def tanh(): TensorFR @diff = shift { (k: TensorFR => Unit) =>
      val y = TensorFR(x.tanh()); k(y)
      d += y.d; d -= y.x * y.x * y.d
    }
    def exp(): TensorFR @diff = shift { (k: TensorFR => Unit) =>
      val y = TensorFR(x.exp()); k(y)
      d += y.d * y.x
    }
    def log(): TensorFR @diff = shift { (k: TensorFR => Unit) =>
      val y = TensorFR(x.log()); k(y)
      d += y.d / x
    }
    def sqrt(): TensorFR @diff = shift { (k: TensorFR => Unit) =>
      val y = TensorFR(x.sqrt()); k(y)
      d += y.d / y.x / 2
    }
    def square(): TensorFR @diff = shift { (k: TensorFR => Unit) =>
      val y = TensorFR(x.square()); k(y)
      d += y.d * x * 2
    }
    def relu(inPlace: Boolean = false): TensorFR @diff = shift { (k: TensorFR => Unit) =>
      if (inPlace) {
        x.relu(inPlace); k(this)
        backend.relu_grad(new TensorR(x.x, d.x), new TensorR(x.x, d.x), inPlace)
        backend.relu_grad(new TensorR(x.x, d.d), new TensorR(x.x, d.d), inPlace)
      } else {
        val y = TensorFR(x.relu(inPlace)); k(y)
        backend.relu_grad(new TensorR(x.x, d.x), new TensorR(y.x.x, y.d.x), inPlace)
        backend.relu_grad(new TensorR(x.x, d.d), new TensorR(y.x.x, y.d.d), inPlace)
      }
    }
    def hardTanh(min_val: Float = -1.0f, max_val: Float = 1.0f, inPlace: Boolean = false): TensorFR @diff = shift { (k: TensorFR => Unit) =>
      if (inPlace) {
        x.hardTanh(min_val, max_val, inPlace); k(this)
        backend.hardTanh_grad(new TensorR(x.x, d.x), new TensorR(x.x, d.x), min_val, max_val, inPlace)
        backend.hardTanh_grad(new TensorR(x.x, d.d), new TensorR(x.x, d.d), min_val, max_val, inPlace)
      } else {
        val y = TensorFR(x.hardTanh(min_val, max_val, inPlace)); k(y)
        backend.hardTanh_grad(new TensorR(x.x, d.x), new TensorR(y.x.x, y.d.x), min_val, max_val, inPlace)
        backend.hardTanh_grad(new TensorR(x.x, d.d), new TensorR(y.x.x, y.d.d), min_val, max_val, inPlace)
      }
    }

    def conv2D_batch(kernel: TensorFR, bias: Option[TensorFR] = None, strides: Seq[Int] = Seq(1,1), pads: Seq[Int] = Seq(0,0)): TensorFR @diff = shift { (k: TensorFR => Unit) =>
      val (out, opInput, counterId) =  x.conv2D_batch(kernel.x, bias.map(_.x), strides, pads)
      val y = TensorFR(out); val opInputFR = opInput.map(TensorFR(_)); k(y)

      generateRawComment("conv2D back-propagate sec order")
      val paddings = pads.size match {
        case 2 => (pads(0), pads(1))
        case 4 => (pads(0), pads(2))
        case 1 => (pads(0), pads(0))
      }
      val stridess = strides.size match {
        case 2 => (strides(0), strides(1))
      }
      backend.conv2D_batch_grad(
        new TensorR(this.x.x, this.d.x),
        opInputFR map (v => new TensorR(v.x.x, v.d.x)),
        new TensorR(kernel.x.x, kernel.d.x),
        new TensorR(y.x.x, y.d.x),
        bias map (v => new TensorR(v.x.x, v.d.x)),
        paddings, stridess, (1, 1), counterId)
      backend.conv2D_batch_grad(
        new TensorR(this.x.d, this.d.d), //
        opInputFR map (v => new TensorR(v.x.d, v.d.d)),
        new TensorR(kernel.x.d, kernel.d.d), //
        new TensorR(y.x.x, y.d.x), //
        None,
        paddings, stridess, (1, 1), counterId + 1)
      backend.conv2D_batch_grad(
        new TensorR(this.x.x, this.d.d), //
        opInputFR map (v => new TensorR(v.x.x, v.d.d)),
        new TensorR(kernel.x.x, kernel.d.d), //
        new TensorR(y.x.d, y.d.d), //
        bias map (v => new TensorR(v.d.x, v.d.d)),
        paddings, stridess, (1, 1), counterId + 2)
    }
  }

  // reset for gradients and hessian_vector
  def gradHessV(f: TensorFR => TensorFR @diff)(start: TensorF) = {
    val x = TensorFR(start)
    reset {
      val temp = f(x)
      temp.d.x.setAsOne()
      ()
    }
    val gradient: Tensor = x.d.x
    val hessian_vector: Tensor = x.d.d
    (gradient, hessian_vector)
  }
  def gradHessV(f: () => TensorFR @diff) = {
    val result = Tensor.scalar(0)
    reset {
      val temp = f()
      // Assume that result is scalar
      result.copy_data(temp.x.x)
      temp.d.x.setAsOne()
    }
    result
  }
  def getGradient(x: TensorFR): Tensor = x.d.x
  def getHessV(x: TensorFR): Tensor = x.d.d

}

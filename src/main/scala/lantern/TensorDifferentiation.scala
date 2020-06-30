package lantern

import scala.util.continuations._

import scala.collection.mutable.ArrayBuffer
import scala.math._

import lms.core.stub._
import lms.macros.SourceContext
import lms.core.virtualize

trait TensorDsl extends DslCPP with Diff {

  /**
    Memory Management:
      finally we used a temporary solution called "memory arena". The base code will claim a large piece of code for the whole program.
      internally, every malloc will borrow memory from this arena.
      By using getAllocMem and setAllocMem, we can selectively return a big trunk of memory after one iteration of training.
   **/

  var debug: Boolean = true
  implicit def Seq2SeqRep(x: Seq[Int]) = x map (unit(_))
  implicit def SeqRB2SeqRBOps[T](s: Seq[T]): SeqRBOps[T] = SeqRBOps(s)
  @virtualize
  case class SeqRBOps[T](s: Seq[T]) {
    def forallR(f: T => Rep[Boolean]): Rep[Boolean] = s.foldLeft(None: Option[Rep[Boolean]]) {
      case (None, r) => Some(f(r))
      case (Some(l), r) => Some(l && f(r))
    }.getOrElse(true)
    def count(f: T => Rep[Boolean]): Rep[Int] = s.foldLeft(unit(0)){case (l, r) => if (f(r)) (l+1) else l}
  }

  def convSize(size: Rep[Int], kernelSize: Rep[Int], strideSize: Int, pad: Int = 0) = (size + 2 * pad - kernelSize)/strideSize + 1
  @virtualize
  def resizeDim(scalarCount: Rep[Int], toDims: Seq[Rep[Int]]): Seq[Rep[Int]] = {
    val count = var_new(0)
    val prod = var_new(1)
    def check(dimseq: Seq[Rep[Int]]): Unit =
      if (dimseq.size > 0) {
        if (dimseq.head < 0) { count += 1 }
        else prod *= dimseq.head
        check(dimseq.tail)
      }
    check(toDims)
    assertC(count < 2, "cannot have 2 or more -1s in resize!!")
    if (count == 0) assertC(prod == scalarCount, "must same size!!")
    toDims.map(x => if (x > 0) x else scalarCount / prod)
  }

  @virtualize
  def mmax(a: Rep[Int], b: Rep[Int]) = if (a >= b) a else b
  def mmax(a: Int, b: Int) = if (a >= b) a else b

  // Hacky fix for the lack of effect system working with slices and aliases in LMS_clean
  // def slice[T: Manifest](arr: Rep[Array[T]], off: Rep[Int]) = uncheckedPure[Array[T]](arr, "+", off)
  def slice[T: Manifest](arr: Rep[Array[T]], off: Rep[Int]) = sliceReadWrite(arr, off)
  // sliceRead: slice an array only for read
  def sliceRead[T: Manifest](arr: Rep[Array[T]], off: Rep[Int]) = uncheckedEffect[Array[T]](arr, "+", off)(arr)()
  // sliceWrite: slice an array only for write
  def sliceWrite[T: Manifest](arr: Rep[Array[T]], off: Rep[Int]) = uncheckedEffect[Array[T]](arr, "+", off)()(arr)
  // sliceReadWrite: slice an array for both read and write
  def sliceReadWrite[T: Manifest](arr: Rep[Array[T]], off: Rep[Int]) = uncheckedEffect[Array[T]](arr, "+", off)(arr)(arr)

  @virtualize
  def assertC(cond: => Rep[Boolean], msg: => String, args: Rep[Any]*): Unit = if (debug) {
    if (!cond) { printf(msg + "\\n", args : _*); exit(1) }
  }

  object Random {
    def rand() = unchecked[Float]("(float)rand()/RAND_MAX")
    def srand(seed: Option[Int] = None) = unchecked[Unit]("srand(",seed.map(_.toString).getOrElse("time(NULL)"),")")
  }

  case class Dimensions(val dims: Seq[Rep[Int]], broadcasted: Rep[Boolean] = unit(false)) {
    def apply(idx: Int) = {
      if (!dims.indices.contains(idx)) throw new IndexOutOfBoundsException(s"Index $idx out of bounds")
      else dims(idx)
    }
    // `head` and `last` have default value 1 for scalar tensors.
    def head: Rep[Int] = dims.headOption.getOrElse(1)
    def last: Rep[Int] = dims.lastOption.getOrElse(1)
    def get(idx: Int): Rep[Int] = if (!dims.indices.contains(idx)) 1 else dims(idx)
    def reverse: Seq[Rep[Int]] = dims.reverse

    // get scalarCount and strides
    val (scalarCount +: strides): Seq[Rep[Int]] = (dims :\ Seq[Rep[Int]](1)) {
      case (dim, seq@(t +: q)) => (dim * t) +: seq
    }

    // override def toString = dims mkString " x "
    override def toString = dims.foldLeft(""){case (l, r) => l + " x " + r.toString}

    lazy val product1: Rep[Int] = dims.foldLeft(unit(1)){case (l, r) => l * r}
    lazy val sum1: Rep[Int] = dims.foldLeft(unit(0)){case (l, r) => l + r}
    @virtualize
    def filterProduct(f: (Rep[Int] => Rep[Boolean])): Rep[Int] = dims.foldLeft(unit(1)){case (l, r) => if (f(r)) (l * r) else l}
    @virtualize
    def filterSum(f: (Rep[Int] => Rep[Boolean])): Rep[Int] = dims.foldLeft(unit(0)){case (l, r) => if (f(r)) (l + r) else l}
  }

  implicit def Dimensions2Seq(x: Dimensions) = x.dims
  implicit def Seq2Dimensions(x: Seq[Rep[Int]]) = Dimensions(x)

  abstract class DataLoop {
    def foreach(f: Rep[Int] => Unit): Unit
  }

  @virtualize
  object DataLoop {
    def apply(size: Int) = if (size <= 3) {
      new DataLoop {
        def foreach(f: Rep[Int] => Unit) = {
          for (i <- 0 until size: Range) f(unit(i))
        }
      }
    } else {
      new DataLoop {
        def foreach(f: Rep[Int] => Unit) = {
          for (i <- 0 until size: Rep[Range]) f(i)
        }
      }
    }

    def apply(size: Rep[Int]) = {
      new DataLoop {
        def foreach(f: Rep[Int] => Unit) = {
          for (i <- 0 until size) f(i)
        }
      }
    }
  }

  /**
    * A code generation backend for tensor operations.
    *
    * Note: Eventually, a tensor operation IR may be introduced to enable analyses and
    * transformations such as operator fusion and matrix chain multiplication optimization.
    */
  trait Backend extends Serializable {
    // Setup the backend. This is the first function that will be called in DSL programs.
    def setup(): Unit

    // Cleanup the backend. This is the last function that will be called in DSL programs.
    def cleanup(): Unit

    // Allocate an array with the specified length.
    def mallocArray[T: Manifest](length: Rep[Int]): Rep[Array[T]]

    // Copy data from one array to another.
    // NOTE: This function is intentionally not defined generically to simplify the codegen implementation.
    // The only user of this function is currently `copyTensorData`.
    def copyFloatArray(dest: Rep[Array[Float]], src: Rep[Array[Float]], length: Rep[Int]): Unit

    // Copy data from one tensor to another.
    @virtualize
    def copyTensorData(dest: Tensor, src: Tensor): Unit = {
      assertC(dest.scalarCount == src.scalarCount,
        "Tensors do not have same scalar count: %d, %d", dest.scalarCount, src.scalarCount)
      copyFloatArray(dest.data, src.data, dest.scalarCount)
    }

    // wrap array to tensor
    def arrayToTensor(array: Rep[Array[Float]], dims: Rep[Int]*): Tensor

    // Initialize a tensor with the specified dimensions and scalar values.
    def makeTensor(dims: Seq[Rep[Int]], scalars: Float*): Tensor

    // Initialize a tensor with the specified dimensions and repeated value.
    def fill(dims: Seq[Rep[Int]], value: Rep[Float]): Tensor

    // Initialize a tensor with the specified bias tensor at the specified dimension.
    def fillWithBias(dims: Seq[Rep[Int]], bias: Tensor, dim: Int): Tensor

    // Fill a tensor in-place with the specified value.
    def fillInPlace(x: Tensor, value: Rep[Float])

    // Initialize a tensor with scalars sampled from a zero-centered uniform distribution.
    // By default, the uniform distribution is over [-0.5, 0.5].
    def randinit(dims: Seq[Int], scale: Float = 1.0f, seed: Option[Int] = None): Tensor

    def clipAt(x: Tensor, bound: Float): Unit
    def mutate(x: Tensor, delta: Rep[Int] => Rep[Float]): Unit
    def mapInPlace(x: Tensor, op: Rep[Float] => Rep[Float]): Unit
    def changeTo(x: Tensor, gen: Rep[Int] => Rep[Float]): Unit
    def map(x: Tensor, op: Rep[Float] => Rep[Float]): Tensor
    def fold(init: Rep[Float])(x: Tensor, op: (Rep[Float], Rep[Float]) => Rep[Float]): Rep[Float]

    // Compute vector-vector dot product, i.e. inner product.
    // [V] dot [V] => [1] (scalar)
    def vectorVectorDot(x: Tensor, y: Tensor): Tensor

    // Compute matrix-vector dot product.
    // [M1 x M2] dot [M2] => [M1]
    def matrixVectorDot(x: Tensor, y: Tensor): Tensor

    // Compute matrix-matrix dot product.
    // [M1 x M2] dot [M2 x M3] => [M1 x M3]
    def matrixMatrixDot(x: Tensor, y: Tensor): Tensor

    def dot(x: Tensor, y: Tensor): Tensor =
      (x.rank, y.rank) match {
        case (1, 1) => vectorVectorDot(x, y)
        case (2, 1) => matrixVectorDot(x, y)
        case (2, 2) => matrixMatrixDot(x, y)
        case _ => throw new IllegalArgumentException(s"Incompatible shapes: ${x.shape}, ${y.shape}")
      }

    def dot_grad(x: TensorR, y: TensorR, output: TensorR): Unit

    // `x` is matrix, `y` is dims(1)-sized vector, `output` is dims(0)-sized vector
    // this function updates `this` so that x += output * y, where * is Cartesian product
    def add_cartesian(x: Tensor, y: Tensor, output: Tensor): Unit

    // `x` is vector of size `a`, `output` is vector of size `b`, `y` is 2-D matrix of size (b x a)
    // this function updates `x`, so that x += y^T dot output.
    def add_composition(x: Tensor, y: Tensor, output: Tensor): Unit

    // x += y^T dot output
    def add_dotTrans1(x: Tensor, y: Tensor, output: Tensor): Unit
    // x += y dot output^T
    def add_dotTrans2(x: Tensor, y: Tensor, output: Tensor): Unit

    // Elementwise addition.
    def +(x: Tensor, y: Rep[Float]): Tensor
    // Also return dimensions to track whether broadcasting happened for the two operands
    def +(x: Tensor, y: Tensor): (Tensor, Dimensions, Dimensions)
    // back prop for + (may fuse the gradient of the two operands for efficiency)
    def add_grad(x: TensorR, y: TensorR, output: TensorR, xShape: Dimensions, yShape: Dimensions): Unit

    // In-place elementwise addition.
    def +=(x: Tensor, y: Rep[Float]): Unit
    // Flexible broadcasting and/or reducing of y to fit into the shape of x.
    def +=(x: Tensor, y: Tensor): Unit

    // Elementwise subtraction.
    def -(x: Tensor, y: Rep[Float]): Tensor
    def -(x: Tensor, y: Tensor): (Tensor, Dimensions, Dimensions)
    def minus_grad(x: TensorR, y: TensorR, output: TensorR, xShape: Dimensions, yShape: Dimensions): Unit

    // In-place elementwise subtraction.
    def -=(x: Tensor, y: Rep[Float]): Unit
    def -=(x: Tensor, y: Tensor): Unit

    // Elementwise multiplication.
    def *(x: Tensor, y: Rep[Float]): Tensor
    def *(x: Tensor, y: Tensor): (Tensor, Dimensions, Dimensions)
    def mul_grad(x: TensorR, y: TensorR, output: TensorR, xShape: Dimensions, yShape: Dimensions): Unit

    // In-place elementwise multiplication.
    def *=(x: Tensor, y: Rep[Float]): Unit
    def *=(x: Tensor, y: Tensor): Unit

    // Elementwise division.
    def /(x: Tensor, y: Rep[Float]): Tensor
    def /(x: Tensor, y: Tensor): (Tensor, Dimensions, Dimensions)
    def div_grad(x: TensorR, y: TensorR, output: TensorR, xShape: Dimensions, yShape: Dimensions): Unit

    // In-place elementwise division.
    def /=(x: Tensor, y: Rep[Float]): Unit
    def /=(x: Tensor, y: Tensor): Unit

    // Ans: plusBias is less general than elementwise + with broadcasting, since it is assume that
    // the bias may be broadcasted, while the other tensor (call it main tensor) doesn't need to.
    // That resulted in easier implementation in cuDNN API calls.
    // It also carries the assumption that the main tensor is not used by other ops until it was added to the bias,
    // so an optimization can be done, such that plusBias is in-place (directly changing the main tensor).
    def plusBias(main: Tensor, bias: Tensor): Tensor
    def plusBias_grad(main: TensorR, bias: TensorR): Unit

    // plusEqual assumes that adder is of the same shape as base, and addition can be done inPlace
    def plusEqual(base: Tensor, adder: Tensor): Tensor
    def plusEqual_grad(base: TensorR, adder: TensorR): Unit

    // output = x * alpha + y * beta (can be in-place if output is x or y)
    def geam(x: Tensor, transX: Boolean, alpha: Rep[Float], y: Tensor, transY: Boolean, beta: Rep[Float], output: Tensor): Unit
    def trans(x: Tensor): Tensor
    def trans_grad(x: TensorR, y: TensorR): Unit
    def permute(x: Tensor, dims: Int*): Tensor
    def permute_grad(x: TensorR, y: TensorR, dims: Int*): Unit

    def gemm(x: Tensor, transX: Boolean, y: Tensor, transY: Boolean, alpha: Float): Tensor
    def gemm_grad(x: TensorR, transX: Boolean, y: TensorR, transY: Boolean, alpha: Float, output: TensorR): Unit

    /**
      * 2D convolution.
      * @param input Input with shape [batchSize, inChannels, iW, iH].
      * @param kernel Kernel with shape [outChannels, inChannels, kW, kH].
      * @param bias Optional bias with shape [outChannels].
      * @param strides Kernel strides of length two: [strideWidth, strideHeight].
      * @param pads Padding of length four: [padTop, padBottom, padLeft, padRight].
      * @return Result of 2D convolution.
      */
    // NOTE: cuDNN accepts only two padding arguments: [padVertical, padHorizontal].
    def conv2D_batch(input: Tensor, kernel: Tensor, bias: Option[Tensor], strides: Seq[Int], pads: Seq[Int]): (Tensor, Option[Tensor], Int)

    def conv2D_batch_grad(input: TensorR, finput: Option[TensorR], filter: TensorR, res: TensorR, bias: Option[TensorR] = None,
                          padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int), counter: Int): Unit

    def conv2DTraining(input: TensorR, kernel: TensorR, bias: Option[TensorR], resShape: Seq[Rep[Int]], strides: Seq[Int], pads: Seq[Int]): TensorR@diff

    def maxPool2D_batch(input: Tensor, kernel: Seq[Int], strides: Seq[Int], pads: Option[Seq[Int]]): (Tensor, Option[Rep[Array[Int]]])
    def maxPool2D_batch_grad(input: TensorR, output: TensorR, sidx: Option[Rep[Array[Int]]], kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Unit

    def averagePool2D_batch(input: Tensor, kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Tensor
    def averagePool2D_batch_grad(input: TensorR, output: TensorR, kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Unit

    def batchNormInference(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): Tensor
    def batchNormTraining(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): (Tensor, Option[Tensor], Option[Tensor], Int)
    def batchNorm_grad(input: TensorR, res: TensorR, scale: TensorR, bias: TensorR, saveMean: Option[Tensor], saveInvVariance: Option[Tensor], counterId: Int): Unit

    def batchNorm1DInference(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): Tensor
    def batchNorm1DTraining(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): (Tensor, Option[Tensor], Option[Tensor], Int)
    def batchNorm1D_grad(input: TensorR, res: TensorR, scale: TensorR, bias: TensorR, saveMean: Option[Tensor], saveInvVariance: Option[Tensor], counterId: Int): Unit

    def dropout(input: Tensor, prob: Float = 0.5f): (Tensor, Rep[Array[Float]], Rep[Int])
    def dropout_grad(input: TensorR, output: TensorR, prob: Float, helper: Rep[Array[Float]], size: Rep[Int]): Unit

    // multihead attention
    // TODO - add default size descriptors, default shape
    // Then in apply() check whether query.shape is equal to the default shape
//    case class MultiheadAttnConfig(weights: TensorR, numHeads: Int, embedDim:Int,
//                                   defaultDevQSeqArray: Rep[Array[Int]], defaultKSeqArray: Rep[Array[Int]], defaultLoWinIdx: Rep[Array[Int]], defaultHiWinIdx: Rep[Array[Int]], bias: Boolean,
//                                   dropoutRate :Float = 0.0f, smScaler: Float = 1.0f, residuals: Boolean)

    abstract class MultiheadAttnConfig {
      val weights: TensorR
      val numHeads: Int
      val embedDim: Int
      val defaultQSeqArray: Rep[Array[Int]]
      val defaultKSeqArray: Rep[Array[Int]]
      val bias: Boolean
      val dropoutRate: Float
      val smScaler: Float
      val residualConnection: Boolean
    }


    def multiheadAttention(query: TensorR, key: TensorR, value: TensorR, weights: TensorR, numHeads: Int, embedDim:Int, 
      devqSeqArray: Rep[Array[Int]], devkSeqArray: Rep[Array[Int]], loWinIdx: Rep[Array[Int]], hiWinIdx: Rep[Array[Int]], bias: Boolean, 
      dropoutRate :Float = 0.0f, smScaler: Float = 1.0f, residuals: Boolean): (Tensor, Rep[Array[Float]], Rep[Int], Rep[Array[Float]],
       Rep[Int], Rep[Int], Rep[Array[Int]], Rep[Array[Int]])

    def multiheadAttention_grad(output: TensorR, query: TensorR, key: TensorR, value: TensorR, weights: TensorR, numHeads: Int, embedDim:Int, 
      qSeqArray: Rep[Array[Int]], kSeqArray: Rep[Array[Int]], devQSeqArray: Rep[Array[Int]], devKSeqArray: Rep[Array[Int]], loWinIdx: Rep[Array[Int]], 
      hiWinIdx: Rep[Array[Int]], bias: Boolean, dropoutRate: Float = 0.0f, smScaler: Float = 1.0f, devWkSpace: Rep[Array[Float]], sizeWkSpace: Rep[Int], devReserve: Rep[Array[Float]], 
      sizeReserve: Rep[Int], sizeWeights: Rep[Int], residuals: Boolean): Unit

    // inplace mask (input is of size Batch * c * d * Time, lengths are the actual length of each sequence in batch)
    def mask4D(input: Tensor, lengths: Rep[Array[Int]]): Tensor

    // Activation functions.
    def relu(x: Tensor, inPlace: Boolean = false): Tensor
    def tanh(x: Tensor): Tensor
    def sigmoid(x: Tensor): Tensor
    def hardTanh(x: Tensor, min_val: Float = -1.0f, max_val: Float = 1.0f, inPlace: Boolean = false): Tensor
    def exp(x: Tensor): Tensor
    def log(x: Tensor): Tensor
    def sqrt(x: Tensor): Tensor
    def square(x: Tensor): Tensor

    def relu_grad(input: TensorR, res: TensorR, inPlace: Boolean = false): Unit
    def tanh_grad(input: TensorR, res: TensorR): Unit
    def sigmoid_grad(input: TensorR, res: TensorR): Unit
    def hardTanh_grad(input: TensorR, res: TensorR, min_val: Float = -1.0f, max_val: Float = 1.0f, inPlace: Boolean = false): Unit
    def exp_grad(input: TensorR, res: TensorR): Unit
    def log_grad(input: TensorR, res: TensorR): Unit
    def sqrt_grad(input: TensorR, res: TensorR): Unit
    def square_grad(input: TensorR, res: TensorR): Unit

    // Softmax functions.
    def softmax(x: Tensor, dim: Int = 1): Tensor
    def logSoftmax(x: Tensor, dim: Int = 1): Tensor

    def softmax_grad(input: TensorR, res: TensorR, dim: Int = 1): Unit
    def logSoftmax_grad(input: TensorR, res: TensorR, dim: Int = 1): Unit

    // Loss functions.
    def nllLoss(x: Tensor, target: Rep[Array[Int]]): Tensor
    def nllLoss_grad(input: TensorR, res: TensorR, target: Rep[Array[Int]]): Unit
    def mseLoss(x: Tensor, target: Rep[Array[Float]]): Tensor
    def mseLoss_grad(input: TensorR, res: TensorR, target: Rep[Array[Float]]): Unit

    // CTCLoss
    def ctcLoss(prob: TensorR, inputLengths: Rep[Array[Int]], labels: Rep[Array[Int]], labelLengths: Rep[Array[Int]]): Tensor

    // Reduction operations.
    def sum(x: Tensor): Tensor
    def sum_grad(input: TensorR, res: TensorR): Unit
    def mean(x: Tensor): Tensor
    def mean_grad(input: TensorR, res: TensorR): Unit

    // TODO: Add more ops:
    // - Reduction operators (e.g. sum).
    //   - Reduction op GPU implementations are non-trivial.
    //   - Roll out own reduction op kernels? There may be significant boilerplate.
    //   - Use thrust library reduction ops? Need to consider device_vector initialization overhead.
    // - Fused multiply add operations?

    // Reduction on one dimension
    def sum(x: Tensor, dim: Int): Tensor
    def sum_grad(input: TensorR, output: TensorR, dim: Int): Unit

    // concatenate
    def concat(dim: Int, tensors: Seq[Tensor]): Tensor
    def concat_grad(dim: Int, tensorRs: Seq[TensorR], output: TensorR): Unit

    // repeat on the first dimension with some number of context
    def repeat0(in: Tensor, context: Int): Tensor
    def repeat0_grad(in: TensorR, out: TensorR, context: Int): Unit

    // various kinds of gradient descent
    def adagrad_update(tr: TensorR, t: Tensor, learning_rate: Float, gradClip: Float, descent: Boolean): Unit
    def momentum_update(tr: TensorR, t: Tensor, learning_rate: Float, momentum: Float, gradClip: Float, nesterov: Boolean, descent: Boolean): Unit
  }

  // The current backend for code generation.
  // To switch code generation to a different backend, simply change this value
  // in your DSL program.
  var backend: Backend

  class Tensor(val data: Rep[Array[Float]], val dimensions: Seq[Rep[Int]]) extends Serializable {

    def shape = Dimensions(dimensions)
    val rank = dimensions.length
    assertC (rank > 0, "Tensors need to have nonEmpty dimensions")
    val scalarCount = shape.scalarCount
    val isScalar = scalarCount == unit(1)

    assertC(scalarCount != 0, "Tensor cannot have scalar count 0")

    def apply(i: Rep[Int]): Tensor = new Tensor(sliceReadWrite(data, i * shape.strides(0)), shape.tail)
    // Slice: i inclued, j excluded
    def apply(i: Rep[Int], j: Rep[Int]): Tensor = new Tensor(sliceReadWrite(data, i * shape.strides(0)), (j - i) +: shape.tail)

    def clipAt(bound: Float) = backend.clipAt(this, bound)
    def mutate(delta: Rep[Int] => Rep[Float]) = backend.mutate(this, delta)
    def mapInPlace(op: Rep[Float] => Rep[Float]) = backend.mapInPlace(this, op)
    def changeTo(gen: Rep[Int] => Rep[Float]) = backend.changeTo(this, gen)
    def map(op: Rep[Float] => Rep[Float]) = backend.map(this, op)
    def fold(init: Rep[Float])(op: (Rep[Float], Rep[Float]) => Rep[Float]) = backend.fold(init)(this, op)

    // bias may need to be broadcasted; plus is inPlace
    def plusBias(that: Tensor): Tensor = backend.plusBias(this, that)
    // that is the same size as this; plus is inPlace
    def plusEqual(that: Tensor): Tensor = backend.plusEqual(this, that)

    // Elementwise addition.
    def +(that: Rep[Float]): Tensor = backend.+(this, that)
    def +(that: Tensor): Tensor = backend.+(this, that)._1

    // In-place elementwise addition.
    def +=(that: Rep[Float]): Unit = backend.+=(this, that)
    def += (that: Tensor): Unit = backend.+=(this, that)

    // Elementwise subtraction.
    def -(that: Rep[Float]): Tensor = backend.-(this, that)
    def -(that: Tensor): Tensor = backend.-(this, that)._1

    // In-place elementwise subtraction.
    def -=(that: Rep[Float]): Unit = backend.-=(this, that)
    def -= (that: Tensor): Unit = backend.-=(this, that)

    // Elementwise multiplication.
    def *(that: Rep[Float]): Tensor = backend.*(this, that)
    def *(that: Tensor): Tensor = backend.*(this, that)._1

    // In-place elementwise multiplication.
    def *=(that: Rep[Float]): Unit = backend.*=(this, that)
    def *= (that: Tensor): Unit = backend.*=(this, that)

    // Elementwise division.
    def /(that: Rep[Float]): Tensor = backend./(this, that)
    def /(that: Tensor): Tensor = backend./(this, that)._1

    // In-place elementwise division.
    def /=(that: Rep[Float]): Unit = backend./=(this, that)
    def /= (that: Tensor): Unit = backend./=(this, that)

    def fillInPlace(value: Rep[Float]): Unit = backend.fillInPlace(this, value)
    def setAsOne() = fillInPlace(1)
    def clear() = fillInPlace(0)

    // Copy data from another tensor, in-place.
    def copy_data(src: Tensor) = backend.copyTensorData(this, src)

    // `dot` represents the following:
    // - vector-vector dot product.
    //   [V] dot [V] => [1] (scalar)
    // - matrix-vector multiplication.
    //   [M1 x M2] dot [M2] => [M1]
    // - matrix-matrix multiplication.
    //   [M1 x M2] dot [M2 x M3] => [M1 x M3]
    def dot(that: Tensor) = {
      generate_comment(s"dot: ${this.shape.seq}, ${that.shape.seq}")
      (this.rank, that.rank) match {
        case (1, 1) => assertC(this.shape(0) == that.shape(0), s"Incompatible shapes: ${this.shape}, ${that.shape}")
        case (2, 1) | (2, 2) => assertC(this.shape(1) == that.shape(0), s"Incompatible shapes: ${this.shape}, ${that.shape}")
        case _ => throw new IllegalArgumentException(
          s"Only vector-vector, matrix-vector, and matrix-matrix multiplication are allowed (actual shapes: ${this.shape}, ${that.shape})")
      }
      backend.dot(this, that)
    }

    // `this` is 2-D matrix, `that` is dims(1)-sized vector, `y` is dims(0)-sized vector
    // this function updates `this` so that this += that * y, where * is Cartesian product
    def add_cartesian(that: Tensor, y: Tensor): Unit = backend.add_cartesian(this, that, y)

    // `this` is vector of size `a`, `y` is vector of size `b`, that is 2-D matrix of size (b x a)
    // this function updates `this`, so that this += that^T dot y.
    def add_composition(that: Tensor, y: Tensor): Unit = backend.add_composition(this, that, y)

    // this += that^T dot y
    def add_dotTrans1(that: Tensor, y: Tensor): Unit = backend.add_dotTrans1(this, that, y)
    // this += that dot y^T
    def add_dotTrans2(that: Tensor, y: Tensor): Unit = backend.add_dotTrans2(this, that, y)

    def gemm(that: Tensor, transX: Boolean, transY: Boolean, alpha: Float): Tensor = {
      generate_comment(s"gemm: ${this.shape.seq}, ${that.shape.seq}")
      backend.gemm(this, transX, that, transY, alpha)
    }

    def trans() = backend.trans(this)
    def permute(dims: Int*) = backend.permute(this, dims: _*)

    def exp() = backend.exp(this)
    def log() = backend.log(this)
    def sqrt() = backend.sqrt(this)
    def square() = backend.square(this)

    def mask4D(lengths: Rep[Array[Int]]): Tensor = backend.mask4D(this, lengths)

    def relu(inPlace: Boolean = false) = backend.relu(this, inPlace)
    def tanh() = backend.tanh(this)
    def sigmoid() = backend.sigmoid(this)
    def hardTanh(min_val: Float = -1.0f, max_val: Float = 1.0f, inPlace : Boolean = false) = backend.hardTanh(this, min_val, max_val, inPlace)

    // NOTE: sum all elements
    def sum() = backend.sum(this)

    // sum over one dimension
    def sum(dim: Int) = backend.sum(this, dim)

    def mean() = backend.mean(this)

    def batchNormInference(scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): Tensor =
      backend.batchNormInference(this, scale, bias, runningMean, runningVar)

    def batchNormTraining(scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): (Tensor, Option[Tensor], Option[Tensor], Int) =
      backend.batchNormTraining(this, scale, bias, runningMean, runningVar)

    def batchNorm1DInference(scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): Tensor =
      backend.batchNorm1DInference(this, scale, bias, runningMean, runningVar)

    def batchNorm1DTraining(scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): (Tensor, Option[Tensor], Option[Tensor], Int) =
      backend.batchNorm1DTraining(this, scale, bias, runningMean, runningVar)

    @virtualize
    def batchNormAv() = {
      assert(this.rank == 4, "tensor for batch normal averaging should have 4 dimensions")
      val base: Rep[Float] = this.shape(0) * this.shape(2) * this.shape(3) * 1.0f
      val res = Tensor.zeros(this.shape(1), 1, 1)

      for (batch <- DataLoop(this.shape(0))) {
        val offsetBatch = batch * this.shape.strides(0)
        for (channel <- DataLoop(this.shape(1))) {
          val offset = offsetBatch + channel * this.shape.strides(1)
          for (lower <- DataLoop(this.shape.strides(1))) {
            res.data(channel) = res.data(channel) + this.data(offset + lower)
          }
        }
      }
      res / base
    }

    // mark: (Fei Wang) HERE: more gardening below
    // only for debugging, maybe remove?
    @virtualize
    def check(limit: Float) = {
      val idx = var_new(0)
      while (idx < this.scalarCount && -limit < this.data(idx) && this.data(idx) < limit) {
        idx += 1
      }
      idx != this.scalarCount
    }

    @virtualize
    def max() = this.fold(scala.Float.MinValue)((agg, x) => if (x > agg) x else agg)

    @virtualize
    def max2D(dim: Int) = {
      assert (this.rank == 2, "Only deal with 2D tensor")
      assert (dim == 0 || dim == 1, "dim must be in range of this.nbDims")

      if (dim == 0) ???
      else {
        val res = backend.mallocArray[Float](this.shape(0))
        val offset = var_new(0)
        for (i <- DataLoop(this.shape(0))) {
          val max = var_new(scala.Float.MinValue)
          for (j <- DataLoop(this.shape(1))) {
            if (this.data(offset) > max) max = this.data(offset)
            offset += 1
          }
          res(i) = max
        }
        Tensor(res, this.shape(0))
      }
    }

    // FIXME: Proper tensor
    @virtualize
    def maxIndex() = {
      assert(this.rank == 1, s"rank is ${this.rank}")
      val vMax = var_new(this.data(0))
      val iMax = var_new(0)
      for (idx <- 1 until this.scalarCount: Rep[Range]) {
        if (this.data(idx) > vMax) {
          iMax = idx
          vMax = this.data(idx)
        }
      }
      iMax
    }

    @virtualize
    def softmax_batch(dim: Int = 1) = {
      if (dim < 0) backend.softmax(this, this.rank + dim)
      else backend.softmax(this, dim)
    }

    // deprecated (older version that only works for 1D case), should remove
    @virtualize
    def softmax() = {
      assert(this.rank == 1, "TODO: softmax only handles 1d vectors so far: " + this.rank)

      val m = this.max
      val normalized = this.map(x => x - m)
      val nor_exp = normalized.exp()
      nor_exp / nor_exp.sum()
    }

    @virtualize  // batched log softmax
    def logSoftmaxB(dim: Int = 1) = {
      if (dim < 0) backend.logSoftmax(this, dim + this.rank)
      else backend.logSoftmax(this, dim)
    }

    def nllLossB(target: Rep[Array[Int]]) = backend.nllLoss(this, target)

    def mseLoss(target: Rep[Array[Float]]) = backend.mseLoss(this, target)

    def ctcLoss(inputLengths: Rep[Array[Int]], labels: Rep[Array[Int]], labelLengths: Rep[Array[Int]]): Tensor =
      backend.ctcLoss(TensorR(this), inputLengths, labels, labelLengths)

    @virtualize
    def resize(dims: Rep[Int]*) = Tensor(this.data, resizeDim(this.scalarCount, dims): _*)
    @virtualize
    def resizeNoCheck(dims: Rep[Int]*) = Tensor(this.data, dims: _*)

    @virtualize
    // NOTE: this function is fixed to run on CPU!
    def amax() = {
      val res = var_new[Float](0.0f)
      for (i <- DataLoop(this.scalarCount)) __assign(res, if (Math.abs(res) > Math.abs(this.data(i))) res else this.data(i))
      res
    }

    def printHead(count: Int = 10, msg: String = ""): Unit = {
      if (msg != "") {
        // printf(s"$msg (size ${this.shape.seq.map(quote) mkString " x "})\\n")
        val frm = s"$msg (size " + this.shape.seq.map(_ => "%d").mkString(" x ") + ")\\n"
        printf(frm, this.shape.seq: _*)
      }

      printf(s"Max Abs: ${format}|| ", this.amax())
      for (i <- 0 until count: Rep[Range]) {
        printf(format, this.data(i))
      }
      printf("\\n")
    }

    def printPane(msg: String = ""): Unit = {
      assert(this.rank == 4, "printPane only applies to Tensors of rank 4")
      printf(s"${msg} --> Pane # - ${this.shape(2)} x ${this.shape(3)}\\n")
      for (k <- 0 until this.shape(2): Rep[Range]) {
        for (l <- 0 until this.shape(3): Rep[Range]) {
          printf(format, this.data(k * this.shape.strides(2) + l))
        }
        printf("\\n")
      }
      printf("\\n\\n")
    }

    def print(msg: String = ""): Unit = {
      if (msg != "")
        printf(s"$msg (size ${this.shape.seq mkString " x "})\\n")
      if (this.rank == 4) this.print4D
      else if (this.rank == 3) this.print3D
      else this.printRaw(this.shape.lastOption.getOrElse(unit(1)))
    }

    val format = "%.5f "
    def print4D() = {
      val idx = var_new(1)
      for (i <- 0 until this.shape(0): Rep[Range]) {
        val idx1 = var_new(1)
        for (j <- 0 until this.shape(1): Rep[Range]) {
          printf(s"Pane #(%d, %d) - ${this.shape(2)} x ${this.shape(3)}\\n", idx, idx1)
          for (k <- 0 until this.shape(2): Rep[Range]) {
            for (l <- 0 until this.shape(3): Rep[Range]) {
              printf(format, this.data(i * this.shape.strides(0) + j * this.shape.strides(1) + k * this.shape.strides(2) + l))
            }
            printf("\\n")
          }
          printf("\\n\\n")
          idx1 += 1
        }
        idx += 1
      }
    }

    def print3D() = {
      val idx = var_new(1)
      for (i <- 0 until this.shape(0): Rep[Range]) {
        printf(s"Pane #%d - ${this.shape(1)} x ${this.shape(2)}\\n", idx)
        for (k <- 0 until this.shape(1): Rep[Range]) {
          for (l <- 0 until this.shape(2): Rep[Range]) {
            printf(format, this.data(i * this.shape.strides(0) + k * this.shape.strides(1) + l))
          }
          printf("\\n")
        }
        printf("\\n\\n")
        idx += 1
      }
    }

    @virtualize
    def printRaw(row: Rep[Int] = 10) = {
      for (i <- 0 until this.scalarCount: Rep[Range]) {
        printf(format, data(i))
        val imod = i % row
        if (imod == row - 1)
          printf("\\n")
      }
      printf("\\n")
    }

    @virtualize
    def addMul(that: Tensor, y: Tensor) = {
      assert(this.rank == 2 && that.rank == 2 && y.rank == 2, s"Dimensions: ${this.shape.seq} - ${that.shape.seq} - ${y.shape.seq}")
      assertC(this.shape(0) == that.shape(0) && this.shape(1) == y.shape(1) && that.shape(1) == y.shape(0), s"Dimensions: ${this.shape.seq} + ${that.shape.seq} * ${y.shape.seq}")
      var offThis = var_new(0)
      var offThatR = var_new(0)
      var offYC = var_new(0)
      for (i <- DataLoop(this.shape(0))) {
        val offYR = var_new(offYC)
        for (j <- DataLoop(this.shape(1))) {
          val offY = var_new(offYR)
          val offThat = var_new(offThatR)
          for (k <- DataLoop(that.shape(1))) {
            this.data(offThis) = this.data(offThis) + that.data(offThat) * y.data(offY)
            offThat += 1
            offY += y.shape.strides(0)
          }
          offThis += 1
          offYR += 1
        }
        offThatR += that.shape.strides(0)
        offYC *= 0
      }
    }

    // private function to get data with default to the only element
    def getAt(i: Rep[Int]) = {
      if (this.isScalar) data(0)
      else data(i)
    }

    def square(t: Rep[Float]) = t * t
    def add_mult(a: Tensor, b: Tensor) = {
      assertC(Tensor.dimCompatible(a, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, b), "dim not Compatible in add_mult")

      val dims0M = mmax(shape.head, mmax(a.shape.head, b.shape.head))
      val dims1M = mmax(shape.get(1), mmax(a.shape.get(1), b.shape.get(1)))
      for (i <- DataLoop(dims0M * dims1M)) {
        if (this.isScalar) { data(0) = data(0) + a.getAt(i) * b.getAt(i) }
        else { data(i) = data(i) + a.getAt(i) * b.getAt(i) }
      }
    }

    def addMul(a: Rep[Float], b: Tensor) = {
      assert(this.shape == b.shape)

      generate_comment("Generate code for addMul")
      for (i <- DataLoop(this.scalarCount)) {
        this.data(i) = this.data(i) + a * b.data(i)
      }
    }

    def cmulAdd(a: Float, b: Tensor) = {
      assert(this.shape == b.shape)
      for (i <- DataLoop(this.scalarCount))
        this.data(i) = a * this.data(i) + b.data(i)

      this // FIXME ??
    }

    def add_div(a: Tensor, b: Tensor) = {
      assertC(Tensor.dimCompatible(a, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, b), "dim not Compatible in add_div")
      val dims0M = mmax(shape.head, mmax(a.shape.head, b.shape.head))
      val dims1M = mmax(shape.get(1), mmax(a.shape.get(1), b.shape.get(1)))
      for (i <- DataLoop(dims0M * dims1M)) {
        if (this.isScalar) { data(0) = data(0) + a.getAt(i) / b.getAt(i) }
        else { data(i) = data(i) + a.getAt(i) / b.getAt(i) }
      }
    }

    def minus_mult_div_square(a: Tensor, b: Tensor, c: Tensor) = {
      assertC(Tensor.dimCompatible(a, b)    && Tensor.dimCompatible(a, c)    && Tensor.dimCompatible(c, b)    &&
        Tensor.dimCompatible(this, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, c),
        "dim not competible in minus_mult_div_square")
      val dims0M = mmax(shape.head, mmax(a.shape.head, b.shape.head))
      val dims1M = mmax(shape.get(1), mmax(a.shape.get(1), b.shape.get(1)))
      for (i <- DataLoop(dims0M * dims1M)) {
        if (this.isScalar) { data(0) = data(0) - a.getAt(i) * b.getAt(i) / square(c.getAt(i)) }
        else { data(i) = data(i) - a.getAt(i) * b.getAt(i) / square(c.getAt(i)) }
      }
    }

    def add_oneMinusSquare_mult(a: Tensor, b: Tensor) = {
      assertC(Tensor.dimCompatible(a, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, b), "dim not Compatible in add_oneMinusSquare_mult")
      val dims0M = mmax(shape.head, mmax(a.shape.head, b.shape.head))
      val dims1M = mmax(shape.get(1), mmax(a.shape.get(1), b.shape.get(1)))
      for (i <- DataLoop(dims0M * dims1M)) {
        if (this.isScalar) { data(0) = data(0) + (1.0f - square(a.getAt(i))) * b.getAt(i) }
        else { data(i) = data(i) + (1.0f - square(a.getAt(i))) * b.getAt(i) }
      }
    }

    def oneMinusThenMult(t: Rep[Float]) = (1.0f - t) * t

    def add_oneMinusThenMult_mult(a: Tensor, b: Tensor) = {
      assertC(Tensor.dimCompatible(a, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, b), "dim not Compatible in add_oneMinusThenMult_mult")
      val dims0M = mmax(shape.head, mmax(a.shape.head, b.shape.head))
      val dims1M = mmax(shape.get(1), mmax(a.shape.get(1), b.shape.get(1)))
      for (i <- DataLoop(dims0M * dims1M)) {
        if (this.isScalar) { data(0) = data(0) + oneMinusThenMult(a.getAt(i)) * b.getAt(i) }
        else { data(i) = data(i) + oneMinusThenMult(a.getAt(i)) * b.getAt(i) }
      }
    }

    @virtualize
    def conv2D_batch(kernel: Tensor, bias: Option[Tensor], strides: Seq[Int], pads: Seq[Int]): (Tensor, Option[Tensor], Int) = {
      backend.conv2D_batch(this, kernel, bias, strides, pads)
    }

    @virtualize
    def averagePool2D_batch(kernels: Seq[Int], strides: Seq[Int], paddings: Option[Seq[Int]] = None): Tensor = paddings match {
      case Some(pads) => backend.averagePool2D_batch(this, kernels, strides, pads)
      case None => backend.averagePool2D_batch(this, kernels, strides, Seq(0,0,0,0))
    }

    @virtualize
    def maxPool(strideRow: Int, strideCol: Int) = {
      assert(this.rank == 3)

      val resHeight = this.shape(1) / strideRow
      val resWidth = this.shape(2) / strideCol
      val res = Tensor.fill(Seq(this.shape(0), resHeight, resWidth), scala.Float.MinValue)

      // FIXME adhoc transform tensor to be using generic type!
      val savedIdx = NewArray[Int](res.scalarCount)

      val oidxW = var_new(0)  // walks by channel in res
      val iidx = var_new(0)   // walks by 1 in input (this.data)
      for (ichan <- DataLoop(this.shape(0))) {
        val oidx = var_new(oidxW)  // walks by row in res
        for (ox <- DataLoop(res.shape(1))) {
          for (sx <- DataLoop(strideRow)) {
            val oidx2 = var_new(oidx)  // walks by 1 in res
            for (oy <- DataLoop(res.shape(2))) {
              for (sy <- DataLoop(strideCol)) {
                if (this.data(iidx) > res.data(oidx2)) {
                  res.data(oidx2) = this.data(iidx)
                  savedIdx(oidx2) = iidx
                }
                iidx += 1
              }
              oidx2 += 1
            }
          }
          oidx += res.shape.strides(1)
        }
        oidxW += res.shape.strides(0)
      }

      (res, savedIdx)
    }

    def maxPool2D_batch(kernels: Seq[Int], strides: Seq[Int], pads: Option[Seq[Int]]): (Tensor, Option[Rep[Array[Int]]]) =
      backend.maxPool2D_batch(this, kernels, strides, pads)

    def dropout(prob: Float = 0.5f) =
      backend.dropout(this, prob)

    @virtualize
    def concat(dim: Int, others: Tensor*) = {
      assert(others.size >= 1, "there should be at least one tensor in others")
      assert(dim >= 0 && dim < this.rank, s"dim should be within range of ${this.rank}")
      assert(others.forall(x => x.rank == this.rank), "all tensors should have the same number of dimensions")
      assertC(others.forallR{t=> (0 until this.rank: Range).forallR{i => t.shape(i) == this.shape(i) || i == dim}},
              "all dimensions except the concatenation dimension should be the same")

      generate_comment("back prop for concat")
      backend.concat(dim: Int, this +: others)
    }

    def repeat0(context: Int): Tensor = backend.repeat0(this, context)

    @virtualize
    def global_ave_batch() = {
      assert(this.rank == 4, "assume this is Tensor with 4D (batch * channel * width * height")
      val resTensor = Tensor.zeros(this.shape.take(2): _*)

      val scale = this.shape.strides(1)
      val offset = var_new(0)                      // offset of this, should step by this.strides(2)
      val res_offset = var_new(0)                  // offset of res, should step by 1
      // looping over each batch
      for (batch <- DataLoop(this.shape(0))) {
        // looping over each channel
        for (channel <- DataLoop(this.shape(1))) {
          // get average of a channel
          val sum = var_new(0.0f)
          val offset_a = var_new(offset)           // offset of this for averaging, should step by this.strides(3)
          for (i <- DataLoop(this.shape(2))) {
            for (j <- DataLoop(this.shape(3))) {
              sum += this.data(offset_a + j)
            }
            offset_a += this.shape.strides(2)
          }
          resTensor.data(res_offset) = sum / scale
          offset += this.shape.strides(1)
          res_offset += 1
        }
      }
      resTensor
    }

    @virtualize
    def global_ave() = {
      // the result should be 1D (channel), by averaging the numbers in (width * height)
      assert(this.rank == 3, "assume this is Tensor with 3D (channel * width * height)")

      val res = backend.mallocArray[Float](this.shape(0))
      // looping over each channel
      for (channel <- DataLoop(this.shape(0))) {
        val offset = var_new(this.shape.strides(0) * channel)
        val sum = var_new(0.0f)
        for (i <- DataLoop(this.shape(1))) {
          for (j <- DataLoop(this.shape(2))) {
            sum += this.data(offset + j)
          }
          offset += this.shape.strides(1)
        }
        res(channel) = sum / (this.shape.strides(0))
      }
      Tensor(res, this.shape(0))
    }

    // FIXME: the MNIST example precomput the mean and std
    // I thought that normalize would need to compute it first and then
    // modify the data to match the one requested.
    // SO here what is expected is to have mean = 0 and std = 1 knowing that
    // the current mean is m and the current std is s
    @virtualize
    // Force this function to run on CPU
    def normalize(m: Float, s: Float, inPlace: Boolean = false) = {
      if (inPlace) {
        for (i <- DataLoop(this.scalarCount)) this.data(i) = (this.data(i) - m) / s
        this
      } else {
        val res = NewArray[Float](this.scalarCount)
        for (i <- DataLoop(this.scalarCount)) res(i) = (this.data(i) - m) / s
        Tensor(res, this.shape: _*)
      }
    }

    // fused ops
    def linearTanh(x: Tensor, b: Tensor) = {
      // this is W. We want (W.dot(x)+b).tanh()
      assert(this.rank == 2 && x.rank == 1 && b.rank == 1, "limited generalization")
      generate_comment("forward for linearTanh")
      // val res_dot = backend.mallocArray[Float](this.shape(0))
      // val res_add = backend.mallocArray[Float](this.shape(0))
      val res_tanh = backend.mallocArray[Float](this.shape(0))
      val offSet = var_new(0)
      for (i <- DataLoop(this.shape(0))) {
        val sum = var_new(0.0f)
        for (j <- DataLoop(this.shape(1))) {
          sum += this.data(offSet + j) * x.data(j)
        }
        // res_dot(i) = sum
        // res_add(i) = b.data(i) + sum
        res_tanh(i) = Math.tanh(b.data(i) + sum).toFloat
        offSet += this.shape(1)
      }
      (Tensor(res_tanh, this.shape(0)))
    }

    def linear2Tanh(x: Tensor, W2: Tensor, x2: Tensor, b: Tensor) = {
      // this is W. We want (W.dot(x) + W2.dot(x2) + b).tanh()
      assert(this.rank == 2 && x.rank == 1 && W2.rank == 2 && x2.rank == 1 && b.rank == 1, "limited generalization")
      generate_comment("forward for linear2Tanh")
      val res_tanh = backend.mallocArray[Float](this.shape(0))
      val offSet = var_new(0)
      for (i <- DataLoop(this.shape(0))) {
        val sum = var_new(0.0f)
        for (j <- DataLoop(this.shape(1))) {
          sum += this.data(offSet + j) * x.data(j)
          sum += W2.data(offSet + j) * x2.data(j)
        }
        res_tanh(i) = Math.tanh(b.data(i) + sum).toFloat
        offSet += this.shape(1)
      }
      Tensor(res_tanh, this.shape(0))
    }
  }

  object Tensor {
    // this function actually SHOULD NOT be different for different backend
    def apply(data: Rep[Array[Float]], dims: Rep[Int]*) =
      backend.arrayToTensor(data, dims: _*)

    def dimCompatible(a: Tensor, b: Tensor) = {
      (a.shape == b.shape) || a.isScalar || b.isScalar
    }

    @virtualize
    def dimBroadcast(a: Seq[Rep[Int]], b: Seq[Rep[Int]]): Option[(Dimensions, Dimensions, Dimensions)] = {
      val header: Seq[Rep[Int]] = if (a.size > b.size) a.take(a.size - b.size) else b.take(b.size - a.size)
      val body: Seq[(Rep[Int], Rep[Int])] = (a.reverse zip b.reverse).reverse
      val comp: Rep[Boolean] = body.forallR{case (x, y) => x == unit(1) || y == unit(1) || x == y}
      assertC(comp, s"dimensions not compatible for broadcasting ${"%d," * a.size} with ${"%d," * b.size}", (a ++ b): _*)
      val Body: Seq[Rep[Int]] = body.map{case (x, y) => if (x <= y) y else x}
      val res: Seq[Rep[Int]] = header ++ Body
      if (a.size > b.size) {
        val shapeB = Seq.fill(a.size - b.size)(unit(1)) ++ b
        val sameA = (a zip res).forallR{case (x, y) => x == y}
        val sameB = (shapeB zip res).forallR{case (x, y) => x == y}
        Some((Dimensions(a, !sameA), Dimensions(shapeB, !sameB), Dimensions(res)))
      } else {
        val shapeA = Seq.fill(b.size - a.size)(unit(1)) ++ a
        val sameA = (shapeA zip res).forallR{case (x, y) => x == y}
        val sameB = (b zip res).forallR{case (x, y) => x == y}
        Some((Dimensions(shapeA, !sameA), Dimensions(b, !sameB), Dimensions(res)))
      }
    }

    def randseed(seed: Int) = unchecked[Unit]("srand(", seed, ")")
    def randseed() = unchecked[Unit]("srand(time(NULL))")
    def rand(dims: Int*) = randinit(dims.toSeq, 1.0f, None)
    def rand(dims: Seq[Int], scale: Float) = randinit(dims.toSeq, scale, None)
    def randinit(dim0: Int): Tensor = randinit(Seq(dim0), 1.0f, None)
    def randinit(dim0: Int, seed: Option[Int]): Tensor = randinit(Seq(dim0), 1.0f, seed)
    def randinit(dim0: Int, dim1: Int, scale: Float): Tensor = randinit(Seq(dim0, dim1), scale, None)
    def randinit(dims: Seq[Int], scale: Float = 1.0f, seed: Option[Int] = None): Tensor =
      backend.randinit(dims, scale, seed)

    def randn(dim0: Int, dim1: Int = 1, scale: Float = 1.0f, offset: Int = 0) = {
      val res = backend.mallocArray[Float](dim0 * dim1)
      for (i <- (0 until dim0 * dim1): Rep[Range]) res(i) = unchecked[Float]("d(gen)") * scale
      Tensor(res, dim0, dim1)
    }

    def randnorm(dims: Int*) = {
      val res = backend.mallocArray[Float](dims.product)
      for (i <- (0 until dims.product): Rep[Range]) res(i) = unchecked[Float]("d(gen)")
      Tensor(res, dims: _*)
    }

    def randPositive(dims: Int*) = {
      val scalarCount = dims.product
      val res = backend.mallocArray[Float](scalarCount)
      for (i <- (0 until scalarCount): Rep[Range]) res(i) = Random.rand()
      new Tensor(res, dims)
    }

    def fill(dims: Seq[Rep[Int]], value: Rep[Float]): Tensor = backend.fill(dims, value)

    def fill(dims: Seq[Rep[Int]], fFill: Seq[Rep[Int]] => Rep[Float]) = {
      val scalarCount = dims.product1
      val res = backend.mallocArray[Float](scalarCount)

      var idx = var_new(0)
      def innerFill(args: Seq[Rep[Int]]) = {
        res(idx) = fFill(args)
        idx += 1
      }

      val dum = (dims :\ innerFill _) {
        case (up, f) =>
          (args: Seq[Rep[Int]]) => {
            for (i <- 0 until up: Rep[Range]) {
              f(args :+ i)
            }
          }
      }
      dum(Seq[Rep[Int]]())
      new Tensor(res, dims)
    }

    def fillWithBias(dims: Seq[Rep[Int]], bias: Tensor, dim: Int) = backend.fillWithBias(dims, bias, dim)

    def scalar(value: Rep[Float]) = fill(Seq(1), value)

    def zeros(dims: Rep[Int]*): Tensor =
      new Tensor(backend.mallocArray[Float](dims.product1), dims)
    def zeros_like(that: Tensor) = zeros(that.shape: _*)
    def ones(dims: Rep[Int]*) = fill(dims, 1.0f)
    def ones_like(that: Tensor) = ones(that.shape: _*)
    def halves(dims: Int*) = fill(dims, 0.5f)

    def expand(vector: Tensor, dim1: Int) = {
      assert(vector.rank == 1)
      val res = backend.mallocArray[Float](vector.shape(0) * dim1)
      val off = var_new(0)
      for (j <- (0 until dim1): Rep[Range]){
        for (i <- (0 until vector.shape(0)): Rep[Range]) {
          res(off) = vector.data(i)
          off += 1
        }
      }
      new Tensor(res, unit(dim1) +: vector.shape)
    }

    def copy(tensor: Tensor) = {
      val res = backend.mallocArray[Float](tensor.scalarCount)
      for (i <- DataLoop(tensor.scalarCount)) res(i) = tensor.data(i)
      new Tensor(res, tensor.shape)
    }

    def fromData(scalars: Float*): Tensor = backend.makeTensor(Seq(scalars.length), scalars: _*)

    def fromData(dims: Seq[Int], scalars: Float*): Tensor = backend.makeTensor(dims, scalars: _*)

    @virtualize
    def ifShapeEqual(a: Dimensions, b: Dimensions): Rep[Boolean] = {
      if (a.dims.size == b.dims.size) {
        (a.dims zip b.dims).forallR{case (a, b) => a == b}
      } else false
    }

    def assertShapeEqual(a: Dimensions, b: Dimensions, errorPrefix: String = "") = {
      assertC(ifShapeEqual(a, b), s"$errorPrefix: tensor shapes are not equal %s, %s \\n", a.toString, b.toString)
    }

    def assertShapeNotEqual(a: Dimensions, b: Dimensions, errorPrefix: String = "") = {
      assertC(!ifShapeEqual(a, b), s"$errorPrefix: tensor shapes are equal %s, %s \\n", a.toString, b.toString)
    }

    @virtualize
    def assertEqual(a: Tensor, b: Tensor, mark: String = "", tal: Float = 0.0001f): Unit = {
      val errorPrefix = if (mark != "") s"ERROR ($mark)" else "ERROR"
      assertShapeEqual(a.shape, b.shape)

      val i = var_new(0)
      while (i < a.scalarCount && { val diff = a.data(i) - b.data(i); diff > -tal && diff < tal }) {
        i += 1
      }
      if (i < a.scalarCount) {
        printf("%s: tensor data are not equal at index %d, %.4f != %.4f\\n", errorPrefix, i, a.data(i), b.data(i))
        exit(1)
      }
    }
  }

  // Tensor type is the similar to NumR, just replace RFloat with Tensor
  // also Tensor internally use array, which is mutable by default
  // so both field are val (not var) and can be updated by += -= *= /= setAsOne()
  // all instances of vectors will be shepherded by c++ smart pointers, alleviating the memory leak problem
  // type diff = cps[Unit]
  class TensorR(val x: Tensor, val d: Tensor) extends Serializable {
    var isInput: Boolean = false // true if it is an input (no need to compute gradient)

    def apply(i: Rep[Int]) = new TensorR(x(i), d(i))
    def apply(i: Int, j: Int) = new TensorR(x(i, j), d(i, j))

    def clip_grad(bound: Float) = {
      d.clipAt(bound)
    }

    def plusBias(bias: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      backend.plusBias(this.x, bias.x); k(this)  // note: plusBias is in-place
      backend.plusBias_grad(this, bias)
    }

    // plusEqual assumes that "that" is of the same dimension as "this", and addition can be done in place
    def plusEqual(that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      backend.plusEqual(this.x, that.x); k(this)
      backend.plusEqual_grad(this, that)
    }

    def + (that: Rep[Float]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x + that); k(y)
      this.d += y.d
    }
    def + (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val (ya, xShape, yShape) = backend.+(x, that.x)
      val y = TensorR(ya); k(y)
      generate_comment("back prop for + op")
      backend.add_grad(this, that, y, xShape, yShape)
    }

    def - (that: Rep[Float]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x - that); k(y)
      this.d += y.d
    }
    def - (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val (ya, xShape, yShape) = backend.-(x, that.x)
      val y = TensorR(ya); k(y)
      generate_comment("back prop for - op")
      backend.minus_grad(this, that, y, xShape, yShape)
    }

    // this is element wise multiplication
    def * (that: Rep[Float]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x * that); k(y)
      backend.geam(this.d, false, 1.0f, y.d, false, that, this.d)
    }
    def * (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val (ya, xShape, yShape) = backend.*(x, that.x)
      val y = TensorR(ya); k(y)
      generate_comment("backprop for * op")
      backend.mul_grad(this, that, y, xShape, yShape)
    }

    // element wise division
    def / (that: Rep[Float]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x / that); k(y)
      this.d += y.d / that
    }
    def / (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val (ya, xShape, yShape) = backend./(x, that.x)
      val y = TensorR(ya); k(y)
      generate_comment("backprop for / op")
      backend.div_grad(this, that, y, xShape, yShape)
    }

    // `dot` represents the following:
    // - vector-vector dot product.
    //   [V] dot [V] => [1] (scalar)
    // - matrix-vector multiplication.
    //   [M1 x M2] dot [M2] => [M1]
    // - matrix-matrix multiplication.
    //   [M1 x M2] dot [M2 x M3] => [M1 x M3]
    def dot(that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val res = backend.dot(x, that.x)
      val y = TensorR(res); k(y)

      // back-propagate
      backend.dot_grad(this, that, y)
    }

    def gemm(that: TensorR, transX: Boolean, transY: Boolean, alpha: Float): TensorR @diff = shift { (k: TensorR => Unit) =>
      generate_comment("foward of gemm")
      val ty = TensorR(x.gemm(that.x, transX, transY, alpha)); k(ty)
      generate_comment(s"backprop for gemm ${x.shape.seq}, ${that.x.shape.seq}")
      backend.gemm_grad(this, transX, that, transY, alpha, ty)
    }

    def trans(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.trans()); k(y)
      // back-propagate
      backend.trans_grad(this, y)
    }

    def permute(dims: Int*): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.permute(dims: _*)); k(y)
      generate_comment(s"backprop for permute ${dims}")
      backend.permute_grad(this, y, dims: _*)
    }

    def exp(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(backend.exp(x)); k(y)
      generate_comment("backprop for exp")
      backend.exp_grad(this, y)
    }

    def log(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(backend.log(x)); k(y)
      generate_comment("backprop for log")
      backend.log_grad(this, y)
    }

    def sqrt(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(backend.sqrt(x)); k(y)
      generate_comment("backprop for sqrt")
      backend.sqrt_grad(this, y)
    }

    def square(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.square()); k(y)
      generate_comment("backprop for square")
      backend.square_grad(this, y)
    }

    def mask4D(lengths: Rep[Array[Int]]): TensorR @diff = shift { (k: TensorR => Unit) =>
      x.mask4D(lengths); k(this)
      generate_comment("backprop for mask4D, not sure if gradient should be masked as well?")
    }

    def relu(inPlace: Boolean = false): TensorR @diff = shift { (k: TensorR => Unit) =>
      if (inPlace) {
        this.x.relu(inPlace); k(this)
        backend.relu_grad(this, this, inPlace)
      } else {
        val y = TensorR(this.x.relu(inPlace)); k(y)
        backend.relu_grad(this, y, inPlace)
      }
    }

    def hardTanh(min_val: Float = -1.0f, max_val: Float = 1.0f, inPlace : Boolean = false): TensorR @diff = shift { (k: TensorR => Unit) =>
      if (inPlace) {
        this.x.hardTanh(min_val, max_val, inPlace); k(this)
        backend.hardTanh_grad(this, this, min_val, max_val, inPlace)
      } else {
        val y = TensorR(this.x.hardTanh(min_val, max_val, inPlace)); k(y)
        backend.hardTanh_grad(this, y, min_val, max_val, inPlace)
      }
    }

    def tanh(): TensorR @diff = shift { (k : TensorR => Unit) =>
      val y = TensorR(x.tanh()); k(y)
      backend.tanh_grad(this, y)
    }

    def sigmoid(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.sigmoid()); k(y)
      backend.sigmoid_grad(this, y)
    }

    def sum(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = new TensorR(x.sum(), Tensor.zeros(1)); k(y)
      generate_comment("'sum' gradient.")
      backend.sum_grad(this, y)
    }

    def mean(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = new TensorR(x.mean(), Tensor.zeros(1)); k(y)
      generate_comment("'mean' gradient")
      backend.mean_grad(this, y)
    }

    def sum(dim: Int): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.sum(dim)); k(y)
      // backprop
      backend.sum_grad(this, y, dim)
    }

    def batchNorm(scale: TensorR, bias: TensorR, runningMean: Tensor, runningVar: Tensor): TensorR @diff =
      shift { (k: TensorR => Unit) =>
        val (y, saveMean, saveInvVariance, counterId) = x.batchNormTraining(scale.x, bias.x, runningMean, runningVar)
        val ty = TensorR(y); k(ty);
        backend.batchNorm_grad(this, ty, scale, bias, saveMean, saveInvVariance, counterId)
      }

    def batchNorm1D(scale: TensorR, bias: TensorR, runningMean: Tensor, runningVar: Tensor): TensorR @diff =
      shift { (k: TensorR => Unit) =>
        val (y, saveMean, saveInvVariance, counterId) = x.batchNorm1DTraining(scale.x, bias.x, runningMean, runningVar)
        val ty = TensorR(y); k(ty);
        backend.batchNorm1D_grad(this, ty, scale, bias, saveMean, saveInvVariance, counterId)
      }

    def batchNormAv(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.batchNormAv()); k(y)

      // back-propagate
      val base: Rep[Float] = this.x.shape(0) * this.x.shape(2) * this.x.shape(3) * 1.0f

      for (batch <- DataLoop(this.x.shape(0))) {
        val offsetBatch = batch * this.x.shape.strides(0)
        for (channel <- DataLoop(this.x.shape(1))) {
          val offset = offsetBatch + channel * this.x.shape.strides(1)
          for (lower <- DataLoop(this.x.shape.strides(1))) {
            this.d.data(offset + lower) += y.d.data(channel) / base
          }
        }
      }
    }

    def softmax_batch(dim: Int = 1): TensorR @diff = shift { (k: TensorR => Unit) =>
      val adjust_dim = if (dim < 0) this.x.rank + dim else dim
      val y = TensorR(x.softmax_batch(adjust_dim)); k(y)
      backend.softmax_grad(this, y, adjust_dim)
    }

    def logSoftmaxB(dim: Int = 1): TensorR @diff = shift { (k: TensorR => Unit) =>
      val adjust_dim = if (dim < 0) this.x.rank + dim else dim
      val y = TensorR(x.logSoftmaxB(adjust_dim)); k(y)
      backend.logSoftmax_grad(this, y, adjust_dim)
    }

    def resize(dims: Rep[Int]*) = {
      val newDims = resizeDim(this.x.scalarCount, dims)
      new TensorR(new Tensor(this.x.data, newDims), new Tensor(this.d.data, newDims))
    }

    def resizeNoCheck(dims: Rep[Int]*) = new TensorR(new Tensor(this.x.data, dims), new Tensor(this.d.data, dims))

    def nllLossB(target: Rep[Array[Int]]): TensorR @diff = shift { (k: TensorR => Unit) =>
      assert (this.x.rank == 2, s"nllLossB() function only takes tensor of rank 2, got ${this.x.shape}")
      val y = TensorR(x.nllLossB(target)); k(y)
      generate_comment("'nllLossB' gradient.")
      backend.nllLoss_grad(this, y, target)
    }

    def mseLoss(target: Rep[Array[Float]]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.mseLoss(target)); k(y)
      generate_comment("'mseLoss' gradient")
      backend.mseLoss_grad(this, y, target)
    }

    def ctcLoss(inputLengths: Rep[Array[Int]], labels: Rep[Array[Int]], labelLengths: Rep[Array[Int]]): Tensor =
      backend.ctcLoss(this, inputLengths, labels, labelLengths)

    @virtualize
    def averagePoolBK(kernels: Seq[Int], strides: Seq[Int], pads: Option[Seq[Int]] = None): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(this.x.averagePool2D_batch(kernels, strides, pads))
      k(y)

      // back prop
      backend.averagePool2D_batch_grad(this, y, kernels, strides, pads match {case None => Seq(0,0,0,0); case Some(pads) => pads})
    }

    // @virtualize  // conv with batch, bias, and pads
    def convBBP(kernel: TensorR, bias: Option[TensorR], strides: Seq[Int], pads: Seq[Int]): TensorR@diff = shift { (k: TensorR => Unit) =>
      assert(this.isInput || this.d.scalarCount == this.x.scalarCount, "For convBBP, THIS is either input or intermediate stage")
      assert(this.x.rank == 4, "For convBBP, THIS is dim 4: batch, channel, row, col")
      val (output, finputOption, counterId) = bias match {
        case Some(bias) => backend.conv2D_batch(x, kernel.x, Some(bias.x), strides, pads)
        case None =>       backend.conv2D_batch(x, kernel.x, None, strides, pads)
      }
      val y = TensorR(output); k(y)

      generate_comment("conv2D back-propagate")
      val paddings = if (pads.size == 2) (pads(0), pads(1)) else {if (pads.size == 4) (pads(0), pads(2)) else {if (pads.size == 1) (pads(0), pads(0)) else ???}}
      val stridess = if (strides.size == 2) (strides(0), strides(1)) else ???
      finputOption match {
        case None => backend.conv2D_batch_grad(this, None, kernel, y, bias, paddings, stridess, dilations = (1, 1), counterId)
        case Some(finput) => backend.conv2D_batch_grad(this, Some(TensorR(finput)), kernel, y, bias, paddings, stridess, dilations = (1, 1), counterId)
      }
    }

    // def convBBP(kernel: TensorR, bias: Option[TensorR], strides: Seq[Int], pads: Seq[Int]): TensorR@diff = {
    //   // verification
    //   assert(this.isInput || this.d.scalarCount == this.x.scalarCount, "For convBBP, THIS is either input or intermediate stage")
    //   assert(this.x.rank == 4, "For convBBP, THIS is dim 4: batch, channel, row, col")
    //   assert(kernel.x.rank == 4, s"Kernel must be 4-D, but got ${kernel.x.rank}")
    //   assert(kernel.x.shape(1) == input.x.shape(1), s"In-channel count mismatch: input.shape[1] ${input.x.shape(1)} should match kernel.shape[1] ${kernel.x.shape(1)}")

    //   val (padH, padW) = if (pads.size == 1) (pads(0), pads(0)) else {if (pads.size == 2) (pads(0), pads(1)) else {if (pads.size == 4) (pads(0), pads(2)) else ???}}
    //   assertC(input.x.shape(2) + 2 * padH >= kernel.x.shape(2) && input.x.shape(3) + 2 * padW >= kernel.x.shape(3), "Error")
    //   bias match {
    //     case Some(bias) =>
    //       assert(bias.x.rank == 1, s"Bias should be 1-D, got ${bias.x.shape}")
    //       assert(bias.x.shape(0) == kernel.x.shape(0), "Bias length must equal number of out-channels")
    //     case None => ()
    //   }
    //   assert(strides.size == 2, "Strides should have length 2: [strideRow, strideColumn]")
    //   val ((strideRow:Int) :: (strideCol:Int) :: Nil) = strides.take(2).toList
    //   assert(strideRow >= 1, "Row stride must be at least 1")
    //   assert(strideCol >= 1, "Column stride must be at least 1")

    //   // compute result shape
    //   val resWidth = convSize(input.x.shape(2) + padH * 2, kernel.x.shape(2), strideRow)
    //   val resHeight = convSize(input.x.shape(3) + padW * 2, kernel.x.shape(3), strideCol)
    //   val resShape = Seq(input.x.shape(0), kernel.x.shape(0), resWidth, resHeight)
    //   backend.conv2DTraining(this, kernel, bias, resShape, strides, Seq(padH, padW))
    // }

    @virtualize  // maxpool with kernel size potentially different from strides, and works with batch dimension! can have optional paddings
    def maxPoolBK(kernels: Seq[Int], strides: Seq[Int], pads: Option[Seq[Int]] = None): TensorR @diff = shift { (k: TensorR => Unit) =>
      val (y, sidx) = backend.maxPool2D_batch(x, kernels, strides, pads)
      val ty = TensorR(y)
      k(ty)

      // back propagate
      pads match {
        case None => backend.maxPool2D_batch_grad(this, ty, sidx, kernels, strides, Seq(0, 0, 0, 0))
        case Some(pads) => backend.maxPool2D_batch_grad(this, ty, sidx, kernels, strides, pads)
      }
    }

    @virtualize
    def maxPool(strideRow: Int, strideCol: Int): TensorR @diff = shift { (k: TensorR => Unit) =>
      val (y, sidx) = this.x.maxPool(strideRow, strideCol)

      val ty = TensorR(y)
      k(ty)

      for (i <- 0 until y.scalarCount: Rep[Range]) {
        this.d.data(sidx(i)) += ty.d.data(i)
      }
    }

    @virtualize
    def concat(dim: Int, others: TensorR*): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = this.x.concat(dim, others.map(t => t.x): _*)
      val ty = TensorR(y)
      k(ty)

      // back propagate
      backend.concat_grad(dim, this +: others, ty)
    }

    def repeat0(context: Int): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(this.x.repeat0(context)); k(y)
      generate_comment("back prop for repeat0")
      backend.repeat0_grad(this, y, context)
    }

    @virtualize
    def global_ave_batch(): TensorR @diff = shift { k: (TensorR => Unit) =>
      val y = this.x.global_ave_batch()
      val ty = TensorR(y)
      k(ty)

      // back propagate
      val scale = 1.0f / this.x.shape.strides(1)
      val offset = var_new(0)                      // offset of this, should step by this.x.strides(2)
      val res_offset = var_new(0)                  // offset of res, should step by 1
      // looping over each batch
      for (batch <- DataLoop(this.x.shape(0))) {
        // looping over each channel
        for (channel <- DataLoop(this.x.shape(1))) {
          // reflect gradient of ty to this, by scale
          val offset_a = var_new(offset)           // offset of this, should step by this.x.strides(3)
          for (i <- DataLoop(this.x.shape(2))) {
            for (j <- DataLoop(this.x.shape(3))) {
              this.d.data(offset_a + j) += ty.d.data(res_offset) * scale
            }
            offset_a += this.x.shape.strides(2)
          }
          offset += this.x.shape.strides(1)
          res_offset += 1
        }
      }
    }

    @virtualize
    def dropout(prob: Float): TensorR @diff = shift { (k: TensorR => Unit) =>
      val (y, helper, size) = backend.dropout(this.x, prob)
      val ty = TensorR(y); k(ty)
      // back prop
      backend.dropout_grad(this, ty, prob, helper, size)
    }

    @virtualize
    def multiheadAttention(key: TensorR, value: TensorR, weights: TensorR, numHeads: Int, embedDim:Int, qSeqArray: Rep[Array[Int]], kSeqArray: Rep[Array[Int]], 
    loWinIdx: Rep[Array[Int]], hiWinIdx: Rep[Array[Int]], bias: Boolean, dropoutRate: Float = 0.0f, smScaler: Float = 1.0f, residuals: Boolean = false): TensorR @diff = shift{ (k: TensorR => Unit) =>
      val (y, devWkSpace, sizeWkSpace, devReserve, sizeReserve, sizeWeights, devQSeqArray, devKSeqArray) = 
      backend.multiheadAttention(this, key, value, weights, numHeads, embedDim, qSeqArray, kSeqArray, loWinIdx, hiWinIdx, bias, dropoutRate, smScaler, residuals)
      val ty = TensorR(y); k(ty)
      // backprop
      backend.multiheadAttention_grad(ty, this, key, value, weights, numHeads, embedDim, qSeqArray, kSeqArray, devQSeqArray, devKSeqArray, loWinIdx, hiWinIdx,
      bias, dropoutRate, smScaler, devWkSpace, sizeWkSpace, devReserve, sizeReserve, sizeWeights, residuals)
    }

    def print(msg: String = "", derivative: Boolean = false): Unit = {
      this.x.print(msg)
      if (derivative) {
        if (msg == "")
          printf("=========\\n")
        this.d.print(s"derivative $msg")
      }
    }

    def clear_all() = {
      x.clear()
      d.clear()
    }

    def clear_grad() = {
      d.clear()
    }
  }

  object TensorR {
    def apply(a: Tensor, isInput: Boolean = false): TensorR = {
      val d = if (isInput) Tensor.zeros(2) else Tensor.zeros_like(a)
      val res = new TensorR(a, d)
      res.isInput = isInput
      res
    }
    def apply(a: Rep[Array[Float]], dim0: Int, dim1: Int): TensorR = {
      new TensorR(Tensor(a, dim0, dim1), Tensor.zeros(dim0, dim1))
    }

    // def apply(dim0: Int, dim1: Int): TensorR = {
    //   new TensorR(Tensor.zeros(dim0, dim1), Tensor.zeros(dim0, dim1))
    // }

  }

  /**
   * In LMS, the `fun` function reifies Rep[T] => Rep[U] functions to Rep[T => U] functions.
   * In Lantern, the `FUN*` functions reify (TensorR => Unit) functions to (TensorR => Unit)
   *    functions, where staged functions (Rep[Array[Float] => Unit]) are created from (Rep[Array[Float]] => Rep[Unit]) functions
   */
  def FUNc(f: TensorR => Unit): (TensorR => Unit) = { (x:TensorR) =>
    val dims = x.x.shape.toSeq
    val f1 = fun("&", { (x0: Rep[Array[Float]], x1: Rep[Array[Float]]) =>
      f(new TensorR(Tensor(x0, dims: _*), Tensor(x1, dims: _*)))
    })
    f1(x.x.data, x.d.data)
  }

  /**
   * comment out this version until we can support array of arrays in LMS_clean
    def FUNc(f: TensorR => Unit): (TensorR => Unit) = { (x:TensorR) =>
      val dims = x.x.shape.toSeq
      val f1 = fun("&", { (x: Rep[Array[Array[Float]]]) =>
        tempFixEffect(f)(x)(dims)
      })
      val in = NewArray[Array[Float]](2)
      in(0) = x.x.data; in(1) = x.d.data
      f1(in) // f1 should take Array[Array[Float]] and update the gradient of x
    }
   */


  /**
   * With FUNc, we can define the conditional in Lantern. The conditional takes a Rep[Boolean] flag,
   * 2 branches of type (=> TensorR @diff) because they must not be evaluated when passed in as arguments
   * The conditional captures a continuation (if type TensorR => Unit) and must lift it to a staged function
   * via our FUNc method.
   * Then in IF, we create a staged if (via `if` under @virtualize macro). Within each branch, the staged
   * continuation is applied to the left branch or the right branch.
   * Note that creating the staged function via `FUNc` is necessary because otherwise the code after conditional
   * will be duplicated in both branches.
   */
  @virtualize
  def IF(c: Rep[Boolean])(a: =>TensorR @diff)(b: =>TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>
    val k1 = FUNc(k)
    if (c) RST(k1(a)) else RST(k1(b))
  }

  // This is the WRONG way to construct conditional, where the continuation of the conditional is duplicated.
  // FIXME(feiw): it is currently used by DeepSpeech?? Fix it later
  @virtualize
  def If(c: Boolean)(a: => TensorR @diff)(b: => TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>
    if (c) RST(k(a)) else RST(k(b))
  }

  /**
   * comment out until we can handle array of arrays in LMS_clean
    def FUNm(f: ArrayBuffer[TensorR] => Unit): (ArrayBuffer[TensorR] => Unit) = { (x: ArrayBuffer[TensorR]) =>
      val dims = x.map(_.x.shape.toSeq)
      val f1 = fun("&", { (x: Rep[Array[Array[Float]]]) =>
        val tensors = ArrayBuffer[TensorR]()
        for (u <- (0 until dims.length): Range) {
          tensors.append(new TensorR(Tensor(x(u * 2), dims(u) : _*), Tensor(x(u*2+1), dims(u) : _*)))
        }
        f(tensors)
      })
      val in = NewArray[Array[Float]](2 * dims.length)
      for (u <- (0 until dims.length): Range) {
        in(u*2) = x(u).x.data; in(u*2+1) = x(u).d.data
      }
      f1(in)
    }
   */


  def buildArrayBuffer(size: Int, dims: ArrayBuffer[Seq[Rep[Int]]], xs: Rep[Array[Float]]*) = ArrayBuffer[TensorR](
    ((0 until size): Range).map(i => new TensorR(Tensor(xs(i * 2), dims(i): _*), Tensor(xs(i * 2 + 1), dims(i): _*))): _*)

  /**
   * This is a variation of `FUN*` that reifies (TensorR* => Unit) functions, where there are undetermined number of TensorRs as parameters.
   * For instance, if a conditional returns more than one TensorRs, then the continuation of the conditional will take more than
   * one TensorRs as parameters. To lift that continuation as staged function, we will need FUNm
   * In FUNc, the internal staged function takes 2 Rep[Array[Float]]. In this function, the internal staged function
   * needs to take some even number of Rep[Array[Float]].
   *
   * Ideally, we can stage a function of type Rep[Array[Array[Float]] => Unit. However, the effect system of lms-clean
   * is not handling aliases well, thus not handling Array of Array well. So as a compromize, we are going to have multiple
   * staged functions for different number of parameters.
   */
  def FUNm(f: ArrayBuffer[TensorR] => Unit): (ArrayBuffer[TensorR] => Unit) = { (xs: ArrayBuffer[TensorR]) =>
    val dims = xs.map(_.x.shape.toSeq)
    xs.length match {
      case n if n == 2 => // there are 2 TensorRs
        val f1 = fun("&", { (x00: Rep[Array[Float]], x01: Rep[Array[Float]],
                             x10: Rep[Array[Float]], x11: Rep[Array[Float]]) =>
          f(buildArrayBuffer(n, dims, x00, x01, x10, x11))
        })
        f1(xs(0).x.data, xs(0).d.data, xs(1).x.data, xs(1).d.data)
      case n if n == 3 => // there are 3 TensorRs
        val f1 = fun("&", { (x00: Rep[Array[Float]], x01: Rep[Array[Float]],
                             x10: Rep[Array[Float]], x11: Rep[Array[Float]],
                             x20: Rep[Array[Float]], x21: Rep[Array[Float]]) =>
          f(buildArrayBuffer(n, dims, x00, x01, x10, x11, x20, x21))
        })
        f1(xs(0).x.data, xs(0).d.data, xs(1).x.data, xs(1).d.data, xs(2).x.data, xs(2).d.data)
      case n if n == 4 => // there are 4 TensorRs
        val f1 = fun("&", { (x00: Rep[Array[Float]], x01: Rep[Array[Float]],
                             x10: Rep[Array[Float]], x11: Rep[Array[Float]],
                             x20: Rep[Array[Float]], x21: Rep[Array[Float]],
                             x30: Rep[Array[Float]], x31: Rep[Array[Float]]) =>
          f(buildArrayBuffer(n, dims, x00, x01, x10, x11, x20, x21, x30, x31))
        })
        f1(xs(0).x.data, xs(0).d.data, xs(1).x.data, xs(1).d.data, xs(2).x.data, xs(2).d.data, xs(3).x.data, xs(3).d.data)
      case n => System.out.println(s"$n number of TensorRs is not yet supported"); ???
    }
  }

  @virtualize
  def IFm(c: Rep[Boolean])(a: => ArrayBuffer[TensorR] @diff)(b: => ArrayBuffer[TensorR] @diff): ArrayBuffer[TensorR] @diff =
    shift { k: (ArrayBuffer[TensorR] => Unit) =>
      val k1 = FUNm(k)
      if (c) RST(k1(a)) else RST(k1(b))
    }

  /**
   * For the Loop construct, we don't need to lift the continuation of the Loop to staged functions (since it is only needed once)
   * However, we do need to create a recursive function for Loop, which depends on generating staged functions :)
   * If the signature of the recursive function is different, we will have to implement another FUN* function for it.
   *
   * The expected loop behavior is that we run a loop body over a init tensor many times (thinking of the init tensor as the hidden state of RNN).
   */
  @virtualize
  def LOOP(init: TensorR)(c: TensorR => Rep[Boolean])(b: TensorR => TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>
    lazy val loop: TensorR => Unit = FUNc { (x: TensorR) =>
      if (c(x)) RST(loop(b(x))) else RST(k(x))
    }
    loop(init)
  }

  /**
   * Comment this out until array of arrays is supported in LMS_clean
    def FUNs(f: Rep[Int] => TensorR => Unit): (Rep[Int] => TensorR => Unit) = { (i: Rep[Int]) => (x:TensorR) =>
      val dims = x.x.shape.toSeq
      val f1 = fun("&", { (i: Rep[Int], x: Rep[Array[Array[Float]]]) =>
        // tempFixEffect(f(i))(x)(dims)
        f(i)(new TensorR(Tensor(x(0), dims: _*), Tensor(x(1), dims: _*)))
      })
      val in = NewArray[Array[Float]](2)
      in(0) = x.x.data; in(1) = x.d.data
      // in(0) = uncheckedRead[Array[Float]](x.x.data)(x.x.data);
      // in(1) = uncheckedRead[Array[Float]](x.d.data)(x.d.data)
      f1(i, in)
    }
   */


  /**
   * If we want to keep a counter (Rep[Int]) in the recursive function (recursive loop body application), we will have to
   * implement a variation of FUNc with the additional Rep[Int] argument.
   */
  def FUNs(f: Rep[Int] => TensorR => Unit): (Rep[Int] => TensorR => Unit) = { (i: Rep[Int]) => (x: TensorR) =>
    val dims = x.x.shape.toSeq
    val f1 = fun("&", { (i: Rep[Int], x0: Rep[Array[Float]], x1: Rep[Array[Float]]) =>
      f(i)(new TensorR(Tensor(x0, dims: _*), Tensor(x1, dims: _*)))
    })
    f1(i, x.x.data, x.d.data)
  }

  @virtualize
  def LOOPS(init: TensorR)(c: Rep[Int])(b: Rep[Int] => TensorR => TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>
    lazy val loop: Rep[Int] => TensorR => Unit = FUNs { (i: Rep[Int]) => (x: TensorR) =>
      if (i < c) { RST(loop(i+1)(b(i)(x))) } else RST(k(x))
    }
    loop(0)(init)
  }

  /**
   * comment out until Array of Array is supported by LMS_clean
    def FUNsm(f: Rep[Int] => ArrayBuffer[TensorR] => Unit): (Rep[Int] => ArrayBuffer[TensorR] => Unit) = { (i: Rep[Int]) => (x:ArrayBuffer[TensorR]) =>
      val dims = x.map(_.x.shape.seq)
      val f1 = fun("&", { (i: Rep[Int], x: Rep[Array[Array[Float]]]) =>
        val tensors = ArrayBuffer[TensorR]()
        for (u <- (0 until dims.length): Range) {
          tensors.append(new TensorR(Tensor(x(u*2), dims(u) : _*), Tensor(x(u*2+1), dims(u) : _*)))
        }
        f(i)(tensors)
      })
      val in = NewArray[Array[Float]](2 * dims.length)
      for (u <- (0 until dims.length): Range) {
        in(u*2) = x(u).x.data; in(u*2+1) = x(u).d.data
      }
      f1(i, in)
    }
  */

  /**
   * Similarly, if the recursive loop body takes multiple TensorR as inputs, then another variation of FUN that handles multiple TensorRs is needed
   */
  def FUNsm(f: Rep[Int] => ArrayBuffer[TensorR] => Unit): (Rep[Int] => ArrayBuffer[TensorR] => Unit) = { (i: Rep[Int]) => (xs:ArrayBuffer[TensorR]) =>
    val dims = xs.map(_.x.shape.seq)
    xs.length match {
      case n if n == 2 => // 2 TensorRs
        val f1 = fun("&", { (i: Rep[Int], x00: Rep[Array[Float]], x01: Rep[Array[Float]],
                                          x10: Rep[Array[Float]], x11: Rep[Array[Float]]) =>
          f(i)(buildArrayBuffer(n, dims, x00, x01, x10, x11))
        })
        f1(i, xs(0).x.data, xs(0).d.data, xs(1).x.data, xs(1).d.data)
      case n if n == 3 => // 3 TensorRs
        val f1 = fun("&", { (i: Rep[Int], x00: Rep[Array[Float]], x01: Rep[Array[Float]],
                                          x10: Rep[Array[Float]], x11: Rep[Array[Float]],
                                          x20: Rep[Array[Float]], x21: Rep[Array[Float]]) =>
          f(i)(buildArrayBuffer(n, dims, x00, x01, x10, x11, x20, x21))
        })
        f1(i, xs(0).x.data, xs(0).d.data, xs(1).x.data, xs(1).d.data, xs(2).x.data, xs(2).d.data)
      case n if n == 4 => // 4 TensorRs
        val f1 = fun("&", { (i: Rep[Int], x00: Rep[Array[Float]], x01: Rep[Array[Float]],
                                          x10: Rep[Array[Float]], x11: Rep[Array[Float]],
                                          x20: Rep[Array[Float]], x21: Rep[Array[Float]],
                                          x30: Rep[Array[Float]], x31: Rep[Array[Float]]) =>
          f(i)(buildArrayBuffer(n, dims, x00, x01, x10, x11, x20, x21, x30, x31))
        })
        f1(i, xs(0).x.data, xs(0).d.data, xs(1).x.data, xs(1).d.data, xs(2).x.data, xs(2).d.data, xs(3).x.data, xs(3).d.data)
      case n => System.out.println(s"$n number of TensorRs is not yet supported"); ???
    }
  }

  @virtualize
  def LOOPSM(init: ArrayBuffer[TensorR])(c: Rep[Int])(b: Rep[Int] => ArrayBuffer[TensorR] => ArrayBuffer[TensorR] @diff):
    ArrayBuffer[TensorR] @diff = shift { k: (ArrayBuffer[TensorR] => Unit) =>
    lazy val loop: Rep[Int] => ArrayBuffer[TensorR] => Unit = FUNsm { (i: Rep[Int]) => (x: ArrayBuffer[TensorR]) =>
      if (i < c) {
        RST(loop(i+1)(b(i)(x)))
      } else {
        RST(k(x))
      }
    }
    loop(0)(init)
  }

  /**
   * Previous recursive functions (in LoopS and LoopSM) has been used to simulate loops (like recurrent neural networks)
   * Starting here, we need to use recursive functions to simulate recursion in the machine learning model.
   * NOTE(feiw): this part is very tricky and kind-of hard to understand :)
   *
   * In the LoopS and LoopSM cases, the recursions in the models are tail-recursions (making them essentially recurrent).
   * In the LoopL and LoopT cases, the recursions in the models are no longer tail-recursions.
   * In non-tail-recursive calls, the continuation after the recursive call are effectively "stacked up".
   * Another way to view it is that, in the recurrent cases, the continuation of the Loop* remain unchanged, and simply applies to the final value of the loop
   * However, in the recurrent cases, the continuation of the Loop* keeps growing, until the recursion reaches the base case.
   * For that reason, we need FUNl function that takes the continuation as parameter as well :)
   * We also have a FUN0 function but it is not use yet.
   *
   * Now for this category of FUN*, we are doing these things:
   * 1. The continuation parameter needs to be lifted as staged function (see val k_staged)
   * 2. The argument function `f` has to be staged (see val f1)
   */
  def FUN0(f: ((TensorR => Unit) => TensorR => Unit)): ((TensorR => Unit) => TensorR => Unit) = { k: (TensorR => Unit) => (x: TensorR) =>
    val dims = x.x.shape.toSeq
    // Stage the k1 continuation
    val k_staged = fun("&", { (x0: Rep[Array[Float]], x1: Rep[Array[Float]]) =>
      k(new TensorR(Tensor(x0, dims: _*), Tensor(x1, dims: _*)))
    })
    // Stage the `f` function argument
    val f_staged = fun("&", { (k_staged: Rep[(Array[Float], Array[Float]) => Unit], x0: Rep[Array[Float]], x1: Rep[Array[Float]]) =>
      // just wrap on t1
      val k_wrapped: (TensorR => Unit) = { (x: TensorR) => k_staged(x.x.data, x.d.data) }
      // apply `f` so that it can be reified
      f(k_wrapped)(new TensorR(Tensor(x0, dims: _*), Tensor(x1, dims: _*)))
    })
    f_staged(k_staged, x.x.data, x.d.data)
  }

  def FUNl(f: (Rep[Int] => (TensorR => Unit) => TensorR => Unit)): (Rep[Int] => (TensorR => Unit) => TensorR => Unit) = {i: Rep[Int] => k: (TensorR => Unit) => (x: TensorR) =>
    val dims = x.x.shape.toSeq
    // Stage the continuation parameter k1
    val k_staged: Rep[(Array[Float], Array[Float]) => Unit] = fun("&", { (x1: Rep[Array[Float]], x2: Rep[Array[Float]]) =>
      k(new TensorR(Tensor(x1, dims: _*), Tensor(x2, dims: _*)))
    })
    // Stage the argument function f
    val f_staged = fun("&", { (i: Rep[Int], k_staged: Rep[(Array[Float], Array[Float]) => Unit], x0: Rep[Array[Float]], x1: Rep[Array[Float]]) =>
      // t2 simply wrap on top of the staged t1 function, so that it can be feed to `f` in the next line
      val k_wrapped: (TensorR => Unit) = { (x: TensorR) => k_staged(x.x.data, x.d.data) }
      // apply the `f` function here so that it can by reified.
      f(i)(k_wrapped)(new TensorR(Tensor(x0, dims: _*), Tensor(x1, dims: _*)))
    })
    f_staged(i, k_staged, x.x.data, x.d.data)
  }

  @virtualize
  def LOOPL(init: TensorR)(c: Rep[Int])(b: Rep[Int] => TensorR => TensorR @diff): TensorR @diff = shift { k: (TensorR => Unit) =>
    lazy val loop: Rep[Int] => (TensorR => Unit) => TensorR => Unit = FUNl{ (gc: Rep[Int]) => (k: TensorR => Unit) => (x: TensorR) =>
      def sh_loop: (Rep[Int] => TensorR @diff) = (i: Rep[Int]) => shift{(k: TensorR => Unit) => loop(i)(k)(x)}
      RST(k (IF(gc < c) { b(gc)(sh_loop(gc+1)) } { init }) )
    }
    loop(0)(k)(init)
  }

  @virtualize
  def LOOPT(start: Rep[Int])(init: TensorR)(lch: Rep[Array[Int]], rch: Rep[Array[Int]])(b: (TensorR, TensorR, Rep[Int]) => TensorR @diff): TensorR @diff = shift {
    k: (TensorR => Unit) =>
      lazy val tree: Rep[Int] => (TensorR => Unit) => TensorR => Unit = FUNl{ (i: Rep[Int]) => (k: TensorR => Unit) => (x: TensorR) =>
        def sh_tree: (Rep[Int] => TensorR @diff) = (i: Rep[Int]) => shift{(k: TensorR => Unit) => tree(i)(k)(x)}
        RST(k( IF(i >= 0) { b(sh_tree(lch(i)), sh_tree(rch(i)), i) } { init } ))
      }
      tree(start)(k)(init)
  }

  /**
   * Comment this out until array of array is handled by LMS_clean
    def FUNlm(f: (Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit)):
    (Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit) = {i: Rep[Int] => k1: (ArrayBuffer[TensorR] => Unit) => (x: ArrayBuffer[TensorR]) =>

      val length = x.length
      val dims = x.map(_.x.shape.toSeq)
      val f1 = fun("&", { (i: Rep[Int], t1: Rep[Array[Array[Float]] => Unit], xx: Rep[Array[Array[Float]]]) =>
        val t2: (ArrayBuffer[TensorR] => Unit) = { (x: ArrayBuffer[TensorR]) =>
          val aa = NewArray[Array[Float]](2*length)
          for (u <- (0 until length): Range) {
            aa(u*2) = x(u).x.data; aa(u*2+1) = x(u).d.data
          }
          t1(aa)
        }
        val t3: (ArrayBuffer[TensorR] => Unit) = f(i)(t2)
        val tensors = ArrayBuffer[TensorR]()
        for (u <- (0 until length): Range) {
          tensors.append(new TensorR(Tensor(xx(u*2), dims(u): _*), Tensor(xx(u*2+1), dims(u): _*)))
        }
        t3(tensors)
      })
      val k2: Rep[Array[Array[Float]] => Unit] = fun(Capture("&"), { (x: Rep[Array[Array[Float]]]) =>
        val tensors = ArrayBuffer[TensorR]()
        for (u <- (0 until length): Range) {
          tensors.append(new TensorR(Tensor(x(u*2), dims(u): _*), Tensor(x(u*2+1), dims(u): _*)))
        }
        k1(tensors)
      })
      val arrays = NewArray[Array[Float]](2*length)
      for (u <- (0 until length): Range) {
        arrays(u*2) = x(u).x.data; arrays(u*2+1) = x(u).d.data
      }
      f1(i, k2, arrays)
    }
   */


  def FUNlm(f: Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit): (Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit) = {
    i: Rep[Int] => k: (ArrayBuffer[TensorR] => Unit) => xs: ArrayBuffer[TensorR] =>

    val dims = xs.map(_.x.shape.toSeq)
    xs.length match {
      case n if n == 2 =>
        // Stage k
        val k_staged = fun("&", { (x00: Rep[Array[Float]], x01: Rep[Array[Float]], x10: Rep[Array[Float]], x11: Rep[Array[Float]]) =>
          k(buildArrayBuffer(n, dims, x00, x01, x10, x11))
        })
        // Stage f
        val f_staged = fun("&", { (i: Rep[Int], k_staged: Rep[(Array[Float], Array[Float], Array[Float], Array[Float]) => Unit],
                                   x00: Rep[Array[Float]], x01: Rep[Array[Float]], x10: Rep[Array[Float]], x11: Rep[Array[Float]]) =>
          val k_wrapped: (ArrayBuffer[TensorR] => Unit) = { xs: ArrayBuffer[TensorR] =>
            k_staged(xs(0).x.data, xs(0).d.data, xs(1).x.data, xs(1).d.data)
          }
          f(i)(k_wrapped)(buildArrayBuffer(n, dims, x00, x01, x10, x11))
        })
        f_staged(i, k_staged, xs(0).x.data, xs(0).d.data, xs(1).x.data, xs(1).d.data)

      case n if n == 3 =>
        // Stage k
        val k_staged = fun("&", { (x00: Rep[Array[Float]], x01: Rep[Array[Float]], x10: Rep[Array[Float]], x11: Rep[Array[Float]], x20: Rep[Array[Float]], x21: Rep[Array[Float]]) =>
          k(buildArrayBuffer(n, dims, x00, x01, x10, x11, x20, x21))
        })
        // stage f
        val f_staged = fun("&", { (i: Rep[Int], k_staged: Rep[(Array[Float], Array[Float], Array[Float], Array[Float], Array[Float], Array[Float]) => Unit],
                                   x00: Rep[Array[Float]], x01: Rep[Array[Float]], x10: Rep[Array[Float]], x11: Rep[Array[Float]], x20: Rep[Array[Float]], x21: Rep[Array[Float]]) =>
          val k_wrapped: (ArrayBuffer[TensorR] => Unit) = { xs: ArrayBuffer[TensorR] =>
            k_staged(xs(0).x.data, xs(0).d.data, xs(1).x.data, xs(1).d.data, xs(2).x.data, xs(2).d.data)
          }
          f(i)(k_wrapped)(buildArrayBuffer(n, dims, x00, x01, x10, x11, x20, x21))
        })
        f_staged(i, k_staged, xs(0).x.data, xs(0).d.data, xs(1).x.data, xs(1).d.data, xs(2).x.data, xs(2).d.data)

      case n if n == 4 =>
        // Stage k
        val k_staged = fun("&", { (x00: Rep[Array[Float]], x01: Rep[Array[Float]], x10: Rep[Array[Float]], x11: Rep[Array[Float]],
                                   x20: Rep[Array[Float]], x21: Rep[Array[Float]], x30: Rep[Array[Float]], x31: Rep[Array[Float]]) =>
          k(buildArrayBuffer(n, dims, x00, x01, x10, x11, x20, x21, x30, x31))
        })
        // stage f
        val f_staged = fun("&", { (i: Rep[Int], k_staged: Rep[(Array[Float], Array[Float], Array[Float], Array[Float], Array[Float], Array[Float], Array[Float], Array[Float]) => Unit],
                                   x00: Rep[Array[Float]], x01: Rep[Array[Float]], x10: Rep[Array[Float]], x11: Rep[Array[Float]],
                                   x20: Rep[Array[Float]], x21: Rep[Array[Float]], x30: Rep[Array[Float]], x31: Rep[Array[Float]]) =>
          val k_wrapped: (ArrayBuffer[TensorR] => Unit) = { xs: ArrayBuffer[TensorR] =>
            k_staged(xs(0).x.data, xs(0).d.data, xs(1).x.data, xs(1).d.data, xs(2).x.data, xs(2).d.data, xs(3).x.data, xs(3).d.data)
          }
          f(i)(k_wrapped)(buildArrayBuffer(n, dims, x00, x01, x10, x11, x20, x21, x30, x31))
        })

      case n => System.out.println(s"$n number of TensorRs is not yet supported"); ???
    }
  }

  @virtualize
  def LOOPLM(start: Rep[Int])(init: ArrayBuffer[TensorR])(c: Rep[Int])(b: Rep[Int] => ArrayBuffer[TensorR] => ArrayBuffer[TensorR] @diff):
    ArrayBuffer[TensorR] @diff = shift { k: (ArrayBuffer[TensorR] => Unit) =>

    lazy val loop: Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit = FUNlm { (i: Rep[Int]) => (k: ArrayBuffer[TensorR] => Unit) => (x: ArrayBuffer[TensorR]) =>

      def sh_loop: (Rep[Int] => ArrayBuffer[TensorR] @diff) = (i: Rep[Int]) => shift{ (k: ArrayBuffer[TensorR] => Unit) => loop(i)(k)(x) }

      RST(k( IFm(i < c) { b(i)(sh_loop(i+1)) } { init } ))
    }
    loop(start)(k)(init)
  }

  @virtualize
  def LOOPTM(start: Rep[Int])(init: ArrayBuffer[TensorR])(lch: Rep[Array[Int]], rch: Rep[Array[Int]])
    (b: (ArrayBuffer[TensorR], ArrayBuffer[TensorR], Rep[Int]) => ArrayBuffer[TensorR] @diff): ArrayBuffer[TensorR] @diff = shift { k: (ArrayBuffer[TensorR] => Unit) =>

      lazy val tree: Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit = FUNlm { (i: Rep[Int]) => (k: ArrayBuffer[TensorR] => Unit) => (x: ArrayBuffer[TensorR]) =>

        def sh_tree: (Rep[Int] => ArrayBuffer[TensorR] => ArrayBuffer[TensorR] @diff) = (i: Rep[Int]) => (x: ArrayBuffer[TensorR]) => shift{(k: ArrayBuffer[TensorR] => Unit) => tree(i)(k)(x)}

        RST(k( IFm (i >= 0) { b(sh_tree(lch(i))(init), sh_tree(rch(i))(init), i) } { init } ))
      }
      tree(start)(k)(init)
  }

  def gradR(f: TensorR => TensorR @diff)(x: Tensor): Tensor = {
    val x1 = new TensorR(x, Tensor.zeros_like(x))
    reset {
      val y = f(x1)
      y.d.setAsOne()
      ()
    }
    x1.d
  }

  // same as gradR function, except that we return the final result of f, not the gradient of input
  // gradient of input is supposed to be dummy value here
  // gradient of useful tensors are in closure, and can be accessed directly from outside of this function
  def gradR_loss(f: TensorR => TensorR @diff)(x: Tensor): Tensor = {
    val x1 = TensorR(x) // this should be a dummy tensor
    // val result = Tensor.zeros(1)                  // this should be the loss
    generate_comment("allocate memory to save the final loss in CPU Tensor")
    val res = backend.mallocArray[Float](1)
    val result = Tensor(res, 1)
    reset {
      val y = f(x1)
      generate_comment("make sure the size of loss is 1")
      assertC(y.x.scalarCount == unit(1), "Loss function must return a Tensor of size 1, got %d\\n", y.x.scalarCount)
      y.d.setAsOne()
      generate_comment(s"backend is $backend")
      backend.copyFloatArray(res, y.x.data, 1)
      // if (backend.isInstanceOf[BackendCPU]) BackendCPU().copyFloatArray(res, y.x.data, 1)
      // else unchecked[Unit]("CUDA_CALL(cudaMemcpy(", res, ", ", y.x.data, ", ", 1, " * sizeof(float), cudaMemcpyDeviceToHost))")
    }
    result
  }

  def getMallocAddr(): Rep[Long] = {
    unchecked[Long]("(long)mallocAddr")
  }

  def resetMallocAddr(addr: Rep[Long]) = {
    unchecked[Unit]("memset((void*)", addr, ", 0, ", getMallocAddr() - addr, ")")
    unchecked[Unit]("mallocAddr = (void*)", addr)
  }
}

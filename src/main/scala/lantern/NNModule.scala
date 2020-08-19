package lantern

import scala.util.continuations._
import scala.util.continuations

import lms.core.stub._
import lms.macros.SourceContext
import lms.core.virtualize

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

    /**
      * A wrapper for non-parameter tensors.
      *
      * All `TensorR` fields are implicitly assumed to be parameters.
      * To declare a non-parameter `TensorR` field, wrap it using `Nonparameter`.
      * Note: this is a slight hack. A more robust solution is an explicit parameter registration system.
      */
    case class Nonparameter(t: TensorR)
    implicit def nonparameterToTensorR(n: Nonparameter): TensorR = n.t

    def forEachNamedParameter(f: (String, (TensorR, Option[Tensor])) => Unit): Unit = {
      parameters.foreach{ case (k, v) => f(k, v) }
      for ((_, module) <- modules) module.forEachNamedParameter(f)
    }

    def enrichParameter(): Unit = {
      for ((k, (tensorR, _)) <- parameters) parameters(k) = (tensorR, Some(Tensor.zeros_like(tensorR.x)))
      for ((_, module) <- modules) module.enrichParameter()
    }

    def forEachParameter(f: TensorR => Unit) =
      forEachNamedParameter { case (_, (tensorR, _)) => f(tensorR) }
    def forEachPairParameter(f: (TensorR, Option[Tensor]) => Unit) =
      forEachNamedParameter { case (_, (tr, t)) => f(tr, t) }

    // Implicit parameter registration system.
    def registerParameters(nameScope: String, parent: Option[Module] = None): Unit = {
      def oops[T](field: Field)(read: Field => T) = {
        val acc = field.isAccessible
        field.setAccessible(true)
        val res = read(field)
        field.setAccessible(acc)
        res
      }

      val allFields1: Array[java.lang.reflect.Field] = this.getClass.getDeclaredFields
      val allFields = allFields1.filter { field => oops[Boolean](field) {
        x => x.get(this) != this && (parent match { case None => true; case Some(p) => x.get(this) != p })
      }}

      // this part may take more than needed. For instance, all Option or ArrayBuffer type are included due to type erasure
      val subParameters = allFields.filter { f =>
        classOf[Option[TensorR]].isAssignableFrom(f.getType) ||
        classOf[TensorR].isAssignableFrom(f.getType) ||
        classOf[Seq[TensorR]].isAssignableFrom(f.getType)
      }
      val subModules = allFields.filter { f =>
        classOf[Module].isAssignableFrom(f.getType) && oops[Boolean](f) { _.get(this) != this } ||
        classOf[Option[Module]].isAssignableFrom(f.getType) ||
        classOf[Seq[Module]].isAssignableFrom(f.getType)
      }

      subParameters.foreach { field => oops[Unit](field) { x =>
        val field = x.get(this)
        val name = x.getName
        val fullName = s"${nameScope}${name}"
        field match {
          case t: TensorR => parameters.update(fullName, (t, None))
          case arr@((a: TensorR) +: rest) =>
            arr.asInstanceOf[Seq[TensorR]].zipWithIndex.foreach { case (t, i) =>
              val name = s"${fullName}_index_$i"
              parameters.update(name, (t, None))
            }
          case Some(t: TensorR) => parameters.update(fullName, (t, None))
          case _ => ()
        }
      }}
      subModules.foreach { field => oops[Unit](field) { x =>
        val field = x.get(this)
        val name = x.getName
        val fullName = s"${nameScope}${name}"
        field match {
          case t: Module => modules.update(fullName, t)
          case arr@((a: Module) +: rest) =>
            arr.asInstanceOf[Seq[Module]].zipWithIndex.foreach { case (t, i) =>
              val name = s"${fullName}_index_$i"
              modules.update(name, t)
            }
          case Some(t: Module) => modules.update(fullName, t)
          case _ => ()
        }
      }}

      modules.foreach {case (name, module) => module.registerParameters(s"${name}/", Some(this))}
    }
  }

  case class Linear1D(val inSize: Int, val outSize: Int, val bias: Boolean = true, val name: String = "linear1d") extends Module {
    val scale: Float = 1.0f / sqrt(inSize).toFloat
    val weight = TensorR(Tensor.rand(Seq(inSize, outSize), scale))
    val biasOp = if (bias) Some(TensorR(Tensor.zeros(outSize))) else None
//    def apply(in: TensorR): TensorR @diff = if (bias) in.dot(weight) plusBias biasOp.get else in.dot(weight)
    def apply(in: TensorR): TensorR @diff = if (bias) in.dot(weight) plusBias_v2 biasOp.get else in.dot(weight)
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

  case class Conv2D(val inChannel: Int, val outChannel: Int, val kernelSize: Seq[Int], val stride: Seq[Int] = Seq(1, 1), val pad: Seq[Int] = Seq(0, 0),
    val dilation: Seq[Int] = Seq(1, 1), val useBias: Boolean = true, val name: String = "conv2d") extends Module {
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
    def apply(in: TensorR): TensorR @diff = in.convBBP(kernel, bias, stride, pad)
  }

  case class Conv2Dn(val inChannel: Int, val outChannel: Int, val kernelSize: Seq[Int], val stride: Seq[Int] = Seq(1, 1), val pad: Seq[Int] = Seq(0, 0),
    val useBias: Boolean = true, val name: String = "conv2d") extends Module {
    assert(kernelSize.size == 2, "kernel_size should be Seq[Int] of size 2")
    assert(stride.size == 2, "stride should be Seq[Int] of size 2")
    // normal initialization with mean 0.0 and std 0.01
    val kernel = TensorR(Tensor.randnorm(outChannel, inChannel, kernelSize.head, kernelSize.last))
    val bias = if (useBias) Some(TensorR(Tensor.zeros(outChannel))) else None
    def apply(in: TensorR): TensorR @diff = in.convBBP(kernel, bias, stride, pad)
  }

  case class BatchNorm1D(dimSize: Int, name: String = "batch_norm_1d") extends Module {
    val scale: TensorR = TensorR(Tensor.ones(dimSize))
    val bias: TensorR = TensorR(Tensor.zeros(dimSize))
    val runningMean: Tensor = Tensor.zeros(dimSize)
    val runningVar: Tensor = Tensor.zeros(dimSize)
    @virtualize
    def apply(in: TensorR): TensorR @diff = {
      assert(in.x.rank == 2)
      assert(in.x.shape.dims(1) == unit(dimSize), "BatchNorm1D input should be rank2, with shape 1 same as dimSize, got %d : %d") //, in.x.shape(1), dimSize)
      in.batchNorm1D(scale, bias, runningMean, runningVar)
    }
  }

  case class BatchNorm2D(num_features: Int, eps: Float =1e-05f, momentum: Float =0.1f, affine: Boolean = true,
    track_running_stats: Boolean = true, name: String = "batch_norm_2d") extends Module {
    assert(affine && track_running_stats, "TODO: not yet handling them to be false")
    val scale: TensorR = TensorR(Tensor.ones(num_features))
    val bias: TensorR = TensorR(Tensor.zeros(num_features))
    val runningMean: Tensor = Tensor.zeros(num_features)
    val runningVar: Tensor = Tensor.zeros(num_features)
    def apply(in: TensorR): TensorR @diff = {
      assert(in.x.rank == 4 && in.x.shape(1) == unit(num_features), s"BatchNorm2D input should be rank 2, with shape 1 same as num_features, got ${in.x.shape} : ${num_features}")
      in.batchNorm(scale, bias, runningMean, runningVar)
    }
  }

  abstract class RnnCell extends Module {
    def init(batchSize: Rep[Int]): ArrayBuffer[TensorR]
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
    def init(batchSize: Rep[Int]) = ArrayBuffer(TensorR(Tensor.zeros(batchSize, hiddenSize)))
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
    def init(batchSize: Rep[Int]) = ArrayBuffer(TensorR(Tensor.zeros(batchSize, hiddenSize)), TensorR(Tensor.zeros(batchSize, hiddenSize)))
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

  /** MultiheadAttention_v2
   * @param embedDim Vector Embedding size of Query. All Q, K, V are projected to embedDim. Also, the final output size.
   * @param numHeads Number of attention heads
   * @param dropout dropout rate
   * @param bias whether to use bias or not in projections
   * @param qDim (initial) Vector Embedding size of Query (defaults to embedDim)
   * @param kDim (initial) Vector Embedding size of Key (defaults to embedDim)
   * @param vDim (initial) Vector Embedding size of Value (defaults to embedDim)
   */
  case class MultiheadAttention_v2(embedDim: Int, numHeads: Int, dropout: Float, bias: Boolean = true, qDim: Option[Int] = None,
                                   kDim: Option[Int] = None, vDim: Option[Int] = None, name: String = "mha-v2") extends Module {
    val headDim: Int = embedDim / numHeads

    // input projection weights
    // TODO - make bias = bias (just set false for testing)
    val qProjLayer = Linear1D(inSize = qDim.getOrElse(embedDim), outSize = embedDim, bias)
    val kProjLayer = Linear1D(inSize = kDim.getOrElse(embedDim), outSize = embedDim, bias)
    val vProjLayer = Linear1D(inSize = vDim.getOrElse(embedDim), outSize = embedDim, bias)


    val finalLinear = Linear1D(embedDim, embedDim, bias)

    def apply(query: TensorR, key: TensorR, value: TensorR, attnMask: Option[Rep[Array[Int]]] = None) = {
      // attnMask should be in the correct device (i.e. GPU Array if GPU, o.w. CPU)
      // expected shape L (seqlen) x N (batchsize) x V (embedding size)
      val batchSize = query.x.shape(1)
      val srcLen = key.x.shape(0)
      val tgtLen = query.x.shape(0)

      // TODO - can optimize when kProj = vProj etc. by combining weights and doing a single matmul (but need a tensor chunk op next)
      val qProj = qProjLayer(query.resizeNoCheck(tgtLen * batchSize, query.x.shape(2))).resizeNoCheck(tgtLen, batchSize * numHeads, headDim)
      val kProj = kProjLayer(key.resizeNoCheck(srcLen * batchSize, key.x.shape(2))).resizeNoCheck(srcLen, batchSize * numHeads, headDim)
      val vProj = vProjLayer(value.resizeNoCheck(srcLen * batchSize, value.x.shape(2))).resizeNoCheck(srcLen, batchSize * numHeads, headDim)

      // scaling Q
      val scaling: Rep[Float] = 1 / Math.sqrt(headDim).toFloat
      val q1 = qProj * scaling

//      val q2 = q1.permute(1, 0, 2)
//      val k2 = kProj.permute(1, 2, 0) // transposed
//      val v2 = vProj.permute(1, 0, 2)

      // output shape - tgtLen, batchSize * numHeads, srcLen
      val attnWeightsRaw = q1.bmm(kProj, batchDim = 1, transX = false, transY = true)
      val attnWeightsMasked = attnMask match {
        case Some(mask) => attnWeightsRaw.maskedFill(mask, Float.MinValue, dim0=0, dim1=2)
        case _ => attnWeightsRaw
      }

      val attnWeights = attnWeightsMasked.softmax_v2()
      val attnWeightsAfterDropout = if (dropout == 0) attnWeights else attnWeights.dropout_v2(dropout)

      // output after below bmm - tgtLen, batchSize * numHeads, headDim
      val attnOut = attnWeightsAfterDropout.bmm(vProj, batchDim = 1, transX = false, transY = false).resizeNoCheck(tgtLen * batchSize, embedDim)
      finalLinear(attnOut).resizeNoCheck(tgtLen, batchSize, embedDim)
    }

//    def apply(qkv: TensorR, attnMask: Boolean) = ??? // self attention

//    def apply(q: TensorR, kv: TensorR, attnMask: Boolean) = ??? // encoder-decoder attention
  }

  /** MultiheadAttention
   *
   * @param embedDim Vector Embedding size of Query. All Q, K, V are projected to embedDim. Also, the final output size.
   * @param numHeads Number of attention heads
   * @param kDim (initial) Vector Embedding size of Key
   * @param vDim (initial) Vector Embedding size of Value
   * @param bias whether to use bias or not in projections
   * @param dropOut dropout rate
   * @param residualConnection whether to use the residual connection (see original paper)
   */

  case class MultiheadAttention(val embedDim: Int, val numHeads: Int, kDim: Int, vDim: Int,
                                val bias: Boolean = false, val dropOut: Float = 0.0f, residualConnection: Boolean = false,
                                maxSeqLenQ: Rep[Int], maxSeqLenK: Rep[Int], maxBatchSize: Rep[Int], maxBeamSize: Rep[Int],
                                val name: String = "MultiHeadAttn") extends Module {
    // embedDim should be divisible by numHeads
    // embedDim is the final output size
    val config: MultiheadAttnConfig = backend.multiheadAttentionInit(embedDim, numHeads, kDim, vDim, bias, dropOut, residualConnection, maxSeqLenQ, maxSeqLenK, maxBatchSize, maxBeamSize)

    // weights of attention model
    val sizeWeights: Int = embedDim * kDim + embedDim * vDim + embedDim * embedDim + {
      if (bias) embedDim * 3 else 0
    }
    val weights: TensorR = TensorR(Tensor.rand(sizeWeights))
    val finalLinear = Linear1D(inSize = embedDim, outSize = embedDim, bias=bias)

    def apply(query: TensorR, key: TensorR, value: TensorR, attnMask: Boolean = false) = {
      // Assumes shape = [T(time) N(batch) B(beamsize) V(vector-embed)]
      // TODO - take attn_mask (i.e. loWinIdx, hiWinIdx) as arg (intead of boolean)
      // set up attention window
//      val loWinIdx = NewArray[Int](query.x.shape(0))
//      val hiWinIdx = NewArray[Int](query.x.shape(0))
//      val qSeqArray = NewArray[Int](query.x.shape(1))
//      val kSeqArray = NewArray[Int](key.x.shape(1))
//
//      for (i <- 0 until query.x.shape(0)) {
//        loWinIdx(i) = 0
//        if (attnMask)
//          hiWinIdx(i) = i + 1
//        else
//          hiWinIdx(i) = key.x.shape(0)
//      }
//
//      for (i <- 0 until query.x.shape(1)) {
//        qSeqArray(i) = query.x.shape(0)
//        kSeqArray(i) = key.x.shape(0)
//      }

      val step1 = query.multiheadAttention(key, value, weights, attnMask, config)
      finalLinear(step1.resizeNoCheck(query.x.shape(0) * query.x.shape(1) * query.x.shape(2), embedDim)).resizeNoCheck(query.x.shape(0), query.x.shape(1), query.x.shape(2), embedDim)
    }
  }

//  case class LayerNorm(dim_size: Int, epsilon: Float = 0.00005, featureDim: Int = 2, name: String = "Layer Norm") extends Module {
//    // performs layer norm on the last dimension
//    val weights = TensorR(Tensor.ones(dim_size))
//    val bias = TensorR(Tensor.zeros(dim_size))
//
//    def apply(input: TensorR) = {
//      val mean = (input.sum(featureDim) / dim_size).resizeNoCheck(input.x.shape(0), input.x.shape(1), 1) // TODO - assumes 3d Tensor
//      val mean_squared = mean * mean
//      val squared = input * input
//      val squared_mean = (squared.sum(featureDim) / dim_size).resizeNoCheck(input.x.shape(0), input.x.shape(1), 1) // TODO - assumes 3d Tensor
//
//      val variance = (squared_mean - mean_squared + epsilon).sqrt()
//      val normalized = (input - mean) / variance
//
//      normalized * weights + bias
//    }
//  }

  case class LayerNorm(dim_size: Int, eps: Float = 0.00001, name: String = "Layer Norm") extends Module {
    // performs layer norm on the last dimension
    val gamma = TensorR(Tensor.ones(dim_size))
    val beta = TensorR(Tensor.zeros(dim_size))

    def apply(input: TensorR) = {
      input.layerNorm(eps, gamma, beta)
    }
  }

  case class TransformerEncoderLayer(embedDim: Int, nheads: Int, dimFeedForward: Int, dropOut: Float = 0.0f,
                                     name: String = "transformer-encoder-layer") extends Module {
//    val enMHA = MultiheadAttention(embedDim, nheads, embedDim, embedDim, bias = true, dropOut, residualConnection = true,
//      maxSeqLen, maxSeqLen, maxBatchSize, maxBeamSize)
    val enMHA = MultiheadAttention_v2(embedDim, nheads, dropOut, bias = true)
    val enLinear1 = Linear1D(inSize = embedDim, outSize = dimFeedForward)
    val enLinear2 = Linear1D(inSize = dimFeedForward, outSize = embedDim)
    val enLayerNorm1 = LayerNorm(embedDim)
    val enLayerNorm2 = LayerNorm(embedDim)

    def apply(src: TensorR, attnMask: Option[Rep[Array[Int]]] = None) = {
      val step1 = enMHA(src, src, src, attnMask)
      val step2 = src + step1.dropout_v2(dropOut)
      val step3 = enLayerNorm1(step2)
      val step4 = enLinear1(step3.resizeNoCheck(src.x.shape(0)*src.x.shape(1), embedDim))
      val step5 = step4.relu_v2().dropout_v2(dropOut) // TODO - can fuse
      val step6 = enLinear2(step5).resizeNoCheck(src.x.shape: _*)
      val step7 = step6.dropout_v2(dropOut) + step3
      enLayerNorm2(step7)
    }
  }

  case class TransformerDecoderLayer(embedDim: Int, nheads: Int, dimFeedForward: Int, dropOut: Float = 0.0f,
                                     name: String = "transformer-decoder-layer") extends Module {
//    val deMHA1 = MultiheadAttention(embedDim, nheads, embedDim, embedDim, bias = true, dropOut, residualConnection = true,
//      maxSeqLen, maxSeqLen, maxBatchSize, maxBeamSize)
//    val deMHA2 = MultiheadAttention(embedDim, nheads, embedDim, embedDim, bias = true, dropOut, residualConnection = true,
//      maxSeqLen, maxSeqLen, maxBatchSize, maxBeamSize)
    val deMHA1 = MultiheadAttention_v2(embedDim, nheads, dropOut, bias = true)
    val deMHA2 = MultiheadAttention_v2(embedDim, nheads, dropOut, bias = true)
    val deLinear1 = Linear1D(inSize = embedDim, outSize = dimFeedForward)
    val deLinear2 = Linear1D(inSize = dimFeedForward, outSize = embedDim)
    val deLayerNorm1 = LayerNorm(embedDim)
    val deLayerNorm2 = LayerNorm(embedDim)
    val deLayerNorm3 = LayerNorm(embedDim)

    def apply(tgt: TensorR, memory: TensorR, attnMask: Option[Rep[Array[Int]]] = None) = {
      val step1 = deMHA1(tgt, tgt, tgt, attnMask)
      val step2 = tgt + step1.dropout_v2(dropOut)
      val step3 = deLayerNorm1(step2)
      val step4 = deMHA2(step3, memory, memory, attnMask)
      val step5 = step4.dropout_v2(dropOut) + step3
      val step6 = deLayerNorm2(step5)
      val step7 = deLinear1(step6.resizeNoCheck(tgt.x.shape(0)*tgt.x.shape(1), embedDim))
      val step8 = step7.relu_v2().dropout_v2(dropOut) // TODO - can easily fuse relu + dropout
      val step9 = deLinear2(step8).resizeNoCheck(tgt.x.shape: _*)
      val step10 = step6 + step9.dropout_v2(dropOut)
      deLayerNorm3(step10)
    }
  }

  case class TransformerEncoder(embedDim: Int, nheads: Int, dimFeedForward: Int, dropOut: Float = 0.0f, numLayers: Int = 1,
                                name: String = "transformer-encoder") extends Module {
    val layers = (0 until numLayers: Range) map (_ => TransformerEncoderLayer(embedDim, nheads, dimFeedForward, dropOut))

    val encoderNorm = LayerNorm(embedDim)

    def apply(src: TensorR, attnMask: Option[Rep[Array[Int]]] = None) = {
      @scala.annotation.tailrec
      def stack(prev: => TensorR @diff, i: Int = 0): TensorR @diff = {
        if (i==numLayers)
          prev
        else
          stack(layers(i)(prev, attnMask), i + 1)
      }

      val out = stack(layers(0)(src, attnMask), 1)
      encoderNorm(out)
    }
  }

  case class TransformerDecoder(embedDim: Int, nheads: Int, dimFeedForward: Int, dropOut: Float = 0.0f, numLayers: Int = 1,
                                name: String = "transformer-decoder") extends Module {
    val layers = (0 until numLayers : Range) map (_ => TransformerDecoderLayer(embedDim, nheads, dimFeedForward, dropOut))

    val decoderNorm = LayerNorm(embedDim)

    def apply(tgt: TensorR, memory: TensorR, attnMask: Option[Rep[Array[Int]]] = None) = {
      @scala.annotation.tailrec
      def stack(prev: => TensorR @diff, i: Int = 0): TensorR @diff = {
        if (i==numLayers)
          prev
        else
          stack(layers(i)(prev, memory, attnMask), i + 1)
      }

      val out = stack(layers(0)(tgt, memory, attnMask), 1)
      decoderNorm(out)
    }
  }

  // TODO - remove seqLen (no longer required; used in the previous implementation)
  case class Transformer(embedDim: Int, seqLen: Int, nheads: Int = 8, numEncoderLayers: Int = 6,
                         numDecoderLayers: Int = 6, dimFeedForward: Int = 2048, dropOut: Float = 0.0f,
                         name: String = "transformer") extends Module {
    // val blocks = (0 until numBlocks: Range) map (_ => TransformerBlock(embedDim, nheads, dimFeedForward, dropOut))
    val encoderStack = TransformerEncoder(embedDim, nheads, dimFeedForward, dropOut, numEncoderLayers)
    val decoderStack = TransformerDecoder(embedDim, nheads, dimFeedForward, dropOut, numDecoderLayers)

    def apply(src: TensorR, tgt: TensorR, maskGPU: Option[Rep[Array[Int]]] = None) = {
      val encoderOut = encoderStack(src, attnMask = None)
      val decoderOut = decoderStack(tgt, encoderOut, attnMask = maskGPU)
      decoderOut
    }
  }

  case class Embedding(numEmbeddings: Int, embeddingDim: Int, paddingIdx: Int = -1, name: String = "embedding") extends Module {
    val weights = TensorR(Tensor.rand(numEmbeddings, embeddingDim))
    // TODO - padding idx is not supported yet (padding embedding should be all zeros)

    def apply(indices: Rep[Array[Int]], indices_shape: Seq[Rep[Int]]) = {
      // TODO - indices array must be on the correct device
      weights.embedding(indices, indices_shape)
    }
  }

  abstract class Optim {
    val module: Module
    module.registerParameters(s"${module.name}/")
    def step_func: (TensorR, Option[Tensor]) => Unit
    def zero_grad() = module.forEachParameter(_.clear_grad())
    def step() = module.forEachPairParameter(step_func)
    def show() = module.forEachNamedParameter { case (name, (tr, ot)) => tr.d.printHead(5, name) }
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

  case class SGD_Momentum(val module: Module, val learning_rate: Float, val momentum: Float = 0.9f, val gradClip: Float = 400.0f, val nesterov: Boolean = false, val descent: Boolean = true) extends Optim {
    module.enrichParameter()
    @virtualize
    def step_func = { case (tr, Some(t)) =>
      backend.momentum_update(tr, t, learning_rate, momentum, gradClip, nesterov, descent)
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

  def RNNRelu(inputSize: Int, hiddenSize: Int, numLayers: Int = 1,
              dropout: Float = 0f, bidirectional: Boolean = false, name: String = "rnn_relu") =
    RNNBase(RnnReluMode, inputSize, hiddenSize, numLayers, dropout, bidirectional, name)
 
  def RNNTanh(inputSize: Int, hiddenSize: Int, numLayers: Int = 1,
              dropout: Float = 0f, bidirectional: Boolean = false, name: String = "rnn_tanh") =
    RNNBase(RnnTanhMode, inputSize, hiddenSize, numLayers, dropout, bidirectional, name)
 
  def LSTM(inputSize: Int, hiddenSize: Int, numLayers: Int = 1,
           dropout: Float = 0f, bidirectional: Boolean = false, name: String = "lstm") =
    RNNBase(LstmMode, inputSize, hiddenSize, numLayers, dropout, bidirectional, name)

  def GRU(inputSize: Int, hiddenSize: Int, numLayers: Int = 1,
          dropout: Float = 0f, bidirectional: Boolean = false, name: String = "gru") =
    RNNBase(GruMode, inputSize, hiddenSize, numLayers, dropout, bidirectional, name)

  case class RNNBase (val mode: RnnMode, val inputSize: Int, val hiddenSize: Int, val numLayers: Int = 1,
    val dropout: Float = 0f, val bidirectional: Boolean = false, val name: String = "rnn") extends Module {

    assert(inputSize >= 1, "Input size must be at least 1")
    assert(hiddenSize >= 1, "Hidden size must be at least 1")
    assert(numLayers >= 1, "Number of layers must be at least 1")
    assert(dropout >= 0 && dropout <= 1, "Dropout must be between 0 and 1")
    
    val w_ih = ArrayBuffer[TensorR]()
    val w_hh = ArrayBuffer[TensorR]()
    val b_ih = ArrayBuffer[TensorR]()
    val b_hh = ArrayBuffer[TensorR]()

    // Reverse parameters for bidirectional RNNs.
    val w_ih_reverse = ArrayBuffer[TensorR]()
    val w_hh_reverse = ArrayBuffer[TensorR]()
    val b_ih_reverse = ArrayBuffer[TensorR]()
    val b_hh_reverse = ArrayBuffer[TensorR]()

    val numDirections = if (bidirectional) 2 else 1
    val gateSize = hiddenSize * mode.numGates

    def getParameterSize(): Int = {
      val w_ih_size = gateSize * inputSize + (numLayers - 1) * gateSize * hiddenSize // * numDirections
      val w_hh_size = numLayers * gateSize * hiddenSize
      val b_ih_size = numLayers * gateSize
      val b_hh_size = numLayers * gateSize
      val total = w_ih_size + w_hh_size + b_ih_size + b_hh_size
      if (bidirectional) total * 2 else total
    }

    // Initialize parameter buffer.
    // cuDNN requires that all parameters are stored in a contiguous buffer.
    //    val parameterBuffer = Nonparameter(TensorR(Tensor.fill(Seq(getParameterSize()), 0.01f)))
    val parameterBuffer = Nonparameter(TensorR(Tensor.rand(Seq(getParameterSize()), 0.01f)))

    def setupParameters(): Unit = {
      // Helper function that computes offsets for individual parameter tensors.
      var offset = var_new(0)
      def getParameter(dims: Int*): TensorR = {
        val x = Tensor(slice(parameterBuffer.x.data, offset), dims: _*)
        val d = Tensor(slice(parameterBuffer.d.data, offset), dims: _*)
        offset += dims.product
        new TensorR(x, d)
      }

      for (layer <- (0 until numLayers): Range) {
        val layerInputSize = if (layer == 0) inputSize else hiddenSize
        w_ih += getParameter(gateSize, layerInputSize)
        w_hh += getParameter(gateSize, hiddenSize)
        b_ih += getParameter(gateSize)
        b_hh += getParameter(gateSize)
        if (bidirectional) {
          w_ih_reverse += getParameter(gateSize, layerInputSize)
          w_hh_reverse += getParameter(gateSize, hiddenSize)
          b_ih_reverse += getParameter(gateSize)
          b_hh_reverse += getParameter(gateSize)
        }
      }
    }
    setupParameters()

    def apply(input: TensorR, hx: Option[TensorR] = None, cx: Option[TensorR] = None): TensorR @diff = shift { (k: TensorR => Unit) =>
      assert(input.x.rank == 3, "RNN input should have rank 3: [seqLength x batchSize x inputSize]")
      val seqLength = input.x.shape(0)
      val batchSize = input.x.shape(1)
      val inputSize = input.x.shape(2)

      hx match {
        case None =>
        case Some(hx) =>
          assert(hx.x.rank == 3, "RNN hidden state should have rank 3: [numLayers x batchSize x hiddenSize]")
          assert(batchSize == hx.x.shape(1), "RNN hidden state second dimension should equal input second dimension (batch size)")
      }

      val (y, hy, reserve, reserveSize, counter) = BackendCudnn().cudnnRNNForwardTraining(
        mode, input.x, hx.map(_.x), cx.map(_.x), parameterBuffer.x,
        numLayers, hiddenSize, dropout, bidirectional)
      val output = TensorR(y)
      k(output)

      BackendCudnn().cudnnRNNBackward(
        mode, input, hx.map(_.x), cx.map(_.x), parameterBuffer, output,
        numLayers, hiddenSize, dropout, bidirectional, reserve, reserveSize, counter)
    }
  }
}

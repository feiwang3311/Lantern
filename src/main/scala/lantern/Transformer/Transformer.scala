package lantern

import lms.core.stub._
import lms.macros.SourceContext
import lms.core.virtualize

import scala.sys.process._

import java.io.PrintWriter;
import java.io.File;

object Transformer {

  val driver = new LanternDriverCudnn[String, Unit] with ScannerOpsExp with TimerOpsExp {
    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {
      case class MultiheadAttention(val embedDim: Int, val numHeads: Int, kDim: Int, vDim: Int,
                                    val bias: Boolean = false, val dropOut: Float = 0.0f, residuals: Boolean = false,
                                    val name: String = "MultiHeadAttn") extends Module {
        // note - pytorch: all q, k, v is transormed to embedDim size. embedDim = numHeads * h_i(size).
        // Therefore embedDim should be divisible by numHeads
        // embedDim is the final output size

        // weights of attention model
        // TODO - would be better if we can get the sizeWeights from cudnn side
        val sizeWeights = embedDim * kDim + embedDim * kDim + embedDim * vDim + {
          if (bias) 0 else embedDim * 3
        }
        val weights: TensorR = TensorR(Tensor.rand(sizeWeights)).toGPU()
        val finalLinear = Linear1D(inSize = embedDim, outSize = embedDim)


        def apply(query: TensorR, key: TensorR, value: TensorR, attnMask: Boolean = false) = {
          // [T(time) N(batch) B(beamsize) V(vector-embed)]
          // TODO - take attn_mask (i.e. loWinIdx, hiWinIdx) as arg
          // assert rank 4
          // set up attention window without mask
          val loWinIdx = NewArray[Int](query.x.shape(1))
          val hiWinIdx = NewArray[Int](query.x.shape(1))
          val qSeqArray = NewArray[Int](query.x.shape(1))
          val kSeqArray = NewArray[Int](key.x.shape(1))

          for (i <- 0 until query.x.shape(1)) {
            loWinIdx(i) = 0
            if (attnMask)
              hiWinIdx(i) = i + 1 // end idx is exclusive
            else
              hiWinIdx(i) = query.x.shape(0)
            qSeqArray(i) = query.x.shape(0)
            kSeqArray(i) = key.x.shape(0)
          }

          val step1 = query.multiheadAttention(key, value, weights, numHeads, embedDim, qSeqArray, kSeqArray, loWinIdx, hiWinIdx, bias, dropOut, 1.0f, residuals)
          finalLinear(step1.resize(-1, embedDim)).resize(query.x.shape(0), query.x.shape(1), query.x.shape(2), embedDim)
        }
      }

      case class LayerNorm(dim_size: Int, epsilon: Float = 0.00005, featureDim: Int = 3, name: String = "Layer Norm") extends Module {
        // performs layer norm on the last dimension
        val weights = TensorR(Tensor.ones(dim_size))
        val bias = TensorR(Tensor.zeros(dim_size))

        def apply(input: TensorR) = {
          val mean = (input.sum(featureDim) / dim_size).resize(input.x.shape(0), input.x.shape(1),input.x.shape(2), 1)
          val mean_squared = mean * mean
          val squared = input * input
          val squared_mean = (squared.sum(featureDim) / dim_size).resize(input.x.shape(0), input.x.shape(1),input.x.shape(2), 1)

          val variance = (squared_mean - mean_squared + epsilon).sqrt()
          val normalized = (input - mean) / variance

          normalized * weights + bias
        }
      }

      case class TransformerEncoderLayer(embedDim: Int, nheads: Int, dimFeedForward: Int, dropOut: Float = 0.0f,
                                         name: String = "transformer-encoder-layer") extends Module {
        val enMHA = MultiheadAttention(embedDim, nheads, embedDim, embedDim, bias = true, dropOut, residuals = true)
        val enLinear1 = Linear1D(inSize = embedDim, outSize = dimFeedForward)
        val enLinear2 = Linear1D(inSize = dimFeedForward, outSize = embedDim)
        val enLayerNorm1 = LayerNorm(embedDim)
        val enLayerNorm2 = LayerNorm(embedDim)

        def apply(src: TensorR, attnMask: Boolean = false) = {
          val step1 = enMHA(src, src, src)
          val step2 = enLayerNorm1(step1)
          val step3 = enLinear1(step2.resize(-1, embedDim))
          val step4 = step3.relu()
          val step5 = enLinear2(step4).resize(src.x.shape: _*)
          val step6 = step5 + step2
          enLayerNorm2(step6)
        }
      }

      case class TransformerDecoderLayer(embedDim: Int, nheads: Int, dimFeedForward: Int, dropOut: Float = 0.0f,
                                         name: String = "transformer-decoder-layer") extends Module {
        val deMHA1 = MultiheadAttention(embedDim, nheads, embedDim, embedDim, bias = true, dropOut, residuals = true)
        val deMHA2 = MultiheadAttention(embedDim, nheads, embedDim, embedDim, bias = true, dropOut, residuals = true)
        val deLinear1 = Linear1D(inSize = embedDim, outSize = dimFeedForward)
        val deLinear2 = Linear1D(inSize = dimFeedForward, outSize = embedDim)
        val deLayerNorm1 = LayerNorm(embedDim)
        val deLayerNorm2 = LayerNorm(embedDim)
        val deLayerNorm3 = LayerNorm(embedDim)

        def apply(tgt: TensorR, memory: TensorR, attnMask: Boolean = false) = {
          val step1 = deMHA1(tgt, tgt, tgt)
          val step2 = deLayerNorm1(step1)
          val step3 = deMHA2(step2, memory, memory)
          val step4 = deLayerNorm2(step3)
          val step5 = deLinear1(step4.resize(-1, embedDim))
          val step6 = step5.relu()
          val step7 = deLinear2(step6).resize(tgt.x.shape: _*)
          val step8 = step7 + step4
          deLayerNorm3(step8)
        }
      }

      case class TransformerEncoder(embedDim: Int, nheads: Int, dimFeedForward: Int, dropOut: Float = 0.0f,
                                    numLayers: Int = 1, name: String = "transformer-encoder") extends Module {
        // TODO - initialize numLayers encoderLayers (below is hardcoded)
        val layer1 = TransformerEncoderLayer(embedDim, nheads, dimFeedForward, dropOut)
        val layer2 = TransformerEncoderLayer(embedDim, nheads, dimFeedForward, dropOut)
        val layer3 = TransformerEncoderLayer(embedDim, nheads, dimFeedForward, dropOut)
        val layer4 = TransformerEncoderLayer(embedDim, nheads, dimFeedForward, dropOut)

        // TODO - check this with the original paper
        val encoderNorm = LayerNorm(embedDim)

        def apply(src: TensorR, attnMask: Boolean = false) = {
          val step1 = layer1(src, attnMask)
          val step2 = layer2(src, attnMask)
          val step3 = layer3(src, attnMask)
          val step4 = layer4(src, attnMask)
          encoderNorm(step4)
        }
      }

      case class TransformerDecoder(embedDim: Int, nheads: Int, dimFeedForward: Int, dropOut: Float = 0.0f,
                                    numLayers: Int = 1, name: String = "transformer-decoder") extends Module {
        // TODO - initialize numLayers encoderLayers (below is hardcoded)
        val layer1 = TransformerDecoderLayer(embedDim, nheads, dimFeedForward, dropOut)
        val layer2 = TransformerDecoderLayer(embedDim, nheads, dimFeedForward, dropOut)
        val layer3 = TransformerDecoderLayer(embedDim, nheads, dimFeedForward, dropOut)
        val layer4 = TransformerDecoderLayer(embedDim, nheads, dimFeedForward, dropOut)

        // TODO - check this with the original paper
        val decoderNorm = LayerNorm(embedDim)

        def apply(tgt: TensorR, memory: TensorR, attnMask: Boolean = true) = {
          val step1 = layer1(tgt, memory, attnMask)
          val step2 = layer2(tgt, memory, attnMask)
          val step3 = layer3(tgt, memory, attnMask)
          val step4 = layer4(tgt, memory, attnMask)
          decoderNorm(step4)
        }
      }

      case class Transformer(embedDim: Int, seqLen: Int, nheads: Int = 8, numEncoderLayers: Int = 6,
                             numDecoderLayers: Int = 6, dimFeedForward: Int = 2048, dropOut: Float = 0.0f,
                             name: String = "transformer") extends Module {
        val finalLinear = Linear1D(inSize = embedDim * seqLen, outSize = 1)
        // val blocks = (0 until numBlocks: Range) map (_ => TransformerBlock(embedDim, nheads, dimFeedForward, dropOut))
        val encoderStack = TransformerEncoder(embedDim, nheads, dimFeedForward, dropOut, numEncoderLayers)
        val decoderStack = TransformerDecoder(embedDim, nheads, dimFeedForward, dropOut, numEncoderLayers)

        def apply(src: TensorR, tgt: TensorR) = {
          val encoderOut = encoderStack(src, attnMask = false)
          val decoderOut = decoderStack(tgt, encoderOut, attnMask = true)

          // calculate the final output layer using decoder output
          finalLinear(decoderOut.permute(1, 2, 0, 3).resize(-1, embedDim * seqLen)) // this is a dummy final layer to produce a single output value
        }
      }

      // model
      // requirements: qsize = ksize and vsize * numHeads = embedDim
      //             val qsize = 500
      //             val ksize = 50
      //             val vsize = 50
      //             val embedDim = 500
      //             val numHeads = 5
      //             val batchSize = 10
      //             val beamSize = 1
      //             val seqLen = 500 // both klen and qlen
      //             val dropOut = 0.1f
      //
      //             case class Model(val name: String = "test_model") extends Module {
      //                 val mha = MultiheadAttention(embedDim, numHeads, ksize, vsize, true, dropOut, false, true)
      //                 val linear = Linear1D(inSize = embedDim * seqLen, outSize = 1)
      //
      //                 def apply(q: TensorR, k: TensorR, v: TensorR) = {
      //                     val step1 = mha(q, k, v)
      //                     linear(step1.permute(1, 2, 0, 3).resize(-1, q.x.shape(0) * q.x.shape(3)))
      //                 }
      //             }
      //
      //             val model = Model()
      //
      //             val q = TensorR(Tensor.rand(Seq(seqLen, batchSize, beamSize, qsize) : _*)).toGPU()
      //             val k = TensorR(Tensor.rand(Seq(seqLen, batchSize, beamSize, ksize) : _*)).toGPU()
      //             val v = TensorR(Tensor.rand(Seq(seqLen, batchSize, beamSize, vsize) : _*)).toGPU()
      //
      //             val opt = SGD(model, learning_rate = 0.0005f, gradClip = 1000.0f)
      //
      //
      //             def lossFun(query: TensorR, key: TensorR, value: TensorR) = { (batchIndex:TensorR) =>
      //             //     trainTimer.startTimer
      //                 val res = model(query, key, value)
      //                 res.sum()
      //             }
      //
      //
      //             val num_iter = 5
      //             for(i <- 0 until num_iter: Rep[Range]) {
      //                 val trainTimer = Timer2()
      //                 trainTimer.startTimer
      //                 val loss = gradR_loss(lossFun(q, k, v))(Tensor.zeros(4))
      //                 opt.step()
      //                 val delta = trainTimer.getElapsedTime
      //                 printf("Training iter done in %ldms \\n", delta/1000L)
      //                 printf("loss = %f\n", loss.toCPU().data(0))
      //             }

      val embedDim = 100
      val seqLen = 50
      val batchSize = 100
      val nheads = 4
      val numBlocks = 4
      val dimFeedForward = 200
      val dropOut = 0.1f

      val model = Transformer(embedDim, seqLen, nheads, 4, 4, dimFeedForward, dropOut)

      val src = TensorR(Tensor.rand(Seq(seqLen, batchSize, 1, embedDim): _*))
      val tgt = TensorR(Tensor.rand(Seq(seqLen, batchSize, 1, embedDim): _*))

      def lossFun(src: TensorR, tgt: TensorR) = { (batchIndex: TensorR) =>
        val res = model(src, tgt)
        res.sum()
      }

      val opt = SGD(model, learning_rate = 0.0005f, gradClip = 1.0f)

      val num_iter = 5
      for (i <- 0 until num_iter: Rep[Range]) {
        val trainTimer = Timer2()
        trainTimer.startTimer

        val loss = gradR_loss(lossFun(src, tgt))(Tensor.zeros(4))
        opt.step()
        val delta = trainTimer.getElapsedTime
        printf("Training iter done in %ldms \\n", delta / 1000L)

        printf("loss = %f\n", loss.toCPU().data(0))
      }


    }
  }

  // def compile: Int = {
  //     val io = new ProcessIO(
  //     _.close, // stdin
  //     stdout => try { Source.fromInputStream(stdout).getLines.foreach(System.out.println(_)) } finally { stdout.close },
  //     stderr => try { Source.fromInputStream(stderr).getLines.foreach(System.out.println(_)) } finally { stderr.close }
  //     )

  //     // Compile query
  //     val sources = "multihead_lantern.cu"
  //     val executableName = "lantern_out.o"
  //     val libraryFlags = Seq("-lcuda", "-lcublas", "-lcudnn") mkString(" ")
  //     val compilerFlags = Seq("-std=c++11", "--expt-extended-lambda") mkString(" ")
  //     val cmd = s"nvcc $compilerFlags $sources -o $executableName $libraryFlags"
  //     infoCompileTime(s"Compilation command: $cmd")
  //     val proc = cmd.run(io)
  //     proc.exitValue
  // }

  def main(args: Array[String]) = {
    val code_file = new PrintWriter(new File("src/out/Transformers/Lantern/transformer.cu"))
    code_file.println(driver.code)
    code_file.flush()
  }
}
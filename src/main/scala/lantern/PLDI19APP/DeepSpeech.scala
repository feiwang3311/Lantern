package lantern
package PLDI19App

import lms.core.stub._
import lms.macros.SourceContext
import lms.core.virtualize

import scala.collection.mutable.ArrayBuffer
import scala.collection.Seq
import scala.math._

import java.io.PrintWriter
import java.io.File

object DeepSpeech {

  val root_dir = "src/out/PLDI19evaluation/"
  val gpu_file_dir = "deepspeech2/lantern/Lantern.cu"
  val data_dir = "/scratch-ml00/wang603/deepspeechData/deepspeech_train.bin"

  val deepspeechGPU = new LanternDriverCudnn[String, Unit] {

    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {
      debug = false

      Random.srand(Some(42))
      val dataTimer = Timer2()
      dataTimer.startTimer

      // Reference: https://github.com/SeanNaren/deepspeech.pytorch/blob/c959d29c381e5bef7cdfb0cd420ddacd89d11520/model.py#L80
      case class BatchRNN(val name: String = "batch_rnn",
                          inputSize: Int, hiddenSize: Int, rnnMode: RnnMode = RnnTanhMode,
                          bidirectional: Boolean = true, useBatchNorm: Boolean = false, numLayers: Int = 1) extends Module {
        val rnn = RNNBase(rnnMode, inputSize, hiddenSize, numLayers = numLayers, bidirectional = bidirectional)
        val batchNorm: Option[BatchNorm1D] = if (useBatchNorm) Some(BatchNorm1D(inputSize)) else None

        def apply(input: TensorR): TensorR @diff = {
          val in1 = If (useBatchNorm) {
            val input2D = input.resizeNoCheck(input.x.shape(0) * input.x.shape(1), input.x.shape(2))
            val inputBN = batchNorm.get.apply(input2D)
            inputBN.resizeNoCheck(input.x.shape(0), input.x.shape(1), input.x.shape(2))
          } { input }
          val output = rnn(in1)
          If (bidirectional) {output.resizeNoCheck(output.x.shape(0), output.x.shape(1), 2, output.x.shape(2) / 2).sum(2)} {output}
        }
      }

      // Reference: https://github.com/SeanNaren/deepspeech.pytorch/blob/c959d29c381e5bef7cdfb0cd420ddacd89d11520/model.py#L105
      case class Lookahead(val name: String = "lookahead", numFeatures: Int, context: Int) extends Module {
        assert(context >= 1, "Context size must be at least 1")
        val weight = TensorR(Tensor.rand(Seq(numFeatures, context + 1), scala.math.sqrt(context + 1).toFloat))

        // TODO (Fei Wang): this could be optimized by a user-defined kernel?
        def apply(input: TensorR): TensorR @diff = {
          val padding = TensorR(Tensor.zeros((unit(context) +: input.x.shape.drop(1)): _*))
          val x = input.concat(0, padding)
          val xs = x.repeat0(context).permute(0, 2, 3, 1)
          (xs * weight).sum(3)
        }
      }

      // Reference: https://github.com/SeanNaren/deepspeech.pytorch/blob/c959d29c381e5bef7cdfb0cd420ddacd89d11520/model.py#L145
      case class DeepSpeech(val name: String = "deepspeech",
                            rnnMode: RnnMode = RnnTanhMode, labels: String = "abc",
                            rnnHiddenSize: Int = 1024, numLayers: Int = 3,
                            sampleRate: Int = 16000, windowSize: Float = 0.02f,
                            bidirectional: Boolean = true, context: Int = 20) extends Module with Serializable {

        assert(rnnHiddenSize >= 1, "RNN hidden size must be at least 1")
        assert(numLayers >= 1, "Number of RNN layers must be at least 1")

        val numClasses = labels.length

        val conv = new Module with Serializable {
          val name = "conv"
          val conv1 = Conv2D(1, 32, Seq(41, 11), stride = Seq(2, 2), useBias = false)
          val bn1 = BatchNorm2D(32)
          val conv2 = Conv2D(32, 32, Seq(21, 11), stride = Seq(2, 1), useBias = false)
          val bn2 = BatchNorm2D(32)
          def apply(in: TensorR): TensorR @diff = {
            // NOTE: This function assume that the lengths array is already on GPU
            val step1 = conv1(in)
            val step2 = bn1(step1).hardTanh(0, 20, inPlace = true)
            val step3 = conv2(step2)
            bn2(step3).hardTanh(0, 20, inPlace = true)
          }
        }

        val rnnInputSize: Int = {
          var tmp: Int = (floor((sampleRate * windowSize) / 2) + 1).toInt
          tmp = (floor(tmp - 41) / 2 + 1).toInt
          tmp = (floor(tmp - 21) / 2 + 1).toInt
          tmp *= 32
          tmp
        }

        printf("initial rnn input size is %d \\n", rnnInputSize)
        val rnns: Seq[BatchRNN] = for (layer <- 0 until numLayers: Range) yield {
          if (layer == 0) BatchRNN(s"batch_rnn${layer}", rnnInputSize, rnnHiddenSize, rnnMode, bidirectional, useBatchNorm = false)
          else BatchRNN(s"batch_rnn${layer}", rnnHiddenSize, rnnHiddenSize, rnnMode, bidirectional, useBatchNorm = false)
        }

        val lookahead: Option[Lookahead] = if (bidirectional) None else Some(Lookahead(numFeatures = rnnHiddenSize, context = context))

        val fc = new Module with Serializable {
          val name: String = "fully_connected"
          val bn = BatchNorm1D(rnnHiddenSize)
          val linear = Linear1D(rnnHiddenSize, numClasses, bias=false)
          def apply(in: TensorR): TensorR @diff = {
            val shape0 = in.x.shape(0)
            val shape1 = in.x.shape(1)
            val shape2 = in.x.shape(2)
            val in2D = in.resizeNoCheck(shape0 * shape1, shape2)
            val out2D = linear(bn(in2D))
            out2D.resizeNoCheck(shape0, shape1, numClasses)
          }
        }

        def apply(input: TensorR): TensorR @diff = {
          // input is B * C * D * T
          // generate_comment("before getting length info") // line 1117
          // val outputLengths = getSeqLens(lengths, input.x.shape(0))
          // val outputLengthsGPU = outputLengths.toGPU(input.x.shape(0))
          // generate_comment("after getting length info") // line 1138
          val step1 = conv(input)
          generate_comment("after conv ops")  // line 1480
          val step2 = step1.resizeNoCheck(step1.x.shape(0), step1.x.shape(1) * step1.x.shape(2), step1.x.shape(3))  // step2 is B * CD * T
          val step3 = step2.permute(2, 0, 1) // step3 is T * B * (CD)
          generate_comment("after resize and permute") // line 1576

          def rec(rnns: Seq[BatchRNN], in: TensorR): TensorR @diff = If (rnns.isEmpty) {in} {rec(rnns.tail, rnns.head(in))}
          val step4 = rec(rnns, step3)
          //generate_comment("after RNN layers")// line 8711

          val step5 = If (bidirectional) {step4} { lookahead.get.apply(step4).hardTanh(0, 20, inPlace=true) }
          //generate_comment("after maybe lookahead") // line 8450
          // TODO igore eval_mode (which needs a softmax layer) for now
          fc(step5)  // T * B * num_alphabet
        }
      }

      val labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "
      val net = DeepSpeech(labels = labels, bidirectional = true)
      // TODO: PyTorch DeepSpeech model uses SGD with Nesterov momentum.
      // val opt = SGD(net, learning_rate = 3e-8f, gradClip = 1000.0f)
      val opt = SGD_Momentum(net, learning_rate = 3e-6f, momentum = 0.9f, gradClip = 400.0f, nesterov = true)

      def lossFun(input: TensorR, percent: Rep[Array[Float]], target: Rep[Array[Int]], targetSize: Rep[Array[Int]]) = { (dummy: TensorR) =>
        val probs = net(input).softmax_batch(2)
        generate_comment("before CTC loss") // line 8572
        val outputLength = NewArray[Int](probs.x.shape(1))
        for (i <- 0 until probs.x.shape(1)) outputLength(i) = unchecked[Int]("(int)", percent(i) * probs.x.shape(0))
        val loss = probs.ctcLoss(outputLength, target, targetSize)
        generate_comment("after CTC loss") // line 8641
        TensorR(loss)
      }

      // Training
      val nbEpoch = 1

      // TODO: Replace with real data loader.
      val data = new DeepSpeechDataLoader(data_dir, true)
      val batchSize = data.batchSize

      val prepareTime = dataTimer.getElapsedTime / 1e6f
      printf("Data reading (all prepare time) in %lf sec\\n", prepareTime)

      val loss_save = NewArray[Double](nbEpoch)
      val time_save = NewArray[Double](nbEpoch)

      val addr = getMallocAddr()
      val addrCuda = getCudaMallocAddr()

      generate_comment("training loop starts here")
      for (epoch <- 0 until nbEpoch: Rep[Range]) {
        val trainTimer = Timer2()
        var imgIdx = var_new(0)
        var trainLoss = var_new(0.0f)
        printf("Start training epoch %d\\n", epoch + 1)
        trainTimer.startTimer

        data.foreachBatch { (batchIndex: Rep[Int], input: Tensor, percent: Rep[Array[Float]], target: Rep[Array[Int]], targetLength: Rep[Array[Int]]) =>

          imgIdx += batchSize
          val inputR = TensorR(input.toGPU(), isInput = true)
          val loss = gradR_loss(lossFun(inputR, percent, target, targetLength))(Tensor.zeros(1))
          trainLoss += loss.toCPU().data(0)
          opt.step()

          // selective printing
          if (imgIdx % (batchSize * 20) == 0) {
            printf(s"Train epoch %d: [%d/%d (%.0f%%)]\\tAverage Loss: %.6f\\n", epoch, imgIdx, data.length, 100.0 * imgIdx /data.length, trainLoss/imgIdx)
            unchecked[Unit]("fflush(stdout)")
          }
          resetMallocAddr(addr)
          resetCudaMallocAddr(addrCuda)
        }
        val delta = trainTimer.getElapsedTime
        printf("Training completed in %ldms (%ld us/images)\\n", delta/1000L, delta/data.length)
        time_save(epoch) = delta / 1000000L
        loss_save(epoch) = trainLoss / data.length
      }

      val totalTime = dataTimer.getElapsedTime / 1e6f
      val loopTime = totalTime - prepareTime
      val timePerEpoc = loopTime / nbEpoch

      // report median time
      unchecked[Unit]("sort(", time_save, ", ", time_save, " + ", nbEpoch, ")")
      val median_time =  time_save(nbEpoch / 2)

      val fp2 = openf(a, "w")
      fprintf(fp2, "unit: %s\\n", "1 epoch")
      for (i <- (0 until loss_save.length): Rep[Range]) {
        fprintf(fp2, "%lf\\n", loss_save(i))
      }
      fprintf(fp2, "run time: %lf %lf\\n", prepareTime, median_time)
      closef(fp2)
    }
  }

  def main(args: Array[String]) {
    val gpu_file = new PrintWriter(new File(root_dir + gpu_file_dir))
    gpu_file.println(deepspeechGPU.code)
    gpu_file.flush()
  }
}

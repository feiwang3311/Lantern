package lantern
package PLDI19App

import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

import scala.collection.mutable.ArrayBuffer
import scala.collection.Seq
import scala.math._

import java.io.PrintWriter
import java.io.File

object DeepSpeech {

  val root_dir = "src/out/PLDI19evaluation/"
  val gpu_file_dir = "deepspeech2/lantern/Lantern.cu"
  // val data_dir: String = "/u/data/u99/wang603/TiarkMlEnv/SampleData/deepspeech_train.bin"
  val data_dir: String = "/scratch/wu636/training/speech_recognition/data/test/deepspeech_train.bin"

  val deepspeechGPU = new LanternDriverCudnn[String, Unit] {

    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {
      Random.srand(Some(42))
      val dataTimer = Timer2()
      dataTimer.startTimer

      // Reference: https://github.com/SeanNaren/deepspeech.pytorch/blob/c959d29c381e5bef7cdfb0cd420ddacd89d11520/model.py#L80
      case class BatchRNN(val name: String = "batch_rnn",
                          inputSize: Int, hiddenSize: Int, rnnMode: RnnMode = LstmMode,
                          bidirectional: Boolean = false, useBatchNorm: Boolean = true) extends Module {
        val rnn = RNNBase(rnnMode, inputSize, hiddenSize, bidirectional = bidirectional)
        val batchNorm: Option[BatchNorm1D] = if (useBatchNorm) Some(BatchNorm1D(inputSize)) else None

        // we don't actually use outputLengths here. The pytorch imp needs it for pack_padded_sequence and pad_packed_sequence
        def apply(input: TensorR, outputLengths: Rep[Array[Int]]): TensorR @diff = {
          val in1 = IF (useBatchNorm) {
            val input2D = input.resize(input.x.shape(0) * input.x.shape(1), input.x.shape(2))
            val inputBN = batchNorm.get.apply(input2D)
            inputBN.resize(input.x.shape(0), input.x.shape(1), input.x.shape(2))
          } { input }
          val output = rnn(in1)
          IF (bidirectional) {output.resize(output.x.shape(0), output.x.shape(1), 2, -1).sum(2)} {output}
        }
      }

      // Reference: https://github.com/SeanNaren/deepspeech.pytorch/blob/c959d29c381e5bef7cdfb0cd420ddacd89d11520/model.py#L105
      case class Lookahead(val name: String = "lookahead", numFeatures: Int, context: Int) extends Module {
        assert(context >= 1, "Context size must be at least 1")
        val weight = TensorR(Tensor.rand(Seq(numFeatures, context + 1), scala.math.sqrt(context + 1).toFloat))

        // TODO (Fei Wang): this could be optimized by a user-defined kernel?
        def apply(input: TensorR): TensorR @diff = {
          val padding = TensorR(Tensor.zeros((unit(context) +: input.x.shape.drop(1)): _*))
          val xt = input.resize((unit(1) +: input.x.shape): _*).concat(1, padding.resize((unit(1) +: padding.x.shape): _*))
          val x = xt.resize(input.x.shape(0) + padding.x.shape(0), input.x.shape(1), input.x.shape(2))
          // val x = input.concat(0, padding)
          val xs = x.repeat0(context).permute(0, 2, 3, 1)
          (xs mul_sub weight).sum(3)
        }
      }

      // Reference: https://github.com/SeanNaren/deepspeech.pytorch/blob/c959d29c381e5bef7cdfb0cd420ddacd89d11520/model.py#L145
      case class DeepSpeech(val name: String = "deepspeech",
                            rnnMode: RnnMode = LstmMode, labels: String = "abc",
                            rnnHiddenSize: Int = 768, numLayers: Int = 5,
                            sampleRate: Int = 16000, windowSize: Float = 0.02f,
                            bidirectional: Boolean = true, context: Int = 20) extends Module with Serializable {

        assert(rnnHiddenSize >= 1, "RNN hidden size must be at least 1")
        assert(numLayers >= 1, "Number of RNN layers must be at least 1")

        val numClasses = labels.length

        val conv = new Module with Serializable {
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

        val rnnInputSize: Int = {
          var tmp: Int = (floor((sampleRate * windowSize) / 2) + 1).toInt
          tmp = (floor(tmp + 2 * 20 - 41) / 2 + 1).toInt
          tmp = (floor(tmp + 2 * 10 - 21) / 2 + 1).toInt
          tmp *= 32
          tmp
        }

        val rnns: Seq[BatchRNN] = for (layer <- 0 until numLayers: Range) yield {
          if (layer == 0) BatchRNN(s"batch_rnn${layer}", rnnInputSize, rnnHiddenSize, rnnMode, bidirectional, useBatchNorm = false)
          else BatchRNN(s"batch_rnn${layer}", rnnHiddenSize, rnnHiddenSize, rnnMode, bidirectional)
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
            val in2D = in.resize(shape0 * shape1, shape2)
            val out2D = linear(bn(in2D))
            out2D.resize(shape0, shape1, numClasses)
          }
        }

        def getSeqLens(lengths: Rep[Array[Int]], size: Rep[Int]) = {
          conv.modules.foldLeft(lengths) { case(ls, (_, m)) =>
            if (m.isInstanceOf[Conv2D]) {
              val mm = m.asInstanceOf[Conv2D]
              val ls_next = NewArray[Int](size)
              for (i <- 0 until size) ls_next(i) = (ls(i) + 2 * mm.pad(1) - mm.dilation(1) * (mm.kernelSize(1) - 1) - 1) / mm.stride(1) + 1
              // ls.map(x => (x + 2 * mm.pad(1) - mm.dilation(1) * (mm.kernelSize(1) - 1) - 1) / mm.stride(1) + 1)
              ls_next } else ls
          }
        }

        def apply(input: TensorR, lengths: Rep[Array[Int]]): (TensorR, Rep[Array[Int]]) @diff = {
          // input is B * C * D * T
          generateRawComment("before getting length info") // line 1117
          val outputLengths = getSeqLens(lengths, input.x.shape(0))
          val outputLengthsGPU = outputLengths.toGPU(input.x.shape(0))
          generateRawComment("after getting length info") // line 1138
          val step1 = conv(input, outputLengthsGPU)  // TODO (Fei Wang): this is a potential error for the pytorch implementation
          generateRawComment("after conv ops")  // line 1480
          val step2 = step1.resize(step1.x.shape(0), step1.x.shape(1) * step1.x.shape(2), step1.x.shape(3))  // step2 is B * CD * T
          val step3 = step2.permute(2, 0, 1) // step3 is T * B * (CD)
          generateRawComment("after resize and permute") // line 1576

          def rec(rnns: Seq[BatchRNN], in: TensorR): TensorR @diff = IF (rnns.isEmpty) {in} {rec(rnns.tail, rnns.head(in, outputLengthsGPU))}
          val step4 = rec(rnns, step3)
          generateRawComment("after RNN layers")// line 8711

          val step5 = IF (bidirectional) {step4} { lookahead.get.apply(step4).hardTanh(0, 20, inPlace=true) }
          generateRawComment("after bidirectional sum") // line 8450
          // TODO igore eval_mode (which needs a softmax layer) for now
          (fc(step5), outputLengthsGPU)  // B * T * num_alphabet
        }
      }

      val labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "
      val net = DeepSpeech(labels = labels, bidirectional = false)
      net.registerParameters(s"${net.name}/")
      // TODO: PyTorch DeepSpeech model uses SGD with Nesterov momentum.
      val opt = SGD(net, learning_rate = 3e-8f, gradClip = 1000.0f)
      // val opt = SGD_Momentum(net, learning_rate = 3e-4f, momentum = 0.9f, gradClip = 400.0f, nesterov = true)

      def lossFun(input: TensorR, inputLengths: Rep[Array[Int]], target: Rep[Array[Int]], targetSize: Rep[Array[Int]]) = { (dummy: TensorR) =>
        val (probs, outputLength) = net(input, inputLengths)
        val probs1 = probs.softmax_batch(2)
        generateRawComment("before CTC loss")// line 8572
        val loss = probs1.ctcLoss(outputLength.toCPU(input.x.shape(0)), target, targetSize)
        generateRawComment("after CTC loss")// line 8641
        TensorR(loss)
      }

      // Training
      val nbEpoch = 10

      // TODO: Replace with real data loader.
      val data = new Dataset.DeepSpeechDataLoader(data_dir, true)
      val batchSize = data.batchSize

      val prepareTime = dataTimer.getElapsedTime / 1e6f
      printf("Data reading (all prepare time) in %lf sec\\n", prepareTime)

      val loss_save = NewArray[Double](nbEpoch)

      val addr = getMallocAddr()
      val addrCuda = getCudaMallocAddr()

      generateRawComment("training loop starts here")
      for (epoch <- 0 until nbEpoch: Rep[Range]) {
        val trainTimer = Timer2()
        var imgIdx = var_new(0)
        var trainLoss = var_new(0.0f)
        printf("Start training epoch %d\\n", epoch + 1)
        trainTimer.startTimer

        data.foreachBatch { (batchIndex: Rep[Int], input: Tensor, inputLength: Rep[Array[Int]], target: Rep[Array[Int]], targetLength: Rep[Array[Int]]) =>

          imgIdx += batchSize
          val inputR = TensorR(input.toGPU(), isInput = true)
          val loss = gradR_loss(lossFun(inputR, inputLength, target, targetLength))(Tensor.zeros(1))
          trainLoss += loss.data(0)
          // opt.perform{case (name, (tr, ot)) => tr.d.toCPU().printHead(5, name)}
          // error("stop")
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

        loss_save(epoch) = trainLoss / data.length
      }

      val totalTime = dataTimer.getElapsedTime / 1e6f
      val loopTime = totalTime - prepareTime
      val timePerEpoc = loopTime / nbEpoch

      val fp2 = openf(a, "w")
      fprintf(fp2, "unit: %s\\n", "1 epoch")
      for (i <- (0 until loss_save.length): Rep[Range]) {
        fprintf(fp2, "%lf\\n", loss_save(i))
      }
      fprintf(fp2, "run time: %lf %lf\\n", prepareTime, timePerEpoc)
      closef(fp2)
    }
  }

  def main(args: Array[String]) {
    val gpu_file = new PrintWriter(new File(root_dir + gpu_file_dir))
    gpu_file.println(deepspeechGPU.code)
    gpu_file.flush()
  }
}

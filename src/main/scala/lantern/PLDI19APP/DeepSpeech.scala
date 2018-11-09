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
  val cpu_file_dir = "deepspeech/lantern/Lantern.cpp"
  val gpu_file_dir = "deepspeech/lantern/Lantern.cu"
  // TODO: Specify data directory.
  val data_dir: String = ???

  val deepspeechGPU = new LanternDriverCudnn[String, Unit] {

    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {
      Random.srand(Some(42))
      val dataTimer = Timer2()
      dataTimer.startTimer

      val batchSize = 20

      case class BatchNorm1D(dimSize: Int, name: String = "batch_norm_1d") extends Module {
        val scale: TensorR = TensorR(Tensor.ones(dimSize))
        val bias: TensorR = TensorR(Tensor.zeros(dimSize))
        val runningMean: Tensor = Tensor.zeros(dimSize)
        val runningVar: Tensor = Tensor.zeros(dimSize)
        def apply(in: TensorR): TensorR @diff = {
          assert(in.x.rank == 2 && in.x.shape(1) == dimSize, s"BatchNorm1D input should be rank2, with shape 1 same as dimSize, got ${in.x.shape} : ${dimSize}")
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
          assert(in.x.rank == 4 && in.x.shape(1) == num_features, s"BatchNorm2D input should be rank 2, with shape 1 same as num_features, got ${in.x.shape} : ${num_features}")
          in.batchNorm(scale, bias, runningMean, runningVar)
        }
      }

      // case class MaskConv(seq_module: Module) extends Module {
      //   def apply(x: TensorR, lengths: Array[Int]) = {
      //     // x: input of size B, C, D, T
      //     // lengths: the actual length of each sequence in the batch
      //     val xx = seq_module(x)
      //     val mask = Tensor.zeros(x.shape: _*)
      //     lengths zipWithIndex foreach { case (length, index) =>
      //       if (mask(i).shape(2) - length > 0) ??? mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
      //           x = x.masked_fill(mask, 0)
      //       return x, lengths
      //     }
      //   }
      // }

      // mask(input, lengths) //:param x: The input of size BxCxDxT :param lengths: The actual length of each sequence in the batch

      // def forward(self, x, lengths):
      //   """
      //   :param x: The input of size BxCxDxT
      //   :param lengths: The actual length of each sequence in the batch
      //   :return: Masked output from the module
      //   """
      //   for module in self.seq_module:
      //       x = module(x)
      //       mask = torch.ByteTensor(x.size()).fill_(0)
      //       if x.is_cuda:
      //           mask = mask.cuda()
      //       for i, length in enumerate(lengths):
      //           length = length.item()
      //           if (mask[i].size(2) - length) > 0:
      //               mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
      //       x = x.masked_fill(mask, 0)
      //   return x, lengths

      // Reference: https://github.com/SeanNaren/deepspeech.pytorch/blob/c959d29c381e5bef7cdfb0cd420ddacd89d11520/model.py#L80
      case class BatchRNN(val name: String = "batch_rnn",
                          inputSize: Int, hiddenSize: Int, rnnMode: RnnMode = LstmMode,
                          bidirectional: Boolean = false, useBatchNorm: Boolean = true) extends Module {
        val rnn = RNNBase(rnnMode, inputSize, hiddenSize, bidirectional = bidirectional)
        val batchNorm: Option[BatchNorm1D] = if (useBatchNorm) Some(BatchNorm1D(inputSize)) else None

        def apply(input: TensorR): TensorR @diff = {
          val in1 = batchNorm match {
            case None => input
            case Some(batchNorm) => batchNorm(input)
          }
          val output = rnn(in1)
          val timeD = output.x.shape(0)
          val batchD = output.x.shape(1)
          // TODO (Fei Wang) implementation using if else has compilation error
          (bidirectional) match {
            case true => output.resize(timeD, batchD, 2, -1).sum(2)
            case false => output
          }
        }
      }

      // Reference: https://github.com/SeanNaren/deepspeech.pytorch/blob/c959d29c381e5bef7cdfb0cd420ddacd89d11520/model.py#L105
      // TODO: Implement.
      case class Lookahead(val name: String = "lookahead", numFeatures: Int, context: Int) extends Module {
        assert(context >= 1, "Context size must be at least 1")
        val weight = TensorR(Tensor.rand(Seq(numFeatures, context + 1), scala.math.sqrt(context + 1).toFloat))

        // TODO (Fei Wang): this could be optimized by a user-defined kernel?
        def apply(input: TensorR): TensorR @diff = {
          val padding = TensorR(Tensor.zeros((context +: input.x.shape.drop(1)): _*))
          val x = input.concat(0, padding)
          val xs = (0 until input.x.shape(0): Range) map (i => x(i, i + context + 1))
          val xc = xs.head.concat(0, xs.tail: _*).permute(0, 2, 3, 1)
          (x mul_sub weight).sum(3)
        }
      }

      // Reference: https://github.com/SeanNaren/deepspeech.pytorch/blob/c959d29c381e5bef7cdfb0cd420ddacd89d11520/model.py#L145
      case class DeepSpeech(val name: String = "deepspeech",
                            rnnMode: RnnMode = LstmMode, labels: String = "abc",
                            rnnHiddenSize: Int = 768, numLayers: Int = 5,
                            sampleRate: Int = 16000, windowSize: Float = 0.02f,
                            bidirectional: Boolean = true, context: Int = 20) extends Module {

        assert(rnnHiddenSize >= 1, "RNN hidden size must be at least 1")
        assert(numLayers >= 1, "Number of RNN layers must be at least 1")

        val numClasses = labels.length

        val conv = new Module {
          val name = "conv"
          val conv1 = Conv2D(1, 32, Seq(41, 11), stride = Seq(2, 2), pad = Seq(20, 5))
          val bn1 = BatchNorm2D(32)
          val conv2 = Conv2D(32, 32, Seq(21, 11), stride = Seq(2, 1), pad = Seq(10, 5))
          val bn2 = BatchNorm2D(32)
          def apply(in: TensorR): TensorR @diff = {
            val step1 = bn1(conv1(in)).hardTanh(0, 20, inPlace = true)
            bn2(conv2(step1)).hardTanh(0, 20, inPlace = true)
          }
        }

        val rnnInputSize: Int = {
          var tmp: Int = (floor((sampleRate * windowSize) / 2) + 1).toInt
          tmp = (floor((sampleRate * windowSize) / 2) + 1).toInt
          tmp = (floor(tmp + 2 * 20 - 41) / 2 + 1).toInt
          tmp = (floor(tmp + 2 * 10 - 21) / 2 + 1).toInt
          tmp *= 32
          tmp
        }

        val rnns = ArrayBuffer[Module]()
        rnns += BatchRNN(s"batch_rnn0", rnnInputSize, rnnHiddenSize, rnnMode, bidirectional, useBatchNorm = false)
        for (layer <- (0 until numLayers - 1): Range) {
          rnns += BatchRNN(s"batch_rnn${layer + 1}", rnnHiddenSize, rnnHiddenSize, rnnMode, bidirectional)
        }

        val lookahead: Option[Module] = if (bidirectional) None else Some(Lookahead(numFeatures = rnnHiddenSize, context = context))

        val fc = new Module {
          val name: String = "fully_connected"
          val bn = BatchNorm1D(rnnHiddenSize)
          val linear = Linear1D(rnnHiddenSize, numClasses, bias=false)
          def apply(in: TensorR) = {
            linear(bn(in))
          }
        }
        // val inferenceSoftmax: Module = ???

        // class InferenceBatchSoftmax(nn.Module):
        //   def forward(self, input_):
        //     if not self.training:
        //       return F.softmax(input_, dim=-1)
        //     else:
        //       return input_

        // TODO: Implement.
        def apply(input: TensorR, lengths: Array[Int]): TensorR @diff = {
          val outputLengths = getSeqLens(lengths)
          val x = conv(input)
          x
        }
            // def forward(self, x, lengths):
            //   lengths = lengths.cpu().int()
            //   output_lengths = self.get_seq_lens(lengths)
            //   x, _ = self.conv(x, output_lengths)

            //   sizes = x.size()
            //   x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
            //   x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

            //   for rnn in self.rnns:
            //       x = rnn(x, output_lengths)

            //   if not self._bidirectional:  # no need for lookahead layer in bidirectional
            //       x = self.lookahead(x)

            //   x = self.fc(x)
            //   x = x.transpose(0, 1)
            //   # identity in training mode, softmax in eval mode
            //   x = self.inference_softmax(x)
            //   return x, output_lengths

        def getSeqLens(lengths: Array[Int]) = {
          conv.modules.foldLeft(lengths) { case(ls, (_, m)) =>
            if (m.isInstanceOf[Conv2D]) {
              val mm = m.asInstanceOf[Conv2D]
              ls.map(x => (x + 2 * mm.pad(1) - mm.dilation(1) * (mm.kernelSize(1) - 1) - 1) / mm.stride(1) + 1)
            } else ls
          }
        }

      }

      val net = DeepSpeech()
      // TODO: PyTorch DeepSpeech model uses SGD with Nesterov momentum.
      val opt = SGD(net, learning_rate = 3e-4f, gradClip = 1000.0f)

      // def lossFun(input: TensorR, target: Rep[Array[Int]]) = { (dummy: TensorR) =>
      //   val res = net(input).logSoftmaxB().nllLossB(target)
      //   res.sum()
      // }

      // // Training
      // val nbEpoch = 4

      // // TODO: Replace with real data loader.
      // val train = new Dataset.DataLoaderTest("dummy_input", "dummy_output", dims = Seq())

      // val prepareTime = dataTimer.getElapsedTime / 1e6f
      // printf("Data normalized (all prepare time) in %lf sec\\n", prepareTime)

      // val loss_save = NewArray[Double](nbEpoch)

      // val addr = getMallocAddr() // remember current allocation pointer here
      // val addrCuda = getCudaMallocAddr()

      // generateRawComment("training loop starts here")
      // for (epoch <- 0 until nbEpoch: Rep[Range]) {
      //   val trainTimer = Timer2()
      //   var imgIdx = var_new(0)
      //   var trainLoss = var_new(0.0f)
      //   printf("Start training epoch %d\\n", epoch + 1)
      //   trainTimer.startTimer

      //   train.foreachBatch(batchSize) { (batchIndex: Rep[Int], input: Tensor, target: Rep[Array[Int]]) =>
      //     imgIdx += batchSize
      //     val inputR = TensorR(input.toGPU(), isInput = true)
      //     val targetR = target.toGPU(batchSize)
      //     val loss = gradR_loss(lossFun(inputR, targetR))(Tensor.zeros(1))
      //     trainLoss += loss.data(0)
      //     opt.perform{case (name, (tr, ot)) => tr.d.toCPU().printHead(5, name)}
      //     error("stop")
      //     opt.step()

      //     // selective printing
      //     if (imgIdx % (train.length / 10) == 0) {
      //       printf(s"Train epoch %d: [%d/%d (%.0f%%)]\\tAverage Loss: %.6f\\n", epoch, imgIdx, train.length, 100.0 * imgIdx /train.length, trainLoss/imgIdx)
      //       unchecked[Unit]("fflush(stdout)")
      //     }
      //     resetMallocAddr(addr)
      //     resetCudaMallocAddr(addrCuda)
      //   }
      //   val delta = trainTimer.getElapsedTime
      //   printf("Training completed in %ldms (%ld us/images)\\n", delta/1000L, delta/train.length)

      //   loss_save(epoch) = trainLoss / train.length
      // }

      // val totalTime = dataTimer.getElapsedTime / 1e6f
      // val loopTime = totalTime - prepareTime
      // val timePerEpoc = loopTime / nbEpoch

      // val fp2 = openf(a, "w")
      // fprintf(fp2, "unit: %s\\n", "1 epoch")
      // for (i <- (0 until loss_save.length): Rep[Range]) {
      //   fprintf(fp2, "%lf\\n", loss_save(i))
      // }
      // fprintf(fp2, "run time: %lf %lf\\n", prepareTime, timePerEpoc)
      // closef(fp2)
    }
  }

  def main(args: Array[String]) {
    val gpu_file = new PrintWriter(new File(root_dir + gpu_file_dir))
    gpu_file.println(deepspeechGPU.code)
    gpu_file.flush()
  }
}

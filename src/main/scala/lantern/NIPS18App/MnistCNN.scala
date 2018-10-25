package lantern
package NIPS18App

import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

import scala.collection.mutable.ArrayBuffer
import scala.collection.Seq
import scala.math._

import java.io.PrintWriter;
import java.io.File;

object MnistCNN {

  val root_dir = "src/out/NIPS18evaluation/"
  val cpu_file_dir = "evaluationCNN/Lantern/Lantern.cpp"
  val gpu_file_dir = "evaluationCNN/Lantern/Lantern.cu"

  val mnistCPU = new LanternDriverC[String, Unit] {

    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {
      Random.srand(Some(42))
      val dataTimer = Timer2()
      dataTimer.startTimer

      val (batchSize, iChan1, iRow1, iCol1) = (100, 1, 28, 28)

      case class MNIST(val name: String = "mnist") extends Module {
        val conv1 = Conv2D(inChannel = 1, outChannel = 10, kernelSize = Seq(5, 5))
        val conv2 = Conv2D(inChannel = 10, outChannel = 20, kernelSize = Seq(5, 5))
        val linear1 = Linear1D(inSize = 320, outSize = 50)
        val linear2 = Linear1D(inSize = 50, outSize = 10)

        def apply(in: TensorR): TensorR @diff = {
          val step0 = conv1(in)
          val step1 = step0.relu().maxPoolBK(kernels = Seq(2,2), strides = Seq(2,2), None)
          val step2 = conv2(step1).relu().maxPoolBK(kernels = Seq(2,2), strides = Seq(2,2), None)
          val step3 = linear1(step2.resize(-1, 320)).dropout(0.5f)
          linear2(step3)
        }
      }
      val net = MNIST()
      val opt = SGD(net, learning_rate = 0.0005f, gradClip = 1000.0f)

      def lossFun(input: TensorR, target: Rep[Array[Int]]) = { (batchIndex: TensorR) =>
        val res = net(input).logSoftmaxB().nllLossB(target)
        res.sum()
      }

      // Training
      val nbEpoch = 4

      val tot1 = NewArray[Long](2)
      val tot2 = NewArray[Long](2)

      val train = new Dataset.DataLoader("mnist", true, mean = 0.1307f, std = 0.3081f, Seq(iChan1, iRow1, iCol1))
      train.normalize()

      val prepareTime = dataTimer.getElapsedTime / 1e6f
      printf("Data normalized (all prepare time) in %lf sec\\n", prepareTime)

      val loss_save = NewArray[Double](nbEpoch)

      val addr = getMallocAddr() // remember current allocation pointer here

      generateRawComment("training loop starts here")
      for (epoch <- 0 until nbEpoch: Rep[Range]) {
        val trainTimer = Timer2()
        var imgIdx = var_new(0)
        var trainLoss = var_new(0.0f)
        printf("Start training epoch %d\\n", epoch + 1)
        trainTimer.startTimer

        train.foreachBatch(batchSize) { (batchIndex: Rep[Int], input: Tensor, target: Rep[Array[Int]]) =>
          imgIdx += batchSize
          val inputR = TensorR(input, isInput=true)
          val loss = gradR_loss(lossFun(inputR, target))(Tensor.zeros(4))
          trainLoss += loss.data(0)
          opt.step()

          // selective printing
          if (imgIdx % (train.length / 10) == 0) {
            printf(s"Train epoch %d: [%d/%d (%.0f%%)]\\tAverage Loss: %.6f\\n", epoch, imgIdx, train.length, 100.0 * imgIdx /train.length, trainLoss/imgIdx)
            unchecked[Unit]("fflush(stdout)")
          }
          resetMallocAddr(addr)
        }
        val delta = trainTimer.getElapsedTime
        printf("Training completed in %ldms (%ld us/images)\\n", delta/1000L, delta/train.length)

        loss_save(epoch) = trainLoss / train.length
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


  val mnistGPU = new LanternDriverCudnn[String, Unit] {

    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {
      Random.srand(Some(42))
      val dataTimer = Timer2()
      dataTimer.startTimer

      val (batchSize, iChan1, iRow1, iCol1) = (100, 1, 28, 28)

      case class MNIST(val name: String = "mnist") extends Module {
        val conv1 = Conv2D(inChannel = 1, outChannel = 10, kernelSize = Seq(5, 5))
        val conv2 = Conv2D(inChannel = 10, outChannel = 20, kernelSize = Seq(5, 5))
        val linear1 = Linear1D(inSize = 320, outSize = 50)
        val linear2 = Linear1D(inSize = 50, outSize = 10)

        def apply(in: TensorR): TensorR @diff = {
          val step0 = conv1(in)
          val step1 = step0.relu().maxPoolBK(kernels = Seq(2,2), strides = Seq(2,2), None)
          val step2 = conv2(step1).relu().maxPoolBK(kernels = Seq(2,2), strides = Seq(2,2), None).resize(-1, 320)
          val step3 = linear1(step2).dropout(0.5f)
          linear2(step3)
        }
      }
      val net = MNIST()
      val opt = SGD(net, learning_rate = 0.0005f, gradClip = 1000.0f)

      def lossFun(input: TensorR, target: Rep[Array[Int]]) = { (batchIndex: TensorR) =>
        val res = net(input).logSoftmaxB().nllLossB(target)
        res.sum()
      }

      // Training
      val nbEpoch = 4

      val tot1 = NewArray[Long](2)
      val tot2 = NewArray[Long](2)

      val train = new Dataset.DataLoader("mnist", true, mean = 0.1307f, std = 0.3081f, Seq(iChan1, iRow1, iCol1))
      // Move target labels to GPU.
      train.target.moveToGPU(train.tlen)
      train.normalize()

      val prepareTime = dataTimer.getElapsedTime / 1e6f
      printf("Data normalized (all prepare time) in %lf sec\\n", prepareTime)

      val loss_save = NewArray[Double](nbEpoch)

      val addr = getMallocAddr() // remember current allocation pointer here
      val addrCuda = getCudaMallocAddr()

      generateRawComment("training loop starts here")
      for (epoch <- 0 until nbEpoch: Rep[Range]) {
        val trainTimer = Timer2()
        var imgIdx = var_new(0)
        var trainLoss = var_new(0.0f)
        printf("Start training epoch %d\\n", epoch + 1)
        trainTimer.startTimer

        train.foreachBatch(batchSize) { (batchIndex: Rep[Int], input: Tensor, target: Rep[Array[Int]]) =>
          imgIdx += batchSize
          val inputR = TensorR(input.toGPU(), isInput=true)
          val loss = gradR_loss(lossFun(inputR, target))(Tensor.zeros(4))
          trainLoss += loss.data(0)
          opt.step()

          // selective printing
          if (imgIdx % (train.length / 10) == 0) {
            printf(s"Train epoch %d: [%d/%d (%.0f%%)]\\tAverage Loss: %.6f\\n", epoch, imgIdx, train.length, 100.0 * imgIdx /train.length, trainLoss/imgIdx)
            unchecked[Unit]("fflush(stdout)")
          }
          resetMallocAddr(addr)
          resetCudaMallocAddr(addrCuda)
        }
        val delta = trainTimer.getElapsedTime
        printf("Training completed in %ldms (%ld us/images)\\n", delta/1000L, delta/train.length)

        loss_save(epoch) = trainLoss / train.length
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
    val cpu_cnn_file = new PrintWriter(new File(root_dir + cpu_file_dir))
    val gpu_cnn_file = new PrintWriter(new File(root_dir + gpu_file_dir))
    cpu_cnn_file.println(mnistCPU.code)
    cpu_cnn_file.flush()
    gpu_cnn_file.println(mnistGPU.code)
    gpu_cnn_file.flush()
  }
}

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

  val root_dir = "src/out/ICFP18evaluation/"
  val root_dir2 = "src/out/NIPS18evaluation/"
  val file_dir = "evaluationCNN/Lantern/Lantern.cpp"

  val mnist  = new LanternDriverC[String, Unit] {

    // From the MNIST pytorch example
    val mean = 0.1307f
    val std = 0.3081f

    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {
      printf("Here we go!! Go MNIST!!!!\\n")
      Random.srand(Some(42))

      //move timer here to track all of the prepare time
      val dataTimer = Timer2()
      dataTimer.startTimer

      val variables = ArrayBuffer[TensorR]()

      // input size
      val (iChan1, iRow1, iCol1) = (1, 28, 28)

      // Layer 1
      val (inChan1, outChan1, kRow1, kCol1, sRow1, sCol1) = (iChan1, 10, 5, 5, 1, 1)

      // stride maxpool
      val (smRow1, smCol1) = (2, 2)

      // FIXME scale based on PyTorch
      val varConv1 = TensorR(Tensor.rand(Seq(outChan1, inChan1, kRow1, kCol1), 1.0f / sqrt(inChan1 * kRow1 * kCol1).toFloat))
      variables += varConv1

      // input size
      val (iChan2, iRow2, iCol2) = (outChan1, convSize(iRow1, kRow1, sRow1)/smRow1, convSize(iCol1, kCol1, sCol1)/smCol1)

      // Layer 2
      val (inChan2, outChan2, kRow2, kCol2, sRow2, sCol2) = (outChan1, 20, 5, 5, 1, 1)

      // stride maxpool
      val (smRow2, smCol2) = (2, 2)

      val varConv2 = TensorR(Tensor.rand(Seq(outChan2, inChan2, kRow2, kCol2), 1.0f / sqrt(inChan2 * kRow2 * kCol2).toFloat))
      variables += varConv2

      // Layer 3
      val (oRow2, oCol2) = (convSize(iRow2, kRow2, sRow2)/smRow2, convSize(iCol2, kCol2, sCol2)/smCol2)
      val (in3, out3) = (outChan2 * oRow2 * oCol2, 50)  // 320

      val varA1 = TensorR(Tensor.rand(Seq(out3, in3), 1.0f / sqrt(in3).toFloat))
      val varB1 = TensorR(Tensor.rand(Seq(out3), 1.0f / sqrt(in3).toFloat))
      variables += varA1
      variables += varB1

      // Layer 4
      val (in4, out4) = (out3, 10)

      val varA2 = TensorR(Tensor.rand(Seq(out4, in4), 1.0f / sqrt(in4).toFloat))
      val varB2 = TensorR(Tensor.rand(Seq(out4), 1.0f / sqrt(in4).toFloat))
      variables += varA2
      variables += varB2

      // Training
      val nbEpoch = 10
      val lr = 0.0005f
      val mom = 0.0f

      val momentum = if (mom > 0.0f) variables map(tR => Tensor.zeros_like(tR.d)) else ArrayBuffer[Tensor]()

      val tot1 = NewArray[Long](2)
      val tot2 = NewArray[Long](2)

      val train = new Dataset.DataLoader("mnist", true, mean, std, Seq(iChan1, iRow1, iCol1))
      printf("Start normalize\\n")
      train.normalize()

      // we skip tests for the experiments
      //val test = new DataLoader("mnist", false, iChan1, iRow1, iCol1)
      //test.normalize()

      def trainFun(input: TensorR, target: Rep[Int]) = { (dummy: TensorR) =>
        val resL1 = input.conv(varConv1, sRow1, sCol1, tot1).maxPool(smRow1, smCol1).relu()
        val resL2 = resL1.conv(varConv2, sRow2, sCol2, tot2).maxPool(smRow2, smCol2).relu()
        val resL3 = ((varA1 dot resL2.resize(in3)) + varB1).relu().dropout(0.5f)
        val resL4 = (varA2 dot resL3) + varB2
        val res = resL4.logSoftmax()
        res.nllLoss(target)
      }

      val prepareTime = dataTimer.getElapsedTime / 1e6f
      printf("Data normalized (all prepare time) in %lf sec\\n", prepareTime)

      val loss_save = NewArray[Double](nbEpoch)

      val addr = getMallocAddr() // remember current allocation pointer here
      for (epoch <- 0 until nbEpoch: Rep[Range]) {

        val trainTimer = Timer2()
        var trainLoss = var_new(0.0f)
        printf("Start training epoch %d\\n", epoch + 1)
        trainTimer.startTimer
        train foreach { (idx: Rep[Int], input: Tensor, target: Rep[Int]) =>

          val inputR = TensorR(input , isInput=true)
          val loss = gradR_loss(trainFun(inputR, target))(Tensor.scalar(0.0f))
          trainLoss += loss.data(0)

          // Update weights
          for ((weight, idx) <- variables.zipWithIndex) {
            val d = if (mom > 0.0f) {
              printf("TBT\\n")
              exit()
              val sMom = momentum(idx)
              sMom.cmulAdd(mom, weight.d)
            } else {
              weight.d
            }

            weight.x.addMul(-lr, d)
            weight.clear_grad()
          }

          val imgIdx = idx + 1
          if (imgIdx %  (train.length / 10) == 0) {
            printf(s"Train epoch %d: [%d/%d (%.0f%%)]\\tAverage Loss: %.6f\\n", epoch, imgIdx, train.length, 100.0 * imgIdx /train.length, trainLoss/imgIdx)
            unchecked[Unit]("fflush(stdout)")
          }
          resetMallocAddr(addr)
        }
        val delta = trainTimer.getElapsedTime
        printf("Training completed in %ldms (%ld us/images)\\n", delta/1000L, delta/train.length)

        loss_save(epoch) = trainLoss / train.length

        /* skip tests
        def testFun(input: Tensor) = {
          val (resL1, _) = input.conv2D(conv1, sRow1, sCol1).maxPool(smRow1, smCol1)
          val (resL2, _) = resL1.relu().conv2D(conv2, sRow2, sCol2).maxPool(smRow2, smCol2)
          val resL3 = ((a1 dot resL2.relu().resize(in3)) + b1).relu()
          val resL4 = (a2 dot resL3) + b2
          resL4.logSoftmax()
        }

        printf("\\nStart testing:\\n")
        val testTimer = Timer2()
        testTimer.startTimer
        imgIdx = var_new(0)
        var testLoss = var_new(0.0)
        val correct = var_new(0)
        test foreach { (input: Tensor, target: Rep[Int]) =>
          imgIdx += 1
          val res = testFun(input)

          testLoss += res.nllLoss(target).data(0)
          if (res.maxIndex() == target)
            correct += 1
        }
        printf("Test set: Average loss: %.4f, Acurracy: %d/%d (%.0f) in %ldms\\n", testLoss / test.length, correct, test.length, 100.0 * correct / test.length, testTimer.getElapsedTime/1000L)
        printf("\\n\\n")
        */
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

  val mnist2  = new DslDriverC[String, Unit] with NNModule {

    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {

      Random.srand(Some(42))
      val dataTimer = Timer2()
      dataTimer.startTimer

      val (batch, iChan1, iRow1, iCol1) = (100, 1, 28, 28)

      case class MNIST(val name: String = "mnist") extends Module {
        val conv1 = Conv2D(inChannel = 1, outChannel = 10, kernelSize = Seq(5, 5))
        val conv2 = Conv2D(inChannel = 10, outChannel = 20, kernelSize = Seq(5, 5))
        val linear1 = Linear1D(inSize = 320, outSize = 50)
        val linear2 = Linear1D(inSize = 50, outSize = 10)

        def apply(in: TensorR): TensorR @diff = {
          val step1 = conv1(in).relu().maxPoolBK(kernels = Seq(2,2), strides = Seq(2,2), None)
          val step2 = conv2(step1).relu().maxPoolBK(kernels = Seq(2,2), strides = Seq(2,2), None)
          val step3 = linear1(step2.resize(-1, 320)).dropout(0.5f)
          linear2(step3)
        }
      }
      val net = MNIST()
      val opt = SGD(net, learning_rate = 0.0005f, gradClip = 1000.0f)

      def lossFun(input: TensorR, target: Rep[Array[Int]]) = { (dummy: TensorR) =>
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
      for (epoch <- 0 until nbEpoch: Rep[Range]) {

        val trainTimer = Timer2()
        var imgIdx = var_new(0)
        var trainLoss = var_new(0.0f)
        printf("Start training epoch %d\\n", epoch + 1)
        trainTimer.startTimer
        train.foreachBatch(batch){ (dummy: Rep[Int], input: Tensor, target: Rep[Array[Int]]) =>
          imgIdx += batch
          val inputR = TensorR(input , isInput=true)
          val loss = gradR_loss(lossFun(inputR, target))(Tensor.scalar(0.0f))
          trainLoss += loss.data(0)

          opt.step()

          // selective printing
          if (imgIdx %  (train.length / 10) == 0) {
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

  def main(args: Array[String]) {
    val cnn_file = new PrintWriter(new File(root_dir2 + file_dir))
    cnn_file.println(mnist2.code)
    cnn_file.flush()
  }
}

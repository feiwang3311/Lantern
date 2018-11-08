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

object TestNet {

  val root_dir = "src/out/PLDI19evaluation/"
  val cpu_file_dir = "squeezenet/lantern/test.cpp"
  val gpu_file_dir = "squeezenet/lantern/test.cu"
  val data_dir_x = "../../cifar10_data/cifar-10-batches-bin/small_batch_x.bin"
  val data_dir_y = "../../cifar10_data/cifar-10-batches-bin/small_batch_y.bin"

  val squeezenetCPU = new LanternDriverC[String, Unit] {

    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {
      Random.srand(Some(42))
      val dataTimer = Timer2()
      dataTimer.startTimer

      val (batchSize, iChan1, iRow1, iCol1) = (64, 3, 32, 32)

      case class FireModule(val name: String = "squeezenet_fire", val inChannel: Int, val squeezeDepth: Int, val expandDepth: Int) extends Module {
        val convs1 = Conv2D(inChannel = inChannel, outChannel = squeezeDepth, kernelSize = Seq(1, 1))
        val conve1 = Conv2D(inChannel = squeezeDepth, outChannel = expandDepth, kernelSize = Seq(1, 1))
        val conve2 = Conv2D(inChannel = squeezeDepth, outChannel = expandDepth, kernelSize = Seq(3, 3), pad = 1)
        def apply(in: TensorR): TensorR @diff = {
          val step_squeeze = convs1(in).relu(false)
          val step_expand1 = conve1(step_squeeze).relu(true)
          val step_expand2 = conve2(step_squeeze).relu(true)
          val res = step_expand1.concat(dim = 1, step_expand2)
          res
        }
      }

      case class SqueezeNetCifar10(val name: String = "squeezenet_cifar10", val num_classes: Int) extends Module {
        val conv1 = Conv2D(inChannel = 3, outChannel = 96, kernelSize = Seq(3, 3), pad = 1)
        val fire1 = FireModule(inChannel = 96, squeezeDepth = 16, expandDepth = 64)
        // val fire1_convs1 = Conv2D(inChannel = 96, outChannel = 16, kernelSize = Seq(1, 1))
        // val fire1_conve1 = Conv2D(inChannel = 16, outChannel = 64, kernelSize = Seq(1, 1))
        // val fire1_conve2 = Conv2D(inChannel = 16, outChannel = 64, kernelSize = Seq(3, 3), pad = 1)
        val fire2 = FireModule(inChannel = 128, squeezeDepth = 16, expandDepth = 64)
        val fire3 = FireModule(inChannel = 128, squeezeDepth = 32, expandDepth = 128)
        val fire4 = FireModule(inChannel = 256, squeezeDepth = 32, expandDepth = 128)
        val fire5 = FireModule(inChannel = 256, squeezeDepth = 48, expandDepth = 192)
        val fire6 = FireModule(inChannel = 384, squeezeDepth = 48, expandDepth = 192)
        val fire7 = FireModule(inChannel = 384, squeezeDepth = 64, expandDepth = 256)
        val fire8 = FireModule(inChannel = 512, squeezeDepth = 64, expandDepth = 256)
        val conv2 = Conv2D(inChannel = 512, outChannel = num_classes, kernelSize = Seq(4, 4))

        def apply(in: TensorR): TensorR @diff = {
          // in.x.printHead(10, "forward, input")
          // conv1.kernel.x.printHead(10, "conv1 weight")
          val step0 = conv1(in).relu(true).maxPoolBK(kernels = Seq(2, 2), strides = Seq(2, 2), None)
          // step0.x.printHead(10, "forward, after conv1")
          val step1 = fire1(step0)
          // val fire1_squeeze = fire1_convs1(step0).relu()
          // fire1_squeeze.x.printHead(10, "forward, fire1_squeeze")
          // val fire1_expand1 = fire1_conve1(fire1_squeeze).relu()
          // fire1_expand1.x.printHead(10, "forward, fire1_expand1")
          // val fire1_expand2 = fire1_conve2(fire1_squeeze).relu()
          // fire1_expand2.x.printHead(10, "forward, fire1_expand2")
          // val step1 = fire1_expand1.concat(dim = 1, fire1_expand2)
          // step1.x.printHead(10, "forward, after fire1")
          val step2 = fire2(step1)
          // step2.x.printHead(10, "forward, after fire2")
          val step3 = fire3(step2).maxPoolBK(kernels = Seq(2, 2), strides = Seq(2, 2), None)
          // step3.x.printHead(10, "forward, after fire3")
          val step4 = fire4(step3)
          val step5 = fire5(step4)
          // step5.x.printHead(10, "forward, after fire5")
          val step6 = fire6(step5)
          val step7 = fire7(step6).maxPoolBK(kernels = Seq(2, 2), strides = Seq(2, 2), None)
          val step8 = fire8(step7) //.averagePoolBK(kernels = Seq(4, 4), strides = Seq(1, 1), None)
          // val step8 = fire1_expand2.averagePoolBK(kernels = Seq(16, 16), strides = Seq(1, 1), None)
          val logits = conv2(step8)
          logits.resize(-1, num_classes)
        }
      }

      val net = SqueezeNetCifar10(num_classes = 10)
      val opt = SGD(net, learning_rate = 0.005f, gradClip = 1000.0f)

      def lossFun(input: TensorR, target: Rep[Array[Int]]) = { (dummy: TensorR) =>
        val res = net(input).logSoftmaxB().nllLossB(target)
        res.sum()
      }

      // Training
      val nbEpoch = 100

      val tot1 = NewArray[Long](2)
      val tot2 = NewArray[Long](2)

      val train = new Dataset.DataLoaderTest(data_dir_x, data_dir_y, Seq(iChan1, iRow1, iCol1))

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
          val loss = gradR_loss(lossFun(inputR, target))(Tensor.zeros(1))
          trainLoss += loss.data(0)
          // opt.perform{case (name, (tr, ot)) => tr.d.printHead(5, name)}
          // error("stop")
          opt.step()

          resetMallocAddr(addr)
        }
        val delta = trainTimer.getElapsedTime
        printf("Training completed in %ldms (%ld us/images), loss is %f\\n", delta/1000L, delta/train.length, trainLoss / train.length)
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
    cpu_cnn_file.println(squeezenetCPU.code)
    cpu_cnn_file.flush()
  }
}

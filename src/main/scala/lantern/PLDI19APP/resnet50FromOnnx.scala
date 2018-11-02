package lantern
package PLDI18App

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

object Resnet50Onnx {

  val root_dir = "src/out/PLDI19evaluation/"
  val inference_cpu_file_dir = "resnet50/lantern/LanternOnnxInference.cpp"
  val cpu_file_dir = "resnet50/lantern/LanternOnnx.cpp"
  val gpu_file_dir = "resnet50/lantern/LanternOnnx.cu"
  val model_file = "resnet50/resnet50.onnx"
  val relative_data_dir = "../../cifar10_data/cifar-10-batches-bin/data_batch_1.bin"

  val resnet50InferenceCPU = new LanternDriverC[String, Unit] with ONNXLib {
    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {
      // reading ONNX model
      val model = readONNX(root_dir + model_file)
      val (func, x_dims, y_dims) = (model.inference_func(model.initializer_map_tensor), model.x_dims, model.y_dims)

      val (batchSize, iChan1, iRow1, iCol1) = (64, 3, 32, 32)
      val train = new Dataset.Cifar10DataLoader(relative_data_dir, true, Seq(iChan1, iRow1, iCol1))

      train.foreachBatch(batchSize) { (batchIndex: Rep[Int], input: Tensor, target: Rep[Array[Int]]) =>
        input.printHead(10, "input")
        val out = func(input)
        out.printHead(10, "output")
        error("stop")
      }
    }
  }

  // // to here
  // val resnetCPU = new LanternDriverC[String, Unit] with ONNXLib {
  //   @virtualize
  //   def snippet(a: Rep[String]): Rep[Unit] = {
  //     Random.srand(Some(42))
  //     val dataTimer = Timer2()
  //     dataTimer.startTimer

  //     // reading ONNX model
  //     val model = readONNX(model_file)
  //     def lossFun(input: TensorR, target: Rep[Array[Int]]) = { (dummy: TensorR) =>
  //       val res = model.training_func(input).logSoftmaxB().nllLossB(target)
  //       res.sum() / 64.0f
  //     }

  //     val (batchSize, iChan1, iRow1, iCol1) = (64, 3, 32, 32)
  //     val train = new Dataset.Cifar10DataLoader(relative_data_dir, true, Seq(iChan1, iRow1, iCol1))

  //     val prepareTime = dataTimer.getElapsedTime / 1e6f
  //     printf("Data normalized (all prepare time) in %lf sec\\n", prepareTime)

  //     // Training
  //     val nbEpoch = 4
  //     val loss_save = NewArray[Double](nbEpoch)
  //     val addr = getMallocAddr() // remember current allocation pointer here

  //     generateRawComment("training loop starts here")
  //     for (epoch <- 0 until nbEpoch: Rep[Range]) {
  //       val trainTimer = Timer2()
  //       var trainLoss = var_new(0.0f)
  //       printf("Start training epoch %d\\n", epoch + 1)
  //       trainTimer.startTimer

  //       train.foreachBatch(batchSize) { (batchIndex: Rep[Int], input: Tensor, target: Rep[Array[Int]]) =>
  //         val inputR = TensorR(input, isInput=true)
  //         val loss = gradR_loss(lossFun(inputR, target))(Tensor.zeros(1))
  //         // loss.print("loss")
  //         trainLoss += loss.data(0)
  //         model.initializer_map_tensorR foreach { case (name, tr) =>
  //           tr.d.changeTo { i =>
  //             tr.x.data(i) = tr.x.data(i) - 0.005f * tr.d.data(i)
  //             0.0f
  //           }
  //         }
  //         // model.initializer_map_tensorR.toList.sortBy(x => x._1.toInt).foreach {
  //         //   case (name, tr) => tr.x.printHead(10, name)
  //         // }

  //         // selective printing
  //         if ((batchIndex + 1) % (train.length / batchSize / 10) == 0) {
  //           val trained = batchIndex * batchSize
  //           val total = train.length
  //           printf(s"Train epoch %d: [%d/%d (%.0f%%)] Average Loss: %.6f\\n", epoch, trained, total, 100.0*trained/total, trainLoss/batchIndex)
  //           unchecked[Unit]("fflush(stdout)")
  //         }
  //         resetMallocAddr(addr)
  //       }
  //       val delta = trainTimer.getElapsedTime
  //       printf("Training completed in %ldms (%ld us/images)\\n", delta/1000L, delta/train.length)
  //       error("stop")
  //       loss_save(epoch) = trainLoss / train.length
  //     }

  //   }
  // }

  def main(args: Array[String]) {
    val resnet_cpu_inference_file = new PrintWriter(new File(root_dir + inference_cpu_file_dir))
    resnet_cpu_inference_file.println(resnet50InferenceCPU.code)
    // val squeezenet_file = new PrintWriter(new File(root_dir + cpu_file_dir))
    // squeezenet_file.println(squeezenetCPU.code)
    resnet_cpu_inference_file.flush()
  }

}

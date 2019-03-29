package lantern
package NIPS19App

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

object SecOrder {

  val root_dir = "src/out/NIPS19evaluation/basic/"
  val cpu_secOrder_file_dir = "lantern/LanternSecOrder.cpp"
  val gpu_secOrder_file_dir = "lantern/LanternSecOrder.cu"

  val secOrderCPU = new LanternDriverC[String, Unit] with TensorSecOrderApi {
    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {
      // set inputs and vectors for Hessian
      val input1 = Tensor.fromData(Seq(1,1,2,2), 1,2,3,4)
      val inputd1 = Tensor.fromData(Seq(1,1,2,2), 0,0,0,0)
      val input = TensorFR(new TensorF(input1, inputd1))

      val kernel1 = Tensor.fromData(Seq(1,1,2,2), 0.177578f, 0.153097f, -0.454294f, 0.442411f)
      val kerneld1 = Tensor.fromData(Seq(1,1,2,2), 0.4f, 0.5f, 0.6f, 0.7f)
      val kernel = TensorFR(new TensorF(kernel1, kerneld1))

      // compute gradient and hessV
      val res: Tensor = gradHessV { () =>
        input.conv2D_batch(kernel).tanh().sum()
      }

      // correctness assertion
      Tensor.assertEqual(res, Tensor.scalar(0.711658f))
      Tensor.assertEqual(getGradient(kernel),
        Tensor.fromData(Seq(1,1,2,2), 0.493542f, 0.987085f, 1.480627f, 1.974169f))
      Tensor.assertEqual(getHessV(kernel),
        Tensor.fromData(Seq(1,1,2,2), -4.214802f,  -8.429605f, -12.644407f, -16.859209f))
    }
  }

  val secOrderGPU = new LanternDriverCudnn[String, Unit] with TensorSecOrderApi {
    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {
      // set inputs and vectors for Hessian
      val input1 = Tensor.rand(100, 3, 60, 60)
      val inputd1 = Tensor.rand(100, 3, 60, 60)
      val input = TensorFR(new TensorF(input1, inputd1))

      val kernel1 = Tensor.rand(10, 3, 4, 4)
      val kerneld1 = Tensor.rand(10, 3, 4 ,4)
      val kernel = TensorFR(new TensorF(kernel1, kerneld1))

      val addr = getMallocAddr() // remember current allocation pointer here
      val addrCuda = getCudaMallocAddr()

      val timer = Timer2()
      timer.startTimer
      // compute gradient and hessV
      for (i <- (0 until 100): Rep[Range]) {
        val res = gradHessV { () =>
          input.conv2D_batch(kernel).tanh().sum()
        }
        printf("%d ", i)
        resetMallocAddr(addr)
        resetCudaMallocAddr(addrCuda)
      }
      val runTime = timer.getElapsedTime / 1e6f
      printf("sec order in %lf sec\\n", runTime)
      // correctness assertion
//      backend = BackendCPU()
//      Tensor.assertEqual(res.toCPU(), Tensor.scalar(0.711658f))
//      Tensor.assertEqual(getGradient(kernel).toCPU(),
//        Tensor.fromData(Seq(1,1,2,2), 0.493542f, 0.987085f, 1.480627f, 1.974169f))
//      Tensor.assertEqual(getHessV(kernel).toCPU(),
//        Tensor.fromData(Seq(1,1,2,2), -4.214802f,  -8.429605f, -12.644407f, -16.859209f))
    }
  }
//  val squeezenetInferenceGPU = new LanternDriverCudnn[String, Unit] with ONNXLib {
//    @virtualize
//    def snippet(a: Rep[String]): Rep[Unit] = {
//      debug = false
//      // init timer
//      Random.srand(Some(42))
//      val dataTimer = Timer2()
//      dataTimer.startTimer
//
//      // set up data
//      val (batchSize, iChan1, iRow1, iCol1) = (64, 3, 32, 32)
//      val train = new Cifar10DataLoader(relative_data_dir, true, Seq(iChan1, iRow1, iCol1))
//      val prepareTime = dataTimer.getElapsedTime / 1e6f
//      printf("Data reading in %lf sec\\n", prepareTime)
//
//      // reading ONNX model
//      val model = readONNX(root_dir + model_file)
//      val initMap = model.initializer_map_tensor.map{case (name, tr) => (name, tr.toGPU())}
//      val (func, x_dims, y_dims) = (model.inference_func(initMap), model.x_dims, model.y_dims)
//
//      // Inferencing
//      val nbEpoch = 4
//      val addr = getMallocAddr() // remember current allocation pointer here
//      val addrCuda = getCudaMallocAddr()
//
//      generateRawComment("inferencing loop starts here")
//      for (epoch <- 0 until nbEpoch: Rep[Range]) {
//        val trainTimer = Timer2()
//        printf("Start inferencing epoch %d\\n", epoch + 1)
//        trainTimer.startTimer
//
//        train.foreachBatch(batchSize) { (batchIndex: Rep[Int], input: Tensor, target: Rep[Array[Int]]) =>
//          func(input.toGPU())
//          resetMallocAddr(addr)
//          resetCudaMallocAddr(addrCuda)
//        }
//        val delta = trainTimer.getElapsedTime
//        printf("Inferencing completed in %ldms (%ld us/images)\\n", delta/1000L, delta/train.length)
//      }
//    }
//  }
//
//  val squeezenetTrainingCPU = new LanternDriverC[String, Unit] with ONNXLib {
//    @virtualize
//    def snippet(a: Rep[String]): Rep[Unit] = {
//      debug = false
//      // init timer
//      Random.srand(Some(42))
//      val dataTimer = Timer2()
//      dataTimer.startTimer
//      val learning_rate = 0.005f
//
//      // set up data
//      val (batchSize, iChan1, iRow1, iCol1) = (64, 3, 32, 32)
//      val train = new Cifar10DataLoader(relative_data_dir, true, Seq(iChan1, iRow1, iCol1))
//      val prepareTime = dataTimer.getElapsedTime / 1e6f
//      printf("Data reading in %lf sec\\n", prepareTime)
//
//      // reading ONNX model
//      val model = readONNX(root_dir + model_file)
//      val (func, parameters) = model.training_func(model.initializer_map_tensor)
//      def lossFun(input: TensorR, target: Rep[Array[Int]]) = { (dummy: TensorR) =>
//        val res = func(input).logSoftmaxB(1).nllLossB(target)
//        res.mean()
//      }
//
//      // Training
//      val nbEpoch = 4
//      val loss_save = NewArray[Double](nbEpoch)
//      val addr = getMallocAddr() // remember current allocation pointer here
//
//      generateRawComment("training loop starts here")
//      for (epoch <- 0 until nbEpoch: Rep[Range]) {
//        val trainTimer = Timer2()
//        var trainLoss = var_new(0.0f)
//        printf("Start training epoch %d\\n", epoch + 1)
//        trainTimer.startTimer
//
//        train.foreachBatch(batchSize) { (batchIndex: Rep[Int], input: Tensor, target: Rep[Array[Int]]) =>
//          val inputR = TensorR(input, isInput=true)
//          val loss = gradR_loss(lossFun(inputR, target))(Tensor.zeros(1))
//          trainLoss += loss.data(0)
//          parameters foreach { case (name, tr) =>
//            tr.d.changeTo { i =>
//              tr.x.data(i) = tr.x.data(i) - learning_rate * tr.d.data(i)
//              0.0f
//            }
//          }
//          // model.initializer_map_tensorR.toList.sortBy(x => x._1.toInt).foreach {
//          //   case (name, tr) => tr.x.printHead(10, name)
//          // }
//
//          // selective printing
//          if ((batchIndex + 1) % (train.length / batchSize / 10) == 0) {
//            val trained = batchIndex * batchSize
//            val total = train.length
//            printf(s"Train epoch %d: [%d/%d (%.0f%%)] Average Loss: %.6f\\n", epoch, trained, total, 100.0*trained/total, trainLoss/batchIndex)
//            unchecked[Unit]("fflush(stdout)")
//          }
//          resetMallocAddr(addr)
//        }
//        val delta = trainTimer.getElapsedTime
//        printf("Training completed in %ldms (%ld us/images)\\n", delta/1000L, delta/train.length)
//        loss_save(epoch) = trainLoss / train.length
//      }
//    }
//  }
//
//  val squeezenetTrainingGPU = new LanternDriverCudnn[String, Unit] with ONNXLib {
//    @virtualize
//    def snippet(a: Rep[String]): Rep[Unit] = {
//      debug = false
//      // init timer
//      Random.srand(Some(42))
//      val dataTimer = Timer2()
//      dataTimer.startTimer
//      val learning_rate = 0.005f
//
//      // set up data
//      val (batchSize, iChan1, iRow1, iCol1) = (64, 3, 32, 32)
//      val train = new Cifar10DataLoader(relative_data_dir, true, Seq(iChan1, iRow1, iCol1))
//      val prepareTime = dataTimer.getElapsedTime / 1e6f
//      printf("Data reading in %lf sec\\n", prepareTime)
//
//      // reading ONNX model
//      val model = readONNX(root_dir + model_file)
//      val initMap = model.initializer_map_tensor.map{case (name, tr) => (name, tr.toGPU())}
//      val (func, parameters) = model.training_func(initMap)
//      def lossFun(input: TensorR, target: Rep[Array[Int]]) = { (dummy: TensorR) =>
//        val res = func(input).logSoftmaxB(1).nllLossB(target)
//        res.mean()
//      }
//
//      // Training
//      val nbEpoch = 4
//      val loss_save = NewArray[Double](nbEpoch)
//      val time_save = NewArray[Double](nbEpoch)
//
//      val addr = getMallocAddr() // remember current allocation pointer here
//      val addrCuda = getCudaMallocAddr()
//
//      generateRawComment("training loop starts here")
//      for (epoch <- 0 until nbEpoch: Rep[Range]) {
//        val trainTimer = Timer2()
//        var trainLoss = var_new(0.0f)
//        printf("Start training epoch %d\\n", epoch + 1)
//        trainTimer.startTimer
//
//        train.foreachBatch(batchSize) { (batchIndex: Rep[Int], input: Tensor, target: Rep[Array[Int]]) =>
//          val inputR = TensorR(input.toGPU(), isInput=true)
//          val targetR = target.toGPU(batchSize)
//          val loss = gradR_loss(lossFun(inputR, targetR))(Tensor.zeros(1))  // loss is guaranteed to be on CPU
//          trainLoss += loss.toCPU().data(0)
//          parameters foreach { case (name, tr) =>
//            backend.geam(tr.x, false, 1.0f, tr.d, false, -1.0f * learning_rate, tr.x)
//            tr.clear_grad()
//          }
//
//          // selective printing
//          if ((batchIndex + 1) % (train.length / batchSize / 10) == 0) {
//            val trained = batchIndex * batchSize
//            val total = train.length
//            printf(s"Train epoch %d: [%d/%d (%.0f%%)] Average Loss: %.6f\\n", epoch, trained, total, 100.0*trained/total, trainLoss/batchIndex)
//            unchecked[Unit]("fflush(stdout)")
//          }
//          resetMallocAddr(addr)
//          resetCudaMallocAddr(addrCuda)
//        }
//        val delta = trainTimer.getElapsedTime
//        time_save(epoch) = delta / 1000000L
//        printf("Training completed in %ldms (%ld us/images)\\n", delta/1000L, delta/train.length)
//        loss_save(epoch) = trainLoss / train.length
//      }
//
//      val totalTime = dataTimer.getElapsedTime / 1e6f
//      val loopTime = totalTime - prepareTime
//      val timePerEpoc = loopTime / nbEpoch
//
//      // get median time of epochs
//      unchecked[Unit]("sort(", time_save, ", ", time_save, " + ", nbEpoch, ")")
//      val median_time =  time_save(nbEpoch / 2)
//
//      val fp2 = openf(a, "w")
//      fprintf(fp2, "unit: %s\\n", "1 epoch")
//      for (i <- (0 until loss_save.length): Rep[Range]) {
//        fprintf(fp2, "%lf\\n", loss_save(i))
//      }
//      fprintf(fp2, "run time: %lf %lf\\n", prepareTime, median_time)
//      closef(fp2)
//    }
//  }

  def main(args: Array[String]) {
    val secOrderCPUfile = new PrintWriter(new File(root_dir + cpu_secOrder_file_dir))
    secOrderCPUfile.println(secOrderCPU.code)
    secOrderCPUfile.flush()
    val secOrderGPUfile = new PrintWriter(new File(root_dir + gpu_secOrder_file_dir))
    secOrderGPUfile.println(secOrderGPU.code)
    secOrderGPUfile.flush()
   // val squeezenet_cpu_training_file = new PrintWriter(new File(root_dir + cpu_training_file_dir))
   // squeezenet_cpu_training_file.println(squeezenetTrainingCPU.code)
   // squeezenet_cpu_training_file.flush()
   // val squeezenet_gpu_inference_file = new PrintWriter(new File(root_dir + gpu_inference_file_dir))
   // squeezenet_gpu_inference_file.println(squeezenetInferenceGPU.code)
   // squeezenet_gpu_inference_file.flush()
   // val squeezenet_gpu_training_file = new PrintWriter(new File(root_dir + gpu_training_file_dir))
   // squeezenet_gpu_training_file.println(squeezenetTrainingGPU.code)
   // squeezenet_gpu_training_file.flush()
  }
}

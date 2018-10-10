package lantern

import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

import scala.collection.mutable.ArrayBuffer
import scala.collection.{Seq => NSeq}
import scala.math._

import org.scalatest.FunSuite

import java.io.PrintWriter;
import java.io.File;

class MnistCNNTest extends FunSuite {

  val root_dir = "src/out/ICFP18evaluation/"
  val file_dir = "evaluationCNN/Lantern/Lantern.cpp"

  val mnist  = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {

    // From the MNIST pytorch example
    val mean = 0.1307f
    val std = 0.3081f

    class DataLoader(name: String, train: Boolean, dims: Int*) {

      def open(path: Rep[String]) = uncheckedPure[Int]("open(",path,",0)")
      def filelen(fd: Rep[Int]) = uncheckedPure[Long]("fsize(",fd,")") // FIXME: fresh name
      def mmap[T:Typ](fd: Rep[Int], len: Rep[Long]) = uncheckedPure[Array[T]]("(",codegen.remap(typ[T]),"*)mmap(0, ",len,", PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, ",fd,", 0)")

      val fd = open(s"../data/bin/${name}_${if (train) "train" else "test"}.bin")
      val len = filelen(fd)
      val data = mmap[Float](fd, len)
      val dLength = (len/4L).toInt

      val tfd = open(s"../data/bin/${name}_${if (train) "train" else "test"}_target.bin")
      val tlen = filelen(tfd)
      val target = mmap[Int](tfd, tlen)
      val length = (tlen/4L).toInt

      @virtualize
      def normalize() = {
        this.foreach { (t, d) =>
          t.normalize(mean, std, inPlace = true)
        }
      }


      @virtualize
      def foreach(f: (Tensor, Rep[Int]) => Unit) = {
        var off = var_new(0)
        for (img <- 0 until length: Rep[Range]) {
          val dataPtr = slice(data, off)
          val t = Tensor(dataPtr, dims : _*)
          f(t, target(img))
          off += t.scalarCount
        }
        assertC(off == dLength, "Data length doesn't match\\n")
      }
    }

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
      val varConv1 = TensorR(Tensor.rand(1.0f / sqrt(inChan1 * kRow1 * kCol1).toFloat, outChan1, inChan1, kRow1, kCol1))
      variables += varConv1

      // input size
      val (iChan2, iRow2, iCol2) = (outChan1, convSize(iRow1, kRow1, sRow1)/smRow1, convSize(iCol1, kCol1, sCol1)/smCol1)

      // Layer 2
      val (inChan2, outChan2, kRow2, kCol2, sRow2, sCol2) = (outChan1, 20, 5, 5, 1, 1)

      // stride maxpool
      val (smRow2, smCol2) = (2, 2)

      val varConv2 = TensorR(Tensor.rand(1.0f / sqrt(inChan2 * kRow2 * kCol2).toFloat, outChan2, inChan2, kRow2, kCol2))
      variables += varConv2

      // Layer 3
      val (oRow2, oCol2) = (convSize(iRow2, kRow2, sRow2)/smRow2, convSize(iCol2, kCol2, sCol2)/smCol2)
      val (in3, out3) = (outChan2 * oRow2 * oCol2, 50)  // 320

      val varA1 = TensorR(Tensor.rand(1.0f / sqrt(in3).toFloat, out3, in3))
      val varB1 = TensorR(Tensor.rand(1.0f / sqrt(in3).toFloat, out3))
      variables += varA1
      variables += varB1

      // Layer 4
      val (in4, out4) = (out3, 10)

      val varA2 = TensorR(Tensor.rand(1.0f / sqrt(in4).toFloat, out4, in4))
      val varB2 = TensorR(Tensor.rand(1.0f / sqrt(in4).toFloat, out4))
      variables += varA2
      variables += varB2

      // Training
      val nbEpoch = 10
      val lr = 0.0005f
      val mom = 0.0f

      val momentum = if (mom > 0.0f) variables map(tR => Tensor.zeros(tR.d)) else ArrayBuffer[Tensor]()

      val tot1 = NewArray[Long](2)
      val tot2 = NewArray[Long](2)

      val train = new DataLoader("mnist", true, iChan1, iRow1, iCol1)
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
        var imgIdx = var_new(0)
        var trainLoss = var_new(0.0f)
        printf("Start training epoch %d\\n", epoch + 1)
        trainTimer.startTimer
        train foreach { (input: Tensor, target: Rep[Int]) =>
          imgIdx += 1

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

  test("generate_code_for_mnist_cnn") {
    val cnn_file = new PrintWriter(new File(root_dir + file_dir))
    cnn_file.println(mnist.code)
    cnn_file.flush()
  }

  val mnist2  = new DslDriverC[String, Unit] with NNModule with ScannerLowerExp {

    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {

      Random.srand(Some(42))
      val dataTimer = Timer2()
      dataTimer.startTimer

      case class MNIST(val name: String = "mnist") extends Module {
        val (batch, iChan1, iRow1, iCol1) = (1, 1, 28, 28)
        val conv1 = regModuleWithName("conv1")(Conv2D(inChannel = 1, outChannel = 10, kernelSize = NSeq(5, 5)))
        val conv2 = regModuleWithName("conv2")(Conv2D(inChannel = 10, outChannel = 20, kernelSize = NSeq(5, 5)))
        val linear1 = regModuleWithName("linear1")(Linear1D(inSize = 320, outSize = 50))
        val linear2 = regModuleWithName("linear2")(Linear1D(inSize = 50, outSize = 10))

        def apply(in: TensorR): TensorR @diff = {
          val step1 = conv1(in).relu().maxPoolBK(kernels = NSeq(2,2), strides = NSeq(2,2))
          val step2 = conv2(step1).relu().maxPoolBK(kernels = NSeq(2,2), strides = NSeq(2,2))
          val step3 = linear1(step2.resize(320)).dropout(0.5f)
          linear2(step3)
        }
      }
      val net = MNIST("model")
      val opt = SGD(net, learning_rate = 0.0005f, gradClip = 1000.0f)

      def lossFun(input: TensorR, target: Rep[Int]) = { (dummy: TensorR) =>
        val res = net(input).logSoftmax()
        res.nllLoss(target)
      }

      // Training
      val nbEpoch = 10
      val lr = 0.0005f

      val tot1 = NewArray[Long](2)
      val tot2 = NewArray[Long](2)

      val train = new Dataset.DataLoader("mnist", true, mean = 0.1307f, std = 0.3081f, net.batch, net.iChan1, net.iRow1, net.iCol1)
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
        train foreach { (dummy: Rep[Int], input: Tensor, target: Rep[Int]) =>
          imgIdx += 1
          val inputR = TensorR(input , isInput=true)
          val loss = gradR_loss(lossFun(inputR, target))(Tensor.scalar(0.0f))
          trainLoss += loss.data(0)

          // Update weights
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

  test("mnist_cnn_2") {
    val cnn_file = new PrintWriter(new File(root_dir + file_dir))
    cnn_file.println(mnist2.code)
    cnn_file.flush()
  }
}

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

class MnistCNN extends FunSuite {

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
          off += t.nbElem
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
      val iChan1 = 1
      val iRow1 = 28
      val iCol1 = 28

      //System.out.println(s"Input size: $iChan1 x $iRow1 x $iCol1")

      // TODO create modules
      // Layer 1
      val inChan1 = iChan1
      val outChan1 = 10
      val kRow1 = 5
      val kCol1 = 5

      // stride conv
      val sRow1 = 1
      val sCol1 = 1

      // stride maxpool
      val smRow1 = 2
      val smCol1 = 2

      // FIXME scale based on PyTorch
      val conv1 = Tensor.rand(1.0f / sqrt(inChan1 * kRow1 * kCol1).toFloat, outChan1, inChan1, kRow1, kCol1)
      val varConv1 = TensorR(conv1)
      variables += varConv1

      // input size
      val iChan2 = outChan1
      val iRow2 = convSize(iRow1, kRow1, sRow1)/smRow1
      val iCol2 = convSize(iCol1, kCol1, sCol1)/smCol1

      //System.out.println(s"Layer 1 output size: $iChan2 x $iRow2 x $iCol2")

      // Layer 2
      val inChan2 = outChan1
      val outChan2 = 20
      val kRow2 = 5
      val kCol2 = 5

      // stride conv
      val sRow2 = 1
      val sCol2 = 1

      // stride maxpool
      val smRow2 = 2
      val smCol2 = 2

      val conv2 = Tensor.rand(1.0f / sqrt(inChan2 * kRow2 * kCol2).toFloat, outChan2, inChan2, kRow2, kCol2)
      val varConv2 = TensorR(conv2)
      variables += varConv2

      // Layer 3
      val oRow2 = convSize(iRow2, kRow2, sRow2)/smRow2
      val oCol2 = convSize(iCol2, kCol2, sCol2)/smCol2
      val in3 = 320
      val out3 = 50

      //System.out.println(s"Layer 2 output size: $outChan2 x $oRow2 x $oCol2")

      assert(in3 == outChan2 * oRow2 * oCol2, s"The input of the first Linear layer should be $in3, got ${outChan2 * oRow2 * oCol2}")

      val a1 = Tensor.rand(1.0f / sqrt(in3).toFloat, out3, in3)
      val b1 = Tensor.rand(1.0f / sqrt(in3).toFloat, out3)
      val varA1 = TensorR(a1)
      val varB1 = TensorR(b1)
      variables += varA1
      variables += varB1

      // Layer 4
      val in4 = out3
      val out4 = 10

      val a2 = Tensor.rand(1.0f / sqrt(in4).toFloat, out4, in4)
      val b2 = Tensor.rand(1.0f / sqrt(in4).toFloat, out4)
      val varA2 = TensorR(a2)
      val varB2 = TensorR(b2)
      variables += varA2
      variables += varB2

      // Training
      val nbEpoch = 10
      val lr = 0.0005f
      val mom = 0.0f

      val momentum = if (mom > 0.0f) variables map(tR => Tensor.zeros(tR.d)) else ArrayBuffer[Tensor]()

      val tot1 = NewArray[Long](2)
      val tot2 = NewArray[Long](2)

      //val dataTimer = Timer2()
      //dataTimer.startTimer

      val train = new DataLoader("mnist", true, iChan1, iRow1, iCol1)
      printf("Start normalize\\n")
      train.normalize()

      def trainFun(input: TensorR, target: Rep[Int]) = { (dummy: TensorR) =>
        val resL1 = input.conv(varConv1, sRow1, sCol1, tot1).maxPool(smRow1, smCol1).relu()
        val resL2 = resL1.conv(varConv2, sRow2, sCol2, tot2).maxPool(smRow2, smCol2).relu()
        val resL3 = ((varA1 dot resL2.resize(in3)) + varB1).relu().dropout(0.5f)
        val resL4 = (varA2 dot resL3) + varB2
        val res = resL4.logSoftmax()
        res.nllLoss(target)
      }

      // we skip tests for the experiments
      //val test = new DataLoader("mnist", false, iChan1, iRow1, iCol1)
      //test.normalize()

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
          // assertC(0 <= target && target <= 9, "Target should be a number between 0 and 9, got %d\\n", target)

          val inputR = TensorR(input , isInput=true)
          val loss = gradR_loss(trainFun(inputR, target))(Tensor.scalar(0.0f))
          trainLoss += loss.data(0)

          // for ((weight, idx) <- variables.zipWithIndex) {
          //   weight.print(s"Variable ${idx + 1}", derivative = true)
          // }

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

            // printf("Weight before %.10f -", weight.x.data(0))
            weight.x.addMul(-lr, d)
            // if (weight.x.check(5.0f)) {
            //   printf("Iteration %d\\n", imgIdx)
            //   weight.print(s"Weight of variable ${idx + 1} diverged!!!", derivative = true)
            //   exit()
            // }
            // printf("%.10f weigth after (%.10f - %.5f)\\n", weight.x.data(0), weight.d.data(0), lr)
            weight.clear_grad()
          }

          // for ((weight, idx) <- variables.zipWithIndex) {
          //   weight.print(s"Variable ${idx + 1}")
          // }

          if (imgIdx %  (train.length / 10) == 0) {
            printf(s"Train epoch %d: [%d/%d (%.0f%%)]\\tAverage Loss: %.6f\\n", epoch, imgIdx, train.length, 100.0 * imgIdx /train.length, trainLoss/imgIdx)
            // printf("Conv1 fwd %ld us/image - bwd %ld us/image\\n", tot1(0)/imgIdx, tot1(1)/imgIdx)
            // printf("Conv2 fwd %ld us/image - bwd %ld us/image\\n", tot2(0)/imgIdx, tot2(1)/imgIdx)
            unchecked[Unit]("fflush(stdout)")
          }
          resetMallocAddr(addr)
        }
        val delta = trainTimer.getElapsedTime
        printf("Training completed in %ldms (%ld us/images)\\n", delta/1000L, delta/train.length)

        // save trainLoss / train.length to loss_save
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
        //printf("loss_saver is %lf \\n", loss_save(i))
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

}

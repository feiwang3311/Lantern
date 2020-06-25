package lantern

import scala.util.continuations._

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.{Map => MutableMap}
import scala.math._

import lms.core.stub._
import lms.macros.SourceContext
import lms.core.virtualize

trait Dataset extends TensorDsl with ScannerOps with TimerOps {

  class DataLoader(name: String, train: Boolean, mean: Float, std: Float, dims: Seq[Int]) {

    val fd = open(s"../data/bin/${name}_${if (train) "train" else "test"}.bin")
    val len = filelen(fd)
    val data = mmap[Float](fd, len)
    val dLength = (len/4L).toInt

    val tfd = open(s"../data/bin/${name}_${if (train) "train" else "test"}_target.bin")
    val tlen = filelen(tfd)
    val target = mmap[Int](tfd, tlen)
    val length: Rep[Int] = tlen.toInt/4

    def dataset = new Tensor(data, Seq(60000, dims(1), dims(2)))

    @virtualize
    def normalize() = {
      this.foreach { (i, t, d) =>
        t.normalize(mean, std, inPlace = true)
      }
    }

    @virtualize
    def foreach(f: (Rep[Int], Tensor, Rep[Int]) => Unit) = {
      var off = var_new(0)
      for (index <- 0 until length: Rep[Range]) {
        val dataPtr = sliceRead(data, off)
        val t = Tensor(dataPtr, dims : _*)
        f(index, t, target(index))
        off += t.scalarCount
      }
      assertC(off == dLength, "Data length doesn't match\\n")
    }

    @virtualize
    def foreachBatch(batchSize: Int)(f: (Rep[Int], Tensor, Rep[Array[Int]]) => Unit) = {
      var off = var_new(0)
      for (batchIndex <- 0 until (length / batchSize): Rep[Range]) {
        val dataPtr = sliceRead(data, off)
        val t = Tensor(dataPtr, (batchSize +: dims.toSeq): _*)
        val targets = sliceRead(target, batchIndex * batchSize)
        f(batchIndex, t, targets)
        off += t.scalarCount
      }
    }
  }

  class Cifar10DataLoader(name: String, train: Boolean, dims: Seq[Int]) {

    val fd = open(name)
    val len = filelen(fd)
    val data = mmap[Char](fd, len)
    // each entry is target + image
    val entrySize = (dims.product + 1)
    val dLength = (len/entrySize.toLong).toInt
    val length = dLength

    val x = NewArray[Float](dLength * dims.product)
    val y = NewArray[Int](dLength)

    for (i <- (0 until dLength): Rep[Range]) {
      y(i) = unchecked[Int]("(int32_t)(unsigned char)", data(i * entrySize))
      for (j <- (0 until dims.product): Rep[Range]) {
        x(i * dims.product + j) = uncheckedPure[Float]("(float)(unsigned char)", data(i * entrySize + 1 + j)) / 255.0f
      }
    }

    @virtualize
    def foreachBatch(batchSize: Int)(f: (Rep[Int], Tensor, Rep[Array[Int]]) => Unit) = {
      for (batchIndex <- 0 until (dLength / batchSize): Rep[Range]) {
        val dataPtr = slice(x, batchIndex * batchSize * dims.product)
        val targets = slice(y, batchIndex * batchSize)
        val t = Tensor(dataPtr, (batchSize +: dims.toSeq): _*)
        f(batchIndex, t, targets)
      }
    }
  }

  @virtualize
  class DeepSpeechDataLoader(name: String, train: Boolean) {

    // open file
    val fd = open(name)
    val len = filelen(fd)
    printf("file size is %ld\\n", len)

    val data = mmap[Char](fd, len)
    object reader {
      val pointer = var_new(unchecked[Long]("(long)", data))
      def nextI(size: Rep[Int] = 1): Rep[Array[Int]] = {
        val temp: Rep[Long] = pointer
        val intArray = unchecked[Array[Int]]("(int32_t*) ", temp)
        pointer += 4 * size
        intArray
      }
      def nextInt(): Rep[Int] = nextI()(0)
      def nextF(size: Rep[Int] = 1): Rep[Array[Float]] = {
        val temp: Rep[Long] = pointer
        val floatArray = unchecked[Array[Float]]("(float*) ", temp)
        pointer += 4 * size
        floatArray
      }
    }

    // get batchSize and numBatches
    val batchSize = reader.nextInt  // batchSize is 32, and numBatches is 5
    val num_Batches = reader.nextInt
    val numBatches = 200
    val length = batchSize * numBatches
    printf("data size is %d batches, %d batch size\\n", numBatches, batchSize)

    // get array to store information for each batch
    val freqSizes: Rep[Array[Int]] = NewArray[Int](numBatches)
    val maxLengths: Rep[Array[Int]] = NewArray[Int](numBatches)
    // get array of arrays to store the pointers to data
    val inputs: Rep[Array[Array[Float]]] = NewArray[Array[Float]](numBatches)
    val percents: Rep[Array[Array[Float]]] = NewArray[Array[Float]](numBatches)
    // val inputSizes: Rep[Array[Array[Int]]] = NewArray[Array[Int]](numBatches)
    // val inputs = NewArray[Tensor](numBatches)
    // val percents = NewArray[Tensor](numBatches)
    val targetSizes: Rep[Array[Array[Int]]] = NewArray[Array[Int]](numBatches)
    val targets: Rep[Array[Array[Int]]] = NewArray[Array[Int]](numBatches)

    generate_comment("load data by batchs")
    for (batch <- (0 until numBatches: Rep[Range])) {
      // First, get frequency_size and max_length
      freqSizes(batch) = reader.nextInt  // freqSize is 161, and maxLength is 229
      maxLengths(batch) = reader.nextInt
      // then the sound tensor of float [batchSize * 1 * freqSize * maxLength]
      inputs(batch) = reader.nextF(batchSize * freqSizes(batch) * maxLengths(batch))
      // then the percentage tensor of float [batchSize] (percentage of padding for each sound)
      percents(batch) = reader.nextF(batchSize)

      // then the targetSize tensor of Int[batchSize]
      targetSizes(batch) = reader.nextI(batchSize)
      val sumTargetSize: Rep[Int] = unchecked[Int]("accumulate(", targetSizes(batch), ", ", targetSizes(batch), " + ", batchSize, ", 0)")
      // then the targets tensor of Int[sum(targetSize)]
      targets(batch) = reader.nextI(sumTargetSize)
    }

    @virtualize
    // the lossFun takes a Batch (Tensor), inputLengths, labels, labelLengths (all Rep[Array[Int]])
    def foreachBatch(f: (Rep[Int], Tensor, Rep[Array[Float]], Rep[Array[Int]], Rep[Array[Int]]) => Unit) = {
      for (batchIndex <- 0 until numBatches: Rep[Range]) {
        val maxLength = maxLengths(batchIndex)
        val freqSize = freqSizes(batchIndex)
        val input: Tensor = Tensor(inputs(batchIndex), batchSize, 1, freqSize, maxLength)
        val percent: Rep[Array[Float]] = percents(batchIndex)
        val target: Rep[Array[Int]] = targets(batchIndex)
        val targetSize: Rep[Array[Int]] = targetSizes(batchIndex)
        f(batchIndex, input, percent, target, targetSize)
      }
    }
  }
}

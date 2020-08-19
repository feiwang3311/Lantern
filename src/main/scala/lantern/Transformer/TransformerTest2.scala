package lantern
package Transformer

import lms.core.stub._
import lms.macros.SourceContext
import lms.core.virtualize

import scala.sys.process._

import java.io.PrintWriter;
import java.io.File;

object TransformerTest2 {

  val driver = new LanternDriverCudnn[String, Unit] with TimerOpsExp {

    @virtualize
    class TranslationDataLoader(name: String) {

      // open file
      val fd = open(name)
      val len = filelen(fd)
      printf("file size is %ld\\n", len)

      val raw_data = mmap[Char](fd, len)

      object reader {
        val pointer = var_new(unchecked[Long]("(long)", raw_data))

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

      val numElems = reader.nextInt() // Number of data elements
      val numBatches = reader.nextInt()
      val batchSize = reader.nextInt() // assumes there's no last batch with size < batch_size (in the rest of the processing)
      val vocabSize = reader.nextInt()
      val maxSeqLen = reader.nextInt()

      val srcLens = NewArray[Int](numBatches)
      val tgtLens = NewArray[Int](numBatches)
//      val testtt = numElems - numBatches * batchSize
      val data = NewArray[Int](numElems - numBatches * batchSize) // - numBatches * batchSize because we remove the end token from tgt
      // to keep track of pos in data array
      val srcBatch = NewArray[Int](numBatches)
      val tgtBatch = NewArray[Int](numBatches)

      val dataPos = var_new[Int](0)
      val tgtLensSum = var_new[Int](0)

      val END_TOKEN = 3
      val PADDING_TOKEN = 1
      val START_TOKEN = 1

      for(b <- 0 until numBatches: Rep[Range]) {
        val srcLen = reader.nextInt()
        srcLens(b) = srcLen
        srcBatch(b) = dataPos
        // batch is the fastest moving dimension
        // copy as it is because our models operate on Tensor shape seqLen x batchSize
        for(i <- 0 until srcLen: Rep[Range]) {
          for(j <- 0 until batchSize: Rep[Range]) {
            data(dataPos) = reader.nextInt()
            dataPos += 1
          }
        }

        val tgtLen = reader.nextInt() - 1
        tgtLensSum += tgtLen
        tgtLens(b) = tgtLen
        tgtBatch(b) = dataPos
        for(j <- 0 until tgtLen: Rep[Range]) {
          for(i <- 0 until batchSize: Rep[Range]) {
            data(dataPos) = reader.nextInt()
            dataPos += 1
          }
        }

        for(i <- 0 until batchSize: Rep[Range]) {
          val temp = reader.nextInt()
          assertC(temp != PADDING_TOKEN || temp != END_TOKEN, s"last token id must be either padding (1) or end token (2) but got ${temp}")
        }
      }

      // create the target (target output of the decoder - used in computing loss)
      // this is tgt sequence shifted to right (i.e., without the starting token)
      // shape = numBatches x batchSize x tgtLen (tgtLen is not the same for all)
      val target = NewArray[Int](tgtLensSum * batchSize * numBatches)
      val targetBatch = NewArray[Int](numBatches) // to locate the batch elems in target
      val pos = var_new[Int](0)
      for(b <- 0 until numBatches: Rep[Range]) {
        targetBatch(pos) = pos
        for(i <- 0 until batchSize: Rep[Range]) {
          for(j <- 0 until tgtLens(b)-1: Rep[Range]) {
            target(pos) = data(tgtBatch(b) + (j+1) * batchSize + i) // j + 1 because skipping the start token
            pos += 1
          }
          target(pos) = if (target(pos - 1) == END_TOKEN || target(pos - 1) == PADDING_TOKEN) PADDING_TOKEN else END_TOKEN
          pos += 1
        }
      }

      @virtualize
      def forEachBatch(f: (Rep[Array[Int]], Rep[Array[Int]], Rep[Int], Rep[Int], Rep[Array[Int]]) => Unit): Unit = {
        // TODO - make this until numBatches
        for(i <- 0 until numBatches: Rep[Range]) {
          val srcGPU = sliceRead(data, srcBatch(i)).toGPU(batchSize * srcLens(i))
          val tgtGPU = sliceRead(data, tgtBatch(i)).toGPU(batchSize * tgtLens(i))
          val targetGPU = sliceRead(target, targetBatch(i)).toGPU(batchSize * tgtLens(i))

          val srcLen = srcLens(i)
          val tgtLen = tgtLens(i)
          f(srcGPU, tgtGPU, srcLen, tgtLen, targetGPU)
        }
      }

    }

    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {
      val dataPath = "/u/ml00_s2/tabeysin/data/translation/dataset.bin"
      val dataLoader = new TranslationDataLoader(dataPath)

      val batchSize = dataLoader.batchSize
      val embedDim = 256
      val maxSeqLen = dataLoader.maxSeqLen
      val nheads = 8
      val numEncoderLayers = 6
      val numDecoderLayers = 6
      val dimFeedForward = 512
      val dropOut = 0.1f
      val vocabSize = 9521
      assertC(vocabSize == dataLoader.vocabSize, "vocab sizes do not match")

      // positional embeddings
      // create a cpu array containing positional embeddings (for a given max len)
      // use slices of that and create tensors (move to GPU) and use it in the forward pass
      // (not performance critical since this happens in initialization (just once)
      val posEmbeddingsCPU = NewArray[Float](maxSeqLen * embedDim)
      for (pos <- 0 until maxSeqLen: Rep[Range]; i<- 0 until embedDim: Rep[Range]) {
        val floorVal = i / 2
        if (i % 2 == 0)
          posEmbeddingsCPU(pos * embedDim + i) = Math.sin(pos * Math.exp(-floorVal * Math.log(10000) / embedDim)).toFloat
        else
          posEmbeddingsCPU(pos * embedDim + i) = Math.cos(pos * Math.exp(-floorVal * Math.log(10000) / embedDim)).toFloat
      }

      val posEmbeddingsGPU = posEmbeddingsCPU.toGPU(embedDim * maxSeqLen)

      case class Model(name: String = "transformer-seq2seq") extends Module {
        // the word embeddings should be model parameters as well
        // embeddings are shared (encoder and decoder)
        val embedding = Embedding(vocabSize, embedDim, dataLoader.PADDING_TOKEN)
        val transformer = Transformer(embedDim, -1, nheads, numEncoderLayers, numDecoderLayers, dimFeedForward, dropOut)
        val linear = Linear1D(inSize = embedDim, outSize = vocabSize, bias = false)

        // should take two Tensors with word ids, but embedding layer is not implemented yet
        def apply(srcIdx: Rep[Array[Int]], tgtIdx: Rep[Array[Int]], srcShape: Seq[Rep[Int]], tgtShape: Seq[Rep[Int]]) = {
//          printf("src shape (%d, %d)\n", srcShape.head, srcShape.last)
//          printf("tgt shape (%d, %d)\n", tgtShape.head, tgtShape.last)

          val srcEmb = embedding(srcIdx, srcShape)
          val tgtEmb = embedding(tgtIdx, tgtShape)

          val srcPosEmbeddings = TensorR(Tensor(posEmbeddingsGPU, srcShape.head, 1, 1), isInput = true)
          val tgtPosEmbeddings = TensorR(Tensor(posEmbeddingsGPU, tgtShape.head, 1, 1), isInput = true)

          // mask - tgtLen x tgtLen
          val mask = NewGPUArray[Int](tgtShape.head * tgtShape.head)
          val blockSize = (tgtShape.head + 31) / 32
          create_attention_mask(mask, blockSize, tgtShape.head, tgtShape.head)

          val src = srcEmb + srcPosEmbeddings
          val tgt = tgtEmb + tgtPosEmbeddings

          val tgtLen = tgt.x.shape(0)
          val batchSize = tgt.x.shape(1)

          val result = transformer(src, tgt, Some(mask))
          val temp = linear(result.resizeNoCheck(tgtLen * batchSize, embedDim))
          temp.logSoftmax_v2(1).resizeNoCheck(tgtLen, batchSize, vocabSize)
        }
      }

      val model = Model()

      def lossFun(srcIdx: Rep[Array[Int]], tgtIdx: Rep[Array[Int]], srcShape: Seq[Rep[Int]], tgtShape: Seq[Rep[Int]], target: Rep[Array[Int]]) = { (batchIndex: TensorR) =>
        val res = model(srcIdx, tgtIdx, srcShape, tgtShape)
        val loss = res.resizeNoCheck(res.x.shape(0) * res.x.shape(1), res.x.shape(2)).nllLossB(target)
        loss.sum()
      }

//      val opt = SGD(model, learning_rate = 0.01f, gradClip = 1.0f)
      val opt = Adagrad(model, learning_rate = 0.01f, gradClip = 1.0f)

      val numEpochs = 3

      for(i <- 0 until numEpochs: Rep[Range]) {
        val trainTimer = Timer2()
        trainTimer.startTimer
        val totalLoss = var_new[Float](0)
        val addr = getMallocAddr()
        val addrCuda = getCudaMallocAddr()
        dataLoader.forEachBatch{ (srcIdx: Rep[Array[Int]], tgtIdx: Rep[Array[Int]], srcLen: Rep[Int], tgtLen: Rep[Int], target: Rep[Array[Int]]) => {
//          printf("begin mini batch\n")
            val loss = gradR_loss(lossFun(srcIdx, tgtIdx, Seq(srcLen, batchSize), Seq(tgtLen, batchSize), target))(Tensor.zeros(4))
            totalLoss += loss.toCPU().data(0)
//            printf("loss (mini batch) = %f\n", loss.toCPU().data(0))
            opt.step()
            resetMallocAddr(addr)
            resetCudaMallocAddr(addrCuda)
//          printf("mini batch end\n")
          }
        }
        unchecked[Unit]("cudaDeviceSynchronize()")
        val delta = trainTimer.getElapsedTime
        printf("Training iter done in %ldms \\n", delta / 1000L)
        printf("loss = %f\n", totalLoss)
      }
    }
  }

  def main(args: Array[String]): Unit = {
    import sys.process._
    val code_file = new PrintWriter(new File("src/out/Transformers/Lantern/transformer2.cu"))
    code_file.println(driver.code)
    code_file.flush()

    val logger = ProcessLogger(
      (o: String) => println("out " + o),
      (e: String) => println("err " + e))

    "nvcc src/out/Transformers/Lantern/transformer2.cu -o  src/out/Transformers/Lantern/transformer2 -Isrc/main/cpp/headers/ -I../lms-clean/src/main/scala/lms/thirdparty/thirdpartyAdaptor/ -lcuda -lcublas -lcudnn" ! logger;
    "./src/out/Transformers/Lantern/transformer2 q" ! logger;
  }
}
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

object SoundRead {

  val root_dir = "src/out/PLDI19evaluation/"
  val cpu_file_dir = "deepspeech2/lantern/Lantern.cpp"
  val gpu_file_dir = "deepspeech2/lantern/Lantern.cu"
  // TODO: Specify data directory.
  val data_dir: String = "../deepspeech_train.bin"

  val deepspeechCPU = new LanternDriverC[String, Unit] {

    @virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {
      val train = new Dataset.DeepSpeechDataLoader(data_dir, true)
      printf("batchSize is %d, and numBatches is %d\\n", train.batchSize, train.numBatches)
      printf("freqSize is %d, and maxLength is %d\\n", train.freqSizes(0), train.maxLengths(0))
      printf("freqSize2 is %d, and maxLength2 is %d\\n", train.freqSizes(1), train.maxLengths(1))
    }
  }

  def main(args: Array[String]) {
    val cpu_file = new PrintWriter(new File(root_dir + cpu_file_dir))
    cpu_file.println(deepspeechCPU.code)
    cpu_file.flush()
    // val gpu_file = new PrintWriter(new File(root_dir + gpu_file_dir))
    // gpu_file.println(deepspeechGPU.code)
    // gpu_file.flush()
  }
}
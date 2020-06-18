package lantern

import lms.core.stub._
import lms.macros.SourceContext
import lms.core.virtualize

import scala.sys.process._

import java.io.PrintWriter;
import java.io.File;

object Transformer {

    val driver = new LanternDriverCudnn[String, Unit] with ScannerOpsExp with TimerOpsExp {
        @virtualize
        def snippet(a: Rep[String]): Rep[Unit] = {
            case class MultiheadAttention(val embedDim: Int, val numHeads: Int, kDim: Int, vDim: Int, val bias: Boolean = false, val dropOut: Float = 0.0f, val name:String ="MultiHeadAttn") extends Module {
                // note - pytorch: all q, k, v is transormed to embedDim size. embedDim = numHeads * h_i(size). Therefore embedDim should be divisible by numHeads
                // attention mask (loWinIdx, hiWinIdx)
                // attention descriptor
                // k,q,v,o desc

                // weights of attention model
                // TODO - would be better if we can get the sizeWeights from cudnn side
                val sizeWeights = embedDim * kDim + embedDim * kDim + embedDim * vDim + {if (bias) 0 else embedDim * 3}
                val weights: TensorR = TensorR(Tensor.rand(sizeWeights)).toGPU()

                def apply(query: TensorR, key: TensorR, value: TensorR) = {
                    // [T(time) N(batch) B(beamsize) V(vector-embed)]
                    // TODO - take attn_mask (i.e. loWinIdx, hiWinIdx) as arg
                    // assert rank 4
                    // set up attention window without mask
                    val loWinIdx = NewArray[Int](query.x.shape(1))
                    val hiWinIdx = NewArray[Int](query.x.shape(1))
                    val qSeqArray = NewArray[Int](query.x.shape(1))
                    val kSeqArray = NewArray[Int](key.x.shape(1))
                    
                    for(i <- 0 until query.x.shape(1)) {
                        loWinIdx(i) = 0
                        hiWinIdx(i) = query.x.shape(0)
                        qSeqArray(i) = query.x.shape(0)
                        kSeqArray(i) = key.x.shape(0)
                    }

                    query.multiheadAttention(key, value, weights, numHeads, embedDim, qSeqArray, kSeqArray, loWinIdx, hiWinIdx, bias, dropOut, 1.0f)
                }
            }

            

            // model
            // requirements: qsize = ksize and vsize * numHeads = embedDim
            val qsize = 50
            val ksize = 50
            val vsize = 50
            val embedDim = 100
            val numHeads = 2
            val batchSize = 10
            val beamSize = 1
            val seqLen = 4 // both klen and qlen
            val dropOut = 0.1f

            case class Model(val name: String = "test_model") extends Module {
                val mha = MultiheadAttention(embedDim, numHeads, ksize, vsize, true, dropOut)
                val linear = Linear1D(inSize = embedDim * seqLen, outSize = 1)

                def apply(q: TensorR, k: TensorR, v: TensorR) = {
                    val step1 = mha(q, k, v)
                    linear(step1.permute(1, 2, 0, 3).resize(-1, q.x.shape(0) * q.x.shape(3)))
                }
            }

            val model = Model()

            val q = TensorR(Tensor.rand(Seq(seqLen, batchSize, beamSize, qsize) : _*)).toGPU()
            val k = TensorR(Tensor.rand(Seq(seqLen, batchSize, beamSize, ksize) : _*)).toGPU()
            val v = TensorR(Tensor.rand(Seq(seqLen, batchSize, beamSize, vsize) : _*)).toGPU()
            
            val opt = SGD(model, learning_rate = 0.0005f, gradClip = 1000.0f)


            def lossFun(query: TensorR, key: TensorR, value: TensorR) = { (batchIndex:TensorR) =>
                val res = model(query, key, value)
                res.sum()
            }

            
            val num_iter = 5
            for(i <- 0 until num_iter: Rep[Range]) {
                val loss = gradR_loss(lossFun(q, k, v))(Tensor.zeros(4))
                opt.step()
                printf("loss = %f\n", loss.toCPU().data(0))
            }

        }
    }

    // def compile: Int = {
    //     val io = new ProcessIO(
    //     _.close, // stdin
    //     stdout => try { Source.fromInputStream(stdout).getLines.foreach(System.out.println(_)) } finally { stdout.close },
    //     stderr => try { Source.fromInputStream(stderr).getLines.foreach(System.out.println(_)) } finally { stderr.close }
    //     )

    //     // Compile query
    //     val sources = "multihead_lantern.cu"
    //     val executableName = "lantern_out.o"
    //     val libraryFlags = Seq("-lcuda", "-lcublas", "-lcudnn") mkString(" ")
    //     val compilerFlags = Seq("-std=c++11", "--expt-extended-lambda") mkString(" ")
    //     val cmd = s"nvcc $compilerFlags $sources -o $executableName $libraryFlags"
    //     infoCompileTime(s"Compilation command: $cmd")
    //     val proc = cmd.run(io)
    //     proc.exitValue
    // }

    def main(args :Array[String]) = {
        val code_file = new PrintWriter(new File("src/out/Transformers/Lantern/multihead_lantern.cu"))
        code_file.println(driver.code)
        code_file.flush()
    }
}
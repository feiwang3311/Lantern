package scala.virtualization.lms
package common

import org.scala_lang.virtualized.SourceContext

trait GPUOps extends ArrayOps {
  object NewGPUArray {
    // Allocate an array of the specified size on the GPU.
    def apply[T: Manifest](scalarCount: Rep[Int]): Rep[Array[T]] = gpu_array_new(scalarCount)
  }
  object GPUArray {
    def apply[T: Manifest](xs: Rep[T]*) = gpu_array_fromseq(xs)
  }
  def gpu_array_new[T: Manifest](scalarCount: Rep[Int]): Rep[Array[T]]
  def gpu_array_fromseq[T: Manifest](xs: Seq[Rep[T]]): Rep[Array[T]]
}

trait GPUOpsExp extends ArrayOpsExp {
  case class GPUArrayNew[T: Manifest](scalarCount: Rep[Int]) extends Def[Array[T]] {
    val m = manifest[T]
  }
  case class GPUArrayFromSeq[T: Manifest](xs: Seq[Exp[T]]) extends Def[Array[T]] {
    val m = manifest[T]
    val mArray = manifest[Array[T]]
  }
  def gpu_array_new[T: Manifest](scalarCount: Exp[Int]) = reflectMutable(GPUArrayNew(scalarCount))
  def gpu_array_fromseq[T: Manifest](xs: Seq[Rep[T]]): Rep[Array[T]] = /*reflectMutable(*/ GPUArrayFromSeq(xs) /*)*/
}

trait CudaGenGPUOps extends CGenBase {
  val IR: GPUOpsExp
  import IR._

  // Allocate GPU memory.
  def getCudaMallocString(buffer: String, count: String, dataType: String): String =
    "CUDA_CALL(cudaMalloc((void **)&" + buffer + ", " + count + " * sizeof(" + dataType + ")))"

  // Allocate GPU memory from memory arena.
  def getCudaMallocArenaString(count: String, dataType: String): String =
    "(" + dataType + "*)myGpuMalloc(" + count + " * sizeof(" + dataType + "))"

  // Copy an array from host to device.
  def cudaMemcpyHostToDevice(dest: String, src: String, count: String, dataType: String): String =
    s"CUDA_CALL(cudaMemcpy($dest, $src, $count * sizeof($dataType), cudaMemcpyHostToDevice))"

  // Copy an array from device to host.
  def cudaMemcpyDeviceToHost(dest: String, src: String, count: String, dataType: String): String =
    s"CUDA_CALL(cudaMemcpy($dest, $src, $count * sizeof($dataType), cudaMemcpyDeviceToHost))"

  // Allocate unified memory, accessible by CPU and GPU.
  // FIXME: I encountered "bus error" when performing CPU ops on managed memory:
  //     Thread 1 "snippet" received signal SIGBUS, Bus error.
  //     Snippet (x0=<optimized out>) at snippet.cpp:144
  //     144	float x32 = x30 - x31;
  // I wonder if others can replicate this issue.
  def getCudaMallocManagedString(buffer: String, count: String, dataType: String): String =
  "CUDA_CALL(cudaMallocManaged((void **)&" + buffer + ", " + count + " * sizeof(" + dataType + ")));"

  override def emitNode(sym: Sym[Any], rhs: Def[Any]) = rhs match {
    case a@GPUArrayNew(n) =>
      val dataType = remap(a.m)
      emitValDef(sym, getCudaMallocArenaString(quote(n), dataType))
    case a@GPUArrayFromSeq(xs) =>
      val dataType = remap(a.m)
      val count = xs.length.toString
      // Allocate GPU array.
      emitValDef(sym, getCudaMallocArenaString(count, dataType))
      // Initialize CPU array.
      val src = fresh(a.mArray)
      emitNode(src, ArrayFromSeq(xs)(a.m))
      // Copy CPU array to GPU array.
      stream.println(cudaMemcpyHostToDevice(quote(sym), quote(src), count, dataType))
    case _ => super.emitNode(sym,rhs)
  }
}

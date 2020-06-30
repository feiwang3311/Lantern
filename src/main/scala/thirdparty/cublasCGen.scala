package lantern.thirdparty

import lms.core._
import lms.util._
import lms.core.stub._
import lms.core.Backend._
import lms.core.virtualize
import lms.core.utils.time
import lms.macros.SourceContext
import lms.thirdparty._

import lantern.collection.mutable.{StackArrayOps}

trait CCodeGenCuBLAS extends ExtendedCCodeGen with CudaCodeGenLibFunction {

  override def remap(m: Manifest[_]): String = m.runtimeClass.getName match {
    case s: String if s.endsWith("SizeT") => "size_t"
    case s: String if s.endsWith("Dim3") => "dim3"
    case _ => super.remap(m)
  }

  override def shallow(n: Node): Unit = n match {
    case Node(s, "myGpuMalloc-f", List(size), _) =>
      emit(s"(${remap(typeMap.getOrElse(s, manifest[Unknown]))})myGpuMalloc("); shallow(size); emit(")")
    case Node(s, "myGpuFree-f", List(size), _) =>
      emit("myGpuFree("); shallow(size); emit(")")


    case Node(s, "cudnn-finalize", _, _) =>
      emit("CUDNN_FINALIZE()")
    case _ => super.shallow(n)
  }

}

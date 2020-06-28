package lantern.collection.mutable

import lms.core._
import lms.util._
import lms.core.stub._
import lms.core.Backend._
import lms.core.virtualize
import lms.core.utils.time
import lms.macros.SourceContext
import lms.collection.mutable.{ArrayOps}

trait StackArrayOps extends ArrayOps { b: Base =>

  def NewStackArray[T:Manifest](x: Rep[Int]): Rep[Array[T]] = {
    Wrap[Array[T]](Adapter.g.reflectMutable("NewStackArray", Unwrap(x)))
  }

  def StackArray[T:Manifest](xs: Rep[T]*): Rep[Array[T]] = {
    Wrap[Array[T]](Adapter.g.reflectMutable("StackArray", xs.map(Unwrap(_)): _*))
  }
}

trait CCodeGenStackArray extends ExtendedCCodeGen {

  override def traverse(n: Node): Unit = n match {
    case n @ Node(s, "NewStackArray", List(x), _) =>
      val tpe = remap(typeMap.get(s).map(_.typeArguments.head).getOrElse(manifest[Unknown]))
      emit(s"$tpe "); shallow(s); emit("["); shallow(x); emitln("];")
    case n @ Node(s, "StackArray", xs, _) =>
      val tpe = remap(typeMap.get(s).map(_.typeArguments.head).getOrElse(manifest[Unknown]))
      emit(s"$tpe "); shallow(s); emit("[] = {"); shallow(xs.head)
      xs.tail.foreach(x => {emit(", "); shallow(x)}); emitln("};")
    case _ => super.traverse(n)
  }

  override def mayInline(n: Node): Boolean = n match {
    case Node(s, "NewStackArray", _, _) => false
    case Node(s, "StackArray", _, _) => false
    case _ => super.mayInline(n)
  }
}

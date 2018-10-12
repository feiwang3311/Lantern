package scala.virtualization.lms
package common

import util.OverloadHack

import org.scala_lang.virtualized.SourceContext

// TODO: clean up at least, maybe add to LMS?
trait UtilOps extends Base with OverloadHack {

  implicit class HashCls[T:Manifest](o: Rep[T]) {
    def HashCode(implicit pos: SourceContext): Rep[Long] = infix_HashCode(o)
    def HashCode(len: Rep[Int])(implicit pos: SourceContext): Rep[Long] = o match {
      case s: Rep[String] => infix_HashCodeS(s, len)
      // case _ => infix_HashCode(o) // FIXME: is this an ok dispatch?
    }
  }

  def infix_HashCode[T:Manifest](a: Rep[T])(implicit pos: SourceContext): Rep[Long]
  def infix_HashCodeS(s: Rep[String], len: Rep[Int])(implicit v: Overloaded1, pos: SourceContext): Rep[Long]
}

trait UtilOpsExp extends UtilOps with BaseExp {
  case class ObjHashCode[T:Manifest](o: Rep[T])(implicit pos: SourceContext) extends Def[Long] { def m = manifest[T] }
  case class StrSubHashCode(o: Rep[String], len: Rep[Int])(implicit pos: SourceContext) extends Def[Long]
  def infix_HashCode[T:Manifest](o: Rep[T])(implicit pos: SourceContext) = ObjHashCode(o)
  def infix_HashCodeS(o: Rep[String], len: Rep[Int])(implicit v: Overloaded1, pos: SourceContext) = StrSubHashCode(o,len)

  override def mirror[A:Manifest](e: Def[A], f: Transformer)(implicit pos: SourceContext): Exp[A] = (e match {
    case e@ObjHashCode(a) => infix_HashCode(f(a))(e.m, pos)
    case e@StrSubHashCode(o,len) => infix_HashCodeS(f(o),f(len))
    case _ => super.mirror(e,f)
  }).asInstanceOf[Exp[A]]
}

trait ScalaGenUtilOps extends ScalaGenBase {
  val IR: UtilOpsExp
  import IR._

  override def emitNode(sym: Sym[Any], rhs: Def[Any]) = rhs match {
    case ObjHashCode(o) => emitValDef(sym, src"$o.##")
    case _ => super.emitNode(sym, rhs)
  }
}

trait CGenUtilOps extends CGenBase {
  val IR: UtilOpsExp
  import IR._

  override def emitNode(sym: Sym[Any], rhs: Def[Any]) = rhs match {
    case StrSubHashCode(o,len) => emitValDef(sym, src"hash($o,$len)")
    case _ => super.emitNode(sym, rhs)
  }
}

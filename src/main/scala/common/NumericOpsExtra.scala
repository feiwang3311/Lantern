package scala.virtualization.lms
package common

import org.scala_lang.virtualized.SourceContext

trait NumericOpsExtra extends NumericOps {

  this: PrimitiveOps =>

  implicit def numericToNumericOpsExtra[T:Numeric:Manifest](n: Rep[T]) = new NumericOpsExtraCls(n)

  class NumericOpsExtraCls[T:Numeric:Manifest](lhs: Rep[T]) {
    def toDouble(implicit pos: SourceContext) = numeric_toDouble(lhs)
  }

  def numeric_toDouble[T:Numeric:Manifest](lhs: Rep[T])(implicit pos: SourceContext): Rep[Double]

}

trait NumericOpsExtraExp extends NumericOpsExp with NumericOpsExtra {
  this: PrimitiveOpsExp =>

  case class NumericToDouble[T:Numeric:Manifest](lhs: Exp[T]) extends DefMN[Double]

  def numeric_toDouble[T:Numeric:Manifest](lhs: Rep[T])(implicit pos: SourceContext): Exp[Double] = NumericToDouble(lhs)

  // override def mirror[A:Manifest](e: Def[A], f: Transformer)(implicit pos: SourceContext): Exp[A] = (e match {
  //   case e@NumericToDouble(l) => numeric_toDouble(f(l))(e.aev.asInstanceOf[Numeric[A]], mtype(e.mev), pos)
  //   case _ => super.mirror(e,f)
  // }).asInstanceOf[Exp[A]]
}

trait CLikeGenNumericOpsExtra extends CLikeGenBase {
  val IR: NumericOpsExtraExp
  import IR._

  override def emitNode(sym: Sym[Any], rhs: Def[Any]) = {
    rhs match {
      case NumericToDouble(a) =>
        emitValDef(sym, src"(double)$a")
      case _ => super.emitNode(sym, rhs)
    }
  }
}

trait CGenNumericOpsExtra extends CGenBase with CLikeGenNumericOpsExtra

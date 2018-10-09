package lantern

import scala.collection.{Seq => NSeq}

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

trait TestExp extends DslOps {
  type Size = Int

  object Size {
    def zero: Size = 0
  }

  abstract class DataLoop {
    def foreach(f: Rep[Int] => Unit): Unit
  }

  @virtualize
  object DataLoop {
    def apply(size: Int) = if (size <= 1) {
      new DataLoop {
        def foreach(f: Rep[Int] => Unit) = {
          for (i <- 0 until size: Range) f(unit(i))
        }
      }
    } else {
      new DataLoop {
        def foreach(f: Rep[Int] => Unit) = {
          for (i <- 0 until size: Rep[Range]) f(i)
        }
      }
    }
  }

  class Dimensions(val dims: NSeq[Size]) {
    def apply(idx: Int) = dims(idx)

    val (nbElem +: strides) = (dims :\ NSeq[Int](1)) {
      case (dim, seq@(t +: q)) => (dim * t) +: seq
    }

    override def toString = dims mkString " x "
    override def equals(o: Any) = o match {
      case t: Dimensions => this.dims == t.dims
      case _ => false
    }
  }

  object Dimensions {
    def apply(x: Size*) = new Dimensions(x)
  }

  val debug: Rep[Boolean] = false
  class Test[T:Ordering:Numeric:Typ](val data: Rep[Array[T]], val dims: Dimensions) {

    val nbElem = dims.nbElem

    def exit() = unchecked[Unit]("exit(0)")

    def format = typ[T] match {
      case t if t == typ[Int] => "%d "
      case t if t == typ[Float] => "%.3f "
      case t if t == typ[Double] => "%.3f "
    }
    val zero = implicitly[Numeric[T]].zero
    val one = implicitly[Numeric[T]].one

    val num = implicitly[Numeric[T]]
    def log(x: T) = Math.log(num.toDouble(x))

    @virtualize
    def assertC(b: Rep[Boolean], msg: String, x: Rep[Any]) = {
      if (debug && !b) { printf(s"Assert failed $msg\\n", x); exit() }
    }

    @virtualize
    def apply(x: Rep[Size]*) = {
      // Fei: should we make sure that length of x is the same as length of Dimensions 
      val idx: Rep[Size] = ((x zip (if (x.length == 1) NSeq(1) else dims.strides)) :\ (0: Rep[Int])) { (c, agg) => agg + c._1 * c._2 }
      assertC(0 <= idx && idx < nbElem, s"Out of bound: %d not in [0, ${nbElem}]", idx)
      data(idx)
    }

    @virtualize
    def update(idx: Rep[Int], v: Rep[T]) = this.data(idx) = v

    @virtualize
    def +(that: Test[T]) = {
      assert(this.dims == that.dims, s"Dimension mismatch for +: ${this.dims} != ${that.dims}")
      val arr = NewArray[T](this.nbElem)
      for (x <- DataLoop(this.nbElem))
        arr(x) = this(x) + that(x)

      new Test(arr, this.dims)
    }

    @virtualize
    def clipAt(bound: Rep[T]) = {
      for (i <- DataLoop(this.nbElem)) {
        if (this(i) > bound) this(i) = bound
        if (this(i) + bound < zero)
          this(i) = zero - bound
      }
    }


    @virtualize
    def printRaw(row: Int = 10) = {
      for (i <- DataLoop(this.nbElem)) {
        printf(format, this(i))
        val imod = i % row
        if (imod == row - 1)
          printf("\\n")
      }
      printf("\\n")
    }
  }
}
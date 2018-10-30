package lantern

import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

trait Diff {
  type diff = cps[Unit]
  def RST(a: =>Unit @diff) = continuations.reset { a; () }
}

trait DiffApi extends DslOps with Diff {

  type RDouble = Rep[Double]

  class NumF(val x: RDouble, val d: RDouble = 1.0) {
    def +(that: NumF) =
      new NumF(this.x + that.x, this.d + that.d)
    def *(that: NumF) =
      new NumF(this.x * that.x, this.d * that.x + that.d * this.x)
    override def toString = (x,d).toString
  }
  class NumFF(val x: RDouble, val d: NumF) {
    def +(that: NumFF) =
      new NumFF(this.x + that.x, this.d + that.d)
    def *(that: NumFF) =
      new NumFF(this.x * that.x,
        this.d * new NumF(that.x, that.d.x) + that.d * new NumF(this.x, this.d.x))
    override def toString = (x,d).toString
  }

  class NumR(val x: RDouble, val d: Var[Double]) extends Serializable {
    def +(that: NumR): NumR @diff = shift { (k: NumR => Unit) =>
      val y = new NumR(x + that.x, var_new(0.0)); k(y)
      this.d += y.d; that.d += y.d }
    def *(that: NumR): NumR @diff = shift { (k: NumR => Unit) =>
      // is it worth optimizing x*x --> 2*x (this == that)?
      val y = new NumR(x * that.x, var_new(0.0)); k(y)
      this.d += that.x * y.d; that.d += this.x * y.d }
    def -(that: NumR): NumR @diff = shift{ (k: NumR => Unit) =>
      val y = new NumR(x - that.x, var_new(0.0)); k(y)
      this.d += y.d; that.d -= y.d}
    def print() = {
      printf("the value is %f\n", x)
      printf("the gradient is %f\n", d)
    }
  }

  // difference between static var and staged var:
  // static var won't work for nested scopes!!!
  class NumRV(val x: RDouble, var d: RDouble) {
    def +(that: NumRV): NumRV @diff = shift { (k: NumRV => Unit) =>
      val y = new NumRV(x + that.x, 0.0); k(y)
      this.d += y.d; that.d += y.d }
    def *(that: NumRV): NumRV @diff = shift { (k: NumRV => Unit) =>
      val y = new NumRV(x * that.x, 0.0); k(y)
      this.d += that.x * y.d; that.d += this.x * y.d }
  }

  def getNumR(x: RDouble) = new NumR(x, var_new(0.0))

  // Note: we make the generated function return the accumulated deltaVar
  // and add it to the var after calling the continuation. Slightly different
  // than in the upstaged version. The main reason is that we don't (want to)
  // have NumR objects in the generated code and that we can't (easily) pass
  // a mutable var to a function with reference semantics (we could with
  // explicit boxing, and in C/C++ we could just pass the address)
  def FUN(f: NumR => Unit): (NumR => Unit) = { (x: NumR) =>
    val f1 = fun { (x:Rep[Double]) =>
      val deltaVar = var_new(0.0)
      f(new NumR(x, deltaVar))
      readVar(deltaVar)
    }
    x.d += f1(x.x)
  }

  @virtualize
  def IF(c: Rep[Boolean])(a: =>NumR @diff)(b: =>NumR @diff): NumR @diff = shift { k:(NumR => Unit) =>
    val k1 = FUN(k)

    if (c) RST(k1(a)) else RST(k1(b))
  }

  // basic loop constructs
  @virtualize
  def LOOP(init: NumR)(c: NumR => Rep[Boolean])(b: NumR => NumR @diff): NumR @diff = shift { k:(NumR => Unit) =>
    val k1 = FUN(k)

    lazy val loop: NumR => Unit = FUN { (x: NumR) =>
      if (c(x)) RST(loop(b(x))) else RST(k1(x))
    }
    loop(init)
  }

  // loop constructs with a counter (loop iteration number is pre-fixed)
  @virtualize
  def LOOPC(init: NumR)(c: Rep[Int])(b: NumR => NumR @diff): NumR @diff = shift { k:(NumR => Unit) =>

    var gc = 0

    lazy val loop: NumR => Unit = FUN { (x: NumR) =>
      if (gc < c) { gc += 1; RST(loop(b(x))) } else RST(k(x))
    }
    loop(init)

  }

  // loop constructs with a counter and the counter is used by loop body
  @virtualize
  def LOOPCC(init: NumR)(c: Rep[Int])(b: Rep[Int] => NumR => NumR @diff): NumR @diff = shift { k:(NumR => Unit) =>
    var gc = 0
    lazy val loop: NumR => Unit = FUN { (x: NumR) =>
      if (gc < c) { gc += 1; RST(loop(b(gc-1)(x))) } else RST(k(x))
    }
    loop(init)
  }

  // iterate an array (very similar to LOOPCC, but substantiate the situation as array)
  @virtualize
  def LOOPA(init: NumR)(a: Rep[Array[Double]])(b: Rep[Int] => NumR => NumR @diff): NumR @diff = shift { k: (NumR => Unit) =>
    var gc = 0
    val bound = a.length
    lazy val loop: NumR => Unit = FUN { (x: NumR) =>
      if (gc < bound) {gc += 1; RST(loop(b(gc-1)(x)))} else RST(k(x))
    }
    loop(init)
  }

  // the FUN function lift continuations (type NumR => Unit) to Rep type
  // this FUNL function lift continuations composer (type (NumR => Unit) => (NumR => Unit)) to Rep type
  def FUNL(f: ((NumR => Unit) => NumR => Unit)): ((NumR => Unit) => NumR => Unit) = {k1: (NumR => Unit) => (x: NumR) =>
    /* we have to have the continuation to be dynamic here:
       meaning that we almost have to have Rep[NumR => Unit] type as the fun parameter
       but we do extra trick to equivalently transform between Rep[NumR => Unit] and Rep[Double => Double]
    */
    val f1 = fun { (t1: Rep[Double => Double], xx: Rep[Double]) =>
      val t2: (NumR => Unit) = { (x: NumR) => x.d += t1(x.x) }
      val t3: (NumR => Unit) = f(t2)
      val deltaVar = var_new(0.0)
      t3(new NumR(xx, deltaVar))
      readVar(deltaVar)
    };
    val k2: Rep[Double => Double] = fun { (x: Rep[Double]) =>
      val deltaVar = var_new(0.0)
      k1(new NumR(x, deltaVar))
      readVar(deltaVar)
    }
    x.d += f1(k2, x.x)
  }

  // erroneous implementation by using var as indexer, you can ignore this function
  @virtualize
  def LOOPL4(init: NumR)(c: Rep[Int])(b: Rep[Int] => NumR => NumR @diff): NumR @diff = shift { k: (NumR => Unit) =>
    var gc = 0
    lazy val loop: (NumR => Unit) => NumR => Unit = FUNL { (k: NumR => Unit) => (x: NumR) =>
      if (gc < c) { gc += 1; loop((x: NumR) => RST(k(b(gc-1)(x))))(x)  } else { RST(k(x)) }
    }                                               // Problem! gc is a var, so it changes all the time
    loop(k)(init)                                   // so all recursive call will use the same value of gc in the last recursion
  }                                                 // Trying to multiply all elements in a list will endup multiplying the last element many times

  // similar to FUNL, but a Rep[Int] is carried along, so that we don't use a var as indexer
  // avoiding the mistake of the function above.
  def FUNL1(f: (Rep[Int] => (NumR => Unit) => NumR => Unit)): (Rep[Int] => (NumR => Unit) => NumR => Unit) = { i: Rep[Int] => k1: (NumR => Unit) => (x: NumR) =>

    val f1 = fun { (i: Rep[Int], t1: Rep[Double => Double], xx: Rep[Double]) =>
      val t2: (NumR => Unit) = { (x: NumR) => x.d += t1(x.x) }
      val t3: (NumR => Unit) = f(i)(t2)
      val deltaVar = var_new(0.0)
      t3(new NumR(xx, deltaVar))
      readVar(deltaVar)
    }
    val k2: Rep[Double => Double] = fun { (x: Rep[Double]) =>
      val deltaVar = var_new(0.0)
      k1(new NumR(x, deltaVar))
      readVar(deltaVar)
    }
    x.d += f1(i, k2, x.x)
  }

  // fix LOOPL4 by using the FUNL1 function.
  // compared with normal loops, loop of this function stack computations in continuation,
  // until the end of loop.
  // In that sense, an array is traversed in the opposite order.
  // this loop is important for traversing functional list, or any recursive data structures.
  @virtualize
  def LOOPL5(start: Rep[Int])(init: NumR)(c: Rep[Int])(b: Rep[Int] => NumR => NumR @diff): NumR @diff = shift { k: (NumR => Unit) =>
    lazy val loop: Rep[Int] => (NumR => Unit) => NumR => Unit = FUNL1 { (gc: Rep[Int]) => (k: NumR => Unit) => (x: NumR) =>
      def sh_loop: Rep[Int] => NumR => NumR @diff = (i: Rep[Int]) => (x: NumR) => shift{(k: NumR => Unit) => loop(i)(k)(x)}
      RST(k(IF(gc < c) {b(gc)(sh_loop(gc+1)(x))}{x} ))
    }
    loop(start)(k)(init)
  }

  // iterate an array in reverse order by stacking continuations.
  // this is the same as LOOPL5, but using an array as input.
  @virtualize
  def LOOPL6(start: Rep[Int])(init: NumR)(a: Rep[Array[Double]])(b: Rep[Int] => NumR => NumR @diff): NumR @diff = shift { k: (NumR => Unit) =>
    lazy val loop: Rep[Int] => (NumR => Unit) => NumR => Unit = FUNL1 { (gc: Rep[Int]) => (k: NumR => Unit) => (x: NumR) =>
      def sh_loop: Rep[Int] => NumR => NumR @diff = (i: Rep[Int]) => (x: NumR) => shift{(k: NumR => Unit) => loop(i)(k)(x)}
      RST(k(IF(gc < a.length) {b(gc)(sh_loop(gc+1)(x))}{x}))
    }
    loop(start)(k)(init)
  }

  /*
    Note: summarize for list recursive continuation:

    Scenario A: for an array, the desired operation is like:
    def loop(x) = if (done) kf(x) else kf(loop(b(x))) // note "b" happens within loop, "kf" is the final continuation
    cps version:
    def loop(x, k) = if (done) k(x) else loop(b(x), k); loop(x, kf)
    can be simplified as:
    def loop(x) = if (done) x else loop(b(x)); kf(loop(x))

    Scenario B: for a list, the desired operation is like:
    def loop(x) = if (done) kf(x) else kf(b(loop(x))) // note "b" happens outside of loop, because the head of list should be the last element to process
    cps version:
    def loop(x, k) = if (done) k(x) else loop(x, r => k(b(r))); loop(x, kf)
    can be simplified as:
    def loop(k) = if (done) k else loop(r => k(b(r))); loop(kf)(x)

    Now you see the key difference is that in Scenario A, the recursion is changing "x", but in Scenario B, the recursion is changing "k"
    so the type of loop should be different: "Double => Double" in Scenario A, "(Double => Double) => (Double => Double)" in Scenario B.
    In our type setting, "Double => Double" is equivalent with "NumR => Unit", because updating NumR.d with NumR.x is like "Double => Double"
    and correspondingly, "(Double => Double) => (Double => Double)" is like "(NumR => Unit) => (NumR => Unit)".
    In this way, the LOOP* constructs work with NumR type, while the recursive closure "fun" in "FUN" works with simple Double types.

    To further expand this to tree recursion:  we want:
    def T(node, x) = if (isEmpty(node)) kf(x) else kf(b(T(node.left, x), T(node.right, x)))
      here we have node as extra parameter because tree shaped data structure uses node. In contract, list recursion did use node parameter
      because list structure is simply linear, so that keeping a index (int) is enough
    cps version:
    def T(node, k, x) = if (isEmpty(node)) k(x) else T(node.left, l => T(node.right, r => k(b(l, r, node.t)) ,x) ,x); T(root, kf, x)
    where node.t is the encoding of the current node value.
    the operation b takes three parameters, the result of left tree, the result of right tree, and t.
    some of the parameter can be null, and b can be different based on whether the node is leaf or non-leaf

    The cps version is very similar to List recursion, with the key operation to modify continuations.
    In fact, the list recursion can (should) also explicitly add an index, which is like this:
    def loop(i, k, x) = if (i == length) k(x) else loop(i+1, r => k(b(r)), x); loop(0, kf, x)

  */

  @virtualize
  def TREE(init: NumR)(data: Rep[Array[Double]], lch: Rep[Array[Int]], rch: Rep[Array[Int]])(b: (NumR, NumR, NumR) => NumR @diff): NumR @diff = shift {
    k: (NumR => Unit) =>

    // NOTE here the encode function is just wrapping Double as NumR, so it is not explicitly given.
    // However, for general case, encode should be given, as a function (Rep[Int] => NumR)
    // NOTE b can take three NumRs as parameters, all take two NumRs and a Rep[Int],
    // if it take a Rep[Int], then b internally call encode() function to transform Rep[Int] to NumR
    // NOTE data is not really needed in this function, if encode is given and b takes a Rep[Int] parameter
    // I add data here only because encode uses data.
    // If data is not given, a size-of-data needs to be provided, for the bound check

    val bound1 = lch.length  // bound of non-leaf node
    val bound  = data.length // bound of leaf node
    val encode: (Rep[Int] => NumR) = {x: Rep[Int] => new NumR(data(x), var_new(0.0))}

    lazy val tree: Rep[Int] => (NumR => Unit) => NumR => Unit = FUNL1 { (i: Rep[Int]) => (k: NumR => Unit) => (x: NumR) =>
      def sh_tree = (i: Rep[Int]) => shift{k: (NumR => Unit) => tree(i)(k)(x)}
      RST(k( IF(i < bound) {b(sh_tree(lch(i)), sh_tree(rch(i)), encode(i))}{x}))
    }
    tree(0)(k)(init)
  }

  @virtualize // NOTE: this version assume that children array use very large number for leaf nodes
  def TREE1(init: NumR)(bound: Rep[Int], lch: Rep[Array[Int]], rch: Rep[Array[Int]])(b: (NumR, NumR, Rep[Int]) => NumR @diff): NumR @diff = shift {
    k: (NumR => Unit) =>

    lazy val tree: Rep[Int] => (NumR => Unit) => NumR => Unit = FUNL1 { (i: Rep[Int]) => (k: NumR => Unit) => (x: NumR) =>
      def sh_tree = (i: Rep[Int]) => shift{k: (NumR => Unit) => tree(i)(k)(x)}
      RST(k( IF(i < bound) {b(sh_tree(lch(i)), sh_tree(rch(i)), i)}{x} ))
    }
    tree(0)(k)(init)
  }

  @virtualize // NOTE: this version cannot handle empty trees // assume that children array use -1 for leaf nodes
  def TREE2(init: NumR)(lch: Rep[Array[Int]], rch: Rep[Array[Int]])(b: (NumR, NumR, Rep[Int]) => NumR @diff): NumR @diff = shift {
    k: (NumR => Unit) =>

    lazy val tree: Rep[Int] => (NumR => Unit) => NumR => Unit = FUNL1 { (i: Rep[Int]) => (k: NumR => Unit) => (x: NumR) =>
      def sh_tree = (i: Rep[Int]) => shift{ k: (NumR => Unit) => tree(i)(k)(x) }
      RST(k( IF(i >= 0){ b(sh_tree(lch(i)), sh_tree(rch(i)), i) } { x } ))
    }
    tree(0)(k)(init)
  }

  @virtualize
  def Rec(init: NumR)(b: NumR => NumR @diff): NumR @diff = shift { (k: NumR => Unit) =>
    lazy val rec: (NumR => Unit) => NumR => Unit = FUNL { (k: NumR => Unit) => (x: NumR) =>
      RST{k(b(x))}
    }
    rec(k)(init)
  }

  def gradRV(f: NumRV => NumRV @diff)(x: Rep[Double]): Rep[Double] = {
    val x1 = new NumRV(x, 0.0)
    reset { f(x1).d = 1.0 }
    x1.d
  }
  def gradR(f: NumR => NumR @diff)(x: RDouble): Rep[Double] = {
    val x1 = new NumR(x, var_new(0.0))
    reset {
      val r = f(x1)
      var_assign(r.d, 1.0); ()
    }
    x1.d
  }

  def gradF(f: NumF => NumF)(x: RDouble) = {
    val x1 = new NumF(x, 1.0)
    f(x1).d
  }

  def gradFF(f: NumFF => NumFF)(x: RDouble) = {
    val x1 = new NumFF(x, new NumF(1.0, 0.0))
    f(x1).d.d
  }
}
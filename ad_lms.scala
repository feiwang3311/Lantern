import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

object LMS {

  trait DiffApi extends Dsl {

    type diff = cps[Unit]

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


    // Note: we make the generated function return the accumulated deltaVar
    // and add it to the var after calling the continuation. Slightly different
    // than in the unstaged version. The main reason is that we don't (want to)
    // have NumR objects in the generated code and that we can't (easily) pass
    // a mutable var to a function with reference semantics (we could with 
    // explicit boxing, and in C/C++ we could just pass the address)
    def FUN(f: NumR => Unit): (NumR => Unit) = {
      val f1 = fun { (x:Rep[Double]) => 
        val deltaVar = var_new(0.0)
        f(new NumR(x, deltaVar))
        readVar(deltaVar)
      };
      { (x:NumR) => x.d += f1(x.x) }
    }

    def RST(a: =>Unit @diff) = continuations.reset { a; () }

    @virtualize
    def IF(c: Rep[Boolean])(a: =>NumR @diff)(b: =>NumR @diff): NumR @diff = shift { k:(NumR => Unit) =>
      val k1 = FUN(k)

      if (c) RST(k1(a)) else RST(k1(b))
    }

    @virtualize
    def LOOP(init: NumR)(c: NumR => Rep[Boolean])(b: NumR => NumR @diff): NumR @diff = shift { k:(NumR => Unit) =>
      val k1 = FUN(k)

      lazy val loop: NumR => Unit = FUN { (x: NumR) =>
        if (c(x)) RST(loop(b(x))) else RST(k1(x))
      }
      loop(init)
    }

    @virtualize
    def LOOPC(init: NumR)(c: Rep[Int])(b: NumR => NumR @diff): NumR @diff = shift { k:(NumR => Unit) =>

      var gc = 0

      lazy val loop: NumR => Unit = FUN { (x: NumR) =>
        if (gc < c) { gc += 1; RST(loop(b(x))) } else RST(k(x))
      }
      loop(init)

    }

    @virtualize
    def LOOPCC(init: NumR)(c: Rep[Int])(b: Rep[Int] => NumR => NumR @diff): NumR @diff = shift { k:(NumR => Unit) =>
      var gc = 0
      lazy val loop: NumR => Unit = FUN { (x: NumR) =>
        if (gc < c) { gc += 1; RST(loop(b(gc-1)(x))) } else RST(k(x))
      }
      loop(init)
    }

    @virtualize
    def LOOPA(init: NumR)(a: Rep[Array[Double]])(b: Rep[Int] => NumR => NumR @diff): NumR @diff = shift { k: (NumR => Unit) =>
      var gc = 0
      val bound = a.length
      lazy val loop: NumR => Unit = FUN { (x : NumR) =>
        if (gc < bound) {gc += 1; RST(loop(b(gc-1)(x)))} else RST(k(x))
      }
      loop(init)
    }
/*
    @virtualize
    def LOOPL(init: NumR)(c: Rep[Int])(b: Rep[Int] => NumR => NumR @diff): NumR @diff = shift { k:(NumR => Unit) =>
      var gc = 0
      lazy val loop: NumR => Unit = FUN { (x: NumR) =>
        if (gc < c) { gc += 1; RST(b(gc-1)(loop(x))) } else RST(k(x))
      }                                   // type error, loop(x) is unit type
      loop(init)
    }

    @virtualize
    def LOOPL1(init: NumR)(c: Rep[Int])(b: Rep[Int] => NumR => NumR @diff): NumR @diff = shift { k: (NumR => Unit) =>
      var gc = 0
      lazy val loop: NumR => NumR @diff = FUNL { (x: NumR) => // How to design the FUNL, when the type is NumR => NumR @diff?
        if (gc < c) { gc += 1; b(gc-1)(loop(x)) } else x
      }
      RST(k(loop(init)))
    }
*/
/*
    @virtualize
    def LOOPL2(init: NumR)(c: Rep[Int])(b: Rep[Int] => NumR => NumR @diff): NumR @diff = shift { k: (NumR => Unit) =>
      var gc = 0
      var k_g = k
      lazy val loop: NumR => Unit = FUN { (x: NumR) =>
        if (gc < c) { gc += 1; k_g = { (x: NumR) => k_g(b(gc-1)(x)) }; loop(x) } else RST(k_g(x))
      }
      loop(init)
    }
*/
    def FUNL(f: ((NumR => Unit) => (NumR => Unit))): ((NumR => Unit) => (NumR => Unit)) = {
      /* we have to have the continuation to be dynamic here:
         meaning that we almost have to have Rep[NumR => Unit] type as the fun parameter
         but we do extra trick to equvilently transfrom between Rep[NumR => Unit] and Rep[Double => Double]
      */
      val f1 = fun { (t1: Rep[Double => Double]) =>
        val t2: (NumR => Unit) = { (x: NumR) => x.d += t1(x.x) }
        val t3: (NumR => Unit) = f(t2)
        fun {(x: Rep[Double]) => 
          val deltaVar = var_new(0.0)
          t3(new NumR(x, deltaVar))
          readVar(deltaVar)
        }
      };

      {k1: (NumR => Unit) => 
        {
          val k2: Rep[Double => Double] = fun { (x: Rep[Double]) =>
            val deltaVar = var_new(0.0)
            k1(new NumR(x, deltaVar))
            readVar(deltaVar)
          }
          val k3: Rep[Double => Double] = f1(k2)
          val k4: (NumR => Unit) = {(x: NumR) => x.d += k3(x.x)}
          k4
        } 
      }
    }

    @virtualize
    def LOOPL4(init: NumR)(c: Rep[Int])(b: Rep[Int] => NumR => NumR @diff): NumR @diff = shift { k: (NumR => Unit) =>
      var gc = 0
      lazy val loop: (NumR => Unit) => NumR => Unit = FUNL { (k: NumR => Unit) => (x: NumR) =>
        if (gc < c) { gc += 1; loop((x: NumR) => RST(k(b(gc-1)(x))))(x)  } else { RST(k(x)) } 
      }                                               // Problem! gc is a var, so it changes all the time
      loop(k)(init)                                   // so all recursive call will use the same value of gc in the last recursion
    }                                                 // Trying to multiply all elements in a list will endup multiplying the last element many times

    def FUNL1(f: (Rep[Int] => (NumR => Unit) => (NumR => Unit))): (Rep[Int] => (NumR => Unit) => (NumR => Unit)) = {      

      val f1 = fun { (yy: Rep[(Int, (Double => Double), Double)]) => 
        val i: Rep[Int] = tuple3_get1(yy)
        val t1: Rep[Double => Double] = tuple3_get2(yy)
        val xx: Rep[Double] = tuple3_get3(yy)
        val t2: (NumR => Unit) = { (x: NumR) => x.d += t1(x.x) }
        val t3: (NumR => Unit) = f(i)(t2)

        val deltaVar = var_new(0.0)
        t3(new NumR(xx, deltaVar))
        readVar(deltaVar)
      };

      {i: Rep[Int] => k1: (NumR => Unit) => 
        {
          val k2: Rep[Double => Double] = fun { (x: Rep[Double]) =>
            val deltaVar = var_new(0.0)
            k1(new NumR(x, deltaVar))
            readVar(deltaVar)
          }
          val k4: (NumR => Unit) = {(x: NumR) => 
            x.d += f1((i, k2, x.x))
          }
          k4
        } 
      }
    }

    @virtualize
    def LOOPL5(init: NumR)(c: Rep[Int])(b: Rep[Int] => NumR => NumR @diff): NumR @diff = shift { k: (NumR => Unit) =>
      lazy val loop: Rep[Int] => (NumR => Unit) => NumR => Unit = FUNL1 { (gc: Rep[Int]) => (k: NumR => Unit) => (x: NumR) =>
        if (gc < c) { loop(gc+1)((x: NumR) => RST(k(b(gc)(x))))(x)  } else { RST(k(x)) }
      }
      loop(0)(k)(init)
    }    

    /*
      Note: summerize for list recursive continuation:

      Senario A: for an array, the desired operation is like:
      def loop(x) = if (done) kf(x) else kf(loop(b(x))) // note "b" happens within loop, "kf" is the final continuation
      cps version:
      def loop(x, k) = if (done) k(x) else loop(b(x), k); loop(x, kf)
      can be simplifed as:
      def loop(x) = if (done) x else loop(b(x)); kf(loop(x))

      Senario B: for a list, the desired operation is like:
      def loop(x) = if (done) kf(x) else kf(b(loop(x))) // note "b" happens outside of loop, because the head of list should be the last element to process
      cps version:
      def loop(x, k) = if (done) k(x) else loop(x, r => k(b(r))); loop(x, kf)
      can be simplified as:
      def loop(k) = if (done) k else loop(r => k(b(r))); loop(kf)(x)

      Now you see the key difference is that in Senario A, the recursion is changing "x", but in Senario B, the recursion is changing "k"
      so the type of loop should be different: "Double => Double" in Senario A, "(Double => Double) => (Double => Double)" in Senario B.
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
        if (i < bound) { tree(lch(i))((l: NumR) => tree(rch(i))((r: NumR) => RST(k(b(l, r, encode(i)))))(x))(x) } else { RST(k(x)) }
      }                                            
      tree(0)(k)(init)
    }

    @virtualize // NOTE: this version assume that children array use very large number for leaf nodes
    def TREE1(init: NumR)(bound: Rep[Int], lch: Rep[Array[Int]], rch: Rep[Array[Int]])(b: (NumR, NumR, Rep[Int]) => NumR @diff): NumR @diff = shift {
      k: (NumR => Unit) =>

      lazy val tree: Rep[Int] => (NumR => Unit) => NumR => Unit = FUNL1 { (i: Rep[Int]) => (k: NumR => Unit) => (x: NumR) =>
        if (i < bound) { tree(lch(i))((l: NumR) => tree(rch(i))((r: NumR) => RST(k(b(l, r, i))))(x))(x) } else { RST(k(x)) }
      }
      tree(0)(k)(init)
    }

    @virtualize // NOTE: this version cannot handle empty trees // assume that children array use -1 for leaf nodes
    def TREE2(init: NumR)(lch: Rep[Array[Int]], rch: Rep[Array[Int]])(b: (NumR, NumR, Rep[Int]) => NumR @diff): NumR @diff = shift {
      k: (NumR => Unit) =>

      lazy val tree: Rep[Int] => (NumR => Unit) => NumR => Unit = FUNL1 { (i: Rep[Int]) => (k: NumR => Unit) => (x: NumR) =>
        if (i >= 0) { tree(lch(i))((l: NumR) => tree(rch(i))((r: NumR) => RST(k(b(l, r, i))))(x))(x) } else { RST(k(x)) }
      }
      tree(0)(k)(init)
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
        printf("result of model is %f\n", r.x)
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

  def main(args: Array[String]): Unit = {

    import java.io.PrintWriter;
    import java.io.File;    
    
    val gr1 = new DslDriver[Double,Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        gradR(x => x + x*x*x)(x)
      }
    }

    val grv1 = new DslDriver[Double,Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        gradRV(x => x + x*x*x)(x)
      }
    }

    val gf1 = new DslDriver[Double,Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        gradF(x => x + x*x*x)(x)
      }
    }

    val gff1 = new DslDriver[Double,Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        gradFF(x => x + x*x*x)(x)
      }
    }
/*
    println("---- reverse mode with dynamic vars ---- \n")
    println(gr1.code)
    println("---- reverse mode with static vars ---- \n")
    println(grv1.code)
    println("---- forward mode ---- \n")
    println(gf1.code)
    println("---- forward mode second order ---- \n")
    println(gff1.code)

    // reverse mode
    for (x <- 0 until 10) {
      assert(gr1.eval(x) == 1 + 3*x*x)
    }

    // reverse mode (with static accumulation of reverse 
    // computation -- only for straightline code)
    for (x <- 0 until 10) {
      assert(grv1.eval(x) == 1 + 3*x*x)
    }

    // forward mode
    for (x <- 0 until 10) {
      assert(gf1.eval(x) == 1 + 3*x*x)
    }

    // forward of forward mode
    for (x <- 0 until 10) {
      assert(gff1.eval(x) == 6*x)
    }
*/
    // test 2 -- conditional
    val gr2 = new DslDriver[Double,Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        val minus_1 = (new NumR(-1.0,var_new(0.0)))
        gradR(x => IF (x.x > 0.0) { minus_1*x*x } { x*x })(x)
      }
    }
/*
    println("---- reverse mode with dynamic vars ---- \n")
    println(gr2.code)
    // NOTE: in the generated code, code motion has pushed
    // x3 = {x4: (Double) => ... } into both if branches.
    // (suboptimal in terms of code size)

    // reverse mode
    for (x <- -10 until 10) {
      assert(gr2.eval(x) == (if (x > 0.0) -2*x else 2*x))
    }
*/
    // test 3 -- loop using fold
    def fr(x: Double): Double = {
      // Divide by 2.0 until less than 1.0
      if (x > 1.0) fr(0.5 * x) else x 
    }
    // Hand-coded correct derivative
    def gfr(x: Double): Double = {
      if (x > 1.0) 0.5 * gfr(0.5 * x) else 1.0
    }


    val gr3 = new DslDriver[Double,Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        val half = (new NumR(0.5,var_new(0.0)))
        val res = gradR(x => LOOP(x)(x1 => x1.x > 1.0)(x1 => half * x1))(x)
        println(readVar(half.d))
        res
      }
    }
/*
    println("---- reverse mode with dynamic vars ---- \n")
    println(gr3.code)
    println(gr3.eval(10))


    // NOTE: in the generated code, code motion has pushed
    // x3 = {x4: (Double) => ... } into both if branches.
    // (suboptimal in terms of code size)

    // reverse mode
    for (x <- 0 until 10) {
      assert(gr3.eval(x) == gfr(x))
    }

    println("done")
*/    
    val gr4 = new DslDriver[Double, Double] with DiffApi {
      def snippet(x: Rep[Double]): Rep[Double] = {
        val half = new NumR(0.5, var_new(0.0))
        val res = gradR(x => LOOPC(x)(3)(x1 =>{half * x1}))(x)
        println(readVar(half.d))
        res
      }
    }

    //println(gr4.code)
    //println(gr4.eval(10))

    val gr7 = new DslDriver[Double, Double] with DiffApi {

      def snippet(x: Rep[Double]): Rep[Double] = {
        val half = new NumR(0.5, var_new(0.0))
        val res = gradR(x => LOOPCC(x)(3)(i => x1 => {
          println(i)
          half * x1 }))(x)
        println(readVar(half.d))
        res
      }
    }

    //println(gr7.code)
    //println(gr7.eval(10))

    val gr8 = new DslDriver[Double, Double] with DiffApi {

      def snippet(x: Rep[Double]): Rep[Double] = {
        val array = NewArray[Double](3)
        for (i <- (0 until 3): Rep[Range]) array(i) = i + 2
        val model: NumR => NumR @diff = { (x: NumR) =>
          LOOPA(x)(array)(i => x1 => {
            val t = new NumR(array(i), var_new(0.0))
            t * x1
          })
        }
        val res = gradR(model)(x)
        res
      }
    }

    //println(gr8.code)
    //println(gr8.eval(10))

    val gr9 = new DslDriver[Double, Double] with DiffApi {

      def snippet(x: Rep[Double]): Rep[Double] = {
        // preprocess data (wrap Array as RepArray)
        val arr = scala.Array(1.5,3.0,4.0)
        val arra = staticData(arr)
        
        val model: NumR => NumR @diff = { (x: NumR) =>
          LOOPA(x)(arra)(i => x1 => {
            val t = new NumR(arra(i), var_new(0.0))
            t * x1
            })
        }
        val res = gradR(model)(x)
        res
      }
    }

    //println(gr9.code)
    //println(gr9.eval(10))

    val gr10 = new DslDriver[Double, Double] with DiffApi {

      def snippet(x: Rep[Double]): Rep[Double] = {
        // preprocess data
        val arr = scala.Array(1.5, 2.0, 3.0)
        val arra = staticData(arr)
        val length = arr.length

        // maybe we can use loopcc, just use arra by closure
        val model: NumR => NumR @diff = { (x: NumR) =>
          LOOPCC(x)(length)(i => x1 => {
            val t = new NumR(arra(i), var_new(0.0))
            t * x1
            })
        }
        val res = gradR(model)(x)
        res
        /*
          Note: It is interesting to note that the recursive function body can make use of closured array data
                but recursive guard (the if condition of LOOP) cannot, because the def of LOOP is before the presence of data
                So the length has to be passed in as a parameter explicitly
          Is there a better way to do it??
        */
      }
    }

    //println(gr10.code)
    //println(gr10.eval(10))

    val gr11 = new DslDriver[Double, Double] with DiffApi {

      def snippet(x: Rep[Double]): Rep[Double] = {
        // represent list as array
        val arr = scala.Array(4.0, 3.0, 1.5, 2.0)
        //val arr = scala.Array[Double]()
        val arra = mutableStaticData(arr)

        // create a model that recursively use the data in arr (originated from list)
        def model: NumR => NumR @diff = { (x: NumR) =>
          LOOPL5(x)(arra.length)(i => x1 => new NumR(arra(i), var_new(0.0)) * x1)
        }
        val res = gradR(model)(x)
        res
      }
    }
    
    //println(gr11.code)
    /*val p = new PrintWriter(new File("gr11.scala"))
    p.println(gr11.code)
    p.flush()*/
    //println(gr11.eval(2))

    val gr12 = new DslDriver[Double, Double] with DiffApi {

      def snippet(x: Rep[Double]): Rep[Double] = {
        // represent tree as arrays
        /*    5
             / \
            3   4
        */
        val data = scala.Array[Double](5.0, 3.0, 4.0)
        val lch  = scala.Array[Int](1, 100, 100) // use very large number to mean non nodes
        val rch  = scala.Array[Int](2, 100, 100)
        val data1 = mutableStaticData(data)
        val lch1  = mutableStaticData(lch)
        val rch1  = mutableStaticData(rch)

        // create a model that recursively use the data (originated from tree)
        def model: NumR => NumR @diff = { (x: NumR) =>
          TREE1(x)(data1.length, lch1, rch1){ (l: NumR, r: NumR, i: Rep[Int]) =>
            l * r * new NumR(data1(i), var_new(0.0))
          }
        }

        val res = gradR(model)(x)
        res
      }
    }

    //println(gr12.code)
    //println(gr12.eval(2))

    val gr12_2 = new DslDriver[Double, Double] with DiffApi {

      def snippet(x: Rep[Double]): Rep[Double] = {
        val A = scala.Array
        val data = A[Double](5.0, 3.0, 4.0)
        val lch  = A[Int](1, 100, 100)
        val rch  = A[Int](2, 100, 100)
        val data1 = mutableStaticData(data)
        val lch1  = mutableStaticData(lch)
        val rch1  = mutableStaticData(rch)

        val para = new NumR(2.0, var_new(0.0))
        def model: NumR => NumR @diff = { (x: NumR) =>
          TREE1(x)(data1.length, lch1, rch1){ (l: NumR, r: NumR, i: Rep[Int]) =>
            l * r * new NumR(data1(i), var_new(0.0)) * para
          }
        }

        val res = gradR(model)(x)
        printf("the grad of para is %f\n", readVar(para.d))
        res
      }
    }

    //println(gr12_2.code)
    //printf("grad of x is %f\n", gr12_2.eval(1))

    val gr12_3 = new DslDriver[Double, Double] with DiffApi {

      @virtualize
      def snippet(x: Rep[Double]): Rep[Double] = {
        val A = scala.Array
        val data = A[Double](0.0, 3.0, 4.0)
        val lch  = A[Int](1, 100, 100)
        val rch  = A[Int](2, 100, 100)
        val data1 = mutableStaticData(data)
        val lch1  = mutableStaticData(lch)
        val rch1  = mutableStaticData(rch)

        val para  = new NumR(2.0, var_new(0.0))
        val paral = new NumR(1.5, var_new(0.0))
        val parar = new NumR(1.6, var_new(0.0))
        val bias  = new NumR(2.5, var_new(0.0))
        def model: NumR => NumR @diff = { (dummy: NumR) =>
          TREE1(dummy)(data1.length, lch1, rch1){ (l: NumR, r: NumR, i: Rep[Int]) =>
            IF (lch1(i) > data1.length) {para * new NumR(data1(i), var_new(0.0)) + bias}
            {paral * l + parar * r + bias}
          }
        }

        val res = gradR(model)(x)
        printf("the grad of para is %f\n",  readVar(para.d))
        printf("the grad of paral is %f\n", readVar(paral.d))
        printf("the grad of parar is %f\n", readVar(parar.d))
        printf("the grad of bias is %f\n",  readVar(bias.d))
        0.0
      }
    }
/*
    val p1 = new PrintWriter(new File("temp/gr12_3.scala"))
    p1.println(gr12_3.code)
    p1.flush()*/
    gr12_3.eval(0.0)
    //printf("grad of x is %f\n", gr12_3.eval(1))

    val gr12_4 = new DslDriver[Double, Double] with DiffApi {

      @virtualize
      def snippet(x: Rep[Double]): Rep[Double] = {
        val A = scala.Array
        val data = A[Double](0.0, 3.0, 4.0)
        val lch  = A[Int](1, -1, -1)
        val rch  = A[Int](2, -1, -1)
        val data1 = mutableStaticData(data)
        val lch1  = mutableStaticData(lch)
        val rch1  = mutableStaticData(rch)

        val para  = new NumR(2.0, var_new(0.0))
        val paral = new NumR(1.5, var_new(0.0))
        val parar = new NumR(1.6, var_new(0.0))
        val bias  = new NumR(2.5, var_new(0.0))
        def model: NumR => NumR @diff = { (dummy: NumR) =>
          TREE2(dummy)(lch1, rch1){ (l: NumR, r: NumR, i: Rep[Int]) =>
            IF (lch1(i) < 0) {para * new NumR(data1(i), var_new(0.0)) + bias}
            {paral * l + parar * r + bias}
          }
        }

        val res = gradR(model)(x)
        printf("the grad of para is %f\n",  readVar(para.d))
        printf("the grad of paral is %f\n", readVar(paral.d))
        printf("the grad of parar is %f\n", readVar(parar.d))
        printf("the grad of bias is %f\n",  readVar(bias.d))
        0.0
      }
    }

/*
    val p2 = new PrintWriter(new File("temp/gr12_4.scala"))
    p2.println(gr12_4.code)
    p2.flush()
    gr12_4.eval(0.0)()
    //printf("grad of x is %f\n", gr12_4.eval(1))
*/

  }
}
package lantern

import scala.util.continuations._

import lms.core.stub._
import lms.macros.SourceContext
import lms.core.virtualize
import lms.core.Backend._
import lms.core.Graph

import java.io.File
import java.io.PrintWriter

trait LanternGenC extends DslGenC {
  val IR: DslExp
  import IR._

  override def remap(m: Manifest[_]): String = m.toString match {
    case f if f.startsWith("scala.Function") =>
      val targs = m.typeArguments.dropRight(1)
      val res = remap(m.typeArguments.last)
      def remapInFunction[A](m: Manifest[A]): Array[String] = {
        val s = m.toString
        if (s.startsWith("scala.Tuple")) m.typeArguments.map(t => remap(t)).toArray
        else scala.Array(remap(m))
      }
      val targsUnboxed = targs.flatMap(t => remapInFunction(t))
      "function<" + res + "(" + targsUnboxed.mkString(",") + ")>"
    case _ => super.remap(m)
  }

  override def quoteBlock1(y: lms.core.Backend.Block, argType: Boolean = false) = {
    def eff = quoteEff(y.ein)
    def typed(s:lms.core.Backend.Sym) = if (argType) s"${remap(typeMap(s))} ${quote(s)}" else quote(s)
    def ltyped(xs:List[lms.core.Backend.Sym]) = xs.map(typed(_)).mkString(", ")
    def paren(s:String) = if (argType) "("+s+")" else s
    if (y.in.length == 0) {
      quoteBlock(traverse(y))
    } else {
      val xs = y.in
      val l = captureLines(traverse(y))
      val b = l.mkString("\n")
      s"[&]${paren(ltyped(xs))} {$b}"
    }
  }

  override def shallow(n: Node): String = n match {
    case n @ Node(s, op, List(x), _) if op.startsWith("new Array[") =>
      def parse(op: String): String = {
        if (op.startsWith("Array[")) {
          val inner = op.drop(6).dropRight(1)
          parse(inner) + "*"
        } else op.toLowerCase() // NOTE: only applies to simple numeric types
      }
      val ctype = parse(op.drop(4))
      s"(${ctype})myMalloc(${shallow1(x)} * sizeof(${ctype.dropRight(1)}))"
    case _ => super.shallow(n)
  }

  override def shallow(n: lms.core.Backend.Def): String = n match {
    case InlineSym(t: Node) => shallow(t)
    case b: lms.core.Backend.Block => quoteBlock1(b)
    case _ => quote(n)
  }

  override def emitAll(g: Graph, name: String)(m1:Manifest[_],m2:Manifest[_]): Unit = {
    init(g)
    val arg = quote(g.block.in.head)
    val efs = "" //quoteEff(g.block.ein)
    val stt = dce.statics.toList.map(quoteStatic).mkString(", ")
    val (ms1, ms2) = (remap(m1), remap(m2))
    val functionName = name
    stream.println("""
    #include <fcntl.h>
    #include <errno.h>
    #include <err.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <sys/time.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <stdbool.h>
    #include <stdint.h>
    #include <unistd.h>
    #include <assert.h>
    #include <functional>
    #include <math.h>
    #include <memory>
    #include <random>
    #include <cblas.h>
    #include <algorithm>
    #include <numeric>
    using namespace std;

    #ifndef MAP_FILE
    #define MAP_FILE MAP_SHARED
    #endif

    long fsize(int fd) {
      struct stat stat;
      int res = fstat(fd,&stat);
      return stat.st_size;
    }
    int printll(char* s) {
      while (*s != '\n' && *s != ',' && *s != '\t') {
        putchar(*s++);
      }
      return 0;
    }
    long hash(char *str0, int len)
    {
      unsigned char* str = (unsigned char*)str0;
      unsigned long hash = 5381;
      int c;

      while ((c = *str++) && len--)
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

      return hash;
    }

    long HEAP_SIZE_CPU = 1073741826;
    void *mallocBase = calloc(HEAP_SIZE_CPU, 1);
    void *mallocAddr = mallocBase;
    void *waterMark = mallocBase;
    void *myMalloc(size_t bytes) {
      void *res = mallocAddr;
      mallocAddr = (void *)((char *)mallocAddr + bytes);
      if ((long)mallocAddr >= (long)mallocBase + HEAP_SIZE_CPU) {
        fprintf(stderr, "CPU memory breached limit of HEAP_SIZE_CPU\n"); abort();
      }
      return res;
    }
    int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1) {
      long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
      result->tv_sec = diff / 1000000;
      result->tv_usec = diff % 1000000;
      return (diff < 0);
    }

    void Snippet(char*);
    //std::random_device rd{};
    //std::mt19937 gen{rd()};
    //std::normal_distribution<> d{0, 0.01};

    int main(int argc, char *argv[])
    {
      if (argc != 2) {
        printf("usage: query <filename>\n");
        return 0;
      }
      Snippet(argv[1]);
      return 0;
    }
    /*****************************************
    Emitting C Generated Code
    *******************************************/
    """)
    stream.println(s"$ms2 $functionName($ms1 $arg) {")
    apply(g)
    stream.println("}")
    stream.println("""
    /*****************************************
    End of C Generated Code
    *******************************************/
    """)
  }
}

// abstract class DslDriverBase[A: Manifest, B: Manifest] extends DslExp { self =>
//   // The C-like code generator.
//   // val codegen: DslGenBase {
//   //   val IR: self.type
//   // }

//   val dir = "/tmp"
//   val fileName = s"lantern-snippet-${scala.util.Random.alphanumeric.take(4).mkString}"

//   // The code snippet to compile.
//   def snippet(x: Rep[A]): Rep[B]

//   def eval(a: A)
//   // Note: is it possible to implement `eval` returning `B`?
//   // def eval(a: A): B

//   // lazy val code: String = {
//   //   val source = new java.io.StringWriter()
//   //   codegen.emitSource[A,B](snippet, "Snippet", new java.io.PrintWriter(source))
//   //   source.toString
//   // }
// }

// // /**
//   * A wrapper around `DslDriverBase` that provides a `wrapper` function that performs backend setup/cleanup.
//   * Extend this instead of `DslDriverBase` for correct backend management.
//   */
// trait LanternDriver[A, B] extends DslDriverBase[A, B] with TensorDsl { self =>
//   // Hacky workaround to support trait type parameters with context bounds.
//   // `trait LanternDriver[A: Manifest, B: Manifest]` doesn't work.
//   // These must be overridden in subclasses.
//   implicit def manifestA: Manifest[A]
//   implicit def manifestB: Manifest[B]

//   def wrapper(x: Rep[A]): Rep[B] = {
//     generate_comment("Backend setup.")
//     backend.setup()
//     val result = snippet(x)

//     generate_comment("Backend cleanup.")
//     backend.cleanup()
//     result
//   }

//   // override lazy val code: String = {
//   //   val source = new java.io.StringWriter()
//   //   codegen.emitSource(wrapper, "Snippet", new java.io.PrintWriter(source))
//   //   source.toString
//   // }
// }

// abstract class LanternDriverC[A: Manifest, B: Manifest] extends DslDriverC[A, B] with LanternDriver[A, B] with TensorDslCPU /*with NNModule with Dataset with ONNXLib*/ { self =>
//   override def manifestA: Manifest[A] = manifest[A]
//   override def manifestB: Manifest[B] = manifest[B]
// }


// TODO: bad design!! NNModule should not depend on backend!
abstract class LanternDriverBase[A: Manifest, B: Manifest] extends DslDriverC[A, B]
with TensorDsl with NNModule with NNModuleCudnn with Dataset with ONNXLib with ScannerOpsExp with TimerOpsExp { q =>
  override val codegen = new LanternGenC {
    val IR: q.type = q
  }

  val dir = "/tmp/"
  val fileName = s"lantern-snippet-${scala.util.Random.alphanumeric.take(4).mkString}"

  def codeToFile(name: Option[String] = None) = {
    val outFileName = name match {
      case Some(s) => s
      case None => dir + fileName + ".cpp"
    }
    System.out.println(s"code => $outFileName")
    val outFile = new PrintWriter(new File(outFileName))
    outFile.println(this.code)
    outFile.flush()
  }

  def wrapper(x: Rep[A]): Unit = {
    generate_comment("Backend setup.")
    backend.setup()
    val result = snippet(x)

    generate_comment("Backend cleanup.")
    backend.cleanup()
  }

  override lazy val code: String = {
    val source = new java.io.StringWriter()
    codegen.emitSource(wrapper, "Snippet", new java.io.PrintWriter(source))
    source.toString
  }

}

abstract class LanternDriverC[A: Manifest, B: Manifest] extends LanternDriverBase[A, B] with TensorDslCPU { q =>

  backend = BackendCPU()

  override lazy val f: A => Unit = {
    // TBD: should read result of type B?
    val out = new java.io.PrintWriter("/tmp/snippet.cpp")
    out.println(code)
    out.close
    (new java.io.File("/tmp/snippet")).delete
    import scala.sys.process._
    // TODO: would like to use time("cc") { .. }, but messes with captureOut
    (s"g++ -std=c++11 -O3 /tmp/snippet.cpp -o /tmp/snippet -I /opt/OpenBLAS/include -L /opt/OpenBLAS/lib -lopenblas -lpthread": ProcessBuilder).lines.foreach(Console.println _)
    (a: A) => (s"/tmp/snippet $a": ProcessBuilder).lines.foreach(Console.println _)
  }

}

abstract class LanternDriverCublas[A: Manifest, B: Manifest] extends LanternDriverBase[A, B] with TensorDslCublas { q =>

  override lazy val f: A => Unit = {
    // TBD: should read result of type B?
    val out = new java.io.PrintWriter("/tmp/snippet.cpp")
    out.println(code)
    out.close
    (new java.io.File("/tmp/snippet")).delete
    import scala.sys.process._
    // TODO: would like to use time("cc") { .. }, but messes with captureOut
    (s"nvcc -std=c++11 -O3 /tmp/snippet.cpp -o /tmp/snippet -I /opt/OpenBLAS/include -L /opt/OpenBLAS/lib -lopenblas -lpthread": ProcessBuilder).lines.foreach(Console.println _)
    (a: A) => (s"/tmp/snippet $a": ProcessBuilder).lines.foreach(Console.println _)
  }

}

abstract class LanternDriverCudnn[A: Manifest, B: Manifest] extends LanternDriverBase[A, B] with TensorDslCublas with TensorDslCudnn { q =>
}

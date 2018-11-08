package lantern

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms.common._

trait DslOps extends PrimitiveOps with NumericOpsExtra with BooleanOps
    with LiftString with LiftPrimitives with LiftNumeric with LiftBoolean
    with IfThenElse with Equal with RangeOps with OrderingOps with MiscOps with ArrayOps with StringOps
    with Functions with While with StaticData
    with Variables with LiftVariables with UtilOps with UncheckedOps
    with MathOps with TupleOps with TupledFunctions with CastingOps with ScannerOps {

  // implicit def repStrToSeqOps(a: Rep[String]) = new SeqOpsCls(a.asInstanceOf[Rep[Seq[Char]]])
  implicit class BooleanOps2(lhs: Rep[Boolean]) {
    def &&(rhs: =>Rep[Boolean])(implicit pos: SourceContext) = __ifThenElse(lhs, rhs, unit(false))
  }
  // override def boolean_and(lhs: Rep[Boolean], rhs: Rep[Boolean])(implicit pos: SourceContext): Rep[Boolean] = __ifThenElse(lhs, rhs, unit(false))

  // Raw code/comment operations.
  def generateRawCode(s: String): Rep[Unit]
  def generateRawComment(s: String): Rep[Unit]
  def comment[A:Manifest](s: String, verbose: Boolean = true)(b: => Rep[A]): Rep[A]

  // added by Fei
  def mutableStaticData[T:Manifest](x: T): Rep[T]
  // add for fun with 6 or more parameters
  // def fun[A1:Manifest,A2:Manifest,A3:Manifest,A4:Manifest,A5:Manifest,A6:Manifest,B:Manifest](f: (Rep[A1], Rep[A2], Rep[A3], Rep[A4], Rep[A5], Rep[A6]) => Rep[B]): Rep[((A1,A2,A3,A4,A5,A6))=>B] =
  //   fun((t: Rep[(A1,A2,A3,A4,A5,A6)]) => f(tuple6_get1(t), tuple6_get2(t), tuple6_get3(t), tuple6_get4(t), tuple6_get5(t), tuple6_get6(t)))
  // // def fun[A1:Manifest,A2:Manifest,A3:Manifest,A4:Manifest,A5:Manifest,A6:Manifest,A7:Manifest,B:Manifest](f: (Rep[A1], Rep[A2], Rep[A3], Rep[A4], Rep[A5], Rep[A6], Rep[A7]) => Rep[B]): Rep[((A1,A2,A3,A4,A5,A6,A7))=>B] =
  // //   fun((t: Rep[(A1,A2,A3,A4,A5,A6,A7)]) => f(tuple7_get1(t), tuple7_get2(t), tuple7_get3(t), tuple7_get4(t), tuple7_get5(t), tuple7_get6(t), tuple7_get7(t)))
  // class LambdaOps6[A1:Manifest,A2:Manifest,A3:Manifest,A4:Manifest,A5:Manifest,A6:Manifest,B:Manifest](f: Rep[((A1,A2,A3,A4,A5,A6)) => B]) {
  //   def apply(x1: Rep[A1], x2: Rep[A2], x3: Rep[A3], x4: Rep[A4], x5: Rep[A5], x6: Rep[A6]) = doApply(f,(x1, x2, x3, x4, x5, x6))
  //   def apply(x: Rep[(A1,A2,A3,A4,A5,A6)]): Rep[B] = doApply(f,x)
  // }
  // implicit def toLambdaOps6[A1:Manifest,A2:Manifest,A3:Manifest,A4:Manifest,A5:Manifest,A6:Manifest,B:Manifest](fun: Rep[((A1,A2,A3,A4,A5,A6)) => B]) =
  //   new LambdaOps6(fun)
}

trait DslExp extends DslOps
    with PrimitiveOpsExpOpt with NumericOpsExpOpt with NumericOpsExtraExp with BooleanOpsExp
    with IfThenElseExpOpt with EqualExpBridgeOpt with RangeOpsExp with OrderingOpsExp
    with MiscOpsExp with EffectExp with ArrayOpsExpOpt with StringOpsExp
    with FunctionsRecursiveExp with WhileExp with StaticDataExp
    with UtilOpsExp with UncheckedOpsExp with MathOpsExp
    with TupleOps with TupledFunctionsExp with CastingOpsExp
    with ScannerOpsExp {

  override def boolean_or(lhs: Exp[Boolean], rhs: Exp[Boolean])(implicit pos: SourceContext) : Exp[Boolean] = lhs match {
    case Const(false) => rhs
    case _ => super.boolean_or(lhs, rhs)
  }
  override def boolean_and(lhs: Exp[Boolean], rhs: Exp[Boolean])(implicit pos: SourceContext) : Exp[Boolean] = lhs match {
    case Const(true) => rhs
    case _ => super.boolean_and(lhs, rhs)
  }

  // A raw snippet of code, to be code generated literally.
  case class RawCode(s: String) extends Def[Unit]
  def generateRawCode(s: String) = reflectEffect(RawCode(s))

  case class RawComment(s: String) extends Def[Unit]
  def generateRawComment(s: String) = reflectEffect(RawComment(s))

  case class Comment[A:Manifest](s: String, verbose: Boolean, b: Block[A]) extends Def[A]
  def comment[A:Manifest](s: String, verbose: Boolean)(b: => Rep[A]): Rep[A] = {
    val br = reifyEffects(b)
    val be = summarizeEffects(br)
    super.reflectEffect[A](Comment(s, verbose, br), be)
  }

  override def boundSyms(e: Any): List[Sym[Any]] = e match {
    case Comment(_, _, b) => effectSyms(b)
    case _ => super.boundSyms(e)
  }

  override def array_apply[T:Manifest](x: Exp[Array[T]], n: Exp[Int])(implicit pos: SourceContext): Exp[T] = (x,n) match {
    case (Def(StaticData(x:Array[T])), Const(n)) =>
      val y = x(n)
      if (y.isInstanceOf[Int]) unit(y) else staticData(y)
    // case _ => super.array_apply(x,n)
    // FIXME!!!
    case _ => reflectEffect(ArrayApply(x, n))
  }
  // override def array_apply[T:Manifest](x: Exp[Array[T]], n: Exp[Int])(implicit pos: SourceContext): Exp[T] = reflectEffect(ArrayApply(x, n))

  override def array_update[T:Manifest](x: Exp[Array[T]], n: Exp[Int], y: Exp[T])(implicit pos: SourceContext) = reflectEffect(ArrayUpdate(x,n,y))
  /*
  override def array_update[T:Manifest](x: Exp[Array[T]], n: Exp[Int], y: Exp[T])(implicit pos: SourceContext) = {
    if (context ne null) {
      // find the last modification of array x
      // if it is an assigment at index n with the same value, just do nothing
      val vs = x.asInstanceOf[Sym[Array[T]]]
      //TODO: could use calculateDependencies?

      val rhs = context.reverse.collectFirst {
        //case w @ Def(Reflect(ArrayNew(sz: Exp[T]), _, _)) if w == x => Some(Const(())) // FIXME: bounds check!
        case Def(Reflect(ArrayUpdate(`x`, `n`, `y`), _, _)) => Some(Const(()))
        case Def(Reflect(_, u, _)) if mayWrite(u, List(vs)) => None // not a simple assignment
      }
      rhs.flatten.getOrElse(super.array_update(x,n,y))
    } else {
      reflectEffect(ArrayUpdate(x,n,y))
    }
  }
  */

  // TODO: should this be in LMS?
  override def isPrimitiveType[T](m: Manifest[T]) = (m == manifest[String]) || super.isPrimitiveType(m)

  // should probably add to LMS
  def mutableStaticData[T:Manifest](x: T): Exp[T] = reflectMutable(StaticData(x))

  override def doApply[A:Manifest,B:Manifest](f: Exp[A => B], x: Exp[A])(implicit pos: SourceContext): Exp[B] = {
    val x1 = unbox(x)
    val x1_effects = x1 match {
      case UnboxedTuple(l) => l.foldLeft(Pure())((b,a)=>a match {
        case Def(Lambda(_, _, yy)) => b orElse summarizeEffects(yy)
        case _ => b
        })
      case _ => Pure()
    }
    f match {
      case Def(Lambda(_, _, y)) => reflectEffect(Apply(f, x1), summarizeEffects(y) andAlso x1_effects)
      case _ => reflectEffect(Apply(f, x1), Simple() andAlso x1_effects)
    }
  }

  // add for fun with 6 or more parameters
  // override def unbox[A:Manifest](x : Exp[A])(implicit pos: SourceContext) : Exp[A] = {
  //   val mA = implicitly[Manifest[A]]
  //   x match {
  //     case _ if tupledManifestOf(mA, 6) =>
  //       x match { case t : Rep[(a1,a2,a3,a4,a5,a6)] =>
  //         UnboxedTuple[A](List(
  //           tuple6_get1(t)(mA.typeArguments(0).asInstanceOf[Manifest[a1]], pos),
  //           tuple6_get2(t)(mA.typeArguments(1).asInstanceOf[Manifest[a2]], pos),
  //           tuple6_get3(t)(mA.typeArguments(2).asInstanceOf[Manifest[a2]], pos),
  //           tuple6_get4(t)(mA.typeArguments(3).asInstanceOf[Manifest[a2]], pos),
  //           tuple6_get5(t)(mA.typeArguments(4).asInstanceOf[Manifest[a2]], pos),
  //           tuple6_get6(t)(mA.typeArguments(5).asInstanceOf[Manifest[a2]], pos)))
  //       }
  //     case _ => super.unbox(x)
  //   }
  // }
}

trait DslGPUExp extends DslExp with GPUOpsExp

trait DslGenScala extends ScalaGenNumericOps
    with ScalaGenPrimitiveOps with ScalaGenBooleanOps with ScalaGenIfThenElse
    with ScalaGenEqual with ScalaGenRangeOps with ScalaGenOrderingOps
    with ScalaGenMiscOps with ScalaGenArrayOps with ScalaGenStringOps
    with ScalaGenFunctions with ScalaGenWhile
    with ScalaGenStaticData with ScalaGenVariables
    with ScalaGenUtilOps with ScalaGenMathOps with ScalaGenTupledFunctions
    with ScalaGenCastingOps {
  val IR: DslExp

  import IR._

  override def quote(x: Exp[Any]) = x match {
    case Const('\n') if x.tp == manifest[Char] => "'\\n'"
    case Const('\t') if x.tp == manifest[Char] => "'\\t'"
    case Const(0)    if x.tp == manifest[Char] => "'\\0'"
    case _ => super.quote(x)
  }

  override def emitNode(sym: Sym[Any], rhs: Def[Any]) = rhs match {
    case afs@ArrayFromSeq(xs) =>
      stream.println(remap(afs.m) + " " + quote(sym) + "[" + xs.length + "] = {" + (xs mkString ",") + "}")
    case Assign(Variable(a), b) =>
      emitAssignment(a.asInstanceOf[Sym[Variable[Any]]], quote(b))
    case IfThenElse(c,Block(Const(true)),Block(Const(false))) =>
      emitValDef(sym, quote(c))
    case PrintF(f:String,xs) =>
      emitValDef(sym, src"printf(${Const(f)::xs})")
    case RawCode(s) =>
      stream.println(s)
    case RawComment(s) =>
      stream.println("// "+s)
    case Comment(s, verbose, b) =>
      stream.println("val " + quote(sym) + " = {")
      stream.println("//#" + s)
      if (verbose) {
        stream.println("// generated code for " + s.replace('_', ' '))
      } else {
        stream.println("// generated code")
      }
      emitBlock(b)
      stream.println(quote(getBlockResult(b)))
      stream.println("//#" + s)
      stream.println("}")
    // case FieldApply() => super.emitNode(sym, rhs)
    // case FieldApply(a, "_1") => emitValDef(sym, quote(a) + "._1")
    // case FieldApply(a, "_2") => emitValDef(sym, quote(a) + "._2")
    case _ => super.emitNode(sym, rhs)
  }

  override def getFreeDataExp[A](sym: Sym[A], rhs: Def[A]): List[(Sym[Any],Any)] = rhs match {
    case Reflect(StaticData(x), _, _) => List((sym,x))
    case _ => super.getFreeDataExp(sym, rhs)
  }
}

// TODO: currently part of this is specific to the query tests. generalize? move?
trait DslGenBase extends CGenNumericOpsExtra
    with CGenPrimitiveOps with CGenBooleanOps with CGenIfThenElse
    with CGenEqual with CGenRangeOps with CGenOrderingOps
    with CGenMiscOps with CGenArrayOps with CGenStringOps
    with CGenFunctions with CGenWhile
    with CGenStaticData with CGenVariables
    with CGenUtilOps with CGenUncheckedOps with CGenMathOps with CGenTupledFunctions
    with CGenCastingOps {
  val IR: DslExp
  import IR._

  def getMallocString(count: String, dataType: String): String = {
      "(" + dataType + "*)malloc(" + count + " * sizeof(" + dataType + "));"
  }

  def getMallocArenaString(count: String, dataType: String): String = {
      "(" + dataType + "*)myMalloc(" + count + " * sizeof(" + dataType + "));"
  }

  // In LMS code, it was "remap(m) + addRef(m)" which would put an extra "*"
  override def remapWithRef[A](m: Manifest[A]): String = remap(m) + " "

  def unwrapTupleStr(s: String): Array[String] = {
    if (s.startsWith("scala.Tuple")) s.slice(s.indexOf("[")+1,s.length-1).filter(c => c != ' ').split(",")
    else scala.Array(s)
  }

  override def remap[A](m: Manifest[A]): String = m.toString match {
    case "Any" => "NOOOOOOOOOO"
    case "java.lang.String" => "char*"
    case "Char" => "char"

    case "Array[Char]" => "char*"
    case "Array[Double]" => "double*"
    case "Array[Int]"    => "int*"
    case "Array[Long]"    => "int64_t*"
    case "Array[Float]"  => "float*"
    case "Array[Array[Int]]" => "int**"
    case "Array[Array[Double]]" => "double**"
    case "Array[Array[Float]]"  => "float**"

    case f if f.startsWith("scala.Function") =>
      val targs = m.typeArguments.dropRight(1)
      val res = remap(m.typeArguments.last)
      // val targsUnboxed = targs.flatMap(t => unwrapTupleStr(remap(t)))
      // val sep = if (targsUnboxed.length > 0) "," else ""
      def remapInFunction[A](m: Manifest[A]): Array[String] = {
        val s = m.toString
        if (s.startsWith("scala.Tuple")) m.typeArguments.map(t => remap(t)).toArray
        else scala.Array(remap(m))
      }
      val targsUnboxed = targs.flatMap(t => remapInFunction(t))
      "function<" + res + "(" + targsUnboxed.mkString(",") + ")>"

    // scala.Function1[Array[Double], Array[Double]] --> function<double*(double*)>
    case _ => super.remap(m)
  }

  override def format(s: Exp[Any]): String = {
    remap(s.tp) match {
      case "uint16_t" => "%c"
      case "bool" | "int8_t" | "int16_t" | "int32_t" => "%d"
      case "int64_t" => "%ld"
      case "float" | "double" => "%f"
      case "string" => "%s"
      case "char*" => "%s"
      case "char" => "%c"
      case "void" => "%c"
      case _ =>
        import scala.virtualization.lms.internal.GenerationFailedException
        throw new GenerationFailedException("CGenMiscOps: cannot print type " + remap(s.tp))
    }
  }
  override def quoteRawString(s: Exp[Any]): String = {
    remap(s.tp) match {
      case "string" => quote(s) + ".c_str()"
      case _ => quote(s)
    }
  }
  // we treat string as a primitive type to prevent memory management on strings
  // strings are always stack allocated and freed automatically at the scope exit
  override def isPrimitiveType(tpe: String) : Boolean = {
    tpe match {
      case "char*" => true
      case "char" => true
      case _ => super.isPrimitiveType(tpe)
    }
  }
  // XX: from LMS 1.0
  override def emitValDef(sym: Sym[Any], rhs: String): Unit = {
    if (!isVoidType(sym.tp))
      stream.println(remapWithRef(sym.tp) + quote(sym) + " = " + rhs + ";")
    else // we might still want the RHS for its effects
      stream.println(rhs + ";")
  }

  override def quote(x: Exp[Any]) = x match {
    case Const(s: String) => "\""+s.replace("\"", "\\\"")+"\"" // TODO: more escapes?
    case Const('\n') if x.tp == manifest[Char] => "'\\n'"
    case Const('\t') if x.tp == manifest[Char] => "'\\t'"
    case Const(0)    if x.tp == manifest[Char] => "'\\0'"
    case _ => super.quote(x)
  }

  override def emitNode(sym: Sym[Any], rhs: Def[Any]) = rhs match {
    case CharToInt(s) => stream.println(s"int32_t ${quote{sym}} = (int32_t) ${quote(s)};")
    case Error(s) => stream.println("assert(false && " + quote(s) + ");")
    case afs@ArrayFromSeq(xs) =>
      stream.println(remap(afs.m) + " " + quote(sym) + "[" + xs.length + "] = {" + (xs map quote mkString ",") + "};")
    case a@ArrayNew(n) =>
      val arrType = remap(a.m)
      // emitValDef(sym, getMallocString(quote(n), arrType))
      emitValDef(sym, getMallocArenaString(quote(n), arrType))
      // stream.println("unique_ptr<" + arrType + "[]> " + quote(sym) + "(new " + arrType + "[" + quote(n) + "]);")
      // stream.println("shared_ptr<" + arrType + "[]> " + quote(sym) + "(new " + arrType + "[" + quote(n) + "]);")
    case ArrayApply(x,n) => emitValDef(sym, quote(x) + "[" + quote(n) + "]")
    case ArrayUpdate(x,n,y) => stream.println(quote(x) + "[" + quote(n) + "] = " + quote(y) + ";")
    case PrintLn(s) => stream.println("printf(\"" + format(s) + "\\n\"," + quoteRawString(s) + ");")
    case StringCharAt(s,i) => emitValDef(sym, "%s[%s]".format(quote(s), quote(i)))
    case RawCode(s) =>
      stream.println(s)
    case RawComment(s) =>
      stream.println("// "+s)
    case Comment(s, verbose, b) =>
      stream.println("//#" + s)
      if (verbose) {
        stream.println("// generated code for " + s.replace('_', ' '))
      } else {
        stream.println("// generated code")
      }
      emitBlock(b)
      emitValDef(sym, quote(getBlockResult(b)))
      stream.println("//#" + s)
    case MathTanh(x) => emitValDef(sym, src"tanh($x)")
    // // add for fun with 6 or more parameters
    // case FieldApply(UnboxedTuple(vars), "_6") => emitValDef(sym, quote(vars(5))){}
    case _ => super.emitNode(sym,rhs)
  }

  // List of header files, to be imported in the code template.
  def templateHeaders: Seq[String] = Seq(
    "<assert.h>", "<err.h>", "<errno.h>", "<fcntl.h>", "<functional>",
    "<math.h>", "<memory>", "<random>", "<stdint.h>", "<stdio.h>",
    "<sys/mman.h>", "<sys/stat.h>", "<sys/time.h>", "<time.h>", "<unistd.h>", "<cblas.h>")

  // Raw code, to be included in the code template at file scope, before the main function.
  def templateRawCode: String = ""

  override def emitSource[A:Manifest](args: List[Sym[_]], body: Block[A], functionName: String, out: java.io.PrintWriter) = {
    withStream(out) {
      stream.println(templateHeaders.map(x => s"#include $x").mkString("\n"))
      stream.println(raw"""
        |using namespace std;
        |#ifndef MAP_FILE
        |#define MAP_FILE MAP_SHARED
        |#endif
        |
        |int fsize(int fd) {
        |  struct stat stat;
        |  int res = fstat(fd, &stat);
        |  return stat.st_size;
        |}
        |
        |int printll(char *s) {
        |  while (*s != '\n' && *s != ',' && *s != '\t') {
        |    putchar(*s++);
        |  }
        |  return 0;
        |}
        |
        |long hash(char *str0, int len) {
        |  unsigned char *str = (unsigned char *)str0;
        |  unsigned long hash = 5381;
        |  int c;
        |
        |  while ((c = *str++) && len--)
        |    hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
        |
        |  return hash;
        |}
        |
        |long HEAP_SIZE_CPU = 1073741826; // 1048576; // 536870912; // 268435456; // 2097152; 1610612739; // 4294967304; //
        |void *mallocBase = calloc(HEAP_SIZE_CPU, 1);
        |void *mallocAddr = mallocBase;
        |void *waterMark = mallocBase;
        |void *myMalloc(size_t bytes) {
        |  void *res = mallocAddr;
        |  mallocAddr = (void *)((char *)mallocAddr + bytes);
        |  if ((long)mallocAddr >= (long)mallocBase + HEAP_SIZE_CPU)
        |    fprintf(stderr, "CPU memory breached limit of HEAP_SIZE_CPU\n");
        |  return res;
        |}
        |
        |long HEAP_SIZE = 4294967304; // this is for GPU
        |
        |int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1) {
        |  long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
        |  result->tv_sec = diff / 1000000;
        |  result->tv_usec = diff % 1000000;
        |  return (diff < 0);
        |}
        |
        |$templateRawCode
        |
        |void Snippet(char *);
        |
        |std::random_device rd{};
        |std::mt19937 gen{rd()};
        |std::normal_distribution<> d{0, 0.01};
        |
        |int main(int argc, char *argv[]) {
        |  if (argc != 2) {
        |    printf("usage: query <filename>\n");
        |    return 0;
        |  }
        |  Snippet(argv[1]);
        |  return 0;
        |}
        |""".stripMargin)
    }
    super.emitSource[A](args, body, functionName, out)
  }
}

trait DslGenC extends DslGenBase {
  val IR: DslExp
  import IR._
}

trait DslGenCublas extends DslGenBase with CudaGenGPUOps {
  val IR: DslGPUExp
  import IR._

  override def templateHeaders: Seq[String] =
    super.templateHeaders ++ Seq("<cuda.h>", "<cuda_runtime.h>", "<cublas_v2.h>")

  override def templateRawCode: String = super.templateRawCode +
    """
      |#define CUDA_CALL(f) { \
      |  cudaError_t err = (f); \
      |  if (err != cudaSuccess) { \
      |    fprintf(stderr, "CUDA error occurred: %s (%s:%d)\n", \
      |            cudaGetErrorString(err), __FILE__, __LINE__); \
      |    exit(err); \
      |  } \
      |}
      |
      |#define CUBLAS_CALL(f) { \
      |  cublasStatus_t stat = (f); \
      |  if (stat != CUBLAS_STATUS_SUCCESS) { \
      |    fprintf(stderr, "cuBLAS error occurred: %d (%s:%d)\n", \
      |            stat, __FILE__, __LINE__); \
      |    exit(stat); \
      |  } \
      |}
      |
      |void *gpuMallocBase;
      |void *gpuMallocAddr;
      |
      |// Alignment boundary size, in bytes.
      |constexpr int N = 4; // 16
      |void *myGpuMalloc(size_t bytes) {
      |  bytes = ((bytes + (1 << N) - 1) >> N) << N;
      |  void *res = gpuMallocAddr;
      |  gpuMallocAddr = (void *)((char *)gpuMallocAddr + bytes);
      |  if ((long)gpuMallocAddr >= (long)gpuMallocBase + HEAP_SIZE)
      |    fprintf(stderr, "GPU breached memory limit of HEAP_SIZE\n");
      |  return res;
      |}
      |
      |template <typename T>
      |__global__ void arrayUpdate(T *data, int index, T value) {
      |  data[index] = value;
      |}
      |
      |__global__ void arrayFill(float *data, float value) {
      |  int tid = threadIdx.x + blockIdx.x * blockDim.x;
      |  data[tid] = value;
      |}
      |__global__ void arrayFill_greg(float* data, float value, int size) {
      |  int stride = gridDim.x * blockDim.x;
      |  int tid = threadIdx.x + blockIdx.x * blockDim.x;
      |  for (int i = tid; i < size; i += stride) data[i] = value;
      |}
      |
      |__global__ void hardTanh(float* in, float* out, float min_val, float max_val, int size) {
      |  int tid = threadIdx.x + blockIdx.x * blockDim.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (int i = tid; i < size; i += stride) {
      |    out[i] = in[i] < min_val ? min_val : (in[i] > max_val ? max_val : in[i]);
      |  }
      |}
      |
      |__global__ void hardTanh_grad(float* in_x, float* in_d, float* out_d, float min_val, float max_val, int size, bool inplace) {
      |  int tid = threadIdx.x + blockIdx.x * blockDim.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (int i = tid; i < size; i += stride) {
      |    if (inplace) {
      |      if (in_x[i] < min_val || in_x[i] > max_val) in_d[i] = 0;
      |    } else {
      |      if (in_x[i] >= min_val && in_x[i] <= max_val) in_d[i] += out_d[i];
      |    }
      |  }
      |}
      |
      |__global__ void nllLoss(float *x, int x_stride, float *y, int* target) {
      |  int tid = threadIdx.x + blockIdx.x * blockDim.x;
      |  int offset = tid * x_stride + target[tid];
      |  y[tid] = -1 * x[offset];
      |}
      |
      |__global__ void nllLoss_grad(int x_stride, float *yGrad, int* target, float* xGrad) {
      |  int tid = threadIdx.x + blockIdx.x * blockDim.x;
      |  int offset = tid * x_stride + target[tid];
      |  xGrad[offset] += -1 * yGrad[tid];
      |}
      |
      | // only for 4D tensor in and 3D tensor out
      |__global__ void sum_grad(float* in, int inSize0, int inSize1, int inSize2, int inSize3, int nElement,
      |                         float* out, int outStride0, int outStride1, int outStride2, int dim) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (int i = tid; i < nElement; i += stride) {
      |    int inOff2 = i / inSize3;
      |    int inDim3 = i - inOff2 * inSize3;
      |    int inOff1 = inOff2 / inSize2;
      |    int inDim2 = inOff2 - inOff1 * inSize2;
      |    int inDim0 = inOff1 / inSize1;
      |    int inDim1 = inOff1 - inDim0 * inSize1;
      |    int outOff = 0;
      |    if (dim == 0) outOff = inDim1 * outStride0 + inDim2 * outStride1 + inDim3 * outStride2;
      |    if (dim == 1) outOff = inDim0 * outStride0 + inDim2 * outStride1 + inDim3 * outStride2;
      |    if (dim == 2) outOff = inDim0 * outStride0 + inDim1 * outStride1 + inDim3 * outStride2;
      |    if (dim == 3) outOff = inDim0 * outStride0 + inDim1 * outStride1 + inDim2 * outStride2;
      |    in[i] += out[outOff];
      |  }
      |}
      |
      |//following - https://github.com/torch/cutorch/blob/master/lib/THC/THCTensorMath.cuh#L49
      |static inline __device__ int compute(int outputSize0, int outputSize1, int outputSize2, int outputSize3,
      |                                     int outputStride0, int outputStride1, int outputStride2, int outputStride3,
      |                                     const int dimSize, const int concatDim, int linearIndex) {
      |  int offset = 0;
      |  int curDimSize = 3 == concatDim ? dimSize : outputSize3;
      |  int nextDimIndex = linearIndex / curDimSize;
      |  int curDimIndex = linearIndex - curDimSize * nextDimIndex;
      |  int curDimOffset = curDimIndex * outputStride3;
      |  offset += curDimOffset;
      |  linearIndex = nextDimIndex;
      |  curDimSize = 2 == concatDim ? dimSize : outputSize2;
      |  nextDimIndex = linearIndex / curDimSize;
      |  curDimIndex = linearIndex - curDimSize * nextDimIndex;
      |  curDimOffset = curDimIndex * outputStride2;
      |  offset += curDimOffset;
      |  linearIndex = nextDimIndex;
      |  curDimSize = 1 == concatDim ? dimSize : outputSize1;
      |  nextDimIndex = linearIndex / curDimSize;
      |  curDimIndex = linearIndex - curDimSize * nextDimIndex;
      |  curDimOffset = curDimIndex * outputStride1;
      |  offset += curDimOffset;
      |  linearIndex = nextDimIndex;
      |  return offset + linearIndex * outputStride0;
      |//  for (int i = 3; i >= 1; i--) {
      |//    int curDimSize = i == concatDim ? dimSize : outputSize[i];
      |//    int nextDimIndex = linearIndex / curDimSize;
      |//    int curDimIndex = linearIndex - curDimSize * nextDimIndex;
      |//    int curDimOffset = curDimIndex * outputStride[i];
      |//    offset += curDimOffset;
      |//    linearIndex = nextDimIndex;
      |//  }
      |//  return offset + linearIndex * outputStride[0];
      |}
      |
      |// TODO: Only for Dim of rank 4, and only for 2 inputs, and only for concat at dim = 1
      |__global__ void concat2D_1D_greg(float* in1, int dimSize1, int nElement1,
      |                                 float* in2, int dimSize2, int nElement2,
      |                                 float* out, int concatDim,
      |                                 int outSize0, int outSize1, int outSize2, int outSize3,
      |                                 int outStride0, int outStride1, int outStride2, int outStride3) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int nElement = blockIdx.y == 0 ? nElement1 : nElement2;
      |  if (tid >= nElement) return;
      |  float* data = blockIdx.y == 0 ? in1 : in2;
      |  int offset = blockIdx.y == 0 ? 0 : dimSize1;
      |  int dimSize = blockIdx.y == 0 ? dimSize1 : dimSize2;
      |  int dataOffset = offset * outStride1;
      |  int stride = gridDim.x * blockDim.x;
      |  while (tid < nElement) {
      |    int elementOffset = compute(outSize0, outSize1, outSize2, outSize3,
      |                                outStride0, outStride1, outStride2, outStride3, dimSize, concatDim, tid);
      |    out[dataOffset + elementOffset] = data[tid];
      |    tid += stride;
      |  }
      |}
      |
      |// TODO: Only for Dim of rank 4, and only for 2 inputs, and only for concat at dim = 1
      |__global__ void concat2D_1D_greg_grad(float* in1, int dimSize1, int nElement1,
      |                                      float* in2, int dimSize2, int nElement2,
      |                                      float* out, int concatDim,
      |                                      int outSize0, int outSize1, int outSize2, int outSize3,
      |                                      int outStride0, int outStride1, int outStride2, int outStride3) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int nElement = blockIdx.y == 0 ? nElement1 : nElement2;
      |  if (tid >= nElement) return;
      |  float* data = blockIdx.y == 0 ? in1 : in2;
      |  int offset = blockIdx.y == 0 ? 0 : dimSize1;
      |  int dimSize = blockIdx.y == 0 ? dimSize1 : dimSize2;
      |  int dataOffset = offset * outStride1;
      |  int stride = gridDim.x * blockDim.x;
      |  while (tid < nElement) {
      |    int elementOffset = compute(outSize0, outSize1, outSize2, outSize3,
      |                                outStride0, outStride1, outStride2, outStride3, dimSize, concatDim, tid);
      |    data[tid] += out[dataOffset + elementOffset];
      |    tid += stride;
      |  }
      |}
      |
      |__global__ void concat2D_1D_loop(float* in1, float* in2, float* out, int sizeLow, int sizeHigh, int sizeDim1, int sizeDim2) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  if (tid >= sizeLow) return;
      |  if (blockIdx.y < sizeHigh) { // the first input
      |    int index_out = tid + blockIdx.y * sizeLow * (sizeDim1 + sizeDim2);
      |    int index_in1 = tid + blockIdx.y * sizeLow * sizeDim1;
      |    for (int i = 0; i < sizeDim1; i++) {
      |      out[index_out] = in1[index_in1];
      |      index_out += sizeLow; index_in1 += sizeLow;
      |    }
      |  } else { // the second input
      |    int index_out = tid + (blockIdx.y - sizeHigh) * sizeLow * (sizeDim1 + sizeDim2) + sizeLow * sizeDim1;
      |    int index_in2 = tid + (blockIdx.y - sizeHigh) * sizeLow * sizeDim2;
      |    for (int i = 0; i < sizeDim2; i++) {
      |      out[index_out] = in2[index_in2];
      |      index_out += sizeLow; index_in2 += sizeLow;
      |    }
      |  }
      |}
      |
      |__global__ void concat2D_1D_loop_grad(float* in1, float* in2, float* out, int sizeLow, int sizeHigh, int sizeDim1, int sizeDim2) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  if (tid >= sizeLow) return;
      |  if (blockIdx.y < sizeHigh) { // the first input
      |    int index_out = tid + blockIdx.y * sizeLow * (sizeDim1 + sizeDim2);
      |    int index_in1 = tid + blockIdx.y * sizeLow * sizeDim1;
      |    for (int i = 0; i < sizeDim1; i++) {
      |      in1[index_in1] += out[index_out];
      |      index_out += sizeLow; index_in1 += sizeLow;
      |    }
      |  } else { // the second input
      |    int index_out = tid + (blockIdx.y - sizeHigh) * sizeLow * (sizeDim1 + sizeDim2) + sizeLow * sizeDim1;
      |    int index_in2 = tid + (blockIdx.y - sizeHigh) * sizeLow * sizeDim2;
      |    for (int i = 0; i < sizeDim2; i++) {
      |      in2[index_in2] += out[index_out];
      |      index_out += sizeLow; index_in2 += sizeLow;
      |    }
      |  }
      |}
      |
      |__global__ void concat2D_1D(float* in1, float* in2, float* out, int dim2, int bound) {
      |  int tid = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
      |  if (blockIdx.x < bound * dim2) {
      |    int subid = blockIdx.y * bound * dim2 * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
      |    out[tid] = in1[subid];
      |  } else {
      |    int subid = blockIdx.y * (gridDim.x - bound * dim2) * blockDim.x + (blockIdx.x - bound * dim2) * blockDim.x + threadIdx.x;
      |    out[tid] = in2[subid];
      |  }
      |}
      |
      |__global__ void concat2D_1D_grad(float* in1, float* in2, float* out, int dim2, int bound) {
      |  int tid = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
      |  if (blockIdx.x < bound * dim2) {
      |    int subid = blockIdx.y * bound * dim2 * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
      |    in1[subid] += out[tid];
      |  } else {
      |    int subid = blockIdx.y * (gridDim.x - bound * dim2) * blockDim.x + (blockIdx.x - bound * dim2) * blockDim.x + threadIdx.x;
      |    in2[subid] += out[tid];
      |  }
      |}
      |
      |__global__ void adagrad_update_1D_1D(float* x, float* d, float* m, float clip, float lr, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  if (tid < size) {
      |    if (d[tid] > clip) d[tid] = clip;
      |    if (d[tid] < -clip) d[tid] = -clip;
      |    m[tid] += d[tid] * d[tid];
      |    x[tid] -= lr * d[tid] / sqrt(m[tid] + 0.00000001);
      |    d[tid] = 0;
      |  }
      |}
      |
      |__global__ void elementwise_1D_1D_mul(float* in1, float* in2, float* out, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  if (tid < size) out[tid] = in1[tid] * in2[tid];
      |}
      |
      |__global__ void elementwise_1D_1D_mul_mutate(float* in1, float* in2, float* out, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  if (tid < size) out[tid] += in1[tid] * in2[tid];
      |}
      |
      |__global__ void elementwise_1D_1D_add(float* in1, float* in2, float* out, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  if (tid < size) out[tid] = in1[tid] + in2[tid];
      |}
      |
      |__global__ void elementwise_1D_1D_minus(float* in1, float* in2, float* out, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  if (tid < size) out[tid] = in1[tid] - in2[tid];
      |}
      |
      |__global__ void elementwise_1D_1D_div(float* in1, float* in2, float* out, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  if (tid < size) out[tid] = in1[tid] / in2[tid];
      |}
      |
      |__global__ void elementwise_1D_1D_exp(float* in, float* out, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  if (tid < size) out[tid] = exp(in[tid]);
      |}
      |__global__ void elementwise_1D_1D_log(float* in, float* out, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  if (tid < size) out[tid] = log(in[tid]);
      |}
      |__global__ void elementwise_1D_1D_sqrt(float* in, float* out, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  if (tid < size) out[tid] = sqrt(in[tid]);
      |}
      |
      |__global__ void elementwise_1D_1D_square(float* in, float* out, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  if (tid < size) out[tid] = in[tid] * in[tid];
      |}
      |
      |__global__ void elementwise_1D_1D_exp_grad(float* in_x, float* in_d, float* out_x, float * out_d, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  if (tid < size) in_d[tid] += out_d[tid] * out_x[tid];
      |}
      |__global__ void elementwise_1D_1D_log_grad(float* in_x, float* in_d, float* out_x, float * out_d, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  if (tid < size) in_d[tid] += out_d[tid] / in_x[tid];
      |}
      |__global__ void elementwise_1D_1D_sqrt_grad(float* in_x, float* in_d, float* out_x, float * out_d, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  if (tid < size) in_d[tid] += out_d[tid] / out_x[tid] / 2;
      |}
      |
      |__global__ void elementwise_1D_1D_square_grad(float* in_x, float* in_d, float* out_x, float * out_d, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  if (tid < size) in_d[tid] += out_d[tid] * 2 * in_x[tid];
      |}
      |
      |// From: https://github.com/pytorch/pytorch/blob/master/aten/src/THC/THCIntegerDivider.cuh
      |// Result of div/mod operation stored together.
      |template <typename Value>
      |struct DivMod {
      |  Value div, mod;
      |
      |  __host__ __device__ DivMod(Value div, Value mod) : div(div), mod(mod) { }
      |};
      |
      |// Base case: we only have an implementation for uint32_t for now.  For
      |// everything else, we use plain division.
      |template <typename Value>
      |struct IntDivider {
      |  IntDivider() { }  // Dummy constructor for arrays.
      |  IntDivider(Value d) : divisor(d) { }
      |
      |  __host__ __device__ inline Value div(Value n) const { return n / divisor; }
      |  __host__ __device__ inline Value mod(Value n) const { return n % divisor; }
      |  __host__ __device__ inline DivMod<Value> divmod(Value n) const {
      |    return DivMod<Value>(n / divisor, n % divisor);
      |  }
      |
      |  Value divisor;
      |};
      |
      |// Implement fast integer division.
      |template <>
      |struct IntDivider<unsigned int> {
      |  static_assert(sizeof(unsigned int) == 4, "Assumes 32-bit unsigned int.");
      |
      |  IntDivider() { }  // Dummy constructor for arrays.
      |
      |  IntDivider(unsigned int d) : divisor(d) {
      |    assert(divisor >= 1 && divisor <= INT32_MAX);
      |
      |    // TODO: gcc/clang has __builtin_clz() but it's not portable.
      |    for (shift = 0; shift < 32; shift++) if ((1U << shift) >= divisor) break;
      |
      |    uint64_t one = 1;
      |    uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;
      |    m1 = magic;
      |    assert(m1 > 0 && m1 == magic);  // m1 must fit in 32 bits.
      |  }
      |
      |  __host__ __device__ inline unsigned int div(unsigned int n) const {
      |#ifdef __CUDA_ARCH__
      |    // 't' is the higher 32-bits of unsigned 32-bit multiplication of 'n' and
      |    // 'm1'.
      |    unsigned int t = __umulhi(n, m1);
      |    return (t + n) >> shift;
      |#else
      |    // Using uint64_t so that the addition does not overflow.
      |    uint64_t t = ((uint64_t) n * m1) >> 32;
      |    return (t + n) >> shift;
      |#endif
      |  }
      |
      |  __host__ __device__ inline unsigned int mod(unsigned int n) const {
      |    return n - div(n) * divisor;
      |  }
      |
      |  __host__ __device__ inline DivMod<unsigned int> divmod(unsigned int n) const {
      |    unsigned int q = div(n);
      |    return DivMod<unsigned int>(q, n - q * divisor);
      |  }
      |
      |  unsigned int divisor;  // d above.
      |  unsigned int m1;  // Magic number: m' above.
      |  unsigned int shift;  // Shift amounts.
      |};
      |
      |// From: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/OffsetCalculator.cuh
      |/// OffsetCalculator calculates the offset in bytes of a linear index for NARGS
      |/// operands that share the same shape, but may have different strides.
      |
      |template <int NARGS>
      |struct OffsetCalculator {
      |  static constexpr int MAX_DIMS = 25;
      |
      |  // The offset for each argument (in bytes). Wrapper around fixed-size array.
      |  struct offsets_t {
      |    __host__ __device__ uint32_t& operator[](int idx) {
      |      return values[idx];
      |    }
      |    uint32_t values[NARGS];
      |  };
      |
      |
      |  // OffsetCalculator(int dims, const int64_t* sizes, const int64_t* const* strides) : dims(dims) {
      |  OffsetCalculator(int dims, const int32_t* sizes, const int32_t* const* strides) : dims(dims) {
      |    for (int i = 0; i < MAX_DIMS; ++i) {
      |      if (i < dims) {
      |        sizes_[i] = IntDivider<uint32_t>(sizes[i]);
      |      } else {
      |        sizes_[i] = IntDivider<uint32_t>(1);
      |      }
      |      for (int arg = 0; arg < NARGS; arg++) {
      |        strides_[i][arg] = i < dims ? strides[arg][i] : 0;
      |      }
      |    }
      |  }
      |
      |  __host__ __device__ offsets_t get(uint32_t linear_idx) const {
      |    offsets_t offsets;
      |#pragma unroll
      |    for (int arg = 0; arg < NARGS; arg++) {
      |      offsets[arg] = 0;
      |    }
      |
      |#pragma unroll
      |    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      |      if (dim == dims) {
      |        break;
      |      }
      |      auto divmod = sizes_[dim].divmod(linear_idx);
      |      linear_idx = divmod.div;
      |
      |#pragma unroll
      |      for (int arg = 0; arg < NARGS; arg++) {
      |        offsets[arg] += divmod.mod * strides_[dim][arg];
      |      }
      |    }
      |    return offsets;
      |  }
      |
      |  void print() {
      |    for (auto i = 1; i < 128; i++) {
      |      auto offsets = get(i);
      |      printf("offsets[%d]: ", i);
      |      for (auto arg = 0; arg < NARGS; arg++) {
      |        printf("%d ", offsets[arg]);
      |      }
      |      printf("\n");
      |    }
      |  }
      |
      |  int dims;
      |  IntDivider<uint32_t> sizes_[MAX_DIMS];
      |  uint32_t strides_[MAX_DIMS][NARGS];
      |};
      |
      |// From: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/Loops.cuh
      |template<int nt, int vt, typename func_t>
      |__launch_bounds__(nt, 4)
      |__global__ void elementwise_kernel(int N, func_t f) {
      |  int tid = threadIdx.x;
      |  int nv = nt * vt;
      |  int idx = nv * blockIdx.x + tid;
      |#pragma unroll
      |  for (int i = 0; i < vt; i++) {
      |    if (idx < N) {
      |      f(idx);
      |      idx += nt;
      |    }
      |  }
      |}
      |
      |template<int nt, int vt, typename func_t>
      |static void launch_kernel(int64_t N, const func_t& f) {
      |  if (N == 0) {
      |    return;
      |  }
      |  dim3 block(nt);
      |  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
      |  elementwise_kernel<nt, vt, func_t><<<grid, block, 0>>>(N, f);
      |}
      |
      |template<typename func_t>
      |void gpu_unary_kernel(float *res, float *x,
      |                      int32_t resRank, const int32_t resScalarCount,
      |                      const int32_t* resShape, const int32_t* const* strides,
      |                      const func_t& f) {
      |  OffsetCalculator<2> calc(resRank, resShape, strides);
      |  launch_kernel<128, 4>(resScalarCount, [=]__device__(int idx) {
      |    auto offsets = calc.get(idx);
      |    float* out = &res[offsets[0]];
      |    float* in = &x[offsets[1]];
      |    *out = f(*in);
      |  });
      |}
      |
      |template<typename func_t>
      |void gpu_binary_kernel(float *res, float *x, float *y,
      |                       int32_t resRank, const int32_t resScalarCount,
      |                       const int32_t* resShape, const int32_t* const* strides,
      |                       const func_t& f) {
      |  OffsetCalculator<3> calc(resRank, resShape, strides);
      |  launch_kernel<128, 4>(resScalarCount, [=]__device__(int idx) {
      |    auto offsets = calc.get(idx);
      |    float* out = &res[offsets[0]];
      |    float* in1 = &x[offsets[1]];
      |    float* in2 = &y[offsets[2]];
      |    *out = f(*in1, *in2);
      |  });
      |}
      |""".stripMargin

}

trait DslGenCudnn extends DslGenCublas {
  val IR: DslGPUExp
  import IR._

  override def templateHeaders: Seq[String] = super.templateHeaders ++ Seq("<cudnn.h>")
  override def templateRawCode: String = super.templateRawCode +
    """
      |#define CUDNN_CALL(f) { \
      |  cudnnStatus_t stat = (f); \
      |  if (stat != CUDNN_STATUS_SUCCESS) { \
      |    fprintf(stderr, "cuDNN error occurred: %d (%s:%d)\n", \
      |            stat, __FILE__, __LINE__); \
      |    exit(stat); \
      |  } \
      |}
      |""".stripMargin
}

abstract class DslDriverScala[A: Manifest, B: Manifest] extends DslOps with DslExp with CompileScala { self =>
  val codegen = new DslGenScala {
    val IR: self.type = self
  }

  def snippet(x: Rep[A]): Rep[B]

  lazy val f = compile[A,B](snippet)
  def precompile: Unit = f

  // def precompileSilently: Unit = utils.devnull(f)

  def eval(x: A): B = f(x)

  lazy val code: String = {
    val source = new java.io.StringWriter()
    codegen.emitSource[A,B](snippet, "Snippet", new java.io.PrintWriter(source))
    source.toString
  }
}

abstract class DslDriverBase[A: Manifest, B: Manifest] extends DslExp { self =>
  // The C-like code generator.
  val codegen: DslGenBase {
    val IR: self.type
  }

  val dir = "/tmp"
  val fileName = s"lantern-snippet-${scala.util.Random.alphanumeric.take(4).mkString}"

  // The code snippet to compile.
  def snippet(x: Rep[A]): Rep[B]

  def eval(a: A)
  // Note: is it possible to implement `eval` returning `B`?
  // def eval(a: A): B

  lazy val code: String = {
    val source = new java.io.StringWriter()
    codegen.emitSource[A,B](snippet, "Snippet", new java.io.PrintWriter(source))
    source.toString
  }
}

abstract class DslDriverC[A: Manifest, B: Manifest] extends DslDriverBase[A, B] { self =>
  override val codegen = new DslGenC {
    val IR: self.type = self
  }

  override def eval(a: A) {
    val cppFileName = s"$dir/$fileName.cpp"
    val binaryFileName = s"$dir/$fileName"
    val out = new java.io.PrintWriter(cppFileName)
    out.println(code)
    out.close()

    new java.io.File(binaryFileName).delete
    import scala.sys.process._
    System.out.println("Compile C++ code")
    (s"g++ -std=c++11 -O1 $cppFileName -o $binaryFileName -I /opt/OpenBLAS/include -L /opt/OpenBLAS/lib -lopenblas -lpthread": ProcessBuilder).lines.foreach(System.out.println) //-std=c99
    System.out.println("Run C++ code")
    (s"$binaryFileName $a": ProcessBuilder).lines.foreach(System.out.println)
  }
}

abstract class DslDriverCuda[A: Manifest, B: Manifest] extends DslDriverBase[A, B] with DslGPUExp {
  def nvccArguments: Seq[String] = Seq("-ccbin gcc-5", "-std=c++11", "-O1", "--expt-extended-lambda", "-Wno-deprecated-gpu-targets", "-lstdc++")
}

abstract class DslDriverCublas[A: Manifest, B: Manifest] extends DslDriverCuda[A, B] { self =>
  override val codegen = new DslGenCublas {
    val IR: self.type = self
  }

  override def nvccArguments: Seq[String] = super.nvccArguments ++ Seq("-lcublas")

  override def eval(a: A) {
    val cudaFileName = s"$dir/$fileName.cu"
    val binaryFileName = s"$dir/$fileName"
    val out = new java.io.PrintWriter(cudaFileName)
    out.println(code)
    out.close()

    new java.io.File(binaryFileName).delete
    import scala.sys.process._
    System.out.println("Compile C++ (cuBLAS) code")
    (s"nvcc $cudaFileName -o $binaryFileName ${nvccArguments.mkString(" ")}": ProcessBuilder).lines.foreach(System.out.println) //-std=c99
    System.out.println("Run C++ (cuBLAS) code")
    (s"$binaryFileName $a": ProcessBuilder).lines.foreach(System.out.println)
  }
}

abstract class DslDriverCudnn[A: Manifest, B: Manifest] extends DslDriverCuda[A, B] { self =>
  override val codegen = new DslGenCudnn {
    val IR: self.type = self
  }

  override def nvccArguments: Seq[String] = super.nvccArguments ++ Seq("-lcublas", "-lcudnn")

  override def eval(a: A) {
    val cudaFileName = s"$dir/$fileName.cu"
    val binaryFileName = s"$dir/$fileName"
    val out = new java.io.PrintWriter(cudaFileName)
    out.println(code)
    out.close()

    new java.io.File(binaryFileName).delete
    import scala.sys.process._
    System.out.println("Compile C++ (cuBLAS & cuDNN) code")
    (s"nvcc $cudaFileName -o $binaryFileName ${nvccArguments.mkString(" ")}": ProcessBuilder).lines.foreach(System.out.println) //-std=c99
    System.out.println("Run C++ (cuBLAS & cuDNN) code")
    (s"$binaryFileName $a": ProcessBuilder).lines.foreach(System.out.println)
  }
}

/**
  * A wrapper around `DslDriverBase` that provides a `wrapper` function that performs backend setup/cleanup.
  * Extend this instead of `DslDriverBase` for correct backend management.
  */
trait LanternDriver[A, B] extends DslDriverBase[A, B] with TensorDsl with DslExp { self =>
  // Hacky workaround to support trait type parameters with context bounds.
  // `trait LanternDriver[A: Manifest, B: Manifest]` doesn't work.
  // These must be overridden in subclasses.
  implicit def manifestA: Manifest[A]
  implicit def manifestB: Manifest[B]

  def wrapper(x: Rep[A]): Rep[B] = {
    generateRawComment("Backend setup.")
    backend.setup()
    val result = snippet(x)

    generateRawComment("Backend cleanup.")
    backend.cleanup()
    result
  }

  override lazy val code: String = {
    val source = new java.io.StringWriter()
    codegen.emitSource(wrapper, "Snippet", new java.io.PrintWriter(source))
    source.toString
  }
}

abstract class LanternDriverC[A: Manifest, B: Manifest] extends DslDriverC[A, B] with LanternDriver[A, B] with NNModule { self =>
  override def manifestA: Manifest[A] = manifest[A]
  override def manifestB: Manifest[B] = manifest[B]
}

abstract class LanternDriverCublas[A: Manifest, B: Manifest] extends DslDriverCublas[A, B] with LanternDriver[A, B] with TensorDslCublas with NNModuleCublas { self =>
  override def manifestA: Manifest[A] = manifest[A]
  override def manifestB: Manifest[B] = manifest[B]
}

abstract class LanternDriverCudnn[A: Manifest, B: Manifest] extends DslDriverCudnn[A, B] with LanternDriver[A, B] with TensorDslCudnn with NNModuleCudnn { self =>
  override def manifestA: Manifest[A] = manifest[A]
  override def manifestB: Manifest[B] = manifest[B]
}

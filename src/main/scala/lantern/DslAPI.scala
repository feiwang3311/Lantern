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
}

trait GPUOps extends DslOps {
  object NewGPUArray {
    // Allocate an array of the specified size on the GPU.
    def apply[T: Manifest](scalarCount: Rep[Int]): Rep[Array[T]] = gpu_array_new(scalarCount)
  }
  def gpu_array_new[T: Manifest](scalarCount: Rep[Int]): Rep[Array[T]]
}

trait GPUOpsExp extends DslExp {
  case class GPUArrayNew[T: Manifest](scalarCount: Rep[Int]) extends Def[Array[T]] {
    val m = manifest[T]
  }
  def gpu_array_new[T: Manifest](scalarCount: Exp[Int]) = reflectMutable(GPUArrayNew(scalarCount))
}

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
    case afs@ArrayFromSeq(xs) => stream.println(remap(afs.m) + " " + quote(sym) + "[" + xs.length + "] = {" + (xs mkString ",") + "}; // ;)")
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

    /*
    case "Array[Char]"   => "unique_ptr<char[]>"
    case "Array[Double]" => "unique_ptr<double[]>"
    case "Array[Int]"    => "unique_ptr<int[]>"
    case "Array[Long]"    => "unique_ptr<int64_t[]>"
    case "Array[Float]"  => "unique_ptr<float[]>"
    case "Array[Array[Int]]"  => "unique_ptr<int*[]>"
    case "Array[Array[Double]]" => "unique_ptr<double*[]>"
    case "Array[Array[Float]]"  => "unique_ptr<float*[]>"
    */
    /*
    case "Array[Char]"   => "shared_ptr<char[]>"
    case "Array[Double]" => "shared_ptr<double[]>"
    case "Array[Int]"    => "shared_ptr<int[]>"
    case "Array[Long]"    => "unique_ptr<int64_t[]>"
    case "Array[Float]"  => "shared_ptr<float[]>"
    case "Array[Array[Int]]"  => "shared_ptr<int*[]>"
    case "Array[Array[Double]]" => "shared_ptr<double*[]>"
    case "Array[Array[Float]]"  => "shared_ptr<float*[]>"
    */

    case f if f.startsWith("scala.Function") =>
      val targs = m.typeArguments.dropRight(1)
      val res = remap(m.typeArguments.last)
      val targsUnboxed = targs.flatMap(t => unwrapTupleStr(remap(t)))
      val sep = if (targsUnboxed.length > 0) "," else ""
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
    case Error(s) => stream.println("assert(false && " + quote(s) + ");")
    case afs@ArrayFromSeq(xs) =>
      stream.println(remap(afs.m) + " " + quote(sym) + "[" + xs.length + "] = {" + (xs map quote mkString ",") + "}; // ;)")
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
        |int HEAP_SIZE = 1073741826; // 1048576; // 2147483652; // 536870912; // 268435456; // 2097152;
        |void *mallocBase = malloc(HEAP_SIZE);
        |void *mallocAddr = mallocBase;
        |void *waterMark = mallocBase;
        |void *myMalloc(size_t bytes) {
        |  void *res = mallocAddr;
        |  mallocAddr = (void *)((char *)mallocAddr + bytes);
        |  return res;
        |}
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
        |std::normal_distribution<> d{0, 1};
        |
        |int main(int argc, char *argv[]) {
        |  if (argc != 2) {
        |    printf("usage: query <filename>\n");
        |    return 0;
        |  }
        |  Snippet(argv[1]);
        |  return 0;
        |}
      """.stripMargin)
    }
    super.emitSource[A](args, body, functionName, out)
  }
}

trait DslGenC extends DslGenBase {
  val IR: DslExp
  import IR._
}

trait DslGenCublas extends DslGenBase {
  val IR: DslExp
  import IR._

  // Allocate GPU memory.
  def getCudaMallocString(buffer: String, count: String, dataType: String): String =
    "CUDA_CALL(cudaMalloc((void **)&" + buffer + ", " + count + " * sizeof(" + dataType + ")));"

  // Allocate GPU memory from memory arena.
  def getCudaMallocArenaString(count: String, dataType: String): String =
    "(" + dataType + "*)myGpuMalloc(" + count + " * sizeof(" + dataType + "));"

  // Allocate unified memory, accessible by CPU and GPU.
  // FIXME: I encountered "bus error" when performing CPU ops on managed memory:
  //     Thread 1 "snippet" received signal SIGBUS, Bus error.
  //     Snippet (x0=<optimized out>) at snippet.cpp:144
  //     144	float x32 = x30 - x31;
  // I wonder if others can replicate this issue.
  def getCudaMallocManagedString(buffer: String, count: String, dataType: String): String =
    "CUDA_CALL(cudaMallocManaged((void **)&" + buffer + ", " + count + " * sizeof(" + dataType + ")));"

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
      |void *gpuMallocAddr;
      |void *myGpuMalloc(size_t bytes) {
      |  void *res = gpuMallocAddr;
      |  gpuMallocAddr = (void *)((char *)gpuMallocAddr + bytes);
      |  return res;
      |}
      |
      |template <typename T>
      |__global__ void arrayUpdate(T *data, int index, T value) {
      |  data[index] = value;
      |}
      |
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
    """.stripMargin
}

trait DslGenCudnn extends DslGenCublas {
  val IR: DslExp
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
    """.stripMargin
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

abstract class DslDriverCublas[A: Manifest, B: Manifest] extends DslDriverBase[A, B] with GPUOps with GPUOpsExp { self =>
  override val codegen = new DslGenCublas {
    val IR: self.type = self

    override def emitNode(sym: Sym[Any], rhs: Def[Any]) = rhs match {
      case a@GPUArrayNew(n) =>
        val dataType = remap(a.m)
        emitValDef(sym, getCudaMallocArenaString(quote(n), dataType))
      case _ => super.emitNode(sym,rhs)
    }
  }

  override def eval(a: A) {
    val cudaFileName = s"$dir/$fileName.cu"
    val binaryFileName = s"$dir/$fileName"
    val out = new java.io.PrintWriter(cudaFileName)
    out.println(code)
    out.close()

    new java.io.File(binaryFileName).delete
    import scala.sys.process._
    System.out.println("Compile C++ (cuBLAS) code")
    (s"nvcc -std=c++11 -O1 $cudaFileName -o $binaryFileName -lcublas": ProcessBuilder).lines.foreach(System.out.println) //-std=c99
    System.out.println("Run C++ (cuBLAS) code")
    (s"$binaryFileName $a": ProcessBuilder).lines.foreach(System.out.println)
  }
}

abstract class DslDriverCudnn[A: Manifest, B: Manifest] extends DslDriverBase[A, B] with GPUOps with GPUOpsExp { self =>
  override val codegen = new DslGenCudnn {
    val IR: self.type = self

    override def emitNode(sym: Sym[Any], rhs: Def[Any]) = rhs match {
      case a@GPUArrayNew(n) =>
        val dataType = remap(a.m)
        emitValDef(sym, getCudaMallocArenaString(quote(n), dataType))
      case _ => super.emitNode(sym,rhs)
    }
  }

  override def eval(a: A) {
    val cudaFileName = s"$dir/$fileName.cu"
    val binaryFileName = s"$dir/$fileName"
    val out = new java.io.PrintWriter(cudaFileName)
    out.println(code)
    out.close()

    new java.io.File(binaryFileName).delete
    import scala.sys.process._
    System.out.println("Compile C++ (cuBLAS & cuDNN) code")
    (s"nvcc -std=c++11 -O1 $cudaFileName -o $binaryFileName -lcublas -lcudnn": ProcessBuilder).lines.foreach(System.out.println) //-std=c99
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

abstract class LanternDriverC[A: Manifest, B: Manifest] extends DslDriverC[A, B] with LanternDriver[A, B] { self =>
  override def manifestA: Manifest[A] = manifest[A]
  override def manifestB: Manifest[B] = manifest[B]
}

abstract class LanternDriverCublas[A: Manifest, B: Manifest] extends DslDriverCublas[A, B] with LanternDriver[A, B] with TensorDslCublas { self =>
  override def manifestA: Manifest[A] = manifest[A]
  override def manifestB: Manifest[B] = manifest[B]
}

abstract class LanternDriverCudnn[A: Manifest, B: Manifest] extends DslDriverCudnn[A, B] with LanternDriver[A, B] with TensorDslCudnn { self =>
  override def manifestA: Manifest[A] = manifest[A]
  override def manifestB: Manifest[B] = manifest[B]
}

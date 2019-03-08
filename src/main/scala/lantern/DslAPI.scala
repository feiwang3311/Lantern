package lantern

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms.common._

import java.nio._
import java.nio.file._
import java.io._

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
    with PrimitiveOpsExpOpt with NumericOpsExpOpt with NumericOpsExtraExp with BooleanOpsExpOpt
    with IfThenElseExpOpt with EqualExpBridgeOpt with RangeOpsExp with OrderingOpsExpOpt
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
    "<sys/mman.h>", "<sys/stat.h>", "<sys/time.h>", "<time.h>", "<unistd.h>", "<cblas.h>", "<algorithm>", "<numeric>")

  // Raw code, to be included in the code template at file scope, before the main function.
  def templateRawCode: String = ""

  def preamble = raw"""
        |using namespace std;
        |#ifndef MAP_FILE
        |#define MAP_FILE MAP_SHARED
        |#endif
        |
        |long fsize(int fd) {
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
        |  if ((long)mallocAddr >= (long)mallocBase + HEAP_SIZE_CPU) {
        |    fprintf(stderr, "CPU memory breached limit of HEAP_SIZE_CPU\n"); abort();
        |  }
        |  return res;
        |}
        |
        |long HEAP_SIZE = 8589934608; //  4294967304; // this is for GPU
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
        |""".stripMargin

  override def emitSource[A:Manifest](args: List[Sym[_]], body: Block[A], functionName: String, out: java.io.PrintWriter) = {
    withStream(out) {
      stream.println(templateHeaders.map(x => s"#include $x").mkString("\n"))
      stream.println(preamble)
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
      |  if ((long)gpuMallocAddr >= (long)gpuMallocBase + HEAP_SIZE) {
      |    fprintf(stderr, "GPU breached memory limit of HEAP_SIZE\n"); abort();
      |  }
      |  return res;
      |}
      |
      |void myGpuFree(size_t bytes) {
      |  bytes = ((bytes + (1 << N) - 1) >> N) << N;
      |  gpuMallocAddr = (void *)((char *)gpuMallocAddr - bytes);
      |  cudaMemset((void*)gpuMallocAddr, 0, bytes);
      |  return;
      |}
      |
      |template <typename T>
      |__global__ void arrayUpdate(T *data, int index, T value) {
      |  data[index] = value;
      |}
      |
      |__global__ void arrayFill(float* data, float value, int size) {
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
      | // only for 4D tensor in and 3D tensor out (TODO: incorrect!)
      |__global__ void sum_optimization(float* in, int inStr0, int inStr1, int inStr2, int inStr3,
      |                                 float* out, int outStr0, int outStr1, int outStr2,
      |                                 int dim, int nElementOut, int dimSize) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (int i = tid; i < nElementOut; i += stride) {
      |    int outOff0 = i / outStr0;
      |    int outOff1temp = i - outOff0 * outStr0;
      |    int outOff1 = outOff1temp / outStr1;
      |    int outOff2 = outOff1temp - outOff1 * outStr1;
      |    for (int j = 0; j < dimSize; j++) {
      |      int inOff; 
      |      if (dim == 0) inOff = j * inStr0 + outOff0 * inStr1 + outOff1 * inStr2 + outOff2 * inStr3;
      |      if (dim == 1) inOff = outOff0 * inStr0 + j * inStr1 + outOff1 * inStr2 + outOff2 * inStr3;
      |      if (dim == 2) inOff = outOff0 * inStr0 + outOff1 * inStr1 + j * inStr2 + outOff2 * inStr3;
      |      if (dim == 3) inOff = outOff0 * inStr0 + outOff1 * inStr1 + outOff2 * inStr2 + j * inStr3;
      |      out[i] += in[inOff];
      |    }
      |  }
      |}
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
      |template <int Dims>
      |static inline __device__ int compute(const int outputSizes[Dims], const int outputStrides[Dims],
      |                                     const int dimSize, const int concatDim, int linearIndex) {
      |  int offset = 0;
      |  #pragma unroll
      |  for (int i = Dims - 1; i >= 1; --i) {
      |    int curDimSize = i == concatDim? dimSize : outputSizes[i];
      |    int nextDimIndex = linearIndex / curDimSize;
      |    int curDimIndex = linearIndex - curDimSize * nextDimIndex;
      |    int curDimOffset = curDimIndex * outputStrides[i];
      |    offset += curDimOffset;
      |    linearIndex = nextDimIndex;
      |  }
      |  return offset + linearIndex * outputStrides[0];
      |}
      |
      |// TODO: Only for Dim of rank 4, and only for 2 inputs
      |__global__ void concat2D_1D_greg(float* in1, int dimSize1, int nElement1,
      |                                 float* in2, int dimSize2, int nElement2,
      |                                 float* out, int concatDim,
      |                                 int outSize0, int outSize1, int outSize2, int outSize3,
      |                                 int outStride0, int outStride1, int outStride2, int outStride3) {
      |  int outSizes[] = {outSize0, outSize1, outSize2, outSize3};
      |  int outStrides[] = {outStride0, outStride1, outStride2, outStride3};
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int nElement = blockIdx.y == 0 ? nElement1 : nElement2;
      |  if (tid >= nElement) return;
      |  float* data = blockIdx.y == 0 ? in1 : in2;
      |  int offset = blockIdx.y == 0 ? 0 : dimSize1;
      |  int dimSize = blockIdx.y == 0 ? dimSize1 : dimSize2;
      |  int dataOffset = offset * outStrides[concatDim];
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < nElement; tid += stride) {
      |    int elementOffset = compute<4>(outSizes, //0, outSize1, outSize2, outSize3,
      |                                   outStrides, //0, outStride1, outStride2, outStride3,
      |                                   dimSize, concatDim, tid);
      |    out[dataOffset + elementOffset] = data[tid];
      |  }
      |}
      |
      |// TODO: Only for Dim of rank 4, and only for 2 inputs, and only for concat at dim = 1
      |__global__ void concat2D_1D_greg_grad(float* in1, int dimSize1, int nElement1,
      |                                      float* in2, int dimSize2, int nElement2,
      |                                      float* out, int concatDim,
      |                                      int outSize0, int outSize1, int outSize2, int outSize3,
      |                                      int outStride0, int outStride1, int outStride2, int outStride3) {
      |  int outSizes[] = {outSize0, outSize1, outSize2, outSize3};
      |  int outStrides[] = {outStride0, outStride1, outStride2, outStride3};
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int nElement = blockIdx.y == 0 ? nElement1 : nElement2;
      |  if (tid >= nElement) return;
      |  float* data = blockIdx.y == 0 ? in1 : in2;
      |  int offset = blockIdx.y == 0 ? 0 : dimSize1;
      |  int dimSize = blockIdx.y == 0 ? dimSize1 : dimSize2;
      |  int dataOffset = offset * outStride1;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < nElement; tid += stride) {
      |    int elementOffset = compute<4>(outSizes, //0, outSize1, outSize2, outSize3,
      |                                   outStrides, //0, outStride1, outStride2, outStride3,
      |                                   dimSize, concatDim, tid);
      |    data[tid] += out[dataOffset + elementOffset];
      |  }
      |}
      |
      |__global__ void repeat0(float* in, float* out, int outStride0, int outStride1, int outScalarCount) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < outScalarCount; tid += stride) {
      |    int linearIndex = tid;
      |    int outIndex0 = linearIndex / outStride0;
      |    linearIndex = linearIndex - outIndex0 * outStride0;
      |    int outIndex1 = linearIndex / outStride1;
      |    int outIndex2 = linearIndex - outIndex1 * outStride1;
      |    int inIndex = outIndex2 + (outIndex0 + outIndex1) * outStride1;
      |    out[tid] = in[inIndex];
      |  }
      |}
      |
      |__global__ void shift0(float* in, float* out, int inDim0, int inStride0, int inStride1, int inScalarCount) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < inScalarCount; tid += stride) {
      |    int linearIndex = tid;
      |    int inIndex0 = linearIndex / inStride0;
      |    linearIndex = linearIndex - inIndex0 * inStride0;
      |    int inIndex1 = linearIndex / inStride1;
      |    if (inIndex0 + inIndex1 >= inDim0) return;
      |    out[tid + inIndex1 * inStride0] = in[tid];
      |  }
      |}
      |
      |__global__ void adagrad_update_1D_1D(float* x, float* d, float* m, float clip, float lr, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride) {
      |    if (d[tid] > clip) d[tid] = clip;
      |    if (d[tid] < -clip) d[tid] = -clip;
      |    m[tid] += d[tid] * d[tid];
      |    x[tid] -= lr * d[tid] / sqrt(m[tid] + 0.00000001);
      |    d[tid] = 0;
      |  }
      |}
      |
      |__global__ void momentum_update_1D_1D(float* x, float* d, float* m, float learning_rate, float momentum, float gradClip, bool nesterov, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride) {
      |    float temp = d[tid];
      |    if (temp > gradClip) temp = gradClip;
      |    if (temp < -gradClip) temp = -gradClip;
      |    m[tid] *= momentum;
      |    m[tid] += temp;
      |    if (nesterov) { temp += momentum * m[tid]; }
      |    else { temp = m[tid]; }
      |    x[tid] -= learning_rate * temp;
      |    d[tid] = 0;
      |  }
      |}
      |
      |__global__ void addScalarInArrayInPlace(float* in, float* add, float scale, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride)
      |    if (tid < size) in[tid] += add[0] * scale;
      |}
      |
      |__global__ void addScalar(float* in, float* out, float add, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride)
      |    if (tid < size) out[tid] = in[tid] + add;
      |}
      |__global__ void minusScalar(float* in, float* out, float minus, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride)
      |    if (tid < size) out[tid] = in[tid] - minus;
      |}
      |__global__ void multScalar(float* in, float* out, float mult, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride)
      |    if (tid < size) out[tid] = in[tid] * mult;
      |}
      |__global__ void divScalar(float* in, float* out, float div, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride)
      |    if (tid < size) out[tid] = in[tid] / div;
      |}
      |
      |__global__ void elementwise_1D_1D_mul(float* in1, float* in2, float* out, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride)
      |    if (tid < size) out[tid] = in1[tid] * in2[tid];
      |}
      |
      |__global__ void elementwise_1D_1D_mul_mutate(float* in1, float* in2, float* out, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride)
      |    if (tid < size) out[tid] += in1[tid] * in2[tid];
      |}
      |
      |__global__ void elementwise_1D_1D_add(float* in1, float* in2, float* out, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride)
      |    if (tid < size) out[tid] = in1[tid] + in2[tid];
      |}
      |
      |__global__ void elementwise_1D_1D_minus(float* in1, float* in2, float* out, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride)
      |    if (tid < size) out[tid] = in1[tid] - in2[tid];
      |}
      |
      |__global__ void elementwise_1D_1D_div(float* in1, float* in2, float* out, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride)
      |    if (tid < size) out[tid] = in1[tid] / in2[tid];
      |}
      |
      |__global__ void elementwise_1D_1D_exp(float* in, float* out, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride)
      |    if (tid < size) out[tid] = exp(in[tid]);
      |}
      |__global__ void elementwise_1D_1D_log(float* in, float* out, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride)
      |    if (tid < size) out[tid] = log(in[tid]);
      |}
      |__global__ void elementwise_1D_1D_sqrt(float* in, float* out, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride)
      |    if (tid < size) out[tid] = sqrt(in[tid]);
      |}
      |
      |__global__ void elementwise_1D_1D_square(float* in, float* out, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride)
      |    if (tid < size) out[tid] = in[tid] * in[tid];
      |}
      |
      |__global__ void elementwise_1D_1D_exp_grad(float* in_x, float* in_d, float* out_x, float * out_d, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride)
      |    if (tid < size) in_d[tid] += out_d[tid] * out_x[tid];
      |}
      |
      |__global__ void elementwise_1D_1D_log_grad(float* in_x, float* in_d, float* out_x, float * out_d, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride)
      |    if (tid < size) in_d[tid] += out_d[tid] / in_x[tid];
      |}
      |
      |__global__ void elementwise_1D_1D_sqrt_grad(float* in_x, float* in_d, float* out_x, float * out_d, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride)
      |    if (tid < size) in_d[tid] += out_d[tid] / out_x[tid] / 2;
      |}
      |
      |__global__ void elementwise_1D_1D_square_grad(float* in_x, float* in_d, float* out_x, float * out_d, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride)
      |    if (tid < size) in_d[tid] += out_d[tid] * 2 * in_x[tid];
      |}
      |
      |__global__ void clipAt(float* in, float bound, int size) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < size; tid += stride)
      |    if (tid < size) {
      |      if (in[tid] > bound) in[tid] = bound;
      |      if (in[tid] < -bound) in[tid] = -bound;
      |    }
      |}
      |
      |__global__ void mask4D(float* in, int* mask, int xstrides0, int xstrides1, int xstrides2, int xstrides3, int scalarCount) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < scalarCount; tid += stride) {
      |    int linearIndex = tid;
      |    int xindex0 = linearIndex / xstrides0;
      |    linearIndex = linearIndex - xstrides0 * xindex0;
      |    int xindex1 = linearIndex / xstrides1;
      |    linearIndex = linearIndex - xstrides1 * xindex1;
      |    int xindex2 = linearIndex / xstrides2;
      |    int xindex3 = linearIndex - xstrides2 * xindex2;
      |    if (xindex3 >= mask[xindex0]) in[tid] = 0;
      |  }
      |}
      |
      |__global__ void mul_sub(float* in1, float* in2, float* out, int in1ScalarCount, int in2ScalarCount) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < in1ScalarCount; tid += stride) {
      |    out[tid] = in1[tid] * in2[tid % in2ScalarCount];
      |  }
      |}
      |
      |__global__ void mul_sub_grad(float* in1_x, float* in1_d, float* in2_x, float* in2_d, float* out, int in1ScalarCount, int in2ScalarCount) {
      |  int tid = blockIdx.x * blockDim.x + threadIdx.x;
      |  int stride = gridDim.x * blockDim.x;
      |  for (; tid < in1ScalarCount; tid += stride) {
      |    int index = tid % in2ScalarCount;
      |    in1_d[tid] += out[tid] * in2_x[index];
      |    in2_d[tid] = in1_x[tid] * out[tid];  // this is the temp array, need to be reduced!
      |  }
      |}
      |
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

abstract class LanternDriverC[A: Manifest, B: Manifest] extends DslDriverC[A, B] with LanternDriver[A, B] with TensorDslCPU with NNModule with Dataset with ONNXLib { self =>
  override def manifestA: Manifest[A] = manifest[A]
  override def manifestB: Manifest[B] = manifest[B]
}

abstract class LanternDriverCublas[A: Manifest, B: Manifest] extends DslDriverCublas[A, B] with LanternDriver[A, B] with TensorDslCublas with NNModuleCublas with Dataset { self =>
  override def manifestA: Manifest[A] = manifest[A]
  override def manifestB: Manifest[B] = manifest[B]
  override val codegen = new DslGenCudnn {
    val IR: self.type = self

    override def templateRawCode: String = {
      super.templateRawCode +
      (concatMap.values mkString("\n\n")) + (mask4dKernelMap.values map(_._1) mkString("\n\n")) +
      (permuteKernelMap.values map(_._1) mkString("\n\n")) + (permuteGradKernelMap.values map(_._1) mkString("\n\n")) +
      (mulSubKernelMap.values map(_._1) mkString("\n\n")) + (mulSubGradKernelMap.values map(_._1) mkString("\n\n"))
    }
  }
}

abstract class LanternDriverCudnn[A: Manifest, B: Manifest] extends DslDriverCudnn[A, B] with LanternDriver[A, B] with TensorDslCudnn with NNModuleCudnn  with Dataset { self =>
  override def manifestA: Manifest[A] = manifest[A]
  override def manifestB: Manifest[B] = manifest[B]

  override val codegen = new DslGenCudnn {
    val IR: self.type = self

    override def templateRawCode: String = {
      super.templateRawCode +
      (concatMap.values mkString("\n\n")) + (mask4dKernelMap.values map(_._1) mkString("\n\n")) +
      (permuteKernelMap.values map(_._1) mkString("\n\n")) + (permuteGradKernelMap.values map(_._1) mkString("\n\n")) +
      (mulSubKernelMap.values map(_._1) mkString("\n\n")) + (mulSubGradKernelMap.values map(_._1) mkString("\n\n")) +
      (elementWiseWithBroadCastKernelMap.values map(_._1) mkString("\n\n"))
    }
  }
}


// library code generation
trait DslGenBaseLib extends DslGenBase {
  val IR: DslExp
  import IR._

  override def preamble = raw"""
        |using namespace std;
        |#ifndef MAP_FILE
        |#define MAP_FILE MAP_SHARED
        |#endif
        |
        |long fsize(int fd) {
        |  struct stat stat;
        |  int res = fstat(fd, &stat);
        |  return stat.st_size;
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
        |long HEAP_SIZE = 8589934608; //  4294967304; // this is for GPU
        |
        |$templateRawCode
        |
        |""".stripMargin
}

trait DslGenLibC extends DslGenBaseLib {
  val IR: DslExp
  import IR._
}

abstract class DslDriverBaseLib[A: Manifest, B: Manifest, C:Manifest] extends DslExp { self =>
  val codegen: DslGenBaseLib {
    val IR: self.type
  }

  def libDir: String
  def libFileName: String
  def libInferenceFuncName: String

  // The code snippet to compile.
  def snippet(x: Rep[A], y: Rep[B]): Rep[C]

  def generateLib: Unit

  lazy val code: String = {
    val source = new java.io.StringWriter()
    codegen.emitSource2[A,B,C](snippet, libInferenceFuncName, new java.io.PrintWriter(source))
    source.toString
  }
}

abstract class DslDriverLibC[A: Manifest, B: Manifest, C:Manifest] extends DslDriverBaseLib[A, B, C] { self =>
  override val codegen = new DslGenLibC {
    val IR: self.type = self
  }
}

trait LanternDriverLib[A, B, C] extends DslDriverBaseLib[A, B, C] with TensorDsl with DslExp { self =>
  implicit def manifestA: Manifest[A]
  implicit def manifestB: Manifest[B]
  implicit def manifestC: Manifest[C]

  def wrapper(x: Rep[A], y: Rep[B]): Rep[C] = {
    generateRawComment("Backend setup.")
    backend.setup()
    val result = snippet(x, y)

    generateRawComment("Backend cleanup.")
    backend.cleanup()
    result
  }

  override lazy val code: String = {
    val source = new java.io.StringWriter()
    codegen.emitSource2(wrapper, libInferenceFuncName, new java.io.PrintWriter(source))
    source.toString
  }

  override def generateLib {
    val dir = new File(libDir)
    if (!dir.exists()) dir.mkdirs()
    val cppFileName = s"$libDir/$libFileName.cpp"
    val headerFileName = s"$libDir/$libFileName.h"
    val binaryFileName = s"$libDir/$libFileName.so"
    val out = new java.io.PrintWriter(cppFileName)
    out.println(code)
    out.close()
    val out2 = new java.io.PrintWriter(headerFileName)
    out2.println(s"void $libInferenceFuncName(float* in, float* out);")
    out2.println(s"void *myMalloc(size_t bytes);")
    out2.println(s"long fsize(int fd);")
    out2.close()

    // new java.io.File(binaryFileName).delete
    // import scala.sys.process._
    // System.out.println(s"Compile C++ code $cppFileName into $binaryFileName as dynamic library")
    // (s"g++ -c -std=c++11 -O3 $cppFileName -o $binaryFileName -fPIC -I /opt/OpenBLAS/include -L /opt/OpenBLAS/lib -lopenblas -lpthread": ProcessBuilder).lines.foreach(System.out.println) //-std=c99
  }
}

abstract class LanternDriverLibC[A: Manifest, B: Manifest, C: Manifest] extends DslDriverLibC[A, B, C] with LanternDriverLib[A, B, C] with TensorDslCPU with NNModule with Dataset with ONNXLib { self =>
  override def manifestA: Manifest[A] = manifest[A]
  override def manifestB: Manifest[B] = manifest[B]
  override def manifestC: Manifest[C] = manifest[C]
}

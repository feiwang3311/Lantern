package lantern

import scala.util.continuations._

import lms.core.stub._
import lms.macros.SourceContext
import lms.core.virtualize
import lms.core.Backend
import lms.core.Backend._
import lms.core.Graph

import java.io.File
import java.io.PrintWriter

import lms.thirdparty._
import lantern.thirdparty._
import lantern.collection.mutable._

trait LanternGenC extends DslGenCPP {
  val IR: DslExp
  import IR._

  class Unknown

  def isInt(d: Backend.Def): Boolean = d match {
    case Backend.Const(x: Int) => true
    case s : Backend.Sym => typeMap.get(s).fold(false)(_ == manifest[Int])
    case _ => false
  }
  override def shallow(n: Node): Unit = n match {
    case n @ Node(s, "NewArray", List(x), _) if isInt(x) =>
      val ctype = remap(typeMap.get(s).map(_.typeArguments.head).getOrElse(manifest[Unknown]))
      emit(s"($ctype*)myMalloc("); shallow(x); emit(s" * sizeof($ctype))")
      // emit(s"($ctype*)myMalloc(${shallow(x)} * sizeof($ctype))")
    case _ => super.shallow(n)
  }

  def templateHeaders: Seq[String] = Seq(
    "<assert.h>", "<err.h>", "<errno.h>", "<fcntl.h>", "<functional>",
    "<math.h>", "<memory>", "<random>", "<stdint.h>", "<stdio.h>", "<stdlib.h>", "<string.h>", "<stdbool.h>",
    "<sys/mman.h>", "<sys/stat.h>", "<sys/time.h>", "<time.h>", "<unistd.h>", "<cblas.h>", "<algorithm>", "<numeric>")
  def templateRawCode: String = ""

  override def emitAll(ng: Graph, name: String)(m1:Manifest[_],m2:Manifest[_]): Unit = {
    val g = init(ng)
    val arg = quote(g.block.in.head)
    val efs = "" //quoteEff(g.block.ein)
    val stt = dce.statics.toList.map(quoteStatic).mkString(", ")
    val (ms1, ms2) = (remap(m1), remap(m2))
    val functionName = name
    stream.println(raw"""
    |${templateHeaders.map(x => s"#include $x").mkString("\n")}
    |
    |using namespace std;
    |#ifndef MAP_FILE
    |#define MAP_FILE MAP_SHARED
    |#endif
    |
    |long fsize(int fd) {
    |  struct stat stat;
    |  int res = fstat(fd,&stat);
    |  return stat.st_size;
    |}
    |int printll(char* s) {
    |  while (*s != '\n' && *s != ',' && *s != '\t') {
    |    putchar(*s++);
    |  }
    |  return 0;
    |}
    |long hash(char *str0, int len) {
    |  unsigned char* str = (unsigned char*)str0;
    |  unsigned long hash = 5381;
    |  int c;
    |
    |  while ((c = *str++) && len--)
    |    hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    |
    |  return hash;
    |}
    |
    |long HEAP_SIZE_CPU = 1073741826;
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
    |int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1) {
    |  long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
    |  result->tv_sec = diff / 1000000;
    |  result->tv_usec = diff % 1000000;
    |  return (diff < 0);
    |}
    |
    |$templateRawCode
    |
    |void Snippet(char*);
    |
    |int main(int argc, char *argv[]) {
    |  if (argc != 2) {
    |    printf("usage: query <filename>\n");
    |    return 0;
    |  }
    |  Snippet(argv[1]);
    |  return 0;
    |}
    |/*****************************************
    |Emitting C Generated Code
    |*******************************************/
   """.stripMargin)
    val src = run(name, g)
    src.writeTo(stream)
    stream.println("""
    /*****************************************
    End of C Generated Code
    *******************************************/
    """)
  }
}

trait LanternGenCublas extends LanternGenC {
  val IR: DslExp
  import IR._

  override def shallow(n: Node): Unit = n match {
    case n @ Node(s, "NewGpuArray", List(x), _) =>
      val ctype = remap(typeMap.get(s).map(_.typeArguments.head).getOrElse(manifest[Unknown]))
      emit(s"($ctype*)myGpuMalloc("); shallow(x); emit(s" * sizeof($ctype))")
      // s"($ctype*)myGpuMalloc(${shallow(x)} * sizeof($ctype))"
    case n @ Node(s, op, List(x,y,size),_) if op.startsWith("h2dCopy[") =>
      // TODO: this is a bad fix for shallow(x), which doesn't work if x is static float array like {1.0, 2.0}
      // The issue is in c++ -std=c++11, we can use `(const float[]){1.0, 2.0}` to initalize a static float array namelessly
      // But we cannot do so in cuda code, which is compiled via nvcc!
      // I found no way to initialize a nameless float array in cuda, thus we cannot use `shallow(x)`!
      // We must to `traverse(x)' first, and use the result of it in the cudaMemcpy
      val ty = op.substring(8, op.length - 1).toLowerCase
      x match {
        case nx @ InlineSym(nnx @ Node(sx, "Array" , xs, _)) =>
          traverse(nnx)
          emit(s"CUDA_CALL(cudaMemcpy("); shallow(y); emit(s", ${quote(sx)}, "); shallow(size); emit(s" * sizeof($ty), cudaMemcpyHostToDevice))")
        case _ =>
          emit(s"CUDA_CALL(cudaMemcpy("); shallow(y); emit(", "); shallow(x); emit(", "); shallow(size); emit(s" * sizeof($ty), cudaMemcpyHostToDevice))")
      }
      // s"CUDA_CALL(cudaMemcpy(${shallow(y)}, ${shallow(x)}, ${shallow(size)}*sizeof(${ty}), cudaMemcpyHostToDevice))"
    case n @ Node(s, op, List(x,y,size),_) if op.startsWith("d2hCopy[") =>
      val ty = op.substring(8, op.length - 1).toLowerCase // op.drop(8).dropRight(1).toLowerCase
      emit(s"CUDA_CALL(cudaMemcpy("); shallow(y); emit(", "); shallow(x); emit(", "); shallow(size); emit(s" * sizeof($ty), cudaMemcpyDeviceToHost))")
      // s"CUDA_CALL(cudaMemcpy(${shallow(y)}, ${shallow(x)}, ${shallow(size)}*sizeof(${ty}), cudaMemcpyDeviceToHost))"
    case n @ Node(s, op, List(x,y,size),_) if op.startsWith("d2dCopy[") =>
      val ty = op.substring(8, op.length - 1).toLowerCase // op.drop(8).dropRight(1).toLowerCase
      emit(s"CUDA_CALL(cudaMemcpy("); shallow(y); emit(", "); shallow(x); emit(", "); shallow(size); emit(s" * sizeof($ty), cudaMemcpyDeviceToDevice))")
      // s"CUDA_CALL(cudaMemcpy(${shallow(y)}, ${shallow(x)}, ${shallow(size)}*sizeof(${ty}), cudaMemcpyDeviceToDevice))"
    case _ => super.shallow(n)
  }

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
      |  if ((long)gpuMallocAddr > (long)gpuMallocBase + HEAP_SIZE) {
      |    fprintf(stderr, "GPU breached memory limit of HEAP_SIZE\n");
      |    // try to throw a SegFault here so that I can use gdb to find where the error is:
      |    int *foo = (int*)-1;
      |    printf("%d\n", *foo);
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
      |#define AVAIL_GPU_MEM ((long)gpuMallocBase + HEAP_SIZE - (long)gpuMallocAddr)
      |#define CAP_AVAIL(claim) min(AVAIL_GPU_MEM, claim)
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

trait LanternGenCudnn extends LanternGenCublas {
  val IR: DslExp
  import IR._

  def convAlgoTemplate(x: Int): String =
  s"""
    |cudnnConvolutionFwdAlgo_t       algo_$x      = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;     bool init_algo_$x      = false;
    |cudnnConvolutionBwdDataAlgo_t   algo_bwd_$x  = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;   bool init_algo_bwd_$x  = false;
    |cudnnConvolutionBwdFilterAlgo_t algo_bwf_$x  = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0; bool init_algo_bwf_$x  = false;
   """.stripMargin

  def convOpIndex() = ((0 until 12): Range).toList
  def buildConvAlgoTemplate(): String = convOpIndex.map(convAlgoTemplate).mkString("\n")

  override def templateHeaders: Seq[String] = super.templateHeaders ++ Seq("<cudnn.h>")
  override def templateRawCode: String = super.templateRawCode + buildConvAlgoTemplate() +
     """
      |
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

// TODO: bad design!! NNModule should not depend on backend!
abstract class LanternDriverBase[A: Manifest, B: Manifest] extends DslDriverCPP[A, B]
  with TensorDsl with NNModule with Dataset with ONNXLib with ScannerOpsExp with TimerOpsExp { q =>
  override val codegen = new LanternGenC {
    val IR: q.type = q
  }

  val dir = "/tmp/"
  val fileName = s"lantern-snippet-${scala.util.Random.alphanumeric.take(4).mkString}"
  val filetype = ".cpp"

  def codeToFile(name: Option[String] = None) = {
    val outFileName = name match {
      case Some(s) => s
      case None => dir + fileName + filetype
    }
    System.out.println(s"code => $outFileName")
    val outFile = new PrintWriter(new File(outFileName))
    outFile.println(this.code)
    outFile.flush()
  }

  override def wrapper(x: Rep[A]): Rep[B] = {
    generate_comment("Backend setup.")
    backend.setup()
    val result = snippet(x)

    generate_comment("Backend cleanup.")
    backend.cleanup()
    result
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
  override val codegen = new LanternGenCublas {
    val IR: q.type = q
    override def templateRawCode: String = super.templateRawCode + (permutationKernelMap.values map (_._1) mkString("\n\n"))
  }

  backend = BackendCublas()

  override val filetype = ".cu"

  override lazy val f: A => Unit = {
    // TBD: should read result of type B?
    val out = new java.io.PrintWriter("/tmp/snippet.cu")
    out.println(code)
    out.close
    (new java.io.File("/tmp/snippet")).delete
    import scala.sys.process._
    // TODO: would like to use time("cc") { .. }, but messes with captureOut
    (s"nvcc -std=c++11 -O3 /tmp/snippet.cu -o /tmp/snippet --expt-extended-lambda -Wno-deprecated-gpu-targets -I /opt/OpenBLAS/include -L /opt/OpenBLAS/lib -lopenblas -lstdc++ -lcublas": ProcessBuilder).lines.foreach(Console.println _)
    (a: A) => (s"/tmp/snippet $a": ProcessBuilder).lines.foreach(Console.println _)
  }

}

abstract class LanternDriverCudnn[A: Manifest, B: Manifest] extends LanternDriverCublas[A, B] with NNModuleCudnn with TensorDslCudnn { q =>
  override val codegen = new LanternGenCudnn with CCodeGenCuBLAS with CCodeGenCuDNN with CCodeGenStackArray with CCodeGenCMacro with CCodeGenLibStruct with CCodeGenLibFunction {
    val IR: q.type = q

    override def convOpIndex() = convOpIndexSet.toList
    override def templateRawCode: String = super.templateRawCode +
      (permutationKernelMap.values map (_._1) mkString("\n\n")) +
      (elementWiseWithBroadCastKernelMap.values map(_._1) mkString("\n\n")) +
      (elementWiseUpdateWithBroadCastKernelMap.values map(_._1) mkString("\n\n"))
  }
  backend = BackendCudnn()
  override lazy val f: A => Unit = {
    // TBD: should read result of type B?
    val out = new java.io.PrintWriter("/tmp/snippet.cu")
    out.println(code)
    out.close
    (new java.io.File("/tmp/snippet")).delete
    import scala.sys.process._
    // TODO: would like to use time("cc") { .. }, but messes with captureOut
    (s"nvcc -std=c++11 -O3 /tmp/snippet.cu -o /tmp/snippet --expt-extended-lambda -Wno-deprecated-gpu-targets -I /opt/OpenBLAS/include -L /opt/OpenBLAS/lib -lopenblas -lstdc++ -lcublas -lcudnn": ProcessBuilder).lines.foreach(Console.println _)
    (a: A) => (s"/tmp/snippet $a": ProcessBuilder).lines.foreach(Console.println _)
  }
}

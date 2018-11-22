#include <assert.h>
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <functional>
#include <math.h>
#include <memory>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <cblas.h>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

using namespace std;
#ifndef MAP_FILE
#define MAP_FILE MAP_SHARED
#endif

long fsize(int fd) {
  struct stat stat;
  int res = fstat(fd, &stat);
  return stat.st_size;
}

int printll(char *s) {
  while (*s != '\n' && *s != ',' && *s != '\t') {
    putchar(*s++);
  }
  return 0;
}

long hash(char *str0, int len) {
  unsigned char *str = (unsigned char *)str0;
  unsigned long hash = 5381;
  int c;

  while ((c = *str++) && len--)
    hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

  return hash;
}

long HEAP_SIZE_CPU = 1073741826; // 1048576; // 536870912; // 268435456; // 2097152; 1610612739; // 4294967304; //
void *mallocBase = calloc(HEAP_SIZE_CPU, 1);
void *mallocAddr = mallocBase;
void *waterMark = mallocBase;
void *myMalloc(size_t bytes) {
  void *res = mallocAddr;
  mallocAddr = (void *)((char *)mallocAddr + bytes);
  if ((long)mallocAddr >= (long)mallocBase + HEAP_SIZE_CPU)
    fprintf(stderr, "CPU memory breached limit of HEAP_SIZE_CPU\n");
  return res;
}

long HEAP_SIZE = 8589934608; //  4294967304; // this is for GPU

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1) {
  long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
  result->tv_sec = diff / 1000000;
  result->tv_usec = diff % 1000000;
  return (diff < 0);
}


#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error occurred: %s (%s:%d)\n", \
            cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(err); \
  } \
}

#define CUBLAS_CALL(f) { \
  cublasStatus_t stat = (f); \
  if (stat != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error occurred: %d (%s:%d)\n", \
            stat, __FILE__, __LINE__); \
    exit(stat); \
  } \
}

void *gpuMallocBase;
void *gpuMallocAddr;

// Alignment boundary size, in bytes.
constexpr int N = 4; // 16
void *myGpuMalloc(size_t bytes) {
  bytes = ((bytes + (1 << N) - 1) >> N) << N;
  void *res = gpuMallocAddr;
  gpuMallocAddr = (void *)((char *)gpuMallocAddr + bytes);
  if ((long)gpuMallocAddr >= (long)gpuMallocBase + HEAP_SIZE)
    fprintf(stderr, "GPU breached memory limit of HEAP_SIZE\n");
  return res;
}

template <typename T>
__global__ void arrayUpdate(T *data, int index, T value) {
  data[index] = value;
}

__global__ void arrayFill(float* data, float value, int size) {
  int stride = gridDim.x * blockDim.x;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < size; i += stride) data[i] = value;
}

__global__ void hardTanh(float* in, float* out, float min_val, float max_val, int size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < size; i += stride) {
    out[i] = in[i] < min_val ? min_val : (in[i] > max_val ? max_val : in[i]);
  }
}

__global__ void hardTanh_grad(float* in_x, float* in_d, float* out_d, float min_val, float max_val, int size, bool inplace) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < size; i += stride) {
    if (inplace) {
      if (in_x[i] < min_val || in_x[i] > max_val) in_d[i] = 0;
    } else {
      if (in_x[i] >= min_val && in_x[i] <= max_val) in_d[i] += out_d[i];
    }
  }
}

__global__ void nllLoss(float *x, int x_stride, float *y, int* target) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int offset = tid * x_stride + target[tid];
  y[tid] = -1 * x[offset];
}

__global__ void nllLoss_grad(int x_stride, float *yGrad, int* target, float* xGrad) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int offset = tid * x_stride + target[tid];
  xGrad[offset] += -1 * yGrad[tid];
}

 // only for 4D tensor in and 3D tensor out
__global__ void sum_grad(float* in, int inSize0, int inSize1, int inSize2, int inSize3, int nElement,
                         float* out, int outStride0, int outStride1, int outStride2, int dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < nElement; i += stride) {
    int inOff2 = i / inSize3;
    int inDim3 = i - inOff2 * inSize3;
    int inOff1 = inOff2 / inSize2;
    int inDim2 = inOff2 - inOff1 * inSize2;
    int inDim0 = inOff1 / inSize1;
    int inDim1 = inOff1 - inDim0 * inSize1;
    int outOff = 0;
    if (dim == 0) outOff = inDim1 * outStride0 + inDim2 * outStride1 + inDim3 * outStride2;
    if (dim == 1) outOff = inDim0 * outStride0 + inDim2 * outStride1 + inDim3 * outStride2;
    if (dim == 2) outOff = inDim0 * outStride0 + inDim1 * outStride1 + inDim3 * outStride2;
    if (dim == 3) outOff = inDim0 * outStride0 + inDim1 * outStride1 + inDim2 * outStride2;
    in[i] += out[outOff];
  }
}

//following - https://github.com/torch/cutorch/blob/master/lib/THC/THCTensorMath.cuh#L49
static inline __device__ int compute(int outputSize0, int outputSize1, int outputSize2, int outputSize3,
                                     int outputStride0, int outputStride1, int outputStride2, int outputStride3,
                                     const int dimSize, const int concatDim, int linearIndex) {
  int offset = 0;
  int curDimSize = 3 == concatDim ? dimSize : outputSize3;
  int nextDimIndex = linearIndex / curDimSize;
  int curDimIndex = linearIndex - curDimSize * nextDimIndex;
  int curDimOffset = curDimIndex * outputStride3;
  offset += curDimOffset;
  linearIndex = nextDimIndex;
  curDimSize = 2 == concatDim ? dimSize : outputSize2;
  nextDimIndex = linearIndex / curDimSize;
  curDimIndex = linearIndex - curDimSize * nextDimIndex;
  curDimOffset = curDimIndex * outputStride2;
  offset += curDimOffset;
  linearIndex = nextDimIndex;
  curDimSize = 1 == concatDim ? dimSize : outputSize1;
  nextDimIndex = linearIndex / curDimSize;
  curDimIndex = linearIndex - curDimSize * nextDimIndex;
  curDimOffset = curDimIndex * outputStride1;
  offset += curDimOffset;
  linearIndex = nextDimIndex;
  return offset + linearIndex * outputStride0;
//  for (int i = 3; i >= 1; i--) {
//    int curDimSize = i == concatDim ? dimSize : outputSize[i];
//    int nextDimIndex = linearIndex / curDimSize;
//    int curDimIndex = linearIndex - curDimSize * nextDimIndex;
//    int curDimOffset = curDimIndex * outputStride[i];
//    offset += curDimOffset;
//    linearIndex = nextDimIndex;
//  }
//  return offset + linearIndex * outputStride[0];
}

// TODO: Only for Dim of rank 4, and only for 2 inputs, and only for concat at dim = 1
__global__ void concat2D_1D_greg(float* in1, int dimSize1, int nElement1,
                                 float* in2, int dimSize2, int nElement2,
                                 float* out, int concatDim,
                                 int outSize0, int outSize1, int outSize2, int outSize3,
                                 int outStride0, int outStride1, int outStride2, int outStride3) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nElement = blockIdx.y == 0 ? nElement1 : nElement2;
  if (tid >= nElement) return;
  float* data = blockIdx.y == 0 ? in1 : in2;
  int offset = blockIdx.y == 0 ? 0 : dimSize1;
  int dimSize = blockIdx.y == 0 ? dimSize1 : dimSize2;
  int dataOffset = offset * outStride1;
  int stride = gridDim.x * blockDim.x;
  while (tid < nElement) {
    int elementOffset = compute(outSize0, outSize1, outSize2, outSize3,
                                outStride0, outStride1, outStride2, outStride3, dimSize, concatDim, tid);
    out[dataOffset + elementOffset] = data[tid];
    tid += stride;
  }
}

// TODO: Only for Dim of rank 4, and only for 2 inputs, and only for concat at dim = 1
__global__ void concat2D_1D_greg_grad(float* in1, int dimSize1, int nElement1,
                                      float* in2, int dimSize2, int nElement2,
                                      float* out, int concatDim,
                                      int outSize0, int outSize1, int outSize2, int outSize3,
                                      int outStride0, int outStride1, int outStride2, int outStride3) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nElement = blockIdx.y == 0 ? nElement1 : nElement2;
  if (tid >= nElement) return;
  float* data = blockIdx.y == 0 ? in1 : in2;
  int offset = blockIdx.y == 0 ? 0 : dimSize1;
  int dimSize = blockIdx.y == 0 ? dimSize1 : dimSize2;
  int dataOffset = offset * outStride1;
  int stride = gridDim.x * blockDim.x;
  while (tid < nElement) {
    int elementOffset = compute(outSize0, outSize1, outSize2, outSize3,
                                outStride0, outStride1, outStride2, outStride3, dimSize, concatDim, tid);
    data[tid] += out[dataOffset + elementOffset];
    tid += stride;
  }
}

__global__ void repeat0(float* in, float* out, int outStride0, int outStride1, int outScalarCount) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < outScalarCount; tid += stride) {
    int linearIndex = tid;
    int outIndex0 = linearIndex / outStride0;
    linearIndex = linearIndex - outIndex0 * outStride0;
    int outIndex1 = linearIndex / outStride1;
    int outIndex2 = linearIndex - outIndex1 * outStride1;
    int inIndex = outIndex2 + (outIndex0 + outIndex1) * outStride1;
    out[tid] = in[inIndex];
  }
}

__global__ void shift0(float* in, float* out, int inDim0, int inStride0, int inStride1, int inScalarCount) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < inScalarCount; tid += stride) {
    int linearIndex = tid;
    int inIndex0 = linearIndex / inStride0;
    linearIndex = linearIndex - inIndex0 * inStride0;
    int inIndex1 = linearIndex / inStride1;
    if (inIndex0 + inIndex1 >= inDim0) return;
    out[tid + inIndex1 * inStride0] = in[tid];
  }
}

__global__ void adagrad_update_1D_1D(float* x, float* d, float* m, float clip, float lr, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride) {
    if (d[tid] > clip) d[tid] = clip;
    if (d[tid] < -clip) d[tid] = -clip;
    m[tid] += d[tid] * d[tid];
    x[tid] -= lr * d[tid] / sqrt(m[tid] + 0.00000001);
    d[tid] = 0;
  }
}

__global__ void momentum_update_1D_1D(float* x, float* d, float* m, float learning_rate, float momentum, float gradClip, bool nesterov, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride) {
    float temp = d[tid];
    if (temp > gradClip) temp = gradClip;
    if (temp < -gradClip) temp = -gradClip;
    m[tid] *= momentum;
    m[tid] += temp;
    if (nesterov) { temp += momentum * m[tid]; }
    else { temp = m[tid]; }
    x[tid] -= learning_rate * temp;
    d[tid] = 0;
  }
}

__global__ void addScalar(float* in, float* out, float add, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) out[tid] = in[tid] + add;
}
__global__ void minusScalar(float* in, float* out, float minus, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) out[tid] = in[tid] - minus;
}
__global__ void multScalar(float* in, float* out, float mult, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) out[tid] = in[tid] * mult;
}
__global__ void divScalar(float* in, float* out, float div, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) out[tid] = in[tid] / div;
}

__global__ void elementwise_1D_1D_mul(float* in1, float* in2, float* out, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) out[tid] = in1[tid] * in2[tid];
}

__global__ void elementwise_1D_1D_mul_mutate(float* in1, float* in2, float* out, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) out[tid] += in1[tid] * in2[tid];
}

__global__ void elementwise_1D_1D_add(float* in1, float* in2, float* out, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) out[tid] = in1[tid] + in2[tid];
}

__global__ void elementwise_1D_1D_minus(float* in1, float* in2, float* out, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) out[tid] = in1[tid] - in2[tid];
}

__global__ void elementwise_1D_1D_div(float* in1, float* in2, float* out, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) out[tid] = in1[tid] / in2[tid];
}

__global__ void elementwise_1D_1D_exp(float* in, float* out, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) out[tid] = exp(in[tid]);
}
__global__ void elementwise_1D_1D_log(float* in, float* out, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) out[tid] = log(in[tid]);
}
__global__ void elementwise_1D_1D_sqrt(float* in, float* out, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) out[tid] = sqrt(in[tid]);
}

__global__ void elementwise_1D_1D_square(float* in, float* out, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) out[tid] = in[tid] * in[tid];
}

__global__ void elementwise_1D_1D_exp_grad(float* in_x, float* in_d, float* out_x, float * out_d, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) in_d[tid] += out_d[tid] * out_x[tid];
}

__global__ void elementwise_1D_1D_log_grad(float* in_x, float* in_d, float* out_x, float * out_d, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) in_d[tid] += out_d[tid] / in_x[tid];
}

__global__ void elementwise_1D_1D_sqrt_grad(float* in_x, float* in_d, float* out_x, float * out_d, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) in_d[tid] += out_d[tid] / out_x[tid] / 2;
}

__global__ void elementwise_1D_1D_square_grad(float* in_x, float* in_d, float* out_x, float * out_d, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) in_d[tid] += out_d[tid] * 2 * in_x[tid];
}

__global__ void clipAt(float* in, float bound, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) {
      if (in[tid] > bound) in[tid] = bound;
      if (in[tid] < -bound) in[tid] = -bound;
    }
}

__global__ void mask4D(float* in, int* mask, int xstrides0, int xstrides1, int xstrides2, int xstrides3, int scalarCount) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < scalarCount; tid += stride) {
    int linearIndex = tid;
    int xindex0 = linearIndex / xstrides0;
    linearIndex = linearIndex - xstrides0 * xindex0;
    int xindex1 = linearIndex / xstrides1;
    linearIndex = linearIndex - xstrides1 * xindex1;
    int xindex2 = linearIndex / xstrides2;
    int xindex3 = linearIndex - xstrides2 * xindex2;
    if (xindex3 >= mask[xindex0]) in[tid] = 0;
  }
}

__global__ void mul_sub(float* in1, float* in2, float* out, int in1ScalarCount, int in2ScalarCount) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < in1ScalarCount; tid += stride) {
    out[tid] = in1[tid] * in2[tid % in2ScalarCount];
  }
}

__global__ void mul_sub_grad(float* in1_x, float* in1_d, float* in2_x, float* in2_d, float* out, int in1ScalarCount, int in2ScalarCount) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < in1ScalarCount; tid += stride) {
    int index = tid % in2ScalarCount;
    in1_d[tid] += out[tid] * in2_x[index];
    in2_d[tid] = in1_x[tid] * out[tid];  // this is the temp array, need to be reduced!
  }
}

// From: https://github.com/pytorch/pytorch/blob/master/aten/src/THC/THCIntegerDivider.cuh
// Result of div/mod operation stored together.
template <typename Value>
struct DivMod {
  Value div, mod;

  __host__ __device__ DivMod(Value div, Value mod) : div(div), mod(mod) { }
};

// Base case: we only have an implementation for uint32_t for now.  For
// everything else, we use plain division.
template <typename Value>
struct IntDivider {
  IntDivider() { }  // Dummy constructor for arrays.
  IntDivider(Value d) : divisor(d) { }

  __host__ __device__ inline Value div(Value n) const { return n / divisor; }
  __host__ __device__ inline Value mod(Value n) const { return n % divisor; }
  __host__ __device__ inline DivMod<Value> divmod(Value n) const {
    return DivMod<Value>(n / divisor, n % divisor);
  }

  Value divisor;
};

// Implement fast integer division.
template <>
struct IntDivider<unsigned int> {
  static_assert(sizeof(unsigned int) == 4, "Assumes 32-bit unsigned int.");

  IntDivider() { }  // Dummy constructor for arrays.

  IntDivider(unsigned int d) : divisor(d) {
    assert(divisor >= 1 && divisor <= INT32_MAX);

    // TODO: gcc/clang has __builtin_clz() but it's not portable.
    for (shift = 0; shift < 32; shift++) if ((1U << shift) >= divisor) break;

    uint64_t one = 1;
    uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;
    m1 = magic;
    assert(m1 > 0 && m1 == magic);  // m1 must fit in 32 bits.
  }

  __host__ __device__ inline unsigned int div(unsigned int n) const {
#ifdef __CUDA_ARCH__
    // 't' is the higher 32-bits of unsigned 32-bit multiplication of 'n' and
    // 'm1'.
    unsigned int t = __umulhi(n, m1);
    return (t + n) >> shift;
#else
    // Using uint64_t so that the addition does not overflow.
    uint64_t t = ((uint64_t) n * m1) >> 32;
    return (t + n) >> shift;
#endif
  }

  __host__ __device__ inline unsigned int mod(unsigned int n) const {
    return n - div(n) * divisor;
  }

  __host__ __device__ inline DivMod<unsigned int> divmod(unsigned int n) const {
    unsigned int q = div(n);
    return DivMod<unsigned int>(q, n - q * divisor);
  }

  unsigned int divisor;  // d above.
  unsigned int m1;  // Magic number: m' above.
  unsigned int shift;  // Shift amounts.
};

// From: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/OffsetCalculator.cuh
/// OffsetCalculator calculates the offset in bytes of a linear index for NARGS
/// operands that share the same shape, but may have different strides.

template <int NARGS>
struct OffsetCalculator {
  static constexpr int MAX_DIMS = 25;

  // The offset for each argument (in bytes). Wrapper around fixed-size array.
  struct offsets_t {
    __host__ __device__ uint32_t& operator[](int idx) {
      return values[idx];
    }
    uint32_t values[NARGS];
  };


  // OffsetCalculator(int dims, const int64_t* sizes, const int64_t* const* strides) : dims(dims) {
  OffsetCalculator(int dims, const int32_t* sizes, const int32_t* const* strides) : dims(dims) {
    for (int i = 0; i < MAX_DIMS; ++i) {
      if (i < dims) {
        sizes_[i] = IntDivider<uint32_t>(sizes[i]);
      } else {
        sizes_[i] = IntDivider<uint32_t>(1);
      }
      for (int arg = 0; arg < NARGS; arg++) {
        strides_[i][arg] = i < dims ? strides[arg][i] : 0;
      }
    }
  }

  __host__ __device__ offsets_t get(uint32_t linear_idx) const {
    offsets_t offsets;
#pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = 0;
    }

#pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims) {
        break;
      }
      auto divmod = sizes_[dim].divmod(linear_idx);
      linear_idx = divmod.div;

#pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] += divmod.mod * strides_[dim][arg];
      }
    }
    return offsets;
  }

  void print() {
    for (auto i = 1; i < 128; i++) {
      auto offsets = get(i);
      printf("offsets[%d]: ", i);
      for (auto arg = 0; arg < NARGS; arg++) {
        printf("%d ", offsets[arg]);
      }
      printf("\n");
    }
  }

  int dims;
  IntDivider<uint32_t> sizes_[MAX_DIMS];
  uint32_t strides_[MAX_DIMS][NARGS];
};

// From: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/Loops.cuh
template<int nt, int vt, typename func_t>
__launch_bounds__(nt, 4)
__global__ void elementwise_kernel(int N, func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template<int nt, int vt, typename func_t>
static void launch_kernel(int64_t N, const func_t& f) {
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  elementwise_kernel<nt, vt, func_t><<<grid, block, 0>>>(N, f);
}

template<typename func_t>
void gpu_unary_kernel(float *res, float *x,
                      int32_t resRank, const int32_t resScalarCount,
                      const int32_t* resShape, const int32_t* const* strides,
                      const func_t& f) {
  OffsetCalculator<2> calc(resRank, resShape, strides);
  launch_kernel<128, 4>(resScalarCount, [=]__device__(int idx) {
    auto offsets = calc.get(idx);
    float* out = &res[offsets[0]];
    float* in = &x[offsets[1]];
    *out = f(*in);
  });
}

template<typename func_t>
void gpu_binary_kernel(float *res, float *x, float *y,
                       int32_t resRank, const int32_t resScalarCount,
                       const int32_t* resShape, const int32_t* const* strides,
                       const func_t& f) {
  OffsetCalculator<3> calc(resRank, resShape, strides);
  launch_kernel<128, 4>(resScalarCount, [=]__device__(int idx) {
    auto offsets = calc.get(idx);
    float* out = &res[offsets[0]];
    float* in1 = &x[offsets[1]];
    float* in2 = &y[offsets[2]];
    *out = f(*in1, *in2);
  });
}

#define CUDNN_CALL(f) { \
  cudnnStatus_t stat = (f); \
  if (stat != CUDNN_STATUS_SUCCESS) { \
    fprintf(stderr, "cuDNN error occurred: %d (%s:%d)\n", \
            stat, __FILE__, __LINE__); \
    exit(stat); \
  } \
}


void Snippet(char *);

std::random_device rd{};
std::mt19937 gen{rd()};
std::normal_distribution<> d{0, 0.01};

int main(int argc, char *argv[]) {
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
void Snippet(char*  x0) {
// Backend setup.
cublasHandle_t cublasHandle;
CUBLAS_CALL(cublasCreate(&cublasHandle));
CUDA_CALL(cudaMalloc(&gpuMallocBase, HEAP_SIZE));
CUDA_CALL(cudaMemset(gpuMallocBase, 0, HEAP_SIZE));
gpuMallocAddr = gpuMallocBase;
      
cudnnHandle_t cudnnHandle;
CUDNN_CALL(cudnnCreate(&cudnnHandle));
srand(42);
struct timeval begin_0, end_0, diff_0;
gettimeofday(&begin_0, NULL);
float* x8 = (float*)myMalloc(14432 * sizeof(float));;
for(int x10=0; x10 < 14432; x10++) {
float x11 = (float)rand()/RAND_MAX;
float x12 = x11 - 0.5f;
float x13 = x12 * 0.23068394f;
x8[x10] = x13;

}
// Tensor 'toGPU' invocation.
float* x18 = (float*)myGpuMalloc(14432 * sizeof(float));
CUDA_CALL(cudaMemcpy(x18, x8, 14432 * sizeof(float), cudaMemcpyHostToDevice));
float* x20 = (float*)myGpuMalloc(14432 * sizeof(float));
float* x21 = (float*)myGpuMalloc(32 * sizeof(float));
arrayFill<<<28, 512>>>(x21, 1.0f, 32);
float* x23 = (float*)myGpuMalloc(32 * sizeof(float));
float* x24 = (float*)myGpuMalloc(32 * sizeof(float));
float* x25 = (float*)myGpuMalloc(32 * sizeof(float));
float* x26 = (float*)myGpuMalloc(32 * sizeof(float));
float* x27 = (float*)myGpuMalloc(32 * sizeof(float));
float* x28 = (float*)myMalloc(236544 * sizeof(float));;
for(int x30=0; x30 < 236544; x30++) {
float x31 = (float)rand()/RAND_MAX;
float x32 = x31 - 0.5f;
float x33 = x32 * 0.05698029f;
x28[x30] = x33;

}
// Tensor 'toGPU' invocation.
float* x38 = (float*)myGpuMalloc(236544 * sizeof(float));
CUDA_CALL(cudaMemcpy(x38, x28, 236544 * sizeof(float), cudaMemcpyHostToDevice));
float* x40 = (float*)myGpuMalloc(236544 * sizeof(float));
float* x41 = (float*)myGpuMalloc(32 * sizeof(float));
arrayFill<<<28, 512>>>(x41, 1.0f, 32);
float* x43 = (float*)myGpuMalloc(32 * sizeof(float));
float* x44 = (float*)myGpuMalloc(32 * sizeof(float));
float* x45 = (float*)myGpuMalloc(32 * sizeof(float));
float* x46 = (float*)myGpuMalloc(32 * sizeof(float));
float* x47 = (float*)myGpuMalloc(32 * sizeof(float));
printf("initial rnn input size is %d \n",672);
printf("inputSize for batchRNN is %d\n",672);
int32_t x50 = 0;
float* x51 = (float*)myGpuMalloc(3477504 * sizeof(float));
arrayFill<<<28, 512>>>(x51, 0.01f, 3477504);
float* x53 = (float*)myGpuMalloc(3477504 * sizeof(float));
int32_t x54 = x50;
float* x55 = x51+x54;
float* x56 = x53+x54;
x50 += 688128;
int32_t x58 = x50;
float* x59 = x51+x58;
float* x60 = x53+x58;
x50 += 1048576;
int32_t x62 = x50;
float* x63 = x51+x62;
float* x64 = x53+x62;
x50 += 688128;
int32_t x66 = x50;
float* x67 = x51+x66;
float* x68 = x53+x66;
x50 += 1048576;
int32_t x70 = x50;
float* x71 = x51+x70;
float* x72 = x53+x70;
x50 += 1024;
int32_t x74 = x50;
float* x75 = x51+x74;
float* x76 = x53+x74;
x50 += 1024;
int32_t x78 = x50;
float* x79 = x51+x78;
float* x80 = x53+x78;
x50 += 1024;
int32_t x82 = x50;
float* x83 = x51+x82;
float* x84 = x53+x82;
x50 += 1024;
printf("inputSize for batchRNN is %d\n",1024);
int32_t x87 = 0;
float* x88 = (float*)myGpuMalloc(4198400 * sizeof(float));
arrayFill<<<28, 512>>>(x88, 0.01f, 4198400);
float* x90 = (float*)myGpuMalloc(4198400 * sizeof(float));
int32_t x91 = x87;
float* x92 = x88+x91;
float* x93 = x90+x91;
x87 += 1048576;
int32_t x95 = x87;
float* x96 = x88+x95;
float* x97 = x90+x95;
x87 += 1048576;
int32_t x99 = x87;
float* x100 = x88+x99;
float* x101 = x90+x99;
x87 += 1048576;
int32_t x103 = x87;
float* x104 = x88+x103;
float* x105 = x90+x103;
x87 += 1048576;
int32_t x107 = x87;
float* x108 = x88+x107;
float* x109 = x90+x107;
x87 += 1024;
int32_t x111 = x87;
float* x112 = x88+x111;
float* x113 = x90+x111;
x87 += 1024;
int32_t x115 = x87;
float* x116 = x88+x115;
float* x117 = x90+x115;
x87 += 1024;
int32_t x119 = x87;
float* x120 = x88+x119;
float* x121 = x90+x119;
x87 += 1024;
printf("inputSize for batchRNN is %d\n",1024);
int32_t x124 = 0;
float* x125 = (float*)myGpuMalloc(4198400 * sizeof(float));
arrayFill<<<28, 512>>>(x125, 0.01f, 4198400);
float* x127 = (float*)myGpuMalloc(4198400 * sizeof(float));
int32_t x128 = x124;
float* x129 = x125+x128;
float* x130 = x127+x128;
x124 += 1048576;
int32_t x132 = x124;
float* x133 = x125+x132;
float* x134 = x127+x132;
x124 += 1048576;
int32_t x136 = x124;
float* x137 = x125+x136;
float* x138 = x127+x136;
x124 += 1048576;
int32_t x140 = x124;
float* x141 = x125+x140;
float* x142 = x127+x140;
x124 += 1048576;
int32_t x144 = x124;
float* x145 = x125+x144;
float* x146 = x127+x144;
x124 += 1024;
int32_t x148 = x124;
float* x149 = x125+x148;
float* x150 = x127+x148;
x124 += 1024;
int32_t x152 = x124;
float* x153 = x125+x152;
float* x154 = x127+x152;
x124 += 1024;
int32_t x156 = x124;
float* x157 = x125+x156;
float* x158 = x127+x156;
x124 += 1024;
float* x160 = (float*)myGpuMalloc(1024 * sizeof(float));
arrayFill<<<28, 512>>>(x160, 1.0f, 1024);
float* x162 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x163 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x164 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x165 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x166 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x167 = (float*)myMalloc(29696 * sizeof(float));;
for(int x169=0; x169 < 29696; x169++) {
float x170 = (float)rand()/RAND_MAX;
float x171 = x170 - 0.5f;
float x172 = x171 * 0.03125f;
x167[x169] = x172;

}
// Tensor 'toGPU' invocation.
float* x177 = (float*)myGpuMalloc(29696 * sizeof(float));
CUDA_CALL(cudaMemcpy(x177, x167, 29696 * sizeof(float), cudaMemcpyHostToDevice));
float* x179 = (float*)myGpuMalloc(29696 * sizeof(float));
int32_t x180 = open("/scratch/wu636/training/speech_recognition/data/test/deepspeech_train.bin",0);
int64_t x181 = fsize(x180);
printf("file size is %ld\n",x181);
char* x183 = (char*)mmap(0, x181, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x180, 0);
int64_t x184 = (long)x183;
int64_t x185 = x184;
int64_t x186 = x185;
int* x187 = (int32_t*) x186;
int64_t x188 = (int64_t)4;
x185 += x188;
int32_t x190 = x187[0];
int64_t x191 = x185;
int* x192 = (int32_t*) x191;
x185 += x188;
int32_t x194 = x192[0];
printf("data size is %d batches, %d batch size\n",200,x190);
int* x197 = (int32_t*)myMalloc(200 * sizeof(int32_t));;
int* x198 = (int32_t*)myMalloc(200 * sizeof(int32_t));;
float** x199 = (float**)myMalloc(200 * sizeof(float*));;
float** x200 = (float**)myMalloc(200 * sizeof(float*));;
int** x201 = (int**)myMalloc(200 * sizeof(int*));;
int** x202 = (int**)myMalloc(200 * sizeof(int*));;
// load data by batchs
int32_t x228 = 4 * x190;
int64_t x229 = (int64_t)x228;
for(int x205=0; x205 < 200; x205++) {
int64_t x206 = x185;
int* x207 = (int32_t*) x206;
x185 += x188;
int32_t x209 = x207[0];
x197[x205] = x209;
int64_t x211 = x185;
int* x212 = (int32_t*) x211;
x185 += x188;
int32_t x214 = x212[0];
x198[x205] = x214;
int32_t x216 = x197[x205];
int32_t x218 = x198[x205];
int64_t x220 = x185;
float* x221 = (float*) x220;
int32_t x217 = x190 * x216;
int32_t x219 = x217 * x218;
int32_t x222 = 4 * x219;
int64_t x223 = (int64_t)x222;
x185 += x223;
x199[x205] = x221;
int64_t x226 = x185;
float* x227 = (float*) x226;
x185 += x229;
x200[x205] = x227;
int64_t x232 = x185;
int* x233 = (int32_t*) x232;
x185 += x229;
x201[x205] = x233;
int* x236 = x201[x205];
int* x237 = x201[x205];
int32_t x238 = accumulate(x236, x237 + x190, 0);
int64_t x239 = x185;
int* x240 = (int32_t*) x239;
int32_t x241 = 4 * x238;
int64_t x242 = (int64_t)x241;
x185 += x242;
x202[x205] = x240;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x249 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
float x250 = (float)x249;
float x251 = x250 / 1000000.0f;
printf("Data reading (all prepare time) in %lf sec\n",x251);
double* x253 = (double*)myMalloc(1 * sizeof(double));;
double* x254 = (double*)myMalloc(1 * sizeof(double));;
int64_t x255 = (long)mallocAddr;
int64_t x256 = (long)gpuMallocAddr;
// training loop starts here
int32_t x304 = x190 * 32;
bool x363 = x190 < 0;
bool x397 = x190 > 0;
int32_t x446 = 2048 / 2;
bool x462 = x446 < 0;
bool x493 = x446 > 0;
int32_t x1329 = x190 * 20;
int32_t x195 = x190 * 200;
double x1334 = (double)x195;
int64_t x1357 = (int64_t)x195;
float x1364 = (float)x195;
for(int x259=0; x259 < 1; x259++) {
struct timeval begin_1, end_1, diff_1;
int32_t x261 = 0;
int32_t x262 = x261;
int32_t x263 = x262;
float x264 = 0.0f;
float x265 = x264;
float x266 = x265;
int32_t x267 = x259 + 1;
printf("Start training epoch %d\n",x267);
gettimeofday(&begin_1, NULL);
for(int x270=0; x270 < 200; x270++) {
int32_t x271 = x198[x270];
int32_t x272 = x197[x270];
float* x273 = x199[x270];
float* x276 = x200[x270];
int* x277 = x202[x270];
int* x278 = x201[x270];
x263 += x190;
// Tensor 'toGPU' invocation.
int32_t x274 = x272 * x271;
int32_t x275 = x190 * x274;
float* x281 = (float*)myGpuMalloc(x275 * sizeof(float));
CUDA_CALL(cudaMemcpy(x281, x273, x275 * sizeof(float), cudaMemcpyHostToDevice));
float* x283 = (float*)myGpuMalloc(2 * sizeof(float));
float* x284 = (float*)myGpuMalloc(1 * sizeof(float));
float* x285 = (float*)myGpuMalloc(1 * sizeof(float));
// allocate memory to save the final loss in CPU Tensor
float* x287 = (float*)myGpuMalloc(1 * sizeof(float));
bool x288 = x272 >= 41;
bool x290;
if (x288) {
bool x289 = x271 >= 11;
x290 = x289;
} else {
x290 = false;
}
if (x290) {
} else {
assert(false && "ERROR not specified");
}
int32_t x298 = x271 - 11;
int32_t x299 = x298 / 2;
int32_t x300 = x299 + 1;
int32_t x295 = x272 - 41;
int32_t x296 = x295 / 2;
int32_t x297 = x296 + 1;
int32_t x305 = x304 * x297;
int32_t x306 = x305 * x300;
float* x307 = (float*)myGpuMalloc(x306 * sizeof(float));
float* x308 = (float*)myMalloc(1 * sizeof(float));;
x308[0] = 0.0f;
float* x310 = (float*)myMalloc(1 * sizeof(float));;
x310[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x190, 1, x272, x271));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    32, 1, 41, 11));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x190, 32, x297, x300));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 2, 2, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
// Algorithm.
cudnnConvolutionFwdAlgo_t algo;
CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
    cudnnHandle,
    in_desc, filt_desc, conv_desc, out_desc,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
// Execute convolution.
CUDNN_CALL(cudnnConvolutionForward(
    cudnnHandle,
    x310, in_desc, x281, filt_desc, x18,
    conv_desc, algo, ws_data, ws_size,
    x308, out_desc, x307));
};
float* x313 = (float*)myGpuMalloc(x306 * sizeof(float));
int32_t x301 = x297 * x300;
int32_t x302 = 32 * x301;
int32_t x303 = x190 * x302;
float* x314 = (float*)myGpuMalloc(x303 * sizeof(float));
float* x315 = (float*)myGpuMalloc(32 * sizeof(float));
float* x316 = (float*)myGpuMalloc(32 * sizeof(float));
float* x317 = (float*)myMalloc(1 * sizeof(float));;
x317[0] = 0.0f;
float* x319 = (float*)myMalloc(1 * sizeof(float));;
x319[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x190, 32, x297, x300));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x190, 32, x297, x300));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 32, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x319, x317, in_desc, x307, out_desc, x314, sbmv_desc, x21,
    x24, 0.1, x26, x27, 1.0E-5,
    x315, x316));
};
float* x322 = (float*)myGpuMalloc(x306 * sizeof(float));
hardTanh<<<28, 512>>>(x314, x314, 0.0, 20.0, true);
bool x324 = x297 >= 21;
bool x326;
if (x324) {
bool x325 = x300 >= 11;
x326 = x325;
} else {
x326 = false;
}
if (x326) {
} else {
assert(false && "ERROR not specified");
}
int32_t x334 = x300 - 11;
int32_t x335 = x334 / 1;
int32_t x336 = x335 + 1;
int32_t x331 = x297 - 21;
int32_t x332 = x331 / 2;
int32_t x333 = x332 + 1;
int32_t x340 = x304 * x333;
int32_t x341 = x340 * x336;
float* x342 = (float*)myGpuMalloc(x341 * sizeof(float));
float* x343 = (float*)myMalloc(1 * sizeof(float));;
x343[0] = 0.0f;
float* x345 = (float*)myMalloc(1 * sizeof(float));;
x345[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x190, 32, x297, x300));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    32, 32, 21, 11));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x190, 32, x333, x336));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 2, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
// Algorithm.
cudnnConvolutionFwdAlgo_t algo;
CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
    cudnnHandle,
    in_desc, filt_desc, conv_desc, out_desc,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
// Execute convolution.
CUDNN_CALL(cudnnConvolutionForward(
    cudnnHandle,
    x345, in_desc, x314, filt_desc, x38,
    conv_desc, algo, ws_data, ws_size,
    x343, out_desc, x342));
};
float* x348 = (float*)myGpuMalloc(x341 * sizeof(float));
int32_t x337 = x333 * x336;
int32_t x338 = 32 * x337;
int32_t x339 = x190 * x338;
float* x349 = (float*)myGpuMalloc(x339 * sizeof(float));
float* x350 = (float*)myGpuMalloc(32 * sizeof(float));
float* x351 = (float*)myGpuMalloc(32 * sizeof(float));
float* x352 = (float*)myMalloc(1 * sizeof(float));;
x352[0] = 0.0f;
float* x354 = (float*)myMalloc(1 * sizeof(float));;
x354[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x190, 32, x333, x336));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x190, 32, x333, x336));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 32, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x354, x352, in_desc, x342, out_desc, x349, sbmv_desc, x41,
    x44, 0.1, x46, x47, 1.0E-5,
    x350, x351));
};
float* x357 = (float*)myGpuMalloc(x341 * sizeof(float));
hardTanh<<<28, 512>>>(x349, x349, 0.0, 20.0, true);
// after conv ops
int32_t x361 = 0;
int32_t x362 = 1;
if (x363) {
x361 += 1;
} else {
x362 *= x190;
}
int32_t x360 = 32 * x333;
bool x369 = x360 < 0;
if (x369) {
x361 += 1;
} else {
x362 *= x360;
}
bool x375 = x336 < 0;
if (x375) {
x361 += 1;
} else {
x362 *= x336;
}
int32_t x381 = x361;
bool x382 = x381 >= 2;
if (x382) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x388 = x381 == 0;
if (x388) {
int32_t x389 = x362;
bool x390 = x389 == x339;
if (x390) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x401;
if (x397) {
x401 = x190;
} else {
int32_t x398 = x362;
int32_t x399 = x339 / x398;
x401 = x399;
}
bool x402 = x360 > 0;
int32_t x406;
if (x402) {
x406 = x360;
} else {
int32_t x403 = x362;
int32_t x404 = x339 / x403;
x406 = x404;
}
bool x407 = x336 > 0;
int32_t x411;
if (x407) {
x411 = x336;
} else {
int32_t x408 = x362;
int32_t x409 = x339 / x408;
x411 = x409;
}
int32_t x412 = x406 * x411;
int32_t x413 = x401 * x412;
float* x414 = (float*)myGpuMalloc(x413 * sizeof(float));
int* x417 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
int32_t x415 = x401 * x406;
x417[2] = x415;
x417[0] = x406;
x417[1] = 1;
x417[3] = 1;
float* x422 = (float*)myMalloc(1 * sizeof(float));;
x422[0] = 1.0f;
float* x424 = (float*)myMalloc(0 * sizeof(float));;
x424[0] = 0.0f;
int32_t x426 = x417[0];
int32_t x427 = x417[1];
int32_t x428 = x417[2];
int32_t x429 = x417[3];

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
    in_desc, CUDNN_DATA_FLOAT,
    x401, x406, x411, 1,
    x412, x411, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
    out_desc, CUDNN_DATA_FLOAT,
    x401, x406, x411, 1,
    x426, x427, x428, x429));

CUDNN_CALL(cudnnTransformTensor(
    cudnnHandle, x422, in_desc, x349, x424, out_desc, x414));
};
int32_t x431 = x411 * x401;
int32_t x432 = x431 * x406;
float* x433 = (float*)myGpuMalloc(x432 * sizeof(float));
// after resize and permute
float* x435 = (float*)NULL;
float* x436 = (float*)NULL;
float* x437 = (float*)NULL;
int32_t x440 = x431 * 2048;
float* x441 = (float*)myGpuMalloc(x440 * sizeof(float));
float* x442 = (float*)NULL;
int32_t x443 = 0;

{
size_t dropoutStateSize;
CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize));
void* dropoutStates = myGpuMalloc(dropoutStateSize);

cudnnDropoutDescriptor_t dropout_desc;
CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
CUDNN_CALL(cudnnSetDropoutDescriptor(
    dropout_desc, cudnnHandle, 0.0, dropoutStates, dropoutStateSize, time(NULL)));

cudnnRNNDescriptor_t rnn_desc;
CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
CUDNN_CALL(cudnnSetRNNDescriptor(
    cudnnHandle, rnn_desc,
    /*hiddenSize*/ 1024, /*numLayers*/ 1,
    dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
    CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH));
int32_t seqLength = x411;
int32_t batchSize = x401;
int32_t inputSize = x406;

cudnnTensorDescriptor_t x_descs[seqLength];
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
int x_dims[] = {batchSize, inputSize, 1};
int x_strides[] = {x_dims[1] * x_dims[2], x_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    x_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims, x_strides));
for (int i = 0; i < seqLength; i++) {
  x_descs[i] = x_desc;
}

// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
// The second dimension must match the first dimension of the tensors described in xDesc.
// The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
cudnnTensorDescriptor_t hx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc));
int hx_dims[] = {2, batchSize, 1024};
int hx_strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    hx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims, hx_strides));

cudnnTensorDescriptor_t cx_desc = hx_desc;

size_t paramsSize;
CUDNN_CALL(cudnnGetRNNParamsSize(
    cudnnHandle, rnn_desc, x_descs[0], &paramsSize, CUDNN_DATA_FLOAT));
assert(paramsSize / sizeof(float) == 3477504 && "Expected parameter size mismatch");

cudnnFilterDescriptor_t w_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
int w_dims[] = {int(paramsSize / sizeof(float)), 1, 1};
CUDNN_CALL(cudnnSetFilterNdDescriptor(
    w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims));

cudnnTensorDescriptor_t y_descs[seqLength];
cudnnTensorDescriptor_t y_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
int y_dims[] = {batchSize, 2048, 1};
int y_strides[] = {y_dims[1] * y_dims[2], y_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    y_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims, y_strides));
for (int i = 0; i < seqLength; i++) {
  y_descs[i] = y_desc;
}

cudnnTensorDescriptor_t hy_desc = hx_desc;
cudnnTensorDescriptor_t cy_desc = cx_desc;

size_t workspaceSize;
CUDNN_CALL(cudnnGetRNNWorkspaceSize(
    cudnnHandle, rnn_desc, seqLength, x_descs, &workspaceSize));
void* workspace = myGpuMalloc(workspaceSize);

// Reserve space used by `ForwardTraining` function.
size_t reserveSize;
CUDNN_CALL(cudnnGetRNNTrainingReserveSize(
    cudnnHandle, rnn_desc, seqLength, x_descs, &reserveSize));
void* reserveSpace = myGpuMalloc(reserveSize);
x442 = (float*)reserveSpace;
x443 = (int)reserveSize;
CUDNN_CALL(cudnnRNNForwardTraining(
    cudnnHandle, rnn_desc, seqLength, x_descs, x414,
    hx_desc,x435, cx_desc,x436, w_desc, x51, y_descs, x441,
    hy_desc,x437, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
};
float* x445 = (float*)myGpuMalloc(x440 * sizeof(float));
int32_t x447 = 0;
int32_t x448 = 1;
bool x449 = x411 < 0;
if (x449) {
x447 += 1;
} else {
x448 *= x411;
}
bool x455 = x401 < 0;
if (x455) {
x447 += 1;
} else {
x448 *= x401;
}
x448 *= 2;
if (x462) {
x447 += 1;
} else {
x448 *= x446;
}
int32_t x468 = x447;
bool x469 = x468 >= 2;
if (x469) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x474 = x468 == 0;
int32_t x438 = x401 * 2048;
int32_t x439 = x411 * x438;
if (x474) {
int32_t x475 = x448;
bool x476 = x475 == x439;
if (x476) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x483 = x411 > 0;
int32_t x487;
if (x483) {
x487 = x411;
} else {
int32_t x484 = x448;
int32_t x485 = x439 / x484;
x487 = x485;
}
bool x488 = x401 > 0;
int32_t x492;
if (x488) {
x492 = x401;
} else {
int32_t x489 = x448;
int32_t x490 = x439 / x489;
x492 = x490;
}
int32_t x497;
if (x493) {
x497 = x446;
} else {
int32_t x494 = x448;
int32_t x495 = x439 / x494;
x497 = x495;
}
int32_t x501 = 0;
int32_t x502 = 1;
bool x503 = x487 < 0;
if (x503) {
x501 += 1;
} else {
x502 *= x487;
}
bool x509 = x492 < 0;
if (x509) {
x501 += 1;
} else {
x502 *= x492;
}
x502 *= 2;
bool x516 = x497 < 0;
if (x516) {
x501 += 1;
} else {
x502 *= x497;
}
int32_t x522 = x501;
bool x523 = x522 >= 2;
if (x523) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x528 = x522 == 0;
int32_t x498 = 2 * x497;
int32_t x499 = x492 * x498;
int32_t x500 = x487 * x499;
if (x528) {
int32_t x529 = x502;
bool x530 = x529 == x500;
if (x530) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x537 = x487 > 0;
int32_t x541;
if (x537) {
x541 = x487;
} else {
int32_t x538 = x502;
int32_t x539 = x500 / x538;
x541 = x539;
}
bool x542 = x492 > 0;
int32_t x546;
if (x542) {
x546 = x492;
} else {
int32_t x543 = x502;
int32_t x544 = x500 / x543;
x546 = x544;
}
bool x547 = x497 > 0;
int32_t x551;
if (x547) {
x551 = x497;
} else {
int32_t x548 = x502;
int32_t x549 = x500 / x548;
x551 = x549;
}
int32_t x557 = x541 * x546;
int32_t x558 = x557 * x551;
float* x559 = (float*)myGpuMalloc(x558 * sizeof(float));
float* x560 = (float*)myMalloc(1 * sizeof(float));;
x560[0] = 0.0f;
float* x562 = (float*)myMalloc(1 * sizeof(float));;
x562[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x541, x546, 2, x551));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x541, x546, 1, x551));

cudnnReduceTensorDescriptor_t reduce_desc;
CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&reduce_desc));
CUDNN_CALL(cudnnSetReduceTensorDescriptor(
    reduce_desc, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
    CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));

void *indices = nullptr; // Don't store indices.

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetReductionWorkspaceSize(
    cudnnHandle, reduce_desc, x_desc, out_desc, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnReduceTensor(
    cudnnHandle, reduce_desc, indices, 0, ws_data, ws_size,
    x562, x_desc, x441, x560, out_desc, x559));
};
float* x565 = (float*)myGpuMalloc(x558 * sizeof(float));
float* x566 = (float*)NULL;
float* x567 = (float*)NULL;
float* x568 = (float*)NULL;
int32_t x571 = x557 * 2048;
float* x572 = (float*)myGpuMalloc(x571 * sizeof(float));
float* x573 = (float*)NULL;
int32_t x574 = 0;

{
size_t dropoutStateSize;
CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize));
void* dropoutStates = myGpuMalloc(dropoutStateSize);

cudnnDropoutDescriptor_t dropout_desc;
CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
CUDNN_CALL(cudnnSetDropoutDescriptor(
    dropout_desc, cudnnHandle, 0.0, dropoutStates, dropoutStateSize, time(NULL)));

cudnnRNNDescriptor_t rnn_desc;
CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
CUDNN_CALL(cudnnSetRNNDescriptor(
    cudnnHandle, rnn_desc,
    /*hiddenSize*/ 1024, /*numLayers*/ 1,
    dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
    CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH));
int32_t seqLength = x541;
int32_t batchSize = x546;
int32_t inputSize = x551;

cudnnTensorDescriptor_t x_descs[seqLength];
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
int x_dims[] = {batchSize, inputSize, 1};
int x_strides[] = {x_dims[1] * x_dims[2], x_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    x_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims, x_strides));
for (int i = 0; i < seqLength; i++) {
  x_descs[i] = x_desc;
}

// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
// The second dimension must match the first dimension of the tensors described in xDesc.
// The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
cudnnTensorDescriptor_t hx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc));
int hx_dims[] = {2, batchSize, 1024};
int hx_strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    hx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims, hx_strides));

cudnnTensorDescriptor_t cx_desc = hx_desc;

size_t paramsSize;
CUDNN_CALL(cudnnGetRNNParamsSize(
    cudnnHandle, rnn_desc, x_descs[0], &paramsSize, CUDNN_DATA_FLOAT));
assert(paramsSize / sizeof(float) == 4198400 && "Expected parameter size mismatch");

cudnnFilterDescriptor_t w_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
int w_dims[] = {int(paramsSize / sizeof(float)), 1, 1};
CUDNN_CALL(cudnnSetFilterNdDescriptor(
    w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims));

cudnnTensorDescriptor_t y_descs[seqLength];
cudnnTensorDescriptor_t y_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
int y_dims[] = {batchSize, 2048, 1};
int y_strides[] = {y_dims[1] * y_dims[2], y_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    y_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims, y_strides));
for (int i = 0; i < seqLength; i++) {
  y_descs[i] = y_desc;
}

cudnnTensorDescriptor_t hy_desc = hx_desc;
cudnnTensorDescriptor_t cy_desc = cx_desc;

size_t workspaceSize;
CUDNN_CALL(cudnnGetRNNWorkspaceSize(
    cudnnHandle, rnn_desc, seqLength, x_descs, &workspaceSize));
void* workspace = myGpuMalloc(workspaceSize);

// Reserve space used by `ForwardTraining` function.
size_t reserveSize;
CUDNN_CALL(cudnnGetRNNTrainingReserveSize(
    cudnnHandle, rnn_desc, seqLength, x_descs, &reserveSize));
void* reserveSpace = myGpuMalloc(reserveSize);
x573 = (float*)reserveSpace;
x574 = (int)reserveSize;
CUDNN_CALL(cudnnRNNForwardTraining(
    cudnnHandle, rnn_desc, seqLength, x_descs, x559,
    hx_desc,x566, cx_desc,x567, w_desc, x88, y_descs, x572,
    hy_desc,x568, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
};
float* x576 = (float*)myGpuMalloc(x571 * sizeof(float));
int32_t x577 = 0;
int32_t x578 = 1;
bool x579 = x541 < 0;
if (x579) {
x577 += 1;
} else {
x578 *= x541;
}
bool x585 = x546 < 0;
if (x585) {
x577 += 1;
} else {
x578 *= x546;
}
x578 *= 2;
if (x462) {
x577 += 1;
} else {
x578 *= x446;
}
int32_t x597 = x577;
bool x598 = x597 >= 2;
if (x598) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x603 = x597 == 0;
int32_t x569 = x546 * 2048;
int32_t x570 = x541 * x569;
if (x603) {
int32_t x604 = x578;
bool x605 = x604 == x570;
if (x605) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x612 = x541 > 0;
int32_t x616;
if (x612) {
x616 = x541;
} else {
int32_t x613 = x578;
int32_t x614 = x570 / x613;
x616 = x614;
}
bool x617 = x546 > 0;
int32_t x621;
if (x617) {
x621 = x546;
} else {
int32_t x618 = x578;
int32_t x619 = x570 / x618;
x621 = x619;
}
int32_t x625;
if (x493) {
x625 = x446;
} else {
int32_t x622 = x578;
int32_t x623 = x570 / x622;
x625 = x623;
}
int32_t x629 = 0;
int32_t x630 = 1;
bool x631 = x616 < 0;
if (x631) {
x629 += 1;
} else {
x630 *= x616;
}
bool x637 = x621 < 0;
if (x637) {
x629 += 1;
} else {
x630 *= x621;
}
x630 *= 2;
bool x644 = x625 < 0;
if (x644) {
x629 += 1;
} else {
x630 *= x625;
}
int32_t x650 = x629;
bool x651 = x650 >= 2;
if (x651) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x656 = x650 == 0;
int32_t x626 = 2 * x625;
int32_t x627 = x621 * x626;
int32_t x628 = x616 * x627;
if (x656) {
int32_t x657 = x630;
bool x658 = x657 == x628;
if (x658) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x665 = x616 > 0;
int32_t x669;
if (x665) {
x669 = x616;
} else {
int32_t x666 = x630;
int32_t x667 = x628 / x666;
x669 = x667;
}
bool x670 = x621 > 0;
int32_t x674;
if (x670) {
x674 = x621;
} else {
int32_t x671 = x630;
int32_t x672 = x628 / x671;
x674 = x672;
}
bool x675 = x625 > 0;
int32_t x679;
if (x675) {
x679 = x625;
} else {
int32_t x676 = x630;
int32_t x677 = x628 / x676;
x679 = x677;
}
int32_t x685 = x669 * x674;
int32_t x686 = x685 * x679;
float* x687 = (float*)myGpuMalloc(x686 * sizeof(float));
float* x688 = (float*)myMalloc(1 * sizeof(float));;
x688[0] = 0.0f;
float* x690 = (float*)myMalloc(1 * sizeof(float));;
x690[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x669, x674, 2, x679));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x669, x674, 1, x679));

cudnnReduceTensorDescriptor_t reduce_desc;
CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&reduce_desc));
CUDNN_CALL(cudnnSetReduceTensorDescriptor(
    reduce_desc, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
    CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));

void *indices = nullptr; // Don't store indices.

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetReductionWorkspaceSize(
    cudnnHandle, reduce_desc, x_desc, out_desc, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnReduceTensor(
    cudnnHandle, reduce_desc, indices, 0, ws_data, ws_size,
    x690, x_desc, x572, x688, out_desc, x687));
};
float* x693 = (float*)myGpuMalloc(x686 * sizeof(float));
float* x694 = (float*)NULL;
float* x695 = (float*)NULL;
float* x696 = (float*)NULL;
int32_t x699 = x685 * 2048;
float* x700 = (float*)myGpuMalloc(x699 * sizeof(float));
float* x701 = (float*)NULL;
int32_t x702 = 0;

{
size_t dropoutStateSize;
CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize));
void* dropoutStates = myGpuMalloc(dropoutStateSize);

cudnnDropoutDescriptor_t dropout_desc;
CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
CUDNN_CALL(cudnnSetDropoutDescriptor(
    dropout_desc, cudnnHandle, 0.0, dropoutStates, dropoutStateSize, time(NULL)));

cudnnRNNDescriptor_t rnn_desc;
CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
CUDNN_CALL(cudnnSetRNNDescriptor(
    cudnnHandle, rnn_desc,
    /*hiddenSize*/ 1024, /*numLayers*/ 1,
    dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
    CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH));
int32_t seqLength = x669;
int32_t batchSize = x674;
int32_t inputSize = x679;

cudnnTensorDescriptor_t x_descs[seqLength];
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
int x_dims[] = {batchSize, inputSize, 1};
int x_strides[] = {x_dims[1] * x_dims[2], x_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    x_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims, x_strides));
for (int i = 0; i < seqLength; i++) {
  x_descs[i] = x_desc;
}

// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
// The second dimension must match the first dimension of the tensors described in xDesc.
// The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
cudnnTensorDescriptor_t hx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc));
int hx_dims[] = {2, batchSize, 1024};
int hx_strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    hx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims, hx_strides));

cudnnTensorDescriptor_t cx_desc = hx_desc;

size_t paramsSize;
CUDNN_CALL(cudnnGetRNNParamsSize(
    cudnnHandle, rnn_desc, x_descs[0], &paramsSize, CUDNN_DATA_FLOAT));
assert(paramsSize / sizeof(float) == 4198400 && "Expected parameter size mismatch");

cudnnFilterDescriptor_t w_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
int w_dims[] = {int(paramsSize / sizeof(float)), 1, 1};
CUDNN_CALL(cudnnSetFilterNdDescriptor(
    w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims));

cudnnTensorDescriptor_t y_descs[seqLength];
cudnnTensorDescriptor_t y_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
int y_dims[] = {batchSize, 2048, 1};
int y_strides[] = {y_dims[1] * y_dims[2], y_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    y_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims, y_strides));
for (int i = 0; i < seqLength; i++) {
  y_descs[i] = y_desc;
}

cudnnTensorDescriptor_t hy_desc = hx_desc;
cudnnTensorDescriptor_t cy_desc = cx_desc;

size_t workspaceSize;
CUDNN_CALL(cudnnGetRNNWorkspaceSize(
    cudnnHandle, rnn_desc, seqLength, x_descs, &workspaceSize));
void* workspace = myGpuMalloc(workspaceSize);

// Reserve space used by `ForwardTraining` function.
size_t reserveSize;
CUDNN_CALL(cudnnGetRNNTrainingReserveSize(
    cudnnHandle, rnn_desc, seqLength, x_descs, &reserveSize));
void* reserveSpace = myGpuMalloc(reserveSize);
x701 = (float*)reserveSpace;
x702 = (int)reserveSize;
CUDNN_CALL(cudnnRNNForwardTraining(
    cudnnHandle, rnn_desc, seqLength, x_descs, x687,
    hx_desc,x694, cx_desc,x695, w_desc, x125, y_descs, x700,
    hy_desc,x696, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
};
float* x704 = (float*)myGpuMalloc(x699 * sizeof(float));
int32_t x705 = 0;
int32_t x706 = 1;
bool x707 = x669 < 0;
if (x707) {
x705 += 1;
} else {
x706 *= x669;
}
bool x713 = x674 < 0;
if (x713) {
x705 += 1;
} else {
x706 *= x674;
}
x706 *= 2;
if (x462) {
x705 += 1;
} else {
x706 *= x446;
}
int32_t x725 = x705;
bool x726 = x725 >= 2;
if (x726) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x731 = x725 == 0;
int32_t x697 = x674 * 2048;
int32_t x698 = x669 * x697;
if (x731) {
int32_t x732 = x706;
bool x733 = x732 == x698;
if (x733) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x740 = x669 > 0;
int32_t x744;
if (x740) {
x744 = x669;
} else {
int32_t x741 = x706;
int32_t x742 = x698 / x741;
x744 = x742;
}
bool x745 = x674 > 0;
int32_t x749;
if (x745) {
x749 = x674;
} else {
int32_t x746 = x706;
int32_t x747 = x698 / x746;
x749 = x747;
}
int32_t x753;
if (x493) {
x753 = x446;
} else {
int32_t x750 = x706;
int32_t x751 = x698 / x750;
x753 = x751;
}
int32_t x757 = 0;
int32_t x758 = 1;
bool x759 = x744 < 0;
if (x759) {
x757 += 1;
} else {
x758 *= x744;
}
bool x765 = x749 < 0;
if (x765) {
x757 += 1;
} else {
x758 *= x749;
}
x758 *= 2;
bool x772 = x753 < 0;
if (x772) {
x757 += 1;
} else {
x758 *= x753;
}
int32_t x778 = x757;
bool x779 = x778 >= 2;
if (x779) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x784 = x778 == 0;
int32_t x754 = 2 * x753;
int32_t x755 = x749 * x754;
int32_t x756 = x744 * x755;
if (x784) {
int32_t x785 = x758;
bool x786 = x785 == x756;
if (x786) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x793 = x744 > 0;
int32_t x797;
if (x793) {
x797 = x744;
} else {
int32_t x794 = x758;
int32_t x795 = x756 / x794;
x797 = x795;
}
bool x798 = x749 > 0;
int32_t x802;
if (x798) {
x802 = x749;
} else {
int32_t x799 = x758;
int32_t x800 = x756 / x799;
x802 = x800;
}
bool x803 = x753 > 0;
int32_t x807;
if (x803) {
x807 = x753;
} else {
int32_t x804 = x758;
int32_t x805 = x756 / x804;
x807 = x805;
}
int32_t x813 = x797 * x802;
int32_t x814 = x813 * x807;
float* x815 = (float*)myGpuMalloc(x814 * sizeof(float));
float* x816 = (float*)myMalloc(1 * sizeof(float));;
x816[0] = 0.0f;
float* x818 = (float*)myMalloc(1 * sizeof(float));;
x818[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x797, x802, 2, x807));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x797, x802, 1, x807));

cudnnReduceTensorDescriptor_t reduce_desc;
CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&reduce_desc));
CUDNN_CALL(cudnnSetReduceTensorDescriptor(
    reduce_desc, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
    CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));

void *indices = nullptr; // Don't store indices.

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetReductionWorkspaceSize(
    cudnnHandle, reduce_desc, x_desc, out_desc, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnReduceTensor(
    cudnnHandle, reduce_desc, indices, 0, ws_data, ws_size,
    x818, x_desc, x700, x816, out_desc, x815));
};
float* x821 = (float*)myGpuMalloc(x814 * sizeof(float));
// after RNN layers
// after maybe lookahead
int32_t x824 = 0;
int32_t x825 = 1;
bool x826 = x813 < 0;
if (x826) {
x824 += 1;
} else {
x825 *= x813;
}
bool x832 = x807 < 0;
if (x832) {
x824 += 1;
} else {
x825 *= x807;
}
int32_t x838 = x824;
bool x839 = x838 >= 2;
if (x839) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x844 = x838 == 0;
int32_t x811 = x802 * x807;
int32_t x812 = x797 * x811;
if (x844) {
int32_t x845 = x825;
bool x846 = x845 == x812;
if (x846) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x853 = x813 > 0;
int32_t x857;
if (x853) {
x857 = x813;
} else {
int32_t x854 = x825;
int32_t x855 = x812 / x854;
x857 = x855;
}
bool x858 = x807 > 0;
int32_t x862;
if (x858) {
x862 = x807;
} else {
int32_t x859 = x825;
int32_t x860 = x812 / x859;
x862 = x860;
}
bool x864 = x862 == 1024;
if (x864) {
} else {
assert(false && "BatchNorm1D input should be rank2, with shape 1 same as dimSize, got %d : %d");
}
bool x869 = 1024 == x862;
if (x869) {
} else {
assert(false && "scale should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(857) x Sym(862)");
}
if (x869) {
} else {
assert(false && "bias should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(857) x Sym(862)");
}
if (x869) {
} else {
assert(false && "runningMean should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(857) x Sym(862)");
}
if (x869) {
} else {
assert(false && "runningVar should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(857) x Sym(862)");
}
int32_t x863 = x857 * x862;
float* x883 = (float*)myGpuMalloc(x863 * sizeof(float));
float* x884 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x885 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x886 = (float*)myMalloc(1 * sizeof(float));;
x886[0] = 0.0f;
float* x888 = (float*)myMalloc(1 * sizeof(float));;
x888[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x857, x862, 1, 1));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,
    x888, x886, in_desc, x815, in_desc, x883, sbmv_desc, x160,
    x163, 0.1, x165, x166, 1.0E-5,
    x884, x885));
};
float* x891 = (float*)myGpuMalloc(x863 * sizeof(float));
int32_t x892 = x857 * 29;
float* x893 = (float*)myGpuMalloc(x892 * sizeof(float));
float* x894 = (float*)myMalloc(1 * sizeof(float));;
x894[0] = 0.0f;
float* x896 = (float*)myMalloc(1 * sizeof(float));;
x896[0] = 1.0f;
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 29,x857,1024,x896,x177,29,x883,1024,x894,x893,29));
float* x899 = (float*)myGpuMalloc(x892 * sizeof(float));
int32_t x900 = 0;
int32_t x901 = 1;
bool x902 = x797 < 0;
if (x902) {
x900 += 1;
} else {
x901 *= x797;
}
bool x908 = x802 < 0;
if (x908) {
x900 += 1;
} else {
x901 *= x802;
}
x901 *= 29;
int32_t x915 = x900;
bool x916 = x915 >= 2;
if (x916) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x921 = x915 == 0;
if (x921) {
int32_t x922 = x901;
bool x923 = x922 == x892;
if (x923) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x930 = x797 > 0;
int32_t x934;
if (x930) {
x934 = x797;
} else {
int32_t x931 = x901;
int32_t x932 = x892 / x931;
x934 = x932;
}
bool x935 = x802 > 0;
int32_t x939;
if (x935) {
x939 = x802;
} else {
int32_t x936 = x901;
int32_t x937 = x892 / x936;
x939 = x937;
}
int32_t x943 = 0;
int32_t x944 = 1;
int32_t x942 = x934 * x939;
bool x945 = x942 < 0;
if (x945) {
x943 += 1;
} else {
x944 *= x942;
}
x944 *= 29;
x944 *= 1;
x944 *= 1;
int32_t x954 = x943;
bool x955 = x954 >= 2;
if (x955) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x960 = x954 == 0;
int32_t x940 = x939 * 29;
int32_t x941 = x934 * x940;
if (x960) {
int32_t x961 = x944;
bool x962 = x961 == x941;
if (x962) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x969 = x942 > 0;
int32_t x973;
if (x969) {
x973 = x942;
} else {
int32_t x970 = x944;
int32_t x971 = x941 / x970;
x973 = x971;
}
float* x975 = (float*)myMalloc(1 * sizeof(float));;
x975[0] = 0.0f;
float* x977 = (float*)myMalloc(1 * sizeof(float));;
x977[0] = 1.0f;
int32_t x974 = x973 * 29;
float* x979 = (float*)myGpuMalloc(x974 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x973, 29, 1, 1));
CUDNN_CALL(cudnnSoftmaxForward(
    cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
    x977, x_desc, x893, x975, x_desc, x979));
};
int32_t x981 = 0;
int32_t x982 = 1;
bool x983 = x934 < 0;
if (x983) {
x981 += 1;
} else {
x982 *= x934;
}
bool x989 = x939 < 0;
if (x989) {
x981 += 1;
} else {
x982 *= x939;
}
x982 *= 29;
int32_t x996 = x981;
bool x997 = x996 >= 2;
if (x997) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1002 = x996 == 0;
if (x1002) {
int32_t x1003 = x982;
bool x1004 = x1003 == x974;
if (x1004) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x1011 = x934 > 0;
int32_t x1015;
if (x1011) {
x1015 = x934;
} else {
int32_t x1012 = x982;
int32_t x1013 = x974 / x1012;
x1015 = x1013;
}
bool x1016 = x939 > 0;
int32_t x1020;
if (x1016) {
x1020 = x939;
} else {
int32_t x1017 = x982;
int32_t x1018 = x974 / x1017;
x1020 = x1018;
}
int32_t x1023 = x1015 * x1020;
int32_t x1024 = x1023 * 29;
float* x1025 = (float*)myGpuMalloc(x1024 * sizeof(float));
// before CTC loss
int* x1027 = (int32_t*)myMalloc(x1020 * sizeof(int32_t));;
float x1031 = (float)x1015;
for(int x1029=0; x1029 < x1020; x1029++) {
float x1030 = x276[x1029];
float x1032 = x1030 * x1031;
int32_t x1033 = (int)x1032;
x1027[x1029] = x1033;

}
bool x1037 = x1020 <= 256;
if (x1037) {
} else {
printf("'cudnnGetCTCLossWorkspaceSize' requires batch size less than 256, got %d\n\n",x1020);
assert(false && "");
}
float* x1043 = (float*)myGpuMalloc(x1020 * sizeof(float));

{
cudnnTensorDescriptor_t probs_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&probs_desc));
int probs_dims[] = {x1015, x1020, 29};
int probs_strides[] = {probs_dims[1] * probs_dims[2], probs_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    probs_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, probs_dims, probs_strides));

cudnnTensorDescriptor_t grad_desc = probs_desc;

cudnnCTCLossDescriptor_t ctc_desc;
CUDNN_CALL(cudnnCreateCTCLossDescriptor(&ctc_desc));
CUDNN_CALL(cudnnSetCTCLossDescriptor(ctc_desc, CUDNN_DATA_FLOAT));
size_t wsSize;
CUDNN_CALL(cudnnGetCTCLossWorkspaceSize(
    cudnnHandle, probs_desc, grad_desc, x277, x278, x1027,
    CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, &wsSize));
void *ws = myGpuMalloc(wsSize);

CUDNN_CALL(cudnnCTCLoss(
    cudnnHandle, probs_desc, x979, x277, x278, x1027,
    x1043, grad_desc, x1025, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, ws, wsSize));
};
float* x1045 = (float*)myGpuMalloc(1 * sizeof(float));
float* x1046 = (float*)myMalloc(1 * sizeof(float));;
x1046[0] = 0.0f;
float* x1048 = (float*)myMalloc(1 * sizeof(float));;
x1048[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x1020, 1, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1, 1, 1));

cudnnReduceTensorDescriptor_t reduce_desc;
CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&reduce_desc));
CUDNN_CALL(cudnnSetReduceTensorDescriptor(
    reduce_desc, CUDNN_REDUCE_TENSOR_AVG, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
    CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));

void *indices = nullptr; // Don't store indices.

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetReductionWorkspaceSize(
    cudnnHandle, reduce_desc, x_desc, out_desc, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnReduceTensor(
    cudnnHandle, reduce_desc, indices, 0, ws_data, ws_size,
    x1048, x_desc, x1043, x1046, out_desc, x1045));
};
// after CTC loss
float* x1052 = (float*)myGpuMalloc(1 * sizeof(float));
// make sure the size of loss is 1
arrayFill<<<28, 512>>>(x1052, 1.0f, 1);
// backend is lantern.TensorDslCudnn$BackendCudnn@105e9b0f
CUDA_CALL(cudaMemcpy(x287, x1045, 1 * sizeof(float), cudaMemcpyDeviceToDevice));
int32_t x1057 = 0;
int32_t x1058 = 1;
if (x945) {
x1057 += 1;
} else {
x1058 *= x942;
}
x1058 *= 29;
x1058 *= 1;
x1058 *= 1;
int32_t x1067 = x1057;
bool x1068 = x1067 >= 2;
if (x1068) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1073 = x1067 == 0;
if (x1073) {
int32_t x1074 = x1058;
bool x1075 = x1074 == x941;
if (x1075) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1085;
if (x969) {
x1085 = x942;
} else {
int32_t x1082 = x1058;
int32_t x1083 = x941 / x1082;
x1085 = x1083;
}
int32_t x1087 = 0;
int32_t x1088 = 1;
if (x945) {
x1087 += 1;
} else {
x1088 *= x942;
}
x1088 *= 29;
x1088 *= 1;
x1088 *= 1;
int32_t x1097 = x1087;
bool x1098 = x1097 >= 2;
if (x1098) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1103 = x1097 == 0;
if (x1103) {
int32_t x1104 = x1088;
bool x1105 = x1104 == x941;
if (x1105) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1115;
if (x969) {
x1115 = x942;
} else {
int32_t x1112 = x1088;
int32_t x1113 = x941 / x1112;
x1115 = x1113;
}
int32_t x1117 = 0;
int32_t x1118 = 1;
bool x1119 = x1023 < 0;
if (x1119) {
x1117 += 1;
} else {
x1118 *= x1023;
}
x1118 *= 29;
x1118 *= 1;
x1118 *= 1;
int32_t x1128 = x1117;
bool x1129 = x1128 >= 2;
if (x1129) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1134 = x1128 == 0;
if (x1134) {
int32_t x1135 = x1118;
int32_t x1021 = x1020 * 29;
int32_t x1022 = x1015 * x1021;
bool x1136 = x1135 == x1022;
if (x1136) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x1143 = x1023 > 0;
int32_t x1147;
if (x1143) {
x1147 = x1023;
} else {
int32_t x1144 = x1118;
int32_t x1021 = x1020 * 29;
int32_t x1022 = x1015 * x1021;
int32_t x1145 = x1022 / x1144;
x1147 = x1145;
}
int32_t x1149 = 0;
int32_t x1150 = 1;
if (x1119) {
x1149 += 1;
} else {
x1150 *= x1023;
}
x1150 *= 29;
x1150 *= 1;
x1150 *= 1;
int32_t x1159 = x1149;
bool x1160 = x1159 >= 2;
if (x1160) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1165 = x1159 == 0;
if (x1165) {
int32_t x1166 = x1150;
int32_t x1021 = x1020 * 29;
int32_t x1022 = x1015 * x1021;
bool x1167 = x1166 == x1022;
if (x1167) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1177;
if (x1143) {
x1177 = x1023;
} else {
int32_t x1174 = x1150;
int32_t x1021 = x1020 * 29;
int32_t x1022 = x1015 * x1021;
int32_t x1175 = x1022 / x1174;
x1177 = x1175;
}
bool x1179 = x1085 == x1147;
bool x1180;
if (x1179) {
x1180 = true;
} else {
x1180 = false;
}
bool x1181;
if (x1180) {
x1181 = true;
} else {
x1181 = false;
}
bool x1182;
if (x1181) {
x1182 = true;
} else {
x1182 = false;
}
if (x1182) {
} else {
printf("$errorPrefix: tensor shapes are not equal %s, %s\n\n"," x Sym(1085) x Const(29) x Const(1) x Const(1)"," x Sym(1147) x Const(29) x Const(1) x Const(1)");
assert(false && "");
}
bool x1188 = x1115 == x1177;
bool x1189;
if (x1188) {
x1189 = true;
} else {
x1189 = false;
}
bool x1190;
if (x1189) {
x1190 = true;
} else {
x1190 = false;
}
bool x1191;
if (x1190) {
x1191 = true;
} else {
x1191 = false;
}
if (x1191) {
} else {
printf("$errorPrefix: tensor shapes are not equal %s, %s\n\n"," x Sym(1115) x Const(29) x Const(1) x Const(1)"," x Sym(1177) x Const(29) x Const(1) x Const(1)");
assert(false && "");
}
float* x1197 = (float*)myMalloc(1 * sizeof(float));;
x1197[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x1085, 29, 1, 1));
CUDNN_CALL(cudnnSoftmaxBackward(
    cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
    x1197, x_desc, x979, x_desc, x1025,
    x1197, x_desc, x899));
};
float* x1200 = (float*)myMalloc(1 * sizeof(float));;
x1200[0] = 0.0f;
float* x1202 = (float*)myMalloc(1 * sizeof(float));;
x1202[0] = 1.0f;
// backprop of matrix-matrix-dot
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, x862,x857,29,x1202,x177,29,x899,29,x1202,x891,x862));
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 29,x862,x857,x1202,x899,29,x883,x862,x1202,x179,29));
float* x1207 = (float*)myMalloc(1 * sizeof(float));;
x1207[0] = 0.0f;
float* x1209 = (float*)myMalloc(1 * sizeof(float));;
x1209[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x857, x862, 1, 1));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,
    x1209, x1209, x1209, x1209, in_desc, x815,
    in_desc, x891, in_desc, x821, sbmv_desc, x160,
    x162,x164, 1.0E-5, x884, x885));
};
// backprop for sum on dim op
sum_grad<<<28, 512>>>(x704, x744, x749, 2, x753, x756, x821, x811, x807, 1, 2);
;
float* x1214 = (float*)NULL;
float* x1215 = (float*)NULL;

{
size_t dropoutStateSize;
CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize));
void* dropoutStates = myGpuMalloc(dropoutStateSize);

cudnnDropoutDescriptor_t dropout_desc;
CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
CUDNN_CALL(cudnnSetDropoutDescriptor(
    dropout_desc, cudnnHandle, 0.0, dropoutStates, dropoutStateSize, time(NULL)));

cudnnRNNDescriptor_t rnn_desc;
CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
CUDNN_CALL(cudnnSetRNNDescriptor(
    cudnnHandle, rnn_desc,
    /*hiddenSize*/ 1024, /*numLayers*/ 1,
    dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
    CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH));
int32_t seqLength = x669;
int32_t batchSize = x674;
int32_t inputSize = x679;

cudnnTensorDescriptor_t dx_descs[seqLength];
cudnnTensorDescriptor_t dx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&dx_desc));
int x_dims[] = {batchSize, inputSize, 1};
int x_strides[] = {x_dims[1] * x_dims[2], x_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    dx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims, x_strides));
for (int i = 0; i < seqLength; i++) {
  dx_descs[i] = dx_desc;
}

// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
// The second dimension must match the first dimension of the tensors described in xDesc.
// The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
cudnnTensorDescriptor_t hx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc));
int hx_dims[] = {2, batchSize, 1024};
int hx_strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    hx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims, hx_strides));

cudnnTensorDescriptor_t cx_desc = hx_desc;

size_t paramsSize;
CUDNN_CALL(cudnnGetRNNParamsSize(
    cudnnHandle, rnn_desc, dx_descs[0], &paramsSize, CUDNN_DATA_FLOAT));
assert(paramsSize / sizeof(float) == 4198400 && "Expected parameter size mismatch");

cudnnFilterDescriptor_t w_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
int w_dims[] = {int(paramsSize / sizeof(float)), 1, 1};
CUDNN_CALL(cudnnSetFilterNdDescriptor(
    w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims));

cudnnTensorDescriptor_t y_descs[seqLength];
cudnnTensorDescriptor_t y_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
int y_dims[] = {batchSize, 2048, 1};
int y_strides[] = {y_dims[1] * y_dims[2], y_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    y_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims, y_strides));
for (int i = 0; i < seqLength; i++) {
  y_descs[i] = y_desc;
}

cudnnTensorDescriptor_t dhx_desc = hx_desc;
cudnnTensorDescriptor_t hy_desc = hx_desc;
cudnnTensorDescriptor_t dhy_desc = hy_desc;

cudnnTensorDescriptor_t dcx_desc = cx_desc;
cudnnTensorDescriptor_t cy_desc = cx_desc;
cudnnTensorDescriptor_t dcy_desc = cy_desc;

size_t workspaceSize;
CUDNN_CALL(cudnnGetRNNWorkspaceSize(
    cudnnHandle, rnn_desc, seqLength, dx_descs, &workspaceSize));
void* workspace = myGpuMalloc(workspaceSize);
CUDNN_CALL(cudnnRNNBackwardData(
    cudnnHandle, rnn_desc, seqLength, y_descs, x700, y_descs, x704,
    dhy_desc, NULL, dcy_desc, NULL, w_desc, x125, hx_desc, x1214,
    cx_desc, x1215, dx_descs, x693, dhx_desc, NULL, dcx_desc, NULL,
    workspace, workspaceSize, x701, x702));
};
float* x1217 = (float*)NULL;

{
size_t dropoutStateSize;
CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize));
void* dropoutStates = myGpuMalloc(dropoutStateSize);

cudnnDropoutDescriptor_t dropout_desc;
CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
CUDNN_CALL(cudnnSetDropoutDescriptor(
    dropout_desc, cudnnHandle, 0.0, dropoutStates, dropoutStateSize, time(NULL)));

cudnnRNNDescriptor_t rnn_desc;
CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
CUDNN_CALL(cudnnSetRNNDescriptor(
    cudnnHandle, rnn_desc,
    /*hiddenSize*/ 1024, /*numLayers*/ 1,
    dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
    CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH));
int32_t seqLength = x669;
int32_t batchSize = x674;
int32_t inputSize = x679;

cudnnTensorDescriptor_t x_descs[seqLength];
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
int x_dims[] = {batchSize, inputSize, 1};
int x_strides[] = {x_dims[1] * x_dims[2], x_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    x_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims, x_strides));
for (int i = 0; i < seqLength; i++) {
  x_descs[i] = x_desc;
}

// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
// The second dimension must match the first dimension of the tensors described in xDesc.
// The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
cudnnTensorDescriptor_t hx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc));
int hx_dims[] = {2, batchSize, 1024};
int hx_strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    hx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims, hx_strides));

size_t paramsSize;
CUDNN_CALL(cudnnGetRNNParamsSize(
    cudnnHandle, rnn_desc, x_descs[0], &paramsSize, CUDNN_DATA_FLOAT));
// printf("paramsSize: %zu\n", paramsSize / sizeof(float));
assert(paramsSize / sizeof(float) == 4198400 && "Expected parameter size mismatch");

cudnnFilterDescriptor_t dw_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&dw_desc));
int w_dims[] = {int(paramsSize / sizeof(float)), 1, 1};
CUDNN_CALL(cudnnSetFilterNdDescriptor(
    dw_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims));

cudnnTensorDescriptor_t y_descs[seqLength];
cudnnTensorDescriptor_t y_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
int y_dims[] = {batchSize, 2048, 1};
int y_strides[] = {y_dims[1] * y_dims[2], y_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    y_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims, y_strides));
for (int i = 0; i < seqLength; i++) {
  y_descs[i] = y_desc;
}

size_t workspaceSize;
CUDNN_CALL(cudnnGetRNNWorkspaceSize(
    cudnnHandle, rnn_desc, seqLength, x_descs, &workspaceSize));
void* workspace = myGpuMalloc(workspaceSize);
CUDNN_CALL(cudnnRNNBackwardWeights(
    cudnnHandle, rnn_desc, seqLength, x_descs, x687, hx_desc, x1217,
    y_descs, x700, workspace, workspaceSize,
    dw_desc, x127, x701, x702));
};
// backprop for sum on dim op
int32_t x683 = x674 * x679;
sum_grad<<<28, 512>>>(x576, x616, x621, 2, x625, x628, x693, x683, x679, 1, 2);
;
float* x1221 = (float*)NULL;
float* x1222 = (float*)NULL;

{
size_t dropoutStateSize;
CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize));
void* dropoutStates = myGpuMalloc(dropoutStateSize);

cudnnDropoutDescriptor_t dropout_desc;
CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
CUDNN_CALL(cudnnSetDropoutDescriptor(
    dropout_desc, cudnnHandle, 0.0, dropoutStates, dropoutStateSize, time(NULL)));

cudnnRNNDescriptor_t rnn_desc;
CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
CUDNN_CALL(cudnnSetRNNDescriptor(
    cudnnHandle, rnn_desc,
    /*hiddenSize*/ 1024, /*numLayers*/ 1,
    dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
    CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH));
int32_t seqLength = x541;
int32_t batchSize = x546;
int32_t inputSize = x551;

cudnnTensorDescriptor_t dx_descs[seqLength];
cudnnTensorDescriptor_t dx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&dx_desc));
int x_dims[] = {batchSize, inputSize, 1};
int x_strides[] = {x_dims[1] * x_dims[2], x_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    dx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims, x_strides));
for (int i = 0; i < seqLength; i++) {
  dx_descs[i] = dx_desc;
}

// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
// The second dimension must match the first dimension of the tensors described in xDesc.
// The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
cudnnTensorDescriptor_t hx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc));
int hx_dims[] = {2, batchSize, 1024};
int hx_strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    hx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims, hx_strides));

cudnnTensorDescriptor_t cx_desc = hx_desc;

size_t paramsSize;
CUDNN_CALL(cudnnGetRNNParamsSize(
    cudnnHandle, rnn_desc, dx_descs[0], &paramsSize, CUDNN_DATA_FLOAT));
assert(paramsSize / sizeof(float) == 4198400 && "Expected parameter size mismatch");

cudnnFilterDescriptor_t w_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
int w_dims[] = {int(paramsSize / sizeof(float)), 1, 1};
CUDNN_CALL(cudnnSetFilterNdDescriptor(
    w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims));

cudnnTensorDescriptor_t y_descs[seqLength];
cudnnTensorDescriptor_t y_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
int y_dims[] = {batchSize, 2048, 1};
int y_strides[] = {y_dims[1] * y_dims[2], y_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    y_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims, y_strides));
for (int i = 0; i < seqLength; i++) {
  y_descs[i] = y_desc;
}

cudnnTensorDescriptor_t dhx_desc = hx_desc;
cudnnTensorDescriptor_t hy_desc = hx_desc;
cudnnTensorDescriptor_t dhy_desc = hy_desc;

cudnnTensorDescriptor_t dcx_desc = cx_desc;
cudnnTensorDescriptor_t cy_desc = cx_desc;
cudnnTensorDescriptor_t dcy_desc = cy_desc;

size_t workspaceSize;
CUDNN_CALL(cudnnGetRNNWorkspaceSize(
    cudnnHandle, rnn_desc, seqLength, dx_descs, &workspaceSize));
void* workspace = myGpuMalloc(workspaceSize);
CUDNN_CALL(cudnnRNNBackwardData(
    cudnnHandle, rnn_desc, seqLength, y_descs, x572, y_descs, x576,
    dhy_desc, NULL, dcy_desc, NULL, w_desc, x88, hx_desc, x1221,
    cx_desc, x1222, dx_descs, x565, dhx_desc, NULL, dcx_desc, NULL,
    workspace, workspaceSize, x573, x574));
};
float* x1224 = (float*)NULL;

{
size_t dropoutStateSize;
CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize));
void* dropoutStates = myGpuMalloc(dropoutStateSize);

cudnnDropoutDescriptor_t dropout_desc;
CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
CUDNN_CALL(cudnnSetDropoutDescriptor(
    dropout_desc, cudnnHandle, 0.0, dropoutStates, dropoutStateSize, time(NULL)));

cudnnRNNDescriptor_t rnn_desc;
CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
CUDNN_CALL(cudnnSetRNNDescriptor(
    cudnnHandle, rnn_desc,
    /*hiddenSize*/ 1024, /*numLayers*/ 1,
    dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
    CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH));
int32_t seqLength = x541;
int32_t batchSize = x546;
int32_t inputSize = x551;

cudnnTensorDescriptor_t x_descs[seqLength];
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
int x_dims[] = {batchSize, inputSize, 1};
int x_strides[] = {x_dims[1] * x_dims[2], x_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    x_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims, x_strides));
for (int i = 0; i < seqLength; i++) {
  x_descs[i] = x_desc;
}

// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
// The second dimension must match the first dimension of the tensors described in xDesc.
// The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
cudnnTensorDescriptor_t hx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc));
int hx_dims[] = {2, batchSize, 1024};
int hx_strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    hx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims, hx_strides));

size_t paramsSize;
CUDNN_CALL(cudnnGetRNNParamsSize(
    cudnnHandle, rnn_desc, x_descs[0], &paramsSize, CUDNN_DATA_FLOAT));
// printf("paramsSize: %zu\n", paramsSize / sizeof(float));
assert(paramsSize / sizeof(float) == 4198400 && "Expected parameter size mismatch");

cudnnFilterDescriptor_t dw_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&dw_desc));
int w_dims[] = {int(paramsSize / sizeof(float)), 1, 1};
CUDNN_CALL(cudnnSetFilterNdDescriptor(
    dw_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims));

cudnnTensorDescriptor_t y_descs[seqLength];
cudnnTensorDescriptor_t y_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
int y_dims[] = {batchSize, 2048, 1};
int y_strides[] = {y_dims[1] * y_dims[2], y_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    y_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims, y_strides));
for (int i = 0; i < seqLength; i++) {
  y_descs[i] = y_desc;
}

size_t workspaceSize;
CUDNN_CALL(cudnnGetRNNWorkspaceSize(
    cudnnHandle, rnn_desc, seqLength, x_descs, &workspaceSize));
void* workspace = myGpuMalloc(workspaceSize);
CUDNN_CALL(cudnnRNNBackwardWeights(
    cudnnHandle, rnn_desc, seqLength, x_descs, x559, hx_desc, x1224,
    y_descs, x572, workspace, workspaceSize,
    dw_desc, x90, x573, x574));
};
// backprop for sum on dim op
int32_t x555 = x546 * x551;
sum_grad<<<28, 512>>>(x445, x487, x492, 2, x497, x500, x565, x555, x551, 1, 2);
;
float* x1228 = (float*)NULL;
float* x1229 = (float*)NULL;

{
size_t dropoutStateSize;
CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize));
void* dropoutStates = myGpuMalloc(dropoutStateSize);

cudnnDropoutDescriptor_t dropout_desc;
CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
CUDNN_CALL(cudnnSetDropoutDescriptor(
    dropout_desc, cudnnHandle, 0.0, dropoutStates, dropoutStateSize, time(NULL)));

cudnnRNNDescriptor_t rnn_desc;
CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
CUDNN_CALL(cudnnSetRNNDescriptor(
    cudnnHandle, rnn_desc,
    /*hiddenSize*/ 1024, /*numLayers*/ 1,
    dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
    CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH));
int32_t seqLength = x411;
int32_t batchSize = x401;
int32_t inputSize = x406;

cudnnTensorDescriptor_t dx_descs[seqLength];
cudnnTensorDescriptor_t dx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&dx_desc));
int x_dims[] = {batchSize, inputSize, 1};
int x_strides[] = {x_dims[1] * x_dims[2], x_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    dx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims, x_strides));
for (int i = 0; i < seqLength; i++) {
  dx_descs[i] = dx_desc;
}

// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
// The second dimension must match the first dimension of the tensors described in xDesc.
// The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
cudnnTensorDescriptor_t hx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc));
int hx_dims[] = {2, batchSize, 1024};
int hx_strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    hx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims, hx_strides));

cudnnTensorDescriptor_t cx_desc = hx_desc;

size_t paramsSize;
CUDNN_CALL(cudnnGetRNNParamsSize(
    cudnnHandle, rnn_desc, dx_descs[0], &paramsSize, CUDNN_DATA_FLOAT));
assert(paramsSize / sizeof(float) == 3477504 && "Expected parameter size mismatch");

cudnnFilterDescriptor_t w_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
int w_dims[] = {int(paramsSize / sizeof(float)), 1, 1};
CUDNN_CALL(cudnnSetFilterNdDescriptor(
    w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims));

cudnnTensorDescriptor_t y_descs[seqLength];
cudnnTensorDescriptor_t y_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
int y_dims[] = {batchSize, 2048, 1};
int y_strides[] = {y_dims[1] * y_dims[2], y_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    y_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims, y_strides));
for (int i = 0; i < seqLength; i++) {
  y_descs[i] = y_desc;
}

cudnnTensorDescriptor_t dhx_desc = hx_desc;
cudnnTensorDescriptor_t hy_desc = hx_desc;
cudnnTensorDescriptor_t dhy_desc = hy_desc;

cudnnTensorDescriptor_t dcx_desc = cx_desc;
cudnnTensorDescriptor_t cy_desc = cx_desc;
cudnnTensorDescriptor_t dcy_desc = cy_desc;

size_t workspaceSize;
CUDNN_CALL(cudnnGetRNNWorkspaceSize(
    cudnnHandle, rnn_desc, seqLength, dx_descs, &workspaceSize));
void* workspace = myGpuMalloc(workspaceSize);
CUDNN_CALL(cudnnRNNBackwardData(
    cudnnHandle, rnn_desc, seqLength, y_descs, x441, y_descs, x445,
    dhy_desc, NULL, dcy_desc, NULL, w_desc, x51, hx_desc, x1228,
    cx_desc, x1229, dx_descs, x433, dhx_desc, NULL, dcx_desc, NULL,
    workspace, workspaceSize, x442, x443));
};
float* x1231 = (float*)NULL;

{
size_t dropoutStateSize;
CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize));
void* dropoutStates = myGpuMalloc(dropoutStateSize);

cudnnDropoutDescriptor_t dropout_desc;
CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
CUDNN_CALL(cudnnSetDropoutDescriptor(
    dropout_desc, cudnnHandle, 0.0, dropoutStates, dropoutStateSize, time(NULL)));

cudnnRNNDescriptor_t rnn_desc;
CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
CUDNN_CALL(cudnnSetRNNDescriptor(
    cudnnHandle, rnn_desc,
    /*hiddenSize*/ 1024, /*numLayers*/ 1,
    dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
    CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH));
int32_t seqLength = x411;
int32_t batchSize = x401;
int32_t inputSize = x406;

cudnnTensorDescriptor_t x_descs[seqLength];
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
int x_dims[] = {batchSize, inputSize, 1};
int x_strides[] = {x_dims[1] * x_dims[2], x_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    x_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims, x_strides));
for (int i = 0; i < seqLength; i++) {
  x_descs[i] = x_desc;
}

// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
// The second dimension must match the first dimension of the tensors described in xDesc.
// The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
cudnnTensorDescriptor_t hx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc));
int hx_dims[] = {2, batchSize, 1024};
int hx_strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    hx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims, hx_strides));

size_t paramsSize;
CUDNN_CALL(cudnnGetRNNParamsSize(
    cudnnHandle, rnn_desc, x_descs[0], &paramsSize, CUDNN_DATA_FLOAT));
// printf("paramsSize: %zu\n", paramsSize / sizeof(float));
assert(paramsSize / sizeof(float) == 3477504 && "Expected parameter size mismatch");

cudnnFilterDescriptor_t dw_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&dw_desc));
int w_dims[] = {int(paramsSize / sizeof(float)), 1, 1};
CUDNN_CALL(cudnnSetFilterNdDescriptor(
    dw_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims));

cudnnTensorDescriptor_t y_descs[seqLength];
cudnnTensorDescriptor_t y_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
int y_dims[] = {batchSize, 2048, 1};
int y_strides[] = {y_dims[1] * y_dims[2], y_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    y_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims, y_strides));
for (int i = 0; i < seqLength; i++) {
  y_descs[i] = y_desc;
}

size_t workspaceSize;
CUDNN_CALL(cudnnGetRNNWorkspaceSize(
    cudnnHandle, rnn_desc, seqLength, x_descs, &workspaceSize));
void* workspace = myGpuMalloc(workspaceSize);
CUDNN_CALL(cudnnRNNBackwardWeights(
    cudnnHandle, rnn_desc, seqLength, x_descs, x414, hx_desc, x1231,
    y_descs, x441, workspace, workspaceSize,
    dw_desc, x53, x442, x443));
};
// backprop for permute WrappedArray(2, 0, 1)
int* x1234 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x1234[2] = x415;
x1234[0] = x406;
x1234[1] = 1;
x1234[3] = 1;
float* x1239 = (float*)myMalloc(1 * sizeof(float));;
x1239[0] = 1.0f;
int32_t x1241 = x1234[0];
int32_t x1242 = x1234[1];
int32_t x1243 = x1234[2];
int32_t x1244 = x1234[3];

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
    in_desc, CUDNN_DATA_FLOAT,
    x401, x406, x411, 1,
    x1241, x1242, x1243, x1244));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
    out_desc, CUDNN_DATA_FLOAT,
    x401, x406, x411, 1,
    x412, x411, 1, 1));

CUDNN_CALL(cudnnTransformTensor(
    cudnnHandle, x1239, in_desc, x433, x1239, out_desc, x357));
};
hardTanh_grad<<<28, 512>>>(x349, x357, x357, 0.0, 20.0, x339, true);
float* x1247 = (float*)myMalloc(1 * sizeof(float));;
x1247[0] = 0.0f;
float* x1249 = (float*)myMalloc(1 * sizeof(float));;
x1249[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x190, 32, x333, x336));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x190, 32, x333, x336));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 32, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1249, x1249, x1249, x1249, in_desc, x342,
    out_desc, x357, in_desc, x348, sbmv_desc, x41,
    x43,x45, 1.0E-5, x350, x351));
};
// conv2D back-propagate
float* x1253 = (float*)myMalloc(1 * sizeof(float));;
x1253[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    32, 32, 21, 11));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x190, 32, x297, x300));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x190, 32, x333, x336));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 2, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
// Algorithm.
cudnnConvolutionBwdDataAlgo_t algo;
CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(
    cudnnHandle,
    filt_desc, grad_out_desc, conv_desc, grad_in_desc,
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algo));
// algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1253, filt_desc, x38, grad_out_desc, x348,
    conv_desc, algo, ws_data, ws_size,
    x1253, grad_in_desc, x322));
};
float* x1256 = (float*)myMalloc(1 * sizeof(float));;
x1256[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    32, 32, 21, 11));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x190, 32, x333, x336));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x190, 32, x297, x300));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 2, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
// Algorithm.
cudnnConvolutionBwdFilterAlgo_t algo;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle,
    in_desc, grad_out_desc, conv_desc, grad_filt_desc,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algo));
algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1256, in_desc, x314, grad_out_desc, x348,
    conv_desc, algo, ws_data, ws_size,
    x1256, grad_filt_desc, x40));
};
hardTanh_grad<<<28, 512>>>(x314, x322, x322, 0.0, 20.0, x303, true);
float* x1260 = (float*)myMalloc(1 * sizeof(float));;
x1260[0] = 0.0f;
float* x1262 = (float*)myMalloc(1 * sizeof(float));;
x1262[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x190, 32, x297, x300));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x190, 32, x297, x300));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 32, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1262, x1262, x1262, x1262, in_desc, x307,
    out_desc, x322, in_desc, x313, sbmv_desc, x21,
    x23,x25, 1.0E-5, x315, x316));
};
// conv2D back-propagate
float* x1266 = (float*)myMalloc(1 * sizeof(float));;
x1266[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    32, 1, 41, 11));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x190, 32, x297, x300));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x190, 1, x272, x271));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 2, 2, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
// Algorithm.
cudnnConvolutionBwdFilterAlgo_t algo;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle,
    in_desc, grad_out_desc, conv_desc, grad_filt_desc,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algo));
algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1266, in_desc, x281, grad_out_desc, x313,
    conv_desc, algo, ws_data, ws_size,
    x1266, grad_filt_desc, x20));
};
// Tensor 'toCPU' invocation.
float* x1270 = (float*)myMalloc(1 * sizeof(float));;
CUDA_CALL(cudaMemcpy(x1270, x287, 1 * sizeof(float), cudaMemcpyDeviceToHost));
float x1272 = x1270[0];
x266 += x1272;
float* x1274 = (float*)myMalloc(1 * sizeof(float));;
x1274[0] = 1.0f;
float* x1276 = (float*)myMalloc(1 * sizeof(float));;
x1276[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 451,32,x1274,x18,451,x1276, x20, 451, x18,451));
arrayFill<<<28, 512>>>(x20, 0.0f, 14432);
float* x1280 = (float*)myMalloc(1 * sizeof(float));;
x1280[0] = 1.0f;
float* x1282 = (float*)myMalloc(1 * sizeof(float));;
x1282[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 7392,32,x1280,x38,7392,x1282, x40, 7392, x38,7392));
arrayFill<<<28, 512>>>(x40, 0.0f, 236544);
float* x1286 = (float*)myMalloc(1 * sizeof(float));;
x1286[0] = 1.0f;
float* x1288 = (float*)myMalloc(1 * sizeof(float));;
x1288[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1286,x41,1,x1288, x43, 1, x41,1));
arrayFill<<<28, 512>>>(x43, 0.0f, 32);
float* x1292 = (float*)myMalloc(1 * sizeof(float));;
x1292[0] = 1.0f;
float* x1294 = (float*)myMalloc(1 * sizeof(float));;
x1294[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1292,x44,1,x1294, x45, 1, x44,1));
arrayFill<<<28, 512>>>(x45, 0.0f, 32);
float* x1298 = (float*)myMalloc(1 * sizeof(float));;
x1298[0] = 1.0f;
float* x1300 = (float*)myMalloc(1 * sizeof(float));;
x1300[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1298,x24,1,x1300, x25, 1, x24,1));
arrayFill<<<28, 512>>>(x25, 0.0f, 32);
float* x1304 = (float*)myMalloc(1 * sizeof(float));;
x1304[0] = 1.0f;
float* x1306 = (float*)myMalloc(1 * sizeof(float));;
x1306[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1304,x21,1,x1306, x23, 1, x21,1));
arrayFill<<<28, 512>>>(x23, 0.0f, 32);
float* x1310 = (float*)myMalloc(1 * sizeof(float));;
x1310[0] = 1.0f;
float* x1312 = (float*)myMalloc(1 * sizeof(float));;
x1312[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x1310,x160,1,x1312, x162, 1, x160,1));
arrayFill<<<28, 512>>>(x162, 0.0f, 1024);
float* x1316 = (float*)myMalloc(1 * sizeof(float));;
x1316[0] = 1.0f;
float* x1318 = (float*)myMalloc(1 * sizeof(float));;
x1318[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x1316,x163,1,x1318, x164, 1, x163,1));
arrayFill<<<28, 512>>>(x164, 0.0f, 1024);
float* x1322 = (float*)myMalloc(1 * sizeof(float));;
x1322[0] = 1.0f;
float* x1324 = (float*)myMalloc(1 * sizeof(float));;
x1324[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 29,1024,x1322,x177,29,x1324, x179, 29, x177,29));
arrayFill<<<28, 512>>>(x179, 0.0f, 29696);
int32_t x1328 = x263;
int32_t x1330 = x1328 % x1329;
bool x1331 = x1330 == 0;
if (x1331) {
float x1336 = x266;
double x1332 = (double)x1328;
double x1333 = 100.0 * x1332;
double x1335 = x1333 / x1334;
float x1337 = (float)x1328;
float x1338 = x1336 / x1337;
printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x259,x1328,x195,x1335,x1338);
fflush(stdout);
} else {
}
int64_t x1343 = (long)mallocAddr;
int64_t x1344 = x1343 - x255;
memset((void*)x255, 0, x1344);
mallocAddr = (void*)x255;
int64_t x1347 = (long)gpuMallocAddr;
int64_t x1348 = x1347 - x256;
cudaMemset((void*)x256, 0, x1348);
gpuMallocAddr = (void*)x256;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x1355 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x1356 = x1355 / 1000LL;
int64_t x1358 = x1355 / x1357;
printf("Training completed in %ldms (%ld us/images)\n",x1356,x1358);
double x1360 = (double)x1355;
double x1361 = x1360 / 1000000.0;
x254[x259] = x1361;
float x1363 = x266;
float x1365 = x1363 / x1364;
double x1366 = (double)x1365;
x253[x259] = x1366;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x1372 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
sort(x254, x254 + 1);
double x1378 = x254[0];
int64_t x1379 = (long)fopen(x0, "w");
fprintf((FILE *)x1379, "unit: %s\n", "1 epoch");
for(int x1381=0; x1381 < 1; x1381++) {
double x1382 = x253[x1381];
fprintf((FILE *)x1379, "%lf\n", x1382);

}
fprintf((FILE *)x1379, "run time: %lf %lf\n", x251, x1378);
fclose((FILE*)x1379);
// Backend cleanup.
CUBLAS_CALL(cublasDestroy(cublasHandle));
CUDA_CALL(cudaFree(gpuMallocBase));
      
CUDNN_CALL(cudnnDestroy(cudnnHandle));
}
/*****************************************
  End of C Generated Code                  
*******************************************/


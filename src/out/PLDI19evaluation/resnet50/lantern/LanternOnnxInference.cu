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
// Tensor 'toGPU' invocation.
float* x276 = (float*)myGpuMalloc(262144 * sizeof(float));
int32_t x5 = open("/home/fei/bitbucket/Lantern/src/out/PLDI19evaluation/resnet50/resnet50.onnx.bin",0);
int64_t x6 = fsize(x5);
float* x7 = (float*)mmap(0, x6, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x5, 0);
float* x8 = x7+5205440;
CUDA_CALL(cudaMemcpy(x276, x8, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x279 = (float*)myGpuMalloc(256 * sizeof(float));
float* x9 = x7+148672;
CUDA_CALL(cudaMemcpy(x279, x9, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x282 = (float*)myGpuMalloc(128 * sizeof(float));
float* x10 = x7+816064;
CUDA_CALL(cudaMemcpy(x282, x10, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x285 = (float*)myGpuMalloc(128 * sizeof(float));
float* x11 = x7+950080;
CUDA_CALL(cudaMemcpy(x285, x11, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x288 = (float*)myGpuMalloc(64 * sizeof(float));
float* x12 = x7+94784;
CUDA_CALL(cudaMemcpy(x288, x12, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x291 = (float*)myGpuMalloc(32768 * sizeof(float));
float* x13 = x7+220608;
CUDA_CALL(cudaMemcpy(x291, x13, 32768 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x294 = (float*)myGpuMalloc(512 * sizeof(float));
float* x14 = x7+22495680;
CUDA_CALL(cudaMemcpy(x294, x14, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x297 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x15 = x7+2964928;
CUDA_CALL(cudaMemcpy(x297, x15, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x300 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x16 = x7+4348352;
CUDA_CALL(cudaMemcpy(x300, x16, 589824 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x303 = (float*)myGpuMalloc(512 * sizeof(float));
float* x17 = x7+20133312;
CUDA_CALL(cudaMemcpy(x303, x17, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x306 = (float*)myGpuMalloc(256 * sizeof(float));
float* x18 = x7+2169536;
CUDA_CALL(cudaMemcpy(x306, x18, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x309 = (float*)myGpuMalloc(128 * sizeof(float));
float* x19 = x7+668224;
CUDA_CALL(cudaMemcpy(x309, x19, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x312 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x20 = x7+2432448;
CUDA_CALL(cudaMemcpy(x312, x20, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x315 = (float*)myGpuMalloc(512 * sizeof(float));
float* x21 = x7+1446336;
CUDA_CALL(cudaMemcpy(x315, x21, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x318 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x22 = x7+4081088;
CUDA_CALL(cudaMemcpy(x318, x22, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x321 = (float*)myGpuMalloc(256 * sizeof(float));
float* x23 = x7+1578688;
CUDA_CALL(cudaMemcpy(x321, x23, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x324 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x24 = x7+6325696;
CUDA_CALL(cudaMemcpy(x324, x24, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x327 = (float*)myGpuMalloc(512 * sizeof(float));
float* x25 = x7+602048;
CUDA_CALL(cudaMemcpy(x327, x25, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x330 = (float*)myGpuMalloc(64 * sizeof(float));
float* x26 = x7+165888;
CUDA_CALL(cudaMemcpy(x330, x26, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x333 = (float*)myGpuMalloc(512 * sizeof(float));
float* x27 = x7+1164736;
CUDA_CALL(cudaMemcpy(x333, x27, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x336 = (float*)myGpuMalloc(64 * sizeof(float));
float* x28 = x7+6080;
CUDA_CALL(cudaMemcpy(x336, x28, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x339 = (float*)myGpuMalloc(147456 * sizeof(float));
float* x29 = x7+253888;
CUDA_CALL(cudaMemcpy(x339, x29, 147456 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x342 = (float*)myGpuMalloc(2359296 * sizeof(float));
float* x30 = x7+20135360;
CUDA_CALL(cudaMemcpy(x342, x30, 2359296 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x345 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x31 = x7+2960832;
CUDA_CALL(cudaMemcpy(x345, x31, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x348 = (float*)myGpuMalloc(256 * sizeof(float));
float* x32 = x7+3227072;
CUDA_CALL(cudaMemcpy(x348, x32, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x351 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x33 = x7+3228096;
CUDA_CALL(cudaMemcpy(x351, x33, 589824 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x354 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x34 = x7+43456;
CUDA_CALL(cudaMemcpy(x354, x34, 16384 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x357 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x35 = x7+22496704;
CUDA_CALL(cudaMemcpy(x357, x35, 1048576 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x360 = (float*)myGpuMalloc(2359296 * sizeof(float));
float* x36 = x7+9092544;
CUDA_CALL(cudaMemcpy(x360, x36, 2359296 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x363 = (float*)myGpuMalloc(128 * sizeof(float));
float* x37 = x7+816320;
CUDA_CALL(cudaMemcpy(x363, x37, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x366 = (float*)myGpuMalloc(256 * sizeof(float));
float* x38 = x7+60608;
CUDA_CALL(cudaMemcpy(x366, x38, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x369 = (float*)myGpuMalloc(256 * sizeof(float));
float* x39 = x7+219584;
CUDA_CALL(cudaMemcpy(x369, x39, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x372 = (float*)myGpuMalloc(128 * sizeof(float));
float* x40 = x7+1379392;
CUDA_CALL(cudaMemcpy(x372, x40, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x375 = (float*)myGpuMalloc(128 * sizeof(float));
float* x41 = x7+1231296;
CUDA_CALL(cudaMemcpy(x375, x41, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x378 = (float*)myGpuMalloc(64 * sizeof(float));
float* x42 = x7+1856;
CUDA_CALL(cudaMemcpy(x378, x42, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x381 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x43 = x7+1098176;
CUDA_CALL(cudaMemcpy(x381, x43, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x384 = (float*)myGpuMalloc(512 * sizeof(float));
float* x44 = x7+601536;
CUDA_CALL(cudaMemcpy(x384, x44, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x387 = (float*)myGpuMalloc(128 * sizeof(float));
float* x45 = x7+401728;
CUDA_CALL(cudaMemcpy(x387, x45, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x390 = (float*)myGpuMalloc(64 * sizeof(float));
float* x46 = x7+131904;
CUDA_CALL(cudaMemcpy(x390, x46, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x393 = (float*)myGpuMalloc(128 * sizeof(float));
float* x47 = x7+949696;
CUDA_CALL(cudaMemcpy(x393, x47, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x396 = (float*)myGpuMalloc(512 * sizeof(float));
float* x48 = x7+15664576;
CUDA_CALL(cudaMemcpy(x396, x48, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x399 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x49 = x7+18027968;
CUDA_CALL(cudaMemcpy(x399, x49, 1048576 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x402 = (float*)myGpuMalloc(10 * sizeof(float));
float* x50 = x7+23573952;
CUDA_CALL(cudaMemcpy(x402, x50, 10 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x405 = (float*)myGpuMalloc(64 * sizeof(float));
float* x51 = x7+43264;
CUDA_CALL(cudaMemcpy(x405, x51, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x408 = (float*)myGpuMalloc(512 * sizeof(float));
float* x52 = x7+11453376;
CUDA_CALL(cudaMemcpy(x408, x52, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x411 = (float*)myGpuMalloc(64 * sizeof(float));
float* x53 = x7+6272;
CUDA_CALL(cudaMemcpy(x411, x53, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x414 = (float*)myGpuMalloc(512 * sizeof(float));
float* x54 = x7+882112;
CUDA_CALL(cudaMemcpy(x414, x54, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x417 = (float*)myGpuMalloc(64 * sizeof(float));
float* x55 = x7+6144;
CUDA_CALL(cudaMemcpy(x417, x55, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x420 = (float*)myGpuMalloc(512 * sizeof(float));
float* x56 = x7+1445824;
CUDA_CALL(cudaMemcpy(x420, x56, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x423 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x57 = x7+1379776;
CUDA_CALL(cudaMemcpy(x423, x57, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x426 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x58 = x7+3818944;
CUDA_CALL(cudaMemcpy(x426, x58, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x429 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x59 = x7+5202368;
CUDA_CALL(cudaMemcpy(x429, x59, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x432 = (float*)myGpuMalloc(256 * sizeof(float));
float* x60 = x7+148416;
CUDA_CALL(cudaMemcpy(x432, x60, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x435 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x61 = x7+7441856;
CUDA_CALL(cudaMemcpy(x435, x61, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x438 = (float*)myGpuMalloc(64 * sizeof(float));
float* x62 = x7+94720;
CUDA_CALL(cudaMemcpy(x438, x62, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x441 = (float*)myGpuMalloc(128 * sizeof(float));
float* x63 = x7+1097792;
CUDA_CALL(cudaMemcpy(x441, x63, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x444 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x64 = x7+12504512;
CUDA_CALL(cudaMemcpy(x444, x64, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x447 = (float*)myGpuMalloc(256 * sizeof(float));
float* x65 = x7+4938944;
CUDA_CALL(cudaMemcpy(x447, x65, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x450 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x66 = x7+14611904;
CUDA_CALL(cudaMemcpy(x450, x66, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x453 = (float*)myGpuMalloc(512 * sizeof(float));
float* x67 = x7+15666112;
CUDA_CALL(cudaMemcpy(x453, x67, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x456 = (float*)myGpuMalloc(512 * sizeof(float));
float* x68 = x7+18026432;
CUDA_CALL(cudaMemcpy(x456, x68, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x459 = (float*)myGpuMalloc(512 * sizeof(float));
float* x69 = x7+9091520;
CUDA_CALL(cudaMemcpy(x459, x69, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x462 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x70 = x7+19080640;
CUDA_CALL(cudaMemcpy(x462, x70, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x465 = (float*)myGpuMalloc(256 * sizeof(float));
float* x71 = x7+6588608;
CUDA_CALL(cudaMemcpy(x465, x71, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x468 = (float*)myGpuMalloc(256 * sizeof(float));
float* x72 = x7+8299456;
CUDA_CALL(cudaMemcpy(x468, x72, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x471 = (float*)myGpuMalloc(256 * sizeof(float));
float* x73 = x7+60352;
CUDA_CALL(cudaMemcpy(x471, x73, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x474 = (float*)myGpuMalloc(64 * sizeof(float));
float* x74 = x7+202944;
CUDA_CALL(cudaMemcpy(x474, x74, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x477 = (float*)myGpuMalloc(36864 * sizeof(float));
float* x75 = x7+166080;
CUDA_CALL(cudaMemcpy(x477, x75, 36864 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x480 = (float*)myGpuMalloc(256 * sizeof(float));
float* x76 = x7+6058432;
CUDA_CALL(cudaMemcpy(x480, x76, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x483 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x77 = x7+2436544;
CUDA_CALL(cudaMemcpy(x483, x77, 524288 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x486 = (float*)myGpuMalloc(256 * sizeof(float));
float* x78 = x7+77248;
CUDA_CALL(cudaMemcpy(x486, x78, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x489 = (float*)myGpuMalloc(256 * sizeof(float));
float* x79 = x7+6587840;
CUDA_CALL(cudaMemcpy(x489, x79, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x492 = (float*)myGpuMalloc(512 * sizeof(float));
float* x80 = x7+20133824;
CUDA_CALL(cudaMemcpy(x492, x80, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x495 = (float*)myGpuMalloc(128 * sizeof(float));
float* x81 = x7+1379264;
CUDA_CALL(cudaMemcpy(x495, x81, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x498 = (float*)myGpuMalloc(256 * sizeof(float));
float* x82 = x7+7708608;
CUDA_CALL(cudaMemcpy(x498, x82, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x501 = (float*)myGpuMalloc(64 * sizeof(float));
float* x83 = x7+165824;
CUDA_CALL(cudaMemcpy(x501, x83, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x504 = (float*)myGpuMalloc(512 * sizeof(float));
float* x84 = x7+1164224;
CUDA_CALL(cudaMemcpy(x504, x84, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x507 = (float*)myGpuMalloc(36864 * sizeof(float));
float* x85 = x7+94912;
CUDA_CALL(cudaMemcpy(x507, x85, 36864 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x510 = (float*)myGpuMalloc(128 * sizeof(float));
float* x86 = x7+253376;
CUDA_CALL(cudaMemcpy(x510, x86, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x513 = (float*)myGpuMalloc(256 * sizeof(float));
float* x87 = x7+7708096;
CUDA_CALL(cudaMemcpy(x513, x87, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x516 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x88 = x7+2962880;
CUDA_CALL(cudaMemcpy(x516, x88, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x519 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x89 = x7+203200;
CUDA_CALL(cudaMemcpy(x519, x89, 16384 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x522 = (float*)myGpuMalloc(512 * sizeof(float));
float* x90 = x7+883648;
CUDA_CALL(cudaMemcpy(x522, x90, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x525 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x91 = x7+6059456;
CUDA_CALL(cudaMemcpy(x525, x91, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x528 = (float*)myGpuMalloc(36864 * sizeof(float));
float* x92 = x7+6336;
CUDA_CALL(cudaMemcpy(x528, x92, 36864 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x531 = (float*)myGpuMalloc(256 * sizeof(float));
float* x93 = x7+148928;
CUDA_CALL(cudaMemcpy(x531, x93, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x534 = (float*)myGpuMalloc(256 * sizeof(float));
float* x94 = x7+5467584;
CUDA_CALL(cudaMemcpy(x534, x94, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x537 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x95 = x7+8563136;
CUDA_CALL(cudaMemcpy(x537, x95, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x540 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x96 = x7+19076544;
CUDA_CALL(cudaMemcpy(x540, x96, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x543 = (float*)myGpuMalloc(128 * sizeof(float));
float* x97 = x7+816192;
CUDA_CALL(cudaMemcpy(x543, x97, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x546 = (float*)myGpuMalloc(256 * sizeof(float));
float* x98 = x7+3818176;
CUDA_CALL(cudaMemcpy(x546, x98, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x549 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x99 = x7+8299968;
CUDA_CALL(cudaMemcpy(x549, x99, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x552 = (float*)myGpuMalloc(256 * sizeof(float));
float* x100 = x7+5468352;
CUDA_CALL(cudaMemcpy(x552, x100, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x555 = (float*)myGpuMalloc(256 * sizeof(float));
float* x101 = x7+2170048;
CUDA_CALL(cudaMemcpy(x555, x101, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x558 = (float*)myGpuMalloc(128 * sizeof(float));
float* x102 = x7+668352;
CUDA_CALL(cudaMemcpy(x558, x102, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x561 = (float*)myGpuMalloc(512 * sizeof(float));
float* x103 = x7+468928;
CUDA_CALL(cudaMemcpy(x561, x103, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x564 = (float*)myGpuMalloc(64 * sizeof(float));
float* x104 = x7+94848;
CUDA_CALL(cudaMemcpy(x564, x104, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x567 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x105 = x7+23545280;
CUDA_CALL(cudaMemcpy(x567, x105, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x570 = (float*)myGpuMalloc(256 * sizeof(float));
float* x106 = x7+7179456;
CUDA_CALL(cudaMemcpy(x570, x106, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x573 = (float*)myGpuMalloc(64 * sizeof(float));
float* x107 = x7+43328;
CUDA_CALL(cudaMemcpy(x573, x107, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x576 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x108 = x7+401856;
CUDA_CALL(cudaMemcpy(x576, x108, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x579 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x109 = x7+14609856;
CUDA_CALL(cudaMemcpy(x579, x109, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x582 = (float*)myGpuMalloc(256 * sizeof(float));
float* x110 = x7+2169280;
CUDA_CALL(cudaMemcpy(x582, x110, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x585 = (float*)myGpuMalloc(256 * sizeof(float));
float* x111 = x7+7178944;
CUDA_CALL(cudaMemcpy(x585, x111, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x588 = (float*)myGpuMalloc(64 * sizeof(float));
float* x112 = x7+1920;
CUDA_CALL(cudaMemcpy(x588, x112, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x591 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x113 = x7+816576;
CUDA_CALL(cudaMemcpy(x591, x113, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x594 = (float*)myGpuMalloc(128 * sizeof(float));
float* x114 = x7+949952;
CUDA_CALL(cudaMemcpy(x594, x114, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x597 = (float*)myGpuMalloc(512 * sizeof(float));
float* x115 = x7+11452864;
CUDA_CALL(cudaMemcpy(x597, x115, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x600 = (float*)myGpuMalloc(64 * sizeof(float));
float* x116 = x7+6208;
CUDA_CALL(cudaMemcpy(x600, x116, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x603 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x117 = x7+12506560;
CUDA_CALL(cudaMemcpy(x603, x117, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x606 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x118 = x7+4939200;
CUDA_CALL(cudaMemcpy(x606, x118, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x609 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x119 = x7+2433472;
CUDA_CALL(cudaMemcpy(x609, x119, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x612 = (float*)myGpuMalloc(64 * sizeof(float));
float* x120 = x7+203136;
CUDA_CALL(cudaMemcpy(x612, x120, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x615 = (float*)myGpuMalloc(512 * sizeof(float));
float* x121 = x7+601024;
CUDA_CALL(cudaMemcpy(x615, x121, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x618 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x122 = x7+7442880;
CUDA_CALL(cudaMemcpy(x618, x122, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x621 = (float*)myGpuMalloc(512 * sizeof(float));
float* x123 = x7+9092032;
CUDA_CALL(cudaMemcpy(x621, x123, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x624 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x124 = x7+8564160;
CUDA_CALL(cudaMemcpy(x624, x124, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x627 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x125 = x7+23551424;
CUDA_CALL(cudaMemcpy(x627, x125, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x630 = (float*)myGpuMalloc(256 * sizeof(float));
float* x126 = x7+4938688;
CUDA_CALL(cudaMemcpy(x630, x126, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x633 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x127 = x7+14613952;
CUDA_CALL(cudaMemcpy(x633, x127, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x636 = (float*)myGpuMalloc(256 * sizeof(float));
float* x128 = x7+60096;
CUDA_CALL(cudaMemcpy(x636, x128, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x639 = (float*)myGpuMalloc(128 * sizeof(float));
float* x129 = x7+1097664;
CUDA_CALL(cudaMemcpy(x639, x129, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x642 = (float*)myGpuMalloc(128 * sizeof(float));
float* x130 = x7+401600;
CUDA_CALL(cudaMemcpy(x642, x130, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x645 = (float*)myGpuMalloc(256 * sizeof(float));
float* x131 = x7+4347328;
CUDA_CALL(cudaMemcpy(x645, x131, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x648 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x132 = x7+132032;
CUDA_CALL(cudaMemcpy(x648, x132, 16384 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x651 = (float*)myGpuMalloc(256 * sizeof(float));
float* x133 = x7+1578944;
CUDA_CALL(cudaMemcpy(x651, x133, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x654 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x134 = x7+1165760;
CUDA_CALL(cudaMemcpy(x654, x134, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x657 = (float*)myGpuMalloc(256 * sizeof(float));
float* x135 = x7+220352;
CUDA_CALL(cudaMemcpy(x657, x135, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x660 = (float*)myGpuMalloc(128 * sizeof(float));
float* x136 = x7+253760;
CUDA_CALL(cudaMemcpy(x660, x136, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x663 = (float*)myGpuMalloc(64 * sizeof(float));
float* x137 = x7+203008;
CUDA_CALL(cudaMemcpy(x663, x137, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x666 = (float*)myGpuMalloc(256 * sizeof(float));
float* x138 = x7+6058688;
CUDA_CALL(cudaMemcpy(x666, x138, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x669 = (float*)myGpuMalloc(512 * sizeof(float));
float* x139 = x7+15665088;
CUDA_CALL(cudaMemcpy(x669, x139, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x672 = (float*)myGpuMalloc(512 * sizeof(float));
float* x140 = x7+18026944;
CUDA_CALL(cudaMemcpy(x672, x140, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x675 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x141 = x7+8566208;
CUDA_CALL(cudaMemcpy(x675, x141, 524288 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x678 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x142 = x7+5203392;
CUDA_CALL(cudaMemcpy(x678, x142, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x681 = (float*)myGpuMalloc(256 * sizeof(float));
float* x143 = x7+8298944;
CUDA_CALL(cudaMemcpy(x681, x143, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x684 = (float*)myGpuMalloc(64 * sizeof(float));
float* x144 = x7+94656;
CUDA_CALL(cudaMemcpy(x684, x144, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x687 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x145 = x7+4084160;
CUDA_CALL(cudaMemcpy(x687, x145, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x690 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x146 = x7+19078592;
CUDA_CALL(cudaMemcpy(x690, x146, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x693 = (float*)myGpuMalloc(512 * sizeof(float));
float* x147 = x7+467392;
CUDA_CALL(cudaMemcpy(x693, x147, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x696 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x148 = x7+6322624;
CUDA_CALL(cudaMemcpy(x696, x148, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x699 = (float*)myGpuMalloc(512 * sizeof(float));
float* x149 = x7+883136;
CUDA_CALL(cudaMemcpy(x699, x149, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x702 = (float*)myGpuMalloc(128 * sizeof(float));
float* x150 = x7+1379648;
CUDA_CALL(cudaMemcpy(x702, x150, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x705 = (float*)myGpuMalloc(512 * sizeof(float));
float* x151 = x7+468416;
CUDA_CALL(cudaMemcpy(x705, x151, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x708 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x152 = x7+149440;
CUDA_CALL(cudaMemcpy(x708, x152, 16384 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x711 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x153 = x7+7445952;
CUDA_CALL(cudaMemcpy(x711, x153, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x714 = (float*)myGpuMalloc(1728 * sizeof(float));
float* x154 = x7+0;
CUDA_CALL(cudaMemcpy(x714, x154, 1728 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x717 = (float*)myGpuMalloc(64 * sizeof(float));
float* x155 = x7+131840;
CUDA_CALL(cudaMemcpy(x717, x155, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x720 = (float*)myGpuMalloc(512 * sizeof(float));
float* x156 = x7+15665600;
CUDA_CALL(cudaMemcpy(x720, x156, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x723 = (float*)myGpuMalloc(2359296 * sizeof(float));
float* x157 = x7+15666624;
CUDA_CALL(cudaMemcpy(x723, x157, 2359296 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x726 = (float*)myGpuMalloc(512 * sizeof(float));
float* x158 = x7+1445312;
CUDA_CALL(cudaMemcpy(x726, x158, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x729 = (float*)myGpuMalloc(256 * sizeof(float));
float* x159 = x7+3227840;
CUDA_CALL(cudaMemcpy(x729, x159, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x732 = (float*)myGpuMalloc(64 * sizeof(float));
float* x160 = x7+43392;
CUDA_CALL(cudaMemcpy(x732, x160, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x735 = (float*)myGpuMalloc(512 * sizeof(float));
float* x161 = x7+11452352;
CUDA_CALL(cudaMemcpy(x735, x161, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x738 = (float*)myGpuMalloc(512 * sizeof(float));
float* x162 = x7+18025920;
CUDA_CALL(cudaMemcpy(x738, x162, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x741 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x163 = x7+6324672;
CUDA_CALL(cudaMemcpy(x741, x163, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x744 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x164 = x7+60864;
CUDA_CALL(cudaMemcpy(x744, x164, 16384 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x747 = (float*)myGpuMalloc(256 * sizeof(float));
float* x165 = x7+5468096;
CUDA_CALL(cudaMemcpy(x747, x165, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x750 = (float*)myGpuMalloc(64 * sizeof(float));
float* x166 = x7+43200;
CUDA_CALL(cudaMemcpy(x750, x166, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x753 = (float*)myGpuMalloc(147456 * sizeof(float));
float* x167 = x7+1231808;
CUDA_CALL(cudaMemcpy(x753, x167, 147456 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x756 = (float*)myGpuMalloc(256 * sizeof(float));
float* x168 = x7+149184;
CUDA_CALL(cudaMemcpy(x756, x168, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x759 = (float*)myGpuMalloc(512 * sizeof(float));
float* x169 = x7+1163712;
CUDA_CALL(cudaMemcpy(x759, x169, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x762 = (float*)myGpuMalloc(256 * sizeof(float));
float* x170 = x7+7178688;
CUDA_CALL(cudaMemcpy(x762, x170, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x765 = (float*)myGpuMalloc(512 * sizeof(float));
float* x171 = x7+22495168;
CUDA_CALL(cudaMemcpy(x765, x171, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x768 = (float*)myGpuMalloc(128 * sizeof(float));
float* x172 = x7+949824;
CUDA_CALL(cudaMemcpy(x768, x172, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x771 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x173 = x7+78272;
CUDA_CALL(cudaMemcpy(x771, x173, 16384 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x774 = (float*)myGpuMalloc(128 * sizeof(float));
float* x174 = x7+253504;
CUDA_CALL(cudaMemcpy(x774, x174, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x777 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x175 = x7+14607808;
CUDA_CALL(cudaMemcpy(x777, x175, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x780 = (float*)myGpuMalloc(256 * sizeof(float));
float* x176 = x7+4348096;
CUDA_CALL(cudaMemcpy(x780, x176, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x783 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x177 = x7+1579456;
CUDA_CALL(cudaMemcpy(x783, x177, 589824 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x786 = (float*)myGpuMalloc(256 * sizeof(float));
float* x178 = x7+7708864;
CUDA_CALL(cudaMemcpy(x786, x178, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x789 = (float*)myGpuMalloc(128 * sizeof(float));
float* x179 = x7+668480;
CUDA_CALL(cudaMemcpy(x789, x179, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x792 = (float*)myGpuMalloc(256 * sizeof(float));
float* x180 = x7+4347840;
CUDA_CALL(cudaMemcpy(x792, x180, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x795 = (float*)myGpuMalloc(64 * sizeof(float));
float* x181 = x7+203072;
CUDA_CALL(cudaMemcpy(x795, x181, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x798 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x182 = x7+1447360;
CUDA_CALL(cudaMemcpy(x798, x182, 131072 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x801 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x183 = x7+23547328;
CUDA_CALL(cudaMemcpy(x801, x183, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x804 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x184 = x7+4083136;
CUDA_CALL(cudaMemcpy(x804, x184, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x807 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x185 = x7+8565184;
CUDA_CALL(cudaMemcpy(x807, x185, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x810 = (float*)myGpuMalloc(256 * sizeof(float));
float* x186 = x7+220096;
CUDA_CALL(cudaMemcpy(x810, x186, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x813 = (float*)myGpuMalloc(256 * sizeof(float));
float* x187 = x7+6588096;
CUDA_CALL(cudaMemcpy(x813, x187, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x816 = (float*)myGpuMalloc(256 * sizeof(float));
float* x188 = x7+6058944;
CUDA_CALL(cudaMemcpy(x816, x188, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x819 = (float*)myGpuMalloc(64 * sizeof(float));
float* x189 = x7+166016;
CUDA_CALL(cudaMemcpy(x819, x189, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x822 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x190 = x7+5204416;
CUDA_CALL(cudaMemcpy(x822, x190, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x825 = (float*)myGpuMalloc(256 * sizeof(float));
float* x191 = x7+8299200;
CUDA_CALL(cudaMemcpy(x825, x191, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x828 = (float*)myGpuMalloc(128 * sizeof(float));
float* x192 = x7+401472;
CUDA_CALL(cudaMemcpy(x828, x192, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x831 = (float*)myGpuMalloc(147456 * sizeof(float));
float* x193 = x7+950208;
CUDA_CALL(cudaMemcpy(x831, x193, 147456 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x834 = (float*)myGpuMalloc(256 * sizeof(float));
float* x194 = x7+4938432;
CUDA_CALL(cudaMemcpy(x834, x194, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x837 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x195 = x7+12508608;
CUDA_CALL(cudaMemcpy(x837, x195, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x840 = (float*)myGpuMalloc(512 * sizeof(float));
float* x196 = x7+22494656;
CUDA_CALL(cudaMemcpy(x840, x196, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x843 = (float*)myGpuMalloc(512 * sizeof(float));
float* x197 = x7+18027456;
CUDA_CALL(cudaMemcpy(x843, x197, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x846 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x198 = x7+884160;
CUDA_CALL(cudaMemcpy(x846, x198, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x849 = (float*)myGpuMalloc(256 * sizeof(float));
float* x199 = x7+4347584;
CUDA_CALL(cudaMemcpy(x849, x199, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x852 = (float*)myGpuMalloc(256 * sizeof(float));
float* x200 = x7+1579200;
CUDA_CALL(cudaMemcpy(x852, x200, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x855 = (float*)myGpuMalloc(256 * sizeof(float));
float* x201 = x7+59840;
CUDA_CALL(cudaMemcpy(x855, x201, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x858 = (float*)myGpuMalloc(256 * sizeof(float));
float* x202 = x7+3818432;
CUDA_CALL(cudaMemcpy(x858, x202, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x861 = (float*)myGpuMalloc(512 * sizeof(float));
float* x203 = x7+9090496;
CUDA_CALL(cudaMemcpy(x861, x203, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x864 = (float*)myGpuMalloc(512 * sizeof(float));
float* x204 = x7+22496192;
CUDA_CALL(cudaMemcpy(x864, x204, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x867 = (float*)myGpuMalloc(256 * sizeof(float));
float* x205 = x7+77504;
CUDA_CALL(cudaMemcpy(x867, x205, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x870 = (float*)myGpuMalloc(128 * sizeof(float));
float* x206 = x7+253632;
CUDA_CALL(cudaMemcpy(x870, x206, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x873 = (float*)myGpuMalloc(512 * sizeof(float));
float* x207 = x7+11451840;
CUDA_CALL(cudaMemcpy(x873, x207, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x876 = (float*)myGpuMalloc(64 * sizeof(float));
float* x208 = x7+1728;
CUDA_CALL(cudaMemcpy(x876, x208, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x879 = (float*)myGpuMalloc(512 * sizeof(float));
float* x209 = x7+600512;
CUDA_CALL(cudaMemcpy(x879, x209, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x882 = (float*)myGpuMalloc(64 * sizeof(float));
float* x210 = x7+131776;
CUDA_CALL(cudaMemcpy(x882, x210, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x885 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x211 = x7+7443904;
CUDA_CALL(cudaMemcpy(x885, x211, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x888 = (float*)myGpuMalloc(512 * sizeof(float));
float* x212 = x7+467904;
CUDA_CALL(cudaMemcpy(x888, x212, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x891 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x213 = x7+2963904;
CUDA_CALL(cudaMemcpy(x891, x213, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x894 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x214 = x7+11453888;
CUDA_CALL(cudaMemcpy(x894, x214, 1048576 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x897 = (float*)myGpuMalloc(512 * sizeof(float));
float* x215 = x7+20134336;
CUDA_CALL(cudaMemcpy(x897, x215, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x900 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x216 = x7+12510656;
CUDA_CALL(cudaMemcpy(x900, x216, 2097152 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x903 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x217 = x7+14616000;
CUDA_CALL(cudaMemcpy(x903, x217, 1048576 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x906 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x218 = x7+2434496;
CUDA_CALL(cudaMemcpy(x906, x218, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x909 = (float*)myGpuMalloc(128 * sizeof(float));
float* x219 = x7+1097920;
CUDA_CALL(cudaMemcpy(x909, x219, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x912 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x220 = x7+4085184;
CUDA_CALL(cudaMemcpy(x912, x220, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x915 = (float*)myGpuMalloc(256 * sizeof(float));
float* x221 = x7+3227328;
CUDA_CALL(cudaMemcpy(x915, x221, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x918 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x222 = x7+2961856;
CUDA_CALL(cudaMemcpy(x918, x222, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x921 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x223 = x7+7179712;
CUDA_CALL(cudaMemcpy(x921, x223, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x924 = (float*)myGpuMalloc(128 * sizeof(float));
float* x224 = x7+668096;
CUDA_CALL(cudaMemcpy(x924, x224, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x927 = (float*)myGpuMalloc(512 * sizeof(float));
float* x225 = x7+1165248;
CUDA_CALL(cudaMemcpy(x927, x225, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x930 = (float*)myGpuMalloc(512 * sizeof(float));
float* x226 = x7+9091008;
CUDA_CALL(cudaMemcpy(x930, x226, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x933 = (float*)myGpuMalloc(128 * sizeof(float));
float* x227 = x7+816448;
CUDA_CALL(cudaMemcpy(x933, x227, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x936 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x228 = x7+7709120;
CUDA_CALL(cudaMemcpy(x936, x228, 589824 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x939 = (float*)myGpuMalloc(20480 * sizeof(float));
float* x229 = x7+23553472;
CUDA_CALL(cudaMemcpy(x939, x229, 20480 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x942 = (float*)myGpuMalloc(256 * sizeof(float));
float* x230 = x7+4938176;
CUDA_CALL(cudaMemcpy(x942, x230, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x945 = (float*)myGpuMalloc(256 * sizeof(float));
float* x231 = x7+2169792;
CUDA_CALL(cudaMemcpy(x945, x231, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x948 = (float*)myGpuMalloc(256 * sizeof(float));
float* x232 = x7+6059200;
CUDA_CALL(cudaMemcpy(x948, x232, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x951 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x233 = x7+6323648;
CUDA_CALL(cudaMemcpy(x951, x233, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x954 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x234 = x7+4082112;
CUDA_CALL(cudaMemcpy(x954, x234, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x957 = (float*)myGpuMalloc(4096 * sizeof(float));
float* x235 = x7+1984;
CUDA_CALL(cudaMemcpy(x957, x235, 4096 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x960 = (float*)myGpuMalloc(512 * sizeof(float));
float* x236 = x7+1446848;
CUDA_CALL(cudaMemcpy(x960, x236, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x963 = (float*)myGpuMalloc(147456 * sizeof(float));
float* x237 = x7+668608;
CUDA_CALL(cudaMemcpy(x963, x237, 147456 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x966 = (float*)myGpuMalloc(128 * sizeof(float));
float* x238 = x7+1231552;
CUDA_CALL(cudaMemcpy(x966, x238, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x969 = (float*)myGpuMalloc(256 * sizeof(float));
float* x239 = x7+3818688;
CUDA_CALL(cudaMemcpy(x969, x239, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x972 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x240 = x7+6321600;
CUDA_CALL(cudaMemcpy(x972, x240, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x975 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x241 = x7+12502464;
CUDA_CALL(cudaMemcpy(x975, x241, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x978 = (float*)myGpuMalloc(256 * sizeof(float));
float* x242 = x7+8299712;
CUDA_CALL(cudaMemcpy(x978, x242, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x981 = (float*)myGpuMalloc(256 * sizeof(float));
float* x243 = x7+5467840;
CUDA_CALL(cudaMemcpy(x981, x243, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x984 = (float*)myGpuMalloc(128 * sizeof(float));
float* x244 = x7+1231424;
CUDA_CALL(cudaMemcpy(x984, x244, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x987 = (float*)myGpuMalloc(256 * sizeof(float));
float* x245 = x7+78016;
CUDA_CALL(cudaMemcpy(x987, x245, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x990 = (float*)myGpuMalloc(64 * sizeof(float));
float* x246 = x7+131968;
CUDA_CALL(cudaMemcpy(x990, x246, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x993 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x247 = x7+19082688;
CUDA_CALL(cudaMemcpy(x993, x247, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x996 = (float*)myGpuMalloc(512 * sizeof(float));
float* x248 = x7+882624;
CUDA_CALL(cudaMemcpy(x996, x248, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x999 = (float*)myGpuMalloc(256 * sizeof(float));
float* x249 = x7+219840;
CUDA_CALL(cudaMemcpy(x999, x249, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1002 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x250 = x7+8562112;
CUDA_CALL(cudaMemcpy(x1002, x250, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1005 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x251 = x7+5468608;
CUDA_CALL(cudaMemcpy(x1005, x251, 589824 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1008 = (float*)myGpuMalloc(256 * sizeof(float));
float* x252 = x7+7179200;
CUDA_CALL(cudaMemcpy(x1008, x252, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1011 = (float*)myGpuMalloc(64 * sizeof(float));
float* x253 = x7+1792;
CUDA_CALL(cudaMemcpy(x1011, x253, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1014 = (float*)myGpuMalloc(128 * sizeof(float));
float* x254 = x7+401344;
CUDA_CALL(cudaMemcpy(x1014, x254, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1017 = (float*)myGpuMalloc(256 * sizeof(float));
float* x255 = x7+7708352;
CUDA_CALL(cudaMemcpy(x1017, x255, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1020 = (float*)myGpuMalloc(256 * sizeof(float));
float* x256 = x7+6588352;
CUDA_CALL(cudaMemcpy(x1020, x256, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1023 = (float*)myGpuMalloc(512 * sizeof(float));
float* x257 = x7+20134848;
CUDA_CALL(cudaMemcpy(x1023, x257, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1026 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x258 = x7+602560;
CUDA_CALL(cudaMemcpy(x1026, x258, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1029 = (float*)myGpuMalloc(64 * sizeof(float));
float* x259 = x7+165952;
CUDA_CALL(cudaMemcpy(x1029, x259, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1032 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x260 = x7+469440;
CUDA_CALL(cudaMemcpy(x1032, x260, 131072 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1035 = (float*)myGpuMalloc(256 * sizeof(float));
float* x261 = x7+3227584;
CUDA_CALL(cudaMemcpy(x1035, x261, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1038 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x262 = x7+23549376;
CUDA_CALL(cudaMemcpy(x1038, x262, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1041 = (float*)myGpuMalloc(128 * sizeof(float));
float* x263 = x7+1231680;
CUDA_CALL(cudaMemcpy(x1041, x263, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1044 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x264 = x7+6588864;
CUDA_CALL(cudaMemcpy(x1044, x264, 589824 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1047 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x265 = x7+5201344;
CUDA_CALL(cudaMemcpy(x1047, x265, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1050 = (float*)myGpuMalloc(256 * sizeof(float));
float* x266 = x7+77760;
CUDA_CALL(cudaMemcpy(x1050, x266, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1053 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x267 = x7+19084736;
CUDA_CALL(cudaMemcpy(x1053, x267, 1048576 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1056 = (float*)myGpuMalloc(128 * sizeof(float));
float* x268 = x7+1098048;
CUDA_CALL(cudaMemcpy(x1056, x268, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1059 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x269 = x7+2435520;
CUDA_CALL(cudaMemcpy(x1059, x269, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1062 = (float*)myGpuMalloc(128 * sizeof(float));
float* x270 = x7+1379520;
CUDA_CALL(cudaMemcpy(x1062, x270, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1065 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x271 = x7+2170304;
CUDA_CALL(cudaMemcpy(x1065, x271, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1068 = (float*)myGpuMalloc(256 * sizeof(float));
float* x272 = x7+1578432;
CUDA_CALL(cudaMemcpy(x1068, x272, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1071 = (float*)myGpuMalloc(256 * sizeof(float));
float* x273 = x7+3817920;
CUDA_CALL(cudaMemcpy(x1071, x273, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1074 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x274 = x7+7444928;
CUDA_CALL(cudaMemcpy(x1074, x274, 1024 * sizeof(float), cudaMemcpyHostToDevice));
int32_t x1076 = open("../../cifar10_data/cifar-10-batches-bin/data_batch_1.bin",0);
int64_t x1077 = fsize(x1076);
int64_t x1079 = x1077 / 3073LL;
int32_t x1080 = (int32_t)x1079;
int32_t x1081 = x1080 * 3072;
float* x1082 = (float*)myMalloc(x1081 * sizeof(float));;
int* x1083 = (int32_t*)myMalloc(x1080 * sizeof(int32_t));;
char* x1078 = (char*)mmap(0, x1077, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x1076, 0);
for(int x1085=0; x1085 < x1080; x1085++) {
int32_t x1086 = x1085 * 3073;
char x1087 = x1078[x1086];
int32_t x1088 = (int32_t)(unsigned char)x1087;
x1083[x1085] = x1088;
int32_t x1094 = x1086 + 1;
int32_t x1092 = x1085 * 3072;
for(int x1091=0; x1091 < 3072; x1091++) {
int32_t x1095 = x1094 + x1091;
char x1096 = x1078[x1095];
int32_t x1093 = x1092 + x1091;
float x1097 = (float)(unsigned char)x1096;
float x1098 = x1097 / 255.0f;
x1082[x1093] = x1098;

}

}
int32_t x1104 = x1080 / 64;
int32_t x1138 = 31 / 1;
int32_t x1139 = x1138 + 1;
int32_t x1143 = 4096 * x1139;
int32_t x1144 = x1143 * x1139;
int32_t x1140 = x1139 * x1139;
int32_t x1141 = 64 * x1140;
int32_t x1142 = 64 * x1141;
int32_t x1167 = x1139 - 2;
int32_t x1168 = x1167 / 2;
int32_t x1169 = x1168 + 1;
int32_t x1173 = 4096 * x1169;
int32_t x1174 = x1173 * x1169;
bool x1177 = x1169 >= 1;
bool x1178;
if (x1177) {
x1178 = x1177;
} else {
x1178 = false;
}
int32_t x1183 = x1168 / 1;
int32_t x1184 = x1183 + 1;
int32_t x1188 = 4096 * x1184;
int32_t x1189 = x1188 * x1184;
int32_t x1185 = x1184 * x1184;
int32_t x1186 = 64 * x1185;
int32_t x1187 = 64 * x1186;
int32_t x1208 = x1184 + 2;
bool x1209 = x1208 >= 3;
bool x1210;
if (x1209) {
x1210 = x1209;
} else {
x1210 = false;
}
int32_t x1215 = x1208 - 3;
int32_t x1216 = x1215 / 1;
int32_t x1217 = x1216 + 1;
int32_t x1221 = 4096 * x1217;
int32_t x1222 = x1221 * x1217;
int32_t x1218 = x1217 * x1217;
int32_t x1219 = 64 * x1218;
int32_t x1220 = 64 * x1219;
bool x1241 = x1217 >= 1;
bool x1242;
if (x1241) {
x1242 = x1241;
} else {
x1242 = false;
}
int32_t x1247 = x1216 / 1;
int32_t x1248 = x1247 + 1;
int32_t x1252 = 16384 * x1248;
int32_t x1253 = x1252 * x1248;
int32_t x1249 = x1248 * x1248;
int32_t x1250 = 256 * x1249;
int32_t x1251 = 64 * x1250;
int32_t x1271 = 16384 * x1184;
int32_t x1272 = x1271 * x1184;
int32_t x1269 = 256 * x1185;
int32_t x1270 = 64 * x1269;
bool x1285 = x1184 == 1;
bool x1286 = x1184 == x1248;
bool x1287 = x1285 || x1286;
bool x1288;
if (x1287) {
x1288 = x1287;
} else {
x1288 = false;
}
bool x1304 = x1248 >= 1;
bool x1305;
if (x1304) {
x1305 = x1304;
} else {
x1305 = false;
}
int32_t x1310 = x1247 / 1;
int32_t x1311 = x1310 + 1;
int32_t x1315 = 4096 * x1311;
int32_t x1316 = x1315 * x1311;
int32_t x1312 = x1311 * x1311;
int32_t x1313 = 64 * x1312;
int32_t x1314 = 64 * x1313;
int32_t x1335 = x1311 + 2;
bool x1336 = x1335 >= 3;
bool x1337;
if (x1336) {
x1337 = x1336;
} else {
x1337 = false;
}
int32_t x1342 = x1335 - 3;
int32_t x1343 = x1342 / 1;
int32_t x1344 = x1343 + 1;
int32_t x1348 = 4096 * x1344;
int32_t x1349 = x1348 * x1344;
int32_t x1345 = x1344 * x1344;
int32_t x1346 = 64 * x1345;
int32_t x1347 = 64 * x1346;
bool x1368 = x1344 >= 1;
bool x1369;
if (x1368) {
x1369 = x1368;
} else {
x1369 = false;
}
int32_t x1374 = x1343 / 1;
int32_t x1375 = x1374 + 1;
int32_t x1379 = 16384 * x1375;
int32_t x1380 = x1379 * x1375;
int32_t x1376 = x1375 * x1375;
int32_t x1377 = 256 * x1376;
int32_t x1378 = 64 * x1377;
bool x1393 = x1248 == 1;
bool x1394 = x1248 == x1375;
bool x1395 = x1393 || x1394;
bool x1396;
if (x1395) {
x1396 = x1395;
} else {
x1396 = false;
}
bool x1412 = x1375 >= 1;
bool x1413;
if (x1412) {
x1413 = x1412;
} else {
x1413 = false;
}
int32_t x1418 = x1374 / 1;
int32_t x1419 = x1418 + 1;
int32_t x1423 = 4096 * x1419;
int32_t x1424 = x1423 * x1419;
int32_t x1420 = x1419 * x1419;
int32_t x1421 = 64 * x1420;
int32_t x1422 = 64 * x1421;
int32_t x1443 = x1419 + 2;
bool x1444 = x1443 >= 3;
bool x1445;
if (x1444) {
x1445 = x1444;
} else {
x1445 = false;
}
int32_t x1450 = x1443 - 3;
int32_t x1451 = x1450 / 1;
int32_t x1452 = x1451 + 1;
int32_t x1456 = 4096 * x1452;
int32_t x1457 = x1456 * x1452;
int32_t x1453 = x1452 * x1452;
int32_t x1454 = 64 * x1453;
int32_t x1455 = 64 * x1454;
bool x1476 = x1452 >= 1;
bool x1477;
if (x1476) {
x1477 = x1476;
} else {
x1477 = false;
}
int32_t x1482 = x1451 / 1;
int32_t x1483 = x1482 + 1;
int32_t x1487 = 16384 * x1483;
int32_t x1488 = x1487 * x1483;
int32_t x1484 = x1483 * x1483;
int32_t x1485 = 256 * x1484;
int32_t x1486 = 64 * x1485;
bool x1501 = x1375 == 1;
bool x1502 = x1375 == x1483;
bool x1503 = x1501 || x1502;
bool x1504;
if (x1503) {
x1504 = x1503;
} else {
x1504 = false;
}
bool x1520 = x1483 >= 1;
bool x1521;
if (x1520) {
x1521 = x1520;
} else {
x1521 = false;
}
int32_t x1526 = x1482 / 1;
int32_t x1527 = x1526 + 1;
int32_t x1531 = 8192 * x1527;
int32_t x1532 = x1531 * x1527;
int32_t x1528 = x1527 * x1527;
int32_t x1529 = 128 * x1528;
int32_t x1530 = 64 * x1529;
int32_t x1551 = x1527 + 2;
bool x1552 = x1551 >= 3;
bool x1553;
if (x1552) {
x1553 = x1552;
} else {
x1553 = false;
}
int32_t x1558 = x1551 - 3;
int32_t x1559 = x1558 / 2;
int32_t x1560 = x1559 + 1;
int32_t x1564 = 8192 * x1560;
int32_t x1565 = x1564 * x1560;
int32_t x1561 = x1560 * x1560;
int32_t x1562 = 128 * x1561;
int32_t x1563 = 64 * x1562;
bool x1584 = x1560 >= 1;
bool x1585;
if (x1584) {
x1585 = x1584;
} else {
x1585 = false;
}
int32_t x1590 = x1559 / 1;
int32_t x1591 = x1590 + 1;
int32_t x1595 = 32768 * x1591;
int32_t x1596 = x1595 * x1591;
int32_t x1592 = x1591 * x1591;
int32_t x1593 = 512 * x1592;
int32_t x1594 = 64 * x1593;
int32_t x1612 = x1482 / 2;
int32_t x1613 = x1612 + 1;
int32_t x1617 = 32768 * x1613;
int32_t x1618 = x1617 * x1613;
int32_t x1614 = x1613 * x1613;
int32_t x1615 = 512 * x1614;
int32_t x1616 = 64 * x1615;
bool x1631 = x1613 == 1;
bool x1632 = x1613 == x1591;
bool x1633 = x1631 || x1632;
bool x1634;
if (x1633) {
x1634 = x1633;
} else {
x1634 = false;
}
bool x1650 = x1591 >= 1;
bool x1651;
if (x1650) {
x1651 = x1650;
} else {
x1651 = false;
}
int32_t x1656 = x1590 / 1;
int32_t x1657 = x1656 + 1;
int32_t x1661 = 8192 * x1657;
int32_t x1662 = x1661 * x1657;
int32_t x1658 = x1657 * x1657;
int32_t x1659 = 128 * x1658;
int32_t x1660 = 64 * x1659;
int32_t x1681 = x1657 + 2;
bool x1682 = x1681 >= 3;
bool x1683;
if (x1682) {
x1683 = x1682;
} else {
x1683 = false;
}
int32_t x1688 = x1681 - 3;
int32_t x1689 = x1688 / 1;
int32_t x1690 = x1689 + 1;
int32_t x1694 = 8192 * x1690;
int32_t x1695 = x1694 * x1690;
int32_t x1691 = x1690 * x1690;
int32_t x1692 = 128 * x1691;
int32_t x1693 = 64 * x1692;
bool x1714 = x1690 >= 1;
bool x1715;
if (x1714) {
x1715 = x1714;
} else {
x1715 = false;
}
int32_t x1720 = x1689 / 1;
int32_t x1721 = x1720 + 1;
int32_t x1725 = 32768 * x1721;
int32_t x1726 = x1725 * x1721;
int32_t x1722 = x1721 * x1721;
int32_t x1723 = 512 * x1722;
int32_t x1724 = 64 * x1723;
bool x1739 = x1591 == 1;
bool x1740 = x1591 == x1721;
bool x1741 = x1739 || x1740;
bool x1742;
if (x1741) {
x1742 = x1741;
} else {
x1742 = false;
}
bool x1758 = x1721 >= 1;
bool x1759;
if (x1758) {
x1759 = x1758;
} else {
x1759 = false;
}
int32_t x1764 = x1720 / 1;
int32_t x1765 = x1764 + 1;
int32_t x1769 = 8192 * x1765;
int32_t x1770 = x1769 * x1765;
int32_t x1766 = x1765 * x1765;
int32_t x1767 = 128 * x1766;
int32_t x1768 = 64 * x1767;
int32_t x1789 = x1765 + 2;
bool x1790 = x1789 >= 3;
bool x1791;
if (x1790) {
x1791 = x1790;
} else {
x1791 = false;
}
int32_t x1796 = x1789 - 3;
int32_t x1797 = x1796 / 1;
int32_t x1798 = x1797 + 1;
int32_t x1802 = 8192 * x1798;
int32_t x1803 = x1802 * x1798;
int32_t x1799 = x1798 * x1798;
int32_t x1800 = 128 * x1799;
int32_t x1801 = 64 * x1800;
bool x1822 = x1798 >= 1;
bool x1823;
if (x1822) {
x1823 = x1822;
} else {
x1823 = false;
}
int32_t x1828 = x1797 / 1;
int32_t x1829 = x1828 + 1;
int32_t x1833 = 32768 * x1829;
int32_t x1834 = x1833 * x1829;
int32_t x1830 = x1829 * x1829;
int32_t x1831 = 512 * x1830;
int32_t x1832 = 64 * x1831;
bool x1847 = x1721 == 1;
bool x1848 = x1721 == x1829;
bool x1849 = x1847 || x1848;
bool x1850;
if (x1849) {
x1850 = x1849;
} else {
x1850 = false;
}
bool x1866 = x1829 >= 1;
bool x1867;
if (x1866) {
x1867 = x1866;
} else {
x1867 = false;
}
int32_t x1872 = x1828 / 1;
int32_t x1873 = x1872 + 1;
int32_t x1877 = 8192 * x1873;
int32_t x1878 = x1877 * x1873;
int32_t x1874 = x1873 * x1873;
int32_t x1875 = 128 * x1874;
int32_t x1876 = 64 * x1875;
int32_t x1897 = x1873 + 2;
bool x1898 = x1897 >= 3;
bool x1899;
if (x1898) {
x1899 = x1898;
} else {
x1899 = false;
}
int32_t x1904 = x1897 - 3;
int32_t x1905 = x1904 / 1;
int32_t x1906 = x1905 + 1;
int32_t x1910 = 8192 * x1906;
int32_t x1911 = x1910 * x1906;
int32_t x1907 = x1906 * x1906;
int32_t x1908 = 128 * x1907;
int32_t x1909 = 64 * x1908;
bool x1930 = x1906 >= 1;
bool x1931;
if (x1930) {
x1931 = x1930;
} else {
x1931 = false;
}
int32_t x1936 = x1905 / 1;
int32_t x1937 = x1936 + 1;
int32_t x1941 = 32768 * x1937;
int32_t x1942 = x1941 * x1937;
int32_t x1938 = x1937 * x1937;
int32_t x1939 = 512 * x1938;
int32_t x1940 = 64 * x1939;
bool x1955 = x1829 == 1;
bool x1956 = x1829 == x1937;
bool x1957 = x1955 || x1956;
bool x1958;
if (x1957) {
x1958 = x1957;
} else {
x1958 = false;
}
bool x1974 = x1937 >= 1;
bool x1975;
if (x1974) {
x1975 = x1974;
} else {
x1975 = false;
}
int32_t x1980 = x1936 / 1;
int32_t x1981 = x1980 + 1;
int32_t x1985 = 16384 * x1981;
int32_t x1986 = x1985 * x1981;
int32_t x1982 = x1981 * x1981;
int32_t x1983 = 256 * x1982;
int32_t x1984 = 64 * x1983;
int32_t x2005 = x1981 + 2;
bool x2006 = x2005 >= 3;
bool x2007;
if (x2006) {
x2007 = x2006;
} else {
x2007 = false;
}
int32_t x2012 = x2005 - 3;
int32_t x2013 = x2012 / 2;
int32_t x2014 = x2013 + 1;
int32_t x2018 = 16384 * x2014;
int32_t x2019 = x2018 * x2014;
int32_t x2015 = x2014 * x2014;
int32_t x2016 = 256 * x2015;
int32_t x2017 = 64 * x2016;
bool x2038 = x2014 >= 1;
bool x2039;
if (x2038) {
x2039 = x2038;
} else {
x2039 = false;
}
int32_t x2044 = x2013 / 1;
int32_t x2045 = x2044 + 1;
int32_t x2049 = 65536 * x2045;
int32_t x2050 = x2049 * x2045;
int32_t x2046 = x2045 * x2045;
int32_t x2047 = 1024 * x2046;
int32_t x2048 = 64 * x2047;
int32_t x2066 = x1936 / 2;
int32_t x2067 = x2066 + 1;
int32_t x2071 = 65536 * x2067;
int32_t x2072 = x2071 * x2067;
int32_t x2068 = x2067 * x2067;
int32_t x2069 = 1024 * x2068;
int32_t x2070 = 64 * x2069;
bool x2085 = x2067 == 1;
bool x2086 = x2067 == x2045;
bool x2087 = x2085 || x2086;
bool x2088;
if (x2087) {
x2088 = x2087;
} else {
x2088 = false;
}
bool x2104 = x2045 >= 1;
bool x2105;
if (x2104) {
x2105 = x2104;
} else {
x2105 = false;
}
int32_t x2110 = x2044 / 1;
int32_t x2111 = x2110 + 1;
int32_t x2115 = 16384 * x2111;
int32_t x2116 = x2115 * x2111;
int32_t x2112 = x2111 * x2111;
int32_t x2113 = 256 * x2112;
int32_t x2114 = 64 * x2113;
int32_t x2135 = x2111 + 2;
bool x2136 = x2135 >= 3;
bool x2137;
if (x2136) {
x2137 = x2136;
} else {
x2137 = false;
}
int32_t x2142 = x2135 - 3;
int32_t x2143 = x2142 / 1;
int32_t x2144 = x2143 + 1;
int32_t x2148 = 16384 * x2144;
int32_t x2149 = x2148 * x2144;
int32_t x2145 = x2144 * x2144;
int32_t x2146 = 256 * x2145;
int32_t x2147 = 64 * x2146;
bool x2168 = x2144 >= 1;
bool x2169;
if (x2168) {
x2169 = x2168;
} else {
x2169 = false;
}
int32_t x2174 = x2143 / 1;
int32_t x2175 = x2174 + 1;
int32_t x2179 = 65536 * x2175;
int32_t x2180 = x2179 * x2175;
int32_t x2176 = x2175 * x2175;
int32_t x2177 = 1024 * x2176;
int32_t x2178 = 64 * x2177;
bool x2193 = x2045 == 1;
bool x2194 = x2045 == x2175;
bool x2195 = x2193 || x2194;
bool x2196;
if (x2195) {
x2196 = x2195;
} else {
x2196 = false;
}
bool x2212 = x2175 >= 1;
bool x2213;
if (x2212) {
x2213 = x2212;
} else {
x2213 = false;
}
int32_t x2218 = x2174 / 1;
int32_t x2219 = x2218 + 1;
int32_t x2223 = 16384 * x2219;
int32_t x2224 = x2223 * x2219;
int32_t x2220 = x2219 * x2219;
int32_t x2221 = 256 * x2220;
int32_t x2222 = 64 * x2221;
int32_t x2243 = x2219 + 2;
bool x2244 = x2243 >= 3;
bool x2245;
if (x2244) {
x2245 = x2244;
} else {
x2245 = false;
}
int32_t x2250 = x2243 - 3;
int32_t x2251 = x2250 / 1;
int32_t x2252 = x2251 + 1;
int32_t x2256 = 16384 * x2252;
int32_t x2257 = x2256 * x2252;
int32_t x2253 = x2252 * x2252;
int32_t x2254 = 256 * x2253;
int32_t x2255 = 64 * x2254;
bool x2276 = x2252 >= 1;
bool x2277;
if (x2276) {
x2277 = x2276;
} else {
x2277 = false;
}
int32_t x2282 = x2251 / 1;
int32_t x2283 = x2282 + 1;
int32_t x2287 = 65536 * x2283;
int32_t x2288 = x2287 * x2283;
int32_t x2284 = x2283 * x2283;
int32_t x2285 = 1024 * x2284;
int32_t x2286 = 64 * x2285;
bool x2301 = x2175 == 1;
bool x2302 = x2175 == x2283;
bool x2303 = x2301 || x2302;
bool x2304;
if (x2303) {
x2304 = x2303;
} else {
x2304 = false;
}
bool x2320 = x2283 >= 1;
bool x2321;
if (x2320) {
x2321 = x2320;
} else {
x2321 = false;
}
int32_t x2326 = x2282 / 1;
int32_t x2327 = x2326 + 1;
int32_t x2331 = 16384 * x2327;
int32_t x2332 = x2331 * x2327;
int32_t x2328 = x2327 * x2327;
int32_t x2329 = 256 * x2328;
int32_t x2330 = 64 * x2329;
int32_t x2351 = x2327 + 2;
bool x2352 = x2351 >= 3;
bool x2353;
if (x2352) {
x2353 = x2352;
} else {
x2353 = false;
}
int32_t x2358 = x2351 - 3;
int32_t x2359 = x2358 / 1;
int32_t x2360 = x2359 + 1;
int32_t x2364 = 16384 * x2360;
int32_t x2365 = x2364 * x2360;
int32_t x2361 = x2360 * x2360;
int32_t x2362 = 256 * x2361;
int32_t x2363 = 64 * x2362;
bool x2384 = x2360 >= 1;
bool x2385;
if (x2384) {
x2385 = x2384;
} else {
x2385 = false;
}
int32_t x2390 = x2359 / 1;
int32_t x2391 = x2390 + 1;
int32_t x2395 = 65536 * x2391;
int32_t x2396 = x2395 * x2391;
int32_t x2392 = x2391 * x2391;
int32_t x2393 = 1024 * x2392;
int32_t x2394 = 64 * x2393;
bool x2409 = x2283 == 1;
bool x2410 = x2283 == x2391;
bool x2411 = x2409 || x2410;
bool x2412;
if (x2411) {
x2412 = x2411;
} else {
x2412 = false;
}
bool x2428 = x2391 >= 1;
bool x2429;
if (x2428) {
x2429 = x2428;
} else {
x2429 = false;
}
int32_t x2434 = x2390 / 1;
int32_t x2435 = x2434 + 1;
int32_t x2439 = 16384 * x2435;
int32_t x2440 = x2439 * x2435;
int32_t x2436 = x2435 * x2435;
int32_t x2437 = 256 * x2436;
int32_t x2438 = 64 * x2437;
int32_t x2459 = x2435 + 2;
bool x2460 = x2459 >= 3;
bool x2461;
if (x2460) {
x2461 = x2460;
} else {
x2461 = false;
}
int32_t x2466 = x2459 - 3;
int32_t x2467 = x2466 / 1;
int32_t x2468 = x2467 + 1;
int32_t x2472 = 16384 * x2468;
int32_t x2473 = x2472 * x2468;
int32_t x2469 = x2468 * x2468;
int32_t x2470 = 256 * x2469;
int32_t x2471 = 64 * x2470;
bool x2492 = x2468 >= 1;
bool x2493;
if (x2492) {
x2493 = x2492;
} else {
x2493 = false;
}
int32_t x2498 = x2467 / 1;
int32_t x2499 = x2498 + 1;
int32_t x2503 = 65536 * x2499;
int32_t x2504 = x2503 * x2499;
int32_t x2500 = x2499 * x2499;
int32_t x2501 = 1024 * x2500;
int32_t x2502 = 64 * x2501;
bool x2517 = x2391 == 1;
bool x2518 = x2391 == x2499;
bool x2519 = x2517 || x2518;
bool x2520;
if (x2519) {
x2520 = x2519;
} else {
x2520 = false;
}
bool x2536 = x2499 >= 1;
bool x2537;
if (x2536) {
x2537 = x2536;
} else {
x2537 = false;
}
int32_t x2542 = x2498 / 1;
int32_t x2543 = x2542 + 1;
int32_t x2547 = 16384 * x2543;
int32_t x2548 = x2547 * x2543;
int32_t x2544 = x2543 * x2543;
int32_t x2545 = 256 * x2544;
int32_t x2546 = 64 * x2545;
int32_t x2567 = x2543 + 2;
bool x2568 = x2567 >= 3;
bool x2569;
if (x2568) {
x2569 = x2568;
} else {
x2569 = false;
}
int32_t x2574 = x2567 - 3;
int32_t x2575 = x2574 / 1;
int32_t x2576 = x2575 + 1;
int32_t x2580 = 16384 * x2576;
int32_t x2581 = x2580 * x2576;
int32_t x2577 = x2576 * x2576;
int32_t x2578 = 256 * x2577;
int32_t x2579 = 64 * x2578;
bool x2600 = x2576 >= 1;
bool x2601;
if (x2600) {
x2601 = x2600;
} else {
x2601 = false;
}
int32_t x2606 = x2575 / 1;
int32_t x2607 = x2606 + 1;
int32_t x2611 = 65536 * x2607;
int32_t x2612 = x2611 * x2607;
int32_t x2608 = x2607 * x2607;
int32_t x2609 = 1024 * x2608;
int32_t x2610 = 64 * x2609;
bool x2625 = x2499 == 1;
bool x2626 = x2499 == x2607;
bool x2627 = x2625 || x2626;
bool x2628;
if (x2627) {
x2628 = x2627;
} else {
x2628 = false;
}
bool x2644 = x2607 >= 1;
bool x2645;
if (x2644) {
x2645 = x2644;
} else {
x2645 = false;
}
int32_t x2650 = x2606 / 1;
int32_t x2651 = x2650 + 1;
int32_t x2655 = 32768 * x2651;
int32_t x2656 = x2655 * x2651;
int32_t x2652 = x2651 * x2651;
int32_t x2653 = 512 * x2652;
int32_t x2654 = 64 * x2653;
int32_t x2675 = x2651 + 2;
bool x2676 = x2675 >= 3;
bool x2677;
if (x2676) {
x2677 = x2676;
} else {
x2677 = false;
}
int32_t x2682 = x2675 - 3;
int32_t x2683 = x2682 / 2;
int32_t x2684 = x2683 + 1;
int32_t x2688 = 32768 * x2684;
int32_t x2689 = x2688 * x2684;
int32_t x2685 = x2684 * x2684;
int32_t x2686 = 512 * x2685;
int32_t x2687 = 64 * x2686;
bool x2708 = x2684 >= 1;
bool x2709;
if (x2708) {
x2709 = x2708;
} else {
x2709 = false;
}
int32_t x2714 = x2683 / 1;
int32_t x2715 = x2714 + 1;
int32_t x2719 = 131072 * x2715;
int32_t x2720 = x2719 * x2715;
int32_t x2716 = x2715 * x2715;
int32_t x2717 = 2048 * x2716;
int32_t x2718 = 64 * x2717;
int32_t x2736 = x2606 / 2;
int32_t x2737 = x2736 + 1;
int32_t x2741 = 131072 * x2737;
int32_t x2742 = x2741 * x2737;
int32_t x2738 = x2737 * x2737;
int32_t x2739 = 2048 * x2738;
int32_t x2740 = 64 * x2739;
bool x2755 = x2737 == 1;
bool x2756 = x2737 == x2715;
bool x2757 = x2755 || x2756;
bool x2758;
if (x2757) {
x2758 = x2757;
} else {
x2758 = false;
}
bool x2774 = x2715 >= 1;
bool x2775;
if (x2774) {
x2775 = x2774;
} else {
x2775 = false;
}
int32_t x2780 = x2714 / 1;
int32_t x2781 = x2780 + 1;
int32_t x2785 = 32768 * x2781;
int32_t x2786 = x2785 * x2781;
int32_t x2782 = x2781 * x2781;
int32_t x2783 = 512 * x2782;
int32_t x2784 = 64 * x2783;
int32_t x2805 = x2781 + 2;
bool x2806 = x2805 >= 3;
bool x2807;
if (x2806) {
x2807 = x2806;
} else {
x2807 = false;
}
int32_t x2812 = x2805 - 3;
int32_t x2813 = x2812 / 1;
int32_t x2814 = x2813 + 1;
int32_t x2818 = 32768 * x2814;
int32_t x2819 = x2818 * x2814;
int32_t x2815 = x2814 * x2814;
int32_t x2816 = 512 * x2815;
int32_t x2817 = 64 * x2816;
bool x2838 = x2814 >= 1;
bool x2839;
if (x2838) {
x2839 = x2838;
} else {
x2839 = false;
}
int32_t x2844 = x2813 / 1;
int32_t x2845 = x2844 + 1;
int32_t x2849 = 131072 * x2845;
int32_t x2850 = x2849 * x2845;
int32_t x2846 = x2845 * x2845;
int32_t x2847 = 2048 * x2846;
int32_t x2848 = 64 * x2847;
bool x2863 = x2715 == 1;
bool x2864 = x2715 == x2845;
bool x2865 = x2863 || x2864;
bool x2866;
if (x2865) {
x2866 = x2865;
} else {
x2866 = false;
}
bool x2882 = x2845 >= 1;
bool x2883;
if (x2882) {
x2883 = x2882;
} else {
x2883 = false;
}
int32_t x2888 = x2844 / 1;
int32_t x2889 = x2888 + 1;
int32_t x2893 = 32768 * x2889;
int32_t x2894 = x2893 * x2889;
int32_t x2890 = x2889 * x2889;
int32_t x2891 = 512 * x2890;
int32_t x2892 = 64 * x2891;
int32_t x2913 = x2889 + 2;
bool x2914 = x2913 >= 3;
bool x2915;
if (x2914) {
x2915 = x2914;
} else {
x2915 = false;
}
int32_t x2920 = x2913 - 3;
int32_t x2921 = x2920 / 1;
int32_t x2922 = x2921 + 1;
int32_t x2926 = 32768 * x2922;
int32_t x2927 = x2926 * x2922;
int32_t x2923 = x2922 * x2922;
int32_t x2924 = 512 * x2923;
int32_t x2925 = 64 * x2924;
bool x2946 = x2922 >= 1;
bool x2947;
if (x2946) {
x2947 = x2946;
} else {
x2947 = false;
}
int32_t x2952 = x2921 / 1;
int32_t x2953 = x2952 + 1;
int32_t x2957 = 131072 * x2953;
int32_t x2958 = x2957 * x2953;
int32_t x2954 = x2953 * x2953;
int32_t x2955 = 2048 * x2954;
int32_t x2956 = 64 * x2955;
bool x2971 = x2845 == 1;
bool x2972 = x2845 == x2953;
bool x2973 = x2971 || x2972;
bool x2974;
if (x2973) {
x2974 = x2973;
} else {
x2974 = false;
}
bool x2990 = x2953 >= 2;
bool x2991;
if (x2990) {
x2991 = x2990;
} else {
x2991 = false;
}
int32_t x3000 = x2953 - 2;
int32_t x3001 = x3000 / 1;
int32_t x3002 = x3001 + 1;
int32_t x3006 = 131072 * x3002;
int32_t x3007 = x3006 * x3002;
int32_t x3003 = x3002 * x3002;
int32_t x3004 = 2048 * x3003;
int32_t x3005 = 64 * x3004;
for(int x1106=0; x1106 < x1104; x1106++) {
int32_t x1107 = x1106 * 64;
int32_t x1108 = x1107 * 3072;
float* x1109 = x1082+x1108;
int* x1110 = x1083+x1107;
printf("input (size Const(64) x Const(3) x Const(32) x Const(32))\n");
float x1112 = 0.0f;
for(int x1114=0; x1114 < 196608; x1114++) {
float x1115 = x1112;
float x1117 = x1109[x1114];
float x1116 = fabs(x1115);
float x1118 = fabs(x1117);
bool x1119 = x1116 > x1118;
float x1122;
if (x1119) {
x1122 = x1115;
} else {
float x1120 = x1109[x1114];
x1122 = x1120;
}
x1112 = x1122;

}
float x1126 = x1112;
printf("Max Abs: %.5f || ",x1126);
for(int x1129=0; x1129 < 10; x1129++) {
float x1130 = x1109[x1129];
printf("%.5f ",x1130);

}
printf("\n");
// Tensor 'toGPU' invocation.
float* x1136 = (float*)myGpuMalloc(196608 * sizeof(float));
CUDA_CALL(cudaMemcpy(x1136, x1109, 196608 * sizeof(float), cudaMemcpyHostToDevice));
float* x1145 = (float*)myGpuMalloc(x1144 * sizeof(float));
float* x1146 = (float*)myMalloc(1 * sizeof(float));;
x1146[0] = 0.0f;
float* x1148 = (float*)myMalloc(1 * sizeof(float));;
x1148[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 3, 32, 32));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 3, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1139, x1139));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x1148, in_desc, x1136, filt_desc, x714,
    conv_desc, algo, ws_data, ws_size,
    x1146, out_desc, x1145));
};
float* x1151 = (float*)myGpuMalloc(x1142 * sizeof(float));
float* x1152 = (float*)myMalloc(1 * sizeof(float));;
x1152[0] = 0.0f;
float* x1154 = (float*)myMalloc(1 * sizeof(float));;
x1154[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1139, x1139));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1139, x1139));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1154, x1154, in_desc, x1145, out_desc, x1151, sbmv_desc, x876,
    x1011, x378, x588, 1.0E-5));
};
float* x1157 = (float*)myMalloc(1 * sizeof(float));;
x1157[0] = 0.0f;
float* x1159 = (float*)myMalloc(1 * sizeof(float));;
x1159[0] = 1.0f;
float* x1161 = (float*)myGpuMalloc(x1142 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1139, x1139));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1159, x_desc, x1151, x1157, x_desc, x1161));
};
float* x1163 = (float*)myMalloc(1 * sizeof(float));;
x1163[0] = 0.0f;
float* x1165 = (float*)myMalloc(1 * sizeof(float));;
x1165[0] = 1.0f;
float* x1175 = (float*)myGpuMalloc(x1174 * sizeof(float));

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1139, x1139) );

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1169, x1169));

cudnnPoolingDescriptor_t poolingDesc;
CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));
CUDNN_CALL(cudnnSetPooling2dDescriptor(
    poolingDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
    2, 2, 0,
    0, 2, 2
));
CUDNN_CALL(cudnnPoolingForward(
    cudnnHandle, 
    poolingDesc, 
    x1165, in_desc, x1161, x1163, out_desc, x1175));
};
if (x1178) {
} else {
assert(false && "ERROR not specified");
}
float* x1190 = (float*)myGpuMalloc(x1189 * sizeof(float));
float* x1191 = (float*)myMalloc(1 * sizeof(float));;
x1191[0] = 0.0f;
float* x1193 = (float*)myMalloc(1 * sizeof(float));;
x1193[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1169, x1169));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1184, x1184));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x1193, in_desc, x1175, filt_desc, x957,
    conv_desc, algo, ws_data, ws_size,
    x1191, out_desc, x1190));
};
float* x1196 = (float*)myGpuMalloc(x1187 * sizeof(float));
float* x1197 = (float*)myMalloc(1 * sizeof(float));;
x1197[0] = 0.0f;
float* x1199 = (float*)myMalloc(1 * sizeof(float));;
x1199[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1184, x1184));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1184, x1184));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1199, x1199, in_desc, x1190, out_desc, x1196, sbmv_desc, x336,
    x417, x600, x411, 1.0E-5));
};
float* x1202 = (float*)myMalloc(1 * sizeof(float));;
x1202[0] = 0.0f;
float* x1204 = (float*)myMalloc(1 * sizeof(float));;
x1204[0] = 1.0f;
float* x1206 = (float*)myGpuMalloc(x1187 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1184, x1184));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1204, x_desc, x1196, x1202, x_desc, x1206));
};
if (x1210) {
} else {
assert(false && "ERROR not specified");
}
float* x1223 = (float*)myGpuMalloc(x1222 * sizeof(float));
float* x1224 = (float*)myMalloc(1 * sizeof(float));;
x1224[0] = 0.0f;
float* x1226 = (float*)myMalloc(1 * sizeof(float));;
x1226[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1184, x1184));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1217, x1217));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x1226, in_desc, x1206, filt_desc, x528,
    conv_desc, algo, ws_data, ws_size,
    x1224, out_desc, x1223));
};
float* x1229 = (float*)myGpuMalloc(x1220 * sizeof(float));
float* x1230 = (float*)myMalloc(1 * sizeof(float));;
x1230[0] = 0.0f;
float* x1232 = (float*)myMalloc(1 * sizeof(float));;
x1232[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1217, x1217));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1217, x1217));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1232, x1232, in_desc, x1223, out_desc, x1229, sbmv_desc, x750,
    x405, x573, x732, 1.0E-5));
};
float* x1235 = (float*)myMalloc(1 * sizeof(float));;
x1235[0] = 0.0f;
float* x1237 = (float*)myMalloc(1 * sizeof(float));;
x1237[0] = 1.0f;
float* x1239 = (float*)myGpuMalloc(x1220 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1217, x1217));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1237, x_desc, x1229, x1235, x_desc, x1239));
};
if (x1242) {
} else {
assert(false && "ERROR not specified");
}
float* x1254 = (float*)myGpuMalloc(x1253 * sizeof(float));
float* x1255 = (float*)myMalloc(1 * sizeof(float));;
x1255[0] = 0.0f;
float* x1257 = (float*)myMalloc(1 * sizeof(float));;
x1257[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1217, x1217));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1248, x1248));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x1257, in_desc, x1239, filt_desc, x354,
    conv_desc, algo, ws_data, ws_size,
    x1255, out_desc, x1254));
};
float* x1260 = (float*)myGpuMalloc(x1251 * sizeof(float));
float* x1261 = (float*)myMalloc(1 * sizeof(float));;
x1261[0] = 0.0f;
float* x1263 = (float*)myMalloc(1 * sizeof(float));;
x1263[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1248, x1248));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1248, x1248));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1263, x1263, in_desc, x1254, out_desc, x1260, sbmv_desc, x855,
    x636, x471, x366, 1.0E-5));
};
if (x1178) {
} else {
assert(false && "ERROR not specified");
}
float* x1273 = (float*)myGpuMalloc(x1272 * sizeof(float));
float* x1274 = (float*)myMalloc(1 * sizeof(float));;
x1274[0] = 0.0f;
float* x1276 = (float*)myMalloc(1 * sizeof(float));;
x1276[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1169, x1169));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1184, x1184));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x1276, in_desc, x1175, filt_desc, x744,
    conv_desc, algo, ws_data, ws_size,
    x1274, out_desc, x1273));
};
float* x1279 = (float*)myGpuMalloc(x1270 * sizeof(float));
float* x1280 = (float*)myMalloc(1 * sizeof(float));;
x1280[0] = 0.0f;
float* x1282 = (float*)myMalloc(1 * sizeof(float));;
x1282[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1184, x1184));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1184, x1184));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1282, x1282, in_desc, x1273, out_desc, x1279, sbmv_desc, x486,
    x867, x1050, x987, 1.0E-5));
};
if (x1288) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1184) x Sym(1184), res:  x Const(64) x Const(256) x Sym(1248) x Sym(1248)");
}
float* x1293 = (float*)myMalloc(1 * sizeof(float));;
x1293[0] = 1.0f;
float* x1295 = (float*)myMalloc(1 * sizeof(float));;
x1295[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1184, x1184));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1248, x1248));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1293, bias_desc, x1279, x1295, out_desc, x1260));
};
float* x1298 = (float*)myMalloc(1 * sizeof(float));;
x1298[0] = 0.0f;
float* x1300 = (float*)myMalloc(1 * sizeof(float));;
x1300[0] = 1.0f;
float* x1302 = (float*)myGpuMalloc(x1251 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1248, x1248));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1300, x_desc, x1260, x1298, x_desc, x1302));
};
if (x1305) {
} else {
assert(false && "ERROR not specified");
}
float* x1317 = (float*)myGpuMalloc(x1316 * sizeof(float));
float* x1318 = (float*)myMalloc(1 * sizeof(float));;
x1318[0] = 0.0f;
float* x1320 = (float*)myMalloc(1 * sizeof(float));;
x1320[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1248, x1248));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1311, x1311));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x1320, in_desc, x1302, filt_desc, x771,
    conv_desc, algo, ws_data, ws_size,
    x1318, out_desc, x1317));
};
float* x1323 = (float*)myGpuMalloc(x1314 * sizeof(float));
float* x1324 = (float*)myMalloc(1 * sizeof(float));;
x1324[0] = 0.0f;
float* x1326 = (float*)myMalloc(1 * sizeof(float));;
x1326[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1311, x1311));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1311, x1311));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1326, x1326, in_desc, x1317, out_desc, x1323, sbmv_desc, x684,
    x438, x288, x564, 1.0E-5));
};
float* x1329 = (float*)myMalloc(1 * sizeof(float));;
x1329[0] = 0.0f;
float* x1331 = (float*)myMalloc(1 * sizeof(float));;
x1331[0] = 1.0f;
float* x1333 = (float*)myGpuMalloc(x1314 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1311, x1311));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1331, x_desc, x1323, x1329, x_desc, x1333));
};
if (x1337) {
} else {
assert(false && "ERROR not specified");
}
float* x1350 = (float*)myGpuMalloc(x1349 * sizeof(float));
float* x1351 = (float*)myMalloc(1 * sizeof(float));;
x1351[0] = 0.0f;
float* x1353 = (float*)myMalloc(1 * sizeof(float));;
x1353[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1311, x1311));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1344, x1344));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x1353, in_desc, x1333, filt_desc, x507,
    conv_desc, algo, ws_data, ws_size,
    x1351, out_desc, x1350));
};
float* x1356 = (float*)myGpuMalloc(x1347 * sizeof(float));
float* x1357 = (float*)myMalloc(1 * sizeof(float));;
x1357[0] = 0.0f;
float* x1359 = (float*)myMalloc(1 * sizeof(float));;
x1359[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1344, x1344));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1344, x1344));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1359, x1359, in_desc, x1350, out_desc, x1356, sbmv_desc, x882,
    x717, x390, x990, 1.0E-5));
};
float* x1362 = (float*)myMalloc(1 * sizeof(float));;
x1362[0] = 0.0f;
float* x1364 = (float*)myMalloc(1 * sizeof(float));;
x1364[0] = 1.0f;
float* x1366 = (float*)myGpuMalloc(x1347 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1344, x1344));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1364, x_desc, x1356, x1362, x_desc, x1366));
};
if (x1369) {
} else {
assert(false && "ERROR not specified");
}
float* x1381 = (float*)myGpuMalloc(x1380 * sizeof(float));
float* x1382 = (float*)myMalloc(1 * sizeof(float));;
x1382[0] = 0.0f;
float* x1384 = (float*)myMalloc(1 * sizeof(float));;
x1384[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1344, x1344));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1375, x1375));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x1384, in_desc, x1366, filt_desc, x648,
    conv_desc, algo, ws_data, ws_size,
    x1382, out_desc, x1381));
};
float* x1387 = (float*)myGpuMalloc(x1378 * sizeof(float));
float* x1388 = (float*)myMalloc(1 * sizeof(float));;
x1388[0] = 0.0f;
float* x1390 = (float*)myMalloc(1 * sizeof(float));;
x1390[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1375, x1375));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1375, x1375));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1390, x1390, in_desc, x1381, out_desc, x1387, sbmv_desc, x432,
    x279, x531, x756, 1.0E-5));
};
if (x1396) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1248) x Sym(1248), res:  x Const(64) x Const(256) x Sym(1375) x Sym(1375)");
}
float* x1401 = (float*)myMalloc(1 * sizeof(float));;
x1401[0] = 1.0f;
float* x1403 = (float*)myMalloc(1 * sizeof(float));;
x1403[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1248, x1248));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1375, x1375));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1401, bias_desc, x1302, x1403, out_desc, x1387));
};
float* x1406 = (float*)myMalloc(1 * sizeof(float));;
x1406[0] = 0.0f;
float* x1408 = (float*)myMalloc(1 * sizeof(float));;
x1408[0] = 1.0f;
float* x1410 = (float*)myGpuMalloc(x1378 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1375, x1375));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1408, x_desc, x1387, x1406, x_desc, x1410));
};
if (x1413) {
} else {
assert(false && "ERROR not specified");
}
float* x1425 = (float*)myGpuMalloc(x1424 * sizeof(float));
float* x1426 = (float*)myMalloc(1 * sizeof(float));;
x1426[0] = 0.0f;
float* x1428 = (float*)myMalloc(1 * sizeof(float));;
x1428[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1375, x1375));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1419, x1419));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x1428, in_desc, x1410, filt_desc, x708,
    conv_desc, algo, ws_data, ws_size,
    x1426, out_desc, x1425));
};
float* x1431 = (float*)myGpuMalloc(x1422 * sizeof(float));
float* x1432 = (float*)myMalloc(1 * sizeof(float));;
x1432[0] = 0.0f;
float* x1434 = (float*)myMalloc(1 * sizeof(float));;
x1434[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1419, x1419));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1419, x1419));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1434, x1434, in_desc, x1425, out_desc, x1431, sbmv_desc, x501,
    x330, x1029, x819, 1.0E-5));
};
float* x1437 = (float*)myMalloc(1 * sizeof(float));;
x1437[0] = 0.0f;
float* x1439 = (float*)myMalloc(1 * sizeof(float));;
x1439[0] = 1.0f;
float* x1441 = (float*)myGpuMalloc(x1422 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1419, x1419));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1439, x_desc, x1431, x1437, x_desc, x1441));
};
if (x1445) {
} else {
assert(false && "ERROR not specified");
}
float* x1458 = (float*)myGpuMalloc(x1457 * sizeof(float));
float* x1459 = (float*)myMalloc(1 * sizeof(float));;
x1459[0] = 0.0f;
float* x1461 = (float*)myMalloc(1 * sizeof(float));;
x1461[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1419, x1419));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1452, x1452));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x1461, in_desc, x1441, filt_desc, x477,
    conv_desc, algo, ws_data, ws_size,
    x1459, out_desc, x1458));
};
float* x1464 = (float*)myGpuMalloc(x1455 * sizeof(float));
float* x1465 = (float*)myMalloc(1 * sizeof(float));;
x1465[0] = 0.0f;
float* x1467 = (float*)myMalloc(1 * sizeof(float));;
x1467[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1452, x1452));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1452, x1452));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1467, x1467, in_desc, x1458, out_desc, x1464, sbmv_desc, x474,
    x663, x795, x612, 1.0E-5));
};
float* x1470 = (float*)myMalloc(1 * sizeof(float));;
x1470[0] = 0.0f;
float* x1472 = (float*)myMalloc(1 * sizeof(float));;
x1472[0] = 1.0f;
float* x1474 = (float*)myGpuMalloc(x1455 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1452, x1452));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1472, x_desc, x1464, x1470, x_desc, x1474));
};
if (x1477) {
} else {
assert(false && "ERROR not specified");
}
float* x1489 = (float*)myGpuMalloc(x1488 * sizeof(float));
float* x1490 = (float*)myMalloc(1 * sizeof(float));;
x1490[0] = 0.0f;
float* x1492 = (float*)myMalloc(1 * sizeof(float));;
x1492[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1452, x1452));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1483, x1483));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x1492, in_desc, x1474, filt_desc, x519,
    conv_desc, algo, ws_data, ws_size,
    x1490, out_desc, x1489));
};
float* x1495 = (float*)myGpuMalloc(x1486 * sizeof(float));
float* x1496 = (float*)myMalloc(1 * sizeof(float));;
x1496[0] = 0.0f;
float* x1498 = (float*)myMalloc(1 * sizeof(float));;
x1498[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1483, x1483));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1483, x1483));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1498, x1498, in_desc, x1489, out_desc, x1495, sbmv_desc, x369,
    x999, x810, x657, 1.0E-5));
};
if (x1504) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1375) x Sym(1375), res:  x Const(64) x Const(256) x Sym(1483) x Sym(1483)");
}
float* x1509 = (float*)myMalloc(1 * sizeof(float));;
x1509[0] = 1.0f;
float* x1511 = (float*)myMalloc(1 * sizeof(float));;
x1511[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1375, x1375));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1483, x1483));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1509, bias_desc, x1410, x1511, out_desc, x1495));
};
float* x1514 = (float*)myMalloc(1 * sizeof(float));;
x1514[0] = 0.0f;
float* x1516 = (float*)myMalloc(1 * sizeof(float));;
x1516[0] = 1.0f;
float* x1518 = (float*)myGpuMalloc(x1486 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1483, x1483));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1516, x_desc, x1495, x1514, x_desc, x1518));
};
if (x1521) {
} else {
assert(false && "ERROR not specified");
}
float* x1533 = (float*)myGpuMalloc(x1532 * sizeof(float));
float* x1534 = (float*)myMalloc(1 * sizeof(float));;
x1534[0] = 0.0f;
float* x1536 = (float*)myMalloc(1 * sizeof(float));;
x1536[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1483, x1483));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1527, x1527));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x1536, in_desc, x1518, filt_desc, x291,
    conv_desc, algo, ws_data, ws_size,
    x1534, out_desc, x1533));
};
float* x1539 = (float*)myGpuMalloc(x1530 * sizeof(float));
float* x1540 = (float*)myMalloc(1 * sizeof(float));;
x1540[0] = 0.0f;
float* x1542 = (float*)myMalloc(1 * sizeof(float));;
x1542[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1527, x1527));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1527, x1527));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1542, x1542, in_desc, x1533, out_desc, x1539, sbmv_desc, x510,
    x774, x870, x660, 1.0E-5));
};
float* x1545 = (float*)myMalloc(1 * sizeof(float));;
x1545[0] = 0.0f;
float* x1547 = (float*)myMalloc(1 * sizeof(float));;
x1547[0] = 1.0f;
float* x1549 = (float*)myGpuMalloc(x1530 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1527, x1527));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1547, x_desc, x1539, x1545, x_desc, x1549));
};
if (x1553) {
} else {
assert(false && "ERROR not specified");
}
float* x1566 = (float*)myGpuMalloc(x1565 * sizeof(float));
float* x1567 = (float*)myMalloc(1 * sizeof(float));;
x1567[0] = 0.0f;
float* x1569 = (float*)myMalloc(1 * sizeof(float));;
x1569[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1527, x1527));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1560, x1560));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 2, 2, 1, 1,
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
    x1569, in_desc, x1549, filt_desc, x339,
    conv_desc, algo, ws_data, ws_size,
    x1567, out_desc, x1566));
};
float* x1572 = (float*)myGpuMalloc(x1563 * sizeof(float));
float* x1573 = (float*)myMalloc(1 * sizeof(float));;
x1573[0] = 0.0f;
float* x1575 = (float*)myMalloc(1 * sizeof(float));;
x1575[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1560, x1560));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1560, x1560));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1575, x1575, in_desc, x1566, out_desc, x1572, sbmv_desc, x1014,
    x828, x642, x387, 1.0E-5));
};
float* x1578 = (float*)myMalloc(1 * sizeof(float));;
x1578[0] = 0.0f;
float* x1580 = (float*)myMalloc(1 * sizeof(float));;
x1580[0] = 1.0f;
float* x1582 = (float*)myGpuMalloc(x1563 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1560, x1560));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1580, x_desc, x1572, x1578, x_desc, x1582));
};
if (x1585) {
} else {
assert(false && "ERROR not specified");
}
float* x1597 = (float*)myGpuMalloc(x1596 * sizeof(float));
float* x1598 = (float*)myMalloc(1 * sizeof(float));;
x1598[0] = 0.0f;
float* x1600 = (float*)myMalloc(1 * sizeof(float));;
x1600[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1560, x1560));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1591, x1591));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x1600, in_desc, x1582, filt_desc, x576,
    conv_desc, algo, ws_data, ws_size,
    x1598, out_desc, x1597));
};
float* x1603 = (float*)myGpuMalloc(x1594 * sizeof(float));
float* x1604 = (float*)myMalloc(1 * sizeof(float));;
x1604[0] = 0.0f;
float* x1606 = (float*)myMalloc(1 * sizeof(float));;
x1606[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1591, x1591));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1591, x1591));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1606, x1606, in_desc, x1597, out_desc, x1603, sbmv_desc, x693,
    x888, x705, x561, 1.0E-5));
};
if (x1521) {
} else {
assert(false && "ERROR not specified");
}
float* x1619 = (float*)myGpuMalloc(x1618 * sizeof(float));
float* x1620 = (float*)myMalloc(1 * sizeof(float));;
x1620[0] = 0.0f;
float* x1622 = (float*)myMalloc(1 * sizeof(float));;
x1622[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1483, x1483));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1613, x1613));

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
    x1622, in_desc, x1518, filt_desc, x1032,
    conv_desc, algo, ws_data, ws_size,
    x1620, out_desc, x1619));
};
float* x1625 = (float*)myGpuMalloc(x1616 * sizeof(float));
float* x1626 = (float*)myMalloc(1 * sizeof(float));;
x1626[0] = 0.0f;
float* x1628 = (float*)myMalloc(1 * sizeof(float));;
x1628[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1613, x1613));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1613, x1613));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1628, x1628, in_desc, x1619, out_desc, x1625, sbmv_desc, x879,
    x615, x384, x327, 1.0E-5));
};
if (x1634) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(1613) x Sym(1613), res:  x Const(64) x Const(512) x Sym(1591) x Sym(1591)");
}
float* x1639 = (float*)myMalloc(1 * sizeof(float));;
x1639[0] = 1.0f;
float* x1641 = (float*)myMalloc(1 * sizeof(float));;
x1641[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1613, x1613));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1591, x1591));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1639, bias_desc, x1625, x1641, out_desc, x1603));
};
float* x1644 = (float*)myMalloc(1 * sizeof(float));;
x1644[0] = 0.0f;
float* x1646 = (float*)myMalloc(1 * sizeof(float));;
x1646[0] = 1.0f;
float* x1648 = (float*)myGpuMalloc(x1594 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1591, x1591));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1646, x_desc, x1603, x1644, x_desc, x1648));
};
if (x1651) {
} else {
assert(false && "ERROR not specified");
}
float* x1663 = (float*)myGpuMalloc(x1662 * sizeof(float));
float* x1664 = (float*)myMalloc(1 * sizeof(float));;
x1664[0] = 0.0f;
float* x1666 = (float*)myMalloc(1 * sizeof(float));;
x1666[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1591, x1591));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1657, x1657));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x1666, in_desc, x1648, filt_desc, x1026,
    conv_desc, algo, ws_data, ws_size,
    x1664, out_desc, x1663));
};
float* x1669 = (float*)myGpuMalloc(x1660 * sizeof(float));
float* x1670 = (float*)myMalloc(1 * sizeof(float));;
x1670[0] = 0.0f;
float* x1672 = (float*)myMalloc(1 * sizeof(float));;
x1672[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1657, x1657));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1657, x1657));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1672, x1672, in_desc, x1663, out_desc, x1669, sbmv_desc, x924,
    x309, x558, x789, 1.0E-5));
};
float* x1675 = (float*)myMalloc(1 * sizeof(float));;
x1675[0] = 0.0f;
float* x1677 = (float*)myMalloc(1 * sizeof(float));;
x1677[0] = 1.0f;
float* x1679 = (float*)myGpuMalloc(x1660 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1657, x1657));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1677, x_desc, x1669, x1675, x_desc, x1679));
};
if (x1683) {
} else {
assert(false && "ERROR not specified");
}
float* x1696 = (float*)myGpuMalloc(x1695 * sizeof(float));
float* x1697 = (float*)myMalloc(1 * sizeof(float));;
x1697[0] = 0.0f;
float* x1699 = (float*)myMalloc(1 * sizeof(float));;
x1699[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1657, x1657));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1690, x1690));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x1699, in_desc, x1679, filt_desc, x963,
    conv_desc, algo, ws_data, ws_size,
    x1697, out_desc, x1696));
};
float* x1702 = (float*)myGpuMalloc(x1693 * sizeof(float));
float* x1703 = (float*)myMalloc(1 * sizeof(float));;
x1703[0] = 0.0f;
float* x1705 = (float*)myMalloc(1 * sizeof(float));;
x1705[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1690, x1690));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1690, x1690));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1705, x1705, in_desc, x1696, out_desc, x1702, sbmv_desc, x282,
    x543, x363, x933, 1.0E-5));
};
float* x1708 = (float*)myMalloc(1 * sizeof(float));;
x1708[0] = 0.0f;
float* x1710 = (float*)myMalloc(1 * sizeof(float));;
x1710[0] = 1.0f;
float* x1712 = (float*)myGpuMalloc(x1693 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1690, x1690));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1710, x_desc, x1702, x1708, x_desc, x1712));
};
if (x1715) {
} else {
assert(false && "ERROR not specified");
}
float* x1727 = (float*)myGpuMalloc(x1726 * sizeof(float));
float* x1728 = (float*)myMalloc(1 * sizeof(float));;
x1728[0] = 0.0f;
float* x1730 = (float*)myMalloc(1 * sizeof(float));;
x1730[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1690, x1690));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1721, x1721));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x1730, in_desc, x1712, filt_desc, x591,
    conv_desc, algo, ws_data, ws_size,
    x1728, out_desc, x1727));
};
float* x1733 = (float*)myGpuMalloc(x1724 * sizeof(float));
float* x1734 = (float*)myMalloc(1 * sizeof(float));;
x1734[0] = 0.0f;
float* x1736 = (float*)myMalloc(1 * sizeof(float));;
x1736[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1721, x1721));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1721, x1721));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1736, x1736, in_desc, x1727, out_desc, x1733, sbmv_desc, x414,
    x996, x699, x522, 1.0E-5));
};
if (x1742) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(1591) x Sym(1591), res:  x Const(64) x Const(512) x Sym(1721) x Sym(1721)");
}
float* x1747 = (float*)myMalloc(1 * sizeof(float));;
x1747[0] = 1.0f;
float* x1749 = (float*)myMalloc(1 * sizeof(float));;
x1749[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1591, x1591));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1721, x1721));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1747, bias_desc, x1648, x1749, out_desc, x1733));
};
float* x1752 = (float*)myMalloc(1 * sizeof(float));;
x1752[0] = 0.0f;
float* x1754 = (float*)myMalloc(1 * sizeof(float));;
x1754[0] = 1.0f;
float* x1756 = (float*)myGpuMalloc(x1724 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1721, x1721));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1754, x_desc, x1733, x1752, x_desc, x1756));
};
if (x1759) {
} else {
assert(false && "ERROR not specified");
}
float* x1771 = (float*)myGpuMalloc(x1770 * sizeof(float));
float* x1772 = (float*)myMalloc(1 * sizeof(float));;
x1772[0] = 0.0f;
float* x1774 = (float*)myMalloc(1 * sizeof(float));;
x1774[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1721, x1721));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1765, x1765));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x1774, in_desc, x1756, filt_desc, x846,
    conv_desc, algo, ws_data, ws_size,
    x1772, out_desc, x1771));
};
float* x1777 = (float*)myGpuMalloc(x1768 * sizeof(float));
float* x1778 = (float*)myMalloc(1 * sizeof(float));;
x1778[0] = 0.0f;
float* x1780 = (float*)myMalloc(1 * sizeof(float));;
x1780[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1765, x1765));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1765, x1765));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1780, x1780, in_desc, x1771, out_desc, x1777, sbmv_desc, x393,
    x768, x594, x285, 1.0E-5));
};
float* x1783 = (float*)myMalloc(1 * sizeof(float));;
x1783[0] = 0.0f;
float* x1785 = (float*)myMalloc(1 * sizeof(float));;
x1785[0] = 1.0f;
float* x1787 = (float*)myGpuMalloc(x1768 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1765, x1765));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1785, x_desc, x1777, x1783, x_desc, x1787));
};
if (x1791) {
} else {
assert(false && "ERROR not specified");
}
float* x1804 = (float*)myGpuMalloc(x1803 * sizeof(float));
float* x1805 = (float*)myMalloc(1 * sizeof(float));;
x1805[0] = 0.0f;
float* x1807 = (float*)myMalloc(1 * sizeof(float));;
x1807[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1765, x1765));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1798, x1798));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x1807, in_desc, x1787, filt_desc, x831,
    conv_desc, algo, ws_data, ws_size,
    x1805, out_desc, x1804));
};
float* x1810 = (float*)myGpuMalloc(x1801 * sizeof(float));
float* x1811 = (float*)myMalloc(1 * sizeof(float));;
x1811[0] = 0.0f;
float* x1813 = (float*)myMalloc(1 * sizeof(float));;
x1813[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1798, x1798));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1798, x1798));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1813, x1813, in_desc, x1804, out_desc, x1810, sbmv_desc, x639,
    x441, x909, x1056, 1.0E-5));
};
float* x1816 = (float*)myMalloc(1 * sizeof(float));;
x1816[0] = 0.0f;
float* x1818 = (float*)myMalloc(1 * sizeof(float));;
x1818[0] = 1.0f;
float* x1820 = (float*)myGpuMalloc(x1801 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1798, x1798));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1818, x_desc, x1810, x1816, x_desc, x1820));
};
if (x1823) {
} else {
assert(false && "ERROR not specified");
}
float* x1835 = (float*)myGpuMalloc(x1834 * sizeof(float));
float* x1836 = (float*)myMalloc(1 * sizeof(float));;
x1836[0] = 0.0f;
float* x1838 = (float*)myMalloc(1 * sizeof(float));;
x1838[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1798, x1798));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1829, x1829));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x1838, in_desc, x1820, filt_desc, x381,
    conv_desc, algo, ws_data, ws_size,
    x1836, out_desc, x1835));
};
float* x1841 = (float*)myGpuMalloc(x1832 * sizeof(float));
float* x1842 = (float*)myMalloc(1 * sizeof(float));;
x1842[0] = 0.0f;
float* x1844 = (float*)myMalloc(1 * sizeof(float));;
x1844[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1829, x1829));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1829, x1829));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1844, x1844, in_desc, x1835, out_desc, x1841, sbmv_desc, x759,
    x504, x333, x927, 1.0E-5));
};
if (x1850) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(1721) x Sym(1721), res:  x Const(64) x Const(512) x Sym(1829) x Sym(1829)");
}
float* x1855 = (float*)myMalloc(1 * sizeof(float));;
x1855[0] = 1.0f;
float* x1857 = (float*)myMalloc(1 * sizeof(float));;
x1857[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1721, x1721));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1829, x1829));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1855, bias_desc, x1756, x1857, out_desc, x1841));
};
float* x1860 = (float*)myMalloc(1 * sizeof(float));;
x1860[0] = 0.0f;
float* x1862 = (float*)myMalloc(1 * sizeof(float));;
x1862[0] = 1.0f;
float* x1864 = (float*)myGpuMalloc(x1832 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1829, x1829));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1862, x_desc, x1841, x1860, x_desc, x1864));
};
if (x1867) {
} else {
assert(false && "ERROR not specified");
}
float* x1879 = (float*)myGpuMalloc(x1878 * sizeof(float));
float* x1880 = (float*)myMalloc(1 * sizeof(float));;
x1880[0] = 0.0f;
float* x1882 = (float*)myMalloc(1 * sizeof(float));;
x1882[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1829, x1829));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1873, x1873));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x1882, in_desc, x1864, filt_desc, x654,
    conv_desc, algo, ws_data, ws_size,
    x1880, out_desc, x1879));
};
float* x1885 = (float*)myGpuMalloc(x1876 * sizeof(float));
float* x1886 = (float*)myMalloc(1 * sizeof(float));;
x1886[0] = 0.0f;
float* x1888 = (float*)myMalloc(1 * sizeof(float));;
x1888[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1873, x1873));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1873, x1873));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1888, x1888, in_desc, x1879, out_desc, x1885, sbmv_desc, x375,
    x984, x966, x1041, 1.0E-5));
};
float* x1891 = (float*)myMalloc(1 * sizeof(float));;
x1891[0] = 0.0f;
float* x1893 = (float*)myMalloc(1 * sizeof(float));;
x1893[0] = 1.0f;
float* x1895 = (float*)myGpuMalloc(x1876 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1873, x1873));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1893, x_desc, x1885, x1891, x_desc, x1895));
};
if (x1899) {
} else {
assert(false && "ERROR not specified");
}
float* x1912 = (float*)myGpuMalloc(x1911 * sizeof(float));
float* x1913 = (float*)myMalloc(1 * sizeof(float));;
x1913[0] = 0.0f;
float* x1915 = (float*)myMalloc(1 * sizeof(float));;
x1915[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1873, x1873));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1906, x1906));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x1915, in_desc, x1895, filt_desc, x753,
    conv_desc, algo, ws_data, ws_size,
    x1913, out_desc, x1912));
};
float* x1918 = (float*)myGpuMalloc(x1909 * sizeof(float));
float* x1919 = (float*)myMalloc(1 * sizeof(float));;
x1919[0] = 0.0f;
float* x1921 = (float*)myMalloc(1 * sizeof(float));;
x1921[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1906, x1906));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1906, x1906));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1921, x1921, in_desc, x1912, out_desc, x1918, sbmv_desc, x495,
    x372, x1062, x702, 1.0E-5));
};
float* x1924 = (float*)myMalloc(1 * sizeof(float));;
x1924[0] = 0.0f;
float* x1926 = (float*)myMalloc(1 * sizeof(float));;
x1926[0] = 1.0f;
float* x1928 = (float*)myGpuMalloc(x1909 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1906, x1906));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1926, x_desc, x1918, x1924, x_desc, x1928));
};
if (x1931) {
} else {
assert(false && "ERROR not specified");
}
float* x1943 = (float*)myGpuMalloc(x1942 * sizeof(float));
float* x1944 = (float*)myMalloc(1 * sizeof(float));;
x1944[0] = 0.0f;
float* x1946 = (float*)myMalloc(1 * sizeof(float));;
x1946[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1906, x1906));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1937, x1937));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x1946, in_desc, x1928, filt_desc, x423,
    conv_desc, algo, ws_data, ws_size,
    x1944, out_desc, x1943));
};
float* x1949 = (float*)myGpuMalloc(x1940 * sizeof(float));
float* x1950 = (float*)myMalloc(1 * sizeof(float));;
x1950[0] = 0.0f;
float* x1952 = (float*)myMalloc(1 * sizeof(float));;
x1952[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1937, x1937));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1937, x1937));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1952, x1952, in_desc, x1943, out_desc, x1949, sbmv_desc, x726,
    x420, x315, x960, 1.0E-5));
};
if (x1958) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(1829) x Sym(1829), res:  x Const(64) x Const(512) x Sym(1937) x Sym(1937)");
}
float* x1963 = (float*)myMalloc(1 * sizeof(float));;
x1963[0] = 1.0f;
float* x1965 = (float*)myMalloc(1 * sizeof(float));;
x1965[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1829, x1829));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1937, x1937));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1963, bias_desc, x1864, x1965, out_desc, x1949));
};
float* x1968 = (float*)myMalloc(1 * sizeof(float));;
x1968[0] = 0.0f;
float* x1970 = (float*)myMalloc(1 * sizeof(float));;
x1970[0] = 1.0f;
float* x1972 = (float*)myGpuMalloc(x1940 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1937, x1937));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1970, x_desc, x1949, x1968, x_desc, x1972));
};
if (x1975) {
} else {
assert(false && "ERROR not specified");
}
float* x1987 = (float*)myGpuMalloc(x1986 * sizeof(float));
float* x1988 = (float*)myMalloc(1 * sizeof(float));;
x1988[0] = 0.0f;
float* x1990 = (float*)myMalloc(1 * sizeof(float));;
x1990[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1937, x1937));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1981, x1981));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x1990, in_desc, x1972, filt_desc, x798,
    conv_desc, algo, ws_data, ws_size,
    x1988, out_desc, x1987));
};
float* x1993 = (float*)myGpuMalloc(x1984 * sizeof(float));
float* x1994 = (float*)myMalloc(1 * sizeof(float));;
x1994[0] = 0.0f;
float* x1996 = (float*)myMalloc(1 * sizeof(float));;
x1996[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1981, x1981));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1981, x1981));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1996, x1996, in_desc, x1987, out_desc, x1993, sbmv_desc, x1068,
    x321, x651, x852, 1.0E-5));
};
float* x1999 = (float*)myMalloc(1 * sizeof(float));;
x1999[0] = 0.0f;
float* x2001 = (float*)myMalloc(1 * sizeof(float));;
x2001[0] = 1.0f;
float* x2003 = (float*)myGpuMalloc(x1984 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1981, x1981));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2001, x_desc, x1993, x1999, x_desc, x2003));
};
if (x2007) {
} else {
assert(false && "ERROR not specified");
}
float* x2020 = (float*)myGpuMalloc(x2019 * sizeof(float));
float* x2021 = (float*)myMalloc(1 * sizeof(float));;
x2021[0] = 0.0f;
float* x2023 = (float*)myMalloc(1 * sizeof(float));;
x2023[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1981, x1981));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2014, x2014));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 2, 2, 1, 1,
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
    x2023, in_desc, x2003, filt_desc, x783,
    conv_desc, algo, ws_data, ws_size,
    x2021, out_desc, x2020));
};
float* x2026 = (float*)myGpuMalloc(x2017 * sizeof(float));
float* x2027 = (float*)myMalloc(1 * sizeof(float));;
x2027[0] = 0.0f;
float* x2029 = (float*)myMalloc(1 * sizeof(float));;
x2029[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2014, x2014));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2014, x2014));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2029, x2029, in_desc, x2020, out_desc, x2026, sbmv_desc, x582,
    x306, x945, x555, 1.0E-5));
};
float* x2032 = (float*)myMalloc(1 * sizeof(float));;
x2032[0] = 0.0f;
float* x2034 = (float*)myMalloc(1 * sizeof(float));;
x2034[0] = 1.0f;
float* x2036 = (float*)myGpuMalloc(x2017 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2014, x2014));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2034, x_desc, x2026, x2032, x_desc, x2036));
};
if (x2039) {
} else {
assert(false && "ERROR not specified");
}
float* x2051 = (float*)myGpuMalloc(x2050 * sizeof(float));
float* x2052 = (float*)myMalloc(1 * sizeof(float));;
x2052[0] = 0.0f;
float* x2054 = (float*)myMalloc(1 * sizeof(float));;
x2054[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2014, x2014));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2045, x2045));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x2054, in_desc, x2036, filt_desc, x1065,
    conv_desc, algo, ws_data, ws_size,
    x2052, out_desc, x2051));
};
float* x2057 = (float*)myGpuMalloc(x2048 * sizeof(float));
float* x2058 = (float*)myMalloc(1 * sizeof(float));;
x2058[0] = 0.0f;
float* x2060 = (float*)myMalloc(1 * sizeof(float));;
x2060[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2045, x2045));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2045, x2045));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2060, x2060, in_desc, x2051, out_desc, x2057, sbmv_desc, x312,
    x609, x906, x1059, 1.0E-5));
};
if (x1975) {
} else {
assert(false && "ERROR not specified");
}
float* x2073 = (float*)myGpuMalloc(x2072 * sizeof(float));
float* x2074 = (float*)myMalloc(1 * sizeof(float));;
x2074[0] = 0.0f;
float* x2076 = (float*)myMalloc(1 * sizeof(float));;
x2076[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1937, x1937));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2067, x2067));

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
    x2076, in_desc, x1972, filt_desc, x483,
    conv_desc, algo, ws_data, ws_size,
    x2074, out_desc, x2073));
};
float* x2079 = (float*)myGpuMalloc(x2070 * sizeof(float));
float* x2080 = (float*)myMalloc(1 * sizeof(float));;
x2080[0] = 0.0f;
float* x2082 = (float*)myMalloc(1 * sizeof(float));;
x2082[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2067, x2067));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2067, x2067));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2082, x2082, in_desc, x2073, out_desc, x2079, sbmv_desc, x345,
    x918, x516, x891, 1.0E-5));
};
if (x2088) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2067) x Sym(2067), res:  x Const(64) x Const(1024) x Sym(2045) x Sym(2045)");
}
float* x2093 = (float*)myMalloc(1 * sizeof(float));;
x2093[0] = 1.0f;
float* x2095 = (float*)myMalloc(1 * sizeof(float));;
x2095[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2067, x2067));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2045, x2045));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2093, bias_desc, x2079, x2095, out_desc, x2057));
};
float* x2098 = (float*)myMalloc(1 * sizeof(float));;
x2098[0] = 0.0f;
float* x2100 = (float*)myMalloc(1 * sizeof(float));;
x2100[0] = 1.0f;
float* x2102 = (float*)myGpuMalloc(x2048 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2045, x2045));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2100, x_desc, x2057, x2098, x_desc, x2102));
};
if (x2105) {
} else {
assert(false && "ERROR not specified");
}
float* x2117 = (float*)myGpuMalloc(x2116 * sizeof(float));
float* x2118 = (float*)myMalloc(1 * sizeof(float));;
x2118[0] = 0.0f;
float* x2120 = (float*)myMalloc(1 * sizeof(float));;
x2120[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2045, x2045));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2111, x2111));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x2120, in_desc, x2102, filt_desc, x297,
    conv_desc, algo, ws_data, ws_size,
    x2118, out_desc, x2117));
};
float* x2123 = (float*)myGpuMalloc(x2114 * sizeof(float));
float* x2124 = (float*)myMalloc(1 * sizeof(float));;
x2124[0] = 0.0f;
float* x2126 = (float*)myMalloc(1 * sizeof(float));;
x2126[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2111, x2111));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2111, x2111));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2126, x2126, in_desc, x2117, out_desc, x2123, sbmv_desc, x348,
    x915, x1035, x729, 1.0E-5));
};
float* x2129 = (float*)myMalloc(1 * sizeof(float));;
x2129[0] = 0.0f;
float* x2131 = (float*)myMalloc(1 * sizeof(float));;
x2131[0] = 1.0f;
float* x2133 = (float*)myGpuMalloc(x2114 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2111, x2111));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2131, x_desc, x2123, x2129, x_desc, x2133));
};
if (x2137) {
} else {
assert(false && "ERROR not specified");
}
float* x2150 = (float*)myGpuMalloc(x2149 * sizeof(float));
float* x2151 = (float*)myMalloc(1 * sizeof(float));;
x2151[0] = 0.0f;
float* x2153 = (float*)myMalloc(1 * sizeof(float));;
x2153[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2111, x2111));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2144, x2144));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x2153, in_desc, x2133, filt_desc, x351,
    conv_desc, algo, ws_data, ws_size,
    x2151, out_desc, x2150));
};
float* x2156 = (float*)myGpuMalloc(x2147 * sizeof(float));
float* x2157 = (float*)myMalloc(1 * sizeof(float));;
x2157[0] = 0.0f;
float* x2159 = (float*)myMalloc(1 * sizeof(float));;
x2159[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2144, x2144));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2144, x2144));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2159, x2159, in_desc, x2150, out_desc, x2156, sbmv_desc, x1071,
    x546, x858, x969, 1.0E-5));
};
float* x2162 = (float*)myMalloc(1 * sizeof(float));;
x2162[0] = 0.0f;
float* x2164 = (float*)myMalloc(1 * sizeof(float));;
x2164[0] = 1.0f;
float* x2166 = (float*)myGpuMalloc(x2147 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2144, x2144));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2164, x_desc, x2156, x2162, x_desc, x2166));
};
if (x2169) {
} else {
assert(false && "ERROR not specified");
}
float* x2181 = (float*)myGpuMalloc(x2180 * sizeof(float));
float* x2182 = (float*)myMalloc(1 * sizeof(float));;
x2182[0] = 0.0f;
float* x2184 = (float*)myMalloc(1 * sizeof(float));;
x2184[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2144, x2144));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2175, x2175));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x2184, in_desc, x2166, filt_desc, x426,
    conv_desc, algo, ws_data, ws_size,
    x2182, out_desc, x2181));
};
float* x2187 = (float*)myGpuMalloc(x2178 * sizeof(float));
float* x2188 = (float*)myMalloc(1 * sizeof(float));;
x2188[0] = 0.0f;
float* x2190 = (float*)myMalloc(1 * sizeof(float));;
x2190[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2175, x2175));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2175, x2175));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2190, x2190, in_desc, x2181, out_desc, x2187, sbmv_desc, x318,
    x954, x804, x687, 1.0E-5));
};
if (x2196) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2045) x Sym(2045), res:  x Const(64) x Const(1024) x Sym(2175) x Sym(2175)");
}
float* x2201 = (float*)myMalloc(1 * sizeof(float));;
x2201[0] = 1.0f;
float* x2203 = (float*)myMalloc(1 * sizeof(float));;
x2203[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2045, x2045));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2175, x2175));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2201, bias_desc, x2102, x2203, out_desc, x2187));
};
float* x2206 = (float*)myMalloc(1 * sizeof(float));;
x2206[0] = 0.0f;
float* x2208 = (float*)myMalloc(1 * sizeof(float));;
x2208[0] = 1.0f;
float* x2210 = (float*)myGpuMalloc(x2178 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2175, x2175));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2208, x_desc, x2187, x2206, x_desc, x2210));
};
if (x2213) {
} else {
assert(false && "ERROR not specified");
}
float* x2225 = (float*)myGpuMalloc(x2224 * sizeof(float));
float* x2226 = (float*)myMalloc(1 * sizeof(float));;
x2226[0] = 0.0f;
float* x2228 = (float*)myMalloc(1 * sizeof(float));;
x2228[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2175, x2175));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2219, x2219));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x2228, in_desc, x2210, filt_desc, x912,
    conv_desc, algo, ws_data, ws_size,
    x2226, out_desc, x2225));
};
float* x2231 = (float*)myGpuMalloc(x2222 * sizeof(float));
float* x2232 = (float*)myMalloc(1 * sizeof(float));;
x2232[0] = 0.0f;
float* x2234 = (float*)myMalloc(1 * sizeof(float));;
x2234[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2219, x2219));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2219, x2219));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2234, x2234, in_desc, x2225, out_desc, x2231, sbmv_desc, x645,
    x849, x792, x780, 1.0E-5));
};
float* x2237 = (float*)myMalloc(1 * sizeof(float));;
x2237[0] = 0.0f;
float* x2239 = (float*)myMalloc(1 * sizeof(float));;
x2239[0] = 1.0f;
float* x2241 = (float*)myGpuMalloc(x2222 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2219, x2219));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2239, x_desc, x2231, x2237, x_desc, x2241));
};
if (x2245) {
} else {
assert(false && "ERROR not specified");
}
float* x2258 = (float*)myGpuMalloc(x2257 * sizeof(float));
float* x2259 = (float*)myMalloc(1 * sizeof(float));;
x2259[0] = 0.0f;
float* x2261 = (float*)myMalloc(1 * sizeof(float));;
x2261[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2219, x2219));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2252, x2252));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x2261, in_desc, x2241, filt_desc, x300,
    conv_desc, algo, ws_data, ws_size,
    x2259, out_desc, x2258));
};
float* x2264 = (float*)myGpuMalloc(x2255 * sizeof(float));
float* x2265 = (float*)myMalloc(1 * sizeof(float));;
x2265[0] = 0.0f;
float* x2267 = (float*)myMalloc(1 * sizeof(float));;
x2267[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2252, x2252));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2252, x2252));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2267, x2267, in_desc, x2258, out_desc, x2264, sbmv_desc, x942,
    x834, x630, x447, 1.0E-5));
};
float* x2270 = (float*)myMalloc(1 * sizeof(float));;
x2270[0] = 0.0f;
float* x2272 = (float*)myMalloc(1 * sizeof(float));;
x2272[0] = 1.0f;
float* x2274 = (float*)myGpuMalloc(x2255 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2252, x2252));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2272, x_desc, x2264, x2270, x_desc, x2274));
};
if (x2277) {
} else {
assert(false && "ERROR not specified");
}
float* x2289 = (float*)myGpuMalloc(x2288 * sizeof(float));
float* x2290 = (float*)myMalloc(1 * sizeof(float));;
x2290[0] = 0.0f;
float* x2292 = (float*)myMalloc(1 * sizeof(float));;
x2292[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2252, x2252));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2283, x2283));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x2292, in_desc, x2274, filt_desc, x606,
    conv_desc, algo, ws_data, ws_size,
    x2290, out_desc, x2289));
};
float* x2295 = (float*)myGpuMalloc(x2286 * sizeof(float));
float* x2296 = (float*)myMalloc(1 * sizeof(float));;
x2296[0] = 0.0f;
float* x2298 = (float*)myMalloc(1 * sizeof(float));;
x2298[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2283, x2283));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2283, x2283));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2298, x2298, in_desc, x2289, out_desc, x2295, sbmv_desc, x1047,
    x429, x678, x822, 1.0E-5));
};
if (x2304) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2175) x Sym(2175), res:  x Const(64) x Const(1024) x Sym(2283) x Sym(2283)");
}
float* x2309 = (float*)myMalloc(1 * sizeof(float));;
x2309[0] = 1.0f;
float* x2311 = (float*)myMalloc(1 * sizeof(float));;
x2311[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2175, x2175));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2283, x2283));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2309, bias_desc, x2210, x2311, out_desc, x2295));
};
float* x2314 = (float*)myMalloc(1 * sizeof(float));;
x2314[0] = 0.0f;
float* x2316 = (float*)myMalloc(1 * sizeof(float));;
x2316[0] = 1.0f;
float* x2318 = (float*)myGpuMalloc(x2286 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2283, x2283));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2316, x_desc, x2295, x2314, x_desc, x2318));
};
if (x2321) {
} else {
assert(false && "ERROR not specified");
}
float* x2333 = (float*)myGpuMalloc(x2332 * sizeof(float));
float* x2334 = (float*)myMalloc(1 * sizeof(float));;
x2334[0] = 0.0f;
float* x2336 = (float*)myMalloc(1 * sizeof(float));;
x2336[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2283, x2283));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2327, x2327));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x2336, in_desc, x2318, filt_desc, x276,
    conv_desc, algo, ws_data, ws_size,
    x2334, out_desc, x2333));
};
float* x2339 = (float*)myGpuMalloc(x2330 * sizeof(float));
float* x2340 = (float*)myMalloc(1 * sizeof(float));;
x2340[0] = 0.0f;
float* x2342 = (float*)myMalloc(1 * sizeof(float));;
x2342[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2327, x2327));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2327, x2327));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2342, x2342, in_desc, x2333, out_desc, x2339, sbmv_desc, x534,
    x981, x747, x552, 1.0E-5));
};
float* x2345 = (float*)myMalloc(1 * sizeof(float));;
x2345[0] = 0.0f;
float* x2347 = (float*)myMalloc(1 * sizeof(float));;
x2347[0] = 1.0f;
float* x2349 = (float*)myGpuMalloc(x2330 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2327, x2327));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2347, x_desc, x2339, x2345, x_desc, x2349));
};
if (x2353) {
} else {
assert(false && "ERROR not specified");
}
float* x2366 = (float*)myGpuMalloc(x2365 * sizeof(float));
float* x2367 = (float*)myMalloc(1 * sizeof(float));;
x2367[0] = 0.0f;
float* x2369 = (float*)myMalloc(1 * sizeof(float));;
x2369[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2327, x2327));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2360, x2360));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x2369, in_desc, x2349, filt_desc, x1005,
    conv_desc, algo, ws_data, ws_size,
    x2367, out_desc, x2366));
};
float* x2372 = (float*)myGpuMalloc(x2363 * sizeof(float));
float* x2373 = (float*)myMalloc(1 * sizeof(float));;
x2373[0] = 0.0f;
float* x2375 = (float*)myMalloc(1 * sizeof(float));;
x2375[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2360, x2360));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2360, x2360));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2375, x2375, in_desc, x2366, out_desc, x2372, sbmv_desc, x480,
    x666, x816, x948, 1.0E-5));
};
float* x2378 = (float*)myMalloc(1 * sizeof(float));;
x2378[0] = 0.0f;
float* x2380 = (float*)myMalloc(1 * sizeof(float));;
x2380[0] = 1.0f;
float* x2382 = (float*)myGpuMalloc(x2363 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2360, x2360));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2380, x_desc, x2372, x2378, x_desc, x2382));
};
if (x2385) {
} else {
assert(false && "ERROR not specified");
}
float* x2397 = (float*)myGpuMalloc(x2396 * sizeof(float));
float* x2398 = (float*)myMalloc(1 * sizeof(float));;
x2398[0] = 0.0f;
float* x2400 = (float*)myMalloc(1 * sizeof(float));;
x2400[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2360, x2360));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2391, x2391));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x2400, in_desc, x2382, filt_desc, x525,
    conv_desc, algo, ws_data, ws_size,
    x2398, out_desc, x2397));
};
float* x2403 = (float*)myGpuMalloc(x2394 * sizeof(float));
float* x2404 = (float*)myMalloc(1 * sizeof(float));;
x2404[0] = 0.0f;
float* x2406 = (float*)myMalloc(1 * sizeof(float));;
x2406[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2391, x2391));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2391, x2391));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2406, x2406, in_desc, x2397, out_desc, x2403, sbmv_desc, x972,
    x696, x951, x741, 1.0E-5));
};
if (x2412) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2283) x Sym(2283), res:  x Const(64) x Const(1024) x Sym(2391) x Sym(2391)");
}
float* x2417 = (float*)myMalloc(1 * sizeof(float));;
x2417[0] = 1.0f;
float* x2419 = (float*)myMalloc(1 * sizeof(float));;
x2419[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2283, x2283));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2391, x2391));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2417, bias_desc, x2318, x2419, out_desc, x2403));
};
float* x2422 = (float*)myMalloc(1 * sizeof(float));;
x2422[0] = 0.0f;
float* x2424 = (float*)myMalloc(1 * sizeof(float));;
x2424[0] = 1.0f;
float* x2426 = (float*)myGpuMalloc(x2394 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2391, x2391));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2424, x_desc, x2403, x2422, x_desc, x2426));
};
if (x2429) {
} else {
assert(false && "ERROR not specified");
}
float* x2441 = (float*)myGpuMalloc(x2440 * sizeof(float));
float* x2442 = (float*)myMalloc(1 * sizeof(float));;
x2442[0] = 0.0f;
float* x2444 = (float*)myMalloc(1 * sizeof(float));;
x2444[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2391, x2391));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2435, x2435));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x2444, in_desc, x2426, filt_desc, x324,
    conv_desc, algo, ws_data, ws_size,
    x2442, out_desc, x2441));
};
float* x2447 = (float*)myGpuMalloc(x2438 * sizeof(float));
float* x2448 = (float*)myMalloc(1 * sizeof(float));;
x2448[0] = 0.0f;
float* x2450 = (float*)myMalloc(1 * sizeof(float));;
x2450[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2435, x2435));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2435, x2435));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2450, x2450, in_desc, x2441, out_desc, x2447, sbmv_desc, x489,
    x813, x1020, x465, 1.0E-5));
};
float* x2453 = (float*)myMalloc(1 * sizeof(float));;
x2453[0] = 0.0f;
float* x2455 = (float*)myMalloc(1 * sizeof(float));;
x2455[0] = 1.0f;
float* x2457 = (float*)myGpuMalloc(x2438 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2435, x2435));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2455, x_desc, x2447, x2453, x_desc, x2457));
};
if (x2461) {
} else {
assert(false && "ERROR not specified");
}
float* x2474 = (float*)myGpuMalloc(x2473 * sizeof(float));
float* x2475 = (float*)myMalloc(1 * sizeof(float));;
x2475[0] = 0.0f;
float* x2477 = (float*)myMalloc(1 * sizeof(float));;
x2477[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2435, x2435));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2468, x2468));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x2477, in_desc, x2457, filt_desc, x1044,
    conv_desc, algo, ws_data, ws_size,
    x2475, out_desc, x2474));
};
float* x2480 = (float*)myGpuMalloc(x2471 * sizeof(float));
float* x2481 = (float*)myMalloc(1 * sizeof(float));;
x2481[0] = 0.0f;
float* x2483 = (float*)myMalloc(1 * sizeof(float));;
x2483[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2468, x2468));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2468, x2468));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2483, x2483, in_desc, x2474, out_desc, x2480, sbmv_desc, x762,
    x585, x1008, x570, 1.0E-5));
};
float* x2486 = (float*)myMalloc(1 * sizeof(float));;
x2486[0] = 0.0f;
float* x2488 = (float*)myMalloc(1 * sizeof(float));;
x2488[0] = 1.0f;
float* x2490 = (float*)myGpuMalloc(x2471 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2468, x2468));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2488, x_desc, x2480, x2486, x_desc, x2490));
};
if (x2493) {
} else {
assert(false && "ERROR not specified");
}
float* x2505 = (float*)myGpuMalloc(x2504 * sizeof(float));
float* x2506 = (float*)myMalloc(1 * sizeof(float));;
x2506[0] = 0.0f;
float* x2508 = (float*)myMalloc(1 * sizeof(float));;
x2508[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2468, x2468));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2499, x2499));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x2508, in_desc, x2490, filt_desc, x921,
    conv_desc, algo, ws_data, ws_size,
    x2506, out_desc, x2505));
};
float* x2511 = (float*)myGpuMalloc(x2502 * sizeof(float));
float* x2512 = (float*)myMalloc(1 * sizeof(float));;
x2512[0] = 0.0f;
float* x2514 = (float*)myMalloc(1 * sizeof(float));;
x2514[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2499, x2499));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2499, x2499));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2514, x2514, in_desc, x2505, out_desc, x2511, sbmv_desc, x435,
    x618, x885, x1074, 1.0E-5));
};
if (x2520) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2391) x Sym(2391), res:  x Const(64) x Const(1024) x Sym(2499) x Sym(2499)");
}
float* x2525 = (float*)myMalloc(1 * sizeof(float));;
x2525[0] = 1.0f;
float* x2527 = (float*)myMalloc(1 * sizeof(float));;
x2527[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2391, x2391));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2499, x2499));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2525, bias_desc, x2426, x2527, out_desc, x2511));
};
float* x2530 = (float*)myMalloc(1 * sizeof(float));;
x2530[0] = 0.0f;
float* x2532 = (float*)myMalloc(1 * sizeof(float));;
x2532[0] = 1.0f;
float* x2534 = (float*)myGpuMalloc(x2502 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2499, x2499));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2532, x_desc, x2511, x2530, x_desc, x2534));
};
if (x2537) {
} else {
assert(false && "ERROR not specified");
}
float* x2549 = (float*)myGpuMalloc(x2548 * sizeof(float));
float* x2550 = (float*)myMalloc(1 * sizeof(float));;
x2550[0] = 0.0f;
float* x2552 = (float*)myMalloc(1 * sizeof(float));;
x2552[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2499, x2499));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2543, x2543));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x2552, in_desc, x2534, filt_desc, x711,
    conv_desc, algo, ws_data, ws_size,
    x2550, out_desc, x2549));
};
float* x2555 = (float*)myGpuMalloc(x2546 * sizeof(float));
float* x2556 = (float*)myMalloc(1 * sizeof(float));;
x2556[0] = 0.0f;
float* x2558 = (float*)myMalloc(1 * sizeof(float));;
x2558[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2543, x2543));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2543, x2543));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2558, x2558, in_desc, x2549, out_desc, x2555, sbmv_desc, x513,
    x1017, x498, x786, 1.0E-5));
};
float* x2561 = (float*)myMalloc(1 * sizeof(float));;
x2561[0] = 0.0f;
float* x2563 = (float*)myMalloc(1 * sizeof(float));;
x2563[0] = 1.0f;
float* x2565 = (float*)myGpuMalloc(x2546 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2543, x2543));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2563, x_desc, x2555, x2561, x_desc, x2565));
};
if (x2569) {
} else {
assert(false && "ERROR not specified");
}
float* x2582 = (float*)myGpuMalloc(x2581 * sizeof(float));
float* x2583 = (float*)myMalloc(1 * sizeof(float));;
x2583[0] = 0.0f;
float* x2585 = (float*)myMalloc(1 * sizeof(float));;
x2585[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2543, x2543));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2576, x2576));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x2585, in_desc, x2565, filt_desc, x936,
    conv_desc, algo, ws_data, ws_size,
    x2583, out_desc, x2582));
};
float* x2588 = (float*)myGpuMalloc(x2579 * sizeof(float));
float* x2589 = (float*)myMalloc(1 * sizeof(float));;
x2589[0] = 0.0f;
float* x2591 = (float*)myMalloc(1 * sizeof(float));;
x2591[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2576, x2576));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2576, x2576));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2591, x2591, in_desc, x2582, out_desc, x2588, sbmv_desc, x681,
    x825, x468, x978, 1.0E-5));
};
float* x2594 = (float*)myMalloc(1 * sizeof(float));;
x2594[0] = 0.0f;
float* x2596 = (float*)myMalloc(1 * sizeof(float));;
x2596[0] = 1.0f;
float* x2598 = (float*)myGpuMalloc(x2579 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2576, x2576));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2596, x_desc, x2588, x2594, x_desc, x2598));
};
if (x2601) {
} else {
assert(false && "ERROR not specified");
}
float* x2613 = (float*)myGpuMalloc(x2612 * sizeof(float));
float* x2614 = (float*)myMalloc(1 * sizeof(float));;
x2614[0] = 0.0f;
float* x2616 = (float*)myMalloc(1 * sizeof(float));;
x2616[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2576, x2576));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2607, x2607));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x2616, in_desc, x2598, filt_desc, x549,
    conv_desc, algo, ws_data, ws_size,
    x2614, out_desc, x2613));
};
float* x2619 = (float*)myGpuMalloc(x2610 * sizeof(float));
float* x2620 = (float*)myMalloc(1 * sizeof(float));;
x2620[0] = 0.0f;
float* x2622 = (float*)myMalloc(1 * sizeof(float));;
x2622[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2607, x2607));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2607, x2607));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2622, x2622, in_desc, x2613, out_desc, x2619, sbmv_desc, x1002,
    x537, x624, x807, 1.0E-5));
};
if (x2628) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2499) x Sym(2499), res:  x Const(64) x Const(1024) x Sym(2607) x Sym(2607)");
}
float* x2633 = (float*)myMalloc(1 * sizeof(float));;
x2633[0] = 1.0f;
float* x2635 = (float*)myMalloc(1 * sizeof(float));;
x2635[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2499, x2499));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2607, x2607));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2633, bias_desc, x2534, x2635, out_desc, x2619));
};
float* x2638 = (float*)myMalloc(1 * sizeof(float));;
x2638[0] = 0.0f;
float* x2640 = (float*)myMalloc(1 * sizeof(float));;
x2640[0] = 1.0f;
float* x2642 = (float*)myGpuMalloc(x2610 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2607, x2607));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2640, x_desc, x2619, x2638, x_desc, x2642));
};
if (x2645) {
} else {
assert(false && "ERROR not specified");
}
float* x2657 = (float*)myGpuMalloc(x2656 * sizeof(float));
float* x2658 = (float*)myMalloc(1 * sizeof(float));;
x2658[0] = 0.0f;
float* x2660 = (float*)myMalloc(1 * sizeof(float));;
x2660[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2607, x2607));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2651, x2651));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x2660, in_desc, x2642, filt_desc, x675,
    conv_desc, algo, ws_data, ws_size,
    x2658, out_desc, x2657));
};
float* x2663 = (float*)myGpuMalloc(x2654 * sizeof(float));
float* x2664 = (float*)myMalloc(1 * sizeof(float));;
x2664[0] = 0.0f;
float* x2666 = (float*)myMalloc(1 * sizeof(float));;
x2666[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2651, x2651));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2651, x2651));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2666, x2666, in_desc, x2657, out_desc, x2663, sbmv_desc, x861,
    x930, x459, x621, 1.0E-5));
};
float* x2669 = (float*)myMalloc(1 * sizeof(float));;
x2669[0] = 0.0f;
float* x2671 = (float*)myMalloc(1 * sizeof(float));;
x2671[0] = 1.0f;
float* x2673 = (float*)myGpuMalloc(x2654 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2651, x2651));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2671, x_desc, x2663, x2669, x_desc, x2673));
};
if (x2677) {
} else {
assert(false && "ERROR not specified");
}
float* x2690 = (float*)myGpuMalloc(x2689 * sizeof(float));
float* x2691 = (float*)myMalloc(1 * sizeof(float));;
x2691[0] = 0.0f;
float* x2693 = (float*)myMalloc(1 * sizeof(float));;
x2693[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2651, x2651));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2684, x2684));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 2, 2, 1, 1,
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
    x2693, in_desc, x2673, filt_desc, x360,
    conv_desc, algo, ws_data, ws_size,
    x2691, out_desc, x2690));
};
float* x2696 = (float*)myGpuMalloc(x2687 * sizeof(float));
float* x2697 = (float*)myMalloc(1 * sizeof(float));;
x2697[0] = 0.0f;
float* x2699 = (float*)myMalloc(1 * sizeof(float));;
x2699[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2684, x2684));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2684, x2684));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2699, x2699, in_desc, x2690, out_desc, x2696, sbmv_desc, x873,
    x735, x597, x408, 1.0E-5));
};
float* x2702 = (float*)myMalloc(1 * sizeof(float));;
x2702[0] = 0.0f;
float* x2704 = (float*)myMalloc(1 * sizeof(float));;
x2704[0] = 1.0f;
float* x2706 = (float*)myGpuMalloc(x2687 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2684, x2684));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2704, x_desc, x2696, x2702, x_desc, x2706));
};
if (x2709) {
} else {
assert(false && "ERROR not specified");
}
float* x2721 = (float*)myGpuMalloc(x2720 * sizeof(float));
float* x2722 = (float*)myMalloc(1 * sizeof(float));;
x2722[0] = 0.0f;
float* x2724 = (float*)myMalloc(1 * sizeof(float));;
x2724[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2684, x2684));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2715, x2715));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x2724, in_desc, x2706, filt_desc, x894,
    conv_desc, algo, ws_data, ws_size,
    x2722, out_desc, x2721));
};
float* x2727 = (float*)myGpuMalloc(x2718 * sizeof(float));
float* x2728 = (float*)myMalloc(1 * sizeof(float));;
x2728[0] = 0.0f;
float* x2730 = (float*)myMalloc(1 * sizeof(float));;
x2730[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2715, x2715));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2715, x2715));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2730, x2730, in_desc, x2721, out_desc, x2727, sbmv_desc, x975,
    x444, x603, x837, 1.0E-5));
};
if (x2645) {
} else {
assert(false && "ERROR not specified");
}
float* x2743 = (float*)myGpuMalloc(x2742 * sizeof(float));
float* x2744 = (float*)myMalloc(1 * sizeof(float));;
x2744[0] = 0.0f;
float* x2746 = (float*)myMalloc(1 * sizeof(float));;
x2746[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2607, x2607));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2737, x2737));

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
    x2746, in_desc, x2642, filt_desc, x900,
    conv_desc, algo, ws_data, ws_size,
    x2744, out_desc, x2743));
};
float* x2749 = (float*)myGpuMalloc(x2740 * sizeof(float));
float* x2750 = (float*)myMalloc(1 * sizeof(float));;
x2750[0] = 0.0f;
float* x2752 = (float*)myMalloc(1 * sizeof(float));;
x2752[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2737, x2737));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2737, x2737));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2752, x2752, in_desc, x2743, out_desc, x2749, sbmv_desc, x777,
    x579, x450, x633, 1.0E-5));
};
if (x2758) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(2737) x Sym(2737), res:  x Const(64) x Const(2048) x Sym(2715) x Sym(2715)");
}
float* x2763 = (float*)myMalloc(1 * sizeof(float));;
x2763[0] = 1.0f;
float* x2765 = (float*)myMalloc(1 * sizeof(float));;
x2765[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2737, x2737));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2715, x2715));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2763, bias_desc, x2749, x2765, out_desc, x2727));
};
float* x2768 = (float*)myMalloc(1 * sizeof(float));;
x2768[0] = 0.0f;
float* x2770 = (float*)myMalloc(1 * sizeof(float));;
x2770[0] = 1.0f;
float* x2772 = (float*)myGpuMalloc(x2718 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2715, x2715));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2770, x_desc, x2727, x2768, x_desc, x2772));
};
if (x2775) {
} else {
assert(false && "ERROR not specified");
}
float* x2787 = (float*)myGpuMalloc(x2786 * sizeof(float));
float* x2788 = (float*)myMalloc(1 * sizeof(float));;
x2788[0] = 0.0f;
float* x2790 = (float*)myMalloc(1 * sizeof(float));;
x2790[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2715, x2715));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 2048, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2781, x2781));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x2790, in_desc, x2772, filt_desc, x903,
    conv_desc, algo, ws_data, ws_size,
    x2788, out_desc, x2787));
};
float* x2793 = (float*)myGpuMalloc(x2784 * sizeof(float));
float* x2794 = (float*)myMalloc(1 * sizeof(float));;
x2794[0] = 0.0f;
float* x2796 = (float*)myMalloc(1 * sizeof(float));;
x2796[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2781, x2781));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2781, x2781));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2796, x2796, in_desc, x2787, out_desc, x2793, sbmv_desc, x396,
    x669, x720, x453, 1.0E-5));
};
float* x2799 = (float*)myMalloc(1 * sizeof(float));;
x2799[0] = 0.0f;
float* x2801 = (float*)myMalloc(1 * sizeof(float));;
x2801[0] = 1.0f;
float* x2803 = (float*)myGpuMalloc(x2784 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2781, x2781));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2801, x_desc, x2793, x2799, x_desc, x2803));
};
if (x2807) {
} else {
assert(false && "ERROR not specified");
}
float* x2820 = (float*)myGpuMalloc(x2819 * sizeof(float));
float* x2821 = (float*)myMalloc(1 * sizeof(float));;
x2821[0] = 0.0f;
float* x2823 = (float*)myMalloc(1 * sizeof(float));;
x2823[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2781, x2781));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2814, x2814));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x2823, in_desc, x2803, filt_desc, x723,
    conv_desc, algo, ws_data, ws_size,
    x2821, out_desc, x2820));
};
float* x2826 = (float*)myGpuMalloc(x2817 * sizeof(float));
float* x2827 = (float*)myMalloc(1 * sizeof(float));;
x2827[0] = 0.0f;
float* x2829 = (float*)myMalloc(1 * sizeof(float));;
x2829[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2814, x2814));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2814, x2814));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2829, x2829, in_desc, x2820, out_desc, x2826, sbmv_desc, x738,
    x456, x672, x843, 1.0E-5));
};
float* x2832 = (float*)myMalloc(1 * sizeof(float));;
x2832[0] = 0.0f;
float* x2834 = (float*)myMalloc(1 * sizeof(float));;
x2834[0] = 1.0f;
float* x2836 = (float*)myGpuMalloc(x2817 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2814, x2814));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2834, x_desc, x2826, x2832, x_desc, x2836));
};
if (x2839) {
} else {
assert(false && "ERROR not specified");
}
float* x2851 = (float*)myGpuMalloc(x2850 * sizeof(float));
float* x2852 = (float*)myMalloc(1 * sizeof(float));;
x2852[0] = 0.0f;
float* x2854 = (float*)myMalloc(1 * sizeof(float));;
x2854[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2814, x2814));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2845, x2845));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x2854, in_desc, x2836, filt_desc, x399,
    conv_desc, algo, ws_data, ws_size,
    x2852, out_desc, x2851));
};
float* x2857 = (float*)myGpuMalloc(x2848 * sizeof(float));
float* x2858 = (float*)myMalloc(1 * sizeof(float));;
x2858[0] = 0.0f;
float* x2860 = (float*)myMalloc(1 * sizeof(float));;
x2860[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2845, x2845));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2845, x2845));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2860, x2860, in_desc, x2851, out_desc, x2857, sbmv_desc, x540,
    x690, x462, x993, 1.0E-5));
};
if (x2866) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(2715) x Sym(2715), res:  x Const(64) x Const(2048) x Sym(2845) x Sym(2845)");
}
float* x2871 = (float*)myMalloc(1 * sizeof(float));;
x2871[0] = 1.0f;
float* x2873 = (float*)myMalloc(1 * sizeof(float));;
x2873[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2715, x2715));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2845, x2845));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2871, bias_desc, x2772, x2873, out_desc, x2857));
};
float* x2876 = (float*)myMalloc(1 * sizeof(float));;
x2876[0] = 0.0f;
float* x2878 = (float*)myMalloc(1 * sizeof(float));;
x2878[0] = 1.0f;
float* x2880 = (float*)myGpuMalloc(x2848 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2845, x2845));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2878, x_desc, x2857, x2876, x_desc, x2880));
};
if (x2883) {
} else {
assert(false && "ERROR not specified");
}
float* x2895 = (float*)myGpuMalloc(x2894 * sizeof(float));
float* x2896 = (float*)myMalloc(1 * sizeof(float));;
x2896[0] = 0.0f;
float* x2898 = (float*)myMalloc(1 * sizeof(float));;
x2898[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2845, x2845));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 2048, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2889, x2889));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x2898, in_desc, x2880, filt_desc, x1053,
    conv_desc, algo, ws_data, ws_size,
    x2896, out_desc, x2895));
};
float* x2901 = (float*)myGpuMalloc(x2892 * sizeof(float));
float* x2902 = (float*)myMalloc(1 * sizeof(float));;
x2902[0] = 0.0f;
float* x2904 = (float*)myMalloc(1 * sizeof(float));;
x2904[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2889, x2889));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2889, x2889));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2904, x2904, in_desc, x2895, out_desc, x2901, sbmv_desc, x303,
    x492, x897, x1023, 1.0E-5));
};
float* x2907 = (float*)myMalloc(1 * sizeof(float));;
x2907[0] = 0.0f;
float* x2909 = (float*)myMalloc(1 * sizeof(float));;
x2909[0] = 1.0f;
float* x2911 = (float*)myGpuMalloc(x2892 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2889, x2889));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2909, x_desc, x2901, x2907, x_desc, x2911));
};
if (x2915) {
} else {
assert(false && "ERROR not specified");
}
float* x2928 = (float*)myGpuMalloc(x2927 * sizeof(float));
float* x2929 = (float*)myMalloc(1 * sizeof(float));;
x2929[0] = 0.0f;
float* x2931 = (float*)myMalloc(1 * sizeof(float));;
x2931[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2889, x2889));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2922, x2922));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x2931, in_desc, x2911, filt_desc, x342,
    conv_desc, algo, ws_data, ws_size,
    x2929, out_desc, x2928));
};
float* x2934 = (float*)myGpuMalloc(x2925 * sizeof(float));
float* x2935 = (float*)myMalloc(1 * sizeof(float));;
x2935[0] = 0.0f;
float* x2937 = (float*)myMalloc(1 * sizeof(float));;
x2937[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2922, x2922));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2922, x2922));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2937, x2937, in_desc, x2928, out_desc, x2934, sbmv_desc, x840,
    x765, x294, x864, 1.0E-5));
};
float* x2940 = (float*)myMalloc(1 * sizeof(float));;
x2940[0] = 0.0f;
float* x2942 = (float*)myMalloc(1 * sizeof(float));;
x2942[0] = 1.0f;
float* x2944 = (float*)myGpuMalloc(x2925 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2922, x2922));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2942, x_desc, x2934, x2940, x_desc, x2944));
};
if (x2947) {
} else {
assert(false && "ERROR not specified");
}
float* x2959 = (float*)myGpuMalloc(x2958 * sizeof(float));
float* x2960 = (float*)myMalloc(1 * sizeof(float));;
x2960[0] = 0.0f;
float* x2962 = (float*)myMalloc(1 * sizeof(float));;
x2962[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2922, x2922));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2953, x2953));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x2962, in_desc, x2944, filt_desc, x357,
    conv_desc, algo, ws_data, ws_size,
    x2960, out_desc, x2959));
};
float* x2965 = (float*)myGpuMalloc(x2956 * sizeof(float));
float* x2966 = (float*)myMalloc(1 * sizeof(float));;
x2966[0] = 0.0f;
float* x2968 = (float*)myMalloc(1 * sizeof(float));;
x2968[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2953, x2953));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2953, x2953));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2968, x2968, in_desc, x2959, out_desc, x2965, sbmv_desc, x567,
    x801, x1038, x627, 1.0E-5));
};
if (x2974) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(2845) x Sym(2845), res:  x Const(64) x Const(2048) x Sym(2953) x Sym(2953)");
}
float* x2979 = (float*)myMalloc(1 * sizeof(float));;
x2979[0] = 1.0f;
float* x2981 = (float*)myMalloc(1 * sizeof(float));;
x2981[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2845, x2845));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2953, x2953));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2979, bias_desc, x2880, x2981, out_desc, x2965));
};
float* x2984 = (float*)myMalloc(1 * sizeof(float));;
x2984[0] = 0.0f;
float* x2986 = (float*)myMalloc(1 * sizeof(float));;
x2986[0] = 1.0f;
float* x2988 = (float*)myGpuMalloc(x2956 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2953, x2953));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2986, x_desc, x2965, x2984, x_desc, x2988));
};
if (x2991) {
} else {
assert(false && "Image too small for averagePool_batch:  x Const(64) x Const(2048) x Sym(2953) x Sym(2953)|(2,2)");
}
float* x2996 = (float*)myMalloc(1 * sizeof(float));;
x2996[0] = 0.0f;
float* x2998 = (float*)myMalloc(1 * sizeof(float));;
x2998[0] = 1.0f;
float* x3008 = (float*)myGpuMalloc(x3007 * sizeof(float));

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2953, x2953) );

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3002, x3002));

cudnnPoolingDescriptor_t poolingDesc;
CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));
CUDNN_CALL(cudnnSetPooling2dDescriptor(
    poolingDesc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN,
    2, 2, 0,
    0, 1, 1
));
CUDNN_CALL(cudnnPoolingForward(
    cudnnHandle, 
    poolingDesc, 
    x2998, in_desc, x2988, x2996, out_desc, x3008));
};
int32_t x3010 = 0;
int32_t x3011 = 1;
x3011 *= 64;
x3010 += 1;
int32_t x3014 = x3010;
bool x3015 = x3014 >= 2;
if (x3015) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3021 = x3014 == 0;
if (x3021) {
int32_t x3022 = x3011;
bool x3023 = x3022 == x3005;
if (x3023) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3030 = x3011;
// gemm: List(Const(64), Sym(3031)), Vector(Const(10), Const(2048))
float* x3034 = (float*)myGpuMalloc(640 * sizeof(float));
float* x3035 = (float*)myMalloc(1 * sizeof(float));;
x3035[0] = 0.0f;
float* x3037 = (float*)myMalloc(1 * sizeof(float));;
x3037[0] = 1.0f;
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 10,64,2048,x3037,x939,2048,x3008,2048,x3035,x3034,10));
float* x3040 = (float*)myMalloc(1 * sizeof(float));;
x3040[0] = 1.0f;
float* x3042 = (float*)myMalloc(1 * sizeof(float));;
x3042[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 10, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 10, 1, 1));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x3040, bias_desc, x402, x3042, out_desc, x3034));
};
// Tensor 'toCPU' invocation.
float* x3046 = (float*)myMalloc(640 * sizeof(float));;
CUDA_CALL(cudaMemcpy(x3046, x3034, 640 * sizeof(float), cudaMemcpyDeviceToHost));
printf("output (size Const(64) x Const(10))\n");
float x3049 = 0.0f;
for(int x3051=0; x3051 < 640; x3051++) {
float x3052 = x3049;
float x3054 = x3046[x3051];
float x3053 = fabs(x3052);
float x3055 = fabs(x3054);
bool x3056 = x3053 > x3055;
float x3059;
if (x3056) {
x3059 = x3052;
} else {
float x3057 = x3046[x3051];
x3059 = x3057;
}
x3049 = x3059;

}
float x3063 = x3049;
printf("Max Abs: %.5f || ",x3063);
for(int x3065=0; x3065 < 10; x3065++) {
float x3066 = x3046[x3065];
printf("%.5f ",x3066);

}
printf("\n");
assert(false && "stop");

}
// Backend cleanup.
CUBLAS_CALL(cublasDestroy(cublasHandle));
CUDA_CALL(cudaFree(gpuMallocBase));
      
CUDNN_CALL(cudnnDestroy(cudnnHandle));
}
/*****************************************
  End of C Generated Code                  
*******************************************/


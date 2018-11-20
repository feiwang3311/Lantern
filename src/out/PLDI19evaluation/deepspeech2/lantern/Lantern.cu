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

__global__ void arrayFill_greg(float* data, float value, int size) {
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
float* x21 = (float*)myMalloc(32 * sizeof(float));;
for(int x23=0; x23 < 32; x23++) {
x21[x23] = 1.0f;

}
// Tensor 'toGPU' invocation.
float* x28 = (float*)myGpuMalloc(32 * sizeof(float));
CUDA_CALL(cudaMemcpy(x28, x21, 32 * sizeof(float), cudaMemcpyHostToDevice));
float* x30 = (float*)myGpuMalloc(32 * sizeof(float));
float* x31 = (float*)myGpuMalloc(32 * sizeof(float));
float* x32 = (float*)myGpuMalloc(32 * sizeof(float));
float* x33 = (float*)myGpuMalloc(32 * sizeof(float));
float* x34 = (float*)myGpuMalloc(32 * sizeof(float));
float* x35 = (float*)myMalloc(236544 * sizeof(float));;
for(int x37=0; x37 < 236544; x37++) {
float x38 = (float)rand()/RAND_MAX;
float x39 = x38 - 0.5f;
float x40 = x39 * 0.05698029f;
x35[x37] = x40;

}
// Tensor 'toGPU' invocation.
float* x45 = (float*)myGpuMalloc(236544 * sizeof(float));
CUDA_CALL(cudaMemcpy(x45, x35, 236544 * sizeof(float), cudaMemcpyHostToDevice));
float* x47 = (float*)myGpuMalloc(236544 * sizeof(float));
float* x48 = (float*)myMalloc(32 * sizeof(float));;
for(int x49=0; x49 < 32; x49++) {
x48[x49] = 1.0f;

}
// Tensor 'toGPU' invocation.
float* x54 = (float*)myGpuMalloc(32 * sizeof(float));
CUDA_CALL(cudaMemcpy(x54, x48, 32 * sizeof(float), cudaMemcpyHostToDevice));
float* x56 = (float*)myGpuMalloc(32 * sizeof(float));
float* x57 = (float*)myGpuMalloc(32 * sizeof(float));
float* x58 = (float*)myGpuMalloc(32 * sizeof(float));
float* x59 = (float*)myGpuMalloc(32 * sizeof(float));
float* x60 = (float*)myGpuMalloc(32 * sizeof(float));
printf("initial rnn input size is %d \n",672);
printf("inputSize for batchRNN is %d\n",672);
int32_t x63 = 0;
float* x64 = (float*)myMalloc(3477504 * sizeof(float));;
for(int x66=0; x66 < 3477504; x66++) {
x64[x66] = 0.01f;

}
// Tensor 'toGPU' invocation.
float* x71 = (float*)myGpuMalloc(3477504 * sizeof(float));
CUDA_CALL(cudaMemcpy(x71, x64, 3477504 * sizeof(float), cudaMemcpyHostToDevice));
float* x73 = (float*)myGpuMalloc(3477504 * sizeof(float));
int32_t x74 = x63;
float* x75 = x71+x74;
float* x76 = x73+x74;
x63 += 688128;
int32_t x78 = x63;
float* x79 = x71+x78;
float* x80 = x73+x78;
x63 += 1048576;
int32_t x82 = x63;
float* x83 = x71+x82;
float* x84 = x73+x82;
x63 += 688128;
int32_t x86 = x63;
float* x87 = x71+x86;
float* x88 = x73+x86;
x63 += 1048576;
int32_t x90 = x63;
float* x91 = x71+x90;
float* x92 = x73+x90;
x63 += 1024;
int32_t x94 = x63;
float* x95 = x71+x94;
float* x96 = x73+x94;
x63 += 1024;
int32_t x98 = x63;
float* x99 = x71+x98;
float* x100 = x73+x98;
x63 += 1024;
int32_t x102 = x63;
float* x103 = x71+x102;
float* x104 = x73+x102;
x63 += 1024;
printf("inputSize for batchRNN is %d\n",1024);
int32_t x107 = 0;
float* x108 = (float*)myMalloc(4198400 * sizeof(float));;
for(int x110=0; x110 < 4198400; x110++) {
x108[x110] = 0.01f;

}
// Tensor 'toGPU' invocation.
float* x115 = (float*)myGpuMalloc(4198400 * sizeof(float));
CUDA_CALL(cudaMemcpy(x115, x108, 4198400 * sizeof(float), cudaMemcpyHostToDevice));
float* x117 = (float*)myGpuMalloc(4198400 * sizeof(float));
int32_t x118 = x107;
float* x119 = x115+x118;
float* x120 = x117+x118;
x107 += 1048576;
int32_t x122 = x107;
float* x123 = x115+x122;
float* x124 = x117+x122;
x107 += 1048576;
int32_t x126 = x107;
float* x127 = x115+x126;
float* x128 = x117+x126;
x107 += 1048576;
int32_t x130 = x107;
float* x131 = x115+x130;
float* x132 = x117+x130;
x107 += 1048576;
int32_t x134 = x107;
float* x135 = x115+x134;
float* x136 = x117+x134;
x107 += 1024;
int32_t x138 = x107;
float* x139 = x115+x138;
float* x140 = x117+x138;
x107 += 1024;
int32_t x142 = x107;
float* x143 = x115+x142;
float* x144 = x117+x142;
x107 += 1024;
int32_t x146 = x107;
float* x147 = x115+x146;
float* x148 = x117+x146;
x107 += 1024;
printf("inputSize for batchRNN is %d\n",1024);
int32_t x151 = 0;
float* x152 = (float*)myMalloc(4198400 * sizeof(float));;
for(int x153=0; x153 < 4198400; x153++) {
x152[x153] = 0.01f;

}
// Tensor 'toGPU' invocation.
float* x158 = (float*)myGpuMalloc(4198400 * sizeof(float));
CUDA_CALL(cudaMemcpy(x158, x152, 4198400 * sizeof(float), cudaMemcpyHostToDevice));
float* x160 = (float*)myGpuMalloc(4198400 * sizeof(float));
int32_t x161 = x151;
float* x162 = x158+x161;
float* x163 = x160+x161;
x151 += 1048576;
int32_t x165 = x151;
float* x166 = x158+x165;
float* x167 = x160+x165;
x151 += 1048576;
int32_t x169 = x151;
float* x170 = x158+x169;
float* x171 = x160+x169;
x151 += 1048576;
int32_t x173 = x151;
float* x174 = x158+x173;
float* x175 = x160+x173;
x151 += 1048576;
int32_t x177 = x151;
float* x178 = x158+x177;
float* x179 = x160+x177;
x151 += 1024;
int32_t x181 = x151;
float* x182 = x158+x181;
float* x183 = x160+x181;
x151 += 1024;
int32_t x185 = x151;
float* x186 = x158+x185;
float* x187 = x160+x185;
x151 += 1024;
int32_t x189 = x151;
float* x190 = x158+x189;
float* x191 = x160+x189;
x151 += 1024;
float* x193 = (float*)myMalloc(1024 * sizeof(float));;
for(int x195=0; x195 < 1024; x195++) {
x193[x195] = 1.0f;

}
// Tensor 'toGPU' invocation.
float* x200 = (float*)myGpuMalloc(1024 * sizeof(float));
CUDA_CALL(cudaMemcpy(x200, x193, 1024 * sizeof(float), cudaMemcpyHostToDevice));
float* x202 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x203 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x204 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x205 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x206 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x207 = (float*)myMalloc(29696 * sizeof(float));;
for(int x209=0; x209 < 29696; x209++) {
float x210 = (float)rand()/RAND_MAX;
float x211 = x210 - 0.5f;
float x212 = x211 * 0.03125f;
x207[x209] = x212;

}
// Tensor 'toGPU' invocation.
float* x217 = (float*)myGpuMalloc(29696 * sizeof(float));
CUDA_CALL(cudaMemcpy(x217, x207, 29696 * sizeof(float), cudaMemcpyHostToDevice));
float* x219 = (float*)myGpuMalloc(29696 * sizeof(float));
int32_t x220 = open("/scratch/wu636/training/speech_recognition/data/test/deepspeech_train.bin",0);
int64_t x221 = fsize(x220);
printf("file size is %ld\n",x221);
char* x223 = (char*)mmap(0, x221, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x220, 0);
int64_t x224 = (long)x223;
int64_t x225 = x224;
int64_t x226 = x225;
int* x227 = (int32_t*) x226;
int64_t x228 = (int64_t)4;
x225 += x228;
int32_t x230 = x227[0];
int64_t x231 = x225;
int* x232 = (int32_t*) x231;
x225 += x228;
int32_t x234 = x232[0];
printf("data size is %d batches, %d batch size\n",200,x230);
int* x237 = (int32_t*)myMalloc(200 * sizeof(int32_t));;
int* x238 = (int32_t*)myMalloc(200 * sizeof(int32_t));;
float** x239 = (float**)myMalloc(200 * sizeof(float*));;
float** x240 = (float**)myMalloc(200 * sizeof(float*));;
int** x241 = (int**)myMalloc(200 * sizeof(int*));;
int** x242 = (int**)myMalloc(200 * sizeof(int*));;
// load data by batchs
int32_t x268 = 4 * x230;
int64_t x269 = (int64_t)x268;
for(int x245=0; x245 < 200; x245++) {
int64_t x246 = x225;
int* x247 = (int32_t*) x246;
x225 += x228;
int32_t x249 = x247[0];
x237[x245] = x249;
int64_t x251 = x225;
int* x252 = (int32_t*) x251;
x225 += x228;
int32_t x254 = x252[0];
x238[x245] = x254;
int32_t x256 = x237[x245];
int32_t x258 = x238[x245];
int64_t x260 = x225;
float* x261 = (float*) x260;
int32_t x257 = x230 * x256;
int32_t x259 = x257 * x258;
int32_t x262 = 4 * x259;
int64_t x263 = (int64_t)x262;
x225 += x263;
x239[x245] = x261;
int64_t x266 = x225;
float* x267 = (float*) x266;
x225 += x269;
x240[x245] = x267;
int64_t x272 = x225;
int* x273 = (int32_t*) x272;
x225 += x269;
x241[x245] = x273;
int* x276 = x241[x245];
int* x277 = x241[x245];
int32_t x278 = accumulate(x276, x277 + x230, 0);
int64_t x279 = x225;
int* x280 = (int32_t*) x279;
int32_t x281 = 4 * x278;
int64_t x282 = (int64_t)x281;
x225 += x282;
x242[x245] = x280;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x289 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
float x290 = (float)x289;
float x291 = x290 / 1000000.0f;
printf("Data reading (all prepare time) in %lf sec\n",x291);
double* x293 = (double*)myMalloc(1 * sizeof(double));;
double* x294 = (double*)myMalloc(1 * sizeof(double));;
int64_t x295 = (long)mallocAddr;
int64_t x296 = (long)gpuMallocAddr;
// training loop starts here
int32_t x344 = x230 * 32;
bool x403 = x230 < 0;
bool x437 = x230 > 0;
int32_t x486 = 2048 / 2;
bool x502 = x486 < 0;
bool x533 = x486 > 0;
int32_t x1366 = x230 * 20;
int32_t x235 = x230 * 200;
double x1371 = (double)x235;
int64_t x1394 = (int64_t)x235;
float x1401 = (float)x235;
for(int x299=0; x299 < 1; x299++) {
struct timeval begin_1, end_1, diff_1;
int32_t x301 = 0;
int32_t x302 = x301;
int32_t x303 = x302;
float x304 = 0.0f;
float x305 = x304;
float x306 = x305;
int32_t x307 = x299 + 1;
printf("Start training epoch %d\n",x307);
gettimeofday(&begin_1, NULL);
for(int x310=0; x310 < 200; x310++) {
int32_t x311 = x238[x310];
int32_t x312 = x237[x310];
float* x313 = x239[x310];
float* x316 = x240[x310];
int* x317 = x242[x310];
int* x318 = x241[x310];
x303 += x230;
// Tensor 'toGPU' invocation.
int32_t x314 = x312 * x311;
int32_t x315 = x230 * x314;
float* x321 = (float*)myGpuMalloc(x315 * sizeof(float));
CUDA_CALL(cudaMemcpy(x321, x313, x315 * sizeof(float), cudaMemcpyHostToDevice));
float* x323 = (float*)myGpuMalloc(2 * sizeof(float));
float* x324 = (float*)myGpuMalloc(1 * sizeof(float));
float* x325 = (float*)myGpuMalloc(1 * sizeof(float));
// allocate memory to save the final loss in CPU Tensor
float* x327 = (float*)myMalloc(1 * sizeof(float));;
bool x328 = x312 >= 41;
bool x330;
if (x328) {
bool x329 = x311 >= 11;
x330 = x329;
} else {
x330 = false;
}
if (x330) {
} else {
assert(false && "ERROR not specified");
}
int32_t x338 = x311 - 11;
int32_t x339 = x338 / 2;
int32_t x340 = x339 + 1;
int32_t x335 = x312 - 41;
int32_t x336 = x335 / 2;
int32_t x337 = x336 + 1;
int32_t x345 = x344 * x337;
int32_t x346 = x345 * x340;
float* x347 = (float*)myGpuMalloc(x346 * sizeof(float));
float* x348 = (float*)myMalloc(1 * sizeof(float));;
x348[0] = 0.0f;
float* x350 = (float*)myMalloc(1 * sizeof(float));;
x350[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x230, 1, x312, x311));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    32, 1, 41, 11));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x230, 32, x337, x340));

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
    x350, in_desc, x321, filt_desc, x18,
    conv_desc, algo, ws_data, ws_size,
    x348, out_desc, x347));
};
float* x353 = (float*)myGpuMalloc(x346 * sizeof(float));
int32_t x341 = x337 * x340;
int32_t x342 = 32 * x341;
int32_t x343 = x230 * x342;
float* x354 = (float*)myGpuMalloc(x343 * sizeof(float));
float* x355 = (float*)myGpuMalloc(32 * sizeof(float));
float* x356 = (float*)myGpuMalloc(32 * sizeof(float));
float* x357 = (float*)myMalloc(1 * sizeof(float));;
x357[0] = 0.0f;
float* x359 = (float*)myMalloc(1 * sizeof(float));;
x359[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x230, 32, x337, x340));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x230, 32, x337, x340));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 32, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x359, x357, in_desc, x347, out_desc, x354, sbmv_desc, x28,
    x31, 0.1, x33, x34, 1.0E-5,
    x355, x356));
};
float* x362 = (float*)myGpuMalloc(x346 * sizeof(float));
hardTanh<<<28, 512>>>(x354, x354, 0.0, 20.0, true);
bool x364 = x337 >= 21;
bool x366;
if (x364) {
bool x365 = x340 >= 11;
x366 = x365;
} else {
x366 = false;
}
if (x366) {
} else {
assert(false && "ERROR not specified");
}
int32_t x374 = x340 - 11;
int32_t x375 = x374 / 1;
int32_t x376 = x375 + 1;
int32_t x371 = x337 - 21;
int32_t x372 = x371 / 2;
int32_t x373 = x372 + 1;
int32_t x380 = x344 * x373;
int32_t x381 = x380 * x376;
float* x382 = (float*)myGpuMalloc(x381 * sizeof(float));
float* x383 = (float*)myMalloc(1 * sizeof(float));;
x383[0] = 0.0f;
float* x385 = (float*)myMalloc(1 * sizeof(float));;
x385[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x230, 32, x337, x340));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    32, 32, 21, 11));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x230, 32, x373, x376));

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
    x385, in_desc, x354, filt_desc, x45,
    conv_desc, algo, ws_data, ws_size,
    x383, out_desc, x382));
};
float* x388 = (float*)myGpuMalloc(x381 * sizeof(float));
int32_t x377 = x373 * x376;
int32_t x378 = 32 * x377;
int32_t x379 = x230 * x378;
float* x389 = (float*)myGpuMalloc(x379 * sizeof(float));
float* x390 = (float*)myGpuMalloc(32 * sizeof(float));
float* x391 = (float*)myGpuMalloc(32 * sizeof(float));
float* x392 = (float*)myMalloc(1 * sizeof(float));;
x392[0] = 0.0f;
float* x394 = (float*)myMalloc(1 * sizeof(float));;
x394[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x230, 32, x373, x376));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x230, 32, x373, x376));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 32, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x394, x392, in_desc, x382, out_desc, x389, sbmv_desc, x54,
    x57, 0.1, x59, x60, 1.0E-5,
    x390, x391));
};
float* x397 = (float*)myGpuMalloc(x381 * sizeof(float));
hardTanh<<<28, 512>>>(x389, x389, 0.0, 20.0, true);
// after conv ops
int32_t x401 = 0;
int32_t x402 = 1;
if (x403) {
x401 += 1;
} else {
x402 *= x230;
}
int32_t x400 = 32 * x373;
bool x409 = x400 < 0;
if (x409) {
x401 += 1;
} else {
x402 *= x400;
}
bool x415 = x376 < 0;
if (x415) {
x401 += 1;
} else {
x402 *= x376;
}
int32_t x421 = x401;
bool x422 = x421 >= 2;
if (x422) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x428 = x421 == 0;
if (x428) {
int32_t x429 = x402;
bool x430 = x429 == x379;
if (x430) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x441;
if (x437) {
x441 = x230;
} else {
int32_t x438 = x402;
int32_t x439 = x379 / x438;
x441 = x439;
}
bool x442 = x400 > 0;
int32_t x446;
if (x442) {
x446 = x400;
} else {
int32_t x443 = x402;
int32_t x444 = x379 / x443;
x446 = x444;
}
bool x447 = x376 > 0;
int32_t x451;
if (x447) {
x451 = x376;
} else {
int32_t x448 = x402;
int32_t x449 = x379 / x448;
x451 = x449;
}
int32_t x452 = x446 * x451;
int32_t x453 = x441 * x452;
float* x454 = (float*)myGpuMalloc(x453 * sizeof(float));
int* x457 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
int32_t x455 = x441 * x446;
x457[2] = x455;
x457[0] = x446;
x457[1] = 1;
x457[3] = 1;
float* x462 = (float*)myMalloc(1 * sizeof(float));;
x462[0] = 1.0f;
float* x464 = (float*)myMalloc(0 * sizeof(float));;
x464[0] = 0.0f;
int32_t x466 = x457[0];
int32_t x467 = x457[1];
int32_t x468 = x457[2];
int32_t x469 = x457[3];

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
    in_desc, CUDNN_DATA_FLOAT,
    x441, x446, x451, 1,
    x452, x451, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
    out_desc, CUDNN_DATA_FLOAT,
    x441, x446, x451, 1,
    x466, x467, x468, x469));

CUDNN_CALL(cudnnTransformTensor(
    cudnnHandle, x462, in_desc, x389, x464, out_desc, x454));
};
int32_t x471 = x451 * x441;
int32_t x472 = x471 * x446;
float* x473 = (float*)myGpuMalloc(x472 * sizeof(float));
// after resize and permute
float* x475 = (float*)NULL;
float* x476 = (float*)NULL;
float* x477 = (float*)NULL;
int32_t x480 = x471 * 2048;
float* x481 = (float*)myGpuMalloc(x480 * sizeof(float));
float* x482 = (float*)NULL;
int32_t x483 = 0;

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
int32_t seqLength = x451;
int32_t batchSize = x441;
int32_t inputSize = x446;

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
x482 = (float*)reserveSpace;
x483 = (int)reserveSize;
CUDNN_CALL(cudnnRNNForwardTraining(
    cudnnHandle, rnn_desc, seqLength, x_descs, x454,
    hx_desc,x475, cx_desc,x476, w_desc, x71, y_descs, x481,
    hy_desc,x477, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
};
float* x485 = (float*)myGpuMalloc(x480 * sizeof(float));
int32_t x487 = 0;
int32_t x488 = 1;
bool x489 = x451 < 0;
if (x489) {
x487 += 1;
} else {
x488 *= x451;
}
bool x495 = x441 < 0;
if (x495) {
x487 += 1;
} else {
x488 *= x441;
}
x488 *= 2;
if (x502) {
x487 += 1;
} else {
x488 *= x486;
}
int32_t x508 = x487;
bool x509 = x508 >= 2;
if (x509) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x514 = x508 == 0;
int32_t x478 = x441 * 2048;
int32_t x479 = x451 * x478;
if (x514) {
int32_t x515 = x488;
bool x516 = x515 == x479;
if (x516) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x523 = x451 > 0;
int32_t x527;
if (x523) {
x527 = x451;
} else {
int32_t x524 = x488;
int32_t x525 = x479 / x524;
x527 = x525;
}
bool x528 = x441 > 0;
int32_t x532;
if (x528) {
x532 = x441;
} else {
int32_t x529 = x488;
int32_t x530 = x479 / x529;
x532 = x530;
}
int32_t x537;
if (x533) {
x537 = x486;
} else {
int32_t x534 = x488;
int32_t x535 = x479 / x534;
x537 = x535;
}
int32_t x541 = 0;
int32_t x542 = 1;
bool x543 = x527 < 0;
if (x543) {
x541 += 1;
} else {
x542 *= x527;
}
bool x549 = x532 < 0;
if (x549) {
x541 += 1;
} else {
x542 *= x532;
}
x542 *= 2;
bool x556 = x537 < 0;
if (x556) {
x541 += 1;
} else {
x542 *= x537;
}
int32_t x562 = x541;
bool x563 = x562 >= 2;
if (x563) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x568 = x562 == 0;
int32_t x538 = 2 * x537;
int32_t x539 = x532 * x538;
int32_t x540 = x527 * x539;
if (x568) {
int32_t x569 = x542;
bool x570 = x569 == x540;
if (x570) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x577 = x527 > 0;
int32_t x581;
if (x577) {
x581 = x527;
} else {
int32_t x578 = x542;
int32_t x579 = x540 / x578;
x581 = x579;
}
bool x582 = x532 > 0;
int32_t x586;
if (x582) {
x586 = x532;
} else {
int32_t x583 = x542;
int32_t x584 = x540 / x583;
x586 = x584;
}
bool x587 = x537 > 0;
int32_t x591;
if (x587) {
x591 = x537;
} else {
int32_t x588 = x542;
int32_t x589 = x540 / x588;
x591 = x589;
}
int32_t x597 = x581 * x586;
int32_t x598 = x597 * x591;
float* x599 = (float*)myGpuMalloc(x598 * sizeof(float));
float* x600 = (float*)myMalloc(1 * sizeof(float));;
x600[0] = 0.0f;
float* x602 = (float*)myMalloc(1 * sizeof(float));;
x602[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x581, x586, 2, x591));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x581, x586, 1, x591));

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
    x602, x_desc, x481, x600, out_desc, x599));
};
float* x605 = (float*)myGpuMalloc(x598 * sizeof(float));
float* x606 = (float*)NULL;
float* x607 = (float*)NULL;
float* x608 = (float*)NULL;
int32_t x611 = x597 * 2048;
float* x612 = (float*)myGpuMalloc(x611 * sizeof(float));
float* x613 = (float*)NULL;
int32_t x614 = 0;

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
int32_t seqLength = x581;
int32_t batchSize = x586;
int32_t inputSize = x591;

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
x613 = (float*)reserveSpace;
x614 = (int)reserveSize;
CUDNN_CALL(cudnnRNNForwardTraining(
    cudnnHandle, rnn_desc, seqLength, x_descs, x599,
    hx_desc,x606, cx_desc,x607, w_desc, x115, y_descs, x612,
    hy_desc,x608, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
};
float* x616 = (float*)myGpuMalloc(x611 * sizeof(float));
int32_t x617 = 0;
int32_t x618 = 1;
bool x619 = x581 < 0;
if (x619) {
x617 += 1;
} else {
x618 *= x581;
}
bool x625 = x586 < 0;
if (x625) {
x617 += 1;
} else {
x618 *= x586;
}
x618 *= 2;
if (x502) {
x617 += 1;
} else {
x618 *= x486;
}
int32_t x637 = x617;
bool x638 = x637 >= 2;
if (x638) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x643 = x637 == 0;
int32_t x609 = x586 * 2048;
int32_t x610 = x581 * x609;
if (x643) {
int32_t x644 = x618;
bool x645 = x644 == x610;
if (x645) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x652 = x581 > 0;
int32_t x656;
if (x652) {
x656 = x581;
} else {
int32_t x653 = x618;
int32_t x654 = x610 / x653;
x656 = x654;
}
bool x657 = x586 > 0;
int32_t x661;
if (x657) {
x661 = x586;
} else {
int32_t x658 = x618;
int32_t x659 = x610 / x658;
x661 = x659;
}
int32_t x665;
if (x533) {
x665 = x486;
} else {
int32_t x662 = x618;
int32_t x663 = x610 / x662;
x665 = x663;
}
int32_t x669 = 0;
int32_t x670 = 1;
bool x671 = x656 < 0;
if (x671) {
x669 += 1;
} else {
x670 *= x656;
}
bool x677 = x661 < 0;
if (x677) {
x669 += 1;
} else {
x670 *= x661;
}
x670 *= 2;
bool x684 = x665 < 0;
if (x684) {
x669 += 1;
} else {
x670 *= x665;
}
int32_t x690 = x669;
bool x691 = x690 >= 2;
if (x691) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x696 = x690 == 0;
int32_t x666 = 2 * x665;
int32_t x667 = x661 * x666;
int32_t x668 = x656 * x667;
if (x696) {
int32_t x697 = x670;
bool x698 = x697 == x668;
if (x698) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x705 = x656 > 0;
int32_t x709;
if (x705) {
x709 = x656;
} else {
int32_t x706 = x670;
int32_t x707 = x668 / x706;
x709 = x707;
}
bool x710 = x661 > 0;
int32_t x714;
if (x710) {
x714 = x661;
} else {
int32_t x711 = x670;
int32_t x712 = x668 / x711;
x714 = x712;
}
bool x715 = x665 > 0;
int32_t x719;
if (x715) {
x719 = x665;
} else {
int32_t x716 = x670;
int32_t x717 = x668 / x716;
x719 = x717;
}
int32_t x725 = x709 * x714;
int32_t x726 = x725 * x719;
float* x727 = (float*)myGpuMalloc(x726 * sizeof(float));
float* x728 = (float*)myMalloc(1 * sizeof(float));;
x728[0] = 0.0f;
float* x730 = (float*)myMalloc(1 * sizeof(float));;
x730[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x709, x714, 2, x719));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x709, x714, 1, x719));

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
    x730, x_desc, x612, x728, out_desc, x727));
};
float* x733 = (float*)myGpuMalloc(x726 * sizeof(float));
float* x734 = (float*)NULL;
float* x735 = (float*)NULL;
float* x736 = (float*)NULL;
int32_t x739 = x725 * 2048;
float* x740 = (float*)myGpuMalloc(x739 * sizeof(float));
float* x741 = (float*)NULL;
int32_t x742 = 0;

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
int32_t seqLength = x709;
int32_t batchSize = x714;
int32_t inputSize = x719;

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
x741 = (float*)reserveSpace;
x742 = (int)reserveSize;
CUDNN_CALL(cudnnRNNForwardTraining(
    cudnnHandle, rnn_desc, seqLength, x_descs, x727,
    hx_desc,x734, cx_desc,x735, w_desc, x158, y_descs, x740,
    hy_desc,x736, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
};
float* x744 = (float*)myGpuMalloc(x739 * sizeof(float));
int32_t x745 = 0;
int32_t x746 = 1;
bool x747 = x709 < 0;
if (x747) {
x745 += 1;
} else {
x746 *= x709;
}
bool x753 = x714 < 0;
if (x753) {
x745 += 1;
} else {
x746 *= x714;
}
x746 *= 2;
if (x502) {
x745 += 1;
} else {
x746 *= x486;
}
int32_t x765 = x745;
bool x766 = x765 >= 2;
if (x766) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x771 = x765 == 0;
int32_t x737 = x714 * 2048;
int32_t x738 = x709 * x737;
if (x771) {
int32_t x772 = x746;
bool x773 = x772 == x738;
if (x773) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x780 = x709 > 0;
int32_t x784;
if (x780) {
x784 = x709;
} else {
int32_t x781 = x746;
int32_t x782 = x738 / x781;
x784 = x782;
}
bool x785 = x714 > 0;
int32_t x789;
if (x785) {
x789 = x714;
} else {
int32_t x786 = x746;
int32_t x787 = x738 / x786;
x789 = x787;
}
int32_t x793;
if (x533) {
x793 = x486;
} else {
int32_t x790 = x746;
int32_t x791 = x738 / x790;
x793 = x791;
}
int32_t x797 = 0;
int32_t x798 = 1;
bool x799 = x784 < 0;
if (x799) {
x797 += 1;
} else {
x798 *= x784;
}
bool x805 = x789 < 0;
if (x805) {
x797 += 1;
} else {
x798 *= x789;
}
x798 *= 2;
bool x812 = x793 < 0;
if (x812) {
x797 += 1;
} else {
x798 *= x793;
}
int32_t x818 = x797;
bool x819 = x818 >= 2;
if (x819) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x824 = x818 == 0;
int32_t x794 = 2 * x793;
int32_t x795 = x789 * x794;
int32_t x796 = x784 * x795;
if (x824) {
int32_t x825 = x798;
bool x826 = x825 == x796;
if (x826) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x833 = x784 > 0;
int32_t x837;
if (x833) {
x837 = x784;
} else {
int32_t x834 = x798;
int32_t x835 = x796 / x834;
x837 = x835;
}
bool x838 = x789 > 0;
int32_t x842;
if (x838) {
x842 = x789;
} else {
int32_t x839 = x798;
int32_t x840 = x796 / x839;
x842 = x840;
}
bool x843 = x793 > 0;
int32_t x847;
if (x843) {
x847 = x793;
} else {
int32_t x844 = x798;
int32_t x845 = x796 / x844;
x847 = x845;
}
int32_t x853 = x837 * x842;
int32_t x854 = x853 * x847;
float* x855 = (float*)myGpuMalloc(x854 * sizeof(float));
float* x856 = (float*)myMalloc(1 * sizeof(float));;
x856[0] = 0.0f;
float* x858 = (float*)myMalloc(1 * sizeof(float));;
x858[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x837, x842, 2, x847));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x837, x842, 1, x847));

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
    x858, x_desc, x740, x856, out_desc, x855));
};
float* x861 = (float*)myGpuMalloc(x854 * sizeof(float));
// after RNN layers
// after maybe lookahead
int32_t x864 = 0;
int32_t x865 = 1;
bool x866 = x853 < 0;
if (x866) {
x864 += 1;
} else {
x865 *= x853;
}
bool x872 = x847 < 0;
if (x872) {
x864 += 1;
} else {
x865 *= x847;
}
int32_t x878 = x864;
bool x879 = x878 >= 2;
if (x879) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x884 = x878 == 0;
int32_t x851 = x842 * x847;
int32_t x852 = x837 * x851;
if (x884) {
int32_t x885 = x865;
bool x886 = x885 == x852;
if (x886) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x893 = x853 > 0;
int32_t x897;
if (x893) {
x897 = x853;
} else {
int32_t x894 = x865;
int32_t x895 = x852 / x894;
x897 = x895;
}
bool x898 = x847 > 0;
int32_t x902;
if (x898) {
x902 = x847;
} else {
int32_t x899 = x865;
int32_t x900 = x852 / x899;
x902 = x900;
}
bool x904 = x902 == 1024;
if (x904) {
} else {
assert(false && "BatchNorm1D input should be rank2, with shape 1 same as dimSize, got %d : %d");
}
bool x909 = 1024 == x902;
if (x909) {
} else {
assert(false && "scale should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(897) x Sym(902)");
}
if (x909) {
} else {
assert(false && "bias should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(897) x Sym(902)");
}
if (x909) {
} else {
assert(false && "runningMean should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(897) x Sym(902)");
}
if (x909) {
} else {
assert(false && "runningVar should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(897) x Sym(902)");
}
int32_t x903 = x897 * x902;
float* x923 = (float*)myGpuMalloc(x903 * sizeof(float));
float* x924 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x925 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x926 = (float*)myMalloc(1 * sizeof(float));;
x926[0] = 0.0f;
float* x928 = (float*)myMalloc(1 * sizeof(float));;
x928[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x897, x902, 1, 1));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,
    x928, x926, in_desc, x855, in_desc, x923, sbmv_desc, x200,
    x203, 0.1, x205, x206, 1.0E-5,
    x924, x925));
};
float* x931 = (float*)myGpuMalloc(x903 * sizeof(float));
int32_t x932 = x897 * 29;
float* x933 = (float*)myGpuMalloc(x932 * sizeof(float));
float* x934 = (float*)myMalloc(1 * sizeof(float));;
x934[0] = 0.0f;
float* x936 = (float*)myMalloc(1 * sizeof(float));;
x936[0] = 1.0f;
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 29,x897,1024,x936,x217,29,x923,1024,x934,x933,29));
float* x939 = (float*)myGpuMalloc(x932 * sizeof(float));
int32_t x940 = 0;
int32_t x941 = 1;
bool x942 = x837 < 0;
if (x942) {
x940 += 1;
} else {
x941 *= x837;
}
bool x948 = x842 < 0;
if (x948) {
x940 += 1;
} else {
x941 *= x842;
}
x941 *= 29;
int32_t x955 = x940;
bool x956 = x955 >= 2;
if (x956) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x961 = x955 == 0;
if (x961) {
int32_t x962 = x941;
bool x963 = x962 == x932;
if (x963) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x970 = x837 > 0;
int32_t x974;
if (x970) {
x974 = x837;
} else {
int32_t x971 = x941;
int32_t x972 = x932 / x971;
x974 = x972;
}
bool x975 = x842 > 0;
int32_t x979;
if (x975) {
x979 = x842;
} else {
int32_t x976 = x941;
int32_t x977 = x932 / x976;
x979 = x977;
}
int32_t x983 = 0;
int32_t x984 = 1;
int32_t x982 = x974 * x979;
bool x985 = x982 < 0;
if (x985) {
x983 += 1;
} else {
x984 *= x982;
}
x984 *= 29;
x984 *= 1;
x984 *= 1;
int32_t x994 = x983;
bool x995 = x994 >= 2;
if (x995) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1000 = x994 == 0;
int32_t x980 = x979 * 29;
int32_t x981 = x974 * x980;
if (x1000) {
int32_t x1001 = x984;
bool x1002 = x1001 == x981;
if (x1002) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x1009 = x982 > 0;
int32_t x1013;
if (x1009) {
x1013 = x982;
} else {
int32_t x1010 = x984;
int32_t x1011 = x981 / x1010;
x1013 = x1011;
}
float* x1015 = (float*)myMalloc(1 * sizeof(float));;
x1015[0] = 0.0f;
float* x1017 = (float*)myMalloc(1 * sizeof(float));;
x1017[0] = 1.0f;
int32_t x1014 = x1013 * 29;
float* x1019 = (float*)myGpuMalloc(x1014 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x1013, 29, 1, 1));
CUDNN_CALL(cudnnSoftmaxForward(
    cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
    x1017, x_desc, x933, x1015, x_desc, x1019));
};
int32_t x1021 = 0;
int32_t x1022 = 1;
bool x1023 = x974 < 0;
if (x1023) {
x1021 += 1;
} else {
x1022 *= x974;
}
bool x1029 = x979 < 0;
if (x1029) {
x1021 += 1;
} else {
x1022 *= x979;
}
x1022 *= 29;
int32_t x1036 = x1021;
bool x1037 = x1036 >= 2;
if (x1037) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1042 = x1036 == 0;
if (x1042) {
int32_t x1043 = x1022;
bool x1044 = x1043 == x1014;
if (x1044) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x1051 = x974 > 0;
int32_t x1055;
if (x1051) {
x1055 = x974;
} else {
int32_t x1052 = x1022;
int32_t x1053 = x1014 / x1052;
x1055 = x1053;
}
bool x1056 = x979 > 0;
int32_t x1060;
if (x1056) {
x1060 = x979;
} else {
int32_t x1057 = x1022;
int32_t x1058 = x1014 / x1057;
x1060 = x1058;
}
int32_t x1063 = x1055 * x1060;
int32_t x1064 = x1063 * 29;
float* x1065 = (float*)myGpuMalloc(x1064 * sizeof(float));
// before CTC loss
int* x1067 = (int32_t*)myMalloc(x1060 * sizeof(int32_t));;
float x1071 = (float)x1055;
for(int x1069=0; x1069 < x1060; x1069++) {
float x1070 = x316[x1069];
float x1072 = x1070 * x1071;
int32_t x1073 = (int)x1072;
x1067[x1069] = x1073;

}
bool x1077 = x1060 <= 256;
if (x1077) {
} else {
printf("'cudnnGetCTCLossWorkspaceSize' requires batch size less than 256, got %d\n\n",x1060);
assert(false && "");
}
float* x1083 = (float*)myGpuMalloc(x1060 * sizeof(float));

{
cudnnTensorDescriptor_t probs_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&probs_desc));
int probs_dims[] = {x1055, x1060, 29};
int probs_strides[] = {probs_dims[1] * probs_dims[2], probs_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    probs_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, probs_dims, probs_strides));

cudnnTensorDescriptor_t grad_desc = probs_desc;

cudnnCTCLossDescriptor_t ctc_desc;
CUDNN_CALL(cudnnCreateCTCLossDescriptor(&ctc_desc));
CUDNN_CALL(cudnnSetCTCLossDescriptor(ctc_desc, CUDNN_DATA_FLOAT));
size_t wsSize;
CUDNN_CALL(cudnnGetCTCLossWorkspaceSize(
    cudnnHandle, probs_desc, grad_desc, x317, x318, x1067,
    CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, &wsSize));
void *ws = myGpuMalloc(wsSize);

CUDNN_CALL(cudnnCTCLoss(
    cudnnHandle, probs_desc, x1019, x317, x318, x1067,
    x1083, grad_desc, x1065, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, ws, wsSize));
};
float* x1085 = (float*)myGpuMalloc(1 * sizeof(float));
float* x1086 = (float*)myMalloc(1 * sizeof(float));;
x1086[0] = 0.0f;
float* x1088 = (float*)myMalloc(1 * sizeof(float));;
x1088[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x1060, 1, 1, 1));

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
    x1088, x_desc, x1083, x1086, out_desc, x1085));
};
// after CTC loss
float* x1092 = (float*)myGpuMalloc(1 * sizeof(float));
// make sure the size of loss is 1
arrayFill_greg<<<28, 512>>>(x1092, 1.0f, 1);
// backend is lantern.TensorDslCudnn$BackendCudnn@2c114c25
CUDA_CALL(cudaMemcpy(x327, x1085, 1 * sizeof(float), cudaMemcpyDeviceToHost));
int32_t x1097 = 0;
int32_t x1098 = 1;
if (x985) {
x1097 += 1;
} else {
x1098 *= x982;
}
x1098 *= 29;
x1098 *= 1;
x1098 *= 1;
int32_t x1107 = x1097;
bool x1108 = x1107 >= 2;
if (x1108) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1113 = x1107 == 0;
if (x1113) {
int32_t x1114 = x1098;
bool x1115 = x1114 == x981;
if (x1115) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1125;
if (x1009) {
x1125 = x982;
} else {
int32_t x1122 = x1098;
int32_t x1123 = x981 / x1122;
x1125 = x1123;
}
int32_t x1127 = 0;
int32_t x1128 = 1;
if (x985) {
x1127 += 1;
} else {
x1128 *= x982;
}
x1128 *= 29;
x1128 *= 1;
x1128 *= 1;
int32_t x1137 = x1127;
bool x1138 = x1137 >= 2;
if (x1138) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1143 = x1137 == 0;
if (x1143) {
int32_t x1144 = x1128;
bool x1145 = x1144 == x981;
if (x1145) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1155;
if (x1009) {
x1155 = x982;
} else {
int32_t x1152 = x1128;
int32_t x1153 = x981 / x1152;
x1155 = x1153;
}
int32_t x1157 = 0;
int32_t x1158 = 1;
bool x1159 = x1063 < 0;
if (x1159) {
x1157 += 1;
} else {
x1158 *= x1063;
}
x1158 *= 29;
x1158 *= 1;
x1158 *= 1;
int32_t x1168 = x1157;
bool x1169 = x1168 >= 2;
if (x1169) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1174 = x1168 == 0;
if (x1174) {
int32_t x1175 = x1158;
int32_t x1061 = x1060 * 29;
int32_t x1062 = x1055 * x1061;
bool x1176 = x1175 == x1062;
if (x1176) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x1183 = x1063 > 0;
int32_t x1187;
if (x1183) {
x1187 = x1063;
} else {
int32_t x1184 = x1158;
int32_t x1061 = x1060 * 29;
int32_t x1062 = x1055 * x1061;
int32_t x1185 = x1062 / x1184;
x1187 = x1185;
}
int32_t x1189 = 0;
int32_t x1190 = 1;
if (x1159) {
x1189 += 1;
} else {
x1190 *= x1063;
}
x1190 *= 29;
x1190 *= 1;
x1190 *= 1;
int32_t x1199 = x1189;
bool x1200 = x1199 >= 2;
if (x1200) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1205 = x1199 == 0;
if (x1205) {
int32_t x1206 = x1190;
int32_t x1061 = x1060 * 29;
int32_t x1062 = x1055 * x1061;
bool x1207 = x1206 == x1062;
if (x1207) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1217;
if (x1183) {
x1217 = x1063;
} else {
int32_t x1214 = x1190;
int32_t x1061 = x1060 * 29;
int32_t x1062 = x1055 * x1061;
int32_t x1215 = x1062 / x1214;
x1217 = x1215;
}
bool x1219 = x1125 == x1187;
bool x1220;
if (x1219) {
x1220 = true;
} else {
x1220 = false;
}
bool x1221;
if (x1220) {
x1221 = true;
} else {
x1221 = false;
}
bool x1222;
if (x1221) {
x1222 = true;
} else {
x1222 = false;
}
if (x1222) {
} else {
printf("$errorPrefix: tensor shapes are not equal %s, %s\n\n"," x Sym(1125) x Const(29) x Const(1) x Const(1)"," x Sym(1187) x Const(29) x Const(1) x Const(1)");
assert(false && "");
}
bool x1228 = x1155 == x1217;
bool x1229;
if (x1228) {
x1229 = true;
} else {
x1229 = false;
}
bool x1230;
if (x1229) {
x1230 = true;
} else {
x1230 = false;
}
bool x1231;
if (x1230) {
x1231 = true;
} else {
x1231 = false;
}
if (x1231) {
} else {
printf("$errorPrefix: tensor shapes are not equal %s, %s\n\n"," x Sym(1155) x Const(29) x Const(1) x Const(1)"," x Sym(1217) x Const(29) x Const(1) x Const(1)");
assert(false && "");
}
float* x1237 = (float*)myMalloc(1 * sizeof(float));;
x1237[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x1125, 29, 1, 1));
CUDNN_CALL(cudnnSoftmaxBackward(
    cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
    x1237, x_desc, x1019, x_desc, x1065,
    x1237, x_desc, x939));
};
float* x1240 = (float*)myMalloc(1 * sizeof(float));;
x1240[0] = 0.0f;
float* x1242 = (float*)myMalloc(1 * sizeof(float));;
x1242[0] = 1.0f;
// backprop of matrix-matrix-dot
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, x902,x897,29,x1242,x217,29,x939,29,x1242,x931,x902));
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 29,x902,x897,x1242,x939,29,x923,x902,x1242,x219,29));
float* x1247 = (float*)myMalloc(1 * sizeof(float));;
x1247[0] = 0.0f;
float* x1249 = (float*)myMalloc(1 * sizeof(float));;
x1249[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x897, x902, 1, 1));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,
    x1249, x1249, x1249, x1249, in_desc, x855,
    in_desc, x931, in_desc, x861, sbmv_desc, x200,
    x202,x204, 1.0E-5, x924, x925));
};
// backprop for sum on dim op
sum_grad<<<28, 512>>>(x744, x784, x789, 2, x793, x796, x861, x851, x847, 1, 2);
;
float* x1254 = (float*)NULL;
float* x1255 = (float*)NULL;

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
int32_t seqLength = x709;
int32_t batchSize = x714;
int32_t inputSize = x719;

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
    cudnnHandle, rnn_desc, seqLength, y_descs, x740, y_descs, x744,
    dhy_desc, NULL, dcy_desc, NULL, w_desc, x158, hx_desc, x1254,
    cx_desc, x1255, dx_descs, x733, dhx_desc, NULL, dcx_desc, NULL,
    workspace, workspaceSize, x741, x742));
};
float* x1257 = (float*)NULL;

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
int32_t seqLength = x709;
int32_t batchSize = x714;
int32_t inputSize = x719;

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
    cudnnHandle, rnn_desc, seqLength, x_descs, x727, hx_desc, x1257,
    y_descs, x740, workspace, workspaceSize,
    dw_desc, x160, x741, x742));
};
// backprop for sum on dim op
int32_t x723 = x714 * x719;
sum_grad<<<28, 512>>>(x616, x656, x661, 2, x665, x668, x733, x723, x719, 1, 2);
;
float* x1261 = (float*)NULL;
float* x1262 = (float*)NULL;

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
int32_t seqLength = x581;
int32_t batchSize = x586;
int32_t inputSize = x591;

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
    cudnnHandle, rnn_desc, seqLength, y_descs, x612, y_descs, x616,
    dhy_desc, NULL, dcy_desc, NULL, w_desc, x115, hx_desc, x1261,
    cx_desc, x1262, dx_descs, x605, dhx_desc, NULL, dcx_desc, NULL,
    workspace, workspaceSize, x613, x614));
};
float* x1264 = (float*)NULL;

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
int32_t seqLength = x581;
int32_t batchSize = x586;
int32_t inputSize = x591;

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
    cudnnHandle, rnn_desc, seqLength, x_descs, x599, hx_desc, x1264,
    y_descs, x612, workspace, workspaceSize,
    dw_desc, x117, x613, x614));
};
// backprop for sum on dim op
int32_t x595 = x586 * x591;
sum_grad<<<28, 512>>>(x485, x527, x532, 2, x537, x540, x605, x595, x591, 1, 2);
;
float* x1268 = (float*)NULL;
float* x1269 = (float*)NULL;

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
int32_t seqLength = x451;
int32_t batchSize = x441;
int32_t inputSize = x446;

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
    cudnnHandle, rnn_desc, seqLength, y_descs, x481, y_descs, x485,
    dhy_desc, NULL, dcy_desc, NULL, w_desc, x71, hx_desc, x1268,
    cx_desc, x1269, dx_descs, x473, dhx_desc, NULL, dcx_desc, NULL,
    workspace, workspaceSize, x482, x483));
};
float* x1271 = (float*)NULL;

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
int32_t seqLength = x451;
int32_t batchSize = x441;
int32_t inputSize = x446;

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
    cudnnHandle, rnn_desc, seqLength, x_descs, x454, hx_desc, x1271,
    y_descs, x481, workspace, workspaceSize,
    dw_desc, x73, x482, x483));
};
// backprop for permute WrappedArray(2, 0, 1)
int* x1274 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x1274[2] = x455;
x1274[0] = x446;
x1274[1] = 1;
x1274[3] = 1;
float* x1279 = (float*)myMalloc(1 * sizeof(float));;
x1279[0] = 1.0f;
int32_t x1281 = x1274[0];
int32_t x1282 = x1274[1];
int32_t x1283 = x1274[2];
int32_t x1284 = x1274[3];

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
    in_desc, CUDNN_DATA_FLOAT,
    x441, x446, x451, 1,
    x1281, x1282, x1283, x1284));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
    out_desc, CUDNN_DATA_FLOAT,
    x441, x446, x451, 1,
    x452, x451, 1, 1));

CUDNN_CALL(cudnnTransformTensor(
    cudnnHandle, x1279, in_desc, x473, x1279, out_desc, x397));
};
hardTanh_grad<<<28, 512>>>(x389, x397, x397, 0.0, 20.0, x379, true);
float* x1287 = (float*)myMalloc(1 * sizeof(float));;
x1287[0] = 0.0f;
float* x1289 = (float*)myMalloc(1 * sizeof(float));;
x1289[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x230, 32, x373, x376));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x230, 32, x373, x376));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 32, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1289, x1289, x1289, x1289, in_desc, x382,
    out_desc, x397, in_desc, x388, sbmv_desc, x54,
    x56,x58, 1.0E-5, x390, x391));
};
// conv2D back-propagate
float* x1293 = (float*)myMalloc(1 * sizeof(float));;
x1293[0] = 1.0f;

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
    x230, 32, x337, x340));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x230, 32, x373, x376));

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
    x1293, filt_desc, x45, grad_out_desc, x388,
    conv_desc, algo, ws_data, ws_size,
    x1293, grad_in_desc, x362));
};
float* x1296 = (float*)myMalloc(1 * sizeof(float));;
x1296[0] = 1.0f;

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
    x230, 32, x373, x376));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x230, 32, x337, x340));

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
    x1296, in_desc, x354, grad_out_desc, x388,
    conv_desc, algo, ws_data, ws_size,
    x1296, grad_filt_desc, x47));
};
hardTanh_grad<<<28, 512>>>(x354, x362, x362, 0.0, 20.0, x343, true);
float* x1300 = (float*)myMalloc(1 * sizeof(float));;
x1300[0] = 0.0f;
float* x1302 = (float*)myMalloc(1 * sizeof(float));;
x1302[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x230, 32, x337, x340));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x230, 32, x337, x340));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 32, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1302, x1302, x1302, x1302, in_desc, x347,
    out_desc, x362, in_desc, x353, sbmv_desc, x28,
    x30,x32, 1.0E-5, x355, x356));
};
// conv2D back-propagate
float* x1306 = (float*)myMalloc(1 * sizeof(float));;
x1306[0] = 1.0f;

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
    x230, 32, x337, x340));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x230, 1, x312, x311));

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
    x1306, in_desc, x321, grad_out_desc, x353,
    conv_desc, algo, ws_data, ws_size,
    x1306, grad_filt_desc, x20));
};
float x1309 = x327[0];
x306 += x1309;
float* x1311 = (float*)myMalloc(1 * sizeof(float));;
x1311[0] = 1.0f;
float* x1313 = (float*)myMalloc(1 * sizeof(float));;
x1313[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 451,32,x1311,x18,451,x1313, x20, 451, x18,451));
arrayFill_greg<<<28, 512>>>(x20, 0.0f, 14432);
float* x1317 = (float*)myMalloc(1 * sizeof(float));;
x1317[0] = 1.0f;
float* x1319 = (float*)myMalloc(1 * sizeof(float));;
x1319[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 7392,32,x1317,x45,7392,x1319, x47, 7392, x45,7392));
arrayFill_greg<<<28, 512>>>(x47, 0.0f, 236544);
float* x1323 = (float*)myMalloc(1 * sizeof(float));;
x1323[0] = 1.0f;
float* x1325 = (float*)myMalloc(1 * sizeof(float));;
x1325[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1323,x54,1,x1325, x56, 1, x54,1));
arrayFill_greg<<<28, 512>>>(x56, 0.0f, 32);
float* x1329 = (float*)myMalloc(1 * sizeof(float));;
x1329[0] = 1.0f;
float* x1331 = (float*)myMalloc(1 * sizeof(float));;
x1331[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1329,x57,1,x1331, x58, 1, x57,1));
arrayFill_greg<<<28, 512>>>(x58, 0.0f, 32);
float* x1335 = (float*)myMalloc(1 * sizeof(float));;
x1335[0] = 1.0f;
float* x1337 = (float*)myMalloc(1 * sizeof(float));;
x1337[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1335,x31,1,x1337, x32, 1, x31,1));
arrayFill_greg<<<28, 512>>>(x32, 0.0f, 32);
float* x1341 = (float*)myMalloc(1 * sizeof(float));;
x1341[0] = 1.0f;
float* x1343 = (float*)myMalloc(1 * sizeof(float));;
x1343[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1341,x28,1,x1343, x30, 1, x28,1));
arrayFill_greg<<<28, 512>>>(x30, 0.0f, 32);
float* x1347 = (float*)myMalloc(1 * sizeof(float));;
x1347[0] = 1.0f;
float* x1349 = (float*)myMalloc(1 * sizeof(float));;
x1349[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x1347,x200,1,x1349, x202, 1, x200,1));
arrayFill_greg<<<28, 512>>>(x202, 0.0f, 1024);
float* x1353 = (float*)myMalloc(1 * sizeof(float));;
x1353[0] = 1.0f;
float* x1355 = (float*)myMalloc(1 * sizeof(float));;
x1355[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x1353,x203,1,x1355, x204, 1, x203,1));
arrayFill_greg<<<28, 512>>>(x204, 0.0f, 1024);
float* x1359 = (float*)myMalloc(1 * sizeof(float));;
x1359[0] = 1.0f;
float* x1361 = (float*)myMalloc(1 * sizeof(float));;
x1361[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 29,1024,x1359,x217,29,x1361, x219, 29, x217,29));
arrayFill_greg<<<28, 512>>>(x219, 0.0f, 29696);
int32_t x1365 = x303;
int32_t x1367 = x1365 % x1366;
bool x1368 = x1367 == 0;
if (x1368) {
float x1373 = x306;
double x1369 = (double)x1365;
double x1370 = 100.0 * x1369;
double x1372 = x1370 / x1371;
float x1374 = (float)x1365;
float x1375 = x1373 / x1374;
printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x299,x1365,x235,x1372,x1375);
fflush(stdout);
} else {
}
int64_t x1380 = (long)mallocAddr;
int64_t x1381 = x1380 - x295;
memset((void*)x295, 0, x1381);
mallocAddr = (void*)x295;
int64_t x1384 = (long)gpuMallocAddr;
int64_t x1385 = x1384 - x296;
cudaMemset((void*)x296, 0, x1385);
gpuMallocAddr = (void*)x296;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x1392 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x1393 = x1392 / 1000LL;
int64_t x1395 = x1392 / x1394;
printf("Training completed in %ldms (%ld us/images)\n",x1393,x1395);
double x1397 = (double)x1392;
double x1398 = x1397 / 1000000.0;
x294[x299] = x1398;
float x1400 = x306;
float x1402 = x1400 / x1401;
double x1403 = (double)x1402;
x293[x299] = x1403;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x1409 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
sort(x294, x294 + 1);
double x1415 = x294[0];
int64_t x1416 = (long)fopen(x0, "w");
fprintf((FILE *)x1416, "unit: %s\n", "1 epoch");
for(int x1418=0; x1418 < 1; x1418++) {
double x1419 = x293[x1418];
fprintf((FILE *)x1416, "%lf\n", x1419);

}
fprintf((FILE *)x1416, "run time: %lf %lf\n", x291, x1415);
fclose((FILE*)x1416);
// Backend cleanup.
CUBLAS_CALL(cublasDestroy(cublasHandle));
CUDA_CALL(cudaFree(gpuMallocBase));
      
CUDNN_CALL(cudnnDestroy(cudnnHandle));
}
/*****************************************
  End of C Generated Code                  
*******************************************/


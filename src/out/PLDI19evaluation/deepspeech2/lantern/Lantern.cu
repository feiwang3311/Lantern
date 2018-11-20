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
bool x501 = 2 < 0;
int32_t x486 = 2048 / 2;
bool x507 = x486 < 0;
bool x538 = 2 > 0;
bool x543 = x486 > 0;
bool x1010 = 29 < 0;
bool x1041 = 29 > 0;
bool x1063 = 1 < 0;
bool x1099 = 1 > 0;
int32_t x1582 = x230 * 20;
int32_t x235 = x230 * 200;
double x1587 = (double)x235;
int64_t x1610 = (int64_t)x235;
float x1617 = (float)x235;
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
if (x501) {
x487 += 1;
} else {
x488 *= 2;
}
if (x507) {
x487 += 1;
} else {
x488 *= x486;
}
int32_t x513 = x487;
bool x514 = x513 >= 2;
if (x514) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x519 = x513 == 0;
int32_t x478 = x441 * 2048;
int32_t x479 = x451 * x478;
if (x519) {
int32_t x520 = x488;
bool x521 = x520 == x479;
if (x521) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x528 = x451 > 0;
int32_t x532;
if (x528) {
x532 = x451;
} else {
int32_t x529 = x488;
int32_t x530 = x479 / x529;
x532 = x530;
}
bool x533 = x441 > 0;
int32_t x537;
if (x533) {
x537 = x441;
} else {
int32_t x534 = x488;
int32_t x535 = x479 / x534;
x537 = x535;
}
int32_t x542;
if (x538) {
x542 = 2;
} else {
int32_t x539 = x488;
int32_t x540 = x479 / x539;
x542 = x540;
}
int32_t x547;
if (x543) {
x547 = x486;
} else {
int32_t x544 = x488;
int32_t x545 = x479 / x544;
x547 = x545;
}
int32_t x551 = 0;
int32_t x552 = 1;
bool x553 = x532 < 0;
if (x553) {
x551 += 1;
} else {
x552 *= x532;
}
bool x559 = x537 < 0;
if (x559) {
x551 += 1;
} else {
x552 *= x537;
}
bool x565 = x542 < 0;
if (x565) {
x551 += 1;
} else {
x552 *= x542;
}
bool x571 = x547 < 0;
if (x571) {
x551 += 1;
} else {
x552 *= x547;
}
int32_t x577 = x551;
bool x578 = x577 >= 2;
if (x578) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x583 = x577 == 0;
int32_t x548 = x542 * x547;
int32_t x549 = x537 * x548;
int32_t x550 = x532 * x549;
if (x583) {
int32_t x584 = x552;
bool x585 = x584 == x550;
if (x585) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x592 = x532 > 0;
int32_t x596;
if (x592) {
x596 = x532;
} else {
int32_t x593 = x552;
int32_t x594 = x550 / x593;
x596 = x594;
}
bool x597 = x537 > 0;
int32_t x601;
if (x597) {
x601 = x537;
} else {
int32_t x598 = x552;
int32_t x599 = x550 / x598;
x601 = x599;
}
bool x602 = x542 > 0;
int32_t x606;
if (x602) {
x606 = x542;
} else {
int32_t x603 = x552;
int32_t x604 = x550 / x603;
x606 = x604;
}
bool x607 = x547 > 0;
int32_t x611;
if (x607) {
x611 = x547;
} else {
int32_t x608 = x552;
int32_t x609 = x550 / x608;
x611 = x609;
}
int32_t x617 = x596 * x601;
int32_t x618 = x617 * x611;
float* x619 = (float*)myGpuMalloc(x618 * sizeof(float));
float* x620 = (float*)myMalloc(1 * sizeof(float));;
x620[0] = 0.0f;
float* x622 = (float*)myMalloc(1 * sizeof(float));;
x622[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x596, x601, x606, x611));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x596, x601, 1, x611));

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
    x622, x_desc, x481, x620, out_desc, x619));
};
float* x625 = (float*)myGpuMalloc(x618 * sizeof(float));
float* x626 = (float*)NULL;
float* x627 = (float*)NULL;
float* x628 = (float*)NULL;
int32_t x631 = x617 * 2048;
float* x632 = (float*)myGpuMalloc(x631 * sizeof(float));
float* x633 = (float*)NULL;
int32_t x634 = 0;

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
int32_t seqLength = x596;
int32_t batchSize = x601;
int32_t inputSize = x611;

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
x633 = (float*)reserveSpace;
x634 = (int)reserveSize;
CUDNN_CALL(cudnnRNNForwardTraining(
    cudnnHandle, rnn_desc, seqLength, x_descs, x619,
    hx_desc,x626, cx_desc,x627, w_desc, x115, y_descs, x632,
    hy_desc,x628, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
};
float* x636 = (float*)myGpuMalloc(x631 * sizeof(float));
int32_t x637 = 0;
int32_t x638 = 1;
bool x639 = x596 < 0;
if (x639) {
x637 += 1;
} else {
x638 *= x596;
}
bool x645 = x601 < 0;
if (x645) {
x637 += 1;
} else {
x638 *= x601;
}
if (x501) {
x637 += 1;
} else {
x638 *= 2;
}
if (x507) {
x637 += 1;
} else {
x638 *= x486;
}
int32_t x661 = x637;
bool x662 = x661 >= 2;
if (x662) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x667 = x661 == 0;
int32_t x629 = x601 * 2048;
int32_t x630 = x596 * x629;
if (x667) {
int32_t x668 = x638;
bool x669 = x668 == x630;
if (x669) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x676 = x596 > 0;
int32_t x680;
if (x676) {
x680 = x596;
} else {
int32_t x677 = x638;
int32_t x678 = x630 / x677;
x680 = x678;
}
bool x681 = x601 > 0;
int32_t x685;
if (x681) {
x685 = x601;
} else {
int32_t x682 = x638;
int32_t x683 = x630 / x682;
x685 = x683;
}
int32_t x689;
if (x538) {
x689 = 2;
} else {
int32_t x686 = x638;
int32_t x687 = x630 / x686;
x689 = x687;
}
int32_t x693;
if (x543) {
x693 = x486;
} else {
int32_t x690 = x638;
int32_t x691 = x630 / x690;
x693 = x691;
}
int32_t x697 = 0;
int32_t x698 = 1;
bool x699 = x680 < 0;
if (x699) {
x697 += 1;
} else {
x698 *= x680;
}
bool x705 = x685 < 0;
if (x705) {
x697 += 1;
} else {
x698 *= x685;
}
bool x711 = x689 < 0;
if (x711) {
x697 += 1;
} else {
x698 *= x689;
}
bool x717 = x693 < 0;
if (x717) {
x697 += 1;
} else {
x698 *= x693;
}
int32_t x723 = x697;
bool x724 = x723 >= 2;
if (x724) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x729 = x723 == 0;
int32_t x694 = x689 * x693;
int32_t x695 = x685 * x694;
int32_t x696 = x680 * x695;
if (x729) {
int32_t x730 = x698;
bool x731 = x730 == x696;
if (x731) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x738 = x680 > 0;
int32_t x742;
if (x738) {
x742 = x680;
} else {
int32_t x739 = x698;
int32_t x740 = x696 / x739;
x742 = x740;
}
bool x743 = x685 > 0;
int32_t x747;
if (x743) {
x747 = x685;
} else {
int32_t x744 = x698;
int32_t x745 = x696 / x744;
x747 = x745;
}
bool x748 = x689 > 0;
int32_t x752;
if (x748) {
x752 = x689;
} else {
int32_t x749 = x698;
int32_t x750 = x696 / x749;
x752 = x750;
}
bool x753 = x693 > 0;
int32_t x757;
if (x753) {
x757 = x693;
} else {
int32_t x754 = x698;
int32_t x755 = x696 / x754;
x757 = x755;
}
int32_t x763 = x742 * x747;
int32_t x764 = x763 * x757;
float* x765 = (float*)myGpuMalloc(x764 * sizeof(float));
float* x766 = (float*)myMalloc(1 * sizeof(float));;
x766[0] = 0.0f;
float* x768 = (float*)myMalloc(1 * sizeof(float));;
x768[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x742, x747, x752, x757));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x742, x747, 1, x757));

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
    x768, x_desc, x632, x766, out_desc, x765));
};
float* x771 = (float*)myGpuMalloc(x764 * sizeof(float));
float* x772 = (float*)NULL;
float* x773 = (float*)NULL;
float* x774 = (float*)NULL;
int32_t x777 = x763 * 2048;
float* x778 = (float*)myGpuMalloc(x777 * sizeof(float));
float* x779 = (float*)NULL;
int32_t x780 = 0;

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
int32_t seqLength = x742;
int32_t batchSize = x747;
int32_t inputSize = x757;

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
x779 = (float*)reserveSpace;
x780 = (int)reserveSize;
CUDNN_CALL(cudnnRNNForwardTraining(
    cudnnHandle, rnn_desc, seqLength, x_descs, x765,
    hx_desc,x772, cx_desc,x773, w_desc, x158, y_descs, x778,
    hy_desc,x774, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
};
float* x782 = (float*)myGpuMalloc(x777 * sizeof(float));
int32_t x783 = 0;
int32_t x784 = 1;
bool x785 = x742 < 0;
if (x785) {
x783 += 1;
} else {
x784 *= x742;
}
bool x791 = x747 < 0;
if (x791) {
x783 += 1;
} else {
x784 *= x747;
}
if (x501) {
x783 += 1;
} else {
x784 *= 2;
}
if (x507) {
x783 += 1;
} else {
x784 *= x486;
}
int32_t x807 = x783;
bool x808 = x807 >= 2;
if (x808) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x813 = x807 == 0;
int32_t x775 = x747 * 2048;
int32_t x776 = x742 * x775;
if (x813) {
int32_t x814 = x784;
bool x815 = x814 == x776;
if (x815) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x822 = x742 > 0;
int32_t x826;
if (x822) {
x826 = x742;
} else {
int32_t x823 = x784;
int32_t x824 = x776 / x823;
x826 = x824;
}
bool x827 = x747 > 0;
int32_t x831;
if (x827) {
x831 = x747;
} else {
int32_t x828 = x784;
int32_t x829 = x776 / x828;
x831 = x829;
}
int32_t x835;
if (x538) {
x835 = 2;
} else {
int32_t x832 = x784;
int32_t x833 = x776 / x832;
x835 = x833;
}
int32_t x839;
if (x543) {
x839 = x486;
} else {
int32_t x836 = x784;
int32_t x837 = x776 / x836;
x839 = x837;
}
int32_t x843 = 0;
int32_t x844 = 1;
bool x845 = x826 < 0;
if (x845) {
x843 += 1;
} else {
x844 *= x826;
}
bool x851 = x831 < 0;
if (x851) {
x843 += 1;
} else {
x844 *= x831;
}
bool x857 = x835 < 0;
if (x857) {
x843 += 1;
} else {
x844 *= x835;
}
bool x863 = x839 < 0;
if (x863) {
x843 += 1;
} else {
x844 *= x839;
}
int32_t x869 = x843;
bool x870 = x869 >= 2;
if (x870) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x875 = x869 == 0;
int32_t x840 = x835 * x839;
int32_t x841 = x831 * x840;
int32_t x842 = x826 * x841;
if (x875) {
int32_t x876 = x844;
bool x877 = x876 == x842;
if (x877) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x884 = x826 > 0;
int32_t x888;
if (x884) {
x888 = x826;
} else {
int32_t x885 = x844;
int32_t x886 = x842 / x885;
x888 = x886;
}
bool x889 = x831 > 0;
int32_t x893;
if (x889) {
x893 = x831;
} else {
int32_t x890 = x844;
int32_t x891 = x842 / x890;
x893 = x891;
}
bool x894 = x835 > 0;
int32_t x898;
if (x894) {
x898 = x835;
} else {
int32_t x895 = x844;
int32_t x896 = x842 / x895;
x898 = x896;
}
bool x899 = x839 > 0;
int32_t x903;
if (x899) {
x903 = x839;
} else {
int32_t x900 = x844;
int32_t x901 = x842 / x900;
x903 = x901;
}
int32_t x909 = x888 * x893;
int32_t x910 = x909 * x903;
float* x911 = (float*)myGpuMalloc(x910 * sizeof(float));
float* x912 = (float*)myMalloc(1 * sizeof(float));;
x912[0] = 0.0f;
float* x914 = (float*)myMalloc(1 * sizeof(float));;
x914[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x888, x893, x898, x903));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x888, x893, 1, x903));

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
    x914, x_desc, x778, x912, out_desc, x911));
};
float* x917 = (float*)myGpuMalloc(x910 * sizeof(float));
// after RNN layers
// after maybe lookahead
int32_t x920 = 0;
int32_t x921 = 1;
bool x922 = x909 < 0;
if (x922) {
x920 += 1;
} else {
x921 *= x909;
}
bool x928 = x903 < 0;
if (x928) {
x920 += 1;
} else {
x921 *= x903;
}
int32_t x934 = x920;
bool x935 = x934 >= 2;
if (x935) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x940 = x934 == 0;
int32_t x907 = x893 * x903;
int32_t x908 = x888 * x907;
if (x940) {
int32_t x941 = x921;
bool x942 = x941 == x908;
if (x942) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x949 = x909 > 0;
int32_t x953;
if (x949) {
x953 = x909;
} else {
int32_t x950 = x921;
int32_t x951 = x908 / x950;
x953 = x951;
}
bool x954 = x903 > 0;
int32_t x958;
if (x954) {
x958 = x903;
} else {
int32_t x955 = x921;
int32_t x956 = x908 / x955;
x958 = x956;
}
bool x960 = x958 == 1024;
if (x960) {
} else {
assert(false && "BatchNorm1D input should be rank2, with shape 1 same as dimSize, got %d : %d");
}
bool x965 = 1024 == x958;
if (x965) {
} else {
assert(false && "scale should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(953) x Sym(958)");
}
if (x965) {
} else {
assert(false && "bias should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(953) x Sym(958)");
}
if (x965) {
} else {
assert(false && "runningMean should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(953) x Sym(958)");
}
if (x965) {
} else {
assert(false && "runningVar should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(953) x Sym(958)");
}
int32_t x959 = x953 * x958;
float* x979 = (float*)myGpuMalloc(x959 * sizeof(float));
float* x980 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x981 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x982 = (float*)myMalloc(1 * sizeof(float));;
x982[0] = 0.0f;
float* x984 = (float*)myMalloc(1 * sizeof(float));;
x984[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x953, x958, 1, 1));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,
    x984, x982, in_desc, x911, in_desc, x979, sbmv_desc, x200,
    x203, 0.1, x205, x206, 1.0E-5,
    x980, x981));
};
float* x987 = (float*)myGpuMalloc(x959 * sizeof(float));
int32_t x988 = x953 * 29;
float* x989 = (float*)myGpuMalloc(x988 * sizeof(float));
float* x990 = (float*)myMalloc(1 * sizeof(float));;
x990[0] = 0.0f;
float* x992 = (float*)myMalloc(1 * sizeof(float));;
x992[0] = 1.0f;
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 29,x953,1024,x992,x217,29,x979,1024,x990,x989,29));
float* x995 = (float*)myGpuMalloc(x988 * sizeof(float));
int32_t x996 = 0;
int32_t x997 = 1;
bool x998 = x888 < 0;
if (x998) {
x996 += 1;
} else {
x997 *= x888;
}
bool x1004 = x893 < 0;
if (x1004) {
x996 += 1;
} else {
x997 *= x893;
}
if (x1010) {
x996 += 1;
} else {
x997 *= 29;
}
int32_t x1016 = x996;
bool x1017 = x1016 >= 2;
if (x1017) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1022 = x1016 == 0;
if (x1022) {
int32_t x1023 = x997;
bool x1024 = x1023 == x988;
if (x1024) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x1031 = x888 > 0;
int32_t x1035;
if (x1031) {
x1035 = x888;
} else {
int32_t x1032 = x997;
int32_t x1033 = x988 / x1032;
x1035 = x1033;
}
bool x1036 = x893 > 0;
int32_t x1040;
if (x1036) {
x1040 = x893;
} else {
int32_t x1037 = x997;
int32_t x1038 = x988 / x1037;
x1040 = x1038;
}
int32_t x1045;
if (x1041) {
x1045 = 29;
} else {
int32_t x1042 = x997;
int32_t x1043 = x988 / x1042;
x1045 = x1043;
}
int32_t x1049 = 0;
int32_t x1050 = 1;
int32_t x1048 = x1035 * x1040;
bool x1051 = x1048 < 0;
if (x1051) {
x1049 += 1;
} else {
x1050 *= x1048;
}
bool x1057 = x1045 < 0;
if (x1057) {
x1049 += 1;
} else {
x1050 *= x1045;
}
if (x1063) {
x1049 += 1;
} else {
x1050 *= 1;
}
if (x1063) {
x1049 += 1;
} else {
x1050 *= 1;
}
int32_t x1074 = x1049;
bool x1075 = x1074 >= 2;
if (x1075) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1080 = x1074 == 0;
int32_t x1046 = x1040 * x1045;
int32_t x1047 = x1035 * x1046;
if (x1080) {
int32_t x1081 = x1050;
bool x1082 = x1081 == x1047;
if (x1082) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x1089 = x1048 > 0;
int32_t x1093;
if (x1089) {
x1093 = x1048;
} else {
int32_t x1090 = x1050;
int32_t x1091 = x1047 / x1090;
x1093 = x1091;
}
bool x1094 = x1045 > 0;
int32_t x1098;
if (x1094) {
x1098 = x1045;
} else {
int32_t x1095 = x1050;
int32_t x1096 = x1047 / x1095;
x1098 = x1096;
}
int32_t x1103;
if (x1099) {
x1103 = 1;
} else {
int32_t x1100 = x1050;
int32_t x1101 = x1047 / x1100;
x1103 = x1101;
}
int32_t x1107;
if (x1099) {
x1107 = 1;
} else {
int32_t x1104 = x1050;
int32_t x1105 = x1047 / x1104;
x1107 = x1105;
}
float* x1111 = (float*)myMalloc(1 * sizeof(float));;
x1111[0] = 0.0f;
float* x1113 = (float*)myMalloc(1 * sizeof(float));;
x1113[0] = 1.0f;
int32_t x1108 = x1103 * x1107;
int32_t x1109 = x1098 * x1108;
int32_t x1110 = x1093 * x1109;
float* x1115 = (float*)myGpuMalloc(x1110 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x1093, x1098, x1103, x1107));
CUDNN_CALL(cudnnSoftmaxForward(
    cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
    x1113, x_desc, x989, x1111, x_desc, x1115));
};
int32_t x1117 = 0;
int32_t x1118 = 1;
bool x1119 = x1035 < 0;
if (x1119) {
x1117 += 1;
} else {
x1118 *= x1035;
}
bool x1125 = x1040 < 0;
if (x1125) {
x1117 += 1;
} else {
x1118 *= x1040;
}
if (x1057) {
x1117 += 1;
} else {
x1118 *= x1045;
}
int32_t x1136 = x1117;
bool x1137 = x1136 >= 2;
if (x1137) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1142 = x1136 == 0;
if (x1142) {
int32_t x1143 = x1118;
bool x1144 = x1143 == x1110;
if (x1144) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x1151 = x1035 > 0;
int32_t x1155;
if (x1151) {
x1155 = x1035;
} else {
int32_t x1152 = x1118;
int32_t x1153 = x1110 / x1152;
x1155 = x1153;
}
bool x1156 = x1040 > 0;
int32_t x1160;
if (x1156) {
x1160 = x1040;
} else {
int32_t x1157 = x1118;
int32_t x1158 = x1110 / x1157;
x1160 = x1158;
}
int32_t x1164;
if (x1094) {
x1164 = x1045;
} else {
int32_t x1161 = x1118;
int32_t x1162 = x1110 / x1161;
x1164 = x1162;
}
int32_t x1167 = x1155 * x1160;
int32_t x1168 = x1167 * x1164;
float* x1169 = (float*)myGpuMalloc(x1168 * sizeof(float));
// before CTC loss
int* x1171 = (int32_t*)myMalloc(x1160 * sizeof(int32_t));;
float x1175 = (float)x1155;
for(int x1173=0; x1173 < x1160; x1173++) {
float x1174 = x316[x1173];
float x1176 = x1174 * x1175;
int32_t x1177 = (int)x1176;
x1171[x1173] = x1177;

}
bool x1181 = x1160 <= 256;
if (x1181) {
} else {
printf("'cudnnGetCTCLossWorkspaceSize' requires batch size less than 256, got %d\n\n",x1160);
assert(false && "");
}
float* x1187 = (float*)myGpuMalloc(x1160 * sizeof(float));

{
cudnnTensorDescriptor_t probs_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&probs_desc));
int probs_dims[] = {x1155, x1160, x1164};
int probs_strides[] = {probs_dims[1] * probs_dims[2], probs_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    probs_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, probs_dims, probs_strides));

cudnnTensorDescriptor_t grad_desc = probs_desc;

cudnnCTCLossDescriptor_t ctc_desc;
CUDNN_CALL(cudnnCreateCTCLossDescriptor(&ctc_desc));
CUDNN_CALL(cudnnSetCTCLossDescriptor(ctc_desc, CUDNN_DATA_FLOAT));
size_t wsSize;
CUDNN_CALL(cudnnGetCTCLossWorkspaceSize(
    cudnnHandle, probs_desc, grad_desc, x317, x318, x1171,
    CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, &wsSize));
void *ws = myGpuMalloc(wsSize);

CUDNN_CALL(cudnnCTCLoss(
    cudnnHandle, probs_desc, x1115, x317, x318, x1171,
    x1187, grad_desc, x1169, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, ws, wsSize));
};
float* x1189 = (float*)myGpuMalloc(1 * sizeof(float));
float* x1190 = (float*)myMalloc(1 * sizeof(float));;
x1190[0] = 0.0f;
float* x1192 = (float*)myMalloc(1 * sizeof(float));;
x1192[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x1160, 1, 1, 1));

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
    x1192, x_desc, x1187, x1190, out_desc, x1189));
};
// after CTC loss
float* x1196 = (float*)myGpuMalloc(1 * sizeof(float));
// make sure the size of loss is 1
arrayFill_greg<<<28, 512>>>(x1196, 1.0f, 1);
// backend is lantern.TensorDslCudnn$BackendCudnn@c5a15cd
CUDA_CALL(cudaMemcpy(x327, x1189, 1 * sizeof(float), cudaMemcpyDeviceToHost));
int32_t x1201 = 0;
int32_t x1202 = 1;
if (x1051) {
x1201 += 1;
} else {
x1202 *= x1048;
}
if (x1057) {
x1201 += 1;
} else {
x1202 *= x1045;
}
if (x1063) {
x1201 += 1;
} else {
x1202 *= 1;
}
if (x1063) {
x1201 += 1;
} else {
x1202 *= 1;
}
int32_t x1223 = x1201;
bool x1224 = x1223 >= 2;
if (x1224) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1229 = x1223 == 0;
if (x1229) {
int32_t x1230 = x1202;
bool x1231 = x1230 == x1047;
if (x1231) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1241;
if (x1089) {
x1241 = x1048;
} else {
int32_t x1238 = x1202;
int32_t x1239 = x1047 / x1238;
x1241 = x1239;
}
int32_t x1245;
if (x1094) {
x1245 = x1045;
} else {
int32_t x1242 = x1202;
int32_t x1243 = x1047 / x1242;
x1245 = x1243;
}
int32_t x1249;
if (x1099) {
x1249 = 1;
} else {
int32_t x1246 = x1202;
int32_t x1247 = x1047 / x1246;
x1249 = x1247;
}
int32_t x1253;
if (x1099) {
x1253 = 1;
} else {
int32_t x1250 = x1202;
int32_t x1251 = x1047 / x1250;
x1253 = x1251;
}
int32_t x1257 = 0;
int32_t x1258 = 1;
if (x1051) {
x1257 += 1;
} else {
x1258 *= x1048;
}
if (x1057) {
x1257 += 1;
} else {
x1258 *= x1045;
}
if (x1063) {
x1257 += 1;
} else {
x1258 *= 1;
}
if (x1063) {
x1257 += 1;
} else {
x1258 *= 1;
}
int32_t x1279 = x1257;
bool x1280 = x1279 >= 2;
if (x1280) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1285 = x1279 == 0;
if (x1285) {
int32_t x1286 = x1258;
bool x1287 = x1286 == x1047;
if (x1287) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1297;
if (x1089) {
x1297 = x1048;
} else {
int32_t x1294 = x1258;
int32_t x1295 = x1047 / x1294;
x1297 = x1295;
}
int32_t x1301;
if (x1094) {
x1301 = x1045;
} else {
int32_t x1298 = x1258;
int32_t x1299 = x1047 / x1298;
x1301 = x1299;
}
int32_t x1305;
if (x1099) {
x1305 = 1;
} else {
int32_t x1302 = x1258;
int32_t x1303 = x1047 / x1302;
x1305 = x1303;
}
int32_t x1309;
if (x1099) {
x1309 = 1;
} else {
int32_t x1306 = x1258;
int32_t x1307 = x1047 / x1306;
x1309 = x1307;
}
int32_t x1313 = 0;
int32_t x1314 = 1;
bool x1315 = x1167 < 0;
if (x1315) {
x1313 += 1;
} else {
x1314 *= x1167;
}
bool x1321 = x1164 < 0;
if (x1321) {
x1313 += 1;
} else {
x1314 *= x1164;
}
if (x1063) {
x1313 += 1;
} else {
x1314 *= 1;
}
if (x1063) {
x1313 += 1;
} else {
x1314 *= 1;
}
int32_t x1337 = x1313;
bool x1338 = x1337 >= 2;
if (x1338) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1343 = x1337 == 0;
if (x1343) {
int32_t x1344 = x1314;
int32_t x1165 = x1160 * x1164;
int32_t x1166 = x1155 * x1165;
bool x1345 = x1344 == x1166;
if (x1345) {
} else {
assert(false && "must same size!!");
}
} else {
}
bool x1352 = x1167 > 0;
int32_t x1356;
if (x1352) {
x1356 = x1167;
} else {
int32_t x1353 = x1314;
int32_t x1165 = x1160 * x1164;
int32_t x1166 = x1155 * x1165;
int32_t x1354 = x1166 / x1353;
x1356 = x1354;
}
bool x1357 = x1164 > 0;
int32_t x1361;
if (x1357) {
x1361 = x1164;
} else {
int32_t x1358 = x1314;
int32_t x1165 = x1160 * x1164;
int32_t x1166 = x1155 * x1165;
int32_t x1359 = x1166 / x1358;
x1361 = x1359;
}
int32_t x1365;
if (x1099) {
x1365 = 1;
} else {
int32_t x1362 = x1314;
int32_t x1165 = x1160 * x1164;
int32_t x1166 = x1155 * x1165;
int32_t x1363 = x1166 / x1362;
x1365 = x1363;
}
int32_t x1369;
if (x1099) {
x1369 = 1;
} else {
int32_t x1366 = x1314;
int32_t x1165 = x1160 * x1164;
int32_t x1166 = x1155 * x1165;
int32_t x1367 = x1166 / x1366;
x1369 = x1367;
}
int32_t x1373 = 0;
int32_t x1374 = 1;
if (x1315) {
x1373 += 1;
} else {
x1374 *= x1167;
}
if (x1321) {
x1373 += 1;
} else {
x1374 *= x1164;
}
if (x1063) {
x1373 += 1;
} else {
x1374 *= 1;
}
if (x1063) {
x1373 += 1;
} else {
x1374 *= 1;
}
int32_t x1395 = x1373;
bool x1396 = x1395 >= 2;
if (x1396) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x1401 = x1395 == 0;
if (x1401) {
int32_t x1402 = x1374;
int32_t x1165 = x1160 * x1164;
int32_t x1166 = x1155 * x1165;
bool x1403 = x1402 == x1166;
if (x1403) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x1413;
if (x1352) {
x1413 = x1167;
} else {
int32_t x1410 = x1374;
int32_t x1165 = x1160 * x1164;
int32_t x1166 = x1155 * x1165;
int32_t x1411 = x1166 / x1410;
x1413 = x1411;
}
int32_t x1417;
if (x1357) {
x1417 = x1164;
} else {
int32_t x1414 = x1374;
int32_t x1165 = x1160 * x1164;
int32_t x1166 = x1155 * x1165;
int32_t x1415 = x1166 / x1414;
x1417 = x1415;
}
int32_t x1421;
if (x1099) {
x1421 = 1;
} else {
int32_t x1418 = x1374;
int32_t x1165 = x1160 * x1164;
int32_t x1166 = x1155 * x1165;
int32_t x1419 = x1166 / x1418;
x1421 = x1419;
}
int32_t x1425;
if (x1099) {
x1425 = 1;
} else {
int32_t x1422 = x1374;
int32_t x1165 = x1160 * x1164;
int32_t x1166 = x1155 * x1165;
int32_t x1423 = x1166 / x1422;
x1425 = x1423;
}
bool x1429 = x1241 == x1356;
bool x1431;
if (x1429) {
bool x1430 = x1245 == x1361;
x1431 = x1430;
} else {
x1431 = false;
}
bool x1433;
if (x1431) {
bool x1432 = x1249 == x1365;
x1433 = x1432;
} else {
x1433 = false;
}
bool x1435;
if (x1433) {
bool x1434 = x1253 == x1369;
x1435 = x1434;
} else {
x1435 = false;
}
if (x1435) {
} else {
printf("$errorPrefix: tensor shapes are not equal %s, %s\n\n"," x Sym(1241) x Sym(1245) x Sym(1249) x Sym(1253)"," x Sym(1356) x Sym(1361) x Sym(1365) x Sym(1369)");
assert(false && "");
}
bool x1441 = x1297 == x1413;
bool x1443;
if (x1441) {
bool x1442 = x1301 == x1417;
x1443 = x1442;
} else {
x1443 = false;
}
bool x1445;
if (x1443) {
bool x1444 = x1305 == x1421;
x1445 = x1444;
} else {
x1445 = false;
}
bool x1447;
if (x1445) {
bool x1446 = x1309 == x1425;
x1447 = x1446;
} else {
x1447 = false;
}
if (x1447) {
} else {
printf("$errorPrefix: tensor shapes are not equal %s, %s\n\n"," x Sym(1297) x Sym(1301) x Sym(1305) x Sym(1309)"," x Sym(1413) x Sym(1417) x Sym(1421) x Sym(1425)");
assert(false && "");
}
float* x1453 = (float*)myMalloc(1 * sizeof(float));;
x1453[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x1241, x1245, x1249, x1253));
CUDNN_CALL(cudnnSoftmaxBackward(
    cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
    x1453, x_desc, x1115, x_desc, x1169,
    x1453, x_desc, x995));
};
float* x1456 = (float*)myMalloc(1 * sizeof(float));;
x1456[0] = 0.0f;
float* x1458 = (float*)myMalloc(1 * sizeof(float));;
x1458[0] = 1.0f;
// backprop of matrix-matrix-dot
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, x958,x953,29,x1458,x217,29,x995,29,x1458,x987,x958));
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 29,x958,x953,x1458,x995,29,x979,x958,x1458,x219,29));
float* x1463 = (float*)myMalloc(1 * sizeof(float));;
x1463[0] = 0.0f;
float* x1465 = (float*)myMalloc(1 * sizeof(float));;
x1465[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x953, x958, 1, 1));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,
    x1465, x1465, x1465, x1465, in_desc, x911,
    in_desc, x987, in_desc, x917, sbmv_desc, x200,
    x202,x204, 1.0E-5, x980, x981));
};
// backprop for sum on dim op
sum_grad<<<28, 512>>>(x782, x826, x831, x835, x839, x842, x917, x907, x903, 1, 2);
;
float* x1470 = (float*)NULL;
float* x1471 = (float*)NULL;

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
int32_t seqLength = x742;
int32_t batchSize = x747;
int32_t inputSize = x757;

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
    cudnnHandle, rnn_desc, seqLength, y_descs, x778, y_descs, x782,
    dhy_desc, NULL, dcy_desc, NULL, w_desc, x158, hx_desc, x1470,
    cx_desc, x1471, dx_descs, x771, dhx_desc, NULL, dcx_desc, NULL,
    workspace, workspaceSize, x779, x780));
};
float* x1473 = (float*)NULL;

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
int32_t seqLength = x742;
int32_t batchSize = x747;
int32_t inputSize = x757;

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
    cudnnHandle, rnn_desc, seqLength, x_descs, x765, hx_desc, x1473,
    y_descs, x778, workspace, workspaceSize,
    dw_desc, x160, x779, x780));
};
// backprop for sum on dim op
int32_t x761 = x747 * x757;
sum_grad<<<28, 512>>>(x636, x680, x685, x689, x693, x696, x771, x761, x757, 1, 2);
;
float* x1477 = (float*)NULL;
float* x1478 = (float*)NULL;

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
int32_t seqLength = x596;
int32_t batchSize = x601;
int32_t inputSize = x611;

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
    cudnnHandle, rnn_desc, seqLength, y_descs, x632, y_descs, x636,
    dhy_desc, NULL, dcy_desc, NULL, w_desc, x115, hx_desc, x1477,
    cx_desc, x1478, dx_descs, x625, dhx_desc, NULL, dcx_desc, NULL,
    workspace, workspaceSize, x633, x634));
};
float* x1480 = (float*)NULL;

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
int32_t seqLength = x596;
int32_t batchSize = x601;
int32_t inputSize = x611;

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
    cudnnHandle, rnn_desc, seqLength, x_descs, x619, hx_desc, x1480,
    y_descs, x632, workspace, workspaceSize,
    dw_desc, x117, x633, x634));
};
// backprop for sum on dim op
int32_t x615 = x601 * x611;
sum_grad<<<28, 512>>>(x485, x532, x537, x542, x547, x550, x625, x615, x611, 1, 2);
;
float* x1484 = (float*)NULL;
float* x1485 = (float*)NULL;

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
    dhy_desc, NULL, dcy_desc, NULL, w_desc, x71, hx_desc, x1484,
    cx_desc, x1485, dx_descs, x473, dhx_desc, NULL, dcx_desc, NULL,
    workspace, workspaceSize, x482, x483));
};
float* x1487 = (float*)NULL;

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
    cudnnHandle, rnn_desc, seqLength, x_descs, x454, hx_desc, x1487,
    y_descs, x481, workspace, workspaceSize,
    dw_desc, x73, x482, x483));
};
// backprop for permute WrappedArray(2, 0, 1)
int* x1490 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x1490[2] = x455;
x1490[0] = x446;
x1490[1] = 1;
x1490[3] = 1;
float* x1495 = (float*)myMalloc(1 * sizeof(float));;
x1495[0] = 1.0f;
int32_t x1497 = x1490[0];
int32_t x1498 = x1490[1];
int32_t x1499 = x1490[2];
int32_t x1500 = x1490[3];

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
    in_desc, CUDNN_DATA_FLOAT,
    x441, x446, x451, 1,
    x1497, x1498, x1499, x1500));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
    out_desc, CUDNN_DATA_FLOAT,
    x441, x446, x451, 1,
    x452, x451, 1, 1));

CUDNN_CALL(cudnnTransformTensor(
    cudnnHandle, x1495, in_desc, x473, x1495, out_desc, x397));
};
hardTanh_grad<<<28, 512>>>(x389, x397, x397, 0.0, 20.0, x379, true);
float* x1503 = (float*)myMalloc(1 * sizeof(float));;
x1503[0] = 0.0f;
float* x1505 = (float*)myMalloc(1 * sizeof(float));;
x1505[0] = 1.0f;

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
    x1505, x1505, x1505, x1505, in_desc, x382,
    out_desc, x397, in_desc, x388, sbmv_desc, x54,
    x56,x58, 1.0E-5, x390, x391));
};
// conv2D back-propagate
float* x1509 = (float*)myMalloc(1 * sizeof(float));;
x1509[0] = 1.0f;

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
    x1509, filt_desc, x45, grad_out_desc, x388,
    conv_desc, algo, ws_data, ws_size,
    x1509, grad_in_desc, x362));
};
float* x1512 = (float*)myMalloc(1 * sizeof(float));;
x1512[0] = 1.0f;

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
    x1512, in_desc, x354, grad_out_desc, x388,
    conv_desc, algo, ws_data, ws_size,
    x1512, grad_filt_desc, x47));
};
hardTanh_grad<<<28, 512>>>(x354, x362, x362, 0.0, 20.0, x343, true);
float* x1516 = (float*)myMalloc(1 * sizeof(float));;
x1516[0] = 0.0f;
float* x1518 = (float*)myMalloc(1 * sizeof(float));;
x1518[0] = 1.0f;

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
    x1518, x1518, x1518, x1518, in_desc, x347,
    out_desc, x362, in_desc, x353, sbmv_desc, x28,
    x30,x32, 1.0E-5, x355, x356));
};
// conv2D back-propagate
float* x1522 = (float*)myMalloc(1 * sizeof(float));;
x1522[0] = 1.0f;

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
    x1522, in_desc, x321, grad_out_desc, x353,
    conv_desc, algo, ws_data, ws_size,
    x1522, grad_filt_desc, x20));
};
float x1525 = x327[0];
x306 += x1525;
float* x1527 = (float*)myMalloc(1 * sizeof(float));;
x1527[0] = 1.0f;
float* x1529 = (float*)myMalloc(1 * sizeof(float));;
x1529[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 451,32,x1527,x18,451,x1529, x20, 451, x18,451));
arrayFill_greg<<<28, 512>>>(x20, 0.0f, 14432);
float* x1533 = (float*)myMalloc(1 * sizeof(float));;
x1533[0] = 1.0f;
float* x1535 = (float*)myMalloc(1 * sizeof(float));;
x1535[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 7392,32,x1533,x45,7392,x1535, x47, 7392, x45,7392));
arrayFill_greg<<<28, 512>>>(x47, 0.0f, 236544);
float* x1539 = (float*)myMalloc(1 * sizeof(float));;
x1539[0] = 1.0f;
float* x1541 = (float*)myMalloc(1 * sizeof(float));;
x1541[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1539,x54,1,x1541, x56, 1, x54,1));
arrayFill_greg<<<28, 512>>>(x56, 0.0f, 32);
float* x1545 = (float*)myMalloc(1 * sizeof(float));;
x1545[0] = 1.0f;
float* x1547 = (float*)myMalloc(1 * sizeof(float));;
x1547[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1545,x57,1,x1547, x58, 1, x57,1));
arrayFill_greg<<<28, 512>>>(x58, 0.0f, 32);
float* x1551 = (float*)myMalloc(1 * sizeof(float));;
x1551[0] = 1.0f;
float* x1553 = (float*)myMalloc(1 * sizeof(float));;
x1553[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1551,x31,1,x1553, x32, 1, x31,1));
arrayFill_greg<<<28, 512>>>(x32, 0.0f, 32);
float* x1557 = (float*)myMalloc(1 * sizeof(float));;
x1557[0] = 1.0f;
float* x1559 = (float*)myMalloc(1 * sizeof(float));;
x1559[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1557,x28,1,x1559, x30, 1, x28,1));
arrayFill_greg<<<28, 512>>>(x30, 0.0f, 32);
float* x1563 = (float*)myMalloc(1 * sizeof(float));;
x1563[0] = 1.0f;
float* x1565 = (float*)myMalloc(1 * sizeof(float));;
x1565[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x1563,x200,1,x1565, x202, 1, x200,1));
arrayFill_greg<<<28, 512>>>(x202, 0.0f, 1024);
float* x1569 = (float*)myMalloc(1 * sizeof(float));;
x1569[0] = 1.0f;
float* x1571 = (float*)myMalloc(1 * sizeof(float));;
x1571[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x1569,x203,1,x1571, x204, 1, x203,1));
arrayFill_greg<<<28, 512>>>(x204, 0.0f, 1024);
float* x1575 = (float*)myMalloc(1 * sizeof(float));;
x1575[0] = 1.0f;
float* x1577 = (float*)myMalloc(1 * sizeof(float));;
x1577[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 29,1024,x1575,x217,29,x1577, x219, 29, x217,29));
arrayFill_greg<<<28, 512>>>(x219, 0.0f, 29696);
int32_t x1581 = x303;
int32_t x1583 = x1581 % x1582;
bool x1584 = x1583 == 0;
if (x1584) {
float x1589 = x306;
double x1585 = (double)x1581;
double x1586 = 100.0 * x1585;
double x1588 = x1586 / x1587;
float x1590 = (float)x1581;
float x1591 = x1589 / x1590;
printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x299,x1581,x235,x1588,x1591);
fflush(stdout);
} else {
}
int64_t x1596 = (long)mallocAddr;
int64_t x1597 = x1596 - x295;
memset((void*)x295, 0, x1597);
mallocAddr = (void*)x295;
int64_t x1600 = (long)gpuMallocAddr;
int64_t x1601 = x1600 - x296;
cudaMemset((void*)x296, 0, x1601);
gpuMallocAddr = (void*)x296;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x1608 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x1609 = x1608 / 1000LL;
int64_t x1611 = x1608 / x1610;
printf("Training completed in %ldms (%ld us/images)\n",x1609,x1611);
double x1613 = (double)x1608;
double x1614 = x1613 / 1000000.0;
x294[x299] = x1614;
float x1616 = x306;
float x1618 = x1616 / x1617;
double x1619 = (double)x1618;
x293[x299] = x1619;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x1625 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
sort(x294, x294 + 1);
double x1631 = x294[0];
int64_t x1632 = (long)fopen(x0, "w");
fprintf((FILE *)x1632, "unit: %s\n", "1 epoch");
for(int x1634=0; x1634 < 1; x1634++) {
double x1635 = x293[x1634];
fprintf((FILE *)x1632, "%lf\n", x1635);

}
fprintf((FILE *)x1632, "run time: %lf %lf\n", x291, x1631);
fclose((FILE*)x1632);
// Backend cleanup.
CUBLAS_CALL(cublasDestroy(cublasHandle));
CUDA_CALL(cudaFree(gpuMallocBase));
      
CUDNN_CALL(cudnnDestroy(cudnnHandle));
}
/*****************************************
  End of C Generated Code                  
*******************************************/


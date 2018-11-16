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
int32_t x435 = 2048 / 2;
bool x481 = x435 == 1024;
bool x486 = 1024 == x435;
bool x537 = x230 <= 256;
int32_t x436 = 2 * x435;
int32_t x437 = x230 * x436;
int32_t x439 = x230 * x435;
int32_t x686 = x230 * 20;
int32_t x235 = x230 * 200;
double x691 = (double)x235;
int64_t x714 = (int64_t)x235;
float x721 = (float)x235;
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
CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
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
CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
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
int32_t x400 = 32 * x373;
int32_t x401 = x400 * x376;
int32_t x402 = x230 * x401;
float* x403 = (float*)myGpuMalloc(x402 * sizeof(float));
int* x406 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
int32_t x404 = x230 * x400;
x406[2] = x404;
x406[0] = x400;
x406[1] = 1;
x406[3] = 1;
float* x411 = (float*)myMalloc(1 * sizeof(float));;
x411[0] = 1.0f;
float* x413 = (float*)myMalloc(0 * sizeof(float));;
x413[0] = 0.0f;
int32_t x415 = x406[0];
int32_t x416 = x406[1];
int32_t x417 = x406[2];
int32_t x418 = x406[3];

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
    in_desc, CUDNN_DATA_FLOAT,
    x230, x400, x376, 1,
    x401, x376, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
    out_desc, CUDNN_DATA_FLOAT,
    x230, x400, x376, 1,
    x415, x416, x417, x418));

CUDNN_CALL(cudnnTransformTensor(
    cudnnHandle, x411, in_desc, x389, x413, out_desc, x403));
};
int32_t x420 = x376 * x230;
int32_t x421 = x420 * x400;
float* x422 = (float*)myGpuMalloc(x421 * sizeof(float));
// after resize and permute
float* x424 = (float*)NULL;
float* x425 = (float*)NULL;
float* x426 = (float*)NULL;
int32_t x429 = x420 * 2048;
float* x430 = (float*)myGpuMalloc(x429 * sizeof(float));
float* x431 = (float*)NULL;
int32_t x432 = 0;

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
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
int32_t seqLength = x376;
int32_t batchSize = x230;
int32_t inputSize = x400;

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
x431 = (float*)reserveSpace;
x432 = (int)reserveSize;
CUDNN_CALL(cudnnRNNForwardTraining(
    cudnnHandle, rnn_desc, seqLength, x_descs, x403,
    hx_desc,x424, cx_desc,x425, w_desc, x71, y_descs, x430,
    hy_desc,x426, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
};
float* x434 = (float*)myGpuMalloc(x429 * sizeof(float));
int32_t x441 = x420 * x435;
float* x442 = (float*)myGpuMalloc(x441 * sizeof(float));
float* x443 = (float*)myMalloc(1 * sizeof(float));;
x443[0] = 0.0f;
float* x445 = (float*)myMalloc(1 * sizeof(float));;
x445[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x376, x230, 2, x435));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x376, x230, 1, x435));

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
    x445, x_desc, x430, x443, out_desc, x442));
};
float* x448 = (float*)myGpuMalloc(x441 * sizeof(float));
float* x449 = (float*)NULL;
float* x450 = (float*)NULL;
float* x451 = (float*)NULL;
float* x452 = (float*)myGpuMalloc(x429 * sizeof(float));
float* x453 = (float*)NULL;
int32_t x454 = 0;

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
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
int32_t seqLength = x376;
int32_t batchSize = x230;
int32_t inputSize = x435;

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
x453 = (float*)reserveSpace;
x454 = (int)reserveSize;
CUDNN_CALL(cudnnRNNForwardTraining(
    cudnnHandle, rnn_desc, seqLength, x_descs, x442,
    hx_desc,x449, cx_desc,x450, w_desc, x115, y_descs, x452,
    hy_desc,x451, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
};
float* x456 = (float*)myGpuMalloc(x429 * sizeof(float));
float* x457 = (float*)myGpuMalloc(x441 * sizeof(float));
float* x458 = (float*)myMalloc(1 * sizeof(float));;
x458[0] = 0.0f;
float* x460 = (float*)myMalloc(1 * sizeof(float));;
x460[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x376, x230, 2, x435));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x376, x230, 1, x435));

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
    x460, x_desc, x452, x458, out_desc, x457));
};
float* x463 = (float*)myGpuMalloc(x441 * sizeof(float));
float* x464 = (float*)NULL;
float* x465 = (float*)NULL;
float* x466 = (float*)NULL;
float* x467 = (float*)myGpuMalloc(x429 * sizeof(float));
float* x468 = (float*)NULL;
int32_t x469 = 0;

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
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
int32_t seqLength = x376;
int32_t batchSize = x230;
int32_t inputSize = x435;

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
x468 = (float*)reserveSpace;
x469 = (int)reserveSize;
CUDNN_CALL(cudnnRNNForwardTraining(
    cudnnHandle, rnn_desc, seqLength, x_descs, x457,
    hx_desc,x464, cx_desc,x465, w_desc, x158, y_descs, x467,
    hy_desc,x466, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
};
float* x471 = (float*)myGpuMalloc(x429 * sizeof(float));
float* x472 = (float*)myGpuMalloc(x441 * sizeof(float));
float* x473 = (float*)myMalloc(1 * sizeof(float));;
x473[0] = 0.0f;
float* x475 = (float*)myMalloc(1 * sizeof(float));;
x475[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x376, x230, 2, x435));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x376, x230, 1, x435));

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
    x475, x_desc, x467, x473, out_desc, x472));
};
float* x478 = (float*)myGpuMalloc(x441 * sizeof(float));
// after RNN layers
// after maybe lookahead
if (x481) {
} else {
assert(false && "BatchNorm1D input should be rank2, with shape 1 same as dimSize, got %d : %d");
}
if (x486) {
} else {
assert(false && "scale should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(420) x Sym(435)");
}
if (x486) {
} else {
assert(false && "bias should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(420) x Sym(435)");
}
if (x486) {
} else {
assert(false && "runningMean should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(420) x Sym(435)");
}
if (x486) {
} else {
assert(false && "runningVar should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(420) x Sym(435)");
}
float* x500 = (float*)myGpuMalloc(x441 * sizeof(float));
float* x501 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x502 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x503 = (float*)myMalloc(1 * sizeof(float));;
x503[0] = 0.0f;
float* x505 = (float*)myMalloc(1 * sizeof(float));;
x505[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x420, x435, 1, 1));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,
    x505, x503, in_desc, x472, in_desc, x500, sbmv_desc, x200,
    x203, 0.1, x205, x206, 1.0E-5,
    x501, x502));
};
float* x508 = (float*)myGpuMalloc(x441 * sizeof(float));
int32_t x509 = x420 * 29;
float* x510 = (float*)myGpuMalloc(x509 * sizeof(float));
float* x511 = (float*)myMalloc(1 * sizeof(float));;
x511[0] = 0.0f;
float* x513 = (float*)myMalloc(1 * sizeof(float));;
x513[0] = 1.0f;
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 29,x420,1024,x513,x217,29,x500,1024,x511,x510,29));
float* x516 = (float*)myGpuMalloc(x509 * sizeof(float));
float* x519 = (float*)myMalloc(1 * sizeof(float));;
x519[0] = 0.0f;
float* x521 = (float*)myMalloc(1 * sizeof(float));;
x521[0] = 1.0f;
float* x523 = (float*)myGpuMalloc(x509 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x420, 29, 1, 1));
CUDNN_CALL(cudnnSoftmaxForward(
    cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
    x521, x_desc, x510, x519, x_desc, x523));
};
float* x525 = (float*)myGpuMalloc(x509 * sizeof(float));
// before CTC loss
int* x527 = (int32_t*)myMalloc(x230 * sizeof(int32_t));;
float x531 = (float)x376;
for(int x529=0; x529 < x230; x529++) {
float x530 = x316[x529];
float x532 = x530 * x531;
int32_t x533 = (int)x532;
x527[x529] = x533;

}
if (x537) {
} else {
printf("'cudnnGetCTCLossWorkspaceSize' requires batch size less than 256, got %d\n\n",x230);
assert(false && "");
}
float* x543 = (float*)myGpuMalloc(x230 * sizeof(float));

{
cudnnTensorDescriptor_t probs_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&probs_desc));
int probs_dims[] = {x376, x230, 29};
int probs_strides[] = {probs_dims[1] * probs_dims[2], probs_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    probs_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, probs_dims, probs_strides));

cudnnTensorDescriptor_t grad_desc = probs_desc;

cudnnCTCLossDescriptor_t ctc_desc;
CUDNN_CALL(cudnnCreateCTCLossDescriptor(&ctc_desc));
CUDNN_CALL(cudnnSetCTCLossDescriptor(ctc_desc, CUDNN_DATA_FLOAT));
size_t wsSize;
CUDNN_CALL(cudnnGetCTCLossWorkspaceSize(
    cudnnHandle, probs_desc, grad_desc, x317, x318, x527,
    CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, &wsSize));
void *ws = myGpuMalloc(wsSize);

CUDNN_CALL(cudnnCTCLoss(
    cudnnHandle, probs_desc, x523, x317, x318, x527,
    x543, grad_desc, x525, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, ws, wsSize));
};
float* x545 = (float*)myGpuMalloc(1 * sizeof(float));
float* x546 = (float*)myMalloc(1 * sizeof(float));;
x546[0] = 0.0f;
float* x548 = (float*)myMalloc(1 * sizeof(float));;
x548[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x230, 1, 1, 1));

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
    x548, x_desc, x543, x546, out_desc, x545));
};
// after CTC loss
float* x552 = (float*)myGpuMalloc(1 * sizeof(float));
// make sure the size of loss is 1
arrayFill_greg<<<28, 512>>>(x552, 1.0f, 1);
// backend is lantern.TensorDslCudnn$BackendCudnn@3ddce270
CUDA_CALL(cudaMemcpy(x327, x545, 1 * sizeof(float), cudaMemcpyDeviceToHost));
float* x557 = (float*)myMalloc(1 * sizeof(float));;
x557[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x420, 29, 1, 1));
CUDNN_CALL(cudnnSoftmaxBackward(
    cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
    x557, x_desc, x523, x_desc, x525,
    x557, x_desc, x516));
};
float* x560 = (float*)myMalloc(1 * sizeof(float));;
x560[0] = 0.0f;
float* x562 = (float*)myMalloc(1 * sizeof(float));;
x562[0] = 1.0f;
// backprop of matrix-matrix-dot
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, x435,x420,29,x562,x217,29,x516,29,x562,x508,x435));
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 29,x435,x420,x562,x516,29,x500,x435,x562,x219,29));
float* x567 = (float*)myMalloc(1 * sizeof(float));;
x567[0] = 0.0f;
float* x569 = (float*)myMalloc(1 * sizeof(float));;
x569[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x420, x435, 1, 1));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,
    x569, x569, x569, x569, in_desc, x472,
    in_desc, x508, in_desc, x478, sbmv_desc, x200,
    x202,x204, 1.0E-5, x501, x502));
};
// backprop for sum on dim op
int32_t x438 = x376 * x437;
sum_grad<<<28, 512>>>(x471, x376, x230, 2, x435, x438, x478, x439, x435, 1, 2);
;
float* x574 = (float*)NULL;
float* x575 = (float*)NULL;

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
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
int32_t seqLength = x376;
int32_t batchSize = x230;
int32_t inputSize = x435;

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
    cudnnHandle, rnn_desc, seqLength, y_descs, x467, y_descs, x471,
    dhy_desc, NULL, dcy_desc, NULL, w_desc, x158, hx_desc, x574,
    cx_desc, x575, dx_descs, x463, dhx_desc, NULL, dcx_desc, NULL,
    workspace, workspaceSize, x468, x469));
};
float* x577 = (float*)NULL;

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
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
int32_t seqLength = x376;
int32_t batchSize = x230;
int32_t inputSize = x435;

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
    cudnnHandle, rnn_desc, seqLength, x_descs, x457, hx_desc, x577,
    y_descs, x467, workspace, workspaceSize,
    dw_desc, x160, x468, x469));
};
// backprop for sum on dim op
sum_grad<<<28, 512>>>(x456, x376, x230, 2, x435, x438, x463, x439, x435, 1, 2);
;
float* x581 = (float*)NULL;
float* x582 = (float*)NULL;

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
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
int32_t seqLength = x376;
int32_t batchSize = x230;
int32_t inputSize = x435;

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
    cudnnHandle, rnn_desc, seqLength, y_descs, x452, y_descs, x456,
    dhy_desc, NULL, dcy_desc, NULL, w_desc, x115, hx_desc, x581,
    cx_desc, x582, dx_descs, x448, dhx_desc, NULL, dcx_desc, NULL,
    workspace, workspaceSize, x453, x454));
};
float* x584 = (float*)NULL;

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
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
int32_t seqLength = x376;
int32_t batchSize = x230;
int32_t inputSize = x435;

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
    cudnnHandle, rnn_desc, seqLength, x_descs, x442, hx_desc, x584,
    y_descs, x452, workspace, workspaceSize,
    dw_desc, x117, x453, x454));
};
// backprop for sum on dim op
sum_grad<<<28, 512>>>(x434, x376, x230, 2, x435, x438, x448, x439, x435, 1, 2);
;
float* x588 = (float*)NULL;
float* x589 = (float*)NULL;

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
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
int32_t seqLength = x376;
int32_t batchSize = x230;
int32_t inputSize = x400;

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
    cudnnHandle, rnn_desc, seqLength, y_descs, x430, y_descs, x434,
    dhy_desc, NULL, dcy_desc, NULL, w_desc, x71, hx_desc, x588,
    cx_desc, x589, dx_descs, x422, dhx_desc, NULL, dcx_desc, NULL,
    workspace, workspaceSize, x431, x432));
};
float* x591 = (float*)NULL;

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
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
int32_t seqLength = x376;
int32_t batchSize = x230;
int32_t inputSize = x400;

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
    cudnnHandle, rnn_desc, seqLength, x_descs, x403, hx_desc, x591,
    y_descs, x430, workspace, workspaceSize,
    dw_desc, x73, x431, x432));
};
// backprop for permute WrappedArray(2, 0, 1)
int* x594 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x594[2] = x404;
x594[0] = x400;
x594[1] = 1;
x594[3] = 1;
float* x599 = (float*)myMalloc(1 * sizeof(float));;
x599[0] = 1.0f;
int32_t x601 = x594[0];
int32_t x602 = x594[1];
int32_t x603 = x594[2];
int32_t x604 = x594[3];

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
    in_desc, CUDNN_DATA_FLOAT,
    x230, x400, x376, 1,
    x601, x602, x603, x604));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
    out_desc, CUDNN_DATA_FLOAT,
    x230, x400, x376, 1,
    x401, x376, 1, 1));

CUDNN_CALL(cudnnTransformTensor(
    cudnnHandle, x599, in_desc, x422, x599, out_desc, x397));
};
hardTanh_grad<<<28, 512>>>(x389, x397, x397, 0.0, 20.0, x379, true);
float* x607 = (float*)myMalloc(1 * sizeof(float));;
x607[0] = 0.0f;
float* x609 = (float*)myMalloc(1 * sizeof(float));;
x609[0] = 1.0f;

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
    x609, x609, x609, x609, in_desc, x382,
    out_desc, x397, in_desc, x388, sbmv_desc, x54,
    x56,x58, 1.0E-5, x390, x391));
};
// conv2D back-propagate
float* x613 = (float*)myMalloc(1 * sizeof(float));;
x613[0] = 1.0f;

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
CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
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
    x613, filt_desc, x45, grad_out_desc, x388,
    conv_desc, algo, ws_data, ws_size,
    x613, grad_in_desc, x362));
};
float* x616 = (float*)myMalloc(1 * sizeof(float));;
x616[0] = 1.0f;

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
CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
// Algorithm.
cudnnConvolutionBwdFilterAlgo_t algo;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle,
    in_desc, grad_out_desc, conv_desc, grad_filt_desc,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algo));
//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x616, in_desc, x354, grad_out_desc, x388,
    conv_desc, algo, ws_data, ws_size,
    x616, grad_filt_desc, x47));
};
hardTanh_grad<<<28, 512>>>(x354, x362, x362, 0.0, 20.0, x343, true);
float* x620 = (float*)myMalloc(1 * sizeof(float));;
x620[0] = 0.0f;
float* x622 = (float*)myMalloc(1 * sizeof(float));;
x622[0] = 1.0f;

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
    x622, x622, x622, x622, in_desc, x347,
    out_desc, x362, in_desc, x353, sbmv_desc, x28,
    x30,x32, 1.0E-5, x355, x356));
};
// conv2D back-propagate
float* x626 = (float*)myMalloc(1 * sizeof(float));;
x626[0] = 1.0f;

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
CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
// Algorithm.
cudnnConvolutionBwdFilterAlgo_t algo;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle,
    in_desc, grad_out_desc, conv_desc, grad_filt_desc,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algo));
//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x626, in_desc, x321, grad_out_desc, x353,
    conv_desc, algo, ws_data, ws_size,
    x626, grad_filt_desc, x20));
};
float x629 = x327[0];
x306 += x629;
float* x631 = (float*)myMalloc(1 * sizeof(float));;
x631[0] = 1.0f;
float* x633 = (float*)myMalloc(1 * sizeof(float));;
x633[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 451,32,x631,x18,451,x633, x20, 451, x18,451));
arrayFill_greg<<<28, 512>>>(x20, 0.0f, 14432);
float* x637 = (float*)myMalloc(1 * sizeof(float));;
x637[0] = 1.0f;
float* x639 = (float*)myMalloc(1 * sizeof(float));;
x639[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 7392,32,x637,x45,7392,x639, x47, 7392, x45,7392));
arrayFill_greg<<<28, 512>>>(x47, 0.0f, 236544);
float* x643 = (float*)myMalloc(1 * sizeof(float));;
x643[0] = 1.0f;
float* x645 = (float*)myMalloc(1 * sizeof(float));;
x645[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x643,x54,1,x645, x56, 1, x54,1));
arrayFill_greg<<<28, 512>>>(x56, 0.0f, 32);
float* x649 = (float*)myMalloc(1 * sizeof(float));;
x649[0] = 1.0f;
float* x651 = (float*)myMalloc(1 * sizeof(float));;
x651[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x649,x57,1,x651, x58, 1, x57,1));
arrayFill_greg<<<28, 512>>>(x58, 0.0f, 32);
float* x655 = (float*)myMalloc(1 * sizeof(float));;
x655[0] = 1.0f;
float* x657 = (float*)myMalloc(1 * sizeof(float));;
x657[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x655,x31,1,x657, x32, 1, x31,1));
arrayFill_greg<<<28, 512>>>(x32, 0.0f, 32);
float* x661 = (float*)myMalloc(1 * sizeof(float));;
x661[0] = 1.0f;
float* x663 = (float*)myMalloc(1 * sizeof(float));;
x663[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x661,x28,1,x663, x30, 1, x28,1));
arrayFill_greg<<<28, 512>>>(x30, 0.0f, 32);
float* x667 = (float*)myMalloc(1 * sizeof(float));;
x667[0] = 1.0f;
float* x669 = (float*)myMalloc(1 * sizeof(float));;
x669[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x667,x200,1,x669, x202, 1, x200,1));
arrayFill_greg<<<28, 512>>>(x202, 0.0f, 1024);
float* x673 = (float*)myMalloc(1 * sizeof(float));;
x673[0] = 1.0f;
float* x675 = (float*)myMalloc(1 * sizeof(float));;
x675[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x673,x203,1,x675, x204, 1, x203,1));
arrayFill_greg<<<28, 512>>>(x204, 0.0f, 1024);
float* x679 = (float*)myMalloc(1 * sizeof(float));;
x679[0] = 1.0f;
float* x681 = (float*)myMalloc(1 * sizeof(float));;
x681[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 29,1024,x679,x217,29,x681, x219, 29, x217,29));
arrayFill_greg<<<28, 512>>>(x219, 0.0f, 29696);
int32_t x685 = x303;
int32_t x687 = x685 % x686;
bool x688 = x687 == 0;
if (x688) {
float x693 = x306;
double x689 = (double)x685;
double x690 = 100.0 * x689;
double x692 = x690 / x691;
float x694 = (float)x685;
float x695 = x693 / x694;
printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x299,x685,x235,x692,x695);
fflush(stdout);
} else {
}
int64_t x700 = (long)mallocAddr;
int64_t x701 = x700 - x295;
memset((void*)x295, 0, x701);
mallocAddr = (void*)x295;
int64_t x704 = (long)gpuMallocAddr;
int64_t x705 = x704 - x296;
cudaMemset((void*)x296, 0, x705);
gpuMallocAddr = (void*)x296;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x712 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x713 = x712 / 1000LL;
int64_t x715 = x712 / x714;
printf("Training completed in %ldms (%ld us/images)\n",x713,x715);
double x717 = (double)x712;
double x718 = x717 / 1000000.0;
x294[x299] = x718;
float x720 = x306;
float x722 = x720 / x721;
double x723 = (double)x722;
x293[x299] = x723;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x729 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
sort(x294, x294 + 1);
double x735 = x294[0];
int64_t x736 = (long)fopen(x0, "w");
fprintf((FILE *)x736, "unit: %s\n", "1 epoch");
for(int x738=0; x738 < 1; x738++) {
double x739 = x293[x738];
fprintf((FILE *)x736, "%lf\n", x739);

}
fprintf((FILE *)x736, "run time: %lf %lf\n", x291, x735);
fclose((FILE*)x736);
// Backend cleanup.
CUBLAS_CALL(cublasDestroy(cublasHandle));
CUDA_CALL(cudaFree(gpuMallocBase));
      
CUDNN_CALL(cudnnDestroy(cudnnHandle));
}
/*****************************************
  End of C Generated Code                  
*******************************************/


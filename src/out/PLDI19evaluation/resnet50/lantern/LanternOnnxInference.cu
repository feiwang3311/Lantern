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
// Tensor 'toGPU' invocation.
float* x276 = (float*)myGpuMalloc(262144 * sizeof(float));
int32_t x5 = open("/u/data/u99/wang603/TiarkMlEnv/Lantern/src/out/PLDI19evaluation/resnet50/resnet50.onnx.bin",0);
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
bool x1136 = 34 >= 3;
bool x1137;
if (x1136) {
x1137 = x1136;
} else {
x1137 = false;
}
int32_t x1142 = 31 / 1;
int32_t x1143 = x1142 + 1;
int32_t x1147 = 4096 * x1143;
int32_t x1148 = x1147 * x1143;
int32_t x1144 = x1143 * x1143;
int32_t x1145 = 64 * x1144;
int32_t x1146 = 64 * x1145;
int32_t x1171 = x1143 - 2;
int32_t x1172 = x1171 / 2;
int32_t x1173 = x1172 + 1;
int32_t x1177 = 4096 * x1173;
int32_t x1178 = x1177 * x1173;
bool x1181 = x1173 >= 1;
bool x1182;
if (x1181) {
x1182 = x1181;
} else {
x1182 = false;
}
int32_t x1187 = x1172 / 1;
int32_t x1188 = x1187 + 1;
int32_t x1192 = 4096 * x1188;
int32_t x1193 = x1192 * x1188;
int32_t x1189 = x1188 * x1188;
int32_t x1190 = 64 * x1189;
int32_t x1191 = 64 * x1190;
int32_t x1212 = x1188 + 2;
bool x1213 = x1212 >= 3;
bool x1214;
if (x1213) {
x1214 = x1213;
} else {
x1214 = false;
}
int32_t x1219 = x1212 - 3;
int32_t x1220 = x1219 / 1;
int32_t x1221 = x1220 + 1;
int32_t x1225 = 4096 * x1221;
int32_t x1226 = x1225 * x1221;
int32_t x1222 = x1221 * x1221;
int32_t x1223 = 64 * x1222;
int32_t x1224 = 64 * x1223;
bool x1245 = x1221 >= 1;
bool x1246;
if (x1245) {
x1246 = x1245;
} else {
x1246 = false;
}
int32_t x1251 = x1220 / 1;
int32_t x1252 = x1251 + 1;
int32_t x1256 = 16384 * x1252;
int32_t x1257 = x1256 * x1252;
int32_t x1253 = x1252 * x1252;
int32_t x1254 = 256 * x1253;
int32_t x1255 = 64 * x1254;
int32_t x1275 = 16384 * x1188;
int32_t x1276 = x1275 * x1188;
int32_t x1273 = 256 * x1189;
int32_t x1274 = 64 * x1273;
bool x1289 = x1188 == 1;
bool x1290 = x1188 == x1252;
bool x1291 = x1289 || x1290;
bool x1292;
if (x1291) {
x1292 = x1291;
} else {
x1292 = false;
}
bool x1308 = x1252 >= 1;
bool x1309;
if (x1308) {
x1309 = x1308;
} else {
x1309 = false;
}
int32_t x1314 = x1251 / 1;
int32_t x1315 = x1314 + 1;
int32_t x1319 = 4096 * x1315;
int32_t x1320 = x1319 * x1315;
int32_t x1316 = x1315 * x1315;
int32_t x1317 = 64 * x1316;
int32_t x1318 = 64 * x1317;
int32_t x1339 = x1315 + 2;
bool x1340 = x1339 >= 3;
bool x1341;
if (x1340) {
x1341 = x1340;
} else {
x1341 = false;
}
int32_t x1346 = x1339 - 3;
int32_t x1347 = x1346 / 1;
int32_t x1348 = x1347 + 1;
int32_t x1352 = 4096 * x1348;
int32_t x1353 = x1352 * x1348;
int32_t x1349 = x1348 * x1348;
int32_t x1350 = 64 * x1349;
int32_t x1351 = 64 * x1350;
bool x1372 = x1348 >= 1;
bool x1373;
if (x1372) {
x1373 = x1372;
} else {
x1373 = false;
}
int32_t x1378 = x1347 / 1;
int32_t x1379 = x1378 + 1;
int32_t x1383 = 16384 * x1379;
int32_t x1384 = x1383 * x1379;
int32_t x1380 = x1379 * x1379;
int32_t x1381 = 256 * x1380;
int32_t x1382 = 64 * x1381;
bool x1397 = x1252 == 1;
bool x1398 = x1252 == x1379;
bool x1399 = x1397 || x1398;
bool x1400;
if (x1399) {
x1400 = x1399;
} else {
x1400 = false;
}
bool x1416 = x1379 >= 1;
bool x1417;
if (x1416) {
x1417 = x1416;
} else {
x1417 = false;
}
int32_t x1422 = x1378 / 1;
int32_t x1423 = x1422 + 1;
int32_t x1427 = 4096 * x1423;
int32_t x1428 = x1427 * x1423;
int32_t x1424 = x1423 * x1423;
int32_t x1425 = 64 * x1424;
int32_t x1426 = 64 * x1425;
int32_t x1447 = x1423 + 2;
bool x1448 = x1447 >= 3;
bool x1449;
if (x1448) {
x1449 = x1448;
} else {
x1449 = false;
}
int32_t x1454 = x1447 - 3;
int32_t x1455 = x1454 / 1;
int32_t x1456 = x1455 + 1;
int32_t x1460 = 4096 * x1456;
int32_t x1461 = x1460 * x1456;
int32_t x1457 = x1456 * x1456;
int32_t x1458 = 64 * x1457;
int32_t x1459 = 64 * x1458;
bool x1480 = x1456 >= 1;
bool x1481;
if (x1480) {
x1481 = x1480;
} else {
x1481 = false;
}
int32_t x1486 = x1455 / 1;
int32_t x1487 = x1486 + 1;
int32_t x1491 = 16384 * x1487;
int32_t x1492 = x1491 * x1487;
int32_t x1488 = x1487 * x1487;
int32_t x1489 = 256 * x1488;
int32_t x1490 = 64 * x1489;
bool x1505 = x1379 == 1;
bool x1506 = x1379 == x1487;
bool x1507 = x1505 || x1506;
bool x1508;
if (x1507) {
x1508 = x1507;
} else {
x1508 = false;
}
bool x1524 = x1487 >= 1;
bool x1525;
if (x1524) {
x1525 = x1524;
} else {
x1525 = false;
}
int32_t x1530 = x1486 / 1;
int32_t x1531 = x1530 + 1;
int32_t x1535 = 8192 * x1531;
int32_t x1536 = x1535 * x1531;
int32_t x1532 = x1531 * x1531;
int32_t x1533 = 128 * x1532;
int32_t x1534 = 64 * x1533;
int32_t x1555 = x1531 + 2;
bool x1556 = x1555 >= 3;
bool x1557;
if (x1556) {
x1557 = x1556;
} else {
x1557 = false;
}
int32_t x1562 = x1555 - 3;
int32_t x1563 = x1562 / 2;
int32_t x1564 = x1563 + 1;
int32_t x1568 = 8192 * x1564;
int32_t x1569 = x1568 * x1564;
int32_t x1565 = x1564 * x1564;
int32_t x1566 = 128 * x1565;
int32_t x1567 = 64 * x1566;
bool x1588 = x1564 >= 1;
bool x1589;
if (x1588) {
x1589 = x1588;
} else {
x1589 = false;
}
int32_t x1594 = x1563 / 1;
int32_t x1595 = x1594 + 1;
int32_t x1599 = 32768 * x1595;
int32_t x1600 = x1599 * x1595;
int32_t x1596 = x1595 * x1595;
int32_t x1597 = 512 * x1596;
int32_t x1598 = 64 * x1597;
int32_t x1616 = x1486 / 2;
int32_t x1617 = x1616 + 1;
int32_t x1621 = 32768 * x1617;
int32_t x1622 = x1621 * x1617;
int32_t x1618 = x1617 * x1617;
int32_t x1619 = 512 * x1618;
int32_t x1620 = 64 * x1619;
bool x1635 = x1617 == 1;
bool x1636 = x1617 == x1595;
bool x1637 = x1635 || x1636;
bool x1638;
if (x1637) {
x1638 = x1637;
} else {
x1638 = false;
}
bool x1654 = x1595 >= 1;
bool x1655;
if (x1654) {
x1655 = x1654;
} else {
x1655 = false;
}
int32_t x1660 = x1594 / 1;
int32_t x1661 = x1660 + 1;
int32_t x1665 = 8192 * x1661;
int32_t x1666 = x1665 * x1661;
int32_t x1662 = x1661 * x1661;
int32_t x1663 = 128 * x1662;
int32_t x1664 = 64 * x1663;
int32_t x1685 = x1661 + 2;
bool x1686 = x1685 >= 3;
bool x1687;
if (x1686) {
x1687 = x1686;
} else {
x1687 = false;
}
int32_t x1692 = x1685 - 3;
int32_t x1693 = x1692 / 1;
int32_t x1694 = x1693 + 1;
int32_t x1698 = 8192 * x1694;
int32_t x1699 = x1698 * x1694;
int32_t x1695 = x1694 * x1694;
int32_t x1696 = 128 * x1695;
int32_t x1697 = 64 * x1696;
bool x1718 = x1694 >= 1;
bool x1719;
if (x1718) {
x1719 = x1718;
} else {
x1719 = false;
}
int32_t x1724 = x1693 / 1;
int32_t x1725 = x1724 + 1;
int32_t x1729 = 32768 * x1725;
int32_t x1730 = x1729 * x1725;
int32_t x1726 = x1725 * x1725;
int32_t x1727 = 512 * x1726;
int32_t x1728 = 64 * x1727;
bool x1743 = x1595 == 1;
bool x1744 = x1595 == x1725;
bool x1745 = x1743 || x1744;
bool x1746;
if (x1745) {
x1746 = x1745;
} else {
x1746 = false;
}
bool x1762 = x1725 >= 1;
bool x1763;
if (x1762) {
x1763 = x1762;
} else {
x1763 = false;
}
int32_t x1768 = x1724 / 1;
int32_t x1769 = x1768 + 1;
int32_t x1773 = 8192 * x1769;
int32_t x1774 = x1773 * x1769;
int32_t x1770 = x1769 * x1769;
int32_t x1771 = 128 * x1770;
int32_t x1772 = 64 * x1771;
int32_t x1793 = x1769 + 2;
bool x1794 = x1793 >= 3;
bool x1795;
if (x1794) {
x1795 = x1794;
} else {
x1795 = false;
}
int32_t x1800 = x1793 - 3;
int32_t x1801 = x1800 / 1;
int32_t x1802 = x1801 + 1;
int32_t x1806 = 8192 * x1802;
int32_t x1807 = x1806 * x1802;
int32_t x1803 = x1802 * x1802;
int32_t x1804 = 128 * x1803;
int32_t x1805 = 64 * x1804;
bool x1826 = x1802 >= 1;
bool x1827;
if (x1826) {
x1827 = x1826;
} else {
x1827 = false;
}
int32_t x1832 = x1801 / 1;
int32_t x1833 = x1832 + 1;
int32_t x1837 = 32768 * x1833;
int32_t x1838 = x1837 * x1833;
int32_t x1834 = x1833 * x1833;
int32_t x1835 = 512 * x1834;
int32_t x1836 = 64 * x1835;
bool x1851 = x1725 == 1;
bool x1852 = x1725 == x1833;
bool x1853 = x1851 || x1852;
bool x1854;
if (x1853) {
x1854 = x1853;
} else {
x1854 = false;
}
bool x1870 = x1833 >= 1;
bool x1871;
if (x1870) {
x1871 = x1870;
} else {
x1871 = false;
}
int32_t x1876 = x1832 / 1;
int32_t x1877 = x1876 + 1;
int32_t x1881 = 8192 * x1877;
int32_t x1882 = x1881 * x1877;
int32_t x1878 = x1877 * x1877;
int32_t x1879 = 128 * x1878;
int32_t x1880 = 64 * x1879;
int32_t x1901 = x1877 + 2;
bool x1902 = x1901 >= 3;
bool x1903;
if (x1902) {
x1903 = x1902;
} else {
x1903 = false;
}
int32_t x1908 = x1901 - 3;
int32_t x1909 = x1908 / 1;
int32_t x1910 = x1909 + 1;
int32_t x1914 = 8192 * x1910;
int32_t x1915 = x1914 * x1910;
int32_t x1911 = x1910 * x1910;
int32_t x1912 = 128 * x1911;
int32_t x1913 = 64 * x1912;
bool x1934 = x1910 >= 1;
bool x1935;
if (x1934) {
x1935 = x1934;
} else {
x1935 = false;
}
int32_t x1940 = x1909 / 1;
int32_t x1941 = x1940 + 1;
int32_t x1945 = 32768 * x1941;
int32_t x1946 = x1945 * x1941;
int32_t x1942 = x1941 * x1941;
int32_t x1943 = 512 * x1942;
int32_t x1944 = 64 * x1943;
bool x1959 = x1833 == 1;
bool x1960 = x1833 == x1941;
bool x1961 = x1959 || x1960;
bool x1962;
if (x1961) {
x1962 = x1961;
} else {
x1962 = false;
}
bool x1978 = x1941 >= 1;
bool x1979;
if (x1978) {
x1979 = x1978;
} else {
x1979 = false;
}
int32_t x1984 = x1940 / 1;
int32_t x1985 = x1984 + 1;
int32_t x1989 = 16384 * x1985;
int32_t x1990 = x1989 * x1985;
int32_t x1986 = x1985 * x1985;
int32_t x1987 = 256 * x1986;
int32_t x1988 = 64 * x1987;
int32_t x2009 = x1985 + 2;
bool x2010 = x2009 >= 3;
bool x2011;
if (x2010) {
x2011 = x2010;
} else {
x2011 = false;
}
int32_t x2016 = x2009 - 3;
int32_t x2017 = x2016 / 2;
int32_t x2018 = x2017 + 1;
int32_t x2022 = 16384 * x2018;
int32_t x2023 = x2022 * x2018;
int32_t x2019 = x2018 * x2018;
int32_t x2020 = 256 * x2019;
int32_t x2021 = 64 * x2020;
bool x2042 = x2018 >= 1;
bool x2043;
if (x2042) {
x2043 = x2042;
} else {
x2043 = false;
}
int32_t x2048 = x2017 / 1;
int32_t x2049 = x2048 + 1;
int32_t x2053 = 65536 * x2049;
int32_t x2054 = x2053 * x2049;
int32_t x2050 = x2049 * x2049;
int32_t x2051 = 1024 * x2050;
int32_t x2052 = 64 * x2051;
int32_t x2070 = x1940 / 2;
int32_t x2071 = x2070 + 1;
int32_t x2075 = 65536 * x2071;
int32_t x2076 = x2075 * x2071;
int32_t x2072 = x2071 * x2071;
int32_t x2073 = 1024 * x2072;
int32_t x2074 = 64 * x2073;
bool x2089 = x2071 == 1;
bool x2090 = x2071 == x2049;
bool x2091 = x2089 || x2090;
bool x2092;
if (x2091) {
x2092 = x2091;
} else {
x2092 = false;
}
bool x2108 = x2049 >= 1;
bool x2109;
if (x2108) {
x2109 = x2108;
} else {
x2109 = false;
}
int32_t x2114 = x2048 / 1;
int32_t x2115 = x2114 + 1;
int32_t x2119 = 16384 * x2115;
int32_t x2120 = x2119 * x2115;
int32_t x2116 = x2115 * x2115;
int32_t x2117 = 256 * x2116;
int32_t x2118 = 64 * x2117;
int32_t x2139 = x2115 + 2;
bool x2140 = x2139 >= 3;
bool x2141;
if (x2140) {
x2141 = x2140;
} else {
x2141 = false;
}
int32_t x2146 = x2139 - 3;
int32_t x2147 = x2146 / 1;
int32_t x2148 = x2147 + 1;
int32_t x2152 = 16384 * x2148;
int32_t x2153 = x2152 * x2148;
int32_t x2149 = x2148 * x2148;
int32_t x2150 = 256 * x2149;
int32_t x2151 = 64 * x2150;
bool x2172 = x2148 >= 1;
bool x2173;
if (x2172) {
x2173 = x2172;
} else {
x2173 = false;
}
int32_t x2178 = x2147 / 1;
int32_t x2179 = x2178 + 1;
int32_t x2183 = 65536 * x2179;
int32_t x2184 = x2183 * x2179;
int32_t x2180 = x2179 * x2179;
int32_t x2181 = 1024 * x2180;
int32_t x2182 = 64 * x2181;
bool x2197 = x2049 == 1;
bool x2198 = x2049 == x2179;
bool x2199 = x2197 || x2198;
bool x2200;
if (x2199) {
x2200 = x2199;
} else {
x2200 = false;
}
bool x2216 = x2179 >= 1;
bool x2217;
if (x2216) {
x2217 = x2216;
} else {
x2217 = false;
}
int32_t x2222 = x2178 / 1;
int32_t x2223 = x2222 + 1;
int32_t x2227 = 16384 * x2223;
int32_t x2228 = x2227 * x2223;
int32_t x2224 = x2223 * x2223;
int32_t x2225 = 256 * x2224;
int32_t x2226 = 64 * x2225;
int32_t x2247 = x2223 + 2;
bool x2248 = x2247 >= 3;
bool x2249;
if (x2248) {
x2249 = x2248;
} else {
x2249 = false;
}
int32_t x2254 = x2247 - 3;
int32_t x2255 = x2254 / 1;
int32_t x2256 = x2255 + 1;
int32_t x2260 = 16384 * x2256;
int32_t x2261 = x2260 * x2256;
int32_t x2257 = x2256 * x2256;
int32_t x2258 = 256 * x2257;
int32_t x2259 = 64 * x2258;
bool x2280 = x2256 >= 1;
bool x2281;
if (x2280) {
x2281 = x2280;
} else {
x2281 = false;
}
int32_t x2286 = x2255 / 1;
int32_t x2287 = x2286 + 1;
int32_t x2291 = 65536 * x2287;
int32_t x2292 = x2291 * x2287;
int32_t x2288 = x2287 * x2287;
int32_t x2289 = 1024 * x2288;
int32_t x2290 = 64 * x2289;
bool x2305 = x2179 == 1;
bool x2306 = x2179 == x2287;
bool x2307 = x2305 || x2306;
bool x2308;
if (x2307) {
x2308 = x2307;
} else {
x2308 = false;
}
bool x2324 = x2287 >= 1;
bool x2325;
if (x2324) {
x2325 = x2324;
} else {
x2325 = false;
}
int32_t x2330 = x2286 / 1;
int32_t x2331 = x2330 + 1;
int32_t x2335 = 16384 * x2331;
int32_t x2336 = x2335 * x2331;
int32_t x2332 = x2331 * x2331;
int32_t x2333 = 256 * x2332;
int32_t x2334 = 64 * x2333;
int32_t x2355 = x2331 + 2;
bool x2356 = x2355 >= 3;
bool x2357;
if (x2356) {
x2357 = x2356;
} else {
x2357 = false;
}
int32_t x2362 = x2355 - 3;
int32_t x2363 = x2362 / 1;
int32_t x2364 = x2363 + 1;
int32_t x2368 = 16384 * x2364;
int32_t x2369 = x2368 * x2364;
int32_t x2365 = x2364 * x2364;
int32_t x2366 = 256 * x2365;
int32_t x2367 = 64 * x2366;
bool x2388 = x2364 >= 1;
bool x2389;
if (x2388) {
x2389 = x2388;
} else {
x2389 = false;
}
int32_t x2394 = x2363 / 1;
int32_t x2395 = x2394 + 1;
int32_t x2399 = 65536 * x2395;
int32_t x2400 = x2399 * x2395;
int32_t x2396 = x2395 * x2395;
int32_t x2397 = 1024 * x2396;
int32_t x2398 = 64 * x2397;
bool x2413 = x2287 == 1;
bool x2414 = x2287 == x2395;
bool x2415 = x2413 || x2414;
bool x2416;
if (x2415) {
x2416 = x2415;
} else {
x2416 = false;
}
bool x2432 = x2395 >= 1;
bool x2433;
if (x2432) {
x2433 = x2432;
} else {
x2433 = false;
}
int32_t x2438 = x2394 / 1;
int32_t x2439 = x2438 + 1;
int32_t x2443 = 16384 * x2439;
int32_t x2444 = x2443 * x2439;
int32_t x2440 = x2439 * x2439;
int32_t x2441 = 256 * x2440;
int32_t x2442 = 64 * x2441;
int32_t x2463 = x2439 + 2;
bool x2464 = x2463 >= 3;
bool x2465;
if (x2464) {
x2465 = x2464;
} else {
x2465 = false;
}
int32_t x2470 = x2463 - 3;
int32_t x2471 = x2470 / 1;
int32_t x2472 = x2471 + 1;
int32_t x2476 = 16384 * x2472;
int32_t x2477 = x2476 * x2472;
int32_t x2473 = x2472 * x2472;
int32_t x2474 = 256 * x2473;
int32_t x2475 = 64 * x2474;
bool x2496 = x2472 >= 1;
bool x2497;
if (x2496) {
x2497 = x2496;
} else {
x2497 = false;
}
int32_t x2502 = x2471 / 1;
int32_t x2503 = x2502 + 1;
int32_t x2507 = 65536 * x2503;
int32_t x2508 = x2507 * x2503;
int32_t x2504 = x2503 * x2503;
int32_t x2505 = 1024 * x2504;
int32_t x2506 = 64 * x2505;
bool x2521 = x2395 == 1;
bool x2522 = x2395 == x2503;
bool x2523 = x2521 || x2522;
bool x2524;
if (x2523) {
x2524 = x2523;
} else {
x2524 = false;
}
bool x2540 = x2503 >= 1;
bool x2541;
if (x2540) {
x2541 = x2540;
} else {
x2541 = false;
}
int32_t x2546 = x2502 / 1;
int32_t x2547 = x2546 + 1;
int32_t x2551 = 16384 * x2547;
int32_t x2552 = x2551 * x2547;
int32_t x2548 = x2547 * x2547;
int32_t x2549 = 256 * x2548;
int32_t x2550 = 64 * x2549;
int32_t x2571 = x2547 + 2;
bool x2572 = x2571 >= 3;
bool x2573;
if (x2572) {
x2573 = x2572;
} else {
x2573 = false;
}
int32_t x2578 = x2571 - 3;
int32_t x2579 = x2578 / 1;
int32_t x2580 = x2579 + 1;
int32_t x2584 = 16384 * x2580;
int32_t x2585 = x2584 * x2580;
int32_t x2581 = x2580 * x2580;
int32_t x2582 = 256 * x2581;
int32_t x2583 = 64 * x2582;
bool x2604 = x2580 >= 1;
bool x2605;
if (x2604) {
x2605 = x2604;
} else {
x2605 = false;
}
int32_t x2610 = x2579 / 1;
int32_t x2611 = x2610 + 1;
int32_t x2615 = 65536 * x2611;
int32_t x2616 = x2615 * x2611;
int32_t x2612 = x2611 * x2611;
int32_t x2613 = 1024 * x2612;
int32_t x2614 = 64 * x2613;
bool x2629 = x2503 == 1;
bool x2630 = x2503 == x2611;
bool x2631 = x2629 || x2630;
bool x2632;
if (x2631) {
x2632 = x2631;
} else {
x2632 = false;
}
bool x2648 = x2611 >= 1;
bool x2649;
if (x2648) {
x2649 = x2648;
} else {
x2649 = false;
}
int32_t x2654 = x2610 / 1;
int32_t x2655 = x2654 + 1;
int32_t x2659 = 32768 * x2655;
int32_t x2660 = x2659 * x2655;
int32_t x2656 = x2655 * x2655;
int32_t x2657 = 512 * x2656;
int32_t x2658 = 64 * x2657;
int32_t x2679 = x2655 + 2;
bool x2680 = x2679 >= 3;
bool x2681;
if (x2680) {
x2681 = x2680;
} else {
x2681 = false;
}
int32_t x2686 = x2679 - 3;
int32_t x2687 = x2686 / 2;
int32_t x2688 = x2687 + 1;
int32_t x2692 = 32768 * x2688;
int32_t x2693 = x2692 * x2688;
int32_t x2689 = x2688 * x2688;
int32_t x2690 = 512 * x2689;
int32_t x2691 = 64 * x2690;
bool x2712 = x2688 >= 1;
bool x2713;
if (x2712) {
x2713 = x2712;
} else {
x2713 = false;
}
int32_t x2718 = x2687 / 1;
int32_t x2719 = x2718 + 1;
int32_t x2723 = 131072 * x2719;
int32_t x2724 = x2723 * x2719;
int32_t x2720 = x2719 * x2719;
int32_t x2721 = 2048 * x2720;
int32_t x2722 = 64 * x2721;
int32_t x2740 = x2610 / 2;
int32_t x2741 = x2740 + 1;
int32_t x2745 = 131072 * x2741;
int32_t x2746 = x2745 * x2741;
int32_t x2742 = x2741 * x2741;
int32_t x2743 = 2048 * x2742;
int32_t x2744 = 64 * x2743;
bool x2759 = x2741 == 1;
bool x2760 = x2741 == x2719;
bool x2761 = x2759 || x2760;
bool x2762;
if (x2761) {
x2762 = x2761;
} else {
x2762 = false;
}
bool x2778 = x2719 >= 1;
bool x2779;
if (x2778) {
x2779 = x2778;
} else {
x2779 = false;
}
int32_t x2784 = x2718 / 1;
int32_t x2785 = x2784 + 1;
int32_t x2789 = 32768 * x2785;
int32_t x2790 = x2789 * x2785;
int32_t x2786 = x2785 * x2785;
int32_t x2787 = 512 * x2786;
int32_t x2788 = 64 * x2787;
int32_t x2809 = x2785 + 2;
bool x2810 = x2809 >= 3;
bool x2811;
if (x2810) {
x2811 = x2810;
} else {
x2811 = false;
}
int32_t x2816 = x2809 - 3;
int32_t x2817 = x2816 / 1;
int32_t x2818 = x2817 + 1;
int32_t x2822 = 32768 * x2818;
int32_t x2823 = x2822 * x2818;
int32_t x2819 = x2818 * x2818;
int32_t x2820 = 512 * x2819;
int32_t x2821 = 64 * x2820;
bool x2842 = x2818 >= 1;
bool x2843;
if (x2842) {
x2843 = x2842;
} else {
x2843 = false;
}
int32_t x2848 = x2817 / 1;
int32_t x2849 = x2848 + 1;
int32_t x2853 = 131072 * x2849;
int32_t x2854 = x2853 * x2849;
int32_t x2850 = x2849 * x2849;
int32_t x2851 = 2048 * x2850;
int32_t x2852 = 64 * x2851;
bool x2867 = x2719 == 1;
bool x2868 = x2719 == x2849;
bool x2869 = x2867 || x2868;
bool x2870;
if (x2869) {
x2870 = x2869;
} else {
x2870 = false;
}
bool x2886 = x2849 >= 1;
bool x2887;
if (x2886) {
x2887 = x2886;
} else {
x2887 = false;
}
int32_t x2892 = x2848 / 1;
int32_t x2893 = x2892 + 1;
int32_t x2897 = 32768 * x2893;
int32_t x2898 = x2897 * x2893;
int32_t x2894 = x2893 * x2893;
int32_t x2895 = 512 * x2894;
int32_t x2896 = 64 * x2895;
int32_t x2917 = x2893 + 2;
bool x2918 = x2917 >= 3;
bool x2919;
if (x2918) {
x2919 = x2918;
} else {
x2919 = false;
}
int32_t x2924 = x2917 - 3;
int32_t x2925 = x2924 / 1;
int32_t x2926 = x2925 + 1;
int32_t x2930 = 32768 * x2926;
int32_t x2931 = x2930 * x2926;
int32_t x2927 = x2926 * x2926;
int32_t x2928 = 512 * x2927;
int32_t x2929 = 64 * x2928;
bool x2950 = x2926 >= 1;
bool x2951;
if (x2950) {
x2951 = x2950;
} else {
x2951 = false;
}
int32_t x2956 = x2925 / 1;
int32_t x2957 = x2956 + 1;
int32_t x2961 = 131072 * x2957;
int32_t x2962 = x2961 * x2957;
int32_t x2958 = x2957 * x2957;
int32_t x2959 = 2048 * x2958;
int32_t x2960 = 64 * x2959;
bool x2975 = x2849 == 1;
bool x2976 = x2849 == x2957;
bool x2977 = x2975 || x2976;
bool x2978;
if (x2977) {
x2978 = x2977;
} else {
x2978 = false;
}
bool x2994 = x2957 >= 2;
bool x2995;
if (x2994) {
x2995 = x2994;
} else {
x2995 = false;
}
int32_t x3004 = x2957 - 2;
int32_t x3005 = x3004 / 1;
int32_t x3006 = x3005 + 1;
int32_t x3010 = 131072 * x3006;
int32_t x3011 = x3010 * x3006;
for(int x1106=0; x1106 < x1104; x1106++) {
int32_t x1107 = x1106 * 64;
int32_t x1108 = x1107 * 3072;
float* x1109 = x1082+x1108;
int* x1110 = x1083+x1107;
printf("input (size Const(64) x Const(3) x Const(32) x Const(32))\n");
float x1112 = 0.0f;
for(int x1114=0; x1114 < 196608; x1114++) {
float x1115 = x1112;
float x1116 = x1109[x1114];
float x1117 = fabs(x1116);
float x1118 = fabs(x1115);
bool x1119 = x1117 > x1118;
float x1120;
if (x1119) {
x1120 = x1116;
} else {
x1120 = x1115;
}
x1112 = x1120;

}
float x1124 = x1112;
printf("Max Abs: %.5f || ",x1124);
for(int x1127=0; x1127 < 10; x1127++) {
float x1128 = x1109[x1127];
printf("%.5f ",x1128);

}
printf("\n");
// Tensor 'toGPU' invocation.
float* x1134 = (float*)myGpuMalloc(196608 * sizeof(float));
CUDA_CALL(cudaMemcpy(x1134, x1109, 196608 * sizeof(float), cudaMemcpyHostToDevice));
if (x1137) {
} else {
assert(false && "ERROR not specified");
}
float* x1149 = (float*)myGpuMalloc(x1148 * sizeof(float));
float* x1150 = (float*)myMalloc(1 * sizeof(float));;
x1150[0] = 0.0f;
float* x1152 = (float*)myMalloc(1 * sizeof(float));;
x1152[0] = 1.0f;

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
    64, 64, x1143, x1143));

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
    x1152, in_desc, x1134, filt_desc, x714,
    conv_desc, algo, ws_data, ws_size,
    x1150, out_desc, x1149));
};
float* x1155 = (float*)myGpuMalloc(x1146 * sizeof(float));
float* x1156 = (float*)myMalloc(1 * sizeof(float));;
x1156[0] = 0.0f;
float* x1158 = (float*)myMalloc(1 * sizeof(float));;
x1158[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1143, x1143));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1143, x1143));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1158, x1158, in_desc, x1149, out_desc, x1155, sbmv_desc, x876,
    x1011, x378, x588, 1.0E-5));
};
float* x1161 = (float*)myMalloc(1 * sizeof(float));;
x1161[0] = 0.0f;
float* x1163 = (float*)myMalloc(1 * sizeof(float));;
x1163[0] = 1.0f;
float* x1165 = (float*)myGpuMalloc(x1146 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1143, x1143));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1163, x_desc, x1155, x1161, x_desc, x1165));
};
float* x1167 = (float*)myMalloc(1 * sizeof(float));;
x1167[0] = 0.0f;
float* x1169 = (float*)myMalloc(1 * sizeof(float));;
x1169[0] = 1.0f;
float* x1179 = (float*)myGpuMalloc(x1178 * sizeof(float));

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1143, x1143) );

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1173, x1173));

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
    x1169, in_desc, x1165, x1167, out_desc, x1179));
};
if (x1182) {
} else {
assert(false && "ERROR not specified");
}
float* x1194 = (float*)myGpuMalloc(x1193 * sizeof(float));
float* x1195 = (float*)myMalloc(1 * sizeof(float));;
x1195[0] = 0.0f;
float* x1197 = (float*)myMalloc(1 * sizeof(float));;
x1197[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1173, x1173));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1188, x1188));

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
    x1197, in_desc, x1179, filt_desc, x957,
    conv_desc, algo, ws_data, ws_size,
    x1195, out_desc, x1194));
};
float* x1200 = (float*)myGpuMalloc(x1191 * sizeof(float));
float* x1201 = (float*)myMalloc(1 * sizeof(float));;
x1201[0] = 0.0f;
float* x1203 = (float*)myMalloc(1 * sizeof(float));;
x1203[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1188, x1188));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1188, x1188));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1203, x1203, in_desc, x1194, out_desc, x1200, sbmv_desc, x336,
    x417, x600, x411, 1.0E-5));
};
float* x1206 = (float*)myMalloc(1 * sizeof(float));;
x1206[0] = 0.0f;
float* x1208 = (float*)myMalloc(1 * sizeof(float));;
x1208[0] = 1.0f;
float* x1210 = (float*)myGpuMalloc(x1191 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1188, x1188));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1208, x_desc, x1200, x1206, x_desc, x1210));
};
if (x1214) {
} else {
assert(false && "ERROR not specified");
}
float* x1227 = (float*)myGpuMalloc(x1226 * sizeof(float));
float* x1228 = (float*)myMalloc(1 * sizeof(float));;
x1228[0] = 0.0f;
float* x1230 = (float*)myMalloc(1 * sizeof(float));;
x1230[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1188, x1188));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1221, x1221));

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
    x1230, in_desc, x1210, filt_desc, x528,
    conv_desc, algo, ws_data, ws_size,
    x1228, out_desc, x1227));
};
float* x1233 = (float*)myGpuMalloc(x1224 * sizeof(float));
float* x1234 = (float*)myMalloc(1 * sizeof(float));;
x1234[0] = 0.0f;
float* x1236 = (float*)myMalloc(1 * sizeof(float));;
x1236[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1221, x1221));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1221, x1221));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1236, x1236, in_desc, x1227, out_desc, x1233, sbmv_desc, x750,
    x405, x573, x732, 1.0E-5));
};
float* x1239 = (float*)myMalloc(1 * sizeof(float));;
x1239[0] = 0.0f;
float* x1241 = (float*)myMalloc(1 * sizeof(float));;
x1241[0] = 1.0f;
float* x1243 = (float*)myGpuMalloc(x1224 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1221, x1221));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1241, x_desc, x1233, x1239, x_desc, x1243));
};
if (x1246) {
} else {
assert(false && "ERROR not specified");
}
float* x1258 = (float*)myGpuMalloc(x1257 * sizeof(float));
float* x1259 = (float*)myMalloc(1 * sizeof(float));;
x1259[0] = 0.0f;
float* x1261 = (float*)myMalloc(1 * sizeof(float));;
x1261[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1221, x1221));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1252, x1252));

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
    x1261, in_desc, x1243, filt_desc, x354,
    conv_desc, algo, ws_data, ws_size,
    x1259, out_desc, x1258));
};
float* x1264 = (float*)myGpuMalloc(x1255 * sizeof(float));
float* x1265 = (float*)myMalloc(1 * sizeof(float));;
x1265[0] = 0.0f;
float* x1267 = (float*)myMalloc(1 * sizeof(float));;
x1267[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1252, x1252));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1252, x1252));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1267, x1267, in_desc, x1258, out_desc, x1264, sbmv_desc, x855,
    x636, x471, x366, 1.0E-5));
};
if (x1182) {
} else {
assert(false && "ERROR not specified");
}
float* x1277 = (float*)myGpuMalloc(x1276 * sizeof(float));
float* x1278 = (float*)myMalloc(1 * sizeof(float));;
x1278[0] = 0.0f;
float* x1280 = (float*)myMalloc(1 * sizeof(float));;
x1280[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1173, x1173));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1188, x1188));

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
    x1280, in_desc, x1179, filt_desc, x744,
    conv_desc, algo, ws_data, ws_size,
    x1278, out_desc, x1277));
};
float* x1283 = (float*)myGpuMalloc(x1274 * sizeof(float));
float* x1284 = (float*)myMalloc(1 * sizeof(float));;
x1284[0] = 0.0f;
float* x1286 = (float*)myMalloc(1 * sizeof(float));;
x1286[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1188, x1188));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1188, x1188));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1286, x1286, in_desc, x1277, out_desc, x1283, sbmv_desc, x486,
    x867, x1050, x987, 1.0E-5));
};
if (x1292) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1188) x Sym(1188), res:  x Const(64) x Const(256) x Sym(1252) x Sym(1252)");
}
float* x1297 = (float*)myMalloc(1 * sizeof(float));;
x1297[0] = 1.0f;
float* x1299 = (float*)myMalloc(1 * sizeof(float));;
x1299[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1188, x1188));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1252, x1252));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1297, bias_desc, x1283, x1299, out_desc, x1264));
};
float* x1302 = (float*)myMalloc(1 * sizeof(float));;
x1302[0] = 0.0f;
float* x1304 = (float*)myMalloc(1 * sizeof(float));;
x1304[0] = 1.0f;
float* x1306 = (float*)myGpuMalloc(x1255 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1252, x1252));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1304, x_desc, x1264, x1302, x_desc, x1306));
};
if (x1309) {
} else {
assert(false && "ERROR not specified");
}
float* x1321 = (float*)myGpuMalloc(x1320 * sizeof(float));
float* x1322 = (float*)myMalloc(1 * sizeof(float));;
x1322[0] = 0.0f;
float* x1324 = (float*)myMalloc(1 * sizeof(float));;
x1324[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1252, x1252));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1315, x1315));

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
    x1324, in_desc, x1306, filt_desc, x771,
    conv_desc, algo, ws_data, ws_size,
    x1322, out_desc, x1321));
};
float* x1327 = (float*)myGpuMalloc(x1318 * sizeof(float));
float* x1328 = (float*)myMalloc(1 * sizeof(float));;
x1328[0] = 0.0f;
float* x1330 = (float*)myMalloc(1 * sizeof(float));;
x1330[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1315, x1315));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1315, x1315));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1330, x1330, in_desc, x1321, out_desc, x1327, sbmv_desc, x684,
    x438, x288, x564, 1.0E-5));
};
float* x1333 = (float*)myMalloc(1 * sizeof(float));;
x1333[0] = 0.0f;
float* x1335 = (float*)myMalloc(1 * sizeof(float));;
x1335[0] = 1.0f;
float* x1337 = (float*)myGpuMalloc(x1318 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1315, x1315));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1335, x_desc, x1327, x1333, x_desc, x1337));
};
if (x1341) {
} else {
assert(false && "ERROR not specified");
}
float* x1354 = (float*)myGpuMalloc(x1353 * sizeof(float));
float* x1355 = (float*)myMalloc(1 * sizeof(float));;
x1355[0] = 0.0f;
float* x1357 = (float*)myMalloc(1 * sizeof(float));;
x1357[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1315, x1315));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1348, x1348));

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
    x1357, in_desc, x1337, filt_desc, x507,
    conv_desc, algo, ws_data, ws_size,
    x1355, out_desc, x1354));
};
float* x1360 = (float*)myGpuMalloc(x1351 * sizeof(float));
float* x1361 = (float*)myMalloc(1 * sizeof(float));;
x1361[0] = 0.0f;
float* x1363 = (float*)myMalloc(1 * sizeof(float));;
x1363[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1348, x1348));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1348, x1348));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1363, x1363, in_desc, x1354, out_desc, x1360, sbmv_desc, x882,
    x717, x390, x990, 1.0E-5));
};
float* x1366 = (float*)myMalloc(1 * sizeof(float));;
x1366[0] = 0.0f;
float* x1368 = (float*)myMalloc(1 * sizeof(float));;
x1368[0] = 1.0f;
float* x1370 = (float*)myGpuMalloc(x1351 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1348, x1348));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1368, x_desc, x1360, x1366, x_desc, x1370));
};
if (x1373) {
} else {
assert(false && "ERROR not specified");
}
float* x1385 = (float*)myGpuMalloc(x1384 * sizeof(float));
float* x1386 = (float*)myMalloc(1 * sizeof(float));;
x1386[0] = 0.0f;
float* x1388 = (float*)myMalloc(1 * sizeof(float));;
x1388[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1348, x1348));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1379, x1379));

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
    x1388, in_desc, x1370, filt_desc, x648,
    conv_desc, algo, ws_data, ws_size,
    x1386, out_desc, x1385));
};
float* x1391 = (float*)myGpuMalloc(x1382 * sizeof(float));
float* x1392 = (float*)myMalloc(1 * sizeof(float));;
x1392[0] = 0.0f;
float* x1394 = (float*)myMalloc(1 * sizeof(float));;
x1394[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1379, x1379));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1379, x1379));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1394, x1394, in_desc, x1385, out_desc, x1391, sbmv_desc, x432,
    x279, x531, x756, 1.0E-5));
};
if (x1400) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1252) x Sym(1252), res:  x Const(64) x Const(256) x Sym(1379) x Sym(1379)");
}
float* x1405 = (float*)myMalloc(1 * sizeof(float));;
x1405[0] = 1.0f;
float* x1407 = (float*)myMalloc(1 * sizeof(float));;
x1407[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1252, x1252));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1379, x1379));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1405, bias_desc, x1306, x1407, out_desc, x1391));
};
float* x1410 = (float*)myMalloc(1 * sizeof(float));;
x1410[0] = 0.0f;
float* x1412 = (float*)myMalloc(1 * sizeof(float));;
x1412[0] = 1.0f;
float* x1414 = (float*)myGpuMalloc(x1382 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1379, x1379));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1412, x_desc, x1391, x1410, x_desc, x1414));
};
if (x1417) {
} else {
assert(false && "ERROR not specified");
}
float* x1429 = (float*)myGpuMalloc(x1428 * sizeof(float));
float* x1430 = (float*)myMalloc(1 * sizeof(float));;
x1430[0] = 0.0f;
float* x1432 = (float*)myMalloc(1 * sizeof(float));;
x1432[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1379, x1379));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1423, x1423));

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
    x1432, in_desc, x1414, filt_desc, x708,
    conv_desc, algo, ws_data, ws_size,
    x1430, out_desc, x1429));
};
float* x1435 = (float*)myGpuMalloc(x1426 * sizeof(float));
float* x1436 = (float*)myMalloc(1 * sizeof(float));;
x1436[0] = 0.0f;
float* x1438 = (float*)myMalloc(1 * sizeof(float));;
x1438[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1423, x1423));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1423, x1423));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1438, x1438, in_desc, x1429, out_desc, x1435, sbmv_desc, x501,
    x330, x1029, x819, 1.0E-5));
};
float* x1441 = (float*)myMalloc(1 * sizeof(float));;
x1441[0] = 0.0f;
float* x1443 = (float*)myMalloc(1 * sizeof(float));;
x1443[0] = 1.0f;
float* x1445 = (float*)myGpuMalloc(x1426 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1423, x1423));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1443, x_desc, x1435, x1441, x_desc, x1445));
};
if (x1449) {
} else {
assert(false && "ERROR not specified");
}
float* x1462 = (float*)myGpuMalloc(x1461 * sizeof(float));
float* x1463 = (float*)myMalloc(1 * sizeof(float));;
x1463[0] = 0.0f;
float* x1465 = (float*)myMalloc(1 * sizeof(float));;
x1465[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1423, x1423));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1456, x1456));

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
    x1465, in_desc, x1445, filt_desc, x477,
    conv_desc, algo, ws_data, ws_size,
    x1463, out_desc, x1462));
};
float* x1468 = (float*)myGpuMalloc(x1459 * sizeof(float));
float* x1469 = (float*)myMalloc(1 * sizeof(float));;
x1469[0] = 0.0f;
float* x1471 = (float*)myMalloc(1 * sizeof(float));;
x1471[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1456, x1456));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1456, x1456));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1471, x1471, in_desc, x1462, out_desc, x1468, sbmv_desc, x474,
    x663, x795, x612, 1.0E-5));
};
float* x1474 = (float*)myMalloc(1 * sizeof(float));;
x1474[0] = 0.0f;
float* x1476 = (float*)myMalloc(1 * sizeof(float));;
x1476[0] = 1.0f;
float* x1478 = (float*)myGpuMalloc(x1459 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1456, x1456));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1476, x_desc, x1468, x1474, x_desc, x1478));
};
if (x1481) {
} else {
assert(false && "ERROR not specified");
}
float* x1493 = (float*)myGpuMalloc(x1492 * sizeof(float));
float* x1494 = (float*)myMalloc(1 * sizeof(float));;
x1494[0] = 0.0f;
float* x1496 = (float*)myMalloc(1 * sizeof(float));;
x1496[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1456, x1456));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1487, x1487));

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
    x1496, in_desc, x1478, filt_desc, x519,
    conv_desc, algo, ws_data, ws_size,
    x1494, out_desc, x1493));
};
float* x1499 = (float*)myGpuMalloc(x1490 * sizeof(float));
float* x1500 = (float*)myMalloc(1 * sizeof(float));;
x1500[0] = 0.0f;
float* x1502 = (float*)myMalloc(1 * sizeof(float));;
x1502[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1487, x1487));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1487, x1487));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1502, x1502, in_desc, x1493, out_desc, x1499, sbmv_desc, x369,
    x999, x810, x657, 1.0E-5));
};
if (x1508) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1379) x Sym(1379), res:  x Const(64) x Const(256) x Sym(1487) x Sym(1487)");
}
float* x1513 = (float*)myMalloc(1 * sizeof(float));;
x1513[0] = 1.0f;
float* x1515 = (float*)myMalloc(1 * sizeof(float));;
x1515[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1379, x1379));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1487, x1487));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1513, bias_desc, x1414, x1515, out_desc, x1499));
};
float* x1518 = (float*)myMalloc(1 * sizeof(float));;
x1518[0] = 0.0f;
float* x1520 = (float*)myMalloc(1 * sizeof(float));;
x1520[0] = 1.0f;
float* x1522 = (float*)myGpuMalloc(x1490 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1487, x1487));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1520, x_desc, x1499, x1518, x_desc, x1522));
};
if (x1525) {
} else {
assert(false && "ERROR not specified");
}
float* x1537 = (float*)myGpuMalloc(x1536 * sizeof(float));
float* x1538 = (float*)myMalloc(1 * sizeof(float));;
x1538[0] = 0.0f;
float* x1540 = (float*)myMalloc(1 * sizeof(float));;
x1540[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1487, x1487));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1531, x1531));

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
    x1540, in_desc, x1522, filt_desc, x291,
    conv_desc, algo, ws_data, ws_size,
    x1538, out_desc, x1537));
};
float* x1543 = (float*)myGpuMalloc(x1534 * sizeof(float));
float* x1544 = (float*)myMalloc(1 * sizeof(float));;
x1544[0] = 0.0f;
float* x1546 = (float*)myMalloc(1 * sizeof(float));;
x1546[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1531, x1531));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1531, x1531));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1546, x1546, in_desc, x1537, out_desc, x1543, sbmv_desc, x510,
    x774, x870, x660, 1.0E-5));
};
float* x1549 = (float*)myMalloc(1 * sizeof(float));;
x1549[0] = 0.0f;
float* x1551 = (float*)myMalloc(1 * sizeof(float));;
x1551[0] = 1.0f;
float* x1553 = (float*)myGpuMalloc(x1534 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1531, x1531));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1551, x_desc, x1543, x1549, x_desc, x1553));
};
if (x1557) {
} else {
assert(false && "ERROR not specified");
}
float* x1570 = (float*)myGpuMalloc(x1569 * sizeof(float));
float* x1571 = (float*)myMalloc(1 * sizeof(float));;
x1571[0] = 0.0f;
float* x1573 = (float*)myMalloc(1 * sizeof(float));;
x1573[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1531, x1531));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1564, x1564));

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
    x1573, in_desc, x1553, filt_desc, x339,
    conv_desc, algo, ws_data, ws_size,
    x1571, out_desc, x1570));
};
float* x1576 = (float*)myGpuMalloc(x1567 * sizeof(float));
float* x1577 = (float*)myMalloc(1 * sizeof(float));;
x1577[0] = 0.0f;
float* x1579 = (float*)myMalloc(1 * sizeof(float));;
x1579[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1564, x1564));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1564, x1564));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1579, x1579, in_desc, x1570, out_desc, x1576, sbmv_desc, x1014,
    x828, x642, x387, 1.0E-5));
};
float* x1582 = (float*)myMalloc(1 * sizeof(float));;
x1582[0] = 0.0f;
float* x1584 = (float*)myMalloc(1 * sizeof(float));;
x1584[0] = 1.0f;
float* x1586 = (float*)myGpuMalloc(x1567 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1564, x1564));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1584, x_desc, x1576, x1582, x_desc, x1586));
};
if (x1589) {
} else {
assert(false && "ERROR not specified");
}
float* x1601 = (float*)myGpuMalloc(x1600 * sizeof(float));
float* x1602 = (float*)myMalloc(1 * sizeof(float));;
x1602[0] = 0.0f;
float* x1604 = (float*)myMalloc(1 * sizeof(float));;
x1604[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1564, x1564));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1595, x1595));

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
    x1604, in_desc, x1586, filt_desc, x576,
    conv_desc, algo, ws_data, ws_size,
    x1602, out_desc, x1601));
};
float* x1607 = (float*)myGpuMalloc(x1598 * sizeof(float));
float* x1608 = (float*)myMalloc(1 * sizeof(float));;
x1608[0] = 0.0f;
float* x1610 = (float*)myMalloc(1 * sizeof(float));;
x1610[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1595, x1595));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1595, x1595));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1610, x1610, in_desc, x1601, out_desc, x1607, sbmv_desc, x693,
    x888, x705, x561, 1.0E-5));
};
if (x1525) {
} else {
assert(false && "ERROR not specified");
}
float* x1623 = (float*)myGpuMalloc(x1622 * sizeof(float));
float* x1624 = (float*)myMalloc(1 * sizeof(float));;
x1624[0] = 0.0f;
float* x1626 = (float*)myMalloc(1 * sizeof(float));;
x1626[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1487, x1487));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1617, x1617));

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
    x1626, in_desc, x1522, filt_desc, x1032,
    conv_desc, algo, ws_data, ws_size,
    x1624, out_desc, x1623));
};
float* x1629 = (float*)myGpuMalloc(x1620 * sizeof(float));
float* x1630 = (float*)myMalloc(1 * sizeof(float));;
x1630[0] = 0.0f;
float* x1632 = (float*)myMalloc(1 * sizeof(float));;
x1632[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1617, x1617));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1617, x1617));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1632, x1632, in_desc, x1623, out_desc, x1629, sbmv_desc, x879,
    x615, x384, x327, 1.0E-5));
};
if (x1638) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(1617) x Sym(1617), res:  x Const(64) x Const(512) x Sym(1595) x Sym(1595)");
}
float* x1643 = (float*)myMalloc(1 * sizeof(float));;
x1643[0] = 1.0f;
float* x1645 = (float*)myMalloc(1 * sizeof(float));;
x1645[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1617, x1617));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1595, x1595));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1643, bias_desc, x1629, x1645, out_desc, x1607));
};
float* x1648 = (float*)myMalloc(1 * sizeof(float));;
x1648[0] = 0.0f;
float* x1650 = (float*)myMalloc(1 * sizeof(float));;
x1650[0] = 1.0f;
float* x1652 = (float*)myGpuMalloc(x1598 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1595, x1595));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1650, x_desc, x1607, x1648, x_desc, x1652));
};
if (x1655) {
} else {
assert(false && "ERROR not specified");
}
float* x1667 = (float*)myGpuMalloc(x1666 * sizeof(float));
float* x1668 = (float*)myMalloc(1 * sizeof(float));;
x1668[0] = 0.0f;
float* x1670 = (float*)myMalloc(1 * sizeof(float));;
x1670[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1595, x1595));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1661, x1661));

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
    x1670, in_desc, x1652, filt_desc, x1026,
    conv_desc, algo, ws_data, ws_size,
    x1668, out_desc, x1667));
};
float* x1673 = (float*)myGpuMalloc(x1664 * sizeof(float));
float* x1674 = (float*)myMalloc(1 * sizeof(float));;
x1674[0] = 0.0f;
float* x1676 = (float*)myMalloc(1 * sizeof(float));;
x1676[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1661, x1661));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1661, x1661));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1676, x1676, in_desc, x1667, out_desc, x1673, sbmv_desc, x924,
    x309, x558, x789, 1.0E-5));
};
float* x1679 = (float*)myMalloc(1 * sizeof(float));;
x1679[0] = 0.0f;
float* x1681 = (float*)myMalloc(1 * sizeof(float));;
x1681[0] = 1.0f;
float* x1683 = (float*)myGpuMalloc(x1664 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1661, x1661));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1681, x_desc, x1673, x1679, x_desc, x1683));
};
if (x1687) {
} else {
assert(false && "ERROR not specified");
}
float* x1700 = (float*)myGpuMalloc(x1699 * sizeof(float));
float* x1701 = (float*)myMalloc(1 * sizeof(float));;
x1701[0] = 0.0f;
float* x1703 = (float*)myMalloc(1 * sizeof(float));;
x1703[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1661, x1661));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1694, x1694));

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
    x1703, in_desc, x1683, filt_desc, x963,
    conv_desc, algo, ws_data, ws_size,
    x1701, out_desc, x1700));
};
float* x1706 = (float*)myGpuMalloc(x1697 * sizeof(float));
float* x1707 = (float*)myMalloc(1 * sizeof(float));;
x1707[0] = 0.0f;
float* x1709 = (float*)myMalloc(1 * sizeof(float));;
x1709[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1694, x1694));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1694, x1694));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1709, x1709, in_desc, x1700, out_desc, x1706, sbmv_desc, x282,
    x543, x363, x933, 1.0E-5));
};
float* x1712 = (float*)myMalloc(1 * sizeof(float));;
x1712[0] = 0.0f;
float* x1714 = (float*)myMalloc(1 * sizeof(float));;
x1714[0] = 1.0f;
float* x1716 = (float*)myGpuMalloc(x1697 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1694, x1694));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1714, x_desc, x1706, x1712, x_desc, x1716));
};
if (x1719) {
} else {
assert(false && "ERROR not specified");
}
float* x1731 = (float*)myGpuMalloc(x1730 * sizeof(float));
float* x1732 = (float*)myMalloc(1 * sizeof(float));;
x1732[0] = 0.0f;
float* x1734 = (float*)myMalloc(1 * sizeof(float));;
x1734[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1694, x1694));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1725, x1725));

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
    x1734, in_desc, x1716, filt_desc, x591,
    conv_desc, algo, ws_data, ws_size,
    x1732, out_desc, x1731));
};
float* x1737 = (float*)myGpuMalloc(x1728 * sizeof(float));
float* x1738 = (float*)myMalloc(1 * sizeof(float));;
x1738[0] = 0.0f;
float* x1740 = (float*)myMalloc(1 * sizeof(float));;
x1740[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1725, x1725));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1725, x1725));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1740, x1740, in_desc, x1731, out_desc, x1737, sbmv_desc, x414,
    x996, x699, x522, 1.0E-5));
};
if (x1746) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(1595) x Sym(1595), res:  x Const(64) x Const(512) x Sym(1725) x Sym(1725)");
}
float* x1751 = (float*)myMalloc(1 * sizeof(float));;
x1751[0] = 1.0f;
float* x1753 = (float*)myMalloc(1 * sizeof(float));;
x1753[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1595, x1595));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1725, x1725));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1751, bias_desc, x1652, x1753, out_desc, x1737));
};
float* x1756 = (float*)myMalloc(1 * sizeof(float));;
x1756[0] = 0.0f;
float* x1758 = (float*)myMalloc(1 * sizeof(float));;
x1758[0] = 1.0f;
float* x1760 = (float*)myGpuMalloc(x1728 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1725, x1725));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1758, x_desc, x1737, x1756, x_desc, x1760));
};
if (x1763) {
} else {
assert(false && "ERROR not specified");
}
float* x1775 = (float*)myGpuMalloc(x1774 * sizeof(float));
float* x1776 = (float*)myMalloc(1 * sizeof(float));;
x1776[0] = 0.0f;
float* x1778 = (float*)myMalloc(1 * sizeof(float));;
x1778[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1725, x1725));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1769, x1769));

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
    x1778, in_desc, x1760, filt_desc, x846,
    conv_desc, algo, ws_data, ws_size,
    x1776, out_desc, x1775));
};
float* x1781 = (float*)myGpuMalloc(x1772 * sizeof(float));
float* x1782 = (float*)myMalloc(1 * sizeof(float));;
x1782[0] = 0.0f;
float* x1784 = (float*)myMalloc(1 * sizeof(float));;
x1784[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1769, x1769));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1769, x1769));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1784, x1784, in_desc, x1775, out_desc, x1781, sbmv_desc, x393,
    x768, x594, x285, 1.0E-5));
};
float* x1787 = (float*)myMalloc(1 * sizeof(float));;
x1787[0] = 0.0f;
float* x1789 = (float*)myMalloc(1 * sizeof(float));;
x1789[0] = 1.0f;
float* x1791 = (float*)myGpuMalloc(x1772 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1769, x1769));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1789, x_desc, x1781, x1787, x_desc, x1791));
};
if (x1795) {
} else {
assert(false && "ERROR not specified");
}
float* x1808 = (float*)myGpuMalloc(x1807 * sizeof(float));
float* x1809 = (float*)myMalloc(1 * sizeof(float));;
x1809[0] = 0.0f;
float* x1811 = (float*)myMalloc(1 * sizeof(float));;
x1811[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1769, x1769));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1802, x1802));

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
    x1811, in_desc, x1791, filt_desc, x831,
    conv_desc, algo, ws_data, ws_size,
    x1809, out_desc, x1808));
};
float* x1814 = (float*)myGpuMalloc(x1805 * sizeof(float));
float* x1815 = (float*)myMalloc(1 * sizeof(float));;
x1815[0] = 0.0f;
float* x1817 = (float*)myMalloc(1 * sizeof(float));;
x1817[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1802, x1802));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1802, x1802));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1817, x1817, in_desc, x1808, out_desc, x1814, sbmv_desc, x639,
    x441, x909, x1056, 1.0E-5));
};
float* x1820 = (float*)myMalloc(1 * sizeof(float));;
x1820[0] = 0.0f;
float* x1822 = (float*)myMalloc(1 * sizeof(float));;
x1822[0] = 1.0f;
float* x1824 = (float*)myGpuMalloc(x1805 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1802, x1802));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1822, x_desc, x1814, x1820, x_desc, x1824));
};
if (x1827) {
} else {
assert(false && "ERROR not specified");
}
float* x1839 = (float*)myGpuMalloc(x1838 * sizeof(float));
float* x1840 = (float*)myMalloc(1 * sizeof(float));;
x1840[0] = 0.0f;
float* x1842 = (float*)myMalloc(1 * sizeof(float));;
x1842[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1802, x1802));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1833, x1833));

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
    x1842, in_desc, x1824, filt_desc, x381,
    conv_desc, algo, ws_data, ws_size,
    x1840, out_desc, x1839));
};
float* x1845 = (float*)myGpuMalloc(x1836 * sizeof(float));
float* x1846 = (float*)myMalloc(1 * sizeof(float));;
x1846[0] = 0.0f;
float* x1848 = (float*)myMalloc(1 * sizeof(float));;
x1848[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1833, x1833));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1833, x1833));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1848, x1848, in_desc, x1839, out_desc, x1845, sbmv_desc, x759,
    x504, x333, x927, 1.0E-5));
};
if (x1854) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(1725) x Sym(1725), res:  x Const(64) x Const(512) x Sym(1833) x Sym(1833)");
}
float* x1859 = (float*)myMalloc(1 * sizeof(float));;
x1859[0] = 1.0f;
float* x1861 = (float*)myMalloc(1 * sizeof(float));;
x1861[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1725, x1725));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1833, x1833));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1859, bias_desc, x1760, x1861, out_desc, x1845));
};
float* x1864 = (float*)myMalloc(1 * sizeof(float));;
x1864[0] = 0.0f;
float* x1866 = (float*)myMalloc(1 * sizeof(float));;
x1866[0] = 1.0f;
float* x1868 = (float*)myGpuMalloc(x1836 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1833, x1833));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1866, x_desc, x1845, x1864, x_desc, x1868));
};
if (x1871) {
} else {
assert(false && "ERROR not specified");
}
float* x1883 = (float*)myGpuMalloc(x1882 * sizeof(float));
float* x1884 = (float*)myMalloc(1 * sizeof(float));;
x1884[0] = 0.0f;
float* x1886 = (float*)myMalloc(1 * sizeof(float));;
x1886[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1833, x1833));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1877, x1877));

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
    x1886, in_desc, x1868, filt_desc, x654,
    conv_desc, algo, ws_data, ws_size,
    x1884, out_desc, x1883));
};
float* x1889 = (float*)myGpuMalloc(x1880 * sizeof(float));
float* x1890 = (float*)myMalloc(1 * sizeof(float));;
x1890[0] = 0.0f;
float* x1892 = (float*)myMalloc(1 * sizeof(float));;
x1892[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1877, x1877));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1877, x1877));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1892, x1892, in_desc, x1883, out_desc, x1889, sbmv_desc, x375,
    x984, x966, x1041, 1.0E-5));
};
float* x1895 = (float*)myMalloc(1 * sizeof(float));;
x1895[0] = 0.0f;
float* x1897 = (float*)myMalloc(1 * sizeof(float));;
x1897[0] = 1.0f;
float* x1899 = (float*)myGpuMalloc(x1880 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1877, x1877));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1897, x_desc, x1889, x1895, x_desc, x1899));
};
if (x1903) {
} else {
assert(false && "ERROR not specified");
}
float* x1916 = (float*)myGpuMalloc(x1915 * sizeof(float));
float* x1917 = (float*)myMalloc(1 * sizeof(float));;
x1917[0] = 0.0f;
float* x1919 = (float*)myMalloc(1 * sizeof(float));;
x1919[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1877, x1877));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1910, x1910));

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
    x1919, in_desc, x1899, filt_desc, x753,
    conv_desc, algo, ws_data, ws_size,
    x1917, out_desc, x1916));
};
float* x1922 = (float*)myGpuMalloc(x1913 * sizeof(float));
float* x1923 = (float*)myMalloc(1 * sizeof(float));;
x1923[0] = 0.0f;
float* x1925 = (float*)myMalloc(1 * sizeof(float));;
x1925[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1910, x1910));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1910, x1910));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1925, x1925, in_desc, x1916, out_desc, x1922, sbmv_desc, x495,
    x372, x1062, x702, 1.0E-5));
};
float* x1928 = (float*)myMalloc(1 * sizeof(float));;
x1928[0] = 0.0f;
float* x1930 = (float*)myMalloc(1 * sizeof(float));;
x1930[0] = 1.0f;
float* x1932 = (float*)myGpuMalloc(x1913 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1910, x1910));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1930, x_desc, x1922, x1928, x_desc, x1932));
};
if (x1935) {
} else {
assert(false && "ERROR not specified");
}
float* x1947 = (float*)myGpuMalloc(x1946 * sizeof(float));
float* x1948 = (float*)myMalloc(1 * sizeof(float));;
x1948[0] = 0.0f;
float* x1950 = (float*)myMalloc(1 * sizeof(float));;
x1950[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1910, x1910));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1941, x1941));

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
    x1950, in_desc, x1932, filt_desc, x423,
    conv_desc, algo, ws_data, ws_size,
    x1948, out_desc, x1947));
};
float* x1953 = (float*)myGpuMalloc(x1944 * sizeof(float));
float* x1954 = (float*)myMalloc(1 * sizeof(float));;
x1954[0] = 0.0f;
float* x1956 = (float*)myMalloc(1 * sizeof(float));;
x1956[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1941, x1941));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1941, x1941));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1956, x1956, in_desc, x1947, out_desc, x1953, sbmv_desc, x726,
    x420, x315, x960, 1.0E-5));
};
if (x1962) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(1833) x Sym(1833), res:  x Const(64) x Const(512) x Sym(1941) x Sym(1941)");
}
float* x1967 = (float*)myMalloc(1 * sizeof(float));;
x1967[0] = 1.0f;
float* x1969 = (float*)myMalloc(1 * sizeof(float));;
x1969[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1833, x1833));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1941, x1941));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1967, bias_desc, x1868, x1969, out_desc, x1953));
};
float* x1972 = (float*)myMalloc(1 * sizeof(float));;
x1972[0] = 0.0f;
float* x1974 = (float*)myMalloc(1 * sizeof(float));;
x1974[0] = 1.0f;
float* x1976 = (float*)myGpuMalloc(x1944 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1941, x1941));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1974, x_desc, x1953, x1972, x_desc, x1976));
};
if (x1979) {
} else {
assert(false && "ERROR not specified");
}
float* x1991 = (float*)myGpuMalloc(x1990 * sizeof(float));
float* x1992 = (float*)myMalloc(1 * sizeof(float));;
x1992[0] = 0.0f;
float* x1994 = (float*)myMalloc(1 * sizeof(float));;
x1994[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1941, x1941));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1985, x1985));

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
    x1994, in_desc, x1976, filt_desc, x798,
    conv_desc, algo, ws_data, ws_size,
    x1992, out_desc, x1991));
};
float* x1997 = (float*)myGpuMalloc(x1988 * sizeof(float));
float* x1998 = (float*)myMalloc(1 * sizeof(float));;
x1998[0] = 0.0f;
float* x2000 = (float*)myMalloc(1 * sizeof(float));;
x2000[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1985, x1985));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1985, x1985));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2000, x2000, in_desc, x1991, out_desc, x1997, sbmv_desc, x1068,
    x321, x651, x852, 1.0E-5));
};
float* x2003 = (float*)myMalloc(1 * sizeof(float));;
x2003[0] = 0.0f;
float* x2005 = (float*)myMalloc(1 * sizeof(float));;
x2005[0] = 1.0f;
float* x2007 = (float*)myGpuMalloc(x1988 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1985, x1985));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2005, x_desc, x1997, x2003, x_desc, x2007));
};
if (x2011) {
} else {
assert(false && "ERROR not specified");
}
float* x2024 = (float*)myGpuMalloc(x2023 * sizeof(float));
float* x2025 = (float*)myMalloc(1 * sizeof(float));;
x2025[0] = 0.0f;
float* x2027 = (float*)myMalloc(1 * sizeof(float));;
x2027[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1985, x1985));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2018, x2018));

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
    x2027, in_desc, x2007, filt_desc, x783,
    conv_desc, algo, ws_data, ws_size,
    x2025, out_desc, x2024));
};
float* x2030 = (float*)myGpuMalloc(x2021 * sizeof(float));
float* x2031 = (float*)myMalloc(1 * sizeof(float));;
x2031[0] = 0.0f;
float* x2033 = (float*)myMalloc(1 * sizeof(float));;
x2033[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2018, x2018));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2018, x2018));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2033, x2033, in_desc, x2024, out_desc, x2030, sbmv_desc, x582,
    x306, x945, x555, 1.0E-5));
};
float* x2036 = (float*)myMalloc(1 * sizeof(float));;
x2036[0] = 0.0f;
float* x2038 = (float*)myMalloc(1 * sizeof(float));;
x2038[0] = 1.0f;
float* x2040 = (float*)myGpuMalloc(x2021 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2018, x2018));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2038, x_desc, x2030, x2036, x_desc, x2040));
};
if (x2043) {
} else {
assert(false && "ERROR not specified");
}
float* x2055 = (float*)myGpuMalloc(x2054 * sizeof(float));
float* x2056 = (float*)myMalloc(1 * sizeof(float));;
x2056[0] = 0.0f;
float* x2058 = (float*)myMalloc(1 * sizeof(float));;
x2058[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2018, x2018));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2049, x2049));

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
    x2058, in_desc, x2040, filt_desc, x1065,
    conv_desc, algo, ws_data, ws_size,
    x2056, out_desc, x2055));
};
float* x2061 = (float*)myGpuMalloc(x2052 * sizeof(float));
float* x2062 = (float*)myMalloc(1 * sizeof(float));;
x2062[0] = 0.0f;
float* x2064 = (float*)myMalloc(1 * sizeof(float));;
x2064[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2049, x2049));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2049, x2049));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2064, x2064, in_desc, x2055, out_desc, x2061, sbmv_desc, x312,
    x609, x906, x1059, 1.0E-5));
};
if (x1979) {
} else {
assert(false && "ERROR not specified");
}
float* x2077 = (float*)myGpuMalloc(x2076 * sizeof(float));
float* x2078 = (float*)myMalloc(1 * sizeof(float));;
x2078[0] = 0.0f;
float* x2080 = (float*)myMalloc(1 * sizeof(float));;
x2080[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1941, x1941));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2071, x2071));

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
    x2080, in_desc, x1976, filt_desc, x483,
    conv_desc, algo, ws_data, ws_size,
    x2078, out_desc, x2077));
};
float* x2083 = (float*)myGpuMalloc(x2074 * sizeof(float));
float* x2084 = (float*)myMalloc(1 * sizeof(float));;
x2084[0] = 0.0f;
float* x2086 = (float*)myMalloc(1 * sizeof(float));;
x2086[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2071, x2071));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2071, x2071));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2086, x2086, in_desc, x2077, out_desc, x2083, sbmv_desc, x345,
    x918, x516, x891, 1.0E-5));
};
if (x2092) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2071) x Sym(2071), res:  x Const(64) x Const(1024) x Sym(2049) x Sym(2049)");
}
float* x2097 = (float*)myMalloc(1 * sizeof(float));;
x2097[0] = 1.0f;
float* x2099 = (float*)myMalloc(1 * sizeof(float));;
x2099[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2071, x2071));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2049, x2049));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2097, bias_desc, x2083, x2099, out_desc, x2061));
};
float* x2102 = (float*)myMalloc(1 * sizeof(float));;
x2102[0] = 0.0f;
float* x2104 = (float*)myMalloc(1 * sizeof(float));;
x2104[0] = 1.0f;
float* x2106 = (float*)myGpuMalloc(x2052 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2049, x2049));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2104, x_desc, x2061, x2102, x_desc, x2106));
};
if (x2109) {
} else {
assert(false && "ERROR not specified");
}
float* x2121 = (float*)myGpuMalloc(x2120 * sizeof(float));
float* x2122 = (float*)myMalloc(1 * sizeof(float));;
x2122[0] = 0.0f;
float* x2124 = (float*)myMalloc(1 * sizeof(float));;
x2124[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2049, x2049));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2115, x2115));

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
    x2124, in_desc, x2106, filt_desc, x297,
    conv_desc, algo, ws_data, ws_size,
    x2122, out_desc, x2121));
};
float* x2127 = (float*)myGpuMalloc(x2118 * sizeof(float));
float* x2128 = (float*)myMalloc(1 * sizeof(float));;
x2128[0] = 0.0f;
float* x2130 = (float*)myMalloc(1 * sizeof(float));;
x2130[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2115, x2115));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2115, x2115));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2130, x2130, in_desc, x2121, out_desc, x2127, sbmv_desc, x348,
    x915, x1035, x729, 1.0E-5));
};
float* x2133 = (float*)myMalloc(1 * sizeof(float));;
x2133[0] = 0.0f;
float* x2135 = (float*)myMalloc(1 * sizeof(float));;
x2135[0] = 1.0f;
float* x2137 = (float*)myGpuMalloc(x2118 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2115, x2115));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2135, x_desc, x2127, x2133, x_desc, x2137));
};
if (x2141) {
} else {
assert(false && "ERROR not specified");
}
float* x2154 = (float*)myGpuMalloc(x2153 * sizeof(float));
float* x2155 = (float*)myMalloc(1 * sizeof(float));;
x2155[0] = 0.0f;
float* x2157 = (float*)myMalloc(1 * sizeof(float));;
x2157[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2115, x2115));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2148, x2148));

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
    x2157, in_desc, x2137, filt_desc, x351,
    conv_desc, algo, ws_data, ws_size,
    x2155, out_desc, x2154));
};
float* x2160 = (float*)myGpuMalloc(x2151 * sizeof(float));
float* x2161 = (float*)myMalloc(1 * sizeof(float));;
x2161[0] = 0.0f;
float* x2163 = (float*)myMalloc(1 * sizeof(float));;
x2163[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2148, x2148));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2148, x2148));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2163, x2163, in_desc, x2154, out_desc, x2160, sbmv_desc, x1071,
    x546, x858, x969, 1.0E-5));
};
float* x2166 = (float*)myMalloc(1 * sizeof(float));;
x2166[0] = 0.0f;
float* x2168 = (float*)myMalloc(1 * sizeof(float));;
x2168[0] = 1.0f;
float* x2170 = (float*)myGpuMalloc(x2151 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2148, x2148));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2168, x_desc, x2160, x2166, x_desc, x2170));
};
if (x2173) {
} else {
assert(false && "ERROR not specified");
}
float* x2185 = (float*)myGpuMalloc(x2184 * sizeof(float));
float* x2186 = (float*)myMalloc(1 * sizeof(float));;
x2186[0] = 0.0f;
float* x2188 = (float*)myMalloc(1 * sizeof(float));;
x2188[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2148, x2148));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2179, x2179));

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
    x2188, in_desc, x2170, filt_desc, x426,
    conv_desc, algo, ws_data, ws_size,
    x2186, out_desc, x2185));
};
float* x2191 = (float*)myGpuMalloc(x2182 * sizeof(float));
float* x2192 = (float*)myMalloc(1 * sizeof(float));;
x2192[0] = 0.0f;
float* x2194 = (float*)myMalloc(1 * sizeof(float));;
x2194[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2179, x2179));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2179, x2179));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2194, x2194, in_desc, x2185, out_desc, x2191, sbmv_desc, x318,
    x954, x804, x687, 1.0E-5));
};
if (x2200) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2049) x Sym(2049), res:  x Const(64) x Const(1024) x Sym(2179) x Sym(2179)");
}
float* x2205 = (float*)myMalloc(1 * sizeof(float));;
x2205[0] = 1.0f;
float* x2207 = (float*)myMalloc(1 * sizeof(float));;
x2207[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2049, x2049));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2179, x2179));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2205, bias_desc, x2106, x2207, out_desc, x2191));
};
float* x2210 = (float*)myMalloc(1 * sizeof(float));;
x2210[0] = 0.0f;
float* x2212 = (float*)myMalloc(1 * sizeof(float));;
x2212[0] = 1.0f;
float* x2214 = (float*)myGpuMalloc(x2182 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2179, x2179));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2212, x_desc, x2191, x2210, x_desc, x2214));
};
if (x2217) {
} else {
assert(false && "ERROR not specified");
}
float* x2229 = (float*)myGpuMalloc(x2228 * sizeof(float));
float* x2230 = (float*)myMalloc(1 * sizeof(float));;
x2230[0] = 0.0f;
float* x2232 = (float*)myMalloc(1 * sizeof(float));;
x2232[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2179, x2179));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2223, x2223));

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
    x2232, in_desc, x2214, filt_desc, x912,
    conv_desc, algo, ws_data, ws_size,
    x2230, out_desc, x2229));
};
float* x2235 = (float*)myGpuMalloc(x2226 * sizeof(float));
float* x2236 = (float*)myMalloc(1 * sizeof(float));;
x2236[0] = 0.0f;
float* x2238 = (float*)myMalloc(1 * sizeof(float));;
x2238[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2223, x2223));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2223, x2223));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2238, x2238, in_desc, x2229, out_desc, x2235, sbmv_desc, x645,
    x849, x792, x780, 1.0E-5));
};
float* x2241 = (float*)myMalloc(1 * sizeof(float));;
x2241[0] = 0.0f;
float* x2243 = (float*)myMalloc(1 * sizeof(float));;
x2243[0] = 1.0f;
float* x2245 = (float*)myGpuMalloc(x2226 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2223, x2223));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2243, x_desc, x2235, x2241, x_desc, x2245));
};
if (x2249) {
} else {
assert(false && "ERROR not specified");
}
float* x2262 = (float*)myGpuMalloc(x2261 * sizeof(float));
float* x2263 = (float*)myMalloc(1 * sizeof(float));;
x2263[0] = 0.0f;
float* x2265 = (float*)myMalloc(1 * sizeof(float));;
x2265[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2223, x2223));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2256, x2256));

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
    x2265, in_desc, x2245, filt_desc, x300,
    conv_desc, algo, ws_data, ws_size,
    x2263, out_desc, x2262));
};
float* x2268 = (float*)myGpuMalloc(x2259 * sizeof(float));
float* x2269 = (float*)myMalloc(1 * sizeof(float));;
x2269[0] = 0.0f;
float* x2271 = (float*)myMalloc(1 * sizeof(float));;
x2271[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2256, x2256));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2256, x2256));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2271, x2271, in_desc, x2262, out_desc, x2268, sbmv_desc, x942,
    x834, x630, x447, 1.0E-5));
};
float* x2274 = (float*)myMalloc(1 * sizeof(float));;
x2274[0] = 0.0f;
float* x2276 = (float*)myMalloc(1 * sizeof(float));;
x2276[0] = 1.0f;
float* x2278 = (float*)myGpuMalloc(x2259 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2256, x2256));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2276, x_desc, x2268, x2274, x_desc, x2278));
};
if (x2281) {
} else {
assert(false && "ERROR not specified");
}
float* x2293 = (float*)myGpuMalloc(x2292 * sizeof(float));
float* x2294 = (float*)myMalloc(1 * sizeof(float));;
x2294[0] = 0.0f;
float* x2296 = (float*)myMalloc(1 * sizeof(float));;
x2296[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2256, x2256));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2287, x2287));

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
    x2296, in_desc, x2278, filt_desc, x606,
    conv_desc, algo, ws_data, ws_size,
    x2294, out_desc, x2293));
};
float* x2299 = (float*)myGpuMalloc(x2290 * sizeof(float));
float* x2300 = (float*)myMalloc(1 * sizeof(float));;
x2300[0] = 0.0f;
float* x2302 = (float*)myMalloc(1 * sizeof(float));;
x2302[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2287, x2287));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2287, x2287));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2302, x2302, in_desc, x2293, out_desc, x2299, sbmv_desc, x1047,
    x429, x678, x822, 1.0E-5));
};
if (x2308) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2179) x Sym(2179), res:  x Const(64) x Const(1024) x Sym(2287) x Sym(2287)");
}
float* x2313 = (float*)myMalloc(1 * sizeof(float));;
x2313[0] = 1.0f;
float* x2315 = (float*)myMalloc(1 * sizeof(float));;
x2315[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2179, x2179));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2287, x2287));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2313, bias_desc, x2214, x2315, out_desc, x2299));
};
float* x2318 = (float*)myMalloc(1 * sizeof(float));;
x2318[0] = 0.0f;
float* x2320 = (float*)myMalloc(1 * sizeof(float));;
x2320[0] = 1.0f;
float* x2322 = (float*)myGpuMalloc(x2290 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2287, x2287));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2320, x_desc, x2299, x2318, x_desc, x2322));
};
if (x2325) {
} else {
assert(false && "ERROR not specified");
}
float* x2337 = (float*)myGpuMalloc(x2336 * sizeof(float));
float* x2338 = (float*)myMalloc(1 * sizeof(float));;
x2338[0] = 0.0f;
float* x2340 = (float*)myMalloc(1 * sizeof(float));;
x2340[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2287, x2287));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2331, x2331));

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
    x2340, in_desc, x2322, filt_desc, x276,
    conv_desc, algo, ws_data, ws_size,
    x2338, out_desc, x2337));
};
float* x2343 = (float*)myGpuMalloc(x2334 * sizeof(float));
float* x2344 = (float*)myMalloc(1 * sizeof(float));;
x2344[0] = 0.0f;
float* x2346 = (float*)myMalloc(1 * sizeof(float));;
x2346[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2331, x2331));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2331, x2331));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2346, x2346, in_desc, x2337, out_desc, x2343, sbmv_desc, x534,
    x981, x747, x552, 1.0E-5));
};
float* x2349 = (float*)myMalloc(1 * sizeof(float));;
x2349[0] = 0.0f;
float* x2351 = (float*)myMalloc(1 * sizeof(float));;
x2351[0] = 1.0f;
float* x2353 = (float*)myGpuMalloc(x2334 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2331, x2331));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2351, x_desc, x2343, x2349, x_desc, x2353));
};
if (x2357) {
} else {
assert(false && "ERROR not specified");
}
float* x2370 = (float*)myGpuMalloc(x2369 * sizeof(float));
float* x2371 = (float*)myMalloc(1 * sizeof(float));;
x2371[0] = 0.0f;
float* x2373 = (float*)myMalloc(1 * sizeof(float));;
x2373[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2331, x2331));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2364, x2364));

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
    x2373, in_desc, x2353, filt_desc, x1005,
    conv_desc, algo, ws_data, ws_size,
    x2371, out_desc, x2370));
};
float* x2376 = (float*)myGpuMalloc(x2367 * sizeof(float));
float* x2377 = (float*)myMalloc(1 * sizeof(float));;
x2377[0] = 0.0f;
float* x2379 = (float*)myMalloc(1 * sizeof(float));;
x2379[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2364, x2364));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2364, x2364));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2379, x2379, in_desc, x2370, out_desc, x2376, sbmv_desc, x480,
    x666, x816, x948, 1.0E-5));
};
float* x2382 = (float*)myMalloc(1 * sizeof(float));;
x2382[0] = 0.0f;
float* x2384 = (float*)myMalloc(1 * sizeof(float));;
x2384[0] = 1.0f;
float* x2386 = (float*)myGpuMalloc(x2367 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2364, x2364));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2384, x_desc, x2376, x2382, x_desc, x2386));
};
if (x2389) {
} else {
assert(false && "ERROR not specified");
}
float* x2401 = (float*)myGpuMalloc(x2400 * sizeof(float));
float* x2402 = (float*)myMalloc(1 * sizeof(float));;
x2402[0] = 0.0f;
float* x2404 = (float*)myMalloc(1 * sizeof(float));;
x2404[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2364, x2364));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2395, x2395));

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
    x2404, in_desc, x2386, filt_desc, x525,
    conv_desc, algo, ws_data, ws_size,
    x2402, out_desc, x2401));
};
float* x2407 = (float*)myGpuMalloc(x2398 * sizeof(float));
float* x2408 = (float*)myMalloc(1 * sizeof(float));;
x2408[0] = 0.0f;
float* x2410 = (float*)myMalloc(1 * sizeof(float));;
x2410[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2395, x2395));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2395, x2395));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2410, x2410, in_desc, x2401, out_desc, x2407, sbmv_desc, x972,
    x696, x951, x741, 1.0E-5));
};
if (x2416) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2287) x Sym(2287), res:  x Const(64) x Const(1024) x Sym(2395) x Sym(2395)");
}
float* x2421 = (float*)myMalloc(1 * sizeof(float));;
x2421[0] = 1.0f;
float* x2423 = (float*)myMalloc(1 * sizeof(float));;
x2423[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2287, x2287));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2395, x2395));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2421, bias_desc, x2322, x2423, out_desc, x2407));
};
float* x2426 = (float*)myMalloc(1 * sizeof(float));;
x2426[0] = 0.0f;
float* x2428 = (float*)myMalloc(1 * sizeof(float));;
x2428[0] = 1.0f;
float* x2430 = (float*)myGpuMalloc(x2398 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2395, x2395));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2428, x_desc, x2407, x2426, x_desc, x2430));
};
if (x2433) {
} else {
assert(false && "ERROR not specified");
}
float* x2445 = (float*)myGpuMalloc(x2444 * sizeof(float));
float* x2446 = (float*)myMalloc(1 * sizeof(float));;
x2446[0] = 0.0f;
float* x2448 = (float*)myMalloc(1 * sizeof(float));;
x2448[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2395, x2395));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2439, x2439));

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
    x2448, in_desc, x2430, filt_desc, x324,
    conv_desc, algo, ws_data, ws_size,
    x2446, out_desc, x2445));
};
float* x2451 = (float*)myGpuMalloc(x2442 * sizeof(float));
float* x2452 = (float*)myMalloc(1 * sizeof(float));;
x2452[0] = 0.0f;
float* x2454 = (float*)myMalloc(1 * sizeof(float));;
x2454[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2439, x2439));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2439, x2439));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2454, x2454, in_desc, x2445, out_desc, x2451, sbmv_desc, x489,
    x813, x1020, x465, 1.0E-5));
};
float* x2457 = (float*)myMalloc(1 * sizeof(float));;
x2457[0] = 0.0f;
float* x2459 = (float*)myMalloc(1 * sizeof(float));;
x2459[0] = 1.0f;
float* x2461 = (float*)myGpuMalloc(x2442 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2439, x2439));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2459, x_desc, x2451, x2457, x_desc, x2461));
};
if (x2465) {
} else {
assert(false && "ERROR not specified");
}
float* x2478 = (float*)myGpuMalloc(x2477 * sizeof(float));
float* x2479 = (float*)myMalloc(1 * sizeof(float));;
x2479[0] = 0.0f;
float* x2481 = (float*)myMalloc(1 * sizeof(float));;
x2481[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2439, x2439));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2472, x2472));

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
    x2481, in_desc, x2461, filt_desc, x1044,
    conv_desc, algo, ws_data, ws_size,
    x2479, out_desc, x2478));
};
float* x2484 = (float*)myGpuMalloc(x2475 * sizeof(float));
float* x2485 = (float*)myMalloc(1 * sizeof(float));;
x2485[0] = 0.0f;
float* x2487 = (float*)myMalloc(1 * sizeof(float));;
x2487[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2472, x2472));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2472, x2472));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2487, x2487, in_desc, x2478, out_desc, x2484, sbmv_desc, x762,
    x585, x1008, x570, 1.0E-5));
};
float* x2490 = (float*)myMalloc(1 * sizeof(float));;
x2490[0] = 0.0f;
float* x2492 = (float*)myMalloc(1 * sizeof(float));;
x2492[0] = 1.0f;
float* x2494 = (float*)myGpuMalloc(x2475 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2472, x2472));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2492, x_desc, x2484, x2490, x_desc, x2494));
};
if (x2497) {
} else {
assert(false && "ERROR not specified");
}
float* x2509 = (float*)myGpuMalloc(x2508 * sizeof(float));
float* x2510 = (float*)myMalloc(1 * sizeof(float));;
x2510[0] = 0.0f;
float* x2512 = (float*)myMalloc(1 * sizeof(float));;
x2512[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2472, x2472));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2503, x2503));

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
    x2512, in_desc, x2494, filt_desc, x921,
    conv_desc, algo, ws_data, ws_size,
    x2510, out_desc, x2509));
};
float* x2515 = (float*)myGpuMalloc(x2506 * sizeof(float));
float* x2516 = (float*)myMalloc(1 * sizeof(float));;
x2516[0] = 0.0f;
float* x2518 = (float*)myMalloc(1 * sizeof(float));;
x2518[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2503, x2503));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2503, x2503));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2518, x2518, in_desc, x2509, out_desc, x2515, sbmv_desc, x435,
    x618, x885, x1074, 1.0E-5));
};
if (x2524) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2395) x Sym(2395), res:  x Const(64) x Const(1024) x Sym(2503) x Sym(2503)");
}
float* x2529 = (float*)myMalloc(1 * sizeof(float));;
x2529[0] = 1.0f;
float* x2531 = (float*)myMalloc(1 * sizeof(float));;
x2531[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2395, x2395));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2503, x2503));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2529, bias_desc, x2430, x2531, out_desc, x2515));
};
float* x2534 = (float*)myMalloc(1 * sizeof(float));;
x2534[0] = 0.0f;
float* x2536 = (float*)myMalloc(1 * sizeof(float));;
x2536[0] = 1.0f;
float* x2538 = (float*)myGpuMalloc(x2506 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2503, x2503));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2536, x_desc, x2515, x2534, x_desc, x2538));
};
if (x2541) {
} else {
assert(false && "ERROR not specified");
}
float* x2553 = (float*)myGpuMalloc(x2552 * sizeof(float));
float* x2554 = (float*)myMalloc(1 * sizeof(float));;
x2554[0] = 0.0f;
float* x2556 = (float*)myMalloc(1 * sizeof(float));;
x2556[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2503, x2503));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2547, x2547));

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
    x2556, in_desc, x2538, filt_desc, x711,
    conv_desc, algo, ws_data, ws_size,
    x2554, out_desc, x2553));
};
float* x2559 = (float*)myGpuMalloc(x2550 * sizeof(float));
float* x2560 = (float*)myMalloc(1 * sizeof(float));;
x2560[0] = 0.0f;
float* x2562 = (float*)myMalloc(1 * sizeof(float));;
x2562[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2547, x2547));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2547, x2547));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2562, x2562, in_desc, x2553, out_desc, x2559, sbmv_desc, x513,
    x1017, x498, x786, 1.0E-5));
};
float* x2565 = (float*)myMalloc(1 * sizeof(float));;
x2565[0] = 0.0f;
float* x2567 = (float*)myMalloc(1 * sizeof(float));;
x2567[0] = 1.0f;
float* x2569 = (float*)myGpuMalloc(x2550 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2547, x2547));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2567, x_desc, x2559, x2565, x_desc, x2569));
};
if (x2573) {
} else {
assert(false && "ERROR not specified");
}
float* x2586 = (float*)myGpuMalloc(x2585 * sizeof(float));
float* x2587 = (float*)myMalloc(1 * sizeof(float));;
x2587[0] = 0.0f;
float* x2589 = (float*)myMalloc(1 * sizeof(float));;
x2589[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2547, x2547));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2580, x2580));

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
    x2589, in_desc, x2569, filt_desc, x936,
    conv_desc, algo, ws_data, ws_size,
    x2587, out_desc, x2586));
};
float* x2592 = (float*)myGpuMalloc(x2583 * sizeof(float));
float* x2593 = (float*)myMalloc(1 * sizeof(float));;
x2593[0] = 0.0f;
float* x2595 = (float*)myMalloc(1 * sizeof(float));;
x2595[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2580, x2580));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2580, x2580));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2595, x2595, in_desc, x2586, out_desc, x2592, sbmv_desc, x681,
    x825, x468, x978, 1.0E-5));
};
float* x2598 = (float*)myMalloc(1 * sizeof(float));;
x2598[0] = 0.0f;
float* x2600 = (float*)myMalloc(1 * sizeof(float));;
x2600[0] = 1.0f;
float* x2602 = (float*)myGpuMalloc(x2583 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2580, x2580));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2600, x_desc, x2592, x2598, x_desc, x2602));
};
if (x2605) {
} else {
assert(false && "ERROR not specified");
}
float* x2617 = (float*)myGpuMalloc(x2616 * sizeof(float));
float* x2618 = (float*)myMalloc(1 * sizeof(float));;
x2618[0] = 0.0f;
float* x2620 = (float*)myMalloc(1 * sizeof(float));;
x2620[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2580, x2580));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2611, x2611));

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
    x2620, in_desc, x2602, filt_desc, x549,
    conv_desc, algo, ws_data, ws_size,
    x2618, out_desc, x2617));
};
float* x2623 = (float*)myGpuMalloc(x2614 * sizeof(float));
float* x2624 = (float*)myMalloc(1 * sizeof(float));;
x2624[0] = 0.0f;
float* x2626 = (float*)myMalloc(1 * sizeof(float));;
x2626[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2611, x2611));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2611, x2611));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2626, x2626, in_desc, x2617, out_desc, x2623, sbmv_desc, x1002,
    x537, x624, x807, 1.0E-5));
};
if (x2632) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2503) x Sym(2503), res:  x Const(64) x Const(1024) x Sym(2611) x Sym(2611)");
}
float* x2637 = (float*)myMalloc(1 * sizeof(float));;
x2637[0] = 1.0f;
float* x2639 = (float*)myMalloc(1 * sizeof(float));;
x2639[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2503, x2503));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2611, x2611));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2637, bias_desc, x2538, x2639, out_desc, x2623));
};
float* x2642 = (float*)myMalloc(1 * sizeof(float));;
x2642[0] = 0.0f;
float* x2644 = (float*)myMalloc(1 * sizeof(float));;
x2644[0] = 1.0f;
float* x2646 = (float*)myGpuMalloc(x2614 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2611, x2611));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2644, x_desc, x2623, x2642, x_desc, x2646));
};
if (x2649) {
} else {
assert(false && "ERROR not specified");
}
float* x2661 = (float*)myGpuMalloc(x2660 * sizeof(float));
float* x2662 = (float*)myMalloc(1 * sizeof(float));;
x2662[0] = 0.0f;
float* x2664 = (float*)myMalloc(1 * sizeof(float));;
x2664[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2611, x2611));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2655, x2655));

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
    x2664, in_desc, x2646, filt_desc, x675,
    conv_desc, algo, ws_data, ws_size,
    x2662, out_desc, x2661));
};
float* x2667 = (float*)myGpuMalloc(x2658 * sizeof(float));
float* x2668 = (float*)myMalloc(1 * sizeof(float));;
x2668[0] = 0.0f;
float* x2670 = (float*)myMalloc(1 * sizeof(float));;
x2670[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2655, x2655));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2655, x2655));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2670, x2670, in_desc, x2661, out_desc, x2667, sbmv_desc, x861,
    x930, x459, x621, 1.0E-5));
};
float* x2673 = (float*)myMalloc(1 * sizeof(float));;
x2673[0] = 0.0f;
float* x2675 = (float*)myMalloc(1 * sizeof(float));;
x2675[0] = 1.0f;
float* x2677 = (float*)myGpuMalloc(x2658 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2655, x2655));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2675, x_desc, x2667, x2673, x_desc, x2677));
};
if (x2681) {
} else {
assert(false && "ERROR not specified");
}
float* x2694 = (float*)myGpuMalloc(x2693 * sizeof(float));
float* x2695 = (float*)myMalloc(1 * sizeof(float));;
x2695[0] = 0.0f;
float* x2697 = (float*)myMalloc(1 * sizeof(float));;
x2697[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2655, x2655));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2688, x2688));

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
    x2697, in_desc, x2677, filt_desc, x360,
    conv_desc, algo, ws_data, ws_size,
    x2695, out_desc, x2694));
};
float* x2700 = (float*)myGpuMalloc(x2691 * sizeof(float));
float* x2701 = (float*)myMalloc(1 * sizeof(float));;
x2701[0] = 0.0f;
float* x2703 = (float*)myMalloc(1 * sizeof(float));;
x2703[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2688, x2688));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2688, x2688));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2703, x2703, in_desc, x2694, out_desc, x2700, sbmv_desc, x873,
    x735, x597, x408, 1.0E-5));
};
float* x2706 = (float*)myMalloc(1 * sizeof(float));;
x2706[0] = 0.0f;
float* x2708 = (float*)myMalloc(1 * sizeof(float));;
x2708[0] = 1.0f;
float* x2710 = (float*)myGpuMalloc(x2691 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2688, x2688));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2708, x_desc, x2700, x2706, x_desc, x2710));
};
if (x2713) {
} else {
assert(false && "ERROR not specified");
}
float* x2725 = (float*)myGpuMalloc(x2724 * sizeof(float));
float* x2726 = (float*)myMalloc(1 * sizeof(float));;
x2726[0] = 0.0f;
float* x2728 = (float*)myMalloc(1 * sizeof(float));;
x2728[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2688, x2688));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2719, x2719));

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
    x2728, in_desc, x2710, filt_desc, x894,
    conv_desc, algo, ws_data, ws_size,
    x2726, out_desc, x2725));
};
float* x2731 = (float*)myGpuMalloc(x2722 * sizeof(float));
float* x2732 = (float*)myMalloc(1 * sizeof(float));;
x2732[0] = 0.0f;
float* x2734 = (float*)myMalloc(1 * sizeof(float));;
x2734[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2719, x2719));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2719, x2719));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2734, x2734, in_desc, x2725, out_desc, x2731, sbmv_desc, x975,
    x444, x603, x837, 1.0E-5));
};
if (x2649) {
} else {
assert(false && "ERROR not specified");
}
float* x2747 = (float*)myGpuMalloc(x2746 * sizeof(float));
float* x2748 = (float*)myMalloc(1 * sizeof(float));;
x2748[0] = 0.0f;
float* x2750 = (float*)myMalloc(1 * sizeof(float));;
x2750[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2611, x2611));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2741, x2741));

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
    x2750, in_desc, x2646, filt_desc, x900,
    conv_desc, algo, ws_data, ws_size,
    x2748, out_desc, x2747));
};
float* x2753 = (float*)myGpuMalloc(x2744 * sizeof(float));
float* x2754 = (float*)myMalloc(1 * sizeof(float));;
x2754[0] = 0.0f;
float* x2756 = (float*)myMalloc(1 * sizeof(float));;
x2756[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2741, x2741));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2741, x2741));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2756, x2756, in_desc, x2747, out_desc, x2753, sbmv_desc, x777,
    x579, x450, x633, 1.0E-5));
};
if (x2762) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(2741) x Sym(2741), res:  x Const(64) x Const(2048) x Sym(2719) x Sym(2719)");
}
float* x2767 = (float*)myMalloc(1 * sizeof(float));;
x2767[0] = 1.0f;
float* x2769 = (float*)myMalloc(1 * sizeof(float));;
x2769[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2741, x2741));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2719, x2719));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2767, bias_desc, x2753, x2769, out_desc, x2731));
};
float* x2772 = (float*)myMalloc(1 * sizeof(float));;
x2772[0] = 0.0f;
float* x2774 = (float*)myMalloc(1 * sizeof(float));;
x2774[0] = 1.0f;
float* x2776 = (float*)myGpuMalloc(x2722 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2719, x2719));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2774, x_desc, x2731, x2772, x_desc, x2776));
};
if (x2779) {
} else {
assert(false && "ERROR not specified");
}
float* x2791 = (float*)myGpuMalloc(x2790 * sizeof(float));
float* x2792 = (float*)myMalloc(1 * sizeof(float));;
x2792[0] = 0.0f;
float* x2794 = (float*)myMalloc(1 * sizeof(float));;
x2794[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2719, x2719));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 2048, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2785, x2785));

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
    x2794, in_desc, x2776, filt_desc, x903,
    conv_desc, algo, ws_data, ws_size,
    x2792, out_desc, x2791));
};
float* x2797 = (float*)myGpuMalloc(x2788 * sizeof(float));
float* x2798 = (float*)myMalloc(1 * sizeof(float));;
x2798[0] = 0.0f;
float* x2800 = (float*)myMalloc(1 * sizeof(float));;
x2800[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2785, x2785));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2785, x2785));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2800, x2800, in_desc, x2791, out_desc, x2797, sbmv_desc, x396,
    x669, x720, x453, 1.0E-5));
};
float* x2803 = (float*)myMalloc(1 * sizeof(float));;
x2803[0] = 0.0f;
float* x2805 = (float*)myMalloc(1 * sizeof(float));;
x2805[0] = 1.0f;
float* x2807 = (float*)myGpuMalloc(x2788 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2785, x2785));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2805, x_desc, x2797, x2803, x_desc, x2807));
};
if (x2811) {
} else {
assert(false && "ERROR not specified");
}
float* x2824 = (float*)myGpuMalloc(x2823 * sizeof(float));
float* x2825 = (float*)myMalloc(1 * sizeof(float));;
x2825[0] = 0.0f;
float* x2827 = (float*)myMalloc(1 * sizeof(float));;
x2827[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2785, x2785));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2818, x2818));

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
    x2827, in_desc, x2807, filt_desc, x723,
    conv_desc, algo, ws_data, ws_size,
    x2825, out_desc, x2824));
};
float* x2830 = (float*)myGpuMalloc(x2821 * sizeof(float));
float* x2831 = (float*)myMalloc(1 * sizeof(float));;
x2831[0] = 0.0f;
float* x2833 = (float*)myMalloc(1 * sizeof(float));;
x2833[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2818, x2818));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2818, x2818));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2833, x2833, in_desc, x2824, out_desc, x2830, sbmv_desc, x738,
    x456, x672, x843, 1.0E-5));
};
float* x2836 = (float*)myMalloc(1 * sizeof(float));;
x2836[0] = 0.0f;
float* x2838 = (float*)myMalloc(1 * sizeof(float));;
x2838[0] = 1.0f;
float* x2840 = (float*)myGpuMalloc(x2821 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2818, x2818));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2838, x_desc, x2830, x2836, x_desc, x2840));
};
if (x2843) {
} else {
assert(false && "ERROR not specified");
}
float* x2855 = (float*)myGpuMalloc(x2854 * sizeof(float));
float* x2856 = (float*)myMalloc(1 * sizeof(float));;
x2856[0] = 0.0f;
float* x2858 = (float*)myMalloc(1 * sizeof(float));;
x2858[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2818, x2818));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2849, x2849));

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
    x2858, in_desc, x2840, filt_desc, x399,
    conv_desc, algo, ws_data, ws_size,
    x2856, out_desc, x2855));
};
float* x2861 = (float*)myGpuMalloc(x2852 * sizeof(float));
float* x2862 = (float*)myMalloc(1 * sizeof(float));;
x2862[0] = 0.0f;
float* x2864 = (float*)myMalloc(1 * sizeof(float));;
x2864[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2849, x2849));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2849, x2849));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2864, x2864, in_desc, x2855, out_desc, x2861, sbmv_desc, x540,
    x690, x462, x993, 1.0E-5));
};
if (x2870) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(2719) x Sym(2719), res:  x Const(64) x Const(2048) x Sym(2849) x Sym(2849)");
}
float* x2875 = (float*)myMalloc(1 * sizeof(float));;
x2875[0] = 1.0f;
float* x2877 = (float*)myMalloc(1 * sizeof(float));;
x2877[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2719, x2719));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2849, x2849));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2875, bias_desc, x2776, x2877, out_desc, x2861));
};
float* x2880 = (float*)myMalloc(1 * sizeof(float));;
x2880[0] = 0.0f;
float* x2882 = (float*)myMalloc(1 * sizeof(float));;
x2882[0] = 1.0f;
float* x2884 = (float*)myGpuMalloc(x2852 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2849, x2849));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2882, x_desc, x2861, x2880, x_desc, x2884));
};
if (x2887) {
} else {
assert(false && "ERROR not specified");
}
float* x2899 = (float*)myGpuMalloc(x2898 * sizeof(float));
float* x2900 = (float*)myMalloc(1 * sizeof(float));;
x2900[0] = 0.0f;
float* x2902 = (float*)myMalloc(1 * sizeof(float));;
x2902[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2849, x2849));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 2048, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2893, x2893));

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
    x2902, in_desc, x2884, filt_desc, x1053,
    conv_desc, algo, ws_data, ws_size,
    x2900, out_desc, x2899));
};
float* x2905 = (float*)myGpuMalloc(x2896 * sizeof(float));
float* x2906 = (float*)myMalloc(1 * sizeof(float));;
x2906[0] = 0.0f;
float* x2908 = (float*)myMalloc(1 * sizeof(float));;
x2908[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2893, x2893));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2893, x2893));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2908, x2908, in_desc, x2899, out_desc, x2905, sbmv_desc, x303,
    x492, x897, x1023, 1.0E-5));
};
float* x2911 = (float*)myMalloc(1 * sizeof(float));;
x2911[0] = 0.0f;
float* x2913 = (float*)myMalloc(1 * sizeof(float));;
x2913[0] = 1.0f;
float* x2915 = (float*)myGpuMalloc(x2896 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2893, x2893));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2913, x_desc, x2905, x2911, x_desc, x2915));
};
if (x2919) {
} else {
assert(false && "ERROR not specified");
}
float* x2932 = (float*)myGpuMalloc(x2931 * sizeof(float));
float* x2933 = (float*)myMalloc(1 * sizeof(float));;
x2933[0] = 0.0f;
float* x2935 = (float*)myMalloc(1 * sizeof(float));;
x2935[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2893, x2893));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2926, x2926));

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
    x2935, in_desc, x2915, filt_desc, x342,
    conv_desc, algo, ws_data, ws_size,
    x2933, out_desc, x2932));
};
float* x2938 = (float*)myGpuMalloc(x2929 * sizeof(float));
float* x2939 = (float*)myMalloc(1 * sizeof(float));;
x2939[0] = 0.0f;
float* x2941 = (float*)myMalloc(1 * sizeof(float));;
x2941[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2926, x2926));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2926, x2926));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2941, x2941, in_desc, x2932, out_desc, x2938, sbmv_desc, x840,
    x765, x294, x864, 1.0E-5));
};
float* x2944 = (float*)myMalloc(1 * sizeof(float));;
x2944[0] = 0.0f;
float* x2946 = (float*)myMalloc(1 * sizeof(float));;
x2946[0] = 1.0f;
float* x2948 = (float*)myGpuMalloc(x2929 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2926, x2926));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2946, x_desc, x2938, x2944, x_desc, x2948));
};
if (x2951) {
} else {
assert(false && "ERROR not specified");
}
float* x2963 = (float*)myGpuMalloc(x2962 * sizeof(float));
float* x2964 = (float*)myMalloc(1 * sizeof(float));;
x2964[0] = 0.0f;
float* x2966 = (float*)myMalloc(1 * sizeof(float));;
x2966[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2926, x2926));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2957, x2957));

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
    x2966, in_desc, x2948, filt_desc, x357,
    conv_desc, algo, ws_data, ws_size,
    x2964, out_desc, x2963));
};
float* x2969 = (float*)myGpuMalloc(x2960 * sizeof(float));
float* x2970 = (float*)myMalloc(1 * sizeof(float));;
x2970[0] = 0.0f;
float* x2972 = (float*)myMalloc(1 * sizeof(float));;
x2972[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2957, x2957));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2957, x2957));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2972, x2972, in_desc, x2963, out_desc, x2969, sbmv_desc, x567,
    x801, x1038, x627, 1.0E-5));
};
if (x2978) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(2849) x Sym(2849), res:  x Const(64) x Const(2048) x Sym(2957) x Sym(2957)");
}
float* x2983 = (float*)myMalloc(1 * sizeof(float));;
x2983[0] = 1.0f;
float* x2985 = (float*)myMalloc(1 * sizeof(float));;
x2985[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2849, x2849));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2957, x2957));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2983, bias_desc, x2884, x2985, out_desc, x2969));
};
float* x2988 = (float*)myMalloc(1 * sizeof(float));;
x2988[0] = 0.0f;
float* x2990 = (float*)myMalloc(1 * sizeof(float));;
x2990[0] = 1.0f;
float* x2992 = (float*)myGpuMalloc(x2960 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2957, x2957));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2990, x_desc, x2969, x2988, x_desc, x2992));
};
if (x2995) {
} else {
assert(false && "Image too small for averagePool_batch:  x Const(64) x Const(2048) x Sym(2957) x Sym(2957)|(2,2)");
}
float* x3000 = (float*)myMalloc(1 * sizeof(float));;
x3000[0] = 0.0f;
float* x3002 = (float*)myMalloc(1 * sizeof(float));;
x3002[0] = 1.0f;
float* x3012 = (float*)myGpuMalloc(x3011 * sizeof(float));

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2957, x2957) );

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3006, x3006));

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
    x3002, in_desc, x2992, x3000, out_desc, x3012));
};
// gemm: List(Const(64), Const(2048)), Vector(Const(10), Const(2048))
float* x3015 = (float*)myGpuMalloc(640 * sizeof(float));
float* x3016 = (float*)myMalloc(1 * sizeof(float));;
x3016[0] = 0.0f;
float* x3018 = (float*)myMalloc(1 * sizeof(float));;
x3018[0] = 1.0f;
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 10,64,2048,x3018,x939,2048,x3012,2048,x3016,x3015,10));
float* x3021 = (float*)myMalloc(1 * sizeof(float));;
x3021[0] = 1.0f;
float* x3023 = (float*)myMalloc(1 * sizeof(float));;
x3023[0] = 1.0f;

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
    cudnnHandle, x3021, bias_desc, x402, x3023, out_desc, x3015));
};
// Tensor 'toCPU' invocation.
float* x3027 = (float*)myMalloc(640 * sizeof(float));;
CUDA_CALL(cudaMemcpy(x3027, x3015, 640 * sizeof(float), cudaMemcpyDeviceToHost));
printf("output (size Const(64) x Const(10))\n");
float x3030 = 0.0f;
for(int x3032=0; x3032 < 640; x3032++) {
float x3033 = x3030;
float x3034 = x3027[x3032];
float x3035 = fabs(x3034);
float x3036 = fabs(x3033);
bool x3037 = x3035 > x3036;
float x3038;
if (x3037) {
x3038 = x3034;
} else {
x3038 = x3033;
}
x3030 = x3038;

}
float x3042 = x3030;
printf("Max Abs: %.5f || ",x3042);
for(int x3044=0; x3044 < 10; x3044++) {
float x3045 = x3027[x3044];
printf("%.5f ",x3045);

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


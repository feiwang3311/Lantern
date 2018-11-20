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
int32_t x1136 = 31 / 1;
int32_t x1137 = x1136 + 1;
int32_t x1141 = 4096 * x1137;
int32_t x1142 = x1141 * x1137;
int32_t x1138 = x1137 * x1137;
int32_t x1139 = 64 * x1138;
int32_t x1140 = 64 * x1139;
int32_t x1165 = x1137 - 2;
int32_t x1166 = x1165 / 2;
int32_t x1167 = x1166 + 1;
int32_t x1171 = 4096 * x1167;
int32_t x1172 = x1171 * x1167;
bool x1175 = x1167 >= 1;
bool x1176;
if (x1175) {
x1176 = x1175;
} else {
x1176 = false;
}
int32_t x1181 = x1166 / 1;
int32_t x1182 = x1181 + 1;
int32_t x1186 = 4096 * x1182;
int32_t x1187 = x1186 * x1182;
int32_t x1183 = x1182 * x1182;
int32_t x1184 = 64 * x1183;
int32_t x1185 = 64 * x1184;
int32_t x1206 = x1182 + 2;
bool x1207 = x1206 >= 3;
bool x1208;
if (x1207) {
x1208 = x1207;
} else {
x1208 = false;
}
int32_t x1213 = x1206 - 3;
int32_t x1214 = x1213 / 1;
int32_t x1215 = x1214 + 1;
int32_t x1219 = 4096 * x1215;
int32_t x1220 = x1219 * x1215;
int32_t x1216 = x1215 * x1215;
int32_t x1217 = 64 * x1216;
int32_t x1218 = 64 * x1217;
bool x1239 = x1215 >= 1;
bool x1240;
if (x1239) {
x1240 = x1239;
} else {
x1240 = false;
}
int32_t x1245 = x1214 / 1;
int32_t x1246 = x1245 + 1;
int32_t x1250 = 16384 * x1246;
int32_t x1251 = x1250 * x1246;
int32_t x1247 = x1246 * x1246;
int32_t x1248 = 256 * x1247;
int32_t x1249 = 64 * x1248;
int32_t x1269 = 16384 * x1182;
int32_t x1270 = x1269 * x1182;
int32_t x1267 = 256 * x1183;
int32_t x1268 = 64 * x1267;
bool x1283 = x1182 == 1;
bool x1284 = x1182 == x1246;
bool x1285 = x1283 || x1284;
bool x1286;
if (x1285) {
x1286 = x1285;
} else {
x1286 = false;
}
bool x1302 = x1246 >= 1;
bool x1303;
if (x1302) {
x1303 = x1302;
} else {
x1303 = false;
}
int32_t x1308 = x1245 / 1;
int32_t x1309 = x1308 + 1;
int32_t x1313 = 4096 * x1309;
int32_t x1314 = x1313 * x1309;
int32_t x1310 = x1309 * x1309;
int32_t x1311 = 64 * x1310;
int32_t x1312 = 64 * x1311;
int32_t x1333 = x1309 + 2;
bool x1334 = x1333 >= 3;
bool x1335;
if (x1334) {
x1335 = x1334;
} else {
x1335 = false;
}
int32_t x1340 = x1333 - 3;
int32_t x1341 = x1340 / 1;
int32_t x1342 = x1341 + 1;
int32_t x1346 = 4096 * x1342;
int32_t x1347 = x1346 * x1342;
int32_t x1343 = x1342 * x1342;
int32_t x1344 = 64 * x1343;
int32_t x1345 = 64 * x1344;
bool x1366 = x1342 >= 1;
bool x1367;
if (x1366) {
x1367 = x1366;
} else {
x1367 = false;
}
int32_t x1372 = x1341 / 1;
int32_t x1373 = x1372 + 1;
int32_t x1377 = 16384 * x1373;
int32_t x1378 = x1377 * x1373;
int32_t x1374 = x1373 * x1373;
int32_t x1375 = 256 * x1374;
int32_t x1376 = 64 * x1375;
bool x1391 = x1246 == 1;
bool x1392 = x1246 == x1373;
bool x1393 = x1391 || x1392;
bool x1394;
if (x1393) {
x1394 = x1393;
} else {
x1394 = false;
}
bool x1410 = x1373 >= 1;
bool x1411;
if (x1410) {
x1411 = x1410;
} else {
x1411 = false;
}
int32_t x1416 = x1372 / 1;
int32_t x1417 = x1416 + 1;
int32_t x1421 = 4096 * x1417;
int32_t x1422 = x1421 * x1417;
int32_t x1418 = x1417 * x1417;
int32_t x1419 = 64 * x1418;
int32_t x1420 = 64 * x1419;
int32_t x1441 = x1417 + 2;
bool x1442 = x1441 >= 3;
bool x1443;
if (x1442) {
x1443 = x1442;
} else {
x1443 = false;
}
int32_t x1448 = x1441 - 3;
int32_t x1449 = x1448 / 1;
int32_t x1450 = x1449 + 1;
int32_t x1454 = 4096 * x1450;
int32_t x1455 = x1454 * x1450;
int32_t x1451 = x1450 * x1450;
int32_t x1452 = 64 * x1451;
int32_t x1453 = 64 * x1452;
bool x1474 = x1450 >= 1;
bool x1475;
if (x1474) {
x1475 = x1474;
} else {
x1475 = false;
}
int32_t x1480 = x1449 / 1;
int32_t x1481 = x1480 + 1;
int32_t x1485 = 16384 * x1481;
int32_t x1486 = x1485 * x1481;
int32_t x1482 = x1481 * x1481;
int32_t x1483 = 256 * x1482;
int32_t x1484 = 64 * x1483;
bool x1499 = x1373 == 1;
bool x1500 = x1373 == x1481;
bool x1501 = x1499 || x1500;
bool x1502;
if (x1501) {
x1502 = x1501;
} else {
x1502 = false;
}
bool x1518 = x1481 >= 1;
bool x1519;
if (x1518) {
x1519 = x1518;
} else {
x1519 = false;
}
int32_t x1524 = x1480 / 1;
int32_t x1525 = x1524 + 1;
int32_t x1529 = 8192 * x1525;
int32_t x1530 = x1529 * x1525;
int32_t x1526 = x1525 * x1525;
int32_t x1527 = 128 * x1526;
int32_t x1528 = 64 * x1527;
int32_t x1549 = x1525 + 2;
bool x1550 = x1549 >= 3;
bool x1551;
if (x1550) {
x1551 = x1550;
} else {
x1551 = false;
}
int32_t x1556 = x1549 - 3;
int32_t x1557 = x1556 / 2;
int32_t x1558 = x1557 + 1;
int32_t x1562 = 8192 * x1558;
int32_t x1563 = x1562 * x1558;
int32_t x1559 = x1558 * x1558;
int32_t x1560 = 128 * x1559;
int32_t x1561 = 64 * x1560;
bool x1582 = x1558 >= 1;
bool x1583;
if (x1582) {
x1583 = x1582;
} else {
x1583 = false;
}
int32_t x1588 = x1557 / 1;
int32_t x1589 = x1588 + 1;
int32_t x1593 = 32768 * x1589;
int32_t x1594 = x1593 * x1589;
int32_t x1590 = x1589 * x1589;
int32_t x1591 = 512 * x1590;
int32_t x1592 = 64 * x1591;
int32_t x1610 = x1480 / 2;
int32_t x1611 = x1610 + 1;
int32_t x1615 = 32768 * x1611;
int32_t x1616 = x1615 * x1611;
int32_t x1612 = x1611 * x1611;
int32_t x1613 = 512 * x1612;
int32_t x1614 = 64 * x1613;
bool x1629 = x1611 == 1;
bool x1630 = x1611 == x1589;
bool x1631 = x1629 || x1630;
bool x1632;
if (x1631) {
x1632 = x1631;
} else {
x1632 = false;
}
bool x1648 = x1589 >= 1;
bool x1649;
if (x1648) {
x1649 = x1648;
} else {
x1649 = false;
}
int32_t x1654 = x1588 / 1;
int32_t x1655 = x1654 + 1;
int32_t x1659 = 8192 * x1655;
int32_t x1660 = x1659 * x1655;
int32_t x1656 = x1655 * x1655;
int32_t x1657 = 128 * x1656;
int32_t x1658 = 64 * x1657;
int32_t x1679 = x1655 + 2;
bool x1680 = x1679 >= 3;
bool x1681;
if (x1680) {
x1681 = x1680;
} else {
x1681 = false;
}
int32_t x1686 = x1679 - 3;
int32_t x1687 = x1686 / 1;
int32_t x1688 = x1687 + 1;
int32_t x1692 = 8192 * x1688;
int32_t x1693 = x1692 * x1688;
int32_t x1689 = x1688 * x1688;
int32_t x1690 = 128 * x1689;
int32_t x1691 = 64 * x1690;
bool x1712 = x1688 >= 1;
bool x1713;
if (x1712) {
x1713 = x1712;
} else {
x1713 = false;
}
int32_t x1718 = x1687 / 1;
int32_t x1719 = x1718 + 1;
int32_t x1723 = 32768 * x1719;
int32_t x1724 = x1723 * x1719;
int32_t x1720 = x1719 * x1719;
int32_t x1721 = 512 * x1720;
int32_t x1722 = 64 * x1721;
bool x1737 = x1589 == 1;
bool x1738 = x1589 == x1719;
bool x1739 = x1737 || x1738;
bool x1740;
if (x1739) {
x1740 = x1739;
} else {
x1740 = false;
}
bool x1756 = x1719 >= 1;
bool x1757;
if (x1756) {
x1757 = x1756;
} else {
x1757 = false;
}
int32_t x1762 = x1718 / 1;
int32_t x1763 = x1762 + 1;
int32_t x1767 = 8192 * x1763;
int32_t x1768 = x1767 * x1763;
int32_t x1764 = x1763 * x1763;
int32_t x1765 = 128 * x1764;
int32_t x1766 = 64 * x1765;
int32_t x1787 = x1763 + 2;
bool x1788 = x1787 >= 3;
bool x1789;
if (x1788) {
x1789 = x1788;
} else {
x1789 = false;
}
int32_t x1794 = x1787 - 3;
int32_t x1795 = x1794 / 1;
int32_t x1796 = x1795 + 1;
int32_t x1800 = 8192 * x1796;
int32_t x1801 = x1800 * x1796;
int32_t x1797 = x1796 * x1796;
int32_t x1798 = 128 * x1797;
int32_t x1799 = 64 * x1798;
bool x1820 = x1796 >= 1;
bool x1821;
if (x1820) {
x1821 = x1820;
} else {
x1821 = false;
}
int32_t x1826 = x1795 / 1;
int32_t x1827 = x1826 + 1;
int32_t x1831 = 32768 * x1827;
int32_t x1832 = x1831 * x1827;
int32_t x1828 = x1827 * x1827;
int32_t x1829 = 512 * x1828;
int32_t x1830 = 64 * x1829;
bool x1845 = x1719 == 1;
bool x1846 = x1719 == x1827;
bool x1847 = x1845 || x1846;
bool x1848;
if (x1847) {
x1848 = x1847;
} else {
x1848 = false;
}
bool x1864 = x1827 >= 1;
bool x1865;
if (x1864) {
x1865 = x1864;
} else {
x1865 = false;
}
int32_t x1870 = x1826 / 1;
int32_t x1871 = x1870 + 1;
int32_t x1875 = 8192 * x1871;
int32_t x1876 = x1875 * x1871;
int32_t x1872 = x1871 * x1871;
int32_t x1873 = 128 * x1872;
int32_t x1874 = 64 * x1873;
int32_t x1895 = x1871 + 2;
bool x1896 = x1895 >= 3;
bool x1897;
if (x1896) {
x1897 = x1896;
} else {
x1897 = false;
}
int32_t x1902 = x1895 - 3;
int32_t x1903 = x1902 / 1;
int32_t x1904 = x1903 + 1;
int32_t x1908 = 8192 * x1904;
int32_t x1909 = x1908 * x1904;
int32_t x1905 = x1904 * x1904;
int32_t x1906 = 128 * x1905;
int32_t x1907 = 64 * x1906;
bool x1928 = x1904 >= 1;
bool x1929;
if (x1928) {
x1929 = x1928;
} else {
x1929 = false;
}
int32_t x1934 = x1903 / 1;
int32_t x1935 = x1934 + 1;
int32_t x1939 = 32768 * x1935;
int32_t x1940 = x1939 * x1935;
int32_t x1936 = x1935 * x1935;
int32_t x1937 = 512 * x1936;
int32_t x1938 = 64 * x1937;
bool x1953 = x1827 == 1;
bool x1954 = x1827 == x1935;
bool x1955 = x1953 || x1954;
bool x1956;
if (x1955) {
x1956 = x1955;
} else {
x1956 = false;
}
bool x1972 = x1935 >= 1;
bool x1973;
if (x1972) {
x1973 = x1972;
} else {
x1973 = false;
}
int32_t x1978 = x1934 / 1;
int32_t x1979 = x1978 + 1;
int32_t x1983 = 16384 * x1979;
int32_t x1984 = x1983 * x1979;
int32_t x1980 = x1979 * x1979;
int32_t x1981 = 256 * x1980;
int32_t x1982 = 64 * x1981;
int32_t x2003 = x1979 + 2;
bool x2004 = x2003 >= 3;
bool x2005;
if (x2004) {
x2005 = x2004;
} else {
x2005 = false;
}
int32_t x2010 = x2003 - 3;
int32_t x2011 = x2010 / 2;
int32_t x2012 = x2011 + 1;
int32_t x2016 = 16384 * x2012;
int32_t x2017 = x2016 * x2012;
int32_t x2013 = x2012 * x2012;
int32_t x2014 = 256 * x2013;
int32_t x2015 = 64 * x2014;
bool x2036 = x2012 >= 1;
bool x2037;
if (x2036) {
x2037 = x2036;
} else {
x2037 = false;
}
int32_t x2042 = x2011 / 1;
int32_t x2043 = x2042 + 1;
int32_t x2047 = 65536 * x2043;
int32_t x2048 = x2047 * x2043;
int32_t x2044 = x2043 * x2043;
int32_t x2045 = 1024 * x2044;
int32_t x2046 = 64 * x2045;
int32_t x2064 = x1934 / 2;
int32_t x2065 = x2064 + 1;
int32_t x2069 = 65536 * x2065;
int32_t x2070 = x2069 * x2065;
int32_t x2066 = x2065 * x2065;
int32_t x2067 = 1024 * x2066;
int32_t x2068 = 64 * x2067;
bool x2083 = x2065 == 1;
bool x2084 = x2065 == x2043;
bool x2085 = x2083 || x2084;
bool x2086;
if (x2085) {
x2086 = x2085;
} else {
x2086 = false;
}
bool x2102 = x2043 >= 1;
bool x2103;
if (x2102) {
x2103 = x2102;
} else {
x2103 = false;
}
int32_t x2108 = x2042 / 1;
int32_t x2109 = x2108 + 1;
int32_t x2113 = 16384 * x2109;
int32_t x2114 = x2113 * x2109;
int32_t x2110 = x2109 * x2109;
int32_t x2111 = 256 * x2110;
int32_t x2112 = 64 * x2111;
int32_t x2133 = x2109 + 2;
bool x2134 = x2133 >= 3;
bool x2135;
if (x2134) {
x2135 = x2134;
} else {
x2135 = false;
}
int32_t x2140 = x2133 - 3;
int32_t x2141 = x2140 / 1;
int32_t x2142 = x2141 + 1;
int32_t x2146 = 16384 * x2142;
int32_t x2147 = x2146 * x2142;
int32_t x2143 = x2142 * x2142;
int32_t x2144 = 256 * x2143;
int32_t x2145 = 64 * x2144;
bool x2166 = x2142 >= 1;
bool x2167;
if (x2166) {
x2167 = x2166;
} else {
x2167 = false;
}
int32_t x2172 = x2141 / 1;
int32_t x2173 = x2172 + 1;
int32_t x2177 = 65536 * x2173;
int32_t x2178 = x2177 * x2173;
int32_t x2174 = x2173 * x2173;
int32_t x2175 = 1024 * x2174;
int32_t x2176 = 64 * x2175;
bool x2191 = x2043 == 1;
bool x2192 = x2043 == x2173;
bool x2193 = x2191 || x2192;
bool x2194;
if (x2193) {
x2194 = x2193;
} else {
x2194 = false;
}
bool x2210 = x2173 >= 1;
bool x2211;
if (x2210) {
x2211 = x2210;
} else {
x2211 = false;
}
int32_t x2216 = x2172 / 1;
int32_t x2217 = x2216 + 1;
int32_t x2221 = 16384 * x2217;
int32_t x2222 = x2221 * x2217;
int32_t x2218 = x2217 * x2217;
int32_t x2219 = 256 * x2218;
int32_t x2220 = 64 * x2219;
int32_t x2241 = x2217 + 2;
bool x2242 = x2241 >= 3;
bool x2243;
if (x2242) {
x2243 = x2242;
} else {
x2243 = false;
}
int32_t x2248 = x2241 - 3;
int32_t x2249 = x2248 / 1;
int32_t x2250 = x2249 + 1;
int32_t x2254 = 16384 * x2250;
int32_t x2255 = x2254 * x2250;
int32_t x2251 = x2250 * x2250;
int32_t x2252 = 256 * x2251;
int32_t x2253 = 64 * x2252;
bool x2274 = x2250 >= 1;
bool x2275;
if (x2274) {
x2275 = x2274;
} else {
x2275 = false;
}
int32_t x2280 = x2249 / 1;
int32_t x2281 = x2280 + 1;
int32_t x2285 = 65536 * x2281;
int32_t x2286 = x2285 * x2281;
int32_t x2282 = x2281 * x2281;
int32_t x2283 = 1024 * x2282;
int32_t x2284 = 64 * x2283;
bool x2299 = x2173 == 1;
bool x2300 = x2173 == x2281;
bool x2301 = x2299 || x2300;
bool x2302;
if (x2301) {
x2302 = x2301;
} else {
x2302 = false;
}
bool x2318 = x2281 >= 1;
bool x2319;
if (x2318) {
x2319 = x2318;
} else {
x2319 = false;
}
int32_t x2324 = x2280 / 1;
int32_t x2325 = x2324 + 1;
int32_t x2329 = 16384 * x2325;
int32_t x2330 = x2329 * x2325;
int32_t x2326 = x2325 * x2325;
int32_t x2327 = 256 * x2326;
int32_t x2328 = 64 * x2327;
int32_t x2349 = x2325 + 2;
bool x2350 = x2349 >= 3;
bool x2351;
if (x2350) {
x2351 = x2350;
} else {
x2351 = false;
}
int32_t x2356 = x2349 - 3;
int32_t x2357 = x2356 / 1;
int32_t x2358 = x2357 + 1;
int32_t x2362 = 16384 * x2358;
int32_t x2363 = x2362 * x2358;
int32_t x2359 = x2358 * x2358;
int32_t x2360 = 256 * x2359;
int32_t x2361 = 64 * x2360;
bool x2382 = x2358 >= 1;
bool x2383;
if (x2382) {
x2383 = x2382;
} else {
x2383 = false;
}
int32_t x2388 = x2357 / 1;
int32_t x2389 = x2388 + 1;
int32_t x2393 = 65536 * x2389;
int32_t x2394 = x2393 * x2389;
int32_t x2390 = x2389 * x2389;
int32_t x2391 = 1024 * x2390;
int32_t x2392 = 64 * x2391;
bool x2407 = x2281 == 1;
bool x2408 = x2281 == x2389;
bool x2409 = x2407 || x2408;
bool x2410;
if (x2409) {
x2410 = x2409;
} else {
x2410 = false;
}
bool x2426 = x2389 >= 1;
bool x2427;
if (x2426) {
x2427 = x2426;
} else {
x2427 = false;
}
int32_t x2432 = x2388 / 1;
int32_t x2433 = x2432 + 1;
int32_t x2437 = 16384 * x2433;
int32_t x2438 = x2437 * x2433;
int32_t x2434 = x2433 * x2433;
int32_t x2435 = 256 * x2434;
int32_t x2436 = 64 * x2435;
int32_t x2457 = x2433 + 2;
bool x2458 = x2457 >= 3;
bool x2459;
if (x2458) {
x2459 = x2458;
} else {
x2459 = false;
}
int32_t x2464 = x2457 - 3;
int32_t x2465 = x2464 / 1;
int32_t x2466 = x2465 + 1;
int32_t x2470 = 16384 * x2466;
int32_t x2471 = x2470 * x2466;
int32_t x2467 = x2466 * x2466;
int32_t x2468 = 256 * x2467;
int32_t x2469 = 64 * x2468;
bool x2490 = x2466 >= 1;
bool x2491;
if (x2490) {
x2491 = x2490;
} else {
x2491 = false;
}
int32_t x2496 = x2465 / 1;
int32_t x2497 = x2496 + 1;
int32_t x2501 = 65536 * x2497;
int32_t x2502 = x2501 * x2497;
int32_t x2498 = x2497 * x2497;
int32_t x2499 = 1024 * x2498;
int32_t x2500 = 64 * x2499;
bool x2515 = x2389 == 1;
bool x2516 = x2389 == x2497;
bool x2517 = x2515 || x2516;
bool x2518;
if (x2517) {
x2518 = x2517;
} else {
x2518 = false;
}
bool x2534 = x2497 >= 1;
bool x2535;
if (x2534) {
x2535 = x2534;
} else {
x2535 = false;
}
int32_t x2540 = x2496 / 1;
int32_t x2541 = x2540 + 1;
int32_t x2545 = 16384 * x2541;
int32_t x2546 = x2545 * x2541;
int32_t x2542 = x2541 * x2541;
int32_t x2543 = 256 * x2542;
int32_t x2544 = 64 * x2543;
int32_t x2565 = x2541 + 2;
bool x2566 = x2565 >= 3;
bool x2567;
if (x2566) {
x2567 = x2566;
} else {
x2567 = false;
}
int32_t x2572 = x2565 - 3;
int32_t x2573 = x2572 / 1;
int32_t x2574 = x2573 + 1;
int32_t x2578 = 16384 * x2574;
int32_t x2579 = x2578 * x2574;
int32_t x2575 = x2574 * x2574;
int32_t x2576 = 256 * x2575;
int32_t x2577 = 64 * x2576;
bool x2598 = x2574 >= 1;
bool x2599;
if (x2598) {
x2599 = x2598;
} else {
x2599 = false;
}
int32_t x2604 = x2573 / 1;
int32_t x2605 = x2604 + 1;
int32_t x2609 = 65536 * x2605;
int32_t x2610 = x2609 * x2605;
int32_t x2606 = x2605 * x2605;
int32_t x2607 = 1024 * x2606;
int32_t x2608 = 64 * x2607;
bool x2623 = x2497 == 1;
bool x2624 = x2497 == x2605;
bool x2625 = x2623 || x2624;
bool x2626;
if (x2625) {
x2626 = x2625;
} else {
x2626 = false;
}
bool x2642 = x2605 >= 1;
bool x2643;
if (x2642) {
x2643 = x2642;
} else {
x2643 = false;
}
int32_t x2648 = x2604 / 1;
int32_t x2649 = x2648 + 1;
int32_t x2653 = 32768 * x2649;
int32_t x2654 = x2653 * x2649;
int32_t x2650 = x2649 * x2649;
int32_t x2651 = 512 * x2650;
int32_t x2652 = 64 * x2651;
int32_t x2673 = x2649 + 2;
bool x2674 = x2673 >= 3;
bool x2675;
if (x2674) {
x2675 = x2674;
} else {
x2675 = false;
}
int32_t x2680 = x2673 - 3;
int32_t x2681 = x2680 / 2;
int32_t x2682 = x2681 + 1;
int32_t x2686 = 32768 * x2682;
int32_t x2687 = x2686 * x2682;
int32_t x2683 = x2682 * x2682;
int32_t x2684 = 512 * x2683;
int32_t x2685 = 64 * x2684;
bool x2706 = x2682 >= 1;
bool x2707;
if (x2706) {
x2707 = x2706;
} else {
x2707 = false;
}
int32_t x2712 = x2681 / 1;
int32_t x2713 = x2712 + 1;
int32_t x2717 = 131072 * x2713;
int32_t x2718 = x2717 * x2713;
int32_t x2714 = x2713 * x2713;
int32_t x2715 = 2048 * x2714;
int32_t x2716 = 64 * x2715;
int32_t x2734 = x2604 / 2;
int32_t x2735 = x2734 + 1;
int32_t x2739 = 131072 * x2735;
int32_t x2740 = x2739 * x2735;
int32_t x2736 = x2735 * x2735;
int32_t x2737 = 2048 * x2736;
int32_t x2738 = 64 * x2737;
bool x2753 = x2735 == 1;
bool x2754 = x2735 == x2713;
bool x2755 = x2753 || x2754;
bool x2756;
if (x2755) {
x2756 = x2755;
} else {
x2756 = false;
}
bool x2772 = x2713 >= 1;
bool x2773;
if (x2772) {
x2773 = x2772;
} else {
x2773 = false;
}
int32_t x2778 = x2712 / 1;
int32_t x2779 = x2778 + 1;
int32_t x2783 = 32768 * x2779;
int32_t x2784 = x2783 * x2779;
int32_t x2780 = x2779 * x2779;
int32_t x2781 = 512 * x2780;
int32_t x2782 = 64 * x2781;
int32_t x2803 = x2779 + 2;
bool x2804 = x2803 >= 3;
bool x2805;
if (x2804) {
x2805 = x2804;
} else {
x2805 = false;
}
int32_t x2810 = x2803 - 3;
int32_t x2811 = x2810 / 1;
int32_t x2812 = x2811 + 1;
int32_t x2816 = 32768 * x2812;
int32_t x2817 = x2816 * x2812;
int32_t x2813 = x2812 * x2812;
int32_t x2814 = 512 * x2813;
int32_t x2815 = 64 * x2814;
bool x2836 = x2812 >= 1;
bool x2837;
if (x2836) {
x2837 = x2836;
} else {
x2837 = false;
}
int32_t x2842 = x2811 / 1;
int32_t x2843 = x2842 + 1;
int32_t x2847 = 131072 * x2843;
int32_t x2848 = x2847 * x2843;
int32_t x2844 = x2843 * x2843;
int32_t x2845 = 2048 * x2844;
int32_t x2846 = 64 * x2845;
bool x2861 = x2713 == 1;
bool x2862 = x2713 == x2843;
bool x2863 = x2861 || x2862;
bool x2864;
if (x2863) {
x2864 = x2863;
} else {
x2864 = false;
}
bool x2880 = x2843 >= 1;
bool x2881;
if (x2880) {
x2881 = x2880;
} else {
x2881 = false;
}
int32_t x2886 = x2842 / 1;
int32_t x2887 = x2886 + 1;
int32_t x2891 = 32768 * x2887;
int32_t x2892 = x2891 * x2887;
int32_t x2888 = x2887 * x2887;
int32_t x2889 = 512 * x2888;
int32_t x2890 = 64 * x2889;
int32_t x2911 = x2887 + 2;
bool x2912 = x2911 >= 3;
bool x2913;
if (x2912) {
x2913 = x2912;
} else {
x2913 = false;
}
int32_t x2918 = x2911 - 3;
int32_t x2919 = x2918 / 1;
int32_t x2920 = x2919 + 1;
int32_t x2924 = 32768 * x2920;
int32_t x2925 = x2924 * x2920;
int32_t x2921 = x2920 * x2920;
int32_t x2922 = 512 * x2921;
int32_t x2923 = 64 * x2922;
bool x2944 = x2920 >= 1;
bool x2945;
if (x2944) {
x2945 = x2944;
} else {
x2945 = false;
}
int32_t x2950 = x2919 / 1;
int32_t x2951 = x2950 + 1;
int32_t x2955 = 131072 * x2951;
int32_t x2956 = x2955 * x2951;
int32_t x2952 = x2951 * x2951;
int32_t x2953 = 2048 * x2952;
int32_t x2954 = 64 * x2953;
bool x2969 = x2843 == 1;
bool x2970 = x2843 == x2951;
bool x2971 = x2969 || x2970;
bool x2972;
if (x2971) {
x2972 = x2971;
} else {
x2972 = false;
}
bool x2988 = x2951 >= 2;
bool x2989;
if (x2988) {
x2989 = x2988;
} else {
x2989 = false;
}
int32_t x2998 = x2951 - 2;
int32_t x2999 = x2998 / 1;
int32_t x3000 = x2999 + 1;
int32_t x3004 = 131072 * x3000;
int32_t x3005 = x3004 * x3000;
int32_t x3001 = x3000 * x3000;
int32_t x3002 = 2048 * x3001;
int32_t x3003 = 64 * x3002;
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
float* x1143 = (float*)myGpuMalloc(x1142 * sizeof(float));
float* x1144 = (float*)myMalloc(1 * sizeof(float));;
x1144[0] = 0.0f;
float* x1146 = (float*)myMalloc(1 * sizeof(float));;
x1146[0] = 1.0f;

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
    64, 64, x1137, x1137));

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
    x1146, in_desc, x1134, filt_desc, x714,
    conv_desc, algo, ws_data, ws_size,
    x1144, out_desc, x1143));
};
float* x1149 = (float*)myGpuMalloc(x1140 * sizeof(float));
float* x1150 = (float*)myMalloc(1 * sizeof(float));;
x1150[0] = 0.0f;
float* x1152 = (float*)myMalloc(1 * sizeof(float));;
x1152[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1137, x1137));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1137, x1137));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1152, x1152, in_desc, x1143, out_desc, x1149, sbmv_desc, x876,
    x1011, x378, x588, 1.0E-5));
};
float* x1155 = (float*)myMalloc(1 * sizeof(float));;
x1155[0] = 0.0f;
float* x1157 = (float*)myMalloc(1 * sizeof(float));;
x1157[0] = 1.0f;
float* x1159 = (float*)myGpuMalloc(x1140 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1137, x1137));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1157, x_desc, x1149, x1155, x_desc, x1159));
};
float* x1161 = (float*)myMalloc(1 * sizeof(float));;
x1161[0] = 0.0f;
float* x1163 = (float*)myMalloc(1 * sizeof(float));;
x1163[0] = 1.0f;
float* x1173 = (float*)myGpuMalloc(x1172 * sizeof(float));

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1137, x1137) );

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1167, x1167));

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
    x1163, in_desc, x1159, x1161, out_desc, x1173));
};
if (x1176) {
} else {
assert(false && "ERROR not specified");
}
float* x1188 = (float*)myGpuMalloc(x1187 * sizeof(float));
float* x1189 = (float*)myMalloc(1 * sizeof(float));;
x1189[0] = 0.0f;
float* x1191 = (float*)myMalloc(1 * sizeof(float));;
x1191[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1167, x1167));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1182, x1182));

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
    x1191, in_desc, x1173, filt_desc, x957,
    conv_desc, algo, ws_data, ws_size,
    x1189, out_desc, x1188));
};
float* x1194 = (float*)myGpuMalloc(x1185 * sizeof(float));
float* x1195 = (float*)myMalloc(1 * sizeof(float));;
x1195[0] = 0.0f;
float* x1197 = (float*)myMalloc(1 * sizeof(float));;
x1197[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1182, x1182));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1182, x1182));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1197, x1197, in_desc, x1188, out_desc, x1194, sbmv_desc, x336,
    x417, x600, x411, 1.0E-5));
};
float* x1200 = (float*)myMalloc(1 * sizeof(float));;
x1200[0] = 0.0f;
float* x1202 = (float*)myMalloc(1 * sizeof(float));;
x1202[0] = 1.0f;
float* x1204 = (float*)myGpuMalloc(x1185 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1182, x1182));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1202, x_desc, x1194, x1200, x_desc, x1204));
};
if (x1208) {
} else {
assert(false && "ERROR not specified");
}
float* x1221 = (float*)myGpuMalloc(x1220 * sizeof(float));
float* x1222 = (float*)myMalloc(1 * sizeof(float));;
x1222[0] = 0.0f;
float* x1224 = (float*)myMalloc(1 * sizeof(float));;
x1224[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1182, x1182));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1215, x1215));

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
    x1224, in_desc, x1204, filt_desc, x528,
    conv_desc, algo, ws_data, ws_size,
    x1222, out_desc, x1221));
};
float* x1227 = (float*)myGpuMalloc(x1218 * sizeof(float));
float* x1228 = (float*)myMalloc(1 * sizeof(float));;
x1228[0] = 0.0f;
float* x1230 = (float*)myMalloc(1 * sizeof(float));;
x1230[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1215, x1215));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1215, x1215));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1230, x1230, in_desc, x1221, out_desc, x1227, sbmv_desc, x750,
    x405, x573, x732, 1.0E-5));
};
float* x1233 = (float*)myMalloc(1 * sizeof(float));;
x1233[0] = 0.0f;
float* x1235 = (float*)myMalloc(1 * sizeof(float));;
x1235[0] = 1.0f;
float* x1237 = (float*)myGpuMalloc(x1218 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1215, x1215));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1235, x_desc, x1227, x1233, x_desc, x1237));
};
if (x1240) {
} else {
assert(false && "ERROR not specified");
}
float* x1252 = (float*)myGpuMalloc(x1251 * sizeof(float));
float* x1253 = (float*)myMalloc(1 * sizeof(float));;
x1253[0] = 0.0f;
float* x1255 = (float*)myMalloc(1 * sizeof(float));;
x1255[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1215, x1215));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1246, x1246));

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
    x1255, in_desc, x1237, filt_desc, x354,
    conv_desc, algo, ws_data, ws_size,
    x1253, out_desc, x1252));
};
float* x1258 = (float*)myGpuMalloc(x1249 * sizeof(float));
float* x1259 = (float*)myMalloc(1 * sizeof(float));;
x1259[0] = 0.0f;
float* x1261 = (float*)myMalloc(1 * sizeof(float));;
x1261[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1246, x1246));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1246, x1246));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1261, x1261, in_desc, x1252, out_desc, x1258, sbmv_desc, x855,
    x636, x471, x366, 1.0E-5));
};
if (x1176) {
} else {
assert(false && "ERROR not specified");
}
float* x1271 = (float*)myGpuMalloc(x1270 * sizeof(float));
float* x1272 = (float*)myMalloc(1 * sizeof(float));;
x1272[0] = 0.0f;
float* x1274 = (float*)myMalloc(1 * sizeof(float));;
x1274[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1167, x1167));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1182, x1182));

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
    x1274, in_desc, x1173, filt_desc, x744,
    conv_desc, algo, ws_data, ws_size,
    x1272, out_desc, x1271));
};
float* x1277 = (float*)myGpuMalloc(x1268 * sizeof(float));
float* x1278 = (float*)myMalloc(1 * sizeof(float));;
x1278[0] = 0.0f;
float* x1280 = (float*)myMalloc(1 * sizeof(float));;
x1280[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1182, x1182));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1182, x1182));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1280, x1280, in_desc, x1271, out_desc, x1277, sbmv_desc, x486,
    x867, x1050, x987, 1.0E-5));
};
if (x1286) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1182) x Sym(1182), res:  x Const(64) x Const(256) x Sym(1246) x Sym(1246)");
}
float* x1291 = (float*)myMalloc(1 * sizeof(float));;
x1291[0] = 1.0f;
float* x1293 = (float*)myMalloc(1 * sizeof(float));;
x1293[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1182, x1182));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1246, x1246));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1291, bias_desc, x1277, x1293, out_desc, x1258));
};
float* x1296 = (float*)myMalloc(1 * sizeof(float));;
x1296[0] = 0.0f;
float* x1298 = (float*)myMalloc(1 * sizeof(float));;
x1298[0] = 1.0f;
float* x1300 = (float*)myGpuMalloc(x1249 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1246, x1246));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1298, x_desc, x1258, x1296, x_desc, x1300));
};
if (x1303) {
} else {
assert(false && "ERROR not specified");
}
float* x1315 = (float*)myGpuMalloc(x1314 * sizeof(float));
float* x1316 = (float*)myMalloc(1 * sizeof(float));;
x1316[0] = 0.0f;
float* x1318 = (float*)myMalloc(1 * sizeof(float));;
x1318[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1246, x1246));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1309, x1309));

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
    x1318, in_desc, x1300, filt_desc, x771,
    conv_desc, algo, ws_data, ws_size,
    x1316, out_desc, x1315));
};
float* x1321 = (float*)myGpuMalloc(x1312 * sizeof(float));
float* x1322 = (float*)myMalloc(1 * sizeof(float));;
x1322[0] = 0.0f;
float* x1324 = (float*)myMalloc(1 * sizeof(float));;
x1324[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1309, x1309));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1309, x1309));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1324, x1324, in_desc, x1315, out_desc, x1321, sbmv_desc, x684,
    x438, x288, x564, 1.0E-5));
};
float* x1327 = (float*)myMalloc(1 * sizeof(float));;
x1327[0] = 0.0f;
float* x1329 = (float*)myMalloc(1 * sizeof(float));;
x1329[0] = 1.0f;
float* x1331 = (float*)myGpuMalloc(x1312 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1309, x1309));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1329, x_desc, x1321, x1327, x_desc, x1331));
};
if (x1335) {
} else {
assert(false && "ERROR not specified");
}
float* x1348 = (float*)myGpuMalloc(x1347 * sizeof(float));
float* x1349 = (float*)myMalloc(1 * sizeof(float));;
x1349[0] = 0.0f;
float* x1351 = (float*)myMalloc(1 * sizeof(float));;
x1351[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1309, x1309));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1342, x1342));

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
    x1351, in_desc, x1331, filt_desc, x507,
    conv_desc, algo, ws_data, ws_size,
    x1349, out_desc, x1348));
};
float* x1354 = (float*)myGpuMalloc(x1345 * sizeof(float));
float* x1355 = (float*)myMalloc(1 * sizeof(float));;
x1355[0] = 0.0f;
float* x1357 = (float*)myMalloc(1 * sizeof(float));;
x1357[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1342, x1342));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1342, x1342));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1357, x1357, in_desc, x1348, out_desc, x1354, sbmv_desc, x882,
    x717, x390, x990, 1.0E-5));
};
float* x1360 = (float*)myMalloc(1 * sizeof(float));;
x1360[0] = 0.0f;
float* x1362 = (float*)myMalloc(1 * sizeof(float));;
x1362[0] = 1.0f;
float* x1364 = (float*)myGpuMalloc(x1345 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1342, x1342));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1362, x_desc, x1354, x1360, x_desc, x1364));
};
if (x1367) {
} else {
assert(false && "ERROR not specified");
}
float* x1379 = (float*)myGpuMalloc(x1378 * sizeof(float));
float* x1380 = (float*)myMalloc(1 * sizeof(float));;
x1380[0] = 0.0f;
float* x1382 = (float*)myMalloc(1 * sizeof(float));;
x1382[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1342, x1342));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1373, x1373));

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
    x1382, in_desc, x1364, filt_desc, x648,
    conv_desc, algo, ws_data, ws_size,
    x1380, out_desc, x1379));
};
float* x1385 = (float*)myGpuMalloc(x1376 * sizeof(float));
float* x1386 = (float*)myMalloc(1 * sizeof(float));;
x1386[0] = 0.0f;
float* x1388 = (float*)myMalloc(1 * sizeof(float));;
x1388[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1373, x1373));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1373, x1373));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1388, x1388, in_desc, x1379, out_desc, x1385, sbmv_desc, x432,
    x279, x531, x756, 1.0E-5));
};
if (x1394) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1246) x Sym(1246), res:  x Const(64) x Const(256) x Sym(1373) x Sym(1373)");
}
float* x1399 = (float*)myMalloc(1 * sizeof(float));;
x1399[0] = 1.0f;
float* x1401 = (float*)myMalloc(1 * sizeof(float));;
x1401[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1246, x1246));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1373, x1373));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1399, bias_desc, x1300, x1401, out_desc, x1385));
};
float* x1404 = (float*)myMalloc(1 * sizeof(float));;
x1404[0] = 0.0f;
float* x1406 = (float*)myMalloc(1 * sizeof(float));;
x1406[0] = 1.0f;
float* x1408 = (float*)myGpuMalloc(x1376 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1373, x1373));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1406, x_desc, x1385, x1404, x_desc, x1408));
};
if (x1411) {
} else {
assert(false && "ERROR not specified");
}
float* x1423 = (float*)myGpuMalloc(x1422 * sizeof(float));
float* x1424 = (float*)myMalloc(1 * sizeof(float));;
x1424[0] = 0.0f;
float* x1426 = (float*)myMalloc(1 * sizeof(float));;
x1426[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1373, x1373));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1417, x1417));

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
    x1426, in_desc, x1408, filt_desc, x708,
    conv_desc, algo, ws_data, ws_size,
    x1424, out_desc, x1423));
};
float* x1429 = (float*)myGpuMalloc(x1420 * sizeof(float));
float* x1430 = (float*)myMalloc(1 * sizeof(float));;
x1430[0] = 0.0f;
float* x1432 = (float*)myMalloc(1 * sizeof(float));;
x1432[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1417, x1417));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1417, x1417));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1432, x1432, in_desc, x1423, out_desc, x1429, sbmv_desc, x501,
    x330, x1029, x819, 1.0E-5));
};
float* x1435 = (float*)myMalloc(1 * sizeof(float));;
x1435[0] = 0.0f;
float* x1437 = (float*)myMalloc(1 * sizeof(float));;
x1437[0] = 1.0f;
float* x1439 = (float*)myGpuMalloc(x1420 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1417, x1417));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1437, x_desc, x1429, x1435, x_desc, x1439));
};
if (x1443) {
} else {
assert(false && "ERROR not specified");
}
float* x1456 = (float*)myGpuMalloc(x1455 * sizeof(float));
float* x1457 = (float*)myMalloc(1 * sizeof(float));;
x1457[0] = 0.0f;
float* x1459 = (float*)myMalloc(1 * sizeof(float));;
x1459[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1417, x1417));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1450, x1450));

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
    x1459, in_desc, x1439, filt_desc, x477,
    conv_desc, algo, ws_data, ws_size,
    x1457, out_desc, x1456));
};
float* x1462 = (float*)myGpuMalloc(x1453 * sizeof(float));
float* x1463 = (float*)myMalloc(1 * sizeof(float));;
x1463[0] = 0.0f;
float* x1465 = (float*)myMalloc(1 * sizeof(float));;
x1465[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1450, x1450));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1450, x1450));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1465, x1465, in_desc, x1456, out_desc, x1462, sbmv_desc, x474,
    x663, x795, x612, 1.0E-5));
};
float* x1468 = (float*)myMalloc(1 * sizeof(float));;
x1468[0] = 0.0f;
float* x1470 = (float*)myMalloc(1 * sizeof(float));;
x1470[0] = 1.0f;
float* x1472 = (float*)myGpuMalloc(x1453 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1450, x1450));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1470, x_desc, x1462, x1468, x_desc, x1472));
};
if (x1475) {
} else {
assert(false && "ERROR not specified");
}
float* x1487 = (float*)myGpuMalloc(x1486 * sizeof(float));
float* x1488 = (float*)myMalloc(1 * sizeof(float));;
x1488[0] = 0.0f;
float* x1490 = (float*)myMalloc(1 * sizeof(float));;
x1490[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1450, x1450));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1481, x1481));

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
    x1490, in_desc, x1472, filt_desc, x519,
    conv_desc, algo, ws_data, ws_size,
    x1488, out_desc, x1487));
};
float* x1493 = (float*)myGpuMalloc(x1484 * sizeof(float));
float* x1494 = (float*)myMalloc(1 * sizeof(float));;
x1494[0] = 0.0f;
float* x1496 = (float*)myMalloc(1 * sizeof(float));;
x1496[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1481, x1481));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1481, x1481));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1496, x1496, in_desc, x1487, out_desc, x1493, sbmv_desc, x369,
    x999, x810, x657, 1.0E-5));
};
if (x1502) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1373) x Sym(1373), res:  x Const(64) x Const(256) x Sym(1481) x Sym(1481)");
}
float* x1507 = (float*)myMalloc(1 * sizeof(float));;
x1507[0] = 1.0f;
float* x1509 = (float*)myMalloc(1 * sizeof(float));;
x1509[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1373, x1373));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1481, x1481));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1507, bias_desc, x1408, x1509, out_desc, x1493));
};
float* x1512 = (float*)myMalloc(1 * sizeof(float));;
x1512[0] = 0.0f;
float* x1514 = (float*)myMalloc(1 * sizeof(float));;
x1514[0] = 1.0f;
float* x1516 = (float*)myGpuMalloc(x1484 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1481, x1481));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1514, x_desc, x1493, x1512, x_desc, x1516));
};
if (x1519) {
} else {
assert(false && "ERROR not specified");
}
float* x1531 = (float*)myGpuMalloc(x1530 * sizeof(float));
float* x1532 = (float*)myMalloc(1 * sizeof(float));;
x1532[0] = 0.0f;
float* x1534 = (float*)myMalloc(1 * sizeof(float));;
x1534[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1481, x1481));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1525, x1525));

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
    x1534, in_desc, x1516, filt_desc, x291,
    conv_desc, algo, ws_data, ws_size,
    x1532, out_desc, x1531));
};
float* x1537 = (float*)myGpuMalloc(x1528 * sizeof(float));
float* x1538 = (float*)myMalloc(1 * sizeof(float));;
x1538[0] = 0.0f;
float* x1540 = (float*)myMalloc(1 * sizeof(float));;
x1540[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1525, x1525));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1525, x1525));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1540, x1540, in_desc, x1531, out_desc, x1537, sbmv_desc, x510,
    x774, x870, x660, 1.0E-5));
};
float* x1543 = (float*)myMalloc(1 * sizeof(float));;
x1543[0] = 0.0f;
float* x1545 = (float*)myMalloc(1 * sizeof(float));;
x1545[0] = 1.0f;
float* x1547 = (float*)myGpuMalloc(x1528 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1525, x1525));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1545, x_desc, x1537, x1543, x_desc, x1547));
};
if (x1551) {
} else {
assert(false && "ERROR not specified");
}
float* x1564 = (float*)myGpuMalloc(x1563 * sizeof(float));
float* x1565 = (float*)myMalloc(1 * sizeof(float));;
x1565[0] = 0.0f;
float* x1567 = (float*)myMalloc(1 * sizeof(float));;
x1567[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1525, x1525));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1558, x1558));

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
    x1567, in_desc, x1547, filt_desc, x339,
    conv_desc, algo, ws_data, ws_size,
    x1565, out_desc, x1564));
};
float* x1570 = (float*)myGpuMalloc(x1561 * sizeof(float));
float* x1571 = (float*)myMalloc(1 * sizeof(float));;
x1571[0] = 0.0f;
float* x1573 = (float*)myMalloc(1 * sizeof(float));;
x1573[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1558, x1558));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1558, x1558));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1573, x1573, in_desc, x1564, out_desc, x1570, sbmv_desc, x1014,
    x828, x642, x387, 1.0E-5));
};
float* x1576 = (float*)myMalloc(1 * sizeof(float));;
x1576[0] = 0.0f;
float* x1578 = (float*)myMalloc(1 * sizeof(float));;
x1578[0] = 1.0f;
float* x1580 = (float*)myGpuMalloc(x1561 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1558, x1558));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1578, x_desc, x1570, x1576, x_desc, x1580));
};
if (x1583) {
} else {
assert(false && "ERROR not specified");
}
float* x1595 = (float*)myGpuMalloc(x1594 * sizeof(float));
float* x1596 = (float*)myMalloc(1 * sizeof(float));;
x1596[0] = 0.0f;
float* x1598 = (float*)myMalloc(1 * sizeof(float));;
x1598[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1558, x1558));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1589, x1589));

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
    x1598, in_desc, x1580, filt_desc, x576,
    conv_desc, algo, ws_data, ws_size,
    x1596, out_desc, x1595));
};
float* x1601 = (float*)myGpuMalloc(x1592 * sizeof(float));
float* x1602 = (float*)myMalloc(1 * sizeof(float));;
x1602[0] = 0.0f;
float* x1604 = (float*)myMalloc(1 * sizeof(float));;
x1604[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1589, x1589));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1589, x1589));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1604, x1604, in_desc, x1595, out_desc, x1601, sbmv_desc, x693,
    x888, x705, x561, 1.0E-5));
};
if (x1519) {
} else {
assert(false && "ERROR not specified");
}
float* x1617 = (float*)myGpuMalloc(x1616 * sizeof(float));
float* x1618 = (float*)myMalloc(1 * sizeof(float));;
x1618[0] = 0.0f;
float* x1620 = (float*)myMalloc(1 * sizeof(float));;
x1620[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1481, x1481));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1611, x1611));

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
    x1620, in_desc, x1516, filt_desc, x1032,
    conv_desc, algo, ws_data, ws_size,
    x1618, out_desc, x1617));
};
float* x1623 = (float*)myGpuMalloc(x1614 * sizeof(float));
float* x1624 = (float*)myMalloc(1 * sizeof(float));;
x1624[0] = 0.0f;
float* x1626 = (float*)myMalloc(1 * sizeof(float));;
x1626[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1611, x1611));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1611, x1611));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1626, x1626, in_desc, x1617, out_desc, x1623, sbmv_desc, x879,
    x615, x384, x327, 1.0E-5));
};
if (x1632) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(1611) x Sym(1611), res:  x Const(64) x Const(512) x Sym(1589) x Sym(1589)");
}
float* x1637 = (float*)myMalloc(1 * sizeof(float));;
x1637[0] = 1.0f;
float* x1639 = (float*)myMalloc(1 * sizeof(float));;
x1639[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1611, x1611));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1589, x1589));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1637, bias_desc, x1623, x1639, out_desc, x1601));
};
float* x1642 = (float*)myMalloc(1 * sizeof(float));;
x1642[0] = 0.0f;
float* x1644 = (float*)myMalloc(1 * sizeof(float));;
x1644[0] = 1.0f;
float* x1646 = (float*)myGpuMalloc(x1592 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1589, x1589));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1644, x_desc, x1601, x1642, x_desc, x1646));
};
if (x1649) {
} else {
assert(false && "ERROR not specified");
}
float* x1661 = (float*)myGpuMalloc(x1660 * sizeof(float));
float* x1662 = (float*)myMalloc(1 * sizeof(float));;
x1662[0] = 0.0f;
float* x1664 = (float*)myMalloc(1 * sizeof(float));;
x1664[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1589, x1589));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1655, x1655));

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
    x1664, in_desc, x1646, filt_desc, x1026,
    conv_desc, algo, ws_data, ws_size,
    x1662, out_desc, x1661));
};
float* x1667 = (float*)myGpuMalloc(x1658 * sizeof(float));
float* x1668 = (float*)myMalloc(1 * sizeof(float));;
x1668[0] = 0.0f;
float* x1670 = (float*)myMalloc(1 * sizeof(float));;
x1670[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1655, x1655));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1655, x1655));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1670, x1670, in_desc, x1661, out_desc, x1667, sbmv_desc, x924,
    x309, x558, x789, 1.0E-5));
};
float* x1673 = (float*)myMalloc(1 * sizeof(float));;
x1673[0] = 0.0f;
float* x1675 = (float*)myMalloc(1 * sizeof(float));;
x1675[0] = 1.0f;
float* x1677 = (float*)myGpuMalloc(x1658 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1655, x1655));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1675, x_desc, x1667, x1673, x_desc, x1677));
};
if (x1681) {
} else {
assert(false && "ERROR not specified");
}
float* x1694 = (float*)myGpuMalloc(x1693 * sizeof(float));
float* x1695 = (float*)myMalloc(1 * sizeof(float));;
x1695[0] = 0.0f;
float* x1697 = (float*)myMalloc(1 * sizeof(float));;
x1697[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1655, x1655));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1688, x1688));

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
    x1697, in_desc, x1677, filt_desc, x963,
    conv_desc, algo, ws_data, ws_size,
    x1695, out_desc, x1694));
};
float* x1700 = (float*)myGpuMalloc(x1691 * sizeof(float));
float* x1701 = (float*)myMalloc(1 * sizeof(float));;
x1701[0] = 0.0f;
float* x1703 = (float*)myMalloc(1 * sizeof(float));;
x1703[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1688, x1688));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1688, x1688));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1703, x1703, in_desc, x1694, out_desc, x1700, sbmv_desc, x282,
    x543, x363, x933, 1.0E-5));
};
float* x1706 = (float*)myMalloc(1 * sizeof(float));;
x1706[0] = 0.0f;
float* x1708 = (float*)myMalloc(1 * sizeof(float));;
x1708[0] = 1.0f;
float* x1710 = (float*)myGpuMalloc(x1691 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1688, x1688));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1708, x_desc, x1700, x1706, x_desc, x1710));
};
if (x1713) {
} else {
assert(false && "ERROR not specified");
}
float* x1725 = (float*)myGpuMalloc(x1724 * sizeof(float));
float* x1726 = (float*)myMalloc(1 * sizeof(float));;
x1726[0] = 0.0f;
float* x1728 = (float*)myMalloc(1 * sizeof(float));;
x1728[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1688, x1688));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1719, x1719));

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
    x1728, in_desc, x1710, filt_desc, x591,
    conv_desc, algo, ws_data, ws_size,
    x1726, out_desc, x1725));
};
float* x1731 = (float*)myGpuMalloc(x1722 * sizeof(float));
float* x1732 = (float*)myMalloc(1 * sizeof(float));;
x1732[0] = 0.0f;
float* x1734 = (float*)myMalloc(1 * sizeof(float));;
x1734[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1719, x1719));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1719, x1719));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1734, x1734, in_desc, x1725, out_desc, x1731, sbmv_desc, x414,
    x996, x699, x522, 1.0E-5));
};
if (x1740) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(1589) x Sym(1589), res:  x Const(64) x Const(512) x Sym(1719) x Sym(1719)");
}
float* x1745 = (float*)myMalloc(1 * sizeof(float));;
x1745[0] = 1.0f;
float* x1747 = (float*)myMalloc(1 * sizeof(float));;
x1747[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1589, x1589));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1719, x1719));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1745, bias_desc, x1646, x1747, out_desc, x1731));
};
float* x1750 = (float*)myMalloc(1 * sizeof(float));;
x1750[0] = 0.0f;
float* x1752 = (float*)myMalloc(1 * sizeof(float));;
x1752[0] = 1.0f;
float* x1754 = (float*)myGpuMalloc(x1722 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1719, x1719));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1752, x_desc, x1731, x1750, x_desc, x1754));
};
if (x1757) {
} else {
assert(false && "ERROR not specified");
}
float* x1769 = (float*)myGpuMalloc(x1768 * sizeof(float));
float* x1770 = (float*)myMalloc(1 * sizeof(float));;
x1770[0] = 0.0f;
float* x1772 = (float*)myMalloc(1 * sizeof(float));;
x1772[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1719, x1719));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1763, x1763));

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
    x1772, in_desc, x1754, filt_desc, x846,
    conv_desc, algo, ws_data, ws_size,
    x1770, out_desc, x1769));
};
float* x1775 = (float*)myGpuMalloc(x1766 * sizeof(float));
float* x1776 = (float*)myMalloc(1 * sizeof(float));;
x1776[0] = 0.0f;
float* x1778 = (float*)myMalloc(1 * sizeof(float));;
x1778[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1763, x1763));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1763, x1763));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1778, x1778, in_desc, x1769, out_desc, x1775, sbmv_desc, x393,
    x768, x594, x285, 1.0E-5));
};
float* x1781 = (float*)myMalloc(1 * sizeof(float));;
x1781[0] = 0.0f;
float* x1783 = (float*)myMalloc(1 * sizeof(float));;
x1783[0] = 1.0f;
float* x1785 = (float*)myGpuMalloc(x1766 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1763, x1763));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1783, x_desc, x1775, x1781, x_desc, x1785));
};
if (x1789) {
} else {
assert(false && "ERROR not specified");
}
float* x1802 = (float*)myGpuMalloc(x1801 * sizeof(float));
float* x1803 = (float*)myMalloc(1 * sizeof(float));;
x1803[0] = 0.0f;
float* x1805 = (float*)myMalloc(1 * sizeof(float));;
x1805[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1763, x1763));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1796, x1796));

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
    x1805, in_desc, x1785, filt_desc, x831,
    conv_desc, algo, ws_data, ws_size,
    x1803, out_desc, x1802));
};
float* x1808 = (float*)myGpuMalloc(x1799 * sizeof(float));
float* x1809 = (float*)myMalloc(1 * sizeof(float));;
x1809[0] = 0.0f;
float* x1811 = (float*)myMalloc(1 * sizeof(float));;
x1811[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1796, x1796));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1796, x1796));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1811, x1811, in_desc, x1802, out_desc, x1808, sbmv_desc, x639,
    x441, x909, x1056, 1.0E-5));
};
float* x1814 = (float*)myMalloc(1 * sizeof(float));;
x1814[0] = 0.0f;
float* x1816 = (float*)myMalloc(1 * sizeof(float));;
x1816[0] = 1.0f;
float* x1818 = (float*)myGpuMalloc(x1799 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1796, x1796));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1816, x_desc, x1808, x1814, x_desc, x1818));
};
if (x1821) {
} else {
assert(false && "ERROR not specified");
}
float* x1833 = (float*)myGpuMalloc(x1832 * sizeof(float));
float* x1834 = (float*)myMalloc(1 * sizeof(float));;
x1834[0] = 0.0f;
float* x1836 = (float*)myMalloc(1 * sizeof(float));;
x1836[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1796, x1796));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1827, x1827));

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
    x1836, in_desc, x1818, filt_desc, x381,
    conv_desc, algo, ws_data, ws_size,
    x1834, out_desc, x1833));
};
float* x1839 = (float*)myGpuMalloc(x1830 * sizeof(float));
float* x1840 = (float*)myMalloc(1 * sizeof(float));;
x1840[0] = 0.0f;
float* x1842 = (float*)myMalloc(1 * sizeof(float));;
x1842[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1827, x1827));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1827, x1827));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1842, x1842, in_desc, x1833, out_desc, x1839, sbmv_desc, x759,
    x504, x333, x927, 1.0E-5));
};
if (x1848) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(1719) x Sym(1719), res:  x Const(64) x Const(512) x Sym(1827) x Sym(1827)");
}
float* x1853 = (float*)myMalloc(1 * sizeof(float));;
x1853[0] = 1.0f;
float* x1855 = (float*)myMalloc(1 * sizeof(float));;
x1855[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1719, x1719));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1827, x1827));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1853, bias_desc, x1754, x1855, out_desc, x1839));
};
float* x1858 = (float*)myMalloc(1 * sizeof(float));;
x1858[0] = 0.0f;
float* x1860 = (float*)myMalloc(1 * sizeof(float));;
x1860[0] = 1.0f;
float* x1862 = (float*)myGpuMalloc(x1830 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1827, x1827));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1860, x_desc, x1839, x1858, x_desc, x1862));
};
if (x1865) {
} else {
assert(false && "ERROR not specified");
}
float* x1877 = (float*)myGpuMalloc(x1876 * sizeof(float));
float* x1878 = (float*)myMalloc(1 * sizeof(float));;
x1878[0] = 0.0f;
float* x1880 = (float*)myMalloc(1 * sizeof(float));;
x1880[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1827, x1827));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1871, x1871));

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
    x1880, in_desc, x1862, filt_desc, x654,
    conv_desc, algo, ws_data, ws_size,
    x1878, out_desc, x1877));
};
float* x1883 = (float*)myGpuMalloc(x1874 * sizeof(float));
float* x1884 = (float*)myMalloc(1 * sizeof(float));;
x1884[0] = 0.0f;
float* x1886 = (float*)myMalloc(1 * sizeof(float));;
x1886[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1871, x1871));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1871, x1871));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1886, x1886, in_desc, x1877, out_desc, x1883, sbmv_desc, x375,
    x984, x966, x1041, 1.0E-5));
};
float* x1889 = (float*)myMalloc(1 * sizeof(float));;
x1889[0] = 0.0f;
float* x1891 = (float*)myMalloc(1 * sizeof(float));;
x1891[0] = 1.0f;
float* x1893 = (float*)myGpuMalloc(x1874 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1871, x1871));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1891, x_desc, x1883, x1889, x_desc, x1893));
};
if (x1897) {
} else {
assert(false && "ERROR not specified");
}
float* x1910 = (float*)myGpuMalloc(x1909 * sizeof(float));
float* x1911 = (float*)myMalloc(1 * sizeof(float));;
x1911[0] = 0.0f;
float* x1913 = (float*)myMalloc(1 * sizeof(float));;
x1913[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1871, x1871));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1904, x1904));

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
    x1913, in_desc, x1893, filt_desc, x753,
    conv_desc, algo, ws_data, ws_size,
    x1911, out_desc, x1910));
};
float* x1916 = (float*)myGpuMalloc(x1907 * sizeof(float));
float* x1917 = (float*)myMalloc(1 * sizeof(float));;
x1917[0] = 0.0f;
float* x1919 = (float*)myMalloc(1 * sizeof(float));;
x1919[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1904, x1904));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1904, x1904));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1919, x1919, in_desc, x1910, out_desc, x1916, sbmv_desc, x495,
    x372, x1062, x702, 1.0E-5));
};
float* x1922 = (float*)myMalloc(1 * sizeof(float));;
x1922[0] = 0.0f;
float* x1924 = (float*)myMalloc(1 * sizeof(float));;
x1924[0] = 1.0f;
float* x1926 = (float*)myGpuMalloc(x1907 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1904, x1904));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1924, x_desc, x1916, x1922, x_desc, x1926));
};
if (x1929) {
} else {
assert(false && "ERROR not specified");
}
float* x1941 = (float*)myGpuMalloc(x1940 * sizeof(float));
float* x1942 = (float*)myMalloc(1 * sizeof(float));;
x1942[0] = 0.0f;
float* x1944 = (float*)myMalloc(1 * sizeof(float));;
x1944[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1904, x1904));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1935, x1935));

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
    x1944, in_desc, x1926, filt_desc, x423,
    conv_desc, algo, ws_data, ws_size,
    x1942, out_desc, x1941));
};
float* x1947 = (float*)myGpuMalloc(x1938 * sizeof(float));
float* x1948 = (float*)myMalloc(1 * sizeof(float));;
x1948[0] = 0.0f;
float* x1950 = (float*)myMalloc(1 * sizeof(float));;
x1950[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1935, x1935));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1935, x1935));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1950, x1950, in_desc, x1941, out_desc, x1947, sbmv_desc, x726,
    x420, x315, x960, 1.0E-5));
};
if (x1956) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(1827) x Sym(1827), res:  x Const(64) x Const(512) x Sym(1935) x Sym(1935)");
}
float* x1961 = (float*)myMalloc(1 * sizeof(float));;
x1961[0] = 1.0f;
float* x1963 = (float*)myMalloc(1 * sizeof(float));;
x1963[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1827, x1827));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1935, x1935));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1961, bias_desc, x1862, x1963, out_desc, x1947));
};
float* x1966 = (float*)myMalloc(1 * sizeof(float));;
x1966[0] = 0.0f;
float* x1968 = (float*)myMalloc(1 * sizeof(float));;
x1968[0] = 1.0f;
float* x1970 = (float*)myGpuMalloc(x1938 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1935, x1935));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1968, x_desc, x1947, x1966, x_desc, x1970));
};
if (x1973) {
} else {
assert(false && "ERROR not specified");
}
float* x1985 = (float*)myGpuMalloc(x1984 * sizeof(float));
float* x1986 = (float*)myMalloc(1 * sizeof(float));;
x1986[0] = 0.0f;
float* x1988 = (float*)myMalloc(1 * sizeof(float));;
x1988[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1935, x1935));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1979, x1979));

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
    x1988, in_desc, x1970, filt_desc, x798,
    conv_desc, algo, ws_data, ws_size,
    x1986, out_desc, x1985));
};
float* x1991 = (float*)myGpuMalloc(x1982 * sizeof(float));
float* x1992 = (float*)myMalloc(1 * sizeof(float));;
x1992[0] = 0.0f;
float* x1994 = (float*)myMalloc(1 * sizeof(float));;
x1994[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1979, x1979));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1979, x1979));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1994, x1994, in_desc, x1985, out_desc, x1991, sbmv_desc, x1068,
    x321, x651, x852, 1.0E-5));
};
float* x1997 = (float*)myMalloc(1 * sizeof(float));;
x1997[0] = 0.0f;
float* x1999 = (float*)myMalloc(1 * sizeof(float));;
x1999[0] = 1.0f;
float* x2001 = (float*)myGpuMalloc(x1982 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1979, x1979));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1999, x_desc, x1991, x1997, x_desc, x2001));
};
if (x2005) {
} else {
assert(false && "ERROR not specified");
}
float* x2018 = (float*)myGpuMalloc(x2017 * sizeof(float));
float* x2019 = (float*)myMalloc(1 * sizeof(float));;
x2019[0] = 0.0f;
float* x2021 = (float*)myMalloc(1 * sizeof(float));;
x2021[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1979, x1979));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2012, x2012));

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
    x2021, in_desc, x2001, filt_desc, x783,
    conv_desc, algo, ws_data, ws_size,
    x2019, out_desc, x2018));
};
float* x2024 = (float*)myGpuMalloc(x2015 * sizeof(float));
float* x2025 = (float*)myMalloc(1 * sizeof(float));;
x2025[0] = 0.0f;
float* x2027 = (float*)myMalloc(1 * sizeof(float));;
x2027[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2012, x2012));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2012, x2012));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2027, x2027, in_desc, x2018, out_desc, x2024, sbmv_desc, x582,
    x306, x945, x555, 1.0E-5));
};
float* x2030 = (float*)myMalloc(1 * sizeof(float));;
x2030[0] = 0.0f;
float* x2032 = (float*)myMalloc(1 * sizeof(float));;
x2032[0] = 1.0f;
float* x2034 = (float*)myGpuMalloc(x2015 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2012, x2012));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2032, x_desc, x2024, x2030, x_desc, x2034));
};
if (x2037) {
} else {
assert(false && "ERROR not specified");
}
float* x2049 = (float*)myGpuMalloc(x2048 * sizeof(float));
float* x2050 = (float*)myMalloc(1 * sizeof(float));;
x2050[0] = 0.0f;
float* x2052 = (float*)myMalloc(1 * sizeof(float));;
x2052[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2012, x2012));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2043, x2043));

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
    x2052, in_desc, x2034, filt_desc, x1065,
    conv_desc, algo, ws_data, ws_size,
    x2050, out_desc, x2049));
};
float* x2055 = (float*)myGpuMalloc(x2046 * sizeof(float));
float* x2056 = (float*)myMalloc(1 * sizeof(float));;
x2056[0] = 0.0f;
float* x2058 = (float*)myMalloc(1 * sizeof(float));;
x2058[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2043, x2043));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2043, x2043));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2058, x2058, in_desc, x2049, out_desc, x2055, sbmv_desc, x312,
    x609, x906, x1059, 1.0E-5));
};
if (x1973) {
} else {
assert(false && "ERROR not specified");
}
float* x2071 = (float*)myGpuMalloc(x2070 * sizeof(float));
float* x2072 = (float*)myMalloc(1 * sizeof(float));;
x2072[0] = 0.0f;
float* x2074 = (float*)myMalloc(1 * sizeof(float));;
x2074[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1935, x1935));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2065, x2065));

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
    x2074, in_desc, x1970, filt_desc, x483,
    conv_desc, algo, ws_data, ws_size,
    x2072, out_desc, x2071));
};
float* x2077 = (float*)myGpuMalloc(x2068 * sizeof(float));
float* x2078 = (float*)myMalloc(1 * sizeof(float));;
x2078[0] = 0.0f;
float* x2080 = (float*)myMalloc(1 * sizeof(float));;
x2080[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2065, x2065));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2065, x2065));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2080, x2080, in_desc, x2071, out_desc, x2077, sbmv_desc, x345,
    x918, x516, x891, 1.0E-5));
};
if (x2086) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2065) x Sym(2065), res:  x Const(64) x Const(1024) x Sym(2043) x Sym(2043)");
}
float* x2091 = (float*)myMalloc(1 * sizeof(float));;
x2091[0] = 1.0f;
float* x2093 = (float*)myMalloc(1 * sizeof(float));;
x2093[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2065, x2065));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2043, x2043));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2091, bias_desc, x2077, x2093, out_desc, x2055));
};
float* x2096 = (float*)myMalloc(1 * sizeof(float));;
x2096[0] = 0.0f;
float* x2098 = (float*)myMalloc(1 * sizeof(float));;
x2098[0] = 1.0f;
float* x2100 = (float*)myGpuMalloc(x2046 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2043, x2043));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2098, x_desc, x2055, x2096, x_desc, x2100));
};
if (x2103) {
} else {
assert(false && "ERROR not specified");
}
float* x2115 = (float*)myGpuMalloc(x2114 * sizeof(float));
float* x2116 = (float*)myMalloc(1 * sizeof(float));;
x2116[0] = 0.0f;
float* x2118 = (float*)myMalloc(1 * sizeof(float));;
x2118[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2043, x2043));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2109, x2109));

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
    x2118, in_desc, x2100, filt_desc, x297,
    conv_desc, algo, ws_data, ws_size,
    x2116, out_desc, x2115));
};
float* x2121 = (float*)myGpuMalloc(x2112 * sizeof(float));
float* x2122 = (float*)myMalloc(1 * sizeof(float));;
x2122[0] = 0.0f;
float* x2124 = (float*)myMalloc(1 * sizeof(float));;
x2124[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2109, x2109));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2109, x2109));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2124, x2124, in_desc, x2115, out_desc, x2121, sbmv_desc, x348,
    x915, x1035, x729, 1.0E-5));
};
float* x2127 = (float*)myMalloc(1 * sizeof(float));;
x2127[0] = 0.0f;
float* x2129 = (float*)myMalloc(1 * sizeof(float));;
x2129[0] = 1.0f;
float* x2131 = (float*)myGpuMalloc(x2112 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2109, x2109));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2129, x_desc, x2121, x2127, x_desc, x2131));
};
if (x2135) {
} else {
assert(false && "ERROR not specified");
}
float* x2148 = (float*)myGpuMalloc(x2147 * sizeof(float));
float* x2149 = (float*)myMalloc(1 * sizeof(float));;
x2149[0] = 0.0f;
float* x2151 = (float*)myMalloc(1 * sizeof(float));;
x2151[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2109, x2109));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2142, x2142));

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
    x2151, in_desc, x2131, filt_desc, x351,
    conv_desc, algo, ws_data, ws_size,
    x2149, out_desc, x2148));
};
float* x2154 = (float*)myGpuMalloc(x2145 * sizeof(float));
float* x2155 = (float*)myMalloc(1 * sizeof(float));;
x2155[0] = 0.0f;
float* x2157 = (float*)myMalloc(1 * sizeof(float));;
x2157[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2142, x2142));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2142, x2142));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2157, x2157, in_desc, x2148, out_desc, x2154, sbmv_desc, x1071,
    x546, x858, x969, 1.0E-5));
};
float* x2160 = (float*)myMalloc(1 * sizeof(float));;
x2160[0] = 0.0f;
float* x2162 = (float*)myMalloc(1 * sizeof(float));;
x2162[0] = 1.0f;
float* x2164 = (float*)myGpuMalloc(x2145 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2142, x2142));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2162, x_desc, x2154, x2160, x_desc, x2164));
};
if (x2167) {
} else {
assert(false && "ERROR not specified");
}
float* x2179 = (float*)myGpuMalloc(x2178 * sizeof(float));
float* x2180 = (float*)myMalloc(1 * sizeof(float));;
x2180[0] = 0.0f;
float* x2182 = (float*)myMalloc(1 * sizeof(float));;
x2182[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2142, x2142));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2173, x2173));

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
    x2182, in_desc, x2164, filt_desc, x426,
    conv_desc, algo, ws_data, ws_size,
    x2180, out_desc, x2179));
};
float* x2185 = (float*)myGpuMalloc(x2176 * sizeof(float));
float* x2186 = (float*)myMalloc(1 * sizeof(float));;
x2186[0] = 0.0f;
float* x2188 = (float*)myMalloc(1 * sizeof(float));;
x2188[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2173, x2173));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2173, x2173));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2188, x2188, in_desc, x2179, out_desc, x2185, sbmv_desc, x318,
    x954, x804, x687, 1.0E-5));
};
if (x2194) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2043) x Sym(2043), res:  x Const(64) x Const(1024) x Sym(2173) x Sym(2173)");
}
float* x2199 = (float*)myMalloc(1 * sizeof(float));;
x2199[0] = 1.0f;
float* x2201 = (float*)myMalloc(1 * sizeof(float));;
x2201[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2043, x2043));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2173, x2173));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2199, bias_desc, x2100, x2201, out_desc, x2185));
};
float* x2204 = (float*)myMalloc(1 * sizeof(float));;
x2204[0] = 0.0f;
float* x2206 = (float*)myMalloc(1 * sizeof(float));;
x2206[0] = 1.0f;
float* x2208 = (float*)myGpuMalloc(x2176 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2173, x2173));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2206, x_desc, x2185, x2204, x_desc, x2208));
};
if (x2211) {
} else {
assert(false && "ERROR not specified");
}
float* x2223 = (float*)myGpuMalloc(x2222 * sizeof(float));
float* x2224 = (float*)myMalloc(1 * sizeof(float));;
x2224[0] = 0.0f;
float* x2226 = (float*)myMalloc(1 * sizeof(float));;
x2226[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2173, x2173));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2217, x2217));

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
    x2226, in_desc, x2208, filt_desc, x912,
    conv_desc, algo, ws_data, ws_size,
    x2224, out_desc, x2223));
};
float* x2229 = (float*)myGpuMalloc(x2220 * sizeof(float));
float* x2230 = (float*)myMalloc(1 * sizeof(float));;
x2230[0] = 0.0f;
float* x2232 = (float*)myMalloc(1 * sizeof(float));;
x2232[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2217, x2217));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2217, x2217));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2232, x2232, in_desc, x2223, out_desc, x2229, sbmv_desc, x645,
    x849, x792, x780, 1.0E-5));
};
float* x2235 = (float*)myMalloc(1 * sizeof(float));;
x2235[0] = 0.0f;
float* x2237 = (float*)myMalloc(1 * sizeof(float));;
x2237[0] = 1.0f;
float* x2239 = (float*)myGpuMalloc(x2220 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2217, x2217));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2237, x_desc, x2229, x2235, x_desc, x2239));
};
if (x2243) {
} else {
assert(false && "ERROR not specified");
}
float* x2256 = (float*)myGpuMalloc(x2255 * sizeof(float));
float* x2257 = (float*)myMalloc(1 * sizeof(float));;
x2257[0] = 0.0f;
float* x2259 = (float*)myMalloc(1 * sizeof(float));;
x2259[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2217, x2217));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2250, x2250));

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
    x2259, in_desc, x2239, filt_desc, x300,
    conv_desc, algo, ws_data, ws_size,
    x2257, out_desc, x2256));
};
float* x2262 = (float*)myGpuMalloc(x2253 * sizeof(float));
float* x2263 = (float*)myMalloc(1 * sizeof(float));;
x2263[0] = 0.0f;
float* x2265 = (float*)myMalloc(1 * sizeof(float));;
x2265[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2250, x2250));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2250, x2250));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2265, x2265, in_desc, x2256, out_desc, x2262, sbmv_desc, x942,
    x834, x630, x447, 1.0E-5));
};
float* x2268 = (float*)myMalloc(1 * sizeof(float));;
x2268[0] = 0.0f;
float* x2270 = (float*)myMalloc(1 * sizeof(float));;
x2270[0] = 1.0f;
float* x2272 = (float*)myGpuMalloc(x2253 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2250, x2250));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2270, x_desc, x2262, x2268, x_desc, x2272));
};
if (x2275) {
} else {
assert(false && "ERROR not specified");
}
float* x2287 = (float*)myGpuMalloc(x2286 * sizeof(float));
float* x2288 = (float*)myMalloc(1 * sizeof(float));;
x2288[0] = 0.0f;
float* x2290 = (float*)myMalloc(1 * sizeof(float));;
x2290[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2250, x2250));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2281, x2281));

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
    x2290, in_desc, x2272, filt_desc, x606,
    conv_desc, algo, ws_data, ws_size,
    x2288, out_desc, x2287));
};
float* x2293 = (float*)myGpuMalloc(x2284 * sizeof(float));
float* x2294 = (float*)myMalloc(1 * sizeof(float));;
x2294[0] = 0.0f;
float* x2296 = (float*)myMalloc(1 * sizeof(float));;
x2296[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2281, x2281));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2281, x2281));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2296, x2296, in_desc, x2287, out_desc, x2293, sbmv_desc, x1047,
    x429, x678, x822, 1.0E-5));
};
if (x2302) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2173) x Sym(2173), res:  x Const(64) x Const(1024) x Sym(2281) x Sym(2281)");
}
float* x2307 = (float*)myMalloc(1 * sizeof(float));;
x2307[0] = 1.0f;
float* x2309 = (float*)myMalloc(1 * sizeof(float));;
x2309[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2173, x2173));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2281, x2281));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2307, bias_desc, x2208, x2309, out_desc, x2293));
};
float* x2312 = (float*)myMalloc(1 * sizeof(float));;
x2312[0] = 0.0f;
float* x2314 = (float*)myMalloc(1 * sizeof(float));;
x2314[0] = 1.0f;
float* x2316 = (float*)myGpuMalloc(x2284 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2281, x2281));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2314, x_desc, x2293, x2312, x_desc, x2316));
};
if (x2319) {
} else {
assert(false && "ERROR not specified");
}
float* x2331 = (float*)myGpuMalloc(x2330 * sizeof(float));
float* x2332 = (float*)myMalloc(1 * sizeof(float));;
x2332[0] = 0.0f;
float* x2334 = (float*)myMalloc(1 * sizeof(float));;
x2334[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2281, x2281));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2325, x2325));

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
    x2334, in_desc, x2316, filt_desc, x276,
    conv_desc, algo, ws_data, ws_size,
    x2332, out_desc, x2331));
};
float* x2337 = (float*)myGpuMalloc(x2328 * sizeof(float));
float* x2338 = (float*)myMalloc(1 * sizeof(float));;
x2338[0] = 0.0f;
float* x2340 = (float*)myMalloc(1 * sizeof(float));;
x2340[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2325, x2325));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2325, x2325));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2340, x2340, in_desc, x2331, out_desc, x2337, sbmv_desc, x534,
    x981, x747, x552, 1.0E-5));
};
float* x2343 = (float*)myMalloc(1 * sizeof(float));;
x2343[0] = 0.0f;
float* x2345 = (float*)myMalloc(1 * sizeof(float));;
x2345[0] = 1.0f;
float* x2347 = (float*)myGpuMalloc(x2328 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2325, x2325));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2345, x_desc, x2337, x2343, x_desc, x2347));
};
if (x2351) {
} else {
assert(false && "ERROR not specified");
}
float* x2364 = (float*)myGpuMalloc(x2363 * sizeof(float));
float* x2365 = (float*)myMalloc(1 * sizeof(float));;
x2365[0] = 0.0f;
float* x2367 = (float*)myMalloc(1 * sizeof(float));;
x2367[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2325, x2325));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2358, x2358));

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
    x2367, in_desc, x2347, filt_desc, x1005,
    conv_desc, algo, ws_data, ws_size,
    x2365, out_desc, x2364));
};
float* x2370 = (float*)myGpuMalloc(x2361 * sizeof(float));
float* x2371 = (float*)myMalloc(1 * sizeof(float));;
x2371[0] = 0.0f;
float* x2373 = (float*)myMalloc(1 * sizeof(float));;
x2373[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2358, x2358));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2358, x2358));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2373, x2373, in_desc, x2364, out_desc, x2370, sbmv_desc, x480,
    x666, x816, x948, 1.0E-5));
};
float* x2376 = (float*)myMalloc(1 * sizeof(float));;
x2376[0] = 0.0f;
float* x2378 = (float*)myMalloc(1 * sizeof(float));;
x2378[0] = 1.0f;
float* x2380 = (float*)myGpuMalloc(x2361 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2358, x2358));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2378, x_desc, x2370, x2376, x_desc, x2380));
};
if (x2383) {
} else {
assert(false && "ERROR not specified");
}
float* x2395 = (float*)myGpuMalloc(x2394 * sizeof(float));
float* x2396 = (float*)myMalloc(1 * sizeof(float));;
x2396[0] = 0.0f;
float* x2398 = (float*)myMalloc(1 * sizeof(float));;
x2398[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2358, x2358));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2389, x2389));

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
    x2398, in_desc, x2380, filt_desc, x525,
    conv_desc, algo, ws_data, ws_size,
    x2396, out_desc, x2395));
};
float* x2401 = (float*)myGpuMalloc(x2392 * sizeof(float));
float* x2402 = (float*)myMalloc(1 * sizeof(float));;
x2402[0] = 0.0f;
float* x2404 = (float*)myMalloc(1 * sizeof(float));;
x2404[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2389, x2389));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2389, x2389));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2404, x2404, in_desc, x2395, out_desc, x2401, sbmv_desc, x972,
    x696, x951, x741, 1.0E-5));
};
if (x2410) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2281) x Sym(2281), res:  x Const(64) x Const(1024) x Sym(2389) x Sym(2389)");
}
float* x2415 = (float*)myMalloc(1 * sizeof(float));;
x2415[0] = 1.0f;
float* x2417 = (float*)myMalloc(1 * sizeof(float));;
x2417[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2281, x2281));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2389, x2389));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2415, bias_desc, x2316, x2417, out_desc, x2401));
};
float* x2420 = (float*)myMalloc(1 * sizeof(float));;
x2420[0] = 0.0f;
float* x2422 = (float*)myMalloc(1 * sizeof(float));;
x2422[0] = 1.0f;
float* x2424 = (float*)myGpuMalloc(x2392 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2389, x2389));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2422, x_desc, x2401, x2420, x_desc, x2424));
};
if (x2427) {
} else {
assert(false && "ERROR not specified");
}
float* x2439 = (float*)myGpuMalloc(x2438 * sizeof(float));
float* x2440 = (float*)myMalloc(1 * sizeof(float));;
x2440[0] = 0.0f;
float* x2442 = (float*)myMalloc(1 * sizeof(float));;
x2442[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2389, x2389));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2433, x2433));

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
    x2442, in_desc, x2424, filt_desc, x324,
    conv_desc, algo, ws_data, ws_size,
    x2440, out_desc, x2439));
};
float* x2445 = (float*)myGpuMalloc(x2436 * sizeof(float));
float* x2446 = (float*)myMalloc(1 * sizeof(float));;
x2446[0] = 0.0f;
float* x2448 = (float*)myMalloc(1 * sizeof(float));;
x2448[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2433, x2433));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2433, x2433));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2448, x2448, in_desc, x2439, out_desc, x2445, sbmv_desc, x489,
    x813, x1020, x465, 1.0E-5));
};
float* x2451 = (float*)myMalloc(1 * sizeof(float));;
x2451[0] = 0.0f;
float* x2453 = (float*)myMalloc(1 * sizeof(float));;
x2453[0] = 1.0f;
float* x2455 = (float*)myGpuMalloc(x2436 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2433, x2433));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2453, x_desc, x2445, x2451, x_desc, x2455));
};
if (x2459) {
} else {
assert(false && "ERROR not specified");
}
float* x2472 = (float*)myGpuMalloc(x2471 * sizeof(float));
float* x2473 = (float*)myMalloc(1 * sizeof(float));;
x2473[0] = 0.0f;
float* x2475 = (float*)myMalloc(1 * sizeof(float));;
x2475[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2433, x2433));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2466, x2466));

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
    x2475, in_desc, x2455, filt_desc, x1044,
    conv_desc, algo, ws_data, ws_size,
    x2473, out_desc, x2472));
};
float* x2478 = (float*)myGpuMalloc(x2469 * sizeof(float));
float* x2479 = (float*)myMalloc(1 * sizeof(float));;
x2479[0] = 0.0f;
float* x2481 = (float*)myMalloc(1 * sizeof(float));;
x2481[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2466, x2466));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2466, x2466));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2481, x2481, in_desc, x2472, out_desc, x2478, sbmv_desc, x762,
    x585, x1008, x570, 1.0E-5));
};
float* x2484 = (float*)myMalloc(1 * sizeof(float));;
x2484[0] = 0.0f;
float* x2486 = (float*)myMalloc(1 * sizeof(float));;
x2486[0] = 1.0f;
float* x2488 = (float*)myGpuMalloc(x2469 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2466, x2466));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2486, x_desc, x2478, x2484, x_desc, x2488));
};
if (x2491) {
} else {
assert(false && "ERROR not specified");
}
float* x2503 = (float*)myGpuMalloc(x2502 * sizeof(float));
float* x2504 = (float*)myMalloc(1 * sizeof(float));;
x2504[0] = 0.0f;
float* x2506 = (float*)myMalloc(1 * sizeof(float));;
x2506[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2466, x2466));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2497, x2497));

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
    x2506, in_desc, x2488, filt_desc, x921,
    conv_desc, algo, ws_data, ws_size,
    x2504, out_desc, x2503));
};
float* x2509 = (float*)myGpuMalloc(x2500 * sizeof(float));
float* x2510 = (float*)myMalloc(1 * sizeof(float));;
x2510[0] = 0.0f;
float* x2512 = (float*)myMalloc(1 * sizeof(float));;
x2512[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2497, x2497));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2497, x2497));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2512, x2512, in_desc, x2503, out_desc, x2509, sbmv_desc, x435,
    x618, x885, x1074, 1.0E-5));
};
if (x2518) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2389) x Sym(2389), res:  x Const(64) x Const(1024) x Sym(2497) x Sym(2497)");
}
float* x2523 = (float*)myMalloc(1 * sizeof(float));;
x2523[0] = 1.0f;
float* x2525 = (float*)myMalloc(1 * sizeof(float));;
x2525[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2389, x2389));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2497, x2497));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2523, bias_desc, x2424, x2525, out_desc, x2509));
};
float* x2528 = (float*)myMalloc(1 * sizeof(float));;
x2528[0] = 0.0f;
float* x2530 = (float*)myMalloc(1 * sizeof(float));;
x2530[0] = 1.0f;
float* x2532 = (float*)myGpuMalloc(x2500 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2497, x2497));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2530, x_desc, x2509, x2528, x_desc, x2532));
};
if (x2535) {
} else {
assert(false && "ERROR not specified");
}
float* x2547 = (float*)myGpuMalloc(x2546 * sizeof(float));
float* x2548 = (float*)myMalloc(1 * sizeof(float));;
x2548[0] = 0.0f;
float* x2550 = (float*)myMalloc(1 * sizeof(float));;
x2550[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2497, x2497));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2541, x2541));

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
    x2550, in_desc, x2532, filt_desc, x711,
    conv_desc, algo, ws_data, ws_size,
    x2548, out_desc, x2547));
};
float* x2553 = (float*)myGpuMalloc(x2544 * sizeof(float));
float* x2554 = (float*)myMalloc(1 * sizeof(float));;
x2554[0] = 0.0f;
float* x2556 = (float*)myMalloc(1 * sizeof(float));;
x2556[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2541, x2541));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2541, x2541));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2556, x2556, in_desc, x2547, out_desc, x2553, sbmv_desc, x513,
    x1017, x498, x786, 1.0E-5));
};
float* x2559 = (float*)myMalloc(1 * sizeof(float));;
x2559[0] = 0.0f;
float* x2561 = (float*)myMalloc(1 * sizeof(float));;
x2561[0] = 1.0f;
float* x2563 = (float*)myGpuMalloc(x2544 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2541, x2541));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2561, x_desc, x2553, x2559, x_desc, x2563));
};
if (x2567) {
} else {
assert(false && "ERROR not specified");
}
float* x2580 = (float*)myGpuMalloc(x2579 * sizeof(float));
float* x2581 = (float*)myMalloc(1 * sizeof(float));;
x2581[0] = 0.0f;
float* x2583 = (float*)myMalloc(1 * sizeof(float));;
x2583[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2541, x2541));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2574, x2574));

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
    x2583, in_desc, x2563, filt_desc, x936,
    conv_desc, algo, ws_data, ws_size,
    x2581, out_desc, x2580));
};
float* x2586 = (float*)myGpuMalloc(x2577 * sizeof(float));
float* x2587 = (float*)myMalloc(1 * sizeof(float));;
x2587[0] = 0.0f;
float* x2589 = (float*)myMalloc(1 * sizeof(float));;
x2589[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2574, x2574));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2574, x2574));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2589, x2589, in_desc, x2580, out_desc, x2586, sbmv_desc, x681,
    x825, x468, x978, 1.0E-5));
};
float* x2592 = (float*)myMalloc(1 * sizeof(float));;
x2592[0] = 0.0f;
float* x2594 = (float*)myMalloc(1 * sizeof(float));;
x2594[0] = 1.0f;
float* x2596 = (float*)myGpuMalloc(x2577 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2574, x2574));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2594, x_desc, x2586, x2592, x_desc, x2596));
};
if (x2599) {
} else {
assert(false && "ERROR not specified");
}
float* x2611 = (float*)myGpuMalloc(x2610 * sizeof(float));
float* x2612 = (float*)myMalloc(1 * sizeof(float));;
x2612[0] = 0.0f;
float* x2614 = (float*)myMalloc(1 * sizeof(float));;
x2614[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2574, x2574));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2605, x2605));

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
    x2614, in_desc, x2596, filt_desc, x549,
    conv_desc, algo, ws_data, ws_size,
    x2612, out_desc, x2611));
};
float* x2617 = (float*)myGpuMalloc(x2608 * sizeof(float));
float* x2618 = (float*)myMalloc(1 * sizeof(float));;
x2618[0] = 0.0f;
float* x2620 = (float*)myMalloc(1 * sizeof(float));;
x2620[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2605, x2605));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2605, x2605));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2620, x2620, in_desc, x2611, out_desc, x2617, sbmv_desc, x1002,
    x537, x624, x807, 1.0E-5));
};
if (x2626) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2497) x Sym(2497), res:  x Const(64) x Const(1024) x Sym(2605) x Sym(2605)");
}
float* x2631 = (float*)myMalloc(1 * sizeof(float));;
x2631[0] = 1.0f;
float* x2633 = (float*)myMalloc(1 * sizeof(float));;
x2633[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2497, x2497));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2605, x2605));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2631, bias_desc, x2532, x2633, out_desc, x2617));
};
float* x2636 = (float*)myMalloc(1 * sizeof(float));;
x2636[0] = 0.0f;
float* x2638 = (float*)myMalloc(1 * sizeof(float));;
x2638[0] = 1.0f;
float* x2640 = (float*)myGpuMalloc(x2608 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2605, x2605));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2638, x_desc, x2617, x2636, x_desc, x2640));
};
if (x2643) {
} else {
assert(false && "ERROR not specified");
}
float* x2655 = (float*)myGpuMalloc(x2654 * sizeof(float));
float* x2656 = (float*)myMalloc(1 * sizeof(float));;
x2656[0] = 0.0f;
float* x2658 = (float*)myMalloc(1 * sizeof(float));;
x2658[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2605, x2605));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2649, x2649));

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
    x2658, in_desc, x2640, filt_desc, x675,
    conv_desc, algo, ws_data, ws_size,
    x2656, out_desc, x2655));
};
float* x2661 = (float*)myGpuMalloc(x2652 * sizeof(float));
float* x2662 = (float*)myMalloc(1 * sizeof(float));;
x2662[0] = 0.0f;
float* x2664 = (float*)myMalloc(1 * sizeof(float));;
x2664[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2649, x2649));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2649, x2649));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2664, x2664, in_desc, x2655, out_desc, x2661, sbmv_desc, x861,
    x930, x459, x621, 1.0E-5));
};
float* x2667 = (float*)myMalloc(1 * sizeof(float));;
x2667[0] = 0.0f;
float* x2669 = (float*)myMalloc(1 * sizeof(float));;
x2669[0] = 1.0f;
float* x2671 = (float*)myGpuMalloc(x2652 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2649, x2649));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2669, x_desc, x2661, x2667, x_desc, x2671));
};
if (x2675) {
} else {
assert(false && "ERROR not specified");
}
float* x2688 = (float*)myGpuMalloc(x2687 * sizeof(float));
float* x2689 = (float*)myMalloc(1 * sizeof(float));;
x2689[0] = 0.0f;
float* x2691 = (float*)myMalloc(1 * sizeof(float));;
x2691[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2649, x2649));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2682, x2682));

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
    x2691, in_desc, x2671, filt_desc, x360,
    conv_desc, algo, ws_data, ws_size,
    x2689, out_desc, x2688));
};
float* x2694 = (float*)myGpuMalloc(x2685 * sizeof(float));
float* x2695 = (float*)myMalloc(1 * sizeof(float));;
x2695[0] = 0.0f;
float* x2697 = (float*)myMalloc(1 * sizeof(float));;
x2697[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2682, x2682));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2682, x2682));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2697, x2697, in_desc, x2688, out_desc, x2694, sbmv_desc, x873,
    x735, x597, x408, 1.0E-5));
};
float* x2700 = (float*)myMalloc(1 * sizeof(float));;
x2700[0] = 0.0f;
float* x2702 = (float*)myMalloc(1 * sizeof(float));;
x2702[0] = 1.0f;
float* x2704 = (float*)myGpuMalloc(x2685 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2682, x2682));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2702, x_desc, x2694, x2700, x_desc, x2704));
};
if (x2707) {
} else {
assert(false && "ERROR not specified");
}
float* x2719 = (float*)myGpuMalloc(x2718 * sizeof(float));
float* x2720 = (float*)myMalloc(1 * sizeof(float));;
x2720[0] = 0.0f;
float* x2722 = (float*)myMalloc(1 * sizeof(float));;
x2722[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2682, x2682));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2713, x2713));

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
    x2722, in_desc, x2704, filt_desc, x894,
    conv_desc, algo, ws_data, ws_size,
    x2720, out_desc, x2719));
};
float* x2725 = (float*)myGpuMalloc(x2716 * sizeof(float));
float* x2726 = (float*)myMalloc(1 * sizeof(float));;
x2726[0] = 0.0f;
float* x2728 = (float*)myMalloc(1 * sizeof(float));;
x2728[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2713, x2713));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2713, x2713));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2728, x2728, in_desc, x2719, out_desc, x2725, sbmv_desc, x975,
    x444, x603, x837, 1.0E-5));
};
if (x2643) {
} else {
assert(false && "ERROR not specified");
}
float* x2741 = (float*)myGpuMalloc(x2740 * sizeof(float));
float* x2742 = (float*)myMalloc(1 * sizeof(float));;
x2742[0] = 0.0f;
float* x2744 = (float*)myMalloc(1 * sizeof(float));;
x2744[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2605, x2605));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2735, x2735));

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
    x2744, in_desc, x2640, filt_desc, x900,
    conv_desc, algo, ws_data, ws_size,
    x2742, out_desc, x2741));
};
float* x2747 = (float*)myGpuMalloc(x2738 * sizeof(float));
float* x2748 = (float*)myMalloc(1 * sizeof(float));;
x2748[0] = 0.0f;
float* x2750 = (float*)myMalloc(1 * sizeof(float));;
x2750[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2735, x2735));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2735, x2735));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2750, x2750, in_desc, x2741, out_desc, x2747, sbmv_desc, x777,
    x579, x450, x633, 1.0E-5));
};
if (x2756) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(2735) x Sym(2735), res:  x Const(64) x Const(2048) x Sym(2713) x Sym(2713)");
}
float* x2761 = (float*)myMalloc(1 * sizeof(float));;
x2761[0] = 1.0f;
float* x2763 = (float*)myMalloc(1 * sizeof(float));;
x2763[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2735, x2735));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2713, x2713));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2761, bias_desc, x2747, x2763, out_desc, x2725));
};
float* x2766 = (float*)myMalloc(1 * sizeof(float));;
x2766[0] = 0.0f;
float* x2768 = (float*)myMalloc(1 * sizeof(float));;
x2768[0] = 1.0f;
float* x2770 = (float*)myGpuMalloc(x2716 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2713, x2713));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2768, x_desc, x2725, x2766, x_desc, x2770));
};
if (x2773) {
} else {
assert(false && "ERROR not specified");
}
float* x2785 = (float*)myGpuMalloc(x2784 * sizeof(float));
float* x2786 = (float*)myMalloc(1 * sizeof(float));;
x2786[0] = 0.0f;
float* x2788 = (float*)myMalloc(1 * sizeof(float));;
x2788[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2713, x2713));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 2048, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2779, x2779));

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
    x2788, in_desc, x2770, filt_desc, x903,
    conv_desc, algo, ws_data, ws_size,
    x2786, out_desc, x2785));
};
float* x2791 = (float*)myGpuMalloc(x2782 * sizeof(float));
float* x2792 = (float*)myMalloc(1 * sizeof(float));;
x2792[0] = 0.0f;
float* x2794 = (float*)myMalloc(1 * sizeof(float));;
x2794[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2779, x2779));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2779, x2779));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2794, x2794, in_desc, x2785, out_desc, x2791, sbmv_desc, x396,
    x669, x720, x453, 1.0E-5));
};
float* x2797 = (float*)myMalloc(1 * sizeof(float));;
x2797[0] = 0.0f;
float* x2799 = (float*)myMalloc(1 * sizeof(float));;
x2799[0] = 1.0f;
float* x2801 = (float*)myGpuMalloc(x2782 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2779, x2779));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2799, x_desc, x2791, x2797, x_desc, x2801));
};
if (x2805) {
} else {
assert(false && "ERROR not specified");
}
float* x2818 = (float*)myGpuMalloc(x2817 * sizeof(float));
float* x2819 = (float*)myMalloc(1 * sizeof(float));;
x2819[0] = 0.0f;
float* x2821 = (float*)myMalloc(1 * sizeof(float));;
x2821[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2779, x2779));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2812, x2812));

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
    x2821, in_desc, x2801, filt_desc, x723,
    conv_desc, algo, ws_data, ws_size,
    x2819, out_desc, x2818));
};
float* x2824 = (float*)myGpuMalloc(x2815 * sizeof(float));
float* x2825 = (float*)myMalloc(1 * sizeof(float));;
x2825[0] = 0.0f;
float* x2827 = (float*)myMalloc(1 * sizeof(float));;
x2827[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2812, x2812));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2812, x2812));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2827, x2827, in_desc, x2818, out_desc, x2824, sbmv_desc, x738,
    x456, x672, x843, 1.0E-5));
};
float* x2830 = (float*)myMalloc(1 * sizeof(float));;
x2830[0] = 0.0f;
float* x2832 = (float*)myMalloc(1 * sizeof(float));;
x2832[0] = 1.0f;
float* x2834 = (float*)myGpuMalloc(x2815 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2812, x2812));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2832, x_desc, x2824, x2830, x_desc, x2834));
};
if (x2837) {
} else {
assert(false && "ERROR not specified");
}
float* x2849 = (float*)myGpuMalloc(x2848 * sizeof(float));
float* x2850 = (float*)myMalloc(1 * sizeof(float));;
x2850[0] = 0.0f;
float* x2852 = (float*)myMalloc(1 * sizeof(float));;
x2852[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2812, x2812));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2843, x2843));

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
    x2852, in_desc, x2834, filt_desc, x399,
    conv_desc, algo, ws_data, ws_size,
    x2850, out_desc, x2849));
};
float* x2855 = (float*)myGpuMalloc(x2846 * sizeof(float));
float* x2856 = (float*)myMalloc(1 * sizeof(float));;
x2856[0] = 0.0f;
float* x2858 = (float*)myMalloc(1 * sizeof(float));;
x2858[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2843, x2843));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2843, x2843));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2858, x2858, in_desc, x2849, out_desc, x2855, sbmv_desc, x540,
    x690, x462, x993, 1.0E-5));
};
if (x2864) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(2713) x Sym(2713), res:  x Const(64) x Const(2048) x Sym(2843) x Sym(2843)");
}
float* x2869 = (float*)myMalloc(1 * sizeof(float));;
x2869[0] = 1.0f;
float* x2871 = (float*)myMalloc(1 * sizeof(float));;
x2871[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2713, x2713));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2843, x2843));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2869, bias_desc, x2770, x2871, out_desc, x2855));
};
float* x2874 = (float*)myMalloc(1 * sizeof(float));;
x2874[0] = 0.0f;
float* x2876 = (float*)myMalloc(1 * sizeof(float));;
x2876[0] = 1.0f;
float* x2878 = (float*)myGpuMalloc(x2846 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2843, x2843));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2876, x_desc, x2855, x2874, x_desc, x2878));
};
if (x2881) {
} else {
assert(false && "ERROR not specified");
}
float* x2893 = (float*)myGpuMalloc(x2892 * sizeof(float));
float* x2894 = (float*)myMalloc(1 * sizeof(float));;
x2894[0] = 0.0f;
float* x2896 = (float*)myMalloc(1 * sizeof(float));;
x2896[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2843, x2843));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 2048, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2887, x2887));

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
    x2896, in_desc, x2878, filt_desc, x1053,
    conv_desc, algo, ws_data, ws_size,
    x2894, out_desc, x2893));
};
float* x2899 = (float*)myGpuMalloc(x2890 * sizeof(float));
float* x2900 = (float*)myMalloc(1 * sizeof(float));;
x2900[0] = 0.0f;
float* x2902 = (float*)myMalloc(1 * sizeof(float));;
x2902[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2887, x2887));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2887, x2887));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2902, x2902, in_desc, x2893, out_desc, x2899, sbmv_desc, x303,
    x492, x897, x1023, 1.0E-5));
};
float* x2905 = (float*)myMalloc(1 * sizeof(float));;
x2905[0] = 0.0f;
float* x2907 = (float*)myMalloc(1 * sizeof(float));;
x2907[0] = 1.0f;
float* x2909 = (float*)myGpuMalloc(x2890 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2887, x2887));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2907, x_desc, x2899, x2905, x_desc, x2909));
};
if (x2913) {
} else {
assert(false && "ERROR not specified");
}
float* x2926 = (float*)myGpuMalloc(x2925 * sizeof(float));
float* x2927 = (float*)myMalloc(1 * sizeof(float));;
x2927[0] = 0.0f;
float* x2929 = (float*)myMalloc(1 * sizeof(float));;
x2929[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2887, x2887));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2920, x2920));

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
    x2929, in_desc, x2909, filt_desc, x342,
    conv_desc, algo, ws_data, ws_size,
    x2927, out_desc, x2926));
};
float* x2932 = (float*)myGpuMalloc(x2923 * sizeof(float));
float* x2933 = (float*)myMalloc(1 * sizeof(float));;
x2933[0] = 0.0f;
float* x2935 = (float*)myMalloc(1 * sizeof(float));;
x2935[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2920, x2920));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2920, x2920));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2935, x2935, in_desc, x2926, out_desc, x2932, sbmv_desc, x840,
    x765, x294, x864, 1.0E-5));
};
float* x2938 = (float*)myMalloc(1 * sizeof(float));;
x2938[0] = 0.0f;
float* x2940 = (float*)myMalloc(1 * sizeof(float));;
x2940[0] = 1.0f;
float* x2942 = (float*)myGpuMalloc(x2923 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2920, x2920));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2940, x_desc, x2932, x2938, x_desc, x2942));
};
if (x2945) {
} else {
assert(false && "ERROR not specified");
}
float* x2957 = (float*)myGpuMalloc(x2956 * sizeof(float));
float* x2958 = (float*)myMalloc(1 * sizeof(float));;
x2958[0] = 0.0f;
float* x2960 = (float*)myMalloc(1 * sizeof(float));;
x2960[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2920, x2920));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2951, x2951));

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
    x2960, in_desc, x2942, filt_desc, x357,
    conv_desc, algo, ws_data, ws_size,
    x2958, out_desc, x2957));
};
float* x2963 = (float*)myGpuMalloc(x2954 * sizeof(float));
float* x2964 = (float*)myMalloc(1 * sizeof(float));;
x2964[0] = 0.0f;
float* x2966 = (float*)myMalloc(1 * sizeof(float));;
x2966[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2951, x2951));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2951, x2951));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2966, x2966, in_desc, x2957, out_desc, x2963, sbmv_desc, x567,
    x801, x1038, x627, 1.0E-5));
};
if (x2972) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(2843) x Sym(2843), res:  x Const(64) x Const(2048) x Sym(2951) x Sym(2951)");
}
float* x2977 = (float*)myMalloc(1 * sizeof(float));;
x2977[0] = 1.0f;
float* x2979 = (float*)myMalloc(1 * sizeof(float));;
x2979[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2843, x2843));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2951, x2951));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2977, bias_desc, x2878, x2979, out_desc, x2963));
};
float* x2982 = (float*)myMalloc(1 * sizeof(float));;
x2982[0] = 0.0f;
float* x2984 = (float*)myMalloc(1 * sizeof(float));;
x2984[0] = 1.0f;
float* x2986 = (float*)myGpuMalloc(x2954 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2951, x2951));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2984, x_desc, x2963, x2982, x_desc, x2986));
};
if (x2989) {
} else {
assert(false && "Image too small for averagePool_batch:  x Const(64) x Const(2048) x Sym(2951) x Sym(2951)|(2,2)");
}
float* x2994 = (float*)myMalloc(1 * sizeof(float));;
x2994[0] = 0.0f;
float* x2996 = (float*)myMalloc(1 * sizeof(float));;
x2996[0] = 1.0f;
float* x3006 = (float*)myGpuMalloc(x3005 * sizeof(float));

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x2951, x2951) );

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3000, x3000));

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
    x2996, in_desc, x2986, x2994, out_desc, x3006));
};
int32_t x3008 = 0;
int32_t x3009 = 1;
x3009 *= 64;
x3009 *= 2048;
int32_t x3012 = x3008;
bool x3013 = x3012 >= 2;
if (x3013) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3019 = x3012 == 0;
if (x3019) {
int32_t x3020 = x3009;
bool x3021 = x3020 == x3003;
if (x3021) {
} else {
assert(false && "must same size!!");
}
} else {
}
// gemm: List(Const(64), Const(2048)), Vector(Const(10), Const(2048))
float* x3029 = (float*)myGpuMalloc(640 * sizeof(float));
float* x3030 = (float*)myMalloc(1 * sizeof(float));;
x3030[0] = 0.0f;
float* x3032 = (float*)myMalloc(1 * sizeof(float));;
x3032[0] = 1.0f;
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 10,64,2048,x3032,x939,2048,x3006,2048,x3030,x3029,10));
float* x3035 = (float*)myMalloc(1 * sizeof(float));;
x3035[0] = 1.0f;
float* x3037 = (float*)myMalloc(1 * sizeof(float));;
x3037[0] = 1.0f;

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
    cudnnHandle, x3035, bias_desc, x402, x3037, out_desc, x3029));
};
// Tensor 'toCPU' invocation.
float* x3041 = (float*)myMalloc(640 * sizeof(float));;
CUDA_CALL(cudaMemcpy(x3041, x3029, 640 * sizeof(float), cudaMemcpyDeviceToHost));
printf("output (size Const(64) x Const(10))\n");
float x3044 = 0.0f;
for(int x3046=0; x3046 < 640; x3046++) {
float x3047 = x3044;
float x3048 = x3041[x3046];
float x3049 = fabs(x3048);
float x3050 = fabs(x3047);
bool x3051 = x3049 > x3050;
float x3052;
if (x3051) {
x3052 = x3048;
} else {
x3052 = x3047;
}
x3044 = x3052;

}
float x3056 = x3044;
printf("Max Abs: %.5f || ",x3056);
for(int x3058=0; x3058 < 10; x3058++) {
float x3059 = x3041[x3058];
printf("%.5f ",x3059);

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


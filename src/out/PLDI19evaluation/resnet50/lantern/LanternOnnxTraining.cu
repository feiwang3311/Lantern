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
int32_t x7 = open("../../cifar10_data/cifar-10-batches-bin/data_batch_1.bin",0);
int64_t x8 = fsize(x7);
int64_t x10 = x8 / 3073LL;
int32_t x11 = (int32_t)x10;
int32_t x12 = x11 * 3072;
float* x13 = (float*)myMalloc(x12 * sizeof(float));;
int* x14 = (int32_t*)myMalloc(x11 * sizeof(int32_t));;
char* x9 = (char*)mmap(0, x8, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x7, 0);
for(int x16=0; x16 < x11; x16++) {
int32_t x17 = x16 * 3073;
char x18 = x9[x17];
int32_t x19 = (int32_t)(unsigned char)x18;
x14[x16] = x19;
int32_t x25 = x17 + 1;
int32_t x23 = x16 * 3072;
for(int x22=0; x22 < 3072; x22++) {
int32_t x26 = x25 + x22;
char x27 = x9[x26];
int32_t x24 = x23 + x22;
float x28 = (float)(unsigned char)x27;
float x29 = x28 / 255.0f;
x13[x24] = x29;

}

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x37 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
float x38 = (float)x37;
float x39 = x38 / 1000000.0f;
printf("Data normalized (all prepare time) in %lf sec\n",x39);
// Tensor 'toGPU' invocation.
float* x313 = (float*)myGpuMalloc(262144 * sizeof(float));
int32_t x42 = open("/u/data/u99/wang603/TiarkMlEnv/Lantern/src/out/PLDI19evaluation/resnet50/resnet50.onnx.bin",0);
int64_t x43 = fsize(x42);
float* x44 = (float*)mmap(0, x43, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x42, 0);
float* x45 = x44+5205440;
CUDA_CALL(cudaMemcpy(x313, x45, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x316 = (float*)myGpuMalloc(256 * sizeof(float));
float* x46 = x44+148672;
CUDA_CALL(cudaMemcpy(x316, x46, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x319 = (float*)myGpuMalloc(128 * sizeof(float));
float* x47 = x44+816064;
CUDA_CALL(cudaMemcpy(x319, x47, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x322 = (float*)myGpuMalloc(128 * sizeof(float));
float* x48 = x44+950080;
CUDA_CALL(cudaMemcpy(x322, x48, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x325 = (float*)myGpuMalloc(64 * sizeof(float));
float* x49 = x44+94784;
CUDA_CALL(cudaMemcpy(x325, x49, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x328 = (float*)myGpuMalloc(32768 * sizeof(float));
float* x50 = x44+220608;
CUDA_CALL(cudaMemcpy(x328, x50, 32768 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x331 = (float*)myGpuMalloc(512 * sizeof(float));
float* x51 = x44+22495680;
CUDA_CALL(cudaMemcpy(x331, x51, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x334 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x52 = x44+2964928;
CUDA_CALL(cudaMemcpy(x334, x52, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x337 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x53 = x44+4348352;
CUDA_CALL(cudaMemcpy(x337, x53, 589824 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x340 = (float*)myGpuMalloc(512 * sizeof(float));
float* x54 = x44+20133312;
CUDA_CALL(cudaMemcpy(x340, x54, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x343 = (float*)myGpuMalloc(256 * sizeof(float));
float* x55 = x44+2169536;
CUDA_CALL(cudaMemcpy(x343, x55, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x346 = (float*)myGpuMalloc(128 * sizeof(float));
float* x56 = x44+668224;
CUDA_CALL(cudaMemcpy(x346, x56, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x349 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x57 = x44+2432448;
CUDA_CALL(cudaMemcpy(x349, x57, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x352 = (float*)myGpuMalloc(512 * sizeof(float));
float* x58 = x44+1446336;
CUDA_CALL(cudaMemcpy(x352, x58, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x355 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x59 = x44+4081088;
CUDA_CALL(cudaMemcpy(x355, x59, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x358 = (float*)myGpuMalloc(256 * sizeof(float));
float* x60 = x44+1578688;
CUDA_CALL(cudaMemcpy(x358, x60, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x361 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x61 = x44+6325696;
CUDA_CALL(cudaMemcpy(x361, x61, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x364 = (float*)myGpuMalloc(512 * sizeof(float));
float* x62 = x44+602048;
CUDA_CALL(cudaMemcpy(x364, x62, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x367 = (float*)myGpuMalloc(64 * sizeof(float));
float* x63 = x44+165888;
CUDA_CALL(cudaMemcpy(x367, x63, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x370 = (float*)myGpuMalloc(512 * sizeof(float));
float* x64 = x44+1164736;
CUDA_CALL(cudaMemcpy(x370, x64, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x373 = (float*)myGpuMalloc(64 * sizeof(float));
float* x65 = x44+6080;
CUDA_CALL(cudaMemcpy(x373, x65, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x376 = (float*)myGpuMalloc(147456 * sizeof(float));
float* x66 = x44+253888;
CUDA_CALL(cudaMemcpy(x376, x66, 147456 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x379 = (float*)myGpuMalloc(2359296 * sizeof(float));
float* x67 = x44+20135360;
CUDA_CALL(cudaMemcpy(x379, x67, 2359296 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x382 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x68 = x44+2960832;
CUDA_CALL(cudaMemcpy(x382, x68, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x385 = (float*)myGpuMalloc(256 * sizeof(float));
float* x69 = x44+3227072;
CUDA_CALL(cudaMemcpy(x385, x69, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x388 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x70 = x44+3228096;
CUDA_CALL(cudaMemcpy(x388, x70, 589824 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x391 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x71 = x44+43456;
CUDA_CALL(cudaMemcpy(x391, x71, 16384 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x394 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x72 = x44+22496704;
CUDA_CALL(cudaMemcpy(x394, x72, 1048576 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x397 = (float*)myGpuMalloc(2359296 * sizeof(float));
float* x73 = x44+9092544;
CUDA_CALL(cudaMemcpy(x397, x73, 2359296 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x400 = (float*)myGpuMalloc(128 * sizeof(float));
float* x74 = x44+816320;
CUDA_CALL(cudaMemcpy(x400, x74, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x403 = (float*)myGpuMalloc(256 * sizeof(float));
float* x75 = x44+60608;
CUDA_CALL(cudaMemcpy(x403, x75, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x406 = (float*)myGpuMalloc(256 * sizeof(float));
float* x76 = x44+219584;
CUDA_CALL(cudaMemcpy(x406, x76, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x409 = (float*)myGpuMalloc(128 * sizeof(float));
float* x77 = x44+1379392;
CUDA_CALL(cudaMemcpy(x409, x77, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x412 = (float*)myGpuMalloc(128 * sizeof(float));
float* x78 = x44+1231296;
CUDA_CALL(cudaMemcpy(x412, x78, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x415 = (float*)myGpuMalloc(64 * sizeof(float));
float* x79 = x44+1856;
CUDA_CALL(cudaMemcpy(x415, x79, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x418 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x80 = x44+1098176;
CUDA_CALL(cudaMemcpy(x418, x80, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x421 = (float*)myGpuMalloc(512 * sizeof(float));
float* x81 = x44+601536;
CUDA_CALL(cudaMemcpy(x421, x81, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x424 = (float*)myGpuMalloc(128 * sizeof(float));
float* x82 = x44+401728;
CUDA_CALL(cudaMemcpy(x424, x82, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x427 = (float*)myGpuMalloc(64 * sizeof(float));
float* x83 = x44+131904;
CUDA_CALL(cudaMemcpy(x427, x83, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x430 = (float*)myGpuMalloc(128 * sizeof(float));
float* x84 = x44+949696;
CUDA_CALL(cudaMemcpy(x430, x84, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x433 = (float*)myGpuMalloc(512 * sizeof(float));
float* x85 = x44+15664576;
CUDA_CALL(cudaMemcpy(x433, x85, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x436 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x86 = x44+18027968;
CUDA_CALL(cudaMemcpy(x436, x86, 1048576 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x439 = (float*)myGpuMalloc(10 * sizeof(float));
float* x87 = x44+23573952;
CUDA_CALL(cudaMemcpy(x439, x87, 10 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x442 = (float*)myGpuMalloc(64 * sizeof(float));
float* x88 = x44+43264;
CUDA_CALL(cudaMemcpy(x442, x88, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x445 = (float*)myGpuMalloc(512 * sizeof(float));
float* x89 = x44+11453376;
CUDA_CALL(cudaMemcpy(x445, x89, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x448 = (float*)myGpuMalloc(64 * sizeof(float));
float* x90 = x44+6272;
CUDA_CALL(cudaMemcpy(x448, x90, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x451 = (float*)myGpuMalloc(512 * sizeof(float));
float* x91 = x44+882112;
CUDA_CALL(cudaMemcpy(x451, x91, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x454 = (float*)myGpuMalloc(64 * sizeof(float));
float* x92 = x44+6144;
CUDA_CALL(cudaMemcpy(x454, x92, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x457 = (float*)myGpuMalloc(512 * sizeof(float));
float* x93 = x44+1445824;
CUDA_CALL(cudaMemcpy(x457, x93, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x460 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x94 = x44+1379776;
CUDA_CALL(cudaMemcpy(x460, x94, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x463 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x95 = x44+3818944;
CUDA_CALL(cudaMemcpy(x463, x95, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x466 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x96 = x44+5202368;
CUDA_CALL(cudaMemcpy(x466, x96, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x469 = (float*)myGpuMalloc(256 * sizeof(float));
float* x97 = x44+148416;
CUDA_CALL(cudaMemcpy(x469, x97, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x472 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x98 = x44+7441856;
CUDA_CALL(cudaMemcpy(x472, x98, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x475 = (float*)myGpuMalloc(64 * sizeof(float));
float* x99 = x44+94720;
CUDA_CALL(cudaMemcpy(x475, x99, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x478 = (float*)myGpuMalloc(128 * sizeof(float));
float* x100 = x44+1097792;
CUDA_CALL(cudaMemcpy(x478, x100, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x481 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x101 = x44+12504512;
CUDA_CALL(cudaMemcpy(x481, x101, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x484 = (float*)myGpuMalloc(256 * sizeof(float));
float* x102 = x44+4938944;
CUDA_CALL(cudaMemcpy(x484, x102, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x487 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x103 = x44+14611904;
CUDA_CALL(cudaMemcpy(x487, x103, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x490 = (float*)myGpuMalloc(512 * sizeof(float));
float* x104 = x44+15666112;
CUDA_CALL(cudaMemcpy(x490, x104, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x493 = (float*)myGpuMalloc(512 * sizeof(float));
float* x105 = x44+18026432;
CUDA_CALL(cudaMemcpy(x493, x105, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x496 = (float*)myGpuMalloc(512 * sizeof(float));
float* x106 = x44+9091520;
CUDA_CALL(cudaMemcpy(x496, x106, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x499 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x107 = x44+19080640;
CUDA_CALL(cudaMemcpy(x499, x107, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x502 = (float*)myGpuMalloc(256 * sizeof(float));
float* x108 = x44+6588608;
CUDA_CALL(cudaMemcpy(x502, x108, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x505 = (float*)myGpuMalloc(256 * sizeof(float));
float* x109 = x44+8299456;
CUDA_CALL(cudaMemcpy(x505, x109, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x508 = (float*)myGpuMalloc(256 * sizeof(float));
float* x110 = x44+60352;
CUDA_CALL(cudaMemcpy(x508, x110, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x511 = (float*)myGpuMalloc(64 * sizeof(float));
float* x111 = x44+202944;
CUDA_CALL(cudaMemcpy(x511, x111, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x514 = (float*)myGpuMalloc(36864 * sizeof(float));
float* x112 = x44+166080;
CUDA_CALL(cudaMemcpy(x514, x112, 36864 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x517 = (float*)myGpuMalloc(256 * sizeof(float));
float* x113 = x44+6058432;
CUDA_CALL(cudaMemcpy(x517, x113, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x520 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x114 = x44+2436544;
CUDA_CALL(cudaMemcpy(x520, x114, 524288 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x523 = (float*)myGpuMalloc(256 * sizeof(float));
float* x115 = x44+77248;
CUDA_CALL(cudaMemcpy(x523, x115, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x526 = (float*)myGpuMalloc(256 * sizeof(float));
float* x116 = x44+6587840;
CUDA_CALL(cudaMemcpy(x526, x116, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x529 = (float*)myGpuMalloc(512 * sizeof(float));
float* x117 = x44+20133824;
CUDA_CALL(cudaMemcpy(x529, x117, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x532 = (float*)myGpuMalloc(128 * sizeof(float));
float* x118 = x44+1379264;
CUDA_CALL(cudaMemcpy(x532, x118, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x535 = (float*)myGpuMalloc(256 * sizeof(float));
float* x119 = x44+7708608;
CUDA_CALL(cudaMemcpy(x535, x119, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x538 = (float*)myGpuMalloc(64 * sizeof(float));
float* x120 = x44+165824;
CUDA_CALL(cudaMemcpy(x538, x120, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x541 = (float*)myGpuMalloc(512 * sizeof(float));
float* x121 = x44+1164224;
CUDA_CALL(cudaMemcpy(x541, x121, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x544 = (float*)myGpuMalloc(36864 * sizeof(float));
float* x122 = x44+94912;
CUDA_CALL(cudaMemcpy(x544, x122, 36864 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x547 = (float*)myGpuMalloc(128 * sizeof(float));
float* x123 = x44+253376;
CUDA_CALL(cudaMemcpy(x547, x123, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x550 = (float*)myGpuMalloc(256 * sizeof(float));
float* x124 = x44+7708096;
CUDA_CALL(cudaMemcpy(x550, x124, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x553 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x125 = x44+2962880;
CUDA_CALL(cudaMemcpy(x553, x125, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x556 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x126 = x44+203200;
CUDA_CALL(cudaMemcpy(x556, x126, 16384 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x559 = (float*)myGpuMalloc(512 * sizeof(float));
float* x127 = x44+883648;
CUDA_CALL(cudaMemcpy(x559, x127, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x562 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x128 = x44+6059456;
CUDA_CALL(cudaMemcpy(x562, x128, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x565 = (float*)myGpuMalloc(36864 * sizeof(float));
float* x129 = x44+6336;
CUDA_CALL(cudaMemcpy(x565, x129, 36864 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x568 = (float*)myGpuMalloc(256 * sizeof(float));
float* x130 = x44+148928;
CUDA_CALL(cudaMemcpy(x568, x130, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x571 = (float*)myGpuMalloc(256 * sizeof(float));
float* x131 = x44+5467584;
CUDA_CALL(cudaMemcpy(x571, x131, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x574 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x132 = x44+8563136;
CUDA_CALL(cudaMemcpy(x574, x132, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x577 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x133 = x44+19076544;
CUDA_CALL(cudaMemcpy(x577, x133, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x580 = (float*)myGpuMalloc(128 * sizeof(float));
float* x134 = x44+816192;
CUDA_CALL(cudaMemcpy(x580, x134, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x583 = (float*)myGpuMalloc(256 * sizeof(float));
float* x135 = x44+3818176;
CUDA_CALL(cudaMemcpy(x583, x135, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x586 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x136 = x44+8299968;
CUDA_CALL(cudaMemcpy(x586, x136, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x589 = (float*)myGpuMalloc(256 * sizeof(float));
float* x137 = x44+5468352;
CUDA_CALL(cudaMemcpy(x589, x137, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x592 = (float*)myGpuMalloc(256 * sizeof(float));
float* x138 = x44+2170048;
CUDA_CALL(cudaMemcpy(x592, x138, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x595 = (float*)myGpuMalloc(128 * sizeof(float));
float* x139 = x44+668352;
CUDA_CALL(cudaMemcpy(x595, x139, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x598 = (float*)myGpuMalloc(512 * sizeof(float));
float* x140 = x44+468928;
CUDA_CALL(cudaMemcpy(x598, x140, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x601 = (float*)myGpuMalloc(64 * sizeof(float));
float* x141 = x44+94848;
CUDA_CALL(cudaMemcpy(x601, x141, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x604 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x142 = x44+23545280;
CUDA_CALL(cudaMemcpy(x604, x142, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x607 = (float*)myGpuMalloc(256 * sizeof(float));
float* x143 = x44+7179456;
CUDA_CALL(cudaMemcpy(x607, x143, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x610 = (float*)myGpuMalloc(64 * sizeof(float));
float* x144 = x44+43328;
CUDA_CALL(cudaMemcpy(x610, x144, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x613 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x145 = x44+401856;
CUDA_CALL(cudaMemcpy(x613, x145, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x616 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x146 = x44+14609856;
CUDA_CALL(cudaMemcpy(x616, x146, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x619 = (float*)myGpuMalloc(256 * sizeof(float));
float* x147 = x44+2169280;
CUDA_CALL(cudaMemcpy(x619, x147, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x622 = (float*)myGpuMalloc(256 * sizeof(float));
float* x148 = x44+7178944;
CUDA_CALL(cudaMemcpy(x622, x148, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x625 = (float*)myGpuMalloc(64 * sizeof(float));
float* x149 = x44+1920;
CUDA_CALL(cudaMemcpy(x625, x149, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x628 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x150 = x44+816576;
CUDA_CALL(cudaMemcpy(x628, x150, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x631 = (float*)myGpuMalloc(128 * sizeof(float));
float* x151 = x44+949952;
CUDA_CALL(cudaMemcpy(x631, x151, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x634 = (float*)myGpuMalloc(512 * sizeof(float));
float* x152 = x44+11452864;
CUDA_CALL(cudaMemcpy(x634, x152, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x637 = (float*)myGpuMalloc(64 * sizeof(float));
float* x153 = x44+6208;
CUDA_CALL(cudaMemcpy(x637, x153, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x640 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x154 = x44+12506560;
CUDA_CALL(cudaMemcpy(x640, x154, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x643 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x155 = x44+4939200;
CUDA_CALL(cudaMemcpy(x643, x155, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x646 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x156 = x44+2433472;
CUDA_CALL(cudaMemcpy(x646, x156, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x649 = (float*)myGpuMalloc(64 * sizeof(float));
float* x157 = x44+203136;
CUDA_CALL(cudaMemcpy(x649, x157, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x652 = (float*)myGpuMalloc(512 * sizeof(float));
float* x158 = x44+601024;
CUDA_CALL(cudaMemcpy(x652, x158, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x655 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x159 = x44+7442880;
CUDA_CALL(cudaMemcpy(x655, x159, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x658 = (float*)myGpuMalloc(512 * sizeof(float));
float* x160 = x44+9092032;
CUDA_CALL(cudaMemcpy(x658, x160, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x661 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x161 = x44+8564160;
CUDA_CALL(cudaMemcpy(x661, x161, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x664 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x162 = x44+23551424;
CUDA_CALL(cudaMemcpy(x664, x162, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x667 = (float*)myGpuMalloc(256 * sizeof(float));
float* x163 = x44+4938688;
CUDA_CALL(cudaMemcpy(x667, x163, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x670 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x164 = x44+14613952;
CUDA_CALL(cudaMemcpy(x670, x164, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x673 = (float*)myGpuMalloc(256 * sizeof(float));
float* x165 = x44+60096;
CUDA_CALL(cudaMemcpy(x673, x165, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x676 = (float*)myGpuMalloc(128 * sizeof(float));
float* x166 = x44+1097664;
CUDA_CALL(cudaMemcpy(x676, x166, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x679 = (float*)myGpuMalloc(128 * sizeof(float));
float* x167 = x44+401600;
CUDA_CALL(cudaMemcpy(x679, x167, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x682 = (float*)myGpuMalloc(256 * sizeof(float));
float* x168 = x44+4347328;
CUDA_CALL(cudaMemcpy(x682, x168, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x685 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x169 = x44+132032;
CUDA_CALL(cudaMemcpy(x685, x169, 16384 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x688 = (float*)myGpuMalloc(256 * sizeof(float));
float* x170 = x44+1578944;
CUDA_CALL(cudaMemcpy(x688, x170, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x691 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x171 = x44+1165760;
CUDA_CALL(cudaMemcpy(x691, x171, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x694 = (float*)myGpuMalloc(256 * sizeof(float));
float* x172 = x44+220352;
CUDA_CALL(cudaMemcpy(x694, x172, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x697 = (float*)myGpuMalloc(128 * sizeof(float));
float* x173 = x44+253760;
CUDA_CALL(cudaMemcpy(x697, x173, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x700 = (float*)myGpuMalloc(64 * sizeof(float));
float* x174 = x44+203008;
CUDA_CALL(cudaMemcpy(x700, x174, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x703 = (float*)myGpuMalloc(256 * sizeof(float));
float* x175 = x44+6058688;
CUDA_CALL(cudaMemcpy(x703, x175, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x706 = (float*)myGpuMalloc(512 * sizeof(float));
float* x176 = x44+15665088;
CUDA_CALL(cudaMemcpy(x706, x176, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x709 = (float*)myGpuMalloc(512 * sizeof(float));
float* x177 = x44+18026944;
CUDA_CALL(cudaMemcpy(x709, x177, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x712 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x178 = x44+8566208;
CUDA_CALL(cudaMemcpy(x712, x178, 524288 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x715 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x179 = x44+5203392;
CUDA_CALL(cudaMemcpy(x715, x179, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x718 = (float*)myGpuMalloc(256 * sizeof(float));
float* x180 = x44+8298944;
CUDA_CALL(cudaMemcpy(x718, x180, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x721 = (float*)myGpuMalloc(64 * sizeof(float));
float* x181 = x44+94656;
CUDA_CALL(cudaMemcpy(x721, x181, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x724 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x182 = x44+4084160;
CUDA_CALL(cudaMemcpy(x724, x182, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x727 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x183 = x44+19078592;
CUDA_CALL(cudaMemcpy(x727, x183, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x730 = (float*)myGpuMalloc(512 * sizeof(float));
float* x184 = x44+467392;
CUDA_CALL(cudaMemcpy(x730, x184, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x733 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x185 = x44+6322624;
CUDA_CALL(cudaMemcpy(x733, x185, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x736 = (float*)myGpuMalloc(512 * sizeof(float));
float* x186 = x44+883136;
CUDA_CALL(cudaMemcpy(x736, x186, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x739 = (float*)myGpuMalloc(128 * sizeof(float));
float* x187 = x44+1379648;
CUDA_CALL(cudaMemcpy(x739, x187, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x742 = (float*)myGpuMalloc(512 * sizeof(float));
float* x188 = x44+468416;
CUDA_CALL(cudaMemcpy(x742, x188, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x745 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x189 = x44+149440;
CUDA_CALL(cudaMemcpy(x745, x189, 16384 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x748 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x190 = x44+7445952;
CUDA_CALL(cudaMemcpy(x748, x190, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x751 = (float*)myGpuMalloc(1728 * sizeof(float));
float* x191 = x44+0;
CUDA_CALL(cudaMemcpy(x751, x191, 1728 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x754 = (float*)myGpuMalloc(64 * sizeof(float));
float* x192 = x44+131840;
CUDA_CALL(cudaMemcpy(x754, x192, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x757 = (float*)myGpuMalloc(512 * sizeof(float));
float* x193 = x44+15665600;
CUDA_CALL(cudaMemcpy(x757, x193, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x760 = (float*)myGpuMalloc(2359296 * sizeof(float));
float* x194 = x44+15666624;
CUDA_CALL(cudaMemcpy(x760, x194, 2359296 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x763 = (float*)myGpuMalloc(512 * sizeof(float));
float* x195 = x44+1445312;
CUDA_CALL(cudaMemcpy(x763, x195, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x766 = (float*)myGpuMalloc(256 * sizeof(float));
float* x196 = x44+3227840;
CUDA_CALL(cudaMemcpy(x766, x196, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x769 = (float*)myGpuMalloc(64 * sizeof(float));
float* x197 = x44+43392;
CUDA_CALL(cudaMemcpy(x769, x197, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x772 = (float*)myGpuMalloc(512 * sizeof(float));
float* x198 = x44+11452352;
CUDA_CALL(cudaMemcpy(x772, x198, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x775 = (float*)myGpuMalloc(512 * sizeof(float));
float* x199 = x44+18025920;
CUDA_CALL(cudaMemcpy(x775, x199, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x778 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x200 = x44+6324672;
CUDA_CALL(cudaMemcpy(x778, x200, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x781 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x201 = x44+60864;
CUDA_CALL(cudaMemcpy(x781, x201, 16384 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x784 = (float*)myGpuMalloc(256 * sizeof(float));
float* x202 = x44+5468096;
CUDA_CALL(cudaMemcpy(x784, x202, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x787 = (float*)myGpuMalloc(64 * sizeof(float));
float* x203 = x44+43200;
CUDA_CALL(cudaMemcpy(x787, x203, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x790 = (float*)myGpuMalloc(147456 * sizeof(float));
float* x204 = x44+1231808;
CUDA_CALL(cudaMemcpy(x790, x204, 147456 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x793 = (float*)myGpuMalloc(256 * sizeof(float));
float* x205 = x44+149184;
CUDA_CALL(cudaMemcpy(x793, x205, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x796 = (float*)myGpuMalloc(512 * sizeof(float));
float* x206 = x44+1163712;
CUDA_CALL(cudaMemcpy(x796, x206, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x799 = (float*)myGpuMalloc(256 * sizeof(float));
float* x207 = x44+7178688;
CUDA_CALL(cudaMemcpy(x799, x207, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x802 = (float*)myGpuMalloc(512 * sizeof(float));
float* x208 = x44+22495168;
CUDA_CALL(cudaMemcpy(x802, x208, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x805 = (float*)myGpuMalloc(128 * sizeof(float));
float* x209 = x44+949824;
CUDA_CALL(cudaMemcpy(x805, x209, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x808 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x210 = x44+78272;
CUDA_CALL(cudaMemcpy(x808, x210, 16384 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x811 = (float*)myGpuMalloc(128 * sizeof(float));
float* x211 = x44+253504;
CUDA_CALL(cudaMemcpy(x811, x211, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x814 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x212 = x44+14607808;
CUDA_CALL(cudaMemcpy(x814, x212, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x817 = (float*)myGpuMalloc(256 * sizeof(float));
float* x213 = x44+4348096;
CUDA_CALL(cudaMemcpy(x817, x213, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x820 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x214 = x44+1579456;
CUDA_CALL(cudaMemcpy(x820, x214, 589824 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x823 = (float*)myGpuMalloc(256 * sizeof(float));
float* x215 = x44+7708864;
CUDA_CALL(cudaMemcpy(x823, x215, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x826 = (float*)myGpuMalloc(128 * sizeof(float));
float* x216 = x44+668480;
CUDA_CALL(cudaMemcpy(x826, x216, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x829 = (float*)myGpuMalloc(256 * sizeof(float));
float* x217 = x44+4347840;
CUDA_CALL(cudaMemcpy(x829, x217, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x832 = (float*)myGpuMalloc(64 * sizeof(float));
float* x218 = x44+203072;
CUDA_CALL(cudaMemcpy(x832, x218, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x835 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x219 = x44+1447360;
CUDA_CALL(cudaMemcpy(x835, x219, 131072 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x838 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x220 = x44+23547328;
CUDA_CALL(cudaMemcpy(x838, x220, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x841 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x221 = x44+4083136;
CUDA_CALL(cudaMemcpy(x841, x221, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x844 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x222 = x44+8565184;
CUDA_CALL(cudaMemcpy(x844, x222, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x847 = (float*)myGpuMalloc(256 * sizeof(float));
float* x223 = x44+220096;
CUDA_CALL(cudaMemcpy(x847, x223, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x850 = (float*)myGpuMalloc(256 * sizeof(float));
float* x224 = x44+6588096;
CUDA_CALL(cudaMemcpy(x850, x224, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x853 = (float*)myGpuMalloc(256 * sizeof(float));
float* x225 = x44+6058944;
CUDA_CALL(cudaMemcpy(x853, x225, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x856 = (float*)myGpuMalloc(64 * sizeof(float));
float* x226 = x44+166016;
CUDA_CALL(cudaMemcpy(x856, x226, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x859 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x227 = x44+5204416;
CUDA_CALL(cudaMemcpy(x859, x227, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x862 = (float*)myGpuMalloc(256 * sizeof(float));
float* x228 = x44+8299200;
CUDA_CALL(cudaMemcpy(x862, x228, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x865 = (float*)myGpuMalloc(128 * sizeof(float));
float* x229 = x44+401472;
CUDA_CALL(cudaMemcpy(x865, x229, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x868 = (float*)myGpuMalloc(147456 * sizeof(float));
float* x230 = x44+950208;
CUDA_CALL(cudaMemcpy(x868, x230, 147456 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x871 = (float*)myGpuMalloc(256 * sizeof(float));
float* x231 = x44+4938432;
CUDA_CALL(cudaMemcpy(x871, x231, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x874 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x232 = x44+12508608;
CUDA_CALL(cudaMemcpy(x874, x232, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x877 = (float*)myGpuMalloc(512 * sizeof(float));
float* x233 = x44+22494656;
CUDA_CALL(cudaMemcpy(x877, x233, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x880 = (float*)myGpuMalloc(512 * sizeof(float));
float* x234 = x44+18027456;
CUDA_CALL(cudaMemcpy(x880, x234, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x883 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x235 = x44+884160;
CUDA_CALL(cudaMemcpy(x883, x235, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x886 = (float*)myGpuMalloc(256 * sizeof(float));
float* x236 = x44+4347584;
CUDA_CALL(cudaMemcpy(x886, x236, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x889 = (float*)myGpuMalloc(256 * sizeof(float));
float* x237 = x44+1579200;
CUDA_CALL(cudaMemcpy(x889, x237, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x892 = (float*)myGpuMalloc(256 * sizeof(float));
float* x238 = x44+59840;
CUDA_CALL(cudaMemcpy(x892, x238, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x895 = (float*)myGpuMalloc(256 * sizeof(float));
float* x239 = x44+3818432;
CUDA_CALL(cudaMemcpy(x895, x239, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x898 = (float*)myGpuMalloc(512 * sizeof(float));
float* x240 = x44+9090496;
CUDA_CALL(cudaMemcpy(x898, x240, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x901 = (float*)myGpuMalloc(512 * sizeof(float));
float* x241 = x44+22496192;
CUDA_CALL(cudaMemcpy(x901, x241, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x904 = (float*)myGpuMalloc(256 * sizeof(float));
float* x242 = x44+77504;
CUDA_CALL(cudaMemcpy(x904, x242, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x907 = (float*)myGpuMalloc(128 * sizeof(float));
float* x243 = x44+253632;
CUDA_CALL(cudaMemcpy(x907, x243, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x910 = (float*)myGpuMalloc(512 * sizeof(float));
float* x244 = x44+11451840;
CUDA_CALL(cudaMemcpy(x910, x244, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x913 = (float*)myGpuMalloc(64 * sizeof(float));
float* x245 = x44+1728;
CUDA_CALL(cudaMemcpy(x913, x245, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x916 = (float*)myGpuMalloc(512 * sizeof(float));
float* x246 = x44+600512;
CUDA_CALL(cudaMemcpy(x916, x246, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x919 = (float*)myGpuMalloc(64 * sizeof(float));
float* x247 = x44+131776;
CUDA_CALL(cudaMemcpy(x919, x247, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x922 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x248 = x44+7443904;
CUDA_CALL(cudaMemcpy(x922, x248, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x925 = (float*)myGpuMalloc(512 * sizeof(float));
float* x249 = x44+467904;
CUDA_CALL(cudaMemcpy(x925, x249, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x928 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x250 = x44+2963904;
CUDA_CALL(cudaMemcpy(x928, x250, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x931 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x251 = x44+11453888;
CUDA_CALL(cudaMemcpy(x931, x251, 1048576 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x934 = (float*)myGpuMalloc(512 * sizeof(float));
float* x252 = x44+20134336;
CUDA_CALL(cudaMemcpy(x934, x252, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x937 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x253 = x44+12510656;
CUDA_CALL(cudaMemcpy(x937, x253, 2097152 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x940 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x254 = x44+14616000;
CUDA_CALL(cudaMemcpy(x940, x254, 1048576 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x943 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x255 = x44+2434496;
CUDA_CALL(cudaMemcpy(x943, x255, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x946 = (float*)myGpuMalloc(128 * sizeof(float));
float* x256 = x44+1097920;
CUDA_CALL(cudaMemcpy(x946, x256, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x949 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x257 = x44+4085184;
CUDA_CALL(cudaMemcpy(x949, x257, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x952 = (float*)myGpuMalloc(256 * sizeof(float));
float* x258 = x44+3227328;
CUDA_CALL(cudaMemcpy(x952, x258, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x955 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x259 = x44+2961856;
CUDA_CALL(cudaMemcpy(x955, x259, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x958 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x260 = x44+7179712;
CUDA_CALL(cudaMemcpy(x958, x260, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x961 = (float*)myGpuMalloc(128 * sizeof(float));
float* x261 = x44+668096;
CUDA_CALL(cudaMemcpy(x961, x261, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x964 = (float*)myGpuMalloc(512 * sizeof(float));
float* x262 = x44+1165248;
CUDA_CALL(cudaMemcpy(x964, x262, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x967 = (float*)myGpuMalloc(512 * sizeof(float));
float* x263 = x44+9091008;
CUDA_CALL(cudaMemcpy(x967, x263, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x970 = (float*)myGpuMalloc(128 * sizeof(float));
float* x264 = x44+816448;
CUDA_CALL(cudaMemcpy(x970, x264, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x973 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x265 = x44+7709120;
CUDA_CALL(cudaMemcpy(x973, x265, 589824 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x976 = (float*)myGpuMalloc(20480 * sizeof(float));
float* x266 = x44+23553472;
CUDA_CALL(cudaMemcpy(x976, x266, 20480 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x979 = (float*)myGpuMalloc(256 * sizeof(float));
float* x267 = x44+4938176;
CUDA_CALL(cudaMemcpy(x979, x267, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x982 = (float*)myGpuMalloc(256 * sizeof(float));
float* x268 = x44+2169792;
CUDA_CALL(cudaMemcpy(x982, x268, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x985 = (float*)myGpuMalloc(256 * sizeof(float));
float* x269 = x44+6059200;
CUDA_CALL(cudaMemcpy(x985, x269, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x988 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x270 = x44+6323648;
CUDA_CALL(cudaMemcpy(x988, x270, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x991 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x271 = x44+4082112;
CUDA_CALL(cudaMemcpy(x991, x271, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x994 = (float*)myGpuMalloc(4096 * sizeof(float));
float* x272 = x44+1984;
CUDA_CALL(cudaMemcpy(x994, x272, 4096 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x997 = (float*)myGpuMalloc(512 * sizeof(float));
float* x273 = x44+1446848;
CUDA_CALL(cudaMemcpy(x997, x273, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1000 = (float*)myGpuMalloc(147456 * sizeof(float));
float* x274 = x44+668608;
CUDA_CALL(cudaMemcpy(x1000, x274, 147456 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1003 = (float*)myGpuMalloc(128 * sizeof(float));
float* x275 = x44+1231552;
CUDA_CALL(cudaMemcpy(x1003, x275, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1006 = (float*)myGpuMalloc(256 * sizeof(float));
float* x276 = x44+3818688;
CUDA_CALL(cudaMemcpy(x1006, x276, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1009 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x277 = x44+6321600;
CUDA_CALL(cudaMemcpy(x1009, x277, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1012 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x278 = x44+12502464;
CUDA_CALL(cudaMemcpy(x1012, x278, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1015 = (float*)myGpuMalloc(256 * sizeof(float));
float* x279 = x44+8299712;
CUDA_CALL(cudaMemcpy(x1015, x279, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1018 = (float*)myGpuMalloc(256 * sizeof(float));
float* x280 = x44+5467840;
CUDA_CALL(cudaMemcpy(x1018, x280, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1021 = (float*)myGpuMalloc(128 * sizeof(float));
float* x281 = x44+1231424;
CUDA_CALL(cudaMemcpy(x1021, x281, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1024 = (float*)myGpuMalloc(256 * sizeof(float));
float* x282 = x44+78016;
CUDA_CALL(cudaMemcpy(x1024, x282, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1027 = (float*)myGpuMalloc(64 * sizeof(float));
float* x283 = x44+131968;
CUDA_CALL(cudaMemcpy(x1027, x283, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1030 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x284 = x44+19082688;
CUDA_CALL(cudaMemcpy(x1030, x284, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1033 = (float*)myGpuMalloc(512 * sizeof(float));
float* x285 = x44+882624;
CUDA_CALL(cudaMemcpy(x1033, x285, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1036 = (float*)myGpuMalloc(256 * sizeof(float));
float* x286 = x44+219840;
CUDA_CALL(cudaMemcpy(x1036, x286, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1039 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x287 = x44+8562112;
CUDA_CALL(cudaMemcpy(x1039, x287, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1042 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x288 = x44+5468608;
CUDA_CALL(cudaMemcpy(x1042, x288, 589824 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1045 = (float*)myGpuMalloc(256 * sizeof(float));
float* x289 = x44+7179200;
CUDA_CALL(cudaMemcpy(x1045, x289, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1048 = (float*)myGpuMalloc(64 * sizeof(float));
float* x290 = x44+1792;
CUDA_CALL(cudaMemcpy(x1048, x290, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1051 = (float*)myGpuMalloc(128 * sizeof(float));
float* x291 = x44+401344;
CUDA_CALL(cudaMemcpy(x1051, x291, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1054 = (float*)myGpuMalloc(256 * sizeof(float));
float* x292 = x44+7708352;
CUDA_CALL(cudaMemcpy(x1054, x292, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1057 = (float*)myGpuMalloc(256 * sizeof(float));
float* x293 = x44+6588352;
CUDA_CALL(cudaMemcpy(x1057, x293, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1060 = (float*)myGpuMalloc(512 * sizeof(float));
float* x294 = x44+20134848;
CUDA_CALL(cudaMemcpy(x1060, x294, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1063 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x295 = x44+602560;
CUDA_CALL(cudaMemcpy(x1063, x295, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1066 = (float*)myGpuMalloc(64 * sizeof(float));
float* x296 = x44+165952;
CUDA_CALL(cudaMemcpy(x1066, x296, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1069 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x297 = x44+469440;
CUDA_CALL(cudaMemcpy(x1069, x297, 131072 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1072 = (float*)myGpuMalloc(256 * sizeof(float));
float* x298 = x44+3227584;
CUDA_CALL(cudaMemcpy(x1072, x298, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1075 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x299 = x44+23549376;
CUDA_CALL(cudaMemcpy(x1075, x299, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1078 = (float*)myGpuMalloc(128 * sizeof(float));
float* x300 = x44+1231680;
CUDA_CALL(cudaMemcpy(x1078, x300, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1081 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x301 = x44+6588864;
CUDA_CALL(cudaMemcpy(x1081, x301, 589824 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1084 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x302 = x44+5201344;
CUDA_CALL(cudaMemcpy(x1084, x302, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1087 = (float*)myGpuMalloc(256 * sizeof(float));
float* x303 = x44+77760;
CUDA_CALL(cudaMemcpy(x1087, x303, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1090 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x304 = x44+19084736;
CUDA_CALL(cudaMemcpy(x1090, x304, 1048576 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1093 = (float*)myGpuMalloc(128 * sizeof(float));
float* x305 = x44+1098048;
CUDA_CALL(cudaMemcpy(x1093, x305, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1096 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x306 = x44+2435520;
CUDA_CALL(cudaMemcpy(x1096, x306, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1099 = (float*)myGpuMalloc(128 * sizeof(float));
float* x307 = x44+1379520;
CUDA_CALL(cudaMemcpy(x1099, x307, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1102 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x308 = x44+2170304;
CUDA_CALL(cudaMemcpy(x1102, x308, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1105 = (float*)myGpuMalloc(256 * sizeof(float));
float* x309 = x44+1578432;
CUDA_CALL(cudaMemcpy(x1105, x309, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1108 = (float*)myGpuMalloc(256 * sizeof(float));
float* x310 = x44+3817920;
CUDA_CALL(cudaMemcpy(x1108, x310, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1111 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x311 = x44+7444928;
CUDA_CALL(cudaMemcpy(x1111, x311, 1024 * sizeof(float), cudaMemcpyHostToDevice));
float* x1113 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1114 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1115 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1116 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1117 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1118 = (float*)myGpuMalloc(32768 * sizeof(float));
float* x1119 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1120 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1121 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x1122 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1123 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1124 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1125 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1126 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1127 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1128 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1129 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1130 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1131 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1132 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1133 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1134 = (float*)myGpuMalloc(147456 * sizeof(float));
float* x1135 = (float*)myGpuMalloc(2359296 * sizeof(float));
float* x1136 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1137 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1138 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x1139 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x1140 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1141 = (float*)myGpuMalloc(2359296 * sizeof(float));
float* x1142 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1143 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1144 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1145 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1146 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1147 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1148 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x1149 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1150 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1151 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1152 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1153 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1154 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1155 = (float*)myGpuMalloc(10 * sizeof(float));
float* x1156 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1157 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1158 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1159 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1160 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1161 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1162 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x1163 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1164 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1165 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1166 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1167 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1168 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1169 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x1170 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1171 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x1172 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1173 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1174 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1175 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x1176 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1177 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1178 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1179 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1180 = (float*)myGpuMalloc(36864 * sizeof(float));
float* x1181 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1182 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1183 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1184 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1185 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1186 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1187 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1188 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1189 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1190 = (float*)myGpuMalloc(36864 * sizeof(float));
float* x1191 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1192 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1193 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1194 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x1195 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1196 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1197 = (float*)myGpuMalloc(36864 * sizeof(float));
float* x1198 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1199 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1200 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1201 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x1202 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1203 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1204 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1205 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1206 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1207 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1208 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1209 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1210 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x1211 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1212 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1213 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x1214 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x1215 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1216 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1217 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1218 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x1219 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1220 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1221 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1222 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x1223 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1224 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1225 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1226 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1227 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1228 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1229 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1230 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x1231 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1232 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x1233 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1234 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1235 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1236 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1237 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x1238 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1239 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x1240 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1241 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1242 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1243 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1244 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1245 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1246 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1247 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1248 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1249 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1250 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1251 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x1252 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1253 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1254 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1255 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1256 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1257 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x1258 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1259 = (float*)myGpuMalloc(1728 * sizeof(float));
float* x1260 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1261 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1262 = (float*)myGpuMalloc(2359296 * sizeof(float));
float* x1263 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1264 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1265 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1266 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1267 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1268 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1269 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x1270 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1271 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1272 = (float*)myGpuMalloc(147456 * sizeof(float));
float* x1273 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1274 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1275 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1276 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1277 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1278 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x1279 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1280 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x1281 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1282 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x1283 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1284 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1285 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1286 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1287 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x1288 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x1289 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1290 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1291 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1292 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1293 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1294 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1295 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1296 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1297 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1298 = (float*)myGpuMalloc(147456 * sizeof(float));
float* x1299 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1300 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x1301 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1302 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1303 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x1304 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1305 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1306 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1307 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1308 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1309 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1310 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1311 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1312 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1313 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1314 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1315 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1316 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1317 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1318 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1319 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1320 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1321 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1322 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1323 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1324 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1325 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1326 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1327 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1328 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1329 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1330 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1331 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1332 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1333 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x1334 = (float*)myGpuMalloc(20480 * sizeof(float));
float* x1335 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1336 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1337 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1338 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1339 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1340 = (float*)myGpuMalloc(4096 * sizeof(float));
float* x1341 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1342 = (float*)myGpuMalloc(147456 * sizeof(float));
float* x1343 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1344 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1345 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1346 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x1347 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1348 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1349 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1350 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1351 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1352 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x1353 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1354 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1355 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1356 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x1357 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1358 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1359 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1360 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1361 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1362 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1363 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x1364 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1365 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x1366 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1367 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x1368 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1369 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x1370 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1371 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1372 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1373 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1374 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1375 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1376 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1377 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1378 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1379 = (float*)myGpuMalloc(1024 * sizeof(float));
double* x1380 = (double*)myMalloc(4 * sizeof(double));;
double* x1381 = (double*)myMalloc(4 * sizeof(double));;
int64_t x1382 = (long)mallocAddr;
int64_t x1383 = (long)gpuMallocAddr;
// training loop starts here
int32_t x1394 = x11 / 64;
int32_t x1411 = 31 / 1;
int32_t x1412 = x1411 + 1;
int32_t x1416 = 4096 * x1412;
int32_t x1417 = x1416 * x1412;
int32_t x1413 = x1412 * x1412;
int32_t x1414 = 64 * x1413;
int32_t x1415 = 64 * x1414;
int32_t x1443 = x1412 - 2;
int32_t x1444 = x1443 / 2;
int32_t x1445 = x1444 + 1;
int32_t x1449 = 4096 * x1445;
int32_t x1450 = x1449 * x1445;
bool x1454 = x1445 >= 1;
bool x1455;
if (x1454) {
x1455 = x1454;
} else {
x1455 = false;
}
int32_t x1460 = x1444 / 1;
int32_t x1461 = x1460 + 1;
int32_t x1465 = 4096 * x1461;
int32_t x1466 = x1465 * x1461;
int32_t x1462 = x1461 * x1461;
int32_t x1463 = 64 * x1462;
int32_t x1464 = 64 * x1463;
int32_t x1488 = x1461 + 2;
bool x1489 = x1488 >= 3;
bool x1490;
if (x1489) {
x1490 = x1489;
} else {
x1490 = false;
}
int32_t x1495 = x1488 - 3;
int32_t x1496 = x1495 / 1;
int32_t x1497 = x1496 + 1;
int32_t x1501 = 4096 * x1497;
int32_t x1502 = x1501 * x1497;
int32_t x1498 = x1497 * x1497;
int32_t x1499 = 64 * x1498;
int32_t x1500 = 64 * x1499;
bool x1524 = x1497 >= 1;
bool x1525;
if (x1524) {
x1525 = x1524;
} else {
x1525 = false;
}
int32_t x1530 = x1496 / 1;
int32_t x1531 = x1530 + 1;
int32_t x1535 = 16384 * x1531;
int32_t x1536 = x1535 * x1531;
int32_t x1532 = x1531 * x1531;
int32_t x1533 = 256 * x1532;
int32_t x1534 = 64 * x1533;
int32_t x1558 = 16384 * x1461;
int32_t x1559 = x1558 * x1461;
int32_t x1556 = 256 * x1462;
int32_t x1557 = 64 * x1556;
bool x1576 = x1461 == 1;
bool x1577 = x1461 == x1531;
bool x1578 = x1576 || x1577;
bool x1579;
if (x1578) {
x1579 = x1578;
} else {
x1579 = false;
}
bool x1594 = x1531 >= 1;
bool x1595;
if (x1594) {
x1595 = x1594;
} else {
x1595 = false;
}
int32_t x1600 = x1530 / 1;
int32_t x1601 = x1600 + 1;
int32_t x1605 = 4096 * x1601;
int32_t x1606 = x1605 * x1601;
int32_t x1602 = x1601 * x1601;
int32_t x1603 = 64 * x1602;
int32_t x1604 = 64 * x1603;
int32_t x1628 = x1601 + 2;
bool x1629 = x1628 >= 3;
bool x1630;
if (x1629) {
x1630 = x1629;
} else {
x1630 = false;
}
int32_t x1635 = x1628 - 3;
int32_t x1636 = x1635 / 1;
int32_t x1637 = x1636 + 1;
int32_t x1641 = 4096 * x1637;
int32_t x1642 = x1641 * x1637;
int32_t x1638 = x1637 * x1637;
int32_t x1639 = 64 * x1638;
int32_t x1640 = 64 * x1639;
bool x1664 = x1637 >= 1;
bool x1665;
if (x1664) {
x1665 = x1664;
} else {
x1665 = false;
}
int32_t x1670 = x1636 / 1;
int32_t x1671 = x1670 + 1;
int32_t x1675 = 16384 * x1671;
int32_t x1676 = x1675 * x1671;
int32_t x1672 = x1671 * x1671;
int32_t x1673 = 256 * x1672;
int32_t x1674 = 64 * x1673;
bool x1693 = x1531 == 1;
bool x1694 = x1531 == x1671;
bool x1695 = x1693 || x1694;
bool x1696;
if (x1695) {
x1696 = x1695;
} else {
x1696 = false;
}
bool x1711 = x1671 >= 1;
bool x1712;
if (x1711) {
x1712 = x1711;
} else {
x1712 = false;
}
int32_t x1717 = x1670 / 1;
int32_t x1718 = x1717 + 1;
int32_t x1722 = 4096 * x1718;
int32_t x1723 = x1722 * x1718;
int32_t x1719 = x1718 * x1718;
int32_t x1720 = 64 * x1719;
int32_t x1721 = 64 * x1720;
int32_t x1745 = x1718 + 2;
bool x1746 = x1745 >= 3;
bool x1747;
if (x1746) {
x1747 = x1746;
} else {
x1747 = false;
}
int32_t x1752 = x1745 - 3;
int32_t x1753 = x1752 / 1;
int32_t x1754 = x1753 + 1;
int32_t x1758 = 4096 * x1754;
int32_t x1759 = x1758 * x1754;
int32_t x1755 = x1754 * x1754;
int32_t x1756 = 64 * x1755;
int32_t x1757 = 64 * x1756;
bool x1781 = x1754 >= 1;
bool x1782;
if (x1781) {
x1782 = x1781;
} else {
x1782 = false;
}
int32_t x1787 = x1753 / 1;
int32_t x1788 = x1787 + 1;
int32_t x1792 = 16384 * x1788;
int32_t x1793 = x1792 * x1788;
int32_t x1789 = x1788 * x1788;
int32_t x1790 = 256 * x1789;
int32_t x1791 = 64 * x1790;
bool x1810 = x1671 == 1;
bool x1811 = x1671 == x1788;
bool x1812 = x1810 || x1811;
bool x1813;
if (x1812) {
x1813 = x1812;
} else {
x1813 = false;
}
bool x1828 = x1788 >= 1;
bool x1829;
if (x1828) {
x1829 = x1828;
} else {
x1829 = false;
}
int32_t x1834 = x1787 / 1;
int32_t x1835 = x1834 + 1;
int32_t x1839 = 8192 * x1835;
int32_t x1840 = x1839 * x1835;
int32_t x1836 = x1835 * x1835;
int32_t x1837 = 128 * x1836;
int32_t x1838 = 64 * x1837;
int32_t x1862 = x1835 + 2;
bool x1863 = x1862 >= 3;
bool x1864;
if (x1863) {
x1864 = x1863;
} else {
x1864 = false;
}
int32_t x1869 = x1862 - 3;
int32_t x1870 = x1869 / 2;
int32_t x1871 = x1870 + 1;
int32_t x1875 = 8192 * x1871;
int32_t x1876 = x1875 * x1871;
int32_t x1872 = x1871 * x1871;
int32_t x1873 = 128 * x1872;
int32_t x1874 = 64 * x1873;
bool x1898 = x1871 >= 1;
bool x1899;
if (x1898) {
x1899 = x1898;
} else {
x1899 = false;
}
int32_t x1904 = x1870 / 1;
int32_t x1905 = x1904 + 1;
int32_t x1909 = 32768 * x1905;
int32_t x1910 = x1909 * x1905;
int32_t x1906 = x1905 * x1905;
int32_t x1907 = 512 * x1906;
int32_t x1908 = 64 * x1907;
int32_t x1930 = x1787 / 2;
int32_t x1931 = x1930 + 1;
int32_t x1935 = 32768 * x1931;
int32_t x1936 = x1935 * x1931;
int32_t x1932 = x1931 * x1931;
int32_t x1933 = 512 * x1932;
int32_t x1934 = 64 * x1933;
bool x1953 = x1931 == 1;
bool x1954 = x1931 == x1905;
bool x1955 = x1953 || x1954;
bool x1956;
if (x1955) {
x1956 = x1955;
} else {
x1956 = false;
}
bool x1971 = x1905 >= 1;
bool x1972;
if (x1971) {
x1972 = x1971;
} else {
x1972 = false;
}
int32_t x1977 = x1904 / 1;
int32_t x1978 = x1977 + 1;
int32_t x1982 = 8192 * x1978;
int32_t x1983 = x1982 * x1978;
int32_t x1979 = x1978 * x1978;
int32_t x1980 = 128 * x1979;
int32_t x1981 = 64 * x1980;
int32_t x2005 = x1978 + 2;
bool x2006 = x2005 >= 3;
bool x2007;
if (x2006) {
x2007 = x2006;
} else {
x2007 = false;
}
int32_t x2012 = x2005 - 3;
int32_t x2013 = x2012 / 1;
int32_t x2014 = x2013 + 1;
int32_t x2018 = 8192 * x2014;
int32_t x2019 = x2018 * x2014;
int32_t x2015 = x2014 * x2014;
int32_t x2016 = 128 * x2015;
int32_t x2017 = 64 * x2016;
bool x2041 = x2014 >= 1;
bool x2042;
if (x2041) {
x2042 = x2041;
} else {
x2042 = false;
}
int32_t x2047 = x2013 / 1;
int32_t x2048 = x2047 + 1;
int32_t x2052 = 32768 * x2048;
int32_t x2053 = x2052 * x2048;
int32_t x2049 = x2048 * x2048;
int32_t x2050 = 512 * x2049;
int32_t x2051 = 64 * x2050;
bool x2070 = x1905 == 1;
bool x2071 = x1905 == x2048;
bool x2072 = x2070 || x2071;
bool x2073;
if (x2072) {
x2073 = x2072;
} else {
x2073 = false;
}
bool x2088 = x2048 >= 1;
bool x2089;
if (x2088) {
x2089 = x2088;
} else {
x2089 = false;
}
int32_t x2094 = x2047 / 1;
int32_t x2095 = x2094 + 1;
int32_t x2099 = 8192 * x2095;
int32_t x2100 = x2099 * x2095;
int32_t x2096 = x2095 * x2095;
int32_t x2097 = 128 * x2096;
int32_t x2098 = 64 * x2097;
int32_t x2122 = x2095 + 2;
bool x2123 = x2122 >= 3;
bool x2124;
if (x2123) {
x2124 = x2123;
} else {
x2124 = false;
}
int32_t x2129 = x2122 - 3;
int32_t x2130 = x2129 / 1;
int32_t x2131 = x2130 + 1;
int32_t x2135 = 8192 * x2131;
int32_t x2136 = x2135 * x2131;
int32_t x2132 = x2131 * x2131;
int32_t x2133 = 128 * x2132;
int32_t x2134 = 64 * x2133;
bool x2158 = x2131 >= 1;
bool x2159;
if (x2158) {
x2159 = x2158;
} else {
x2159 = false;
}
int32_t x2164 = x2130 / 1;
int32_t x2165 = x2164 + 1;
int32_t x2169 = 32768 * x2165;
int32_t x2170 = x2169 * x2165;
int32_t x2166 = x2165 * x2165;
int32_t x2167 = 512 * x2166;
int32_t x2168 = 64 * x2167;
bool x2187 = x2048 == 1;
bool x2188 = x2048 == x2165;
bool x2189 = x2187 || x2188;
bool x2190;
if (x2189) {
x2190 = x2189;
} else {
x2190 = false;
}
bool x2205 = x2165 >= 1;
bool x2206;
if (x2205) {
x2206 = x2205;
} else {
x2206 = false;
}
int32_t x2211 = x2164 / 1;
int32_t x2212 = x2211 + 1;
int32_t x2216 = 8192 * x2212;
int32_t x2217 = x2216 * x2212;
int32_t x2213 = x2212 * x2212;
int32_t x2214 = 128 * x2213;
int32_t x2215 = 64 * x2214;
int32_t x2239 = x2212 + 2;
bool x2240 = x2239 >= 3;
bool x2241;
if (x2240) {
x2241 = x2240;
} else {
x2241 = false;
}
int32_t x2246 = x2239 - 3;
int32_t x2247 = x2246 / 1;
int32_t x2248 = x2247 + 1;
int32_t x2252 = 8192 * x2248;
int32_t x2253 = x2252 * x2248;
int32_t x2249 = x2248 * x2248;
int32_t x2250 = 128 * x2249;
int32_t x2251 = 64 * x2250;
bool x2275 = x2248 >= 1;
bool x2276;
if (x2275) {
x2276 = x2275;
} else {
x2276 = false;
}
int32_t x2281 = x2247 / 1;
int32_t x2282 = x2281 + 1;
int32_t x2286 = 32768 * x2282;
int32_t x2287 = x2286 * x2282;
int32_t x2283 = x2282 * x2282;
int32_t x2284 = 512 * x2283;
int32_t x2285 = 64 * x2284;
bool x2304 = x2165 == 1;
bool x2305 = x2165 == x2282;
bool x2306 = x2304 || x2305;
bool x2307;
if (x2306) {
x2307 = x2306;
} else {
x2307 = false;
}
bool x2322 = x2282 >= 1;
bool x2323;
if (x2322) {
x2323 = x2322;
} else {
x2323 = false;
}
int32_t x2328 = x2281 / 1;
int32_t x2329 = x2328 + 1;
int32_t x2333 = 16384 * x2329;
int32_t x2334 = x2333 * x2329;
int32_t x2330 = x2329 * x2329;
int32_t x2331 = 256 * x2330;
int32_t x2332 = 64 * x2331;
int32_t x2356 = x2329 + 2;
bool x2357 = x2356 >= 3;
bool x2358;
if (x2357) {
x2358 = x2357;
} else {
x2358 = false;
}
int32_t x2363 = x2356 - 3;
int32_t x2364 = x2363 / 2;
int32_t x2365 = x2364 + 1;
int32_t x2369 = 16384 * x2365;
int32_t x2370 = x2369 * x2365;
int32_t x2366 = x2365 * x2365;
int32_t x2367 = 256 * x2366;
int32_t x2368 = 64 * x2367;
bool x2392 = x2365 >= 1;
bool x2393;
if (x2392) {
x2393 = x2392;
} else {
x2393 = false;
}
int32_t x2398 = x2364 / 1;
int32_t x2399 = x2398 + 1;
int32_t x2403 = 65536 * x2399;
int32_t x2404 = x2403 * x2399;
int32_t x2400 = x2399 * x2399;
int32_t x2401 = 1024 * x2400;
int32_t x2402 = 64 * x2401;
int32_t x2424 = x2281 / 2;
int32_t x2425 = x2424 + 1;
int32_t x2429 = 65536 * x2425;
int32_t x2430 = x2429 * x2425;
int32_t x2426 = x2425 * x2425;
int32_t x2427 = 1024 * x2426;
int32_t x2428 = 64 * x2427;
bool x2447 = x2425 == 1;
bool x2448 = x2425 == x2399;
bool x2449 = x2447 || x2448;
bool x2450;
if (x2449) {
x2450 = x2449;
} else {
x2450 = false;
}
bool x2465 = x2399 >= 1;
bool x2466;
if (x2465) {
x2466 = x2465;
} else {
x2466 = false;
}
int32_t x2471 = x2398 / 1;
int32_t x2472 = x2471 + 1;
int32_t x2476 = 16384 * x2472;
int32_t x2477 = x2476 * x2472;
int32_t x2473 = x2472 * x2472;
int32_t x2474 = 256 * x2473;
int32_t x2475 = 64 * x2474;
int32_t x2499 = x2472 + 2;
bool x2500 = x2499 >= 3;
bool x2501;
if (x2500) {
x2501 = x2500;
} else {
x2501 = false;
}
int32_t x2506 = x2499 - 3;
int32_t x2507 = x2506 / 1;
int32_t x2508 = x2507 + 1;
int32_t x2512 = 16384 * x2508;
int32_t x2513 = x2512 * x2508;
int32_t x2509 = x2508 * x2508;
int32_t x2510 = 256 * x2509;
int32_t x2511 = 64 * x2510;
bool x2535 = x2508 >= 1;
bool x2536;
if (x2535) {
x2536 = x2535;
} else {
x2536 = false;
}
int32_t x2541 = x2507 / 1;
int32_t x2542 = x2541 + 1;
int32_t x2546 = 65536 * x2542;
int32_t x2547 = x2546 * x2542;
int32_t x2543 = x2542 * x2542;
int32_t x2544 = 1024 * x2543;
int32_t x2545 = 64 * x2544;
bool x2564 = x2399 == 1;
bool x2565 = x2399 == x2542;
bool x2566 = x2564 || x2565;
bool x2567;
if (x2566) {
x2567 = x2566;
} else {
x2567 = false;
}
bool x2582 = x2542 >= 1;
bool x2583;
if (x2582) {
x2583 = x2582;
} else {
x2583 = false;
}
int32_t x2588 = x2541 / 1;
int32_t x2589 = x2588 + 1;
int32_t x2593 = 16384 * x2589;
int32_t x2594 = x2593 * x2589;
int32_t x2590 = x2589 * x2589;
int32_t x2591 = 256 * x2590;
int32_t x2592 = 64 * x2591;
int32_t x2616 = x2589 + 2;
bool x2617 = x2616 >= 3;
bool x2618;
if (x2617) {
x2618 = x2617;
} else {
x2618 = false;
}
int32_t x2623 = x2616 - 3;
int32_t x2624 = x2623 / 1;
int32_t x2625 = x2624 + 1;
int32_t x2629 = 16384 * x2625;
int32_t x2630 = x2629 * x2625;
int32_t x2626 = x2625 * x2625;
int32_t x2627 = 256 * x2626;
int32_t x2628 = 64 * x2627;
bool x2652 = x2625 >= 1;
bool x2653;
if (x2652) {
x2653 = x2652;
} else {
x2653 = false;
}
int32_t x2658 = x2624 / 1;
int32_t x2659 = x2658 + 1;
int32_t x2663 = 65536 * x2659;
int32_t x2664 = x2663 * x2659;
int32_t x2660 = x2659 * x2659;
int32_t x2661 = 1024 * x2660;
int32_t x2662 = 64 * x2661;
bool x2681 = x2542 == 1;
bool x2682 = x2542 == x2659;
bool x2683 = x2681 || x2682;
bool x2684;
if (x2683) {
x2684 = x2683;
} else {
x2684 = false;
}
bool x2699 = x2659 >= 1;
bool x2700;
if (x2699) {
x2700 = x2699;
} else {
x2700 = false;
}
int32_t x2705 = x2658 / 1;
int32_t x2706 = x2705 + 1;
int32_t x2710 = 16384 * x2706;
int32_t x2711 = x2710 * x2706;
int32_t x2707 = x2706 * x2706;
int32_t x2708 = 256 * x2707;
int32_t x2709 = 64 * x2708;
int32_t x2733 = x2706 + 2;
bool x2734 = x2733 >= 3;
bool x2735;
if (x2734) {
x2735 = x2734;
} else {
x2735 = false;
}
int32_t x2740 = x2733 - 3;
int32_t x2741 = x2740 / 1;
int32_t x2742 = x2741 + 1;
int32_t x2746 = 16384 * x2742;
int32_t x2747 = x2746 * x2742;
int32_t x2743 = x2742 * x2742;
int32_t x2744 = 256 * x2743;
int32_t x2745 = 64 * x2744;
bool x2769 = x2742 >= 1;
bool x2770;
if (x2769) {
x2770 = x2769;
} else {
x2770 = false;
}
int32_t x2775 = x2741 / 1;
int32_t x2776 = x2775 + 1;
int32_t x2780 = 65536 * x2776;
int32_t x2781 = x2780 * x2776;
int32_t x2777 = x2776 * x2776;
int32_t x2778 = 1024 * x2777;
int32_t x2779 = 64 * x2778;
bool x2798 = x2659 == 1;
bool x2799 = x2659 == x2776;
bool x2800 = x2798 || x2799;
bool x2801;
if (x2800) {
x2801 = x2800;
} else {
x2801 = false;
}
bool x2816 = x2776 >= 1;
bool x2817;
if (x2816) {
x2817 = x2816;
} else {
x2817 = false;
}
int32_t x2822 = x2775 / 1;
int32_t x2823 = x2822 + 1;
int32_t x2827 = 16384 * x2823;
int32_t x2828 = x2827 * x2823;
int32_t x2824 = x2823 * x2823;
int32_t x2825 = 256 * x2824;
int32_t x2826 = 64 * x2825;
int32_t x2850 = x2823 + 2;
bool x2851 = x2850 >= 3;
bool x2852;
if (x2851) {
x2852 = x2851;
} else {
x2852 = false;
}
int32_t x2857 = x2850 - 3;
int32_t x2858 = x2857 / 1;
int32_t x2859 = x2858 + 1;
int32_t x2863 = 16384 * x2859;
int32_t x2864 = x2863 * x2859;
int32_t x2860 = x2859 * x2859;
int32_t x2861 = 256 * x2860;
int32_t x2862 = 64 * x2861;
bool x2886 = x2859 >= 1;
bool x2887;
if (x2886) {
x2887 = x2886;
} else {
x2887 = false;
}
int32_t x2892 = x2858 / 1;
int32_t x2893 = x2892 + 1;
int32_t x2897 = 65536 * x2893;
int32_t x2898 = x2897 * x2893;
int32_t x2894 = x2893 * x2893;
int32_t x2895 = 1024 * x2894;
int32_t x2896 = 64 * x2895;
bool x2915 = x2776 == 1;
bool x2916 = x2776 == x2893;
bool x2917 = x2915 || x2916;
bool x2918;
if (x2917) {
x2918 = x2917;
} else {
x2918 = false;
}
bool x2933 = x2893 >= 1;
bool x2934;
if (x2933) {
x2934 = x2933;
} else {
x2934 = false;
}
int32_t x2939 = x2892 / 1;
int32_t x2940 = x2939 + 1;
int32_t x2944 = 16384 * x2940;
int32_t x2945 = x2944 * x2940;
int32_t x2941 = x2940 * x2940;
int32_t x2942 = 256 * x2941;
int32_t x2943 = 64 * x2942;
int32_t x2967 = x2940 + 2;
bool x2968 = x2967 >= 3;
bool x2969;
if (x2968) {
x2969 = x2968;
} else {
x2969 = false;
}
int32_t x2974 = x2967 - 3;
int32_t x2975 = x2974 / 1;
int32_t x2976 = x2975 + 1;
int32_t x2980 = 16384 * x2976;
int32_t x2981 = x2980 * x2976;
int32_t x2977 = x2976 * x2976;
int32_t x2978 = 256 * x2977;
int32_t x2979 = 64 * x2978;
bool x3003 = x2976 >= 1;
bool x3004;
if (x3003) {
x3004 = x3003;
} else {
x3004 = false;
}
int32_t x3009 = x2975 / 1;
int32_t x3010 = x3009 + 1;
int32_t x3014 = 65536 * x3010;
int32_t x3015 = x3014 * x3010;
int32_t x3011 = x3010 * x3010;
int32_t x3012 = 1024 * x3011;
int32_t x3013 = 64 * x3012;
bool x3032 = x2893 == 1;
bool x3033 = x2893 == x3010;
bool x3034 = x3032 || x3033;
bool x3035;
if (x3034) {
x3035 = x3034;
} else {
x3035 = false;
}
bool x3050 = x3010 >= 1;
bool x3051;
if (x3050) {
x3051 = x3050;
} else {
x3051 = false;
}
int32_t x3056 = x3009 / 1;
int32_t x3057 = x3056 + 1;
int32_t x3061 = 32768 * x3057;
int32_t x3062 = x3061 * x3057;
int32_t x3058 = x3057 * x3057;
int32_t x3059 = 512 * x3058;
int32_t x3060 = 64 * x3059;
int32_t x3084 = x3057 + 2;
bool x3085 = x3084 >= 3;
bool x3086;
if (x3085) {
x3086 = x3085;
} else {
x3086 = false;
}
int32_t x3091 = x3084 - 3;
int32_t x3092 = x3091 / 2;
int32_t x3093 = x3092 + 1;
int32_t x3097 = 32768 * x3093;
int32_t x3098 = x3097 * x3093;
int32_t x3094 = x3093 * x3093;
int32_t x3095 = 512 * x3094;
int32_t x3096 = 64 * x3095;
bool x3120 = x3093 >= 1;
bool x3121;
if (x3120) {
x3121 = x3120;
} else {
x3121 = false;
}
int32_t x3126 = x3092 / 1;
int32_t x3127 = x3126 + 1;
int32_t x3131 = 131072 * x3127;
int32_t x3132 = x3131 * x3127;
int32_t x3128 = x3127 * x3127;
int32_t x3129 = 2048 * x3128;
int32_t x3130 = 64 * x3129;
int32_t x3152 = x3009 / 2;
int32_t x3153 = x3152 + 1;
int32_t x3157 = 131072 * x3153;
int32_t x3158 = x3157 * x3153;
int32_t x3154 = x3153 * x3153;
int32_t x3155 = 2048 * x3154;
int32_t x3156 = 64 * x3155;
bool x3175 = x3153 == 1;
bool x3176 = x3153 == x3127;
bool x3177 = x3175 || x3176;
bool x3178;
if (x3177) {
x3178 = x3177;
} else {
x3178 = false;
}
bool x3193 = x3127 >= 1;
bool x3194;
if (x3193) {
x3194 = x3193;
} else {
x3194 = false;
}
int32_t x3199 = x3126 / 1;
int32_t x3200 = x3199 + 1;
int32_t x3204 = 32768 * x3200;
int32_t x3205 = x3204 * x3200;
int32_t x3201 = x3200 * x3200;
int32_t x3202 = 512 * x3201;
int32_t x3203 = 64 * x3202;
int32_t x3227 = x3200 + 2;
bool x3228 = x3227 >= 3;
bool x3229;
if (x3228) {
x3229 = x3228;
} else {
x3229 = false;
}
int32_t x3234 = x3227 - 3;
int32_t x3235 = x3234 / 1;
int32_t x3236 = x3235 + 1;
int32_t x3240 = 32768 * x3236;
int32_t x3241 = x3240 * x3236;
int32_t x3237 = x3236 * x3236;
int32_t x3238 = 512 * x3237;
int32_t x3239 = 64 * x3238;
bool x3263 = x3236 >= 1;
bool x3264;
if (x3263) {
x3264 = x3263;
} else {
x3264 = false;
}
int32_t x3269 = x3235 / 1;
int32_t x3270 = x3269 + 1;
int32_t x3274 = 131072 * x3270;
int32_t x3275 = x3274 * x3270;
int32_t x3271 = x3270 * x3270;
int32_t x3272 = 2048 * x3271;
int32_t x3273 = 64 * x3272;
bool x3292 = x3127 == 1;
bool x3293 = x3127 == x3270;
bool x3294 = x3292 || x3293;
bool x3295;
if (x3294) {
x3295 = x3294;
} else {
x3295 = false;
}
bool x3310 = x3270 >= 1;
bool x3311;
if (x3310) {
x3311 = x3310;
} else {
x3311 = false;
}
int32_t x3316 = x3269 / 1;
int32_t x3317 = x3316 + 1;
int32_t x3321 = 32768 * x3317;
int32_t x3322 = x3321 * x3317;
int32_t x3318 = x3317 * x3317;
int32_t x3319 = 512 * x3318;
int32_t x3320 = 64 * x3319;
int32_t x3344 = x3317 + 2;
bool x3345 = x3344 >= 3;
bool x3346;
if (x3345) {
x3346 = x3345;
} else {
x3346 = false;
}
int32_t x3351 = x3344 - 3;
int32_t x3352 = x3351 / 1;
int32_t x3353 = x3352 + 1;
int32_t x3357 = 32768 * x3353;
int32_t x3358 = x3357 * x3353;
int32_t x3354 = x3353 * x3353;
int32_t x3355 = 512 * x3354;
int32_t x3356 = 64 * x3355;
bool x3380 = x3353 >= 1;
bool x3381;
if (x3380) {
x3381 = x3380;
} else {
x3381 = false;
}
int32_t x3386 = x3352 / 1;
int32_t x3387 = x3386 + 1;
int32_t x3391 = 131072 * x3387;
int32_t x3392 = x3391 * x3387;
int32_t x3388 = x3387 * x3387;
int32_t x3389 = 2048 * x3388;
int32_t x3390 = 64 * x3389;
bool x3409 = x3270 == 1;
bool x3410 = x3270 == x3387;
bool x3411 = x3409 || x3410;
bool x3412;
if (x3411) {
x3412 = x3411;
} else {
x3412 = false;
}
bool x3427 = x3387 >= 2;
bool x3428;
if (x3427) {
x3428 = x3427;
} else {
x3428 = false;
}
int32_t x3437 = x3387 - 2;
int32_t x3438 = x3437 / 1;
int32_t x3439 = x3438 + 1;
int32_t x3443 = 131072 * x3439;
int32_t x3444 = x3443 * x3439;
int32_t x3440 = x3439 * x3439;
int32_t x3441 = 2048 * x3440;
int32_t x3442 = 64 * x3441;
bool x3627 = true || false;
bool x3629;
if (x3627) {
bool x3628 = true || true;
x3629 = x3628;
} else {
x3629 = false;
}
bool x3630;
if (x3629) {
bool x3628 = true || true;
x3630 = x3628;
} else {
x3630 = false;
}
bool x3631;
if (x3630) {
bool x3628 = true || true;
x3631 = x3628;
} else {
x3631 = false;
}
float x3626 = 1.0f / 64.0f;
bool x3751 = x3387 == x3270;
bool x3752;
if (x3751) {
x3752 = x3751;
} else {
x3752 = false;
}
bool x3753 = x3387 == 1;
bool x3754 = x3753 || x3751;
bool x3755;
if (x3754) {
x3755 = x3754;
} else {
x3755 = false;
}
bool x3822 = x3270 == x3127;
bool x3823;
if (x3822) {
x3823 = x3822;
} else {
x3823 = false;
}
bool x3824 = x3409 || x3822;
bool x3825;
if (x3824) {
x3825 = x3824;
} else {
x3825 = false;
}
bool x3892 = x3127 == x3153;
bool x3893;
if (x3892) {
x3893 = x3892;
} else {
x3893 = false;
}
bool x3894 = x3292 || x3892;
bool x3895;
if (x3894) {
x3895 = x3894;
} else {
x3895 = false;
}
bool x3974 = x3010 == x2893;
bool x3975;
if (x3974) {
x3975 = x3974;
} else {
x3975 = false;
}
bool x3976 = x3010 == 1;
bool x3977 = x3976 || x3974;
bool x3978;
if (x3977) {
x3978 = x3977;
} else {
x3978 = false;
}
bool x4045 = x2893 == x2776;
bool x4046;
if (x4045) {
x4046 = x4045;
} else {
x4046 = false;
}
bool x4047 = x3032 || x4045;
bool x4048;
if (x4047) {
x4048 = x4047;
} else {
x4048 = false;
}
bool x4115 = x2776 == x2659;
bool x4116;
if (x4115) {
x4116 = x4115;
} else {
x4116 = false;
}
bool x4117 = x2915 || x4115;
bool x4118;
if (x4117) {
x4118 = x4117;
} else {
x4118 = false;
}
bool x4185 = x2659 == x2542;
bool x4186;
if (x4185) {
x4186 = x4185;
} else {
x4186 = false;
}
bool x4187 = x2798 || x4185;
bool x4188;
if (x4187) {
x4188 = x4187;
} else {
x4188 = false;
}
bool x4255 = x2542 == x2399;
bool x4256;
if (x4255) {
x4256 = x4255;
} else {
x4256 = false;
}
bool x4257 = x2681 || x4255;
bool x4258;
if (x4257) {
x4258 = x4257;
} else {
x4258 = false;
}
bool x4325 = x2399 == x2425;
bool x4326;
if (x4325) {
x4326 = x4325;
} else {
x4326 = false;
}
bool x4327 = x2564 || x4325;
bool x4328;
if (x4327) {
x4328 = x4327;
} else {
x4328 = false;
}
bool x4407 = x2282 == x2165;
bool x4408;
if (x4407) {
x4408 = x4407;
} else {
x4408 = false;
}
bool x4409 = x2282 == 1;
bool x4410 = x4409 || x4407;
bool x4411;
if (x4410) {
x4411 = x4410;
} else {
x4411 = false;
}
bool x4478 = x2165 == x2048;
bool x4479;
if (x4478) {
x4479 = x4478;
} else {
x4479 = false;
}
bool x4480 = x2304 || x4478;
bool x4481;
if (x4480) {
x4481 = x4480;
} else {
x4481 = false;
}
bool x4548 = x2048 == x1905;
bool x4549;
if (x4548) {
x4549 = x4548;
} else {
x4549 = false;
}
bool x4550 = x2187 || x4548;
bool x4551;
if (x4550) {
x4551 = x4550;
} else {
x4551 = false;
}
bool x4618 = x1905 == x1931;
bool x4619;
if (x4618) {
x4619 = x4618;
} else {
x4619 = false;
}
bool x4620 = x2070 || x4618;
bool x4621;
if (x4620) {
x4621 = x4620;
} else {
x4621 = false;
}
bool x4700 = x1788 == x1671;
bool x4701;
if (x4700) {
x4701 = x4700;
} else {
x4701 = false;
}
bool x4702 = x1788 == 1;
bool x4703 = x4702 || x4700;
bool x4704;
if (x4703) {
x4704 = x4703;
} else {
x4704 = false;
}
bool x4771 = x1671 == x1531;
bool x4772;
if (x4771) {
x4772 = x4771;
} else {
x4772 = false;
}
bool x4773 = x1810 || x4771;
bool x4774;
if (x4773) {
x4774 = x4773;
} else {
x4774 = false;
}
bool x4841 = x1531 == x1461;
bool x4842;
if (x4841) {
x4842 = x4841;
} else {
x4842 = false;
}
bool x4843 = x1693 || x4841;
bool x4844;
if (x4843) {
x4844 = x4843;
} else {
x4844 = false;
}
int32_t x6542 = x1394 / 10;
double x6547 = (double)x11;
int64_t x6573 = (int64_t)x11;
float x6577 = (float)x11;
for(int x1386=0; x1386 < 4; x1386++) {
struct timeval begin_1, end_1, diff_1;
float x1388 = 0.0f;
float x1389 = x1388;
float x1390 = x1389;
int32_t x1391 = x1386 + 1;
printf("Start training epoch %d\n",x1391);
gettimeofday(&begin_1, NULL);
for(int x1396=0; x1396 < x1394; x1396++) {
int32_t x1397 = x1396 * 64;
int32_t x1398 = x1397 * 3072;
float* x1399 = x13+x1398;
int* x1400 = x14+x1397;
// Tensor 'toGPU' invocation.
float* x1402 = (float*)myGpuMalloc(196608 * sizeof(float));
CUDA_CALL(cudaMemcpy(x1402, x1399, 196608 * sizeof(float), cudaMemcpyHostToDevice));
float* x1404 = (float*)myGpuMalloc(2 * sizeof(float));
int* x1405 = (int32_t*)myGpuMalloc(64 * sizeof(int32_t));
CUDA_CALL(cudaMemcpy(x1405, x1400, 64 * sizeof(int32_t), cudaMemcpyHostToDevice));
float* x1407 = (float*)myGpuMalloc(1 * sizeof(float));
float* x1408 = (float*)myGpuMalloc(1 * sizeof(float));
// allocate memory to save the final loss in CPU Tensor
float* x1410 = (float*)myMalloc(1 * sizeof(float));;
float* x1418 = (float*)myGpuMalloc(x1417 * sizeof(float));
float* x1419 = (float*)myMalloc(1 * sizeof(float));;
x1419[0] = 0.0f;
float* x1421 = (float*)myMalloc(1 * sizeof(float));;
x1421[0] = 1.0f;

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
    64, 64, x1412, x1412));

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
    x1421, in_desc, x1402, filt_desc, x751,
    conv_desc, algo, ws_data, ws_size,
    x1419, out_desc, x1418));
};
float* x1424 = (float*)myGpuMalloc(x1417 * sizeof(float));
float* x1425 = (float*)myGpuMalloc(x1415 * sizeof(float));
float* x1426 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1427 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1428 = (float*)myMalloc(1 * sizeof(float));;
x1428[0] = 0.0f;
float* x1430 = (float*)myMalloc(1 * sizeof(float));;
x1430[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1412, x1412));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1412, x1412));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1430, x1428, in_desc, x1418, out_desc, x1425, sbmv_desc, x913,
    x1048, 0.1, x415, x625, 1.0E-5,
    x1426, x1427));
};
float* x1433 = (float*)myGpuMalloc(x1417 * sizeof(float));
float* x1434 = (float*)myMalloc(1 * sizeof(float));;
x1434[0] = 0.0f;
float* x1436 = (float*)myMalloc(1 * sizeof(float));;
x1436[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1412, x1412));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1436, x_desc, x1425, x1434, x_desc, x1425));
};
float* x1439 = (float*)myMalloc(1 * sizeof(float));;
x1439[0] = 0.0f;
float* x1441 = (float*)myMalloc(1 * sizeof(float));;
x1441[0] = 1.0f;
float* x1451 = (float*)myGpuMalloc(x1450 * sizeof(float));

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1412, x1412) );

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1445, x1445));

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
    x1441, in_desc, x1425, x1439, out_desc, x1451));
};
float* x1453 = (float*)myGpuMalloc(x1450 * sizeof(float));
if (x1455) {
} else {
assert(false && "ERROR not specified");
}
float* x1467 = (float*)myGpuMalloc(x1466 * sizeof(float));
float* x1468 = (float*)myMalloc(1 * sizeof(float));;
x1468[0] = 0.0f;
float* x1470 = (float*)myMalloc(1 * sizeof(float));;
x1470[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1445, x1445));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1461, x1461));

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
    x1470, in_desc, x1451, filt_desc, x994,
    conv_desc, algo, ws_data, ws_size,
    x1468, out_desc, x1467));
};
float* x1473 = (float*)myGpuMalloc(x1466 * sizeof(float));
float* x1474 = (float*)myGpuMalloc(x1464 * sizeof(float));
float* x1475 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1476 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1477 = (float*)myMalloc(1 * sizeof(float));;
x1477[0] = 0.0f;
float* x1479 = (float*)myMalloc(1 * sizeof(float));;
x1479[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1461, x1461));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1461, x1461));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1479, x1477, in_desc, x1467, out_desc, x1474, sbmv_desc, x373,
    x454, 0.1, x637, x448, 1.0E-5,
    x1475, x1476));
};
float* x1482 = (float*)myGpuMalloc(x1466 * sizeof(float));
float* x1483 = (float*)myMalloc(1 * sizeof(float));;
x1483[0] = 0.0f;
float* x1485 = (float*)myMalloc(1 * sizeof(float));;
x1485[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1461, x1461));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1485, x_desc, x1474, x1483, x_desc, x1474));
};
if (x1490) {
} else {
assert(false && "ERROR not specified");
}
float* x1503 = (float*)myGpuMalloc(x1502 * sizeof(float));
float* x1504 = (float*)myMalloc(1 * sizeof(float));;
x1504[0] = 0.0f;
float* x1506 = (float*)myMalloc(1 * sizeof(float));;
x1506[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1461, x1461));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1497, x1497));

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
    x1506, in_desc, x1474, filt_desc, x565,
    conv_desc, algo, ws_data, ws_size,
    x1504, out_desc, x1503));
};
float* x1509 = (float*)myGpuMalloc(x1502 * sizeof(float));
float* x1510 = (float*)myGpuMalloc(x1500 * sizeof(float));
float* x1511 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1512 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1513 = (float*)myMalloc(1 * sizeof(float));;
x1513[0] = 0.0f;
float* x1515 = (float*)myMalloc(1 * sizeof(float));;
x1515[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1497, x1497));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1497, x1497));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1515, x1513, in_desc, x1503, out_desc, x1510, sbmv_desc, x787,
    x442, 0.1, x610, x769, 1.0E-5,
    x1511, x1512));
};
float* x1518 = (float*)myGpuMalloc(x1502 * sizeof(float));
float* x1519 = (float*)myMalloc(1 * sizeof(float));;
x1519[0] = 0.0f;
float* x1521 = (float*)myMalloc(1 * sizeof(float));;
x1521[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1497, x1497));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1521, x_desc, x1510, x1519, x_desc, x1510));
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
    64, 64, x1497, x1497));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1531, x1531));

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
    x1540, in_desc, x1510, filt_desc, x391,
    conv_desc, algo, ws_data, ws_size,
    x1538, out_desc, x1537));
};
float* x1543 = (float*)myGpuMalloc(x1536 * sizeof(float));
float* x1544 = (float*)myGpuMalloc(x1534 * sizeof(float));
float* x1545 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1546 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1547 = (float*)myMalloc(1 * sizeof(float));;
x1547[0] = 0.0f;
float* x1549 = (float*)myMalloc(1 * sizeof(float));;
x1549[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1531, x1531));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1531, x1531));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1549, x1547, in_desc, x1537, out_desc, x1544, sbmv_desc, x892,
    x673, 0.1, x508, x403, 1.0E-5,
    x1545, x1546));
};
float* x1552 = (float*)myGpuMalloc(x1536 * sizeof(float));
if (x1455) {
} else {
assert(false && "ERROR not specified");
}
float* x1560 = (float*)myGpuMalloc(x1559 * sizeof(float));
float* x1561 = (float*)myMalloc(1 * sizeof(float));;
x1561[0] = 0.0f;
float* x1563 = (float*)myMalloc(1 * sizeof(float));;
x1563[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1445, x1445));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1461, x1461));

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
    x1563, in_desc, x1451, filt_desc, x781,
    conv_desc, algo, ws_data, ws_size,
    x1561, out_desc, x1560));
};
float* x1566 = (float*)myGpuMalloc(x1559 * sizeof(float));
float* x1567 = (float*)myGpuMalloc(x1557 * sizeof(float));
float* x1568 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1569 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1570 = (float*)myMalloc(1 * sizeof(float));;
x1570[0] = 0.0f;
float* x1572 = (float*)myMalloc(1 * sizeof(float));;
x1572[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1461, x1461));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1461, x1461));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1572, x1570, in_desc, x1560, out_desc, x1567, sbmv_desc, x523,
    x904, 0.1, x1087, x1024, 1.0E-5,
    x1568, x1569));
};
float* x1575 = (float*)myGpuMalloc(x1559 * sizeof(float));
if (x1579) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1461) x Sym(1461), res:  x Const(64) x Const(256) x Sym(1531) x Sym(1531)");
}
float* x1584 = (float*)myMalloc(1 * sizeof(float));;
x1584[0] = 1.0f;
float* x1586 = (float*)myMalloc(1 * sizeof(float));;
x1586[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1461, x1461));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1531, x1531));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1584, bias_desc, x1567, x1586, out_desc, x1544));
};
float* x1589 = (float*)myMalloc(1 * sizeof(float));;
x1589[0] = 0.0f;
float* x1591 = (float*)myMalloc(1 * sizeof(float));;
x1591[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1531, x1531));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1591, x_desc, x1544, x1589, x_desc, x1544));
};
if (x1595) {
} else {
assert(false && "ERROR not specified");
}
float* x1607 = (float*)myGpuMalloc(x1606 * sizeof(float));
float* x1608 = (float*)myMalloc(1 * sizeof(float));;
x1608[0] = 0.0f;
float* x1610 = (float*)myMalloc(1 * sizeof(float));;
x1610[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1531, x1531));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1601, x1601));

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
    x1610, in_desc, x1544, filt_desc, x808,
    conv_desc, algo, ws_data, ws_size,
    x1608, out_desc, x1607));
};
float* x1613 = (float*)myGpuMalloc(x1606 * sizeof(float));
float* x1614 = (float*)myGpuMalloc(x1604 * sizeof(float));
float* x1615 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1616 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1617 = (float*)myMalloc(1 * sizeof(float));;
x1617[0] = 0.0f;
float* x1619 = (float*)myMalloc(1 * sizeof(float));;
x1619[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1601, x1601));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1601, x1601));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1619, x1617, in_desc, x1607, out_desc, x1614, sbmv_desc, x721,
    x475, 0.1, x325, x601, 1.0E-5,
    x1615, x1616));
};
float* x1622 = (float*)myGpuMalloc(x1606 * sizeof(float));
float* x1623 = (float*)myMalloc(1 * sizeof(float));;
x1623[0] = 0.0f;
float* x1625 = (float*)myMalloc(1 * sizeof(float));;
x1625[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1601, x1601));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1625, x_desc, x1614, x1623, x_desc, x1614));
};
if (x1630) {
} else {
assert(false && "ERROR not specified");
}
float* x1643 = (float*)myGpuMalloc(x1642 * sizeof(float));
float* x1644 = (float*)myMalloc(1 * sizeof(float));;
x1644[0] = 0.0f;
float* x1646 = (float*)myMalloc(1 * sizeof(float));;
x1646[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1601, x1601));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1637, x1637));

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
    x1646, in_desc, x1614, filt_desc, x544,
    conv_desc, algo, ws_data, ws_size,
    x1644, out_desc, x1643));
};
float* x1649 = (float*)myGpuMalloc(x1642 * sizeof(float));
float* x1650 = (float*)myGpuMalloc(x1640 * sizeof(float));
float* x1651 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1652 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1653 = (float*)myMalloc(1 * sizeof(float));;
x1653[0] = 0.0f;
float* x1655 = (float*)myMalloc(1 * sizeof(float));;
x1655[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1637, x1637));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1637, x1637));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1655, x1653, in_desc, x1643, out_desc, x1650, sbmv_desc, x919,
    x754, 0.1, x427, x1027, 1.0E-5,
    x1651, x1652));
};
float* x1658 = (float*)myGpuMalloc(x1642 * sizeof(float));
float* x1659 = (float*)myMalloc(1 * sizeof(float));;
x1659[0] = 0.0f;
float* x1661 = (float*)myMalloc(1 * sizeof(float));;
x1661[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1637, x1637));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1661, x_desc, x1650, x1659, x_desc, x1650));
};
if (x1665) {
} else {
assert(false && "ERROR not specified");
}
float* x1677 = (float*)myGpuMalloc(x1676 * sizeof(float));
float* x1678 = (float*)myMalloc(1 * sizeof(float));;
x1678[0] = 0.0f;
float* x1680 = (float*)myMalloc(1 * sizeof(float));;
x1680[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1637, x1637));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1671, x1671));

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
    x1680, in_desc, x1650, filt_desc, x685,
    conv_desc, algo, ws_data, ws_size,
    x1678, out_desc, x1677));
};
float* x1683 = (float*)myGpuMalloc(x1676 * sizeof(float));
float* x1684 = (float*)myGpuMalloc(x1674 * sizeof(float));
float* x1685 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1686 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1687 = (float*)myMalloc(1 * sizeof(float));;
x1687[0] = 0.0f;
float* x1689 = (float*)myMalloc(1 * sizeof(float));;
x1689[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1671, x1671));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1671, x1671));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1689, x1687, in_desc, x1677, out_desc, x1684, sbmv_desc, x469,
    x316, 0.1, x568, x793, 1.0E-5,
    x1685, x1686));
};
float* x1692 = (float*)myGpuMalloc(x1676 * sizeof(float));
if (x1696) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1531) x Sym(1531), res:  x Const(64) x Const(256) x Sym(1671) x Sym(1671)");
}
float* x1701 = (float*)myMalloc(1 * sizeof(float));;
x1701[0] = 1.0f;
float* x1703 = (float*)myMalloc(1 * sizeof(float));;
x1703[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1531, x1531));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1671, x1671));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1701, bias_desc, x1544, x1703, out_desc, x1684));
};
float* x1706 = (float*)myMalloc(1 * sizeof(float));;
x1706[0] = 0.0f;
float* x1708 = (float*)myMalloc(1 * sizeof(float));;
x1708[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1671, x1671));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1708, x_desc, x1684, x1706, x_desc, x1684));
};
if (x1712) {
} else {
assert(false && "ERROR not specified");
}
float* x1724 = (float*)myGpuMalloc(x1723 * sizeof(float));
float* x1725 = (float*)myMalloc(1 * sizeof(float));;
x1725[0] = 0.0f;
float* x1727 = (float*)myMalloc(1 * sizeof(float));;
x1727[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1671, x1671));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1718, x1718));

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
    x1727, in_desc, x1684, filt_desc, x745,
    conv_desc, algo, ws_data, ws_size,
    x1725, out_desc, x1724));
};
float* x1730 = (float*)myGpuMalloc(x1723 * sizeof(float));
float* x1731 = (float*)myGpuMalloc(x1721 * sizeof(float));
float* x1732 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1733 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1734 = (float*)myMalloc(1 * sizeof(float));;
x1734[0] = 0.0f;
float* x1736 = (float*)myMalloc(1 * sizeof(float));;
x1736[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1718, x1718));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1718, x1718));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1736, x1734, in_desc, x1724, out_desc, x1731, sbmv_desc, x538,
    x367, 0.1, x1066, x856, 1.0E-5,
    x1732, x1733));
};
float* x1739 = (float*)myGpuMalloc(x1723 * sizeof(float));
float* x1740 = (float*)myMalloc(1 * sizeof(float));;
x1740[0] = 0.0f;
float* x1742 = (float*)myMalloc(1 * sizeof(float));;
x1742[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1718, x1718));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1742, x_desc, x1731, x1740, x_desc, x1731));
};
if (x1747) {
} else {
assert(false && "ERROR not specified");
}
float* x1760 = (float*)myGpuMalloc(x1759 * sizeof(float));
float* x1761 = (float*)myMalloc(1 * sizeof(float));;
x1761[0] = 0.0f;
float* x1763 = (float*)myMalloc(1 * sizeof(float));;
x1763[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1718, x1718));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1754, x1754));

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
    x1763, in_desc, x1731, filt_desc, x514,
    conv_desc, algo, ws_data, ws_size,
    x1761, out_desc, x1760));
};
float* x1766 = (float*)myGpuMalloc(x1759 * sizeof(float));
float* x1767 = (float*)myGpuMalloc(x1757 * sizeof(float));
float* x1768 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1769 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1770 = (float*)myMalloc(1 * sizeof(float));;
x1770[0] = 0.0f;
float* x1772 = (float*)myMalloc(1 * sizeof(float));;
x1772[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1754, x1754));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1754, x1754));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1772, x1770, in_desc, x1760, out_desc, x1767, sbmv_desc, x511,
    x700, 0.1, x832, x649, 1.0E-5,
    x1768, x1769));
};
float* x1775 = (float*)myGpuMalloc(x1759 * sizeof(float));
float* x1776 = (float*)myMalloc(1 * sizeof(float));;
x1776[0] = 0.0f;
float* x1778 = (float*)myMalloc(1 * sizeof(float));;
x1778[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1754, x1754));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1778, x_desc, x1767, x1776, x_desc, x1767));
};
if (x1782) {
} else {
assert(false && "ERROR not specified");
}
float* x1794 = (float*)myGpuMalloc(x1793 * sizeof(float));
float* x1795 = (float*)myMalloc(1 * sizeof(float));;
x1795[0] = 0.0f;
float* x1797 = (float*)myMalloc(1 * sizeof(float));;
x1797[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1754, x1754));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1788, x1788));

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
    x1797, in_desc, x1767, filt_desc, x556,
    conv_desc, algo, ws_data, ws_size,
    x1795, out_desc, x1794));
};
float* x1800 = (float*)myGpuMalloc(x1793 * sizeof(float));
float* x1801 = (float*)myGpuMalloc(x1791 * sizeof(float));
float* x1802 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1803 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1804 = (float*)myMalloc(1 * sizeof(float));;
x1804[0] = 0.0f;
float* x1806 = (float*)myMalloc(1 * sizeof(float));;
x1806[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1788, x1788));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1788, x1788));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1806, x1804, in_desc, x1794, out_desc, x1801, sbmv_desc, x406,
    x1036, 0.1, x847, x694, 1.0E-5,
    x1802, x1803));
};
float* x1809 = (float*)myGpuMalloc(x1793 * sizeof(float));
if (x1813) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1671) x Sym(1671), res:  x Const(64) x Const(256) x Sym(1788) x Sym(1788)");
}
float* x1818 = (float*)myMalloc(1 * sizeof(float));;
x1818[0] = 1.0f;
float* x1820 = (float*)myMalloc(1 * sizeof(float));;
x1820[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1671, x1671));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1788, x1788));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1818, bias_desc, x1684, x1820, out_desc, x1801));
};
float* x1823 = (float*)myMalloc(1 * sizeof(float));;
x1823[0] = 0.0f;
float* x1825 = (float*)myMalloc(1 * sizeof(float));;
x1825[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1788, x1788));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1825, x_desc, x1801, x1823, x_desc, x1801));
};
if (x1829) {
} else {
assert(false && "ERROR not specified");
}
float* x1841 = (float*)myGpuMalloc(x1840 * sizeof(float));
float* x1842 = (float*)myMalloc(1 * sizeof(float));;
x1842[0] = 0.0f;
float* x1844 = (float*)myMalloc(1 * sizeof(float));;
x1844[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1788, x1788));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1835, x1835));

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
    x1844, in_desc, x1801, filt_desc, x328,
    conv_desc, algo, ws_data, ws_size,
    x1842, out_desc, x1841));
};
float* x1847 = (float*)myGpuMalloc(x1840 * sizeof(float));
float* x1848 = (float*)myGpuMalloc(x1838 * sizeof(float));
float* x1849 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1850 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1851 = (float*)myMalloc(1 * sizeof(float));;
x1851[0] = 0.0f;
float* x1853 = (float*)myMalloc(1 * sizeof(float));;
x1853[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1835, x1835));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1835, x1835));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1853, x1851, in_desc, x1841, out_desc, x1848, sbmv_desc, x547,
    x811, 0.1, x907, x697, 1.0E-5,
    x1849, x1850));
};
float* x1856 = (float*)myGpuMalloc(x1840 * sizeof(float));
float* x1857 = (float*)myMalloc(1 * sizeof(float));;
x1857[0] = 0.0f;
float* x1859 = (float*)myMalloc(1 * sizeof(float));;
x1859[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1835, x1835));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1859, x_desc, x1848, x1857, x_desc, x1848));
};
if (x1864) {
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
    64, 128, x1835, x1835));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1871, x1871));

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
    x1880, in_desc, x1848, filt_desc, x376,
    conv_desc, algo, ws_data, ws_size,
    x1878, out_desc, x1877));
};
float* x1883 = (float*)myGpuMalloc(x1876 * sizeof(float));
float* x1884 = (float*)myGpuMalloc(x1874 * sizeof(float));
float* x1885 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1886 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1887 = (float*)myMalloc(1 * sizeof(float));;
x1887[0] = 0.0f;
float* x1889 = (float*)myMalloc(1 * sizeof(float));;
x1889[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1889, x1887, in_desc, x1877, out_desc, x1884, sbmv_desc, x1051,
    x865, 0.1, x679, x424, 1.0E-5,
    x1885, x1886));
};
float* x1892 = (float*)myGpuMalloc(x1876 * sizeof(float));
float* x1893 = (float*)myMalloc(1 * sizeof(float));;
x1893[0] = 0.0f;
float* x1895 = (float*)myMalloc(1 * sizeof(float));;
x1895[0] = 1.0f;

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
    x1895, x_desc, x1884, x1893, x_desc, x1884));
};
if (x1899) {
} else {
assert(false && "ERROR not specified");
}
float* x1911 = (float*)myGpuMalloc(x1910 * sizeof(float));
float* x1912 = (float*)myMalloc(1 * sizeof(float));;
x1912[0] = 0.0f;
float* x1914 = (float*)myMalloc(1 * sizeof(float));;
x1914[0] = 1.0f;

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
    512, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1905, x1905));

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
    x1914, in_desc, x1884, filt_desc, x613,
    conv_desc, algo, ws_data, ws_size,
    x1912, out_desc, x1911));
};
float* x1917 = (float*)myGpuMalloc(x1910 * sizeof(float));
float* x1918 = (float*)myGpuMalloc(x1908 * sizeof(float));
float* x1919 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1920 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1921 = (float*)myMalloc(1 * sizeof(float));;
x1921[0] = 0.0f;
float* x1923 = (float*)myMalloc(1 * sizeof(float));;
x1923[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1905, x1905));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1905, x1905));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1923, x1921, in_desc, x1911, out_desc, x1918, sbmv_desc, x730,
    x925, 0.1, x742, x598, 1.0E-5,
    x1919, x1920));
};
float* x1926 = (float*)myGpuMalloc(x1910 * sizeof(float));
if (x1829) {
} else {
assert(false && "ERROR not specified");
}
float* x1937 = (float*)myGpuMalloc(x1936 * sizeof(float));
float* x1938 = (float*)myMalloc(1 * sizeof(float));;
x1938[0] = 0.0f;
float* x1940 = (float*)myMalloc(1 * sizeof(float));;
x1940[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1788, x1788));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1931, x1931));

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
    x1940, in_desc, x1801, filt_desc, x1069,
    conv_desc, algo, ws_data, ws_size,
    x1938, out_desc, x1937));
};
float* x1943 = (float*)myGpuMalloc(x1936 * sizeof(float));
float* x1944 = (float*)myGpuMalloc(x1934 * sizeof(float));
float* x1945 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1946 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1947 = (float*)myMalloc(1 * sizeof(float));;
x1947[0] = 0.0f;
float* x1949 = (float*)myMalloc(1 * sizeof(float));;
x1949[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1931, x1931));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1931, x1931));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1949, x1947, in_desc, x1937, out_desc, x1944, sbmv_desc, x916,
    x652, 0.1, x421, x364, 1.0E-5,
    x1945, x1946));
};
float* x1952 = (float*)myGpuMalloc(x1936 * sizeof(float));
if (x1956) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(1931) x Sym(1931), res:  x Const(64) x Const(512) x Sym(1905) x Sym(1905)");
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
    64, 512, x1931, x1931));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1905, x1905));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1961, bias_desc, x1944, x1963, out_desc, x1918));
};
float* x1966 = (float*)myMalloc(1 * sizeof(float));;
x1966[0] = 0.0f;
float* x1968 = (float*)myMalloc(1 * sizeof(float));;
x1968[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1905, x1905));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1968, x_desc, x1918, x1966, x_desc, x1918));
};
if (x1972) {
} else {
assert(false && "ERROR not specified");
}
float* x1984 = (float*)myGpuMalloc(x1983 * sizeof(float));
float* x1985 = (float*)myMalloc(1 * sizeof(float));;
x1985[0] = 0.0f;
float* x1987 = (float*)myMalloc(1 * sizeof(float));;
x1987[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1905, x1905));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1978, x1978));

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
    x1987, in_desc, x1918, filt_desc, x1063,
    conv_desc, algo, ws_data, ws_size,
    x1985, out_desc, x1984));
};
float* x1990 = (float*)myGpuMalloc(x1983 * sizeof(float));
float* x1991 = (float*)myGpuMalloc(x1981 * sizeof(float));
float* x1992 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1993 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1994 = (float*)myMalloc(1 * sizeof(float));;
x1994[0] = 0.0f;
float* x1996 = (float*)myMalloc(1 * sizeof(float));;
x1996[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1978, x1978));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1978, x1978));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1996, x1994, in_desc, x1984, out_desc, x1991, sbmv_desc, x961,
    x346, 0.1, x595, x826, 1.0E-5,
    x1992, x1993));
};
float* x1999 = (float*)myGpuMalloc(x1983 * sizeof(float));
float* x2000 = (float*)myMalloc(1 * sizeof(float));;
x2000[0] = 0.0f;
float* x2002 = (float*)myMalloc(1 * sizeof(float));;
x2002[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1978, x1978));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2002, x_desc, x1991, x2000, x_desc, x1991));
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
    64, 128, x1978, x1978));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2014, x2014));

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
    x2023, in_desc, x1991, filt_desc, x1000,
    conv_desc, algo, ws_data, ws_size,
    x2021, out_desc, x2020));
};
float* x2026 = (float*)myGpuMalloc(x2019 * sizeof(float));
float* x2027 = (float*)myGpuMalloc(x2017 * sizeof(float));
float* x2028 = (float*)myGpuMalloc(128 * sizeof(float));
float* x2029 = (float*)myGpuMalloc(128 * sizeof(float));
float* x2030 = (float*)myMalloc(1 * sizeof(float));;
x2030[0] = 0.0f;
float* x2032 = (float*)myMalloc(1 * sizeof(float));;
x2032[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2014, x2014));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2014, x2014));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2032, x2030, in_desc, x2020, out_desc, x2027, sbmv_desc, x319,
    x580, 0.1, x400, x970, 1.0E-5,
    x2028, x2029));
};
float* x2035 = (float*)myGpuMalloc(x2019 * sizeof(float));
float* x2036 = (float*)myMalloc(1 * sizeof(float));;
x2036[0] = 0.0f;
float* x2038 = (float*)myMalloc(1 * sizeof(float));;
x2038[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2014, x2014));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2038, x_desc, x2027, x2036, x_desc, x2027));
};
if (x2042) {
} else {
assert(false && "ERROR not specified");
}
float* x2054 = (float*)myGpuMalloc(x2053 * sizeof(float));
float* x2055 = (float*)myMalloc(1 * sizeof(float));;
x2055[0] = 0.0f;
float* x2057 = (float*)myMalloc(1 * sizeof(float));;
x2057[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2014, x2014));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2048, x2048));

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
    x2057, in_desc, x2027, filt_desc, x628,
    conv_desc, algo, ws_data, ws_size,
    x2055, out_desc, x2054));
};
float* x2060 = (float*)myGpuMalloc(x2053 * sizeof(float));
float* x2061 = (float*)myGpuMalloc(x2051 * sizeof(float));
float* x2062 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2063 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2064 = (float*)myMalloc(1 * sizeof(float));;
x2064[0] = 0.0f;
float* x2066 = (float*)myMalloc(1 * sizeof(float));;
x2066[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2048, x2048));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2048, x2048));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2066, x2064, in_desc, x2054, out_desc, x2061, sbmv_desc, x451,
    x1033, 0.1, x736, x559, 1.0E-5,
    x2062, x2063));
};
float* x2069 = (float*)myGpuMalloc(x2053 * sizeof(float));
if (x2073) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(1905) x Sym(1905), res:  x Const(64) x Const(512) x Sym(2048) x Sym(2048)");
}
float* x2078 = (float*)myMalloc(1 * sizeof(float));;
x2078[0] = 1.0f;
float* x2080 = (float*)myMalloc(1 * sizeof(float));;
x2080[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1905, x1905));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2048, x2048));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2078, bias_desc, x1918, x2080, out_desc, x2061));
};
float* x2083 = (float*)myMalloc(1 * sizeof(float));;
x2083[0] = 0.0f;
float* x2085 = (float*)myMalloc(1 * sizeof(float));;
x2085[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2048, x2048));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2085, x_desc, x2061, x2083, x_desc, x2061));
};
if (x2089) {
} else {
assert(false && "ERROR not specified");
}
float* x2101 = (float*)myGpuMalloc(x2100 * sizeof(float));
float* x2102 = (float*)myMalloc(1 * sizeof(float));;
x2102[0] = 0.0f;
float* x2104 = (float*)myMalloc(1 * sizeof(float));;
x2104[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2048, x2048));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2095, x2095));

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
    x2104, in_desc, x2061, filt_desc, x883,
    conv_desc, algo, ws_data, ws_size,
    x2102, out_desc, x2101));
};
float* x2107 = (float*)myGpuMalloc(x2100 * sizeof(float));
float* x2108 = (float*)myGpuMalloc(x2098 * sizeof(float));
float* x2109 = (float*)myGpuMalloc(128 * sizeof(float));
float* x2110 = (float*)myGpuMalloc(128 * sizeof(float));
float* x2111 = (float*)myMalloc(1 * sizeof(float));;
x2111[0] = 0.0f;
float* x2113 = (float*)myMalloc(1 * sizeof(float));;
x2113[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2095, x2095));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2095, x2095));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2113, x2111, in_desc, x2101, out_desc, x2108, sbmv_desc, x430,
    x805, 0.1, x631, x322, 1.0E-5,
    x2109, x2110));
};
float* x2116 = (float*)myGpuMalloc(x2100 * sizeof(float));
float* x2117 = (float*)myMalloc(1 * sizeof(float));;
x2117[0] = 0.0f;
float* x2119 = (float*)myMalloc(1 * sizeof(float));;
x2119[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2095, x2095));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2119, x_desc, x2108, x2117, x_desc, x2108));
};
if (x2124) {
} else {
assert(false && "ERROR not specified");
}
float* x2137 = (float*)myGpuMalloc(x2136 * sizeof(float));
float* x2138 = (float*)myMalloc(1 * sizeof(float));;
x2138[0] = 0.0f;
float* x2140 = (float*)myMalloc(1 * sizeof(float));;
x2140[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2095, x2095));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2131, x2131));

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
    x2140, in_desc, x2108, filt_desc, x868,
    conv_desc, algo, ws_data, ws_size,
    x2138, out_desc, x2137));
};
float* x2143 = (float*)myGpuMalloc(x2136 * sizeof(float));
float* x2144 = (float*)myGpuMalloc(x2134 * sizeof(float));
float* x2145 = (float*)myGpuMalloc(128 * sizeof(float));
float* x2146 = (float*)myGpuMalloc(128 * sizeof(float));
float* x2147 = (float*)myMalloc(1 * sizeof(float));;
x2147[0] = 0.0f;
float* x2149 = (float*)myMalloc(1 * sizeof(float));;
x2149[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2131, x2131));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2131, x2131));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2149, x2147, in_desc, x2137, out_desc, x2144, sbmv_desc, x676,
    x478, 0.1, x946, x1093, 1.0E-5,
    x2145, x2146));
};
float* x2152 = (float*)myGpuMalloc(x2136 * sizeof(float));
float* x2153 = (float*)myMalloc(1 * sizeof(float));;
x2153[0] = 0.0f;
float* x2155 = (float*)myMalloc(1 * sizeof(float));;
x2155[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2131, x2131));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2155, x_desc, x2144, x2153, x_desc, x2144));
};
if (x2159) {
} else {
assert(false && "ERROR not specified");
}
float* x2171 = (float*)myGpuMalloc(x2170 * sizeof(float));
float* x2172 = (float*)myMalloc(1 * sizeof(float));;
x2172[0] = 0.0f;
float* x2174 = (float*)myMalloc(1 * sizeof(float));;
x2174[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2131, x2131));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2165, x2165));

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
    x2174, in_desc, x2144, filt_desc, x418,
    conv_desc, algo, ws_data, ws_size,
    x2172, out_desc, x2171));
};
float* x2177 = (float*)myGpuMalloc(x2170 * sizeof(float));
float* x2178 = (float*)myGpuMalloc(x2168 * sizeof(float));
float* x2179 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2180 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2181 = (float*)myMalloc(1 * sizeof(float));;
x2181[0] = 0.0f;
float* x2183 = (float*)myMalloc(1 * sizeof(float));;
x2183[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2165, x2165));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2165, x2165));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2183, x2181, in_desc, x2171, out_desc, x2178, sbmv_desc, x796,
    x541, 0.1, x370, x964, 1.0E-5,
    x2179, x2180));
};
float* x2186 = (float*)myGpuMalloc(x2170 * sizeof(float));
if (x2190) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(2048) x Sym(2048), res:  x Const(64) x Const(512) x Sym(2165) x Sym(2165)");
}
float* x2195 = (float*)myMalloc(1 * sizeof(float));;
x2195[0] = 1.0f;
float* x2197 = (float*)myMalloc(1 * sizeof(float));;
x2197[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2048, x2048));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2165, x2165));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2195, bias_desc, x2061, x2197, out_desc, x2178));
};
float* x2200 = (float*)myMalloc(1 * sizeof(float));;
x2200[0] = 0.0f;
float* x2202 = (float*)myMalloc(1 * sizeof(float));;
x2202[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2165, x2165));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2202, x_desc, x2178, x2200, x_desc, x2178));
};
if (x2206) {
} else {
assert(false && "ERROR not specified");
}
float* x2218 = (float*)myGpuMalloc(x2217 * sizeof(float));
float* x2219 = (float*)myMalloc(1 * sizeof(float));;
x2219[0] = 0.0f;
float* x2221 = (float*)myMalloc(1 * sizeof(float));;
x2221[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2165, x2165));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2212, x2212));

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
    x2221, in_desc, x2178, filt_desc, x691,
    conv_desc, algo, ws_data, ws_size,
    x2219, out_desc, x2218));
};
float* x2224 = (float*)myGpuMalloc(x2217 * sizeof(float));
float* x2225 = (float*)myGpuMalloc(x2215 * sizeof(float));
float* x2226 = (float*)myGpuMalloc(128 * sizeof(float));
float* x2227 = (float*)myGpuMalloc(128 * sizeof(float));
float* x2228 = (float*)myMalloc(1 * sizeof(float));;
x2228[0] = 0.0f;
float* x2230 = (float*)myMalloc(1 * sizeof(float));;
x2230[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2212, x2212));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2212, x2212));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2230, x2228, in_desc, x2218, out_desc, x2225, sbmv_desc, x412,
    x1021, 0.1, x1003, x1078, 1.0E-5,
    x2226, x2227));
};
float* x2233 = (float*)myGpuMalloc(x2217 * sizeof(float));
float* x2234 = (float*)myMalloc(1 * sizeof(float));;
x2234[0] = 0.0f;
float* x2236 = (float*)myMalloc(1 * sizeof(float));;
x2236[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2212, x2212));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2236, x_desc, x2225, x2234, x_desc, x2225));
};
if (x2241) {
} else {
assert(false && "ERROR not specified");
}
float* x2254 = (float*)myGpuMalloc(x2253 * sizeof(float));
float* x2255 = (float*)myMalloc(1 * sizeof(float));;
x2255[0] = 0.0f;
float* x2257 = (float*)myMalloc(1 * sizeof(float));;
x2257[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2212, x2212));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2248, x2248));

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
    x2257, in_desc, x2225, filt_desc, x790,
    conv_desc, algo, ws_data, ws_size,
    x2255, out_desc, x2254));
};
float* x2260 = (float*)myGpuMalloc(x2253 * sizeof(float));
float* x2261 = (float*)myGpuMalloc(x2251 * sizeof(float));
float* x2262 = (float*)myGpuMalloc(128 * sizeof(float));
float* x2263 = (float*)myGpuMalloc(128 * sizeof(float));
float* x2264 = (float*)myMalloc(1 * sizeof(float));;
x2264[0] = 0.0f;
float* x2266 = (float*)myMalloc(1 * sizeof(float));;
x2266[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2248, x2248));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2248, x2248));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2266, x2264, in_desc, x2254, out_desc, x2261, sbmv_desc, x532,
    x409, 0.1, x1099, x739, 1.0E-5,
    x2262, x2263));
};
float* x2269 = (float*)myGpuMalloc(x2253 * sizeof(float));
float* x2270 = (float*)myMalloc(1 * sizeof(float));;
x2270[0] = 0.0f;
float* x2272 = (float*)myMalloc(1 * sizeof(float));;
x2272[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2248, x2248));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2272, x_desc, x2261, x2270, x_desc, x2261));
};
if (x2276) {
} else {
assert(false && "ERROR not specified");
}
float* x2288 = (float*)myGpuMalloc(x2287 * sizeof(float));
float* x2289 = (float*)myMalloc(1 * sizeof(float));;
x2289[0] = 0.0f;
float* x2291 = (float*)myMalloc(1 * sizeof(float));;
x2291[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2248, x2248));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2282, x2282));

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
    x2291, in_desc, x2261, filt_desc, x460,
    conv_desc, algo, ws_data, ws_size,
    x2289, out_desc, x2288));
};
float* x2294 = (float*)myGpuMalloc(x2287 * sizeof(float));
float* x2295 = (float*)myGpuMalloc(x2285 * sizeof(float));
float* x2296 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2297 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2298 = (float*)myMalloc(1 * sizeof(float));;
x2298[0] = 0.0f;
float* x2300 = (float*)myMalloc(1 * sizeof(float));;
x2300[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2282, x2282));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2282, x2282));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2300, x2298, in_desc, x2288, out_desc, x2295, sbmv_desc, x763,
    x457, 0.1, x352, x997, 1.0E-5,
    x2296, x2297));
};
float* x2303 = (float*)myGpuMalloc(x2287 * sizeof(float));
if (x2307) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(2165) x Sym(2165), res:  x Const(64) x Const(512) x Sym(2282) x Sym(2282)");
}
float* x2312 = (float*)myMalloc(1 * sizeof(float));;
x2312[0] = 1.0f;
float* x2314 = (float*)myMalloc(1 * sizeof(float));;
x2314[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2165, x2165));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2282, x2282));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2312, bias_desc, x2178, x2314, out_desc, x2295));
};
float* x2317 = (float*)myMalloc(1 * sizeof(float));;
x2317[0] = 0.0f;
float* x2319 = (float*)myMalloc(1 * sizeof(float));;
x2319[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2282, x2282));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2319, x_desc, x2295, x2317, x_desc, x2295));
};
if (x2323) {
} else {
assert(false && "ERROR not specified");
}
float* x2335 = (float*)myGpuMalloc(x2334 * sizeof(float));
float* x2336 = (float*)myMalloc(1 * sizeof(float));;
x2336[0] = 0.0f;
float* x2338 = (float*)myMalloc(1 * sizeof(float));;
x2338[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2282, x2282));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2329, x2329));

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
    x2338, in_desc, x2295, filt_desc, x835,
    conv_desc, algo, ws_data, ws_size,
    x2336, out_desc, x2335));
};
float* x2341 = (float*)myGpuMalloc(x2334 * sizeof(float));
float* x2342 = (float*)myGpuMalloc(x2332 * sizeof(float));
float* x2343 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2344 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2345 = (float*)myMalloc(1 * sizeof(float));;
x2345[0] = 0.0f;
float* x2347 = (float*)myMalloc(1 * sizeof(float));;
x2347[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2329, x2329));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2329, x2329));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2347, x2345, in_desc, x2335, out_desc, x2342, sbmv_desc, x1105,
    x358, 0.1, x688, x889, 1.0E-5,
    x2343, x2344));
};
float* x2350 = (float*)myGpuMalloc(x2334 * sizeof(float));
float* x2351 = (float*)myMalloc(1 * sizeof(float));;
x2351[0] = 0.0f;
float* x2353 = (float*)myMalloc(1 * sizeof(float));;
x2353[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2329, x2329));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2353, x_desc, x2342, x2351, x_desc, x2342));
};
if (x2358) {
} else {
assert(false && "ERROR not specified");
}
float* x2371 = (float*)myGpuMalloc(x2370 * sizeof(float));
float* x2372 = (float*)myMalloc(1 * sizeof(float));;
x2372[0] = 0.0f;
float* x2374 = (float*)myMalloc(1 * sizeof(float));;
x2374[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2329, x2329));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2365, x2365));

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
    x2374, in_desc, x2342, filt_desc, x820,
    conv_desc, algo, ws_data, ws_size,
    x2372, out_desc, x2371));
};
float* x2377 = (float*)myGpuMalloc(x2370 * sizeof(float));
float* x2378 = (float*)myGpuMalloc(x2368 * sizeof(float));
float* x2379 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2380 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2381 = (float*)myMalloc(1 * sizeof(float));;
x2381[0] = 0.0f;
float* x2383 = (float*)myMalloc(1 * sizeof(float));;
x2383[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2365, x2365));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2365, x2365));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2383, x2381, in_desc, x2371, out_desc, x2378, sbmv_desc, x619,
    x343, 0.1, x982, x592, 1.0E-5,
    x2379, x2380));
};
float* x2386 = (float*)myGpuMalloc(x2370 * sizeof(float));
float* x2387 = (float*)myMalloc(1 * sizeof(float));;
x2387[0] = 0.0f;
float* x2389 = (float*)myMalloc(1 * sizeof(float));;
x2389[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2365, x2365));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2389, x_desc, x2378, x2387, x_desc, x2378));
};
if (x2393) {
} else {
assert(false && "ERROR not specified");
}
float* x2405 = (float*)myGpuMalloc(x2404 * sizeof(float));
float* x2406 = (float*)myMalloc(1 * sizeof(float));;
x2406[0] = 0.0f;
float* x2408 = (float*)myMalloc(1 * sizeof(float));;
x2408[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2365, x2365));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2399, x2399));

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
    x2408, in_desc, x2378, filt_desc, x1102,
    conv_desc, algo, ws_data, ws_size,
    x2406, out_desc, x2405));
};
float* x2411 = (float*)myGpuMalloc(x2404 * sizeof(float));
float* x2412 = (float*)myGpuMalloc(x2402 * sizeof(float));
float* x2413 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2414 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2415 = (float*)myMalloc(1 * sizeof(float));;
x2415[0] = 0.0f;
float* x2417 = (float*)myMalloc(1 * sizeof(float));;
x2417[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2399, x2399));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2399, x2399));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2417, x2415, in_desc, x2405, out_desc, x2412, sbmv_desc, x349,
    x646, 0.1, x943, x1096, 1.0E-5,
    x2413, x2414));
};
float* x2420 = (float*)myGpuMalloc(x2404 * sizeof(float));
if (x2323) {
} else {
assert(false && "ERROR not specified");
}
float* x2431 = (float*)myGpuMalloc(x2430 * sizeof(float));
float* x2432 = (float*)myMalloc(1 * sizeof(float));;
x2432[0] = 0.0f;
float* x2434 = (float*)myMalloc(1 * sizeof(float));;
x2434[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2282, x2282));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2425, x2425));

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
    x2434, in_desc, x2295, filt_desc, x520,
    conv_desc, algo, ws_data, ws_size,
    x2432, out_desc, x2431));
};
float* x2437 = (float*)myGpuMalloc(x2430 * sizeof(float));
float* x2438 = (float*)myGpuMalloc(x2428 * sizeof(float));
float* x2439 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2440 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2441 = (float*)myMalloc(1 * sizeof(float));;
x2441[0] = 0.0f;
float* x2443 = (float*)myMalloc(1 * sizeof(float));;
x2443[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2425, x2425));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2425, x2425));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2443, x2441, in_desc, x2431, out_desc, x2438, sbmv_desc, x382,
    x955, 0.1, x553, x928, 1.0E-5,
    x2439, x2440));
};
float* x2446 = (float*)myGpuMalloc(x2430 * sizeof(float));
if (x2450) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2425) x Sym(2425), res:  x Const(64) x Const(1024) x Sym(2399) x Sym(2399)");
}
float* x2455 = (float*)myMalloc(1 * sizeof(float));;
x2455[0] = 1.0f;
float* x2457 = (float*)myMalloc(1 * sizeof(float));;
x2457[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2425, x2425));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2399, x2399));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2455, bias_desc, x2438, x2457, out_desc, x2412));
};
float* x2460 = (float*)myMalloc(1 * sizeof(float));;
x2460[0] = 0.0f;
float* x2462 = (float*)myMalloc(1 * sizeof(float));;
x2462[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2399, x2399));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2462, x_desc, x2412, x2460, x_desc, x2412));
};
if (x2466) {
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
    64, 1024, x2399, x2399));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2472, x2472));

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
    x2481, in_desc, x2412, filt_desc, x334,
    conv_desc, algo, ws_data, ws_size,
    x2479, out_desc, x2478));
};
float* x2484 = (float*)myGpuMalloc(x2477 * sizeof(float));
float* x2485 = (float*)myGpuMalloc(x2475 * sizeof(float));
float* x2486 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2487 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2488 = (float*)myMalloc(1 * sizeof(float));;
x2488[0] = 0.0f;
float* x2490 = (float*)myMalloc(1 * sizeof(float));;
x2490[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2490, x2488, in_desc, x2478, out_desc, x2485, sbmv_desc, x385,
    x952, 0.1, x1072, x766, 1.0E-5,
    x2486, x2487));
};
float* x2493 = (float*)myGpuMalloc(x2477 * sizeof(float));
float* x2494 = (float*)myMalloc(1 * sizeof(float));;
x2494[0] = 0.0f;
float* x2496 = (float*)myMalloc(1 * sizeof(float));;
x2496[0] = 1.0f;

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
    x2496, x_desc, x2485, x2494, x_desc, x2485));
};
if (x2501) {
} else {
assert(false && "ERROR not specified");
}
float* x2514 = (float*)myGpuMalloc(x2513 * sizeof(float));
float* x2515 = (float*)myMalloc(1 * sizeof(float));;
x2515[0] = 0.0f;
float* x2517 = (float*)myMalloc(1 * sizeof(float));;
x2517[0] = 1.0f;

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
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2508, x2508));

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
    x2517, in_desc, x2485, filt_desc, x388,
    conv_desc, algo, ws_data, ws_size,
    x2515, out_desc, x2514));
};
float* x2520 = (float*)myGpuMalloc(x2513 * sizeof(float));
float* x2521 = (float*)myGpuMalloc(x2511 * sizeof(float));
float* x2522 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2523 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2524 = (float*)myMalloc(1 * sizeof(float));;
x2524[0] = 0.0f;
float* x2526 = (float*)myMalloc(1 * sizeof(float));;
x2526[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2508, x2508));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2508, x2508));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2526, x2524, in_desc, x2514, out_desc, x2521, sbmv_desc, x1108,
    x583, 0.1, x895, x1006, 1.0E-5,
    x2522, x2523));
};
float* x2529 = (float*)myGpuMalloc(x2513 * sizeof(float));
float* x2530 = (float*)myMalloc(1 * sizeof(float));;
x2530[0] = 0.0f;
float* x2532 = (float*)myMalloc(1 * sizeof(float));;
x2532[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2508, x2508));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2532, x_desc, x2521, x2530, x_desc, x2521));
};
if (x2536) {
} else {
assert(false && "ERROR not specified");
}
float* x2548 = (float*)myGpuMalloc(x2547 * sizeof(float));
float* x2549 = (float*)myMalloc(1 * sizeof(float));;
x2549[0] = 0.0f;
float* x2551 = (float*)myMalloc(1 * sizeof(float));;
x2551[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2508, x2508));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2542, x2542));

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
    x2551, in_desc, x2521, filt_desc, x463,
    conv_desc, algo, ws_data, ws_size,
    x2549, out_desc, x2548));
};
float* x2554 = (float*)myGpuMalloc(x2547 * sizeof(float));
float* x2555 = (float*)myGpuMalloc(x2545 * sizeof(float));
float* x2556 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2557 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2558 = (float*)myMalloc(1 * sizeof(float));;
x2558[0] = 0.0f;
float* x2560 = (float*)myMalloc(1 * sizeof(float));;
x2560[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2542, x2542));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2542, x2542));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2560, x2558, in_desc, x2548, out_desc, x2555, sbmv_desc, x355,
    x991, 0.1, x841, x724, 1.0E-5,
    x2556, x2557));
};
float* x2563 = (float*)myGpuMalloc(x2547 * sizeof(float));
if (x2567) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2399) x Sym(2399), res:  x Const(64) x Const(1024) x Sym(2542) x Sym(2542)");
}
float* x2572 = (float*)myMalloc(1 * sizeof(float));;
x2572[0] = 1.0f;
float* x2574 = (float*)myMalloc(1 * sizeof(float));;
x2574[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2399, x2399));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2542, x2542));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2572, bias_desc, x2412, x2574, out_desc, x2555));
};
float* x2577 = (float*)myMalloc(1 * sizeof(float));;
x2577[0] = 0.0f;
float* x2579 = (float*)myMalloc(1 * sizeof(float));;
x2579[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2542, x2542));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2579, x_desc, x2555, x2577, x_desc, x2555));
};
if (x2583) {
} else {
assert(false && "ERROR not specified");
}
float* x2595 = (float*)myGpuMalloc(x2594 * sizeof(float));
float* x2596 = (float*)myMalloc(1 * sizeof(float));;
x2596[0] = 0.0f;
float* x2598 = (float*)myMalloc(1 * sizeof(float));;
x2598[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2542, x2542));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2589, x2589));

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
    x2598, in_desc, x2555, filt_desc, x949,
    conv_desc, algo, ws_data, ws_size,
    x2596, out_desc, x2595));
};
float* x2601 = (float*)myGpuMalloc(x2594 * sizeof(float));
float* x2602 = (float*)myGpuMalloc(x2592 * sizeof(float));
float* x2603 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2604 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2605 = (float*)myMalloc(1 * sizeof(float));;
x2605[0] = 0.0f;
float* x2607 = (float*)myMalloc(1 * sizeof(float));;
x2607[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2589, x2589));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2589, x2589));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2607, x2605, in_desc, x2595, out_desc, x2602, sbmv_desc, x682,
    x886, 0.1, x829, x817, 1.0E-5,
    x2603, x2604));
};
float* x2610 = (float*)myGpuMalloc(x2594 * sizeof(float));
float* x2611 = (float*)myMalloc(1 * sizeof(float));;
x2611[0] = 0.0f;
float* x2613 = (float*)myMalloc(1 * sizeof(float));;
x2613[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2589, x2589));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2613, x_desc, x2602, x2611, x_desc, x2602));
};
if (x2618) {
} else {
assert(false && "ERROR not specified");
}
float* x2631 = (float*)myGpuMalloc(x2630 * sizeof(float));
float* x2632 = (float*)myMalloc(1 * sizeof(float));;
x2632[0] = 0.0f;
float* x2634 = (float*)myMalloc(1 * sizeof(float));;
x2634[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2589, x2589));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2625, x2625));

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
    x2634, in_desc, x2602, filt_desc, x337,
    conv_desc, algo, ws_data, ws_size,
    x2632, out_desc, x2631));
};
float* x2637 = (float*)myGpuMalloc(x2630 * sizeof(float));
float* x2638 = (float*)myGpuMalloc(x2628 * sizeof(float));
float* x2639 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2640 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2641 = (float*)myMalloc(1 * sizeof(float));;
x2641[0] = 0.0f;
float* x2643 = (float*)myMalloc(1 * sizeof(float));;
x2643[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2625, x2625));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2625, x2625));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2643, x2641, in_desc, x2631, out_desc, x2638, sbmv_desc, x979,
    x871, 0.1, x667, x484, 1.0E-5,
    x2639, x2640));
};
float* x2646 = (float*)myGpuMalloc(x2630 * sizeof(float));
float* x2647 = (float*)myMalloc(1 * sizeof(float));;
x2647[0] = 0.0f;
float* x2649 = (float*)myMalloc(1 * sizeof(float));;
x2649[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2625, x2625));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2649, x_desc, x2638, x2647, x_desc, x2638));
};
if (x2653) {
} else {
assert(false && "ERROR not specified");
}
float* x2665 = (float*)myGpuMalloc(x2664 * sizeof(float));
float* x2666 = (float*)myMalloc(1 * sizeof(float));;
x2666[0] = 0.0f;
float* x2668 = (float*)myMalloc(1 * sizeof(float));;
x2668[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2625, x2625));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2659, x2659));

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
    x2668, in_desc, x2638, filt_desc, x643,
    conv_desc, algo, ws_data, ws_size,
    x2666, out_desc, x2665));
};
float* x2671 = (float*)myGpuMalloc(x2664 * sizeof(float));
float* x2672 = (float*)myGpuMalloc(x2662 * sizeof(float));
float* x2673 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2674 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2675 = (float*)myMalloc(1 * sizeof(float));;
x2675[0] = 0.0f;
float* x2677 = (float*)myMalloc(1 * sizeof(float));;
x2677[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2659, x2659));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2659, x2659));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2677, x2675, in_desc, x2665, out_desc, x2672, sbmv_desc, x1084,
    x466, 0.1, x715, x859, 1.0E-5,
    x2673, x2674));
};
float* x2680 = (float*)myGpuMalloc(x2664 * sizeof(float));
if (x2684) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2542) x Sym(2542), res:  x Const(64) x Const(1024) x Sym(2659) x Sym(2659)");
}
float* x2689 = (float*)myMalloc(1 * sizeof(float));;
x2689[0] = 1.0f;
float* x2691 = (float*)myMalloc(1 * sizeof(float));;
x2691[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2542, x2542));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2659, x2659));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2689, bias_desc, x2555, x2691, out_desc, x2672));
};
float* x2694 = (float*)myMalloc(1 * sizeof(float));;
x2694[0] = 0.0f;
float* x2696 = (float*)myMalloc(1 * sizeof(float));;
x2696[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2659, x2659));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2696, x_desc, x2672, x2694, x_desc, x2672));
};
if (x2700) {
} else {
assert(false && "ERROR not specified");
}
float* x2712 = (float*)myGpuMalloc(x2711 * sizeof(float));
float* x2713 = (float*)myMalloc(1 * sizeof(float));;
x2713[0] = 0.0f;
float* x2715 = (float*)myMalloc(1 * sizeof(float));;
x2715[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2659, x2659));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2706, x2706));

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
    x2715, in_desc, x2672, filt_desc, x313,
    conv_desc, algo, ws_data, ws_size,
    x2713, out_desc, x2712));
};
float* x2718 = (float*)myGpuMalloc(x2711 * sizeof(float));
float* x2719 = (float*)myGpuMalloc(x2709 * sizeof(float));
float* x2720 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2721 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2722 = (float*)myMalloc(1 * sizeof(float));;
x2722[0] = 0.0f;
float* x2724 = (float*)myMalloc(1 * sizeof(float));;
x2724[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2706, x2706));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2706, x2706));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2724, x2722, in_desc, x2712, out_desc, x2719, sbmv_desc, x571,
    x1018, 0.1, x784, x589, 1.0E-5,
    x2720, x2721));
};
float* x2727 = (float*)myGpuMalloc(x2711 * sizeof(float));
float* x2728 = (float*)myMalloc(1 * sizeof(float));;
x2728[0] = 0.0f;
float* x2730 = (float*)myMalloc(1 * sizeof(float));;
x2730[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2706, x2706));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2730, x_desc, x2719, x2728, x_desc, x2719));
};
if (x2735) {
} else {
assert(false && "ERROR not specified");
}
float* x2748 = (float*)myGpuMalloc(x2747 * sizeof(float));
float* x2749 = (float*)myMalloc(1 * sizeof(float));;
x2749[0] = 0.0f;
float* x2751 = (float*)myMalloc(1 * sizeof(float));;
x2751[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2706, x2706));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2742, x2742));

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
    x2751, in_desc, x2719, filt_desc, x1042,
    conv_desc, algo, ws_data, ws_size,
    x2749, out_desc, x2748));
};
float* x2754 = (float*)myGpuMalloc(x2747 * sizeof(float));
float* x2755 = (float*)myGpuMalloc(x2745 * sizeof(float));
float* x2756 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2757 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2758 = (float*)myMalloc(1 * sizeof(float));;
x2758[0] = 0.0f;
float* x2760 = (float*)myMalloc(1 * sizeof(float));;
x2760[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2742, x2742));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2742, x2742));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2760, x2758, in_desc, x2748, out_desc, x2755, sbmv_desc, x517,
    x703, 0.1, x853, x985, 1.0E-5,
    x2756, x2757));
};
float* x2763 = (float*)myGpuMalloc(x2747 * sizeof(float));
float* x2764 = (float*)myMalloc(1 * sizeof(float));;
x2764[0] = 0.0f;
float* x2766 = (float*)myMalloc(1 * sizeof(float));;
x2766[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2742, x2742));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2766, x_desc, x2755, x2764, x_desc, x2755));
};
if (x2770) {
} else {
assert(false && "ERROR not specified");
}
float* x2782 = (float*)myGpuMalloc(x2781 * sizeof(float));
float* x2783 = (float*)myMalloc(1 * sizeof(float));;
x2783[0] = 0.0f;
float* x2785 = (float*)myMalloc(1 * sizeof(float));;
x2785[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2742, x2742));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2776, x2776));

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
    x2785, in_desc, x2755, filt_desc, x562,
    conv_desc, algo, ws_data, ws_size,
    x2783, out_desc, x2782));
};
float* x2788 = (float*)myGpuMalloc(x2781 * sizeof(float));
float* x2789 = (float*)myGpuMalloc(x2779 * sizeof(float));
float* x2790 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2791 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2792 = (float*)myMalloc(1 * sizeof(float));;
x2792[0] = 0.0f;
float* x2794 = (float*)myMalloc(1 * sizeof(float));;
x2794[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2776, x2776));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2776, x2776));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2794, x2792, in_desc, x2782, out_desc, x2789, sbmv_desc, x1009,
    x733, 0.1, x988, x778, 1.0E-5,
    x2790, x2791));
};
float* x2797 = (float*)myGpuMalloc(x2781 * sizeof(float));
if (x2801) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2659) x Sym(2659), res:  x Const(64) x Const(1024) x Sym(2776) x Sym(2776)");
}
float* x2806 = (float*)myMalloc(1 * sizeof(float));;
x2806[0] = 1.0f;
float* x2808 = (float*)myMalloc(1 * sizeof(float));;
x2808[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2659, x2659));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2776, x2776));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2806, bias_desc, x2672, x2808, out_desc, x2789));
};
float* x2811 = (float*)myMalloc(1 * sizeof(float));;
x2811[0] = 0.0f;
float* x2813 = (float*)myMalloc(1 * sizeof(float));;
x2813[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2776, x2776));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2813, x_desc, x2789, x2811, x_desc, x2789));
};
if (x2817) {
} else {
assert(false && "ERROR not specified");
}
float* x2829 = (float*)myGpuMalloc(x2828 * sizeof(float));
float* x2830 = (float*)myMalloc(1 * sizeof(float));;
x2830[0] = 0.0f;
float* x2832 = (float*)myMalloc(1 * sizeof(float));;
x2832[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2776, x2776));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2823, x2823));

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
    x2832, in_desc, x2789, filt_desc, x361,
    conv_desc, algo, ws_data, ws_size,
    x2830, out_desc, x2829));
};
float* x2835 = (float*)myGpuMalloc(x2828 * sizeof(float));
float* x2836 = (float*)myGpuMalloc(x2826 * sizeof(float));
float* x2837 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2838 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2839 = (float*)myMalloc(1 * sizeof(float));;
x2839[0] = 0.0f;
float* x2841 = (float*)myMalloc(1 * sizeof(float));;
x2841[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2823, x2823));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2823, x2823));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2841, x2839, in_desc, x2829, out_desc, x2836, sbmv_desc, x526,
    x850, 0.1, x1057, x502, 1.0E-5,
    x2837, x2838));
};
float* x2844 = (float*)myGpuMalloc(x2828 * sizeof(float));
float* x2845 = (float*)myMalloc(1 * sizeof(float));;
x2845[0] = 0.0f;
float* x2847 = (float*)myMalloc(1 * sizeof(float));;
x2847[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2823, x2823));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2847, x_desc, x2836, x2845, x_desc, x2836));
};
if (x2852) {
} else {
assert(false && "ERROR not specified");
}
float* x2865 = (float*)myGpuMalloc(x2864 * sizeof(float));
float* x2866 = (float*)myMalloc(1 * sizeof(float));;
x2866[0] = 0.0f;
float* x2868 = (float*)myMalloc(1 * sizeof(float));;
x2868[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2823, x2823));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2859, x2859));

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
    x2868, in_desc, x2836, filt_desc, x1081,
    conv_desc, algo, ws_data, ws_size,
    x2866, out_desc, x2865));
};
float* x2871 = (float*)myGpuMalloc(x2864 * sizeof(float));
float* x2872 = (float*)myGpuMalloc(x2862 * sizeof(float));
float* x2873 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2874 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2875 = (float*)myMalloc(1 * sizeof(float));;
x2875[0] = 0.0f;
float* x2877 = (float*)myMalloc(1 * sizeof(float));;
x2877[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2859, x2859));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2859, x2859));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2877, x2875, in_desc, x2865, out_desc, x2872, sbmv_desc, x799,
    x622, 0.1, x1045, x607, 1.0E-5,
    x2873, x2874));
};
float* x2880 = (float*)myGpuMalloc(x2864 * sizeof(float));
float* x2881 = (float*)myMalloc(1 * sizeof(float));;
x2881[0] = 0.0f;
float* x2883 = (float*)myMalloc(1 * sizeof(float));;
x2883[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2859, x2859));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2883, x_desc, x2872, x2881, x_desc, x2872));
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
    64, 256, x2859, x2859));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2893, x2893));

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
    x2902, in_desc, x2872, filt_desc, x958,
    conv_desc, algo, ws_data, ws_size,
    x2900, out_desc, x2899));
};
float* x2905 = (float*)myGpuMalloc(x2898 * sizeof(float));
float* x2906 = (float*)myGpuMalloc(x2896 * sizeof(float));
float* x2907 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2908 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2909 = (float*)myMalloc(1 * sizeof(float));;
x2909[0] = 0.0f;
float* x2911 = (float*)myMalloc(1 * sizeof(float));;
x2911[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2893, x2893));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2893, x2893));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2911, x2909, in_desc, x2899, out_desc, x2906, sbmv_desc, x472,
    x655, 0.1, x922, x1111, 1.0E-5,
    x2907, x2908));
};
float* x2914 = (float*)myGpuMalloc(x2898 * sizeof(float));
if (x2918) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2776) x Sym(2776), res:  x Const(64) x Const(1024) x Sym(2893) x Sym(2893)");
}
float* x2923 = (float*)myMalloc(1 * sizeof(float));;
x2923[0] = 1.0f;
float* x2925 = (float*)myMalloc(1 * sizeof(float));;
x2925[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2776, x2776));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2893, x2893));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2923, bias_desc, x2789, x2925, out_desc, x2906));
};
float* x2928 = (float*)myMalloc(1 * sizeof(float));;
x2928[0] = 0.0f;
float* x2930 = (float*)myMalloc(1 * sizeof(float));;
x2930[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2893, x2893));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2930, x_desc, x2906, x2928, x_desc, x2906));
};
if (x2934) {
} else {
assert(false && "ERROR not specified");
}
float* x2946 = (float*)myGpuMalloc(x2945 * sizeof(float));
float* x2947 = (float*)myMalloc(1 * sizeof(float));;
x2947[0] = 0.0f;
float* x2949 = (float*)myMalloc(1 * sizeof(float));;
x2949[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2893, x2893));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2940, x2940));

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
    x2949, in_desc, x2906, filt_desc, x748,
    conv_desc, algo, ws_data, ws_size,
    x2947, out_desc, x2946));
};
float* x2952 = (float*)myGpuMalloc(x2945 * sizeof(float));
float* x2953 = (float*)myGpuMalloc(x2943 * sizeof(float));
float* x2954 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2955 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2956 = (float*)myMalloc(1 * sizeof(float));;
x2956[0] = 0.0f;
float* x2958 = (float*)myMalloc(1 * sizeof(float));;
x2958[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2940, x2940));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2940, x2940));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2958, x2956, in_desc, x2946, out_desc, x2953, sbmv_desc, x550,
    x1054, 0.1, x535, x823, 1.0E-5,
    x2954, x2955));
};
float* x2961 = (float*)myGpuMalloc(x2945 * sizeof(float));
float* x2962 = (float*)myMalloc(1 * sizeof(float));;
x2962[0] = 0.0f;
float* x2964 = (float*)myMalloc(1 * sizeof(float));;
x2964[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2940, x2940));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2964, x_desc, x2953, x2962, x_desc, x2953));
};
if (x2969) {
} else {
assert(false && "ERROR not specified");
}
float* x2982 = (float*)myGpuMalloc(x2981 * sizeof(float));
float* x2983 = (float*)myMalloc(1 * sizeof(float));;
x2983[0] = 0.0f;
float* x2985 = (float*)myMalloc(1 * sizeof(float));;
x2985[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2940, x2940));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2976, x2976));

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
    x2985, in_desc, x2953, filt_desc, x973,
    conv_desc, algo, ws_data, ws_size,
    x2983, out_desc, x2982));
};
float* x2988 = (float*)myGpuMalloc(x2981 * sizeof(float));
float* x2989 = (float*)myGpuMalloc(x2979 * sizeof(float));
float* x2990 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2991 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2992 = (float*)myMalloc(1 * sizeof(float));;
x2992[0] = 0.0f;
float* x2994 = (float*)myMalloc(1 * sizeof(float));;
x2994[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2976, x2976));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2976, x2976));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2994, x2992, in_desc, x2982, out_desc, x2989, sbmv_desc, x718,
    x862, 0.1, x505, x1015, 1.0E-5,
    x2990, x2991));
};
float* x2997 = (float*)myGpuMalloc(x2981 * sizeof(float));
float* x2998 = (float*)myMalloc(1 * sizeof(float));;
x2998[0] = 0.0f;
float* x3000 = (float*)myMalloc(1 * sizeof(float));;
x3000[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2976, x2976));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x3000, x_desc, x2989, x2998, x_desc, x2989));
};
if (x3004) {
} else {
assert(false && "ERROR not specified");
}
float* x3016 = (float*)myGpuMalloc(x3015 * sizeof(float));
float* x3017 = (float*)myMalloc(1 * sizeof(float));;
x3017[0] = 0.0f;
float* x3019 = (float*)myMalloc(1 * sizeof(float));;
x3019[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2976, x2976));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x3010, x3010));

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
    x3019, in_desc, x2989, filt_desc, x586,
    conv_desc, algo, ws_data, ws_size,
    x3017, out_desc, x3016));
};
float* x3022 = (float*)myGpuMalloc(x3015 * sizeof(float));
float* x3023 = (float*)myGpuMalloc(x3013 * sizeof(float));
float* x3024 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x3025 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x3026 = (float*)myMalloc(1 * sizeof(float));;
x3026[0] = 0.0f;
float* x3028 = (float*)myMalloc(1 * sizeof(float));;
x3028[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x3010, x3010));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x3010, x3010));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3028, x3026, in_desc, x3016, out_desc, x3023, sbmv_desc, x1039,
    x574, 0.1, x661, x844, 1.0E-5,
    x3024, x3025));
};
float* x3031 = (float*)myGpuMalloc(x3015 * sizeof(float));
if (x3035) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2893) x Sym(2893), res:  x Const(64) x Const(1024) x Sym(3010) x Sym(3010)");
}
float* x3040 = (float*)myMalloc(1 * sizeof(float));;
x3040[0] = 1.0f;
float* x3042 = (float*)myMalloc(1 * sizeof(float));;
x3042[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2893, x2893));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x3010, x3010));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x3040, bias_desc, x2906, x3042, out_desc, x3023));
};
float* x3045 = (float*)myMalloc(1 * sizeof(float));;
x3045[0] = 0.0f;
float* x3047 = (float*)myMalloc(1 * sizeof(float));;
x3047[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x3010, x3010));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x3047, x_desc, x3023, x3045, x_desc, x3023));
};
if (x3051) {
} else {
assert(false && "ERROR not specified");
}
float* x3063 = (float*)myGpuMalloc(x3062 * sizeof(float));
float* x3064 = (float*)myMalloc(1 * sizeof(float));;
x3064[0] = 0.0f;
float* x3066 = (float*)myMalloc(1 * sizeof(float));;
x3066[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x3010, x3010));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3057, x3057));

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
    x3066, in_desc, x3023, filt_desc, x712,
    conv_desc, algo, ws_data, ws_size,
    x3064, out_desc, x3063));
};
float* x3069 = (float*)myGpuMalloc(x3062 * sizeof(float));
float* x3070 = (float*)myGpuMalloc(x3060 * sizeof(float));
float* x3071 = (float*)myGpuMalloc(512 * sizeof(float));
float* x3072 = (float*)myGpuMalloc(512 * sizeof(float));
float* x3073 = (float*)myMalloc(1 * sizeof(float));;
x3073[0] = 0.0f;
float* x3075 = (float*)myMalloc(1 * sizeof(float));;
x3075[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3057, x3057));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3057, x3057));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3075, x3073, in_desc, x3063, out_desc, x3070, sbmv_desc, x898,
    x967, 0.1, x496, x658, 1.0E-5,
    x3071, x3072));
};
float* x3078 = (float*)myGpuMalloc(x3062 * sizeof(float));
float* x3079 = (float*)myMalloc(1 * sizeof(float));;
x3079[0] = 0.0f;
float* x3081 = (float*)myMalloc(1 * sizeof(float));;
x3081[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3057, x3057));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x3081, x_desc, x3070, x3079, x_desc, x3070));
};
if (x3086) {
} else {
assert(false && "ERROR not specified");
}
float* x3099 = (float*)myGpuMalloc(x3098 * sizeof(float));
float* x3100 = (float*)myMalloc(1 * sizeof(float));;
x3100[0] = 0.0f;
float* x3102 = (float*)myMalloc(1 * sizeof(float));;
x3102[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3057, x3057));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3093, x3093));

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
    x3102, in_desc, x3070, filt_desc, x397,
    conv_desc, algo, ws_data, ws_size,
    x3100, out_desc, x3099));
};
float* x3105 = (float*)myGpuMalloc(x3098 * sizeof(float));
float* x3106 = (float*)myGpuMalloc(x3096 * sizeof(float));
float* x3107 = (float*)myGpuMalloc(512 * sizeof(float));
float* x3108 = (float*)myGpuMalloc(512 * sizeof(float));
float* x3109 = (float*)myMalloc(1 * sizeof(float));;
x3109[0] = 0.0f;
float* x3111 = (float*)myMalloc(1 * sizeof(float));;
x3111[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3093, x3093));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3093, x3093));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3111, x3109, in_desc, x3099, out_desc, x3106, sbmv_desc, x910,
    x772, 0.1, x634, x445, 1.0E-5,
    x3107, x3108));
};
float* x3114 = (float*)myGpuMalloc(x3098 * sizeof(float));
float* x3115 = (float*)myMalloc(1 * sizeof(float));;
x3115[0] = 0.0f;
float* x3117 = (float*)myMalloc(1 * sizeof(float));;
x3117[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3093, x3093));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x3117, x_desc, x3106, x3115, x_desc, x3106));
};
if (x3121) {
} else {
assert(false && "ERROR not specified");
}
float* x3133 = (float*)myGpuMalloc(x3132 * sizeof(float));
float* x3134 = (float*)myMalloc(1 * sizeof(float));;
x3134[0] = 0.0f;
float* x3136 = (float*)myMalloc(1 * sizeof(float));;
x3136[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3093, x3093));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3127, x3127));

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
    x3136, in_desc, x3106, filt_desc, x931,
    conv_desc, algo, ws_data, ws_size,
    x3134, out_desc, x3133));
};
float* x3139 = (float*)myGpuMalloc(x3132 * sizeof(float));
float* x3140 = (float*)myGpuMalloc(x3130 * sizeof(float));
float* x3141 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x3142 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x3143 = (float*)myMalloc(1 * sizeof(float));;
x3143[0] = 0.0f;
float* x3145 = (float*)myMalloc(1 * sizeof(float));;
x3145[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3127, x3127));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3127, x3127));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3145, x3143, in_desc, x3133, out_desc, x3140, sbmv_desc, x1012,
    x481, 0.1, x640, x874, 1.0E-5,
    x3141, x3142));
};
float* x3148 = (float*)myGpuMalloc(x3132 * sizeof(float));
if (x3051) {
} else {
assert(false && "ERROR not specified");
}
float* x3159 = (float*)myGpuMalloc(x3158 * sizeof(float));
float* x3160 = (float*)myMalloc(1 * sizeof(float));;
x3160[0] = 0.0f;
float* x3162 = (float*)myMalloc(1 * sizeof(float));;
x3162[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x3010, x3010));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3153, x3153));

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
    x3162, in_desc, x3023, filt_desc, x937,
    conv_desc, algo, ws_data, ws_size,
    x3160, out_desc, x3159));
};
float* x3165 = (float*)myGpuMalloc(x3158 * sizeof(float));
float* x3166 = (float*)myGpuMalloc(x3156 * sizeof(float));
float* x3167 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x3168 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x3169 = (float*)myMalloc(1 * sizeof(float));;
x3169[0] = 0.0f;
float* x3171 = (float*)myMalloc(1 * sizeof(float));;
x3171[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3153, x3153));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3153, x3153));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3171, x3169, in_desc, x3159, out_desc, x3166, sbmv_desc, x814,
    x616, 0.1, x487, x670, 1.0E-5,
    x3167, x3168));
};
float* x3174 = (float*)myGpuMalloc(x3158 * sizeof(float));
if (x3178) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(3153) x Sym(3153), res:  x Const(64) x Const(2048) x Sym(3127) x Sym(3127)");
}
float* x3183 = (float*)myMalloc(1 * sizeof(float));;
x3183[0] = 1.0f;
float* x3185 = (float*)myMalloc(1 * sizeof(float));;
x3185[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3153, x3153));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3127, x3127));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x3183, bias_desc, x3166, x3185, out_desc, x3140));
};
float* x3188 = (float*)myMalloc(1 * sizeof(float));;
x3188[0] = 0.0f;
float* x3190 = (float*)myMalloc(1 * sizeof(float));;
x3190[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3127, x3127));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x3190, x_desc, x3140, x3188, x_desc, x3140));
};
if (x3194) {
} else {
assert(false && "ERROR not specified");
}
float* x3206 = (float*)myGpuMalloc(x3205 * sizeof(float));
float* x3207 = (float*)myMalloc(1 * sizeof(float));;
x3207[0] = 0.0f;
float* x3209 = (float*)myMalloc(1 * sizeof(float));;
x3209[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3127, x3127));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 2048, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3200, x3200));

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
    x3209, in_desc, x3140, filt_desc, x940,
    conv_desc, algo, ws_data, ws_size,
    x3207, out_desc, x3206));
};
float* x3212 = (float*)myGpuMalloc(x3205 * sizeof(float));
float* x3213 = (float*)myGpuMalloc(x3203 * sizeof(float));
float* x3214 = (float*)myGpuMalloc(512 * sizeof(float));
float* x3215 = (float*)myGpuMalloc(512 * sizeof(float));
float* x3216 = (float*)myMalloc(1 * sizeof(float));;
x3216[0] = 0.0f;
float* x3218 = (float*)myMalloc(1 * sizeof(float));;
x3218[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3200, x3200));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3200, x3200));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3218, x3216, in_desc, x3206, out_desc, x3213, sbmv_desc, x433,
    x706, 0.1, x757, x490, 1.0E-5,
    x3214, x3215));
};
float* x3221 = (float*)myGpuMalloc(x3205 * sizeof(float));
float* x3222 = (float*)myMalloc(1 * sizeof(float));;
x3222[0] = 0.0f;
float* x3224 = (float*)myMalloc(1 * sizeof(float));;
x3224[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3200, x3200));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x3224, x_desc, x3213, x3222, x_desc, x3213));
};
if (x3229) {
} else {
assert(false && "ERROR not specified");
}
float* x3242 = (float*)myGpuMalloc(x3241 * sizeof(float));
float* x3243 = (float*)myMalloc(1 * sizeof(float));;
x3243[0] = 0.0f;
float* x3245 = (float*)myMalloc(1 * sizeof(float));;
x3245[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3200, x3200));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3236, x3236));

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
    x3245, in_desc, x3213, filt_desc, x760,
    conv_desc, algo, ws_data, ws_size,
    x3243, out_desc, x3242));
};
float* x3248 = (float*)myGpuMalloc(x3241 * sizeof(float));
float* x3249 = (float*)myGpuMalloc(x3239 * sizeof(float));
float* x3250 = (float*)myGpuMalloc(512 * sizeof(float));
float* x3251 = (float*)myGpuMalloc(512 * sizeof(float));
float* x3252 = (float*)myMalloc(1 * sizeof(float));;
x3252[0] = 0.0f;
float* x3254 = (float*)myMalloc(1 * sizeof(float));;
x3254[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3236, x3236));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3236, x3236));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3254, x3252, in_desc, x3242, out_desc, x3249, sbmv_desc, x775,
    x493, 0.1, x709, x880, 1.0E-5,
    x3250, x3251));
};
float* x3257 = (float*)myGpuMalloc(x3241 * sizeof(float));
float* x3258 = (float*)myMalloc(1 * sizeof(float));;
x3258[0] = 0.0f;
float* x3260 = (float*)myMalloc(1 * sizeof(float));;
x3260[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3236, x3236));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x3260, x_desc, x3249, x3258, x_desc, x3249));
};
if (x3264) {
} else {
assert(false && "ERROR not specified");
}
float* x3276 = (float*)myGpuMalloc(x3275 * sizeof(float));
float* x3277 = (float*)myMalloc(1 * sizeof(float));;
x3277[0] = 0.0f;
float* x3279 = (float*)myMalloc(1 * sizeof(float));;
x3279[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3236, x3236));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3270, x3270));

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
    x3279, in_desc, x3249, filt_desc, x436,
    conv_desc, algo, ws_data, ws_size,
    x3277, out_desc, x3276));
};
float* x3282 = (float*)myGpuMalloc(x3275 * sizeof(float));
float* x3283 = (float*)myGpuMalloc(x3273 * sizeof(float));
float* x3284 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x3285 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x3286 = (float*)myMalloc(1 * sizeof(float));;
x3286[0] = 0.0f;
float* x3288 = (float*)myMalloc(1 * sizeof(float));;
x3288[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3270, x3270));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3270, x3270));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3288, x3286, in_desc, x3276, out_desc, x3283, sbmv_desc, x577,
    x727, 0.1, x499, x1030, 1.0E-5,
    x3284, x3285));
};
float* x3291 = (float*)myGpuMalloc(x3275 * sizeof(float));
if (x3295) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(3127) x Sym(3127), res:  x Const(64) x Const(2048) x Sym(3270) x Sym(3270)");
}
float* x3300 = (float*)myMalloc(1 * sizeof(float));;
x3300[0] = 1.0f;
float* x3302 = (float*)myMalloc(1 * sizeof(float));;
x3302[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3127, x3127));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3270, x3270));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x3300, bias_desc, x3140, x3302, out_desc, x3283));
};
float* x3305 = (float*)myMalloc(1 * sizeof(float));;
x3305[0] = 0.0f;
float* x3307 = (float*)myMalloc(1 * sizeof(float));;
x3307[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3270, x3270));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x3307, x_desc, x3283, x3305, x_desc, x3283));
};
if (x3311) {
} else {
assert(false && "ERROR not specified");
}
float* x3323 = (float*)myGpuMalloc(x3322 * sizeof(float));
float* x3324 = (float*)myMalloc(1 * sizeof(float));;
x3324[0] = 0.0f;
float* x3326 = (float*)myMalloc(1 * sizeof(float));;
x3326[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3270, x3270));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 2048, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3317, x3317));

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
    x3326, in_desc, x3283, filt_desc, x1090,
    conv_desc, algo, ws_data, ws_size,
    x3324, out_desc, x3323));
};
float* x3329 = (float*)myGpuMalloc(x3322 * sizeof(float));
float* x3330 = (float*)myGpuMalloc(x3320 * sizeof(float));
float* x3331 = (float*)myGpuMalloc(512 * sizeof(float));
float* x3332 = (float*)myGpuMalloc(512 * sizeof(float));
float* x3333 = (float*)myMalloc(1 * sizeof(float));;
x3333[0] = 0.0f;
float* x3335 = (float*)myMalloc(1 * sizeof(float));;
x3335[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3317, x3317));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3317, x3317));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3335, x3333, in_desc, x3323, out_desc, x3330, sbmv_desc, x340,
    x529, 0.1, x934, x1060, 1.0E-5,
    x3331, x3332));
};
float* x3338 = (float*)myGpuMalloc(x3322 * sizeof(float));
float* x3339 = (float*)myMalloc(1 * sizeof(float));;
x3339[0] = 0.0f;
float* x3341 = (float*)myMalloc(1 * sizeof(float));;
x3341[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3317, x3317));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x3341, x_desc, x3330, x3339, x_desc, x3330));
};
if (x3346) {
} else {
assert(false && "ERROR not specified");
}
float* x3359 = (float*)myGpuMalloc(x3358 * sizeof(float));
float* x3360 = (float*)myMalloc(1 * sizeof(float));;
x3360[0] = 0.0f;
float* x3362 = (float*)myMalloc(1 * sizeof(float));;
x3362[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3317, x3317));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3353, x3353));

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
    x3362, in_desc, x3330, filt_desc, x379,
    conv_desc, algo, ws_data, ws_size,
    x3360, out_desc, x3359));
};
float* x3365 = (float*)myGpuMalloc(x3358 * sizeof(float));
float* x3366 = (float*)myGpuMalloc(x3356 * sizeof(float));
float* x3367 = (float*)myGpuMalloc(512 * sizeof(float));
float* x3368 = (float*)myGpuMalloc(512 * sizeof(float));
float* x3369 = (float*)myMalloc(1 * sizeof(float));;
x3369[0] = 0.0f;
float* x3371 = (float*)myMalloc(1 * sizeof(float));;
x3371[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3353, x3353));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3353, x3353));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3371, x3369, in_desc, x3359, out_desc, x3366, sbmv_desc, x877,
    x802, 0.1, x331, x901, 1.0E-5,
    x3367, x3368));
};
float* x3374 = (float*)myGpuMalloc(x3358 * sizeof(float));
float* x3375 = (float*)myMalloc(1 * sizeof(float));;
x3375[0] = 0.0f;
float* x3377 = (float*)myMalloc(1 * sizeof(float));;
x3377[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3353, x3353));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x3377, x_desc, x3366, x3375, x_desc, x3366));
};
if (x3381) {
} else {
assert(false && "ERROR not specified");
}
float* x3393 = (float*)myGpuMalloc(x3392 * sizeof(float));
float* x3394 = (float*)myMalloc(1 * sizeof(float));;
x3394[0] = 0.0f;
float* x3396 = (float*)myMalloc(1 * sizeof(float));;
x3396[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3353, x3353));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3387, x3387));

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
    x3396, in_desc, x3366, filt_desc, x394,
    conv_desc, algo, ws_data, ws_size,
    x3394, out_desc, x3393));
};
float* x3399 = (float*)myGpuMalloc(x3392 * sizeof(float));
float* x3400 = (float*)myGpuMalloc(x3390 * sizeof(float));
float* x3401 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x3402 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x3403 = (float*)myMalloc(1 * sizeof(float));;
x3403[0] = 0.0f;
float* x3405 = (float*)myMalloc(1 * sizeof(float));;
x3405[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3387, x3387));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3387, x3387));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3405, x3403, in_desc, x3393, out_desc, x3400, sbmv_desc, x604,
    x838, 0.1, x1075, x664, 1.0E-5,
    x3401, x3402));
};
float* x3408 = (float*)myGpuMalloc(x3392 * sizeof(float));
if (x3412) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(3270) x Sym(3270), res:  x Const(64) x Const(2048) x Sym(3387) x Sym(3387)");
}
float* x3417 = (float*)myMalloc(1 * sizeof(float));;
x3417[0] = 1.0f;
float* x3419 = (float*)myMalloc(1 * sizeof(float));;
x3419[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3270, x3270));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3387, x3387));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x3417, bias_desc, x3283, x3419, out_desc, x3400));
};
float* x3422 = (float*)myMalloc(1 * sizeof(float));;
x3422[0] = 0.0f;
float* x3424 = (float*)myMalloc(1 * sizeof(float));;
x3424[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3387, x3387));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x3424, x_desc, x3400, x3422, x_desc, x3400));
};
if (x3428) {
} else {
assert(false && "Image too small for averagePool_batch:  x Const(64) x Const(2048) x Sym(3387) x Sym(3387)|(2,2)");
}
float* x3433 = (float*)myMalloc(1 * sizeof(float));;
x3433[0] = 0.0f;
float* x3435 = (float*)myMalloc(1 * sizeof(float));;
x3435[0] = 1.0f;
float* x3445 = (float*)myGpuMalloc(x3444 * sizeof(float));

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3387, x3387) );

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3439, x3439));

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
    x3435, in_desc, x3400, x3433, out_desc, x3445));
};
float* x3447 = (float*)myGpuMalloc(x3444 * sizeof(float));
int32_t x3448 = 0;
int32_t x3449 = 1;
x3449 *= 64;
x3449 *= 2048;
int32_t x3452 = x3448;
bool x3453 = x3452 >= 2;
if (x3453) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3459 = x3452 == 0;
if (x3459) {
int32_t x3460 = x3449;
bool x3461 = x3460 == x3442;
if (x3461) {
} else {
assert(false && "must same size!!");
}
} else {
}
// foward of gemm
// gemm: List(Const(64), Const(2048)), Vector(Const(10), Const(2048))
float* x3470 = (float*)myGpuMalloc(640 * sizeof(float));
float* x3471 = (float*)myMalloc(1 * sizeof(float));;
x3471[0] = 0.0f;
float* x3473 = (float*)myMalloc(1 * sizeof(float));;
x3473[0] = 1.0f;
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 10,64,2048,x3473,x976,2048,x3445,2048,x3471,x3470,10));
float* x3476 = (float*)myGpuMalloc(640 * sizeof(float));
float* x3477 = (float*)myMalloc(1 * sizeof(float));;
x3477[0] = 1.0f;
float* x3479 = (float*)myMalloc(1 * sizeof(float));;
x3479[0] = 1.0f;

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
    cudnnHandle, x3477, bias_desc, x439, x3479, out_desc, x3470));
};
int32_t x3482 = 0;
int32_t x3483 = 1;
x3483 *= 64;
x3483 *= 10;
x3483 *= 1;
x3483 *= 1;
int32_t x3488 = x3482;
bool x3489 = x3488 >= 2;
if (x3489) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3494 = x3488 == 0;
if (x3494) {
int32_t x3495 = x3483;
bool x3496 = x3495 == 640;
if (x3496) {
} else {
assert(false && "must same size!!");
}
} else {
}
float* x3503 = (float*)myMalloc(1 * sizeof(float));;
x3503[0] = 0.0f;
float* x3505 = (float*)myMalloc(1 * sizeof(float));;
x3505[0] = 1.0f;
float* x3507 = (float*)myGpuMalloc(640 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 10, 1, 1));
CUDNN_CALL(cudnnSoftmaxForward(
    cudnnHandle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL,
    x3505, x_desc, x3470, x3503, x_desc, x3507));
};
int32_t x3509 = 0;
int32_t x3510 = 1;
x3510 *= 64;
x3510 *= 10;
int32_t x3513 = x3509;
bool x3514 = x3513 >= 2;
if (x3514) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3519 = x3513 == 0;
if (x3519) {
int32_t x3520 = x3510;
bool x3521 = x3520 == 640;
if (x3521) {
} else {
assert(false && "must same size!!");
}
} else {
}
float* x3528 = (float*)myGpuMalloc(640 * sizeof(float));
float* x3529 = (float*)myGpuMalloc(64 * sizeof(float));
nllLoss<<<64, 1>>>(x3507, 10, x3529, x1405);
float* x3531 = (float*)myGpuMalloc(64 * sizeof(float));
int32_t x3532 = 0;
int32_t x3533 = 1;
x3533 *= 64;
x3533 *= 1;
x3533 *= 1;
x3533 *= 1;
int32_t x3538 = x3532;
bool x3539 = x3538 >= 2;
if (x3539) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3544 = x3538 == 0;
if (x3544) {
int32_t x3545 = x3533;
bool x3546 = x3545 == 64;
if (x3546) {
} else {
assert(false && "must same size!!");
}
} else {
}
float* x3553 = (float*)myGpuMalloc(1 * sizeof(float));
float* x3554 = (float*)myMalloc(1 * sizeof(float));;
x3554[0] = 0.0f;
float* x3556 = (float*)myMalloc(1 * sizeof(float));;
x3556[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1, 1, 1));

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
    x3556, x_desc, x3529, x3554, out_desc, x3553));
};
int32_t x3559 = 0;
int32_t x3560 = 1;
x3560 *= 1;
int32_t x3562 = x3559;
bool x3563 = x3562 >= 2;
if (x3563) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3568 = x3562 == 0;
if (x3568) {
int32_t x3569 = x3560;
bool x3570 = x3569 == 1;
if (x3570) {
} else {
assert(false && "must same size!!");
}
} else {
}
float* x3577 = (float*)myGpuMalloc(1 * sizeof(float));
// make sure the size of loss is 1
arrayFill_greg<<<28, 512>>>(x3577, 1.0f, 1);
// backend is lantern.TensorDslCudnn$BackendCudnn@27c50b7b
CUDA_CALL(cudaMemcpy(x1410, x3553, 1 * sizeof(float), cudaMemcpyDeviceToHost));
// 'mean' gradient
// backprop for mean op
int32_t x3584 = 0;
int32_t x3585 = 1;
x3585 *= 1;
x3585 *= 1;
x3585 *= 1;
x3585 *= 1;
int32_t x3590 = x3584;
bool x3591 = x3590 >= 2;
if (x3591) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3596 = x3590 == 0;
if (x3596) {
int32_t x3597 = x3585;
bool x3598 = x3597 == 1;
if (x3598) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3605 = 0;
int32_t x3606 = 1;
x3606 *= 64;
x3606 *= 1;
x3606 *= 1;
x3606 *= 1;
int32_t x3611 = x3605;
bool x3612 = x3611 >= 2;
if (x3612) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3617 = x3611 == 0;
if (x3617) {
int32_t x3618 = x3606;
bool x3619 = x3618 == 64;
if (x3619) {
} else {
assert(false && "must same size!!");
}
} else {
}
if (x3631) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(1) x Const(1) x Const(1) x Const(1), res:  x Const(64) x Const(1) x Const(1) x Const(1)");
}
float* x3636 = (float*)myMalloc(1 * sizeof(float));;
x3636[0] = x3626;
float* x3638 = (float*)myMalloc(1 * sizeof(float));;
x3638[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1, 1, 1));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x3636, bias_desc, x3577, x3638, out_desc, x3531));
};
// 'nllLossB' gradient.
nllLoss_grad<<<64, 1>>>(10, x3531, x1405, x3528);
int32_t x3643 = 0;
int32_t x3644 = 1;
x3644 *= 64;
x3644 *= 10;
x3644 *= 1;
x3644 *= 1;
int32_t x3649 = x3643;
bool x3650 = x3649 >= 2;
if (x3650) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3655 = x3649 == 0;
if (x3655) {
int32_t x3656 = x3644;
bool x3657 = x3656 == 640;
if (x3657) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3664 = 0;
int32_t x3665 = 1;
x3665 *= 64;
x3665 *= 10;
x3665 *= 1;
x3665 *= 1;
int32_t x3670 = x3664;
bool x3671 = x3670 >= 2;
if (x3671) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3676 = x3670 == 0;
if (x3676) {
int32_t x3677 = x3665;
bool x3678 = x3677 == 640;
if (x3678) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3685 = 0;
int32_t x3686 = 1;
x3686 *= 64;
x3686 *= 10;
x3686 *= 1;
x3686 *= 1;
int32_t x3691 = x3685;
bool x3692 = x3691 >= 2;
if (x3692) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3697 = x3691 == 0;
if (x3697) {
int32_t x3698 = x3686;
bool x3699 = x3698 == 640;
if (x3699) {
} else {
assert(false && "must same size!!");
}
} else {
}
int32_t x3706 = 0;
int32_t x3707 = 1;
x3707 *= 64;
x3707 *= 10;
x3707 *= 1;
x3707 *= 1;
int32_t x3712 = x3706;
bool x3713 = x3712 >= 2;
if (x3713) {
printf("cannot have 2 or more -1s in resize!!\n");
assert(false && "");
} else {
}
bool x3718 = x3712 == 0;
if (x3718) {
int32_t x3719 = x3707;
bool x3720 = x3719 == 640;
if (x3720) {
} else {
assert(false && "must same size!!");
}
} else {
}
float* x3727 = (float*)myMalloc(1 * sizeof(float));;
x3727[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 10, 1, 1));
CUDNN_CALL(cudnnSoftmaxBackward(
    cudnnHandle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL,
    x3727, x_desc, x3507, x_desc, x3528,
    x3727, x_desc, x3476));
};
float* x3730 = (float*)myMalloc(1 * sizeof(float));;
x3730[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 10, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 10, 1, 1));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x3730, grad_out_desc, x3476,
    x3730, grad_bias_desc, x1155));
};
// backprop for gemm List(Const(64), Const(2048)), Vector(Const(10), Const(2048))
float* x3734 = (float*)myMalloc(1 * sizeof(float));;
x3734[0] = 1.0f;
float* x3736 = (float*)myMalloc(1 * sizeof(float));;
x3736[0] = 1.0f;
// backprop of gemm
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2048,64,10,x3734,x976,2048,x3476,10,x3736,x3447,2048));
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 2048,10,64,x3734,x3445,2048,x3476,10,x3736,x1334,2048));
float* x3741 = (float*)myMalloc(1 * sizeof(float));;
x3741[0] = 0.0f;
float* x3743 = (float*)myMalloc(1 * sizeof(float));;
x3743[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3387, x3387));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3439, x3439));

cudnnPoolingDescriptor_t poolingDesc;
CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));
CUDNN_CALL(cudnnSetPooling2dDescriptor(
    poolingDesc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN,
    2, 2, 0,
    0, 1, 1
));
CUDNN_CALL(cudnnPoolingBackward(
    cudnnHandle, 
    poolingDesc, 
    x3743, out_desc, x3445, out_desc, x3447, in_desc, x3400  , x3741, in_desc, x3408));
};
float* x3746 = (float*)myMalloc(1 * sizeof(float));;
x3746[0] = 1.0f;
float* x3748 = (float*)myMalloc(1 * sizeof(float));;
x3748[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3387, x3387));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3746, x_desc, x3400, x_desc, x3408, x_desc, x3400,
    x3748, x_desc, x3408));
};
if (x3752) {
if (x3755) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(3387) x Sym(3387), res:  x Const(64) x Const(2048) x Sym(3270) x Sym(3270)");
}
float* x3760 = (float*)myMalloc(1 * sizeof(float));;
x3760[0] = 1.0f;
float* x3762 = (float*)myMalloc(1 * sizeof(float));;
x3762[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3387, x3387));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3270, x3270));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x3760, bias_desc, x3408, x3762, out_desc, x3291));
};
} else {
float* x3766 = (float*)myMalloc(1 * sizeof(float));;
x3766[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3270, x3270));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3387, x3387));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x3766, grad_out_desc, x3408,
    x3766, grad_bias_desc, x3291));
};
}
float* x3771 = (float*)myMalloc(1 * sizeof(float));;
x3771[0] = 0.0f;
float* x3773 = (float*)myMalloc(1 * sizeof(float));;
x3773[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3387, x3387));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3387, x3387));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3773, x3773, x3773, x3773, in_desc, x3393,
    out_desc, x3408, in_desc, x3399, sbmv_desc, x604,
    x1210,x1288, 1.0E-5, x3401, x3402));
};
// conv2D back-propagate
float* x3777 = (float*)myMalloc(1 * sizeof(float));;
x3777[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3353, x3353));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3387, x3387));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x3777, filt_desc, x394, grad_out_desc, x3399,
    conv_desc, algo, ws_data, ws_size,
    x3777, grad_in_desc, x3374));
};
float* x3780 = (float*)myMalloc(1 * sizeof(float));;
x3780[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3387, x3387));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3353, x3353));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x3780, in_desc, x3366, grad_out_desc, x3399,
    conv_desc, algo, ws_data, ws_size,
    x3780, grad_filt_desc, x1140));
};
float* x3783 = (float*)myMalloc(1 * sizeof(float));;
x3783[0] = 1.0f;
float* x3785 = (float*)myMalloc(1 * sizeof(float));;
x3785[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3353, x3353));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3783, x_desc, x3366, x_desc, x3374, x_desc, x3366,
    x3785, x_desc, x3374));
};
float* x3788 = (float*)myMalloc(1 * sizeof(float));;
x3788[0] = 0.0f;
float* x3790 = (float*)myMalloc(1 * sizeof(float));;
x3790[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3353, x3353));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3353, x3353));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3790, x3790, x3790, x3790, in_desc, x3359,
    out_desc, x3374, in_desc, x3365, sbmv_desc, x877,
    x1301,x1276, 1.0E-5, x3367, x3368));
};
// conv2D back-propagate
float* x3794 = (float*)myMalloc(1 * sizeof(float));;
x3794[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3317, x3317));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3353, x3353));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x3794, filt_desc, x379, grad_out_desc, x3365,
    conv_desc, algo, ws_data, ws_size,
    x3794, grad_in_desc, x3338));
};
float* x3797 = (float*)myMalloc(1 * sizeof(float));;
x3797[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3353, x3353));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3317, x3317));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x3797, in_desc, x3330, grad_out_desc, x3365,
    conv_desc, algo, ws_data, ws_size,
    x3797, grad_filt_desc, x1135));
};
float* x3800 = (float*)myMalloc(1 * sizeof(float));;
x3800[0] = 1.0f;
float* x3802 = (float*)myMalloc(1 * sizeof(float));;
x3802[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3317, x3317));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3800, x_desc, x3330, x_desc, x3338, x_desc, x3330,
    x3802, x_desc, x3338));
};
float* x3805 = (float*)myMalloc(1 * sizeof(float));;
x3805[0] = 0.0f;
float* x3807 = (float*)myMalloc(1 * sizeof(float));;
x3807[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3317, x3317));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3317, x3317));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3807, x3807, x3807, x3807, in_desc, x3323,
    out_desc, x3338, in_desc, x3329, sbmv_desc, x340,
    x1122,x1185, 1.0E-5, x3331, x3332));
};
// conv2D back-propagate
float* x3811 = (float*)myMalloc(1 * sizeof(float));;
x3811[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 2048, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3270, x3270));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3317, x3317));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x3811, filt_desc, x1090, grad_out_desc, x3329,
    conv_desc, algo, ws_data, ws_size,
    x3811, grad_in_desc, x3291));
};
float* x3814 = (float*)myMalloc(1 * sizeof(float));;
x3814[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 2048, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3317, x3317));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3270, x3270));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x3814, in_desc, x3283, grad_out_desc, x3329,
    conv_desc, algo, ws_data, ws_size,
    x3814, grad_filt_desc, x1372));
};
float* x3817 = (float*)myMalloc(1 * sizeof(float));;
x3817[0] = 1.0f;
float* x3819 = (float*)myMalloc(1 * sizeof(float));;
x3819[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3270, x3270));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3817, x_desc, x3283, x_desc, x3291, x_desc, x3283,
    x3819, x_desc, x3291));
};
if (x3823) {
if (x3825) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(3270) x Sym(3270), res:  x Const(64) x Const(2048) x Sym(3127) x Sym(3127)");
}
float* x3830 = (float*)myMalloc(1 * sizeof(float));;
x3830[0] = 1.0f;
float* x3832 = (float*)myMalloc(1 * sizeof(float));;
x3832[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3270, x3270));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3127, x3127));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x3830, bias_desc, x3291, x3832, out_desc, x3148));
};
} else {
float* x3836 = (float*)myMalloc(1 * sizeof(float));;
x3836[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3127, x3127));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3270, x3270));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x3836, grad_out_desc, x3291,
    x3836, grad_bias_desc, x3148));
};
}
float* x3841 = (float*)myMalloc(1 * sizeof(float));;
x3841[0] = 0.0f;
float* x3843 = (float*)myMalloc(1 * sizeof(float));;
x3843[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3270, x3270));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3270, x3270));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3843, x3843, x3843, x3843, in_desc, x3276,
    out_desc, x3291, in_desc, x3282, sbmv_desc, x577,
    x1201,x1251, 1.0E-5, x3284, x3285));
};
// conv2D back-propagate
float* x3847 = (float*)myMalloc(1 * sizeof(float));;
x3847[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3236, x3236));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3270, x3270));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x3847, filt_desc, x436, grad_out_desc, x3282,
    conv_desc, algo, ws_data, ws_size,
    x3847, grad_in_desc, x3257));
};
float* x3850 = (float*)myMalloc(1 * sizeof(float));;
x3850[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3270, x3270));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3236, x3236));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x3850, in_desc, x3249, grad_out_desc, x3282,
    conv_desc, algo, ws_data, ws_size,
    x3850, grad_filt_desc, x1154));
};
float* x3853 = (float*)myMalloc(1 * sizeof(float));;
x3853[0] = 1.0f;
float* x3855 = (float*)myMalloc(1 * sizeof(float));;
x3855[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3236, x3236));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3853, x_desc, x3249, x_desc, x3257, x_desc, x3249,
    x3855, x_desc, x3257));
};
float* x3858 = (float*)myMalloc(1 * sizeof(float));;
x3858[0] = 0.0f;
float* x3860 = (float*)myMalloc(1 * sizeof(float));;
x3860[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3236, x3236));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3236, x3236));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3860, x3860, x3860, x3860, in_desc, x3242,
    out_desc, x3257, in_desc, x3248, sbmv_desc, x775,
    x1267,x1173, 1.0E-5, x3250, x3251));
};
// conv2D back-propagate
float* x3864 = (float*)myMalloc(1 * sizeof(float));;
x3864[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3200, x3200));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3236, x3236));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x3864, filt_desc, x760, grad_out_desc, x3248,
    conv_desc, algo, ws_data, ws_size,
    x3864, grad_in_desc, x3221));
};
float* x3867 = (float*)myMalloc(1 * sizeof(float));;
x3867[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3236, x3236));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3200, x3200));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x3867, in_desc, x3213, grad_out_desc, x3248,
    conv_desc, algo, ws_data, ws_size,
    x3867, grad_filt_desc, x1262));
};
float* x3870 = (float*)myMalloc(1 * sizeof(float));;
x3870[0] = 1.0f;
float* x3872 = (float*)myMalloc(1 * sizeof(float));;
x3872[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3200, x3200));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3870, x_desc, x3213, x_desc, x3221, x_desc, x3213,
    x3872, x_desc, x3221));
};
float* x3875 = (float*)myMalloc(1 * sizeof(float));;
x3875[0] = 0.0f;
float* x3877 = (float*)myMalloc(1 * sizeof(float));;
x3877[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3200, x3200));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3200, x3200));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3877, x3877, x3877, x3877, in_desc, x3206,
    out_desc, x3221, in_desc, x3212, sbmv_desc, x433,
    x1153,x1244, 1.0E-5, x3214, x3215));
};
// conv2D back-propagate
float* x3881 = (float*)myMalloc(1 * sizeof(float));;
x3881[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 2048, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3127, x3127));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3200, x3200));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x3881, filt_desc, x940, grad_out_desc, x3212,
    conv_desc, algo, ws_data, ws_size,
    x3881, grad_in_desc, x3148));
};
float* x3884 = (float*)myMalloc(1 * sizeof(float));;
x3884[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 2048, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3200, x3200));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3127, x3127));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x3884, in_desc, x3140, grad_out_desc, x3212,
    conv_desc, algo, ws_data, ws_size,
    x3884, grad_filt_desc, x1322));
};
float* x3887 = (float*)myMalloc(1 * sizeof(float));;
x3887[0] = 1.0f;
float* x3889 = (float*)myMalloc(1 * sizeof(float));;
x3889[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3127, x3127));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3887, x_desc, x3140, x_desc, x3148, x_desc, x3140,
    x3889, x_desc, x3148));
};
if (x3893) {
if (x3895) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(3127) x Sym(3127), res:  x Const(64) x Const(2048) x Sym(3153) x Sym(3153)");
}
float* x3900 = (float*)myMalloc(1 * sizeof(float));;
x3900[0] = 1.0f;
float* x3902 = (float*)myMalloc(1 * sizeof(float));;
x3902[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3127, x3127));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3153, x3153));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x3900, bias_desc, x3148, x3902, out_desc, x3174));
};
} else {
float* x3906 = (float*)myMalloc(1 * sizeof(float));;
x3906[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3153, x3153));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3127, x3127));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x3906, grad_out_desc, x3148,
    x3906, grad_bias_desc, x3174));
};
}
float* x3911 = (float*)myMalloc(1 * sizeof(float));;
x3911[0] = 0.0f;
float* x3913 = (float*)myMalloc(1 * sizeof(float));;
x3913[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3153, x3153));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3153, x3153));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3913, x3913, x3913, x3913, in_desc, x3159,
    out_desc, x3174, in_desc, x3165, sbmv_desc, x814,
    x1280,x1214, 1.0E-5, x3167, x3168));
};
// conv2D back-propagate
float* x3917 = (float*)myMalloc(1 * sizeof(float));;
x3917[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 1024, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x3010, x3010));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3153, x3153));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 2, 2, 1, 1,
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
    x3917, filt_desc, x937, grad_out_desc, x3165,
    conv_desc, algo, ws_data, ws_size,
    x3917, grad_in_desc, x3031));
};
float* x3920 = (float*)myMalloc(1 * sizeof(float));;
x3920[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 1024, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3153, x3153));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x3010, x3010));

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
    x3920, in_desc, x3023, grad_out_desc, x3165,
    conv_desc, algo, ws_data, ws_size,
    x3920, grad_filt_desc, x1321));
};
float* x3923 = (float*)myMalloc(1 * sizeof(float));;
x3923[0] = 0.0f;
float* x3925 = (float*)myMalloc(1 * sizeof(float));;
x3925[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3127, x3127));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3127, x3127));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3925, x3925, x3925, x3925, in_desc, x3133,
    out_desc, x3148, in_desc, x3139, sbmv_desc, x1012,
    x1346,x1169, 1.0E-5, x3141, x3142));
};
// conv2D back-propagate
float* x3929 = (float*)myMalloc(1 * sizeof(float));;
x3929[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3093, x3093));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3127, x3127));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x3929, filt_desc, x931, grad_out_desc, x3139,
    conv_desc, algo, ws_data, ws_size,
    x3929, grad_in_desc, x3114));
};
float* x3932 = (float*)myMalloc(1 * sizeof(float));;
x3932[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, x3127, x3127));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3093, x3093));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x3932, in_desc, x3106, grad_out_desc, x3139,
    conv_desc, algo, ws_data, ws_size,
    x3932, grad_filt_desc, x1319));
};
float* x3935 = (float*)myMalloc(1 * sizeof(float));;
x3935[0] = 1.0f;
float* x3937 = (float*)myMalloc(1 * sizeof(float));;
x3937[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3093, x3093));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3935, x_desc, x3106, x_desc, x3114, x_desc, x3106,
    x3937, x_desc, x3114));
};
float* x3940 = (float*)myMalloc(1 * sizeof(float));;
x3940[0] = 0.0f;
float* x3942 = (float*)myMalloc(1 * sizeof(float));;
x3942[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3093, x3093));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3093, x3093));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3942, x3942, x3942, x3942, in_desc, x3099,
    out_desc, x3114, in_desc, x3105, sbmv_desc, x910,
    x1312,x1266, 1.0E-5, x3107, x3108));
};
// conv2D back-propagate
float* x3946 = (float*)myMalloc(1 * sizeof(float));;
x3946[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3057, x3057));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3093, x3093));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 2, 2, 1, 1,
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
    x3946, filt_desc, x397, grad_out_desc, x3105,
    conv_desc, algo, ws_data, ws_size,
    x3946, grad_in_desc, x3078));
};
float* x3949 = (float*)myMalloc(1 * sizeof(float));;
x3949[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3093, x3093));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3057, x3057));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 2, 2, 1, 1,
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
    x3949, in_desc, x3070, grad_out_desc, x3105,
    conv_desc, algo, ws_data, ws_size,
    x3949, grad_filt_desc, x1141));
};
float* x3952 = (float*)myMalloc(1 * sizeof(float));;
x3952[0] = 1.0f;
float* x3954 = (float*)myMalloc(1 * sizeof(float));;
x3954[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3057, x3057));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3952, x_desc, x3070, x_desc, x3078, x_desc, x3070,
    x3954, x_desc, x3078));
};
float* x3957 = (float*)myMalloc(1 * sizeof(float));;
x3957[0] = 0.0f;
float* x3959 = (float*)myMalloc(1 * sizeof(float));;
x3959[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3057, x3057));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3057, x3057));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3959, x3959, x3959, x3959, in_desc, x3063,
    out_desc, x3078, in_desc, x3069, sbmv_desc, x898,
    x1308,x1331, 1.0E-5, x3071, x3072));
};
// conv2D back-propagate
float* x3963 = (float*)myMalloc(1 * sizeof(float));;
x3963[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 1024, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x3010, x3010));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3057, x3057));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x3963, filt_desc, x712, grad_out_desc, x3069,
    conv_desc, algo, ws_data, ws_size,
    x3963, grad_in_desc, x3031));
};
float* x3966 = (float*)myMalloc(1 * sizeof(float));;
x3966[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 1024, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x3057, x3057));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x3010, x3010));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x3966, in_desc, x3023, grad_out_desc, x3069,
    conv_desc, algo, ws_data, ws_size,
    x3966, grad_filt_desc, x1246));
};
float* x3969 = (float*)myMalloc(1 * sizeof(float));;
x3969[0] = 1.0f;
float* x3971 = (float*)myMalloc(1 * sizeof(float));;
x3971[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x3010, x3010));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3969, x_desc, x3023, x_desc, x3031, x_desc, x3023,
    x3971, x_desc, x3031));
};
if (x3975) {
if (x3978) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(3010) x Sym(3010), res:  x Const(64) x Const(1024) x Sym(2893) x Sym(2893)");
}
float* x3983 = (float*)myMalloc(1 * sizeof(float));;
x3983[0] = 1.0f;
float* x3985 = (float*)myMalloc(1 * sizeof(float));;
x3985[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x3010, x3010));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2893, x2893));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x3983, bias_desc, x3031, x3985, out_desc, x2914));
};
} else {
float* x3989 = (float*)myMalloc(1 * sizeof(float));;
x3989[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2893, x2893));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x3010, x3010));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x3989, grad_out_desc, x3031,
    x3989, grad_bias_desc, x2914));
};
}
float* x3994 = (float*)myMalloc(1 * sizeof(float));;
x3994[0] = 0.0f;
float* x3996 = (float*)myMalloc(1 * sizeof(float));;
x3996[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x3010, x3010));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x3010, x3010));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3996, x3996, x3996, x3996, in_desc, x3016,
    out_desc, x3031, in_desc, x3022, sbmv_desc, x1039,
    x1355,x1200, 1.0E-5, x3024, x3025));
};
// conv2D back-propagate
float* x4000 = (float*)myMalloc(1 * sizeof(float));;
x4000[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2976, x2976));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x3010, x3010));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4000, filt_desc, x586, grad_out_desc, x3022,
    conv_desc, algo, ws_data, ws_size,
    x4000, grad_in_desc, x2997));
};
float* x4003 = (float*)myMalloc(1 * sizeof(float));;
x4003[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x3010, x3010));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2976, x2976));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4003, in_desc, x2989, grad_out_desc, x3022,
    conv_desc, algo, ws_data, ws_size,
    x4003, grad_filt_desc, x1204));
};
float* x4006 = (float*)myMalloc(1 * sizeof(float));;
x4006[0] = 1.0f;
float* x4008 = (float*)myMalloc(1 * sizeof(float));;
x4008[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2976, x2976));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4006, x_desc, x2989, x_desc, x2997, x_desc, x2989,
    x4008, x_desc, x2997));
};
float* x4011 = (float*)myMalloc(1 * sizeof(float));;
x4011[0] = 0.0f;
float* x4013 = (float*)myMalloc(1 * sizeof(float));;
x4013[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2976, x2976));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2976, x2976));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4013, x4013, x4013, x4013, in_desc, x2982,
    out_desc, x2997, in_desc, x2988, sbmv_desc, x718,
    x1248,x1296, 1.0E-5, x2990, x2991));
};
// conv2D back-propagate
float* x4017 = (float*)myMalloc(1 * sizeof(float));;
x4017[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2940, x2940));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2976, x2976));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4017, filt_desc, x973, grad_out_desc, x2988,
    conv_desc, algo, ws_data, ws_size,
    x4017, grad_in_desc, x2961));
};
float* x4020 = (float*)myMalloc(1 * sizeof(float));;
x4020[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2976, x2976));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2940, x2940));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4020, in_desc, x2953, grad_out_desc, x2988,
    conv_desc, algo, ws_data, ws_size,
    x4020, grad_filt_desc, x1333));
};
float* x4023 = (float*)myMalloc(1 * sizeof(float));;
x4023[0] = 1.0f;
float* x4025 = (float*)myMalloc(1 * sizeof(float));;
x4025[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2940, x2940));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4023, x_desc, x2953, x_desc, x2961, x_desc, x2953,
    x4025, x_desc, x2961));
};
float* x4028 = (float*)myMalloc(1 * sizeof(float));;
x4028[0] = 0.0f;
float* x4030 = (float*)myMalloc(1 * sizeof(float));;
x4030[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2940, x2940));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2940, x2940));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4030, x4030, x4030, x4030, in_desc, x2946,
    out_desc, x2961, in_desc, x2952, sbmv_desc, x550,
    x1192,x1360, 1.0E-5, x2954, x2955));
};
// conv2D back-propagate
float* x4034 = (float*)myMalloc(1 * sizeof(float));;
x4034[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2893, x2893));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2940, x2940));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4034, filt_desc, x748, grad_out_desc, x2952,
    conv_desc, algo, ws_data, ws_size,
    x4034, grad_in_desc, x2914));
};
float* x4037 = (float*)myMalloc(1 * sizeof(float));;
x4037[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2940, x2940));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2893, x2893));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4037, in_desc, x2906, grad_out_desc, x2952,
    conv_desc, algo, ws_data, ws_size,
    x4037, grad_filt_desc, x1258));
};
float* x4040 = (float*)myMalloc(1 * sizeof(float));;
x4040[0] = 1.0f;
float* x4042 = (float*)myMalloc(1 * sizeof(float));;
x4042[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2893, x2893));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4040, x_desc, x2906, x_desc, x2914, x_desc, x2906,
    x4042, x_desc, x2914));
};
if (x4046) {
if (x4048) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2893) x Sym(2893), res:  x Const(64) x Const(1024) x Sym(2776) x Sym(2776)");
}
float* x4053 = (float*)myMalloc(1 * sizeof(float));;
x4053[0] = 1.0f;
float* x4055 = (float*)myMalloc(1 * sizeof(float));;
x4055[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2893, x2893));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2776, x2776));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x4053, bias_desc, x2914, x4055, out_desc, x2797));
};
} else {
float* x4059 = (float*)myMalloc(1 * sizeof(float));;
x4059[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2776, x2776));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2893, x2893));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x4059, grad_out_desc, x2914,
    x4059, grad_bias_desc, x2797));
};
}
float* x4064 = (float*)myMalloc(1 * sizeof(float));;
x4064[0] = 0.0f;
float* x4066 = (float*)myMalloc(1 * sizeof(float));;
x4066[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2893, x2893));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2893, x2893));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4066, x4066, x4066, x4066, in_desc, x2899,
    out_desc, x2914, in_desc, x2905, sbmv_desc, x472,
    x1166,x1227, 1.0E-5, x2907, x2908));
};
// conv2D back-propagate
float* x4070 = (float*)myMalloc(1 * sizeof(float));;
x4070[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2859, x2859));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2893, x2893));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4070, filt_desc, x958, grad_out_desc, x2905,
    conv_desc, algo, ws_data, ws_size,
    x4070, grad_in_desc, x2880));
};
float* x4073 = (float*)myMalloc(1 * sizeof(float));;
x4073[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2893, x2893));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2859, x2859));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4073, in_desc, x2872, grad_out_desc, x2905,
    conv_desc, algo, ws_data, ws_size,
    x4073, grad_filt_desc, x1328));
};
float* x4076 = (float*)myMalloc(1 * sizeof(float));;
x4076[0] = 1.0f;
float* x4078 = (float*)myMalloc(1 * sizeof(float));;
x4078[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2859, x2859));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4076, x_desc, x2872, x_desc, x2880, x_desc, x2872,
    x4078, x_desc, x2880));
};
float* x4081 = (float*)myMalloc(1 * sizeof(float));;
x4081[0] = 0.0f;
float* x4083 = (float*)myMalloc(1 * sizeof(float));;
x4083[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2859, x2859));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2859, x2859));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4083, x4083, x4083, x4083, in_desc, x2865,
    out_desc, x2880, in_desc, x2871, sbmv_desc, x799,
    x1275,x1216, 1.0E-5, x2873, x2874));
};
// conv2D back-propagate
float* x4087 = (float*)myMalloc(1 * sizeof(float));;
x4087[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2823, x2823));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2859, x2859));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4087, filt_desc, x1081, grad_out_desc, x2871,
    conv_desc, algo, ws_data, ws_size,
    x4087, grad_in_desc, x2844));
};
float* x4090 = (float*)myMalloc(1 * sizeof(float));;
x4090[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2859, x2859));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2823, x2823));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4090, in_desc, x2836, grad_out_desc, x2871,
    conv_desc, algo, ws_data, ws_size,
    x4090, grad_filt_desc, x1369));
};
float* x4093 = (float*)myMalloc(1 * sizeof(float));;
x4093[0] = 1.0f;
float* x4095 = (float*)myMalloc(1 * sizeof(float));;
x4095[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2823, x2823));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4093, x_desc, x2836, x_desc, x2844, x_desc, x2836,
    x4095, x_desc, x2844));
};
float* x4098 = (float*)myMalloc(1 * sizeof(float));;
x4098[0] = 0.0f;
float* x4100 = (float*)myMalloc(1 * sizeof(float));;
x4100[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2823, x2823));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2823, x2823));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4100, x4100, x4100, x4100, in_desc, x2829,
    out_desc, x2844, in_desc, x2835, sbmv_desc, x526,
    x1184,x1292, 1.0E-5, x2837, x2838));
};
// conv2D back-propagate
float* x4104 = (float*)myMalloc(1 * sizeof(float));;
x4104[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2776, x2776));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2823, x2823));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4104, filt_desc, x361, grad_out_desc, x2835,
    conv_desc, algo, ws_data, ws_size,
    x4104, grad_in_desc, x2797));
};
float* x4107 = (float*)myMalloc(1 * sizeof(float));;
x4107[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2823, x2823));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2776, x2776));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4107, in_desc, x2789, grad_out_desc, x2835,
    conv_desc, algo, ws_data, ws_size,
    x4107, grad_filt_desc, x1129));
};
float* x4110 = (float*)myMalloc(1 * sizeof(float));;
x4110[0] = 1.0f;
float* x4112 = (float*)myMalloc(1 * sizeof(float));;
x4112[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2776, x2776));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4110, x_desc, x2789, x_desc, x2797, x_desc, x2789,
    x4112, x_desc, x2797));
};
if (x4116) {
if (x4118) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2776) x Sym(2776), res:  x Const(64) x Const(1024) x Sym(2659) x Sym(2659)");
}
float* x4123 = (float*)myMalloc(1 * sizeof(float));;
x4123[0] = 1.0f;
float* x4125 = (float*)myMalloc(1 * sizeof(float));;
x4125[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2776, x2776));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2659, x2659));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x4123, bias_desc, x2797, x4125, out_desc, x2680));
};
} else {
float* x4129 = (float*)myMalloc(1 * sizeof(float));;
x4129[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2659, x2659));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2776, x2776));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x4129, grad_out_desc, x2797,
    x4129, grad_bias_desc, x2680));
};
}
float* x4134 = (float*)myMalloc(1 * sizeof(float));;
x4134[0] = 0.0f;
float* x4136 = (float*)myMalloc(1 * sizeof(float));;
x4136[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2776, x2776));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2776, x2776));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4136, x4136, x4136, x4136, in_desc, x2782,
    out_desc, x2797, in_desc, x2788, sbmv_desc, x1009,
    x1345,x1253, 1.0E-5, x2790, x2791));
};
// conv2D back-propagate
float* x4140 = (float*)myMalloc(1 * sizeof(float));;
x4140[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2742, x2742));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2776, x2776));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4140, filt_desc, x562, grad_out_desc, x2788,
    conv_desc, algo, ws_data, ws_size,
    x4140, grad_in_desc, x2763));
};
float* x4143 = (float*)myMalloc(1 * sizeof(float));;
x4143[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2776, x2776));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2742, x2742));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4143, in_desc, x2755, grad_out_desc, x2788,
    conv_desc, algo, ws_data, ws_size,
    x4143, grad_filt_desc, x1196));
};
float* x4146 = (float*)myMalloc(1 * sizeof(float));;
x4146[0] = 1.0f;
float* x4148 = (float*)myMalloc(1 * sizeof(float));;
x4148[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2742, x2742));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4146, x_desc, x2755, x_desc, x2763, x_desc, x2755,
    x4148, x_desc, x2763));
};
float* x4151 = (float*)myMalloc(1 * sizeof(float));;
x4151[0] = 0.0f;
float* x4153 = (float*)myMalloc(1 * sizeof(float));;
x4153[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2742, x2742));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2742, x2742));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4153, x4153, x4153, x4153, in_desc, x2748,
    out_desc, x2763, in_desc, x2754, sbmv_desc, x517,
    x1181,x1243, 1.0E-5, x2756, x2757));
};
// conv2D back-propagate
float* x4157 = (float*)myMalloc(1 * sizeof(float));;
x4157[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2706, x2706));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2742, x2742));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4157, filt_desc, x1042, grad_out_desc, x2754,
    conv_desc, algo, ws_data, ws_size,
    x4157, grad_in_desc, x2727));
};
float* x4160 = (float*)myMalloc(1 * sizeof(float));;
x4160[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2742, x2742));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2706, x2706));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4160, in_desc, x2719, grad_out_desc, x2754,
    conv_desc, algo, ws_data, ws_size,
    x4160, grad_filt_desc, x1356));
};
float* x4163 = (float*)myMalloc(1 * sizeof(float));;
x4163[0] = 1.0f;
float* x4165 = (float*)myMalloc(1 * sizeof(float));;
x4165[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2706, x2706));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4163, x_desc, x2719, x_desc, x2727, x_desc, x2719,
    x4165, x_desc, x2727));
};
float* x4168 = (float*)myMalloc(1 * sizeof(float));;
x4168[0] = 0.0f;
float* x4170 = (float*)myMalloc(1 * sizeof(float));;
x4170[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2706, x2706));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2706, x2706));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4170, x4170, x4170, x4170, in_desc, x2712,
    out_desc, x2727, in_desc, x2718, sbmv_desc, x571,
    x1199,x1348, 1.0E-5, x2720, x2721));
};
// conv2D back-propagate
float* x4174 = (float*)myMalloc(1 * sizeof(float));;
x4174[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2659, x2659));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2706, x2706));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4174, filt_desc, x313, grad_out_desc, x2718,
    conv_desc, algo, ws_data, ws_size,
    x4174, grad_in_desc, x2680));
};
float* x4177 = (float*)myMalloc(1 * sizeof(float));;
x4177[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2706, x2706));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2659, x2659));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4177, in_desc, x2672, grad_out_desc, x2718,
    conv_desc, algo, ws_data, ws_size,
    x4177, grad_filt_desc, x1113));
};
float* x4180 = (float*)myMalloc(1 * sizeof(float));;
x4180[0] = 1.0f;
float* x4182 = (float*)myMalloc(1 * sizeof(float));;
x4182[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2659, x2659));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4180, x_desc, x2672, x_desc, x2680, x_desc, x2672,
    x4182, x_desc, x2680));
};
if (x4186) {
if (x4188) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2659) x Sym(2659), res:  x Const(64) x Const(1024) x Sym(2542) x Sym(2542)");
}
float* x4193 = (float*)myMalloc(1 * sizeof(float));;
x4193[0] = 1.0f;
float* x4195 = (float*)myMalloc(1 * sizeof(float));;
x4195[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2659, x2659));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2542, x2542));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x4193, bias_desc, x2680, x4195, out_desc, x2563));
};
} else {
float* x4199 = (float*)myMalloc(1 * sizeof(float));;
x4199[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2542, x2542));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2659, x2659));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x4199, grad_out_desc, x2680,
    x4199, grad_bias_desc, x2563));
};
}
float* x4204 = (float*)myMalloc(1 * sizeof(float));;
x4204[0] = 0.0f;
float* x4206 = (float*)myMalloc(1 * sizeof(float));;
x4206[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2659, x2659));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2659, x2659));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4206, x4206, x4206, x4206, in_desc, x2665,
    out_desc, x2680, in_desc, x2671, sbmv_desc, x1084,
    x1370,x1164, 1.0E-5, x2673, x2674));
};
// conv2D back-propagate
float* x4210 = (float*)myMalloc(1 * sizeof(float));;
x4210[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2625, x2625));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2659, x2659));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4210, filt_desc, x643, grad_out_desc, x2671,
    conv_desc, algo, ws_data, ws_size,
    x4210, grad_in_desc, x2646));
};
float* x4213 = (float*)myMalloc(1 * sizeof(float));;
x4213[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2659, x2659));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2625, x2625));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4213, in_desc, x2638, grad_out_desc, x2671,
    conv_desc, algo, ws_data, ws_size,
    x4213, grad_filt_desc, x1223));
};
float* x4216 = (float*)myMalloc(1 * sizeof(float));;
x4216[0] = 1.0f;
float* x4218 = (float*)myMalloc(1 * sizeof(float));;
x4218[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2625, x2625));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4216, x_desc, x2638, x_desc, x2646, x_desc, x2638,
    x4218, x_desc, x2646));
};
float* x4221 = (float*)myMalloc(1 * sizeof(float));;
x4221[0] = 0.0f;
float* x4223 = (float*)myMalloc(1 * sizeof(float));;
x4223[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2625, x2625));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2625, x2625));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4223, x4223, x4223, x4223, in_desc, x2631,
    out_desc, x2646, in_desc, x2637, sbmv_desc, x979,
    x1335,x1299, 1.0E-5, x2639, x2640));
};
// conv2D back-propagate
float* x4227 = (float*)myMalloc(1 * sizeof(float));;
x4227[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2589, x2589));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2625, x2625));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4227, filt_desc, x337, grad_out_desc, x2637,
    conv_desc, algo, ws_data, ws_size,
    x4227, grad_in_desc, x2610));
};
float* x4230 = (float*)myMalloc(1 * sizeof(float));;
x4230[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2625, x2625));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2589, x2589));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4230, in_desc, x2602, grad_out_desc, x2637,
    conv_desc, algo, ws_data, ws_size,
    x4230, grad_filt_desc, x1121));
};
float* x4233 = (float*)myMalloc(1 * sizeof(float));;
x4233[0] = 1.0f;
float* x4235 = (float*)myMalloc(1 * sizeof(float));;
x4235[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2589, x2589));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4233, x_desc, x2602, x_desc, x2610, x_desc, x2602,
    x4235, x_desc, x2610));
};
float* x4238 = (float*)myMalloc(1 * sizeof(float));;
x4238[0] = 0.0f;
float* x4240 = (float*)myMalloc(1 * sizeof(float));;
x4240[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2589, x2589));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2589, x2589));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4240, x4240, x4240, x4240, in_desc, x2595,
    out_desc, x2610, in_desc, x2601, sbmv_desc, x682,
    x1236,x1304, 1.0E-5, x2603, x2604));
};
// conv2D back-propagate
float* x4244 = (float*)myMalloc(1 * sizeof(float));;
x4244[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2542, x2542));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2589, x2589));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4244, filt_desc, x949, grad_out_desc, x2601,
    conv_desc, algo, ws_data, ws_size,
    x4244, grad_in_desc, x2563));
};
float* x4247 = (float*)myMalloc(1 * sizeof(float));;
x4247[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2589, x2589));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2542, x2542));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4247, in_desc, x2555, grad_out_desc, x2601,
    conv_desc, algo, ws_data, ws_size,
    x4247, grad_filt_desc, x1325));
};
float* x4250 = (float*)myMalloc(1 * sizeof(float));;
x4250[0] = 1.0f;
float* x4252 = (float*)myMalloc(1 * sizeof(float));;
x4252[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2542, x2542));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4250, x_desc, x2555, x_desc, x2563, x_desc, x2555,
    x4252, x_desc, x2563));
};
if (x4256) {
if (x4258) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2542) x Sym(2542), res:  x Const(64) x Const(1024) x Sym(2399) x Sym(2399)");
}
float* x4263 = (float*)myMalloc(1 * sizeof(float));;
x4263[0] = 1.0f;
float* x4265 = (float*)myMalloc(1 * sizeof(float));;
x4265[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2542, x2542));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2399, x2399));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x4263, bias_desc, x2563, x4265, out_desc, x2420));
};
} else {
float* x4269 = (float*)myMalloc(1 * sizeof(float));;
x4269[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2399, x2399));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2542, x2542));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x4269, grad_out_desc, x2563,
    x4269, grad_bias_desc, x2420));
};
}
float* x4274 = (float*)myMalloc(1 * sizeof(float));;
x4274[0] = 0.0f;
float* x4276 = (float*)myMalloc(1 * sizeof(float));;
x4276[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2542, x2542));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2542, x2542));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4276, x4276, x4276, x4276, in_desc, x2548,
    out_desc, x2563, in_desc, x2554, sbmv_desc, x355,
    x1127,x1339, 1.0E-5, x2556, x2557));
};
// conv2D back-propagate
float* x4280 = (float*)myMalloc(1 * sizeof(float));;
x4280[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2508, x2508));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2542, x2542));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4280, filt_desc, x463, grad_out_desc, x2554,
    conv_desc, algo, ws_data, ws_size,
    x4280, grad_in_desc, x2529));
};
float* x4283 = (float*)myMalloc(1 * sizeof(float));;
x4283[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2542, x2542));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2508, x2508));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4283, in_desc, x2521, grad_out_desc, x2554,
    conv_desc, algo, ws_data, ws_size,
    x4283, grad_filt_desc, x1163));
};
float* x4286 = (float*)myMalloc(1 * sizeof(float));;
x4286[0] = 1.0f;
float* x4288 = (float*)myMalloc(1 * sizeof(float));;
x4288[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2508, x2508));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4286, x_desc, x2521, x_desc, x2529, x_desc, x2521,
    x4288, x_desc, x2529));
};
float* x4291 = (float*)myMalloc(1 * sizeof(float));;
x4291[0] = 0.0f;
float* x4293 = (float*)myMalloc(1 * sizeof(float));;
x4293[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2508, x2508));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2508, x2508));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4293, x4293, x4293, x4293, in_desc, x2514,
    out_desc, x2529, in_desc, x2520, sbmv_desc, x1108,
    x1378,x1203, 1.0E-5, x2522, x2523));
};
// conv2D back-propagate
float* x4297 = (float*)myMalloc(1 * sizeof(float));;
x4297[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2472, x2472));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2508, x2508));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4297, filt_desc, x388, grad_out_desc, x2520,
    conv_desc, algo, ws_data, ws_size,
    x4297, grad_in_desc, x2493));
};
float* x4300 = (float*)myMalloc(1 * sizeof(float));;
x4300[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2508, x2508));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2472, x2472));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4300, in_desc, x2485, grad_out_desc, x2520,
    conv_desc, algo, ws_data, ws_size,
    x4300, grad_filt_desc, x1138));
};
float* x4303 = (float*)myMalloc(1 * sizeof(float));;
x4303[0] = 1.0f;
float* x4305 = (float*)myMalloc(1 * sizeof(float));;
x4305[0] = 0.0f;

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
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4303, x_desc, x2485, x_desc, x2493, x_desc, x2485,
    x4305, x_desc, x2493));
};
float* x4308 = (float*)myMalloc(1 * sizeof(float));;
x4308[0] = 0.0f;
float* x4310 = (float*)myMalloc(1 * sizeof(float));;
x4310[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4310, x4310, x4310, x4310, in_desc, x2478,
    out_desc, x2493, in_desc, x2484, sbmv_desc, x385,
    x1137,x1326, 1.0E-5, x2486, x2487));
};
// conv2D back-propagate
float* x4314 = (float*)myMalloc(1 * sizeof(float));;
x4314[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2399, x2399));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2472, x2472));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4314, filt_desc, x334, grad_out_desc, x2484,
    conv_desc, algo, ws_data, ws_size,
    x4314, grad_in_desc, x2420));
};
float* x4317 = (float*)myMalloc(1 * sizeof(float));;
x4317[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2472, x2472));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2399, x2399));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4317, in_desc, x2412, grad_out_desc, x2484,
    conv_desc, algo, ws_data, ws_size,
    x4317, grad_filt_desc, x1120));
};
float* x4320 = (float*)myMalloc(1 * sizeof(float));;
x4320[0] = 1.0f;
float* x4322 = (float*)myMalloc(1 * sizeof(float));;
x4322[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2399, x2399));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4320, x_desc, x2412, x_desc, x2420, x_desc, x2412,
    x4322, x_desc, x2420));
};
if (x4326) {
if (x4328) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2399) x Sym(2399), res:  x Const(64) x Const(1024) x Sym(2425) x Sym(2425)");
}
float* x4333 = (float*)myMalloc(1 * sizeof(float));;
x4333[0] = 1.0f;
float* x4335 = (float*)myMalloc(1 * sizeof(float));;
x4335[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2399, x2399));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2425, x2425));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x4333, bias_desc, x2420, x4335, out_desc, x2446));
};
} else {
float* x4339 = (float*)myMalloc(1 * sizeof(float));;
x4339[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2425, x2425));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2399, x2399));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x4339, grad_out_desc, x2420,
    x4339, grad_bias_desc, x2446));
};
}
float* x4344 = (float*)myMalloc(1 * sizeof(float));;
x4344[0] = 0.0f;
float* x4346 = (float*)myMalloc(1 * sizeof(float));;
x4346[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2425, x2425));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2425, x2425));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4346, x4346, x4346, x4346, in_desc, x2431,
    out_desc, x2446, in_desc, x2437, sbmv_desc, x382,
    x1136,x1327, 1.0E-5, x2439, x2440));
};
// conv2D back-propagate
float* x4350 = (float*)myMalloc(1 * sizeof(float));;
x4350[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 512, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2282, x2282));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2425, x2425));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 2, 2, 1, 1,
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
    x4350, filt_desc, x520, grad_out_desc, x2437,
    conv_desc, algo, ws_data, ws_size,
    x4350, grad_in_desc, x2303));
};
float* x4353 = (float*)myMalloc(1 * sizeof(float));;
x4353[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 512, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2425, x2425));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2282, x2282));

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
    x4353, in_desc, x2295, grad_out_desc, x2437,
    conv_desc, algo, ws_data, ws_size,
    x4353, grad_filt_desc, x1182));
};
float* x4356 = (float*)myMalloc(1 * sizeof(float));;
x4356[0] = 0.0f;
float* x4358 = (float*)myMalloc(1 * sizeof(float));;
x4358[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2399, x2399));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2399, x2399));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4358, x4358, x4358, x4358, in_desc, x2405,
    out_desc, x2420, in_desc, x2411, sbmv_desc, x349,
    x1125,x1224, 1.0E-5, x2413, x2414));
};
// conv2D back-propagate
float* x4362 = (float*)myMalloc(1 * sizeof(float));;
x4362[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2365, x2365));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2399, x2399));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4362, filt_desc, x1102, grad_out_desc, x2411,
    conv_desc, algo, ws_data, ws_size,
    x4362, grad_in_desc, x2386));
};
float* x4365 = (float*)myMalloc(1 * sizeof(float));;
x4365[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, x2399, x2399));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2365, x2365));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4365, in_desc, x2378, grad_out_desc, x2411,
    conv_desc, algo, ws_data, ws_size,
    x4365, grad_filt_desc, x1376));
};
float* x4368 = (float*)myMalloc(1 * sizeof(float));;
x4368[0] = 1.0f;
float* x4370 = (float*)myMalloc(1 * sizeof(float));;
x4370[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2365, x2365));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4368, x_desc, x2378, x_desc, x2386, x_desc, x2378,
    x4370, x_desc, x2386));
};
float* x4373 = (float*)myMalloc(1 * sizeof(float));;
x4373[0] = 0.0f;
float* x4375 = (float*)myMalloc(1 * sizeof(float));;
x4375[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2365, x2365));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2365, x2365));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4375, x4375, x4375, x4375, in_desc, x2371,
    out_desc, x2386, in_desc, x2377, sbmv_desc, x619,
    x1215,x1123, 1.0E-5, x2379, x2380));
};
// conv2D back-propagate
float* x4379 = (float*)myMalloc(1 * sizeof(float));;
x4379[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2329, x2329));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2365, x2365));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 2, 2, 1, 1,
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
    x4379, filt_desc, x820, grad_out_desc, x2377,
    conv_desc, algo, ws_data, ws_size,
    x4379, grad_in_desc, x2350));
};
float* x4382 = (float*)myMalloc(1 * sizeof(float));;
x4382[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2365, x2365));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2329, x2329));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 2, 2, 1, 1,
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
    x4382, in_desc, x2342, grad_out_desc, x2377,
    conv_desc, algo, ws_data, ws_size,
    x4382, grad_filt_desc, x1282));
};
float* x4385 = (float*)myMalloc(1 * sizeof(float));;
x4385[0] = 1.0f;
float* x4387 = (float*)myMalloc(1 * sizeof(float));;
x4387[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2329, x2329));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4385, x_desc, x2342, x_desc, x2350, x_desc, x2342,
    x4387, x_desc, x2350));
};
float* x4390 = (float*)myMalloc(1 * sizeof(float));;
x4390[0] = 0.0f;
float* x4392 = (float*)myMalloc(1 * sizeof(float));;
x4392[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2329, x2329));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2329, x2329));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4392, x4392, x4392, x4392, in_desc, x2335,
    out_desc, x2350, in_desc, x2341, sbmv_desc, x1105,
    x1377,x1128, 1.0E-5, x2343, x2344));
};
// conv2D back-propagate
float* x4396 = (float*)myMalloc(1 * sizeof(float));;
x4396[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 512, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2282, x2282));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2329, x2329));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4396, filt_desc, x835, grad_out_desc, x2341,
    conv_desc, algo, ws_data, ws_size,
    x4396, grad_in_desc, x2303));
};
float* x4399 = (float*)myMalloc(1 * sizeof(float));;
x4399[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 512, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x2329, x2329));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2282, x2282));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4399, in_desc, x2295, grad_out_desc, x2341,
    conv_desc, algo, ws_data, ws_size,
    x4399, grad_filt_desc, x1287));
};
float* x4402 = (float*)myMalloc(1 * sizeof(float));;
x4402[0] = 1.0f;
float* x4404 = (float*)myMalloc(1 * sizeof(float));;
x4404[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2282, x2282));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4402, x_desc, x2295, x_desc, x2303, x_desc, x2295,
    x4404, x_desc, x2303));
};
if (x4408) {
if (x4411) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(2282) x Sym(2282), res:  x Const(64) x Const(512) x Sym(2165) x Sym(2165)");
}
float* x4416 = (float*)myMalloc(1 * sizeof(float));;
x4416[0] = 1.0f;
float* x4418 = (float*)myMalloc(1 * sizeof(float));;
x4418[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2282, x2282));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2165, x2165));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x4416, bias_desc, x2303, x4418, out_desc, x2186));
};
} else {
float* x4422 = (float*)myMalloc(1 * sizeof(float));;
x4422[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2165, x2165));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2282, x2282));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x4422, grad_out_desc, x2303,
    x4422, grad_bias_desc, x2186));
};
}
float* x4427 = (float*)myMalloc(1 * sizeof(float));;
x4427[0] = 0.0f;
float* x4429 = (float*)myMalloc(1 * sizeof(float));;
x4429[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2282, x2282));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2282, x2282));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4429, x4429, x4429, x4429, in_desc, x2288,
    out_desc, x2303, in_desc, x2294, sbmv_desc, x763,
    x1263,x1161, 1.0E-5, x2296, x2297));
};
// conv2D back-propagate
float* x4433 = (float*)myMalloc(1 * sizeof(float));;
x4433[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2248, x2248));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2282, x2282));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4433, filt_desc, x460, grad_out_desc, x2294,
    conv_desc, algo, ws_data, ws_size,
    x4433, grad_in_desc, x2269));
};
float* x4436 = (float*)myMalloc(1 * sizeof(float));;
x4436[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2282, x2282));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2248, x2248));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4436, in_desc, x2261, grad_out_desc, x2294,
    conv_desc, algo, ws_data, ws_size,
    x4436, grad_filt_desc, x1162));
};
float* x4439 = (float*)myMalloc(1 * sizeof(float));;
x4439[0] = 1.0f;
float* x4441 = (float*)myMalloc(1 * sizeof(float));;
x4441[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2248, x2248));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4439, x_desc, x2261, x_desc, x2269, x_desc, x2261,
    x4441, x_desc, x2269));
};
float* x4444 = (float*)myMalloc(1 * sizeof(float));;
x4444[0] = 0.0f;
float* x4446 = (float*)myMalloc(1 * sizeof(float));;
x4446[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2248, x2248));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2248, x2248));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4446, x4446, x4446, x4446, in_desc, x2254,
    out_desc, x2269, in_desc, x2260, sbmv_desc, x532,
    x1186,x1145, 1.0E-5, x2262, x2263));
};
// conv2D back-propagate
float* x4450 = (float*)myMalloc(1 * sizeof(float));;
x4450[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2212, x2212));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2248, x2248));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4450, filt_desc, x790, grad_out_desc, x2260,
    conv_desc, algo, ws_data, ws_size,
    x4450, grad_in_desc, x2233));
};
float* x4453 = (float*)myMalloc(1 * sizeof(float));;
x4453[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2248, x2248));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2212, x2212));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4453, in_desc, x2225, grad_out_desc, x2260,
    conv_desc, algo, ws_data, ws_size,
    x4453, grad_filt_desc, x1272));
};
float* x4456 = (float*)myMalloc(1 * sizeof(float));;
x4456[0] = 1.0f;
float* x4458 = (float*)myMalloc(1 * sizeof(float));;
x4458[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2212, x2212));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4456, x_desc, x2225, x_desc, x2233, x_desc, x2225,
    x4458, x_desc, x2233));
};
float* x4461 = (float*)myMalloc(1 * sizeof(float));;
x4461[0] = 0.0f;
float* x4463 = (float*)myMalloc(1 * sizeof(float));;
x4463[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2212, x2212));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2212, x2212));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4463, x4463, x4463, x4463, in_desc, x2218,
    out_desc, x2233, in_desc, x2224, sbmv_desc, x412,
    x1146,x1349, 1.0E-5, x2226, x2227));
};
// conv2D back-propagate
float* x4467 = (float*)myMalloc(1 * sizeof(float));;
x4467[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2165, x2165));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2212, x2212));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4467, filt_desc, x691, grad_out_desc, x2224,
    conv_desc, algo, ws_data, ws_size,
    x4467, grad_in_desc, x2186));
};
float* x4470 = (float*)myMalloc(1 * sizeof(float));;
x4470[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2212, x2212));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2165, x2165));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4470, in_desc, x2178, grad_out_desc, x2224,
    conv_desc, algo, ws_data, ws_size,
    x4470, grad_filt_desc, x1239));
};
float* x4473 = (float*)myMalloc(1 * sizeof(float));;
x4473[0] = 1.0f;
float* x4475 = (float*)myMalloc(1 * sizeof(float));;
x4475[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2165, x2165));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4473, x_desc, x2178, x_desc, x2186, x_desc, x2178,
    x4475, x_desc, x2186));
};
if (x4479) {
if (x4481) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(2165) x Sym(2165), res:  x Const(64) x Const(512) x Sym(2048) x Sym(2048)");
}
float* x4486 = (float*)myMalloc(1 * sizeof(float));;
x4486[0] = 1.0f;
float* x4488 = (float*)myMalloc(1 * sizeof(float));;
x4488[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2165, x2165));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2048, x2048));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x4486, bias_desc, x2186, x4488, out_desc, x2069));
};
} else {
float* x4492 = (float*)myMalloc(1 * sizeof(float));;
x4492[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2048, x2048));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2165, x2165));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x4492, grad_out_desc, x2186,
    x4492, grad_bias_desc, x2069));
};
}
float* x4497 = (float*)myMalloc(1 * sizeof(float));;
x4497[0] = 0.0f;
float* x4499 = (float*)myMalloc(1 * sizeof(float));;
x4499[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2165, x2165));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2165, x2165));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4499, x4499, x4499, x4499, in_desc, x2171,
    out_desc, x2186, in_desc, x2177, sbmv_desc, x796,
    x1274,x1189, 1.0E-5, x2179, x2180));
};
// conv2D back-propagate
float* x4503 = (float*)myMalloc(1 * sizeof(float));;
x4503[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2131, x2131));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2165, x2165));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4503, filt_desc, x418, grad_out_desc, x2177,
    conv_desc, algo, ws_data, ws_size,
    x4503, grad_in_desc, x2152));
};
float* x4506 = (float*)myMalloc(1 * sizeof(float));;
x4506[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2165, x2165));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2131, x2131));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4506, in_desc, x2144, grad_out_desc, x2177,
    conv_desc, algo, ws_data, ws_size,
    x4506, grad_filt_desc, x1148));
};
float* x4509 = (float*)myMalloc(1 * sizeof(float));;
x4509[0] = 1.0f;
float* x4511 = (float*)myMalloc(1 * sizeof(float));;
x4511[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2131, x2131));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4509, x_desc, x2144, x_desc, x2152, x_desc, x2144,
    x4511, x_desc, x2152));
};
float* x4514 = (float*)myMalloc(1 * sizeof(float));;
x4514[0] = 0.0f;
float* x4516 = (float*)myMalloc(1 * sizeof(float));;
x4516[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2131, x2131));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2131, x2131));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4516, x4516, x4516, x4516, in_desc, x2137,
    out_desc, x2152, in_desc, x2143, sbmv_desc, x676,
    x1234,x1168, 1.0E-5, x2145, x2146));
};
// conv2D back-propagate
float* x4520 = (float*)myMalloc(1 * sizeof(float));;
x4520[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2095, x2095));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2131, x2131));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4520, filt_desc, x868, grad_out_desc, x2143,
    conv_desc, algo, ws_data, ws_size,
    x4520, grad_in_desc, x2116));
};
float* x4523 = (float*)myMalloc(1 * sizeof(float));;
x4523[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2131, x2131));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2095, x2095));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4523, in_desc, x2108, grad_out_desc, x2143,
    conv_desc, algo, ws_data, ws_size,
    x4523, grad_filt_desc, x1298));
};
float* x4526 = (float*)myMalloc(1 * sizeof(float));;
x4526[0] = 1.0f;
float* x4528 = (float*)myMalloc(1 * sizeof(float));;
x4528[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2095, x2095));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4526, x_desc, x2108, x_desc, x2116, x_desc, x2108,
    x4528, x_desc, x2116));
};
float* x4531 = (float*)myMalloc(1 * sizeof(float));;
x4531[0] = 0.0f;
float* x4533 = (float*)myMalloc(1 * sizeof(float));;
x4533[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2095, x2095));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2095, x2095));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4533, x4533, x4533, x4533, in_desc, x2101,
    out_desc, x2116, in_desc, x2107, sbmv_desc, x430,
    x1152,x1277, 1.0E-5, x2109, x2110));
};
// conv2D back-propagate
float* x4537 = (float*)myMalloc(1 * sizeof(float));;
x4537[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2048, x2048));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2095, x2095));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4537, filt_desc, x883, grad_out_desc, x2107,
    conv_desc, algo, ws_data, ws_size,
    x4537, grad_in_desc, x2069));
};
float* x4540 = (float*)myMalloc(1 * sizeof(float));;
x4540[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2095, x2095));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2048, x2048));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4540, in_desc, x2061, grad_out_desc, x2107,
    conv_desc, algo, ws_data, ws_size,
    x4540, grad_filt_desc, x1303));
};
float* x4543 = (float*)myMalloc(1 * sizeof(float));;
x4543[0] = 1.0f;
float* x4545 = (float*)myMalloc(1 * sizeof(float));;
x4545[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2048, x2048));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4543, x_desc, x2061, x_desc, x2069, x_desc, x2061,
    x4545, x_desc, x2069));
};
if (x4549) {
if (x4551) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(2048) x Sym(2048), res:  x Const(64) x Const(512) x Sym(1905) x Sym(1905)");
}
float* x4556 = (float*)myMalloc(1 * sizeof(float));;
x4556[0] = 1.0f;
float* x4558 = (float*)myMalloc(1 * sizeof(float));;
x4558[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2048, x2048));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1905, x1905));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x4556, bias_desc, x2069, x4558, out_desc, x1926));
};
} else {
float* x4562 = (float*)myMalloc(1 * sizeof(float));;
x4562[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1905, x1905));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2048, x2048));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x4562, grad_out_desc, x2069,
    x4562, grad_bias_desc, x1926));
};
}
float* x4567 = (float*)myMalloc(1 * sizeof(float));;
x4567[0] = 0.0f;
float* x4569 = (float*)myMalloc(1 * sizeof(float));;
x4569[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2048, x2048));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2048, x2048));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4569, x4569, x4569, x4569, in_desc, x2054,
    out_desc, x2069, in_desc, x2060, sbmv_desc, x451,
    x1159,x1353, 1.0E-5, x2062, x2063));
};
// conv2D back-propagate
float* x4573 = (float*)myMalloc(1 * sizeof(float));;
x4573[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2014, x2014));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2048, x2048));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4573, filt_desc, x628, grad_out_desc, x2060,
    conv_desc, algo, ws_data, ws_size,
    x4573, grad_in_desc, x2035));
};
float* x4576 = (float*)myMalloc(1 * sizeof(float));;
x4576[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x2048, x2048));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2014, x2014));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4576, in_desc, x2027, grad_out_desc, x2060,
    conv_desc, algo, ws_data, ws_size,
    x4576, grad_filt_desc, x1218));
};
float* x4579 = (float*)myMalloc(1 * sizeof(float));;
x4579[0] = 1.0f;
float* x4581 = (float*)myMalloc(1 * sizeof(float));;
x4581[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2014, x2014));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4579, x_desc, x2027, x_desc, x2035, x_desc, x2027,
    x4581, x_desc, x2035));
};
float* x4584 = (float*)myMalloc(1 * sizeof(float));;
x4584[0] = 0.0f;
float* x4586 = (float*)myMalloc(1 * sizeof(float));;
x4586[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2014, x2014));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2014, x2014));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4586, x4586, x4586, x4586, in_desc, x2020,
    out_desc, x2035, in_desc, x2026, sbmv_desc, x319,
    x1115,x1202, 1.0E-5, x2028, x2029));
};
// conv2D back-propagate
float* x4590 = (float*)myMalloc(1 * sizeof(float));;
x4590[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1978, x1978));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2014, x2014));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4590, filt_desc, x1000, grad_out_desc, x2026,
    conv_desc, algo, ws_data, ws_size,
    x4590, grad_in_desc, x1999));
};
float* x4593 = (float*)myMalloc(1 * sizeof(float));;
x4593[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x2014, x2014));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1978, x1978));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4593, in_desc, x1991, grad_out_desc, x2026,
    conv_desc, algo, ws_data, ws_size,
    x4593, grad_filt_desc, x1342));
};
float* x4596 = (float*)myMalloc(1 * sizeof(float));;
x4596[0] = 1.0f;
float* x4598 = (float*)myMalloc(1 * sizeof(float));;
x4598[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1978, x1978));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4596, x_desc, x1991, x_desc, x1999, x_desc, x1991,
    x4598, x_desc, x1999));
};
float* x4601 = (float*)myMalloc(1 * sizeof(float));;
x4601[0] = 0.0f;
float* x4603 = (float*)myMalloc(1 * sizeof(float));;
x4603[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1978, x1978));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1978, x1978));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4603, x4603, x4603, x4603, in_desc, x1984,
    out_desc, x1999, in_desc, x1990, sbmv_desc, x961,
    x1329,x1124, 1.0E-5, x1992, x1993));
};
// conv2D back-propagate
float* x4607 = (float*)myMalloc(1 * sizeof(float));;
x4607[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1905, x1905));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1978, x1978));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4607, filt_desc, x1063, grad_out_desc, x1990,
    conv_desc, algo, ws_data, ws_size,
    x4607, grad_in_desc, x1926));
};
float* x4610 = (float*)myMalloc(1 * sizeof(float));;
x4610[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1978, x1978));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1905, x1905));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4610, in_desc, x1918, grad_out_desc, x1990,
    conv_desc, algo, ws_data, ws_size,
    x4610, grad_filt_desc, x1363));
};
float* x4613 = (float*)myMalloc(1 * sizeof(float));;
x4613[0] = 1.0f;
float* x4615 = (float*)myMalloc(1 * sizeof(float));;
x4615[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1905, x1905));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4613, x_desc, x1918, x_desc, x1926, x_desc, x1918,
    x4615, x_desc, x1926));
};
if (x4619) {
if (x4621) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(1905) x Sym(1905), res:  x Const(64) x Const(512) x Sym(1931) x Sym(1931)");
}
float* x4626 = (float*)myMalloc(1 * sizeof(float));;
x4626[0] = 1.0f;
float* x4628 = (float*)myMalloc(1 * sizeof(float));;
x4628[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1905, x1905));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1931, x1931));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x4626, bias_desc, x1926, x4628, out_desc, x1952));
};
} else {
float* x4632 = (float*)myMalloc(1 * sizeof(float));;
x4632[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1931, x1931));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1905, x1905));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x4632, grad_out_desc, x1926,
    x4632, grad_bias_desc, x1952));
};
}
float* x4637 = (float*)myMalloc(1 * sizeof(float));;
x4637[0] = 0.0f;
float* x4639 = (float*)myMalloc(1 * sizeof(float));;
x4639[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1931, x1931));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1931, x1931));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4639, x4639, x4639, x4639, in_desc, x1937,
    out_desc, x1952, in_desc, x1943, sbmv_desc, x916,
    x1314,x1226, 1.0E-5, x1945, x1946));
};
// conv2D back-propagate
float* x4643 = (float*)myMalloc(1 * sizeof(float));;
x4643[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 256, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1788, x1788));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1931, x1931));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 2, 2, 1, 1,
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
    x4643, filt_desc, x1069, grad_out_desc, x1943,
    conv_desc, algo, ws_data, ws_size,
    x4643, grad_in_desc, x1809));
};
float* x4646 = (float*)myMalloc(1 * sizeof(float));;
x4646[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 256, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1931, x1931));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1788, x1788));

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
    x4646, in_desc, x1801, grad_out_desc, x1943,
    conv_desc, algo, ws_data, ws_size,
    x4646, grad_filt_desc, x1365));
};
float* x4649 = (float*)myMalloc(1 * sizeof(float));;
x4649[0] = 0.0f;
float* x4651 = (float*)myMalloc(1 * sizeof(float));;
x4651[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1905, x1905));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1905, x1905));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4651, x4651, x4651, x4651, in_desc, x1911,
    out_desc, x1926, in_desc, x1917, sbmv_desc, x730,
    x1252,x1317, 1.0E-5, x1919, x1920));
};
// conv2D back-propagate
float* x4655 = (float*)myMalloc(1 * sizeof(float));;
x4655[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1871, x1871));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1905, x1905));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4655, filt_desc, x613, grad_out_desc, x1917,
    conv_desc, algo, ws_data, ws_size,
    x4655, grad_in_desc, x1892));
};
float* x4658 = (float*)myMalloc(1 * sizeof(float));;
x4658[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, x1905, x1905));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1871, x1871));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4658, in_desc, x1884, grad_out_desc, x1917,
    conv_desc, algo, ws_data, ws_size,
    x4658, grad_filt_desc, x1213));
};
float* x4661 = (float*)myMalloc(1 * sizeof(float));;
x4661[0] = 1.0f;
float* x4663 = (float*)myMalloc(1 * sizeof(float));;
x4663[0] = 0.0f;

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
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4661, x_desc, x1884, x_desc, x1892, x_desc, x1884,
    x4663, x_desc, x1892));
};
float* x4666 = (float*)myMalloc(1 * sizeof(float));;
x4666[0] = 0.0f;
float* x4668 = (float*)myMalloc(1 * sizeof(float));;
x4668[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4668, x4668, x4668, x4668, in_desc, x1877,
    out_desc, x1892, in_desc, x1883, sbmv_desc, x1051,
    x1359,x1297, 1.0E-5, x1885, x1886));
};
// conv2D back-propagate
float* x4672 = (float*)myMalloc(1 * sizeof(float));;
x4672[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1835, x1835));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1871, x1871));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 2, 2, 1, 1,
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
    x4672, filt_desc, x376, grad_out_desc, x1883,
    conv_desc, algo, ws_data, ws_size,
    x4672, grad_in_desc, x1856));
};
float* x4675 = (float*)myMalloc(1 * sizeof(float));;
x4675[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1871, x1871));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1835, x1835));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 2, 2, 1, 1,
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
    x4675, in_desc, x1848, grad_out_desc, x1883,
    conv_desc, algo, ws_data, ws_size,
    x4675, grad_filt_desc, x1134));
};
float* x4678 = (float*)myMalloc(1 * sizeof(float));;
x4678[0] = 1.0f;
float* x4680 = (float*)myMalloc(1 * sizeof(float));;
x4680[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1835, x1835));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4678, x_desc, x1848, x_desc, x1856, x_desc, x1848,
    x4680, x_desc, x1856));
};
float* x4683 = (float*)myMalloc(1 * sizeof(float));;
x4683[0] = 0.0f;
float* x4685 = (float*)myMalloc(1 * sizeof(float));;
x4685[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1835, x1835));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1835, x1835));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4685, x4685, x4685, x4685, in_desc, x1841,
    out_desc, x1856, in_desc, x1847, sbmv_desc, x547,
    x1191,x1279, 1.0E-5, x1849, x1850));
};
// conv2D back-propagate
float* x4689 = (float*)myMalloc(1 * sizeof(float));;
x4689[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 256, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1788, x1788));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1835, x1835));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4689, filt_desc, x328, grad_out_desc, x1847,
    conv_desc, algo, ws_data, ws_size,
    x4689, grad_in_desc, x1809));
};
float* x4692 = (float*)myMalloc(1 * sizeof(float));;
x4692[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 256, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, x1835, x1835));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1788, x1788));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4692, in_desc, x1801, grad_out_desc, x1847,
    conv_desc, algo, ws_data, ws_size,
    x4692, grad_filt_desc, x1118));
};
float* x4695 = (float*)myMalloc(1 * sizeof(float));;
x4695[0] = 1.0f;
float* x4697 = (float*)myMalloc(1 * sizeof(float));;
x4697[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1788, x1788));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4695, x_desc, x1801, x_desc, x1809, x_desc, x1801,
    x4697, x_desc, x1809));
};
if (x4701) {
if (x4704) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1788) x Sym(1788), res:  x Const(64) x Const(256) x Sym(1671) x Sym(1671)");
}
float* x4709 = (float*)myMalloc(1 * sizeof(float));;
x4709[0] = 1.0f;
float* x4711 = (float*)myMalloc(1 * sizeof(float));;
x4711[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1788, x1788));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1671, x1671));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x4709, bias_desc, x1809, x4711, out_desc, x1692));
};
} else {
float* x4715 = (float*)myMalloc(1 * sizeof(float));;
x4715[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1671, x1671));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1788, x1788));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x4715, grad_out_desc, x1809,
    x4715, grad_bias_desc, x1692));
};
}
float* x4720 = (float*)myMalloc(1 * sizeof(float));;
x4720[0] = 0.0f;
float* x4722 = (float*)myMalloc(1 * sizeof(float));;
x4722[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1788, x1788));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1788, x1788));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4722, x4722, x4722, x4722, in_desc, x1794,
    out_desc, x1809, in_desc, x1800, sbmv_desc, x406,
    x1144,x1354, 1.0E-5, x1802, x1803));
};
// conv2D back-propagate
float* x4726 = (float*)myMalloc(1 * sizeof(float));;
x4726[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1754, x1754));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1788, x1788));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4726, filt_desc, x556, grad_out_desc, x1800,
    conv_desc, algo, ws_data, ws_size,
    x4726, grad_in_desc, x1775));
};
float* x4729 = (float*)myMalloc(1 * sizeof(float));;
x4729[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1788, x1788));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1754, x1754));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4729, in_desc, x1767, grad_out_desc, x1800,
    conv_desc, algo, ws_data, ws_size,
    x4729, grad_filt_desc, x1194));
};
float* x4732 = (float*)myMalloc(1 * sizeof(float));;
x4732[0] = 1.0f;
float* x4734 = (float*)myMalloc(1 * sizeof(float));;
x4734[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1754, x1754));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4732, x_desc, x1767, x_desc, x1775, x_desc, x1767,
    x4734, x_desc, x1775));
};
float* x4737 = (float*)myMalloc(1 * sizeof(float));;
x4737[0] = 0.0f;
float* x4739 = (float*)myMalloc(1 * sizeof(float));;
x4739[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1754, x1754));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1754, x1754));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4739, x4739, x4739, x4739, in_desc, x1760,
    out_desc, x1775, in_desc, x1766, sbmv_desc, x511,
    x1179,x1242, 1.0E-5, x1768, x1769));
};
// conv2D back-propagate
float* x4743 = (float*)myMalloc(1 * sizeof(float));;
x4743[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1718, x1718));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1754, x1754));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4743, filt_desc, x514, grad_out_desc, x1766,
    conv_desc, algo, ws_data, ws_size,
    x4743, grad_in_desc, x1739));
};
float* x4746 = (float*)myMalloc(1 * sizeof(float));;
x4746[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1754, x1754));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1718, x1718));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4746, in_desc, x1731, grad_out_desc, x1766,
    conv_desc, algo, ws_data, ws_size,
    x4746, grad_filt_desc, x1180));
};
float* x4749 = (float*)myMalloc(1 * sizeof(float));;
x4749[0] = 1.0f;
float* x4751 = (float*)myMalloc(1 * sizeof(float));;
x4751[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1718, x1718));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4749, x_desc, x1731, x_desc, x1739, x_desc, x1731,
    x4751, x_desc, x1739));
};
float* x4754 = (float*)myMalloc(1 * sizeof(float));;
x4754[0] = 0.0f;
float* x4756 = (float*)myMalloc(1 * sizeof(float));;
x4756[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1718, x1718));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1718, x1718));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4756, x4756, x4756, x4756, in_desc, x1724,
    out_desc, x1739, in_desc, x1730, sbmv_desc, x538,
    x1188,x1131, 1.0E-5, x1732, x1733));
};
// conv2D back-propagate
float* x4760 = (float*)myMalloc(1 * sizeof(float));;
x4760[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 256, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1671, x1671));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1718, x1718));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4760, filt_desc, x745, grad_out_desc, x1730,
    conv_desc, algo, ws_data, ws_size,
    x4760, grad_in_desc, x1692));
};
float* x4763 = (float*)myMalloc(1 * sizeof(float));;
x4763[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 256, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1718, x1718));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1671, x1671));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4763, in_desc, x1684, grad_out_desc, x1730,
    conv_desc, algo, ws_data, ws_size,
    x4763, grad_filt_desc, x1257));
};
float* x4766 = (float*)myMalloc(1 * sizeof(float));;
x4766[0] = 1.0f;
float* x4768 = (float*)myMalloc(1 * sizeof(float));;
x4768[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1671, x1671));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4766, x_desc, x1684, x_desc, x1692, x_desc, x1684,
    x4768, x_desc, x1692));
};
if (x4772) {
if (x4774) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1671) x Sym(1671), res:  x Const(64) x Const(256) x Sym(1531) x Sym(1531)");
}
float* x4779 = (float*)myMalloc(1 * sizeof(float));;
x4779[0] = 1.0f;
float* x4781 = (float*)myMalloc(1 * sizeof(float));;
x4781[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1671, x1671));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1531, x1531));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x4779, bias_desc, x1692, x4781, out_desc, x1552));
};
} else {
float* x4785 = (float*)myMalloc(1 * sizeof(float));;
x4785[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1531, x1531));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1671, x1671));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x4785, grad_out_desc, x1692,
    x4785, grad_bias_desc, x1552));
};
}
float* x4790 = (float*)myMalloc(1 * sizeof(float));;
x4790[0] = 0.0f;
float* x4792 = (float*)myMalloc(1 * sizeof(float));;
x4792[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1671, x1671));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1671, x1671));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4792, x4792, x4792, x4792, in_desc, x1677,
    out_desc, x1692, in_desc, x1683, sbmv_desc, x469,
    x1165,x1114, 1.0E-5, x1685, x1686));
};
// conv2D back-propagate
float* x4796 = (float*)myMalloc(1 * sizeof(float));;
x4796[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1637, x1637));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1671, x1671));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4796, filt_desc, x685, grad_out_desc, x1683,
    conv_desc, algo, ws_data, ws_size,
    x4796, grad_in_desc, x1658));
};
float* x4799 = (float*)myMalloc(1 * sizeof(float));;
x4799[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1671, x1671));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1637, x1637));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4799, in_desc, x1650, grad_out_desc, x1683,
    conv_desc, algo, ws_data, ws_size,
    x4799, grad_filt_desc, x1237));
};
float* x4802 = (float*)myMalloc(1 * sizeof(float));;
x4802[0] = 1.0f;
float* x4804 = (float*)myMalloc(1 * sizeof(float));;
x4804[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1637, x1637));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4802, x_desc, x1650, x_desc, x1658, x_desc, x1650,
    x4804, x_desc, x1658));
};
float* x4807 = (float*)myMalloc(1 * sizeof(float));;
x4807[0] = 0.0f;
float* x4809 = (float*)myMalloc(1 * sizeof(float));;
x4809[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1637, x1637));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1637, x1637));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4809, x4809, x4809, x4809, in_desc, x1643,
    out_desc, x1658, in_desc, x1649, sbmv_desc, x919,
    x1315,x1260, 1.0E-5, x1651, x1652));
};
// conv2D back-propagate
float* x4813 = (float*)myMalloc(1 * sizeof(float));;
x4813[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1601, x1601));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1637, x1637));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4813, filt_desc, x544, grad_out_desc, x1649,
    conv_desc, algo, ws_data, ws_size,
    x4813, grad_in_desc, x1622));
};
float* x4816 = (float*)myMalloc(1 * sizeof(float));;
x4816[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1637, x1637));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1601, x1601));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4816, in_desc, x1614, grad_out_desc, x1649,
    conv_desc, algo, ws_data, ws_size,
    x4816, grad_filt_desc, x1190));
};
float* x4819 = (float*)myMalloc(1 * sizeof(float));;
x4819[0] = 1.0f;
float* x4821 = (float*)myMalloc(1 * sizeof(float));;
x4821[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1601, x1601));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4819, x_desc, x1614, x_desc, x1622, x_desc, x1614,
    x4821, x_desc, x1622));
};
float* x4824 = (float*)myMalloc(1 * sizeof(float));;
x4824[0] = 0.0f;
float* x4826 = (float*)myMalloc(1 * sizeof(float));;
x4826[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1601, x1601));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1601, x1601));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4826, x4826, x4826, x4826, in_desc, x1607,
    out_desc, x1622, in_desc, x1613, sbmv_desc, x721,
    x1249,x1167, 1.0E-5, x1615, x1616));
};
// conv2D back-propagate
float* x4830 = (float*)myMalloc(1 * sizeof(float));;
x4830[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 256, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1531, x1531));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1601, x1601));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4830, filt_desc, x808, grad_out_desc, x1613,
    conv_desc, algo, ws_data, ws_size,
    x4830, grad_in_desc, x1552));
};
float* x4833 = (float*)myMalloc(1 * sizeof(float));;
x4833[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 256, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1601, x1601));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1531, x1531));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4833, in_desc, x1544, grad_out_desc, x1613,
    conv_desc, algo, ws_data, ws_size,
    x4833, grad_filt_desc, x1278));
};
float* x4836 = (float*)myMalloc(1 * sizeof(float));;
x4836[0] = 1.0f;
float* x4838 = (float*)myMalloc(1 * sizeof(float));;
x4838[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1531, x1531));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4836, x_desc, x1544, x_desc, x1552, x_desc, x1544,
    x4838, x_desc, x1552));
};
if (x4842) {
if (x4844) {
} else {
assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1531) x Sym(1531), res:  x Const(64) x Const(256) x Sym(1461) x Sym(1461)");
}
float* x4849 = (float*)myMalloc(1 * sizeof(float));;
x4849[0] = 1.0f;
float* x4851 = (float*)myMalloc(1 * sizeof(float));;
x4851[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1531, x1531));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1461, x1461));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x4849, bias_desc, x1552, x4851, out_desc, x1575));
};
} else {
float* x4855 = (float*)myMalloc(1 * sizeof(float));;
x4855[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1461, x1461));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1531, x1531));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x4855, grad_out_desc, x1552,
    x4855, grad_bias_desc, x1575));
};
}
float* x4860 = (float*)myMalloc(1 * sizeof(float));;
x4860[0] = 0.0f;
float* x4862 = (float*)myMalloc(1 * sizeof(float));;
x4862[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1461, x1461));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1461, x1461));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4862, x4862, x4862, x4862, in_desc, x1560,
    out_desc, x1575, in_desc, x1566, sbmv_desc, x523,
    x1183,x1310, 1.0E-5, x1568, x1569));
};
// conv2D back-propagate
float* x4866 = (float*)myMalloc(1 * sizeof(float));;
x4866[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1445, x1445));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1461, x1461));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4866, filt_desc, x781, grad_out_desc, x1566,
    conv_desc, algo, ws_data, ws_size,
    x4866, grad_in_desc, x1453));
};
float* x4869 = (float*)myMalloc(1 * sizeof(float));;
x4869[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1461, x1461));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1445, x1445));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4869, in_desc, x1451, grad_out_desc, x1566,
    conv_desc, algo, ws_data, ws_size,
    x4869, grad_filt_desc, x1269));
};
float* x4872 = (float*)myMalloc(1 * sizeof(float));;
x4872[0] = 0.0f;
float* x4874 = (float*)myMalloc(1 * sizeof(float));;
x4874[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1531, x1531));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1531, x1531));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4874, x4874, x4874, x4874, in_desc, x1537,
    out_desc, x1552, in_desc, x1543, sbmv_desc, x892,
    x1306,x1233, 1.0E-5, x1545, x1546));
};
// conv2D back-propagate
float* x4878 = (float*)myMalloc(1 * sizeof(float));;
x4878[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1497, x1497));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1531, x1531));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4878, filt_desc, x391, grad_out_desc, x1543,
    conv_desc, algo, ws_data, ws_size,
    x4878, grad_in_desc, x1518));
};
float* x4881 = (float*)myMalloc(1 * sizeof(float));;
x4881[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, x1531, x1531));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1497, x1497));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4881, in_desc, x1510, grad_out_desc, x1543,
    conv_desc, algo, ws_data, ws_size,
    x4881, grad_filt_desc, x1139));
};
float* x4884 = (float*)myMalloc(1 * sizeof(float));;
x4884[0] = 1.0f;
float* x4886 = (float*)myMalloc(1 * sizeof(float));;
x4886[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1497, x1497));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4884, x_desc, x1510, x_desc, x1518, x_desc, x1510,
    x4886, x_desc, x1518));
};
float* x4889 = (float*)myMalloc(1 * sizeof(float));;
x4889[0] = 0.0f;
float* x4891 = (float*)myMalloc(1 * sizeof(float));;
x4891[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1497, x1497));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1497, x1497));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4891, x4891, x4891, x4891, in_desc, x1503,
    out_desc, x1518, in_desc, x1509, sbmv_desc, x787,
    x1271,x1156, 1.0E-5, x1511, x1512));
};
// conv2D back-propagate
float* x4895 = (float*)myMalloc(1 * sizeof(float));;
x4895[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1461, x1461));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1497, x1497));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4895, filt_desc, x565, grad_out_desc, x1509,
    conv_desc, algo, ws_data, ws_size,
    x4895, grad_in_desc, x1482));
};
float* x4898 = (float*)myMalloc(1 * sizeof(float));;
x4898[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1497, x1497));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1461, x1461));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4898, in_desc, x1474, grad_out_desc, x1509,
    conv_desc, algo, ws_data, ws_size,
    x4898, grad_filt_desc, x1197));
};
float* x4901 = (float*)myMalloc(1 * sizeof(float));;
x4901[0] = 1.0f;
float* x4903 = (float*)myMalloc(1 * sizeof(float));;
x4903[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1461, x1461));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4901, x_desc, x1474, x_desc, x1482, x_desc, x1474,
    x4903, x_desc, x1482));
};
float* x4906 = (float*)myMalloc(1 * sizeof(float));;
x4906[0] = 0.0f;
float* x4908 = (float*)myMalloc(1 * sizeof(float));;
x4908[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1461, x1461));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1461, x1461));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4908, x4908, x4908, x4908, in_desc, x1467,
    out_desc, x1482, in_desc, x1473, sbmv_desc, x373,
    x1133,x1160, 1.0E-5, x1475, x1476));
};
// conv2D back-propagate
float* x4912 = (float*)myMalloc(1 * sizeof(float));;
x4912[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1445, x1445));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1461, x1461));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4912, filt_desc, x994, grad_out_desc, x1473,
    conv_desc, algo, ws_data, ws_size,
    x4912, grad_in_desc, x1453));
};
float* x4915 = (float*)myMalloc(1 * sizeof(float));;
x4915[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1461, x1461));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1445, x1445));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
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
    x4915, in_desc, x1451, grad_out_desc, x1473,
    conv_desc, algo, ws_data, ws_size,
    x4915, grad_filt_desc, x1340));
};
float* x4918 = (float*)myMalloc(1 * sizeof(float));;
x4918[0] = 0.0f;
float* x4920 = (float*)myMalloc(1 * sizeof(float));;
x4920[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1412, x1412));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1445, x1445));

cudnnPoolingDescriptor_t poolingDesc;
CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));
CUDNN_CALL(cudnnSetPooling2dDescriptor(
    poolingDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
    2, 2, 0,
    0, 2, 2
));
CUDNN_CALL(cudnnPoolingBackward(
    cudnnHandle, 
    poolingDesc, 
    x4920, out_desc, x1451, out_desc, x1453, in_desc, x1425  , x4918, in_desc, x1433));
};
float* x4923 = (float*)myMalloc(1 * sizeof(float));;
x4923[0] = 1.0f;
float* x4925 = (float*)myMalloc(1 * sizeof(float));;
x4925[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1412, x1412));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x4923, x_desc, x1425, x_desc, x1433, x_desc, x1425,
    x4925, x_desc, x1433));
};
float* x4928 = (float*)myMalloc(1 * sizeof(float));;
x4928[0] = 0.0f;
float* x4930 = (float*)myMalloc(1 * sizeof(float));;
x4930[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1412, x1412));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1412, x1412));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x4930, x4930, x4930, x4930, in_desc, x1418,
    out_desc, x1433, in_desc, x1424, sbmv_desc, x913,
    x1313,x1358, 1.0E-5, x1426, x1427));
};
// conv2D back-propagate
float* x4934 = (float*)myMalloc(1 * sizeof(float));;
x4934[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 3, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, x1412, x1412));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 3, 32, 32));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
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
    x4934, in_desc, x1402, grad_out_desc, x1424,
    conv_desc, algo, ws_data, ws_size,
    x4934, grad_filt_desc, x1259));
};
float x4937 = x1410[0];
x1390 += x4937;
float* x4939 = (float*)myMalloc(1 * sizeof(float));;
x4939[0] = 1.0f;
float* x4941 = (float*)myMalloc(1 * sizeof(float));;
x4941[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,256,x4939,x313,1024,x4941, x1113, 1024, x313,1024));
arrayFill_greg<<<28, 512>>>(x1113, 0.0f, 262144);
float* x4945 = (float*)myMalloc(1 * sizeof(float));;
x4945[0] = 1.0f;
float* x4947 = (float*)myMalloc(1 * sizeof(float));;
x4947[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4945,x316,1,x4947, x1114, 1, x316,1));
arrayFill_greg<<<28, 512>>>(x1114, 0.0f, 256);
float* x4951 = (float*)myMalloc(1 * sizeof(float));;
x4951[0] = 1.0f;
float* x4953 = (float*)myMalloc(1 * sizeof(float));;
x4953[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4951,x319,1,x4953, x1115, 1, x319,1));
arrayFill_greg<<<28, 512>>>(x1115, 0.0f, 128);
float* x4957 = (float*)myMalloc(1 * sizeof(float));;
x4957[0] = 1.0f;
float* x4959 = (float*)myMalloc(1 * sizeof(float));;
x4959[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4957,x322,1,x4959, x1116, 1, x322,1));
arrayFill_greg<<<28, 512>>>(x1116, 0.0f, 128);
float* x4963 = (float*)myMalloc(1 * sizeof(float));;
x4963[0] = 1.0f;
float* x4965 = (float*)myMalloc(1 * sizeof(float));;
x4965[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4963,x325,1,x4965, x1117, 1, x325,1));
arrayFill_greg<<<28, 512>>>(x1117, 0.0f, 64);
float* x4969 = (float*)myMalloc(1 * sizeof(float));;
x4969[0] = 1.0f;
float* x4971 = (float*)myMalloc(1 * sizeof(float));;
x4971[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,128,x4969,x328,256,x4971, x1118, 256, x328,256));
arrayFill_greg<<<28, 512>>>(x1118, 0.0f, 32768);
float* x4975 = (float*)myMalloc(1 * sizeof(float));;
x4975[0] = 1.0f;
float* x4977 = (float*)myMalloc(1 * sizeof(float));;
x4977[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4975,x331,1,x4977, x1119, 1, x331,1));
arrayFill_greg<<<28, 512>>>(x1119, 0.0f, 512);
float* x4981 = (float*)myMalloc(1 * sizeof(float));;
x4981[0] = 1.0f;
float* x4983 = (float*)myMalloc(1 * sizeof(float));;
x4983[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,256,x4981,x334,1024,x4983, x1120, 1024, x334,1024));
arrayFill_greg<<<28, 512>>>(x1120, 0.0f, 262144);
float* x4987 = (float*)myMalloc(1 * sizeof(float));;
x4987[0] = 1.0f;
float* x4989 = (float*)myMalloc(1 * sizeof(float));;
x4989[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x4987,x337,2304,x4989, x1121, 2304, x337,2304));
arrayFill_greg<<<28, 512>>>(x1121, 0.0f, 589824);
float* x4993 = (float*)myMalloc(1 * sizeof(float));;
x4993[0] = 1.0f;
float* x4995 = (float*)myMalloc(1 * sizeof(float));;
x4995[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4993,x340,1,x4995, x1122, 1, x340,1));
arrayFill_greg<<<28, 512>>>(x1122, 0.0f, 512);
float* x4999 = (float*)myMalloc(1 * sizeof(float));;
x4999[0] = 1.0f;
float* x5001 = (float*)myMalloc(1 * sizeof(float));;
x5001[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4999,x343,1,x5001, x1123, 1, x343,1));
arrayFill_greg<<<28, 512>>>(x1123, 0.0f, 256);
float* x5005 = (float*)myMalloc(1 * sizeof(float));;
x5005[0] = 1.0f;
float* x5007 = (float*)myMalloc(1 * sizeof(float));;
x5007[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5005,x346,1,x5007, x1124, 1, x346,1));
arrayFill_greg<<<28, 512>>>(x1124, 0.0f, 128);
float* x5011 = (float*)myMalloc(1 * sizeof(float));;
x5011[0] = 1.0f;
float* x5013 = (float*)myMalloc(1 * sizeof(float));;
x5013[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5011,x349,1,x5013, x1125, 1, x349,1));
arrayFill_greg<<<28, 512>>>(x1125, 0.0f, 1024);
float* x5017 = (float*)myMalloc(1 * sizeof(float));;
x5017[0] = 1.0f;
float* x5019 = (float*)myMalloc(1 * sizeof(float));;
x5019[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5017,x352,1,x5019, x1126, 1, x352,1));
arrayFill_greg<<<28, 512>>>(x1126, 0.0f, 512);
float* x5023 = (float*)myMalloc(1 * sizeof(float));;
x5023[0] = 1.0f;
float* x5025 = (float*)myMalloc(1 * sizeof(float));;
x5025[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5023,x355,1,x5025, x1127, 1, x355,1));
arrayFill_greg<<<28, 512>>>(x1127, 0.0f, 1024);
float* x5029 = (float*)myMalloc(1 * sizeof(float));;
x5029[0] = 1.0f;
float* x5031 = (float*)myMalloc(1 * sizeof(float));;
x5031[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5029,x358,1,x5031, x1128, 1, x358,1));
arrayFill_greg<<<28, 512>>>(x1128, 0.0f, 256);
float* x5035 = (float*)myMalloc(1 * sizeof(float));;
x5035[0] = 1.0f;
float* x5037 = (float*)myMalloc(1 * sizeof(float));;
x5037[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,256,x5035,x361,1024,x5037, x1129, 1024, x361,1024));
arrayFill_greg<<<28, 512>>>(x1129, 0.0f, 262144);
float* x5041 = (float*)myMalloc(1 * sizeof(float));;
x5041[0] = 1.0f;
float* x5043 = (float*)myMalloc(1 * sizeof(float));;
x5043[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5041,x364,1,x5043, x1130, 1, x364,1));
arrayFill_greg<<<28, 512>>>(x1130, 0.0f, 512);
float* x5047 = (float*)myMalloc(1 * sizeof(float));;
x5047[0] = 1.0f;
float* x5049 = (float*)myMalloc(1 * sizeof(float));;
x5049[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5047,x367,1,x5049, x1131, 1, x367,1));
arrayFill_greg<<<28, 512>>>(x1131, 0.0f, 64);
float* x5053 = (float*)myMalloc(1 * sizeof(float));;
x5053[0] = 1.0f;
float* x5055 = (float*)myMalloc(1 * sizeof(float));;
x5055[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5053,x370,1,x5055, x1132, 1, x370,1));
arrayFill_greg<<<28, 512>>>(x1132, 0.0f, 512);
float* x5059 = (float*)myMalloc(1 * sizeof(float));;
x5059[0] = 1.0f;
float* x5061 = (float*)myMalloc(1 * sizeof(float));;
x5061[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5059,x373,1,x5061, x1133, 1, x373,1));
arrayFill_greg<<<28, 512>>>(x1133, 0.0f, 64);
float* x5065 = (float*)myMalloc(1 * sizeof(float));;
x5065[0] = 1.0f;
float* x5067 = (float*)myMalloc(1 * sizeof(float));;
x5067[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1152,128,x5065,x376,1152,x5067, x1134, 1152, x376,1152));
arrayFill_greg<<<28, 512>>>(x1134, 0.0f, 147456);
float* x5071 = (float*)myMalloc(1 * sizeof(float));;
x5071[0] = 1.0f;
float* x5073 = (float*)myMalloc(1 * sizeof(float));;
x5073[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 4608,512,x5071,x379,4608,x5073, x1135, 4608, x379,4608));
arrayFill_greg<<<28, 512>>>(x1135, 0.0f, 2359296);
float* x5077 = (float*)myMalloc(1 * sizeof(float));;
x5077[0] = 1.0f;
float* x5079 = (float*)myMalloc(1 * sizeof(float));;
x5079[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5077,x382,1,x5079, x1136, 1, x382,1));
arrayFill_greg<<<28, 512>>>(x1136, 0.0f, 1024);
float* x5083 = (float*)myMalloc(1 * sizeof(float));;
x5083[0] = 1.0f;
float* x5085 = (float*)myMalloc(1 * sizeof(float));;
x5085[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5083,x385,1,x5085, x1137, 1, x385,1));
arrayFill_greg<<<28, 512>>>(x1137, 0.0f, 256);
float* x5089 = (float*)myMalloc(1 * sizeof(float));;
x5089[0] = 1.0f;
float* x5091 = (float*)myMalloc(1 * sizeof(float));;
x5091[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x5089,x388,2304,x5091, x1138, 2304, x388,2304));
arrayFill_greg<<<28, 512>>>(x1138, 0.0f, 589824);
float* x5095 = (float*)myMalloc(1 * sizeof(float));;
x5095[0] = 1.0f;
float* x5097 = (float*)myMalloc(1 * sizeof(float));;
x5097[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x5095,x391,64,x5097, x1139, 64, x391,64));
arrayFill_greg<<<28, 512>>>(x1139, 0.0f, 16384);
float* x5101 = (float*)myMalloc(1 * sizeof(float));;
x5101[0] = 1.0f;
float* x5103 = (float*)myMalloc(1 * sizeof(float));;
x5103[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,2048,x5101,x394,512,x5103, x1140, 512, x394,512));
arrayFill_greg<<<28, 512>>>(x1140, 0.0f, 1048576);
float* x5107 = (float*)myMalloc(1 * sizeof(float));;
x5107[0] = 1.0f;
float* x5109 = (float*)myMalloc(1 * sizeof(float));;
x5109[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 4608,512,x5107,x397,4608,x5109, x1141, 4608, x397,4608));
arrayFill_greg<<<28, 512>>>(x1141, 0.0f, 2359296);
float* x5113 = (float*)myMalloc(1 * sizeof(float));;
x5113[0] = 1.0f;
float* x5115 = (float*)myMalloc(1 * sizeof(float));;
x5115[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5113,x400,1,x5115, x1142, 1, x400,1));
arrayFill_greg<<<28, 512>>>(x1142, 0.0f, 128);
float* x5119 = (float*)myMalloc(1 * sizeof(float));;
x5119[0] = 1.0f;
float* x5121 = (float*)myMalloc(1 * sizeof(float));;
x5121[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5119,x403,1,x5121, x1143, 1, x403,1));
arrayFill_greg<<<28, 512>>>(x1143, 0.0f, 256);
float* x5125 = (float*)myMalloc(1 * sizeof(float));;
x5125[0] = 1.0f;
float* x5127 = (float*)myMalloc(1 * sizeof(float));;
x5127[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5125,x406,1,x5127, x1144, 1, x406,1));
arrayFill_greg<<<28, 512>>>(x1144, 0.0f, 256);
float* x5131 = (float*)myMalloc(1 * sizeof(float));;
x5131[0] = 1.0f;
float* x5133 = (float*)myMalloc(1 * sizeof(float));;
x5133[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5131,x409,1,x5133, x1145, 1, x409,1));
arrayFill_greg<<<28, 512>>>(x1145, 0.0f, 128);
float* x5137 = (float*)myMalloc(1 * sizeof(float));;
x5137[0] = 1.0f;
float* x5139 = (float*)myMalloc(1 * sizeof(float));;
x5139[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5137,x412,1,x5139, x1146, 1, x412,1));
arrayFill_greg<<<28, 512>>>(x1146, 0.0f, 128);
float* x5143 = (float*)myMalloc(1 * sizeof(float));;
x5143[0] = 1.0f;
float* x5145 = (float*)myMalloc(1 * sizeof(float));;
x5145[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5143,x415,1,x5145, x1147, 1, x415,1));
arrayFill_greg<<<28, 512>>>(x1147, 0.0f, 64);
float* x5149 = (float*)myMalloc(1 * sizeof(float));;
x5149[0] = 1.0f;
float* x5151 = (float*)myMalloc(1 * sizeof(float));;
x5151[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,512,x5149,x418,128,x5151, x1148, 128, x418,128));
arrayFill_greg<<<28, 512>>>(x1148, 0.0f, 65536);
float* x5155 = (float*)myMalloc(1 * sizeof(float));;
x5155[0] = 1.0f;
float* x5157 = (float*)myMalloc(1 * sizeof(float));;
x5157[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5155,x421,1,x5157, x1149, 1, x421,1));
arrayFill_greg<<<28, 512>>>(x1149, 0.0f, 512);
float* x5161 = (float*)myMalloc(1 * sizeof(float));;
x5161[0] = 1.0f;
float* x5163 = (float*)myMalloc(1 * sizeof(float));;
x5163[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5161,x424,1,x5163, x1150, 1, x424,1));
arrayFill_greg<<<28, 512>>>(x1150, 0.0f, 128);
float* x5167 = (float*)myMalloc(1 * sizeof(float));;
x5167[0] = 1.0f;
float* x5169 = (float*)myMalloc(1 * sizeof(float));;
x5169[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5167,x427,1,x5169, x1151, 1, x427,1));
arrayFill_greg<<<28, 512>>>(x1151, 0.0f, 64);
float* x5173 = (float*)myMalloc(1 * sizeof(float));;
x5173[0] = 1.0f;
float* x5175 = (float*)myMalloc(1 * sizeof(float));;
x5175[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5173,x430,1,x5175, x1152, 1, x430,1));
arrayFill_greg<<<28, 512>>>(x1152, 0.0f, 128);
float* x5179 = (float*)myMalloc(1 * sizeof(float));;
x5179[0] = 1.0f;
float* x5181 = (float*)myMalloc(1 * sizeof(float));;
x5181[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5179,x433,1,x5181, x1153, 1, x433,1));
arrayFill_greg<<<28, 512>>>(x1153, 0.0f, 512);
float* x5185 = (float*)myMalloc(1 * sizeof(float));;
x5185[0] = 1.0f;
float* x5187 = (float*)myMalloc(1 * sizeof(float));;
x5187[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,2048,x5185,x436,512,x5187, x1154, 512, x436,512));
arrayFill_greg<<<28, 512>>>(x1154, 0.0f, 1048576);
float* x5191 = (float*)myMalloc(1 * sizeof(float));;
x5191[0] = 1.0f;
float* x5193 = (float*)myMalloc(1 * sizeof(float));;
x5193[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,10,x5191,x439,1,x5193, x1155, 1, x439,1));
arrayFill_greg<<<28, 512>>>(x1155, 0.0f, 10);
float* x5197 = (float*)myMalloc(1 * sizeof(float));;
x5197[0] = 1.0f;
float* x5199 = (float*)myMalloc(1 * sizeof(float));;
x5199[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5197,x442,1,x5199, x1156, 1, x442,1));
arrayFill_greg<<<28, 512>>>(x1156, 0.0f, 64);
float* x5203 = (float*)myMalloc(1 * sizeof(float));;
x5203[0] = 1.0f;
float* x5205 = (float*)myMalloc(1 * sizeof(float));;
x5205[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5203,x445,1,x5205, x1157, 1, x445,1));
arrayFill_greg<<<28, 512>>>(x1157, 0.0f, 512);
float* x5209 = (float*)myMalloc(1 * sizeof(float));;
x5209[0] = 1.0f;
float* x5211 = (float*)myMalloc(1 * sizeof(float));;
x5211[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5209,x448,1,x5211, x1158, 1, x448,1));
arrayFill_greg<<<28, 512>>>(x1158, 0.0f, 64);
float* x5215 = (float*)myMalloc(1 * sizeof(float));;
x5215[0] = 1.0f;
float* x5217 = (float*)myMalloc(1 * sizeof(float));;
x5217[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5215,x451,1,x5217, x1159, 1, x451,1));
arrayFill_greg<<<28, 512>>>(x1159, 0.0f, 512);
float* x5221 = (float*)myMalloc(1 * sizeof(float));;
x5221[0] = 1.0f;
float* x5223 = (float*)myMalloc(1 * sizeof(float));;
x5223[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5221,x454,1,x5223, x1160, 1, x454,1));
arrayFill_greg<<<28, 512>>>(x1160, 0.0f, 64);
float* x5227 = (float*)myMalloc(1 * sizeof(float));;
x5227[0] = 1.0f;
float* x5229 = (float*)myMalloc(1 * sizeof(float));;
x5229[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5227,x457,1,x5229, x1161, 1, x457,1));
arrayFill_greg<<<28, 512>>>(x1161, 0.0f, 512);
float* x5233 = (float*)myMalloc(1 * sizeof(float));;
x5233[0] = 1.0f;
float* x5235 = (float*)myMalloc(1 * sizeof(float));;
x5235[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,512,x5233,x460,128,x5235, x1162, 128, x460,128));
arrayFill_greg<<<28, 512>>>(x1162, 0.0f, 65536);
float* x5239 = (float*)myMalloc(1 * sizeof(float));;
x5239[0] = 1.0f;
float* x5241 = (float*)myMalloc(1 * sizeof(float));;
x5241[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x5239,x463,256,x5241, x1163, 256, x463,256));
arrayFill_greg<<<28, 512>>>(x1163, 0.0f, 262144);
float* x5245 = (float*)myMalloc(1 * sizeof(float));;
x5245[0] = 1.0f;
float* x5247 = (float*)myMalloc(1 * sizeof(float));;
x5247[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5245,x466,1,x5247, x1164, 1, x466,1));
arrayFill_greg<<<28, 512>>>(x1164, 0.0f, 1024);
float* x5251 = (float*)myMalloc(1 * sizeof(float));;
x5251[0] = 1.0f;
float* x5253 = (float*)myMalloc(1 * sizeof(float));;
x5253[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5251,x469,1,x5253, x1165, 1, x469,1));
arrayFill_greg<<<28, 512>>>(x1165, 0.0f, 256);
float* x5257 = (float*)myMalloc(1 * sizeof(float));;
x5257[0] = 1.0f;
float* x5259 = (float*)myMalloc(1 * sizeof(float));;
x5259[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5257,x472,1,x5259, x1166, 1, x472,1));
arrayFill_greg<<<28, 512>>>(x1166, 0.0f, 1024);
float* x5263 = (float*)myMalloc(1 * sizeof(float));;
x5263[0] = 1.0f;
float* x5265 = (float*)myMalloc(1 * sizeof(float));;
x5265[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5263,x475,1,x5265, x1167, 1, x475,1));
arrayFill_greg<<<28, 512>>>(x1167, 0.0f, 64);
float* x5269 = (float*)myMalloc(1 * sizeof(float));;
x5269[0] = 1.0f;
float* x5271 = (float*)myMalloc(1 * sizeof(float));;
x5271[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5269,x478,1,x5271, x1168, 1, x478,1));
arrayFill_greg<<<28, 512>>>(x1168, 0.0f, 128);
float* x5275 = (float*)myMalloc(1 * sizeof(float));;
x5275[0] = 1.0f;
float* x5277 = (float*)myMalloc(1 * sizeof(float));;
x5277[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5275,x481,1,x5277, x1169, 1, x481,1));
arrayFill_greg<<<28, 512>>>(x1169, 0.0f, 2048);
float* x5281 = (float*)myMalloc(1 * sizeof(float));;
x5281[0] = 1.0f;
float* x5283 = (float*)myMalloc(1 * sizeof(float));;
x5283[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5281,x484,1,x5283, x1170, 1, x484,1));
arrayFill_greg<<<28, 512>>>(x1170, 0.0f, 256);
float* x5287 = (float*)myMalloc(1 * sizeof(float));;
x5287[0] = 1.0f;
float* x5289 = (float*)myMalloc(1 * sizeof(float));;
x5289[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5287,x487,1,x5289, x1171, 1, x487,1));
arrayFill_greg<<<28, 512>>>(x1171, 0.0f, 2048);
float* x5293 = (float*)myMalloc(1 * sizeof(float));;
x5293[0] = 1.0f;
float* x5295 = (float*)myMalloc(1 * sizeof(float));;
x5295[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5293,x490,1,x5295, x1172, 1, x490,1));
arrayFill_greg<<<28, 512>>>(x1172, 0.0f, 512);
float* x5299 = (float*)myMalloc(1 * sizeof(float));;
x5299[0] = 1.0f;
float* x5301 = (float*)myMalloc(1 * sizeof(float));;
x5301[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5299,x493,1,x5301, x1173, 1, x493,1));
arrayFill_greg<<<28, 512>>>(x1173, 0.0f, 512);
float* x5305 = (float*)myMalloc(1 * sizeof(float));;
x5305[0] = 1.0f;
float* x5307 = (float*)myMalloc(1 * sizeof(float));;
x5307[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5305,x496,1,x5307, x1174, 1, x496,1));
arrayFill_greg<<<28, 512>>>(x1174, 0.0f, 512);
float* x5311 = (float*)myMalloc(1 * sizeof(float));;
x5311[0] = 1.0f;
float* x5313 = (float*)myMalloc(1 * sizeof(float));;
x5313[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5311,x499,1,x5313, x1175, 1, x499,1));
arrayFill_greg<<<28, 512>>>(x1175, 0.0f, 2048);
float* x5317 = (float*)myMalloc(1 * sizeof(float));;
x5317[0] = 1.0f;
float* x5319 = (float*)myMalloc(1 * sizeof(float));;
x5319[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5317,x502,1,x5319, x1176, 1, x502,1));
arrayFill_greg<<<28, 512>>>(x1176, 0.0f, 256);
float* x5323 = (float*)myMalloc(1 * sizeof(float));;
x5323[0] = 1.0f;
float* x5325 = (float*)myMalloc(1 * sizeof(float));;
x5325[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5323,x505,1,x5325, x1177, 1, x505,1));
arrayFill_greg<<<28, 512>>>(x1177, 0.0f, 256);
float* x5329 = (float*)myMalloc(1 * sizeof(float));;
x5329[0] = 1.0f;
float* x5331 = (float*)myMalloc(1 * sizeof(float));;
x5331[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5329,x508,1,x5331, x1178, 1, x508,1));
arrayFill_greg<<<28, 512>>>(x1178, 0.0f, 256);
float* x5335 = (float*)myMalloc(1 * sizeof(float));;
x5335[0] = 1.0f;
float* x5337 = (float*)myMalloc(1 * sizeof(float));;
x5337[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5335,x511,1,x5337, x1179, 1, x511,1));
arrayFill_greg<<<28, 512>>>(x1179, 0.0f, 64);
float* x5341 = (float*)myMalloc(1 * sizeof(float));;
x5341[0] = 1.0f;
float* x5343 = (float*)myMalloc(1 * sizeof(float));;
x5343[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 576,64,x5341,x514,576,x5343, x1180, 576, x514,576));
arrayFill_greg<<<28, 512>>>(x1180, 0.0f, 36864);
float* x5347 = (float*)myMalloc(1 * sizeof(float));;
x5347[0] = 1.0f;
float* x5349 = (float*)myMalloc(1 * sizeof(float));;
x5349[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5347,x517,1,x5349, x1181, 1, x517,1));
arrayFill_greg<<<28, 512>>>(x1181, 0.0f, 256);
float* x5353 = (float*)myMalloc(1 * sizeof(float));;
x5353[0] = 1.0f;
float* x5355 = (float*)myMalloc(1 * sizeof(float));;
x5355[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,1024,x5353,x520,512,x5355, x1182, 512, x520,512));
arrayFill_greg<<<28, 512>>>(x1182, 0.0f, 524288);
float* x5359 = (float*)myMalloc(1 * sizeof(float));;
x5359[0] = 1.0f;
float* x5361 = (float*)myMalloc(1 * sizeof(float));;
x5361[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5359,x523,1,x5361, x1183, 1, x523,1));
arrayFill_greg<<<28, 512>>>(x1183, 0.0f, 256);
float* x5365 = (float*)myMalloc(1 * sizeof(float));;
x5365[0] = 1.0f;
float* x5367 = (float*)myMalloc(1 * sizeof(float));;
x5367[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5365,x526,1,x5367, x1184, 1, x526,1));
arrayFill_greg<<<28, 512>>>(x1184, 0.0f, 256);
float* x5371 = (float*)myMalloc(1 * sizeof(float));;
x5371[0] = 1.0f;
float* x5373 = (float*)myMalloc(1 * sizeof(float));;
x5373[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5371,x529,1,x5373, x1185, 1, x529,1));
arrayFill_greg<<<28, 512>>>(x1185, 0.0f, 512);
float* x5377 = (float*)myMalloc(1 * sizeof(float));;
x5377[0] = 1.0f;
float* x5379 = (float*)myMalloc(1 * sizeof(float));;
x5379[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5377,x532,1,x5379, x1186, 1, x532,1));
arrayFill_greg<<<28, 512>>>(x1186, 0.0f, 128);
float* x5383 = (float*)myMalloc(1 * sizeof(float));;
x5383[0] = 1.0f;
float* x5385 = (float*)myMalloc(1 * sizeof(float));;
x5385[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5383,x535,1,x5385, x1187, 1, x535,1));
arrayFill_greg<<<28, 512>>>(x1187, 0.0f, 256);
float* x5389 = (float*)myMalloc(1 * sizeof(float));;
x5389[0] = 1.0f;
float* x5391 = (float*)myMalloc(1 * sizeof(float));;
x5391[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5389,x538,1,x5391, x1188, 1, x538,1));
arrayFill_greg<<<28, 512>>>(x1188, 0.0f, 64);
float* x5395 = (float*)myMalloc(1 * sizeof(float));;
x5395[0] = 1.0f;
float* x5397 = (float*)myMalloc(1 * sizeof(float));;
x5397[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5395,x541,1,x5397, x1189, 1, x541,1));
arrayFill_greg<<<28, 512>>>(x1189, 0.0f, 512);
float* x5401 = (float*)myMalloc(1 * sizeof(float));;
x5401[0] = 1.0f;
float* x5403 = (float*)myMalloc(1 * sizeof(float));;
x5403[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 576,64,x5401,x544,576,x5403, x1190, 576, x544,576));
arrayFill_greg<<<28, 512>>>(x1190, 0.0f, 36864);
float* x5407 = (float*)myMalloc(1 * sizeof(float));;
x5407[0] = 1.0f;
float* x5409 = (float*)myMalloc(1 * sizeof(float));;
x5409[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5407,x547,1,x5409, x1191, 1, x547,1));
arrayFill_greg<<<28, 512>>>(x1191, 0.0f, 128);
float* x5413 = (float*)myMalloc(1 * sizeof(float));;
x5413[0] = 1.0f;
float* x5415 = (float*)myMalloc(1 * sizeof(float));;
x5415[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5413,x550,1,x5415, x1192, 1, x550,1));
arrayFill_greg<<<28, 512>>>(x1192, 0.0f, 256);
float* x5419 = (float*)myMalloc(1 * sizeof(float));;
x5419[0] = 1.0f;
float* x5421 = (float*)myMalloc(1 * sizeof(float));;
x5421[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5419,x553,1,x5421, x1193, 1, x553,1));
arrayFill_greg<<<28, 512>>>(x1193, 0.0f, 1024);
float* x5425 = (float*)myMalloc(1 * sizeof(float));;
x5425[0] = 1.0f;
float* x5427 = (float*)myMalloc(1 * sizeof(float));;
x5427[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x5425,x556,64,x5427, x1194, 64, x556,64));
arrayFill_greg<<<28, 512>>>(x1194, 0.0f, 16384);
float* x5431 = (float*)myMalloc(1 * sizeof(float));;
x5431[0] = 1.0f;
float* x5433 = (float*)myMalloc(1 * sizeof(float));;
x5433[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5431,x559,1,x5433, x1195, 1, x559,1));
arrayFill_greg<<<28, 512>>>(x1195, 0.0f, 512);
float* x5437 = (float*)myMalloc(1 * sizeof(float));;
x5437[0] = 1.0f;
float* x5439 = (float*)myMalloc(1 * sizeof(float));;
x5439[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x5437,x562,256,x5439, x1196, 256, x562,256));
arrayFill_greg<<<28, 512>>>(x1196, 0.0f, 262144);
float* x5443 = (float*)myMalloc(1 * sizeof(float));;
x5443[0] = 1.0f;
float* x5445 = (float*)myMalloc(1 * sizeof(float));;
x5445[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 576,64,x5443,x565,576,x5445, x1197, 576, x565,576));
arrayFill_greg<<<28, 512>>>(x1197, 0.0f, 36864);
float* x5449 = (float*)myMalloc(1 * sizeof(float));;
x5449[0] = 1.0f;
float* x5451 = (float*)myMalloc(1 * sizeof(float));;
x5451[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5449,x568,1,x5451, x1198, 1, x568,1));
arrayFill_greg<<<28, 512>>>(x1198, 0.0f, 256);
float* x5455 = (float*)myMalloc(1 * sizeof(float));;
x5455[0] = 1.0f;
float* x5457 = (float*)myMalloc(1 * sizeof(float));;
x5457[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5455,x571,1,x5457, x1199, 1, x571,1));
arrayFill_greg<<<28, 512>>>(x1199, 0.0f, 256);
float* x5461 = (float*)myMalloc(1 * sizeof(float));;
x5461[0] = 1.0f;
float* x5463 = (float*)myMalloc(1 * sizeof(float));;
x5463[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5461,x574,1,x5463, x1200, 1, x574,1));
arrayFill_greg<<<28, 512>>>(x1200, 0.0f, 1024);
float* x5467 = (float*)myMalloc(1 * sizeof(float));;
x5467[0] = 1.0f;
float* x5469 = (float*)myMalloc(1 * sizeof(float));;
x5469[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5467,x577,1,x5469, x1201, 1, x577,1));
arrayFill_greg<<<28, 512>>>(x1201, 0.0f, 2048);
float* x5473 = (float*)myMalloc(1 * sizeof(float));;
x5473[0] = 1.0f;
float* x5475 = (float*)myMalloc(1 * sizeof(float));;
x5475[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5473,x580,1,x5475, x1202, 1, x580,1));
arrayFill_greg<<<28, 512>>>(x1202, 0.0f, 128);
float* x5479 = (float*)myMalloc(1 * sizeof(float));;
x5479[0] = 1.0f;
float* x5481 = (float*)myMalloc(1 * sizeof(float));;
x5481[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5479,x583,1,x5481, x1203, 1, x583,1));
arrayFill_greg<<<28, 512>>>(x1203, 0.0f, 256);
float* x5485 = (float*)myMalloc(1 * sizeof(float));;
x5485[0] = 1.0f;
float* x5487 = (float*)myMalloc(1 * sizeof(float));;
x5487[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x5485,x586,256,x5487, x1204, 256, x586,256));
arrayFill_greg<<<28, 512>>>(x1204, 0.0f, 262144);
float* x5491 = (float*)myMalloc(1 * sizeof(float));;
x5491[0] = 1.0f;
float* x5493 = (float*)myMalloc(1 * sizeof(float));;
x5493[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5491,x589,1,x5493, x1205, 1, x589,1));
arrayFill_greg<<<28, 512>>>(x1205, 0.0f, 256);
float* x5497 = (float*)myMalloc(1 * sizeof(float));;
x5497[0] = 1.0f;
float* x5499 = (float*)myMalloc(1 * sizeof(float));;
x5499[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5497,x592,1,x5499, x1206, 1, x592,1));
arrayFill_greg<<<28, 512>>>(x1206, 0.0f, 256);
float* x5503 = (float*)myMalloc(1 * sizeof(float));;
x5503[0] = 1.0f;
float* x5505 = (float*)myMalloc(1 * sizeof(float));;
x5505[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5503,x595,1,x5505, x1207, 1, x595,1));
arrayFill_greg<<<28, 512>>>(x1207, 0.0f, 128);
float* x5509 = (float*)myMalloc(1 * sizeof(float));;
x5509[0] = 1.0f;
float* x5511 = (float*)myMalloc(1 * sizeof(float));;
x5511[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5509,x598,1,x5511, x1208, 1, x598,1));
arrayFill_greg<<<28, 512>>>(x1208, 0.0f, 512);
float* x5515 = (float*)myMalloc(1 * sizeof(float));;
x5515[0] = 1.0f;
float* x5517 = (float*)myMalloc(1 * sizeof(float));;
x5517[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5515,x601,1,x5517, x1209, 1, x601,1));
arrayFill_greg<<<28, 512>>>(x1209, 0.0f, 64);
float* x5521 = (float*)myMalloc(1 * sizeof(float));;
x5521[0] = 1.0f;
float* x5523 = (float*)myMalloc(1 * sizeof(float));;
x5523[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5521,x604,1,x5523, x1210, 1, x604,1));
arrayFill_greg<<<28, 512>>>(x1210, 0.0f, 2048);
float* x5527 = (float*)myMalloc(1 * sizeof(float));;
x5527[0] = 1.0f;
float* x5529 = (float*)myMalloc(1 * sizeof(float));;
x5529[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5527,x607,1,x5529, x1211, 1, x607,1));
arrayFill_greg<<<28, 512>>>(x1211, 0.0f, 256);
float* x5533 = (float*)myMalloc(1 * sizeof(float));;
x5533[0] = 1.0f;
float* x5535 = (float*)myMalloc(1 * sizeof(float));;
x5535[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5533,x610,1,x5535, x1212, 1, x610,1));
arrayFill_greg<<<28, 512>>>(x1212, 0.0f, 64);
float* x5539 = (float*)myMalloc(1 * sizeof(float));;
x5539[0] = 1.0f;
float* x5541 = (float*)myMalloc(1 * sizeof(float));;
x5541[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,512,x5539,x613,128,x5541, x1213, 128, x613,128));
arrayFill_greg<<<28, 512>>>(x1213, 0.0f, 65536);
float* x5545 = (float*)myMalloc(1 * sizeof(float));;
x5545[0] = 1.0f;
float* x5547 = (float*)myMalloc(1 * sizeof(float));;
x5547[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5545,x616,1,x5547, x1214, 1, x616,1));
arrayFill_greg<<<28, 512>>>(x1214, 0.0f, 2048);
float* x5551 = (float*)myMalloc(1 * sizeof(float));;
x5551[0] = 1.0f;
float* x5553 = (float*)myMalloc(1 * sizeof(float));;
x5553[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5551,x619,1,x5553, x1215, 1, x619,1));
arrayFill_greg<<<28, 512>>>(x1215, 0.0f, 256);
float* x5557 = (float*)myMalloc(1 * sizeof(float));;
x5557[0] = 1.0f;
float* x5559 = (float*)myMalloc(1 * sizeof(float));;
x5559[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5557,x622,1,x5559, x1216, 1, x622,1));
arrayFill_greg<<<28, 512>>>(x1216, 0.0f, 256);
float* x5563 = (float*)myMalloc(1 * sizeof(float));;
x5563[0] = 1.0f;
float* x5565 = (float*)myMalloc(1 * sizeof(float));;
x5565[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5563,x625,1,x5565, x1217, 1, x625,1));
arrayFill_greg<<<28, 512>>>(x1217, 0.0f, 64);
float* x5569 = (float*)myMalloc(1 * sizeof(float));;
x5569[0] = 1.0f;
float* x5571 = (float*)myMalloc(1 * sizeof(float));;
x5571[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,512,x5569,x628,128,x5571, x1218, 128, x628,128));
arrayFill_greg<<<28, 512>>>(x1218, 0.0f, 65536);
float* x5575 = (float*)myMalloc(1 * sizeof(float));;
x5575[0] = 1.0f;
float* x5577 = (float*)myMalloc(1 * sizeof(float));;
x5577[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5575,x631,1,x5577, x1219, 1, x631,1));
arrayFill_greg<<<28, 512>>>(x1219, 0.0f, 128);
float* x5581 = (float*)myMalloc(1 * sizeof(float));;
x5581[0] = 1.0f;
float* x5583 = (float*)myMalloc(1 * sizeof(float));;
x5583[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5581,x634,1,x5583, x1220, 1, x634,1));
arrayFill_greg<<<28, 512>>>(x1220, 0.0f, 512);
float* x5587 = (float*)myMalloc(1 * sizeof(float));;
x5587[0] = 1.0f;
float* x5589 = (float*)myMalloc(1 * sizeof(float));;
x5589[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5587,x637,1,x5589, x1221, 1, x637,1));
arrayFill_greg<<<28, 512>>>(x1221, 0.0f, 64);
float* x5593 = (float*)myMalloc(1 * sizeof(float));;
x5593[0] = 1.0f;
float* x5595 = (float*)myMalloc(1 * sizeof(float));;
x5595[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5593,x640,1,x5595, x1222, 1, x640,1));
arrayFill_greg<<<28, 512>>>(x1222, 0.0f, 2048);
float* x5599 = (float*)myMalloc(1 * sizeof(float));;
x5599[0] = 1.0f;
float* x5601 = (float*)myMalloc(1 * sizeof(float));;
x5601[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x5599,x643,256,x5601, x1223, 256, x643,256));
arrayFill_greg<<<28, 512>>>(x1223, 0.0f, 262144);
float* x5605 = (float*)myMalloc(1 * sizeof(float));;
x5605[0] = 1.0f;
float* x5607 = (float*)myMalloc(1 * sizeof(float));;
x5607[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5605,x646,1,x5607, x1224, 1, x646,1));
arrayFill_greg<<<28, 512>>>(x1224, 0.0f, 1024);
float* x5611 = (float*)myMalloc(1 * sizeof(float));;
x5611[0] = 1.0f;
float* x5613 = (float*)myMalloc(1 * sizeof(float));;
x5613[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5611,x649,1,x5613, x1225, 1, x649,1));
arrayFill_greg<<<28, 512>>>(x1225, 0.0f, 64);
float* x5617 = (float*)myMalloc(1 * sizeof(float));;
x5617[0] = 1.0f;
float* x5619 = (float*)myMalloc(1 * sizeof(float));;
x5619[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5617,x652,1,x5619, x1226, 1, x652,1));
arrayFill_greg<<<28, 512>>>(x1226, 0.0f, 512);
float* x5623 = (float*)myMalloc(1 * sizeof(float));;
x5623[0] = 1.0f;
float* x5625 = (float*)myMalloc(1 * sizeof(float));;
x5625[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5623,x655,1,x5625, x1227, 1, x655,1));
arrayFill_greg<<<28, 512>>>(x1227, 0.0f, 1024);
float* x5629 = (float*)myMalloc(1 * sizeof(float));;
x5629[0] = 1.0f;
float* x5631 = (float*)myMalloc(1 * sizeof(float));;
x5631[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5629,x658,1,x5631, x1228, 1, x658,1));
arrayFill_greg<<<28, 512>>>(x1228, 0.0f, 512);
float* x5635 = (float*)myMalloc(1 * sizeof(float));;
x5635[0] = 1.0f;
float* x5637 = (float*)myMalloc(1 * sizeof(float));;
x5637[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5635,x661,1,x5637, x1229, 1, x661,1));
arrayFill_greg<<<28, 512>>>(x1229, 0.0f, 1024);
float* x5641 = (float*)myMalloc(1 * sizeof(float));;
x5641[0] = 1.0f;
float* x5643 = (float*)myMalloc(1 * sizeof(float));;
x5643[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5641,x664,1,x5643, x1230, 1, x664,1));
arrayFill_greg<<<28, 512>>>(x1230, 0.0f, 2048);
float* x5647 = (float*)myMalloc(1 * sizeof(float));;
x5647[0] = 1.0f;
float* x5649 = (float*)myMalloc(1 * sizeof(float));;
x5649[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5647,x667,1,x5649, x1231, 1, x667,1));
arrayFill_greg<<<28, 512>>>(x1231, 0.0f, 256);
float* x5653 = (float*)myMalloc(1 * sizeof(float));;
x5653[0] = 1.0f;
float* x5655 = (float*)myMalloc(1 * sizeof(float));;
x5655[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5653,x670,1,x5655, x1232, 1, x670,1));
arrayFill_greg<<<28, 512>>>(x1232, 0.0f, 2048);
float* x5659 = (float*)myMalloc(1 * sizeof(float));;
x5659[0] = 1.0f;
float* x5661 = (float*)myMalloc(1 * sizeof(float));;
x5661[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5659,x673,1,x5661, x1233, 1, x673,1));
arrayFill_greg<<<28, 512>>>(x1233, 0.0f, 256);
float* x5665 = (float*)myMalloc(1 * sizeof(float));;
x5665[0] = 1.0f;
float* x5667 = (float*)myMalloc(1 * sizeof(float));;
x5667[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5665,x676,1,x5667, x1234, 1, x676,1));
arrayFill_greg<<<28, 512>>>(x1234, 0.0f, 128);
float* x5671 = (float*)myMalloc(1 * sizeof(float));;
x5671[0] = 1.0f;
float* x5673 = (float*)myMalloc(1 * sizeof(float));;
x5673[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5671,x679,1,x5673, x1235, 1, x679,1));
arrayFill_greg<<<28, 512>>>(x1235, 0.0f, 128);
float* x5677 = (float*)myMalloc(1 * sizeof(float));;
x5677[0] = 1.0f;
float* x5679 = (float*)myMalloc(1 * sizeof(float));;
x5679[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5677,x682,1,x5679, x1236, 1, x682,1));
arrayFill_greg<<<28, 512>>>(x1236, 0.0f, 256);
float* x5683 = (float*)myMalloc(1 * sizeof(float));;
x5683[0] = 1.0f;
float* x5685 = (float*)myMalloc(1 * sizeof(float));;
x5685[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x5683,x685,64,x5685, x1237, 64, x685,64));
arrayFill_greg<<<28, 512>>>(x1237, 0.0f, 16384);
float* x5689 = (float*)myMalloc(1 * sizeof(float));;
x5689[0] = 1.0f;
float* x5691 = (float*)myMalloc(1 * sizeof(float));;
x5691[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5689,x688,1,x5691, x1238, 1, x688,1));
arrayFill_greg<<<28, 512>>>(x1238, 0.0f, 256);
float* x5695 = (float*)myMalloc(1 * sizeof(float));;
x5695[0] = 1.0f;
float* x5697 = (float*)myMalloc(1 * sizeof(float));;
x5697[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,128,x5695,x691,512,x5697, x1239, 512, x691,512));
arrayFill_greg<<<28, 512>>>(x1239, 0.0f, 65536);
float* x5701 = (float*)myMalloc(1 * sizeof(float));;
x5701[0] = 1.0f;
float* x5703 = (float*)myMalloc(1 * sizeof(float));;
x5703[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5701,x694,1,x5703, x1240, 1, x694,1));
arrayFill_greg<<<28, 512>>>(x1240, 0.0f, 256);
float* x5707 = (float*)myMalloc(1 * sizeof(float));;
x5707[0] = 1.0f;
float* x5709 = (float*)myMalloc(1 * sizeof(float));;
x5709[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5707,x697,1,x5709, x1241, 1, x697,1));
arrayFill_greg<<<28, 512>>>(x1241, 0.0f, 128);
float* x5713 = (float*)myMalloc(1 * sizeof(float));;
x5713[0] = 1.0f;
float* x5715 = (float*)myMalloc(1 * sizeof(float));;
x5715[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5713,x700,1,x5715, x1242, 1, x700,1));
arrayFill_greg<<<28, 512>>>(x1242, 0.0f, 64);
float* x5719 = (float*)myMalloc(1 * sizeof(float));;
x5719[0] = 1.0f;
float* x5721 = (float*)myMalloc(1 * sizeof(float));;
x5721[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5719,x703,1,x5721, x1243, 1, x703,1));
arrayFill_greg<<<28, 512>>>(x1243, 0.0f, 256);
float* x5725 = (float*)myMalloc(1 * sizeof(float));;
x5725[0] = 1.0f;
float* x5727 = (float*)myMalloc(1 * sizeof(float));;
x5727[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5725,x706,1,x5727, x1244, 1, x706,1));
arrayFill_greg<<<28, 512>>>(x1244, 0.0f, 512);
float* x5731 = (float*)myMalloc(1 * sizeof(float));;
x5731[0] = 1.0f;
float* x5733 = (float*)myMalloc(1 * sizeof(float));;
x5733[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5731,x709,1,x5733, x1245, 1, x709,1));
arrayFill_greg<<<28, 512>>>(x1245, 0.0f, 512);
float* x5737 = (float*)myMalloc(1 * sizeof(float));;
x5737[0] = 1.0f;
float* x5739 = (float*)myMalloc(1 * sizeof(float));;
x5739[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,512,x5737,x712,1024,x5739, x1246, 1024, x712,1024));
arrayFill_greg<<<28, 512>>>(x1246, 0.0f, 524288);
float* x5743 = (float*)myMalloc(1 * sizeof(float));;
x5743[0] = 1.0f;
float* x5745 = (float*)myMalloc(1 * sizeof(float));;
x5745[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5743,x715,1,x5745, x1247, 1, x715,1));
arrayFill_greg<<<28, 512>>>(x1247, 0.0f, 1024);
float* x5749 = (float*)myMalloc(1 * sizeof(float));;
x5749[0] = 1.0f;
float* x5751 = (float*)myMalloc(1 * sizeof(float));;
x5751[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5749,x718,1,x5751, x1248, 1, x718,1));
arrayFill_greg<<<28, 512>>>(x1248, 0.0f, 256);
float* x5755 = (float*)myMalloc(1 * sizeof(float));;
x5755[0] = 1.0f;
float* x5757 = (float*)myMalloc(1 * sizeof(float));;
x5757[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5755,x721,1,x5757, x1249, 1, x721,1));
arrayFill_greg<<<28, 512>>>(x1249, 0.0f, 64);
float* x5761 = (float*)myMalloc(1 * sizeof(float));;
x5761[0] = 1.0f;
float* x5763 = (float*)myMalloc(1 * sizeof(float));;
x5763[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5761,x724,1,x5763, x1250, 1, x724,1));
arrayFill_greg<<<28, 512>>>(x1250, 0.0f, 1024);
float* x5767 = (float*)myMalloc(1 * sizeof(float));;
x5767[0] = 1.0f;
float* x5769 = (float*)myMalloc(1 * sizeof(float));;
x5769[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5767,x727,1,x5769, x1251, 1, x727,1));
arrayFill_greg<<<28, 512>>>(x1251, 0.0f, 2048);
float* x5773 = (float*)myMalloc(1 * sizeof(float));;
x5773[0] = 1.0f;
float* x5775 = (float*)myMalloc(1 * sizeof(float));;
x5775[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5773,x730,1,x5775, x1252, 1, x730,1));
arrayFill_greg<<<28, 512>>>(x1252, 0.0f, 512);
float* x5779 = (float*)myMalloc(1 * sizeof(float));;
x5779[0] = 1.0f;
float* x5781 = (float*)myMalloc(1 * sizeof(float));;
x5781[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5779,x733,1,x5781, x1253, 1, x733,1));
arrayFill_greg<<<28, 512>>>(x1253, 0.0f, 1024);
float* x5785 = (float*)myMalloc(1 * sizeof(float));;
x5785[0] = 1.0f;
float* x5787 = (float*)myMalloc(1 * sizeof(float));;
x5787[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5785,x736,1,x5787, x1254, 1, x736,1));
arrayFill_greg<<<28, 512>>>(x1254, 0.0f, 512);
float* x5791 = (float*)myMalloc(1 * sizeof(float));;
x5791[0] = 1.0f;
float* x5793 = (float*)myMalloc(1 * sizeof(float));;
x5793[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5791,x739,1,x5793, x1255, 1, x739,1));
arrayFill_greg<<<28, 512>>>(x1255, 0.0f, 128);
float* x5797 = (float*)myMalloc(1 * sizeof(float));;
x5797[0] = 1.0f;
float* x5799 = (float*)myMalloc(1 * sizeof(float));;
x5799[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5797,x742,1,x5799, x1256, 1, x742,1));
arrayFill_greg<<<28, 512>>>(x1256, 0.0f, 512);
float* x5803 = (float*)myMalloc(1 * sizeof(float));;
x5803[0] = 1.0f;
float* x5805 = (float*)myMalloc(1 * sizeof(float));;
x5805[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,64,x5803,x745,256,x5805, x1257, 256, x745,256));
arrayFill_greg<<<28, 512>>>(x1257, 0.0f, 16384);
float* x5809 = (float*)myMalloc(1 * sizeof(float));;
x5809[0] = 1.0f;
float* x5811 = (float*)myMalloc(1 * sizeof(float));;
x5811[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,256,x5809,x748,1024,x5811, x1258, 1024, x748,1024));
arrayFill_greg<<<28, 512>>>(x1258, 0.0f, 262144);
float* x5815 = (float*)myMalloc(1 * sizeof(float));;
x5815[0] = 1.0f;
float* x5817 = (float*)myMalloc(1 * sizeof(float));;
x5817[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 27,64,x5815,x751,27,x5817, x1259, 27, x751,27));
arrayFill_greg<<<28, 512>>>(x1259, 0.0f, 1728);
float* x5821 = (float*)myMalloc(1 * sizeof(float));;
x5821[0] = 1.0f;
float* x5823 = (float*)myMalloc(1 * sizeof(float));;
x5823[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5821,x754,1,x5823, x1260, 1, x754,1));
arrayFill_greg<<<28, 512>>>(x1260, 0.0f, 64);
float* x5827 = (float*)myMalloc(1 * sizeof(float));;
x5827[0] = 1.0f;
float* x5829 = (float*)myMalloc(1 * sizeof(float));;
x5829[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5827,x757,1,x5829, x1261, 1, x757,1));
arrayFill_greg<<<28, 512>>>(x1261, 0.0f, 512);
float* x5833 = (float*)myMalloc(1 * sizeof(float));;
x5833[0] = 1.0f;
float* x5835 = (float*)myMalloc(1 * sizeof(float));;
x5835[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 4608,512,x5833,x760,4608,x5835, x1262, 4608, x760,4608));
arrayFill_greg<<<28, 512>>>(x1262, 0.0f, 2359296);
float* x5839 = (float*)myMalloc(1 * sizeof(float));;
x5839[0] = 1.0f;
float* x5841 = (float*)myMalloc(1 * sizeof(float));;
x5841[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5839,x763,1,x5841, x1263, 1, x763,1));
arrayFill_greg<<<28, 512>>>(x1263, 0.0f, 512);
float* x5845 = (float*)myMalloc(1 * sizeof(float));;
x5845[0] = 1.0f;
float* x5847 = (float*)myMalloc(1 * sizeof(float));;
x5847[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5845,x766,1,x5847, x1264, 1, x766,1));
arrayFill_greg<<<28, 512>>>(x1264, 0.0f, 256);
float* x5851 = (float*)myMalloc(1 * sizeof(float));;
x5851[0] = 1.0f;
float* x5853 = (float*)myMalloc(1 * sizeof(float));;
x5853[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5851,x769,1,x5853, x1265, 1, x769,1));
arrayFill_greg<<<28, 512>>>(x1265, 0.0f, 64);
float* x5857 = (float*)myMalloc(1 * sizeof(float));;
x5857[0] = 1.0f;
float* x5859 = (float*)myMalloc(1 * sizeof(float));;
x5859[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5857,x772,1,x5859, x1266, 1, x772,1));
arrayFill_greg<<<28, 512>>>(x1266, 0.0f, 512);
float* x5863 = (float*)myMalloc(1 * sizeof(float));;
x5863[0] = 1.0f;
float* x5865 = (float*)myMalloc(1 * sizeof(float));;
x5865[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5863,x775,1,x5865, x1267, 1, x775,1));
arrayFill_greg<<<28, 512>>>(x1267, 0.0f, 512);
float* x5869 = (float*)myMalloc(1 * sizeof(float));;
x5869[0] = 1.0f;
float* x5871 = (float*)myMalloc(1 * sizeof(float));;
x5871[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5869,x778,1,x5871, x1268, 1, x778,1));
arrayFill_greg<<<28, 512>>>(x1268, 0.0f, 1024);
float* x5875 = (float*)myMalloc(1 * sizeof(float));;
x5875[0] = 1.0f;
float* x5877 = (float*)myMalloc(1 * sizeof(float));;
x5877[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x5875,x781,64,x5877, x1269, 64, x781,64));
arrayFill_greg<<<28, 512>>>(x1269, 0.0f, 16384);
float* x5881 = (float*)myMalloc(1 * sizeof(float));;
x5881[0] = 1.0f;
float* x5883 = (float*)myMalloc(1 * sizeof(float));;
x5883[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5881,x784,1,x5883, x1270, 1, x784,1));
arrayFill_greg<<<28, 512>>>(x1270, 0.0f, 256);
float* x5887 = (float*)myMalloc(1 * sizeof(float));;
x5887[0] = 1.0f;
float* x5889 = (float*)myMalloc(1 * sizeof(float));;
x5889[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5887,x787,1,x5889, x1271, 1, x787,1));
arrayFill_greg<<<28, 512>>>(x1271, 0.0f, 64);
float* x5893 = (float*)myMalloc(1 * sizeof(float));;
x5893[0] = 1.0f;
float* x5895 = (float*)myMalloc(1 * sizeof(float));;
x5895[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1152,128,x5893,x790,1152,x5895, x1272, 1152, x790,1152));
arrayFill_greg<<<28, 512>>>(x1272, 0.0f, 147456);
float* x5899 = (float*)myMalloc(1 * sizeof(float));;
x5899[0] = 1.0f;
float* x5901 = (float*)myMalloc(1 * sizeof(float));;
x5901[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5899,x793,1,x5901, x1273, 1, x793,1));
arrayFill_greg<<<28, 512>>>(x1273, 0.0f, 256);
float* x5905 = (float*)myMalloc(1 * sizeof(float));;
x5905[0] = 1.0f;
float* x5907 = (float*)myMalloc(1 * sizeof(float));;
x5907[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5905,x796,1,x5907, x1274, 1, x796,1));
arrayFill_greg<<<28, 512>>>(x1274, 0.0f, 512);
float* x5911 = (float*)myMalloc(1 * sizeof(float));;
x5911[0] = 1.0f;
float* x5913 = (float*)myMalloc(1 * sizeof(float));;
x5913[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5911,x799,1,x5913, x1275, 1, x799,1));
arrayFill_greg<<<28, 512>>>(x1275, 0.0f, 256);
float* x5917 = (float*)myMalloc(1 * sizeof(float));;
x5917[0] = 1.0f;
float* x5919 = (float*)myMalloc(1 * sizeof(float));;
x5919[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5917,x802,1,x5919, x1276, 1, x802,1));
arrayFill_greg<<<28, 512>>>(x1276, 0.0f, 512);
float* x5923 = (float*)myMalloc(1 * sizeof(float));;
x5923[0] = 1.0f;
float* x5925 = (float*)myMalloc(1 * sizeof(float));;
x5925[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5923,x805,1,x5925, x1277, 1, x805,1));
arrayFill_greg<<<28, 512>>>(x1277, 0.0f, 128);
float* x5929 = (float*)myMalloc(1 * sizeof(float));;
x5929[0] = 1.0f;
float* x5931 = (float*)myMalloc(1 * sizeof(float));;
x5931[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,64,x5929,x808,256,x5931, x1278, 256, x808,256));
arrayFill_greg<<<28, 512>>>(x1278, 0.0f, 16384);
float* x5935 = (float*)myMalloc(1 * sizeof(float));;
x5935[0] = 1.0f;
float* x5937 = (float*)myMalloc(1 * sizeof(float));;
x5937[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5935,x811,1,x5937, x1279, 1, x811,1));
arrayFill_greg<<<28, 512>>>(x1279, 0.0f, 128);
float* x5941 = (float*)myMalloc(1 * sizeof(float));;
x5941[0] = 1.0f;
float* x5943 = (float*)myMalloc(1 * sizeof(float));;
x5943[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5941,x814,1,x5943, x1280, 1, x814,1));
arrayFill_greg<<<28, 512>>>(x1280, 0.0f, 2048);
float* x5947 = (float*)myMalloc(1 * sizeof(float));;
x5947[0] = 1.0f;
float* x5949 = (float*)myMalloc(1 * sizeof(float));;
x5949[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5947,x817,1,x5949, x1281, 1, x817,1));
arrayFill_greg<<<28, 512>>>(x1281, 0.0f, 256);
float* x5953 = (float*)myMalloc(1 * sizeof(float));;
x5953[0] = 1.0f;
float* x5955 = (float*)myMalloc(1 * sizeof(float));;
x5955[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x5953,x820,2304,x5955, x1282, 2304, x820,2304));
arrayFill_greg<<<28, 512>>>(x1282, 0.0f, 589824);
float* x5959 = (float*)myMalloc(1 * sizeof(float));;
x5959[0] = 1.0f;
float* x5961 = (float*)myMalloc(1 * sizeof(float));;
x5961[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5959,x823,1,x5961, x1283, 1, x823,1));
arrayFill_greg<<<28, 512>>>(x1283, 0.0f, 256);
float* x5965 = (float*)myMalloc(1 * sizeof(float));;
x5965[0] = 1.0f;
float* x5967 = (float*)myMalloc(1 * sizeof(float));;
x5967[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5965,x826,1,x5967, x1284, 1, x826,1));
arrayFill_greg<<<28, 512>>>(x1284, 0.0f, 128);
float* x5971 = (float*)myMalloc(1 * sizeof(float));;
x5971[0] = 1.0f;
float* x5973 = (float*)myMalloc(1 * sizeof(float));;
x5973[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5971,x829,1,x5973, x1285, 1, x829,1));
arrayFill_greg<<<28, 512>>>(x1285, 0.0f, 256);
float* x5977 = (float*)myMalloc(1 * sizeof(float));;
x5977[0] = 1.0f;
float* x5979 = (float*)myMalloc(1 * sizeof(float));;
x5979[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5977,x832,1,x5979, x1286, 1, x832,1));
arrayFill_greg<<<28, 512>>>(x1286, 0.0f, 64);
float* x5983 = (float*)myMalloc(1 * sizeof(float));;
x5983[0] = 1.0f;
float* x5985 = (float*)myMalloc(1 * sizeof(float));;
x5985[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,256,x5983,x835,512,x5985, x1287, 512, x835,512));
arrayFill_greg<<<28, 512>>>(x1287, 0.0f, 131072);
float* x5989 = (float*)myMalloc(1 * sizeof(float));;
x5989[0] = 1.0f;
float* x5991 = (float*)myMalloc(1 * sizeof(float));;
x5991[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5989,x838,1,x5991, x1288, 1, x838,1));
arrayFill_greg<<<28, 512>>>(x1288, 0.0f, 2048);
float* x5995 = (float*)myMalloc(1 * sizeof(float));;
x5995[0] = 1.0f;
float* x5997 = (float*)myMalloc(1 * sizeof(float));;
x5997[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5995,x841,1,x5997, x1289, 1, x841,1));
arrayFill_greg<<<28, 512>>>(x1289, 0.0f, 1024);
float* x6001 = (float*)myMalloc(1 * sizeof(float));;
x6001[0] = 1.0f;
float* x6003 = (float*)myMalloc(1 * sizeof(float));;
x6003[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6001,x844,1,x6003, x1290, 1, x844,1));
arrayFill_greg<<<28, 512>>>(x1290, 0.0f, 1024);
float* x6007 = (float*)myMalloc(1 * sizeof(float));;
x6007[0] = 1.0f;
float* x6009 = (float*)myMalloc(1 * sizeof(float));;
x6009[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6007,x847,1,x6009, x1291, 1, x847,1));
arrayFill_greg<<<28, 512>>>(x1291, 0.0f, 256);
float* x6013 = (float*)myMalloc(1 * sizeof(float));;
x6013[0] = 1.0f;
float* x6015 = (float*)myMalloc(1 * sizeof(float));;
x6015[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6013,x850,1,x6015, x1292, 1, x850,1));
arrayFill_greg<<<28, 512>>>(x1292, 0.0f, 256);
float* x6019 = (float*)myMalloc(1 * sizeof(float));;
x6019[0] = 1.0f;
float* x6021 = (float*)myMalloc(1 * sizeof(float));;
x6021[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6019,x853,1,x6021, x1293, 1, x853,1));
arrayFill_greg<<<28, 512>>>(x1293, 0.0f, 256);
float* x6025 = (float*)myMalloc(1 * sizeof(float));;
x6025[0] = 1.0f;
float* x6027 = (float*)myMalloc(1 * sizeof(float));;
x6027[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x6025,x856,1,x6027, x1294, 1, x856,1));
arrayFill_greg<<<28, 512>>>(x1294, 0.0f, 64);
float* x6031 = (float*)myMalloc(1 * sizeof(float));;
x6031[0] = 1.0f;
float* x6033 = (float*)myMalloc(1 * sizeof(float));;
x6033[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6031,x859,1,x6033, x1295, 1, x859,1));
arrayFill_greg<<<28, 512>>>(x1295, 0.0f, 1024);
float* x6037 = (float*)myMalloc(1 * sizeof(float));;
x6037[0] = 1.0f;
float* x6039 = (float*)myMalloc(1 * sizeof(float));;
x6039[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6037,x862,1,x6039, x1296, 1, x862,1));
arrayFill_greg<<<28, 512>>>(x1296, 0.0f, 256);
float* x6043 = (float*)myMalloc(1 * sizeof(float));;
x6043[0] = 1.0f;
float* x6045 = (float*)myMalloc(1 * sizeof(float));;
x6045[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x6043,x865,1,x6045, x1297, 1, x865,1));
arrayFill_greg<<<28, 512>>>(x1297, 0.0f, 128);
float* x6049 = (float*)myMalloc(1 * sizeof(float));;
x6049[0] = 1.0f;
float* x6051 = (float*)myMalloc(1 * sizeof(float));;
x6051[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1152,128,x6049,x868,1152,x6051, x1298, 1152, x868,1152));
arrayFill_greg<<<28, 512>>>(x1298, 0.0f, 147456);
float* x6055 = (float*)myMalloc(1 * sizeof(float));;
x6055[0] = 1.0f;
float* x6057 = (float*)myMalloc(1 * sizeof(float));;
x6057[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6055,x871,1,x6057, x1299, 1, x871,1));
arrayFill_greg<<<28, 512>>>(x1299, 0.0f, 256);
float* x6061 = (float*)myMalloc(1 * sizeof(float));;
x6061[0] = 1.0f;
float* x6063 = (float*)myMalloc(1 * sizeof(float));;
x6063[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x6061,x874,1,x6063, x1300, 1, x874,1));
arrayFill_greg<<<28, 512>>>(x1300, 0.0f, 2048);
float* x6067 = (float*)myMalloc(1 * sizeof(float));;
x6067[0] = 1.0f;
float* x6069 = (float*)myMalloc(1 * sizeof(float));;
x6069[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x6067,x877,1,x6069, x1301, 1, x877,1));
arrayFill_greg<<<28, 512>>>(x1301, 0.0f, 512);
float* x6073 = (float*)myMalloc(1 * sizeof(float));;
x6073[0] = 1.0f;
float* x6075 = (float*)myMalloc(1 * sizeof(float));;
x6075[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x6073,x880,1,x6075, x1302, 1, x880,1));
arrayFill_greg<<<28, 512>>>(x1302, 0.0f, 512);
float* x6079 = (float*)myMalloc(1 * sizeof(float));;
x6079[0] = 1.0f;
float* x6081 = (float*)myMalloc(1 * sizeof(float));;
x6081[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,128,x6079,x883,512,x6081, x1303, 512, x883,512));
arrayFill_greg<<<28, 512>>>(x1303, 0.0f, 65536);
float* x6085 = (float*)myMalloc(1 * sizeof(float));;
x6085[0] = 1.0f;
float* x6087 = (float*)myMalloc(1 * sizeof(float));;
x6087[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6085,x886,1,x6087, x1304, 1, x886,1));
arrayFill_greg<<<28, 512>>>(x1304, 0.0f, 256);
float* x6091 = (float*)myMalloc(1 * sizeof(float));;
x6091[0] = 1.0f;
float* x6093 = (float*)myMalloc(1 * sizeof(float));;
x6093[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6091,x889,1,x6093, x1305, 1, x889,1));
arrayFill_greg<<<28, 512>>>(x1305, 0.0f, 256);
float* x6097 = (float*)myMalloc(1 * sizeof(float));;
x6097[0] = 1.0f;
float* x6099 = (float*)myMalloc(1 * sizeof(float));;
x6099[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6097,x892,1,x6099, x1306, 1, x892,1));
arrayFill_greg<<<28, 512>>>(x1306, 0.0f, 256);
float* x6103 = (float*)myMalloc(1 * sizeof(float));;
x6103[0] = 1.0f;
float* x6105 = (float*)myMalloc(1 * sizeof(float));;
x6105[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6103,x895,1,x6105, x1307, 1, x895,1));
arrayFill_greg<<<28, 512>>>(x1307, 0.0f, 256);
float* x6109 = (float*)myMalloc(1 * sizeof(float));;
x6109[0] = 1.0f;
float* x6111 = (float*)myMalloc(1 * sizeof(float));;
x6111[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x6109,x898,1,x6111, x1308, 1, x898,1));
arrayFill_greg<<<28, 512>>>(x1308, 0.0f, 512);
float* x6115 = (float*)myMalloc(1 * sizeof(float));;
x6115[0] = 1.0f;
float* x6117 = (float*)myMalloc(1 * sizeof(float));;
x6117[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x6115,x901,1,x6117, x1309, 1, x901,1));
arrayFill_greg<<<28, 512>>>(x1309, 0.0f, 512);
float* x6121 = (float*)myMalloc(1 * sizeof(float));;
x6121[0] = 1.0f;
float* x6123 = (float*)myMalloc(1 * sizeof(float));;
x6123[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6121,x904,1,x6123, x1310, 1, x904,1));
arrayFill_greg<<<28, 512>>>(x1310, 0.0f, 256);
float* x6127 = (float*)myMalloc(1 * sizeof(float));;
x6127[0] = 1.0f;
float* x6129 = (float*)myMalloc(1 * sizeof(float));;
x6129[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x6127,x907,1,x6129, x1311, 1, x907,1));
arrayFill_greg<<<28, 512>>>(x1311, 0.0f, 128);
float* x6133 = (float*)myMalloc(1 * sizeof(float));;
x6133[0] = 1.0f;
float* x6135 = (float*)myMalloc(1 * sizeof(float));;
x6135[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x6133,x910,1,x6135, x1312, 1, x910,1));
arrayFill_greg<<<28, 512>>>(x1312, 0.0f, 512);
float* x6139 = (float*)myMalloc(1 * sizeof(float));;
x6139[0] = 1.0f;
float* x6141 = (float*)myMalloc(1 * sizeof(float));;
x6141[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x6139,x913,1,x6141, x1313, 1, x913,1));
arrayFill_greg<<<28, 512>>>(x1313, 0.0f, 64);
float* x6145 = (float*)myMalloc(1 * sizeof(float));;
x6145[0] = 1.0f;
float* x6147 = (float*)myMalloc(1 * sizeof(float));;
x6147[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x6145,x916,1,x6147, x1314, 1, x916,1));
arrayFill_greg<<<28, 512>>>(x1314, 0.0f, 512);
float* x6151 = (float*)myMalloc(1 * sizeof(float));;
x6151[0] = 1.0f;
float* x6153 = (float*)myMalloc(1 * sizeof(float));;
x6153[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x6151,x919,1,x6153, x1315, 1, x919,1));
arrayFill_greg<<<28, 512>>>(x1315, 0.0f, 64);
float* x6157 = (float*)myMalloc(1 * sizeof(float));;
x6157[0] = 1.0f;
float* x6159 = (float*)myMalloc(1 * sizeof(float));;
x6159[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6157,x922,1,x6159, x1316, 1, x922,1));
arrayFill_greg<<<28, 512>>>(x1316, 0.0f, 1024);
float* x6163 = (float*)myMalloc(1 * sizeof(float));;
x6163[0] = 1.0f;
float* x6165 = (float*)myMalloc(1 * sizeof(float));;
x6165[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x6163,x925,1,x6165, x1317, 1, x925,1));
arrayFill_greg<<<28, 512>>>(x1317, 0.0f, 512);
float* x6169 = (float*)myMalloc(1 * sizeof(float));;
x6169[0] = 1.0f;
float* x6171 = (float*)myMalloc(1 * sizeof(float));;
x6171[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6169,x928,1,x6171, x1318, 1, x928,1));
arrayFill_greg<<<28, 512>>>(x1318, 0.0f, 1024);
float* x6175 = (float*)myMalloc(1 * sizeof(float));;
x6175[0] = 1.0f;
float* x6177 = (float*)myMalloc(1 * sizeof(float));;
x6177[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,2048,x6175,x931,512,x6177, x1319, 512, x931,512));
arrayFill_greg<<<28, 512>>>(x1319, 0.0f, 1048576);
float* x6181 = (float*)myMalloc(1 * sizeof(float));;
x6181[0] = 1.0f;
float* x6183 = (float*)myMalloc(1 * sizeof(float));;
x6183[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x6181,x934,1,x6183, x1320, 1, x934,1));
arrayFill_greg<<<28, 512>>>(x1320, 0.0f, 512);
float* x6187 = (float*)myMalloc(1 * sizeof(float));;
x6187[0] = 1.0f;
float* x6189 = (float*)myMalloc(1 * sizeof(float));;
x6189[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,2048,x6187,x937,1024,x6189, x1321, 1024, x937,1024));
arrayFill_greg<<<28, 512>>>(x1321, 0.0f, 2097152);
float* x6193 = (float*)myMalloc(1 * sizeof(float));;
x6193[0] = 1.0f;
float* x6195 = (float*)myMalloc(1 * sizeof(float));;
x6195[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2048,512,x6193,x940,2048,x6195, x1322, 2048, x940,2048));
arrayFill_greg<<<28, 512>>>(x1322, 0.0f, 1048576);
float* x6199 = (float*)myMalloc(1 * sizeof(float));;
x6199[0] = 1.0f;
float* x6201 = (float*)myMalloc(1 * sizeof(float));;
x6201[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6199,x943,1,x6201, x1323, 1, x943,1));
arrayFill_greg<<<28, 512>>>(x1323, 0.0f, 1024);
float* x6205 = (float*)myMalloc(1 * sizeof(float));;
x6205[0] = 1.0f;
float* x6207 = (float*)myMalloc(1 * sizeof(float));;
x6207[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x6205,x946,1,x6207, x1324, 1, x946,1));
arrayFill_greg<<<28, 512>>>(x1324, 0.0f, 128);
float* x6211 = (float*)myMalloc(1 * sizeof(float));;
x6211[0] = 1.0f;
float* x6213 = (float*)myMalloc(1 * sizeof(float));;
x6213[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,256,x6211,x949,1024,x6213, x1325, 1024, x949,1024));
arrayFill_greg<<<28, 512>>>(x1325, 0.0f, 262144);
float* x6217 = (float*)myMalloc(1 * sizeof(float));;
x6217[0] = 1.0f;
float* x6219 = (float*)myMalloc(1 * sizeof(float));;
x6219[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6217,x952,1,x6219, x1326, 1, x952,1));
arrayFill_greg<<<28, 512>>>(x1326, 0.0f, 256);
float* x6223 = (float*)myMalloc(1 * sizeof(float));;
x6223[0] = 1.0f;
float* x6225 = (float*)myMalloc(1 * sizeof(float));;
x6225[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6223,x955,1,x6225, x1327, 1, x955,1));
arrayFill_greg<<<28, 512>>>(x1327, 0.0f, 1024);
float* x6229 = (float*)myMalloc(1 * sizeof(float));;
x6229[0] = 1.0f;
float* x6231 = (float*)myMalloc(1 * sizeof(float));;
x6231[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x6229,x958,256,x6231, x1328, 256, x958,256));
arrayFill_greg<<<28, 512>>>(x1328, 0.0f, 262144);
float* x6235 = (float*)myMalloc(1 * sizeof(float));;
x6235[0] = 1.0f;
float* x6237 = (float*)myMalloc(1 * sizeof(float));;
x6237[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x6235,x961,1,x6237, x1329, 1, x961,1));
arrayFill_greg<<<28, 512>>>(x1329, 0.0f, 128);
float* x6241 = (float*)myMalloc(1 * sizeof(float));;
x6241[0] = 1.0f;
float* x6243 = (float*)myMalloc(1 * sizeof(float));;
x6243[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x6241,x964,1,x6243, x1330, 1, x964,1));
arrayFill_greg<<<28, 512>>>(x1330, 0.0f, 512);
float* x6247 = (float*)myMalloc(1 * sizeof(float));;
x6247[0] = 1.0f;
float* x6249 = (float*)myMalloc(1 * sizeof(float));;
x6249[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x6247,x967,1,x6249, x1331, 1, x967,1));
arrayFill_greg<<<28, 512>>>(x1331, 0.0f, 512);
float* x6253 = (float*)myMalloc(1 * sizeof(float));;
x6253[0] = 1.0f;
float* x6255 = (float*)myMalloc(1 * sizeof(float));;
x6255[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x6253,x970,1,x6255, x1332, 1, x970,1));
arrayFill_greg<<<28, 512>>>(x1332, 0.0f, 128);
float* x6259 = (float*)myMalloc(1 * sizeof(float));;
x6259[0] = 1.0f;
float* x6261 = (float*)myMalloc(1 * sizeof(float));;
x6261[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x6259,x973,2304,x6261, x1333, 2304, x973,2304));
arrayFill_greg<<<28, 512>>>(x1333, 0.0f, 589824);
float* x6265 = (float*)myMalloc(1 * sizeof(float));;
x6265[0] = 1.0f;
float* x6267 = (float*)myMalloc(1 * sizeof(float));;
x6267[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2048,10,x6265,x976,2048,x6267, x1334, 2048, x976,2048));
arrayFill_greg<<<28, 512>>>(x1334, 0.0f, 20480);
float* x6271 = (float*)myMalloc(1 * sizeof(float));;
x6271[0] = 1.0f;
float* x6273 = (float*)myMalloc(1 * sizeof(float));;
x6273[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6271,x979,1,x6273, x1335, 1, x979,1));
arrayFill_greg<<<28, 512>>>(x1335, 0.0f, 256);
float* x6277 = (float*)myMalloc(1 * sizeof(float));;
x6277[0] = 1.0f;
float* x6279 = (float*)myMalloc(1 * sizeof(float));;
x6279[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6277,x982,1,x6279, x1336, 1, x982,1));
arrayFill_greg<<<28, 512>>>(x1336, 0.0f, 256);
float* x6283 = (float*)myMalloc(1 * sizeof(float));;
x6283[0] = 1.0f;
float* x6285 = (float*)myMalloc(1 * sizeof(float));;
x6285[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6283,x985,1,x6285, x1337, 1, x985,1));
arrayFill_greg<<<28, 512>>>(x1337, 0.0f, 256);
float* x6289 = (float*)myMalloc(1 * sizeof(float));;
x6289[0] = 1.0f;
float* x6291 = (float*)myMalloc(1 * sizeof(float));;
x6291[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6289,x988,1,x6291, x1338, 1, x988,1));
arrayFill_greg<<<28, 512>>>(x1338, 0.0f, 1024);
float* x6295 = (float*)myMalloc(1 * sizeof(float));;
x6295[0] = 1.0f;
float* x6297 = (float*)myMalloc(1 * sizeof(float));;
x6297[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6295,x991,1,x6297, x1339, 1, x991,1));
arrayFill_greg<<<28, 512>>>(x1339, 0.0f, 1024);
float* x6301 = (float*)myMalloc(1 * sizeof(float));;
x6301[0] = 1.0f;
float* x6303 = (float*)myMalloc(1 * sizeof(float));;
x6303[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,64,x6301,x994,64,x6303, x1340, 64, x994,64));
arrayFill_greg<<<28, 512>>>(x1340, 0.0f, 4096);
float* x6307 = (float*)myMalloc(1 * sizeof(float));;
x6307[0] = 1.0f;
float* x6309 = (float*)myMalloc(1 * sizeof(float));;
x6309[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x6307,x997,1,x6309, x1341, 1, x997,1));
arrayFill_greg<<<28, 512>>>(x1341, 0.0f, 512);
float* x6313 = (float*)myMalloc(1 * sizeof(float));;
x6313[0] = 1.0f;
float* x6315 = (float*)myMalloc(1 * sizeof(float));;
x6315[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1152,128,x6313,x1000,1152,x6315, x1342, 1152, x1000,1152));
arrayFill_greg<<<28, 512>>>(x1342, 0.0f, 147456);
float* x6319 = (float*)myMalloc(1 * sizeof(float));;
x6319[0] = 1.0f;
float* x6321 = (float*)myMalloc(1 * sizeof(float));;
x6321[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x6319,x1003,1,x6321, x1343, 1, x1003,1));
arrayFill_greg<<<28, 512>>>(x1343, 0.0f, 128);
float* x6325 = (float*)myMalloc(1 * sizeof(float));;
x6325[0] = 1.0f;
float* x6327 = (float*)myMalloc(1 * sizeof(float));;
x6327[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6325,x1006,1,x6327, x1344, 1, x1006,1));
arrayFill_greg<<<28, 512>>>(x1344, 0.0f, 256);
float* x6331 = (float*)myMalloc(1 * sizeof(float));;
x6331[0] = 1.0f;
float* x6333 = (float*)myMalloc(1 * sizeof(float));;
x6333[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6331,x1009,1,x6333, x1345, 1, x1009,1));
arrayFill_greg<<<28, 512>>>(x1345, 0.0f, 1024);
float* x6337 = (float*)myMalloc(1 * sizeof(float));;
x6337[0] = 1.0f;
float* x6339 = (float*)myMalloc(1 * sizeof(float));;
x6339[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x6337,x1012,1,x6339, x1346, 1, x1012,1));
arrayFill_greg<<<28, 512>>>(x1346, 0.0f, 2048);
float* x6343 = (float*)myMalloc(1 * sizeof(float));;
x6343[0] = 1.0f;
float* x6345 = (float*)myMalloc(1 * sizeof(float));;
x6345[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6343,x1015,1,x6345, x1347, 1, x1015,1));
arrayFill_greg<<<28, 512>>>(x1347, 0.0f, 256);
float* x6349 = (float*)myMalloc(1 * sizeof(float));;
x6349[0] = 1.0f;
float* x6351 = (float*)myMalloc(1 * sizeof(float));;
x6351[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6349,x1018,1,x6351, x1348, 1, x1018,1));
arrayFill_greg<<<28, 512>>>(x1348, 0.0f, 256);
float* x6355 = (float*)myMalloc(1 * sizeof(float));;
x6355[0] = 1.0f;
float* x6357 = (float*)myMalloc(1 * sizeof(float));;
x6357[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x6355,x1021,1,x6357, x1349, 1, x1021,1));
arrayFill_greg<<<28, 512>>>(x1349, 0.0f, 128);
float* x6361 = (float*)myMalloc(1 * sizeof(float));;
x6361[0] = 1.0f;
float* x6363 = (float*)myMalloc(1 * sizeof(float));;
x6363[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6361,x1024,1,x6363, x1350, 1, x1024,1));
arrayFill_greg<<<28, 512>>>(x1350, 0.0f, 256);
float* x6367 = (float*)myMalloc(1 * sizeof(float));;
x6367[0] = 1.0f;
float* x6369 = (float*)myMalloc(1 * sizeof(float));;
x6369[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x6367,x1027,1,x6369, x1351, 1, x1027,1));
arrayFill_greg<<<28, 512>>>(x1351, 0.0f, 64);
float* x6373 = (float*)myMalloc(1 * sizeof(float));;
x6373[0] = 1.0f;
float* x6375 = (float*)myMalloc(1 * sizeof(float));;
x6375[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x6373,x1030,1,x6375, x1352, 1, x1030,1));
arrayFill_greg<<<28, 512>>>(x1352, 0.0f, 2048);
float* x6379 = (float*)myMalloc(1 * sizeof(float));;
x6379[0] = 1.0f;
float* x6381 = (float*)myMalloc(1 * sizeof(float));;
x6381[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x6379,x1033,1,x6381, x1353, 1, x1033,1));
arrayFill_greg<<<28, 512>>>(x1353, 0.0f, 512);
float* x6385 = (float*)myMalloc(1 * sizeof(float));;
x6385[0] = 1.0f;
float* x6387 = (float*)myMalloc(1 * sizeof(float));;
x6387[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6385,x1036,1,x6387, x1354, 1, x1036,1));
arrayFill_greg<<<28, 512>>>(x1354, 0.0f, 256);
float* x6391 = (float*)myMalloc(1 * sizeof(float));;
x6391[0] = 1.0f;
float* x6393 = (float*)myMalloc(1 * sizeof(float));;
x6393[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6391,x1039,1,x6393, x1355, 1, x1039,1));
arrayFill_greg<<<28, 512>>>(x1355, 0.0f, 1024);
float* x6397 = (float*)myMalloc(1 * sizeof(float));;
x6397[0] = 1.0f;
float* x6399 = (float*)myMalloc(1 * sizeof(float));;
x6399[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x6397,x1042,2304,x6399, x1356, 2304, x1042,2304));
arrayFill_greg<<<28, 512>>>(x1356, 0.0f, 589824);
float* x6403 = (float*)myMalloc(1 * sizeof(float));;
x6403[0] = 1.0f;
float* x6405 = (float*)myMalloc(1 * sizeof(float));;
x6405[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6403,x1045,1,x6405, x1357, 1, x1045,1));
arrayFill_greg<<<28, 512>>>(x1357, 0.0f, 256);
float* x6409 = (float*)myMalloc(1 * sizeof(float));;
x6409[0] = 1.0f;
float* x6411 = (float*)myMalloc(1 * sizeof(float));;
x6411[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x6409,x1048,1,x6411, x1358, 1, x1048,1));
arrayFill_greg<<<28, 512>>>(x1358, 0.0f, 64);
float* x6415 = (float*)myMalloc(1 * sizeof(float));;
x6415[0] = 1.0f;
float* x6417 = (float*)myMalloc(1 * sizeof(float));;
x6417[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x6415,x1051,1,x6417, x1359, 1, x1051,1));
arrayFill_greg<<<28, 512>>>(x1359, 0.0f, 128);
float* x6421 = (float*)myMalloc(1 * sizeof(float));;
x6421[0] = 1.0f;
float* x6423 = (float*)myMalloc(1 * sizeof(float));;
x6423[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6421,x1054,1,x6423, x1360, 1, x1054,1));
arrayFill_greg<<<28, 512>>>(x1360, 0.0f, 256);
float* x6427 = (float*)myMalloc(1 * sizeof(float));;
x6427[0] = 1.0f;
float* x6429 = (float*)myMalloc(1 * sizeof(float));;
x6429[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6427,x1057,1,x6429, x1361, 1, x1057,1));
arrayFill_greg<<<28, 512>>>(x1361, 0.0f, 256);
float* x6433 = (float*)myMalloc(1 * sizeof(float));;
x6433[0] = 1.0f;
float* x6435 = (float*)myMalloc(1 * sizeof(float));;
x6435[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x6433,x1060,1,x6435, x1362, 1, x1060,1));
arrayFill_greg<<<28, 512>>>(x1362, 0.0f, 512);
float* x6439 = (float*)myMalloc(1 * sizeof(float));;
x6439[0] = 1.0f;
float* x6441 = (float*)myMalloc(1 * sizeof(float));;
x6441[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,128,x6439,x1063,512,x6441, x1363, 512, x1063,512));
arrayFill_greg<<<28, 512>>>(x1363, 0.0f, 65536);
float* x6445 = (float*)myMalloc(1 * sizeof(float));;
x6445[0] = 1.0f;
float* x6447 = (float*)myMalloc(1 * sizeof(float));;
x6447[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x6445,x1066,1,x6447, x1364, 1, x1066,1));
arrayFill_greg<<<28, 512>>>(x1364, 0.0f, 64);
float* x6451 = (float*)myMalloc(1 * sizeof(float));;
x6451[0] = 1.0f;
float* x6453 = (float*)myMalloc(1 * sizeof(float));;
x6453[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,512,x6451,x1069,256,x6453, x1365, 256, x1069,256));
arrayFill_greg<<<28, 512>>>(x1365, 0.0f, 131072);
float* x6457 = (float*)myMalloc(1 * sizeof(float));;
x6457[0] = 1.0f;
float* x6459 = (float*)myMalloc(1 * sizeof(float));;
x6459[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6457,x1072,1,x6459, x1366, 1, x1072,1));
arrayFill_greg<<<28, 512>>>(x1366, 0.0f, 256);
float* x6463 = (float*)myMalloc(1 * sizeof(float));;
x6463[0] = 1.0f;
float* x6465 = (float*)myMalloc(1 * sizeof(float));;
x6465[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x6463,x1075,1,x6465, x1367, 1, x1075,1));
arrayFill_greg<<<28, 512>>>(x1367, 0.0f, 2048);
float* x6469 = (float*)myMalloc(1 * sizeof(float));;
x6469[0] = 1.0f;
float* x6471 = (float*)myMalloc(1 * sizeof(float));;
x6471[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x6469,x1078,1,x6471, x1368, 1, x1078,1));
arrayFill_greg<<<28, 512>>>(x1368, 0.0f, 128);
float* x6475 = (float*)myMalloc(1 * sizeof(float));;
x6475[0] = 1.0f;
float* x6477 = (float*)myMalloc(1 * sizeof(float));;
x6477[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x6475,x1081,2304,x6477, x1369, 2304, x1081,2304));
arrayFill_greg<<<28, 512>>>(x1369, 0.0f, 589824);
float* x6481 = (float*)myMalloc(1 * sizeof(float));;
x6481[0] = 1.0f;
float* x6483 = (float*)myMalloc(1 * sizeof(float));;
x6483[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6481,x1084,1,x6483, x1370, 1, x1084,1));
arrayFill_greg<<<28, 512>>>(x1370, 0.0f, 1024);
float* x6487 = (float*)myMalloc(1 * sizeof(float));;
x6487[0] = 1.0f;
float* x6489 = (float*)myMalloc(1 * sizeof(float));;
x6489[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6487,x1087,1,x6489, x1371, 1, x1087,1));
arrayFill_greg<<<28, 512>>>(x1371, 0.0f, 256);
float* x6493 = (float*)myMalloc(1 * sizeof(float));;
x6493[0] = 1.0f;
float* x6495 = (float*)myMalloc(1 * sizeof(float));;
x6495[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2048,512,x6493,x1090,2048,x6495, x1372, 2048, x1090,2048));
arrayFill_greg<<<28, 512>>>(x1372, 0.0f, 1048576);
float* x6499 = (float*)myMalloc(1 * sizeof(float));;
x6499[0] = 1.0f;
float* x6501 = (float*)myMalloc(1 * sizeof(float));;
x6501[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x6499,x1093,1,x6501, x1373, 1, x1093,1));
arrayFill_greg<<<28, 512>>>(x1373, 0.0f, 128);
float* x6505 = (float*)myMalloc(1 * sizeof(float));;
x6505[0] = 1.0f;
float* x6507 = (float*)myMalloc(1 * sizeof(float));;
x6507[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6505,x1096,1,x6507, x1374, 1, x1096,1));
arrayFill_greg<<<28, 512>>>(x1374, 0.0f, 1024);
float* x6511 = (float*)myMalloc(1 * sizeof(float));;
x6511[0] = 1.0f;
float* x6513 = (float*)myMalloc(1 * sizeof(float));;
x6513[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x6511,x1099,1,x6513, x1375, 1, x1099,1));
arrayFill_greg<<<28, 512>>>(x1375, 0.0f, 128);
float* x6517 = (float*)myMalloc(1 * sizeof(float));;
x6517[0] = 1.0f;
float* x6519 = (float*)myMalloc(1 * sizeof(float));;
x6519[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x6517,x1102,256,x6519, x1376, 256, x1102,256));
arrayFill_greg<<<28, 512>>>(x1376, 0.0f, 262144);
float* x6523 = (float*)myMalloc(1 * sizeof(float));;
x6523[0] = 1.0f;
float* x6525 = (float*)myMalloc(1 * sizeof(float));;
x6525[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6523,x1105,1,x6525, x1377, 1, x1105,1));
arrayFill_greg<<<28, 512>>>(x1377, 0.0f, 256);
float* x6529 = (float*)myMalloc(1 * sizeof(float));;
x6529[0] = 1.0f;
float* x6531 = (float*)myMalloc(1 * sizeof(float));;
x6531[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6529,x1108,1,x6531, x1378, 1, x1108,1));
arrayFill_greg<<<28, 512>>>(x1378, 0.0f, 256);
float* x6535 = (float*)myMalloc(1 * sizeof(float));;
x6535[0] = 1.0f;
float* x6537 = (float*)myMalloc(1 * sizeof(float));;
x6537[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6535,x1111,1,x6537, x1379, 1, x1111,1));
arrayFill_greg<<<28, 512>>>(x1379, 0.0f, 1024);
int32_t x6541 = x1396 + 1;
int32_t x6543 = x6541 % x6542;
bool x6544 = x6543 == 0;
if (x6544) {
float x6549 = x1390;
double x6545 = (double)x1397;
double x6546 = 100.0 * x6545;
double x6548 = x6546 / x6547;
float x6550 = (float)x1396;
float x6551 = x6549 / x6550;
printf("Train epoch %d: [%d/%d (%.0f%%)] Average Loss: %.6f\n",x1386,x1397,x11,x6548,x6551);
fflush(stdout);
} else {
}
int64_t x6556 = (long)mallocAddr;
int64_t x6557 = x6556 - x1382;
memset((void*)x1382, 0, x6557);
mallocAddr = (void*)x1382;
int64_t x6560 = (long)gpuMallocAddr;
int64_t x6561 = x6560 - x1383;
cudaMemset((void*)x1383, 0, x6561);
gpuMallocAddr = (void*)x1383;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x6568 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
double x6569 = (double)x6568;
double x6570 = x6569 / 1000000.0;
x1381[x1386] = x6570;
int64_t x6572 = x6568 / 1000LL;
int64_t x6574 = x6568 / x6573;
printf("Training completed in %ldms (%ld us/images)\n",x6572,x6574);
float x6576 = x1390;
float x6578 = x6576 / x6577;
double x6579 = (double)x6578;
x1380[x1386] = x6579;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x6585 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
sort(x1381, x1381 + 4);
double x6591 = x1381[2];
int64_t x6592 = (long)fopen(x0, "w");
fprintf((FILE *)x6592, "unit: %s\n", "1 epoch");
for(int x6594=0; x6594 < 4; x6594++) {
double x6595 = x1380[x6594];
fprintf((FILE *)x6592, "%lf\n", x6595);

}
fprintf((FILE *)x6592, "run time: %lf %lf\n", x39, x6591);
fclose((FILE*)x6592);
// Backend cleanup.
CUBLAS_CALL(cublasDestroy(cublasHandle));
CUDA_CALL(cudaFree(gpuMallocBase));
      
CUDNN_CALL(cudnnDestroy(cudnnHandle));
}
/*****************************************
  End of C Generated Code                  
*******************************************/


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

int fsize(int fd) {
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

long HEAP_SIZE = 4294967304; // this is for GPU

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

__global__ void arrayFill(float *data, float value) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  data[tid] = value;
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

__global__ void concat2D_1D_loop(float* in1, float* in2, float* out, int sizeLow, int sizeHigh, int sizeDim1, int sizeDim2) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= sizeLow) return;
  if (blockIdx.y < sizeHigh) { // the first input
    int index_out = tid + blockIdx.y * sizeLow * (sizeDim1 + sizeDim2);
    int index_in1 = tid + blockIdx.y * sizeLow * sizeDim1;
    for (int i = 0; i < sizeDim1; i++) {
      out[index_out] = in1[index_in1];
      index_out += sizeLow; index_in1 += sizeLow;
    }
  } else { // the second input
    int index_out = tid + (blockIdx.y - sizeHigh) * sizeLow * (sizeDim1 + sizeDim2) + sizeLow * sizeDim1;
    int index_in2 = tid + (blockIdx.y - sizeHigh) * sizeLow * sizeDim2;
    for (int i = 0; i < sizeDim2; i++) {
      out[index_out] = in2[index_in2];
      index_out += sizeLow; index_in2 += sizeLow;
    }
  }
}

__global__ void concat2D_1D_loop_grad(float* in1, float* in2, float* out, int sizeLow, int sizeHigh, int sizeDim1, int sizeDim2) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= sizeLow) return;
  if (blockIdx.y < sizeHigh) { // the first input
    int index_out = tid + blockIdx.y * sizeLow * (sizeDim1 + sizeDim2);
    int index_in1 = tid + blockIdx.y * sizeLow * sizeDim1;
    for (int i = 0; i < sizeDim1; i++) {
      in1[index_in1] += out[index_out];
      index_out += sizeLow; index_in1 += sizeLow;
    }
  } else { // the second input
    int index_out = tid + (blockIdx.y - sizeHigh) * sizeLow * (sizeDim1 + sizeDim2) + sizeLow * sizeDim1;
    int index_in2 = tid + (blockIdx.y - sizeHigh) * sizeLow * sizeDim2;
    for (int i = 0; i < sizeDim2; i++) {
      in2[index_in2] += out[index_out];
      index_out += sizeLow; index_in2 += sizeLow;
    }
  }
}

__global__ void concat2D_1D(float* in1, float* in2, float* out, int dim2, int bound) {
  int tid = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.x < bound * dim2) {
    int subid = blockIdx.y * bound * dim2 * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
    out[tid] = in1[subid];
  } else {
    int subid = blockIdx.y * (gridDim.x - bound * dim2) * blockDim.x + (blockIdx.x - bound * dim2) * blockDim.x + threadIdx.x;
    out[tid] = in2[subid];
  }
}

__global__ void concat2D_1D_grad(float* in1, float* in2, float* out, int dim2, int bound) {
  int tid = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.x < bound * dim2) {
    int subid = blockIdx.y * bound * dim2 * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
    in1[subid] += out[tid];
  } else {
    int subid = blockIdx.y * (gridDim.x - bound * dim2) * blockDim.x + (blockIdx.x - bound * dim2) * blockDim.x + threadIdx.x;
    in2[subid] += out[tid];
  }
}

__global__ void adagrad_update_1D_1D(float* x, float* d, float* m, float clip, float lr, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    if (d[tid] > clip) d[tid] = clip;
    if (d[tid] < -clip) d[tid] = -clip;
    m[tid] += d[tid] * d[tid];
    x[tid] -= lr * d[tid] / sqrt(m[tid] + 0.00000001);
    d[tid] = 0;
  }
}

__global__ void elementwise_1D_1D_mul(float* in1, float* in2, float* out, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) out[tid] = in1[tid] * in2[tid];
}

__global__ void elementwise_1D_1D_mul_mutate(float* in1, float* in2, float* out, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) out[tid] += in1[tid] * in2[tid];
}

__global__ void elementwise_1D_1D_add(float* in1, float* in2, float* out, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) out[tid] = in1[tid] + in2[tid];
}

__global__ void elementwise_1D_1D_minus(float* in1, float* in2, float* out, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) out[tid] = in1[tid] - in2[tid];
}

__global__ void elementwise_1D_1D_div(float* in1, float* in2, float* out, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) out[tid] = in1[tid] / in2[tid];
}

__global__ void elementwise_1D_1D_exp(float* in, float* out, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) out[tid] = exp(in[tid]);
}
__global__ void elementwise_1D_1D_log(float* in, float* out, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) out[tid] = log(in[tid]);
}
__global__ void elementwise_1D_1D_sqrt(float* in, float* out, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) out[tid] = sqrt(in[tid]);
}

__global__ void elementwise_1D_1D_square(float* in, float* out, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) out[tid] = in[tid] * in[tid];
}

__global__ void elementwise_1D_1D_exp_grad(float* in_x, float* in_d, float* out_x, float * out_d, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) in_d[tid] += out_d[tid] * out_x[tid];
}
__global__ void elementwise_1D_1D_log_grad(float* in_x, float* in_d, float* out_x, float * out_d, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) in_d[tid] += out_d[tid] / in_x[tid];
}
__global__ void elementwise_1D_1D_sqrt_grad(float* in_x, float* in_d, float* out_x, float * out_d, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) in_d[tid] += out_d[tid] / out_x[tid] / 2;
}

__global__ void elementwise_1D_1D_square_grad(float* in_x, float* in_d, float* out_x, float * out_d, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) in_d[tid] += out_d[tid] * 2 * in_x[tid];
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
int32_t x8 = fsize(x7);
int64_t x10 = (int64_t)x8;
int64_t x11 = x10 / 3073LL;
int32_t x12 = (int32_t)x11;
int32_t x13 = x12 * 3072;
float* x14 = (float*)myMalloc(x13 * sizeof(float));;
int* x15 = (int32_t*)myMalloc(x12 * sizeof(int32_t));;
char* x9 = (char*)mmap(0, x8, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x7, 0);
for(int x17=0; x17 < x12; x17++) {
int32_t x18 = x17 * 3073;
char x19 = x9[x18];
int32_t x20 = (int32_t)(unsigned char)x19;
x15[x17] = x20;
int32_t x26 = x18 + 1;
int32_t x24 = x17 * 3072;
for(int x23=0; x23 < 3072; x23++) {
int32_t x27 = x26 + x23;
char x28 = x9[x27];
int32_t x25 = x24 + x23;
float x29 = (float)(unsigned char)x28;
float x30 = x29 / 255.0f;
x14[x25] = x30;

}

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x38 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
float x39 = (float)x38;
float x40 = x39 / 1000000.0f;
printf("Data normalized (all prepare time) in %lf sec\n",x40);
// Tensor 'toGPU' invocation.
float* x313 = (float*)myGpuMalloc(262144 * sizeof(float));
int32_t x42 = open("/home/fei/bitbucket/Lantern/src/out/PLDI19evaluation/resnet50/resnet50.onnx.bin",0);
int32_t x43 = fsize(x42);
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
int32_t x1394 = x12 / 64;
int32_t x5239 = x1394 / 10;
double x5244 = (double)x12;
int64_t x5270 = (int64_t)x12;
float x5274 = (float)x12;
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
float* x1399 = x14+x1398;
int* x1400 = x15+x1397;
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
float* x1411 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1412 = (float*)myMalloc(1 * sizeof(float));;
x1412[0] = 0.0f;
float* x1414 = (float*)myMalloc(1 * sizeof(float));;
x1414[0] = 1.0f;

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
    64, 64, 32, 32));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1414, in_desc, x1402, filt_desc, x751,
    conv_desc, algo, ws_data, ws_size,
    x1412, out_desc, x1411));
};
float* x1417 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1418 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1419 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1420 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1421 = (float*)myMalloc(1 * sizeof(float));;
x1421[0] = 0.0f;
float* x1423 = (float*)myMalloc(1 * sizeof(float));;
x1423[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 32, 32));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 32, 32));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1423, x1421, in_desc, x1411, out_desc, x1418, sbmv_desc, x913,
    x1048, 0.1, x415, x625, 1.0E-5,
    x1419, x1420));
};
float* x1426 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1427 = (float*)myMalloc(1 * sizeof(float));;
x1427[0] = 0.0f;
float* x1429 = (float*)myMalloc(1 * sizeof(float));;
x1429[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 32, 32));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1429, x_desc, x1418, x1427, x_desc, x1418));
};
float* x1432 = (float*)myMalloc(1 * sizeof(float));;
x1432[0] = 0.0f;
float* x1434 = (float*)myMalloc(1 * sizeof(float));;
x1434[0] = 1.0f;
float* x1436 = (float*)myGpuMalloc(1048576 * sizeof(float));

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 32, 32));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

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
    x1434, in_desc, x1418, x1432, out_desc, x1436));
};
float* x1438 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1439 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1440 = (float*)myMalloc(1 * sizeof(float));;
x1440[0] = 0.0f;
float* x1442 = (float*)myMalloc(1 * sizeof(float));;
x1442[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1442, in_desc, x1436, filt_desc, x994,
    conv_desc, algo, ws_data, ws_size,
    x1440, out_desc, x1439));
};
float* x1445 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1446 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1447 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1448 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1449 = (float*)myMalloc(1 * sizeof(float));;
x1449[0] = 0.0f;
float* x1451 = (float*)myMalloc(1 * sizeof(float));;
x1451[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1451, x1449, in_desc, x1439, out_desc, x1446, sbmv_desc, x373,
    x454, 0.1, x637, x448, 1.0E-5,
    x1447, x1448));
};
float* x1454 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1455 = (float*)myMalloc(1 * sizeof(float));;
x1455[0] = 0.0f;
float* x1457 = (float*)myMalloc(1 * sizeof(float));;
x1457[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1457, x_desc, x1446, x1455, x_desc, x1446));
};
float* x1460 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1461 = (float*)myMalloc(1 * sizeof(float));;
x1461[0] = 0.0f;
float* x1463 = (float*)myMalloc(1 * sizeof(float));;
x1463[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1463, in_desc, x1446, filt_desc, x565,
    conv_desc, algo, ws_data, ws_size,
    x1461, out_desc, x1460));
};
float* x1466 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1467 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1468 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1469 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1470 = (float*)myMalloc(1 * sizeof(float));;
x1470[0] = 0.0f;
float* x1472 = (float*)myMalloc(1 * sizeof(float));;
x1472[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1472, x1470, in_desc, x1460, out_desc, x1467, sbmv_desc, x787,
    x442, 0.1, x610, x769, 1.0E-5,
    x1468, x1469));
};
float* x1475 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1476 = (float*)myMalloc(1 * sizeof(float));;
x1476[0] = 0.0f;
float* x1478 = (float*)myMalloc(1 * sizeof(float));;
x1478[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1478, x_desc, x1467, x1476, x_desc, x1467));
};
float* x1481 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1482 = (float*)myMalloc(1 * sizeof(float));;
x1482[0] = 0.0f;
float* x1484 = (float*)myMalloc(1 * sizeof(float));;
x1484[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1484, in_desc, x1467, filt_desc, x391,
    conv_desc, algo, ws_data, ws_size,
    x1482, out_desc, x1481));
};
float* x1487 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1488 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1489 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1490 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1491 = (float*)myMalloc(1 * sizeof(float));;
x1491[0] = 0.0f;
float* x1493 = (float*)myMalloc(1 * sizeof(float));;
x1493[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1493, x1491, in_desc, x1481, out_desc, x1488, sbmv_desc, x892,
    x673, 0.1, x508, x403, 1.0E-5,
    x1489, x1490));
};
float* x1496 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1497 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1498 = (float*)myMalloc(1 * sizeof(float));;
x1498[0] = 0.0f;
float* x1500 = (float*)myMalloc(1 * sizeof(float));;
x1500[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1500, in_desc, x1436, filt_desc, x781,
    conv_desc, algo, ws_data, ws_size,
    x1498, out_desc, x1497));
};
float* x1503 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1504 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1505 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1506 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1507 = (float*)myMalloc(1 * sizeof(float));;
x1507[0] = 0.0f;
float* x1509 = (float*)myMalloc(1 * sizeof(float));;
x1509[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1509, x1507, in_desc, x1497, out_desc, x1504, sbmv_desc, x523,
    x904, 0.1, x1087, x1024, 1.0E-5,
    x1505, x1506));
};
float* x1512 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1513 = (float*)myMalloc(1 * sizeof(float));;
x1513[0] = 1.0f;
float* x1515 = (float*)myMalloc(1 * sizeof(float));;
x1515[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1513, bias_desc, x1504, x1515, out_desc, x1488));
};
float* x1518 = (float*)myMalloc(1 * sizeof(float));;
x1518[0] = 0.0f;
float* x1520 = (float*)myMalloc(1 * sizeof(float));;
x1520[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1520, x_desc, x1488, x1518, x_desc, x1488));
};
float* x1523 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1524 = (float*)myMalloc(1 * sizeof(float));;
x1524[0] = 0.0f;
float* x1526 = (float*)myMalloc(1 * sizeof(float));;
x1526[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1526, in_desc, x1488, filt_desc, x808,
    conv_desc, algo, ws_data, ws_size,
    x1524, out_desc, x1523));
};
float* x1529 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1530 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1531 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1532 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1533 = (float*)myMalloc(1 * sizeof(float));;
x1533[0] = 0.0f;
float* x1535 = (float*)myMalloc(1 * sizeof(float));;
x1535[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1535, x1533, in_desc, x1523, out_desc, x1530, sbmv_desc, x721,
    x475, 0.1, x325, x601, 1.0E-5,
    x1531, x1532));
};
float* x1538 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1539 = (float*)myMalloc(1 * sizeof(float));;
x1539[0] = 0.0f;
float* x1541 = (float*)myMalloc(1 * sizeof(float));;
x1541[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1541, x_desc, x1530, x1539, x_desc, x1530));
};
float* x1544 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1545 = (float*)myMalloc(1 * sizeof(float));;
x1545[0] = 0.0f;
float* x1547 = (float*)myMalloc(1 * sizeof(float));;
x1547[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1547, in_desc, x1530, filt_desc, x544,
    conv_desc, algo, ws_data, ws_size,
    x1545, out_desc, x1544));
};
float* x1550 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1551 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1552 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1553 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1554 = (float*)myMalloc(1 * sizeof(float));;
x1554[0] = 0.0f;
float* x1556 = (float*)myMalloc(1 * sizeof(float));;
x1556[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1556, x1554, in_desc, x1544, out_desc, x1551, sbmv_desc, x919,
    x754, 0.1, x427, x1027, 1.0E-5,
    x1552, x1553));
};
float* x1559 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1560 = (float*)myMalloc(1 * sizeof(float));;
x1560[0] = 0.0f;
float* x1562 = (float*)myMalloc(1 * sizeof(float));;
x1562[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1562, x_desc, x1551, x1560, x_desc, x1551));
};
float* x1565 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1566 = (float*)myMalloc(1 * sizeof(float));;
x1566[0] = 0.0f;
float* x1568 = (float*)myMalloc(1 * sizeof(float));;
x1568[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1568, in_desc, x1551, filt_desc, x685,
    conv_desc, algo, ws_data, ws_size,
    x1566, out_desc, x1565));
};
float* x1571 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1572 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1573 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1574 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1575 = (float*)myMalloc(1 * sizeof(float));;
x1575[0] = 0.0f;
float* x1577 = (float*)myMalloc(1 * sizeof(float));;
x1577[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1577, x1575, in_desc, x1565, out_desc, x1572, sbmv_desc, x469,
    x316, 0.1, x568, x793, 1.0E-5,
    x1573, x1574));
};
float* x1580 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1581 = (float*)myMalloc(1 * sizeof(float));;
x1581[0] = 1.0f;
float* x1583 = (float*)myMalloc(1 * sizeof(float));;
x1583[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1581, bias_desc, x1488, x1583, out_desc, x1572));
};
float* x1586 = (float*)myMalloc(1 * sizeof(float));;
x1586[0] = 0.0f;
float* x1588 = (float*)myMalloc(1 * sizeof(float));;
x1588[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1588, x_desc, x1572, x1586, x_desc, x1572));
};
float* x1591 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1592 = (float*)myMalloc(1 * sizeof(float));;
x1592[0] = 0.0f;
float* x1594 = (float*)myMalloc(1 * sizeof(float));;
x1594[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1594, in_desc, x1572, filt_desc, x745,
    conv_desc, algo, ws_data, ws_size,
    x1592, out_desc, x1591));
};
float* x1597 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1598 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1599 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1600 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1601 = (float*)myMalloc(1 * sizeof(float));;
x1601[0] = 0.0f;
float* x1603 = (float*)myMalloc(1 * sizeof(float));;
x1603[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1603, x1601, in_desc, x1591, out_desc, x1598, sbmv_desc, x538,
    x367, 0.1, x1066, x856, 1.0E-5,
    x1599, x1600));
};
float* x1606 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1607 = (float*)myMalloc(1 * sizeof(float));;
x1607[0] = 0.0f;
float* x1609 = (float*)myMalloc(1 * sizeof(float));;
x1609[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1609, x_desc, x1598, x1607, x_desc, x1598));
};
float* x1612 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1613 = (float*)myMalloc(1 * sizeof(float));;
x1613[0] = 0.0f;
float* x1615 = (float*)myMalloc(1 * sizeof(float));;
x1615[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 64, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1615, in_desc, x1598, filt_desc, x514,
    conv_desc, algo, ws_data, ws_size,
    x1613, out_desc, x1612));
};
float* x1618 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1619 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1620 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1621 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1622 = (float*)myMalloc(1 * sizeof(float));;
x1622[0] = 0.0f;
float* x1624 = (float*)myMalloc(1 * sizeof(float));;
x1624[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1624, x1622, in_desc, x1612, out_desc, x1619, sbmv_desc, x511,
    x700, 0.1, x832, x649, 1.0E-5,
    x1620, x1621));
};
float* x1627 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1628 = (float*)myMalloc(1 * sizeof(float));;
x1628[0] = 0.0f;
float* x1630 = (float*)myMalloc(1 * sizeof(float));;
x1630[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1630, x_desc, x1619, x1628, x_desc, x1619));
};
float* x1633 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1634 = (float*)myMalloc(1 * sizeof(float));;
x1634[0] = 0.0f;
float* x1636 = (float*)myMalloc(1 * sizeof(float));;
x1636[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1636, in_desc, x1619, filt_desc, x556,
    conv_desc, algo, ws_data, ws_size,
    x1634, out_desc, x1633));
};
float* x1639 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1640 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1641 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1642 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1643 = (float*)myMalloc(1 * sizeof(float));;
x1643[0] = 0.0f;
float* x1645 = (float*)myMalloc(1 * sizeof(float));;
x1645[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1645, x1643, in_desc, x1633, out_desc, x1640, sbmv_desc, x406,
    x1036, 0.1, x847, x694, 1.0E-5,
    x1641, x1642));
};
float* x1648 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1649 = (float*)myMalloc(1 * sizeof(float));;
x1649[0] = 1.0f;
float* x1651 = (float*)myMalloc(1 * sizeof(float));;
x1651[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1649, bias_desc, x1572, x1651, out_desc, x1640));
};
float* x1654 = (float*)myMalloc(1 * sizeof(float));;
x1654[0] = 0.0f;
float* x1656 = (float*)myMalloc(1 * sizeof(float));;
x1656[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1656, x_desc, x1640, x1654, x_desc, x1640));
};
float* x1659 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1660 = (float*)myMalloc(1 * sizeof(float));;
x1660[0] = 0.0f;
float* x1662 = (float*)myMalloc(1 * sizeof(float));;
x1662[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1662, in_desc, x1640, filt_desc, x328,
    conv_desc, algo, ws_data, ws_size,
    x1660, out_desc, x1659));
};
float* x1665 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1666 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1667 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1668 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1669 = (float*)myMalloc(1 * sizeof(float));;
x1669[0] = 0.0f;
float* x1671 = (float*)myMalloc(1 * sizeof(float));;
x1671[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1671, x1669, in_desc, x1659, out_desc, x1666, sbmv_desc, x547,
    x811, 0.1, x907, x697, 1.0E-5,
    x1667, x1668));
};
float* x1674 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1675 = (float*)myMalloc(1 * sizeof(float));;
x1675[0] = 0.0f;
float* x1677 = (float*)myMalloc(1 * sizeof(float));;
x1677[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1677, x_desc, x1666, x1675, x_desc, x1666));
};
float* x1680 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1681 = (float*)myMalloc(1 * sizeof(float));;
x1681[0] = 0.0f;
float* x1683 = (float*)myMalloc(1 * sizeof(float));;
x1683[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 2, 2, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1683, in_desc, x1666, filt_desc, x376,
    conv_desc, algo, ws_data, ws_size,
    x1681, out_desc, x1680));
};
float* x1686 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1687 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1688 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1689 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1690 = (float*)myMalloc(1 * sizeof(float));;
x1690[0] = 0.0f;
float* x1692 = (float*)myMalloc(1 * sizeof(float));;
x1692[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1692, x1690, in_desc, x1680, out_desc, x1687, sbmv_desc, x1051,
    x865, 0.1, x679, x424, 1.0E-5,
    x1688, x1689));
};
float* x1695 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1696 = (float*)myMalloc(1 * sizeof(float));;
x1696[0] = 0.0f;
float* x1698 = (float*)myMalloc(1 * sizeof(float));;
x1698[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1698, x_desc, x1687, x1696, x_desc, x1687));
};
float* x1701 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1702 = (float*)myMalloc(1 * sizeof(float));;
x1702[0] = 0.0f;
float* x1704 = (float*)myMalloc(1 * sizeof(float));;
x1704[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1704, in_desc, x1687, filt_desc, x613,
    conv_desc, algo, ws_data, ws_size,
    x1702, out_desc, x1701));
};
float* x1707 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1708 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1709 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1710 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1711 = (float*)myMalloc(1 * sizeof(float));;
x1711[0] = 0.0f;
float* x1713 = (float*)myMalloc(1 * sizeof(float));;
x1713[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1713, x1711, in_desc, x1701, out_desc, x1708, sbmv_desc, x730,
    x925, 0.1, x742, x598, 1.0E-5,
    x1709, x1710));
};
float* x1716 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1717 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1718 = (float*)myMalloc(1 * sizeof(float));;
x1718[0] = 0.0f;
float* x1720 = (float*)myMalloc(1 * sizeof(float));;
x1720[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 2, 2, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1720, in_desc, x1640, filt_desc, x1069,
    conv_desc, algo, ws_data, ws_size,
    x1718, out_desc, x1717));
};
float* x1723 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1724 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1725 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1726 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1727 = (float*)myMalloc(1 * sizeof(float));;
x1727[0] = 0.0f;
float* x1729 = (float*)myMalloc(1 * sizeof(float));;
x1729[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1729, x1727, in_desc, x1717, out_desc, x1724, sbmv_desc, x916,
    x652, 0.1, x421, x364, 1.0E-5,
    x1725, x1726));
};
float* x1732 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1733 = (float*)myMalloc(1 * sizeof(float));;
x1733[0] = 1.0f;
float* x1735 = (float*)myMalloc(1 * sizeof(float));;
x1735[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1733, bias_desc, x1724, x1735, out_desc, x1708));
};
float* x1738 = (float*)myMalloc(1 * sizeof(float));;
x1738[0] = 0.0f;
float* x1740 = (float*)myMalloc(1 * sizeof(float));;
x1740[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1740, x_desc, x1708, x1738, x_desc, x1708));
};
float* x1743 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1744 = (float*)myMalloc(1 * sizeof(float));;
x1744[0] = 0.0f;
float* x1746 = (float*)myMalloc(1 * sizeof(float));;
x1746[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1746, in_desc, x1708, filt_desc, x1063,
    conv_desc, algo, ws_data, ws_size,
    x1744, out_desc, x1743));
};
float* x1749 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1750 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1751 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1752 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1753 = (float*)myMalloc(1 * sizeof(float));;
x1753[0] = 0.0f;
float* x1755 = (float*)myMalloc(1 * sizeof(float));;
x1755[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1755, x1753, in_desc, x1743, out_desc, x1750, sbmv_desc, x961,
    x346, 0.1, x595, x826, 1.0E-5,
    x1751, x1752));
};
float* x1758 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1759 = (float*)myMalloc(1 * sizeof(float));;
x1759[0] = 0.0f;
float* x1761 = (float*)myMalloc(1 * sizeof(float));;
x1761[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1761, x_desc, x1750, x1759, x_desc, x1750));
};
float* x1764 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1765 = (float*)myMalloc(1 * sizeof(float));;
x1765[0] = 0.0f;
float* x1767 = (float*)myMalloc(1 * sizeof(float));;
x1767[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1767, in_desc, x1750, filt_desc, x1000,
    conv_desc, algo, ws_data, ws_size,
    x1765, out_desc, x1764));
};
float* x1770 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1771 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1772 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1773 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1774 = (float*)myMalloc(1 * sizeof(float));;
x1774[0] = 0.0f;
float* x1776 = (float*)myMalloc(1 * sizeof(float));;
x1776[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1776, x1774, in_desc, x1764, out_desc, x1771, sbmv_desc, x319,
    x580, 0.1, x400, x970, 1.0E-5,
    x1772, x1773));
};
float* x1779 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1780 = (float*)myMalloc(1 * sizeof(float));;
x1780[0] = 0.0f;
float* x1782 = (float*)myMalloc(1 * sizeof(float));;
x1782[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1782, x_desc, x1771, x1780, x_desc, x1771));
};
float* x1785 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1786 = (float*)myMalloc(1 * sizeof(float));;
x1786[0] = 0.0f;
float* x1788 = (float*)myMalloc(1 * sizeof(float));;
x1788[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1788, in_desc, x1771, filt_desc, x628,
    conv_desc, algo, ws_data, ws_size,
    x1786, out_desc, x1785));
};
float* x1791 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1792 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1793 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1794 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1795 = (float*)myMalloc(1 * sizeof(float));;
x1795[0] = 0.0f;
float* x1797 = (float*)myMalloc(1 * sizeof(float));;
x1797[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1797, x1795, in_desc, x1785, out_desc, x1792, sbmv_desc, x451,
    x1033, 0.1, x736, x559, 1.0E-5,
    x1793, x1794));
};
float* x1800 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1801 = (float*)myMalloc(1 * sizeof(float));;
x1801[0] = 1.0f;
float* x1803 = (float*)myMalloc(1 * sizeof(float));;
x1803[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1801, bias_desc, x1708, x1803, out_desc, x1792));
};
float* x1806 = (float*)myMalloc(1 * sizeof(float));;
x1806[0] = 0.0f;
float* x1808 = (float*)myMalloc(1 * sizeof(float));;
x1808[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1808, x_desc, x1792, x1806, x_desc, x1792));
};
float* x1811 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1812 = (float*)myMalloc(1 * sizeof(float));;
x1812[0] = 0.0f;
float* x1814 = (float*)myMalloc(1 * sizeof(float));;
x1814[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1814, in_desc, x1792, filt_desc, x883,
    conv_desc, algo, ws_data, ws_size,
    x1812, out_desc, x1811));
};
float* x1817 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1818 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1819 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1820 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1821 = (float*)myMalloc(1 * sizeof(float));;
x1821[0] = 0.0f;
float* x1823 = (float*)myMalloc(1 * sizeof(float));;
x1823[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1823, x1821, in_desc, x1811, out_desc, x1818, sbmv_desc, x430,
    x805, 0.1, x631, x322, 1.0E-5,
    x1819, x1820));
};
float* x1826 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1827 = (float*)myMalloc(1 * sizeof(float));;
x1827[0] = 0.0f;
float* x1829 = (float*)myMalloc(1 * sizeof(float));;
x1829[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1829, x_desc, x1818, x1827, x_desc, x1818));
};
float* x1832 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1833 = (float*)myMalloc(1 * sizeof(float));;
x1833[0] = 0.0f;
float* x1835 = (float*)myMalloc(1 * sizeof(float));;
x1835[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1835, in_desc, x1818, filt_desc, x868,
    conv_desc, algo, ws_data, ws_size,
    x1833, out_desc, x1832));
};
float* x1838 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1839 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1840 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1841 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1842 = (float*)myMalloc(1 * sizeof(float));;
x1842[0] = 0.0f;
float* x1844 = (float*)myMalloc(1 * sizeof(float));;
x1844[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1844, x1842, in_desc, x1832, out_desc, x1839, sbmv_desc, x676,
    x478, 0.1, x946, x1093, 1.0E-5,
    x1840, x1841));
};
float* x1847 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1848 = (float*)myMalloc(1 * sizeof(float));;
x1848[0] = 0.0f;
float* x1850 = (float*)myMalloc(1 * sizeof(float));;
x1850[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1850, x_desc, x1839, x1848, x_desc, x1839));
};
float* x1853 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1854 = (float*)myMalloc(1 * sizeof(float));;
x1854[0] = 0.0f;
float* x1856 = (float*)myMalloc(1 * sizeof(float));;
x1856[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1856, in_desc, x1839, filt_desc, x418,
    conv_desc, algo, ws_data, ws_size,
    x1854, out_desc, x1853));
};
float* x1859 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1860 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1861 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1862 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1863 = (float*)myMalloc(1 * sizeof(float));;
x1863[0] = 0.0f;
float* x1865 = (float*)myMalloc(1 * sizeof(float));;
x1865[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1865, x1863, in_desc, x1853, out_desc, x1860, sbmv_desc, x796,
    x541, 0.1, x370, x964, 1.0E-5,
    x1861, x1862));
};
float* x1868 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1869 = (float*)myMalloc(1 * sizeof(float));;
x1869[0] = 1.0f;
float* x1871 = (float*)myMalloc(1 * sizeof(float));;
x1871[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1869, bias_desc, x1792, x1871, out_desc, x1860));
};
float* x1874 = (float*)myMalloc(1 * sizeof(float));;
x1874[0] = 0.0f;
float* x1876 = (float*)myMalloc(1 * sizeof(float));;
x1876[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1876, x_desc, x1860, x1874, x_desc, x1860));
};
float* x1879 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1880 = (float*)myMalloc(1 * sizeof(float));;
x1880[0] = 0.0f;
float* x1882 = (float*)myMalloc(1 * sizeof(float));;
x1882[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1882, in_desc, x1860, filt_desc, x691,
    conv_desc, algo, ws_data, ws_size,
    x1880, out_desc, x1879));
};
float* x1885 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1886 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1887 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1888 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1889 = (float*)myMalloc(1 * sizeof(float));;
x1889[0] = 0.0f;
float* x1891 = (float*)myMalloc(1 * sizeof(float));;
x1891[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1891, x1889, in_desc, x1879, out_desc, x1886, sbmv_desc, x412,
    x1021, 0.1, x1003, x1078, 1.0E-5,
    x1887, x1888));
};
float* x1894 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1895 = (float*)myMalloc(1 * sizeof(float));;
x1895[0] = 0.0f;
float* x1897 = (float*)myMalloc(1 * sizeof(float));;
x1897[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1897, x_desc, x1886, x1895, x_desc, x1886));
};
float* x1900 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1901 = (float*)myMalloc(1 * sizeof(float));;
x1901[0] = 0.0f;
float* x1903 = (float*)myMalloc(1 * sizeof(float));;
x1903[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 128, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1903, in_desc, x1886, filt_desc, x790,
    conv_desc, algo, ws_data, ws_size,
    x1901, out_desc, x1900));
};
float* x1906 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1907 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1908 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1909 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1910 = (float*)myMalloc(1 * sizeof(float));;
x1910[0] = 0.0f;
float* x1912 = (float*)myMalloc(1 * sizeof(float));;
x1912[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1912, x1910, in_desc, x1900, out_desc, x1907, sbmv_desc, x532,
    x409, 0.1, x1099, x739, 1.0E-5,
    x1908, x1909));
};
float* x1915 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1916 = (float*)myMalloc(1 * sizeof(float));;
x1916[0] = 0.0f;
float* x1918 = (float*)myMalloc(1 * sizeof(float));;
x1918[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1918, x_desc, x1907, x1916, x_desc, x1907));
};
float* x1921 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1922 = (float*)myMalloc(1 * sizeof(float));;
x1922[0] = 0.0f;
float* x1924 = (float*)myMalloc(1 * sizeof(float));;
x1924[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1924, in_desc, x1907, filt_desc, x460,
    conv_desc, algo, ws_data, ws_size,
    x1922, out_desc, x1921));
};
float* x1927 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1928 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1929 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1930 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1931 = (float*)myMalloc(1 * sizeof(float));;
x1931[0] = 0.0f;
float* x1933 = (float*)myMalloc(1 * sizeof(float));;
x1933[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1933, x1931, in_desc, x1921, out_desc, x1928, sbmv_desc, x763,
    x457, 0.1, x352, x997, 1.0E-5,
    x1929, x1930));
};
float* x1936 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1937 = (float*)myMalloc(1 * sizeof(float));;
x1937[0] = 1.0f;
float* x1939 = (float*)myMalloc(1 * sizeof(float));;
x1939[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x1937, bias_desc, x1860, x1939, out_desc, x1928));
};
float* x1942 = (float*)myMalloc(1 * sizeof(float));;
x1942[0] = 0.0f;
float* x1944 = (float*)myMalloc(1 * sizeof(float));;
x1944[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1944, x_desc, x1928, x1942, x_desc, x1928));
};
float* x1947 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1948 = (float*)myMalloc(1 * sizeof(float));;
x1948[0] = 0.0f;
float* x1950 = (float*)myMalloc(1 * sizeof(float));;
x1950[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1950, in_desc, x1928, filt_desc, x835,
    conv_desc, algo, ws_data, ws_size,
    x1948, out_desc, x1947));
};
float* x1953 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1954 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1955 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1956 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1957 = (float*)myMalloc(1 * sizeof(float));;
x1957[0] = 0.0f;
float* x1959 = (float*)myMalloc(1 * sizeof(float));;
x1959[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1959, x1957, in_desc, x1947, out_desc, x1954, sbmv_desc, x1105,
    x358, 0.1, x688, x889, 1.0E-5,
    x1955, x1956));
};
float* x1962 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1963 = (float*)myMalloc(1 * sizeof(float));;
x1963[0] = 0.0f;
float* x1965 = (float*)myMalloc(1 * sizeof(float));;
x1965[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1965, x_desc, x1954, x1963, x_desc, x1954));
};
float* x1968 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1969 = (float*)myMalloc(1 * sizeof(float));;
x1969[0] = 0.0f;
float* x1971 = (float*)myMalloc(1 * sizeof(float));;
x1971[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 2, 2, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1971, in_desc, x1954, filt_desc, x820,
    conv_desc, algo, ws_data, ws_size,
    x1969, out_desc, x1968));
};
float* x1974 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1975 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1976 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1977 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1978 = (float*)myMalloc(1 * sizeof(float));;
x1978[0] = 0.0f;
float* x1980 = (float*)myMalloc(1 * sizeof(float));;
x1980[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1980, x1978, in_desc, x1968, out_desc, x1975, sbmv_desc, x619,
    x343, 0.1, x982, x592, 1.0E-5,
    x1976, x1977));
};
float* x1983 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1984 = (float*)myMalloc(1 * sizeof(float));;
x1984[0] = 0.0f;
float* x1986 = (float*)myMalloc(1 * sizeof(float));;
x1986[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1986, x_desc, x1975, x1984, x_desc, x1975));
};
float* x1989 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1990 = (float*)myMalloc(1 * sizeof(float));;
x1990[0] = 0.0f;
float* x1992 = (float*)myMalloc(1 * sizeof(float));;
x1992[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x1992, in_desc, x1975, filt_desc, x1102,
    conv_desc, algo, ws_data, ws_size,
    x1990, out_desc, x1989));
};
float* x1995 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1996 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1997 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1998 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x1999 = (float*)myMalloc(1 * sizeof(float));;
x1999[0] = 0.0f;
float* x2001 = (float*)myMalloc(1 * sizeof(float));;
x2001[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2001, x1999, in_desc, x1989, out_desc, x1996, sbmv_desc, x349,
    x646, 0.1, x943, x1096, 1.0E-5,
    x1997, x1998));
};
float* x2004 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2005 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2006 = (float*)myMalloc(1 * sizeof(float));;
x2006[0] = 0.0f;
float* x2008 = (float*)myMalloc(1 * sizeof(float));;
x2008[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 2, 2, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2008, in_desc, x1928, filt_desc, x520,
    conv_desc, algo, ws_data, ws_size,
    x2006, out_desc, x2005));
};
float* x2011 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2012 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2013 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2014 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2015 = (float*)myMalloc(1 * sizeof(float));;
x2015[0] = 0.0f;
float* x2017 = (float*)myMalloc(1 * sizeof(float));;
x2017[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2017, x2015, in_desc, x2005, out_desc, x2012, sbmv_desc, x382,
    x955, 0.1, x553, x928, 1.0E-5,
    x2013, x2014));
};
float* x2020 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2021 = (float*)myMalloc(1 * sizeof(float));;
x2021[0] = 1.0f;
float* x2023 = (float*)myMalloc(1 * sizeof(float));;
x2023[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2021, bias_desc, x2012, x2023, out_desc, x1996));
};
float* x2026 = (float*)myMalloc(1 * sizeof(float));;
x2026[0] = 0.0f;
float* x2028 = (float*)myMalloc(1 * sizeof(float));;
x2028[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2028, x_desc, x1996, x2026, x_desc, x1996));
};
float* x2031 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2032 = (float*)myMalloc(1 * sizeof(float));;
x2032[0] = 0.0f;
float* x2034 = (float*)myMalloc(1 * sizeof(float));;
x2034[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2034, in_desc, x1996, filt_desc, x334,
    conv_desc, algo, ws_data, ws_size,
    x2032, out_desc, x2031));
};
float* x2037 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2038 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2039 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2040 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2041 = (float*)myMalloc(1 * sizeof(float));;
x2041[0] = 0.0f;
float* x2043 = (float*)myMalloc(1 * sizeof(float));;
x2043[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2043, x2041, in_desc, x2031, out_desc, x2038, sbmv_desc, x385,
    x952, 0.1, x1072, x766, 1.0E-5,
    x2039, x2040));
};
float* x2046 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2047 = (float*)myMalloc(1 * sizeof(float));;
x2047[0] = 0.0f;
float* x2049 = (float*)myMalloc(1 * sizeof(float));;
x2049[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2049, x_desc, x2038, x2047, x_desc, x2038));
};
float* x2052 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2053 = (float*)myMalloc(1 * sizeof(float));;
x2053[0] = 0.0f;
float* x2055 = (float*)myMalloc(1 * sizeof(float));;
x2055[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2055, in_desc, x2038, filt_desc, x388,
    conv_desc, algo, ws_data, ws_size,
    x2053, out_desc, x2052));
};
float* x2058 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2059 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2060 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2061 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2062 = (float*)myMalloc(1 * sizeof(float));;
x2062[0] = 0.0f;
float* x2064 = (float*)myMalloc(1 * sizeof(float));;
x2064[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2064, x2062, in_desc, x2052, out_desc, x2059, sbmv_desc, x1108,
    x583, 0.1, x895, x1006, 1.0E-5,
    x2060, x2061));
};
float* x2067 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2068 = (float*)myMalloc(1 * sizeof(float));;
x2068[0] = 0.0f;
float* x2070 = (float*)myMalloc(1 * sizeof(float));;
x2070[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2070, x_desc, x2059, x2068, x_desc, x2059));
};
float* x2073 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2074 = (float*)myMalloc(1 * sizeof(float));;
x2074[0] = 0.0f;
float* x2076 = (float*)myMalloc(1 * sizeof(float));;
x2076[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2076, in_desc, x2059, filt_desc, x463,
    conv_desc, algo, ws_data, ws_size,
    x2074, out_desc, x2073));
};
float* x2079 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2080 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2081 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2082 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2083 = (float*)myMalloc(1 * sizeof(float));;
x2083[0] = 0.0f;
float* x2085 = (float*)myMalloc(1 * sizeof(float));;
x2085[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2085, x2083, in_desc, x2073, out_desc, x2080, sbmv_desc, x355,
    x991, 0.1, x841, x724, 1.0E-5,
    x2081, x2082));
};
float* x2088 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2089 = (float*)myMalloc(1 * sizeof(float));;
x2089[0] = 1.0f;
float* x2091 = (float*)myMalloc(1 * sizeof(float));;
x2091[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2089, bias_desc, x1996, x2091, out_desc, x2080));
};
float* x2094 = (float*)myMalloc(1 * sizeof(float));;
x2094[0] = 0.0f;
float* x2096 = (float*)myMalloc(1 * sizeof(float));;
x2096[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2096, x_desc, x2080, x2094, x_desc, x2080));
};
float* x2099 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2100 = (float*)myMalloc(1 * sizeof(float));;
x2100[0] = 0.0f;
float* x2102 = (float*)myMalloc(1 * sizeof(float));;
x2102[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2102, in_desc, x2080, filt_desc, x949,
    conv_desc, algo, ws_data, ws_size,
    x2100, out_desc, x2099));
};
float* x2105 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2106 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2107 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2108 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2109 = (float*)myMalloc(1 * sizeof(float));;
x2109[0] = 0.0f;
float* x2111 = (float*)myMalloc(1 * sizeof(float));;
x2111[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2111, x2109, in_desc, x2099, out_desc, x2106, sbmv_desc, x682,
    x886, 0.1, x829, x817, 1.0E-5,
    x2107, x2108));
};
float* x2114 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2115 = (float*)myMalloc(1 * sizeof(float));;
x2115[0] = 0.0f;
float* x2117 = (float*)myMalloc(1 * sizeof(float));;
x2117[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2117, x_desc, x2106, x2115, x_desc, x2106));
};
float* x2120 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2121 = (float*)myMalloc(1 * sizeof(float));;
x2121[0] = 0.0f;
float* x2123 = (float*)myMalloc(1 * sizeof(float));;
x2123[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2123, in_desc, x2106, filt_desc, x337,
    conv_desc, algo, ws_data, ws_size,
    x2121, out_desc, x2120));
};
float* x2126 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2127 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2128 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2129 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2130 = (float*)myMalloc(1 * sizeof(float));;
x2130[0] = 0.0f;
float* x2132 = (float*)myMalloc(1 * sizeof(float));;
x2132[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2132, x2130, in_desc, x2120, out_desc, x2127, sbmv_desc, x979,
    x871, 0.1, x667, x484, 1.0E-5,
    x2128, x2129));
};
float* x2135 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2136 = (float*)myMalloc(1 * sizeof(float));;
x2136[0] = 0.0f;
float* x2138 = (float*)myMalloc(1 * sizeof(float));;
x2138[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2138, x_desc, x2127, x2136, x_desc, x2127));
};
float* x2141 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2142 = (float*)myMalloc(1 * sizeof(float));;
x2142[0] = 0.0f;
float* x2144 = (float*)myMalloc(1 * sizeof(float));;
x2144[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2144, in_desc, x2127, filt_desc, x643,
    conv_desc, algo, ws_data, ws_size,
    x2142, out_desc, x2141));
};
float* x2147 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2148 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2149 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2150 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2151 = (float*)myMalloc(1 * sizeof(float));;
x2151[0] = 0.0f;
float* x2153 = (float*)myMalloc(1 * sizeof(float));;
x2153[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2153, x2151, in_desc, x2141, out_desc, x2148, sbmv_desc, x1084,
    x466, 0.1, x715, x859, 1.0E-5,
    x2149, x2150));
};
float* x2156 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2157 = (float*)myMalloc(1 * sizeof(float));;
x2157[0] = 1.0f;
float* x2159 = (float*)myMalloc(1 * sizeof(float));;
x2159[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2157, bias_desc, x2080, x2159, out_desc, x2148));
};
float* x2162 = (float*)myMalloc(1 * sizeof(float));;
x2162[0] = 0.0f;
float* x2164 = (float*)myMalloc(1 * sizeof(float));;
x2164[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2164, x_desc, x2148, x2162, x_desc, x2148));
};
float* x2167 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2168 = (float*)myMalloc(1 * sizeof(float));;
x2168[0] = 0.0f;
float* x2170 = (float*)myMalloc(1 * sizeof(float));;
x2170[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2170, in_desc, x2148, filt_desc, x313,
    conv_desc, algo, ws_data, ws_size,
    x2168, out_desc, x2167));
};
float* x2173 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2174 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2175 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2176 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2177 = (float*)myMalloc(1 * sizeof(float));;
x2177[0] = 0.0f;
float* x2179 = (float*)myMalloc(1 * sizeof(float));;
x2179[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2179, x2177, in_desc, x2167, out_desc, x2174, sbmv_desc, x571,
    x1018, 0.1, x784, x589, 1.0E-5,
    x2175, x2176));
};
float* x2182 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2183 = (float*)myMalloc(1 * sizeof(float));;
x2183[0] = 0.0f;
float* x2185 = (float*)myMalloc(1 * sizeof(float));;
x2185[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2185, x_desc, x2174, x2183, x_desc, x2174));
};
float* x2188 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2189 = (float*)myMalloc(1 * sizeof(float));;
x2189[0] = 0.0f;
float* x2191 = (float*)myMalloc(1 * sizeof(float));;
x2191[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2191, in_desc, x2174, filt_desc, x1042,
    conv_desc, algo, ws_data, ws_size,
    x2189, out_desc, x2188));
};
float* x2194 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2195 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2196 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2197 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2198 = (float*)myMalloc(1 * sizeof(float));;
x2198[0] = 0.0f;
float* x2200 = (float*)myMalloc(1 * sizeof(float));;
x2200[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2200, x2198, in_desc, x2188, out_desc, x2195, sbmv_desc, x517,
    x703, 0.1, x853, x985, 1.0E-5,
    x2196, x2197));
};
float* x2203 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2204 = (float*)myMalloc(1 * sizeof(float));;
x2204[0] = 0.0f;
float* x2206 = (float*)myMalloc(1 * sizeof(float));;
x2206[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2206, x_desc, x2195, x2204, x_desc, x2195));
};
float* x2209 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2210 = (float*)myMalloc(1 * sizeof(float));;
x2210[0] = 0.0f;
float* x2212 = (float*)myMalloc(1 * sizeof(float));;
x2212[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2212, in_desc, x2195, filt_desc, x562,
    conv_desc, algo, ws_data, ws_size,
    x2210, out_desc, x2209));
};
float* x2215 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2216 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2217 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2218 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2219 = (float*)myMalloc(1 * sizeof(float));;
x2219[0] = 0.0f;
float* x2221 = (float*)myMalloc(1 * sizeof(float));;
x2221[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2221, x2219, in_desc, x2209, out_desc, x2216, sbmv_desc, x1009,
    x733, 0.1, x988, x778, 1.0E-5,
    x2217, x2218));
};
float* x2224 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2225 = (float*)myMalloc(1 * sizeof(float));;
x2225[0] = 1.0f;
float* x2227 = (float*)myMalloc(1 * sizeof(float));;
x2227[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2225, bias_desc, x2148, x2227, out_desc, x2216));
};
float* x2230 = (float*)myMalloc(1 * sizeof(float));;
x2230[0] = 0.0f;
float* x2232 = (float*)myMalloc(1 * sizeof(float));;
x2232[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2232, x_desc, x2216, x2230, x_desc, x2216));
};
float* x2235 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2236 = (float*)myMalloc(1 * sizeof(float));;
x2236[0] = 0.0f;
float* x2238 = (float*)myMalloc(1 * sizeof(float));;
x2238[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2238, in_desc, x2216, filt_desc, x361,
    conv_desc, algo, ws_data, ws_size,
    x2236, out_desc, x2235));
};
float* x2241 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2242 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2243 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2244 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2245 = (float*)myMalloc(1 * sizeof(float));;
x2245[0] = 0.0f;
float* x2247 = (float*)myMalloc(1 * sizeof(float));;
x2247[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2247, x2245, in_desc, x2235, out_desc, x2242, sbmv_desc, x526,
    x850, 0.1, x1057, x502, 1.0E-5,
    x2243, x2244));
};
float* x2250 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2251 = (float*)myMalloc(1 * sizeof(float));;
x2251[0] = 0.0f;
float* x2253 = (float*)myMalloc(1 * sizeof(float));;
x2253[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2253, x_desc, x2242, x2251, x_desc, x2242));
};
float* x2256 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2257 = (float*)myMalloc(1 * sizeof(float));;
x2257[0] = 0.0f;
float* x2259 = (float*)myMalloc(1 * sizeof(float));;
x2259[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2259, in_desc, x2242, filt_desc, x1081,
    conv_desc, algo, ws_data, ws_size,
    x2257, out_desc, x2256));
};
float* x2262 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2263 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2264 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2265 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2266 = (float*)myMalloc(1 * sizeof(float));;
x2266[0] = 0.0f;
float* x2268 = (float*)myMalloc(1 * sizeof(float));;
x2268[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2268, x2266, in_desc, x2256, out_desc, x2263, sbmv_desc, x799,
    x622, 0.1, x1045, x607, 1.0E-5,
    x2264, x2265));
};
float* x2271 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2272 = (float*)myMalloc(1 * sizeof(float));;
x2272[0] = 0.0f;
float* x2274 = (float*)myMalloc(1 * sizeof(float));;
x2274[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2274, x_desc, x2263, x2272, x_desc, x2263));
};
float* x2277 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2278 = (float*)myMalloc(1 * sizeof(float));;
x2278[0] = 0.0f;
float* x2280 = (float*)myMalloc(1 * sizeof(float));;
x2280[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2280, in_desc, x2263, filt_desc, x958,
    conv_desc, algo, ws_data, ws_size,
    x2278, out_desc, x2277));
};
float* x2283 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2284 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2285 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2286 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2287 = (float*)myMalloc(1 * sizeof(float));;
x2287[0] = 0.0f;
float* x2289 = (float*)myMalloc(1 * sizeof(float));;
x2289[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2289, x2287, in_desc, x2277, out_desc, x2284, sbmv_desc, x472,
    x655, 0.1, x922, x1111, 1.0E-5,
    x2285, x2286));
};
float* x2292 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2293 = (float*)myMalloc(1 * sizeof(float));;
x2293[0] = 1.0f;
float* x2295 = (float*)myMalloc(1 * sizeof(float));;
x2295[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2293, bias_desc, x2216, x2295, out_desc, x2284));
};
float* x2298 = (float*)myMalloc(1 * sizeof(float));;
x2298[0] = 0.0f;
float* x2300 = (float*)myMalloc(1 * sizeof(float));;
x2300[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2300, x_desc, x2284, x2298, x_desc, x2284));
};
float* x2303 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2304 = (float*)myMalloc(1 * sizeof(float));;
x2304[0] = 0.0f;
float* x2306 = (float*)myMalloc(1 * sizeof(float));;
x2306[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2306, in_desc, x2284, filt_desc, x748,
    conv_desc, algo, ws_data, ws_size,
    x2304, out_desc, x2303));
};
float* x2309 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2310 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2311 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2312 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2313 = (float*)myMalloc(1 * sizeof(float));;
x2313[0] = 0.0f;
float* x2315 = (float*)myMalloc(1 * sizeof(float));;
x2315[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2315, x2313, in_desc, x2303, out_desc, x2310, sbmv_desc, x550,
    x1054, 0.1, x535, x823, 1.0E-5,
    x2311, x2312));
};
float* x2318 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2319 = (float*)myMalloc(1 * sizeof(float));;
x2319[0] = 0.0f;
float* x2321 = (float*)myMalloc(1 * sizeof(float));;
x2321[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2321, x_desc, x2310, x2319, x_desc, x2310));
};
float* x2324 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2325 = (float*)myMalloc(1 * sizeof(float));;
x2325[0] = 0.0f;
float* x2327 = (float*)myMalloc(1 * sizeof(float));;
x2327[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 256, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2327, in_desc, x2310, filt_desc, x973,
    conv_desc, algo, ws_data, ws_size,
    x2325, out_desc, x2324));
};
float* x2330 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2331 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2332 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2333 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2334 = (float*)myMalloc(1 * sizeof(float));;
x2334[0] = 0.0f;
float* x2336 = (float*)myMalloc(1 * sizeof(float));;
x2336[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2336, x2334, in_desc, x2324, out_desc, x2331, sbmv_desc, x718,
    x862, 0.1, x505, x1015, 1.0E-5,
    x2332, x2333));
};
float* x2339 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2340 = (float*)myMalloc(1 * sizeof(float));;
x2340[0] = 0.0f;
float* x2342 = (float*)myMalloc(1 * sizeof(float));;
x2342[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2342, x_desc, x2331, x2340, x_desc, x2331));
};
float* x2345 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2346 = (float*)myMalloc(1 * sizeof(float));;
x2346[0] = 0.0f;
float* x2348 = (float*)myMalloc(1 * sizeof(float));;
x2348[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    1024, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2348, in_desc, x2331, filt_desc, x586,
    conv_desc, algo, ws_data, ws_size,
    x2346, out_desc, x2345));
};
float* x2351 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2352 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2353 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2354 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2355 = (float*)myMalloc(1 * sizeof(float));;
x2355[0] = 0.0f;
float* x2357 = (float*)myMalloc(1 * sizeof(float));;
x2357[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2357, x2355, in_desc, x2345, out_desc, x2352, sbmv_desc, x1039,
    x574, 0.1, x661, x844, 1.0E-5,
    x2353, x2354));
};
float* x2360 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2361 = (float*)myMalloc(1 * sizeof(float));;
x2361[0] = 1.0f;
float* x2363 = (float*)myMalloc(1 * sizeof(float));;
x2363[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2361, bias_desc, x2284, x2363, out_desc, x2352));
};
float* x2366 = (float*)myMalloc(1 * sizeof(float));;
x2366[0] = 0.0f;
float* x2368 = (float*)myMalloc(1 * sizeof(float));;
x2368[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2368, x_desc, x2352, x2366, x_desc, x2352));
};
float* x2371 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2372 = (float*)myMalloc(1 * sizeof(float));;
x2372[0] = 0.0f;
float* x2374 = (float*)myMalloc(1 * sizeof(float));;
x2374[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2374, in_desc, x2352, filt_desc, x712,
    conv_desc, algo, ws_data, ws_size,
    x2372, out_desc, x2371));
};
float* x2377 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2378 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2379 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2380 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2381 = (float*)myMalloc(1 * sizeof(float));;
x2381[0] = 0.0f;
float* x2383 = (float*)myMalloc(1 * sizeof(float));;
x2383[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2383, x2381, in_desc, x2371, out_desc, x2378, sbmv_desc, x898,
    x967, 0.1, x496, x658, 1.0E-5,
    x2379, x2380));
};
float* x2386 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2387 = (float*)myMalloc(1 * sizeof(float));;
x2387[0] = 0.0f;
float* x2389 = (float*)myMalloc(1 * sizeof(float));;
x2389[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 4, 4));

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
float* x2392 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2393 = (float*)myMalloc(1 * sizeof(float));;
x2393[0] = 0.0f;
float* x2395 = (float*)myMalloc(1 * sizeof(float));;
x2395[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 2, 2, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2395, in_desc, x2378, filt_desc, x397,
    conv_desc, algo, ws_data, ws_size,
    x2393, out_desc, x2392));
};
float* x2398 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2399 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2400 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2401 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2402 = (float*)myMalloc(1 * sizeof(float));;
x2402[0] = 0.0f;
float* x2404 = (float*)myMalloc(1 * sizeof(float));;
x2404[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2404, x2402, in_desc, x2392, out_desc, x2399, sbmv_desc, x910,
    x772, 0.1, x634, x445, 1.0E-5,
    x2400, x2401));
};
float* x2407 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2408 = (float*)myMalloc(1 * sizeof(float));;
x2408[0] = 0.0f;
float* x2410 = (float*)myMalloc(1 * sizeof(float));;
x2410[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2410, x_desc, x2399, x2408, x_desc, x2399));
};
float* x2413 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2414 = (float*)myMalloc(1 * sizeof(float));;
x2414[0] = 0.0f;
float* x2416 = (float*)myMalloc(1 * sizeof(float));;
x2416[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2416, in_desc, x2399, filt_desc, x931,
    conv_desc, algo, ws_data, ws_size,
    x2414, out_desc, x2413));
};
float* x2419 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2420 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2421 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x2422 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x2423 = (float*)myMalloc(1 * sizeof(float));;
x2423[0] = 0.0f;
float* x2425 = (float*)myMalloc(1 * sizeof(float));;
x2425[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2425, x2423, in_desc, x2413, out_desc, x2420, sbmv_desc, x1012,
    x481, 0.1, x640, x874, 1.0E-5,
    x2421, x2422));
};
float* x2428 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2429 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2430 = (float*)myMalloc(1 * sizeof(float));;
x2430[0] = 0.0f;
float* x2432 = (float*)myMalloc(1 * sizeof(float));;
x2432[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 1024, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 2, 2, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2432, in_desc, x2352, filt_desc, x937,
    conv_desc, algo, ws_data, ws_size,
    x2430, out_desc, x2429));
};
float* x2435 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2436 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2437 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x2438 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x2439 = (float*)myMalloc(1 * sizeof(float));;
x2439[0] = 0.0f;
float* x2441 = (float*)myMalloc(1 * sizeof(float));;
x2441[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2441, x2439, in_desc, x2429, out_desc, x2436, sbmv_desc, x814,
    x616, 0.1, x487, x670, 1.0E-5,
    x2437, x2438));
};
float* x2444 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2445 = (float*)myMalloc(1 * sizeof(float));;
x2445[0] = 1.0f;
float* x2447 = (float*)myMalloc(1 * sizeof(float));;
x2447[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2445, bias_desc, x2436, x2447, out_desc, x2420));
};
float* x2450 = (float*)myMalloc(1 * sizeof(float));;
x2450[0] = 0.0f;
float* x2452 = (float*)myMalloc(1 * sizeof(float));;
x2452[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2452, x_desc, x2420, x2450, x_desc, x2420));
};
float* x2455 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2456 = (float*)myMalloc(1 * sizeof(float));;
x2456[0] = 0.0f;
float* x2458 = (float*)myMalloc(1 * sizeof(float));;
x2458[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 2048, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2458, in_desc, x2420, filt_desc, x940,
    conv_desc, algo, ws_data, ws_size,
    x2456, out_desc, x2455));
};
float* x2461 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2462 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2463 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2464 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2465 = (float*)myMalloc(1 * sizeof(float));;
x2465[0] = 0.0f;
float* x2467 = (float*)myMalloc(1 * sizeof(float));;
x2467[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2467, x2465, in_desc, x2455, out_desc, x2462, sbmv_desc, x433,
    x706, 0.1, x757, x490, 1.0E-5,
    x2463, x2464));
};
float* x2470 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2471 = (float*)myMalloc(1 * sizeof(float));;
x2471[0] = 0.0f;
float* x2473 = (float*)myMalloc(1 * sizeof(float));;
x2473[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2473, x_desc, x2462, x2471, x_desc, x2462));
};
float* x2476 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2477 = (float*)myMalloc(1 * sizeof(float));;
x2477[0] = 0.0f;
float* x2479 = (float*)myMalloc(1 * sizeof(float));;
x2479[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2479, in_desc, x2462, filt_desc, x760,
    conv_desc, algo, ws_data, ws_size,
    x2477, out_desc, x2476));
};
float* x2482 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2483 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2484 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2485 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2486 = (float*)myMalloc(1 * sizeof(float));;
x2486[0] = 0.0f;
float* x2488 = (float*)myMalloc(1 * sizeof(float));;
x2488[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2488, x2486, in_desc, x2476, out_desc, x2483, sbmv_desc, x775,
    x493, 0.1, x709, x880, 1.0E-5,
    x2484, x2485));
};
float* x2491 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2492 = (float*)myMalloc(1 * sizeof(float));;
x2492[0] = 0.0f;
float* x2494 = (float*)myMalloc(1 * sizeof(float));;
x2494[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2494, x_desc, x2483, x2492, x_desc, x2483));
};
float* x2497 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2498 = (float*)myMalloc(1 * sizeof(float));;
x2498[0] = 0.0f;
float* x2500 = (float*)myMalloc(1 * sizeof(float));;
x2500[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2500, in_desc, x2483, filt_desc, x436,
    conv_desc, algo, ws_data, ws_size,
    x2498, out_desc, x2497));
};
float* x2503 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2504 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2505 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x2506 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x2507 = (float*)myMalloc(1 * sizeof(float));;
x2507[0] = 0.0f;
float* x2509 = (float*)myMalloc(1 * sizeof(float));;
x2509[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2509, x2507, in_desc, x2497, out_desc, x2504, sbmv_desc, x577,
    x727, 0.1, x499, x1030, 1.0E-5,
    x2505, x2506));
};
float* x2512 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2513 = (float*)myMalloc(1 * sizeof(float));;
x2513[0] = 1.0f;
float* x2515 = (float*)myMalloc(1 * sizeof(float));;
x2515[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2513, bias_desc, x2420, x2515, out_desc, x2504));
};
float* x2518 = (float*)myMalloc(1 * sizeof(float));;
x2518[0] = 0.0f;
float* x2520 = (float*)myMalloc(1 * sizeof(float));;
x2520[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2520, x_desc, x2504, x2518, x_desc, x2504));
};
float* x2523 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2524 = (float*)myMalloc(1 * sizeof(float));;
x2524[0] = 0.0f;
float* x2526 = (float*)myMalloc(1 * sizeof(float));;
x2526[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 2048, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2526, in_desc, x2504, filt_desc, x1090,
    conv_desc, algo, ws_data, ws_size,
    x2524, out_desc, x2523));
};
float* x2529 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2530 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2531 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2532 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2533 = (float*)myMalloc(1 * sizeof(float));;
x2533[0] = 0.0f;
float* x2535 = (float*)myMalloc(1 * sizeof(float));;
x2535[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2535, x2533, in_desc, x2523, out_desc, x2530, sbmv_desc, x340,
    x529, 0.1, x934, x1060, 1.0E-5,
    x2531, x2532));
};
float* x2538 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2539 = (float*)myMalloc(1 * sizeof(float));;
x2539[0] = 0.0f;
float* x2541 = (float*)myMalloc(1 * sizeof(float));;
x2541[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2541, x_desc, x2530, x2539, x_desc, x2530));
};
float* x2544 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2545 = (float*)myMalloc(1 * sizeof(float));;
x2545[0] = 0.0f;
float* x2547 = (float*)myMalloc(1 * sizeof(float));;
x2547[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    512, 512, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2547, in_desc, x2530, filt_desc, x379,
    conv_desc, algo, ws_data, ws_size,
    x2545, out_desc, x2544));
};
float* x2550 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2551 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2552 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2553 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2554 = (float*)myMalloc(1 * sizeof(float));;
x2554[0] = 0.0f;
float* x2556 = (float*)myMalloc(1 * sizeof(float));;
x2556[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2556, x2554, in_desc, x2544, out_desc, x2551, sbmv_desc, x877,
    x802, 0.1, x331, x901, 1.0E-5,
    x2552, x2553));
};
float* x2559 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2560 = (float*)myMalloc(1 * sizeof(float));;
x2560[0] = 0.0f;
float* x2562 = (float*)myMalloc(1 * sizeof(float));;
x2562[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2562, x_desc, x2551, x2560, x_desc, x2551));
};
float* x2565 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2566 = (float*)myMalloc(1 * sizeof(float));;
x2566[0] = 0.0f;
float* x2568 = (float*)myMalloc(1 * sizeof(float));;
x2568[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    2048, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2568, in_desc, x2551, filt_desc, x394,
    conv_desc, algo, ws_data, ws_size,
    x2566, out_desc, x2565));
};
float* x2571 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2572 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2573 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x2574 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x2575 = (float*)myMalloc(1 * sizeof(float));;
x2575[0] = 0.0f;
float* x2577 = (float*)myMalloc(1 * sizeof(float));;
x2577[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2577, x2575, in_desc, x2565, out_desc, x2572, sbmv_desc, x604,
    x838, 0.1, x1075, x664, 1.0E-5,
    x2573, x2574));
};
float* x2580 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2581 = (float*)myMalloc(1 * sizeof(float));;
x2581[0] = 1.0f;
float* x2583 = (float*)myMalloc(1 * sizeof(float));;
x2583[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2581, bias_desc, x2504, x2583, out_desc, x2572));
};
float* x2586 = (float*)myMalloc(1 * sizeof(float));;
x2586[0] = 0.0f;
float* x2588 = (float*)myMalloc(1 * sizeof(float));;
x2588[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x2588, x_desc, x2572, x2586, x_desc, x2572));
};
float* x2591 = (float*)myMalloc(1 * sizeof(float));;
x2591[0] = 0.0f;
float* x2593 = (float*)myMalloc(1 * sizeof(float));;
x2593[0] = 1.0f;
float* x2595 = (float*)myGpuMalloc(131072 * sizeof(float));

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 1, 1));

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
    x2593, in_desc, x2572, x2591, out_desc, x2595));
};
float* x2597 = (float*)myGpuMalloc(131072 * sizeof(float));
// resize to WrappedArray(64, -1)
// resize to WrappedArray(64, -1)
// gemm: ArrayBuffer(64, 2048), Vector(10, 2048)
float* x2601 = (float*)myGpuMalloc(640 * sizeof(float));
float* x2602 = (float*)myMalloc(1 * sizeof(float));;
x2602[0] = 0.0f;
float* x2604 = (float*)myMalloc(1 * sizeof(float));;
x2604[0] = 1.0f;
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 10,64,2048,x2604,x976,2048,x2595,2048,x2602,x2601,10));
float* x2607 = (float*)myGpuMalloc(640 * sizeof(float));
float* x2608 = (float*)myMalloc(1 * sizeof(float));;
x2608[0] = 1.0f;
float* x2610 = (float*)myMalloc(1 * sizeof(float));;
x2610[0] = 1.0f;

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
    cudnnHandle, x2608, bias_desc, x439, x2610, out_desc, x2601));
};
// resize to WrappedArray(64, 10, 1, 1)
float* x2614 = (float*)myMalloc(1 * sizeof(float));;
x2614[0] = 0.0f;
float* x2616 = (float*)myMalloc(1 * sizeof(float));;
x2616[0] = 1.0f;
float* x2618 = (float*)myGpuMalloc(640 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 10, 1, 1));
CUDNN_CALL(cudnnSoftmaxForward(
    cudnnHandle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL,
    x2616, x_desc, x2601, x2614, x_desc, x2618));
};
// resize to WrappedArray(64, 10)
float* x2621 = (float*)myGpuMalloc(640 * sizeof(float));
float* x2622 = (float*)myGpuMalloc(64 * sizeof(float));
nllLoss<<<64, 1>>>(x2618,10,x2622,x1405);
float* x2624 = (float*)myGpuMalloc(64 * sizeof(float));
// resize to ArrayBuffer(64, 1, 1, 1)
float* x2626 = (float*)myGpuMalloc(1 * sizeof(float));
float* x2627 = (float*)myMalloc(1 * sizeof(float));;
x2627[0] = 0.0f;
float* x2629 = (float*)myMalloc(1 * sizeof(float));;
x2629[0] = 1.0f;

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
    x2629, x_desc, x2622, x2627, out_desc, x2626));
};
// resize to WrappedArray(1)
float* x2633 = (float*)myGpuMalloc(1 * sizeof(float));
arrayFill_greg<<<1, 512>>>(x2633, 1.0f, 1);
// backend is lantern.TensorDslCudnn$BackendCudnn@1b9947d9
CUDA_CALL(cudaMemcpy(x1410, x2626, 1 * sizeof(float), cudaMemcpyDeviceToHost));
// 'mean' gradient
// backprop for mean op
// resize to WrappedArray(1, 1, 1, 1)
// resize to ArrayBuffer(64, 1, 1, 1)
float* x2641 = (float*)myMalloc(1 * sizeof(float));;
x2641[0] = 0.015625f;
float* x2643 = (float*)myMalloc(1 * sizeof(float));;
x2643[0] = 1.0f;

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
    cudnnHandle, x2641, bias_desc, x2633, x2643, out_desc, x2624));
};
// 'nllLossB' gradient.
nllLoss_grad<<<64, 1>>>(10,x2624,x1405,x2621);
// resize to WrappedArray(64, 10, 1, 1)
// resize to WrappedArray(64, 10, 1, 1)
// resize to WrappedArray(64, 10, 1, 1)
// resize to WrappedArray(64, 10, 1, 1)
float* x2652 = (float*)myMalloc(1 * sizeof(float));;
x2652[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 10, 1, 1));
CUDNN_CALL(cudnnSoftmaxBackward(
    cudnnHandle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL,
    x2652, x_desc, x2618, x_desc, x2621,
    x2652, x_desc, x2607));
};
float* x2655 = (float*)myMalloc(1 * sizeof(float));;
x2655[0] = 1.0f;

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
    cudnnHandle, x2655, grad_out_desc, x2607,
    x2655, grad_bias_desc, x1155));
};
// backprop for gemm ArrayBuffer(64, 2048), Vector(10, 2048)
float* x2659 = (float*)myMalloc(1 * sizeof(float));;
x2659[0] = 1.0f;
float* x2661 = (float*)myMalloc(1 * sizeof(float));;
x2661[0] = 1.0f;
// backprop of gemm
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2048,64,10,x2659,x976,2048,x2607,10,x2661,x2597,2048));
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 2048,10,64,x2659,x2595,2048,x2607,10,x2661,x1334,2048));
float* x2666 = (float*)myMalloc(1 * sizeof(float));;
x2666[0] = 0.0f;
float* x2668 = (float*)myMalloc(1 * sizeof(float));;
x2668[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 1, 1));

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
    x2668, out_desc, x2595, out_desc, x2597, in_desc, x2572  , x2666, in_desc, x2580));
};
float* x2671 = (float*)myMalloc(1 * sizeof(float));;
x2671[0] = 1.0f;
float* x2673 = (float*)myMalloc(1 * sizeof(float));;
x2673[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x2671, x_desc, x2572, x_desc, x2580, x_desc, x2572,
    x2673, x_desc, x2580));
};
float* x2676 = (float*)myMalloc(1 * sizeof(float));;
x2676[0] = 1.0f;
float* x2678 = (float*)myMalloc(1 * sizeof(float));;
x2678[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2676, bias_desc, x2580, x2678, out_desc, x2512));
};
float* x2681 = (float*)myMalloc(1 * sizeof(float));;
x2681[0] = 0.0f;
float* x2683 = (float*)myMalloc(1 * sizeof(float));;
x2683[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2683, x2683, x2683, x2683, in_desc, x2565,
    out_desc, x2580, in_desc, x2571, sbmv_desc, x604,
    x1210,x1288, 1.0E-5, x2573, x2574));
};
// conv2D back-propagate
float* x2687 = (float*)myMalloc(1 * sizeof(float));;
x2687[0] = 1.0f;

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
    64, 512, 2, 2));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2687, filt_desc, x394, grad_out_desc, x2571,
    conv_desc, algo, ws_data, ws_size,
    x2687, grad_in_desc, x2559));
};
float* x2690 = (float*)myMalloc(1 * sizeof(float));;
x2690[0] = 1.0f;

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
    64, 2048, 2, 2));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2690, in_desc, x2551, grad_out_desc, x2571,
    conv_desc, algo, ws_data, ws_size,
    x2690, grad_filt_desc, x1140));
};
float* x2693 = (float*)myMalloc(1 * sizeof(float));;
x2693[0] = 1.0f;
float* x2695 = (float*)myMalloc(1 * sizeof(float));;
x2695[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x2693, x_desc, x2551, x_desc, x2559, x_desc, x2551,
    x2695, x_desc, x2559));
};
float* x2698 = (float*)myMalloc(1 * sizeof(float));;
x2698[0] = 0.0f;
float* x2700 = (float*)myMalloc(1 * sizeof(float));;
x2700[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2700, x2700, x2700, x2700, in_desc, x2544,
    out_desc, x2559, in_desc, x2550, sbmv_desc, x877,
    x1301,x1276, 1.0E-5, x2552, x2553));
};
// conv2D back-propagate
float* x2704 = (float*)myMalloc(1 * sizeof(float));;
x2704[0] = 1.0f;

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
    64, 512, 2, 2));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2704, filt_desc, x379, grad_out_desc, x2550,
    conv_desc, algo, ws_data, ws_size,
    x2704, grad_in_desc, x2538));
};
float* x2707 = (float*)myMalloc(1 * sizeof(float));;
x2707[0] = 1.0f;

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
    64, 512, 2, 2));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2707, in_desc, x2530, grad_out_desc, x2550,
    conv_desc, algo, ws_data, ws_size,
    x2707, grad_filt_desc, x1135));
};
float* x2710 = (float*)myMalloc(1 * sizeof(float));;
x2710[0] = 1.0f;
float* x2712 = (float*)myMalloc(1 * sizeof(float));;
x2712[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x2710, x_desc, x2530, x_desc, x2538, x_desc, x2530,
    x2712, x_desc, x2538));
};
float* x2715 = (float*)myMalloc(1 * sizeof(float));;
x2715[0] = 0.0f;
float* x2717 = (float*)myMalloc(1 * sizeof(float));;
x2717[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2717, x2717, x2717, x2717, in_desc, x2523,
    out_desc, x2538, in_desc, x2529, sbmv_desc, x340,
    x1122,x1185, 1.0E-5, x2531, x2532));
};
// conv2D back-propagate
float* x2721 = (float*)myMalloc(1 * sizeof(float));;
x2721[0] = 1.0f;

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
    64, 2048, 2, 2));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2721, filt_desc, x1090, grad_out_desc, x2529,
    conv_desc, algo, ws_data, ws_size,
    x2721, grad_in_desc, x2512));
};
float* x2724 = (float*)myMalloc(1 * sizeof(float));;
x2724[0] = 1.0f;

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
    64, 512, 2, 2));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2724, in_desc, x2504, grad_out_desc, x2529,
    conv_desc, algo, ws_data, ws_size,
    x2724, grad_filt_desc, x1372));
};
float* x2727 = (float*)myMalloc(1 * sizeof(float));;
x2727[0] = 1.0f;
float* x2729 = (float*)myMalloc(1 * sizeof(float));;
x2729[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x2727, x_desc, x2504, x_desc, x2512, x_desc, x2504,
    x2729, x_desc, x2512));
};
float* x2732 = (float*)myMalloc(1 * sizeof(float));;
x2732[0] = 1.0f;
float* x2734 = (float*)myMalloc(1 * sizeof(float));;
x2734[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2732, bias_desc, x2512, x2734, out_desc, x2428));
};
float* x2737 = (float*)myMalloc(1 * sizeof(float));;
x2737[0] = 0.0f;
float* x2739 = (float*)myMalloc(1 * sizeof(float));;
x2739[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2739, x2739, x2739, x2739, in_desc, x2497,
    out_desc, x2512, in_desc, x2503, sbmv_desc, x577,
    x1201,x1251, 1.0E-5, x2505, x2506));
};
// conv2D back-propagate
float* x2743 = (float*)myMalloc(1 * sizeof(float));;
x2743[0] = 1.0f;

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
    64, 512, 2, 2));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2743, filt_desc, x436, grad_out_desc, x2503,
    conv_desc, algo, ws_data, ws_size,
    x2743, grad_in_desc, x2491));
};
float* x2746 = (float*)myMalloc(1 * sizeof(float));;
x2746[0] = 1.0f;

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
    64, 2048, 2, 2));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2746, in_desc, x2483, grad_out_desc, x2503,
    conv_desc, algo, ws_data, ws_size,
    x2746, grad_filt_desc, x1154));
};
float* x2749 = (float*)myMalloc(1 * sizeof(float));;
x2749[0] = 1.0f;
float* x2751 = (float*)myMalloc(1 * sizeof(float));;
x2751[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x2749, x_desc, x2483, x_desc, x2491, x_desc, x2483,
    x2751, x_desc, x2491));
};
float* x2754 = (float*)myMalloc(1 * sizeof(float));;
x2754[0] = 0.0f;
float* x2756 = (float*)myMalloc(1 * sizeof(float));;
x2756[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2756, x2756, x2756, x2756, in_desc, x2476,
    out_desc, x2491, in_desc, x2482, sbmv_desc, x775,
    x1267,x1173, 1.0E-5, x2484, x2485));
};
// conv2D back-propagate
float* x2760 = (float*)myMalloc(1 * sizeof(float));;
x2760[0] = 1.0f;

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
    64, 512, 2, 2));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2760, filt_desc, x760, grad_out_desc, x2482,
    conv_desc, algo, ws_data, ws_size,
    x2760, grad_in_desc, x2470));
};
float* x2763 = (float*)myMalloc(1 * sizeof(float));;
x2763[0] = 1.0f;

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
    64, 512, 2, 2));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2763, in_desc, x2462, grad_out_desc, x2482,
    conv_desc, algo, ws_data, ws_size,
    x2763, grad_filt_desc, x1262));
};
float* x2766 = (float*)myMalloc(1 * sizeof(float));;
x2766[0] = 1.0f;
float* x2768 = (float*)myMalloc(1 * sizeof(float));;
x2768[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x2766, x_desc, x2462, x_desc, x2470, x_desc, x2462,
    x2768, x_desc, x2470));
};
float* x2771 = (float*)myMalloc(1 * sizeof(float));;
x2771[0] = 0.0f;
float* x2773 = (float*)myMalloc(1 * sizeof(float));;
x2773[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2773, x2773, x2773, x2773, in_desc, x2455,
    out_desc, x2470, in_desc, x2461, sbmv_desc, x433,
    x1153,x1244, 1.0E-5, x2463, x2464));
};
// conv2D back-propagate
float* x2777 = (float*)myMalloc(1 * sizeof(float));;
x2777[0] = 1.0f;

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
    64, 2048, 2, 2));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2777, filt_desc, x940, grad_out_desc, x2461,
    conv_desc, algo, ws_data, ws_size,
    x2777, grad_in_desc, x2428));
};
float* x2780 = (float*)myMalloc(1 * sizeof(float));;
x2780[0] = 1.0f;

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
    64, 512, 2, 2));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2780, in_desc, x2420, grad_out_desc, x2461,
    conv_desc, algo, ws_data, ws_size,
    x2780, grad_filt_desc, x1322));
};
float* x2783 = (float*)myMalloc(1 * sizeof(float));;
x2783[0] = 1.0f;
float* x2785 = (float*)myMalloc(1 * sizeof(float));;
x2785[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x2783, x_desc, x2420, x_desc, x2428, x_desc, x2420,
    x2785, x_desc, x2428));
};
float* x2788 = (float*)myMalloc(1 * sizeof(float));;
x2788[0] = 1.0f;
float* x2790 = (float*)myMalloc(1 * sizeof(float));;
x2790[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2788, bias_desc, x2428, x2790, out_desc, x2444));
};
float* x2793 = (float*)myMalloc(1 * sizeof(float));;
x2793[0] = 0.0f;
float* x2795 = (float*)myMalloc(1 * sizeof(float));;
x2795[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2795, x2795, x2795, x2795, in_desc, x2429,
    out_desc, x2444, in_desc, x2435, sbmv_desc, x814,
    x1280,x1214, 1.0E-5, x2437, x2438));
};
// conv2D back-propagate
float* x2799 = (float*)myMalloc(1 * sizeof(float));;
x2799[0] = 1.0f;

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
    64, 1024, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 2, 2, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2799, filt_desc, x937, grad_out_desc, x2435,
    conv_desc, algo, ws_data, ws_size,
    x2799, grad_in_desc, x2360));
};
float* x2802 = (float*)myMalloc(1 * sizeof(float));;
x2802[0] = 1.0f;

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
    64, 2048, 2, 2));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 2, 2, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2802, in_desc, x2352, grad_out_desc, x2435,
    conv_desc, algo, ws_data, ws_size,
    x2802, grad_filt_desc, x1321));
};
float* x2805 = (float*)myMalloc(1 * sizeof(float));;
x2805[0] = 0.0f;
float* x2807 = (float*)myMalloc(1 * sizeof(float));;
x2807[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 2048, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2807, x2807, x2807, x2807, in_desc, x2413,
    out_desc, x2428, in_desc, x2419, sbmv_desc, x1012,
    x1346,x1169, 1.0E-5, x2421, x2422));
};
// conv2D back-propagate
float* x2811 = (float*)myMalloc(1 * sizeof(float));;
x2811[0] = 1.0f;

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
    64, 512, 2, 2));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2811, filt_desc, x931, grad_out_desc, x2419,
    conv_desc, algo, ws_data, ws_size,
    x2811, grad_in_desc, x2407));
};
float* x2814 = (float*)myMalloc(1 * sizeof(float));;
x2814[0] = 1.0f;

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
    64, 2048, 2, 2));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2814, in_desc, x2399, grad_out_desc, x2419,
    conv_desc, algo, ws_data, ws_size,
    x2814, grad_filt_desc, x1319));
};
float* x2817 = (float*)myMalloc(1 * sizeof(float));;
x2817[0] = 1.0f;
float* x2819 = (float*)myMalloc(1 * sizeof(float));;
x2819[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x2817, x_desc, x2399, x_desc, x2407, x_desc, x2399,
    x2819, x_desc, x2407));
};
float* x2822 = (float*)myMalloc(1 * sizeof(float));;
x2822[0] = 0.0f;
float* x2824 = (float*)myMalloc(1 * sizeof(float));;
x2824[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2824, x2824, x2824, x2824, in_desc, x2392,
    out_desc, x2407, in_desc, x2398, sbmv_desc, x910,
    x1312,x1266, 1.0E-5, x2400, x2401));
};
// conv2D back-propagate
float* x2828 = (float*)myMalloc(1 * sizeof(float));;
x2828[0] = 1.0f;

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
    64, 512, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 2, 2));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 2, 2, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2828, filt_desc, x397, grad_out_desc, x2398,
    conv_desc, algo, ws_data, ws_size,
    x2828, grad_in_desc, x2386));
};
float* x2831 = (float*)myMalloc(1 * sizeof(float));;
x2831[0] = 1.0f;

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
    64, 512, 2, 2));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 2, 2, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2831, in_desc, x2378, grad_out_desc, x2398,
    conv_desc, algo, ws_data, ws_size,
    x2831, grad_filt_desc, x1141));
};
float* x2834 = (float*)myMalloc(1 * sizeof(float));;
x2834[0] = 1.0f;
float* x2836 = (float*)myMalloc(1 * sizeof(float));;
x2836[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x2834, x_desc, x2378, x_desc, x2386, x_desc, x2378,
    x2836, x_desc, x2386));
};
float* x2839 = (float*)myMalloc(1 * sizeof(float));;
x2839[0] = 0.0f;
float* x2841 = (float*)myMalloc(1 * sizeof(float));;
x2841[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2841, x2841, x2841, x2841, in_desc, x2371,
    out_desc, x2386, in_desc, x2377, sbmv_desc, x898,
    x1308,x1331, 1.0E-5, x2379, x2380));
};
// conv2D back-propagate
float* x2845 = (float*)myMalloc(1 * sizeof(float));;
x2845[0] = 1.0f;

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
    64, 1024, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2845, filt_desc, x712, grad_out_desc, x2377,
    conv_desc, algo, ws_data, ws_size,
    x2845, grad_in_desc, x2360));
};
float* x2848 = (float*)myMalloc(1 * sizeof(float));;
x2848[0] = 1.0f;

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
    64, 512, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2848, in_desc, x2352, grad_out_desc, x2377,
    conv_desc, algo, ws_data, ws_size,
    x2848, grad_filt_desc, x1246));
};
float* x2851 = (float*)myMalloc(1 * sizeof(float));;
x2851[0] = 1.0f;
float* x2853 = (float*)myMalloc(1 * sizeof(float));;
x2853[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x2851, x_desc, x2352, x_desc, x2360, x_desc, x2352,
    x2853, x_desc, x2360));
};
float* x2856 = (float*)myMalloc(1 * sizeof(float));;
x2856[0] = 1.0f;
float* x2858 = (float*)myMalloc(1 * sizeof(float));;
x2858[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2856, bias_desc, x2360, x2858, out_desc, x2292));
};
float* x2861 = (float*)myMalloc(1 * sizeof(float));;
x2861[0] = 0.0f;
float* x2863 = (float*)myMalloc(1 * sizeof(float));;
x2863[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2863, x2863, x2863, x2863, in_desc, x2345,
    out_desc, x2360, in_desc, x2351, sbmv_desc, x1039,
    x1355,x1200, 1.0E-5, x2353, x2354));
};
// conv2D back-propagate
float* x2867 = (float*)myMalloc(1 * sizeof(float));;
x2867[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2867, filt_desc, x586, grad_out_desc, x2351,
    conv_desc, algo, ws_data, ws_size,
    x2867, grad_in_desc, x2339));
};
float* x2870 = (float*)myMalloc(1 * sizeof(float));;
x2870[0] = 1.0f;

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
    64, 1024, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2870, in_desc, x2331, grad_out_desc, x2351,
    conv_desc, algo, ws_data, ws_size,
    x2870, grad_filt_desc, x1204));
};
float* x2873 = (float*)myMalloc(1 * sizeof(float));;
x2873[0] = 1.0f;
float* x2875 = (float*)myMalloc(1 * sizeof(float));;
x2875[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x2873, x_desc, x2331, x_desc, x2339, x_desc, x2331,
    x2875, x_desc, x2339));
};
float* x2878 = (float*)myMalloc(1 * sizeof(float));;
x2878[0] = 0.0f;
float* x2880 = (float*)myMalloc(1 * sizeof(float));;
x2880[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2880, x2880, x2880, x2880, in_desc, x2324,
    out_desc, x2339, in_desc, x2330, sbmv_desc, x718,
    x1248,x1296, 1.0E-5, x2332, x2333));
};
// conv2D back-propagate
float* x2884 = (float*)myMalloc(1 * sizeof(float));;
x2884[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2884, filt_desc, x973, grad_out_desc, x2330,
    conv_desc, algo, ws_data, ws_size,
    x2884, grad_in_desc, x2318));
};
float* x2887 = (float*)myMalloc(1 * sizeof(float));;
x2887[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2887, in_desc, x2310, grad_out_desc, x2330,
    conv_desc, algo, ws_data, ws_size,
    x2887, grad_filt_desc, x1333));
};
float* x2890 = (float*)myMalloc(1 * sizeof(float));;
x2890[0] = 1.0f;
float* x2892 = (float*)myMalloc(1 * sizeof(float));;
x2892[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x2890, x_desc, x2310, x_desc, x2318, x_desc, x2310,
    x2892, x_desc, x2318));
};
float* x2895 = (float*)myMalloc(1 * sizeof(float));;
x2895[0] = 0.0f;
float* x2897 = (float*)myMalloc(1 * sizeof(float));;
x2897[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2897, x2897, x2897, x2897, in_desc, x2303,
    out_desc, x2318, in_desc, x2309, sbmv_desc, x550,
    x1192,x1360, 1.0E-5, x2311, x2312));
};
// conv2D back-propagate
float* x2901 = (float*)myMalloc(1 * sizeof(float));;
x2901[0] = 1.0f;

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
    64, 1024, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2901, filt_desc, x748, grad_out_desc, x2309,
    conv_desc, algo, ws_data, ws_size,
    x2901, grad_in_desc, x2292));
};
float* x2904 = (float*)myMalloc(1 * sizeof(float));;
x2904[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2904, in_desc, x2284, grad_out_desc, x2309,
    conv_desc, algo, ws_data, ws_size,
    x2904, grad_filt_desc, x1258));
};
float* x2907 = (float*)myMalloc(1 * sizeof(float));;
x2907[0] = 1.0f;
float* x2909 = (float*)myMalloc(1 * sizeof(float));;
x2909[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x2907, x_desc, x2284, x_desc, x2292, x_desc, x2284,
    x2909, x_desc, x2292));
};
float* x2912 = (float*)myMalloc(1 * sizeof(float));;
x2912[0] = 1.0f;
float* x2914 = (float*)myMalloc(1 * sizeof(float));;
x2914[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2912, bias_desc, x2292, x2914, out_desc, x2224));
};
float* x2917 = (float*)myMalloc(1 * sizeof(float));;
x2917[0] = 0.0f;
float* x2919 = (float*)myMalloc(1 * sizeof(float));;
x2919[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2919, x2919, x2919, x2919, in_desc, x2277,
    out_desc, x2292, in_desc, x2283, sbmv_desc, x472,
    x1166,x1227, 1.0E-5, x2285, x2286));
};
// conv2D back-propagate
float* x2923 = (float*)myMalloc(1 * sizeof(float));;
x2923[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2923, filt_desc, x958, grad_out_desc, x2283,
    conv_desc, algo, ws_data, ws_size,
    x2923, grad_in_desc, x2271));
};
float* x2926 = (float*)myMalloc(1 * sizeof(float));;
x2926[0] = 1.0f;

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
    64, 1024, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2926, in_desc, x2263, grad_out_desc, x2283,
    conv_desc, algo, ws_data, ws_size,
    x2926, grad_filt_desc, x1328));
};
float* x2929 = (float*)myMalloc(1 * sizeof(float));;
x2929[0] = 1.0f;
float* x2931 = (float*)myMalloc(1 * sizeof(float));;
x2931[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x2929, x_desc, x2263, x_desc, x2271, x_desc, x2263,
    x2931, x_desc, x2271));
};
float* x2934 = (float*)myMalloc(1 * sizeof(float));;
x2934[0] = 0.0f;
float* x2936 = (float*)myMalloc(1 * sizeof(float));;
x2936[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2936, x2936, x2936, x2936, in_desc, x2256,
    out_desc, x2271, in_desc, x2262, sbmv_desc, x799,
    x1275,x1216, 1.0E-5, x2264, x2265));
};
// conv2D back-propagate
float* x2940 = (float*)myMalloc(1 * sizeof(float));;
x2940[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2940, filt_desc, x1081, grad_out_desc, x2262,
    conv_desc, algo, ws_data, ws_size,
    x2940, grad_in_desc, x2250));
};
float* x2943 = (float*)myMalloc(1 * sizeof(float));;
x2943[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2943, in_desc, x2242, grad_out_desc, x2262,
    conv_desc, algo, ws_data, ws_size,
    x2943, grad_filt_desc, x1369));
};
float* x2946 = (float*)myMalloc(1 * sizeof(float));;
x2946[0] = 1.0f;
float* x2948 = (float*)myMalloc(1 * sizeof(float));;
x2948[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x2946, x_desc, x2242, x_desc, x2250, x_desc, x2242,
    x2948, x_desc, x2250));
};
float* x2951 = (float*)myMalloc(1 * sizeof(float));;
x2951[0] = 0.0f;
float* x2953 = (float*)myMalloc(1 * sizeof(float));;
x2953[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2953, x2953, x2953, x2953, in_desc, x2235,
    out_desc, x2250, in_desc, x2241, sbmv_desc, x526,
    x1184,x1292, 1.0E-5, x2243, x2244));
};
// conv2D back-propagate
float* x2957 = (float*)myMalloc(1 * sizeof(float));;
x2957[0] = 1.0f;

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
    64, 1024, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2957, filt_desc, x361, grad_out_desc, x2241,
    conv_desc, algo, ws_data, ws_size,
    x2957, grad_in_desc, x2224));
};
float* x2960 = (float*)myMalloc(1 * sizeof(float));;
x2960[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2960, in_desc, x2216, grad_out_desc, x2241,
    conv_desc, algo, ws_data, ws_size,
    x2960, grad_filt_desc, x1129));
};
float* x2963 = (float*)myMalloc(1 * sizeof(float));;
x2963[0] = 1.0f;
float* x2965 = (float*)myMalloc(1 * sizeof(float));;
x2965[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x2963, x_desc, x2216, x_desc, x2224, x_desc, x2216,
    x2965, x_desc, x2224));
};
float* x2968 = (float*)myMalloc(1 * sizeof(float));;
x2968[0] = 1.0f;
float* x2970 = (float*)myMalloc(1 * sizeof(float));;
x2970[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x2968, bias_desc, x2224, x2970, out_desc, x2156));
};
float* x2973 = (float*)myMalloc(1 * sizeof(float));;
x2973[0] = 0.0f;
float* x2975 = (float*)myMalloc(1 * sizeof(float));;
x2975[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2975, x2975, x2975, x2975, in_desc, x2209,
    out_desc, x2224, in_desc, x2215, sbmv_desc, x1009,
    x1345,x1253, 1.0E-5, x2217, x2218));
};
// conv2D back-propagate
float* x2979 = (float*)myMalloc(1 * sizeof(float));;
x2979[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2979, filt_desc, x562, grad_out_desc, x2215,
    conv_desc, algo, ws_data, ws_size,
    x2979, grad_in_desc, x2203));
};
float* x2982 = (float*)myMalloc(1 * sizeof(float));;
x2982[0] = 1.0f;

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
    64, 1024, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2982, in_desc, x2195, grad_out_desc, x2215,
    conv_desc, algo, ws_data, ws_size,
    x2982, grad_filt_desc, x1196));
};
float* x2985 = (float*)myMalloc(1 * sizeof(float));;
x2985[0] = 1.0f;
float* x2987 = (float*)myMalloc(1 * sizeof(float));;
x2987[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x2985, x_desc, x2195, x_desc, x2203, x_desc, x2195,
    x2987, x_desc, x2203));
};
float* x2990 = (float*)myMalloc(1 * sizeof(float));;
x2990[0] = 0.0f;
float* x2992 = (float*)myMalloc(1 * sizeof(float));;
x2992[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2992, x2992, x2992, x2992, in_desc, x2188,
    out_desc, x2203, in_desc, x2194, sbmv_desc, x517,
    x1181,x1243, 1.0E-5, x2196, x2197));
};
// conv2D back-propagate
float* x2996 = (float*)myMalloc(1 * sizeof(float));;
x2996[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2996, filt_desc, x1042, grad_out_desc, x2194,
    conv_desc, algo, ws_data, ws_size,
    x2996, grad_in_desc, x2182));
};
float* x2999 = (float*)myMalloc(1 * sizeof(float));;
x2999[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x2999, in_desc, x2174, grad_out_desc, x2194,
    conv_desc, algo, ws_data, ws_size,
    x2999, grad_filt_desc, x1356));
};
float* x3002 = (float*)myMalloc(1 * sizeof(float));;
x3002[0] = 1.0f;
float* x3004 = (float*)myMalloc(1 * sizeof(float));;
x3004[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3002, x_desc, x2174, x_desc, x2182, x_desc, x2174,
    x3004, x_desc, x2182));
};
float* x3007 = (float*)myMalloc(1 * sizeof(float));;
x3007[0] = 0.0f;
float* x3009 = (float*)myMalloc(1 * sizeof(float));;
x3009[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3009, x3009, x3009, x3009, in_desc, x2167,
    out_desc, x2182, in_desc, x2173, sbmv_desc, x571,
    x1199,x1348, 1.0E-5, x2175, x2176));
};
// conv2D back-propagate
float* x3013 = (float*)myMalloc(1 * sizeof(float));;
x3013[0] = 1.0f;

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
    64, 1024, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3013, filt_desc, x313, grad_out_desc, x2173,
    conv_desc, algo, ws_data, ws_size,
    x3013, grad_in_desc, x2156));
};
float* x3016 = (float*)myMalloc(1 * sizeof(float));;
x3016[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3016, in_desc, x2148, grad_out_desc, x2173,
    conv_desc, algo, ws_data, ws_size,
    x3016, grad_filt_desc, x1113));
};
float* x3019 = (float*)myMalloc(1 * sizeof(float));;
x3019[0] = 1.0f;
float* x3021 = (float*)myMalloc(1 * sizeof(float));;
x3021[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3019, x_desc, x2148, x_desc, x2156, x_desc, x2148,
    x3021, x_desc, x2156));
};
float* x3024 = (float*)myMalloc(1 * sizeof(float));;
x3024[0] = 1.0f;
float* x3026 = (float*)myMalloc(1 * sizeof(float));;
x3026[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x3024, bias_desc, x2156, x3026, out_desc, x2088));
};
float* x3029 = (float*)myMalloc(1 * sizeof(float));;
x3029[0] = 0.0f;
float* x3031 = (float*)myMalloc(1 * sizeof(float));;
x3031[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3031, x3031, x3031, x3031, in_desc, x2141,
    out_desc, x2156, in_desc, x2147, sbmv_desc, x1084,
    x1370,x1164, 1.0E-5, x2149, x2150));
};
// conv2D back-propagate
float* x3035 = (float*)myMalloc(1 * sizeof(float));;
x3035[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3035, filt_desc, x643, grad_out_desc, x2147,
    conv_desc, algo, ws_data, ws_size,
    x3035, grad_in_desc, x2135));
};
float* x3038 = (float*)myMalloc(1 * sizeof(float));;
x3038[0] = 1.0f;

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
    64, 1024, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3038, in_desc, x2127, grad_out_desc, x2147,
    conv_desc, algo, ws_data, ws_size,
    x3038, grad_filt_desc, x1223));
};
float* x3041 = (float*)myMalloc(1 * sizeof(float));;
x3041[0] = 1.0f;
float* x3043 = (float*)myMalloc(1 * sizeof(float));;
x3043[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3041, x_desc, x2127, x_desc, x2135, x_desc, x2127,
    x3043, x_desc, x2135));
};
float* x3046 = (float*)myMalloc(1 * sizeof(float));;
x3046[0] = 0.0f;
float* x3048 = (float*)myMalloc(1 * sizeof(float));;
x3048[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3048, x3048, x3048, x3048, in_desc, x2120,
    out_desc, x2135, in_desc, x2126, sbmv_desc, x979,
    x1335,x1299, 1.0E-5, x2128, x2129));
};
// conv2D back-propagate
float* x3052 = (float*)myMalloc(1 * sizeof(float));;
x3052[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3052, filt_desc, x337, grad_out_desc, x2126,
    conv_desc, algo, ws_data, ws_size,
    x3052, grad_in_desc, x2114));
};
float* x3055 = (float*)myMalloc(1 * sizeof(float));;
x3055[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3055, in_desc, x2106, grad_out_desc, x2126,
    conv_desc, algo, ws_data, ws_size,
    x3055, grad_filt_desc, x1121));
};
float* x3058 = (float*)myMalloc(1 * sizeof(float));;
x3058[0] = 1.0f;
float* x3060 = (float*)myMalloc(1 * sizeof(float));;
x3060[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3058, x_desc, x2106, x_desc, x2114, x_desc, x2106,
    x3060, x_desc, x2114));
};
float* x3063 = (float*)myMalloc(1 * sizeof(float));;
x3063[0] = 0.0f;
float* x3065 = (float*)myMalloc(1 * sizeof(float));;
x3065[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3065, x3065, x3065, x3065, in_desc, x2099,
    out_desc, x2114, in_desc, x2105, sbmv_desc, x682,
    x1236,x1304, 1.0E-5, x2107, x2108));
};
// conv2D back-propagate
float* x3069 = (float*)myMalloc(1 * sizeof(float));;
x3069[0] = 1.0f;

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
    64, 1024, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3069, filt_desc, x949, grad_out_desc, x2105,
    conv_desc, algo, ws_data, ws_size,
    x3069, grad_in_desc, x2088));
};
float* x3072 = (float*)myMalloc(1 * sizeof(float));;
x3072[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3072, in_desc, x2080, grad_out_desc, x2105,
    conv_desc, algo, ws_data, ws_size,
    x3072, grad_filt_desc, x1325));
};
float* x3075 = (float*)myMalloc(1 * sizeof(float));;
x3075[0] = 1.0f;
float* x3077 = (float*)myMalloc(1 * sizeof(float));;
x3077[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3075, x_desc, x2080, x_desc, x2088, x_desc, x2080,
    x3077, x_desc, x2088));
};
float* x3080 = (float*)myMalloc(1 * sizeof(float));;
x3080[0] = 1.0f;
float* x3082 = (float*)myMalloc(1 * sizeof(float));;
x3082[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x3080, bias_desc, x2088, x3082, out_desc, x2004));
};
float* x3085 = (float*)myMalloc(1 * sizeof(float));;
x3085[0] = 0.0f;
float* x3087 = (float*)myMalloc(1 * sizeof(float));;
x3087[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3087, x3087, x3087, x3087, in_desc, x2073,
    out_desc, x2088, in_desc, x2079, sbmv_desc, x355,
    x1127,x1339, 1.0E-5, x2081, x2082));
};
// conv2D back-propagate
float* x3091 = (float*)myMalloc(1 * sizeof(float));;
x3091[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3091, filt_desc, x463, grad_out_desc, x2079,
    conv_desc, algo, ws_data, ws_size,
    x3091, grad_in_desc, x2067));
};
float* x3094 = (float*)myMalloc(1 * sizeof(float));;
x3094[0] = 1.0f;

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
    64, 1024, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3094, in_desc, x2059, grad_out_desc, x2079,
    conv_desc, algo, ws_data, ws_size,
    x3094, grad_filt_desc, x1163));
};
float* x3097 = (float*)myMalloc(1 * sizeof(float));;
x3097[0] = 1.0f;
float* x3099 = (float*)myMalloc(1 * sizeof(float));;
x3099[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3097, x_desc, x2059, x_desc, x2067, x_desc, x2059,
    x3099, x_desc, x2067));
};
float* x3102 = (float*)myMalloc(1 * sizeof(float));;
x3102[0] = 0.0f;
float* x3104 = (float*)myMalloc(1 * sizeof(float));;
x3104[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3104, x3104, x3104, x3104, in_desc, x2052,
    out_desc, x2067, in_desc, x2058, sbmv_desc, x1108,
    x1378,x1203, 1.0E-5, x2060, x2061));
};
// conv2D back-propagate
float* x3108 = (float*)myMalloc(1 * sizeof(float));;
x3108[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3108, filt_desc, x388, grad_out_desc, x2058,
    conv_desc, algo, ws_data, ws_size,
    x3108, grad_in_desc, x2046));
};
float* x3111 = (float*)myMalloc(1 * sizeof(float));;
x3111[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3111, in_desc, x2038, grad_out_desc, x2058,
    conv_desc, algo, ws_data, ws_size,
    x3111, grad_filt_desc, x1138));
};
float* x3114 = (float*)myMalloc(1 * sizeof(float));;
x3114[0] = 1.0f;
float* x3116 = (float*)myMalloc(1 * sizeof(float));;
x3116[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3114, x_desc, x2038, x_desc, x2046, x_desc, x2038,
    x3116, x_desc, x2046));
};
float* x3119 = (float*)myMalloc(1 * sizeof(float));;
x3119[0] = 0.0f;
float* x3121 = (float*)myMalloc(1 * sizeof(float));;
x3121[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3121, x3121, x3121, x3121, in_desc, x2031,
    out_desc, x2046, in_desc, x2037, sbmv_desc, x385,
    x1137,x1326, 1.0E-5, x2039, x2040));
};
// conv2D back-propagate
float* x3125 = (float*)myMalloc(1 * sizeof(float));;
x3125[0] = 1.0f;

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
    64, 1024, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3125, filt_desc, x334, grad_out_desc, x2037,
    conv_desc, algo, ws_data, ws_size,
    x3125, grad_in_desc, x2004));
};
float* x3128 = (float*)myMalloc(1 * sizeof(float));;
x3128[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3128, in_desc, x1996, grad_out_desc, x2037,
    conv_desc, algo, ws_data, ws_size,
    x3128, grad_filt_desc, x1120));
};
float* x3131 = (float*)myMalloc(1 * sizeof(float));;
x3131[0] = 1.0f;
float* x3133 = (float*)myMalloc(1 * sizeof(float));;
x3133[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3131, x_desc, x1996, x_desc, x2004, x_desc, x1996,
    x3133, x_desc, x2004));
};
float* x3136 = (float*)myMalloc(1 * sizeof(float));;
x3136[0] = 1.0f;
float* x3138 = (float*)myMalloc(1 * sizeof(float));;
x3138[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x3136, bias_desc, x2004, x3138, out_desc, x2020));
};
float* x3141 = (float*)myMalloc(1 * sizeof(float));;
x3141[0] = 0.0f;
float* x3143 = (float*)myMalloc(1 * sizeof(float));;
x3143[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3143, x3143, x3143, x3143, in_desc, x2005,
    out_desc, x2020, in_desc, x2011, sbmv_desc, x382,
    x1136,x1327, 1.0E-5, x2013, x2014));
};
// conv2D back-propagate
float* x3147 = (float*)myMalloc(1 * sizeof(float));;
x3147[0] = 1.0f;

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
    64, 512, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 2, 2, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3147, filt_desc, x520, grad_out_desc, x2011,
    conv_desc, algo, ws_data, ws_size,
    x3147, grad_in_desc, x1936));
};
float* x3150 = (float*)myMalloc(1 * sizeof(float));;
x3150[0] = 1.0f;

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
    64, 1024, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 2, 2, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3150, in_desc, x1928, grad_out_desc, x2011,
    conv_desc, algo, ws_data, ws_size,
    x3150, grad_filt_desc, x1182));
};
float* x3153 = (float*)myMalloc(1 * sizeof(float));;
x3153[0] = 0.0f;
float* x3155 = (float*)myMalloc(1 * sizeof(float));;
x3155[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3155, x3155, x3155, x3155, in_desc, x1989,
    out_desc, x2004, in_desc, x1995, sbmv_desc, x349,
    x1125,x1224, 1.0E-5, x1997, x1998));
};
// conv2D back-propagate
float* x3159 = (float*)myMalloc(1 * sizeof(float));;
x3159[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3159, filt_desc, x1102, grad_out_desc, x1995,
    conv_desc, algo, ws_data, ws_size,
    x3159, grad_in_desc, x1983));
};
float* x3162 = (float*)myMalloc(1 * sizeof(float));;
x3162[0] = 1.0f;

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
    64, 1024, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3162, in_desc, x1975, grad_out_desc, x1995,
    conv_desc, algo, ws_data, ws_size,
    x3162, grad_filt_desc, x1376));
};
float* x3165 = (float*)myMalloc(1 * sizeof(float));;
x3165[0] = 1.0f;
float* x3167 = (float*)myMalloc(1 * sizeof(float));;
x3167[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3165, x_desc, x1975, x_desc, x1983, x_desc, x1975,
    x3167, x_desc, x1983));
};
float* x3170 = (float*)myMalloc(1 * sizeof(float));;
x3170[0] = 0.0f;
float* x3172 = (float*)myMalloc(1 * sizeof(float));;
x3172[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3172, x3172, x3172, x3172, in_desc, x1968,
    out_desc, x1983, in_desc, x1974, sbmv_desc, x619,
    x1215,x1123, 1.0E-5, x1976, x1977));
};
// conv2D back-propagate
float* x3176 = (float*)myMalloc(1 * sizeof(float));;
x3176[0] = 1.0f;

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
    64, 256, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 2, 2, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3176, filt_desc, x820, grad_out_desc, x1974,
    conv_desc, algo, ws_data, ws_size,
    x3176, grad_in_desc, x1962));
};
float* x3179 = (float*)myMalloc(1 * sizeof(float));;
x3179[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 2, 2, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3179, in_desc, x1954, grad_out_desc, x1974,
    conv_desc, algo, ws_data, ws_size,
    x3179, grad_filt_desc, x1282));
};
float* x3182 = (float*)myMalloc(1 * sizeof(float));;
x3182[0] = 1.0f;
float* x3184 = (float*)myMalloc(1 * sizeof(float));;
x3184[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3182, x_desc, x1954, x_desc, x1962, x_desc, x1954,
    x3184, x_desc, x1962));
};
float* x3187 = (float*)myMalloc(1 * sizeof(float));;
x3187[0] = 0.0f;
float* x3189 = (float*)myMalloc(1 * sizeof(float));;
x3189[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3189, x3189, x3189, x3189, in_desc, x1947,
    out_desc, x1962, in_desc, x1953, sbmv_desc, x1105,
    x1377,x1128, 1.0E-5, x1955, x1956));
};
// conv2D back-propagate
float* x3193 = (float*)myMalloc(1 * sizeof(float));;
x3193[0] = 1.0f;

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
    64, 512, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3193, filt_desc, x835, grad_out_desc, x1953,
    conv_desc, algo, ws_data, ws_size,
    x3193, grad_in_desc, x1936));
};
float* x3196 = (float*)myMalloc(1 * sizeof(float));;
x3196[0] = 1.0f;

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
    64, 256, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3196, in_desc, x1928, grad_out_desc, x1953,
    conv_desc, algo, ws_data, ws_size,
    x3196, grad_filt_desc, x1287));
};
float* x3199 = (float*)myMalloc(1 * sizeof(float));;
x3199[0] = 1.0f;
float* x3201 = (float*)myMalloc(1 * sizeof(float));;
x3201[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3199, x_desc, x1928, x_desc, x1936, x_desc, x1928,
    x3201, x_desc, x1936));
};
float* x3204 = (float*)myMalloc(1 * sizeof(float));;
x3204[0] = 1.0f;
float* x3206 = (float*)myMalloc(1 * sizeof(float));;
x3206[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x3204, bias_desc, x1936, x3206, out_desc, x1868));
};
float* x3209 = (float*)myMalloc(1 * sizeof(float));;
x3209[0] = 0.0f;
float* x3211 = (float*)myMalloc(1 * sizeof(float));;
x3211[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3211, x3211, x3211, x3211, in_desc, x1921,
    out_desc, x1936, in_desc, x1927, sbmv_desc, x763,
    x1263,x1161, 1.0E-5, x1929, x1930));
};
// conv2D back-propagate
float* x3215 = (float*)myMalloc(1 * sizeof(float));;
x3215[0] = 1.0f;

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
    64, 128, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3215, filt_desc, x460, grad_out_desc, x1927,
    conv_desc, algo, ws_data, ws_size,
    x3215, grad_in_desc, x1915));
};
float* x3218 = (float*)myMalloc(1 * sizeof(float));;
x3218[0] = 1.0f;

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
    64, 512, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3218, in_desc, x1907, grad_out_desc, x1927,
    conv_desc, algo, ws_data, ws_size,
    x3218, grad_filt_desc, x1162));
};
float* x3221 = (float*)myMalloc(1 * sizeof(float));;
x3221[0] = 1.0f;
float* x3223 = (float*)myMalloc(1 * sizeof(float));;
x3223[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3221, x_desc, x1907, x_desc, x1915, x_desc, x1907,
    x3223, x_desc, x1915));
};
float* x3226 = (float*)myMalloc(1 * sizeof(float));;
x3226[0] = 0.0f;
float* x3228 = (float*)myMalloc(1 * sizeof(float));;
x3228[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3228, x3228, x3228, x3228, in_desc, x1900,
    out_desc, x1915, in_desc, x1906, sbmv_desc, x532,
    x1186,x1145, 1.0E-5, x1908, x1909));
};
// conv2D back-propagate
float* x3232 = (float*)myMalloc(1 * sizeof(float));;
x3232[0] = 1.0f;

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
    64, 128, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3232, filt_desc, x790, grad_out_desc, x1906,
    conv_desc, algo, ws_data, ws_size,
    x3232, grad_in_desc, x1894));
};
float* x3235 = (float*)myMalloc(1 * sizeof(float));;
x3235[0] = 1.0f;

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
    64, 128, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3235, in_desc, x1886, grad_out_desc, x1906,
    conv_desc, algo, ws_data, ws_size,
    x3235, grad_filt_desc, x1272));
};
float* x3238 = (float*)myMalloc(1 * sizeof(float));;
x3238[0] = 1.0f;
float* x3240 = (float*)myMalloc(1 * sizeof(float));;
x3240[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3238, x_desc, x1886, x_desc, x1894, x_desc, x1886,
    x3240, x_desc, x1894));
};
float* x3243 = (float*)myMalloc(1 * sizeof(float));;
x3243[0] = 0.0f;
float* x3245 = (float*)myMalloc(1 * sizeof(float));;
x3245[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3245, x3245, x3245, x3245, in_desc, x1879,
    out_desc, x1894, in_desc, x1885, sbmv_desc, x412,
    x1146,x1349, 1.0E-5, x1887, x1888));
};
// conv2D back-propagate
float* x3249 = (float*)myMalloc(1 * sizeof(float));;
x3249[0] = 1.0f;

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
    64, 512, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3249, filt_desc, x691, grad_out_desc, x1885,
    conv_desc, algo, ws_data, ws_size,
    x3249, grad_in_desc, x1868));
};
float* x3252 = (float*)myMalloc(1 * sizeof(float));;
x3252[0] = 1.0f;

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
    64, 128, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3252, in_desc, x1860, grad_out_desc, x1885,
    conv_desc, algo, ws_data, ws_size,
    x3252, grad_filt_desc, x1239));
};
float* x3255 = (float*)myMalloc(1 * sizeof(float));;
x3255[0] = 1.0f;
float* x3257 = (float*)myMalloc(1 * sizeof(float));;
x3257[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3255, x_desc, x1860, x_desc, x1868, x_desc, x1860,
    x3257, x_desc, x1868));
};
float* x3260 = (float*)myMalloc(1 * sizeof(float));;
x3260[0] = 1.0f;
float* x3262 = (float*)myMalloc(1 * sizeof(float));;
x3262[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x3260, bias_desc, x1868, x3262, out_desc, x1800));
};
float* x3265 = (float*)myMalloc(1 * sizeof(float));;
x3265[0] = 0.0f;
float* x3267 = (float*)myMalloc(1 * sizeof(float));;
x3267[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3267, x3267, x3267, x3267, in_desc, x1853,
    out_desc, x1868, in_desc, x1859, sbmv_desc, x796,
    x1274,x1189, 1.0E-5, x1861, x1862));
};
// conv2D back-propagate
float* x3271 = (float*)myMalloc(1 * sizeof(float));;
x3271[0] = 1.0f;

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
    64, 128, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3271, filt_desc, x418, grad_out_desc, x1859,
    conv_desc, algo, ws_data, ws_size,
    x3271, grad_in_desc, x1847));
};
float* x3274 = (float*)myMalloc(1 * sizeof(float));;
x3274[0] = 1.0f;

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
    64, 512, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3274, in_desc, x1839, grad_out_desc, x1859,
    conv_desc, algo, ws_data, ws_size,
    x3274, grad_filt_desc, x1148));
};
float* x3277 = (float*)myMalloc(1 * sizeof(float));;
x3277[0] = 1.0f;
float* x3279 = (float*)myMalloc(1 * sizeof(float));;
x3279[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3277, x_desc, x1839, x_desc, x1847, x_desc, x1839,
    x3279, x_desc, x1847));
};
float* x3282 = (float*)myMalloc(1 * sizeof(float));;
x3282[0] = 0.0f;
float* x3284 = (float*)myMalloc(1 * sizeof(float));;
x3284[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3284, x3284, x3284, x3284, in_desc, x1832,
    out_desc, x1847, in_desc, x1838, sbmv_desc, x676,
    x1234,x1168, 1.0E-5, x1840, x1841));
};
// conv2D back-propagate
float* x3288 = (float*)myMalloc(1 * sizeof(float));;
x3288[0] = 1.0f;

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
    64, 128, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3288, filt_desc, x868, grad_out_desc, x1838,
    conv_desc, algo, ws_data, ws_size,
    x3288, grad_in_desc, x1826));
};
float* x3291 = (float*)myMalloc(1 * sizeof(float));;
x3291[0] = 1.0f;

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
    64, 128, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3291, in_desc, x1818, grad_out_desc, x1838,
    conv_desc, algo, ws_data, ws_size,
    x3291, grad_filt_desc, x1298));
};
float* x3294 = (float*)myMalloc(1 * sizeof(float));;
x3294[0] = 1.0f;
float* x3296 = (float*)myMalloc(1 * sizeof(float));;
x3296[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3294, x_desc, x1818, x_desc, x1826, x_desc, x1818,
    x3296, x_desc, x1826));
};
float* x3299 = (float*)myMalloc(1 * sizeof(float));;
x3299[0] = 0.0f;
float* x3301 = (float*)myMalloc(1 * sizeof(float));;
x3301[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3301, x3301, x3301, x3301, in_desc, x1811,
    out_desc, x1826, in_desc, x1817, sbmv_desc, x430,
    x1152,x1277, 1.0E-5, x1819, x1820));
};
// conv2D back-propagate
float* x3305 = (float*)myMalloc(1 * sizeof(float));;
x3305[0] = 1.0f;

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
    64, 512, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3305, filt_desc, x883, grad_out_desc, x1817,
    conv_desc, algo, ws_data, ws_size,
    x3305, grad_in_desc, x1800));
};
float* x3308 = (float*)myMalloc(1 * sizeof(float));;
x3308[0] = 1.0f;

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
    64, 128, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3308, in_desc, x1792, grad_out_desc, x1817,
    conv_desc, algo, ws_data, ws_size,
    x3308, grad_filt_desc, x1303));
};
float* x3311 = (float*)myMalloc(1 * sizeof(float));;
x3311[0] = 1.0f;
float* x3313 = (float*)myMalloc(1 * sizeof(float));;
x3313[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3311, x_desc, x1792, x_desc, x1800, x_desc, x1792,
    x3313, x_desc, x1800));
};
float* x3316 = (float*)myMalloc(1 * sizeof(float));;
x3316[0] = 1.0f;
float* x3318 = (float*)myMalloc(1 * sizeof(float));;
x3318[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x3316, bias_desc, x1800, x3318, out_desc, x1716));
};
float* x3321 = (float*)myMalloc(1 * sizeof(float));;
x3321[0] = 0.0f;
float* x3323 = (float*)myMalloc(1 * sizeof(float));;
x3323[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3323, x3323, x3323, x3323, in_desc, x1785,
    out_desc, x1800, in_desc, x1791, sbmv_desc, x451,
    x1159,x1353, 1.0E-5, x1793, x1794));
};
// conv2D back-propagate
float* x3327 = (float*)myMalloc(1 * sizeof(float));;
x3327[0] = 1.0f;

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
    64, 128, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3327, filt_desc, x628, grad_out_desc, x1791,
    conv_desc, algo, ws_data, ws_size,
    x3327, grad_in_desc, x1779));
};
float* x3330 = (float*)myMalloc(1 * sizeof(float));;
x3330[0] = 1.0f;

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
    64, 512, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3330, in_desc, x1771, grad_out_desc, x1791,
    conv_desc, algo, ws_data, ws_size,
    x3330, grad_filt_desc, x1218));
};
float* x3333 = (float*)myMalloc(1 * sizeof(float));;
x3333[0] = 1.0f;
float* x3335 = (float*)myMalloc(1 * sizeof(float));;
x3335[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3333, x_desc, x1771, x_desc, x1779, x_desc, x1771,
    x3335, x_desc, x1779));
};
float* x3338 = (float*)myMalloc(1 * sizeof(float));;
x3338[0] = 0.0f;
float* x3340 = (float*)myMalloc(1 * sizeof(float));;
x3340[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3340, x3340, x3340, x3340, in_desc, x1764,
    out_desc, x1779, in_desc, x1770, sbmv_desc, x319,
    x1115,x1202, 1.0E-5, x1772, x1773));
};
// conv2D back-propagate
float* x3344 = (float*)myMalloc(1 * sizeof(float));;
x3344[0] = 1.0f;

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
    64, 128, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3344, filt_desc, x1000, grad_out_desc, x1770,
    conv_desc, algo, ws_data, ws_size,
    x3344, grad_in_desc, x1758));
};
float* x3347 = (float*)myMalloc(1 * sizeof(float));;
x3347[0] = 1.0f;

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
    64, 128, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3347, in_desc, x1750, grad_out_desc, x1770,
    conv_desc, algo, ws_data, ws_size,
    x3347, grad_filt_desc, x1342));
};
float* x3350 = (float*)myMalloc(1 * sizeof(float));;
x3350[0] = 1.0f;
float* x3352 = (float*)myMalloc(1 * sizeof(float));;
x3352[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3350, x_desc, x1750, x_desc, x1758, x_desc, x1750,
    x3352, x_desc, x1758));
};
float* x3355 = (float*)myMalloc(1 * sizeof(float));;
x3355[0] = 0.0f;
float* x3357 = (float*)myMalloc(1 * sizeof(float));;
x3357[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3357, x3357, x3357, x3357, in_desc, x1743,
    out_desc, x1758, in_desc, x1749, sbmv_desc, x961,
    x1329,x1124, 1.0E-5, x1751, x1752));
};
// conv2D back-propagate
float* x3361 = (float*)myMalloc(1 * sizeof(float));;
x3361[0] = 1.0f;

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
    64, 512, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3361, filt_desc, x1063, grad_out_desc, x1749,
    conv_desc, algo, ws_data, ws_size,
    x3361, grad_in_desc, x1716));
};
float* x3364 = (float*)myMalloc(1 * sizeof(float));;
x3364[0] = 1.0f;

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
    64, 128, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3364, in_desc, x1708, grad_out_desc, x1749,
    conv_desc, algo, ws_data, ws_size,
    x3364, grad_filt_desc, x1363));
};
float* x3367 = (float*)myMalloc(1 * sizeof(float));;
x3367[0] = 1.0f;
float* x3369 = (float*)myMalloc(1 * sizeof(float));;
x3369[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3367, x_desc, x1708, x_desc, x1716, x_desc, x1708,
    x3369, x_desc, x1716));
};
float* x3372 = (float*)myMalloc(1 * sizeof(float));;
x3372[0] = 1.0f;
float* x3374 = (float*)myMalloc(1 * sizeof(float));;
x3374[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x3372, bias_desc, x1716, x3374, out_desc, x1732));
};
float* x3377 = (float*)myMalloc(1 * sizeof(float));;
x3377[0] = 0.0f;
float* x3379 = (float*)myMalloc(1 * sizeof(float));;
x3379[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3379, x3379, x3379, x3379, in_desc, x1717,
    out_desc, x1732, in_desc, x1723, sbmv_desc, x916,
    x1314,x1226, 1.0E-5, x1725, x1726));
};
// conv2D back-propagate
float* x3383 = (float*)myMalloc(1 * sizeof(float));;
x3383[0] = 1.0f;

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
    64, 256, 16, 16));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 2, 2, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3383, filt_desc, x1069, grad_out_desc, x1723,
    conv_desc, algo, ws_data, ws_size,
    x3383, grad_in_desc, x1648));
};
float* x3386 = (float*)myMalloc(1 * sizeof(float));;
x3386[0] = 1.0f;

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
    64, 512, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 2, 2, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3386, in_desc, x1640, grad_out_desc, x1723,
    conv_desc, algo, ws_data, ws_size,
    x3386, grad_filt_desc, x1365));
};
float* x3389 = (float*)myMalloc(1 * sizeof(float));;
x3389[0] = 0.0f;
float* x3391 = (float*)myMalloc(1 * sizeof(float));;
x3391[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 512, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3391, x3391, x3391, x3391, in_desc, x1701,
    out_desc, x1716, in_desc, x1707, sbmv_desc, x730,
    x1252,x1317, 1.0E-5, x1709, x1710));
};
// conv2D back-propagate
float* x3395 = (float*)myMalloc(1 * sizeof(float));;
x3395[0] = 1.0f;

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
    64, 128, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3395, filt_desc, x613, grad_out_desc, x1707,
    conv_desc, algo, ws_data, ws_size,
    x3395, grad_in_desc, x1695));
};
float* x3398 = (float*)myMalloc(1 * sizeof(float));;
x3398[0] = 1.0f;

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
    64, 512, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3398, in_desc, x1687, grad_out_desc, x1707,
    conv_desc, algo, ws_data, ws_size,
    x3398, grad_filt_desc, x1213));
};
float* x3401 = (float*)myMalloc(1 * sizeof(float));;
x3401[0] = 1.0f;
float* x3403 = (float*)myMalloc(1 * sizeof(float));;
x3403[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3401, x_desc, x1687, x_desc, x1695, x_desc, x1687,
    x3403, x_desc, x1695));
};
float* x3406 = (float*)myMalloc(1 * sizeof(float));;
x3406[0] = 0.0f;
float* x3408 = (float*)myMalloc(1 * sizeof(float));;
x3408[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3408, x3408, x3408, x3408, in_desc, x1680,
    out_desc, x1695, in_desc, x1686, sbmv_desc, x1051,
    x1359,x1297, 1.0E-5, x1688, x1689));
};
// conv2D back-propagate
float* x3412 = (float*)myMalloc(1 * sizeof(float));;
x3412[0] = 1.0f;

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
    64, 128, 16, 16));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 2, 2, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3412, filt_desc, x376, grad_out_desc, x1686,
    conv_desc, algo, ws_data, ws_size,
    x3412, grad_in_desc, x1674));
};
float* x3415 = (float*)myMalloc(1 * sizeof(float));;
x3415[0] = 1.0f;

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
    64, 128, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 2, 2, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3415, in_desc, x1666, grad_out_desc, x1686,
    conv_desc, algo, ws_data, ws_size,
    x3415, grad_filt_desc, x1134));
};
float* x3418 = (float*)myMalloc(1 * sizeof(float));;
x3418[0] = 1.0f;
float* x3420 = (float*)myMalloc(1 * sizeof(float));;
x3420[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3418, x_desc, x1666, x_desc, x1674, x_desc, x1666,
    x3420, x_desc, x1674));
};
float* x3423 = (float*)myMalloc(1 * sizeof(float));;
x3423[0] = 0.0f;
float* x3425 = (float*)myMalloc(1 * sizeof(float));;
x3425[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3425, x3425, x3425, x3425, in_desc, x1659,
    out_desc, x1674, in_desc, x1665, sbmv_desc, x547,
    x1191,x1279, 1.0E-5, x1667, x1668));
};
// conv2D back-propagate
float* x3429 = (float*)myMalloc(1 * sizeof(float));;
x3429[0] = 1.0f;

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
    64, 256, 16, 16));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3429, filt_desc, x328, grad_out_desc, x1665,
    conv_desc, algo, ws_data, ws_size,
    x3429, grad_in_desc, x1648));
};
float* x3432 = (float*)myMalloc(1 * sizeof(float));;
x3432[0] = 1.0f;

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
    64, 128, 16, 16));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3432, in_desc, x1640, grad_out_desc, x1665,
    conv_desc, algo, ws_data, ws_size,
    x3432, grad_filt_desc, x1118));
};
float* x3435 = (float*)myMalloc(1 * sizeof(float));;
x3435[0] = 1.0f;
float* x3437 = (float*)myMalloc(1 * sizeof(float));;
x3437[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3435, x_desc, x1640, x_desc, x1648, x_desc, x1640,
    x3437, x_desc, x1648));
};
float* x3440 = (float*)myMalloc(1 * sizeof(float));;
x3440[0] = 1.0f;
float* x3442 = (float*)myMalloc(1 * sizeof(float));;
x3442[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x3440, bias_desc, x1648, x3442, out_desc, x1580));
};
float* x3445 = (float*)myMalloc(1 * sizeof(float));;
x3445[0] = 0.0f;
float* x3447 = (float*)myMalloc(1 * sizeof(float));;
x3447[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3447, x3447, x3447, x3447, in_desc, x1633,
    out_desc, x1648, in_desc, x1639, sbmv_desc, x406,
    x1144,x1354, 1.0E-5, x1641, x1642));
};
// conv2D back-propagate
float* x3451 = (float*)myMalloc(1 * sizeof(float));;
x3451[0] = 1.0f;

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
    64, 64, 16, 16));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3451, filt_desc, x556, grad_out_desc, x1639,
    conv_desc, algo, ws_data, ws_size,
    x3451, grad_in_desc, x1627));
};
float* x3454 = (float*)myMalloc(1 * sizeof(float));;
x3454[0] = 1.0f;

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
    64, 256, 16, 16));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3454, in_desc, x1619, grad_out_desc, x1639,
    conv_desc, algo, ws_data, ws_size,
    x3454, grad_filt_desc, x1194));
};
float* x3457 = (float*)myMalloc(1 * sizeof(float));;
x3457[0] = 1.0f;
float* x3459 = (float*)myMalloc(1 * sizeof(float));;
x3459[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3457, x_desc, x1619, x_desc, x1627, x_desc, x1619,
    x3459, x_desc, x1627));
};
float* x3462 = (float*)myMalloc(1 * sizeof(float));;
x3462[0] = 0.0f;
float* x3464 = (float*)myMalloc(1 * sizeof(float));;
x3464[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3464, x3464, x3464, x3464, in_desc, x1612,
    out_desc, x1627, in_desc, x1618, sbmv_desc, x511,
    x1179,x1242, 1.0E-5, x1620, x1621));
};
// conv2D back-propagate
float* x3468 = (float*)myMalloc(1 * sizeof(float));;
x3468[0] = 1.0f;

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
    64, 64, 16, 16));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3468, filt_desc, x514, grad_out_desc, x1618,
    conv_desc, algo, ws_data, ws_size,
    x3468, grad_in_desc, x1606));
};
float* x3471 = (float*)myMalloc(1 * sizeof(float));;
x3471[0] = 1.0f;

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
    64, 64, 16, 16));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3471, in_desc, x1598, grad_out_desc, x1618,
    conv_desc, algo, ws_data, ws_size,
    x3471, grad_filt_desc, x1180));
};
float* x3474 = (float*)myMalloc(1 * sizeof(float));;
x3474[0] = 1.0f;
float* x3476 = (float*)myMalloc(1 * sizeof(float));;
x3476[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3474, x_desc, x1598, x_desc, x1606, x_desc, x1598,
    x3476, x_desc, x1606));
};
float* x3479 = (float*)myMalloc(1 * sizeof(float));;
x3479[0] = 0.0f;
float* x3481 = (float*)myMalloc(1 * sizeof(float));;
x3481[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3481, x3481, x3481, x3481, in_desc, x1591,
    out_desc, x1606, in_desc, x1597, sbmv_desc, x538,
    x1188,x1131, 1.0E-5, x1599, x1600));
};
// conv2D back-propagate
float* x3485 = (float*)myMalloc(1 * sizeof(float));;
x3485[0] = 1.0f;

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
    64, 256, 16, 16));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3485, filt_desc, x745, grad_out_desc, x1597,
    conv_desc, algo, ws_data, ws_size,
    x3485, grad_in_desc, x1580));
};
float* x3488 = (float*)myMalloc(1 * sizeof(float));;
x3488[0] = 1.0f;

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
    64, 64, 16, 16));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3488, in_desc, x1572, grad_out_desc, x1597,
    conv_desc, algo, ws_data, ws_size,
    x3488, grad_filt_desc, x1257));
};
float* x3491 = (float*)myMalloc(1 * sizeof(float));;
x3491[0] = 1.0f;
float* x3493 = (float*)myMalloc(1 * sizeof(float));;
x3493[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3491, x_desc, x1572, x_desc, x1580, x_desc, x1572,
    x3493, x_desc, x1580));
};
float* x3496 = (float*)myMalloc(1 * sizeof(float));;
x3496[0] = 1.0f;
float* x3498 = (float*)myMalloc(1 * sizeof(float));;
x3498[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x3496, bias_desc, x1580, x3498, out_desc, x1496));
};
float* x3501 = (float*)myMalloc(1 * sizeof(float));;
x3501[0] = 0.0f;
float* x3503 = (float*)myMalloc(1 * sizeof(float));;
x3503[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3503, x3503, x3503, x3503, in_desc, x1565,
    out_desc, x1580, in_desc, x1571, sbmv_desc, x469,
    x1165,x1114, 1.0E-5, x1573, x1574));
};
// conv2D back-propagate
float* x3507 = (float*)myMalloc(1 * sizeof(float));;
x3507[0] = 1.0f;

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
    64, 64, 16, 16));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3507, filt_desc, x685, grad_out_desc, x1571,
    conv_desc, algo, ws_data, ws_size,
    x3507, grad_in_desc, x1559));
};
float* x3510 = (float*)myMalloc(1 * sizeof(float));;
x3510[0] = 1.0f;

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
    64, 256, 16, 16));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3510, in_desc, x1551, grad_out_desc, x1571,
    conv_desc, algo, ws_data, ws_size,
    x3510, grad_filt_desc, x1237));
};
float* x3513 = (float*)myMalloc(1 * sizeof(float));;
x3513[0] = 1.0f;
float* x3515 = (float*)myMalloc(1 * sizeof(float));;
x3515[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3513, x_desc, x1551, x_desc, x1559, x_desc, x1551,
    x3515, x_desc, x1559));
};
float* x3518 = (float*)myMalloc(1 * sizeof(float));;
x3518[0] = 0.0f;
float* x3520 = (float*)myMalloc(1 * sizeof(float));;
x3520[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3520, x3520, x3520, x3520, in_desc, x1544,
    out_desc, x1559, in_desc, x1550, sbmv_desc, x919,
    x1315,x1260, 1.0E-5, x1552, x1553));
};
// conv2D back-propagate
float* x3524 = (float*)myMalloc(1 * sizeof(float));;
x3524[0] = 1.0f;

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
    64, 64, 16, 16));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3524, filt_desc, x544, grad_out_desc, x1550,
    conv_desc, algo, ws_data, ws_size,
    x3524, grad_in_desc, x1538));
};
float* x3527 = (float*)myMalloc(1 * sizeof(float));;
x3527[0] = 1.0f;

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
    64, 64, 16, 16));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3527, in_desc, x1530, grad_out_desc, x1550,
    conv_desc, algo, ws_data, ws_size,
    x3527, grad_filt_desc, x1190));
};
float* x3530 = (float*)myMalloc(1 * sizeof(float));;
x3530[0] = 1.0f;
float* x3532 = (float*)myMalloc(1 * sizeof(float));;
x3532[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3530, x_desc, x1530, x_desc, x1538, x_desc, x1530,
    x3532, x_desc, x1538));
};
float* x3535 = (float*)myMalloc(1 * sizeof(float));;
x3535[0] = 0.0f;
float* x3537 = (float*)myMalloc(1 * sizeof(float));;
x3537[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3537, x3537, x3537, x3537, in_desc, x1523,
    out_desc, x1538, in_desc, x1529, sbmv_desc, x721,
    x1249,x1167, 1.0E-5, x1531, x1532));
};
// conv2D back-propagate
float* x3541 = (float*)myMalloc(1 * sizeof(float));;
x3541[0] = 1.0f;

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
    64, 256, 16, 16));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3541, filt_desc, x808, grad_out_desc, x1529,
    conv_desc, algo, ws_data, ws_size,
    x3541, grad_in_desc, x1496));
};
float* x3544 = (float*)myMalloc(1 * sizeof(float));;
x3544[0] = 1.0f;

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
    64, 64, 16, 16));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3544, in_desc, x1488, grad_out_desc, x1529,
    conv_desc, algo, ws_data, ws_size,
    x3544, grad_filt_desc, x1278));
};
float* x3547 = (float*)myMalloc(1 * sizeof(float));;
x3547[0] = 1.0f;
float* x3549 = (float*)myMalloc(1 * sizeof(float));;
x3549[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3547, x_desc, x1488, x_desc, x1496, x_desc, x1488,
    x3549, x_desc, x1496));
};
float* x3552 = (float*)myMalloc(1 * sizeof(float));;
x3552[0] = 1.0f;
float* x3554 = (float*)myMalloc(1 * sizeof(float));;
x3554[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x3552, bias_desc, x1496, x3554, out_desc, x1512));
};
float* x3557 = (float*)myMalloc(1 * sizeof(float));;
x3557[0] = 0.0f;
float* x3559 = (float*)myMalloc(1 * sizeof(float));;
x3559[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3559, x3559, x3559, x3559, in_desc, x1497,
    out_desc, x1512, in_desc, x1503, sbmv_desc, x523,
    x1183,x1310, 1.0E-5, x1505, x1506));
};
// conv2D back-propagate
float* x3563 = (float*)myMalloc(1 * sizeof(float));;
x3563[0] = 1.0f;

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
    64, 64, 16, 16));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3563, filt_desc, x781, grad_out_desc, x1503,
    conv_desc, algo, ws_data, ws_size,
    x3563, grad_in_desc, x1438));
};
float* x3566 = (float*)myMalloc(1 * sizeof(float));;
x3566[0] = 1.0f;

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
    64, 256, 16, 16));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3566, in_desc, x1436, grad_out_desc, x1503,
    conv_desc, algo, ws_data, ws_size,
    x3566, grad_filt_desc, x1269));
};
float* x3569 = (float*)myMalloc(1 * sizeof(float));;
x3569[0] = 0.0f;
float* x3571 = (float*)myMalloc(1 * sizeof(float));;
x3571[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3571, x3571, x3571, x3571, in_desc, x1481,
    out_desc, x1496, in_desc, x1487, sbmv_desc, x892,
    x1306,x1233, 1.0E-5, x1489, x1490));
};
// conv2D back-propagate
float* x3575 = (float*)myMalloc(1 * sizeof(float));;
x3575[0] = 1.0f;

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
    64, 64, 16, 16));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3575, filt_desc, x391, grad_out_desc, x1487,
    conv_desc, algo, ws_data, ws_size,
    x3575, grad_in_desc, x1475));
};
float* x3578 = (float*)myMalloc(1 * sizeof(float));;
x3578[0] = 1.0f;

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
    64, 256, 16, 16));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3578, in_desc, x1467, grad_out_desc, x1487,
    conv_desc, algo, ws_data, ws_size,
    x3578, grad_filt_desc, x1139));
};
float* x3581 = (float*)myMalloc(1 * sizeof(float));;
x3581[0] = 1.0f;
float* x3583 = (float*)myMalloc(1 * sizeof(float));;
x3583[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3581, x_desc, x1467, x_desc, x1475, x_desc, x1467,
    x3583, x_desc, x1475));
};
float* x3586 = (float*)myMalloc(1 * sizeof(float));;
x3586[0] = 0.0f;
float* x3588 = (float*)myMalloc(1 * sizeof(float));;
x3588[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3588, x3588, x3588, x3588, in_desc, x1460,
    out_desc, x1475, in_desc, x1466, sbmv_desc, x787,
    x1271,x1156, 1.0E-5, x1468, x1469));
};
// conv2D back-propagate
float* x3592 = (float*)myMalloc(1 * sizeof(float));;
x3592[0] = 1.0f;

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
    64, 64, 16, 16));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3592, filt_desc, x565, grad_out_desc, x1466,
    conv_desc, algo, ws_data, ws_size,
    x3592, grad_in_desc, x1454));
};
float* x3595 = (float*)myMalloc(1 * sizeof(float));;
x3595[0] = 1.0f;

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
    64, 64, 16, 16));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    1, 1, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3595, in_desc, x1446, grad_out_desc, x1466,
    conv_desc, algo, ws_data, ws_size,
    x3595, grad_filt_desc, x1197));
};
float* x3598 = (float*)myMalloc(1 * sizeof(float));;
x3598[0] = 1.0f;
float* x3600 = (float*)myMalloc(1 * sizeof(float));;
x3600[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3598, x_desc, x1446, x_desc, x1454, x_desc, x1446,
    x3600, x_desc, x1454));
};
float* x3603 = (float*)myMalloc(1 * sizeof(float));;
x3603[0] = 0.0f;
float* x3605 = (float*)myMalloc(1 * sizeof(float));;
x3605[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3605, x3605, x3605, x3605, in_desc, x1439,
    out_desc, x1454, in_desc, x1445, sbmv_desc, x373,
    x1133,x1160, 1.0E-5, x1447, x1448));
};
// conv2D back-propagate
float* x3609 = (float*)myMalloc(1 * sizeof(float));;
x3609[0] = 1.0f;

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
    64, 64, 16, 16));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3609, filt_desc, x994, grad_out_desc, x1445,
    conv_desc, algo, ws_data, ws_size,
    x3609, grad_in_desc, x1438));
};
float* x3612 = (float*)myMalloc(1 * sizeof(float));;
x3612[0] = 1.0f;

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
    64, 64, 16, 16));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 1, 1, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
    x3612, in_desc, x1436, grad_out_desc, x1445,
    conv_desc, algo, ws_data, ws_size,
    x3612, grad_filt_desc, x1340));
};
float* x3615 = (float*)myMalloc(1 * sizeof(float));;
x3615[0] = 0.0f;
float* x3617 = (float*)myMalloc(1 * sizeof(float));;
x3617[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 32, 32));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

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
    x3617, out_desc, x1436, out_desc, x1438, in_desc, x1418  , x3615, in_desc, x1426));
};
float* x3620 = (float*)myMalloc(1 * sizeof(float));;
x3620[0] = 1.0f;
float* x3622 = (float*)myMalloc(1 * sizeof(float));;
x3622[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 32, 32));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x3620, x_desc, x1418, x_desc, x1426, x_desc, x1418,
    x3622, x_desc, x1426));
};
float* x3625 = (float*)myMalloc(1 * sizeof(float));;
x3625[0] = 0.0f;
float* x3627 = (float*)myMalloc(1 * sizeof(float));;
x3627[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 32, 32));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 32, 32));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x3627, x3627, x3627, x3627, in_desc, x1411,
    out_desc, x1426, in_desc, x1417, sbmv_desc, x913,
    x1313,x1358, 1.0E-5, x1419, x1420));
};
// conv2D back-propagate
float* x3631 = (float*)myMalloc(1 * sizeof(float));;
x3631[0] = 1.0f;

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
    64, 64, 32, 32));

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
    x3631, in_desc, x1402, grad_out_desc, x1417,
    conv_desc, algo, ws_data, ws_size,
    x3631, grad_filt_desc, x1259));
};
float x3634 = x1410[0];
x1390 += x3634;
float* x3636 = (float*)myMalloc(1 * sizeof(float));;
x3636[0] = 1.0f;
float* x3638 = (float*)myMalloc(1 * sizeof(float));;
x3638[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,256,x3636,x313,1024,x3638, x1113, 1024, x313,1024));
arrayFill_greg<<<52, 512>>>(x1113, 0.0f, 262144);
float* x3642 = (float*)myMalloc(1 * sizeof(float));;
x3642[0] = 1.0f;
float* x3644 = (float*)myMalloc(1 * sizeof(float));;
x3644[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x3642,x316,1,x3644, x1114, 1, x316,1));
arrayFill_greg<<<1, 512>>>(x1114, 0.0f, 256);
float* x3648 = (float*)myMalloc(1 * sizeof(float));;
x3648[0] = 1.0f;
float* x3650 = (float*)myMalloc(1 * sizeof(float));;
x3650[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x3648,x319,1,x3650, x1115, 1, x319,1));
arrayFill_greg<<<1, 512>>>(x1115, 0.0f, 128);
float* x3654 = (float*)myMalloc(1 * sizeof(float));;
x3654[0] = 1.0f;
float* x3656 = (float*)myMalloc(1 * sizeof(float));;
x3656[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x3654,x322,1,x3656, x1116, 1, x322,1));
arrayFill_greg<<<1, 512>>>(x1116, 0.0f, 128);
float* x3660 = (float*)myMalloc(1 * sizeof(float));;
x3660[0] = 1.0f;
float* x3662 = (float*)myMalloc(1 * sizeof(float));;
x3662[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x3660,x325,1,x3662, x1117, 1, x325,1));
arrayFill_greg<<<1, 512>>>(x1117, 0.0f, 64);
float* x3666 = (float*)myMalloc(1 * sizeof(float));;
x3666[0] = 1.0f;
float* x3668 = (float*)myMalloc(1 * sizeof(float));;
x3668[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,128,x3666,x328,256,x3668, x1118, 256, x328,256));
arrayFill_greg<<<7, 512>>>(x1118, 0.0f, 32768);
float* x3672 = (float*)myMalloc(1 * sizeof(float));;
x3672[0] = 1.0f;
float* x3674 = (float*)myMalloc(1 * sizeof(float));;
x3674[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3672,x331,1,x3674, x1119, 1, x331,1));
arrayFill_greg<<<1, 512>>>(x1119, 0.0f, 512);
float* x3678 = (float*)myMalloc(1 * sizeof(float));;
x3678[0] = 1.0f;
float* x3680 = (float*)myMalloc(1 * sizeof(float));;
x3680[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,256,x3678,x334,1024,x3680, x1120, 1024, x334,1024));
arrayFill_greg<<<52, 512>>>(x1120, 0.0f, 262144);
float* x3684 = (float*)myMalloc(1 * sizeof(float));;
x3684[0] = 1.0f;
float* x3686 = (float*)myMalloc(1 * sizeof(float));;
x3686[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x3684,x337,2304,x3686, x1121, 2304, x337,2304));
arrayFill_greg<<<116, 512>>>(x1121, 0.0f, 589824);
float* x3690 = (float*)myMalloc(1 * sizeof(float));;
x3690[0] = 1.0f;
float* x3692 = (float*)myMalloc(1 * sizeof(float));;
x3692[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3690,x340,1,x3692, x1122, 1, x340,1));
arrayFill_greg<<<1, 512>>>(x1122, 0.0f, 512);
float* x3696 = (float*)myMalloc(1 * sizeof(float));;
x3696[0] = 1.0f;
float* x3698 = (float*)myMalloc(1 * sizeof(float));;
x3698[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x3696,x343,1,x3698, x1123, 1, x343,1));
arrayFill_greg<<<1, 512>>>(x1123, 0.0f, 256);
float* x3702 = (float*)myMalloc(1 * sizeof(float));;
x3702[0] = 1.0f;
float* x3704 = (float*)myMalloc(1 * sizeof(float));;
x3704[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x3702,x346,1,x3704, x1124, 1, x346,1));
arrayFill_greg<<<1, 512>>>(x1124, 0.0f, 128);
float* x3708 = (float*)myMalloc(1 * sizeof(float));;
x3708[0] = 1.0f;
float* x3710 = (float*)myMalloc(1 * sizeof(float));;
x3710[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x3708,x349,1,x3710, x1125, 1, x349,1));
arrayFill_greg<<<1, 512>>>(x1125, 0.0f, 1024);
float* x3714 = (float*)myMalloc(1 * sizeof(float));;
x3714[0] = 1.0f;
float* x3716 = (float*)myMalloc(1 * sizeof(float));;
x3716[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3714,x352,1,x3716, x1126, 1, x352,1));
arrayFill_greg<<<1, 512>>>(x1126, 0.0f, 512);
float* x3720 = (float*)myMalloc(1 * sizeof(float));;
x3720[0] = 1.0f;
float* x3722 = (float*)myMalloc(1 * sizeof(float));;
x3722[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x3720,x355,1,x3722, x1127, 1, x355,1));
arrayFill_greg<<<1, 512>>>(x1127, 0.0f, 1024);
float* x3726 = (float*)myMalloc(1 * sizeof(float));;
x3726[0] = 1.0f;
float* x3728 = (float*)myMalloc(1 * sizeof(float));;
x3728[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x3726,x358,1,x3728, x1128, 1, x358,1));
arrayFill_greg<<<1, 512>>>(x1128, 0.0f, 256);
float* x3732 = (float*)myMalloc(1 * sizeof(float));;
x3732[0] = 1.0f;
float* x3734 = (float*)myMalloc(1 * sizeof(float));;
x3734[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,256,x3732,x361,1024,x3734, x1129, 1024, x361,1024));
arrayFill_greg<<<52, 512>>>(x1129, 0.0f, 262144);
float* x3738 = (float*)myMalloc(1 * sizeof(float));;
x3738[0] = 1.0f;
float* x3740 = (float*)myMalloc(1 * sizeof(float));;
x3740[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3738,x364,1,x3740, x1130, 1, x364,1));
arrayFill_greg<<<1, 512>>>(x1130, 0.0f, 512);
float* x3744 = (float*)myMalloc(1 * sizeof(float));;
x3744[0] = 1.0f;
float* x3746 = (float*)myMalloc(1 * sizeof(float));;
x3746[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x3744,x367,1,x3746, x1131, 1, x367,1));
arrayFill_greg<<<1, 512>>>(x1131, 0.0f, 64);
float* x3750 = (float*)myMalloc(1 * sizeof(float));;
x3750[0] = 1.0f;
float* x3752 = (float*)myMalloc(1 * sizeof(float));;
x3752[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3750,x370,1,x3752, x1132, 1, x370,1));
arrayFill_greg<<<1, 512>>>(x1132, 0.0f, 512);
float* x3756 = (float*)myMalloc(1 * sizeof(float));;
x3756[0] = 1.0f;
float* x3758 = (float*)myMalloc(1 * sizeof(float));;
x3758[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x3756,x373,1,x3758, x1133, 1, x373,1));
arrayFill_greg<<<1, 512>>>(x1133, 0.0f, 64);
float* x3762 = (float*)myMalloc(1 * sizeof(float));;
x3762[0] = 1.0f;
float* x3764 = (float*)myMalloc(1 * sizeof(float));;
x3764[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1152,128,x3762,x376,1152,x3764, x1134, 1152, x376,1152));
arrayFill_greg<<<29, 512>>>(x1134, 0.0f, 147456);
float* x3768 = (float*)myMalloc(1 * sizeof(float));;
x3768[0] = 1.0f;
float* x3770 = (float*)myMalloc(1 * sizeof(float));;
x3770[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 4608,512,x3768,x379,4608,x3770, x1135, 4608, x379,4608));
arrayFill_greg<<<461, 512>>>(x1135, 0.0f, 2359296);
float* x3774 = (float*)myMalloc(1 * sizeof(float));;
x3774[0] = 1.0f;
float* x3776 = (float*)myMalloc(1 * sizeof(float));;
x3776[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x3774,x382,1,x3776, x1136, 1, x382,1));
arrayFill_greg<<<1, 512>>>(x1136, 0.0f, 1024);
float* x3780 = (float*)myMalloc(1 * sizeof(float));;
x3780[0] = 1.0f;
float* x3782 = (float*)myMalloc(1 * sizeof(float));;
x3782[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x3780,x385,1,x3782, x1137, 1, x385,1));
arrayFill_greg<<<1, 512>>>(x1137, 0.0f, 256);
float* x3786 = (float*)myMalloc(1 * sizeof(float));;
x3786[0] = 1.0f;
float* x3788 = (float*)myMalloc(1 * sizeof(float));;
x3788[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x3786,x388,2304,x3788, x1138, 2304, x388,2304));
arrayFill_greg<<<116, 512>>>(x1138, 0.0f, 589824);
float* x3792 = (float*)myMalloc(1 * sizeof(float));;
x3792[0] = 1.0f;
float* x3794 = (float*)myMalloc(1 * sizeof(float));;
x3794[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x3792,x391,64,x3794, x1139, 64, x391,64));
arrayFill_greg<<<4, 512>>>(x1139, 0.0f, 16384);
float* x3798 = (float*)myMalloc(1 * sizeof(float));;
x3798[0] = 1.0f;
float* x3800 = (float*)myMalloc(1 * sizeof(float));;
x3800[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,2048,x3798,x394,512,x3800, x1140, 512, x394,512));
arrayFill_greg<<<205, 512>>>(x1140, 0.0f, 1048576);
float* x3804 = (float*)myMalloc(1 * sizeof(float));;
x3804[0] = 1.0f;
float* x3806 = (float*)myMalloc(1 * sizeof(float));;
x3806[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 4608,512,x3804,x397,4608,x3806, x1141, 4608, x397,4608));
arrayFill_greg<<<461, 512>>>(x1141, 0.0f, 2359296);
float* x3810 = (float*)myMalloc(1 * sizeof(float));;
x3810[0] = 1.0f;
float* x3812 = (float*)myMalloc(1 * sizeof(float));;
x3812[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x3810,x400,1,x3812, x1142, 1, x400,1));
arrayFill_greg<<<1, 512>>>(x1142, 0.0f, 128);
float* x3816 = (float*)myMalloc(1 * sizeof(float));;
x3816[0] = 1.0f;
float* x3818 = (float*)myMalloc(1 * sizeof(float));;
x3818[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x3816,x403,1,x3818, x1143, 1, x403,1));
arrayFill_greg<<<1, 512>>>(x1143, 0.0f, 256);
float* x3822 = (float*)myMalloc(1 * sizeof(float));;
x3822[0] = 1.0f;
float* x3824 = (float*)myMalloc(1 * sizeof(float));;
x3824[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x3822,x406,1,x3824, x1144, 1, x406,1));
arrayFill_greg<<<1, 512>>>(x1144, 0.0f, 256);
float* x3828 = (float*)myMalloc(1 * sizeof(float));;
x3828[0] = 1.0f;
float* x3830 = (float*)myMalloc(1 * sizeof(float));;
x3830[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x3828,x409,1,x3830, x1145, 1, x409,1));
arrayFill_greg<<<1, 512>>>(x1145, 0.0f, 128);
float* x3834 = (float*)myMalloc(1 * sizeof(float));;
x3834[0] = 1.0f;
float* x3836 = (float*)myMalloc(1 * sizeof(float));;
x3836[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x3834,x412,1,x3836, x1146, 1, x412,1));
arrayFill_greg<<<1, 512>>>(x1146, 0.0f, 128);
float* x3840 = (float*)myMalloc(1 * sizeof(float));;
x3840[0] = 1.0f;
float* x3842 = (float*)myMalloc(1 * sizeof(float));;
x3842[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x3840,x415,1,x3842, x1147, 1, x415,1));
arrayFill_greg<<<1, 512>>>(x1147, 0.0f, 64);
float* x3846 = (float*)myMalloc(1 * sizeof(float));;
x3846[0] = 1.0f;
float* x3848 = (float*)myMalloc(1 * sizeof(float));;
x3848[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,512,x3846,x418,128,x3848, x1148, 128, x418,128));
arrayFill_greg<<<13, 512>>>(x1148, 0.0f, 65536);
float* x3852 = (float*)myMalloc(1 * sizeof(float));;
x3852[0] = 1.0f;
float* x3854 = (float*)myMalloc(1 * sizeof(float));;
x3854[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3852,x421,1,x3854, x1149, 1, x421,1));
arrayFill_greg<<<1, 512>>>(x1149, 0.0f, 512);
float* x3858 = (float*)myMalloc(1 * sizeof(float));;
x3858[0] = 1.0f;
float* x3860 = (float*)myMalloc(1 * sizeof(float));;
x3860[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x3858,x424,1,x3860, x1150, 1, x424,1));
arrayFill_greg<<<1, 512>>>(x1150, 0.0f, 128);
float* x3864 = (float*)myMalloc(1 * sizeof(float));;
x3864[0] = 1.0f;
float* x3866 = (float*)myMalloc(1 * sizeof(float));;
x3866[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x3864,x427,1,x3866, x1151, 1, x427,1));
arrayFill_greg<<<1, 512>>>(x1151, 0.0f, 64);
float* x3870 = (float*)myMalloc(1 * sizeof(float));;
x3870[0] = 1.0f;
float* x3872 = (float*)myMalloc(1 * sizeof(float));;
x3872[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x3870,x430,1,x3872, x1152, 1, x430,1));
arrayFill_greg<<<1, 512>>>(x1152, 0.0f, 128);
float* x3876 = (float*)myMalloc(1 * sizeof(float));;
x3876[0] = 1.0f;
float* x3878 = (float*)myMalloc(1 * sizeof(float));;
x3878[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3876,x433,1,x3878, x1153, 1, x433,1));
arrayFill_greg<<<1, 512>>>(x1153, 0.0f, 512);
float* x3882 = (float*)myMalloc(1 * sizeof(float));;
x3882[0] = 1.0f;
float* x3884 = (float*)myMalloc(1 * sizeof(float));;
x3884[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,2048,x3882,x436,512,x3884, x1154, 512, x436,512));
arrayFill_greg<<<205, 512>>>(x1154, 0.0f, 1048576);
float* x3888 = (float*)myMalloc(1 * sizeof(float));;
x3888[0] = 1.0f;
float* x3890 = (float*)myMalloc(1 * sizeof(float));;
x3890[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,10,x3888,x439,1,x3890, x1155, 1, x439,1));
arrayFill_greg<<<1, 512>>>(x1155, 0.0f, 10);
float* x3894 = (float*)myMalloc(1 * sizeof(float));;
x3894[0] = 1.0f;
float* x3896 = (float*)myMalloc(1 * sizeof(float));;
x3896[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x3894,x442,1,x3896, x1156, 1, x442,1));
arrayFill_greg<<<1, 512>>>(x1156, 0.0f, 64);
float* x3900 = (float*)myMalloc(1 * sizeof(float));;
x3900[0] = 1.0f;
float* x3902 = (float*)myMalloc(1 * sizeof(float));;
x3902[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3900,x445,1,x3902, x1157, 1, x445,1));
arrayFill_greg<<<1, 512>>>(x1157, 0.0f, 512);
float* x3906 = (float*)myMalloc(1 * sizeof(float));;
x3906[0] = 1.0f;
float* x3908 = (float*)myMalloc(1 * sizeof(float));;
x3908[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x3906,x448,1,x3908, x1158, 1, x448,1));
arrayFill_greg<<<1, 512>>>(x1158, 0.0f, 64);
float* x3912 = (float*)myMalloc(1 * sizeof(float));;
x3912[0] = 1.0f;
float* x3914 = (float*)myMalloc(1 * sizeof(float));;
x3914[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3912,x451,1,x3914, x1159, 1, x451,1));
arrayFill_greg<<<1, 512>>>(x1159, 0.0f, 512);
float* x3918 = (float*)myMalloc(1 * sizeof(float));;
x3918[0] = 1.0f;
float* x3920 = (float*)myMalloc(1 * sizeof(float));;
x3920[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x3918,x454,1,x3920, x1160, 1, x454,1));
arrayFill_greg<<<1, 512>>>(x1160, 0.0f, 64);
float* x3924 = (float*)myMalloc(1 * sizeof(float));;
x3924[0] = 1.0f;
float* x3926 = (float*)myMalloc(1 * sizeof(float));;
x3926[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3924,x457,1,x3926, x1161, 1, x457,1));
arrayFill_greg<<<1, 512>>>(x1161, 0.0f, 512);
float* x3930 = (float*)myMalloc(1 * sizeof(float));;
x3930[0] = 1.0f;
float* x3932 = (float*)myMalloc(1 * sizeof(float));;
x3932[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,512,x3930,x460,128,x3932, x1162, 128, x460,128));
arrayFill_greg<<<13, 512>>>(x1162, 0.0f, 65536);
float* x3936 = (float*)myMalloc(1 * sizeof(float));;
x3936[0] = 1.0f;
float* x3938 = (float*)myMalloc(1 * sizeof(float));;
x3938[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x3936,x463,256,x3938, x1163, 256, x463,256));
arrayFill_greg<<<52, 512>>>(x1163, 0.0f, 262144);
float* x3942 = (float*)myMalloc(1 * sizeof(float));;
x3942[0] = 1.0f;
float* x3944 = (float*)myMalloc(1 * sizeof(float));;
x3944[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x3942,x466,1,x3944, x1164, 1, x466,1));
arrayFill_greg<<<1, 512>>>(x1164, 0.0f, 1024);
float* x3948 = (float*)myMalloc(1 * sizeof(float));;
x3948[0] = 1.0f;
float* x3950 = (float*)myMalloc(1 * sizeof(float));;
x3950[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x3948,x469,1,x3950, x1165, 1, x469,1));
arrayFill_greg<<<1, 512>>>(x1165, 0.0f, 256);
float* x3954 = (float*)myMalloc(1 * sizeof(float));;
x3954[0] = 1.0f;
float* x3956 = (float*)myMalloc(1 * sizeof(float));;
x3956[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x3954,x472,1,x3956, x1166, 1, x472,1));
arrayFill_greg<<<1, 512>>>(x1166, 0.0f, 1024);
float* x3960 = (float*)myMalloc(1 * sizeof(float));;
x3960[0] = 1.0f;
float* x3962 = (float*)myMalloc(1 * sizeof(float));;
x3962[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x3960,x475,1,x3962, x1167, 1, x475,1));
arrayFill_greg<<<1, 512>>>(x1167, 0.0f, 64);
float* x3966 = (float*)myMalloc(1 * sizeof(float));;
x3966[0] = 1.0f;
float* x3968 = (float*)myMalloc(1 * sizeof(float));;
x3968[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x3966,x478,1,x3968, x1168, 1, x478,1));
arrayFill_greg<<<1, 512>>>(x1168, 0.0f, 128);
float* x3972 = (float*)myMalloc(1 * sizeof(float));;
x3972[0] = 1.0f;
float* x3974 = (float*)myMalloc(1 * sizeof(float));;
x3974[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x3972,x481,1,x3974, x1169, 1, x481,1));
arrayFill_greg<<<1, 512>>>(x1169, 0.0f, 2048);
float* x3978 = (float*)myMalloc(1 * sizeof(float));;
x3978[0] = 1.0f;
float* x3980 = (float*)myMalloc(1 * sizeof(float));;
x3980[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x3978,x484,1,x3980, x1170, 1, x484,1));
arrayFill_greg<<<1, 512>>>(x1170, 0.0f, 256);
float* x3984 = (float*)myMalloc(1 * sizeof(float));;
x3984[0] = 1.0f;
float* x3986 = (float*)myMalloc(1 * sizeof(float));;
x3986[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x3984,x487,1,x3986, x1171, 1, x487,1));
arrayFill_greg<<<1, 512>>>(x1171, 0.0f, 2048);
float* x3990 = (float*)myMalloc(1 * sizeof(float));;
x3990[0] = 1.0f;
float* x3992 = (float*)myMalloc(1 * sizeof(float));;
x3992[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3990,x490,1,x3992, x1172, 1, x490,1));
arrayFill_greg<<<1, 512>>>(x1172, 0.0f, 512);
float* x3996 = (float*)myMalloc(1 * sizeof(float));;
x3996[0] = 1.0f;
float* x3998 = (float*)myMalloc(1 * sizeof(float));;
x3998[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3996,x493,1,x3998, x1173, 1, x493,1));
arrayFill_greg<<<1, 512>>>(x1173, 0.0f, 512);
float* x4002 = (float*)myMalloc(1 * sizeof(float));;
x4002[0] = 1.0f;
float* x4004 = (float*)myMalloc(1 * sizeof(float));;
x4004[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4002,x496,1,x4004, x1174, 1, x496,1));
arrayFill_greg<<<1, 512>>>(x1174, 0.0f, 512);
float* x4008 = (float*)myMalloc(1 * sizeof(float));;
x4008[0] = 1.0f;
float* x4010 = (float*)myMalloc(1 * sizeof(float));;
x4010[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4008,x499,1,x4010, x1175, 1, x499,1));
arrayFill_greg<<<1, 512>>>(x1175, 0.0f, 2048);
float* x4014 = (float*)myMalloc(1 * sizeof(float));;
x4014[0] = 1.0f;
float* x4016 = (float*)myMalloc(1 * sizeof(float));;
x4016[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4014,x502,1,x4016, x1176, 1, x502,1));
arrayFill_greg<<<1, 512>>>(x1176, 0.0f, 256);
float* x4020 = (float*)myMalloc(1 * sizeof(float));;
x4020[0] = 1.0f;
float* x4022 = (float*)myMalloc(1 * sizeof(float));;
x4022[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4020,x505,1,x4022, x1177, 1, x505,1));
arrayFill_greg<<<1, 512>>>(x1177, 0.0f, 256);
float* x4026 = (float*)myMalloc(1 * sizeof(float));;
x4026[0] = 1.0f;
float* x4028 = (float*)myMalloc(1 * sizeof(float));;
x4028[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4026,x508,1,x4028, x1178, 1, x508,1));
arrayFill_greg<<<1, 512>>>(x1178, 0.0f, 256);
float* x4032 = (float*)myMalloc(1 * sizeof(float));;
x4032[0] = 1.0f;
float* x4034 = (float*)myMalloc(1 * sizeof(float));;
x4034[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4032,x511,1,x4034, x1179, 1, x511,1));
arrayFill_greg<<<1, 512>>>(x1179, 0.0f, 64);
float* x4038 = (float*)myMalloc(1 * sizeof(float));;
x4038[0] = 1.0f;
float* x4040 = (float*)myMalloc(1 * sizeof(float));;
x4040[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 576,64,x4038,x514,576,x4040, x1180, 576, x514,576));
arrayFill_greg<<<8, 512>>>(x1180, 0.0f, 36864);
float* x4044 = (float*)myMalloc(1 * sizeof(float));;
x4044[0] = 1.0f;
float* x4046 = (float*)myMalloc(1 * sizeof(float));;
x4046[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4044,x517,1,x4046, x1181, 1, x517,1));
arrayFill_greg<<<1, 512>>>(x1181, 0.0f, 256);
float* x4050 = (float*)myMalloc(1 * sizeof(float));;
x4050[0] = 1.0f;
float* x4052 = (float*)myMalloc(1 * sizeof(float));;
x4052[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,1024,x4050,x520,512,x4052, x1182, 512, x520,512));
arrayFill_greg<<<103, 512>>>(x1182, 0.0f, 524288);
float* x4056 = (float*)myMalloc(1 * sizeof(float));;
x4056[0] = 1.0f;
float* x4058 = (float*)myMalloc(1 * sizeof(float));;
x4058[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4056,x523,1,x4058, x1183, 1, x523,1));
arrayFill_greg<<<1, 512>>>(x1183, 0.0f, 256);
float* x4062 = (float*)myMalloc(1 * sizeof(float));;
x4062[0] = 1.0f;
float* x4064 = (float*)myMalloc(1 * sizeof(float));;
x4064[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4062,x526,1,x4064, x1184, 1, x526,1));
arrayFill_greg<<<1, 512>>>(x1184, 0.0f, 256);
float* x4068 = (float*)myMalloc(1 * sizeof(float));;
x4068[0] = 1.0f;
float* x4070 = (float*)myMalloc(1 * sizeof(float));;
x4070[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4068,x529,1,x4070, x1185, 1, x529,1));
arrayFill_greg<<<1, 512>>>(x1185, 0.0f, 512);
float* x4074 = (float*)myMalloc(1 * sizeof(float));;
x4074[0] = 1.0f;
float* x4076 = (float*)myMalloc(1 * sizeof(float));;
x4076[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4074,x532,1,x4076, x1186, 1, x532,1));
arrayFill_greg<<<1, 512>>>(x1186, 0.0f, 128);
float* x4080 = (float*)myMalloc(1 * sizeof(float));;
x4080[0] = 1.0f;
float* x4082 = (float*)myMalloc(1 * sizeof(float));;
x4082[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4080,x535,1,x4082, x1187, 1, x535,1));
arrayFill_greg<<<1, 512>>>(x1187, 0.0f, 256);
float* x4086 = (float*)myMalloc(1 * sizeof(float));;
x4086[0] = 1.0f;
float* x4088 = (float*)myMalloc(1 * sizeof(float));;
x4088[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4086,x538,1,x4088, x1188, 1, x538,1));
arrayFill_greg<<<1, 512>>>(x1188, 0.0f, 64);
float* x4092 = (float*)myMalloc(1 * sizeof(float));;
x4092[0] = 1.0f;
float* x4094 = (float*)myMalloc(1 * sizeof(float));;
x4094[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4092,x541,1,x4094, x1189, 1, x541,1));
arrayFill_greg<<<1, 512>>>(x1189, 0.0f, 512);
float* x4098 = (float*)myMalloc(1 * sizeof(float));;
x4098[0] = 1.0f;
float* x4100 = (float*)myMalloc(1 * sizeof(float));;
x4100[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 576,64,x4098,x544,576,x4100, x1190, 576, x544,576));
arrayFill_greg<<<8, 512>>>(x1190, 0.0f, 36864);
float* x4104 = (float*)myMalloc(1 * sizeof(float));;
x4104[0] = 1.0f;
float* x4106 = (float*)myMalloc(1 * sizeof(float));;
x4106[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4104,x547,1,x4106, x1191, 1, x547,1));
arrayFill_greg<<<1, 512>>>(x1191, 0.0f, 128);
float* x4110 = (float*)myMalloc(1 * sizeof(float));;
x4110[0] = 1.0f;
float* x4112 = (float*)myMalloc(1 * sizeof(float));;
x4112[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4110,x550,1,x4112, x1192, 1, x550,1));
arrayFill_greg<<<1, 512>>>(x1192, 0.0f, 256);
float* x4116 = (float*)myMalloc(1 * sizeof(float));;
x4116[0] = 1.0f;
float* x4118 = (float*)myMalloc(1 * sizeof(float));;
x4118[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4116,x553,1,x4118, x1193, 1, x553,1));
arrayFill_greg<<<1, 512>>>(x1193, 0.0f, 1024);
float* x4122 = (float*)myMalloc(1 * sizeof(float));;
x4122[0] = 1.0f;
float* x4124 = (float*)myMalloc(1 * sizeof(float));;
x4124[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x4122,x556,64,x4124, x1194, 64, x556,64));
arrayFill_greg<<<4, 512>>>(x1194, 0.0f, 16384);
float* x4128 = (float*)myMalloc(1 * sizeof(float));;
x4128[0] = 1.0f;
float* x4130 = (float*)myMalloc(1 * sizeof(float));;
x4130[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4128,x559,1,x4130, x1195, 1, x559,1));
arrayFill_greg<<<1, 512>>>(x1195, 0.0f, 512);
float* x4134 = (float*)myMalloc(1 * sizeof(float));;
x4134[0] = 1.0f;
float* x4136 = (float*)myMalloc(1 * sizeof(float));;
x4136[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x4134,x562,256,x4136, x1196, 256, x562,256));
arrayFill_greg<<<52, 512>>>(x1196, 0.0f, 262144);
float* x4140 = (float*)myMalloc(1 * sizeof(float));;
x4140[0] = 1.0f;
float* x4142 = (float*)myMalloc(1 * sizeof(float));;
x4142[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 576,64,x4140,x565,576,x4142, x1197, 576, x565,576));
arrayFill_greg<<<8, 512>>>(x1197, 0.0f, 36864);
float* x4146 = (float*)myMalloc(1 * sizeof(float));;
x4146[0] = 1.0f;
float* x4148 = (float*)myMalloc(1 * sizeof(float));;
x4148[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4146,x568,1,x4148, x1198, 1, x568,1));
arrayFill_greg<<<1, 512>>>(x1198, 0.0f, 256);
float* x4152 = (float*)myMalloc(1 * sizeof(float));;
x4152[0] = 1.0f;
float* x4154 = (float*)myMalloc(1 * sizeof(float));;
x4154[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4152,x571,1,x4154, x1199, 1, x571,1));
arrayFill_greg<<<1, 512>>>(x1199, 0.0f, 256);
float* x4158 = (float*)myMalloc(1 * sizeof(float));;
x4158[0] = 1.0f;
float* x4160 = (float*)myMalloc(1 * sizeof(float));;
x4160[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4158,x574,1,x4160, x1200, 1, x574,1));
arrayFill_greg<<<1, 512>>>(x1200, 0.0f, 1024);
float* x4164 = (float*)myMalloc(1 * sizeof(float));;
x4164[0] = 1.0f;
float* x4166 = (float*)myMalloc(1 * sizeof(float));;
x4166[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4164,x577,1,x4166, x1201, 1, x577,1));
arrayFill_greg<<<1, 512>>>(x1201, 0.0f, 2048);
float* x4170 = (float*)myMalloc(1 * sizeof(float));;
x4170[0] = 1.0f;
float* x4172 = (float*)myMalloc(1 * sizeof(float));;
x4172[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4170,x580,1,x4172, x1202, 1, x580,1));
arrayFill_greg<<<1, 512>>>(x1202, 0.0f, 128);
float* x4176 = (float*)myMalloc(1 * sizeof(float));;
x4176[0] = 1.0f;
float* x4178 = (float*)myMalloc(1 * sizeof(float));;
x4178[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4176,x583,1,x4178, x1203, 1, x583,1));
arrayFill_greg<<<1, 512>>>(x1203, 0.0f, 256);
float* x4182 = (float*)myMalloc(1 * sizeof(float));;
x4182[0] = 1.0f;
float* x4184 = (float*)myMalloc(1 * sizeof(float));;
x4184[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x4182,x586,256,x4184, x1204, 256, x586,256));
arrayFill_greg<<<52, 512>>>(x1204, 0.0f, 262144);
float* x4188 = (float*)myMalloc(1 * sizeof(float));;
x4188[0] = 1.0f;
float* x4190 = (float*)myMalloc(1 * sizeof(float));;
x4190[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4188,x589,1,x4190, x1205, 1, x589,1));
arrayFill_greg<<<1, 512>>>(x1205, 0.0f, 256);
float* x4194 = (float*)myMalloc(1 * sizeof(float));;
x4194[0] = 1.0f;
float* x4196 = (float*)myMalloc(1 * sizeof(float));;
x4196[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4194,x592,1,x4196, x1206, 1, x592,1));
arrayFill_greg<<<1, 512>>>(x1206, 0.0f, 256);
float* x4200 = (float*)myMalloc(1 * sizeof(float));;
x4200[0] = 1.0f;
float* x4202 = (float*)myMalloc(1 * sizeof(float));;
x4202[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4200,x595,1,x4202, x1207, 1, x595,1));
arrayFill_greg<<<1, 512>>>(x1207, 0.0f, 128);
float* x4206 = (float*)myMalloc(1 * sizeof(float));;
x4206[0] = 1.0f;
float* x4208 = (float*)myMalloc(1 * sizeof(float));;
x4208[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4206,x598,1,x4208, x1208, 1, x598,1));
arrayFill_greg<<<1, 512>>>(x1208, 0.0f, 512);
float* x4212 = (float*)myMalloc(1 * sizeof(float));;
x4212[0] = 1.0f;
float* x4214 = (float*)myMalloc(1 * sizeof(float));;
x4214[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4212,x601,1,x4214, x1209, 1, x601,1));
arrayFill_greg<<<1, 512>>>(x1209, 0.0f, 64);
float* x4218 = (float*)myMalloc(1 * sizeof(float));;
x4218[0] = 1.0f;
float* x4220 = (float*)myMalloc(1 * sizeof(float));;
x4220[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4218,x604,1,x4220, x1210, 1, x604,1));
arrayFill_greg<<<1, 512>>>(x1210, 0.0f, 2048);
float* x4224 = (float*)myMalloc(1 * sizeof(float));;
x4224[0] = 1.0f;
float* x4226 = (float*)myMalloc(1 * sizeof(float));;
x4226[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4224,x607,1,x4226, x1211, 1, x607,1));
arrayFill_greg<<<1, 512>>>(x1211, 0.0f, 256);
float* x4230 = (float*)myMalloc(1 * sizeof(float));;
x4230[0] = 1.0f;
float* x4232 = (float*)myMalloc(1 * sizeof(float));;
x4232[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4230,x610,1,x4232, x1212, 1, x610,1));
arrayFill_greg<<<1, 512>>>(x1212, 0.0f, 64);
float* x4236 = (float*)myMalloc(1 * sizeof(float));;
x4236[0] = 1.0f;
float* x4238 = (float*)myMalloc(1 * sizeof(float));;
x4238[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,512,x4236,x613,128,x4238, x1213, 128, x613,128));
arrayFill_greg<<<13, 512>>>(x1213, 0.0f, 65536);
float* x4242 = (float*)myMalloc(1 * sizeof(float));;
x4242[0] = 1.0f;
float* x4244 = (float*)myMalloc(1 * sizeof(float));;
x4244[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4242,x616,1,x4244, x1214, 1, x616,1));
arrayFill_greg<<<1, 512>>>(x1214, 0.0f, 2048);
float* x4248 = (float*)myMalloc(1 * sizeof(float));;
x4248[0] = 1.0f;
float* x4250 = (float*)myMalloc(1 * sizeof(float));;
x4250[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4248,x619,1,x4250, x1215, 1, x619,1));
arrayFill_greg<<<1, 512>>>(x1215, 0.0f, 256);
float* x4254 = (float*)myMalloc(1 * sizeof(float));;
x4254[0] = 1.0f;
float* x4256 = (float*)myMalloc(1 * sizeof(float));;
x4256[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4254,x622,1,x4256, x1216, 1, x622,1));
arrayFill_greg<<<1, 512>>>(x1216, 0.0f, 256);
float* x4260 = (float*)myMalloc(1 * sizeof(float));;
x4260[0] = 1.0f;
float* x4262 = (float*)myMalloc(1 * sizeof(float));;
x4262[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4260,x625,1,x4262, x1217, 1, x625,1));
arrayFill_greg<<<1, 512>>>(x1217, 0.0f, 64);
float* x4266 = (float*)myMalloc(1 * sizeof(float));;
x4266[0] = 1.0f;
float* x4268 = (float*)myMalloc(1 * sizeof(float));;
x4268[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,512,x4266,x628,128,x4268, x1218, 128, x628,128));
arrayFill_greg<<<13, 512>>>(x1218, 0.0f, 65536);
float* x4272 = (float*)myMalloc(1 * sizeof(float));;
x4272[0] = 1.0f;
float* x4274 = (float*)myMalloc(1 * sizeof(float));;
x4274[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4272,x631,1,x4274, x1219, 1, x631,1));
arrayFill_greg<<<1, 512>>>(x1219, 0.0f, 128);
float* x4278 = (float*)myMalloc(1 * sizeof(float));;
x4278[0] = 1.0f;
float* x4280 = (float*)myMalloc(1 * sizeof(float));;
x4280[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4278,x634,1,x4280, x1220, 1, x634,1));
arrayFill_greg<<<1, 512>>>(x1220, 0.0f, 512);
float* x4284 = (float*)myMalloc(1 * sizeof(float));;
x4284[0] = 1.0f;
float* x4286 = (float*)myMalloc(1 * sizeof(float));;
x4286[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4284,x637,1,x4286, x1221, 1, x637,1));
arrayFill_greg<<<1, 512>>>(x1221, 0.0f, 64);
float* x4290 = (float*)myMalloc(1 * sizeof(float));;
x4290[0] = 1.0f;
float* x4292 = (float*)myMalloc(1 * sizeof(float));;
x4292[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4290,x640,1,x4292, x1222, 1, x640,1));
arrayFill_greg<<<1, 512>>>(x1222, 0.0f, 2048);
float* x4296 = (float*)myMalloc(1 * sizeof(float));;
x4296[0] = 1.0f;
float* x4298 = (float*)myMalloc(1 * sizeof(float));;
x4298[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x4296,x643,256,x4298, x1223, 256, x643,256));
arrayFill_greg<<<52, 512>>>(x1223, 0.0f, 262144);
float* x4302 = (float*)myMalloc(1 * sizeof(float));;
x4302[0] = 1.0f;
float* x4304 = (float*)myMalloc(1 * sizeof(float));;
x4304[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4302,x646,1,x4304, x1224, 1, x646,1));
arrayFill_greg<<<1, 512>>>(x1224, 0.0f, 1024);
float* x4308 = (float*)myMalloc(1 * sizeof(float));;
x4308[0] = 1.0f;
float* x4310 = (float*)myMalloc(1 * sizeof(float));;
x4310[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4308,x649,1,x4310, x1225, 1, x649,1));
arrayFill_greg<<<1, 512>>>(x1225, 0.0f, 64);
float* x4314 = (float*)myMalloc(1 * sizeof(float));;
x4314[0] = 1.0f;
float* x4316 = (float*)myMalloc(1 * sizeof(float));;
x4316[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4314,x652,1,x4316, x1226, 1, x652,1));
arrayFill_greg<<<1, 512>>>(x1226, 0.0f, 512);
float* x4320 = (float*)myMalloc(1 * sizeof(float));;
x4320[0] = 1.0f;
float* x4322 = (float*)myMalloc(1 * sizeof(float));;
x4322[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4320,x655,1,x4322, x1227, 1, x655,1));
arrayFill_greg<<<1, 512>>>(x1227, 0.0f, 1024);
float* x4326 = (float*)myMalloc(1 * sizeof(float));;
x4326[0] = 1.0f;
float* x4328 = (float*)myMalloc(1 * sizeof(float));;
x4328[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4326,x658,1,x4328, x1228, 1, x658,1));
arrayFill_greg<<<1, 512>>>(x1228, 0.0f, 512);
float* x4332 = (float*)myMalloc(1 * sizeof(float));;
x4332[0] = 1.0f;
float* x4334 = (float*)myMalloc(1 * sizeof(float));;
x4334[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4332,x661,1,x4334, x1229, 1, x661,1));
arrayFill_greg<<<1, 512>>>(x1229, 0.0f, 1024);
float* x4338 = (float*)myMalloc(1 * sizeof(float));;
x4338[0] = 1.0f;
float* x4340 = (float*)myMalloc(1 * sizeof(float));;
x4340[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4338,x664,1,x4340, x1230, 1, x664,1));
arrayFill_greg<<<1, 512>>>(x1230, 0.0f, 2048);
float* x4344 = (float*)myMalloc(1 * sizeof(float));;
x4344[0] = 1.0f;
float* x4346 = (float*)myMalloc(1 * sizeof(float));;
x4346[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4344,x667,1,x4346, x1231, 1, x667,1));
arrayFill_greg<<<1, 512>>>(x1231, 0.0f, 256);
float* x4350 = (float*)myMalloc(1 * sizeof(float));;
x4350[0] = 1.0f;
float* x4352 = (float*)myMalloc(1 * sizeof(float));;
x4352[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4350,x670,1,x4352, x1232, 1, x670,1));
arrayFill_greg<<<1, 512>>>(x1232, 0.0f, 2048);
float* x4356 = (float*)myMalloc(1 * sizeof(float));;
x4356[0] = 1.0f;
float* x4358 = (float*)myMalloc(1 * sizeof(float));;
x4358[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4356,x673,1,x4358, x1233, 1, x673,1));
arrayFill_greg<<<1, 512>>>(x1233, 0.0f, 256);
float* x4362 = (float*)myMalloc(1 * sizeof(float));;
x4362[0] = 1.0f;
float* x4364 = (float*)myMalloc(1 * sizeof(float));;
x4364[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4362,x676,1,x4364, x1234, 1, x676,1));
arrayFill_greg<<<1, 512>>>(x1234, 0.0f, 128);
float* x4368 = (float*)myMalloc(1 * sizeof(float));;
x4368[0] = 1.0f;
float* x4370 = (float*)myMalloc(1 * sizeof(float));;
x4370[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4368,x679,1,x4370, x1235, 1, x679,1));
arrayFill_greg<<<1, 512>>>(x1235, 0.0f, 128);
float* x4374 = (float*)myMalloc(1 * sizeof(float));;
x4374[0] = 1.0f;
float* x4376 = (float*)myMalloc(1 * sizeof(float));;
x4376[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4374,x682,1,x4376, x1236, 1, x682,1));
arrayFill_greg<<<1, 512>>>(x1236, 0.0f, 256);
float* x4380 = (float*)myMalloc(1 * sizeof(float));;
x4380[0] = 1.0f;
float* x4382 = (float*)myMalloc(1 * sizeof(float));;
x4382[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x4380,x685,64,x4382, x1237, 64, x685,64));
arrayFill_greg<<<4, 512>>>(x1237, 0.0f, 16384);
float* x4386 = (float*)myMalloc(1 * sizeof(float));;
x4386[0] = 1.0f;
float* x4388 = (float*)myMalloc(1 * sizeof(float));;
x4388[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4386,x688,1,x4388, x1238, 1, x688,1));
arrayFill_greg<<<1, 512>>>(x1238, 0.0f, 256);
float* x4392 = (float*)myMalloc(1 * sizeof(float));;
x4392[0] = 1.0f;
float* x4394 = (float*)myMalloc(1 * sizeof(float));;
x4394[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,128,x4392,x691,512,x4394, x1239, 512, x691,512));
arrayFill_greg<<<13, 512>>>(x1239, 0.0f, 65536);
float* x4398 = (float*)myMalloc(1 * sizeof(float));;
x4398[0] = 1.0f;
float* x4400 = (float*)myMalloc(1 * sizeof(float));;
x4400[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4398,x694,1,x4400, x1240, 1, x694,1));
arrayFill_greg<<<1, 512>>>(x1240, 0.0f, 256);
float* x4404 = (float*)myMalloc(1 * sizeof(float));;
x4404[0] = 1.0f;
float* x4406 = (float*)myMalloc(1 * sizeof(float));;
x4406[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4404,x697,1,x4406, x1241, 1, x697,1));
arrayFill_greg<<<1, 512>>>(x1241, 0.0f, 128);
float* x4410 = (float*)myMalloc(1 * sizeof(float));;
x4410[0] = 1.0f;
float* x4412 = (float*)myMalloc(1 * sizeof(float));;
x4412[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4410,x700,1,x4412, x1242, 1, x700,1));
arrayFill_greg<<<1, 512>>>(x1242, 0.0f, 64);
float* x4416 = (float*)myMalloc(1 * sizeof(float));;
x4416[0] = 1.0f;
float* x4418 = (float*)myMalloc(1 * sizeof(float));;
x4418[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4416,x703,1,x4418, x1243, 1, x703,1));
arrayFill_greg<<<1, 512>>>(x1243, 0.0f, 256);
float* x4422 = (float*)myMalloc(1 * sizeof(float));;
x4422[0] = 1.0f;
float* x4424 = (float*)myMalloc(1 * sizeof(float));;
x4424[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4422,x706,1,x4424, x1244, 1, x706,1));
arrayFill_greg<<<1, 512>>>(x1244, 0.0f, 512);
float* x4428 = (float*)myMalloc(1 * sizeof(float));;
x4428[0] = 1.0f;
float* x4430 = (float*)myMalloc(1 * sizeof(float));;
x4430[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4428,x709,1,x4430, x1245, 1, x709,1));
arrayFill_greg<<<1, 512>>>(x1245, 0.0f, 512);
float* x4434 = (float*)myMalloc(1 * sizeof(float));;
x4434[0] = 1.0f;
float* x4436 = (float*)myMalloc(1 * sizeof(float));;
x4436[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,512,x4434,x712,1024,x4436, x1246, 1024, x712,1024));
arrayFill_greg<<<103, 512>>>(x1246, 0.0f, 524288);
float* x4440 = (float*)myMalloc(1 * sizeof(float));;
x4440[0] = 1.0f;
float* x4442 = (float*)myMalloc(1 * sizeof(float));;
x4442[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4440,x715,1,x4442, x1247, 1, x715,1));
arrayFill_greg<<<1, 512>>>(x1247, 0.0f, 1024);
float* x4446 = (float*)myMalloc(1 * sizeof(float));;
x4446[0] = 1.0f;
float* x4448 = (float*)myMalloc(1 * sizeof(float));;
x4448[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4446,x718,1,x4448, x1248, 1, x718,1));
arrayFill_greg<<<1, 512>>>(x1248, 0.0f, 256);
float* x4452 = (float*)myMalloc(1 * sizeof(float));;
x4452[0] = 1.0f;
float* x4454 = (float*)myMalloc(1 * sizeof(float));;
x4454[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4452,x721,1,x4454, x1249, 1, x721,1));
arrayFill_greg<<<1, 512>>>(x1249, 0.0f, 64);
float* x4458 = (float*)myMalloc(1 * sizeof(float));;
x4458[0] = 1.0f;
float* x4460 = (float*)myMalloc(1 * sizeof(float));;
x4460[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4458,x724,1,x4460, x1250, 1, x724,1));
arrayFill_greg<<<1, 512>>>(x1250, 0.0f, 1024);
float* x4464 = (float*)myMalloc(1 * sizeof(float));;
x4464[0] = 1.0f;
float* x4466 = (float*)myMalloc(1 * sizeof(float));;
x4466[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4464,x727,1,x4466, x1251, 1, x727,1));
arrayFill_greg<<<1, 512>>>(x1251, 0.0f, 2048);
float* x4470 = (float*)myMalloc(1 * sizeof(float));;
x4470[0] = 1.0f;
float* x4472 = (float*)myMalloc(1 * sizeof(float));;
x4472[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4470,x730,1,x4472, x1252, 1, x730,1));
arrayFill_greg<<<1, 512>>>(x1252, 0.0f, 512);
float* x4476 = (float*)myMalloc(1 * sizeof(float));;
x4476[0] = 1.0f;
float* x4478 = (float*)myMalloc(1 * sizeof(float));;
x4478[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4476,x733,1,x4478, x1253, 1, x733,1));
arrayFill_greg<<<1, 512>>>(x1253, 0.0f, 1024);
float* x4482 = (float*)myMalloc(1 * sizeof(float));;
x4482[0] = 1.0f;
float* x4484 = (float*)myMalloc(1 * sizeof(float));;
x4484[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4482,x736,1,x4484, x1254, 1, x736,1));
arrayFill_greg<<<1, 512>>>(x1254, 0.0f, 512);
float* x4488 = (float*)myMalloc(1 * sizeof(float));;
x4488[0] = 1.0f;
float* x4490 = (float*)myMalloc(1 * sizeof(float));;
x4490[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4488,x739,1,x4490, x1255, 1, x739,1));
arrayFill_greg<<<1, 512>>>(x1255, 0.0f, 128);
float* x4494 = (float*)myMalloc(1 * sizeof(float));;
x4494[0] = 1.0f;
float* x4496 = (float*)myMalloc(1 * sizeof(float));;
x4496[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4494,x742,1,x4496, x1256, 1, x742,1));
arrayFill_greg<<<1, 512>>>(x1256, 0.0f, 512);
float* x4500 = (float*)myMalloc(1 * sizeof(float));;
x4500[0] = 1.0f;
float* x4502 = (float*)myMalloc(1 * sizeof(float));;
x4502[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,64,x4500,x745,256,x4502, x1257, 256, x745,256));
arrayFill_greg<<<4, 512>>>(x1257, 0.0f, 16384);
float* x4506 = (float*)myMalloc(1 * sizeof(float));;
x4506[0] = 1.0f;
float* x4508 = (float*)myMalloc(1 * sizeof(float));;
x4508[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,256,x4506,x748,1024,x4508, x1258, 1024, x748,1024));
arrayFill_greg<<<52, 512>>>(x1258, 0.0f, 262144);
float* x4512 = (float*)myMalloc(1 * sizeof(float));;
x4512[0] = 1.0f;
float* x4514 = (float*)myMalloc(1 * sizeof(float));;
x4514[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 27,64,x4512,x751,27,x4514, x1259, 27, x751,27));
arrayFill_greg<<<1, 512>>>(x1259, 0.0f, 1728);
float* x4518 = (float*)myMalloc(1 * sizeof(float));;
x4518[0] = 1.0f;
float* x4520 = (float*)myMalloc(1 * sizeof(float));;
x4520[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4518,x754,1,x4520, x1260, 1, x754,1));
arrayFill_greg<<<1, 512>>>(x1260, 0.0f, 64);
float* x4524 = (float*)myMalloc(1 * sizeof(float));;
x4524[0] = 1.0f;
float* x4526 = (float*)myMalloc(1 * sizeof(float));;
x4526[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4524,x757,1,x4526, x1261, 1, x757,1));
arrayFill_greg<<<1, 512>>>(x1261, 0.0f, 512);
float* x4530 = (float*)myMalloc(1 * sizeof(float));;
x4530[0] = 1.0f;
float* x4532 = (float*)myMalloc(1 * sizeof(float));;
x4532[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 4608,512,x4530,x760,4608,x4532, x1262, 4608, x760,4608));
arrayFill_greg<<<461, 512>>>(x1262, 0.0f, 2359296);
float* x4536 = (float*)myMalloc(1 * sizeof(float));;
x4536[0] = 1.0f;
float* x4538 = (float*)myMalloc(1 * sizeof(float));;
x4538[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4536,x763,1,x4538, x1263, 1, x763,1));
arrayFill_greg<<<1, 512>>>(x1263, 0.0f, 512);
float* x4542 = (float*)myMalloc(1 * sizeof(float));;
x4542[0] = 1.0f;
float* x4544 = (float*)myMalloc(1 * sizeof(float));;
x4544[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4542,x766,1,x4544, x1264, 1, x766,1));
arrayFill_greg<<<1, 512>>>(x1264, 0.0f, 256);
float* x4548 = (float*)myMalloc(1 * sizeof(float));;
x4548[0] = 1.0f;
float* x4550 = (float*)myMalloc(1 * sizeof(float));;
x4550[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4548,x769,1,x4550, x1265, 1, x769,1));
arrayFill_greg<<<1, 512>>>(x1265, 0.0f, 64);
float* x4554 = (float*)myMalloc(1 * sizeof(float));;
x4554[0] = 1.0f;
float* x4556 = (float*)myMalloc(1 * sizeof(float));;
x4556[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4554,x772,1,x4556, x1266, 1, x772,1));
arrayFill_greg<<<1, 512>>>(x1266, 0.0f, 512);
float* x4560 = (float*)myMalloc(1 * sizeof(float));;
x4560[0] = 1.0f;
float* x4562 = (float*)myMalloc(1 * sizeof(float));;
x4562[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4560,x775,1,x4562, x1267, 1, x775,1));
arrayFill_greg<<<1, 512>>>(x1267, 0.0f, 512);
float* x4566 = (float*)myMalloc(1 * sizeof(float));;
x4566[0] = 1.0f;
float* x4568 = (float*)myMalloc(1 * sizeof(float));;
x4568[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4566,x778,1,x4568, x1268, 1, x778,1));
arrayFill_greg<<<1, 512>>>(x1268, 0.0f, 1024);
float* x4572 = (float*)myMalloc(1 * sizeof(float));;
x4572[0] = 1.0f;
float* x4574 = (float*)myMalloc(1 * sizeof(float));;
x4574[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x4572,x781,64,x4574, x1269, 64, x781,64));
arrayFill_greg<<<4, 512>>>(x1269, 0.0f, 16384);
float* x4578 = (float*)myMalloc(1 * sizeof(float));;
x4578[0] = 1.0f;
float* x4580 = (float*)myMalloc(1 * sizeof(float));;
x4580[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4578,x784,1,x4580, x1270, 1, x784,1));
arrayFill_greg<<<1, 512>>>(x1270, 0.0f, 256);
float* x4584 = (float*)myMalloc(1 * sizeof(float));;
x4584[0] = 1.0f;
float* x4586 = (float*)myMalloc(1 * sizeof(float));;
x4586[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4584,x787,1,x4586, x1271, 1, x787,1));
arrayFill_greg<<<1, 512>>>(x1271, 0.0f, 64);
float* x4590 = (float*)myMalloc(1 * sizeof(float));;
x4590[0] = 1.0f;
float* x4592 = (float*)myMalloc(1 * sizeof(float));;
x4592[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1152,128,x4590,x790,1152,x4592, x1272, 1152, x790,1152));
arrayFill_greg<<<29, 512>>>(x1272, 0.0f, 147456);
float* x4596 = (float*)myMalloc(1 * sizeof(float));;
x4596[0] = 1.0f;
float* x4598 = (float*)myMalloc(1 * sizeof(float));;
x4598[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4596,x793,1,x4598, x1273, 1, x793,1));
arrayFill_greg<<<1, 512>>>(x1273, 0.0f, 256);
float* x4602 = (float*)myMalloc(1 * sizeof(float));;
x4602[0] = 1.0f;
float* x4604 = (float*)myMalloc(1 * sizeof(float));;
x4604[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4602,x796,1,x4604, x1274, 1, x796,1));
arrayFill_greg<<<1, 512>>>(x1274, 0.0f, 512);
float* x4608 = (float*)myMalloc(1 * sizeof(float));;
x4608[0] = 1.0f;
float* x4610 = (float*)myMalloc(1 * sizeof(float));;
x4610[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4608,x799,1,x4610, x1275, 1, x799,1));
arrayFill_greg<<<1, 512>>>(x1275, 0.0f, 256);
float* x4614 = (float*)myMalloc(1 * sizeof(float));;
x4614[0] = 1.0f;
float* x4616 = (float*)myMalloc(1 * sizeof(float));;
x4616[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4614,x802,1,x4616, x1276, 1, x802,1));
arrayFill_greg<<<1, 512>>>(x1276, 0.0f, 512);
float* x4620 = (float*)myMalloc(1 * sizeof(float));;
x4620[0] = 1.0f;
float* x4622 = (float*)myMalloc(1 * sizeof(float));;
x4622[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4620,x805,1,x4622, x1277, 1, x805,1));
arrayFill_greg<<<1, 512>>>(x1277, 0.0f, 128);
float* x4626 = (float*)myMalloc(1 * sizeof(float));;
x4626[0] = 1.0f;
float* x4628 = (float*)myMalloc(1 * sizeof(float));;
x4628[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,64,x4626,x808,256,x4628, x1278, 256, x808,256));
arrayFill_greg<<<4, 512>>>(x1278, 0.0f, 16384);
float* x4632 = (float*)myMalloc(1 * sizeof(float));;
x4632[0] = 1.0f;
float* x4634 = (float*)myMalloc(1 * sizeof(float));;
x4634[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4632,x811,1,x4634, x1279, 1, x811,1));
arrayFill_greg<<<1, 512>>>(x1279, 0.0f, 128);
float* x4638 = (float*)myMalloc(1 * sizeof(float));;
x4638[0] = 1.0f;
float* x4640 = (float*)myMalloc(1 * sizeof(float));;
x4640[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4638,x814,1,x4640, x1280, 1, x814,1));
arrayFill_greg<<<1, 512>>>(x1280, 0.0f, 2048);
float* x4644 = (float*)myMalloc(1 * sizeof(float));;
x4644[0] = 1.0f;
float* x4646 = (float*)myMalloc(1 * sizeof(float));;
x4646[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4644,x817,1,x4646, x1281, 1, x817,1));
arrayFill_greg<<<1, 512>>>(x1281, 0.0f, 256);
float* x4650 = (float*)myMalloc(1 * sizeof(float));;
x4650[0] = 1.0f;
float* x4652 = (float*)myMalloc(1 * sizeof(float));;
x4652[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x4650,x820,2304,x4652, x1282, 2304, x820,2304));
arrayFill_greg<<<116, 512>>>(x1282, 0.0f, 589824);
float* x4656 = (float*)myMalloc(1 * sizeof(float));;
x4656[0] = 1.0f;
float* x4658 = (float*)myMalloc(1 * sizeof(float));;
x4658[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4656,x823,1,x4658, x1283, 1, x823,1));
arrayFill_greg<<<1, 512>>>(x1283, 0.0f, 256);
float* x4662 = (float*)myMalloc(1 * sizeof(float));;
x4662[0] = 1.0f;
float* x4664 = (float*)myMalloc(1 * sizeof(float));;
x4664[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4662,x826,1,x4664, x1284, 1, x826,1));
arrayFill_greg<<<1, 512>>>(x1284, 0.0f, 128);
float* x4668 = (float*)myMalloc(1 * sizeof(float));;
x4668[0] = 1.0f;
float* x4670 = (float*)myMalloc(1 * sizeof(float));;
x4670[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4668,x829,1,x4670, x1285, 1, x829,1));
arrayFill_greg<<<1, 512>>>(x1285, 0.0f, 256);
float* x4674 = (float*)myMalloc(1 * sizeof(float));;
x4674[0] = 1.0f;
float* x4676 = (float*)myMalloc(1 * sizeof(float));;
x4676[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4674,x832,1,x4676, x1286, 1, x832,1));
arrayFill_greg<<<1, 512>>>(x1286, 0.0f, 64);
float* x4680 = (float*)myMalloc(1 * sizeof(float));;
x4680[0] = 1.0f;
float* x4682 = (float*)myMalloc(1 * sizeof(float));;
x4682[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,256,x4680,x835,512,x4682, x1287, 512, x835,512));
arrayFill_greg<<<26, 512>>>(x1287, 0.0f, 131072);
float* x4686 = (float*)myMalloc(1 * sizeof(float));;
x4686[0] = 1.0f;
float* x4688 = (float*)myMalloc(1 * sizeof(float));;
x4688[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4686,x838,1,x4688, x1288, 1, x838,1));
arrayFill_greg<<<1, 512>>>(x1288, 0.0f, 2048);
float* x4692 = (float*)myMalloc(1 * sizeof(float));;
x4692[0] = 1.0f;
float* x4694 = (float*)myMalloc(1 * sizeof(float));;
x4694[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4692,x841,1,x4694, x1289, 1, x841,1));
arrayFill_greg<<<1, 512>>>(x1289, 0.0f, 1024);
float* x4698 = (float*)myMalloc(1 * sizeof(float));;
x4698[0] = 1.0f;
float* x4700 = (float*)myMalloc(1 * sizeof(float));;
x4700[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4698,x844,1,x4700, x1290, 1, x844,1));
arrayFill_greg<<<1, 512>>>(x1290, 0.0f, 1024);
float* x4704 = (float*)myMalloc(1 * sizeof(float));;
x4704[0] = 1.0f;
float* x4706 = (float*)myMalloc(1 * sizeof(float));;
x4706[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4704,x847,1,x4706, x1291, 1, x847,1));
arrayFill_greg<<<1, 512>>>(x1291, 0.0f, 256);
float* x4710 = (float*)myMalloc(1 * sizeof(float));;
x4710[0] = 1.0f;
float* x4712 = (float*)myMalloc(1 * sizeof(float));;
x4712[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4710,x850,1,x4712, x1292, 1, x850,1));
arrayFill_greg<<<1, 512>>>(x1292, 0.0f, 256);
float* x4716 = (float*)myMalloc(1 * sizeof(float));;
x4716[0] = 1.0f;
float* x4718 = (float*)myMalloc(1 * sizeof(float));;
x4718[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4716,x853,1,x4718, x1293, 1, x853,1));
arrayFill_greg<<<1, 512>>>(x1293, 0.0f, 256);
float* x4722 = (float*)myMalloc(1 * sizeof(float));;
x4722[0] = 1.0f;
float* x4724 = (float*)myMalloc(1 * sizeof(float));;
x4724[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4722,x856,1,x4724, x1294, 1, x856,1));
arrayFill_greg<<<1, 512>>>(x1294, 0.0f, 64);
float* x4728 = (float*)myMalloc(1 * sizeof(float));;
x4728[0] = 1.0f;
float* x4730 = (float*)myMalloc(1 * sizeof(float));;
x4730[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4728,x859,1,x4730, x1295, 1, x859,1));
arrayFill_greg<<<1, 512>>>(x1295, 0.0f, 1024);
float* x4734 = (float*)myMalloc(1 * sizeof(float));;
x4734[0] = 1.0f;
float* x4736 = (float*)myMalloc(1 * sizeof(float));;
x4736[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4734,x862,1,x4736, x1296, 1, x862,1));
arrayFill_greg<<<1, 512>>>(x1296, 0.0f, 256);
float* x4740 = (float*)myMalloc(1 * sizeof(float));;
x4740[0] = 1.0f;
float* x4742 = (float*)myMalloc(1 * sizeof(float));;
x4742[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4740,x865,1,x4742, x1297, 1, x865,1));
arrayFill_greg<<<1, 512>>>(x1297, 0.0f, 128);
float* x4746 = (float*)myMalloc(1 * sizeof(float));;
x4746[0] = 1.0f;
float* x4748 = (float*)myMalloc(1 * sizeof(float));;
x4748[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1152,128,x4746,x868,1152,x4748, x1298, 1152, x868,1152));
arrayFill_greg<<<29, 512>>>(x1298, 0.0f, 147456);
float* x4752 = (float*)myMalloc(1 * sizeof(float));;
x4752[0] = 1.0f;
float* x4754 = (float*)myMalloc(1 * sizeof(float));;
x4754[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4752,x871,1,x4754, x1299, 1, x871,1));
arrayFill_greg<<<1, 512>>>(x1299, 0.0f, 256);
float* x4758 = (float*)myMalloc(1 * sizeof(float));;
x4758[0] = 1.0f;
float* x4760 = (float*)myMalloc(1 * sizeof(float));;
x4760[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4758,x874,1,x4760, x1300, 1, x874,1));
arrayFill_greg<<<1, 512>>>(x1300, 0.0f, 2048);
float* x4764 = (float*)myMalloc(1 * sizeof(float));;
x4764[0] = 1.0f;
float* x4766 = (float*)myMalloc(1 * sizeof(float));;
x4766[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4764,x877,1,x4766, x1301, 1, x877,1));
arrayFill_greg<<<1, 512>>>(x1301, 0.0f, 512);
float* x4770 = (float*)myMalloc(1 * sizeof(float));;
x4770[0] = 1.0f;
float* x4772 = (float*)myMalloc(1 * sizeof(float));;
x4772[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4770,x880,1,x4772, x1302, 1, x880,1));
arrayFill_greg<<<1, 512>>>(x1302, 0.0f, 512);
float* x4776 = (float*)myMalloc(1 * sizeof(float));;
x4776[0] = 1.0f;
float* x4778 = (float*)myMalloc(1 * sizeof(float));;
x4778[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,128,x4776,x883,512,x4778, x1303, 512, x883,512));
arrayFill_greg<<<13, 512>>>(x1303, 0.0f, 65536);
float* x4782 = (float*)myMalloc(1 * sizeof(float));;
x4782[0] = 1.0f;
float* x4784 = (float*)myMalloc(1 * sizeof(float));;
x4784[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4782,x886,1,x4784, x1304, 1, x886,1));
arrayFill_greg<<<1, 512>>>(x1304, 0.0f, 256);
float* x4788 = (float*)myMalloc(1 * sizeof(float));;
x4788[0] = 1.0f;
float* x4790 = (float*)myMalloc(1 * sizeof(float));;
x4790[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4788,x889,1,x4790, x1305, 1, x889,1));
arrayFill_greg<<<1, 512>>>(x1305, 0.0f, 256);
float* x4794 = (float*)myMalloc(1 * sizeof(float));;
x4794[0] = 1.0f;
float* x4796 = (float*)myMalloc(1 * sizeof(float));;
x4796[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4794,x892,1,x4796, x1306, 1, x892,1));
arrayFill_greg<<<1, 512>>>(x1306, 0.0f, 256);
float* x4800 = (float*)myMalloc(1 * sizeof(float));;
x4800[0] = 1.0f;
float* x4802 = (float*)myMalloc(1 * sizeof(float));;
x4802[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4800,x895,1,x4802, x1307, 1, x895,1));
arrayFill_greg<<<1, 512>>>(x1307, 0.0f, 256);
float* x4806 = (float*)myMalloc(1 * sizeof(float));;
x4806[0] = 1.0f;
float* x4808 = (float*)myMalloc(1 * sizeof(float));;
x4808[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4806,x898,1,x4808, x1308, 1, x898,1));
arrayFill_greg<<<1, 512>>>(x1308, 0.0f, 512);
float* x4812 = (float*)myMalloc(1 * sizeof(float));;
x4812[0] = 1.0f;
float* x4814 = (float*)myMalloc(1 * sizeof(float));;
x4814[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4812,x901,1,x4814, x1309, 1, x901,1));
arrayFill_greg<<<1, 512>>>(x1309, 0.0f, 512);
float* x4818 = (float*)myMalloc(1 * sizeof(float));;
x4818[0] = 1.0f;
float* x4820 = (float*)myMalloc(1 * sizeof(float));;
x4820[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4818,x904,1,x4820, x1310, 1, x904,1));
arrayFill_greg<<<1, 512>>>(x1310, 0.0f, 256);
float* x4824 = (float*)myMalloc(1 * sizeof(float));;
x4824[0] = 1.0f;
float* x4826 = (float*)myMalloc(1 * sizeof(float));;
x4826[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4824,x907,1,x4826, x1311, 1, x907,1));
arrayFill_greg<<<1, 512>>>(x1311, 0.0f, 128);
float* x4830 = (float*)myMalloc(1 * sizeof(float));;
x4830[0] = 1.0f;
float* x4832 = (float*)myMalloc(1 * sizeof(float));;
x4832[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4830,x910,1,x4832, x1312, 1, x910,1));
arrayFill_greg<<<1, 512>>>(x1312, 0.0f, 512);
float* x4836 = (float*)myMalloc(1 * sizeof(float));;
x4836[0] = 1.0f;
float* x4838 = (float*)myMalloc(1 * sizeof(float));;
x4838[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4836,x913,1,x4838, x1313, 1, x913,1));
arrayFill_greg<<<1, 512>>>(x1313, 0.0f, 64);
float* x4842 = (float*)myMalloc(1 * sizeof(float));;
x4842[0] = 1.0f;
float* x4844 = (float*)myMalloc(1 * sizeof(float));;
x4844[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4842,x916,1,x4844, x1314, 1, x916,1));
arrayFill_greg<<<1, 512>>>(x1314, 0.0f, 512);
float* x4848 = (float*)myMalloc(1 * sizeof(float));;
x4848[0] = 1.0f;
float* x4850 = (float*)myMalloc(1 * sizeof(float));;
x4850[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4848,x919,1,x4850, x1315, 1, x919,1));
arrayFill_greg<<<1, 512>>>(x1315, 0.0f, 64);
float* x4854 = (float*)myMalloc(1 * sizeof(float));;
x4854[0] = 1.0f;
float* x4856 = (float*)myMalloc(1 * sizeof(float));;
x4856[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4854,x922,1,x4856, x1316, 1, x922,1));
arrayFill_greg<<<1, 512>>>(x1316, 0.0f, 1024);
float* x4860 = (float*)myMalloc(1 * sizeof(float));;
x4860[0] = 1.0f;
float* x4862 = (float*)myMalloc(1 * sizeof(float));;
x4862[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4860,x925,1,x4862, x1317, 1, x925,1));
arrayFill_greg<<<1, 512>>>(x1317, 0.0f, 512);
float* x4866 = (float*)myMalloc(1 * sizeof(float));;
x4866[0] = 1.0f;
float* x4868 = (float*)myMalloc(1 * sizeof(float));;
x4868[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4866,x928,1,x4868, x1318, 1, x928,1));
arrayFill_greg<<<1, 512>>>(x1318, 0.0f, 1024);
float* x4872 = (float*)myMalloc(1 * sizeof(float));;
x4872[0] = 1.0f;
float* x4874 = (float*)myMalloc(1 * sizeof(float));;
x4874[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,2048,x4872,x931,512,x4874, x1319, 512, x931,512));
arrayFill_greg<<<205, 512>>>(x1319, 0.0f, 1048576);
float* x4878 = (float*)myMalloc(1 * sizeof(float));;
x4878[0] = 1.0f;
float* x4880 = (float*)myMalloc(1 * sizeof(float));;
x4880[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4878,x934,1,x4880, x1320, 1, x934,1));
arrayFill_greg<<<1, 512>>>(x1320, 0.0f, 512);
float* x4884 = (float*)myMalloc(1 * sizeof(float));;
x4884[0] = 1.0f;
float* x4886 = (float*)myMalloc(1 * sizeof(float));;
x4886[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,2048,x4884,x937,1024,x4886, x1321, 1024, x937,1024));
arrayFill_greg<<<410, 512>>>(x1321, 0.0f, 2097152);
float* x4890 = (float*)myMalloc(1 * sizeof(float));;
x4890[0] = 1.0f;
float* x4892 = (float*)myMalloc(1 * sizeof(float));;
x4892[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2048,512,x4890,x940,2048,x4892, x1322, 2048, x940,2048));
arrayFill_greg<<<205, 512>>>(x1322, 0.0f, 1048576);
float* x4896 = (float*)myMalloc(1 * sizeof(float));;
x4896[0] = 1.0f;
float* x4898 = (float*)myMalloc(1 * sizeof(float));;
x4898[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4896,x943,1,x4898, x1323, 1, x943,1));
arrayFill_greg<<<1, 512>>>(x1323, 0.0f, 1024);
float* x4902 = (float*)myMalloc(1 * sizeof(float));;
x4902[0] = 1.0f;
float* x4904 = (float*)myMalloc(1 * sizeof(float));;
x4904[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4902,x946,1,x4904, x1324, 1, x946,1));
arrayFill_greg<<<1, 512>>>(x1324, 0.0f, 128);
float* x4908 = (float*)myMalloc(1 * sizeof(float));;
x4908[0] = 1.0f;
float* x4910 = (float*)myMalloc(1 * sizeof(float));;
x4910[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,256,x4908,x949,1024,x4910, x1325, 1024, x949,1024));
arrayFill_greg<<<52, 512>>>(x1325, 0.0f, 262144);
float* x4914 = (float*)myMalloc(1 * sizeof(float));;
x4914[0] = 1.0f;
float* x4916 = (float*)myMalloc(1 * sizeof(float));;
x4916[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4914,x952,1,x4916, x1326, 1, x952,1));
arrayFill_greg<<<1, 512>>>(x1326, 0.0f, 256);
float* x4920 = (float*)myMalloc(1 * sizeof(float));;
x4920[0] = 1.0f;
float* x4922 = (float*)myMalloc(1 * sizeof(float));;
x4922[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4920,x955,1,x4922, x1327, 1, x955,1));
arrayFill_greg<<<1, 512>>>(x1327, 0.0f, 1024);
float* x4926 = (float*)myMalloc(1 * sizeof(float));;
x4926[0] = 1.0f;
float* x4928 = (float*)myMalloc(1 * sizeof(float));;
x4928[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x4926,x958,256,x4928, x1328, 256, x958,256));
arrayFill_greg<<<52, 512>>>(x1328, 0.0f, 262144);
float* x4932 = (float*)myMalloc(1 * sizeof(float));;
x4932[0] = 1.0f;
float* x4934 = (float*)myMalloc(1 * sizeof(float));;
x4934[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4932,x961,1,x4934, x1329, 1, x961,1));
arrayFill_greg<<<1, 512>>>(x1329, 0.0f, 128);
float* x4938 = (float*)myMalloc(1 * sizeof(float));;
x4938[0] = 1.0f;
float* x4940 = (float*)myMalloc(1 * sizeof(float));;
x4940[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4938,x964,1,x4940, x1330, 1, x964,1));
arrayFill_greg<<<1, 512>>>(x1330, 0.0f, 512);
float* x4944 = (float*)myMalloc(1 * sizeof(float));;
x4944[0] = 1.0f;
float* x4946 = (float*)myMalloc(1 * sizeof(float));;
x4946[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4944,x967,1,x4946, x1331, 1, x967,1));
arrayFill_greg<<<1, 512>>>(x1331, 0.0f, 512);
float* x4950 = (float*)myMalloc(1 * sizeof(float));;
x4950[0] = 1.0f;
float* x4952 = (float*)myMalloc(1 * sizeof(float));;
x4952[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4950,x970,1,x4952, x1332, 1, x970,1));
arrayFill_greg<<<1, 512>>>(x1332, 0.0f, 128);
float* x4956 = (float*)myMalloc(1 * sizeof(float));;
x4956[0] = 1.0f;
float* x4958 = (float*)myMalloc(1 * sizeof(float));;
x4958[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x4956,x973,2304,x4958, x1333, 2304, x973,2304));
arrayFill_greg<<<116, 512>>>(x1333, 0.0f, 589824);
float* x4962 = (float*)myMalloc(1 * sizeof(float));;
x4962[0] = 1.0f;
float* x4964 = (float*)myMalloc(1 * sizeof(float));;
x4964[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2048,10,x4962,x976,2048,x4964, x1334, 2048, x976,2048));
arrayFill_greg<<<5, 512>>>(x1334, 0.0f, 20480);
float* x4968 = (float*)myMalloc(1 * sizeof(float));;
x4968[0] = 1.0f;
float* x4970 = (float*)myMalloc(1 * sizeof(float));;
x4970[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4968,x979,1,x4970, x1335, 1, x979,1));
arrayFill_greg<<<1, 512>>>(x1335, 0.0f, 256);
float* x4974 = (float*)myMalloc(1 * sizeof(float));;
x4974[0] = 1.0f;
float* x4976 = (float*)myMalloc(1 * sizeof(float));;
x4976[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4974,x982,1,x4976, x1336, 1, x982,1));
arrayFill_greg<<<1, 512>>>(x1336, 0.0f, 256);
float* x4980 = (float*)myMalloc(1 * sizeof(float));;
x4980[0] = 1.0f;
float* x4982 = (float*)myMalloc(1 * sizeof(float));;
x4982[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4980,x985,1,x4982, x1337, 1, x985,1));
arrayFill_greg<<<1, 512>>>(x1337, 0.0f, 256);
float* x4986 = (float*)myMalloc(1 * sizeof(float));;
x4986[0] = 1.0f;
float* x4988 = (float*)myMalloc(1 * sizeof(float));;
x4988[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4986,x988,1,x4988, x1338, 1, x988,1));
arrayFill_greg<<<1, 512>>>(x1338, 0.0f, 1024);
float* x4992 = (float*)myMalloc(1 * sizeof(float));;
x4992[0] = 1.0f;
float* x4994 = (float*)myMalloc(1 * sizeof(float));;
x4994[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4992,x991,1,x4994, x1339, 1, x991,1));
arrayFill_greg<<<1, 512>>>(x1339, 0.0f, 1024);
float* x4998 = (float*)myMalloc(1 * sizeof(float));;
x4998[0] = 1.0f;
float* x5000 = (float*)myMalloc(1 * sizeof(float));;
x5000[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,64,x4998,x994,64,x5000, x1340, 64, x994,64));
arrayFill_greg<<<1, 512>>>(x1340, 0.0f, 4096);
float* x5004 = (float*)myMalloc(1 * sizeof(float));;
x5004[0] = 1.0f;
float* x5006 = (float*)myMalloc(1 * sizeof(float));;
x5006[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5004,x997,1,x5006, x1341, 1, x997,1));
arrayFill_greg<<<1, 512>>>(x1341, 0.0f, 512);
float* x5010 = (float*)myMalloc(1 * sizeof(float));;
x5010[0] = 1.0f;
float* x5012 = (float*)myMalloc(1 * sizeof(float));;
x5012[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1152,128,x5010,x1000,1152,x5012, x1342, 1152, x1000,1152));
arrayFill_greg<<<29, 512>>>(x1342, 0.0f, 147456);
float* x5016 = (float*)myMalloc(1 * sizeof(float));;
x5016[0] = 1.0f;
float* x5018 = (float*)myMalloc(1 * sizeof(float));;
x5018[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5016,x1003,1,x5018, x1343, 1, x1003,1));
arrayFill_greg<<<1, 512>>>(x1343, 0.0f, 128);
float* x5022 = (float*)myMalloc(1 * sizeof(float));;
x5022[0] = 1.0f;
float* x5024 = (float*)myMalloc(1 * sizeof(float));;
x5024[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5022,x1006,1,x5024, x1344, 1, x1006,1));
arrayFill_greg<<<1, 512>>>(x1344, 0.0f, 256);
float* x5028 = (float*)myMalloc(1 * sizeof(float));;
x5028[0] = 1.0f;
float* x5030 = (float*)myMalloc(1 * sizeof(float));;
x5030[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5028,x1009,1,x5030, x1345, 1, x1009,1));
arrayFill_greg<<<1, 512>>>(x1345, 0.0f, 1024);
float* x5034 = (float*)myMalloc(1 * sizeof(float));;
x5034[0] = 1.0f;
float* x5036 = (float*)myMalloc(1 * sizeof(float));;
x5036[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5034,x1012,1,x5036, x1346, 1, x1012,1));
arrayFill_greg<<<1, 512>>>(x1346, 0.0f, 2048);
float* x5040 = (float*)myMalloc(1 * sizeof(float));;
x5040[0] = 1.0f;
float* x5042 = (float*)myMalloc(1 * sizeof(float));;
x5042[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5040,x1015,1,x5042, x1347, 1, x1015,1));
arrayFill_greg<<<1, 512>>>(x1347, 0.0f, 256);
float* x5046 = (float*)myMalloc(1 * sizeof(float));;
x5046[0] = 1.0f;
float* x5048 = (float*)myMalloc(1 * sizeof(float));;
x5048[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5046,x1018,1,x5048, x1348, 1, x1018,1));
arrayFill_greg<<<1, 512>>>(x1348, 0.0f, 256);
float* x5052 = (float*)myMalloc(1 * sizeof(float));;
x5052[0] = 1.0f;
float* x5054 = (float*)myMalloc(1 * sizeof(float));;
x5054[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5052,x1021,1,x5054, x1349, 1, x1021,1));
arrayFill_greg<<<1, 512>>>(x1349, 0.0f, 128);
float* x5058 = (float*)myMalloc(1 * sizeof(float));;
x5058[0] = 1.0f;
float* x5060 = (float*)myMalloc(1 * sizeof(float));;
x5060[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5058,x1024,1,x5060, x1350, 1, x1024,1));
arrayFill_greg<<<1, 512>>>(x1350, 0.0f, 256);
float* x5064 = (float*)myMalloc(1 * sizeof(float));;
x5064[0] = 1.0f;
float* x5066 = (float*)myMalloc(1 * sizeof(float));;
x5066[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5064,x1027,1,x5066, x1351, 1, x1027,1));
arrayFill_greg<<<1, 512>>>(x1351, 0.0f, 64);
float* x5070 = (float*)myMalloc(1 * sizeof(float));;
x5070[0] = 1.0f;
float* x5072 = (float*)myMalloc(1 * sizeof(float));;
x5072[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5070,x1030,1,x5072, x1352, 1, x1030,1));
arrayFill_greg<<<1, 512>>>(x1352, 0.0f, 2048);
float* x5076 = (float*)myMalloc(1 * sizeof(float));;
x5076[0] = 1.0f;
float* x5078 = (float*)myMalloc(1 * sizeof(float));;
x5078[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5076,x1033,1,x5078, x1353, 1, x1033,1));
arrayFill_greg<<<1, 512>>>(x1353, 0.0f, 512);
float* x5082 = (float*)myMalloc(1 * sizeof(float));;
x5082[0] = 1.0f;
float* x5084 = (float*)myMalloc(1 * sizeof(float));;
x5084[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5082,x1036,1,x5084, x1354, 1, x1036,1));
arrayFill_greg<<<1, 512>>>(x1354, 0.0f, 256);
float* x5088 = (float*)myMalloc(1 * sizeof(float));;
x5088[0] = 1.0f;
float* x5090 = (float*)myMalloc(1 * sizeof(float));;
x5090[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5088,x1039,1,x5090, x1355, 1, x1039,1));
arrayFill_greg<<<1, 512>>>(x1355, 0.0f, 1024);
float* x5094 = (float*)myMalloc(1 * sizeof(float));;
x5094[0] = 1.0f;
float* x5096 = (float*)myMalloc(1 * sizeof(float));;
x5096[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x5094,x1042,2304,x5096, x1356, 2304, x1042,2304));
arrayFill_greg<<<116, 512>>>(x1356, 0.0f, 589824);
float* x5100 = (float*)myMalloc(1 * sizeof(float));;
x5100[0] = 1.0f;
float* x5102 = (float*)myMalloc(1 * sizeof(float));;
x5102[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5100,x1045,1,x5102, x1357, 1, x1045,1));
arrayFill_greg<<<1, 512>>>(x1357, 0.0f, 256);
float* x5106 = (float*)myMalloc(1 * sizeof(float));;
x5106[0] = 1.0f;
float* x5108 = (float*)myMalloc(1 * sizeof(float));;
x5108[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5106,x1048,1,x5108, x1358, 1, x1048,1));
arrayFill_greg<<<1, 512>>>(x1358, 0.0f, 64);
float* x5112 = (float*)myMalloc(1 * sizeof(float));;
x5112[0] = 1.0f;
float* x5114 = (float*)myMalloc(1 * sizeof(float));;
x5114[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5112,x1051,1,x5114, x1359, 1, x1051,1));
arrayFill_greg<<<1, 512>>>(x1359, 0.0f, 128);
float* x5118 = (float*)myMalloc(1 * sizeof(float));;
x5118[0] = 1.0f;
float* x5120 = (float*)myMalloc(1 * sizeof(float));;
x5120[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5118,x1054,1,x5120, x1360, 1, x1054,1));
arrayFill_greg<<<1, 512>>>(x1360, 0.0f, 256);
float* x5124 = (float*)myMalloc(1 * sizeof(float));;
x5124[0] = 1.0f;
float* x5126 = (float*)myMalloc(1 * sizeof(float));;
x5126[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5124,x1057,1,x5126, x1361, 1, x1057,1));
arrayFill_greg<<<1, 512>>>(x1361, 0.0f, 256);
float* x5130 = (float*)myMalloc(1 * sizeof(float));;
x5130[0] = 1.0f;
float* x5132 = (float*)myMalloc(1 * sizeof(float));;
x5132[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5130,x1060,1,x5132, x1362, 1, x1060,1));
arrayFill_greg<<<1, 512>>>(x1362, 0.0f, 512);
float* x5136 = (float*)myMalloc(1 * sizeof(float));;
x5136[0] = 1.0f;
float* x5138 = (float*)myMalloc(1 * sizeof(float));;
x5138[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,128,x5136,x1063,512,x5138, x1363, 512, x1063,512));
arrayFill_greg<<<13, 512>>>(x1363, 0.0f, 65536);
float* x5142 = (float*)myMalloc(1 * sizeof(float));;
x5142[0] = 1.0f;
float* x5144 = (float*)myMalloc(1 * sizeof(float));;
x5144[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5142,x1066,1,x5144, x1364, 1, x1066,1));
arrayFill_greg<<<1, 512>>>(x1364, 0.0f, 64);
float* x5148 = (float*)myMalloc(1 * sizeof(float));;
x5148[0] = 1.0f;
float* x5150 = (float*)myMalloc(1 * sizeof(float));;
x5150[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,512,x5148,x1069,256,x5150, x1365, 256, x1069,256));
arrayFill_greg<<<26, 512>>>(x1365, 0.0f, 131072);
float* x5154 = (float*)myMalloc(1 * sizeof(float));;
x5154[0] = 1.0f;
float* x5156 = (float*)myMalloc(1 * sizeof(float));;
x5156[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5154,x1072,1,x5156, x1366, 1, x1072,1));
arrayFill_greg<<<1, 512>>>(x1366, 0.0f, 256);
float* x5160 = (float*)myMalloc(1 * sizeof(float));;
x5160[0] = 1.0f;
float* x5162 = (float*)myMalloc(1 * sizeof(float));;
x5162[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5160,x1075,1,x5162, x1367, 1, x1075,1));
arrayFill_greg<<<1, 512>>>(x1367, 0.0f, 2048);
float* x5166 = (float*)myMalloc(1 * sizeof(float));;
x5166[0] = 1.0f;
float* x5168 = (float*)myMalloc(1 * sizeof(float));;
x5168[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5166,x1078,1,x5168, x1368, 1, x1078,1));
arrayFill_greg<<<1, 512>>>(x1368, 0.0f, 128);
float* x5172 = (float*)myMalloc(1 * sizeof(float));;
x5172[0] = 1.0f;
float* x5174 = (float*)myMalloc(1 * sizeof(float));;
x5174[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x5172,x1081,2304,x5174, x1369, 2304, x1081,2304));
arrayFill_greg<<<116, 512>>>(x1369, 0.0f, 589824);
float* x5178 = (float*)myMalloc(1 * sizeof(float));;
x5178[0] = 1.0f;
float* x5180 = (float*)myMalloc(1 * sizeof(float));;
x5180[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5178,x1084,1,x5180, x1370, 1, x1084,1));
arrayFill_greg<<<1, 512>>>(x1370, 0.0f, 1024);
float* x5184 = (float*)myMalloc(1 * sizeof(float));;
x5184[0] = 1.0f;
float* x5186 = (float*)myMalloc(1 * sizeof(float));;
x5186[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5184,x1087,1,x5186, x1371, 1, x1087,1));
arrayFill_greg<<<1, 512>>>(x1371, 0.0f, 256);
float* x5190 = (float*)myMalloc(1 * sizeof(float));;
x5190[0] = 1.0f;
float* x5192 = (float*)myMalloc(1 * sizeof(float));;
x5192[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2048,512,x5190,x1090,2048,x5192, x1372, 2048, x1090,2048));
arrayFill_greg<<<205, 512>>>(x1372, 0.0f, 1048576);
float* x5196 = (float*)myMalloc(1 * sizeof(float));;
x5196[0] = 1.0f;
float* x5198 = (float*)myMalloc(1 * sizeof(float));;
x5198[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5196,x1093,1,x5198, x1373, 1, x1093,1));
arrayFill_greg<<<1, 512>>>(x1373, 0.0f, 128);
float* x5202 = (float*)myMalloc(1 * sizeof(float));;
x5202[0] = 1.0f;
float* x5204 = (float*)myMalloc(1 * sizeof(float));;
x5204[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5202,x1096,1,x5204, x1374, 1, x1096,1));
arrayFill_greg<<<1, 512>>>(x1374, 0.0f, 1024);
float* x5208 = (float*)myMalloc(1 * sizeof(float));;
x5208[0] = 1.0f;
float* x5210 = (float*)myMalloc(1 * sizeof(float));;
x5210[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5208,x1099,1,x5210, x1375, 1, x1099,1));
arrayFill_greg<<<1, 512>>>(x1375, 0.0f, 128);
float* x5214 = (float*)myMalloc(1 * sizeof(float));;
x5214[0] = 1.0f;
float* x5216 = (float*)myMalloc(1 * sizeof(float));;
x5216[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x5214,x1102,256,x5216, x1376, 256, x1102,256));
arrayFill_greg<<<52, 512>>>(x1376, 0.0f, 262144);
float* x5220 = (float*)myMalloc(1 * sizeof(float));;
x5220[0] = 1.0f;
float* x5222 = (float*)myMalloc(1 * sizeof(float));;
x5222[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5220,x1105,1,x5222, x1377, 1, x1105,1));
arrayFill_greg<<<1, 512>>>(x1377, 0.0f, 256);
float* x5226 = (float*)myMalloc(1 * sizeof(float));;
x5226[0] = 1.0f;
float* x5228 = (float*)myMalloc(1 * sizeof(float));;
x5228[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5226,x1108,1,x5228, x1378, 1, x1108,1));
arrayFill_greg<<<1, 512>>>(x1378, 0.0f, 256);
float* x5232 = (float*)myMalloc(1 * sizeof(float));;
x5232[0] = 1.0f;
float* x5234 = (float*)myMalloc(1 * sizeof(float));;
x5234[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5232,x1111,1,x5234, x1379, 1, x1111,1));
arrayFill_greg<<<1, 512>>>(x1379, 0.0f, 1024);
int32_t x5238 = x1396 + 1;
int32_t x5240 = x5238 % x5239;
bool x5241 = x5240 == 0;
if (x5241) {
float x5246 = x1390;
double x5242 = (double)x1397;
double x5243 = 100.0 * x5242;
double x5245 = x5243 / x5244;
float x5247 = (float)x1396;
float x5248 = x5246 / x5247;
printf("Train epoch %d: [%d/%d (%.0f%%)] Average Loss: %.6f\n",x1386,x1397,x12,x5245,x5248);
fflush(stdout);
} else {
}
int64_t x5253 = (long)mallocAddr;
int64_t x5254 = x5253 - x1382;
memset((void*)x1382, 0, x5254);
mallocAddr = (void*)x1382;
int64_t x5257 = (long)gpuMallocAddr;
int64_t x5258 = x5257 - x1383;
cudaMemset((void*)x1383, 0, x5258);
gpuMallocAddr = (void*)x1383;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x5265 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
double x5266 = (double)x5265;
double x5267 = x5266 / 1000000.0;
x1381[x1386] = x5267;
int64_t x5269 = x5265 / 1000LL;
int64_t x5271 = x5265 / x5270;
printf("Training completed in %ldms (%ld us/images)\n",x5269,x5271);
float x5273 = x1390;
float x5275 = x5273 / x5274;
double x5276 = (double)x5275;
x1380[x1386] = x5276;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x5282 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
sort(x1381, x1381 + 4);
double x5288 = x1381[2];
int64_t x5289 = (long)fopen(x0, "w");
fprintf((FILE *)x5289, "unit: %s\n", "1 epoch");
for(int x5291=0; x5291 < 4; x5291++) {
double x5292 = x1380[x5291];
fprintf((FILE *)x5289, "%lf\n", x5292);

}
fprintf((FILE *)x5289, "run time: %lf %lf\n", x40, x5288);
fclose((FILE*)x5289);
// Backend cleanup.
CUBLAS_CALL(cublasDestroy(cublasHandle));
CUDA_CALL(cudaFree(gpuMallocBase));
      
CUDNN_CALL(cudnnDestroy(cudnnHandle));
}
/*****************************************
  End of C Generated Code                  
*******************************************/


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
printf("Data reading in %lf sec\n",x40);
// Tensor 'toGPU' invocation.
float* x98 = (float*)myGpuMalloc(32768 * sizeof(float));
int32_t x42 = open("/u/data/u99/wang603/TiarkMlEnv/Lantern/src/out/PLDI19evaluation/squeezenet/squeezenetCifar10.onnx.bin",0);
int32_t x43 = fsize(x42);
float* x44 = (float*)mmap(0, x43, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x42, 0);
float* x45 = x44+526720;
CUDA_CALL(cudaMemcpy(x98, x45, 32768 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x101 = (float*)myGpuMalloc(48 * sizeof(float));
float* x46 = x44+245136;
CUDA_CALL(cudaMemcpy(x101, x46, 48 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x104 = (float*)myGpuMalloc(64 * sizeof(float));
float* x47 = x44+17696;
CUDA_CALL(cudaMemcpy(x104, x47, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x107 = (float*)myGpuMalloc(81920 * sizeof(float));
float* x48 = x44+723904;
CUDA_CALL(cudaMemcpy(x107, x48, 81920 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x110 = (float*)myGpuMalloc(64 * sizeof(float));
float* x49 = x44+14544;
CUDA_CALL(cudaMemcpy(x110, x49, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x113 = (float*)myGpuMalloc(36864 * sizeof(float));
float* x50 = x44+35392;
CUDA_CALL(cudaMemcpy(x113, x50, 36864 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x116 = (float*)myGpuMalloc(4096 * sizeof(float));
float* x51 = x44+80608;
CUDA_CALL(cudaMemcpy(x116, x51, 4096 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x119 = (float*)myGpuMalloc(16 * sizeof(float));
float* x52 = x44+4224;
CUDA_CALL(cudaMemcpy(x119, x52, 16 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x122 = (float*)myGpuMalloc(64 * sizeof(float));
float* x53 = x44+362304;
CUDA_CALL(cudaMemcpy(x122, x53, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x125 = (float*)myGpuMalloc(4096 * sizeof(float));
float* x54 = x44+27040;
CUDA_CALL(cudaMemcpy(x125, x54, 4096 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x128 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x55 = x44+16672;
CUDA_CALL(cudaMemcpy(x128, x55, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x131 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x56 = x44+14608;
CUDA_CALL(cudaMemcpy(x131, x56, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x134 = (float*)myGpuMalloc(256 * sizeof(float));
float* x57 = x44+526464;
CUDA_CALL(cudaMemcpy(x134, x57, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x137 = (float*)myGpuMalloc(18432 * sizeof(float));
float* x58 = x44+226704;
CUDA_CALL(cudaMemcpy(x137, x58, 18432 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x140 = (float*)myGpuMalloc(32 * sizeof(float));
float* x59 = x44+80576;
CUDA_CALL(cudaMemcpy(x140, x59, 32 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x143 = (float*)myGpuMalloc(128 * sizeof(float));
float* x60 = x44+121696;
CUDA_CALL(cudaMemcpy(x143, x60, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x146 = (float*)myGpuMalloc(256 * sizeof(float));
float* x61 = x44+723648;
CUDA_CALL(cudaMemcpy(x146, x61, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x149 = (float*)myGpuMalloc(82944 * sizeof(float));
float* x62 = x44+254592;
CUDA_CALL(cudaMemcpy(x149, x62, 82944 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x152 = (float*)myGpuMalloc(9216 * sizeof(float));
float* x63 = x44+17760;
CUDA_CALL(cudaMemcpy(x152, x63, 9216 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x155 = (float*)myGpuMalloc(64 * sizeof(float));
float* x64 = x44+559488;
CUDA_CALL(cudaMemcpy(x155, x64, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x158 = (float*)myGpuMalloc(128 * sizeof(float));
float* x65 = x44+84704;
CUDA_CALL(cudaMemcpy(x158, x65, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x161 = (float*)myGpuMalloc(9216 * sizeof(float));
float* x66 = x44+245184;
CUDA_CALL(cudaMemcpy(x161, x66, 9216 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x164 = (float*)myGpuMalloc(32 * sizeof(float));
float* x67 = x44+31136;
CUDA_CALL(cudaMemcpy(x164, x67, 32 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x167 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x68 = x44+4240;
CUDA_CALL(cudaMemcpy(x167, x68, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x170 = (float*)myGpuMalloc(16 * sizeof(float));
float* x69 = x44+16656;
CUDA_CALL(cudaMemcpy(x170, x69, 16 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x173 = (float*)myGpuMalloc(256 * sizeof(float));
float* x70 = x44+575936;
CUDA_CALL(cudaMemcpy(x173, x70, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x176 = (float*)myGpuMalloc(8192 * sizeof(float));
float* x71 = x44+72384;
CUDA_CALL(cudaMemcpy(x176, x71, 8192 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x179 = (float*)myGpuMalloc(147456 * sizeof(float));
float* x72 = x44+379008;
CUDA_CALL(cudaMemcpy(x179, x72, 147456 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x182 = (float*)myGpuMalloc(192 * sizeof(float));
float* x73 = x44+226512;
CUDA_CALL(cudaMemcpy(x182, x73, 192 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x185 = (float*)myGpuMalloc(147456 * sizeof(float));
float* x74 = x44+576192;
CUDA_CALL(cudaMemcpy(x185, x74, 147456 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x188 = (float*)myGpuMalloc(64 * sizeof(float));
float* x75 = x44+5264;
CUDA_CALL(cudaMemcpy(x188, x75, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x191 = (float*)myGpuMalloc(192 * sizeof(float));
float* x76 = x44+254400;
CUDA_CALL(cudaMemcpy(x191, x76, 192 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x194 = (float*)myGpuMalloc(2592 * sizeof(float));
float* x77 = x44+0;
CUDA_CALL(cudaMemcpy(x194, x77, 2592 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x197 = (float*)myGpuMalloc(24576 * sizeof(float));
float* x78 = x44+337728;
CUDA_CALL(cudaMemcpy(x197, x78, 24576 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x200 = (float*)myGpuMalloc(4096 * sizeof(float));
float* x79 = x44+31168;
CUDA_CALL(cudaMemcpy(x200, x79, 4096 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x203 = (float*)myGpuMalloc(36864 * sizeof(float));
float* x80 = x44+84832;
CUDA_CALL(cudaMemcpy(x203, x80, 36864 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x206 = (float*)myGpuMalloc(64 * sizeof(float));
float* x81 = x44+26976;
CUDA_CALL(cudaMemcpy(x206, x81, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x209 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x82 = x44+559552;
CUDA_CALL(cudaMemcpy(x209, x82, 16384 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x212 = (float*)myGpuMalloc(82944 * sizeof(float));
float* x83 = x44+143568;
CUDA_CALL(cudaMemcpy(x212, x83, 82944 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x215 = (float*)myGpuMalloc(256 * sizeof(float));
float* x84 = x44+378752;
CUDA_CALL(cudaMemcpy(x215, x84, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x218 = (float*)myGpuMalloc(128 * sizeof(float));
float* x85 = x44+72256;
CUDA_CALL(cudaMemcpy(x218, x85, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x221 = (float*)myGpuMalloc(12288 * sizeof(float));
float* x86 = x44+121824;
CUDA_CALL(cudaMemcpy(x221, x86, 12288 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x224 = (float*)myGpuMalloc(96 * sizeof(float));
float* x87 = x44+2592;
CUDA_CALL(cudaMemcpy(x224, x87, 96 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x227 = (float*)myGpuMalloc(192 * sizeof(float));
float* x88 = x44+337536;
CUDA_CALL(cudaMemcpy(x227, x88, 192 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x230 = (float*)myGpuMalloc(128 * sizeof(float));
float* x89 = x44+35264;
CUDA_CALL(cudaMemcpy(x230, x89, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x233 = (float*)myGpuMalloc(192 * sizeof(float));
float* x90 = x44+143376;
CUDA_CALL(cudaMemcpy(x233, x90, 192 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x236 = (float*)myGpuMalloc(9216 * sizeof(float));
float* x91 = x44+5328;
CUDA_CALL(cudaMemcpy(x236, x91, 9216 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x239 = (float*)myGpuMalloc(9216 * sizeof(float));
float* x92 = x44+134160;
CUDA_CALL(cudaMemcpy(x239, x92, 9216 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x242 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x93 = x44+362368;
CUDA_CALL(cudaMemcpy(x242, x93, 16384 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x245 = (float*)myGpuMalloc(1536 * sizeof(float));
float* x94 = x44+2688;
CUDA_CALL(cudaMemcpy(x245, x94, 1536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x248 = (float*)myGpuMalloc(10 * sizeof(float));
float* x95 = x44+805824;
CUDA_CALL(cudaMemcpy(x248, x95, 10 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x251 = (float*)myGpuMalloc(48 * sizeof(float));
float* x96 = x44+134112;
CUDA_CALL(cudaMemcpy(x251, x96, 48 * sizeof(float), cudaMemcpyHostToDevice));
float* x253 = (float*)myGpuMalloc(32768 * sizeof(float));
float* x254 = (float*)myGpuMalloc(48 * sizeof(float));
float* x255 = (float*)myGpuMalloc(64 * sizeof(float));
float* x256 = (float*)myGpuMalloc(81920 * sizeof(float));
float* x257 = (float*)myGpuMalloc(64 * sizeof(float));
float* x258 = (float*)myGpuMalloc(36864 * sizeof(float));
float* x259 = (float*)myGpuMalloc(4096 * sizeof(float));
float* x260 = (float*)myGpuMalloc(16 * sizeof(float));
float* x261 = (float*)myGpuMalloc(64 * sizeof(float));
float* x262 = (float*)myGpuMalloc(4096 * sizeof(float));
float* x263 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x264 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x265 = (float*)myGpuMalloc(256 * sizeof(float));
float* x266 = (float*)myGpuMalloc(18432 * sizeof(float));
float* x267 = (float*)myGpuMalloc(32 * sizeof(float));
float* x268 = (float*)myGpuMalloc(128 * sizeof(float));
float* x269 = (float*)myGpuMalloc(256 * sizeof(float));
float* x270 = (float*)myGpuMalloc(82944 * sizeof(float));
float* x271 = (float*)myGpuMalloc(9216 * sizeof(float));
float* x272 = (float*)myGpuMalloc(64 * sizeof(float));
float* x273 = (float*)myGpuMalloc(128 * sizeof(float));
float* x274 = (float*)myGpuMalloc(9216 * sizeof(float));
float* x275 = (float*)myGpuMalloc(32 * sizeof(float));
float* x276 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x277 = (float*)myGpuMalloc(16 * sizeof(float));
float* x278 = (float*)myGpuMalloc(256 * sizeof(float));
float* x279 = (float*)myGpuMalloc(8192 * sizeof(float));
float* x280 = (float*)myGpuMalloc(147456 * sizeof(float));
float* x281 = (float*)myGpuMalloc(192 * sizeof(float));
float* x282 = (float*)myGpuMalloc(147456 * sizeof(float));
float* x283 = (float*)myGpuMalloc(64 * sizeof(float));
float* x284 = (float*)myGpuMalloc(192 * sizeof(float));
float* x285 = (float*)myGpuMalloc(2592 * sizeof(float));
float* x286 = (float*)myGpuMalloc(24576 * sizeof(float));
float* x287 = (float*)myGpuMalloc(4096 * sizeof(float));
float* x288 = (float*)myGpuMalloc(36864 * sizeof(float));
float* x289 = (float*)myGpuMalloc(64 * sizeof(float));
float* x290 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x291 = (float*)myGpuMalloc(82944 * sizeof(float));
float* x292 = (float*)myGpuMalloc(256 * sizeof(float));
float* x293 = (float*)myGpuMalloc(128 * sizeof(float));
float* x294 = (float*)myGpuMalloc(12288 * sizeof(float));
float* x295 = (float*)myGpuMalloc(96 * sizeof(float));
float* x296 = (float*)myGpuMalloc(192 * sizeof(float));
float* x297 = (float*)myGpuMalloc(128 * sizeof(float));
float* x298 = (float*)myGpuMalloc(192 * sizeof(float));
float* x299 = (float*)myGpuMalloc(9216 * sizeof(float));
float* x300 = (float*)myGpuMalloc(9216 * sizeof(float));
float* x301 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x302 = (float*)myGpuMalloc(1536 * sizeof(float));
float* x303 = (float*)myGpuMalloc(10 * sizeof(float));
float* x304 = (float*)myGpuMalloc(48 * sizeof(float));
double* x305 = (double*)myMalloc(4 * sizeof(double));;
double* x306 = (double*)myMalloc(4 * sizeof(double));;
int64_t x307 = (long)mallocAddr;
int64_t x308 = (long)gpuMallocAddr;
// training loop starts here
int32_t x319 = x12 / 64;
int32_t x1576 = x319 / 10;
double x1581 = (double)x12;
int64_t x1607 = (int64_t)x12;
float x1611 = (float)x12;
for(int x311=0; x311 < 4; x311++) {
struct timeval begin_1, end_1, diff_1;
float x313 = 0.0f;
float x314 = x313;
float x315 = x314;
int32_t x316 = x311 + 1;
printf("Start training epoch %d\n",x316);
gettimeofday(&begin_1, NULL);
for(int x321=0; x321 < x319; x321++) {
int32_t x322 = x321 * 64;
int32_t x323 = x322 * 3072;
float* x324 = x14+x323;
int* x325 = x15+x322;
// Tensor 'toGPU' invocation.
float* x327 = (float*)myGpuMalloc(196608 * sizeof(float));
CUDA_CALL(cudaMemcpy(x327, x324, 196608 * sizeof(float), cudaMemcpyHostToDevice));
float* x329 = (float*)myGpuMalloc(2 * sizeof(float));
int* x330 = (int32_t*)myGpuMalloc(64 * sizeof(int32_t));
CUDA_CALL(cudaMemcpy(x330, x325, 64 * sizeof(int32_t), cudaMemcpyHostToDevice));
float* x332 = (float*)myGpuMalloc(1 * sizeof(float));
float* x333 = (float*)myGpuMalloc(1 * sizeof(float));
// allocate memory to save the final loss in CPU Tensor
float* x335 = (float*)myMalloc(1 * sizeof(float));;
float* x336 = (float*)myGpuMalloc(6291456 * sizeof(float));
float* x337 = (float*)myMalloc(1 * sizeof(float));;
x337[0] = 0.0f;
float* x339 = (float*)myMalloc(1 * sizeof(float));;
x339[0] = 1.0f;

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
    96, 3, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 96, 32, 32));

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
    x339, in_desc, x327, filt_desc, x194,
    conv_desc, algo, ws_data, ws_size,
    x337, out_desc, x336));
};
float* x342 = (float*)myMalloc(1 * sizeof(float));;
x342[0] = 1.0f;
float* x344 = (float*)myMalloc(1 * sizeof(float));;
x344[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 96, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 96, 32, 32));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x342, bias_desc, x224, x344, out_desc, x336));
};
float* x347 = (float*)myGpuMalloc(6291456 * sizeof(float));
float* x348 = (float*)myMalloc(1 * sizeof(float));;
x348[0] = 0.0f;
float* x350 = (float*)myMalloc(1 * sizeof(float));;
x350[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 96, 32, 32));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x350, x_desc, x336, x348, x_desc, x336));
};
float* x353 = (float*)myMalloc(1 * sizeof(float));;
x353[0] = 0.0f;
float* x355 = (float*)myMalloc(1 * sizeof(float));;
x355[0] = 1.0f;
float* x357 = (float*)myGpuMalloc(1572864 * sizeof(float));

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 96, 32, 32));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 96, 16, 16));

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
    x355, in_desc, x336, x353, out_desc, x357));
};
float* x359 = (float*)myGpuMalloc(1572864 * sizeof(float));
float* x360 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x361 = (float*)myMalloc(1 * sizeof(float));;
x361[0] = 0.0f;
float* x363 = (float*)myMalloc(1 * sizeof(float));;
x363[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 96, 16, 16));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    16, 96, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

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
    x363, in_desc, x357, filt_desc, x245,
    conv_desc, algo, ws_data, ws_size,
    x361, out_desc, x360));
};
float* x366 = (float*)myMalloc(1 * sizeof(float));;
x366[0] = 1.0f;
float* x368 = (float*)myMalloc(1 * sizeof(float));;
x368[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 16, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x366, bias_desc, x119, x368, out_desc, x360));
};
float* x371 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x372 = (float*)myMalloc(1 * sizeof(float));;
x372[0] = 0.0f;
float* x374 = (float*)myMalloc(1 * sizeof(float));;
x374[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x374, x_desc, x360, x372, x_desc, x360));
};
float* x377 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x378 = (float*)myMalloc(1 * sizeof(float));;
x378[0] = 0.0f;
float* x380 = (float*)myMalloc(1 * sizeof(float));;
x380[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 16, 1, 1));

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
    x380, in_desc, x360, filt_desc, x167,
    conv_desc, algo, ws_data, ws_size,
    x378, out_desc, x377));
};
float* x383 = (float*)myMalloc(1 * sizeof(float));;
x383[0] = 1.0f;
float* x385 = (float*)myMalloc(1 * sizeof(float));;
x385[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x383, bias_desc, x188, x385, out_desc, x377));
};
float* x388 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x389 = (float*)myMalloc(1 * sizeof(float));;
x389[0] = 0.0f;
float* x391 = (float*)myMalloc(1 * sizeof(float));;
x391[0] = 1.0f;

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
    x391, x_desc, x377, x389, x_desc, x377));
};
float* x394 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x395 = (float*)myMalloc(1 * sizeof(float));;
x395[0] = 0.0f;
float* x397 = (float*)myMalloc(1 * sizeof(float));;
x397[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 16, 3, 3));

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
    x397, in_desc, x360, filt_desc, x236,
    conv_desc, algo, ws_data, ws_size,
    x395, out_desc, x394));
};
float* x400 = (float*)myMalloc(1 * sizeof(float));;
x400[0] = 1.0f;
float* x402 = (float*)myMalloc(1 * sizeof(float));;
x402[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x400, bias_desc, x110, x402, out_desc, x394));
};
float* x405 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x406 = (float*)myMalloc(1 * sizeof(float));;
x406[0] = 0.0f;
float* x408 = (float*)myMalloc(1 * sizeof(float));;
x408[0] = 1.0f;

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
    x408, x_desc, x394, x406, x_desc, x394));
};
float* x411 = (float*)myGpuMalloc(2097152 * sizeof(float));
{
dim3 grid(28, 2);
concat2D_1D_greg<<<grid, 512>>>(x377, 64, 1048576, x394, 64, 1048576, x411, 1, 64, 128, 16, 16, 32768, 256, 16, 1);
};
float* x413 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x414 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x415 = (float*)myMalloc(1 * sizeof(float));;
x415[0] = 0.0f;
float* x417 = (float*)myMalloc(1 * sizeof(float));;
x417[0] = 1.0f;

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
    16, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

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
    x417, in_desc, x411, filt_desc, x131,
    conv_desc, algo, ws_data, ws_size,
    x415, out_desc, x414));
};
float* x420 = (float*)myMalloc(1 * sizeof(float));;
x420[0] = 1.0f;
float* x422 = (float*)myMalloc(1 * sizeof(float));;
x422[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 16, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x420, bias_desc, x170, x422, out_desc, x414));
};
float* x425 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x426 = (float*)myMalloc(1 * sizeof(float));;
x426[0] = 0.0f;
float* x428 = (float*)myMalloc(1 * sizeof(float));;
x428[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x428, x_desc, x414, x426, x_desc, x414));
};
float* x431 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x432 = (float*)myMalloc(1 * sizeof(float));;
x432[0] = 0.0f;
float* x434 = (float*)myMalloc(1 * sizeof(float));;
x434[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 16, 1, 1));

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
    x434, in_desc, x414, filt_desc, x128,
    conv_desc, algo, ws_data, ws_size,
    x432, out_desc, x431));
};
float* x437 = (float*)myMalloc(1 * sizeof(float));;
x437[0] = 1.0f;
float* x439 = (float*)myMalloc(1 * sizeof(float));;
x439[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x437, bias_desc, x104, x439, out_desc, x431));
};
float* x442 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x443 = (float*)myMalloc(1 * sizeof(float));;
x443[0] = 0.0f;
float* x445 = (float*)myMalloc(1 * sizeof(float));;
x445[0] = 1.0f;

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
    x445, x_desc, x431, x443, x_desc, x431));
};
float* x448 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x449 = (float*)myMalloc(1 * sizeof(float));;
x449[0] = 0.0f;
float* x451 = (float*)myMalloc(1 * sizeof(float));;
x451[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 16, 3, 3));

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
    x451, in_desc, x414, filt_desc, x152,
    conv_desc, algo, ws_data, ws_size,
    x449, out_desc, x448));
};
float* x454 = (float*)myMalloc(1 * sizeof(float));;
x454[0] = 1.0f;
float* x456 = (float*)myMalloc(1 * sizeof(float));;
x456[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x454, bias_desc, x206, x456, out_desc, x448));
};
float* x459 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x460 = (float*)myMalloc(1 * sizeof(float));;
x460[0] = 0.0f;
float* x462 = (float*)myMalloc(1 * sizeof(float));;
x462[0] = 1.0f;

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
    x462, x_desc, x448, x460, x_desc, x448));
};
float* x465 = (float*)myGpuMalloc(2097152 * sizeof(float));
{
dim3 grid(28, 2);
concat2D_1D_greg<<<grid, 512>>>(x431, 64, 1048576, x448, 64, 1048576, x465, 1, 64, 128, 16, 16, 32768, 256, 16, 1);
};
float* x467 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x468 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x469 = (float*)myMalloc(1 * sizeof(float));;
x469[0] = 0.0f;
float* x471 = (float*)myMalloc(1 * sizeof(float));;
x471[0] = 1.0f;

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
    32, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 16, 16));

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
    x471, in_desc, x465, filt_desc, x125,
    conv_desc, algo, ws_data, ws_size,
    x469, out_desc, x468));
};
float* x474 = (float*)myMalloc(1 * sizeof(float));;
x474[0] = 1.0f;
float* x476 = (float*)myMalloc(1 * sizeof(float));;
x476[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 32, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 16, 16));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x474, bias_desc, x164, x476, out_desc, x468));
};
float* x479 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x480 = (float*)myMalloc(1 * sizeof(float));;
x480[0] = 0.0f;
float* x482 = (float*)myMalloc(1 * sizeof(float));;
x482[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x482, x_desc, x468, x480, x_desc, x468));
};
float* x485 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x486 = (float*)myMalloc(1 * sizeof(float));;
x486[0] = 0.0f;
float* x488 = (float*)myMalloc(1 * sizeof(float));;
x488[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 16, 16));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 32, 1, 1));

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
    x488, in_desc, x468, filt_desc, x200,
    conv_desc, algo, ws_data, ws_size,
    x486, out_desc, x485));
};
float* x491 = (float*)myMalloc(1 * sizeof(float));;
x491[0] = 1.0f;
float* x493 = (float*)myMalloc(1 * sizeof(float));;
x493[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x491, bias_desc, x230, x493, out_desc, x485));
};
float* x496 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x497 = (float*)myMalloc(1 * sizeof(float));;
x497[0] = 0.0f;
float* x499 = (float*)myMalloc(1 * sizeof(float));;
x499[0] = 1.0f;

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
    x499, x_desc, x485, x497, x_desc, x485));
};
float* x502 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x503 = (float*)myMalloc(1 * sizeof(float));;
x503[0] = 0.0f;
float* x505 = (float*)myMalloc(1 * sizeof(float));;
x505[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 16, 16));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 32, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

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
    x505, in_desc, x468, filt_desc, x113,
    conv_desc, algo, ws_data, ws_size,
    x503, out_desc, x502));
};
float* x508 = (float*)myMalloc(1 * sizeof(float));;
x508[0] = 1.0f;
float* x510 = (float*)myMalloc(1 * sizeof(float));;
x510[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x508, bias_desc, x218, x510, out_desc, x502));
};
float* x513 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x514 = (float*)myMalloc(1 * sizeof(float));;
x514[0] = 0.0f;
float* x516 = (float*)myMalloc(1 * sizeof(float));;
x516[0] = 1.0f;

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
    x516, x_desc, x502, x514, x_desc, x502));
};
float* x519 = (float*)myGpuMalloc(4194304 * sizeof(float));
{
dim3 grid(28, 2);
concat2D_1D_greg<<<grid, 512>>>(x485, 128, 2097152, x502, 128, 2097152, x519, 1, 64, 256, 16, 16, 65536, 256, 16, 1);
};
float* x521 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x522 = (float*)myMalloc(1 * sizeof(float));;
x522[0] = 0.0f;
float* x524 = (float*)myMalloc(1 * sizeof(float));;
x524[0] = 1.0f;
float* x526 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    64, 256, 8, 8));

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
    x524, in_desc, x519, x522, out_desc, x526));
};
float* x528 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x529 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x530 = (float*)myMalloc(1 * sizeof(float));;
x530[0] = 0.0f;
float* x532 = (float*)myMalloc(1 * sizeof(float));;
x532[0] = 1.0f;

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
    32, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 8, 8));

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
    x532, in_desc, x526, filt_desc, x176,
    conv_desc, algo, ws_data, ws_size,
    x530, out_desc, x529));
};
float* x535 = (float*)myMalloc(1 * sizeof(float));;
x535[0] = 1.0f;
float* x537 = (float*)myMalloc(1 * sizeof(float));;
x537[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 32, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x535, bias_desc, x140, x537, out_desc, x529));
};
float* x540 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x541 = (float*)myMalloc(1 * sizeof(float));;
x541[0] = 0.0f;
float* x543 = (float*)myMalloc(1 * sizeof(float));;
x543[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x543, x_desc, x529, x541, x_desc, x529));
};
float* x546 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x547 = (float*)myMalloc(1 * sizeof(float));;
x547[0] = 0.0f;
float* x549 = (float*)myMalloc(1 * sizeof(float));;
x549[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 32, 1, 1));

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
    x549, in_desc, x529, filt_desc, x116,
    conv_desc, algo, ws_data, ws_size,
    x547, out_desc, x546));
};
float* x552 = (float*)myMalloc(1 * sizeof(float));;
x552[0] = 1.0f;
float* x554 = (float*)myMalloc(1 * sizeof(float));;
x554[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x552, bias_desc, x158, x554, out_desc, x546));
};
float* x557 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x558 = (float*)myMalloc(1 * sizeof(float));;
x558[0] = 0.0f;
float* x560 = (float*)myMalloc(1 * sizeof(float));;
x560[0] = 1.0f;

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
    x560, x_desc, x546, x558, x_desc, x546));
};
float* x563 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x564 = (float*)myMalloc(1 * sizeof(float));;
x564[0] = 0.0f;
float* x566 = (float*)myMalloc(1 * sizeof(float));;
x566[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 32, 3, 3));

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
    x566, in_desc, x529, filt_desc, x203,
    conv_desc, algo, ws_data, ws_size,
    x564, out_desc, x563));
};
float* x569 = (float*)myMalloc(1 * sizeof(float));;
x569[0] = 1.0f;
float* x571 = (float*)myMalloc(1 * sizeof(float));;
x571[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x569, bias_desc, x143, x571, out_desc, x563));
};
float* x574 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x575 = (float*)myMalloc(1 * sizeof(float));;
x575[0] = 0.0f;
float* x577 = (float*)myMalloc(1 * sizeof(float));;
x577[0] = 1.0f;

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
    x577, x_desc, x563, x575, x_desc, x563));
};
float* x580 = (float*)myGpuMalloc(1048576 * sizeof(float));
{
dim3 grid(28, 2);
concat2D_1D_greg<<<grid, 512>>>(x546, 128, 524288, x563, 128, 524288, x580, 1, 64, 256, 8, 8, 16384, 64, 8, 1);
};
float* x582 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x583 = (float*)myGpuMalloc(196608 * sizeof(float));
float* x584 = (float*)myMalloc(1 * sizeof(float));;
x584[0] = 0.0f;
float* x586 = (float*)myMalloc(1 * sizeof(float));;
x586[0] = 1.0f;

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
    48, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

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
    x586, in_desc, x580, filt_desc, x221,
    conv_desc, algo, ws_data, ws_size,
    x584, out_desc, x583));
};
float* x589 = (float*)myMalloc(1 * sizeof(float));;
x589[0] = 1.0f;
float* x591 = (float*)myMalloc(1 * sizeof(float));;
x591[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 48, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x589, bias_desc, x251, x591, out_desc, x583));
};
float* x594 = (float*)myGpuMalloc(196608 * sizeof(float));
float* x595 = (float*)myMalloc(1 * sizeof(float));;
x595[0] = 0.0f;
float* x597 = (float*)myMalloc(1 * sizeof(float));;
x597[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x597, x_desc, x583, x595, x_desc, x583));
};
float* x600 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x601 = (float*)myMalloc(1 * sizeof(float));;
x601[0] = 0.0f;
float* x603 = (float*)myMalloc(1 * sizeof(float));;
x603[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    192, 48, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

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
    x603, in_desc, x583, filt_desc, x239,
    conv_desc, algo, ws_data, ws_size,
    x601, out_desc, x600));
};
float* x606 = (float*)myMalloc(1 * sizeof(float));;
x606[0] = 1.0f;
float* x608 = (float*)myMalloc(1 * sizeof(float));;
x608[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 192, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x606, bias_desc, x233, x608, out_desc, x600));
};
float* x611 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x612 = (float*)myMalloc(1 * sizeof(float));;
x612[0] = 0.0f;
float* x614 = (float*)myMalloc(1 * sizeof(float));;
x614[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x614, x_desc, x600, x612, x_desc, x600));
};
float* x617 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x618 = (float*)myMalloc(1 * sizeof(float));;
x618[0] = 0.0f;
float* x620 = (float*)myMalloc(1 * sizeof(float));;
x620[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    192, 48, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

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
    x620, in_desc, x583, filt_desc, x212,
    conv_desc, algo, ws_data, ws_size,
    x618, out_desc, x617));
};
float* x623 = (float*)myMalloc(1 * sizeof(float));;
x623[0] = 1.0f;
float* x625 = (float*)myMalloc(1 * sizeof(float));;
x625[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 192, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x623, bias_desc, x182, x625, out_desc, x617));
};
float* x628 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x629 = (float*)myMalloc(1 * sizeof(float));;
x629[0] = 0.0f;
float* x631 = (float*)myMalloc(1 * sizeof(float));;
x631[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x631, x_desc, x617, x629, x_desc, x617));
};
float* x634 = (float*)myGpuMalloc(1572864 * sizeof(float));
{
dim3 grid(28, 2);
concat2D_1D_greg<<<grid, 512>>>(x600, 192, 786432, x617, 192, 786432, x634, 1, 64, 384, 8, 8, 24576, 64, 8, 1);
};
float* x636 = (float*)myGpuMalloc(1572864 * sizeof(float));
float* x637 = (float*)myGpuMalloc(196608 * sizeof(float));
float* x638 = (float*)myMalloc(1 * sizeof(float));;
x638[0] = 0.0f;
float* x640 = (float*)myMalloc(1 * sizeof(float));;
x640[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 384, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    48, 384, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

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
    x640, in_desc, x634, filt_desc, x137,
    conv_desc, algo, ws_data, ws_size,
    x638, out_desc, x637));
};
float* x643 = (float*)myMalloc(1 * sizeof(float));;
x643[0] = 1.0f;
float* x645 = (float*)myMalloc(1 * sizeof(float));;
x645[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 48, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x643, bias_desc, x101, x645, out_desc, x637));
};
float* x648 = (float*)myGpuMalloc(196608 * sizeof(float));
float* x649 = (float*)myMalloc(1 * sizeof(float));;
x649[0] = 0.0f;
float* x651 = (float*)myMalloc(1 * sizeof(float));;
x651[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x651, x_desc, x637, x649, x_desc, x637));
};
float* x654 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x655 = (float*)myMalloc(1 * sizeof(float));;
x655[0] = 0.0f;
float* x657 = (float*)myMalloc(1 * sizeof(float));;
x657[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    192, 48, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

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
    x657, in_desc, x637, filt_desc, x161,
    conv_desc, algo, ws_data, ws_size,
    x655, out_desc, x654));
};
float* x660 = (float*)myMalloc(1 * sizeof(float));;
x660[0] = 1.0f;
float* x662 = (float*)myMalloc(1 * sizeof(float));;
x662[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 192, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x660, bias_desc, x191, x662, out_desc, x654));
};
float* x665 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x666 = (float*)myMalloc(1 * sizeof(float));;
x666[0] = 0.0f;
float* x668 = (float*)myMalloc(1 * sizeof(float));;
x668[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x668, x_desc, x654, x666, x_desc, x654));
};
float* x671 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x672 = (float*)myMalloc(1 * sizeof(float));;
x672[0] = 0.0f;
float* x674 = (float*)myMalloc(1 * sizeof(float));;
x674[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    192, 48, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

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
    x674, in_desc, x637, filt_desc, x149,
    conv_desc, algo, ws_data, ws_size,
    x672, out_desc, x671));
};
float* x677 = (float*)myMalloc(1 * sizeof(float));;
x677[0] = 1.0f;
float* x679 = (float*)myMalloc(1 * sizeof(float));;
x679[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 192, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x677, bias_desc, x227, x679, out_desc, x671));
};
float* x682 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x683 = (float*)myMalloc(1 * sizeof(float));;
x683[0] = 0.0f;
float* x685 = (float*)myMalloc(1 * sizeof(float));;
x685[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x685, x_desc, x671, x683, x_desc, x671));
};
float* x688 = (float*)myGpuMalloc(1572864 * sizeof(float));
{
dim3 grid(28, 2);
concat2D_1D_greg<<<grid, 512>>>(x654, 192, 786432, x671, 192, 786432, x688, 1, 64, 384, 8, 8, 24576, 64, 8, 1);
};
float* x690 = (float*)myGpuMalloc(1572864 * sizeof(float));
float* x691 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x692 = (float*)myMalloc(1 * sizeof(float));;
x692[0] = 0.0f;
float* x694 = (float*)myMalloc(1 * sizeof(float));;
x694[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 384, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 384, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 8, 8));

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
    x694, in_desc, x688, filt_desc, x197,
    conv_desc, algo, ws_data, ws_size,
    x692, out_desc, x691));
};
float* x697 = (float*)myMalloc(1 * sizeof(float));;
x697[0] = 1.0f;
float* x699 = (float*)myMalloc(1 * sizeof(float));;
x699[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x697, bias_desc, x122, x699, out_desc, x691));
};
float* x702 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x703 = (float*)myMalloc(1 * sizeof(float));;
x703[0] = 0.0f;
float* x705 = (float*)myMalloc(1 * sizeof(float));;
x705[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x705, x_desc, x691, x703, x_desc, x691));
};
float* x708 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x709 = (float*)myMalloc(1 * sizeof(float));;
x709[0] = 0.0f;
float* x711 = (float*)myMalloc(1 * sizeof(float));;
x711[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

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
    x711, in_desc, x691, filt_desc, x242,
    conv_desc, algo, ws_data, ws_size,
    x709, out_desc, x708));
};
float* x714 = (float*)myMalloc(1 * sizeof(float));;
x714[0] = 1.0f;
float* x716 = (float*)myMalloc(1 * sizeof(float));;
x716[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x714, bias_desc, x215, x716, out_desc, x708));
};
float* x719 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x720 = (float*)myMalloc(1 * sizeof(float));;
x720[0] = 0.0f;
float* x722 = (float*)myMalloc(1 * sizeof(float));;
x722[0] = 1.0f;

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
    x722, x_desc, x708, x720, x_desc, x708));
};
float* x725 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x726 = (float*)myMalloc(1 * sizeof(float));;
x726[0] = 0.0f;
float* x728 = (float*)myMalloc(1 * sizeof(float));;
x728[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 8, 8));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 3, 3));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

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
    x728, in_desc, x691, filt_desc, x179,
    conv_desc, algo, ws_data, ws_size,
    x726, out_desc, x725));
};
float* x731 = (float*)myMalloc(1 * sizeof(float));;
x731[0] = 1.0f;
float* x733 = (float*)myMalloc(1 * sizeof(float));;
x733[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x731, bias_desc, x134, x733, out_desc, x725));
};
float* x736 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x737 = (float*)myMalloc(1 * sizeof(float));;
x737[0] = 0.0f;
float* x739 = (float*)myMalloc(1 * sizeof(float));;
x739[0] = 1.0f;

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
    x739, x_desc, x725, x737, x_desc, x725));
};
float* x742 = (float*)myGpuMalloc(2097152 * sizeof(float));
{
dim3 grid(28, 2);
concat2D_1D_greg<<<grid, 512>>>(x708, 256, 1048576, x725, 256, 1048576, x742, 1, 64, 512, 8, 8, 32768, 64, 8, 1);
};
float* x744 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x745 = (float*)myMalloc(1 * sizeof(float));;
x745[0] = 0.0f;
float* x747 = (float*)myMalloc(1 * sizeof(float));;
x747[0] = 1.0f;
float* x749 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    64, 512, 4, 4));

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
    x747, in_desc, x742, x745, out_desc, x749));
};
float* x751 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x752 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x753 = (float*)myMalloc(1 * sizeof(float));;
x753[0] = 0.0f;
float* x755 = (float*)myMalloc(1 * sizeof(float));;
x755[0] = 1.0f;

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
    64, 512, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 4, 4));

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
    x755, in_desc, x749, filt_desc, x98,
    conv_desc, algo, ws_data, ws_size,
    x753, out_desc, x752));
};
float* x758 = (float*)myMalloc(1 * sizeof(float));;
x758[0] = 1.0f;
float* x760 = (float*)myMalloc(1 * sizeof(float));;
x760[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 4, 4));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x758, bias_desc, x155, x760, out_desc, x752));
};
float* x763 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x764 = (float*)myMalloc(1 * sizeof(float));;
x764[0] = 0.0f;
float* x766 = (float*)myMalloc(1 * sizeof(float));;
x766[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x766, x_desc, x752, x764, x_desc, x752));
};
float* x769 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x770 = (float*)myMalloc(1 * sizeof(float));;
x770[0] = 0.0f;
float* x772 = (float*)myMalloc(1 * sizeof(float));;
x772[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 1, 1));

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
    x772, in_desc, x752, filt_desc, x209,
    conv_desc, algo, ws_data, ws_size,
    x770, out_desc, x769));
};
float* x775 = (float*)myMalloc(1 * sizeof(float));;
x775[0] = 1.0f;
float* x777 = (float*)myMalloc(1 * sizeof(float));;
x777[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x775, bias_desc, x173, x777, out_desc, x769));
};
float* x780 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x781 = (float*)myMalloc(1 * sizeof(float));;
x781[0] = 0.0f;
float* x783 = (float*)myMalloc(1 * sizeof(float));;
x783[0] = 1.0f;

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
    x783, x_desc, x769, x781, x_desc, x769));
};
float* x786 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x787 = (float*)myMalloc(1 * sizeof(float));;
x787[0] = 0.0f;
float* x789 = (float*)myMalloc(1 * sizeof(float));;
x789[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 4, 4));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 3, 3));

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
    x789, in_desc, x752, filt_desc, x185,
    conv_desc, algo, ws_data, ws_size,
    x787, out_desc, x786));
};
float* x792 = (float*)myMalloc(1 * sizeof(float));;
x792[0] = 1.0f;
float* x794 = (float*)myMalloc(1 * sizeof(float));;
x794[0] = 1.0f;

{
cudnnTensorDescriptor_t bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x792, bias_desc, x146, x794, out_desc, x786));
};
float* x797 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x798 = (float*)myMalloc(1 * sizeof(float));;
x798[0] = 0.0f;
float* x800 = (float*)myMalloc(1 * sizeof(float));;
x800[0] = 1.0f;

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
    x800, x_desc, x786, x798, x_desc, x786));
};
float* x803 = (float*)myGpuMalloc(524288 * sizeof(float));
{
dim3 grid(28, 2);
concat2D_1D_greg<<<grid, 512>>>(x769, 256, 262144, x786, 256, 262144, x803, 1, 64, 512, 4, 4, 8192, 16, 4, 1);
};
float* x805 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x806 = (float*)myGpuMalloc(640 * sizeof(float));
float* x807 = (float*)myMalloc(1 * sizeof(float));;
x807[0] = 0.0f;
float* x809 = (float*)myMalloc(1 * sizeof(float));;
x809[0] = 1.0f;

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
    10, 512, 4, 4));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 10, 1, 1));

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
    x809, in_desc, x803, filt_desc, x107,
    conv_desc, algo, ws_data, ws_size,
    x807, out_desc, x806));
};
float* x812 = (float*)myMalloc(1 * sizeof(float));;
x812[0] = 1.0f;
float* x814 = (float*)myMalloc(1 * sizeof(float));;
x814[0] = 1.0f;

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
    cudnnHandle, x812, bias_desc, x248, x814, out_desc, x806));
};
float* x817 = (float*)myGpuMalloc(640 * sizeof(float));
// resize to WrappedArray(64, 10)
// resize to WrappedArray(64, 10)
float* x820 = (float*)myMalloc(1 * sizeof(float));;
x820[0] = 0.0f;
float* x822 = (float*)myMalloc(1 * sizeof(float));;
x822[0] = 1.0f;
float* x824 = (float*)myGpuMalloc(640 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 10, 1, 1));
CUDNN_CALL(cudnnSoftmaxForward(
    cudnnHandle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL,
    x822, x_desc, x806, x820, x_desc, x824));
};
float* x826 = (float*)myGpuMalloc(640 * sizeof(float));
float* x827 = (float*)myGpuMalloc(64 * sizeof(float));
nllLoss<<<64, 1>>>(x824,10,x827,x330);
float* x829 = (float*)myGpuMalloc(64 * sizeof(float));
// resize to ArrayBuffer(64, 1, 1, 1)
float* x831 = (float*)myGpuMalloc(1 * sizeof(float));
float* x832 = (float*)myMalloc(1 * sizeof(float));;
x832[0] = 0.0f;
float* x834 = (float*)myMalloc(1 * sizeof(float));;
x834[0] = 1.0f;

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
    x834, x_desc, x827, x832, out_desc, x831));
};
// resize to WrappedArray(1)
float* x838 = (float*)myGpuMalloc(1 * sizeof(float));
arrayFill_greg<<<1, 512>>>(x838, 1.0f, 1);
// backend is lantern.TensorDslCudnn$BackendCudnn@45b1bfcc
CUDA_CALL(cudaMemcpy(x335, x831, 1 * sizeof(float), cudaMemcpyDeviceToHost));
// 'mean' gradient
// backprop for mean op
// resize to WrappedArray(1, 1, 1, 1)
// resize to ArrayBuffer(64, 1, 1, 1)
float* x846 = (float*)myMalloc(1 * sizeof(float));;
x846[0] = 0.015625f;
float* x848 = (float*)myMalloc(1 * sizeof(float));;
x848[0] = 1.0f;

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
    cudnnHandle, x846, bias_desc, x838, x848, out_desc, x829));
};
// 'nllLossB' gradient.
nllLoss_grad<<<64, 1>>>(10,x829,x330,x826);
float* x853 = (float*)myMalloc(1 * sizeof(float));;
x853[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 10, 1, 1));
CUDNN_CALL(cudnnSoftmaxBackward(
    cudnnHandle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL,
    x853, x_desc, x824, x_desc, x826,
    x853, x_desc, x817));
};
// conv2D back-propagate
float* x857 = (float*)myMalloc(1 * sizeof(float));;
x857[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    10, 512, 4, 4));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 10, 1, 1));

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
    x857, filt_desc, x107, grad_out_desc, x817,
    conv_desc, algo, ws_data, ws_size,
    x857, grad_in_desc, x805));
};
float* x860 = (float*)myMalloc(1 * sizeof(float));;
x860[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    10, 512, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 10, 1, 1));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 4, 4));

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
    x860, in_desc, x803, grad_out_desc, x817,
    conv_desc, algo, ws_data, ws_size,
    x860, grad_filt_desc, x256));
};
float* x863 = (float*)myMalloc(1 * sizeof(float));;
x863[0] = 1.0f;

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
    cudnnHandle, x863, grad_out_desc, x817,
    x863, grad_bias_desc, x303));
};
{
dim3 grid(28, 2);
concat2D_1D_greg_grad<<<grid, 512>>>(x780, 256, 262144, x797, 256, 262144, x805, 1, 64, 512, 4, 4, 8192, 16, 4, 1);
};
float* x867 = (float*)myMalloc(1 * sizeof(float));;
x867[0] = 1.0f;
float* x869 = (float*)myMalloc(1 * sizeof(float));;
x869[0] = 0.0f;

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
    x867, x_desc, x786, x_desc, x797, x_desc, x786,
    x869, x_desc, x797));
};
// conv2D back-propagate
float* x873 = (float*)myMalloc(1 * sizeof(float));;
x873[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 4, 4));

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
    x873, filt_desc, x185, grad_out_desc, x797,
    conv_desc, algo, ws_data, ws_size,
    x873, grad_in_desc, x763));
};
float* x876 = (float*)myMalloc(1 * sizeof(float));;
x876[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 4, 4));

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
    x876, in_desc, x752, grad_out_desc, x797,
    conv_desc, algo, ws_data, ws_size,
    x876, grad_filt_desc, x282));
};
float* x879 = (float*)myMalloc(1 * sizeof(float));;
x879[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x879, grad_out_desc, x797,
    x879, grad_bias_desc, x269));
};
float* x882 = (float*)myMalloc(1 * sizeof(float));;
x882[0] = 1.0f;
float* x884 = (float*)myMalloc(1 * sizeof(float));;
x884[0] = 0.0f;

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
    x882, x_desc, x769, x_desc, x780, x_desc, x769,
    x884, x_desc, x780));
};
// conv2D back-propagate
float* x888 = (float*)myMalloc(1 * sizeof(float));;
x888[0] = 1.0f;

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
    64, 64, 4, 4));

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
    x888, filt_desc, x209, grad_out_desc, x780,
    conv_desc, algo, ws_data, ws_size,
    x888, grad_in_desc, x763));
};
float* x891 = (float*)myMalloc(1 * sizeof(float));;
x891[0] = 1.0f;

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
    64, 256, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 4, 4));

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
    x891, in_desc, x752, grad_out_desc, x780,
    conv_desc, algo, ws_data, ws_size,
    x891, grad_filt_desc, x290));
};
float* x894 = (float*)myMalloc(1 * sizeof(float));;
x894[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 4, 4));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x894, grad_out_desc, x780,
    x894, grad_bias_desc, x278));
};
float* x897 = (float*)myMalloc(1 * sizeof(float));;
x897[0] = 1.0f;
float* x899 = (float*)myMalloc(1 * sizeof(float));;
x899[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 4, 4));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x897, x_desc, x752, x_desc, x763, x_desc, x752,
    x899, x_desc, x763));
};
// conv2D back-propagate
float* x903 = (float*)myMalloc(1 * sizeof(float));;
x903[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 512, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 4, 4));

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
    x903, filt_desc, x98, grad_out_desc, x763,
    conv_desc, algo, ws_data, ws_size,
    x903, grad_in_desc, x751));
};
float* x906 = (float*)myMalloc(1 * sizeof(float));;
x906[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 512, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 4, 4));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 4, 4));

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
    x906, in_desc, x749, grad_out_desc, x763,
    conv_desc, algo, ws_data, ws_size,
    x906, grad_filt_desc, x253));
};
float* x909 = (float*)myMalloc(1 * sizeof(float));;
x909[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 4, 4));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x909, grad_out_desc, x763,
    x909, grad_bias_desc, x272));
};
float* x912 = (float*)myMalloc(1 * sizeof(float));;
x912[0] = 0.0f;
float* x914 = (float*)myMalloc(1 * sizeof(float));;
x914[0] = 1.0f;

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
    64, 512, 4, 4));

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
    x914, out_desc, x749, out_desc, x751, in_desc, x742  , x912, in_desc, x744));
};
{
dim3 grid(28, 2);
concat2D_1D_greg_grad<<<grid, 512>>>(x719, 256, 1048576, x736, 256, 1048576, x744, 1, 64, 512, 8, 8, 32768, 64, 8, 1);
};
float* x918 = (float*)myMalloc(1 * sizeof(float));;
x918[0] = 1.0f;
float* x920 = (float*)myMalloc(1 * sizeof(float));;
x920[0] = 0.0f;

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
    x918, x_desc, x725, x_desc, x736, x_desc, x725,
    x920, x_desc, x736));
};
// conv2D back-propagate
float* x924 = (float*)myMalloc(1 * sizeof(float));;
x924[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

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
    x924, filt_desc, x179, grad_out_desc, x736,
    conv_desc, algo, ws_data, ws_size,
    x924, grad_in_desc, x702));
};
float* x927 = (float*)myMalloc(1 * sizeof(float));;
x927[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    256, 64, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 8, 8));

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
    x927, in_desc, x691, grad_out_desc, x736,
    conv_desc, algo, ws_data, ws_size,
    x927, grad_filt_desc, x280));
};
float* x930 = (float*)myMalloc(1 * sizeof(float));;
x930[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x930, grad_out_desc, x736,
    x930, grad_bias_desc, x265));
};
float* x933 = (float*)myMalloc(1 * sizeof(float));;
x933[0] = 1.0f;
float* x935 = (float*)myMalloc(1 * sizeof(float));;
x935[0] = 0.0f;

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
    x933, x_desc, x708, x_desc, x719, x_desc, x708,
    x935, x_desc, x719));
};
// conv2D back-propagate
float* x939 = (float*)myMalloc(1 * sizeof(float));;
x939[0] = 1.0f;

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
    64, 64, 8, 8));

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
    x939, filt_desc, x242, grad_out_desc, x719,
    conv_desc, algo, ws_data, ws_size,
    x939, grad_in_desc, x702));
};
float* x942 = (float*)myMalloc(1 * sizeof(float));;
x942[0] = 1.0f;

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
    64, 256, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 8, 8));

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
    x942, in_desc, x691, grad_out_desc, x719,
    conv_desc, algo, ws_data, ws_size,
    x942, grad_filt_desc, x301));
};
float* x945 = (float*)myMalloc(1 * sizeof(float));;
x945[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 256, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x945, grad_out_desc, x719,
    x945, grad_bias_desc, x292));
};
float* x948 = (float*)myMalloc(1 * sizeof(float));;
x948[0] = 1.0f;
float* x950 = (float*)myMalloc(1 * sizeof(float));;
x950[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x948, x_desc, x691, x_desc, x702, x_desc, x691,
    x950, x_desc, x702));
};
// conv2D back-propagate
float* x954 = (float*)myMalloc(1 * sizeof(float));;
x954[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 384, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 384, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 8, 8));

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
    x954, filt_desc, x197, grad_out_desc, x702,
    conv_desc, algo, ws_data, ws_size,
    x954, grad_in_desc, x690));
};
float* x957 = (float*)myMalloc(1 * sizeof(float));;
x957[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 384, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 384, 8, 8));

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
    x957, in_desc, x688, grad_out_desc, x702,
    conv_desc, algo, ws_data, ws_size,
    x957, grad_filt_desc, x286));
};
float* x960 = (float*)myMalloc(1 * sizeof(float));;
x960[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 8, 8));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x960, grad_out_desc, x702,
    x960, grad_bias_desc, x261));
};
{
dim3 grid(28, 2);
concat2D_1D_greg_grad<<<grid, 512>>>(x665, 192, 786432, x682, 192, 786432, x690, 1, 64, 384, 8, 8, 24576, 64, 8, 1);
};
float* x964 = (float*)myMalloc(1 * sizeof(float));;
x964[0] = 1.0f;
float* x966 = (float*)myMalloc(1 * sizeof(float));;
x966[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x964, x_desc, x671, x_desc, x682, x_desc, x671,
    x966, x_desc, x682));
};
// conv2D back-propagate
float* x970 = (float*)myMalloc(1 * sizeof(float));;
x970[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    192, 48, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

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
    x970, filt_desc, x149, grad_out_desc, x682,
    conv_desc, algo, ws_data, ws_size,
    x970, grad_in_desc, x648));
};
float* x973 = (float*)myMalloc(1 * sizeof(float));;
x973[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    192, 48, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

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
    x973, in_desc, x637, grad_out_desc, x682,
    conv_desc, algo, ws_data, ws_size,
    x973, grad_filt_desc, x270));
};
float* x976 = (float*)myMalloc(1 * sizeof(float));;
x976[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 192, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x976, grad_out_desc, x682,
    x976, grad_bias_desc, x296));
};
float* x979 = (float*)myMalloc(1 * sizeof(float));;
x979[0] = 1.0f;
float* x981 = (float*)myMalloc(1 * sizeof(float));;
x981[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x979, x_desc, x654, x_desc, x665, x_desc, x654,
    x981, x_desc, x665));
};
// conv2D back-propagate
float* x985 = (float*)myMalloc(1 * sizeof(float));;
x985[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    192, 48, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

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
    x985, filt_desc, x161, grad_out_desc, x665,
    conv_desc, algo, ws_data, ws_size,
    x985, grad_in_desc, x648));
};
float* x988 = (float*)myMalloc(1 * sizeof(float));;
x988[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    192, 48, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

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
    x988, in_desc, x637, grad_out_desc, x665,
    conv_desc, algo, ws_data, ws_size,
    x988, grad_filt_desc, x274));
};
float* x991 = (float*)myMalloc(1 * sizeof(float));;
x991[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 192, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x991, grad_out_desc, x665,
    x991, grad_bias_desc, x284));
};
float* x994 = (float*)myMalloc(1 * sizeof(float));;
x994[0] = 1.0f;
float* x996 = (float*)myMalloc(1 * sizeof(float));;
x996[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x994, x_desc, x637, x_desc, x648, x_desc, x637,
    x996, x_desc, x648));
};
// conv2D back-propagate
float* x1000 = (float*)myMalloc(1 * sizeof(float));;
x1000[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    48, 384, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 384, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

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
    x1000, filt_desc, x137, grad_out_desc, x648,
    conv_desc, algo, ws_data, ws_size,
    x1000, grad_in_desc, x636));
};
float* x1003 = (float*)myMalloc(1 * sizeof(float));;
x1003[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    48, 384, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 384, 8, 8));

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
    x1003, in_desc, x634, grad_out_desc, x648,
    conv_desc, algo, ws_data, ws_size,
    x1003, grad_filt_desc, x266));
};
float* x1006 = (float*)myMalloc(1 * sizeof(float));;
x1006[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 48, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x1006, grad_out_desc, x648,
    x1006, grad_bias_desc, x254));
};
{
dim3 grid(28, 2);
concat2D_1D_greg_grad<<<grid, 512>>>(x611, 192, 786432, x628, 192, 786432, x636, 1, 64, 384, 8, 8, 24576, 64, 8, 1);
};
float* x1010 = (float*)myMalloc(1 * sizeof(float));;
x1010[0] = 1.0f;
float* x1012 = (float*)myMalloc(1 * sizeof(float));;
x1012[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x1010, x_desc, x617, x_desc, x628, x_desc, x617,
    x1012, x_desc, x628));
};
// conv2D back-propagate
float* x1016 = (float*)myMalloc(1 * sizeof(float));;
x1016[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    192, 48, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

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
    x1016, filt_desc, x212, grad_out_desc, x628,
    conv_desc, algo, ws_data, ws_size,
    x1016, grad_in_desc, x594));
};
float* x1019 = (float*)myMalloc(1 * sizeof(float));;
x1019[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    192, 48, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

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
    x1019, in_desc, x583, grad_out_desc, x628,
    conv_desc, algo, ws_data, ws_size,
    x1019, grad_filt_desc, x291));
};
float* x1022 = (float*)myMalloc(1 * sizeof(float));;
x1022[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 192, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x1022, grad_out_desc, x628,
    x1022, grad_bias_desc, x281));
};
float* x1025 = (float*)myMalloc(1 * sizeof(float));;
x1025[0] = 1.0f;
float* x1027 = (float*)myMalloc(1 * sizeof(float));;
x1027[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x1025, x_desc, x600, x_desc, x611, x_desc, x600,
    x1027, x_desc, x611));
};
// conv2D back-propagate
float* x1031 = (float*)myMalloc(1 * sizeof(float));;
x1031[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    192, 48, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

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
    x1031, filt_desc, x239, grad_out_desc, x611,
    conv_desc, algo, ws_data, ws_size,
    x1031, grad_in_desc, x594));
};
float* x1034 = (float*)myMalloc(1 * sizeof(float));;
x1034[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    192, 48, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

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
    x1034, in_desc, x583, grad_out_desc, x611,
    conv_desc, algo, ws_data, ws_size,
    x1034, grad_filt_desc, x300));
};
float* x1037 = (float*)myMalloc(1 * sizeof(float));;
x1037[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 192, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 192, 8, 8));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x1037, grad_out_desc, x611,
    x1037, grad_bias_desc, x298));
};
float* x1040 = (float*)myMalloc(1 * sizeof(float));;
x1040[0] = 1.0f;
float* x1042 = (float*)myMalloc(1 * sizeof(float));;
x1042[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x1040, x_desc, x583, x_desc, x594, x_desc, x583,
    x1042, x_desc, x594));
};
// conv2D back-propagate
float* x1046 = (float*)myMalloc(1 * sizeof(float));;
x1046[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    48, 256, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

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
    x1046, filt_desc, x221, grad_out_desc, x594,
    conv_desc, algo, ws_data, ws_size,
    x1046, grad_in_desc, x582));
};
float* x1049 = (float*)myMalloc(1 * sizeof(float));;
x1049[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    48, 256, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

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
    x1049, in_desc, x580, grad_out_desc, x594,
    conv_desc, algo, ws_data, ws_size,
    x1049, grad_filt_desc, x294));
};
float* x1052 = (float*)myMalloc(1 * sizeof(float));;
x1052[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 48, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x1052, grad_out_desc, x594,
    x1052, grad_bias_desc, x304));
};
{
dim3 grid(28, 2);
concat2D_1D_greg_grad<<<grid, 512>>>(x557, 128, 524288, x574, 128, 524288, x582, 1, 64, 256, 8, 8, 16384, 64, 8, 1);
};
float* x1056 = (float*)myMalloc(1 * sizeof(float));;
x1056[0] = 1.0f;
float* x1058 = (float*)myMalloc(1 * sizeof(float));;
x1058[0] = 0.0f;

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
    x1056, x_desc, x563, x_desc, x574, x_desc, x563,
    x1058, x_desc, x574));
};
// conv2D back-propagate
float* x1062 = (float*)myMalloc(1 * sizeof(float));;
x1062[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 32, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 8, 8));

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
    x1062, filt_desc, x203, grad_out_desc, x574,
    conv_desc, algo, ws_data, ws_size,
    x1062, grad_in_desc, x540));
};
float* x1065 = (float*)myMalloc(1 * sizeof(float));;
x1065[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 32, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 8, 8));

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
    x1065, in_desc, x529, grad_out_desc, x574,
    conv_desc, algo, ws_data, ws_size,
    x1065, grad_filt_desc, x288));
};
float* x1068 = (float*)myMalloc(1 * sizeof(float));;
x1068[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x1068, grad_out_desc, x574,
    x1068, grad_bias_desc, x268));
};
float* x1071 = (float*)myMalloc(1 * sizeof(float));;
x1071[0] = 1.0f;
float* x1073 = (float*)myMalloc(1 * sizeof(float));;
x1073[0] = 0.0f;

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
    x1071, x_desc, x546, x_desc, x557, x_desc, x546,
    x1073, x_desc, x557));
};
// conv2D back-propagate
float* x1077 = (float*)myMalloc(1 * sizeof(float));;
x1077[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 32, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 8, 8));

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
    x1077, filt_desc, x116, grad_out_desc, x557,
    conv_desc, algo, ws_data, ws_size,
    x1077, grad_in_desc, x540));
};
float* x1080 = (float*)myMalloc(1 * sizeof(float));;
x1080[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 32, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 8, 8));

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
    x1080, in_desc, x529, grad_out_desc, x557,
    conv_desc, algo, ws_data, ws_size,
    x1080, grad_filt_desc, x259));
};
float* x1083 = (float*)myMalloc(1 * sizeof(float));;
x1083[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 8, 8));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x1083, grad_out_desc, x557,
    x1083, grad_bias_desc, x273));
};
float* x1086 = (float*)myMalloc(1 * sizeof(float));;
x1086[0] = 1.0f;
float* x1088 = (float*)myMalloc(1 * sizeof(float));;
x1088[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 8, 8));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x1086, x_desc, x529, x_desc, x540, x_desc, x529,
    x1088, x_desc, x540));
};
// conv2D back-propagate
float* x1092 = (float*)myMalloc(1 * sizeof(float));;
x1092[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    32, 256, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 8, 8));

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
    x1092, filt_desc, x176, grad_out_desc, x540,
    conv_desc, algo, ws_data, ws_size,
    x1092, grad_in_desc, x528));
};
float* x1095 = (float*)myMalloc(1 * sizeof(float));;
x1095[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    32, 256, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 8, 8));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 8, 8));

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
    x1095, in_desc, x526, grad_out_desc, x540,
    conv_desc, algo, ws_data, ws_size,
    x1095, grad_filt_desc, x279));
};
float* x1098 = (float*)myMalloc(1 * sizeof(float));;
x1098[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 32, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 8, 8));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x1098, grad_out_desc, x540,
    x1098, grad_bias_desc, x267));
};
float* x1101 = (float*)myMalloc(1 * sizeof(float));;
x1101[0] = 0.0f;
float* x1103 = (float*)myMalloc(1 * sizeof(float));;
x1103[0] = 1.0f;

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
    64, 256, 8, 8));

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
    x1103, out_desc, x526, out_desc, x528, in_desc, x519  , x1101, in_desc, x521));
};
{
dim3 grid(28, 2);
concat2D_1D_greg_grad<<<grid, 512>>>(x496, 128, 2097152, x513, 128, 2097152, x521, 1, 64, 256, 16, 16, 65536, 256, 16, 1);
};
float* x1107 = (float*)myMalloc(1 * sizeof(float));;
x1107[0] = 1.0f;
float* x1109 = (float*)myMalloc(1 * sizeof(float));;
x1109[0] = 0.0f;

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
    x1107, x_desc, x502, x_desc, x513, x_desc, x502,
    x1109, x_desc, x513));
};
// conv2D back-propagate
float* x1113 = (float*)myMalloc(1 * sizeof(float));;
x1113[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 32, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 16, 16));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

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
    x1113, filt_desc, x113, grad_out_desc, x513,
    conv_desc, algo, ws_data, ws_size,
    x1113, grad_in_desc, x479));
};
float* x1116 = (float*)myMalloc(1 * sizeof(float));;
x1116[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 32, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 16, 16));

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
    x1116, in_desc, x468, grad_out_desc, x513,
    conv_desc, algo, ws_data, ws_size,
    x1116, grad_filt_desc, x258));
};
float* x1119 = (float*)myMalloc(1 * sizeof(float));;
x1119[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x1119, grad_out_desc, x513,
    x1119, grad_bias_desc, x293));
};
float* x1122 = (float*)myMalloc(1 * sizeof(float));;
x1122[0] = 1.0f;
float* x1124 = (float*)myMalloc(1 * sizeof(float));;
x1124[0] = 0.0f;

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
    x1122, x_desc, x485, x_desc, x496, x_desc, x485,
    x1124, x_desc, x496));
};
// conv2D back-propagate
float* x1128 = (float*)myMalloc(1 * sizeof(float));;
x1128[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 32, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 16, 16));

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
    x1128, filt_desc, x200, grad_out_desc, x496,
    conv_desc, algo, ws_data, ws_size,
    x1128, grad_in_desc, x479));
};
float* x1131 = (float*)myMalloc(1 * sizeof(float));;
x1131[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    128, 32, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 16, 16));

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
    x1131, in_desc, x468, grad_out_desc, x496,
    conv_desc, algo, ws_data, ws_size,
    x1131, grad_filt_desc, x287));
};
float* x1134 = (float*)myMalloc(1 * sizeof(float));;
x1134[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 128, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x1134, grad_out_desc, x496,
    x1134, grad_bias_desc, x297));
};
float* x1137 = (float*)myMalloc(1 * sizeof(float));;
x1137[0] = 1.0f;
float* x1139 = (float*)myMalloc(1 * sizeof(float));;
x1139[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x1137, x_desc, x468, x_desc, x479, x_desc, x468,
    x1139, x_desc, x479));
};
// conv2D back-propagate
float* x1143 = (float*)myMalloc(1 * sizeof(float));;
x1143[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    32, 128, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 16, 16));

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
    x1143, filt_desc, x125, grad_out_desc, x479,
    conv_desc, algo, ws_data, ws_size,
    x1143, grad_in_desc, x467));
};
float* x1146 = (float*)myMalloc(1 * sizeof(float));;
x1146[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    32, 128, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 16, 16));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

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
    x1146, in_desc, x465, grad_out_desc, x479,
    conv_desc, algo, ws_data, ws_size,
    x1146, grad_filt_desc, x262));
};
float* x1149 = (float*)myMalloc(1 * sizeof(float));;
x1149[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 32, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 32, 16, 16));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x1149, grad_out_desc, x479,
    x1149, grad_bias_desc, x275));
};
{
dim3 grid(28, 2);
concat2D_1D_greg_grad<<<grid, 512>>>(x442, 64, 1048576, x459, 64, 1048576, x467, 1, 64, 128, 16, 16, 32768, 256, 16, 1);
};
float* x1153 = (float*)myMalloc(1 * sizeof(float));;
x1153[0] = 1.0f;
float* x1155 = (float*)myMalloc(1 * sizeof(float));;
x1155[0] = 0.0f;

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
    x1153, x_desc, x448, x_desc, x459, x_desc, x448,
    x1155, x_desc, x459));
};
// conv2D back-propagate
float* x1159 = (float*)myMalloc(1 * sizeof(float));;
x1159[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 16, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

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
    x1159, filt_desc, x152, grad_out_desc, x459,
    conv_desc, algo, ws_data, ws_size,
    x1159, grad_in_desc, x425));
};
float* x1162 = (float*)myMalloc(1 * sizeof(float));;
x1162[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 16, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

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
    x1162, in_desc, x414, grad_out_desc, x459,
    conv_desc, algo, ws_data, ws_size,
    x1162, grad_filt_desc, x271));
};
float* x1165 = (float*)myMalloc(1 * sizeof(float));;
x1165[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x1165, grad_out_desc, x459,
    x1165, grad_bias_desc, x289));
};
float* x1168 = (float*)myMalloc(1 * sizeof(float));;
x1168[0] = 1.0f;
float* x1170 = (float*)myMalloc(1 * sizeof(float));;
x1170[0] = 0.0f;

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
    x1168, x_desc, x431, x_desc, x442, x_desc, x431,
    x1170, x_desc, x442));
};
// conv2D back-propagate
float* x1174 = (float*)myMalloc(1 * sizeof(float));;
x1174[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 16, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

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
    x1174, filt_desc, x128, grad_out_desc, x442,
    conv_desc, algo, ws_data, ws_size,
    x1174, grad_in_desc, x425));
};
float* x1177 = (float*)myMalloc(1 * sizeof(float));;
x1177[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 16, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

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
    x1177, in_desc, x414, grad_out_desc, x442,
    conv_desc, algo, ws_data, ws_size,
    x1177, grad_filt_desc, x263));
};
float* x1180 = (float*)myMalloc(1 * sizeof(float));;
x1180[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x1180, grad_out_desc, x442,
    x1180, grad_bias_desc, x255));
};
float* x1183 = (float*)myMalloc(1 * sizeof(float));;
x1183[0] = 1.0f;
float* x1185 = (float*)myMalloc(1 * sizeof(float));;
x1185[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x1183, x_desc, x414, x_desc, x425, x_desc, x414,
    x1185, x_desc, x425));
};
// conv2D back-propagate
float* x1189 = (float*)myMalloc(1 * sizeof(float));;
x1189[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    16, 128, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

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
    x1189, filt_desc, x131, grad_out_desc, x425,
    conv_desc, algo, ws_data, ws_size,
    x1189, grad_in_desc, x413));
};
float* x1192 = (float*)myMalloc(1 * sizeof(float));;
x1192[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    16, 128, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 128, 16, 16));

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
    x1192, in_desc, x411, grad_out_desc, x425,
    conv_desc, algo, ws_data, ws_size,
    x1192, grad_filt_desc, x264));
};
float* x1195 = (float*)myMalloc(1 * sizeof(float));;
x1195[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 16, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x1195, grad_out_desc, x425,
    x1195, grad_bias_desc, x277));
};
{
dim3 grid(28, 2);
concat2D_1D_greg_grad<<<grid, 512>>>(x388, 64, 1048576, x405, 64, 1048576, x413, 1, 64, 128, 16, 16, 32768, 256, 16, 1);
};
float* x1199 = (float*)myMalloc(1 * sizeof(float));;
x1199[0] = 1.0f;
float* x1201 = (float*)myMalloc(1 * sizeof(float));;
x1201[0] = 0.0f;

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
    x1199, x_desc, x394, x_desc, x405, x_desc, x394,
    x1201, x_desc, x405));
};
// conv2D back-propagate
float* x1205 = (float*)myMalloc(1 * sizeof(float));;
x1205[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 16, 3, 3));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

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
    x1205, filt_desc, x236, grad_out_desc, x405,
    conv_desc, algo, ws_data, ws_size,
    x1205, grad_in_desc, x371));
};
float* x1208 = (float*)myMalloc(1 * sizeof(float));;
x1208[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 16, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

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
    x1208, in_desc, x360, grad_out_desc, x405,
    conv_desc, algo, ws_data, ws_size,
    x1208, grad_filt_desc, x299));
};
float* x1211 = (float*)myMalloc(1 * sizeof(float));;
x1211[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x1211, grad_out_desc, x405,
    x1211, grad_bias_desc, x257));
};
float* x1214 = (float*)myMalloc(1 * sizeof(float));;
x1214[0] = 1.0f;
float* x1216 = (float*)myMalloc(1 * sizeof(float));;
x1216[0] = 0.0f;

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
    x1214, x_desc, x377, x_desc, x388, x_desc, x377,
    x1216, x_desc, x388));
};
// conv2D back-propagate
float* x1220 = (float*)myMalloc(1 * sizeof(float));;
x1220[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 16, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

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
    x1220, filt_desc, x167, grad_out_desc, x388,
    conv_desc, algo, ws_data, ws_size,
    x1220, grad_in_desc, x371));
};
float* x1223 = (float*)myMalloc(1 * sizeof(float));;
x1223[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    64, 16, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

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
    x1223, in_desc, x360, grad_out_desc, x388,
    conv_desc, algo, ws_data, ws_size,
    x1223, grad_filt_desc, x276));
};
float* x1226 = (float*)myMalloc(1 * sizeof(float));;
x1226[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 64, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 64, 16, 16));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x1226, grad_out_desc, x388,
    x1226, grad_bias_desc, x283));
};
float* x1229 = (float*)myMalloc(1 * sizeof(float));;
x1229[0] = 1.0f;
float* x1231 = (float*)myMalloc(1 * sizeof(float));;
x1231[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x1229, x_desc, x360, x_desc, x371, x_desc, x360,
    x1231, x_desc, x371));
};
// conv2D back-propagate
float* x1235 = (float*)myMalloc(1 * sizeof(float));;
x1235[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    16, 96, 1, 1));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 96, 16, 16));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

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
    x1235, filt_desc, x245, grad_out_desc, x371,
    conv_desc, algo, ws_data, ws_size,
    x1235, grad_in_desc, x359));
};
float* x1238 = (float*)myMalloc(1 * sizeof(float));;
x1238[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    16, 96, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 96, 16, 16));

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
    x1238, in_desc, x357, grad_out_desc, x371,
    conv_desc, algo, ws_data, ws_size,
    x1238, grad_filt_desc, x302));
};
float* x1241 = (float*)myMalloc(1 * sizeof(float));;
x1241[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 16, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x1241, grad_out_desc, x371,
    x1241, grad_bias_desc, x260));
};
float* x1244 = (float*)myMalloc(1 * sizeof(float));;
x1244[0] = 0.0f;
float* x1246 = (float*)myMalloc(1 * sizeof(float));;
x1246[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 96, 32, 32));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 96, 16, 16));

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
    x1246, out_desc, x357, out_desc, x359, in_desc, x336  , x1244, in_desc, x347));
};
float* x1249 = (float*)myMalloc(1 * sizeof(float));;
x1249[0] = 1.0f;
float* x1251 = (float*)myMalloc(1 * sizeof(float));;
x1251[0] = 0.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 96, 32, 32));

cudnnActivationDescriptor_t act_desc;
CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
                                        /*mode=*/ CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/ 0));
CUDNN_CALL(cudnnActivationBackward(
    cudnnHandle, act_desc,
    x1249, x_desc, x336, x_desc, x347, x_desc, x336,
    x1251, x_desc, x347));
};
// conv2D back-propagate
float* x1255 = (float*)myMalloc(1 * sizeof(float));;
x1255[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    96, 3, 3, 3));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 96, 32, 32));

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
    x1255, in_desc, x327, grad_out_desc, x347,
    conv_desc, algo, ws_data, ws_size,
    x1255, grad_filt_desc, x285));
};
float* x1258 = (float*)myMalloc(1 * sizeof(float));;
x1258[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 96, 1, 1));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 96, 32, 32));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x1258, grad_out_desc, x347,
    x1258, grad_bias_desc, x295));
};
float x1261 = x335[0];
x315 += x1261;
float* x1263 = (float*)myMalloc(1 * sizeof(float));;
x1263[0] = 1.0f;
float* x1265 = (float*)myMalloc(1 * sizeof(float));;
x1265[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,64,x1263,x98,512,x1265, x253, 512, x98,512));
arrayFill_greg<<<7, 512>>>(x253, 0.0f, 32768);
float* x1269 = (float*)myMalloc(1 * sizeof(float));;
x1269[0] = 1.0f;
float* x1271 = (float*)myMalloc(1 * sizeof(float));;
x1271[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,48,x1269,x101,1,x1271, x254, 1, x101,1));
arrayFill_greg<<<1, 512>>>(x254, 0.0f, 48);
float* x1275 = (float*)myMalloc(1 * sizeof(float));;
x1275[0] = 1.0f;
float* x1277 = (float*)myMalloc(1 * sizeof(float));;
x1277[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x1275,x104,1,x1277, x255, 1, x104,1));
arrayFill_greg<<<1, 512>>>(x255, 0.0f, 64);
float* x1281 = (float*)myMalloc(1 * sizeof(float));;
x1281[0] = 1.0f;
float* x1283 = (float*)myMalloc(1 * sizeof(float));;
x1283[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 8192,10,x1281,x107,8192,x1283, x256, 8192, x107,8192));
arrayFill_greg<<<17, 512>>>(x256, 0.0f, 81920);
float* x1287 = (float*)myMalloc(1 * sizeof(float));;
x1287[0] = 1.0f;
float* x1289 = (float*)myMalloc(1 * sizeof(float));;
x1289[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x1287,x110,1,x1289, x257, 1, x110,1));
arrayFill_greg<<<1, 512>>>(x257, 0.0f, 64);
float* x1293 = (float*)myMalloc(1 * sizeof(float));;
x1293[0] = 1.0f;
float* x1295 = (float*)myMalloc(1 * sizeof(float));;
x1295[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 288,128,x1293,x113,288,x1295, x258, 288, x113,288));
arrayFill_greg<<<8, 512>>>(x258, 0.0f, 36864);
float* x1299 = (float*)myMalloc(1 * sizeof(float));;
x1299[0] = 1.0f;
float* x1301 = (float*)myMalloc(1 * sizeof(float));;
x1301[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 32,128,x1299,x116,32,x1301, x259, 32, x116,32));
arrayFill_greg<<<1, 512>>>(x259, 0.0f, 4096);
float* x1305 = (float*)myMalloc(1 * sizeof(float));;
x1305[0] = 1.0f;
float* x1307 = (float*)myMalloc(1 * sizeof(float));;
x1307[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,16,x1305,x119,1,x1307, x260, 1, x119,1));
arrayFill_greg<<<1, 512>>>(x260, 0.0f, 16);
float* x1311 = (float*)myMalloc(1 * sizeof(float));;
x1311[0] = 1.0f;
float* x1313 = (float*)myMalloc(1 * sizeof(float));;
x1313[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x1311,x122,1,x1313, x261, 1, x122,1));
arrayFill_greg<<<1, 512>>>(x261, 0.0f, 64);
float* x1317 = (float*)myMalloc(1 * sizeof(float));;
x1317[0] = 1.0f;
float* x1319 = (float*)myMalloc(1 * sizeof(float));;
x1319[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,32,x1317,x125,128,x1319, x262, 128, x125,128));
arrayFill_greg<<<1, 512>>>(x262, 0.0f, 4096);
float* x1323 = (float*)myMalloc(1 * sizeof(float));;
x1323[0] = 1.0f;
float* x1325 = (float*)myMalloc(1 * sizeof(float));;
x1325[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 16,64,x1323,x128,16,x1325, x263, 16, x128,16));
arrayFill_greg<<<1, 512>>>(x263, 0.0f, 1024);
float* x1329 = (float*)myMalloc(1 * sizeof(float));;
x1329[0] = 1.0f;
float* x1331 = (float*)myMalloc(1 * sizeof(float));;
x1331[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,16,x1329,x131,128,x1331, x264, 128, x131,128));
arrayFill_greg<<<1, 512>>>(x264, 0.0f, 2048);
float* x1335 = (float*)myMalloc(1 * sizeof(float));;
x1335[0] = 1.0f;
float* x1337 = (float*)myMalloc(1 * sizeof(float));;
x1337[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x1335,x134,1,x1337, x265, 1, x134,1));
arrayFill_greg<<<1, 512>>>(x265, 0.0f, 256);
float* x1341 = (float*)myMalloc(1 * sizeof(float));;
x1341[0] = 1.0f;
float* x1343 = (float*)myMalloc(1 * sizeof(float));;
x1343[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 384,48,x1341,x137,384,x1343, x266, 384, x137,384));
arrayFill_greg<<<4, 512>>>(x266, 0.0f, 18432);
float* x1347 = (float*)myMalloc(1 * sizeof(float));;
x1347[0] = 1.0f;
float* x1349 = (float*)myMalloc(1 * sizeof(float));;
x1349[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1347,x140,1,x1349, x267, 1, x140,1));
arrayFill_greg<<<1, 512>>>(x267, 0.0f, 32);
float* x1353 = (float*)myMalloc(1 * sizeof(float));;
x1353[0] = 1.0f;
float* x1355 = (float*)myMalloc(1 * sizeof(float));;
x1355[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x1353,x143,1,x1355, x268, 1, x143,1));
arrayFill_greg<<<1, 512>>>(x268, 0.0f, 128);
float* x1359 = (float*)myMalloc(1 * sizeof(float));;
x1359[0] = 1.0f;
float* x1361 = (float*)myMalloc(1 * sizeof(float));;
x1361[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x1359,x146,1,x1361, x269, 1, x146,1));
arrayFill_greg<<<1, 512>>>(x269, 0.0f, 256);
float* x1365 = (float*)myMalloc(1 * sizeof(float));;
x1365[0] = 1.0f;
float* x1367 = (float*)myMalloc(1 * sizeof(float));;
x1367[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 432,192,x1365,x149,432,x1367, x270, 432, x149,432));
arrayFill_greg<<<17, 512>>>(x270, 0.0f, 82944);
float* x1371 = (float*)myMalloc(1 * sizeof(float));;
x1371[0] = 1.0f;
float* x1373 = (float*)myMalloc(1 * sizeof(float));;
x1373[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 144,64,x1371,x152,144,x1373, x271, 144, x152,144));
arrayFill_greg<<<2, 512>>>(x271, 0.0f, 9216);
float* x1377 = (float*)myMalloc(1 * sizeof(float));;
x1377[0] = 1.0f;
float* x1379 = (float*)myMalloc(1 * sizeof(float));;
x1379[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x1377,x155,1,x1379, x272, 1, x155,1));
arrayFill_greg<<<1, 512>>>(x272, 0.0f, 64);
float* x1383 = (float*)myMalloc(1 * sizeof(float));;
x1383[0] = 1.0f;
float* x1385 = (float*)myMalloc(1 * sizeof(float));;
x1385[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x1383,x158,1,x1385, x273, 1, x158,1));
arrayFill_greg<<<1, 512>>>(x273, 0.0f, 128);
float* x1389 = (float*)myMalloc(1 * sizeof(float));;
x1389[0] = 1.0f;
float* x1391 = (float*)myMalloc(1 * sizeof(float));;
x1391[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 48,192,x1389,x161,48,x1391, x274, 48, x161,48));
arrayFill_greg<<<2, 512>>>(x274, 0.0f, 9216);
float* x1395 = (float*)myMalloc(1 * sizeof(float));;
x1395[0] = 1.0f;
float* x1397 = (float*)myMalloc(1 * sizeof(float));;
x1397[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1395,x164,1,x1397, x275, 1, x164,1));
arrayFill_greg<<<1, 512>>>(x275, 0.0f, 32);
float* x1401 = (float*)myMalloc(1 * sizeof(float));;
x1401[0] = 1.0f;
float* x1403 = (float*)myMalloc(1 * sizeof(float));;
x1403[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 16,64,x1401,x167,16,x1403, x276, 16, x167,16));
arrayFill_greg<<<1, 512>>>(x276, 0.0f, 1024);
float* x1407 = (float*)myMalloc(1 * sizeof(float));;
x1407[0] = 1.0f;
float* x1409 = (float*)myMalloc(1 * sizeof(float));;
x1409[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,16,x1407,x170,1,x1409, x277, 1, x170,1));
arrayFill_greg<<<1, 512>>>(x277, 0.0f, 16);
float* x1413 = (float*)myMalloc(1 * sizeof(float));;
x1413[0] = 1.0f;
float* x1415 = (float*)myMalloc(1 * sizeof(float));;
x1415[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x1413,x173,1,x1415, x278, 1, x173,1));
arrayFill_greg<<<1, 512>>>(x278, 0.0f, 256);
float* x1419 = (float*)myMalloc(1 * sizeof(float));;
x1419[0] = 1.0f;
float* x1421 = (float*)myMalloc(1 * sizeof(float));;
x1421[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,32,x1419,x176,256,x1421, x279, 256, x176,256));
arrayFill_greg<<<2, 512>>>(x279, 0.0f, 8192);
float* x1425 = (float*)myMalloc(1 * sizeof(float));;
x1425[0] = 1.0f;
float* x1427 = (float*)myMalloc(1 * sizeof(float));;
x1427[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 576,256,x1425,x179,576,x1427, x280, 576, x179,576));
arrayFill_greg<<<29, 512>>>(x280, 0.0f, 147456);
float* x1431 = (float*)myMalloc(1 * sizeof(float));;
x1431[0] = 1.0f;
float* x1433 = (float*)myMalloc(1 * sizeof(float));;
x1433[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,192,x1431,x182,1,x1433, x281, 1, x182,1));
arrayFill_greg<<<1, 512>>>(x281, 0.0f, 192);
float* x1437 = (float*)myMalloc(1 * sizeof(float));;
x1437[0] = 1.0f;
float* x1439 = (float*)myMalloc(1 * sizeof(float));;
x1439[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 576,256,x1437,x185,576,x1439, x282, 576, x185,576));
arrayFill_greg<<<29, 512>>>(x282, 0.0f, 147456);
float* x1443 = (float*)myMalloc(1 * sizeof(float));;
x1443[0] = 1.0f;
float* x1445 = (float*)myMalloc(1 * sizeof(float));;
x1445[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x1443,x188,1,x1445, x283, 1, x188,1));
arrayFill_greg<<<1, 512>>>(x283, 0.0f, 64);
float* x1449 = (float*)myMalloc(1 * sizeof(float));;
x1449[0] = 1.0f;
float* x1451 = (float*)myMalloc(1 * sizeof(float));;
x1451[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,192,x1449,x191,1,x1451, x284, 1, x191,1));
arrayFill_greg<<<1, 512>>>(x284, 0.0f, 192);
float* x1455 = (float*)myMalloc(1 * sizeof(float));;
x1455[0] = 1.0f;
float* x1457 = (float*)myMalloc(1 * sizeof(float));;
x1457[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 27,96,x1455,x194,27,x1457, x285, 27, x194,27));
arrayFill_greg<<<1, 512>>>(x285, 0.0f, 2592);
float* x1461 = (float*)myMalloc(1 * sizeof(float));;
x1461[0] = 1.0f;
float* x1463 = (float*)myMalloc(1 * sizeof(float));;
x1463[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 384,64,x1461,x197,384,x1463, x286, 384, x197,384));
arrayFill_greg<<<5, 512>>>(x286, 0.0f, 24576);
float* x1467 = (float*)myMalloc(1 * sizeof(float));;
x1467[0] = 1.0f;
float* x1469 = (float*)myMalloc(1 * sizeof(float));;
x1469[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 32,128,x1467,x200,32,x1469, x287, 32, x200,32));
arrayFill_greg<<<1, 512>>>(x287, 0.0f, 4096);
float* x1473 = (float*)myMalloc(1 * sizeof(float));;
x1473[0] = 1.0f;
float* x1475 = (float*)myMalloc(1 * sizeof(float));;
x1475[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 288,128,x1473,x203,288,x1475, x288, 288, x203,288));
arrayFill_greg<<<8, 512>>>(x288, 0.0f, 36864);
float* x1479 = (float*)myMalloc(1 * sizeof(float));;
x1479[0] = 1.0f;
float* x1481 = (float*)myMalloc(1 * sizeof(float));;
x1481[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x1479,x206,1,x1481, x289, 1, x206,1));
arrayFill_greg<<<1, 512>>>(x289, 0.0f, 64);
float* x1485 = (float*)myMalloc(1 * sizeof(float));;
x1485[0] = 1.0f;
float* x1487 = (float*)myMalloc(1 * sizeof(float));;
x1487[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x1485,x209,64,x1487, x290, 64, x209,64));
arrayFill_greg<<<4, 512>>>(x290, 0.0f, 16384);
float* x1491 = (float*)myMalloc(1 * sizeof(float));;
x1491[0] = 1.0f;
float* x1493 = (float*)myMalloc(1 * sizeof(float));;
x1493[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 432,192,x1491,x212,432,x1493, x291, 432, x212,432));
arrayFill_greg<<<17, 512>>>(x291, 0.0f, 82944);
float* x1497 = (float*)myMalloc(1 * sizeof(float));;
x1497[0] = 1.0f;
float* x1499 = (float*)myMalloc(1 * sizeof(float));;
x1499[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x1497,x215,1,x1499, x292, 1, x215,1));
arrayFill_greg<<<1, 512>>>(x292, 0.0f, 256);
float* x1503 = (float*)myMalloc(1 * sizeof(float));;
x1503[0] = 1.0f;
float* x1505 = (float*)myMalloc(1 * sizeof(float));;
x1505[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x1503,x218,1,x1505, x293, 1, x218,1));
arrayFill_greg<<<1, 512>>>(x293, 0.0f, 128);
float* x1509 = (float*)myMalloc(1 * sizeof(float));;
x1509[0] = 1.0f;
float* x1511 = (float*)myMalloc(1 * sizeof(float));;
x1511[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,48,x1509,x221,256,x1511, x294, 256, x221,256));
arrayFill_greg<<<3, 512>>>(x294, 0.0f, 12288);
float* x1515 = (float*)myMalloc(1 * sizeof(float));;
x1515[0] = 1.0f;
float* x1517 = (float*)myMalloc(1 * sizeof(float));;
x1517[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,96,x1515,x224,1,x1517, x295, 1, x224,1));
arrayFill_greg<<<1, 512>>>(x295, 0.0f, 96);
float* x1521 = (float*)myMalloc(1 * sizeof(float));;
x1521[0] = 1.0f;
float* x1523 = (float*)myMalloc(1 * sizeof(float));;
x1523[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,192,x1521,x227,1,x1523, x296, 1, x227,1));
arrayFill_greg<<<1, 512>>>(x296, 0.0f, 192);
float* x1527 = (float*)myMalloc(1 * sizeof(float));;
x1527[0] = 1.0f;
float* x1529 = (float*)myMalloc(1 * sizeof(float));;
x1529[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x1527,x230,1,x1529, x297, 1, x230,1));
arrayFill_greg<<<1, 512>>>(x297, 0.0f, 128);
float* x1533 = (float*)myMalloc(1 * sizeof(float));;
x1533[0] = 1.0f;
float* x1535 = (float*)myMalloc(1 * sizeof(float));;
x1535[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,192,x1533,x233,1,x1535, x298, 1, x233,1));
arrayFill_greg<<<1, 512>>>(x298, 0.0f, 192);
float* x1539 = (float*)myMalloc(1 * sizeof(float));;
x1539[0] = 1.0f;
float* x1541 = (float*)myMalloc(1 * sizeof(float));;
x1541[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 144,64,x1539,x236,144,x1541, x299, 144, x236,144));
arrayFill_greg<<<2, 512>>>(x299, 0.0f, 9216);
float* x1545 = (float*)myMalloc(1 * sizeof(float));;
x1545[0] = 1.0f;
float* x1547 = (float*)myMalloc(1 * sizeof(float));;
x1547[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 48,192,x1545,x239,48,x1547, x300, 48, x239,48));
arrayFill_greg<<<2, 512>>>(x300, 0.0f, 9216);
float* x1551 = (float*)myMalloc(1 * sizeof(float));;
x1551[0] = 1.0f;
float* x1553 = (float*)myMalloc(1 * sizeof(float));;
x1553[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x1551,x242,64,x1553, x301, 64, x242,64));
arrayFill_greg<<<4, 512>>>(x301, 0.0f, 16384);
float* x1557 = (float*)myMalloc(1 * sizeof(float));;
x1557[0] = 1.0f;
float* x1559 = (float*)myMalloc(1 * sizeof(float));;
x1559[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 96,16,x1557,x245,96,x1559, x302, 96, x245,96));
arrayFill_greg<<<1, 512>>>(x302, 0.0f, 1536);
float* x1563 = (float*)myMalloc(1 * sizeof(float));;
x1563[0] = 1.0f;
float* x1565 = (float*)myMalloc(1 * sizeof(float));;
x1565[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,10,x1563,x248,1,x1565, x303, 1, x248,1));
arrayFill_greg<<<1, 512>>>(x303, 0.0f, 10);
float* x1569 = (float*)myMalloc(1 * sizeof(float));;
x1569[0] = 1.0f;
float* x1571 = (float*)myMalloc(1 * sizeof(float));;
x1571[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,48,x1569,x251,1,x1571, x304, 1, x251,1));
arrayFill_greg<<<1, 512>>>(x304, 0.0f, 48);
int32_t x1575 = x321 + 1;
int32_t x1577 = x1575 % x1576;
bool x1578 = x1577 == 0;
if (x1578) {
float x1583 = x315;
double x1579 = (double)x322;
double x1580 = 100.0 * x1579;
double x1582 = x1580 / x1581;
float x1584 = (float)x321;
float x1585 = x1583 / x1584;
printf("Train epoch %d: [%d/%d (%.0f%%)] Average Loss: %.6f\n",x311,x322,x12,x1582,x1585);
fflush(stdout);
} else {
}
int64_t x1590 = (long)mallocAddr;
int64_t x1591 = x1590 - x307;
memset((void*)x307, 0, x1591);
mallocAddr = (void*)x307;
int64_t x1594 = (long)gpuMallocAddr;
int64_t x1595 = x1594 - x308;
cudaMemset((void*)x308, 0, x1595);
gpuMallocAddr = (void*)x308;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x1602 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
double x1603 = (double)x1602;
double x1604 = x1603 / 1000000.0;
x306[x311] = x1604;
int64_t x1606 = x1602 / 1000LL;
int64_t x1608 = x1602 / x1607;
printf("Training completed in %ldms (%ld us/images)\n",x1606,x1608);
float x1610 = x315;
float x1612 = x1610 / x1611;
double x1613 = (double)x1612;
x305[x311] = x1613;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x1619 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
sort(x306, x306 + 4);
double x1625 = x306[2];
int64_t x1626 = (long)fopen(x0, "w");
fprintf((FILE *)x1626, "unit: %s\n", "1 epoch");
for(int x1628=0; x1628 < 4; x1628++) {
double x1629 = x305[x1628];
fprintf((FILE *)x1626, "%lf\n", x1629);

}
fprintf((FILE *)x1626, "run time: %lf %lf\n", x40, x1625);
fclose((FILE*)x1626);
// Backend cleanup.
CUBLAS_CALL(cublasDestroy(cublasHandle));
CUDA_CALL(cudaFree(gpuMallocBase));
      
CUDNN_CALL(cudnnDestroy(cudnnHandle));
}
/*****************************************
  End of C Generated Code                  
*******************************************/


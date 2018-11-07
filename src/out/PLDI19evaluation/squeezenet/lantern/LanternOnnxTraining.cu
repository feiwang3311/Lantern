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

// dimSize: size on concat dimension. Only for Dim of rank 4
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
// dimSize: size on concat dimension. Only for Dim of rank 4
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
int64_t x306 = (long)mallocAddr;
int64_t x307 = (long)gpuMallocAddr;
// training loop starts here
int32_t x318 = x12 / 64;
int32_t x1678 = x318 / 10;
double x1683 = (double)x12;
int64_t x1706 = (int64_t)x12;
float x1710 = (float)x12;
for(int x310=0; x310 < 4; x310++) {
struct timeval begin_1, end_1, diff_1;
float x312 = 0.0f;
float x313 = x312;
float x314 = x313;
int32_t x315 = x310 + 1;
printf("Start training epoch %d\n",x315);
gettimeofday(&begin_1, NULL);
for(int x320=0; x320 < x318; x320++) {
int32_t x321 = x320 * 64;
int32_t x322 = x321 * 3072;
float* x323 = x14+x322;
int* x324 = x15+x321;
// Tensor 'toGPU' invocation.
float* x326 = (float*)myGpuMalloc(196608 * sizeof(float));
CUDA_CALL(cudaMemcpy(x326, x323, 196608 * sizeof(float), cudaMemcpyHostToDevice));
float* x328 = (float*)myGpuMalloc(2 * sizeof(float));
int* x329 = (int32_t*)myGpuMalloc(64 * sizeof(int32_t));
CUDA_CALL(cudaMemcpy(x329, x324, 64 * sizeof(int32_t), cudaMemcpyHostToDevice));
float* x331 = (float*)myGpuMalloc(1 * sizeof(float));
float* x332 = (float*)myGpuMalloc(1 * sizeof(float));
// allocate memory to save the final loss in CPU Tensor
float* x334 = (float*)myMalloc(1 * sizeof(float));;
float* x335 = (float*)myGpuMalloc(6291456 * sizeof(float));
float* x336 = (float*)myMalloc(1 * sizeof(float));;
x336[0] = 0.0f;
float* x338 = (float*)myMalloc(1 * sizeof(float));;
x338[0] = 1.0f;

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
    x338, in_desc, x326, filt_desc, x194,
    conv_desc, algo, ws_data, ws_size,
    x336, out_desc, x335));
};
float* x341 = (float*)myMalloc(1 * sizeof(float));;
x341[0] = 1.0f;
float* x343 = (float*)myMalloc(1 * sizeof(float));;
x343[0] = 1.0f;

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
    cudnnHandle, x341, bias_desc, x224, x343, out_desc, x335));
};
float* x346 = (float*)myGpuMalloc(6291456 * sizeof(float));
float* x347 = (float*)myMalloc(1 * sizeof(float));;
x347[0] = 0.0f;
float* x349 = (float*)myMalloc(1 * sizeof(float));;
x349[0] = 1.0f;
float* x351 = (float*)myGpuMalloc(6291456 * sizeof(float));

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
    x349, x_desc, x335, x347, x_desc, x351));
};
float* x353 = (float*)myGpuMalloc(6291456 * sizeof(float));
float* x354 = (float*)myMalloc(1 * sizeof(float));;
x354[0] = 0.0f;
float* x356 = (float*)myMalloc(1 * sizeof(float));;
x356[0] = 1.0f;
float* x358 = (float*)myGpuMalloc(1572864 * sizeof(float));

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
    x356, in_desc, x351, x354, out_desc, x358));
};
float* x360 = (float*)myGpuMalloc(1572864 * sizeof(float));
float* x361 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x362 = (float*)myMalloc(1 * sizeof(float));;
x362[0] = 0.0f;
float* x364 = (float*)myMalloc(1 * sizeof(float));;
x364[0] = 1.0f;

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
    x364, in_desc, x358, filt_desc, x245,
    conv_desc, algo, ws_data, ws_size,
    x362, out_desc, x361));
};
float* x367 = (float*)myMalloc(1 * sizeof(float));;
x367[0] = 1.0f;
float* x369 = (float*)myMalloc(1 * sizeof(float));;
x369[0] = 1.0f;

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
    cudnnHandle, x367, bias_desc, x119, x369, out_desc, x361));
};
float* x372 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x373 = (float*)myMalloc(1 * sizeof(float));;
x373[0] = 0.0f;
float* x375 = (float*)myMalloc(1 * sizeof(float));;
x375[0] = 1.0f;
float* x377 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x375, x_desc, x361, x373, x_desc, x377));
};
float* x379 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x380 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x381 = (float*)myMalloc(1 * sizeof(float));;
x381[0] = 0.0f;
float* x383 = (float*)myMalloc(1 * sizeof(float));;
x383[0] = 1.0f;

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
    x383, in_desc, x377, filt_desc, x167,
    conv_desc, algo, ws_data, ws_size,
    x381, out_desc, x380));
};
float* x386 = (float*)myMalloc(1 * sizeof(float));;
x386[0] = 1.0f;
float* x388 = (float*)myMalloc(1 * sizeof(float));;
x388[0] = 1.0f;

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
    cudnnHandle, x386, bias_desc, x188, x388, out_desc, x380));
};
float* x391 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x392 = (float*)myMalloc(1 * sizeof(float));;
x392[0] = 0.0f;
float* x394 = (float*)myMalloc(1 * sizeof(float));;
x394[0] = 1.0f;
float* x396 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x394, x_desc, x380, x392, x_desc, x396));
};
float* x398 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x399 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x400 = (float*)myMalloc(1 * sizeof(float));;
x400[0] = 0.0f;
float* x402 = (float*)myMalloc(1 * sizeof(float));;
x402[0] = 1.0f;

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
    x402, in_desc, x377, filt_desc, x236,
    conv_desc, algo, ws_data, ws_size,
    x400, out_desc, x399));
};
float* x405 = (float*)myMalloc(1 * sizeof(float));;
x405[0] = 1.0f;
float* x407 = (float*)myMalloc(1 * sizeof(float));;
x407[0] = 1.0f;

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
    cudnnHandle, x405, bias_desc, x110, x407, out_desc, x399));
};
float* x410 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x411 = (float*)myMalloc(1 * sizeof(float));;
x411[0] = 0.0f;
float* x413 = (float*)myMalloc(1 * sizeof(float));;
x413[0] = 1.0f;
float* x415 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x413, x_desc, x399, x411, x_desc, x415));
};
float* x417 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x418 = (float*)myGpuMalloc(2097152 * sizeof(float));
int* x419 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x419[0] = 64;
x419[1] = 128;
x419[2] = 16;
x419[3] = 16;
int* x424 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x424[0] = 32768;
x424[1] = 256;
x424[2] = 16;
x424[3] = 1;
{
dim3 grid(400, 2);
concat2D_1D_greg<<<grid, 512>>>(x396, 64, 1048576, x415, 64, 1048576, x418, 1, 64, 128, 16, 16, 32768, 256, 16, 1);
};
float* x430 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x431 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x432 = (float*)myMalloc(1 * sizeof(float));;
x432[0] = 0.0f;
float* x434 = (float*)myMalloc(1 * sizeof(float));;
x434[0] = 1.0f;

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
    x434, in_desc, x418, filt_desc, x131,
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
    1, 16, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 16, 16, 16));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x437, bias_desc, x170, x439, out_desc, x431));
};
float* x442 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x443 = (float*)myMalloc(1 * sizeof(float));;
x443[0] = 0.0f;
float* x445 = (float*)myMalloc(1 * sizeof(float));;
x445[0] = 1.0f;
float* x447 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x445, x_desc, x431, x443, x_desc, x447));
};
float* x449 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x450 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x451 = (float*)myMalloc(1 * sizeof(float));;
x451[0] = 0.0f;
float* x453 = (float*)myMalloc(1 * sizeof(float));;
x453[0] = 1.0f;

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
    x453, in_desc, x447, filt_desc, x128,
    conv_desc, algo, ws_data, ws_size,
    x451, out_desc, x450));
};
float* x456 = (float*)myMalloc(1 * sizeof(float));;
x456[0] = 1.0f;
float* x458 = (float*)myMalloc(1 * sizeof(float));;
x458[0] = 1.0f;

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
    cudnnHandle, x456, bias_desc, x104, x458, out_desc, x450));
};
float* x461 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x462 = (float*)myMalloc(1 * sizeof(float));;
x462[0] = 0.0f;
float* x464 = (float*)myMalloc(1 * sizeof(float));;
x464[0] = 1.0f;
float* x466 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x464, x_desc, x450, x462, x_desc, x466));
};
float* x468 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x469 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x470 = (float*)myMalloc(1 * sizeof(float));;
x470[0] = 0.0f;
float* x472 = (float*)myMalloc(1 * sizeof(float));;
x472[0] = 1.0f;

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
    x472, in_desc, x447, filt_desc, x152,
    conv_desc, algo, ws_data, ws_size,
    x470, out_desc, x469));
};
float* x475 = (float*)myMalloc(1 * sizeof(float));;
x475[0] = 1.0f;
float* x477 = (float*)myMalloc(1 * sizeof(float));;
x477[0] = 1.0f;

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
    cudnnHandle, x475, bias_desc, x206, x477, out_desc, x469));
};
float* x480 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x481 = (float*)myMalloc(1 * sizeof(float));;
x481[0] = 0.0f;
float* x483 = (float*)myMalloc(1 * sizeof(float));;
x483[0] = 1.0f;
float* x485 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x483, x_desc, x469, x481, x_desc, x485));
};
float* x487 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x488 = (float*)myGpuMalloc(2097152 * sizeof(float));
int* x489 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x489[0] = 64;
x489[1] = 128;
x489[2] = 16;
x489[3] = 16;
int* x494 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x494[0] = 32768;
x494[1] = 256;
x494[2] = 16;
x494[3] = 1;
{
dim3 grid(400, 2);
concat2D_1D_greg<<<grid, 512>>>(x466, 64, 1048576, x485, 64, 1048576, x488, 1, 64, 128, 16, 16, 32768, 256, 16, 1);
};
float* x500 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x501 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x502 = (float*)myMalloc(1 * sizeof(float));;
x502[0] = 0.0f;
float* x504 = (float*)myMalloc(1 * sizeof(float));;
x504[0] = 1.0f;

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
    x504, in_desc, x488, filt_desc, x125,
    conv_desc, algo, ws_data, ws_size,
    x502, out_desc, x501));
};
float* x507 = (float*)myMalloc(1 * sizeof(float));;
x507[0] = 1.0f;
float* x509 = (float*)myMalloc(1 * sizeof(float));;
x509[0] = 1.0f;

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
    cudnnHandle, x507, bias_desc, x164, x509, out_desc, x501));
};
float* x512 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x513 = (float*)myMalloc(1 * sizeof(float));;
x513[0] = 0.0f;
float* x515 = (float*)myMalloc(1 * sizeof(float));;
x515[0] = 1.0f;
float* x517 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x515, x_desc, x501, x513, x_desc, x517));
};
float* x519 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x520 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x521 = (float*)myMalloc(1 * sizeof(float));;
x521[0] = 0.0f;
float* x523 = (float*)myMalloc(1 * sizeof(float));;
x523[0] = 1.0f;

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
    x523, in_desc, x517, filt_desc, x200,
    conv_desc, algo, ws_data, ws_size,
    x521, out_desc, x520));
};
float* x526 = (float*)myMalloc(1 * sizeof(float));;
x526[0] = 1.0f;
float* x528 = (float*)myMalloc(1 * sizeof(float));;
x528[0] = 1.0f;

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
    cudnnHandle, x526, bias_desc, x230, x528, out_desc, x520));
};
float* x531 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x532 = (float*)myMalloc(1 * sizeof(float));;
x532[0] = 0.0f;
float* x534 = (float*)myMalloc(1 * sizeof(float));;
x534[0] = 1.0f;
float* x536 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x534, x_desc, x520, x532, x_desc, x536));
};
float* x538 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x539 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x540 = (float*)myMalloc(1 * sizeof(float));;
x540[0] = 0.0f;
float* x542 = (float*)myMalloc(1 * sizeof(float));;
x542[0] = 1.0f;

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
    x542, in_desc, x517, filt_desc, x113,
    conv_desc, algo, ws_data, ws_size,
    x540, out_desc, x539));
};
float* x545 = (float*)myMalloc(1 * sizeof(float));;
x545[0] = 1.0f;
float* x547 = (float*)myMalloc(1 * sizeof(float));;
x547[0] = 1.0f;

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
    cudnnHandle, x545, bias_desc, x218, x547, out_desc, x539));
};
float* x550 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x551 = (float*)myMalloc(1 * sizeof(float));;
x551[0] = 0.0f;
float* x553 = (float*)myMalloc(1 * sizeof(float));;
x553[0] = 1.0f;
float* x555 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x553, x_desc, x539, x551, x_desc, x555));
};
float* x557 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x558 = (float*)myGpuMalloc(4194304 * sizeof(float));
int* x559 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x559[0] = 64;
x559[1] = 256;
x559[2] = 16;
x559[3] = 16;
int* x564 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x564[0] = 65536;
x564[1] = 256;
x564[2] = 16;
x564[3] = 1;
{
dim3 grid(400, 2);
concat2D_1D_greg<<<grid, 512>>>(x536, 128, 2097152, x555, 128, 2097152, x558, 1, 64, 256, 16, 16, 65536, 256, 16, 1);
};
float* x570 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x571 = (float*)myMalloc(1 * sizeof(float));;
x571[0] = 0.0f;
float* x573 = (float*)myMalloc(1 * sizeof(float));;
x573[0] = 1.0f;
float* x575 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x573, in_desc, x558, x571, out_desc, x575));
};
float* x577 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x578 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x579 = (float*)myMalloc(1 * sizeof(float));;
x579[0] = 0.0f;
float* x581 = (float*)myMalloc(1 * sizeof(float));;
x581[0] = 1.0f;

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
    x581, in_desc, x575, filt_desc, x176,
    conv_desc, algo, ws_data, ws_size,
    x579, out_desc, x578));
};
float* x584 = (float*)myMalloc(1 * sizeof(float));;
x584[0] = 1.0f;
float* x586 = (float*)myMalloc(1 * sizeof(float));;
x586[0] = 1.0f;

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
    cudnnHandle, x584, bias_desc, x140, x586, out_desc, x578));
};
float* x589 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x590 = (float*)myMalloc(1 * sizeof(float));;
x590[0] = 0.0f;
float* x592 = (float*)myMalloc(1 * sizeof(float));;
x592[0] = 1.0f;
float* x594 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x592, x_desc, x578, x590, x_desc, x594));
};
float* x596 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x597 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x598 = (float*)myMalloc(1 * sizeof(float));;
x598[0] = 0.0f;
float* x600 = (float*)myMalloc(1 * sizeof(float));;
x600[0] = 1.0f;

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
    x600, in_desc, x594, filt_desc, x116,
    conv_desc, algo, ws_data, ws_size,
    x598, out_desc, x597));
};
float* x603 = (float*)myMalloc(1 * sizeof(float));;
x603[0] = 1.0f;
float* x605 = (float*)myMalloc(1 * sizeof(float));;
x605[0] = 1.0f;

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
    cudnnHandle, x603, bias_desc, x158, x605, out_desc, x597));
};
float* x608 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x609 = (float*)myMalloc(1 * sizeof(float));;
x609[0] = 0.0f;
float* x611 = (float*)myMalloc(1 * sizeof(float));;
x611[0] = 1.0f;
float* x613 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x611, x_desc, x597, x609, x_desc, x613));
};
float* x615 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x616 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x617 = (float*)myMalloc(1 * sizeof(float));;
x617[0] = 0.0f;
float* x619 = (float*)myMalloc(1 * sizeof(float));;
x619[0] = 1.0f;

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
    x619, in_desc, x594, filt_desc, x203,
    conv_desc, algo, ws_data, ws_size,
    x617, out_desc, x616));
};
float* x622 = (float*)myMalloc(1 * sizeof(float));;
x622[0] = 1.0f;
float* x624 = (float*)myMalloc(1 * sizeof(float));;
x624[0] = 1.0f;

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
    cudnnHandle, x622, bias_desc, x143, x624, out_desc, x616));
};
float* x627 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x628 = (float*)myMalloc(1 * sizeof(float));;
x628[0] = 0.0f;
float* x630 = (float*)myMalloc(1 * sizeof(float));;
x630[0] = 1.0f;
float* x632 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x630, x_desc, x616, x628, x_desc, x632));
};
float* x634 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x635 = (float*)myGpuMalloc(1048576 * sizeof(float));
int* x636 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x636[0] = 64;
x636[1] = 256;
x636[2] = 8;
x636[3] = 8;
int* x641 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x641[0] = 16384;
x641[1] = 64;
x641[2] = 8;
x641[3] = 1;
{
dim3 grid(400, 2);
concat2D_1D_greg<<<grid, 512>>>(x613, 128, 524288, x632, 128, 524288, x635, 1, 64, 256, 8, 8, 16384, 64, 8, 1);
};
float* x647 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x648 = (float*)myGpuMalloc(196608 * sizeof(float));
float* x649 = (float*)myMalloc(1 * sizeof(float));;
x649[0] = 0.0f;
float* x651 = (float*)myMalloc(1 * sizeof(float));;
x651[0] = 1.0f;

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
    x651, in_desc, x635, filt_desc, x221,
    conv_desc, algo, ws_data, ws_size,
    x649, out_desc, x648));
};
float* x654 = (float*)myMalloc(1 * sizeof(float));;
x654[0] = 1.0f;
float* x656 = (float*)myMalloc(1 * sizeof(float));;
x656[0] = 1.0f;

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
    cudnnHandle, x654, bias_desc, x251, x656, out_desc, x648));
};
float* x659 = (float*)myGpuMalloc(196608 * sizeof(float));
float* x660 = (float*)myMalloc(1 * sizeof(float));;
x660[0] = 0.0f;
float* x662 = (float*)myMalloc(1 * sizeof(float));;
x662[0] = 1.0f;
float* x664 = (float*)myGpuMalloc(196608 * sizeof(float));

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
    x662, x_desc, x648, x660, x_desc, x664));
};
float* x666 = (float*)myGpuMalloc(196608 * sizeof(float));
float* x667 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x668 = (float*)myMalloc(1 * sizeof(float));;
x668[0] = 0.0f;
float* x670 = (float*)myMalloc(1 * sizeof(float));;
x670[0] = 1.0f;

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
    x670, in_desc, x664, filt_desc, x239,
    conv_desc, algo, ws_data, ws_size,
    x668, out_desc, x667));
};
float* x673 = (float*)myMalloc(1 * sizeof(float));;
x673[0] = 1.0f;
float* x675 = (float*)myMalloc(1 * sizeof(float));;
x675[0] = 1.0f;

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
    cudnnHandle, x673, bias_desc, x233, x675, out_desc, x667));
};
float* x678 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x679 = (float*)myMalloc(1 * sizeof(float));;
x679[0] = 0.0f;
float* x681 = (float*)myMalloc(1 * sizeof(float));;
x681[0] = 1.0f;
float* x683 = (float*)myGpuMalloc(786432 * sizeof(float));

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
    x681, x_desc, x667, x679, x_desc, x683));
};
float* x685 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x686 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x687 = (float*)myMalloc(1 * sizeof(float));;
x687[0] = 0.0f;
float* x689 = (float*)myMalloc(1 * sizeof(float));;
x689[0] = 1.0f;

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
    x689, in_desc, x664, filt_desc, x212,
    conv_desc, algo, ws_data, ws_size,
    x687, out_desc, x686));
};
float* x692 = (float*)myMalloc(1 * sizeof(float));;
x692[0] = 1.0f;
float* x694 = (float*)myMalloc(1 * sizeof(float));;
x694[0] = 1.0f;

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
    cudnnHandle, x692, bias_desc, x182, x694, out_desc, x686));
};
float* x697 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x698 = (float*)myMalloc(1 * sizeof(float));;
x698[0] = 0.0f;
float* x700 = (float*)myMalloc(1 * sizeof(float));;
x700[0] = 1.0f;
float* x702 = (float*)myGpuMalloc(786432 * sizeof(float));

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
    x700, x_desc, x686, x698, x_desc, x702));
};
float* x704 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x705 = (float*)myGpuMalloc(1572864 * sizeof(float));
int* x706 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x706[0] = 64;
x706[1] = 384;
x706[2] = 8;
x706[3] = 8;
int* x711 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x711[0] = 24576;
x711[1] = 64;
x711[2] = 8;
x711[3] = 1;
{
dim3 grid(400, 2);
concat2D_1D_greg<<<grid, 512>>>(x683, 192, 786432, x702, 192, 786432, x705, 1, 64, 384, 8, 8, 24576, 64, 8, 1);
};
float* x717 = (float*)myGpuMalloc(1572864 * sizeof(float));
float* x718 = (float*)myGpuMalloc(196608 * sizeof(float));
float* x719 = (float*)myMalloc(1 * sizeof(float));;
x719[0] = 0.0f;
float* x721 = (float*)myMalloc(1 * sizeof(float));;
x721[0] = 1.0f;

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
    x721, in_desc, x705, filt_desc, x137,
    conv_desc, algo, ws_data, ws_size,
    x719, out_desc, x718));
};
float* x724 = (float*)myMalloc(1 * sizeof(float));;
x724[0] = 1.0f;
float* x726 = (float*)myMalloc(1 * sizeof(float));;
x726[0] = 1.0f;

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
    cudnnHandle, x724, bias_desc, x101, x726, out_desc, x718));
};
float* x729 = (float*)myGpuMalloc(196608 * sizeof(float));
float* x730 = (float*)myMalloc(1 * sizeof(float));;
x730[0] = 0.0f;
float* x732 = (float*)myMalloc(1 * sizeof(float));;
x732[0] = 1.0f;
float* x734 = (float*)myGpuMalloc(196608 * sizeof(float));

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
    x732, x_desc, x718, x730, x_desc, x734));
};
float* x736 = (float*)myGpuMalloc(196608 * sizeof(float));
float* x737 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x738 = (float*)myMalloc(1 * sizeof(float));;
x738[0] = 0.0f;
float* x740 = (float*)myMalloc(1 * sizeof(float));;
x740[0] = 1.0f;

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
    x740, in_desc, x734, filt_desc, x161,
    conv_desc, algo, ws_data, ws_size,
    x738, out_desc, x737));
};
float* x743 = (float*)myMalloc(1 * sizeof(float));;
x743[0] = 1.0f;
float* x745 = (float*)myMalloc(1 * sizeof(float));;
x745[0] = 1.0f;

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
    cudnnHandle, x743, bias_desc, x191, x745, out_desc, x737));
};
float* x748 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x749 = (float*)myMalloc(1 * sizeof(float));;
x749[0] = 0.0f;
float* x751 = (float*)myMalloc(1 * sizeof(float));;
x751[0] = 1.0f;
float* x753 = (float*)myGpuMalloc(786432 * sizeof(float));

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
    x751, x_desc, x737, x749, x_desc, x753));
};
float* x755 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x756 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x757 = (float*)myMalloc(1 * sizeof(float));;
x757[0] = 0.0f;
float* x759 = (float*)myMalloc(1 * sizeof(float));;
x759[0] = 1.0f;

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
    x759, in_desc, x734, filt_desc, x149,
    conv_desc, algo, ws_data, ws_size,
    x757, out_desc, x756));
};
float* x762 = (float*)myMalloc(1 * sizeof(float));;
x762[0] = 1.0f;
float* x764 = (float*)myMalloc(1 * sizeof(float));;
x764[0] = 1.0f;

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
    cudnnHandle, x762, bias_desc, x227, x764, out_desc, x756));
};
float* x767 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x768 = (float*)myMalloc(1 * sizeof(float));;
x768[0] = 0.0f;
float* x770 = (float*)myMalloc(1 * sizeof(float));;
x770[0] = 1.0f;
float* x772 = (float*)myGpuMalloc(786432 * sizeof(float));

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
    x770, x_desc, x756, x768, x_desc, x772));
};
float* x774 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x775 = (float*)myGpuMalloc(1572864 * sizeof(float));
int* x776 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x776[0] = 64;
x776[1] = 384;
x776[2] = 8;
x776[3] = 8;
int* x781 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x781[0] = 24576;
x781[1] = 64;
x781[2] = 8;
x781[3] = 1;
{
dim3 grid(400, 2);
concat2D_1D_greg<<<grid, 512>>>(x753, 192, 786432, x772, 192, 786432, x775, 1, 64, 384, 8, 8, 24576, 64, 8, 1);
};
float* x787 = (float*)myGpuMalloc(1572864 * sizeof(float));
float* x788 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x789 = (float*)myMalloc(1 * sizeof(float));;
x789[0] = 0.0f;
float* x791 = (float*)myMalloc(1 * sizeof(float));;
x791[0] = 1.0f;

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
    x791, in_desc, x775, filt_desc, x197,
    conv_desc, algo, ws_data, ws_size,
    x789, out_desc, x788));
};
float* x794 = (float*)myMalloc(1 * sizeof(float));;
x794[0] = 1.0f;
float* x796 = (float*)myMalloc(1 * sizeof(float));;
x796[0] = 1.0f;

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
    cudnnHandle, x794, bias_desc, x122, x796, out_desc, x788));
};
float* x799 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x800 = (float*)myMalloc(1 * sizeof(float));;
x800[0] = 0.0f;
float* x802 = (float*)myMalloc(1 * sizeof(float));;
x802[0] = 1.0f;
float* x804 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x802, x_desc, x788, x800, x_desc, x804));
};
float* x806 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x807 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x808 = (float*)myMalloc(1 * sizeof(float));;
x808[0] = 0.0f;
float* x810 = (float*)myMalloc(1 * sizeof(float));;
x810[0] = 1.0f;

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
    x810, in_desc, x804, filt_desc, x242,
    conv_desc, algo, ws_data, ws_size,
    x808, out_desc, x807));
};
float* x813 = (float*)myMalloc(1 * sizeof(float));;
x813[0] = 1.0f;
float* x815 = (float*)myMalloc(1 * sizeof(float));;
x815[0] = 1.0f;

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
    cudnnHandle, x813, bias_desc, x215, x815, out_desc, x807));
};
float* x818 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x819 = (float*)myMalloc(1 * sizeof(float));;
x819[0] = 0.0f;
float* x821 = (float*)myMalloc(1 * sizeof(float));;
x821[0] = 1.0f;
float* x823 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x821, x_desc, x807, x819, x_desc, x823));
};
float* x825 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x826 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x827 = (float*)myMalloc(1 * sizeof(float));;
x827[0] = 0.0f;
float* x829 = (float*)myMalloc(1 * sizeof(float));;
x829[0] = 1.0f;

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
    x829, in_desc, x804, filt_desc, x179,
    conv_desc, algo, ws_data, ws_size,
    x827, out_desc, x826));
};
float* x832 = (float*)myMalloc(1 * sizeof(float));;
x832[0] = 1.0f;
float* x834 = (float*)myMalloc(1 * sizeof(float));;
x834[0] = 1.0f;

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
    cudnnHandle, x832, bias_desc, x134, x834, out_desc, x826));
};
float* x837 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x838 = (float*)myMalloc(1 * sizeof(float));;
x838[0] = 0.0f;
float* x840 = (float*)myMalloc(1 * sizeof(float));;
x840[0] = 1.0f;
float* x842 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x840, x_desc, x826, x838, x_desc, x842));
};
float* x844 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x845 = (float*)myGpuMalloc(2097152 * sizeof(float));
int* x846 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x846[0] = 64;
x846[1] = 512;
x846[2] = 8;
x846[3] = 8;
int* x851 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x851[0] = 32768;
x851[1] = 64;
x851[2] = 8;
x851[3] = 1;
{
dim3 grid(400, 2);
concat2D_1D_greg<<<grid, 512>>>(x823, 256, 1048576, x842, 256, 1048576, x845, 1, 64, 512, 8, 8, 32768, 64, 8, 1);
};
float* x857 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x858 = (float*)myMalloc(1 * sizeof(float));;
x858[0] = 0.0f;
float* x860 = (float*)myMalloc(1 * sizeof(float));;
x860[0] = 1.0f;
float* x862 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x860, in_desc, x845, x858, out_desc, x862));
};
float* x864 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x865 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x866 = (float*)myMalloc(1 * sizeof(float));;
x866[0] = 0.0f;
float* x868 = (float*)myMalloc(1 * sizeof(float));;
x868[0] = 1.0f;

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
    x868, in_desc, x862, filt_desc, x98,
    conv_desc, algo, ws_data, ws_size,
    x866, out_desc, x865));
};
float* x871 = (float*)myMalloc(1 * sizeof(float));;
x871[0] = 1.0f;
float* x873 = (float*)myMalloc(1 * sizeof(float));;
x873[0] = 1.0f;

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
    cudnnHandle, x871, bias_desc, x155, x873, out_desc, x865));
};
float* x876 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x877 = (float*)myMalloc(1 * sizeof(float));;
x877[0] = 0.0f;
float* x879 = (float*)myMalloc(1 * sizeof(float));;
x879[0] = 1.0f;
float* x881 = (float*)myGpuMalloc(65536 * sizeof(float));

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
    x879, x_desc, x865, x877, x_desc, x881));
};
float* x883 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x884 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x885 = (float*)myMalloc(1 * sizeof(float));;
x885[0] = 0.0f;
float* x887 = (float*)myMalloc(1 * sizeof(float));;
x887[0] = 1.0f;

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
    x887, in_desc, x881, filt_desc, x209,
    conv_desc, algo, ws_data, ws_size,
    x885, out_desc, x884));
};
float* x890 = (float*)myMalloc(1 * sizeof(float));;
x890[0] = 1.0f;
float* x892 = (float*)myMalloc(1 * sizeof(float));;
x892[0] = 1.0f;

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
    cudnnHandle, x890, bias_desc, x173, x892, out_desc, x884));
};
float* x895 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x896 = (float*)myMalloc(1 * sizeof(float));;
x896[0] = 0.0f;
float* x898 = (float*)myMalloc(1 * sizeof(float));;
x898[0] = 1.0f;
float* x900 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x898, x_desc, x884, x896, x_desc, x900));
};
float* x902 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x903 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x904 = (float*)myMalloc(1 * sizeof(float));;
x904[0] = 0.0f;
float* x906 = (float*)myMalloc(1 * sizeof(float));;
x906[0] = 1.0f;

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
    x906, in_desc, x881, filt_desc, x185,
    conv_desc, algo, ws_data, ws_size,
    x904, out_desc, x903));
};
float* x909 = (float*)myMalloc(1 * sizeof(float));;
x909[0] = 1.0f;
float* x911 = (float*)myMalloc(1 * sizeof(float));;
x911[0] = 1.0f;

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
    cudnnHandle, x909, bias_desc, x146, x911, out_desc, x903));
};
float* x914 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x915 = (float*)myMalloc(1 * sizeof(float));;
x915[0] = 0.0f;
float* x917 = (float*)myMalloc(1 * sizeof(float));;
x917[0] = 1.0f;
float* x919 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x917, x_desc, x903, x915, x_desc, x919));
};
float* x921 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x922 = (float*)myGpuMalloc(524288 * sizeof(float));
int* x923 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x923[0] = 64;
x923[1] = 512;
x923[2] = 4;
x923[3] = 4;
int* x928 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x928[0] = 8192;
x928[1] = 16;
x928[2] = 4;
x928[3] = 1;
{
dim3 grid(400, 2);
concat2D_1D_greg<<<grid, 512>>>(x900, 256, 262144, x919, 256, 262144, x922, 1, 64, 512, 4, 4, 8192, 16, 4, 1);
};
float* x934 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x935 = (float*)myGpuMalloc(640 * sizeof(float));
float* x936 = (float*)myMalloc(1 * sizeof(float));;
x936[0] = 0.0f;
float* x938 = (float*)myMalloc(1 * sizeof(float));;
x938[0] = 1.0f;

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
    x938, in_desc, x922, filt_desc, x107,
    conv_desc, algo, ws_data, ws_size,
    x936, out_desc, x935));
};
float* x941 = (float*)myMalloc(1 * sizeof(float));;
x941[0] = 1.0f;
float* x943 = (float*)myMalloc(1 * sizeof(float));;
x943[0] = 1.0f;

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
    cudnnHandle, x941, bias_desc, x248, x943, out_desc, x935));
};
float* x946 = (float*)myGpuMalloc(640 * sizeof(float));
// resize to WrappedArray(64, 10)
// resize to WrappedArray(64, 10)
float* x949 = (float*)myMalloc(1 * sizeof(float));;
x949[0] = 0.0f;
float* x951 = (float*)myMalloc(1 * sizeof(float));;
x951[0] = 1.0f;
float* x953 = (float*)myGpuMalloc(640 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 10, 1, 1));
CUDNN_CALL(cudnnSoftmaxForward(
    cudnnHandle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL,
    x951, x_desc, x935, x949, x_desc, x953));
};
float* x955 = (float*)myGpuMalloc(640 * sizeof(float));
float* x956 = (float*)myGpuMalloc(64 * sizeof(float));
nllLoss<<<64, 1>>>(x953,10,x956,x329);
float* x958 = (float*)myGpuMalloc(64 * sizeof(float));
// resize to ArrayBuffer(64, 1, 1, 1)
float* x960 = (float*)myGpuMalloc(1 * sizeof(float));
float* x961 = (float*)myMalloc(1 * sizeof(float));;
x961[0] = 0.0f;
float* x963 = (float*)myMalloc(1 * sizeof(float));;
x963[0] = 1.0f;

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
    x963, x_desc, x956, x961, out_desc, x960));
};
// resize to WrappedArray(1)
float* x967 = (float*)myGpuMalloc(1 * sizeof(float));
arrayFill_greg<<<1, 512>>>(x967, 1.0f, 1);
// backend is lantern.TensorDslCudnn$BackendCudnn@485d3311
CUDA_CALL(cudaMemcpy(x334, x960, 1 * sizeof(float), cudaMemcpyDeviceToHost));
// 'mean' gradient
// backprop for mean op
float* x973 = (float*)myMalloc(1 * sizeof(float));;
x973[0] = 0.015625f;
float* x975 = (float*)myMalloc(1 * sizeof(float));;
x975[0] = 1.0f;

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
    cudnnHandle, x973, bias_desc, x967, x975, out_desc, x958));
};
// 'nllLossB' gradient.
nllLoss_grad<<<64, 1>>>(10,x958,x329,x955);
float* x980 = (float*)myMalloc(1 * sizeof(float));;
x980[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 10, 1, 1));
CUDNN_CALL(cudnnSoftmaxBackward(
    cudnnHandle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL,
    x980, x_desc, x953, x_desc, x955,
    x980, x_desc, x946));
};
// conv2D back-propagate
float* x984 = (float*)myMalloc(1 * sizeof(float));;
x984[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x984, filt_desc, x107, grad_out_desc, x946,
    conv_desc, algo, ws_data, ws_size,
    x984, grad_in_desc, x934));
};
float* x987 = (float*)myMalloc(1 * sizeof(float));;
x987[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x987, in_desc, x922, grad_out_desc, x946,
    conv_desc, algo, ws_data, ws_size,
    x987, grad_filt_desc, x256));
};
float* x990 = (float*)myMalloc(1 * sizeof(float));;
x990[0] = 1.0f;

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
    cudnnHandle, x990, grad_out_desc, x946,
    x990, grad_bias_desc, x303));
};
{
dim3 grid(400, 2);
concat2D_1D_greg_grad<<<grid, 512>>>(x902, 256, 262144, x921, 256, 262144, x934, 1, 64, 512, 4, 4, 8192, 16, 4, 1);
};
float* x994 = (float*)myMalloc(1 * sizeof(float));;
x994[0] = 1.0f;
float* x996 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x994, x_desc, x919, x_desc, x921, x_desc, x903,
    x994, x_desc, x914));
};
// conv2D back-propagate
float* x999 = (float*)myMalloc(1 * sizeof(float));;
x999[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x999, filt_desc, x185, grad_out_desc, x914,
    conv_desc, algo, ws_data, ws_size,
    x999, grad_in_desc, x883));
};
float* x1002 = (float*)myMalloc(1 * sizeof(float));;
x1002[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1002, in_desc, x881, grad_out_desc, x914,
    conv_desc, algo, ws_data, ws_size,
    x1002, grad_filt_desc, x282));
};
float* x1005 = (float*)myMalloc(1 * sizeof(float));;
x1005[0] = 1.0f;

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
    cudnnHandle, x1005, grad_out_desc, x914,
    x1005, grad_bias_desc, x269));
};
float* x1008 = (float*)myMalloc(1 * sizeof(float));;
x1008[0] = 1.0f;
float* x1010 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1008, x_desc, x900, x_desc, x902, x_desc, x884,
    x1008, x_desc, x895));
};
// conv2D back-propagate
float* x1013 = (float*)myMalloc(1 * sizeof(float));;
x1013[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1013, filt_desc, x209, grad_out_desc, x895,
    conv_desc, algo, ws_data, ws_size,
    x1013, grad_in_desc, x883));
};
float* x1016 = (float*)myMalloc(1 * sizeof(float));;
x1016[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1016, in_desc, x881, grad_out_desc, x895,
    conv_desc, algo, ws_data, ws_size,
    x1016, grad_filt_desc, x290));
};
float* x1019 = (float*)myMalloc(1 * sizeof(float));;
x1019[0] = 1.0f;

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
    cudnnHandle, x1019, grad_out_desc, x895,
    x1019, grad_bias_desc, x278));
};
float* x1022 = (float*)myMalloc(1 * sizeof(float));;
x1022[0] = 1.0f;
float* x1024 = (float*)myGpuMalloc(65536 * sizeof(float));

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
    x1022, x_desc, x881, x_desc, x883, x_desc, x865,
    x1022, x_desc, x876));
};
// conv2D back-propagate
float* x1027 = (float*)myMalloc(1 * sizeof(float));;
x1027[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1027, filt_desc, x98, grad_out_desc, x876,
    conv_desc, algo, ws_data, ws_size,
    x1027, grad_in_desc, x864));
};
float* x1030 = (float*)myMalloc(1 * sizeof(float));;
x1030[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1030, in_desc, x862, grad_out_desc, x876,
    conv_desc, algo, ws_data, ws_size,
    x1030, grad_filt_desc, x253));
};
float* x1033 = (float*)myMalloc(1 * sizeof(float));;
x1033[0] = 1.0f;

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
    cudnnHandle, x1033, grad_out_desc, x876,
    x1033, grad_bias_desc, x272));
};
float* x1036 = (float*)myMalloc(1 * sizeof(float));;
x1036[0] = 0.0f;
float* x1038 = (float*)myMalloc(1 * sizeof(float));;
x1038[0] = 1.0f;

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
    x1038, out_desc, x862, out_desc, x864, in_desc, x845  , x1036, in_desc, x857));
};
{
dim3 grid(400, 2);
concat2D_1D_greg_grad<<<grid, 512>>>(x825, 256, 1048576, x844, 256, 1048576, x857, 1, 64, 512, 8, 8, 32768, 64, 8, 1);
};
float* x1042 = (float*)myMalloc(1 * sizeof(float));;
x1042[0] = 1.0f;
float* x1044 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1042, x_desc, x842, x_desc, x844, x_desc, x826,
    x1042, x_desc, x837));
};
// conv2D back-propagate
float* x1047 = (float*)myMalloc(1 * sizeof(float));;
x1047[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1047, filt_desc, x179, grad_out_desc, x837,
    conv_desc, algo, ws_data, ws_size,
    x1047, grad_in_desc, x806));
};
float* x1050 = (float*)myMalloc(1 * sizeof(float));;
x1050[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1050, in_desc, x804, grad_out_desc, x837,
    conv_desc, algo, ws_data, ws_size,
    x1050, grad_filt_desc, x280));
};
float* x1053 = (float*)myMalloc(1 * sizeof(float));;
x1053[0] = 1.0f;

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
    cudnnHandle, x1053, grad_out_desc, x837,
    x1053, grad_bias_desc, x265));
};
float* x1056 = (float*)myMalloc(1 * sizeof(float));;
x1056[0] = 1.0f;
float* x1058 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1056, x_desc, x823, x_desc, x825, x_desc, x807,
    x1056, x_desc, x818));
};
// conv2D back-propagate
float* x1061 = (float*)myMalloc(1 * sizeof(float));;
x1061[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1061, filt_desc, x242, grad_out_desc, x818,
    conv_desc, algo, ws_data, ws_size,
    x1061, grad_in_desc, x806));
};
float* x1064 = (float*)myMalloc(1 * sizeof(float));;
x1064[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1064, in_desc, x804, grad_out_desc, x818,
    conv_desc, algo, ws_data, ws_size,
    x1064, grad_filt_desc, x301));
};
float* x1067 = (float*)myMalloc(1 * sizeof(float));;
x1067[0] = 1.0f;

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
    cudnnHandle, x1067, grad_out_desc, x818,
    x1067, grad_bias_desc, x292));
};
float* x1070 = (float*)myMalloc(1 * sizeof(float));;
x1070[0] = 1.0f;
float* x1072 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1070, x_desc, x804, x_desc, x806, x_desc, x788,
    x1070, x_desc, x799));
};
// conv2D back-propagate
float* x1075 = (float*)myMalloc(1 * sizeof(float));;
x1075[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1075, filt_desc, x197, grad_out_desc, x799,
    conv_desc, algo, ws_data, ws_size,
    x1075, grad_in_desc, x787));
};
float* x1078 = (float*)myMalloc(1 * sizeof(float));;
x1078[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1078, in_desc, x775, grad_out_desc, x799,
    conv_desc, algo, ws_data, ws_size,
    x1078, grad_filt_desc, x286));
};
float* x1081 = (float*)myMalloc(1 * sizeof(float));;
x1081[0] = 1.0f;

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
    cudnnHandle, x1081, grad_out_desc, x799,
    x1081, grad_bias_desc, x261));
};
{
dim3 grid(400, 2);
concat2D_1D_greg_grad<<<grid, 512>>>(x755, 192, 786432, x774, 192, 786432, x787, 1, 64, 384, 8, 8, 24576, 64, 8, 1);
};
float* x1085 = (float*)myMalloc(1 * sizeof(float));;
x1085[0] = 1.0f;
float* x1087 = (float*)myGpuMalloc(786432 * sizeof(float));

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
    x1085, x_desc, x772, x_desc, x774, x_desc, x756,
    x1085, x_desc, x767));
};
// conv2D back-propagate
float* x1090 = (float*)myMalloc(1 * sizeof(float));;
x1090[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1090, filt_desc, x149, grad_out_desc, x767,
    conv_desc, algo, ws_data, ws_size,
    x1090, grad_in_desc, x736));
};
float* x1093 = (float*)myMalloc(1 * sizeof(float));;
x1093[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1093, in_desc, x734, grad_out_desc, x767,
    conv_desc, algo, ws_data, ws_size,
    x1093, grad_filt_desc, x270));
};
float* x1096 = (float*)myMalloc(1 * sizeof(float));;
x1096[0] = 1.0f;

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
    cudnnHandle, x1096, grad_out_desc, x767,
    x1096, grad_bias_desc, x296));
};
float* x1099 = (float*)myMalloc(1 * sizeof(float));;
x1099[0] = 1.0f;
float* x1101 = (float*)myGpuMalloc(786432 * sizeof(float));

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
    x1099, x_desc, x753, x_desc, x755, x_desc, x737,
    x1099, x_desc, x748));
};
// conv2D back-propagate
float* x1104 = (float*)myMalloc(1 * sizeof(float));;
x1104[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1104, filt_desc, x161, grad_out_desc, x748,
    conv_desc, algo, ws_data, ws_size,
    x1104, grad_in_desc, x736));
};
float* x1107 = (float*)myMalloc(1 * sizeof(float));;
x1107[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1107, in_desc, x734, grad_out_desc, x748,
    conv_desc, algo, ws_data, ws_size,
    x1107, grad_filt_desc, x274));
};
float* x1110 = (float*)myMalloc(1 * sizeof(float));;
x1110[0] = 1.0f;

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
    cudnnHandle, x1110, grad_out_desc, x748,
    x1110, grad_bias_desc, x284));
};
float* x1113 = (float*)myMalloc(1 * sizeof(float));;
x1113[0] = 1.0f;
float* x1115 = (float*)myGpuMalloc(196608 * sizeof(float));

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
    x1113, x_desc, x734, x_desc, x736, x_desc, x718,
    x1113, x_desc, x729));
};
// conv2D back-propagate
float* x1118 = (float*)myMalloc(1 * sizeof(float));;
x1118[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1118, filt_desc, x137, grad_out_desc, x729,
    conv_desc, algo, ws_data, ws_size,
    x1118, grad_in_desc, x717));
};
float* x1121 = (float*)myMalloc(1 * sizeof(float));;
x1121[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1121, in_desc, x705, grad_out_desc, x729,
    conv_desc, algo, ws_data, ws_size,
    x1121, grad_filt_desc, x266));
};
float* x1124 = (float*)myMalloc(1 * sizeof(float));;
x1124[0] = 1.0f;

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
    cudnnHandle, x1124, grad_out_desc, x729,
    x1124, grad_bias_desc, x254));
};
{
dim3 grid(400, 2);
concat2D_1D_greg_grad<<<grid, 512>>>(x685, 192, 786432, x704, 192, 786432, x717, 1, 64, 384, 8, 8, 24576, 64, 8, 1);
};
float* x1128 = (float*)myMalloc(1 * sizeof(float));;
x1128[0] = 1.0f;
float* x1130 = (float*)myGpuMalloc(786432 * sizeof(float));

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
    x1128, x_desc, x702, x_desc, x704, x_desc, x686,
    x1128, x_desc, x697));
};
// conv2D back-propagate
float* x1133 = (float*)myMalloc(1 * sizeof(float));;
x1133[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1133, filt_desc, x212, grad_out_desc, x697,
    conv_desc, algo, ws_data, ws_size,
    x1133, grad_in_desc, x666));
};
float* x1136 = (float*)myMalloc(1 * sizeof(float));;
x1136[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1136, in_desc, x664, grad_out_desc, x697,
    conv_desc, algo, ws_data, ws_size,
    x1136, grad_filt_desc, x291));
};
float* x1139 = (float*)myMalloc(1 * sizeof(float));;
x1139[0] = 1.0f;

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
    cudnnHandle, x1139, grad_out_desc, x697,
    x1139, grad_bias_desc, x281));
};
float* x1142 = (float*)myMalloc(1 * sizeof(float));;
x1142[0] = 1.0f;
float* x1144 = (float*)myGpuMalloc(786432 * sizeof(float));

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
    x1142, x_desc, x683, x_desc, x685, x_desc, x667,
    x1142, x_desc, x678));
};
// conv2D back-propagate
float* x1147 = (float*)myMalloc(1 * sizeof(float));;
x1147[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1147, filt_desc, x239, grad_out_desc, x678,
    conv_desc, algo, ws_data, ws_size,
    x1147, grad_in_desc, x666));
};
float* x1150 = (float*)myMalloc(1 * sizeof(float));;
x1150[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1150, in_desc, x664, grad_out_desc, x678,
    conv_desc, algo, ws_data, ws_size,
    x1150, grad_filt_desc, x300));
};
float* x1153 = (float*)myMalloc(1 * sizeof(float));;
x1153[0] = 1.0f;

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
    cudnnHandle, x1153, grad_out_desc, x678,
    x1153, grad_bias_desc, x298));
};
float* x1156 = (float*)myMalloc(1 * sizeof(float));;
x1156[0] = 1.0f;
float* x1158 = (float*)myGpuMalloc(196608 * sizeof(float));

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
    x1156, x_desc, x664, x_desc, x666, x_desc, x648,
    x1156, x_desc, x659));
};
// conv2D back-propagate
float* x1161 = (float*)myMalloc(1 * sizeof(float));;
x1161[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1161, filt_desc, x221, grad_out_desc, x659,
    conv_desc, algo, ws_data, ws_size,
    x1161, grad_in_desc, x647));
};
float* x1164 = (float*)myMalloc(1 * sizeof(float));;
x1164[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1164, in_desc, x635, grad_out_desc, x659,
    conv_desc, algo, ws_data, ws_size,
    x1164, grad_filt_desc, x294));
};
float* x1167 = (float*)myMalloc(1 * sizeof(float));;
x1167[0] = 1.0f;

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
    cudnnHandle, x1167, grad_out_desc, x659,
    x1167, grad_bias_desc, x304));
};
{
dim3 grid(400, 2);
concat2D_1D_greg_grad<<<grid, 512>>>(x615, 128, 524288, x634, 128, 524288, x647, 1, 64, 256, 8, 8, 16384, 64, 8, 1);
};
float* x1171 = (float*)myMalloc(1 * sizeof(float));;
x1171[0] = 1.0f;
float* x1173 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1171, x_desc, x632, x_desc, x634, x_desc, x616,
    x1171, x_desc, x627));
};
// conv2D back-propagate
float* x1176 = (float*)myMalloc(1 * sizeof(float));;
x1176[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1176, filt_desc, x203, grad_out_desc, x627,
    conv_desc, algo, ws_data, ws_size,
    x1176, grad_in_desc, x596));
};
float* x1179 = (float*)myMalloc(1 * sizeof(float));;
x1179[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1179, in_desc, x594, grad_out_desc, x627,
    conv_desc, algo, ws_data, ws_size,
    x1179, grad_filt_desc, x288));
};
float* x1182 = (float*)myMalloc(1 * sizeof(float));;
x1182[0] = 1.0f;

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
    cudnnHandle, x1182, grad_out_desc, x627,
    x1182, grad_bias_desc, x268));
};
float* x1185 = (float*)myMalloc(1 * sizeof(float));;
x1185[0] = 1.0f;
float* x1187 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1185, x_desc, x613, x_desc, x615, x_desc, x597,
    x1185, x_desc, x608));
};
// conv2D back-propagate
float* x1190 = (float*)myMalloc(1 * sizeof(float));;
x1190[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1190, filt_desc, x116, grad_out_desc, x608,
    conv_desc, algo, ws_data, ws_size,
    x1190, grad_in_desc, x596));
};
float* x1193 = (float*)myMalloc(1 * sizeof(float));;
x1193[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1193, in_desc, x594, grad_out_desc, x608,
    conv_desc, algo, ws_data, ws_size,
    x1193, grad_filt_desc, x259));
};
float* x1196 = (float*)myMalloc(1 * sizeof(float));;
x1196[0] = 1.0f;

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
    cudnnHandle, x1196, grad_out_desc, x608,
    x1196, grad_bias_desc, x273));
};
float* x1199 = (float*)myMalloc(1 * sizeof(float));;
x1199[0] = 1.0f;
float* x1201 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x1199, x_desc, x594, x_desc, x596, x_desc, x578,
    x1199, x_desc, x589));
};
// conv2D back-propagate
float* x1204 = (float*)myMalloc(1 * sizeof(float));;
x1204[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1204, filt_desc, x176, grad_out_desc, x589,
    conv_desc, algo, ws_data, ws_size,
    x1204, grad_in_desc, x577));
};
float* x1207 = (float*)myMalloc(1 * sizeof(float));;
x1207[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1207, in_desc, x575, grad_out_desc, x589,
    conv_desc, algo, ws_data, ws_size,
    x1207, grad_filt_desc, x279));
};
float* x1210 = (float*)myMalloc(1 * sizeof(float));;
x1210[0] = 1.0f;

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
    cudnnHandle, x1210, grad_out_desc, x589,
    x1210, grad_bias_desc, x267));
};
float* x1213 = (float*)myMalloc(1 * sizeof(float));;
x1213[0] = 0.0f;
float* x1215 = (float*)myMalloc(1 * sizeof(float));;
x1215[0] = 1.0f;

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
    x1215, out_desc, x575, out_desc, x577, in_desc, x558  , x1213, in_desc, x570));
};
{
dim3 grid(400, 2);
concat2D_1D_greg_grad<<<grid, 512>>>(x538, 128, 2097152, x557, 128, 2097152, x570, 1, 64, 256, 16, 16, 65536, 256, 16, 1);
};
float* x1219 = (float*)myMalloc(1 * sizeof(float));;
x1219[0] = 1.0f;
float* x1221 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x1219, x_desc, x555, x_desc, x557, x_desc, x539,
    x1219, x_desc, x550));
};
// conv2D back-propagate
float* x1224 = (float*)myMalloc(1 * sizeof(float));;
x1224[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1224, filt_desc, x113, grad_out_desc, x550,
    conv_desc, algo, ws_data, ws_size,
    x1224, grad_in_desc, x519));
};
float* x1227 = (float*)myMalloc(1 * sizeof(float));;
x1227[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1227, in_desc, x517, grad_out_desc, x550,
    conv_desc, algo, ws_data, ws_size,
    x1227, grad_filt_desc, x258));
};
float* x1230 = (float*)myMalloc(1 * sizeof(float));;
x1230[0] = 1.0f;

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
    cudnnHandle, x1230, grad_out_desc, x550,
    x1230, grad_bias_desc, x293));
};
float* x1233 = (float*)myMalloc(1 * sizeof(float));;
x1233[0] = 1.0f;
float* x1235 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x1233, x_desc, x536, x_desc, x538, x_desc, x520,
    x1233, x_desc, x531));
};
// conv2D back-propagate
float* x1238 = (float*)myMalloc(1 * sizeof(float));;
x1238[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1238, filt_desc, x200, grad_out_desc, x531,
    conv_desc, algo, ws_data, ws_size,
    x1238, grad_in_desc, x519));
};
float* x1241 = (float*)myMalloc(1 * sizeof(float));;
x1241[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1241, in_desc, x517, grad_out_desc, x531,
    conv_desc, algo, ws_data, ws_size,
    x1241, grad_filt_desc, x287));
};
float* x1244 = (float*)myMalloc(1 * sizeof(float));;
x1244[0] = 1.0f;

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
    cudnnHandle, x1244, grad_out_desc, x531,
    x1244, grad_bias_desc, x297));
};
float* x1247 = (float*)myMalloc(1 * sizeof(float));;
x1247[0] = 1.0f;
float* x1249 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1247, x_desc, x517, x_desc, x519, x_desc, x501,
    x1247, x_desc, x512));
};
// conv2D back-propagate
float* x1252 = (float*)myMalloc(1 * sizeof(float));;
x1252[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1252, filt_desc, x125, grad_out_desc, x512,
    conv_desc, algo, ws_data, ws_size,
    x1252, grad_in_desc, x500));
};
float* x1255 = (float*)myMalloc(1 * sizeof(float));;
x1255[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1255, in_desc, x488, grad_out_desc, x512,
    conv_desc, algo, ws_data, ws_size,
    x1255, grad_filt_desc, x262));
};
float* x1258 = (float*)myMalloc(1 * sizeof(float));;
x1258[0] = 1.0f;

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
    cudnnHandle, x1258, grad_out_desc, x512,
    x1258, grad_bias_desc, x275));
};
{
dim3 grid(400, 2);
concat2D_1D_greg_grad<<<grid, 512>>>(x468, 64, 1048576, x487, 64, 1048576, x500, 1, 64, 128, 16, 16, 32768, 256, 16, 1);
};
float* x1262 = (float*)myMalloc(1 * sizeof(float));;
x1262[0] = 1.0f;
float* x1264 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1262, x_desc, x485, x_desc, x487, x_desc, x469,
    x1262, x_desc, x480));
};
// conv2D back-propagate
float* x1267 = (float*)myMalloc(1 * sizeof(float));;
x1267[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1267, filt_desc, x152, grad_out_desc, x480,
    conv_desc, algo, ws_data, ws_size,
    x1267, grad_in_desc, x449));
};
float* x1270 = (float*)myMalloc(1 * sizeof(float));;
x1270[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1270, in_desc, x447, grad_out_desc, x480,
    conv_desc, algo, ws_data, ws_size,
    x1270, grad_filt_desc, x271));
};
float* x1273 = (float*)myMalloc(1 * sizeof(float));;
x1273[0] = 1.0f;

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
    cudnnHandle, x1273, grad_out_desc, x480,
    x1273, grad_bias_desc, x289));
};
float* x1276 = (float*)myMalloc(1 * sizeof(float));;
x1276[0] = 1.0f;
float* x1278 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1276, x_desc, x466, x_desc, x468, x_desc, x450,
    x1276, x_desc, x461));
};
// conv2D back-propagate
float* x1281 = (float*)myMalloc(1 * sizeof(float));;
x1281[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1281, filt_desc, x128, grad_out_desc, x461,
    conv_desc, algo, ws_data, ws_size,
    x1281, grad_in_desc, x449));
};
float* x1284 = (float*)myMalloc(1 * sizeof(float));;
x1284[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1284, in_desc, x447, grad_out_desc, x461,
    conv_desc, algo, ws_data, ws_size,
    x1284, grad_filt_desc, x263));
};
float* x1287 = (float*)myMalloc(1 * sizeof(float));;
x1287[0] = 1.0f;

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
    cudnnHandle, x1287, grad_out_desc, x461,
    x1287, grad_bias_desc, x255));
};
float* x1290 = (float*)myMalloc(1 * sizeof(float));;
x1290[0] = 1.0f;
float* x1292 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1290, x_desc, x447, x_desc, x449, x_desc, x431,
    x1290, x_desc, x442));
};
// conv2D back-propagate
float* x1295 = (float*)myMalloc(1 * sizeof(float));;
x1295[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1295, filt_desc, x131, grad_out_desc, x442,
    conv_desc, algo, ws_data, ws_size,
    x1295, grad_in_desc, x430));
};
float* x1298 = (float*)myMalloc(1 * sizeof(float));;
x1298[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1298, in_desc, x418, grad_out_desc, x442,
    conv_desc, algo, ws_data, ws_size,
    x1298, grad_filt_desc, x264));
};
float* x1301 = (float*)myMalloc(1 * sizeof(float));;
x1301[0] = 1.0f;

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
    cudnnHandle, x1301, grad_out_desc, x442,
    x1301, grad_bias_desc, x277));
};
{
dim3 grid(400, 2);
concat2D_1D_greg_grad<<<grid, 512>>>(x398, 64, 1048576, x417, 64, 1048576, x430, 1, 64, 128, 16, 16, 32768, 256, 16, 1);
};
float* x1305 = (float*)myMalloc(1 * sizeof(float));;
x1305[0] = 1.0f;
float* x1307 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1305, x_desc, x415, x_desc, x417, x_desc, x399,
    x1305, x_desc, x410));
};
// conv2D back-propagate
float* x1310 = (float*)myMalloc(1 * sizeof(float));;
x1310[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1310, filt_desc, x236, grad_out_desc, x410,
    conv_desc, algo, ws_data, ws_size,
    x1310, grad_in_desc, x379));
};
float* x1313 = (float*)myMalloc(1 * sizeof(float));;
x1313[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1313, in_desc, x377, grad_out_desc, x410,
    conv_desc, algo, ws_data, ws_size,
    x1313, grad_filt_desc, x299));
};
float* x1316 = (float*)myMalloc(1 * sizeof(float));;
x1316[0] = 1.0f;

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
    cudnnHandle, x1316, grad_out_desc, x410,
    x1316, grad_bias_desc, x257));
};
float* x1319 = (float*)myMalloc(1 * sizeof(float));;
x1319[0] = 1.0f;
float* x1321 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1319, x_desc, x396, x_desc, x398, x_desc, x380,
    x1319, x_desc, x391));
};
// conv2D back-propagate
float* x1324 = (float*)myMalloc(1 * sizeof(float));;
x1324[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1324, filt_desc, x167, grad_out_desc, x391,
    conv_desc, algo, ws_data, ws_size,
    x1324, grad_in_desc, x379));
};
float* x1327 = (float*)myMalloc(1 * sizeof(float));;
x1327[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1327, in_desc, x377, grad_out_desc, x391,
    conv_desc, algo, ws_data, ws_size,
    x1327, grad_filt_desc, x276));
};
float* x1330 = (float*)myMalloc(1 * sizeof(float));;
x1330[0] = 1.0f;

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
    cudnnHandle, x1330, grad_out_desc, x391,
    x1330, grad_bias_desc, x283));
};
float* x1333 = (float*)myMalloc(1 * sizeof(float));;
x1333[0] = 1.0f;
float* x1335 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1333, x_desc, x377, x_desc, x379, x_desc, x361,
    x1333, x_desc, x372));
};
// conv2D back-propagate
float* x1338 = (float*)myMalloc(1 * sizeof(float));;
x1338[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x1338, filt_desc, x245, grad_out_desc, x372,
    conv_desc, algo, ws_data, ws_size,
    x1338, grad_in_desc, x360));
};
float* x1341 = (float*)myMalloc(1 * sizeof(float));;
x1341[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1341, in_desc, x358, grad_out_desc, x372,
    conv_desc, algo, ws_data, ws_size,
    x1341, grad_filt_desc, x302));
};
float* x1344 = (float*)myMalloc(1 * sizeof(float));;
x1344[0] = 1.0f;

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
    cudnnHandle, x1344, grad_out_desc, x372,
    x1344, grad_bias_desc, x260));
};
float* x1347 = (float*)myMalloc(1 * sizeof(float));;
x1347[0] = 0.0f;
float* x1349 = (float*)myMalloc(1 * sizeof(float));;
x1349[0] = 1.0f;

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
    x1349, out_desc, x358, out_desc, x360, in_desc, x351  , x1347, in_desc, x353));
};
float* x1352 = (float*)myMalloc(1 * sizeof(float));;
x1352[0] = 1.0f;
float* x1354 = (float*)myGpuMalloc(6291456 * sizeof(float));

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
    x1352, x_desc, x351, x_desc, x353, x_desc, x335,
    x1352, x_desc, x346));
};
// conv2D back-propagate
float* x1357 = (float*)myMalloc(1 * sizeof(float));;
x1357[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x1357, in_desc, x326, grad_out_desc, x346,
    conv_desc, algo, ws_data, ws_size,
    x1357, grad_filt_desc, x285));
};
float* x1360 = (float*)myMalloc(1 * sizeof(float));;
x1360[0] = 1.0f;

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
    cudnnHandle, x1360, grad_out_desc, x346,
    x1360, grad_bias_desc, x295));
};
float x1363 = x334[0];
x314 += x1363;
float* x1365 = (float*)myMalloc(1 * sizeof(float));;
x1365[0] = 1.0f;
float* x1367 = (float*)myMalloc(1 * sizeof(float));;
x1367[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,64,x1365,x98,512,x1367, x253, 512, x98,512));
arrayFill_greg<<<7, 512>>>(x253, 0.0f, 32768);
float* x1371 = (float*)myMalloc(1 * sizeof(float));;
x1371[0] = 1.0f;
float* x1373 = (float*)myMalloc(1 * sizeof(float));;
x1373[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,48,x1371,x101,1,x1373, x254, 1, x101,1));
arrayFill_greg<<<1, 512>>>(x254, 0.0f, 48);
float* x1377 = (float*)myMalloc(1 * sizeof(float));;
x1377[0] = 1.0f;
float* x1379 = (float*)myMalloc(1 * sizeof(float));;
x1379[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x1377,x104,1,x1379, x255, 1, x104,1));
arrayFill_greg<<<1, 512>>>(x255, 0.0f, 64);
float* x1383 = (float*)myMalloc(1 * sizeof(float));;
x1383[0] = 1.0f;
float* x1385 = (float*)myMalloc(1 * sizeof(float));;
x1385[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 8192,10,x1383,x107,8192,x1385, x256, 8192, x107,8192));
arrayFill_greg<<<17, 512>>>(x256, 0.0f, 81920);
float* x1389 = (float*)myMalloc(1 * sizeof(float));;
x1389[0] = 1.0f;
float* x1391 = (float*)myMalloc(1 * sizeof(float));;
x1391[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x1389,x110,1,x1391, x257, 1, x110,1));
arrayFill_greg<<<1, 512>>>(x257, 0.0f, 64);
float* x1395 = (float*)myMalloc(1 * sizeof(float));;
x1395[0] = 1.0f;
float* x1397 = (float*)myMalloc(1 * sizeof(float));;
x1397[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 288,128,x1395,x113,288,x1397, x258, 288, x113,288));
arrayFill_greg<<<8, 512>>>(x258, 0.0f, 36864);
float* x1401 = (float*)myMalloc(1 * sizeof(float));;
x1401[0] = 1.0f;
float* x1403 = (float*)myMalloc(1 * sizeof(float));;
x1403[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 32,128,x1401,x116,32,x1403, x259, 32, x116,32));
arrayFill_greg<<<1, 512>>>(x259, 0.0f, 4096);
float* x1407 = (float*)myMalloc(1 * sizeof(float));;
x1407[0] = 1.0f;
float* x1409 = (float*)myMalloc(1 * sizeof(float));;
x1409[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,16,x1407,x119,1,x1409, x260, 1, x119,1));
arrayFill_greg<<<1, 512>>>(x260, 0.0f, 16);
float* x1413 = (float*)myMalloc(1 * sizeof(float));;
x1413[0] = 1.0f;
float* x1415 = (float*)myMalloc(1 * sizeof(float));;
x1415[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x1413,x122,1,x1415, x261, 1, x122,1));
arrayFill_greg<<<1, 512>>>(x261, 0.0f, 64);
float* x1419 = (float*)myMalloc(1 * sizeof(float));;
x1419[0] = 1.0f;
float* x1421 = (float*)myMalloc(1 * sizeof(float));;
x1421[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,32,x1419,x125,128,x1421, x262, 128, x125,128));
arrayFill_greg<<<1, 512>>>(x262, 0.0f, 4096);
float* x1425 = (float*)myMalloc(1 * sizeof(float));;
x1425[0] = 1.0f;
float* x1427 = (float*)myMalloc(1 * sizeof(float));;
x1427[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 16,64,x1425,x128,16,x1427, x263, 16, x128,16));
arrayFill_greg<<<1, 512>>>(x263, 0.0f, 1024);
float* x1431 = (float*)myMalloc(1 * sizeof(float));;
x1431[0] = 1.0f;
float* x1433 = (float*)myMalloc(1 * sizeof(float));;
x1433[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,16,x1431,x131,128,x1433, x264, 128, x131,128));
arrayFill_greg<<<1, 512>>>(x264, 0.0f, 2048);
float* x1437 = (float*)myMalloc(1 * sizeof(float));;
x1437[0] = 1.0f;
float* x1439 = (float*)myMalloc(1 * sizeof(float));;
x1439[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x1437,x134,1,x1439, x265, 1, x134,1));
arrayFill_greg<<<1, 512>>>(x265, 0.0f, 256);
float* x1443 = (float*)myMalloc(1 * sizeof(float));;
x1443[0] = 1.0f;
float* x1445 = (float*)myMalloc(1 * sizeof(float));;
x1445[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 384,48,x1443,x137,384,x1445, x266, 384, x137,384));
arrayFill_greg<<<4, 512>>>(x266, 0.0f, 18432);
float* x1449 = (float*)myMalloc(1 * sizeof(float));;
x1449[0] = 1.0f;
float* x1451 = (float*)myMalloc(1 * sizeof(float));;
x1451[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1449,x140,1,x1451, x267, 1, x140,1));
arrayFill_greg<<<1, 512>>>(x267, 0.0f, 32);
float* x1455 = (float*)myMalloc(1 * sizeof(float));;
x1455[0] = 1.0f;
float* x1457 = (float*)myMalloc(1 * sizeof(float));;
x1457[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x1455,x143,1,x1457, x268, 1, x143,1));
arrayFill_greg<<<1, 512>>>(x268, 0.0f, 128);
float* x1461 = (float*)myMalloc(1 * sizeof(float));;
x1461[0] = 1.0f;
float* x1463 = (float*)myMalloc(1 * sizeof(float));;
x1463[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x1461,x146,1,x1463, x269, 1, x146,1));
arrayFill_greg<<<1, 512>>>(x269, 0.0f, 256);
float* x1467 = (float*)myMalloc(1 * sizeof(float));;
x1467[0] = 1.0f;
float* x1469 = (float*)myMalloc(1 * sizeof(float));;
x1469[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 432,192,x1467,x149,432,x1469, x270, 432, x149,432));
arrayFill_greg<<<17, 512>>>(x270, 0.0f, 82944);
float* x1473 = (float*)myMalloc(1 * sizeof(float));;
x1473[0] = 1.0f;
float* x1475 = (float*)myMalloc(1 * sizeof(float));;
x1475[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 144,64,x1473,x152,144,x1475, x271, 144, x152,144));
arrayFill_greg<<<2, 512>>>(x271, 0.0f, 9216);
float* x1479 = (float*)myMalloc(1 * sizeof(float));;
x1479[0] = 1.0f;
float* x1481 = (float*)myMalloc(1 * sizeof(float));;
x1481[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x1479,x155,1,x1481, x272, 1, x155,1));
arrayFill_greg<<<1, 512>>>(x272, 0.0f, 64);
float* x1485 = (float*)myMalloc(1 * sizeof(float));;
x1485[0] = 1.0f;
float* x1487 = (float*)myMalloc(1 * sizeof(float));;
x1487[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x1485,x158,1,x1487, x273, 1, x158,1));
arrayFill_greg<<<1, 512>>>(x273, 0.0f, 128);
float* x1491 = (float*)myMalloc(1 * sizeof(float));;
x1491[0] = 1.0f;
float* x1493 = (float*)myMalloc(1 * sizeof(float));;
x1493[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 48,192,x1491,x161,48,x1493, x274, 48, x161,48));
arrayFill_greg<<<2, 512>>>(x274, 0.0f, 9216);
float* x1497 = (float*)myMalloc(1 * sizeof(float));;
x1497[0] = 1.0f;
float* x1499 = (float*)myMalloc(1 * sizeof(float));;
x1499[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1497,x164,1,x1499, x275, 1, x164,1));
arrayFill_greg<<<1, 512>>>(x275, 0.0f, 32);
float* x1503 = (float*)myMalloc(1 * sizeof(float));;
x1503[0] = 1.0f;
float* x1505 = (float*)myMalloc(1 * sizeof(float));;
x1505[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 16,64,x1503,x167,16,x1505, x276, 16, x167,16));
arrayFill_greg<<<1, 512>>>(x276, 0.0f, 1024);
float* x1509 = (float*)myMalloc(1 * sizeof(float));;
x1509[0] = 1.0f;
float* x1511 = (float*)myMalloc(1 * sizeof(float));;
x1511[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,16,x1509,x170,1,x1511, x277, 1, x170,1));
arrayFill_greg<<<1, 512>>>(x277, 0.0f, 16);
float* x1515 = (float*)myMalloc(1 * sizeof(float));;
x1515[0] = 1.0f;
float* x1517 = (float*)myMalloc(1 * sizeof(float));;
x1517[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x1515,x173,1,x1517, x278, 1, x173,1));
arrayFill_greg<<<1, 512>>>(x278, 0.0f, 256);
float* x1521 = (float*)myMalloc(1 * sizeof(float));;
x1521[0] = 1.0f;
float* x1523 = (float*)myMalloc(1 * sizeof(float));;
x1523[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,32,x1521,x176,256,x1523, x279, 256, x176,256));
arrayFill_greg<<<2, 512>>>(x279, 0.0f, 8192);
float* x1527 = (float*)myMalloc(1 * sizeof(float));;
x1527[0] = 1.0f;
float* x1529 = (float*)myMalloc(1 * sizeof(float));;
x1529[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 576,256,x1527,x179,576,x1529, x280, 576, x179,576));
arrayFill_greg<<<29, 512>>>(x280, 0.0f, 147456);
float* x1533 = (float*)myMalloc(1 * sizeof(float));;
x1533[0] = 1.0f;
float* x1535 = (float*)myMalloc(1 * sizeof(float));;
x1535[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,192,x1533,x182,1,x1535, x281, 1, x182,1));
arrayFill_greg<<<1, 512>>>(x281, 0.0f, 192);
float* x1539 = (float*)myMalloc(1 * sizeof(float));;
x1539[0] = 1.0f;
float* x1541 = (float*)myMalloc(1 * sizeof(float));;
x1541[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 576,256,x1539,x185,576,x1541, x282, 576, x185,576));
arrayFill_greg<<<29, 512>>>(x282, 0.0f, 147456);
float* x1545 = (float*)myMalloc(1 * sizeof(float));;
x1545[0] = 1.0f;
float* x1547 = (float*)myMalloc(1 * sizeof(float));;
x1547[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x1545,x188,1,x1547, x283, 1, x188,1));
arrayFill_greg<<<1, 512>>>(x283, 0.0f, 64);
float* x1551 = (float*)myMalloc(1 * sizeof(float));;
x1551[0] = 1.0f;
float* x1553 = (float*)myMalloc(1 * sizeof(float));;
x1553[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,192,x1551,x191,1,x1553, x284, 1, x191,1));
arrayFill_greg<<<1, 512>>>(x284, 0.0f, 192);
float* x1557 = (float*)myMalloc(1 * sizeof(float));;
x1557[0] = 1.0f;
float* x1559 = (float*)myMalloc(1 * sizeof(float));;
x1559[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 27,96,x1557,x194,27,x1559, x285, 27, x194,27));
arrayFill_greg<<<1, 512>>>(x285, 0.0f, 2592);
float* x1563 = (float*)myMalloc(1 * sizeof(float));;
x1563[0] = 1.0f;
float* x1565 = (float*)myMalloc(1 * sizeof(float));;
x1565[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 384,64,x1563,x197,384,x1565, x286, 384, x197,384));
arrayFill_greg<<<5, 512>>>(x286, 0.0f, 24576);
float* x1569 = (float*)myMalloc(1 * sizeof(float));;
x1569[0] = 1.0f;
float* x1571 = (float*)myMalloc(1 * sizeof(float));;
x1571[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 32,128,x1569,x200,32,x1571, x287, 32, x200,32));
arrayFill_greg<<<1, 512>>>(x287, 0.0f, 4096);
float* x1575 = (float*)myMalloc(1 * sizeof(float));;
x1575[0] = 1.0f;
float* x1577 = (float*)myMalloc(1 * sizeof(float));;
x1577[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 288,128,x1575,x203,288,x1577, x288, 288, x203,288));
arrayFill_greg<<<8, 512>>>(x288, 0.0f, 36864);
float* x1581 = (float*)myMalloc(1 * sizeof(float));;
x1581[0] = 1.0f;
float* x1583 = (float*)myMalloc(1 * sizeof(float));;
x1583[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x1581,x206,1,x1583, x289, 1, x206,1));
arrayFill_greg<<<1, 512>>>(x289, 0.0f, 64);
float* x1587 = (float*)myMalloc(1 * sizeof(float));;
x1587[0] = 1.0f;
float* x1589 = (float*)myMalloc(1 * sizeof(float));;
x1589[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x1587,x209,64,x1589, x290, 64, x209,64));
arrayFill_greg<<<4, 512>>>(x290, 0.0f, 16384);
float* x1593 = (float*)myMalloc(1 * sizeof(float));;
x1593[0] = 1.0f;
float* x1595 = (float*)myMalloc(1 * sizeof(float));;
x1595[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 432,192,x1593,x212,432,x1595, x291, 432, x212,432));
arrayFill_greg<<<17, 512>>>(x291, 0.0f, 82944);
float* x1599 = (float*)myMalloc(1 * sizeof(float));;
x1599[0] = 1.0f;
float* x1601 = (float*)myMalloc(1 * sizeof(float));;
x1601[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x1599,x215,1,x1601, x292, 1, x215,1));
arrayFill_greg<<<1, 512>>>(x292, 0.0f, 256);
float* x1605 = (float*)myMalloc(1 * sizeof(float));;
x1605[0] = 1.0f;
float* x1607 = (float*)myMalloc(1 * sizeof(float));;
x1607[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x1605,x218,1,x1607, x293, 1, x218,1));
arrayFill_greg<<<1, 512>>>(x293, 0.0f, 128);
float* x1611 = (float*)myMalloc(1 * sizeof(float));;
x1611[0] = 1.0f;
float* x1613 = (float*)myMalloc(1 * sizeof(float));;
x1613[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,48,x1611,x221,256,x1613, x294, 256, x221,256));
arrayFill_greg<<<3, 512>>>(x294, 0.0f, 12288);
float* x1617 = (float*)myMalloc(1 * sizeof(float));;
x1617[0] = 1.0f;
float* x1619 = (float*)myMalloc(1 * sizeof(float));;
x1619[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,96,x1617,x224,1,x1619, x295, 1, x224,1));
arrayFill_greg<<<1, 512>>>(x295, 0.0f, 96);
float* x1623 = (float*)myMalloc(1 * sizeof(float));;
x1623[0] = 1.0f;
float* x1625 = (float*)myMalloc(1 * sizeof(float));;
x1625[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,192,x1623,x227,1,x1625, x296, 1, x227,1));
arrayFill_greg<<<1, 512>>>(x296, 0.0f, 192);
float* x1629 = (float*)myMalloc(1 * sizeof(float));;
x1629[0] = 1.0f;
float* x1631 = (float*)myMalloc(1 * sizeof(float));;
x1631[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x1629,x230,1,x1631, x297, 1, x230,1));
arrayFill_greg<<<1, 512>>>(x297, 0.0f, 128);
float* x1635 = (float*)myMalloc(1 * sizeof(float));;
x1635[0] = 1.0f;
float* x1637 = (float*)myMalloc(1 * sizeof(float));;
x1637[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,192,x1635,x233,1,x1637, x298, 1, x233,1));
arrayFill_greg<<<1, 512>>>(x298, 0.0f, 192);
float* x1641 = (float*)myMalloc(1 * sizeof(float));;
x1641[0] = 1.0f;
float* x1643 = (float*)myMalloc(1 * sizeof(float));;
x1643[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 144,64,x1641,x236,144,x1643, x299, 144, x236,144));
arrayFill_greg<<<2, 512>>>(x299, 0.0f, 9216);
float* x1647 = (float*)myMalloc(1 * sizeof(float));;
x1647[0] = 1.0f;
float* x1649 = (float*)myMalloc(1 * sizeof(float));;
x1649[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 48,192,x1647,x239,48,x1649, x300, 48, x239,48));
arrayFill_greg<<<2, 512>>>(x300, 0.0f, 9216);
float* x1653 = (float*)myMalloc(1 * sizeof(float));;
x1653[0] = 1.0f;
float* x1655 = (float*)myMalloc(1 * sizeof(float));;
x1655[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x1653,x242,64,x1655, x301, 64, x242,64));
arrayFill_greg<<<4, 512>>>(x301, 0.0f, 16384);
float* x1659 = (float*)myMalloc(1 * sizeof(float));;
x1659[0] = 1.0f;
float* x1661 = (float*)myMalloc(1 * sizeof(float));;
x1661[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 96,16,x1659,x245,96,x1661, x302, 96, x245,96));
arrayFill_greg<<<1, 512>>>(x302, 0.0f, 1536);
float* x1665 = (float*)myMalloc(1 * sizeof(float));;
x1665[0] = 1.0f;
float* x1667 = (float*)myMalloc(1 * sizeof(float));;
x1667[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,10,x1665,x248,1,x1667, x303, 1, x248,1));
arrayFill_greg<<<1, 512>>>(x303, 0.0f, 10);
float* x1671 = (float*)myMalloc(1 * sizeof(float));;
x1671[0] = 1.0f;
float* x1673 = (float*)myMalloc(1 * sizeof(float));;
x1673[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,48,x1671,x251,1,x1673, x304, 1, x251,1));
arrayFill_greg<<<1, 512>>>(x304, 0.0f, 48);
int32_t x1677 = x320 + 1;
int32_t x1679 = x1677 % x1678;
bool x1680 = x1679 == 0;
if (x1680) {
float x1685 = x314;
double x1681 = (double)x321;
double x1682 = 100.0 * x1681;
double x1684 = x1682 / x1683;
float x1686 = (float)x320;
float x1687 = x1685 / x1686;
printf("Train epoch %d: [%d/%d (%.0f%%)] Average Loss: %.6f\n",x310,x321,x12,x1684,x1687);
fflush(stdout);
} else {
}
int64_t x1692 = (long)mallocAddr;
int64_t x1693 = x1692 - x306;
memset((void*)x306, 0, x1693);
mallocAddr = (void*)x306;
int64_t x1696 = (long)gpuMallocAddr;
int64_t x1697 = x1696 - x307;
cudaMemset((void*)x307, 0, x1697);
gpuMallocAddr = (void*)x307;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x1704 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x1705 = x1704 / 1000LL;
int64_t x1707 = x1704 / x1706;
printf("Training completed in %ldms (%ld us/images)\n",x1705,x1707);
float x1709 = x314;
float x1711 = x1709 / x1710;
double x1712 = (double)x1711;
x305[x310] = x1712;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x1718 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
int64_t x1723 = (long)fopen(x0, "w");
fprintf((FILE *)x1723, "unit: %s\n", "1 epoch");
for(int x1725=0; x1725 < 4; x1725++) {
double x1726 = x305[x1725];
fprintf((FILE *)x1723, "%lf\n", x1726);

}
float x1719 = (float)x1718;
float x1720 = x1719 / 1000000.0f;
float x1721 = x1720 - x40;
float x1722 = x1721 / 4.0f;
fprintf((FILE *)x1723, "run time: %lf %lf\n", x40, x1722);
fclose((FILE*)x1723);
// Backend cleanup.
CUBLAS_CALL(cublasDestroy(cublasHandle));
CUDA_CALL(cudaFree(gpuMallocBase));
      
CUDNN_CALL(cudnnDestroy(cudnnHandle));
}
/*****************************************
  End of C Generated Code                  
*******************************************/


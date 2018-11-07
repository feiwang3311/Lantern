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
int64_t x253 = (long)mallocAddr;
int64_t x254 = (long)gpuMallocAddr;
// inferencing loop starts here
int32_t x262 = x12 / 64;
int64_t x837 = (int64_t)x12;
for(int x257=0; x257 < 4; x257++) {
struct timeval begin_1, end_1, diff_1;
int32_t x259 = x257 + 1;
printf("Start inferencing epoch %d\n",x259);
gettimeofday(&begin_1, NULL);
for(int x264=0; x264 < x262; x264++) {
int32_t x265 = x264 * 64;
int32_t x266 = x265 * 3072;
float* x267 = x14+x266;
int* x268 = x15+x265;
// Tensor 'toGPU' invocation.
float* x270 = (float*)myGpuMalloc(196608 * sizeof(float));
CUDA_CALL(cudaMemcpy(x270, x267, 196608 * sizeof(float), cudaMemcpyHostToDevice));
float* x272 = (float*)myGpuMalloc(6291456 * sizeof(float));
float* x273 = (float*)myMalloc(1 * sizeof(float));;
x273[0] = 0.0f;
float* x275 = (float*)myMalloc(1 * sizeof(float));;
x275[0] = 1.0f;

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
    x275, in_desc, x270, filt_desc, x194,
    conv_desc, algo, ws_data, ws_size,
    x273, out_desc, x272));
};
float* x278 = (float*)myMalloc(1 * sizeof(float));;
x278[0] = 1.0f;
float* x280 = (float*)myMalloc(1 * sizeof(float));;
x280[0] = 1.0f;

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
    cudnnHandle, x278, bias_desc, x224, x280, out_desc, x272));
};
float* x283 = (float*)myMalloc(1 * sizeof(float));;
x283[0] = 0.0f;
float* x285 = (float*)myMalloc(1 * sizeof(float));;
x285[0] = 1.0f;
float* x287 = (float*)myGpuMalloc(6291456 * sizeof(float));

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
    x285, x_desc, x272, x283, x_desc, x287));
};
float* x289 = (float*)myMalloc(1 * sizeof(float));;
x289[0] = 0.0f;
float* x291 = (float*)myMalloc(1 * sizeof(float));;
x291[0] = 1.0f;
float* x293 = (float*)myGpuMalloc(1572864 * sizeof(float));

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
    x291, in_desc, x287, x289, out_desc, x293));
};
float* x295 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x296 = (float*)myMalloc(1 * sizeof(float));;
x296[0] = 0.0f;
float* x298 = (float*)myMalloc(1 * sizeof(float));;
x298[0] = 1.0f;

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
    x298, in_desc, x293, filt_desc, x245,
    conv_desc, algo, ws_data, ws_size,
    x296, out_desc, x295));
};
float* x301 = (float*)myMalloc(1 * sizeof(float));;
x301[0] = 1.0f;
float* x303 = (float*)myMalloc(1 * sizeof(float));;
x303[0] = 1.0f;

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
    cudnnHandle, x301, bias_desc, x119, x303, out_desc, x295));
};
float* x306 = (float*)myMalloc(1 * sizeof(float));;
x306[0] = 0.0f;
float* x308 = (float*)myMalloc(1 * sizeof(float));;
x308[0] = 1.0f;
float* x310 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x308, x_desc, x295, x306, x_desc, x310));
};
float* x312 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x313 = (float*)myMalloc(1 * sizeof(float));;
x313[0] = 0.0f;
float* x315 = (float*)myMalloc(1 * sizeof(float));;
x315[0] = 1.0f;

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
    x315, in_desc, x310, filt_desc, x167,
    conv_desc, algo, ws_data, ws_size,
    x313, out_desc, x312));
};
float* x318 = (float*)myMalloc(1 * sizeof(float));;
x318[0] = 1.0f;
float* x320 = (float*)myMalloc(1 * sizeof(float));;
x320[0] = 1.0f;

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
    cudnnHandle, x318, bias_desc, x188, x320, out_desc, x312));
};
float* x323 = (float*)myMalloc(1 * sizeof(float));;
x323[0] = 0.0f;
float* x325 = (float*)myMalloc(1 * sizeof(float));;
x325[0] = 1.0f;
float* x327 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x325, x_desc, x312, x323, x_desc, x327));
};
float* x329 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x330 = (float*)myMalloc(1 * sizeof(float));;
x330[0] = 0.0f;
float* x332 = (float*)myMalloc(1 * sizeof(float));;
x332[0] = 1.0f;

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
    x332, in_desc, x310, filt_desc, x236,
    conv_desc, algo, ws_data, ws_size,
    x330, out_desc, x329));
};
float* x335 = (float*)myMalloc(1 * sizeof(float));;
x335[0] = 1.0f;
float* x337 = (float*)myMalloc(1 * sizeof(float));;
x337[0] = 1.0f;

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
    cudnnHandle, x335, bias_desc, x110, x337, out_desc, x329));
};
float* x340 = (float*)myMalloc(1 * sizeof(float));;
x340[0] = 0.0f;
float* x342 = (float*)myMalloc(1 * sizeof(float));;
x342[0] = 1.0f;
float* x344 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x342, x_desc, x329, x340, x_desc, x344));
};
float* x346 = (float*)myGpuMalloc(2097152 * sizeof(float));
int* x347 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x347[0] = 64;
x347[1] = 128;
x347[2] = 16;
x347[3] = 16;
int* x352 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x352[0] = 32768;
x352[1] = 256;
x352[2] = 16;
x352[3] = 1;
{
dim3 grid(400, 2);
concat2D_1D_greg<<<grid, 512>>>(x327, 64, 1048576, x344, 64, 1048576, x346, 1, 64, 128, 16, 16, 32768, 256, 16, 1);
};
float* x358 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x359 = (float*)myMalloc(1 * sizeof(float));;
x359[0] = 0.0f;
float* x361 = (float*)myMalloc(1 * sizeof(float));;
x361[0] = 1.0f;

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
    x361, in_desc, x346, filt_desc, x131,
    conv_desc, algo, ws_data, ws_size,
    x359, out_desc, x358));
};
float* x364 = (float*)myMalloc(1 * sizeof(float));;
x364[0] = 1.0f;
float* x366 = (float*)myMalloc(1 * sizeof(float));;
x366[0] = 1.0f;

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
    cudnnHandle, x364, bias_desc, x170, x366, out_desc, x358));
};
float* x369 = (float*)myMalloc(1 * sizeof(float));;
x369[0] = 0.0f;
float* x371 = (float*)myMalloc(1 * sizeof(float));;
x371[0] = 1.0f;
float* x373 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x371, x_desc, x358, x369, x_desc, x373));
};
float* x375 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x376 = (float*)myMalloc(1 * sizeof(float));;
x376[0] = 0.0f;
float* x378 = (float*)myMalloc(1 * sizeof(float));;
x378[0] = 1.0f;

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
    x378, in_desc, x373, filt_desc, x128,
    conv_desc, algo, ws_data, ws_size,
    x376, out_desc, x375));
};
float* x381 = (float*)myMalloc(1 * sizeof(float));;
x381[0] = 1.0f;
float* x383 = (float*)myMalloc(1 * sizeof(float));;
x383[0] = 1.0f;

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
    cudnnHandle, x381, bias_desc, x104, x383, out_desc, x375));
};
float* x386 = (float*)myMalloc(1 * sizeof(float));;
x386[0] = 0.0f;
float* x388 = (float*)myMalloc(1 * sizeof(float));;
x388[0] = 1.0f;
float* x390 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x388, x_desc, x375, x386, x_desc, x390));
};
float* x392 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x393 = (float*)myMalloc(1 * sizeof(float));;
x393[0] = 0.0f;
float* x395 = (float*)myMalloc(1 * sizeof(float));;
x395[0] = 1.0f;

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
    x395, in_desc, x373, filt_desc, x152,
    conv_desc, algo, ws_data, ws_size,
    x393, out_desc, x392));
};
float* x398 = (float*)myMalloc(1 * sizeof(float));;
x398[0] = 1.0f;
float* x400 = (float*)myMalloc(1 * sizeof(float));;
x400[0] = 1.0f;

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
    cudnnHandle, x398, bias_desc, x206, x400, out_desc, x392));
};
float* x403 = (float*)myMalloc(1 * sizeof(float));;
x403[0] = 0.0f;
float* x405 = (float*)myMalloc(1 * sizeof(float));;
x405[0] = 1.0f;
float* x407 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x405, x_desc, x392, x403, x_desc, x407));
};
float* x409 = (float*)myGpuMalloc(2097152 * sizeof(float));
int* x410 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x410[0] = 64;
x410[1] = 128;
x410[2] = 16;
x410[3] = 16;
int* x415 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x415[0] = 32768;
x415[1] = 256;
x415[2] = 16;
x415[3] = 1;
{
dim3 grid(400, 2);
concat2D_1D_greg<<<grid, 512>>>(x390, 64, 1048576, x407, 64, 1048576, x409, 1, 64, 128, 16, 16, 32768, 256, 16, 1);
};
float* x421 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x422 = (float*)myMalloc(1 * sizeof(float));;
x422[0] = 0.0f;
float* x424 = (float*)myMalloc(1 * sizeof(float));;
x424[0] = 1.0f;

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
    x424, in_desc, x409, filt_desc, x125,
    conv_desc, algo, ws_data, ws_size,
    x422, out_desc, x421));
};
float* x427 = (float*)myMalloc(1 * sizeof(float));;
x427[0] = 1.0f;
float* x429 = (float*)myMalloc(1 * sizeof(float));;
x429[0] = 1.0f;

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
    cudnnHandle, x427, bias_desc, x164, x429, out_desc, x421));
};
float* x432 = (float*)myMalloc(1 * sizeof(float));;
x432[0] = 0.0f;
float* x434 = (float*)myMalloc(1 * sizeof(float));;
x434[0] = 1.0f;
float* x436 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x434, x_desc, x421, x432, x_desc, x436));
};
float* x438 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x439 = (float*)myMalloc(1 * sizeof(float));;
x439[0] = 0.0f;
float* x441 = (float*)myMalloc(1 * sizeof(float));;
x441[0] = 1.0f;

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
    x441, in_desc, x436, filt_desc, x200,
    conv_desc, algo, ws_data, ws_size,
    x439, out_desc, x438));
};
float* x444 = (float*)myMalloc(1 * sizeof(float));;
x444[0] = 1.0f;
float* x446 = (float*)myMalloc(1 * sizeof(float));;
x446[0] = 1.0f;

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
    cudnnHandle, x444, bias_desc, x230, x446, out_desc, x438));
};
float* x449 = (float*)myMalloc(1 * sizeof(float));;
x449[0] = 0.0f;
float* x451 = (float*)myMalloc(1 * sizeof(float));;
x451[0] = 1.0f;
float* x453 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x451, x_desc, x438, x449, x_desc, x453));
};
float* x455 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x456 = (float*)myMalloc(1 * sizeof(float));;
x456[0] = 0.0f;
float* x458 = (float*)myMalloc(1 * sizeof(float));;
x458[0] = 1.0f;

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
    x458, in_desc, x436, filt_desc, x113,
    conv_desc, algo, ws_data, ws_size,
    x456, out_desc, x455));
};
float* x461 = (float*)myMalloc(1 * sizeof(float));;
x461[0] = 1.0f;
float* x463 = (float*)myMalloc(1 * sizeof(float));;
x463[0] = 1.0f;

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
    cudnnHandle, x461, bias_desc, x218, x463, out_desc, x455));
};
float* x466 = (float*)myMalloc(1 * sizeof(float));;
x466[0] = 0.0f;
float* x468 = (float*)myMalloc(1 * sizeof(float));;
x468[0] = 1.0f;
float* x470 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x468, x_desc, x455, x466, x_desc, x470));
};
float* x472 = (float*)myGpuMalloc(4194304 * sizeof(float));
int* x473 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x473[0] = 64;
x473[1] = 256;
x473[2] = 16;
x473[3] = 16;
int* x478 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x478[0] = 65536;
x478[1] = 256;
x478[2] = 16;
x478[3] = 1;
{
dim3 grid(400, 2);
concat2D_1D_greg<<<grid, 512>>>(x453, 128, 2097152, x470, 128, 2097152, x472, 1, 64, 256, 16, 16, 65536, 256, 16, 1);
};
float* x484 = (float*)myMalloc(1 * sizeof(float));;
x484[0] = 0.0f;
float* x486 = (float*)myMalloc(1 * sizeof(float));;
x486[0] = 1.0f;
float* x488 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x486, in_desc, x472, x484, out_desc, x488));
};
float* x490 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x491 = (float*)myMalloc(1 * sizeof(float));;
x491[0] = 0.0f;
float* x493 = (float*)myMalloc(1 * sizeof(float));;
x493[0] = 1.0f;

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
    x493, in_desc, x488, filt_desc, x176,
    conv_desc, algo, ws_data, ws_size,
    x491, out_desc, x490));
};
float* x496 = (float*)myMalloc(1 * sizeof(float));;
x496[0] = 1.0f;
float* x498 = (float*)myMalloc(1 * sizeof(float));;
x498[0] = 1.0f;

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
    cudnnHandle, x496, bias_desc, x140, x498, out_desc, x490));
};
float* x501 = (float*)myMalloc(1 * sizeof(float));;
x501[0] = 0.0f;
float* x503 = (float*)myMalloc(1 * sizeof(float));;
x503[0] = 1.0f;
float* x505 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x503, x_desc, x490, x501, x_desc, x505));
};
float* x507 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x508 = (float*)myMalloc(1 * sizeof(float));;
x508[0] = 0.0f;
float* x510 = (float*)myMalloc(1 * sizeof(float));;
x510[0] = 1.0f;

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
    x510, in_desc, x505, filt_desc, x116,
    conv_desc, algo, ws_data, ws_size,
    x508, out_desc, x507));
};
float* x513 = (float*)myMalloc(1 * sizeof(float));;
x513[0] = 1.0f;
float* x515 = (float*)myMalloc(1 * sizeof(float));;
x515[0] = 1.0f;

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
    cudnnHandle, x513, bias_desc, x158, x515, out_desc, x507));
};
float* x518 = (float*)myMalloc(1 * sizeof(float));;
x518[0] = 0.0f;
float* x520 = (float*)myMalloc(1 * sizeof(float));;
x520[0] = 1.0f;
float* x522 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x520, x_desc, x507, x518, x_desc, x522));
};
float* x524 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x525 = (float*)myMalloc(1 * sizeof(float));;
x525[0] = 0.0f;
float* x527 = (float*)myMalloc(1 * sizeof(float));;
x527[0] = 1.0f;

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
    x527, in_desc, x505, filt_desc, x203,
    conv_desc, algo, ws_data, ws_size,
    x525, out_desc, x524));
};
float* x530 = (float*)myMalloc(1 * sizeof(float));;
x530[0] = 1.0f;
float* x532 = (float*)myMalloc(1 * sizeof(float));;
x532[0] = 1.0f;

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
    cudnnHandle, x530, bias_desc, x143, x532, out_desc, x524));
};
float* x535 = (float*)myMalloc(1 * sizeof(float));;
x535[0] = 0.0f;
float* x537 = (float*)myMalloc(1 * sizeof(float));;
x537[0] = 1.0f;
float* x539 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x537, x_desc, x524, x535, x_desc, x539));
};
float* x541 = (float*)myGpuMalloc(1048576 * sizeof(float));
int* x542 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x542[0] = 64;
x542[1] = 256;
x542[2] = 8;
x542[3] = 8;
int* x547 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x547[0] = 16384;
x547[1] = 64;
x547[2] = 8;
x547[3] = 1;
{
dim3 grid(400, 2);
concat2D_1D_greg<<<grid, 512>>>(x522, 128, 524288, x539, 128, 524288, x541, 1, 64, 256, 8, 8, 16384, 64, 8, 1);
};
float* x553 = (float*)myGpuMalloc(196608 * sizeof(float));
float* x554 = (float*)myMalloc(1 * sizeof(float));;
x554[0] = 0.0f;
float* x556 = (float*)myMalloc(1 * sizeof(float));;
x556[0] = 1.0f;

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
    x556, in_desc, x541, filt_desc, x221,
    conv_desc, algo, ws_data, ws_size,
    x554, out_desc, x553));
};
float* x559 = (float*)myMalloc(1 * sizeof(float));;
x559[0] = 1.0f;
float* x561 = (float*)myMalloc(1 * sizeof(float));;
x561[0] = 1.0f;

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
    cudnnHandle, x559, bias_desc, x251, x561, out_desc, x553));
};
float* x564 = (float*)myMalloc(1 * sizeof(float));;
x564[0] = 0.0f;
float* x566 = (float*)myMalloc(1 * sizeof(float));;
x566[0] = 1.0f;
float* x568 = (float*)myGpuMalloc(196608 * sizeof(float));

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
    x566, x_desc, x553, x564, x_desc, x568));
};
float* x570 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x571 = (float*)myMalloc(1 * sizeof(float));;
x571[0] = 0.0f;
float* x573 = (float*)myMalloc(1 * sizeof(float));;
x573[0] = 1.0f;

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
    x573, in_desc, x568, filt_desc, x239,
    conv_desc, algo, ws_data, ws_size,
    x571, out_desc, x570));
};
float* x576 = (float*)myMalloc(1 * sizeof(float));;
x576[0] = 1.0f;
float* x578 = (float*)myMalloc(1 * sizeof(float));;
x578[0] = 1.0f;

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
    cudnnHandle, x576, bias_desc, x233, x578, out_desc, x570));
};
float* x581 = (float*)myMalloc(1 * sizeof(float));;
x581[0] = 0.0f;
float* x583 = (float*)myMalloc(1 * sizeof(float));;
x583[0] = 1.0f;
float* x585 = (float*)myGpuMalloc(786432 * sizeof(float));

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
    x583, x_desc, x570, x581, x_desc, x585));
};
float* x587 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x588 = (float*)myMalloc(1 * sizeof(float));;
x588[0] = 0.0f;
float* x590 = (float*)myMalloc(1 * sizeof(float));;
x590[0] = 1.0f;

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
    x590, in_desc, x568, filt_desc, x212,
    conv_desc, algo, ws_data, ws_size,
    x588, out_desc, x587));
};
float* x593 = (float*)myMalloc(1 * sizeof(float));;
x593[0] = 1.0f;
float* x595 = (float*)myMalloc(1 * sizeof(float));;
x595[0] = 1.0f;

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
    cudnnHandle, x593, bias_desc, x182, x595, out_desc, x587));
};
float* x598 = (float*)myMalloc(1 * sizeof(float));;
x598[0] = 0.0f;
float* x600 = (float*)myMalloc(1 * sizeof(float));;
x600[0] = 1.0f;
float* x602 = (float*)myGpuMalloc(786432 * sizeof(float));

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
    x600, x_desc, x587, x598, x_desc, x602));
};
float* x604 = (float*)myGpuMalloc(1572864 * sizeof(float));
int* x605 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x605[0] = 64;
x605[1] = 384;
x605[2] = 8;
x605[3] = 8;
int* x610 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x610[0] = 24576;
x610[1] = 64;
x610[2] = 8;
x610[3] = 1;
{
dim3 grid(400, 2);
concat2D_1D_greg<<<grid, 512>>>(x585, 192, 786432, x602, 192, 786432, x604, 1, 64, 384, 8, 8, 24576, 64, 8, 1);
};
float* x616 = (float*)myGpuMalloc(196608 * sizeof(float));
float* x617 = (float*)myMalloc(1 * sizeof(float));;
x617[0] = 0.0f;
float* x619 = (float*)myMalloc(1 * sizeof(float));;
x619[0] = 1.0f;

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
    x619, in_desc, x604, filt_desc, x137,
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
    1, 48, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 48, 8, 8));

CUDNN_CALL(cudnnAddTensor(
    cudnnHandle, x622, bias_desc, x101, x624, out_desc, x616));
};
float* x627 = (float*)myMalloc(1 * sizeof(float));;
x627[0] = 0.0f;
float* x629 = (float*)myMalloc(1 * sizeof(float));;
x629[0] = 1.0f;
float* x631 = (float*)myGpuMalloc(196608 * sizeof(float));

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
    x629, x_desc, x616, x627, x_desc, x631));
};
float* x633 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x634 = (float*)myMalloc(1 * sizeof(float));;
x634[0] = 0.0f;
float* x636 = (float*)myMalloc(1 * sizeof(float));;
x636[0] = 1.0f;

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
    x636, in_desc, x631, filt_desc, x161,
    conv_desc, algo, ws_data, ws_size,
    x634, out_desc, x633));
};
float* x639 = (float*)myMalloc(1 * sizeof(float));;
x639[0] = 1.0f;
float* x641 = (float*)myMalloc(1 * sizeof(float));;
x641[0] = 1.0f;

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
    cudnnHandle, x639, bias_desc, x191, x641, out_desc, x633));
};
float* x644 = (float*)myMalloc(1 * sizeof(float));;
x644[0] = 0.0f;
float* x646 = (float*)myMalloc(1 * sizeof(float));;
x646[0] = 1.0f;
float* x648 = (float*)myGpuMalloc(786432 * sizeof(float));

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
    x646, x_desc, x633, x644, x_desc, x648));
};
float* x650 = (float*)myGpuMalloc(786432 * sizeof(float));
float* x651 = (float*)myMalloc(1 * sizeof(float));;
x651[0] = 0.0f;
float* x653 = (float*)myMalloc(1 * sizeof(float));;
x653[0] = 1.0f;

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
    x653, in_desc, x631, filt_desc, x149,
    conv_desc, algo, ws_data, ws_size,
    x651, out_desc, x650));
};
float* x656 = (float*)myMalloc(1 * sizeof(float));;
x656[0] = 1.0f;
float* x658 = (float*)myMalloc(1 * sizeof(float));;
x658[0] = 1.0f;

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
    cudnnHandle, x656, bias_desc, x227, x658, out_desc, x650));
};
float* x661 = (float*)myMalloc(1 * sizeof(float));;
x661[0] = 0.0f;
float* x663 = (float*)myMalloc(1 * sizeof(float));;
x663[0] = 1.0f;
float* x665 = (float*)myGpuMalloc(786432 * sizeof(float));

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
    x663, x_desc, x650, x661, x_desc, x665));
};
float* x667 = (float*)myGpuMalloc(1572864 * sizeof(float));
int* x668 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x668[0] = 64;
x668[1] = 384;
x668[2] = 8;
x668[3] = 8;
int* x673 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x673[0] = 24576;
x673[1] = 64;
x673[2] = 8;
x673[3] = 1;
{
dim3 grid(400, 2);
concat2D_1D_greg<<<grid, 512>>>(x648, 192, 786432, x665, 192, 786432, x667, 1, 64, 384, 8, 8, 24576, 64, 8, 1);
};
float* x679 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x680 = (float*)myMalloc(1 * sizeof(float));;
x680[0] = 0.0f;
float* x682 = (float*)myMalloc(1 * sizeof(float));;
x682[0] = 1.0f;

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
    x682, in_desc, x667, filt_desc, x197,
    conv_desc, algo, ws_data, ws_size,
    x680, out_desc, x679));
};
float* x685 = (float*)myMalloc(1 * sizeof(float));;
x685[0] = 1.0f;
float* x687 = (float*)myMalloc(1 * sizeof(float));;
x687[0] = 1.0f;

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
    cudnnHandle, x685, bias_desc, x122, x687, out_desc, x679));
};
float* x690 = (float*)myMalloc(1 * sizeof(float));;
x690[0] = 0.0f;
float* x692 = (float*)myMalloc(1 * sizeof(float));;
x692[0] = 1.0f;
float* x694 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x692, x_desc, x679, x690, x_desc, x694));
};
float* x696 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x697 = (float*)myMalloc(1 * sizeof(float));;
x697[0] = 0.0f;
float* x699 = (float*)myMalloc(1 * sizeof(float));;
x699[0] = 1.0f;

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
    x699, in_desc, x694, filt_desc, x242,
    conv_desc, algo, ws_data, ws_size,
    x697, out_desc, x696));
};
float* x702 = (float*)myMalloc(1 * sizeof(float));;
x702[0] = 1.0f;
float* x704 = (float*)myMalloc(1 * sizeof(float));;
x704[0] = 1.0f;

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
    cudnnHandle, x702, bias_desc, x215, x704, out_desc, x696));
};
float* x707 = (float*)myMalloc(1 * sizeof(float));;
x707[0] = 0.0f;
float* x709 = (float*)myMalloc(1 * sizeof(float));;
x709[0] = 1.0f;
float* x711 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x709, x_desc, x696, x707, x_desc, x711));
};
float* x713 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x714 = (float*)myMalloc(1 * sizeof(float));;
x714[0] = 0.0f;
float* x716 = (float*)myMalloc(1 * sizeof(float));;
x716[0] = 1.0f;

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
    x716, in_desc, x694, filt_desc, x179,
    conv_desc, algo, ws_data, ws_size,
    x714, out_desc, x713));
};
float* x719 = (float*)myMalloc(1 * sizeof(float));;
x719[0] = 1.0f;
float* x721 = (float*)myMalloc(1 * sizeof(float));;
x721[0] = 1.0f;

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
    cudnnHandle, x719, bias_desc, x134, x721, out_desc, x713));
};
float* x724 = (float*)myMalloc(1 * sizeof(float));;
x724[0] = 0.0f;
float* x726 = (float*)myMalloc(1 * sizeof(float));;
x726[0] = 1.0f;
float* x728 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x726, x_desc, x713, x724, x_desc, x728));
};
float* x730 = (float*)myGpuMalloc(2097152 * sizeof(float));
int* x731 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x731[0] = 64;
x731[1] = 512;
x731[2] = 8;
x731[3] = 8;
int* x736 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x736[0] = 32768;
x736[1] = 64;
x736[2] = 8;
x736[3] = 1;
{
dim3 grid(400, 2);
concat2D_1D_greg<<<grid, 512>>>(x711, 256, 1048576, x728, 256, 1048576, x730, 1, 64, 512, 8, 8, 32768, 64, 8, 1);
};
float* x742 = (float*)myMalloc(1 * sizeof(float));;
x742[0] = 0.0f;
float* x744 = (float*)myMalloc(1 * sizeof(float));;
x744[0] = 1.0f;
float* x746 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x744, in_desc, x730, x742, out_desc, x746));
};
float* x748 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x749 = (float*)myMalloc(1 * sizeof(float));;
x749[0] = 0.0f;
float* x751 = (float*)myMalloc(1 * sizeof(float));;
x751[0] = 1.0f;

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
    x751, in_desc, x746, filt_desc, x98,
    conv_desc, algo, ws_data, ws_size,
    x749, out_desc, x748));
};
float* x754 = (float*)myMalloc(1 * sizeof(float));;
x754[0] = 1.0f;
float* x756 = (float*)myMalloc(1 * sizeof(float));;
x756[0] = 1.0f;

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
    cudnnHandle, x754, bias_desc, x155, x756, out_desc, x748));
};
float* x759 = (float*)myMalloc(1 * sizeof(float));;
x759[0] = 0.0f;
float* x761 = (float*)myMalloc(1 * sizeof(float));;
x761[0] = 1.0f;
float* x763 = (float*)myGpuMalloc(65536 * sizeof(float));

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
    x761, x_desc, x748, x759, x_desc, x763));
};
float* x765 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x766 = (float*)myMalloc(1 * sizeof(float));;
x766[0] = 0.0f;
float* x768 = (float*)myMalloc(1 * sizeof(float));;
x768[0] = 1.0f;

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
    x768, in_desc, x763, filt_desc, x209,
    conv_desc, algo, ws_data, ws_size,
    x766, out_desc, x765));
};
float* x771 = (float*)myMalloc(1 * sizeof(float));;
x771[0] = 1.0f;
float* x773 = (float*)myMalloc(1 * sizeof(float));;
x773[0] = 1.0f;

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
    cudnnHandle, x771, bias_desc, x173, x773, out_desc, x765));
};
float* x776 = (float*)myMalloc(1 * sizeof(float));;
x776[0] = 0.0f;
float* x778 = (float*)myMalloc(1 * sizeof(float));;
x778[0] = 1.0f;
float* x780 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x778, x_desc, x765, x776, x_desc, x780));
};
float* x782 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x783 = (float*)myMalloc(1 * sizeof(float));;
x783[0] = 0.0f;
float* x785 = (float*)myMalloc(1 * sizeof(float));;
x785[0] = 1.0f;

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
    x785, in_desc, x763, filt_desc, x185,
    conv_desc, algo, ws_data, ws_size,
    x783, out_desc, x782));
};
float* x788 = (float*)myMalloc(1 * sizeof(float));;
x788[0] = 1.0f;
float* x790 = (float*)myMalloc(1 * sizeof(float));;
x790[0] = 1.0f;

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
    cudnnHandle, x788, bias_desc, x146, x790, out_desc, x782));
};
float* x793 = (float*)myMalloc(1 * sizeof(float));;
x793[0] = 0.0f;
float* x795 = (float*)myMalloc(1 * sizeof(float));;
x795[0] = 1.0f;
float* x797 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x795, x_desc, x782, x793, x_desc, x797));
};
float* x799 = (float*)myGpuMalloc(524288 * sizeof(float));
int* x800 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x800[0] = 64;
x800[1] = 512;
x800[2] = 4;
x800[3] = 4;
int* x805 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x805[0] = 8192;
x805[1] = 16;
x805[2] = 4;
x805[3] = 1;
{
dim3 grid(400, 2);
concat2D_1D_greg<<<grid, 512>>>(x780, 256, 262144, x797, 256, 262144, x799, 1, 64, 512, 4, 4, 8192, 16, 4, 1);
};
float* x811 = (float*)myGpuMalloc(640 * sizeof(float));
float* x812 = (float*)myMalloc(1 * sizeof(float));;
x812[0] = 0.0f;
float* x814 = (float*)myMalloc(1 * sizeof(float));;
x814[0] = 1.0f;

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
    x814, in_desc, x799, filt_desc, x107,
    conv_desc, algo, ws_data, ws_size,
    x812, out_desc, x811));
};
float* x817 = (float*)myMalloc(1 * sizeof(float));;
x817[0] = 1.0f;
float* x819 = (float*)myMalloc(1 * sizeof(float));;
x819[0] = 1.0f;

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
    cudnnHandle, x817, bias_desc, x248, x819, out_desc, x811));
};
// resize to WrappedArray(64, 10)
int64_t x823 = (long)mallocAddr;
int64_t x824 = x823 - x253;
memset((void*)x253, 0, x824);
mallocAddr = (void*)x253;
int64_t x827 = (long)gpuMallocAddr;
int64_t x828 = x827 - x254;
cudaMemset((void*)x254, 0, x828);
gpuMallocAddr = (void*)x254;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x835 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x836 = x835 / 1000LL;
int64_t x838 = x835 / x837;
printf("Inferencing completed in %ldms (%ld us/images)\n",x836,x838);

}
// Backend cleanup.
CUBLAS_CALL(cublasDestroy(cublasHandle));
CUDA_CALL(cudaFree(gpuMallocBase));
      
CUDNN_CALL(cudnnDestroy(cudnnHandle));
}
/*****************************************
  End of C Generated Code                  
*******************************************/


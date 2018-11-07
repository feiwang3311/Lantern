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
// Tensor 'toGPU' invocation.
float* x275 = (float*)myGpuMalloc(262144 * sizeof(float));
int32_t x4 = open("/u/data/u99/wang603/TiarkMlEnv/Lantern/src/out/PLDI19evaluation/resnet50/resnet50.onnx.bin",0);
int32_t x5 = fsize(x4);
float* x6 = (float*)mmap(0, x5, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x4, 0);
float* x7 = x6+5205440;
CUDA_CALL(cudaMemcpy(x275, x7, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x278 = (float*)myGpuMalloc(256 * sizeof(float));
float* x8 = x6+148672;
CUDA_CALL(cudaMemcpy(x278, x8, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x281 = (float*)myGpuMalloc(128 * sizeof(float));
float* x9 = x6+816064;
CUDA_CALL(cudaMemcpy(x281, x9, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x284 = (float*)myGpuMalloc(128 * sizeof(float));
float* x10 = x6+950080;
CUDA_CALL(cudaMemcpy(x284, x10, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x287 = (float*)myGpuMalloc(64 * sizeof(float));
float* x11 = x6+94784;
CUDA_CALL(cudaMemcpy(x287, x11, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x290 = (float*)myGpuMalloc(32768 * sizeof(float));
float* x12 = x6+220608;
CUDA_CALL(cudaMemcpy(x290, x12, 32768 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x293 = (float*)myGpuMalloc(512 * sizeof(float));
float* x13 = x6+22495680;
CUDA_CALL(cudaMemcpy(x293, x13, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x296 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x14 = x6+2964928;
CUDA_CALL(cudaMemcpy(x296, x14, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x299 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x15 = x6+4348352;
CUDA_CALL(cudaMemcpy(x299, x15, 589824 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x302 = (float*)myGpuMalloc(512 * sizeof(float));
float* x16 = x6+20133312;
CUDA_CALL(cudaMemcpy(x302, x16, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x305 = (float*)myGpuMalloc(256 * sizeof(float));
float* x17 = x6+2169536;
CUDA_CALL(cudaMemcpy(x305, x17, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x308 = (float*)myGpuMalloc(128 * sizeof(float));
float* x18 = x6+668224;
CUDA_CALL(cudaMemcpy(x308, x18, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x311 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x19 = x6+2432448;
CUDA_CALL(cudaMemcpy(x311, x19, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x314 = (float*)myGpuMalloc(512 * sizeof(float));
float* x20 = x6+1446336;
CUDA_CALL(cudaMemcpy(x314, x20, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x317 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x21 = x6+4081088;
CUDA_CALL(cudaMemcpy(x317, x21, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x320 = (float*)myGpuMalloc(256 * sizeof(float));
float* x22 = x6+1578688;
CUDA_CALL(cudaMemcpy(x320, x22, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x323 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x23 = x6+6325696;
CUDA_CALL(cudaMemcpy(x323, x23, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x326 = (float*)myGpuMalloc(512 * sizeof(float));
float* x24 = x6+602048;
CUDA_CALL(cudaMemcpy(x326, x24, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x329 = (float*)myGpuMalloc(64 * sizeof(float));
float* x25 = x6+165888;
CUDA_CALL(cudaMemcpy(x329, x25, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x332 = (float*)myGpuMalloc(512 * sizeof(float));
float* x26 = x6+1164736;
CUDA_CALL(cudaMemcpy(x332, x26, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x335 = (float*)myGpuMalloc(64 * sizeof(float));
float* x27 = x6+6080;
CUDA_CALL(cudaMemcpy(x335, x27, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x338 = (float*)myGpuMalloc(147456 * sizeof(float));
float* x28 = x6+253888;
CUDA_CALL(cudaMemcpy(x338, x28, 147456 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x341 = (float*)myGpuMalloc(2359296 * sizeof(float));
float* x29 = x6+20135360;
CUDA_CALL(cudaMemcpy(x341, x29, 2359296 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x344 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x30 = x6+2960832;
CUDA_CALL(cudaMemcpy(x344, x30, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x347 = (float*)myGpuMalloc(256 * sizeof(float));
float* x31 = x6+3227072;
CUDA_CALL(cudaMemcpy(x347, x31, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x350 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x32 = x6+3228096;
CUDA_CALL(cudaMemcpy(x350, x32, 589824 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x353 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x33 = x6+43456;
CUDA_CALL(cudaMemcpy(x353, x33, 16384 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x356 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x34 = x6+22496704;
CUDA_CALL(cudaMemcpy(x356, x34, 1048576 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x359 = (float*)myGpuMalloc(2359296 * sizeof(float));
float* x35 = x6+9092544;
CUDA_CALL(cudaMemcpy(x359, x35, 2359296 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x362 = (float*)myGpuMalloc(128 * sizeof(float));
float* x36 = x6+816320;
CUDA_CALL(cudaMemcpy(x362, x36, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x365 = (float*)myGpuMalloc(256 * sizeof(float));
float* x37 = x6+60608;
CUDA_CALL(cudaMemcpy(x365, x37, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x368 = (float*)myGpuMalloc(256 * sizeof(float));
float* x38 = x6+219584;
CUDA_CALL(cudaMemcpy(x368, x38, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x371 = (float*)myGpuMalloc(128 * sizeof(float));
float* x39 = x6+1379392;
CUDA_CALL(cudaMemcpy(x371, x39, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x374 = (float*)myGpuMalloc(128 * sizeof(float));
float* x40 = x6+1231296;
CUDA_CALL(cudaMemcpy(x374, x40, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x377 = (float*)myGpuMalloc(64 * sizeof(float));
float* x41 = x6+1856;
CUDA_CALL(cudaMemcpy(x377, x41, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x380 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x42 = x6+1098176;
CUDA_CALL(cudaMemcpy(x380, x42, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x383 = (float*)myGpuMalloc(512 * sizeof(float));
float* x43 = x6+601536;
CUDA_CALL(cudaMemcpy(x383, x43, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x386 = (float*)myGpuMalloc(128 * sizeof(float));
float* x44 = x6+401728;
CUDA_CALL(cudaMemcpy(x386, x44, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x389 = (float*)myGpuMalloc(64 * sizeof(float));
float* x45 = x6+131904;
CUDA_CALL(cudaMemcpy(x389, x45, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x392 = (float*)myGpuMalloc(128 * sizeof(float));
float* x46 = x6+949696;
CUDA_CALL(cudaMemcpy(x392, x46, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x395 = (float*)myGpuMalloc(512 * sizeof(float));
float* x47 = x6+15664576;
CUDA_CALL(cudaMemcpy(x395, x47, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x398 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x48 = x6+18027968;
CUDA_CALL(cudaMemcpy(x398, x48, 1048576 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x401 = (float*)myGpuMalloc(10 * sizeof(float));
float* x49 = x6+23573952;
CUDA_CALL(cudaMemcpy(x401, x49, 10 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x404 = (float*)myGpuMalloc(64 * sizeof(float));
float* x50 = x6+43264;
CUDA_CALL(cudaMemcpy(x404, x50, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x407 = (float*)myGpuMalloc(512 * sizeof(float));
float* x51 = x6+11453376;
CUDA_CALL(cudaMemcpy(x407, x51, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x410 = (float*)myGpuMalloc(64 * sizeof(float));
float* x52 = x6+6272;
CUDA_CALL(cudaMemcpy(x410, x52, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x413 = (float*)myGpuMalloc(512 * sizeof(float));
float* x53 = x6+882112;
CUDA_CALL(cudaMemcpy(x413, x53, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x416 = (float*)myGpuMalloc(64 * sizeof(float));
float* x54 = x6+6144;
CUDA_CALL(cudaMemcpy(x416, x54, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x419 = (float*)myGpuMalloc(512 * sizeof(float));
float* x55 = x6+1445824;
CUDA_CALL(cudaMemcpy(x419, x55, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x422 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x56 = x6+1379776;
CUDA_CALL(cudaMemcpy(x422, x56, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x425 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x57 = x6+3818944;
CUDA_CALL(cudaMemcpy(x425, x57, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x428 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x58 = x6+5202368;
CUDA_CALL(cudaMemcpy(x428, x58, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x431 = (float*)myGpuMalloc(256 * sizeof(float));
float* x59 = x6+148416;
CUDA_CALL(cudaMemcpy(x431, x59, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x434 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x60 = x6+7441856;
CUDA_CALL(cudaMemcpy(x434, x60, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x437 = (float*)myGpuMalloc(64 * sizeof(float));
float* x61 = x6+94720;
CUDA_CALL(cudaMemcpy(x437, x61, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x440 = (float*)myGpuMalloc(128 * sizeof(float));
float* x62 = x6+1097792;
CUDA_CALL(cudaMemcpy(x440, x62, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x443 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x63 = x6+12504512;
CUDA_CALL(cudaMemcpy(x443, x63, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x446 = (float*)myGpuMalloc(256 * sizeof(float));
float* x64 = x6+4938944;
CUDA_CALL(cudaMemcpy(x446, x64, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x449 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x65 = x6+14611904;
CUDA_CALL(cudaMemcpy(x449, x65, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x452 = (float*)myGpuMalloc(512 * sizeof(float));
float* x66 = x6+15666112;
CUDA_CALL(cudaMemcpy(x452, x66, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x455 = (float*)myGpuMalloc(512 * sizeof(float));
float* x67 = x6+18026432;
CUDA_CALL(cudaMemcpy(x455, x67, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x458 = (float*)myGpuMalloc(512 * sizeof(float));
float* x68 = x6+9091520;
CUDA_CALL(cudaMemcpy(x458, x68, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x461 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x69 = x6+19080640;
CUDA_CALL(cudaMemcpy(x461, x69, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x464 = (float*)myGpuMalloc(256 * sizeof(float));
float* x70 = x6+6588608;
CUDA_CALL(cudaMemcpy(x464, x70, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x467 = (float*)myGpuMalloc(256 * sizeof(float));
float* x71 = x6+8299456;
CUDA_CALL(cudaMemcpy(x467, x71, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x470 = (float*)myGpuMalloc(256 * sizeof(float));
float* x72 = x6+60352;
CUDA_CALL(cudaMemcpy(x470, x72, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x473 = (float*)myGpuMalloc(64 * sizeof(float));
float* x73 = x6+202944;
CUDA_CALL(cudaMemcpy(x473, x73, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x476 = (float*)myGpuMalloc(36864 * sizeof(float));
float* x74 = x6+166080;
CUDA_CALL(cudaMemcpy(x476, x74, 36864 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x479 = (float*)myGpuMalloc(256 * sizeof(float));
float* x75 = x6+6058432;
CUDA_CALL(cudaMemcpy(x479, x75, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x482 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x76 = x6+2436544;
CUDA_CALL(cudaMemcpy(x482, x76, 524288 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x485 = (float*)myGpuMalloc(256 * sizeof(float));
float* x77 = x6+77248;
CUDA_CALL(cudaMemcpy(x485, x77, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x488 = (float*)myGpuMalloc(256 * sizeof(float));
float* x78 = x6+6587840;
CUDA_CALL(cudaMemcpy(x488, x78, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x491 = (float*)myGpuMalloc(512 * sizeof(float));
float* x79 = x6+20133824;
CUDA_CALL(cudaMemcpy(x491, x79, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x494 = (float*)myGpuMalloc(128 * sizeof(float));
float* x80 = x6+1379264;
CUDA_CALL(cudaMemcpy(x494, x80, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x497 = (float*)myGpuMalloc(256 * sizeof(float));
float* x81 = x6+7708608;
CUDA_CALL(cudaMemcpy(x497, x81, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x500 = (float*)myGpuMalloc(64 * sizeof(float));
float* x82 = x6+165824;
CUDA_CALL(cudaMemcpy(x500, x82, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x503 = (float*)myGpuMalloc(512 * sizeof(float));
float* x83 = x6+1164224;
CUDA_CALL(cudaMemcpy(x503, x83, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x506 = (float*)myGpuMalloc(36864 * sizeof(float));
float* x84 = x6+94912;
CUDA_CALL(cudaMemcpy(x506, x84, 36864 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x509 = (float*)myGpuMalloc(128 * sizeof(float));
float* x85 = x6+253376;
CUDA_CALL(cudaMemcpy(x509, x85, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x512 = (float*)myGpuMalloc(256 * sizeof(float));
float* x86 = x6+7708096;
CUDA_CALL(cudaMemcpy(x512, x86, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x515 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x87 = x6+2962880;
CUDA_CALL(cudaMemcpy(x515, x87, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x518 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x88 = x6+203200;
CUDA_CALL(cudaMemcpy(x518, x88, 16384 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x521 = (float*)myGpuMalloc(512 * sizeof(float));
float* x89 = x6+883648;
CUDA_CALL(cudaMemcpy(x521, x89, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x524 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x90 = x6+6059456;
CUDA_CALL(cudaMemcpy(x524, x90, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x527 = (float*)myGpuMalloc(36864 * sizeof(float));
float* x91 = x6+6336;
CUDA_CALL(cudaMemcpy(x527, x91, 36864 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x530 = (float*)myGpuMalloc(256 * sizeof(float));
float* x92 = x6+148928;
CUDA_CALL(cudaMemcpy(x530, x92, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x533 = (float*)myGpuMalloc(256 * sizeof(float));
float* x93 = x6+5467584;
CUDA_CALL(cudaMemcpy(x533, x93, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x536 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x94 = x6+8563136;
CUDA_CALL(cudaMemcpy(x536, x94, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x539 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x95 = x6+19076544;
CUDA_CALL(cudaMemcpy(x539, x95, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x542 = (float*)myGpuMalloc(128 * sizeof(float));
float* x96 = x6+816192;
CUDA_CALL(cudaMemcpy(x542, x96, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x545 = (float*)myGpuMalloc(256 * sizeof(float));
float* x97 = x6+3818176;
CUDA_CALL(cudaMemcpy(x545, x97, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x548 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x98 = x6+8299968;
CUDA_CALL(cudaMemcpy(x548, x98, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x551 = (float*)myGpuMalloc(256 * sizeof(float));
float* x99 = x6+5468352;
CUDA_CALL(cudaMemcpy(x551, x99, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x554 = (float*)myGpuMalloc(256 * sizeof(float));
float* x100 = x6+2170048;
CUDA_CALL(cudaMemcpy(x554, x100, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x557 = (float*)myGpuMalloc(128 * sizeof(float));
float* x101 = x6+668352;
CUDA_CALL(cudaMemcpy(x557, x101, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x560 = (float*)myGpuMalloc(512 * sizeof(float));
float* x102 = x6+468928;
CUDA_CALL(cudaMemcpy(x560, x102, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x563 = (float*)myGpuMalloc(64 * sizeof(float));
float* x103 = x6+94848;
CUDA_CALL(cudaMemcpy(x563, x103, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x566 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x104 = x6+23545280;
CUDA_CALL(cudaMemcpy(x566, x104, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x569 = (float*)myGpuMalloc(256 * sizeof(float));
float* x105 = x6+7179456;
CUDA_CALL(cudaMemcpy(x569, x105, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x572 = (float*)myGpuMalloc(64 * sizeof(float));
float* x106 = x6+43328;
CUDA_CALL(cudaMemcpy(x572, x106, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x575 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x107 = x6+401856;
CUDA_CALL(cudaMemcpy(x575, x107, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x578 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x108 = x6+14609856;
CUDA_CALL(cudaMemcpy(x578, x108, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x581 = (float*)myGpuMalloc(256 * sizeof(float));
float* x109 = x6+2169280;
CUDA_CALL(cudaMemcpy(x581, x109, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x584 = (float*)myGpuMalloc(256 * sizeof(float));
float* x110 = x6+7178944;
CUDA_CALL(cudaMemcpy(x584, x110, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x587 = (float*)myGpuMalloc(64 * sizeof(float));
float* x111 = x6+1920;
CUDA_CALL(cudaMemcpy(x587, x111, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x590 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x112 = x6+816576;
CUDA_CALL(cudaMemcpy(x590, x112, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x593 = (float*)myGpuMalloc(128 * sizeof(float));
float* x113 = x6+949952;
CUDA_CALL(cudaMemcpy(x593, x113, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x596 = (float*)myGpuMalloc(512 * sizeof(float));
float* x114 = x6+11452864;
CUDA_CALL(cudaMemcpy(x596, x114, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x599 = (float*)myGpuMalloc(64 * sizeof(float));
float* x115 = x6+6208;
CUDA_CALL(cudaMemcpy(x599, x115, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x602 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x116 = x6+12506560;
CUDA_CALL(cudaMemcpy(x602, x116, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x605 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x117 = x6+4939200;
CUDA_CALL(cudaMemcpy(x605, x117, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x608 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x118 = x6+2433472;
CUDA_CALL(cudaMemcpy(x608, x118, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x611 = (float*)myGpuMalloc(64 * sizeof(float));
float* x119 = x6+203136;
CUDA_CALL(cudaMemcpy(x611, x119, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x614 = (float*)myGpuMalloc(512 * sizeof(float));
float* x120 = x6+601024;
CUDA_CALL(cudaMemcpy(x614, x120, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x617 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x121 = x6+7442880;
CUDA_CALL(cudaMemcpy(x617, x121, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x620 = (float*)myGpuMalloc(512 * sizeof(float));
float* x122 = x6+9092032;
CUDA_CALL(cudaMemcpy(x620, x122, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x623 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x123 = x6+8564160;
CUDA_CALL(cudaMemcpy(x623, x123, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x626 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x124 = x6+23551424;
CUDA_CALL(cudaMemcpy(x626, x124, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x629 = (float*)myGpuMalloc(256 * sizeof(float));
float* x125 = x6+4938688;
CUDA_CALL(cudaMemcpy(x629, x125, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x632 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x126 = x6+14613952;
CUDA_CALL(cudaMemcpy(x632, x126, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x635 = (float*)myGpuMalloc(256 * sizeof(float));
float* x127 = x6+60096;
CUDA_CALL(cudaMemcpy(x635, x127, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x638 = (float*)myGpuMalloc(128 * sizeof(float));
float* x128 = x6+1097664;
CUDA_CALL(cudaMemcpy(x638, x128, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x641 = (float*)myGpuMalloc(128 * sizeof(float));
float* x129 = x6+401600;
CUDA_CALL(cudaMemcpy(x641, x129, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x644 = (float*)myGpuMalloc(256 * sizeof(float));
float* x130 = x6+4347328;
CUDA_CALL(cudaMemcpy(x644, x130, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x647 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x131 = x6+132032;
CUDA_CALL(cudaMemcpy(x647, x131, 16384 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x650 = (float*)myGpuMalloc(256 * sizeof(float));
float* x132 = x6+1578944;
CUDA_CALL(cudaMemcpy(x650, x132, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x653 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x133 = x6+1165760;
CUDA_CALL(cudaMemcpy(x653, x133, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x656 = (float*)myGpuMalloc(256 * sizeof(float));
float* x134 = x6+220352;
CUDA_CALL(cudaMemcpy(x656, x134, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x659 = (float*)myGpuMalloc(128 * sizeof(float));
float* x135 = x6+253760;
CUDA_CALL(cudaMemcpy(x659, x135, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x662 = (float*)myGpuMalloc(64 * sizeof(float));
float* x136 = x6+203008;
CUDA_CALL(cudaMemcpy(x662, x136, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x665 = (float*)myGpuMalloc(256 * sizeof(float));
float* x137 = x6+6058688;
CUDA_CALL(cudaMemcpy(x665, x137, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x668 = (float*)myGpuMalloc(512 * sizeof(float));
float* x138 = x6+15665088;
CUDA_CALL(cudaMemcpy(x668, x138, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x671 = (float*)myGpuMalloc(512 * sizeof(float));
float* x139 = x6+18026944;
CUDA_CALL(cudaMemcpy(x671, x139, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x674 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x140 = x6+8566208;
CUDA_CALL(cudaMemcpy(x674, x140, 524288 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x677 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x141 = x6+5203392;
CUDA_CALL(cudaMemcpy(x677, x141, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x680 = (float*)myGpuMalloc(256 * sizeof(float));
float* x142 = x6+8298944;
CUDA_CALL(cudaMemcpy(x680, x142, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x683 = (float*)myGpuMalloc(64 * sizeof(float));
float* x143 = x6+94656;
CUDA_CALL(cudaMemcpy(x683, x143, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x686 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x144 = x6+4084160;
CUDA_CALL(cudaMemcpy(x686, x144, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x689 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x145 = x6+19078592;
CUDA_CALL(cudaMemcpy(x689, x145, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x692 = (float*)myGpuMalloc(512 * sizeof(float));
float* x146 = x6+467392;
CUDA_CALL(cudaMemcpy(x692, x146, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x695 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x147 = x6+6322624;
CUDA_CALL(cudaMemcpy(x695, x147, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x698 = (float*)myGpuMalloc(512 * sizeof(float));
float* x148 = x6+883136;
CUDA_CALL(cudaMemcpy(x698, x148, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x701 = (float*)myGpuMalloc(128 * sizeof(float));
float* x149 = x6+1379648;
CUDA_CALL(cudaMemcpy(x701, x149, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x704 = (float*)myGpuMalloc(512 * sizeof(float));
float* x150 = x6+468416;
CUDA_CALL(cudaMemcpy(x704, x150, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x707 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x151 = x6+149440;
CUDA_CALL(cudaMemcpy(x707, x151, 16384 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x710 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x152 = x6+7445952;
CUDA_CALL(cudaMemcpy(x710, x152, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x713 = (float*)myGpuMalloc(1728 * sizeof(float));
float* x153 = x6+0;
CUDA_CALL(cudaMemcpy(x713, x153, 1728 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x716 = (float*)myGpuMalloc(64 * sizeof(float));
float* x154 = x6+131840;
CUDA_CALL(cudaMemcpy(x716, x154, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x719 = (float*)myGpuMalloc(512 * sizeof(float));
float* x155 = x6+15665600;
CUDA_CALL(cudaMemcpy(x719, x155, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x722 = (float*)myGpuMalloc(2359296 * sizeof(float));
float* x156 = x6+15666624;
CUDA_CALL(cudaMemcpy(x722, x156, 2359296 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x725 = (float*)myGpuMalloc(512 * sizeof(float));
float* x157 = x6+1445312;
CUDA_CALL(cudaMemcpy(x725, x157, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x728 = (float*)myGpuMalloc(256 * sizeof(float));
float* x158 = x6+3227840;
CUDA_CALL(cudaMemcpy(x728, x158, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x731 = (float*)myGpuMalloc(64 * sizeof(float));
float* x159 = x6+43392;
CUDA_CALL(cudaMemcpy(x731, x159, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x734 = (float*)myGpuMalloc(512 * sizeof(float));
float* x160 = x6+11452352;
CUDA_CALL(cudaMemcpy(x734, x160, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x737 = (float*)myGpuMalloc(512 * sizeof(float));
float* x161 = x6+18025920;
CUDA_CALL(cudaMemcpy(x737, x161, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x740 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x162 = x6+6324672;
CUDA_CALL(cudaMemcpy(x740, x162, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x743 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x163 = x6+60864;
CUDA_CALL(cudaMemcpy(x743, x163, 16384 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x746 = (float*)myGpuMalloc(256 * sizeof(float));
float* x164 = x6+5468096;
CUDA_CALL(cudaMemcpy(x746, x164, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x749 = (float*)myGpuMalloc(64 * sizeof(float));
float* x165 = x6+43200;
CUDA_CALL(cudaMemcpy(x749, x165, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x752 = (float*)myGpuMalloc(147456 * sizeof(float));
float* x166 = x6+1231808;
CUDA_CALL(cudaMemcpy(x752, x166, 147456 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x755 = (float*)myGpuMalloc(256 * sizeof(float));
float* x167 = x6+149184;
CUDA_CALL(cudaMemcpy(x755, x167, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x758 = (float*)myGpuMalloc(512 * sizeof(float));
float* x168 = x6+1163712;
CUDA_CALL(cudaMemcpy(x758, x168, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x761 = (float*)myGpuMalloc(256 * sizeof(float));
float* x169 = x6+7178688;
CUDA_CALL(cudaMemcpy(x761, x169, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x764 = (float*)myGpuMalloc(512 * sizeof(float));
float* x170 = x6+22495168;
CUDA_CALL(cudaMemcpy(x764, x170, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x767 = (float*)myGpuMalloc(128 * sizeof(float));
float* x171 = x6+949824;
CUDA_CALL(cudaMemcpy(x767, x171, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x770 = (float*)myGpuMalloc(16384 * sizeof(float));
float* x172 = x6+78272;
CUDA_CALL(cudaMemcpy(x770, x172, 16384 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x773 = (float*)myGpuMalloc(128 * sizeof(float));
float* x173 = x6+253504;
CUDA_CALL(cudaMemcpy(x773, x173, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x776 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x174 = x6+14607808;
CUDA_CALL(cudaMemcpy(x776, x174, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x779 = (float*)myGpuMalloc(256 * sizeof(float));
float* x175 = x6+4348096;
CUDA_CALL(cudaMemcpy(x779, x175, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x782 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x176 = x6+1579456;
CUDA_CALL(cudaMemcpy(x782, x176, 589824 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x785 = (float*)myGpuMalloc(256 * sizeof(float));
float* x177 = x6+7708864;
CUDA_CALL(cudaMemcpy(x785, x177, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x788 = (float*)myGpuMalloc(128 * sizeof(float));
float* x178 = x6+668480;
CUDA_CALL(cudaMemcpy(x788, x178, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x791 = (float*)myGpuMalloc(256 * sizeof(float));
float* x179 = x6+4347840;
CUDA_CALL(cudaMemcpy(x791, x179, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x794 = (float*)myGpuMalloc(64 * sizeof(float));
float* x180 = x6+203072;
CUDA_CALL(cudaMemcpy(x794, x180, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x797 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x181 = x6+1447360;
CUDA_CALL(cudaMemcpy(x797, x181, 131072 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x800 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x182 = x6+23547328;
CUDA_CALL(cudaMemcpy(x800, x182, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x803 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x183 = x6+4083136;
CUDA_CALL(cudaMemcpy(x803, x183, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x806 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x184 = x6+8565184;
CUDA_CALL(cudaMemcpy(x806, x184, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x809 = (float*)myGpuMalloc(256 * sizeof(float));
float* x185 = x6+220096;
CUDA_CALL(cudaMemcpy(x809, x185, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x812 = (float*)myGpuMalloc(256 * sizeof(float));
float* x186 = x6+6588096;
CUDA_CALL(cudaMemcpy(x812, x186, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x815 = (float*)myGpuMalloc(256 * sizeof(float));
float* x187 = x6+6058944;
CUDA_CALL(cudaMemcpy(x815, x187, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x818 = (float*)myGpuMalloc(64 * sizeof(float));
float* x188 = x6+166016;
CUDA_CALL(cudaMemcpy(x818, x188, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x821 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x189 = x6+5204416;
CUDA_CALL(cudaMemcpy(x821, x189, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x824 = (float*)myGpuMalloc(256 * sizeof(float));
float* x190 = x6+8299200;
CUDA_CALL(cudaMemcpy(x824, x190, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x827 = (float*)myGpuMalloc(128 * sizeof(float));
float* x191 = x6+401472;
CUDA_CALL(cudaMemcpy(x827, x191, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x830 = (float*)myGpuMalloc(147456 * sizeof(float));
float* x192 = x6+950208;
CUDA_CALL(cudaMemcpy(x830, x192, 147456 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x833 = (float*)myGpuMalloc(256 * sizeof(float));
float* x193 = x6+4938432;
CUDA_CALL(cudaMemcpy(x833, x193, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x836 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x194 = x6+12508608;
CUDA_CALL(cudaMemcpy(x836, x194, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x839 = (float*)myGpuMalloc(512 * sizeof(float));
float* x195 = x6+22494656;
CUDA_CALL(cudaMemcpy(x839, x195, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x842 = (float*)myGpuMalloc(512 * sizeof(float));
float* x196 = x6+18027456;
CUDA_CALL(cudaMemcpy(x842, x196, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x845 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x197 = x6+884160;
CUDA_CALL(cudaMemcpy(x845, x197, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x848 = (float*)myGpuMalloc(256 * sizeof(float));
float* x198 = x6+4347584;
CUDA_CALL(cudaMemcpy(x848, x198, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x851 = (float*)myGpuMalloc(256 * sizeof(float));
float* x199 = x6+1579200;
CUDA_CALL(cudaMemcpy(x851, x199, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x854 = (float*)myGpuMalloc(256 * sizeof(float));
float* x200 = x6+59840;
CUDA_CALL(cudaMemcpy(x854, x200, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x857 = (float*)myGpuMalloc(256 * sizeof(float));
float* x201 = x6+3818432;
CUDA_CALL(cudaMemcpy(x857, x201, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x860 = (float*)myGpuMalloc(512 * sizeof(float));
float* x202 = x6+9090496;
CUDA_CALL(cudaMemcpy(x860, x202, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x863 = (float*)myGpuMalloc(512 * sizeof(float));
float* x203 = x6+22496192;
CUDA_CALL(cudaMemcpy(x863, x203, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x866 = (float*)myGpuMalloc(256 * sizeof(float));
float* x204 = x6+77504;
CUDA_CALL(cudaMemcpy(x866, x204, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x869 = (float*)myGpuMalloc(128 * sizeof(float));
float* x205 = x6+253632;
CUDA_CALL(cudaMemcpy(x869, x205, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x872 = (float*)myGpuMalloc(512 * sizeof(float));
float* x206 = x6+11451840;
CUDA_CALL(cudaMemcpy(x872, x206, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x875 = (float*)myGpuMalloc(64 * sizeof(float));
float* x207 = x6+1728;
CUDA_CALL(cudaMemcpy(x875, x207, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x878 = (float*)myGpuMalloc(512 * sizeof(float));
float* x208 = x6+600512;
CUDA_CALL(cudaMemcpy(x878, x208, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x881 = (float*)myGpuMalloc(64 * sizeof(float));
float* x209 = x6+131776;
CUDA_CALL(cudaMemcpy(x881, x209, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x884 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x210 = x6+7443904;
CUDA_CALL(cudaMemcpy(x884, x210, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x887 = (float*)myGpuMalloc(512 * sizeof(float));
float* x211 = x6+467904;
CUDA_CALL(cudaMemcpy(x887, x211, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x890 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x212 = x6+2963904;
CUDA_CALL(cudaMemcpy(x890, x212, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x893 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x213 = x6+11453888;
CUDA_CALL(cudaMemcpy(x893, x213, 1048576 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x896 = (float*)myGpuMalloc(512 * sizeof(float));
float* x214 = x6+20134336;
CUDA_CALL(cudaMemcpy(x896, x214, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x899 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x215 = x6+12510656;
CUDA_CALL(cudaMemcpy(x899, x215, 2097152 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x902 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x216 = x6+14616000;
CUDA_CALL(cudaMemcpy(x902, x216, 1048576 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x905 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x217 = x6+2434496;
CUDA_CALL(cudaMemcpy(x905, x217, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x908 = (float*)myGpuMalloc(128 * sizeof(float));
float* x218 = x6+1097920;
CUDA_CALL(cudaMemcpy(x908, x218, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x911 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x219 = x6+4085184;
CUDA_CALL(cudaMemcpy(x911, x219, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x914 = (float*)myGpuMalloc(256 * sizeof(float));
float* x220 = x6+3227328;
CUDA_CALL(cudaMemcpy(x914, x220, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x917 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x221 = x6+2961856;
CUDA_CALL(cudaMemcpy(x917, x221, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x920 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x222 = x6+7179712;
CUDA_CALL(cudaMemcpy(x920, x222, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x923 = (float*)myGpuMalloc(128 * sizeof(float));
float* x223 = x6+668096;
CUDA_CALL(cudaMemcpy(x923, x223, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x926 = (float*)myGpuMalloc(512 * sizeof(float));
float* x224 = x6+1165248;
CUDA_CALL(cudaMemcpy(x926, x224, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x929 = (float*)myGpuMalloc(512 * sizeof(float));
float* x225 = x6+9091008;
CUDA_CALL(cudaMemcpy(x929, x225, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x932 = (float*)myGpuMalloc(128 * sizeof(float));
float* x226 = x6+816448;
CUDA_CALL(cudaMemcpy(x932, x226, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x935 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x227 = x6+7709120;
CUDA_CALL(cudaMemcpy(x935, x227, 589824 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x938 = (float*)myGpuMalloc(20480 * sizeof(float));
float* x228 = x6+23553472;
CUDA_CALL(cudaMemcpy(x938, x228, 20480 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x941 = (float*)myGpuMalloc(256 * sizeof(float));
float* x229 = x6+4938176;
CUDA_CALL(cudaMemcpy(x941, x229, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x944 = (float*)myGpuMalloc(256 * sizeof(float));
float* x230 = x6+2169792;
CUDA_CALL(cudaMemcpy(x944, x230, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x947 = (float*)myGpuMalloc(256 * sizeof(float));
float* x231 = x6+6059200;
CUDA_CALL(cudaMemcpy(x947, x231, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x950 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x232 = x6+6323648;
CUDA_CALL(cudaMemcpy(x950, x232, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x953 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x233 = x6+4082112;
CUDA_CALL(cudaMemcpy(x953, x233, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x956 = (float*)myGpuMalloc(4096 * sizeof(float));
float* x234 = x6+1984;
CUDA_CALL(cudaMemcpy(x956, x234, 4096 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x959 = (float*)myGpuMalloc(512 * sizeof(float));
float* x235 = x6+1446848;
CUDA_CALL(cudaMemcpy(x959, x235, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x962 = (float*)myGpuMalloc(147456 * sizeof(float));
float* x236 = x6+668608;
CUDA_CALL(cudaMemcpy(x962, x236, 147456 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x965 = (float*)myGpuMalloc(128 * sizeof(float));
float* x237 = x6+1231552;
CUDA_CALL(cudaMemcpy(x965, x237, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x968 = (float*)myGpuMalloc(256 * sizeof(float));
float* x238 = x6+3818688;
CUDA_CALL(cudaMemcpy(x968, x238, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x971 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x239 = x6+6321600;
CUDA_CALL(cudaMemcpy(x971, x239, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x974 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x240 = x6+12502464;
CUDA_CALL(cudaMemcpy(x974, x240, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x977 = (float*)myGpuMalloc(256 * sizeof(float));
float* x241 = x6+8299712;
CUDA_CALL(cudaMemcpy(x977, x241, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x980 = (float*)myGpuMalloc(256 * sizeof(float));
float* x242 = x6+5467840;
CUDA_CALL(cudaMemcpy(x980, x242, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x983 = (float*)myGpuMalloc(128 * sizeof(float));
float* x243 = x6+1231424;
CUDA_CALL(cudaMemcpy(x983, x243, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x986 = (float*)myGpuMalloc(256 * sizeof(float));
float* x244 = x6+78016;
CUDA_CALL(cudaMemcpy(x986, x244, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x989 = (float*)myGpuMalloc(64 * sizeof(float));
float* x245 = x6+131968;
CUDA_CALL(cudaMemcpy(x989, x245, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x992 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x246 = x6+19082688;
CUDA_CALL(cudaMemcpy(x992, x246, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x995 = (float*)myGpuMalloc(512 * sizeof(float));
float* x247 = x6+882624;
CUDA_CALL(cudaMemcpy(x995, x247, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x998 = (float*)myGpuMalloc(256 * sizeof(float));
float* x248 = x6+219840;
CUDA_CALL(cudaMemcpy(x998, x248, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1001 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x249 = x6+8562112;
CUDA_CALL(cudaMemcpy(x1001, x249, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1004 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x250 = x6+5468608;
CUDA_CALL(cudaMemcpy(x1004, x250, 589824 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1007 = (float*)myGpuMalloc(256 * sizeof(float));
float* x251 = x6+7179200;
CUDA_CALL(cudaMemcpy(x1007, x251, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1010 = (float*)myGpuMalloc(64 * sizeof(float));
float* x252 = x6+1792;
CUDA_CALL(cudaMemcpy(x1010, x252, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1013 = (float*)myGpuMalloc(128 * sizeof(float));
float* x253 = x6+401344;
CUDA_CALL(cudaMemcpy(x1013, x253, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1016 = (float*)myGpuMalloc(256 * sizeof(float));
float* x254 = x6+7708352;
CUDA_CALL(cudaMemcpy(x1016, x254, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1019 = (float*)myGpuMalloc(256 * sizeof(float));
float* x255 = x6+6588352;
CUDA_CALL(cudaMemcpy(x1019, x255, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1022 = (float*)myGpuMalloc(512 * sizeof(float));
float* x256 = x6+20134848;
CUDA_CALL(cudaMemcpy(x1022, x256, 512 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1025 = (float*)myGpuMalloc(65536 * sizeof(float));
float* x257 = x6+602560;
CUDA_CALL(cudaMemcpy(x1025, x257, 65536 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1028 = (float*)myGpuMalloc(64 * sizeof(float));
float* x258 = x6+165952;
CUDA_CALL(cudaMemcpy(x1028, x258, 64 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1031 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x259 = x6+469440;
CUDA_CALL(cudaMemcpy(x1031, x259, 131072 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1034 = (float*)myGpuMalloc(256 * sizeof(float));
float* x260 = x6+3227584;
CUDA_CALL(cudaMemcpy(x1034, x260, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1037 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x261 = x6+23549376;
CUDA_CALL(cudaMemcpy(x1037, x261, 2048 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1040 = (float*)myGpuMalloc(128 * sizeof(float));
float* x262 = x6+1231680;
CUDA_CALL(cudaMemcpy(x1040, x262, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1043 = (float*)myGpuMalloc(589824 * sizeof(float));
float* x263 = x6+6588864;
CUDA_CALL(cudaMemcpy(x1043, x263, 589824 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1046 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x264 = x6+5201344;
CUDA_CALL(cudaMemcpy(x1046, x264, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1049 = (float*)myGpuMalloc(256 * sizeof(float));
float* x265 = x6+77760;
CUDA_CALL(cudaMemcpy(x1049, x265, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1052 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x266 = x6+19084736;
CUDA_CALL(cudaMemcpy(x1052, x266, 1048576 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1055 = (float*)myGpuMalloc(128 * sizeof(float));
float* x267 = x6+1098048;
CUDA_CALL(cudaMemcpy(x1055, x267, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1058 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x268 = x6+2435520;
CUDA_CALL(cudaMemcpy(x1058, x268, 1024 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1061 = (float*)myGpuMalloc(128 * sizeof(float));
float* x269 = x6+1379520;
CUDA_CALL(cudaMemcpy(x1061, x269, 128 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1064 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x270 = x6+2170304;
CUDA_CALL(cudaMemcpy(x1064, x270, 262144 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1067 = (float*)myGpuMalloc(256 * sizeof(float));
float* x271 = x6+1578432;
CUDA_CALL(cudaMemcpy(x1067, x271, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1070 = (float*)myGpuMalloc(256 * sizeof(float));
float* x272 = x6+3817920;
CUDA_CALL(cudaMemcpy(x1070, x272, 256 * sizeof(float), cudaMemcpyHostToDevice));
// Tensor 'toGPU' invocation.
float* x1073 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x273 = x6+7444928;
CUDA_CALL(cudaMemcpy(x1073, x273, 1024 * sizeof(float), cudaMemcpyHostToDevice));
int32_t x1075 = open("../../cifar10_data/cifar-10-batches-bin/data_batch_1.bin",0);
int32_t x1076 = fsize(x1075);
int64_t x1078 = (int64_t)x1076;
int64_t x1079 = x1078 / 3073LL;
int32_t x1080 = (int32_t)x1079;
int32_t x1081 = x1080 * 3072;
float* x1082 = (float*)myMalloc(x1081 * sizeof(float));;
int* x1083 = (int32_t*)myMalloc(x1080 * sizeof(int32_t));;
char* x1077 = (char*)mmap(0, x1076, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x1075, 0);
for(int x1085=0; x1085 < x1080; x1085++) {
int32_t x1086 = x1085 * 3073;
char x1087 = x1077[x1086];
int32_t x1088 = (int32_t)(unsigned char)x1087;
x1083[x1085] = x1088;
int32_t x1094 = x1086 + 1;
int32_t x1092 = x1085 * 3072;
for(int x1091=0; x1091 < 3072; x1091++) {
int32_t x1095 = x1094 + x1091;
char x1096 = x1077[x1095];
int32_t x1093 = x1092 + x1091;
float x1097 = (float)(unsigned char)x1096;
float x1098 = x1097 / 255.0f;
x1082[x1093] = x1098;

}

}
int32_t x1104 = x1080 / 64;
for(int x1106=0; x1106 < x1104; x1106++) {
int32_t x1107 = x1106 * 64;
int32_t x1108 = x1107 * 3072;
float* x1109 = x1082+x1108;
int* x1110 = x1083+x1107;
printf("input (size 64 x 3 x 32 x 32)\n");
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
float* x1136 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1137 = (float*)myMalloc(1 * sizeof(float));;
x1137[0] = 0.0f;
float* x1139 = (float*)myMalloc(1 * sizeof(float));;
x1139[0] = 1.0f;

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
    x1139, in_desc, x1134, filt_desc, x713,
    conv_desc, algo, ws_data, ws_size,
    x1137, out_desc, x1136));
};
float* x1142 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1143 = (float*)myMalloc(1 * sizeof(float));;
x1143[0] = 0.0f;
float* x1145 = (float*)myMalloc(1 * sizeof(float));;
x1145[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1145, x1145, in_desc, x1136, out_desc, x1142, sbmv_desc, x875,
    x1010, x377, x587, 1.0E-5));
};
float* x1148 = (float*)myMalloc(1 * sizeof(float));;
x1148[0] = 0.0f;
float* x1150 = (float*)myMalloc(1 * sizeof(float));;
x1150[0] = 1.0f;
float* x1152 = (float*)myGpuMalloc(4194304 * sizeof(float));

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
    x1150, x_desc, x1142, x1148, x_desc, x1152));
};
float* x1154 = (float*)myMalloc(1 * sizeof(float));;
x1154[0] = 0.0f;
float* x1156 = (float*)myMalloc(1 * sizeof(float));;
x1156[0] = 1.0f;
float* x1158 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1156, in_desc, x1152, x1154, out_desc, x1158));
};
float* x1160 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1161 = (float*)myMalloc(1 * sizeof(float));;
x1161[0] = 0.0f;
float* x1163 = (float*)myMalloc(1 * sizeof(float));;
x1163[0] = 1.0f;

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
    x1163, in_desc, x1158, filt_desc, x956,
    conv_desc, algo, ws_data, ws_size,
    x1161, out_desc, x1160));
};
float* x1166 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1167 = (float*)myMalloc(1 * sizeof(float));;
x1167[0] = 0.0f;
float* x1169 = (float*)myMalloc(1 * sizeof(float));;
x1169[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1169, x1169, in_desc, x1160, out_desc, x1166, sbmv_desc, x335,
    x416, x599, x410, 1.0E-5));
};
float* x1172 = (float*)myMalloc(1 * sizeof(float));;
x1172[0] = 0.0f;
float* x1174 = (float*)myMalloc(1 * sizeof(float));;
x1174[0] = 1.0f;
float* x1176 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1174, x_desc, x1166, x1172, x_desc, x1176));
};
float* x1178 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1179 = (float*)myMalloc(1 * sizeof(float));;
x1179[0] = 0.0f;
float* x1181 = (float*)myMalloc(1 * sizeof(float));;
x1181[0] = 1.0f;

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
    x1181, in_desc, x1176, filt_desc, x527,
    conv_desc, algo, ws_data, ws_size,
    x1179, out_desc, x1178));
};
float* x1184 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1185 = (float*)myMalloc(1 * sizeof(float));;
x1185[0] = 0.0f;
float* x1187 = (float*)myMalloc(1 * sizeof(float));;
x1187[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1187, x1187, in_desc, x1178, out_desc, x1184, sbmv_desc, x749,
    x404, x572, x731, 1.0E-5));
};
float* x1190 = (float*)myMalloc(1 * sizeof(float));;
x1190[0] = 0.0f;
float* x1192 = (float*)myMalloc(1 * sizeof(float));;
x1192[0] = 1.0f;
float* x1194 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1192, x_desc, x1184, x1190, x_desc, x1194));
};
float* x1196 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1197 = (float*)myMalloc(1 * sizeof(float));;
x1197[0] = 0.0f;
float* x1199 = (float*)myMalloc(1 * sizeof(float));;
x1199[0] = 1.0f;

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
    x1199, in_desc, x1194, filt_desc, x353,
    conv_desc, algo, ws_data, ws_size,
    x1197, out_desc, x1196));
};
float* x1202 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1203 = (float*)myMalloc(1 * sizeof(float));;
x1203[0] = 0.0f;
float* x1205 = (float*)myMalloc(1 * sizeof(float));;
x1205[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1205, x1205, in_desc, x1196, out_desc, x1202, sbmv_desc, x854,
    x635, x470, x365, 1.0E-5));
};
float* x1208 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1209 = (float*)myMalloc(1 * sizeof(float));;
x1209[0] = 0.0f;
float* x1211 = (float*)myMalloc(1 * sizeof(float));;
x1211[0] = 1.0f;

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
    x1211, in_desc, x1158, filt_desc, x743,
    conv_desc, algo, ws_data, ws_size,
    x1209, out_desc, x1208));
};
float* x1214 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1215 = (float*)myMalloc(1 * sizeof(float));;
x1215[0] = 0.0f;
float* x1217 = (float*)myMalloc(1 * sizeof(float));;
x1217[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1217, x1217, in_desc, x1208, out_desc, x1214, sbmv_desc, x485,
    x866, x1049, x986, 1.0E-5));
};
float* x1220 = (float*)myMalloc(1 * sizeof(float));;
x1220[0] = 1.0f;
float* x1222 = (float*)myMalloc(1 * sizeof(float));;
x1222[0] = 1.0f;

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
    cudnnHandle, x1220, bias_desc, x1214, x1222, out_desc, x1202));
};
float* x1225 = (float*)myMalloc(1 * sizeof(float));;
x1225[0] = 0.0f;
float* x1227 = (float*)myMalloc(1 * sizeof(float));;
x1227[0] = 1.0f;
float* x1229 = (float*)myGpuMalloc(4194304 * sizeof(float));

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
    x1227, x_desc, x1202, x1225, x_desc, x1229));
};
float* x1231 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1232 = (float*)myMalloc(1 * sizeof(float));;
x1232[0] = 0.0f;
float* x1234 = (float*)myMalloc(1 * sizeof(float));;
x1234[0] = 1.0f;

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
    x1234, in_desc, x1229, filt_desc, x770,
    conv_desc, algo, ws_data, ws_size,
    x1232, out_desc, x1231));
};
float* x1237 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1238 = (float*)myMalloc(1 * sizeof(float));;
x1238[0] = 0.0f;
float* x1240 = (float*)myMalloc(1 * sizeof(float));;
x1240[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1240, x1240, in_desc, x1231, out_desc, x1237, sbmv_desc, x683,
    x437, x287, x563, 1.0E-5));
};
float* x1243 = (float*)myMalloc(1 * sizeof(float));;
x1243[0] = 0.0f;
float* x1245 = (float*)myMalloc(1 * sizeof(float));;
x1245[0] = 1.0f;
float* x1247 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1245, x_desc, x1237, x1243, x_desc, x1247));
};
float* x1249 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1250 = (float*)myMalloc(1 * sizeof(float));;
x1250[0] = 0.0f;
float* x1252 = (float*)myMalloc(1 * sizeof(float));;
x1252[0] = 1.0f;

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
    x1252, in_desc, x1247, filt_desc, x506,
    conv_desc, algo, ws_data, ws_size,
    x1250, out_desc, x1249));
};
float* x1255 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1256 = (float*)myMalloc(1 * sizeof(float));;
x1256[0] = 0.0f;
float* x1258 = (float*)myMalloc(1 * sizeof(float));;
x1258[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1258, x1258, in_desc, x1249, out_desc, x1255, sbmv_desc, x881,
    x716, x389, x989, 1.0E-5));
};
float* x1261 = (float*)myMalloc(1 * sizeof(float));;
x1261[0] = 0.0f;
float* x1263 = (float*)myMalloc(1 * sizeof(float));;
x1263[0] = 1.0f;
float* x1265 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1263, x_desc, x1255, x1261, x_desc, x1265));
};
float* x1267 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1268 = (float*)myMalloc(1 * sizeof(float));;
x1268[0] = 0.0f;
float* x1270 = (float*)myMalloc(1 * sizeof(float));;
x1270[0] = 1.0f;

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
    x1270, in_desc, x1265, filt_desc, x647,
    conv_desc, algo, ws_data, ws_size,
    x1268, out_desc, x1267));
};
float* x1273 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1274 = (float*)myMalloc(1 * sizeof(float));;
x1274[0] = 0.0f;
float* x1276 = (float*)myMalloc(1 * sizeof(float));;
x1276[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1276, x1276, in_desc, x1267, out_desc, x1273, sbmv_desc, x431,
    x278, x530, x755, 1.0E-5));
};
float* x1279 = (float*)myMalloc(1 * sizeof(float));;
x1279[0] = 1.0f;
float* x1281 = (float*)myMalloc(1 * sizeof(float));;
x1281[0] = 1.0f;

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
    cudnnHandle, x1279, bias_desc, x1229, x1281, out_desc, x1273));
};
float* x1284 = (float*)myMalloc(1 * sizeof(float));;
x1284[0] = 0.0f;
float* x1286 = (float*)myMalloc(1 * sizeof(float));;
x1286[0] = 1.0f;
float* x1288 = (float*)myGpuMalloc(4194304 * sizeof(float));

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
    x1286, x_desc, x1273, x1284, x_desc, x1288));
};
float* x1290 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1291 = (float*)myMalloc(1 * sizeof(float));;
x1291[0] = 0.0f;
float* x1293 = (float*)myMalloc(1 * sizeof(float));;
x1293[0] = 1.0f;

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
    x1293, in_desc, x1288, filt_desc, x707,
    conv_desc, algo, ws_data, ws_size,
    x1291, out_desc, x1290));
};
float* x1296 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1297 = (float*)myMalloc(1 * sizeof(float));;
x1297[0] = 0.0f;
float* x1299 = (float*)myMalloc(1 * sizeof(float));;
x1299[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1299, x1299, in_desc, x1290, out_desc, x1296, sbmv_desc, x500,
    x329, x1028, x818, 1.0E-5));
};
float* x1302 = (float*)myMalloc(1 * sizeof(float));;
x1302[0] = 0.0f;
float* x1304 = (float*)myMalloc(1 * sizeof(float));;
x1304[0] = 1.0f;
float* x1306 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1304, x_desc, x1296, x1302, x_desc, x1306));
};
float* x1308 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1309 = (float*)myMalloc(1 * sizeof(float));;
x1309[0] = 0.0f;
float* x1311 = (float*)myMalloc(1 * sizeof(float));;
x1311[0] = 1.0f;

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
    x1311, in_desc, x1306, filt_desc, x476,
    conv_desc, algo, ws_data, ws_size,
    x1309, out_desc, x1308));
};
float* x1314 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1315 = (float*)myMalloc(1 * sizeof(float));;
x1315[0] = 0.0f;
float* x1317 = (float*)myMalloc(1 * sizeof(float));;
x1317[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1317, x1317, in_desc, x1308, out_desc, x1314, sbmv_desc, x473,
    x662, x794, x611, 1.0E-5));
};
float* x1320 = (float*)myMalloc(1 * sizeof(float));;
x1320[0] = 0.0f;
float* x1322 = (float*)myMalloc(1 * sizeof(float));;
x1322[0] = 1.0f;
float* x1324 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1322, x_desc, x1314, x1320, x_desc, x1324));
};
float* x1326 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1327 = (float*)myMalloc(1 * sizeof(float));;
x1327[0] = 0.0f;
float* x1329 = (float*)myMalloc(1 * sizeof(float));;
x1329[0] = 1.0f;

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
    x1329, in_desc, x1324, filt_desc, x518,
    conv_desc, algo, ws_data, ws_size,
    x1327, out_desc, x1326));
};
float* x1332 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1333 = (float*)myMalloc(1 * sizeof(float));;
x1333[0] = 0.0f;
float* x1335 = (float*)myMalloc(1 * sizeof(float));;
x1335[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1335, x1335, in_desc, x1326, out_desc, x1332, sbmv_desc, x368,
    x998, x809, x656, 1.0E-5));
};
float* x1338 = (float*)myMalloc(1 * sizeof(float));;
x1338[0] = 1.0f;
float* x1340 = (float*)myMalloc(1 * sizeof(float));;
x1340[0] = 1.0f;

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
    cudnnHandle, x1338, bias_desc, x1288, x1340, out_desc, x1332));
};
float* x1343 = (float*)myMalloc(1 * sizeof(float));;
x1343[0] = 0.0f;
float* x1345 = (float*)myMalloc(1 * sizeof(float));;
x1345[0] = 1.0f;
float* x1347 = (float*)myGpuMalloc(4194304 * sizeof(float));

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
    x1345, x_desc, x1332, x1343, x_desc, x1347));
};
float* x1349 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1350 = (float*)myMalloc(1 * sizeof(float));;
x1350[0] = 0.0f;
float* x1352 = (float*)myMalloc(1 * sizeof(float));;
x1352[0] = 1.0f;

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
    x1352, in_desc, x1347, filt_desc, x290,
    conv_desc, algo, ws_data, ws_size,
    x1350, out_desc, x1349));
};
float* x1355 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1356 = (float*)myMalloc(1 * sizeof(float));;
x1356[0] = 0.0f;
float* x1358 = (float*)myMalloc(1 * sizeof(float));;
x1358[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1358, x1358, in_desc, x1349, out_desc, x1355, sbmv_desc, x509,
    x773, x869, x659, 1.0E-5));
};
float* x1361 = (float*)myMalloc(1 * sizeof(float));;
x1361[0] = 0.0f;
float* x1363 = (float*)myMalloc(1 * sizeof(float));;
x1363[0] = 1.0f;
float* x1365 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x1363, x_desc, x1355, x1361, x_desc, x1365));
};
float* x1367 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1368 = (float*)myMalloc(1 * sizeof(float));;
x1368[0] = 0.0f;
float* x1370 = (float*)myMalloc(1 * sizeof(float));;
x1370[0] = 1.0f;

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
    x1370, in_desc, x1365, filt_desc, x338,
    conv_desc, algo, ws_data, ws_size,
    x1368, out_desc, x1367));
};
float* x1373 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1374 = (float*)myMalloc(1 * sizeof(float));;
x1374[0] = 0.0f;
float* x1376 = (float*)myMalloc(1 * sizeof(float));;
x1376[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1376, x1376, in_desc, x1367, out_desc, x1373, sbmv_desc, x1013,
    x827, x641, x386, 1.0E-5));
};
float* x1379 = (float*)myMalloc(1 * sizeof(float));;
x1379[0] = 0.0f;
float* x1381 = (float*)myMalloc(1 * sizeof(float));;
x1381[0] = 1.0f;
float* x1383 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1381, x_desc, x1373, x1379, x_desc, x1383));
};
float* x1385 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1386 = (float*)myMalloc(1 * sizeof(float));;
x1386[0] = 0.0f;
float* x1388 = (float*)myMalloc(1 * sizeof(float));;
x1388[0] = 1.0f;

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
    x1388, in_desc, x1383, filt_desc, x575,
    conv_desc, algo, ws_data, ws_size,
    x1386, out_desc, x1385));
};
float* x1391 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1392 = (float*)myMalloc(1 * sizeof(float));;
x1392[0] = 0.0f;
float* x1394 = (float*)myMalloc(1 * sizeof(float));;
x1394[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1394, x1394, in_desc, x1385, out_desc, x1391, sbmv_desc, x692,
    x887, x704, x560, 1.0E-5));
};
float* x1397 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1398 = (float*)myMalloc(1 * sizeof(float));;
x1398[0] = 0.0f;
float* x1400 = (float*)myMalloc(1 * sizeof(float));;
x1400[0] = 1.0f;

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
    x1400, in_desc, x1347, filt_desc, x1031,
    conv_desc, algo, ws_data, ws_size,
    x1398, out_desc, x1397));
};
float* x1403 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1404 = (float*)myMalloc(1 * sizeof(float));;
x1404[0] = 0.0f;
float* x1406 = (float*)myMalloc(1 * sizeof(float));;
x1406[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1406, x1406, in_desc, x1397, out_desc, x1403, sbmv_desc, x878,
    x614, x383, x326, 1.0E-5));
};
float* x1409 = (float*)myMalloc(1 * sizeof(float));;
x1409[0] = 1.0f;
float* x1411 = (float*)myMalloc(1 * sizeof(float));;
x1411[0] = 1.0f;

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
    cudnnHandle, x1409, bias_desc, x1403, x1411, out_desc, x1391));
};
float* x1414 = (float*)myMalloc(1 * sizeof(float));;
x1414[0] = 0.0f;
float* x1416 = (float*)myMalloc(1 * sizeof(float));;
x1416[0] = 1.0f;
float* x1418 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x1416, x_desc, x1391, x1414, x_desc, x1418));
};
float* x1420 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1421 = (float*)myMalloc(1 * sizeof(float));;
x1421[0] = 0.0f;
float* x1423 = (float*)myMalloc(1 * sizeof(float));;
x1423[0] = 1.0f;

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
    x1423, in_desc, x1418, filt_desc, x1025,
    conv_desc, algo, ws_data, ws_size,
    x1421, out_desc, x1420));
};
float* x1426 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1427 = (float*)myMalloc(1 * sizeof(float));;
x1427[0] = 0.0f;
float* x1429 = (float*)myMalloc(1 * sizeof(float));;
x1429[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1429, x1429, in_desc, x1420, out_desc, x1426, sbmv_desc, x923,
    x308, x557, x788, 1.0E-5));
};
float* x1432 = (float*)myMalloc(1 * sizeof(float));;
x1432[0] = 0.0f;
float* x1434 = (float*)myMalloc(1 * sizeof(float));;
x1434[0] = 1.0f;
float* x1436 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1434, x_desc, x1426, x1432, x_desc, x1436));
};
float* x1438 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1439 = (float*)myMalloc(1 * sizeof(float));;
x1439[0] = 0.0f;
float* x1441 = (float*)myMalloc(1 * sizeof(float));;
x1441[0] = 1.0f;

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
    x1441, in_desc, x1436, filt_desc, x962,
    conv_desc, algo, ws_data, ws_size,
    x1439, out_desc, x1438));
};
float* x1444 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1445 = (float*)myMalloc(1 * sizeof(float));;
x1445[0] = 0.0f;
float* x1447 = (float*)myMalloc(1 * sizeof(float));;
x1447[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1447, x1447, in_desc, x1438, out_desc, x1444, sbmv_desc, x281,
    x542, x362, x932, 1.0E-5));
};
float* x1450 = (float*)myMalloc(1 * sizeof(float));;
x1450[0] = 0.0f;
float* x1452 = (float*)myMalloc(1 * sizeof(float));;
x1452[0] = 1.0f;
float* x1454 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1452, x_desc, x1444, x1450, x_desc, x1454));
};
float* x1456 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1457 = (float*)myMalloc(1 * sizeof(float));;
x1457[0] = 0.0f;
float* x1459 = (float*)myMalloc(1 * sizeof(float));;
x1459[0] = 1.0f;

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
    x1459, in_desc, x1454, filt_desc, x590,
    conv_desc, algo, ws_data, ws_size,
    x1457, out_desc, x1456));
};
float* x1462 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1463 = (float*)myMalloc(1 * sizeof(float));;
x1463[0] = 0.0f;
float* x1465 = (float*)myMalloc(1 * sizeof(float));;
x1465[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1465, x1465, in_desc, x1456, out_desc, x1462, sbmv_desc, x413,
    x995, x698, x521, 1.0E-5));
};
float* x1468 = (float*)myMalloc(1 * sizeof(float));;
x1468[0] = 1.0f;
float* x1470 = (float*)myMalloc(1 * sizeof(float));;
x1470[0] = 1.0f;

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
    cudnnHandle, x1468, bias_desc, x1418, x1470, out_desc, x1462));
};
float* x1473 = (float*)myMalloc(1 * sizeof(float));;
x1473[0] = 0.0f;
float* x1475 = (float*)myMalloc(1 * sizeof(float));;
x1475[0] = 1.0f;
float* x1477 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x1475, x_desc, x1462, x1473, x_desc, x1477));
};
float* x1479 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1480 = (float*)myMalloc(1 * sizeof(float));;
x1480[0] = 0.0f;
float* x1482 = (float*)myMalloc(1 * sizeof(float));;
x1482[0] = 1.0f;

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
    x1482, in_desc, x1477, filt_desc, x845,
    conv_desc, algo, ws_data, ws_size,
    x1480, out_desc, x1479));
};
float* x1485 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1486 = (float*)myMalloc(1 * sizeof(float));;
x1486[0] = 0.0f;
float* x1488 = (float*)myMalloc(1 * sizeof(float));;
x1488[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1488, x1488, in_desc, x1479, out_desc, x1485, sbmv_desc, x392,
    x767, x593, x284, 1.0E-5));
};
float* x1491 = (float*)myMalloc(1 * sizeof(float));;
x1491[0] = 0.0f;
float* x1493 = (float*)myMalloc(1 * sizeof(float));;
x1493[0] = 1.0f;
float* x1495 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1493, x_desc, x1485, x1491, x_desc, x1495));
};
float* x1497 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1498 = (float*)myMalloc(1 * sizeof(float));;
x1498[0] = 0.0f;
float* x1500 = (float*)myMalloc(1 * sizeof(float));;
x1500[0] = 1.0f;

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
    x1500, in_desc, x1495, filt_desc, x830,
    conv_desc, algo, ws_data, ws_size,
    x1498, out_desc, x1497));
};
float* x1503 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1504 = (float*)myMalloc(1 * sizeof(float));;
x1504[0] = 0.0f;
float* x1506 = (float*)myMalloc(1 * sizeof(float));;
x1506[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1506, x1506, in_desc, x1497, out_desc, x1503, sbmv_desc, x638,
    x440, x908, x1055, 1.0E-5));
};
float* x1509 = (float*)myMalloc(1 * sizeof(float));;
x1509[0] = 0.0f;
float* x1511 = (float*)myMalloc(1 * sizeof(float));;
x1511[0] = 1.0f;
float* x1513 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1511, x_desc, x1503, x1509, x_desc, x1513));
};
float* x1515 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1516 = (float*)myMalloc(1 * sizeof(float));;
x1516[0] = 0.0f;
float* x1518 = (float*)myMalloc(1 * sizeof(float));;
x1518[0] = 1.0f;

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
    x1518, in_desc, x1513, filt_desc, x380,
    conv_desc, algo, ws_data, ws_size,
    x1516, out_desc, x1515));
};
float* x1521 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1522 = (float*)myMalloc(1 * sizeof(float));;
x1522[0] = 0.0f;
float* x1524 = (float*)myMalloc(1 * sizeof(float));;
x1524[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1524, x1524, in_desc, x1515, out_desc, x1521, sbmv_desc, x758,
    x503, x332, x926, 1.0E-5));
};
float* x1527 = (float*)myMalloc(1 * sizeof(float));;
x1527[0] = 1.0f;
float* x1529 = (float*)myMalloc(1 * sizeof(float));;
x1529[0] = 1.0f;

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
    cudnnHandle, x1527, bias_desc, x1477, x1529, out_desc, x1521));
};
float* x1532 = (float*)myMalloc(1 * sizeof(float));;
x1532[0] = 0.0f;
float* x1534 = (float*)myMalloc(1 * sizeof(float));;
x1534[0] = 1.0f;
float* x1536 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x1534, x_desc, x1521, x1532, x_desc, x1536));
};
float* x1538 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1539 = (float*)myMalloc(1 * sizeof(float));;
x1539[0] = 0.0f;
float* x1541 = (float*)myMalloc(1 * sizeof(float));;
x1541[0] = 1.0f;

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
    x1541, in_desc, x1536, filt_desc, x653,
    conv_desc, algo, ws_data, ws_size,
    x1539, out_desc, x1538));
};
float* x1544 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1545 = (float*)myMalloc(1 * sizeof(float));;
x1545[0] = 0.0f;
float* x1547 = (float*)myMalloc(1 * sizeof(float));;
x1547[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1547, x1547, in_desc, x1538, out_desc, x1544, sbmv_desc, x374,
    x983, x965, x1040, 1.0E-5));
};
float* x1550 = (float*)myMalloc(1 * sizeof(float));;
x1550[0] = 0.0f;
float* x1552 = (float*)myMalloc(1 * sizeof(float));;
x1552[0] = 1.0f;
float* x1554 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1552, x_desc, x1544, x1550, x_desc, x1554));
};
float* x1556 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1557 = (float*)myMalloc(1 * sizeof(float));;
x1557[0] = 0.0f;
float* x1559 = (float*)myMalloc(1 * sizeof(float));;
x1559[0] = 1.0f;

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
    x1559, in_desc, x1554, filt_desc, x752,
    conv_desc, algo, ws_data, ws_size,
    x1557, out_desc, x1556));
};
float* x1562 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1563 = (float*)myMalloc(1 * sizeof(float));;
x1563[0] = 0.0f;
float* x1565 = (float*)myMalloc(1 * sizeof(float));;
x1565[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1565, x1565, in_desc, x1556, out_desc, x1562, sbmv_desc, x494,
    x371, x1061, x701, 1.0E-5));
};
float* x1568 = (float*)myMalloc(1 * sizeof(float));;
x1568[0] = 0.0f;
float* x1570 = (float*)myMalloc(1 * sizeof(float));;
x1570[0] = 1.0f;
float* x1572 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1570, x_desc, x1562, x1568, x_desc, x1572));
};
float* x1574 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1575 = (float*)myMalloc(1 * sizeof(float));;
x1575[0] = 0.0f;
float* x1577 = (float*)myMalloc(1 * sizeof(float));;
x1577[0] = 1.0f;

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
    x1577, in_desc, x1572, filt_desc, x422,
    conv_desc, algo, ws_data, ws_size,
    x1575, out_desc, x1574));
};
float* x1580 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1581 = (float*)myMalloc(1 * sizeof(float));;
x1581[0] = 0.0f;
float* x1583 = (float*)myMalloc(1 * sizeof(float));;
x1583[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1583, x1583, in_desc, x1574, out_desc, x1580, sbmv_desc, x725,
    x419, x314, x959, 1.0E-5));
};
float* x1586 = (float*)myMalloc(1 * sizeof(float));;
x1586[0] = 1.0f;
float* x1588 = (float*)myMalloc(1 * sizeof(float));;
x1588[0] = 1.0f;

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
    cudnnHandle, x1586, bias_desc, x1536, x1588, out_desc, x1580));
};
float* x1591 = (float*)myMalloc(1 * sizeof(float));;
x1591[0] = 0.0f;
float* x1593 = (float*)myMalloc(1 * sizeof(float));;
x1593[0] = 1.0f;
float* x1595 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x1593, x_desc, x1580, x1591, x_desc, x1595));
};
float* x1597 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1598 = (float*)myMalloc(1 * sizeof(float));;
x1598[0] = 0.0f;
float* x1600 = (float*)myMalloc(1 * sizeof(float));;
x1600[0] = 1.0f;

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
    x1600, in_desc, x1595, filt_desc, x797,
    conv_desc, algo, ws_data, ws_size,
    x1598, out_desc, x1597));
};
float* x1603 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1604 = (float*)myMalloc(1 * sizeof(float));;
x1604[0] = 0.0f;
float* x1606 = (float*)myMalloc(1 * sizeof(float));;
x1606[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1606, x1606, in_desc, x1597, out_desc, x1603, sbmv_desc, x1067,
    x320, x650, x851, 1.0E-5));
};
float* x1609 = (float*)myMalloc(1 * sizeof(float));;
x1609[0] = 0.0f;
float* x1611 = (float*)myMalloc(1 * sizeof(float));;
x1611[0] = 1.0f;
float* x1613 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1611, x_desc, x1603, x1609, x_desc, x1613));
};
float* x1615 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1616 = (float*)myMalloc(1 * sizeof(float));;
x1616[0] = 0.0f;
float* x1618 = (float*)myMalloc(1 * sizeof(float));;
x1618[0] = 1.0f;

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
    x1618, in_desc, x1613, filt_desc, x782,
    conv_desc, algo, ws_data, ws_size,
    x1616, out_desc, x1615));
};
float* x1621 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1622 = (float*)myMalloc(1 * sizeof(float));;
x1622[0] = 0.0f;
float* x1624 = (float*)myMalloc(1 * sizeof(float));;
x1624[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1624, x1624, in_desc, x1615, out_desc, x1621, sbmv_desc, x581,
    x305, x944, x554, 1.0E-5));
};
float* x1627 = (float*)myMalloc(1 * sizeof(float));;
x1627[0] = 0.0f;
float* x1629 = (float*)myMalloc(1 * sizeof(float));;
x1629[0] = 1.0f;
float* x1631 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1629, x_desc, x1621, x1627, x_desc, x1631));
};
float* x1633 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1634 = (float*)myMalloc(1 * sizeof(float));;
x1634[0] = 0.0f;
float* x1636 = (float*)myMalloc(1 * sizeof(float));;
x1636[0] = 1.0f;

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
    x1636, in_desc, x1631, filt_desc, x1064,
    conv_desc, algo, ws_data, ws_size,
    x1634, out_desc, x1633));
};
float* x1639 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1640 = (float*)myMalloc(1 * sizeof(float));;
x1640[0] = 0.0f;
float* x1642 = (float*)myMalloc(1 * sizeof(float));;
x1642[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1642, x1642, in_desc, x1633, out_desc, x1639, sbmv_desc, x311,
    x608, x905, x1058, 1.0E-5));
};
float* x1645 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1646 = (float*)myMalloc(1 * sizeof(float));;
x1646[0] = 0.0f;
float* x1648 = (float*)myMalloc(1 * sizeof(float));;
x1648[0] = 1.0f;

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
    x1648, in_desc, x1595, filt_desc, x482,
    conv_desc, algo, ws_data, ws_size,
    x1646, out_desc, x1645));
};
float* x1651 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1652 = (float*)myMalloc(1 * sizeof(float));;
x1652[0] = 0.0f;
float* x1654 = (float*)myMalloc(1 * sizeof(float));;
x1654[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1654, x1654, in_desc, x1645, out_desc, x1651, sbmv_desc, x344,
    x917, x515, x890, 1.0E-5));
};
float* x1657 = (float*)myMalloc(1 * sizeof(float));;
x1657[0] = 1.0f;
float* x1659 = (float*)myMalloc(1 * sizeof(float));;
x1659[0] = 1.0f;

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
    cudnnHandle, x1657, bias_desc, x1651, x1659, out_desc, x1639));
};
float* x1662 = (float*)myMalloc(1 * sizeof(float));;
x1662[0] = 0.0f;
float* x1664 = (float*)myMalloc(1 * sizeof(float));;
x1664[0] = 1.0f;
float* x1666 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1664, x_desc, x1639, x1662, x_desc, x1666));
};
float* x1668 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1669 = (float*)myMalloc(1 * sizeof(float));;
x1669[0] = 0.0f;
float* x1671 = (float*)myMalloc(1 * sizeof(float));;
x1671[0] = 1.0f;

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
    x1671, in_desc, x1666, filt_desc, x296,
    conv_desc, algo, ws_data, ws_size,
    x1669, out_desc, x1668));
};
float* x1674 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1675 = (float*)myMalloc(1 * sizeof(float));;
x1675[0] = 0.0f;
float* x1677 = (float*)myMalloc(1 * sizeof(float));;
x1677[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1677, x1677, in_desc, x1668, out_desc, x1674, sbmv_desc, x347,
    x914, x1034, x728, 1.0E-5));
};
float* x1680 = (float*)myMalloc(1 * sizeof(float));;
x1680[0] = 0.0f;
float* x1682 = (float*)myMalloc(1 * sizeof(float));;
x1682[0] = 1.0f;
float* x1684 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1682, x_desc, x1674, x1680, x_desc, x1684));
};
float* x1686 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1687 = (float*)myMalloc(1 * sizeof(float));;
x1687[0] = 0.0f;
float* x1689 = (float*)myMalloc(1 * sizeof(float));;
x1689[0] = 1.0f;

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
    x1689, in_desc, x1684, filt_desc, x350,
    conv_desc, algo, ws_data, ws_size,
    x1687, out_desc, x1686));
};
float* x1692 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1693 = (float*)myMalloc(1 * sizeof(float));;
x1693[0] = 0.0f;
float* x1695 = (float*)myMalloc(1 * sizeof(float));;
x1695[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1695, x1695, in_desc, x1686, out_desc, x1692, sbmv_desc, x1070,
    x545, x857, x968, 1.0E-5));
};
float* x1698 = (float*)myMalloc(1 * sizeof(float));;
x1698[0] = 0.0f;
float* x1700 = (float*)myMalloc(1 * sizeof(float));;
x1700[0] = 1.0f;
float* x1702 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1700, x_desc, x1692, x1698, x_desc, x1702));
};
float* x1704 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1705 = (float*)myMalloc(1 * sizeof(float));;
x1705[0] = 0.0f;
float* x1707 = (float*)myMalloc(1 * sizeof(float));;
x1707[0] = 1.0f;

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
    x1707, in_desc, x1702, filt_desc, x425,
    conv_desc, algo, ws_data, ws_size,
    x1705, out_desc, x1704));
};
float* x1710 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1711 = (float*)myMalloc(1 * sizeof(float));;
x1711[0] = 0.0f;
float* x1713 = (float*)myMalloc(1 * sizeof(float));;
x1713[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1713, x1713, in_desc, x1704, out_desc, x1710, sbmv_desc, x317,
    x953, x803, x686, 1.0E-5));
};
float* x1716 = (float*)myMalloc(1 * sizeof(float));;
x1716[0] = 1.0f;
float* x1718 = (float*)myMalloc(1 * sizeof(float));;
x1718[0] = 1.0f;

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
    cudnnHandle, x1716, bias_desc, x1666, x1718, out_desc, x1710));
};
float* x1721 = (float*)myMalloc(1 * sizeof(float));;
x1721[0] = 0.0f;
float* x1723 = (float*)myMalloc(1 * sizeof(float));;
x1723[0] = 1.0f;
float* x1725 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1723, x_desc, x1710, x1721, x_desc, x1725));
};
float* x1727 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1728 = (float*)myMalloc(1 * sizeof(float));;
x1728[0] = 0.0f;
float* x1730 = (float*)myMalloc(1 * sizeof(float));;
x1730[0] = 1.0f;

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
    x1730, in_desc, x1725, filt_desc, x911,
    conv_desc, algo, ws_data, ws_size,
    x1728, out_desc, x1727));
};
float* x1733 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1734 = (float*)myMalloc(1 * sizeof(float));;
x1734[0] = 0.0f;
float* x1736 = (float*)myMalloc(1 * sizeof(float));;
x1736[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1736, x1736, in_desc, x1727, out_desc, x1733, sbmv_desc, x644,
    x848, x791, x779, 1.0E-5));
};
float* x1739 = (float*)myMalloc(1 * sizeof(float));;
x1739[0] = 0.0f;
float* x1741 = (float*)myMalloc(1 * sizeof(float));;
x1741[0] = 1.0f;
float* x1743 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1741, x_desc, x1733, x1739, x_desc, x1743));
};
float* x1745 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1746 = (float*)myMalloc(1 * sizeof(float));;
x1746[0] = 0.0f;
float* x1748 = (float*)myMalloc(1 * sizeof(float));;
x1748[0] = 1.0f;

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
    x1748, in_desc, x1743, filt_desc, x299,
    conv_desc, algo, ws_data, ws_size,
    x1746, out_desc, x1745));
};
float* x1751 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1752 = (float*)myMalloc(1 * sizeof(float));;
x1752[0] = 0.0f;
float* x1754 = (float*)myMalloc(1 * sizeof(float));;
x1754[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1754, x1754, in_desc, x1745, out_desc, x1751, sbmv_desc, x941,
    x833, x629, x446, 1.0E-5));
};
float* x1757 = (float*)myMalloc(1 * sizeof(float));;
x1757[0] = 0.0f;
float* x1759 = (float*)myMalloc(1 * sizeof(float));;
x1759[0] = 1.0f;
float* x1761 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1759, x_desc, x1751, x1757, x_desc, x1761));
};
float* x1763 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1764 = (float*)myMalloc(1 * sizeof(float));;
x1764[0] = 0.0f;
float* x1766 = (float*)myMalloc(1 * sizeof(float));;
x1766[0] = 1.0f;

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
    x1766, in_desc, x1761, filt_desc, x605,
    conv_desc, algo, ws_data, ws_size,
    x1764, out_desc, x1763));
};
float* x1769 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1770 = (float*)myMalloc(1 * sizeof(float));;
x1770[0] = 0.0f;
float* x1772 = (float*)myMalloc(1 * sizeof(float));;
x1772[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1772, x1772, in_desc, x1763, out_desc, x1769, sbmv_desc, x1046,
    x428, x677, x821, 1.0E-5));
};
float* x1775 = (float*)myMalloc(1 * sizeof(float));;
x1775[0] = 1.0f;
float* x1777 = (float*)myMalloc(1 * sizeof(float));;
x1777[0] = 1.0f;

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
    cudnnHandle, x1775, bias_desc, x1725, x1777, out_desc, x1769));
};
float* x1780 = (float*)myMalloc(1 * sizeof(float));;
x1780[0] = 0.0f;
float* x1782 = (float*)myMalloc(1 * sizeof(float));;
x1782[0] = 1.0f;
float* x1784 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1782, x_desc, x1769, x1780, x_desc, x1784));
};
float* x1786 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1787 = (float*)myMalloc(1 * sizeof(float));;
x1787[0] = 0.0f;
float* x1789 = (float*)myMalloc(1 * sizeof(float));;
x1789[0] = 1.0f;

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
    x1789, in_desc, x1784, filt_desc, x275,
    conv_desc, algo, ws_data, ws_size,
    x1787, out_desc, x1786));
};
float* x1792 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1793 = (float*)myMalloc(1 * sizeof(float));;
x1793[0] = 0.0f;
float* x1795 = (float*)myMalloc(1 * sizeof(float));;
x1795[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1795, x1795, in_desc, x1786, out_desc, x1792, sbmv_desc, x533,
    x980, x746, x551, 1.0E-5));
};
float* x1798 = (float*)myMalloc(1 * sizeof(float));;
x1798[0] = 0.0f;
float* x1800 = (float*)myMalloc(1 * sizeof(float));;
x1800[0] = 1.0f;
float* x1802 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1800, x_desc, x1792, x1798, x_desc, x1802));
};
float* x1804 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1805 = (float*)myMalloc(1 * sizeof(float));;
x1805[0] = 0.0f;
float* x1807 = (float*)myMalloc(1 * sizeof(float));;
x1807[0] = 1.0f;

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
    x1807, in_desc, x1802, filt_desc, x1004,
    conv_desc, algo, ws_data, ws_size,
    x1805, out_desc, x1804));
};
float* x1810 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1811 = (float*)myMalloc(1 * sizeof(float));;
x1811[0] = 0.0f;
float* x1813 = (float*)myMalloc(1 * sizeof(float));;
x1813[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1813, x1813, in_desc, x1804, out_desc, x1810, sbmv_desc, x479,
    x665, x815, x947, 1.0E-5));
};
float* x1816 = (float*)myMalloc(1 * sizeof(float));;
x1816[0] = 0.0f;
float* x1818 = (float*)myMalloc(1 * sizeof(float));;
x1818[0] = 1.0f;
float* x1820 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1818, x_desc, x1810, x1816, x_desc, x1820));
};
float* x1822 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1823 = (float*)myMalloc(1 * sizeof(float));;
x1823[0] = 0.0f;
float* x1825 = (float*)myMalloc(1 * sizeof(float));;
x1825[0] = 1.0f;

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
    x1825, in_desc, x1820, filt_desc, x524,
    conv_desc, algo, ws_data, ws_size,
    x1823, out_desc, x1822));
};
float* x1828 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1829 = (float*)myMalloc(1 * sizeof(float));;
x1829[0] = 0.0f;
float* x1831 = (float*)myMalloc(1 * sizeof(float));;
x1831[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1831, x1831, in_desc, x1822, out_desc, x1828, sbmv_desc, x971,
    x695, x950, x740, 1.0E-5));
};
float* x1834 = (float*)myMalloc(1 * sizeof(float));;
x1834[0] = 1.0f;
float* x1836 = (float*)myMalloc(1 * sizeof(float));;
x1836[0] = 1.0f;

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
    cudnnHandle, x1834, bias_desc, x1784, x1836, out_desc, x1828));
};
float* x1839 = (float*)myMalloc(1 * sizeof(float));;
x1839[0] = 0.0f;
float* x1841 = (float*)myMalloc(1 * sizeof(float));;
x1841[0] = 1.0f;
float* x1843 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1841, x_desc, x1828, x1839, x_desc, x1843));
};
float* x1845 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1846 = (float*)myMalloc(1 * sizeof(float));;
x1846[0] = 0.0f;
float* x1848 = (float*)myMalloc(1 * sizeof(float));;
x1848[0] = 1.0f;

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
    x1848, in_desc, x1843, filt_desc, x323,
    conv_desc, algo, ws_data, ws_size,
    x1846, out_desc, x1845));
};
float* x1851 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1852 = (float*)myMalloc(1 * sizeof(float));;
x1852[0] = 0.0f;
float* x1854 = (float*)myMalloc(1 * sizeof(float));;
x1854[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1854, x1854, in_desc, x1845, out_desc, x1851, sbmv_desc, x488,
    x812, x1019, x464, 1.0E-5));
};
float* x1857 = (float*)myMalloc(1 * sizeof(float));;
x1857[0] = 0.0f;
float* x1859 = (float*)myMalloc(1 * sizeof(float));;
x1859[0] = 1.0f;
float* x1861 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1859, x_desc, x1851, x1857, x_desc, x1861));
};
float* x1863 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1864 = (float*)myMalloc(1 * sizeof(float));;
x1864[0] = 0.0f;
float* x1866 = (float*)myMalloc(1 * sizeof(float));;
x1866[0] = 1.0f;

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
    x1866, in_desc, x1861, filt_desc, x1043,
    conv_desc, algo, ws_data, ws_size,
    x1864, out_desc, x1863));
};
float* x1869 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1870 = (float*)myMalloc(1 * sizeof(float));;
x1870[0] = 0.0f;
float* x1872 = (float*)myMalloc(1 * sizeof(float));;
x1872[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1872, x1872, in_desc, x1863, out_desc, x1869, sbmv_desc, x761,
    x584, x1007, x569, 1.0E-5));
};
float* x1875 = (float*)myMalloc(1 * sizeof(float));;
x1875[0] = 0.0f;
float* x1877 = (float*)myMalloc(1 * sizeof(float));;
x1877[0] = 1.0f;
float* x1879 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1877, x_desc, x1869, x1875, x_desc, x1879));
};
float* x1881 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1882 = (float*)myMalloc(1 * sizeof(float));;
x1882[0] = 0.0f;
float* x1884 = (float*)myMalloc(1 * sizeof(float));;
x1884[0] = 1.0f;

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
    x1884, in_desc, x1879, filt_desc, x920,
    conv_desc, algo, ws_data, ws_size,
    x1882, out_desc, x1881));
};
float* x1887 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1888 = (float*)myMalloc(1 * sizeof(float));;
x1888[0] = 0.0f;
float* x1890 = (float*)myMalloc(1 * sizeof(float));;
x1890[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1890, x1890, in_desc, x1881, out_desc, x1887, sbmv_desc, x434,
    x617, x884, x1073, 1.0E-5));
};
float* x1893 = (float*)myMalloc(1 * sizeof(float));;
x1893[0] = 1.0f;
float* x1895 = (float*)myMalloc(1 * sizeof(float));;
x1895[0] = 1.0f;

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
    cudnnHandle, x1893, bias_desc, x1843, x1895, out_desc, x1887));
};
float* x1898 = (float*)myMalloc(1 * sizeof(float));;
x1898[0] = 0.0f;
float* x1900 = (float*)myMalloc(1 * sizeof(float));;
x1900[0] = 1.0f;
float* x1902 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1900, x_desc, x1887, x1898, x_desc, x1902));
};
float* x1904 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1905 = (float*)myMalloc(1 * sizeof(float));;
x1905[0] = 0.0f;
float* x1907 = (float*)myMalloc(1 * sizeof(float));;
x1907[0] = 1.0f;

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
    x1907, in_desc, x1902, filt_desc, x710,
    conv_desc, algo, ws_data, ws_size,
    x1905, out_desc, x1904));
};
float* x1910 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1911 = (float*)myMalloc(1 * sizeof(float));;
x1911[0] = 0.0f;
float* x1913 = (float*)myMalloc(1 * sizeof(float));;
x1913[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1913, x1913, in_desc, x1904, out_desc, x1910, sbmv_desc, x512,
    x1016, x497, x785, 1.0E-5));
};
float* x1916 = (float*)myMalloc(1 * sizeof(float));;
x1916[0] = 0.0f;
float* x1918 = (float*)myMalloc(1 * sizeof(float));;
x1918[0] = 1.0f;
float* x1920 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1918, x_desc, x1910, x1916, x_desc, x1920));
};
float* x1922 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1923 = (float*)myMalloc(1 * sizeof(float));;
x1923[0] = 0.0f;
float* x1925 = (float*)myMalloc(1 * sizeof(float));;
x1925[0] = 1.0f;

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
    x1925, in_desc, x1920, filt_desc, x935,
    conv_desc, algo, ws_data, ws_size,
    x1923, out_desc, x1922));
};
float* x1928 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1929 = (float*)myMalloc(1 * sizeof(float));;
x1929[0] = 0.0f;
float* x1931 = (float*)myMalloc(1 * sizeof(float));;
x1931[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1931, x1931, in_desc, x1922, out_desc, x1928, sbmv_desc, x680,
    x824, x467, x977, 1.0E-5));
};
float* x1934 = (float*)myMalloc(1 * sizeof(float));;
x1934[0] = 0.0f;
float* x1936 = (float*)myMalloc(1 * sizeof(float));;
x1936[0] = 1.0f;
float* x1938 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1936, x_desc, x1928, x1934, x_desc, x1938));
};
float* x1940 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1941 = (float*)myMalloc(1 * sizeof(float));;
x1941[0] = 0.0f;
float* x1943 = (float*)myMalloc(1 * sizeof(float));;
x1943[0] = 1.0f;

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
    x1943, in_desc, x1938, filt_desc, x548,
    conv_desc, algo, ws_data, ws_size,
    x1941, out_desc, x1940));
};
float* x1946 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1947 = (float*)myMalloc(1 * sizeof(float));;
x1947[0] = 0.0f;
float* x1949 = (float*)myMalloc(1 * sizeof(float));;
x1949[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1949, x1949, in_desc, x1940, out_desc, x1946, sbmv_desc, x1001,
    x536, x623, x806, 1.0E-5));
};
float* x1952 = (float*)myMalloc(1 * sizeof(float));;
x1952[0] = 1.0f;
float* x1954 = (float*)myMalloc(1 * sizeof(float));;
x1954[0] = 1.0f;

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
    cudnnHandle, x1952, bias_desc, x1902, x1954, out_desc, x1946));
};
float* x1957 = (float*)myMalloc(1 * sizeof(float));;
x1957[0] = 0.0f;
float* x1959 = (float*)myMalloc(1 * sizeof(float));;
x1959[0] = 1.0f;
float* x1961 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1959, x_desc, x1946, x1957, x_desc, x1961));
};
float* x1963 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1964 = (float*)myMalloc(1 * sizeof(float));;
x1964[0] = 0.0f;
float* x1966 = (float*)myMalloc(1 * sizeof(float));;
x1966[0] = 1.0f;

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
    x1966, in_desc, x1961, filt_desc, x674,
    conv_desc, algo, ws_data, ws_size,
    x1964, out_desc, x1963));
};
float* x1969 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1970 = (float*)myMalloc(1 * sizeof(float));;
x1970[0] = 0.0f;
float* x1972 = (float*)myMalloc(1 * sizeof(float));;
x1972[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1972, x1972, in_desc, x1963, out_desc, x1969, sbmv_desc, x860,
    x929, x458, x620, 1.0E-5));
};
float* x1975 = (float*)myMalloc(1 * sizeof(float));;
x1975[0] = 0.0f;
float* x1977 = (float*)myMalloc(1 * sizeof(float));;
x1977[0] = 1.0f;
float* x1979 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1977, x_desc, x1969, x1975, x_desc, x1979));
};
float* x1981 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x1982 = (float*)myMalloc(1 * sizeof(float));;
x1982[0] = 0.0f;
float* x1984 = (float*)myMalloc(1 * sizeof(float));;
x1984[0] = 1.0f;

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
    x1984, in_desc, x1979, filt_desc, x359,
    conv_desc, algo, ws_data, ws_size,
    x1982, out_desc, x1981));
};
float* x1987 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x1988 = (float*)myMalloc(1 * sizeof(float));;
x1988[0] = 0.0f;
float* x1990 = (float*)myMalloc(1 * sizeof(float));;
x1990[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x1990, x1990, in_desc, x1981, out_desc, x1987, sbmv_desc, x872,
    x734, x596, x407, 1.0E-5));
};
float* x1993 = (float*)myMalloc(1 * sizeof(float));;
x1993[0] = 0.0f;
float* x1995 = (float*)myMalloc(1 * sizeof(float));;
x1995[0] = 1.0f;
float* x1997 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x1995, x_desc, x1987, x1993, x_desc, x1997));
};
float* x1999 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2000 = (float*)myMalloc(1 * sizeof(float));;
x2000[0] = 0.0f;
float* x2002 = (float*)myMalloc(1 * sizeof(float));;
x2002[0] = 1.0f;

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
    x2002, in_desc, x1997, filt_desc, x893,
    conv_desc, algo, ws_data, ws_size,
    x2000, out_desc, x1999));
};
float* x2005 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2006 = (float*)myMalloc(1 * sizeof(float));;
x2006[0] = 0.0f;
float* x2008 = (float*)myMalloc(1 * sizeof(float));;
x2008[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2008, x2008, in_desc, x1999, out_desc, x2005, sbmv_desc, x974,
    x443, x602, x836, 1.0E-5));
};
float* x2011 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2012 = (float*)myMalloc(1 * sizeof(float));;
x2012[0] = 0.0f;
float* x2014 = (float*)myMalloc(1 * sizeof(float));;
x2014[0] = 1.0f;

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
    x2014, in_desc, x1961, filt_desc, x899,
    conv_desc, algo, ws_data, ws_size,
    x2012, out_desc, x2011));
};
float* x2017 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2018 = (float*)myMalloc(1 * sizeof(float));;
x2018[0] = 0.0f;
float* x2020 = (float*)myMalloc(1 * sizeof(float));;
x2020[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2020, x2020, in_desc, x2011, out_desc, x2017, sbmv_desc, x776,
    x578, x449, x632, 1.0E-5));
};
float* x2023 = (float*)myMalloc(1 * sizeof(float));;
x2023[0] = 1.0f;
float* x2025 = (float*)myMalloc(1 * sizeof(float));;
x2025[0] = 1.0f;

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
    cudnnHandle, x2023, bias_desc, x2017, x2025, out_desc, x2005));
};
float* x2028 = (float*)myMalloc(1 * sizeof(float));;
x2028[0] = 0.0f;
float* x2030 = (float*)myMalloc(1 * sizeof(float));;
x2030[0] = 1.0f;
float* x2032 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x2030, x_desc, x2005, x2028, x_desc, x2032));
};
float* x2034 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2035 = (float*)myMalloc(1 * sizeof(float));;
x2035[0] = 0.0f;
float* x2037 = (float*)myMalloc(1 * sizeof(float));;
x2037[0] = 1.0f;

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
    x2037, in_desc, x2032, filt_desc, x902,
    conv_desc, algo, ws_data, ws_size,
    x2035, out_desc, x2034));
};
float* x2040 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2041 = (float*)myMalloc(1 * sizeof(float));;
x2041[0] = 0.0f;
float* x2043 = (float*)myMalloc(1 * sizeof(float));;
x2043[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2043, x2043, in_desc, x2034, out_desc, x2040, sbmv_desc, x395,
    x668, x719, x452, 1.0E-5));
};
float* x2046 = (float*)myMalloc(1 * sizeof(float));;
x2046[0] = 0.0f;
float* x2048 = (float*)myMalloc(1 * sizeof(float));;
x2048[0] = 1.0f;
float* x2050 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2048, x_desc, x2040, x2046, x_desc, x2050));
};
float* x2052 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2053 = (float*)myMalloc(1 * sizeof(float));;
x2053[0] = 0.0f;
float* x2055 = (float*)myMalloc(1 * sizeof(float));;
x2055[0] = 1.0f;

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
    x2055, in_desc, x2050, filt_desc, x722,
    conv_desc, algo, ws_data, ws_size,
    x2053, out_desc, x2052));
};
float* x2058 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2059 = (float*)myMalloc(1 * sizeof(float));;
x2059[0] = 0.0f;
float* x2061 = (float*)myMalloc(1 * sizeof(float));;
x2061[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2061, x2061, in_desc, x2052, out_desc, x2058, sbmv_desc, x737,
    x455, x671, x842, 1.0E-5));
};
float* x2064 = (float*)myMalloc(1 * sizeof(float));;
x2064[0] = 0.0f;
float* x2066 = (float*)myMalloc(1 * sizeof(float));;
x2066[0] = 1.0f;
float* x2068 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2066, x_desc, x2058, x2064, x_desc, x2068));
};
float* x2070 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2071 = (float*)myMalloc(1 * sizeof(float));;
x2071[0] = 0.0f;
float* x2073 = (float*)myMalloc(1 * sizeof(float));;
x2073[0] = 1.0f;

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
    x2073, in_desc, x2068, filt_desc, x398,
    conv_desc, algo, ws_data, ws_size,
    x2071, out_desc, x2070));
};
float* x2076 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2077 = (float*)myMalloc(1 * sizeof(float));;
x2077[0] = 0.0f;
float* x2079 = (float*)myMalloc(1 * sizeof(float));;
x2079[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2079, x2079, in_desc, x2070, out_desc, x2076, sbmv_desc, x539,
    x689, x461, x992, 1.0E-5));
};
float* x2082 = (float*)myMalloc(1 * sizeof(float));;
x2082[0] = 1.0f;
float* x2084 = (float*)myMalloc(1 * sizeof(float));;
x2084[0] = 1.0f;

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
    cudnnHandle, x2082, bias_desc, x2032, x2084, out_desc, x2076));
};
float* x2087 = (float*)myMalloc(1 * sizeof(float));;
x2087[0] = 0.0f;
float* x2089 = (float*)myMalloc(1 * sizeof(float));;
x2089[0] = 1.0f;
float* x2091 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x2089, x_desc, x2076, x2087, x_desc, x2091));
};
float* x2093 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2094 = (float*)myMalloc(1 * sizeof(float));;
x2094[0] = 0.0f;
float* x2096 = (float*)myMalloc(1 * sizeof(float));;
x2096[0] = 1.0f;

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
    x2096, in_desc, x2091, filt_desc, x1052,
    conv_desc, algo, ws_data, ws_size,
    x2094, out_desc, x2093));
};
float* x2099 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2100 = (float*)myMalloc(1 * sizeof(float));;
x2100[0] = 0.0f;
float* x2102 = (float*)myMalloc(1 * sizeof(float));;
x2102[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2102, x2102, in_desc, x2093, out_desc, x2099, sbmv_desc, x302,
    x491, x896, x1022, 1.0E-5));
};
float* x2105 = (float*)myMalloc(1 * sizeof(float));;
x2105[0] = 0.0f;
float* x2107 = (float*)myMalloc(1 * sizeof(float));;
x2107[0] = 1.0f;
float* x2109 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2107, x_desc, x2099, x2105, x_desc, x2109));
};
float* x2111 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2112 = (float*)myMalloc(1 * sizeof(float));;
x2112[0] = 0.0f;
float* x2114 = (float*)myMalloc(1 * sizeof(float));;
x2114[0] = 1.0f;

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
    x2114, in_desc, x2109, filt_desc, x341,
    conv_desc, algo, ws_data, ws_size,
    x2112, out_desc, x2111));
};
float* x2117 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2118 = (float*)myMalloc(1 * sizeof(float));;
x2118[0] = 0.0f;
float* x2120 = (float*)myMalloc(1 * sizeof(float));;
x2120[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2120, x2120, in_desc, x2111, out_desc, x2117, sbmv_desc, x839,
    x764, x293, x863, 1.0E-5));
};
float* x2123 = (float*)myMalloc(1 * sizeof(float));;
x2123[0] = 0.0f;
float* x2125 = (float*)myMalloc(1 * sizeof(float));;
x2125[0] = 1.0f;
float* x2127 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2125, x_desc, x2117, x2123, x_desc, x2127));
};
float* x2129 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2130 = (float*)myMalloc(1 * sizeof(float));;
x2130[0] = 0.0f;
float* x2132 = (float*)myMalloc(1 * sizeof(float));;
x2132[0] = 1.0f;

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
    x2132, in_desc, x2127, filt_desc, x356,
    conv_desc, algo, ws_data, ws_size,
    x2130, out_desc, x2129));
};
float* x2135 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2136 = (float*)myMalloc(1 * sizeof(float));;
x2136[0] = 0.0f;
float* x2138 = (float*)myMalloc(1 * sizeof(float));;
x2138[0] = 1.0f;

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

CUDNN_CALL(cudnnBatchNormalizationForwardInference(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x2138, x2138, in_desc, x2129, out_desc, x2135, sbmv_desc, x566,
    x800, x1037, x626, 1.0E-5));
};
float* x2141 = (float*)myMalloc(1 * sizeof(float));;
x2141[0] = 1.0f;
float* x2143 = (float*)myMalloc(1 * sizeof(float));;
x2143[0] = 1.0f;

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
    cudnnHandle, x2141, bias_desc, x2091, x2143, out_desc, x2135));
};
float* x2146 = (float*)myMalloc(1 * sizeof(float));;
x2146[0] = 0.0f;
float* x2148 = (float*)myMalloc(1 * sizeof(float));;
x2148[0] = 1.0f;
float* x2150 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x2148, x_desc, x2135, x2146, x_desc, x2150));
};
float* x2152 = (float*)myMalloc(1 * sizeof(float));;
x2152[0] = 0.0f;
float* x2154 = (float*)myMalloc(1 * sizeof(float));;
x2154[0] = 1.0f;
float* x2156 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2154, in_desc, x2150, x2152, out_desc, x2156));
};
// resize to WrappedArray(64, 2048)
// gemm: WrappedArray(64, 2048), Vector(10, 2048)
float* x2160 = (float*)myGpuMalloc(640 * sizeof(float));
float* x2161 = (float*)myMalloc(1 * sizeof(float));;
x2161[0] = 0.0f;
float* x2163 = (float*)myMalloc(1 * sizeof(float));;
x2163[0] = 1.0f;
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 10,64,2048,x2163,x938,2048,x2156,2048,x2161,x2160,10));
float* x2166 = (float*)myMalloc(1 * sizeof(float));;
x2166[0] = 1.0f;
float* x2168 = (float*)myMalloc(1 * sizeof(float));;
x2168[0] = 1.0f;

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
    cudnnHandle, x2166, bias_desc, x401, x2168, out_desc, x2160));
};
// Tensor 'toCPU' invocation.
float* x2172 = (float*)myMalloc(640 * sizeof(float));;
CUDA_CALL(cudaMemcpy(x2172, x2160, 640 * sizeof(float), cudaMemcpyDeviceToHost));
printf("output (size 64 x 10)\n");
float x2175 = 0.0f;
for(int x2177=0; x2177 < 640; x2177++) {
float x2178 = x2175;
float x2179 = x2172[x2177];
float x2180 = fabs(x2179);
float x2181 = fabs(x2178);
bool x2182 = x2180 > x2181;
float x2183;
if (x2182) {
x2183 = x2179;
} else {
x2183 = x2178;
}
x2175 = x2183;

}
float x2187 = x2175;
printf("Max Abs: %.5f || ",x2187);
for(int x2189=0; x2189 < 10; x2189++) {
float x2190 = x2172[x2189];
printf("%.5f ",x2190);

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


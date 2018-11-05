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

long HEAP_SIZE = 4294967304; // 1073741826; // 1048576; // 536870912; // 268435456; // 2097152; 1610612739; //
void *mallocBase = calloc(HEAP_SIZE, 1);
void *mallocAddr = mallocBase;
void *waterMark = mallocBase;
void *myMalloc(size_t bytes) {
  void *res = mallocAddr;
  mallocAddr = (void *)((char *)mallocAddr + bytes);
  if ((long)mallocAddr >= (long)mallocBase + HEAP_SIZE)
    fprintf(stderr, "CPU memory breached limit of HEAP_SIZE\n");
  return res;
}

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

__global__ void concat(float* in1, float* in2, float* out, int bound) {
  int tid = blockIdx.x * blockDim.x * blockDim.y * blockDim.z +
            threadIdx.z * blockDim.y * blockDim.x +
            threadIdx.y * blockDim.x + threadIdx.x;
  if (threadIdx.z < bound) {
    int subid = blockIdx.x * blockDim.x * blockDim.y * bound +
                threadIdx.z * blockDim.y * blockDim.x +
                threadIdx.y * blockDim.x + threadIdx.x;
    out[tid] = in1[subid];
  } else {
    int subid = blockIdx.x * blockDim.x * blockDim.y * (blockDim.z - bound) +
                (threadIdx.z - bound) * blockDim.y * blockDim.x +
                threadIdx.y * blockDim.x + threadIdx.x;
    out[tid] = in2[subid];
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

__global__ void concat_grad(float* in1, float* in2, float* out, int bound) {
  int tid = blockIdx.x * blockDim.x * blockDim.y * blockDim.z +
            threadIdx.z * blockDim.y * blockDim.x +
            threadIdx.y * blockDim.x + threadIdx.x;
  if (threadIdx.z < bound) {
    int subid = blockIdx.x * blockDim.x * blockDim.y * bound +
                threadIdx.z * blockDim.y * blockDim.x +
                threadIdx.y * blockDim.x + threadIdx.x;
     in1[subid] += out[tid];
  } else {
    int subid = blockIdx.x * blockDim.x * blockDim.y * (blockDim.z - bound) +
                (threadIdx.z - bound) * blockDim.y * blockDim.x +
                threadIdx.y * blockDim.x + threadIdx.x;
    in2[subid] += out[tid];
  }
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
int32_t x4 = open("/home/fei/bitbucket/Lantern/src/out/PLDI19evaluation/resnet50/resnet50.onnx.bin",0);
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
float* x1133 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1134 = (float*)myMalloc(1 * sizeof(float));;
x1134[0] = 0.0f;
float* x1136 = (float*)myMalloc(1 * sizeof(float));;
x1136[0] = 1.0f;

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
    x1136, in_desc, x1109, filt_desc, x713,
    conv_desc, algo, ws_data, ws_size,
    x1134, out_desc, x1133));
};
float* x1139 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1140 = (float*)myMalloc(1 * sizeof(float));;
x1140[0] = 0.0f;
float* x1142 = (float*)myMalloc(1 * sizeof(float));;
x1142[0] = 1.0f;

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
    x1142, x1142, in_desc, x1133, out_desc, x1139, sbmv_desc, x875,
    x1010, x377, x587, 1.0E-5));
};
float* x1145 = (float*)myMalloc(1 * sizeof(float));;
x1145[0] = 0.0f;
float* x1147 = (float*)myMalloc(1 * sizeof(float));;
x1147[0] = 1.0f;
float* x1149 = (float*)myGpuMalloc(4194304 * sizeof(float));

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
    x1147, x_desc, x1139, x1145, x_desc, x1149));
};
float* x1151 = (float*)myMalloc(1 * sizeof(float));;
x1151[0] = 0.0f;
float* x1153 = (float*)myMalloc(1 * sizeof(float));;
x1153[0] = 1.0f;
float* x1155 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1153, in_desc, x1149, x1151, out_desc, x1155));
};
float* x1157 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1158 = (float*)myMalloc(1 * sizeof(float));;
x1158[0] = 0.0f;
float* x1160 = (float*)myMalloc(1 * sizeof(float));;
x1160[0] = 1.0f;

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
    x1160, in_desc, x1155, filt_desc, x956,
    conv_desc, algo, ws_data, ws_size,
    x1158, out_desc, x1157));
};
float* x1163 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1164 = (float*)myMalloc(1 * sizeof(float));;
x1164[0] = 0.0f;
float* x1166 = (float*)myMalloc(1 * sizeof(float));;
x1166[0] = 1.0f;

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
    x1166, x1166, in_desc, x1157, out_desc, x1163, sbmv_desc, x335,
    x416, x599, x410, 1.0E-5));
};
float* x1169 = (float*)myMalloc(1 * sizeof(float));;
x1169[0] = 0.0f;
float* x1171 = (float*)myMalloc(1 * sizeof(float));;
x1171[0] = 1.0f;
float* x1173 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1171, x_desc, x1163, x1169, x_desc, x1173));
};
float* x1175 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1176 = (float*)myMalloc(1 * sizeof(float));;
x1176[0] = 0.0f;
float* x1178 = (float*)myMalloc(1 * sizeof(float));;
x1178[0] = 1.0f;

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
    x1178, in_desc, x1173, filt_desc, x527,
    conv_desc, algo, ws_data, ws_size,
    x1176, out_desc, x1175));
};
float* x1181 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1182 = (float*)myMalloc(1 * sizeof(float));;
x1182[0] = 0.0f;
float* x1184 = (float*)myMalloc(1 * sizeof(float));;
x1184[0] = 1.0f;

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
    x1184, x1184, in_desc, x1175, out_desc, x1181, sbmv_desc, x749,
    x404, x572, x731, 1.0E-5));
};
float* x1187 = (float*)myMalloc(1 * sizeof(float));;
x1187[0] = 0.0f;
float* x1189 = (float*)myMalloc(1 * sizeof(float));;
x1189[0] = 1.0f;
float* x1191 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1189, x_desc, x1181, x1187, x_desc, x1191));
};
float* x1193 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1194 = (float*)myMalloc(1 * sizeof(float));;
x1194[0] = 0.0f;
float* x1196 = (float*)myMalloc(1 * sizeof(float));;
x1196[0] = 1.0f;

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
    x1196, in_desc, x1191, filt_desc, x353,
    conv_desc, algo, ws_data, ws_size,
    x1194, out_desc, x1193));
};
float* x1199 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1200 = (float*)myMalloc(1 * sizeof(float));;
x1200[0] = 0.0f;
float* x1202 = (float*)myMalloc(1 * sizeof(float));;
x1202[0] = 1.0f;

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
    x1202, x1202, in_desc, x1193, out_desc, x1199, sbmv_desc, x854,
    x635, x470, x365, 1.0E-5));
};
float* x1205 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1206 = (float*)myMalloc(1 * sizeof(float));;
x1206[0] = 0.0f;
float* x1208 = (float*)myMalloc(1 * sizeof(float));;
x1208[0] = 1.0f;

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
    x1208, in_desc, x1155, filt_desc, x743,
    conv_desc, algo, ws_data, ws_size,
    x1206, out_desc, x1205));
};
float* x1211 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1212 = (float*)myMalloc(1 * sizeof(float));;
x1212[0] = 0.0f;
float* x1214 = (float*)myMalloc(1 * sizeof(float));;
x1214[0] = 1.0f;

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
    x1214, x1214, in_desc, x1205, out_desc, x1211, sbmv_desc, x485,
    x866, x1049, x986, 1.0E-5));
};
float* x1217 = (float*)myMalloc(1 * sizeof(float));;
x1217[0] = 1.0f;
float* x1219 = (float*)myMalloc(1 * sizeof(float));;
x1219[0] = 1.0f;

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
    cudnnHandle, x1217, bias_desc, x1211, x1219, out_desc, x1199));
};
float* x1222 = (float*)myMalloc(1 * sizeof(float));;
x1222[0] = 0.0f;
float* x1224 = (float*)myMalloc(1 * sizeof(float));;
x1224[0] = 1.0f;
float* x1226 = (float*)myGpuMalloc(4194304 * sizeof(float));

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
    x1224, x_desc, x1199, x1222, x_desc, x1226));
};
float* x1228 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1229 = (float*)myMalloc(1 * sizeof(float));;
x1229[0] = 0.0f;
float* x1231 = (float*)myMalloc(1 * sizeof(float));;
x1231[0] = 1.0f;

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
    x1231, in_desc, x1226, filt_desc, x770,
    conv_desc, algo, ws_data, ws_size,
    x1229, out_desc, x1228));
};
float* x1234 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1235 = (float*)myMalloc(1 * sizeof(float));;
x1235[0] = 0.0f;
float* x1237 = (float*)myMalloc(1 * sizeof(float));;
x1237[0] = 1.0f;

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
    x1237, x1237, in_desc, x1228, out_desc, x1234, sbmv_desc, x683,
    x437, x287, x563, 1.0E-5));
};
float* x1240 = (float*)myMalloc(1 * sizeof(float));;
x1240[0] = 0.0f;
float* x1242 = (float*)myMalloc(1 * sizeof(float));;
x1242[0] = 1.0f;
float* x1244 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1242, x_desc, x1234, x1240, x_desc, x1244));
};
float* x1246 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1247 = (float*)myMalloc(1 * sizeof(float));;
x1247[0] = 0.0f;
float* x1249 = (float*)myMalloc(1 * sizeof(float));;
x1249[0] = 1.0f;

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
    x1249, in_desc, x1244, filt_desc, x506,
    conv_desc, algo, ws_data, ws_size,
    x1247, out_desc, x1246));
};
float* x1252 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1253 = (float*)myMalloc(1 * sizeof(float));;
x1253[0] = 0.0f;
float* x1255 = (float*)myMalloc(1 * sizeof(float));;
x1255[0] = 1.0f;

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
    x1255, x1255, in_desc, x1246, out_desc, x1252, sbmv_desc, x881,
    x716, x389, x989, 1.0E-5));
};
float* x1258 = (float*)myMalloc(1 * sizeof(float));;
x1258[0] = 0.0f;
float* x1260 = (float*)myMalloc(1 * sizeof(float));;
x1260[0] = 1.0f;
float* x1262 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1260, x_desc, x1252, x1258, x_desc, x1262));
};
float* x1264 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1265 = (float*)myMalloc(1 * sizeof(float));;
x1265[0] = 0.0f;
float* x1267 = (float*)myMalloc(1 * sizeof(float));;
x1267[0] = 1.0f;

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
    x1267, in_desc, x1262, filt_desc, x647,
    conv_desc, algo, ws_data, ws_size,
    x1265, out_desc, x1264));
};
float* x1270 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1271 = (float*)myMalloc(1 * sizeof(float));;
x1271[0] = 0.0f;
float* x1273 = (float*)myMalloc(1 * sizeof(float));;
x1273[0] = 1.0f;

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
    x1273, x1273, in_desc, x1264, out_desc, x1270, sbmv_desc, x431,
    x278, x530, x755, 1.0E-5));
};
float* x1276 = (float*)myMalloc(1 * sizeof(float));;
x1276[0] = 1.0f;
float* x1278 = (float*)myMalloc(1 * sizeof(float));;
x1278[0] = 1.0f;

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
    cudnnHandle, x1276, bias_desc, x1226, x1278, out_desc, x1270));
};
float* x1281 = (float*)myMalloc(1 * sizeof(float));;
x1281[0] = 0.0f;
float* x1283 = (float*)myMalloc(1 * sizeof(float));;
x1283[0] = 1.0f;
float* x1285 = (float*)myGpuMalloc(4194304 * sizeof(float));

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
    x1283, x_desc, x1270, x1281, x_desc, x1285));
};
float* x1287 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1288 = (float*)myMalloc(1 * sizeof(float));;
x1288[0] = 0.0f;
float* x1290 = (float*)myMalloc(1 * sizeof(float));;
x1290[0] = 1.0f;

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
    x1290, in_desc, x1285, filt_desc, x707,
    conv_desc, algo, ws_data, ws_size,
    x1288, out_desc, x1287));
};
float* x1293 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1294 = (float*)myMalloc(1 * sizeof(float));;
x1294[0] = 0.0f;
float* x1296 = (float*)myMalloc(1 * sizeof(float));;
x1296[0] = 1.0f;

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
    x1296, x1296, in_desc, x1287, out_desc, x1293, sbmv_desc, x500,
    x329, x1028, x818, 1.0E-5));
};
float* x1299 = (float*)myMalloc(1 * sizeof(float));;
x1299[0] = 0.0f;
float* x1301 = (float*)myMalloc(1 * sizeof(float));;
x1301[0] = 1.0f;
float* x1303 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1301, x_desc, x1293, x1299, x_desc, x1303));
};
float* x1305 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1306 = (float*)myMalloc(1 * sizeof(float));;
x1306[0] = 0.0f;
float* x1308 = (float*)myMalloc(1 * sizeof(float));;
x1308[0] = 1.0f;

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
    x1308, in_desc, x1303, filt_desc, x476,
    conv_desc, algo, ws_data, ws_size,
    x1306, out_desc, x1305));
};
float* x1311 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1312 = (float*)myMalloc(1 * sizeof(float));;
x1312[0] = 0.0f;
float* x1314 = (float*)myMalloc(1 * sizeof(float));;
x1314[0] = 1.0f;

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
    x1314, x1314, in_desc, x1305, out_desc, x1311, sbmv_desc, x473,
    x662, x794, x611, 1.0E-5));
};
float* x1317 = (float*)myMalloc(1 * sizeof(float));;
x1317[0] = 0.0f;
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
CUDNN_CALL(cudnnActivationForward(
    cudnnHandle, act_desc,
    x1319, x_desc, x1311, x1317, x_desc, x1321));
};
float* x1323 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1324 = (float*)myMalloc(1 * sizeof(float));;
x1324[0] = 0.0f;
float* x1326 = (float*)myMalloc(1 * sizeof(float));;
x1326[0] = 1.0f;

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
    x1326, in_desc, x1321, filt_desc, x518,
    conv_desc, algo, ws_data, ws_size,
    x1324, out_desc, x1323));
};
float* x1329 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1330 = (float*)myMalloc(1 * sizeof(float));;
x1330[0] = 0.0f;
float* x1332 = (float*)myMalloc(1 * sizeof(float));;
x1332[0] = 1.0f;

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
    x1332, x1332, in_desc, x1323, out_desc, x1329, sbmv_desc, x368,
    x998, x809, x656, 1.0E-5));
};
float* x1335 = (float*)myMalloc(1 * sizeof(float));;
x1335[0] = 1.0f;
float* x1337 = (float*)myMalloc(1 * sizeof(float));;
x1337[0] = 1.0f;

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
    cudnnHandle, x1335, bias_desc, x1285, x1337, out_desc, x1329));
};
float* x1340 = (float*)myMalloc(1 * sizeof(float));;
x1340[0] = 0.0f;
float* x1342 = (float*)myMalloc(1 * sizeof(float));;
x1342[0] = 1.0f;
float* x1344 = (float*)myGpuMalloc(4194304 * sizeof(float));

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
    x1342, x_desc, x1329, x1340, x_desc, x1344));
};
float* x1346 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1347 = (float*)myMalloc(1 * sizeof(float));;
x1347[0] = 0.0f;
float* x1349 = (float*)myMalloc(1 * sizeof(float));;
x1349[0] = 1.0f;

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
    x1349, in_desc, x1344, filt_desc, x290,
    conv_desc, algo, ws_data, ws_size,
    x1347, out_desc, x1346));
};
float* x1352 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1353 = (float*)myMalloc(1 * sizeof(float));;
x1353[0] = 0.0f;
float* x1355 = (float*)myMalloc(1 * sizeof(float));;
x1355[0] = 1.0f;

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
    x1355, x1355, in_desc, x1346, out_desc, x1352, sbmv_desc, x509,
    x773, x869, x659, 1.0E-5));
};
float* x1358 = (float*)myMalloc(1 * sizeof(float));;
x1358[0] = 0.0f;
float* x1360 = (float*)myMalloc(1 * sizeof(float));;
x1360[0] = 1.0f;
float* x1362 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x1360, x_desc, x1352, x1358, x_desc, x1362));
};
float* x1364 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1365 = (float*)myMalloc(1 * sizeof(float));;
x1365[0] = 0.0f;
float* x1367 = (float*)myMalloc(1 * sizeof(float));;
x1367[0] = 1.0f;

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
    x1367, in_desc, x1362, filt_desc, x338,
    conv_desc, algo, ws_data, ws_size,
    x1365, out_desc, x1364));
};
float* x1370 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1371 = (float*)myMalloc(1 * sizeof(float));;
x1371[0] = 0.0f;
float* x1373 = (float*)myMalloc(1 * sizeof(float));;
x1373[0] = 1.0f;

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
    x1373, x1373, in_desc, x1364, out_desc, x1370, sbmv_desc, x1013,
    x827, x641, x386, 1.0E-5));
};
float* x1376 = (float*)myMalloc(1 * sizeof(float));;
x1376[0] = 0.0f;
float* x1378 = (float*)myMalloc(1 * sizeof(float));;
x1378[0] = 1.0f;
float* x1380 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1378, x_desc, x1370, x1376, x_desc, x1380));
};
float* x1382 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1383 = (float*)myMalloc(1 * sizeof(float));;
x1383[0] = 0.0f;
float* x1385 = (float*)myMalloc(1 * sizeof(float));;
x1385[0] = 1.0f;

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
    x1385, in_desc, x1380, filt_desc, x575,
    conv_desc, algo, ws_data, ws_size,
    x1383, out_desc, x1382));
};
float* x1388 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1389 = (float*)myMalloc(1 * sizeof(float));;
x1389[0] = 0.0f;
float* x1391 = (float*)myMalloc(1 * sizeof(float));;
x1391[0] = 1.0f;

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
    x1391, x1391, in_desc, x1382, out_desc, x1388, sbmv_desc, x692,
    x887, x704, x560, 1.0E-5));
};
float* x1394 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1395 = (float*)myMalloc(1 * sizeof(float));;
x1395[0] = 0.0f;
float* x1397 = (float*)myMalloc(1 * sizeof(float));;
x1397[0] = 1.0f;

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
    x1397, in_desc, x1344, filt_desc, x1031,
    conv_desc, algo, ws_data, ws_size,
    x1395, out_desc, x1394));
};
float* x1400 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1401 = (float*)myMalloc(1 * sizeof(float));;
x1401[0] = 0.0f;
float* x1403 = (float*)myMalloc(1 * sizeof(float));;
x1403[0] = 1.0f;

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
    x1403, x1403, in_desc, x1394, out_desc, x1400, sbmv_desc, x878,
    x614, x383, x326, 1.0E-5));
};
float* x1406 = (float*)myMalloc(1 * sizeof(float));;
x1406[0] = 1.0f;
float* x1408 = (float*)myMalloc(1 * sizeof(float));;
x1408[0] = 1.0f;

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
    cudnnHandle, x1406, bias_desc, x1400, x1408, out_desc, x1388));
};
float* x1411 = (float*)myMalloc(1 * sizeof(float));;
x1411[0] = 0.0f;
float* x1413 = (float*)myMalloc(1 * sizeof(float));;
x1413[0] = 1.0f;
float* x1415 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x1413, x_desc, x1388, x1411, x_desc, x1415));
};
float* x1417 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1418 = (float*)myMalloc(1 * sizeof(float));;
x1418[0] = 0.0f;
float* x1420 = (float*)myMalloc(1 * sizeof(float));;
x1420[0] = 1.0f;

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
    x1420, in_desc, x1415, filt_desc, x1025,
    conv_desc, algo, ws_data, ws_size,
    x1418, out_desc, x1417));
};
float* x1423 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1424 = (float*)myMalloc(1 * sizeof(float));;
x1424[0] = 0.0f;
float* x1426 = (float*)myMalloc(1 * sizeof(float));;
x1426[0] = 1.0f;

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
    x1426, x1426, in_desc, x1417, out_desc, x1423, sbmv_desc, x923,
    x308, x557, x788, 1.0E-5));
};
float* x1429 = (float*)myMalloc(1 * sizeof(float));;
x1429[0] = 0.0f;
float* x1431 = (float*)myMalloc(1 * sizeof(float));;
x1431[0] = 1.0f;
float* x1433 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1431, x_desc, x1423, x1429, x_desc, x1433));
};
float* x1435 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1436 = (float*)myMalloc(1 * sizeof(float));;
x1436[0] = 0.0f;
float* x1438 = (float*)myMalloc(1 * sizeof(float));;
x1438[0] = 1.0f;

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
    x1438, in_desc, x1433, filt_desc, x962,
    conv_desc, algo, ws_data, ws_size,
    x1436, out_desc, x1435));
};
float* x1441 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1442 = (float*)myMalloc(1 * sizeof(float));;
x1442[0] = 0.0f;
float* x1444 = (float*)myMalloc(1 * sizeof(float));;
x1444[0] = 1.0f;

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
    x1444, x1444, in_desc, x1435, out_desc, x1441, sbmv_desc, x281,
    x542, x362, x932, 1.0E-5));
};
float* x1447 = (float*)myMalloc(1 * sizeof(float));;
x1447[0] = 0.0f;
float* x1449 = (float*)myMalloc(1 * sizeof(float));;
x1449[0] = 1.0f;
float* x1451 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1449, x_desc, x1441, x1447, x_desc, x1451));
};
float* x1453 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1454 = (float*)myMalloc(1 * sizeof(float));;
x1454[0] = 0.0f;
float* x1456 = (float*)myMalloc(1 * sizeof(float));;
x1456[0] = 1.0f;

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
    x1456, in_desc, x1451, filt_desc, x590,
    conv_desc, algo, ws_data, ws_size,
    x1454, out_desc, x1453));
};
float* x1459 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1460 = (float*)myMalloc(1 * sizeof(float));;
x1460[0] = 0.0f;
float* x1462 = (float*)myMalloc(1 * sizeof(float));;
x1462[0] = 1.0f;

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
    x1462, x1462, in_desc, x1453, out_desc, x1459, sbmv_desc, x413,
    x995, x698, x521, 1.0E-5));
};
float* x1465 = (float*)myMalloc(1 * sizeof(float));;
x1465[0] = 1.0f;
float* x1467 = (float*)myMalloc(1 * sizeof(float));;
x1467[0] = 1.0f;

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
    cudnnHandle, x1465, bias_desc, x1415, x1467, out_desc, x1459));
};
float* x1470 = (float*)myMalloc(1 * sizeof(float));;
x1470[0] = 0.0f;
float* x1472 = (float*)myMalloc(1 * sizeof(float));;
x1472[0] = 1.0f;
float* x1474 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x1472, x_desc, x1459, x1470, x_desc, x1474));
};
float* x1476 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1477 = (float*)myMalloc(1 * sizeof(float));;
x1477[0] = 0.0f;
float* x1479 = (float*)myMalloc(1 * sizeof(float));;
x1479[0] = 1.0f;

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
    x1479, in_desc, x1474, filt_desc, x845,
    conv_desc, algo, ws_data, ws_size,
    x1477, out_desc, x1476));
};
float* x1482 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1483 = (float*)myMalloc(1 * sizeof(float));;
x1483[0] = 0.0f;
float* x1485 = (float*)myMalloc(1 * sizeof(float));;
x1485[0] = 1.0f;

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
    x1485, x1485, in_desc, x1476, out_desc, x1482, sbmv_desc, x392,
    x767, x593, x284, 1.0E-5));
};
float* x1488 = (float*)myMalloc(1 * sizeof(float));;
x1488[0] = 0.0f;
float* x1490 = (float*)myMalloc(1 * sizeof(float));;
x1490[0] = 1.0f;
float* x1492 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1490, x_desc, x1482, x1488, x_desc, x1492));
};
float* x1494 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1495 = (float*)myMalloc(1 * sizeof(float));;
x1495[0] = 0.0f;
float* x1497 = (float*)myMalloc(1 * sizeof(float));;
x1497[0] = 1.0f;

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
    x1497, in_desc, x1492, filt_desc, x830,
    conv_desc, algo, ws_data, ws_size,
    x1495, out_desc, x1494));
};
float* x1500 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1501 = (float*)myMalloc(1 * sizeof(float));;
x1501[0] = 0.0f;
float* x1503 = (float*)myMalloc(1 * sizeof(float));;
x1503[0] = 1.0f;

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
    x1503, x1503, in_desc, x1494, out_desc, x1500, sbmv_desc, x638,
    x440, x908, x1055, 1.0E-5));
};
float* x1506 = (float*)myMalloc(1 * sizeof(float));;
x1506[0] = 0.0f;
float* x1508 = (float*)myMalloc(1 * sizeof(float));;
x1508[0] = 1.0f;
float* x1510 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1508, x_desc, x1500, x1506, x_desc, x1510));
};
float* x1512 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1513 = (float*)myMalloc(1 * sizeof(float));;
x1513[0] = 0.0f;
float* x1515 = (float*)myMalloc(1 * sizeof(float));;
x1515[0] = 1.0f;

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
    x1515, in_desc, x1510, filt_desc, x380,
    conv_desc, algo, ws_data, ws_size,
    x1513, out_desc, x1512));
};
float* x1518 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1519 = (float*)myMalloc(1 * sizeof(float));;
x1519[0] = 0.0f;
float* x1521 = (float*)myMalloc(1 * sizeof(float));;
x1521[0] = 1.0f;

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
    x1521, x1521, in_desc, x1512, out_desc, x1518, sbmv_desc, x758,
    x503, x332, x926, 1.0E-5));
};
float* x1524 = (float*)myMalloc(1 * sizeof(float));;
x1524[0] = 1.0f;
float* x1526 = (float*)myMalloc(1 * sizeof(float));;
x1526[0] = 1.0f;

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
    cudnnHandle, x1524, bias_desc, x1474, x1526, out_desc, x1518));
};
float* x1529 = (float*)myMalloc(1 * sizeof(float));;
x1529[0] = 0.0f;
float* x1531 = (float*)myMalloc(1 * sizeof(float));;
x1531[0] = 1.0f;
float* x1533 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x1531, x_desc, x1518, x1529, x_desc, x1533));
};
float* x1535 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1536 = (float*)myMalloc(1 * sizeof(float));;
x1536[0] = 0.0f;
float* x1538 = (float*)myMalloc(1 * sizeof(float));;
x1538[0] = 1.0f;

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
    x1538, in_desc, x1533, filt_desc, x653,
    conv_desc, algo, ws_data, ws_size,
    x1536, out_desc, x1535));
};
float* x1541 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1542 = (float*)myMalloc(1 * sizeof(float));;
x1542[0] = 0.0f;
float* x1544 = (float*)myMalloc(1 * sizeof(float));;
x1544[0] = 1.0f;

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
    x1544, x1544, in_desc, x1535, out_desc, x1541, sbmv_desc, x374,
    x983, x965, x1040, 1.0E-5));
};
float* x1547 = (float*)myMalloc(1 * sizeof(float));;
x1547[0] = 0.0f;
float* x1549 = (float*)myMalloc(1 * sizeof(float));;
x1549[0] = 1.0f;
float* x1551 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1549, x_desc, x1541, x1547, x_desc, x1551));
};
float* x1553 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1554 = (float*)myMalloc(1 * sizeof(float));;
x1554[0] = 0.0f;
float* x1556 = (float*)myMalloc(1 * sizeof(float));;
x1556[0] = 1.0f;

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
    x1556, in_desc, x1551, filt_desc, x752,
    conv_desc, algo, ws_data, ws_size,
    x1554, out_desc, x1553));
};
float* x1559 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1560 = (float*)myMalloc(1 * sizeof(float));;
x1560[0] = 0.0f;
float* x1562 = (float*)myMalloc(1 * sizeof(float));;
x1562[0] = 1.0f;

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
    x1562, x1562, in_desc, x1553, out_desc, x1559, sbmv_desc, x494,
    x371, x1061, x701, 1.0E-5));
};
float* x1565 = (float*)myMalloc(1 * sizeof(float));;
x1565[0] = 0.0f;
float* x1567 = (float*)myMalloc(1 * sizeof(float));;
x1567[0] = 1.0f;
float* x1569 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1567, x_desc, x1559, x1565, x_desc, x1569));
};
float* x1571 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1572 = (float*)myMalloc(1 * sizeof(float));;
x1572[0] = 0.0f;
float* x1574 = (float*)myMalloc(1 * sizeof(float));;
x1574[0] = 1.0f;

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
    x1574, in_desc, x1569, filt_desc, x422,
    conv_desc, algo, ws_data, ws_size,
    x1572, out_desc, x1571));
};
float* x1577 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1578 = (float*)myMalloc(1 * sizeof(float));;
x1578[0] = 0.0f;
float* x1580 = (float*)myMalloc(1 * sizeof(float));;
x1580[0] = 1.0f;

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
    x1580, x1580, in_desc, x1571, out_desc, x1577, sbmv_desc, x725,
    x419, x314, x959, 1.0E-5));
};
float* x1583 = (float*)myMalloc(1 * sizeof(float));;
x1583[0] = 1.0f;
float* x1585 = (float*)myMalloc(1 * sizeof(float));;
x1585[0] = 1.0f;

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
    cudnnHandle, x1583, bias_desc, x1533, x1585, out_desc, x1577));
};
float* x1588 = (float*)myMalloc(1 * sizeof(float));;
x1588[0] = 0.0f;
float* x1590 = (float*)myMalloc(1 * sizeof(float));;
x1590[0] = 1.0f;
float* x1592 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x1590, x_desc, x1577, x1588, x_desc, x1592));
};
float* x1594 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1595 = (float*)myMalloc(1 * sizeof(float));;
x1595[0] = 0.0f;
float* x1597 = (float*)myMalloc(1 * sizeof(float));;
x1597[0] = 1.0f;

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
    x1597, in_desc, x1592, filt_desc, x797,
    conv_desc, algo, ws_data, ws_size,
    x1595, out_desc, x1594));
};
float* x1600 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1601 = (float*)myMalloc(1 * sizeof(float));;
x1601[0] = 0.0f;
float* x1603 = (float*)myMalloc(1 * sizeof(float));;
x1603[0] = 1.0f;

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
    x1603, x1603, in_desc, x1594, out_desc, x1600, sbmv_desc, x1067,
    x320, x650, x851, 1.0E-5));
};
float* x1606 = (float*)myMalloc(1 * sizeof(float));;
x1606[0] = 0.0f;
float* x1608 = (float*)myMalloc(1 * sizeof(float));;
x1608[0] = 1.0f;
float* x1610 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1608, x_desc, x1600, x1606, x_desc, x1610));
};
float* x1612 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1613 = (float*)myMalloc(1 * sizeof(float));;
x1613[0] = 0.0f;
float* x1615 = (float*)myMalloc(1 * sizeof(float));;
x1615[0] = 1.0f;

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
    x1615, in_desc, x1610, filt_desc, x782,
    conv_desc, algo, ws_data, ws_size,
    x1613, out_desc, x1612));
};
float* x1618 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1619 = (float*)myMalloc(1 * sizeof(float));;
x1619[0] = 0.0f;
float* x1621 = (float*)myMalloc(1 * sizeof(float));;
x1621[0] = 1.0f;

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
    x1621, x1621, in_desc, x1612, out_desc, x1618, sbmv_desc, x581,
    x305, x944, x554, 1.0E-5));
};
float* x1624 = (float*)myMalloc(1 * sizeof(float));;
x1624[0] = 0.0f;
float* x1626 = (float*)myMalloc(1 * sizeof(float));;
x1626[0] = 1.0f;
float* x1628 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1626, x_desc, x1618, x1624, x_desc, x1628));
};
float* x1630 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1631 = (float*)myMalloc(1 * sizeof(float));;
x1631[0] = 0.0f;
float* x1633 = (float*)myMalloc(1 * sizeof(float));;
x1633[0] = 1.0f;

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
    x1633, in_desc, x1628, filt_desc, x1064,
    conv_desc, algo, ws_data, ws_size,
    x1631, out_desc, x1630));
};
float* x1636 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1637 = (float*)myMalloc(1 * sizeof(float));;
x1637[0] = 0.0f;
float* x1639 = (float*)myMalloc(1 * sizeof(float));;
x1639[0] = 1.0f;

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
    x1639, x1639, in_desc, x1630, out_desc, x1636, sbmv_desc, x311,
    x608, x905, x1058, 1.0E-5));
};
float* x1642 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1643 = (float*)myMalloc(1 * sizeof(float));;
x1643[0] = 0.0f;
float* x1645 = (float*)myMalloc(1 * sizeof(float));;
x1645[0] = 1.0f;

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
    x1645, in_desc, x1592, filt_desc, x482,
    conv_desc, algo, ws_data, ws_size,
    x1643, out_desc, x1642));
};
float* x1648 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1649 = (float*)myMalloc(1 * sizeof(float));;
x1649[0] = 0.0f;
float* x1651 = (float*)myMalloc(1 * sizeof(float));;
x1651[0] = 1.0f;

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
    x1651, x1651, in_desc, x1642, out_desc, x1648, sbmv_desc, x344,
    x917, x515, x890, 1.0E-5));
};
float* x1654 = (float*)myMalloc(1 * sizeof(float));;
x1654[0] = 1.0f;
float* x1656 = (float*)myMalloc(1 * sizeof(float));;
x1656[0] = 1.0f;

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
    cudnnHandle, x1654, bias_desc, x1648, x1656, out_desc, x1636));
};
float* x1659 = (float*)myMalloc(1 * sizeof(float));;
x1659[0] = 0.0f;
float* x1661 = (float*)myMalloc(1 * sizeof(float));;
x1661[0] = 1.0f;
float* x1663 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1661, x_desc, x1636, x1659, x_desc, x1663));
};
float* x1665 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1666 = (float*)myMalloc(1 * sizeof(float));;
x1666[0] = 0.0f;
float* x1668 = (float*)myMalloc(1 * sizeof(float));;
x1668[0] = 1.0f;

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
    x1668, in_desc, x1663, filt_desc, x296,
    conv_desc, algo, ws_data, ws_size,
    x1666, out_desc, x1665));
};
float* x1671 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1672 = (float*)myMalloc(1 * sizeof(float));;
x1672[0] = 0.0f;
float* x1674 = (float*)myMalloc(1 * sizeof(float));;
x1674[0] = 1.0f;

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
    x1674, x1674, in_desc, x1665, out_desc, x1671, sbmv_desc, x347,
    x914, x1034, x728, 1.0E-5));
};
float* x1677 = (float*)myMalloc(1 * sizeof(float));;
x1677[0] = 0.0f;
float* x1679 = (float*)myMalloc(1 * sizeof(float));;
x1679[0] = 1.0f;
float* x1681 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1679, x_desc, x1671, x1677, x_desc, x1681));
};
float* x1683 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1684 = (float*)myMalloc(1 * sizeof(float));;
x1684[0] = 0.0f;
float* x1686 = (float*)myMalloc(1 * sizeof(float));;
x1686[0] = 1.0f;

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
    x1686, in_desc, x1681, filt_desc, x350,
    conv_desc, algo, ws_data, ws_size,
    x1684, out_desc, x1683));
};
float* x1689 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1690 = (float*)myMalloc(1 * sizeof(float));;
x1690[0] = 0.0f;
float* x1692 = (float*)myMalloc(1 * sizeof(float));;
x1692[0] = 1.0f;

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
    x1692, x1692, in_desc, x1683, out_desc, x1689, sbmv_desc, x1070,
    x545, x857, x968, 1.0E-5));
};
float* x1695 = (float*)myMalloc(1 * sizeof(float));;
x1695[0] = 0.0f;
float* x1697 = (float*)myMalloc(1 * sizeof(float));;
x1697[0] = 1.0f;
float* x1699 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1697, x_desc, x1689, x1695, x_desc, x1699));
};
float* x1701 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1702 = (float*)myMalloc(1 * sizeof(float));;
x1702[0] = 0.0f;
float* x1704 = (float*)myMalloc(1 * sizeof(float));;
x1704[0] = 1.0f;

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
    x1704, in_desc, x1699, filt_desc, x425,
    conv_desc, algo, ws_data, ws_size,
    x1702, out_desc, x1701));
};
float* x1707 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1708 = (float*)myMalloc(1 * sizeof(float));;
x1708[0] = 0.0f;
float* x1710 = (float*)myMalloc(1 * sizeof(float));;
x1710[0] = 1.0f;

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
    x1710, x1710, in_desc, x1701, out_desc, x1707, sbmv_desc, x317,
    x953, x803, x686, 1.0E-5));
};
float* x1713 = (float*)myMalloc(1 * sizeof(float));;
x1713[0] = 1.0f;
float* x1715 = (float*)myMalloc(1 * sizeof(float));;
x1715[0] = 1.0f;

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
    cudnnHandle, x1713, bias_desc, x1663, x1715, out_desc, x1707));
};
float* x1718 = (float*)myMalloc(1 * sizeof(float));;
x1718[0] = 0.0f;
float* x1720 = (float*)myMalloc(1 * sizeof(float));;
x1720[0] = 1.0f;
float* x1722 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1720, x_desc, x1707, x1718, x_desc, x1722));
};
float* x1724 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1725 = (float*)myMalloc(1 * sizeof(float));;
x1725[0] = 0.0f;
float* x1727 = (float*)myMalloc(1 * sizeof(float));;
x1727[0] = 1.0f;

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
    x1727, in_desc, x1722, filt_desc, x911,
    conv_desc, algo, ws_data, ws_size,
    x1725, out_desc, x1724));
};
float* x1730 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1731 = (float*)myMalloc(1 * sizeof(float));;
x1731[0] = 0.0f;
float* x1733 = (float*)myMalloc(1 * sizeof(float));;
x1733[0] = 1.0f;

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
    x1733, x1733, in_desc, x1724, out_desc, x1730, sbmv_desc, x644,
    x848, x791, x779, 1.0E-5));
};
float* x1736 = (float*)myMalloc(1 * sizeof(float));;
x1736[0] = 0.0f;
float* x1738 = (float*)myMalloc(1 * sizeof(float));;
x1738[0] = 1.0f;
float* x1740 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1738, x_desc, x1730, x1736, x_desc, x1740));
};
float* x1742 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1743 = (float*)myMalloc(1 * sizeof(float));;
x1743[0] = 0.0f;
float* x1745 = (float*)myMalloc(1 * sizeof(float));;
x1745[0] = 1.0f;

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
    x1745, in_desc, x1740, filt_desc, x299,
    conv_desc, algo, ws_data, ws_size,
    x1743, out_desc, x1742));
};
float* x1748 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1749 = (float*)myMalloc(1 * sizeof(float));;
x1749[0] = 0.0f;
float* x1751 = (float*)myMalloc(1 * sizeof(float));;
x1751[0] = 1.0f;

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
    x1751, x1751, in_desc, x1742, out_desc, x1748, sbmv_desc, x941,
    x833, x629, x446, 1.0E-5));
};
float* x1754 = (float*)myMalloc(1 * sizeof(float));;
x1754[0] = 0.0f;
float* x1756 = (float*)myMalloc(1 * sizeof(float));;
x1756[0] = 1.0f;
float* x1758 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1756, x_desc, x1748, x1754, x_desc, x1758));
};
float* x1760 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1761 = (float*)myMalloc(1 * sizeof(float));;
x1761[0] = 0.0f;
float* x1763 = (float*)myMalloc(1 * sizeof(float));;
x1763[0] = 1.0f;

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
    x1763, in_desc, x1758, filt_desc, x605,
    conv_desc, algo, ws_data, ws_size,
    x1761, out_desc, x1760));
};
float* x1766 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1767 = (float*)myMalloc(1 * sizeof(float));;
x1767[0] = 0.0f;
float* x1769 = (float*)myMalloc(1 * sizeof(float));;
x1769[0] = 1.0f;

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
    x1769, x1769, in_desc, x1760, out_desc, x1766, sbmv_desc, x1046,
    x428, x677, x821, 1.0E-5));
};
float* x1772 = (float*)myMalloc(1 * sizeof(float));;
x1772[0] = 1.0f;
float* x1774 = (float*)myMalloc(1 * sizeof(float));;
x1774[0] = 1.0f;

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
    cudnnHandle, x1772, bias_desc, x1722, x1774, out_desc, x1766));
};
float* x1777 = (float*)myMalloc(1 * sizeof(float));;
x1777[0] = 0.0f;
float* x1779 = (float*)myMalloc(1 * sizeof(float));;
x1779[0] = 1.0f;
float* x1781 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1779, x_desc, x1766, x1777, x_desc, x1781));
};
float* x1783 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1784 = (float*)myMalloc(1 * sizeof(float));;
x1784[0] = 0.0f;
float* x1786 = (float*)myMalloc(1 * sizeof(float));;
x1786[0] = 1.0f;

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
    x1786, in_desc, x1781, filt_desc, x275,
    conv_desc, algo, ws_data, ws_size,
    x1784, out_desc, x1783));
};
float* x1789 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1790 = (float*)myMalloc(1 * sizeof(float));;
x1790[0] = 0.0f;
float* x1792 = (float*)myMalloc(1 * sizeof(float));;
x1792[0] = 1.0f;

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
    x1792, x1792, in_desc, x1783, out_desc, x1789, sbmv_desc, x533,
    x980, x746, x551, 1.0E-5));
};
float* x1795 = (float*)myMalloc(1 * sizeof(float));;
x1795[0] = 0.0f;
float* x1797 = (float*)myMalloc(1 * sizeof(float));;
x1797[0] = 1.0f;
float* x1799 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1797, x_desc, x1789, x1795, x_desc, x1799));
};
float* x1801 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1802 = (float*)myMalloc(1 * sizeof(float));;
x1802[0] = 0.0f;
float* x1804 = (float*)myMalloc(1 * sizeof(float));;
x1804[0] = 1.0f;

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
    x1804, in_desc, x1799, filt_desc, x1004,
    conv_desc, algo, ws_data, ws_size,
    x1802, out_desc, x1801));
};
float* x1807 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1808 = (float*)myMalloc(1 * sizeof(float));;
x1808[0] = 0.0f;
float* x1810 = (float*)myMalloc(1 * sizeof(float));;
x1810[0] = 1.0f;

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
    x1810, x1810, in_desc, x1801, out_desc, x1807, sbmv_desc, x479,
    x665, x815, x947, 1.0E-5));
};
float* x1813 = (float*)myMalloc(1 * sizeof(float));;
x1813[0] = 0.0f;
float* x1815 = (float*)myMalloc(1 * sizeof(float));;
x1815[0] = 1.0f;
float* x1817 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1815, x_desc, x1807, x1813, x_desc, x1817));
};
float* x1819 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1820 = (float*)myMalloc(1 * sizeof(float));;
x1820[0] = 0.0f;
float* x1822 = (float*)myMalloc(1 * sizeof(float));;
x1822[0] = 1.0f;

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
    x1822, in_desc, x1817, filt_desc, x524,
    conv_desc, algo, ws_data, ws_size,
    x1820, out_desc, x1819));
};
float* x1825 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1826 = (float*)myMalloc(1 * sizeof(float));;
x1826[0] = 0.0f;
float* x1828 = (float*)myMalloc(1 * sizeof(float));;
x1828[0] = 1.0f;

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
    x1828, x1828, in_desc, x1819, out_desc, x1825, sbmv_desc, x971,
    x695, x950, x740, 1.0E-5));
};
float* x1831 = (float*)myMalloc(1 * sizeof(float));;
x1831[0] = 1.0f;
float* x1833 = (float*)myMalloc(1 * sizeof(float));;
x1833[0] = 1.0f;

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
    cudnnHandle, x1831, bias_desc, x1781, x1833, out_desc, x1825));
};
float* x1836 = (float*)myMalloc(1 * sizeof(float));;
x1836[0] = 0.0f;
float* x1838 = (float*)myMalloc(1 * sizeof(float));;
x1838[0] = 1.0f;
float* x1840 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1838, x_desc, x1825, x1836, x_desc, x1840));
};
float* x1842 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1843 = (float*)myMalloc(1 * sizeof(float));;
x1843[0] = 0.0f;
float* x1845 = (float*)myMalloc(1 * sizeof(float));;
x1845[0] = 1.0f;

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
    x1845, in_desc, x1840, filt_desc, x323,
    conv_desc, algo, ws_data, ws_size,
    x1843, out_desc, x1842));
};
float* x1848 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1849 = (float*)myMalloc(1 * sizeof(float));;
x1849[0] = 0.0f;
float* x1851 = (float*)myMalloc(1 * sizeof(float));;
x1851[0] = 1.0f;

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
    x1851, x1851, in_desc, x1842, out_desc, x1848, sbmv_desc, x488,
    x812, x1019, x464, 1.0E-5));
};
float* x1854 = (float*)myMalloc(1 * sizeof(float));;
x1854[0] = 0.0f;
float* x1856 = (float*)myMalloc(1 * sizeof(float));;
x1856[0] = 1.0f;
float* x1858 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1856, x_desc, x1848, x1854, x_desc, x1858));
};
float* x1860 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1861 = (float*)myMalloc(1 * sizeof(float));;
x1861[0] = 0.0f;
float* x1863 = (float*)myMalloc(1 * sizeof(float));;
x1863[0] = 1.0f;

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
    x1863, in_desc, x1858, filt_desc, x1043,
    conv_desc, algo, ws_data, ws_size,
    x1861, out_desc, x1860));
};
float* x1866 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1867 = (float*)myMalloc(1 * sizeof(float));;
x1867[0] = 0.0f;
float* x1869 = (float*)myMalloc(1 * sizeof(float));;
x1869[0] = 1.0f;

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
    x1869, x1869, in_desc, x1860, out_desc, x1866, sbmv_desc, x761,
    x584, x1007, x569, 1.0E-5));
};
float* x1872 = (float*)myMalloc(1 * sizeof(float));;
x1872[0] = 0.0f;
float* x1874 = (float*)myMalloc(1 * sizeof(float));;
x1874[0] = 1.0f;
float* x1876 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1874, x_desc, x1866, x1872, x_desc, x1876));
};
float* x1878 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1879 = (float*)myMalloc(1 * sizeof(float));;
x1879[0] = 0.0f;
float* x1881 = (float*)myMalloc(1 * sizeof(float));;
x1881[0] = 1.0f;

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
    x1881, in_desc, x1876, filt_desc, x920,
    conv_desc, algo, ws_data, ws_size,
    x1879, out_desc, x1878));
};
float* x1884 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1885 = (float*)myMalloc(1 * sizeof(float));;
x1885[0] = 0.0f;
float* x1887 = (float*)myMalloc(1 * sizeof(float));;
x1887[0] = 1.0f;

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
    x1887, x1887, in_desc, x1878, out_desc, x1884, sbmv_desc, x434,
    x617, x884, x1073, 1.0E-5));
};
float* x1890 = (float*)myMalloc(1 * sizeof(float));;
x1890[0] = 1.0f;
float* x1892 = (float*)myMalloc(1 * sizeof(float));;
x1892[0] = 1.0f;

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
    cudnnHandle, x1890, bias_desc, x1840, x1892, out_desc, x1884));
};
float* x1895 = (float*)myMalloc(1 * sizeof(float));;
x1895[0] = 0.0f;
float* x1897 = (float*)myMalloc(1 * sizeof(float));;
x1897[0] = 1.0f;
float* x1899 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1897, x_desc, x1884, x1895, x_desc, x1899));
};
float* x1901 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1902 = (float*)myMalloc(1 * sizeof(float));;
x1902[0] = 0.0f;
float* x1904 = (float*)myMalloc(1 * sizeof(float));;
x1904[0] = 1.0f;

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
    x1904, in_desc, x1899, filt_desc, x710,
    conv_desc, algo, ws_data, ws_size,
    x1902, out_desc, x1901));
};
float* x1907 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1908 = (float*)myMalloc(1 * sizeof(float));;
x1908[0] = 0.0f;
float* x1910 = (float*)myMalloc(1 * sizeof(float));;
x1910[0] = 1.0f;

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
    x1910, x1910, in_desc, x1901, out_desc, x1907, sbmv_desc, x512,
    x1016, x497, x785, 1.0E-5));
};
float* x1913 = (float*)myMalloc(1 * sizeof(float));;
x1913[0] = 0.0f;
float* x1915 = (float*)myMalloc(1 * sizeof(float));;
x1915[0] = 1.0f;
float* x1917 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1915, x_desc, x1907, x1913, x_desc, x1917));
};
float* x1919 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1920 = (float*)myMalloc(1 * sizeof(float));;
x1920[0] = 0.0f;
float* x1922 = (float*)myMalloc(1 * sizeof(float));;
x1922[0] = 1.0f;

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
    x1922, in_desc, x1917, filt_desc, x935,
    conv_desc, algo, ws_data, ws_size,
    x1920, out_desc, x1919));
};
float* x1925 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x1926 = (float*)myMalloc(1 * sizeof(float));;
x1926[0] = 0.0f;
float* x1928 = (float*)myMalloc(1 * sizeof(float));;
x1928[0] = 1.0f;

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
    x1928, x1928, in_desc, x1919, out_desc, x1925, sbmv_desc, x680,
    x824, x467, x977, 1.0E-5));
};
float* x1931 = (float*)myMalloc(1 * sizeof(float));;
x1931[0] = 0.0f;
float* x1933 = (float*)myMalloc(1 * sizeof(float));;
x1933[0] = 1.0f;
float* x1935 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x1933, x_desc, x1925, x1931, x_desc, x1935));
};
float* x1937 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1938 = (float*)myMalloc(1 * sizeof(float));;
x1938[0] = 0.0f;
float* x1940 = (float*)myMalloc(1 * sizeof(float));;
x1940[0] = 1.0f;

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
    x1940, in_desc, x1935, filt_desc, x548,
    conv_desc, algo, ws_data, ws_size,
    x1938, out_desc, x1937));
};
float* x1943 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1944 = (float*)myMalloc(1 * sizeof(float));;
x1944[0] = 0.0f;
float* x1946 = (float*)myMalloc(1 * sizeof(float));;
x1946[0] = 1.0f;

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
    x1946, x1946, in_desc, x1937, out_desc, x1943, sbmv_desc, x1001,
    x536, x623, x806, 1.0E-5));
};
float* x1949 = (float*)myMalloc(1 * sizeof(float));;
x1949[0] = 1.0f;
float* x1951 = (float*)myMalloc(1 * sizeof(float));;
x1951[0] = 1.0f;

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
    cudnnHandle, x1949, bias_desc, x1899, x1951, out_desc, x1943));
};
float* x1954 = (float*)myMalloc(1 * sizeof(float));;
x1954[0] = 0.0f;
float* x1956 = (float*)myMalloc(1 * sizeof(float));;
x1956[0] = 1.0f;
float* x1958 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1956, x_desc, x1943, x1954, x_desc, x1958));
};
float* x1960 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1961 = (float*)myMalloc(1 * sizeof(float));;
x1961[0] = 0.0f;
float* x1963 = (float*)myMalloc(1 * sizeof(float));;
x1963[0] = 1.0f;

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
    x1963, in_desc, x1958, filt_desc, x674,
    conv_desc, algo, ws_data, ws_size,
    x1961, out_desc, x1960));
};
float* x1966 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1967 = (float*)myMalloc(1 * sizeof(float));;
x1967[0] = 0.0f;
float* x1969 = (float*)myMalloc(1 * sizeof(float));;
x1969[0] = 1.0f;

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
    x1969, x1969, in_desc, x1960, out_desc, x1966, sbmv_desc, x860,
    x929, x458, x620, 1.0E-5));
};
float* x1972 = (float*)myMalloc(1 * sizeof(float));;
x1972[0] = 0.0f;
float* x1974 = (float*)myMalloc(1 * sizeof(float));;
x1974[0] = 1.0f;
float* x1976 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1974, x_desc, x1966, x1972, x_desc, x1976));
};
float* x1978 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x1979 = (float*)myMalloc(1 * sizeof(float));;
x1979[0] = 0.0f;
float* x1981 = (float*)myMalloc(1 * sizeof(float));;
x1981[0] = 1.0f;

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
    x1981, in_desc, x1976, filt_desc, x359,
    conv_desc, algo, ws_data, ws_size,
    x1979, out_desc, x1978));
};
float* x1984 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x1985 = (float*)myMalloc(1 * sizeof(float));;
x1985[0] = 0.0f;
float* x1987 = (float*)myMalloc(1 * sizeof(float));;
x1987[0] = 1.0f;

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
    x1987, x1987, in_desc, x1978, out_desc, x1984, sbmv_desc, x872,
    x734, x596, x407, 1.0E-5));
};
float* x1990 = (float*)myMalloc(1 * sizeof(float));;
x1990[0] = 0.0f;
float* x1992 = (float*)myMalloc(1 * sizeof(float));;
x1992[0] = 1.0f;
float* x1994 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x1992, x_desc, x1984, x1990, x_desc, x1994));
};
float* x1996 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1997 = (float*)myMalloc(1 * sizeof(float));;
x1997[0] = 0.0f;
float* x1999 = (float*)myMalloc(1 * sizeof(float));;
x1999[0] = 1.0f;

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
    x1999, in_desc, x1994, filt_desc, x893,
    conv_desc, algo, ws_data, ws_size,
    x1997, out_desc, x1996));
};
float* x2002 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2003 = (float*)myMalloc(1 * sizeof(float));;
x2003[0] = 0.0f;
float* x2005 = (float*)myMalloc(1 * sizeof(float));;
x2005[0] = 1.0f;

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
    x2005, x2005, in_desc, x1996, out_desc, x2002, sbmv_desc, x974,
    x443, x602, x836, 1.0E-5));
};
float* x2008 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2009 = (float*)myMalloc(1 * sizeof(float));;
x2009[0] = 0.0f;
float* x2011 = (float*)myMalloc(1 * sizeof(float));;
x2011[0] = 1.0f;

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
    x2011, in_desc, x1958, filt_desc, x899,
    conv_desc, algo, ws_data, ws_size,
    x2009, out_desc, x2008));
};
float* x2014 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2015 = (float*)myMalloc(1 * sizeof(float));;
x2015[0] = 0.0f;
float* x2017 = (float*)myMalloc(1 * sizeof(float));;
x2017[0] = 1.0f;

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
    x2017, x2017, in_desc, x2008, out_desc, x2014, sbmv_desc, x776,
    x578, x449, x632, 1.0E-5));
};
float* x2020 = (float*)myMalloc(1 * sizeof(float));;
x2020[0] = 1.0f;
float* x2022 = (float*)myMalloc(1 * sizeof(float));;
x2022[0] = 1.0f;

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
    cudnnHandle, x2020, bias_desc, x2014, x2022, out_desc, x2002));
};
float* x2025 = (float*)myMalloc(1 * sizeof(float));;
x2025[0] = 0.0f;
float* x2027 = (float*)myMalloc(1 * sizeof(float));;
x2027[0] = 1.0f;
float* x2029 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x2027, x_desc, x2002, x2025, x_desc, x2029));
};
float* x2031 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2032 = (float*)myMalloc(1 * sizeof(float));;
x2032[0] = 0.0f;
float* x2034 = (float*)myMalloc(1 * sizeof(float));;
x2034[0] = 1.0f;

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
    x2034, in_desc, x2029, filt_desc, x902,
    conv_desc, algo, ws_data, ws_size,
    x2032, out_desc, x2031));
};
float* x2037 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2038 = (float*)myMalloc(1 * sizeof(float));;
x2038[0] = 0.0f;
float* x2040 = (float*)myMalloc(1 * sizeof(float));;
x2040[0] = 1.0f;

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
    x2040, x2040, in_desc, x2031, out_desc, x2037, sbmv_desc, x395,
    x668, x719, x452, 1.0E-5));
};
float* x2043 = (float*)myMalloc(1 * sizeof(float));;
x2043[0] = 0.0f;
float* x2045 = (float*)myMalloc(1 * sizeof(float));;
x2045[0] = 1.0f;
float* x2047 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2045, x_desc, x2037, x2043, x_desc, x2047));
};
float* x2049 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2050 = (float*)myMalloc(1 * sizeof(float));;
x2050[0] = 0.0f;
float* x2052 = (float*)myMalloc(1 * sizeof(float));;
x2052[0] = 1.0f;

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
    x2052, in_desc, x2047, filt_desc, x722,
    conv_desc, algo, ws_data, ws_size,
    x2050, out_desc, x2049));
};
float* x2055 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2056 = (float*)myMalloc(1 * sizeof(float));;
x2056[0] = 0.0f;
float* x2058 = (float*)myMalloc(1 * sizeof(float));;
x2058[0] = 1.0f;

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
    x2058, x2058, in_desc, x2049, out_desc, x2055, sbmv_desc, x737,
    x455, x671, x842, 1.0E-5));
};
float* x2061 = (float*)myMalloc(1 * sizeof(float));;
x2061[0] = 0.0f;
float* x2063 = (float*)myMalloc(1 * sizeof(float));;
x2063[0] = 1.0f;
float* x2065 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2063, x_desc, x2055, x2061, x_desc, x2065));
};
float* x2067 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2068 = (float*)myMalloc(1 * sizeof(float));;
x2068[0] = 0.0f;
float* x2070 = (float*)myMalloc(1 * sizeof(float));;
x2070[0] = 1.0f;

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
    x2070, in_desc, x2065, filt_desc, x398,
    conv_desc, algo, ws_data, ws_size,
    x2068, out_desc, x2067));
};
float* x2073 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2074 = (float*)myMalloc(1 * sizeof(float));;
x2074[0] = 0.0f;
float* x2076 = (float*)myMalloc(1 * sizeof(float));;
x2076[0] = 1.0f;

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
    x2076, x2076, in_desc, x2067, out_desc, x2073, sbmv_desc, x539,
    x689, x461, x992, 1.0E-5));
};
float* x2079 = (float*)myMalloc(1 * sizeof(float));;
x2079[0] = 1.0f;
float* x2081 = (float*)myMalloc(1 * sizeof(float));;
x2081[0] = 1.0f;

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
    cudnnHandle, x2079, bias_desc, x2029, x2081, out_desc, x2073));
};
float* x2084 = (float*)myMalloc(1 * sizeof(float));;
x2084[0] = 0.0f;
float* x2086 = (float*)myMalloc(1 * sizeof(float));;
x2086[0] = 1.0f;
float* x2088 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x2086, x_desc, x2073, x2084, x_desc, x2088));
};
float* x2090 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2091 = (float*)myMalloc(1 * sizeof(float));;
x2091[0] = 0.0f;
float* x2093 = (float*)myMalloc(1 * sizeof(float));;
x2093[0] = 1.0f;

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
    x2093, in_desc, x2088, filt_desc, x1052,
    conv_desc, algo, ws_data, ws_size,
    x2091, out_desc, x2090));
};
float* x2096 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2097 = (float*)myMalloc(1 * sizeof(float));;
x2097[0] = 0.0f;
float* x2099 = (float*)myMalloc(1 * sizeof(float));;
x2099[0] = 1.0f;

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
    x2099, x2099, in_desc, x2090, out_desc, x2096, sbmv_desc, x302,
    x491, x896, x1022, 1.0E-5));
};
float* x2102 = (float*)myMalloc(1 * sizeof(float));;
x2102[0] = 0.0f;
float* x2104 = (float*)myMalloc(1 * sizeof(float));;
x2104[0] = 1.0f;
float* x2106 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2104, x_desc, x2096, x2102, x_desc, x2106));
};
float* x2108 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2109 = (float*)myMalloc(1 * sizeof(float));;
x2109[0] = 0.0f;
float* x2111 = (float*)myMalloc(1 * sizeof(float));;
x2111[0] = 1.0f;

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
    x2111, in_desc, x2106, filt_desc, x341,
    conv_desc, algo, ws_data, ws_size,
    x2109, out_desc, x2108));
};
float* x2114 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2115 = (float*)myMalloc(1 * sizeof(float));;
x2115[0] = 0.0f;
float* x2117 = (float*)myMalloc(1 * sizeof(float));;
x2117[0] = 1.0f;

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
    x2117, x2117, in_desc, x2108, out_desc, x2114, sbmv_desc, x839,
    x764, x293, x863, 1.0E-5));
};
float* x2120 = (float*)myMalloc(1 * sizeof(float));;
x2120[0] = 0.0f;
float* x2122 = (float*)myMalloc(1 * sizeof(float));;
x2122[0] = 1.0f;
float* x2124 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2122, x_desc, x2114, x2120, x_desc, x2124));
};
float* x2126 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2127 = (float*)myMalloc(1 * sizeof(float));;
x2127[0] = 0.0f;
float* x2129 = (float*)myMalloc(1 * sizeof(float));;
x2129[0] = 1.0f;

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
    x2129, in_desc, x2124, filt_desc, x356,
    conv_desc, algo, ws_data, ws_size,
    x2127, out_desc, x2126));
};
float* x2132 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2133 = (float*)myMalloc(1 * sizeof(float));;
x2133[0] = 0.0f;
float* x2135 = (float*)myMalloc(1 * sizeof(float));;
x2135[0] = 1.0f;

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
    x2135, x2135, in_desc, x2126, out_desc, x2132, sbmv_desc, x566,
    x800, x1037, x626, 1.0E-5));
};
float* x2138 = (float*)myMalloc(1 * sizeof(float));;
x2138[0] = 1.0f;
float* x2140 = (float*)myMalloc(1 * sizeof(float));;
x2140[0] = 1.0f;

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
    cudnnHandle, x2138, bias_desc, x2088, x2140, out_desc, x2132));
};
float* x2143 = (float*)myMalloc(1 * sizeof(float));;
x2143[0] = 0.0f;
float* x2145 = (float*)myMalloc(1 * sizeof(float));;
x2145[0] = 1.0f;
float* x2147 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x2145, x_desc, x2132, x2143, x_desc, x2147));
};
float* x2149 = (float*)myMalloc(1 * sizeof(float));;
x2149[0] = 0.0f;
float* x2151 = (float*)myMalloc(1 * sizeof(float));;
x2151[0] = 1.0f;
float* x2153 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2151, in_desc, x2147, x2149, out_desc, x2153));
};
// resize to WrappedArray(64, -1)
// gemm: ArrayBuffer(64, 2048), Vector(10, 2048)
float* x2157 = (float*)myGpuMalloc(640 * sizeof(float));
float* x2158 = (float*)myMalloc(1 * sizeof(float));;
x2158[0] = 0.0f;
float* x2160 = (float*)myMalloc(1 * sizeof(float));;
x2160[0] = 1.0f;
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 10,64,2048,x2160,x938,2048,x2153,2048,x2158,x2157,10));
float* x2163 = (float*)myMalloc(1 * sizeof(float));;
x2163[0] = 1.0f;
float* x2165 = (float*)myMalloc(1 * sizeof(float));;
x2165[0] = 1.0f;

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
    cudnnHandle, x2163, bias_desc, x401, x2165, out_desc, x2157));
};
// Tensor 'toCPU' invocation.
float* x2169 = (float*)myMalloc(640 * sizeof(float));;
CUDA_CALL(cudaMemcpy(x2169, x2157, 640 * sizeof(float), cudaMemcpyDeviceToHost));
printf("output (size 64 x 10)\n");
float x2172 = 0.0f;
for(int x2174=0; x2174 < 640; x2174++) {
float x2175 = x2172;
float x2176 = x2169[x2174];
float x2177 = fabs(x2176);
float x2178 = fabs(x2175);
bool x2179 = x2177 > x2178;
float x2180;
if (x2179) {
x2180 = x2176;
} else {
x2180 = x2175;
}
x2172 = x2180;

}
float x2184 = x2172;
printf("Max Abs: %.5f || ",x2184);
for(int x2186=0; x2186 < 10; x2186++) {
float x2187 = x2169[x2186];
printf("%.5f ",x2187);

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


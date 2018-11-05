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
int64_t x1381 = (long)mallocAddr;
int64_t x1382 = (long)gpuMallocAddr;
// training loop starts here
int32_t x1393 = x12 / 64;
int32_t x5247 = x1393 / 10;
double x5252 = (double)x12;
int64_t x5275 = (int64_t)x12;
float x5279 = (float)x12;
for(int x1385=0; x1385 < 4; x1385++) {
struct timeval begin_1, end_1, diff_1;
float x1387 = 0.0f;
float x1388 = x1387;
float x1389 = x1388;
int32_t x1390 = x1385 + 1;
printf("Start training epoch %d\n",x1390);
gettimeofday(&begin_1, NULL);
for(int x1395=0; x1395 < x1393; x1395++) {
int32_t x1396 = x1395 * 64;
int32_t x1397 = x1396 * 3072;
float* x1398 = x14+x1397;
int* x1399 = x15+x1396;
// Tensor 'toGPU' invocation.
float* x1401 = (float*)myGpuMalloc(196608 * sizeof(float));
CUDA_CALL(cudaMemcpy(x1401, x1398, 196608 * sizeof(float), cudaMemcpyHostToDevice));
float* x1403 = (float*)myGpuMalloc(2 * sizeof(float));
int* x1404 = (int32_t*)myGpuMalloc(64 * sizeof(int32_t));
CUDA_CALL(cudaMemcpy(x1404, x1399, 64 * sizeof(int32_t), cudaMemcpyHostToDevice));
float* x1406 = (float*)myGpuMalloc(1 * sizeof(float));
float* x1407 = (float*)myGpuMalloc(1 * sizeof(float));
// allocate memory to save the final loss in CPU Tensor
float* x1409 = (float*)myMalloc(1 * sizeof(float));;
float* x1410 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1411 = (float*)myMalloc(1 * sizeof(float));;
x1411[0] = 0.0f;
float* x1413 = (float*)myMalloc(1 * sizeof(float));;
x1413[0] = 1.0f;

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
    x1413, in_desc, x1401, filt_desc, x751,
    conv_desc, algo, ws_data, ws_size,
    x1411, out_desc, x1410));
};
float* x1416 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1417 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1418 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1419 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1420 = (float*)myMalloc(1 * sizeof(float));;
x1420[0] = 0.0f;
float* x1422 = (float*)myMalloc(1 * sizeof(float));;
x1422[0] = 1.0f;

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
    x1422, x1420, in_desc, x1410, out_desc, x1417, sbmv_desc, x913,
    x1048, 0.1, x415, x625, 1.0E-5,
    x1418, x1419));
};
float* x1425 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1426 = (float*)myMalloc(1 * sizeof(float));;
x1426[0] = 0.0f;
float* x1428 = (float*)myMalloc(1 * sizeof(float));;
x1428[0] = 1.0f;
float* x1430 = (float*)myGpuMalloc(4194304 * sizeof(float));

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
    x1428, x_desc, x1417, x1426, x_desc, x1430));
};
float* x1432 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1433 = (float*)myMalloc(1 * sizeof(float));;
x1433[0] = 0.0f;
float* x1435 = (float*)myMalloc(1 * sizeof(float));;
x1435[0] = 1.0f;
float* x1437 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1435, in_desc, x1430, x1433, out_desc, x1437));
};
float* x1439 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1440 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1441 = (float*)myMalloc(1 * sizeof(float));;
x1441[0] = 0.0f;
float* x1443 = (float*)myMalloc(1 * sizeof(float));;
x1443[0] = 1.0f;

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
    x1443, in_desc, x1437, filt_desc, x994,
    conv_desc, algo, ws_data, ws_size,
    x1441, out_desc, x1440));
};
float* x1446 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1447 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1448 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1449 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1450 = (float*)myMalloc(1 * sizeof(float));;
x1450[0] = 0.0f;
float* x1452 = (float*)myMalloc(1 * sizeof(float));;
x1452[0] = 1.0f;

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
    x1452, x1450, in_desc, x1440, out_desc, x1447, sbmv_desc, x373,
    x454, 0.1, x637, x448, 1.0E-5,
    x1448, x1449));
};
float* x1455 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1456 = (float*)myMalloc(1 * sizeof(float));;
x1456[0] = 0.0f;
float* x1458 = (float*)myMalloc(1 * sizeof(float));;
x1458[0] = 1.0f;
float* x1460 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1458, x_desc, x1447, x1456, x_desc, x1460));
};
float* x1462 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1463 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1464 = (float*)myMalloc(1 * sizeof(float));;
x1464[0] = 0.0f;
float* x1466 = (float*)myMalloc(1 * sizeof(float));;
x1466[0] = 1.0f;

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
    x1466, in_desc, x1460, filt_desc, x565,
    conv_desc, algo, ws_data, ws_size,
    x1464, out_desc, x1463));
};
float* x1469 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1470 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1471 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1472 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1473 = (float*)myMalloc(1 * sizeof(float));;
x1473[0] = 0.0f;
float* x1475 = (float*)myMalloc(1 * sizeof(float));;
x1475[0] = 1.0f;

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
    x1475, x1473, in_desc, x1463, out_desc, x1470, sbmv_desc, x787,
    x442, 0.1, x610, x769, 1.0E-5,
    x1471, x1472));
};
float* x1478 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1479 = (float*)myMalloc(1 * sizeof(float));;
x1479[0] = 0.0f;
float* x1481 = (float*)myMalloc(1 * sizeof(float));;
x1481[0] = 1.0f;
float* x1483 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1481, x_desc, x1470, x1479, x_desc, x1483));
};
float* x1485 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1486 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1487 = (float*)myMalloc(1 * sizeof(float));;
x1487[0] = 0.0f;
float* x1489 = (float*)myMalloc(1 * sizeof(float));;
x1489[0] = 1.0f;

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
    x1489, in_desc, x1483, filt_desc, x391,
    conv_desc, algo, ws_data, ws_size,
    x1487, out_desc, x1486));
};
float* x1492 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1493 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1494 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1495 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1496 = (float*)myMalloc(1 * sizeof(float));;
x1496[0] = 0.0f;
float* x1498 = (float*)myMalloc(1 * sizeof(float));;
x1498[0] = 1.0f;

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
    x1498, x1496, in_desc, x1486, out_desc, x1493, sbmv_desc, x892,
    x673, 0.1, x508, x403, 1.0E-5,
    x1494, x1495));
};
float* x1501 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1502 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1503 = (float*)myMalloc(1 * sizeof(float));;
x1503[0] = 0.0f;
float* x1505 = (float*)myMalloc(1 * sizeof(float));;
x1505[0] = 1.0f;

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
    x1505, in_desc, x1437, filt_desc, x781,
    conv_desc, algo, ws_data, ws_size,
    x1503, out_desc, x1502));
};
float* x1508 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1509 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1510 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1511 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1512 = (float*)myMalloc(1 * sizeof(float));;
x1512[0] = 0.0f;
float* x1514 = (float*)myMalloc(1 * sizeof(float));;
x1514[0] = 1.0f;

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
    x1514, x1512, in_desc, x1502, out_desc, x1509, sbmv_desc, x523,
    x904, 0.1, x1087, x1024, 1.0E-5,
    x1510, x1511));
};
float* x1517 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1518 = (float*)myMalloc(1 * sizeof(float));;
x1518[0] = 1.0f;
float* x1520 = (float*)myMalloc(1 * sizeof(float));;
x1520[0] = 1.0f;

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
    cudnnHandle, x1518, bias_desc, x1509, x1520, out_desc, x1493));
};
float* x1523 = (float*)myMalloc(1 * sizeof(float));;
x1523[0] = 0.0f;
float* x1525 = (float*)myMalloc(1 * sizeof(float));;
x1525[0] = 1.0f;
float* x1527 = (float*)myGpuMalloc(4194304 * sizeof(float));

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
    x1525, x_desc, x1493, x1523, x_desc, x1527));
};
float* x1529 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1530 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1531 = (float*)myMalloc(1 * sizeof(float));;
x1531[0] = 0.0f;
float* x1533 = (float*)myMalloc(1 * sizeof(float));;
x1533[0] = 1.0f;

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
    x1533, in_desc, x1527, filt_desc, x808,
    conv_desc, algo, ws_data, ws_size,
    x1531, out_desc, x1530));
};
float* x1536 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1537 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1538 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1539 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1540 = (float*)myMalloc(1 * sizeof(float));;
x1540[0] = 0.0f;
float* x1542 = (float*)myMalloc(1 * sizeof(float));;
x1542[0] = 1.0f;

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
    x1542, x1540, in_desc, x1530, out_desc, x1537, sbmv_desc, x721,
    x475, 0.1, x325, x601, 1.0E-5,
    x1538, x1539));
};
float* x1545 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1546 = (float*)myMalloc(1 * sizeof(float));;
x1546[0] = 0.0f;
float* x1548 = (float*)myMalloc(1 * sizeof(float));;
x1548[0] = 1.0f;
float* x1550 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1548, x_desc, x1537, x1546, x_desc, x1550));
};
float* x1552 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1553 = (float*)myGpuMalloc(1048576 * sizeof(float));
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
    x1556, in_desc, x1550, filt_desc, x544,
    conv_desc, algo, ws_data, ws_size,
    x1554, out_desc, x1553));
};
float* x1559 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1560 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1561 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1562 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1563 = (float*)myMalloc(1 * sizeof(float));;
x1563[0] = 0.0f;
float* x1565 = (float*)myMalloc(1 * sizeof(float));;
x1565[0] = 1.0f;

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
    x1565, x1563, in_desc, x1553, out_desc, x1560, sbmv_desc, x919,
    x754, 0.1, x427, x1027, 1.0E-5,
    x1561, x1562));
};
float* x1568 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1569 = (float*)myMalloc(1 * sizeof(float));;
x1569[0] = 0.0f;
float* x1571 = (float*)myMalloc(1 * sizeof(float));;
x1571[0] = 1.0f;
float* x1573 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1571, x_desc, x1560, x1569, x_desc, x1573));
};
float* x1575 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1576 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1577 = (float*)myMalloc(1 * sizeof(float));;
x1577[0] = 0.0f;
float* x1579 = (float*)myMalloc(1 * sizeof(float));;
x1579[0] = 1.0f;

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
    x1579, in_desc, x1573, filt_desc, x685,
    conv_desc, algo, ws_data, ws_size,
    x1577, out_desc, x1576));
};
float* x1582 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1583 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1584 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1585 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1586 = (float*)myMalloc(1 * sizeof(float));;
x1586[0] = 0.0f;
float* x1588 = (float*)myMalloc(1 * sizeof(float));;
x1588[0] = 1.0f;

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
    x1588, x1586, in_desc, x1576, out_desc, x1583, sbmv_desc, x469,
    x316, 0.1, x568, x793, 1.0E-5,
    x1584, x1585));
};
float* x1591 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1592 = (float*)myMalloc(1 * sizeof(float));;
x1592[0] = 1.0f;
float* x1594 = (float*)myMalloc(1 * sizeof(float));;
x1594[0] = 1.0f;

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
    cudnnHandle, x1592, bias_desc, x1527, x1594, out_desc, x1583));
};
float* x1597 = (float*)myMalloc(1 * sizeof(float));;
x1597[0] = 0.0f;
float* x1599 = (float*)myMalloc(1 * sizeof(float));;
x1599[0] = 1.0f;
float* x1601 = (float*)myGpuMalloc(4194304 * sizeof(float));

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
    x1599, x_desc, x1583, x1597, x_desc, x1601));
};
float* x1603 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1604 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1605 = (float*)myMalloc(1 * sizeof(float));;
x1605[0] = 0.0f;
float* x1607 = (float*)myMalloc(1 * sizeof(float));;
x1607[0] = 1.0f;

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
    x1607, in_desc, x1601, filt_desc, x745,
    conv_desc, algo, ws_data, ws_size,
    x1605, out_desc, x1604));
};
float* x1610 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1611 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1612 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1613 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1614 = (float*)myMalloc(1 * sizeof(float));;
x1614[0] = 0.0f;
float* x1616 = (float*)myMalloc(1 * sizeof(float));;
x1616[0] = 1.0f;

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
    x1616, x1614, in_desc, x1604, out_desc, x1611, sbmv_desc, x538,
    x367, 0.1, x1066, x856, 1.0E-5,
    x1612, x1613));
};
float* x1619 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1620 = (float*)myMalloc(1 * sizeof(float));;
x1620[0] = 0.0f;
float* x1622 = (float*)myMalloc(1 * sizeof(float));;
x1622[0] = 1.0f;
float* x1624 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1622, x_desc, x1611, x1620, x_desc, x1624));
};
float* x1626 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1627 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1628 = (float*)myMalloc(1 * sizeof(float));;
x1628[0] = 0.0f;
float* x1630 = (float*)myMalloc(1 * sizeof(float));;
x1630[0] = 1.0f;

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
    x1630, in_desc, x1624, filt_desc, x514,
    conv_desc, algo, ws_data, ws_size,
    x1628, out_desc, x1627));
};
float* x1633 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1634 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1635 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1636 = (float*)myGpuMalloc(64 * sizeof(float));
float* x1637 = (float*)myMalloc(1 * sizeof(float));;
x1637[0] = 0.0f;
float* x1639 = (float*)myMalloc(1 * sizeof(float));;
x1639[0] = 1.0f;

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
    x1639, x1637, in_desc, x1627, out_desc, x1634, sbmv_desc, x511,
    x700, 0.1, x832, x649, 1.0E-5,
    x1635, x1636));
};
float* x1642 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1643 = (float*)myMalloc(1 * sizeof(float));;
x1643[0] = 0.0f;
float* x1645 = (float*)myMalloc(1 * sizeof(float));;
x1645[0] = 1.0f;
float* x1647 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x1645, x_desc, x1634, x1643, x_desc, x1647));
};
float* x1649 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1650 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1651 = (float*)myMalloc(1 * sizeof(float));;
x1651[0] = 0.0f;
float* x1653 = (float*)myMalloc(1 * sizeof(float));;
x1653[0] = 1.0f;

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
    x1653, in_desc, x1647, filt_desc, x556,
    conv_desc, algo, ws_data, ws_size,
    x1651, out_desc, x1650));
};
float* x1656 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1657 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1658 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1659 = (float*)myGpuMalloc(256 * sizeof(float));
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
    x1662, x1660, in_desc, x1650, out_desc, x1657, sbmv_desc, x406,
    x1036, 0.1, x847, x694, 1.0E-5,
    x1658, x1659));
};
float* x1665 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1666 = (float*)myMalloc(1 * sizeof(float));;
x1666[0] = 1.0f;
float* x1668 = (float*)myMalloc(1 * sizeof(float));;
x1668[0] = 1.0f;

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
    cudnnHandle, x1666, bias_desc, x1601, x1668, out_desc, x1657));
};
float* x1671 = (float*)myMalloc(1 * sizeof(float));;
x1671[0] = 0.0f;
float* x1673 = (float*)myMalloc(1 * sizeof(float));;
x1673[0] = 1.0f;
float* x1675 = (float*)myGpuMalloc(4194304 * sizeof(float));

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
    x1673, x_desc, x1657, x1671, x_desc, x1675));
};
float* x1677 = (float*)myGpuMalloc(4194304 * sizeof(float));
float* x1678 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1679 = (float*)myMalloc(1 * sizeof(float));;
x1679[0] = 0.0f;
float* x1681 = (float*)myMalloc(1 * sizeof(float));;
x1681[0] = 1.0f;

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
    x1681, in_desc, x1675, filt_desc, x328,
    conv_desc, algo, ws_data, ws_size,
    x1679, out_desc, x1678));
};
float* x1684 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1685 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1686 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1687 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1688 = (float*)myMalloc(1 * sizeof(float));;
x1688[0] = 0.0f;
float* x1690 = (float*)myMalloc(1 * sizeof(float));;
x1690[0] = 1.0f;

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
    x1690, x1688, in_desc, x1678, out_desc, x1685, sbmv_desc, x547,
    x811, 0.1, x907, x697, 1.0E-5,
    x1686, x1687));
};
float* x1693 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1694 = (float*)myMalloc(1 * sizeof(float));;
x1694[0] = 0.0f;
float* x1696 = (float*)myMalloc(1 * sizeof(float));;
x1696[0] = 1.0f;
float* x1698 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x1696, x_desc, x1685, x1694, x_desc, x1698));
};
float* x1700 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1701 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1702 = (float*)myMalloc(1 * sizeof(float));;
x1702[0] = 0.0f;
float* x1704 = (float*)myMalloc(1 * sizeof(float));;
x1704[0] = 1.0f;

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
    x1704, in_desc, x1698, filt_desc, x376,
    conv_desc, algo, ws_data, ws_size,
    x1702, out_desc, x1701));
};
float* x1707 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1708 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1709 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1710 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1711 = (float*)myMalloc(1 * sizeof(float));;
x1711[0] = 0.0f;
float* x1713 = (float*)myMalloc(1 * sizeof(float));;
x1713[0] = 1.0f;

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
    x1713, x1711, in_desc, x1701, out_desc, x1708, sbmv_desc, x1051,
    x865, 0.1, x679, x424, 1.0E-5,
    x1709, x1710));
};
float* x1716 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1717 = (float*)myMalloc(1 * sizeof(float));;
x1717[0] = 0.0f;
float* x1719 = (float*)myMalloc(1 * sizeof(float));;
x1719[0] = 1.0f;
float* x1721 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1719, x_desc, x1708, x1717, x_desc, x1721));
};
float* x1723 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1724 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1725 = (float*)myMalloc(1 * sizeof(float));;
x1725[0] = 0.0f;
float* x1727 = (float*)myMalloc(1 * sizeof(float));;
x1727[0] = 1.0f;

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
    x1727, in_desc, x1721, filt_desc, x613,
    conv_desc, algo, ws_data, ws_size,
    x1725, out_desc, x1724));
};
float* x1730 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1731 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1732 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1733 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1734 = (float*)myMalloc(1 * sizeof(float));;
x1734[0] = 0.0f;
float* x1736 = (float*)myMalloc(1 * sizeof(float));;
x1736[0] = 1.0f;

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
    x1736, x1734, in_desc, x1724, out_desc, x1731, sbmv_desc, x730,
    x925, 0.1, x742, x598, 1.0E-5,
    x1732, x1733));
};
float* x1739 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1740 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1741 = (float*)myMalloc(1 * sizeof(float));;
x1741[0] = 0.0f;
float* x1743 = (float*)myMalloc(1 * sizeof(float));;
x1743[0] = 1.0f;

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
    x1743, in_desc, x1675, filt_desc, x1069,
    conv_desc, algo, ws_data, ws_size,
    x1741, out_desc, x1740));
};
float* x1746 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1747 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1748 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1749 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1750 = (float*)myMalloc(1 * sizeof(float));;
x1750[0] = 0.0f;
float* x1752 = (float*)myMalloc(1 * sizeof(float));;
x1752[0] = 1.0f;

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
    x1752, x1750, in_desc, x1740, out_desc, x1747, sbmv_desc, x916,
    x652, 0.1, x421, x364, 1.0E-5,
    x1748, x1749));
};
float* x1755 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1756 = (float*)myMalloc(1 * sizeof(float));;
x1756[0] = 1.0f;
float* x1758 = (float*)myMalloc(1 * sizeof(float));;
x1758[0] = 1.0f;

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
    cudnnHandle, x1756, bias_desc, x1747, x1758, out_desc, x1731));
};
float* x1761 = (float*)myMalloc(1 * sizeof(float));;
x1761[0] = 0.0f;
float* x1763 = (float*)myMalloc(1 * sizeof(float));;
x1763[0] = 1.0f;
float* x1765 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x1763, x_desc, x1731, x1761, x_desc, x1765));
};
float* x1767 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1768 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1769 = (float*)myMalloc(1 * sizeof(float));;
x1769[0] = 0.0f;
float* x1771 = (float*)myMalloc(1 * sizeof(float));;
x1771[0] = 1.0f;

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
    x1771, in_desc, x1765, filt_desc, x1063,
    conv_desc, algo, ws_data, ws_size,
    x1769, out_desc, x1768));
};
float* x1774 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1775 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1776 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1777 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1778 = (float*)myMalloc(1 * sizeof(float));;
x1778[0] = 0.0f;
float* x1780 = (float*)myMalloc(1 * sizeof(float));;
x1780[0] = 1.0f;

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
    x1780, x1778, in_desc, x1768, out_desc, x1775, sbmv_desc, x961,
    x346, 0.1, x595, x826, 1.0E-5,
    x1776, x1777));
};
float* x1783 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1784 = (float*)myMalloc(1 * sizeof(float));;
x1784[0] = 0.0f;
float* x1786 = (float*)myMalloc(1 * sizeof(float));;
x1786[0] = 1.0f;
float* x1788 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1786, x_desc, x1775, x1784, x_desc, x1788));
};
float* x1790 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1791 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1792 = (float*)myMalloc(1 * sizeof(float));;
x1792[0] = 0.0f;
float* x1794 = (float*)myMalloc(1 * sizeof(float));;
x1794[0] = 1.0f;

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
    x1794, in_desc, x1788, filt_desc, x1000,
    conv_desc, algo, ws_data, ws_size,
    x1792, out_desc, x1791));
};
float* x1797 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1798 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1799 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1800 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1801 = (float*)myMalloc(1 * sizeof(float));;
x1801[0] = 0.0f;
float* x1803 = (float*)myMalloc(1 * sizeof(float));;
x1803[0] = 1.0f;

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
    x1803, x1801, in_desc, x1791, out_desc, x1798, sbmv_desc, x319,
    x580, 0.1, x400, x970, 1.0E-5,
    x1799, x1800));
};
float* x1806 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1807 = (float*)myMalloc(1 * sizeof(float));;
x1807[0] = 0.0f;
float* x1809 = (float*)myMalloc(1 * sizeof(float));;
x1809[0] = 1.0f;
float* x1811 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1809, x_desc, x1798, x1807, x_desc, x1811));
};
float* x1813 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1814 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1815 = (float*)myMalloc(1 * sizeof(float));;
x1815[0] = 0.0f;
float* x1817 = (float*)myMalloc(1 * sizeof(float));;
x1817[0] = 1.0f;

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
    x1817, in_desc, x1811, filt_desc, x628,
    conv_desc, algo, ws_data, ws_size,
    x1815, out_desc, x1814));
};
float* x1820 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1821 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1822 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1823 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1824 = (float*)myMalloc(1 * sizeof(float));;
x1824[0] = 0.0f;
float* x1826 = (float*)myMalloc(1 * sizeof(float));;
x1826[0] = 1.0f;

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
    x1826, x1824, in_desc, x1814, out_desc, x1821, sbmv_desc, x451,
    x1033, 0.1, x736, x559, 1.0E-5,
    x1822, x1823));
};
float* x1829 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1830 = (float*)myMalloc(1 * sizeof(float));;
x1830[0] = 1.0f;
float* x1832 = (float*)myMalloc(1 * sizeof(float));;
x1832[0] = 1.0f;

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
    cudnnHandle, x1830, bias_desc, x1765, x1832, out_desc, x1821));
};
float* x1835 = (float*)myMalloc(1 * sizeof(float));;
x1835[0] = 0.0f;
float* x1837 = (float*)myMalloc(1 * sizeof(float));;
x1837[0] = 1.0f;
float* x1839 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x1837, x_desc, x1821, x1835, x_desc, x1839));
};
float* x1841 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1842 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1843 = (float*)myMalloc(1 * sizeof(float));;
x1843[0] = 0.0f;
float* x1845 = (float*)myMalloc(1 * sizeof(float));;
x1845[0] = 1.0f;

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
    x1845, in_desc, x1839, filt_desc, x883,
    conv_desc, algo, ws_data, ws_size,
    x1843, out_desc, x1842));
};
float* x1848 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1849 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1850 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1851 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1852 = (float*)myMalloc(1 * sizeof(float));;
x1852[0] = 0.0f;
float* x1854 = (float*)myMalloc(1 * sizeof(float));;
x1854[0] = 1.0f;

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
    x1854, x1852, in_desc, x1842, out_desc, x1849, sbmv_desc, x430,
    x805, 0.1, x631, x322, 1.0E-5,
    x1850, x1851));
};
float* x1857 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1858 = (float*)myMalloc(1 * sizeof(float));;
x1858[0] = 0.0f;
float* x1860 = (float*)myMalloc(1 * sizeof(float));;
x1860[0] = 1.0f;
float* x1862 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1860, x_desc, x1849, x1858, x_desc, x1862));
};
float* x1864 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1865 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1866 = (float*)myMalloc(1 * sizeof(float));;
x1866[0] = 0.0f;
float* x1868 = (float*)myMalloc(1 * sizeof(float));;
x1868[0] = 1.0f;

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
    x1868, in_desc, x1862, filt_desc, x868,
    conv_desc, algo, ws_data, ws_size,
    x1866, out_desc, x1865));
};
float* x1871 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1872 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1873 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1874 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1875 = (float*)myMalloc(1 * sizeof(float));;
x1875[0] = 0.0f;
float* x1877 = (float*)myMalloc(1 * sizeof(float));;
x1877[0] = 1.0f;

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
    x1877, x1875, in_desc, x1865, out_desc, x1872, sbmv_desc, x676,
    x478, 0.1, x946, x1093, 1.0E-5,
    x1873, x1874));
};
float* x1880 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1881 = (float*)myMalloc(1 * sizeof(float));;
x1881[0] = 0.0f;
float* x1883 = (float*)myMalloc(1 * sizeof(float));;
x1883[0] = 1.0f;
float* x1885 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1883, x_desc, x1872, x1881, x_desc, x1885));
};
float* x1887 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1888 = (float*)myGpuMalloc(2097152 * sizeof(float));
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
    x1891, in_desc, x1885, filt_desc, x418,
    conv_desc, algo, ws_data, ws_size,
    x1889, out_desc, x1888));
};
float* x1894 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1895 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1896 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1897 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1898 = (float*)myMalloc(1 * sizeof(float));;
x1898[0] = 0.0f;
float* x1900 = (float*)myMalloc(1 * sizeof(float));;
x1900[0] = 1.0f;

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
    x1900, x1898, in_desc, x1888, out_desc, x1895, sbmv_desc, x796,
    x541, 0.1, x370, x964, 1.0E-5,
    x1896, x1897));
};
float* x1903 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1904 = (float*)myMalloc(1 * sizeof(float));;
x1904[0] = 1.0f;
float* x1906 = (float*)myMalloc(1 * sizeof(float));;
x1906[0] = 1.0f;

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
    cudnnHandle, x1904, bias_desc, x1839, x1906, out_desc, x1895));
};
float* x1909 = (float*)myMalloc(1 * sizeof(float));;
x1909[0] = 0.0f;
float* x1911 = (float*)myMalloc(1 * sizeof(float));;
x1911[0] = 1.0f;
float* x1913 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x1911, x_desc, x1895, x1909, x_desc, x1913));
};
float* x1915 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1916 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1917 = (float*)myMalloc(1 * sizeof(float));;
x1917[0] = 0.0f;
float* x1919 = (float*)myMalloc(1 * sizeof(float));;
x1919[0] = 1.0f;

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
    x1919, in_desc, x1913, filt_desc, x691,
    conv_desc, algo, ws_data, ws_size,
    x1917, out_desc, x1916));
};
float* x1922 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1923 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1924 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1925 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1926 = (float*)myMalloc(1 * sizeof(float));;
x1926[0] = 0.0f;
float* x1928 = (float*)myMalloc(1 * sizeof(float));;
x1928[0] = 1.0f;

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
    x1928, x1926, in_desc, x1916, out_desc, x1923, sbmv_desc, x412,
    x1021, 0.1, x1003, x1078, 1.0E-5,
    x1924, x1925));
};
float* x1931 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1932 = (float*)myMalloc(1 * sizeof(float));;
x1932[0] = 0.0f;
float* x1934 = (float*)myMalloc(1 * sizeof(float));;
x1934[0] = 1.0f;
float* x1936 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1934, x_desc, x1923, x1932, x_desc, x1936));
};
float* x1938 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1939 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1940 = (float*)myMalloc(1 * sizeof(float));;
x1940[0] = 0.0f;
float* x1942 = (float*)myMalloc(1 * sizeof(float));;
x1942[0] = 1.0f;

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
    x1942, in_desc, x1936, filt_desc, x790,
    conv_desc, algo, ws_data, ws_size,
    x1940, out_desc, x1939));
};
float* x1945 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1946 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1947 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1948 = (float*)myGpuMalloc(128 * sizeof(float));
float* x1949 = (float*)myMalloc(1 * sizeof(float));;
x1949[0] = 0.0f;
float* x1951 = (float*)myMalloc(1 * sizeof(float));;
x1951[0] = 1.0f;

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
    x1951, x1949, in_desc, x1939, out_desc, x1946, sbmv_desc, x532,
    x409, 0.1, x1099, x739, 1.0E-5,
    x1947, x1948));
};
float* x1954 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1955 = (float*)myMalloc(1 * sizeof(float));;
x1955[0] = 0.0f;
float* x1957 = (float*)myMalloc(1 * sizeof(float));;
x1957[0] = 1.0f;
float* x1959 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x1957, x_desc, x1946, x1955, x_desc, x1959));
};
float* x1961 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x1962 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1963 = (float*)myMalloc(1 * sizeof(float));;
x1963[0] = 0.0f;
float* x1965 = (float*)myMalloc(1 * sizeof(float));;
x1965[0] = 1.0f;

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
    x1965, in_desc, x1959, filt_desc, x460,
    conv_desc, algo, ws_data, ws_size,
    x1963, out_desc, x1962));
};
float* x1968 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1969 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1970 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1971 = (float*)myGpuMalloc(512 * sizeof(float));
float* x1972 = (float*)myMalloc(1 * sizeof(float));;
x1972[0] = 0.0f;
float* x1974 = (float*)myMalloc(1 * sizeof(float));;
x1974[0] = 1.0f;

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
    x1974, x1972, in_desc, x1962, out_desc, x1969, sbmv_desc, x763,
    x457, 0.1, x352, x997, 1.0E-5,
    x1970, x1971));
};
float* x1977 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1978 = (float*)myMalloc(1 * sizeof(float));;
x1978[0] = 1.0f;
float* x1980 = (float*)myMalloc(1 * sizeof(float));;
x1980[0] = 1.0f;

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
    cudnnHandle, x1978, bias_desc, x1913, x1980, out_desc, x1969));
};
float* x1983 = (float*)myMalloc(1 * sizeof(float));;
x1983[0] = 0.0f;
float* x1985 = (float*)myMalloc(1 * sizeof(float));;
x1985[0] = 1.0f;
float* x1987 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x1985, x_desc, x1969, x1983, x_desc, x1987));
};
float* x1989 = (float*)myGpuMalloc(2097152 * sizeof(float));
float* x1990 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1991 = (float*)myMalloc(1 * sizeof(float));;
x1991[0] = 0.0f;
float* x1993 = (float*)myMalloc(1 * sizeof(float));;
x1993[0] = 1.0f;

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
    x1993, in_desc, x1987, filt_desc, x835,
    conv_desc, algo, ws_data, ws_size,
    x1991, out_desc, x1990));
};
float* x1996 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1997 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x1998 = (float*)myGpuMalloc(256 * sizeof(float));
float* x1999 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2000 = (float*)myMalloc(1 * sizeof(float));;
x2000[0] = 0.0f;
float* x2002 = (float*)myMalloc(1 * sizeof(float));;
x2002[0] = 1.0f;

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
    x2002, x2000, in_desc, x1990, out_desc, x1997, sbmv_desc, x1105,
    x358, 0.1, x688, x889, 1.0E-5,
    x1998, x1999));
};
float* x2005 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2006 = (float*)myMalloc(1 * sizeof(float));;
x2006[0] = 0.0f;
float* x2008 = (float*)myMalloc(1 * sizeof(float));;
x2008[0] = 1.0f;
float* x2010 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x2008, x_desc, x1997, x2006, x_desc, x2010));
};
float* x2012 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2013 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2014 = (float*)myMalloc(1 * sizeof(float));;
x2014[0] = 0.0f;
float* x2016 = (float*)myMalloc(1 * sizeof(float));;
x2016[0] = 1.0f;

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
    x2016, in_desc, x2010, filt_desc, x820,
    conv_desc, algo, ws_data, ws_size,
    x2014, out_desc, x2013));
};
float* x2019 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2020 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2021 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2022 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2023 = (float*)myMalloc(1 * sizeof(float));;
x2023[0] = 0.0f;
float* x2025 = (float*)myMalloc(1 * sizeof(float));;
x2025[0] = 1.0f;

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
    x2025, x2023, in_desc, x2013, out_desc, x2020, sbmv_desc, x619,
    x343, 0.1, x982, x592, 1.0E-5,
    x2021, x2022));
};
float* x2028 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2029 = (float*)myMalloc(1 * sizeof(float));;
x2029[0] = 0.0f;
float* x2031 = (float*)myMalloc(1 * sizeof(float));;
x2031[0] = 1.0f;
float* x2033 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x2031, x_desc, x2020, x2029, x_desc, x2033));
};
float* x2035 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2036 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2037 = (float*)myMalloc(1 * sizeof(float));;
x2037[0] = 0.0f;
float* x2039 = (float*)myMalloc(1 * sizeof(float));;
x2039[0] = 1.0f;

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
    x2039, in_desc, x2033, filt_desc, x1102,
    conv_desc, algo, ws_data, ws_size,
    x2037, out_desc, x2036));
};
float* x2042 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2043 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2044 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2045 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2046 = (float*)myMalloc(1 * sizeof(float));;
x2046[0] = 0.0f;
float* x2048 = (float*)myMalloc(1 * sizeof(float));;
x2048[0] = 1.0f;

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
    x2048, x2046, in_desc, x2036, out_desc, x2043, sbmv_desc, x349,
    x646, 0.1, x943, x1096, 1.0E-5,
    x2044, x2045));
};
float* x2051 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2052 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2053 = (float*)myMalloc(1 * sizeof(float));;
x2053[0] = 0.0f;
float* x2055 = (float*)myMalloc(1 * sizeof(float));;
x2055[0] = 1.0f;

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
    x2055, in_desc, x1987, filt_desc, x520,
    conv_desc, algo, ws_data, ws_size,
    x2053, out_desc, x2052));
};
float* x2058 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2059 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2060 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2061 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2062 = (float*)myMalloc(1 * sizeof(float));;
x2062[0] = 0.0f;
float* x2064 = (float*)myMalloc(1 * sizeof(float));;
x2064[0] = 1.0f;

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
    x2064, x2062, in_desc, x2052, out_desc, x2059, sbmv_desc, x382,
    x955, 0.1, x553, x928, 1.0E-5,
    x2060, x2061));
};
float* x2067 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2068 = (float*)myMalloc(1 * sizeof(float));;
x2068[0] = 1.0f;
float* x2070 = (float*)myMalloc(1 * sizeof(float));;
x2070[0] = 1.0f;

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
    cudnnHandle, x2068, bias_desc, x2059, x2070, out_desc, x2043));
};
float* x2073 = (float*)myMalloc(1 * sizeof(float));;
x2073[0] = 0.0f;
float* x2075 = (float*)myMalloc(1 * sizeof(float));;
x2075[0] = 1.0f;
float* x2077 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x2075, x_desc, x2043, x2073, x_desc, x2077));
};
float* x2079 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2080 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2081 = (float*)myMalloc(1 * sizeof(float));;
x2081[0] = 0.0f;
float* x2083 = (float*)myMalloc(1 * sizeof(float));;
x2083[0] = 1.0f;

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
    x2083, in_desc, x2077, filt_desc, x334,
    conv_desc, algo, ws_data, ws_size,
    x2081, out_desc, x2080));
};
float* x2086 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2087 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2088 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2089 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2090 = (float*)myMalloc(1 * sizeof(float));;
x2090[0] = 0.0f;
float* x2092 = (float*)myMalloc(1 * sizeof(float));;
x2092[0] = 1.0f;

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
    x2092, x2090, in_desc, x2080, out_desc, x2087, sbmv_desc, x385,
    x952, 0.1, x1072, x766, 1.0E-5,
    x2088, x2089));
};
float* x2095 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2096 = (float*)myMalloc(1 * sizeof(float));;
x2096[0] = 0.0f;
float* x2098 = (float*)myMalloc(1 * sizeof(float));;
x2098[0] = 1.0f;
float* x2100 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x2098, x_desc, x2087, x2096, x_desc, x2100));
};
float* x2102 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2103 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2104 = (float*)myMalloc(1 * sizeof(float));;
x2104[0] = 0.0f;
float* x2106 = (float*)myMalloc(1 * sizeof(float));;
x2106[0] = 1.0f;

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
    x2106, in_desc, x2100, filt_desc, x388,
    conv_desc, algo, ws_data, ws_size,
    x2104, out_desc, x2103));
};
float* x2109 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2110 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2111 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2112 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2113 = (float*)myMalloc(1 * sizeof(float));;
x2113[0] = 0.0f;
float* x2115 = (float*)myMalloc(1 * sizeof(float));;
x2115[0] = 1.0f;

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
    x2115, x2113, in_desc, x2103, out_desc, x2110, sbmv_desc, x1108,
    x583, 0.1, x895, x1006, 1.0E-5,
    x2111, x2112));
};
float* x2118 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2119 = (float*)myMalloc(1 * sizeof(float));;
x2119[0] = 0.0f;
float* x2121 = (float*)myMalloc(1 * sizeof(float));;
x2121[0] = 1.0f;
float* x2123 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x2121, x_desc, x2110, x2119, x_desc, x2123));
};
float* x2125 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2126 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2127 = (float*)myMalloc(1 * sizeof(float));;
x2127[0] = 0.0f;
float* x2129 = (float*)myMalloc(1 * sizeof(float));;
x2129[0] = 1.0f;

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
    x2129, in_desc, x2123, filt_desc, x463,
    conv_desc, algo, ws_data, ws_size,
    x2127, out_desc, x2126));
};
float* x2132 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2133 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2134 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2135 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2136 = (float*)myMalloc(1 * sizeof(float));;
x2136[0] = 0.0f;
float* x2138 = (float*)myMalloc(1 * sizeof(float));;
x2138[0] = 1.0f;

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
    x2138, x2136, in_desc, x2126, out_desc, x2133, sbmv_desc, x355,
    x991, 0.1, x841, x724, 1.0E-5,
    x2134, x2135));
};
float* x2141 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2142 = (float*)myMalloc(1 * sizeof(float));;
x2142[0] = 1.0f;
float* x2144 = (float*)myMalloc(1 * sizeof(float));;
x2144[0] = 1.0f;

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
    cudnnHandle, x2142, bias_desc, x2077, x2144, out_desc, x2133));
};
float* x2147 = (float*)myMalloc(1 * sizeof(float));;
x2147[0] = 0.0f;
float* x2149 = (float*)myMalloc(1 * sizeof(float));;
x2149[0] = 1.0f;
float* x2151 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x2149, x_desc, x2133, x2147, x_desc, x2151));
};
float* x2153 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2154 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2155 = (float*)myMalloc(1 * sizeof(float));;
x2155[0] = 0.0f;
float* x2157 = (float*)myMalloc(1 * sizeof(float));;
x2157[0] = 1.0f;

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
    x2157, in_desc, x2151, filt_desc, x949,
    conv_desc, algo, ws_data, ws_size,
    x2155, out_desc, x2154));
};
float* x2160 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2161 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2162 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2163 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2164 = (float*)myMalloc(1 * sizeof(float));;
x2164[0] = 0.0f;
float* x2166 = (float*)myMalloc(1 * sizeof(float));;
x2166[0] = 1.0f;

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
    x2166, x2164, in_desc, x2154, out_desc, x2161, sbmv_desc, x682,
    x886, 0.1, x829, x817, 1.0E-5,
    x2162, x2163));
};
float* x2169 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2170 = (float*)myMalloc(1 * sizeof(float));;
x2170[0] = 0.0f;
float* x2172 = (float*)myMalloc(1 * sizeof(float));;
x2172[0] = 1.0f;
float* x2174 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x2172, x_desc, x2161, x2170, x_desc, x2174));
};
float* x2176 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2177 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2178 = (float*)myMalloc(1 * sizeof(float));;
x2178[0] = 0.0f;
float* x2180 = (float*)myMalloc(1 * sizeof(float));;
x2180[0] = 1.0f;

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
    x2180, in_desc, x2174, filt_desc, x337,
    conv_desc, algo, ws_data, ws_size,
    x2178, out_desc, x2177));
};
float* x2183 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2184 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2185 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2186 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2187 = (float*)myMalloc(1 * sizeof(float));;
x2187[0] = 0.0f;
float* x2189 = (float*)myMalloc(1 * sizeof(float));;
x2189[0] = 1.0f;

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
    x2189, x2187, in_desc, x2177, out_desc, x2184, sbmv_desc, x979,
    x871, 0.1, x667, x484, 1.0E-5,
    x2185, x2186));
};
float* x2192 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2193 = (float*)myMalloc(1 * sizeof(float));;
x2193[0] = 0.0f;
float* x2195 = (float*)myMalloc(1 * sizeof(float));;
x2195[0] = 1.0f;
float* x2197 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x2195, x_desc, x2184, x2193, x_desc, x2197));
};
float* x2199 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2200 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2201 = (float*)myMalloc(1 * sizeof(float));;
x2201[0] = 0.0f;
float* x2203 = (float*)myMalloc(1 * sizeof(float));;
x2203[0] = 1.0f;

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
    x2203, in_desc, x2197, filt_desc, x643,
    conv_desc, algo, ws_data, ws_size,
    x2201, out_desc, x2200));
};
float* x2206 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2207 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2208 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2209 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2210 = (float*)myMalloc(1 * sizeof(float));;
x2210[0] = 0.0f;
float* x2212 = (float*)myMalloc(1 * sizeof(float));;
x2212[0] = 1.0f;

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
    x2212, x2210, in_desc, x2200, out_desc, x2207, sbmv_desc, x1084,
    x466, 0.1, x715, x859, 1.0E-5,
    x2208, x2209));
};
float* x2215 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2216 = (float*)myMalloc(1 * sizeof(float));;
x2216[0] = 1.0f;
float* x2218 = (float*)myMalloc(1 * sizeof(float));;
x2218[0] = 1.0f;

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
    cudnnHandle, x2216, bias_desc, x2151, x2218, out_desc, x2207));
};
float* x2221 = (float*)myMalloc(1 * sizeof(float));;
x2221[0] = 0.0f;
float* x2223 = (float*)myMalloc(1 * sizeof(float));;
x2223[0] = 1.0f;
float* x2225 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x2223, x_desc, x2207, x2221, x_desc, x2225));
};
float* x2227 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2228 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2229 = (float*)myMalloc(1 * sizeof(float));;
x2229[0] = 0.0f;
float* x2231 = (float*)myMalloc(1 * sizeof(float));;
x2231[0] = 1.0f;

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
    x2231, in_desc, x2225, filt_desc, x313,
    conv_desc, algo, ws_data, ws_size,
    x2229, out_desc, x2228));
};
float* x2234 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2235 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2236 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2237 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2238 = (float*)myMalloc(1 * sizeof(float));;
x2238[0] = 0.0f;
float* x2240 = (float*)myMalloc(1 * sizeof(float));;
x2240[0] = 1.0f;

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
    x2240, x2238, in_desc, x2228, out_desc, x2235, sbmv_desc, x571,
    x1018, 0.1, x784, x589, 1.0E-5,
    x2236, x2237));
};
float* x2243 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2244 = (float*)myMalloc(1 * sizeof(float));;
x2244[0] = 0.0f;
float* x2246 = (float*)myMalloc(1 * sizeof(float));;
x2246[0] = 1.0f;
float* x2248 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x2246, x_desc, x2235, x2244, x_desc, x2248));
};
float* x2250 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2251 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2252 = (float*)myMalloc(1 * sizeof(float));;
x2252[0] = 0.0f;
float* x2254 = (float*)myMalloc(1 * sizeof(float));;
x2254[0] = 1.0f;

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
    x2254, in_desc, x2248, filt_desc, x1042,
    conv_desc, algo, ws_data, ws_size,
    x2252, out_desc, x2251));
};
float* x2257 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2258 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2259 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2260 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2261 = (float*)myMalloc(1 * sizeof(float));;
x2261[0] = 0.0f;
float* x2263 = (float*)myMalloc(1 * sizeof(float));;
x2263[0] = 1.0f;

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
    x2263, x2261, in_desc, x2251, out_desc, x2258, sbmv_desc, x517,
    x703, 0.1, x853, x985, 1.0E-5,
    x2259, x2260));
};
float* x2266 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2267 = (float*)myMalloc(1 * sizeof(float));;
x2267[0] = 0.0f;
float* x2269 = (float*)myMalloc(1 * sizeof(float));;
x2269[0] = 1.0f;
float* x2271 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x2269, x_desc, x2258, x2267, x_desc, x2271));
};
float* x2273 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2274 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2275 = (float*)myMalloc(1 * sizeof(float));;
x2275[0] = 0.0f;
float* x2277 = (float*)myMalloc(1 * sizeof(float));;
x2277[0] = 1.0f;

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
    x2277, in_desc, x2271, filt_desc, x562,
    conv_desc, algo, ws_data, ws_size,
    x2275, out_desc, x2274));
};
float* x2280 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2281 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2282 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2283 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2284 = (float*)myMalloc(1 * sizeof(float));;
x2284[0] = 0.0f;
float* x2286 = (float*)myMalloc(1 * sizeof(float));;
x2286[0] = 1.0f;

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
    x2286, x2284, in_desc, x2274, out_desc, x2281, sbmv_desc, x1009,
    x733, 0.1, x988, x778, 1.0E-5,
    x2282, x2283));
};
float* x2289 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2290 = (float*)myMalloc(1 * sizeof(float));;
x2290[0] = 1.0f;
float* x2292 = (float*)myMalloc(1 * sizeof(float));;
x2292[0] = 1.0f;

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
    cudnnHandle, x2290, bias_desc, x2225, x2292, out_desc, x2281));
};
float* x2295 = (float*)myMalloc(1 * sizeof(float));;
x2295[0] = 0.0f;
float* x2297 = (float*)myMalloc(1 * sizeof(float));;
x2297[0] = 1.0f;
float* x2299 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x2297, x_desc, x2281, x2295, x_desc, x2299));
};
float* x2301 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2302 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2303 = (float*)myMalloc(1 * sizeof(float));;
x2303[0] = 0.0f;
float* x2305 = (float*)myMalloc(1 * sizeof(float));;
x2305[0] = 1.0f;

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
    x2305, in_desc, x2299, filt_desc, x361,
    conv_desc, algo, ws_data, ws_size,
    x2303, out_desc, x2302));
};
float* x2308 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2309 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2310 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2311 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2312 = (float*)myMalloc(1 * sizeof(float));;
x2312[0] = 0.0f;
float* x2314 = (float*)myMalloc(1 * sizeof(float));;
x2314[0] = 1.0f;

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
    x2314, x2312, in_desc, x2302, out_desc, x2309, sbmv_desc, x526,
    x850, 0.1, x1057, x502, 1.0E-5,
    x2310, x2311));
};
float* x2317 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2318 = (float*)myMalloc(1 * sizeof(float));;
x2318[0] = 0.0f;
float* x2320 = (float*)myMalloc(1 * sizeof(float));;
x2320[0] = 1.0f;
float* x2322 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x2320, x_desc, x2309, x2318, x_desc, x2322));
};
float* x2324 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2325 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2326 = (float*)myMalloc(1 * sizeof(float));;
x2326[0] = 0.0f;
float* x2328 = (float*)myMalloc(1 * sizeof(float));;
x2328[0] = 1.0f;

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
    x2328, in_desc, x2322, filt_desc, x1081,
    conv_desc, algo, ws_data, ws_size,
    x2326, out_desc, x2325));
};
float* x2331 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2332 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2333 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2334 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2335 = (float*)myMalloc(1 * sizeof(float));;
x2335[0] = 0.0f;
float* x2337 = (float*)myMalloc(1 * sizeof(float));;
x2337[0] = 1.0f;

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
    x2337, x2335, in_desc, x2325, out_desc, x2332, sbmv_desc, x799,
    x622, 0.1, x1045, x607, 1.0E-5,
    x2333, x2334));
};
float* x2340 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2341 = (float*)myMalloc(1 * sizeof(float));;
x2341[0] = 0.0f;
float* x2343 = (float*)myMalloc(1 * sizeof(float));;
x2343[0] = 1.0f;
float* x2345 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x2343, x_desc, x2332, x2341, x_desc, x2345));
};
float* x2347 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2348 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2349 = (float*)myMalloc(1 * sizeof(float));;
x2349[0] = 0.0f;
float* x2351 = (float*)myMalloc(1 * sizeof(float));;
x2351[0] = 1.0f;

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
    x2351, in_desc, x2345, filt_desc, x958,
    conv_desc, algo, ws_data, ws_size,
    x2349, out_desc, x2348));
};
float* x2354 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2355 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2356 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2357 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2358 = (float*)myMalloc(1 * sizeof(float));;
x2358[0] = 0.0f;
float* x2360 = (float*)myMalloc(1 * sizeof(float));;
x2360[0] = 1.0f;

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
    x2360, x2358, in_desc, x2348, out_desc, x2355, sbmv_desc, x472,
    x655, 0.1, x922, x1111, 1.0E-5,
    x2356, x2357));
};
float* x2363 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2364 = (float*)myMalloc(1 * sizeof(float));;
x2364[0] = 1.0f;
float* x2366 = (float*)myMalloc(1 * sizeof(float));;
x2366[0] = 1.0f;

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
    cudnnHandle, x2364, bias_desc, x2299, x2366, out_desc, x2355));
};
float* x2369 = (float*)myMalloc(1 * sizeof(float));;
x2369[0] = 0.0f;
float* x2371 = (float*)myMalloc(1 * sizeof(float));;
x2371[0] = 1.0f;
float* x2373 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x2371, x_desc, x2355, x2369, x_desc, x2373));
};
float* x2375 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2376 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2377 = (float*)myMalloc(1 * sizeof(float));;
x2377[0] = 0.0f;
float* x2379 = (float*)myMalloc(1 * sizeof(float));;
x2379[0] = 1.0f;

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
    x2379, in_desc, x2373, filt_desc, x748,
    conv_desc, algo, ws_data, ws_size,
    x2377, out_desc, x2376));
};
float* x2382 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2383 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2384 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2385 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2386 = (float*)myMalloc(1 * sizeof(float));;
x2386[0] = 0.0f;
float* x2388 = (float*)myMalloc(1 * sizeof(float));;
x2388[0] = 1.0f;

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
    x2388, x2386, in_desc, x2376, out_desc, x2383, sbmv_desc, x550,
    x1054, 0.1, x535, x823, 1.0E-5,
    x2384, x2385));
};
float* x2391 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2392 = (float*)myMalloc(1 * sizeof(float));;
x2392[0] = 0.0f;
float* x2394 = (float*)myMalloc(1 * sizeof(float));;
x2394[0] = 1.0f;
float* x2396 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x2394, x_desc, x2383, x2392, x_desc, x2396));
};
float* x2398 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2399 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2400 = (float*)myMalloc(1 * sizeof(float));;
x2400[0] = 0.0f;
float* x2402 = (float*)myMalloc(1 * sizeof(float));;
x2402[0] = 1.0f;

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
    x2402, in_desc, x2396, filt_desc, x973,
    conv_desc, algo, ws_data, ws_size,
    x2400, out_desc, x2399));
};
float* x2405 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2406 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2407 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2408 = (float*)myGpuMalloc(256 * sizeof(float));
float* x2409 = (float*)myMalloc(1 * sizeof(float));;
x2409[0] = 0.0f;
float* x2411 = (float*)myMalloc(1 * sizeof(float));;
x2411[0] = 1.0f;

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
    x2411, x2409, in_desc, x2399, out_desc, x2406, sbmv_desc, x718,
    x862, 0.1, x505, x1015, 1.0E-5,
    x2407, x2408));
};
float* x2414 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2415 = (float*)myMalloc(1 * sizeof(float));;
x2415[0] = 0.0f;
float* x2417 = (float*)myMalloc(1 * sizeof(float));;
x2417[0] = 1.0f;
float* x2419 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x2417, x_desc, x2406, x2415, x_desc, x2419));
};
float* x2421 = (float*)myGpuMalloc(262144 * sizeof(float));
float* x2422 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2423 = (float*)myMalloc(1 * sizeof(float));;
x2423[0] = 0.0f;
float* x2425 = (float*)myMalloc(1 * sizeof(float));;
x2425[0] = 1.0f;

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
    x2425, in_desc, x2419, filt_desc, x586,
    conv_desc, algo, ws_data, ws_size,
    x2423, out_desc, x2422));
};
float* x2428 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2429 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2430 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2431 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x2432 = (float*)myMalloc(1 * sizeof(float));;
x2432[0] = 0.0f;
float* x2434 = (float*)myMalloc(1 * sizeof(float));;
x2434[0] = 1.0f;

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
    x2434, x2432, in_desc, x2422, out_desc, x2429, sbmv_desc, x1039,
    x574, 0.1, x661, x844, 1.0E-5,
    x2430, x2431));
};
float* x2437 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2438 = (float*)myMalloc(1 * sizeof(float));;
x2438[0] = 1.0f;
float* x2440 = (float*)myMalloc(1 * sizeof(float));;
x2440[0] = 1.0f;

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
    cudnnHandle, x2438, bias_desc, x2373, x2440, out_desc, x2429));
};
float* x2443 = (float*)myMalloc(1 * sizeof(float));;
x2443[0] = 0.0f;
float* x2445 = (float*)myMalloc(1 * sizeof(float));;
x2445[0] = 1.0f;
float* x2447 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x2445, x_desc, x2429, x2443, x_desc, x2447));
};
float* x2449 = (float*)myGpuMalloc(1048576 * sizeof(float));
float* x2450 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2451 = (float*)myMalloc(1 * sizeof(float));;
x2451[0] = 0.0f;
float* x2453 = (float*)myMalloc(1 * sizeof(float));;
x2453[0] = 1.0f;

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
    x2453, in_desc, x2447, filt_desc, x712,
    conv_desc, algo, ws_data, ws_size,
    x2451, out_desc, x2450));
};
float* x2456 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2457 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2458 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2459 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2460 = (float*)myMalloc(1 * sizeof(float));;
x2460[0] = 0.0f;
float* x2462 = (float*)myMalloc(1 * sizeof(float));;
x2462[0] = 1.0f;

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
    x2462, x2460, in_desc, x2450, out_desc, x2457, sbmv_desc, x898,
    x967, 0.1, x496, x658, 1.0E-5,
    x2458, x2459));
};
float* x2465 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2466 = (float*)myMalloc(1 * sizeof(float));;
x2466[0] = 0.0f;
float* x2468 = (float*)myMalloc(1 * sizeof(float));;
x2468[0] = 1.0f;
float* x2470 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x2468, x_desc, x2457, x2466, x_desc, x2470));
};
float* x2472 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2473 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2474 = (float*)myMalloc(1 * sizeof(float));;
x2474[0] = 0.0f;
float* x2476 = (float*)myMalloc(1 * sizeof(float));;
x2476[0] = 1.0f;

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
    x2476, in_desc, x2470, filt_desc, x397,
    conv_desc, algo, ws_data, ws_size,
    x2474, out_desc, x2473));
};
float* x2479 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2480 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2481 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2482 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2483 = (float*)myMalloc(1 * sizeof(float));;
x2483[0] = 0.0f;
float* x2485 = (float*)myMalloc(1 * sizeof(float));;
x2485[0] = 1.0f;

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
    x2485, x2483, in_desc, x2473, out_desc, x2480, sbmv_desc, x910,
    x772, 0.1, x634, x445, 1.0E-5,
    x2481, x2482));
};
float* x2488 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2489 = (float*)myMalloc(1 * sizeof(float));;
x2489[0] = 0.0f;
float* x2491 = (float*)myMalloc(1 * sizeof(float));;
x2491[0] = 1.0f;
float* x2493 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2491, x_desc, x2480, x2489, x_desc, x2493));
};
float* x2495 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2496 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2497 = (float*)myMalloc(1 * sizeof(float));;
x2497[0] = 0.0f;
float* x2499 = (float*)myMalloc(1 * sizeof(float));;
x2499[0] = 1.0f;

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
    x2499, in_desc, x2493, filt_desc, x931,
    conv_desc, algo, ws_data, ws_size,
    x2497, out_desc, x2496));
};
float* x2502 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2503 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2504 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x2505 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x2506 = (float*)myMalloc(1 * sizeof(float));;
x2506[0] = 0.0f;
float* x2508 = (float*)myMalloc(1 * sizeof(float));;
x2508[0] = 1.0f;

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
    x2508, x2506, in_desc, x2496, out_desc, x2503, sbmv_desc, x1012,
    x481, 0.1, x640, x874, 1.0E-5,
    x2504, x2505));
};
float* x2511 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2512 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2513 = (float*)myMalloc(1 * sizeof(float));;
x2513[0] = 0.0f;
float* x2515 = (float*)myMalloc(1 * sizeof(float));;
x2515[0] = 1.0f;

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
    x2515, in_desc, x2447, filt_desc, x937,
    conv_desc, algo, ws_data, ws_size,
    x2513, out_desc, x2512));
};
float* x2518 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2519 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2520 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x2521 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x2522 = (float*)myMalloc(1 * sizeof(float));;
x2522[0] = 0.0f;
float* x2524 = (float*)myMalloc(1 * sizeof(float));;
x2524[0] = 1.0f;

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
    x2524, x2522, in_desc, x2512, out_desc, x2519, sbmv_desc, x814,
    x616, 0.1, x487, x670, 1.0E-5,
    x2520, x2521));
};
float* x2527 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2528 = (float*)myMalloc(1 * sizeof(float));;
x2528[0] = 1.0f;
float* x2530 = (float*)myMalloc(1 * sizeof(float));;
x2530[0] = 1.0f;

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
    cudnnHandle, x2528, bias_desc, x2519, x2530, out_desc, x2503));
};
float* x2533 = (float*)myMalloc(1 * sizeof(float));;
x2533[0] = 0.0f;
float* x2535 = (float*)myMalloc(1 * sizeof(float));;
x2535[0] = 1.0f;
float* x2537 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x2535, x_desc, x2503, x2533, x_desc, x2537));
};
float* x2539 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2540 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2541 = (float*)myMalloc(1 * sizeof(float));;
x2541[0] = 0.0f;
float* x2543 = (float*)myMalloc(1 * sizeof(float));;
x2543[0] = 1.0f;

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
    x2543, in_desc, x2537, filt_desc, x940,
    conv_desc, algo, ws_data, ws_size,
    x2541, out_desc, x2540));
};
float* x2546 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2547 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2548 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2549 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2550 = (float*)myMalloc(1 * sizeof(float));;
x2550[0] = 0.0f;
float* x2552 = (float*)myMalloc(1 * sizeof(float));;
x2552[0] = 1.0f;

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
    x2552, x2550, in_desc, x2540, out_desc, x2547, sbmv_desc, x433,
    x706, 0.1, x757, x490, 1.0E-5,
    x2548, x2549));
};
float* x2555 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2556 = (float*)myMalloc(1 * sizeof(float));;
x2556[0] = 0.0f;
float* x2558 = (float*)myMalloc(1 * sizeof(float));;
x2558[0] = 1.0f;
float* x2560 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2558, x_desc, x2547, x2556, x_desc, x2560));
};
float* x2562 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2563 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2564 = (float*)myMalloc(1 * sizeof(float));;
x2564[0] = 0.0f;
float* x2566 = (float*)myMalloc(1 * sizeof(float));;
x2566[0] = 1.0f;

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
    x2566, in_desc, x2560, filt_desc, x760,
    conv_desc, algo, ws_data, ws_size,
    x2564, out_desc, x2563));
};
float* x2569 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2570 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2571 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2572 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2573 = (float*)myMalloc(1 * sizeof(float));;
x2573[0] = 0.0f;
float* x2575 = (float*)myMalloc(1 * sizeof(float));;
x2575[0] = 1.0f;

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
    x2575, x2573, in_desc, x2563, out_desc, x2570, sbmv_desc, x775,
    x493, 0.1, x709, x880, 1.0E-5,
    x2571, x2572));
};
float* x2578 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2579 = (float*)myMalloc(1 * sizeof(float));;
x2579[0] = 0.0f;
float* x2581 = (float*)myMalloc(1 * sizeof(float));;
x2581[0] = 1.0f;
float* x2583 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2581, x_desc, x2570, x2579, x_desc, x2583));
};
float* x2585 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2586 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2587 = (float*)myMalloc(1 * sizeof(float));;
x2587[0] = 0.0f;
float* x2589 = (float*)myMalloc(1 * sizeof(float));;
x2589[0] = 1.0f;

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
    x2589, in_desc, x2583, filt_desc, x436,
    conv_desc, algo, ws_data, ws_size,
    x2587, out_desc, x2586));
};
float* x2592 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2593 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2594 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x2595 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x2596 = (float*)myMalloc(1 * sizeof(float));;
x2596[0] = 0.0f;
float* x2598 = (float*)myMalloc(1 * sizeof(float));;
x2598[0] = 1.0f;

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
    x2598, x2596, in_desc, x2586, out_desc, x2593, sbmv_desc, x577,
    x727, 0.1, x499, x1030, 1.0E-5,
    x2594, x2595));
};
float* x2601 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2602 = (float*)myMalloc(1 * sizeof(float));;
x2602[0] = 1.0f;
float* x2604 = (float*)myMalloc(1 * sizeof(float));;
x2604[0] = 1.0f;

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
    cudnnHandle, x2602, bias_desc, x2537, x2604, out_desc, x2593));
};
float* x2607 = (float*)myMalloc(1 * sizeof(float));;
x2607[0] = 0.0f;
float* x2609 = (float*)myMalloc(1 * sizeof(float));;
x2609[0] = 1.0f;
float* x2611 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x2609, x_desc, x2593, x2607, x_desc, x2611));
};
float* x2613 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2614 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2615 = (float*)myMalloc(1 * sizeof(float));;
x2615[0] = 0.0f;
float* x2617 = (float*)myMalloc(1 * sizeof(float));;
x2617[0] = 1.0f;

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
    x2617, in_desc, x2611, filt_desc, x1090,
    conv_desc, algo, ws_data, ws_size,
    x2615, out_desc, x2614));
};
float* x2620 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2621 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2622 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2623 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2624 = (float*)myMalloc(1 * sizeof(float));;
x2624[0] = 0.0f;
float* x2626 = (float*)myMalloc(1 * sizeof(float));;
x2626[0] = 1.0f;

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
    x2626, x2624, in_desc, x2614, out_desc, x2621, sbmv_desc, x340,
    x529, 0.1, x934, x1060, 1.0E-5,
    x2622, x2623));
};
float* x2629 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2630 = (float*)myMalloc(1 * sizeof(float));;
x2630[0] = 0.0f;
float* x2632 = (float*)myMalloc(1 * sizeof(float));;
x2632[0] = 1.0f;
float* x2634 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2632, x_desc, x2621, x2630, x_desc, x2634));
};
float* x2636 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2637 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2638 = (float*)myMalloc(1 * sizeof(float));;
x2638[0] = 0.0f;
float* x2640 = (float*)myMalloc(1 * sizeof(float));;
x2640[0] = 1.0f;

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
    x2640, in_desc, x2634, filt_desc, x379,
    conv_desc, algo, ws_data, ws_size,
    x2638, out_desc, x2637));
};
float* x2643 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2644 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2645 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2646 = (float*)myGpuMalloc(512 * sizeof(float));
float* x2647 = (float*)myMalloc(1 * sizeof(float));;
x2647[0] = 0.0f;
float* x2649 = (float*)myMalloc(1 * sizeof(float));;
x2649[0] = 1.0f;

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
    x2649, x2647, in_desc, x2637, out_desc, x2644, sbmv_desc, x877,
    x802, 0.1, x331, x901, 1.0E-5,
    x2645, x2646));
};
float* x2652 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2653 = (float*)myMalloc(1 * sizeof(float));;
x2653[0] = 0.0f;
float* x2655 = (float*)myMalloc(1 * sizeof(float));;
x2655[0] = 1.0f;
float* x2657 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2655, x_desc, x2644, x2653, x_desc, x2657));
};
float* x2659 = (float*)myGpuMalloc(131072 * sizeof(float));
float* x2660 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2661 = (float*)myMalloc(1 * sizeof(float));;
x2661[0] = 0.0f;
float* x2663 = (float*)myMalloc(1 * sizeof(float));;
x2663[0] = 1.0f;

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
    x2663, in_desc, x2657, filt_desc, x394,
    conv_desc, algo, ws_data, ws_size,
    x2661, out_desc, x2660));
};
float* x2666 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2667 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2668 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x2669 = (float*)myGpuMalloc(2048 * sizeof(float));
float* x2670 = (float*)myMalloc(1 * sizeof(float));;
x2670[0] = 0.0f;
float* x2672 = (float*)myMalloc(1 * sizeof(float));;
x2672[0] = 1.0f;

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
    x2672, x2670, in_desc, x2660, out_desc, x2667, sbmv_desc, x604,
    x838, 0.1, x1075, x664, 1.0E-5,
    x2668, x2669));
};
float* x2675 = (float*)myGpuMalloc(524288 * sizeof(float));
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
    cudnnHandle, x2676, bias_desc, x2611, x2678, out_desc, x2667));
};
float* x2681 = (float*)myMalloc(1 * sizeof(float));;
x2681[0] = 0.0f;
float* x2683 = (float*)myMalloc(1 * sizeof(float));;
x2683[0] = 1.0f;
float* x2685 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x2683, x_desc, x2667, x2681, x_desc, x2685));
};
float* x2687 = (float*)myGpuMalloc(524288 * sizeof(float));
float* x2688 = (float*)myMalloc(1 * sizeof(float));;
x2688[0] = 0.0f;
float* x2690 = (float*)myMalloc(1 * sizeof(float));;
x2690[0] = 1.0f;
float* x2692 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2690, in_desc, x2685, x2688, out_desc, x2692));
};
float* x2694 = (float*)myGpuMalloc(131072 * sizeof(float));
// resize to WrappedArray(64, -1)
// resize to WrappedArray(64, -1)
// gemm: ArrayBuffer(64, 2048), Vector(10, 2048)
float* x2698 = (float*)myGpuMalloc(640 * sizeof(float));
float* x2699 = (float*)myMalloc(1 * sizeof(float));;
x2699[0] = 0.0f;
float* x2701 = (float*)myMalloc(1 * sizeof(float));;
x2701[0] = 1.0f;
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 10,64,2048,x2701,x976,2048,x2692,2048,x2699,x2698,10));
float* x2704 = (float*)myGpuMalloc(640 * sizeof(float));
float* x2705 = (float*)myMalloc(1 * sizeof(float));;
x2705[0] = 1.0f;
float* x2707 = (float*)myMalloc(1 * sizeof(float));;
x2707[0] = 1.0f;

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
    cudnnHandle, x2705, bias_desc, x439, x2707, out_desc, x2698));
};
float* x2710 = (float*)myMalloc(1 * sizeof(float));;
x2710[0] = 0.0f;
float* x2712 = (float*)myMalloc(1 * sizeof(float));;
x2712[0] = 1.0f;
float* x2714 = (float*)myGpuMalloc(640 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 10, 1, 1));
CUDNN_CALL(cudnnSoftmaxForward(
    cudnnHandle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL,
    x2712, x_desc, x2698, x2710, x_desc, x2714));
};
float* x2716 = (float*)myGpuMalloc(640 * sizeof(float));
float* x2717 = (float*)myGpuMalloc(64 * sizeof(float));
nllLoss<<<64, 1>>>(x2714,10,x2717,x1399);
float* x2719 = (float*)myGpuMalloc(64 * sizeof(float));
// resize to ArrayBuffer(64, 1, 1, 1)
float* x2721 = (float*)myGpuMalloc(1 * sizeof(float));
float* x2722 = (float*)myMalloc(1 * sizeof(float));;
x2722[0] = 0.0f;
float* x2724 = (float*)myMalloc(1 * sizeof(float));;
x2724[0] = 1.0f;

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
    x2724, x_desc, x2717, x2722, out_desc, x2721));
};
// resize to WrappedArray(1)
float* x2728 = (float*)myGpuMalloc(1 * sizeof(float));
arrayFill<<<1, 1>>>(x2728, 1.0f);
// backend is lantern.TensorDslCudnn$BackendCudnn@52bc2ed2
CUDA_CALL(cudaMemcpy(x1409, x2721, 1 * sizeof(float), cudaMemcpyDeviceToHost));
// 'mean' gradient
// backprop for mean op
float* x2734 = (float*)myMalloc(1 * sizeof(float));;
x2734[0] = 0.015625f;
float* x2736 = (float*)myMalloc(1 * sizeof(float));;
x2736[0] = 1.0f;

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
    cudnnHandle, x2734, bias_desc, x2728, x2736, out_desc, x2719));
};
// 'nllLossB' gradient.
nllLoss_grad<<<64, 1>>>(10,x2719,x1399,x2716);
float* x2741 = (float*)myMalloc(1 * sizeof(float));;
x2741[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 10, 1, 1));
CUDNN_CALL(cudnnSoftmaxBackward(
    cudnnHandle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL,
    x2741, x_desc, x2714, x_desc, x2716,
    x2741, x_desc, x2704));
};
float* x2744 = (float*)myMalloc(1 * sizeof(float));;
x2744[0] = 1.0f;

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
    cudnnHandle, x2744, grad_out_desc, x2704,
    x2744, grad_bias_desc, x1155));
};
// backprop for gemm ArrayBuffer(64, 2048), Vector(10, 2048)
float* x2748 = (float*)myMalloc(1 * sizeof(float));;
x2748[0] = 1.0f;
float* x2750 = (float*)myMalloc(1 * sizeof(float));;
x2750[0] = 1.0f;
// backprop of gemm
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2048,64,10,x2748,x976,2048,x2704,10,x2750,x2694,2048));
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 2048,10,64,x2748,x2692,2048,x2704,10,x2750,x1334,2048));
float* x2755 = (float*)myMalloc(1 * sizeof(float));;
x2755[0] = 0.0f;
float* x2757 = (float*)myMalloc(1 * sizeof(float));;
x2757[0] = 1.0f;

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
    x2757, out_desc, x2692, out_desc, x2694, in_desc, x2685  , x2755, in_desc, x2687));
};
float* x2760 = (float*)myMalloc(1 * sizeof(float));;
x2760[0] = 1.0f;
float* x2762 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x2760, x_desc, x2685, x_desc, x2687, x_desc, x2667,
    x2760, x_desc, x2675));
};
float* x2764 = (float*)myMalloc(1 * sizeof(float));;
x2764[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x2764, grad_out_desc, x2675,
    x2764, grad_bias_desc, x2613));
};
float* x2767 = (float*)myMalloc(1 * sizeof(float));;
x2767[0] = 0.0f;
float* x2769 = (float*)myMalloc(1 * sizeof(float));;
x2769[0] = 1.0f;

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
    x2769, x2769, x2769, x2769, in_desc, x2660,
    out_desc, x2675, in_desc, x2666, sbmv_desc, x604,
    x1210,x1288, 1.0E-5, x2668, x2669));
};
// conv2D back-propagate
float* x2773 = (float*)myMalloc(1 * sizeof(float));;
x2773[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x2773, filt_desc, x394, grad_out_desc, x2666,
    conv_desc, algo, ws_data, ws_size,
    x2773, grad_in_desc, x2659));
};
float* x2776 = (float*)myMalloc(1 * sizeof(float));;
x2776[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x2776, in_desc, x2657, grad_out_desc, x2666,
    conv_desc, algo, ws_data, ws_size,
    x2776, grad_filt_desc, x1140));
};
float* x2779 = (float*)myMalloc(1 * sizeof(float));;
x2779[0] = 1.0f;
float* x2781 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2779, x_desc, x2657, x_desc, x2659, x_desc, x2644,
    x2779, x_desc, x2652));
};
float* x2783 = (float*)myMalloc(1 * sizeof(float));;
x2783[0] = 0.0f;
float* x2785 = (float*)myMalloc(1 * sizeof(float));;
x2785[0] = 1.0f;

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
    x2785, x2785, x2785, x2785, in_desc, x2637,
    out_desc, x2652, in_desc, x2643, sbmv_desc, x877,
    x1301,x1276, 1.0E-5, x2645, x2646));
};
// conv2D back-propagate
float* x2789 = (float*)myMalloc(1 * sizeof(float));;
x2789[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x2789, filt_desc, x379, grad_out_desc, x2643,
    conv_desc, algo, ws_data, ws_size,
    x2789, grad_in_desc, x2636));
};
float* x2792 = (float*)myMalloc(1 * sizeof(float));;
x2792[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x2792, in_desc, x2634, grad_out_desc, x2643,
    conv_desc, algo, ws_data, ws_size,
    x2792, grad_filt_desc, x1135));
};
float* x2795 = (float*)myMalloc(1 * sizeof(float));;
x2795[0] = 1.0f;
float* x2797 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2795, x_desc, x2634, x_desc, x2636, x_desc, x2621,
    x2795, x_desc, x2629));
};
float* x2799 = (float*)myMalloc(1 * sizeof(float));;
x2799[0] = 0.0f;
float* x2801 = (float*)myMalloc(1 * sizeof(float));;
x2801[0] = 1.0f;

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
    x2801, x2801, x2801, x2801, in_desc, x2614,
    out_desc, x2629, in_desc, x2620, sbmv_desc, x340,
    x1122,x1185, 1.0E-5, x2622, x2623));
};
// conv2D back-propagate
float* x2805 = (float*)myMalloc(1 * sizeof(float));;
x2805[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x2805, filt_desc, x1090, grad_out_desc, x2620,
    conv_desc, algo, ws_data, ws_size,
    x2805, grad_in_desc, x2613));
};
float* x2808 = (float*)myMalloc(1 * sizeof(float));;
x2808[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x2808, in_desc, x2611, grad_out_desc, x2620,
    conv_desc, algo, ws_data, ws_size,
    x2808, grad_filt_desc, x1372));
};
float* x2811 = (float*)myMalloc(1 * sizeof(float));;
x2811[0] = 1.0f;
float* x2813 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x2811, x_desc, x2611, x_desc, x2613, x_desc, x2593,
    x2811, x_desc, x2601));
};
float* x2815 = (float*)myMalloc(1 * sizeof(float));;
x2815[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x2815, grad_out_desc, x2601,
    x2815, grad_bias_desc, x2539));
};
float* x2818 = (float*)myMalloc(1 * sizeof(float));;
x2818[0] = 0.0f;
float* x2820 = (float*)myMalloc(1 * sizeof(float));;
x2820[0] = 1.0f;

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
    x2820, x2820, x2820, x2820, in_desc, x2586,
    out_desc, x2601, in_desc, x2592, sbmv_desc, x577,
    x1201,x1251, 1.0E-5, x2594, x2595));
};
// conv2D back-propagate
float* x2824 = (float*)myMalloc(1 * sizeof(float));;
x2824[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x2824, filt_desc, x436, grad_out_desc, x2592,
    conv_desc, algo, ws_data, ws_size,
    x2824, grad_in_desc, x2585));
};
float* x2827 = (float*)myMalloc(1 * sizeof(float));;
x2827[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x2827, in_desc, x2583, grad_out_desc, x2592,
    conv_desc, algo, ws_data, ws_size,
    x2827, grad_filt_desc, x1154));
};
float* x2830 = (float*)myMalloc(1 * sizeof(float));;
x2830[0] = 1.0f;
float* x2832 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2830, x_desc, x2583, x_desc, x2585, x_desc, x2570,
    x2830, x_desc, x2578));
};
float* x2834 = (float*)myMalloc(1 * sizeof(float));;
x2834[0] = 0.0f;
float* x2836 = (float*)myMalloc(1 * sizeof(float));;
x2836[0] = 1.0f;

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
    x2836, x2836, x2836, x2836, in_desc, x2563,
    out_desc, x2578, in_desc, x2569, sbmv_desc, x775,
    x1267,x1173, 1.0E-5, x2571, x2572));
};
// conv2D back-propagate
float* x2840 = (float*)myMalloc(1 * sizeof(float));;
x2840[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x2840, filt_desc, x760, grad_out_desc, x2569,
    conv_desc, algo, ws_data, ws_size,
    x2840, grad_in_desc, x2562));
};
float* x2843 = (float*)myMalloc(1 * sizeof(float));;
x2843[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x2843, in_desc, x2560, grad_out_desc, x2569,
    conv_desc, algo, ws_data, ws_size,
    x2843, grad_filt_desc, x1262));
};
float* x2846 = (float*)myMalloc(1 * sizeof(float));;
x2846[0] = 1.0f;
float* x2848 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2846, x_desc, x2560, x_desc, x2562, x_desc, x2547,
    x2846, x_desc, x2555));
};
float* x2850 = (float*)myMalloc(1 * sizeof(float));;
x2850[0] = 0.0f;
float* x2852 = (float*)myMalloc(1 * sizeof(float));;
x2852[0] = 1.0f;

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
    x2852, x2852, x2852, x2852, in_desc, x2540,
    out_desc, x2555, in_desc, x2546, sbmv_desc, x433,
    x1153,x1244, 1.0E-5, x2548, x2549));
};
// conv2D back-propagate
float* x2856 = (float*)myMalloc(1 * sizeof(float));;
x2856[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x2856, filt_desc, x940, grad_out_desc, x2546,
    conv_desc, algo, ws_data, ws_size,
    x2856, grad_in_desc, x2539));
};
float* x2859 = (float*)myMalloc(1 * sizeof(float));;
x2859[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x2859, in_desc, x2537, grad_out_desc, x2546,
    conv_desc, algo, ws_data, ws_size,
    x2859, grad_filt_desc, x1322));
};
float* x2862 = (float*)myMalloc(1 * sizeof(float));;
x2862[0] = 1.0f;
float* x2864 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x2862, x_desc, x2537, x_desc, x2539, x_desc, x2503,
    x2862, x_desc, x2511));
};
float* x2866 = (float*)myMalloc(1 * sizeof(float));;
x2866[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 2048, 2, 2));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x2866, grad_out_desc, x2511,
    x2866, grad_bias_desc, x2527));
};
float* x2869 = (float*)myMalloc(1 * sizeof(float));;
x2869[0] = 0.0f;
float* x2871 = (float*)myMalloc(1 * sizeof(float));;
x2871[0] = 1.0f;

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
    x2871, x2871, x2871, x2871, in_desc, x2512,
    out_desc, x2527, in_desc, x2518, sbmv_desc, x814,
    x1280,x1214, 1.0E-5, x2520, x2521));
};
// conv2D back-propagate
float* x2875 = (float*)myMalloc(1 * sizeof(float));;
x2875[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x2875, filt_desc, x937, grad_out_desc, x2518,
    conv_desc, algo, ws_data, ws_size,
    x2875, grad_in_desc, x2449));
};
float* x2878 = (float*)myMalloc(1 * sizeof(float));;
x2878[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x2878, in_desc, x2447, grad_out_desc, x2518,
    conv_desc, algo, ws_data, ws_size,
    x2878, grad_filt_desc, x1321));
};
float* x2881 = (float*)myMalloc(1 * sizeof(float));;
x2881[0] = 0.0f;
float* x2883 = (float*)myMalloc(1 * sizeof(float));;
x2883[0] = 1.0f;

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
    x2883, x2883, x2883, x2883, in_desc, x2496,
    out_desc, x2511, in_desc, x2502, sbmv_desc, x1012,
    x1346,x1169, 1.0E-5, x2504, x2505));
};
// conv2D back-propagate
float* x2887 = (float*)myMalloc(1 * sizeof(float));;
x2887[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x2887, filt_desc, x931, grad_out_desc, x2502,
    conv_desc, algo, ws_data, ws_size,
    x2887, grad_in_desc, x2495));
};
float* x2890 = (float*)myMalloc(1 * sizeof(float));;
x2890[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x2890, in_desc, x2493, grad_out_desc, x2502,
    conv_desc, algo, ws_data, ws_size,
    x2890, grad_filt_desc, x1319));
};
float* x2893 = (float*)myMalloc(1 * sizeof(float));;
x2893[0] = 1.0f;
float* x2895 = (float*)myGpuMalloc(131072 * sizeof(float));

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
    x2893, x_desc, x2493, x_desc, x2495, x_desc, x2480,
    x2893, x_desc, x2488));
};
float* x2897 = (float*)myMalloc(1 * sizeof(float));;
x2897[0] = 0.0f;
float* x2899 = (float*)myMalloc(1 * sizeof(float));;
x2899[0] = 1.0f;

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
    x2899, x2899, x2899, x2899, in_desc, x2473,
    out_desc, x2488, in_desc, x2479, sbmv_desc, x910,
    x1312,x1266, 1.0E-5, x2481, x2482));
};
// conv2D back-propagate
float* x2903 = (float*)myMalloc(1 * sizeof(float));;
x2903[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x2903, filt_desc, x397, grad_out_desc, x2479,
    conv_desc, algo, ws_data, ws_size,
    x2903, grad_in_desc, x2472));
};
float* x2906 = (float*)myMalloc(1 * sizeof(float));;
x2906[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x2906, in_desc, x2470, grad_out_desc, x2479,
    conv_desc, algo, ws_data, ws_size,
    x2906, grad_filt_desc, x1141));
};
float* x2909 = (float*)myMalloc(1 * sizeof(float));;
x2909[0] = 1.0f;
float* x2911 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x2909, x_desc, x2470, x_desc, x2472, x_desc, x2457,
    x2909, x_desc, x2465));
};
float* x2913 = (float*)myMalloc(1 * sizeof(float));;
x2913[0] = 0.0f;
float* x2915 = (float*)myMalloc(1 * sizeof(float));;
x2915[0] = 1.0f;

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
    x2915, x2915, x2915, x2915, in_desc, x2450,
    out_desc, x2465, in_desc, x2456, sbmv_desc, x898,
    x1308,x1331, 1.0E-5, x2458, x2459));
};
// conv2D back-propagate
float* x2919 = (float*)myMalloc(1 * sizeof(float));;
x2919[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x2919, filt_desc, x712, grad_out_desc, x2456,
    conv_desc, algo, ws_data, ws_size,
    x2919, grad_in_desc, x2449));
};
float* x2922 = (float*)myMalloc(1 * sizeof(float));;
x2922[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x2922, in_desc, x2447, grad_out_desc, x2456,
    conv_desc, algo, ws_data, ws_size,
    x2922, grad_filt_desc, x1246));
};
float* x2925 = (float*)myMalloc(1 * sizeof(float));;
x2925[0] = 1.0f;
float* x2927 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x2925, x_desc, x2447, x_desc, x2449, x_desc, x2429,
    x2925, x_desc, x2437));
};
float* x2929 = (float*)myMalloc(1 * sizeof(float));;
x2929[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x2929, grad_out_desc, x2437,
    x2929, grad_bias_desc, x2375));
};
float* x2932 = (float*)myMalloc(1 * sizeof(float));;
x2932[0] = 0.0f;
float* x2934 = (float*)myMalloc(1 * sizeof(float));;
x2934[0] = 1.0f;

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
    x2934, x2934, x2934, x2934, in_desc, x2422,
    out_desc, x2437, in_desc, x2428, sbmv_desc, x1039,
    x1355,x1200, 1.0E-5, x2430, x2431));
};
// conv2D back-propagate
float* x2938 = (float*)myMalloc(1 * sizeof(float));;
x2938[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x2938, filt_desc, x586, grad_out_desc, x2428,
    conv_desc, algo, ws_data, ws_size,
    x2938, grad_in_desc, x2421));
};
float* x2941 = (float*)myMalloc(1 * sizeof(float));;
x2941[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x2941, in_desc, x2419, grad_out_desc, x2428,
    conv_desc, algo, ws_data, ws_size,
    x2941, grad_filt_desc, x1204));
};
float* x2944 = (float*)myMalloc(1 * sizeof(float));;
x2944[0] = 1.0f;
float* x2946 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x2944, x_desc, x2419, x_desc, x2421, x_desc, x2406,
    x2944, x_desc, x2414));
};
float* x2948 = (float*)myMalloc(1 * sizeof(float));;
x2948[0] = 0.0f;
float* x2950 = (float*)myMalloc(1 * sizeof(float));;
x2950[0] = 1.0f;

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
    x2950, x2950, x2950, x2950, in_desc, x2399,
    out_desc, x2414, in_desc, x2405, sbmv_desc, x718,
    x1248,x1296, 1.0E-5, x2407, x2408));
};
// conv2D back-propagate
float* x2954 = (float*)myMalloc(1 * sizeof(float));;
x2954[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x2954, filt_desc, x973, grad_out_desc, x2405,
    conv_desc, algo, ws_data, ws_size,
    x2954, grad_in_desc, x2398));
};
float* x2957 = (float*)myMalloc(1 * sizeof(float));;
x2957[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x2957, in_desc, x2396, grad_out_desc, x2405,
    conv_desc, algo, ws_data, ws_size,
    x2957, grad_filt_desc, x1333));
};
float* x2960 = (float*)myMalloc(1 * sizeof(float));;
x2960[0] = 1.0f;
float* x2962 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x2960, x_desc, x2396, x_desc, x2398, x_desc, x2383,
    x2960, x_desc, x2391));
};
float* x2964 = (float*)myMalloc(1 * sizeof(float));;
x2964[0] = 0.0f;
float* x2966 = (float*)myMalloc(1 * sizeof(float));;
x2966[0] = 1.0f;

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
    x2966, x2966, x2966, x2966, in_desc, x2376,
    out_desc, x2391, in_desc, x2382, sbmv_desc, x550,
    x1192,x1360, 1.0E-5, x2384, x2385));
};
// conv2D back-propagate
float* x2970 = (float*)myMalloc(1 * sizeof(float));;
x2970[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x2970, filt_desc, x748, grad_out_desc, x2382,
    conv_desc, algo, ws_data, ws_size,
    x2970, grad_in_desc, x2375));
};
float* x2973 = (float*)myMalloc(1 * sizeof(float));;
x2973[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x2973, in_desc, x2373, grad_out_desc, x2382,
    conv_desc, algo, ws_data, ws_size,
    x2973, grad_filt_desc, x1258));
};
float* x2976 = (float*)myMalloc(1 * sizeof(float));;
x2976[0] = 1.0f;
float* x2978 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x2976, x_desc, x2373, x_desc, x2375, x_desc, x2355,
    x2976, x_desc, x2363));
};
float* x2980 = (float*)myMalloc(1 * sizeof(float));;
x2980[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x2980, grad_out_desc, x2363,
    x2980, grad_bias_desc, x2301));
};
float* x2983 = (float*)myMalloc(1 * sizeof(float));;
x2983[0] = 0.0f;
float* x2985 = (float*)myMalloc(1 * sizeof(float));;
x2985[0] = 1.0f;

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
    x2985, x2985, x2985, x2985, in_desc, x2348,
    out_desc, x2363, in_desc, x2354, sbmv_desc, x472,
    x1166,x1227, 1.0E-5, x2356, x2357));
};
// conv2D back-propagate
float* x2989 = (float*)myMalloc(1 * sizeof(float));;
x2989[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x2989, filt_desc, x958, grad_out_desc, x2354,
    conv_desc, algo, ws_data, ws_size,
    x2989, grad_in_desc, x2347));
};
float* x2992 = (float*)myMalloc(1 * sizeof(float));;
x2992[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x2992, in_desc, x2345, grad_out_desc, x2354,
    conv_desc, algo, ws_data, ws_size,
    x2992, grad_filt_desc, x1328));
};
float* x2995 = (float*)myMalloc(1 * sizeof(float));;
x2995[0] = 1.0f;
float* x2997 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x2995, x_desc, x2345, x_desc, x2347, x_desc, x2332,
    x2995, x_desc, x2340));
};
float* x2999 = (float*)myMalloc(1 * sizeof(float));;
x2999[0] = 0.0f;
float* x3001 = (float*)myMalloc(1 * sizeof(float));;
x3001[0] = 1.0f;

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
    x3001, x3001, x3001, x3001, in_desc, x2325,
    out_desc, x2340, in_desc, x2331, sbmv_desc, x799,
    x1275,x1216, 1.0E-5, x2333, x2334));
};
// conv2D back-propagate
float* x3005 = (float*)myMalloc(1 * sizeof(float));;
x3005[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3005, filt_desc, x1081, grad_out_desc, x2331,
    conv_desc, algo, ws_data, ws_size,
    x3005, grad_in_desc, x2324));
};
float* x3008 = (float*)myMalloc(1 * sizeof(float));;
x3008[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3008, in_desc, x2322, grad_out_desc, x2331,
    conv_desc, algo, ws_data, ws_size,
    x3008, grad_filt_desc, x1369));
};
float* x3011 = (float*)myMalloc(1 * sizeof(float));;
x3011[0] = 1.0f;
float* x3013 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x3011, x_desc, x2322, x_desc, x2324, x_desc, x2309,
    x3011, x_desc, x2317));
};
float* x3015 = (float*)myMalloc(1 * sizeof(float));;
x3015[0] = 0.0f;
float* x3017 = (float*)myMalloc(1 * sizeof(float));;
x3017[0] = 1.0f;

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
    x3017, x3017, x3017, x3017, in_desc, x2302,
    out_desc, x2317, in_desc, x2308, sbmv_desc, x526,
    x1184,x1292, 1.0E-5, x2310, x2311));
};
// conv2D back-propagate
float* x3021 = (float*)myMalloc(1 * sizeof(float));;
x3021[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3021, filt_desc, x361, grad_out_desc, x2308,
    conv_desc, algo, ws_data, ws_size,
    x3021, grad_in_desc, x2301));
};
float* x3024 = (float*)myMalloc(1 * sizeof(float));;
x3024[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3024, in_desc, x2299, grad_out_desc, x2308,
    conv_desc, algo, ws_data, ws_size,
    x3024, grad_filt_desc, x1129));
};
float* x3027 = (float*)myMalloc(1 * sizeof(float));;
x3027[0] = 1.0f;
float* x3029 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x3027, x_desc, x2299, x_desc, x2301, x_desc, x2281,
    x3027, x_desc, x2289));
};
float* x3031 = (float*)myMalloc(1 * sizeof(float));;
x3031[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x3031, grad_out_desc, x2289,
    x3031, grad_bias_desc, x2227));
};
float* x3034 = (float*)myMalloc(1 * sizeof(float));;
x3034[0] = 0.0f;
float* x3036 = (float*)myMalloc(1 * sizeof(float));;
x3036[0] = 1.0f;

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
    x3036, x3036, x3036, x3036, in_desc, x2274,
    out_desc, x2289, in_desc, x2280, sbmv_desc, x1009,
    x1345,x1253, 1.0E-5, x2282, x2283));
};
// conv2D back-propagate
float* x3040 = (float*)myMalloc(1 * sizeof(float));;
x3040[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3040, filt_desc, x562, grad_out_desc, x2280,
    conv_desc, algo, ws_data, ws_size,
    x3040, grad_in_desc, x2273));
};
float* x3043 = (float*)myMalloc(1 * sizeof(float));;
x3043[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3043, in_desc, x2271, grad_out_desc, x2280,
    conv_desc, algo, ws_data, ws_size,
    x3043, grad_filt_desc, x1196));
};
float* x3046 = (float*)myMalloc(1 * sizeof(float));;
x3046[0] = 1.0f;
float* x3048 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x3046, x_desc, x2271, x_desc, x2273, x_desc, x2258,
    x3046, x_desc, x2266));
};
float* x3050 = (float*)myMalloc(1 * sizeof(float));;
x3050[0] = 0.0f;
float* x3052 = (float*)myMalloc(1 * sizeof(float));;
x3052[0] = 1.0f;

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
    x3052, x3052, x3052, x3052, in_desc, x2251,
    out_desc, x2266, in_desc, x2257, sbmv_desc, x517,
    x1181,x1243, 1.0E-5, x2259, x2260));
};
// conv2D back-propagate
float* x3056 = (float*)myMalloc(1 * sizeof(float));;
x3056[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3056, filt_desc, x1042, grad_out_desc, x2257,
    conv_desc, algo, ws_data, ws_size,
    x3056, grad_in_desc, x2250));
};
float* x3059 = (float*)myMalloc(1 * sizeof(float));;
x3059[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3059, in_desc, x2248, grad_out_desc, x2257,
    conv_desc, algo, ws_data, ws_size,
    x3059, grad_filt_desc, x1356));
};
float* x3062 = (float*)myMalloc(1 * sizeof(float));;
x3062[0] = 1.0f;
float* x3064 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x3062, x_desc, x2248, x_desc, x2250, x_desc, x2235,
    x3062, x_desc, x2243));
};
float* x3066 = (float*)myMalloc(1 * sizeof(float));;
x3066[0] = 0.0f;
float* x3068 = (float*)myMalloc(1 * sizeof(float));;
x3068[0] = 1.0f;

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
    x3068, x3068, x3068, x3068, in_desc, x2228,
    out_desc, x2243, in_desc, x2234, sbmv_desc, x571,
    x1199,x1348, 1.0E-5, x2236, x2237));
};
// conv2D back-propagate
float* x3072 = (float*)myMalloc(1 * sizeof(float));;
x3072[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3072, filt_desc, x313, grad_out_desc, x2234,
    conv_desc, algo, ws_data, ws_size,
    x3072, grad_in_desc, x2227));
};
float* x3075 = (float*)myMalloc(1 * sizeof(float));;
x3075[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3075, in_desc, x2225, grad_out_desc, x2234,
    conv_desc, algo, ws_data, ws_size,
    x3075, grad_filt_desc, x1113));
};
float* x3078 = (float*)myMalloc(1 * sizeof(float));;
x3078[0] = 1.0f;
float* x3080 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x3078, x_desc, x2225, x_desc, x2227, x_desc, x2207,
    x3078, x_desc, x2215));
};
float* x3082 = (float*)myMalloc(1 * sizeof(float));;
x3082[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x3082, grad_out_desc, x2215,
    x3082, grad_bias_desc, x2153));
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
    x3087, x3087, x3087, x3087, in_desc, x2200,
    out_desc, x2215, in_desc, x2206, sbmv_desc, x1084,
    x1370,x1164, 1.0E-5, x2208, x2209));
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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3091, filt_desc, x643, grad_out_desc, x2206,
    conv_desc, algo, ws_data, ws_size,
    x3091, grad_in_desc, x2199));
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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3094, in_desc, x2197, grad_out_desc, x2206,
    conv_desc, algo, ws_data, ws_size,
    x3094, grad_filt_desc, x1223));
};
float* x3097 = (float*)myMalloc(1 * sizeof(float));;
x3097[0] = 1.0f;
float* x3099 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x3097, x_desc, x2197, x_desc, x2199, x_desc, x2184,
    x3097, x_desc, x2192));
};
float* x3101 = (float*)myMalloc(1 * sizeof(float));;
x3101[0] = 0.0f;
float* x3103 = (float*)myMalloc(1 * sizeof(float));;
x3103[0] = 1.0f;

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
    x3103, x3103, x3103, x3103, in_desc, x2177,
    out_desc, x2192, in_desc, x2183, sbmv_desc, x979,
    x1335,x1299, 1.0E-5, x2185, x2186));
};
// conv2D back-propagate
float* x3107 = (float*)myMalloc(1 * sizeof(float));;
x3107[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3107, filt_desc, x337, grad_out_desc, x2183,
    conv_desc, algo, ws_data, ws_size,
    x3107, grad_in_desc, x2176));
};
float* x3110 = (float*)myMalloc(1 * sizeof(float));;
x3110[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3110, in_desc, x2174, grad_out_desc, x2183,
    conv_desc, algo, ws_data, ws_size,
    x3110, grad_filt_desc, x1121));
};
float* x3113 = (float*)myMalloc(1 * sizeof(float));;
x3113[0] = 1.0f;
float* x3115 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x3113, x_desc, x2174, x_desc, x2176, x_desc, x2161,
    x3113, x_desc, x2169));
};
float* x3117 = (float*)myMalloc(1 * sizeof(float));;
x3117[0] = 0.0f;
float* x3119 = (float*)myMalloc(1 * sizeof(float));;
x3119[0] = 1.0f;

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
    x3119, x3119, x3119, x3119, in_desc, x2154,
    out_desc, x2169, in_desc, x2160, sbmv_desc, x682,
    x1236,x1304, 1.0E-5, x2162, x2163));
};
// conv2D back-propagate
float* x3123 = (float*)myMalloc(1 * sizeof(float));;
x3123[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3123, filt_desc, x949, grad_out_desc, x2160,
    conv_desc, algo, ws_data, ws_size,
    x3123, grad_in_desc, x2153));
};
float* x3126 = (float*)myMalloc(1 * sizeof(float));;
x3126[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3126, in_desc, x2151, grad_out_desc, x2160,
    conv_desc, algo, ws_data, ws_size,
    x3126, grad_filt_desc, x1325));
};
float* x3129 = (float*)myMalloc(1 * sizeof(float));;
x3129[0] = 1.0f;
float* x3131 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x3129, x_desc, x2151, x_desc, x2153, x_desc, x2133,
    x3129, x_desc, x2141));
};
float* x3133 = (float*)myMalloc(1 * sizeof(float));;
x3133[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x3133, grad_out_desc, x2141,
    x3133, grad_bias_desc, x2079));
};
float* x3136 = (float*)myMalloc(1 * sizeof(float));;
x3136[0] = 0.0f;
float* x3138 = (float*)myMalloc(1 * sizeof(float));;
x3138[0] = 1.0f;

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
    x3138, x3138, x3138, x3138, in_desc, x2126,
    out_desc, x2141, in_desc, x2132, sbmv_desc, x355,
    x1127,x1339, 1.0E-5, x2134, x2135));
};
// conv2D back-propagate
float* x3142 = (float*)myMalloc(1 * sizeof(float));;
x3142[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3142, filt_desc, x463, grad_out_desc, x2132,
    conv_desc, algo, ws_data, ws_size,
    x3142, grad_in_desc, x2125));
};
float* x3145 = (float*)myMalloc(1 * sizeof(float));;
x3145[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3145, in_desc, x2123, grad_out_desc, x2132,
    conv_desc, algo, ws_data, ws_size,
    x3145, grad_filt_desc, x1163));
};
float* x3148 = (float*)myMalloc(1 * sizeof(float));;
x3148[0] = 1.0f;
float* x3150 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x3148, x_desc, x2123, x_desc, x2125, x_desc, x2110,
    x3148, x_desc, x2118));
};
float* x3152 = (float*)myMalloc(1 * sizeof(float));;
x3152[0] = 0.0f;
float* x3154 = (float*)myMalloc(1 * sizeof(float));;
x3154[0] = 1.0f;

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
    x3154, x3154, x3154, x3154, in_desc, x2103,
    out_desc, x2118, in_desc, x2109, sbmv_desc, x1108,
    x1378,x1203, 1.0E-5, x2111, x2112));
};
// conv2D back-propagate
float* x3158 = (float*)myMalloc(1 * sizeof(float));;
x3158[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3158, filt_desc, x388, grad_out_desc, x2109,
    conv_desc, algo, ws_data, ws_size,
    x3158, grad_in_desc, x2102));
};
float* x3161 = (float*)myMalloc(1 * sizeof(float));;
x3161[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3161, in_desc, x2100, grad_out_desc, x2109,
    conv_desc, algo, ws_data, ws_size,
    x3161, grad_filt_desc, x1138));
};
float* x3164 = (float*)myMalloc(1 * sizeof(float));;
x3164[0] = 1.0f;
float* x3166 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x3164, x_desc, x2100, x_desc, x2102, x_desc, x2087,
    x3164, x_desc, x2095));
};
float* x3168 = (float*)myMalloc(1 * sizeof(float));;
x3168[0] = 0.0f;
float* x3170 = (float*)myMalloc(1 * sizeof(float));;
x3170[0] = 1.0f;

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
    x3170, x3170, x3170, x3170, in_desc, x2080,
    out_desc, x2095, in_desc, x2086, sbmv_desc, x385,
    x1137,x1326, 1.0E-5, x2088, x2089));
};
// conv2D back-propagate
float* x3174 = (float*)myMalloc(1 * sizeof(float));;
x3174[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3174, filt_desc, x334, grad_out_desc, x2086,
    conv_desc, algo, ws_data, ws_size,
    x3174, grad_in_desc, x2079));
};
float* x3177 = (float*)myMalloc(1 * sizeof(float));;
x3177[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3177, in_desc, x2077, grad_out_desc, x2086,
    conv_desc, algo, ws_data, ws_size,
    x3177, grad_filt_desc, x1120));
};
float* x3180 = (float*)myMalloc(1 * sizeof(float));;
x3180[0] = 1.0f;
float* x3182 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x3180, x_desc, x2077, x_desc, x2079, x_desc, x2043,
    x3180, x_desc, x2051));
};
float* x3184 = (float*)myMalloc(1 * sizeof(float));;
x3184[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 1024, 4, 4));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x3184, grad_out_desc, x2051,
    x3184, grad_bias_desc, x2067));
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
    x3189, x3189, x3189, x3189, in_desc, x2052,
    out_desc, x2067, in_desc, x2058, sbmv_desc, x382,
    x1136,x1327, 1.0E-5, x2060, x2061));
};
// conv2D back-propagate
float* x3193 = (float*)myMalloc(1 * sizeof(float));;
x3193[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3193, filt_desc, x520, grad_out_desc, x2058,
    conv_desc, algo, ws_data, ws_size,
    x3193, grad_in_desc, x1989));
};
float* x3196 = (float*)myMalloc(1 * sizeof(float));;
x3196[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3196, in_desc, x1987, grad_out_desc, x2058,
    conv_desc, algo, ws_data, ws_size,
    x3196, grad_filt_desc, x1182));
};
float* x3199 = (float*)myMalloc(1 * sizeof(float));;
x3199[0] = 0.0f;
float* x3201 = (float*)myMalloc(1 * sizeof(float));;
x3201[0] = 1.0f;

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
    x3201, x3201, x3201, x3201, in_desc, x2036,
    out_desc, x2051, in_desc, x2042, sbmv_desc, x349,
    x1125,x1224, 1.0E-5, x2044, x2045));
};
// conv2D back-propagate
float* x3205 = (float*)myMalloc(1 * sizeof(float));;
x3205[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3205, filt_desc, x1102, grad_out_desc, x2042,
    conv_desc, algo, ws_data, ws_size,
    x3205, grad_in_desc, x2035));
};
float* x3208 = (float*)myMalloc(1 * sizeof(float));;
x3208[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3208, in_desc, x2033, grad_out_desc, x2042,
    conv_desc, algo, ws_data, ws_size,
    x3208, grad_filt_desc, x1376));
};
float* x3211 = (float*)myMalloc(1 * sizeof(float));;
x3211[0] = 1.0f;
float* x3213 = (float*)myGpuMalloc(262144 * sizeof(float));

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
    x3211, x_desc, x2033, x_desc, x2035, x_desc, x2020,
    x3211, x_desc, x2028));
};
float* x3215 = (float*)myMalloc(1 * sizeof(float));;
x3215[0] = 0.0f;
float* x3217 = (float*)myMalloc(1 * sizeof(float));;
x3217[0] = 1.0f;

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
    x3217, x3217, x3217, x3217, in_desc, x2013,
    out_desc, x2028, in_desc, x2019, sbmv_desc, x619,
    x1215,x1123, 1.0E-5, x2021, x2022));
};
// conv2D back-propagate
float* x3221 = (float*)myMalloc(1 * sizeof(float));;
x3221[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3221, filt_desc, x820, grad_out_desc, x2019,
    conv_desc, algo, ws_data, ws_size,
    x3221, grad_in_desc, x2012));
};
float* x3224 = (float*)myMalloc(1 * sizeof(float));;
x3224[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3224, in_desc, x2010, grad_out_desc, x2019,
    conv_desc, algo, ws_data, ws_size,
    x3224, grad_filt_desc, x1282));
};
float* x3227 = (float*)myMalloc(1 * sizeof(float));;
x3227[0] = 1.0f;
float* x3229 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x3227, x_desc, x2010, x_desc, x2012, x_desc, x1997,
    x3227, x_desc, x2005));
};
float* x3231 = (float*)myMalloc(1 * sizeof(float));;
x3231[0] = 0.0f;
float* x3233 = (float*)myMalloc(1 * sizeof(float));;
x3233[0] = 1.0f;

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
    x3233, x3233, x3233, x3233, in_desc, x1990,
    out_desc, x2005, in_desc, x1996, sbmv_desc, x1105,
    x1377,x1128, 1.0E-5, x1998, x1999));
};
// conv2D back-propagate
float* x3237 = (float*)myMalloc(1 * sizeof(float));;
x3237[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3237, filt_desc, x835, grad_out_desc, x1996,
    conv_desc, algo, ws_data, ws_size,
    x3237, grad_in_desc, x1989));
};
float* x3240 = (float*)myMalloc(1 * sizeof(float));;
x3240[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3240, in_desc, x1987, grad_out_desc, x1996,
    conv_desc, algo, ws_data, ws_size,
    x3240, grad_filt_desc, x1287));
};
float* x3243 = (float*)myMalloc(1 * sizeof(float));;
x3243[0] = 1.0f;
float* x3245 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x3243, x_desc, x1987, x_desc, x1989, x_desc, x1969,
    x3243, x_desc, x1977));
};
float* x3247 = (float*)myMalloc(1 * sizeof(float));;
x3247[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x3247, grad_out_desc, x1977,
    x3247, grad_bias_desc, x1915));
};
float* x3250 = (float*)myMalloc(1 * sizeof(float));;
x3250[0] = 0.0f;
float* x3252 = (float*)myMalloc(1 * sizeof(float));;
x3252[0] = 1.0f;

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
    x3252, x3252, x3252, x3252, in_desc, x1962,
    out_desc, x1977, in_desc, x1968, sbmv_desc, x763,
    x1263,x1161, 1.0E-5, x1970, x1971));
};
// conv2D back-propagate
float* x3256 = (float*)myMalloc(1 * sizeof(float));;
x3256[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3256, filt_desc, x460, grad_out_desc, x1968,
    conv_desc, algo, ws_data, ws_size,
    x3256, grad_in_desc, x1961));
};
float* x3259 = (float*)myMalloc(1 * sizeof(float));;
x3259[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3259, in_desc, x1959, grad_out_desc, x1968,
    conv_desc, algo, ws_data, ws_size,
    x3259, grad_filt_desc, x1162));
};
float* x3262 = (float*)myMalloc(1 * sizeof(float));;
x3262[0] = 1.0f;
float* x3264 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x3262, x_desc, x1959, x_desc, x1961, x_desc, x1946,
    x3262, x_desc, x1954));
};
float* x3266 = (float*)myMalloc(1 * sizeof(float));;
x3266[0] = 0.0f;
float* x3268 = (float*)myMalloc(1 * sizeof(float));;
x3268[0] = 1.0f;

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
    x3268, x3268, x3268, x3268, in_desc, x1939,
    out_desc, x1954, in_desc, x1945, sbmv_desc, x532,
    x1186,x1145, 1.0E-5, x1947, x1948));
};
// conv2D back-propagate
float* x3272 = (float*)myMalloc(1 * sizeof(float));;
x3272[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3272, filt_desc, x790, grad_out_desc, x1945,
    conv_desc, algo, ws_data, ws_size,
    x3272, grad_in_desc, x1938));
};
float* x3275 = (float*)myMalloc(1 * sizeof(float));;
x3275[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3275, in_desc, x1936, grad_out_desc, x1945,
    conv_desc, algo, ws_data, ws_size,
    x3275, grad_filt_desc, x1272));
};
float* x3278 = (float*)myMalloc(1 * sizeof(float));;
x3278[0] = 1.0f;
float* x3280 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x3278, x_desc, x1936, x_desc, x1938, x_desc, x1923,
    x3278, x_desc, x1931));
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
    x3284, x3284, x3284, x3284, in_desc, x1916,
    out_desc, x1931, in_desc, x1922, sbmv_desc, x412,
    x1146,x1349, 1.0E-5, x1924, x1925));
};
// conv2D back-propagate
float* x3288 = (float*)myMalloc(1 * sizeof(float));;
x3288[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3288, filt_desc, x691, grad_out_desc, x1922,
    conv_desc, algo, ws_data, ws_size,
    x3288, grad_in_desc, x1915));
};
float* x3291 = (float*)myMalloc(1 * sizeof(float));;
x3291[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3291, in_desc, x1913, grad_out_desc, x1922,
    conv_desc, algo, ws_data, ws_size,
    x3291, grad_filt_desc, x1239));
};
float* x3294 = (float*)myMalloc(1 * sizeof(float));;
x3294[0] = 1.0f;
float* x3296 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x3294, x_desc, x1913, x_desc, x1915, x_desc, x1895,
    x3294, x_desc, x1903));
};
float* x3298 = (float*)myMalloc(1 * sizeof(float));;
x3298[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x3298, grad_out_desc, x1903,
    x3298, grad_bias_desc, x1841));
};
float* x3301 = (float*)myMalloc(1 * sizeof(float));;
x3301[0] = 0.0f;
float* x3303 = (float*)myMalloc(1 * sizeof(float));;
x3303[0] = 1.0f;

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
    x3303, x3303, x3303, x3303, in_desc, x1888,
    out_desc, x1903, in_desc, x1894, sbmv_desc, x796,
    x1274,x1189, 1.0E-5, x1896, x1897));
};
// conv2D back-propagate
float* x3307 = (float*)myMalloc(1 * sizeof(float));;
x3307[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3307, filt_desc, x418, grad_out_desc, x1894,
    conv_desc, algo, ws_data, ws_size,
    x3307, grad_in_desc, x1887));
};
float* x3310 = (float*)myMalloc(1 * sizeof(float));;
x3310[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3310, in_desc, x1885, grad_out_desc, x1894,
    conv_desc, algo, ws_data, ws_size,
    x3310, grad_filt_desc, x1148));
};
float* x3313 = (float*)myMalloc(1 * sizeof(float));;
x3313[0] = 1.0f;
float* x3315 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x3313, x_desc, x1885, x_desc, x1887, x_desc, x1872,
    x3313, x_desc, x1880));
};
float* x3317 = (float*)myMalloc(1 * sizeof(float));;
x3317[0] = 0.0f;
float* x3319 = (float*)myMalloc(1 * sizeof(float));;
x3319[0] = 1.0f;

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
    x3319, x3319, x3319, x3319, in_desc, x1865,
    out_desc, x1880, in_desc, x1871, sbmv_desc, x676,
    x1234,x1168, 1.0E-5, x1873, x1874));
};
// conv2D back-propagate
float* x3323 = (float*)myMalloc(1 * sizeof(float));;
x3323[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3323, filt_desc, x868, grad_out_desc, x1871,
    conv_desc, algo, ws_data, ws_size,
    x3323, grad_in_desc, x1864));
};
float* x3326 = (float*)myMalloc(1 * sizeof(float));;
x3326[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3326, in_desc, x1862, grad_out_desc, x1871,
    conv_desc, algo, ws_data, ws_size,
    x3326, grad_filt_desc, x1298));
};
float* x3329 = (float*)myMalloc(1 * sizeof(float));;
x3329[0] = 1.0f;
float* x3331 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x3329, x_desc, x1862, x_desc, x1864, x_desc, x1849,
    x3329, x_desc, x1857));
};
float* x3333 = (float*)myMalloc(1 * sizeof(float));;
x3333[0] = 0.0f;
float* x3335 = (float*)myMalloc(1 * sizeof(float));;
x3335[0] = 1.0f;

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
    x3335, x3335, x3335, x3335, in_desc, x1842,
    out_desc, x1857, in_desc, x1848, sbmv_desc, x430,
    x1152,x1277, 1.0E-5, x1850, x1851));
};
// conv2D back-propagate
float* x3339 = (float*)myMalloc(1 * sizeof(float));;
x3339[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3339, filt_desc, x883, grad_out_desc, x1848,
    conv_desc, algo, ws_data, ws_size,
    x3339, grad_in_desc, x1841));
};
float* x3342 = (float*)myMalloc(1 * sizeof(float));;
x3342[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3342, in_desc, x1839, grad_out_desc, x1848,
    conv_desc, algo, ws_data, ws_size,
    x3342, grad_filt_desc, x1303));
};
float* x3345 = (float*)myMalloc(1 * sizeof(float));;
x3345[0] = 1.0f;
float* x3347 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x3345, x_desc, x1839, x_desc, x1841, x_desc, x1821,
    x3345, x_desc, x1829));
};
float* x3349 = (float*)myMalloc(1 * sizeof(float));;
x3349[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x3349, grad_out_desc, x1829,
    x3349, grad_bias_desc, x1767));
};
float* x3352 = (float*)myMalloc(1 * sizeof(float));;
x3352[0] = 0.0f;
float* x3354 = (float*)myMalloc(1 * sizeof(float));;
x3354[0] = 1.0f;

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
    x3354, x3354, x3354, x3354, in_desc, x1814,
    out_desc, x1829, in_desc, x1820, sbmv_desc, x451,
    x1159,x1353, 1.0E-5, x1822, x1823));
};
// conv2D back-propagate
float* x3358 = (float*)myMalloc(1 * sizeof(float));;
x3358[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3358, filt_desc, x628, grad_out_desc, x1820,
    conv_desc, algo, ws_data, ws_size,
    x3358, grad_in_desc, x1813));
};
float* x3361 = (float*)myMalloc(1 * sizeof(float));;
x3361[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3361, in_desc, x1811, grad_out_desc, x1820,
    conv_desc, algo, ws_data, ws_size,
    x3361, grad_filt_desc, x1218));
};
float* x3364 = (float*)myMalloc(1 * sizeof(float));;
x3364[0] = 1.0f;
float* x3366 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x3364, x_desc, x1811, x_desc, x1813, x_desc, x1798,
    x3364, x_desc, x1806));
};
float* x3368 = (float*)myMalloc(1 * sizeof(float));;
x3368[0] = 0.0f;
float* x3370 = (float*)myMalloc(1 * sizeof(float));;
x3370[0] = 1.0f;

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
    x3370, x3370, x3370, x3370, in_desc, x1791,
    out_desc, x1806, in_desc, x1797, sbmv_desc, x319,
    x1115,x1202, 1.0E-5, x1799, x1800));
};
// conv2D back-propagate
float* x3374 = (float*)myMalloc(1 * sizeof(float));;
x3374[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3374, filt_desc, x1000, grad_out_desc, x1797,
    conv_desc, algo, ws_data, ws_size,
    x3374, grad_in_desc, x1790));
};
float* x3377 = (float*)myMalloc(1 * sizeof(float));;
x3377[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3377, in_desc, x1788, grad_out_desc, x1797,
    conv_desc, algo, ws_data, ws_size,
    x3377, grad_filt_desc, x1342));
};
float* x3380 = (float*)myMalloc(1 * sizeof(float));;
x3380[0] = 1.0f;
float* x3382 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x3380, x_desc, x1788, x_desc, x1790, x_desc, x1775,
    x3380, x_desc, x1783));
};
float* x3384 = (float*)myMalloc(1 * sizeof(float));;
x3384[0] = 0.0f;
float* x3386 = (float*)myMalloc(1 * sizeof(float));;
x3386[0] = 1.0f;

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
    x3386, x3386, x3386, x3386, in_desc, x1768,
    out_desc, x1783, in_desc, x1774, sbmv_desc, x961,
    x1329,x1124, 1.0E-5, x1776, x1777));
};
// conv2D back-propagate
float* x3390 = (float*)myMalloc(1 * sizeof(float));;
x3390[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3390, filt_desc, x1063, grad_out_desc, x1774,
    conv_desc, algo, ws_data, ws_size,
    x3390, grad_in_desc, x1767));
};
float* x3393 = (float*)myMalloc(1 * sizeof(float));;
x3393[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3393, in_desc, x1765, grad_out_desc, x1774,
    conv_desc, algo, ws_data, ws_size,
    x3393, grad_filt_desc, x1363));
};
float* x3396 = (float*)myMalloc(1 * sizeof(float));;
x3396[0] = 1.0f;
float* x3398 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x3396, x_desc, x1765, x_desc, x1767, x_desc, x1731,
    x3396, x_desc, x1739));
};
float* x3400 = (float*)myMalloc(1 * sizeof(float));;
x3400[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 512, 8, 8));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x3400, grad_out_desc, x1739,
    x3400, grad_bias_desc, x1755));
};
float* x3403 = (float*)myMalloc(1 * sizeof(float));;
x3403[0] = 0.0f;
float* x3405 = (float*)myMalloc(1 * sizeof(float));;
x3405[0] = 1.0f;

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
    x3405, x3405, x3405, x3405, in_desc, x1740,
    out_desc, x1755, in_desc, x1746, sbmv_desc, x916,
    x1314,x1226, 1.0E-5, x1748, x1749));
};
// conv2D back-propagate
float* x3409 = (float*)myMalloc(1 * sizeof(float));;
x3409[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3409, filt_desc, x1069, grad_out_desc, x1746,
    conv_desc, algo, ws_data, ws_size,
    x3409, grad_in_desc, x1677));
};
float* x3412 = (float*)myMalloc(1 * sizeof(float));;
x3412[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3412, in_desc, x1675, grad_out_desc, x1746,
    conv_desc, algo, ws_data, ws_size,
    x3412, grad_filt_desc, x1365));
};
float* x3415 = (float*)myMalloc(1 * sizeof(float));;
x3415[0] = 0.0f;
float* x3417 = (float*)myMalloc(1 * sizeof(float));;
x3417[0] = 1.0f;

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
    x3417, x3417, x3417, x3417, in_desc, x1724,
    out_desc, x1739, in_desc, x1730, sbmv_desc, x730,
    x1252,x1317, 1.0E-5, x1732, x1733));
};
// conv2D back-propagate
float* x3421 = (float*)myMalloc(1 * sizeof(float));;
x3421[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3421, filt_desc, x613, grad_out_desc, x1730,
    conv_desc, algo, ws_data, ws_size,
    x3421, grad_in_desc, x1723));
};
float* x3424 = (float*)myMalloc(1 * sizeof(float));;
x3424[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3424, in_desc, x1721, grad_out_desc, x1730,
    conv_desc, algo, ws_data, ws_size,
    x3424, grad_filt_desc, x1213));
};
float* x3427 = (float*)myMalloc(1 * sizeof(float));;
x3427[0] = 1.0f;
float* x3429 = (float*)myGpuMalloc(524288 * sizeof(float));

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
    x3427, x_desc, x1721, x_desc, x1723, x_desc, x1708,
    x3427, x_desc, x1716));
};
float* x3431 = (float*)myMalloc(1 * sizeof(float));;
x3431[0] = 0.0f;
float* x3433 = (float*)myMalloc(1 * sizeof(float));;
x3433[0] = 1.0f;

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
    x3433, x3433, x3433, x3433, in_desc, x1701,
    out_desc, x1716, in_desc, x1707, sbmv_desc, x1051,
    x1359,x1297, 1.0E-5, x1709, x1710));
};
// conv2D back-propagate
float* x3437 = (float*)myMalloc(1 * sizeof(float));;
x3437[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3437, filt_desc, x376, grad_out_desc, x1707,
    conv_desc, algo, ws_data, ws_size,
    x3437, grad_in_desc, x1700));
};
float* x3440 = (float*)myMalloc(1 * sizeof(float));;
x3440[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3440, in_desc, x1698, grad_out_desc, x1707,
    conv_desc, algo, ws_data, ws_size,
    x3440, grad_filt_desc, x1134));
};
float* x3443 = (float*)myMalloc(1 * sizeof(float));;
x3443[0] = 1.0f;
float* x3445 = (float*)myGpuMalloc(2097152 * sizeof(float));

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
    x3443, x_desc, x1698, x_desc, x1700, x_desc, x1685,
    x3443, x_desc, x1693));
};
float* x3447 = (float*)myMalloc(1 * sizeof(float));;
x3447[0] = 0.0f;
float* x3449 = (float*)myMalloc(1 * sizeof(float));;
x3449[0] = 1.0f;

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
    x3449, x3449, x3449, x3449, in_desc, x1678,
    out_desc, x1693, in_desc, x1684, sbmv_desc, x547,
    x1191,x1279, 1.0E-5, x1686, x1687));
};
// conv2D back-propagate
float* x3453 = (float*)myMalloc(1 * sizeof(float));;
x3453[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3453, filt_desc, x328, grad_out_desc, x1684,
    conv_desc, algo, ws_data, ws_size,
    x3453, grad_in_desc, x1677));
};
float* x3456 = (float*)myMalloc(1 * sizeof(float));;
x3456[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3456, in_desc, x1675, grad_out_desc, x1684,
    conv_desc, algo, ws_data, ws_size,
    x3456, grad_filt_desc, x1118));
};
float* x3459 = (float*)myMalloc(1 * sizeof(float));;
x3459[0] = 1.0f;
float* x3461 = (float*)myGpuMalloc(4194304 * sizeof(float));

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
    x3459, x_desc, x1675, x_desc, x1677, x_desc, x1657,
    x3459, x_desc, x1665));
};
float* x3463 = (float*)myMalloc(1 * sizeof(float));;
x3463[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x3463, grad_out_desc, x1665,
    x3463, grad_bias_desc, x1603));
};
float* x3466 = (float*)myMalloc(1 * sizeof(float));;
x3466[0] = 0.0f;
float* x3468 = (float*)myMalloc(1 * sizeof(float));;
x3468[0] = 1.0f;

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
    x3468, x3468, x3468, x3468, in_desc, x1650,
    out_desc, x1665, in_desc, x1656, sbmv_desc, x406,
    x1144,x1354, 1.0E-5, x1658, x1659));
};
// conv2D back-propagate
float* x3472 = (float*)myMalloc(1 * sizeof(float));;
x3472[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3472, filt_desc, x556, grad_out_desc, x1656,
    conv_desc, algo, ws_data, ws_size,
    x3472, grad_in_desc, x1649));
};
float* x3475 = (float*)myMalloc(1 * sizeof(float));;
x3475[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3475, in_desc, x1647, grad_out_desc, x1656,
    conv_desc, algo, ws_data, ws_size,
    x3475, grad_filt_desc, x1194));
};
float* x3478 = (float*)myMalloc(1 * sizeof(float));;
x3478[0] = 1.0f;
float* x3480 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x3478, x_desc, x1647, x_desc, x1649, x_desc, x1634,
    x3478, x_desc, x1642));
};
float* x3482 = (float*)myMalloc(1 * sizeof(float));;
x3482[0] = 0.0f;
float* x3484 = (float*)myMalloc(1 * sizeof(float));;
x3484[0] = 1.0f;

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
    x3484, x3484, x3484, x3484, in_desc, x1627,
    out_desc, x1642, in_desc, x1633, sbmv_desc, x511,
    x1179,x1242, 1.0E-5, x1635, x1636));
};
// conv2D back-propagate
float* x3488 = (float*)myMalloc(1 * sizeof(float));;
x3488[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3488, filt_desc, x514, grad_out_desc, x1633,
    conv_desc, algo, ws_data, ws_size,
    x3488, grad_in_desc, x1626));
};
float* x3491 = (float*)myMalloc(1 * sizeof(float));;
x3491[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3491, in_desc, x1624, grad_out_desc, x1633,
    conv_desc, algo, ws_data, ws_size,
    x3491, grad_filt_desc, x1180));
};
float* x3494 = (float*)myMalloc(1 * sizeof(float));;
x3494[0] = 1.0f;
float* x3496 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x3494, x_desc, x1624, x_desc, x1626, x_desc, x1611,
    x3494, x_desc, x1619));
};
float* x3498 = (float*)myMalloc(1 * sizeof(float));;
x3498[0] = 0.0f;
float* x3500 = (float*)myMalloc(1 * sizeof(float));;
x3500[0] = 1.0f;

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
    x3500, x3500, x3500, x3500, in_desc, x1604,
    out_desc, x1619, in_desc, x1610, sbmv_desc, x538,
    x1188,x1131, 1.0E-5, x1612, x1613));
};
// conv2D back-propagate
float* x3504 = (float*)myMalloc(1 * sizeof(float));;
x3504[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3504, filt_desc, x745, grad_out_desc, x1610,
    conv_desc, algo, ws_data, ws_size,
    x3504, grad_in_desc, x1603));
};
float* x3507 = (float*)myMalloc(1 * sizeof(float));;
x3507[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3507, in_desc, x1601, grad_out_desc, x1610,
    conv_desc, algo, ws_data, ws_size,
    x3507, grad_filt_desc, x1257));
};
float* x3510 = (float*)myMalloc(1 * sizeof(float));;
x3510[0] = 1.0f;
float* x3512 = (float*)myGpuMalloc(4194304 * sizeof(float));

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
    x3510, x_desc, x1601, x_desc, x1603, x_desc, x1583,
    x3510, x_desc, x1591));
};
float* x3514 = (float*)myMalloc(1 * sizeof(float));;
x3514[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x3514, grad_out_desc, x1591,
    x3514, grad_bias_desc, x1529));
};
float* x3517 = (float*)myMalloc(1 * sizeof(float));;
x3517[0] = 0.0f;
float* x3519 = (float*)myMalloc(1 * sizeof(float));;
x3519[0] = 1.0f;

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
    x3519, x3519, x3519, x3519, in_desc, x1576,
    out_desc, x1591, in_desc, x1582, sbmv_desc, x469,
    x1165,x1114, 1.0E-5, x1584, x1585));
};
// conv2D back-propagate
float* x3523 = (float*)myMalloc(1 * sizeof(float));;
x3523[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3523, filt_desc, x685, grad_out_desc, x1582,
    conv_desc, algo, ws_data, ws_size,
    x3523, grad_in_desc, x1575));
};
float* x3526 = (float*)myMalloc(1 * sizeof(float));;
x3526[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3526, in_desc, x1573, grad_out_desc, x1582,
    conv_desc, algo, ws_data, ws_size,
    x3526, grad_filt_desc, x1237));
};
float* x3529 = (float*)myMalloc(1 * sizeof(float));;
x3529[0] = 1.0f;
float* x3531 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x3529, x_desc, x1573, x_desc, x1575, x_desc, x1560,
    x3529, x_desc, x1568));
};
float* x3533 = (float*)myMalloc(1 * sizeof(float));;
x3533[0] = 0.0f;
float* x3535 = (float*)myMalloc(1 * sizeof(float));;
x3535[0] = 1.0f;

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
    x3535, x3535, x3535, x3535, in_desc, x1553,
    out_desc, x1568, in_desc, x1559, sbmv_desc, x919,
    x1315,x1260, 1.0E-5, x1561, x1562));
};
// conv2D back-propagate
float* x3539 = (float*)myMalloc(1 * sizeof(float));;
x3539[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3539, filt_desc, x544, grad_out_desc, x1559,
    conv_desc, algo, ws_data, ws_size,
    x3539, grad_in_desc, x1552));
};
float* x3542 = (float*)myMalloc(1 * sizeof(float));;
x3542[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3542, in_desc, x1550, grad_out_desc, x1559,
    conv_desc, algo, ws_data, ws_size,
    x3542, grad_filt_desc, x1190));
};
float* x3545 = (float*)myMalloc(1 * sizeof(float));;
x3545[0] = 1.0f;
float* x3547 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x3545, x_desc, x1550, x_desc, x1552, x_desc, x1537,
    x3545, x_desc, x1545));
};
float* x3549 = (float*)myMalloc(1 * sizeof(float));;
x3549[0] = 0.0f;
float* x3551 = (float*)myMalloc(1 * sizeof(float));;
x3551[0] = 1.0f;

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
    x3551, x3551, x3551, x3551, in_desc, x1530,
    out_desc, x1545, in_desc, x1536, sbmv_desc, x721,
    x1249,x1167, 1.0E-5, x1538, x1539));
};
// conv2D back-propagate
float* x3555 = (float*)myMalloc(1 * sizeof(float));;
x3555[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3555, filt_desc, x808, grad_out_desc, x1536,
    conv_desc, algo, ws_data, ws_size,
    x3555, grad_in_desc, x1529));
};
float* x3558 = (float*)myMalloc(1 * sizeof(float));;
x3558[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3558, in_desc, x1527, grad_out_desc, x1536,
    conv_desc, algo, ws_data, ws_size,
    x3558, grad_filt_desc, x1278));
};
float* x3561 = (float*)myMalloc(1 * sizeof(float));;
x3561[0] = 1.0f;
float* x3563 = (float*)myGpuMalloc(4194304 * sizeof(float));

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
    x3561, x_desc, x1527, x_desc, x1529, x_desc, x1493,
    x3561, x_desc, x1501));
};
float* x3565 = (float*)myMalloc(1 * sizeof(float));;
x3565[0] = 1.0f;

{
cudnnTensorDescriptor_t grad_bias_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    64, 256, 16, 16));

CUDNN_CALL(cudnnConvolutionBackwardBias(
    cudnnHandle, x3565, grad_out_desc, x1501,
    x3565, grad_bias_desc, x1517));
};
float* x3568 = (float*)myMalloc(1 * sizeof(float));;
x3568[0] = 0.0f;
float* x3570 = (float*)myMalloc(1 * sizeof(float));;
x3570[0] = 1.0f;

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
    x3570, x3570, x3570, x3570, in_desc, x1502,
    out_desc, x1517, in_desc, x1508, sbmv_desc, x523,
    x1183,x1310, 1.0E-5, x1510, x1511));
};
// conv2D back-propagate
float* x3574 = (float*)myMalloc(1 * sizeof(float));;
x3574[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3574, filt_desc, x781, grad_out_desc, x1508,
    conv_desc, algo, ws_data, ws_size,
    x3574, grad_in_desc, x1439));
};
float* x3577 = (float*)myMalloc(1 * sizeof(float));;
x3577[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3577, in_desc, x1437, grad_out_desc, x1508,
    conv_desc, algo, ws_data, ws_size,
    x3577, grad_filt_desc, x1269));
};
float* x3580 = (float*)myMalloc(1 * sizeof(float));;
x3580[0] = 0.0f;
float* x3582 = (float*)myMalloc(1 * sizeof(float));;
x3582[0] = 1.0f;

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
    x3582, x3582, x3582, x3582, in_desc, x1486,
    out_desc, x1501, in_desc, x1492, sbmv_desc, x892,
    x1306,x1233, 1.0E-5, x1494, x1495));
};
// conv2D back-propagate
float* x3586 = (float*)myMalloc(1 * sizeof(float));;
x3586[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3586, filt_desc, x391, grad_out_desc, x1492,
    conv_desc, algo, ws_data, ws_size,
    x3586, grad_in_desc, x1485));
};
float* x3589 = (float*)myMalloc(1 * sizeof(float));;
x3589[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3589, in_desc, x1483, grad_out_desc, x1492,
    conv_desc, algo, ws_data, ws_size,
    x3589, grad_filt_desc, x1139));
};
float* x3592 = (float*)myMalloc(1 * sizeof(float));;
x3592[0] = 1.0f;
float* x3594 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x3592, x_desc, x1483, x_desc, x1485, x_desc, x1470,
    x3592, x_desc, x1478));
};
float* x3596 = (float*)myMalloc(1 * sizeof(float));;
x3596[0] = 0.0f;
float* x3598 = (float*)myMalloc(1 * sizeof(float));;
x3598[0] = 1.0f;

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
    x3598, x3598, x3598, x3598, in_desc, x1463,
    out_desc, x1478, in_desc, x1469, sbmv_desc, x787,
    x1271,x1156, 1.0E-5, x1471, x1472));
};
// conv2D back-propagate
float* x3602 = (float*)myMalloc(1 * sizeof(float));;
x3602[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3602, filt_desc, x565, grad_out_desc, x1469,
    conv_desc, algo, ws_data, ws_size,
    x3602, grad_in_desc, x1462));
};
float* x3605 = (float*)myMalloc(1 * sizeof(float));;
x3605[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3605, in_desc, x1460, grad_out_desc, x1469,
    conv_desc, algo, ws_data, ws_size,
    x3605, grad_filt_desc, x1197));
};
float* x3608 = (float*)myMalloc(1 * sizeof(float));;
x3608[0] = 1.0f;
float* x3610 = (float*)myGpuMalloc(1048576 * sizeof(float));

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
    x3608, x_desc, x1460, x_desc, x1462, x_desc, x1447,
    x3608, x_desc, x1455));
};
float* x3612 = (float*)myMalloc(1 * sizeof(float));;
x3612[0] = 0.0f;
float* x3614 = (float*)myMalloc(1 * sizeof(float));;
x3614[0] = 1.0f;

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
    x3614, x3614, x3614, x3614, in_desc, x1440,
    out_desc, x1455, in_desc, x1446, sbmv_desc, x373,
    x1133,x1160, 1.0E-5, x1448, x1449));
};
// conv2D back-propagate
float* x3618 = (float*)myMalloc(1 * sizeof(float));;
x3618[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardData(
    cudnnHandle,
    x3618, filt_desc, x994, grad_out_desc, x1446,
    conv_desc, algo, ws_data, ws_size,
    x3618, grad_in_desc, x1439));
};
float* x3621 = (float*)myMalloc(1 * sizeof(float));;
x3621[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3621, in_desc, x1437, grad_out_desc, x1446,
    conv_desc, algo, ws_data, ws_size,
    x3621, grad_filt_desc, x1340));
};
float* x3624 = (float*)myMalloc(1 * sizeof(float));;
x3624[0] = 0.0f;
float* x3626 = (float*)myMalloc(1 * sizeof(float));;
x3626[0] = 1.0f;

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
    x3626, out_desc, x1437, out_desc, x1439, in_desc, x1430  , x3624, in_desc, x1432));
};
float* x3629 = (float*)myMalloc(1 * sizeof(float));;
x3629[0] = 1.0f;
float* x3631 = (float*)myGpuMalloc(4194304 * sizeof(float));

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
    x3629, x_desc, x1430, x_desc, x1432, x_desc, x1417,
    x3629, x_desc, x1425));
};
float* x3633 = (float*)myMalloc(1 * sizeof(float));;
x3633[0] = 0.0f;
float* x3635 = (float*)myMalloc(1 * sizeof(float));;
x3635[0] = 1.0f;

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
    x3635, x3635, x3635, x3635, in_desc, x1410,
    out_desc, x1425, in_desc, x1416, sbmv_desc, x913,
    x1313,x1358, 1.0E-5, x1418, x1419));
};
// conv2D back-propagate
float* x3639 = (float*)myMalloc(1 * sizeof(float));;
x3639[0] = 1.0f;

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

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnConvolutionBackwardFilter(
    cudnnHandle,
    x3639, in_desc, x1401, grad_out_desc, x1416,
    conv_desc, algo, ws_data, ws_size,
    x3639, grad_filt_desc, x1259));
};
float x3642 = x1409[0];
x1389 += x3642;
float* x3644 = (float*)myMalloc(1 * sizeof(float));;
x3644[0] = 1.0f;
float* x3646 = (float*)myMalloc(1 * sizeof(float));;
x3646[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,256,x3644,x313,1024,x3646, x1113, 1024, x313,1024));
arrayFill<<<262144, 1>>>(x1113, 0.0f);
float* x3650 = (float*)myMalloc(1 * sizeof(float));;
x3650[0] = 1.0f;
float* x3652 = (float*)myMalloc(1 * sizeof(float));;
x3652[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x3650,x316,1,x3652, x1114, 1, x316,1));
arrayFill<<<256, 1>>>(x1114, 0.0f);
float* x3656 = (float*)myMalloc(1 * sizeof(float));;
x3656[0] = 1.0f;
float* x3658 = (float*)myMalloc(1 * sizeof(float));;
x3658[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x3656,x319,1,x3658, x1115, 1, x319,1));
arrayFill<<<128, 1>>>(x1115, 0.0f);
float* x3662 = (float*)myMalloc(1 * sizeof(float));;
x3662[0] = 1.0f;
float* x3664 = (float*)myMalloc(1 * sizeof(float));;
x3664[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x3662,x322,1,x3664, x1116, 1, x322,1));
arrayFill<<<128, 1>>>(x1116, 0.0f);
float* x3668 = (float*)myMalloc(1 * sizeof(float));;
x3668[0] = 1.0f;
float* x3670 = (float*)myMalloc(1 * sizeof(float));;
x3670[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x3668,x325,1,x3670, x1117, 1, x325,1));
arrayFill<<<64, 1>>>(x1117, 0.0f);
float* x3674 = (float*)myMalloc(1 * sizeof(float));;
x3674[0] = 1.0f;
float* x3676 = (float*)myMalloc(1 * sizeof(float));;
x3676[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,128,x3674,x328,256,x3676, x1118, 256, x328,256));
arrayFill<<<32768, 1>>>(x1118, 0.0f);
float* x3680 = (float*)myMalloc(1 * sizeof(float));;
x3680[0] = 1.0f;
float* x3682 = (float*)myMalloc(1 * sizeof(float));;
x3682[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3680,x331,1,x3682, x1119, 1, x331,1));
arrayFill<<<512, 1>>>(x1119, 0.0f);
float* x3686 = (float*)myMalloc(1 * sizeof(float));;
x3686[0] = 1.0f;
float* x3688 = (float*)myMalloc(1 * sizeof(float));;
x3688[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,256,x3686,x334,1024,x3688, x1120, 1024, x334,1024));
arrayFill<<<262144, 1>>>(x1120, 0.0f);
float* x3692 = (float*)myMalloc(1 * sizeof(float));;
x3692[0] = 1.0f;
float* x3694 = (float*)myMalloc(1 * sizeof(float));;
x3694[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x3692,x337,2304,x3694, x1121, 2304, x337,2304));
arrayFill<<<589824, 1>>>(x1121, 0.0f);
float* x3698 = (float*)myMalloc(1 * sizeof(float));;
x3698[0] = 1.0f;
float* x3700 = (float*)myMalloc(1 * sizeof(float));;
x3700[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3698,x340,1,x3700, x1122, 1, x340,1));
arrayFill<<<512, 1>>>(x1122, 0.0f);
float* x3704 = (float*)myMalloc(1 * sizeof(float));;
x3704[0] = 1.0f;
float* x3706 = (float*)myMalloc(1 * sizeof(float));;
x3706[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x3704,x343,1,x3706, x1123, 1, x343,1));
arrayFill<<<256, 1>>>(x1123, 0.0f);
float* x3710 = (float*)myMalloc(1 * sizeof(float));;
x3710[0] = 1.0f;
float* x3712 = (float*)myMalloc(1 * sizeof(float));;
x3712[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x3710,x346,1,x3712, x1124, 1, x346,1));
arrayFill<<<128, 1>>>(x1124, 0.0f);
float* x3716 = (float*)myMalloc(1 * sizeof(float));;
x3716[0] = 1.0f;
float* x3718 = (float*)myMalloc(1 * sizeof(float));;
x3718[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x3716,x349,1,x3718, x1125, 1, x349,1));
arrayFill<<<1024, 1>>>(x1125, 0.0f);
float* x3722 = (float*)myMalloc(1 * sizeof(float));;
x3722[0] = 1.0f;
float* x3724 = (float*)myMalloc(1 * sizeof(float));;
x3724[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3722,x352,1,x3724, x1126, 1, x352,1));
arrayFill<<<512, 1>>>(x1126, 0.0f);
float* x3728 = (float*)myMalloc(1 * sizeof(float));;
x3728[0] = 1.0f;
float* x3730 = (float*)myMalloc(1 * sizeof(float));;
x3730[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x3728,x355,1,x3730, x1127, 1, x355,1));
arrayFill<<<1024, 1>>>(x1127, 0.0f);
float* x3734 = (float*)myMalloc(1 * sizeof(float));;
x3734[0] = 1.0f;
float* x3736 = (float*)myMalloc(1 * sizeof(float));;
x3736[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x3734,x358,1,x3736, x1128, 1, x358,1));
arrayFill<<<256, 1>>>(x1128, 0.0f);
float* x3740 = (float*)myMalloc(1 * sizeof(float));;
x3740[0] = 1.0f;
float* x3742 = (float*)myMalloc(1 * sizeof(float));;
x3742[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,256,x3740,x361,1024,x3742, x1129, 1024, x361,1024));
arrayFill<<<262144, 1>>>(x1129, 0.0f);
float* x3746 = (float*)myMalloc(1 * sizeof(float));;
x3746[0] = 1.0f;
float* x3748 = (float*)myMalloc(1 * sizeof(float));;
x3748[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3746,x364,1,x3748, x1130, 1, x364,1));
arrayFill<<<512, 1>>>(x1130, 0.0f);
float* x3752 = (float*)myMalloc(1 * sizeof(float));;
x3752[0] = 1.0f;
float* x3754 = (float*)myMalloc(1 * sizeof(float));;
x3754[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x3752,x367,1,x3754, x1131, 1, x367,1));
arrayFill<<<64, 1>>>(x1131, 0.0f);
float* x3758 = (float*)myMalloc(1 * sizeof(float));;
x3758[0] = 1.0f;
float* x3760 = (float*)myMalloc(1 * sizeof(float));;
x3760[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3758,x370,1,x3760, x1132, 1, x370,1));
arrayFill<<<512, 1>>>(x1132, 0.0f);
float* x3764 = (float*)myMalloc(1 * sizeof(float));;
x3764[0] = 1.0f;
float* x3766 = (float*)myMalloc(1 * sizeof(float));;
x3766[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x3764,x373,1,x3766, x1133, 1, x373,1));
arrayFill<<<64, 1>>>(x1133, 0.0f);
float* x3770 = (float*)myMalloc(1 * sizeof(float));;
x3770[0] = 1.0f;
float* x3772 = (float*)myMalloc(1 * sizeof(float));;
x3772[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1152,128,x3770,x376,1152,x3772, x1134, 1152, x376,1152));
arrayFill<<<147456, 1>>>(x1134, 0.0f);
float* x3776 = (float*)myMalloc(1 * sizeof(float));;
x3776[0] = 1.0f;
float* x3778 = (float*)myMalloc(1 * sizeof(float));;
x3778[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 4608,512,x3776,x379,4608,x3778, x1135, 4608, x379,4608));
arrayFill<<<2359296, 1>>>(x1135, 0.0f);
float* x3782 = (float*)myMalloc(1 * sizeof(float));;
x3782[0] = 1.0f;
float* x3784 = (float*)myMalloc(1 * sizeof(float));;
x3784[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x3782,x382,1,x3784, x1136, 1, x382,1));
arrayFill<<<1024, 1>>>(x1136, 0.0f);
float* x3788 = (float*)myMalloc(1 * sizeof(float));;
x3788[0] = 1.0f;
float* x3790 = (float*)myMalloc(1 * sizeof(float));;
x3790[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x3788,x385,1,x3790, x1137, 1, x385,1));
arrayFill<<<256, 1>>>(x1137, 0.0f);
float* x3794 = (float*)myMalloc(1 * sizeof(float));;
x3794[0] = 1.0f;
float* x3796 = (float*)myMalloc(1 * sizeof(float));;
x3796[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x3794,x388,2304,x3796, x1138, 2304, x388,2304));
arrayFill<<<589824, 1>>>(x1138, 0.0f);
float* x3800 = (float*)myMalloc(1 * sizeof(float));;
x3800[0] = 1.0f;
float* x3802 = (float*)myMalloc(1 * sizeof(float));;
x3802[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x3800,x391,64,x3802, x1139, 64, x391,64));
arrayFill<<<16384, 1>>>(x1139, 0.0f);
float* x3806 = (float*)myMalloc(1 * sizeof(float));;
x3806[0] = 1.0f;
float* x3808 = (float*)myMalloc(1 * sizeof(float));;
x3808[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,2048,x3806,x394,512,x3808, x1140, 512, x394,512));
arrayFill<<<1048576, 1>>>(x1140, 0.0f);
float* x3812 = (float*)myMalloc(1 * sizeof(float));;
x3812[0] = 1.0f;
float* x3814 = (float*)myMalloc(1 * sizeof(float));;
x3814[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 4608,512,x3812,x397,4608,x3814, x1141, 4608, x397,4608));
arrayFill<<<2359296, 1>>>(x1141, 0.0f);
float* x3818 = (float*)myMalloc(1 * sizeof(float));;
x3818[0] = 1.0f;
float* x3820 = (float*)myMalloc(1 * sizeof(float));;
x3820[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x3818,x400,1,x3820, x1142, 1, x400,1));
arrayFill<<<128, 1>>>(x1142, 0.0f);
float* x3824 = (float*)myMalloc(1 * sizeof(float));;
x3824[0] = 1.0f;
float* x3826 = (float*)myMalloc(1 * sizeof(float));;
x3826[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x3824,x403,1,x3826, x1143, 1, x403,1));
arrayFill<<<256, 1>>>(x1143, 0.0f);
float* x3830 = (float*)myMalloc(1 * sizeof(float));;
x3830[0] = 1.0f;
float* x3832 = (float*)myMalloc(1 * sizeof(float));;
x3832[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x3830,x406,1,x3832, x1144, 1, x406,1));
arrayFill<<<256, 1>>>(x1144, 0.0f);
float* x3836 = (float*)myMalloc(1 * sizeof(float));;
x3836[0] = 1.0f;
float* x3838 = (float*)myMalloc(1 * sizeof(float));;
x3838[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x3836,x409,1,x3838, x1145, 1, x409,1));
arrayFill<<<128, 1>>>(x1145, 0.0f);
float* x3842 = (float*)myMalloc(1 * sizeof(float));;
x3842[0] = 1.0f;
float* x3844 = (float*)myMalloc(1 * sizeof(float));;
x3844[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x3842,x412,1,x3844, x1146, 1, x412,1));
arrayFill<<<128, 1>>>(x1146, 0.0f);
float* x3848 = (float*)myMalloc(1 * sizeof(float));;
x3848[0] = 1.0f;
float* x3850 = (float*)myMalloc(1 * sizeof(float));;
x3850[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x3848,x415,1,x3850, x1147, 1, x415,1));
arrayFill<<<64, 1>>>(x1147, 0.0f);
float* x3854 = (float*)myMalloc(1 * sizeof(float));;
x3854[0] = 1.0f;
float* x3856 = (float*)myMalloc(1 * sizeof(float));;
x3856[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,512,x3854,x418,128,x3856, x1148, 128, x418,128));
arrayFill<<<65536, 1>>>(x1148, 0.0f);
float* x3860 = (float*)myMalloc(1 * sizeof(float));;
x3860[0] = 1.0f;
float* x3862 = (float*)myMalloc(1 * sizeof(float));;
x3862[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3860,x421,1,x3862, x1149, 1, x421,1));
arrayFill<<<512, 1>>>(x1149, 0.0f);
float* x3866 = (float*)myMalloc(1 * sizeof(float));;
x3866[0] = 1.0f;
float* x3868 = (float*)myMalloc(1 * sizeof(float));;
x3868[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x3866,x424,1,x3868, x1150, 1, x424,1));
arrayFill<<<128, 1>>>(x1150, 0.0f);
float* x3872 = (float*)myMalloc(1 * sizeof(float));;
x3872[0] = 1.0f;
float* x3874 = (float*)myMalloc(1 * sizeof(float));;
x3874[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x3872,x427,1,x3874, x1151, 1, x427,1));
arrayFill<<<64, 1>>>(x1151, 0.0f);
float* x3878 = (float*)myMalloc(1 * sizeof(float));;
x3878[0] = 1.0f;
float* x3880 = (float*)myMalloc(1 * sizeof(float));;
x3880[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x3878,x430,1,x3880, x1152, 1, x430,1));
arrayFill<<<128, 1>>>(x1152, 0.0f);
float* x3884 = (float*)myMalloc(1 * sizeof(float));;
x3884[0] = 1.0f;
float* x3886 = (float*)myMalloc(1 * sizeof(float));;
x3886[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3884,x433,1,x3886, x1153, 1, x433,1));
arrayFill<<<512, 1>>>(x1153, 0.0f);
float* x3890 = (float*)myMalloc(1 * sizeof(float));;
x3890[0] = 1.0f;
float* x3892 = (float*)myMalloc(1 * sizeof(float));;
x3892[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,2048,x3890,x436,512,x3892, x1154, 512, x436,512));
arrayFill<<<1048576, 1>>>(x1154, 0.0f);
float* x3896 = (float*)myMalloc(1 * sizeof(float));;
x3896[0] = 1.0f;
float* x3898 = (float*)myMalloc(1 * sizeof(float));;
x3898[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,10,x3896,x439,1,x3898, x1155, 1, x439,1));
arrayFill<<<10, 1>>>(x1155, 0.0f);
float* x3902 = (float*)myMalloc(1 * sizeof(float));;
x3902[0] = 1.0f;
float* x3904 = (float*)myMalloc(1 * sizeof(float));;
x3904[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x3902,x442,1,x3904, x1156, 1, x442,1));
arrayFill<<<64, 1>>>(x1156, 0.0f);
float* x3908 = (float*)myMalloc(1 * sizeof(float));;
x3908[0] = 1.0f;
float* x3910 = (float*)myMalloc(1 * sizeof(float));;
x3910[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3908,x445,1,x3910, x1157, 1, x445,1));
arrayFill<<<512, 1>>>(x1157, 0.0f);
float* x3914 = (float*)myMalloc(1 * sizeof(float));;
x3914[0] = 1.0f;
float* x3916 = (float*)myMalloc(1 * sizeof(float));;
x3916[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x3914,x448,1,x3916, x1158, 1, x448,1));
arrayFill<<<64, 1>>>(x1158, 0.0f);
float* x3920 = (float*)myMalloc(1 * sizeof(float));;
x3920[0] = 1.0f;
float* x3922 = (float*)myMalloc(1 * sizeof(float));;
x3922[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3920,x451,1,x3922, x1159, 1, x451,1));
arrayFill<<<512, 1>>>(x1159, 0.0f);
float* x3926 = (float*)myMalloc(1 * sizeof(float));;
x3926[0] = 1.0f;
float* x3928 = (float*)myMalloc(1 * sizeof(float));;
x3928[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x3926,x454,1,x3928, x1160, 1, x454,1));
arrayFill<<<64, 1>>>(x1160, 0.0f);
float* x3932 = (float*)myMalloc(1 * sizeof(float));;
x3932[0] = 1.0f;
float* x3934 = (float*)myMalloc(1 * sizeof(float));;
x3934[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3932,x457,1,x3934, x1161, 1, x457,1));
arrayFill<<<512, 1>>>(x1161, 0.0f);
float* x3938 = (float*)myMalloc(1 * sizeof(float));;
x3938[0] = 1.0f;
float* x3940 = (float*)myMalloc(1 * sizeof(float));;
x3940[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,512,x3938,x460,128,x3940, x1162, 128, x460,128));
arrayFill<<<65536, 1>>>(x1162, 0.0f);
float* x3944 = (float*)myMalloc(1 * sizeof(float));;
x3944[0] = 1.0f;
float* x3946 = (float*)myMalloc(1 * sizeof(float));;
x3946[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x3944,x463,256,x3946, x1163, 256, x463,256));
arrayFill<<<262144, 1>>>(x1163, 0.0f);
float* x3950 = (float*)myMalloc(1 * sizeof(float));;
x3950[0] = 1.0f;
float* x3952 = (float*)myMalloc(1 * sizeof(float));;
x3952[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x3950,x466,1,x3952, x1164, 1, x466,1));
arrayFill<<<1024, 1>>>(x1164, 0.0f);
float* x3956 = (float*)myMalloc(1 * sizeof(float));;
x3956[0] = 1.0f;
float* x3958 = (float*)myMalloc(1 * sizeof(float));;
x3958[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x3956,x469,1,x3958, x1165, 1, x469,1));
arrayFill<<<256, 1>>>(x1165, 0.0f);
float* x3962 = (float*)myMalloc(1 * sizeof(float));;
x3962[0] = 1.0f;
float* x3964 = (float*)myMalloc(1 * sizeof(float));;
x3964[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x3962,x472,1,x3964, x1166, 1, x472,1));
arrayFill<<<1024, 1>>>(x1166, 0.0f);
float* x3968 = (float*)myMalloc(1 * sizeof(float));;
x3968[0] = 1.0f;
float* x3970 = (float*)myMalloc(1 * sizeof(float));;
x3970[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x3968,x475,1,x3970, x1167, 1, x475,1));
arrayFill<<<64, 1>>>(x1167, 0.0f);
float* x3974 = (float*)myMalloc(1 * sizeof(float));;
x3974[0] = 1.0f;
float* x3976 = (float*)myMalloc(1 * sizeof(float));;
x3976[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x3974,x478,1,x3976, x1168, 1, x478,1));
arrayFill<<<128, 1>>>(x1168, 0.0f);
float* x3980 = (float*)myMalloc(1 * sizeof(float));;
x3980[0] = 1.0f;
float* x3982 = (float*)myMalloc(1 * sizeof(float));;
x3982[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x3980,x481,1,x3982, x1169, 1, x481,1));
arrayFill<<<2048, 1>>>(x1169, 0.0f);
float* x3986 = (float*)myMalloc(1 * sizeof(float));;
x3986[0] = 1.0f;
float* x3988 = (float*)myMalloc(1 * sizeof(float));;
x3988[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x3986,x484,1,x3988, x1170, 1, x484,1));
arrayFill<<<256, 1>>>(x1170, 0.0f);
float* x3992 = (float*)myMalloc(1 * sizeof(float));;
x3992[0] = 1.0f;
float* x3994 = (float*)myMalloc(1 * sizeof(float));;
x3994[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x3992,x487,1,x3994, x1171, 1, x487,1));
arrayFill<<<2048, 1>>>(x1171, 0.0f);
float* x3998 = (float*)myMalloc(1 * sizeof(float));;
x3998[0] = 1.0f;
float* x4000 = (float*)myMalloc(1 * sizeof(float));;
x4000[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x3998,x490,1,x4000, x1172, 1, x490,1));
arrayFill<<<512, 1>>>(x1172, 0.0f);
float* x4004 = (float*)myMalloc(1 * sizeof(float));;
x4004[0] = 1.0f;
float* x4006 = (float*)myMalloc(1 * sizeof(float));;
x4006[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4004,x493,1,x4006, x1173, 1, x493,1));
arrayFill<<<512, 1>>>(x1173, 0.0f);
float* x4010 = (float*)myMalloc(1 * sizeof(float));;
x4010[0] = 1.0f;
float* x4012 = (float*)myMalloc(1 * sizeof(float));;
x4012[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4010,x496,1,x4012, x1174, 1, x496,1));
arrayFill<<<512, 1>>>(x1174, 0.0f);
float* x4016 = (float*)myMalloc(1 * sizeof(float));;
x4016[0] = 1.0f;
float* x4018 = (float*)myMalloc(1 * sizeof(float));;
x4018[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4016,x499,1,x4018, x1175, 1, x499,1));
arrayFill<<<2048, 1>>>(x1175, 0.0f);
float* x4022 = (float*)myMalloc(1 * sizeof(float));;
x4022[0] = 1.0f;
float* x4024 = (float*)myMalloc(1 * sizeof(float));;
x4024[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4022,x502,1,x4024, x1176, 1, x502,1));
arrayFill<<<256, 1>>>(x1176, 0.0f);
float* x4028 = (float*)myMalloc(1 * sizeof(float));;
x4028[0] = 1.0f;
float* x4030 = (float*)myMalloc(1 * sizeof(float));;
x4030[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4028,x505,1,x4030, x1177, 1, x505,1));
arrayFill<<<256, 1>>>(x1177, 0.0f);
float* x4034 = (float*)myMalloc(1 * sizeof(float));;
x4034[0] = 1.0f;
float* x4036 = (float*)myMalloc(1 * sizeof(float));;
x4036[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4034,x508,1,x4036, x1178, 1, x508,1));
arrayFill<<<256, 1>>>(x1178, 0.0f);
float* x4040 = (float*)myMalloc(1 * sizeof(float));;
x4040[0] = 1.0f;
float* x4042 = (float*)myMalloc(1 * sizeof(float));;
x4042[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4040,x511,1,x4042, x1179, 1, x511,1));
arrayFill<<<64, 1>>>(x1179, 0.0f);
float* x4046 = (float*)myMalloc(1 * sizeof(float));;
x4046[0] = 1.0f;
float* x4048 = (float*)myMalloc(1 * sizeof(float));;
x4048[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 576,64,x4046,x514,576,x4048, x1180, 576, x514,576));
arrayFill<<<36864, 1>>>(x1180, 0.0f);
float* x4052 = (float*)myMalloc(1 * sizeof(float));;
x4052[0] = 1.0f;
float* x4054 = (float*)myMalloc(1 * sizeof(float));;
x4054[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4052,x517,1,x4054, x1181, 1, x517,1));
arrayFill<<<256, 1>>>(x1181, 0.0f);
float* x4058 = (float*)myMalloc(1 * sizeof(float));;
x4058[0] = 1.0f;
float* x4060 = (float*)myMalloc(1 * sizeof(float));;
x4060[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,1024,x4058,x520,512,x4060, x1182, 512, x520,512));
arrayFill<<<524288, 1>>>(x1182, 0.0f);
float* x4064 = (float*)myMalloc(1 * sizeof(float));;
x4064[0] = 1.0f;
float* x4066 = (float*)myMalloc(1 * sizeof(float));;
x4066[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4064,x523,1,x4066, x1183, 1, x523,1));
arrayFill<<<256, 1>>>(x1183, 0.0f);
float* x4070 = (float*)myMalloc(1 * sizeof(float));;
x4070[0] = 1.0f;
float* x4072 = (float*)myMalloc(1 * sizeof(float));;
x4072[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4070,x526,1,x4072, x1184, 1, x526,1));
arrayFill<<<256, 1>>>(x1184, 0.0f);
float* x4076 = (float*)myMalloc(1 * sizeof(float));;
x4076[0] = 1.0f;
float* x4078 = (float*)myMalloc(1 * sizeof(float));;
x4078[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4076,x529,1,x4078, x1185, 1, x529,1));
arrayFill<<<512, 1>>>(x1185, 0.0f);
float* x4082 = (float*)myMalloc(1 * sizeof(float));;
x4082[0] = 1.0f;
float* x4084 = (float*)myMalloc(1 * sizeof(float));;
x4084[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4082,x532,1,x4084, x1186, 1, x532,1));
arrayFill<<<128, 1>>>(x1186, 0.0f);
float* x4088 = (float*)myMalloc(1 * sizeof(float));;
x4088[0] = 1.0f;
float* x4090 = (float*)myMalloc(1 * sizeof(float));;
x4090[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4088,x535,1,x4090, x1187, 1, x535,1));
arrayFill<<<256, 1>>>(x1187, 0.0f);
float* x4094 = (float*)myMalloc(1 * sizeof(float));;
x4094[0] = 1.0f;
float* x4096 = (float*)myMalloc(1 * sizeof(float));;
x4096[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4094,x538,1,x4096, x1188, 1, x538,1));
arrayFill<<<64, 1>>>(x1188, 0.0f);
float* x4100 = (float*)myMalloc(1 * sizeof(float));;
x4100[0] = 1.0f;
float* x4102 = (float*)myMalloc(1 * sizeof(float));;
x4102[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4100,x541,1,x4102, x1189, 1, x541,1));
arrayFill<<<512, 1>>>(x1189, 0.0f);
float* x4106 = (float*)myMalloc(1 * sizeof(float));;
x4106[0] = 1.0f;
float* x4108 = (float*)myMalloc(1 * sizeof(float));;
x4108[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 576,64,x4106,x544,576,x4108, x1190, 576, x544,576));
arrayFill<<<36864, 1>>>(x1190, 0.0f);
float* x4112 = (float*)myMalloc(1 * sizeof(float));;
x4112[0] = 1.0f;
float* x4114 = (float*)myMalloc(1 * sizeof(float));;
x4114[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4112,x547,1,x4114, x1191, 1, x547,1));
arrayFill<<<128, 1>>>(x1191, 0.0f);
float* x4118 = (float*)myMalloc(1 * sizeof(float));;
x4118[0] = 1.0f;
float* x4120 = (float*)myMalloc(1 * sizeof(float));;
x4120[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4118,x550,1,x4120, x1192, 1, x550,1));
arrayFill<<<256, 1>>>(x1192, 0.0f);
float* x4124 = (float*)myMalloc(1 * sizeof(float));;
x4124[0] = 1.0f;
float* x4126 = (float*)myMalloc(1 * sizeof(float));;
x4126[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4124,x553,1,x4126, x1193, 1, x553,1));
arrayFill<<<1024, 1>>>(x1193, 0.0f);
float* x4130 = (float*)myMalloc(1 * sizeof(float));;
x4130[0] = 1.0f;
float* x4132 = (float*)myMalloc(1 * sizeof(float));;
x4132[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x4130,x556,64,x4132, x1194, 64, x556,64));
arrayFill<<<16384, 1>>>(x1194, 0.0f);
float* x4136 = (float*)myMalloc(1 * sizeof(float));;
x4136[0] = 1.0f;
float* x4138 = (float*)myMalloc(1 * sizeof(float));;
x4138[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4136,x559,1,x4138, x1195, 1, x559,1));
arrayFill<<<512, 1>>>(x1195, 0.0f);
float* x4142 = (float*)myMalloc(1 * sizeof(float));;
x4142[0] = 1.0f;
float* x4144 = (float*)myMalloc(1 * sizeof(float));;
x4144[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x4142,x562,256,x4144, x1196, 256, x562,256));
arrayFill<<<262144, 1>>>(x1196, 0.0f);
float* x4148 = (float*)myMalloc(1 * sizeof(float));;
x4148[0] = 1.0f;
float* x4150 = (float*)myMalloc(1 * sizeof(float));;
x4150[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 576,64,x4148,x565,576,x4150, x1197, 576, x565,576));
arrayFill<<<36864, 1>>>(x1197, 0.0f);
float* x4154 = (float*)myMalloc(1 * sizeof(float));;
x4154[0] = 1.0f;
float* x4156 = (float*)myMalloc(1 * sizeof(float));;
x4156[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4154,x568,1,x4156, x1198, 1, x568,1));
arrayFill<<<256, 1>>>(x1198, 0.0f);
float* x4160 = (float*)myMalloc(1 * sizeof(float));;
x4160[0] = 1.0f;
float* x4162 = (float*)myMalloc(1 * sizeof(float));;
x4162[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4160,x571,1,x4162, x1199, 1, x571,1));
arrayFill<<<256, 1>>>(x1199, 0.0f);
float* x4166 = (float*)myMalloc(1 * sizeof(float));;
x4166[0] = 1.0f;
float* x4168 = (float*)myMalloc(1 * sizeof(float));;
x4168[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4166,x574,1,x4168, x1200, 1, x574,1));
arrayFill<<<1024, 1>>>(x1200, 0.0f);
float* x4172 = (float*)myMalloc(1 * sizeof(float));;
x4172[0] = 1.0f;
float* x4174 = (float*)myMalloc(1 * sizeof(float));;
x4174[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4172,x577,1,x4174, x1201, 1, x577,1));
arrayFill<<<2048, 1>>>(x1201, 0.0f);
float* x4178 = (float*)myMalloc(1 * sizeof(float));;
x4178[0] = 1.0f;
float* x4180 = (float*)myMalloc(1 * sizeof(float));;
x4180[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4178,x580,1,x4180, x1202, 1, x580,1));
arrayFill<<<128, 1>>>(x1202, 0.0f);
float* x4184 = (float*)myMalloc(1 * sizeof(float));;
x4184[0] = 1.0f;
float* x4186 = (float*)myMalloc(1 * sizeof(float));;
x4186[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4184,x583,1,x4186, x1203, 1, x583,1));
arrayFill<<<256, 1>>>(x1203, 0.0f);
float* x4190 = (float*)myMalloc(1 * sizeof(float));;
x4190[0] = 1.0f;
float* x4192 = (float*)myMalloc(1 * sizeof(float));;
x4192[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x4190,x586,256,x4192, x1204, 256, x586,256));
arrayFill<<<262144, 1>>>(x1204, 0.0f);
float* x4196 = (float*)myMalloc(1 * sizeof(float));;
x4196[0] = 1.0f;
float* x4198 = (float*)myMalloc(1 * sizeof(float));;
x4198[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4196,x589,1,x4198, x1205, 1, x589,1));
arrayFill<<<256, 1>>>(x1205, 0.0f);
float* x4202 = (float*)myMalloc(1 * sizeof(float));;
x4202[0] = 1.0f;
float* x4204 = (float*)myMalloc(1 * sizeof(float));;
x4204[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4202,x592,1,x4204, x1206, 1, x592,1));
arrayFill<<<256, 1>>>(x1206, 0.0f);
float* x4208 = (float*)myMalloc(1 * sizeof(float));;
x4208[0] = 1.0f;
float* x4210 = (float*)myMalloc(1 * sizeof(float));;
x4210[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4208,x595,1,x4210, x1207, 1, x595,1));
arrayFill<<<128, 1>>>(x1207, 0.0f);
float* x4214 = (float*)myMalloc(1 * sizeof(float));;
x4214[0] = 1.0f;
float* x4216 = (float*)myMalloc(1 * sizeof(float));;
x4216[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4214,x598,1,x4216, x1208, 1, x598,1));
arrayFill<<<512, 1>>>(x1208, 0.0f);
float* x4220 = (float*)myMalloc(1 * sizeof(float));;
x4220[0] = 1.0f;
float* x4222 = (float*)myMalloc(1 * sizeof(float));;
x4222[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4220,x601,1,x4222, x1209, 1, x601,1));
arrayFill<<<64, 1>>>(x1209, 0.0f);
float* x4226 = (float*)myMalloc(1 * sizeof(float));;
x4226[0] = 1.0f;
float* x4228 = (float*)myMalloc(1 * sizeof(float));;
x4228[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4226,x604,1,x4228, x1210, 1, x604,1));
arrayFill<<<2048, 1>>>(x1210, 0.0f);
float* x4232 = (float*)myMalloc(1 * sizeof(float));;
x4232[0] = 1.0f;
float* x4234 = (float*)myMalloc(1 * sizeof(float));;
x4234[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4232,x607,1,x4234, x1211, 1, x607,1));
arrayFill<<<256, 1>>>(x1211, 0.0f);
float* x4238 = (float*)myMalloc(1 * sizeof(float));;
x4238[0] = 1.0f;
float* x4240 = (float*)myMalloc(1 * sizeof(float));;
x4240[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4238,x610,1,x4240, x1212, 1, x610,1));
arrayFill<<<64, 1>>>(x1212, 0.0f);
float* x4244 = (float*)myMalloc(1 * sizeof(float));;
x4244[0] = 1.0f;
float* x4246 = (float*)myMalloc(1 * sizeof(float));;
x4246[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,512,x4244,x613,128,x4246, x1213, 128, x613,128));
arrayFill<<<65536, 1>>>(x1213, 0.0f);
float* x4250 = (float*)myMalloc(1 * sizeof(float));;
x4250[0] = 1.0f;
float* x4252 = (float*)myMalloc(1 * sizeof(float));;
x4252[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4250,x616,1,x4252, x1214, 1, x616,1));
arrayFill<<<2048, 1>>>(x1214, 0.0f);
float* x4256 = (float*)myMalloc(1 * sizeof(float));;
x4256[0] = 1.0f;
float* x4258 = (float*)myMalloc(1 * sizeof(float));;
x4258[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4256,x619,1,x4258, x1215, 1, x619,1));
arrayFill<<<256, 1>>>(x1215, 0.0f);
float* x4262 = (float*)myMalloc(1 * sizeof(float));;
x4262[0] = 1.0f;
float* x4264 = (float*)myMalloc(1 * sizeof(float));;
x4264[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4262,x622,1,x4264, x1216, 1, x622,1));
arrayFill<<<256, 1>>>(x1216, 0.0f);
float* x4268 = (float*)myMalloc(1 * sizeof(float));;
x4268[0] = 1.0f;
float* x4270 = (float*)myMalloc(1 * sizeof(float));;
x4270[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4268,x625,1,x4270, x1217, 1, x625,1));
arrayFill<<<64, 1>>>(x1217, 0.0f);
float* x4274 = (float*)myMalloc(1 * sizeof(float));;
x4274[0] = 1.0f;
float* x4276 = (float*)myMalloc(1 * sizeof(float));;
x4276[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,512,x4274,x628,128,x4276, x1218, 128, x628,128));
arrayFill<<<65536, 1>>>(x1218, 0.0f);
float* x4280 = (float*)myMalloc(1 * sizeof(float));;
x4280[0] = 1.0f;
float* x4282 = (float*)myMalloc(1 * sizeof(float));;
x4282[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4280,x631,1,x4282, x1219, 1, x631,1));
arrayFill<<<128, 1>>>(x1219, 0.0f);
float* x4286 = (float*)myMalloc(1 * sizeof(float));;
x4286[0] = 1.0f;
float* x4288 = (float*)myMalloc(1 * sizeof(float));;
x4288[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4286,x634,1,x4288, x1220, 1, x634,1));
arrayFill<<<512, 1>>>(x1220, 0.0f);
float* x4292 = (float*)myMalloc(1 * sizeof(float));;
x4292[0] = 1.0f;
float* x4294 = (float*)myMalloc(1 * sizeof(float));;
x4294[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4292,x637,1,x4294, x1221, 1, x637,1));
arrayFill<<<64, 1>>>(x1221, 0.0f);
float* x4298 = (float*)myMalloc(1 * sizeof(float));;
x4298[0] = 1.0f;
float* x4300 = (float*)myMalloc(1 * sizeof(float));;
x4300[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4298,x640,1,x4300, x1222, 1, x640,1));
arrayFill<<<2048, 1>>>(x1222, 0.0f);
float* x4304 = (float*)myMalloc(1 * sizeof(float));;
x4304[0] = 1.0f;
float* x4306 = (float*)myMalloc(1 * sizeof(float));;
x4306[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x4304,x643,256,x4306, x1223, 256, x643,256));
arrayFill<<<262144, 1>>>(x1223, 0.0f);
float* x4310 = (float*)myMalloc(1 * sizeof(float));;
x4310[0] = 1.0f;
float* x4312 = (float*)myMalloc(1 * sizeof(float));;
x4312[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4310,x646,1,x4312, x1224, 1, x646,1));
arrayFill<<<1024, 1>>>(x1224, 0.0f);
float* x4316 = (float*)myMalloc(1 * sizeof(float));;
x4316[0] = 1.0f;
float* x4318 = (float*)myMalloc(1 * sizeof(float));;
x4318[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4316,x649,1,x4318, x1225, 1, x649,1));
arrayFill<<<64, 1>>>(x1225, 0.0f);
float* x4322 = (float*)myMalloc(1 * sizeof(float));;
x4322[0] = 1.0f;
float* x4324 = (float*)myMalloc(1 * sizeof(float));;
x4324[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4322,x652,1,x4324, x1226, 1, x652,1));
arrayFill<<<512, 1>>>(x1226, 0.0f);
float* x4328 = (float*)myMalloc(1 * sizeof(float));;
x4328[0] = 1.0f;
float* x4330 = (float*)myMalloc(1 * sizeof(float));;
x4330[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4328,x655,1,x4330, x1227, 1, x655,1));
arrayFill<<<1024, 1>>>(x1227, 0.0f);
float* x4334 = (float*)myMalloc(1 * sizeof(float));;
x4334[0] = 1.0f;
float* x4336 = (float*)myMalloc(1 * sizeof(float));;
x4336[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4334,x658,1,x4336, x1228, 1, x658,1));
arrayFill<<<512, 1>>>(x1228, 0.0f);
float* x4340 = (float*)myMalloc(1 * sizeof(float));;
x4340[0] = 1.0f;
float* x4342 = (float*)myMalloc(1 * sizeof(float));;
x4342[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4340,x661,1,x4342, x1229, 1, x661,1));
arrayFill<<<1024, 1>>>(x1229, 0.0f);
float* x4346 = (float*)myMalloc(1 * sizeof(float));;
x4346[0] = 1.0f;
float* x4348 = (float*)myMalloc(1 * sizeof(float));;
x4348[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4346,x664,1,x4348, x1230, 1, x664,1));
arrayFill<<<2048, 1>>>(x1230, 0.0f);
float* x4352 = (float*)myMalloc(1 * sizeof(float));;
x4352[0] = 1.0f;
float* x4354 = (float*)myMalloc(1 * sizeof(float));;
x4354[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4352,x667,1,x4354, x1231, 1, x667,1));
arrayFill<<<256, 1>>>(x1231, 0.0f);
float* x4358 = (float*)myMalloc(1 * sizeof(float));;
x4358[0] = 1.0f;
float* x4360 = (float*)myMalloc(1 * sizeof(float));;
x4360[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4358,x670,1,x4360, x1232, 1, x670,1));
arrayFill<<<2048, 1>>>(x1232, 0.0f);
float* x4364 = (float*)myMalloc(1 * sizeof(float));;
x4364[0] = 1.0f;
float* x4366 = (float*)myMalloc(1 * sizeof(float));;
x4366[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4364,x673,1,x4366, x1233, 1, x673,1));
arrayFill<<<256, 1>>>(x1233, 0.0f);
float* x4370 = (float*)myMalloc(1 * sizeof(float));;
x4370[0] = 1.0f;
float* x4372 = (float*)myMalloc(1 * sizeof(float));;
x4372[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4370,x676,1,x4372, x1234, 1, x676,1));
arrayFill<<<128, 1>>>(x1234, 0.0f);
float* x4376 = (float*)myMalloc(1 * sizeof(float));;
x4376[0] = 1.0f;
float* x4378 = (float*)myMalloc(1 * sizeof(float));;
x4378[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4376,x679,1,x4378, x1235, 1, x679,1));
arrayFill<<<128, 1>>>(x1235, 0.0f);
float* x4382 = (float*)myMalloc(1 * sizeof(float));;
x4382[0] = 1.0f;
float* x4384 = (float*)myMalloc(1 * sizeof(float));;
x4384[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4382,x682,1,x4384, x1236, 1, x682,1));
arrayFill<<<256, 1>>>(x1236, 0.0f);
float* x4388 = (float*)myMalloc(1 * sizeof(float));;
x4388[0] = 1.0f;
float* x4390 = (float*)myMalloc(1 * sizeof(float));;
x4390[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x4388,x685,64,x4390, x1237, 64, x685,64));
arrayFill<<<16384, 1>>>(x1237, 0.0f);
float* x4394 = (float*)myMalloc(1 * sizeof(float));;
x4394[0] = 1.0f;
float* x4396 = (float*)myMalloc(1 * sizeof(float));;
x4396[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4394,x688,1,x4396, x1238, 1, x688,1));
arrayFill<<<256, 1>>>(x1238, 0.0f);
float* x4400 = (float*)myMalloc(1 * sizeof(float));;
x4400[0] = 1.0f;
float* x4402 = (float*)myMalloc(1 * sizeof(float));;
x4402[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,128,x4400,x691,512,x4402, x1239, 512, x691,512));
arrayFill<<<65536, 1>>>(x1239, 0.0f);
float* x4406 = (float*)myMalloc(1 * sizeof(float));;
x4406[0] = 1.0f;
float* x4408 = (float*)myMalloc(1 * sizeof(float));;
x4408[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4406,x694,1,x4408, x1240, 1, x694,1));
arrayFill<<<256, 1>>>(x1240, 0.0f);
float* x4412 = (float*)myMalloc(1 * sizeof(float));;
x4412[0] = 1.0f;
float* x4414 = (float*)myMalloc(1 * sizeof(float));;
x4414[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4412,x697,1,x4414, x1241, 1, x697,1));
arrayFill<<<128, 1>>>(x1241, 0.0f);
float* x4418 = (float*)myMalloc(1 * sizeof(float));;
x4418[0] = 1.0f;
float* x4420 = (float*)myMalloc(1 * sizeof(float));;
x4420[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4418,x700,1,x4420, x1242, 1, x700,1));
arrayFill<<<64, 1>>>(x1242, 0.0f);
float* x4424 = (float*)myMalloc(1 * sizeof(float));;
x4424[0] = 1.0f;
float* x4426 = (float*)myMalloc(1 * sizeof(float));;
x4426[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4424,x703,1,x4426, x1243, 1, x703,1));
arrayFill<<<256, 1>>>(x1243, 0.0f);
float* x4430 = (float*)myMalloc(1 * sizeof(float));;
x4430[0] = 1.0f;
float* x4432 = (float*)myMalloc(1 * sizeof(float));;
x4432[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4430,x706,1,x4432, x1244, 1, x706,1));
arrayFill<<<512, 1>>>(x1244, 0.0f);
float* x4436 = (float*)myMalloc(1 * sizeof(float));;
x4436[0] = 1.0f;
float* x4438 = (float*)myMalloc(1 * sizeof(float));;
x4438[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4436,x709,1,x4438, x1245, 1, x709,1));
arrayFill<<<512, 1>>>(x1245, 0.0f);
float* x4442 = (float*)myMalloc(1 * sizeof(float));;
x4442[0] = 1.0f;
float* x4444 = (float*)myMalloc(1 * sizeof(float));;
x4444[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,512,x4442,x712,1024,x4444, x1246, 1024, x712,1024));
arrayFill<<<524288, 1>>>(x1246, 0.0f);
float* x4448 = (float*)myMalloc(1 * sizeof(float));;
x4448[0] = 1.0f;
float* x4450 = (float*)myMalloc(1 * sizeof(float));;
x4450[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4448,x715,1,x4450, x1247, 1, x715,1));
arrayFill<<<1024, 1>>>(x1247, 0.0f);
float* x4454 = (float*)myMalloc(1 * sizeof(float));;
x4454[0] = 1.0f;
float* x4456 = (float*)myMalloc(1 * sizeof(float));;
x4456[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4454,x718,1,x4456, x1248, 1, x718,1));
arrayFill<<<256, 1>>>(x1248, 0.0f);
float* x4460 = (float*)myMalloc(1 * sizeof(float));;
x4460[0] = 1.0f;
float* x4462 = (float*)myMalloc(1 * sizeof(float));;
x4462[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4460,x721,1,x4462, x1249, 1, x721,1));
arrayFill<<<64, 1>>>(x1249, 0.0f);
float* x4466 = (float*)myMalloc(1 * sizeof(float));;
x4466[0] = 1.0f;
float* x4468 = (float*)myMalloc(1 * sizeof(float));;
x4468[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4466,x724,1,x4468, x1250, 1, x724,1));
arrayFill<<<1024, 1>>>(x1250, 0.0f);
float* x4472 = (float*)myMalloc(1 * sizeof(float));;
x4472[0] = 1.0f;
float* x4474 = (float*)myMalloc(1 * sizeof(float));;
x4474[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4472,x727,1,x4474, x1251, 1, x727,1));
arrayFill<<<2048, 1>>>(x1251, 0.0f);
float* x4478 = (float*)myMalloc(1 * sizeof(float));;
x4478[0] = 1.0f;
float* x4480 = (float*)myMalloc(1 * sizeof(float));;
x4480[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4478,x730,1,x4480, x1252, 1, x730,1));
arrayFill<<<512, 1>>>(x1252, 0.0f);
float* x4484 = (float*)myMalloc(1 * sizeof(float));;
x4484[0] = 1.0f;
float* x4486 = (float*)myMalloc(1 * sizeof(float));;
x4486[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4484,x733,1,x4486, x1253, 1, x733,1));
arrayFill<<<1024, 1>>>(x1253, 0.0f);
float* x4490 = (float*)myMalloc(1 * sizeof(float));;
x4490[0] = 1.0f;
float* x4492 = (float*)myMalloc(1 * sizeof(float));;
x4492[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4490,x736,1,x4492, x1254, 1, x736,1));
arrayFill<<<512, 1>>>(x1254, 0.0f);
float* x4496 = (float*)myMalloc(1 * sizeof(float));;
x4496[0] = 1.0f;
float* x4498 = (float*)myMalloc(1 * sizeof(float));;
x4498[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4496,x739,1,x4498, x1255, 1, x739,1));
arrayFill<<<128, 1>>>(x1255, 0.0f);
float* x4502 = (float*)myMalloc(1 * sizeof(float));;
x4502[0] = 1.0f;
float* x4504 = (float*)myMalloc(1 * sizeof(float));;
x4504[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4502,x742,1,x4504, x1256, 1, x742,1));
arrayFill<<<512, 1>>>(x1256, 0.0f);
float* x4508 = (float*)myMalloc(1 * sizeof(float));;
x4508[0] = 1.0f;
float* x4510 = (float*)myMalloc(1 * sizeof(float));;
x4510[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,64,x4508,x745,256,x4510, x1257, 256, x745,256));
arrayFill<<<16384, 1>>>(x1257, 0.0f);
float* x4514 = (float*)myMalloc(1 * sizeof(float));;
x4514[0] = 1.0f;
float* x4516 = (float*)myMalloc(1 * sizeof(float));;
x4516[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,256,x4514,x748,1024,x4516, x1258, 1024, x748,1024));
arrayFill<<<262144, 1>>>(x1258, 0.0f);
float* x4520 = (float*)myMalloc(1 * sizeof(float));;
x4520[0] = 1.0f;
float* x4522 = (float*)myMalloc(1 * sizeof(float));;
x4522[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 27,64,x4520,x751,27,x4522, x1259, 27, x751,27));
arrayFill<<<1728, 1>>>(x1259, 0.0f);
float* x4526 = (float*)myMalloc(1 * sizeof(float));;
x4526[0] = 1.0f;
float* x4528 = (float*)myMalloc(1 * sizeof(float));;
x4528[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4526,x754,1,x4528, x1260, 1, x754,1));
arrayFill<<<64, 1>>>(x1260, 0.0f);
float* x4532 = (float*)myMalloc(1 * sizeof(float));;
x4532[0] = 1.0f;
float* x4534 = (float*)myMalloc(1 * sizeof(float));;
x4534[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4532,x757,1,x4534, x1261, 1, x757,1));
arrayFill<<<512, 1>>>(x1261, 0.0f);
float* x4538 = (float*)myMalloc(1 * sizeof(float));;
x4538[0] = 1.0f;
float* x4540 = (float*)myMalloc(1 * sizeof(float));;
x4540[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 4608,512,x4538,x760,4608,x4540, x1262, 4608, x760,4608));
arrayFill<<<2359296, 1>>>(x1262, 0.0f);
float* x4544 = (float*)myMalloc(1 * sizeof(float));;
x4544[0] = 1.0f;
float* x4546 = (float*)myMalloc(1 * sizeof(float));;
x4546[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4544,x763,1,x4546, x1263, 1, x763,1));
arrayFill<<<512, 1>>>(x1263, 0.0f);
float* x4550 = (float*)myMalloc(1 * sizeof(float));;
x4550[0] = 1.0f;
float* x4552 = (float*)myMalloc(1 * sizeof(float));;
x4552[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4550,x766,1,x4552, x1264, 1, x766,1));
arrayFill<<<256, 1>>>(x1264, 0.0f);
float* x4556 = (float*)myMalloc(1 * sizeof(float));;
x4556[0] = 1.0f;
float* x4558 = (float*)myMalloc(1 * sizeof(float));;
x4558[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4556,x769,1,x4558, x1265, 1, x769,1));
arrayFill<<<64, 1>>>(x1265, 0.0f);
float* x4562 = (float*)myMalloc(1 * sizeof(float));;
x4562[0] = 1.0f;
float* x4564 = (float*)myMalloc(1 * sizeof(float));;
x4564[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4562,x772,1,x4564, x1266, 1, x772,1));
arrayFill<<<512, 1>>>(x1266, 0.0f);
float* x4568 = (float*)myMalloc(1 * sizeof(float));;
x4568[0] = 1.0f;
float* x4570 = (float*)myMalloc(1 * sizeof(float));;
x4570[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4568,x775,1,x4570, x1267, 1, x775,1));
arrayFill<<<512, 1>>>(x1267, 0.0f);
float* x4574 = (float*)myMalloc(1 * sizeof(float));;
x4574[0] = 1.0f;
float* x4576 = (float*)myMalloc(1 * sizeof(float));;
x4576[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4574,x778,1,x4576, x1268, 1, x778,1));
arrayFill<<<1024, 1>>>(x1268, 0.0f);
float* x4580 = (float*)myMalloc(1 * sizeof(float));;
x4580[0] = 1.0f;
float* x4582 = (float*)myMalloc(1 * sizeof(float));;
x4582[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x4580,x781,64,x4582, x1269, 64, x781,64));
arrayFill<<<16384, 1>>>(x1269, 0.0f);
float* x4586 = (float*)myMalloc(1 * sizeof(float));;
x4586[0] = 1.0f;
float* x4588 = (float*)myMalloc(1 * sizeof(float));;
x4588[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4586,x784,1,x4588, x1270, 1, x784,1));
arrayFill<<<256, 1>>>(x1270, 0.0f);
float* x4592 = (float*)myMalloc(1 * sizeof(float));;
x4592[0] = 1.0f;
float* x4594 = (float*)myMalloc(1 * sizeof(float));;
x4594[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4592,x787,1,x4594, x1271, 1, x787,1));
arrayFill<<<64, 1>>>(x1271, 0.0f);
float* x4598 = (float*)myMalloc(1 * sizeof(float));;
x4598[0] = 1.0f;
float* x4600 = (float*)myMalloc(1 * sizeof(float));;
x4600[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1152,128,x4598,x790,1152,x4600, x1272, 1152, x790,1152));
arrayFill<<<147456, 1>>>(x1272, 0.0f);
float* x4604 = (float*)myMalloc(1 * sizeof(float));;
x4604[0] = 1.0f;
float* x4606 = (float*)myMalloc(1 * sizeof(float));;
x4606[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4604,x793,1,x4606, x1273, 1, x793,1));
arrayFill<<<256, 1>>>(x1273, 0.0f);
float* x4610 = (float*)myMalloc(1 * sizeof(float));;
x4610[0] = 1.0f;
float* x4612 = (float*)myMalloc(1 * sizeof(float));;
x4612[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4610,x796,1,x4612, x1274, 1, x796,1));
arrayFill<<<512, 1>>>(x1274, 0.0f);
float* x4616 = (float*)myMalloc(1 * sizeof(float));;
x4616[0] = 1.0f;
float* x4618 = (float*)myMalloc(1 * sizeof(float));;
x4618[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4616,x799,1,x4618, x1275, 1, x799,1));
arrayFill<<<256, 1>>>(x1275, 0.0f);
float* x4622 = (float*)myMalloc(1 * sizeof(float));;
x4622[0] = 1.0f;
float* x4624 = (float*)myMalloc(1 * sizeof(float));;
x4624[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4622,x802,1,x4624, x1276, 1, x802,1));
arrayFill<<<512, 1>>>(x1276, 0.0f);
float* x4628 = (float*)myMalloc(1 * sizeof(float));;
x4628[0] = 1.0f;
float* x4630 = (float*)myMalloc(1 * sizeof(float));;
x4630[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4628,x805,1,x4630, x1277, 1, x805,1));
arrayFill<<<128, 1>>>(x1277, 0.0f);
float* x4634 = (float*)myMalloc(1 * sizeof(float));;
x4634[0] = 1.0f;
float* x4636 = (float*)myMalloc(1 * sizeof(float));;
x4636[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,64,x4634,x808,256,x4636, x1278, 256, x808,256));
arrayFill<<<16384, 1>>>(x1278, 0.0f);
float* x4640 = (float*)myMalloc(1 * sizeof(float));;
x4640[0] = 1.0f;
float* x4642 = (float*)myMalloc(1 * sizeof(float));;
x4642[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4640,x811,1,x4642, x1279, 1, x811,1));
arrayFill<<<128, 1>>>(x1279, 0.0f);
float* x4646 = (float*)myMalloc(1 * sizeof(float));;
x4646[0] = 1.0f;
float* x4648 = (float*)myMalloc(1 * sizeof(float));;
x4648[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4646,x814,1,x4648, x1280, 1, x814,1));
arrayFill<<<2048, 1>>>(x1280, 0.0f);
float* x4652 = (float*)myMalloc(1 * sizeof(float));;
x4652[0] = 1.0f;
float* x4654 = (float*)myMalloc(1 * sizeof(float));;
x4654[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4652,x817,1,x4654, x1281, 1, x817,1));
arrayFill<<<256, 1>>>(x1281, 0.0f);
float* x4658 = (float*)myMalloc(1 * sizeof(float));;
x4658[0] = 1.0f;
float* x4660 = (float*)myMalloc(1 * sizeof(float));;
x4660[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x4658,x820,2304,x4660, x1282, 2304, x820,2304));
arrayFill<<<589824, 1>>>(x1282, 0.0f);
float* x4664 = (float*)myMalloc(1 * sizeof(float));;
x4664[0] = 1.0f;
float* x4666 = (float*)myMalloc(1 * sizeof(float));;
x4666[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4664,x823,1,x4666, x1283, 1, x823,1));
arrayFill<<<256, 1>>>(x1283, 0.0f);
float* x4670 = (float*)myMalloc(1 * sizeof(float));;
x4670[0] = 1.0f;
float* x4672 = (float*)myMalloc(1 * sizeof(float));;
x4672[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4670,x826,1,x4672, x1284, 1, x826,1));
arrayFill<<<128, 1>>>(x1284, 0.0f);
float* x4676 = (float*)myMalloc(1 * sizeof(float));;
x4676[0] = 1.0f;
float* x4678 = (float*)myMalloc(1 * sizeof(float));;
x4678[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4676,x829,1,x4678, x1285, 1, x829,1));
arrayFill<<<256, 1>>>(x1285, 0.0f);
float* x4682 = (float*)myMalloc(1 * sizeof(float));;
x4682[0] = 1.0f;
float* x4684 = (float*)myMalloc(1 * sizeof(float));;
x4684[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4682,x832,1,x4684, x1286, 1, x832,1));
arrayFill<<<64, 1>>>(x1286, 0.0f);
float* x4688 = (float*)myMalloc(1 * sizeof(float));;
x4688[0] = 1.0f;
float* x4690 = (float*)myMalloc(1 * sizeof(float));;
x4690[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,256,x4688,x835,512,x4690, x1287, 512, x835,512));
arrayFill<<<131072, 1>>>(x1287, 0.0f);
float* x4694 = (float*)myMalloc(1 * sizeof(float));;
x4694[0] = 1.0f;
float* x4696 = (float*)myMalloc(1 * sizeof(float));;
x4696[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4694,x838,1,x4696, x1288, 1, x838,1));
arrayFill<<<2048, 1>>>(x1288, 0.0f);
float* x4700 = (float*)myMalloc(1 * sizeof(float));;
x4700[0] = 1.0f;
float* x4702 = (float*)myMalloc(1 * sizeof(float));;
x4702[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4700,x841,1,x4702, x1289, 1, x841,1));
arrayFill<<<1024, 1>>>(x1289, 0.0f);
float* x4706 = (float*)myMalloc(1 * sizeof(float));;
x4706[0] = 1.0f;
float* x4708 = (float*)myMalloc(1 * sizeof(float));;
x4708[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4706,x844,1,x4708, x1290, 1, x844,1));
arrayFill<<<1024, 1>>>(x1290, 0.0f);
float* x4712 = (float*)myMalloc(1 * sizeof(float));;
x4712[0] = 1.0f;
float* x4714 = (float*)myMalloc(1 * sizeof(float));;
x4714[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4712,x847,1,x4714, x1291, 1, x847,1));
arrayFill<<<256, 1>>>(x1291, 0.0f);
float* x4718 = (float*)myMalloc(1 * sizeof(float));;
x4718[0] = 1.0f;
float* x4720 = (float*)myMalloc(1 * sizeof(float));;
x4720[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4718,x850,1,x4720, x1292, 1, x850,1));
arrayFill<<<256, 1>>>(x1292, 0.0f);
float* x4724 = (float*)myMalloc(1 * sizeof(float));;
x4724[0] = 1.0f;
float* x4726 = (float*)myMalloc(1 * sizeof(float));;
x4726[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4724,x853,1,x4726, x1293, 1, x853,1));
arrayFill<<<256, 1>>>(x1293, 0.0f);
float* x4730 = (float*)myMalloc(1 * sizeof(float));;
x4730[0] = 1.0f;
float* x4732 = (float*)myMalloc(1 * sizeof(float));;
x4732[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4730,x856,1,x4732, x1294, 1, x856,1));
arrayFill<<<64, 1>>>(x1294, 0.0f);
float* x4736 = (float*)myMalloc(1 * sizeof(float));;
x4736[0] = 1.0f;
float* x4738 = (float*)myMalloc(1 * sizeof(float));;
x4738[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4736,x859,1,x4738, x1295, 1, x859,1));
arrayFill<<<1024, 1>>>(x1295, 0.0f);
float* x4742 = (float*)myMalloc(1 * sizeof(float));;
x4742[0] = 1.0f;
float* x4744 = (float*)myMalloc(1 * sizeof(float));;
x4744[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4742,x862,1,x4744, x1296, 1, x862,1));
arrayFill<<<256, 1>>>(x1296, 0.0f);
float* x4748 = (float*)myMalloc(1 * sizeof(float));;
x4748[0] = 1.0f;
float* x4750 = (float*)myMalloc(1 * sizeof(float));;
x4750[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4748,x865,1,x4750, x1297, 1, x865,1));
arrayFill<<<128, 1>>>(x1297, 0.0f);
float* x4754 = (float*)myMalloc(1 * sizeof(float));;
x4754[0] = 1.0f;
float* x4756 = (float*)myMalloc(1 * sizeof(float));;
x4756[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1152,128,x4754,x868,1152,x4756, x1298, 1152, x868,1152));
arrayFill<<<147456, 1>>>(x1298, 0.0f);
float* x4760 = (float*)myMalloc(1 * sizeof(float));;
x4760[0] = 1.0f;
float* x4762 = (float*)myMalloc(1 * sizeof(float));;
x4762[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4760,x871,1,x4762, x1299, 1, x871,1));
arrayFill<<<256, 1>>>(x1299, 0.0f);
float* x4766 = (float*)myMalloc(1 * sizeof(float));;
x4766[0] = 1.0f;
float* x4768 = (float*)myMalloc(1 * sizeof(float));;
x4768[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x4766,x874,1,x4768, x1300, 1, x874,1));
arrayFill<<<2048, 1>>>(x1300, 0.0f);
float* x4772 = (float*)myMalloc(1 * sizeof(float));;
x4772[0] = 1.0f;
float* x4774 = (float*)myMalloc(1 * sizeof(float));;
x4774[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4772,x877,1,x4774, x1301, 1, x877,1));
arrayFill<<<512, 1>>>(x1301, 0.0f);
float* x4778 = (float*)myMalloc(1 * sizeof(float));;
x4778[0] = 1.0f;
float* x4780 = (float*)myMalloc(1 * sizeof(float));;
x4780[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4778,x880,1,x4780, x1302, 1, x880,1));
arrayFill<<<512, 1>>>(x1302, 0.0f);
float* x4784 = (float*)myMalloc(1 * sizeof(float));;
x4784[0] = 1.0f;
float* x4786 = (float*)myMalloc(1 * sizeof(float));;
x4786[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,128,x4784,x883,512,x4786, x1303, 512, x883,512));
arrayFill<<<65536, 1>>>(x1303, 0.0f);
float* x4790 = (float*)myMalloc(1 * sizeof(float));;
x4790[0] = 1.0f;
float* x4792 = (float*)myMalloc(1 * sizeof(float));;
x4792[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4790,x886,1,x4792, x1304, 1, x886,1));
arrayFill<<<256, 1>>>(x1304, 0.0f);
float* x4796 = (float*)myMalloc(1 * sizeof(float));;
x4796[0] = 1.0f;
float* x4798 = (float*)myMalloc(1 * sizeof(float));;
x4798[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4796,x889,1,x4798, x1305, 1, x889,1));
arrayFill<<<256, 1>>>(x1305, 0.0f);
float* x4802 = (float*)myMalloc(1 * sizeof(float));;
x4802[0] = 1.0f;
float* x4804 = (float*)myMalloc(1 * sizeof(float));;
x4804[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4802,x892,1,x4804, x1306, 1, x892,1));
arrayFill<<<256, 1>>>(x1306, 0.0f);
float* x4808 = (float*)myMalloc(1 * sizeof(float));;
x4808[0] = 1.0f;
float* x4810 = (float*)myMalloc(1 * sizeof(float));;
x4810[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4808,x895,1,x4810, x1307, 1, x895,1));
arrayFill<<<256, 1>>>(x1307, 0.0f);
float* x4814 = (float*)myMalloc(1 * sizeof(float));;
x4814[0] = 1.0f;
float* x4816 = (float*)myMalloc(1 * sizeof(float));;
x4816[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4814,x898,1,x4816, x1308, 1, x898,1));
arrayFill<<<512, 1>>>(x1308, 0.0f);
float* x4820 = (float*)myMalloc(1 * sizeof(float));;
x4820[0] = 1.0f;
float* x4822 = (float*)myMalloc(1 * sizeof(float));;
x4822[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4820,x901,1,x4822, x1309, 1, x901,1));
arrayFill<<<512, 1>>>(x1309, 0.0f);
float* x4826 = (float*)myMalloc(1 * sizeof(float));;
x4826[0] = 1.0f;
float* x4828 = (float*)myMalloc(1 * sizeof(float));;
x4828[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4826,x904,1,x4828, x1310, 1, x904,1));
arrayFill<<<256, 1>>>(x1310, 0.0f);
float* x4832 = (float*)myMalloc(1 * sizeof(float));;
x4832[0] = 1.0f;
float* x4834 = (float*)myMalloc(1 * sizeof(float));;
x4834[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4832,x907,1,x4834, x1311, 1, x907,1));
arrayFill<<<128, 1>>>(x1311, 0.0f);
float* x4838 = (float*)myMalloc(1 * sizeof(float));;
x4838[0] = 1.0f;
float* x4840 = (float*)myMalloc(1 * sizeof(float));;
x4840[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4838,x910,1,x4840, x1312, 1, x910,1));
arrayFill<<<512, 1>>>(x1312, 0.0f);
float* x4844 = (float*)myMalloc(1 * sizeof(float));;
x4844[0] = 1.0f;
float* x4846 = (float*)myMalloc(1 * sizeof(float));;
x4846[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4844,x913,1,x4846, x1313, 1, x913,1));
arrayFill<<<64, 1>>>(x1313, 0.0f);
float* x4850 = (float*)myMalloc(1 * sizeof(float));;
x4850[0] = 1.0f;
float* x4852 = (float*)myMalloc(1 * sizeof(float));;
x4852[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4850,x916,1,x4852, x1314, 1, x916,1));
arrayFill<<<512, 1>>>(x1314, 0.0f);
float* x4856 = (float*)myMalloc(1 * sizeof(float));;
x4856[0] = 1.0f;
float* x4858 = (float*)myMalloc(1 * sizeof(float));;
x4858[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4856,x919,1,x4858, x1315, 1, x919,1));
arrayFill<<<64, 1>>>(x1315, 0.0f);
float* x4862 = (float*)myMalloc(1 * sizeof(float));;
x4862[0] = 1.0f;
float* x4864 = (float*)myMalloc(1 * sizeof(float));;
x4864[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4862,x922,1,x4864, x1316, 1, x922,1));
arrayFill<<<1024, 1>>>(x1316, 0.0f);
float* x4868 = (float*)myMalloc(1 * sizeof(float));;
x4868[0] = 1.0f;
float* x4870 = (float*)myMalloc(1 * sizeof(float));;
x4870[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4868,x925,1,x4870, x1317, 1, x925,1));
arrayFill<<<512, 1>>>(x1317, 0.0f);
float* x4874 = (float*)myMalloc(1 * sizeof(float));;
x4874[0] = 1.0f;
float* x4876 = (float*)myMalloc(1 * sizeof(float));;
x4876[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4874,x928,1,x4876, x1318, 1, x928,1));
arrayFill<<<1024, 1>>>(x1318, 0.0f);
float* x4880 = (float*)myMalloc(1 * sizeof(float));;
x4880[0] = 1.0f;
float* x4882 = (float*)myMalloc(1 * sizeof(float));;
x4882[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,2048,x4880,x931,512,x4882, x1319, 512, x931,512));
arrayFill<<<1048576, 1>>>(x1319, 0.0f);
float* x4886 = (float*)myMalloc(1 * sizeof(float));;
x4886[0] = 1.0f;
float* x4888 = (float*)myMalloc(1 * sizeof(float));;
x4888[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4886,x934,1,x4888, x1320, 1, x934,1));
arrayFill<<<512, 1>>>(x1320, 0.0f);
float* x4892 = (float*)myMalloc(1 * sizeof(float));;
x4892[0] = 1.0f;
float* x4894 = (float*)myMalloc(1 * sizeof(float));;
x4894[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,2048,x4892,x937,1024,x4894, x1321, 1024, x937,1024));
arrayFill<<<2097152, 1>>>(x1321, 0.0f);
float* x4898 = (float*)myMalloc(1 * sizeof(float));;
x4898[0] = 1.0f;
float* x4900 = (float*)myMalloc(1 * sizeof(float));;
x4900[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2048,512,x4898,x940,2048,x4900, x1322, 2048, x940,2048));
arrayFill<<<1048576, 1>>>(x1322, 0.0f);
float* x4904 = (float*)myMalloc(1 * sizeof(float));;
x4904[0] = 1.0f;
float* x4906 = (float*)myMalloc(1 * sizeof(float));;
x4906[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4904,x943,1,x4906, x1323, 1, x943,1));
arrayFill<<<1024, 1>>>(x1323, 0.0f);
float* x4910 = (float*)myMalloc(1 * sizeof(float));;
x4910[0] = 1.0f;
float* x4912 = (float*)myMalloc(1 * sizeof(float));;
x4912[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4910,x946,1,x4912, x1324, 1, x946,1));
arrayFill<<<128, 1>>>(x1324, 0.0f);
float* x4916 = (float*)myMalloc(1 * sizeof(float));;
x4916[0] = 1.0f;
float* x4918 = (float*)myMalloc(1 * sizeof(float));;
x4918[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,256,x4916,x949,1024,x4918, x1325, 1024, x949,1024));
arrayFill<<<262144, 1>>>(x1325, 0.0f);
float* x4922 = (float*)myMalloc(1 * sizeof(float));;
x4922[0] = 1.0f;
float* x4924 = (float*)myMalloc(1 * sizeof(float));;
x4924[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4922,x952,1,x4924, x1326, 1, x952,1));
arrayFill<<<256, 1>>>(x1326, 0.0f);
float* x4928 = (float*)myMalloc(1 * sizeof(float));;
x4928[0] = 1.0f;
float* x4930 = (float*)myMalloc(1 * sizeof(float));;
x4930[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4928,x955,1,x4930, x1327, 1, x955,1));
arrayFill<<<1024, 1>>>(x1327, 0.0f);
float* x4934 = (float*)myMalloc(1 * sizeof(float));;
x4934[0] = 1.0f;
float* x4936 = (float*)myMalloc(1 * sizeof(float));;
x4936[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x4934,x958,256,x4936, x1328, 256, x958,256));
arrayFill<<<262144, 1>>>(x1328, 0.0f);
float* x4940 = (float*)myMalloc(1 * sizeof(float));;
x4940[0] = 1.0f;
float* x4942 = (float*)myMalloc(1 * sizeof(float));;
x4942[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4940,x961,1,x4942, x1329, 1, x961,1));
arrayFill<<<128, 1>>>(x1329, 0.0f);
float* x4946 = (float*)myMalloc(1 * sizeof(float));;
x4946[0] = 1.0f;
float* x4948 = (float*)myMalloc(1 * sizeof(float));;
x4948[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4946,x964,1,x4948, x1330, 1, x964,1));
arrayFill<<<512, 1>>>(x1330, 0.0f);
float* x4952 = (float*)myMalloc(1 * sizeof(float));;
x4952[0] = 1.0f;
float* x4954 = (float*)myMalloc(1 * sizeof(float));;
x4954[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4952,x967,1,x4954, x1331, 1, x967,1));
arrayFill<<<512, 1>>>(x1331, 0.0f);
float* x4958 = (float*)myMalloc(1 * sizeof(float));;
x4958[0] = 1.0f;
float* x4960 = (float*)myMalloc(1 * sizeof(float));;
x4960[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4958,x970,1,x4960, x1332, 1, x970,1));
arrayFill<<<128, 1>>>(x1332, 0.0f);
float* x4964 = (float*)myMalloc(1 * sizeof(float));;
x4964[0] = 1.0f;
float* x4966 = (float*)myMalloc(1 * sizeof(float));;
x4966[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x4964,x973,2304,x4966, x1333, 2304, x973,2304));
arrayFill<<<589824, 1>>>(x1333, 0.0f);
float* x4970 = (float*)myMalloc(1 * sizeof(float));;
x4970[0] = 1.0f;
float* x4972 = (float*)myMalloc(1 * sizeof(float));;
x4972[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2048,10,x4970,x976,2048,x4972, x1334, 2048, x976,2048));
arrayFill<<<20480, 1>>>(x1334, 0.0f);
float* x4976 = (float*)myMalloc(1 * sizeof(float));;
x4976[0] = 1.0f;
float* x4978 = (float*)myMalloc(1 * sizeof(float));;
x4978[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4976,x979,1,x4978, x1335, 1, x979,1));
arrayFill<<<256, 1>>>(x1335, 0.0f);
float* x4982 = (float*)myMalloc(1 * sizeof(float));;
x4982[0] = 1.0f;
float* x4984 = (float*)myMalloc(1 * sizeof(float));;
x4984[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4982,x982,1,x4984, x1336, 1, x982,1));
arrayFill<<<256, 1>>>(x1336, 0.0f);
float* x4988 = (float*)myMalloc(1 * sizeof(float));;
x4988[0] = 1.0f;
float* x4990 = (float*)myMalloc(1 * sizeof(float));;
x4990[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4988,x985,1,x4990, x1337, 1, x985,1));
arrayFill<<<256, 1>>>(x1337, 0.0f);
float* x4994 = (float*)myMalloc(1 * sizeof(float));;
x4994[0] = 1.0f;
float* x4996 = (float*)myMalloc(1 * sizeof(float));;
x4996[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4994,x988,1,x4996, x1338, 1, x988,1));
arrayFill<<<1024, 1>>>(x1338, 0.0f);
float* x5000 = (float*)myMalloc(1 * sizeof(float));;
x5000[0] = 1.0f;
float* x5002 = (float*)myMalloc(1 * sizeof(float));;
x5002[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5000,x991,1,x5002, x1339, 1, x991,1));
arrayFill<<<1024, 1>>>(x1339, 0.0f);
float* x5006 = (float*)myMalloc(1 * sizeof(float));;
x5006[0] = 1.0f;
float* x5008 = (float*)myMalloc(1 * sizeof(float));;
x5008[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,64,x5006,x994,64,x5008, x1340, 64, x994,64));
arrayFill<<<4096, 1>>>(x1340, 0.0f);
float* x5012 = (float*)myMalloc(1 * sizeof(float));;
x5012[0] = 1.0f;
float* x5014 = (float*)myMalloc(1 * sizeof(float));;
x5014[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5012,x997,1,x5014, x1341, 1, x997,1));
arrayFill<<<512, 1>>>(x1341, 0.0f);
float* x5018 = (float*)myMalloc(1 * sizeof(float));;
x5018[0] = 1.0f;
float* x5020 = (float*)myMalloc(1 * sizeof(float));;
x5020[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1152,128,x5018,x1000,1152,x5020, x1342, 1152, x1000,1152));
arrayFill<<<147456, 1>>>(x1342, 0.0f);
float* x5024 = (float*)myMalloc(1 * sizeof(float));;
x5024[0] = 1.0f;
float* x5026 = (float*)myMalloc(1 * sizeof(float));;
x5026[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5024,x1003,1,x5026, x1343, 1, x1003,1));
arrayFill<<<128, 1>>>(x1343, 0.0f);
float* x5030 = (float*)myMalloc(1 * sizeof(float));;
x5030[0] = 1.0f;
float* x5032 = (float*)myMalloc(1 * sizeof(float));;
x5032[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5030,x1006,1,x5032, x1344, 1, x1006,1));
arrayFill<<<256, 1>>>(x1344, 0.0f);
float* x5036 = (float*)myMalloc(1 * sizeof(float));;
x5036[0] = 1.0f;
float* x5038 = (float*)myMalloc(1 * sizeof(float));;
x5038[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5036,x1009,1,x5038, x1345, 1, x1009,1));
arrayFill<<<1024, 1>>>(x1345, 0.0f);
float* x5042 = (float*)myMalloc(1 * sizeof(float));;
x5042[0] = 1.0f;
float* x5044 = (float*)myMalloc(1 * sizeof(float));;
x5044[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5042,x1012,1,x5044, x1346, 1, x1012,1));
arrayFill<<<2048, 1>>>(x1346, 0.0f);
float* x5048 = (float*)myMalloc(1 * sizeof(float));;
x5048[0] = 1.0f;
float* x5050 = (float*)myMalloc(1 * sizeof(float));;
x5050[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5048,x1015,1,x5050, x1347, 1, x1015,1));
arrayFill<<<256, 1>>>(x1347, 0.0f);
float* x5054 = (float*)myMalloc(1 * sizeof(float));;
x5054[0] = 1.0f;
float* x5056 = (float*)myMalloc(1 * sizeof(float));;
x5056[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5054,x1018,1,x5056, x1348, 1, x1018,1));
arrayFill<<<256, 1>>>(x1348, 0.0f);
float* x5060 = (float*)myMalloc(1 * sizeof(float));;
x5060[0] = 1.0f;
float* x5062 = (float*)myMalloc(1 * sizeof(float));;
x5062[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5060,x1021,1,x5062, x1349, 1, x1021,1));
arrayFill<<<128, 1>>>(x1349, 0.0f);
float* x5066 = (float*)myMalloc(1 * sizeof(float));;
x5066[0] = 1.0f;
float* x5068 = (float*)myMalloc(1 * sizeof(float));;
x5068[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5066,x1024,1,x5068, x1350, 1, x1024,1));
arrayFill<<<256, 1>>>(x1350, 0.0f);
float* x5072 = (float*)myMalloc(1 * sizeof(float));;
x5072[0] = 1.0f;
float* x5074 = (float*)myMalloc(1 * sizeof(float));;
x5074[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5072,x1027,1,x5074, x1351, 1, x1027,1));
arrayFill<<<64, 1>>>(x1351, 0.0f);
float* x5078 = (float*)myMalloc(1 * sizeof(float));;
x5078[0] = 1.0f;
float* x5080 = (float*)myMalloc(1 * sizeof(float));;
x5080[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5078,x1030,1,x5080, x1352, 1, x1030,1));
arrayFill<<<2048, 1>>>(x1352, 0.0f);
float* x5084 = (float*)myMalloc(1 * sizeof(float));;
x5084[0] = 1.0f;
float* x5086 = (float*)myMalloc(1 * sizeof(float));;
x5086[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5084,x1033,1,x5086, x1353, 1, x1033,1));
arrayFill<<<512, 1>>>(x1353, 0.0f);
float* x5090 = (float*)myMalloc(1 * sizeof(float));;
x5090[0] = 1.0f;
float* x5092 = (float*)myMalloc(1 * sizeof(float));;
x5092[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5090,x1036,1,x5092, x1354, 1, x1036,1));
arrayFill<<<256, 1>>>(x1354, 0.0f);
float* x5096 = (float*)myMalloc(1 * sizeof(float));;
x5096[0] = 1.0f;
float* x5098 = (float*)myMalloc(1 * sizeof(float));;
x5098[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5096,x1039,1,x5098, x1355, 1, x1039,1));
arrayFill<<<1024, 1>>>(x1355, 0.0f);
float* x5102 = (float*)myMalloc(1 * sizeof(float));;
x5102[0] = 1.0f;
float* x5104 = (float*)myMalloc(1 * sizeof(float));;
x5104[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x5102,x1042,2304,x5104, x1356, 2304, x1042,2304));
arrayFill<<<589824, 1>>>(x1356, 0.0f);
float* x5108 = (float*)myMalloc(1 * sizeof(float));;
x5108[0] = 1.0f;
float* x5110 = (float*)myMalloc(1 * sizeof(float));;
x5110[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5108,x1045,1,x5110, x1357, 1, x1045,1));
arrayFill<<<256, 1>>>(x1357, 0.0f);
float* x5114 = (float*)myMalloc(1 * sizeof(float));;
x5114[0] = 1.0f;
float* x5116 = (float*)myMalloc(1 * sizeof(float));;
x5116[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5114,x1048,1,x5116, x1358, 1, x1048,1));
arrayFill<<<64, 1>>>(x1358, 0.0f);
float* x5120 = (float*)myMalloc(1 * sizeof(float));;
x5120[0] = 1.0f;
float* x5122 = (float*)myMalloc(1 * sizeof(float));;
x5122[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5120,x1051,1,x5122, x1359, 1, x1051,1));
arrayFill<<<128, 1>>>(x1359, 0.0f);
float* x5126 = (float*)myMalloc(1 * sizeof(float));;
x5126[0] = 1.0f;
float* x5128 = (float*)myMalloc(1 * sizeof(float));;
x5128[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5126,x1054,1,x5128, x1360, 1, x1054,1));
arrayFill<<<256, 1>>>(x1360, 0.0f);
float* x5132 = (float*)myMalloc(1 * sizeof(float));;
x5132[0] = 1.0f;
float* x5134 = (float*)myMalloc(1 * sizeof(float));;
x5134[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5132,x1057,1,x5134, x1361, 1, x1057,1));
arrayFill<<<256, 1>>>(x1361, 0.0f);
float* x5138 = (float*)myMalloc(1 * sizeof(float));;
x5138[0] = 1.0f;
float* x5140 = (float*)myMalloc(1 * sizeof(float));;
x5140[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5138,x1060,1,x5140, x1362, 1, x1060,1));
arrayFill<<<512, 1>>>(x1362, 0.0f);
float* x5144 = (float*)myMalloc(1 * sizeof(float));;
x5144[0] = 1.0f;
float* x5146 = (float*)myMalloc(1 * sizeof(float));;
x5146[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,128,x5144,x1063,512,x5146, x1363, 512, x1063,512));
arrayFill<<<65536, 1>>>(x1363, 0.0f);
float* x5150 = (float*)myMalloc(1 * sizeof(float));;
x5150[0] = 1.0f;
float* x5152 = (float*)myMalloc(1 * sizeof(float));;
x5152[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5150,x1066,1,x5152, x1364, 1, x1066,1));
arrayFill<<<64, 1>>>(x1364, 0.0f);
float* x5156 = (float*)myMalloc(1 * sizeof(float));;
x5156[0] = 1.0f;
float* x5158 = (float*)myMalloc(1 * sizeof(float));;
x5158[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,512,x5156,x1069,256,x5158, x1365, 256, x1069,256));
arrayFill<<<131072, 1>>>(x1365, 0.0f);
float* x5162 = (float*)myMalloc(1 * sizeof(float));;
x5162[0] = 1.0f;
float* x5164 = (float*)myMalloc(1 * sizeof(float));;
x5164[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5162,x1072,1,x5164, x1366, 1, x1072,1));
arrayFill<<<256, 1>>>(x1366, 0.0f);
float* x5168 = (float*)myMalloc(1 * sizeof(float));;
x5168[0] = 1.0f;
float* x5170 = (float*)myMalloc(1 * sizeof(float));;
x5170[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5168,x1075,1,x5170, x1367, 1, x1075,1));
arrayFill<<<2048, 1>>>(x1367, 0.0f);
float* x5174 = (float*)myMalloc(1 * sizeof(float));;
x5174[0] = 1.0f;
float* x5176 = (float*)myMalloc(1 * sizeof(float));;
x5176[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5174,x1078,1,x5176, x1368, 1, x1078,1));
arrayFill<<<128, 1>>>(x1368, 0.0f);
float* x5180 = (float*)myMalloc(1 * sizeof(float));;
x5180[0] = 1.0f;
float* x5182 = (float*)myMalloc(1 * sizeof(float));;
x5182[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x5180,x1081,2304,x5182, x1369, 2304, x1081,2304));
arrayFill<<<589824, 1>>>(x1369, 0.0f);
float* x5186 = (float*)myMalloc(1 * sizeof(float));;
x5186[0] = 1.0f;
float* x5188 = (float*)myMalloc(1 * sizeof(float));;
x5188[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5186,x1084,1,x5188, x1370, 1, x1084,1));
arrayFill<<<1024, 1>>>(x1370, 0.0f);
float* x5192 = (float*)myMalloc(1 * sizeof(float));;
x5192[0] = 1.0f;
float* x5194 = (float*)myMalloc(1 * sizeof(float));;
x5194[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5192,x1087,1,x5194, x1371, 1, x1087,1));
arrayFill<<<256, 1>>>(x1371, 0.0f);
float* x5198 = (float*)myMalloc(1 * sizeof(float));;
x5198[0] = 1.0f;
float* x5200 = (float*)myMalloc(1 * sizeof(float));;
x5200[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2048,512,x5198,x1090,2048,x5200, x1372, 2048, x1090,2048));
arrayFill<<<1048576, 1>>>(x1372, 0.0f);
float* x5204 = (float*)myMalloc(1 * sizeof(float));;
x5204[0] = 1.0f;
float* x5206 = (float*)myMalloc(1 * sizeof(float));;
x5206[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5204,x1093,1,x5206, x1373, 1, x1093,1));
arrayFill<<<128, 1>>>(x1373, 0.0f);
float* x5210 = (float*)myMalloc(1 * sizeof(float));;
x5210[0] = 1.0f;
float* x5212 = (float*)myMalloc(1 * sizeof(float));;
x5212[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5210,x1096,1,x5212, x1374, 1, x1096,1));
arrayFill<<<1024, 1>>>(x1374, 0.0f);
float* x5216 = (float*)myMalloc(1 * sizeof(float));;
x5216[0] = 1.0f;
float* x5218 = (float*)myMalloc(1 * sizeof(float));;
x5218[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5216,x1099,1,x5218, x1375, 1, x1099,1));
arrayFill<<<128, 1>>>(x1375, 0.0f);
float* x5222 = (float*)myMalloc(1 * sizeof(float));;
x5222[0] = 1.0f;
float* x5224 = (float*)myMalloc(1 * sizeof(float));;
x5224[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x5222,x1102,256,x5224, x1376, 256, x1102,256));
arrayFill<<<262144, 1>>>(x1376, 0.0f);
float* x5228 = (float*)myMalloc(1 * sizeof(float));;
x5228[0] = 1.0f;
float* x5230 = (float*)myMalloc(1 * sizeof(float));;
x5230[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5228,x1105,1,x5230, x1377, 1, x1105,1));
arrayFill<<<256, 1>>>(x1377, 0.0f);
float* x5234 = (float*)myMalloc(1 * sizeof(float));;
x5234[0] = 1.0f;
float* x5236 = (float*)myMalloc(1 * sizeof(float));;
x5236[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5234,x1108,1,x5236, x1378, 1, x1108,1));
arrayFill<<<256, 1>>>(x1378, 0.0f);
float* x5240 = (float*)myMalloc(1 * sizeof(float));;
x5240[0] = 1.0f;
float* x5242 = (float*)myMalloc(1 * sizeof(float));;
x5242[0] = -0.005f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5240,x1111,1,x5242, x1379, 1, x1111,1));
arrayFill<<<1024, 1>>>(x1379, 0.0f);
int32_t x5246 = x1395 + 1;
int32_t x5248 = x5246 % x5247;
bool x5249 = x5248 == 0;
if (x5249) {
float x5254 = x1389;
double x5250 = (double)x1396;
double x5251 = 100.0 * x5250;
double x5253 = x5251 / x5252;
float x5255 = (float)x1395;
float x5256 = x5254 / x5255;
printf("Train epoch %d: [%d/%d (%.0f%%)] Average Loss: %.6f\n",x1385,x1396,x12,x5253,x5256);
fflush(stdout);
} else {
}
int64_t x5261 = (long)mallocAddr;
int64_t x5262 = x5261 - x1381;
memset((void*)x1381, 0, x5262);
mallocAddr = (void*)x1381;
int64_t x5265 = (long)gpuMallocAddr;
int64_t x5266 = x5265 - x1382;
cudaMemset((void*)x1382, 0, x5266);
gpuMallocAddr = (void*)x1382;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x5273 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x5274 = x5273 / 1000LL;
int64_t x5276 = x5273 / x5275;
printf("Training completed in %ldms (%ld us/images)\n",x5274,x5276);
float x5278 = x1389;
float x5280 = x5278 / x5279;
double x5281 = (double)x5280;
x1380[x1385] = x5281;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x5287 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
int64_t x5292 = (long)fopen(x0, "w");
fprintf((FILE *)x5292, "unit: %s\n", "1 epoch");
for(int x5294=0; x5294 < 4; x5294++) {
double x5295 = x1380[x5294];
fprintf((FILE *)x5292, "%lf\n", x5295);

}
float x5288 = (float)x5287;
float x5289 = x5288 / 1000000.0f;
float x5290 = x5289 - x40;
float x5291 = x5290 / 4.0f;
fprintf((FILE *)x5292, "run time: %lf %lf\n", x40, x5291);
fclose((FILE*)x5292);
// Backend cleanup.
CUBLAS_CALL(cublasDestroy(cublasHandle));
CUDA_CALL(cudaFree(gpuMallocBase));
      
CUDNN_CALL(cudnnDestroy(cudnnHandle));
}
/*****************************************
  End of C Generated Code                  
*******************************************/


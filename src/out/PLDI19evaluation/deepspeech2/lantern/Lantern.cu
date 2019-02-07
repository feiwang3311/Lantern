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

void myGpuFree(size_t bytes) {
  bytes = ((bytes + (1 << N) - 1) >> N) << N;
  gpuMallocAddr = (void *)((char *)gpuMallocAddr - bytes);
  cudaMemset((void*)gpuMallocAddr, 0, bytes);
  return;
}

template <typename T>
__global__ void arrayUpdate(T *data, int index, T value) {
  data[index] = value;
}

__global__ void arrayFill(float* data, float value, int size) {
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
template <int Dims>
static inline __device__ int compute(const int outputSizes[Dims], const int outputStrides[Dims],
                                     const int dimSize, const int concatDim, int linearIndex) {
  int offset = 0;
  #pragma unroll
  for (int i = Dims - 1; i >= 1; --i) {
    int curDimSize = i == concatDim? dimSize : outputSizes[i];
    int nextDimIndex = linearIndex / curDimSize;
    int curDimIndex = linearIndex - curDimSize * nextDimIndex;
    int curDimOffset = curDimIndex * outputStrides[i];
    offset += curDimOffset;
    linearIndex = nextDimIndex;
  }
  return offset + linearIndex * outputStrides[0];
}

// TODO: Only for Dim of rank 4, and only for 2 inputs
__global__ void concat2D_1D_greg(float* in1, int dimSize1, int nElement1,
                                 float* in2, int dimSize2, int nElement2,
                                 float* out, int concatDim,
                                 int outSize0, int outSize1, int outSize2, int outSize3,
                                 int outStride0, int outStride1, int outStride2, int outStride3) {
  int outSizes[] = {outSize0, outSize1, outSize2, outSize3};
  int outStrides[] = {outStride0, outStride1, outStride2, outStride3};
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nElement = blockIdx.y == 0 ? nElement1 : nElement2;
  if (tid >= nElement) return;
  float* data = blockIdx.y == 0 ? in1 : in2;
  int offset = blockIdx.y == 0 ? 0 : dimSize1;
  int dimSize = blockIdx.y == 0 ? dimSize1 : dimSize2;
  int dataOffset = offset * outStrides[concatDim];
  int stride = gridDim.x * blockDim.x;
  for (; tid < nElement; tid += stride) {
    int elementOffset = compute<4>(outSizes, //0, outSize1, outSize2, outSize3,
                                   outStrides, //0, outStride1, outStride2, outStride3,
                                   dimSize, concatDim, tid);
    out[dataOffset + elementOffset] = data[tid];
  }
}

// TODO: Only for Dim of rank 4, and only for 2 inputs, and only for concat at dim = 1
__global__ void concat2D_1D_greg_grad(float* in1, int dimSize1, int nElement1,
                                      float* in2, int dimSize2, int nElement2,
                                      float* out, int concatDim,
                                      int outSize0, int outSize1, int outSize2, int outSize3,
                                      int outStride0, int outStride1, int outStride2, int outStride3) {
  int outSizes[] = {outSize0, outSize1, outSize2, outSize3};
  int outStrides[] = {outStride0, outStride1, outStride2, outStride3};
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nElement = blockIdx.y == 0 ? nElement1 : nElement2;
  if (tid >= nElement) return;
  float* data = blockIdx.y == 0 ? in1 : in2;
  int offset = blockIdx.y == 0 ? 0 : dimSize1;
  int dimSize = blockIdx.y == 0 ? dimSize1 : dimSize2;
  int dataOffset = offset * outStride1;
  int stride = gridDim.x * blockDim.x;
  for (; tid < nElement; tid += stride) {
    int elementOffset = compute<4>(outSizes, //0, outSize1, outSize2, outSize3,
                                   outStrides, //0, outStride1, outStride2, outStride3,
                                   dimSize, concatDim, tid);
    data[tid] += out[dataOffset + elementOffset];
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

__global__ void addScalarInArrayInPlace(float* in, float* add, float scale, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) in[tid] += add[0] * scale;
}

__global__ void addScalar(float* in, float* out, float add, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) out[tid] = in[tid] + add;
}
__global__ void minusScalar(float* in, float* out, float minus, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) out[tid] = in[tid] - minus;
}
__global__ void multScalar(float* in, float* out, float mult, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) out[tid] = in[tid] * mult;
}
__global__ void divScalar(float* in, float* out, float div, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) out[tid] = in[tid] / div;
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

__global__ void clipAt(float* in, float bound, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride)
    if (tid < size) {
      if (in[tid] > bound) in[tid] = bound;
      if (in[tid] < -bound) in[tid] = -bound;
    }
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
float* x7 = (float*)myMalloc(14432 * sizeof(float));;
for(int x9=0; x9 < 14432; x9++) {
float x10 = (float)rand()/RAND_MAX;
float x11 = x10 - 0.5f;
float x12 = x11 * 0.23068394f;
x7[x9] = x12;

}
// Tensor 'toGPU' invocation.
float* x17 = (float*)myGpuMalloc(14432 * sizeof(float));
CUDA_CALL(cudaMemcpy(x17, x7, 14432 * sizeof(float), cudaMemcpyHostToDevice));
float* x19 = (float*)myGpuMalloc(14432 * sizeof(float));
float* x20 = (float*)myGpuMalloc(32 * sizeof(float));
arrayFill<<<28, 512>>>(x20, 1.0f, 32);
float* x22 = (float*)myGpuMalloc(32 * sizeof(float));
float* x23 = (float*)myGpuMalloc(32 * sizeof(float));
float* x24 = (float*)myGpuMalloc(32 * sizeof(float));
float* x25 = (float*)myGpuMalloc(32 * sizeof(float));
float* x26 = (float*)myGpuMalloc(32 * sizeof(float));
float* x27 = (float*)myMalloc(236544 * sizeof(float));;
for(int x29=0; x29 < 236544; x29++) {
float x30 = (float)rand()/RAND_MAX;
float x31 = x30 - 0.5f;
float x32 = x31 * 0.05698029f;
x27[x29] = x32;

}
// Tensor 'toGPU' invocation.
float* x37 = (float*)myGpuMalloc(236544 * sizeof(float));
CUDA_CALL(cudaMemcpy(x37, x27, 236544 * sizeof(float), cudaMemcpyHostToDevice));
float* x39 = (float*)myGpuMalloc(236544 * sizeof(float));
float* x40 = (float*)myGpuMalloc(32 * sizeof(float));
arrayFill<<<28, 512>>>(x40, 1.0f, 32);
float* x42 = (float*)myGpuMalloc(32 * sizeof(float));
float* x43 = (float*)myGpuMalloc(32 * sizeof(float));
float* x44 = (float*)myGpuMalloc(32 * sizeof(float));
float* x45 = (float*)myGpuMalloc(32 * sizeof(float));
float* x46 = (float*)myGpuMalloc(32 * sizeof(float));
printf("initial rnn input size is %d \n",672);
int32_t x48 = 0;
float* x49 = (float*)myGpuMalloc(3477504 * sizeof(float));
arrayFill<<<28, 512>>>(x49, 0.01f, 3477504);
float* x51 = (float*)myGpuMalloc(3477504 * sizeof(float));
int32_t x52 = x48;
float* x53 = x49+x52;
float* x54 = x51+x52;
x48 += 688128;
int32_t x56 = x48;
float* x57 = x49+x56;
float* x58 = x51+x56;
x48 += 1048576;
int32_t x60 = x48;
float* x61 = x49+x60;
float* x62 = x51+x60;
x48 += 688128;
int32_t x64 = x48;
float* x65 = x49+x64;
float* x66 = x51+x64;
x48 += 1048576;
int32_t x68 = x48;
float* x69 = x49+x68;
float* x70 = x51+x68;
x48 += 1024;
int32_t x72 = x48;
float* x73 = x49+x72;
float* x74 = x51+x72;
x48 += 1024;
int32_t x76 = x48;
float* x77 = x49+x76;
float* x78 = x51+x76;
x48 += 1024;
int32_t x80 = x48;
float* x81 = x49+x80;
float* x82 = x51+x80;
x48 += 1024;
int32_t x84 = 0;
float* x85 = (float*)myGpuMalloc(4198400 * sizeof(float));
arrayFill<<<28, 512>>>(x85, 0.01f, 4198400);
float* x87 = (float*)myGpuMalloc(4198400 * sizeof(float));
int32_t x88 = x84;
float* x89 = x85+x88;
float* x90 = x87+x88;
x84 += 1048576;
int32_t x92 = x84;
float* x93 = x85+x92;
float* x94 = x87+x92;
x84 += 1048576;
int32_t x96 = x84;
float* x97 = x85+x96;
float* x98 = x87+x96;
x84 += 1048576;
int32_t x100 = x84;
float* x101 = x85+x100;
float* x102 = x87+x100;
x84 += 1048576;
int32_t x104 = x84;
float* x105 = x85+x104;
float* x106 = x87+x104;
x84 += 1024;
int32_t x108 = x84;
float* x109 = x85+x108;
float* x110 = x87+x108;
x84 += 1024;
int32_t x112 = x84;
float* x113 = x85+x112;
float* x114 = x87+x112;
x84 += 1024;
int32_t x116 = x84;
float* x117 = x85+x116;
float* x118 = x87+x116;
x84 += 1024;
int32_t x120 = 0;
float* x121 = (float*)myGpuMalloc(4198400 * sizeof(float));
arrayFill<<<28, 512>>>(x121, 0.01f, 4198400);
float* x123 = (float*)myGpuMalloc(4198400 * sizeof(float));
int32_t x124 = x120;
float* x125 = x121+x124;
float* x126 = x123+x124;
x120 += 1048576;
int32_t x128 = x120;
float* x129 = x121+x128;
float* x130 = x123+x128;
x120 += 1048576;
int32_t x132 = x120;
float* x133 = x121+x132;
float* x134 = x123+x132;
x120 += 1048576;
int32_t x136 = x120;
float* x137 = x121+x136;
float* x138 = x123+x136;
x120 += 1048576;
int32_t x140 = x120;
float* x141 = x121+x140;
float* x142 = x123+x140;
x120 += 1024;
int32_t x144 = x120;
float* x145 = x121+x144;
float* x146 = x123+x144;
x120 += 1024;
int32_t x148 = x120;
float* x149 = x121+x148;
float* x150 = x123+x148;
x120 += 1024;
int32_t x152 = x120;
float* x153 = x121+x152;
float* x154 = x123+x152;
x120 += 1024;
float* x156 = (float*)myGpuMalloc(1024 * sizeof(float));
arrayFill<<<28, 512>>>(x156, 1.0f, 1024);
float* x158 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x159 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x160 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x161 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x162 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x163 = (float*)myMalloc(29696 * sizeof(float));;
for(int x165=0; x165 < 29696; x165++) {
float x166 = (float)rand()/RAND_MAX;
float x167 = x166 - 0.5f;
float x168 = x167 * 0.03125f;
x163[x165] = x168;

}
// Tensor 'toGPU' invocation.
float* x173 = (float*)myGpuMalloc(29696 * sizeof(float));
CUDA_CALL(cudaMemcpy(x173, x163, 29696 * sizeof(float), cudaMemcpyHostToDevice));
float* x175 = (float*)myGpuMalloc(29696 * sizeof(float));
int32_t x176 = open("/scratch-ml00/wang603/deepspeechData/deepspeech_train.bin",0);
int64_t x177 = fsize(x176);
printf("file size is %ld\n",x177);
char* x179 = (char*)mmap(0, x177, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x176, 0);
int64_t x180 = (long)x179;
int64_t x181 = x180;
int64_t x182 = x181;
int* x183 = (int32_t*) x182;
int64_t x184 = (int64_t)4;
x181 += x184;
int32_t x186 = x183[0];
int64_t x187 = x181;
int* x188 = (int32_t*) x187;
x181 += x184;
int32_t x190 = x188[0];
printf("data size is %d batches, %d batch size\n",200,x186);
int* x193 = (int32_t*)myMalloc(200 * sizeof(int32_t));;
int* x194 = (int32_t*)myMalloc(200 * sizeof(int32_t));;
float** x195 = (float**)myMalloc(200 * sizeof(float*));;
float** x196 = (float**)myMalloc(200 * sizeof(float*));;
int** x197 = (int**)myMalloc(200 * sizeof(int*));;
int** x198 = (int**)myMalloc(200 * sizeof(int*));;
// load data by batchs
int32_t x224 = 4 * x186;
int64_t x225 = (int64_t)x224;
for(int x201=0; x201 < 200; x201++) {
int64_t x202 = x181;
int* x203 = (int32_t*) x202;
x181 += x184;
int32_t x205 = x203[0];
x193[x201] = x205;
int64_t x207 = x181;
int* x208 = (int32_t*) x207;
x181 += x184;
int32_t x210 = x208[0];
x194[x201] = x210;
int32_t x212 = x193[x201];
int32_t x214 = x194[x201];
int64_t x216 = x181;
float* x217 = (float*) x216;
int32_t x213 = x186 * x212;
int32_t x215 = x213 * x214;
int32_t x218 = 4 * x215;
int64_t x219 = (int64_t)x218;
x181 += x219;
x195[x201] = x217;
int64_t x222 = x181;
float* x223 = (float*) x222;
x181 += x225;
x196[x201] = x223;
int64_t x228 = x181;
int* x229 = (int32_t*) x228;
x181 += x225;
x197[x201] = x229;
int* x232 = x197[x201];
int* x233 = x197[x201];
int32_t x234 = accumulate(x232, x233 + x186, 0);
int64_t x235 = x181;
int* x236 = (int32_t*) x235;
int32_t x237 = 4 * x234;
int64_t x238 = (int64_t)x237;
x181 += x238;
x198[x201] = x236;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x245 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
float x246 = (float)x245;
float x247 = x246 / 1000000.0f;
printf("Data reading (all prepare time) in %lf sec\n",x247);
double* x249 = (double*)myMalloc(1 * sizeof(double));;
double* x250 = (double*)myMalloc(1 * sizeof(double));;
int64_t x251 = (long)mallocAddr;
int64_t x252 = (long)gpuMallocAddr;
// training loop starts here
int32_t x296 = x186 * 32;
int32_t x383 = 2048 / 2;
int32_t x384 = 2 * x383;
int32_t x385 = x186 * x384;
int32_t x387 = x186 * x383;
int32_t x615 = x186 * 20;
int32_t x191 = x186 * 200;
double x620 = (double)x191;
int64_t x643 = (int64_t)x191;
float x650 = (float)x191;
for(int x255=0; x255 < 1; x255++) {
struct timeval begin_1, end_1, diff_1;
int32_t x257 = 0;
int32_t x258 = x257;
int32_t x259 = x258;
float x260 = 0.0f;
float x261 = x260;
float x262 = x261;
int32_t x263 = x255 + 1;
printf("Start training epoch %d\n",x263);
gettimeofday(&begin_1, NULL);
for(int x266=0; x266 < 200; x266++) {
int32_t x267 = x194[x266];
int32_t x268 = x193[x266];
float* x269 = x195[x266];
float* x272 = x196[x266];
int* x273 = x198[x266];
int* x274 = x197[x266];
x259 += x186;
// Tensor 'toGPU' invocation.
int32_t x270 = x268 * x267;
int32_t x271 = x186 * x270;
float* x277 = (float*)myGpuMalloc(x271 * sizeof(float));
CUDA_CALL(cudaMemcpy(x277, x269, x271 * sizeof(float), cudaMemcpyHostToDevice));
float* x279 = (float*)myGpuMalloc(2 * sizeof(float));
float* x280 = (float*)myGpuMalloc(1 * sizeof(float));
float* x281 = (float*)myGpuMalloc(1 * sizeof(float));
// allocate memory to save the final loss in CPU Tensor
float* x283 = (float*)myGpuMalloc(1 * sizeof(float));
int32_t x290 = x267 - 11;
int32_t x291 = x290 / 2;
int32_t x292 = x291 + 1;
int32_t x287 = x268 - 41;
int32_t x288 = x287 / 2;
int32_t x289 = x288 + 1;
int32_t x297 = x296 * x289;
int32_t x298 = x297 * x292;
float* x299 = (float*)myGpuMalloc(x298 * sizeof(float));
float* x300 = (float*)myMalloc(1 * sizeof(float));;
x300[0] = 0.0f;
float* x302 = (float*)myMalloc(1 * sizeof(float));;
x302[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x186, 1, x268, x267));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    32, 1, 41, 11));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x186, 32, x289, x292));

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
    x302, in_desc, x277, filt_desc, x17,
    conv_desc, algo, ws_data, ws_size,
    x300, out_desc, x299));
};
float* x305 = (float*)myGpuMalloc(x298 * sizeof(float));
int32_t x293 = x289 * x292;
int32_t x294 = 32 * x293;
int32_t x295 = x186 * x294;
float* x306 = (float*)myGpuMalloc(x295 * sizeof(float));
float* x307 = (float*)myGpuMalloc(32 * sizeof(float));
float* x308 = (float*)myGpuMalloc(32 * sizeof(float));
float* x309 = (float*)myMalloc(1 * sizeof(float));;
x309[0] = 0.0f;
float* x311 = (float*)myMalloc(1 * sizeof(float));;
x311[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x186, 32, x289, x292));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x186, 32, x289, x292));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 32, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x311, x309, in_desc, x299, out_desc, x306, sbmv_desc, x20,
    x23, 0.1, x25, x26, 1.0E-5,
    x307, x308));
};
float* x314 = (float*)myGpuMalloc(x298 * sizeof(float));
hardTanh<<<28, 512>>>(x306, x306, 0.0, 20.0, true);
int32_t x322 = x292 - 11;
int32_t x323 = x322 / 1;
int32_t x324 = x323 + 1;
int32_t x319 = x289 - 21;
int32_t x320 = x319 / 2;
int32_t x321 = x320 + 1;
int32_t x328 = x296 * x321;
int32_t x329 = x328 * x324;
float* x330 = (float*)myGpuMalloc(x329 * sizeof(float));
float* x331 = (float*)myMalloc(1 * sizeof(float));;
x331[0] = 0.0f;
float* x333 = (float*)myMalloc(1 * sizeof(float));;
x333[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x186, 32, x289, x292));

cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    32, 32, 21, 11));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x186, 32, x321, x324));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 2, 1, 1, 1,
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
    x333, in_desc, x306, filt_desc, x37,
    conv_desc, algo, ws_data, ws_size,
    x331, out_desc, x330));
};
float* x336 = (float*)myGpuMalloc(x329 * sizeof(float));
int32_t x325 = x321 * x324;
int32_t x326 = 32 * x325;
int32_t x327 = x186 * x326;
float* x337 = (float*)myGpuMalloc(x327 * sizeof(float));
float* x338 = (float*)myGpuMalloc(32 * sizeof(float));
float* x339 = (float*)myGpuMalloc(32 * sizeof(float));
float* x340 = (float*)myMalloc(1 * sizeof(float));;
x340[0] = 0.0f;
float* x342 = (float*)myMalloc(1 * sizeof(float));;
x342[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x186, 32, x321, x324));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x186, 32, x321, x324));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 32, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x342, x340, in_desc, x330, out_desc, x337, sbmv_desc, x40,
    x43, 0.1, x45, x46, 1.0E-5,
    x338, x339));
};
float* x345 = (float*)myGpuMalloc(x329 * sizeof(float));
hardTanh<<<28, 512>>>(x337, x337, 0.0, 20.0, true);
// after conv ops
int32_t x348 = 32 * x321;
int32_t x349 = x348 * x324;
int32_t x350 = x186 * x349;
float* x351 = (float*)myGpuMalloc(x350 * sizeof(float));
int* x354 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
int32_t x352 = x186 * x348;
x354[2] = x352;
x354[0] = x348;
x354[1] = 1;
x354[3] = 1;
float* x359 = (float*)myMalloc(1 * sizeof(float));;
x359[0] = 1.0f;
float* x361 = (float*)myMalloc(0 * sizeof(float));;
x361[0] = 0.0f;
int32_t x363 = x354[0];
int32_t x364 = x354[1];
int32_t x365 = x354[2];
int32_t x366 = x354[3];

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
    in_desc, CUDNN_DATA_FLOAT,
    x186, x348, x324, 1,
    x349, x324, 1, 1));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
    out_desc, CUDNN_DATA_FLOAT,
    x186, x348, x324, 1,
    x363, x364, x365, x366));

CUDNN_CALL(cudnnTransformTensor(
    cudnnHandle, x359, in_desc, x337, x361, out_desc, x351));
};
int32_t x368 = x324 * x186;
int32_t x369 = x368 * x348;
float* x370 = (float*)myGpuMalloc(x369 * sizeof(float));
// after resize and permute
float* x372 = (float*)NULL;
float* x373 = (float*)NULL;
float* x374 = (float*)NULL;
int32_t x377 = x368 * 2048;
float* x378 = (float*)myGpuMalloc(x377 * sizeof(float));
float* x379 = (float*)NULL;
int32_t x380 = 0;

{
size_t dropoutStateSize;
CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize));
void* dropoutStates = myGpuMalloc(dropoutStateSize);

cudnnDropoutDescriptor_t dropout_desc;
CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
CUDNN_CALL(cudnnSetDropoutDescriptor(
    dropout_desc, cudnnHandle, 0.0, dropoutStates, dropoutStateSize, time(NULL)));

cudnnRNNDescriptor_t rnn_desc;
CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
CUDNN_CALL(cudnnSetRNNDescriptor(
    cudnnHandle, rnn_desc,
    /*hiddenSize*/ 1024, /*numLayers*/ 1,
    dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
    CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH));
int32_t seqLength = x324;
int32_t batchSize = x186;
int32_t inputSize = x348;

cudnnTensorDescriptor_t x_descs[seqLength];
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
int x_dims[] = {batchSize, inputSize, 1};
int x_strides[] = {x_dims[1] * x_dims[2], x_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    x_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims, x_strides));
for (int i = 0; i < seqLength; i++) {
  x_descs[i] = x_desc;
}

// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
// The second dimension must match the first dimension of the tensors described in xDesc.
// The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
cudnnTensorDescriptor_t hx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc));
int hx_dims[] = {2, batchSize, 1024};
int hx_strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    hx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims, hx_strides));

cudnnTensorDescriptor_t cx_desc = hx_desc;

size_t paramsSize;
CUDNN_CALL(cudnnGetRNNParamsSize(
    cudnnHandle, rnn_desc, x_descs[0], &paramsSize, CUDNN_DATA_FLOAT));
#ifdef DEBUG
assert(paramsSize / sizeof(float) == 3477504 && "Expected parameter size mismatch");
#endif

cudnnFilterDescriptor_t w_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
int w_dims[] = {int(paramsSize / sizeof(float)), 1, 1};
CUDNN_CALL(cudnnSetFilterNdDescriptor(
    w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims));

cudnnTensorDescriptor_t y_descs[seqLength];
cudnnTensorDescriptor_t y_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
int y_dims[] = {batchSize, 2048, 1};
int y_strides[] = {y_dims[1] * y_dims[2], y_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    y_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims, y_strides));
for (int i = 0; i < seqLength; i++) {
  y_descs[i] = y_desc;
}

cudnnTensorDescriptor_t hy_desc = hx_desc;
cudnnTensorDescriptor_t cy_desc = cx_desc;

size_t workspaceSize;
CUDNN_CALL(cudnnGetRNNWorkspaceSize(
    cudnnHandle, rnn_desc, seqLength, x_descs, &workspaceSize));

// Reserve space used by `ForwardTraining` function.
size_t reserveSize;
CUDNN_CALL(cudnnGetRNNTrainingReserveSize(
    cudnnHandle, rnn_desc, seqLength, x_descs, &reserveSize));
void* reserveSpace = myGpuMalloc(reserveSize);
x379 = (float*)reserveSpace;
x380 = (int)reserveSize;
void* workspace = myGpuMalloc(workspaceSize);
CUDNN_CALL(cudnnRNNForwardTraining(
    cudnnHandle, rnn_desc, seqLength, x_descs, x351,
    hx_desc,x372, cx_desc,x373, w_desc, x49, y_descs, x378,
    hy_desc,x374, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
myGpuFree(workspaceSize);
};
float* x382 = (float*)myGpuMalloc(x377 * sizeof(float));
int32_t x389 = x368 * x383;
float* x390 = (float*)myGpuMalloc(x389 * sizeof(float));
float* x391 = (float*)myMalloc(1 * sizeof(float));;
x391[0] = 0.0f;
float* x393 = (float*)myMalloc(1 * sizeof(float));;
x393[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x324, x186, 2, x383));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x324, x186, 1, x383));

cudnnReduceTensorDescriptor_t reduce_desc;
CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&reduce_desc));
CUDNN_CALL(cudnnSetReduceTensorDescriptor(
    reduce_desc, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
    CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));

void *indices = nullptr; // Don't store indices.

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetReductionWorkspaceSize(
    cudnnHandle, reduce_desc, x_desc, out_desc, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnReduceTensor(
    cudnnHandle, reduce_desc, indices, 0, ws_data, ws_size,
    x393, x_desc, x378, x391, out_desc, x390));
};
float* x396 = (float*)myGpuMalloc(x389 * sizeof(float));
float* x397 = (float*)NULL;
float* x398 = (float*)NULL;
float* x399 = (float*)NULL;
float* x400 = (float*)myGpuMalloc(x377 * sizeof(float));
float* x401 = (float*)NULL;
int32_t x402 = 0;

{
size_t dropoutStateSize;
CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize));
void* dropoutStates = myGpuMalloc(dropoutStateSize);

cudnnDropoutDescriptor_t dropout_desc;
CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
CUDNN_CALL(cudnnSetDropoutDescriptor(
    dropout_desc, cudnnHandle, 0.0, dropoutStates, dropoutStateSize, time(NULL)));

cudnnRNNDescriptor_t rnn_desc;
CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
CUDNN_CALL(cudnnSetRNNDescriptor(
    cudnnHandle, rnn_desc,
    /*hiddenSize*/ 1024, /*numLayers*/ 1,
    dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
    CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH));
int32_t seqLength = x324;
int32_t batchSize = x186;
int32_t inputSize = x383;

cudnnTensorDescriptor_t x_descs[seqLength];
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
int x_dims[] = {batchSize, inputSize, 1};
int x_strides[] = {x_dims[1] * x_dims[2], x_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    x_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims, x_strides));
for (int i = 0; i < seqLength; i++) {
  x_descs[i] = x_desc;
}

// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
// The second dimension must match the first dimension of the tensors described in xDesc.
// The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
cudnnTensorDescriptor_t hx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc));
int hx_dims[] = {2, batchSize, 1024};
int hx_strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    hx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims, hx_strides));

cudnnTensorDescriptor_t cx_desc = hx_desc;

size_t paramsSize;
CUDNN_CALL(cudnnGetRNNParamsSize(
    cudnnHandle, rnn_desc, x_descs[0], &paramsSize, CUDNN_DATA_FLOAT));
#ifdef DEBUG
assert(paramsSize / sizeof(float) == 4198400 && "Expected parameter size mismatch");
#endif

cudnnFilterDescriptor_t w_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
int w_dims[] = {int(paramsSize / sizeof(float)), 1, 1};
CUDNN_CALL(cudnnSetFilterNdDescriptor(
    w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims));

cudnnTensorDescriptor_t y_descs[seqLength];
cudnnTensorDescriptor_t y_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
int y_dims[] = {batchSize, 2048, 1};
int y_strides[] = {y_dims[1] * y_dims[2], y_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    y_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims, y_strides));
for (int i = 0; i < seqLength; i++) {
  y_descs[i] = y_desc;
}

cudnnTensorDescriptor_t hy_desc = hx_desc;
cudnnTensorDescriptor_t cy_desc = cx_desc;

size_t workspaceSize;
CUDNN_CALL(cudnnGetRNNWorkspaceSize(
    cudnnHandle, rnn_desc, seqLength, x_descs, &workspaceSize));

// Reserve space used by `ForwardTraining` function.
size_t reserveSize;
CUDNN_CALL(cudnnGetRNNTrainingReserveSize(
    cudnnHandle, rnn_desc, seqLength, x_descs, &reserveSize));
void* reserveSpace = myGpuMalloc(reserveSize);
x401 = (float*)reserveSpace;
x402 = (int)reserveSize;
void* workspace = myGpuMalloc(workspaceSize);
CUDNN_CALL(cudnnRNNForwardTraining(
    cudnnHandle, rnn_desc, seqLength, x_descs, x390,
    hx_desc,x397, cx_desc,x398, w_desc, x85, y_descs, x400,
    hy_desc,x399, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
myGpuFree(workspaceSize);
};
float* x404 = (float*)myGpuMalloc(x377 * sizeof(float));
float* x405 = (float*)myGpuMalloc(x389 * sizeof(float));
float* x406 = (float*)myMalloc(1 * sizeof(float));;
x406[0] = 0.0f;
float* x408 = (float*)myMalloc(1 * sizeof(float));;
x408[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x324, x186, 2, x383));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x324, x186, 1, x383));

cudnnReduceTensorDescriptor_t reduce_desc;
CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&reduce_desc));
CUDNN_CALL(cudnnSetReduceTensorDescriptor(
    reduce_desc, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
    CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));

void *indices = nullptr; // Don't store indices.

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetReductionWorkspaceSize(
    cudnnHandle, reduce_desc, x_desc, out_desc, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnReduceTensor(
    cudnnHandle, reduce_desc, indices, 0, ws_data, ws_size,
    x408, x_desc, x400, x406, out_desc, x405));
};
float* x411 = (float*)myGpuMalloc(x389 * sizeof(float));
float* x412 = (float*)NULL;
float* x413 = (float*)NULL;
float* x414 = (float*)NULL;
float* x415 = (float*)myGpuMalloc(x377 * sizeof(float));
float* x416 = (float*)NULL;
int32_t x417 = 0;

{
size_t dropoutStateSize;
CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize));
void* dropoutStates = myGpuMalloc(dropoutStateSize);

cudnnDropoutDescriptor_t dropout_desc;
CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
CUDNN_CALL(cudnnSetDropoutDescriptor(
    dropout_desc, cudnnHandle, 0.0, dropoutStates, dropoutStateSize, time(NULL)));

cudnnRNNDescriptor_t rnn_desc;
CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
CUDNN_CALL(cudnnSetRNNDescriptor(
    cudnnHandle, rnn_desc,
    /*hiddenSize*/ 1024, /*numLayers*/ 1,
    dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
    CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH));
int32_t seqLength = x324;
int32_t batchSize = x186;
int32_t inputSize = x383;

cudnnTensorDescriptor_t x_descs[seqLength];
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
int x_dims[] = {batchSize, inputSize, 1};
int x_strides[] = {x_dims[1] * x_dims[2], x_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    x_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims, x_strides));
for (int i = 0; i < seqLength; i++) {
  x_descs[i] = x_desc;
}

// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
// The second dimension must match the first dimension of the tensors described in xDesc.
// The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
cudnnTensorDescriptor_t hx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc));
int hx_dims[] = {2, batchSize, 1024};
int hx_strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    hx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims, hx_strides));

cudnnTensorDescriptor_t cx_desc = hx_desc;

size_t paramsSize;
CUDNN_CALL(cudnnGetRNNParamsSize(
    cudnnHandle, rnn_desc, x_descs[0], &paramsSize, CUDNN_DATA_FLOAT));
#ifdef DEBUG
assert(paramsSize / sizeof(float) == 4198400 && "Expected parameter size mismatch");
#endif

cudnnFilterDescriptor_t w_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
int w_dims[] = {int(paramsSize / sizeof(float)), 1, 1};
CUDNN_CALL(cudnnSetFilterNdDescriptor(
    w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims));

cudnnTensorDescriptor_t y_descs[seqLength];
cudnnTensorDescriptor_t y_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
int y_dims[] = {batchSize, 2048, 1};
int y_strides[] = {y_dims[1] * y_dims[2], y_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    y_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims, y_strides));
for (int i = 0; i < seqLength; i++) {
  y_descs[i] = y_desc;
}

cudnnTensorDescriptor_t hy_desc = hx_desc;
cudnnTensorDescriptor_t cy_desc = cx_desc;

size_t workspaceSize;
CUDNN_CALL(cudnnGetRNNWorkspaceSize(
    cudnnHandle, rnn_desc, seqLength, x_descs, &workspaceSize));

// Reserve space used by `ForwardTraining` function.
size_t reserveSize;
CUDNN_CALL(cudnnGetRNNTrainingReserveSize(
    cudnnHandle, rnn_desc, seqLength, x_descs, &reserveSize));
void* reserveSpace = myGpuMalloc(reserveSize);
x416 = (float*)reserveSpace;
x417 = (int)reserveSize;
void* workspace = myGpuMalloc(workspaceSize);
CUDNN_CALL(cudnnRNNForwardTraining(
    cudnnHandle, rnn_desc, seqLength, x_descs, x405,
    hx_desc,x412, cx_desc,x413, w_desc, x121, y_descs, x415,
    hy_desc,x414, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
myGpuFree(workspaceSize);
};
float* x419 = (float*)myGpuMalloc(x377 * sizeof(float));
float* x420 = (float*)myGpuMalloc(x389 * sizeof(float));
float* x421 = (float*)myMalloc(1 * sizeof(float));;
x421[0] = 0.0f;
float* x423 = (float*)myMalloc(1 * sizeof(float));;
x423[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x324, x186, 2, x383));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x324, x186, 1, x383));

cudnnReduceTensorDescriptor_t reduce_desc;
CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&reduce_desc));
CUDNN_CALL(cudnnSetReduceTensorDescriptor(
    reduce_desc, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
    CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));

void *indices = nullptr; // Don't store indices.

// Workspace.
size_t ws_size;
CUDNN_CALL(cudnnGetReductionWorkspaceSize(
    cudnnHandle, reduce_desc, x_desc, out_desc, &ws_size));
void *ws_data = myGpuMalloc(ws_size);
CUDNN_CALL(cudnnReduceTensor(
    cudnnHandle, reduce_desc, indices, 0, ws_data, ws_size,
    x423, x_desc, x415, x421, out_desc, x420));
};
float* x426 = (float*)myGpuMalloc(x389 * sizeof(float));
// after RNN layers
// after maybe lookahead
float* x431 = (float*)myGpuMalloc(x389 * sizeof(float));
float* x432 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x433 = (float*)myGpuMalloc(1024 * sizeof(float));
float* x434 = (float*)myMalloc(1 * sizeof(float));;
x434[0] = 0.0f;
float* x436 = (float*)myMalloc(1 * sizeof(float));;
x436[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x368, x383, 1, 1));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
    cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,
    x436, x434, in_desc, x420, in_desc, x431, sbmv_desc, x156,
    x159, 0.1, x161, x162, 1.0E-5,
    x432, x433));
};
float* x439 = (float*)myGpuMalloc(x389 * sizeof(float));
int32_t x440 = x368 * 29;
float* x441 = (float*)myGpuMalloc(x440 * sizeof(float));
float* x442 = (float*)myMalloc(1 * sizeof(float));;
x442[0] = 0.0f;
float* x444 = (float*)myMalloc(1 * sizeof(float));;
x444[0] = 1.0f;
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 29,x368,1024,x444,x173,29,x431,1024,x442,x441,29));
float* x447 = (float*)myGpuMalloc(x440 * sizeof(float));
float* x450 = (float*)myMalloc(1 * sizeof(float));;
x450[0] = 0.0f;
float* x452 = (float*)myMalloc(1 * sizeof(float));;
x452[0] = 1.0f;
float* x454 = (float*)myGpuMalloc(x440 * sizeof(float));

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x368, 29, 1, 1));
CUDNN_CALL(cudnnSoftmaxForward(
    cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
    x452, x_desc, x441, x450, x_desc, x454));
};
float* x456 = (float*)myGpuMalloc(x440 * sizeof(float));
// before CTC loss
int* x458 = (int32_t*)myMalloc(x186 * sizeof(int32_t));;
float x462 = (float)x324;
for(int x460=0; x460 < x186; x460++) {
float x461 = x272[x460];
float x463 = x461 * x462;
int32_t x464 = (int)x463;
x458[x460] = x464;

}
float* x469 = (float*)myGpuMalloc(x186 * sizeof(float));

{
cudnnTensorDescriptor_t probs_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&probs_desc));
int probs_dims[] = {x324, x186, 29};
int probs_strides[] = {probs_dims[1] * probs_dims[2], probs_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    probs_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, probs_dims, probs_strides));

cudnnTensorDescriptor_t grad_desc = probs_desc;

cudnnCTCLossDescriptor_t ctc_desc;
CUDNN_CALL(cudnnCreateCTCLossDescriptor(&ctc_desc));
CUDNN_CALL(cudnnSetCTCLossDescriptor(ctc_desc, CUDNN_DATA_FLOAT));
size_t wsSize;
CUDNN_CALL(cudnnGetCTCLossWorkspaceSize(
    cudnnHandle, probs_desc, grad_desc, x273, x274, x458,
    CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, &wsSize));
void *ws = myGpuMalloc(wsSize);

CUDNN_CALL(cudnnCTCLoss(
    cudnnHandle, probs_desc, x454, x273, x274, x458,
    x469, grad_desc, x456, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, ws, wsSize));
};
float* x471 = (float*)myGpuMalloc(1 * sizeof(float));
float* x472 = (float*)myMalloc(1 * sizeof(float));;
x472[0] = 0.0f;
float* x474 = (float*)myMalloc(1 * sizeof(float));;
x474[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x186, 1, 1, 1));

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
    x474, x_desc, x469, x472, out_desc, x471));
};
// after CTC loss
float* x478 = (float*)myGpuMalloc(1 * sizeof(float));
// make sure the size of loss is 1
arrayFill<<<28, 512>>>(x478, 1.0f, 1);
// backend is lantern.TensorDslCudnn$BackendCudnn@2d338b23
CUDA_CALL(cudaMemcpy(x283, x471, 1 * sizeof(float), cudaMemcpyDeviceToDevice));
float* x483 = (float*)myMalloc(1 * sizeof(float));;
x483[0] = 1.0f;

{
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x368, 29, 1, 1));
CUDNN_CALL(cudnnSoftmaxBackward(
    cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
    x483, x_desc, x454, x_desc, x456,
    x483, x_desc, x447));
};
float* x486 = (float*)myMalloc(1 * sizeof(float));;
x486[0] = 0.0f;
float* x488 = (float*)myMalloc(1 * sizeof(float));;
x488[0] = 1.0f;
// backprop of matrix-matrix-dot
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, x383,x368,29,x488,x173,29,x447,29,x488,x439,x383));
CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 29,x383,x368,x488,x447,29,x431,x383,x488,x175,29));
float* x493 = (float*)myMalloc(1 * sizeof(float));;
x493[0] = 0.0f;
float* x495 = (float*)myMalloc(1 * sizeof(float));;
x495[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x368, x383, 1, 1));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 1024, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,
    x495, x495, x495, x495, in_desc, x420,
    in_desc, x439, in_desc, x426, sbmv_desc, x156,
    x158,x160, 1.0E-5, x432, x433));
};
// backprop for sum on dim op
int32_t x386 = x324 * x385;
sum_grad<<<28, 512>>>(x419, x324, x186, 2, x383, x386, x426, x387, x383, 1, 2);
;
float* x500 = (float*)NULL;
float* x501 = (float*)NULL;

{
size_t dropoutStateSize;
CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize));
void* dropoutStates = myGpuMalloc(dropoutStateSize);

cudnnDropoutDescriptor_t dropout_desc;
CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
CUDNN_CALL(cudnnSetDropoutDescriptor(
    dropout_desc, cudnnHandle, 0.0, dropoutStates, dropoutStateSize, time(NULL)));

cudnnRNNDescriptor_t rnn_desc;
CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
CUDNN_CALL(cudnnSetRNNDescriptor(
    cudnnHandle, rnn_desc,
    /*hiddenSize*/ 1024, /*numLayers*/ 1,
    dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
    CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH));
int32_t seqLength = x324;
int32_t batchSize = x186;
int32_t inputSize = x383;

cudnnTensorDescriptor_t dx_descs[seqLength];
cudnnTensorDescriptor_t dx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&dx_desc));
int x_dims[] = {batchSize, inputSize, 1};
int x_strides[] = {x_dims[1] * x_dims[2], x_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    dx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims, x_strides));
for (int i = 0; i < seqLength; i++) {
  dx_descs[i] = dx_desc;
}

// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
// The second dimension must match the first dimension of the tensors described in xDesc.
// The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
cudnnTensorDescriptor_t hx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc));
int hx_dims[] = {2, batchSize, 1024};
int hx_strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    hx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims, hx_strides));

cudnnTensorDescriptor_t cx_desc = hx_desc;

size_t paramsSize;
CUDNN_CALL(cudnnGetRNNParamsSize(
    cudnnHandle, rnn_desc, dx_descs[0], &paramsSize, CUDNN_DATA_FLOAT));
#ifdef DEBUG
assert(paramsSize / sizeof(float) == 4198400 && "Expected parameter size mismatch");
#endif

cudnnFilterDescriptor_t w_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
int w_dims[] = {int(paramsSize / sizeof(float)), 1, 1};
CUDNN_CALL(cudnnSetFilterNdDescriptor(
    w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims));

cudnnTensorDescriptor_t y_descs[seqLength];
cudnnTensorDescriptor_t y_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
int y_dims[] = {batchSize, 2048, 1};
int y_strides[] = {y_dims[1] * y_dims[2], y_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    y_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims, y_strides));
for (int i = 0; i < seqLength; i++) {
  y_descs[i] = y_desc;
}

cudnnTensorDescriptor_t dhx_desc = hx_desc;
cudnnTensorDescriptor_t hy_desc = hx_desc;
cudnnTensorDescriptor_t dhy_desc = hy_desc;

cudnnTensorDescriptor_t dcx_desc = cx_desc;
cudnnTensorDescriptor_t cy_desc = cx_desc;
cudnnTensorDescriptor_t dcy_desc = cy_desc;

size_t workspaceSize;
CUDNN_CALL(cudnnGetRNNWorkspaceSize(
    cudnnHandle, rnn_desc, seqLength, dx_descs, &workspaceSize));
void* workspace = myGpuMalloc(workspaceSize);
CUDNN_CALL(cudnnRNNBackwardData(
    cudnnHandle, rnn_desc, seqLength, y_descs, x415, y_descs, x419,
    dhy_desc, NULL, dcy_desc, NULL, w_desc, x121, hx_desc, x500,
    cx_desc, x501, dx_descs, x411, dhx_desc, NULL, dcx_desc, NULL,
    workspace, workspaceSize, x416, x417));
myGpuFree(workspaceSize);
};
float* x503 = (float*)NULL;

{
size_t dropoutStateSize;
CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize));
void* dropoutStates = myGpuMalloc(dropoutStateSize);

cudnnDropoutDescriptor_t dropout_desc;
CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
CUDNN_CALL(cudnnSetDropoutDescriptor(
    dropout_desc, cudnnHandle, 0.0, dropoutStates, dropoutStateSize, time(NULL)));

cudnnRNNDescriptor_t rnn_desc;
CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
CUDNN_CALL(cudnnSetRNNDescriptor(
    cudnnHandle, rnn_desc,
    /*hiddenSize*/ 1024, /*numLayers*/ 1,
    dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
    CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH));
int32_t seqLength = x324;
int32_t batchSize = x186;
int32_t inputSize = x383;

cudnnTensorDescriptor_t x_descs[seqLength];
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
int x_dims[] = {batchSize, inputSize, 1};
int x_strides[] = {x_dims[1] * x_dims[2], x_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    x_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims, x_strides));
for (int i = 0; i < seqLength; i++) {
  x_descs[i] = x_desc;
}

// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
// The second dimension must match the first dimension of the tensors described in xDesc.
// The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
cudnnTensorDescriptor_t hx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc));
int hx_dims[] = {2, batchSize, 1024};
int hx_strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    hx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims, hx_strides));

size_t paramsSize;
CUDNN_CALL(cudnnGetRNNParamsSize(
    cudnnHandle, rnn_desc, x_descs[0], &paramsSize, CUDNN_DATA_FLOAT));
#ifdef DEBUG
assert(paramsSize / sizeof(float) == 4198400 && "Expected parameter size mismatch");
#endif

cudnnFilterDescriptor_t dw_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&dw_desc));
int w_dims[] = {int(paramsSize / sizeof(float)), 1, 1};
CUDNN_CALL(cudnnSetFilterNdDescriptor(
    dw_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims));

cudnnTensorDescriptor_t y_descs[seqLength];
cudnnTensorDescriptor_t y_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
int y_dims[] = {batchSize, 2048, 1};
int y_strides[] = {y_dims[1] * y_dims[2], y_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    y_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims, y_strides));
for (int i = 0; i < seqLength; i++) {
  y_descs[i] = y_desc;
}

size_t workspaceSize;
CUDNN_CALL(cudnnGetRNNWorkspaceSize(
    cudnnHandle, rnn_desc, seqLength, x_descs, &workspaceSize));
void* workspace = myGpuMalloc(workspaceSize);
CUDNN_CALL(cudnnRNNBackwardWeights(
    cudnnHandle, rnn_desc, seqLength, x_descs, x405, hx_desc, x503,
    y_descs, x415, workspace, workspaceSize,
    dw_desc, x123, x416, x417));
myGpuFree(workspaceSize);
};
// backprop for sum on dim op
sum_grad<<<28, 512>>>(x404, x324, x186, 2, x383, x386, x411, x387, x383, 1, 2);
;
float* x507 = (float*)NULL;
float* x508 = (float*)NULL;

{
size_t dropoutStateSize;
CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize));
void* dropoutStates = myGpuMalloc(dropoutStateSize);

cudnnDropoutDescriptor_t dropout_desc;
CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
CUDNN_CALL(cudnnSetDropoutDescriptor(
    dropout_desc, cudnnHandle, 0.0, dropoutStates, dropoutStateSize, time(NULL)));

cudnnRNNDescriptor_t rnn_desc;
CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
CUDNN_CALL(cudnnSetRNNDescriptor(
    cudnnHandle, rnn_desc,
    /*hiddenSize*/ 1024, /*numLayers*/ 1,
    dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
    CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH));
int32_t seqLength = x324;
int32_t batchSize = x186;
int32_t inputSize = x383;

cudnnTensorDescriptor_t dx_descs[seqLength];
cudnnTensorDescriptor_t dx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&dx_desc));
int x_dims[] = {batchSize, inputSize, 1};
int x_strides[] = {x_dims[1] * x_dims[2], x_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    dx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims, x_strides));
for (int i = 0; i < seqLength; i++) {
  dx_descs[i] = dx_desc;
}

// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
// The second dimension must match the first dimension of the tensors described in xDesc.
// The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
cudnnTensorDescriptor_t hx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc));
int hx_dims[] = {2, batchSize, 1024};
int hx_strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    hx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims, hx_strides));

cudnnTensorDescriptor_t cx_desc = hx_desc;

size_t paramsSize;
CUDNN_CALL(cudnnGetRNNParamsSize(
    cudnnHandle, rnn_desc, dx_descs[0], &paramsSize, CUDNN_DATA_FLOAT));
#ifdef DEBUG
assert(paramsSize / sizeof(float) == 4198400 && "Expected parameter size mismatch");
#endif

cudnnFilterDescriptor_t w_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
int w_dims[] = {int(paramsSize / sizeof(float)), 1, 1};
CUDNN_CALL(cudnnSetFilterNdDescriptor(
    w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims));

cudnnTensorDescriptor_t y_descs[seqLength];
cudnnTensorDescriptor_t y_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
int y_dims[] = {batchSize, 2048, 1};
int y_strides[] = {y_dims[1] * y_dims[2], y_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    y_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims, y_strides));
for (int i = 0; i < seqLength; i++) {
  y_descs[i] = y_desc;
}

cudnnTensorDescriptor_t dhx_desc = hx_desc;
cudnnTensorDescriptor_t hy_desc = hx_desc;
cudnnTensorDescriptor_t dhy_desc = hy_desc;

cudnnTensorDescriptor_t dcx_desc = cx_desc;
cudnnTensorDescriptor_t cy_desc = cx_desc;
cudnnTensorDescriptor_t dcy_desc = cy_desc;

size_t workspaceSize;
CUDNN_CALL(cudnnGetRNNWorkspaceSize(
    cudnnHandle, rnn_desc, seqLength, dx_descs, &workspaceSize));
void* workspace = myGpuMalloc(workspaceSize);
CUDNN_CALL(cudnnRNNBackwardData(
    cudnnHandle, rnn_desc, seqLength, y_descs, x400, y_descs, x404,
    dhy_desc, NULL, dcy_desc, NULL, w_desc, x85, hx_desc, x507,
    cx_desc, x508, dx_descs, x396, dhx_desc, NULL, dcx_desc, NULL,
    workspace, workspaceSize, x401, x402));
myGpuFree(workspaceSize);
};
float* x510 = (float*)NULL;

{
size_t dropoutStateSize;
CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize));
void* dropoutStates = myGpuMalloc(dropoutStateSize);

cudnnDropoutDescriptor_t dropout_desc;
CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
CUDNN_CALL(cudnnSetDropoutDescriptor(
    dropout_desc, cudnnHandle, 0.0, dropoutStates, dropoutStateSize, time(NULL)));

cudnnRNNDescriptor_t rnn_desc;
CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
CUDNN_CALL(cudnnSetRNNDescriptor(
    cudnnHandle, rnn_desc,
    /*hiddenSize*/ 1024, /*numLayers*/ 1,
    dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
    CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH));
int32_t seqLength = x324;
int32_t batchSize = x186;
int32_t inputSize = x383;

cudnnTensorDescriptor_t x_descs[seqLength];
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
int x_dims[] = {batchSize, inputSize, 1};
int x_strides[] = {x_dims[1] * x_dims[2], x_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    x_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims, x_strides));
for (int i = 0; i < seqLength; i++) {
  x_descs[i] = x_desc;
}

// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
// The second dimension must match the first dimension of the tensors described in xDesc.
// The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
cudnnTensorDescriptor_t hx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc));
int hx_dims[] = {2, batchSize, 1024};
int hx_strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    hx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims, hx_strides));

size_t paramsSize;
CUDNN_CALL(cudnnGetRNNParamsSize(
    cudnnHandle, rnn_desc, x_descs[0], &paramsSize, CUDNN_DATA_FLOAT));
#ifdef DEBUG
assert(paramsSize / sizeof(float) == 4198400 && "Expected parameter size mismatch");
#endif

cudnnFilterDescriptor_t dw_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&dw_desc));
int w_dims[] = {int(paramsSize / sizeof(float)), 1, 1};
CUDNN_CALL(cudnnSetFilterNdDescriptor(
    dw_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims));

cudnnTensorDescriptor_t y_descs[seqLength];
cudnnTensorDescriptor_t y_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
int y_dims[] = {batchSize, 2048, 1};
int y_strides[] = {y_dims[1] * y_dims[2], y_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    y_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims, y_strides));
for (int i = 0; i < seqLength; i++) {
  y_descs[i] = y_desc;
}

size_t workspaceSize;
CUDNN_CALL(cudnnGetRNNWorkspaceSize(
    cudnnHandle, rnn_desc, seqLength, x_descs, &workspaceSize));
void* workspace = myGpuMalloc(workspaceSize);
CUDNN_CALL(cudnnRNNBackwardWeights(
    cudnnHandle, rnn_desc, seqLength, x_descs, x390, hx_desc, x510,
    y_descs, x400, workspace, workspaceSize,
    dw_desc, x87, x401, x402));
myGpuFree(workspaceSize);
};
// backprop for sum on dim op
sum_grad<<<28, 512>>>(x382, x324, x186, 2, x383, x386, x396, x387, x383, 1, 2);
;
float* x514 = (float*)NULL;
float* x515 = (float*)NULL;

{
size_t dropoutStateSize;
CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize));
void* dropoutStates = myGpuMalloc(dropoutStateSize);

cudnnDropoutDescriptor_t dropout_desc;
CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
CUDNN_CALL(cudnnSetDropoutDescriptor(
    dropout_desc, cudnnHandle, 0.0, dropoutStates, dropoutStateSize, time(NULL)));

cudnnRNNDescriptor_t rnn_desc;
CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
CUDNN_CALL(cudnnSetRNNDescriptor(
    cudnnHandle, rnn_desc,
    /*hiddenSize*/ 1024, /*numLayers*/ 1,
    dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
    CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH));
int32_t seqLength = x324;
int32_t batchSize = x186;
int32_t inputSize = x348;

cudnnTensorDescriptor_t dx_descs[seqLength];
cudnnTensorDescriptor_t dx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&dx_desc));
int x_dims[] = {batchSize, inputSize, 1};
int x_strides[] = {x_dims[1] * x_dims[2], x_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    dx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims, x_strides));
for (int i = 0; i < seqLength; i++) {
  dx_descs[i] = dx_desc;
}

// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
// The second dimension must match the first dimension of the tensors described in xDesc.
// The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
cudnnTensorDescriptor_t hx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc));
int hx_dims[] = {2, batchSize, 1024};
int hx_strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    hx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims, hx_strides));

cudnnTensorDescriptor_t cx_desc = hx_desc;

size_t paramsSize;
CUDNN_CALL(cudnnGetRNNParamsSize(
    cudnnHandle, rnn_desc, dx_descs[0], &paramsSize, CUDNN_DATA_FLOAT));
#ifdef DEBUG
assert(paramsSize / sizeof(float) == 3477504 && "Expected parameter size mismatch");
#endif

cudnnFilterDescriptor_t w_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
int w_dims[] = {int(paramsSize / sizeof(float)), 1, 1};
CUDNN_CALL(cudnnSetFilterNdDescriptor(
    w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims));

cudnnTensorDescriptor_t y_descs[seqLength];
cudnnTensorDescriptor_t y_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
int y_dims[] = {batchSize, 2048, 1};
int y_strides[] = {y_dims[1] * y_dims[2], y_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    y_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims, y_strides));
for (int i = 0; i < seqLength; i++) {
  y_descs[i] = y_desc;
}

cudnnTensorDescriptor_t dhx_desc = hx_desc;
cudnnTensorDescriptor_t hy_desc = hx_desc;
cudnnTensorDescriptor_t dhy_desc = hy_desc;

cudnnTensorDescriptor_t dcx_desc = cx_desc;
cudnnTensorDescriptor_t cy_desc = cx_desc;
cudnnTensorDescriptor_t dcy_desc = cy_desc;

size_t workspaceSize;
CUDNN_CALL(cudnnGetRNNWorkspaceSize(
    cudnnHandle, rnn_desc, seqLength, dx_descs, &workspaceSize));
void* workspace = myGpuMalloc(workspaceSize);
CUDNN_CALL(cudnnRNNBackwardData(
    cudnnHandle, rnn_desc, seqLength, y_descs, x378, y_descs, x382,
    dhy_desc, NULL, dcy_desc, NULL, w_desc, x49, hx_desc, x514,
    cx_desc, x515, dx_descs, x370, dhx_desc, NULL, dcx_desc, NULL,
    workspace, workspaceSize, x379, x380));
myGpuFree(workspaceSize);
};
float* x517 = (float*)NULL;

{
size_t dropoutStateSize;
CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize));
void* dropoutStates = myGpuMalloc(dropoutStateSize);

cudnnDropoutDescriptor_t dropout_desc;
CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
CUDNN_CALL(cudnnSetDropoutDescriptor(
    dropout_desc, cudnnHandle, 0.0, dropoutStates, dropoutStateSize, time(NULL)));

cudnnRNNDescriptor_t rnn_desc;
CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
CUDNN_CALL(cudnnSetRNNDescriptor(
    cudnnHandle, rnn_desc,
    /*hiddenSize*/ 1024, /*numLayers*/ 1,
    dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
    CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_TENSOR_OP_MATH));
int32_t seqLength = x324;
int32_t batchSize = x186;
int32_t inputSize = x348;

cudnnTensorDescriptor_t x_descs[seqLength];
cudnnTensorDescriptor_t x_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
int x_dims[] = {batchSize, inputSize, 1};
int x_strides[] = {x_dims[1] * x_dims[2], x_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    x_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims, x_strides));
for (int i = 0; i < seqLength; i++) {
  x_descs[i] = x_desc;
}

// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
// The second dimension must match the first dimension of the tensors described in xDesc.
// The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
cudnnTensorDescriptor_t hx_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc));
int hx_dims[] = {2, batchSize, 1024};
int hx_strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    hx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims, hx_strides));

size_t paramsSize;
CUDNN_CALL(cudnnGetRNNParamsSize(
    cudnnHandle, rnn_desc, x_descs[0], &paramsSize, CUDNN_DATA_FLOAT));
#ifdef DEBUG
assert(paramsSize / sizeof(float) == 3477504 && "Expected parameter size mismatch");
#endif

cudnnFilterDescriptor_t dw_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&dw_desc));
int w_dims[] = {int(paramsSize / sizeof(float)), 1, 1};
CUDNN_CALL(cudnnSetFilterNdDescriptor(
    dw_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims));

cudnnTensorDescriptor_t y_descs[seqLength];
cudnnTensorDescriptor_t y_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
int y_dims[] = {batchSize, 2048, 1};
int y_strides[] = {y_dims[1] * y_dims[2], y_dims[2], 1};
CUDNN_CALL(cudnnSetTensorNdDescriptor(
    y_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims, y_strides));
for (int i = 0; i < seqLength; i++) {
  y_descs[i] = y_desc;
}

size_t workspaceSize;
CUDNN_CALL(cudnnGetRNNWorkspaceSize(
    cudnnHandle, rnn_desc, seqLength, x_descs, &workspaceSize));
void* workspace = myGpuMalloc(workspaceSize);
CUDNN_CALL(cudnnRNNBackwardWeights(
    cudnnHandle, rnn_desc, seqLength, x_descs, x351, hx_desc, x517,
    y_descs, x378, workspace, workspaceSize,
    dw_desc, x51, x379, x380));
myGpuFree(workspaceSize);
};
// backprop for permute WrappedArray(2, 0, 1)
int* x520 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
x520[2] = x352;
x520[0] = x348;
x520[1] = 1;
x520[3] = 1;
float* x525 = (float*)myMalloc(1 * sizeof(float));;
x525[0] = 1.0f;
int32_t x527 = x520[0];
int32_t x528 = x520[1];
int32_t x529 = x520[2];
int32_t x530 = x520[3];

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
    in_desc, CUDNN_DATA_FLOAT,
    x186, x348, x324, 1,
    x527, x528, x529, x530));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
    out_desc, CUDNN_DATA_FLOAT,
    x186, x348, x324, 1,
    x349, x324, 1, 1));

CUDNN_CALL(cudnnTransformTensor(
    cudnnHandle, x525, in_desc, x370, x525, out_desc, x345));
};
hardTanh_grad<<<28, 512>>>(x337, x345, x345, 0.0, 20.0, x327, true);
float* x533 = (float*)myMalloc(1 * sizeof(float));;
x533[0] = 0.0f;
float* x535 = (float*)myMalloc(1 * sizeof(float));;
x535[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x186, 32, x321, x324));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x186, 32, x321, x324));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 32, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x535, x535, x535, x535, in_desc, x330,
    out_desc, x345, in_desc, x336, sbmv_desc, x40,
    x42,x44, 1.0E-5, x338, x339));
};
// conv2D back-propagate
float* x539 = (float*)myMalloc(1 * sizeof(float));;
x539[0] = 1.0f;

{
cudnnFilterDescriptor_t filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    32, 32, 21, 11));

cudnnTensorDescriptor_t grad_in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x186, 32, x289, x292));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x186, 32, x321, x324));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 2, 1, 1, 1,
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
    x539, filt_desc, x37, grad_out_desc, x336,
    conv_desc, algo, ws_data, ws_size,
    x539, grad_in_desc, x314));
};
float* x542 = (float*)myMalloc(1 * sizeof(float));;
x542[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    32, 32, 21, 11));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x186, 32, x321, x324));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x186, 32, x289, x292));

cudnnConvolutionDescriptor_t conv_desc;
CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
CUDNN_CALL(cudnnSetConvolution2dDescriptor(
    conv_desc,
    0, 0, 2, 1, 1, 1,
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
    x542, in_desc, x306, grad_out_desc, x336,
    conv_desc, algo, ws_data, ws_size,
    x542, grad_filt_desc, x39));
};
hardTanh_grad<<<28, 512>>>(x306, x314, x314, 0.0, 20.0, x295, true);
float* x546 = (float*)myMalloc(1 * sizeof(float));;
x546[0] = 0.0f;
float* x548 = (float*)myMalloc(1 * sizeof(float));;
x548[0] = 1.0f;

{
cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x186, 32, x289, x292));

cudnnTensorDescriptor_t out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x186, 32, x289, x292));

cudnnTensorDescriptor_t sbmv_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    1, 32, 1, 1));

CUDNN_CALL(cudnnBatchNormalizationBackward(
    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
    x548, x548, x548, x548, in_desc, x299,
    out_desc, x314, in_desc, x305, sbmv_desc, x20,
    x22,x24, 1.0E-5, x307, x308));
};
// conv2D back-propagate
float* x552 = (float*)myMalloc(1 * sizeof(float));;
x552[0] = 1.0f;

{
cudnnFilterDescriptor_t grad_filt_desc;
CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
CUDNN_CALL(cudnnSetFilter4dDescriptor(
    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    32, 1, 41, 11));

cudnnTensorDescriptor_t grad_out_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x186, 32, x289, x292));

cudnnTensorDescriptor_t in_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    x186, 1, x268, x267));

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
    x552, in_desc, x277, grad_out_desc, x305,
    conv_desc, algo, ws_data, ws_size,
    x552, grad_filt_desc, x19));
};
// Tensor 'toCPU' invocation.
float* x556 = (float*)myMalloc(1 * sizeof(float));;
CUDA_CALL(cudaMemcpy(x556, x283, 1 * sizeof(float), cudaMemcpyDeviceToHost));
float x558 = x556[0];
x262 += x558;
float* x560 = (float*)myMalloc(1 * sizeof(float));;
x560[0] = 1.0f;
float* x562 = (float*)myMalloc(1 * sizeof(float));;
x562[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 451,32,x560,x17,451,x562, x19, 451, x17,451));
arrayFill<<<28, 512>>>(x19, 0.0f, 14432);
float* x566 = (float*)myMalloc(1 * sizeof(float));;
x566[0] = 1.0f;
float* x568 = (float*)myMalloc(1 * sizeof(float));;
x568[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 7392,32,x566,x37,7392,x568, x39, 7392, x37,7392));
arrayFill<<<28, 512>>>(x39, 0.0f, 236544);
float* x572 = (float*)myMalloc(1 * sizeof(float));;
x572[0] = 1.0f;
float* x574 = (float*)myMalloc(1 * sizeof(float));;
x574[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x572,x40,1,x574, x42, 1, x40,1));
arrayFill<<<28, 512>>>(x42, 0.0f, 32);
float* x578 = (float*)myMalloc(1 * sizeof(float));;
x578[0] = 1.0f;
float* x580 = (float*)myMalloc(1 * sizeof(float));;
x580[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x578,x43,1,x580, x44, 1, x43,1));
arrayFill<<<28, 512>>>(x44, 0.0f, 32);
float* x584 = (float*)myMalloc(1 * sizeof(float));;
x584[0] = 1.0f;
float* x586 = (float*)myMalloc(1 * sizeof(float));;
x586[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x584,x23,1,x586, x24, 1, x23,1));
arrayFill<<<28, 512>>>(x24, 0.0f, 32);
float* x590 = (float*)myMalloc(1 * sizeof(float));;
x590[0] = 1.0f;
float* x592 = (float*)myMalloc(1 * sizeof(float));;
x592[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x590,x20,1,x592, x22, 1, x20,1));
arrayFill<<<28, 512>>>(x22, 0.0f, 32);
float* x596 = (float*)myMalloc(1 * sizeof(float));;
x596[0] = 1.0f;
float* x598 = (float*)myMalloc(1 * sizeof(float));;
x598[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x596,x156,1,x598, x158, 1, x156,1));
arrayFill<<<28, 512>>>(x158, 0.0f, 1024);
float* x602 = (float*)myMalloc(1 * sizeof(float));;
x602[0] = 1.0f;
float* x604 = (float*)myMalloc(1 * sizeof(float));;
x604[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x602,x159,1,x604, x160, 1, x159,1));
arrayFill<<<28, 512>>>(x160, 0.0f, 1024);
float* x608 = (float*)myMalloc(1 * sizeof(float));;
x608[0] = 1.0f;
float* x610 = (float*)myMalloc(1 * sizeof(float));;
x610[0] = -3.0E-8f;
CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 29,1024,x608,x173,29,x610, x175, 29, x173,29));
arrayFill<<<28, 512>>>(x175, 0.0f, 29696);
int32_t x614 = x259;
int32_t x616 = x614 % x615;
bool x617 = x616 == 0;
if (x617) {
float x622 = x262;
double x618 = (double)x614;
double x619 = 100.0 * x618;
double x621 = x619 / x620;
float x623 = (float)x614;
float x624 = x622 / x623;
printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x255,x614,x191,x621,x624);
fflush(stdout);
} else {
}
int64_t x629 = (long)mallocAddr;
int64_t x630 = x629 - x251;
memset((void*)x251, 0, x630);
mallocAddr = (void*)x251;
int64_t x633 = (long)gpuMallocAddr;
int64_t x634 = x633 - x252;
cudaMemset((void*)x252, 0, x634);
gpuMallocAddr = (void*)x252;

}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x641 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x642 = x641 / 1000LL;
int64_t x644 = x641 / x643;
printf("Training completed in %ldms (%ld us/images)\n",x642,x644);
double x646 = (double)x641;
double x647 = x646 / 1000000.0;
x250[x255] = x647;
float x649 = x262;
float x651 = x649 / x650;
double x652 = (double)x651;
x249[x255] = x652;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x658 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
sort(x250, x250 + 1);
double x664 = x250[0];
int64_t x665 = (long)fopen(x0, "w");
fprintf((FILE *)x665, "unit: %s\n", "1 epoch");
for(int x667=0; x667 < 1; x667++) {
double x668 = x249[x667];
fprintf((FILE *)x665, "%lf\n", x668);

}
fprintf((FILE *)x665, "run time: %lf %lf\n", x247, x664);
fclose((FILE*)x665);
// Backend cleanup.
CUBLAS_CALL(cublasDestroy(cublasHandle));
CUDA_CALL(cudaFree(gpuMallocBase));
      
CUDNN_CALL(cudnnDestroy(cudnnHandle));
}
/*****************************************
  End of C Generated Code                  
*******************************************/


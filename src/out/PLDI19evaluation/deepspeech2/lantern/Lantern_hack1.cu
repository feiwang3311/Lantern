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

// only for 4D tensor in and 3D tensor out (TODO: incorrect!)
__global__ void sum_optimization(float* in, int inStr0, int inStr1, int inStr2, int inStr3,
        float* out, int outStr0, int outStr1, int outStr2,
        int dim, int nElementOut, int dimSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < nElementOut; i += stride) {
        int outOff0 = i / outStr0;
        int outOff1temp = i - outOff0 * outStr0;
        int outOff1 = outOff1temp / outStr1;
        int outOff2 = outOff1temp - outOff1 * outStr1;
        for (int j = 0; j < dimSize; j++) {
            int inOff; 
            if (dim == 0) inOff = j * inStr0 + outOff0 * inStr1 + outOff1 * inStr2 + outOff2 * inStr3;
            if (dim == 1) inOff = outOff0 * inStr0 + j * inStr1 + outOff1 * inStr2 + outOff2 * inStr3;
            if (dim == 2) inOff = outOff0 * inStr0 + outOff1 * inStr1 + j * inStr2 + outOff2 * inStr3;
            if (dim == 3) inOff = outOff0 * inStr0 + outOff1 * inStr1 + outOff2 * inStr2 + j * inStr3;
            out[i] += in[inOff];
        }
    }
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
    float* x48 = (float*)myMalloc(3477504 * sizeof(float));;
    for(int x50=0; x50 < 3477504; x50++) {
        float x51 = (float)rand()/RAND_MAX;
        float x52 = x51 - 0.5f;
        float x53 = x52 * 0.01f;
        x48[x50] = x53;

    }
    // Tensor 'toGPU' invocation.
    float* x58 = (float*)myGpuMalloc(3477504 * sizeof(float));
    CUDA_CALL(cudaMemcpy(x58, x48, 3477504 * sizeof(float), cudaMemcpyHostToDevice));
    float* x60 = (float*)myGpuMalloc(3477504 * sizeof(float));
    int32_t x61 = 0;
    int32_t x62 = x61;
    float* x63 = x58+x62;
    float* x64 = x60+x62;
    x61 += 688128;
    int32_t x66 = x61;
    float* x67 = x58+x66;
    float* x68 = x60+x66;
    x61 += 1048576;
    int32_t x70 = x61;
    float* x71 = x58+x70;
    float* x72 = x60+x70;
    x61 += 1024;
    int32_t x74 = x61;
    float* x75 = x58+x74;
    float* x76 = x60+x74;
    x61 += 1024;
    int32_t x78 = x61;
    float* x79 = x58+x78;
    float* x80 = x60+x78;
    x61 += 688128;
    int32_t x82 = x61;
    float* x83 = x58+x82;
    float* x84 = x60+x82;
    x61 += 1048576;
    int32_t x86 = x61;
    float* x87 = x58+x86;
    float* x88 = x60+x86;
    x61 += 1024;
    int32_t x90 = x61;
    float* x91 = x58+x90;
    float* x92 = x60+x90;
    x61 += 1024;
    float* x94 = (float*)myMalloc(4198400 * sizeof(float));;
    for(int x96=0; x96 < 4198400; x96++) {
        float x97 = (float)rand()/RAND_MAX;
        float x98 = x97 - 0.5f;
        float x99 = x98 * 0.01f;
        x94[x96] = x99;

    }
    // Tensor 'toGPU' invocation.
    float* x104 = (float*)myGpuMalloc(4198400 * sizeof(float));
    CUDA_CALL(cudaMemcpy(x104, x94, 4198400 * sizeof(float), cudaMemcpyHostToDevice));
    float* x106 = (float*)myGpuMalloc(4198400 * sizeof(float));
    int32_t x107 = 0;
    int32_t x108 = x107;
    float* x109 = x104+x108;
    float* x110 = x106+x108;
    x107 += 1048576;
    int32_t x112 = x107;
    float* x113 = x104+x112;
    float* x114 = x106+x112;
    x107 += 1048576;
    int32_t x116 = x107;
    float* x117 = x104+x116;
    float* x118 = x106+x116;
    x107 += 1024;
    int32_t x120 = x107;
    float* x121 = x104+x120;
    float* x122 = x106+x120;
    x107 += 1024;
    int32_t x124 = x107;
    float* x125 = x104+x124;
    float* x126 = x106+x124;
    x107 += 1048576;
    int32_t x128 = x107;
    float* x129 = x104+x128;
    float* x130 = x106+x128;
    x107 += 1048576;
    int32_t x132 = x107;
    float* x133 = x104+x132;
    float* x134 = x106+x132;
    x107 += 1024;
    int32_t x136 = x107;
    float* x137 = x104+x136;
    float* x138 = x106+x136;
    x107 += 1024;
    float* x140 = (float*)myMalloc(4198400 * sizeof(float));;
    for(int x141=0; x141 < 4198400; x141++) {
        float x142 = (float)rand()/RAND_MAX;
        float x143 = x142 - 0.5f;
        float x144 = x143 * 0.01f;
        x140[x141] = x144;

    }
    // Tensor 'toGPU' invocation.
    float* x149 = (float*)myGpuMalloc(4198400 * sizeof(float));
    CUDA_CALL(cudaMemcpy(x149, x140, 4198400 * sizeof(float), cudaMemcpyHostToDevice));
    float* x151 = (float*)myGpuMalloc(4198400 * sizeof(float));
    int32_t x152 = 0;
    int32_t x153 = x152;
    float* x154 = x149+x153;
    float* x155 = x151+x153;
    x152 += 1048576;
    int32_t x157 = x152;
    float* x158 = x149+x157;
    float* x159 = x151+x157;
    x152 += 1048576;
    int32_t x161 = x152;
    float* x162 = x149+x161;
    float* x163 = x151+x161;
    x152 += 1024;
    int32_t x165 = x152;
    float* x166 = x149+x165;
    float* x167 = x151+x165;
    x152 += 1024;
    int32_t x169 = x152;
    float* x170 = x149+x169;
    float* x171 = x151+x169;
    x152 += 1048576;
    int32_t x173 = x152;
    float* x174 = x149+x173;
    float* x175 = x151+x173;
    x152 += 1048576;
    int32_t x177 = x152;
    float* x178 = x149+x177;
    float* x179 = x151+x177;
    x152 += 1024;
    int32_t x181 = x152;
    float* x182 = x149+x181;
    float* x183 = x151+x181;
    x152 += 1024;
    float* x185 = (float*)myGpuMalloc(1024 * sizeof(float));
    arrayFill<<<28, 512>>>(x185, 1.0f, 1024);
    float* x187 = (float*)myGpuMalloc(1024 * sizeof(float));
    float* x188 = (float*)myGpuMalloc(1024 * sizeof(float));
    float* x189 = (float*)myGpuMalloc(1024 * sizeof(float));
    float* x190 = (float*)myGpuMalloc(1024 * sizeof(float));
    float* x191 = (float*)myGpuMalloc(1024 * sizeof(float));
    float* x192 = (float*)myMalloc(29696 * sizeof(float));;
    for(int x194=0; x194 < 29696; x194++) {
        float x195 = (float)rand()/RAND_MAX;
        float x196 = x195 - 0.5f;
        float x197 = x196 * 0.03125f;
        x192[x194] = x197;

    }
    // Tensor 'toGPU' invocation.
    float* x202 = (float*)myGpuMalloc(29696 * sizeof(float));
    CUDA_CALL(cudaMemcpy(x202, x192, 29696 * sizeof(float), cudaMemcpyHostToDevice));
    float* x204 = (float*)myGpuMalloc(29696 * sizeof(float));
    float* x205 = (float*)myGpuMalloc(14432 * sizeof(float));
    float* x206 = (float*)myGpuMalloc(236544 * sizeof(float));
    float* x207 = (float*)myGpuMalloc(32 * sizeof(float));
    float* x208 = (float*)myGpuMalloc(32 * sizeof(float));
    float* x209 = (float*)myGpuMalloc(32 * sizeof(float));
    float* x210 = (float*)myGpuMalloc(32 * sizeof(float));
    float* x211 = (float*)myGpuMalloc(1024 * sizeof(float));
    float* x212 = (float*)myGpuMalloc(1024 * sizeof(float));
    float* x213 = (float*)myGpuMalloc(29696 * sizeof(float));
    float* x214 = (float*)myGpuMalloc(1048576 * sizeof(float));
    float* x215 = (float*)myGpuMalloc(1048576 * sizeof(float));
    float* x216 = (float*)myGpuMalloc(1024 * sizeof(float));
    float* x217 = (float*)myGpuMalloc(1024 * sizeof(float));
    float* x218 = (float*)myGpuMalloc(1048576 * sizeof(float));
    float* x219 = (float*)myGpuMalloc(1048576 * sizeof(float));
    float* x220 = (float*)myGpuMalloc(1024 * sizeof(float));
    float* x221 = (float*)myGpuMalloc(1024 * sizeof(float));
    float* x222 = (float*)myGpuMalloc(1048576 * sizeof(float));
    float* x223 = (float*)myGpuMalloc(1048576 * sizeof(float));
    float* x224 = (float*)myGpuMalloc(1024 * sizeof(float));
    float* x225 = (float*)myGpuMalloc(1048576 * sizeof(float));
    float* x226 = (float*)myGpuMalloc(1024 * sizeof(float));
    float* x227 = (float*)myGpuMalloc(1048576 * sizeof(float));
    float* x228 = (float*)myGpuMalloc(1024 * sizeof(float));
    float* x229 = (float*)myGpuMalloc(1024 * sizeof(float));
    float* x230 = (float*)myGpuMalloc(1024 * sizeof(float));
    float* x231 = (float*)myGpuMalloc(688128 * sizeof(float));
    float* x232 = (float*)myGpuMalloc(688128 * sizeof(float));
    float* x233 = (float*)myGpuMalloc(1024 * sizeof(float));
    float* x234 = (float*)myGpuMalloc(1048576 * sizeof(float));
    float* x235 = (float*)myGpuMalloc(1024 * sizeof(float));
    float* x236 = (float*)myGpuMalloc(1024 * sizeof(float));
    float* x237 = (float*)myGpuMalloc(1048576 * sizeof(float));
    int32_t x238 = open("/scratch-ml00/wang603/deepspeechData/deepspeech_train.bin",0);
    int64_t x239 = fsize(x238);
    printf("file size is %ld\n",x239);
    char* x241 = (char*)mmap(0, x239, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x238, 0);
    int64_t x242 = (long)x241;
    int64_t x243 = x242;
    int64_t x244 = x243;
    int* x245 = (int32_t*) x244;
    int64_t x246 = (int64_t)4;
    x243 += x246;
    int32_t x248 = x245[0];
    //    int64_t x249 = x243;
    //    int* x250 = (int32_t*) x249;
    x243 += x246;
    //    int32_t x252 = x250[0];
    printf("data size is %d batches, %d batch size\n",200,x248);
    int* x255 = (int32_t*)myMalloc(200 * sizeof(int32_t));;
    int* x256 = (int32_t*)myMalloc(200 * sizeof(int32_t));;
    float** x257 = (float**)myMalloc(200 * sizeof(float*));;
    float** x258 = (float**)myMalloc(200 * sizeof(float*));;
    int** x259 = (int**)myMalloc(200 * sizeof(int*));;
    int** x260 = (int**)myMalloc(200 * sizeof(int*));;
    // load data by batchs
    int32_t x286 = 4 * x248;
    int64_t x287 = (int64_t)x286;
    for(int x263=0; x263 < 200; x263++) {
        int64_t x264 = x243;
        int* x265 = (int32_t*) x264;
        x243 += x246;
        int32_t x267 = x265[0];
        x255[x263] = x267;
        int64_t x269 = x243;
        int* x270 = (int32_t*) x269;
        x243 += x246;
        int32_t x272 = x270[0];
        x256[x263] = x272;
        int32_t x274 = x255[x263];
        int32_t x276 = x256[x263];
        int64_t x278 = x243;
        float* x279 = (float*) x278;
        int32_t x275 = x248 * x274;
        int32_t x277 = x275 * x276;
        int32_t x280 = 4 * x277;
        int64_t x281 = (int64_t)x280;
        x243 += x281;
        x257[x263] = x279;
        int64_t x284 = x243;
        float* x285 = (float*) x284;
        x243 += x287;
        x258[x263] = x285;
        int64_t x290 = x243;
        int* x291 = (int32_t*) x290;
        x243 += x287;
        x259[x263] = x291;
        int* x294 = x259[x263];
        int* x295 = x259[x263];
        int32_t x296 = accumulate(x294, x295 + x248, 0);
        int64_t x297 = x243;
        int* x298 = (int32_t*) x297;
        int32_t x299 = 4 * x296;
        int64_t x300 = (int64_t)x299;
        x243 += x300;
        x260[x263] = x298;

    }
    gettimeofday(&end_0, NULL);
    timeval_subtract(&diff_0, &end_0, &begin_0);;
    int64_t x307 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
    float x308 = (float)x307;
    float x309 = x308 / 1000000.0f;
    printf("Data reading (all prepare time) in %lf sec\n",x309);
    double* x311 = (double*)myMalloc(1 * sizeof(double));;
    double* x312 = (double*)myMalloc(1 * sizeof(double));;

    // training loop starts here
    int32_t x358 = x248 * 32;
    int32_t x450 = 2048 / 2;
    int32_t x454 = x248 * x450;
    int32_t x451 = 2 * x450;
    int32_t x452 = x248 * x451;
    int32_t x657 = x248 * 20;
    int32_t x253 = x248 * 200;
    double x662 = (double)x253;
    int64_t x685 = (int64_t)x253;
    float x692 = (float)x253;
    for(int x317=0; x317 < 1; x317++) {
        struct timeval begin_1, end_1, diff_1;
        int32_t x319 = 0;
        int32_t x320 = x319;
        int32_t x321 = x320;
        float x322 = 0.0f;
        float x323 = x322;
        float x324 = x323;
        int32_t x325 = x317 + 1;
        printf("Start training epoch %d\n",x325);

        // RNN descriptors refactored
        size_t dropoutStateSize_4;
        CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize_4));
        void* dropoutStates_4 = NULL;

        cudnnDropoutDescriptor_t dropout_desc_4;
        CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc_4));
        CUDNN_CALL(cudnnSetDropoutDescriptor(
                    dropout_desc_4, cudnnHandle, 0.0, dropoutStates_4, dropoutStateSize_4, time(NULL)));

        cudnnRNNDescriptor_t rnn_desc_4;
        CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc_4));
        CUDNN_CALL(cudnnSetRNNDescriptor(
                    cudnnHandle, rnn_desc_4,
                    /*hiddenSize*/ 1024, /*numLayers*/ 1,
                    dropout_desc_4, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
                    CUDNN_RNN_RELU, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));         
        CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc_4, CUDNN_TENSOR_OP_MATH));

        int batchSize_4 = 32;
        int inputSize_4 = 672;
        int hiddenSize_4 = 1024;

        cudnnTensorDescriptor_t x_desc_4;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc_4));
        int x_dims_4[] = {batchSize_4, inputSize_4, 1};
        int x_strides_4[] = {x_dims_4[1] * x_dims_4[2], x_dims_4[2], 1};
        CUDNN_CALL(cudnnSetTensorNdDescriptor(
                    x_desc_4, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims_4, x_strides_4));

        size_t paramsSize_4;
        CUDNN_CALL(cudnnGetRNNParamsSize(
                    cudnnHandle, rnn_desc_4, x_desc_4, &paramsSize_4, CUDNN_DATA_FLOAT));
        //#ifdef DEBUG
        //            assert(paramsSize_4 / sizeof(float) == 3477504 && "Expected parameter size mismatch");
        //#endif
        cudnnFilterDescriptor_t w_desc_4;
        CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc_4));
        int w_dims_4[] = {int(paramsSize_4 / sizeof(float)), 1, 1};
        CUDNN_CALL(cudnnSetFilterNdDescriptor(
                    w_desc_4, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims_4));

        cudnnTensorDescriptor_t x_desc_5;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc_5));
        int x_dims_5[] = {batchSize_4, hiddenSize_4, 1};
        int x_strides_5[] = {x_dims_5[1] * x_dims_5[2], x_dims_5[2], 1};
        CUDNN_CALL(cudnnSetTensorNdDescriptor(
                    x_desc_5, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims_5, x_strides_5));

        size_t paramsSize_5;
        CUDNN_CALL(cudnnGetRNNParamsSize(
                    cudnnHandle, rnn_desc_4, x_desc_5, &paramsSize_5, CUDNN_DATA_FLOAT));
        //#ifdef DEBUG
        //            assert(paramsSize_5 / sizeof(float) == 4198400 && "Expected parameter size mismatch");
        //#endif

        cudnnFilterDescriptor_t w_desc_5;
        CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc_5));
        int w_dims_5[] = {int(paramsSize_5 / sizeof(float)), 1, 1};
        CUDNN_CALL(cudnnSetFilterNdDescriptor(
                    w_desc_5, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims_5));


        cudnnTensorDescriptor_t hx_desc_4;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc_4));
        int hx_dims_4[] = {2, batchSize_4, 1024};
        int hx_strides_4[] = {hx_dims_4[1] * hx_dims_4[2], hx_dims_4[2], 1};
        CUDNN_CALL(cudnnSetTensorNdDescriptor(
                    hx_desc_4, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims_4, hx_strides_4));

        cudnnTensorDescriptor_t y_desc_4;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc_4));
        int y_dims_4[] = {batchSize_4, 2048, 1};
        int y_strides_4[] = {y_dims_4[1] * y_dims_4[2], y_dims_4[2], 1};
        CUDNN_CALL(cudnnSetTensorNdDescriptor(
                    y_desc_4, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims_4, y_strides_4));

        int seqLength_4 = 630;
        cudnnTensorDescriptor_t x_descs_4[seqLength_4];
        for (int i = 0; i < seqLength_4; i++) {
            x_descs_4[i] = x_desc_4;
        }

        cudnnTensorDescriptor_t y_descs_4[seqLength_4];
        for (int i = 0; i < seqLength_4; i++) {
            y_descs_4[i] = y_desc_4;
        }

        cudnnTensorDescriptor_t x_descs_5[seqLength_4];
        for (int i = 0; i < seqLength_4; i++) {
            x_descs_5[i] = x_desc_5;
        }

        cudnnTensorDescriptor_t y_descs_5[seqLength_4];
        for (int i = 0; i < seqLength_4; i++) {
            y_descs_5[i] = y_desc_4;
        }

        size_t workspaceSize_x;
        CUDNN_CALL(cudnnGetRNNWorkspaceSize(
                    cudnnHandle, rnn_desc_4, seqLength_4, x_descs_5, &workspaceSize_x));
        void* workspace_x = myGpuMalloc(workspaceSize_x);

        size_t reserveSize_x1;
        CUDNN_CALL(cudnnGetRNNTrainingReserveSize(
                    cudnnHandle, rnn_desc_4, seqLength_4, x_descs_4, &reserveSize_x1));
        void* reserveSpace_x1 = myGpuMalloc(reserveSize_x1);

        size_t reserveSize_x2;
        CUDNN_CALL(cudnnGetRNNTrainingReserveSize(
                    cudnnHandle, rnn_desc_4, seqLength_4, x_descs_5, &reserveSize_x2));
        void* reserveSpace_x2 = myGpuMalloc(reserveSize_x2);

        size_t reserveSize_x3;
        CUDNN_CALL(cudnnGetRNNTrainingReserveSize(
                    cudnnHandle, rnn_desc_4, seqLength_4, x_descs_5, &reserveSize_x3));
        void* reserveSpace_x3 = myGpuMalloc(reserveSize_x3);

        // CNN/batchNorm descriptor refactored
        cudnnTensorDescriptor_t in_desc_0;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_0));

        cudnnFilterDescriptor_t filt_desc_0;
        CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc_0));
        CUDNN_CALL(cudnnSetFilter4dDescriptor(
                    filt_desc_0, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                    32, 1, 41, 11));

        cudnnTensorDescriptor_t out_desc_0;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_0));

        cudnnConvolutionDescriptor_t conv_desc_0;
        CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_0));
        CUDNN_CALL(cudnnSetConvolution2dDescriptor(
                    conv_desc_0,
                    0, 0, 2, 2, 1, 1,
                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc_0, CUDNN_TENSOR_OP_MATH));;

        cudnnTensorDescriptor_t in_desc_1;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_1));

        cudnnTensorDescriptor_t out_desc_1;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_1));

        cudnnTensorDescriptor_t sbmv_desc_1;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc_1));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(
                    sbmv_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                    1, 32, 1, 1));

        cudnnTensorDescriptor_t in_desc_2;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_2));

        cudnnFilterDescriptor_t filt_desc_2;
        CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc_2));
        CUDNN_CALL(cudnnSetFilter4dDescriptor(
                    filt_desc_2, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                    32, 32, 21, 11));

        cudnnTensorDescriptor_t out_desc_2;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_2));

        cudnnConvolutionDescriptor_t conv_desc_2;
        CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_2));
        CUDNN_CALL(cudnnSetConvolution2dDescriptor(
                    conv_desc_2,
                    0, 0, 2, 1, 1, 1,
                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc_2, CUDNN_TENSOR_OP_MATH));;

        cudnnTensorDescriptor_t in_desc_3;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_3));

        cudnnTensorDescriptor_t out_desc_3;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_3));

        cudnnTensorDescriptor_t sbmv_desc_3;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc_3));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(
                    sbmv_desc_3, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                    1, 32, 1, 1));

        cudnnTensorDescriptor_t in_desc_7;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_7));

        cudnnTensorDescriptor_t sbmv_desc_7;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc_7));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(
                    sbmv_desc_7, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                    1, 1024, 1, 1));

        // Other Workspace.
        size_t ws_size_1 = 234248;
        void *ws_data_1 = myGpuMalloc(ws_size_1);

        cudnnTensorDescriptor_t in_desc_trans;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_trans));
        cudnnTensorDescriptor_t out_desc_trans;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_trans));

        cudnnTensorDescriptor_t x_desc_soft;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc_soft));

        cudnnTensorDescriptor_t probs_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&probs_desc));

        cudnnCTCLossDescriptor_t ctc_desc;
        CUDNN_CALL(cudnnCreateCTCLossDescriptor(&ctc_desc));
        CUDNN_CALL(cudnnSetCTCLossDescriptor(ctc_desc, CUDNN_DATA_FLOAT));

        size_t wsSizeCTC = 70129408;
        void *wsCTC = myGpuMalloc(wsSizeCTC);

        cudnnTensorDescriptor_t x_desc_red;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc_red));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(
                    x_desc_red, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                    32, 1, 1, 1));

        cudnnTensorDescriptor_t out_desc_red;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_red));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(
                    out_desc_red, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                    1, 1, 1, 1));

        cudnnReduceTensorDescriptor_t reduce_desc;
        CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&reduce_desc));
        CUDNN_CALL(cudnnSetReduceTensorDescriptor(
                    reduce_desc, CUDNN_REDUCE_TENSOR_AVG, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
                    CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));

        size_t ws_size_red = 128;
        void *ws_data_red = myGpuMalloc(ws_size_red);

        float* x362 = (float*)myMalloc(1 * sizeof(float));;
        x362[0] = 0.0f;
        float* x364 = (float*)myMalloc(1 * sizeof(float));;
        x364[0] = 1.0f;

        int64_t x313 = (long)mallocAddr;
        int64_t x314 = (long)gpuMallocAddr;
        gettimeofday(&begin_1, NULL);

        // loop for one epoch
        for(int x328=0; x328 < 200; x328++) {
            int32_t x329 = x256[x328];
            int32_t x330 = x255[x328];
            float* x331 = x257[x328];
            float* x334 = x258[x328];
            int* x335 = x260[x328];
            int* x336 = x259[x328];
            x321 += x248;
            // Tensor 'toGPU' invocation.
            int32_t x332 = x330 * x329;
            int32_t x333 = x248 * x332;
            float* x339 = (float*)myGpuMalloc(x333 * sizeof(float));
            CUDA_CALL(cudaMemcpyAsync(x339, x331, x333 * sizeof(float), cudaMemcpyHostToDevice));
            float* x341 = (float*)myGpuMalloc(2 * sizeof(float));
            float* x342 = (float*)myGpuMalloc(1 * sizeof(float));
            float* x343 = (float*)myGpuMalloc(1 * sizeof(float));
            // allocate memory to save the final loss in CPU Tensor
            float* x345 = (float*)myGpuMalloc(1 * sizeof(float));
            int32_t x352 = x329 - 11;
            int32_t x353 = x352 / 2;
            int32_t x354 = x353 + 1;
            int32_t x349 = x330 - 41;
            int32_t x350 = x349 / 2;
            int32_t x351 = x350 + 1;
            int32_t x359 = x358 * x351;
            int32_t x360 = x359 * x354;
            float* x361 = (float*)myGpuMalloc(x360 * sizeof(float));

            CUDNN_CALL(cudnnSetTensor4dDescriptor(
                        in_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                        x248, 1, x330, x329));

            CUDNN_CALL(cudnnSetTensor4dDescriptor(
                        out_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                        x248, 32, x351, x354));

            CUDNN_CALL(cudnnConvolutionForward(
                        cudnnHandle,
                        x364, in_desc_0, x339, filt_desc_0, x17,
                        conv_desc_0, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, ws_data_1, ws_size_1,
                        x362, out_desc_0, x361));

            float* x368 = (float*)myGpuMalloc(x360 * sizeof(float));
            int32_t x355 = x351 * x354;
            int32_t x356 = 32 * x355;
            int32_t x357 = x248 * x356;
            float* x369 = (float*)myGpuMalloc(x357 * sizeof(float));
            float* x370 = (float*)myGpuMalloc(32 * sizeof(float));
            float* x371 = (float*)myGpuMalloc(32 * sizeof(float));

            CUDNN_CALL(cudnnSetTensor4dDescriptor(
                        in_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                        x248, 32, x351, x354));

            CUDNN_CALL(cudnnSetTensor4dDescriptor(
                        out_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                        x248, 32, x351, x354));

            CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
                        cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
                        x364, x362, in_desc_1, x361, out_desc_1, x369, sbmv_desc_1, x20,
                        x23, 0.1, x25, x26, 1.0E-5,
                        x370, x371));

            float* x378 = (float*)myGpuMalloc(x360 * sizeof(float));
            hardTanh<<<28, 512>>>(x369, x369, 0.0, 20.0, true);
            int32_t x386 = x354 - 11;
            int32_t x387 = x386 / 1;
            int32_t x388 = x387 + 1;
            int32_t x383 = x351 - 21;
            int32_t x384 = x383 / 2;
            int32_t x385 = x384 + 1;
            int32_t x392 = x358 * x385;
            int32_t x393 = x392 * x388;
            float* x394 = (float*)myGpuMalloc(x393 * sizeof(float));

            CUDNN_CALL(cudnnSetTensor4dDescriptor(
                        in_desc_2, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                        x248, 32, x351, x354));

            CUDNN_CALL(cudnnSetTensor4dDescriptor(
                        out_desc_2, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                        x248, 32, x385, x388));

            CUDNN_CALL(cudnnConvolutionForward(
                        cudnnHandle,
                        x364, in_desc_2, x369, filt_desc_2, x37,
                        conv_desc_2, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, ws_data_1, ws_size_1,
                        x362, out_desc_2, x394));

            float* x401 = (float*)myGpuMalloc(x393 * sizeof(float));
            int32_t x389 = x385 * x388;
            int32_t x390 = 32 * x389;
            int32_t x391 = x248 * x390;
            float* x402 = (float*)myGpuMalloc(x391 * sizeof(float));
            float* x403 = (float*)myGpuMalloc(32 * sizeof(float));
            float* x404 = (float*)myGpuMalloc(32 * sizeof(float));

            CUDNN_CALL(cudnnSetTensor4dDescriptor(
                        in_desc_3, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                        x248, 32, x385, x388));

            CUDNN_CALL(cudnnSetTensor4dDescriptor(
                        out_desc_3, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                        x248, 32, x385, x388));

            CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
                        cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
                        x364, x362, in_desc_3, x394, out_desc_3, x402, sbmv_desc_3, x40,
                        x43, 0.1, x45, x46, 1.0E-5,
                        x403, x404));

            float* x411 = (float*)myGpuMalloc(x393 * sizeof(float));
            hardTanh<<<28, 512>>>(x402, x402, 0.0, 20.0, true);
            // after conv ops
            int32_t x414 = 32 * x385;
            int32_t x415 = x414 * x388;
            int32_t x416 = x248 * x415;
            float* x417 = (float*)myGpuMalloc(x416 * sizeof(float));
            int* x420 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
            int32_t x418 = x248 * x414;
            x420[2] = x418;
            x420[0] = x414;
            x420[1] = 1;
            x420[3] = 1;
            int32_t x429 = x420[0];
            int32_t x430 = x420[1];
            int32_t x431 = x420[2];
            int32_t x432 = x420[3];

            CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
                        in_desc_trans, CUDNN_DATA_FLOAT,
                        x248, x414, x388, 1,
                        x415, x388, 1, 1));

            CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
                        out_desc_trans, CUDNN_DATA_FLOAT,
                        x248, x414, x388, 1,
                        x429, x430, x431, x432));

            CUDNN_CALL(cudnnTransformTensor(
                        cudnnHandle, x364, in_desc_trans, x402, x362, out_desc_trans, x417));

            int32_t x434 = x388 * x248;
            int32_t x435 = x434 * x414;
            float* x436 = (float*)myGpuMalloc(x435 * sizeof(float));
            // after resize and permute
            float* x438 = (float*)NULL;
            float* x439 = (float*)NULL;
            float* x440 = (float*)NULL;
            int32_t x443 = x434 * 2048;
            float* x444 = (float*)myGpuMalloc(x443 * sizeof(float));

            int32_t seqLength_4 = x388;
            //           int32_t batchSize_4 = x248;
            //           int32_t inputSize_4 = x414;

            CUDNN_CALL(cudnnRNNForwardTraining(
                        cudnnHandle, rnn_desc_4, seqLength_4, x_descs_4, x417,
                        hx_desc_4, x438, hx_desc_4, x439, w_desc_4, x58, y_descs_4, x444,
                        hx_desc_4, x440, hx_desc_4, NULL, workspace_x, workspaceSize_x, reserveSpace_x1, reserveSize_x1));

            float* x449 = (float*)myGpuMalloc(x443 * sizeof(float));
            int32_t x455 = x388 * x454;
            float* x456 = (float*)myGpuMalloc(x455 * sizeof(float));
            // optimization for dimension sum if size is small
            int32_t x458 = x434 * x450;
            sum_optimization<<<28, 512>>>(x444, x452, x451, x450, 1, x456, x454, x450, 1, 2, x458, 2);
            float* x460 = (float*)myGpuMalloc(x458 * sizeof(float));
            float* x461 = (float*)NULL;
            float* x462 = (float*)NULL;
            float* x463 = (float*)NULL;
            float* x464 = (float*)myGpuMalloc(x443 * sizeof(float));

            int32_t seqLength_5 = x388;
            //           int32_t batchSize_5 = x248;
            //           int32_t inputSize_5 = x450;

            CUDNN_CALL(cudnnRNNForwardTraining(
                        cudnnHandle, rnn_desc_4, seqLength_5, x_descs_5, x456,
                        hx_desc_4, x461, hx_desc_4, x462, w_desc_5, x104, y_descs_5, x464,
                        hx_desc_4, x463, hx_desc_4, NULL, workspace_x, workspaceSize_x, reserveSpace_x2, reserveSize_x2));

            float* x469 = (float*)myGpuMalloc(x443 * sizeof(float));
            float* x470 = (float*)myGpuMalloc(x455 * sizeof(float));
            // optimization for dimension sum if size is small
            sum_optimization<<<28, 512>>>(x464, x452, x451, x450, 1, x470, x454, x450, 1, 2, x458, 2);
            float* x473 = (float*)myGpuMalloc(x458 * sizeof(float));
            float* x474 = (float*)NULL;
            float* x475 = (float*)NULL;
            float* x476 = (float*)NULL;
            float* x477 = (float*)myGpuMalloc(x443 * sizeof(float));

            int32_t seqLength_6 = x388;
            //           int32_t batchSize_6 = x248;
            //           int32_t inputSize_6 = x450;

            CUDNN_CALL(cudnnRNNForwardTraining(
                        cudnnHandle, rnn_desc_4, seqLength_6, x_descs_5, x470,
                        hx_desc_4, x474, hx_desc_4, x475, w_desc_5, x149, y_descs_5, x477,
                        hx_desc_4, x476, hx_desc_4, NULL, workspace_x, workspaceSize_x, reserveSpace_x3, reserveSize_x3));

            float* x482 = (float*)myGpuMalloc(x443 * sizeof(float));
            float* x483 = (float*)myGpuMalloc(x455 * sizeof(float));
            // optimization for dimension sum if size is small
            sum_optimization<<<28, 512>>>(x477, x452, x451, x450, 1, x483, x454, x450, 1, 2, x458, 2);
            float* x486 = (float*)myGpuMalloc(x458 * sizeof(float));
            float* x489 = (float*)myGpuMalloc(x458 * sizeof(float));
            float* x490 = (float*)myGpuMalloc(1024 * sizeof(float));
            float* x491 = (float*)myGpuMalloc(1024 * sizeof(float));

            CUDNN_CALL(cudnnSetTensor4dDescriptor(
                        in_desc_7, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                        x434, x450, 1, 1));

            CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
                        cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,
                        x364, x362, in_desc_7, x483, in_desc_7, x489, sbmv_desc_7, x185,
                        x188, 0.1, x190, x191, 1.0E-5,
                        x490, x491));

            float* x498 = (float*)myGpuMalloc(x458 * sizeof(float));
            int32_t x499 = x434 * 29;
            float* x500 = (float*)myGpuMalloc(x499 * sizeof(float));
            CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 29,x434,1024,x364,x202,29,x489,1024,x362,x500,29));
            float* x506 = (float*)myGpuMalloc(x499 * sizeof(float));
            float* x513 = (float*)myGpuMalloc(x499 * sizeof(float));

            CUDNN_CALL(cudnnSetTensor4dDescriptor(
                        x_desc_soft, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                        x434, 29, 1, 1));

            CUDNN_CALL(cudnnSoftmaxForward(
                        cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                        x364, x_desc_soft, x500, x362, x_desc_soft, x513));

            float* x515 = (float*)myGpuMalloc(x499 * sizeof(float));
            // before CTC loss
            int* x517 = (int32_t*)myMalloc(x248 * sizeof(int32_t));;
            float x521 = (float)x388;
            for(int x519=0; x519 < x248; x519++) {
                float x520 = x334[x519];
                float x522 = x520 * x521;
                int32_t x523 = (int)x522;
                x517[x519] = x523;

            }
            float* x528 = (float*)myGpuMalloc(x248 * sizeof(float));

            {
                int probs_dims[] = {x388, x248, 29};
                int probs_strides[] = {probs_dims[1] * probs_dims[2], probs_dims[2], 1};
                CUDNN_CALL(cudnnSetTensorNdDescriptor(
                            probs_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, probs_dims, probs_strides));

                CUDNN_CALL(cudnnCTCLoss(
                            cudnnHandle, probs_desc, x513, x335, x336, x517,
                            x528, probs_desc, x515, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, wsCTC, wsSizeCTC));
            };
            float* x530 = (float*)myGpuMalloc(1 * sizeof(float));

            CUDNN_CALL(cudnnReduceTensor(
                        cudnnHandle, reduce_desc, nullptr, 0, ws_data_red, ws_size_red,
                        x364, x_desc_red, x528, x362, out_desc_red, x530));

            // after CTC loss
            float* x537 = (float*)myGpuMalloc(1 * sizeof(float));
            // make sure the size of loss is 1
            arrayFill<<<28, 512>>>(x537, 1.0f, 1);
            // backend is lantern.TensorDslCudnn$BackendCudnn@50648033
            CUDA_CALL(cudaMemcpyAsync(x345, x530, 1 * sizeof(float), cudaMemcpyDeviceToDevice));

            CUDNN_CALL(cudnnSoftmaxBackward(
                        cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                        x364, x_desc_soft, x513, x_desc_soft, x515,
                        x364, x_desc_soft, x506));

            // backprop of matrix-matrix-dot
            CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, x450,x434,29,x364,x202,29,x506,29,x364,x498,x450));
            CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 29,x450,x434,x364,x506,29,x489,x450,x364,x204,29));

            CUDNN_CALL(cudnnBatchNormalizationBackward(
                        cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,
                        x364, x364, x364, x364, in_desc_7, x483,
                        in_desc_7, x498, in_desc_7, x486, sbmv_desc_7, x185,
                        x187,x189, 1.0E-5, x490, x491));
            // backprop for sum on dim op
            int32_t x453 = x388 * x452;
            sum_grad<<<28, 512>>>(x482, x388, x248, 2, x450, x453, x486, x454, x450, 1, 2);
            float* x563 = (float*)NULL;
            float* x564 = (float*)NULL;
            CUDNN_CALL(cudnnRNNBackwardData(
                        cudnnHandle, rnn_desc_4, seqLength_6, y_descs_5, x477, y_descs_5, x482,
                        hx_desc_4, NULL, hx_desc_4, NULL, w_desc_5, x149, hx_desc_4, x563,
                        hx_desc_4, x564, x_descs_5, x473, hx_desc_4, NULL, hx_desc_4, NULL,
                        workspace_x, workspaceSize_x, reserveSpace_x3, reserveSize_x3));
            float* x566 = (float*)NULL;
            CUDNN_CALL(cudnnRNNBackwardWeights(
                        cudnnHandle, rnn_desc_4, seqLength_6, x_descs_5, x470, hx_desc_4, x566,
                        y_descs_5, x477, workspace_x, workspaceSize_x,
                        w_desc_5, x151, reserveSpace_x3, reserveSize_x3));
            // backprop for sum on dim op
            sum_grad<<<28, 512>>>(x469, x388, x248, 2, x450, x453, x473, x454, x450, 1, 2);
            float* x570 = (float*)NULL;
            float* x571 = (float*)NULL;
            CUDNN_CALL(cudnnRNNBackwardData(
                        cudnnHandle, rnn_desc_4, seqLength_5, y_descs_5, x464, y_descs_5, x469,
                        hx_desc_4, NULL, hx_desc_4, NULL, w_desc_5, x104, hx_desc_4, x570,
                        hx_desc_4, x571, x_descs_5, x460, hx_desc_4, NULL, hx_desc_4, NULL,
                        workspace_x, workspaceSize_x, reserveSpace_x2, reserveSize_x2));
            float* x573 = (float*)NULL;
            CUDNN_CALL(cudnnRNNBackwardWeights(
                        cudnnHandle, rnn_desc_4, seqLength_5, x_descs_5, x456, hx_desc_4, x573,
                        y_descs_5, x464, workspace_x, workspaceSize_x,
                        w_desc_5, x106, reserveSpace_x2, reserveSize_x2));
            // backprop for sum on dim op
            sum_grad<<<28, 512>>>(x449, x388, x248, 2, x450, x453, x460, x454, x450, 1, 2);
            float* x577 = (float*)NULL;
            float* x578 = (float*)NULL;
            CUDNN_CALL(cudnnRNNBackwardData(
                        cudnnHandle, rnn_desc_4, seqLength_4, y_descs_4, x444, y_descs_4, x449,
                        hx_desc_4, NULL, hx_desc_4, NULL, w_desc_4, x58, hx_desc_4, x577,
                        hx_desc_4, x578, x_descs_4, x436, hx_desc_4, NULL, hx_desc_4, NULL,
                        workspace_x, workspaceSize_x, reserveSpace_x1, reserveSize_x1));
            float* x580 = (float*)NULL;
            CUDNN_CALL(cudnnRNNBackwardWeights(
                        cudnnHandle, rnn_desc_4, seqLength_4, x_descs_4, x417, hx_desc_4, x580,
                        y_descs_4, x444, workspace_x, workspaceSize_x,
                        w_desc_4, x60, reserveSpace_x1, reserveSize_x1));
            // backprop for permute WrappedArray(2, 0, 1)
            int* x583 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
            x583[2] = x418;
            x583[0] = x414;
            x583[1] = 1;
            x583[3] = 1;
            int32_t x590 = x583[0];
            int32_t x591 = x583[1];
            int32_t x592 = x583[2];
            int32_t x593 = x583[3];

            CUDNN_CALL(cudnnTransformTensor(
                        cudnnHandle, x364, out_desc_trans, x436, x364, in_desc_trans, x411));

            hardTanh_grad<<<28, 512>>>(x402, x411, x411, 0.0, 20.0, x391, true);
            CUDNN_CALL(cudnnBatchNormalizationBackward(
                        cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
                        x364, x364, x364, x364, in_desc_3, x394,
                        out_desc_3, x411, in_desc_3, x401, sbmv_desc_3, x40,
                        x42,x44, 1.0E-5, x403, x404));
            // conv2D back-propagate
            CUDNN_CALL(cudnnConvolutionBackwardData(
                        cudnnHandle,
                        x364, filt_desc_2, x37, out_desc_2, x401,
                        conv_desc_2, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, ws_data_1, ws_size_1,
                        x364, in_desc_2, x378));

            CUDNN_CALL(cudnnConvolutionBackwardFilter(
                        cudnnHandle,
                        x364, in_desc_2, x369, out_desc_2, x401,
                        conv_desc_2, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3, ws_data_1, ws_size_1,
                        x364, filt_desc_2, x39));

            hardTanh_grad<<<28, 512>>>(x369, x378, x378, 0.0, 20.0, x357, true);
            CUDNN_CALL(cudnnBatchNormalizationBackward(
                        cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
                        x364, x364, x364, x364, in_desc_1, x361,
                        out_desc_1, x378, in_desc_1, x368, sbmv_desc_1, x20,
                        x22,x24, 1.0E-5, x370, x371));
            // conv2D back-propagate
            CUDNN_CALL(cudnnConvolutionBackwardFilter(
                        cudnnHandle,
                        x364, in_desc_0, x339, out_desc_0, x368,
                        conv_desc_0, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3, ws_data_1, ws_size_1,
                        x364, filt_desc_0, x19));

            // Tensor 'toCPU' invocation.
            float* x619 = (float*)myMalloc(1 * sizeof(float));;
            CUDA_CALL(cudaMemcpyAsync(x619, x345, 1 * sizeof(float), cudaMemcpyDeviceToHost));
            float x621 = x619[0];
            x324 += x621;
            momentum_update_1D_1D<<<28, 512>>>(x17, x19, x205, 3.0E-8, 0.9, 400.0, true, 14432);
            momentum_update_1D_1D<<<28, 512>>>(x37, x39, x206, 3.0E-8, 0.9, 400.0, true, 236544);
            momentum_update_1D_1D<<<28, 512>>>(x40, x42, x207, 3.0E-8, 0.9, 400.0, true, 32);
            momentum_update_1D_1D<<<28, 512>>>(x43, x44, x208, 3.0E-8, 0.9, 400.0, true, 32);
            momentum_update_1D_1D<<<28, 512>>>(x23, x24, x209, 3.0E-8, 0.9, 400.0, true, 32);
            momentum_update_1D_1D<<<28, 512>>>(x20, x22, x210, 3.0E-8, 0.9, 400.0, true, 32);
            momentum_update_1D_1D<<<28, 512>>>(x185, x187, x211, 3.0E-8, 0.9, 400.0, true, 1024);
            momentum_update_1D_1D<<<28, 512>>>(x188, x189, x212, 3.0E-8, 0.9, 400.0, true, 1024);
            momentum_update_1D_1D<<<28, 512>>>(x202, x204, x213, 3.0E-8, 0.9, 400.0, true, 29696);
            momentum_update_1D_1D<<<28, 512>>>(x174, x175, x214, 3.0E-8, 0.9, 400.0, true, 1048576);
            momentum_update_1D_1D<<<28, 512>>>(x170, x171, x215, 3.0E-8, 0.9, 400.0, true, 1048576);
            momentum_update_1D_1D<<<28, 512>>>(x182, x183, x216, 3.0E-8, 0.9, 400.0, true, 1024);
            momentum_update_1D_1D<<<28, 512>>>(x178, x179, x217, 3.0E-8, 0.9, 400.0, true, 1024);
            momentum_update_1D_1D<<<28, 512>>>(x154, x155, x218, 3.0E-8, 0.9, 400.0, true, 1048576);
            momentum_update_1D_1D<<<28, 512>>>(x158, x159, x219, 3.0E-8, 0.9, 400.0, true, 1048576);
            momentum_update_1D_1D<<<28, 512>>>(x166, x167, x220, 3.0E-8, 0.9, 400.0, true, 1024);
            momentum_update_1D_1D<<<28, 512>>>(x162, x163, x221, 3.0E-8, 0.9, 400.0, true, 1024);
            momentum_update_1D_1D<<<28, 512>>>(x129, x130, x222, 3.0E-8, 0.9, 400.0, true, 1048576);
            momentum_update_1D_1D<<<28, 512>>>(x125, x126, x223, 3.0E-8, 0.9, 400.0, true, 1048576);
            momentum_update_1D_1D<<<28, 512>>>(x137, x138, x224, 3.0E-8, 0.9, 400.0, true, 1024);
            momentum_update_1D_1D<<<28, 512>>>(x109, x110, x225, 3.0E-8, 0.9, 400.0, true, 1048576);
            momentum_update_1D_1D<<<28, 512>>>(x133, x134, x226, 3.0E-8, 0.9, 400.0, true, 1024);
            momentum_update_1D_1D<<<28, 512>>>(x113, x114, x227, 3.0E-8, 0.9, 400.0, true, 1048576);
            momentum_update_1D_1D<<<28, 512>>>(x117, x118, x228, 3.0E-8, 0.9, 400.0, true, 1024);
            momentum_update_1D_1D<<<28, 512>>>(x121, x122, x229, 3.0E-8, 0.9, 400.0, true, 1024);
            momentum_update_1D_1D<<<28, 512>>>(x91, x92, x230, 3.0E-8, 0.9, 400.0, true, 1024);
            momentum_update_1D_1D<<<28, 512>>>(x79, x80, x231, 3.0E-8, 0.9, 400.0, true, 688128);
            momentum_update_1D_1D<<<28, 512>>>(x63, x64, x232, 3.0E-8, 0.9, 400.0, true, 688128);
            momentum_update_1D_1D<<<28, 512>>>(x87, x88, x233, 3.0E-8, 0.9, 400.0, true, 1024);
            momentum_update_1D_1D<<<28, 512>>>(x67, x68, x234, 3.0E-8, 0.9, 400.0, true, 1048576);
            momentum_update_1D_1D<<<28, 512>>>(x71, x72, x235, 3.0E-8, 0.9, 400.0, true, 1024);
            momentum_update_1D_1D<<<28, 512>>>(x75, x76, x236, 3.0E-8, 0.9, 400.0, true, 1024);
            momentum_update_1D_1D<<<28, 512>>>(x83, x84, x237, 3.0E-8, 0.9, 400.0, true, 1048576);
            int32_t x656 = x321;
            int32_t x658 = x656 % x657;
            bool x659 = x658 == 0;
            if (x659) {
                float x664 = x324;
                double x660 = (double)x656;
                double x661 = 100.0 * x660;
                double x663 = x661 / x662;
                float x665 = (float)x656;
                float x666 = x664 / x665;
                printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x317,x656,x253,x663,x666);
                fflush(stdout);
            } else {
            }
            int64_t x671 = (long)mallocAddr;
            int64_t x672 = x671 - x313;
            memset((void*)x313, 0, x672);
            mallocAddr = (void*)x313;
            int64_t x675 = (long)gpuMallocAddr;
            int64_t x676 = x675 - x314;
            cudaMemset((void*)x314, 0, x676);
            gpuMallocAddr = (void*)x314;

        }
        gettimeofday(&end_1, NULL);
        timeval_subtract(&diff_1, &end_1, &begin_1);;
        int64_t x683 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
        int64_t x684 = x683 / 1000LL;
        int64_t x686 = x683 / x685;
        printf("Training completed in %ldms (%ld us/images)\n",x684,x686);
        double x688 = (double)x683;
        double x689 = x688 / 1000000.0;
        x312[x317] = x689;
        float x691 = x324;
        float x693 = x691 / x692;
        double x694 = (double)x693;
        x311[x317] = x694;

    }
    gettimeofday(&end_0, NULL);
    timeval_subtract(&diff_0, &end_0, &begin_0);;
    //    int64_t x700 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
    sort(x312, x312 + 1);
    double x706 = x312[0];
    int64_t x707 = (long)fopen(x0, "w");
    fprintf((FILE *)x707, "unit: %s\n", "1 epoch");
    for(int x709=0; x709 < 1; x709++) {
        double x710 = x311[x709];
        fprintf((FILE *)x707, "%lf\n", x710);

    }
    fprintf((FILE *)x707, "run time: %lf %lf\n", x309, x706);
    fclose((FILE*)x707);
    // Backend cleanup.
    CUBLAS_CALL(cublasDestroy(cublasHandle));
    CUDA_CALL(cudaFree(gpuMallocBase));

    CUDNN_CALL(cudnnDestroy(cudnnHandle));
}
/*****************************************
  End of C Generated Code                  
 *******************************************/


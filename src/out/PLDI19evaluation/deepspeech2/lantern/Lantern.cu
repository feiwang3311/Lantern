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
	printf("inputSize for batchRNN is %d\n",672);
	int32_t x49 = 0;
	float* x50 = (float*)myGpuMalloc(3477504 * sizeof(float));
	arrayFill<<<28, 512>>>(x50, 0.01f, 3477504);
	float* x52 = (float*)myGpuMalloc(3477504 * sizeof(float));
	int32_t x53 = x49;
	float* x54 = x50+x53;
	float* x55 = x52+x53;
	x49 += 688128;
	int32_t x57 = x49;
	float* x58 = x50+x57;
	float* x59 = x52+x57;
	x49 += 1048576;
	int32_t x61 = x49;
	float* x62 = x50+x61;
	float* x63 = x52+x61;
	x49 += 688128;
	int32_t x65 = x49;
	float* x66 = x50+x65;
	float* x67 = x52+x65;
	x49 += 1048576;
	int32_t x69 = x49;
	float* x70 = x50+x69;
	float* x71 = x52+x69;
	x49 += 1024;
	int32_t x73 = x49;
	float* x74 = x50+x73;
	float* x75 = x52+x73;
	x49 += 1024;
	int32_t x77 = x49;
	float* x78 = x50+x77;
	float* x79 = x52+x77;
	x49 += 1024;
	int32_t x81 = x49;
	float* x82 = x50+x81;
	float* x83 = x52+x81;
	x49 += 1024;
	printf("inputSize for batchRNN is %d\n",1024);
	int32_t x86 = 0;
	float* x87 = (float*)myGpuMalloc(4198400 * sizeof(float));
	arrayFill<<<28, 512>>>(x87, 0.01f, 4198400);
	float* x89 = (float*)myGpuMalloc(4198400 * sizeof(float));
	int32_t x90 = x86;
	float* x91 = x87+x90;
	float* x92 = x89+x90;
	x86 += 1048576;
	int32_t x94 = x86;
	float* x95 = x87+x94;
	float* x96 = x89+x94;
	x86 += 1048576;
	int32_t x98 = x86;
	float* x99 = x87+x98;
	float* x100 = x89+x98;
	x86 += 1048576;
	int32_t x102 = x86;
	float* x103 = x87+x102;
	float* x104 = x89+x102;
	x86 += 1048576;
	int32_t x106 = x86;
	float* x107 = x87+x106;
	float* x108 = x89+x106;
	x86 += 1024;
	int32_t x110 = x86;
	float* x111 = x87+x110;
	float* x112 = x89+x110;
	x86 += 1024;
	int32_t x114 = x86;
	float* x115 = x87+x114;
	float* x116 = x89+x114;
	x86 += 1024;
	int32_t x118 = x86;
	float* x119 = x87+x118;
	float* x120 = x89+x118;
	x86 += 1024;
	printf("inputSize for batchRNN is %d\n",1024);
	int32_t x123 = 0;
	float* x124 = (float*)myGpuMalloc(4198400 * sizeof(float));
	arrayFill<<<28, 512>>>(x124, 0.01f, 4198400);
	float* x126 = (float*)myGpuMalloc(4198400 * sizeof(float));
	int32_t x127 = x123;
	float* x128 = x124+x127;
	float* x129 = x126+x127;
	x123 += 1048576;
	int32_t x131 = x123;
	float* x132 = x124+x131;
	float* x133 = x126+x131;
	x123 += 1048576;
	int32_t x135 = x123;
	float* x136 = x124+x135;
	float* x137 = x126+x135;
	x123 += 1048576;
	int32_t x139 = x123;
	float* x140 = x124+x139;
	float* x141 = x126+x139;
	x123 += 1048576;
	int32_t x143 = x123;
	float* x144 = x124+x143;
	float* x145 = x126+x143;
	x123 += 1024;
	int32_t x147 = x123;
	float* x148 = x124+x147;
	float* x149 = x126+x147;
	x123 += 1024;
	int32_t x151 = x123;
	float* x152 = x124+x151;
	float* x153 = x126+x151;
	x123 += 1024;
	int32_t x155 = x123;
	float* x156 = x124+x155;
	float* x157 = x126+x155;
	x123 += 1024;
	float* x159 = (float*)myGpuMalloc(1024 * sizeof(float));
	arrayFill<<<28, 512>>>(x159, 1.0f, 1024);
	float* x161 = (float*)myGpuMalloc(1024 * sizeof(float));
	float* x162 = (float*)myGpuMalloc(1024 * sizeof(float));
	float* x163 = (float*)myGpuMalloc(1024 * sizeof(float));
	float* x164 = (float*)myGpuMalloc(1024 * sizeof(float));
	float* x165 = (float*)myGpuMalloc(1024 * sizeof(float));
	float* x166 = (float*)myMalloc(29696 * sizeof(float));;
	for(int x168=0; x168 < 29696; x168++) {
		float x169 = (float)rand()/RAND_MAX;
		float x170 = x169 - 0.5f;
		float x171 = x170 * 0.03125f;
		x166[x168] = x171;

	}
	// Tensor 'toGPU' invocation.
	float* x176 = (float*)myGpuMalloc(29696 * sizeof(float));
	CUDA_CALL(cudaMemcpy(x176, x166, 29696 * sizeof(float), cudaMemcpyHostToDevice));
	float* x178 = (float*)myGpuMalloc(29696 * sizeof(float));
	int32_t x179 = open("/scratch-ml00/wang603/deepspeechData/deepspeech_train.bin",0);
	int64_t x180 = fsize(x179);
	printf("file size is %ld\n",x180);
	char* x182 = (char*)mmap(0, x180, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x179, 0);
	int64_t x183 = (long)x182;
	int64_t x184 = x183;
	int64_t x185 = x184;
	int* x186 = (int32_t*) x185;
	int64_t x187 = (int64_t)4;
	x184 += x187;
	int32_t x189 = x186[0];
	int64_t x190 = x184;
	int* x191 = (int32_t*) x190;
	x184 += x187;
	int32_t x193 = x191[0];
	printf("data size is %d batches, %d batch size\n",200,x189);
	int* x196 = (int32_t*)myMalloc(200 * sizeof(int32_t));;
	int* x197 = (int32_t*)myMalloc(200 * sizeof(int32_t));;
	float** x198 = (float**)myMalloc(200 * sizeof(float*));;
	float** x199 = (float**)myMalloc(200 * sizeof(float*));;
	int** x200 = (int**)myMalloc(200 * sizeof(int*));;
	int** x201 = (int**)myMalloc(200 * sizeof(int*));;
	// load data by batchs
	int32_t x227 = 4 * x189;
	int64_t x228 = (int64_t)x227;
	for(int x204=0; x204 < 200; x204++) {
		int64_t x205 = x184;
		int* x206 = (int32_t*) x205;
		x184 += x187;
		int32_t x208 = x206[0];
		x196[x204] = x208;
		int64_t x210 = x184;
		int* x211 = (int32_t*) x210;
		x184 += x187;
		int32_t x213 = x211[0];
		x197[x204] = x213;
		int32_t x215 = x196[x204];
		int32_t x217 = x197[x204];
		int64_t x219 = x184;
		float* x220 = (float*) x219;
		int32_t x216 = x189 * x215;
		int32_t x218 = x216 * x217;
		int32_t x221 = 4 * x218;
		int64_t x222 = (int64_t)x221;
		x184 += x222;
		x198[x204] = x220;
		int64_t x225 = x184;
		float* x226 = (float*) x225;
		x184 += x228;
		x199[x204] = x226;
		int64_t x231 = x184;
		int* x232 = (int32_t*) x231;
		x184 += x228;
		x200[x204] = x232;
		int* x235 = x200[x204];
		int* x236 = x200[x204];
		int32_t x237 = accumulate(x235, x236 + x189, 0);
		int64_t x238 = x184;
		int* x239 = (int32_t*) x238;
		int32_t x240 = 4 * x237;
		int64_t x241 = (int64_t)x240;
		x184 += x241;
		x201[x204] = x239;

	}
	gettimeofday(&end_0, NULL);
	timeval_subtract(&diff_0, &end_0, &begin_0);;
	int64_t x248 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
	float x249 = (float)x248;
	float x250 = x249 / 1000000.0f;
	printf("Data reading (all prepare time) in %lf sec\n",x250);
	double* x252 = (double*)myMalloc(1 * sizeof(double));;
	double* x253 = (double*)myMalloc(1 * sizeof(double));;
	int64_t x254 = (long)mallocAddr;
	int64_t x255 = (long)gpuMallocAddr;
	// training loop starts here
	int32_t x299 = x189 * 32;
	int32_t x386 = 2048 / 2;
	int32_t x387 = 2 * x386;
	int32_t x388 = x189 * x387;
	int32_t x390 = x189 * x386;
	int32_t x618 = x189 * 20;
	int32_t x194 = x189 * 200;
	double x623 = (double)x194;
	int64_t x646 = (int64_t)x194;
	float x653 = (float)x194;
	for(int x258=0; x258 < 1; x258++) {
		struct timeval begin_1, end_1, diff_1;
		int32_t x260 = 0;
		int32_t x261 = x260;
		int32_t x262 = x261;
		float x263 = 0.0f;
		float x264 = x263;
		float x265 = x264;
		int32_t x266 = x258 + 1;
		printf("Start training epoch %d\n",x266);
		gettimeofday(&begin_1, NULL);
		for(int x269=0; x269 < 200; x269++) {
			x269 = 199;
			int32_t x270 = x197[x269];
			int32_t x271 = x196[x269];
			float* x272 = x198[x269];
			float* x275 = x199[x269];
			int* x276 = x201[x269];
			int* x277 = x200[x269];
			x262 += x189;
			// Tensor 'toGPU' invocation.
			int32_t x273 = x271 * x270;
			int32_t x274 = x189 * x273;
			float* x280 = (float*)myGpuMalloc(x274 * sizeof(float));
			CUDA_CALL(cudaMemcpy(x280, x272, x274 * sizeof(float), cudaMemcpyHostToDevice));
			float* x282 = (float*)myGpuMalloc(2 * sizeof(float));
			float* x283 = (float*)myGpuMalloc(1 * sizeof(float));
			float* x284 = (float*)myGpuMalloc(1 * sizeof(float));
			// allocate memory to save the final loss in CPU Tensor
			float* x286 = (float*)myGpuMalloc(1 * sizeof(float));
			int32_t x293 = x270 - 11;
			int32_t x294 = x293 / 2;
			int32_t x295 = x294 + 1;
			int32_t x290 = x271 - 41;
			int32_t x291 = x290 / 2;
			int32_t x292 = x291 + 1;
			int32_t x300 = x299 * x292;
			int32_t x301 = x300 * x295;
			float* x302 = (float*)myGpuMalloc(x301 * sizeof(float));
			float* x303 = (float*)myMalloc(1 * sizeof(float));;
			x303[0] = 0.0f;
			float* x305 = (float*)myMalloc(1 * sizeof(float));;
			x305[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x189, 1, x271, x270));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							32, 1, 41, 11));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x189, 32, x292, x295));

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
							x305, in_desc, x280, filt_desc, x17,
							conv_desc, algo, ws_data, ws_size,
							x303, out_desc, x302));
			};
			float* x308 = (float*)myGpuMalloc(x301 * sizeof(float));
			int32_t x296 = x292 * x295;
			int32_t x297 = 32 * x296;
			int32_t x298 = x189 * x297;
			float* x309 = (float*)myGpuMalloc(x298 * sizeof(float));
			float* x310 = (float*)myGpuMalloc(32 * sizeof(float));
			float* x311 = (float*)myGpuMalloc(32 * sizeof(float));
			float* x312 = (float*)myMalloc(1 * sizeof(float));;
			x312[0] = 0.0f;
			float* x314 = (float*)myMalloc(1 * sizeof(float));;
			x314[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x189, 32, x292, x295));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x189, 32, x292, x295));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 32, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x314, x312, in_desc, x302, out_desc, x309, sbmv_desc, x20,
							x23, 0.1, x25, x26, 1.0E-5,
							x310, x311));
			};
			float* x317 = (float*)myGpuMalloc(x301 * sizeof(float));
			hardTanh<<<28, 512>>>(x309, x309, 0.0, 20.0, true);
			int32_t x325 = x295 - 11;
			int32_t x326 = x325 / 1;
			int32_t x327 = x326 + 1;
			int32_t x322 = x292 - 21;
			int32_t x323 = x322 / 2;
			int32_t x324 = x323 + 1;
			int32_t x331 = x299 * x324;
			int32_t x332 = x331 * x327;
			float* x333 = (float*)myGpuMalloc(x332 * sizeof(float));
			float* x334 = (float*)myMalloc(1 * sizeof(float));;
			x334[0] = 0.0f;
			float* x336 = (float*)myMalloc(1 * sizeof(float));;
			x336[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x189, 32, x292, x295));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							32, 32, 21, 11));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x189, 32, x324, x327));

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
							x336, in_desc, x309, filt_desc, x37,
							conv_desc, algo, ws_data, ws_size,
							x334, out_desc, x333));
			};
			float* x339 = (float*)myGpuMalloc(x332 * sizeof(float));
			int32_t x328 = x324 * x327;
			int32_t x329 = 32 * x328;
			int32_t x330 = x189 * x329;
			float* x340 = (float*)myGpuMalloc(x330 * sizeof(float));
			float* x341 = (float*)myGpuMalloc(32 * sizeof(float));
			float* x342 = (float*)myGpuMalloc(32 * sizeof(float));
			float* x343 = (float*)myMalloc(1 * sizeof(float));;
			x343[0] = 0.0f;
			float* x345 = (float*)myMalloc(1 * sizeof(float));;
			x345[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x189, 32, x324, x327));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x189, 32, x324, x327));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 32, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x345, x343, in_desc, x333, out_desc, x340, sbmv_desc, x40,
							x43, 0.1, x45, x46, 1.0E-5,
							x341, x342));
			};
			float* x348 = (float*)myGpuMalloc(x332 * sizeof(float));
			hardTanh<<<28, 512>>>(x340, x340, 0.0, 20.0, true);
			// after conv ops
			int32_t x351 = 32 * x324;
			int32_t x352 = x351 * x327;
			int32_t x353 = x189 * x352;
			float* x354 = (float*)myGpuMalloc(x353 * sizeof(float));
			int* x357 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
			int32_t x355 = x189 * x351;
			x357[2] = x355;
			x357[0] = x351;
			x357[1] = 1;
			x357[3] = 1;
			float* x362 = (float*)myMalloc(1 * sizeof(float));;
			x362[0] = 1.0f;
			float* x364 = (float*)myMalloc(0 * sizeof(float));;
			x364[0] = 0.0f;
			int32_t x366 = x357[0];
			int32_t x367 = x357[1];
			int32_t x368 = x357[2];
			int32_t x369 = x357[3];

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
							in_desc, CUDNN_DATA_FLOAT,
							x189, x351, x327, 1,
							x352, x327, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
							out_desc, CUDNN_DATA_FLOAT,
							x189, x351, x327, 1,
							x366, x367, x368, x369));

				CUDNN_CALL(cudnnTransformTensor(
							cudnnHandle, x362, in_desc, x340, x364, out_desc, x354));
			};
			int32_t x371 = x327 * x189;
			int32_t x372 = x371 * x351;
			float* x373 = (float*)myGpuMalloc(x372 * sizeof(float));
			// after resize and permute
			float* x375 = (float*)NULL;
			float* x376 = (float*)NULL;
			float* x377 = (float*)NULL;
			int32_t x380 = x371 * 2048;
			float* x381 = (float*)myGpuMalloc(x380 * sizeof(float));
			float* x382 = (float*)NULL;
			int32_t x383 = 0;

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
				int32_t seqLength = x327;
				int32_t batchSize = x189;
				int32_t inputSize = x351;

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
				x382 = (float*)reserveSpace;
				x383 = (int)reserveSize;
				void* workspace = myGpuMalloc(workspaceSize);
				CUDNN_CALL(cudnnRNNForwardTraining(
							cudnnHandle, rnn_desc, seqLength, x_descs, x354,
							hx_desc,x375, cx_desc,x376, w_desc, x50, y_descs, x381,
							hy_desc,x377, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
				myGpuFree(workspaceSize);
			};
			float* x385 = (float*)myGpuMalloc(x380 * sizeof(float));
			int32_t x392 = x371 * x386;
			float* x393 = (float*)myGpuMalloc(x392 * sizeof(float));
			float* x394 = (float*)myMalloc(1 * sizeof(float));;
			x394[0] = 0.0f;
			float* x396 = (float*)myMalloc(1 * sizeof(float));;
			x396[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x327, x189, 2, x386));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x327, x189, 1, x386));

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
							x396, x_desc, x381, x394, out_desc, x393));
			};
			float* x399 = (float*)myGpuMalloc(x392 * sizeof(float));
			float* x400 = (float*)NULL;
			float* x401 = (float*)NULL;
			float* x402 = (float*)NULL;
			float* x403 = (float*)myGpuMalloc(x380 * sizeof(float));
			float* x404 = (float*)NULL;
			int32_t x405 = 0;

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
				int32_t seqLength = x327;
				int32_t batchSize = x189;
				int32_t inputSize = x386;

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
				x404 = (float*)reserveSpace;
				x405 = (int)reserveSize;
				void* workspace = myGpuMalloc(workspaceSize);
				CUDNN_CALL(cudnnRNNForwardTraining(
							cudnnHandle, rnn_desc, seqLength, x_descs, x393,
							hx_desc,x400, cx_desc,x401, w_desc, x87, y_descs, x403,
							hy_desc,x402, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
				myGpuFree(workspaceSize);
			};
			float* x407 = (float*)myGpuMalloc(x380 * sizeof(float));
			float* x408 = (float*)myGpuMalloc(x392 * sizeof(float));
			float* x409 = (float*)myMalloc(1 * sizeof(float));;
			x409[0] = 0.0f;
			float* x411 = (float*)myMalloc(1 * sizeof(float));;
			x411[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x327, x189, 2, x386));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x327, x189, 1, x386));

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
							x411, x_desc, x403, x409, out_desc, x408));
			};
			float* x414 = (float*)myGpuMalloc(x392 * sizeof(float));
			float* x415 = (float*)NULL;
			float* x416 = (float*)NULL;
			float* x417 = (float*)NULL;
			float* x418 = (float*)myGpuMalloc(x380 * sizeof(float));
			float* x419 = (float*)NULL;
			int32_t x420 = 0;

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
				int32_t seqLength = x327;
				int32_t batchSize = x189;
				int32_t inputSize = x386;

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
				x419 = (float*)reserveSpace;
				x420 = (int)reserveSize;
				void* workspace = myGpuMalloc(workspaceSize);
				CUDNN_CALL(cudnnRNNForwardTraining(
							cudnnHandle, rnn_desc, seqLength, x_descs, x408,
							hx_desc,x415, cx_desc,x416, w_desc, x124, y_descs, x418,
							hy_desc,x417, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
				myGpuFree(workspaceSize);
			};
			float* x422 = (float*)myGpuMalloc(x380 * sizeof(float));
			float* x423 = (float*)myGpuMalloc(x392 * sizeof(float));
			float* x424 = (float*)myMalloc(1 * sizeof(float));;
			x424[0] = 0.0f;
			float* x426 = (float*)myMalloc(1 * sizeof(float));;
			x426[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x327, x189, 2, x386));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x327, x189, 1, x386));

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
							x426, x_desc, x418, x424, out_desc, x423));
			};
			float* x429 = (float*)myGpuMalloc(x392 * sizeof(float));
			// after RNN layers
			// after maybe lookahead
			float* x434 = (float*)myGpuMalloc(x392 * sizeof(float));
			float* x435 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x436 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x437 = (float*)myMalloc(1 * sizeof(float));;
			x437[0] = 0.0f;
			float* x439 = (float*)myMalloc(1 * sizeof(float));;
			x439[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x371, x386, 1, 1));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 1024, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,
							x439, x437, in_desc, x423, in_desc, x434, sbmv_desc, x159,
							x162, 0.1, x164, x165, 1.0E-5,
							x435, x436));
			};
			float* x442 = (float*)myGpuMalloc(x392 * sizeof(float));
			int32_t x443 = x371 * 29;
			float* x444 = (float*)myGpuMalloc(x443 * sizeof(float));
			float* x445 = (float*)myMalloc(1 * sizeof(float));;
			x445[0] = 0.0f;
			float* x447 = (float*)myMalloc(1 * sizeof(float));;
			x447[0] = 1.0f;
			CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 29,x371,1024,x447,x176,29,x434,1024,x445,x444,29));
			float* x450 = (float*)myGpuMalloc(x443 * sizeof(float));
			float* x453 = (float*)myMalloc(1 * sizeof(float));;
			x453[0] = 0.0f;
			float* x455 = (float*)myMalloc(1 * sizeof(float));;
			x455[0] = 1.0f;
			float* x457 = (float*)myGpuMalloc(x443 * sizeof(float));

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x371, 29, 1, 1));
				CUDNN_CALL(cudnnSoftmaxForward(
							cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
							x455, x_desc, x444, x453, x_desc, x457));
			};
			float* x459 = (float*)myGpuMalloc(x443 * sizeof(float));
			// before CTC loss
			int* x461 = (int32_t*)myMalloc(x189 * sizeof(int32_t));;
			float x465 = (float)x327;
			for(int x463=0; x463 < x189; x463++) {
				float x464 = x275[x463];
				float x466 = x464 * x465;
				int32_t x467 = (int)x466;
				x461[x463] = x467;

			}
			float* x472 = (float*)myGpuMalloc(x189 * sizeof(float));

			{
				cudnnTensorDescriptor_t probs_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&probs_desc));
				int probs_dims[] = {x327, x189, 29};
				int probs_strides[] = {probs_dims[1] * probs_dims[2], probs_dims[2], 1};
				CUDNN_CALL(cudnnSetTensorNdDescriptor(
							probs_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, probs_dims, probs_strides));

				cudnnTensorDescriptor_t grad_desc = probs_desc;

				cudnnCTCLossDescriptor_t ctc_desc;
				CUDNN_CALL(cudnnCreateCTCLossDescriptor(&ctc_desc));
				CUDNN_CALL(cudnnSetCTCLossDescriptor(ctc_desc, CUDNN_DATA_FLOAT));
				size_t wsSize;
				CUDNN_CALL(cudnnGetCTCLossWorkspaceSize(
							cudnnHandle, probs_desc, grad_desc, x276, x277, x461,
							CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, &wsSize));
				void *ws = myGpuMalloc(wsSize);

				CUDNN_CALL(cudnnCTCLoss(
							cudnnHandle, probs_desc, x457, x276, x277, x461,
							x472, grad_desc, x459, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, ws, wsSize));
			};
			float* x474 = (float*)myGpuMalloc(1 * sizeof(float));
			float* x475 = (float*)myMalloc(1 * sizeof(float));;
			x475[0] = 0.0f;
			float* x477 = (float*)myMalloc(1 * sizeof(float));;
			x477[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x189, 1, 1, 1));

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
							x477, x_desc, x472, x475, out_desc, x474));
			};
			// after CTC loss
			float* x481 = (float*)myGpuMalloc(1 * sizeof(float));
			// make sure the size of loss is 1
			arrayFill<<<28, 512>>>(x481, 1.0f, 1);
			// backend is lantern.TensorDslCudnn$BackendCudnn@1aaad5b
			CUDA_CALL(cudaMemcpy(x286, x474, 1 * sizeof(float), cudaMemcpyDeviceToDevice));
			float* x486 = (float*)myMalloc(1 * sizeof(float));;
			x486[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x371, 29, 1, 1));
				CUDNN_CALL(cudnnSoftmaxBackward(
							cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
							x486, x_desc, x457, x_desc, x459,
							x486, x_desc, x450));
			};
			float* x489 = (float*)myMalloc(1 * sizeof(float));;
			x489[0] = 0.0f;
			float* x491 = (float*)myMalloc(1 * sizeof(float));;
			x491[0] = 1.0f;
			// backprop of matrix-matrix-dot
			CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, x386,x371,29,x491,x176,29,x450,29,x491,x442,x386));
			CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 29,x386,x371,x491,x450,29,x434,x386,x491,x178,29));
			float* x496 = (float*)myMalloc(1 * sizeof(float));;
			x496[0] = 0.0f;
			float* x498 = (float*)myMalloc(1 * sizeof(float));;
			x498[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x371, x386, 1, 1));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 1024, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,
							x498, x498, x498, x498, in_desc, x423,
							in_desc, x442, in_desc, x429, sbmv_desc, x159,
							x161,x163, 1.0E-5, x435, x436));
			};
			// backprop for sum on dim op
			int32_t x389 = x327 * x388;
			sum_grad<<<28, 512>>>(x422, x327, x189, 2, x386, x389, x429, x390, x386, 1, 2);
			;
			float* x503 = (float*)NULL;
			float* x504 = (float*)NULL;

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
				int32_t seqLength = x327;
				int32_t batchSize = x189;
				int32_t inputSize = x386;

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
							cudnnHandle, rnn_desc, seqLength, y_descs, x418, y_descs, x422,
							dhy_desc, NULL, dcy_desc, NULL, w_desc, x124, hx_desc, x503,
							cx_desc, x504, dx_descs, x414, dhx_desc, NULL, dcx_desc, NULL,
							workspace, workspaceSize, x419, x420));
				myGpuFree(workspaceSize);
			};
			float* x506 = (float*)NULL;

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
				int32_t seqLength = x327;
				int32_t batchSize = x189;
				int32_t inputSize = x386;

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
							cudnnHandle, rnn_desc, seqLength, x_descs, x408, hx_desc, x506,
							y_descs, x418, workspace, workspaceSize,
							dw_desc, x126, x419, x420));
				myGpuFree(workspaceSize);
			};
			// backprop for sum on dim op
			sum_grad<<<28, 512>>>(x407, x327, x189, 2, x386, x389, x414, x390, x386, 1, 2);
			;
			float* x510 = (float*)NULL;
			float* x511 = (float*)NULL;

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
				int32_t seqLength = x327;
				int32_t batchSize = x189;
				int32_t inputSize = x386;

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
							cudnnHandle, rnn_desc, seqLength, y_descs, x403, y_descs, x407,
							dhy_desc, NULL, dcy_desc, NULL, w_desc, x87, hx_desc, x510,
							cx_desc, x511, dx_descs, x399, dhx_desc, NULL, dcx_desc, NULL,
							workspace, workspaceSize, x404, x405));
				myGpuFree(workspaceSize);
			};
			float* x513 = (float*)NULL;

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
				int32_t seqLength = x327;
				int32_t batchSize = x189;
				int32_t inputSize = x386;

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
							cudnnHandle, rnn_desc, seqLength, x_descs, x393, hx_desc, x513,
							y_descs, x403, workspace, workspaceSize,
							dw_desc, x89, x404, x405));
				myGpuFree(workspaceSize);
			};
			// backprop for sum on dim op
			sum_grad<<<28, 512>>>(x385, x327, x189, 2, x386, x389, x399, x390, x386, 1, 2);
			;
			float* x517 = (float*)NULL;
			float* x518 = (float*)NULL;

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
				int32_t seqLength = x327;
				int32_t batchSize = x189;
				int32_t inputSize = x351;

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
							cudnnHandle, rnn_desc, seqLength, y_descs, x381, y_descs, x385,
							dhy_desc, NULL, dcy_desc, NULL, w_desc, x50, hx_desc, x517,
							cx_desc, x518, dx_descs, x373, dhx_desc, NULL, dcx_desc, NULL,
							workspace, workspaceSize, x382, x383));
				myGpuFree(workspaceSize);
			};
			float* x520 = (float*)NULL;

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
				int32_t seqLength = x327;
				int32_t batchSize = x189;
				int32_t inputSize = x351;

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
							cudnnHandle, rnn_desc, seqLength, x_descs, x354, hx_desc, x520,
							y_descs, x381, workspace, workspaceSize,
							dw_desc, x52, x382, x383));
				myGpuFree(workspaceSize);
			};
			// backprop for permute WrappedArray(2, 0, 1)
			int* x523 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
			x523[2] = x355;
			x523[0] = x351;
			x523[1] = 1;
			x523[3] = 1;
			float* x528 = (float*)myMalloc(1 * sizeof(float));;
			x528[0] = 1.0f;
			int32_t x530 = x523[0];
			int32_t x531 = x523[1];
			int32_t x532 = x523[2];
			int32_t x533 = x523[3];

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
							in_desc, CUDNN_DATA_FLOAT,
							x189, x351, x327, 1,
							x530, x531, x532, x533));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
							out_desc, CUDNN_DATA_FLOAT,
							x189, x351, x327, 1,
							x352, x327, 1, 1));

				CUDNN_CALL(cudnnTransformTensor(
							cudnnHandle, x528, in_desc, x373, x528, out_desc, x348));
			};
			hardTanh_grad<<<28, 512>>>(x340, x348, x348, 0.0, 20.0, x330, true);
			float* x536 = (float*)myMalloc(1 * sizeof(float));;
			x536[0] = 0.0f;
			float* x538 = (float*)myMalloc(1 * sizeof(float));;
			x538[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x189, 32, x324, x327));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x189, 32, x324, x327));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 32, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x538, x538, x538, x538, in_desc, x333,
							out_desc, x348, in_desc, x339, sbmv_desc, x40,
							x42,x44, 1.0E-5, x341, x342));
			};
			// conv2D back-propagate
			float* x542 = (float*)myMalloc(1 * sizeof(float));;
			x542[0] = 1.0f;

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
							x189, 32, x292, x295));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x189, 32, x324, x327));

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
							x542, filt_desc, x37, grad_out_desc, x339,
							conv_desc, algo, ws_data, ws_size,
							x542, grad_in_desc, x317));
			};
			float* x545 = (float*)myMalloc(1 * sizeof(float));;
			x545[0] = 1.0f;

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
							x189, 32, x324, x327));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x189, 32, x292, x295));

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
							x545, in_desc, x309, grad_out_desc, x339,
							conv_desc, algo, ws_data, ws_size,
							x545, grad_filt_desc, x39));
			};
			hardTanh_grad<<<28, 512>>>(x309, x317, x317, 0.0, 20.0, x298, true);
			float* x549 = (float*)myMalloc(1 * sizeof(float));;
			x549[0] = 0.0f;
			float* x551 = (float*)myMalloc(1 * sizeof(float));;
			x551[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x189, 32, x292, x295));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x189, 32, x292, x295));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 32, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x551, x551, x551, x551, in_desc, x302,
							out_desc, x317, in_desc, x308, sbmv_desc, x20,
							x22,x24, 1.0E-5, x310, x311));
			};
			// conv2D back-propagate
			float* x555 = (float*)myMalloc(1 * sizeof(float));;
			x555[0] = 1.0f;

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
							x189, 32, x292, x295));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x189, 1, x271, x270));

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
							x555, in_desc, x280, grad_out_desc, x308,
							conv_desc, algo, ws_data, ws_size,
							x555, grad_filt_desc, x19));
			};
			// Tensor 'toCPU' invocation.
			float* x559 = (float*)myMalloc(1 * sizeof(float));;
			CUDA_CALL(cudaMemcpy(x559, x286, 1 * sizeof(float), cudaMemcpyDeviceToHost));
			float x561 = x559[0];
			x265 += x561;
			float* x563 = (float*)myMalloc(1 * sizeof(float));;
			x563[0] = 1.0f;
			float* x565 = (float*)myMalloc(1 * sizeof(float));;
			x565[0] = -3.0E-8f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 451,32,x563,x17,451,x565, x19, 451, x17,451));
			arrayFill<<<28, 512>>>(x19, 0.0f, 14432);
			float* x569 = (float*)myMalloc(1 * sizeof(float));;
			x569[0] = 1.0f;
			float* x571 = (float*)myMalloc(1 * sizeof(float));;
			x571[0] = -3.0E-8f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 7392,32,x569,x37,7392,x571, x39, 7392, x37,7392));
			arrayFill<<<28, 512>>>(x39, 0.0f, 236544);
			float* x575 = (float*)myMalloc(1 * sizeof(float));;
			x575[0] = 1.0f;
			float* x577 = (float*)myMalloc(1 * sizeof(float));;
			x577[0] = -3.0E-8f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x575,x40,1,x577, x42, 1, x40,1));
			arrayFill<<<28, 512>>>(x42, 0.0f, 32);
			float* x581 = (float*)myMalloc(1 * sizeof(float));;
			x581[0] = 1.0f;
			float* x583 = (float*)myMalloc(1 * sizeof(float));;
			x583[0] = -3.0E-8f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x581,x43,1,x583, x44, 1, x43,1));
			arrayFill<<<28, 512>>>(x44, 0.0f, 32);
			float* x587 = (float*)myMalloc(1 * sizeof(float));;
			x587[0] = 1.0f;
			float* x589 = (float*)myMalloc(1 * sizeof(float));;
			x589[0] = -3.0E-8f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x587,x23,1,x589, x24, 1, x23,1));
			arrayFill<<<28, 512>>>(x24, 0.0f, 32);
			float* x593 = (float*)myMalloc(1 * sizeof(float));;
			x593[0] = 1.0f;
			float* x595 = (float*)myMalloc(1 * sizeof(float));;
			x595[0] = -3.0E-8f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x593,x20,1,x595, x22, 1, x20,1));
			arrayFill<<<28, 512>>>(x22, 0.0f, 32);
			float* x599 = (float*)myMalloc(1 * sizeof(float));;
			x599[0] = 1.0f;
			float* x601 = (float*)myMalloc(1 * sizeof(float));;
			x601[0] = -3.0E-8f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x599,x159,1,x601, x161, 1, x159,1));
			arrayFill<<<28, 512>>>(x161, 0.0f, 1024);
			float* x605 = (float*)myMalloc(1 * sizeof(float));;
			x605[0] = 1.0f;
			float* x607 = (float*)myMalloc(1 * sizeof(float));;
			x607[0] = -3.0E-8f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x605,x162,1,x607, x163, 1, x162,1));
			arrayFill<<<28, 512>>>(x163, 0.0f, 1024);
			float* x611 = (float*)myMalloc(1 * sizeof(float));;
			x611[0] = 1.0f;
			float* x613 = (float*)myMalloc(1 * sizeof(float));;
			x613[0] = -3.0E-8f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 29,1024,x611,x176,29,x613, x178, 29, x176,29));
			arrayFill<<<28, 512>>>(x178, 0.0f, 29696);
			int32_t x617 = x262;
			int32_t x619 = x617 % x618;
			bool x620 = x619 == 0;
			if (x620) {
				float x625 = x265;
				double x621 = (double)x617;
				double x622 = 100.0 * x621;
				double x624 = x622 / x623;
				float x626 = (float)x617;
				float x627 = x625 / x626;
				printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x258,x617,x194,x624,x627);
				fflush(stdout);
			} else {
			}
			int64_t x632 = (long)mallocAddr;
			int64_t x633 = x632 - x254;
			memset((void*)x254, 0, x633);
			mallocAddr = (void*)x254;
			int64_t x636 = (long)gpuMallocAddr;
			int64_t x637 = x636 - x255;
			cudaMemset((void*)x255, 0, x637);
			gpuMallocAddr = (void*)x255;

		}
		gettimeofday(&end_1, NULL);
		timeval_subtract(&diff_1, &end_1, &begin_1);;
		int64_t x644 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
		int64_t x645 = x644 / 1000LL;
		int64_t x647 = x644 / x646;
		printf("Training completed in %ldms (%ld us/images)\n",x645,x647);
		double x649 = (double)x644;
		double x650 = x649 / 1000000.0;
		x253[x258] = x650;
		float x652 = x265;
		float x654 = x652 / x653;
		double x655 = (double)x654;
		x252[x258] = x655;

	}
	gettimeofday(&end_0, NULL);
	timeval_subtract(&diff_0, &end_0, &begin_0);;
	int64_t x661 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
	sort(x253, x253 + 1);
	double x667 = x253[0];
	int64_t x668 = (long)fopen(x0, "w");
	fprintf((FILE *)x668, "unit: %s\n", "1 epoch");
	for(int x670=0; x670 < 1; x670++) {
		double x671 = x252[x670];
		fprintf((FILE *)x668, "%lf\n", x671);

	}
	fprintf((FILE *)x668, "run time: %lf %lf\n", x250, x667);
	fclose((FILE*)x668);
	// Backend cleanup.
	CUBLAS_CALL(cublasDestroy(cublasHandle));
	CUDA_CALL(cudaFree(gpuMallocBase));

	CUDNN_CALL(cudnnDestroy(cudnnHandle));
}
/*****************************************
  End of C Generated Code                  
 *******************************************/


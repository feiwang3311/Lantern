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

long HEAP_SIZE_CPU = 10737418260; // 1048576; // 536870912; // 268435456; // 2097152; 1610612739; // 4294967304; //
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
	int64_t x249 = x243;
	int* x250 = (int32_t*) x249;
	x243 += x246;
	int32_t x252 = x250[0];
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
	int64_t x313 = (long)mallocAddr;
	int64_t x314 = (long)gpuMallocAddr;
	// training loop starts here
	int32_t x358 = x248 * 32;
	int32_t x451 = 2048 / 2;
	int32_t x455 = x248 * x451;
	int32_t x452 = 2 * x451;
	int32_t x453 = x248 * x452;
	int32_t x655 = x248 * 20;
	int32_t x253 = x248 * 200;
	double x660 = (double)x253;
	int64_t x683 = (int64_t)x253;
	float x690 = (float)x253;
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
		gettimeofday(&begin_1, NULL);
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
			CUDA_CALL(cudaMemcpy(x339, x331, x333 * sizeof(float), cudaMemcpyHostToDevice));
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
			float* x362 = (float*)myMalloc(1 * sizeof(float));;
			x362[0] = 0.0f;
			float* x364 = (float*)myMalloc(1 * sizeof(float));;
			x364[0] = 1.0f;

			cudnnTensorDescriptor_t in_desc_0;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_0));
			CUDNN_CALL(cudnnSetTensor4dDescriptor(
						in_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
						x248, 1, x330, x329));

			cudnnFilterDescriptor_t filt_desc_0;
			CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc_0));
			CUDNN_CALL(cudnnSetFilter4dDescriptor(
						filt_desc_0, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
						32, 1, 41, 11));

			cudnnTensorDescriptor_t out_desc_0;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_0));
			CUDNN_CALL(cudnnSetTensor4dDescriptor(
						out_desc_0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
						x248, 32, x351, x354));

			cudnnConvolutionDescriptor_t conv_desc_0;
			CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_0));
			CUDNN_CALL(cudnnSetConvolution2dDescriptor(
						conv_desc_0,
						0, 0, 2, 2, 1, 1,
						CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
			CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc_0, CUDNN_TENSOR_OP_MATH));;

			// Algorithm.
			{
				cudnnConvolutionFwdAlgo_t algo;
				CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
							cudnnHandle,
							in_desc_0, filt_desc_0, conv_desc_0, out_desc_0,
							CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
							cudnnHandle, in_desc_0, filt_desc_0, conv_desc_0, out_desc_0, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				// Execute convolution.
				CUDNN_CALL(cudnnConvolutionForward(
							cudnnHandle,
							x364, in_desc_0, x339, filt_desc_0, x17,
							conv_desc_0, algo, ws_data, ws_size,
							x362, out_desc_0, x361));
			};
			float* x368 = (float*)myGpuMalloc(x360 * sizeof(float));
			int32_t x355 = x351 * x354;
			int32_t x356 = 32 * x355;
			int32_t x357 = x248 * x356;
			float* x369 = (float*)myGpuMalloc(x357 * sizeof(float));
			float* x370 = (float*)myGpuMalloc(32 * sizeof(float));
			float* x371 = (float*)myGpuMalloc(32 * sizeof(float));
			float* x372 = (float*)myMalloc(1 * sizeof(float));;
			x372[0] = 0.0f;
			float* x374 = (float*)myMalloc(1 * sizeof(float));;
			x374[0] = 1.0f;

			cudnnTensorDescriptor_t in_desc_1;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_1));
			CUDNN_CALL(cudnnSetTensor4dDescriptor(
						in_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
						x248, 32, x351, x354));

			cudnnTensorDescriptor_t out_desc_1;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_1));
			CUDNN_CALL(cudnnSetTensor4dDescriptor(
						out_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
						x248, 32, x351, x354));

			cudnnTensorDescriptor_t sbmv_desc_1;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc_1));
			CUDNN_CALL(cudnnSetTensor4dDescriptor(
						sbmv_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
						1, 32, 1, 1));

			;
			CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
						cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
						x374, x372, in_desc_1, x361, out_desc_1, x369, sbmv_desc_1, x20,
						x23, 0.1, x25, x26, 1.0E-5,
						x370, x371));
			;
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
			float* x395 = (float*)myMalloc(1 * sizeof(float));;
			x395[0] = 0.0f;
			float* x397 = (float*)myMalloc(1 * sizeof(float));;
			x397[0] = 1.0f;

			cudnnTensorDescriptor_t in_desc_2;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_2));
			CUDNN_CALL(cudnnSetTensor4dDescriptor(
						in_desc_2, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
						x248, 32, x351, x354));

			cudnnFilterDescriptor_t filt_desc_2;
			CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc_2));
			CUDNN_CALL(cudnnSetFilter4dDescriptor(
						filt_desc_2, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
						32, 32, 21, 11));

			cudnnTensorDescriptor_t out_desc_2;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_2));
			CUDNN_CALL(cudnnSetTensor4dDescriptor(
						out_desc_2, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
						x248, 32, x385, x388));

			cudnnConvolutionDescriptor_t conv_desc_2;
			CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_2));
			CUDNN_CALL(cudnnSetConvolution2dDescriptor(
						conv_desc_2,
						0, 0, 2, 1, 1, 1,
						CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
			CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc_2, CUDNN_TENSOR_OP_MATH));;

			// Algorithm.
			{
				cudnnConvolutionFwdAlgo_t algo;
				CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
							cudnnHandle,
							in_desc_2, filt_desc_2, conv_desc_2, out_desc_2,
							CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
							cudnnHandle, in_desc_2, filt_desc_2, conv_desc_2, out_desc_2, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				// Execute convolution.
				CUDNN_CALL(cudnnConvolutionForward(
							cudnnHandle,
							x397, in_desc_2, x369, filt_desc_2, x37,
							conv_desc_2, algo, ws_data, ws_size,
							x395, out_desc_2, x394));
			};
			float* x401 = (float*)myGpuMalloc(x393 * sizeof(float));
			int32_t x389 = x385 * x388;
			int32_t x390 = 32 * x389;
			int32_t x391 = x248 * x390;
			float* x402 = (float*)myGpuMalloc(x391 * sizeof(float));
			float* x403 = (float*)myGpuMalloc(32 * sizeof(float));
			float* x404 = (float*)myGpuMalloc(32 * sizeof(float));
			float* x405 = (float*)myMalloc(1 * sizeof(float));;
			x405[0] = 0.0f;
			float* x407 = (float*)myMalloc(1 * sizeof(float));;
			x407[0] = 1.0f;

			cudnnTensorDescriptor_t in_desc_3;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_3));
			CUDNN_CALL(cudnnSetTensor4dDescriptor(
						in_desc_3, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
						x248, 32, x385, x388));

			cudnnTensorDescriptor_t out_desc_3;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_3));
			CUDNN_CALL(cudnnSetTensor4dDescriptor(
						out_desc_3, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
						x248, 32, x385, x388));

			cudnnTensorDescriptor_t sbmv_desc_3;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc_3));
			CUDNN_CALL(cudnnSetTensor4dDescriptor(
						sbmv_desc_3, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
						1, 32, 1, 1));

			;
			CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
						cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
						x407, x405, in_desc_3, x394, out_desc_3, x402, sbmv_desc_3, x40,
						x43, 0.1, x45, x46, 1.0E-5,
						x403, x404));
			;
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
			float* x425 = (float*)myMalloc(1 * sizeof(float));;
			x425[0] = 1.0f;
			float* x427 = (float*)myMalloc(0 * sizeof(float));;
			x427[0] = 0.0f;
			int32_t x429 = x420[0];
			int32_t x430 = x420[1];
			int32_t x431 = x420[2];
			int32_t x432 = x420[3];

			cudnnTensorDescriptor_t in_desc_4;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_4));
			CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
						in_desc_4, CUDNN_DATA_FLOAT,
						x248, x414, x388, 1,
						x415, x388, 1, 1));

			cudnnTensorDescriptor_t out_desc_4;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_4));
			CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
						out_desc_4, CUDNN_DATA_FLOAT,
						x248, x414, x388, 1,
						x429, x430, x431, x432));

			;
			CUDNN_CALL(cudnnTransformTensor(
						cudnnHandle, x425, in_desc_4, x402, x427, out_desc_4, x417));
			;
			int32_t x435 = x388 * x248;
			int32_t x436 = x435 * x414;
			float* x437 = (float*)myGpuMalloc(x436 * sizeof(float));
			// after resize and permute
			float* x439 = (float*)NULL;
			float* x440 = (float*)NULL;
			float* x441 = (float*)NULL;
			int32_t x444 = x435 * 2048;
			float* x445 = (float*)myGpuMalloc(x444 * sizeof(float));
			float* x446 = (float*)NULL;
			int32_t x447 = 0;

			size_t dropoutStateSize_5;
			CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize_5));
			void* dropoutStates_5 = NULL;

			cudnnDropoutDescriptor_t dropout_desc_5;
			CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc_5));
			CUDNN_CALL(cudnnSetDropoutDescriptor(
						dropout_desc_5, cudnnHandle, 0.0, dropoutStates_5, dropoutStateSize_5, time(NULL)));

			cudnnRNNDescriptor_t rnn_desc_5;
			CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc_5));
			CUDNN_CALL(cudnnSetRNNDescriptor(
						cudnnHandle, rnn_desc_5,
						/*hiddenSize*/ 1024, /*numLayers*/ 1,
						dropout_desc_5, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
						CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));         
			CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc_5, CUDNN_TENSOR_OP_MATH));
			int32_t seqLength_5 = x388;
			int32_t batchSize_5 = x248;
			int32_t inputSize_5 = x414;

			cudnnTensorDescriptor_t x_descs_5[seqLength_5];
			cudnnTensorDescriptor_t x_desc_5;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc_5));
			int x_dims_5[] = {batchSize_5, inputSize_5, 1};
			int x_strides_5[] = {x_dims_5[1] * x_dims_5[2], x_dims_5[2], 1};
			CUDNN_CALL(cudnnSetTensorNdDescriptor(
						x_desc_5, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims_5, x_strides_5));
			for (int i = 0; i < seqLength_5; i++) {
				x_descs_5[i] = x_desc_5;
			}
			cudnnTensorDescriptor_t hx_desc_5;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc_5));
			int hx_dims_5[] = {2, batchSize_5, 1024};
			int hx_strides_5[] = {hx_dims_5[1] * hx_dims_5[2], hx_dims_5[2], 1};
			CUDNN_CALL(cudnnSetTensorNdDescriptor(
						hx_desc_5, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims_5, hx_strides_5));

			size_t paramsSize_5;
			CUDNN_CALL(cudnnGetRNNParamsSize(
						cudnnHandle, rnn_desc_5, x_descs_5[0], &paramsSize_5, CUDNN_DATA_FLOAT));
#ifdef DEBUG
			assert(paramsSize_5 / sizeof(float) == 3477504 && "Expected parameter size mismatch");
#endif

			cudnnFilterDescriptor_t w_desc_5;
			CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc_5));
			int w_dims_5[] = {int(paramsSize_5 / sizeof(float)), 1, 1};
			CUDNN_CALL(cudnnSetFilterNdDescriptor(
						w_desc_5, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims_5));

			cudnnTensorDescriptor_t y_descs_5[seqLength_5];
			cudnnTensorDescriptor_t y_desc_5;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc_5));
			int y_dims_5[] = {batchSize_5, 2048, 1};
			int y_strides_5[] = {y_dims_5[1] * y_dims_5[2], y_dims_5[2], 1};
			CUDNN_CALL(cudnnSetTensorNdDescriptor(
						y_desc_5, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims_5, y_strides_5));
			for (int i = 0; i < seqLength_5; i++) {
				y_descs_5[i] = y_desc_5;
			}

			size_t workspaceSize_5;
			CUDNN_CALL(cudnnGetRNNWorkspaceSize(
						cudnnHandle, rnn_desc_5, seqLength_5, x_descs_5, &workspaceSize_5));
			void* workspace_5 = myGpuMalloc(workspaceSize_5);
			;

			{// Reserve space used by `ForwardTraining` function.
				size_t reserveSize;
				CUDNN_CALL(cudnnGetRNNTrainingReserveSize(
							cudnnHandle, rnn_desc_5, seqLength_5, x_descs_5, &reserveSize));
				void* reserveSpace = myGpuMalloc(reserveSize);
				x446 = (float*)reserveSpace;
				x447 = (int)reserveSize;
				CUDNN_CALL(cudnnRNNForwardTraining(
							cudnnHandle, rnn_desc_5, seqLength_5, x_descs_5, x417,
							hx_desc_5,x439, hx_desc_5,x440, w_desc_5, x58, y_descs_5, x445,
							hx_desc_5, x441, hx_desc_5, NULL, workspace_5, workspaceSize_5, reserveSpace, reserveSize));
			};
			float* x450 = (float*)myGpuMalloc(x444 * sizeof(float));
			int32_t x456 = x388 * x455;
			float* x457 = (float*)myGpuMalloc(x456 * sizeof(float));
			// optimization for dimension sum if size is small
			int32_t x459 = x435 * x451;
			sum_optimization<<<28, 512>>>(x445, x453, x452, x451, 1, x457, x455, x451, 1, 2, x459, 2);
			;
			float* x461 = (float*)myGpuMalloc(x459 * sizeof(float));
			float* x462 = (float*)NULL;
			float* x463 = (float*)NULL;
			float* x464 = (float*)NULL;
			float* x465 = (float*)myGpuMalloc(x444 * sizeof(float));
			float* x466 = (float*)NULL;
			int32_t x467 = 0;

			size_t dropoutStateSize_6;
			CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize_6));
			void* dropoutStates_6 = NULL;

			cudnnDropoutDescriptor_t dropout_desc_6;
			CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc_6));
			CUDNN_CALL(cudnnSetDropoutDescriptor(
						dropout_desc_6, cudnnHandle, 0.0, dropoutStates_6, dropoutStateSize_6, time(NULL)));

			cudnnRNNDescriptor_t rnn_desc_6;
			CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc_6));
			CUDNN_CALL(cudnnSetRNNDescriptor(
						cudnnHandle, rnn_desc_6,
						/*hiddenSize*/ 1024, /*numLayers*/ 1,
						dropout_desc_6, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
						CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));         
			CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc_6, CUDNN_TENSOR_OP_MATH));
			int32_t seqLength_6 = x388;
			int32_t batchSize_6 = x248;
			int32_t inputSize_6 = x451;

			cudnnTensorDescriptor_t x_descs_6[seqLength_6];
			cudnnTensorDescriptor_t x_desc_6;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc_6));
			int x_dims_6[] = {batchSize_6, inputSize_6, 1};
			int x_strides_6[] = {x_dims_6[1] * x_dims_6[2], x_dims_6[2], 1};
			CUDNN_CALL(cudnnSetTensorNdDescriptor(
						x_desc_6, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims_6, x_strides_6));
			for (int i = 0; i < seqLength_6; i++) {
				x_descs_6[i] = x_desc_6;
			}
			cudnnTensorDescriptor_t hx_desc_6;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc_6));
			int hx_dims_6[] = {2, batchSize_6, 1024};
			int hx_strides_6[] = {hx_dims_6[1] * hx_dims_6[2], hx_dims_6[2], 1};
			CUDNN_CALL(cudnnSetTensorNdDescriptor(
						hx_desc_6, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims_6, hx_strides_6));

			size_t paramsSize_6;
			CUDNN_CALL(cudnnGetRNNParamsSize(
						cudnnHandle, rnn_desc_6, x_descs_6[0], &paramsSize_6, CUDNN_DATA_FLOAT));
#ifdef DEBUG
			assert(paramsSize_6 / sizeof(float) == 4198400 && "Expected parameter size mismatch");
#endif

			cudnnFilterDescriptor_t w_desc_6;
			CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc_6));
			int w_dims_6[] = {int(paramsSize_6 / sizeof(float)), 1, 1};
			CUDNN_CALL(cudnnSetFilterNdDescriptor(
						w_desc_6, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims_6));

			cudnnTensorDescriptor_t y_descs_6[seqLength_6];
			cudnnTensorDescriptor_t y_desc_6;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc_6));
			int y_dims_6[] = {batchSize_6, 2048, 1};
			int y_strides_6[] = {y_dims_6[1] * y_dims_6[2], y_dims_6[2], 1};
			CUDNN_CALL(cudnnSetTensorNdDescriptor(
						y_desc_6, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims_6, y_strides_6));
			for (int i = 0; i < seqLength_6; i++) {
				y_descs_6[i] = y_desc_6;
			}

			size_t workspaceSize_6;
			CUDNN_CALL(cudnnGetRNNWorkspaceSize(
						cudnnHandle, rnn_desc_6, seqLength_6, x_descs_6, &workspaceSize_6));
			void* workspace_6 = myGpuMalloc(workspaceSize_6);
			;

			{// Reserve space used by `ForwardTraining` function.
				size_t reserveSize;
				CUDNN_CALL(cudnnGetRNNTrainingReserveSize(
							cudnnHandle, rnn_desc_6, seqLength_6, x_descs_6, &reserveSize));
				void* reserveSpace = myGpuMalloc(reserveSize);
				x466 = (float*)reserveSpace;
				x467 = (int)reserveSize;
				CUDNN_CALL(cudnnRNNForwardTraining(
							cudnnHandle, rnn_desc_6, seqLength_6, x_descs_6, x457,
							hx_desc_6,x462, hx_desc_6,x463, w_desc_6, x104, y_descs_6, x465,
							hx_desc_6, x464, hx_desc_6, NULL, workspace_6, workspaceSize_6, reserveSpace, reserveSize));
			};
			float* x470 = (float*)myGpuMalloc(x444 * sizeof(float));
			float* x471 = (float*)myGpuMalloc(x456 * sizeof(float));
			// optimization for dimension sum if size is small
			sum_optimization<<<28, 512>>>(x465, x453, x452, x451, 1, x471, x455, x451, 1, 2, x459, 2);
			;
			float* x474 = (float*)myGpuMalloc(x459 * sizeof(float));
			float* x475 = (float*)NULL;
			float* x476 = (float*)NULL;
			float* x477 = (float*)NULL;
			float* x478 = (float*)myGpuMalloc(x444 * sizeof(float));
			float* x479 = (float*)NULL;
			int32_t x480 = 0;

			size_t dropoutStateSize_7;
			CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize_7));
			void* dropoutStates_7 = NULL;

			cudnnDropoutDescriptor_t dropout_desc_7;
			CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc_7));
			CUDNN_CALL(cudnnSetDropoutDescriptor(
						dropout_desc_7, cudnnHandle, 0.0, dropoutStates_7, dropoutStateSize_7, time(NULL)));

			cudnnRNNDescriptor_t rnn_desc_7;
			CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc_7));
			CUDNN_CALL(cudnnSetRNNDescriptor(
						cudnnHandle, rnn_desc_7,
						/*hiddenSize*/ 1024, /*numLayers*/ 1,
						dropout_desc_7, CUDNN_LINEAR_INPUT, CUDNN_BIDIRECTIONAL,
						CUDNN_RNN_TANH, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));         
			CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc_7, CUDNN_TENSOR_OP_MATH));
			int32_t seqLength_7 = x388;
			int32_t batchSize_7 = x248;
			int32_t inputSize_7 = x451;

			cudnnTensorDescriptor_t x_descs_7[seqLength_7];
			cudnnTensorDescriptor_t x_desc_7;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc_7));
			int x_dims_7[] = {batchSize_7, inputSize_7, 1};
			int x_strides_7[] = {x_dims_7[1] * x_dims_7[2], x_dims_7[2], 1};
			CUDNN_CALL(cudnnSetTensorNdDescriptor(
						x_desc_7, CUDNN_DATA_FLOAT, /*nbDims*/ 3, x_dims_7, x_strides_7));
			for (int i = 0; i < seqLength_7; i++) {
				x_descs_7[i] = x_desc_7;
			}
			cudnnTensorDescriptor_t hx_desc_7;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc_7));
			int hx_dims_7[] = {2, batchSize_7, 1024};
			int hx_strides_7[] = {hx_dims_7[1] * hx_dims_7[2], hx_dims_7[2], 1};
			CUDNN_CALL(cudnnSetTensorNdDescriptor(
						hx_desc_7, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims_7, hx_strides_7));

			size_t paramsSize_7;
			CUDNN_CALL(cudnnGetRNNParamsSize(
						cudnnHandle, rnn_desc_7, x_descs_7[0], &paramsSize_7, CUDNN_DATA_FLOAT));
#ifdef DEBUG
			assert(paramsSize_7 / sizeof(float) == 4198400 && "Expected parameter size mismatch");
#endif

			cudnnFilterDescriptor_t w_desc_7;
			CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc_7));
			int w_dims_7[] = {int(paramsSize_7 / sizeof(float)), 1, 1};
			CUDNN_CALL(cudnnSetFilterNdDescriptor(
						w_desc_7, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims_7));

			cudnnTensorDescriptor_t y_descs_7[seqLength_7];
			cudnnTensorDescriptor_t y_desc_7;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc_7));
			int y_dims_7[] = {batchSize_7, 2048, 1};
			int y_strides_7[] = {y_dims_7[1] * y_dims_7[2], y_dims_7[2], 1};
			CUDNN_CALL(cudnnSetTensorNdDescriptor(
						y_desc_7, CUDNN_DATA_FLOAT, /*nbDims*/ 3, y_dims_7, y_strides_7));
			for (int i = 0; i < seqLength_7; i++) {
				y_descs_7[i] = y_desc_7;
			}

			size_t workspaceSize_7;
			CUDNN_CALL(cudnnGetRNNWorkspaceSize(
						cudnnHandle, rnn_desc_7, seqLength_7, x_descs_7, &workspaceSize_7));
			void* workspace_7 = myGpuMalloc(workspaceSize_7);
			;

			{// Reserve space used by `ForwardTraining` function.
				size_t reserveSize;
				CUDNN_CALL(cudnnGetRNNTrainingReserveSize(
							cudnnHandle, rnn_desc_7, seqLength_7, x_descs_7, &reserveSize));
				void* reserveSpace = myGpuMalloc(reserveSize);
				x479 = (float*)reserveSpace;
				x480 = (int)reserveSize;
				CUDNN_CALL(cudnnRNNForwardTraining(
							cudnnHandle, rnn_desc_7, seqLength_7, x_descs_7, x471,
							hx_desc_7,x475, hx_desc_7,x476, w_desc_7, x149, y_descs_7, x478,
							hx_desc_7, x477, hx_desc_7, NULL, workspace_7, workspaceSize_7, reserveSpace, reserveSize));
			};
			float* x483 = (float*)myGpuMalloc(x444 * sizeof(float));
			float* x484 = (float*)myGpuMalloc(x456 * sizeof(float));
			// optimization for dimension sum if size is small
			sum_optimization<<<28, 512>>>(x478, x453, x452, x451, 1, x484, x455, x451, 1, 2, x459, 2);
			;
			float* x487 = (float*)myGpuMalloc(x459 * sizeof(float));
			float* x490 = (float*)myGpuMalloc(x459 * sizeof(float));
			float* x491 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x492 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x493 = (float*)myMalloc(1 * sizeof(float));;
			x493[0] = 0.0f;
			float* x495 = (float*)myMalloc(1 * sizeof(float));;
			x495[0] = 1.0f;

			cudnnTensorDescriptor_t in_desc_8;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_8));
			CUDNN_CALL(cudnnSetTensor4dDescriptor(
						in_desc_8, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
						x435, x451, 1, 1));

			cudnnTensorDescriptor_t sbmv_desc_8;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc_8));
			CUDNN_CALL(cudnnSetTensor4dDescriptor(
						sbmv_desc_8, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
						1, 1024, 1, 1));

			;
			CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
						cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,
						x495, x493, in_desc_8, x484, in_desc_8, x490, sbmv_desc_8, x185,
						x188, 0.1, x190, x191, 1.0E-5,
						x491, x492));
			;
			float* x499 = (float*)myGpuMalloc(x459 * sizeof(float));
			int32_t x500 = x435 * 29;
			float* x501 = (float*)myGpuMalloc(x500 * sizeof(float));
			float* x502 = (float*)myMalloc(1 * sizeof(float));;
			x502[0] = 0.0f;
			float* x504 = (float*)myMalloc(1 * sizeof(float));;
			x504[0] = 1.0f;
			CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 29,x435,1024,x504,x202,29,x490,1024,x502,x501,29));
			float* x507 = (float*)myGpuMalloc(x500 * sizeof(float));
			float* x510 = (float*)myMalloc(1 * sizeof(float));;
			x510[0] = 0.0f;
			float* x512 = (float*)myMalloc(1 * sizeof(float));;
			x512[0] = 1.0f;
			float* x514 = (float*)myGpuMalloc(x500 * sizeof(float));

			cudnnTensorDescriptor_t x_desc_9;
			CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc_9));
			CUDNN_CALL(cudnnSetTensor4dDescriptor(
						x_desc_9, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
						x435, 29, 1, 1));
			;
			CUDNN_CALL(cudnnSoftmaxForward(
						cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
						x512, x_desc_9, x501, x510, x_desc_9, x514));
			;
			float* x517 = (float*)myGpuMalloc(x500 * sizeof(float));
			// before CTC loss
			int* x519 = (int32_t*)myMalloc(x248 * sizeof(int32_t));;
			float x523 = (float)x388;
			for(int x521=0; x521 < x248; x521++) {
				float x522 = x334[x521];
				float x524 = x522 * x523;
				int32_t x525 = (int)x524;
				x519[x521] = x525;

			}
			float* x530 = (float*)myGpuMalloc(x248 * sizeof(float));

			{
				cudnnTensorDescriptor_t probs_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&probs_desc));
				int probs_dims[] = {x388, x248, 29};
				int probs_strides[] = {probs_dims[1] * probs_dims[2], probs_dims[2], 1};
				CUDNN_CALL(cudnnSetTensorNdDescriptor(
							probs_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, probs_dims, probs_strides));

				cudnnTensorDescriptor_t grad_desc = probs_desc;

				cudnnCTCLossDescriptor_t ctc_desc;
				CUDNN_CALL(cudnnCreateCTCLossDescriptor(&ctc_desc));
				CUDNN_CALL(cudnnSetCTCLossDescriptor(ctc_desc, CUDNN_DATA_FLOAT));
				size_t wsSize;
				CUDNN_CALL(cudnnGetCTCLossWorkspaceSize(
							cudnnHandle, probs_desc, grad_desc, x335, x336, x519,
							CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, &wsSize));
				void *ws = myGpuMalloc(wsSize);

				CUDNN_CALL(cudnnCTCLoss(
							cudnnHandle, probs_desc, x514, x335, x336, x519,
							x530, grad_desc, x517, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, ws, wsSize));
			};
			float* x532 = (float*)myGpuMalloc(1 * sizeof(float));
			float* x533 = (float*)myMalloc(1 * sizeof(float));;
			x533[0] = 0.0f;
			float* x535 = (float*)myMalloc(1 * sizeof(float));;
			x535[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x248, 1, 1, 1));

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
							x535, x_desc, x530, x533, out_desc, x532));
			};
			// after CTC loss
			float* x539 = (float*)myGpuMalloc(1 * sizeof(float));
			// make sure the size of loss is 1
			arrayFill<<<28, 512>>>(x539, 1.0f, 1);
			// backend is lantern.TensorDslCudnn$BackendCudnn@23fbaf4a
			CUDA_CALL(cudaMemcpy(x345, x532, 1 * sizeof(float), cudaMemcpyDeviceToDevice));
			float* x544 = (float*)myMalloc(1 * sizeof(float));;
			x544[0] = 1.0f;
			CUDNN_CALL(cudnnSoftmaxBackward(
						cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
						x544, x_desc_9, x514, x_desc_9, x517,
						x544, x_desc_9, x507));
			;
			float* x547 = (float*)myMalloc(1 * sizeof(float));;
			x547[0] = 0.0f;
			float* x549 = (float*)myMalloc(1 * sizeof(float));;
			x549[0] = 1.0f;
			// backprop of matrix-matrix-dot
			float* x552 = (float*)myMalloc(1 * sizeof(float));;
			x552[0] = 1.0f;
			CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, x451,x435,29,x552,x202,29,x507,29,x552,x499,x451));
			float* x555 = (float*)myMalloc(1 * sizeof(float));;
			x555[0] = 1.0f;
			CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 29,x451,x435,x555,x507,29,x490,x451,x555,x204,29));
			float* x558 = (float*)myMalloc(1 * sizeof(float));;
			x558[0] = 0.0f;
			float* x560 = (float*)myMalloc(1 * sizeof(float));;
			x560[0] = 1.0f;
			CUDNN_CALL(cudnnBatchNormalizationBackward(
						cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,
						x560, x560, x560, x560, in_desc_8, x484,
						in_desc_8, x499, in_desc_8, x487, sbmv_desc_8, x185,
						x187,x189, 1.0E-5, x491, x492));
			;
			// backprop for sum on dim op
			int32_t x454 = x388 * x453;
			sum_grad<<<28, 512>>>(x483, x388, x248, 2, x451, x454, x487, x455, x451, 1, 2);
			;
			float* x565 = (float*)NULL;
			float* x566 = (float*)NULL;
			CUDNN_CALL(cudnnRNNBackwardData(
						cudnnHandle, rnn_desc_7, seqLength_7, y_descs_7, x478, y_descs_7, x483,
						hx_desc_7, NULL, hx_desc_7, NULL, w_desc_7, x149, hx_desc_7, x565,
						hx_desc_7, x566, x_descs_7, x474, hx_desc_7, NULL, hx_desc_7, NULL,
						workspace_7, workspaceSize_7, x479, x480));
			;
			float* x568 = (float*)NULL;
			CUDNN_CALL(cudnnRNNBackwardWeights(
						cudnnHandle, rnn_desc_7, seqLength_7, x_descs_7, x471, hx_desc_7, x568,
						y_descs_7, x478, workspace_7, workspaceSize_7,
						w_desc_7, x151, x479, x480));
			;
			// backprop for sum on dim op
			sum_grad<<<28, 512>>>(x470, x388, x248, 2, x451, x454, x474, x455, x451, 1, 2);
			;
			float* x572 = (float*)NULL;
			float* x573 = (float*)NULL;
			CUDNN_CALL(cudnnRNNBackwardData(
						cudnnHandle, rnn_desc_6, seqLength_6, y_descs_6, x465, y_descs_6, x470,
						hx_desc_6, NULL, hx_desc_6, NULL, w_desc_6, x104, hx_desc_6, x572,
						hx_desc_6, x573, x_descs_6, x461, hx_desc_6, NULL, hx_desc_6, NULL,
						workspace_6, workspaceSize_6, x466, x467));
			;
			float* x575 = (float*)NULL;
			CUDNN_CALL(cudnnRNNBackwardWeights(
						cudnnHandle, rnn_desc_6, seqLength_6, x_descs_6, x457, hx_desc_6, x575,
						y_descs_6, x465, workspace_6, workspaceSize_6,
						w_desc_6, x106, x466, x467));
			;
			// backprop for sum on dim op
			sum_grad<<<28, 512>>>(x450, x388, x248, 2, x451, x454, x461, x455, x451, 1, 2);
			;
			float* x579 = (float*)NULL;
			float* x580 = (float*)NULL;
			CUDNN_CALL(cudnnRNNBackwardData(
						cudnnHandle, rnn_desc_5, seqLength_5, y_descs_5, x445, y_descs_5, x450,
						hx_desc_5, NULL, hx_desc_5, NULL, w_desc_5, x58, hx_desc_5, x579,
						hx_desc_5, x580, x_descs_5, x437, hx_desc_5, NULL, hx_desc_5, NULL,
						workspace_5, workspaceSize_5, x446, x447));
			;
			float* x582 = (float*)NULL;
			CUDNN_CALL(cudnnRNNBackwardWeights(
						cudnnHandle, rnn_desc_5, seqLength_5, x_descs_5, x417, hx_desc_5, x582,
						y_descs_5, x445, workspace_5, workspaceSize_5,
						w_desc_5, x60, x446, x447));
			;
			// backprop for permute WrappedArray(2, 0, 1)
			int* x585 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
			x585[2] = x418;
			x585[0] = x414;
			x585[1] = 1;
			x585[3] = 1;
			float* x590 = (float*)myMalloc(1 * sizeof(float));;
			x590[0] = 1.0f;
			CUDNN_CALL(cudnnTransformTensor(
						cudnnHandle, x590, out_desc_4, x437, x590, in_desc_4, x411));
			;
			hardTanh_grad<<<28, 512>>>(x402, x411, x411, 0.0, 20.0, x391, true);
			float* x594 = (float*)myMalloc(1 * sizeof(float));;
			x594[0] = 0.0f;
			float* x596 = (float*)myMalloc(1 * sizeof(float));;
			x596[0] = 1.0f;
			CUDNN_CALL(cudnnBatchNormalizationBackward(
						cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
						x596, x596, x596, x596, in_desc_3, x394,
						out_desc_3, x411, in_desc_3, x401, sbmv_desc_3, x40,
						x42,x44, 1.0E-5, x403, x404));
			;
			// conv2D back-propagate
			float* x600 = (float*)myMalloc(1 * sizeof(float));;
			x600[0] = 1.0f;

			{// Algorithm.
				cudnnConvolutionBwdDataAlgo_t algo;
				CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(
							cudnnHandle,
							filt_desc_2, out_desc_2, conv_desc_2, in_desc_2,
							CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algo));
				// algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
							cudnnHandle, filt_desc_2, out_desc_2, conv_desc_2, in_desc_2, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardData(
							cudnnHandle,
							x600, filt_desc_2, x37, out_desc_2, x401,
							conv_desc_2, algo, ws_data, ws_size,
							x600, in_desc_2, x378));
			};
			float* x603 = (float*)myMalloc(1 * sizeof(float));;
			x603[0] = 1.0f;

			{// Algorithm.
				cudnnConvolutionBwdFilterAlgo_t algo;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(
							cudnnHandle,
							in_desc_2, out_desc_2, conv_desc_2, filt_desc_2,
							CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algo));
				// algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc_2, out_desc_2, conv_desc_2, filt_desc_2, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x603, in_desc_2, x369, out_desc_2, x401,
							conv_desc_2, algo, ws_data, ws_size,
							x603, filt_desc_2, x39));
			};
			hardTanh_grad<<<28, 512>>>(x369, x378, x378, 0.0, 20.0, x357, true);
			float* x607 = (float*)myMalloc(1 * sizeof(float));;
			x607[0] = 0.0f;
			float* x609 = (float*)myMalloc(1 * sizeof(float));;
			x609[0] = 1.0f;
			CUDNN_CALL(cudnnBatchNormalizationBackward(
						cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
						x609, x609, x609, x609, in_desc_1, x361,
						out_desc_1, x378, in_desc_1, x368, sbmv_desc_1, x20,
						x22,x24, 1.0E-5, x370, x371));
			;
			// conv2D back-propagate
			float* x613 = (float*)myMalloc(1 * sizeof(float));;
			x613[0] = 1.0f;

			{// Algorithm.
				cudnnConvolutionBwdFilterAlgo_t algo;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(
							cudnnHandle,
							in_desc_0, out_desc_0, conv_desc_0, filt_desc_0,
							CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algo));
				// algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc_0, out_desc_0, conv_desc_0, filt_desc_0, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x613, in_desc_0, x339, out_desc_0, x368,
							conv_desc_0, algo, ws_data, ws_size,
							x613, filt_desc_0, x19));
			};
			// Tensor 'toCPU' invocation.
			float* x617 = (float*)myMalloc(1 * sizeof(float));;
			CUDA_CALL(cudaMemcpy(x617, x345, 1 * sizeof(float), cudaMemcpyDeviceToHost));
			float x619 = x617[0];
			x324 += x619;
			momentum_update_1D_1D<<<28, 512>>>(x17, x19, x205, 3.0E-8, 0.01, 400.0, true, 14432);
			momentum_update_1D_1D<<<28, 512>>>(x37, x39, x206, 3.0E-8, 0.01, 400.0, true, 236544);
			momentum_update_1D_1D<<<28, 512>>>(x40, x42, x207, 3.0E-8, 0.01, 400.0, true, 32);
			momentum_update_1D_1D<<<28, 512>>>(x43, x44, x208, 3.0E-8, 0.01, 400.0, true, 32);
			momentum_update_1D_1D<<<28, 512>>>(x23, x24, x209, 3.0E-8, 0.01, 400.0, true, 32);
			momentum_update_1D_1D<<<28, 512>>>(x20, x22, x210, 3.0E-8, 0.01, 400.0, true, 32);
			momentum_update_1D_1D<<<28, 512>>>(x185, x187, x211, 3.0E-8, 0.01, 400.0, true, 1024);
			momentum_update_1D_1D<<<28, 512>>>(x188, x189, x212, 3.0E-8, 0.01, 400.0, true, 1024);
			momentum_update_1D_1D<<<28, 512>>>(x202, x204, x213, 3.0E-8, 0.01, 400.0, true, 29696);
			momentum_update_1D_1D<<<28, 512>>>(x174, x175, x214, 3.0E-8, 0.01, 400.0, true, 1048576);
			momentum_update_1D_1D<<<28, 512>>>(x170, x171, x215, 3.0E-8, 0.01, 400.0, true, 1048576);
			momentum_update_1D_1D<<<28, 512>>>(x182, x183, x216, 3.0E-8, 0.01, 400.0, true, 1024);
			momentum_update_1D_1D<<<28, 512>>>(x178, x179, x217, 3.0E-8, 0.01, 400.0, true, 1024);
			momentum_update_1D_1D<<<28, 512>>>(x154, x155, x218, 3.0E-8, 0.01, 400.0, true, 1048576);
			momentum_update_1D_1D<<<28, 512>>>(x158, x159, x219, 3.0E-8, 0.01, 400.0, true, 1048576);
			momentum_update_1D_1D<<<28, 512>>>(x166, x167, x220, 3.0E-8, 0.01, 400.0, true, 1024);
			momentum_update_1D_1D<<<28, 512>>>(x162, x163, x221, 3.0E-8, 0.01, 400.0, true, 1024);
			momentum_update_1D_1D<<<28, 512>>>(x129, x130, x222, 3.0E-8, 0.01, 400.0, true, 1048576);
			momentum_update_1D_1D<<<28, 512>>>(x125, x126, x223, 3.0E-8, 0.01, 400.0, true, 1048576);
			momentum_update_1D_1D<<<28, 512>>>(x137, x138, x224, 3.0E-8, 0.01, 400.0, true, 1024);
			momentum_update_1D_1D<<<28, 512>>>(x109, x110, x225, 3.0E-8, 0.01, 400.0, true, 1048576);
			momentum_update_1D_1D<<<28, 512>>>(x133, x134, x226, 3.0E-8, 0.01, 400.0, true, 1024);
			momentum_update_1D_1D<<<28, 512>>>(x113, x114, x227, 3.0E-8, 0.01, 400.0, true, 1048576);
			momentum_update_1D_1D<<<28, 512>>>(x117, x118, x228, 3.0E-8, 0.01, 400.0, true, 1024);
			momentum_update_1D_1D<<<28, 512>>>(x121, x122, x229, 3.0E-8, 0.01, 400.0, true, 1024);
			momentum_update_1D_1D<<<28, 512>>>(x91, x92, x230, 3.0E-8, 0.01, 400.0, true, 1024);
			momentum_update_1D_1D<<<28, 512>>>(x79, x80, x231, 3.0E-8, 0.01, 400.0, true, 688128);
			momentum_update_1D_1D<<<28, 512>>>(x63, x64, x232, 3.0E-8, 0.01, 400.0, true, 688128);
			momentum_update_1D_1D<<<28, 512>>>(x87, x88, x233, 3.0E-8, 0.01, 400.0, true, 1024);
			momentum_update_1D_1D<<<28, 512>>>(x67, x68, x234, 3.0E-8, 0.01, 400.0, true, 1048576);
			momentum_update_1D_1D<<<28, 512>>>(x71, x72, x235, 3.0E-8, 0.01, 400.0, true, 1024);
			momentum_update_1D_1D<<<28, 512>>>(x75, x76, x236, 3.0E-8, 0.01, 400.0, true, 1024);
			momentum_update_1D_1D<<<28, 512>>>(x83, x84, x237, 3.0E-8, 0.01, 400.0, true, 1048576);
			int32_t x654 = x321;
			int32_t x656 = x654 % x655;
			bool x657 = x656 == 0;
			if (x657) {
				float x662 = x324;
				double x658 = (double)x654;
				double x659 = 100.0 * x658;
				double x661 = x659 / x660;
				float x663 = (float)x654;
				float x664 = x662 / x663;
				printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x317,x654,x253,x661,x664);
				fflush(stdout);
			} else {
			}
			int64_t x669 = (long)mallocAddr;
			int64_t x670 = x669 - x313;
			memset((void*)x313, 0, x670);
			mallocAddr = (void*)x313;
			int64_t x673 = (long)gpuMallocAddr;
			int64_t x674 = x673 - x314;
			cudaMemset((void*)x314, 0, x674);
			gpuMallocAddr = (void*)x314;

		}
		gettimeofday(&end_1, NULL);
		timeval_subtract(&diff_1, &end_1, &begin_1);;
		int64_t x681 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
		int64_t x682 = x681 / 1000LL;
		int64_t x684 = x681 / x683;
		printf("Training completed in %ldms (%ld us/images)\n",x682,x684);
		double x686 = (double)x681;
		double x687 = x686 / 1000000.0;
		x312[x317] = x687;
		float x689 = x324;
		float x691 = x689 / x690;
		double x692 = (double)x691;
		x311[x317] = x692;

	}
	gettimeofday(&end_0, NULL);
	timeval_subtract(&diff_0, &end_0, &begin_0);;
	int64_t x698 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
	sort(x312, x312 + 1);
	double x704 = x312[0];
	int64_t x705 = (long)fopen(x0, "w");
	fprintf((FILE *)x705, "unit: %s\n", "1 epoch");
	for(int x707=0; x707 < 1; x707++) {
		double x708 = x311[x707];
		fprintf((FILE *)x705, "%lf\n", x708);

	}
	fprintf((FILE *)x705, "run time: %lf %lf\n", x309, x704);
	fclose((FILE*)x705);
	// Backend cleanup.
	CUBLAS_CALL(cublasDestroy(cublasHandle));
	CUDA_CALL(cudaFree(gpuMallocBase));

	CUDNN_CALL(cudnnDestroy(cudnnHandle));
}
/*****************************************
  End of C Generated Code                  
 *******************************************/


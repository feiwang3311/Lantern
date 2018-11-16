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
constexpr int N = 5; // 4; // 16
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
	float* x8 = (float*)myMalloc(14432 * sizeof(float));;
	for(int x10=0; x10 < 14432; x10++) {
		float x11 = (float)rand()/RAND_MAX;
		float x12 = x11 - 0.5f;
		float x13 = x12 * 0.23068394f;
		x8[x10] = x13;

	}
	// Tensor 'toGPU' invocation.
	float* x18 = (float*)myGpuMalloc(14432 * sizeof(float));
	CUDA_CALL(cudaMemcpy(x18, x8, 14432 * sizeof(float), cudaMemcpyHostToDevice));
	float* x20 = (float*)myGpuMalloc(14432 * sizeof(float));
	float* x21 = (float*)myMalloc(32 * sizeof(float));;
	for(int x23=0; x23 < 32; x23++) {
		x21[x23] = 1.0f;

	}
	// Tensor 'toGPU' invocation.
	float* x28 = (float*)myGpuMalloc(32 * sizeof(float));
	CUDA_CALL(cudaMemcpy(x28, x21, 32 * sizeof(float), cudaMemcpyHostToDevice));
	float* x30 = (float*)myGpuMalloc(32 * sizeof(float));
	float* x31 = (float*)myGpuMalloc(32 * sizeof(float));
	float* x32 = (float*)myGpuMalloc(32 * sizeof(float));
	float* x33 = (float*)myGpuMalloc(32 * sizeof(float));
	float* x34 = (float*)myGpuMalloc(32 * sizeof(float));
	float* x35 = (float*)myMalloc(236544 * sizeof(float));;
	for(int x37=0; x37 < 236544; x37++) {
		float x38 = (float)rand()/RAND_MAX;
		float x39 = x38 - 0.5f;
		float x40 = x39 * 0.05698029f;
		x35[x37] = x40;

	}
	// Tensor 'toGPU' invocation.
	float* x45 = (float*)myGpuMalloc(236544 * sizeof(float));
	CUDA_CALL(cudaMemcpy(x45, x35, 236544 * sizeof(float), cudaMemcpyHostToDevice));
	float* x47 = (float*)myGpuMalloc(236544 * sizeof(float));
	float* x48 = (float*)myMalloc(32 * sizeof(float));;
	for(int x49=0; x49 < 32; x49++) {
		x48[x49] = 1.0f;

	}
	// Tensor 'toGPU' invocation.
	float* x54 = (float*)myGpuMalloc(32 * sizeof(float));
	CUDA_CALL(cudaMemcpy(x54, x48, 32 * sizeof(float), cudaMemcpyHostToDevice));
	float* x56 = (float*)myGpuMalloc(32 * sizeof(float));
	float* x57 = (float*)myGpuMalloc(32 * sizeof(float));
	float* x58 = (float*)myGpuMalloc(32 * sizeof(float));
	float* x59 = (float*)myGpuMalloc(32 * sizeof(float));
	float* x60 = (float*)myGpuMalloc(32 * sizeof(float));
	int32_t x61 = 0;
	float* x62 = (float*)myMalloc(4788224 * sizeof(float));;
	for(int x64=0; x64 < 4788224; x64++) {
		x62[x64] = 0.01f;

	}
	// Tensor 'toGPU' invocation.
	float* x69 = (float*)myGpuMalloc(4788224 * sizeof(float));
	CUDA_CALL(cudaMemcpy(x69, x62, 4788224 * sizeof(float), cudaMemcpyHostToDevice));
	float* x71 = (float*)myGpuMalloc(4788224 * sizeof(float));
	int32_t x72 = x61;
	float* x73 = x69+x72;
	float* x74 = x71+x72;
	x61 += 1343488;
	int32_t x76 = x61;
	float* x77 = x69+x76;
	float* x78 = x71+x76;
	x61 += 1048576;
	int32_t x80 = x61;
	float* x81 = x69+x80;
	float* x82 = x71+x80;
	x61 += 1343488;
	int32_t x84 = x61;
	float* x85 = x69+x84;
	float* x86 = x71+x84;
	x61 += 1048576;
	int32_t x88 = x61;
	float* x89 = x69+x88;
	float* x90 = x71+x88;
	x61 += 1024;
	int32_t x92 = x61;
	float* x93 = x69+x92;
	float* x94 = x71+x92;
	x61 += 1024;
	int32_t x96 = x61;
	float* x97 = x69+x96;
	float* x98 = x71+x96;
	x61 += 1024;
	int32_t x100 = x61;
	float* x101 = x69+x100;
	float* x102 = x71+x100;
	x61 += 1024;
	int32_t x104 = 0;
	float* x105 = (float*)myMalloc(4198400 * sizeof(float));;
	for(int x107=0; x107 < 4198400; x107++) {
		x105[x107] = 0.01f;

	}
	// Tensor 'toGPU' invocation.
	float* x112 = (float*)myGpuMalloc(4198400 * sizeof(float));
	CUDA_CALL(cudaMemcpy(x112, x105, 4198400 * sizeof(float), cudaMemcpyHostToDevice));
	float* x114 = (float*)myGpuMalloc(4198400 * sizeof(float));
	int32_t x115 = x104;
	float* x116 = x112+x115;
	float* x117 = x114+x115;
	x104 += 1048576;
	int32_t x119 = x104;
	float* x120 = x112+x119;
	float* x121 = x114+x119;
	x104 += 1048576;
	int32_t x123 = x104;
	float* x124 = x112+x123;
	float* x125 = x114+x123;
	x104 += 1048576;
	int32_t x127 = x104;
	float* x128 = x112+x127;
	float* x129 = x114+x127;
	x104 += 1048576;
	int32_t x131 = x104;
	float* x132 = x112+x131;
	float* x133 = x114+x131;
	x104 += 1024;
	int32_t x135 = x104;
	float* x136 = x112+x135;
	float* x137 = x114+x135;
	x104 += 1024;
	int32_t x139 = x104;
	float* x140 = x112+x139;
	float* x141 = x114+x139;
	x104 += 1024;
	int32_t x143 = x104;
	float* x144 = x112+x143;
	float* x145 = x114+x143;
	x104 += 1024;
	int32_t x147 = 0;
	float* x148 = (float*)myMalloc(4198400 * sizeof(float));;
	for(int x149=0; x149 < 4198400; x149++) {
		x148[x149] = 0.01f;

	}
	// Tensor 'toGPU' invocation.
	float* x154 = (float*)myGpuMalloc(4198400 * sizeof(float));
	CUDA_CALL(cudaMemcpy(x154, x148, 4198400 * sizeof(float), cudaMemcpyHostToDevice));
	float* x156 = (float*)myGpuMalloc(4198400 * sizeof(float));
	int32_t x157 = x147;
	float* x158 = x154+x157;
	float* x159 = x156+x157;
	x147 += 1048576;
	int32_t x161 = x147;
	float* x162 = x154+x161;
	float* x163 = x156+x161;
	x147 += 1048576;
	int32_t x165 = x147;
	float* x166 = x154+x165;
	float* x167 = x156+x165;
	x147 += 1048576;
	int32_t x169 = x147;
	float* x170 = x154+x169;
	float* x171 = x156+x169;
	x147 += 1048576;
	int32_t x173 = x147;
	float* x174 = x154+x173;
	float* x175 = x156+x173;
	x147 += 1024;
	int32_t x177 = x147;
	float* x178 = x154+x177;
	float* x179 = x156+x177;
	x147 += 1024;
	int32_t x181 = x147;
	float* x182 = x154+x181;
	float* x183 = x156+x181;
	x147 += 1024;
	int32_t x185 = x147;
	float* x186 = x154+x185;
	float* x187 = x156+x185;
	x147 += 1024;
	float* x189 = (float*)myMalloc(1024 * sizeof(float));;
	for(int x191=0; x191 < 1024; x191++) {
		x189[x191] = 1.0f;

	}
	// Tensor 'toGPU' invocation.
	float* x196 = (float*)myGpuMalloc(1024 * sizeof(float));
	CUDA_CALL(cudaMemcpy(x196, x189, 1024 * sizeof(float), cudaMemcpyHostToDevice));
	float* x198 = (float*)myGpuMalloc(1024 * sizeof(float));
	float* x199 = (float*)myGpuMalloc(1024 * sizeof(float));
	float* x200 = (float*)myGpuMalloc(1024 * sizeof(float));
	float* x201 = (float*)myGpuMalloc(1024 * sizeof(float));
	float* x202 = (float*)myGpuMalloc(1024 * sizeof(float));
	float* x203 = (float*)myMalloc(29696 * sizeof(float));;
	for(int x205=0; x205 < 29696; x205++) {
		float x206 = (float)rand()/RAND_MAX;
		float x207 = x206 - 0.5f;
		float x208 = x207 * 0.03125f;
		x203[x205] = x208;

	}
	// Tensor 'toGPU' invocation.
	float* x213 = (float*)myGpuMalloc(29696 * sizeof(float));
	CUDA_CALL(cudaMemcpy(x213, x203, 29696 * sizeof(float), cudaMemcpyHostToDevice));
	float* x215 = (float*)myGpuMalloc(29696 * sizeof(float));
	int32_t x216 = open("/scratch/wu636/training/speech_recognition/data/test/deepspeech_train.bin",0);
	int64_t x217 = fsize(x216);
	printf("file size is %ld\n",x217);
	char* x219 = (char*)mmap(0, x217, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x216, 0);
	int64_t x220 = (long)x219;
	int64_t x221 = x220;
	int64_t x222 = x221;
	int* x223 = (int32_t*) x222;
	int64_t x224 = (int64_t)4;
	x221 += x224;
	int32_t x226 = x223[0];
	int64_t x227 = x221;
	int* x228 = (int32_t*) x227;
	x221 += x224;
	int32_t x230 = x228[0];
	printf("data size is %d batches, %d batch size\n",200,x226);
	int* x233 = (int32_t*)myMalloc(200 * sizeof(int32_t));;
	int* x234 = (int32_t*)myMalloc(200 * sizeof(int32_t));;
	float** x235 = (float**)myMalloc(200 * sizeof(float*));;
	float** x236 = (float**)myMalloc(200 * sizeof(float*));;
	int** x237 = (int**)myMalloc(200 * sizeof(int*));;
	int** x238 = (int**)myMalloc(200 * sizeof(int*));;
	// load data by batchs
	int32_t x267 = 4 * x226;
	int64_t x268 = (int64_t)x267;
	for(int x241=0; x241 < 200; x241++) {
		int64_t x242 = x221;
		int* x243 = (int32_t*) x242;
		x221 += x224;
		int32_t x245 = x243[0];
		x233[x241] = x245;
		int64_t x247 = x221;
		int* x248 = (int32_t*) x247;
		x221 += x224;
		int32_t x250 = x248[0];
		x234[x241] = x250;
		int32_t x252 = x233[x241];
		int32_t x253 = x234[x241];
		printf("batch %d has freqSize %d maxLength %d\n",x241,x252,x253);
		int32_t x255 = x233[x241];
		int32_t x257 = x234[x241];
		int64_t x259 = x221;
		float* x260 = (float*) x259;
		int32_t x256 = x226 * x255;
		int32_t x258 = x256 * x257;
		int32_t x261 = 4 * x258;
		int64_t x262 = (int64_t)x261;
		x221 += x262;
		x235[x241] = x260;
		int64_t x265 = x221;
		float* x266 = (float*) x265;
		x221 += x268;
		x236[x241] = x266;
		int64_t x271 = x221;
		int* x272 = (int32_t*) x271;
		x221 += x268;
		x237[x241] = x272;
		int* x275 = x237[x241];
		int* x276 = x237[x241];
		int32_t x277 = accumulate(x275, x276 + x226, 0);
		int64_t x278 = x221;
		int* x279 = (int32_t*) x278;
		int32_t x280 = 4 * x277;
		int64_t x281 = (int64_t)x280;
		x221 += x281;
		x238[x241] = x279;

	}
	gettimeofday(&end_0, NULL);
	timeval_subtract(&diff_0, &end_0, &begin_0);;
	int64_t x288 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
	float x289 = (float)x288;
	float x290 = x289 / 1000000.0f;
	printf("Data reading (all prepare time) in %lf sec\n",x290);
	double* x292 = (double*)myMalloc(1 * sizeof(double));;
	double* x293 = (double*)myMalloc(1 * sizeof(double));;
	int64_t x294 = (long)mallocAddr;
	int64_t x295 = (long)gpuMallocAddr;
	// training loop starts here
	int32_t x345 = x226 * 32;
	int32_t x438 = 2048 / 2;
	bool x484 = x438 == 1024;
	bool x489 = 1024 == x438;
	bool x540 = x226 <= 256;
	int32_t x439 = 2 * x438;
	int32_t x440 = x226 * x439;
	int32_t x442 = x226 * x438;
	int32_t x635 = x226 * 20;
	int32_t x231 = x226 * 200;
	double x640 = (double)x231;
	int64_t x663 = (int64_t)x231;
	float x670 = (float)x231;
	for(int x298=0; x298 < 1; x298++) {
		struct timeval begin_1, end_1, diff_1;
		int32_t x300 = 0;
		int32_t x301 = x300;
		int32_t x302 = x301;
		float x303 = 0.0f;
		float x304 = x303;
		float x305 = x304;
		int32_t x306 = x298 + 1;
		printf("Start training epoch %d\n",x306);
		gettimeofday(&begin_1, NULL);
		for(int x309=0; x309 < 200; x309++) {
			int32_t x310 = x234[x309];
			int32_t x311 = x233[x309];
			float* x312 = x235[x309];
			float* x315 = x236[x309];
			int* x316 = x238[x309];
			int* x317 = x237[x309];
			x302 += x226;
			// Tensor 'toGPU' invocation.
			int32_t x313 = x311 * x310;
			int32_t x314 = x226 * x313;
			float* x320 = (float*)myGpuMalloc(x314 * sizeof(float));
			CUDA_CALL(cudaMemcpy(x320, x312, x314 * sizeof(float), cudaMemcpyHostToDevice));
			float* x322 = (float*)myGpuMalloc(2 * sizeof(float));
			float* x323 = (float*)myGpuMalloc(1 * sizeof(float));
			float* x324 = (float*)myGpuMalloc(1 * sizeof(float));
			// allocate memory to save the final loss in CPU Tensor
			float* x326 = (float*)myMalloc(1 * sizeof(float));;
			int32_t x327 = x311 + 40;
			bool x328 = x327 >= 41;
			int32_t x329 = x310 + 10;
			bool x331;
			if (x328) {
				bool x330 = x329 >= 11;
				x331 = x330;
			} else {
				x331 = false;
			}
			if (x331) {
			} else {
				assert(false && "ERROR not specified");
			}
			int32_t x339 = x329 - 11;
			int32_t x340 = x339 / 2;
			int32_t x341 = x340 + 1;
			int32_t x336 = x327 - 41;
			int32_t x337 = x336 / 2;
			int32_t x338 = x337 + 1;
			int32_t x346 = x345 * x338;
			int32_t x347 = x346 * x341;
			float* x348 = (float*)myGpuMalloc(x347 * sizeof(float));
			float* x349 = (float*)myMalloc(1 * sizeof(float));;
			x349[0] = 0.0f;
			float* x351 = (float*)myMalloc(1 * sizeof(float));;
			x351[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x226, 1, x311, x310));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							32, 1, 41, 11));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x226, 32, x338, x341));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							20, 5, 2, 2, 1, 1,
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
							x351, in_desc, x320, filt_desc, x18,
							conv_desc, algo, ws_data, ws_size,
							x349, out_desc, x348));
			};
			float* x354 = (float*)myGpuMalloc(x347 * sizeof(float));
			int32_t x342 = x338 * x341;
			int32_t x343 = 32 * x342;
			int32_t x344 = x226 * x343;
			float* x355 = (float*)myGpuMalloc(x344 * sizeof(float));
			float* x356 = (float*)myGpuMalloc(32 * sizeof(float));
			float* x357 = (float*)myGpuMalloc(32 * sizeof(float));
			float* x358 = (float*)myMalloc(1 * sizeof(float));;
			x358[0] = 0.0f;
			float* x360 = (float*)myMalloc(1 * sizeof(float));;
			x360[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x226, 32, x338, x341));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x226, 32, x338, x341));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 32, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x360, x358, in_desc, x348, out_desc, x355, sbmv_desc, x28,
							x31, 0.1, x33, x34, 1.0E-5,
							x356, x357));
			};
			float* x363 = (float*)myGpuMalloc(x347 * sizeof(float));
			hardTanh<<<28, 512>>>(x355, x355, 0.0, 20.0, true);
			int32_t x365 = x338 + 20;
			bool x366 = x365 >= 21;
			int32_t x367 = x341 + 10;
			bool x369;
			if (x366) {
				bool x368 = x367 >= 11;
				x369 = x368;
			} else {
				x369 = false;
			}
			if (x369) {
			} else {
				assert(false && "ERROR not specified");
			}
			int32_t x377 = x367 - 11;
			int32_t x378 = x377 / 1;
			int32_t x379 = x378 + 1;
			int32_t x374 = x365 - 21;
			int32_t x375 = x374 / 2;
			int32_t x376 = x375 + 1;
			int32_t x383 = x345 * x376;
			int32_t x384 = x383 * x379;
			float* x385 = (float*)myGpuMalloc(x384 * sizeof(float));
			float* x386 = (float*)myMalloc(1 * sizeof(float));;
			x386[0] = 0.0f;
			float* x388 = (float*)myMalloc(1 * sizeof(float));;
			x388[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x226, 32, x338, x341));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							32, 32, 21, 11));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x226, 32, x376, x379));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							10, 5, 2, 1, 1, 1,
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
							x388, in_desc, x355, filt_desc, x45,
							conv_desc, algo, ws_data, ws_size,
							x386, out_desc, x385));
			};
			float* x391 = (float*)myGpuMalloc(x384 * sizeof(float));
			int32_t x380 = x376 * x379;
			int32_t x381 = 32 * x380;
			int32_t x382 = x226 * x381;
			float* x392 = (float*)myGpuMalloc(x382 * sizeof(float));
			float* x393 = (float*)myGpuMalloc(32 * sizeof(float));
			float* x394 = (float*)myGpuMalloc(32 * sizeof(float));
			float* x395 = (float*)myMalloc(1 * sizeof(float));;
			x395[0] = 0.0f;
			float* x397 = (float*)myMalloc(1 * sizeof(float));;
			x397[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x226, 32, x376, x379));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x226, 32, x376, x379));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 32, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x397, x395, in_desc, x385, out_desc, x392, sbmv_desc, x54,
							x57, 0.1, x59, x60, 1.0E-5,
							x393, x394));
			};
			float* x400 = (float*)myGpuMalloc(x384 * sizeof(float));
			hardTanh<<<28, 512>>>(x392, x392, 0.0, 20.0, true);
			// after conv ops
			int32_t x403 = 32 * x376;
			int32_t x404 = x403 * x379;
			int32_t x405 = x226 * x404;
			float* x406 = (float*)myGpuMalloc(x405 * sizeof(float));
			int* x409 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
			int32_t x407 = x226 * x403;
			x409[2] = x407;
			x409[0] = x403;
			x409[1] = 1;
			x409[3] = 1;
			float* x414 = (float*)myMalloc(1 * sizeof(float));;
			x414[0] = 1.0f;
			float* x416 = (float*)myMalloc(0 * sizeof(float));;
			x416[0] = 0.0f;
			int32_t x418 = x409[0];
			int32_t x419 = x409[1];
			int32_t x420 = x409[2];
			int32_t x421 = x409[3];

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
							in_desc, CUDNN_DATA_FLOAT,
							x226, x403, x379, 1,
							x404, x379, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
							out_desc, CUDNN_DATA_FLOAT,
							x226, x403, x379, 1,
							x418, x419, x420, x421));

				CUDNN_CALL(cudnnTransformTensor(
							cudnnHandle, x414, in_desc, x392, x416, out_desc, x406));
			};
			int32_t x423 = x379 * x226;
			int32_t x424 = x423 * x403;
			float* x425 = (float*)myGpuMalloc(x424 * sizeof(float));
			// after resize and permute
			float* x427 = (float*)NULL;
			float* x428 = (float*)NULL;
			float* x429 = (float*)NULL;
			int32_t x432 = x423 * 2048;
			float* x433 = (float*)myGpuMalloc(x432 * sizeof(float));
			float* x434 = (float*)NULL;
			int32_t x435 = 0;

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
				int32_t seqLength = x379;
				int32_t batchSize = x226;
				int32_t inputSize = x403;

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
				assert(paramsSize / sizeof(float) == 4788224 && "Expected parameter size mismatch");

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
				void* workspace = myGpuMalloc(workspaceSize);

				// Reserve space used by `ForwardTraining` function.
				size_t reserveSize;
				CUDNN_CALL(cudnnGetRNNTrainingReserveSize(
							cudnnHandle, rnn_desc, seqLength, x_descs, &reserveSize));
				void* reserveSpace = myGpuMalloc(reserveSize);
				x434 = (float*)reserveSpace;
				x435 = (int)reserveSize;
				CUDNN_CALL(cudnnRNNForwardTraining(
							cudnnHandle, rnn_desc, seqLength, x_descs, x406,
							hx_desc,x427, cx_desc,x428, w_desc, x69, y_descs, x433,
							hy_desc,x429, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
			};
			float* x437 = (float*)myGpuMalloc(x432 * sizeof(float));
			int32_t x444 = x423 * x438;
			float* x445 = (float*)myGpuMalloc(x444 * sizeof(float));
			float* x446 = (float*)myMalloc(1 * sizeof(float));;
			x446[0] = 0.0f;
			float* x448 = (float*)myMalloc(1 * sizeof(float));;
			x448[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x379, x226, 2, x438));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x379, x226, 1, x438));

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
							x448, x_desc, x433, x446, out_desc, x445));
			};
			float* x451 = (float*)myGpuMalloc(x444 * sizeof(float));
			float* x452 = (float*)NULL;
			float* x453 = (float*)NULL;
			float* x454 = (float*)NULL;
			float* x455 = (float*)myGpuMalloc(x432 * sizeof(float));
			float* x456 = (float*)NULL;
			int32_t x457 = 0;

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
				int32_t seqLength = x379;
				int32_t batchSize = x226;
				int32_t inputSize = x438;

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
				assert(paramsSize / sizeof(float) == 4198400 && "Expected parameter size mismatch");

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
				void* workspace = myGpuMalloc(workspaceSize);

				// Reserve space used by `ForwardTraining` function.
				size_t reserveSize;
				CUDNN_CALL(cudnnGetRNNTrainingReserveSize(
							cudnnHandle, rnn_desc, seqLength, x_descs, &reserveSize));
				void* reserveSpace = myGpuMalloc(reserveSize);
				x456 = (float*)reserveSpace;
				x457 = (int)reserveSize;
				CUDNN_CALL(cudnnRNNForwardTraining(
							cudnnHandle, rnn_desc, seqLength, x_descs, x445,
							hx_desc,x452, cx_desc,x453, w_desc, x112, y_descs, x455,
							hy_desc,x454, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
			};
			float* x459 = (float*)myGpuMalloc(x432 * sizeof(float));
			float* x460 = (float*)myGpuMalloc(x444 * sizeof(float));
			float* x461 = (float*)myMalloc(1 * sizeof(float));;
			x461[0] = 0.0f;
			float* x463 = (float*)myMalloc(1 * sizeof(float));;
			x463[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x379, x226, 2, x438));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x379, x226, 1, x438));

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
							x463, x_desc, x455, x461, out_desc, x460));
			};
			float* x466 = (float*)myGpuMalloc(x444 * sizeof(float));
			float* x467 = (float*)NULL;
			float* x468 = (float*)NULL;
			float* x469 = (float*)NULL;
			float* x470 = (float*)myGpuMalloc(x432 * sizeof(float));
			float* x471 = (float*)NULL;
			int32_t x472 = 0;

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
				int32_t seqLength = x379;
				int32_t batchSize = x226;
				int32_t inputSize = x438;

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
				assert(paramsSize / sizeof(float) == 4198400 && "Expected parameter size mismatch");

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
				void* workspace = myGpuMalloc(workspaceSize);

				// Reserve space used by `ForwardTraining` function.
				size_t reserveSize;
				CUDNN_CALL(cudnnGetRNNTrainingReserveSize(
							cudnnHandle, rnn_desc, seqLength, x_descs, &reserveSize));
				void* reserveSpace = myGpuMalloc(reserveSize);
				x471 = (float*)reserveSpace;
				x472 = (int)reserveSize;
				CUDNN_CALL(cudnnRNNForwardTraining(
							cudnnHandle, rnn_desc, seqLength, x_descs, x460,
							hx_desc,x467, cx_desc,x468, w_desc, x154, y_descs, x470,
							hy_desc,x469, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
			};
			float* x474 = (float*)myGpuMalloc(x432 * sizeof(float));
			float* x475 = (float*)myGpuMalloc(x444 * sizeof(float));
			float* x476 = (float*)myMalloc(1 * sizeof(float));;
			x476[0] = 0.0f;
			float* x478 = (float*)myMalloc(1 * sizeof(float));;
			x478[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x379, x226, 2, x438));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x379, x226, 1, x438));

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
							x478, x_desc, x470, x476, out_desc, x475));
			};
			float* x481 = (float*)myGpuMalloc(x444 * sizeof(float));
			// after RNN layers
			// after maybe lookahead
			if (x484) {
			} else {
				assert(false && "BatchNorm1D input should be rank2, with shape 1 same as dimSize, got %d : %d");
			}
			if (x489) {
			} else {
				assert(false && "scale should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(423) x Sym(438)");
			}
			if (x489) {
			} else {
				assert(false && "bias should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(423) x Sym(438)");
			}
			if (x489) {
			} else {
				assert(false && "runningMean should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(423) x Sym(438)");
			}
			if (x489) {
			} else {
				assert(false && "runningVar should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(423) x Sym(438)");
			}
			float* x503 = (float*)myGpuMalloc(x444 * sizeof(float));
			float* x504 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x505 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x506 = (float*)myMalloc(1 * sizeof(float));;
			x506[0] = 0.0f;
			float* x508 = (float*)myMalloc(1 * sizeof(float));;
			x508[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x423, x438, 1, 1));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 1024, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,
							x508, x506, in_desc, x475, in_desc, x503, sbmv_desc, x196,
							x199, 0.1, x201, x202, 1.0E-5,
							x504, x505));
			};
			float* x511 = (float*)myGpuMalloc(x444 * sizeof(float));
			int32_t x512 = x423 * 29;
			float* x513 = (float*)myGpuMalloc(x512 * sizeof(float));
			float* x514 = (float*)myMalloc(1 * sizeof(float));;
			x514[0] = 0.0f;
			float* x516 = (float*)myMalloc(1 * sizeof(float));;
			x516[0] = 1.0f;
			CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 29,x423,1024,x516,x213,29,x503,1024,x514,x513,29));
			float* x519 = (float*)myGpuMalloc(x512 * sizeof(float));
			float* x522 = (float*)myMalloc(1 * sizeof(float));;
			x522[0] = 0.0f;
			float* x524 = (float*)myMalloc(1 * sizeof(float));;
			x524[0] = 1.0f;
			float* x526 = (float*)myGpuMalloc(x512 * sizeof(float));

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x423, 29, 1, 1));
				CUDNN_CALL(cudnnSoftmaxForward(
							cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
							x524, x_desc, x513, x522, x_desc, x526));
			};
			float* x528 = (float*)myGpuMalloc(x512 * sizeof(float));
			// before CTC loss
			int* x530 = (int32_t*)myMalloc(x226 * sizeof(int32_t));;
			float x534 = (float)x379;
			for(int x532=0; x532 < x226; x532++) {
				float x533 = x315[x532];
				float x535 = x533 * x534;
				int32_t x536 = (int)x535;
				x530[x532] = x536;

			}
			if (x540) {
			} else {
				printf("'cudnnGetCTCLossWorkspaceSize' requires batch size less than 256, got %d\n\n",x226);
				assert(false && "");
			}
			float* x546 = (float*)myGpuMalloc(x226 * sizeof(float));

			{
				cudnnTensorDescriptor_t probs_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&probs_desc));
				int probs_dims[] = {x379, x226, 29};
				int probs_strides[] = {probs_dims[1] * probs_dims[2], probs_dims[2], 1};
				CUDNN_CALL(cudnnSetTensorNdDescriptor(
							probs_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, probs_dims, probs_strides));

				cudnnTensorDescriptor_t grad_desc = probs_desc;

				cudnnCTCLossDescriptor_t ctc_desc;
				CUDNN_CALL(cudnnCreateCTCLossDescriptor(&ctc_desc));
				CUDNN_CALL(cudnnSetCTCLossDescriptor(ctc_desc, CUDNN_DATA_FLOAT));
				size_t wsSize;
				CUDNN_CALL(cudnnGetCTCLossWorkspaceSize(
							cudnnHandle, probs_desc, grad_desc, x316, x317, x530,
							CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, &wsSize));
				void *ws = myGpuMalloc(wsSize);

				CUDNN_CALL(cudnnCTCLoss(
							cudnnHandle, probs_desc, x526, x316, x317, x530,
							x546, grad_desc, x528, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, ws, wsSize));
			};
			float* x548 = (float*)myGpuMalloc(1 * sizeof(float));
			float* x549 = (float*)myMalloc(1 * sizeof(float));;
			x549[0] = 0.0f;
			float* x551 = (float*)myMalloc(1 * sizeof(float));;
			x551[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x226, 1, 1, 1));

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
							x551, x_desc, x546, x549, out_desc, x548));
			};
			// after CTC loss
			float* x555 = (float*)myGpuMalloc(1 * sizeof(float));
			// make sure the size of loss is 1
			arrayFill_greg<<<28, 512>>>(x555, 1.0f, 1);
			// backend is lantern.TensorDslCudnn$BackendCudnn@4aa3a4a9
			CUDA_CALL(cudaMemcpy(x326, x548, 1 * sizeof(float), cudaMemcpyDeviceToHost));
			float* x560 = (float*)myMalloc(1 * sizeof(float));;
			x560[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x423, 29, 1, 1));
				CUDNN_CALL(cudnnSoftmaxBackward(
							cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
							x560, x_desc, x526, x_desc, x528,
							x560, x_desc, x519));
			};
			float* x563 = (float*)myMalloc(1 * sizeof(float));;
			x563[0] = 0.0f;
			float* x565 = (float*)myMalloc(1 * sizeof(float));;
			x565[0] = 1.0f;
			// backprop of matrix-matrix-dot
			CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, x438,x423,29,x565,x213,29,x519,29,x565,x511,x438));
			CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 29,x438,x423,x565,x519,29,x503,x438,x565,x215,29));
			float* x570 = (float*)myMalloc(1 * sizeof(float));;
			x570[0] = 0.0f;
			float* x572 = (float*)myMalloc(1 * sizeof(float));;
			x572[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x423, x438, 1, 1));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 1024, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,
							x572, x572, x572, x572, in_desc, x475,
							in_desc, x511, in_desc, x481, sbmv_desc, x196,
							x198,x200, 1.0E-5, x504, x505));
			};
			// backprop for sum on dim op
			int32_t x441 = x379 * x440;
			sum_grad<<<28, 512>>>(x474, x379, x226, 2, x438, x441, x481, x442, x438, 1, 2);
			;
			float* x577 = (float*)NULL;
			float* x578 = (float*)NULL;

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
				int32_t seqLength = x379;
				int32_t batchSize = x226;
				int32_t inputSize = x438;

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
				assert(paramsSize / sizeof(float) == 4198400 && "Expected parameter size mismatch");

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
							cudnnHandle, rnn_desc, seqLength, y_descs, x470, y_descs, x474,
							dhy_desc, NULL, dcy_desc, NULL, w_desc, x154, hx_desc, x577,
							cx_desc, x578, dx_descs, x466, dhx_desc, NULL, dcx_desc, NULL,
							workspace, workspaceSize, x471, x472));
			};
			float* x580 = (float*)NULL;

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
				int32_t seqLength = x379;
				int32_t batchSize = x226;
				int32_t inputSize = x438;

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
				// printf("paramsSize: %zu\n", paramsSize / sizeof(float));
				assert(paramsSize / sizeof(float) == 4198400 && "Expected parameter size mismatch");

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
							cudnnHandle, rnn_desc, seqLength, x_descs, x460, hx_desc, x580,
							y_descs, x470, workspace, workspaceSize,
							dw_desc, x156, x471, x472));
			};
			// backprop for sum on dim op
			sum_grad<<<28, 512>>>(x459, x379, x226, 2, x438, x441, x466, x442, x438, 1, 2);
			;
			float* x584 = (float*)NULL;
			float* x585 = (float*)NULL;

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
				int32_t seqLength = x379;
				int32_t batchSize = x226;
				int32_t inputSize = x438;

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
				assert(paramsSize / sizeof(float) == 4198400 && "Expected parameter size mismatch");

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
							cudnnHandle, rnn_desc, seqLength, y_descs, x455, y_descs, x459,
							dhy_desc, NULL, dcy_desc, NULL, w_desc, x112, hx_desc, x584,
							cx_desc, x585, dx_descs, x451, dhx_desc, NULL, dcx_desc, NULL,
							workspace, workspaceSize, x456, x457));
			};
			float* x587 = (float*)NULL;

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
				int32_t seqLength = x379;
				int32_t batchSize = x226;
				int32_t inputSize = x438;

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
				// printf("paramsSize: %zu\n", paramsSize / sizeof(float));
				assert(paramsSize / sizeof(float) == 4198400 && "Expected parameter size mismatch");

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
							cudnnHandle, rnn_desc, seqLength, x_descs, x445, hx_desc, x587,
							y_descs, x455, workspace, workspaceSize,
							dw_desc, x114, x456, x457));
			};
			// backprop for sum on dim op
			sum_grad<<<28, 512>>>(x437, x379, x226, 2, x438, x441, x451, x442, x438, 1, 2);
			;
			float* x591 = (float*)NULL;
			float* x592 = (float*)NULL;

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
				int32_t seqLength = x379;
				int32_t batchSize = x226;
				int32_t inputSize = x403;

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
				assert(paramsSize / sizeof(float) == 4788224 && "Expected parameter size mismatch");

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
							cudnnHandle, rnn_desc, seqLength, y_descs, x433, y_descs, x437,
							dhy_desc, NULL, dcy_desc, NULL, w_desc, x69, hx_desc, x591,
							cx_desc, x592, dx_descs, x425, dhx_desc, NULL, dcx_desc, NULL,
							workspace, workspaceSize, x434, x435));
			};
			float* x594 = (float*)NULL;

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
				int32_t seqLength = x379;
				int32_t batchSize = x226;
				int32_t inputSize = x403;

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
				// printf("paramsSize: %zu\n", paramsSize / sizeof(float));
				assert(paramsSize / sizeof(float) == 4788224 && "Expected parameter size mismatch");

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
							cudnnHandle, rnn_desc, seqLength, x_descs, x406, hx_desc, x594,
							y_descs, x433, workspace, workspaceSize,
							dw_desc, x71, x434, x435));
			};
			// backprop for permute WrappedArray(2, 0, 1)
			int* x597 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
			x597[2] = x407;
			x597[0] = x403;
			x597[1] = 1;
			x597[3] = 1;
			float* x602 = (float*)myMalloc(1 * sizeof(float));;
			x602[0] = 1.0f;
			int32_t x604 = x597[0];
			int32_t x605 = x597[1];
			int32_t x606 = x597[2];
			int32_t x607 = x597[3];

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
							in_desc, CUDNN_DATA_FLOAT,
							x226, x403, x379, 1,
							x604, x605, x606, x607));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
							out_desc, CUDNN_DATA_FLOAT,
							x226, x403, x379, 1,
							x404, x379, 1, 1));

				CUDNN_CALL(cudnnTransformTensor(
							cudnnHandle, x602, in_desc, x425, x602, out_desc, x400));
			};
			hardTanh_grad<<<28, 512>>>(x392, x400, x400, 0.0, 20.0, x382, true);
			float* x610 = (float*)myMalloc(1 * sizeof(float));;
			x610[0] = 0.0f;
			float* x612 = (float*)myMalloc(1 * sizeof(float));;
			x612[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x226, 32, x376, x379));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x226, 32, x376, x379));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 32, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x612, x612, x612, x612, in_desc, x385,
							out_desc, x400, in_desc, x391, sbmv_desc, x54,
							x56,x58, 1.0E-5, x393, x394));
			};
			// conv2D back-propagate
			float* x616 = (float*)myMalloc(1 * sizeof(float));;
			x616[0] = 1.0f;

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
							x226, 32, x338, x341));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x226, 32, x376, x379));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							10, 5, 2, 1, 1, 1,
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
							x616, filt_desc, x45, grad_out_desc, x391,
							conv_desc, algo, ws_data, ws_size,
							x616, grad_in_desc, x363));
			};
			float* x619 = (float*)myMalloc(1 * sizeof(float));;
			x619[0] = 1.0f;

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
							x226, 32, x376, x379));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x226, 32, x338, x341));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							10, 5, 2, 1, 1, 1,
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
							x619, in_desc, x355, grad_out_desc, x391,
							conv_desc, algo, ws_data, ws_size,
							x619, grad_filt_desc, x47));
			};
			hardTanh_grad<<<28, 512>>>(x355, x363, x363, 0.0, 20.0, x344, true);
			float* x623 = (float*)myMalloc(1 * sizeof(float));;
			x623[0] = 0.0f;
			float* x625 = (float*)myMalloc(1 * sizeof(float));;
			x625[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x226, 32, x338, x341));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x226, 32, x338, x341));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 32, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x625, x625, x625, x625, in_desc, x348,
							out_desc, x363, in_desc, x354, sbmv_desc, x28,
							x30,x32, 1.0E-5, x356, x357));
			};
			// conv2D back-propagate
			float* x629 = (float*)myMalloc(1 * sizeof(float));;
			x629[0] = 1.0f;

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
							x226, 32, x338, x341));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x226, 1, x311, x310));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							20, 5, 2, 2, 1, 1,
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
							x629, in_desc, x320, grad_out_desc, x354,
							conv_desc, algo, ws_data, ws_size,
							x629, grad_filt_desc, x20));
			};
			float x632 = x326[0];
			x305 += x632;
			int32_t x634 = x302;
			int32_t x636 = x634 % x635;
			bool x637 = x636 == 0;
			if (x637) {
				float x642 = x305;
				double x638 = (double)x634;
				double x639 = 100.0 * x638;
				double x641 = x639 / x640;
				float x643 = (float)x634;
				float x644 = x642 / x643;
				printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x298,x634,x231,x641,x644);
				fflush(stdout);
			} else {
			}
			int64_t x649 = (long)mallocAddr;
			int64_t x650 = x649 - x294;
			memset((void*)x294, 0, x650);
			mallocAddr = (void*)x294;
			int64_t x653 = (long)gpuMallocAddr;
			int64_t x654 = x653 - x295;
			cudaMemset((void*)x295, 0, x654);
			gpuMallocAddr = (void*)x295;

		}
		gettimeofday(&end_1, NULL);
		timeval_subtract(&diff_1, &end_1, &begin_1);;
		int64_t x661 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
		int64_t x662 = x661 / 1000LL;
		int64_t x664 = x661 / x663;
		printf("Training completed in %ldms (%ld us/images)\n",x662,x664);
		double x666 = (double)x661;
		double x667 = x666 / 1000000.0;
		x293[x298] = x667;
		float x669 = x305;
		float x671 = x669 / x670;
		double x672 = (double)x671;
		x292[x298] = x672;

	}
	gettimeofday(&end_0, NULL);
	timeval_subtract(&diff_0, &end_0, &begin_0);;
	int64_t x678 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
	sort(x293, x293 + 1);
	double x684 = x293[0];
	int64_t x685 = (long)fopen(x0, "w");
	fprintf((FILE *)x685, "unit: %s\n", "1 epoch");
	for(int x687=0; x687 < 1; x687++) {
		double x688 = x292[x687];
		fprintf((FILE *)x685, "%lf\n", x688);

	}
	fprintf((FILE *)x685, "run time: %lf %lf\n", x290, x684);
	fclose((FILE*)x685);
	// Backend cleanup.
	CUBLAS_CALL(cublasDestroy(cublasHandle));
	CUDA_CALL(cudaFree(gpuMallocBase));

	CUDNN_CALL(cudnnDestroy(cudnnHandle));
}
/*****************************************
  End of C Generated Code                  
 *******************************************/


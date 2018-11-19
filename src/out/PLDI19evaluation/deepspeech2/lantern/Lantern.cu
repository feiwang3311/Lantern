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
	printf("initial rnn input size is %d \n",672);
	printf("inputSize for batchRNN is %d\n",672);
	int32_t x63 = 0;
	float* x64 = (float*)myMalloc(3477504 * sizeof(float));;
	for(int x66=0; x66 < 3477504; x66++) {
		x64[x66] = 0.01f;

	}
	// Tensor 'toGPU' invocation.
	float* x71 = (float*)myGpuMalloc(3477504 * sizeof(float));
	CUDA_CALL(cudaMemcpy(x71, x64, 3477504 * sizeof(float), cudaMemcpyHostToDevice));
	float* x73 = (float*)myGpuMalloc(3477504 * sizeof(float));
	int32_t x74 = x63;
	float* x75 = x71+x74;
	float* x76 = x73+x74;
	x63 += 688128;
	int32_t x78 = x63;
	float* x79 = x71+x78;
	float* x80 = x73+x78;
	x63 += 1048576;
	int32_t x82 = x63;
	float* x83 = x71+x82;
	float* x84 = x73+x82;
	x63 += 688128;
	int32_t x86 = x63;
	float* x87 = x71+x86;
	float* x88 = x73+x86;
	x63 += 1048576;
	int32_t x90 = x63;
	float* x91 = x71+x90;
	float* x92 = x73+x90;
	x63 += 1024;
	int32_t x94 = x63;
	float* x95 = x71+x94;
	float* x96 = x73+x94;
	x63 += 1024;
	int32_t x98 = x63;
	float* x99 = x71+x98;
	float* x100 = x73+x98;
	x63 += 1024;
	int32_t x102 = x63;
	float* x103 = x71+x102;
	float* x104 = x73+x102;
	x63 += 1024;
	printf("inputSize for batchRNN is %d\n",1024);
	int32_t x107 = 0;
	float* x108 = (float*)myMalloc(4198400 * sizeof(float));;
	for(int x110=0; x110 < 4198400; x110++) {
		x108[x110] = 0.01f;

	}
	// Tensor 'toGPU' invocation.
	float* x115 = (float*)myGpuMalloc(4198400 * sizeof(float));
	CUDA_CALL(cudaMemcpy(x115, x108, 4198400 * sizeof(float), cudaMemcpyHostToDevice));
	float* x117 = (float*)myGpuMalloc(4198400 * sizeof(float));
	int32_t x118 = x107;
	float* x119 = x115+x118;
	float* x120 = x117+x118;
	x107 += 1048576;
	int32_t x122 = x107;
	float* x123 = x115+x122;
	float* x124 = x117+x122;
	x107 += 1048576;
	int32_t x126 = x107;
	float* x127 = x115+x126;
	float* x128 = x117+x126;
	x107 += 1048576;
	int32_t x130 = x107;
	float* x131 = x115+x130;
	float* x132 = x117+x130;
	x107 += 1048576;
	int32_t x134 = x107;
	float* x135 = x115+x134;
	float* x136 = x117+x134;
	x107 += 1024;
	int32_t x138 = x107;
	float* x139 = x115+x138;
	float* x140 = x117+x138;
	x107 += 1024;
	int32_t x142 = x107;
	float* x143 = x115+x142;
	float* x144 = x117+x142;
	x107 += 1024;
	int32_t x146 = x107;
	float* x147 = x115+x146;
	float* x148 = x117+x146;
	x107 += 1024;
	printf("inputSize for batchRNN is %d\n",1024);
	int32_t x151 = 0;
	float* x152 = (float*)myMalloc(4198400 * sizeof(float));;
	for(int x153=0; x153 < 4198400; x153++) {
		x152[x153] = 0.01f;

	}
	// Tensor 'toGPU' invocation.
	float* x158 = (float*)myGpuMalloc(4198400 * sizeof(float));
	CUDA_CALL(cudaMemcpy(x158, x152, 4198400 * sizeof(float), cudaMemcpyHostToDevice));
	float* x160 = (float*)myGpuMalloc(4198400 * sizeof(float));
	int32_t x161 = x151;
	float* x162 = x158+x161;
	float* x163 = x160+x161;
	x151 += 1048576;
	int32_t x165 = x151;
	float* x166 = x158+x165;
	float* x167 = x160+x165;
	x151 += 1048576;
	int32_t x169 = x151;
	float* x170 = x158+x169;
	float* x171 = x160+x169;
	x151 += 1048576;
	int32_t x173 = x151;
	float* x174 = x158+x173;
	float* x175 = x160+x173;
	x151 += 1048576;
	int32_t x177 = x151;
	float* x178 = x158+x177;
	float* x179 = x160+x177;
	x151 += 1024;
	int32_t x181 = x151;
	float* x182 = x158+x181;
	float* x183 = x160+x181;
	x151 += 1024;
	int32_t x185 = x151;
	float* x186 = x158+x185;
	float* x187 = x160+x185;
	x151 += 1024;
	int32_t x189 = x151;
	float* x190 = x158+x189;
	float* x191 = x160+x189;
	x151 += 1024;
	float* x193 = (float*)myMalloc(1024 * sizeof(float));;
	for(int x195=0; x195 < 1024; x195++) {
		x193[x195] = 1.0f;

	}
	// Tensor 'toGPU' invocation.
	float* x200 = (float*)myGpuMalloc(1024 * sizeof(float));
	CUDA_CALL(cudaMemcpy(x200, x193, 1024 * sizeof(float), cudaMemcpyHostToDevice));
	float* x202 = (float*)myGpuMalloc(1024 * sizeof(float));
	float* x203 = (float*)myGpuMalloc(1024 * sizeof(float));
	float* x204 = (float*)myGpuMalloc(1024 * sizeof(float));
	float* x205 = (float*)myGpuMalloc(1024 * sizeof(float));
	float* x206 = (float*)myGpuMalloc(1024 * sizeof(float));
	float* x207 = (float*)myMalloc(29696 * sizeof(float));;
	for(int x209=0; x209 < 29696; x209++) {
		float x210 = (float)rand()/RAND_MAX;
		float x211 = x210 - 0.5f;
		float x212 = x211 * 0.03125f;
		x207[x209] = x212;

	}
	// Tensor 'toGPU' invocation.
	float* x217 = (float*)myGpuMalloc(29696 * sizeof(float));
	CUDA_CALL(cudaMemcpy(x217, x207, 29696 * sizeof(float), cudaMemcpyHostToDevice));
	float* x219 = (float*)myGpuMalloc(29696 * sizeof(float));
	int32_t x220 = open("/scratch/wu636/training/speech_recognition/data/test/deepspeech_train.bin",0);
	int64_t x221 = fsize(x220);
	printf("file size is %ld\n",x221);
	char* x223 = (char*)mmap(0, x221, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x220, 0);
	int64_t x224 = (long)x223;
	int64_t x225 = x224;
	int64_t x226 = x225;
	int* x227 = (int32_t*) x226;
	int64_t x228 = (int64_t)4;
	x225 += x228;
	int32_t x230 = x227[0];
	int64_t x231 = x225;
	int* x232 = (int32_t*) x231;
	x225 += x228;
	int32_t x234 = x232[0];
	printf("data size is %d batches, %d batch size\n",200,x230);
	int* x237 = (int32_t*)myMalloc(200 * sizeof(int32_t));;
	int* x238 = (int32_t*)myMalloc(200 * sizeof(int32_t));;
	float** x239 = (float**)myMalloc(200 * sizeof(float*));;
	float** x240 = (float**)myMalloc(200 * sizeof(float*));;
	int** x241 = (int**)myMalloc(200 * sizeof(int*));;
	int** x242 = (int**)myMalloc(200 * sizeof(int*));;
	// load data by batchs
	int32_t x268 = 4 * x230;
	int64_t x269 = (int64_t)x268;
	for(int x245=0; x245 < 200; x245++) {
		int64_t x246 = x225;
		int* x247 = (int32_t*) x246;
		x225 += x228;
		int32_t x249 = x247[0];
		x237[x245] = x249;
		int64_t x251 = x225;
		int* x252 = (int32_t*) x251;
		x225 += x228;
		int32_t x254 = x252[0];
		x238[x245] = x254;
		int32_t x256 = x237[x245];
		int32_t x258 = x238[x245];
		int64_t x260 = x225;
		float* x261 = (float*) x260;
		int32_t x257 = x230 * x256;
		int32_t x259 = x257 * x258;
		int32_t x262 = 4 * x259;
		int64_t x263 = (int64_t)x262;
		x225 += x263;
		x239[x245] = x261;
		int64_t x266 = x225;
		float* x267 = (float*) x266;
		x225 += x269;
		x240[x245] = x267;
		int64_t x272 = x225;
		int* x273 = (int32_t*) x272;
		x225 += x269;
		x241[x245] = x273;
		int* x276 = x241[x245];
		int* x277 = x241[x245];
		int32_t x278 = accumulate(x276, x277 + x230, 0);
		int64_t x279 = x225;
		int* x280 = (int32_t*) x279;
		int32_t x281 = 4 * x278;
		int64_t x282 = (int64_t)x281;
		x225 += x282;
		x242[x245] = x280;

	}
	gettimeofday(&end_0, NULL);
	timeval_subtract(&diff_0, &end_0, &begin_0);;
	int64_t x289 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
	float x290 = (float)x289;
	float x291 = x290 / 1000000.0f;
	printf("Data reading (all prepare time) in %lf sec\n",x291);
	double* x293 = (double*)myMalloc(1 * sizeof(double));;
	double* x294 = (double*)myMalloc(1 * sizeof(double));;
	int64_t x295 = (long)mallocAddr;
	int64_t x296 = (long)gpuMallocAddr;
	// training loop starts here
	int32_t x344 = x230 * 32;
	bool x403 = x230 < 0;
	bool x436 = x230 > 0;
	bool x545 = 2 < 0;
	int32_t x530 = 2048 / 2;
	bool x551 = x530 < 0;
	bool x581 = 2 > 0;
	bool x586 = x530 > 0;
	bool x1247 = 29 < 0;
	bool x1277 = 29 > 0;
	bool x1344 = 1 < 0;
	bool x1379 = 1 > 0;
	int32_t x1857 = x230 * 20;
	int32_t x235 = x230 * 200;
	double x1862 = (double)x235;
	int64_t x1885 = (int64_t)x235;
	float x1892 = (float)x235;
	for(int x299=0; x299 < 1; x299++) {
		struct timeval begin_1, end_1, diff_1;
		int32_t x301 = 0;
		int32_t x302 = x301;
		int32_t x303 = x302;
		float x304 = 0.0f;
		float x305 = x304;
		float x306 = x305;
		int32_t x307 = x299 + 1;
		printf("Start training epoch %d\n",x307);
		gettimeofday(&begin_1, NULL);
		for(int x310=0; x310 < 200; x310++) {
			int32_t x311 = x238[x310];
			int32_t x312 = x237[x310];
			float* x313 = x239[x310];
			float* x316 = x240[x310];
			int* x317 = x242[x310];
			int* x318 = x241[x310];
			x303 += x230;
			// Tensor 'toGPU' invocation.
			int32_t x314 = x312 * x311;
			int32_t x315 = x230 * x314;
			float* x321 = (float*)myGpuMalloc(x315 * sizeof(float));
			CUDA_CALL(cudaMemcpy(x321, x313, x315 * sizeof(float), cudaMemcpyHostToDevice));
			float* x323 = (float*)myGpuMalloc(2 * sizeof(float));
			float* x324 = (float*)myGpuMalloc(1 * sizeof(float));
			float* x325 = (float*)myGpuMalloc(1 * sizeof(float));
			// allocate memory to save the final loss in CPU Tensor
			float* x327 = (float*)myMalloc(1 * sizeof(float));;
			bool x328 = x312 >= 41;
			bool x330;
			if (x328) {
				bool x329 = x311 >= 11;
				x330 = x329;
			} else {
				x330 = false;
			}
			if (x330) {
			} else {
				assert(false && "ERROR not specified");
			}
			int32_t x338 = x311 - 11;
			int32_t x339 = x338 / 2;
			int32_t x340 = x339 + 1;
			int32_t x335 = x312 - 41;
			int32_t x336 = x335 / 2;
			int32_t x337 = x336 + 1;
			int32_t x345 = x344 * x337;
			int32_t x346 = x345 * x340;
			float* x347 = (float*)myGpuMalloc(x346 * sizeof(float));
			float* x348 = (float*)myMalloc(1 * sizeof(float));;
			x348[0] = 0.0f;
			float* x350 = (float*)myMalloc(1 * sizeof(float));;
			x350[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x230, 1, x312, x311));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							32, 1, 41, 11));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x230, 32, x337, x340));

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
							x350, in_desc, x321, filt_desc, x18,
							conv_desc, algo, ws_data, ws_size,
							x348, out_desc, x347));
			};
			float* x353 = (float*)myGpuMalloc(x346 * sizeof(float));
			int32_t x341 = x337 * x340;
			int32_t x342 = 32 * x341;
			int32_t x343 = x230 * x342;
			float* x354 = (float*)myGpuMalloc(x343 * sizeof(float));
			float* x355 = (float*)myGpuMalloc(32 * sizeof(float));
			float* x356 = (float*)myGpuMalloc(32 * sizeof(float));
			float* x357 = (float*)myMalloc(1 * sizeof(float));;
			x357[0] = 0.0f;
			float* x359 = (float*)myMalloc(1 * sizeof(float));;
			x359[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x230, 32, x337, x340));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x230, 32, x337, x340));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 32, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x359, x357, in_desc, x347, out_desc, x354, sbmv_desc, x28,
							x31, 0.1, x33, x34, 1.0E-5,
							x355, x356));
			};
			float* x362 = (float*)myGpuMalloc(x346 * sizeof(float));
			hardTanh<<<28, 512>>>(x354, x354, 0.0, 20.0, true);
			bool x364 = x337 >= 21;
			bool x366;
			if (x364) {
				bool x365 = x340 >= 11;
				x366 = x365;
			} else {
				x366 = false;
			}
			if (x366) {
			} else {
				assert(false && "ERROR not specified");
			}
			int32_t x374 = x340 - 11;
			int32_t x375 = x374 / 1;
			int32_t x376 = x375 + 1;
			int32_t x371 = x337 - 21;
			int32_t x372 = x371 / 2;
			int32_t x373 = x372 + 1;
			int32_t x380 = x344 * x373;
			int32_t x381 = x380 * x376;
			float* x382 = (float*)myGpuMalloc(x381 * sizeof(float));
			float* x383 = (float*)myMalloc(1 * sizeof(float));;
			x383[0] = 0.0f;
			float* x385 = (float*)myMalloc(1 * sizeof(float));;
			x385[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x230, 32, x337, x340));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							32, 32, 21, 11));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x230, 32, x373, x376));

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
							x385, in_desc, x354, filt_desc, x45,
							conv_desc, algo, ws_data, ws_size,
							x383, out_desc, x382));
			};
			float* x388 = (float*)myGpuMalloc(x381 * sizeof(float));
			int32_t x377 = x373 * x376;
			int32_t x378 = 32 * x377;
			int32_t x379 = x230 * x378;
			float* x389 = (float*)myGpuMalloc(x379 * sizeof(float));
			float* x390 = (float*)myGpuMalloc(32 * sizeof(float));
			float* x391 = (float*)myGpuMalloc(32 * sizeof(float));
			float* x392 = (float*)myMalloc(1 * sizeof(float));;
			x392[0] = 0.0f;
			float* x394 = (float*)myMalloc(1 * sizeof(float));;
			x394[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x230, 32, x373, x376));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x230, 32, x373, x376));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 32, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x394, x392, in_desc, x382, out_desc, x389, sbmv_desc, x54,
							x57, 0.1, x59, x60, 1.0E-5,
							x390, x391));
			};
			float* x397 = (float*)myGpuMalloc(x381 * sizeof(float));
			hardTanh<<<28, 512>>>(x389, x389, 0.0, 20.0, true);
			// after conv ops
			int32_t x401 = 0;
			int32_t x402 = 1;
			if (x403) {
				x401 += 1;
			} else {
				x402 *= x230;
			}
			int32_t x400 = 32 * x373;
			bool x409 = x400 < 0;
			if (x409) {
				x401 += 1;
			} else {
				x402 *= x400;
			}
			bool x415 = x376 < 0;
			if (x415) {
				x401 += 1;
			} else {
				x402 *= x376;
			}
			int32_t x421 = x401;
			bool x422 = x421 >= 2;
			if (x422) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x427 = x421 == 0;
			if (x427) {
				int32_t x428 = x402;
				bool x429 = x428 == x379;
				if (x429) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			int32_t x440;
			if (x436) {
				x440 = x230;
			} else {
				int32_t x437 = x402;
				int32_t x438 = x379 / x437;
				x440 = x438;
			}
			bool x441 = x400 > 0;
			int32_t x445;
			if (x441) {
				x445 = x400;
			} else {
				int32_t x442 = x402;
				int32_t x443 = x379 / x442;
				x445 = x443;
			}
			bool x446 = x376 > 0;
			int32_t x450;
			if (x446) {
				x450 = x376;
			} else {
				int32_t x447 = x402;
				int32_t x448 = x379 / x447;
				x450 = x448;
			}
			int32_t x453 = 0;
			int32_t x454 = 1;
			if (x403) {
				x453 += 1;
			} else {
				x454 *= x230;
			}
			if (x409) {
				x453 += 1;
			} else {
				x454 *= x400;
			}
			if (x415) {
				x453 += 1;
			} else {
				x454 *= x376;
			}
			int32_t x470 = x453;
			bool x471 = x470 >= 2;
			if (x471) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x475 = x470 == 0;
			if (x475) {
				int32_t x476 = x454;
				bool x477 = x476 == x379;
				if (x477) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			int32_t x487;
			if (x436) {
				x487 = x230;
			} else {
				int32_t x484 = x454;
				int32_t x485 = x379 / x484;
				x487 = x485;
			}
			int32_t x491;
			if (x441) {
				x491 = x400;
			} else {
				int32_t x488 = x454;
				int32_t x489 = x379 / x488;
				x491 = x489;
			}
			int32_t x495;
			if (x446) {
				x495 = x376;
			} else {
				int32_t x492 = x454;
				int32_t x493 = x379 / x492;
				x495 = x493;
			}
			int32_t x451 = x445 * x450;
			int32_t x452 = x440 * x451;
			float* x498 = (float*)myGpuMalloc(x452 * sizeof(float));
			int* x501 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
			int32_t x499 = x440 * x445;
			x501[2] = x499;
			x501[0] = x445;
			x501[1] = 1;
			x501[3] = 1;
			float* x506 = (float*)myMalloc(1 * sizeof(float));;
			x506[0] = 1.0f;
			float* x508 = (float*)myMalloc(0 * sizeof(float));;
			x508[0] = 0.0f;
			int32_t x510 = x501[0];
			int32_t x511 = x501[1];
			int32_t x512 = x501[2];
			int32_t x513 = x501[3];

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
							in_desc, CUDNN_DATA_FLOAT,
							x440, x445, x450, 1,
							x451, x450, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
							out_desc, CUDNN_DATA_FLOAT,
							x440, x445, x450, 1,
							x510, x511, x512, x513));

				CUDNN_CALL(cudnnTransformTensor(
							cudnnHandle, x506, in_desc, x389, x508, out_desc, x498));
			};
			int32_t x515 = x450 * x440;
			int32_t x516 = x515 * x445;
			float* x517 = (float*)myGpuMalloc(x516 * sizeof(float));
			// after resize and permute
			float* x519 = (float*)NULL;
			float* x520 = (float*)NULL;
			float* x521 = (float*)NULL;
			int32_t x524 = x515 * 2048;
			float* x525 = (float*)myGpuMalloc(x524 * sizeof(float));
			float* x526 = (float*)NULL;
			int32_t x527 = 0;

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
				int32_t seqLength = x450;
				int32_t batchSize = x440;
				int32_t inputSize = x445;

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
				assert(paramsSize / sizeof(float) == 3477504 && "Expected parameter size mismatch");

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
				x526 = (float*)reserveSpace;
				x527 = (int)reserveSize;
				CUDNN_CALL(cudnnRNNForwardTraining(
							cudnnHandle, rnn_desc, seqLength, x_descs, x498,
							hx_desc,x519, cx_desc,x520, w_desc, x71, y_descs, x525,
							hy_desc,x521, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
			};
			float* x529 = (float*)myGpuMalloc(x524 * sizeof(float));
			int32_t x531 = 0;
			int32_t x532 = 1;
			bool x533 = x450 < 0;
			if (x533) {
				x531 += 1;
			} else {
				x532 *= x450;
			}
			bool x539 = x440 < 0;
			if (x539) {
				x531 += 1;
			} else {
				x532 *= x440;
			}
			if (x545) {
				x531 += 1;
			} else {
				x532 *= 2;
			}
			if (x551) {
				x531 += 1;
			} else {
				x532 *= x530;
			}
			int32_t x557 = x531;
			bool x558 = x557 >= 2;
			if (x558) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x562 = x557 == 0;
			int32_t x522 = x440 * 2048;
			int32_t x523 = x450 * x522;
			if (x562) {
				int32_t x563 = x532;
				bool x564 = x563 == x523;
				if (x564) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			bool x571 = x450 > 0;
			int32_t x575;
			if (x571) {
				x575 = x450;
			} else {
				int32_t x572 = x532;
				int32_t x573 = x523 / x572;
				x575 = x573;
			}
			bool x576 = x440 > 0;
			int32_t x580;
			if (x576) {
				x580 = x440;
			} else {
				int32_t x577 = x532;
				int32_t x578 = x523 / x577;
				x580 = x578;
			}
			int32_t x585;
			if (x581) {
				x585 = 2;
			} else {
				int32_t x582 = x532;
				int32_t x583 = x523 / x582;
				x585 = x583;
			}
			int32_t x590;
			if (x586) {
				x590 = x530;
			} else {
				int32_t x587 = x532;
				int32_t x588 = x523 / x587;
				x590 = x588;
			}
			int32_t x594 = 0;
			int32_t x595 = 1;
			if (x533) {
				x594 += 1;
			} else {
				x595 *= x450;
			}
			if (x539) {
				x594 += 1;
			} else {
				x595 *= x440;
			}
			if (x545) {
				x594 += 1;
			} else {
				x595 *= 2;
			}
			if (x551) {
				x594 += 1;
			} else {
				x595 *= x530;
			}
			int32_t x616 = x594;
			bool x617 = x616 >= 2;
			if (x617) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x621 = x616 == 0;
			if (x621) {
				int32_t x622 = x595;
				bool x623 = x622 == x523;
				if (x623) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			int32_t x633;
			if (x571) {
				x633 = x450;
			} else {
				int32_t x630 = x595;
				int32_t x631 = x523 / x630;
				x633 = x631;
			}
			int32_t x637;
			if (x576) {
				x637 = x440;
			} else {
				int32_t x634 = x595;
				int32_t x635 = x523 / x634;
				x637 = x635;
			}
			int32_t x641;
			if (x581) {
				x641 = 2;
			} else {
				int32_t x638 = x595;
				int32_t x639 = x523 / x638;
				x641 = x639;
			}
			int32_t x645;
			if (x586) {
				x645 = x530;
			} else {
				int32_t x642 = x595;
				int32_t x643 = x523 / x642;
				x645 = x643;
			}
			int32_t x649 = 0;
			int32_t x650 = 1;
			bool x651 = x575 < 0;
			if (x651) {
				x649 += 1;
			} else {
				x650 *= x575;
			}
			bool x657 = x580 < 0;
			if (x657) {
				x649 += 1;
			} else {
				x650 *= x580;
			}
			bool x663 = x585 < 0;
			if (x663) {
				x649 += 1;
			} else {
				x650 *= x585;
			}
			bool x669 = x590 < 0;
			if (x669) {
				x649 += 1;
			} else {
				x650 *= x590;
			}
			int32_t x675 = x649;
			bool x676 = x675 >= 2;
			if (x676) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x680 = x675 == 0;
			int32_t x591 = x585 * x590;
			int32_t x592 = x580 * x591;
			int32_t x593 = x575 * x592;
			if (x680) {
				int32_t x681 = x650;
				bool x682 = x681 == x593;
				if (x682) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			bool x689 = x575 > 0;
			int32_t x693;
			if (x689) {
				x693 = x575;
			} else {
				int32_t x690 = x650;
				int32_t x691 = x593 / x690;
				x693 = x691;
			}
			bool x694 = x580 > 0;
			int32_t x698;
			if (x694) {
				x698 = x580;
			} else {
				int32_t x695 = x650;
				int32_t x696 = x593 / x695;
				x698 = x696;
			}
			bool x699 = x585 > 0;
			int32_t x703;
			if (x699) {
				x703 = x585;
			} else {
				int32_t x700 = x650;
				int32_t x701 = x593 / x700;
				x703 = x701;
			}
			bool x704 = x590 > 0;
			int32_t x708;
			if (x704) {
				x708 = x590;
			} else {
				int32_t x705 = x650;
				int32_t x706 = x593 / x705;
				x708 = x706;
			}
			int32_t x714 = x693 * x698;
			int32_t x715 = x714 * x708;
			float* x716 = (float*)myGpuMalloc(x715 * sizeof(float));
			float* x717 = (float*)myMalloc(1 * sizeof(float));;
			x717[0] = 0.0f;
			float* x719 = (float*)myMalloc(1 * sizeof(float));;
			x719[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x693, x698, x703, x708));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x693, x698, 1, x708));

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
							x719, x_desc, x525, x717, out_desc, x716));
			};
			float* x722 = (float*)myGpuMalloc(x715 * sizeof(float));
			float* x723 = (float*)NULL;
			float* x724 = (float*)NULL;
			float* x725 = (float*)NULL;
			int32_t x728 = x714 * 2048;
			float* x729 = (float*)myGpuMalloc(x728 * sizeof(float));
			float* x730 = (float*)NULL;
			int32_t x731 = 0;

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
				int32_t seqLength = x693;
				int32_t batchSize = x698;
				int32_t inputSize = x708;

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
				x730 = (float*)reserveSpace;
				x731 = (int)reserveSize;
				CUDNN_CALL(cudnnRNNForwardTraining(
							cudnnHandle, rnn_desc, seqLength, x_descs, x716,
							hx_desc,x723, cx_desc,x724, w_desc, x115, y_descs, x729,
							hy_desc,x725, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
			};
			float* x733 = (float*)myGpuMalloc(x728 * sizeof(float));
			int32_t x734 = 0;
			int32_t x735 = 1;
			bool x736 = x693 < 0;
			if (x736) {
				x734 += 1;
			} else {
				x735 *= x693;
			}
			bool x742 = x698 < 0;
			if (x742) {
				x734 += 1;
			} else {
				x735 *= x698;
			}
			if (x545) {
				x734 += 1;
			} else {
				x735 *= 2;
			}
			if (x551) {
				x734 += 1;
			} else {
				x735 *= x530;
			}
			int32_t x758 = x734;
			bool x759 = x758 >= 2;
			if (x759) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x763 = x758 == 0;
			int32_t x726 = x698 * 2048;
			int32_t x727 = x693 * x726;
			if (x763) {
				int32_t x764 = x735;
				bool x765 = x764 == x727;
				if (x765) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			bool x772 = x693 > 0;
			int32_t x776;
			if (x772) {
				x776 = x693;
			} else {
				int32_t x773 = x735;
				int32_t x774 = x727 / x773;
				x776 = x774;
			}
			bool x777 = x698 > 0;
			int32_t x781;
			if (x777) {
				x781 = x698;
			} else {
				int32_t x778 = x735;
				int32_t x779 = x727 / x778;
				x781 = x779;
			}
			int32_t x785;
			if (x581) {
				x785 = 2;
			} else {
				int32_t x782 = x735;
				int32_t x783 = x727 / x782;
				x785 = x783;
			}
			int32_t x789;
			if (x586) {
				x789 = x530;
			} else {
				int32_t x786 = x735;
				int32_t x787 = x727 / x786;
				x789 = x787;
			}
			int32_t x793 = 0;
			int32_t x794 = 1;
			if (x736) {
				x793 += 1;
			} else {
				x794 *= x693;
			}
			if (x742) {
				x793 += 1;
			} else {
				x794 *= x698;
			}
			if (x545) {
				x793 += 1;
			} else {
				x794 *= 2;
			}
			if (x551) {
				x793 += 1;
			} else {
				x794 *= x530;
			}
			int32_t x815 = x793;
			bool x816 = x815 >= 2;
			if (x816) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x820 = x815 == 0;
			if (x820) {
				int32_t x821 = x794;
				bool x822 = x821 == x727;
				if (x822) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			int32_t x832;
			if (x772) {
				x832 = x693;
			} else {
				int32_t x829 = x794;
				int32_t x830 = x727 / x829;
				x832 = x830;
			}
			int32_t x836;
			if (x777) {
				x836 = x698;
			} else {
				int32_t x833 = x794;
				int32_t x834 = x727 / x833;
				x836 = x834;
			}
			int32_t x840;
			if (x581) {
				x840 = 2;
			} else {
				int32_t x837 = x794;
				int32_t x838 = x727 / x837;
				x840 = x838;
			}
			int32_t x844;
			if (x586) {
				x844 = x530;
			} else {
				int32_t x841 = x794;
				int32_t x842 = x727 / x841;
				x844 = x842;
			}
			int32_t x848 = 0;
			int32_t x849 = 1;
			bool x850 = x776 < 0;
			if (x850) {
				x848 += 1;
			} else {
				x849 *= x776;
			}
			bool x856 = x781 < 0;
			if (x856) {
				x848 += 1;
			} else {
				x849 *= x781;
			}
			bool x862 = x785 < 0;
			if (x862) {
				x848 += 1;
			} else {
				x849 *= x785;
			}
			bool x868 = x789 < 0;
			if (x868) {
				x848 += 1;
			} else {
				x849 *= x789;
			}
			int32_t x874 = x848;
			bool x875 = x874 >= 2;
			if (x875) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x879 = x874 == 0;
			int32_t x790 = x785 * x789;
			int32_t x791 = x781 * x790;
			int32_t x792 = x776 * x791;
			if (x879) {
				int32_t x880 = x849;
				bool x881 = x880 == x792;
				if (x881) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			bool x888 = x776 > 0;
			int32_t x892;
			if (x888) {
				x892 = x776;
			} else {
				int32_t x889 = x849;
				int32_t x890 = x792 / x889;
				x892 = x890;
			}
			bool x893 = x781 > 0;
			int32_t x897;
			if (x893) {
				x897 = x781;
			} else {
				int32_t x894 = x849;
				int32_t x895 = x792 / x894;
				x897 = x895;
			}
			bool x898 = x785 > 0;
			int32_t x902;
			if (x898) {
				x902 = x785;
			} else {
				int32_t x899 = x849;
				int32_t x900 = x792 / x899;
				x902 = x900;
			}
			bool x903 = x789 > 0;
			int32_t x907;
			if (x903) {
				x907 = x789;
			} else {
				int32_t x904 = x849;
				int32_t x905 = x792 / x904;
				x907 = x905;
			}
			int32_t x913 = x892 * x897;
			int32_t x914 = x913 * x907;
			float* x915 = (float*)myGpuMalloc(x914 * sizeof(float));
			float* x916 = (float*)myMalloc(1 * sizeof(float));;
			x916[0] = 0.0f;
			float* x918 = (float*)myMalloc(1 * sizeof(float));;
			x918[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x892, x897, x902, x907));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x892, x897, 1, x907));

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
							x918, x_desc, x729, x916, out_desc, x915));
			};
			float* x921 = (float*)myGpuMalloc(x914 * sizeof(float));
			float* x922 = (float*)NULL;
			float* x923 = (float*)NULL;
			float* x924 = (float*)NULL;
			int32_t x927 = x913 * 2048;
			float* x928 = (float*)myGpuMalloc(x927 * sizeof(float));
			float* x929 = (float*)NULL;
			int32_t x930 = 0;

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
				int32_t seqLength = x892;
				int32_t batchSize = x897;
				int32_t inputSize = x907;

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
				x929 = (float*)reserveSpace;
				x930 = (int)reserveSize;
				CUDNN_CALL(cudnnRNNForwardTraining(
							cudnnHandle, rnn_desc, seqLength, x_descs, x915,
							hx_desc,x922, cx_desc,x923, w_desc, x158, y_descs, x928,
							hy_desc,x924, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));
			};
			float* x932 = (float*)myGpuMalloc(x927 * sizeof(float));
			int32_t x933 = 0;
			int32_t x934 = 1;
			bool x935 = x892 < 0;
			if (x935) {
				x933 += 1;
			} else {
				x934 *= x892;
			}
			bool x941 = x897 < 0;
			if (x941) {
				x933 += 1;
			} else {
				x934 *= x897;
			}
			if (x545) {
				x933 += 1;
			} else {
				x934 *= 2;
			}
			if (x551) {
				x933 += 1;
			} else {
				x934 *= x530;
			}
			int32_t x957 = x933;
			bool x958 = x957 >= 2;
			if (x958) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x962 = x957 == 0;
			int32_t x925 = x897 * 2048;
			int32_t x926 = x892 * x925;
			if (x962) {
				int32_t x963 = x934;
				bool x964 = x963 == x926;
				if (x964) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			bool x971 = x892 > 0;
			int32_t x975;
			if (x971) {
				x975 = x892;
			} else {
				int32_t x972 = x934;
				int32_t x973 = x926 / x972;
				x975 = x973;
			}
			bool x976 = x897 > 0;
			int32_t x980;
			if (x976) {
				x980 = x897;
			} else {
				int32_t x977 = x934;
				int32_t x978 = x926 / x977;
				x980 = x978;
			}
			int32_t x984;
			if (x581) {
				x984 = 2;
			} else {
				int32_t x981 = x934;
				int32_t x982 = x926 / x981;
				x984 = x982;
			}
			int32_t x988;
			if (x586) {
				x988 = x530;
			} else {
				int32_t x985 = x934;
				int32_t x986 = x926 / x985;
				x988 = x986;
			}
			int32_t x992 = 0;
			int32_t x993 = 1;
			if (x935) {
				x992 += 1;
			} else {
				x993 *= x892;
			}
			if (x941) {
				x992 += 1;
			} else {
				x993 *= x897;
			}
			if (x545) {
				x992 += 1;
			} else {
				x993 *= 2;
			}
			if (x551) {
				x992 += 1;
			} else {
				x993 *= x530;
			}
			int32_t x1014 = x992;
			bool x1015 = x1014 >= 2;
			if (x1015) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x1019 = x1014 == 0;
			if (x1019) {
				int32_t x1020 = x993;
				bool x1021 = x1020 == x926;
				if (x1021) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			int32_t x1031;
			if (x971) {
				x1031 = x892;
			} else {
				int32_t x1028 = x993;
				int32_t x1029 = x926 / x1028;
				x1031 = x1029;
			}
			int32_t x1035;
			if (x976) {
				x1035 = x897;
			} else {
				int32_t x1032 = x993;
				int32_t x1033 = x926 / x1032;
				x1035 = x1033;
			}
			int32_t x1039;
			if (x581) {
				x1039 = 2;
			} else {
				int32_t x1036 = x993;
				int32_t x1037 = x926 / x1036;
				x1039 = x1037;
			}
			int32_t x1043;
			if (x586) {
				x1043 = x530;
			} else {
				int32_t x1040 = x993;
				int32_t x1041 = x926 / x1040;
				x1043 = x1041;
			}
			int32_t x1047 = 0;
			int32_t x1048 = 1;
			bool x1049 = x975 < 0;
			if (x1049) {
				x1047 += 1;
			} else {
				x1048 *= x975;
			}
			bool x1055 = x980 < 0;
			if (x1055) {
				x1047 += 1;
			} else {
				x1048 *= x980;
			}
			bool x1061 = x984 < 0;
			if (x1061) {
				x1047 += 1;
			} else {
				x1048 *= x984;
			}
			bool x1067 = x988 < 0;
			if (x1067) {
				x1047 += 1;
			} else {
				x1048 *= x988;
			}
			int32_t x1073 = x1047;
			bool x1074 = x1073 >= 2;
			if (x1074) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x1078 = x1073 == 0;
			int32_t x989 = x984 * x988;
			int32_t x990 = x980 * x989;
			int32_t x991 = x975 * x990;
			if (x1078) {
				int32_t x1079 = x1048;
				bool x1080 = x1079 == x991;
				if (x1080) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			bool x1087 = x975 > 0;
			int32_t x1091;
			if (x1087) {
				x1091 = x975;
			} else {
				int32_t x1088 = x1048;
				int32_t x1089 = x991 / x1088;
				x1091 = x1089;
			}
			bool x1092 = x980 > 0;
			int32_t x1096;
			if (x1092) {
				x1096 = x980;
			} else {
				int32_t x1093 = x1048;
				int32_t x1094 = x991 / x1093;
				x1096 = x1094;
			}
			bool x1097 = x984 > 0;
			int32_t x1101;
			if (x1097) {
				x1101 = x984;
			} else {
				int32_t x1098 = x1048;
				int32_t x1099 = x991 / x1098;
				x1101 = x1099;
			}
			bool x1102 = x988 > 0;
			int32_t x1106;
			if (x1102) {
				x1106 = x988;
			} else {
				int32_t x1103 = x1048;
				int32_t x1104 = x991 / x1103;
				x1106 = x1104;
			}
			int32_t x1112 = x1091 * x1096;
			int32_t x1113 = x1112 * x1106;
			float* x1114 = (float*)myGpuMalloc(x1113 * sizeof(float));
			float* x1115 = (float*)myMalloc(1 * sizeof(float));;
			x1115[0] = 0.0f;
			float* x1117 = (float*)myMalloc(1 * sizeof(float));;
			x1117[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x1091, x1096, x1101, x1106));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x1091, x1096, 1, x1106));

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
							x1117, x_desc, x928, x1115, out_desc, x1114));
			};
			float* x1120 = (float*)myGpuMalloc(x1113 * sizeof(float));
			// after RNN layers
			// after maybe lookahead
			int32_t x1123 = 0;
			int32_t x1124 = 1;
			bool x1125 = x1112 < 0;
			if (x1125) {
				x1123 += 1;
			} else {
				x1124 *= x1112;
			}
			bool x1131 = x1106 < 0;
			if (x1131) {
				x1123 += 1;
			} else {
				x1124 *= x1106;
			}
			int32_t x1137 = x1123;
			bool x1138 = x1137 >= 2;
			if (x1138) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x1142 = x1137 == 0;
			int32_t x1110 = x1096 * x1106;
			int32_t x1111 = x1091 * x1110;
			if (x1142) {
				int32_t x1143 = x1124;
				bool x1144 = x1143 == x1111;
				if (x1144) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			bool x1151 = x1112 > 0;
			int32_t x1155;
			if (x1151) {
				x1155 = x1112;
			} else {
				int32_t x1152 = x1124;
				int32_t x1153 = x1111 / x1152;
				x1155 = x1153;
			}
			bool x1156 = x1106 > 0;
			int32_t x1160;
			if (x1156) {
				x1160 = x1106;
			} else {
				int32_t x1157 = x1124;
				int32_t x1158 = x1111 / x1157;
				x1160 = x1158;
			}
			int32_t x1162 = 0;
			int32_t x1163 = 1;
			if (x1125) {
				x1162 += 1;
			} else {
				x1163 *= x1112;
			}
			if (x1131) {
				x1162 += 1;
			} else {
				x1163 *= x1106;
			}
			int32_t x1174 = x1162;
			bool x1175 = x1174 >= 2;
			if (x1175) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x1179 = x1174 == 0;
			if (x1179) {
				int32_t x1180 = x1163;
				bool x1181 = x1180 == x1111;
				if (x1181) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			int32_t x1191;
			if (x1151) {
				x1191 = x1112;
			} else {
				int32_t x1188 = x1163;
				int32_t x1189 = x1111 / x1188;
				x1191 = x1189;
			}
			int32_t x1195;
			if (x1156) {
				x1195 = x1106;
			} else {
				int32_t x1192 = x1163;
				int32_t x1193 = x1111 / x1192;
				x1195 = x1193;
			}
			bool x1197 = x1160 == 1024;
			if (x1197) {
			} else {
				assert(false && "BatchNorm1D input should be rank2, with shape 1 same as dimSize, got %d : %d");
			}
			bool x1202 = 1024 == x1160;
			if (x1202) {
			} else {
				assert(false && "scale should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(1155) x Sym(1160)");
			}
			if (x1202) {
			} else {
				assert(false && "bias should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(1155) x Sym(1160)");
			}
			if (x1202) {
			} else {
				assert(false && "runningMean should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(1155) x Sym(1160)");
			}
			if (x1202) {
			} else {
				assert(false && "runningVar should be rank 1 and have the same size as input dim 1, got  x Const(1024) and  x Sym(1155) x Sym(1160)");
			}
			int32_t x1161 = x1155 * x1160;
			float* x1216 = (float*)myGpuMalloc(x1161 * sizeof(float));
			float* x1217 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x1218 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x1219 = (float*)myMalloc(1 * sizeof(float));;
			x1219[0] = 0.0f;
			float* x1221 = (float*)myMalloc(1 * sizeof(float));;
			x1221[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x1155, x1160, 1, 1));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 1024, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,
							x1221, x1219, in_desc, x1114, in_desc, x1216, sbmv_desc, x200,
							x203, 0.1, x205, x206, 1.0E-5,
							x1217, x1218));
			};
			float* x1224 = (float*)myGpuMalloc(x1161 * sizeof(float));
			int32_t x1225 = x1155 * 29;
			float* x1226 = (float*)myGpuMalloc(x1225 * sizeof(float));
			float* x1227 = (float*)myMalloc(1 * sizeof(float));;
			x1227[0] = 0.0f;
			float* x1229 = (float*)myMalloc(1 * sizeof(float));;
			x1229[0] = 1.0f;
			CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 29,x1155,1024,x1229,x217,29,x1216,1024,x1227,x1226,29));
			float* x1232 = (float*)myGpuMalloc(x1225 * sizeof(float));
			int32_t x1233 = 0;
			int32_t x1234 = 1;
			bool x1235 = x1091 < 0;
			if (x1235) {
				x1233 += 1;
			} else {
				x1234 *= x1091;
			}
			bool x1241 = x1096 < 0;
			if (x1241) {
				x1233 += 1;
			} else {
				x1234 *= x1096;
			}
			if (x1247) {
				x1233 += 1;
			} else {
				x1234 *= 29;
			}
			int32_t x1253 = x1233;
			bool x1254 = x1253 >= 2;
			if (x1254) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x1258 = x1253 == 0;
			if (x1258) {
				int32_t x1259 = x1234;
				bool x1260 = x1259 == x1225;
				if (x1260) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			bool x1267 = x1091 > 0;
			int32_t x1271;
			if (x1267) {
				x1271 = x1091;
			} else {
				int32_t x1268 = x1234;
				int32_t x1269 = x1225 / x1268;
				x1271 = x1269;
			}
			bool x1272 = x1096 > 0;
			int32_t x1276;
			if (x1272) {
				x1276 = x1096;
			} else {
				int32_t x1273 = x1234;
				int32_t x1274 = x1225 / x1273;
				x1276 = x1274;
			}
			int32_t x1281;
			if (x1277) {
				x1281 = 29;
			} else {
				int32_t x1278 = x1234;
				int32_t x1279 = x1225 / x1278;
				x1281 = x1279;
			}
			int32_t x1284 = 0;
			int32_t x1285 = 1;
			if (x1235) {
				x1284 += 1;
			} else {
				x1285 *= x1091;
			}
			if (x1241) {
				x1284 += 1;
			} else {
				x1285 *= x1096;
			}
			if (x1247) {
				x1284 += 1;
			} else {
				x1285 *= 29;
			}
			int32_t x1301 = x1284;
			bool x1302 = x1301 >= 2;
			if (x1302) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x1306 = x1301 == 0;
			if (x1306) {
				int32_t x1307 = x1285;
				bool x1308 = x1307 == x1225;
				if (x1308) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			int32_t x1318;
			if (x1267) {
				x1318 = x1091;
			} else {
				int32_t x1315 = x1285;
				int32_t x1316 = x1225 / x1315;
				x1318 = x1316;
			}
			int32_t x1322;
			if (x1272) {
				x1322 = x1096;
			} else {
				int32_t x1319 = x1285;
				int32_t x1320 = x1225 / x1319;
				x1322 = x1320;
			}
			int32_t x1326;
			if (x1277) {
				x1326 = 29;
			} else {
				int32_t x1323 = x1285;
				int32_t x1324 = x1225 / x1323;
				x1326 = x1324;
			}
			int32_t x1330 = 0;
			int32_t x1331 = 1;
			int32_t x1329 = x1271 * x1276;
			bool x1332 = x1329 < 0;
			if (x1332) {
				x1330 += 1;
			} else {
				x1331 *= x1329;
			}
			bool x1338 = x1281 < 0;
			if (x1338) {
				x1330 += 1;
			} else {
				x1331 *= x1281;
			}
			if (x1344) {
				x1330 += 1;
			} else {
				x1331 *= 1;
			}
			if (x1344) {
				x1330 += 1;
			} else {
				x1331 *= 1;
			}
			int32_t x1355 = x1330;
			bool x1356 = x1355 >= 2;
			if (x1356) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x1360 = x1355 == 0;
			int32_t x1282 = x1276 * x1281;
			int32_t x1283 = x1271 * x1282;
			if (x1360) {
				int32_t x1361 = x1331;
				bool x1362 = x1361 == x1283;
				if (x1362) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			bool x1369 = x1329 > 0;
			int32_t x1373;
			if (x1369) {
				x1373 = x1329;
			} else {
				int32_t x1370 = x1331;
				int32_t x1371 = x1283 / x1370;
				x1373 = x1371;
			}
			bool x1374 = x1281 > 0;
			int32_t x1378;
			if (x1374) {
				x1378 = x1281;
			} else {
				int32_t x1375 = x1331;
				int32_t x1376 = x1283 / x1375;
				x1378 = x1376;
			}
			int32_t x1383;
			if (x1379) {
				x1383 = 1;
			} else {
				int32_t x1380 = x1331;
				int32_t x1381 = x1283 / x1380;
				x1383 = x1381;
			}
			int32_t x1387;
			if (x1379) {
				x1387 = 1;
			} else {
				int32_t x1384 = x1331;
				int32_t x1385 = x1283 / x1384;
				x1387 = x1385;
			}
			float* x1391 = (float*)myMalloc(1 * sizeof(float));;
			x1391[0] = 0.0f;
			float* x1393 = (float*)myMalloc(1 * sizeof(float));;
			x1393[0] = 1.0f;
			int32_t x1388 = x1383 * x1387;
			int32_t x1389 = x1378 * x1388;
			int32_t x1390 = x1373 * x1389;
			float* x1395 = (float*)myGpuMalloc(x1390 * sizeof(float));

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x1373, x1378, x1383, x1387));
				CUDNN_CALL(cudnnSoftmaxForward(
							cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
							x1393, x_desc, x1226, x1391, x_desc, x1395));
			};
			int32_t x1397 = 0;
			int32_t x1398 = 1;
			bool x1399 = x1271 < 0;
			if (x1399) {
				x1397 += 1;
			} else {
				x1398 *= x1271;
			}
			bool x1405 = x1276 < 0;
			if (x1405) {
				x1397 += 1;
			} else {
				x1398 *= x1276;
			}
			if (x1338) {
				x1397 += 1;
			} else {
				x1398 *= x1281;
			}
			int32_t x1416 = x1397;
			bool x1417 = x1416 >= 2;
			if (x1417) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x1421 = x1416 == 0;
			if (x1421) {
				int32_t x1422 = x1398;
				bool x1423 = x1422 == x1390;
				if (x1423) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			bool x1430 = x1271 > 0;
			int32_t x1434;
			if (x1430) {
				x1434 = x1271;
			} else {
				int32_t x1431 = x1398;
				int32_t x1432 = x1390 / x1431;
				x1434 = x1432;
			}
			bool x1435 = x1276 > 0;
			int32_t x1439;
			if (x1435) {
				x1439 = x1276;
			} else {
				int32_t x1436 = x1398;
				int32_t x1437 = x1390 / x1436;
				x1439 = x1437;
			}
			int32_t x1443;
			if (x1374) {
				x1443 = x1281;
			} else {
				int32_t x1440 = x1398;
				int32_t x1441 = x1390 / x1440;
				x1443 = x1441;
			}
			int32_t x1446 = x1434 * x1439;
			int32_t x1447 = x1446 * x1443;
			float* x1448 = (float*)myGpuMalloc(x1447 * sizeof(float));
			// before CTC loss
			int* x1450 = (int32_t*)myMalloc(x1439 * sizeof(int32_t));;
			float x1454 = (float)x1434;
			for(int x1452=0; x1452 < x1439; x1452++) {
				float x1453 = x316[x1452];
				float x1455 = x1453 * x1454;
				int32_t x1456 = (int)x1455;
				x1450[x1452] = x1456;

			}
			bool x1460 = x1439 <= 256;
			if (x1460) {
			} else {
				printf("'cudnnGetCTCLossWorkspaceSize' requires batch size less than 256, got %d\n\n",x1439);
				assert(false && "");
			}
			float* x1466 = (float*)myGpuMalloc(x1439 * sizeof(float));

			{
				cudnnTensorDescriptor_t probs_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&probs_desc));
				int probs_dims[] = {x1434, x1439, x1443};
				int probs_strides[] = {probs_dims[1] * probs_dims[2], probs_dims[2], 1};
				CUDNN_CALL(cudnnSetTensorNdDescriptor(
							probs_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, probs_dims, probs_strides));

				cudnnTensorDescriptor_t grad_desc = probs_desc;

				cudnnCTCLossDescriptor_t ctc_desc;
				CUDNN_CALL(cudnnCreateCTCLossDescriptor(&ctc_desc));
				CUDNN_CALL(cudnnSetCTCLossDescriptor(ctc_desc, CUDNN_DATA_FLOAT));
				size_t wsSize;
				CUDNN_CALL(cudnnGetCTCLossWorkspaceSize(
							cudnnHandle, probs_desc, grad_desc, x317, x318, x1450,
							CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, &wsSize));
				void *ws = myGpuMalloc(wsSize);

				CUDNN_CALL(cudnnCTCLoss(
							cudnnHandle, probs_desc, x1395, x317, x318, x1450,
							x1466, grad_desc, x1448, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc, ws, wsSize));
			};
			float* x1468 = (float*)myGpuMalloc(1 * sizeof(float));
			float* x1469 = (float*)myMalloc(1 * sizeof(float));;
			x1469[0] = 0.0f;
			float* x1471 = (float*)myMalloc(1 * sizeof(float));;
			x1471[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x1439, 1, 1, 1));

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
							x1471, x_desc, x1466, x1469, out_desc, x1468));
			};
			// after CTC loss
			float* x1475 = (float*)myGpuMalloc(1 * sizeof(float));
			// make sure the size of loss is 1
			arrayFill_greg<<<28, 512>>>(x1475, 1.0f, 1);
			// backend is lantern.TensorDslCudnn$BackendCudnn@69232924
			CUDA_CALL(cudaMemcpy(x327, x1468, 1 * sizeof(float), cudaMemcpyDeviceToHost));
			int32_t x1480 = 0;
			int32_t x1481 = 1;
			if (x1332) {
				x1480 += 1;
			} else {
				x1481 *= x1329;
			}
			if (x1338) {
				x1480 += 1;
			} else {
				x1481 *= x1281;
			}
			if (x1344) {
				x1480 += 1;
			} else {
				x1481 *= 1;
			}
			if (x1344) {
				x1480 += 1;
			} else {
				x1481 *= 1;
			}
			int32_t x1502 = x1480;
			bool x1503 = x1502 >= 2;
			if (x1503) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x1507 = x1502 == 0;
			if (x1507) {
				int32_t x1508 = x1481;
				bool x1509 = x1508 == x1283;
				if (x1509) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			int32_t x1519;
			if (x1369) {
				x1519 = x1329;
			} else {
				int32_t x1516 = x1481;
				int32_t x1517 = x1283 / x1516;
				x1519 = x1517;
			}
			int32_t x1523;
			if (x1374) {
				x1523 = x1281;
			} else {
				int32_t x1520 = x1481;
				int32_t x1521 = x1283 / x1520;
				x1523 = x1521;
			}
			int32_t x1527;
			if (x1379) {
				x1527 = 1;
			} else {
				int32_t x1524 = x1481;
				int32_t x1525 = x1283 / x1524;
				x1527 = x1525;
			}
			int32_t x1531;
			if (x1379) {
				x1531 = 1;
			} else {
				int32_t x1528 = x1481;
				int32_t x1529 = x1283 / x1528;
				x1531 = x1529;
			}
			int32_t x1535 = 0;
			int32_t x1536 = 1;
			if (x1332) {
				x1535 += 1;
			} else {
				x1536 *= x1329;
			}
			if (x1338) {
				x1535 += 1;
			} else {
				x1536 *= x1281;
			}
			if (x1344) {
				x1535 += 1;
			} else {
				x1536 *= 1;
			}
			if (x1344) {
				x1535 += 1;
			} else {
				x1536 *= 1;
			}
			int32_t x1557 = x1535;
			bool x1558 = x1557 >= 2;
			if (x1558) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x1562 = x1557 == 0;
			if (x1562) {
				int32_t x1563 = x1536;
				int32_t x1327 = x1322 * x1326;
				int32_t x1328 = x1318 * x1327;
				bool x1564 = x1563 == x1328;
				if (x1564) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			int32_t x1574;
			if (x1369) {
				x1574 = x1329;
			} else {
				int32_t x1571 = x1536;
				int32_t x1327 = x1322 * x1326;
				int32_t x1328 = x1318 * x1327;
				int32_t x1572 = x1328 / x1571;
				x1574 = x1572;
			}
			int32_t x1578;
			if (x1374) {
				x1578 = x1281;
			} else {
				int32_t x1575 = x1536;
				int32_t x1327 = x1322 * x1326;
				int32_t x1328 = x1318 * x1327;
				int32_t x1576 = x1328 / x1575;
				x1578 = x1576;
			}
			int32_t x1582;
			if (x1379) {
				x1582 = 1;
			} else {
				int32_t x1579 = x1536;
				int32_t x1327 = x1322 * x1326;
				int32_t x1328 = x1318 * x1327;
				int32_t x1580 = x1328 / x1579;
				x1582 = x1580;
			}
			int32_t x1586;
			if (x1379) {
				x1586 = 1;
			} else {
				int32_t x1583 = x1536;
				int32_t x1327 = x1322 * x1326;
				int32_t x1328 = x1318 * x1327;
				int32_t x1584 = x1328 / x1583;
				x1586 = x1584;
			}
			int32_t x1590 = 0;
			int32_t x1591 = 1;
			bool x1592 = x1446 < 0;
			if (x1592) {
				x1590 += 1;
			} else {
				x1591 *= x1446;
			}
			bool x1598 = x1443 < 0;
			if (x1598) {
				x1590 += 1;
			} else {
				x1591 *= x1443;
			}
			if (x1344) {
				x1590 += 1;
			} else {
				x1591 *= 1;
			}
			if (x1344) {
				x1590 += 1;
			} else {
				x1591 *= 1;
			}
			int32_t x1614 = x1590;
			bool x1615 = x1614 >= 2;
			if (x1615) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x1619 = x1614 == 0;
			if (x1619) {
				int32_t x1620 = x1591;
				int32_t x1444 = x1439 * x1443;
				int32_t x1445 = x1434 * x1444;
				bool x1621 = x1620 == x1445;
				if (x1621) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			bool x1628 = x1446 > 0;
			int32_t x1632;
			if (x1628) {
				x1632 = x1446;
			} else {
				int32_t x1629 = x1591;
				int32_t x1444 = x1439 * x1443;
				int32_t x1445 = x1434 * x1444;
				int32_t x1630 = x1445 / x1629;
				x1632 = x1630;
			}
			bool x1633 = x1443 > 0;
			int32_t x1637;
			if (x1633) {
				x1637 = x1443;
			} else {
				int32_t x1634 = x1591;
				int32_t x1444 = x1439 * x1443;
				int32_t x1445 = x1434 * x1444;
				int32_t x1635 = x1445 / x1634;
				x1637 = x1635;
			}
			int32_t x1641;
			if (x1379) {
				x1641 = 1;
			} else {
				int32_t x1638 = x1591;
				int32_t x1444 = x1439 * x1443;
				int32_t x1445 = x1434 * x1444;
				int32_t x1639 = x1445 / x1638;
				x1641 = x1639;
			}
			int32_t x1645;
			if (x1379) {
				x1645 = 1;
			} else {
				int32_t x1642 = x1591;
				int32_t x1444 = x1439 * x1443;
				int32_t x1445 = x1434 * x1444;
				int32_t x1643 = x1445 / x1642;
				x1645 = x1643;
			}
			int32_t x1649 = 0;
			int32_t x1650 = 1;
			if (x1592) {
				x1649 += 1;
			} else {
				x1650 *= x1446;
			}
			if (x1598) {
				x1649 += 1;
			} else {
				x1650 *= x1443;
			}
			if (x1344) {
				x1649 += 1;
			} else {
				x1650 *= 1;
			}
			if (x1344) {
				x1649 += 1;
			} else {
				x1650 *= 1;
			}
			int32_t x1671 = x1649;
			bool x1672 = x1671 >= 2;
			if (x1672) {
				assert(false && "cannot have 2 or more -1s in resize!!");
			} else {
			}
			bool x1676 = x1671 == 0;
			if (x1676) {
				int32_t x1677 = x1650;
				int32_t x1444 = x1439 * x1443;
				int32_t x1445 = x1434 * x1444;
				bool x1678 = x1677 == x1445;
				if (x1678) {
				} else {
					assert(false && "must same size!!");
				}
			} else {
			}
			int32_t x1688;
			if (x1628) {
				x1688 = x1446;
			} else {
				int32_t x1685 = x1650;
				int32_t x1444 = x1439 * x1443;
				int32_t x1445 = x1434 * x1444;
				int32_t x1686 = x1445 / x1685;
				x1688 = x1686;
			}
			int32_t x1692;
			if (x1633) {
				x1692 = x1443;
			} else {
				int32_t x1689 = x1650;
				int32_t x1444 = x1439 * x1443;
				int32_t x1445 = x1434 * x1444;
				int32_t x1690 = x1445 / x1689;
				x1692 = x1690;
			}
			int32_t x1696;
			if (x1379) {
				x1696 = 1;
			} else {
				int32_t x1693 = x1650;
				int32_t x1444 = x1439 * x1443;
				int32_t x1445 = x1434 * x1444;
				int32_t x1694 = x1445 / x1693;
				x1696 = x1694;
			}
			int32_t x1700;
			if (x1379) {
				x1700 = 1;
			} else {
				int32_t x1697 = x1650;
				int32_t x1444 = x1439 * x1443;
				int32_t x1445 = x1434 * x1444;
				int32_t x1698 = x1445 / x1697;
				x1700 = x1698;
			}
			bool x1704 = x1519 == x1632;
			bool x1706;
			if (x1704) {
				bool x1705 = x1523 == x1637;
				x1706 = x1705;
			} else {
				x1706 = false;
			}
			bool x1708;
			if (x1706) {
				bool x1707 = x1527 == x1641;
				x1708 = x1707;
			} else {
				x1708 = false;
			}
			bool x1710;
			if (x1708) {
				bool x1709 = x1531 == x1645;
				x1710 = x1709;
			} else {
				x1710 = false;
			}
			if (x1710) {
			} else {
				printf("$errorPrefix: tensor shapes are not equal %s, %s\n\n"," x Sym(1519) x Sym(1523) x Sym(1527) x Sym(1531)"," x Sym(1632) x Sym(1637) x Sym(1641) x Sym(1645)");
				assert(false && "");
			}
			bool x1716 = x1574 == x1688;
			bool x1718;
			if (x1716) {
				bool x1717 = x1578 == x1692;
				x1718 = x1717;
			} else {
				x1718 = false;
			}
			bool x1720;
			if (x1718) {
				bool x1719 = x1582 == x1696;
				x1720 = x1719;
			} else {
				x1720 = false;
			}
			bool x1722;
			if (x1720) {
				bool x1721 = x1586 == x1700;
				x1722 = x1721;
			} else {
				x1722 = false;
			}
			if (x1722) {
			} else {
				printf("$errorPrefix: tensor shapes are not equal %s, %s\n\n"," x Sym(1574) x Sym(1578) x Sym(1582) x Sym(1586)"," x Sym(1688) x Sym(1692) x Sym(1696) x Sym(1700)");
				assert(false && "");
			}
			float* x1728 = (float*)myMalloc(1 * sizeof(float));;
			x1728[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x1519, x1523, x1527, x1531));
				CUDNN_CALL(cudnnSoftmaxBackward(
							cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
							x1728, x_desc, x1395, x_desc, x1448,
							x1728, x_desc, x1232));
			};
			float* x1731 = (float*)myMalloc(1 * sizeof(float));;
			x1731[0] = 0.0f;
			float* x1733 = (float*)myMalloc(1 * sizeof(float));;
			x1733[0] = 1.0f;
			// backprop of matrix-matrix-dot
			CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, x1160,x1155,29,x1733,x217,29,x1232,29,x1733,x1224,x1160));
			CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 29,x1160,x1155,x1733,x1232,29,x1216,x1160,x1733,x219,29));
			float* x1738 = (float*)myMalloc(1 * sizeof(float));;
			x1738[0] = 0.0f;
			float* x1740 = (float*)myMalloc(1 * sizeof(float));;
			x1740[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x1155, x1160, 1, 1));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 1024, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,
							x1740, x1740, x1740, x1740, in_desc, x1114,
							in_desc, x1224, in_desc, x1120, sbmv_desc, x200,
							x202,x204, 1.0E-5, x1217, x1218));
			};
			// backprop for sum on dim op
			sum_grad<<<28, 512>>>(x932, x975, x980, x984, x988, x991, x1120, x1110, x1106, 1, 2);
			;
			float* x1745 = (float*)NULL;
			float* x1746 = (float*)NULL;

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
				int32_t seqLength = x892;
				int32_t batchSize = x897;
				int32_t inputSize = x907;

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
							cudnnHandle, rnn_desc, seqLength, y_descs, x928, y_descs, x932,
							dhy_desc, NULL, dcy_desc, NULL, w_desc, x158, hx_desc, x1745,
							cx_desc, x1746, dx_descs, x921, dhx_desc, NULL, dcx_desc, NULL,
							workspace, workspaceSize, x929, x930));
			};
			float* x1748 = (float*)NULL;

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
				int32_t seqLength = x892;
				int32_t batchSize = x897;
				int32_t inputSize = x907;

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
							cudnnHandle, rnn_desc, seqLength, x_descs, x915, hx_desc, x1748,
							y_descs, x928, workspace, workspaceSize,
							dw_desc, x160, x929, x930));
			};
			// backprop for sum on dim op
			int32_t x911 = x897 * x907;
			sum_grad<<<28, 512>>>(x733, x776, x781, x785, x789, x792, x921, x911, x907, 1, 2);
			;
			float* x1752 = (float*)NULL;
			float* x1753 = (float*)NULL;

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
				int32_t seqLength = x693;
				int32_t batchSize = x698;
				int32_t inputSize = x708;

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
							cudnnHandle, rnn_desc, seqLength, y_descs, x729, y_descs, x733,
							dhy_desc, NULL, dcy_desc, NULL, w_desc, x115, hx_desc, x1752,
							cx_desc, x1753, dx_descs, x722, dhx_desc, NULL, dcx_desc, NULL,
							workspace, workspaceSize, x730, x731));
			};
			float* x1755 = (float*)NULL;

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
				int32_t seqLength = x693;
				int32_t batchSize = x698;
				int32_t inputSize = x708;

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
							cudnnHandle, rnn_desc, seqLength, x_descs, x716, hx_desc, x1755,
							y_descs, x729, workspace, workspaceSize,
							dw_desc, x117, x730, x731));
			};
			// backprop for sum on dim op
			int32_t x712 = x698 * x708;
			sum_grad<<<28, 512>>>(x529, x575, x580, x585, x590, x593, x722, x712, x708, 1, 2);
			;
			float* x1759 = (float*)NULL;
			float* x1760 = (float*)NULL;

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
				int32_t seqLength = x450;
				int32_t batchSize = x440;
				int32_t inputSize = x445;

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
				assert(paramsSize / sizeof(float) == 3477504 && "Expected parameter size mismatch");

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
							cudnnHandle, rnn_desc, seqLength, y_descs, x525, y_descs, x529,
							dhy_desc, NULL, dcy_desc, NULL, w_desc, x71, hx_desc, x1759,
							cx_desc, x1760, dx_descs, x517, dhx_desc, NULL, dcx_desc, NULL,
							workspace, workspaceSize, x526, x527));
			};
			float* x1762 = (float*)NULL;

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
				int32_t seqLength = x450;
				int32_t batchSize = x440;
				int32_t inputSize = x445;

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
				assert(paramsSize / sizeof(float) == 3477504 && "Expected parameter size mismatch");

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
							cudnnHandle, rnn_desc, seqLength, x_descs, x498, hx_desc, x1762,
							y_descs, x525, workspace, workspaceSize,
							dw_desc, x73, x526, x527));
			};
			// backprop for permute WrappedArray(2, 0, 1)
			int* x1765 = (int32_t*)myMalloc(4 * sizeof(int32_t));;
			x1765[2] = x499;
			x1765[0] = x445;
			x1765[1] = 1;
			x1765[3] = 1;
			float* x1770 = (float*)myMalloc(1 * sizeof(float));;
			x1770[0] = 1.0f;
			int32_t x1772 = x1765[0];
			int32_t x1773 = x1765[1];
			int32_t x1774 = x1765[2];
			int32_t x1775 = x1765[3];

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
							in_desc, CUDNN_DATA_FLOAT,
							x440, x445, x450, 1,
							x1772, x1773, x1774, x1775));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptorEx(
							out_desc, CUDNN_DATA_FLOAT,
							x440, x445, x450, 1,
							x451, x450, 1, 1));

				CUDNN_CALL(cudnnTransformTensor(
							cudnnHandle, x1770, in_desc, x517, x1770, out_desc, x397));
			};
			hardTanh_grad<<<28, 512>>>(x389, x397, x397, 0.0, 20.0, x379, true);
			float* x1778 = (float*)myMalloc(1 * sizeof(float));;
			x1778[0] = 0.0f;
			float* x1780 = (float*)myMalloc(1 * sizeof(float));;
			x1780[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x230, 32, x373, x376));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x230, 32, x373, x376));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 32, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x1780, x1780, x1780, x1780, in_desc, x382,
							out_desc, x397, in_desc, x388, sbmv_desc, x54,
							x56,x58, 1.0E-5, x390, x391));
			};
			// conv2D back-propagate
			float* x1784 = (float*)myMalloc(1 * sizeof(float));;
			x1784[0] = 1.0f;

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
							x230, 32, x337, x340));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x230, 32, x373, x376));

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
							x1784, filt_desc, x45, grad_out_desc, x388,
							conv_desc, algo, ws_data, ws_size,
							x1784, grad_in_desc, x362));
			};
			float* x1787 = (float*)myMalloc(1 * sizeof(float));;
			x1787[0] = 1.0f;

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
							x230, 32, x373, x376));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x230, 32, x337, x340));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1787, in_desc, x354, grad_out_desc, x388,
							conv_desc, algo, ws_data, ws_size,
							x1787, grad_filt_desc, x47));
			};
			hardTanh_grad<<<28, 512>>>(x354, x362, x362, 0.0, 20.0, x343, true);
			float* x1791 = (float*)myMalloc(1 * sizeof(float));;
			x1791[0] = 0.0f;
			float* x1793 = (float*)myMalloc(1 * sizeof(float));;
			x1793[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x230, 32, x337, x340));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x230, 32, x337, x340));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 32, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x1793, x1793, x1793, x1793, in_desc, x347,
							out_desc, x362, in_desc, x353, sbmv_desc, x28,
							x30,x32, 1.0E-5, x355, x356));
			};
			// conv2D back-propagate
			float* x1797 = (float*)myMalloc(1 * sizeof(float));;
			x1797[0] = 1.0f;

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
							x230, 32, x337, x340));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							x230, 1, x312, x311));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1797, in_desc, x321, grad_out_desc, x353,
							conv_desc, algo, ws_data, ws_size,
							x1797, grad_filt_desc, x20));
			};
			float x1800 = x327[0];
			x306 += x1800;
			float* x1802 = (float*)myMalloc(1 * sizeof(float));;
			x1802[0] = 1.0f;
			float* x1804 = (float*)myMalloc(1 * sizeof(float));;
			x1804[0] = -3.0E-8f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 451,32,x1802,x18,451,x1804, x20, 451, x18,451));
			arrayFill_greg<<<28, 512>>>(x20, 0.0f, 14432);
			float* x1808 = (float*)myMalloc(1 * sizeof(float));;
			x1808[0] = 1.0f;
			float* x1810 = (float*)myMalloc(1 * sizeof(float));;
			x1810[0] = -3.0E-8f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 7392,32,x1808,x45,7392,x1810, x47, 7392, x45,7392));
			arrayFill_greg<<<28, 512>>>(x47, 0.0f, 236544);
			float* x1814 = (float*)myMalloc(1 * sizeof(float));;
			x1814[0] = 1.0f;
			float* x1816 = (float*)myMalloc(1 * sizeof(float));;
			x1816[0] = -3.0E-8f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1814,x54,1,x1816, x56, 1, x54,1));
			arrayFill_greg<<<28, 512>>>(x56, 0.0f, 32);
			float* x1820 = (float*)myMalloc(1 * sizeof(float));;
			x1820[0] = 1.0f;
			float* x1822 = (float*)myMalloc(1 * sizeof(float));;
			x1822[0] = -3.0E-8f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1820,x57,1,x1822, x58, 1, x57,1));
			arrayFill_greg<<<28, 512>>>(x58, 0.0f, 32);
			float* x1826 = (float*)myMalloc(1 * sizeof(float));;
			x1826[0] = 1.0f;
			float* x1828 = (float*)myMalloc(1 * sizeof(float));;
			x1828[0] = -3.0E-8f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1826,x31,1,x1828, x32, 1, x31,1));
			arrayFill_greg<<<28, 512>>>(x32, 0.0f, 32);
			float* x1832 = (float*)myMalloc(1 * sizeof(float));;
			x1832[0] = 1.0f;
			float* x1834 = (float*)myMalloc(1 * sizeof(float));;
			x1834[0] = -3.0E-8f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1832,x28,1,x1834, x30, 1, x28,1));
			arrayFill_greg<<<28, 512>>>(x30, 0.0f, 32);
			float* x1838 = (float*)myMalloc(1 * sizeof(float));;
			x1838[0] = 1.0f;
			float* x1840 = (float*)myMalloc(1 * sizeof(float));;
			x1840[0] = -3.0E-8f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x1838,x200,1,x1840, x202, 1, x200,1));
			arrayFill_greg<<<28, 512>>>(x202, 0.0f, 1024);
			float* x1844 = (float*)myMalloc(1 * sizeof(float));;
			x1844[0] = 1.0f;
			float* x1846 = (float*)myMalloc(1 * sizeof(float));;
			x1846[0] = -3.0E-8f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x1844,x203,1,x1846, x204, 1, x203,1));
			arrayFill_greg<<<28, 512>>>(x204, 0.0f, 1024);
			float* x1850 = (float*)myMalloc(1 * sizeof(float));;
			x1850[0] = 1.0f;
			float* x1852 = (float*)myMalloc(1 * sizeof(float));;
			x1852[0] = -3.0E-8f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 29,1024,x1850,x217,29,x1852, x219, 29, x217,29));
			arrayFill_greg<<<28, 512>>>(x219, 0.0f, 29696);
			int32_t x1856 = x303;
			int32_t x1858 = x1856 % x1857;
			bool x1859 = x1858 == 0;
			if (x1859) {
				float x1864 = x306;
				double x1860 = (double)x1856;
				double x1861 = 100.0 * x1860;
				double x1863 = x1861 / x1862;
				float x1865 = (float)x1856;
				float x1866 = x1864 / x1865;
				printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x299,x1856,x235,x1863,x1866);
				fflush(stdout);
			} else {
			}
			int64_t x1871 = (long)mallocAddr;
			int64_t x1872 = x1871 - x295;
			memset((void*)x295, 0, x1872);
			mallocAddr = (void*)x295;
			int64_t x1875 = (long)gpuMallocAddr;
			int64_t x1876 = x1875 - x296;
			cudaMemset((void*)x296, 0, x1876);
			gpuMallocAddr = (void*)x296;

		}
		gettimeofday(&end_1, NULL);
		timeval_subtract(&diff_1, &end_1, &begin_1);;
		int64_t x1883 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
		int64_t x1884 = x1883 / 1000LL;
		int64_t x1886 = x1883 / x1885;
		printf("Training completed in %ldms (%ld us/images)\n",x1884,x1886);
		double x1888 = (double)x1883;
		double x1889 = x1888 / 1000000.0;
		x294[x299] = x1889;
		float x1891 = x306;
		float x1893 = x1891 / x1892;
		double x1894 = (double)x1893;
		x293[x299] = x1894;

	}
	gettimeofday(&end_0, NULL);
	timeval_subtract(&diff_0, &end_0, &begin_0);;
	int64_t x1900 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
	sort(x294, x294 + 1);
	double x1906 = x294[0];
	int64_t x1907 = (long)fopen(x0, "w");
	fprintf((FILE *)x1907, "unit: %s\n", "1 epoch");
	for(int x1909=0; x1909 < 1; x1909++) {
		double x1910 = x293[x1909];
		fprintf((FILE *)x1907, "%lf\n", x1910);

	}
	fprintf((FILE *)x1907, "run time: %lf %lf\n", x291, x1906);
	fclose((FILE*)x1907);
	// Backend cleanup.
	CUBLAS_CALL(cublasDestroy(cublasHandle));
	CUDA_CALL(cudaFree(gpuMallocBase));

	CUDNN_CALL(cudnnDestroy(cudnnHandle));
}
/*****************************************
  End of C Generated Code
 *******************************************/


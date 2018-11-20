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
	int32_t x7 = open("../../cifar10_data/cifar-10-batches-bin/data_batch_1.bin",0);
	int64_t x8 = fsize(x7);
	int64_t x10 = x8 / 3073LL;
	int32_t x11 = (int32_t)x10;
	int32_t x12 = x11 * 3072;
	float* x13 = (float*)myMalloc(x12 * sizeof(float));;
	int* x14 = (int32_t*)myMalloc(x11 * sizeof(int32_t));;
	char* x9 = (char*)mmap(0, x8, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x7, 0);
	for(int x16=0; x16 < x11; x16++) {
		int32_t x17 = x16 * 3073;
		char x18 = x9[x17];
		int32_t x19 = (int32_t)(unsigned char)x18;
		x14[x16] = x19;
		int32_t x25 = x17 + 1;
		int32_t x23 = x16 * 3072;
		for(int x22=0; x22 < 3072; x22++) {
			int32_t x26 = x25 + x22;
			char x27 = x9[x26];
			int32_t x24 = x23 + x22;
			float x28 = (float)(unsigned char)x27;
			float x29 = x28 / 255.0f;
			x13[x24] = x29;

		}

	}
	gettimeofday(&end_0, NULL);
	timeval_subtract(&diff_0, &end_0, &begin_0);;
	int64_t x37 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
	float x38 = (float)x37;
	float x39 = x38 / 1000000.0f;
	printf("Data normalized (all prepare time) in %lf sec\n",x39);
	// Tensor 'toGPU' invocation.
	float* x313 = (float*)myGpuMalloc(262144 * sizeof(float));
	int32_t x42 = open("/u/data/u99/wang603/TiarkMlEnv/Lantern/src/out/PLDI19evaluation/resnet50/resnet50.onnx.bin",0);
	int64_t x43 = fsize(x42);
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
	double* x1381 = (double*)myMalloc(4 * sizeof(double));;
	int64_t x1382 = (long)mallocAddr;
	int64_t x1383 = (long)gpuMallocAddr;
	// training loop starts here
	int32_t x1394 = x11 / 64;
	bool x1411 = 34 >= 3;
	bool x1412;
	if (x1411) {
		x1412 = x1411;
	} else {
		x1412 = false;
	}
	int32_t x1417 = 31 / 1;
	int32_t x1418 = x1417 + 1;
	int32_t x1422 = 4096 * x1418;
	int32_t x1423 = x1422 * x1418;
	int32_t x1419 = x1418 * x1418;
	int32_t x1420 = 64 * x1419;
	int32_t x1421 = 64 * x1420;
	int32_t x1449 = x1418 - 2;
	int32_t x1450 = x1449 / 2;
	int32_t x1451 = x1450 + 1;
	int32_t x1455 = 4096 * x1451;
	int32_t x1456 = x1455 * x1451;
	bool x1460 = x1451 >= 1;
	bool x1461;
	if (x1460) {
		x1461 = x1460;
	} else {
		x1461 = false;
	}
	int32_t x1466 = x1450 / 1;
	int32_t x1467 = x1466 + 1;
	int32_t x1471 = 4096 * x1467;
	int32_t x1472 = x1471 * x1467;
	int32_t x1468 = x1467 * x1467;
	int32_t x1469 = 64 * x1468;
	int32_t x1470 = 64 * x1469;
	int32_t x1494 = x1467 + 2;
	bool x1495 = x1494 >= 3;
	bool x1496;
	if (x1495) {
		x1496 = x1495;
	} else {
		x1496 = false;
	}
	int32_t x1501 = x1494 - 3;
	int32_t x1502 = x1501 / 1;
	int32_t x1503 = x1502 + 1;
	int32_t x1507 = 4096 * x1503;
	int32_t x1508 = x1507 * x1503;
	int32_t x1504 = x1503 * x1503;
	int32_t x1505 = 64 * x1504;
	int32_t x1506 = 64 * x1505;
	bool x1530 = x1503 >= 1;
	bool x1531;
	if (x1530) {
		x1531 = x1530;
	} else {
		x1531 = false;
	}
	int32_t x1536 = x1502 / 1;
	int32_t x1537 = x1536 + 1;
	int32_t x1541 = 16384 * x1537;
	int32_t x1542 = x1541 * x1537;
	int32_t x1538 = x1537 * x1537;
	int32_t x1539 = 256 * x1538;
	int32_t x1540 = 64 * x1539;
	int32_t x1564 = 16384 * x1467;
	int32_t x1565 = x1564 * x1467;
	int32_t x1562 = 256 * x1468;
	int32_t x1563 = 64 * x1562;
	bool x1582 = x1467 == 1;
	bool x1583 = x1467 == x1537;
	bool x1584 = x1582 || x1583;
	bool x1585;
	if (x1584) {
		x1585 = x1584;
	} else {
		x1585 = false;
	}
	bool x1600 = x1537 >= 1;
	bool x1601;
	if (x1600) {
		x1601 = x1600;
	} else {
		x1601 = false;
	}
	int32_t x1606 = x1536 / 1;
	int32_t x1607 = x1606 + 1;
	int32_t x1611 = 4096 * x1607;
	int32_t x1612 = x1611 * x1607;
	int32_t x1608 = x1607 * x1607;
	int32_t x1609 = 64 * x1608;
	int32_t x1610 = 64 * x1609;
	int32_t x1634 = x1607 + 2;
	bool x1635 = x1634 >= 3;
	bool x1636;
	if (x1635) {
		x1636 = x1635;
	} else {
		x1636 = false;
	}
	int32_t x1641 = x1634 - 3;
	int32_t x1642 = x1641 / 1;
	int32_t x1643 = x1642 + 1;
	int32_t x1647 = 4096 * x1643;
	int32_t x1648 = x1647 * x1643;
	int32_t x1644 = x1643 * x1643;
	int32_t x1645 = 64 * x1644;
	int32_t x1646 = 64 * x1645;
	bool x1670 = x1643 >= 1;
	bool x1671;
	if (x1670) {
		x1671 = x1670;
	} else {
		x1671 = false;
	}
	int32_t x1676 = x1642 / 1;
	int32_t x1677 = x1676 + 1;
	int32_t x1681 = 16384 * x1677;
	int32_t x1682 = x1681 * x1677;
	int32_t x1678 = x1677 * x1677;
	int32_t x1679 = 256 * x1678;
	int32_t x1680 = 64 * x1679;
	bool x1699 = x1537 == 1;
	bool x1700 = x1537 == x1677;
	bool x1701 = x1699 || x1700;
	bool x1702;
	if (x1701) {
		x1702 = x1701;
	} else {
		x1702 = false;
	}
	bool x1717 = x1677 >= 1;
	bool x1718;
	if (x1717) {
		x1718 = x1717;
	} else {
		x1718 = false;
	}
	int32_t x1723 = x1676 / 1;
	int32_t x1724 = x1723 + 1;
	int32_t x1728 = 4096 * x1724;
	int32_t x1729 = x1728 * x1724;
	int32_t x1725 = x1724 * x1724;
	int32_t x1726 = 64 * x1725;
	int32_t x1727 = 64 * x1726;
	int32_t x1751 = x1724 + 2;
	bool x1752 = x1751 >= 3;
	bool x1753;
	if (x1752) {
		x1753 = x1752;
	} else {
		x1753 = false;
	}
	int32_t x1758 = x1751 - 3;
	int32_t x1759 = x1758 / 1;
	int32_t x1760 = x1759 + 1;
	int32_t x1764 = 4096 * x1760;
	int32_t x1765 = x1764 * x1760;
	int32_t x1761 = x1760 * x1760;
	int32_t x1762 = 64 * x1761;
	int32_t x1763 = 64 * x1762;
	bool x1787 = x1760 >= 1;
	bool x1788;
	if (x1787) {
		x1788 = x1787;
	} else {
		x1788 = false;
	}
	int32_t x1793 = x1759 / 1;
	int32_t x1794 = x1793 + 1;
	int32_t x1798 = 16384 * x1794;
	int32_t x1799 = x1798 * x1794;
	int32_t x1795 = x1794 * x1794;
	int32_t x1796 = 256 * x1795;
	int32_t x1797 = 64 * x1796;
	bool x1816 = x1677 == 1;
	bool x1817 = x1677 == x1794;
	bool x1818 = x1816 || x1817;
	bool x1819;
	if (x1818) {
		x1819 = x1818;
	} else {
		x1819 = false;
	}
	bool x1834 = x1794 >= 1;
	bool x1835;
	if (x1834) {
		x1835 = x1834;
	} else {
		x1835 = false;
	}
	int32_t x1840 = x1793 / 1;
	int32_t x1841 = x1840 + 1;
	int32_t x1845 = 8192 * x1841;
	int32_t x1846 = x1845 * x1841;
	int32_t x1842 = x1841 * x1841;
	int32_t x1843 = 128 * x1842;
	int32_t x1844 = 64 * x1843;
	int32_t x1868 = x1841 + 2;
	bool x1869 = x1868 >= 3;
	bool x1870;
	if (x1869) {
		x1870 = x1869;
	} else {
		x1870 = false;
	}
	int32_t x1875 = x1868 - 3;
	int32_t x1876 = x1875 / 2;
	int32_t x1877 = x1876 + 1;
	int32_t x1881 = 8192 * x1877;
	int32_t x1882 = x1881 * x1877;
	int32_t x1878 = x1877 * x1877;
	int32_t x1879 = 128 * x1878;
	int32_t x1880 = 64 * x1879;
	bool x1904 = x1877 >= 1;
	bool x1905;
	if (x1904) {
		x1905 = x1904;
	} else {
		x1905 = false;
	}
	int32_t x1910 = x1876 / 1;
	int32_t x1911 = x1910 + 1;
	int32_t x1915 = 32768 * x1911;
	int32_t x1916 = x1915 * x1911;
	int32_t x1912 = x1911 * x1911;
	int32_t x1913 = 512 * x1912;
	int32_t x1914 = 64 * x1913;
	int32_t x1936 = x1793 / 2;
	int32_t x1937 = x1936 + 1;
	int32_t x1941 = 32768 * x1937;
	int32_t x1942 = x1941 * x1937;
	int32_t x1938 = x1937 * x1937;
	int32_t x1939 = 512 * x1938;
	int32_t x1940 = 64 * x1939;
	bool x1959 = x1937 == 1;
	bool x1960 = x1937 == x1911;
	bool x1961 = x1959 || x1960;
	bool x1962;
	if (x1961) {
		x1962 = x1961;
	} else {
		x1962 = false;
	}
	bool x1977 = x1911 >= 1;
	bool x1978;
	if (x1977) {
		x1978 = x1977;
	} else {
		x1978 = false;
	}
	int32_t x1983 = x1910 / 1;
	int32_t x1984 = x1983 + 1;
	int32_t x1988 = 8192 * x1984;
	int32_t x1989 = x1988 * x1984;
	int32_t x1985 = x1984 * x1984;
	int32_t x1986 = 128 * x1985;
	int32_t x1987 = 64 * x1986;
	int32_t x2011 = x1984 + 2;
	bool x2012 = x2011 >= 3;
	bool x2013;
	if (x2012) {
		x2013 = x2012;
	} else {
		x2013 = false;
	}
	int32_t x2018 = x2011 - 3;
	int32_t x2019 = x2018 / 1;
	int32_t x2020 = x2019 + 1;
	int32_t x2024 = 8192 * x2020;
	int32_t x2025 = x2024 * x2020;
	int32_t x2021 = x2020 * x2020;
	int32_t x2022 = 128 * x2021;
	int32_t x2023 = 64 * x2022;
	bool x2047 = x2020 >= 1;
	bool x2048;
	if (x2047) {
		x2048 = x2047;
	} else {
		x2048 = false;
	}
	int32_t x2053 = x2019 / 1;
	int32_t x2054 = x2053 + 1;
	int32_t x2058 = 32768 * x2054;
	int32_t x2059 = x2058 * x2054;
	int32_t x2055 = x2054 * x2054;
	int32_t x2056 = 512 * x2055;
	int32_t x2057 = 64 * x2056;
	bool x2076 = x1911 == 1;
	bool x2077 = x1911 == x2054;
	bool x2078 = x2076 || x2077;
	bool x2079;
	if (x2078) {
		x2079 = x2078;
	} else {
		x2079 = false;
	}
	bool x2094 = x2054 >= 1;
	bool x2095;
	if (x2094) {
		x2095 = x2094;
	} else {
		x2095 = false;
	}
	int32_t x2100 = x2053 / 1;
	int32_t x2101 = x2100 + 1;
	int32_t x2105 = 8192 * x2101;
	int32_t x2106 = x2105 * x2101;
	int32_t x2102 = x2101 * x2101;
	int32_t x2103 = 128 * x2102;
	int32_t x2104 = 64 * x2103;
	int32_t x2128 = x2101 + 2;
	bool x2129 = x2128 >= 3;
	bool x2130;
	if (x2129) {
		x2130 = x2129;
	} else {
		x2130 = false;
	}
	int32_t x2135 = x2128 - 3;
	int32_t x2136 = x2135 / 1;
	int32_t x2137 = x2136 + 1;
	int32_t x2141 = 8192 * x2137;
	int32_t x2142 = x2141 * x2137;
	int32_t x2138 = x2137 * x2137;
	int32_t x2139 = 128 * x2138;
	int32_t x2140 = 64 * x2139;
	bool x2164 = x2137 >= 1;
	bool x2165;
	if (x2164) {
		x2165 = x2164;
	} else {
		x2165 = false;
	}
	int32_t x2170 = x2136 / 1;
	int32_t x2171 = x2170 + 1;
	int32_t x2175 = 32768 * x2171;
	int32_t x2176 = x2175 * x2171;
	int32_t x2172 = x2171 * x2171;
	int32_t x2173 = 512 * x2172;
	int32_t x2174 = 64 * x2173;
	bool x2193 = x2054 == 1;
	bool x2194 = x2054 == x2171;
	bool x2195 = x2193 || x2194;
	bool x2196;
	if (x2195) {
		x2196 = x2195;
	} else {
		x2196 = false;
	}
	bool x2211 = x2171 >= 1;
	bool x2212;
	if (x2211) {
		x2212 = x2211;
	} else {
		x2212 = false;
	}
	int32_t x2217 = x2170 / 1;
	int32_t x2218 = x2217 + 1;
	int32_t x2222 = 8192 * x2218;
	int32_t x2223 = x2222 * x2218;
	int32_t x2219 = x2218 * x2218;
	int32_t x2220 = 128 * x2219;
	int32_t x2221 = 64 * x2220;
	int32_t x2245 = x2218 + 2;
	bool x2246 = x2245 >= 3;
	bool x2247;
	if (x2246) {
		x2247 = x2246;
	} else {
		x2247 = false;
	}
	int32_t x2252 = x2245 - 3;
	int32_t x2253 = x2252 / 1;
	int32_t x2254 = x2253 + 1;
	int32_t x2258 = 8192 * x2254;
	int32_t x2259 = x2258 * x2254;
	int32_t x2255 = x2254 * x2254;
	int32_t x2256 = 128 * x2255;
	int32_t x2257 = 64 * x2256;
	bool x2281 = x2254 >= 1;
	bool x2282;
	if (x2281) {
		x2282 = x2281;
	} else {
		x2282 = false;
	}
	int32_t x2287 = x2253 / 1;
	int32_t x2288 = x2287 + 1;
	int32_t x2292 = 32768 * x2288;
	int32_t x2293 = x2292 * x2288;
	int32_t x2289 = x2288 * x2288;
	int32_t x2290 = 512 * x2289;
	int32_t x2291 = 64 * x2290;
	bool x2310 = x2171 == 1;
	bool x2311 = x2171 == x2288;
	bool x2312 = x2310 || x2311;
	bool x2313;
	if (x2312) {
		x2313 = x2312;
	} else {
		x2313 = false;
	}
	bool x2328 = x2288 >= 1;
	bool x2329;
	if (x2328) {
		x2329 = x2328;
	} else {
		x2329 = false;
	}
	int32_t x2334 = x2287 / 1;
	int32_t x2335 = x2334 + 1;
	int32_t x2339 = 16384 * x2335;
	int32_t x2340 = x2339 * x2335;
	int32_t x2336 = x2335 * x2335;
	int32_t x2337 = 256 * x2336;
	int32_t x2338 = 64 * x2337;
	int32_t x2362 = x2335 + 2;
	bool x2363 = x2362 >= 3;
	bool x2364;
	if (x2363) {
		x2364 = x2363;
	} else {
		x2364 = false;
	}
	int32_t x2369 = x2362 - 3;
	int32_t x2370 = x2369 / 2;
	int32_t x2371 = x2370 + 1;
	int32_t x2375 = 16384 * x2371;
	int32_t x2376 = x2375 * x2371;
	int32_t x2372 = x2371 * x2371;
	int32_t x2373 = 256 * x2372;
	int32_t x2374 = 64 * x2373;
	bool x2398 = x2371 >= 1;
	bool x2399;
	if (x2398) {
		x2399 = x2398;
	} else {
		x2399 = false;
	}
	int32_t x2404 = x2370 / 1;
	int32_t x2405 = x2404 + 1;
	int32_t x2409 = 65536 * x2405;
	int32_t x2410 = x2409 * x2405;
	int32_t x2406 = x2405 * x2405;
	int32_t x2407 = 1024 * x2406;
	int32_t x2408 = 64 * x2407;
	int32_t x2430 = x2287 / 2;
	int32_t x2431 = x2430 + 1;
	int32_t x2435 = 65536 * x2431;
	int32_t x2436 = x2435 * x2431;
	int32_t x2432 = x2431 * x2431;
	int32_t x2433 = 1024 * x2432;
	int32_t x2434 = 64 * x2433;
	bool x2453 = x2431 == 1;
	bool x2454 = x2431 == x2405;
	bool x2455 = x2453 || x2454;
	bool x2456;
	if (x2455) {
		x2456 = x2455;
	} else {
		x2456 = false;
	}
	bool x2471 = x2405 >= 1;
	bool x2472;
	if (x2471) {
		x2472 = x2471;
	} else {
		x2472 = false;
	}
	int32_t x2477 = x2404 / 1;
	int32_t x2478 = x2477 + 1;
	int32_t x2482 = 16384 * x2478;
	int32_t x2483 = x2482 * x2478;
	int32_t x2479 = x2478 * x2478;
	int32_t x2480 = 256 * x2479;
	int32_t x2481 = 64 * x2480;
	int32_t x2505 = x2478 + 2;
	bool x2506 = x2505 >= 3;
	bool x2507;
	if (x2506) {
		x2507 = x2506;
	} else {
		x2507 = false;
	}
	int32_t x2512 = x2505 - 3;
	int32_t x2513 = x2512 / 1;
	int32_t x2514 = x2513 + 1;
	int32_t x2518 = 16384 * x2514;
	int32_t x2519 = x2518 * x2514;
	int32_t x2515 = x2514 * x2514;
	int32_t x2516 = 256 * x2515;
	int32_t x2517 = 64 * x2516;
	bool x2541 = x2514 >= 1;
	bool x2542;
	if (x2541) {
		x2542 = x2541;
	} else {
		x2542 = false;
	}
	int32_t x2547 = x2513 / 1;
	int32_t x2548 = x2547 + 1;
	int32_t x2552 = 65536 * x2548;
	int32_t x2553 = x2552 * x2548;
	int32_t x2549 = x2548 * x2548;
	int32_t x2550 = 1024 * x2549;
	int32_t x2551 = 64 * x2550;
	bool x2570 = x2405 == 1;
	bool x2571 = x2405 == x2548;
	bool x2572 = x2570 || x2571;
	bool x2573;
	if (x2572) {
		x2573 = x2572;
	} else {
		x2573 = false;
	}
	bool x2588 = x2548 >= 1;
	bool x2589;
	if (x2588) {
		x2589 = x2588;
	} else {
		x2589 = false;
	}
	int32_t x2594 = x2547 / 1;
	int32_t x2595 = x2594 + 1;
	int32_t x2599 = 16384 * x2595;
	int32_t x2600 = x2599 * x2595;
	int32_t x2596 = x2595 * x2595;
	int32_t x2597 = 256 * x2596;
	int32_t x2598 = 64 * x2597;
	int32_t x2622 = x2595 + 2;
	bool x2623 = x2622 >= 3;
	bool x2624;
	if (x2623) {
		x2624 = x2623;
	} else {
		x2624 = false;
	}
	int32_t x2629 = x2622 - 3;
	int32_t x2630 = x2629 / 1;
	int32_t x2631 = x2630 + 1;
	int32_t x2635 = 16384 * x2631;
	int32_t x2636 = x2635 * x2631;
	int32_t x2632 = x2631 * x2631;
	int32_t x2633 = 256 * x2632;
	int32_t x2634 = 64 * x2633;
	bool x2658 = x2631 >= 1;
	bool x2659;
	if (x2658) {
		x2659 = x2658;
	} else {
		x2659 = false;
	}
	int32_t x2664 = x2630 / 1;
	int32_t x2665 = x2664 + 1;
	int32_t x2669 = 65536 * x2665;
	int32_t x2670 = x2669 * x2665;
	int32_t x2666 = x2665 * x2665;
	int32_t x2667 = 1024 * x2666;
	int32_t x2668 = 64 * x2667;
	bool x2687 = x2548 == 1;
	bool x2688 = x2548 == x2665;
	bool x2689 = x2687 || x2688;
	bool x2690;
	if (x2689) {
		x2690 = x2689;
	} else {
		x2690 = false;
	}
	bool x2705 = x2665 >= 1;
	bool x2706;
	if (x2705) {
		x2706 = x2705;
	} else {
		x2706 = false;
	}
	int32_t x2711 = x2664 / 1;
	int32_t x2712 = x2711 + 1;
	int32_t x2716 = 16384 * x2712;
	int32_t x2717 = x2716 * x2712;
	int32_t x2713 = x2712 * x2712;
	int32_t x2714 = 256 * x2713;
	int32_t x2715 = 64 * x2714;
	int32_t x2739 = x2712 + 2;
	bool x2740 = x2739 >= 3;
	bool x2741;
	if (x2740) {
		x2741 = x2740;
	} else {
		x2741 = false;
	}
	int32_t x2746 = x2739 - 3;
	int32_t x2747 = x2746 / 1;
	int32_t x2748 = x2747 + 1;
	int32_t x2752 = 16384 * x2748;
	int32_t x2753 = x2752 * x2748;
	int32_t x2749 = x2748 * x2748;
	int32_t x2750 = 256 * x2749;
	int32_t x2751 = 64 * x2750;
	bool x2775 = x2748 >= 1;
	bool x2776;
	if (x2775) {
		x2776 = x2775;
	} else {
		x2776 = false;
	}
	int32_t x2781 = x2747 / 1;
	int32_t x2782 = x2781 + 1;
	int32_t x2786 = 65536 * x2782;
	int32_t x2787 = x2786 * x2782;
	int32_t x2783 = x2782 * x2782;
	int32_t x2784 = 1024 * x2783;
	int32_t x2785 = 64 * x2784;
	bool x2804 = x2665 == 1;
	bool x2805 = x2665 == x2782;
	bool x2806 = x2804 || x2805;
	bool x2807;
	if (x2806) {
		x2807 = x2806;
	} else {
		x2807 = false;
	}
	bool x2822 = x2782 >= 1;
	bool x2823;
	if (x2822) {
		x2823 = x2822;
	} else {
		x2823 = false;
	}
	int32_t x2828 = x2781 / 1;
	int32_t x2829 = x2828 + 1;
	int32_t x2833 = 16384 * x2829;
	int32_t x2834 = x2833 * x2829;
	int32_t x2830 = x2829 * x2829;
	int32_t x2831 = 256 * x2830;
	int32_t x2832 = 64 * x2831;
	int32_t x2856 = x2829 + 2;
	bool x2857 = x2856 >= 3;
	bool x2858;
	if (x2857) {
		x2858 = x2857;
	} else {
		x2858 = false;
	}
	int32_t x2863 = x2856 - 3;
	int32_t x2864 = x2863 / 1;
	int32_t x2865 = x2864 + 1;
	int32_t x2869 = 16384 * x2865;
	int32_t x2870 = x2869 * x2865;
	int32_t x2866 = x2865 * x2865;
	int32_t x2867 = 256 * x2866;
	int32_t x2868 = 64 * x2867;
	bool x2892 = x2865 >= 1;
	bool x2893;
	if (x2892) {
		x2893 = x2892;
	} else {
		x2893 = false;
	}
	int32_t x2898 = x2864 / 1;
	int32_t x2899 = x2898 + 1;
	int32_t x2903 = 65536 * x2899;
	int32_t x2904 = x2903 * x2899;
	int32_t x2900 = x2899 * x2899;
	int32_t x2901 = 1024 * x2900;
	int32_t x2902 = 64 * x2901;
	bool x2921 = x2782 == 1;
	bool x2922 = x2782 == x2899;
	bool x2923 = x2921 || x2922;
	bool x2924;
	if (x2923) {
		x2924 = x2923;
	} else {
		x2924 = false;
	}
	bool x2939 = x2899 >= 1;
	bool x2940;
	if (x2939) {
		x2940 = x2939;
	} else {
		x2940 = false;
	}
	int32_t x2945 = x2898 / 1;
	int32_t x2946 = x2945 + 1;
	int32_t x2950 = 16384 * x2946;
	int32_t x2951 = x2950 * x2946;
	int32_t x2947 = x2946 * x2946;
	int32_t x2948 = 256 * x2947;
	int32_t x2949 = 64 * x2948;
	int32_t x2973 = x2946 + 2;
	bool x2974 = x2973 >= 3;
	bool x2975;
	if (x2974) {
		x2975 = x2974;
	} else {
		x2975 = false;
	}
	int32_t x2980 = x2973 - 3;
	int32_t x2981 = x2980 / 1;
	int32_t x2982 = x2981 + 1;
	int32_t x2986 = 16384 * x2982;
	int32_t x2987 = x2986 * x2982;
	int32_t x2983 = x2982 * x2982;
	int32_t x2984 = 256 * x2983;
	int32_t x2985 = 64 * x2984;
	bool x3009 = x2982 >= 1;
	bool x3010;
	if (x3009) {
		x3010 = x3009;
	} else {
		x3010 = false;
	}
	int32_t x3015 = x2981 / 1;
	int32_t x3016 = x3015 + 1;
	int32_t x3020 = 65536 * x3016;
	int32_t x3021 = x3020 * x3016;
	int32_t x3017 = x3016 * x3016;
	int32_t x3018 = 1024 * x3017;
	int32_t x3019 = 64 * x3018;
	bool x3038 = x2899 == 1;
	bool x3039 = x2899 == x3016;
	bool x3040 = x3038 || x3039;
	bool x3041;
	if (x3040) {
		x3041 = x3040;
	} else {
		x3041 = false;
	}
	bool x3056 = x3016 >= 1;
	bool x3057;
	if (x3056) {
		x3057 = x3056;
	} else {
		x3057 = false;
	}
	int32_t x3062 = x3015 / 1;
	int32_t x3063 = x3062 + 1;
	int32_t x3067 = 32768 * x3063;
	int32_t x3068 = x3067 * x3063;
	int32_t x3064 = x3063 * x3063;
	int32_t x3065 = 512 * x3064;
	int32_t x3066 = 64 * x3065;
	int32_t x3090 = x3063 + 2;
	bool x3091 = x3090 >= 3;
	bool x3092;
	if (x3091) {
		x3092 = x3091;
	} else {
		x3092 = false;
	}
	int32_t x3097 = x3090 - 3;
	int32_t x3098 = x3097 / 2;
	int32_t x3099 = x3098 + 1;
	int32_t x3103 = 32768 * x3099;
	int32_t x3104 = x3103 * x3099;
	int32_t x3100 = x3099 * x3099;
	int32_t x3101 = 512 * x3100;
	int32_t x3102 = 64 * x3101;
	bool x3126 = x3099 >= 1;
	bool x3127;
	if (x3126) {
		x3127 = x3126;
	} else {
		x3127 = false;
	}
	int32_t x3132 = x3098 / 1;
	int32_t x3133 = x3132 + 1;
	int32_t x3137 = 131072 * x3133;
	int32_t x3138 = x3137 * x3133;
	int32_t x3134 = x3133 * x3133;
	int32_t x3135 = 2048 * x3134;
	int32_t x3136 = 64 * x3135;
	int32_t x3158 = x3015 / 2;
	int32_t x3159 = x3158 + 1;
	int32_t x3163 = 131072 * x3159;
	int32_t x3164 = x3163 * x3159;
	int32_t x3160 = x3159 * x3159;
	int32_t x3161 = 2048 * x3160;
	int32_t x3162 = 64 * x3161;
	bool x3181 = x3159 == 1;
	bool x3182 = x3159 == x3133;
	bool x3183 = x3181 || x3182;
	bool x3184;
	if (x3183) {
		x3184 = x3183;
	} else {
		x3184 = false;
	}
	bool x3199 = x3133 >= 1;
	bool x3200;
	if (x3199) {
		x3200 = x3199;
	} else {
		x3200 = false;
	}
	int32_t x3205 = x3132 / 1;
	int32_t x3206 = x3205 + 1;
	int32_t x3210 = 32768 * x3206;
	int32_t x3211 = x3210 * x3206;
	int32_t x3207 = x3206 * x3206;
	int32_t x3208 = 512 * x3207;
	int32_t x3209 = 64 * x3208;
	int32_t x3233 = x3206 + 2;
	bool x3234 = x3233 >= 3;
	bool x3235;
	if (x3234) {
		x3235 = x3234;
	} else {
		x3235 = false;
	}
	int32_t x3240 = x3233 - 3;
	int32_t x3241 = x3240 / 1;
	int32_t x3242 = x3241 + 1;
	int32_t x3246 = 32768 * x3242;
	int32_t x3247 = x3246 * x3242;
	int32_t x3243 = x3242 * x3242;
	int32_t x3244 = 512 * x3243;
	int32_t x3245 = 64 * x3244;
	bool x3269 = x3242 >= 1;
	bool x3270;
	if (x3269) {
		x3270 = x3269;
	} else {
		x3270 = false;
	}
	int32_t x3275 = x3241 / 1;
	int32_t x3276 = x3275 + 1;
	int32_t x3280 = 131072 * x3276;
	int32_t x3281 = x3280 * x3276;
	int32_t x3277 = x3276 * x3276;
	int32_t x3278 = 2048 * x3277;
	int32_t x3279 = 64 * x3278;
	bool x3298 = x3133 == 1;
	bool x3299 = x3133 == x3276;
	bool x3300 = x3298 || x3299;
	bool x3301;
	if (x3300) {
		x3301 = x3300;
	} else {
		x3301 = false;
	}
	bool x3316 = x3276 >= 1;
	bool x3317;
	if (x3316) {
		x3317 = x3316;
	} else {
		x3317 = false;
	}
	int32_t x3322 = x3275 / 1;
	int32_t x3323 = x3322 + 1;
	int32_t x3327 = 32768 * x3323;
	int32_t x3328 = x3327 * x3323;
	int32_t x3324 = x3323 * x3323;
	int32_t x3325 = 512 * x3324;
	int32_t x3326 = 64 * x3325;
	int32_t x3350 = x3323 + 2;
	bool x3351 = x3350 >= 3;
	bool x3352;
	if (x3351) {
		x3352 = x3351;
	} else {
		x3352 = false;
	}
	int32_t x3357 = x3350 - 3;
	int32_t x3358 = x3357 / 1;
	int32_t x3359 = x3358 + 1;
	int32_t x3363 = 32768 * x3359;
	int32_t x3364 = x3363 * x3359;
	int32_t x3360 = x3359 * x3359;
	int32_t x3361 = 512 * x3360;
	int32_t x3362 = 64 * x3361;
	bool x3386 = x3359 >= 1;
	bool x3387;
	if (x3386) {
		x3387 = x3386;
	} else {
		x3387 = false;
	}
	int32_t x3392 = x3358 / 1;
	int32_t x3393 = x3392 + 1;
	int32_t x3397 = 131072 * x3393;
	int32_t x3398 = x3397 * x3393;
	int32_t x3394 = x3393 * x3393;
	int32_t x3395 = 2048 * x3394;
	int32_t x3396 = 64 * x3395;
	bool x3415 = x3276 == 1;
	bool x3416 = x3276 == x3393;
	bool x3417 = x3415 || x3416;
	bool x3418;
	if (x3417) {
		x3418 = x3417;
	} else {
		x3418 = false;
	}
	bool x3433 = x3393 >= 2;
	bool x3434;
	if (x3433) {
		x3434 = x3433;
	} else {
		x3434 = false;
	}
	int32_t x3443 = x3393 - 2;
	int32_t x3444 = x3443 / 1;
	int32_t x3445 = x3444 + 1;
	int32_t x3449 = 131072 * x3445;
	int32_t x3450 = x3449 * x3445;
	bool x3492 = true || false;
	bool x3494;
	if (x3492) {
		bool x3493 = true || true;
		x3494 = x3493;
	} else {
		x3494 = false;
	}
	bool x3495;
	if (x3494) {
		bool x3493 = true || true;
		x3495 = x3493;
	} else {
		x3495 = false;
	}
	bool x3496;
	if (x3495) {
		bool x3493 = true || true;
		x3496 = x3493;
	} else {
		x3496 = false;
	}
	float x3491 = 1.0f / 64.0f;
	bool x3532 = x3393 == x3276;
	bool x3533;
	if (x3532) {
		x3533 = x3532;
	} else {
		x3533 = false;
	}
	bool x3534 = x3393 == 1;
	bool x3535 = x3534 || x3532;
	bool x3536;
	if (x3535) {
		x3536 = x3535;
	} else {
		x3536 = false;
	}
	bool x3603 = x3276 == x3133;
	bool x3604;
	if (x3603) {
		x3604 = x3603;
	} else {
		x3604 = false;
	}
	bool x3605 = x3415 || x3603;
	bool x3606;
	if (x3605) {
		x3606 = x3605;
	} else {
		x3606 = false;
	}
	bool x3673 = x3133 == x3159;
	bool x3674;
	if (x3673) {
		x3674 = x3673;
	} else {
		x3674 = false;
	}
	bool x3675 = x3298 || x3673;
	bool x3676;
	if (x3675) {
		x3676 = x3675;
	} else {
		x3676 = false;
	}
	bool x3755 = x3016 == x2899;
	bool x3756;
	if (x3755) {
		x3756 = x3755;
	} else {
		x3756 = false;
	}
	bool x3757 = x3016 == 1;
	bool x3758 = x3757 || x3755;
	bool x3759;
	if (x3758) {
		x3759 = x3758;
	} else {
		x3759 = false;
	}
	bool x3826 = x2899 == x2782;
	bool x3827;
	if (x3826) {
		x3827 = x3826;
	} else {
		x3827 = false;
	}
	bool x3828 = x3038 || x3826;
	bool x3829;
	if (x3828) {
		x3829 = x3828;
	} else {
		x3829 = false;
	}
	bool x3896 = x2782 == x2665;
	bool x3897;
	if (x3896) {
		x3897 = x3896;
	} else {
		x3897 = false;
	}
	bool x3898 = x2921 || x3896;
	bool x3899;
	if (x3898) {
		x3899 = x3898;
	} else {
		x3899 = false;
	}
	bool x3966 = x2665 == x2548;
	bool x3967;
	if (x3966) {
		x3967 = x3966;
	} else {
		x3967 = false;
	}
	bool x3968 = x2804 || x3966;
	bool x3969;
	if (x3968) {
		x3969 = x3968;
	} else {
		x3969 = false;
	}
	bool x4036 = x2548 == x2405;
	bool x4037;
	if (x4036) {
		x4037 = x4036;
	} else {
		x4037 = false;
	}
	bool x4038 = x2687 || x4036;
	bool x4039;
	if (x4038) {
		x4039 = x4038;
	} else {
		x4039 = false;
	}
	bool x4106 = x2405 == x2431;
	bool x4107;
	if (x4106) {
		x4107 = x4106;
	} else {
		x4107 = false;
	}
	bool x4108 = x2570 || x4106;
	bool x4109;
	if (x4108) {
		x4109 = x4108;
	} else {
		x4109 = false;
	}
	bool x4188 = x2288 == x2171;
	bool x4189;
	if (x4188) {
		x4189 = x4188;
	} else {
		x4189 = false;
	}
	bool x4190 = x2288 == 1;
	bool x4191 = x4190 || x4188;
	bool x4192;
	if (x4191) {
		x4192 = x4191;
	} else {
		x4192 = false;
	}
	bool x4259 = x2171 == x2054;
	bool x4260;
	if (x4259) {
		x4260 = x4259;
	} else {
		x4260 = false;
	}
	bool x4261 = x2310 || x4259;
	bool x4262;
	if (x4261) {
		x4262 = x4261;
	} else {
		x4262 = false;
	}
	bool x4329 = x2054 == x1911;
	bool x4330;
	if (x4329) {
		x4330 = x4329;
	} else {
		x4330 = false;
	}
	bool x4331 = x2193 || x4329;
	bool x4332;
	if (x4331) {
		x4332 = x4331;
	} else {
		x4332 = false;
	}
	bool x4399 = x1911 == x1937;
	bool x4400;
	if (x4399) {
		x4400 = x4399;
	} else {
		x4400 = false;
	}
	bool x4401 = x2076 || x4399;
	bool x4402;
	if (x4401) {
		x4402 = x4401;
	} else {
		x4402 = false;
	}
	bool x4481 = x1794 == x1677;
	bool x4482;
	if (x4481) {
		x4482 = x4481;
	} else {
		x4482 = false;
	}
	bool x4483 = x1794 == 1;
	bool x4484 = x4483 || x4481;
	bool x4485;
	if (x4484) {
		x4485 = x4484;
	} else {
		x4485 = false;
	}
	bool x4552 = x1677 == x1537;
	bool x4553;
	if (x4552) {
		x4553 = x4552;
	} else {
		x4553 = false;
	}
	bool x4554 = x1816 || x4552;
	bool x4555;
	if (x4554) {
		x4555 = x4554;
	} else {
		x4555 = false;
	}
	bool x4622 = x1537 == x1467;
	bool x4623;
	if (x4622) {
		x4623 = x4622;
	} else {
		x4623 = false;
	}
	bool x4624 = x1699 || x4622;
	bool x4625;
	if (x4624) {
		x4625 = x4624;
	} else {
		x4625 = false;
	}
	int32_t x6323 = x1394 / 10;
	double x6328 = (double)x11;
	int64_t x6354 = (int64_t)x11;
	float x6358 = (float)x11;
	for(int x1386=0; x1386 < 4; x1386++) {
		struct timeval begin_1, end_1, diff_1;
		float x1388 = 0.0f;
		float x1389 = x1388;
		float x1390 = x1389;
		int32_t x1391 = x1386 + 1;
		printf("Start training epoch %d\n",x1391);
		gettimeofday(&begin_1, NULL);
		for(int x1396=0; x1396 < x1394; x1396++) {
			int32_t x1397 = x1396 * 64;
			int32_t x1398 = x1397 * 3072;
			float* x1399 = x13+x1398;
			int* x1400 = x14+x1397;
			// Tensor 'toGPU' invocation.
			float* x1402 = (float*)myGpuMalloc(196608 * sizeof(float));
			CUDA_CALL(cudaMemcpy(x1402, x1399, 196608 * sizeof(float), cudaMemcpyHostToDevice));
			float* x1404 = (float*)myGpuMalloc(2 * sizeof(float));
			int* x1405 = (int32_t*)myGpuMalloc(64 * sizeof(int32_t));
			CUDA_CALL(cudaMemcpy(x1405, x1400, 64 * sizeof(int32_t), cudaMemcpyHostToDevice));
			float* x1407 = (float*)myGpuMalloc(1 * sizeof(float));
			float* x1408 = (float*)myGpuMalloc(1 * sizeof(float));
			// allocate memory to save the final loss in CPU Tensor
			float* x1410 = (float*)myMalloc(1 * sizeof(float));;
			if (x1412) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1424 = (float*)myGpuMalloc(x1423 * sizeof(float));
			float* x1425 = (float*)myMalloc(1 * sizeof(float));;
			x1425[0] = 0.0f;
			float* x1427 = (float*)myMalloc(1 * sizeof(float));;
			x1427[0] = 1.0f;

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
							64, 64, x1418, x1418));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x1427, in_desc, x1402, filt_desc, x751,
							conv_desc, algo, ws_data, ws_size,
							x1425, out_desc, x1424));
			};
			float* x1430 = (float*)myGpuMalloc(x1423 * sizeof(float));
			float* x1431 = (float*)myGpuMalloc(x1421 * sizeof(float));
			float* x1432 = (float*)myGpuMalloc(64 * sizeof(float));
			float* x1433 = (float*)myGpuMalloc(64 * sizeof(float));
			float* x1434 = (float*)myMalloc(1 * sizeof(float));;
			x1434[0] = 0.0f;
			float* x1436 = (float*)myMalloc(1 * sizeof(float));;
			x1436[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1418, x1418));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1418, x1418));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x1436, x1434, in_desc, x1424, out_desc, x1431, sbmv_desc, x913,
							x1048, 0.1, x415, x625, 1.0E-5,
							x1432, x1433));
			};
			float* x1439 = (float*)myGpuMalloc(x1423 * sizeof(float));
			float* x1440 = (float*)myMalloc(1 * sizeof(float));;
			x1440[0] = 0.0f;
			float* x1442 = (float*)myMalloc(1 * sizeof(float));;
			x1442[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1418, x1418));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1442, x_desc, x1431, x1440, x_desc, x1431));
			};
			float* x1445 = (float*)myMalloc(1 * sizeof(float));;
			x1445[0] = 0.0f;
			float* x1447 = (float*)myMalloc(1 * sizeof(float));;
			x1447[0] = 1.0f;
			float* x1457 = (float*)myGpuMalloc(x1456 * sizeof(float));

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1418, x1418) );

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1451, x1451));

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
							x1447, in_desc, x1431, x1445, out_desc, x1457));
			};
			float* x1459 = (float*)myGpuMalloc(x1456 * sizeof(float));
			if (x1461) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1473 = (float*)myGpuMalloc(x1472 * sizeof(float));
			float* x1474 = (float*)myMalloc(1 * sizeof(float));;
			x1474[0] = 0.0f;
			float* x1476 = (float*)myMalloc(1 * sizeof(float));;
			x1476[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1451, x1451));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 64, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1467, x1467));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x1476, in_desc, x1457, filt_desc, x994,
							conv_desc, algo, ws_data, ws_size,
							x1474, out_desc, x1473));
			};
			float* x1479 = (float*)myGpuMalloc(x1472 * sizeof(float));
			float* x1480 = (float*)myGpuMalloc(x1470 * sizeof(float));
			float* x1481 = (float*)myGpuMalloc(64 * sizeof(float));
			float* x1482 = (float*)myGpuMalloc(64 * sizeof(float));
			float* x1483 = (float*)myMalloc(1 * sizeof(float));;
			x1483[0] = 0.0f;
			float* x1485 = (float*)myMalloc(1 * sizeof(float));;
			x1485[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1467, x1467));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1467, x1467));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x1485, x1483, in_desc, x1473, out_desc, x1480, sbmv_desc, x373,
							x454, 0.1, x637, x448, 1.0E-5,
							x1481, x1482));
			};
			float* x1488 = (float*)myGpuMalloc(x1472 * sizeof(float));
			float* x1489 = (float*)myMalloc(1 * sizeof(float));;
			x1489[0] = 0.0f;
			float* x1491 = (float*)myMalloc(1 * sizeof(float));;
			x1491[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1467, x1467));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1491, x_desc, x1480, x1489, x_desc, x1480));
			};
			if (x1496) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1509 = (float*)myGpuMalloc(x1508 * sizeof(float));
			float* x1510 = (float*)myMalloc(1 * sizeof(float));;
			x1510[0] = 0.0f;
			float* x1512 = (float*)myMalloc(1 * sizeof(float));;
			x1512[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1467, x1467));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 64, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1503, x1503));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x1512, in_desc, x1480, filt_desc, x565,
							conv_desc, algo, ws_data, ws_size,
							x1510, out_desc, x1509));
			};
			float* x1515 = (float*)myGpuMalloc(x1508 * sizeof(float));
			float* x1516 = (float*)myGpuMalloc(x1506 * sizeof(float));
			float* x1517 = (float*)myGpuMalloc(64 * sizeof(float));
			float* x1518 = (float*)myGpuMalloc(64 * sizeof(float));
			float* x1519 = (float*)myMalloc(1 * sizeof(float));;
			x1519[0] = 0.0f;
			float* x1521 = (float*)myMalloc(1 * sizeof(float));;
			x1521[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1503, x1503));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1503, x1503));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x1521, x1519, in_desc, x1509, out_desc, x1516, sbmv_desc, x787,
							x442, 0.1, x610, x769, 1.0E-5,
							x1517, x1518));
			};
			float* x1524 = (float*)myGpuMalloc(x1508 * sizeof(float));
			float* x1525 = (float*)myMalloc(1 * sizeof(float));;
			x1525[0] = 0.0f;
			float* x1527 = (float*)myMalloc(1 * sizeof(float));;
			x1527[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1503, x1503));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1527, x_desc, x1516, x1525, x_desc, x1516));
			};
			if (x1531) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1543 = (float*)myGpuMalloc(x1542 * sizeof(float));
			float* x1544 = (float*)myMalloc(1 * sizeof(float));;
			x1544[0] = 0.0f;
			float* x1546 = (float*)myMalloc(1 * sizeof(float));;
			x1546[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1503, x1503));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 64, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1537, x1537));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x1546, in_desc, x1516, filt_desc, x391,
							conv_desc, algo, ws_data, ws_size,
							x1544, out_desc, x1543));
			};
			float* x1549 = (float*)myGpuMalloc(x1542 * sizeof(float));
			float* x1550 = (float*)myGpuMalloc(x1540 * sizeof(float));
			float* x1551 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x1552 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x1553 = (float*)myMalloc(1 * sizeof(float));;
			x1553[0] = 0.0f;
			float* x1555 = (float*)myMalloc(1 * sizeof(float));;
			x1555[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1537, x1537));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1537, x1537));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x1555, x1553, in_desc, x1543, out_desc, x1550, sbmv_desc, x892,
							x673, 0.1, x508, x403, 1.0E-5,
							x1551, x1552));
			};
			float* x1558 = (float*)myGpuMalloc(x1542 * sizeof(float));
			if (x1461) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1566 = (float*)myGpuMalloc(x1565 * sizeof(float));
			float* x1567 = (float*)myMalloc(1 * sizeof(float));;
			x1567[0] = 0.0f;
			float* x1569 = (float*)myMalloc(1 * sizeof(float));;
			x1569[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1451, x1451));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 64, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1467, x1467));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x1569, in_desc, x1457, filt_desc, x781,
							conv_desc, algo, ws_data, ws_size,
							x1567, out_desc, x1566));
			};
			float* x1572 = (float*)myGpuMalloc(x1565 * sizeof(float));
			float* x1573 = (float*)myGpuMalloc(x1563 * sizeof(float));
			float* x1574 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x1575 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x1576 = (float*)myMalloc(1 * sizeof(float));;
			x1576[0] = 0.0f;
			float* x1578 = (float*)myMalloc(1 * sizeof(float));;
			x1578[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1467, x1467));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1467, x1467));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x1578, x1576, in_desc, x1566, out_desc, x1573, sbmv_desc, x523,
							x904, 0.1, x1087, x1024, 1.0E-5,
							x1574, x1575));
			};
			float* x1581 = (float*)myGpuMalloc(x1565 * sizeof(float));
			if (x1585) {
			} else {
				assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1467) x Sym(1467), res:  x Const(64) x Const(256) x Sym(1537) x Sym(1537)");
			}
			float* x1590 = (float*)myMalloc(1 * sizeof(float));;
			x1590[0] = 1.0f;
			float* x1592 = (float*)myMalloc(1 * sizeof(float));;
			x1592[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1467, x1467));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1537, x1537));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x1590, bias_desc, x1573, x1592, out_desc, x1550));
			};
			float* x1595 = (float*)myMalloc(1 * sizeof(float));;
			x1595[0] = 0.0f;
			float* x1597 = (float*)myMalloc(1 * sizeof(float));;
			x1597[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1537, x1537));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1597, x_desc, x1550, x1595, x_desc, x1550));
			};
			if (x1601) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1613 = (float*)myGpuMalloc(x1612 * sizeof(float));
			float* x1614 = (float*)myMalloc(1 * sizeof(float));;
			x1614[0] = 0.0f;
			float* x1616 = (float*)myMalloc(1 * sizeof(float));;
			x1616[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1537, x1537));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 256, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1607, x1607));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x1616, in_desc, x1550, filt_desc, x808,
							conv_desc, algo, ws_data, ws_size,
							x1614, out_desc, x1613));
			};
			float* x1619 = (float*)myGpuMalloc(x1612 * sizeof(float));
			float* x1620 = (float*)myGpuMalloc(x1610 * sizeof(float));
			float* x1621 = (float*)myGpuMalloc(64 * sizeof(float));
			float* x1622 = (float*)myGpuMalloc(64 * sizeof(float));
			float* x1623 = (float*)myMalloc(1 * sizeof(float));;
			x1623[0] = 0.0f;
			float* x1625 = (float*)myMalloc(1 * sizeof(float));;
			x1625[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1607, x1607));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1607, x1607));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x1625, x1623, in_desc, x1613, out_desc, x1620, sbmv_desc, x721,
							x475, 0.1, x325, x601, 1.0E-5,
							x1621, x1622));
			};
			float* x1628 = (float*)myGpuMalloc(x1612 * sizeof(float));
			float* x1629 = (float*)myMalloc(1 * sizeof(float));;
			x1629[0] = 0.0f;
			float* x1631 = (float*)myMalloc(1 * sizeof(float));;
			x1631[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1607, x1607));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1631, x_desc, x1620, x1629, x_desc, x1620));
			};
			if (x1636) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1649 = (float*)myGpuMalloc(x1648 * sizeof(float));
			float* x1650 = (float*)myMalloc(1 * sizeof(float));;
			x1650[0] = 0.0f;
			float* x1652 = (float*)myMalloc(1 * sizeof(float));;
			x1652[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1607, x1607));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 64, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1643, x1643));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x1652, in_desc, x1620, filt_desc, x544,
							conv_desc, algo, ws_data, ws_size,
							x1650, out_desc, x1649));
			};
			float* x1655 = (float*)myGpuMalloc(x1648 * sizeof(float));
			float* x1656 = (float*)myGpuMalloc(x1646 * sizeof(float));
			float* x1657 = (float*)myGpuMalloc(64 * sizeof(float));
			float* x1658 = (float*)myGpuMalloc(64 * sizeof(float));
			float* x1659 = (float*)myMalloc(1 * sizeof(float));;
			x1659[0] = 0.0f;
			float* x1661 = (float*)myMalloc(1 * sizeof(float));;
			x1661[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1643, x1643));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1643, x1643));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x1661, x1659, in_desc, x1649, out_desc, x1656, sbmv_desc, x919,
							x754, 0.1, x427, x1027, 1.0E-5,
							x1657, x1658));
			};
			float* x1664 = (float*)myGpuMalloc(x1648 * sizeof(float));
			float* x1665 = (float*)myMalloc(1 * sizeof(float));;
			x1665[0] = 0.0f;
			float* x1667 = (float*)myMalloc(1 * sizeof(float));;
			x1667[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1643, x1643));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1667, x_desc, x1656, x1665, x_desc, x1656));
			};
			if (x1671) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1683 = (float*)myGpuMalloc(x1682 * sizeof(float));
			float* x1684 = (float*)myMalloc(1 * sizeof(float));;
			x1684[0] = 0.0f;
			float* x1686 = (float*)myMalloc(1 * sizeof(float));;
			x1686[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1643, x1643));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 64, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1677, x1677));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x1686, in_desc, x1656, filt_desc, x685,
							conv_desc, algo, ws_data, ws_size,
							x1684, out_desc, x1683));
			};
			float* x1689 = (float*)myGpuMalloc(x1682 * sizeof(float));
			float* x1690 = (float*)myGpuMalloc(x1680 * sizeof(float));
			float* x1691 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x1692 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x1693 = (float*)myMalloc(1 * sizeof(float));;
			x1693[0] = 0.0f;
			float* x1695 = (float*)myMalloc(1 * sizeof(float));;
			x1695[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1677, x1677));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1677, x1677));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x1695, x1693, in_desc, x1683, out_desc, x1690, sbmv_desc, x469,
							x316, 0.1, x568, x793, 1.0E-5,
							x1691, x1692));
			};
			float* x1698 = (float*)myGpuMalloc(x1682 * sizeof(float));
			if (x1702) {
			} else {
				assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1537) x Sym(1537), res:  x Const(64) x Const(256) x Sym(1677) x Sym(1677)");
			}
			float* x1707 = (float*)myMalloc(1 * sizeof(float));;
			x1707[0] = 1.0f;
			float* x1709 = (float*)myMalloc(1 * sizeof(float));;
			x1709[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1537, x1537));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1677, x1677));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x1707, bias_desc, x1550, x1709, out_desc, x1690));
			};
			float* x1712 = (float*)myMalloc(1 * sizeof(float));;
			x1712[0] = 0.0f;
			float* x1714 = (float*)myMalloc(1 * sizeof(float));;
			x1714[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1677, x1677));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1714, x_desc, x1690, x1712, x_desc, x1690));
			};
			if (x1718) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1730 = (float*)myGpuMalloc(x1729 * sizeof(float));
			float* x1731 = (float*)myMalloc(1 * sizeof(float));;
			x1731[0] = 0.0f;
			float* x1733 = (float*)myMalloc(1 * sizeof(float));;
			x1733[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1677, x1677));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 256, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1724, x1724));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x1733, in_desc, x1690, filt_desc, x745,
							conv_desc, algo, ws_data, ws_size,
							x1731, out_desc, x1730));
			};
			float* x1736 = (float*)myGpuMalloc(x1729 * sizeof(float));
			float* x1737 = (float*)myGpuMalloc(x1727 * sizeof(float));
			float* x1738 = (float*)myGpuMalloc(64 * sizeof(float));
			float* x1739 = (float*)myGpuMalloc(64 * sizeof(float));
			float* x1740 = (float*)myMalloc(1 * sizeof(float));;
			x1740[0] = 0.0f;
			float* x1742 = (float*)myMalloc(1 * sizeof(float));;
			x1742[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1724, x1724));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1724, x1724));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x1742, x1740, in_desc, x1730, out_desc, x1737, sbmv_desc, x538,
							x367, 0.1, x1066, x856, 1.0E-5,
							x1738, x1739));
			};
			float* x1745 = (float*)myGpuMalloc(x1729 * sizeof(float));
			float* x1746 = (float*)myMalloc(1 * sizeof(float));;
			x1746[0] = 0.0f;
			float* x1748 = (float*)myMalloc(1 * sizeof(float));;
			x1748[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1724, x1724));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1748, x_desc, x1737, x1746, x_desc, x1737));
			};
			if (x1753) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1766 = (float*)myGpuMalloc(x1765 * sizeof(float));
			float* x1767 = (float*)myMalloc(1 * sizeof(float));;
			x1767[0] = 0.0f;
			float* x1769 = (float*)myMalloc(1 * sizeof(float));;
			x1769[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1724, x1724));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 64, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1760, x1760));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x1769, in_desc, x1737, filt_desc, x514,
							conv_desc, algo, ws_data, ws_size,
							x1767, out_desc, x1766));
			};
			float* x1772 = (float*)myGpuMalloc(x1765 * sizeof(float));
			float* x1773 = (float*)myGpuMalloc(x1763 * sizeof(float));
			float* x1774 = (float*)myGpuMalloc(64 * sizeof(float));
			float* x1775 = (float*)myGpuMalloc(64 * sizeof(float));
			float* x1776 = (float*)myMalloc(1 * sizeof(float));;
			x1776[0] = 0.0f;
			float* x1778 = (float*)myMalloc(1 * sizeof(float));;
			x1778[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1760, x1760));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1760, x1760));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x1778, x1776, in_desc, x1766, out_desc, x1773, sbmv_desc, x511,
							x700, 0.1, x832, x649, 1.0E-5,
							x1774, x1775));
			};
			float* x1781 = (float*)myGpuMalloc(x1765 * sizeof(float));
			float* x1782 = (float*)myMalloc(1 * sizeof(float));;
			x1782[0] = 0.0f;
			float* x1784 = (float*)myMalloc(1 * sizeof(float));;
			x1784[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1760, x1760));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1784, x_desc, x1773, x1782, x_desc, x1773));
			};
			if (x1788) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1800 = (float*)myGpuMalloc(x1799 * sizeof(float));
			float* x1801 = (float*)myMalloc(1 * sizeof(float));;
			x1801[0] = 0.0f;
			float* x1803 = (float*)myMalloc(1 * sizeof(float));;
			x1803[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1760, x1760));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 64, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1794, x1794));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x1803, in_desc, x1773, filt_desc, x556,
							conv_desc, algo, ws_data, ws_size,
							x1801, out_desc, x1800));
			};
			float* x1806 = (float*)myGpuMalloc(x1799 * sizeof(float));
			float* x1807 = (float*)myGpuMalloc(x1797 * sizeof(float));
			float* x1808 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x1809 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x1810 = (float*)myMalloc(1 * sizeof(float));;
			x1810[0] = 0.0f;
			float* x1812 = (float*)myMalloc(1 * sizeof(float));;
			x1812[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1794, x1794));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1794, x1794));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x1812, x1810, in_desc, x1800, out_desc, x1807, sbmv_desc, x406,
							x1036, 0.1, x847, x694, 1.0E-5,
							x1808, x1809));
			};
			float* x1815 = (float*)myGpuMalloc(x1799 * sizeof(float));
			if (x1819) {
			} else {
				assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1677) x Sym(1677), res:  x Const(64) x Const(256) x Sym(1794) x Sym(1794)");
			}
			float* x1824 = (float*)myMalloc(1 * sizeof(float));;
			x1824[0] = 1.0f;
			float* x1826 = (float*)myMalloc(1 * sizeof(float));;
			x1826[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1677, x1677));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1794, x1794));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x1824, bias_desc, x1690, x1826, out_desc, x1807));
			};
			float* x1829 = (float*)myMalloc(1 * sizeof(float));;
			x1829[0] = 0.0f;
			float* x1831 = (float*)myMalloc(1 * sizeof(float));;
			x1831[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1794, x1794));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1831, x_desc, x1807, x1829, x_desc, x1807));
			};
			if (x1835) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1847 = (float*)myGpuMalloc(x1846 * sizeof(float));
			float* x1848 = (float*)myMalloc(1 * sizeof(float));;
			x1848[0] = 0.0f;
			float* x1850 = (float*)myMalloc(1 * sizeof(float));;
			x1850[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1794, x1794));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							128, 256, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1841, x1841));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x1850, in_desc, x1807, filt_desc, x328,
							conv_desc, algo, ws_data, ws_size,
							x1848, out_desc, x1847));
			};
			float* x1853 = (float*)myGpuMalloc(x1846 * sizeof(float));
			float* x1854 = (float*)myGpuMalloc(x1844 * sizeof(float));
			float* x1855 = (float*)myGpuMalloc(128 * sizeof(float));
			float* x1856 = (float*)myGpuMalloc(128 * sizeof(float));
			float* x1857 = (float*)myMalloc(1 * sizeof(float));;
			x1857[0] = 0.0f;
			float* x1859 = (float*)myMalloc(1 * sizeof(float));;
			x1859[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1841, x1841));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1841, x1841));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x1859, x1857, in_desc, x1847, out_desc, x1854, sbmv_desc, x547,
							x811, 0.1, x907, x697, 1.0E-5,
							x1855, x1856));
			};
			float* x1862 = (float*)myGpuMalloc(x1846 * sizeof(float));
			float* x1863 = (float*)myMalloc(1 * sizeof(float));;
			x1863[0] = 0.0f;
			float* x1865 = (float*)myMalloc(1 * sizeof(float));;
			x1865[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1841, x1841));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1865, x_desc, x1854, x1863, x_desc, x1854));
			};
			if (x1870) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1883 = (float*)myGpuMalloc(x1882 * sizeof(float));
			float* x1884 = (float*)myMalloc(1 * sizeof(float));;
			x1884[0] = 0.0f;
			float* x1886 = (float*)myMalloc(1 * sizeof(float));;
			x1886[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1841, x1841));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							128, 128, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1877, x1877));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 2, 2, 1, 1,
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
							x1886, in_desc, x1854, filt_desc, x376,
							conv_desc, algo, ws_data, ws_size,
							x1884, out_desc, x1883));
			};
			float* x1889 = (float*)myGpuMalloc(x1882 * sizeof(float));
			float* x1890 = (float*)myGpuMalloc(x1880 * sizeof(float));
			float* x1891 = (float*)myGpuMalloc(128 * sizeof(float));
			float* x1892 = (float*)myGpuMalloc(128 * sizeof(float));
			float* x1893 = (float*)myMalloc(1 * sizeof(float));;
			x1893[0] = 0.0f;
			float* x1895 = (float*)myMalloc(1 * sizeof(float));;
			x1895[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1877, x1877));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1877, x1877));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x1895, x1893, in_desc, x1883, out_desc, x1890, sbmv_desc, x1051,
							x865, 0.1, x679, x424, 1.0E-5,
							x1891, x1892));
			};
			float* x1898 = (float*)myGpuMalloc(x1882 * sizeof(float));
			float* x1899 = (float*)myMalloc(1 * sizeof(float));;
			x1899[0] = 0.0f;
			float* x1901 = (float*)myMalloc(1 * sizeof(float));;
			x1901[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1877, x1877));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1901, x_desc, x1890, x1899, x_desc, x1890));
			};
			if (x1905) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1917 = (float*)myGpuMalloc(x1916 * sizeof(float));
			float* x1918 = (float*)myMalloc(1 * sizeof(float));;
			x1918[0] = 0.0f;
			float* x1920 = (float*)myMalloc(1 * sizeof(float));;
			x1920[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1877, x1877));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							512, 128, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1911, x1911));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x1920, in_desc, x1890, filt_desc, x613,
							conv_desc, algo, ws_data, ws_size,
							x1918, out_desc, x1917));
			};
			float* x1923 = (float*)myGpuMalloc(x1916 * sizeof(float));
			float* x1924 = (float*)myGpuMalloc(x1914 * sizeof(float));
			float* x1925 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x1926 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x1927 = (float*)myMalloc(1 * sizeof(float));;
			x1927[0] = 0.0f;
			float* x1929 = (float*)myMalloc(1 * sizeof(float));;
			x1929[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1911, x1911));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1911, x1911));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x1929, x1927, in_desc, x1917, out_desc, x1924, sbmv_desc, x730,
							x925, 0.1, x742, x598, 1.0E-5,
							x1925, x1926));
			};
			float* x1932 = (float*)myGpuMalloc(x1916 * sizeof(float));
			if (x1835) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1943 = (float*)myGpuMalloc(x1942 * sizeof(float));
			float* x1944 = (float*)myMalloc(1 * sizeof(float));;
			x1944[0] = 0.0f;
			float* x1946 = (float*)myMalloc(1 * sizeof(float));;
			x1946[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1794, x1794));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							512, 256, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1937, x1937));

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
							x1946, in_desc, x1807, filt_desc, x1069,
							conv_desc, algo, ws_data, ws_size,
							x1944, out_desc, x1943));
			};
			float* x1949 = (float*)myGpuMalloc(x1942 * sizeof(float));
			float* x1950 = (float*)myGpuMalloc(x1940 * sizeof(float));
			float* x1951 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x1952 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x1953 = (float*)myMalloc(1 * sizeof(float));;
			x1953[0] = 0.0f;
			float* x1955 = (float*)myMalloc(1 * sizeof(float));;
			x1955[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1937, x1937));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1937, x1937));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x1955, x1953, in_desc, x1943, out_desc, x1950, sbmv_desc, x916,
							x652, 0.1, x421, x364, 1.0E-5,
							x1951, x1952));
			};
			float* x1958 = (float*)myGpuMalloc(x1942 * sizeof(float));
			if (x1962) {
			} else {
				assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(1937) x Sym(1937), res:  x Const(64) x Const(512) x Sym(1911) x Sym(1911)");
			}
			float* x1967 = (float*)myMalloc(1 * sizeof(float));;
			x1967[0] = 1.0f;
			float* x1969 = (float*)myMalloc(1 * sizeof(float));;
			x1969[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1937, x1937));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1911, x1911));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x1967, bias_desc, x1950, x1969, out_desc, x1924));
			};
			float* x1972 = (float*)myMalloc(1 * sizeof(float));;
			x1972[0] = 0.0f;
			float* x1974 = (float*)myMalloc(1 * sizeof(float));;
			x1974[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1911, x1911));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1974, x_desc, x1924, x1972, x_desc, x1924));
			};
			if (x1978) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1990 = (float*)myGpuMalloc(x1989 * sizeof(float));
			float* x1991 = (float*)myMalloc(1 * sizeof(float));;
			x1991[0] = 0.0f;
			float* x1993 = (float*)myMalloc(1 * sizeof(float));;
			x1993[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1911, x1911));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							128, 512, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1984, x1984));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x1993, in_desc, x1924, filt_desc, x1063,
							conv_desc, algo, ws_data, ws_size,
							x1991, out_desc, x1990));
			};
			float* x1996 = (float*)myGpuMalloc(x1989 * sizeof(float));
			float* x1997 = (float*)myGpuMalloc(x1987 * sizeof(float));
			float* x1998 = (float*)myGpuMalloc(128 * sizeof(float));
			float* x1999 = (float*)myGpuMalloc(128 * sizeof(float));
			float* x2000 = (float*)myMalloc(1 * sizeof(float));;
			x2000[0] = 0.0f;
			float* x2002 = (float*)myMalloc(1 * sizeof(float));;
			x2002[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1984, x1984));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1984, x1984));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2002, x2000, in_desc, x1990, out_desc, x1997, sbmv_desc, x961,
							x346, 0.1, x595, x826, 1.0E-5,
							x1998, x1999));
			};
			float* x2005 = (float*)myGpuMalloc(x1989 * sizeof(float));
			float* x2006 = (float*)myMalloc(1 * sizeof(float));;
			x2006[0] = 0.0f;
			float* x2008 = (float*)myMalloc(1 * sizeof(float));;
			x2008[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1984, x1984));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2008, x_desc, x1997, x2006, x_desc, x1997));
			};
			if (x2013) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2026 = (float*)myGpuMalloc(x2025 * sizeof(float));
			float* x2027 = (float*)myMalloc(1 * sizeof(float));;
			x2027[0] = 0.0f;
			float* x2029 = (float*)myMalloc(1 * sizeof(float));;
			x2029[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1984, x1984));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							128, 128, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2020, x2020));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x2029, in_desc, x1997, filt_desc, x1000,
							conv_desc, algo, ws_data, ws_size,
							x2027, out_desc, x2026));
			};
			float* x2032 = (float*)myGpuMalloc(x2025 * sizeof(float));
			float* x2033 = (float*)myGpuMalloc(x2023 * sizeof(float));
			float* x2034 = (float*)myGpuMalloc(128 * sizeof(float));
			float* x2035 = (float*)myGpuMalloc(128 * sizeof(float));
			float* x2036 = (float*)myMalloc(1 * sizeof(float));;
			x2036[0] = 0.0f;
			float* x2038 = (float*)myMalloc(1 * sizeof(float));;
			x2038[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2020, x2020));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2020, x2020));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2038, x2036, in_desc, x2026, out_desc, x2033, sbmv_desc, x319,
							x580, 0.1, x400, x970, 1.0E-5,
							x2034, x2035));
			};
			float* x2041 = (float*)myGpuMalloc(x2025 * sizeof(float));
			float* x2042 = (float*)myMalloc(1 * sizeof(float));;
			x2042[0] = 0.0f;
			float* x2044 = (float*)myMalloc(1 * sizeof(float));;
			x2044[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2020, x2020));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2044, x_desc, x2033, x2042, x_desc, x2033));
			};
			if (x2048) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2060 = (float*)myGpuMalloc(x2059 * sizeof(float));
			float* x2061 = (float*)myMalloc(1 * sizeof(float));;
			x2061[0] = 0.0f;
			float* x2063 = (float*)myMalloc(1 * sizeof(float));;
			x2063[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2020, x2020));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							512, 128, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2054, x2054));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x2063, in_desc, x2033, filt_desc, x628,
							conv_desc, algo, ws_data, ws_size,
							x2061, out_desc, x2060));
			};
			float* x2066 = (float*)myGpuMalloc(x2059 * sizeof(float));
			float* x2067 = (float*)myGpuMalloc(x2057 * sizeof(float));
			float* x2068 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x2069 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x2070 = (float*)myMalloc(1 * sizeof(float));;
			x2070[0] = 0.0f;
			float* x2072 = (float*)myMalloc(1 * sizeof(float));;
			x2072[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2054, x2054));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2054, x2054));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2072, x2070, in_desc, x2060, out_desc, x2067, sbmv_desc, x451,
							x1033, 0.1, x736, x559, 1.0E-5,
							x2068, x2069));
			};
			float* x2075 = (float*)myGpuMalloc(x2059 * sizeof(float));
			if (x2079) {
			} else {
				assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(1911) x Sym(1911), res:  x Const(64) x Const(512) x Sym(2054) x Sym(2054)");
			}
			float* x2084 = (float*)myMalloc(1 * sizeof(float));;
			x2084[0] = 1.0f;
			float* x2086 = (float*)myMalloc(1 * sizeof(float));;
			x2086[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1911, x1911));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2054, x2054));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x2084, bias_desc, x1924, x2086, out_desc, x2067));
			};
			float* x2089 = (float*)myMalloc(1 * sizeof(float));;
			x2089[0] = 0.0f;
			float* x2091 = (float*)myMalloc(1 * sizeof(float));;
			x2091[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2054, x2054));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2091, x_desc, x2067, x2089, x_desc, x2067));
			};
			if (x2095) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2107 = (float*)myGpuMalloc(x2106 * sizeof(float));
			float* x2108 = (float*)myMalloc(1 * sizeof(float));;
			x2108[0] = 0.0f;
			float* x2110 = (float*)myMalloc(1 * sizeof(float));;
			x2110[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2054, x2054));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							128, 512, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2101, x2101));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x2110, in_desc, x2067, filt_desc, x883,
							conv_desc, algo, ws_data, ws_size,
							x2108, out_desc, x2107));
			};
			float* x2113 = (float*)myGpuMalloc(x2106 * sizeof(float));
			float* x2114 = (float*)myGpuMalloc(x2104 * sizeof(float));
			float* x2115 = (float*)myGpuMalloc(128 * sizeof(float));
			float* x2116 = (float*)myGpuMalloc(128 * sizeof(float));
			float* x2117 = (float*)myMalloc(1 * sizeof(float));;
			x2117[0] = 0.0f;
			float* x2119 = (float*)myMalloc(1 * sizeof(float));;
			x2119[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2101, x2101));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2101, x2101));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2119, x2117, in_desc, x2107, out_desc, x2114, sbmv_desc, x430,
							x805, 0.1, x631, x322, 1.0E-5,
							x2115, x2116));
			};
			float* x2122 = (float*)myGpuMalloc(x2106 * sizeof(float));
			float* x2123 = (float*)myMalloc(1 * sizeof(float));;
			x2123[0] = 0.0f;
			float* x2125 = (float*)myMalloc(1 * sizeof(float));;
			x2125[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2101, x2101));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2125, x_desc, x2114, x2123, x_desc, x2114));
			};
			if (x2130) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2143 = (float*)myGpuMalloc(x2142 * sizeof(float));
			float* x2144 = (float*)myMalloc(1 * sizeof(float));;
			x2144[0] = 0.0f;
			float* x2146 = (float*)myMalloc(1 * sizeof(float));;
			x2146[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2101, x2101));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							128, 128, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2137, x2137));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x2146, in_desc, x2114, filt_desc, x868,
							conv_desc, algo, ws_data, ws_size,
							x2144, out_desc, x2143));
			};
			float* x2149 = (float*)myGpuMalloc(x2142 * sizeof(float));
			float* x2150 = (float*)myGpuMalloc(x2140 * sizeof(float));
			float* x2151 = (float*)myGpuMalloc(128 * sizeof(float));
			float* x2152 = (float*)myGpuMalloc(128 * sizeof(float));
			float* x2153 = (float*)myMalloc(1 * sizeof(float));;
			x2153[0] = 0.0f;
			float* x2155 = (float*)myMalloc(1 * sizeof(float));;
			x2155[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2137, x2137));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2137, x2137));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2155, x2153, in_desc, x2143, out_desc, x2150, sbmv_desc, x676,
							x478, 0.1, x946, x1093, 1.0E-5,
							x2151, x2152));
			};
			float* x2158 = (float*)myGpuMalloc(x2142 * sizeof(float));
			float* x2159 = (float*)myMalloc(1 * sizeof(float));;
			x2159[0] = 0.0f;
			float* x2161 = (float*)myMalloc(1 * sizeof(float));;
			x2161[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2137, x2137));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2161, x_desc, x2150, x2159, x_desc, x2150));
			};
			if (x2165) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2177 = (float*)myGpuMalloc(x2176 * sizeof(float));
			float* x2178 = (float*)myMalloc(1 * sizeof(float));;
			x2178[0] = 0.0f;
			float* x2180 = (float*)myMalloc(1 * sizeof(float));;
			x2180[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2137, x2137));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							512, 128, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2171, x2171));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x2180, in_desc, x2150, filt_desc, x418,
							conv_desc, algo, ws_data, ws_size,
							x2178, out_desc, x2177));
			};
			float* x2183 = (float*)myGpuMalloc(x2176 * sizeof(float));
			float* x2184 = (float*)myGpuMalloc(x2174 * sizeof(float));
			float* x2185 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x2186 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x2187 = (float*)myMalloc(1 * sizeof(float));;
			x2187[0] = 0.0f;
			float* x2189 = (float*)myMalloc(1 * sizeof(float));;
			x2189[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2171, x2171));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2171, x2171));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2189, x2187, in_desc, x2177, out_desc, x2184, sbmv_desc, x796,
							x541, 0.1, x370, x964, 1.0E-5,
							x2185, x2186));
			};
			float* x2192 = (float*)myGpuMalloc(x2176 * sizeof(float));
			if (x2196) {
			} else {
				assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(2054) x Sym(2054), res:  x Const(64) x Const(512) x Sym(2171) x Sym(2171)");
			}
			float* x2201 = (float*)myMalloc(1 * sizeof(float));;
			x2201[0] = 1.0f;
			float* x2203 = (float*)myMalloc(1 * sizeof(float));;
			x2203[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2054, x2054));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2171, x2171));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x2201, bias_desc, x2067, x2203, out_desc, x2184));
			};
			float* x2206 = (float*)myMalloc(1 * sizeof(float));;
			x2206[0] = 0.0f;
			float* x2208 = (float*)myMalloc(1 * sizeof(float));;
			x2208[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2171, x2171));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2208, x_desc, x2184, x2206, x_desc, x2184));
			};
			if (x2212) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2224 = (float*)myGpuMalloc(x2223 * sizeof(float));
			float* x2225 = (float*)myMalloc(1 * sizeof(float));;
			x2225[0] = 0.0f;
			float* x2227 = (float*)myMalloc(1 * sizeof(float));;
			x2227[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2171, x2171));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							128, 512, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2218, x2218));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x2227, in_desc, x2184, filt_desc, x691,
							conv_desc, algo, ws_data, ws_size,
							x2225, out_desc, x2224));
			};
			float* x2230 = (float*)myGpuMalloc(x2223 * sizeof(float));
			float* x2231 = (float*)myGpuMalloc(x2221 * sizeof(float));
			float* x2232 = (float*)myGpuMalloc(128 * sizeof(float));
			float* x2233 = (float*)myGpuMalloc(128 * sizeof(float));
			float* x2234 = (float*)myMalloc(1 * sizeof(float));;
			x2234[0] = 0.0f;
			float* x2236 = (float*)myMalloc(1 * sizeof(float));;
			x2236[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2218, x2218));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2218, x2218));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2236, x2234, in_desc, x2224, out_desc, x2231, sbmv_desc, x412,
							x1021, 0.1, x1003, x1078, 1.0E-5,
							x2232, x2233));
			};
			float* x2239 = (float*)myGpuMalloc(x2223 * sizeof(float));
			float* x2240 = (float*)myMalloc(1 * sizeof(float));;
			x2240[0] = 0.0f;
			float* x2242 = (float*)myMalloc(1 * sizeof(float));;
			x2242[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2218, x2218));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2242, x_desc, x2231, x2240, x_desc, x2231));
			};
			if (x2247) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2260 = (float*)myGpuMalloc(x2259 * sizeof(float));
			float* x2261 = (float*)myMalloc(1 * sizeof(float));;
			x2261[0] = 0.0f;
			float* x2263 = (float*)myMalloc(1 * sizeof(float));;
			x2263[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2218, x2218));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							128, 128, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2254, x2254));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x2263, in_desc, x2231, filt_desc, x790,
							conv_desc, algo, ws_data, ws_size,
							x2261, out_desc, x2260));
			};
			float* x2266 = (float*)myGpuMalloc(x2259 * sizeof(float));
			float* x2267 = (float*)myGpuMalloc(x2257 * sizeof(float));
			float* x2268 = (float*)myGpuMalloc(128 * sizeof(float));
			float* x2269 = (float*)myGpuMalloc(128 * sizeof(float));
			float* x2270 = (float*)myMalloc(1 * sizeof(float));;
			x2270[0] = 0.0f;
			float* x2272 = (float*)myMalloc(1 * sizeof(float));;
			x2272[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2254, x2254));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2254, x2254));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2272, x2270, in_desc, x2260, out_desc, x2267, sbmv_desc, x532,
							x409, 0.1, x1099, x739, 1.0E-5,
							x2268, x2269));
			};
			float* x2275 = (float*)myGpuMalloc(x2259 * sizeof(float));
			float* x2276 = (float*)myMalloc(1 * sizeof(float));;
			x2276[0] = 0.0f;
			float* x2278 = (float*)myMalloc(1 * sizeof(float));;
			x2278[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2254, x2254));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2278, x_desc, x2267, x2276, x_desc, x2267));
			};
			if (x2282) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2294 = (float*)myGpuMalloc(x2293 * sizeof(float));
			float* x2295 = (float*)myMalloc(1 * sizeof(float));;
			x2295[0] = 0.0f;
			float* x2297 = (float*)myMalloc(1 * sizeof(float));;
			x2297[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2254, x2254));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							512, 128, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2288, x2288));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x2297, in_desc, x2267, filt_desc, x460,
							conv_desc, algo, ws_data, ws_size,
							x2295, out_desc, x2294));
			};
			float* x2300 = (float*)myGpuMalloc(x2293 * sizeof(float));
			float* x2301 = (float*)myGpuMalloc(x2291 * sizeof(float));
			float* x2302 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x2303 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x2304 = (float*)myMalloc(1 * sizeof(float));;
			x2304[0] = 0.0f;
			float* x2306 = (float*)myMalloc(1 * sizeof(float));;
			x2306[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2288, x2288));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2288, x2288));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2306, x2304, in_desc, x2294, out_desc, x2301, sbmv_desc, x763,
							x457, 0.1, x352, x997, 1.0E-5,
							x2302, x2303));
			};
			float* x2309 = (float*)myGpuMalloc(x2293 * sizeof(float));
			if (x2313) {
			} else {
				assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(2171) x Sym(2171), res:  x Const(64) x Const(512) x Sym(2288) x Sym(2288)");
			}
			float* x2318 = (float*)myMalloc(1 * sizeof(float));;
			x2318[0] = 1.0f;
			float* x2320 = (float*)myMalloc(1 * sizeof(float));;
			x2320[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2171, x2171));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2288, x2288));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x2318, bias_desc, x2184, x2320, out_desc, x2301));
			};
			float* x2323 = (float*)myMalloc(1 * sizeof(float));;
			x2323[0] = 0.0f;
			float* x2325 = (float*)myMalloc(1 * sizeof(float));;
			x2325[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2288, x2288));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2325, x_desc, x2301, x2323, x_desc, x2301));
			};
			if (x2329) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2341 = (float*)myGpuMalloc(x2340 * sizeof(float));
			float* x2342 = (float*)myMalloc(1 * sizeof(float));;
			x2342[0] = 0.0f;
			float* x2344 = (float*)myMalloc(1 * sizeof(float));;
			x2344[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2288, x2288));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 512, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2335, x2335));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x2344, in_desc, x2301, filt_desc, x835,
							conv_desc, algo, ws_data, ws_size,
							x2342, out_desc, x2341));
			};
			float* x2347 = (float*)myGpuMalloc(x2340 * sizeof(float));
			float* x2348 = (float*)myGpuMalloc(x2338 * sizeof(float));
			float* x2349 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2350 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2351 = (float*)myMalloc(1 * sizeof(float));;
			x2351[0] = 0.0f;
			float* x2353 = (float*)myMalloc(1 * sizeof(float));;
			x2353[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2335, x2335));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2335, x2335));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2353, x2351, in_desc, x2341, out_desc, x2348, sbmv_desc, x1105,
							x358, 0.1, x688, x889, 1.0E-5,
							x2349, x2350));
			};
			float* x2356 = (float*)myGpuMalloc(x2340 * sizeof(float));
			float* x2357 = (float*)myMalloc(1 * sizeof(float));;
			x2357[0] = 0.0f;
			float* x2359 = (float*)myMalloc(1 * sizeof(float));;
			x2359[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2335, x2335));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2359, x_desc, x2348, x2357, x_desc, x2348));
			};
			if (x2364) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2377 = (float*)myGpuMalloc(x2376 * sizeof(float));
			float* x2378 = (float*)myMalloc(1 * sizeof(float));;
			x2378[0] = 0.0f;
			float* x2380 = (float*)myMalloc(1 * sizeof(float));;
			x2380[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2335, x2335));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 256, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2371, x2371));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 2, 2, 1, 1,
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
							x2380, in_desc, x2348, filt_desc, x820,
							conv_desc, algo, ws_data, ws_size,
							x2378, out_desc, x2377));
			};
			float* x2383 = (float*)myGpuMalloc(x2376 * sizeof(float));
			float* x2384 = (float*)myGpuMalloc(x2374 * sizeof(float));
			float* x2385 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2386 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2387 = (float*)myMalloc(1 * sizeof(float));;
			x2387[0] = 0.0f;
			float* x2389 = (float*)myMalloc(1 * sizeof(float));;
			x2389[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2371, x2371));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2371, x2371));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2389, x2387, in_desc, x2377, out_desc, x2384, sbmv_desc, x619,
							x343, 0.1, x982, x592, 1.0E-5,
							x2385, x2386));
			};
			float* x2392 = (float*)myGpuMalloc(x2376 * sizeof(float));
			float* x2393 = (float*)myMalloc(1 * sizeof(float));;
			x2393[0] = 0.0f;
			float* x2395 = (float*)myMalloc(1 * sizeof(float));;
			x2395[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2371, x2371));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2395, x_desc, x2384, x2393, x_desc, x2384));
			};
			if (x2399) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2411 = (float*)myGpuMalloc(x2410 * sizeof(float));
			float* x2412 = (float*)myMalloc(1 * sizeof(float));;
			x2412[0] = 0.0f;
			float* x2414 = (float*)myMalloc(1 * sizeof(float));;
			x2414[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2371, x2371));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							1024, 256, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2405, x2405));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x2414, in_desc, x2384, filt_desc, x1102,
							conv_desc, algo, ws_data, ws_size,
							x2412, out_desc, x2411));
			};
			float* x2417 = (float*)myGpuMalloc(x2410 * sizeof(float));
			float* x2418 = (float*)myGpuMalloc(x2408 * sizeof(float));
			float* x2419 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x2420 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x2421 = (float*)myMalloc(1 * sizeof(float));;
			x2421[0] = 0.0f;
			float* x2423 = (float*)myMalloc(1 * sizeof(float));;
			x2423[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2405, x2405));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2405, x2405));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 1024, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2423, x2421, in_desc, x2411, out_desc, x2418, sbmv_desc, x349,
							x646, 0.1, x943, x1096, 1.0E-5,
							x2419, x2420));
			};
			float* x2426 = (float*)myGpuMalloc(x2410 * sizeof(float));
			if (x2329) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2437 = (float*)myGpuMalloc(x2436 * sizeof(float));
			float* x2438 = (float*)myMalloc(1 * sizeof(float));;
			x2438[0] = 0.0f;
			float* x2440 = (float*)myMalloc(1 * sizeof(float));;
			x2440[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2288, x2288));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							1024, 512, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2431, x2431));

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
							x2440, in_desc, x2301, filt_desc, x520,
							conv_desc, algo, ws_data, ws_size,
							x2438, out_desc, x2437));
			};
			float* x2443 = (float*)myGpuMalloc(x2436 * sizeof(float));
			float* x2444 = (float*)myGpuMalloc(x2434 * sizeof(float));
			float* x2445 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x2446 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x2447 = (float*)myMalloc(1 * sizeof(float));;
			x2447[0] = 0.0f;
			float* x2449 = (float*)myMalloc(1 * sizeof(float));;
			x2449[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2431, x2431));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2431, x2431));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 1024, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2449, x2447, in_desc, x2437, out_desc, x2444, sbmv_desc, x382,
							x955, 0.1, x553, x928, 1.0E-5,
							x2445, x2446));
			};
			float* x2452 = (float*)myGpuMalloc(x2436 * sizeof(float));
			if (x2456) {
			} else {
				assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2431) x Sym(2431), res:  x Const(64) x Const(1024) x Sym(2405) x Sym(2405)");
			}
			float* x2461 = (float*)myMalloc(1 * sizeof(float));;
			x2461[0] = 1.0f;
			float* x2463 = (float*)myMalloc(1 * sizeof(float));;
			x2463[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2431, x2431));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2405, x2405));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x2461, bias_desc, x2444, x2463, out_desc, x2418));
			};
			float* x2466 = (float*)myMalloc(1 * sizeof(float));;
			x2466[0] = 0.0f;
			float* x2468 = (float*)myMalloc(1 * sizeof(float));;
			x2468[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2405, x2405));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2468, x_desc, x2418, x2466, x_desc, x2418));
			};
			if (x2472) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2484 = (float*)myGpuMalloc(x2483 * sizeof(float));
			float* x2485 = (float*)myMalloc(1 * sizeof(float));;
			x2485[0] = 0.0f;
			float* x2487 = (float*)myMalloc(1 * sizeof(float));;
			x2487[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2405, x2405));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 1024, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2478, x2478));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x2487, in_desc, x2418, filt_desc, x334,
							conv_desc, algo, ws_data, ws_size,
							x2485, out_desc, x2484));
			};
			float* x2490 = (float*)myGpuMalloc(x2483 * sizeof(float));
			float* x2491 = (float*)myGpuMalloc(x2481 * sizeof(float));
			float* x2492 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2493 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2494 = (float*)myMalloc(1 * sizeof(float));;
			x2494[0] = 0.0f;
			float* x2496 = (float*)myMalloc(1 * sizeof(float));;
			x2496[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2478, x2478));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2478, x2478));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2496, x2494, in_desc, x2484, out_desc, x2491, sbmv_desc, x385,
							x952, 0.1, x1072, x766, 1.0E-5,
							x2492, x2493));
			};
			float* x2499 = (float*)myGpuMalloc(x2483 * sizeof(float));
			float* x2500 = (float*)myMalloc(1 * sizeof(float));;
			x2500[0] = 0.0f;
			float* x2502 = (float*)myMalloc(1 * sizeof(float));;
			x2502[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2478, x2478));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2502, x_desc, x2491, x2500, x_desc, x2491));
			};
			if (x2507) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2520 = (float*)myGpuMalloc(x2519 * sizeof(float));
			float* x2521 = (float*)myMalloc(1 * sizeof(float));;
			x2521[0] = 0.0f;
			float* x2523 = (float*)myMalloc(1 * sizeof(float));;
			x2523[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2478, x2478));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 256, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2514, x2514));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x2523, in_desc, x2491, filt_desc, x388,
							conv_desc, algo, ws_data, ws_size,
							x2521, out_desc, x2520));
			};
			float* x2526 = (float*)myGpuMalloc(x2519 * sizeof(float));
			float* x2527 = (float*)myGpuMalloc(x2517 * sizeof(float));
			float* x2528 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2529 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2530 = (float*)myMalloc(1 * sizeof(float));;
			x2530[0] = 0.0f;
			float* x2532 = (float*)myMalloc(1 * sizeof(float));;
			x2532[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2514, x2514));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2514, x2514));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2532, x2530, in_desc, x2520, out_desc, x2527, sbmv_desc, x1108,
							x583, 0.1, x895, x1006, 1.0E-5,
							x2528, x2529));
			};
			float* x2535 = (float*)myGpuMalloc(x2519 * sizeof(float));
			float* x2536 = (float*)myMalloc(1 * sizeof(float));;
			x2536[0] = 0.0f;
			float* x2538 = (float*)myMalloc(1 * sizeof(float));;
			x2538[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2514, x2514));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2538, x_desc, x2527, x2536, x_desc, x2527));
			};
			if (x2542) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2554 = (float*)myGpuMalloc(x2553 * sizeof(float));
			float* x2555 = (float*)myMalloc(1 * sizeof(float));;
			x2555[0] = 0.0f;
			float* x2557 = (float*)myMalloc(1 * sizeof(float));;
			x2557[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2514, x2514));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							1024, 256, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2548, x2548));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x2557, in_desc, x2527, filt_desc, x463,
							conv_desc, algo, ws_data, ws_size,
							x2555, out_desc, x2554));
			};
			float* x2560 = (float*)myGpuMalloc(x2553 * sizeof(float));
			float* x2561 = (float*)myGpuMalloc(x2551 * sizeof(float));
			float* x2562 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x2563 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x2564 = (float*)myMalloc(1 * sizeof(float));;
			x2564[0] = 0.0f;
			float* x2566 = (float*)myMalloc(1 * sizeof(float));;
			x2566[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2548, x2548));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2548, x2548));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 1024, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2566, x2564, in_desc, x2554, out_desc, x2561, sbmv_desc, x355,
							x991, 0.1, x841, x724, 1.0E-5,
							x2562, x2563));
			};
			float* x2569 = (float*)myGpuMalloc(x2553 * sizeof(float));
			if (x2573) {
			} else {
				assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2405) x Sym(2405), res:  x Const(64) x Const(1024) x Sym(2548) x Sym(2548)");
			}
			float* x2578 = (float*)myMalloc(1 * sizeof(float));;
			x2578[0] = 1.0f;
			float* x2580 = (float*)myMalloc(1 * sizeof(float));;
			x2580[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2405, x2405));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2548, x2548));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x2578, bias_desc, x2418, x2580, out_desc, x2561));
			};
			float* x2583 = (float*)myMalloc(1 * sizeof(float));;
			x2583[0] = 0.0f;
			float* x2585 = (float*)myMalloc(1 * sizeof(float));;
			x2585[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2548, x2548));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2585, x_desc, x2561, x2583, x_desc, x2561));
			};
			if (x2589) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2601 = (float*)myGpuMalloc(x2600 * sizeof(float));
			float* x2602 = (float*)myMalloc(1 * sizeof(float));;
			x2602[0] = 0.0f;
			float* x2604 = (float*)myMalloc(1 * sizeof(float));;
			x2604[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2548, x2548));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 1024, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2595, x2595));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x2604, in_desc, x2561, filt_desc, x949,
							conv_desc, algo, ws_data, ws_size,
							x2602, out_desc, x2601));
			};
			float* x2607 = (float*)myGpuMalloc(x2600 * sizeof(float));
			float* x2608 = (float*)myGpuMalloc(x2598 * sizeof(float));
			float* x2609 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2610 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2611 = (float*)myMalloc(1 * sizeof(float));;
			x2611[0] = 0.0f;
			float* x2613 = (float*)myMalloc(1 * sizeof(float));;
			x2613[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2595, x2595));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2595, x2595));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2613, x2611, in_desc, x2601, out_desc, x2608, sbmv_desc, x682,
							x886, 0.1, x829, x817, 1.0E-5,
							x2609, x2610));
			};
			float* x2616 = (float*)myGpuMalloc(x2600 * sizeof(float));
			float* x2617 = (float*)myMalloc(1 * sizeof(float));;
			x2617[0] = 0.0f;
			float* x2619 = (float*)myMalloc(1 * sizeof(float));;
			x2619[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2595, x2595));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2619, x_desc, x2608, x2617, x_desc, x2608));
			};
			if (x2624) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2637 = (float*)myGpuMalloc(x2636 * sizeof(float));
			float* x2638 = (float*)myMalloc(1 * sizeof(float));;
			x2638[0] = 0.0f;
			float* x2640 = (float*)myMalloc(1 * sizeof(float));;
			x2640[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2595, x2595));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 256, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2631, x2631));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x2640, in_desc, x2608, filt_desc, x337,
							conv_desc, algo, ws_data, ws_size,
							x2638, out_desc, x2637));
			};
			float* x2643 = (float*)myGpuMalloc(x2636 * sizeof(float));
			float* x2644 = (float*)myGpuMalloc(x2634 * sizeof(float));
			float* x2645 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2646 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2647 = (float*)myMalloc(1 * sizeof(float));;
			x2647[0] = 0.0f;
			float* x2649 = (float*)myMalloc(1 * sizeof(float));;
			x2649[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2631, x2631));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2631, x2631));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2649, x2647, in_desc, x2637, out_desc, x2644, sbmv_desc, x979,
							x871, 0.1, x667, x484, 1.0E-5,
							x2645, x2646));
			};
			float* x2652 = (float*)myGpuMalloc(x2636 * sizeof(float));
			float* x2653 = (float*)myMalloc(1 * sizeof(float));;
			x2653[0] = 0.0f;
			float* x2655 = (float*)myMalloc(1 * sizeof(float));;
			x2655[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2631, x2631));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2655, x_desc, x2644, x2653, x_desc, x2644));
			};
			if (x2659) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2671 = (float*)myGpuMalloc(x2670 * sizeof(float));
			float* x2672 = (float*)myMalloc(1 * sizeof(float));;
			x2672[0] = 0.0f;
			float* x2674 = (float*)myMalloc(1 * sizeof(float));;
			x2674[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2631, x2631));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							1024, 256, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2665, x2665));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x2674, in_desc, x2644, filt_desc, x643,
							conv_desc, algo, ws_data, ws_size,
							x2672, out_desc, x2671));
			};
			float* x2677 = (float*)myGpuMalloc(x2670 * sizeof(float));
			float* x2678 = (float*)myGpuMalloc(x2668 * sizeof(float));
			float* x2679 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x2680 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x2681 = (float*)myMalloc(1 * sizeof(float));;
			x2681[0] = 0.0f;
			float* x2683 = (float*)myMalloc(1 * sizeof(float));;
			x2683[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2665, x2665));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2665, x2665));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 1024, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2683, x2681, in_desc, x2671, out_desc, x2678, sbmv_desc, x1084,
							x466, 0.1, x715, x859, 1.0E-5,
							x2679, x2680));
			};
			float* x2686 = (float*)myGpuMalloc(x2670 * sizeof(float));
			if (x2690) {
			} else {
				assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2548) x Sym(2548), res:  x Const(64) x Const(1024) x Sym(2665) x Sym(2665)");
			}
			float* x2695 = (float*)myMalloc(1 * sizeof(float));;
			x2695[0] = 1.0f;
			float* x2697 = (float*)myMalloc(1 * sizeof(float));;
			x2697[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2548, x2548));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2665, x2665));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x2695, bias_desc, x2561, x2697, out_desc, x2678));
			};
			float* x2700 = (float*)myMalloc(1 * sizeof(float));;
			x2700[0] = 0.0f;
			float* x2702 = (float*)myMalloc(1 * sizeof(float));;
			x2702[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2665, x2665));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2702, x_desc, x2678, x2700, x_desc, x2678));
			};
			if (x2706) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2718 = (float*)myGpuMalloc(x2717 * sizeof(float));
			float* x2719 = (float*)myMalloc(1 * sizeof(float));;
			x2719[0] = 0.0f;
			float* x2721 = (float*)myMalloc(1 * sizeof(float));;
			x2721[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2665, x2665));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 1024, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2712, x2712));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x2721, in_desc, x2678, filt_desc, x313,
							conv_desc, algo, ws_data, ws_size,
							x2719, out_desc, x2718));
			};
			float* x2724 = (float*)myGpuMalloc(x2717 * sizeof(float));
			float* x2725 = (float*)myGpuMalloc(x2715 * sizeof(float));
			float* x2726 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2727 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2728 = (float*)myMalloc(1 * sizeof(float));;
			x2728[0] = 0.0f;
			float* x2730 = (float*)myMalloc(1 * sizeof(float));;
			x2730[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2712, x2712));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2712, x2712));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2730, x2728, in_desc, x2718, out_desc, x2725, sbmv_desc, x571,
							x1018, 0.1, x784, x589, 1.0E-5,
							x2726, x2727));
			};
			float* x2733 = (float*)myGpuMalloc(x2717 * sizeof(float));
			float* x2734 = (float*)myMalloc(1 * sizeof(float));;
			x2734[0] = 0.0f;
			float* x2736 = (float*)myMalloc(1 * sizeof(float));;
			x2736[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2712, x2712));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2736, x_desc, x2725, x2734, x_desc, x2725));
			};
			if (x2741) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2754 = (float*)myGpuMalloc(x2753 * sizeof(float));
			float* x2755 = (float*)myMalloc(1 * sizeof(float));;
			x2755[0] = 0.0f;
			float* x2757 = (float*)myMalloc(1 * sizeof(float));;
			x2757[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2712, x2712));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 256, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2748, x2748));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x2757, in_desc, x2725, filt_desc, x1042,
							conv_desc, algo, ws_data, ws_size,
							x2755, out_desc, x2754));
			};
			float* x2760 = (float*)myGpuMalloc(x2753 * sizeof(float));
			float* x2761 = (float*)myGpuMalloc(x2751 * sizeof(float));
			float* x2762 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2763 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2764 = (float*)myMalloc(1 * sizeof(float));;
			x2764[0] = 0.0f;
			float* x2766 = (float*)myMalloc(1 * sizeof(float));;
			x2766[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2748, x2748));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2748, x2748));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2766, x2764, in_desc, x2754, out_desc, x2761, sbmv_desc, x517,
							x703, 0.1, x853, x985, 1.0E-5,
							x2762, x2763));
			};
			float* x2769 = (float*)myGpuMalloc(x2753 * sizeof(float));
			float* x2770 = (float*)myMalloc(1 * sizeof(float));;
			x2770[0] = 0.0f;
			float* x2772 = (float*)myMalloc(1 * sizeof(float));;
			x2772[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2748, x2748));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2772, x_desc, x2761, x2770, x_desc, x2761));
			};
			if (x2776) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2788 = (float*)myGpuMalloc(x2787 * sizeof(float));
			float* x2789 = (float*)myMalloc(1 * sizeof(float));;
			x2789[0] = 0.0f;
			float* x2791 = (float*)myMalloc(1 * sizeof(float));;
			x2791[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2748, x2748));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							1024, 256, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2782, x2782));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x2791, in_desc, x2761, filt_desc, x562,
							conv_desc, algo, ws_data, ws_size,
							x2789, out_desc, x2788));
			};
			float* x2794 = (float*)myGpuMalloc(x2787 * sizeof(float));
			float* x2795 = (float*)myGpuMalloc(x2785 * sizeof(float));
			float* x2796 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x2797 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x2798 = (float*)myMalloc(1 * sizeof(float));;
			x2798[0] = 0.0f;
			float* x2800 = (float*)myMalloc(1 * sizeof(float));;
			x2800[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2782, x2782));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2782, x2782));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 1024, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2800, x2798, in_desc, x2788, out_desc, x2795, sbmv_desc, x1009,
							x733, 0.1, x988, x778, 1.0E-5,
							x2796, x2797));
			};
			float* x2803 = (float*)myGpuMalloc(x2787 * sizeof(float));
			if (x2807) {
			} else {
				assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2665) x Sym(2665), res:  x Const(64) x Const(1024) x Sym(2782) x Sym(2782)");
			}
			float* x2812 = (float*)myMalloc(1 * sizeof(float));;
			x2812[0] = 1.0f;
			float* x2814 = (float*)myMalloc(1 * sizeof(float));;
			x2814[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2665, x2665));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2782, x2782));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x2812, bias_desc, x2678, x2814, out_desc, x2795));
			};
			float* x2817 = (float*)myMalloc(1 * sizeof(float));;
			x2817[0] = 0.0f;
			float* x2819 = (float*)myMalloc(1 * sizeof(float));;
			x2819[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2782, x2782));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2819, x_desc, x2795, x2817, x_desc, x2795));
			};
			if (x2823) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2835 = (float*)myGpuMalloc(x2834 * sizeof(float));
			float* x2836 = (float*)myMalloc(1 * sizeof(float));;
			x2836[0] = 0.0f;
			float* x2838 = (float*)myMalloc(1 * sizeof(float));;
			x2838[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2782, x2782));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 1024, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2829, x2829));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x2838, in_desc, x2795, filt_desc, x361,
							conv_desc, algo, ws_data, ws_size,
							x2836, out_desc, x2835));
			};
			float* x2841 = (float*)myGpuMalloc(x2834 * sizeof(float));
			float* x2842 = (float*)myGpuMalloc(x2832 * sizeof(float));
			float* x2843 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2844 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2845 = (float*)myMalloc(1 * sizeof(float));;
			x2845[0] = 0.0f;
			float* x2847 = (float*)myMalloc(1 * sizeof(float));;
			x2847[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2829, x2829));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2829, x2829));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2847, x2845, in_desc, x2835, out_desc, x2842, sbmv_desc, x526,
							x850, 0.1, x1057, x502, 1.0E-5,
							x2843, x2844));
			};
			float* x2850 = (float*)myGpuMalloc(x2834 * sizeof(float));
			float* x2851 = (float*)myMalloc(1 * sizeof(float));;
			x2851[0] = 0.0f;
			float* x2853 = (float*)myMalloc(1 * sizeof(float));;
			x2853[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2829, x2829));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2853, x_desc, x2842, x2851, x_desc, x2842));
			};
			if (x2858) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2871 = (float*)myGpuMalloc(x2870 * sizeof(float));
			float* x2872 = (float*)myMalloc(1 * sizeof(float));;
			x2872[0] = 0.0f;
			float* x2874 = (float*)myMalloc(1 * sizeof(float));;
			x2874[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2829, x2829));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 256, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2865, x2865));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x2874, in_desc, x2842, filt_desc, x1081,
							conv_desc, algo, ws_data, ws_size,
							x2872, out_desc, x2871));
			};
			float* x2877 = (float*)myGpuMalloc(x2870 * sizeof(float));
			float* x2878 = (float*)myGpuMalloc(x2868 * sizeof(float));
			float* x2879 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2880 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2881 = (float*)myMalloc(1 * sizeof(float));;
			x2881[0] = 0.0f;
			float* x2883 = (float*)myMalloc(1 * sizeof(float));;
			x2883[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2865, x2865));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2865, x2865));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2883, x2881, in_desc, x2871, out_desc, x2878, sbmv_desc, x799,
							x622, 0.1, x1045, x607, 1.0E-5,
							x2879, x2880));
			};
			float* x2886 = (float*)myGpuMalloc(x2870 * sizeof(float));
			float* x2887 = (float*)myMalloc(1 * sizeof(float));;
			x2887[0] = 0.0f;
			float* x2889 = (float*)myMalloc(1 * sizeof(float));;
			x2889[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2865, x2865));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2889, x_desc, x2878, x2887, x_desc, x2878));
			};
			if (x2893) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2905 = (float*)myGpuMalloc(x2904 * sizeof(float));
			float* x2906 = (float*)myMalloc(1 * sizeof(float));;
			x2906[0] = 0.0f;
			float* x2908 = (float*)myMalloc(1 * sizeof(float));;
			x2908[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2865, x2865));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							1024, 256, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2899, x2899));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x2908, in_desc, x2878, filt_desc, x958,
							conv_desc, algo, ws_data, ws_size,
							x2906, out_desc, x2905));
			};
			float* x2911 = (float*)myGpuMalloc(x2904 * sizeof(float));
			float* x2912 = (float*)myGpuMalloc(x2902 * sizeof(float));
			float* x2913 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x2914 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x2915 = (float*)myMalloc(1 * sizeof(float));;
			x2915[0] = 0.0f;
			float* x2917 = (float*)myMalloc(1 * sizeof(float));;
			x2917[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2899, x2899));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2899, x2899));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 1024, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2917, x2915, in_desc, x2905, out_desc, x2912, sbmv_desc, x472,
							x655, 0.1, x922, x1111, 1.0E-5,
							x2913, x2914));
			};
			float* x2920 = (float*)myGpuMalloc(x2904 * sizeof(float));
			if (x2924) {
			} else {
				assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2782) x Sym(2782), res:  x Const(64) x Const(1024) x Sym(2899) x Sym(2899)");
			}
			float* x2929 = (float*)myMalloc(1 * sizeof(float));;
			x2929[0] = 1.0f;
			float* x2931 = (float*)myMalloc(1 * sizeof(float));;
			x2931[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2782, x2782));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2899, x2899));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x2929, bias_desc, x2795, x2931, out_desc, x2912));
			};
			float* x2934 = (float*)myMalloc(1 * sizeof(float));;
			x2934[0] = 0.0f;
			float* x2936 = (float*)myMalloc(1 * sizeof(float));;
			x2936[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2899, x2899));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2936, x_desc, x2912, x2934, x_desc, x2912));
			};
			if (x2940) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2952 = (float*)myGpuMalloc(x2951 * sizeof(float));
			float* x2953 = (float*)myMalloc(1 * sizeof(float));;
			x2953[0] = 0.0f;
			float* x2955 = (float*)myMalloc(1 * sizeof(float));;
			x2955[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2899, x2899));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 1024, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2946, x2946));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x2955, in_desc, x2912, filt_desc, x748,
							conv_desc, algo, ws_data, ws_size,
							x2953, out_desc, x2952));
			};
			float* x2958 = (float*)myGpuMalloc(x2951 * sizeof(float));
			float* x2959 = (float*)myGpuMalloc(x2949 * sizeof(float));
			float* x2960 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2961 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2962 = (float*)myMalloc(1 * sizeof(float));;
			x2962[0] = 0.0f;
			float* x2964 = (float*)myMalloc(1 * sizeof(float));;
			x2964[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2946, x2946));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2946, x2946));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x2964, x2962, in_desc, x2952, out_desc, x2959, sbmv_desc, x550,
							x1054, 0.1, x535, x823, 1.0E-5,
							x2960, x2961));
			};
			float* x2967 = (float*)myGpuMalloc(x2951 * sizeof(float));
			float* x2968 = (float*)myMalloc(1 * sizeof(float));;
			x2968[0] = 0.0f;
			float* x2970 = (float*)myMalloc(1 * sizeof(float));;
			x2970[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2946, x2946));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x2970, x_desc, x2959, x2968, x_desc, x2959));
			};
			if (x2975) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x2988 = (float*)myGpuMalloc(x2987 * sizeof(float));
			float* x2989 = (float*)myMalloc(1 * sizeof(float));;
			x2989[0] = 0.0f;
			float* x2991 = (float*)myMalloc(1 * sizeof(float));;
			x2991[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2946, x2946));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 256, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2982, x2982));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x2991, in_desc, x2959, filt_desc, x973,
							conv_desc, algo, ws_data, ws_size,
							x2989, out_desc, x2988));
			};
			float* x2994 = (float*)myGpuMalloc(x2987 * sizeof(float));
			float* x2995 = (float*)myGpuMalloc(x2985 * sizeof(float));
			float* x2996 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2997 = (float*)myGpuMalloc(256 * sizeof(float));
			float* x2998 = (float*)myMalloc(1 * sizeof(float));;
			x2998[0] = 0.0f;
			float* x3000 = (float*)myMalloc(1 * sizeof(float));;
			x3000[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2982, x2982));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2982, x2982));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3000, x2998, in_desc, x2988, out_desc, x2995, sbmv_desc, x718,
							x862, 0.1, x505, x1015, 1.0E-5,
							x2996, x2997));
			};
			float* x3003 = (float*)myGpuMalloc(x2987 * sizeof(float));
			float* x3004 = (float*)myMalloc(1 * sizeof(float));;
			x3004[0] = 0.0f;
			float* x3006 = (float*)myMalloc(1 * sizeof(float));;
			x3006[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2982, x2982));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x3006, x_desc, x2995, x3004, x_desc, x2995));
			};
			if (x3010) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x3022 = (float*)myGpuMalloc(x3021 * sizeof(float));
			float* x3023 = (float*)myMalloc(1 * sizeof(float));;
			x3023[0] = 0.0f;
			float* x3025 = (float*)myMalloc(1 * sizeof(float));;
			x3025[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2982, x2982));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							1024, 256, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x3016, x3016));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3025, in_desc, x2995, filt_desc, x586,
							conv_desc, algo, ws_data, ws_size,
							x3023, out_desc, x3022));
			};
			float* x3028 = (float*)myGpuMalloc(x3021 * sizeof(float));
			float* x3029 = (float*)myGpuMalloc(x3019 * sizeof(float));
			float* x3030 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x3031 = (float*)myGpuMalloc(1024 * sizeof(float));
			float* x3032 = (float*)myMalloc(1 * sizeof(float));;
			x3032[0] = 0.0f;
			float* x3034 = (float*)myMalloc(1 * sizeof(float));;
			x3034[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x3016, x3016));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x3016, x3016));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 1024, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3034, x3032, in_desc, x3022, out_desc, x3029, sbmv_desc, x1039,
							x574, 0.1, x661, x844, 1.0E-5,
							x3030, x3031));
			};
			float* x3037 = (float*)myGpuMalloc(x3021 * sizeof(float));
			if (x3041) {
			} else {
				assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2899) x Sym(2899), res:  x Const(64) x Const(1024) x Sym(3016) x Sym(3016)");
			}
			float* x3046 = (float*)myMalloc(1 * sizeof(float));;
			x3046[0] = 1.0f;
			float* x3048 = (float*)myMalloc(1 * sizeof(float));;
			x3048[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2899, x2899));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x3016, x3016));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x3046, bias_desc, x2912, x3048, out_desc, x3029));
			};
			float* x3051 = (float*)myMalloc(1 * sizeof(float));;
			x3051[0] = 0.0f;
			float* x3053 = (float*)myMalloc(1 * sizeof(float));;
			x3053[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x3016, x3016));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x3053, x_desc, x3029, x3051, x_desc, x3029));
			};
			if (x3057) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x3069 = (float*)myGpuMalloc(x3068 * sizeof(float));
			float* x3070 = (float*)myMalloc(1 * sizeof(float));;
			x3070[0] = 0.0f;
			float* x3072 = (float*)myMalloc(1 * sizeof(float));;
			x3072[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x3016, x3016));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							512, 1024, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3063, x3063));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3072, in_desc, x3029, filt_desc, x712,
							conv_desc, algo, ws_data, ws_size,
							x3070, out_desc, x3069));
			};
			float* x3075 = (float*)myGpuMalloc(x3068 * sizeof(float));
			float* x3076 = (float*)myGpuMalloc(x3066 * sizeof(float));
			float* x3077 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x3078 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x3079 = (float*)myMalloc(1 * sizeof(float));;
			x3079[0] = 0.0f;
			float* x3081 = (float*)myMalloc(1 * sizeof(float));;
			x3081[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3063, x3063));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3063, x3063));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3081, x3079, in_desc, x3069, out_desc, x3076, sbmv_desc, x898,
							x967, 0.1, x496, x658, 1.0E-5,
							x3077, x3078));
			};
			float* x3084 = (float*)myGpuMalloc(x3068 * sizeof(float));
			float* x3085 = (float*)myMalloc(1 * sizeof(float));;
			x3085[0] = 0.0f;
			float* x3087 = (float*)myMalloc(1 * sizeof(float));;
			x3087[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3063, x3063));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x3087, x_desc, x3076, x3085, x_desc, x3076));
			};
			if (x3092) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x3105 = (float*)myGpuMalloc(x3104 * sizeof(float));
			float* x3106 = (float*)myMalloc(1 * sizeof(float));;
			x3106[0] = 0.0f;
			float* x3108 = (float*)myMalloc(1 * sizeof(float));;
			x3108[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3063, x3063));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							512, 512, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3099, x3099));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 2, 2, 1, 1,
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
							x3108, in_desc, x3076, filt_desc, x397,
							conv_desc, algo, ws_data, ws_size,
							x3106, out_desc, x3105));
			};
			float* x3111 = (float*)myGpuMalloc(x3104 * sizeof(float));
			float* x3112 = (float*)myGpuMalloc(x3102 * sizeof(float));
			float* x3113 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x3114 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x3115 = (float*)myMalloc(1 * sizeof(float));;
			x3115[0] = 0.0f;
			float* x3117 = (float*)myMalloc(1 * sizeof(float));;
			x3117[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3099, x3099));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3099, x3099));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3117, x3115, in_desc, x3105, out_desc, x3112, sbmv_desc, x910,
							x772, 0.1, x634, x445, 1.0E-5,
							x3113, x3114));
			};
			float* x3120 = (float*)myGpuMalloc(x3104 * sizeof(float));
			float* x3121 = (float*)myMalloc(1 * sizeof(float));;
			x3121[0] = 0.0f;
			float* x3123 = (float*)myMalloc(1 * sizeof(float));;
			x3123[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3099, x3099));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x3123, x_desc, x3112, x3121, x_desc, x3112));
			};
			if (x3127) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x3139 = (float*)myGpuMalloc(x3138 * sizeof(float));
			float* x3140 = (float*)myMalloc(1 * sizeof(float));;
			x3140[0] = 0.0f;
			float* x3142 = (float*)myMalloc(1 * sizeof(float));;
			x3142[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3099, x3099));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							2048, 512, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3133, x3133));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3142, in_desc, x3112, filt_desc, x931,
							conv_desc, algo, ws_data, ws_size,
							x3140, out_desc, x3139));
			};
			float* x3145 = (float*)myGpuMalloc(x3138 * sizeof(float));
			float* x3146 = (float*)myGpuMalloc(x3136 * sizeof(float));
			float* x3147 = (float*)myGpuMalloc(2048 * sizeof(float));
			float* x3148 = (float*)myGpuMalloc(2048 * sizeof(float));
			float* x3149 = (float*)myMalloc(1 * sizeof(float));;
			x3149[0] = 0.0f;
			float* x3151 = (float*)myMalloc(1 * sizeof(float));;
			x3151[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3133, x3133));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3133, x3133));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 2048, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3151, x3149, in_desc, x3139, out_desc, x3146, sbmv_desc, x1012,
							x481, 0.1, x640, x874, 1.0E-5,
							x3147, x3148));
			};
			float* x3154 = (float*)myGpuMalloc(x3138 * sizeof(float));
			if (x3057) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x3165 = (float*)myGpuMalloc(x3164 * sizeof(float));
			float* x3166 = (float*)myMalloc(1 * sizeof(float));;
			x3166[0] = 0.0f;
			float* x3168 = (float*)myMalloc(1 * sizeof(float));;
			x3168[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x3016, x3016));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							2048, 1024, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3159, x3159));

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
							x3168, in_desc, x3029, filt_desc, x937,
							conv_desc, algo, ws_data, ws_size,
							x3166, out_desc, x3165));
			};
			float* x3171 = (float*)myGpuMalloc(x3164 * sizeof(float));
			float* x3172 = (float*)myGpuMalloc(x3162 * sizeof(float));
			float* x3173 = (float*)myGpuMalloc(2048 * sizeof(float));
			float* x3174 = (float*)myGpuMalloc(2048 * sizeof(float));
			float* x3175 = (float*)myMalloc(1 * sizeof(float));;
			x3175[0] = 0.0f;
			float* x3177 = (float*)myMalloc(1 * sizeof(float));;
			x3177[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3159, x3159));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3159, x3159));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 2048, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3177, x3175, in_desc, x3165, out_desc, x3172, sbmv_desc, x814,
							x616, 0.1, x487, x670, 1.0E-5,
							x3173, x3174));
			};
			float* x3180 = (float*)myGpuMalloc(x3164 * sizeof(float));
			if (x3184) {
			} else {
				assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(3159) x Sym(3159), res:  x Const(64) x Const(2048) x Sym(3133) x Sym(3133)");
			}
			float* x3189 = (float*)myMalloc(1 * sizeof(float));;
			x3189[0] = 1.0f;
			float* x3191 = (float*)myMalloc(1 * sizeof(float));;
			x3191[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3159, x3159));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3133, x3133));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x3189, bias_desc, x3172, x3191, out_desc, x3146));
			};
			float* x3194 = (float*)myMalloc(1 * sizeof(float));;
			x3194[0] = 0.0f;
			float* x3196 = (float*)myMalloc(1 * sizeof(float));;
			x3196[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3133, x3133));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x3196, x_desc, x3146, x3194, x_desc, x3146));
			};
			if (x3200) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x3212 = (float*)myGpuMalloc(x3211 * sizeof(float));
			float* x3213 = (float*)myMalloc(1 * sizeof(float));;
			x3213[0] = 0.0f;
			float* x3215 = (float*)myMalloc(1 * sizeof(float));;
			x3215[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3133, x3133));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							512, 2048, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3206, x3206));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3215, in_desc, x3146, filt_desc, x940,
							conv_desc, algo, ws_data, ws_size,
							x3213, out_desc, x3212));
			};
			float* x3218 = (float*)myGpuMalloc(x3211 * sizeof(float));
			float* x3219 = (float*)myGpuMalloc(x3209 * sizeof(float));
			float* x3220 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x3221 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x3222 = (float*)myMalloc(1 * sizeof(float));;
			x3222[0] = 0.0f;
			float* x3224 = (float*)myMalloc(1 * sizeof(float));;
			x3224[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3206, x3206));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3206, x3206));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3224, x3222, in_desc, x3212, out_desc, x3219, sbmv_desc, x433,
							x706, 0.1, x757, x490, 1.0E-5,
							x3220, x3221));
			};
			float* x3227 = (float*)myGpuMalloc(x3211 * sizeof(float));
			float* x3228 = (float*)myMalloc(1 * sizeof(float));;
			x3228[0] = 0.0f;
			float* x3230 = (float*)myMalloc(1 * sizeof(float));;
			x3230[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3206, x3206));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x3230, x_desc, x3219, x3228, x_desc, x3219));
			};
			if (x3235) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x3248 = (float*)myGpuMalloc(x3247 * sizeof(float));
			float* x3249 = (float*)myMalloc(1 * sizeof(float));;
			x3249[0] = 0.0f;
			float* x3251 = (float*)myMalloc(1 * sizeof(float));;
			x3251[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3206, x3206));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							512, 512, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3242, x3242));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x3251, in_desc, x3219, filt_desc, x760,
							conv_desc, algo, ws_data, ws_size,
							x3249, out_desc, x3248));
			};
			float* x3254 = (float*)myGpuMalloc(x3247 * sizeof(float));
			float* x3255 = (float*)myGpuMalloc(x3245 * sizeof(float));
			float* x3256 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x3257 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x3258 = (float*)myMalloc(1 * sizeof(float));;
			x3258[0] = 0.0f;
			float* x3260 = (float*)myMalloc(1 * sizeof(float));;
			x3260[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3242, x3242));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3242, x3242));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3260, x3258, in_desc, x3248, out_desc, x3255, sbmv_desc, x775,
							x493, 0.1, x709, x880, 1.0E-5,
							x3256, x3257));
			};
			float* x3263 = (float*)myGpuMalloc(x3247 * sizeof(float));
			float* x3264 = (float*)myMalloc(1 * sizeof(float));;
			x3264[0] = 0.0f;
			float* x3266 = (float*)myMalloc(1 * sizeof(float));;
			x3266[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3242, x3242));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x3266, x_desc, x3255, x3264, x_desc, x3255));
			};
			if (x3270) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x3282 = (float*)myGpuMalloc(x3281 * sizeof(float));
			float* x3283 = (float*)myMalloc(1 * sizeof(float));;
			x3283[0] = 0.0f;
			float* x3285 = (float*)myMalloc(1 * sizeof(float));;
			x3285[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3242, x3242));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							2048, 512, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3276, x3276));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3285, in_desc, x3255, filt_desc, x436,
							conv_desc, algo, ws_data, ws_size,
							x3283, out_desc, x3282));
			};
			float* x3288 = (float*)myGpuMalloc(x3281 * sizeof(float));
			float* x3289 = (float*)myGpuMalloc(x3279 * sizeof(float));
			float* x3290 = (float*)myGpuMalloc(2048 * sizeof(float));
			float* x3291 = (float*)myGpuMalloc(2048 * sizeof(float));
			float* x3292 = (float*)myMalloc(1 * sizeof(float));;
			x3292[0] = 0.0f;
			float* x3294 = (float*)myMalloc(1 * sizeof(float));;
			x3294[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3276, x3276));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3276, x3276));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 2048, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3294, x3292, in_desc, x3282, out_desc, x3289, sbmv_desc, x577,
							x727, 0.1, x499, x1030, 1.0E-5,
							x3290, x3291));
			};
			float* x3297 = (float*)myGpuMalloc(x3281 * sizeof(float));
			if (x3301) {
			} else {
				assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(3133) x Sym(3133), res:  x Const(64) x Const(2048) x Sym(3276) x Sym(3276)");
			}
			float* x3306 = (float*)myMalloc(1 * sizeof(float));;
			x3306[0] = 1.0f;
			float* x3308 = (float*)myMalloc(1 * sizeof(float));;
			x3308[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3133, x3133));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3276, x3276));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x3306, bias_desc, x3146, x3308, out_desc, x3289));
			};
			float* x3311 = (float*)myMalloc(1 * sizeof(float));;
			x3311[0] = 0.0f;
			float* x3313 = (float*)myMalloc(1 * sizeof(float));;
			x3313[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3276, x3276));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x3313, x_desc, x3289, x3311, x_desc, x3289));
			};
			if (x3317) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x3329 = (float*)myGpuMalloc(x3328 * sizeof(float));
			float* x3330 = (float*)myMalloc(1 * sizeof(float));;
			x3330[0] = 0.0f;
			float* x3332 = (float*)myMalloc(1 * sizeof(float));;
			x3332[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3276, x3276));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							512, 2048, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3323, x3323));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3332, in_desc, x3289, filt_desc, x1090,
							conv_desc, algo, ws_data, ws_size,
							x3330, out_desc, x3329));
			};
			float* x3335 = (float*)myGpuMalloc(x3328 * sizeof(float));
			float* x3336 = (float*)myGpuMalloc(x3326 * sizeof(float));
			float* x3337 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x3338 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x3339 = (float*)myMalloc(1 * sizeof(float));;
			x3339[0] = 0.0f;
			float* x3341 = (float*)myMalloc(1 * sizeof(float));;
			x3341[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3323, x3323));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3323, x3323));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3341, x3339, in_desc, x3329, out_desc, x3336, sbmv_desc, x340,
							x529, 0.1, x934, x1060, 1.0E-5,
							x3337, x3338));
			};
			float* x3344 = (float*)myGpuMalloc(x3328 * sizeof(float));
			float* x3345 = (float*)myMalloc(1 * sizeof(float));;
			x3345[0] = 0.0f;
			float* x3347 = (float*)myMalloc(1 * sizeof(float));;
			x3347[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3323, x3323));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x3347, x_desc, x3336, x3345, x_desc, x3336));
			};
			if (x3352) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x3365 = (float*)myGpuMalloc(x3364 * sizeof(float));
			float* x3366 = (float*)myMalloc(1 * sizeof(float));;
			x3366[0] = 0.0f;
			float* x3368 = (float*)myMalloc(1 * sizeof(float));;
			x3368[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3323, x3323));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							512, 512, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3359, x3359));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x3368, in_desc, x3336, filt_desc, x379,
							conv_desc, algo, ws_data, ws_size,
							x3366, out_desc, x3365));
			};
			float* x3371 = (float*)myGpuMalloc(x3364 * sizeof(float));
			float* x3372 = (float*)myGpuMalloc(x3362 * sizeof(float));
			float* x3373 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x3374 = (float*)myGpuMalloc(512 * sizeof(float));
			float* x3375 = (float*)myMalloc(1 * sizeof(float));;
			x3375[0] = 0.0f;
			float* x3377 = (float*)myMalloc(1 * sizeof(float));;
			x3377[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3359, x3359));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3359, x3359));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3377, x3375, in_desc, x3365, out_desc, x3372, sbmv_desc, x877,
							x802, 0.1, x331, x901, 1.0E-5,
							x3373, x3374));
			};
			float* x3380 = (float*)myGpuMalloc(x3364 * sizeof(float));
			float* x3381 = (float*)myMalloc(1 * sizeof(float));;
			x3381[0] = 0.0f;
			float* x3383 = (float*)myMalloc(1 * sizeof(float));;
			x3383[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3359, x3359));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x3383, x_desc, x3372, x3381, x_desc, x3372));
			};
			if (x3387) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x3399 = (float*)myGpuMalloc(x3398 * sizeof(float));
			float* x3400 = (float*)myMalloc(1 * sizeof(float));;
			x3400[0] = 0.0f;
			float* x3402 = (float*)myMalloc(1 * sizeof(float));;
			x3402[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3359, x3359));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							2048, 512, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3393, x3393));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3402, in_desc, x3372, filt_desc, x394,
							conv_desc, algo, ws_data, ws_size,
							x3400, out_desc, x3399));
			};
			float* x3405 = (float*)myGpuMalloc(x3398 * sizeof(float));
			float* x3406 = (float*)myGpuMalloc(x3396 * sizeof(float));
			float* x3407 = (float*)myGpuMalloc(2048 * sizeof(float));
			float* x3408 = (float*)myGpuMalloc(2048 * sizeof(float));
			float* x3409 = (float*)myMalloc(1 * sizeof(float));;
			x3409[0] = 0.0f;
			float* x3411 = (float*)myMalloc(1 * sizeof(float));;
			x3411[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3393, x3393));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3393, x3393));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 2048, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3411, x3409, in_desc, x3399, out_desc, x3406, sbmv_desc, x604,
							x838, 0.1, x1075, x664, 1.0E-5,
							x3407, x3408));
			};
			float* x3414 = (float*)myGpuMalloc(x3398 * sizeof(float));
			if (x3418) {
			} else {
				assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(3276) x Sym(3276), res:  x Const(64) x Const(2048) x Sym(3393) x Sym(3393)");
			}
			float* x3423 = (float*)myMalloc(1 * sizeof(float));;
			x3423[0] = 1.0f;
			float* x3425 = (float*)myMalloc(1 * sizeof(float));;
			x3425[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3276, x3276));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3393, x3393));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x3423, bias_desc, x3289, x3425, out_desc, x3406));
			};
			float* x3428 = (float*)myMalloc(1 * sizeof(float));;
			x3428[0] = 0.0f;
			float* x3430 = (float*)myMalloc(1 * sizeof(float));;
			x3430[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3393, x3393));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x3430, x_desc, x3406, x3428, x_desc, x3406));
			};
			if (x3434) {
			} else {
				assert(false && "Image too small for averagePool_batch:  x Const(64) x Const(2048) x Sym(3393) x Sym(3393)|(2,2)");
			}
			float* x3439 = (float*)myMalloc(1 * sizeof(float));;
			x3439[0] = 0.0f;
			float* x3441 = (float*)myMalloc(1 * sizeof(float));;
			x3441[0] = 1.0f;
			float* x3451 = (float*)myGpuMalloc(x3450 * sizeof(float));

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3393, x3393) );

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3445, x3445));

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
							x3441, in_desc, x3406, x3439, out_desc, x3451));
			};
			float* x3453 = (float*)myGpuMalloc(x3450 * sizeof(float));
			// foward of gemm
			// gemm: List(Const(64), Const(2048)), Vector(Const(10), Const(2048))
			float* x3456 = (float*)myGpuMalloc(640 * sizeof(float));
			float* x3457 = (float*)myMalloc(1 * sizeof(float));;
			x3457[0] = 0.0f;
			float* x3459 = (float*)myMalloc(1 * sizeof(float));;
			x3459[0] = 1.0f;
			CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 10,64,2048,x3459,x976,2048,x3451,2048,x3457,x3456,10));
			float* x3462 = (float*)myGpuMalloc(640 * sizeof(float));
			float* x3463 = (float*)myMalloc(1 * sizeof(float));;
			x3463[0] = 1.0f;
			float* x3465 = (float*)myMalloc(1 * sizeof(float));;
			x3465[0] = 1.0f;

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
							cudnnHandle, x3463, bias_desc, x439, x3465, out_desc, x3456));
			};
			float* x3468 = (float*)myMalloc(1 * sizeof(float));;
			x3468[0] = 0.0f;
			float* x3470 = (float*)myMalloc(1 * sizeof(float));;
			x3470[0] = 1.0f;
			float* x3472 = (float*)myGpuMalloc(640 * sizeof(float));

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 10, 1, 1));
				CUDNN_CALL(cudnnSoftmaxForward(
							cudnnHandle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL,
							x3470, x_desc, x3456, x3468, x_desc, x3472));
			};
			float* x3474 = (float*)myGpuMalloc(640 * sizeof(float));
			float* x3475 = (float*)myGpuMalloc(64 * sizeof(float));
			nllLoss<<<64, 1>>>(x3472, 10, x3475, x1405);
			float* x3477 = (float*)myGpuMalloc(64 * sizeof(float));
			float* x3478 = (float*)myGpuMalloc(1 * sizeof(float));
			float* x3479 = (float*)myMalloc(1 * sizeof(float));;
			x3479[0] = 0.0f;
			float* x3481 = (float*)myMalloc(1 * sizeof(float));;
			x3481[0] = 1.0f;

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
							x3481, x_desc, x3475, x3479, out_desc, x3478));
			};
			float* x3484 = (float*)myGpuMalloc(1 * sizeof(float));
			// make sure the size of loss is 1
			arrayFill_greg<<<28, 512>>>(x3484, 1.0f, 1);
			// backend is lantern.TensorDslCudnn$BackendCudnn@27163b33
			CUDA_CALL(cudaMemcpy(x1410, x3478, 1 * sizeof(float), cudaMemcpyDeviceToHost));
			// 'mean' gradient
			// backprop for mean op
			if (x3496) {
			} else {
				assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(1) x Const(1) x Const(1) x Const(1), res:  x Const(64) x Const(1) x Const(1) x Const(1)");
			}
			float* x3501 = (float*)myMalloc(1 * sizeof(float));;
			x3501[0] = x3491;
			float* x3503 = (float*)myMalloc(1 * sizeof(float));;
			x3503[0] = 1.0f;

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
							cudnnHandle, x3501, bias_desc, x3484, x3503, out_desc, x3477));
			};
			// 'nllLossB' gradient.
			nllLoss_grad<<<64, 1>>>(10, x3477, x1405, x3474);
			float* x3508 = (float*)myMalloc(1 * sizeof(float));;
			x3508[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 10, 1, 1));
				CUDNN_CALL(cudnnSoftmaxBackward(
							cudnnHandle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL,
							x3508, x_desc, x3472, x_desc, x3474,
							x3508, x_desc, x3462));
			};
			float* x3511 = (float*)myMalloc(1 * sizeof(float));;
			x3511[0] = 1.0f;

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
							cudnnHandle, x3511, grad_out_desc, x3462,
							x3511, grad_bias_desc, x1155));
			};
			// backprop for gemm List(Const(64), Const(2048)), Vector(Const(10), Const(2048))
			float* x3515 = (float*)myMalloc(1 * sizeof(float));;
			x3515[0] = 1.0f;
			float* x3517 = (float*)myMalloc(1 * sizeof(float));;
			x3517[0] = 1.0f;
			// backprop of gemm
			CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2048,64,10,x3515,x976,2048,x3462,10,x3517,x3453,2048));
			CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 2048,10,64,x3515,x3451,2048,x3462,10,x3517,x1334,2048));
			float* x3522 = (float*)myMalloc(1 * sizeof(float));;
			x3522[0] = 0.0f;
			float* x3524 = (float*)myMalloc(1 * sizeof(float));;
			x3524[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3393, x3393));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3445, x3445));

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
							x3524, out_desc, x3451, out_desc, x3453, in_desc, x3406  , x3522, in_desc, x3414));
			};
			float* x3527 = (float*)myMalloc(1 * sizeof(float));;
			x3527[0] = 1.0f;
			float* x3529 = (float*)myMalloc(1 * sizeof(float));;
			x3529[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3393, x3393));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x3527, x_desc, x3406, x_desc, x3414, x_desc, x3406,
							x3529, x_desc, x3414));
			};
			if (x3533) {
				if (x3536) {
				} else {
					assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(3393) x Sym(3393), res:  x Const(64) x Const(2048) x Sym(3276) x Sym(3276)");
				}
				float* x3541 = (float*)myMalloc(1 * sizeof(float));;
				x3541[0] = 1.0f;
				float* x3543 = (float*)myMalloc(1 * sizeof(float));;
				x3543[0] = 1.0f;

				{
					cudnnTensorDescriptor_t bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 2048, x3393, x3393));

					cudnnTensorDescriptor_t out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 2048, x3276, x3276));

					CUDNN_CALL(cudnnAddTensor(
								cudnnHandle, x3541, bias_desc, x3414, x3543, out_desc, x3297));
				};
			} else {
				float* x3547 = (float*)myMalloc(1 * sizeof(float));;
				x3547[0] = 1.0f;

				{
					cudnnTensorDescriptor_t grad_bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 2048, x3276, x3276));

					cudnnTensorDescriptor_t grad_out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 2048, x3393, x3393));

					CUDNN_CALL(cudnnConvolutionBackwardBias(
								cudnnHandle, x3547, grad_out_desc, x3414,
								x3547, grad_bias_desc, x3297));
				};
			}
			float* x3552 = (float*)myMalloc(1 * sizeof(float));;
			x3552[0] = 0.0f;
			float* x3554 = (float*)myMalloc(1 * sizeof(float));;
			x3554[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3393, x3393));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3393, x3393));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 2048, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3554, x3554, x3554, x3554, in_desc, x3399,
							out_desc, x3414, in_desc, x3405, sbmv_desc, x604,
							x1210,x1288, 1.0E-5, x3407, x3408));
			};
			// conv2D back-propagate
			float* x3558 = (float*)myMalloc(1 * sizeof(float));;
			x3558[0] = 1.0f;

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
							64, 512, x3359, x3359));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3393, x3393));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3558, filt_desc, x394, grad_out_desc, x3405,
							conv_desc, algo, ws_data, ws_size,
							x3558, grad_in_desc, x3380));
			};
			float* x3561 = (float*)myMalloc(1 * sizeof(float));;
			x3561[0] = 1.0f;

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
							64, 2048, x3393, x3393));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3359, x3359));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3561, in_desc, x3372, grad_out_desc, x3405,
							conv_desc, algo, ws_data, ws_size,
							x3561, grad_filt_desc, x1140));
			};
			float* x3564 = (float*)myMalloc(1 * sizeof(float));;
			x3564[0] = 1.0f;
			float* x3566 = (float*)myMalloc(1 * sizeof(float));;
			x3566[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3359, x3359));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x3564, x_desc, x3372, x_desc, x3380, x_desc, x3372,
							x3566, x_desc, x3380));
			};
			float* x3569 = (float*)myMalloc(1 * sizeof(float));;
			x3569[0] = 0.0f;
			float* x3571 = (float*)myMalloc(1 * sizeof(float));;
			x3571[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3359, x3359));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3359, x3359));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3571, x3571, x3571, x3571, in_desc, x3365,
							out_desc, x3380, in_desc, x3371, sbmv_desc, x877,
							x1301,x1276, 1.0E-5, x3373, x3374));
			};
			// conv2D back-propagate
			float* x3575 = (float*)myMalloc(1 * sizeof(float));;
			x3575[0] = 1.0f;

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
							64, 512, x3323, x3323));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3359, x3359));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x3575, filt_desc, x379, grad_out_desc, x3371,
							conv_desc, algo, ws_data, ws_size,
							x3575, grad_in_desc, x3344));
			};
			float* x3578 = (float*)myMalloc(1 * sizeof(float));;
			x3578[0] = 1.0f;

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
							64, 512, x3359, x3359));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3323, x3323));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x3578, in_desc, x3336, grad_out_desc, x3371,
							conv_desc, algo, ws_data, ws_size,
							x3578, grad_filt_desc, x1135));
			};
			float* x3581 = (float*)myMalloc(1 * sizeof(float));;
			x3581[0] = 1.0f;
			float* x3583 = (float*)myMalloc(1 * sizeof(float));;
			x3583[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3323, x3323));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x3581, x_desc, x3336, x_desc, x3344, x_desc, x3336,
							x3583, x_desc, x3344));
			};
			float* x3586 = (float*)myMalloc(1 * sizeof(float));;
			x3586[0] = 0.0f;
			float* x3588 = (float*)myMalloc(1 * sizeof(float));;
			x3588[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3323, x3323));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3323, x3323));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3588, x3588, x3588, x3588, in_desc, x3329,
							out_desc, x3344, in_desc, x3335, sbmv_desc, x340,
							x1122,x1185, 1.0E-5, x3337, x3338));
			};
			// conv2D back-propagate
			float* x3592 = (float*)myMalloc(1 * sizeof(float));;
			x3592[0] = 1.0f;

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
							64, 2048, x3276, x3276));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3323, x3323));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3592, filt_desc, x1090, grad_out_desc, x3335,
							conv_desc, algo, ws_data, ws_size,
							x3592, grad_in_desc, x3297));
			};
			float* x3595 = (float*)myMalloc(1 * sizeof(float));;
			x3595[0] = 1.0f;

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
							64, 512, x3323, x3323));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3276, x3276));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3595, in_desc, x3289, grad_out_desc, x3335,
							conv_desc, algo, ws_data, ws_size,
							x3595, grad_filt_desc, x1372));
			};
			float* x3598 = (float*)myMalloc(1 * sizeof(float));;
			x3598[0] = 1.0f;
			float* x3600 = (float*)myMalloc(1 * sizeof(float));;
			x3600[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3276, x3276));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x3598, x_desc, x3289, x_desc, x3297, x_desc, x3289,
							x3600, x_desc, x3297));
			};
			if (x3604) {
				if (x3606) {
				} else {
					assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(3276) x Sym(3276), res:  x Const(64) x Const(2048) x Sym(3133) x Sym(3133)");
				}
				float* x3611 = (float*)myMalloc(1 * sizeof(float));;
				x3611[0] = 1.0f;
				float* x3613 = (float*)myMalloc(1 * sizeof(float));;
				x3613[0] = 1.0f;

				{
					cudnnTensorDescriptor_t bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 2048, x3276, x3276));

					cudnnTensorDescriptor_t out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 2048, x3133, x3133));

					CUDNN_CALL(cudnnAddTensor(
								cudnnHandle, x3611, bias_desc, x3297, x3613, out_desc, x3154));
				};
			} else {
				float* x3617 = (float*)myMalloc(1 * sizeof(float));;
				x3617[0] = 1.0f;

				{
					cudnnTensorDescriptor_t grad_bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 2048, x3133, x3133));

					cudnnTensorDescriptor_t grad_out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 2048, x3276, x3276));

					CUDNN_CALL(cudnnConvolutionBackwardBias(
								cudnnHandle, x3617, grad_out_desc, x3297,
								x3617, grad_bias_desc, x3154));
				};
			}
			float* x3622 = (float*)myMalloc(1 * sizeof(float));;
			x3622[0] = 0.0f;
			float* x3624 = (float*)myMalloc(1 * sizeof(float));;
			x3624[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3276, x3276));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3276, x3276));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 2048, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3624, x3624, x3624, x3624, in_desc, x3282,
							out_desc, x3297, in_desc, x3288, sbmv_desc, x577,
							x1201,x1251, 1.0E-5, x3290, x3291));
			};
			// conv2D back-propagate
			float* x3628 = (float*)myMalloc(1 * sizeof(float));;
			x3628[0] = 1.0f;

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
							64, 512, x3242, x3242));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3276, x3276));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3628, filt_desc, x436, grad_out_desc, x3288,
							conv_desc, algo, ws_data, ws_size,
							x3628, grad_in_desc, x3263));
			};
			float* x3631 = (float*)myMalloc(1 * sizeof(float));;
			x3631[0] = 1.0f;

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
							64, 2048, x3276, x3276));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3242, x3242));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3631, in_desc, x3255, grad_out_desc, x3288,
							conv_desc, algo, ws_data, ws_size,
							x3631, grad_filt_desc, x1154));
			};
			float* x3634 = (float*)myMalloc(1 * sizeof(float));;
			x3634[0] = 1.0f;
			float* x3636 = (float*)myMalloc(1 * sizeof(float));;
			x3636[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3242, x3242));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x3634, x_desc, x3255, x_desc, x3263, x_desc, x3255,
							x3636, x_desc, x3263));
			};
			float* x3639 = (float*)myMalloc(1 * sizeof(float));;
			x3639[0] = 0.0f;
			float* x3641 = (float*)myMalloc(1 * sizeof(float));;
			x3641[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3242, x3242));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3242, x3242));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3641, x3641, x3641, x3641, in_desc, x3248,
							out_desc, x3263, in_desc, x3254, sbmv_desc, x775,
							x1267,x1173, 1.0E-5, x3256, x3257));
			};
			// conv2D back-propagate
			float* x3645 = (float*)myMalloc(1 * sizeof(float));;
			x3645[0] = 1.0f;

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
							64, 512, x3206, x3206));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3242, x3242));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x3645, filt_desc, x760, grad_out_desc, x3254,
							conv_desc, algo, ws_data, ws_size,
							x3645, grad_in_desc, x3227));
			};
			float* x3648 = (float*)myMalloc(1 * sizeof(float));;
			x3648[0] = 1.0f;

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
							64, 512, x3242, x3242));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3206, x3206));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x3648, in_desc, x3219, grad_out_desc, x3254,
							conv_desc, algo, ws_data, ws_size,
							x3648, grad_filt_desc, x1262));
			};
			float* x3651 = (float*)myMalloc(1 * sizeof(float));;
			x3651[0] = 1.0f;
			float* x3653 = (float*)myMalloc(1 * sizeof(float));;
			x3653[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3206, x3206));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x3651, x_desc, x3219, x_desc, x3227, x_desc, x3219,
							x3653, x_desc, x3227));
			};
			float* x3656 = (float*)myMalloc(1 * sizeof(float));;
			x3656[0] = 0.0f;
			float* x3658 = (float*)myMalloc(1 * sizeof(float));;
			x3658[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3206, x3206));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3206, x3206));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3658, x3658, x3658, x3658, in_desc, x3212,
							out_desc, x3227, in_desc, x3218, sbmv_desc, x433,
							x1153,x1244, 1.0E-5, x3220, x3221));
			};
			// conv2D back-propagate
			float* x3662 = (float*)myMalloc(1 * sizeof(float));;
			x3662[0] = 1.0f;

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
							64, 2048, x3133, x3133));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3206, x3206));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3662, filt_desc, x940, grad_out_desc, x3218,
							conv_desc, algo, ws_data, ws_size,
							x3662, grad_in_desc, x3154));
			};
			float* x3665 = (float*)myMalloc(1 * sizeof(float));;
			x3665[0] = 1.0f;

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
							64, 512, x3206, x3206));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3133, x3133));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3665, in_desc, x3146, grad_out_desc, x3218,
							conv_desc, algo, ws_data, ws_size,
							x3665, grad_filt_desc, x1322));
			};
			float* x3668 = (float*)myMalloc(1 * sizeof(float));;
			x3668[0] = 1.0f;
			float* x3670 = (float*)myMalloc(1 * sizeof(float));;
			x3670[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3133, x3133));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x3668, x_desc, x3146, x_desc, x3154, x_desc, x3146,
							x3670, x_desc, x3154));
			};
			if (x3674) {
				if (x3676) {
				} else {
					assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(2048) x Sym(3133) x Sym(3133), res:  x Const(64) x Const(2048) x Sym(3159) x Sym(3159)");
				}
				float* x3681 = (float*)myMalloc(1 * sizeof(float));;
				x3681[0] = 1.0f;
				float* x3683 = (float*)myMalloc(1 * sizeof(float));;
				x3683[0] = 1.0f;

				{
					cudnnTensorDescriptor_t bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 2048, x3133, x3133));

					cudnnTensorDescriptor_t out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 2048, x3159, x3159));

					CUDNN_CALL(cudnnAddTensor(
								cudnnHandle, x3681, bias_desc, x3154, x3683, out_desc, x3180));
				};
			} else {
				float* x3687 = (float*)myMalloc(1 * sizeof(float));;
				x3687[0] = 1.0f;

				{
					cudnnTensorDescriptor_t grad_bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 2048, x3159, x3159));

					cudnnTensorDescriptor_t grad_out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 2048, x3133, x3133));

					CUDNN_CALL(cudnnConvolutionBackwardBias(
								cudnnHandle, x3687, grad_out_desc, x3154,
								x3687, grad_bias_desc, x3180));
				};
			}
			float* x3692 = (float*)myMalloc(1 * sizeof(float));;
			x3692[0] = 0.0f;
			float* x3694 = (float*)myMalloc(1 * sizeof(float));;
			x3694[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3159, x3159));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3159, x3159));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 2048, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3694, x3694, x3694, x3694, in_desc, x3165,
							out_desc, x3180, in_desc, x3171, sbmv_desc, x814,
							x1280,x1214, 1.0E-5, x3173, x3174));
			};
			// conv2D back-propagate
			float* x3698 = (float*)myMalloc(1 * sizeof(float));;
			x3698[0] = 1.0f;

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
							64, 1024, x3016, x3016));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3159, x3159));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 2, 2, 1, 1,
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
							x3698, filt_desc, x937, grad_out_desc, x3171,
							conv_desc, algo, ws_data, ws_size,
							x3698, grad_in_desc, x3037));
			};
			float* x3701 = (float*)myMalloc(1 * sizeof(float));;
			x3701[0] = 1.0f;

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
							64, 2048, x3159, x3159));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x3016, x3016));

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
							x3701, in_desc, x3029, grad_out_desc, x3171,
							conv_desc, algo, ws_data, ws_size,
							x3701, grad_filt_desc, x1321));
			};
			float* x3704 = (float*)myMalloc(1 * sizeof(float));;
			x3704[0] = 0.0f;
			float* x3706 = (float*)myMalloc(1 * sizeof(float));;
			x3706[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3133, x3133));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3133, x3133));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 2048, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3706, x3706, x3706, x3706, in_desc, x3139,
							out_desc, x3154, in_desc, x3145, sbmv_desc, x1012,
							x1346,x1169, 1.0E-5, x3147, x3148));
			};
			// conv2D back-propagate
			float* x3710 = (float*)myMalloc(1 * sizeof(float));;
			x3710[0] = 1.0f;

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
							64, 512, x3099, x3099));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 2048, x3133, x3133));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3710, filt_desc, x931, grad_out_desc, x3145,
							conv_desc, algo, ws_data, ws_size,
							x3710, grad_in_desc, x3120));
			};
			float* x3713 = (float*)myMalloc(1 * sizeof(float));;
			x3713[0] = 1.0f;

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
							64, 2048, x3133, x3133));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3099, x3099));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3713, in_desc, x3112, grad_out_desc, x3145,
							conv_desc, algo, ws_data, ws_size,
							x3713, grad_filt_desc, x1319));
			};
			float* x3716 = (float*)myMalloc(1 * sizeof(float));;
			x3716[0] = 1.0f;
			float* x3718 = (float*)myMalloc(1 * sizeof(float));;
			x3718[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3099, x3099));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x3716, x_desc, x3112, x_desc, x3120, x_desc, x3112,
							x3718, x_desc, x3120));
			};
			float* x3721 = (float*)myMalloc(1 * sizeof(float));;
			x3721[0] = 0.0f;
			float* x3723 = (float*)myMalloc(1 * sizeof(float));;
			x3723[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3099, x3099));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3099, x3099));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3723, x3723, x3723, x3723, in_desc, x3105,
							out_desc, x3120, in_desc, x3111, sbmv_desc, x910,
							x1312,x1266, 1.0E-5, x3113, x3114));
			};
			// conv2D back-propagate
			float* x3727 = (float*)myMalloc(1 * sizeof(float));;
			x3727[0] = 1.0f;

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
							64, 512, x3063, x3063));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3099, x3099));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 2, 2, 1, 1,
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
							x3727, filt_desc, x397, grad_out_desc, x3111,
							conv_desc, algo, ws_data, ws_size,
							x3727, grad_in_desc, x3084));
			};
			float* x3730 = (float*)myMalloc(1 * sizeof(float));;
			x3730[0] = 1.0f;

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
							64, 512, x3099, x3099));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3063, x3063));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 2, 2, 1, 1,
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
							x3730, in_desc, x3076, grad_out_desc, x3111,
							conv_desc, algo, ws_data, ws_size,
							x3730, grad_filt_desc, x1141));
			};
			float* x3733 = (float*)myMalloc(1 * sizeof(float));;
			x3733[0] = 1.0f;
			float* x3735 = (float*)myMalloc(1 * sizeof(float));;
			x3735[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3063, x3063));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x3733, x_desc, x3076, x_desc, x3084, x_desc, x3076,
							x3735, x_desc, x3084));
			};
			float* x3738 = (float*)myMalloc(1 * sizeof(float));;
			x3738[0] = 0.0f;
			float* x3740 = (float*)myMalloc(1 * sizeof(float));;
			x3740[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3063, x3063));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3063, x3063));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3740, x3740, x3740, x3740, in_desc, x3069,
							out_desc, x3084, in_desc, x3075, sbmv_desc, x898,
							x1308,x1331, 1.0E-5, x3077, x3078));
			};
			// conv2D back-propagate
			float* x3744 = (float*)myMalloc(1 * sizeof(float));;
			x3744[0] = 1.0f;

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
							64, 1024, x3016, x3016));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x3063, x3063));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3744, filt_desc, x712, grad_out_desc, x3075,
							conv_desc, algo, ws_data, ws_size,
							x3744, grad_in_desc, x3037));
			};
			float* x3747 = (float*)myMalloc(1 * sizeof(float));;
			x3747[0] = 1.0f;

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
							64, 512, x3063, x3063));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x3016, x3016));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3747, in_desc, x3029, grad_out_desc, x3075,
							conv_desc, algo, ws_data, ws_size,
							x3747, grad_filt_desc, x1246));
			};
			float* x3750 = (float*)myMalloc(1 * sizeof(float));;
			x3750[0] = 1.0f;
			float* x3752 = (float*)myMalloc(1 * sizeof(float));;
			x3752[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x3016, x3016));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x3750, x_desc, x3029, x_desc, x3037, x_desc, x3029,
							x3752, x_desc, x3037));
			};
			if (x3756) {
				if (x3759) {
				} else {
					assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(3016) x Sym(3016), res:  x Const(64) x Const(1024) x Sym(2899) x Sym(2899)");
				}
				float* x3764 = (float*)myMalloc(1 * sizeof(float));;
				x3764[0] = 1.0f;
				float* x3766 = (float*)myMalloc(1 * sizeof(float));;
				x3766[0] = 1.0f;

				{
					cudnnTensorDescriptor_t bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x3016, x3016));

					cudnnTensorDescriptor_t out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2899, x2899));

					CUDNN_CALL(cudnnAddTensor(
								cudnnHandle, x3764, bias_desc, x3037, x3766, out_desc, x2920));
				};
			} else {
				float* x3770 = (float*)myMalloc(1 * sizeof(float));;
				x3770[0] = 1.0f;

				{
					cudnnTensorDescriptor_t grad_bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2899, x2899));

					cudnnTensorDescriptor_t grad_out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x3016, x3016));

					CUDNN_CALL(cudnnConvolutionBackwardBias(
								cudnnHandle, x3770, grad_out_desc, x3037,
								x3770, grad_bias_desc, x2920));
				};
			}
			float* x3775 = (float*)myMalloc(1 * sizeof(float));;
			x3775[0] = 0.0f;
			float* x3777 = (float*)myMalloc(1 * sizeof(float));;
			x3777[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x3016, x3016));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x3016, x3016));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 1024, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3777, x3777, x3777, x3777, in_desc, x3022,
							out_desc, x3037, in_desc, x3028, sbmv_desc, x1039,
							x1355,x1200, 1.0E-5, x3030, x3031));
			};
			// conv2D back-propagate
			float* x3781 = (float*)myMalloc(1 * sizeof(float));;
			x3781[0] = 1.0f;

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
							64, 256, x2982, x2982));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x3016, x3016));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3781, filt_desc, x586, grad_out_desc, x3028,
							conv_desc, algo, ws_data, ws_size,
							x3781, grad_in_desc, x3003));
			};
			float* x3784 = (float*)myMalloc(1 * sizeof(float));;
			x3784[0] = 1.0f;

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
							64, 1024, x3016, x3016));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2982, x2982));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3784, in_desc, x2995, grad_out_desc, x3028,
							conv_desc, algo, ws_data, ws_size,
							x3784, grad_filt_desc, x1204));
			};
			float* x3787 = (float*)myMalloc(1 * sizeof(float));;
			x3787[0] = 1.0f;
			float* x3789 = (float*)myMalloc(1 * sizeof(float));;
			x3789[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2982, x2982));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x3787, x_desc, x2995, x_desc, x3003, x_desc, x2995,
							x3789, x_desc, x3003));
			};
			float* x3792 = (float*)myMalloc(1 * sizeof(float));;
			x3792[0] = 0.0f;
			float* x3794 = (float*)myMalloc(1 * sizeof(float));;
			x3794[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2982, x2982));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2982, x2982));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3794, x3794, x3794, x3794, in_desc, x2988,
							out_desc, x3003, in_desc, x2994, sbmv_desc, x718,
							x1248,x1296, 1.0E-5, x2996, x2997));
			};
			// conv2D back-propagate
			float* x3798 = (float*)myMalloc(1 * sizeof(float));;
			x3798[0] = 1.0f;

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
							64, 256, x2946, x2946));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2982, x2982));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x3798, filt_desc, x973, grad_out_desc, x2994,
							conv_desc, algo, ws_data, ws_size,
							x3798, grad_in_desc, x2967));
			};
			float* x3801 = (float*)myMalloc(1 * sizeof(float));;
			x3801[0] = 1.0f;

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
							64, 256, x2982, x2982));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2946, x2946));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x3801, in_desc, x2959, grad_out_desc, x2994,
							conv_desc, algo, ws_data, ws_size,
							x3801, grad_filt_desc, x1333));
			};
			float* x3804 = (float*)myMalloc(1 * sizeof(float));;
			x3804[0] = 1.0f;
			float* x3806 = (float*)myMalloc(1 * sizeof(float));;
			x3806[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2946, x2946));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x3804, x_desc, x2959, x_desc, x2967, x_desc, x2959,
							x3806, x_desc, x2967));
			};
			float* x3809 = (float*)myMalloc(1 * sizeof(float));;
			x3809[0] = 0.0f;
			float* x3811 = (float*)myMalloc(1 * sizeof(float));;
			x3811[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2946, x2946));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2946, x2946));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3811, x3811, x3811, x3811, in_desc, x2952,
							out_desc, x2967, in_desc, x2958, sbmv_desc, x550,
							x1192,x1360, 1.0E-5, x2960, x2961));
			};
			// conv2D back-propagate
			float* x3815 = (float*)myMalloc(1 * sizeof(float));;
			x3815[0] = 1.0f;

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
							64, 1024, x2899, x2899));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2946, x2946));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3815, filt_desc, x748, grad_out_desc, x2958,
							conv_desc, algo, ws_data, ws_size,
							x3815, grad_in_desc, x2920));
			};
			float* x3818 = (float*)myMalloc(1 * sizeof(float));;
			x3818[0] = 1.0f;

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
							64, 256, x2946, x2946));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2899, x2899));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3818, in_desc, x2912, grad_out_desc, x2958,
							conv_desc, algo, ws_data, ws_size,
							x3818, grad_filt_desc, x1258));
			};
			float* x3821 = (float*)myMalloc(1 * sizeof(float));;
			x3821[0] = 1.0f;
			float* x3823 = (float*)myMalloc(1 * sizeof(float));;
			x3823[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2899, x2899));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x3821, x_desc, x2912, x_desc, x2920, x_desc, x2912,
							x3823, x_desc, x2920));
			};
			if (x3827) {
				if (x3829) {
				} else {
					assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2899) x Sym(2899), res:  x Const(64) x Const(1024) x Sym(2782) x Sym(2782)");
				}
				float* x3834 = (float*)myMalloc(1 * sizeof(float));;
				x3834[0] = 1.0f;
				float* x3836 = (float*)myMalloc(1 * sizeof(float));;
				x3836[0] = 1.0f;

				{
					cudnnTensorDescriptor_t bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2899, x2899));

					cudnnTensorDescriptor_t out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2782, x2782));

					CUDNN_CALL(cudnnAddTensor(
								cudnnHandle, x3834, bias_desc, x2920, x3836, out_desc, x2803));
				};
			} else {
				float* x3840 = (float*)myMalloc(1 * sizeof(float));;
				x3840[0] = 1.0f;

				{
					cudnnTensorDescriptor_t grad_bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2782, x2782));

					cudnnTensorDescriptor_t grad_out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2899, x2899));

					CUDNN_CALL(cudnnConvolutionBackwardBias(
								cudnnHandle, x3840, grad_out_desc, x2920,
								x3840, grad_bias_desc, x2803));
				};
			}
			float* x3845 = (float*)myMalloc(1 * sizeof(float));;
			x3845[0] = 0.0f;
			float* x3847 = (float*)myMalloc(1 * sizeof(float));;
			x3847[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2899, x2899));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2899, x2899));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 1024, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3847, x3847, x3847, x3847, in_desc, x2905,
							out_desc, x2920, in_desc, x2911, sbmv_desc, x472,
							x1166,x1227, 1.0E-5, x2913, x2914));
			};
			// conv2D back-propagate
			float* x3851 = (float*)myMalloc(1 * sizeof(float));;
			x3851[0] = 1.0f;

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
							64, 256, x2865, x2865));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2899, x2899));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3851, filt_desc, x958, grad_out_desc, x2911,
							conv_desc, algo, ws_data, ws_size,
							x3851, grad_in_desc, x2886));
			};
			float* x3854 = (float*)myMalloc(1 * sizeof(float));;
			x3854[0] = 1.0f;

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
							64, 1024, x2899, x2899));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2865, x2865));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3854, in_desc, x2878, grad_out_desc, x2911,
							conv_desc, algo, ws_data, ws_size,
							x3854, grad_filt_desc, x1328));
			};
			float* x3857 = (float*)myMalloc(1 * sizeof(float));;
			x3857[0] = 1.0f;
			float* x3859 = (float*)myMalloc(1 * sizeof(float));;
			x3859[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2865, x2865));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x3857, x_desc, x2878, x_desc, x2886, x_desc, x2878,
							x3859, x_desc, x2886));
			};
			float* x3862 = (float*)myMalloc(1 * sizeof(float));;
			x3862[0] = 0.0f;
			float* x3864 = (float*)myMalloc(1 * sizeof(float));;
			x3864[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2865, x2865));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2865, x2865));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3864, x3864, x3864, x3864, in_desc, x2871,
							out_desc, x2886, in_desc, x2877, sbmv_desc, x799,
							x1275,x1216, 1.0E-5, x2879, x2880));
			};
			// conv2D back-propagate
			float* x3868 = (float*)myMalloc(1 * sizeof(float));;
			x3868[0] = 1.0f;

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
							64, 256, x2829, x2829));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2865, x2865));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x3868, filt_desc, x1081, grad_out_desc, x2877,
							conv_desc, algo, ws_data, ws_size,
							x3868, grad_in_desc, x2850));
			};
			float* x3871 = (float*)myMalloc(1 * sizeof(float));;
			x3871[0] = 1.0f;

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
							64, 256, x2865, x2865));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2829, x2829));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x3871, in_desc, x2842, grad_out_desc, x2877,
							conv_desc, algo, ws_data, ws_size,
							x3871, grad_filt_desc, x1369));
			};
			float* x3874 = (float*)myMalloc(1 * sizeof(float));;
			x3874[0] = 1.0f;
			float* x3876 = (float*)myMalloc(1 * sizeof(float));;
			x3876[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2829, x2829));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x3874, x_desc, x2842, x_desc, x2850, x_desc, x2842,
							x3876, x_desc, x2850));
			};
			float* x3879 = (float*)myMalloc(1 * sizeof(float));;
			x3879[0] = 0.0f;
			float* x3881 = (float*)myMalloc(1 * sizeof(float));;
			x3881[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2829, x2829));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2829, x2829));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3881, x3881, x3881, x3881, in_desc, x2835,
							out_desc, x2850, in_desc, x2841, sbmv_desc, x526,
							x1184,x1292, 1.0E-5, x2843, x2844));
			};
			// conv2D back-propagate
			float* x3885 = (float*)myMalloc(1 * sizeof(float));;
			x3885[0] = 1.0f;

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
							64, 1024, x2782, x2782));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2829, x2829));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3885, filt_desc, x361, grad_out_desc, x2841,
							conv_desc, algo, ws_data, ws_size,
							x3885, grad_in_desc, x2803));
			};
			float* x3888 = (float*)myMalloc(1 * sizeof(float));;
			x3888[0] = 1.0f;

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
							64, 256, x2829, x2829));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2782, x2782));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3888, in_desc, x2795, grad_out_desc, x2841,
							conv_desc, algo, ws_data, ws_size,
							x3888, grad_filt_desc, x1129));
			};
			float* x3891 = (float*)myMalloc(1 * sizeof(float));;
			x3891[0] = 1.0f;
			float* x3893 = (float*)myMalloc(1 * sizeof(float));;
			x3893[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2782, x2782));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x3891, x_desc, x2795, x_desc, x2803, x_desc, x2795,
							x3893, x_desc, x2803));
			};
			if (x3897) {
				if (x3899) {
				} else {
					assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2782) x Sym(2782), res:  x Const(64) x Const(1024) x Sym(2665) x Sym(2665)");
				}
				float* x3904 = (float*)myMalloc(1 * sizeof(float));;
				x3904[0] = 1.0f;
				float* x3906 = (float*)myMalloc(1 * sizeof(float));;
				x3906[0] = 1.0f;

				{
					cudnnTensorDescriptor_t bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2782, x2782));

					cudnnTensorDescriptor_t out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2665, x2665));

					CUDNN_CALL(cudnnAddTensor(
								cudnnHandle, x3904, bias_desc, x2803, x3906, out_desc, x2686));
				};
			} else {
				float* x3910 = (float*)myMalloc(1 * sizeof(float));;
				x3910[0] = 1.0f;

				{
					cudnnTensorDescriptor_t grad_bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2665, x2665));

					cudnnTensorDescriptor_t grad_out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2782, x2782));

					CUDNN_CALL(cudnnConvolutionBackwardBias(
								cudnnHandle, x3910, grad_out_desc, x2803,
								x3910, grad_bias_desc, x2686));
				};
			}
			float* x3915 = (float*)myMalloc(1 * sizeof(float));;
			x3915[0] = 0.0f;
			float* x3917 = (float*)myMalloc(1 * sizeof(float));;
			x3917[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2782, x2782));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2782, x2782));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 1024, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3917, x3917, x3917, x3917, in_desc, x2788,
							out_desc, x2803, in_desc, x2794, sbmv_desc, x1009,
							x1345,x1253, 1.0E-5, x2796, x2797));
			};
			// conv2D back-propagate
			float* x3921 = (float*)myMalloc(1 * sizeof(float));;
			x3921[0] = 1.0f;

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
							64, 256, x2748, x2748));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2782, x2782));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3921, filt_desc, x562, grad_out_desc, x2794,
							conv_desc, algo, ws_data, ws_size,
							x3921, grad_in_desc, x2769));
			};
			float* x3924 = (float*)myMalloc(1 * sizeof(float));;
			x3924[0] = 1.0f;

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
							64, 1024, x2782, x2782));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2748, x2748));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3924, in_desc, x2761, grad_out_desc, x2794,
							conv_desc, algo, ws_data, ws_size,
							x3924, grad_filt_desc, x1196));
			};
			float* x3927 = (float*)myMalloc(1 * sizeof(float));;
			x3927[0] = 1.0f;
			float* x3929 = (float*)myMalloc(1 * sizeof(float));;
			x3929[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2748, x2748));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x3927, x_desc, x2761, x_desc, x2769, x_desc, x2761,
							x3929, x_desc, x2769));
			};
			float* x3932 = (float*)myMalloc(1 * sizeof(float));;
			x3932[0] = 0.0f;
			float* x3934 = (float*)myMalloc(1 * sizeof(float));;
			x3934[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2748, x2748));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2748, x2748));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3934, x3934, x3934, x3934, in_desc, x2754,
							out_desc, x2769, in_desc, x2760, sbmv_desc, x517,
							x1181,x1243, 1.0E-5, x2762, x2763));
			};
			// conv2D back-propagate
			float* x3938 = (float*)myMalloc(1 * sizeof(float));;
			x3938[0] = 1.0f;

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
							64, 256, x2712, x2712));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2748, x2748));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x3938, filt_desc, x1042, grad_out_desc, x2760,
							conv_desc, algo, ws_data, ws_size,
							x3938, grad_in_desc, x2733));
			};
			float* x3941 = (float*)myMalloc(1 * sizeof(float));;
			x3941[0] = 1.0f;

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
							64, 256, x2748, x2748));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2712, x2712));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x3941, in_desc, x2725, grad_out_desc, x2760,
							conv_desc, algo, ws_data, ws_size,
							x3941, grad_filt_desc, x1356));
			};
			float* x3944 = (float*)myMalloc(1 * sizeof(float));;
			x3944[0] = 1.0f;
			float* x3946 = (float*)myMalloc(1 * sizeof(float));;
			x3946[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2712, x2712));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x3944, x_desc, x2725, x_desc, x2733, x_desc, x2725,
							x3946, x_desc, x2733));
			};
			float* x3949 = (float*)myMalloc(1 * sizeof(float));;
			x3949[0] = 0.0f;
			float* x3951 = (float*)myMalloc(1 * sizeof(float));;
			x3951[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2712, x2712));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2712, x2712));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3951, x3951, x3951, x3951, in_desc, x2718,
							out_desc, x2733, in_desc, x2724, sbmv_desc, x571,
							x1199,x1348, 1.0E-5, x2726, x2727));
			};
			// conv2D back-propagate
			float* x3955 = (float*)myMalloc(1 * sizeof(float));;
			x3955[0] = 1.0f;

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
							64, 1024, x2665, x2665));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2712, x2712));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3955, filt_desc, x313, grad_out_desc, x2724,
							conv_desc, algo, ws_data, ws_size,
							x3955, grad_in_desc, x2686));
			};
			float* x3958 = (float*)myMalloc(1 * sizeof(float));;
			x3958[0] = 1.0f;

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
							64, 256, x2712, x2712));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2665, x2665));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3958, in_desc, x2678, grad_out_desc, x2724,
							conv_desc, algo, ws_data, ws_size,
							x3958, grad_filt_desc, x1113));
			};
			float* x3961 = (float*)myMalloc(1 * sizeof(float));;
			x3961[0] = 1.0f;
			float* x3963 = (float*)myMalloc(1 * sizeof(float));;
			x3963[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2665, x2665));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x3961, x_desc, x2678, x_desc, x2686, x_desc, x2678,
							x3963, x_desc, x2686));
			};
			if (x3967) {
				if (x3969) {
				} else {
					assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2665) x Sym(2665), res:  x Const(64) x Const(1024) x Sym(2548) x Sym(2548)");
				}
				float* x3974 = (float*)myMalloc(1 * sizeof(float));;
				x3974[0] = 1.0f;
				float* x3976 = (float*)myMalloc(1 * sizeof(float));;
				x3976[0] = 1.0f;

				{
					cudnnTensorDescriptor_t bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2665, x2665));

					cudnnTensorDescriptor_t out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2548, x2548));

					CUDNN_CALL(cudnnAddTensor(
								cudnnHandle, x3974, bias_desc, x2686, x3976, out_desc, x2569));
				};
			} else {
				float* x3980 = (float*)myMalloc(1 * sizeof(float));;
				x3980[0] = 1.0f;

				{
					cudnnTensorDescriptor_t grad_bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2548, x2548));

					cudnnTensorDescriptor_t grad_out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2665, x2665));

					CUDNN_CALL(cudnnConvolutionBackwardBias(
								cudnnHandle, x3980, grad_out_desc, x2686,
								x3980, grad_bias_desc, x2569));
				};
			}
			float* x3985 = (float*)myMalloc(1 * sizeof(float));;
			x3985[0] = 0.0f;
			float* x3987 = (float*)myMalloc(1 * sizeof(float));;
			x3987[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2665, x2665));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2665, x2665));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 1024, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x3987, x3987, x3987, x3987, in_desc, x2671,
							out_desc, x2686, in_desc, x2677, sbmv_desc, x1084,
							x1370,x1164, 1.0E-5, x2679, x2680));
			};
			// conv2D back-propagate
			float* x3991 = (float*)myMalloc(1 * sizeof(float));;
			x3991[0] = 1.0f;

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
							64, 256, x2631, x2631));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2665, x2665));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3991, filt_desc, x643, grad_out_desc, x2677,
							conv_desc, algo, ws_data, ws_size,
							x3991, grad_in_desc, x2652));
			};
			float* x3994 = (float*)myMalloc(1 * sizeof(float));;
			x3994[0] = 1.0f;

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
							64, 1024, x2665, x2665));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2631, x2631));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x3994, in_desc, x2644, grad_out_desc, x2677,
							conv_desc, algo, ws_data, ws_size,
							x3994, grad_filt_desc, x1223));
			};
			float* x3997 = (float*)myMalloc(1 * sizeof(float));;
			x3997[0] = 1.0f;
			float* x3999 = (float*)myMalloc(1 * sizeof(float));;
			x3999[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2631, x2631));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x3997, x_desc, x2644, x_desc, x2652, x_desc, x2644,
							x3999, x_desc, x2652));
			};
			float* x4002 = (float*)myMalloc(1 * sizeof(float));;
			x4002[0] = 0.0f;
			float* x4004 = (float*)myMalloc(1 * sizeof(float));;
			x4004[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2631, x2631));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2631, x2631));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4004, x4004, x4004, x4004, in_desc, x2637,
							out_desc, x2652, in_desc, x2643, sbmv_desc, x979,
							x1335,x1299, 1.0E-5, x2645, x2646));
			};
			// conv2D back-propagate
			float* x4008 = (float*)myMalloc(1 * sizeof(float));;
			x4008[0] = 1.0f;

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
							64, 256, x2595, x2595));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2631, x2631));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x4008, filt_desc, x337, grad_out_desc, x2643,
							conv_desc, algo, ws_data, ws_size,
							x4008, grad_in_desc, x2616));
			};
			float* x4011 = (float*)myMalloc(1 * sizeof(float));;
			x4011[0] = 1.0f;

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
							64, 256, x2631, x2631));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2595, x2595));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x4011, in_desc, x2608, grad_out_desc, x2643,
							conv_desc, algo, ws_data, ws_size,
							x4011, grad_filt_desc, x1121));
			};
			float* x4014 = (float*)myMalloc(1 * sizeof(float));;
			x4014[0] = 1.0f;
			float* x4016 = (float*)myMalloc(1 * sizeof(float));;
			x4016[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2595, x2595));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4014, x_desc, x2608, x_desc, x2616, x_desc, x2608,
							x4016, x_desc, x2616));
			};
			float* x4019 = (float*)myMalloc(1 * sizeof(float));;
			x4019[0] = 0.0f;
			float* x4021 = (float*)myMalloc(1 * sizeof(float));;
			x4021[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2595, x2595));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2595, x2595));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4021, x4021, x4021, x4021, in_desc, x2601,
							out_desc, x2616, in_desc, x2607, sbmv_desc, x682,
							x1236,x1304, 1.0E-5, x2609, x2610));
			};
			// conv2D back-propagate
			float* x4025 = (float*)myMalloc(1 * sizeof(float));;
			x4025[0] = 1.0f;

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
							64, 1024, x2548, x2548));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2595, x2595));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4025, filt_desc, x949, grad_out_desc, x2607,
							conv_desc, algo, ws_data, ws_size,
							x4025, grad_in_desc, x2569));
			};
			float* x4028 = (float*)myMalloc(1 * sizeof(float));;
			x4028[0] = 1.0f;

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
							64, 256, x2595, x2595));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2548, x2548));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4028, in_desc, x2561, grad_out_desc, x2607,
							conv_desc, algo, ws_data, ws_size,
							x4028, grad_filt_desc, x1325));
			};
			float* x4031 = (float*)myMalloc(1 * sizeof(float));;
			x4031[0] = 1.0f;
			float* x4033 = (float*)myMalloc(1 * sizeof(float));;
			x4033[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2548, x2548));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4031, x_desc, x2561, x_desc, x2569, x_desc, x2561,
							x4033, x_desc, x2569));
			};
			if (x4037) {
				if (x4039) {
				} else {
					assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2548) x Sym(2548), res:  x Const(64) x Const(1024) x Sym(2405) x Sym(2405)");
				}
				float* x4044 = (float*)myMalloc(1 * sizeof(float));;
				x4044[0] = 1.0f;
				float* x4046 = (float*)myMalloc(1 * sizeof(float));;
				x4046[0] = 1.0f;

				{
					cudnnTensorDescriptor_t bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2548, x2548));

					cudnnTensorDescriptor_t out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2405, x2405));

					CUDNN_CALL(cudnnAddTensor(
								cudnnHandle, x4044, bias_desc, x2569, x4046, out_desc, x2426));
				};
			} else {
				float* x4050 = (float*)myMalloc(1 * sizeof(float));;
				x4050[0] = 1.0f;

				{
					cudnnTensorDescriptor_t grad_bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2405, x2405));

					cudnnTensorDescriptor_t grad_out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2548, x2548));

					CUDNN_CALL(cudnnConvolutionBackwardBias(
								cudnnHandle, x4050, grad_out_desc, x2569,
								x4050, grad_bias_desc, x2426));
				};
			}
			float* x4055 = (float*)myMalloc(1 * sizeof(float));;
			x4055[0] = 0.0f;
			float* x4057 = (float*)myMalloc(1 * sizeof(float));;
			x4057[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2548, x2548));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2548, x2548));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 1024, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4057, x4057, x4057, x4057, in_desc, x2554,
							out_desc, x2569, in_desc, x2560, sbmv_desc, x355,
							x1127,x1339, 1.0E-5, x2562, x2563));
			};
			// conv2D back-propagate
			float* x4061 = (float*)myMalloc(1 * sizeof(float));;
			x4061[0] = 1.0f;

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
							64, 256, x2514, x2514));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2548, x2548));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4061, filt_desc, x463, grad_out_desc, x2560,
							conv_desc, algo, ws_data, ws_size,
							x4061, grad_in_desc, x2535));
			};
			float* x4064 = (float*)myMalloc(1 * sizeof(float));;
			x4064[0] = 1.0f;

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
							64, 1024, x2548, x2548));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2514, x2514));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4064, in_desc, x2527, grad_out_desc, x2560,
							conv_desc, algo, ws_data, ws_size,
							x4064, grad_filt_desc, x1163));
			};
			float* x4067 = (float*)myMalloc(1 * sizeof(float));;
			x4067[0] = 1.0f;
			float* x4069 = (float*)myMalloc(1 * sizeof(float));;
			x4069[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2514, x2514));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4067, x_desc, x2527, x_desc, x2535, x_desc, x2527,
							x4069, x_desc, x2535));
			};
			float* x4072 = (float*)myMalloc(1 * sizeof(float));;
			x4072[0] = 0.0f;
			float* x4074 = (float*)myMalloc(1 * sizeof(float));;
			x4074[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2514, x2514));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2514, x2514));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4074, x4074, x4074, x4074, in_desc, x2520,
							out_desc, x2535, in_desc, x2526, sbmv_desc, x1108,
							x1378,x1203, 1.0E-5, x2528, x2529));
			};
			// conv2D back-propagate
			float* x4078 = (float*)myMalloc(1 * sizeof(float));;
			x4078[0] = 1.0f;

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
							64, 256, x2478, x2478));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2514, x2514));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x4078, filt_desc, x388, grad_out_desc, x2526,
							conv_desc, algo, ws_data, ws_size,
							x4078, grad_in_desc, x2499));
			};
			float* x4081 = (float*)myMalloc(1 * sizeof(float));;
			x4081[0] = 1.0f;

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
							64, 256, x2514, x2514));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2478, x2478));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x4081, in_desc, x2491, grad_out_desc, x2526,
							conv_desc, algo, ws_data, ws_size,
							x4081, grad_filt_desc, x1138));
			};
			float* x4084 = (float*)myMalloc(1 * sizeof(float));;
			x4084[0] = 1.0f;
			float* x4086 = (float*)myMalloc(1 * sizeof(float));;
			x4086[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2478, x2478));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4084, x_desc, x2491, x_desc, x2499, x_desc, x2491,
							x4086, x_desc, x2499));
			};
			float* x4089 = (float*)myMalloc(1 * sizeof(float));;
			x4089[0] = 0.0f;
			float* x4091 = (float*)myMalloc(1 * sizeof(float));;
			x4091[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2478, x2478));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2478, x2478));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4091, x4091, x4091, x4091, in_desc, x2484,
							out_desc, x2499, in_desc, x2490, sbmv_desc, x385,
							x1137,x1326, 1.0E-5, x2492, x2493));
			};
			// conv2D back-propagate
			float* x4095 = (float*)myMalloc(1 * sizeof(float));;
			x4095[0] = 1.0f;

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
							64, 1024, x2405, x2405));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2478, x2478));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4095, filt_desc, x334, grad_out_desc, x2490,
							conv_desc, algo, ws_data, ws_size,
							x4095, grad_in_desc, x2426));
			};
			float* x4098 = (float*)myMalloc(1 * sizeof(float));;
			x4098[0] = 1.0f;

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
							64, 256, x2478, x2478));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2405, x2405));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4098, in_desc, x2418, grad_out_desc, x2490,
							conv_desc, algo, ws_data, ws_size,
							x4098, grad_filt_desc, x1120));
			};
			float* x4101 = (float*)myMalloc(1 * sizeof(float));;
			x4101[0] = 1.0f;
			float* x4103 = (float*)myMalloc(1 * sizeof(float));;
			x4103[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2405, x2405));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4101, x_desc, x2418, x_desc, x2426, x_desc, x2418,
							x4103, x_desc, x2426));
			};
			if (x4107) {
				if (x4109) {
				} else {
					assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(1024) x Sym(2405) x Sym(2405), res:  x Const(64) x Const(1024) x Sym(2431) x Sym(2431)");
				}
				float* x4114 = (float*)myMalloc(1 * sizeof(float));;
				x4114[0] = 1.0f;
				float* x4116 = (float*)myMalloc(1 * sizeof(float));;
				x4116[0] = 1.0f;

				{
					cudnnTensorDescriptor_t bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2405, x2405));

					cudnnTensorDescriptor_t out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2431, x2431));

					CUDNN_CALL(cudnnAddTensor(
								cudnnHandle, x4114, bias_desc, x2426, x4116, out_desc, x2452));
				};
			} else {
				float* x4120 = (float*)myMalloc(1 * sizeof(float));;
				x4120[0] = 1.0f;

				{
					cudnnTensorDescriptor_t grad_bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2431, x2431));

					cudnnTensorDescriptor_t grad_out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 1024, x2405, x2405));

					CUDNN_CALL(cudnnConvolutionBackwardBias(
								cudnnHandle, x4120, grad_out_desc, x2426,
								x4120, grad_bias_desc, x2452));
				};
			}
			float* x4125 = (float*)myMalloc(1 * sizeof(float));;
			x4125[0] = 0.0f;
			float* x4127 = (float*)myMalloc(1 * sizeof(float));;
			x4127[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2431, x2431));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2431, x2431));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 1024, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4127, x4127, x4127, x4127, in_desc, x2437,
							out_desc, x2452, in_desc, x2443, sbmv_desc, x382,
							x1136,x1327, 1.0E-5, x2445, x2446));
			};
			// conv2D back-propagate
			float* x4131 = (float*)myMalloc(1 * sizeof(float));;
			x4131[0] = 1.0f;

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
							64, 512, x2288, x2288));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2431, x2431));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 2, 2, 1, 1,
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
							x4131, filt_desc, x520, grad_out_desc, x2443,
							conv_desc, algo, ws_data, ws_size,
							x4131, grad_in_desc, x2309));
			};
			float* x4134 = (float*)myMalloc(1 * sizeof(float));;
			x4134[0] = 1.0f;

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
							64, 1024, x2431, x2431));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2288, x2288));

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
							x4134, in_desc, x2301, grad_out_desc, x2443,
							conv_desc, algo, ws_data, ws_size,
							x4134, grad_filt_desc, x1182));
			};
			float* x4137 = (float*)myMalloc(1 * sizeof(float));;
			x4137[0] = 0.0f;
			float* x4139 = (float*)myMalloc(1 * sizeof(float));;
			x4139[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2405, x2405));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2405, x2405));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 1024, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4139, x4139, x4139, x4139, in_desc, x2411,
							out_desc, x2426, in_desc, x2417, sbmv_desc, x349,
							x1125,x1224, 1.0E-5, x2419, x2420));
			};
			// conv2D back-propagate
			float* x4143 = (float*)myMalloc(1 * sizeof(float));;
			x4143[0] = 1.0f;

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
							64, 256, x2371, x2371));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 1024, x2405, x2405));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4143, filt_desc, x1102, grad_out_desc, x2417,
							conv_desc, algo, ws_data, ws_size,
							x4143, grad_in_desc, x2392));
			};
			float* x4146 = (float*)myMalloc(1 * sizeof(float));;
			x4146[0] = 1.0f;

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
							64, 1024, x2405, x2405));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2371, x2371));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4146, in_desc, x2384, grad_out_desc, x2417,
							conv_desc, algo, ws_data, ws_size,
							x4146, grad_filt_desc, x1376));
			};
			float* x4149 = (float*)myMalloc(1 * sizeof(float));;
			x4149[0] = 1.0f;
			float* x4151 = (float*)myMalloc(1 * sizeof(float));;
			x4151[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2371, x2371));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4149, x_desc, x2384, x_desc, x2392, x_desc, x2384,
							x4151, x_desc, x2392));
			};
			float* x4154 = (float*)myMalloc(1 * sizeof(float));;
			x4154[0] = 0.0f;
			float* x4156 = (float*)myMalloc(1 * sizeof(float));;
			x4156[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2371, x2371));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2371, x2371));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4156, x4156, x4156, x4156, in_desc, x2377,
							out_desc, x2392, in_desc, x2383, sbmv_desc, x619,
							x1215,x1123, 1.0E-5, x2385, x2386));
			};
			// conv2D back-propagate
			float* x4160 = (float*)myMalloc(1 * sizeof(float));;
			x4160[0] = 1.0f;

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
							64, 256, x2335, x2335));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2371, x2371));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 2, 2, 1, 1,
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
							x4160, filt_desc, x820, grad_out_desc, x2383,
							conv_desc, algo, ws_data, ws_size,
							x4160, grad_in_desc, x2356));
			};
			float* x4163 = (float*)myMalloc(1 * sizeof(float));;
			x4163[0] = 1.0f;

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
							64, 256, x2371, x2371));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2335, x2335));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 2, 2, 1, 1,
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
							x4163, in_desc, x2348, grad_out_desc, x2383,
							conv_desc, algo, ws_data, ws_size,
							x4163, grad_filt_desc, x1282));
			};
			float* x4166 = (float*)myMalloc(1 * sizeof(float));;
			x4166[0] = 1.0f;
			float* x4168 = (float*)myMalloc(1 * sizeof(float));;
			x4168[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2335, x2335));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4166, x_desc, x2348, x_desc, x2356, x_desc, x2348,
							x4168, x_desc, x2356));
			};
			float* x4171 = (float*)myMalloc(1 * sizeof(float));;
			x4171[0] = 0.0f;
			float* x4173 = (float*)myMalloc(1 * sizeof(float));;
			x4173[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2335, x2335));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2335, x2335));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4173, x4173, x4173, x4173, in_desc, x2341,
							out_desc, x2356, in_desc, x2347, sbmv_desc, x1105,
							x1377,x1128, 1.0E-5, x2349, x2350));
			};
			// conv2D back-propagate
			float* x4177 = (float*)myMalloc(1 * sizeof(float));;
			x4177[0] = 1.0f;

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
							64, 512, x2288, x2288));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x2335, x2335));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4177, filt_desc, x835, grad_out_desc, x2347,
							conv_desc, algo, ws_data, ws_size,
							x4177, grad_in_desc, x2309));
			};
			float* x4180 = (float*)myMalloc(1 * sizeof(float));;
			x4180[0] = 1.0f;

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
							64, 256, x2335, x2335));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2288, x2288));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4180, in_desc, x2301, grad_out_desc, x2347,
							conv_desc, algo, ws_data, ws_size,
							x4180, grad_filt_desc, x1287));
			};
			float* x4183 = (float*)myMalloc(1 * sizeof(float));;
			x4183[0] = 1.0f;
			float* x4185 = (float*)myMalloc(1 * sizeof(float));;
			x4185[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2288, x2288));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4183, x_desc, x2301, x_desc, x2309, x_desc, x2301,
							x4185, x_desc, x2309));
			};
			if (x4189) {
				if (x4192) {
				} else {
					assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(2288) x Sym(2288), res:  x Const(64) x Const(512) x Sym(2171) x Sym(2171)");
				}
				float* x4197 = (float*)myMalloc(1 * sizeof(float));;
				x4197[0] = 1.0f;
				float* x4199 = (float*)myMalloc(1 * sizeof(float));;
				x4199[0] = 1.0f;

				{
					cudnnTensorDescriptor_t bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 512, x2288, x2288));

					cudnnTensorDescriptor_t out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 512, x2171, x2171));

					CUDNN_CALL(cudnnAddTensor(
								cudnnHandle, x4197, bias_desc, x2309, x4199, out_desc, x2192));
				};
			} else {
				float* x4203 = (float*)myMalloc(1 * sizeof(float));;
				x4203[0] = 1.0f;

				{
					cudnnTensorDescriptor_t grad_bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 512, x2171, x2171));

					cudnnTensorDescriptor_t grad_out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 512, x2288, x2288));

					CUDNN_CALL(cudnnConvolutionBackwardBias(
								cudnnHandle, x4203, grad_out_desc, x2309,
								x4203, grad_bias_desc, x2192));
				};
			}
			float* x4208 = (float*)myMalloc(1 * sizeof(float));;
			x4208[0] = 0.0f;
			float* x4210 = (float*)myMalloc(1 * sizeof(float));;
			x4210[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2288, x2288));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2288, x2288));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4210, x4210, x4210, x4210, in_desc, x2294,
							out_desc, x2309, in_desc, x2300, sbmv_desc, x763,
							x1263,x1161, 1.0E-5, x2302, x2303));
			};
			// conv2D back-propagate
			float* x4214 = (float*)myMalloc(1 * sizeof(float));;
			x4214[0] = 1.0f;

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
							64, 128, x2254, x2254));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2288, x2288));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4214, filt_desc, x460, grad_out_desc, x2300,
							conv_desc, algo, ws_data, ws_size,
							x4214, grad_in_desc, x2275));
			};
			float* x4217 = (float*)myMalloc(1 * sizeof(float));;
			x4217[0] = 1.0f;

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
							64, 512, x2288, x2288));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2254, x2254));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4217, in_desc, x2267, grad_out_desc, x2300,
							conv_desc, algo, ws_data, ws_size,
							x4217, grad_filt_desc, x1162));
			};
			float* x4220 = (float*)myMalloc(1 * sizeof(float));;
			x4220[0] = 1.0f;
			float* x4222 = (float*)myMalloc(1 * sizeof(float));;
			x4222[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2254, x2254));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4220, x_desc, x2267, x_desc, x2275, x_desc, x2267,
							x4222, x_desc, x2275));
			};
			float* x4225 = (float*)myMalloc(1 * sizeof(float));;
			x4225[0] = 0.0f;
			float* x4227 = (float*)myMalloc(1 * sizeof(float));;
			x4227[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2254, x2254));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2254, x2254));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4227, x4227, x4227, x4227, in_desc, x2260,
							out_desc, x2275, in_desc, x2266, sbmv_desc, x532,
							x1186,x1145, 1.0E-5, x2268, x2269));
			};
			// conv2D back-propagate
			float* x4231 = (float*)myMalloc(1 * sizeof(float));;
			x4231[0] = 1.0f;

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
							64, 128, x2218, x2218));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2254, x2254));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x4231, filt_desc, x790, grad_out_desc, x2266,
							conv_desc, algo, ws_data, ws_size,
							x4231, grad_in_desc, x2239));
			};
			float* x4234 = (float*)myMalloc(1 * sizeof(float));;
			x4234[0] = 1.0f;

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
							64, 128, x2254, x2254));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2218, x2218));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x4234, in_desc, x2231, grad_out_desc, x2266,
							conv_desc, algo, ws_data, ws_size,
							x4234, grad_filt_desc, x1272));
			};
			float* x4237 = (float*)myMalloc(1 * sizeof(float));;
			x4237[0] = 1.0f;
			float* x4239 = (float*)myMalloc(1 * sizeof(float));;
			x4239[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2218, x2218));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4237, x_desc, x2231, x_desc, x2239, x_desc, x2231,
							x4239, x_desc, x2239));
			};
			float* x4242 = (float*)myMalloc(1 * sizeof(float));;
			x4242[0] = 0.0f;
			float* x4244 = (float*)myMalloc(1 * sizeof(float));;
			x4244[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2218, x2218));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2218, x2218));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4244, x4244, x4244, x4244, in_desc, x2224,
							out_desc, x2239, in_desc, x2230, sbmv_desc, x412,
							x1146,x1349, 1.0E-5, x2232, x2233));
			};
			// conv2D back-propagate
			float* x4248 = (float*)myMalloc(1 * sizeof(float));;
			x4248[0] = 1.0f;

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
							64, 512, x2171, x2171));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2218, x2218));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4248, filt_desc, x691, grad_out_desc, x2230,
							conv_desc, algo, ws_data, ws_size,
							x4248, grad_in_desc, x2192));
			};
			float* x4251 = (float*)myMalloc(1 * sizeof(float));;
			x4251[0] = 1.0f;

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
							64, 128, x2218, x2218));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2171, x2171));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4251, in_desc, x2184, grad_out_desc, x2230,
							conv_desc, algo, ws_data, ws_size,
							x4251, grad_filt_desc, x1239));
			};
			float* x4254 = (float*)myMalloc(1 * sizeof(float));;
			x4254[0] = 1.0f;
			float* x4256 = (float*)myMalloc(1 * sizeof(float));;
			x4256[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2171, x2171));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4254, x_desc, x2184, x_desc, x2192, x_desc, x2184,
							x4256, x_desc, x2192));
			};
			if (x4260) {
				if (x4262) {
				} else {
					assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(2171) x Sym(2171), res:  x Const(64) x Const(512) x Sym(2054) x Sym(2054)");
				}
				float* x4267 = (float*)myMalloc(1 * sizeof(float));;
				x4267[0] = 1.0f;
				float* x4269 = (float*)myMalloc(1 * sizeof(float));;
				x4269[0] = 1.0f;

				{
					cudnnTensorDescriptor_t bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 512, x2171, x2171));

					cudnnTensorDescriptor_t out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 512, x2054, x2054));

					CUDNN_CALL(cudnnAddTensor(
								cudnnHandle, x4267, bias_desc, x2192, x4269, out_desc, x2075));
				};
			} else {
				float* x4273 = (float*)myMalloc(1 * sizeof(float));;
				x4273[0] = 1.0f;

				{
					cudnnTensorDescriptor_t grad_bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 512, x2054, x2054));

					cudnnTensorDescriptor_t grad_out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 512, x2171, x2171));

					CUDNN_CALL(cudnnConvolutionBackwardBias(
								cudnnHandle, x4273, grad_out_desc, x2192,
								x4273, grad_bias_desc, x2075));
				};
			}
			float* x4278 = (float*)myMalloc(1 * sizeof(float));;
			x4278[0] = 0.0f;
			float* x4280 = (float*)myMalloc(1 * sizeof(float));;
			x4280[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2171, x2171));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2171, x2171));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4280, x4280, x4280, x4280, in_desc, x2177,
							out_desc, x2192, in_desc, x2183, sbmv_desc, x796,
							x1274,x1189, 1.0E-5, x2185, x2186));
			};
			// conv2D back-propagate
			float* x4284 = (float*)myMalloc(1 * sizeof(float));;
			x4284[0] = 1.0f;

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
							64, 128, x2137, x2137));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2171, x2171));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4284, filt_desc, x418, grad_out_desc, x2183,
							conv_desc, algo, ws_data, ws_size,
							x4284, grad_in_desc, x2158));
			};
			float* x4287 = (float*)myMalloc(1 * sizeof(float));;
			x4287[0] = 1.0f;

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
							64, 512, x2171, x2171));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2137, x2137));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4287, in_desc, x2150, grad_out_desc, x2183,
							conv_desc, algo, ws_data, ws_size,
							x4287, grad_filt_desc, x1148));
			};
			float* x4290 = (float*)myMalloc(1 * sizeof(float));;
			x4290[0] = 1.0f;
			float* x4292 = (float*)myMalloc(1 * sizeof(float));;
			x4292[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2137, x2137));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4290, x_desc, x2150, x_desc, x2158, x_desc, x2150,
							x4292, x_desc, x2158));
			};
			float* x4295 = (float*)myMalloc(1 * sizeof(float));;
			x4295[0] = 0.0f;
			float* x4297 = (float*)myMalloc(1 * sizeof(float));;
			x4297[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2137, x2137));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2137, x2137));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4297, x4297, x4297, x4297, in_desc, x2143,
							out_desc, x2158, in_desc, x2149, sbmv_desc, x676,
							x1234,x1168, 1.0E-5, x2151, x2152));
			};
			// conv2D back-propagate
			float* x4301 = (float*)myMalloc(1 * sizeof(float));;
			x4301[0] = 1.0f;

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
							64, 128, x2101, x2101));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2137, x2137));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x4301, filt_desc, x868, grad_out_desc, x2149,
							conv_desc, algo, ws_data, ws_size,
							x4301, grad_in_desc, x2122));
			};
			float* x4304 = (float*)myMalloc(1 * sizeof(float));;
			x4304[0] = 1.0f;

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
							64, 128, x2137, x2137));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2101, x2101));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x4304, in_desc, x2114, grad_out_desc, x2149,
							conv_desc, algo, ws_data, ws_size,
							x4304, grad_filt_desc, x1298));
			};
			float* x4307 = (float*)myMalloc(1 * sizeof(float));;
			x4307[0] = 1.0f;
			float* x4309 = (float*)myMalloc(1 * sizeof(float));;
			x4309[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2101, x2101));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4307, x_desc, x2114, x_desc, x2122, x_desc, x2114,
							x4309, x_desc, x2122));
			};
			float* x4312 = (float*)myMalloc(1 * sizeof(float));;
			x4312[0] = 0.0f;
			float* x4314 = (float*)myMalloc(1 * sizeof(float));;
			x4314[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2101, x2101));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2101, x2101));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4314, x4314, x4314, x4314, in_desc, x2107,
							out_desc, x2122, in_desc, x2113, sbmv_desc, x430,
							x1152,x1277, 1.0E-5, x2115, x2116));
			};
			// conv2D back-propagate
			float* x4318 = (float*)myMalloc(1 * sizeof(float));;
			x4318[0] = 1.0f;

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
							64, 512, x2054, x2054));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2101, x2101));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4318, filt_desc, x883, grad_out_desc, x2113,
							conv_desc, algo, ws_data, ws_size,
							x4318, grad_in_desc, x2075));
			};
			float* x4321 = (float*)myMalloc(1 * sizeof(float));;
			x4321[0] = 1.0f;

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
							64, 128, x2101, x2101));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2054, x2054));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4321, in_desc, x2067, grad_out_desc, x2113,
							conv_desc, algo, ws_data, ws_size,
							x4321, grad_filt_desc, x1303));
			};
			float* x4324 = (float*)myMalloc(1 * sizeof(float));;
			x4324[0] = 1.0f;
			float* x4326 = (float*)myMalloc(1 * sizeof(float));;
			x4326[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2054, x2054));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4324, x_desc, x2067, x_desc, x2075, x_desc, x2067,
							x4326, x_desc, x2075));
			};
			if (x4330) {
				if (x4332) {
				} else {
					assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(2054) x Sym(2054), res:  x Const(64) x Const(512) x Sym(1911) x Sym(1911)");
				}
				float* x4337 = (float*)myMalloc(1 * sizeof(float));;
				x4337[0] = 1.0f;
				float* x4339 = (float*)myMalloc(1 * sizeof(float));;
				x4339[0] = 1.0f;

				{
					cudnnTensorDescriptor_t bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 512, x2054, x2054));

					cudnnTensorDescriptor_t out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 512, x1911, x1911));

					CUDNN_CALL(cudnnAddTensor(
								cudnnHandle, x4337, bias_desc, x2075, x4339, out_desc, x1932));
				};
			} else {
				float* x4343 = (float*)myMalloc(1 * sizeof(float));;
				x4343[0] = 1.0f;

				{
					cudnnTensorDescriptor_t grad_bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 512, x1911, x1911));

					cudnnTensorDescriptor_t grad_out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 512, x2054, x2054));

					CUDNN_CALL(cudnnConvolutionBackwardBias(
								cudnnHandle, x4343, grad_out_desc, x2075,
								x4343, grad_bias_desc, x1932));
				};
			}
			float* x4348 = (float*)myMalloc(1 * sizeof(float));;
			x4348[0] = 0.0f;
			float* x4350 = (float*)myMalloc(1 * sizeof(float));;
			x4350[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2054, x2054));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2054, x2054));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4350, x4350, x4350, x4350, in_desc, x2060,
							out_desc, x2075, in_desc, x2066, sbmv_desc, x451,
							x1159,x1353, 1.0E-5, x2068, x2069));
			};
			// conv2D back-propagate
			float* x4354 = (float*)myMalloc(1 * sizeof(float));;
			x4354[0] = 1.0f;

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
							64, 128, x2020, x2020));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x2054, x2054));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4354, filt_desc, x628, grad_out_desc, x2066,
							conv_desc, algo, ws_data, ws_size,
							x4354, grad_in_desc, x2041));
			};
			float* x4357 = (float*)myMalloc(1 * sizeof(float));;
			x4357[0] = 1.0f;

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
							64, 512, x2054, x2054));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2020, x2020));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4357, in_desc, x2033, grad_out_desc, x2066,
							conv_desc, algo, ws_data, ws_size,
							x4357, grad_filt_desc, x1218));
			};
			float* x4360 = (float*)myMalloc(1 * sizeof(float));;
			x4360[0] = 1.0f;
			float* x4362 = (float*)myMalloc(1 * sizeof(float));;
			x4362[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2020, x2020));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4360, x_desc, x2033, x_desc, x2041, x_desc, x2033,
							x4362, x_desc, x2041));
			};
			float* x4365 = (float*)myMalloc(1 * sizeof(float));;
			x4365[0] = 0.0f;
			float* x4367 = (float*)myMalloc(1 * sizeof(float));;
			x4367[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2020, x2020));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2020, x2020));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4367, x4367, x4367, x4367, in_desc, x2026,
							out_desc, x2041, in_desc, x2032, sbmv_desc, x319,
							x1115,x1202, 1.0E-5, x2034, x2035));
			};
			// conv2D back-propagate
			float* x4371 = (float*)myMalloc(1 * sizeof(float));;
			x4371[0] = 1.0f;

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
							64, 128, x1984, x1984));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x2020, x2020));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x4371, filt_desc, x1000, grad_out_desc, x2032,
							conv_desc, algo, ws_data, ws_size,
							x4371, grad_in_desc, x2005));
			};
			float* x4374 = (float*)myMalloc(1 * sizeof(float));;
			x4374[0] = 1.0f;

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
							64, 128, x2020, x2020));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1984, x1984));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x4374, in_desc, x1997, grad_out_desc, x2032,
							conv_desc, algo, ws_data, ws_size,
							x4374, grad_filt_desc, x1342));
			};
			float* x4377 = (float*)myMalloc(1 * sizeof(float));;
			x4377[0] = 1.0f;
			float* x4379 = (float*)myMalloc(1 * sizeof(float));;
			x4379[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1984, x1984));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4377, x_desc, x1997, x_desc, x2005, x_desc, x1997,
							x4379, x_desc, x2005));
			};
			float* x4382 = (float*)myMalloc(1 * sizeof(float));;
			x4382[0] = 0.0f;
			float* x4384 = (float*)myMalloc(1 * sizeof(float));;
			x4384[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1984, x1984));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1984, x1984));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4384, x4384, x4384, x4384, in_desc, x1990,
							out_desc, x2005, in_desc, x1996, sbmv_desc, x961,
							x1329,x1124, 1.0E-5, x1998, x1999));
			};
			// conv2D back-propagate
			float* x4388 = (float*)myMalloc(1 * sizeof(float));;
			x4388[0] = 1.0f;

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
							64, 512, x1911, x1911));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1984, x1984));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4388, filt_desc, x1063, grad_out_desc, x1996,
							conv_desc, algo, ws_data, ws_size,
							x4388, grad_in_desc, x1932));
			};
			float* x4391 = (float*)myMalloc(1 * sizeof(float));;
			x4391[0] = 1.0f;

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
							64, 128, x1984, x1984));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1911, x1911));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4391, in_desc, x1924, grad_out_desc, x1996,
							conv_desc, algo, ws_data, ws_size,
							x4391, grad_filt_desc, x1363));
			};
			float* x4394 = (float*)myMalloc(1 * sizeof(float));;
			x4394[0] = 1.0f;
			float* x4396 = (float*)myMalloc(1 * sizeof(float));;
			x4396[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1911, x1911));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4394, x_desc, x1924, x_desc, x1932, x_desc, x1924,
							x4396, x_desc, x1932));
			};
			if (x4400) {
				if (x4402) {
				} else {
					assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(512) x Sym(1911) x Sym(1911), res:  x Const(64) x Const(512) x Sym(1937) x Sym(1937)");
				}
				float* x4407 = (float*)myMalloc(1 * sizeof(float));;
				x4407[0] = 1.0f;
				float* x4409 = (float*)myMalloc(1 * sizeof(float));;
				x4409[0] = 1.0f;

				{
					cudnnTensorDescriptor_t bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 512, x1911, x1911));

					cudnnTensorDescriptor_t out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 512, x1937, x1937));

					CUDNN_CALL(cudnnAddTensor(
								cudnnHandle, x4407, bias_desc, x1932, x4409, out_desc, x1958));
				};
			} else {
				float* x4413 = (float*)myMalloc(1 * sizeof(float));;
				x4413[0] = 1.0f;

				{
					cudnnTensorDescriptor_t grad_bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 512, x1937, x1937));

					cudnnTensorDescriptor_t grad_out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 512, x1911, x1911));

					CUDNN_CALL(cudnnConvolutionBackwardBias(
								cudnnHandle, x4413, grad_out_desc, x1932,
								x4413, grad_bias_desc, x1958));
				};
			}
			float* x4418 = (float*)myMalloc(1 * sizeof(float));;
			x4418[0] = 0.0f;
			float* x4420 = (float*)myMalloc(1 * sizeof(float));;
			x4420[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1937, x1937));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1937, x1937));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4420, x4420, x4420, x4420, in_desc, x1943,
							out_desc, x1958, in_desc, x1949, sbmv_desc, x916,
							x1314,x1226, 1.0E-5, x1951, x1952));
			};
			// conv2D back-propagate
			float* x4424 = (float*)myMalloc(1 * sizeof(float));;
			x4424[0] = 1.0f;

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
							64, 256, x1794, x1794));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1937, x1937));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 2, 2, 1, 1,
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
							x4424, filt_desc, x1069, grad_out_desc, x1949,
							conv_desc, algo, ws_data, ws_size,
							x4424, grad_in_desc, x1815));
			};
			float* x4427 = (float*)myMalloc(1 * sizeof(float));;
			x4427[0] = 1.0f;

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
							64, 512, x1937, x1937));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1794, x1794));

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
							x4427, in_desc, x1807, grad_out_desc, x1949,
							conv_desc, algo, ws_data, ws_size,
							x4427, grad_filt_desc, x1365));
			};
			float* x4430 = (float*)myMalloc(1 * sizeof(float));;
			x4430[0] = 0.0f;
			float* x4432 = (float*)myMalloc(1 * sizeof(float));;
			x4432[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1911, x1911));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1911, x1911));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 512, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4432, x4432, x4432, x4432, in_desc, x1917,
							out_desc, x1932, in_desc, x1923, sbmv_desc, x730,
							x1252,x1317, 1.0E-5, x1925, x1926));
			};
			// conv2D back-propagate
			float* x4436 = (float*)myMalloc(1 * sizeof(float));;
			x4436[0] = 1.0f;

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
							64, 128, x1877, x1877));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1911, x1911));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4436, filt_desc, x613, grad_out_desc, x1923,
							conv_desc, algo, ws_data, ws_size,
							x4436, grad_in_desc, x1898));
			};
			float* x4439 = (float*)myMalloc(1 * sizeof(float));;
			x4439[0] = 1.0f;

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
							64, 512, x1911, x1911));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1877, x1877));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4439, in_desc, x1890, grad_out_desc, x1923,
							conv_desc, algo, ws_data, ws_size,
							x4439, grad_filt_desc, x1213));
			};
			float* x4442 = (float*)myMalloc(1 * sizeof(float));;
			x4442[0] = 1.0f;
			float* x4444 = (float*)myMalloc(1 * sizeof(float));;
			x4444[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1877, x1877));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4442, x_desc, x1890, x_desc, x1898, x_desc, x1890,
							x4444, x_desc, x1898));
			};
			float* x4447 = (float*)myMalloc(1 * sizeof(float));;
			x4447[0] = 0.0f;
			float* x4449 = (float*)myMalloc(1 * sizeof(float));;
			x4449[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1877, x1877));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1877, x1877));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4449, x4449, x4449, x4449, in_desc, x1883,
							out_desc, x1898, in_desc, x1889, sbmv_desc, x1051,
							x1359,x1297, 1.0E-5, x1891, x1892));
			};
			// conv2D back-propagate
			float* x4453 = (float*)myMalloc(1 * sizeof(float));;
			x4453[0] = 1.0f;

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
							64, 128, x1841, x1841));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1877, x1877));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 2, 2, 1, 1,
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
							x4453, filt_desc, x376, grad_out_desc, x1889,
							conv_desc, algo, ws_data, ws_size,
							x4453, grad_in_desc, x1862));
			};
			float* x4456 = (float*)myMalloc(1 * sizeof(float));;
			x4456[0] = 1.0f;

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
							64, 128, x1877, x1877));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1841, x1841));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 2, 2, 1, 1,
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
							x4456, in_desc, x1854, grad_out_desc, x1889,
							conv_desc, algo, ws_data, ws_size,
							x4456, grad_filt_desc, x1134));
			};
			float* x4459 = (float*)myMalloc(1 * sizeof(float));;
			x4459[0] = 1.0f;
			float* x4461 = (float*)myMalloc(1 * sizeof(float));;
			x4461[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1841, x1841));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4459, x_desc, x1854, x_desc, x1862, x_desc, x1854,
							x4461, x_desc, x1862));
			};
			float* x4464 = (float*)myMalloc(1 * sizeof(float));;
			x4464[0] = 0.0f;
			float* x4466 = (float*)myMalloc(1 * sizeof(float));;
			x4466[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1841, x1841));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1841, x1841));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4466, x4466, x4466, x4466, in_desc, x1847,
							out_desc, x1862, in_desc, x1853, sbmv_desc, x547,
							x1191,x1279, 1.0E-5, x1855, x1856));
			};
			// conv2D back-propagate
			float* x4470 = (float*)myMalloc(1 * sizeof(float));;
			x4470[0] = 1.0f;

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
							64, 256, x1794, x1794));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x1841, x1841));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4470, filt_desc, x328, grad_out_desc, x1853,
							conv_desc, algo, ws_data, ws_size,
							x4470, grad_in_desc, x1815));
			};
			float* x4473 = (float*)myMalloc(1 * sizeof(float));;
			x4473[0] = 1.0f;

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
							64, 128, x1841, x1841));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1794, x1794));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4473, in_desc, x1807, grad_out_desc, x1853,
							conv_desc, algo, ws_data, ws_size,
							x4473, grad_filt_desc, x1118));
			};
			float* x4476 = (float*)myMalloc(1 * sizeof(float));;
			x4476[0] = 1.0f;
			float* x4478 = (float*)myMalloc(1 * sizeof(float));;
			x4478[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1794, x1794));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4476, x_desc, x1807, x_desc, x1815, x_desc, x1807,
							x4478, x_desc, x1815));
			};
			if (x4482) {
				if (x4485) {
				} else {
					assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1794) x Sym(1794), res:  x Const(64) x Const(256) x Sym(1677) x Sym(1677)");
				}
				float* x4490 = (float*)myMalloc(1 * sizeof(float));;
				x4490[0] = 1.0f;
				float* x4492 = (float*)myMalloc(1 * sizeof(float));;
				x4492[0] = 1.0f;

				{
					cudnnTensorDescriptor_t bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 256, x1794, x1794));

					cudnnTensorDescriptor_t out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 256, x1677, x1677));

					CUDNN_CALL(cudnnAddTensor(
								cudnnHandle, x4490, bias_desc, x1815, x4492, out_desc, x1698));
				};
			} else {
				float* x4496 = (float*)myMalloc(1 * sizeof(float));;
				x4496[0] = 1.0f;

				{
					cudnnTensorDescriptor_t grad_bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 256, x1677, x1677));

					cudnnTensorDescriptor_t grad_out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 256, x1794, x1794));

					CUDNN_CALL(cudnnConvolutionBackwardBias(
								cudnnHandle, x4496, grad_out_desc, x1815,
								x4496, grad_bias_desc, x1698));
				};
			}
			float* x4501 = (float*)myMalloc(1 * sizeof(float));;
			x4501[0] = 0.0f;
			float* x4503 = (float*)myMalloc(1 * sizeof(float));;
			x4503[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1794, x1794));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1794, x1794));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4503, x4503, x4503, x4503, in_desc, x1800,
							out_desc, x1815, in_desc, x1806, sbmv_desc, x406,
							x1144,x1354, 1.0E-5, x1808, x1809));
			};
			// conv2D back-propagate
			float* x4507 = (float*)myMalloc(1 * sizeof(float));;
			x4507[0] = 1.0f;

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
							64, 64, x1760, x1760));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1794, x1794));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4507, filt_desc, x556, grad_out_desc, x1806,
							conv_desc, algo, ws_data, ws_size,
							x4507, grad_in_desc, x1781));
			};
			float* x4510 = (float*)myMalloc(1 * sizeof(float));;
			x4510[0] = 1.0f;

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
							64, 256, x1794, x1794));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1760, x1760));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4510, in_desc, x1773, grad_out_desc, x1806,
							conv_desc, algo, ws_data, ws_size,
							x4510, grad_filt_desc, x1194));
			};
			float* x4513 = (float*)myMalloc(1 * sizeof(float));;
			x4513[0] = 1.0f;
			float* x4515 = (float*)myMalloc(1 * sizeof(float));;
			x4515[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1760, x1760));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4513, x_desc, x1773, x_desc, x1781, x_desc, x1773,
							x4515, x_desc, x1781));
			};
			float* x4518 = (float*)myMalloc(1 * sizeof(float));;
			x4518[0] = 0.0f;
			float* x4520 = (float*)myMalloc(1 * sizeof(float));;
			x4520[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1760, x1760));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1760, x1760));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4520, x4520, x4520, x4520, in_desc, x1766,
							out_desc, x1781, in_desc, x1772, sbmv_desc, x511,
							x1179,x1242, 1.0E-5, x1774, x1775));
			};
			// conv2D back-propagate
			float* x4524 = (float*)myMalloc(1 * sizeof(float));;
			x4524[0] = 1.0f;

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
							64, 64, x1724, x1724));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1760, x1760));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x4524, filt_desc, x514, grad_out_desc, x1772,
							conv_desc, algo, ws_data, ws_size,
							x4524, grad_in_desc, x1745));
			};
			float* x4527 = (float*)myMalloc(1 * sizeof(float));;
			x4527[0] = 1.0f;

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
							64, 64, x1760, x1760));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1724, x1724));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x4527, in_desc, x1737, grad_out_desc, x1772,
							conv_desc, algo, ws_data, ws_size,
							x4527, grad_filt_desc, x1180));
			};
			float* x4530 = (float*)myMalloc(1 * sizeof(float));;
			x4530[0] = 1.0f;
			float* x4532 = (float*)myMalloc(1 * sizeof(float));;
			x4532[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1724, x1724));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4530, x_desc, x1737, x_desc, x1745, x_desc, x1737,
							x4532, x_desc, x1745));
			};
			float* x4535 = (float*)myMalloc(1 * sizeof(float));;
			x4535[0] = 0.0f;
			float* x4537 = (float*)myMalloc(1 * sizeof(float));;
			x4537[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1724, x1724));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1724, x1724));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4537, x4537, x4537, x4537, in_desc, x1730,
							out_desc, x1745, in_desc, x1736, sbmv_desc, x538,
							x1188,x1131, 1.0E-5, x1738, x1739));
			};
			// conv2D back-propagate
			float* x4541 = (float*)myMalloc(1 * sizeof(float));;
			x4541[0] = 1.0f;

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
							64, 256, x1677, x1677));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1724, x1724));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4541, filt_desc, x745, grad_out_desc, x1736,
							conv_desc, algo, ws_data, ws_size,
							x4541, grad_in_desc, x1698));
			};
			float* x4544 = (float*)myMalloc(1 * sizeof(float));;
			x4544[0] = 1.0f;

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
							64, 64, x1724, x1724));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1677, x1677));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4544, in_desc, x1690, grad_out_desc, x1736,
							conv_desc, algo, ws_data, ws_size,
							x4544, grad_filt_desc, x1257));
			};
			float* x4547 = (float*)myMalloc(1 * sizeof(float));;
			x4547[0] = 1.0f;
			float* x4549 = (float*)myMalloc(1 * sizeof(float));;
			x4549[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1677, x1677));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4547, x_desc, x1690, x_desc, x1698, x_desc, x1690,
							x4549, x_desc, x1698));
			};
			if (x4553) {
				if (x4555) {
				} else {
					assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1677) x Sym(1677), res:  x Const(64) x Const(256) x Sym(1537) x Sym(1537)");
				}
				float* x4560 = (float*)myMalloc(1 * sizeof(float));;
				x4560[0] = 1.0f;
				float* x4562 = (float*)myMalloc(1 * sizeof(float));;
				x4562[0] = 1.0f;

				{
					cudnnTensorDescriptor_t bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 256, x1677, x1677));

					cudnnTensorDescriptor_t out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 256, x1537, x1537));

					CUDNN_CALL(cudnnAddTensor(
								cudnnHandle, x4560, bias_desc, x1698, x4562, out_desc, x1558));
				};
			} else {
				float* x4566 = (float*)myMalloc(1 * sizeof(float));;
				x4566[0] = 1.0f;

				{
					cudnnTensorDescriptor_t grad_bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 256, x1537, x1537));

					cudnnTensorDescriptor_t grad_out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 256, x1677, x1677));

					CUDNN_CALL(cudnnConvolutionBackwardBias(
								cudnnHandle, x4566, grad_out_desc, x1698,
								x4566, grad_bias_desc, x1558));
				};
			}
			float* x4571 = (float*)myMalloc(1 * sizeof(float));;
			x4571[0] = 0.0f;
			float* x4573 = (float*)myMalloc(1 * sizeof(float));;
			x4573[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1677, x1677));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1677, x1677));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4573, x4573, x4573, x4573, in_desc, x1683,
							out_desc, x1698, in_desc, x1689, sbmv_desc, x469,
							x1165,x1114, 1.0E-5, x1691, x1692));
			};
			// conv2D back-propagate
			float* x4577 = (float*)myMalloc(1 * sizeof(float));;
			x4577[0] = 1.0f;

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
							64, 64, x1643, x1643));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1677, x1677));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4577, filt_desc, x685, grad_out_desc, x1689,
							conv_desc, algo, ws_data, ws_size,
							x4577, grad_in_desc, x1664));
			};
			float* x4580 = (float*)myMalloc(1 * sizeof(float));;
			x4580[0] = 1.0f;

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
							64, 256, x1677, x1677));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1643, x1643));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4580, in_desc, x1656, grad_out_desc, x1689,
							conv_desc, algo, ws_data, ws_size,
							x4580, grad_filt_desc, x1237));
			};
			float* x4583 = (float*)myMalloc(1 * sizeof(float));;
			x4583[0] = 1.0f;
			float* x4585 = (float*)myMalloc(1 * sizeof(float));;
			x4585[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1643, x1643));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4583, x_desc, x1656, x_desc, x1664, x_desc, x1656,
							x4585, x_desc, x1664));
			};
			float* x4588 = (float*)myMalloc(1 * sizeof(float));;
			x4588[0] = 0.0f;
			float* x4590 = (float*)myMalloc(1 * sizeof(float));;
			x4590[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1643, x1643));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1643, x1643));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4590, x4590, x4590, x4590, in_desc, x1649,
							out_desc, x1664, in_desc, x1655, sbmv_desc, x919,
							x1315,x1260, 1.0E-5, x1657, x1658));
			};
			// conv2D back-propagate
			float* x4594 = (float*)myMalloc(1 * sizeof(float));;
			x4594[0] = 1.0f;

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
							64, 64, x1607, x1607));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1643, x1643));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x4594, filt_desc, x544, grad_out_desc, x1655,
							conv_desc, algo, ws_data, ws_size,
							x4594, grad_in_desc, x1628));
			};
			float* x4597 = (float*)myMalloc(1 * sizeof(float));;
			x4597[0] = 1.0f;

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
							64, 64, x1643, x1643));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1607, x1607));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x4597, in_desc, x1620, grad_out_desc, x1655,
							conv_desc, algo, ws_data, ws_size,
							x4597, grad_filt_desc, x1190));
			};
			float* x4600 = (float*)myMalloc(1 * sizeof(float));;
			x4600[0] = 1.0f;
			float* x4602 = (float*)myMalloc(1 * sizeof(float));;
			x4602[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1607, x1607));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4600, x_desc, x1620, x_desc, x1628, x_desc, x1620,
							x4602, x_desc, x1628));
			};
			float* x4605 = (float*)myMalloc(1 * sizeof(float));;
			x4605[0] = 0.0f;
			float* x4607 = (float*)myMalloc(1 * sizeof(float));;
			x4607[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1607, x1607));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1607, x1607));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4607, x4607, x4607, x4607, in_desc, x1613,
							out_desc, x1628, in_desc, x1619, sbmv_desc, x721,
							x1249,x1167, 1.0E-5, x1621, x1622));
			};
			// conv2D back-propagate
			float* x4611 = (float*)myMalloc(1 * sizeof(float));;
			x4611[0] = 1.0f;

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
							64, 256, x1537, x1537));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1607, x1607));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4611, filt_desc, x808, grad_out_desc, x1619,
							conv_desc, algo, ws_data, ws_size,
							x4611, grad_in_desc, x1558));
			};
			float* x4614 = (float*)myMalloc(1 * sizeof(float));;
			x4614[0] = 1.0f;

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
							64, 64, x1607, x1607));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1537, x1537));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4614, in_desc, x1550, grad_out_desc, x1619,
							conv_desc, algo, ws_data, ws_size,
							x4614, grad_filt_desc, x1278));
			};
			float* x4617 = (float*)myMalloc(1 * sizeof(float));;
			x4617[0] = 1.0f;
			float* x4619 = (float*)myMalloc(1 * sizeof(float));;
			x4619[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1537, x1537));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4617, x_desc, x1550, x_desc, x1558, x_desc, x1550,
							x4619, x_desc, x1558));
			};
			if (x4623) {
				if (x4625) {
				} else {
					assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(64) x Const(256) x Sym(1537) x Sym(1537), res:  x Const(64) x Const(256) x Sym(1467) x Sym(1467)");
				}
				float* x4630 = (float*)myMalloc(1 * sizeof(float));;
				x4630[0] = 1.0f;
				float* x4632 = (float*)myMalloc(1 * sizeof(float));;
				x4632[0] = 1.0f;

				{
					cudnnTensorDescriptor_t bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 256, x1537, x1537));

					cudnnTensorDescriptor_t out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 256, x1467, x1467));

					CUDNN_CALL(cudnnAddTensor(
								cudnnHandle, x4630, bias_desc, x1558, x4632, out_desc, x1581));
				};
			} else {
				float* x4636 = (float*)myMalloc(1 * sizeof(float));;
				x4636[0] = 1.0f;

				{
					cudnnTensorDescriptor_t grad_bias_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 256, x1467, x1467));

					cudnnTensorDescriptor_t grad_out_desc;
					CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
					CUDNN_CALL(cudnnSetTensor4dDescriptor(
								grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
								64, 256, x1537, x1537));

					CUDNN_CALL(cudnnConvolutionBackwardBias(
								cudnnHandle, x4636, grad_out_desc, x1558,
								x4636, grad_bias_desc, x1581));
				};
			}
			float* x4641 = (float*)myMalloc(1 * sizeof(float));;
			x4641[0] = 0.0f;
			float* x4643 = (float*)myMalloc(1 * sizeof(float));;
			x4643[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1467, x1467));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1467, x1467));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4643, x4643, x4643, x4643, in_desc, x1566,
							out_desc, x1581, in_desc, x1572, sbmv_desc, x523,
							x1183,x1310, 1.0E-5, x1574, x1575));
			};
			// conv2D back-propagate
			float* x4647 = (float*)myMalloc(1 * sizeof(float));;
			x4647[0] = 1.0f;

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
							64, 64, x1451, x1451));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1467, x1467));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4647, filt_desc, x781, grad_out_desc, x1572,
							conv_desc, algo, ws_data, ws_size,
							x4647, grad_in_desc, x1459));
			};
			float* x4650 = (float*)myMalloc(1 * sizeof(float));;
			x4650[0] = 1.0f;

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
							64, 256, x1467, x1467));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1451, x1451));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4650, in_desc, x1457, grad_out_desc, x1572,
							conv_desc, algo, ws_data, ws_size,
							x4650, grad_filt_desc, x1269));
			};
			float* x4653 = (float*)myMalloc(1 * sizeof(float));;
			x4653[0] = 0.0f;
			float* x4655 = (float*)myMalloc(1 * sizeof(float));;
			x4655[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1537, x1537));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1537, x1537));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4655, x4655, x4655, x4655, in_desc, x1543,
							out_desc, x1558, in_desc, x1549, sbmv_desc, x892,
							x1306,x1233, 1.0E-5, x1551, x1552));
			};
			// conv2D back-propagate
			float* x4659 = (float*)myMalloc(1 * sizeof(float));;
			x4659[0] = 1.0f;

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
							64, 64, x1503, x1503));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1537, x1537));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4659, filt_desc, x391, grad_out_desc, x1549,
							conv_desc, algo, ws_data, ws_size,
							x4659, grad_in_desc, x1524));
			};
			float* x4662 = (float*)myMalloc(1 * sizeof(float));;
			x4662[0] = 1.0f;

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
							64, 256, x1537, x1537));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1503, x1503));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4662, in_desc, x1516, grad_out_desc, x1549,
							conv_desc, algo, ws_data, ws_size,
							x4662, grad_filt_desc, x1139));
			};
			float* x4665 = (float*)myMalloc(1 * sizeof(float));;
			x4665[0] = 1.0f;
			float* x4667 = (float*)myMalloc(1 * sizeof(float));;
			x4667[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1503, x1503));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4665, x_desc, x1516, x_desc, x1524, x_desc, x1516,
							x4667, x_desc, x1524));
			};
			float* x4670 = (float*)myMalloc(1 * sizeof(float));;
			x4670[0] = 0.0f;
			float* x4672 = (float*)myMalloc(1 * sizeof(float));;
			x4672[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1503, x1503));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1503, x1503));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4672, x4672, x4672, x4672, in_desc, x1509,
							out_desc, x1524, in_desc, x1515, sbmv_desc, x787,
							x1271,x1156, 1.0E-5, x1517, x1518));
			};
			// conv2D back-propagate
			float* x4676 = (float*)myMalloc(1 * sizeof(float));;
			x4676[0] = 1.0f;

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
							64, 64, x1467, x1467));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1503, x1503));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x4676, filt_desc, x565, grad_out_desc, x1515,
							conv_desc, algo, ws_data, ws_size,
							x4676, grad_in_desc, x1488));
			};
			float* x4679 = (float*)myMalloc(1 * sizeof(float));;
			x4679[0] = 1.0f;

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
							64, 64, x1503, x1503));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1467, x1467));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							1, 1, 1, 1, 1, 1,
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
							x4679, in_desc, x1480, grad_out_desc, x1515,
							conv_desc, algo, ws_data, ws_size,
							x4679, grad_filt_desc, x1197));
			};
			float* x4682 = (float*)myMalloc(1 * sizeof(float));;
			x4682[0] = 1.0f;
			float* x4684 = (float*)myMalloc(1 * sizeof(float));;
			x4684[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1467, x1467));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4682, x_desc, x1480, x_desc, x1488, x_desc, x1480,
							x4684, x_desc, x1488));
			};
			float* x4687 = (float*)myMalloc(1 * sizeof(float));;
			x4687[0] = 0.0f;
			float* x4689 = (float*)myMalloc(1 * sizeof(float));;
			x4689[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1467, x1467));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1467, x1467));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4689, x4689, x4689, x4689, in_desc, x1473,
							out_desc, x1488, in_desc, x1479, sbmv_desc, x373,
							x1133,x1160, 1.0E-5, x1481, x1482));
			};
			// conv2D back-propagate
			float* x4693 = (float*)myMalloc(1 * sizeof(float));;
			x4693[0] = 1.0f;

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
							64, 64, x1451, x1451));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1467, x1467));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4693, filt_desc, x994, grad_out_desc, x1479,
							conv_desc, algo, ws_data, ws_size,
							x4693, grad_in_desc, x1459));
			};
			float* x4696 = (float*)myMalloc(1 * sizeof(float));;
			x4696[0] = 1.0f;

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
							64, 64, x1467, x1467));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1451, x1451));

				cudnnConvolutionDescriptor_t conv_desc;
				CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
				CUDNN_CALL(cudnnSetConvolution2dDescriptor(
							conv_desc,
							0, 0, 1, 1, 1, 1,
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
							x4696, in_desc, x1457, grad_out_desc, x1479,
							conv_desc, algo, ws_data, ws_size,
							x4696, grad_filt_desc, x1340));
			};
			float* x4699 = (float*)myMalloc(1 * sizeof(float));;
			x4699[0] = 0.0f;
			float* x4701 = (float*)myMalloc(1 * sizeof(float));;
			x4701[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1418, x1418));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1451, x1451));

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
							x4701, out_desc, x1457, out_desc, x1459, in_desc, x1431  , x4699, in_desc, x1439));
			};
			float* x4704 = (float*)myMalloc(1 * sizeof(float));;
			x4704[0] = 1.0f;
			float* x4706 = (float*)myMalloc(1 * sizeof(float));;
			x4706[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1418, x1418));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x4704, x_desc, x1431, x_desc, x1439, x_desc, x1431,
							x4706, x_desc, x1439));
			};
			float* x4709 = (float*)myMalloc(1 * sizeof(float));;
			x4709[0] = 0.0f;
			float* x4711 = (float*)myMalloc(1 * sizeof(float));;
			x4711[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1418, x1418));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1418, x1418));

				cudnnTensorDescriptor_t sbmv_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				CUDNN_CALL(cudnnBatchNormalizationBackward(
							cudnnHandle, CUDNN_BATCHNORM_SPATIAL,
							x4711, x4711, x4711, x4711, in_desc, x1424,
							out_desc, x1439, in_desc, x1430, sbmv_desc, x913,
							x1313,x1358, 1.0E-5, x1432, x1433));
			};
			// conv2D back-propagate
			float* x4715 = (float*)myMalloc(1 * sizeof(float));;
			x4715[0] = 1.0f;

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
							64, 64, x1418, x1418));

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
							x4715, in_desc, x1402, grad_out_desc, x1430,
							conv_desc, algo, ws_data, ws_size,
							x4715, grad_filt_desc, x1259));
			};
			float x4718 = x1410[0];
			x1390 += x4718;
			float* x4720 = (float*)myMalloc(1 * sizeof(float));;
			x4720[0] = 1.0f;
			float* x4722 = (float*)myMalloc(1 * sizeof(float));;
			x4722[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,256,x4720,x313,1024,x4722, x1113, 1024, x313,1024));
			arrayFill_greg<<<28, 512>>>(x1113, 0.0f, 262144);
			float* x4726 = (float*)myMalloc(1 * sizeof(float));;
			x4726[0] = 1.0f;
			float* x4728 = (float*)myMalloc(1 * sizeof(float));;
			x4728[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4726,x316,1,x4728, x1114, 1, x316,1));
			arrayFill_greg<<<28, 512>>>(x1114, 0.0f, 256);
			float* x4732 = (float*)myMalloc(1 * sizeof(float));;
			x4732[0] = 1.0f;
			float* x4734 = (float*)myMalloc(1 * sizeof(float));;
			x4734[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4732,x319,1,x4734, x1115, 1, x319,1));
			arrayFill_greg<<<28, 512>>>(x1115, 0.0f, 128);
			float* x4738 = (float*)myMalloc(1 * sizeof(float));;
			x4738[0] = 1.0f;
			float* x4740 = (float*)myMalloc(1 * sizeof(float));;
			x4740[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4738,x322,1,x4740, x1116, 1, x322,1));
			arrayFill_greg<<<28, 512>>>(x1116, 0.0f, 128);
			float* x4744 = (float*)myMalloc(1 * sizeof(float));;
			x4744[0] = 1.0f;
			float* x4746 = (float*)myMalloc(1 * sizeof(float));;
			x4746[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4744,x325,1,x4746, x1117, 1, x325,1));
			arrayFill_greg<<<28, 512>>>(x1117, 0.0f, 64);
			float* x4750 = (float*)myMalloc(1 * sizeof(float));;
			x4750[0] = 1.0f;
			float* x4752 = (float*)myMalloc(1 * sizeof(float));;
			x4752[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,128,x4750,x328,256,x4752, x1118, 256, x328,256));
			arrayFill_greg<<<28, 512>>>(x1118, 0.0f, 32768);
			float* x4756 = (float*)myMalloc(1 * sizeof(float));;
			x4756[0] = 1.0f;
			float* x4758 = (float*)myMalloc(1 * sizeof(float));;
			x4758[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4756,x331,1,x4758, x1119, 1, x331,1));
			arrayFill_greg<<<28, 512>>>(x1119, 0.0f, 512);
			float* x4762 = (float*)myMalloc(1 * sizeof(float));;
			x4762[0] = 1.0f;
			float* x4764 = (float*)myMalloc(1 * sizeof(float));;
			x4764[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,256,x4762,x334,1024,x4764, x1120, 1024, x334,1024));
			arrayFill_greg<<<28, 512>>>(x1120, 0.0f, 262144);
			float* x4768 = (float*)myMalloc(1 * sizeof(float));;
			x4768[0] = 1.0f;
			float* x4770 = (float*)myMalloc(1 * sizeof(float));;
			x4770[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x4768,x337,2304,x4770, x1121, 2304, x337,2304));
			arrayFill_greg<<<28, 512>>>(x1121, 0.0f, 589824);
			float* x4774 = (float*)myMalloc(1 * sizeof(float));;
			x4774[0] = 1.0f;
			float* x4776 = (float*)myMalloc(1 * sizeof(float));;
			x4776[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4774,x340,1,x4776, x1122, 1, x340,1));
			arrayFill_greg<<<28, 512>>>(x1122, 0.0f, 512);
			float* x4780 = (float*)myMalloc(1 * sizeof(float));;
			x4780[0] = 1.0f;
			float* x4782 = (float*)myMalloc(1 * sizeof(float));;
			x4782[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4780,x343,1,x4782, x1123, 1, x343,1));
			arrayFill_greg<<<28, 512>>>(x1123, 0.0f, 256);
			float* x4786 = (float*)myMalloc(1 * sizeof(float));;
			x4786[0] = 1.0f;
			float* x4788 = (float*)myMalloc(1 * sizeof(float));;
			x4788[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4786,x346,1,x4788, x1124, 1, x346,1));
			arrayFill_greg<<<28, 512>>>(x1124, 0.0f, 128);
			float* x4792 = (float*)myMalloc(1 * sizeof(float));;
			x4792[0] = 1.0f;
			float* x4794 = (float*)myMalloc(1 * sizeof(float));;
			x4794[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4792,x349,1,x4794, x1125, 1, x349,1));
			arrayFill_greg<<<28, 512>>>(x1125, 0.0f, 1024);
			float* x4798 = (float*)myMalloc(1 * sizeof(float));;
			x4798[0] = 1.0f;
			float* x4800 = (float*)myMalloc(1 * sizeof(float));;
			x4800[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4798,x352,1,x4800, x1126, 1, x352,1));
			arrayFill_greg<<<28, 512>>>(x1126, 0.0f, 512);
			float* x4804 = (float*)myMalloc(1 * sizeof(float));;
			x4804[0] = 1.0f;
			float* x4806 = (float*)myMalloc(1 * sizeof(float));;
			x4806[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4804,x355,1,x4806, x1127, 1, x355,1));
			arrayFill_greg<<<28, 512>>>(x1127, 0.0f, 1024);
			float* x4810 = (float*)myMalloc(1 * sizeof(float));;
			x4810[0] = 1.0f;
			float* x4812 = (float*)myMalloc(1 * sizeof(float));;
			x4812[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4810,x358,1,x4812, x1128, 1, x358,1));
			arrayFill_greg<<<28, 512>>>(x1128, 0.0f, 256);
			float* x4816 = (float*)myMalloc(1 * sizeof(float));;
			x4816[0] = 1.0f;
			float* x4818 = (float*)myMalloc(1 * sizeof(float));;
			x4818[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,256,x4816,x361,1024,x4818, x1129, 1024, x361,1024));
			arrayFill_greg<<<28, 512>>>(x1129, 0.0f, 262144);
			float* x4822 = (float*)myMalloc(1 * sizeof(float));;
			x4822[0] = 1.0f;
			float* x4824 = (float*)myMalloc(1 * sizeof(float));;
			x4824[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4822,x364,1,x4824, x1130, 1, x364,1));
			arrayFill_greg<<<28, 512>>>(x1130, 0.0f, 512);
			float* x4828 = (float*)myMalloc(1 * sizeof(float));;
			x4828[0] = 1.0f;
			float* x4830 = (float*)myMalloc(1 * sizeof(float));;
			x4830[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4828,x367,1,x4830, x1131, 1, x367,1));
			arrayFill_greg<<<28, 512>>>(x1131, 0.0f, 64);
			float* x4834 = (float*)myMalloc(1 * sizeof(float));;
			x4834[0] = 1.0f;
			float* x4836 = (float*)myMalloc(1 * sizeof(float));;
			x4836[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4834,x370,1,x4836, x1132, 1, x370,1));
			arrayFill_greg<<<28, 512>>>(x1132, 0.0f, 512);
			float* x4840 = (float*)myMalloc(1 * sizeof(float));;
			x4840[0] = 1.0f;
			float* x4842 = (float*)myMalloc(1 * sizeof(float));;
			x4842[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4840,x373,1,x4842, x1133, 1, x373,1));
			arrayFill_greg<<<28, 512>>>(x1133, 0.0f, 64);
			float* x4846 = (float*)myMalloc(1 * sizeof(float));;
			x4846[0] = 1.0f;
			float* x4848 = (float*)myMalloc(1 * sizeof(float));;
			x4848[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1152,128,x4846,x376,1152,x4848, x1134, 1152, x376,1152));
			arrayFill_greg<<<28, 512>>>(x1134, 0.0f, 147456);
			float* x4852 = (float*)myMalloc(1 * sizeof(float));;
			x4852[0] = 1.0f;
			float* x4854 = (float*)myMalloc(1 * sizeof(float));;
			x4854[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 4608,512,x4852,x379,4608,x4854, x1135, 4608, x379,4608));
			arrayFill_greg<<<28, 512>>>(x1135, 0.0f, 2359296);
			float* x4858 = (float*)myMalloc(1 * sizeof(float));;
			x4858[0] = 1.0f;
			float* x4860 = (float*)myMalloc(1 * sizeof(float));;
			x4860[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x4858,x382,1,x4860, x1136, 1, x382,1));
			arrayFill_greg<<<28, 512>>>(x1136, 0.0f, 1024);
			float* x4864 = (float*)myMalloc(1 * sizeof(float));;
			x4864[0] = 1.0f;
			float* x4866 = (float*)myMalloc(1 * sizeof(float));;
			x4866[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4864,x385,1,x4866, x1137, 1, x385,1));
			arrayFill_greg<<<28, 512>>>(x1137, 0.0f, 256);
			float* x4870 = (float*)myMalloc(1 * sizeof(float));;
			x4870[0] = 1.0f;
			float* x4872 = (float*)myMalloc(1 * sizeof(float));;
			x4872[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x4870,x388,2304,x4872, x1138, 2304, x388,2304));
			arrayFill_greg<<<28, 512>>>(x1138, 0.0f, 589824);
			float* x4876 = (float*)myMalloc(1 * sizeof(float));;
			x4876[0] = 1.0f;
			float* x4878 = (float*)myMalloc(1 * sizeof(float));;
			x4878[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x4876,x391,64,x4878, x1139, 64, x391,64));
			arrayFill_greg<<<28, 512>>>(x1139, 0.0f, 16384);
			float* x4882 = (float*)myMalloc(1 * sizeof(float));;
			x4882[0] = 1.0f;
			float* x4884 = (float*)myMalloc(1 * sizeof(float));;
			x4884[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,2048,x4882,x394,512,x4884, x1140, 512, x394,512));
			arrayFill_greg<<<28, 512>>>(x1140, 0.0f, 1048576);
			float* x4888 = (float*)myMalloc(1 * sizeof(float));;
			x4888[0] = 1.0f;
			float* x4890 = (float*)myMalloc(1 * sizeof(float));;
			x4890[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 4608,512,x4888,x397,4608,x4890, x1141, 4608, x397,4608));
			arrayFill_greg<<<28, 512>>>(x1141, 0.0f, 2359296);
			float* x4894 = (float*)myMalloc(1 * sizeof(float));;
			x4894[0] = 1.0f;
			float* x4896 = (float*)myMalloc(1 * sizeof(float));;
			x4896[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4894,x400,1,x4896, x1142, 1, x400,1));
			arrayFill_greg<<<28, 512>>>(x1142, 0.0f, 128);
			float* x4900 = (float*)myMalloc(1 * sizeof(float));;
			x4900[0] = 1.0f;
			float* x4902 = (float*)myMalloc(1 * sizeof(float));;
			x4902[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4900,x403,1,x4902, x1143, 1, x403,1));
			arrayFill_greg<<<28, 512>>>(x1143, 0.0f, 256);
			float* x4906 = (float*)myMalloc(1 * sizeof(float));;
			x4906[0] = 1.0f;
			float* x4908 = (float*)myMalloc(1 * sizeof(float));;
			x4908[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x4906,x406,1,x4908, x1144, 1, x406,1));
			arrayFill_greg<<<28, 512>>>(x1144, 0.0f, 256);
			float* x4912 = (float*)myMalloc(1 * sizeof(float));;
			x4912[0] = 1.0f;
			float* x4914 = (float*)myMalloc(1 * sizeof(float));;
			x4914[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4912,x409,1,x4914, x1145, 1, x409,1));
			arrayFill_greg<<<28, 512>>>(x1145, 0.0f, 128);
			float* x4918 = (float*)myMalloc(1 * sizeof(float));;
			x4918[0] = 1.0f;
			float* x4920 = (float*)myMalloc(1 * sizeof(float));;
			x4920[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4918,x412,1,x4920, x1146, 1, x412,1));
			arrayFill_greg<<<28, 512>>>(x1146, 0.0f, 128);
			float* x4924 = (float*)myMalloc(1 * sizeof(float));;
			x4924[0] = 1.0f;
			float* x4926 = (float*)myMalloc(1 * sizeof(float));;
			x4926[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4924,x415,1,x4926, x1147, 1, x415,1));
			arrayFill_greg<<<28, 512>>>(x1147, 0.0f, 64);
			float* x4930 = (float*)myMalloc(1 * sizeof(float));;
			x4930[0] = 1.0f;
			float* x4932 = (float*)myMalloc(1 * sizeof(float));;
			x4932[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,512,x4930,x418,128,x4932, x1148, 128, x418,128));
			arrayFill_greg<<<28, 512>>>(x1148, 0.0f, 65536);
			float* x4936 = (float*)myMalloc(1 * sizeof(float));;
			x4936[0] = 1.0f;
			float* x4938 = (float*)myMalloc(1 * sizeof(float));;
			x4938[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4936,x421,1,x4938, x1149, 1, x421,1));
			arrayFill_greg<<<28, 512>>>(x1149, 0.0f, 512);
			float* x4942 = (float*)myMalloc(1 * sizeof(float));;
			x4942[0] = 1.0f;
			float* x4944 = (float*)myMalloc(1 * sizeof(float));;
			x4944[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4942,x424,1,x4944, x1150, 1, x424,1));
			arrayFill_greg<<<28, 512>>>(x1150, 0.0f, 128);
			float* x4948 = (float*)myMalloc(1 * sizeof(float));;
			x4948[0] = 1.0f;
			float* x4950 = (float*)myMalloc(1 * sizeof(float));;
			x4950[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4948,x427,1,x4950, x1151, 1, x427,1));
			arrayFill_greg<<<28, 512>>>(x1151, 0.0f, 64);
			float* x4954 = (float*)myMalloc(1 * sizeof(float));;
			x4954[0] = 1.0f;
			float* x4956 = (float*)myMalloc(1 * sizeof(float));;
			x4956[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x4954,x430,1,x4956, x1152, 1, x430,1));
			arrayFill_greg<<<28, 512>>>(x1152, 0.0f, 128);
			float* x4960 = (float*)myMalloc(1 * sizeof(float));;
			x4960[0] = 1.0f;
			float* x4962 = (float*)myMalloc(1 * sizeof(float));;
			x4962[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4960,x433,1,x4962, x1153, 1, x433,1));
			arrayFill_greg<<<28, 512>>>(x1153, 0.0f, 512);
			float* x4966 = (float*)myMalloc(1 * sizeof(float));;
			x4966[0] = 1.0f;
			float* x4968 = (float*)myMalloc(1 * sizeof(float));;
			x4968[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,2048,x4966,x436,512,x4968, x1154, 512, x436,512));
			arrayFill_greg<<<28, 512>>>(x1154, 0.0f, 1048576);
			float* x4972 = (float*)myMalloc(1 * sizeof(float));;
			x4972[0] = 1.0f;
			float* x4974 = (float*)myMalloc(1 * sizeof(float));;
			x4974[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,10,x4972,x439,1,x4974, x1155, 1, x439,1));
			arrayFill_greg<<<28, 512>>>(x1155, 0.0f, 10);
			float* x4978 = (float*)myMalloc(1 * sizeof(float));;
			x4978[0] = 1.0f;
			float* x4980 = (float*)myMalloc(1 * sizeof(float));;
			x4980[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4978,x442,1,x4980, x1156, 1, x442,1));
			arrayFill_greg<<<28, 512>>>(x1156, 0.0f, 64);
			float* x4984 = (float*)myMalloc(1 * sizeof(float));;
			x4984[0] = 1.0f;
			float* x4986 = (float*)myMalloc(1 * sizeof(float));;
			x4986[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4984,x445,1,x4986, x1157, 1, x445,1));
			arrayFill_greg<<<28, 512>>>(x1157, 0.0f, 512);
			float* x4990 = (float*)myMalloc(1 * sizeof(float));;
			x4990[0] = 1.0f;
			float* x4992 = (float*)myMalloc(1 * sizeof(float));;
			x4992[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x4990,x448,1,x4992, x1158, 1, x448,1));
			arrayFill_greg<<<28, 512>>>(x1158, 0.0f, 64);
			float* x4996 = (float*)myMalloc(1 * sizeof(float));;
			x4996[0] = 1.0f;
			float* x4998 = (float*)myMalloc(1 * sizeof(float));;
			x4998[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x4996,x451,1,x4998, x1159, 1, x451,1));
			arrayFill_greg<<<28, 512>>>(x1159, 0.0f, 512);
			float* x5002 = (float*)myMalloc(1 * sizeof(float));;
			x5002[0] = 1.0f;
			float* x5004 = (float*)myMalloc(1 * sizeof(float));;
			x5004[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5002,x454,1,x5004, x1160, 1, x454,1));
			arrayFill_greg<<<28, 512>>>(x1160, 0.0f, 64);
			float* x5008 = (float*)myMalloc(1 * sizeof(float));;
			x5008[0] = 1.0f;
			float* x5010 = (float*)myMalloc(1 * sizeof(float));;
			x5010[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5008,x457,1,x5010, x1161, 1, x457,1));
			arrayFill_greg<<<28, 512>>>(x1161, 0.0f, 512);
			float* x5014 = (float*)myMalloc(1 * sizeof(float));;
			x5014[0] = 1.0f;
			float* x5016 = (float*)myMalloc(1 * sizeof(float));;
			x5016[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,512,x5014,x460,128,x5016, x1162, 128, x460,128));
			arrayFill_greg<<<28, 512>>>(x1162, 0.0f, 65536);
			float* x5020 = (float*)myMalloc(1 * sizeof(float));;
			x5020[0] = 1.0f;
			float* x5022 = (float*)myMalloc(1 * sizeof(float));;
			x5022[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x5020,x463,256,x5022, x1163, 256, x463,256));
			arrayFill_greg<<<28, 512>>>(x1163, 0.0f, 262144);
			float* x5026 = (float*)myMalloc(1 * sizeof(float));;
			x5026[0] = 1.0f;
			float* x5028 = (float*)myMalloc(1 * sizeof(float));;
			x5028[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5026,x466,1,x5028, x1164, 1, x466,1));
			arrayFill_greg<<<28, 512>>>(x1164, 0.0f, 1024);
			float* x5032 = (float*)myMalloc(1 * sizeof(float));;
			x5032[0] = 1.0f;
			float* x5034 = (float*)myMalloc(1 * sizeof(float));;
			x5034[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5032,x469,1,x5034, x1165, 1, x469,1));
			arrayFill_greg<<<28, 512>>>(x1165, 0.0f, 256);
			float* x5038 = (float*)myMalloc(1 * sizeof(float));;
			x5038[0] = 1.0f;
			float* x5040 = (float*)myMalloc(1 * sizeof(float));;
			x5040[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5038,x472,1,x5040, x1166, 1, x472,1));
			arrayFill_greg<<<28, 512>>>(x1166, 0.0f, 1024);
			float* x5044 = (float*)myMalloc(1 * sizeof(float));;
			x5044[0] = 1.0f;
			float* x5046 = (float*)myMalloc(1 * sizeof(float));;
			x5046[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5044,x475,1,x5046, x1167, 1, x475,1));
			arrayFill_greg<<<28, 512>>>(x1167, 0.0f, 64);
			float* x5050 = (float*)myMalloc(1 * sizeof(float));;
			x5050[0] = 1.0f;
			float* x5052 = (float*)myMalloc(1 * sizeof(float));;
			x5052[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5050,x478,1,x5052, x1168, 1, x478,1));
			arrayFill_greg<<<28, 512>>>(x1168, 0.0f, 128);
			float* x5056 = (float*)myMalloc(1 * sizeof(float));;
			x5056[0] = 1.0f;
			float* x5058 = (float*)myMalloc(1 * sizeof(float));;
			x5058[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5056,x481,1,x5058, x1169, 1, x481,1));
			arrayFill_greg<<<28, 512>>>(x1169, 0.0f, 2048);
			float* x5062 = (float*)myMalloc(1 * sizeof(float));;
			x5062[0] = 1.0f;
			float* x5064 = (float*)myMalloc(1 * sizeof(float));;
			x5064[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5062,x484,1,x5064, x1170, 1, x484,1));
			arrayFill_greg<<<28, 512>>>(x1170, 0.0f, 256);
			float* x5068 = (float*)myMalloc(1 * sizeof(float));;
			x5068[0] = 1.0f;
			float* x5070 = (float*)myMalloc(1 * sizeof(float));;
			x5070[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5068,x487,1,x5070, x1171, 1, x487,1));
			arrayFill_greg<<<28, 512>>>(x1171, 0.0f, 2048);
			float* x5074 = (float*)myMalloc(1 * sizeof(float));;
			x5074[0] = 1.0f;
			float* x5076 = (float*)myMalloc(1 * sizeof(float));;
			x5076[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5074,x490,1,x5076, x1172, 1, x490,1));
			arrayFill_greg<<<28, 512>>>(x1172, 0.0f, 512);
			float* x5080 = (float*)myMalloc(1 * sizeof(float));;
			x5080[0] = 1.0f;
			float* x5082 = (float*)myMalloc(1 * sizeof(float));;
			x5082[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5080,x493,1,x5082, x1173, 1, x493,1));
			arrayFill_greg<<<28, 512>>>(x1173, 0.0f, 512);
			float* x5086 = (float*)myMalloc(1 * sizeof(float));;
			x5086[0] = 1.0f;
			float* x5088 = (float*)myMalloc(1 * sizeof(float));;
			x5088[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5086,x496,1,x5088, x1174, 1, x496,1));
			arrayFill_greg<<<28, 512>>>(x1174, 0.0f, 512);
			float* x5092 = (float*)myMalloc(1 * sizeof(float));;
			x5092[0] = 1.0f;
			float* x5094 = (float*)myMalloc(1 * sizeof(float));;
			x5094[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5092,x499,1,x5094, x1175, 1, x499,1));
			arrayFill_greg<<<28, 512>>>(x1175, 0.0f, 2048);
			float* x5098 = (float*)myMalloc(1 * sizeof(float));;
			x5098[0] = 1.0f;
			float* x5100 = (float*)myMalloc(1 * sizeof(float));;
			x5100[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5098,x502,1,x5100, x1176, 1, x502,1));
			arrayFill_greg<<<28, 512>>>(x1176, 0.0f, 256);
			float* x5104 = (float*)myMalloc(1 * sizeof(float));;
			x5104[0] = 1.0f;
			float* x5106 = (float*)myMalloc(1 * sizeof(float));;
			x5106[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5104,x505,1,x5106, x1177, 1, x505,1));
			arrayFill_greg<<<28, 512>>>(x1177, 0.0f, 256);
			float* x5110 = (float*)myMalloc(1 * sizeof(float));;
			x5110[0] = 1.0f;
			float* x5112 = (float*)myMalloc(1 * sizeof(float));;
			x5112[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5110,x508,1,x5112, x1178, 1, x508,1));
			arrayFill_greg<<<28, 512>>>(x1178, 0.0f, 256);
			float* x5116 = (float*)myMalloc(1 * sizeof(float));;
			x5116[0] = 1.0f;
			float* x5118 = (float*)myMalloc(1 * sizeof(float));;
			x5118[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5116,x511,1,x5118, x1179, 1, x511,1));
			arrayFill_greg<<<28, 512>>>(x1179, 0.0f, 64);
			float* x5122 = (float*)myMalloc(1 * sizeof(float));;
			x5122[0] = 1.0f;
			float* x5124 = (float*)myMalloc(1 * sizeof(float));;
			x5124[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 576,64,x5122,x514,576,x5124, x1180, 576, x514,576));
			arrayFill_greg<<<28, 512>>>(x1180, 0.0f, 36864);
			float* x5128 = (float*)myMalloc(1 * sizeof(float));;
			x5128[0] = 1.0f;
			float* x5130 = (float*)myMalloc(1 * sizeof(float));;
			x5130[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5128,x517,1,x5130, x1181, 1, x517,1));
			arrayFill_greg<<<28, 512>>>(x1181, 0.0f, 256);
			float* x5134 = (float*)myMalloc(1 * sizeof(float));;
			x5134[0] = 1.0f;
			float* x5136 = (float*)myMalloc(1 * sizeof(float));;
			x5136[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,1024,x5134,x520,512,x5136, x1182, 512, x520,512));
			arrayFill_greg<<<28, 512>>>(x1182, 0.0f, 524288);
			float* x5140 = (float*)myMalloc(1 * sizeof(float));;
			x5140[0] = 1.0f;
			float* x5142 = (float*)myMalloc(1 * sizeof(float));;
			x5142[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5140,x523,1,x5142, x1183, 1, x523,1));
			arrayFill_greg<<<28, 512>>>(x1183, 0.0f, 256);
			float* x5146 = (float*)myMalloc(1 * sizeof(float));;
			x5146[0] = 1.0f;
			float* x5148 = (float*)myMalloc(1 * sizeof(float));;
			x5148[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5146,x526,1,x5148, x1184, 1, x526,1));
			arrayFill_greg<<<28, 512>>>(x1184, 0.0f, 256);
			float* x5152 = (float*)myMalloc(1 * sizeof(float));;
			x5152[0] = 1.0f;
			float* x5154 = (float*)myMalloc(1 * sizeof(float));;
			x5154[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5152,x529,1,x5154, x1185, 1, x529,1));
			arrayFill_greg<<<28, 512>>>(x1185, 0.0f, 512);
			float* x5158 = (float*)myMalloc(1 * sizeof(float));;
			x5158[0] = 1.0f;
			float* x5160 = (float*)myMalloc(1 * sizeof(float));;
			x5160[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5158,x532,1,x5160, x1186, 1, x532,1));
			arrayFill_greg<<<28, 512>>>(x1186, 0.0f, 128);
			float* x5164 = (float*)myMalloc(1 * sizeof(float));;
			x5164[0] = 1.0f;
			float* x5166 = (float*)myMalloc(1 * sizeof(float));;
			x5166[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5164,x535,1,x5166, x1187, 1, x535,1));
			arrayFill_greg<<<28, 512>>>(x1187, 0.0f, 256);
			float* x5170 = (float*)myMalloc(1 * sizeof(float));;
			x5170[0] = 1.0f;
			float* x5172 = (float*)myMalloc(1 * sizeof(float));;
			x5172[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5170,x538,1,x5172, x1188, 1, x538,1));
			arrayFill_greg<<<28, 512>>>(x1188, 0.0f, 64);
			float* x5176 = (float*)myMalloc(1 * sizeof(float));;
			x5176[0] = 1.0f;
			float* x5178 = (float*)myMalloc(1 * sizeof(float));;
			x5178[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5176,x541,1,x5178, x1189, 1, x541,1));
			arrayFill_greg<<<28, 512>>>(x1189, 0.0f, 512);
			float* x5182 = (float*)myMalloc(1 * sizeof(float));;
			x5182[0] = 1.0f;
			float* x5184 = (float*)myMalloc(1 * sizeof(float));;
			x5184[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 576,64,x5182,x544,576,x5184, x1190, 576, x544,576));
			arrayFill_greg<<<28, 512>>>(x1190, 0.0f, 36864);
			float* x5188 = (float*)myMalloc(1 * sizeof(float));;
			x5188[0] = 1.0f;
			float* x5190 = (float*)myMalloc(1 * sizeof(float));;
			x5190[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5188,x547,1,x5190, x1191, 1, x547,1));
			arrayFill_greg<<<28, 512>>>(x1191, 0.0f, 128);
			float* x5194 = (float*)myMalloc(1 * sizeof(float));;
			x5194[0] = 1.0f;
			float* x5196 = (float*)myMalloc(1 * sizeof(float));;
			x5196[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5194,x550,1,x5196, x1192, 1, x550,1));
			arrayFill_greg<<<28, 512>>>(x1192, 0.0f, 256);
			float* x5200 = (float*)myMalloc(1 * sizeof(float));;
			x5200[0] = 1.0f;
			float* x5202 = (float*)myMalloc(1 * sizeof(float));;
			x5202[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5200,x553,1,x5202, x1193, 1, x553,1));
			arrayFill_greg<<<28, 512>>>(x1193, 0.0f, 1024);
			float* x5206 = (float*)myMalloc(1 * sizeof(float));;
			x5206[0] = 1.0f;
			float* x5208 = (float*)myMalloc(1 * sizeof(float));;
			x5208[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x5206,x556,64,x5208, x1194, 64, x556,64));
			arrayFill_greg<<<28, 512>>>(x1194, 0.0f, 16384);
			float* x5212 = (float*)myMalloc(1 * sizeof(float));;
			x5212[0] = 1.0f;
			float* x5214 = (float*)myMalloc(1 * sizeof(float));;
			x5214[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5212,x559,1,x5214, x1195, 1, x559,1));
			arrayFill_greg<<<28, 512>>>(x1195, 0.0f, 512);
			float* x5218 = (float*)myMalloc(1 * sizeof(float));;
			x5218[0] = 1.0f;
			float* x5220 = (float*)myMalloc(1 * sizeof(float));;
			x5220[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x5218,x562,256,x5220, x1196, 256, x562,256));
			arrayFill_greg<<<28, 512>>>(x1196, 0.0f, 262144);
			float* x5224 = (float*)myMalloc(1 * sizeof(float));;
			x5224[0] = 1.0f;
			float* x5226 = (float*)myMalloc(1 * sizeof(float));;
			x5226[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 576,64,x5224,x565,576,x5226, x1197, 576, x565,576));
			arrayFill_greg<<<28, 512>>>(x1197, 0.0f, 36864);
			float* x5230 = (float*)myMalloc(1 * sizeof(float));;
			x5230[0] = 1.0f;
			float* x5232 = (float*)myMalloc(1 * sizeof(float));;
			x5232[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5230,x568,1,x5232, x1198, 1, x568,1));
			arrayFill_greg<<<28, 512>>>(x1198, 0.0f, 256);
			float* x5236 = (float*)myMalloc(1 * sizeof(float));;
			x5236[0] = 1.0f;
			float* x5238 = (float*)myMalloc(1 * sizeof(float));;
			x5238[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5236,x571,1,x5238, x1199, 1, x571,1));
			arrayFill_greg<<<28, 512>>>(x1199, 0.0f, 256);
			float* x5242 = (float*)myMalloc(1 * sizeof(float));;
			x5242[0] = 1.0f;
			float* x5244 = (float*)myMalloc(1 * sizeof(float));;
			x5244[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5242,x574,1,x5244, x1200, 1, x574,1));
			arrayFill_greg<<<28, 512>>>(x1200, 0.0f, 1024);
			float* x5248 = (float*)myMalloc(1 * sizeof(float));;
			x5248[0] = 1.0f;
			float* x5250 = (float*)myMalloc(1 * sizeof(float));;
			x5250[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5248,x577,1,x5250, x1201, 1, x577,1));
			arrayFill_greg<<<28, 512>>>(x1201, 0.0f, 2048);
			float* x5254 = (float*)myMalloc(1 * sizeof(float));;
			x5254[0] = 1.0f;
			float* x5256 = (float*)myMalloc(1 * sizeof(float));;
			x5256[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5254,x580,1,x5256, x1202, 1, x580,1));
			arrayFill_greg<<<28, 512>>>(x1202, 0.0f, 128);
			float* x5260 = (float*)myMalloc(1 * sizeof(float));;
			x5260[0] = 1.0f;
			float* x5262 = (float*)myMalloc(1 * sizeof(float));;
			x5262[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5260,x583,1,x5262, x1203, 1, x583,1));
			arrayFill_greg<<<28, 512>>>(x1203, 0.0f, 256);
			float* x5266 = (float*)myMalloc(1 * sizeof(float));;
			x5266[0] = 1.0f;
			float* x5268 = (float*)myMalloc(1 * sizeof(float));;
			x5268[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x5266,x586,256,x5268, x1204, 256, x586,256));
			arrayFill_greg<<<28, 512>>>(x1204, 0.0f, 262144);
			float* x5272 = (float*)myMalloc(1 * sizeof(float));;
			x5272[0] = 1.0f;
			float* x5274 = (float*)myMalloc(1 * sizeof(float));;
			x5274[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5272,x589,1,x5274, x1205, 1, x589,1));
			arrayFill_greg<<<28, 512>>>(x1205, 0.0f, 256);
			float* x5278 = (float*)myMalloc(1 * sizeof(float));;
			x5278[0] = 1.0f;
			float* x5280 = (float*)myMalloc(1 * sizeof(float));;
			x5280[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5278,x592,1,x5280, x1206, 1, x592,1));
			arrayFill_greg<<<28, 512>>>(x1206, 0.0f, 256);
			float* x5284 = (float*)myMalloc(1 * sizeof(float));;
			x5284[0] = 1.0f;
			float* x5286 = (float*)myMalloc(1 * sizeof(float));;
			x5286[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5284,x595,1,x5286, x1207, 1, x595,1));
			arrayFill_greg<<<28, 512>>>(x1207, 0.0f, 128);
			float* x5290 = (float*)myMalloc(1 * sizeof(float));;
			x5290[0] = 1.0f;
			float* x5292 = (float*)myMalloc(1 * sizeof(float));;
			x5292[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5290,x598,1,x5292, x1208, 1, x598,1));
			arrayFill_greg<<<28, 512>>>(x1208, 0.0f, 512);
			float* x5296 = (float*)myMalloc(1 * sizeof(float));;
			x5296[0] = 1.0f;
			float* x5298 = (float*)myMalloc(1 * sizeof(float));;
			x5298[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5296,x601,1,x5298, x1209, 1, x601,1));
			arrayFill_greg<<<28, 512>>>(x1209, 0.0f, 64);
			float* x5302 = (float*)myMalloc(1 * sizeof(float));;
			x5302[0] = 1.0f;
			float* x5304 = (float*)myMalloc(1 * sizeof(float));;
			x5304[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5302,x604,1,x5304, x1210, 1, x604,1));
			arrayFill_greg<<<28, 512>>>(x1210, 0.0f, 2048);
			float* x5308 = (float*)myMalloc(1 * sizeof(float));;
			x5308[0] = 1.0f;
			float* x5310 = (float*)myMalloc(1 * sizeof(float));;
			x5310[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5308,x607,1,x5310, x1211, 1, x607,1));
			arrayFill_greg<<<28, 512>>>(x1211, 0.0f, 256);
			float* x5314 = (float*)myMalloc(1 * sizeof(float));;
			x5314[0] = 1.0f;
			float* x5316 = (float*)myMalloc(1 * sizeof(float));;
			x5316[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5314,x610,1,x5316, x1212, 1, x610,1));
			arrayFill_greg<<<28, 512>>>(x1212, 0.0f, 64);
			float* x5320 = (float*)myMalloc(1 * sizeof(float));;
			x5320[0] = 1.0f;
			float* x5322 = (float*)myMalloc(1 * sizeof(float));;
			x5322[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,512,x5320,x613,128,x5322, x1213, 128, x613,128));
			arrayFill_greg<<<28, 512>>>(x1213, 0.0f, 65536);
			float* x5326 = (float*)myMalloc(1 * sizeof(float));;
			x5326[0] = 1.0f;
			float* x5328 = (float*)myMalloc(1 * sizeof(float));;
			x5328[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5326,x616,1,x5328, x1214, 1, x616,1));
			arrayFill_greg<<<28, 512>>>(x1214, 0.0f, 2048);
			float* x5332 = (float*)myMalloc(1 * sizeof(float));;
			x5332[0] = 1.0f;
			float* x5334 = (float*)myMalloc(1 * sizeof(float));;
			x5334[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5332,x619,1,x5334, x1215, 1, x619,1));
			arrayFill_greg<<<28, 512>>>(x1215, 0.0f, 256);
			float* x5338 = (float*)myMalloc(1 * sizeof(float));;
			x5338[0] = 1.0f;
			float* x5340 = (float*)myMalloc(1 * sizeof(float));;
			x5340[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5338,x622,1,x5340, x1216, 1, x622,1));
			arrayFill_greg<<<28, 512>>>(x1216, 0.0f, 256);
			float* x5344 = (float*)myMalloc(1 * sizeof(float));;
			x5344[0] = 1.0f;
			float* x5346 = (float*)myMalloc(1 * sizeof(float));;
			x5346[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5344,x625,1,x5346, x1217, 1, x625,1));
			arrayFill_greg<<<28, 512>>>(x1217, 0.0f, 64);
			float* x5350 = (float*)myMalloc(1 * sizeof(float));;
			x5350[0] = 1.0f;
			float* x5352 = (float*)myMalloc(1 * sizeof(float));;
			x5352[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,512,x5350,x628,128,x5352, x1218, 128, x628,128));
			arrayFill_greg<<<28, 512>>>(x1218, 0.0f, 65536);
			float* x5356 = (float*)myMalloc(1 * sizeof(float));;
			x5356[0] = 1.0f;
			float* x5358 = (float*)myMalloc(1 * sizeof(float));;
			x5358[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5356,x631,1,x5358, x1219, 1, x631,1));
			arrayFill_greg<<<28, 512>>>(x1219, 0.0f, 128);
			float* x5362 = (float*)myMalloc(1 * sizeof(float));;
			x5362[0] = 1.0f;
			float* x5364 = (float*)myMalloc(1 * sizeof(float));;
			x5364[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5362,x634,1,x5364, x1220, 1, x634,1));
			arrayFill_greg<<<28, 512>>>(x1220, 0.0f, 512);
			float* x5368 = (float*)myMalloc(1 * sizeof(float));;
			x5368[0] = 1.0f;
			float* x5370 = (float*)myMalloc(1 * sizeof(float));;
			x5370[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5368,x637,1,x5370, x1221, 1, x637,1));
			arrayFill_greg<<<28, 512>>>(x1221, 0.0f, 64);
			float* x5374 = (float*)myMalloc(1 * sizeof(float));;
			x5374[0] = 1.0f;
			float* x5376 = (float*)myMalloc(1 * sizeof(float));;
			x5376[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5374,x640,1,x5376, x1222, 1, x640,1));
			arrayFill_greg<<<28, 512>>>(x1222, 0.0f, 2048);
			float* x5380 = (float*)myMalloc(1 * sizeof(float));;
			x5380[0] = 1.0f;
			float* x5382 = (float*)myMalloc(1 * sizeof(float));;
			x5382[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x5380,x643,256,x5382, x1223, 256, x643,256));
			arrayFill_greg<<<28, 512>>>(x1223, 0.0f, 262144);
			float* x5386 = (float*)myMalloc(1 * sizeof(float));;
			x5386[0] = 1.0f;
			float* x5388 = (float*)myMalloc(1 * sizeof(float));;
			x5388[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5386,x646,1,x5388, x1224, 1, x646,1));
			arrayFill_greg<<<28, 512>>>(x1224, 0.0f, 1024);
			float* x5392 = (float*)myMalloc(1 * sizeof(float));;
			x5392[0] = 1.0f;
			float* x5394 = (float*)myMalloc(1 * sizeof(float));;
			x5394[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5392,x649,1,x5394, x1225, 1, x649,1));
			arrayFill_greg<<<28, 512>>>(x1225, 0.0f, 64);
			float* x5398 = (float*)myMalloc(1 * sizeof(float));;
			x5398[0] = 1.0f;
			float* x5400 = (float*)myMalloc(1 * sizeof(float));;
			x5400[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5398,x652,1,x5400, x1226, 1, x652,1));
			arrayFill_greg<<<28, 512>>>(x1226, 0.0f, 512);
			float* x5404 = (float*)myMalloc(1 * sizeof(float));;
			x5404[0] = 1.0f;
			float* x5406 = (float*)myMalloc(1 * sizeof(float));;
			x5406[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5404,x655,1,x5406, x1227, 1, x655,1));
			arrayFill_greg<<<28, 512>>>(x1227, 0.0f, 1024);
			float* x5410 = (float*)myMalloc(1 * sizeof(float));;
			x5410[0] = 1.0f;
			float* x5412 = (float*)myMalloc(1 * sizeof(float));;
			x5412[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5410,x658,1,x5412, x1228, 1, x658,1));
			arrayFill_greg<<<28, 512>>>(x1228, 0.0f, 512);
			float* x5416 = (float*)myMalloc(1 * sizeof(float));;
			x5416[0] = 1.0f;
			float* x5418 = (float*)myMalloc(1 * sizeof(float));;
			x5418[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5416,x661,1,x5418, x1229, 1, x661,1));
			arrayFill_greg<<<28, 512>>>(x1229, 0.0f, 1024);
			float* x5422 = (float*)myMalloc(1 * sizeof(float));;
			x5422[0] = 1.0f;
			float* x5424 = (float*)myMalloc(1 * sizeof(float));;
			x5424[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5422,x664,1,x5424, x1230, 1, x664,1));
			arrayFill_greg<<<28, 512>>>(x1230, 0.0f, 2048);
			float* x5428 = (float*)myMalloc(1 * sizeof(float));;
			x5428[0] = 1.0f;
			float* x5430 = (float*)myMalloc(1 * sizeof(float));;
			x5430[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5428,x667,1,x5430, x1231, 1, x667,1));
			arrayFill_greg<<<28, 512>>>(x1231, 0.0f, 256);
			float* x5434 = (float*)myMalloc(1 * sizeof(float));;
			x5434[0] = 1.0f;
			float* x5436 = (float*)myMalloc(1 * sizeof(float));;
			x5436[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5434,x670,1,x5436, x1232, 1, x670,1));
			arrayFill_greg<<<28, 512>>>(x1232, 0.0f, 2048);
			float* x5440 = (float*)myMalloc(1 * sizeof(float));;
			x5440[0] = 1.0f;
			float* x5442 = (float*)myMalloc(1 * sizeof(float));;
			x5442[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5440,x673,1,x5442, x1233, 1, x673,1));
			arrayFill_greg<<<28, 512>>>(x1233, 0.0f, 256);
			float* x5446 = (float*)myMalloc(1 * sizeof(float));;
			x5446[0] = 1.0f;
			float* x5448 = (float*)myMalloc(1 * sizeof(float));;
			x5448[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5446,x676,1,x5448, x1234, 1, x676,1));
			arrayFill_greg<<<28, 512>>>(x1234, 0.0f, 128);
			float* x5452 = (float*)myMalloc(1 * sizeof(float));;
			x5452[0] = 1.0f;
			float* x5454 = (float*)myMalloc(1 * sizeof(float));;
			x5454[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5452,x679,1,x5454, x1235, 1, x679,1));
			arrayFill_greg<<<28, 512>>>(x1235, 0.0f, 128);
			float* x5458 = (float*)myMalloc(1 * sizeof(float));;
			x5458[0] = 1.0f;
			float* x5460 = (float*)myMalloc(1 * sizeof(float));;
			x5460[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5458,x682,1,x5460, x1236, 1, x682,1));
			arrayFill_greg<<<28, 512>>>(x1236, 0.0f, 256);
			float* x5464 = (float*)myMalloc(1 * sizeof(float));;
			x5464[0] = 1.0f;
			float* x5466 = (float*)myMalloc(1 * sizeof(float));;
			x5466[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x5464,x685,64,x5466, x1237, 64, x685,64));
			arrayFill_greg<<<28, 512>>>(x1237, 0.0f, 16384);
			float* x5470 = (float*)myMalloc(1 * sizeof(float));;
			x5470[0] = 1.0f;
			float* x5472 = (float*)myMalloc(1 * sizeof(float));;
			x5472[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5470,x688,1,x5472, x1238, 1, x688,1));
			arrayFill_greg<<<28, 512>>>(x1238, 0.0f, 256);
			float* x5476 = (float*)myMalloc(1 * sizeof(float));;
			x5476[0] = 1.0f;
			float* x5478 = (float*)myMalloc(1 * sizeof(float));;
			x5478[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,128,x5476,x691,512,x5478, x1239, 512, x691,512));
			arrayFill_greg<<<28, 512>>>(x1239, 0.0f, 65536);
			float* x5482 = (float*)myMalloc(1 * sizeof(float));;
			x5482[0] = 1.0f;
			float* x5484 = (float*)myMalloc(1 * sizeof(float));;
			x5484[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5482,x694,1,x5484, x1240, 1, x694,1));
			arrayFill_greg<<<28, 512>>>(x1240, 0.0f, 256);
			float* x5488 = (float*)myMalloc(1 * sizeof(float));;
			x5488[0] = 1.0f;
			float* x5490 = (float*)myMalloc(1 * sizeof(float));;
			x5490[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5488,x697,1,x5490, x1241, 1, x697,1));
			arrayFill_greg<<<28, 512>>>(x1241, 0.0f, 128);
			float* x5494 = (float*)myMalloc(1 * sizeof(float));;
			x5494[0] = 1.0f;
			float* x5496 = (float*)myMalloc(1 * sizeof(float));;
			x5496[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5494,x700,1,x5496, x1242, 1, x700,1));
			arrayFill_greg<<<28, 512>>>(x1242, 0.0f, 64);
			float* x5500 = (float*)myMalloc(1 * sizeof(float));;
			x5500[0] = 1.0f;
			float* x5502 = (float*)myMalloc(1 * sizeof(float));;
			x5502[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5500,x703,1,x5502, x1243, 1, x703,1));
			arrayFill_greg<<<28, 512>>>(x1243, 0.0f, 256);
			float* x5506 = (float*)myMalloc(1 * sizeof(float));;
			x5506[0] = 1.0f;
			float* x5508 = (float*)myMalloc(1 * sizeof(float));;
			x5508[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5506,x706,1,x5508, x1244, 1, x706,1));
			arrayFill_greg<<<28, 512>>>(x1244, 0.0f, 512);
			float* x5512 = (float*)myMalloc(1 * sizeof(float));;
			x5512[0] = 1.0f;
			float* x5514 = (float*)myMalloc(1 * sizeof(float));;
			x5514[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5512,x709,1,x5514, x1245, 1, x709,1));
			arrayFill_greg<<<28, 512>>>(x1245, 0.0f, 512);
			float* x5518 = (float*)myMalloc(1 * sizeof(float));;
			x5518[0] = 1.0f;
			float* x5520 = (float*)myMalloc(1 * sizeof(float));;
			x5520[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,512,x5518,x712,1024,x5520, x1246, 1024, x712,1024));
			arrayFill_greg<<<28, 512>>>(x1246, 0.0f, 524288);
			float* x5524 = (float*)myMalloc(1 * sizeof(float));;
			x5524[0] = 1.0f;
			float* x5526 = (float*)myMalloc(1 * sizeof(float));;
			x5526[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5524,x715,1,x5526, x1247, 1, x715,1));
			arrayFill_greg<<<28, 512>>>(x1247, 0.0f, 1024);
			float* x5530 = (float*)myMalloc(1 * sizeof(float));;
			x5530[0] = 1.0f;
			float* x5532 = (float*)myMalloc(1 * sizeof(float));;
			x5532[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5530,x718,1,x5532, x1248, 1, x718,1));
			arrayFill_greg<<<28, 512>>>(x1248, 0.0f, 256);
			float* x5536 = (float*)myMalloc(1 * sizeof(float));;
			x5536[0] = 1.0f;
			float* x5538 = (float*)myMalloc(1 * sizeof(float));;
			x5538[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5536,x721,1,x5538, x1249, 1, x721,1));
			arrayFill_greg<<<28, 512>>>(x1249, 0.0f, 64);
			float* x5542 = (float*)myMalloc(1 * sizeof(float));;
			x5542[0] = 1.0f;
			float* x5544 = (float*)myMalloc(1 * sizeof(float));;
			x5544[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5542,x724,1,x5544, x1250, 1, x724,1));
			arrayFill_greg<<<28, 512>>>(x1250, 0.0f, 1024);
			float* x5548 = (float*)myMalloc(1 * sizeof(float));;
			x5548[0] = 1.0f;
			float* x5550 = (float*)myMalloc(1 * sizeof(float));;
			x5550[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5548,x727,1,x5550, x1251, 1, x727,1));
			arrayFill_greg<<<28, 512>>>(x1251, 0.0f, 2048);
			float* x5554 = (float*)myMalloc(1 * sizeof(float));;
			x5554[0] = 1.0f;
			float* x5556 = (float*)myMalloc(1 * sizeof(float));;
			x5556[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5554,x730,1,x5556, x1252, 1, x730,1));
			arrayFill_greg<<<28, 512>>>(x1252, 0.0f, 512);
			float* x5560 = (float*)myMalloc(1 * sizeof(float));;
			x5560[0] = 1.0f;
			float* x5562 = (float*)myMalloc(1 * sizeof(float));;
			x5562[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5560,x733,1,x5562, x1253, 1, x733,1));
			arrayFill_greg<<<28, 512>>>(x1253, 0.0f, 1024);
			float* x5566 = (float*)myMalloc(1 * sizeof(float));;
			x5566[0] = 1.0f;
			float* x5568 = (float*)myMalloc(1 * sizeof(float));;
			x5568[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5566,x736,1,x5568, x1254, 1, x736,1));
			arrayFill_greg<<<28, 512>>>(x1254, 0.0f, 512);
			float* x5572 = (float*)myMalloc(1 * sizeof(float));;
			x5572[0] = 1.0f;
			float* x5574 = (float*)myMalloc(1 * sizeof(float));;
			x5574[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5572,x739,1,x5574, x1255, 1, x739,1));
			arrayFill_greg<<<28, 512>>>(x1255, 0.0f, 128);
			float* x5578 = (float*)myMalloc(1 * sizeof(float));;
			x5578[0] = 1.0f;
			float* x5580 = (float*)myMalloc(1 * sizeof(float));;
			x5580[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5578,x742,1,x5580, x1256, 1, x742,1));
			arrayFill_greg<<<28, 512>>>(x1256, 0.0f, 512);
			float* x5584 = (float*)myMalloc(1 * sizeof(float));;
			x5584[0] = 1.0f;
			float* x5586 = (float*)myMalloc(1 * sizeof(float));;
			x5586[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,64,x5584,x745,256,x5586, x1257, 256, x745,256));
			arrayFill_greg<<<28, 512>>>(x1257, 0.0f, 16384);
			float* x5590 = (float*)myMalloc(1 * sizeof(float));;
			x5590[0] = 1.0f;
			float* x5592 = (float*)myMalloc(1 * sizeof(float));;
			x5592[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,256,x5590,x748,1024,x5592, x1258, 1024, x748,1024));
			arrayFill_greg<<<28, 512>>>(x1258, 0.0f, 262144);
			float* x5596 = (float*)myMalloc(1 * sizeof(float));;
			x5596[0] = 1.0f;
			float* x5598 = (float*)myMalloc(1 * sizeof(float));;
			x5598[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 27,64,x5596,x751,27,x5598, x1259, 27, x751,27));
			arrayFill_greg<<<28, 512>>>(x1259, 0.0f, 1728);
			float* x5602 = (float*)myMalloc(1 * sizeof(float));;
			x5602[0] = 1.0f;
			float* x5604 = (float*)myMalloc(1 * sizeof(float));;
			x5604[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5602,x754,1,x5604, x1260, 1, x754,1));
			arrayFill_greg<<<28, 512>>>(x1260, 0.0f, 64);
			float* x5608 = (float*)myMalloc(1 * sizeof(float));;
			x5608[0] = 1.0f;
			float* x5610 = (float*)myMalloc(1 * sizeof(float));;
			x5610[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5608,x757,1,x5610, x1261, 1, x757,1));
			arrayFill_greg<<<28, 512>>>(x1261, 0.0f, 512);
			float* x5614 = (float*)myMalloc(1 * sizeof(float));;
			x5614[0] = 1.0f;
			float* x5616 = (float*)myMalloc(1 * sizeof(float));;
			x5616[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 4608,512,x5614,x760,4608,x5616, x1262, 4608, x760,4608));
			arrayFill_greg<<<28, 512>>>(x1262, 0.0f, 2359296);
			float* x5620 = (float*)myMalloc(1 * sizeof(float));;
			x5620[0] = 1.0f;
			float* x5622 = (float*)myMalloc(1 * sizeof(float));;
			x5622[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5620,x763,1,x5622, x1263, 1, x763,1));
			arrayFill_greg<<<28, 512>>>(x1263, 0.0f, 512);
			float* x5626 = (float*)myMalloc(1 * sizeof(float));;
			x5626[0] = 1.0f;
			float* x5628 = (float*)myMalloc(1 * sizeof(float));;
			x5628[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5626,x766,1,x5628, x1264, 1, x766,1));
			arrayFill_greg<<<28, 512>>>(x1264, 0.0f, 256);
			float* x5632 = (float*)myMalloc(1 * sizeof(float));;
			x5632[0] = 1.0f;
			float* x5634 = (float*)myMalloc(1 * sizeof(float));;
			x5634[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5632,x769,1,x5634, x1265, 1, x769,1));
			arrayFill_greg<<<28, 512>>>(x1265, 0.0f, 64);
			float* x5638 = (float*)myMalloc(1 * sizeof(float));;
			x5638[0] = 1.0f;
			float* x5640 = (float*)myMalloc(1 * sizeof(float));;
			x5640[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5638,x772,1,x5640, x1266, 1, x772,1));
			arrayFill_greg<<<28, 512>>>(x1266, 0.0f, 512);
			float* x5644 = (float*)myMalloc(1 * sizeof(float));;
			x5644[0] = 1.0f;
			float* x5646 = (float*)myMalloc(1 * sizeof(float));;
			x5646[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5644,x775,1,x5646, x1267, 1, x775,1));
			arrayFill_greg<<<28, 512>>>(x1267, 0.0f, 512);
			float* x5650 = (float*)myMalloc(1 * sizeof(float));;
			x5650[0] = 1.0f;
			float* x5652 = (float*)myMalloc(1 * sizeof(float));;
			x5652[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5650,x778,1,x5652, x1268, 1, x778,1));
			arrayFill_greg<<<28, 512>>>(x1268, 0.0f, 1024);
			float* x5656 = (float*)myMalloc(1 * sizeof(float));;
			x5656[0] = 1.0f;
			float* x5658 = (float*)myMalloc(1 * sizeof(float));;
			x5658[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x5656,x781,64,x5658, x1269, 64, x781,64));
			arrayFill_greg<<<28, 512>>>(x1269, 0.0f, 16384);
			float* x5662 = (float*)myMalloc(1 * sizeof(float));;
			x5662[0] = 1.0f;
			float* x5664 = (float*)myMalloc(1 * sizeof(float));;
			x5664[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5662,x784,1,x5664, x1270, 1, x784,1));
			arrayFill_greg<<<28, 512>>>(x1270, 0.0f, 256);
			float* x5668 = (float*)myMalloc(1 * sizeof(float));;
			x5668[0] = 1.0f;
			float* x5670 = (float*)myMalloc(1 * sizeof(float));;
			x5670[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5668,x787,1,x5670, x1271, 1, x787,1));
			arrayFill_greg<<<28, 512>>>(x1271, 0.0f, 64);
			float* x5674 = (float*)myMalloc(1 * sizeof(float));;
			x5674[0] = 1.0f;
			float* x5676 = (float*)myMalloc(1 * sizeof(float));;
			x5676[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1152,128,x5674,x790,1152,x5676, x1272, 1152, x790,1152));
			arrayFill_greg<<<28, 512>>>(x1272, 0.0f, 147456);
			float* x5680 = (float*)myMalloc(1 * sizeof(float));;
			x5680[0] = 1.0f;
			float* x5682 = (float*)myMalloc(1 * sizeof(float));;
			x5682[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5680,x793,1,x5682, x1273, 1, x793,1));
			arrayFill_greg<<<28, 512>>>(x1273, 0.0f, 256);
			float* x5686 = (float*)myMalloc(1 * sizeof(float));;
			x5686[0] = 1.0f;
			float* x5688 = (float*)myMalloc(1 * sizeof(float));;
			x5688[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5686,x796,1,x5688, x1274, 1, x796,1));
			arrayFill_greg<<<28, 512>>>(x1274, 0.0f, 512);
			float* x5692 = (float*)myMalloc(1 * sizeof(float));;
			x5692[0] = 1.0f;
			float* x5694 = (float*)myMalloc(1 * sizeof(float));;
			x5694[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5692,x799,1,x5694, x1275, 1, x799,1));
			arrayFill_greg<<<28, 512>>>(x1275, 0.0f, 256);
			float* x5698 = (float*)myMalloc(1 * sizeof(float));;
			x5698[0] = 1.0f;
			float* x5700 = (float*)myMalloc(1 * sizeof(float));;
			x5700[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5698,x802,1,x5700, x1276, 1, x802,1));
			arrayFill_greg<<<28, 512>>>(x1276, 0.0f, 512);
			float* x5704 = (float*)myMalloc(1 * sizeof(float));;
			x5704[0] = 1.0f;
			float* x5706 = (float*)myMalloc(1 * sizeof(float));;
			x5706[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5704,x805,1,x5706, x1277, 1, x805,1));
			arrayFill_greg<<<28, 512>>>(x1277, 0.0f, 128);
			float* x5710 = (float*)myMalloc(1 * sizeof(float));;
			x5710[0] = 1.0f;
			float* x5712 = (float*)myMalloc(1 * sizeof(float));;
			x5712[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,64,x5710,x808,256,x5712, x1278, 256, x808,256));
			arrayFill_greg<<<28, 512>>>(x1278, 0.0f, 16384);
			float* x5716 = (float*)myMalloc(1 * sizeof(float));;
			x5716[0] = 1.0f;
			float* x5718 = (float*)myMalloc(1 * sizeof(float));;
			x5718[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5716,x811,1,x5718, x1279, 1, x811,1));
			arrayFill_greg<<<28, 512>>>(x1279, 0.0f, 128);
			float* x5722 = (float*)myMalloc(1 * sizeof(float));;
			x5722[0] = 1.0f;
			float* x5724 = (float*)myMalloc(1 * sizeof(float));;
			x5724[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5722,x814,1,x5724, x1280, 1, x814,1));
			arrayFill_greg<<<28, 512>>>(x1280, 0.0f, 2048);
			float* x5728 = (float*)myMalloc(1 * sizeof(float));;
			x5728[0] = 1.0f;
			float* x5730 = (float*)myMalloc(1 * sizeof(float));;
			x5730[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5728,x817,1,x5730, x1281, 1, x817,1));
			arrayFill_greg<<<28, 512>>>(x1281, 0.0f, 256);
			float* x5734 = (float*)myMalloc(1 * sizeof(float));;
			x5734[0] = 1.0f;
			float* x5736 = (float*)myMalloc(1 * sizeof(float));;
			x5736[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x5734,x820,2304,x5736, x1282, 2304, x820,2304));
			arrayFill_greg<<<28, 512>>>(x1282, 0.0f, 589824);
			float* x5740 = (float*)myMalloc(1 * sizeof(float));;
			x5740[0] = 1.0f;
			float* x5742 = (float*)myMalloc(1 * sizeof(float));;
			x5742[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5740,x823,1,x5742, x1283, 1, x823,1));
			arrayFill_greg<<<28, 512>>>(x1283, 0.0f, 256);
			float* x5746 = (float*)myMalloc(1 * sizeof(float));;
			x5746[0] = 1.0f;
			float* x5748 = (float*)myMalloc(1 * sizeof(float));;
			x5748[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5746,x826,1,x5748, x1284, 1, x826,1));
			arrayFill_greg<<<28, 512>>>(x1284, 0.0f, 128);
			float* x5752 = (float*)myMalloc(1 * sizeof(float));;
			x5752[0] = 1.0f;
			float* x5754 = (float*)myMalloc(1 * sizeof(float));;
			x5754[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5752,x829,1,x5754, x1285, 1, x829,1));
			arrayFill_greg<<<28, 512>>>(x1285, 0.0f, 256);
			float* x5758 = (float*)myMalloc(1 * sizeof(float));;
			x5758[0] = 1.0f;
			float* x5760 = (float*)myMalloc(1 * sizeof(float));;
			x5760[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5758,x832,1,x5760, x1286, 1, x832,1));
			arrayFill_greg<<<28, 512>>>(x1286, 0.0f, 64);
			float* x5764 = (float*)myMalloc(1 * sizeof(float));;
			x5764[0] = 1.0f;
			float* x5766 = (float*)myMalloc(1 * sizeof(float));;
			x5766[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,256,x5764,x835,512,x5766, x1287, 512, x835,512));
			arrayFill_greg<<<28, 512>>>(x1287, 0.0f, 131072);
			float* x5770 = (float*)myMalloc(1 * sizeof(float));;
			x5770[0] = 1.0f;
			float* x5772 = (float*)myMalloc(1 * sizeof(float));;
			x5772[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5770,x838,1,x5772, x1288, 1, x838,1));
			arrayFill_greg<<<28, 512>>>(x1288, 0.0f, 2048);
			float* x5776 = (float*)myMalloc(1 * sizeof(float));;
			x5776[0] = 1.0f;
			float* x5778 = (float*)myMalloc(1 * sizeof(float));;
			x5778[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5776,x841,1,x5778, x1289, 1, x841,1));
			arrayFill_greg<<<28, 512>>>(x1289, 0.0f, 1024);
			float* x5782 = (float*)myMalloc(1 * sizeof(float));;
			x5782[0] = 1.0f;
			float* x5784 = (float*)myMalloc(1 * sizeof(float));;
			x5784[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5782,x844,1,x5784, x1290, 1, x844,1));
			arrayFill_greg<<<28, 512>>>(x1290, 0.0f, 1024);
			float* x5788 = (float*)myMalloc(1 * sizeof(float));;
			x5788[0] = 1.0f;
			float* x5790 = (float*)myMalloc(1 * sizeof(float));;
			x5790[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5788,x847,1,x5790, x1291, 1, x847,1));
			arrayFill_greg<<<28, 512>>>(x1291, 0.0f, 256);
			float* x5794 = (float*)myMalloc(1 * sizeof(float));;
			x5794[0] = 1.0f;
			float* x5796 = (float*)myMalloc(1 * sizeof(float));;
			x5796[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5794,x850,1,x5796, x1292, 1, x850,1));
			arrayFill_greg<<<28, 512>>>(x1292, 0.0f, 256);
			float* x5800 = (float*)myMalloc(1 * sizeof(float));;
			x5800[0] = 1.0f;
			float* x5802 = (float*)myMalloc(1 * sizeof(float));;
			x5802[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5800,x853,1,x5802, x1293, 1, x853,1));
			arrayFill_greg<<<28, 512>>>(x1293, 0.0f, 256);
			float* x5806 = (float*)myMalloc(1 * sizeof(float));;
			x5806[0] = 1.0f;
			float* x5808 = (float*)myMalloc(1 * sizeof(float));;
			x5808[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5806,x856,1,x5808, x1294, 1, x856,1));
			arrayFill_greg<<<28, 512>>>(x1294, 0.0f, 64);
			float* x5812 = (float*)myMalloc(1 * sizeof(float));;
			x5812[0] = 1.0f;
			float* x5814 = (float*)myMalloc(1 * sizeof(float));;
			x5814[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5812,x859,1,x5814, x1295, 1, x859,1));
			arrayFill_greg<<<28, 512>>>(x1295, 0.0f, 1024);
			float* x5818 = (float*)myMalloc(1 * sizeof(float));;
			x5818[0] = 1.0f;
			float* x5820 = (float*)myMalloc(1 * sizeof(float));;
			x5820[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5818,x862,1,x5820, x1296, 1, x862,1));
			arrayFill_greg<<<28, 512>>>(x1296, 0.0f, 256);
			float* x5824 = (float*)myMalloc(1 * sizeof(float));;
			x5824[0] = 1.0f;
			float* x5826 = (float*)myMalloc(1 * sizeof(float));;
			x5826[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5824,x865,1,x5826, x1297, 1, x865,1));
			arrayFill_greg<<<28, 512>>>(x1297, 0.0f, 128);
			float* x5830 = (float*)myMalloc(1 * sizeof(float));;
			x5830[0] = 1.0f;
			float* x5832 = (float*)myMalloc(1 * sizeof(float));;
			x5832[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1152,128,x5830,x868,1152,x5832, x1298, 1152, x868,1152));
			arrayFill_greg<<<28, 512>>>(x1298, 0.0f, 147456);
			float* x5836 = (float*)myMalloc(1 * sizeof(float));;
			x5836[0] = 1.0f;
			float* x5838 = (float*)myMalloc(1 * sizeof(float));;
			x5838[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5836,x871,1,x5838, x1299, 1, x871,1));
			arrayFill_greg<<<28, 512>>>(x1299, 0.0f, 256);
			float* x5842 = (float*)myMalloc(1 * sizeof(float));;
			x5842[0] = 1.0f;
			float* x5844 = (float*)myMalloc(1 * sizeof(float));;
			x5844[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x5842,x874,1,x5844, x1300, 1, x874,1));
			arrayFill_greg<<<28, 512>>>(x1300, 0.0f, 2048);
			float* x5848 = (float*)myMalloc(1 * sizeof(float));;
			x5848[0] = 1.0f;
			float* x5850 = (float*)myMalloc(1 * sizeof(float));;
			x5850[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5848,x877,1,x5850, x1301, 1, x877,1));
			arrayFill_greg<<<28, 512>>>(x1301, 0.0f, 512);
			float* x5854 = (float*)myMalloc(1 * sizeof(float));;
			x5854[0] = 1.0f;
			float* x5856 = (float*)myMalloc(1 * sizeof(float));;
			x5856[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5854,x880,1,x5856, x1302, 1, x880,1));
			arrayFill_greg<<<28, 512>>>(x1302, 0.0f, 512);
			float* x5860 = (float*)myMalloc(1 * sizeof(float));;
			x5860[0] = 1.0f;
			float* x5862 = (float*)myMalloc(1 * sizeof(float));;
			x5862[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,128,x5860,x883,512,x5862, x1303, 512, x883,512));
			arrayFill_greg<<<28, 512>>>(x1303, 0.0f, 65536);
			float* x5866 = (float*)myMalloc(1 * sizeof(float));;
			x5866[0] = 1.0f;
			float* x5868 = (float*)myMalloc(1 * sizeof(float));;
			x5868[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5866,x886,1,x5868, x1304, 1, x886,1));
			arrayFill_greg<<<28, 512>>>(x1304, 0.0f, 256);
			float* x5872 = (float*)myMalloc(1 * sizeof(float));;
			x5872[0] = 1.0f;
			float* x5874 = (float*)myMalloc(1 * sizeof(float));;
			x5874[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5872,x889,1,x5874, x1305, 1, x889,1));
			arrayFill_greg<<<28, 512>>>(x1305, 0.0f, 256);
			float* x5878 = (float*)myMalloc(1 * sizeof(float));;
			x5878[0] = 1.0f;
			float* x5880 = (float*)myMalloc(1 * sizeof(float));;
			x5880[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5878,x892,1,x5880, x1306, 1, x892,1));
			arrayFill_greg<<<28, 512>>>(x1306, 0.0f, 256);
			float* x5884 = (float*)myMalloc(1 * sizeof(float));;
			x5884[0] = 1.0f;
			float* x5886 = (float*)myMalloc(1 * sizeof(float));;
			x5886[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5884,x895,1,x5886, x1307, 1, x895,1));
			arrayFill_greg<<<28, 512>>>(x1307, 0.0f, 256);
			float* x5890 = (float*)myMalloc(1 * sizeof(float));;
			x5890[0] = 1.0f;
			float* x5892 = (float*)myMalloc(1 * sizeof(float));;
			x5892[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5890,x898,1,x5892, x1308, 1, x898,1));
			arrayFill_greg<<<28, 512>>>(x1308, 0.0f, 512);
			float* x5896 = (float*)myMalloc(1 * sizeof(float));;
			x5896[0] = 1.0f;
			float* x5898 = (float*)myMalloc(1 * sizeof(float));;
			x5898[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5896,x901,1,x5898, x1309, 1, x901,1));
			arrayFill_greg<<<28, 512>>>(x1309, 0.0f, 512);
			float* x5902 = (float*)myMalloc(1 * sizeof(float));;
			x5902[0] = 1.0f;
			float* x5904 = (float*)myMalloc(1 * sizeof(float));;
			x5904[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5902,x904,1,x5904, x1310, 1, x904,1));
			arrayFill_greg<<<28, 512>>>(x1310, 0.0f, 256);
			float* x5908 = (float*)myMalloc(1 * sizeof(float));;
			x5908[0] = 1.0f;
			float* x5910 = (float*)myMalloc(1 * sizeof(float));;
			x5910[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5908,x907,1,x5910, x1311, 1, x907,1));
			arrayFill_greg<<<28, 512>>>(x1311, 0.0f, 128);
			float* x5914 = (float*)myMalloc(1 * sizeof(float));;
			x5914[0] = 1.0f;
			float* x5916 = (float*)myMalloc(1 * sizeof(float));;
			x5916[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5914,x910,1,x5916, x1312, 1, x910,1));
			arrayFill_greg<<<28, 512>>>(x1312, 0.0f, 512);
			float* x5920 = (float*)myMalloc(1 * sizeof(float));;
			x5920[0] = 1.0f;
			float* x5922 = (float*)myMalloc(1 * sizeof(float));;
			x5922[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5920,x913,1,x5922, x1313, 1, x913,1));
			arrayFill_greg<<<28, 512>>>(x1313, 0.0f, 64);
			float* x5926 = (float*)myMalloc(1 * sizeof(float));;
			x5926[0] = 1.0f;
			float* x5928 = (float*)myMalloc(1 * sizeof(float));;
			x5928[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5926,x916,1,x5928, x1314, 1, x916,1));
			arrayFill_greg<<<28, 512>>>(x1314, 0.0f, 512);
			float* x5932 = (float*)myMalloc(1 * sizeof(float));;
			x5932[0] = 1.0f;
			float* x5934 = (float*)myMalloc(1 * sizeof(float));;
			x5934[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x5932,x919,1,x5934, x1315, 1, x919,1));
			arrayFill_greg<<<28, 512>>>(x1315, 0.0f, 64);
			float* x5938 = (float*)myMalloc(1 * sizeof(float));;
			x5938[0] = 1.0f;
			float* x5940 = (float*)myMalloc(1 * sizeof(float));;
			x5940[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5938,x922,1,x5940, x1316, 1, x922,1));
			arrayFill_greg<<<28, 512>>>(x1316, 0.0f, 1024);
			float* x5944 = (float*)myMalloc(1 * sizeof(float));;
			x5944[0] = 1.0f;
			float* x5946 = (float*)myMalloc(1 * sizeof(float));;
			x5946[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5944,x925,1,x5946, x1317, 1, x925,1));
			arrayFill_greg<<<28, 512>>>(x1317, 0.0f, 512);
			float* x5950 = (float*)myMalloc(1 * sizeof(float));;
			x5950[0] = 1.0f;
			float* x5952 = (float*)myMalloc(1 * sizeof(float));;
			x5952[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5950,x928,1,x5952, x1318, 1, x928,1));
			arrayFill_greg<<<28, 512>>>(x1318, 0.0f, 1024);
			float* x5956 = (float*)myMalloc(1 * sizeof(float));;
			x5956[0] = 1.0f;
			float* x5958 = (float*)myMalloc(1 * sizeof(float));;
			x5958[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,2048,x5956,x931,512,x5958, x1319, 512, x931,512));
			arrayFill_greg<<<28, 512>>>(x1319, 0.0f, 1048576);
			float* x5962 = (float*)myMalloc(1 * sizeof(float));;
			x5962[0] = 1.0f;
			float* x5964 = (float*)myMalloc(1 * sizeof(float));;
			x5964[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x5962,x934,1,x5964, x1320, 1, x934,1));
			arrayFill_greg<<<28, 512>>>(x1320, 0.0f, 512);
			float* x5968 = (float*)myMalloc(1 * sizeof(float));;
			x5968[0] = 1.0f;
			float* x5970 = (float*)myMalloc(1 * sizeof(float));;
			x5970[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,2048,x5968,x937,1024,x5970, x1321, 1024, x937,1024));
			arrayFill_greg<<<28, 512>>>(x1321, 0.0f, 2097152);
			float* x5974 = (float*)myMalloc(1 * sizeof(float));;
			x5974[0] = 1.0f;
			float* x5976 = (float*)myMalloc(1 * sizeof(float));;
			x5976[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2048,512,x5974,x940,2048,x5976, x1322, 2048, x940,2048));
			arrayFill_greg<<<28, 512>>>(x1322, 0.0f, 1048576);
			float* x5980 = (float*)myMalloc(1 * sizeof(float));;
			x5980[0] = 1.0f;
			float* x5982 = (float*)myMalloc(1 * sizeof(float));;
			x5982[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x5980,x943,1,x5982, x1323, 1, x943,1));
			arrayFill_greg<<<28, 512>>>(x1323, 0.0f, 1024);
			float* x5986 = (float*)myMalloc(1 * sizeof(float));;
			x5986[0] = 1.0f;
			float* x5988 = (float*)myMalloc(1 * sizeof(float));;
			x5988[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x5986,x946,1,x5988, x1324, 1, x946,1));
			arrayFill_greg<<<28, 512>>>(x1324, 0.0f, 128);
			float* x5992 = (float*)myMalloc(1 * sizeof(float));;
			x5992[0] = 1.0f;
			float* x5994 = (float*)myMalloc(1 * sizeof(float));;
			x5994[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1024,256,x5992,x949,1024,x5994, x1325, 1024, x949,1024));
			arrayFill_greg<<<28, 512>>>(x1325, 0.0f, 262144);
			float* x5998 = (float*)myMalloc(1 * sizeof(float));;
			x5998[0] = 1.0f;
			float* x6000 = (float*)myMalloc(1 * sizeof(float));;
			x6000[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x5998,x952,1,x6000, x1326, 1, x952,1));
			arrayFill_greg<<<28, 512>>>(x1326, 0.0f, 256);
			float* x6004 = (float*)myMalloc(1 * sizeof(float));;
			x6004[0] = 1.0f;
			float* x6006 = (float*)myMalloc(1 * sizeof(float));;
			x6006[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6004,x955,1,x6006, x1327, 1, x955,1));
			arrayFill_greg<<<28, 512>>>(x1327, 0.0f, 1024);
			float* x6010 = (float*)myMalloc(1 * sizeof(float));;
			x6010[0] = 1.0f;
			float* x6012 = (float*)myMalloc(1 * sizeof(float));;
			x6012[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x6010,x958,256,x6012, x1328, 256, x958,256));
			arrayFill_greg<<<28, 512>>>(x1328, 0.0f, 262144);
			float* x6016 = (float*)myMalloc(1 * sizeof(float));;
			x6016[0] = 1.0f;
			float* x6018 = (float*)myMalloc(1 * sizeof(float));;
			x6018[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x6016,x961,1,x6018, x1329, 1, x961,1));
			arrayFill_greg<<<28, 512>>>(x1329, 0.0f, 128);
			float* x6022 = (float*)myMalloc(1 * sizeof(float));;
			x6022[0] = 1.0f;
			float* x6024 = (float*)myMalloc(1 * sizeof(float));;
			x6024[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x6022,x964,1,x6024, x1330, 1, x964,1));
			arrayFill_greg<<<28, 512>>>(x1330, 0.0f, 512);
			float* x6028 = (float*)myMalloc(1 * sizeof(float));;
			x6028[0] = 1.0f;
			float* x6030 = (float*)myMalloc(1 * sizeof(float));;
			x6030[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x6028,x967,1,x6030, x1331, 1, x967,1));
			arrayFill_greg<<<28, 512>>>(x1331, 0.0f, 512);
			float* x6034 = (float*)myMalloc(1 * sizeof(float));;
			x6034[0] = 1.0f;
			float* x6036 = (float*)myMalloc(1 * sizeof(float));;
			x6036[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x6034,x970,1,x6036, x1332, 1, x970,1));
			arrayFill_greg<<<28, 512>>>(x1332, 0.0f, 128);
			float* x6040 = (float*)myMalloc(1 * sizeof(float));;
			x6040[0] = 1.0f;
			float* x6042 = (float*)myMalloc(1 * sizeof(float));;
			x6042[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x6040,x973,2304,x6042, x1333, 2304, x973,2304));
			arrayFill_greg<<<28, 512>>>(x1333, 0.0f, 589824);
			float* x6046 = (float*)myMalloc(1 * sizeof(float));;
			x6046[0] = 1.0f;
			float* x6048 = (float*)myMalloc(1 * sizeof(float));;
			x6048[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2048,10,x6046,x976,2048,x6048, x1334, 2048, x976,2048));
			arrayFill_greg<<<28, 512>>>(x1334, 0.0f, 20480);
			float* x6052 = (float*)myMalloc(1 * sizeof(float));;
			x6052[0] = 1.0f;
			float* x6054 = (float*)myMalloc(1 * sizeof(float));;
			x6054[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6052,x979,1,x6054, x1335, 1, x979,1));
			arrayFill_greg<<<28, 512>>>(x1335, 0.0f, 256);
			float* x6058 = (float*)myMalloc(1 * sizeof(float));;
			x6058[0] = 1.0f;
			float* x6060 = (float*)myMalloc(1 * sizeof(float));;
			x6060[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6058,x982,1,x6060, x1336, 1, x982,1));
			arrayFill_greg<<<28, 512>>>(x1336, 0.0f, 256);
			float* x6064 = (float*)myMalloc(1 * sizeof(float));;
			x6064[0] = 1.0f;
			float* x6066 = (float*)myMalloc(1 * sizeof(float));;
			x6066[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6064,x985,1,x6066, x1337, 1, x985,1));
			arrayFill_greg<<<28, 512>>>(x1337, 0.0f, 256);
			float* x6070 = (float*)myMalloc(1 * sizeof(float));;
			x6070[0] = 1.0f;
			float* x6072 = (float*)myMalloc(1 * sizeof(float));;
			x6072[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6070,x988,1,x6072, x1338, 1, x988,1));
			arrayFill_greg<<<28, 512>>>(x1338, 0.0f, 1024);
			float* x6076 = (float*)myMalloc(1 * sizeof(float));;
			x6076[0] = 1.0f;
			float* x6078 = (float*)myMalloc(1 * sizeof(float));;
			x6078[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6076,x991,1,x6078, x1339, 1, x991,1));
			arrayFill_greg<<<28, 512>>>(x1339, 0.0f, 1024);
			float* x6082 = (float*)myMalloc(1 * sizeof(float));;
			x6082[0] = 1.0f;
			float* x6084 = (float*)myMalloc(1 * sizeof(float));;
			x6084[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,64,x6082,x994,64,x6084, x1340, 64, x994,64));
			arrayFill_greg<<<28, 512>>>(x1340, 0.0f, 4096);
			float* x6088 = (float*)myMalloc(1 * sizeof(float));;
			x6088[0] = 1.0f;
			float* x6090 = (float*)myMalloc(1 * sizeof(float));;
			x6090[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x6088,x997,1,x6090, x1341, 1, x997,1));
			arrayFill_greg<<<28, 512>>>(x1341, 0.0f, 512);
			float* x6094 = (float*)myMalloc(1 * sizeof(float));;
			x6094[0] = 1.0f;
			float* x6096 = (float*)myMalloc(1 * sizeof(float));;
			x6096[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1152,128,x6094,x1000,1152,x6096, x1342, 1152, x1000,1152));
			arrayFill_greg<<<28, 512>>>(x1342, 0.0f, 147456);
			float* x6100 = (float*)myMalloc(1 * sizeof(float));;
			x6100[0] = 1.0f;
			float* x6102 = (float*)myMalloc(1 * sizeof(float));;
			x6102[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x6100,x1003,1,x6102, x1343, 1, x1003,1));
			arrayFill_greg<<<28, 512>>>(x1343, 0.0f, 128);
			float* x6106 = (float*)myMalloc(1 * sizeof(float));;
			x6106[0] = 1.0f;
			float* x6108 = (float*)myMalloc(1 * sizeof(float));;
			x6108[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6106,x1006,1,x6108, x1344, 1, x1006,1));
			arrayFill_greg<<<28, 512>>>(x1344, 0.0f, 256);
			float* x6112 = (float*)myMalloc(1 * sizeof(float));;
			x6112[0] = 1.0f;
			float* x6114 = (float*)myMalloc(1 * sizeof(float));;
			x6114[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6112,x1009,1,x6114, x1345, 1, x1009,1));
			arrayFill_greg<<<28, 512>>>(x1345, 0.0f, 1024);
			float* x6118 = (float*)myMalloc(1 * sizeof(float));;
			x6118[0] = 1.0f;
			float* x6120 = (float*)myMalloc(1 * sizeof(float));;
			x6120[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x6118,x1012,1,x6120, x1346, 1, x1012,1));
			arrayFill_greg<<<28, 512>>>(x1346, 0.0f, 2048);
			float* x6124 = (float*)myMalloc(1 * sizeof(float));;
			x6124[0] = 1.0f;
			float* x6126 = (float*)myMalloc(1 * sizeof(float));;
			x6126[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6124,x1015,1,x6126, x1347, 1, x1015,1));
			arrayFill_greg<<<28, 512>>>(x1347, 0.0f, 256);
			float* x6130 = (float*)myMalloc(1 * sizeof(float));;
			x6130[0] = 1.0f;
			float* x6132 = (float*)myMalloc(1 * sizeof(float));;
			x6132[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6130,x1018,1,x6132, x1348, 1, x1018,1));
			arrayFill_greg<<<28, 512>>>(x1348, 0.0f, 256);
			float* x6136 = (float*)myMalloc(1 * sizeof(float));;
			x6136[0] = 1.0f;
			float* x6138 = (float*)myMalloc(1 * sizeof(float));;
			x6138[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x6136,x1021,1,x6138, x1349, 1, x1021,1));
			arrayFill_greg<<<28, 512>>>(x1349, 0.0f, 128);
			float* x6142 = (float*)myMalloc(1 * sizeof(float));;
			x6142[0] = 1.0f;
			float* x6144 = (float*)myMalloc(1 * sizeof(float));;
			x6144[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6142,x1024,1,x6144, x1350, 1, x1024,1));
			arrayFill_greg<<<28, 512>>>(x1350, 0.0f, 256);
			float* x6148 = (float*)myMalloc(1 * sizeof(float));;
			x6148[0] = 1.0f;
			float* x6150 = (float*)myMalloc(1 * sizeof(float));;
			x6150[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x6148,x1027,1,x6150, x1351, 1, x1027,1));
			arrayFill_greg<<<28, 512>>>(x1351, 0.0f, 64);
			float* x6154 = (float*)myMalloc(1 * sizeof(float));;
			x6154[0] = 1.0f;
			float* x6156 = (float*)myMalloc(1 * sizeof(float));;
			x6156[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x6154,x1030,1,x6156, x1352, 1, x1030,1));
			arrayFill_greg<<<28, 512>>>(x1352, 0.0f, 2048);
			float* x6160 = (float*)myMalloc(1 * sizeof(float));;
			x6160[0] = 1.0f;
			float* x6162 = (float*)myMalloc(1 * sizeof(float));;
			x6162[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x6160,x1033,1,x6162, x1353, 1, x1033,1));
			arrayFill_greg<<<28, 512>>>(x1353, 0.0f, 512);
			float* x6166 = (float*)myMalloc(1 * sizeof(float));;
			x6166[0] = 1.0f;
			float* x6168 = (float*)myMalloc(1 * sizeof(float));;
			x6168[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6166,x1036,1,x6168, x1354, 1, x1036,1));
			arrayFill_greg<<<28, 512>>>(x1354, 0.0f, 256);
			float* x6172 = (float*)myMalloc(1 * sizeof(float));;
			x6172[0] = 1.0f;
			float* x6174 = (float*)myMalloc(1 * sizeof(float));;
			x6174[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6172,x1039,1,x6174, x1355, 1, x1039,1));
			arrayFill_greg<<<28, 512>>>(x1355, 0.0f, 1024);
			float* x6178 = (float*)myMalloc(1 * sizeof(float));;
			x6178[0] = 1.0f;
			float* x6180 = (float*)myMalloc(1 * sizeof(float));;
			x6180[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x6178,x1042,2304,x6180, x1356, 2304, x1042,2304));
			arrayFill_greg<<<28, 512>>>(x1356, 0.0f, 589824);
			float* x6184 = (float*)myMalloc(1 * sizeof(float));;
			x6184[0] = 1.0f;
			float* x6186 = (float*)myMalloc(1 * sizeof(float));;
			x6186[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6184,x1045,1,x6186, x1357, 1, x1045,1));
			arrayFill_greg<<<28, 512>>>(x1357, 0.0f, 256);
			float* x6190 = (float*)myMalloc(1 * sizeof(float));;
			x6190[0] = 1.0f;
			float* x6192 = (float*)myMalloc(1 * sizeof(float));;
			x6192[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x6190,x1048,1,x6192, x1358, 1, x1048,1));
			arrayFill_greg<<<28, 512>>>(x1358, 0.0f, 64);
			float* x6196 = (float*)myMalloc(1 * sizeof(float));;
			x6196[0] = 1.0f;
			float* x6198 = (float*)myMalloc(1 * sizeof(float));;
			x6198[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x6196,x1051,1,x6198, x1359, 1, x1051,1));
			arrayFill_greg<<<28, 512>>>(x1359, 0.0f, 128);
			float* x6202 = (float*)myMalloc(1 * sizeof(float));;
			x6202[0] = 1.0f;
			float* x6204 = (float*)myMalloc(1 * sizeof(float));;
			x6204[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6202,x1054,1,x6204, x1360, 1, x1054,1));
			arrayFill_greg<<<28, 512>>>(x1360, 0.0f, 256);
			float* x6208 = (float*)myMalloc(1 * sizeof(float));;
			x6208[0] = 1.0f;
			float* x6210 = (float*)myMalloc(1 * sizeof(float));;
			x6210[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6208,x1057,1,x6210, x1361, 1, x1057,1));
			arrayFill_greg<<<28, 512>>>(x1361, 0.0f, 256);
			float* x6214 = (float*)myMalloc(1 * sizeof(float));;
			x6214[0] = 1.0f;
			float* x6216 = (float*)myMalloc(1 * sizeof(float));;
			x6216[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,512,x6214,x1060,1,x6216, x1362, 1, x1060,1));
			arrayFill_greg<<<28, 512>>>(x1362, 0.0f, 512);
			float* x6220 = (float*)myMalloc(1 * sizeof(float));;
			x6220[0] = 1.0f;
			float* x6222 = (float*)myMalloc(1 * sizeof(float));;
			x6222[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,128,x6220,x1063,512,x6222, x1363, 512, x1063,512));
			arrayFill_greg<<<28, 512>>>(x1363, 0.0f, 65536);
			float* x6226 = (float*)myMalloc(1 * sizeof(float));;
			x6226[0] = 1.0f;
			float* x6228 = (float*)myMalloc(1 * sizeof(float));;
			x6228[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x6226,x1066,1,x6228, x1364, 1, x1066,1));
			arrayFill_greg<<<28, 512>>>(x1364, 0.0f, 64);
			float* x6232 = (float*)myMalloc(1 * sizeof(float));;
			x6232[0] = 1.0f;
			float* x6234 = (float*)myMalloc(1 * sizeof(float));;
			x6234[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,512,x6232,x1069,256,x6234, x1365, 256, x1069,256));
			arrayFill_greg<<<28, 512>>>(x1365, 0.0f, 131072);
			float* x6238 = (float*)myMalloc(1 * sizeof(float));;
			x6238[0] = 1.0f;
			float* x6240 = (float*)myMalloc(1 * sizeof(float));;
			x6240[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6238,x1072,1,x6240, x1366, 1, x1072,1));
			arrayFill_greg<<<28, 512>>>(x1366, 0.0f, 256);
			float* x6244 = (float*)myMalloc(1 * sizeof(float));;
			x6244[0] = 1.0f;
			float* x6246 = (float*)myMalloc(1 * sizeof(float));;
			x6246[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,2048,x6244,x1075,1,x6246, x1367, 1, x1075,1));
			arrayFill_greg<<<28, 512>>>(x1367, 0.0f, 2048);
			float* x6250 = (float*)myMalloc(1 * sizeof(float));;
			x6250[0] = 1.0f;
			float* x6252 = (float*)myMalloc(1 * sizeof(float));;
			x6252[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x6250,x1078,1,x6252, x1368, 1, x1078,1));
			arrayFill_greg<<<28, 512>>>(x1368, 0.0f, 128);
			float* x6256 = (float*)myMalloc(1 * sizeof(float));;
			x6256[0] = 1.0f;
			float* x6258 = (float*)myMalloc(1 * sizeof(float));;
			x6258[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2304,256,x6256,x1081,2304,x6258, x1369, 2304, x1081,2304));
			arrayFill_greg<<<28, 512>>>(x1369, 0.0f, 589824);
			float* x6262 = (float*)myMalloc(1 * sizeof(float));;
			x6262[0] = 1.0f;
			float* x6264 = (float*)myMalloc(1 * sizeof(float));;
			x6264[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6262,x1084,1,x6264, x1370, 1, x1084,1));
			arrayFill_greg<<<28, 512>>>(x1370, 0.0f, 1024);
			float* x6268 = (float*)myMalloc(1 * sizeof(float));;
			x6268[0] = 1.0f;
			float* x6270 = (float*)myMalloc(1 * sizeof(float));;
			x6270[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6268,x1087,1,x6270, x1371, 1, x1087,1));
			arrayFill_greg<<<28, 512>>>(x1371, 0.0f, 256);
			float* x6274 = (float*)myMalloc(1 * sizeof(float));;
			x6274[0] = 1.0f;
			float* x6276 = (float*)myMalloc(1 * sizeof(float));;
			x6276[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 2048,512,x6274,x1090,2048,x6276, x1372, 2048, x1090,2048));
			arrayFill_greg<<<28, 512>>>(x1372, 0.0f, 1048576);
			float* x6280 = (float*)myMalloc(1 * sizeof(float));;
			x6280[0] = 1.0f;
			float* x6282 = (float*)myMalloc(1 * sizeof(float));;
			x6282[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x6280,x1093,1,x6282, x1373, 1, x1093,1));
			arrayFill_greg<<<28, 512>>>(x1373, 0.0f, 128);
			float* x6286 = (float*)myMalloc(1 * sizeof(float));;
			x6286[0] = 1.0f;
			float* x6288 = (float*)myMalloc(1 * sizeof(float));;
			x6288[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6286,x1096,1,x6288, x1374, 1, x1096,1));
			arrayFill_greg<<<28, 512>>>(x1374, 0.0f, 1024);
			float* x6292 = (float*)myMalloc(1 * sizeof(float));;
			x6292[0] = 1.0f;
			float* x6294 = (float*)myMalloc(1 * sizeof(float));;
			x6294[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x6292,x1099,1,x6294, x1375, 1, x1099,1));
			arrayFill_greg<<<28, 512>>>(x1375, 0.0f, 128);
			float* x6298 = (float*)myMalloc(1 * sizeof(float));;
			x6298[0] = 1.0f;
			float* x6300 = (float*)myMalloc(1 * sizeof(float));;
			x6300[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,1024,x6298,x1102,256,x6300, x1376, 256, x1102,256));
			arrayFill_greg<<<28, 512>>>(x1376, 0.0f, 262144);
			float* x6304 = (float*)myMalloc(1 * sizeof(float));;
			x6304[0] = 1.0f;
			float* x6306 = (float*)myMalloc(1 * sizeof(float));;
			x6306[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6304,x1105,1,x6306, x1377, 1, x1105,1));
			arrayFill_greg<<<28, 512>>>(x1377, 0.0f, 256);
			float* x6310 = (float*)myMalloc(1 * sizeof(float));;
			x6310[0] = 1.0f;
			float* x6312 = (float*)myMalloc(1 * sizeof(float));;
			x6312[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x6310,x1108,1,x6312, x1378, 1, x1108,1));
			arrayFill_greg<<<28, 512>>>(x1378, 0.0f, 256);
			float* x6316 = (float*)myMalloc(1 * sizeof(float));;
			x6316[0] = 1.0f;
			float* x6318 = (float*)myMalloc(1 * sizeof(float));;
			x6318[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,1024,x6316,x1111,1,x6318, x1379, 1, x1111,1));
			arrayFill_greg<<<28, 512>>>(x1379, 0.0f, 1024);
			int32_t x6322 = x1396 + 1;
			int32_t x6324 = x6322 % x6323;
			bool x6325 = x6324 == 0;
			if (x6325) {
				float x6330 = x1390;
				double x6326 = (double)x1397;
				double x6327 = 100.0 * x6326;
				double x6329 = x6327 / x6328;
				float x6331 = (float)x1396;
				float x6332 = x6330 / x6331;
				printf("Train epoch %d: [%d/%d (%.0f%%)] Average Loss: %.6f\n",x1386,x1397,x11,x6329,x6332);
				fflush(stdout);
			} else {
			}
			int64_t x6337 = (long)mallocAddr;
			int64_t x6338 = x6337 - x1382;
			memset((void*)x1382, 0, x6338);
			mallocAddr = (void*)x1382;
			int64_t x6341 = (long)gpuMallocAddr;
			int64_t x6342 = x6341 - x1383;
			cudaMemset((void*)x1383, 0, x6342);
			gpuMallocAddr = (void*)x1383;

		}
		gettimeofday(&end_1, NULL);
		timeval_subtract(&diff_1, &end_1, &begin_1);;
		int64_t x6349 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
		double x6350 = (double)x6349;
		double x6351 = x6350 / 1000000.0;
		x1381[x1386] = x6351;
		int64_t x6353 = x6349 / 1000LL;
		int64_t x6355 = x6349 / x6354;
		printf("Training completed in %ldms (%ld us/images)\n",x6353,x6355);
		float x6357 = x1390;
		float x6359 = x6357 / x6358;
		double x6360 = (double)x6359;
		x1380[x1386] = x6360;

	}
	gettimeofday(&end_0, NULL);
	timeval_subtract(&diff_0, &end_0, &begin_0);;
	int64_t x6366 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
	sort(x1381, x1381 + 4);
	double x6372 = x1381[2];
	int64_t x6373 = (long)fopen(x0, "w");
	fprintf((FILE *)x6373, "unit: %s\n", "1 epoch");
	for(int x6375=0; x6375 < 4; x6375++) {
		double x6376 = x1380[x6375];
		fprintf((FILE *)x6373, "%lf\n", x6376);

	}
	fprintf((FILE *)x6373, "run time: %lf %lf\n", x39, x6372);
	fclose((FILE*)x6373);
	// Backend cleanup.
	CUBLAS_CALL(cublasDestroy(cublasHandle));
	CUDA_CALL(cudaFree(gpuMallocBase));

	CUDNN_CALL(cudnnDestroy(cudnnHandle));
}
/*****************************************
  End of C Generated Code                  
 *******************************************/


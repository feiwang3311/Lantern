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
	printf("Data reading in %lf sec\n",x39);
	// Tensor 'toGPU' invocation.
	float* x98 = (float*)myGpuMalloc(32768 * sizeof(float));
	int32_t x41 = open("/u/data/u99/wang603/TiarkMlEnv/Lantern/src/out/PLDI19evaluation/squeezenet/squeezenetCifar10.onnx.bin",0);
	int64_t x42 = fsize(x41);
	float* x43 = (float*)mmap(0, x42, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x41, 0);
	float* x45 = x43+526720;
	CUDA_CALL(cudaMemcpy(x98, x45, 32768 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x101 = (float*)myGpuMalloc(48 * sizeof(float));
	float* x46 = x43+245136;
	CUDA_CALL(cudaMemcpy(x101, x46, 48 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x104 = (float*)myGpuMalloc(64 * sizeof(float));
	float* x47 = x43+17696;
	CUDA_CALL(cudaMemcpy(x104, x47, 64 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x107 = (float*)myGpuMalloc(81920 * sizeof(float));
	float* x48 = x43+723904;
	CUDA_CALL(cudaMemcpy(x107, x48, 81920 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x110 = (float*)myGpuMalloc(64 * sizeof(float));
	float* x49 = x43+14544;
	CUDA_CALL(cudaMemcpy(x110, x49, 64 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x113 = (float*)myGpuMalloc(36864 * sizeof(float));
	float* x50 = x43+35392;
	CUDA_CALL(cudaMemcpy(x113, x50, 36864 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x116 = (float*)myGpuMalloc(4096 * sizeof(float));
	float* x51 = x43+80608;
	CUDA_CALL(cudaMemcpy(x116, x51, 4096 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x119 = (float*)myGpuMalloc(16 * sizeof(float));
	float* x52 = x43+4224;
	CUDA_CALL(cudaMemcpy(x119, x52, 16 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x122 = (float*)myGpuMalloc(64 * sizeof(float));
	float* x53 = x43+362304;
	CUDA_CALL(cudaMemcpy(x122, x53, 64 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x125 = (float*)myGpuMalloc(4096 * sizeof(float));
	float* x54 = x43+27040;
	CUDA_CALL(cudaMemcpy(x125, x54, 4096 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x128 = (float*)myGpuMalloc(1024 * sizeof(float));
	float* x55 = x43+16672;
	CUDA_CALL(cudaMemcpy(x128, x55, 1024 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x131 = (float*)myGpuMalloc(2048 * sizeof(float));
	float* x56 = x43+14608;
	CUDA_CALL(cudaMemcpy(x131, x56, 2048 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x134 = (float*)myGpuMalloc(256 * sizeof(float));
	float* x57 = x43+526464;
	CUDA_CALL(cudaMemcpy(x134, x57, 256 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x137 = (float*)myGpuMalloc(18432 * sizeof(float));
	float* x58 = x43+226704;
	CUDA_CALL(cudaMemcpy(x137, x58, 18432 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x140 = (float*)myGpuMalloc(32 * sizeof(float));
	float* x59 = x43+80576;
	CUDA_CALL(cudaMemcpy(x140, x59, 32 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x143 = (float*)myGpuMalloc(128 * sizeof(float));
	float* x60 = x43+121696;
	CUDA_CALL(cudaMemcpy(x143, x60, 128 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x146 = (float*)myGpuMalloc(256 * sizeof(float));
	float* x61 = x43+723648;
	CUDA_CALL(cudaMemcpy(x146, x61, 256 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x149 = (float*)myGpuMalloc(82944 * sizeof(float));
	float* x62 = x43+254592;
	CUDA_CALL(cudaMemcpy(x149, x62, 82944 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x152 = (float*)myGpuMalloc(9216 * sizeof(float));
	float* x63 = x43+17760;
	CUDA_CALL(cudaMemcpy(x152, x63, 9216 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x155 = (float*)myGpuMalloc(64 * sizeof(float));
	float* x64 = x43+559488;
	CUDA_CALL(cudaMemcpy(x155, x64, 64 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x158 = (float*)myGpuMalloc(128 * sizeof(float));
	float* x65 = x43+84704;
	CUDA_CALL(cudaMemcpy(x158, x65, 128 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x161 = (float*)myGpuMalloc(9216 * sizeof(float));
	float* x66 = x43+245184;
	CUDA_CALL(cudaMemcpy(x161, x66, 9216 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x164 = (float*)myGpuMalloc(32 * sizeof(float));
	float* x67 = x43+31136;
	CUDA_CALL(cudaMemcpy(x164, x67, 32 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x167 = (float*)myGpuMalloc(1024 * sizeof(float));
	float* x68 = x43+4240;
	CUDA_CALL(cudaMemcpy(x167, x68, 1024 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x170 = (float*)myGpuMalloc(16 * sizeof(float));
	float* x69 = x43+16656;
	CUDA_CALL(cudaMemcpy(x170, x69, 16 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x173 = (float*)myGpuMalloc(256 * sizeof(float));
	float* x70 = x43+575936;
	CUDA_CALL(cudaMemcpy(x173, x70, 256 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x176 = (float*)myGpuMalloc(8192 * sizeof(float));
	float* x71 = x43+72384;
	CUDA_CALL(cudaMemcpy(x176, x71, 8192 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x179 = (float*)myGpuMalloc(147456 * sizeof(float));
	float* x72 = x43+379008;
	CUDA_CALL(cudaMemcpy(x179, x72, 147456 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x182 = (float*)myGpuMalloc(192 * sizeof(float));
	float* x73 = x43+226512;
	CUDA_CALL(cudaMemcpy(x182, x73, 192 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x185 = (float*)myGpuMalloc(147456 * sizeof(float));
	float* x74 = x43+576192;
	CUDA_CALL(cudaMemcpy(x185, x74, 147456 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x188 = (float*)myGpuMalloc(64 * sizeof(float));
	float* x75 = x43+5264;
	CUDA_CALL(cudaMemcpy(x188, x75, 64 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x191 = (float*)myGpuMalloc(192 * sizeof(float));
	float* x76 = x43+254400;
	CUDA_CALL(cudaMemcpy(x191, x76, 192 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x194 = (float*)myGpuMalloc(2592 * sizeof(float));
	float* x77 = x43+0;
	CUDA_CALL(cudaMemcpy(x194, x77, 2592 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x197 = (float*)myGpuMalloc(24576 * sizeof(float));
	float* x78 = x43+337728;
	CUDA_CALL(cudaMemcpy(x197, x78, 24576 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x200 = (float*)myGpuMalloc(4096 * sizeof(float));
	float* x79 = x43+31168;
	CUDA_CALL(cudaMemcpy(x200, x79, 4096 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x203 = (float*)myGpuMalloc(36864 * sizeof(float));
	float* x80 = x43+84832;
	CUDA_CALL(cudaMemcpy(x203, x80, 36864 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x206 = (float*)myGpuMalloc(64 * sizeof(float));
	float* x81 = x43+26976;
	CUDA_CALL(cudaMemcpy(x206, x81, 64 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x209 = (float*)myGpuMalloc(16384 * sizeof(float));
	float* x82 = x43+559552;
	CUDA_CALL(cudaMemcpy(x209, x82, 16384 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x212 = (float*)myGpuMalloc(82944 * sizeof(float));
	float* x83 = x43+143568;
	CUDA_CALL(cudaMemcpy(x212, x83, 82944 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x215 = (float*)myGpuMalloc(256 * sizeof(float));
	float* x84 = x43+378752;
	CUDA_CALL(cudaMemcpy(x215, x84, 256 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x218 = (float*)myGpuMalloc(128 * sizeof(float));
	float* x85 = x43+72256;
	CUDA_CALL(cudaMemcpy(x218, x85, 128 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x221 = (float*)myGpuMalloc(12288 * sizeof(float));
	float* x86 = x43+121824;
	CUDA_CALL(cudaMemcpy(x221, x86, 12288 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x224 = (float*)myGpuMalloc(96 * sizeof(float));
	float* x87 = x43+2592;
	CUDA_CALL(cudaMemcpy(x224, x87, 96 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x227 = (float*)myGpuMalloc(192 * sizeof(float));
	float* x88 = x43+337536;
	CUDA_CALL(cudaMemcpy(x227, x88, 192 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x230 = (float*)myGpuMalloc(128 * sizeof(float));
	float* x89 = x43+35264;
	CUDA_CALL(cudaMemcpy(x230, x89, 128 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x233 = (float*)myGpuMalloc(192 * sizeof(float));
	float* x90 = x43+143376;
	CUDA_CALL(cudaMemcpy(x233, x90, 192 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x236 = (float*)myGpuMalloc(9216 * sizeof(float));
	float* x91 = x43+5328;
	CUDA_CALL(cudaMemcpy(x236, x91, 9216 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x239 = (float*)myGpuMalloc(9216 * sizeof(float));
	float* x92 = x43+134160;
	CUDA_CALL(cudaMemcpy(x239, x92, 9216 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x242 = (float*)myGpuMalloc(16384 * sizeof(float));
	float* x93 = x43+362368;
	CUDA_CALL(cudaMemcpy(x242, x93, 16384 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x245 = (float*)myGpuMalloc(1536 * sizeof(float));
	float* x94 = x43+2688;
	CUDA_CALL(cudaMemcpy(x245, x94, 1536 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x248 = (float*)myGpuMalloc(10 * sizeof(float));
	float* x95 = x43+805824;
	CUDA_CALL(cudaMemcpy(x248, x95, 10 * sizeof(float), cudaMemcpyHostToDevice));
	// Tensor 'toGPU' invocation.
	float* x251 = (float*)myGpuMalloc(48 * sizeof(float));
	float* x96 = x43+134112;
	CUDA_CALL(cudaMemcpy(x251, x96, 48 * sizeof(float), cudaMemcpyHostToDevice));
	float* x253 = (float*)myGpuMalloc(32768 * sizeof(float));
	float* x254 = (float*)myGpuMalloc(48 * sizeof(float));
	float* x255 = (float*)myGpuMalloc(64 * sizeof(float));
	float* x256 = (float*)myGpuMalloc(81920 * sizeof(float));
	float* x257 = (float*)myGpuMalloc(64 * sizeof(float));
	float* x258 = (float*)myGpuMalloc(36864 * sizeof(float));
	float* x259 = (float*)myGpuMalloc(4096 * sizeof(float));
	float* x260 = (float*)myGpuMalloc(16 * sizeof(float));
	float* x261 = (float*)myGpuMalloc(64 * sizeof(float));
	float* x262 = (float*)myGpuMalloc(4096 * sizeof(float));
	float* x263 = (float*)myGpuMalloc(1024 * sizeof(float));
	float* x264 = (float*)myGpuMalloc(2048 * sizeof(float));
	float* x265 = (float*)myGpuMalloc(256 * sizeof(float));
	float* x266 = (float*)myGpuMalloc(18432 * sizeof(float));
	float* x267 = (float*)myGpuMalloc(32 * sizeof(float));
	float* x268 = (float*)myGpuMalloc(128 * sizeof(float));
	float* x269 = (float*)myGpuMalloc(256 * sizeof(float));
	float* x270 = (float*)myGpuMalloc(82944 * sizeof(float));
	float* x271 = (float*)myGpuMalloc(9216 * sizeof(float));
	float* x272 = (float*)myGpuMalloc(64 * sizeof(float));
	float* x273 = (float*)myGpuMalloc(128 * sizeof(float));
	float* x274 = (float*)myGpuMalloc(9216 * sizeof(float));
	float* x275 = (float*)myGpuMalloc(32 * sizeof(float));
	float* x276 = (float*)myGpuMalloc(1024 * sizeof(float));
	float* x277 = (float*)myGpuMalloc(16 * sizeof(float));
	float* x278 = (float*)myGpuMalloc(256 * sizeof(float));
	float* x279 = (float*)myGpuMalloc(8192 * sizeof(float));
	float* x280 = (float*)myGpuMalloc(147456 * sizeof(float));
	float* x281 = (float*)myGpuMalloc(192 * sizeof(float));
	float* x282 = (float*)myGpuMalloc(147456 * sizeof(float));
	float* x283 = (float*)myGpuMalloc(64 * sizeof(float));
	float* x284 = (float*)myGpuMalloc(192 * sizeof(float));
	float* x285 = (float*)myGpuMalloc(2592 * sizeof(float));
	float* x286 = (float*)myGpuMalloc(24576 * sizeof(float));
	float* x287 = (float*)myGpuMalloc(4096 * sizeof(float));
	float* x288 = (float*)myGpuMalloc(36864 * sizeof(float));
	float* x289 = (float*)myGpuMalloc(64 * sizeof(float));
	float* x290 = (float*)myGpuMalloc(16384 * sizeof(float));
	float* x291 = (float*)myGpuMalloc(82944 * sizeof(float));
	float* x292 = (float*)myGpuMalloc(256 * sizeof(float));
	float* x293 = (float*)myGpuMalloc(128 * sizeof(float));
	float* x294 = (float*)myGpuMalloc(12288 * sizeof(float));
	float* x295 = (float*)myGpuMalloc(96 * sizeof(float));
	float* x296 = (float*)myGpuMalloc(192 * sizeof(float));
	float* x297 = (float*)myGpuMalloc(128 * sizeof(float));
	float* x298 = (float*)myGpuMalloc(192 * sizeof(float));
	float* x299 = (float*)myGpuMalloc(9216 * sizeof(float));
	float* x300 = (float*)myGpuMalloc(9216 * sizeof(float));
	float* x301 = (float*)myGpuMalloc(16384 * sizeof(float));
	float* x302 = (float*)myGpuMalloc(1536 * sizeof(float));
	float* x303 = (float*)myGpuMalloc(10 * sizeof(float));
	float* x304 = (float*)myGpuMalloc(48 * sizeof(float));
	double* x305 = (double*)myMalloc(4 * sizeof(double));;
	double* x306 = (double*)myMalloc(4 * sizeof(double));;
	int64_t x307 = (long)mallocAddr;
	int64_t x308 = (long)gpuMallocAddr;
	// training loop starts here
	int32_t x319 = x11 / 64;
	bool x336 = 34 >= 3;
	bool x337;
	if (x336) {
		x337 = x336;
	} else {
		x337 = false;
	}
	int32_t x342 = 31 / 1;
	int32_t x343 = x342 + 1;
	int32_t x347 = 6144 * x343;
	int32_t x348 = x347 * x343;
	int32_t x370 = x343 - 2;
	int32_t x371 = x370 / 2;
	int32_t x372 = x371 + 1;
	int32_t x376 = 6144 * x372;
	int32_t x377 = x376 * x372;
	bool x381 = x372 >= 1;
	bool x382;
	if (x381) {
		x382 = x381;
	} else {
		x382 = false;
	}
	int32_t x387 = x371 / 1;
	int32_t x388 = x387 + 1;
	int32_t x392 = 1024 * x388;
	int32_t x393 = x392 * x388;
	bool x411 = x388 >= 1;
	bool x412;
	if (x411) {
		x412 = x411;
	} else {
		x412 = false;
	}
	int32_t x417 = x387 / 1;
	int32_t x418 = x417 + 1;
	int32_t x422 = 4096 * x418;
	int32_t x423 = x422 * x418;
	int32_t x441 = x388 + 2;
	bool x442 = x441 >= 3;
	bool x443;
	if (x442) {
		x443 = x442;
	} else {
		x443 = false;
	}
	int32_t x448 = x441 - 3;
	int32_t x449 = x448 / 1;
	int32_t x450 = x449 + 1;
	int32_t x454 = 4096 * x450;
	int32_t x455 = x454 * x450;
	bool x473 = true || false;
	bool x475;
	if (x473) {
		bool x474 = true || true;
		x475 = x474;
	} else {
		x475 = false;
	}
	bool x478;
	if (x475) {
		bool x476 = x450 == x418;
		bool x477 = x476 || false;
		x478 = x477;
	} else {
		x478 = false;
	}
	bool x479;
	if (x478) {
		bool x476 = x450 == x418;
		bool x477 = x476 || false;
		x479 = x477;
	} else {
		x479 = false;
	}
	int32_t x488 = 8192 * x418;
	int32_t x489 = x488 * x418;
	int32_t x419 = x418 * x418;
	int32_t x420 = 64 * x419;
	int32_t x421 = 64 * x420;
	int32_t x451 = x450 * x450;
	int32_t x452 = 64 * x451;
	int32_t x453 = 64 * x452;
	int32_t x486 = 128 * x419;
	bool x493 = x418 >= 1;
	bool x494;
	if (x493) {
		x494 = x493;
	} else {
		x494 = false;
	}
	int32_t x499 = x417 / 1;
	int32_t x500 = x499 + 1;
	int32_t x504 = 1024 * x500;
	int32_t x505 = x504 * x500;
	bool x523 = x500 >= 1;
	bool x524;
	if (x523) {
		x524 = x523;
	} else {
		x524 = false;
	}
	int32_t x529 = x499 / 1;
	int32_t x530 = x529 + 1;
	int32_t x534 = 4096 * x530;
	int32_t x535 = x534 * x530;
	int32_t x553 = x500 + 2;
	bool x554 = x553 >= 3;
	bool x555;
	if (x554) {
		x555 = x554;
	} else {
		x555 = false;
	}
	int32_t x560 = x553 - 3;
	int32_t x561 = x560 / 1;
	int32_t x562 = x561 + 1;
	int32_t x566 = 4096 * x562;
	int32_t x567 = x566 * x562;
	bool x587;
	if (x475) {
		bool x585 = x562 == x530;
		bool x586 = x585 || false;
		x587 = x586;
	} else {
		x587 = false;
	}
	bool x588;
	if (x587) {
		bool x585 = x562 == x530;
		bool x586 = x585 || false;
		x588 = x586;
	} else {
		x588 = false;
	}
	int32_t x597 = 8192 * x530;
	int32_t x598 = x597 * x530;
	int32_t x531 = x530 * x530;
	int32_t x532 = 64 * x531;
	int32_t x533 = 64 * x532;
	int32_t x563 = x562 * x562;
	int32_t x564 = 64 * x563;
	int32_t x565 = 64 * x564;
	int32_t x595 = 128 * x531;
	bool x602 = x530 >= 1;
	bool x603;
	if (x602) {
		x603 = x602;
	} else {
		x603 = false;
	}
	int32_t x608 = x529 / 1;
	int32_t x609 = x608 + 1;
	int32_t x613 = 2048 * x609;
	int32_t x614 = x613 * x609;
	bool x632 = x609 >= 1;
	bool x633;
	if (x632) {
		x633 = x632;
	} else {
		x633 = false;
	}
	int32_t x638 = x608 / 1;
	int32_t x639 = x638 + 1;
	int32_t x643 = 8192 * x639;
	int32_t x644 = x643 * x639;
	int32_t x662 = x609 + 2;
	bool x663 = x662 >= 3;
	bool x664;
	if (x663) {
		x664 = x663;
	} else {
		x664 = false;
	}
	int32_t x669 = x662 - 3;
	int32_t x670 = x669 / 1;
	int32_t x671 = x670 + 1;
	int32_t x675 = 8192 * x671;
	int32_t x676 = x675 * x671;
	bool x696;
	if (x475) {
		bool x694 = x671 == x639;
		bool x695 = x694 || false;
		x696 = x695;
	} else {
		x696 = false;
	}
	bool x697;
	if (x696) {
		bool x694 = x671 == x639;
		bool x695 = x694 || false;
		x697 = x695;
	} else {
		x697 = false;
	}
	int32_t x706 = 16384 * x639;
	int32_t x707 = x706 * x639;
	int32_t x640 = x639 * x639;
	int32_t x641 = 128 * x640;
	int32_t x642 = 64 * x641;
	int32_t x672 = x671 * x671;
	int32_t x673 = 128 * x672;
	int32_t x674 = 64 * x673;
	int32_t x704 = 256 * x640;
	int32_t x715 = x639 - 2;
	int32_t x716 = x715 / 2;
	int32_t x717 = x716 + 1;
	int32_t x721 = 16384 * x717;
	int32_t x722 = x721 * x717;
	bool x726 = x717 >= 1;
	bool x727;
	if (x726) {
		x727 = x726;
	} else {
		x727 = false;
	}
	int32_t x732 = x716 / 1;
	int32_t x733 = x732 + 1;
	int32_t x737 = 2048 * x733;
	int32_t x738 = x737 * x733;
	bool x756 = x733 >= 1;
	bool x757;
	if (x756) {
		x757 = x756;
	} else {
		x757 = false;
	}
	int32_t x762 = x732 / 1;
	int32_t x763 = x762 + 1;
	int32_t x767 = 8192 * x763;
	int32_t x768 = x767 * x763;
	int32_t x786 = x733 + 2;
	bool x787 = x786 >= 3;
	bool x788;
	if (x787) {
		x788 = x787;
	} else {
		x788 = false;
	}
	int32_t x793 = x786 - 3;
	int32_t x794 = x793 / 1;
	int32_t x795 = x794 + 1;
	int32_t x799 = 8192 * x795;
	int32_t x800 = x799 * x795;
	bool x820;
	if (x475) {
		bool x818 = x795 == x763;
		bool x819 = x818 || false;
		x820 = x819;
	} else {
		x820 = false;
	}
	bool x821;
	if (x820) {
		bool x818 = x795 == x763;
		bool x819 = x818 || false;
		x821 = x819;
	} else {
		x821 = false;
	}
	int32_t x830 = 16384 * x763;
	int32_t x831 = x830 * x763;
	int32_t x764 = x763 * x763;
	int32_t x765 = 128 * x764;
	int32_t x766 = 64 * x765;
	int32_t x796 = x795 * x795;
	int32_t x797 = 128 * x796;
	int32_t x798 = 64 * x797;
	int32_t x828 = 256 * x764;
	bool x835 = x763 >= 1;
	bool x836;
	if (x835) {
		x836 = x835;
	} else {
		x836 = false;
	}
	int32_t x841 = x762 / 1;
	int32_t x842 = x841 + 1;
	int32_t x846 = 3072 * x842;
	int32_t x847 = x846 * x842;
	bool x865 = x842 >= 1;
	bool x866;
	if (x865) {
		x866 = x865;
	} else {
		x866 = false;
	}
	int32_t x871 = x841 / 1;
	int32_t x872 = x871 + 1;
	int32_t x876 = 12288 * x872;
	int32_t x877 = x876 * x872;
	int32_t x895 = x842 + 2;
	bool x896 = x895 >= 3;
	bool x897;
	if (x896) {
		x897 = x896;
	} else {
		x897 = false;
	}
	int32_t x902 = x895 - 3;
	int32_t x903 = x902 / 1;
	int32_t x904 = x903 + 1;
	int32_t x908 = 12288 * x904;
	int32_t x909 = x908 * x904;
	bool x929;
	if (x475) {
		bool x927 = x904 == x872;
		bool x928 = x927 || false;
		x929 = x928;
	} else {
		x929 = false;
	}
	bool x930;
	if (x929) {
		bool x927 = x904 == x872;
		bool x928 = x927 || false;
		x930 = x928;
	} else {
		x930 = false;
	}
	int32_t x939 = 24576 * x872;
	int32_t x940 = x939 * x872;
	int32_t x873 = x872 * x872;
	int32_t x874 = 192 * x873;
	int32_t x875 = 64 * x874;
	int32_t x905 = x904 * x904;
	int32_t x906 = 192 * x905;
	int32_t x907 = 64 * x906;
	int32_t x937 = 384 * x873;
	bool x944 = x872 >= 1;
	bool x945;
	if (x944) {
		x945 = x944;
	} else {
		x945 = false;
	}
	int32_t x950 = x871 / 1;
	int32_t x951 = x950 + 1;
	int32_t x955 = 3072 * x951;
	int32_t x956 = x955 * x951;
	bool x974 = x951 >= 1;
	bool x975;
	if (x974) {
		x975 = x974;
	} else {
		x975 = false;
	}
	int32_t x980 = x950 / 1;
	int32_t x981 = x980 + 1;
	int32_t x985 = 12288 * x981;
	int32_t x986 = x985 * x981;
	int32_t x1004 = x951 + 2;
	bool x1005 = x1004 >= 3;
	bool x1006;
	if (x1005) {
		x1006 = x1005;
	} else {
		x1006 = false;
	}
	int32_t x1011 = x1004 - 3;
	int32_t x1012 = x1011 / 1;
	int32_t x1013 = x1012 + 1;
	int32_t x1017 = 12288 * x1013;
	int32_t x1018 = x1017 * x1013;
	bool x1038;
	if (x475) {
		bool x1036 = x1013 == x981;
		bool x1037 = x1036 || false;
		x1038 = x1037;
	} else {
		x1038 = false;
	}
	bool x1039;
	if (x1038) {
		bool x1036 = x1013 == x981;
		bool x1037 = x1036 || false;
		x1039 = x1037;
	} else {
		x1039 = false;
	}
	int32_t x1048 = 24576 * x981;
	int32_t x1049 = x1048 * x981;
	int32_t x982 = x981 * x981;
	int32_t x983 = 192 * x982;
	int32_t x984 = 64 * x983;
	int32_t x1014 = x1013 * x1013;
	int32_t x1015 = 192 * x1014;
	int32_t x1016 = 64 * x1015;
	int32_t x1046 = 384 * x982;
	bool x1053 = x981 >= 1;
	bool x1054;
	if (x1053) {
		x1054 = x1053;
	} else {
		x1054 = false;
	}
	int32_t x1059 = x980 / 1;
	int32_t x1060 = x1059 + 1;
	int32_t x1064 = 4096 * x1060;
	int32_t x1065 = x1064 * x1060;
	bool x1083 = x1060 >= 1;
	bool x1084;
	if (x1083) {
		x1084 = x1083;
	} else {
		x1084 = false;
	}
	int32_t x1089 = x1059 / 1;
	int32_t x1090 = x1089 + 1;
	int32_t x1094 = 16384 * x1090;
	int32_t x1095 = x1094 * x1090;
	int32_t x1113 = x1060 + 2;
	bool x1114 = x1113 >= 3;
	bool x1115;
	if (x1114) {
		x1115 = x1114;
	} else {
		x1115 = false;
	}
	int32_t x1120 = x1113 - 3;
	int32_t x1121 = x1120 / 1;
	int32_t x1122 = x1121 + 1;
	int32_t x1126 = 16384 * x1122;
	int32_t x1127 = x1126 * x1122;
	bool x1147;
	if (x475) {
		bool x1145 = x1122 == x1090;
		bool x1146 = x1145 || false;
		x1147 = x1146;
	} else {
		x1147 = false;
	}
	bool x1148;
	if (x1147) {
		bool x1145 = x1122 == x1090;
		bool x1146 = x1145 || false;
		x1148 = x1146;
	} else {
		x1148 = false;
	}
	int32_t x1157 = 32768 * x1090;
	int32_t x1158 = x1157 * x1090;
	int32_t x1091 = x1090 * x1090;
	int32_t x1092 = 256 * x1091;
	int32_t x1093 = 64 * x1092;
	int32_t x1123 = x1122 * x1122;
	int32_t x1124 = 256 * x1123;
	int32_t x1125 = 64 * x1124;
	int32_t x1155 = 512 * x1091;
	int32_t x1166 = x1090 - 2;
	int32_t x1167 = x1166 / 2;
	int32_t x1168 = x1167 + 1;
	int32_t x1172 = 32768 * x1168;
	int32_t x1173 = x1172 * x1168;
	bool x1177 = x1168 >= 1;
	bool x1178;
	if (x1177) {
		x1178 = x1177;
	} else {
		x1178 = false;
	}
	int32_t x1183 = x1167 / 1;
	int32_t x1184 = x1183 + 1;
	int32_t x1188 = 4096 * x1184;
	int32_t x1189 = x1188 * x1184;
	bool x1207 = x1184 >= 1;
	bool x1208;
	if (x1207) {
		x1208 = x1207;
	} else {
		x1208 = false;
	}
	int32_t x1213 = x1183 / 1;
	int32_t x1214 = x1213 + 1;
	int32_t x1218 = 16384 * x1214;
	int32_t x1219 = x1218 * x1214;
	int32_t x1237 = x1184 + 2;
	bool x1238 = x1237 >= 3;
	bool x1239;
	if (x1238) {
		x1239 = x1238;
	} else {
		x1239 = false;
	}
	int32_t x1244 = x1237 - 3;
	int32_t x1245 = x1244 / 1;
	int32_t x1246 = x1245 + 1;
	int32_t x1250 = 16384 * x1246;
	int32_t x1251 = x1250 * x1246;
	bool x1271;
	if (x475) {
		bool x1269 = x1246 == x1214;
		bool x1270 = x1269 || false;
		x1271 = x1270;
	} else {
		x1271 = false;
	}
	bool x1272;
	if (x1271) {
		bool x1269 = x1246 == x1214;
		bool x1270 = x1269 || false;
		x1272 = x1270;
	} else {
		x1272 = false;
	}
	int32_t x1281 = 32768 * x1214;
	int32_t x1282 = x1281 * x1214;
	int32_t x1215 = x1214 * x1214;
	int32_t x1216 = 256 * x1215;
	int32_t x1217 = 64 * x1216;
	int32_t x1247 = x1246 * x1246;
	int32_t x1248 = 256 * x1247;
	int32_t x1249 = 64 * x1248;
	int32_t x1279 = 512 * x1215;
	bool x1286 = x1214 >= 4;
	bool x1287;
	if (x1286) {
		x1287 = x1286;
	} else {
		x1287 = false;
	}
	int32_t x1292 = x1214 - 4;
	int32_t x1293 = x1292 / 1;
	int32_t x1294 = x1293 + 1;
	int32_t x1298 = 640 * x1294;
	int32_t x1299 = x1298 * x1294;
	bool x1336;
	if (x475) {
		bool x474 = true || true;
		x1336 = x474;
	} else {
		x1336 = false;
	}
	bool x1337;
	if (x1336) {
		bool x474 = true || true;
		x1337 = x474;
	} else {
		x1337 = false;
	}
	float x1335 = 1.0f / 64.0f;
	int32_t x2072 = x319 / 10;
	double x2077 = (double)x11;
	int64_t x2103 = (int64_t)x11;
	float x2107 = (float)x11;
	for(int x311=0; x311 < 4; x311++) {
		struct timeval begin_1, end_1, diff_1;
		float x313 = 0.0f;
		float x314 = x313;
		float x315 = x314;
		int32_t x316 = x311 + 1;
		printf("Start training epoch %d\n",x316);
		gettimeofday(&begin_1, NULL);
		for(int x321=0; x321 < x319; x321++) {
			int32_t x322 = x321 * 64;
			int32_t x323 = x322 * 3072;
			float* x324 = x13+x323;
			int* x325 = x14+x322;
			// Tensor 'toGPU' invocation.
			float* x327 = (float*)myGpuMalloc(196608 * sizeof(float));
			CUDA_CALL(cudaMemcpy(x327, x324, 196608 * sizeof(float), cudaMemcpyHostToDevice));
			float* x329 = (float*)myGpuMalloc(2 * sizeof(float));
			int* x330 = (int32_t*)myGpuMalloc(64 * sizeof(int32_t));
			CUDA_CALL(cudaMemcpy(x330, x325, 64 * sizeof(int32_t), cudaMemcpyHostToDevice));
			float* x332 = (float*)myGpuMalloc(1 * sizeof(float));
			float* x333 = (float*)myGpuMalloc(1 * sizeof(float));
			// allocate memory to save the final loss in CPU Tensor
			float* x335 = (float*)myMalloc(1 * sizeof(float));;
			if (x337) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x349 = (float*)myGpuMalloc(x348 * sizeof(float));
			float* x350 = (float*)myMalloc(1 * sizeof(float));;
			x350[0] = 0.0f;
			float* x352 = (float*)myMalloc(1 * sizeof(float));;
			x352[0] = 1.0f;

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
							96, 3, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 96, x343, x343));

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
							x352, in_desc, x327, filt_desc, x194,
							conv_desc, algo, ws_data, ws_size,
							x350, out_desc, x349));
			};
			float* x355 = (float*)myMalloc(1 * sizeof(float));;
			x355[0] = 1.0f;
			float* x357 = (float*)myMalloc(1 * sizeof(float));;
			x357[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 96, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 96, x343, x343));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x355, bias_desc, x224, x357, out_desc, x349));
			};
			float* x360 = (float*)myGpuMalloc(x348 * sizeof(float));
			float* x361 = (float*)myMalloc(1 * sizeof(float));;
			x361[0] = 0.0f;
			float* x363 = (float*)myMalloc(1 * sizeof(float));;
			x363[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 96, x343, x343));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x363, x_desc, x349, x361, x_desc, x349));
			};
			float* x366 = (float*)myMalloc(1 * sizeof(float));;
			x366[0] = 0.0f;
			float* x368 = (float*)myMalloc(1 * sizeof(float));;
			x368[0] = 1.0f;
			float* x378 = (float*)myGpuMalloc(x377 * sizeof(float));

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 96, x343, x343) );

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 96, x372, x372));

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
							x368, in_desc, x349, x366, out_desc, x378));
			};
			float* x380 = (float*)myGpuMalloc(x377 * sizeof(float));
			if (x382) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x394 = (float*)myGpuMalloc(x393 * sizeof(float));
			float* x395 = (float*)myMalloc(1 * sizeof(float));;
			x395[0] = 0.0f;
			float* x397 = (float*)myMalloc(1 * sizeof(float));;
			x397[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 96, x372, x372));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							16, 96, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x388, x388));

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
							x397, in_desc, x378, filt_desc, x245,
							conv_desc, algo, ws_data, ws_size,
							x395, out_desc, x394));
			};
			float* x400 = (float*)myMalloc(1 * sizeof(float));;
			x400[0] = 1.0f;
			float* x402 = (float*)myMalloc(1 * sizeof(float));;
			x402[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 16, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x388, x388));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x400, bias_desc, x119, x402, out_desc, x394));
			};
			float* x405 = (float*)myGpuMalloc(x393 * sizeof(float));
			float* x406 = (float*)myMalloc(1 * sizeof(float));;
			x406[0] = 0.0f;
			float* x408 = (float*)myMalloc(1 * sizeof(float));;
			x408[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x388, x388));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x408, x_desc, x394, x406, x_desc, x394));
			};
			if (x412) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x424 = (float*)myGpuMalloc(x423 * sizeof(float));
			float* x425 = (float*)myMalloc(1 * sizeof(float));;
			x425[0] = 0.0f;
			float* x427 = (float*)myMalloc(1 * sizeof(float));;
			x427[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x388, x388));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 16, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x418, x418));

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
							x427, in_desc, x394, filt_desc, x167,
							conv_desc, algo, ws_data, ws_size,
							x425, out_desc, x424));
			};
			float* x430 = (float*)myMalloc(1 * sizeof(float));;
			x430[0] = 1.0f;
			float* x432 = (float*)myMalloc(1 * sizeof(float));;
			x432[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x418, x418));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x430, bias_desc, x188, x432, out_desc, x424));
			};
			float* x435 = (float*)myGpuMalloc(x423 * sizeof(float));
			float* x436 = (float*)myMalloc(1 * sizeof(float));;
			x436[0] = 0.0f;
			float* x438 = (float*)myMalloc(1 * sizeof(float));;
			x438[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x418, x418));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x438, x_desc, x424, x436, x_desc, x424));
			};
			if (x443) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x456 = (float*)myGpuMalloc(x455 * sizeof(float));
			float* x457 = (float*)myMalloc(1 * sizeof(float));;
			x457[0] = 0.0f;
			float* x459 = (float*)myMalloc(1 * sizeof(float));;
			x459[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x388, x388));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 16, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x450, x450));

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
							x459, in_desc, x394, filt_desc, x236,
							conv_desc, algo, ws_data, ws_size,
							x457, out_desc, x456));
			};
			float* x462 = (float*)myMalloc(1 * sizeof(float));;
			x462[0] = 1.0f;
			float* x464 = (float*)myMalloc(1 * sizeof(float));;
			x464[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x450, x450));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x462, bias_desc, x110, x464, out_desc, x456));
			};
			float* x467 = (float*)myGpuMalloc(x455 * sizeof(float));
			float* x468 = (float*)myMalloc(1 * sizeof(float));;
			x468[0] = 0.0f;
			float* x470 = (float*)myMalloc(1 * sizeof(float));;
			x470[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x450, x450));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x470, x_desc, x456, x468, x_desc, x456));
			};
			if (x479) {
			} else {
				printf("all dimensions except the concatenation dimension should be the same\n");
				assert(false && "");
			}
			// back prop for concat
			float* x490 = (float*)myGpuMalloc(x489 * sizeof(float));
			{
				dim3 grid(28, 2);
				concat2D_1D_greg<<<grid, 512>>>(x424, 64, x421, x456, 64, x453, x490, 1, 64, 128, x418, x418, x486, x419, x418, 1);
			};
			float* x492 = (float*)myGpuMalloc(x489 * sizeof(float));
			if (x494) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x506 = (float*)myGpuMalloc(x505 * sizeof(float));
			float* x507 = (float*)myMalloc(1 * sizeof(float));;
			x507[0] = 0.0f;
			float* x509 = (float*)myMalloc(1 * sizeof(float));;
			x509[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x418, x418));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							16, 128, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x500, x500));

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
							x509, in_desc, x490, filt_desc, x131,
							conv_desc, algo, ws_data, ws_size,
							x507, out_desc, x506));
			};
			float* x512 = (float*)myMalloc(1 * sizeof(float));;
			x512[0] = 1.0f;
			float* x514 = (float*)myMalloc(1 * sizeof(float));;
			x514[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 16, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x500, x500));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x512, bias_desc, x170, x514, out_desc, x506));
			};
			float* x517 = (float*)myGpuMalloc(x505 * sizeof(float));
			float* x518 = (float*)myMalloc(1 * sizeof(float));;
			x518[0] = 0.0f;
			float* x520 = (float*)myMalloc(1 * sizeof(float));;
			x520[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x500, x500));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x520, x_desc, x506, x518, x_desc, x506));
			};
			if (x524) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x536 = (float*)myGpuMalloc(x535 * sizeof(float));
			float* x537 = (float*)myMalloc(1 * sizeof(float));;
			x537[0] = 0.0f;
			float* x539 = (float*)myMalloc(1 * sizeof(float));;
			x539[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x500, x500));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 16, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x530, x530));

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
							x539, in_desc, x506, filt_desc, x128,
							conv_desc, algo, ws_data, ws_size,
							x537, out_desc, x536));
			};
			float* x542 = (float*)myMalloc(1 * sizeof(float));;
			x542[0] = 1.0f;
			float* x544 = (float*)myMalloc(1 * sizeof(float));;
			x544[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x530, x530));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x542, bias_desc, x104, x544, out_desc, x536));
			};
			float* x547 = (float*)myGpuMalloc(x535 * sizeof(float));
			float* x548 = (float*)myMalloc(1 * sizeof(float));;
			x548[0] = 0.0f;
			float* x550 = (float*)myMalloc(1 * sizeof(float));;
			x550[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x530, x530));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x550, x_desc, x536, x548, x_desc, x536));
			};
			if (x555) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x568 = (float*)myGpuMalloc(x567 * sizeof(float));
			float* x569 = (float*)myMalloc(1 * sizeof(float));;
			x569[0] = 0.0f;
			float* x571 = (float*)myMalloc(1 * sizeof(float));;
			x571[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x500, x500));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 16, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x562, x562));

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
							x571, in_desc, x506, filt_desc, x152,
							conv_desc, algo, ws_data, ws_size,
							x569, out_desc, x568));
			};
			float* x574 = (float*)myMalloc(1 * sizeof(float));;
			x574[0] = 1.0f;
			float* x576 = (float*)myMalloc(1 * sizeof(float));;
			x576[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x562, x562));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x574, bias_desc, x206, x576, out_desc, x568));
			};
			float* x579 = (float*)myGpuMalloc(x567 * sizeof(float));
			float* x580 = (float*)myMalloc(1 * sizeof(float));;
			x580[0] = 0.0f;
			float* x582 = (float*)myMalloc(1 * sizeof(float));;
			x582[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x562, x562));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x582, x_desc, x568, x580, x_desc, x568));
			};
			if (x588) {
			} else {
				printf("all dimensions except the concatenation dimension should be the same\n");
				assert(false && "");
			}
			// back prop for concat
			float* x599 = (float*)myGpuMalloc(x598 * sizeof(float));
			{
				dim3 grid(28, 2);
				concat2D_1D_greg<<<grid, 512>>>(x536, 64, x533, x568, 64, x565, x599, 1, 64, 128, x530, x530, x595, x531, x530, 1);
			};
			float* x601 = (float*)myGpuMalloc(x598 * sizeof(float));
			if (x603) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x615 = (float*)myGpuMalloc(x614 * sizeof(float));
			float* x616 = (float*)myMalloc(1 * sizeof(float));;
			x616[0] = 0.0f;
			float* x618 = (float*)myMalloc(1 * sizeof(float));;
			x618[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x530, x530));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							32, 128, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x609, x609));

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
							x618, in_desc, x599, filt_desc, x125,
							conv_desc, algo, ws_data, ws_size,
							x616, out_desc, x615));
			};
			float* x621 = (float*)myMalloc(1 * sizeof(float));;
			x621[0] = 1.0f;
			float* x623 = (float*)myMalloc(1 * sizeof(float));;
			x623[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 32, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x609, x609));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x621, bias_desc, x164, x623, out_desc, x615));
			};
			float* x626 = (float*)myGpuMalloc(x614 * sizeof(float));
			float* x627 = (float*)myMalloc(1 * sizeof(float));;
			x627[0] = 0.0f;
			float* x629 = (float*)myMalloc(1 * sizeof(float));;
			x629[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x609, x609));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x629, x_desc, x615, x627, x_desc, x615));
			};
			if (x633) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x645 = (float*)myGpuMalloc(x644 * sizeof(float));
			float* x646 = (float*)myMalloc(1 * sizeof(float));;
			x646[0] = 0.0f;
			float* x648 = (float*)myMalloc(1 * sizeof(float));;
			x648[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x609, x609));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							128, 32, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x639, x639));

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
							x648, in_desc, x615, filt_desc, x200,
							conv_desc, algo, ws_data, ws_size,
							x646, out_desc, x645));
			};
			float* x651 = (float*)myMalloc(1 * sizeof(float));;
			x651[0] = 1.0f;
			float* x653 = (float*)myMalloc(1 * sizeof(float));;
			x653[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x639, x639));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x651, bias_desc, x230, x653, out_desc, x645));
			};
			float* x656 = (float*)myGpuMalloc(x644 * sizeof(float));
			float* x657 = (float*)myMalloc(1 * sizeof(float));;
			x657[0] = 0.0f;
			float* x659 = (float*)myMalloc(1 * sizeof(float));;
			x659[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x639, x639));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x659, x_desc, x645, x657, x_desc, x645));
			};
			if (x664) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x677 = (float*)myGpuMalloc(x676 * sizeof(float));
			float* x678 = (float*)myMalloc(1 * sizeof(float));;
			x678[0] = 0.0f;
			float* x680 = (float*)myMalloc(1 * sizeof(float));;
			x680[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x609, x609));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							128, 32, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x671, x671));

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
							x680, in_desc, x615, filt_desc, x113,
							conv_desc, algo, ws_data, ws_size,
							x678, out_desc, x677));
			};
			float* x683 = (float*)myMalloc(1 * sizeof(float));;
			x683[0] = 1.0f;
			float* x685 = (float*)myMalloc(1 * sizeof(float));;
			x685[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x671, x671));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x683, bias_desc, x218, x685, out_desc, x677));
			};
			float* x688 = (float*)myGpuMalloc(x676 * sizeof(float));
			float* x689 = (float*)myMalloc(1 * sizeof(float));;
			x689[0] = 0.0f;
			float* x691 = (float*)myMalloc(1 * sizeof(float));;
			x691[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x671, x671));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x691, x_desc, x677, x689, x_desc, x677));
			};
			if (x697) {
			} else {
				printf("all dimensions except the concatenation dimension should be the same\n");
				assert(false && "");
			}
			// back prop for concat
			float* x708 = (float*)myGpuMalloc(x707 * sizeof(float));
			{
				dim3 grid(28, 2);
				concat2D_1D_greg<<<grid, 512>>>(x645, 128, x642, x677, 128, x674, x708, 1, 64, 256, x639, x639, x704, x640, x639, 1);
			};
			float* x710 = (float*)myGpuMalloc(x707 * sizeof(float));
			float* x711 = (float*)myMalloc(1 * sizeof(float));;
			x711[0] = 0.0f;
			float* x713 = (float*)myMalloc(1 * sizeof(float));;
			x713[0] = 1.0f;
			float* x723 = (float*)myGpuMalloc(x722 * sizeof(float));

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x639, x639) );

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x717, x717));

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
							x713, in_desc, x708, x711, out_desc, x723));
			};
			float* x725 = (float*)myGpuMalloc(x722 * sizeof(float));
			if (x727) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x739 = (float*)myGpuMalloc(x738 * sizeof(float));
			float* x740 = (float*)myMalloc(1 * sizeof(float));;
			x740[0] = 0.0f;
			float* x742 = (float*)myMalloc(1 * sizeof(float));;
			x742[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x717, x717));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							32, 256, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x733, x733));

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
							x742, in_desc, x723, filt_desc, x176,
							conv_desc, algo, ws_data, ws_size,
							x740, out_desc, x739));
			};
			float* x745 = (float*)myMalloc(1 * sizeof(float));;
			x745[0] = 1.0f;
			float* x747 = (float*)myMalloc(1 * sizeof(float));;
			x747[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 32, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x733, x733));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x745, bias_desc, x140, x747, out_desc, x739));
			};
			float* x750 = (float*)myGpuMalloc(x738 * sizeof(float));
			float* x751 = (float*)myMalloc(1 * sizeof(float));;
			x751[0] = 0.0f;
			float* x753 = (float*)myMalloc(1 * sizeof(float));;
			x753[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x733, x733));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x753, x_desc, x739, x751, x_desc, x739));
			};
			if (x757) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x769 = (float*)myGpuMalloc(x768 * sizeof(float));
			float* x770 = (float*)myMalloc(1 * sizeof(float));;
			x770[0] = 0.0f;
			float* x772 = (float*)myMalloc(1 * sizeof(float));;
			x772[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x733, x733));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							128, 32, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x763, x763));

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
							x772, in_desc, x739, filt_desc, x116,
							conv_desc, algo, ws_data, ws_size,
							x770, out_desc, x769));
			};
			float* x775 = (float*)myMalloc(1 * sizeof(float));;
			x775[0] = 1.0f;
			float* x777 = (float*)myMalloc(1 * sizeof(float));;
			x777[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x763, x763));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x775, bias_desc, x158, x777, out_desc, x769));
			};
			float* x780 = (float*)myGpuMalloc(x768 * sizeof(float));
			float* x781 = (float*)myMalloc(1 * sizeof(float));;
			x781[0] = 0.0f;
			float* x783 = (float*)myMalloc(1 * sizeof(float));;
			x783[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x763, x763));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x783, x_desc, x769, x781, x_desc, x769));
			};
			if (x788) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x801 = (float*)myGpuMalloc(x800 * sizeof(float));
			float* x802 = (float*)myMalloc(1 * sizeof(float));;
			x802[0] = 0.0f;
			float* x804 = (float*)myMalloc(1 * sizeof(float));;
			x804[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x733, x733));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							128, 32, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x795, x795));

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
							x804, in_desc, x739, filt_desc, x203,
							conv_desc, algo, ws_data, ws_size,
							x802, out_desc, x801));
			};
			float* x807 = (float*)myMalloc(1 * sizeof(float));;
			x807[0] = 1.0f;
			float* x809 = (float*)myMalloc(1 * sizeof(float));;
			x809[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x795, x795));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x807, bias_desc, x143, x809, out_desc, x801));
			};
			float* x812 = (float*)myGpuMalloc(x800 * sizeof(float));
			float* x813 = (float*)myMalloc(1 * sizeof(float));;
			x813[0] = 0.0f;
			float* x815 = (float*)myMalloc(1 * sizeof(float));;
			x815[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x795, x795));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x815, x_desc, x801, x813, x_desc, x801));
			};
			if (x821) {
			} else {
				printf("all dimensions except the concatenation dimension should be the same\n");
				assert(false && "");
			}
			// back prop for concat
			float* x832 = (float*)myGpuMalloc(x831 * sizeof(float));
			{
				dim3 grid(28, 2);
				concat2D_1D_greg<<<grid, 512>>>(x769, 128, x766, x801, 128, x798, x832, 1, 64, 256, x763, x763, x828, x764, x763, 1);
			};
			float* x834 = (float*)myGpuMalloc(x831 * sizeof(float));
			if (x836) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x848 = (float*)myGpuMalloc(x847 * sizeof(float));
			float* x849 = (float*)myMalloc(1 * sizeof(float));;
			x849[0] = 0.0f;
			float* x851 = (float*)myMalloc(1 * sizeof(float));;
			x851[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x763, x763));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							48, 256, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x842, x842));

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
							x851, in_desc, x832, filt_desc, x221,
							conv_desc, algo, ws_data, ws_size,
							x849, out_desc, x848));
			};
			float* x854 = (float*)myMalloc(1 * sizeof(float));;
			x854[0] = 1.0f;
			float* x856 = (float*)myMalloc(1 * sizeof(float));;
			x856[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 48, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x842, x842));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x854, bias_desc, x251, x856, out_desc, x848));
			};
			float* x859 = (float*)myGpuMalloc(x847 * sizeof(float));
			float* x860 = (float*)myMalloc(1 * sizeof(float));;
			x860[0] = 0.0f;
			float* x862 = (float*)myMalloc(1 * sizeof(float));;
			x862[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x842, x842));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x862, x_desc, x848, x860, x_desc, x848));
			};
			if (x866) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x878 = (float*)myGpuMalloc(x877 * sizeof(float));
			float* x879 = (float*)myMalloc(1 * sizeof(float));;
			x879[0] = 0.0f;
			float* x881 = (float*)myMalloc(1 * sizeof(float));;
			x881[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x842, x842));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							192, 48, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x872, x872));

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
							x881, in_desc, x848, filt_desc, x239,
							conv_desc, algo, ws_data, ws_size,
							x879, out_desc, x878));
			};
			float* x884 = (float*)myMalloc(1 * sizeof(float));;
			x884[0] = 1.0f;
			float* x886 = (float*)myMalloc(1 * sizeof(float));;
			x886[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 192, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x872, x872));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x884, bias_desc, x233, x886, out_desc, x878));
			};
			float* x889 = (float*)myGpuMalloc(x877 * sizeof(float));
			float* x890 = (float*)myMalloc(1 * sizeof(float));;
			x890[0] = 0.0f;
			float* x892 = (float*)myMalloc(1 * sizeof(float));;
			x892[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x872, x872));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x892, x_desc, x878, x890, x_desc, x878));
			};
			if (x897) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x910 = (float*)myGpuMalloc(x909 * sizeof(float));
			float* x911 = (float*)myMalloc(1 * sizeof(float));;
			x911[0] = 0.0f;
			float* x913 = (float*)myMalloc(1 * sizeof(float));;
			x913[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x842, x842));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							192, 48, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x904, x904));

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
							x913, in_desc, x848, filt_desc, x212,
							conv_desc, algo, ws_data, ws_size,
							x911, out_desc, x910));
			};
			float* x916 = (float*)myMalloc(1 * sizeof(float));;
			x916[0] = 1.0f;
			float* x918 = (float*)myMalloc(1 * sizeof(float));;
			x918[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 192, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x904, x904));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x916, bias_desc, x182, x918, out_desc, x910));
			};
			float* x921 = (float*)myGpuMalloc(x909 * sizeof(float));
			float* x922 = (float*)myMalloc(1 * sizeof(float));;
			x922[0] = 0.0f;
			float* x924 = (float*)myMalloc(1 * sizeof(float));;
			x924[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x904, x904));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x924, x_desc, x910, x922, x_desc, x910));
			};
			if (x930) {
			} else {
				printf("all dimensions except the concatenation dimension should be the same\n");
				assert(false && "");
			}
			// back prop for concat
			float* x941 = (float*)myGpuMalloc(x940 * sizeof(float));
			{
				dim3 grid(28, 2);
				concat2D_1D_greg<<<grid, 512>>>(x878, 192, x875, x910, 192, x907, x941, 1, 64, 384, x872, x872, x937, x873, x872, 1);
			};
			float* x943 = (float*)myGpuMalloc(x940 * sizeof(float));
			if (x945) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x957 = (float*)myGpuMalloc(x956 * sizeof(float));
			float* x958 = (float*)myMalloc(1 * sizeof(float));;
			x958[0] = 0.0f;
			float* x960 = (float*)myMalloc(1 * sizeof(float));;
			x960[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 384, x872, x872));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							48, 384, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x951, x951));

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
							x960, in_desc, x941, filt_desc, x137,
							conv_desc, algo, ws_data, ws_size,
							x958, out_desc, x957));
			};
			float* x963 = (float*)myMalloc(1 * sizeof(float));;
			x963[0] = 1.0f;
			float* x965 = (float*)myMalloc(1 * sizeof(float));;
			x965[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 48, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x951, x951));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x963, bias_desc, x101, x965, out_desc, x957));
			};
			float* x968 = (float*)myGpuMalloc(x956 * sizeof(float));
			float* x969 = (float*)myMalloc(1 * sizeof(float));;
			x969[0] = 0.0f;
			float* x971 = (float*)myMalloc(1 * sizeof(float));;
			x971[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x951, x951));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x971, x_desc, x957, x969, x_desc, x957));
			};
			if (x975) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x987 = (float*)myGpuMalloc(x986 * sizeof(float));
			float* x988 = (float*)myMalloc(1 * sizeof(float));;
			x988[0] = 0.0f;
			float* x990 = (float*)myMalloc(1 * sizeof(float));;
			x990[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x951, x951));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							192, 48, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x981, x981));

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
							x990, in_desc, x957, filt_desc, x161,
							conv_desc, algo, ws_data, ws_size,
							x988, out_desc, x987));
			};
			float* x993 = (float*)myMalloc(1 * sizeof(float));;
			x993[0] = 1.0f;
			float* x995 = (float*)myMalloc(1 * sizeof(float));;
			x995[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 192, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x981, x981));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x993, bias_desc, x191, x995, out_desc, x987));
			};
			float* x998 = (float*)myGpuMalloc(x986 * sizeof(float));
			float* x999 = (float*)myMalloc(1 * sizeof(float));;
			x999[0] = 0.0f;
			float* x1001 = (float*)myMalloc(1 * sizeof(float));;
			x1001[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x981, x981));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1001, x_desc, x987, x999, x_desc, x987));
			};
			if (x1006) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1019 = (float*)myGpuMalloc(x1018 * sizeof(float));
			float* x1020 = (float*)myMalloc(1 * sizeof(float));;
			x1020[0] = 0.0f;
			float* x1022 = (float*)myMalloc(1 * sizeof(float));;
			x1022[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x951, x951));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							192, 48, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x1013, x1013));

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
							x1022, in_desc, x957, filt_desc, x149,
							conv_desc, algo, ws_data, ws_size,
							x1020, out_desc, x1019));
			};
			float* x1025 = (float*)myMalloc(1 * sizeof(float));;
			x1025[0] = 1.0f;
			float* x1027 = (float*)myMalloc(1 * sizeof(float));;
			x1027[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 192, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x1013, x1013));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x1025, bias_desc, x227, x1027, out_desc, x1019));
			};
			float* x1030 = (float*)myGpuMalloc(x1018 * sizeof(float));
			float* x1031 = (float*)myMalloc(1 * sizeof(float));;
			x1031[0] = 0.0f;
			float* x1033 = (float*)myMalloc(1 * sizeof(float));;
			x1033[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x1013, x1013));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1033, x_desc, x1019, x1031, x_desc, x1019));
			};
			if (x1039) {
			} else {
				printf("all dimensions except the concatenation dimension should be the same\n");
				assert(false && "");
			}
			// back prop for concat
			float* x1050 = (float*)myGpuMalloc(x1049 * sizeof(float));
			{
				dim3 grid(28, 2);
				concat2D_1D_greg<<<grid, 512>>>(x987, 192, x984, x1019, 192, x1016, x1050, 1, 64, 384, x981, x981, x1046, x982, x981, 1);
			};
			float* x1052 = (float*)myGpuMalloc(x1049 * sizeof(float));
			if (x1054) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1066 = (float*)myGpuMalloc(x1065 * sizeof(float));
			float* x1067 = (float*)myMalloc(1 * sizeof(float));;
			x1067[0] = 0.0f;
			float* x1069 = (float*)myMalloc(1 * sizeof(float));;
			x1069[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 384, x981, x981));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 384, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1060, x1060));

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
							x1069, in_desc, x1050, filt_desc, x197,
							conv_desc, algo, ws_data, ws_size,
							x1067, out_desc, x1066));
			};
			float* x1072 = (float*)myMalloc(1 * sizeof(float));;
			x1072[0] = 1.0f;
			float* x1074 = (float*)myMalloc(1 * sizeof(float));;
			x1074[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1060, x1060));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x1072, bias_desc, x122, x1074, out_desc, x1066));
			};
			float* x1077 = (float*)myGpuMalloc(x1065 * sizeof(float));
			float* x1078 = (float*)myMalloc(1 * sizeof(float));;
			x1078[0] = 0.0f;
			float* x1080 = (float*)myMalloc(1 * sizeof(float));;
			x1080[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1060, x1060));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1080, x_desc, x1066, x1078, x_desc, x1066));
			};
			if (x1084) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1096 = (float*)myGpuMalloc(x1095 * sizeof(float));
			float* x1097 = (float*)myMalloc(1 * sizeof(float));;
			x1097[0] = 0.0f;
			float* x1099 = (float*)myMalloc(1 * sizeof(float));;
			x1099[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1060, x1060));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 64, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1090, x1090));

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
							x1099, in_desc, x1066, filt_desc, x242,
							conv_desc, algo, ws_data, ws_size,
							x1097, out_desc, x1096));
			};
			float* x1102 = (float*)myMalloc(1 * sizeof(float));;
			x1102[0] = 1.0f;
			float* x1104 = (float*)myMalloc(1 * sizeof(float));;
			x1104[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1090, x1090));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x1102, bias_desc, x215, x1104, out_desc, x1096));
			};
			float* x1107 = (float*)myGpuMalloc(x1095 * sizeof(float));
			float* x1108 = (float*)myMalloc(1 * sizeof(float));;
			x1108[0] = 0.0f;
			float* x1110 = (float*)myMalloc(1 * sizeof(float));;
			x1110[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1090, x1090));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1110, x_desc, x1096, x1108, x_desc, x1096));
			};
			if (x1115) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1128 = (float*)myGpuMalloc(x1127 * sizeof(float));
			float* x1129 = (float*)myMalloc(1 * sizeof(float));;
			x1129[0] = 0.0f;
			float* x1131 = (float*)myMalloc(1 * sizeof(float));;
			x1131[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1060, x1060));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 64, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1122, x1122));

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
							x1131, in_desc, x1066, filt_desc, x179,
							conv_desc, algo, ws_data, ws_size,
							x1129, out_desc, x1128));
			};
			float* x1134 = (float*)myMalloc(1 * sizeof(float));;
			x1134[0] = 1.0f;
			float* x1136 = (float*)myMalloc(1 * sizeof(float));;
			x1136[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1122, x1122));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x1134, bias_desc, x134, x1136, out_desc, x1128));
			};
			float* x1139 = (float*)myGpuMalloc(x1127 * sizeof(float));
			float* x1140 = (float*)myMalloc(1 * sizeof(float));;
			x1140[0] = 0.0f;
			float* x1142 = (float*)myMalloc(1 * sizeof(float));;
			x1142[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1122, x1122));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1142, x_desc, x1128, x1140, x_desc, x1128));
			};
			if (x1148) {
			} else {
				printf("all dimensions except the concatenation dimension should be the same\n");
				assert(false && "");
			}
			// back prop for concat
			float* x1159 = (float*)myGpuMalloc(x1158 * sizeof(float));
			{
				dim3 grid(28, 2);
				concat2D_1D_greg<<<grid, 512>>>(x1096, 256, x1093, x1128, 256, x1125, x1159, 1, 64, 512, x1090, x1090, x1155, x1091, x1090, 1);
			};
			float* x1161 = (float*)myGpuMalloc(x1158 * sizeof(float));
			float* x1162 = (float*)myMalloc(1 * sizeof(float));;
			x1162[0] = 0.0f;
			float* x1164 = (float*)myMalloc(1 * sizeof(float));;
			x1164[0] = 1.0f;
			float* x1174 = (float*)myGpuMalloc(x1173 * sizeof(float));

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1090, x1090) );

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1168, x1168));

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
							x1164, in_desc, x1159, x1162, out_desc, x1174));
			};
			float* x1176 = (float*)myGpuMalloc(x1173 * sizeof(float));
			if (x1178) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1190 = (float*)myGpuMalloc(x1189 * sizeof(float));
			float* x1191 = (float*)myMalloc(1 * sizeof(float));;
			x1191[0] = 0.0f;
			float* x1193 = (float*)myMalloc(1 * sizeof(float));;
			x1193[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1168, x1168));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 512, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1184, x1184));

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
							x1193, in_desc, x1174, filt_desc, x98,
							conv_desc, algo, ws_data, ws_size,
							x1191, out_desc, x1190));
			};
			float* x1196 = (float*)myMalloc(1 * sizeof(float));;
			x1196[0] = 1.0f;
			float* x1198 = (float*)myMalloc(1 * sizeof(float));;
			x1198[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1184, x1184));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x1196, bias_desc, x155, x1198, out_desc, x1190));
			};
			float* x1201 = (float*)myGpuMalloc(x1189 * sizeof(float));
			float* x1202 = (float*)myMalloc(1 * sizeof(float));;
			x1202[0] = 0.0f;
			float* x1204 = (float*)myMalloc(1 * sizeof(float));;
			x1204[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1184, x1184));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1204, x_desc, x1190, x1202, x_desc, x1190));
			};
			if (x1208) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1220 = (float*)myGpuMalloc(x1219 * sizeof(float));
			float* x1221 = (float*)myMalloc(1 * sizeof(float));;
			x1221[0] = 0.0f;
			float* x1223 = (float*)myMalloc(1 * sizeof(float));;
			x1223[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1184, x1184));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 64, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1214, x1214));

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
							x1223, in_desc, x1190, filt_desc, x209,
							conv_desc, algo, ws_data, ws_size,
							x1221, out_desc, x1220));
			};
			float* x1226 = (float*)myMalloc(1 * sizeof(float));;
			x1226[0] = 1.0f;
			float* x1228 = (float*)myMalloc(1 * sizeof(float));;
			x1228[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1214, x1214));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x1226, bias_desc, x173, x1228, out_desc, x1220));
			};
			float* x1231 = (float*)myGpuMalloc(x1219 * sizeof(float));
			float* x1232 = (float*)myMalloc(1 * sizeof(float));;
			x1232[0] = 0.0f;
			float* x1234 = (float*)myMalloc(1 * sizeof(float));;
			x1234[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1214, x1214));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1234, x_desc, x1220, x1232, x_desc, x1220));
			};
			if (x1239) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1252 = (float*)myGpuMalloc(x1251 * sizeof(float));
			float* x1253 = (float*)myMalloc(1 * sizeof(float));;
			x1253[0] = 0.0f;
			float* x1255 = (float*)myMalloc(1 * sizeof(float));;
			x1255[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1184, x1184));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 64, 3, 3));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1246, x1246));

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
							x1255, in_desc, x1190, filt_desc, x185,
							conv_desc, algo, ws_data, ws_size,
							x1253, out_desc, x1252));
			};
			float* x1258 = (float*)myMalloc(1 * sizeof(float));;
			x1258[0] = 1.0f;
			float* x1260 = (float*)myMalloc(1 * sizeof(float));;
			x1260[0] = 1.0f;

			{
				cudnnTensorDescriptor_t bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1246, x1246));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x1258, bias_desc, x146, x1260, out_desc, x1252));
			};
			float* x1263 = (float*)myGpuMalloc(x1251 * sizeof(float));
			float* x1264 = (float*)myMalloc(1 * sizeof(float));;
			x1264[0] = 0.0f;
			float* x1266 = (float*)myMalloc(1 * sizeof(float));;
			x1266[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1246, x1246));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationForward(
							cudnnHandle, act_desc,
							x1266, x_desc, x1252, x1264, x_desc, x1252));
			};
			if (x1272) {
			} else {
				printf("all dimensions except the concatenation dimension should be the same\n");
				assert(false && "");
			}
			// back prop for concat
			float* x1283 = (float*)myGpuMalloc(x1282 * sizeof(float));
			{
				dim3 grid(28, 2);
				concat2D_1D_greg<<<grid, 512>>>(x1220, 256, x1217, x1252, 256, x1249, x1283, 1, 64, 512, x1214, x1214, x1279, x1215, x1214, 1);
			};
			float* x1285 = (float*)myGpuMalloc(x1282 * sizeof(float));
			if (x1287) {
			} else {
				assert(false && "ERROR not specified");
			}
			float* x1300 = (float*)myGpuMalloc(x1299 * sizeof(float));
			float* x1301 = (float*)myMalloc(1 * sizeof(float));;
			x1301[0] = 0.0f;
			float* x1303 = (float*)myMalloc(1 * sizeof(float));;
			x1303[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1214, x1214));

				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							10, 512, 4, 4));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 10, x1294, x1294));

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
							x1303, in_desc, x1283, filt_desc, x107,
							conv_desc, algo, ws_data, ws_size,
							x1301, out_desc, x1300));
			};
			float* x1306 = (float*)myMalloc(1 * sizeof(float));;
			x1306[0] = 1.0f;
			float* x1308 = (float*)myMalloc(1 * sizeof(float));;
			x1308[0] = 1.0f;

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
							64, 10, x1294, x1294));

				CUDNN_CALL(cudnnAddTensor(
							cudnnHandle, x1306, bias_desc, x248, x1308, out_desc, x1300));
			};
			float* x1311 = (float*)myGpuMalloc(x1299 * sizeof(float));
			float* x1312 = (float*)myMalloc(1 * sizeof(float));;
			x1312[0] = 0.0f;
			float* x1314 = (float*)myMalloc(1 * sizeof(float));;
			x1314[0] = 1.0f;
			float* x1316 = (float*)myGpuMalloc(640 * sizeof(float));

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 10, 1, 1));
				CUDNN_CALL(cudnnSoftmaxForward(
							cudnnHandle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL,
							x1314, x_desc, x1300, x1312, x_desc, x1316));
			};
			float* x1318 = (float*)myGpuMalloc(640 * sizeof(float));
			float* x1319 = (float*)myGpuMalloc(64 * sizeof(float));
			nllLoss<<<64, 1>>>(x1316, 10, x1319, x330);
			float* x1321 = (float*)myGpuMalloc(64 * sizeof(float));
			float* x1322 = (float*)myGpuMalloc(1 * sizeof(float));
			float* x1323 = (float*)myMalloc(1 * sizeof(float));;
			x1323[0] = 0.0f;
			float* x1325 = (float*)myMalloc(1 * sizeof(float));;
			x1325[0] = 1.0f;

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
							x1325, x_desc, x1319, x1323, out_desc, x1322));
			};
			float* x1328 = (float*)myGpuMalloc(1 * sizeof(float));
			// make sure the size of loss is 1
			arrayFill_greg<<<28, 512>>>(x1328, 1.0f, 1);
			// backend is lantern.TensorDslCudnn$BackendCudnn@41658fc0
			CUDA_CALL(cudaMemcpy(x335, x1322, 1 * sizeof(float), cudaMemcpyDeviceToHost));
			// 'mean' gradient
			// backprop for mean op
			if (x1337) {
			} else {
				assert(false && "bias shape should be equal to res or be 1, got bias:  x Const(1) x Const(1) x Const(1) x Const(1), res:  x Const(64) x Const(1) x Const(1) x Const(1)");
			}
			float* x1342 = (float*)myMalloc(1 * sizeof(float));;
			x1342[0] = x1335;
			float* x1344 = (float*)myMalloc(1 * sizeof(float));;
			x1344[0] = 1.0f;

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
							cudnnHandle, x1342, bias_desc, x1328, x1344, out_desc, x1321));
			};
			// 'nllLossB' gradient.
			nllLoss_grad<<<64, 1>>>(10, x1321, x330, x1318);
			float* x1349 = (float*)myMalloc(1 * sizeof(float));;
			x1349[0] = 1.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 10, 1, 1));
				CUDNN_CALL(cudnnSoftmaxBackward(
							cudnnHandle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL,
							x1349, x_desc, x1316, x_desc, x1318,
							x1349, x_desc, x1311));
			};
			// conv2D back-propagate
			float* x1353 = (float*)myMalloc(1 * sizeof(float));;
			x1353[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							10, 512, 4, 4));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1214, x1214));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 10, x1294, x1294));

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
							x1353, filt_desc, x107, grad_out_desc, x1311,
							conv_desc, algo, ws_data, ws_size,
							x1353, grad_in_desc, x1285));
			};
			float* x1356 = (float*)myMalloc(1 * sizeof(float));;
			x1356[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							10, 512, 4, 4));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 10, x1294, x1294));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1214, x1214));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1356, in_desc, x1283, grad_out_desc, x1311,
							conv_desc, algo, ws_data, ws_size,
							x1356, grad_filt_desc, x256));
			};
			float* x1359 = (float*)myMalloc(1 * sizeof(float));;
			x1359[0] = 1.0f;

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
							64, 10, x1294, x1294));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1359, grad_out_desc, x1311,
							x1359, grad_bias_desc, x303));
			};
			{
				dim3 grid(28, 2);
				concat2D_1D_greg_grad<<<grid, 512>>>(x1231, 256, x1217, x1263, 256, x1249, x1285, 1, 64, 512, x1214, x1214, x1279, x1215, x1214, 1);
			};
			float* x1363 = (float*)myMalloc(1 * sizeof(float));;
			x1363[0] = 1.0f;
			float* x1365 = (float*)myMalloc(1 * sizeof(float));;
			x1365[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1246, x1246));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1363, x_desc, x1252, x_desc, x1263, x_desc, x1252,
							x1365, x_desc, x1263));
			};
			// conv2D back-propagate
			float* x1369 = (float*)myMalloc(1 * sizeof(float));;
			x1369[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 64, 3, 3));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1184, x1184));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1246, x1246));

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
							x1369, filt_desc, x185, grad_out_desc, x1263,
							conv_desc, algo, ws_data, ws_size,
							x1369, grad_in_desc, x1201));
			};
			float* x1372 = (float*)myMalloc(1 * sizeof(float));;
			x1372[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 64, 3, 3));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1246, x1246));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1184, x1184));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1372, in_desc, x1190, grad_out_desc, x1263,
							conv_desc, algo, ws_data, ws_size,
							x1372, grad_filt_desc, x282));
			};
			float* x1375 = (float*)myMalloc(1 * sizeof(float));;
			x1375[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1246, x1246));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1375, grad_out_desc, x1263,
							x1375, grad_bias_desc, x269));
			};
			float* x1378 = (float*)myMalloc(1 * sizeof(float));;
			x1378[0] = 1.0f;
			float* x1380 = (float*)myMalloc(1 * sizeof(float));;
			x1380[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1214, x1214));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1378, x_desc, x1220, x_desc, x1231, x_desc, x1220,
							x1380, x_desc, x1231));
			};
			// conv2D back-propagate
			float* x1384 = (float*)myMalloc(1 * sizeof(float));;
			x1384[0] = 1.0f;

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
							64, 64, x1184, x1184));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1214, x1214));

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
							x1384, filt_desc, x209, grad_out_desc, x1231,
							conv_desc, algo, ws_data, ws_size,
							x1384, grad_in_desc, x1201));
			};
			float* x1387 = (float*)myMalloc(1 * sizeof(float));;
			x1387[0] = 1.0f;

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
							64, 256, x1214, x1214));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1184, x1184));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1387, in_desc, x1190, grad_out_desc, x1231,
							conv_desc, algo, ws_data, ws_size,
							x1387, grad_filt_desc, x290));
			};
			float* x1390 = (float*)myMalloc(1 * sizeof(float));;
			x1390[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1214, x1214));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1390, grad_out_desc, x1231,
							x1390, grad_bias_desc, x278));
			};
			float* x1393 = (float*)myMalloc(1 * sizeof(float));;
			x1393[0] = 1.0f;
			float* x1395 = (float*)myMalloc(1 * sizeof(float));;
			x1395[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1184, x1184));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1393, x_desc, x1190, x_desc, x1201, x_desc, x1190,
							x1395, x_desc, x1201));
			};
			// conv2D back-propagate
			float* x1399 = (float*)myMalloc(1 * sizeof(float));;
			x1399[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 512, 1, 1));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1168, x1168));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1184, x1184));

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
							x1399, filt_desc, x98, grad_out_desc, x1201,
							conv_desc, algo, ws_data, ws_size,
							x1399, grad_in_desc, x1176));
			};
			float* x1402 = (float*)myMalloc(1 * sizeof(float));;
			x1402[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 512, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1184, x1184));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1168, x1168));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1402, in_desc, x1174, grad_out_desc, x1201,
							conv_desc, algo, ws_data, ws_size,
							x1402, grad_filt_desc, x253));
			};
			float* x1405 = (float*)myMalloc(1 * sizeof(float));;
			x1405[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1184, x1184));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1405, grad_out_desc, x1201,
							x1405, grad_bias_desc, x272));
			};
			float* x1408 = (float*)myMalloc(1 * sizeof(float));;
			x1408[0] = 0.0f;
			float* x1410 = (float*)myMalloc(1 * sizeof(float));;
			x1410[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1090, x1090));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 512, x1168, x1168));

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
							x1410, out_desc, x1174, out_desc, x1176, in_desc, x1159  , x1408, in_desc, x1161));
			};
			{
				dim3 grid(28, 2);
				concat2D_1D_greg_grad<<<grid, 512>>>(x1107, 256, x1093, x1139, 256, x1125, x1161, 1, 64, 512, x1090, x1090, x1155, x1091, x1090, 1);
			};
			float* x1414 = (float*)myMalloc(1 * sizeof(float));;
			x1414[0] = 1.0f;
			float* x1416 = (float*)myMalloc(1 * sizeof(float));;
			x1416[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1122, x1122));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1414, x_desc, x1128, x_desc, x1139, x_desc, x1128,
							x1416, x_desc, x1139));
			};
			// conv2D back-propagate
			float* x1420 = (float*)myMalloc(1 * sizeof(float));;
			x1420[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 64, 3, 3));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1060, x1060));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1122, x1122));

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
							x1420, filt_desc, x179, grad_out_desc, x1139,
							conv_desc, algo, ws_data, ws_size,
							x1420, grad_in_desc, x1077));
			};
			float* x1423 = (float*)myMalloc(1 * sizeof(float));;
			x1423[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							256, 64, 3, 3));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1122, x1122));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1060, x1060));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1423, in_desc, x1066, grad_out_desc, x1139,
							conv_desc, algo, ws_data, ws_size,
							x1423, grad_filt_desc, x280));
			};
			float* x1426 = (float*)myMalloc(1 * sizeof(float));;
			x1426[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1122, x1122));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1426, grad_out_desc, x1139,
							x1426, grad_bias_desc, x265));
			};
			float* x1429 = (float*)myMalloc(1 * sizeof(float));;
			x1429[0] = 1.0f;
			float* x1431 = (float*)myMalloc(1 * sizeof(float));;
			x1431[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1090, x1090));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1429, x_desc, x1096, x_desc, x1107, x_desc, x1096,
							x1431, x_desc, x1107));
			};
			// conv2D back-propagate
			float* x1435 = (float*)myMalloc(1 * sizeof(float));;
			x1435[0] = 1.0f;

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
							64, 64, x1060, x1060));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1090, x1090));

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
							x1435, filt_desc, x242, grad_out_desc, x1107,
							conv_desc, algo, ws_data, ws_size,
							x1435, grad_in_desc, x1077));
			};
			float* x1438 = (float*)myMalloc(1 * sizeof(float));;
			x1438[0] = 1.0f;

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
							64, 256, x1090, x1090));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1060, x1060));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1438, in_desc, x1066, grad_out_desc, x1107,
							conv_desc, algo, ws_data, ws_size,
							x1438, grad_filt_desc, x301));
			};
			float* x1441 = (float*)myMalloc(1 * sizeof(float));;
			x1441[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 256, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x1090, x1090));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1441, grad_out_desc, x1107,
							x1441, grad_bias_desc, x292));
			};
			float* x1444 = (float*)myMalloc(1 * sizeof(float));;
			x1444[0] = 1.0f;
			float* x1446 = (float*)myMalloc(1 * sizeof(float));;
			x1446[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1060, x1060));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1444, x_desc, x1066, x_desc, x1077, x_desc, x1066,
							x1446, x_desc, x1077));
			};
			// conv2D back-propagate
			float* x1450 = (float*)myMalloc(1 * sizeof(float));;
			x1450[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 384, 1, 1));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 384, x981, x981));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1060, x1060));

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
							x1450, filt_desc, x197, grad_out_desc, x1077,
							conv_desc, algo, ws_data, ws_size,
							x1450, grad_in_desc, x1052));
			};
			float* x1453 = (float*)myMalloc(1 * sizeof(float));;
			x1453[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 384, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1060, x1060));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 384, x981, x981));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1453, in_desc, x1050, grad_out_desc, x1077,
							conv_desc, algo, ws_data, ws_size,
							x1453, grad_filt_desc, x286));
			};
			float* x1456 = (float*)myMalloc(1 * sizeof(float));;
			x1456[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x1060, x1060));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1456, grad_out_desc, x1077,
							x1456, grad_bias_desc, x261));
			};
			{
				dim3 grid(28, 2);
				concat2D_1D_greg_grad<<<grid, 512>>>(x998, 192, x984, x1030, 192, x1016, x1052, 1, 64, 384, x981, x981, x1046, x982, x981, 1);
			};
			float* x1460 = (float*)myMalloc(1 * sizeof(float));;
			x1460[0] = 1.0f;
			float* x1462 = (float*)myMalloc(1 * sizeof(float));;
			x1462[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x1013, x1013));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1460, x_desc, x1019, x_desc, x1030, x_desc, x1019,
							x1462, x_desc, x1030));
			};
			// conv2D back-propagate
			float* x1466 = (float*)myMalloc(1 * sizeof(float));;
			x1466[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							192, 48, 3, 3));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x951, x951));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x1013, x1013));

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
							x1466, filt_desc, x149, grad_out_desc, x1030,
							conv_desc, algo, ws_data, ws_size,
							x1466, grad_in_desc, x968));
			};
			float* x1469 = (float*)myMalloc(1 * sizeof(float));;
			x1469[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							192, 48, 3, 3));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x1013, x1013));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x951, x951));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1469, in_desc, x957, grad_out_desc, x1030,
							conv_desc, algo, ws_data, ws_size,
							x1469, grad_filt_desc, x270));
			};
			float* x1472 = (float*)myMalloc(1 * sizeof(float));;
			x1472[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 192, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x1013, x1013));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1472, grad_out_desc, x1030,
							x1472, grad_bias_desc, x296));
			};
			float* x1475 = (float*)myMalloc(1 * sizeof(float));;
			x1475[0] = 1.0f;
			float* x1477 = (float*)myMalloc(1 * sizeof(float));;
			x1477[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x981, x981));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1475, x_desc, x987, x_desc, x998, x_desc, x987,
							x1477, x_desc, x998));
			};
			// conv2D back-propagate
			float* x1481 = (float*)myMalloc(1 * sizeof(float));;
			x1481[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							192, 48, 1, 1));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x951, x951));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x981, x981));

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
							x1481, filt_desc, x161, grad_out_desc, x998,
							conv_desc, algo, ws_data, ws_size,
							x1481, grad_in_desc, x968));
			};
			float* x1484 = (float*)myMalloc(1 * sizeof(float));;
			x1484[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							192, 48, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x981, x981));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x951, x951));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1484, in_desc, x957, grad_out_desc, x998,
							conv_desc, algo, ws_data, ws_size,
							x1484, grad_filt_desc, x274));
			};
			float* x1487 = (float*)myMalloc(1 * sizeof(float));;
			x1487[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 192, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x981, x981));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1487, grad_out_desc, x998,
							x1487, grad_bias_desc, x284));
			};
			float* x1490 = (float*)myMalloc(1 * sizeof(float));;
			x1490[0] = 1.0f;
			float* x1492 = (float*)myMalloc(1 * sizeof(float));;
			x1492[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x951, x951));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1490, x_desc, x957, x_desc, x968, x_desc, x957,
							x1492, x_desc, x968));
			};
			// conv2D back-propagate
			float* x1496 = (float*)myMalloc(1 * sizeof(float));;
			x1496[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							48, 384, 1, 1));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 384, x872, x872));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x951, x951));

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
							x1496, filt_desc, x137, grad_out_desc, x968,
							conv_desc, algo, ws_data, ws_size,
							x1496, grad_in_desc, x943));
			};
			float* x1499 = (float*)myMalloc(1 * sizeof(float));;
			x1499[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							48, 384, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x951, x951));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 384, x872, x872));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1499, in_desc, x941, grad_out_desc, x968,
							conv_desc, algo, ws_data, ws_size,
							x1499, grad_filt_desc, x266));
			};
			float* x1502 = (float*)myMalloc(1 * sizeof(float));;
			x1502[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 48, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x951, x951));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1502, grad_out_desc, x968,
							x1502, grad_bias_desc, x254));
			};
			{
				dim3 grid(28, 2);
				concat2D_1D_greg_grad<<<grid, 512>>>(x889, 192, x875, x921, 192, x907, x943, 1, 64, 384, x872, x872, x937, x873, x872, 1);
			};
			float* x1506 = (float*)myMalloc(1 * sizeof(float));;
			x1506[0] = 1.0f;
			float* x1508 = (float*)myMalloc(1 * sizeof(float));;
			x1508[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x904, x904));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1506, x_desc, x910, x_desc, x921, x_desc, x910,
							x1508, x_desc, x921));
			};
			// conv2D back-propagate
			float* x1512 = (float*)myMalloc(1 * sizeof(float));;
			x1512[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							192, 48, 3, 3));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x842, x842));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x904, x904));

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
							x1512, filt_desc, x212, grad_out_desc, x921,
							conv_desc, algo, ws_data, ws_size,
							x1512, grad_in_desc, x859));
			};
			float* x1515 = (float*)myMalloc(1 * sizeof(float));;
			x1515[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							192, 48, 3, 3));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x904, x904));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x842, x842));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1515, in_desc, x848, grad_out_desc, x921,
							conv_desc, algo, ws_data, ws_size,
							x1515, grad_filt_desc, x291));
			};
			float* x1518 = (float*)myMalloc(1 * sizeof(float));;
			x1518[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 192, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x904, x904));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1518, grad_out_desc, x921,
							x1518, grad_bias_desc, x281));
			};
			float* x1521 = (float*)myMalloc(1 * sizeof(float));;
			x1521[0] = 1.0f;
			float* x1523 = (float*)myMalloc(1 * sizeof(float));;
			x1523[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x872, x872));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1521, x_desc, x878, x_desc, x889, x_desc, x878,
							x1523, x_desc, x889));
			};
			// conv2D back-propagate
			float* x1527 = (float*)myMalloc(1 * sizeof(float));;
			x1527[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							192, 48, 1, 1));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x842, x842));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x872, x872));

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
							x1527, filt_desc, x239, grad_out_desc, x889,
							conv_desc, algo, ws_data, ws_size,
							x1527, grad_in_desc, x859));
			};
			float* x1530 = (float*)myMalloc(1 * sizeof(float));;
			x1530[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							192, 48, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x872, x872));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x842, x842));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1530, in_desc, x848, grad_out_desc, x889,
							conv_desc, algo, ws_data, ws_size,
							x1530, grad_filt_desc, x300));
			};
			float* x1533 = (float*)myMalloc(1 * sizeof(float));;
			x1533[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 192, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 192, x872, x872));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1533, grad_out_desc, x889,
							x1533, grad_bias_desc, x298));
			};
			float* x1536 = (float*)myMalloc(1 * sizeof(float));;
			x1536[0] = 1.0f;
			float* x1538 = (float*)myMalloc(1 * sizeof(float));;
			x1538[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x842, x842));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1536, x_desc, x848, x_desc, x859, x_desc, x848,
							x1538, x_desc, x859));
			};
			// conv2D back-propagate
			float* x1542 = (float*)myMalloc(1 * sizeof(float));;
			x1542[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							48, 256, 1, 1));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x763, x763));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x842, x842));

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
							x1542, filt_desc, x221, grad_out_desc, x859,
							conv_desc, algo, ws_data, ws_size,
							x1542, grad_in_desc, x834));
			};
			float* x1545 = (float*)myMalloc(1 * sizeof(float));;
			x1545[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							48, 256, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x842, x842));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x763, x763));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1545, in_desc, x832, grad_out_desc, x859,
							conv_desc, algo, ws_data, ws_size,
							x1545, grad_filt_desc, x294));
			};
			float* x1548 = (float*)myMalloc(1 * sizeof(float));;
			x1548[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 48, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 48, x842, x842));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1548, grad_out_desc, x859,
							x1548, grad_bias_desc, x304));
			};
			{
				dim3 grid(28, 2);
				concat2D_1D_greg_grad<<<grid, 512>>>(x780, 128, x766, x812, 128, x798, x834, 1, 64, 256, x763, x763, x828, x764, x763, 1);
			};
			float* x1552 = (float*)myMalloc(1 * sizeof(float));;
			x1552[0] = 1.0f;
			float* x1554 = (float*)myMalloc(1 * sizeof(float));;
			x1554[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x795, x795));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1552, x_desc, x801, x_desc, x812, x_desc, x801,
							x1554, x_desc, x812));
			};
			// conv2D back-propagate
			float* x1558 = (float*)myMalloc(1 * sizeof(float));;
			x1558[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							128, 32, 3, 3));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x733, x733));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x795, x795));

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
							x1558, filt_desc, x203, grad_out_desc, x812,
							conv_desc, algo, ws_data, ws_size,
							x1558, grad_in_desc, x750));
			};
			float* x1561 = (float*)myMalloc(1 * sizeof(float));;
			x1561[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							128, 32, 3, 3));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x795, x795));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x733, x733));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1561, in_desc, x739, grad_out_desc, x812,
							conv_desc, algo, ws_data, ws_size,
							x1561, grad_filt_desc, x288));
			};
			float* x1564 = (float*)myMalloc(1 * sizeof(float));;
			x1564[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x795, x795));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1564, grad_out_desc, x812,
							x1564, grad_bias_desc, x268));
			};
			float* x1567 = (float*)myMalloc(1 * sizeof(float));;
			x1567[0] = 1.0f;
			float* x1569 = (float*)myMalloc(1 * sizeof(float));;
			x1569[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x763, x763));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1567, x_desc, x769, x_desc, x780, x_desc, x769,
							x1569, x_desc, x780));
			};
			// conv2D back-propagate
			float* x1573 = (float*)myMalloc(1 * sizeof(float));;
			x1573[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							128, 32, 1, 1));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x733, x733));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x763, x763));

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
							x1573, filt_desc, x116, grad_out_desc, x780,
							conv_desc, algo, ws_data, ws_size,
							x1573, grad_in_desc, x750));
			};
			float* x1576 = (float*)myMalloc(1 * sizeof(float));;
			x1576[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							128, 32, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x763, x763));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x733, x733));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1576, in_desc, x739, grad_out_desc, x780,
							conv_desc, algo, ws_data, ws_size,
							x1576, grad_filt_desc, x259));
			};
			float* x1579 = (float*)myMalloc(1 * sizeof(float));;
			x1579[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x763, x763));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1579, grad_out_desc, x780,
							x1579, grad_bias_desc, x273));
			};
			float* x1582 = (float*)myMalloc(1 * sizeof(float));;
			x1582[0] = 1.0f;
			float* x1584 = (float*)myMalloc(1 * sizeof(float));;
			x1584[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x733, x733));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1582, x_desc, x739, x_desc, x750, x_desc, x739,
							x1584, x_desc, x750));
			};
			// conv2D back-propagate
			float* x1588 = (float*)myMalloc(1 * sizeof(float));;
			x1588[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							32, 256, 1, 1));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x717, x717));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x733, x733));

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
							x1588, filt_desc, x176, grad_out_desc, x750,
							conv_desc, algo, ws_data, ws_size,
							x1588, grad_in_desc, x725));
			};
			float* x1591 = (float*)myMalloc(1 * sizeof(float));;
			x1591[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							32, 256, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x733, x733));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x717, x717));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1591, in_desc, x723, grad_out_desc, x750,
							conv_desc, algo, ws_data, ws_size,
							x1591, grad_filt_desc, x279));
			};
			float* x1594 = (float*)myMalloc(1 * sizeof(float));;
			x1594[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 32, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x733, x733));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1594, grad_out_desc, x750,
							x1594, grad_bias_desc, x267));
			};
			float* x1597 = (float*)myMalloc(1 * sizeof(float));;
			x1597[0] = 0.0f;
			float* x1599 = (float*)myMalloc(1 * sizeof(float));;
			x1599[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x639, x639));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 256, x717, x717));

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
							x1599, out_desc, x723, out_desc, x725, in_desc, x708  , x1597, in_desc, x710));
			};
			{
				dim3 grid(28, 2);
				concat2D_1D_greg_grad<<<grid, 512>>>(x656, 128, x642, x688, 128, x674, x710, 1, 64, 256, x639, x639, x704, x640, x639, 1);
			};
			float* x1603 = (float*)myMalloc(1 * sizeof(float));;
			x1603[0] = 1.0f;
			float* x1605 = (float*)myMalloc(1 * sizeof(float));;
			x1605[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x671, x671));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1603, x_desc, x677, x_desc, x688, x_desc, x677,
							x1605, x_desc, x688));
			};
			// conv2D back-propagate
			float* x1609 = (float*)myMalloc(1 * sizeof(float));;
			x1609[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							128, 32, 3, 3));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x609, x609));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x671, x671));

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
							x1609, filt_desc, x113, grad_out_desc, x688,
							conv_desc, algo, ws_data, ws_size,
							x1609, grad_in_desc, x626));
			};
			float* x1612 = (float*)myMalloc(1 * sizeof(float));;
			x1612[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							128, 32, 3, 3));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x671, x671));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x609, x609));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1612, in_desc, x615, grad_out_desc, x688,
							conv_desc, algo, ws_data, ws_size,
							x1612, grad_filt_desc, x258));
			};
			float* x1615 = (float*)myMalloc(1 * sizeof(float));;
			x1615[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x671, x671));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1615, grad_out_desc, x688,
							x1615, grad_bias_desc, x293));
			};
			float* x1618 = (float*)myMalloc(1 * sizeof(float));;
			x1618[0] = 1.0f;
			float* x1620 = (float*)myMalloc(1 * sizeof(float));;
			x1620[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x639, x639));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1618, x_desc, x645, x_desc, x656, x_desc, x645,
							x1620, x_desc, x656));
			};
			// conv2D back-propagate
			float* x1624 = (float*)myMalloc(1 * sizeof(float));;
			x1624[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							128, 32, 1, 1));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x609, x609));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x639, x639));

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
							x1624, filt_desc, x200, grad_out_desc, x656,
							conv_desc, algo, ws_data, ws_size,
							x1624, grad_in_desc, x626));
			};
			float* x1627 = (float*)myMalloc(1 * sizeof(float));;
			x1627[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							128, 32, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x639, x639));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x609, x609));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1627, in_desc, x615, grad_out_desc, x656,
							conv_desc, algo, ws_data, ws_size,
							x1627, grad_filt_desc, x287));
			};
			float* x1630 = (float*)myMalloc(1 * sizeof(float));;
			x1630[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 128, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x639, x639));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1630, grad_out_desc, x656,
							x1630, grad_bias_desc, x297));
			};
			float* x1633 = (float*)myMalloc(1 * sizeof(float));;
			x1633[0] = 1.0f;
			float* x1635 = (float*)myMalloc(1 * sizeof(float));;
			x1635[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x609, x609));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1633, x_desc, x615, x_desc, x626, x_desc, x615,
							x1635, x_desc, x626));
			};
			// conv2D back-propagate
			float* x1639 = (float*)myMalloc(1 * sizeof(float));;
			x1639[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							32, 128, 1, 1));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x530, x530));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x609, x609));

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
							x1639, filt_desc, x125, grad_out_desc, x626,
							conv_desc, algo, ws_data, ws_size,
							x1639, grad_in_desc, x601));
			};
			float* x1642 = (float*)myMalloc(1 * sizeof(float));;
			x1642[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							32, 128, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x609, x609));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x530, x530));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1642, in_desc, x599, grad_out_desc, x626,
							conv_desc, algo, ws_data, ws_size,
							x1642, grad_filt_desc, x262));
			};
			float* x1645 = (float*)myMalloc(1 * sizeof(float));;
			x1645[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 32, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 32, x609, x609));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1645, grad_out_desc, x626,
							x1645, grad_bias_desc, x275));
			};
			{
				dim3 grid(28, 2);
				concat2D_1D_greg_grad<<<grid, 512>>>(x547, 64, x533, x579, 64, x565, x601, 1, 64, 128, x530, x530, x595, x531, x530, 1);
			};
			float* x1649 = (float*)myMalloc(1 * sizeof(float));;
			x1649[0] = 1.0f;
			float* x1651 = (float*)myMalloc(1 * sizeof(float));;
			x1651[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x562, x562));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1649, x_desc, x568, x_desc, x579, x_desc, x568,
							x1651, x_desc, x579));
			};
			// conv2D back-propagate
			float* x1655 = (float*)myMalloc(1 * sizeof(float));;
			x1655[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 16, 3, 3));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x500, x500));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x562, x562));

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
							x1655, filt_desc, x152, grad_out_desc, x579,
							conv_desc, algo, ws_data, ws_size,
							x1655, grad_in_desc, x517));
			};
			float* x1658 = (float*)myMalloc(1 * sizeof(float));;
			x1658[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 16, 3, 3));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x562, x562));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x500, x500));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1658, in_desc, x506, grad_out_desc, x579,
							conv_desc, algo, ws_data, ws_size,
							x1658, grad_filt_desc, x271));
			};
			float* x1661 = (float*)myMalloc(1 * sizeof(float));;
			x1661[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x562, x562));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1661, grad_out_desc, x579,
							x1661, grad_bias_desc, x289));
			};
			float* x1664 = (float*)myMalloc(1 * sizeof(float));;
			x1664[0] = 1.0f;
			float* x1666 = (float*)myMalloc(1 * sizeof(float));;
			x1666[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x530, x530));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1664, x_desc, x536, x_desc, x547, x_desc, x536,
							x1666, x_desc, x547));
			};
			// conv2D back-propagate
			float* x1670 = (float*)myMalloc(1 * sizeof(float));;
			x1670[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 16, 1, 1));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x500, x500));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x530, x530));

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
							x1670, filt_desc, x128, grad_out_desc, x547,
							conv_desc, algo, ws_data, ws_size,
							x1670, grad_in_desc, x517));
			};
			float* x1673 = (float*)myMalloc(1 * sizeof(float));;
			x1673[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 16, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x530, x530));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x500, x500));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1673, in_desc, x506, grad_out_desc, x547,
							conv_desc, algo, ws_data, ws_size,
							x1673, grad_filt_desc, x263));
			};
			float* x1676 = (float*)myMalloc(1 * sizeof(float));;
			x1676[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x530, x530));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1676, grad_out_desc, x547,
							x1676, grad_bias_desc, x255));
			};
			float* x1679 = (float*)myMalloc(1 * sizeof(float));;
			x1679[0] = 1.0f;
			float* x1681 = (float*)myMalloc(1 * sizeof(float));;
			x1681[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x500, x500));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1679, x_desc, x506, x_desc, x517, x_desc, x506,
							x1681, x_desc, x517));
			};
			// conv2D back-propagate
			float* x1685 = (float*)myMalloc(1 * sizeof(float));;
			x1685[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							16, 128, 1, 1));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x418, x418));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x500, x500));

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
							x1685, filt_desc, x131, grad_out_desc, x517,
							conv_desc, algo, ws_data, ws_size,
							x1685, grad_in_desc, x492));
			};
			float* x1688 = (float*)myMalloc(1 * sizeof(float));;
			x1688[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							16, 128, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x500, x500));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 128, x418, x418));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1688, in_desc, x490, grad_out_desc, x517,
							conv_desc, algo, ws_data, ws_size,
							x1688, grad_filt_desc, x264));
			};
			float* x1691 = (float*)myMalloc(1 * sizeof(float));;
			x1691[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 16, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x500, x500));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1691, grad_out_desc, x517,
							x1691, grad_bias_desc, x277));
			};
			{
				dim3 grid(28, 2);
				concat2D_1D_greg_grad<<<grid, 512>>>(x435, 64, x421, x467, 64, x453, x492, 1, 64, 128, x418, x418, x486, x419, x418, 1);
			};
			float* x1695 = (float*)myMalloc(1 * sizeof(float));;
			x1695[0] = 1.0f;
			float* x1697 = (float*)myMalloc(1 * sizeof(float));;
			x1697[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x450, x450));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1695, x_desc, x456, x_desc, x467, x_desc, x456,
							x1697, x_desc, x467));
			};
			// conv2D back-propagate
			float* x1701 = (float*)myMalloc(1 * sizeof(float));;
			x1701[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 16, 3, 3));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x388, x388));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x450, x450));

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
							x1701, filt_desc, x236, grad_out_desc, x467,
							conv_desc, algo, ws_data, ws_size,
							x1701, grad_in_desc, x405));
			};
			float* x1704 = (float*)myMalloc(1 * sizeof(float));;
			x1704[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 16, 3, 3));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x450, x450));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x388, x388));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1704, in_desc, x394, grad_out_desc, x467,
							conv_desc, algo, ws_data, ws_size,
							x1704, grad_filt_desc, x299));
			};
			float* x1707 = (float*)myMalloc(1 * sizeof(float));;
			x1707[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x450, x450));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1707, grad_out_desc, x467,
							x1707, grad_bias_desc, x257));
			};
			float* x1710 = (float*)myMalloc(1 * sizeof(float));;
			x1710[0] = 1.0f;
			float* x1712 = (float*)myMalloc(1 * sizeof(float));;
			x1712[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x418, x418));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1710, x_desc, x424, x_desc, x435, x_desc, x424,
							x1712, x_desc, x435));
			};
			// conv2D back-propagate
			float* x1716 = (float*)myMalloc(1 * sizeof(float));;
			x1716[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 16, 1, 1));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x388, x388));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x418, x418));

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
							x1716, filt_desc, x167, grad_out_desc, x435,
							conv_desc, algo, ws_data, ws_size,
							x1716, grad_in_desc, x405));
			};
			float* x1719 = (float*)myMalloc(1 * sizeof(float));;
			x1719[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							64, 16, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x418, x418));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x388, x388));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1719, in_desc, x394, grad_out_desc, x435,
							conv_desc, algo, ws_data, ws_size,
							x1719, grad_filt_desc, x276));
			};
			float* x1722 = (float*)myMalloc(1 * sizeof(float));;
			x1722[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 64, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 64, x418, x418));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1722, grad_out_desc, x435,
							x1722, grad_bias_desc, x283));
			};
			float* x1725 = (float*)myMalloc(1 * sizeof(float));;
			x1725[0] = 1.0f;
			float* x1727 = (float*)myMalloc(1 * sizeof(float));;
			x1727[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x388, x388));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1725, x_desc, x394, x_desc, x405, x_desc, x394,
							x1727, x_desc, x405));
			};
			// conv2D back-propagate
			float* x1731 = (float*)myMalloc(1 * sizeof(float));;
			x1731[0] = 1.0f;

			{
				cudnnFilterDescriptor_t filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							16, 96, 1, 1));

				cudnnTensorDescriptor_t grad_in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 96, x372, x372));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x388, x388));

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
							x1731, filt_desc, x245, grad_out_desc, x405,
							conv_desc, algo, ws_data, ws_size,
							x1731, grad_in_desc, x380));
			};
			float* x1734 = (float*)myMalloc(1 * sizeof(float));;
			x1734[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							16, 96, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x388, x388));

				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 96, x372, x372));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1734, in_desc, x378, grad_out_desc, x405,
							conv_desc, algo, ws_data, ws_size,
							x1734, grad_filt_desc, x302));
			};
			float* x1737 = (float*)myMalloc(1 * sizeof(float));;
			x1737[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 16, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 16, x388, x388));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1737, grad_out_desc, x405,
							x1737, grad_bias_desc, x260));
			};
			float* x1740 = (float*)myMalloc(1 * sizeof(float));;
			x1740[0] = 0.0f;
			float* x1742 = (float*)myMalloc(1 * sizeof(float));;
			x1742[0] = 1.0f;

			{
				cudnnTensorDescriptor_t in_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 96, x343, x343));

				cudnnTensorDescriptor_t out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 96, x372, x372));

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
							x1742, out_desc, x378, out_desc, x380, in_desc, x349  , x1740, in_desc, x360));
			};
			float* x1745 = (float*)myMalloc(1 * sizeof(float));;
			x1745[0] = 1.0f;
			float* x1747 = (float*)myMalloc(1 * sizeof(float));;
			x1747[0] = 0.0f;

			{
				cudnnTensorDescriptor_t x_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 96, x343, x343));

				cudnnActivationDescriptor_t act_desc;
				CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
				CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
							/*mode=*/ CUDNN_ACTIVATION_RELU,
							/*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
							/*relu_coef=*/ 0));
				CUDNN_CALL(cudnnActivationBackward(
							cudnnHandle, act_desc,
							x1745, x_desc, x349, x_desc, x360, x_desc, x349,
							x1747, x_desc, x360));
			};
			// conv2D back-propagate
			float* x1751 = (float*)myMalloc(1 * sizeof(float));;
			x1751[0] = 1.0f;

			{
				cudnnFilterDescriptor_t grad_filt_desc;
				CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
				CUDNN_CALL(cudnnSetFilter4dDescriptor(
							grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
							96, 3, 3, 3));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 96, x343, x343));

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
				//algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
				// Workspace.
				size_t ws_size;
				CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
							cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
				void *ws_data = myGpuMalloc(ws_size);
				CUDNN_CALL(cudnnConvolutionBackwardFilter(
							cudnnHandle,
							x1751, in_desc, x327, grad_out_desc, x360,
							conv_desc, algo, ws_data, ws_size,
							x1751, grad_filt_desc, x285));
			};
			float* x1754 = (float*)myMalloc(1 * sizeof(float));;
			x1754[0] = 1.0f;

			{
				cudnnTensorDescriptor_t grad_bias_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							1, 96, 1, 1));

				cudnnTensorDescriptor_t grad_out_desc;
				CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
				CUDNN_CALL(cudnnSetTensor4dDescriptor(
							grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
							64, 96, x343, x343));

				CUDNN_CALL(cudnnConvolutionBackwardBias(
							cudnnHandle, x1754, grad_out_desc, x360,
							x1754, grad_bias_desc, x295));
			};
			float x1757 = x335[0];
			x315 += x1757;
			float* x1759 = (float*)myMalloc(1 * sizeof(float));;
			x1759[0] = 1.0f;
			float* x1761 = (float*)myMalloc(1 * sizeof(float));;
			x1761[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 512,64,x1759,x98,512,x1761, x253, 512, x98,512));
			arrayFill_greg<<<28, 512>>>(x253, 0.0f, 32768);
			float* x1765 = (float*)myMalloc(1 * sizeof(float));;
			x1765[0] = 1.0f;
			float* x1767 = (float*)myMalloc(1 * sizeof(float));;
			x1767[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,48,x1765,x101,1,x1767, x254, 1, x101,1));
			arrayFill_greg<<<28, 512>>>(x254, 0.0f, 48);
			float* x1771 = (float*)myMalloc(1 * sizeof(float));;
			x1771[0] = 1.0f;
			float* x1773 = (float*)myMalloc(1 * sizeof(float));;
			x1773[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x1771,x104,1,x1773, x255, 1, x104,1));
			arrayFill_greg<<<28, 512>>>(x255, 0.0f, 64);
			float* x1777 = (float*)myMalloc(1 * sizeof(float));;
			x1777[0] = 1.0f;
			float* x1779 = (float*)myMalloc(1 * sizeof(float));;
			x1779[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 8192,10,x1777,x107,8192,x1779, x256, 8192, x107,8192));
			arrayFill_greg<<<28, 512>>>(x256, 0.0f, 81920);
			float* x1783 = (float*)myMalloc(1 * sizeof(float));;
			x1783[0] = 1.0f;
			float* x1785 = (float*)myMalloc(1 * sizeof(float));;
			x1785[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x1783,x110,1,x1785, x257, 1, x110,1));
			arrayFill_greg<<<28, 512>>>(x257, 0.0f, 64);
			float* x1789 = (float*)myMalloc(1 * sizeof(float));;
			x1789[0] = 1.0f;
			float* x1791 = (float*)myMalloc(1 * sizeof(float));;
			x1791[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 288,128,x1789,x113,288,x1791, x258, 288, x113,288));
			arrayFill_greg<<<28, 512>>>(x258, 0.0f, 36864);
			float* x1795 = (float*)myMalloc(1 * sizeof(float));;
			x1795[0] = 1.0f;
			float* x1797 = (float*)myMalloc(1 * sizeof(float));;
			x1797[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 32,128,x1795,x116,32,x1797, x259, 32, x116,32));
			arrayFill_greg<<<28, 512>>>(x259, 0.0f, 4096);
			float* x1801 = (float*)myMalloc(1 * sizeof(float));;
			x1801[0] = 1.0f;
			float* x1803 = (float*)myMalloc(1 * sizeof(float));;
			x1803[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,16,x1801,x119,1,x1803, x260, 1, x119,1));
			arrayFill_greg<<<28, 512>>>(x260, 0.0f, 16);
			float* x1807 = (float*)myMalloc(1 * sizeof(float));;
			x1807[0] = 1.0f;
			float* x1809 = (float*)myMalloc(1 * sizeof(float));;
			x1809[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x1807,x122,1,x1809, x261, 1, x122,1));
			arrayFill_greg<<<28, 512>>>(x261, 0.0f, 64);
			float* x1813 = (float*)myMalloc(1 * sizeof(float));;
			x1813[0] = 1.0f;
			float* x1815 = (float*)myMalloc(1 * sizeof(float));;
			x1815[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,32,x1813,x125,128,x1815, x262, 128, x125,128));
			arrayFill_greg<<<28, 512>>>(x262, 0.0f, 4096);
			float* x1819 = (float*)myMalloc(1 * sizeof(float));;
			x1819[0] = 1.0f;
			float* x1821 = (float*)myMalloc(1 * sizeof(float));;
			x1821[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 16,64,x1819,x128,16,x1821, x263, 16, x128,16));
			arrayFill_greg<<<28, 512>>>(x263, 0.0f, 1024);
			float* x1825 = (float*)myMalloc(1 * sizeof(float));;
			x1825[0] = 1.0f;
			float* x1827 = (float*)myMalloc(1 * sizeof(float));;
			x1827[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 128,16,x1825,x131,128,x1827, x264, 128, x131,128));
			arrayFill_greg<<<28, 512>>>(x264, 0.0f, 2048);
			float* x1831 = (float*)myMalloc(1 * sizeof(float));;
			x1831[0] = 1.0f;
			float* x1833 = (float*)myMalloc(1 * sizeof(float));;
			x1833[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x1831,x134,1,x1833, x265, 1, x134,1));
			arrayFill_greg<<<28, 512>>>(x265, 0.0f, 256);
			float* x1837 = (float*)myMalloc(1 * sizeof(float));;
			x1837[0] = 1.0f;
			float* x1839 = (float*)myMalloc(1 * sizeof(float));;
			x1839[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 384,48,x1837,x137,384,x1839, x266, 384, x137,384));
			arrayFill_greg<<<28, 512>>>(x266, 0.0f, 18432);
			float* x1843 = (float*)myMalloc(1 * sizeof(float));;
			x1843[0] = 1.0f;
			float* x1845 = (float*)myMalloc(1 * sizeof(float));;
			x1845[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1843,x140,1,x1845, x267, 1, x140,1));
			arrayFill_greg<<<28, 512>>>(x267, 0.0f, 32);
			float* x1849 = (float*)myMalloc(1 * sizeof(float));;
			x1849[0] = 1.0f;
			float* x1851 = (float*)myMalloc(1 * sizeof(float));;
			x1851[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x1849,x143,1,x1851, x268, 1, x143,1));
			arrayFill_greg<<<28, 512>>>(x268, 0.0f, 128);
			float* x1855 = (float*)myMalloc(1 * sizeof(float));;
			x1855[0] = 1.0f;
			float* x1857 = (float*)myMalloc(1 * sizeof(float));;
			x1857[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x1855,x146,1,x1857, x269, 1, x146,1));
			arrayFill_greg<<<28, 512>>>(x269, 0.0f, 256);
			float* x1861 = (float*)myMalloc(1 * sizeof(float));;
			x1861[0] = 1.0f;
			float* x1863 = (float*)myMalloc(1 * sizeof(float));;
			x1863[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 432,192,x1861,x149,432,x1863, x270, 432, x149,432));
			arrayFill_greg<<<28, 512>>>(x270, 0.0f, 82944);
			float* x1867 = (float*)myMalloc(1 * sizeof(float));;
			x1867[0] = 1.0f;
			float* x1869 = (float*)myMalloc(1 * sizeof(float));;
			x1869[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 144,64,x1867,x152,144,x1869, x271, 144, x152,144));
			arrayFill_greg<<<28, 512>>>(x271, 0.0f, 9216);
			float* x1873 = (float*)myMalloc(1 * sizeof(float));;
			x1873[0] = 1.0f;
			float* x1875 = (float*)myMalloc(1 * sizeof(float));;
			x1875[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x1873,x155,1,x1875, x272, 1, x155,1));
			arrayFill_greg<<<28, 512>>>(x272, 0.0f, 64);
			float* x1879 = (float*)myMalloc(1 * sizeof(float));;
			x1879[0] = 1.0f;
			float* x1881 = (float*)myMalloc(1 * sizeof(float));;
			x1881[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x1879,x158,1,x1881, x273, 1, x158,1));
			arrayFill_greg<<<28, 512>>>(x273, 0.0f, 128);
			float* x1885 = (float*)myMalloc(1 * sizeof(float));;
			x1885[0] = 1.0f;
			float* x1887 = (float*)myMalloc(1 * sizeof(float));;
			x1887[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 48,192,x1885,x161,48,x1887, x274, 48, x161,48));
			arrayFill_greg<<<28, 512>>>(x274, 0.0f, 9216);
			float* x1891 = (float*)myMalloc(1 * sizeof(float));;
			x1891[0] = 1.0f;
			float* x1893 = (float*)myMalloc(1 * sizeof(float));;
			x1893[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,32,x1891,x164,1,x1893, x275, 1, x164,1));
			arrayFill_greg<<<28, 512>>>(x275, 0.0f, 32);
			float* x1897 = (float*)myMalloc(1 * sizeof(float));;
			x1897[0] = 1.0f;
			float* x1899 = (float*)myMalloc(1 * sizeof(float));;
			x1899[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 16,64,x1897,x167,16,x1899, x276, 16, x167,16));
			arrayFill_greg<<<28, 512>>>(x276, 0.0f, 1024);
			float* x1903 = (float*)myMalloc(1 * sizeof(float));;
			x1903[0] = 1.0f;
			float* x1905 = (float*)myMalloc(1 * sizeof(float));;
			x1905[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,16,x1903,x170,1,x1905, x277, 1, x170,1));
			arrayFill_greg<<<28, 512>>>(x277, 0.0f, 16);
			float* x1909 = (float*)myMalloc(1 * sizeof(float));;
			x1909[0] = 1.0f;
			float* x1911 = (float*)myMalloc(1 * sizeof(float));;
			x1911[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x1909,x173,1,x1911, x278, 1, x173,1));
			arrayFill_greg<<<28, 512>>>(x278, 0.0f, 256);
			float* x1915 = (float*)myMalloc(1 * sizeof(float));;
			x1915[0] = 1.0f;
			float* x1917 = (float*)myMalloc(1 * sizeof(float));;
			x1917[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,32,x1915,x176,256,x1917, x279, 256, x176,256));
			arrayFill_greg<<<28, 512>>>(x279, 0.0f, 8192);
			float* x1921 = (float*)myMalloc(1 * sizeof(float));;
			x1921[0] = 1.0f;
			float* x1923 = (float*)myMalloc(1 * sizeof(float));;
			x1923[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 576,256,x1921,x179,576,x1923, x280, 576, x179,576));
			arrayFill_greg<<<28, 512>>>(x280, 0.0f, 147456);
			float* x1927 = (float*)myMalloc(1 * sizeof(float));;
			x1927[0] = 1.0f;
			float* x1929 = (float*)myMalloc(1 * sizeof(float));;
			x1929[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,192,x1927,x182,1,x1929, x281, 1, x182,1));
			arrayFill_greg<<<28, 512>>>(x281, 0.0f, 192);
			float* x1933 = (float*)myMalloc(1 * sizeof(float));;
			x1933[0] = 1.0f;
			float* x1935 = (float*)myMalloc(1 * sizeof(float));;
			x1935[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 576,256,x1933,x185,576,x1935, x282, 576, x185,576));
			arrayFill_greg<<<28, 512>>>(x282, 0.0f, 147456);
			float* x1939 = (float*)myMalloc(1 * sizeof(float));;
			x1939[0] = 1.0f;
			float* x1941 = (float*)myMalloc(1 * sizeof(float));;
			x1941[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x1939,x188,1,x1941, x283, 1, x188,1));
			arrayFill_greg<<<28, 512>>>(x283, 0.0f, 64);
			float* x1945 = (float*)myMalloc(1 * sizeof(float));;
			x1945[0] = 1.0f;
			float* x1947 = (float*)myMalloc(1 * sizeof(float));;
			x1947[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,192,x1945,x191,1,x1947, x284, 1, x191,1));
			arrayFill_greg<<<28, 512>>>(x284, 0.0f, 192);
			float* x1951 = (float*)myMalloc(1 * sizeof(float));;
			x1951[0] = 1.0f;
			float* x1953 = (float*)myMalloc(1 * sizeof(float));;
			x1953[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 27,96,x1951,x194,27,x1953, x285, 27, x194,27));
			arrayFill_greg<<<28, 512>>>(x285, 0.0f, 2592);
			float* x1957 = (float*)myMalloc(1 * sizeof(float));;
			x1957[0] = 1.0f;
			float* x1959 = (float*)myMalloc(1 * sizeof(float));;
			x1959[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 384,64,x1957,x197,384,x1959, x286, 384, x197,384));
			arrayFill_greg<<<28, 512>>>(x286, 0.0f, 24576);
			float* x1963 = (float*)myMalloc(1 * sizeof(float));;
			x1963[0] = 1.0f;
			float* x1965 = (float*)myMalloc(1 * sizeof(float));;
			x1965[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 32,128,x1963,x200,32,x1965, x287, 32, x200,32));
			arrayFill_greg<<<28, 512>>>(x287, 0.0f, 4096);
			float* x1969 = (float*)myMalloc(1 * sizeof(float));;
			x1969[0] = 1.0f;
			float* x1971 = (float*)myMalloc(1 * sizeof(float));;
			x1971[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 288,128,x1969,x203,288,x1971, x288, 288, x203,288));
			arrayFill_greg<<<28, 512>>>(x288, 0.0f, 36864);
			float* x1975 = (float*)myMalloc(1 * sizeof(float));;
			x1975[0] = 1.0f;
			float* x1977 = (float*)myMalloc(1 * sizeof(float));;
			x1977[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,64,x1975,x206,1,x1977, x289, 1, x206,1));
			arrayFill_greg<<<28, 512>>>(x289, 0.0f, 64);
			float* x1981 = (float*)myMalloc(1 * sizeof(float));;
			x1981[0] = 1.0f;
			float* x1983 = (float*)myMalloc(1 * sizeof(float));;
			x1983[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x1981,x209,64,x1983, x290, 64, x209,64));
			arrayFill_greg<<<28, 512>>>(x290, 0.0f, 16384);
			float* x1987 = (float*)myMalloc(1 * sizeof(float));;
			x1987[0] = 1.0f;
			float* x1989 = (float*)myMalloc(1 * sizeof(float));;
			x1989[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 432,192,x1987,x212,432,x1989, x291, 432, x212,432));
			arrayFill_greg<<<28, 512>>>(x291, 0.0f, 82944);
			float* x1993 = (float*)myMalloc(1 * sizeof(float));;
			x1993[0] = 1.0f;
			float* x1995 = (float*)myMalloc(1 * sizeof(float));;
			x1995[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,256,x1993,x215,1,x1995, x292, 1, x215,1));
			arrayFill_greg<<<28, 512>>>(x292, 0.0f, 256);
			float* x1999 = (float*)myMalloc(1 * sizeof(float));;
			x1999[0] = 1.0f;
			float* x2001 = (float*)myMalloc(1 * sizeof(float));;
			x2001[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x1999,x218,1,x2001, x293, 1, x218,1));
			arrayFill_greg<<<28, 512>>>(x293, 0.0f, 128);
			float* x2005 = (float*)myMalloc(1 * sizeof(float));;
			x2005[0] = 1.0f;
			float* x2007 = (float*)myMalloc(1 * sizeof(float));;
			x2007[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 256,48,x2005,x221,256,x2007, x294, 256, x221,256));
			arrayFill_greg<<<28, 512>>>(x294, 0.0f, 12288);
			float* x2011 = (float*)myMalloc(1 * sizeof(float));;
			x2011[0] = 1.0f;
			float* x2013 = (float*)myMalloc(1 * sizeof(float));;
			x2013[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,96,x2011,x224,1,x2013, x295, 1, x224,1));
			arrayFill_greg<<<28, 512>>>(x295, 0.0f, 96);
			float* x2017 = (float*)myMalloc(1 * sizeof(float));;
			x2017[0] = 1.0f;
			float* x2019 = (float*)myMalloc(1 * sizeof(float));;
			x2019[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,192,x2017,x227,1,x2019, x296, 1, x227,1));
			arrayFill_greg<<<28, 512>>>(x296, 0.0f, 192);
			float* x2023 = (float*)myMalloc(1 * sizeof(float));;
			x2023[0] = 1.0f;
			float* x2025 = (float*)myMalloc(1 * sizeof(float));;
			x2025[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,128,x2023,x230,1,x2025, x297, 1, x230,1));
			arrayFill_greg<<<28, 512>>>(x297, 0.0f, 128);
			float* x2029 = (float*)myMalloc(1 * sizeof(float));;
			x2029[0] = 1.0f;
			float* x2031 = (float*)myMalloc(1 * sizeof(float));;
			x2031[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,192,x2029,x233,1,x2031, x298, 1, x233,1));
			arrayFill_greg<<<28, 512>>>(x298, 0.0f, 192);
			float* x2035 = (float*)myMalloc(1 * sizeof(float));;
			x2035[0] = 1.0f;
			float* x2037 = (float*)myMalloc(1 * sizeof(float));;
			x2037[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 144,64,x2035,x236,144,x2037, x299, 144, x236,144));
			arrayFill_greg<<<28, 512>>>(x299, 0.0f, 9216);
			float* x2041 = (float*)myMalloc(1 * sizeof(float));;
			x2041[0] = 1.0f;
			float* x2043 = (float*)myMalloc(1 * sizeof(float));;
			x2043[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 48,192,x2041,x239,48,x2043, x300, 48, x239,48));
			arrayFill_greg<<<28, 512>>>(x300, 0.0f, 9216);
			float* x2047 = (float*)myMalloc(1 * sizeof(float));;
			x2047[0] = 1.0f;
			float* x2049 = (float*)myMalloc(1 * sizeof(float));;
			x2049[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 64,256,x2047,x242,64,x2049, x301, 64, x242,64));
			arrayFill_greg<<<28, 512>>>(x301, 0.0f, 16384);
			float* x2053 = (float*)myMalloc(1 * sizeof(float));;
			x2053[0] = 1.0f;
			float* x2055 = (float*)myMalloc(1 * sizeof(float));;
			x2055[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 96,16,x2053,x245,96,x2055, x302, 96, x245,96));
			arrayFill_greg<<<28, 512>>>(x302, 0.0f, 1536);
			float* x2059 = (float*)myMalloc(1 * sizeof(float));;
			x2059[0] = 1.0f;
			float* x2061 = (float*)myMalloc(1 * sizeof(float));;
			x2061[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,10,x2059,x248,1,x2061, x303, 1, x248,1));
			arrayFill_greg<<<28, 512>>>(x303, 0.0f, 10);
			float* x2065 = (float*)myMalloc(1 * sizeof(float));;
			x2065[0] = 1.0f;
			float* x2067 = (float*)myMalloc(1 * sizeof(float));;
			x2067[0] = -0.005f;
			CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1,48,x2065,x251,1,x2067, x304, 1, x251,1));
			arrayFill_greg<<<28, 512>>>(x304, 0.0f, 48);
			int32_t x2071 = x321 + 1;
			int32_t x2073 = x2071 % x2072;
			bool x2074 = x2073 == 0;
			if (x2074) {
				float x2079 = x315;
				double x2075 = (double)x322;
				double x2076 = 100.0 * x2075;
				double x2078 = x2076 / x2077;
				float x2080 = (float)x321;
				float x2081 = x2079 / x2080;
				printf("Train epoch %d: [%d/%d (%.0f%%)] Average Loss: %.6f\n",x311,x322,x11,x2078,x2081);
				fflush(stdout);
			} else {
			}
			int64_t x2086 = (long)mallocAddr;
			int64_t x2087 = x2086 - x307;
			memset((void*)x307, 0, x2087);
			mallocAddr = (void*)x307;
			int64_t x2090 = (long)gpuMallocAddr;
			int64_t x2091 = x2090 - x308;
			cudaMemset((void*)x308, 0, x2091);
			gpuMallocAddr = (void*)x308;

		}
		gettimeofday(&end_1, NULL);
		timeval_subtract(&diff_1, &end_1, &begin_1);;
		int64_t x2098 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
		double x2099 = (double)x2098;
		double x2100 = x2099 / 1000000.0;
		x306[x311] = x2100;
		int64_t x2102 = x2098 / 1000LL;
		int64_t x2104 = x2098 / x2103;
		printf("Training completed in %ldms (%ld us/images)\n",x2102,x2104);
		float x2106 = x315;
		float x2108 = x2106 / x2107;
		double x2109 = (double)x2108;
		x305[x311] = x2109;

	}
	gettimeofday(&end_0, NULL);
	timeval_subtract(&diff_0, &end_0, &begin_0);;
	int64_t x2115 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
	sort(x306, x306 + 4);
	double x2121 = x306[2];
	int64_t x2122 = (long)fopen(x0, "w");
	fprintf((FILE *)x2122, "unit: %s\n", "1 epoch");
	for(int x2124=0; x2124 < 4; x2124++) {
		double x2125 = x305[x2124];
		fprintf((FILE *)x2122, "%lf\n", x2125);

	}
	fprintf((FILE *)x2122, "run time: %lf %lf\n", x39, x2121);
	fclose((FILE*)x2122);
	// Backend cleanup.
	CUBLAS_CALL(cublasDestroy(cublasHandle));
	CUDA_CALL(cudaFree(gpuMallocBase));

	CUDNN_CALL(cudnnDestroy(cudnnHandle));
}
/*****************************************
  End of C Generated Code                  
 *******************************************/


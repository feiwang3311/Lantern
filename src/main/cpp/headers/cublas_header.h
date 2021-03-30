#include <curand_kernel.h>
#include <curand.h>

// thrust is used in embedding backward pass
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/functional.h>

#define NVIDIA_WARP_SIZE 32 // this is typically 32 (for incl. 1080ti s)

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

long HEAP_SIZE = 10737418240; // 1073741824; // 4294967296; // 8589934592; // 10737418240;
// Alignment boundary size, in bytes.
constexpr int N = 4; // 16
void *myGpuMalloc(size_t bytes) {
  bytes = ((bytes + (1 << N) - 1) >> N) << N;
  void *res = gpuMallocAddr;
  gpuMallocAddr = (void *)((char *)gpuMallocAddr + bytes);
  if ((long)gpuMallocAddr > (long)gpuMallocBase + HEAP_SIZE) {
    fprintf(stderr, "GPU breached memory limit of HEAP_SIZE\n");
    // try to throw a SegFault here so that I can use gdb to find where the error is:
    int *foo = (int*)-1;
    printf("%d\n", *foo);
  }
  return res;
}

void myGpuFree(size_t bytes) {
  bytes = ((bytes + (1 << N) - 1) >> N) << N;
  gpuMallocAddr = (void *)((char *)gpuMallocAddr - bytes);
  cudaMemset((void*)gpuMallocAddr, 0, bytes);
  return;
}

#define AVAIL_GPU_MEM ((long)gpuMallocBase + HEAP_SIZE - (long)gpuMallocAddr)
#define CAP_AVAIL(claim) min(AVAIL_GPU_MEM, claim)

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

//  // only for 4D tensor in and 3D tensor out (TODO: incorrect!)
// __global__ void sum_optimization(float* in, int inStr0, int inStr1, int inStr2, int inStr3,
//                                  float* out, int outStr0, int outStr1, int outStr2,
//                                  int dim, int nElementOut, int dimSize) {
//   int tid = blockIdx.x * blockDim.x + threadIdx.x;
//   int stride = gridDim.x * blockDim.x;
//   for (int i = tid; i < nElementOut; i += stride) {
//     int outOff0 = i / outStr0;
//     int outOff1temp = i - outOff0 * outStr0;
//     int outOff1 = outOff1temp / outStr1;
//     int outOff2 = outOff1temp - outOff1 * outStr1;
//     for (int j = 0; j < dimSize; j++) {
//       int inOff;
//       if (dim == 0) inOff = j * inStr0 + outOff0 * inStr1 + outOff1 * inStr2 + outOff2 * inStr3;
//       if (dim == 1) inOff = outOff0 * inStr0 + j * inStr1 + outOff1 * inStr2 + outOff2 * inStr3;
//       if (dim == 2) inOff = outOff0 * inStr0 + outOff1 * inStr1 + j * inStr2 + outOff2 * inStr3;
//       if (dim == 3) inOff = outOff0 * inStr0 + outOff1 * inStr1 + outOff2 * inStr2 + j * inStr3;
//       out[i] += in[inOff];
//     }
//   }
// }

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

/**
 * elementWiseWithBroadCast kernels
 */

__global__ void elementWiseWithBroadCastRank1Add(float* in1, float* in2, float* out, int size,
                int in1Stride0, int in2Stride0, int outStride0) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride) {
    int linearIdx = tid;
    int outIndex0 = linearIdx / outStride0; linearIdx = linearIdx - outIndex0 * outStride0;
    int in1Index = in1Stride0 * outIndex0;
    int in2Index = in2Stride0 * outIndex0;
    out[tid] = in1[in1Index] + in2[in2Index];
  }
}

__global__ void elementWiseWithBroadCastRank1Div(float* in1, float* in2, float* out, int size,
                int in1Stride0, int in2Stride0, int outStride0) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride) {
    int linearIdx = tid;
    int outIndex0 = linearIdx / outStride0; linearIdx = linearIdx - outIndex0 * outStride0;
    int in1Index = in1Stride0 * outIndex0;
    int in2Index = in2Stride0 * outIndex0;
    out[tid] = in1[in1Index] / in2[in2Index];
  }
}

__global__ void elementWiseWithBroadCastRank1Mult(float* in1, float* in2, float* out, int size,
                int in1Stride0, int in2Stride0, int outStride0) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride) {
    int linearIdx = tid;
    int outIndex0 = linearIdx / outStride0; linearIdx = linearIdx - outIndex0 * outStride0;
    int in1Index = in1Stride0 * outIndex0;
    int in2Index = in2Stride0 * outIndex0;
    out[tid] = in1[in1Index] * in2[in2Index];
  }
}

__global__ void elementWiseWithBroadCastRank1Minus(float* in1, float* in2, float* out, int size,
                int in1Stride0, int in2Stride0, int outStride0) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride) {
    int linearIdx = tid;
    int outIndex0 = linearIdx / outStride0; linearIdx = linearIdx - outIndex0 * outStride0;
    int in1Index = in1Stride0 * outIndex0;
    int in2Index = in2Stride0 * outIndex0;
    out[tid] = in1[in1Index] - in2[in2Index];
  }
}

__global__ void elementWiseWithBroadCastRank2Add(float* in1, float* in2, float* out, int size,
                int in1Stride0, int in1Stride1, int in2Stride0, int in2Stride1, int outStride0, int outStride1) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride) {
    int linearIdx = tid;
    int outIndex0 = linearIdx / outStride0; linearIdx = linearIdx - outIndex0 * outStride0;
    int outIndex1 = linearIdx / outStride1; linearIdx = linearIdx - outIndex1 * outStride1;
    int in1Index = in1Stride0 * outIndex0 + in1Stride1 * outIndex1;
    int in2Index = in2Stride0 * outIndex0 + in2Stride1 * outIndex1;
    out[tid] = in1[in1Index] + in2[in2Index];
  }
}

__global__ void elementWiseWithBroadCastRank2Div(float* in1, float* in2, float* out, int size,
                int in1Stride0, int in1Stride1, int in2Stride0, int in2Stride1, int outStride0, int outStride1) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride) {
    int linearIdx = tid;
    int outIndex0 = linearIdx / outStride0; linearIdx = linearIdx - outIndex0 * outStride0;
    int outIndex1 = linearIdx / outStride1; linearIdx = linearIdx - outIndex1 * outStride1;
    int in1Index = in1Stride0 * outIndex0 + in1Stride1 * outIndex1;
    int in2Index = in2Stride0 * outIndex0 + in2Stride1 * outIndex1;
    out[tid] = in1[in1Index] / in2[in2Index];
  }
}

__global__ void elementWiseWithBroadCastRank2Mult(float* in1, float* in2, float* out, int size,
                int in1Stride0, int in1Stride1, int in2Stride0, int in2Stride1, int outStride0, int outStride1) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride) {
    int linearIdx = tid;
    int outIndex0 = linearIdx / outStride0; linearIdx = linearIdx - outIndex0 * outStride0;
    int outIndex1 = linearIdx / outStride1; linearIdx = linearIdx - outIndex1 * outStride1;
    int in1Index = in1Stride0 * outIndex0 + in1Stride1 * outIndex1;
    int in2Index = in2Stride0 * outIndex0 + in2Stride1 * outIndex1;
    out[tid] = in1[in1Index] * in2[in2Index];
  }
}

__global__ void elementWiseWithBroadCastRank2Minus(float* in1, float* in2, float* out, int size,
                int in1Stride0, int in1Stride1, int in2Stride0, int in2Stride1, int outStride0, int outStride1) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride) {
    int linearIdx = tid;
    int outIndex0 = linearIdx / outStride0; linearIdx = linearIdx - outIndex0 * outStride0;
    int outIndex1 = linearIdx / outStride1; linearIdx = linearIdx - outIndex1 * outStride1;
    int in1Index = in1Stride0 * outIndex0 + in1Stride1 * outIndex1;
    int in2Index = in2Stride0 * outIndex0 + in2Stride1 * outIndex1;
    out[tid] = in1[in1Index] - in2[in2Index];
  }
}

__global__ void elementWiseWithBroadCastRank3Add(float* in1, float* in2, float* out, int size,
                int in1Stride0, int in1Stride1, int in1Stride2, int in2Stride0, int in2Stride1, int in2Stride2,
                int outStride0, int outStride1, int outStride2) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride) {
    int linearIdx = tid;
    int outIndex0 = linearIdx / outStride0; linearIdx = linearIdx - outIndex0 * outStride0;
    int outIndex1 = linearIdx / outStride1; linearIdx = linearIdx - outIndex1 * outStride1;
    int outIndex2 = linearIdx / outStride2; linearIdx = linearIdx - outIndex2 * outStride2;
    int in1Index = in1Stride0 * outIndex0 + in1Stride1 * outIndex1 + in1Stride2 * outIndex2;
    int in2Index = in2Stride0 * outIndex0 + in2Stride1 * outIndex1 + in2Stride2 * outIndex2;
    out[tid] = in1[in1Index] + in2[in2Index];
  }
}

__global__ void elementWiseWithBroadCastRank3Div(float* in1, float* in2, float* out, int size,
                int in1Stride0, int in1Stride1, int in1Stride2, int in2Stride0, int in2Stride1, \
                int in2Stride2, int outStride0, int outStride1, int outStride2) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride) {
    int linearIdx = tid;
    int outIndex0 = linearIdx / outStride0; linearIdx = linearIdx - outIndex0 * outStride0;
    int outIndex1 = linearIdx / outStride1; linearIdx = linearIdx - outIndex1 * outStride1;
    int outIndex2 = linearIdx / outStride2; linearIdx = linearIdx - outIndex2 * outStride2;
    int in1Index = in1Stride0 * outIndex0 + in1Stride1 * outIndex1 + in1Stride2 * outIndex2;
    int in2Index = in2Stride0 * outIndex0 + in2Stride1 * outIndex1 + in2Stride2 * outIndex2;
    out[tid] = in1[in1Index] / in2[in2Index];
  }
}

__global__ void elementWiseWithBroadCastRank3Mult(float* in1, float* in2, float* out, int size,
                int in1Stride0, int in1Stride1, int in1Stride2, int in2Stride0, int in2Stride1,
                int in2Stride2, int outStride0, int outStride1, int outStride2) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride) {
    int linearIdx = tid;
    int outIndex0 = linearIdx / outStride0; linearIdx = linearIdx - outIndex0 * outStride0;
    int outIndex1 = linearIdx / outStride1; linearIdx = linearIdx - outIndex1 * outStride1;
    int outIndex2 = linearIdx / outStride2; linearIdx = linearIdx - outIndex2 * outStride2;
    int in1Index = in1Stride0 * outIndex0 + in1Stride1 * outIndex1 + in1Stride2 * outIndex2;
    int in2Index = in2Stride0 * outIndex0 + in2Stride1 * outIndex1 + in2Stride2 * outIndex2;
    out[tid] = in1[in1Index] * in2[in2Index];
  }
}

__global__ void elementWiseWithBroadCastRank3Minus(float* in1, float* in2, float* out, int size,
                int in1Stride0, int in1Stride1, int in1Stride2, int in2Stride0, int in2Stride1,
                int in2Stride2, int outStride0, int outStride1, int outStride2) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride) {
    int linearIdx = tid;
    int outIndex0 = linearIdx / outStride0; linearIdx = linearIdx - outIndex0 * outStride0;
    int outIndex1 = linearIdx / outStride1; linearIdx = linearIdx - outIndex1 * outStride1;
    int outIndex2 = linearIdx / outStride2; linearIdx = linearIdx - outIndex2 * outStride2;
    int in1Index = in1Stride0 * outIndex0 + in1Stride1 * outIndex1 + in1Stride2 * outIndex2;
    int in2Index = in2Stride0 * outIndex0 + in2Stride1 * outIndex1 + in2Stride2 * outIndex2;
    out[tid] = in1[in1Index] - in2[in2Index];
  }
}

__global__ void elementWiseWithBroadCastRank4Add(float* in1, float* in2, float* out, int size,
                int in1Stride0, int in1Stride1, int in1Stride2, int in1Stride3, int in2Stride0,
                int in2Stride1, int in2Stride2, int in2Stride3, int outStride0, int outStride1,
                int outStride2, int outStride3) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride) {
    int linearIdx = tid;
    int outIndex0 = linearIdx / outStride0; linearIdx = linearIdx - outIndex0 * outStride0;
    int outIndex1 = linearIdx / outStride1; linearIdx = linearIdx - outIndex1 * outStride1;
    int outIndex2 = linearIdx / outStride2; linearIdx = linearIdx - outIndex2 * outStride2;
    int outIndex3 = linearIdx / outStride3; linearIdx = linearIdx - outIndex3 * outStride3;
    int in1Index = in1Stride0 * outIndex0 + in1Stride1 * outIndex1 + in1Stride2 * outIndex2 + in1Stride3 * outIndex3;
    int in2Index = in2Stride0 * outIndex0 + in2Stride1 * outIndex1 + in2Stride2 * outIndex2 + in2Stride3 * outIndex3;
    out[tid] = in1[in1Index] + in2[in2Index];
  }
}

__global__ void elementWiseWithBroadCastRank4Div(float* in1, float* in2, float* out, int size,
                int in1Stride0, int in1Stride1, int in1Stride2, int in1Stride3, int in2Stride0,
                int in2Stride1, int in2Stride2, int in2Stride3, int outStride0, int outStride1,
                int outStride2, int outStride3) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride) {
    int linearIdx = tid;
    int outIndex0 = linearIdx / outStride0; linearIdx = linearIdx - outIndex0 * outStride0;
    int outIndex1 = linearIdx / outStride1; linearIdx = linearIdx - outIndex1 * outStride1;
    int outIndex2 = linearIdx / outStride2; linearIdx = linearIdx - outIndex2 * outStride2;
    int outIndex3 = linearIdx / outStride3; linearIdx = linearIdx - outIndex3 * outStride3;
    int in1Index = in1Stride0 * outIndex0 + in1Stride1 * outIndex1 + in1Stride2 * outIndex2 + in1Stride3 * outIndex3;
    int in2Index = in2Stride0 * outIndex0 + in2Stride1 * outIndex1 + in2Stride2 * outIndex2 + in2Stride3 * outIndex3;
    out[tid] = in1[in1Index] / in2[in2Index];
  }
}

__global__ void elementWiseWithBroadCastRank4Mult(float* in1, float* in2, float* out, int size,
                int in1Stride0, int in1Stride1, int in1Stride2, int in1Stride3, int in2Stride0,
                int in2Stride1, int in2Stride2, int in2Stride3, int outStride0, int outStride1,
                int outStride2, int outStride3) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride) {
    int linearIdx = tid;
    int outIndex0 = linearIdx / outStride0; linearIdx = linearIdx - outIndex0 * outStride0;
    int outIndex1 = linearIdx / outStride1; linearIdx = linearIdx - outIndex1 * outStride1;
    int outIndex2 = linearIdx / outStride2; linearIdx = linearIdx - outIndex2 * outStride2;
    int outIndex3 = linearIdx / outStride3; linearIdx = linearIdx - outIndex3 * outStride3;
    int in1Index = in1Stride0 * outIndex0 + in1Stride1 * outIndex1 + in1Stride2 * outIndex2 + in1Stride3 * outIndex3;
    int in2Index = in2Stride0 * outIndex0 + in2Stride1 * outIndex1 + in2Stride2 * outIndex2 + in2Stride3 * outIndex3;
    out[tid] = in1[in1Index] * in2[in2Index];
  }
}

__global__ void elementWiseWithBroadCastRank4Minus(float* in1, float* in2, float* out, int size,
                int in1Stride0, int in1Stride1, int in1Stride2, int in1Stride3, int in2Stride0,
                int in2Stride1, int in2Stride2, int in2Stride3, int outStride0, int outStride1,
                int outStride2, int outStride3) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (; tid < size; tid += stride) {
    int linearIdx = tid;
    int outIndex0 = linearIdx / outStride0; linearIdx = linearIdx - outIndex0 * outStride0;
    int outIndex1 = linearIdx / outStride1; linearIdx = linearIdx - outIndex1 * outStride1;
    int outIndex2 = linearIdx / outStride2; linearIdx = linearIdx - outIndex2 * outStride2;
    int outIndex3 = linearIdx / outStride3; linearIdx = linearIdx - outIndex3 * outStride3;
    int in1Index = in1Stride0 * outIndex0 + in1Stride1 * outIndex1 + in1Stride2 * outIndex2 + in1Stride3 * outIndex3;
    int in2Index = in2Stride0 * outIndex0 + in2Stride1 * outIndex1 + in2Stride2 * outIndex2 + in2Stride3 * outIndex3;
    out[tid] = in1[in1Index] - in2[in2Index];
  }
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

__global__ void softmax(float* input, float* output, int size) {
    // assume gridDim.x equals outerSize
    // assume computing softmax in last dim
    extern __shared__ float buffer[];

    // not vectorized and not unrolled implementation - has room for performance improvement
    float *input_t = input + size * blockIdx.x;
    float *output_t = output + size * blockIdx.x;

    int start = threadIdx.x;
    int end = size;
    int stride = blockDim.x;

    float threadVal = -INFINITY;
    // find the max
    for(int i = start; i < end; i += stride) {
        if (threadVal < input_t[i])
            threadVal = input_t[i];
    }

    buffer[threadIdx.x] = threadVal;
    // printf("%f\n", input_t[threadIdx.x]);
    __syncthreads();

    float warpVal = -INFINITY;
    // reduce
    // first thread reduce the first WARP, second reduces the second WARP etc.
    if (threadIdx.x < blockDim.x / NVIDIA_WARP_SIZE) {
        int lane = threadIdx.x;
        #pragma unroll
        for(int i = 0; i < NVIDIA_WARP_SIZE; i ++) {
            if (warpVal < buffer[lane * NVIDIA_WARP_SIZE + i]) warpVal = buffer[lane * NVIDIA_WARP_SIZE + i];
        }
        buffer[lane] = warpVal;
    }

    __syncthreads();

    // final reduce in the first thread
    if (threadIdx.x == 0) {
        float max = -INFINITY;

        for (int i = 0; i < blockDim.x / NVIDIA_WARP_SIZE; i ++) {
            if (max < buffer[i]) {
                max = buffer[i];
            }
        }
        buffer[0] = max;
    }

    __syncthreads();

    // compute the sum
    // TODO - check whether sequential addressing is better here (To avoid shared memory blank conflicts)?
    threadVal = 0;
    for(int i = start; i < end; i += stride) {
        float expVal = expf(input_t[i] - buffer[0]);
        threadVal += expVal;
        output_t[i] = expVal;
    }

    buffer[threadIdx.x] = threadVal;

    __syncthreads();

    warpVal = 0;
    // reduce
    if (threadIdx.x < blockDim.x / NVIDIA_WARP_SIZE) {
        int lane = threadIdx.x;
        #pragma unroll
        for(int i = 0; i < NVIDIA_WARP_SIZE; i++) {
            warpVal += buffer[lane * NVIDIA_WARP_SIZE + i];
        }
        buffer[lane] = warpVal;
    }

    __syncthreads();

    // final reduce
    if (threadIdx.x == 0) {
        float sum = 0;

        for(int i = 0; i < blockDim.x / NVIDIA_WARP_SIZE; i ++) {
            sum += buffer[i];
        }
        buffer[0] = sum;
    }

    __syncthreads();

    // do the softmax
    for(int i = threadIdx.x; i < size; i += stride) {
        output_t[i] = output_t[i] / buffer[0];
    }
}

__global__ void softmaxGrad(float *gradInput, float *gradOutput, float *output, int size) {
    extern __shared__ float buffer[];

    float *gradInput_t = gradInput + size * blockIdx.x;
    float *gradOutput_t = gradOutput + size * blockIdx.x;
    float *output_t = output + size * blockIdx.x;

    int start = threadIdx.x;
    int end = size;
    int stride = blockDim.x;

    // compute the sum (gradOutput * output sum)
    buffer[threadIdx.x] = 0;
    for(int i=start; i < end; i += stride) {
        buffer[threadIdx.x] += gradOutput_t[i] * output_t[i];
    }

    __syncthreads();

    float warpVal = 0;
    // let's reduce using the first warp
    if (threadIdx.x < blockDim.x / NVIDIA_WARP_SIZE) {
        int lane = threadIdx.x;
        #pragma unroll
        for(int i = 0; i < NVIDIA_WARP_SIZE; i++) {
            warpVal += buffer[lane * NVIDIA_WARP_SIZE + i];
        }
        buffer[lane] = warpVal;
    }

    __syncthreads();

    // final reduce
    if (threadIdx.x == 0) {
        float sum = 0;
        for(int i = 0; i < blockDim.x / NVIDIA_WARP_SIZE; i ++) {
            sum += buffer[i];
        }
        buffer[0] = sum;
    }

    __syncthreads();

    // update the gradient
    for(int i = start; i < end; i += stride) {
        gradInput_t[i] = output_t[i] * (gradOutput_t[i] - buffer[0]);
    }
}

int log2_ceil(int value) {
    int log2_value = 0;
    while ((1 << log2_value) < value) ++log2_value;
    return log2_value;
}

template <int WARP_BATCH, int WARP_SIZE>
__device__ __forceinline__ void warp_reduce_max(float* sum) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            float b = __shfl_xor_sync(0xffffffff, sum[i], offset, WARP_SIZE);
            sum[i] = fmaxf(sum[i], b);
        }
    }
}

template <int WARP_BATCH, int WARP_SIZE>
__device__ __forceinline__ void warp_reduce_add(float* sum) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            float b = __shfl_xor_sync(0xffffffff, sum[i], offset, WARP_SIZE);
            sum[i] += b;
        }
    }
}

// This kernel is for softmax (and logsoftmax) for dimSize (lastdim, also the softmax dim) < 1024
// This is copied from the PyTorch implementation with slight modifications to fit to existing impl.
// aten/src/ATen/native/cuda/PersistentSoftmax.cuh
template <int log2_elements, bool is_log_softmax>
__global__ void softmax_warp_forward(float *dst, float *src, int batch_size, int stride, int element_count)
{
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_forward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < NVIDIA_WARP_SIZE) ? next_power_of_two : NVIDIA_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;

    src += first_batch * stride + local_idx;
    dst += first_batch * stride + local_idx;

    // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
    // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
    // the nested loops.
    // This should have no impact on performance because the loops are unrolled anyway.

    // load data from global memory
    float elements[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                elements[i][it] = src[i*element_count+it*WARP_SIZE];
            } else {
                elements[i][it] = -INFINITY;
            }
        }
    }

    // compute max_value
    float max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        max_value[i] = elements[i][0];
        #pragma unroll
        for (int it = 1;  it < WARP_ITERATIONS;  ++it) {
            max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
        }
    }
    warp_reduce_max<WARP_BATCH, WARP_SIZE>(max_value);

    float sum[WARP_BATCH] { 0.0f };
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            if (is_log_softmax) {
              sum[i] += expf(elements[i][it] - max_value[i]);
            } else {
              elements[i][it] = expf(elements[i][it] - max_value[i]);
              sum[i] += elements[i][it];
            }
        }
    }
    warp_reduce_add<WARP_BATCH, WARP_SIZE>(sum);

    // store result
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
        if (is_log_softmax) sum[i] = std::log(sum[i]);
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                if (is_log_softmax) {
                    dst[i*element_count+it*WARP_SIZE] = elements[i][it] - max_value[i] - sum[i];
                } else {
                    dst[i*element_count+it*WARP_SIZE] = elements[i][it] / sum[i];
                }
            } else {
                break;
            }
        }
    }
}

// Taken from PyTorch with slight modification (to fit to existing impl) - similar to softmax_warp_forward
// aten/src/ATen/native/cuda/PersistentSoftmax.cuh
template <int log2_elements, bool is_log_softmax>
__global__ void softmax_warp_backward(float *gradInput, const float *grad, const float *output, int batch_size, int stride, int element_count)
{
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_backward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < NVIDIA_WARP_SIZE) ? next_power_of_two : NVIDIA_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x % WARP_SIZE;

    // the first element to process by the current thread
    int thread_offset = first_batch * stride + local_idx;
    grad += thread_offset;
    output += thread_offset;
    gradInput += thread_offset;

    // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
    // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
    // the nested loops.
    // This should have no impact on performance because the loops are unrolled anyway.

    // load data from global memory
    float grad_reg[WARP_BATCH][WARP_ITERATIONS];
    float output_reg[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                grad_reg[i][it] = grad[i*element_count+it*WARP_SIZE];
                output_reg[i][it] = output[i*element_count+it*WARP_SIZE];
            } else {
                grad_reg[i][it] = 0.0f;
                output_reg[i][it] = 0.0f;
            }
        }
    }

    float sum[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        sum[i] = grad_reg[i][0] * output_reg[i][0]; // had to change this - (was only grad_reg[i][0])
        #pragma unroll
        for (int it = 1;  it < WARP_ITERATIONS;  ++it) {
            sum[i] += grad_reg[i][it] * output_reg[i][it] ; // had to change this - (was only grad_reg[i][it])
        }
    }
    warp_reduce_add<WARP_BATCH, WARP_SIZE>(sum);

    // store result
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                // compute gradients
                if (is_log_softmax) {
                    gradInput[i*element_count+it*WARP_SIZE] = expf(output_reg[i][it]) * (grad_reg[i][it] - sum[i]); // TODO - check
                } else {
                    gradInput[i*element_count+it*WARP_SIZE] = output_reg[i][it] * (grad_reg[i][it] - sum[i]); // Had to change this from PT impl
                }
            }
        }
    }
}


// Taken from PyTorch with slight modification (to fit to existing impl)
// aten/src/ATen/native/cuda/PersistentSoftmax.cuh
template<bool is_log_softmax>
void dispatch_softmax_forward(float *dst, float *src, int softmax_elements, int softmax_elements_stride, int batch_count)
{
    int log2_elements = log2_ceil(softmax_elements);
    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_forward.
    int warp_size = (next_power_of_two < NVIDIA_WARP_SIZE) ? next_power_of_two : NVIDIA_WARP_SIZE;

    // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_forward.
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);
    // softmax_warp_forward<log2_elements, is_log_softmax>
                // <<<blocks, threads>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);

    switch (log2_elements) {
        case 0: // 1
            softmax_warp_forward<0, is_log_softmax>
                <<<blocks, threads>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
            break;
        case 1: // 2
            softmax_warp_forward<1, is_log_softmax>
                <<<blocks, threads>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
            break;
        case 2: // 4
            softmax_warp_forward<2, is_log_softmax>
                <<<blocks, threads>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
            break;
        case 3: // 8
            softmax_warp_forward<3, is_log_softmax>
                <<<blocks, threads>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
            break;
        case 4: // 16
            softmax_warp_forward<4, is_log_softmax>
                <<<blocks, threads>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
            break;
        case 5: // 32
            softmax_warp_forward<5, is_log_softmax>
                <<<blocks, threads>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
            break;
        case 6: // 64
            softmax_warp_forward<6, is_log_softmax>
                <<<blocks, threads>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
            break;
        case 7: // 128
            softmax_warp_forward<7, is_log_softmax>
                <<<blocks, threads>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
            break;
        case 8: // 256
            softmax_warp_forward<8, is_log_softmax>
                <<<blocks, threads>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
            break;
        case 9: // 512
            softmax_warp_forward<9, is_log_softmax>
                <<<blocks, threads>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
            break;
        case 10: // 1024
            softmax_warp_forward<10, is_log_softmax>
                <<<blocks, threads>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
            break;
        default:
            break;
    }
}

// Taken from PyTorch with slight modification (to fit to existing impl)
// aten/src/ATen/native/cuda/PersistentSoftmax.cuh
template<bool is_log_softmax>
void dispatch_softmax_backward(float *grad_input, const float *grad, const float *output, int softmax_elements, int softmax_elements_stride, int batch_count)
{
    int log2_elements = log2_ceil(softmax_elements);

    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_backward.
    int warp_size = (next_power_of_two < NVIDIA_WARP_SIZE) ? next_power_of_two : NVIDIA_WARP_SIZE;

    // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_backward.
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
        case 0: // 1
            softmax_warp_backward<0, is_log_softmax>
                <<<blocks, threads>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
            break;
        case 1: // 2
            softmax_warp_backward<1, is_log_softmax>
                <<<blocks, threads>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
            break;
        case 2: // 4
            softmax_warp_backward<2, is_log_softmax>
                <<<blocks, threads>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
            break;
        case 3: // 8
            softmax_warp_backward<3, is_log_softmax>
                <<<blocks, threads>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
            break;
        case 4: // 16
            softmax_warp_backward<4, is_log_softmax>
                <<<blocks, threads>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
            break;
        case 5: // 32
            softmax_warp_backward<5, is_log_softmax>
                <<<blocks, threads>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
            break;
        case 6: // 64
            softmax_warp_backward<6, is_log_softmax>
                <<<blocks, threads>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
            break;
        case 7: // 128
            softmax_warp_backward<7, is_log_softmax>
                <<<blocks, threads>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
            break;
        case 8: // 256
            softmax_warp_backward<8, is_log_softmax>
                <<<blocks, threads>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
            break;
        case 9: // 512
            softmax_warp_backward<9, is_log_softmax>
                <<<blocks, threads>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
            break;
        case 10: // 1024
            softmax_warp_backward<10, is_log_softmax>
                <<<blocks, threads>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
            break;
        default:
            break;
    }
}

// Note - p is keep-probability (not the drop probability)
__global__ void dropout(float* input, float *result, float p, bool *mask, int inputSize, long seed, long offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);

    float pinv = 1 / p; // can make this a kernel arg (then can be computed at staging time)

    for(int i = idx; i < inputSize; i += stride) {
        mask[i] = curand_uniform(&state) < p;
        result[i] = input[i] * mask[i] * pinv;
    }
}

__global__ void dropoutGrad(float *y_d, float *x_d, bool *mask, int inputSize, float pinv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < inputSize; i += stride) {
        if (mask[i]) {
            x_d[i] += y_d[i] * pinv;
        }
    }
}

// TODO - this would be an interesting kernel to implement in LMS (can avoid template based optimizations)
// ijSwapped is the case when dim0>dim1 (in this case dim0 is taken as dim1 and dim1 is taken as dim0 and ijSwapped = true)
template
<bool ijSwapped>
__global__ void maskedFill(float *in, float* out, int *mask, float value, int dim0_shape, int dim0_stride, int dim1_shape, int dim1_stride, int offset_size, int input_size) {
    // assumes mask is contiguous and has shape dim0 x dim1 (or dim1 x dim0 if ijSwapped)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // we are collapsing dims (logically)
    // e.g. []...[][i][]...[][j][]..[] ==> [i][j][inner]
    int i = tid / dim0_stride;
    int j = (tid - i*dim0_stride) / dim1_stride;
    int inner_idx = tid - i*dim0_stride - j*dim1_stride;
    int idx = i * dim0_stride + j * dim1_stride + inner_idx;

    while(idx < input_size){
//        printf("tid = %d; i = %d; j = %d; inner_idx=%d; index = %d\n", tid, i, j, inner_idx, idx );
//        printf("mask[%d] = %d\n", (i % dim0_shape) * dim1_shape + (j % dim1_shape), mask[(i % dim0_shape) * dim1_shape + (j % dim1_shape)]);

        // TODO - this mod operations (when computing mask_id) are expensive; can eliminate them for some simple cases (e.g. when we know j < dim1_shape; can eliminate % dim1_shape)
        // Can achieve this using templates
        int mask_id;
        if (ijSwapped)
            mask_id = (j % dim1_shape) * dim0_shape + (i % dim0_shape);
        else
            mask_id = (i % dim0_shape) * dim1_shape + (j % dim1_shape);

        if (mask[mask_id] != 0) {
            out[idx] = value;
        } else {
            out[idx] = in[idx];
        }

        tid += stride;
        i = tid / dim0_stride;
        j = (tid - i*dim0_stride) / dim1_stride;
        inner_idx = tid - i*dim0_stride - j*dim1_stride;
        idx = i * dim0_stride + j * dim1_stride + inner_idx;
    }
}

template
<bool ijSwapped>
__global__ void maskedFillGrad(float *y_d, float *x_d, int *mask, int dim0_shape, int dim0_stride, int dim1_shape, int dim1_stride, int offset_size, int input_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int i = tid / dim0_stride;
    int j = (tid - i*dim0_stride) / dim1_stride;
    int inner_idx = tid - i*dim0_stride - j*dim1_stride;
    int idx = i * dim0_stride + j * dim1_stride + inner_idx;

    // if masked, then gradient is zero (hence, no action)
    while (idx < input_size) {
        int mask_id;
        if (ijSwapped)
            mask_id = (j % dim1_shape) * dim0_shape + (i % dim0_shape);
        else
            mask_id = (i % dim0_shape) * dim1_shape + (j % dim1_shape);
        if (mask[mask_id] == 0) {
            x_d[idx] += y_d[idx];
        }

        tid += stride;
        i = tid / dim0_stride;
        j = (tid - i*dim0_stride) / dim1_stride;
        inner_idx = tid - i*dim0_stride - j*dim1_stride;
        idx = i * dim0_stride + j * dim1_stride + inner_idx;
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

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void permute2D(float *odata, const float *idata, int dimy, int dimx) {

  __shared__ float tile[TILE_DIM][TILE_DIM+1];
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  if (x < dimx)
    for (int j = 0; j < TILE_DIM && j < dimy - y; j += BLOCK_ROWS)
      tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*dimx + x];
  __syncthreads();
  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  if (x < dimy)
    for (int j = 0; j < TILE_DIM && j < dimx-y; j += BLOCK_ROWS)
      odata[(y+j)*dimy + x] += tile[threadIdx.x][threadIdx.y + j];
}

__global__ void permuteSim3D(float* odata, const float* idata, int dim0, int dim1, int dim2) {
  int ioffset = blockIdx.y * dim1 * dim2 + blockIdx.x * dim2;
  int ooffset = blockIdx.x * dim0 * dim2 + blockIdx.y * dim2;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x)
    odata[ooffset + i] += idata[ioffset + i];
}


// case 1: permute3D, dim2to0 dim0to2
__global__ void permute3D_dim2to0_dim0to2(float *odata, const float *idata,
         int dim0, int dim1, int dim2,
         int istr0, int istr1, int ostr0, int ostr1) {

  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int z = blockIdx.z;

  if (x < dim2)
    for (int j = 0; j < TILE_DIM && j < dim0 - y; j += BLOCK_ROWS)
      tile[threadIdx.y+j][threadIdx.x] = idata[z*istr1 + (y+j)*istr0 + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  if (x < dim0)
    for (int j = 0; j < TILE_DIM && j < dim2-y; j += BLOCK_ROWS)
      odata[(y+j)*ostr0 + z*ostr1 + x] += tile[threadIdx.x][threadIdx.y + j];
}


// case 2: permute3D, dim2to1, dim0to2
__global__ void permute3D_dim2to1_dim0to2(float *odata, const float *idata,
         int dim0, int dim1, int dim2,
         int istr0, int istr1, int ostr0, int ostr1) {

  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int z = blockIdx.z;

  if (x < dim2)
    for (int j = 0; j < TILE_DIM && j < dim1 - y; j += BLOCK_ROWS)
      tile[threadIdx.y+j][threadIdx.x] = idata[z*istr0 + (y+j)*istr1 + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  if (x < dim1)
    for (int j = 0; j < TILE_DIM && j < dim2-y; j += BLOCK_ROWS)
      odata[(y+j)*ostr0 + z*ostr1 + x] += tile[threadIdx.x][threadIdx.y + j];
}


// case 3: permute 3D, dim2to0 dim0to1
__global__ void permute3D_dim2to0_dim0to1(float *odata, const float *idata,
         int dim0, int dim1, int dim2,
         int istr0, int istr1, int ostr0, int ostr1) {

  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int z = blockIdx.z;

  if (x < dim2)
    for (int j = 0; j < TILE_DIM && j < dim0 - y; j += BLOCK_ROWS)
      tile[threadIdx.y+j][threadIdx.x] = idata[z*istr1 + (y+j)*istr0 + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  if (x < dim0)
    for (int j = 0; j < TILE_DIM && j < dim2-y; j += BLOCK_ROWS)
      odata[(y+j)*ostr1 + z*ostr0 + x] += tile[threadIdx.x][threadIdx.y + j];
}


// case 4: permute3D dim2to1 dim0to0
__global__ void permute3D_dim2to1_dim0to0(float *odata, const float *idata,
         int dim0, int dim1, int dim2,
         int istr0, int istr1, int ostr0, int ostr1) {

  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int z = blockIdx.z;

  if (x < dim2)
    for (int j = 0; j < TILE_DIM && j < dim1 - y; j += BLOCK_ROWS)
      tile[threadIdx.y+j][threadIdx.x] = idata[z*istr0 + (y+j)*istr1 + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  if (x < dim1)
    for (int j = 0; j < TILE_DIM && j < dim2-y; j += BLOCK_ROWS)
      odata[(y+j)*ostr1 + z*ostr0 + x] += tile[threadIdx.x][threadIdx.y + j];
}


// Permute 4D case 1: dim 4 is unchanged, first three dims are (0, 1, 2)
__global__ void permuteSim4DSim012(float* odata, const float* idata,
      int istr0, int istr1, int istr2,   // elide istr3/ostr3 because that is '1'
      int ostr0, int ostr1, int ostr2) { // actually ostr2 should be the same as istr2 (can remove)

  int ioffset = blockIdx.z * istr0 + blockIdx.y * istr1 + blockIdx.x * istr2;
  int ooffset = blockIdx.z * ostr0 + blockIdx.y * ostr1 + blockIdx.x * ostr2;
  for (int i = threadIdx.x; i < istr2; i += blockDim.x)
    odata[ooffset + i] += idata[ioffset + i];
}

// Permute 4D case 2: dim 4 is unchanged, first three dims are (0, 2, 1)
__global__ void permuteSim4DSim021(float* odata, const float* idata,
      int istr0, int istr1, int istr2,   // elide istr3/ostr3 because that is '1'
      int ostr0, int ostr1, int ostr2) { // actually ostr2 should be the same as istr2 (can remove)

  int ioffset = blockIdx.z * istr0 + blockIdx.y * istr1 + blockIdx.x * istr2;
  int ooffset = blockIdx.z * ostr0 + blockIdx.x * ostr1 + blockIdx.y * ostr2;
  for (int i = threadIdx.x; i < istr2; i += blockDim.x)
    odata[ooffset + i] += idata[ioffset + i];
}

// Permute 4D case 1: dim 4 is unchanged, first three dims are (1, 0, 2)
__global__ void permuteSim4DSim102(float* odata, const float* idata,
      int istr0, int istr1, int istr2,   // elide istr3/ostr3 because that is '1'
      int ostr0, int ostr1, int ostr2) { // actually ostr2 should be the same as istr2 (can remove)

  int ioffset = blockIdx.z * istr0 + blockIdx.y * istr1 + blockIdx.x * istr2;
  int ooffset = blockIdx.y * ostr0 + blockIdx.z * ostr1 + blockIdx.x * ostr2;
  for (int i = threadIdx.x; i < istr2; i += blockDim.x)
    odata[ooffset + i] += idata[ioffset + i];
}

// Permute 4D case 1: dim 4 is unchanged, first three dims are (1, 2, 0)
__global__ void permuteSim4DSim120(float* odata, const float* idata,
      int istr0, int istr1, int istr2,   // elide istr3/ostr3 because that is '1'
      int ostr0, int ostr1, int ostr2) { // actually ostr2 should be the same as istr2 (can remove)

  int ioffset = blockIdx.z * istr0 + blockIdx.y * istr1 + blockIdx.x * istr2;
  int ooffset = blockIdx.y * ostr0 + blockIdx.x * ostr1 + blockIdx.z * ostr2;
  for (int i = threadIdx.x; i < istr2; i += blockDim.x)
    odata[ooffset + i] += idata[ioffset + i];
}

// Permute 4D case 1: dim 4 is unchanged, first three dims are (2, 0, 1)
__global__ void permuteSim4DSim201(float* odata, const float* idata,
      int istr0, int istr1, int istr2,   // elide istr3/ostr3 because that is '1'
      int ostr0, int ostr1, int ostr2) { // actually ostr2 should be the same as istr2 (can remove)

  int ioffset = blockIdx.z * istr0 + blockIdx.y * istr1 + blockIdx.x * istr2;
  int ooffset = blockIdx.x * ostr0 + blockIdx.z * ostr1 + blockIdx.y * ostr2;
  for (int i = threadIdx.x; i < istr2; i += blockDim.x)
    odata[ooffset + i] += idata[ioffset + i];
}

// Permute 4D case 1: dim 4 is unchanged, first three dims are (2, 1, 0)
__global__ void permuteSim4DSim210(float* odata, const float* idata,
      int istr0, int istr1, int istr2,   // elide istr3/ostr3 because that is '1'
      int ostr0, int ostr1, int ostr2) { // actually ostr2 should be the same as istr2 (can remove)

  int ioffset = blockIdx.z * istr0 + blockIdx.y * istr1 + blockIdx.x * istr2;
  int ooffset = blockIdx.x * ostr0 + blockIdx.y * ostr1 + blockIdx.z * ostr2;
  for (int i = threadIdx.x; i < istr2; i += blockDim.x)
    odata[ooffset + i] += idata[ioffset + i];
}

// The reduced sum will only be on first thread of the warp (as opposed to xor shuffle - see warm_reduce_add above)
__inline__ __device__ float warp_reduce_sum(float val) {
   #pragma unroll
  for (int offset = (NVIDIA_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset, NVIDIA_WARP_SIZE);
  }
  return val;
}

// blockDim should be a multiple of 32
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
  const int lid = threadIdx.x % NVIDIA_WARP_SIZE;
  const int wid = threadIdx.x / NVIDIA_WARP_SIZE;
  val = warp_reduce_sum(val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (threadIdx.x < blockDim.x / NVIDIA_WARP_SIZE) ? shared[lid] : 0;
  if (wid == 0) {
    val = warp_reduce_sum(val);
  }
  return val;
}

// Taken with slight modifications from PyTorch - aten/src/ATen/native/cuda/layer_norm_kernel.cu
__global__ void layer_norm_forward(float* x, float* mean, float* rstd, float* gamma, float* beta, float* y, float eps, int vect_size) {
  __shared__ float m_shared[NVIDIA_WARP_SIZE];
  __shared__ float v_shared[NVIDIA_WARP_SIZE];
  const int i = blockIdx.x;
  float sum1 = 0;
  float sum2 = 0;
  for (int j = threadIdx.x; j < vect_size; j += blockDim.x) {
    const int index = i * vect_size + j;
    sum1 += x[index];
    sum2 += x[index] * x[index];
  }

  sum1 = block_reduce_sum(sum1, m_shared);
  sum2 = block_reduce_sum(sum2, v_shared);

  if (threadIdx.x == 0) {
    const float scale = 1.0f / vect_size;
    sum1 = sum1 * scale;
    sum2 = fmaxf(sum2 * scale - sum1 * sum1, 0);
    sum2 = rsqrtf(sum2 + eps);
    // now sum1 and sum2 has mean and rstd (for the blocK) respectively
    // store block mean and rsqrt to be used in backward pass
    mean[i] = sum1;
    rstd[i] = sum2;
//    printf("rstd[%d] = %.3f\n", i, rstd[i]);
  }
  // TODO - should I do a xor shuffle and access sum1, sum2 directly instead of mean[i], rstd[i]. Not that slow due to broadcast? (only one access per WARP)

  __syncthreads();

  for (int j = threadIdx.x; j < vect_size; j += blockDim.x) {
    const int idx = i * vect_size + j;
    y[idx] = (x[idx] - mean[i]) * rstd[i] * gamma[j] + beta[j];
//    printf("y[%d] = %.3f\n", idx, y[idx]);
  }
}

// taken from PyTorch with slight modifications - aten/src/ATen/native/cuda/layer_norm_kernel.cu
__global__ void ComputeInternalGradientsCUDAKernel(int vect_size, float* y_grad, float* x, float* gamma, float* s_grad, float* b_grad) {
  __shared__ float s_grad_shared[NVIDIA_WARP_SIZE];
  __shared__ float b_grad_shared[NVIDIA_WARP_SIZE];

  const int i = blockIdx.x;
  float sum1 = 0;
  float sum2 = 0;
  for (int j = threadIdx.x; j < vect_size; j += blockDim.x) {
    const int index = i * vect_size + j;
    sum1 += y_grad[index] * x[index] * gamma[j];
    sum2 += y_grad[index] * gamma[j];
  }

  sum1 = block_reduce_sum(sum1, s_grad_shared);
  sum2 = block_reduce_sum(sum2, b_grad_shared);
  if (threadIdx.x == 0) {
    s_grad[i] = sum1;
    b_grad[i] = sum2;
  }
}

// taken from PyTorch with slight modifications - aten/src/ATen/native/cuda/layer_norm_kernel.cu
__global__ void ComputeGradientFusedParamsCUDAKernel(int outerSize, int vect_size, float* mean, float* rstd, float* ds, float* db, float* c1, float* c2) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < outerSize) {
    float s = 1.0f / vect_size;
    float a = (db[index] * mean[index] - ds[index]) * rstd[index] * rstd[index] * rstd[index] * s;

    c1[index] = a;
    c2[index] = -(a * mean[index] + db[index] * rstd[index] * s);
  }
}

// taken from PyTorch with slight modifications - aten/src/ATen/native/cuda/layer_norm_kernel.cu
__global__ void LayerNormBackwardCUDAKernel(int vect_size, float* y_grad, float* x, float* gamma, float* a, float* b, float* c, float* x_grad) {
  const int i = blockIdx.x;
  for (int j = threadIdx.x; j < vect_size; j += blockDim.x) {
    const int64_t index = i * vect_size + j;
    x_grad[index] = a[i] * y_grad[index] * gamma[j] + b[i] * x[index] + c[i];
  }
}

// taken from PyTorch with slight modifications - aten/src/ATen/native/cuda/layer_norm_kernel.cu
__global__ void GammaBetaBackwardSimpleCUDAKernel(int outerSize, int vect_size, float* y_grad, float* x, float* mean, float* rstd, float* gamma_grad, float* beta_grad) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < vect_size) {
    float sum1 = 0;
    float sum2 = 0;
    for (int i = 0; i < outerSize; ++i) {
      const int index = i * vect_size + j;
      sum1 += y_grad[index] * (x[index] - mean[i]) * rstd[i];
      sum2 += y_grad[index];
    }
    gamma_grad[j] = sum1;
    beta_grad[j] = sum2;
  }
}

// taken from PyTorch with slight modifications - aten/src/ATen/native/cuda/layer_norm_kernel.cu
__global__ void GammaBetaBackwardCUDAKernel(int outerSize, int vect_size, float* y_grad, float* x, float* mean, float* rstd, float* dg, float* db) {
  __shared__ float g_shared[32][32 + 1]; // +1 to  avoid shared memory bank conflicts
  __shared__ float b_shared[32][32 + 1];

  const int j = blockIdx.x * blockDim.x + threadIdx.x;

  float dg_sum1 = 0;
  float dg_sum2 = 0;
  float db_sum1 = 0;
  float db_sum2 = 0;

  if (j < vect_size) {
    for (int i = threadIdx.y; i < outerSize; i += blockDim.y * 2) {
      const int i1 = i;
      const int i2 = i + blockDim.y;
      const int index1 = i1 * vect_size + j;
      const int index2 = i2 * vect_size + j;

      dg_sum1 += y_grad[index1] * (x[index1] - mean[i1]) * rstd[i1];
      db_sum1 += y_grad[index1];

      if (i2 < outerSize) {
        dg_sum2 += y_grad[index2] * (x[index2] - mean[i2]) * rstd[i2];
        db_sum2 += y_grad[index2];
      }
    }
  }

  g_shared[threadIdx.y][threadIdx.x] = dg_sum1;
  g_shared[threadIdx.y + blockDim.y][threadIdx.x] = dg_sum2;
  b_shared[threadIdx.y][threadIdx.x] = db_sum1;
  b_shared[threadIdx.y + blockDim.y][threadIdx.x] = db_sum2;
  __syncthreads();

  float sum1 = g_shared[threadIdx.x][threadIdx.y];
  float sum2 = b_shared[threadIdx.x][threadIdx.y];

  sum1 = warp_reduce_sum(sum1);
  sum2 = warp_reduce_sum(sum2);

  if (threadIdx.x == 0) {
    const int j = blockIdx.x * blockDim.x + threadIdx.y;
    if (j < vect_size) {
      dg[j] = sum1;
      db[j] = sum2;
    }
  }

  sum1 = g_shared[threadIdx.x][threadIdx.y + blockDim.y];
  sum2 = b_shared[threadIdx.x][threadIdx.y + blockDim.y];
  sum1 = warp_reduce_sum(sum1);
  sum2 = warp_reduce_sum(sum2);

  if (threadIdx.x == 0) {
    const int j = blockIdx.x * blockDim.x + threadIdx.y + blockDim.y;
    if (j < vect_size) {
      dg[j] = sum1;
      db[j] = sum2;
    }
  }
}

// TODO - move this to Scala
// taken from PyTorch with slight modifications - aten/src/ATen/native/cuda/layer_norm_kernel.cu
void layer_norm_grad(float* y_grad, float* x, float* mean, float* rstd, float* gamma, int outerSize, int vect_size,
 float* x_grad, float* gamma_grad, float* beta_grad, float* scale, float* bias, float* s_grad, float* b_grad) {
  ComputeInternalGradientsCUDAKernel<<<outerSize, 512>>>(vect_size, y_grad, x, gamma, s_grad, b_grad);

  // compute number of grids (each having 256 threads) required for outerSize number of computations
  const int B = (outerSize + 256 - 1) / 256;

  ComputeGradientFusedParamsCUDAKernel<<<B, 256>>>(outerSize, vect_size, mean, rstd, s_grad, b_grad, scale, bias);

  LayerNormBackwardCUDAKernel<<<outerSize, 256>>>(vect_size, y_grad, x, gamma, rstd, scale, bias, x_grad);

  if (outerSize < 512) {
    // For small batch size, do colwise reduce directly.
    const int B = (vect_size + 256 - 1) / 256;
    GammaBetaBackwardSimpleCUDAKernel<<<B, 256>>>(outerSize, vect_size, y_grad, x, mean, rstd, gamma_grad, beta_grad);
  } else {
    const int B = (N + 32 - 1) / 32;
    constexpr int kThreadX = 32;
    constexpr int kThreadY = 32 / 2;
    GammaBetaBackwardCUDAKernel<<<B, dim3(kThreadX, kThreadY)>>>(outerSize, vect_size, y_grad, x, mean, rstd, gamma_grad, beta_grad);
  }
}

// assumes contiguous
// in-place operation (hence, no backward pass)
// number of threads = 64
// each threads handles 4 elems (therefore, block work size = 4 * 64 = 256)
// number of blocks = scalarCount / block work size (round up)
__global__ void plus_bias_kernel(float *input, float *bias, float *output, int input_size, int bias_size) {
    int tid = threadIdx.x;

    #pragma unroll
    for(int i = 0; i < 4; i ++) {
        int idx = blockDim.x * blockIdx.x + i * NVIDIA_WARP_SIZE + tid;
        if (idx < input_size)
            output[idx] = input[idx] + bias[idx % bias_size];
    }
}

// ideally this should be handled using sum(dim=?) kernel
__global__ void plus_bias_grad(float *y_grad, float *bias_grad, int outer_size, int bias_size) {
    __shared__ float smm[NVIDIA_WARP_SIZE];

    int tid = threadIdx.x;
    float local_sum = 0;
    for(;tid < outer_size; tid += blockDim.x) {
        local_sum += y_grad[bias_size * tid + blockIdx.x];
    }
    local_sum = block_reduce_sum(local_sum, smm);

    if (threadIdx.x == 0) {
        bias_grad[blockIdx.x] += local_sum;
    }
}

// a typical grid strided loop
__global__ void relu_kernel(float *input, float *output, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(; idx < input_size; idx += stride) {
        if (input[idx] > 0) output[idx] = input[idx];
    }
}

// a typical grid strided loop
__global__ void relu_grad(float *y_grad, float *x_grad, float *x, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(; idx < input_size; idx += stride) {
        if (x[idx] > 0) x_grad[idx] += y_grad[idx];
    }
}

// assumes contiguous
__global__ void embedding_forward(float *embeddings, int *indices, float *output, int embed_size) {
    int posIdx = indices[blockIdx.x];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    for(;tid < embed_size; tid += stride) {
        output[blockIdx.x * embed_size + tid] = embeddings[posIdx*embed_size + tid];
    }
}

// TODO - this is used for cases where indices count < 768 (leader based gradient accumulation)
// from PyTorch - aten/src/ATen/native/cuda/Embedding.cu
// n = indices_size, stride = embedSize, gradients won't be updated for padding_idx
__global__ void embedding_backward_feature_kernel(int* indices, const float* __restrict__ grad, float* __restrict__ grad_weight, int n, int stride, int padding_idx)
{
  extern __shared__ char buf[];
  float* smem = (float*)buf;
  float* my_s = smem + NVIDIA_WARP_SIZE*threadIdx.y;
  int* indices_batch = (int*)(buf + sizeof(float)*NVIDIA_WARP_SIZE*blockDim.y);

  const int s = (int)stride; // OK to make int, we don't expect 2 billion+ embedding row size

  const int f = threadIdx.x + blockIdx.x*blockDim.x; // feature_dim

  for(int batch_start = 0; batch_start < n; batch_start += blockDim.x*blockDim.y)
  {
    // Entire block cooperates to load a batch of 1024 indices to process
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    if(batch_start + tid < n)
      indices_batch[tid] = (int)indices[batch_start + tid];

    int batch_end = batch_start + blockDim.x*blockDim.y < n ?
                    batch_start + blockDim.x*blockDim.y : n;

    // Loop over the batch of <= 1024 loaded indices in chunks of blockDim.y = 32
    for(int chunk_start = batch_start; chunk_start < batch_end; chunk_start += blockDim.y)
    {
      // This does double duty:  it makes sure indices_batch is ready, and it makes sure match-group
      // leaders are done with their accumulates before other warps start loading again.
      __syncthreads();

      int n_this_chunk = (batch_end - chunk_start) < blockDim.y ?
                         (batch_end - chunk_start) : blockDim.y;

      int src_row = chunk_start + threadIdx.y;
      int dst_row = indices_batch[src_row - batch_start]; // This warp's target row in grad_weight

      // All warps load their smem segments with incoming grad data
      if(src_row < n && f < s && dst_row != padding_idx)
        my_s[threadIdx.x] = grad[src_row*stride + f]; // my_s (part of smmem) is of size 32

      __syncthreads();

      // To ensure determinism, we can't just have each warp add its grad data to its dst_row.
      // We need to check if any other warps pulled grad data targeting dst_row.
      // If so, we elect the first warp in each matching group as the leader.
      // Each leader warp serializes the accumulates targeting dst_row in shared memory,
      // then finishes by adding the accumulated buffer to dst_row in grad_weight.
      if(dst_row != padding_idx && src_row < n) // Per-warp exit condition, safe with ballot_sync
      {
        int match_found_this_thread =
          (dst_row == indices_batch[chunk_start - batch_start + threadIdx.x]);
        if(threadIdx.x >= n_this_chunk)
          match_found_this_thread = 0;

        unsigned int matchmask = __ballot_sync(0xffffffff, match_found_this_thread);
        int first_remaining_peer = __ffs(matchmask) - 1;

        if(threadIdx.y == first_remaining_peer) // Nominate lowest-indexed warp as the leader
        {
          matchmask ^= (1 << first_remaining_peer);
          while(matchmask)
          {
            first_remaining_peer = __ffs(matchmask) - 1;
            my_s[threadIdx.x] += smem[threadIdx.x + NVIDIA_WARP_SIZE*first_remaining_peer];
            matchmask ^= (1 << first_remaining_peer);
          }
          if(f < s)
            grad_weight[dst_row*stride + f] += my_s[threadIdx.x];
        }
      }
    }
  }
}

// backward pass for embedding layer
// taken from - aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu
constexpr int MAX_BLOCK_SIZE = 1024;
constexpr int NROWS_PER_THREAD = 10;

__host__ __device__ __forceinline__
int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

__global__
void krn_partials_per_segment(int *ret, const int *segment_offsets, int num_of_segments, int numel) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < num_of_segments) {
    const int idx_start = segment_offsets[id];
    const int idx_end = (id == num_of_segments-1)?numel:segment_offsets[id+1];
    const int size = idx_end - idx_start;
    ret[id] = ceil_div(size, NROWS_PER_THREAD);
  }
}

__global__
void krn_partial_segment_offset(int *ret, const int *partials_per_segment, const int *partials_per_segment_offset, const int *segment_offsets,
        int num_of_segments) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < num_of_segments) {
    int idx = partials_per_segment_offset[id];
    const int num_partials = partials_per_segment[id];
    const int segment_offset = segment_offsets[id];
    for (int i=0; i<num_partials; ++i) {
      ret[idx++] = segment_offset + i * NROWS_PER_THREAD;
    }
  }
}

__global__ void compute_grad_weight(
    int *indices,
    float *gradOutput,
    ptrdiff_t numel,
    int stride,
    int* segment_offsets,
    int num_of_segments,
    float *grad_weight_per_segment,
    const int stride_warped) {

  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int id = gid / stride_warped;
  const int startFeature = gid % stride_warped;
  if (startFeature >= stride) {
    return;
  }
  if (id >= num_of_segments) {
    return;
  }
  const int idx_begin = segment_offsets[id];
  const int idx_end = (id == num_of_segments-1)?numel:segment_offsets[id+1];

  float weight = 0;
  for (int idx=idx_begin; idx < idx_end; ++idx) {
    const int target_row = indices[idx];
    weight += gradOutput[target_row * stride + startFeature];
  }
  grad_weight_per_segment[id * stride + startFeature] = weight;
}

__global__ void sum_and_scatter(
    int *input, float *gradWeight, int stride,
    int* segment_offsets, int num_of_segments,
    const float *grad_weight_per_segment,
    const int *segment_sizes_offsets, int num_of_partial_segments,
    const int padding_idx,
    const int stride_warped) {

  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int id = gid / stride_warped;
  const int startFeature = gid % stride_warped;
  if (startFeature >= stride) {
    return;
  }
  if (id >= num_of_segments) {
    return;
  }

  const int idx_begin = segment_sizes_offsets[id];
  const int idx_end = (id == num_of_segments-1)?num_of_partial_segments:segment_sizes_offsets[id+1];
  float weight = 0;
  for (int idx=idx_begin; idx < idx_end; ++idx) {
    weight += grad_weight_per_segment[idx*stride + startFeature];
  }
  int target_row = input[segment_offsets[id]];
  if (target_row != padding_idx) {
    gradWeight[target_row * stride + startFeature] += weight;
  }
}


void embedding_backward_cuda_kernel(float *grad, float *grad_weight, int embed_size, int *orig_indices, int *sorted_indices, int num_indices, int padding_idx) {
    const ptrdiff_t numel = num_indices;
    const int stride = embed_size;

    // Compute the number of segments and their start position so that we do not have to
    // spawn a warp per index. In this context, a segment is a number of rows that should
    // be summarized.
    // Unit: index in `sorted_indices` and `orig_indices
    int *segment_offsets = (int *) myGpuMalloc(num_indices * sizeof(int));
    int num_of_segments;

    {
        auto sorted_indices_dev = thrust::device_ptr<int>(sorted_indices);
        int *dummy = (int *) myGpuMalloc(num_indices * sizeof(int));
        auto dummy_dev = thrust::device_ptr<int>(dummy);

        auto ends = thrust::unique_by_key_copy(sorted_indices_dev, sorted_indices_dev + numel, thrust::make_counting_iterator(0),
                dummy_dev, thrust::device_ptr<int>(segment_offsets));
        num_of_segments = thrust::get<0>(ends) - dummy_dev;
    }

    // num segments will be number of unique keys
    // and the counts will be the value of first key appearance

    // We split the segments up into sizes of `NROWS_PER_THREAD`
    // Compute the number partial-segments per segment (some partial-segments
    // may not be the full `NROWS_PER_THREAD` number of rows)
    // auto partials_per_segment = at::empty({num_of_segments}, orig_indices.options());
    int *partials_per_segment = (int *) myGpuMalloc(num_of_segments * sizeof(int));

    {
        krn_partials_per_segment<<<ceil_div(num_of_segments, 32), 32>>> (partials_per_segment, segment_offsets, num_of_segments, numel);
    }

    // In order to compute `partial_segment_offset`, which is the start index
    // of each partial-segment in `sorted_indices`, we need to compute the
    // start position of each _segment_ in `partial_segment_offset`.
    // Unit: index in `partial_segment_offset`
    // auto partials_per_segment_offset = at::empty({num_of_segments}, orig_indices.options());
    int *partials_per_segment_offset = (int *) myGpuMalloc(num_of_segments * sizeof(int));

    thrust::exclusive_scan(
        thrust::device_ptr<int>(partials_per_segment),
        thrust::device_ptr<int>(partials_per_segment + num_of_segments),
        thrust::device_ptr<int>(partials_per_segment_offset));

    // The total number of partial-segments is the sum of `partials_per_segment_offset`
    int partials_per_segment_last;
    int partials_per_segment_offset_last;

    cudaMemcpy((void **) &partials_per_segment_last, &partials_per_segment[num_of_segments-1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy((void **) &partials_per_segment_offset_last, &partials_per_segment_offset[num_of_segments-1], sizeof(int), cudaMemcpyDeviceToHost);

    const int num_of_partial_segments = partials_per_segment_last + partials_per_segment_offset_last;
//    const int num_of_partial_segments = partials_per_segment[num_of_segments-1] + partials_per_segment_offset[num_of_segments-1];

    // Now we can compute the start position of each partial-segment
    // Unit: index in `sorted_indices` and `orig_indices`
    // auto partial_segment_offset = at::empty({num_of_partial_segments}, orig_indices.options());
    int *partial_segment_offset = (int *) myGpuMalloc(num_of_partial_segments * sizeof(int));
    {
        krn_partial_segment_offset<<<ceil_div(num_of_segments, 32), 32>>> (partial_segment_offset, partials_per_segment,
                partials_per_segment_offset, segment_offsets, num_of_segments);
    }

    const int stride_warped = ceil_div(stride, NVIDIA_WARP_SIZE)*NVIDIA_WARP_SIZE;
    const int block = std::min(stride_warped, MAX_BLOCK_SIZE);
    const int grid = ceil_div(num_of_partial_segments*stride_warped, block);

    {
        float *grad_weight_per_segment = (float *) myGpuMalloc(num_of_partial_segments * stride * sizeof(float));
        // TODO - can free this memory after the two kernel invocations
        // Compute the sum of each partial-segment and handle bags
        compute_grad_weight<<<grid, block>>>(
            orig_indices,
            grad,
            numel, stride,
            partial_segment_offset,
            num_of_partial_segments,
            grad_weight_per_segment,
            stride_warped);

        // Finally, we sum all the partial-sums and scatter them
        // into `grad_weight`.
        const int grid2 = ceil_div(num_of_segments*stride_warped, block);
            sum_and_scatter<<<grid2, block>>>(
            sorted_indices,
            grad_weight,
            stride,
            segment_offsets,
            num_of_segments, grad_weight_per_segment,
            partials_per_segment_offset,
            num_of_partial_segments,
            padding_idx,
            stride_warped);
    }
}

void embedding_dense_backward_cuda(float *grad, float *grad_weight, int embed_size, int *indices, int num_indices, int padding_idx) {
    int *orig_indices = (int *) myGpuMalloc(num_indices * sizeof(int));

    using device_ptr = thrust::device_ptr<int>;
    // Sort the inputs into sorted with the corresponding indices; we
    // don't need a stable or multidimensional sort, so just use Thrust
    // directly
    {
        // TODO - can create a custom allocator which uses myGpuMalloc
        // auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
        // auto policy = thrust::cuda::par().on(stream);

        // Fill sortedOrigIndices with sequential indices
        auto count_iter = thrust::counting_iterator<int>(0);
        // device_ptr says the pointer is a pointer in the device (it is NOT moving the data from host to device)
        auto orig_data = device_ptr(orig_indices);
        thrust::copy(count_iter, count_iter + num_indices, orig_data);

        // Sort; a stable sort is not required
        auto sorted_data = device_ptr(indices);
        thrust::sort_by_key(sorted_data, sorted_data + num_indices, orig_data, thrust::less<int>());
    }

    return embedding_backward_cuda_kernel(grad, grad_weight, embed_size, orig_indices, indices, num_indices, padding_idx);
}

// mask of shape i_len x j_len
__global__ void create_attention_mask(int *mask, int i_size, int j_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < i_size && j < j_size && i <= j) mask[i*j_size + j] = 1;
}

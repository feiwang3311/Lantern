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

long HEAP_SIZE = 1073741824; // 4294967296; // 8589934592; // 10737418240;
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
    __shared__ float buffer[64];

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
    float *gradInput_t = gradInput + size * blockIdx.x;
    float *gradOutput_t = gradOutput + size * blockIdx.x;
    float *output_t = output + size * blockIdx.x;

    int start = threadIdx.x;
    int end = size;
    int stride = blockDim.x;

    __shared__ float buffer[64];

    // compute the sum (gradOutput * output sum)
    for(int i=start; i < end; i += stride) {
        buffer[threadIdx.x] = gradOutput_t[i] * output_t[i];
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

template <int WARP_BATCH, int WARP_SIZE>
__device__ __forceinline__ void warp_reduce_max(float* sum) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            float b = __shfl_xor_sync(0xffffffff, sum[i], offset, WARP_SIZE);
            if (sum[i] < b) sum[i] = b;
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
                grad_reg[i][it] = float(0);
                output_reg[i][it] = float(0);
            }
        }
    }

    float sum[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        sum[i] = grad_reg[i][0];
        #pragma unroll
        for (int it = 1;  it < WARP_ITERATIONS;  ++it) {
            sum[i] += grad_reg[i][it];
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
                    gradInput[i*element_count+it*WARP_SIZE] = (grad_reg[i][it] - expf(output_reg[i][it]) * sum[i]);
                } else {
                    gradInput[i*element_count+it*WARP_SIZE] = (grad_reg[i][it] - output_reg[i][it] * sum[i]);
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
    // constexpr int log2_elements = log2_ceil(softmax_elements);
    int log2_elements = (int) ceil(log2(softmax_elements));
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
    // int log2_elements = log2_ceil(softmax_elements);
    int log2_elements = (int) ceil(log2(softmax_elements));

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

// this assumes in is contiguous
__global__ void maskedFill3D(float *in, float* out, int *mask, float value, int mask_size, int input_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(; tid < input_size; tid += stride) {
        if (mask[tid % mask_size] != 0) {
            out[tid] = value;
        } else {
            out[tid] = in[tid];
        }
    }
}

// update the gradients of x based on y (y is coming from backward pass)
__global__ void maskedFill3DGrad(float *y_d, float *x_d, int *mask, int mask_size, int input_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // if masked, then gradient is zero (hence, no action)
    for(; tid < input_size; tid += stride) {
        if (mask[tid % mask_size] == 0) {
            x_d[tid] += y_d[tid];
        }
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



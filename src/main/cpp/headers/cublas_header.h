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



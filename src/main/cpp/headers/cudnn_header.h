
#define CUDNN_CALL(f) { \
  cudnnStatus_t stat = (f); \
  if (stat != CUDNN_STATUS_SUCCESS) { \
    fprintf(stderr, "cuDNN error occurred: %s %d (%s:%d)\n", \
            cudnnGetErrorString(stat), stat, __FILE__, __LINE__); \
    exit(stat); \
  } \
}

// FIXME(feiw) handle this global states somewhere else
// def convAlgoTemplate(x: Int): String =
// s"""
// |cudnnConvolutionFwdAlgo_t       algo_$x      = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;     bool init_algo_$x      = false;
// |cudnnConvolutionBwdDataAlgo_t   algo_bwd_$x  = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;   bool init_algo_bwd_$x  = false;
// |cudnnConvolutionBwdFilterAlgo_t algo_bwf_$x  = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0; bool init_algo_bwf_$x  = false;
// """.stripMargin

// def convOpIndex() = ((0 until 12): Range).toList
// def buildConvAlgoTemplate(): String = convOpIndex.map(convAlgoTemplate).mkString("\n")


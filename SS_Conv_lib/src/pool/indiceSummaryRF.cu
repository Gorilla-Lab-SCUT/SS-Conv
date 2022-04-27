/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

c++/cuda source code for average sparse pooling. 
Written by Hongyang Li and Jiehong Lin
# Modified from https://github.com/traveller59/spconv/tree/v1.1
--------------------------------------------------------
*/
#include <ATen/ATen.h>
#include <chrono>
#include <limits>
#include "indice_avgpool.h"
#include "mp_helper.h"
#include "helper_kernel.cu.h"
#include "helper_launch.h"
#include "tensorview.h"
#include <type_traits>
#include "indice_avgpool_functor.h"

template <typename T>
__global__ void summaryRFFwdKernel(const T *indicesIn,
                                   const T *indicesOut, 
                                   int size,
                                   T *num_RF) {
  // if (threadIdx.x == 1){
  //   printf("blockDim.x: %d, gridDim.x: %d \n", blockDim.x, gridDim.x);
  // }
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for(int i = idx; i<size; i+=blockDim.x * gridDim.x){
    // if (threadIdx.x == 1){
    //   printf("iter once \n");
    // }
    num_RF[indicesOut[i]] += 1;
  }
  }

namespace functor {
template <typename T>
struct SummaryRFForwardFunctor<tv::GPU, T> {
  void operator()(  const tv::GPU &d, 
                    tv::TensorView<const T> indices, 
                    tv::TensorView<T> num_RF, 
                    int size
                ) {
    int num_thread = 256;
    int num_iter   = 8;
    int num_block  = size / (256*num_iter)+1;
    // printf("size: %d, num_iter: %d, num_block: %d", size, num_iter, num_block);
    if (size<=0){
      return;
    }
    else{
      summaryRFFwdKernel<T>
                          <<<dim3(num_block, 1), dim3(num_thread, 1)>>>
                            (indices.subview(0).data(), 
                             indices.subview(1).data(),
                             size,
                             num_RF.data());
      TV_CHECK_CUDA_ERR();
    }
  }
};
} // namespace functor

#define DECLARE_GPU_SPECS_T_INDEX(T) \
  template struct functor::SummaryRFForwardFunctor<tv::GPU, T>;

#define DECLARE_GPU_SPECS() DECLARE_GPU_SPECS_T_INDEX(int);

// DECLARE_GPU_SPECS(float);
// DECLARE_GPU_SPECS(double);
// DECLARE_GPU_SPECS(at::Half);
DECLARE_GPU_SPECS();

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_T_INDEX





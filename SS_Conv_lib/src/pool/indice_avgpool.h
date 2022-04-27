/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

c++/cuda source code for average sparse pooling. 
Written by Hongyang Li and Jiehong Lin
# Modified from https://github.com/traveller59/spconv/tree/v1.1
--------------------------------------------------------
*/
#include <cuda_runtime_api.h>
#include <torch/script.h>
#include <torch_utils.h>
#include "timer.h"
#include "indice_avgpool_functor.h"

torch::Tensor indiceSummaryRF(torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numAct);
template <typename T>
torch::Tensor indiceAvgPool(torch::Tensor features, torch::Tensor indicePairs,
                          torch::Tensor indiceNum, int64_t numAct, torch::Tensor summaryrf) {
  auto device = features.device().type();
  auto kernelVolume = indicePairs.size(0);
  auto numInPlanes = features.size(1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  torch::Tensor output = torch::zeros({numAct, numInPlanes}, options);
  double totalTime = 0;
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data<int>()[i];
    if (nHot <= 0) {
      continue;
    }
    // auto timer = spconv::CudaContextTimer<>();
    if (device == torch::kCPU) {
      // functor::SparseAvgPoolForwardFunctor<tv::CPU, T, int> forwardFtor;
      // forwardFtor(tv::CPU(), tv::torch2tv<T>(output),
      //             tv::torch2tv<const T>(features),
      //             tv::torch2tv<const int>(indicePairs).subview(i), nHot);
      return output;
    } else {
      functor::SparseAvgPoolForwardFunctor<tv::GPU, T, int> forwardFtor;
      forwardFtor(tv::TorchGPU(), tv::torch2tv<T>(output),
                  tv::torch2tv<const T>(features),
                  tv::torch2tv<const int>(indicePairs).subview(i), nHot,
                  tv::torch2tv<const int>(summaryrf));
      TV_CHECK_CUDA_ERR();
    }
    // totalTime += timer.report() / 1000.0;
  }
  // std::cout << "maxpool forward time " << totalTime << std::endl;
  return output;
}

template <typename T>
torch::Tensor indiceAvgPoolBackward(torch::Tensor features,
                                  torch::Tensor outFeatures,
                                  torch::Tensor outGrad, torch::Tensor indicePairs,
                                  torch::Tensor indiceNum, torch::Tensor summaryrf) {
  auto device = features.device().type();
  auto numInPlanes = features.size(1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  torch::Tensor inputGrad = torch::zeros(features.sizes(), options);
    auto kernelVolume = indicePairs.size(0);
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data<int>()[i];
    if (nHot <= 0) {
      continue;
    }
    if (device == torch::kCPU) {
      // functor::SparseAvgPoolBackwardFunctor<tv::CPU, T, int> backwardFtor;
      // backwardFtor(tv::CPU(), tv::torch2tv<const T>(outFeatures),
      //              tv::torch2tv<const T>(features),
      //              tv::torch2tv<const T>(outGrad), tv::torch2tv<T>(inputGrad),
      //              tv::torch2tv<const int>(indicePairs).subview(i), nHot,
      //              tv::torch2tv<const int>(summaryrf));
      return inputGrad;
    } else {
      functor::SparseAvgPoolBackwardFunctor<tv::GPU, T, int> backwardFtor;
      backwardFtor(tv::TorchGPU(), tv::torch2tv<const T>(outFeatures),
                   tv::torch2tv<const T>(features),
                   tv::torch2tv<const T>(outGrad), tv::torch2tv<T>(inputGrad),
                   tv::torch2tv<const int>(indicePairs).subview(i), nHot,
                   tv::torch2tv<const int>(summaryrf));
      TV_CHECK_CUDA_ERR();
    }
  }
  return inputGrad;
}

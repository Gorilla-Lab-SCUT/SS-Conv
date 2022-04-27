/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

c++/cuda source code for sparse max pool. 
Written by Hongyang Li and Jiehong Lin
# Modified from https://github.com/traveller59/spconv/tree/v1.1
--------------------------------------------------------
*/
#include <cuda_runtime_api.h>
#include <torch/script.h>
#include "torch_utils.h"
#include "timer.h"
#include "indice_maxpool_functor.h"
template <typename T>
torch::Tensor indiceMaxPool(torch::Tensor features, torch::Tensor indicePairs,
                          torch::Tensor indiceNum, int64_t numAct) {
  auto device = features.device().type();
  auto kernelVolume = indicePairs.size(0);
  auto numInPlanes = features.size(1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  torch::Tensor output = torch::zeros({numAct, numInPlanes}, options);
  // torch::Tensor smallest = torch::tensor({-1e50}, options);
  // output = output + smallest;
  double totalTime = 0;
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data<int>()[i];
    if (nHot <= 0) {
      continue;
    }
    // auto timer = spconv::CudaContextTimer<>();
    if (device == torch::kCPU) {
      functor::SparseMaxPoolForwardFunctor<tv::CPU, T, int> forwardFtor;
      forwardFtor(tv::CPU(), tv::torch2tv<T>(output),
                  tv::torch2tv<const T>(features),
                  tv::torch2tv<const int>(indicePairs).subview(i), nHot);
    } else {
      functor::SparseMaxPoolForwardFunctor<tv::GPU, T, int> forwardFtor;
      forwardFtor(tv::TorchGPU(), tv::torch2tv<T>(output),
                  tv::torch2tv<const T>(features),
                  tv::torch2tv<const int>(indicePairs).subview(i), nHot);
      TV_CHECK_CUDA_ERR();
    }
    // totalTime += timer.report() / 1000.0;
  }
  // std::cout << "maxpool forward time " << totalTime << std::endl;
  return output;
}

template <typename T>
torch::Tensor indiceFieldMaxPool(torch::Tensor features, torch::Tensor indicePairs,
                          torch::Tensor indiceNum, int64_t numAct,
                          torch::Tensor feature_norms) {
  auto device = features.device().type();
  auto kernelVolume = indicePairs.size(0);
  auto numInPlanes = features.size(1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  torch::Tensor output = torch::zeros({numAct, numInPlanes}, options);
  torch::Tensor output_norm = torch::zeros({numAct, numInPlanes}, options);
  torch::Tensor smallest = torch::tensor({-1e50}, options);
  output_norm = output_norm + smallest; 
  output = output + smallest;
  double totalTime = 0;
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data<int>()[i];
    if (nHot <= 0) {
      continue;
    }
    // auto timer = spconv::CudaContextTimer<>();
    if (device == torch::kCPU) {
      functor::SparseFieldMaxPoolForwardFunctor<tv::CPU, T, int> forwardFtor;
      forwardFtor(tv::CPU(), tv::torch2tv<T>(output),
                  tv::torch2tv<const T>(features),
                  tv::torch2tv<const int>(indicePairs).subview(i), nHot,
                  tv::torch2tv<const T>(feature_norms), tv::torch2tv<T>(output_norm));
    } else {
      functor::SparseFieldMaxPoolForwardFunctor<tv::GPU, T, int> forwardFtor;
      forwardFtor(tv::TorchGPU(), tv::torch2tv<T>(output),
                  tv::torch2tv<const T>(features),
                  tv::torch2tv<const int>(indicePairs).subview(i), nHot,
                  tv::torch2tv<const T>(feature_norms), tv::torch2tv<T>(output_norm));
      TV_CHECK_CUDA_ERR();
    }
    // totalTime += timer.report() / 1000.0;
  }
  // std::cout << "maxpool forward time " << totalTime << std::endl;
  return output;
}

template <typename T>
torch::Tensor indiceMaxPoolBackward(torch::Tensor features,
                                  torch::Tensor outFeatures,
                                  torch::Tensor outGrad, torch::Tensor indicePairs,
                                  torch::Tensor indiceNum) {
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
      functor::SparseMaxPoolBackwardFunctor<tv::CPU, T, int> backwardFtor;
      backwardFtor(tv::CPU(), tv::torch2tv<const T>(outFeatures),
                   tv::torch2tv<const T>(features),
                   tv::torch2tv<const T>(outGrad), tv::torch2tv<T>(inputGrad),
                   tv::torch2tv<const int>(indicePairs).subview(i), nHot);
    } else {
      functor::SparseMaxPoolBackwardFunctor<tv::GPU, T, int> backwardFtor;
      backwardFtor(tv::TorchGPU(), tv::torch2tv<const T>(outFeatures),
                   tv::torch2tv<const T>(features),
                   tv::torch2tv<const T>(outGrad), tv::torch2tv<T>(inputGrad),
                   tv::torch2tv<const int>(indicePairs).subview(i), nHot);
      TV_CHECK_CUDA_ERR();
    }
  }
  return inputGrad;
}

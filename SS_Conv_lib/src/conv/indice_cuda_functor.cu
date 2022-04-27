/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

c++/cuda source code for sparse convolution. 
Written by Hongyang Li and Jiehong Lin
# Modified from https://github.com/traveller59/spconv/tree/v1.1
--------------------------------------------------------
*/

#include <ATen/ATen.h>
#include <chrono>
#include <limits>
#include <type_traits>
#include "timer.h"
#include "mp_helper.h"
#include "helper_launch.h"
#include "tensorview.h"
#include "indice_functor.h"
#include "helper_kernel.cu.h"
#include "geometry.h"

template <typename Index, typename IndexGrid, unsigned NDim,
          int KernelMaxVolume = 256>
__global__ void prepareIndicePairsKernel(
    tv::TensorView<const Index> indicesIn, tv::TensorView<Index> indicesOut,
    tv::TensorView<IndexGrid> gridsOut, tv::TensorView<Index> indicePairs,
    tv::TensorView<Index> indiceNum, tv::TensorView<Index> indicePairUnique,
    const tv::SimpleVector<Index, NDim> kernelSize,
    const tv::SimpleVector<Index, NDim> stride,
    const tv::SimpleVector<Index, NDim> padding,
    const tv::SimpleVector<Index, NDim> dilation,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {
  auto numActIn = indicesIn.dim(0);
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index kernelVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    kernelVolume *= kernelSize[i];
  }
  Index numValidPoints = 0;
  Index validPoints[KernelMaxVolume * (NDim + 1)];
  Index *pointPtr = nullptr;
  auto indicePairsDim2 = indicePairs.dim(2);
  Index index;
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    numValidPoints = getValidOutPos<Index, NDim>(
        indicesIn.data() + ix * (NDim + 1) + 1, kernelSize.data(),
        stride.data(), padding.data(), dilation.data(), outSpatialShape.data(),
        validPoints);
    for (Index i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
      auto oldNum = atomicAdd(indiceNum.data() + offset, Index(1));
      indicePairs(offset, 0, oldNum) = ix;
      index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape.data()) +
              spatialVolume * indicesIn(ix, 0);
      indicePairs(offset, 1, oldNum) = index;
      indicePairUnique[offset * indicePairsDim2 + oldNum] = index;
    }
  }
}

template <typename Index, typename IndexGrid, unsigned NDim,
          int KernelMaxVolume = 256>
__global__ void prepareDeConvIndicePairsKernel(
    tv::TensorView<const Index> indicesIn, tv::TensorView<Index> indicesOut,
    tv::TensorView<IndexGrid> gridsOut, tv::TensorView<Index> indicePairs,
    tv::TensorView<Index> indiceNum, tv::TensorView<Index> indicePairUnique,
    const tv::SimpleVector<Index, NDim> kernelSize,
    const tv::SimpleVector<Index, NDim> stride,
    const tv::SimpleVector<Index, NDim> padding,
    const tv::SimpleVector<Index, NDim> dilation,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {
  auto numActIn = indicesIn.dim(0);
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index kernelVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    kernelVolume *= kernelSize[i];
  }
  Index numValidPoints = 0;
  Index validPoints[KernelMaxVolume * (NDim + 1)];
  Index *pointPtr = nullptr;
  auto indicePairsDim2 = indicePairs.dim(2);
  Index index;
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    numValidPoints = getValidOutPosTranspose<Index, NDim>(
        indicesIn.data() + ix * (NDim + 1) + 1, kernelSize.data(),
        stride.data(), padding.data(), dilation.data(), outSpatialShape.data(),
        validPoints);
    for (Index i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
      auto oldNum = atomicAdd(indiceNum.data() + offset, Index(1));
      indicePairs(offset, 0, oldNum) = ix;
      index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape.data()) +
              spatialVolume * indicesIn(ix, 0);
      indicePairs(offset, 1, oldNum) = index;
      indicePairUnique[offset * indicePairsDim2 + oldNum] = index;
    }
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
__global__ void assignGridAndIndiceOutKernel(
    tv::TensorView<Index> indicesOut, tv::TensorView<IndexGrid> gridsOut,
    int numAct, tv::TensorView<Index> indicePairs,
    tv::TensorView<Index> indicePairUnique,
    const tv::SimpleVector<Index, NDim> outSpatialShape, int batchSize) {

  Index index;
  auto indicesOutPtr = indicesOut.data();
  for (int ix : tv::KernelLoopX<int>(numAct)) {
    index = indicePairUnique[ix];
    gridsOut[index] = ix;
    index = tv::rowArrayIdxInv<Index, NDim>(
        index, indicesOutPtr + ix * (NDim + 1) + 1, outSpatialShape.data());
    indicesOut[ix * (NDim + 1)] = index % batchSize;
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
__global__ void
assignIndicePairsKernel(tv::TensorView<Index> indicesOut,
                        tv::TensorView<IndexGrid> gridsOut, int numActIn,
                        tv::TensorView<Index> indicePairs,
                        tv::TensorView<Index> indicePairUnique,
                        const tv::SimpleVector<Index, NDim> outSpatialShape) {

  Index index;
  int kernelVolume = indicePairs.dim(0);
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    for (int i = 0; i < kernelVolume; ++i) {
      index = indicePairs(i, 1, ix);
      if (index > -1) {
        indicePairs(i, 1, ix) = gridsOut[index];
      }
    }
  }
}

template <typename Index, typename IndexGrid, unsigned NDim,
          int KernelMaxVolume = 256>
__global__ void
prepareSubMGridKernel(tv::TensorView<const Index> indicesIn,
                  tv::TensorView<IndexGrid> gridsOut,
                  const tv::SimpleVector<Index, NDim> outSpatialShape) {
  auto numActIn = indicesIn.dim(0);
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index index = 0;
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    index = tv::rowArrayIdx<Index, NDim>(indicesIn.data() + ix * (NDim + 1) + 1,
                                         outSpatialShape.data()) +
            spatialVolume * indicesIn(ix, 0);
    gridsOut[index] = ix;
  }
}

template <typename Index, typename IndexGrid, unsigned NDim,
          int KernelMaxVolume = 256>
__global__ void getSubMIndicePairsKernel(
    tv::TensorView<const Index> indicesIn, tv::TensorView<IndexGrid> gridsOut,
    tv::TensorView<Index> indicePairs, tv::TensorView<Index> indiceNum,
    const tv::SimpleVector<Index, NDim> kernelSize,
    const tv::SimpleVector<Index, NDim> stride,
    const tv::SimpleVector<Index, NDim> padding,
    const tv::SimpleVector<Index, NDim> dilation,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {
  auto numActIn = indicesIn.dim(0);
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index numValidPoints = 0;
  Index validPoints[KernelMaxVolume * (NDim + 1)];
  Index *pointPtr = nullptr;
  Index index = 0;
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    numValidPoints = getValidOutPos<Index, NDim>(
        indicesIn.data() + ix * (NDim + 1) + 1, kernelSize.data(),
        stride.data(), padding.data(), dilation.data(), outSpatialShape.data(),
        validPoints);
    for (int i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
      index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape.data()) +
              spatialVolume * indicesIn(ix, 0);
      if (gridsOut[index] > -1) {
        auto oldNum = atomicAdd(indiceNum.data() + offset, Index(1));
        indicePairs(offset, 1, oldNum) = gridsOut[index];
        indicePairs(offset, 0, oldNum) = ix;
      }
    }
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
__global__ void resetGridKernel(const Index *indicePairUnique,
                                tv::TensorView<IndexGrid> gridsOut,
                                int numAct) {
  for (int ix : tv::KernelLoopX<int>(numAct)) {
    gridsOut[indicePairUnique[ix]] = -1;
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
__global__ void
resetGridSubMKernel(const Index *indices, tv::TensorView<IndexGrid> gridsOut,
                    const tv::SimpleVector<Index, NDim> outSpatialShape,
                    int numAct) {
  int outSpatialShapeReg[NDim];
  for (int i = 0; i < NDim; ++i) {
    outSpatialShapeReg[i] = outSpatialShape[i];
  }
  Index spatialVolume = 1;
  auto indsPtr = indices;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index index;
  for (int ix : tv::KernelLoopX<int>(numAct)) {
    indsPtr = indices + ix * (NDim + 1);
    index = tv::rowArrayIdx<Index, NDim>(indsPtr + 1, outSpatialShapeReg);
    gridsOut[index + spatialVolume * indsPtr[0]] = -1;
  }
}


namespace functor {
template <typename Index, typename IndexGrid, unsigned NDim>
struct CreateConvIndicePairFunctorP1<tv::GPU, Index, IndexGrid, NDim> {
  Index operator()(const tv::GPU &d, tv::TensorView<const Index> indicesIn,
                   tv::TensorView<Index> indicesOut,
                   tv::TensorView<IndexGrid> gridsOut,
                   tv::TensorView<Index> indicePairs,
                   tv::TensorView<Index> indiceNum,
                   tv::TensorView<Index> indicePairUnique,
                   const tv::SimpleVector<Index, NDim> kernelSize,
                   const tv::SimpleVector<Index, NDim> stride,
                   const tv::SimpleVector<Index, NDim> padding,
                   const tv::SimpleVector<Index, NDim> dilation,
                   const tv::SimpleVector<Index, NDim> outSpatialShape,
                   bool transpose) {
    Index batchSize = gridsOut.dim(0);
    auto numActIn = indicesIn.dim(0);
    if (numActIn == 0)
      return 0;
    // auto timer = spconv::CudaContextTimer<>();
    if (transpose)
      prepareDeConvIndicePairsKernel<Index, IndexGrid, NDim, 256>
          <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
             d.stream()>>>(indicesIn, indicesOut, gridsOut, indicePairs,
                           indiceNum, indicePairUnique, kernelSize, stride,
                           padding, dilation, outSpatialShape);
    else
      prepareIndicePairsKernel<Index, IndexGrid, NDim, 256>
          <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
             d.stream()>>>(indicesIn, indicesOut, gridsOut, indicePairs,
                           indiceNum, indicePairUnique, kernelSize, stride,
                           padding, dilation, outSpatialShape);
    TV_CHECK_CUDA_ERR();
    // std::cout << "p1 gene time " << timer.report() / 1000.0 << std::endl;
    return 1;
  }
};

template <typename Index, typename IndexGrid, unsigned NDim>
struct CreateConvIndicePairFunctorP2<tv::GPU, Index, IndexGrid, NDim> {
  Index operator()(const tv::GPU &d, tv::TensorView<const Index> indicesIn,
                   tv::TensorView<Index> indicesOut,
                   tv::TensorView<IndexGrid> gridsOut,
                   tv::TensorView<Index> indicePairs,
                   tv::TensorView<Index> indiceNum,
                   tv::TensorView<Index> indicePairUnique,
                   const tv::SimpleVector<Index, NDim> outSpatialShape,
                   bool transpose, bool resetGrid) {
    Index batchSize = gridsOut.dim(0);
    auto kernelVolume = indicePairs.dim(0);
    auto numActIn = indicesIn.dim(0);
    if (numActIn == 0)
      return 0;
    Index numAct = indicePairUnique.dim(0) - 1;
    assignGridAndIndiceOutKernel<Index, IndexGrid, NDim>
        <<<tv::launch::getBlocks(numAct), tv::launch::CUDA_NUM_THREADS, 0,
           d.stream()>>>(indicesOut, gridsOut, numAct, indicePairs,
                         indicePairUnique, outSpatialShape, batchSize);
    TV_CHECK_CUDA_ERR();
    assignIndicePairsKernel<Index, IndexGrid, NDim>
        <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
           d.stream()>>>(indicesOut, gridsOut, numActIn, indicePairs,
                         indicePairUnique, outSpatialShape);
    TV_CHECK_CUDA_ERR();
    if (resetGrid) {
      resetGridKernel<Index, IndexGrid, NDim>
          <<<tv::launch::getBlocks(numAct), tv::launch::CUDA_NUM_THREADS, 0,
             d.stream()>>>(indicePairUnique.data(), gridsOut, numAct);
      TV_CHECK_CUDA_ERR();
    }
    return numAct;
  }
};

template <typename Index, typename IndexGrid, unsigned NDim>
struct CreateSubMIndicePairFunctor<tv::GPU, Index, IndexGrid, NDim> {
  Index operator()(const tv::GPU &d, tv::TensorView<const Index> indicesIn,
                   tv::TensorView<IndexGrid> gridsOut,
                   tv::TensorView<Index> indicePairs,
                   tv::TensorView<Index> indiceNum,
                   const tv::SimpleVector<Index, NDim> kernelSize,
                   const tv::SimpleVector<Index, NDim> stride,
                   const tv::SimpleVector<Index, NDim> padding,
                   const tv::SimpleVector<Index, NDim> dilation,
                   const tv::SimpleVector<Index, NDim> outSpatialShape,
                   bool transpose, bool resetGrid) {
    auto numActIn = indicesIn.dim(0);
    if (numActIn == 0)
      return 0;
    // auto timer = spconv::CudaContextTimer<>();
    prepareSubMGridKernel<Index, IndexGrid, NDim>
        <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
           d.stream()>>>(indicesIn, gridsOut, outSpatialShape);
    TV_CHECK_CUDA_ERR();
    getSubMIndicePairsKernel<Index, IndexGrid, NDim>
        <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
           d.stream()>>>(indicesIn, gridsOut, indicePairs, indiceNum,
                         kernelSize, stride, padding, dilation, outSpatialShape);
    TV_CHECK_CUDA_ERR();
    // std::cout << "subm gene time " << timer.report() / 1000.0 << std::endl;
    if (resetGrid) {
      resetGridSubMKernel<Index, IndexGrid, NDim>
          <<<tv::launch::getBlocks(numActIn), tv::launch::CUDA_NUM_THREADS, 0,
             d.stream()>>>(indicesIn.data(), gridsOut, outSpatialShape, numActIn);
      TV_CHECK_CUDA_ERR();
    }
    return numActIn;
  }
};
} // namespace functor

#define DECLARE_GPU_SPECS_INDEX_NDIM(Index, NDIM)                              \
  template struct functor::CreateConvIndicePairFunctor<tv::GPU, Index, int,    \
                                                       NDIM>;                  \
  template struct functor::CreateConvIndicePairFunctorP1<tv::GPU, Index, int,  \
                                                         NDIM>;                \
  template struct functor::CreateConvIndicePairFunctorP2<tv::GPU, Index, int,  \
                                                         NDIM>;                \
  template struct functor::CreateSubMIndicePairFunctor<tv::GPU, Index, int,    \
                                                       NDIM>;

#define DECLARE_GPU_INDEX(Index)                                               \
  DECLARE_GPU_SPECS_INDEX_NDIM(Index, 1);                                      \
  DECLARE_GPU_SPECS_INDEX_NDIM(Index, 2);                                      \
  DECLARE_GPU_SPECS_INDEX_NDIM(Index, 3);                                      \
  DECLARE_GPU_SPECS_INDEX_NDIM(Index, 4);

DECLARE_GPU_INDEX(int);

#undef DECLARE_GPU_INDEX
#undef DECLARE_GPU_SPECS_INDEX_NDIM
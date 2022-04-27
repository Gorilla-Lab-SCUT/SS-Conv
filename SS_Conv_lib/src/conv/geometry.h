/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

c++/cuda source code for sparse convolution. 
Written by Hongyang Li and Jiehong Lin
# Modified from https://github.com/traveller59/spconv/tree/v1.1
--------------------------------------------------------
*/

#ifndef SPCONV_GEOMETRY_H_
#define SPCONV_GEOMETRY_H_

#include <iostream>
#include <limits>
#include "tensorview.h"

template <typename Index, unsigned NDim>
TV_HOST_DEVICE Index getValidOutPos(const Index *input_pos,
                                    const Index *kernelSize,
                                    const Index *stride, const Index *padding,
                                    const Index *dilation,
                                    const Index *outSpatialShape, Index *out) {
  Index lowers[NDim];
  Index uppers[NDim];
  Index counter[NDim];
  Index counterSize[NDim];
  Index pointCounter = 0;
  Index val;
  Index numPoints = 1;
  Index m, offset;
  bool valid = false;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    lowers[i] = (input_pos[i] - (kernelSize[i] - 1) * dilation[i] - 1 +
                 stride[i] + padding[i]) /
                stride[i];
    uppers[i] = (input_pos[i] + padding[i]) / stride[i];
  }

#pragma unroll
  for (unsigned i = 0; i < NDim; ++i) {
    counterSize[i] = ((uppers[i] - lowers[i]) / dilation[i] + 1);
    numPoints *= counterSize[i];
  }

#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    counter[i] = 0;
  }
  for (int i = 0; i < numPoints; ++i) {
    valid = true;
    m = 1;
    offset = 0;
#pragma unroll
    for (int j = NDim - 1; j >= 0; --j) {
      val = uppers[j] - counter[j] * dilation[j];
      out[pointCounter * (NDim + 1) + j] = val;
      if (val < 0 || (val > outSpatialShape[j] - 1)) {
        valid = false;
        // break;
      }
      offset += m * (input_pos[j] - val * stride[j] + padding[j]) / dilation[j];
      m *= kernelSize[j];
    }

    out[pointCounter * (NDim + 1) + NDim] = offset;
    if (valid)
      ++pointCounter;
    counter[NDim - 1] += 1;
#pragma unroll
    for (int c = NDim - 1; c >= 0; --c) {
      if (counter[c] == counterSize[c] && c > 0) {
        counter[c - 1] += 1;
        counter[c] = 0;
      }
    }
  }
  return pointCounter;
}

template <typename Index, unsigned NDim>
TV_HOST_DEVICE Index getValidOutPosTranspose(
    const Index *input_pos, const Index *kernelSize, const Index *stride,
    const Index *padding, const Index *dilation, const Index *outSpatialShape,
    Index *out) {
  Index lowers[NDim];
  Index uppers[NDim];
  Index counter[NDim];
  Index counterSize[NDim];
  Index pointCounter = 0;
  Index val;
  Index numPoints = 1;
  Index m, offset;
  bool valid = false;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    lowers[i] = input_pos[i] * stride[i] - padding[i];
    uppers[i] = lowers[i] + (kernelSize[i] - 1) * dilation[i];
  }
#pragma unroll
  for (unsigned i = 0; i < NDim; ++i) {
    counterSize[i] = ((uppers[i] - lowers[i]) / dilation[i] + 1);
    numPoints *= counterSize[i];
  }
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    counter[i] = 0;
  }
  for (int i = 0; i < numPoints; ++i) {
    valid = true;
    m = 1;
    offset = 0;
#pragma unroll
    for (int j = NDim - 1; j >= 0; --j) {
      val = uppers[j] - counter[j] * dilation[j];
      out[pointCounter * (NDim + 1) + j] = val;
      if (val < 0 || (val > outSpatialShape[j] - 1)) {
        valid = false;
        // break;
      }
      offset += m * (val - lowers[j]) / dilation[j];
      m *= kernelSize[j];
    }
    out[pointCounter * (NDim + 1) + NDim] = offset;
    if (valid)
      ++pointCounter;
    counter[NDim - 1] += 1;
#pragma unroll
    for (int c = NDim - 1; c >= 0; --c) {
      if (counter[c] == counterSize[c] && c > 0) {
        counter[c - 1] += 1;
        counter[c] = 0;
      }
    }
  }
  return pointCounter;
}

template <typename Index, typename IndexGrid, unsigned NDim>
Index getIndicePairsConv(tv::TensorView<const Index> indicesIn,
                         tv::TensorView<Index> indicesOut,
                         tv::TensorView<IndexGrid> gridsOut,
                         tv::TensorView<Index> indicePairs,
                         tv::TensorView<Index> indiceNum,
                         const Index *kernelSize, const Index *stride,
                         const Index *padding, const Index *dilation,
                         const Index *outSpatialShape) {
  // indicesOut: num_active * kernelVolume * (NDim + 1)
  Index numAct = 0;
  auto numActIn = indicesIn.dim(0);
  Index batchIdx = 0;
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
  Index validPoints[kernelVolume * (NDim + 1)];
  Index *pointPtr = nullptr;
  for (int j = 0; j < numActIn; ++j) {
    batchIdx = indicesIn(j, 0);
    numValidPoints = getValidOutPos<Index, NDim>(
        indicesIn.data() + j * (NDim + 1) + 1, kernelSize, stride, padding,
        dilation, outSpatialShape, validPoints);
    for (Index i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
      auto index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape) +
                   spatialVolume * batchIdx;
      if (gridsOut[index] == -1) {
        for (unsigned k = 1; k < NDim + 1; ++k) {
          indicesOut(numAct, k) = pointPtr[k - 1];
        }
        indicesOut(numAct, 0) = batchIdx;
        gridsOut[index] = numAct++;
      }
      // indicePairs: [K, 2, L]
      indicePairs(offset, 0, indiceNum[offset]) = j;
      indicePairs(offset, 1, indiceNum[offset]++) = gridsOut[index];
    }
  }
  return numAct;
}

template <typename Index, typename IndexGrid, unsigned NDim>
Index getIndicePairsDeConv(tv::TensorView<const Index> indicesIn,
                           tv::TensorView<Index> indicesOut,
                           tv::TensorView<IndexGrid> gridsOut,
                           tv::TensorView<Index> indicePairs,
                           tv::TensorView<Index> indiceNum,
                           const Index *kernelSize, const Index *stride,
                           const Index *padding, const Index *dilation,
                           const Index *outSpatialShape) {
  Index numAct = 0;
  auto numActIn = indicesIn.dim(0);
  Index batchIdx = 0;
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
  Index validPoints[kernelVolume * (NDim + 1)];
  Index *pointPtr = nullptr;
  for (int j = 0; j < numActIn; ++j) {
    batchIdx = indicesIn(j, 0);
    numValidPoints = getValidOutPosTranspose<Index, NDim>(
        indicesIn.data() + j * (NDim + 1) + 1, kernelSize, stride, padding,
        dilation, outSpatialShape, validPoints);
    for (Index i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
      auto index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape) +
                   spatialVolume * batchIdx;
      if (gridsOut[index] == -1) {
        for (unsigned k = 1; k < NDim + 1; ++k) {
          indicesOut(numAct, k) = pointPtr[k - 1];
        }
        indicesOut(numAct, 0) = batchIdx;
        gridsOut[index] = numAct++;
      }
      // indicePairs: [K, 2, L]
      indicePairs(offset, 0, indiceNum[offset]) = j;
      indicePairs(offset, 1, indiceNum[offset]++) = gridsOut[index];
    }
  }
  return numAct;
}

template <typename Index, typename IndexGrid, unsigned NDim>
Index getIndicePairsSubM(tv::TensorView<const Index> indicesIn,
                         tv::TensorView<IndexGrid> gridsOut,
                         tv::TensorView<Index> indicePairs,
                         tv::TensorView<Index> indiceNum,
                         const Index *const kernelSize,
                         const Index *const stride, const Index *const padding,
                         const Index *dilation, const Index *const outSpatialShape) {
  Index numAct = 0;
  auto numActIn = indicesIn.dim(0);
  Index batchIdx = 0;
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
  Index validPoints[kernelVolume * (NDim + 1)];
  Index *pointPtr = nullptr;
  Index index = 0;
  for (int j = 0; j < numActIn; ++j) {
    index = tv::rowArrayIdx<Index, NDim>(indicesIn.data() + j * (NDim + 1) + 1,
                                         outSpatialShape) +
            spatialVolume * indicesIn(j, 0);
    gridsOut[index] = j;
  }
  for (int j = 0; j < numActIn; ++j) {
    numValidPoints = getValidOutPos<Index, NDim>(
        indicesIn.data() + j * (NDim + 1) + 1, kernelSize, stride, padding,
        dilation, outSpatialShape, validPoints);
    for (Index i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
      index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape) +
              spatialVolume * indicesIn(j, 0);
      if (gridsOut[index] > -1) {
        indicePairs(offset, 0, indiceNum[offset]) = j;
        indicePairs(offset, 1, indiceNum[offset]++) = gridsOut[index];
      }
    }
  }
  return numActIn;
}

#endif
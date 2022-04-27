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
#include <mp_helper.h>
#include <type_traits>
#include "timer.h"
#include "reordering_functor.h"
#include "helper_kernel.cu.h"
#include "helper_launch.h"
#include "tensorview.h"

// see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void gatherGenericKernel(T *buffer, const T *features,
                                    const Index *indices, int size,
                                    int numPlanes) {
  int ILPStrideX[NumILP];
  Index inds[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;

  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < size)
        inds[ilp] = indices[ix + ILPStrideX[ilp]] * numPlanes;
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < size)
          buffer[(ix + ILPStrideX[ilp]) * numPlanes + iy] =
              features[inds[ilp] + iy];
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP, typename VecType>
__global__ void gatherVecKernel(T *buffer, const T *features,
                                const Index *indices, int size, int numPlanes) {
  int ILPStrideX[NumILP];
  Index inds[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;

  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < size)
        inds[ilp] = indices[ix + ILPStrideX[ilp]] * numPlanes;
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < size)
          reinterpret_cast<VecType *>(
              buffer)[(ix + ILPStrideX[ilp]) * numPlanes + iy] =
              reinterpret_cast<const VecType *>(features)[inds[ilp] + iy];
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP,
          typename VecType = int4>
__global__ void gatherVecBlockKernel(T *buffer, const T *features,
                                     const Index *indices, int size,
                                     int numPlanes) {
  int ILPStrideY[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideY[ilp] = ilp * gridDim.y * blockDim.y;
  features += blockIdx.x * NumTLP;
  buffer += blockIdx.x * NumTLP;

  for (int iy : tv::KernelLoopY<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      reinterpret_cast<VecType *>(
          buffer)[(iy + ILPStrideY[ilp]) * numPlanes + threadIdx.x] =
          reinterpret_cast<const VecType *>(
              features)[indices[iy + ILPStrideY[ilp]] * numPlanes +
                        threadIdx.x];
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void scatterAddGenericKernel(T *outFeatures, const T *buffer,
                                        const Index *indices, int size,
                                        int numPlanes) {
  int ILPStrideX[NumILP];
  Index inds[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < size)
        inds[ilp] = indices[ix + ILPStrideX[ilp]] * numPlanes;
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < size) {
          outFeatures[inds[ilp] + iy] +=
              buffer[(ix + ILPStrideX[ilp]) * numPlanes + iy];
        }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP,
          typename VecType = int4>
__global__ void scatterAddVecBlockKernel(T *outFeatures, const T *buffer,
                                         const Index *indices, int size,
                                         int numPlanes) {
  int ILPStrideY[NumILP];
  constexpr int vecloadFactor = sizeof(VecType) / sizeof(T);
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideY[ilp] = ilp * gridDim.y * blockDim.y;
  outFeatures += blockIdx.x * NumTLP;
  buffer += blockIdx.x * NumTLP;
  T buf[vecloadFactor];
  T buf2[vecloadFactor];
  Index idx;
  for (int iy : tv::KernelLoopY<int, NumILP>(size)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      idx = indices[iy + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
      reinterpret_cast<VecType *>(buf)[0] =
          reinterpret_cast<VecType *>(outFeatures)[idx];
      reinterpret_cast<VecType *>(buf2)[0] = reinterpret_cast<const VecType *>(
          buffer)[(iy + ILPStrideY[ilp]) * numPlanes + threadIdx.x];
#pragma unroll
      for (int i = 0; i < vecloadFactor; i++) {
        buf[i] += buf2[i];
      }
      reinterpret_cast<VecType *>(outFeatures)[idx] =
          reinterpret_cast<VecType *>(buf)[0];
    }
  }
}

namespace functor {
template <typename T, typename Index>
struct SparseGatherFunctor<tv::GPU, T, Index> {
  using vecload_type_t =
      std::conditional_t<std::is_same<T, at::Half>::value, int2, int4>;
  using kernel_block_t = mp_list_c<int, 64, 32, 16>;
  void operator()(const tv::GPU &d, tv::TensorView<T> buffer,
                  tv::TensorView<const T> features,
                  tv::TensorView<const Index> indices, int size) {
    if (size <= 0)
      return;
    int numPlanes = features.dim(1);
    bool notFound = true;
    constexpr int vecloadFactor = sizeof(vecload_type_t) / sizeof(T);
    mp_for_each<kernel_block_t>([=, &buffer, &features, &indices,
                                 &notFound](auto NumTLP) {
      constexpr int NumILP = NumTLP / 4;
      // constexpr int NumILP = NumTLP / (64 / (NumTLP / vecloadFactor));
      int nHotBlock = (size / NumTLP) * NumTLP;
      if (notFound) {
        if (numPlanes % NumTLP == 0) {
          if (nHotBlock >= NumTLP) {
            gatherVecBlockKernel<T, Index, int(NumTLP), NumILP, vecload_type_t>
                <<<dim3(numPlanes / NumTLP, size / NumTLP),
                   dim3(NumTLP / vecloadFactor, NumTLP / NumILP), 0,
                   d.stream()>>>(buffer.data(), features.data(), indices.data(),
                                 nHotBlock, numPlanes / vecloadFactor);

            TV_CHECK_CUDA_ERR();
          }
          if (size - nHotBlock > 0) {
            gatherVecKernel<T, Index, int(NumTLP), NumILP, vecload_type_t>
                <<<dim3(1, numPlanes / NumTLP),
                   dim3(NumTLP / NumILP, NumTLP / vecloadFactor), 0,
                   d.stream()>>>(buffer.data() + nHotBlock * numPlanes,
                                 features.data(), indices.data() + nHotBlock,
                                 size - nHotBlock, numPlanes / vecloadFactor);
            TV_CHECK_CUDA_ERR();
          }
          notFound = false;
        }
      }
    });

    if (notFound) {
      constexpr int NumTLP = 64;
      constexpr int NumILP = NumTLP / 4;
      gatherGenericKernel<T, Index, NumTLP, NumILP>
          <<<dim3(tv::launch::DivUp(size, NumTLP),
                  tv::launch::DivUp(numPlanes, NumTLP)),
             dim3(NumTLP / NumILP, NumTLP), 0, d.stream()>>>(
              buffer.data(), features.data(), indices.data(), size, numPlanes);
      TV_CHECK_CUDA_ERR();
    }
  }
};
template <typename T, typename Index>
struct SparseScatterAddFunctor<tv::GPU, T, Index> {
  using vecload_type_t =
      std::conditional_t<std::is_same<T, at::Half>::value, int2, int4>;
  using kernel_block_t = mp_list_c<int, 64, 32, 16>;
  void operator()(const tv::GPU &d, tv::TensorView<T> outFeatures,
                  tv::TensorView<const T> buffer,
                  tv::TensorView<const Index> indices, int size, bool stable) {
    if (size <= 0)
      return;
    int numPlanes = outFeatures.dim(1);
    bool notFound = true;
    constexpr int vecloadFactor =
        sizeof(vecload_type_t) / sizeof(T); // important for half.
    mp_for_each<kernel_block_t>([=, &d, &outFeatures, &buffer, &indices,
                                 &notFound](auto NumTLP) {
      // constexpr int NumILP = NumTLP / (64 / (NumTLP / vecloadFactor));
      constexpr int NumILP = NumTLP / 4;
      int nHotBlock = (size / NumTLP) * NumTLP;
      if (notFound) {
        if (numPlanes % NumTLP == 0) {
          if (nHotBlock >= NumTLP) {
            scatterAddVecBlockKernel<T, Index, int(NumTLP), NumILP,
                                     vecload_type_t>
                <<<dim3(numPlanes / NumTLP, size / NumTLP),
                   dim3(NumTLP / vecloadFactor, NumTLP / NumILP), 0,
                   d.stream()>>>(outFeatures.data(), buffer.data(),
                                 indices.data(), nHotBlock,
                                 numPlanes / vecloadFactor);
            TV_CHECK_CUDA_ERR();
          }
          if (size - nHotBlock > 0) {
            scatterAddGenericKernel<T, Index, int(NumTLP), NumILP>
                <<<dim3(1, numPlanes / NumTLP), dim3(NumTLP / NumILP, NumTLP),
                   0, d.stream()>>>(
                    outFeatures.data(), buffer.data() + nHotBlock * numPlanes,
                    indices.data() + nHotBlock, size - nHotBlock, numPlanes);
            TV_CHECK_CUDA_ERR();
          }
          notFound = false;
        }
      }
    });
    if (notFound) {
      constexpr int NumTLP = 64;
      constexpr int NumILP = NumTLP / 4;
      scatterAddGenericKernel<T, Index, NumTLP, NumILP>
          <<<dim3(tv::launch::DivUp(size, NumTLP),
                  tv::launch::DivUp(numPlanes, NumTLP)),
             dim3(NumTLP / NumILP, NumTLP), 0, d.stream()>>>(
              outFeatures.data(), buffer.data(), indices.data(), size,
              numPlanes);
      TV_CHECK_CUDA_ERR();
    }
  }
};
} // namespace functor


#define DECLARE_GPU_SPECS_T_INDEX(T, Index)                                    \
  template struct functor::SparseGatherFunctor<tv::GPU, T, Index>;             \
  template struct functor::SparseScatterAddFunctor<tv::GPU, T, Index>;

#define DECLARE_GPU_SPECS(T) DECLARE_GPU_SPECS_T_INDEX(T, int);

DECLARE_GPU_SPECS(float);
DECLARE_GPU_SPECS(double);
DECLARE_GPU_SPECS(at::Half);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_T_INDEX
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
#include "indice_avgpool_functor.h"
#include "mp_helper.h"
#include "helper_kernel.cu.h"
#include "helper_launch.h"
#include "tensorview.h"
#include <type_traits>

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void avgPoolFwdBlockKernel(T *outFeatures, const T *inFeatures,
                                      const Index *indicesIn,
                                      const Index *indicesOut, int numHot,
                                      int numPlanes,
                                      const Index *summaryrf) {
  T in, out;
  int ILPStrideY[NumILP];
  Index rf;
  Index idxo, idxi;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideY[ilp] = threadIdx.y + ilp * blockDim.y;
  outFeatures += blockIdx.y * NumTLP;
  inFeatures += blockIdx.y * NumTLP;
  for (int ix = blockIdx.x * blockDim.x; ix < numHot;
       ix += blockDim.x * gridDim.x) {
    {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        idxi = indicesIn[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
        idxo = indicesOut[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
        rf   = summaryrf[indicesOut[ix + ILPStrideY[ilp]]];
        outFeatures[idxo] = outFeatures[idxo] + inFeatures[idxi] / (T)rf;
        // in = inFeatures[idxi];
        // out = outFeatures[idxo];
        // if (in > out) {
        //   outFeatures[idxo] = in;
        // }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void
avgPoolFwdGenericBlockKernel(T *outFeatures, const T *inFeatures,
                             const Index *indicesIn, const Index *indicesOut,
                             int numHot, int numPlanes,
                             const Index *summaryrf) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideX[NumILP];
  Index RI[NumILP];
  Index RO[NumILP];
  Index rf;
  T in, out;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  for (int ix : tv::KernelLoopX<int, NumILP>(numHot)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      RI[ilp] = indicesIn[ix + ILPStrideX[ilp]] * numPlanes;
      RO[ilp] = indicesOut[ix + ILPStrideX[ilp]] * numPlanes;
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        rf = summaryrf[indicesOut[ix + ILPStrideX[ilp]]];
        outFeatures[RO[ilp] + iy] = outFeatures[RO[ilp] + iy] + inFeatures[RI[ilp] + iy] / (T)rf;
        // in = inFeatures[RI[ilp] + iy];
        // out = outFeatures[RO[ilp] + iy];
        // if (in > out) {
        //   outFeatures[RO[ilp] + iy] = in;
        // }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP, typename VecType>
__global__ void avgPoolFwdVecBlockKernel(T *outFeatures, const T *inFeatures,
                                         const Index *indicesIn,
                                         const Index *indicesOut, int numHot,
                                         int numPlanes,
                                         const Index *summaryrf) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideY[NumILP];
  constexpr int vecloadFactor = sizeof(VecType) / sizeof(T);
  T bufi[vecloadFactor];
  T bufo[vecloadFactor];
  Index idxi, idxo;
  Index rf;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideY[ilp] = threadIdx.y + ilp * blockDim.y;
  outFeatures += blockIdx.y * NumTLP;
  inFeatures += blockIdx.y * NumTLP;
  for (int ix = blockIdx.x * blockDim.x * vecloadFactor; ix < numHot;
       ix += blockDim.x * gridDim.x * vecloadFactor) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      idxi = indicesIn[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
      idxo = indicesOut[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
      rf   = summaryrf[indicesOut[ix + ILPStrideY[ilp]]];
      reinterpret_cast<VecType *>(bufo)[0] =
          reinterpret_cast<VecType *>(outFeatures)[idxo];
      reinterpret_cast<VecType *>(bufi)[0] =
          reinterpret_cast<const VecType *>(inFeatures)[idxi];
#pragma unroll
      for (int i = 0; i < vecloadFactor; i++) {
        // if (bufi[i] > bufo[i]) {
        //  bufo[i] = bufi[i];
        //}
        bufo[i] = bufo[i] + bufi[i] / (T)rf;
      }
      reinterpret_cast<VecType *>(outFeatures)[idxo] =
          reinterpret_cast<VecType *>(bufo)[0];
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void avgPoolFwdGenericKernel(T *outFeatures, const T *inFeatures,
                                        const Index *indicesIn,
                                        const Index *indicesOut, int numHot,
                                        int numPlanes,
                                        const Index *summaryrf) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideX[NumILP];
  Index RI[NumILP];
  Index RO[NumILP];
  Index rf;
  T in, out;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  for (int ix : tv::KernelLoopX<int, NumILP>(numHot)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < numHot) {
        RI[ilp] = indicesIn[ix + ILPStrideX[ilp]] * numPlanes;
        RO[ilp] = indicesOut[ix + ILPStrideX[ilp]] * numPlanes;
      }
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < numHot) {
            rf = summaryrf[indicesOut[ix + ILPStrideX[ilp]]];
            outFeatures[RO[ilp] + iy] = outFeatures[RO[ilp] + iy] + inFeatures[RI[ilp] + iy] / (T)rf;
        //   in = inFeatures[RI[ilp] + iy];
        //   out = outFeatures[RO[ilp] + iy];
        //   if (in > out) {
        //     outFeatures[RO[ilp] + iy] = in;
        //   }
        }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void
avgPoolBwdBlockKernel(const T *outFeatures, const T *inFeatures, const T *dout,
                      T *din, const Index *indicesIn, const Index *indicesOut,
                      int numHot, int numPlanes,
                      const Index *summaryrf) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  T in, out;
  Index idxo, idxi;
  Index rf;
  int ILPStrideY[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideY[ilp] = threadIdx.y + ilp * blockDim.y;
  outFeatures += blockIdx.y * NumTLP;
  inFeatures += blockIdx.y * NumTLP;
  dout += blockIdx.y * NumTLP;
  din += blockIdx.y * NumTLP;
  for (int ix = blockIdx.x * blockDim.x; ix < numHot;
       ix += blockDim.x * gridDim.x) {
    {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        idxi = indicesIn[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
        idxo = indicesOut[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
        rf   = summaryrf[indicesOut[ix + ILPStrideY[ilp]]];
        // in = inFeatures[idxi];
        // out = outFeatures[idxo];
        // if (in == out) {
        //   din[idxi] += dout[idxo];
        // }
        din[idxi] += (dout[idxo] / (T)rf);
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void avgPoolBwdGenericBlockKernel(const T *outFeatures,
                                             const T *inFeatures, const T *dout,
                                             T *din, const Index *indicesIn,
                                             const Index *indicesOut,
                                             int numHot, int numPlanes,
                                             const Index *summaryrf) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideX[NumILP];
  Index RI[NumILP];
  Index RO[NumILP];
  Index rf;
  T in, out;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  for (int ix : tv::KernelLoopX<int, NumILP>(numHot)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      RI[ilp] = indicesIn[ix + ILPStrideX[ilp]] * numPlanes;
      RO[ilp] = indicesOut[ix + ILPStrideX[ilp]] * numPlanes;
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        // in = inFeatures[RI[ilp] + iy];
        // out = outFeatures[RO[ilp] + iy];
        // if (in == out) {
        //   din[RI[ilp] + iy] += dout[RO[ilp] + iy];
        // }
        rf = summaryrf[indicesOut[ix + ILPStrideX[ilp]]];
        din[RI[ilp] + iy] += (dout[RO[ilp] + iy] / (T)rf);

      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP, typename VecType>
__global__ void
avgPoolBwdVecBlockKernel(const T *outFeatures, const T *inFeatures,
                         const T *dout, T *din, const Index *indicesIn,
                         const Index *indicesOut, int numHot, int numPlanes,
                         const Index *summaryrf) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideY[NumILP];
  constexpr int vecloadFactor = sizeof(VecType) / sizeof(T);
  T bufi[vecloadFactor];
  T bufo[vecloadFactor];
  T bufdi[vecloadFactor];
  T bufdo[vecloadFactor];
  Index rf;
  Index idxi, idxo;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideY[ilp] = threadIdx.y + ilp * blockDim.y;
  outFeatures += blockIdx.y * NumTLP;
  inFeatures += blockIdx.y * NumTLP;
  for (int ix = blockIdx.x * blockDim.x * vecloadFactor; ix < numHot;
       ix += blockDim.x * gridDim.x * vecloadFactor) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      idxi = indicesIn[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
      idxo = indicesOut[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
      rf   = summaryrf[indicesOut[ix + ILPStrideY[ilp]]];
      reinterpret_cast<VecType *>(bufo)[0] =
          reinterpret_cast<const VecType *>(outFeatures)[idxo];
      reinterpret_cast<VecType *>(bufi)[0] =
          reinterpret_cast<const VecType *>(inFeatures)[idxi];
      reinterpret_cast<VecType *>(bufdo)[0] =
          reinterpret_cast<const VecType *>(dout)[idxo];
      reinterpret_cast<VecType *>(bufdi)[0] = reinterpret_cast<VecType *>(din)[idxi];

#pragma unroll
      for (int i = 0; i < vecloadFactor; i++) {
        // if (bufi[i] == bufo[i]) {
        //   bufdi[i] += bufdo[i];
        // }
        bufdi[i] += (bufdo[i]/(T)rf);
      }
      reinterpret_cast<VecType *>(din)[idxi] = reinterpret_cast<VecType *>(bufdi)[0];
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void
avgPoolBwdGenericKernel(const T *outFeatures, const T *inFeatures,
                        const T *dout, T *din, const Index *indicesIn,
                        const Index *indicesOut, int numHot, int numPlanes,
                        const Index *summaryrf) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideX[NumILP];
  Index RI[NumILP];
  Index RO[NumILP];
  Index rf;
  T in, out;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  for (int ix : tv::KernelLoopX<int, NumILP>(numHot)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < numHot) {
        RI[ilp] = indicesIn[ix + ILPStrideX[ilp]] * numPlanes;
        RO[ilp] = indicesOut[ix + ILPStrideX[ilp]] * numPlanes;
      }
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < numHot) {
        //   in = inFeatures[RI[ilp] + iy];
        //   out = outFeatures[RO[ilp] + iy];
        //   if (in == out) {
        //     din[RI[ilp] + iy] += dout[RO[ilp] + iy];
        //   }
            rf = summaryrf[indicesOut[ix + ILPStrideX[ilp]]];
            din[RI[ilp] + iy] += (dout[RO[ilp] + iy] / (T)rf);
        }
      }
    }
  }
}

namespace functor {
template <typename T, typename Index>
struct SparseAvgPoolForwardFunctor<tv::GPU, T, Index> {
  using vecload_type_t =
      std::conditional_t<std::is_same<T, at::Half>::value, int2, int4>;
  using kernel_block_t = mp_list_c<int, 64, 32, 16>;
  void operator()(const tv::GPU &d, tv::TensorView<T> outFeatures,
                  tv::TensorView<const T> inFeatures,
                  tv::TensorView<const Index> indices, int size,
                  tv::TensorView<const Index> summaryrf) {
    if (size <= 0)
      return;
    int numPlanes = inFeatures.dim(1);
    bool notFound = true;
    constexpr int vecloadFactor = sizeof(vecload_type_t) / sizeof(T);
    mp_for_each<kernel_block_t>([=, &outFeatures, &inFeatures, &indices,
                                 &notFound, &summaryrf](auto NumTLP) {
      constexpr int NumILP = NumTLP / 4;

      int numHotBlock = (size / NumTLP) * NumTLP;
      if (notFound) {
        if (numPlanes % NumTLP == 0) {
          if (numHotBlock >= NumTLP) {
            avgPoolFwdVecBlockKernel<T, Index, int(NumTLP), NumILP, vecload_type_t>
                <<<dim3(std::min(size / NumTLP, 512), numPlanes / NumTLP),
                   dim3(NumTLP / vecloadFactor, NumTLP / NumILP), 0,
                   d.stream()>>>(outFeatures.data(), inFeatures.data(),
                                 indices.subview(0).data(),
                                 indices.subview(1).data(), numHotBlock,
                                 numPlanes / vecloadFactor,
                                 summaryrf.data()
                                );
            TV_CHECK_CUDA_ERR();
          }

          if (size > numHotBlock) {
            avgPoolFwdGenericKernel<T, Index, int(NumTLP), NumILP>
                <<<dim3(1, numPlanes / NumTLP), dim3(NumTLP / NumILP, NumTLP),
                   0, d.stream()>>>(outFeatures.data(), inFeatures.data(),
                                    indices.subview(0).data() + numHotBlock,
                                    indices.subview(1).data() + numHotBlock,
                                    size - numHotBlock, numPlanes,
                                    summaryrf.data());
            TV_CHECK_CUDA_ERR();
          }
          notFound = false;
        }
      }
    });

    if (notFound) {
      constexpr int NumTLP = 64;
      constexpr int NumILP = NumTLP / 4;
      int numHotBlock = (size / NumTLP) * NumTLP;
      if (numHotBlock >= NumTLP) {
        avgPoolFwdGenericBlockKernel<T, Index, NumTLP, NumILP>
            <<<dim3(size / NumTLP, tv::launch::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, d.stream()>>>(
                outFeatures.data(), inFeatures.data(),
                indices.subview(0).data(), indices.subview(1).data(),
                numHotBlock, numPlanes,
                summaryrf.data());
        TV_CHECK_CUDA_ERR();
      }

      if (size > numHotBlock) {
        avgPoolFwdGenericKernel<T, Index, NumTLP, NumILP>
            <<<dim3(1, tv::launch::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, d.stream()>>>(
                outFeatures.data(), inFeatures.data(),
                indices.subview(0).data() + numHotBlock,
                indices.subview(1).data() + numHotBlock, size - numHotBlock,
                numPlanes,
                summaryrf.data());
        TV_CHECK_CUDA_ERR();
      }
    }
  }
};

template <typename T, typename Index>
struct SparseAvgPoolBackwardFunctor<tv::GPU, T, Index> {
  using vecload_type_t =
      std::conditional_t<std::is_same<T, at::Half>::value, int2, int4>;
  using kernel_block_t = mp_list_c<int, 64, 32, 16>;
  void operator()(const tv::GPU &d, tv::TensorView<const T> outFeatures,
                  tv::TensorView<const T> inFeatures,
                  tv::TensorView<const T> dout, tv::TensorView<T> din,
                  tv::TensorView<const Index> indices, int size, tv::TensorView<const Index> summaryrf) {
    if (size <= 0)
      return;
    int numPlanes = inFeatures.dim(1);
    bool notFound = true;
    constexpr int vecloadFactor = sizeof(vecload_type_t) / sizeof(T);
    mp_for_each<kernel_block_t>([=, &outFeatures, &inFeatures, &dout, &din,
                                 &indices, &notFound, &summaryrf](auto NumTLP) {
      constexpr int NumILP = NumTLP / 4;

      int numHotBlock = (size / NumTLP) * NumTLP;
      if (notFound) {
        if (numPlanes % NumTLP == 0) {
          if (numHotBlock >= NumTLP) {
            avgPoolBwdVecBlockKernel<T, Index, int(NumTLP), NumILP, vecload_type_t>
                <<<dim3(std::min(size / NumTLP, 512), numPlanes / NumTLP),
                   dim3(NumTLP / vecloadFactor, NumTLP / NumILP), 0,
                   d.stream()>>>(outFeatures.data(), inFeatures.data(),
                                 dout.data(), din.data(),
                                 indices.subview(0).data(),
                                 indices.subview(1).data(), numHotBlock,
                                 numPlanes / vecloadFactor, summaryrf.data());
            TV_CHECK_CUDA_ERR();
          }

          if (size > numHotBlock) {
            avgPoolBwdGenericKernel<T, Index, int(NumTLP), NumILP>
                <<<dim3(1, numPlanes / NumTLP), dim3(NumTLP / NumILP, NumTLP),
                   0, d.stream()>>>(outFeatures.data(), inFeatures.data(),
                                    dout.data(), din.data(),
                                    indices.subview(0).data() + numHotBlock,
                                    indices.subview(1).data() + numHotBlock,
                                    size - numHotBlock, numPlanes, summaryrf.data());
            TV_CHECK_CUDA_ERR();
          }
          notFound = false;
        }
      }
    });

    if (notFound) {
      constexpr int NumTLP = 64;
      constexpr int NumILP = NumTLP / 4;
      int numHotBlock = (size / NumTLP) * NumTLP;
      if (numHotBlock >= NumTLP) {
        avgPoolBwdGenericBlockKernel<T, Index, NumTLP, NumILP>
            <<<dim3(size / NumTLP, tv::launch::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, d.stream()>>>(
                outFeatures.data(), inFeatures.data(), dout.data(), din.data(),
                indices.subview(0).data(), indices.subview(1).data(),
                numHotBlock, numPlanes, summaryrf.data());
        TV_CHECK_CUDA_ERR();
      }

      if (size > numHotBlock) {
        avgPoolBwdGenericKernel<T, Index, NumTLP, NumILP>
            <<<dim3(1, tv::launch::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, d.stream()>>>(
                outFeatures.data(), inFeatures.data(), dout.data(), din.data(),
                indices.subview(0).data() + numHotBlock,
                indices.subview(1).data() + numHotBlock, size - numHotBlock,
                numPlanes, summaryrf.data());
        TV_CHECK_CUDA_ERR();
      }
    }
  }
};

} // namespace functor

#define DECLARE_GPU_SPECS_T_INDEX(T, Index)                                    \
  template struct functor::SparseAvgPoolForwardFunctor<tv::GPU, T, Index>;     \
  template struct functor::SparseAvgPoolBackwardFunctor<tv::GPU, T, Index>;

#define DECLARE_GPU_SPECS(T) DECLARE_GPU_SPECS_T_INDEX(T, int);

DECLARE_GPU_SPECS(float);
DECLARE_GPU_SPECS(double);
DECLARE_GPU_SPECS(at::Half);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_T_INDEX
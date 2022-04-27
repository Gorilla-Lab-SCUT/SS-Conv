/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

c++/cuda source code for sparse max pool. 
Written by Hongyang Li and Jiehong Lin
# Modified from https://github.com/traveller59/spconv/tree/v1.1
--------------------------------------------------------
*/


#include <ATen/ATen.h>
#include <chrono>
#include <limits>
#include <indice_maxpool_functor.h>
#include <mp_helper.h>
#include <helper_kernel.cu.h>
#include <helper_launch.h>
#include <tensorview.h>
#include <type_traits>

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void maxPoolFwdBlockKernel(T *outFeatures, const T *inFeatures,
                                      const Index *indicesIn,
                                      const Index *indicesOut, int numHot,
                                      int numPlanes) {
  T in, out;
  int ILPStrideY[NumILP];
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
        in = inFeatures[idxi];
        out = outFeatures[idxo];
        if (in > out) {
          outFeatures[idxo] = in;
        }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void
maxPoolFwdGenericBlockKernel(T *outFeatures, const T *inFeatures,
                             const Index *indicesIn, const Index *indicesOut,
                             int numHot, int numPlanes) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideX[NumILP];
  Index RI[NumILP];
  Index RO[NumILP];
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
        in = inFeatures[RI[ilp] + iy];
        out = outFeatures[RO[ilp] + iy];
        if (in > out) {
          outFeatures[RO[ilp] + iy] = in;
        }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP, typename VecType>
__global__ void maxPoolFwdVecBlockKernel(T *outFeatures, const T *inFeatures,
                                         const Index *indicesIn,
                                         const Index *indicesOut, int numHot,
                                         int numPlanes) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideY[NumILP];
  constexpr int vecloadFactor = sizeof(VecType) / sizeof(T);
  T bufi[vecloadFactor];
  T bufo[vecloadFactor];
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
      reinterpret_cast<VecType *>(bufo)[0] =
          reinterpret_cast<VecType *>(outFeatures)[idxo];
      reinterpret_cast<VecType *>(bufi)[0] =
          reinterpret_cast<const VecType *>(inFeatures)[idxi];
#pragma unroll
      for (int i = 0; i < vecloadFactor; i++) {
        if (bufi[i] > bufo[i]) {
          bufo[i] = bufi[i];
        }
      }
      reinterpret_cast<VecType *>(outFeatures)[idxo] =
          reinterpret_cast<VecType *>(bufo)[0];
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void maxPoolFwdGenericKernel(T *outFeatures, const T *inFeatures,
                                        const Index *indicesIn,
                                        const Index *indicesOut, int numHot,
                                        int numPlanes) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideX[NumILP];
  Index RI[NumILP];
  Index RO[NumILP];
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
          in = inFeatures[RI[ilp] + iy];
          out = outFeatures[RO[ilp] + iy];
          if (in > out) {
            outFeatures[RO[ilp] + iy] = in;
          }
        }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void fieldmaxPoolFwdBlockKernel(T *outFeatures, const T *inFeatures,
                                      const Index *indicesIn,
                                      const Index *indicesOut, int numHot,
                                      int numPlanes,
                                      T *outFeature_norms, const T *inFeature_norms) {
  T in, out, in_norm, out_norm;
  int ILPStrideY[NumILP];
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
        in = inFeatures[idxi];
        out = outFeatures[idxo];
        in_norm = inFeature_norms[idxi];
        out_norm = outFeature_norms[idxo];
        if (in_norm > out_norm) {
          outFeatures[idxo] = in;
          outFeature_norms[idxo] = in_norm;
        }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void
fieldmaxPoolFwdGenericBlockKernel(T *outFeatures, const T *inFeatures,
                             const Index *indicesIn, const Index *indicesOut,
                             int numHot, int numPlanes,
                             T *outFeature_norms, const T *inFeature_norms) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideX[NumILP];
  Index RI[NumILP];
  Index RO[NumILP];
  T in, out, in_norm, out_norm;
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
        in = inFeatures[RI[ilp] + iy];
        out = outFeatures[RO[ilp] + iy];
        in_norm = inFeature_norms[RI[ilp] + iy];
        out_norm = outFeature_norms[RO[ilp] + iy];
        if (in_norm > out_norm) {
          outFeatures[RO[ilp] + iy] = in;
          outFeature_norms[RO[ilp] + iy] = in_norm;
        }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP, typename VecType>
__global__ void fieldmaxPoolFwdVecBlockKernel(T *outFeatures, const T *inFeatures,
                                         const Index *indicesIn,
                                         const Index *indicesOut, int numHot,
                                         int numPlanes,
                                         T *outFeature_norms, const T *inFeature_norms) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideY[NumILP];
  constexpr int vecloadFactor = sizeof(VecType) / sizeof(T);
  T bufi[vecloadFactor];
  T bufo[vecloadFactor];
  T bufi_norm[vecloadFactor];
  T bufo_norm[vecloadFactor];
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
      reinterpret_cast<VecType *>(bufo)[0] =
          reinterpret_cast<VecType *>(outFeatures)[idxo];
      reinterpret_cast<VecType *>(bufi)[0] =
          reinterpret_cast<const VecType *>(inFeatures)[idxi];

      reinterpret_cast<VecType *>(bufo_norm)[0] =
          reinterpret_cast<VecType *>(outFeature_norms)[idxo];
      reinterpret_cast<VecType *>(bufi_norm)[0] =
          reinterpret_cast<const VecType *>(inFeature_norms)[idxi];
#pragma unroll
      for (int i = 0; i < vecloadFactor; i++) {
        if (bufi_norm[i] > bufo_norm[i]) {
          bufo[i] = bufi[i];
          bufo_norm[i] = bufi_norm[i];
        }
      }
      reinterpret_cast<VecType *>(outFeatures)[idxo] =
          reinterpret_cast<VecType *>(bufo)[0];
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void fieldmaxPoolFwdGenericKernel(T *outFeatures, const T *inFeatures,
                                        const Index *indicesIn,
                                        const Index *indicesOut, int numHot,
                                        int numPlanes,
                                        T *outFeature_norms, const T *inFeature_norms) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideX[NumILP];
  Index RI[NumILP];
  Index RO[NumILP];
  T in, out, in_norm, out_norm;
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
          in = inFeatures[RI[ilp] + iy];
          out = outFeatures[RO[ilp] + iy];
          in_norm = inFeature_norms[RI[ilp] + iy];
          out_norm = outFeature_norms[RO[ilp] + iy];
          if (in_norm > out_norm) {
            outFeatures[RO[ilp] + iy] = in;
            outFeature_norms[RO[ilp] + iy] = in_norm;
          }
        }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void
maxPoolBwdBlockKernel(const T *outFeatures, const T *inFeatures, const T *dout,
                      T *din, const Index *indicesIn, const Index *indicesOut,
                      int numHot, int numPlanes) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  T in, out;
  Index idxo, idxi;
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
        in = inFeatures[idxi];
        out = outFeatures[idxo];
        if (in == out) {
          din[idxi] += dout[idxo];
        }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void maxPoolBwdGenericBlockKernel(const T *outFeatures,
                                             const T *inFeatures, const T *dout,
                                             T *din, const Index *indicesIn,
                                             const Index *indicesOut,
                                             int numHot, int numPlanes) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideX[NumILP];
  Index RI[NumILP];
  Index RO[NumILP];
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
        in = inFeatures[RI[ilp] + iy];
        out = outFeatures[RO[ilp] + iy];
        if (in == out) {
          din[RI[ilp] + iy] += dout[RO[ilp] + iy];
        }
      }
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP, typename VecType>
__global__ void
maxPoolBwdVecBlockKernel(const T *outFeatures, const T *inFeatures,
                         const T *dout, T *din, const Index *indicesIn,
                         const Index *indicesOut, int numHot, int numPlanes) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideY[NumILP];
  constexpr int vecloadFactor = sizeof(VecType) / sizeof(T);
  T bufi[vecloadFactor];
  T bufo[vecloadFactor];
  T bufdi[vecloadFactor];
  T bufdo[vecloadFactor];
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
      reinterpret_cast<VecType *>(bufo)[0] =
          reinterpret_cast<const VecType *>(outFeatures)[idxo];
      reinterpret_cast<VecType *>(bufi)[0] =
          reinterpret_cast<const VecType *>(inFeatures)[idxi];
      reinterpret_cast<VecType *>(bufdo)[0] =
          reinterpret_cast<const VecType *>(dout)[idxo];
      reinterpret_cast<VecType *>(bufdi)[0] = reinterpret_cast<VecType *>(din)[idxi];

#pragma unroll
      for (int i = 0; i < vecloadFactor; i++) {
        if (bufi[i] == bufo[i]) {
          bufdi[i] += bufdo[i];
        }
      }
      reinterpret_cast<VecType *>(din)[idxi] = reinterpret_cast<VecType *>(bufdi)[0];
    }
  }
}

template <typename T, typename Index, int NumTLP, int NumILP>
__global__ void
maxPoolBwdGenericKernel(const T *outFeatures, const T *inFeatures,
                        const T *dout, T *din, const Index *indicesIn,
                        const Index *indicesOut, int numHot, int numPlanes) {
  // see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
  int ILPStrideX[NumILP];
  Index RI[NumILP];
  Index RO[NumILP];
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
          in = inFeatures[RI[ilp] + iy];
          out = outFeatures[RO[ilp] + iy];
          if (in == out) {
            din[RI[ilp] + iy] += dout[RO[ilp] + iy];
          }
        }
      }
    }
  }
}

namespace functor {
template <typename T, typename Index>
struct SparseMaxPoolForwardFunctor<tv::GPU, T, Index> {
  using vecload_type_t =
      std::conditional_t<std::is_same<T, at::Half>::value, int2, int4>;
  using kernel_block_t = mp_list_c<int, 64, 32, 16>;
  void operator()(const tv::GPU &d, tv::TensorView<T> outFeatures,
                  tv::TensorView<const T> inFeatures,
                  tv::TensorView<const Index> indices, int size) {
    if (size <= 0)
      return;
    int numPlanes = inFeatures.dim(1);
    bool notFound = true;
    constexpr int vecloadFactor = sizeof(vecload_type_t) / sizeof(T);
    mp_for_each<kernel_block_t>([=, &outFeatures, &inFeatures, &indices,
                                 &notFound](auto NumTLP) {
      constexpr int NumILP = NumTLP / 4;

      int numHotBlock = (size / NumTLP) * NumTLP;
      if (notFound) {
        if (numPlanes % NumTLP == 0) {
          if (numHotBlock >= NumTLP) {
            maxPoolFwdVecBlockKernel<T, Index, int(NumTLP), NumILP, vecload_type_t>
                <<<dim3(std::min(size / NumTLP, 512), numPlanes / NumTLP),
                   dim3(NumTLP / vecloadFactor, NumTLP / NumILP), 0,
                   d.stream()>>>(outFeatures.data(), inFeatures.data(),
                                 indices.subview(0).data(),
                                 indices.subview(1).data(), numHotBlock,
                                 numPlanes / vecloadFactor);
            TV_CHECK_CUDA_ERR();
          }

          if (size > numHotBlock) {
            maxPoolFwdGenericKernel<T, Index, int(NumTLP), NumILP>
                <<<dim3(1, numPlanes / NumTLP), dim3(NumTLP / NumILP, NumTLP),
                   0, d.stream()>>>(outFeatures.data(), inFeatures.data(),
                                    indices.subview(0).data() + numHotBlock,
                                    indices.subview(1).data() + numHotBlock,
                                    size - numHotBlock, numPlanes);
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
        maxPoolFwdGenericBlockKernel<T, Index, NumTLP, NumILP>
            <<<dim3(size / NumTLP, tv::launch::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, d.stream()>>>(
                outFeatures.data(), inFeatures.data(),
                indices.subview(0).data(), indices.subview(1).data(),
                numHotBlock, numPlanes);
        TV_CHECK_CUDA_ERR();
      }

      if (size > numHotBlock) {
        maxPoolFwdGenericKernel<T, Index, NumTLP, NumILP>
            <<<dim3(1, tv::launch::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, d.stream()>>>(
                outFeatures.data(), inFeatures.data(),
                indices.subview(0).data() + numHotBlock,
                indices.subview(1).data() + numHotBlock, size - numHotBlock,
                numPlanes);
        TV_CHECK_CUDA_ERR();
      }
    }
  }
};

template <typename T, typename Index>
struct SparseFieldMaxPoolForwardFunctor<tv::GPU, T, Index> {
  using vecload_type_t =
      std::conditional_t<std::is_same<T, at::Half>::value, int2, int4>;
  using kernel_block_t = mp_list_c<int, 64, 32, 16>;
  void operator()(const tv::GPU &d, tv::TensorView<T> outFeatures,
                  tv::TensorView<const T> inFeatures,
                  tv::TensorView<const Index> indices, int size,
                  tv::TensorView<const T> inFeature_norms, tv::TensorView<T> outFeature_norms) {
    if (size <= 0)
      return;
    int numPlanes = inFeatures.dim(1);
    bool notFound = true;
    constexpr int vecloadFactor = sizeof(vecload_type_t) / sizeof(T);
    mp_for_each<kernel_block_t>([=, &outFeatures, &inFeatures, &indices,
                                 &notFound, &inFeature_norms, &outFeature_norms](auto NumTLP) {
      constexpr int NumILP = NumTLP / 4;

      int numHotBlock = (size / NumTLP) * NumTLP;
      if (notFound) {
        if (numPlanes % NumTLP == 0) {
          if (numHotBlock >= NumTLP) {
            fieldmaxPoolFwdVecBlockKernel<T, Index, int(NumTLP), NumILP, vecload_type_t>
                <<<dim3(std::min(size / NumTLP, 512), numPlanes / NumTLP),
                   dim3(NumTLP / vecloadFactor, NumTLP / NumILP), 0,
                   d.stream()>>>(outFeatures.data(), inFeatures.data(),
                                 indices.subview(0).data(),
                                 indices.subview(1).data(), numHotBlock,
                                 numPlanes / vecloadFactor,
                                 outFeature_norms.data(), inFeature_norms.data());
            TV_CHECK_CUDA_ERR();
          }

          if (size > numHotBlock) {
            fieldmaxPoolFwdGenericKernel<T, Index, int(NumTLP), NumILP>
                <<<dim3(1, numPlanes / NumTLP), dim3(NumTLP / NumILP, NumTLP),
                   0, d.stream()>>>(outFeatures.data(), inFeatures.data(),
                                    indices.subview(0).data() + numHotBlock,
                                    indices.subview(1).data() + numHotBlock,
                                    size - numHotBlock, numPlanes,
                                    outFeature_norms.data(), inFeature_norms.data());
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
        fieldmaxPoolFwdGenericBlockKernel<T, Index, NumTLP, NumILP>
            <<<dim3(size / NumTLP, tv::launch::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, d.stream()>>>(
                outFeatures.data(), inFeatures.data(),
                indices.subview(0).data(), indices.subview(1).data(),
                numHotBlock, numPlanes,
                outFeature_norms.data(), inFeature_norms.data());
        TV_CHECK_CUDA_ERR();
      }

      if (size > numHotBlock) {
        fieldmaxPoolFwdGenericKernel<T, Index, NumTLP, NumILP>
            <<<dim3(1, tv::launch::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, d.stream()>>>(
                outFeatures.data(), inFeatures.data(),
                indices.subview(0).data() + numHotBlock,
                indices.subview(1).data() + numHotBlock, size - numHotBlock,
                numPlanes,
                outFeature_norms.data(), inFeature_norms.data());
        TV_CHECK_CUDA_ERR();
      }
    }
  }
};

template <typename T, typename Index>
struct SparseMaxPoolBackwardFunctor<tv::GPU, T, Index> {
  using vecload_type_t =
      std::conditional_t<std::is_same<T, at::Half>::value, int2, int4>;
  using kernel_block_t = mp_list_c<int, 64, 32, 16>;
  void operator()(const tv::GPU &d, tv::TensorView<const T> outFeatures,
                  tv::TensorView<const T> inFeatures,
                  tv::TensorView<const T> dout, tv::TensorView<T> din,
                  tv::TensorView<const Index> indices, int size) {
    if (size <= 0)
      return;
    int numPlanes = inFeatures.dim(1);
    bool notFound = true;
    constexpr int vecloadFactor = sizeof(vecload_type_t) / sizeof(T);
    mp_for_each<kernel_block_t>([=, &outFeatures, &inFeatures, &dout, &din,
                                 &indices, &notFound](auto NumTLP) {
      constexpr int NumILP = NumTLP / 4;

      int numHotBlock = (size / NumTLP) * NumTLP;
      if (notFound) {
        if (numPlanes % NumTLP == 0) {
          if (numHotBlock >= NumTLP) {
            maxPoolBwdVecBlockKernel<T, Index, int(NumTLP), NumILP, vecload_type_t>
                <<<dim3(std::min(size / NumTLP, 512), numPlanes / NumTLP),
                   dim3(NumTLP / vecloadFactor, NumTLP / NumILP), 0,
                   d.stream()>>>(outFeatures.data(), inFeatures.data(),
                                 dout.data(), din.data(),
                                 indices.subview(0).data(),
                                 indices.subview(1).data(), numHotBlock,
                                 numPlanes / vecloadFactor);
            TV_CHECK_CUDA_ERR();
          }

          if (size > numHotBlock) {
            maxPoolBwdGenericKernel<T, Index, int(NumTLP), NumILP>
                <<<dim3(1, numPlanes / NumTLP), dim3(NumTLP / NumILP, NumTLP),
                   0, d.stream()>>>(outFeatures.data(), inFeatures.data(),
                                    dout.data(), din.data(),
                                    indices.subview(0).data() + numHotBlock,
                                    indices.subview(1).data() + numHotBlock,
                                    size - numHotBlock, numPlanes);
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
        maxPoolBwdGenericBlockKernel<T, Index, NumTLP, NumILP>
            <<<dim3(size / NumTLP, tv::launch::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, d.stream()>>>(
                outFeatures.data(), inFeatures.data(), dout.data(), din.data(),
                indices.subview(0).data(), indices.subview(1).data(),
                numHotBlock, numPlanes);
        TV_CHECK_CUDA_ERR();
      }

      if (size > numHotBlock) {
        maxPoolBwdGenericKernel<T, Index, NumTLP, NumILP>
            <<<dim3(1, tv::launch::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, d.stream()>>>(
                outFeatures.data(), inFeatures.data(), dout.data(), din.data(),
                indices.subview(0).data() + numHotBlock,
                indices.subview(1).data() + numHotBlock, size - numHotBlock,
                numPlanes);
        TV_CHECK_CUDA_ERR();
      }
    }
  }
};

} // namespace functor

#define DECLARE_GPU_SPECS_T_INDEX(T, Index)                                    \
  template struct functor::SparseMaxPoolForwardFunctor<tv::GPU, T, Index>;     \
  template struct functor::SparseMaxPoolBackwardFunctor<tv::GPU, T, Index>;    \
  template struct functor::SparseFieldMaxPoolForwardFunctor<tv::GPU, T, Index>;

#define DECLARE_GPU_SPECS(T) DECLARE_GPU_SPECS_T_INDEX(T, int);

DECLARE_GPU_SPECS(float);
DECLARE_GPU_SPECS(double);
DECLARE_GPU_SPECS(at::Half);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_T_INDEX
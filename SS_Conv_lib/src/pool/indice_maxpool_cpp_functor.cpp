/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

c++/cuda source code for sparse max pool. 
Written by Hongyang Li and Jiehong Lin
# Modified from https://github.com/traveller59/spconv/tree/v1.1
--------------------------------------------------------
*/

#include <torch/script.h>
#include "indice_maxpool_functor.h"


namespace functor {
template <typename T, typename Index>
struct SparseMaxPoolForwardFunctor<tv::CPU, T, Index> {
  void operator()(const tv::CPU &d, tv::TensorView<T> outFeatures,
                  tv::TensorView<const T> inFeatures,
                  tv::TensorView<const Index> indices, int size) {
    int stride = outFeatures.dim(1);
    auto outFeaturesData = outFeatures.data();
    auto inFeaturesData = inFeatures.data();
    auto indicesIn = indices.subview(0).data();
    auto indicesOut = indices.subview(1).data();
    Index idxi, idxo;
    for (int row = 0; row < size; row++) {
      idxi = indicesIn[row] * stride;
      idxo = indicesOut[row] * stride;
      for (int plane = 0; plane < stride; ++plane)
        if (outFeaturesData[idxo + plane] < inFeaturesData[idxi + plane])
          outFeaturesData[idxo + plane] = inFeaturesData[idxi + plane];
    }
  }
};

template <typename T, typename Index>
struct SparseFieldMaxPoolForwardFunctor<tv::CPU, T, Index> {
  void operator()(const tv::CPU &d, tv::TensorView<T> outFeatures,
                  tv::TensorView<const T> inFeatures,
                  tv::TensorView<const Index> indices, int size,
                  tv::TensorView<const T> inFeature_norms, tv::TensorView<T> outFeature_norms) {
    int stride = outFeatures.dim(1);
    auto outFeaturesData = outFeatures.data();
    auto inFeaturesData = inFeatures.data();
    auto outFeatureNormsData = outFeature_norms.data();
    auto inFeatureNormsData = inFeature_norms.data();
    auto indicesIn = indices.subview(0).data();
    auto indicesOut = indices.subview(1).data();
    Index idxi, idxo;
    for (int row = 0; row < size; row++) {
      idxi = indicesIn[row] * stride;
      idxo = indicesOut[row] * stride;
      for (int plane = 0; plane < stride; ++plane)
        if (outFeatureNormsData[idxo + plane] < inFeatureNormsData[idxi + plane])
        {
          outFeaturesData[idxo + plane] = inFeaturesData[idxi + plane];
          outFeatureNormsData[idxo + plane] = inFeatureNormsData[idxi + plane];
        }
    }
  }
};

template <typename T, typename Index>
struct SparseMaxPoolBackwardFunctor<tv::CPU, T, Index> {
  void operator()(const tv::CPU &d, tv::TensorView<const T> outFeatures,
                  tv::TensorView<const T> inFeatures,
                  tv::TensorView<const T> dout, tv::TensorView<T> din,
                  tv::TensorView<const Index> indices, int size) {
    int stride = outFeatures.dim(1);
    auto outFeaturesData = outFeatures.data();
    auto inFeaturesData = inFeatures.data();
    auto doutData = dout.data();
    auto dinData = din.data();
    auto indicesIn = indices.subview(0).data();
    auto indicesOut = indices.subview(1).data();
    Index idxi, idxo;
    for (int row = 0; row < size; row++) {
      idxi = indicesIn[row] * stride;
      idxo = indicesOut[row] * stride;
      for (int plane = 0; plane < stride; ++plane)
        if (outFeaturesData[idxo + plane] == inFeaturesData[idxi + plane])
          dinData[idxi + plane] += doutData[idxo + plane];
    }
  }
};
} // namespace functor

#define DECLARE_CPU_SPECS_T_INDEX(T, Index)                                    \
  template struct functor::SparseMaxPoolForwardFunctor<tv::CPU, T, Index>;     \
  template struct functor::SparseMaxPoolBackwardFunctor<tv::CPU, T, Index>;    \
  template struct functor::SparseFieldMaxPoolForwardFunctor<tv::CPU, T, Index>;

#define DECLARE_CPU_SPECS(T)                                                   \
  DECLARE_CPU_SPECS_T_INDEX(T, int);                                           \
  DECLARE_CPU_SPECS_T_INDEX(T, long);

DECLARE_CPU_SPECS(float);
DECLARE_CPU_SPECS(double);
DECLARE_CPU_SPECS(at::Half);

#undef DECLARE_CPU_SPECS
#undef DECLARE_CPU_SPECS_T_INDEX


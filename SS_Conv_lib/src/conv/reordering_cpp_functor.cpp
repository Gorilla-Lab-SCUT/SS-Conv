/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

c++/cuda source code for sparse convolution. 
Written by Hongyang Li and Jiehong Lin
# Modified from https://github.com/traveller59/spconv/tree/v1.1
--------------------------------------------------------
*/

#include <torch/script.h>
#include "reordering_functor.h"

namespace functor {
template <typename T, typename Index>
struct SparseGatherFunctor<tv::CPU, T, Index> {
  void operator()(const tv::CPU& d, tv::TensorView<T> buffer, tv::TensorView<const T> features,
                  tv::TensorView<const Index> indices, int size) {
    int numPlanes = features.dim(1);
    for (int i = 0; i < size; ++i) {
      std::memcpy(buffer.data() + i * numPlanes,
                  features.data() + indices[i] * numPlanes,
                  sizeof(T) * numPlanes);
    }
  }
};

template <typename T, typename Index>
struct SparseScatterAddFunctor<tv::CPU, T, Index> {
  void operator()(const tv::CPU& d, tv::TensorView<T> outFeatures,
                  tv::TensorView<const T> buffer, tv::TensorView<const Index> indices,
                  int size, bool stable) {
    int numPlanes = outFeatures.dim(1);
    const T* buf = buffer.data();
    T* out = outFeatures.data();
    for (int i = 0; i < size; ++i) {
      buf = buffer.data() + i * numPlanes;
      out = outFeatures.data() + indices[i] * numPlanes;
      for (int j = 0; j < numPlanes; ++j){
        out[j] += buf[j];
      }
    }
  }
};

} // namespace functor


#define DECLARE_CPU_SPECS_T_INDEX(T, Index)               \
  template struct functor::SparseGatherFunctor<tv::CPU, T, Index>;  \
  template struct functor::SparseScatterAddFunctor<tv::CPU, T, Index>;

#define DECLARE_CPU_SPECS(T)                                                   \
  DECLARE_CPU_SPECS_T_INDEX(T, int);                                           \
  DECLARE_CPU_SPECS_T_INDEX(T, long);

DECLARE_CPU_SPECS(float);
DECLARE_CPU_SPECS(double);
DECLARE_CPU_SPECS(at::Half);

#undef DECLARE_CPU_SPECS
#undef DECLARE_CPU_SPECS_T_INDEX


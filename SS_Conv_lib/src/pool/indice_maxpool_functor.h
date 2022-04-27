/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

c++/cuda source code for sparse max pool. 
Written by Hongyang Li and Jiehong Lin
# Modified from https://github.com/traveller59/spconv/tree/v1.1
--------------------------------------------------------
*/

#ifndef SPARSE_MAXPOOL_FUNCTOR_H_
#define SPARSE_MAXPOOL_FUNCTOR_H_
#include "tensorview.h"

namespace functor
{
template <typename Device, typename T, typename Index>
struct SparseMaxPoolForwardFunctor
{
    void operator()(const Device& d, tv::TensorView<T> outFeatures,
                  tv::TensorView<const T> inFeatures,
                  tv::TensorView<const Index> indices, int size);
};

template <typename Device, typename T, typename Index>
struct SparseFieldMaxPoolForwardFunctor
{
    void operator()(const Device& d, tv::TensorView<T> outFeatures,
                  tv::TensorView<const T> inFeatures,
                  tv::TensorView<const Index> indices, int size,
                  tv::TensorView<const T> inFeature_norms, tv::TensorView<T> outFeature_norms);
};

template <typename Device, typename T, typename Index>
struct SparseMaxPoolBackwardFunctor
{
    void operator()(const Device& d, tv::TensorView<const T> outFeatures,
                  tv::TensorView<const T> inFeatures,
                  tv::TensorView<const T> dout,
                  tv::TensorView<T> din,
                  tv::TensorView<const Index> indices, int size);
};

} // namespace functor

#endif
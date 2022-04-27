/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

c++/cuda source code for sparse convolution. 
Written by Hongyang Li and Jiehong Lin
# Modified from https://github.com/traveller59/spconv/tree/v1.1
--------------------------------------------------------
*/
#ifndef SPARSE_REORDERING_FUNCTOR_H_
#define SPARSE_REORDERING_FUNCTOR_H_
#include "tensorview.h"

namespace functor
{
template <typename Device, typename T, typename Index>
struct SparseGatherFunctor
{
    void operator()(const Device& d, tv::TensorView<T> buffer, tv::TensorView<const T> features,
                    tv::TensorView<const Index> indices, int size);
};

template <typename Device, typename T, typename Index>
struct SparseScatterAddFunctor
{
    void operator()(const Device& d, tv::TensorView<T> out_features,
                    tv::TensorView<const T> buffer, tv::TensorView<const Index> indices,
                    int size, bool stable=false);
};
} // namespace functor

#endif
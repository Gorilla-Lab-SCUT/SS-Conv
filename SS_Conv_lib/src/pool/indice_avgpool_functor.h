/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

c++/cuda source code for average sparse pooling. 
Written by Hongyang Li and Jiehong Lin
# Modified from https://github.com/traveller59/spconv/tree/v1.1
--------------------------------------------------------
*/

#ifndef SPARSE_AVGPOOL_FUNCTOR_H_
#define SPARSE_AVGPOOL_FUNCTOR_H_
#include "tensorview.h"

namespace functor
{
template <typename Device, typename T, typename Index>
struct SparseAvgPoolForwardFunctor
{
    void operator()(const Device& d, tv::TensorView<T> outFeatures,
                  tv::TensorView<const T> inFeatures,
                  tv::TensorView<const Index> indices, int size, tv::TensorView<const Index> summaryrf);
};

template <typename Device, typename T, typename Index>
struct SparseAvgPoolBackwardFunctor
{
    void operator()(const Device& d, tv::TensorView<const T> outFeatures,
                  tv::TensorView<const T> inFeatures,
                  tv::TensorView<const T> dout,
                  tv::TensorView<T> din,
                  tv::TensorView<const Index> indices, int size,
                  tv::TensorView<const Index> summaryrf);
};


template <typename Device, typename Index>
struct SummaryRFForwardFunctor
{
    void operator()(const Device& d, 
                    tv::TensorView<const Index> indices, 
                    tv::TensorView<Index> num_RF, 
                    int size);
};

} // namespace functor

#endif
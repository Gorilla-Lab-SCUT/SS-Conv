/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

c++/cuda source code for sparse convolution. 
Written by Hongyang Li and Jiehong Lin
# Modified from https://github.com/traveller59/spconv/tree/v1.1
--------------------------------------------------------
*/
#ifndef SPARSE_CONV_INDICE_FUNCTOR_H_
#define SPARSE_CONV_INDICE_FUNCTOR_H_
#include "tensorview.h"

namespace functor
{
template <typename Device, typename Index, typename IndexGrid, unsigned NDim>
struct CreateConvIndicePairFunctorP1
{
    Index operator()(
        const Device& d, tv::TensorView<const Index> indicesIn,
        tv::TensorView<Index> indicesOut, tv::TensorView<IndexGrid> gridsOut,
        tv::TensorView<Index> indicePairs, tv::TensorView<Index> indiceNum,
        tv::TensorView<Index> indicePairUnique,
        const tv::SimpleVector<Index, NDim> kernelSize,
        const tv::SimpleVector<Index, NDim> stride,
        const tv::SimpleVector<Index, NDim> padding,
        const tv::SimpleVector<Index, NDim> dilation,
        const tv::SimpleVector<Index, NDim> outSpatialShape, bool transpose);
};

template <typename Device, typename Index, typename IndexGrid, unsigned NDim>
struct CreateConvIndicePairFunctorP2
{
    Index operator()(
        const Device& d, tv::TensorView<const Index> indicesIn,
        tv::TensorView<Index> indicesOut, tv::TensorView<IndexGrid> gridsOut,
        tv::TensorView<Index> indicePairs, tv::TensorView<Index> indiceNum,
        tv::TensorView<Index> indicePairUnique,
        const tv::SimpleVector<Index, NDim> outSpatialShape, bool transpose,
        bool resetGrid=false);
};

template <typename Device, typename Index, typename IndexGrid, unsigned NDim>
struct CreateConvIndicePairFunctor
{
    Index operator()(
        const Device& d, tv::TensorView<const Index> indicesIn,
        tv::TensorView<Index> indicesOut, tv::TensorView<IndexGrid> gridsOut,
        tv::TensorView<Index> indicePairs, tv::TensorView<Index> indiceNum,
        const tv::SimpleVector<Index, NDim> kernelSize,
        const tv::SimpleVector<Index, NDim> stride,
        const tv::SimpleVector<Index, NDim> padding,
        const tv::SimpleVector<Index, NDim> dilation,
        const tv::SimpleVector<Index, NDim> outSpatialShape, bool transpose, bool resetGrid=false);
};

template <typename Device, typename Index, typename IndexGrid, unsigned NDim>
struct CreateSubMIndicePairFunctor
{
    Index operator()(
        const Device& d, tv::TensorView<const Index> indicesIn, tv::TensorView<IndexGrid> gridsOut,
        tv::TensorView<Index> indicePairs, tv::TensorView<Index> indiceNum,
        const tv::SimpleVector<Index, NDim> kernelSize,
        const tv::SimpleVector<Index, NDim> stride,
        const tv::SimpleVector<Index, NDim> padding,
        const tv::SimpleVector<Index, NDim> dilation,
        const tv::SimpleVector<Index, NDim> outSpatialShape, bool transpose, bool resetGrid=false);
};
} // namespace functor

#endif
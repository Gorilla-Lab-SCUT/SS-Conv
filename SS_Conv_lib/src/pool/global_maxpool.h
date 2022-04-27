/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

c++/cuda source code for max roi-pooling. 
Written by Hongyang Li and Jiehong Lin
Modified https://github.com/dvlab-research/PointGroup
--------------------------------------------------------
*/

#ifndef MAXROIPOOL_H
#define MAXROIPOOL_H
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#include "datatype.h"

//
void global_maxpool_fp(at::Tensor feats_tensor, at::Tensor proposals_offset_tensor, at::Tensor output_feats_tensor, at::Tensor output_maxidx_tensor, int nProposal, int C);

void global_maxpool_fp_cuda(int nProposal, int C, float *feats, int *proposals_offset, float *output_feats, int *output_maxidx);


//
void global_maxpool_bp(at::Tensor d_feats_tensor, at::Tensor proposals_offset_tensor, at::Tensor output_maxidx_tensor, at::Tensor d_output_feats_tensor, int nProposal, int C);

void global_maxpool_bp_cuda(int nProposal, int C, float *d_feats, int *proposals_offset, int *output_maxidx, float *d_output_feats);

#endif //MAXROIPOOL_H

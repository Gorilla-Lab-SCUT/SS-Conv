/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

c++/cuda source code for average roi-pooling. 
Written by Hongyang Li and Jiehong Lin
Modified https://github.com/dvlab-research/PointGroup
--------------------------------------------------------
*/

#include "global_avgpool.h"

void global_avgpool_fp(at::Tensor feats_tensor, at::Tensor proposals_offset_tensor, at::Tensor output_feats_tensor, at::Tensor output_maxidx_tensor, int nProposal, int C){
    float *feats = feats_tensor.data<float>();
    int *proposals_offset = proposals_offset_tensor.data<int>();
    float *output_feats = output_feats_tensor.data<float>();
    int *output_maxidx = output_maxidx_tensor.data<int>();

    global_avgpool_fp_cuda(nProposal, C, feats, proposals_offset, output_feats, output_maxidx);
}


void global_avgpool_bp(at::Tensor d_feats_tensor, at::Tensor proposals_offset_tensor, at::Tensor output_maxidx_tensor, at::Tensor d_output_feats_tensor, int nProposal, int C){
    float *d_feats = d_feats_tensor.data<float>();
    int *proposals_offset = proposals_offset_tensor.data<int>();
    int *output_maxidx = output_maxidx_tensor.data<int>();
    float *d_output_feats = d_output_feats_tensor.data<float>();

    global_avgpool_bp_cuda(nProposal, C, d_feats, proposals_offset, output_maxidx, d_output_feats);
}
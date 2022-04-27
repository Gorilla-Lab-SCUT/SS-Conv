/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

c++/cuda source code for max roi-pooling. 
Written by Hongyang Li and Jiehong Lin
Modified https://github.com/dvlab-research/PointGroup
--------------------------------------------------------
*/

#include <stdio.h>
#include <math.h>
#include "global_maxpool.h"

// fp
__global__ void global_maxpool_fp_cuda_(int nProposal, int C, float *feats, int *proposals_offset, float *output_feats, int *output_maxidx){
    for(int pp_id = blockIdx.x; pp_id < nProposal; pp_id += gridDim.x){
        int start = proposals_offset[pp_id];
        int end = proposals_offset[pp_id + 1];

        for(int plane = threadIdx.x; plane < C; plane += blockDim.x){
            int argmax_idx = -1;
            float max_val = -1e50;

            for(int i = start; i < end; i++){
                if(feats[i * C + plane] > max_val){
                    argmax_idx = i;
                    max_val = feats[i * C + plane];
                }
            }
            output_maxidx[pp_id * C + plane] = argmax_idx;
            output_feats[pp_id * C + plane] = max_val;
        }
    }
}

//input: feats (sumNPoint, C) float
//input: proposals_offset (nProposal + 1) int
//output: output_feats (nProposal, C) float
//output: output_maxidx (nProposal, C) int
void global_maxpool_fp_cuda(int nProposal, int C, float *feats, int *proposals_offset, float *output_feats, int *output_maxidx){
    global_maxpool_fp_cuda_<<<std::min(nProposal, (int)32768), std::min(C, (int)32)>>>(nProposal, C, feats, proposals_offset, output_feats, output_maxidx);
}

// bp
__global__ void global_maxpool_bp_cuda_(int nProposal, int C, float *d_feats, int *proposals_offset, int *output_maxidx, float *d_output_feats){
    for(int pp_id = blockIdx.x; pp_id < nProposal; pp_id += gridDim.x){
        for(int plane = threadIdx.x; plane < C; plane += blockDim.x){
            int argmax_idx = output_maxidx[pp_id * C + plane];
            atomicAdd(&d_feats[argmax_idx * C + plane], d_output_feats[pp_id * C + plane]);
        }
    }
}

//input: d_output_feats (nProposal, C) float
//input: output_maxidx (nProposal, C) int
//input: proposals_offset (nProposal + 1) int
//output: d_feats (sumNPoint, C) float
void global_maxpool_bp_cuda(int nProposal, int C, float *d_feats, int *proposals_offset, int *output_maxidx, float *d_output_feats){
    global_maxpool_bp_cuda_<<<std::min(nProposal, (int)32768), std::min(C, (int)32)>>>(nProposal, C, d_feats, proposals_offset, output_maxidx, d_output_feats);
}

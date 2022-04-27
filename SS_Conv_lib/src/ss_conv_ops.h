/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

The Declarations of Functions for Different Data Types. 
Written by Hongyang Li and Jiehong Lin
Modified https://github.com/dvlab-research/PointGroup
--------------------------------------------------------
*/
#ifndef SSCONV_H
#define SSCONV_H
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include "datatype.h"
#include "global_maxpool.h"
#include "global_avgpool.h"
#include "voxel2point.h"
#include "interpolate.h"



std::vector<torch::Tensor> get_indice_pairs_2d(torch::Tensor indices, int64_t batchSize,
        std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
        std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
        std::vector<int64_t> padding, std::vector<int64_t> dilation,
        std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);
std::vector<torch::Tensor> get_indice_pairs_3d(torch::Tensor indices, int64_t batchSize,
        std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
        std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
        std::vector<int64_t> padding, std::vector<int64_t> dilation,
        std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);
std::vector<torch::Tensor> get_indice_pairs_grid_2d(torch::Tensor indices, torch::Tensor gridOut, int64_t batchSize,
        std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
        std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
        std::vector<int64_t> padding, std::vector<int64_t> dilation,
        std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);
std::vector<torch::Tensor> get_indice_pairs_grid_3d(torch::Tensor indices, torch::Tensor gridOut, int64_t batchSize,
        std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
        std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
        std::vector<int64_t> padding, std::vector<int64_t> dilation,
        std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);
torch::Tensor indice_conv_fp32(torch::Tensor features, torch::Tensor filters,
                       torch::Tensor indicePairs, torch::Tensor indiceNum,
                       int64_t numActOut, int64_t _inverse, int64_t _subM);
std::vector<torch::Tensor> indice_conv_backward_fp32(torch::Tensor features, torch::Tensor filters,
                 torch::Tensor outGrad, torch::Tensor indicePairs, torch::Tensor indiceNum,
                 int64_t _inverse, int64_t _subM);
torch::Tensor indice_conv_half(torch::Tensor features, torch::Tensor filters,
                       torch::Tensor indicePairs, torch::Tensor indiceNum,
                       int64_t numActOut, int64_t _inverse, int64_t _subM);
std::vector<torch::Tensor> indice_conv_backward_half(torch::Tensor features, torch::Tensor filters,
                 torch::Tensor outGrad, torch::Tensor indicePairs, torch::Tensor indiceNum,
                 int64_t _inverse, int64_t _subM);

torch::Tensor indiceMaxPool_fp_float(torch::Tensor features, torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numAct);
torch::Tensor indiceMaxPool_bp_float(torch::Tensor features, torch::Tensor outFeatures, torch::Tensor outGrad, torch::Tensor indicePairs, torch::Tensor indiceNum);
torch::Tensor indiceMaxPool_fp_half(torch::Tensor features, torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numAct);
torch::Tensor indiceMaxPool_bp_half(torch::Tensor features, torch::Tensor outFeatures, torch::Tensor outGrad, torch::Tensor indicePairs, torch::Tensor indiceNum);
torch::Tensor indiceAvgPool_fp_float(torch::Tensor features, torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numAct, torch::Tensor summaryrf);
torch::Tensor indiceAvgPool_bp_float(torch::Tensor features, torch::Tensor outFeatures, torch::Tensor outGrad, torch::Tensor indicePairs, torch::Tensor indiceNum, torch::Tensor summaryrf);
torch::Tensor indiceAvgPool_fp_half(torch::Tensor features, torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numAct, torch::Tensor summaryrf);
torch::Tensor indiceAvgPool_bp_half(torch::Tensor features, torch::Tensor outFeatures, torch::Tensor outGrad, torch::Tensor indicePairs, torch::Tensor indiceNum, torch::Tensor summaryrf);
torch::Tensor indiceSummaryRF(torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numAct);
torch::Tensor indiceFieldMaxPool_fp_float(torch::Tensor features, torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numAct, torch::Tensor feature_norms);
torch::Tensor indiceFieldMaxPool_fp_half(torch::Tensor features, torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numAct, torch::Tensor feature_norms);


void voxelize_idx_3d(/* long N*4 */ at::Tensor coords, /* long M*4 */ at::Tensor output_coords,
                  /* Int N */ at::Tensor input_map, /* Int M*(maxActive+1) */ at::Tensor output_map, Int batchSize, Int mode);
void voxelize_fp_feat(/* cuda float N*C */ at::Tensor feats, // N * 3 -> M * 3 (N >= M)
              /* cuda float M*C */ at::Tensor output_feats,
              /* cuda Int M*(maxActive+1) */ at::Tensor output_map, Int mode, Int nActive, Int maxActive, Int nPlane);
void voxelize_bp_feat(/* cuda float M*C */ at::Tensor d_output_feats, /* cuda float N*C */ at::Tensor d_feats, /* cuda Int M*(maxActive+1) */ at::Tensor output_map,
            Int mode, Int nActive, Int maxActive, Int nPlane);
void point_recover_fp_feat(/* cuda float M*C */ at::Tensor feats, /* cuda float N*C */ at::Tensor output_feats, /* cuda Int M*(maxActive+1) */ at::Tensor idx_map,
                Int nActive, Int maxActive, Int nPlane);
void point_recover_bp_feat(/* cuda float N*C */ at::Tensor d_output_feats, /* cuda float M*C */ at::Tensor d_feats,  /* cuda Int M*(maxActive+1) */ at::Tensor idx_map,
                Int nActive, Int maxActive, Int nPlane);

#endif // SSCONV_H
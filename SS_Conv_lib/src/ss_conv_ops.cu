/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

Template Specialzations for Different Data Types. 
Written by Hongyang Li and Jiehong Lin
Modified https://github.com/dvlab-research/PointGroup
--------------------------------------------------------
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "indice_maxpool.h"
#include "indice_avgpool.h"
#include "indice_conv.h"
#include "voxelize.h"


std::vector<torch::Tensor> get_indice_pairs_2d(torch::Tensor indices, int64_t batchSize,
        std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
        std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
        std::vector<int64_t> padding, std::vector<int64_t> dilation,
        std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose){
    return getIndicePair<2>(indices, batchSize,
        outSpatialShape, spatialShape,
        kernelSize, stride,
        padding, dilation,
        outPadding, _subM, _transpose);
}
std::vector<torch::Tensor> get_indice_pairs_3d(torch::Tensor indices, int64_t batchSize,
        std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
        std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
        std::vector<int64_t> padding, std::vector<int64_t> dilation,
        std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose){
    return getIndicePair<3>(indices, batchSize,
        outSpatialShape, spatialShape,
        kernelSize, stride,
        padding, dilation,
        outPadding, _subM, _transpose);
}
std::vector<torch::Tensor> get_indice_pairs_grid_2d(torch::Tensor indices, torch::Tensor gridOut, int64_t batchSize,
        std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
        std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
        std::vector<int64_t> padding, std::vector<int64_t> dilation,
        std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose){
    return getIndicePairPreGrid<2>(indices, gridOut, batchSize,
        outSpatialShape, spatialShape,
        kernelSize, stride,
        padding, dilation,
        outPadding, _subM, _transpose);
}
std::vector<torch::Tensor> get_indice_pairs_grid_3d(torch::Tensor indices, torch::Tensor gridOut, int64_t batchSize,
        std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
        std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
        std::vector<int64_t> padding, std::vector<int64_t> dilation,
        std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose){
    return getIndicePairPreGrid<3>(indices, gridOut, batchSize,
        outSpatialShape, spatialShape,
        kernelSize, stride,
        padding, dilation,
        outPadding, _subM, _transpose);
}
torch::Tensor indice_conv_fp32(torch::Tensor features, torch::Tensor filters,
                       torch::Tensor indicePairs, torch::Tensor indiceNum,
                       int64_t numActOut, int64_t _inverse, int64_t _subM){
    return indiceConv<float>(features, filters, indicePairs, indiceNum, numActOut, _inverse, _subM);
}
std::vector<torch::Tensor> indice_conv_backward_fp32(torch::Tensor features, torch::Tensor filters,
                 torch::Tensor outGrad, torch::Tensor indicePairs, torch::Tensor indiceNum,
                 int64_t _inverse, int64_t _subM){
    return indiceConvBackward<float>(features, filters, outGrad, indicePairs, indiceNum, _inverse, _subM);
}
torch::Tensor indice_conv_half(torch::Tensor features, torch::Tensor filters,
                       torch::Tensor indicePairs, torch::Tensor indiceNum,
                       int64_t numActOut, int64_t _inverse, int64_t _subM){
    return indiceConv<at::Half>(features, filters, indicePairs, indiceNum, numActOut, _inverse, _subM);
}
std::vector<torch::Tensor> indice_conv_backward_half(torch::Tensor features, torch::Tensor filters,
                 torch::Tensor outGrad, torch::Tensor indicePairs, torch::Tensor indiceNum,
                 int64_t _inverse, int64_t _subM){
    return indiceConvBackward<at::Half>(features, filters, outGrad, indicePairs, indiceNum, _inverse, _subM);
}


torch::Tensor indiceMaxPool_fp_float(torch::Tensor features, torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numAct){
    return indiceMaxPool<float>(features, indicePairs, indiceNum, numAct);
}
torch::Tensor indiceMaxPool_bp_float(torch::Tensor features, torch::Tensor outFeatures, torch::Tensor outGrad, torch::Tensor indicePairs, torch::Tensor indiceNum){
    return indiceMaxPoolBackward<float>(features, outFeatures, outGrad, indicePairs, indiceNum);
}
torch::Tensor indiceMaxPool_fp_half(torch::Tensor features, torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numAct){
    indiceMaxPool<at::Half>(features, indicePairs, indiceNum, numAct);
}
torch::Tensor indiceMaxPool_bp_half(torch::Tensor features, torch::Tensor outFeatures, torch::Tensor outGrad, torch::Tensor indicePairs, torch::Tensor indiceNum){
    indiceMaxPoolBackward<at::Half>(features, outFeatures, outGrad, indicePairs, indiceNum);
}
torch::Tensor indiceFieldMaxPool_fp_float(torch::Tensor features, torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numAct, torch::Tensor feature_norms){
    return indiceFieldMaxPool<float>(features, indicePairs, indiceNum, numAct, feature_norms);
}
torch::Tensor indiceFieldMaxPool_fp_half(torch::Tensor features, torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numAct, torch::Tensor feature_norms){
    return indiceFieldMaxPool<half>(features, indicePairs, indiceNum, numAct, feature_norms);
}
torch::Tensor indiceAvgPool_fp_float(torch::Tensor features, torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numAct, torch::Tensor summaryrf){
    return indiceAvgPool<float>(features, indicePairs, indiceNum, numAct, summaryrf);
}
torch::Tensor indiceAvgPool_bp_float(torch::Tensor features, torch::Tensor outFeatures, torch::Tensor outGrad, torch::Tensor indicePairs, torch::Tensor indiceNum, torch::Tensor summaryrf){
    return indiceAvgPoolBackward<float>(features, outFeatures, outGrad, indicePairs, indiceNum, summaryrf);
}
torch::Tensor indiceAvgPool_fp_half(torch::Tensor features, torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numAct, torch::Tensor summaryrf){
    return indiceAvgPool<at::Half>(features, indicePairs, indiceNum, numAct, summaryrf);
}
torch::Tensor indiceAvgPool_bp_half(torch::Tensor features, torch::Tensor outFeatures, torch::Tensor outGrad, torch::Tensor indicePairs, torch::Tensor indiceNum, torch::Tensor summaryrf){
    return indiceAvgPoolBackward<at::Half>(features, outFeatures, outGrad, indicePairs, indiceNum, summaryrf);
}
torch::Tensor indiceSummaryRF(torch::Tensor indicePairs,
                              torch::Tensor indiceNum, 
                              int64_t numAct){
  auto device             = indicePairs.device().type();
  auto kernelVolume       = indicePairs.size(0);
  auto indicePairNumCpu   = indiceNum.to({torch::kCPU});
  auto options            = torch::TensorOptions().dtype(indicePairs.dtype()).device(indicePairs.device());
  torch::Tensor summarRFs = torch::zeros({numAct}, options);
  for(int i = 0; i<kernelVolume; ++i){
    auto nHot = indicePairNumCpu.data<int>()[i];
    if (nHot <= 0) {
      continue;
    }
    if  (device == torch::kCPU) {
      continue;
    }
    else{
      functor::SummaryRFForwardFunctor<tv::GPU, int> forwardFtor;
      forwardFtor(
        tv::TorchGPU(),
        tv::torch2tv<const int>(indicePairs).subview(i),
        tv::torch2tv<int>(summarRFs),
        nHot
      );
      // printf("summarRFs[0]: %d \n", tv::torch2tv<int>(summarRFs)[0]);
    TV_CHECK_CUDA_ERR();
    }
  }
  return summarRFs;
}

void voxelize_idx_3d(/* long N*4 */ at::Tensor coords, /* long M*4 */ at::Tensor output_coords,
                  /* Int N */ at::Tensor input_map, /* Int M*(maxActive+1) */ at::Tensor output_map, Int batchSize, Int mode){
    voxelize_idx<3>(coords, output_coords, input_map, output_map, batchSize, mode);
}
void voxelize_fp_feat(/* cuda float N*C */ at::Tensor feats, // N * 3 -> M * 3 (N >= M)
              /* cuda float M*C */ at::Tensor output_feats,
              /* cuda Int M*(maxActive+1) */ at::Tensor output_map, Int mode, Int nActive, Int maxActive, Int nPlane){
    voxelize_fp<float>(feats, output_feats, output_map, mode, nActive, maxActive, nPlane);
}
void voxelize_bp_feat(/* cuda float M*C */ at::Tensor d_output_feats, /* cuda float N*C */ at::Tensor d_feats, /* cuda Int M*(maxActive+1) */ at::Tensor output_map,
            Int mode, Int nActive, Int maxActive, Int nPlane){
    voxelize_bp<float>(d_output_feats, d_feats, output_map, mode, nActive, maxActive, nPlane);
}
void point_recover_fp_feat(/* cuda float M*C */ at::Tensor feats, /* cuda float N*C */ at::Tensor output_feats, /* cuda Int M*(maxActive+1) */ at::Tensor idx_map,
                Int nActive, Int maxActive, Int nPlane){
    point_recover_fp<float>(feats, output_feats, idx_map, nActive, maxActive, nPlane);
}
void point_recover_bp_feat(/* cuda float N*C */ at::Tensor d_output_feats, /* cuda float M*C */ at::Tensor d_feats,  /* cuda Int M*(maxActive+1) */ at::Tensor idx_map,
                Int nActive, Int maxActive, Int nPlane){
    point_recover_bp<float>(d_output_feats, d_feats, idx_map, nActive, maxActive, nPlane);
}


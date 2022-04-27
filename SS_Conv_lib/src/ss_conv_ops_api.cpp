/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

C++/CUDA Extention APIs for Sparse Operation. 
Written by Hongyang Li and Jiehong Lin
Modified https://github.com/dvlab-research/PointGroup
--------------------------------------------------------
*/
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "ss_conv_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("get_indice_pairs_2d", &get_indice_pairs_2d, "get_indice_pairs_2d");
    m.def("get_indice_pairs_3d", &get_indice_pairs_3d, "get_indice_pairs_3d");
    m.def("get_indice_pairs_grid_2d", &get_indice_pairs_grid_2d, "get_indice_pairs_grid_2d");
    m.def("get_indice_pairs_grid_3d", &get_indice_pairs_grid_3d, "get_indice_pairs_grid_3d");

    m.def("indice_conv_fp32", &indice_conv_fp32, "indice_conv_fp32");
    m.def("indice_conv_backward_fp32", &indice_conv_backward_fp32, "indice_conv_backward_fp32");
    m.def("indice_conv_half", &indice_conv_half, "indice_conv_half");
    m.def("indice_conv_backward_half", &indice_conv_backward_half, "indice_conv_backward_half");

    m.def("indiceSummaryRF", &indiceSummaryRF, "indiceSummaryRF");
    m.def("indice_avgpool_fp32", &indiceAvgPool_fp_float, "indiceAvgPool_fp_float");
    m.def("indice_avgpool_backward_fp32", &indiceAvgPool_bp_float, "indiceAvgPool_bp_float");
    m.def("indice_avgpool_half", &indiceAvgPool_fp_half, "indiceAvgPool_fp_half");
    m.def("indice_avgpool_backward_half", &indiceAvgPool_bp_half, "indiceAvgPool_bp_half");

    m.def("indice_maxpool_fp32", &indiceMaxPool_fp_float, "indice_maxpool_fp32");
    m.def("indice_maxpool_backward_fp32", &indiceMaxPool_bp_float, "indice_maxpool_backward_fp32");
    m.def("indice_maxpool_half", &indiceMaxPool_fp_half, "indice_maxpool_half");
    m.def("indice_maxpool_backward_half", &indiceMaxPool_bp_half, "indice_maxpool_backward_half");
    m.def("indice_field_maxpool_fp32", &indiceFieldMaxPool_fp_float, "indiceFieldMaxPool_fp_float");
    m.def("indice_field_maxpool_half", &indiceFieldMaxPool_fp_half, "indiceFieldMaxPool_fp_half");

    m.def("global_avgpool_fp", &global_avgpool_fp, "global_avgpool_fp");
    m.def("global_avgpool_bp", &global_avgpool_bp, "global_avgpool_bp");
    m.def("global_maxpool_fp", &global_maxpool_fp, "global_maxpool_fp");
    m.def("global_maxpool_bp", &global_maxpool_bp, "global_maxpool_bp");

    m.def("voxelize_idx", &voxelize_idx_3d, "voxelize_idx");
    m.def("voxel2point_map", &voxel2point_fp, "voxel2point_map");
    m.def("voxelize_fp", &voxelize_fp_feat, "voxelize_fp");
    m.def("voxelize_bp", &voxelize_bp_feat, "voxelize_bp");
    m.def("point_recover_fp", &point_recover_fp_feat, "point_recover_fp");
    m.def("point_recover_bp", &point_recover_bp_feat, "point_recover_bp");

    m.def("three_nn", &three_nn, "three_nn");
    m.def("three_interpolate_fp", &three_interpolate_fp, "three_interpolate_fp");
    m.def("three_interpolate_bp", &three_interpolate_bp, "three_interpolate_bp");

}
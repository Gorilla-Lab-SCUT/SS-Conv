/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

c++/cuda source code for voxel2point mapping. 
Written by Hongyang Li and Jiehong Lin
--------------------------------------------------------
*/

#include "voxel2point.h"

void voxel2point_fp(at::Tensor indices_tensor, at::Tensor ids_tensor, at::Tensor output_maps_tensor, int nPoint, int maxActive, int mode){
    int *indices = indices_tensor.data<int>();
    int *ids = ids_tensor.data<int>();
    int *output_maps = output_maps_tensor.data<int>();

    voxel2point_fp_cuda(nPoint, maxActive, mode, indices, ids, output_maps);
}

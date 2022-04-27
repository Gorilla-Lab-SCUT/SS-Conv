/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

c++/cuda source code for voxel2point mapping. 
Written by Hongyang Li and Jiehong Lin
--------------------------------------------------------
*/

#ifndef VOXEL2POINT_H
#define VOXEL2POINT_H
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#include "datatype.h"

void voxel2point_fp(at::Tensor indices_tensor, at::Tensor ids_tensor, at::Tensor output_maps_tensor, int nPoint, int maxActive, int mode);

void voxel2point_fp_cuda(int nPoint, int maxActive, int mode, int *indices, int *ids, int *output_maps);

#endif
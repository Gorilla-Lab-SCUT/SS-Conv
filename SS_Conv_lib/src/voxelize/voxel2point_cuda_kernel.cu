/*
--------------------------------------------------------
Sparse Steerable Convolution Lib.

c++/cuda source code for voxel2point mapping. 
Written by Hongyang Li and Jiehong Lin
--------------------------------------------------------
*/

#include <stdio.h>
#include <math.h>
#include "voxel2point.h"


// fp
__global__ void voxel2point_fp_cuda_(int nPoint, int maxActive, int mode, int *indices, int *ids, int *output_maps){
    for(int row = blockIdx.x; row < nPoint; row += gridDim.x){
        int indice = indices[row];
        int id = ids[row];
        int *out = output_maps + indice*(maxActive+1);

        if (mode==1) {
            atomicMin(&out[1], id);
            out[0] = 1;
        }

        if (mode==2) {
            atomicMax(&out[1], id);
            out[0] = 1;
        }

        if (mode==3 or mode==4){
            int count = atomicAdd(&out[0], 1);
            out[count+1] = id;
        }
    }
}


void voxel2point_fp_cuda(int nPoint, int maxActive, int mode, int *indices, int *ids, int *output_maps){
    voxel2point_fp_cuda_<<<std::min(nPoint, (int)32768), std::min((int)1, (int)32)>>>(nPoint, maxActive, mode, indices, ids, output_maps);
}


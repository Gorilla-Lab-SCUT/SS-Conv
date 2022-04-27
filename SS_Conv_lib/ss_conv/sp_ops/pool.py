# --------------------------------------------------------
# Sparse Steerable Convolution Lib.

# Functions on Sparse Tensors
# Written by Hongyang Li and Jiehong Lin
# Modified from https://github.com/traveller59/spconv/tree/v1.1
# --------------------------------------------------------
import torch
import torch.nn as nn

from .tensor import SparseTensor
from .ops import get_conv_output_size, get_indice_pairs
from .functional import indice_avgpool, indice_maxpool, indice_fieldmaxpool


class SparseAvgPool(nn.Module):
    def __init__(self,
                 ndim,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 subm=False):
        super(SparseAvgPool, self).__init__()
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size] * ndim
        if not isinstance(stride, (list, tuple)):
            stride = [stride] * ndim
        if not isinstance(padding, (list, tuple)):
            padding = [padding] * ndim
        if not isinstance(dilation, (list, tuple)):
            dilation = [dilation] * ndim

        self.ndim = ndim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.subm = subm
        self.dilation = dilation

    def forward(self, input):
        assert isinstance(input, SparseTensor)
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size

        if not self.subm:
            out_spatial_shape = get_conv_output_size(
                spatial_shape, self.kernel_size, self.stride, self.padding, self.dilation)
        else:
            out_spatial_shape = spatial_shape
        outids, indice_pairs, indice_pairs_num = get_indice_pairs(
            indices, batch_size, spatial_shape, self.kernel_size,
            self.stride, self.padding, self.dilation, 0, self.subm)
        
        out_features = indice_avgpool(features, indice_pairs.to(device),
                                        indice_pairs_num.to(device), outids.shape[0])
        out_tensor = SparseTensor(out_features, outids,
                                             out_spatial_shape, batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


class SparseMaxPool(nn.Module):
    def __init__(self,
                 ndim,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 subm=False):
        super(SparseMaxPool, self).__init__()
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size] * ndim
        if not isinstance(stride, (list, tuple)):
            stride = [stride] * ndim
        if not isinstance(padding, (list, tuple)):
            padding = [padding] * ndim
        if not isinstance(dilation, (list, tuple)):
            dilation = [dilation] * ndim

        self.ndim = ndim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.subm = subm
        self.dilation = dilation

    def forward(self, input):
        assert isinstance(input, SparseTensor)
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        if not self.subm:
            out_spatial_shape = get_conv_output_size(
                spatial_shape, self.kernel_size, self.stride, self.padding, self.dilation)
        else:
            out_spatial_shape = spatial_shape
        outids, indice_pairs, indice_pairs_num = get_indice_pairs(
            indices, batch_size, spatial_shape, self.kernel_size,
            self.stride, self.padding, self.dilation, 0, self.subm)
        
        out_features = indice_maxpool(features, indice_pairs.to(device),
                                        indice_pairs_num.to(device), outids.shape[0])
        out_tensor = SparseTensor(out_features, outids,
                                             out_spatial_shape, batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor

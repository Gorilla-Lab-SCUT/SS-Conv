# --------------------------------------------------------
# Sparse Steerable Convolution Lib.

# Convolutions on Sparse Tensors
# Written by Jiehong Lin
# Modified from https://github.com/traveller59/spconv/tree/v1.1
# --------------------------------------------------------

import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .tensor import SparseTensor
from .ops import get_deconv_output_size, get_conv_output_size, get_indice_pairs
from .functional import indice_subm_conv, indice_inverse_conv, indice_conv


class SparseConvolution(nn.Module):
    def __init__(self,
                 ndim,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 subm=False,
                 output_padding=0,
                 transposed=False,
                 inverse=False,
                 indice_key=None):
        super(SparseConvolution, self).__init__()

        assert groups == 1
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size] * ndim
        if not isinstance(stride, (list, tuple)):
            stride = [stride] * ndim
        if not isinstance(padding, (list, tuple)):
            padding = [padding] * ndim
        if not isinstance(dilation, (list, tuple)):
            dilation = [dilation] * ndim
        if not isinstance(output_padding, (list, tuple)):
            output_padding = [output_padding] * ndim

        for d, s in zip(dilation, stride):
            assert any([s == 1, d == 1]), "don't support this."

        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv1x1 = np.prod(kernel_size) == 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.inverse = inverse
        self.output_padding = output_padding
        self.groups = groups
        self.subm = subm
        self.indice_key = indice_key

        self.init_parameters(bias)


    def calculate_fan_in(self):
        # channel_in: int
        # chennel_out: int
        # kenerl_size: torch.tensor
        fan_in = self.in_channels * self.kernel_size.prod()

        return fan_in

    def init_parameters(self, bias):
        self.weight = Parameter(
            torch.Tensor(*self.kernel_size, self.in_channels, self.out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.calculate_fan_in()
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return self._sparse_conv(input, self.weight)

    def _sparse_conv(self, input, weight):
        assert isinstance(input, SparseTensor)
        features = input.features
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size

        device = features.device

        if not self.subm:
            if self.transposed:
                out_spatial_shape = get_deconv_output_size(
                    spatial_shape, self.kernel_size, self.stride, self.padding, self.dilation, self.output_padding)
            else:
                out_spatial_shape = get_conv_output_size(
                    spatial_shape, self.kernel_size, self.stride, self.padding, self.dilation)

        else:
            out_spatial_shape = spatial_shape

        if self.conv1x1:
            input.features = torch.mm(
                input.features,
                weight.view(self.in_channels, self.out_channels))
            if self.bias is not None:
                input.features += self.bias
            return input

        datas = input.find_indice_pair(self.indice_key)
        if self.inverse:
            assert datas is not None and self.indice_key is not None
            _, outids, indice_pairs, indice_pair_num, out_spatial_shape = datas
            assert indice_pairs.shape[0] == np.prod(self.kernel_size), "inverse conv must have same kernel size as its couple conv"
        else:
            if self.indice_key is not None and datas is not None:
                outids, _, indice_pairs, indice_pair_num, _ = datas
            else:
                outids, indice_pairs, indice_pair_num = get_indice_pairs(
                    indices, batch_size, spatial_shape, self.kernel_size,
                    self.stride, self.padding, self.dilation, self.output_padding, self.subm, self.transposed, grid=input.grid)
                input.indice_dict[self.indice_key] = (outids, indices, indice_pairs, indice_pair_num, spatial_shape)

        if self.subm:
            out_features = indice_subm_conv(features, weight,
                                              indice_pairs.to(device),
                                              indice_pair_num,
                                              outids.shape[0])
        else:
            if self.inverse:
                out_features = indice_inverse_conv(features,
                                            weight, indice_pairs.to(device),
                                            indice_pair_num, outids.shape[0])
            else:
                out_features = indice_conv(features,
                                            weight, indice_pairs.to(device),
                                            indice_pair_num, outids.shape[0])

        if self.bias is not None:
            out_features += self.bias

        out_tensor = SparseTensor(out_features, outids,
                                             out_spatial_shape, batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


# --------------------------------------------------------
# Sparse Steerable Convolution Lib.

# Local & global pooling operations
# Written by Jiehong Lin and Hongyang Li
# --------------------------------------------------------

import torch
import torch.nn as nn

from ss_conv.sp_ops.tensor import SparseTensor
from ss_conv.sp_ops.pool import SparseAvgPool, SparseMaxPool
from ss_conv.sp_ops.functional import global_avgpool, global_maxpool

class AvgPool(SparseAvgPool):
    def __init__(
        self,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        subm=False):
        super(AvgPool, self).__init__(
            ndim=3,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            subm=subm
        )

class MaxPool(SparseMaxPool):
    def __init__(
        self,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        subm=False):
        super(MaxPool, self).__init__(
            ndim=3,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            subm=subm
        )

class MaxPool(SparseMaxPool):
    def __init__(
        self,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        subm=False):
        super(MaxPool, self).__init__(
            ndim=3,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            subm=subm
        )

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.func = global_avgpool

    def forward(self, inputs, offsets=None):
        if isinstance(input, SparseTensor):
            feats = inputs.features
            offsets = inputs.get_offsets() if offsets is None else offsets
        else:
            assert offsets is not None
            feats = inputs
        return self.func(feats, offsets.detach())

class GlobalMaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.func = global_maxpool

    def forward(self, inputs, offsets=None):
        if isinstance(input, SparseTensor):
            feats = inputs.features
            offsets = inputs.get_offsets() if offsets is None else offsets
        else:
            assert offsets is not None
            feats = inputs
        return self.func(feats, offsets.detach())
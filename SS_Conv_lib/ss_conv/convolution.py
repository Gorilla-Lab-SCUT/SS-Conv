# --------------------------------------------------------
# Sparse Steerable Convolution Lib.

# Sparse steerable convolutional operation
# Written by Jiehong Lin and Hongyang Li
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from ss_conv.sp_ops.tensor import SparseTensor
from ss_conv.sp_ops.conv import SparseConvolution
from ss_conv.utils.kernel import SE3Kernel, gaussian_window_wrapper
from ss_conv.utils.utils import Rs2dim


class Convolution(SparseConvolution):
    def __init__(self, Rs_in, Rs_out, kernel_size, 
                 radial_window=gaussian_window_wrapper, 
                 dyn_iso=False, 
                 verbose=False,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 subm=False,
                 output_padding=0,
                 transposed=False,
                 inverse=False,
                 indice_key=None):
        super(Convolution, self).__init__(
            ndim=3,
            in_channels=Rs2dim(Rs_in),
            out_channels=Rs2dim(Rs_out),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            subm=subm,
            output_padding=output_padding,
            transposed=transposed,
            inverse=inverse,
            indice_key=indice_key
        )
        self.kernel = SE3Kernel(Rs_in, Rs_out, kernel_size, radial_window=radial_window, dyn_iso=dyn_iso, verbose=verbose)

    def forward(self, input):
        assert isinstance(input, SparseTensor)
        if self.training:
            weight = self.kernel()
            weight = weight.permute(2,3,4,1,0).to(input.features.device)
            return self._sparse_conv(input, weight)
        else:
            if self.weight is None:
                self.weight = self.kernel()
                self.weight = self.weight.permute(2,3,4,1,0).to(input.features.device)
            return self._sparse_conv(input, self.weight)

    def init_parameters(self, bias):
        self.weight = None
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
            fan_in = self.calculate_fan_in()
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)





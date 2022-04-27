# --------------------------------------------------------
# Sparse Steerable Convolution Lib.

# Sparse SE(3)-equivariant activation
# Written by Hongyang Li and Jiehong Lin
# Modified From https://github.com/tscohen/se3cnn 
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ss_conv.utils.utils import Rs2dim


class Activation(nn.Module):
    def __init__(self, Rs, activation=(None, None), bias=True, checkpoint=True):
        super().__init__()
        '''
        :param Rs: list of tuple (multiplicity, order)
        '''
        self.Rs = Rs
        self.multiplicities = [m for m, _ in self.Rs]
        self.orders = [l for _, l in self.Rs]
        self.checkpoint = checkpoint

        if type(activation) is tuple:
            scalar_activation, gate_activation = activation
        else:
            scalar_activation, gate_activation = activation, activation


        if (scalar_activation is not None and self.multiplicities[0]>0 and self.orders[0]==0):
            self.scalar_act = ScalarActivation([(self.multiplicities[0], scalar_activation)], bias=bias)
        else:
            self.scalar_act = None

        n_non_scalar = sum(self.multiplicities[1:])
        if gate_activation is not None and n_non_scalar > 0:
            self.gate_act = ScalarActivation([(n_non_scalar, gate_activation)], bias=bias)
        else:
            self.gate_act = None
    
    def forward(self, input):
        if self.scalar_act is not None or self.gate_act is not None:
            features = input.features
            if self.checkpoint:
                features = torch.utils.checkpoint.checkpoint(self._gate, features)
            else:
                features = self._gate(features)    
            input.features = features

        return input

    def _gate(self, y):
        N = y.size(0)
        out_dim = Rs2dim(self.Rs)

        if self.gate_act is not None:
            g = y[:, out_dim:]
            g = self.gate_act(g)
            begin_g = 0

        z = y.new_empty((y.size(0), out_dim))
        begin_y = 0 

        for m, l in self.Rs:
            if m == 0:
                continue
            d = 2*l+1
            field_y = y[:, begin_y:begin_y+m*d]

            if l == 0:
                # Scalar activation
                if self.scalar_act is not None:
                    field = self.scalar_act(field_y)
                else:
                    field = field_y
            else:
                if self.gate_act is not None:
                    # crop out corresponding scalar gates
                    field_g = g[:, begin_g:begin_g+m]

                    # reshape channels for broadcasting
                    field_y = field_y.contiguous().view(N, m, d)
                    field_g = field_g.contiguous().view(N, m, 1)

                    # scale non-scalar capsules by gate values
                    field = field_y * field_g
                    field = field.view(N, m*d)

                    begin_g += m
                    del field_g
                else:
                    field = field_y
            z[:, begin_y: begin_y+m*d] = field
            begin_y += m * d
            del field, field_y

        return z


class ScalarActivation(nn.Module):
    def __init__(self, enable, bias=True, inplace=False):
        '''
        Can be used only with scalar fields

        :param enable: list of tuple (dimension, activation function or None)
        :param bool bias: add a bias before the applying the activation
        '''
        super().__init__()

        self.inplace = inplace
        self.enable = []
        for d, act in enable:
            if d == 0:
                continue

            if self.enable and self.enable[-1][1] is act:
                self.enable[-1] = (self.enable[-1][0] + d, act)
            else:
                self.enable.append((d, act))

        nbias = sum([d for d, act in self.enable if act is not None])
        if bias and nbias > 0:
            self.bias = torch.nn.Parameter(torch.zeros(nbias))
        else:
            self.bias = None

    def forward(self, input):  # pylint: disable=W
        '''
        :param input: [ΣNi, feature]
        '''
        begin1 = 0
        begin2 = 0

        if self.inplace:
            output = input
        else:
            output = torch.empty_like(input)

        for d, act in self.enable:
            x = input[:, begin1:begin1 + d].view(-1, d)  # [ΣNi, feature_repr]

            if act is not None:
                if self.bias is not None:
                    x = x + self.bias[begin2:begin2 + d].view(1, d)  # [1, feature_repr]  add bias before the act
                    begin2 += d

                x = act(x)

            if not self.inplace or act is not None:
                output[:, begin1:begin1 + d] = x.reshape(output[:, begin1:begin1 + d].size())

            begin1 += d

        assert begin1 == input.size(1)
        assert self.bias is None or begin2 == self.bias.size(0)

        return output


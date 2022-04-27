# --------------------------------------------------------
# Sparse Steerable Convolution Lib.

# Sparse SE(3)-equivariant batch norm
# Written by Hongyang Li and Jiehong Lin
# Modified From https://github.com/tscohen/se3cnn 
# --------------------------------------------------------

import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    def __init__(self, Rs, eps=1e-5, momentum=0.1, affine=True, reduce='mean'):
        '''
        :param Rs: list of tuple (multiplicity, order)
        '''
        super().__init__()
        self.Rs = Rs 
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.reduce = reduce

        num_scalar = sum(m for m, l in Rs if l == 0)
        num_features = sum(m for m, l in Rs)

        self.register_buffer('running_mean', torch.zeros(num_scalar))
        self.register_buffer('running_var', torch.ones(num_features))

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def __repr__(self):
        return "{} (Rs={}, eps={}, momentum={})".format(
            self.__class__.__name__,
            self.Rs,
            self.eps,
            self.momentum)

    def _roll_avg(self, curr, update):
        return (1 - self.momentum) * curr + self.momentum * update.detach()


    def forward(self, input):  # pylint: disable=W
        features = input.features
        '''
        :param features: [ΣNi, stacked_feature]
        '''

        if self.training:
            new_means = []
            new_vars = []

        fields = []
        ix = 0
        irm = 0
        irv = 0
        iw = 0
        ib = 0

        for m, l in self.Rs:
            d = 2*l+1

            field = features[:, ix: ix + m * d]  # [ΣNi, feature * repr]
            ix += m * d
            field = field.contiguous().view(features.size(0), m, d)  # [ΣNi, feature * repr] --> [ΣNi, feature , repr]

            if d == 1:  # scalars
                if self.training:
                    field_mean = field.mean(0).view(-1)  # [feature]
                    new_means.append(
                        self._roll_avg(self.running_mean[irm:irm+m], field_mean)
                    )
                else:
                    field_mean = self.running_mean[irm: irm + m]
                irm += m
                # [batch, feature, repr, x * y * z]
                field = field - field_mean.view(1, m, 1)

            if self.training:
                field_norm = torch.sum(field ** 2, dim=2)  # [ΣNi, feature , repr] --> [ΣNi, feature]
                if self.reduce == 'mean':
                    field_norm = field_norm.mean(0)  # [feature]
                elif self.reduce == 'max':
                    raise ValueError("Invalid reduce option {}".format(self.reduce))
                else:
                    raise ValueError("Invalid reduce option {}".format(self.reduce))

                # field_norm = field_norm.mean(0)  # [feature]
                new_vars.append(self._roll_avg(self.running_var[irv: irv+m], field_norm))
            else:
                field_norm = self.running_var[irv: irv + m]
            irv += m

            # [batch, feature, repr, x * y * z]
            field_norm = (field_norm + self.eps).pow(-0.5).view(1, m, 1)

            if self.affine:
                weight = self.weight[iw: iw + m]  # [feature]
                iw += m
                # [batch, feature, repr, x * y * z]
                field_norm = field_norm * weight.view(1, m, 1)
            # [ΣNi, feature , repr]
            field = field * field_norm  # [ΣNi, feature , repr] * [1, feature , 1] --> [ΣNi, feature , repr]

            if self.affine and d == 1:  # scalars
                bias = self.bias[ib: ib + m]  # [feature]
                ib += m
                field += bias.view(1, m, 1)  # [ΣNi, feature , repr]
            fields.append(field.view(features.size(0), m * d))

        if ix != features.size(1):
            fmt = "`ix` should have reached features.size(1) ({}), but it ended at {}"
            msg = fmt.format(features.size(1), ix)
            raise AssertionError(msg)

        if self.training:
            assert irm == self.running_mean.numel()
            assert irv == self.running_var.size(0)
        if self.affine:
            assert iw == self.weight.size(0)
            assert ib == self.bias.numel()

        if self.training:
            self.running_mean.copy_(torch.cat(new_means))
            self.running_var.copy_(torch.cat(new_vars))

        new_features = torch.cat(fields, dim=1)  # [batch, stacked_feature]
        input.features = new_features

        return input


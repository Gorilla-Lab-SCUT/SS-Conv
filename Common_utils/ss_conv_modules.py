# --------------------------------------------------------
# Sparse Steerable Convolutions

# Sparse steerable convolutional modules
# Written by Jiehong Lin
# --------------------------------------------------------
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import ss_conv
from ss_conv.convolution import Convolution
from ss_conv.batchnorm import BatchNorm
from ss_conv.activation import Activation
from ss_conv.dropout import Dropout
from ss_conv.utils.kernel import gaussian_window_wrapper
radial_window = partial(gaussian_window_wrapper,mode="compromise", border_dist=0, sigma=0.6)
from ss_conv.sp_ops.tensor import SparseTensor
from ss_conv.sp_ops.voxelize import Point2Voxel, Voxel2Point

from rotation_utils import D_from_matrix

class SS_Conv(nn.Module):
    def __init__(self, irrep_in, irrep_out, kernel_size,
        dropout_p=None,
        radial_window=radial_window, 
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
        use_bn = True,
        bn_momentum=0.01,
        bn_affine=True,
        activation=(None, None),
        activation_bias=False,
        activation_checkpoint=True,
        indice_key=None):
        super().__init__()

        Rs_in = [(m, l) for l, m in enumerate(irrep_in)]
        Rs_out = [(m, l) for l, m in enumerate(irrep_out)]
        Rs_out_with_gate = [(m, l) for l, m in enumerate(irrep_out)]

        if type(activation) is tuple:
            scalar_activation, gate_activation = activation
        else:
            scalar_activation, gate_activation = activation, activation
        n_non_scalar = sum(irrep_out[1:])
        if gate_activation is not None and n_non_scalar > 0:
            Rs_out_with_gate.append((n_non_scalar, 0))

        self.conv = Convolution(
            Rs_in=Rs_in, 
            Rs_out=Rs_out_with_gate, 
            kernel_size=kernel_size, 
            radial_window=radial_window, 
            dyn_iso=dyn_iso, 
            verbose=verbose,
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

        if use_bn:
            self.bn = BatchNorm(
                Rs_out_with_gate, 
                momentum=bn_momentum, 
                affine=bn_affine
            )
        else:
            self.bn = None

        if scalar_activation is not None or gate_activation is not None:
            self.activation = Activation(
                Rs_out,
                activation, 
                bias=activation_bias,
                checkpoint=activation_checkpoint
            )
        else:
            self.activation = None

        if dropout_p is not None and dropout_p>0:
            self.dropout = Dropout(Rs_out, p=dropout_p)
        else:
            self.dropout = None


    def forward(self, x):
        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        if self.activation is not None:
            x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class Feature_Steering_Module(nn.Module):
    def __init__(self, irrep,
        kernel_size=3,
        dropout_p=0.015,
        use_bn=True,
        origin_voxel_extent=None,
        origin_unit_voxel_extent=None,
        voxel_num_limit=[64,64,64],
        unit_voxel_extent=[0.015,0.015,0.015],
        voxelization_mode=4
    ):
        super().__init__()
        self.irrep = irrep
        if origin_voxel_extent is not None or origin_unit_voxel_extent is not None:
            self.v2p = Voxel2Point(
                voxel_extent=origin_voxel_extent,
                unit_voxel_extent=origin_unit_voxel_extent,
            )
        else:
            self.v2p = None
        self.p2v = Point2Voxel(
            voxel_num_limit=voxel_num_limit,
            unit_voxel_extent=unit_voxel_extent,
            voxelization_mode=voxelization_mode          
        )
        self.conv1 = SS_Conv(irrep, irrep, kernel_size,
            padding=kernel_size//2, 
            stride=1,
            subm=False,
            activation=(F.relu, torch.sigmoid),
            dropout_p=dropout_p,
            use_bn=use_bn)
        self.conv2 = SS_Conv(irrep, irrep, kernel_size,
            padding=kernel_size//2, 
            stride=1,
            subm=True,
            activation=(F.relu, torch.sigmoid),
            dropout_p=dropout_p,
            use_bn=use_bn)

    def forward(self, points, feats, ids, r, t, s=None):
        '''
        Param points: N*3
        Param feats: sparse voxel feat or point feat (N*C)
        Param ids: N
        Param r: B*3*3, rotation
        Param t: B*3, translation
        Param s: B or B*1 or B*3, scale
        return valid_points: M*3 (M<N)
        return valid_feats: sparse tensor 
        return valid_ids: M (M<N)
        return flag: M (M<N)
        '''

        if isinstance(feats, SparseTensor):
            assert self.v2p is not None
            if points is None:
                points, feats, ids = self.v2p(feats)
            else:
                feats = self.v2p(feats, points, ids)

        r = r[ids,:,:].contiguous().detach()
        t = t[ids, :].contiguous().detach()
        if s is not None:
            s = s[ids].contiguous().detach()

        # tranform points
        new_points = self._transform_points(points,r,t,s)

        # transform feats
        new_feats = self._transform_feats(feats,r)

        valid_points, valid_feats, valid_ids, flag = self.p2v(new_points, new_feats, ids)

        if torch.sum(flag)>0:
            valid_feats = self.conv1(valid_feats)
            valid_feats = self.conv2(valid_feats)

        return valid_points, valid_feats, valid_ids, flag

    def _transform_points(self, p, r, t, s=None):
        new_p = (p - t).unsqueeze(1) @ r
        if s is not None:
            if len(s.size()) == 1:
                s = s.unsqueeze(1)
            elif len(s.size()) == 2:
                if s.size(1) != 1:
                    s = torch.norm(s,dim=1,keepdim=True)
            new_p = new_p.squeeze(1) / (s+1e-8)
        return new_p

    def _transform_feats(self, f, r):
        new_f = f.new(f.size())
        f = f.unsqueeze(1)
        irrep = self.irrep
        dim = 0

        for l, m in enumerate(irrep):
            D = D_from_matrix(r, l).detach()
            for _ in range(m):
                new_f[:, dim:dim+(2*l+1)] = (f[:, :, dim:dim+(2*l+1)] @ D).squeeze(1)
                dim += (2*l+1)

        assert dim == f.size(2)
        return new_f.contiguous()



if __name__ == "__main__":
    from ss_conv.sp_ops.voxelize import Point2Voxel
    from ss_conv.pool import GlobalAvgPool

    Voc = Point2Voxel()
    Net = SS_Conv((3,), (1,1), 3, padding=1, activation=(F.relu, torch.sigmoid), activation_bias=True, dropout_p=0.1).cuda()
    Net = Net.train()
    Pool = GlobalAvgPool()

    p = torch.rand(2,500,3).cuda().reshape(-1,3)
    ids = torch.arange(2).unsqueeze(1).repeat(1,500).reshape(-1).cuda()

    p1,f1,flag = Voc(p,p, ids)
    f2 = Net(f1)
    f3 = Pool(f2)
    loss = torch.mean(f3)
    loss.backward()

    # Net.eval()
    # f4 = Net(f1)
    # f5 = Net(f1)





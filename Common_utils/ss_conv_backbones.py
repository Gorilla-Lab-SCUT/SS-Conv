# --------------------------------------------------------
# Sparse Steerable Convolutions.

# Sparse steerable convolutional backbones
# Written by Jiehong Lin
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import ss_conv
from ss_conv.pool import AvgPool
from ss_conv_modules import SS_Conv


class _basic_backbone(nn.Module):
    def __init__(self, irreps, strided_layers,
        kernel_size=3,
        dropout_p=0.015,
        use_bn=True,
        pool_type='avg'):
        super(_basic_backbone, self).__init__()

        module_index = 0
        modules = [[] for i in range(len(strided_layers)+1)]
        for i in range(len(irreps)-1):
            irrep_in = irreps[i]
            irrep_out= irreps[i+1]
            subm = False if (i+1) in strided_layers else True

            modules[module_index].append(
                SS_Conv(irrep_in, irrep_out, kernel_size,
                    padding=kernel_size//2, 
                    stride=1,
                    subm=subm,
                    activation=(F.relu, torch.sigmoid),
                    dropout_p=dropout_p,
                    indice_key='key_'+str(module_index),
                    use_bn=use_bn)
            )
            if (i+1) in strided_layers:
                if pool_type == 'avg':
                    modules[module_index].append(
                        AvgPool(kernel_size, stride=2,padding=kernel_size//2)
                    )
                else:
                    assert False
                module_index += 1

        self.module1 = nn.Sequential(*modules[0])
        self.module2 = nn.Sequential(*modules[1])
        self.module3 = nn.Sequential(*modules[2])
        self.module4 = nn.Sequential(*modules[3])
    
    def forward(self, inputs):
        feats1 = self.module1(inputs)
        feats2 = self.module2(feats1)
        feats3 = self.module3(feats2)
        feats4 = self.module4(feats3)

        return feats1, feats2, feats3, feats4


class Plain12(_basic_backbone):
    def __init__(self, irrep_in=(4,),
        dropout_p=0.015,
        use_bn=True,
        pool_type='avg'):
        super(Plain12, self).__init__(
            irreps=[irrep_in] + [ # input
                (1,  1,  1,  1),  # 16  1
                (2,  2,  2,  2),  # 32  2

                (2,  2,  2,  2),  # 32  3
                (2,  2,  2,  2),  # 32  4
                (2,  2,  2,  2),  # 32  5
                (4,  4,  4,  4),  # 64  6

                (4,  4,  4,  4),  # 64  7
                (4,  4,  4,  4),  # 64  8
                (4,  4,  4,  4),  # 64  9
                (8,  8,  8,  8),  # 128 10

                (8,  8,  8,  8),  # 128  11
                (16,  16,  16,  16),  # 256  12
            ],
            strided_layers=[2,6,10],
            dropout_p=dropout_p,
            use_bn=use_bn,
            pool_type=pool_type
        )


class Plain24Lite(_basic_backbone):
    def __init__(self, irrep_in=(4,),
        dropout_p=0.015,
        use_bn=True,
        pool_type='avg'):
        super(Plain24Lite, self).__init__(
            irreps=[irrep_in] + [ # input
                (2,  2,  2,  2),  # 32  1
                (2,  2,  2,  2),  # 32  2
                (2,  2,  2,  2),  # 32  3
                (4,  4,  4,  4),  # 64 4
                (4,  4,  4,  4),  # 64 5
                (4,  4,  4,  4),  # 64 6

                (4,  4,  4,  4),  # 64 7
                (4,  4,  4,  4),  # 64 8
                (4,  4,  4,  4),  # 64 9
                (4,  4,  4,  4),  # 64 10
                (4,  4,  4,  4),  # 64 11
                (4,  4,  4,  4),  # 64 12

                (4,  4,  4,  4),  # 64 13
                (4,  4,  4,  4),  # 64 14
                (4,  4,  4,  4),  # 64 15
                (8,  8,  8,  8),  # 128 16
                (8,  8,  8,  8),  # 128 17
                (8,  8,  8,  8),  # 128 18

                (8,  8,  8,  8),  # 128 19
                (8,  8,  8,  8),  # 128 20
                (8,  8,  8,  8),  # 128 21
                (8,  8,  8,  8),  # 128 22
                (8,  8,  8,  8),  # 128 23
                (8,  8,  8,  8),  # 128 24
            ],
            strided_layers=[6,12,18],
            dropout_p=dropout_p,
            use_bn=use_bn,
            pool_type=pool_type
        )


class Plain24(_basic_backbone):
    def __init__(self, irrep_in=(4,),
        dropout_p=0.015,
        use_bn=True,
        pool_type='avg'):
        super(Plain24, self).__init__(
            irreps=[irrep_in] + [ # input
                (4,  4,  4,  4),  # 64  1
                (4,  4,  4,  4),  # 64  2
                (4,  4,  4,  4),  # 64  3
                (8,  8,  8,  8),  # 128 4
                (8,  8,  8,  8),  # 128 5
                (8,  8,  8,  8),  # 128 6

                (8,  8,  8,  8),  # 128 7
                (8,  8,  8,  8),  # 128 8
                (8,  8,  8,  8),  # 128 9
                (8,  8,  8,  8),  # 128 10
                (8,  8,  8,  8),  # 128 11
                (8,  8,  8,  8),  # 128 12

                (8,  8,  8,  8),  # 128 13
                (8,  8,  8,  8),  # 128 14
                (8,  8,  8,  8),  # 128 15
                (16,  16,  16,  16),  # 256 16
                (16,  16,  16,  16),  # 256 17
                (16,  16,  16,  16),  # 256 18

                (16,  16,  16,  16),  # 256 19
                (16,  16,  16,  16),  # 256 20
                (16,  16,  16,  16),  # 256 21
                (16,  16,  16,  16),  # 256 22
                (16,  16,  16,  16),  # 256 23
                (16,  16,  16,  16),  # 256 24
            ],
            strided_layers=[6,12,18],
            dropout_p=dropout_p,
            use_bn=use_bn,
            pool_type=pool_type
        )


if __name__ == "__main__":
    from ss_conv.sp_ops.voxelize import Point2Voxel
    from ss_conv.pool import GlobalAvgPool

    Voc = Point2Voxel()
    Net = Plain24((3,)).cuda()
    Net = Net.train()
    Pool = GlobalAvgPool()

    p = torch.rand(2,500,3).cuda().reshape(-1,3)
    ids = torch.arange(2).unsqueeze(1).repeat(1,500).reshape(-1).cuda()

    p1,f1,flag = Voc(p,p, ids)
    _,_,_,f2 = Net(f1)
    f3 = Pool(f2)
    loss = torch.mean(f3)
    loss.backward()

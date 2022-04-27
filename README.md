# Sparse Steerable Convolution (SS-Conv)

Code for "Sparse Steerable Convolutions: An Efficient Learning of SE(3)-Equivariant Features for Estimation and Tracking of Object Poses in 3D Space", NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper/2021/file/8c1b6fa97c4288a4514365198566c6fa-Paper.pdf)][[Supp](https://proceedings.neurips.cc/paper/2021/file/8c1b6fa97c4288a4514365198566c6fa-Supplemental.zip)][[Arxiv](https://arxiv.org/abs/2111.07383)]

Created by Jiehong Lin, Hongyang Li, Ke Chen, Jiangbo Lu, and [Kui Jia](http://kuijia.site/).


As a basic component of SE(3)-equivariant deep feature learning, steerable convolution has recently demonstrated its advantages for 3D semantic analysis. The advantages are, however, brought by expensive computations on dense, volumetric data, which prevent its practical use for efficient processing of 3D data that are inherently sparse. In this paper, we propose a novel design of Sparse Steerable Convolution (SS-Conv) to address the shortcoming; SS-Conv greatly accelerates steerable convolution with sparse tensors, while strictly preserving the property of SE(3)-equivariance. 


![image](https://github.com/Gorilla-Lab-SCUT/SS-Conv/blob/main/doc/FigHead.png)

To verify our designs, we conduct thorough experiments on three tasks of 3D object semantic analysis, including [instance-level 6D pose estimation](https://github.com/Gorilla-Lab-SCUT/SS-Conv/tree/main/LineMOD), [category-level 6D pose and size estimation](https://github.com/Gorilla-Lab-SCUT/SS-Conv/tree/main/REAL275), and [category-level 6D pose tracking](https://github.com/Gorilla-Lab-SCUT/SS-Conv/tree/main/REAL275Tracking).

## Citation
If you find our work useful in your research, please consider citing:

    @article{lin2021sparse,
    title={Sparse Steerable Convolutions: An Efficient Learning of SE (3)-Equivariant Features for Estimation and Tracking of Object Poses in 3D Space},
    author={Lin, Jiehong and Li, Hongyang and Chen, Ke and Lu, Jiangbo and Jia, Kui},
    journal={Advances in Neural Information Processing Systems},
    volume={34},
    year={2021}
    }

## Requirements
The code has been tested with
- python 3.6.5
- pytorch 1.3.0
- CUDA 10.2


## Installation
Install our `ss_conv` lib by running the following commands:
```
cd SS_Conv_lib
python setup.py install
```

## Usage
In our `ss_conv` lib, we offer a series of layers for building the sparse steerable CNNs, including `convoluition`, `batchnorm`, `non-linearity activation`, `dropout`, and `pool`, with some operations for data-processing. One could use them by simply importing the library as follows: 

```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

import ss_conv
from ss_conv.convolution import Convolution
from ss_conv.batchnorm import BatchNorm
from ss_conv.activation import Activation
from ss_conv.dropout import Dropout
from ss_conv.pool import AvgPool, MaxPool, GlobalAvgPool, GlobalMaxPool
from ss_conv.sp_ops.voxelize import Point2Voxel, Voxel2Point
```

### Convolution

Here we provide a simple example by defining the common structure `Conv-BatchNorm-Activation-Dropout` based on SS-Conv, as a guideline of the use of `ss_conv` lib. 


```Python
class SS_Conv(nn.Module):
    def __init__(self, irrep_in, irrep_out, kernel_size,
        dropout_p=None,
        stride=1,
        padding=0,
        bias=False,
        use_bn=True,
        activation=(None, None)):
        super().__init__()

        Rs_in = [(m, l) for l, m in enumerate(irrep_in)]
        Rs_out = [(m, l) for l, m in enumerate(irrep_out)]
        Rs_out_with_gate = [(m, l) for l, m in enumerate(irrep_out)]
        # l denotes the order of the irreducible feature
        # m denotes the number of the irreducible feature of order l

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
            stride=stride,
            padding=padding,
            bias=bias)

        if use_bn:
            self.bn = BatchNorm(Rs_out_with_gate)
        else:
            self.bn = None

        if scalar_activation is not None or gate_activation is not None:
            self.activation = Activation(Rs_out, activation)
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
```
`irrep_in` and `irrep_out` are scalar lists to denote the numbers of irreducible representations of order `l=0,1,2,3,4,...`. For example, `irrep_in=(2,2,2,2)` indicates the input feature is formed by 4 kinds of irreducible representations, the orders of which are `0,1,2,3` and the numbers are `2,2,2,2`, respectively, giving a total of `2*(1+3+5+7)=32` feature channels. We could thus initialize the above module and process the features, represented as sparse tensors, as follows:


```Python
# initialize
Net = SS_Conv(irrep_in=(2,2,2,2), 
        irrep_out=(2,2,2,2), 
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        use_bn=True,
        activation=(F.relu, torch.sigmoid),
        dropout_p=0.01).cuda()

# forward
output = Net(sparse_tensor)
```

### Pool
We also provide common pooling oprations to reduce feature spatial sizes, including `average pooling`, `max pooling`, `global average pooling`, and `global max pooling`. 

```Python
# average pooling
avgpool = AvgPool(kernel_size=3, stride=2, padding=1)
output = avgpool(sparse_tensor)         # sparse_tensor.dense(): B*C*V*V*V
                                        # output.dense(): B*C*(V/2)*(V/2)*(V/2)

# max pooling
maxpool = MaxPool(kernel_size=3, stride=2, padding=1)
output = maxpool(sparse_tensor)         # sparse_tensor.dense(): B*C*V*V*V
                                        # output.dense(): B*C*(V/2)*(V/2)*(V/2)

# global average pooling
global_avgpool = GlobalAvgPool()
output = global_avgpool(sparse_tensor)  # sparse_tensor.dense(): B*C*V*V*V
                                        # output: B*C

# global max pooling
global_maxpool = GlobalMaxPool()
output = global_maxpool(sparse_tensor)  # sparse_tensor.dense(): B*C*V*V*V
                                        # output: B*C
```


### Data Processing
We offer `Point2Voxel` and `Voxel2Point` operations for transformations between features of point clouds and voxels (represented as sparse tensors). 

```Python
p = torch.rand(B,N,3).cuda()              # a mini-batch of point clouds: B*N*3
f = torch.rand(B,N,C).cuda()              # a mini-batch of point features: B*N*C
ids = torch.arange(B).unsqueeze(1).repeat(1,N).cuda()

# Point2Voxel
p2v = Point2Voxel(
    voxel_num_limit=[64,64,64],           # spatial size: V*V*V=64*64*64
    unit_voxel_extent=[0.03,0.03,0.03],   # area of each voxel grid: 0.03*0.03*0.03
)                                         # area of the whole voxel: 1.92*1.92*1.92
                                          # valid point coordinates: x,y,z in [-0.96, 0.96]

p = p.reshape(B*N, 3)
f = f.reshape(B*N, C)
ids = ids.reshape(B*N)
valid_p, sparse_tensor, valid_ids, _ = p2v(p,f,ids)   # valid_p: M*3(M<=B*N), points within the voxel
                                                      # sparse_tensor.dense(): B*C*V*V*V=B*C*64*64*64
                                                      # valid_ids: M

# Voxel2Point
v2p = Voxel2Point(unit_voxel_extent=[0.03,0.03,0.03]) # or v2p = Voxel2Point(voxel_extent=[1.92,1.92,1.92])
valid_f = v2p(sparse_tensor, valid_p, valid_ids)      # valid_f: M*C
```




## Applications

- Instance-level 6D Pose Estimation [[LineMOD](https://github.com/Gorilla-Lab-SCUT/SS-Conv/tree/main/LineMOD)]
- Category-level 6D Pose and Size Estimation [[REAL275](https://github.com/Gorilla-Lab-SCUT/SS-Conv/tree/main/REAL275)]
- Category-level 6D Pose Tracking [[REAL275](https://github.com/Gorilla-Lab-SCUT/SS-Conv/tree/main/REAL275Tracking)]




## Contact
`lin.jiehong@mail.scut.edu.cn`

`eeli.hongyang@mail.scut.edu.cn`

## Acknowledgements

Our implementation leverages the code from [ST-Conv](https://github.com/tscohen/se3cnn ), [SP-Conv](https://github.com/traveller59/spconv/tree/v1.1) and [PointGroup](https://github.com/dvlab-research/PointGroup).

## License
Our code is released under MIT License (see LICENSE file for details).



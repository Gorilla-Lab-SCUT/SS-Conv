# Sparse Steerable Convolution (SS-Conv)
Code for "Sparse Steerable Convolutions: An Efficient Learning of SE(3)-Equivariant Features for Estimation and Tracking of Object Poses in 3D Space", NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper/2021/file/8c1b6fa97c4288a4514365198566c6fa-Paper.pdf)][[Supp](https://proceedings.neurips.cc/paper/2021/file/8c1b6fa97c4288a4514365198566c6fa-Supplemental.zip)][[Arxiv](https://arxiv.org/abs/2111.07383)]

Created by Jiehong Lin, Hongyang Li, Ke Chen, Jiangbo Lu, and [Kui Jia](http://kuijia.site/).


As a basic component of SE(3)-equivariant deep feature learning, steerable convolution has recently demonstrated its advantages for 3D semantic analysis. The advantages are, however, brought by expensive computations on dense, volumetric data, which prevent its practical use for efficient processing of 3D data that are inherently sparse. In this paper, we propose a novel design of Sparse Steerable Convolution (SS-Conv) to address the shortcoming; SS-Conv greatly accelerates steerable convolution with sparse tensors, while strictly preserving the property of SE(3)-equivariance. 


![image](https://github.com/Gorilla-Lab-SCUT/SS-Conv/blob/main/doc/FigHead.png)

To verify our designs, we conduct thorough experiments on three tasks of 3D object semantic analysis, including instance-level
6D pose estimation, category-level 6D pose and size estimation, and categorylevel 6D pose tracking.

## Citation
If you find our work useful in your research, please consider citing:

    @article{lin2021sparse,
    title={Sparse Steerable Convolutions: An Efficient Learning of SE (3)-Equivariant Features for Estimation and Tracking of Object Poses in 3D Space},
    author={Lin, Jiehong and Li, Hongyang and Chen, Ke and Lu, Jiangbo and Jia, Kui},
    journal={Advances in Neural Information Processing Systems},
    volume={34},
    year={2021}
    }

# Application: Instance-level 6D Pose Estimation on LineMOD
Code for "Sparse Steerable Convolutions: An Efficient Learning of SE(3)-Equivariant Features for Estimation and Tracking of Object Poses in 3D Space", NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper/2021/file/8c1b6fa97c4288a4514365198566c6fa-Paper.pdf)][[Supp](https://proceedings.neurips.cc/paper/2021/file/8c1b6fa97c4288a4514365198566c6fa-Supplemental.zip)][[Arxiv](https://arxiv.org/abs/2111.07383)]

Created by Jiehong Lin, Hongyang Li, Ke Chen, Jiangbo Lu, and Kui Jia.

![image](https://github.com/Gorilla-Lab-SCUT/SS-Conv/blob/main/doc/FigNetwork.png)

## Requirements
The code has been tested with
- python 3.6.5
- pytorch 1.3.0
- CUDA 10.2

Some python dependent packages：
- [ss_conv](https://github.com/Gorilla-Lab-SCUT/SS-Conv)
- [gorilla 0.2.6.5](https://github.com/Gorilla-Lab-SCUT/gorilla-core) (`pip install gorilla-core`)


## Downloads
- [Dataset](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7), provided by [DenseFusion](https://github.com/j96w/DenseFusion).
- [Models](https://drive.google.com/file/d/1jqrHJO7-8h3LXpPEFoRJ4QylBMUAY9CX/view?usp=sharing) for evaluation, including a simple single-stage model (`tiny_model.py`) used for ablation studies and a two-stage model (`model.py`).  

## Usage

#### Training

```
python train.py --gpu 0 --data YOUR_DATA_ROOT/Linemod_preprocessed --model model 
```

#### Evaluation
```
python test.py --gpu 0 --data YOUR_DATA_ROOT/Linemod_preprocessed --model model
```

## Acknowledgements

Our implementation leverages the code from [DenseFusion](https://github.com/j96w/DenseFusion).
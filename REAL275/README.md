# Application: Category-level 6D Pose and Size Estimation on REAL275
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
- Segmentation predictions [[link](https://drive.google.com/file/d/1RwAbFWw2ITX9mXzLUEBjPy_g-MNdyHET/view?usp=sharing)]
- Trained models and pose predcitions [[link](https://drive.google.com/file/d/1i_F2_7qO5hTKnV8FLDHeYIY0q1H4YD4q/view?usp=sharing)]

## Evaluation
We provide two sets of results based on two different backbones (c.f. `model_lite.py` and `model.py`) for evaluation:
```
python evaluation.py
```

#### Results
|  | niter | IoU25 | IoU75 | 5 degree 2 cm | 5 degree 5 cm | 10 degree 2 cm | 10 degree 5 cm |
|---|---|---|---|---|---|---|---|
| model_lite | 0 | 79.5 | 63.5 | 32.9 | 40.8 | 51.0 | 64.7 |
| model_lite | 1 | 79.2 | 70.5 | 45.8 | 53.1 | 62.2 | 71.8 |
| model_lite | 2 | 79.2 | 71.7 | 47.8 | 55.1 | 65.4 | 74.8 |
| model | 0 | 79.3 | 65.6 | 32.7 | 39.8 | 53.7 | 67.2 |
| model | 1 | 79.2 | 70.2 | 43.9 | 50.3 | 65.0 | 74.0 |
| model | 2 | 78.9 | 71.0 | 46.3 | 53.0 | 67.4 | 76.4 |


## Usage

#### Data Processing

Download the data provided by [NOCS](https://github.com/hughw19/NOCS_CVPR2019) ([real_train](http://download.cs.stanford.edu/orion/nocs/real_train.zip), [real_test](http://download.cs.stanford.edu/orion/nocs/real_test.zip),
[ground truths](http://download.cs.stanford.edu/orion/nocs/gts.zip),
and [mesh models](http://download.cs.stanford.edu/orion/nocs/obj_models.zip)), and unzip them in your data folder as follows:

```
YOUR_DATA_ROOT
└── data
    ├── CAMERA
    │   ├── train
    │   └── val
    ├── Real
    │   ├── train
    │   └── test
    ├── gts
    │   ├── val
    │   └── real_test
    └── obj_models
        ├── train
        ├── val
        ├── real_train
        └── real_test
```

Revise the paprameter `DATA_DIR` in `data_processing.py`, e.g., `DATA_DIR=/YOUR_DATA_ROOT/data`, and run the following scripts to prepare the dataset:

```
python data_processing.py
```

#### Training

```
python train.py --gpu 0 --data YOUR_DATA_ROOT/data --model model 
```

#### Test
```
python test.py --gpu 0 --data YOUR_DATA_ROOT/data --segmentation YOUR_SEG_ROOT/segmentation_results/REAL275 --model model --niter 2
```

## Acknowledgements

Our implementation leverages the code from [NOCS](https://github.com/hughw19/NOCS_CVPR2019), [DualPoseNet](https://github.com/Gorilla-Lab-SCUT/DualPoseNet), and [SPD](https://github.com/mentian/object-deformnet).

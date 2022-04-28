# --------------------------------------------------------
# Sparse Steerable Convolutions

# Common metric utils
# Written by Jiehong Lin
# --------------------------------------------------------

import torch
import torch.nn as nn


def L2_Distance(p1, p2):
    '''
    p1: float B*N*3
    p2: float B*N*3
    return dis: float B*N 
    '''
    return torch.norm(p1 - p2, dim=2)


def Chamfer_Distance(p1, p2):
    '''
    p1: float B*N*3
    p2: float B*N*3
    return dis: float B*N 
    '''
    dis = torch.norm(p1.unsqueeze(2) - p2.unsqueeze(1), dim=3)
    dis1 = torch.min(dis, 2)[0]
    dis2 = torch.min(dis, 1)[0]
    return 0.5*(dis1+dis2)

# --------------------------------------------------------
# Sparse Steerable Convolutions

# A tiny model based on Plain12 for 6D pose estimation
# Written by Jiehong Lin
# --------------------------------------------------------

import torch
import torch.nn as nn

import ss_conv
from ss_conv.pool import GlobalAvgPool
from ss_conv.sp_ops.voxelize import Point2Voxel
from ss_conv_backbones import Plain12

from rotation_utils import normalize_vector, compute_rotation_matrix_from_ortho6d
from metric_utils import L2_Distance, Chamfer_Distance

import ipdb

class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.voxelization = Point2Voxel(
            voxel_num_limit=cfg.voxel_num_limit,
            unit_voxel_extent=cfg.unit_voxel_extent,
            voxelization_mode=cfg.voxelization_mode          
        )
        self.feat_extractor = Plain12(
            irrep_in=(4,),
            dropout_p=cfg.dropout_p,
            use_bn=cfg.use_bn,
            pool_type=cfg.pool_type
        )
        self.global_pool = GlobalAvgPool()
        self.trans_module = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        self.rot_module = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, inputs):
        '''
        inputs: list, containing:
            param input_points: float B*N*3, input point clouds
            param input_feats: float B*N*3, input rgb values
            param model_points: float B*M*3, CAD point clouds
            param target_points: float B*M*3, CAD point clouds transformed by RT
            param sym_flag: float B, 1-symmetric obj, 0-asymmetric obj
            param obj_index: int B, [0, 12], object indexes
        '''

        input_points = inputs['input_points']
        input_feats = inputs['input_feats']
        model_points = inputs['model_points']
        B,N,_ = input_points.size()
        device = input_points.device

        occupied_tensor = torch.ones(B,N,1).float().to(device)
        input_points = input_points.reshape(B*N,3)
        input_feats = torch.cat([occupied_tensor, input_feats], 2).reshape(B*N,4)
        batch_ids = torch.arange(B).unsqueeze(1).repeat(1,N).to(device).reshape(B*N)
        _,input_feats,_,_ = self.voxelization(input_points, input_feats, batch_ids)
        feat_lists = self.feat_extractor(input_feats)

        feats = feat_lists[-1]
        feats = self.global_pool(feats)
        pred_trans = self.trans_module(feats)
        pred_rot_6d = self.rot_module(feats)

        pred_rot_x = normalize_vector(pred_rot_6d[:, :3])
        pred_rot_y = normalize_vector(pred_rot_6d[:, 3:])
        pred_rot = compute_rotation_matrix_from_ortho6d(pred_rot_x, pred_rot_y)

        pred_points = model_points @ pred_rot.transpose(1,2) + pred_trans.unsqueeze(1)

        if self.training:
            end_points = {
                'pred_points': pred_points,
                'target_points': inputs['target_points'],
                'sym_flag': inputs['sym_flag']
            }
        else:
            end_points = {
                'pred_points': pred_points,
                'target_points': inputs['target_points'],
                'sym_flag': inputs['sym_flag'],
                'obj_index': inputs['obj_index']
            }
        return end_points


class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()

    def forward(self, end_points):
        pred_points = end_points['pred_points']
        target_points = end_points['target_points']
        sym_flag = end_points['sym_flag']

        sym_dis = Chamfer_Distance(pred_points, target_points).mean(1)
        asym_dis = L2_Distance(pred_points, target_points).mean(1)
        all_loss = sym_flag * sym_dis + (1-sym_flag) * asym_dis

        losses = {
            "all_loss": all_loss.mean(),
        }
        return losses

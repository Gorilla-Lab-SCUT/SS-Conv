# --------------------------------------------------------
# Sparse Steerable Convolutions

# Model based on Plain24 for 6D pose estimation
# Written by Jiehong Lin
# --------------------------------------------------------
import numpy as np

import torch
import torch.nn as nn

import ss_conv
from ss_conv.pool import GlobalAvgPool
from ss_conv.sp_ops.voxelize import Point2Voxel, Voxel2Point
from ss_conv_backbones import Plain24
from ss_conv_modules import Feature_Steering_Module

from rotation_utils import normalize_vector, compute_rotation_matrix_from_ortho6d
from metric_utils import L2_Distance, Chamfer_Distance


class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.p2v = Point2Voxel(
            voxel_num_limit=cfg.voxel_num_limit,
            unit_voxel_extent=cfg.unit_voxel_extent,
            voxelization_mode=cfg.voxelization_mode          
        )
        self.v2p = Voxel2Point(
            voxel_extent=np.array(cfg.unit_voxel_extent).astype(np.float) \
                * np.array(cfg.voxel_num_limit).astype(np.float)
        )
        self.feat_extractor = Plain24(
            irrep_in=(4,),
            dropout_p=cfg.dropout_p,
            use_bn=cfg.use_bn,
            pool_type=cfg.pool_type
        )
        self.feature_steering = FeatureSteering(
            unit_voxel_extent=cfg.unit_voxel_extent,
            voxel_num_limit=cfg.voxel_num_limit,
            voxelization_mode=cfg.voxelization_mode,
            scales=[2,4,8,16],
        )
        self.pose_estimator1 = PoseEstimator()
        self.pose_estimator2 = PoseEstimator()


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
        points1,input_feats,ids1,flag1 = self.p2v(input_points, input_feats, batch_ids)

        # stage 1
        voxel_feats1 = self.feat_extractor(input_feats)
        point_feats1 = self.voxels2point(points1, voxel_feats1, ids1)
        r1, t1 = self.pose_estimator1(points1, torch.cat(point_feats1,dim=1), ids1)

        if self.training:
            # stage 2
            points2,voxel_feats2,ids2,flag2 = self.feature_steering(points1,point_feats1,ids1, r1, t1)
            point_feats2 = self.voxels2point(points2, voxel_feats2, ids2)
            r2, t2 = self.pose_estimator2(points2, torch.cat(point_feats2,dim=1), ids2)

            flag2 = flag2 * flag1
            r2 = r1.detach() @ r2
            t2 = t1.detach() + (r1.detach() @ t2.unsqueeze(2)).squeeze(2)

            # outputs
            pred_points1 = model_points @ r1.transpose(1,2) + t1.unsqueeze(1)
            pred_points2 = model_points @ r2.transpose(1,2) + t2.unsqueeze(1)

            end_points = {
                'pred_points1': pred_points1,
                'pred_points2': pred_points2,
                'flag1': flag1,
                'flag2': flag2,
                'target_points': inputs['target_points'],
                'sym_flag': inputs['sym_flag'],          
            }

        else:
            # refinement
            if not hasattr(self.cfg, 'niter'):
                self.cfg.niter = 1

            r = r1
            t = t1

            for _ in range(self.cfg.niter):
                points2,voxel_feats2,ids2,flag2 = self.feature_steering(points1,point_feats1,ids1,r,t)
                point_feats2 = self.voxels2point(points2, voxel_feats2, ids2)
                r2, t2 = self.pose_estimator2(points2, torch.cat(point_feats2,dim=1), ids2)

                r2 = r.detach() @ r2
                t2 = t.detach() + (r.detach() @ t2.unsqueeze(2)).squeeze(2)

                # update
                r = (1-flag2.reshape(-1,1,1)) * r + flag2.reshape(-1,1,1) * r2
                t = (1-flag2.reshape(-1,1)) * t + flag2.reshape(-1,1) * t2


            pred_points = model_points @ r.transpose(1,2) + t.unsqueeze(1)
            end_points = {
                'pred_points': pred_points,
                'target_points': inputs['target_points'],
                'sym_flag': inputs['sym_flag'],
                'obj_index': inputs['obj_index']
            }

        return end_points

    def voxels2point(self, points, voxel_feats, ids):
        point_feats = []
        for voxel_feat in voxel_feats:
            point_feats.append(self.v2p(voxel_feat, points, ids))
        return point_feats


class FeatureSteering(nn.Module):
    def __init__(self,
        unit_voxel_extent,
        voxel_num_limit,
        voxelization_mode,
        scales,
    ):
        super().__init__()
        irreps = [
            (8,8,8,8),
            (8,8,8,8),
            (16,16,16,16),
            (16,16,16,16),
        ]

        modules = []
        for k, scale in enumerate(scales):
            scaled_unit_voxel_extent = [m*scale for m in unit_voxel_extent]
            scaled_voxel_num_limit = [int(m/scale) for m in voxel_num_limit]

            modules.append(
                Feature_Steering_Module(
                    irreps[k],
                    voxel_num_limit = scaled_voxel_num_limit,
                    unit_voxel_extent = scaled_unit_voxel_extent,
                    voxelization_mode = voxelization_mode
                )
            )

        self.fs1 = modules[0]
        self.fs2 = modules[1]
        self.fs3 = modules[2]
        self.fs4 = modules[3]
 
    def forward(self, p, f, ids, r, t):
        valid_points, valid_feats1, valid_ids, flag = self.fs1(p, f[0], ids, r, t)
        _, valid_feats2, _, _ = self.fs2(p, f[1], ids, r, t)
        _, valid_feats3, _, _ = self.fs3(p, f[2], ids, r, t)
        _, valid_feats4, _, _ = self.fs4(p, f[3], ids, r, t)
        valid_feats = [valid_feats1, valid_feats2, valid_feats3, valid_feats4]

        return valid_points, valid_feats, valid_ids, flag


class PoseEstimator(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.trans_module = nn.Sequential(
            nn.Linear(dim+3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        self.rot_module = nn.Sequential(
            nn.Linear(dim+3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        self.pool = GlobalAvgPool()

    def forward(self, points, feats, ids):
        offsets = self._get_offsets(ids)

        feats = torch.cat([points, feats], 1)
        pred_trans = self.trans_module(feats) + points
        pred_rot_6d = self.rot_module(feats)
        pred_rot_x = normalize_vector(pred_rot_6d[:, :3])
        pred_rot_y = normalize_vector(pred_rot_6d[:, 3:])

        pred_trans = self.pool(pred_trans, offsets)
        pred_rot_x = self.pool(pred_rot_x, offsets)
        pred_rot_y = self.pool(pred_rot_y, offsets)
        pred_rot = compute_rotation_matrix_from_ortho6d(pred_rot_x, pred_rot_y)

        return pred_rot, pred_trans

    def _get_offsets(self, ids):
        offsets = [0]
        unique_ids = torch.unique(ids)
        for i in unique_ids:
            is_i = (ids==i).sum()
            offsets.append(is_i.item()+offsets[-1])
        offsets = torch.tensor(offsets).int().to(ids.device).detach()
        return offsets


class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()

    def forward(self, end_points):
        pred_points1 = end_points['pred_points1']
        pred_points2 = end_points['pred_points2']
        flag1 = end_points['flag1']
        flag2 = end_points['flag2']
        target_points = end_points['target_points']
        sym_flag = end_points['sym_flag']

        sym_dis1 = Chamfer_Distance(pred_points1, target_points).mean(1)
        asym_dis1 = L2_Distance(pred_points1, target_points).mean(1)
        loss1 = flag1 * (sym_flag * sym_dis1 + (1-sym_flag) * asym_dis1)

        if torch.sum(flag2)>0:
            sym_dis2 = Chamfer_Distance(pred_points2, target_points).mean(1)
            asym_dis2 = L2_Distance(pred_points2, target_points).mean(1)
            loss2 = flag2 * (sym_flag * sym_dis2 + (1-sym_flag) * asym_dis2)
        else:
            loss2 = loss1.new(loss1.size()).zero_()

        all_loss = loss1 + loss2

        losses = {
            "all_loss": all_loss.mean(),
            "stage1": loss1.mean(),
            "stage2": loss2.mean(), 
        }
        return losses
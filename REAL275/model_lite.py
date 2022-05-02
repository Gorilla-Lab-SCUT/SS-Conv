# --------------------------------------------------------
# Sparse Steerable Convolutions

# Model based on Plain24 for category-level 6D pose estimation
# Written by Jiehong Lin
# --------------------------------------------------------

import numpy as np

import torch
import torch.nn as nn

import ss_conv
from ss_conv.pool import GlobalAvgPool
from ss_conv.sp_ops.voxelize import Point2Voxel, Voxel2Point
from ss_conv_backbones import Plain24Lite
from ss_conv_modules import Feature_Steering_Module

from rotation_utils import normalize_vector, compute_rotation_matrix_from_ortho6d

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
        self.feat_extractor = Plain24Lite(
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
        self.pose_estimator1 = PoseEstimator(size_dim=3)
        self.pose_estimator2 = PoseEstimator(size_dim=3)
        self.pool = GlobalAvgPool()


    def forward(self, inputs):
        '''
        inputs: list, containing:
            param pts: float B*N*3, input point clouds
            param rgb: float B*N*3, input rgb values

            during training:
            param translation_label: float B*3
            param rotation_label: float B*3*3
            param size_label: float B*3
        '''
        input_points = inputs['pts']
        input_feats = inputs['rgb']
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
        r1, t1, s1, nocs1 = self.pose_estimator1(points1, torch.cat(point_feats1,dim=1), ids1)

        if self.training:
            # stage 2
            points2,voxel_feats2,ids2,flag2 = self.feature_steering(points1,point_feats1,ids1,r1,t1,s1)
            point_feats2 = self.voxels2point(points2, voxel_feats2, ids2)
            r2, t2, s2, nocs2 = self.pose_estimator2(points2, torch.cat(point_feats2,dim=1), ids2)

            flag2 = flag2 * flag1
            scale = torch.norm(s1, dim=1, keepdim=True).detach()
            r2 = r1.detach() @ r2
            t2 = t1.detach() + (r1.detach() @ t2.unsqueeze(2)).squeeze(2) * scale
            s2 = s2 * scale

            nocs_dis1 = self._get_nocs_dis(nocs1, points1, ids1, inputs)
            nocs_dis2 = self._get_nocs_dis(nocs2, points2, ids2, inputs, r1, t1, s1)

            end_points = {
                'r1': r1,
                't1': t1,
                's1': s1,
                'nocs_dis1': nocs_dis1.squeeze(1),
                'flag1': flag1,

                'r2': r2,
                't2': t2,
                's2': s2,
                'nocs_dis2': nocs_dis2.squeeze(1),
                'flag2': flag2,

                'gt_r': inputs['rotation_label'],
                'gt_t': inputs['translation_label'],
                'gt_s': inputs['size_label'],
            }

        else:
            # refinement
            if not hasattr(self.cfg, 'niter'):
                self.cfg.niter = 1

            r = r1
            t = t1
            s = s1
            
            for _ in range(self.cfg.niter):
                points2,voxel_feats2,ids2,flag2 = self.feature_steering(points1,point_feats1,ids1,r,t,s)
                point_feats2 = self.voxels2point(points2, voxel_feats2, ids2)
                r2, t2, s2, _ = self.pose_estimator2(points2, torch.cat(point_feats2,dim=1), ids2)
                
                flag2 = flag2 * flag1

                scale = torch.norm(s, dim=1, keepdim=True)
                r2 = r.detach() @ r2
                t2 = t.detach() + (r.detach() @ t2.unsqueeze(2)).squeeze(2) * scale
                s2 = s2 * scale

                # update
                r = (1-flag2.reshape(-1,1,1)) * r + flag2.reshape(-1,1,1) * r2
                t = (1-flag2.reshape(-1,1)) * t + flag2.reshape(-1,1) * t2
                s = (1-flag2.reshape(-1,1)) * s + flag2.reshape(-1,1) * s2

            end_points = {
                'rotation': r, 
                'translation': t,
                'size': s,
            }

        return end_points

    def voxels2point(self, points, voxel_feats, ids):
        point_feats = []
        for voxel_feat in voxel_feats:
            point_feats.append(self.v2p(voxel_feat, points, ids))
        return point_feats

    def _get_nocs_dis(self, nocs, p, ids, list, r=None, t=None, s=None):

        def _get_offsets(ids):
            offsets = [0]
            unique_ids = torch.unique(ids)
            for i in unique_ids:
                is_i = (ids==i).sum()
                offsets.append(is_i.item()+offsets[-1])
            offsets = torch.tensor(offsets).int().to(ids.device).detach()
            return offsets

        if r is not None:
            r_ = r[ids, :, :].contiguous().detach()
            p = (p.unsqueeze(1)@(r_.transpose(1,2))).squeeze(1)
        if s is not None:
            s_ = s[ids, :].contiguous().detach()
            p = p*torch.norm(s_,dim=1,keepdim=True)
        if t is not None:
            t_ = t[ids, :].contiguous().detach()
            p = p+t_

        r_ = list['rotation_label'][ids]
        t_ = list['translation_label'][ids]
        s_ = list['size_label'][ids]

        gt_nocs = ((p-t_).unsqueeze(1)@r_).squeeze(1) / torch.norm(s_,dim=1,keepdim=True)
        dis = torch.norm(nocs-gt_nocs.detach(), dim=1, keepdim=True)
        dis = self.pool(dis, _get_offsets(ids))

        return dis


class FeatureSteering(nn.Module):
    def __init__(self,
        unit_voxel_extent,
        voxel_num_limit,
        voxelization_mode,
        scales,
    ):
        super().__init__()
        irreps = [
            (4,4,4,4),
            (4,4,4,4),
            (8,8,8,8),
            (8,8,8,8),
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
 
    def forward(self, p, f, ids, r, t, s):
        valid_points, valid_feats1, valid_ids, flag = self.fs1(p, f[0], ids, r, t, s)
        _, valid_feats2, _, _ = self.fs2(p, f[1], ids, r, t, s)
        _, valid_feats3, _, _ = self.fs3(p, f[2], ids, r, t, s)
        _, valid_feats4, _, _ = self.fs4(p, f[3], ids, r, t, s)
        valid_feats = [valid_feats1, valid_feats2, valid_feats3, valid_feats4]

        return valid_points, valid_feats, valid_ids, flag


class PoseEstimator(nn.Module):
    def __init__(self, input_dim=384, size_dim=3):
        super().__init__()
        self.trans_module = nn.Sequential(
            nn.Linear(input_dim+3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        self.rot_module = nn.Sequential(
            nn.Linear(input_dim+3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        self.size_module = nn.Sequential(
            nn.Linear(input_dim+3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, size_dim)
        )
        self.nocs_module = nn.Sequential(
            nn.Linear(input_dim+3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        self.pool = GlobalAvgPool()

    def forward(self, points, feats, ids):
        offsets = self._get_offsets(ids)

        feats = torch.cat([points, feats], 1)
        pred_trans = self.trans_module(feats) + points
        pred_rot_6d = self.rot_module(feats)
        pred_rot_x = normalize_vector(pred_rot_6d[:, :3])
        pred_rot_y = normalize_vector(pred_rot_6d[:, 3:])
        pred_size = self.size_module(feats)
        pred_nocs = self.nocs_module(feats)

        pred_trans = self.pool(pred_trans, offsets)
        pred_rot_x = self.pool(pred_rot_x, offsets)
        pred_rot_y = self.pool(pred_rot_y, offsets)
        pred_rot = compute_rotation_matrix_from_ortho6d(pred_rot_x, pred_rot_y)
        pred_size = self.pool(pred_size, offsets)

        return pred_rot, pred_trans, pred_size, pred_nocs

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
        self.rw = cfg.rotation_weight
        self.tw = cfg.trans_weight
        self.sw = cfg.size_weight
        self.nw = cfg.nocs_weight

    def forward(self, end_points):
        flag1 = end_points['flag1'].detach()
        r1_loss = flag1*torch.norm(end_points['r1']-end_points['gt_r'], dim=1).mean(1)
        t1_loss = flag1*torch.norm(end_points['t1']-end_points['gt_t'], dim=1)
        s1_loss = flag1*torch.norm(end_points['s1']-end_points['gt_s'], dim=1)
        n1_loss = flag1*end_points['nocs_dis1']

        r1_loss = r1_loss.sum() / flag1.sum()
        t1_loss = t1_loss.sum() / flag1.sum()
        s1_loss = s1_loss.sum() / flag1.sum()
        n1_loss = n1_loss.sum() / flag1.sum()
        loss1 = self.rw*r1_loss + self.tw*t1_loss + self.sw*s1_loss + self.nw*n1_loss

        flag2 = end_points['flag2'].detach()
        if torch.sum(flag2)>0:
            r2_loss = flag2*torch.norm(end_points['r2']-end_points['gt_r'], dim=1).mean(1)
            t2_loss = flag2*torch.norm(end_points['t2']-end_points['gt_t'], dim=1)
            s2_loss = flag2*torch.norm(end_points['s2']-end_points['gt_s'], dim=1)
            n2_loss = flag2*end_points['nocs_dis2']

            r2_loss = r2_loss.sum() / flag2.sum()
            t2_loss = t2_loss.sum() / flag2.sum()
            s2_loss = s2_loss.sum() / flag2.sum()
            n2_loss = n2_loss.sum() / flag2.sum()
        else:
            r2_loss = r1_loss.new(r1_loss.size()).zero_()
            t2_loss = t1_loss.new(t1_loss.size()).zero_()
            s2_loss = s1_loss.new(s1_loss.size()).zero_()
            n2_loss = n1_loss.new(n1_loss.size()).zero_()
        loss2 = self.rw*r2_loss + self.tw*t2_loss + self.sw*s2_loss + self.nw*n2_loss

        losses = {
            "all_loss": loss1+loss2,
            "stage1": loss1,
            "stage2": loss2,
            # "r1": r1_loss,
            # "r2": r2_loss,
            # "t1": t1_loss,
            # "t2": t2_loss,
            # "s1": s1_loss,
            # "s2": s2_loss,
            # "n1": n1_loss,
            # "n2": n2_loss,
        }
        return losses
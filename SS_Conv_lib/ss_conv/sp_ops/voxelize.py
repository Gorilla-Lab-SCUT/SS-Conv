# --------------------------------------------------------
# Sparse Steerable Convolution Lib.

# Functions on Sparse Tensors
# Written by Hongyang Li and Jiehong Lin
# Modified from https://github.com/traveller59/spconv/tree/v1.1
# and https://github.com/dvlab-research/PointGroup
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn

from .tensor import SparseTensor
from .functional import voxelization, voxel2point_mapping, three_nn, three_interpolate


class Point2Voxel(nn.Module):
    def __init__(self,
        voxel_num_limit=[64,64,64],
        unit_voxel_extent=[0.05,0.05,0.05],
        voxelization_mode=4
    ):
        super().__init__()
        self.unit_voxel_extent = torch.FloatTensor(np.array(unit_voxel_extent).astype(np.float))
        self.voxel_num_limit = np.array(voxel_num_limit)
        self.voxelization_mode = voxelization_mode

    def forward(self, point_cloud, feature, batch_ids):
        '''
        point_cloud: N*3
        feature: N*C
        batch_ids: N
        '''
        assert len(point_cloud.size())==2
        N = point_cloud.size(0)
        C = feature.size(1)
        device = point_cloud.device
        
        unique_origin_ids = torch.unique(batch_ids)
        B1 = unique_origin_ids.size(0)

        unit_voxel_extent = self.unit_voxel_extent.reshape(1,3).to(device)
        voxel_num_limit = torch.FloatTensor(self.voxel_num_limit.astype(np.float)).reshape(1,3).to(device)
        total_voxel_extent = voxel_num_limit * unit_voxel_extent
        half_voxel_extent = 0.5 * total_voxel_extent

        # filtering
        valid_flag = (torch.abs(point_cloud) > half_voxel_extent)
        valid_flag = (valid_flag.sum(1) == 0)

        if valid_flag.sum() == 0:
            valid_instance_flag = torch.zeros(B1).float().to(device)
            valid_point_cloud = torch.zeros(B1, 3).float().to(device)
            valid_feature = torch.zeros(B1, C).float().to(device)
            valid_ids = unique_origin_ids
    
        else:
            valid_point_cloud = point_cloud[valid_flag].contiguous()
            valid_feature = feature[valid_flag].contiguous()
            valid_ids = batch_ids[valid_flag].contiguous()

            unique_valid_ids = torch.unique(valid_ids)
            B2 = unique_valid_ids.size(0)

            if B1 == B2:
                valid_instance_flag = torch.ones(B1).float().to(device)
            else:
                unique_origin_ids_ = unique_origin_ids.unsqueeze(1).repeat(1, B2)
                unique_valid_ids_ = unique_valid_ids.unsqueeze(0).repeat(B1, 1)
                valid_instance_flag = (unique_origin_ids_ == unique_valid_ids_).sum(1).float()

                invalid_ids = unique_origin_ids[valid_instance_flag==0].contiguous()
                assert len(invalid_ids) == B1-B2

                valid_point_cloud = torch.cat([valid_point_cloud, torch.zeros(B1-B2, 3).float().to(device)], dim=0)
                valid_feature = torch.cat([valid_feature, torch.zeros(B1-B2, C).float().to(device)], dim=0)
                valid_ids = torch.cat([valid_ids, invalid_ids], dim=0)

            index = torch.sort(valid_ids)[1]
            valid_point_cloud = valid_point_cloud[index].contiguous()
            valid_feature = valid_feature[index].contiguous()
            valid_ids = valid_ids[index].contiguous()         
        
        voxel_index = (valid_point_cloud + half_voxel_extent) / unit_voxel_extent
        voxel_index = torch.cat([valid_ids.unsqueeze(1).int(), voxel_index.int()], dim=1).detach()

        occupied_index, inverse, count = torch.unique(voxel_index, return_inverse=True, return_counts=True, dim=0)
        nActive = occupied_index.size(0)
        if self.voxelization_mode == 1 or self.voxelization_mode == 2:
            maxActive = 1
        else:
            maxActive = torch.max(count).item()
        v2p_maps = voxel2point_mapping(inverse, nActive, maxActive, self.voxelization_mode)

        sparse_feat = voxelization(valid_feature, v2p_maps.detach(), self.voxelization_mode)
        sparse_feat = SparseTensor(sparse_feat, occupied_index.detach(), self.voxel_num_limit.astype(np.int), B1)

        return valid_point_cloud, sparse_feat, valid_ids, valid_instance_flag


class Voxel2Point(nn.Module):
    def __init__(self, voxel_extent=None, unit_voxel_extent=None):
        super().__init__()
        assert voxel_extent is not None or unit_voxel_extent is not None
        self.voxel_extent = np.array(voxel_extent).astype(np.float) if voxel_extent is not None else None
        self.unit_voxel_extent = np.array(unit_voxel_extent).astype(np.float) if unit_voxel_extent is not None else None


    def forward(self, sparse_feat, point_cloud=None, batch_ids=None):
        '''
        sparse_feat: saprse tensor
        point_cloud: N*3
        batch_ids: N
        '''
        vx_points, vx_feats = self._get_vertice_feats(sparse_feat)

        if point_cloud is not None:
            assert batch_ids is not None
            points = torch.cat([batch_ids.float().unsqueeze(1), point_cloud], dim=1)
            feats = self._get_point_feats(points, vx_points, vx_feats)
            return feats
        else:
            point_cloud = vx_points[:,1:].contiguous()
            feats = vx_feats
            batch_ids = vx_points[:,0].contiguous()

            return point_cloud, feats, batch_ids
    
    def _get_vertice_feats(self, sparse_tensor):
        feat = sparse_tensor.features
        device = feat.device

        spatial_shape = np.array(sparse_tensor.spatial_shape).astype(np.float)

        if self.voxel_extent is not None:
            unit_voxel_extent = torch.FloatTensor(self.voxel_extent / spatial_shape).to(device)
            voxel_extent = torch.FloatTensor(self.voxel_extent).to(device)
        else:
            unit_voxel_extent = torch.FloatTensor(self.unit_voxel_extent).to(device)
            voxel_extent = torch.FloatTensor(self.unit_voxel_extent*spatial_shape).to(device)

        occupied_voxel = sparse_tensor.indices.float()
        occupied_voxel[:, 1:] = occupied_voxel[:, 1:] * unit_voxel_extent - 0.5 * voxel_extent + 0.5 * unit_voxel_extent

        return occupied_voxel.contiguous().detach(), feat

    def _get_point_feats(self, target_points, query_points, query_feats):
        """
        :param target_points: (n, 4) tensor of the bxyz positions of the unknown features
        :param query_points: (m, 4) tensor of the bxyz positions of the known features
        :param query_feats: (m, C) tensor of features to be propigated
        :return:
            interpolated_feats: (n, C) tensor of the features of the unknown features
        """
        dist, idx = three_nn(target_points, query_points)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=1, keepdim=True)
        weight = dist_recip / norm
        interpolated_feats = three_interpolate(query_feats, idx, weight)

        return interpolated_feats


if __name__ == "__main__":

    p = torch.rand(5, 100, 3).cuda()
    batch_ids = torch.arange(5).cuda()
    p[2,:,:]=200

    p2v = Point2Voxel()
    v2p = Voxel2Point()

    p = p.reshape(-1,3)
    batch_ids = batch_ids.unsqueeze(1).expand(5,100).reshape(-1)
    p,f,batch_ids,flag = p2v(p,p,batch_ids)
    print(p.size())
    print(flag)

    p,f1,batch_ids = v2p(f,p,batch_ids)
    print(f1.size())

    p,f2,batch_ids = v2p(f,None,None)
    print(f2.size())

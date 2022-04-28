# --------------------------------------------------------
# Sparse Steerable Convolutions

# Dataloder of LineMOD dataset
# Modified from https://github.com/j96w/DenseFusion by Jiehong Lin
# --------------------------------------------------------

import os
import numpy as np
import numpy.ma as ma
import math
import random

import yaml
from PIL import Image
from transforms3d.euler import euler2mat

import torch
import torchvision.transforms as transforms

class LinemodDataset():
    def __init__(self, mode, cfg, root='Linemod_preprocessed'):
        self.npoint = cfg.npoint
        self.num_img_per_epoch = cfg.num_img_per_epoch
        self.total_voxel_extent = cfg.total_voxel_extent
        self.mode = mode
        self.root = root
        self.objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.list_rank = []
        self.meta = {}
        self.pt = {}

        item_count = 0
        for item in self.objlist:
            if self.mode == 'train':
                input_file = open('{0}/data/{1}/train.txt'.format(self.root, '%02d' % item))
            else:
                input_file = open('{0}/data/{1}/test.txt'.format(self.root, '%02d' % item))
            while 1:
                item_count += 1
                input_line = input_file.readline()
                if self.mode == 'test' and item_count % 10 != 0:
                    continue
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                self.list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(self.root, '%02d' % item, input_line))
                self.list_depth.append('{0}/data/{1}/depth/{2}.png'.format(self.root, '%02d' % item, input_line))
                if self.mode == 'eval':
                    self.list_label.append('{0}/segnet_results/{1}_label/{2}_label.png'.format(self.root, '%02d' % item, input_line))
                else:
                    self.list_label.append('{0}/data/{1}/mask/{2}.png'.format(self.root, '%02d' % item, input_line))

                self.list_obj.append(item)
                self.list_rank.append(int(input_line))

            meta_file = open('{0}/data/{1}/gt.yml'.format(self.root, '%02d' % item), 'r')
            self.meta[item] = yaml.load(meta_file, Loader=yaml.FullLoader)
            self.pt[item] = self._ply_vtx('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % item))

            print("Object {0} buffer loaded".format(item))

        self.length = len(self.list_rgb)
        self.img_index = np.arange(self.length)
        print("Total img num: {}".format(len(self.list_rgb)))

        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        # self.img_width = 480
        # self.img_length = 640
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = 500
        self.num_pt_mesh_small = 500
        self.symmetry_obj_idx = [7, 8]


    def __len__(self):
        if self.mode == 'train' and self.num_img_per_epoch != -1:
            return self.num_img_per_epoch
        else:
            return self.length

    def reset(self):
        if self.mode == 'train':
            valid_len = self.length
            required_len = self.__len__()
            if required_len >= valid_len:
                self.img_index = np.random.choice(valid_len, required_len)
            else:
                self.img_index = np.random.choice(valid_len, required_len, replace=False)
        else:
            self.img_index = np.arange(self.length)

    def __getitem__(self, index):
        idx = self.img_index[index]
        img = Image.open(self.list_rgb[idx])
        depth = np.array(Image.open(self.list_depth[idx]))
        label = np.array(Image.open(self.list_label[idx]))
        obj = self.list_obj[idx]
        rank = self.list_rank[idx]

        if obj == 2:
            for i in range(0, len(self.meta[obj][rank])):
                if self.meta[obj][rank][i]['obj_id'] == 2:
                    meta = self.meta[obj][rank][i]
                    break
        else:
            meta = self.meta[obj][rank][0]

        # mask
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        if self.mode == 'eval':
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
        else:
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]
        mask = mask_label * mask_depth
        rmin, rmax, cmin, cmax = self._get_bbox(meta['obj_bb'])

        # choose
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if self.mode == 'train' and len(choose)<32:
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)

        if len(choose)==0:
            return self._return_unvalid_output()

        # gt
        target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
        target_t = np.array(meta['cam_t_m2c']) / 1000.0

        # img
        img = np.array(img)[:, :, :3]
        img = img[rmin:rmax, cmin:cmax, :].astype(np.float32).reshape((-1, 3))[choose, :]
        img = img/255.0 - np.array([0.485, 0.456, 0.406])[np.newaxis,:]

        # pts
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

        cam_scale = 1.0
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        cloud = cloud / 1000.0

        centroid = np.mean(cloud, axis=0)
        cloud = cloud - centroid[np.newaxis, :]
        target_t = target_t - centroid

        if self.mode == 'train':
            a1 = np.random.uniform(-math.pi/36.0, math.pi/36.0)
            a2 = np.random.uniform(-math.pi/36.0, math.pi/36.0)
            a3 = np.random.uniform(-math.pi/36.0, math.pi/36.0)
            aug_r = euler2mat(a1, a2, a3)

            cloud = (cloud - target_t[np.newaxis, :]) @ target_r
            target_t = target_t + np.array([random.uniform(-0.02, 0.02) for i in range(3)])
            target_r = target_r @ aug_r
            cloud = cloud @ target_r.T + target_t[np.newaxis, :]

        # point selection
        valid_idx = (np.abs(cloud[:,0])<self.total_voxel_extent[0]*0.5) & (np.abs(cloud[:,1])<self.total_voxel_extent[1]*0.5) & (np.abs(cloud[:,2])<self.total_voxel_extent[2]*0.5)

        if np.sum(valid_idx) < 32 and self.mode == 'train':
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)

        if np.sum(valid_idx)==0:
            return self._return_unvalid_output()

        cloud = cloud[valid_idx, :]
        img = img[valid_idx, :]
        if cloud.shape[0]>self.npoint:
            choose_idx = np.random.choice(cloud.shape[0], self.npoint, replace=False)
        else:
            choose_idx = np.random.choice(cloud.shape[0], self.npoint)
        cloud = torch.FloatTensor(cloud[choose_idx, :])
        img = torch.FloatTensor(img[choose_idx, :])
        centroid = torch.FloatTensor(centroid)

        # model_points
        model_points = self.pt[obj] / 1000.0
        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh_small)
        model_points = np.delete(model_points, dellist, axis=0)
        model_points = torch.FloatTensor(model_points)

        # target_points
        target = np.dot(model_points, target_r.T)
        target = np.add(target, target_t)
        target = torch.FloatTensor(target)

        if self.objlist.index(obj) in self.symmetry_obj_idx:
            sym_flag = 1
        else:
            sym_flag = 0
        sym_flag = torch.FloatTensor([sym_flag])
        obj_index = torch.IntTensor([self.objlist.index(obj)])

        return {
            'input_points': cloud,
            'input_feats': img,
            'model_points': model_points,
            'target_points': target,
            'sym_flag': sym_flag,
            'obj_index': obj_index,
            'centroid': centroid,     
        }
            
    def _get_bbox(self, bbox):
        border_list = self.border_list
        bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
        if bbx[0] < 0:
            bbx[0] = 0
        if bbx[1] >= 480:
            bbx[1] = 479
        if bbx[2] < 0:
            bbx[2] = 0
        if bbx[3] >= 640:
            bbx[3] = 639
        rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
        r_b = rmax - rmin
        for tt in range(len(border_list)):
            if r_b > border_list[tt] and r_b < border_list[tt + 1]:
                r_b = border_list[tt + 1]
                break
        c_b = cmax - cmin
        for tt in range(len(border_list)):
            if c_b > border_list[tt] and c_b < border_list[tt + 1]:
                c_b = border_list[tt + 1]
                break
        center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
        rmin = center[0] - int(r_b / 2)
        rmax = center[0] + int(r_b / 2)
        cmin = center[1] - int(c_b / 2)
        cmax = center[1] + int(c_b / 2)
        if rmin < 0:
            delt = -rmin
            rmin = 0
            rmax += delt
        if cmin < 0:
            delt = -cmin
            cmin = 0
            cmax += delt
        if rmax > 480:
            delt = rmax - 480
            rmax = 480
            rmin -= delt
        if cmax > 640:
            delt = cmax - 640
            cmax = 640
            cmin -= delt
        return rmin, rmax, cmin, cmax

    def _ply_vtx(self, path):
        f = open(path)
        assert f.readline().strip() == "ply"
        f.readline()
        f.readline()
        N = int(f.readline().split()[-1])
        while f.readline().strip() != "end_header":
            continue
        pts = []
        for _ in range(N):
            pts.append(np.float32(f.readline().split()[:3]))
        return np.array(pts)

    def _return_unvalid_output(self):
        return {
            'input_points': torch.zeros(self.npoint,3).float(),
            'input_feats': torch.zeros(self.npoint, 3).float(),
            'model_points': torch.zeros(self.num_pt_mesh_small, 3).float(),
            'target_points': torch.zeros(self.num_pt_mesh_small, 3).float(),
            'sym_flag': torch.FloatTensor([-1]),
            'obj_index': torch.IntTensor([-1]),
            'centroid': torch.zeros(3).float(),
        }

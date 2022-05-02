# --------------------------------------------------------
# Sparse Steerable Convolutions

# Dataloder of REAL275 dataset
# Written by Jiehong Lin 
# --------------------------------------------------------
import os
import math
import cv2
import numpy as np
import glob
import _pickle as cPickle

import torch
from torch.utils.data import Dataset

from data_utils import depth_fill_missing


class REAL275TrainDataset(Dataset):
    def __init__(self, cfg, root='/data/REAL275'):

        self.npoint = cfg.npoint
        self.num_img_per_epoch = cfg.num_img_per_epoch
        self.total_voxel_extent = cfg.total_voxel_extent
        self.filling_miss = cfg.filling_miss
        self.root = root

        img_list_path = ['camera/train_list.txt', 'real/train_list.txt']
        img_list = []
        for path in img_list_path:
            img_list += [os.path.join(path.split('/')[0], line.rstrip('\n'))
                         for line in open(os.path.join(self.root, path))]
        self.img_list = img_list

        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        self.camera_intrinsics = [577.5, 577.5, 319.5, 239.5]    # [fx, fy, cx, cy]
        self.real_intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
        self.sym_ids = [0, 1, 3]    # 0-indexed
        self.norm_scale = 1000.0    # normalization scale

        self.length = len(self.img_list)
        self.img_index = np.arange(self.length)
        print('{} images found.'.format(self.length))

    def __len__(self):
        if self.num_img_per_epoch != -1:
            return self.num_img_per_epoch
        else:
            return self.length

    def reset(self):
        if self.num_img_per_epoch != -1:
            valid_len = self.length
            required_len = self.__len__()
            if required_len >= valid_len:
                self.img_index = np.random.choice(valid_len, required_len)
            else:
                self.img_index = np.random.choice(valid_len, required_len, replace=False)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.img_list[self.img_index[index]])
        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
        assert(len(gts['class_ids'])==len(gts['instance_ids']))
        num_instance = len(gts['instance_ids'])
        idx = np.random.randint(0, num_instance)

        # depth
        if 'camera' in img_path.split('/'):
            cam_fx, cam_fy, cam_cx, cam_cy = self.camera_intrinsics
            depth = load_composed_depth(img_path)
        else:
            cam_fx, cam_fy, cam_cx, cam_cy = self.real_intrinsics
            depth = load_depth(img_path) #480*640

        if depth is None:
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)

        if self.filling_miss:
            depth = depth_fill_missing(depth, self.norm_scale, 1)

        # mask
        rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx])
        mask = cv2.imread(img_path + '_mask.png')[:, :, 2] #480*640
        mask = np.equal(mask, gts['instance_ids'][idx])
        mask = np.logical_and(mask, depth > 0)
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose)<=0:
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)

        # rgb
        rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        rgb = rgb[:, :, ::-1] #480*640*3
        rgb = rgb[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :]
        rgb = rgb/255.0 - np.array([0.485, 0.456, 0.406])[np.newaxis,:]

        # pts
        xmap = self.xmap
        ymap = self.ymap
        pts2 = depth.copy() / self.norm_scale
        pts0 = (xmap - cam_cx) * pts2 / cam_fx
        pts1 = (ymap - cam_cy) * pts2 / cam_fy
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,2,0)).astype(np.float32) # 480*640*3
        pts = pts[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :]

        centroid = np.mean(pts, axis=0)
        pts = pts - centroid[np.newaxis, :]

        # point selection
        choose = (np.abs(pts[:,0])<self.total_voxel_extent[0]*0.5) & (np.abs(pts[:,1])<self.total_voxel_extent[1]*0.5) & (np.abs(pts[:,2])<self.total_voxel_extent[2]*0.5)
        if np.sum(choose) < 32:
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)
        rgb = rgb[choose, :]
        pts = pts[choose, :]

        if pts.shape[0]>self.npoint:
            choose = np.random.choice(pts.shape[0], self.npoint, replace=False)
        else:
            choose = np.random.choice(pts.shape[0], self.npoint)
        rgb = rgb[choose, :]
        pts = pts[choose, :]

        # gt
        cat_id = gts['class_ids'][idx] - 1 # convert to 0-indexed
        translation = gts['translations'][idx].astype(np.float32) - centroid
        rotation = gts['rotations'][idx].astype(np.float32)
        size = gts['scales'][idx] * gts['sizes'][idx].astype(np.float32)
        if cat_id in self.sym_ids:
            theta_x = rotation[0, 0] + rotation[2, 2]
            theta_y = rotation[0, 2] - rotation[2, 0]
            r_norm = math.sqrt(theta_x**2 + theta_y**2)
            s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                                [0.0,            1.0,  0.0           ],
                                [theta_y/r_norm, 0.0,  theta_x/r_norm]])
            rotation = rotation @ s_map
        # nocs = (pts - translation[np.newaxis, :]) / (np.linalg.norm(gts['scales'][idx])+1e-8) @ rotation

        ret_dict = {}

        ret_dict['pts'] = torch.FloatTensor(pts) # N*3
        ret_dict['rgb'] = torch.FloatTensor(rgb)

        ret_dict['category_label'] = torch.IntTensor([cat_id]).long()
        ret_dict['translation_label'] = torch.FloatTensor(translation)
        ret_dict['rotation_label'] = torch.FloatTensor(rotation)
        ret_dict['size_label'] = torch.FloatTensor(size)

        return ret_dict


class REAL275TestDataset(Dataset):
    def __init__(self, cfg, root='/data/REAL275', seg_results='segmentation_results/REAL275'):

        self.npoint = cfg.npoint
        self.total_voxel_extent = cfg.total_voxel_extent
        self.filling_miss = cfg.filling_miss
        self.root = root

        result_pkl_list = glob.glob(os.path.join(seg_results, 'results_*.pkl'))
        self.result_pkl_list = sorted(result_pkl_list)

        self.length = len(self.result_pkl_list)
        print('{} images found.'.format(self.length))

        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        self.camera_intrinsics = [577.5, 577.5, 319.5, 239.5]    # [fx, fy, cx, cy]
        self.real_intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
        self.sym_ids = [0, 1, 3]    # 0-indexed
        self.norm_scale = 1000.0    # normalization scale

    def __len__(self):
        return len(self.result_pkl_list)

    def __getitem__(self, index):
        path = self.result_pkl_list[index]

        with open(path, 'rb') as f:
            data = cPickle.load(f)
        img_path = data['image_path']
        img_path = os.path.join(self.root[:-4], img_path)
        num_instance = len(data['pred_class_ids'])

        # depth
        cam_fx, cam_fy, cam_cx, cam_cy = self.real_intrinsics
        depth = load_depth(img_path) #480*640
        if self.filling_miss:
            depth = depth_fill_missing(depth, self.norm_scale, 1)

        # pts
        xmap = self.xmap
        ymap = self.ymap
        pts2 = depth.copy() / self.norm_scale
        pts0 = (xmap - cam_cx) * pts2 / cam_fx
        pts1 = (ymap - cam_cy) * pts2 / cam_fy
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,2,0)).astype(np.float32) # 480*640*3

        # rgb
        rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        rgb = rgb[:, :, ::-1] #480*640*3
        rgb = rgb/255.0 - np.array([0.485, 0.456, 0.406])[np.newaxis,np.newaxis,:]

        # mask
        pred_mask = data['pred_masks']

        all_pts = []
        all_rgb = []
        all_centroids = []
        flag = torch.zeros(num_instance)

        for j in range(num_instance):
            inst_mask = 255 * pred_mask[:, :, j].astype('uint8')
            rmin, rmax, cmin, cmax = get_bbox(data['pred_bboxes'][j])
            mask = inst_mask > 0
            mask = np.logical_and(mask, depth>0)
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            instance_pts = pts[rmin:rmax, cmin:cmax, :].copy().reshape((-1, 3))
            instance_rgb = rgb[rmin:rmax, cmin:cmax, :].copy().reshape((-1, 3))

            if len(choose)>= 16:
                instance_pts = instance_pts[choose].copy()
                instance_rgb = instance_rgb[choose].copy()
                centroid = np.mean(instance_pts, axis=0)
                pts_ = instance_pts.copy() - centroid[np.newaxis, :]
                choose = (np.abs(pts_[:,0])<self.total_voxel_extent[0]*0.5) & (np.abs(pts_[:,1])<self.total_voxel_extent[1]*0.5) & (np.abs(pts_[:,2])<self.total_voxel_extent[2]*0.5)

                if np.sum(choose) >= 16:
                    instance_pts = instance_pts[choose].copy()
                    instance_rgb = instance_rgb[choose].copy()
                    centroid = np.mean(instance_pts, axis=0)
                    instance_pts = instance_pts.copy() - centroid[np.newaxis, :]    
                    flag[j] = 1

            if flag[j] == 0:
                continue

            if instance_pts.shape[0]>self.npoint:
                choose = np.random.choice(instance_pts.shape[0], self.npoint, replace=False)
            else:
                choose = np.random.choice(instance_pts.shape[0], self.npoint)
            instance_pts = instance_pts[choose, :]
            instance_rgb = instance_rgb[choose, :]

            all_pts.append(torch.FloatTensor(instance_pts))
            all_rgb.append(torch.FloatTensor(instance_rgb))
            all_centroids.append(torch.FloatTensor(centroid))

        ret_dict = {}
        ret_dict['pts'] = torch.stack(all_pts) # N*3
        ret_dict['rgb'] = torch.stack(all_rgb)
        ret_dict['centroids'] = torch.stack(all_centroids)

        ret_dict['gt_class_ids'] = torch.tensor(data['gt_class_ids'])
        ret_dict['gt_bboxes'] = torch.tensor(data['gt_bboxes'])
        ret_dict['gt_RTs'] = torch.tensor(data['gt_RTs'])
        ret_dict['gt_scales'] = torch.tensor(data['gt_scales'])
        ret_dict['gt_handle_visibility'] = torch.tensor(data['gt_handle_visibility'])

        ret_dict['pred_class_ids'] = torch.tensor(data['pred_class_ids'][flag==1])
        ret_dict['pred_bboxes'] = torch.tensor(data['pred_bboxes'][flag==1])
        ret_dict['pred_scores'] = torch.tensor(data['pred_scores'][flag==1])

        return ret_dict        




def load_depth(img_path):
    """ Load depth image from img_path. """
    depth_path = img_path + '_depth.png'
    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16


def load_composed_depth(img_path):
    """ Load depth image from img_path. """
    img_path_ = img_path.replace('/data/camera/', '/data/camera_full_depths/')
    depth_path = img_path_ + '_composed.png'
    if os.path.exists(depth_path):
        depth = cv2.imread(depth_path, -1)
        if len(depth.shape) == 3:
            # This is encoded depth image, let's convert
            # NOTE: RGB is actually BGR in opencv
            depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
            depth16 = np.where(depth16==32001, 0, depth16)
            depth16 = depth16.astype(np.uint16)
        elif len(depth.shape) == 2 and depth.dtype == 'uint16':
            depth16 = depth
        else:
            assert False, '[ Error ]: Unsupported depth type.'
        return depth16
    else:
        return None


def get_bbox(bbox):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    img_width = 480
    img_length = 640
    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, 440)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


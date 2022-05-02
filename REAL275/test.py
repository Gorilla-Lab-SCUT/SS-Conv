# --------------------------------------------------------
# Sparse Steerable Convolutions

# Evaluation on LinMOD dataset for 6D pose estimation
# Written by Jiehong Lin
# --------------------------------------------------------

import os
import sys
import numpy as np
import random
import argparse
from tqdm import tqdm
import yaml
import pickle as cPickle
import torch
import gorilla
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..', 'Common_utils'))
from evaluation import evaluate


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pose Estimation on REAL275.")
    parser.add_argument("--gpu",
                        type=str,
                        default="0",
                        help="gpu index.")
    parser.add_argument("--config",
                        type=str,
                        default="config.yaml",
                        help="config file.")
    parser.add_argument("--model",
                        type=str,
                        default="model",
                        help="name of training model.")
    parser.add_argument("--data",
                        type=str,
                        default="data",
                        help="data dir.")
    parser.add_argument("--segmentation",
                        type=str,
                        default="segmentation_results/REAL275",
                        help="data dir.")
    parser.add_argument("--epoch",
                        type=int,
                        default=20,
                        help="epoch for testing")
    parser.add_argument("--niter",
                        type=int,
                        default=1,
                        help="iterations of refinement")
    args_cfg = parser.parse_args()
    return args_cfg

def init():
    args = get_parser()
    cfg = gorilla.Config.fromfile(args.config)
    cfg.gpu = args.gpu
    cfg.model_name = args.model

    cfg.model.niter = args.niter
    cfg.test.data_dir = args.data
    cfg.test.seg_dir = args.segmentation
    cfg.test.epoch = args.epoch
    cfg.test.dataset.total_voxel_extent = np.array(cfg.model.voxel_num_limit).astype(np.float) \
        * np.array(cfg.model.unit_voxel_extent)

    print(cfg)

    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpu)

    return cfg


def test(model, save_path, cfg):
    model = model.eval()

    dataset = importlib.import_module(cfg.dataset.name)
    dataset = dataset.REAL275TestDataset(cfg.dataset, cfg.data_dir, cfg.seg_dir)
    dataloder = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.dataloader.bs,
            num_workers=cfg.dataloader.num_workers,
            shuffle=cfg.dataloader.shuffle,
            sampler=None,
            drop_last=cfg.dataloader.drop_last,
            pin_memory=cfg.dataloader.pin_memory
        )

    with tqdm(total=len(dataloder)) as t:
        for i, data in enumerate(dataloder):
            
            with torch.no_grad():
                inputs = {
                    'pts': data['pts'][0].cuda(),
                    'rgb': data['rgb'][0].cuda(),
                }
                pred = model(inputs)

                pred_rotation = pred['rotation']
                pred_translation = pred['translation']  + data['centroids'][0].to(pred_rotation.device)
                pred_size = pred['size']
                pred_scale = torch.norm(pred_size, dim=1, keepdim=True)
                pred_size = pred_size / pred_scale

                num_instance = pred_rotation.size(0)
                pred_RTs =torch.eye(4).unsqueeze(0).repeat(num_instance, 1, 1).float().to(pred_rotation.device)
                pred_RTs[:, :3, 3] = pred_translation
                pred_RTs[:, :3, :3] = pred_rotation * pred_scale.unsqueeze(2)
                pred_scales = pred_size

            # results
            result = {}
            result['gt_class_ids'] = data['gt_class_ids'][0].numpy()
            result['gt_bboxes'] = data['gt_bboxes'][0].numpy()
            result['gt_RTs'] = data['gt_RTs'][0].numpy()
            result['gt_scales'] = data['gt_scales'][0].numpy()
            result['gt_handle_visibility'] = data['gt_handle_visibility'][0].numpy()
            result['pred_class_ids'] = data['pred_class_ids'][0].numpy()
            result['pred_bboxes'] = data['pred_bboxes'][0].numpy()
            result['pred_scores'] = data['pred_scores'][0].numpy()

            result['pred_RTs'] = pred_RTs.detach().cpu().numpy()
            result['pred_scales'] = pred_scales.detach().cpu().numpy()

            path = dataset.result_pkl_list[i]
            with open(os.path.join(save_path, path.split('/')[-1]), 'wb') as f:
                cPickle.dump(result, f)

            t.set_description(
                "Test [{}/{}][{}]: ".format(i+1, dataset.__len__(), num_instance)
            )
            t.update(1)


if __name__ == "__main__":
    cfg = init()
    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    save_path = os.path.join(cfg.log_dir, cfg.model_name + '_results')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # model
    MODEL = importlib.import_module(cfg.model_name)
    model = MODEL.Network(cfg=cfg.model).cuda()
    if cfg.test.epoch == -1:
        checkpoint = os.path.join(cfg.log_dir, cfg.model_name + '_latest.pth')
    else:
        checkpoint = os.path.join(cfg.log_dir, cfg.model_name + '_epoch_' + str(cfg.test.epoch) + '.pth')
    gorilla.solver.load_checkpoint(model=model, filename=checkpoint)

    test(model, save_path, cfg.test)

    evaluate(save_path)




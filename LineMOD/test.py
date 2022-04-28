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
import torch
import gorilla
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..', 'Common_utils'))

def get_parser():
    parser = argparse.ArgumentParser(
        description="Pose Estimation on LineMOD.")
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
                        default="/data/Linemod_preprocessed",
                        help="data dir.")
    parser.add_argument("--epoch",
                        type=int,
                        default=10,
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
    cfg.test.log_dir = cfg.log_dir
    cfg.test.data_dir = args.data
    cfg.test.model_name = cfg.model_name
    cfg.test.epoch = args.epoch
    cfg.test.dataset.total_voxel_extent = np.array(cfg.model.voxel_num_limit).astype(np.float) \
        * np.array(cfg.model.unit_voxel_extent)

    print(cfg)

    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpu)

    return cfg


def test(model, cfg):
    model = model.eval()

    dataset = importlib.import_module(cfg.dataset.name)
    dataset = dataset.LinemodDataset('eval', cfg.dataset, cfg.data_dir)
    dataloder = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.dataloader.bs,
            num_workers=cfg.dataloader.num_workers,
            shuffle=cfg.dataloader.shuffle,
            sampler=None,
            drop_last=cfg.dataloader.drop_last,
            pin_memory=cfg.dataloader.pin_memory
        )

    diameter = []
    meta_file = open(os.path.join(cfg.data_dir, 'models_info.yml'), 'r')
    meta = yaml.load(meta_file, Loader=yaml.FullLoader)
    objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    num_objects = 13
    for obj in objlist:
        diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
    fw = open('{0}/{1}_epoch{2}_eval_results.txt'.format(cfg.log_dir, cfg.model_name, str(cfg.epoch)), 'w')

    success_count = [0 for i in range(num_objects)]
    num_count = [0 for i in range(num_objects)]
    count = 0

    with tqdm(total=len(dataloder)) as t:
        for i, data in enumerate(dataloder):
            with torch.no_grad():
                for key in data:
                    data[key] = data[key].cuda()
            pred = model(data)

            pred_points = pred['pred_points']
            target_points = pred['target_points']
            sym_flag = pred['sym_flag']
            obj_index = pred['obj_index']            

            asym_dis = torch.mean(torch.norm(pred_points - target_points, dim=2), dim=1)
            sym_dis = torch.mean(torch.min(torch.norm(pred_points.unsqueeze(2) - target_points.unsqueeze(1), dim=3), 2)[0], dim=1)

            batch_id = 0
            batch_size = pred_points.size(0)
            for k in range(batch_size):
                count += 1
                if sym_flag[k].item() == -1:
                    fw.write('No.{0} NOT Pass! Lost detection!\n'.format(count))
                    continue
                elif sym_flag[k].item() == 0:
                    dis = asym_dis[batch_id].item()
                elif sym_flag[k].item() == 1:
                    dis = sym_dis[batch_id].item()
                idx = obj_index[k].item()
                num_count[idx] += 1    
                if dis < diameter[idx]:
                    success_count[idx] += 1
                    fw.write('No.{0} Pass! Distance: {1}  ({2})\n'.format(count, dis, idx))
                else:
                    fw.write('No.{0} NOT Pass! Distance: {1}  ({2})\n'.format(count, dis, idx))
            t.set_description(
                "Test [{}/{}][{}/{}] - success rate: {}".format(i+1, len(dataloder), count, dataset.__len__(),  float(sum(success_count)) / sum(num_count))
            )
            t.update(1)

    for i in range(num_objects):
        print('Object {0} success rate: {1}'.format(objlist[i], float(success_count[i]) / num_count[i]))
        fw.write('Object {0} success rate: {1}\n'.format(objlist[i], float(success_count[i]) / num_count[i]))
    print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
    fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
    fw.close()


if __name__ == "__main__":
    cfg = init()
    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    # model
    MODEL = importlib.import_module(cfg.model_name)
    model = MODEL.Network(cfg=cfg.model).cuda()
    if cfg.test.epoch == -1:
        checkpoint = os.path.join(cfg.log_dir, cfg.model_name + '_latest.pth')
    else:
        checkpoint = os.path.join(cfg.log_dir, cfg.model_name + '_epoch_' + str(cfg.test.epoch) + '.pth')
    gorilla.solver.load_checkpoint(model=model, filename=checkpoint)

    test(model, cfg.test)




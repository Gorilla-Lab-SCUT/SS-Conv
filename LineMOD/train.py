# --------------------------------------------------------
# Sparse Steerable Convolutions

# Training on LinMOD dataset for 6D pose estimation
# Written by Jiehong Lin
# --------------------------------------------------------

import os
import sys
import numpy as np
import random
import argparse
import logging
import torch
import gorilla
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..', 'Common_utils'))
import train_utils

import ipdb
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
    parser.add_argument("--checkpoint_epoch",
                        type=int,
                        default=0,
                        help="0:from scratch, -1: from latest epoch, x: from epoch x")
    args_cfg = parser.parse_args()
    return args_cfg


def init():
    args = get_parser()
    cfg = gorilla.Config.fromfile(args.config)
    cfg.gpu = args.gpu
    cfg.model_name = args.model
    cfg.data_dir = args.data

    cfg.train.log_dir = cfg.log_dir
    cfg.train.model_name = cfg.model_name
    cfg.train.checkpoint_epoch = args.checkpoint_epoch

    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    logger = train_utils.get_logger(level_print=logging.INFO, level_save=logging.WARNING, path_file=cfg.log_dir+'/'+args.model+'_training_logger.log')
    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpu)

    return logger, cfg


if __name__ == "__main__":
    logger, cfg = init()

    logger.warning("************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpu))

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    # model
    logger.info("=> creating model ...")
    MODEL = importlib.import_module(cfg.model_name)
    net = MODEL.Network(cfg=cfg.model)
    net = net.cuda()
    loss  = MODEL.Loss(cfg=cfg.train.loss).cuda()

    # dataloder
    train_dataset = importlib.import_module(cfg.train.dataset.name)
    cfg.train.dataset.total_voxel_extent = np.array(cfg.model.voxel_num_limit).astype(np.float) * np.array(cfg.model.unit_voxel_extent)
    cfg.train.dataset.num_img_per_epoch = cfg.train.dataset.num_batch_per_epoch * cfg.train.dataloader.bs
    train_dataset = train_dataset.LinemodDataset('train', cfg.train.dataset, cfg.data_dir)
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.train.dataloader.bs,
            num_workers=cfg.train.dataloader.num_workers,
            shuffle=cfg.train.dataloader.shuffle,
            sampler=None,
            drop_last=cfg.train.dataloader.drop_last,
            pin_memory=cfg.train.dataloader.pin_memory
        )
    dataloaders = {
        "train": train_dataloader,
    }

    # solver
    Trainer = train_utils.Training_Solver(net, loss, dataloaders, logger, cfg.train)
    Trainer.solve()

    logger.info('\nFinish!\n')


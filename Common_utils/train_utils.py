# --------------------------------------------------------
# Sparse Steerable Convolutions

# Common utils for network training
# Written by Jiehong Lin
# --------------------------------------------------------

import os
import logging
from pickletools import optimize
import time
import torch
import gorilla
from tensorboardX import SummaryWriter


def get_logger(level_print, level_save, path_file, name_logger = "logger"):
    # level: logging.INFO / logging.WARN
    logger = logging.getLogger(name_logger)
    logger.setLevel(level = logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # set file handler
    handler_file = logging.FileHandler(path_file)
    handler_file.setLevel(level_save)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    # set console holder
    handler_view = logging.StreamHandler()
    handler_view.setFormatter(formatter)
    handler_view.setLevel(level_print)
    logger.addHandler(handler_view)
    return logger


class tools_writer():
    def __init__(self, dir_project, num_counter, get_sum):
        if not os.path.isdir(dir_project):
            os.makedirs(dir_project)
        if get_sum:
            writer = SummaryWriter(dir_project)
        else:
            writer = None
        self.writer = writer
        self.num_counter = num_counter
        self.list_couter = []
        for i in range(num_counter):
            self.list_couter.append(0)

    def update_scalar(self, list_name, list_value, index_counter, prefix):
        for name, value in zip(list_name, list_value):
            self.writer.add_scalar(prefix+name, float(value), self.list_couter[index_counter])

        self.list_couter[index_counter] += 1

    def refresh(self):
        for i in range(self.num_counter):
            self.list_couter[i] = 0


class Training_Solver(gorilla.solver.BaseSolver):
    def __init__(self, model, loss, dataloaders, logger, cfg, meta=None):
        super(Training_Solver, self).__init__(
            model = model,
            dataloaders = dataloaders,
            cfg = cfg,
            logger = logger,
        )
        self.loss = loss
        self.logger.propagate = 0

        self.iterations_to_write = cfg.iterations_to_write
        self.epochs_to_save = cfg.epochs_to_save
        self.log_dir = cfg.log_dir
        self.model_name = cfg.model_name

        if cfg.checkpoint_epoch == 0:
            self.epoch = 1
        else:
            if cfg.checkpoint_epoch == -1:
                checkpoint = os.path.join(cfg.log_dir, cfg.model_name + '_latest.pth')
            else:
                checkpoint = os.path.join(cfg.log_dir, cfg.model_name + '_epoch_' + str(cfg.checkpoint_epoch) + '.pth')
            gorilla.solver.resume(
                self.model,
                checkpoint,
                optimizer = self.optimizer,
                scheduler = self.lr_scheduler,
                resume_optimizer=True,
                resume_scheduler=True,
                map_location="default",
            )
            self.logger.info("=> loading checkpoint from epoch {} ...".format(self.lr_scheduler.last_epoch))
            self.epoch = self.lr_scheduler.last_epoch + 1

        tb_writer = tools_writer(dir_project=self.log_dir, num_counter=2, get_sum=False)
        tb_writer.writer = self.tb_writer
        self.tb_writer = tb_writer

    def solve(self):
        while self.epoch<=self.cfg.max_epoch:
            self.logger.info('Epoch {} :'.format(self.epoch))

            end = time.time()
            dict_info_train = self.train()
            train_time = time.time()-end

            dict_info = {'train_time(min)': train_time/60.0}
            for key, value in dict_info_train.items():
                if 'loss' in key:
                    dict_info['train_'+key] = value

            ckpt_path = os.path.join(self.log_dir, self.model_name + '_latest.pth')
            gorilla.solver.save_checkpoint(model=self.model, filename=ckpt_path, optimizer=self.optimizer, scheduler=self.lr_scheduler, meta={"epoch": self.epoch})
            if self.epoch % self.epochs_to_save ==0:
                ckpt_path = os.path.join(self.log_dir, self.model_name + '_epoch_' + str(self.epoch) + '.pth')
                gorilla.solver.save_checkpoint(model=self.model, filename=ckpt_path, optimizer=self.optimizer, scheduler=self.lr_scheduler, meta={"epoch": self.epoch})

            prefix = 'Epoch {} - '.format(self.epoch)
            write_info = self.get_logger_info(prefix, dict_info=dict_info)
            self.logger.warning(write_info)
            self.epoch += 1

    def train(self):
        mode = 'train'
        self.model.train()
        self.dataloaders["train"].dataset.reset()

        for i, data in enumerate(self.dataloaders["train"]):

            self.optimizer.zero_grad()
            end = time.time()
            loss, dict_info_step = self.step(data, mode)
            forward_time = time.time()-end

            end = time.time()
            loss.backward()
            self.optimizer.step()
            backward_time = time.time()-end

            dict_info_step.update({
                'T_forward': forward_time,
                'T_backward': backward_time,
            })
            self.log_buffer.update(dict_info_step)

            if i % self.iterations_to_write == 0:
                self.log_buffer.average(self.iterations_to_write)
                prefix = '[{}/{}][{}/{}] Train - '.format(self.epoch, self.cfg.max_epoch, i, len(self.dataloaders["train"]))
                write_info = self.get_logger_info(prefix, dict_info=self.log_buffer._output)
                self.logger.info(write_info)
                self.write_summary(self.log_buffer._output, mode)
            
        dict_info_epoch = self.log_buffer.avg
        self.log_buffer.clear()
        self.lr_scheduler.step()

        return dict_info_epoch

    def evaluate(self):
        pass

    def step(self, data, mode):
        torch.cuda.synchronize()
        for key in data:
            data[key] = data[key].cuda()
        outputs = self.model(data)
        dict_losses = self.loss(outputs)

        keys = list(dict_losses.keys())
        dict_info = {'all_loss': 0}
        if 'all_loss' in keys:
            loss_all = dict_losses['all_loss']
            for key in keys:
                dict_info[key] = float(dict_losses[key].item())
        else:
            loss_all = 0
            for key in keys:
                loss_all += dict_losses[key]
                dict_info[key] = float(dict_losses[key].item())
            dict_info['all_loss'] = float(loss_all.item())

        if mode == 'train':
            dict_info['lr'] = self.lr_scheduler._get_closed_form_lr()[0]

        return loss_all, dict_info

    def get_logger_info(self, prefix, dict_info):
        info = prefix
        for key, value in dict_info.items():
            if 'T_' in key:
                info = info + '{}: {:.3f}\t'.format(key, value)
            else:
                info = info + '{}: {:.5f}\t'.format(key, value)

        return info

    def write_summary(self, dict_info, mode):
        keys   = list(dict_info.keys())
        values = list(dict_info.values())
        if mode == "train":
            self.tb_writer.update_scalar(list_name=keys, list_value=values, index_counter=0, prefix="train_")
        elif mode == "eval":
            self.tb_writer.update_scalar(list_name=keys, list_value=values, index_counter=1, prefix="eval_")
        else:
            assert False
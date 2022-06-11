#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2018/8/2 下午3:23
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : engine.py
import argparse
import os
import shutil
import time

import torch
import torch.distributed as dist

from utils.pyt_utils import load_model, link_file, ensure_dir
from utils.pyt_utils import on_load_checkpoint


class State(object):
    def __init__(self):
        self.epoch = 0
        self.iteration = 0
        self.dataloader = None
        self.model = None
        self.optimizer = None

    def register(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Engine(object):
    def __init__(self, custom_arg, logger, continue_state_object):
        assert continue_state_object is not None, "our proj. only works upon the pretrained weight"
        self.logger = logger
        self.state = State()
        self.devices = None
        self.distributed = False
        self.parser = argparse.ArgumentParser()
        self.inject_default_parser()
        self.args = custom_arg
        self.continue_state_object = continue_state_object

        if 'WORLD_SIZE' in os.environ:
            self.distributed = int(os.environ['WORLD_SIZE']) >= 1
        else:
            self.distributed = self.args.ddp
        self.local_rank = 0 if self.args.local_rank < 0 else self.args.local_rank
        self.gpus = self.args.gpus
        self.world_size = self.args.world_size

        if self.distributed:
            os.environ['MASTER_ADDR'] = '127.0.0.4'
            os.environ['MASTER_PORT'] = '9904'
            dist.init_process_group(backend="nccl",
                                    init_method='env://',
                                    rank=self.local_rank,
                                    world_size=self.world_size)
        else:
            self.world_size = 1

    def inject_default_parser(self):
        p = self.parser
        p.add_argument('-d', '--devices', default='0',
                       help='set data parallel training')
        p.add_argument('-c', '--continue', type=str,
                       dest="continue_fpath",
                       help='continue from one certain checkpoint')
        p.add_argument('-p', '--port', type=str,
                       default='16001',
                       dest="port",
                       help='port for init_process_group')
        p.add_argument('--debug', default=0, type=int,
                       help='whlocal_rankether to use the debug mode')
        p.add_argument('-e', '--epochs', default='last', type=str)

        p.add_argument('-v', '--verbose', default=False, action='store_true')
        p.add_argument('--show_image', '-s', default=True,
                       action='store_true')
        p.add_argument('--save_path', default=None)

    def register_state(self, **kwargs):
        self.state.register(**kwargs)

    def update_iteration(self, epoch, iteration):
        self.state.epoch = epoch
        self.state.iteration = iteration

    def save_checkpoint(self, path):
        self.logger.info("Saving checkpoint to file {}".format(path))
        t_start = time.time()

        state_dict = {}

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in self.state.model.state_dict().items():
            key = k
            if k.split('.')[0] == 'module':
                key = k[7:]
            new_state_dict[key] = v
        state_dict['model'] = new_state_dict
        if self.state.optimizer is not None:
            state_dict['optimizer'] = self.state.optimizer.state_dict()
        state_dict['epoch'] = self.state.epoch
        state_dict['iteration'] = self.state.iteration

        t_iobegin = time.time()
        torch.save(state_dict, path)
        del state_dict
        del new_state_dict
        t_end = time.time()
        self.logger.info(
            "Save checkpoint to file {}, "
            "Time usage:\n\tprepare snapshot: {}, IO: {}".format(
                path, t_iobegin - t_start, t_end - t_iobegin))

    def link_tb(self, source, target):
        ensure_dir(source)
        ensure_dir(target)
        link_file(source, target)

    def save_and_link_checkpoint(self, snapshot_dir, log_dir=None, log_dir_link=None, m_iou=None, name=None):
        ensure_dir(snapshot_dir)
        if name is None:
            current_epoch_checkpoint = os.path.join(snapshot_dir, 'epoch-{}-iou-{}.pth'.format(
                self.state.epoch, m_iou))
        else:
            current_epoch_checkpoint = os.path.join(snapshot_dir, '{}.pth'.format(
                name))

        ''' 如果旧文件存在，先删除 '''
        if os.path.exists(current_epoch_checkpoint):
            os.remove(current_epoch_checkpoint)

        self.save_checkpoint(current_epoch_checkpoint)
        last_epoch_checkpoint = os.path.join(snapshot_dir, 'epoch-last.pth')
        # link_file(current_epoch_checkpoint, last_epoch_checkpoint)
        try:
            shutil.copy(current_epoch_checkpoint, last_epoch_checkpoint)
        except:
            pass

    def restore_checkpoint(self, extra_channel=False, eval=False):
        t_start = time.time()
        continue_state_object = self.continue_state_object
        self.logger.critical('restoring ckpt from pretrained file {}.'.format(continue_state_object))

        if self.distributed:
            tmp = torch.load(continue_state_object,
                             map_location=lambda storage, loc: storage.cuda(self.local_rank))
        else:
            tmp = torch.load(continue_state_object)

        t_ioend = time.time()
        if eval:
            self.state.model = on_load_checkpoint(model=self.state.model, checkpoint=tmp['model'])
        else:
            self.state.model = load_model(self.state.model, tmp['state_dict'], True, strict=True,
                                          extra_channel=extra_channel)

        self.state.epoch = 0  # tmp['epoch'] + 1
        self.state.iteration = 0  # tmp['iteration']
        del tmp
        t_end = time.time()
        self.logger.info("Load checkpoint from file {}, "
                         "Time usage:\n\tIO: {}, restore snapshot: {}".format(self.continue_state_object,
                                                                              t_ioend - t_start, t_end - t_ioend))

    def load_pebal_ckpt(self, ckpt_name, model):
        tmp = torch.load(ckpt_name)
        self.logger.critical('restoring pebal ckpt from {}'.format(ckpt_name))
        state_dict = tmp
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']

        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k
            new_state_dict[name] = v
        state_dict = new_state_dict
        model.module.load_state_dict(state_dict, strict=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            self.logger.warning(
                "A exception occurred during Engine initialization, "
                "give up running process")
            return False

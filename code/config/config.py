import os
import numpy
from easydict import EasyDict

C = EasyDict()
config = C
cfg = C

C.seed = 666

"""Root Directory Config"""
C.repo_name = 'pebal'
C.root_dir = os.path.realpath(".")

"""Data Dir and Weight Dir"""
C.city_root_path = '../dataset/city_scape'  # path/to/your/city_scape
C.coco_root_path = '/media/yuyuan/Applications/pebal/coco'  # path/to/your/coco
C.fishy_root_path = '/media/yuyuan/Applications/pebal/fishyscapes'  # path/to/your/fishy

C.pebal_weight_path = os.path.join(C.root_dir, 'ckpts', 'pebal', 'best_ad_ckpt.pth')
C.pretrained_weight_path = os.path.join(C.root_dir, 'ckpts', 'pretrained_ckpts', 'cityscapes_best.pth')

"""Network Config"""
C.fix_bias = True
C.bn_eps = 1e-5
C.bn_momentum = 0.1

"""Image Config"""
C.num_classes = 19 + 1  # NOTE: 1 more channel for gambler loss
C.image_mean = numpy.array([0.485, 0.456, 0.406])  # 0.485, 0.456, 0.406
C.image_std = numpy.array([0.229, 0.224, 0.225])

C.image_height = 900
C.image_width = 900

C.num_train_imgs = 2975
C.num_eval_imgs = 500

"""Train Config"""
C.lr = 1e-5
C.batch_size = 8
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 1e-4

C.nepochs = 40
C.niters_per_epoch = C.num_train_imgs // C.batch_size
C.num_workers = 8
C.train_scale_array = [0.5, 0.75, 1, 1.5, 1.75, 2.0]
C.void_number = 5
C.warm_up_epoch = 0

"""Eval Config"""
C.eval_epoch = 1
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1, ]  # 0.5, 0.75, 1, 1.5, 1.75
C.eval_flip = False
C.eval_base_size = 800
C.eval_crop_size = 800

"""Display Config"""
C.record_info_iter = 20
C.display_iter = 50

"""Wandb Config"""
# Specify you wandb environment KEY; and paste here
C.wandb_key = "your wandb key"

# Your project [work_space] name
C.proj_name = "OoD_Segmentation"

# Your current experiment name
C.experiment_name = "your_pebal_exp"

# half pretrained_ckpts-loader upload images; loss upload every iteration
C.upload_image_step = [0, int((C.num_train_imgs / C.batch_size) / 2)]

# False for debug; True for visualize
C.wandb_online = True

"""Save Config"""
C.saved_dir = os.path.join(C.root_dir, 'ckpts', C.experiment_name)

if not os.path.exists(C.saved_dir):
    os.mkdir(C.saved_dir)

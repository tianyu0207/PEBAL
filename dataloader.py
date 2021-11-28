import os
import cv2
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils import data
import random
from config import config
from utils.img_utils import generate_random_crop_pos, random_crop_pad_to_shape
from torch.utils.data import Dataset
from collections import namedtuple
from PIL import Image
from typing import Callable, Optional
from typing import Any, Callable, Optional, Tuple
from collections import namedtuple

def normalize(img, mean, std):
    # pytorch pretrained model need the input range: 0-1
    img = img.astype(np.float32) / 255.0
    img = img - mean
    img = img / std
    return img


def random_mirror(img, gt=None):
    if random.random() >= 0.2:
        img = cv2.flip(img, 1)
        if gt is not None:
            gt = cv2.flip(gt, 1)

    return img, gt

def random_scale(img, gt=None, scales=None):
    scale = random.choice(scales)
    # scale = random.uniform(scales[0], scales[-1])
    sh = int(img.shape[0] * scale)
    sw = int(img.shape[1] * scale)
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    if gt is not None:
        gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)

    return img, gt, scale

def SemanticEdgeDetector(gt):
    id255 = np.where(gt == 255)
    no255_gt = np.array(gt)
    no255_gt[id255] = 0
    cgt = cv2.Canny(no255_gt, 5, 5, apertureSize=7)
    edge_radius = 7
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_radius, edge_radius))
    cgt = cv2.dilate(cgt, edge_kernel)
    # print(cgt.max(), cgt.min())
    cgt[cgt>0] = 1
    return cgt


class TrainPre(object):
    def __init__(self, img_mean, img_std,
                 augment=True):
        self.img_mean = img_mean
        self.img_std = img_std
        self.augment = augment

    def __call__(self, img, gt=None):
        # gt = gt - 1     # label 0 is invalid, this operation transfers label 0 to label 255
        if not self.augment:
            return normalize(img, self.img_mean, self.img_std), None, None, None

        img, gt = random_mirror(img, gt)
        if config.train_scale_array is not None:
            img, gt, scale = random_scale(img, gt, config.train_scale_array)

        img = normalize(img, self.img_mean, self.img_std)
        if gt is not None:
            cgt = SemanticEdgeDetector(gt)
        else:
            cgt = None

        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)

        p_img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        if gt is not None:
            p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
            p_cgt, _ = random_crop_pad_to_shape(cgt, crop_pos, crop_size, 255)
        else:
            p_gt = None
            p_cgt = None

        p_img = p_img.transpose(2, 0, 1)
        extra_dict = {}

        return p_img, p_gt, p_cgt, extra_dict


class ValPre(object):
    def __call__(self, img, gt):
        # gt = gt - 1
        extra_dict = {}
        return img, gt, None, extra_dict


def get_train_loader(engine, dataset, train_source, unsupervised=False, collate_fn=None,
                     augment=True):
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': train_source,
                    'eval_source': config.eval_source}

    train_preprocess = TrainPre(config.image_mean, config.image_std, augment=augment)

    if unsupervised is False:
        train_dataset = dataset(data_setting, "train", train_preprocess,
                                config.max_samples, unsupervised=False)
    else:
        train_dataset = dataset(data_setting, "train", train_preprocess,
                                config.max_samples, unsupervised=True)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler,
                                   collate_fn=collate_fn)

    return train_loader, train_sampler


def get_test_loader(dataset, eval_source, config_file):
    data_setting = {'img_root': config_file.img_root_folder,
                    'gt_root': config_file.gt_root_folder,
                    'train_source': config_file.train_source,
                    'eval_source': eval_source}

    val_preprocess = ValPre()

    test_dataset = dataset(data_setting, 'val', val_preprocess)

    return test_dataset



class Cityscapes(Dataset):
    """`
    Cityscapes Dataset http://www.cityscapes-dataset.com/
    Labels based on https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    """
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    labels = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    """Normalization parameters"""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    """Useful information from labels"""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]
    for i in range(len(labels)):
        if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
            ignore_in_eval_ids.append(labels[i].train_id)
    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        if labels[i].train_id not in ignore_in_eval_ids:
            train_ids.append(labels[i].train_id)
            color_palette_train_ids[labels[i].train_id] = labels[i].color
            train_id2id.append(labels[i].id)
    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}

    def __init__(self, root: str = "/home/yu/yu_ssd/city_scape", split: str = "val", mode: str = "gtFine",
                 target_type: str = "semantic_train_id", transform: Optional[Callable] = None,
                 predictions_root: Optional[str] = None) -> None:
        """
        Cityscapes dataset loader
        """
        self.root = root
        self.split = split
        self.mode = 'gtFine' if "fine" in mode.lower() else 'gtCoarse'
        self.transform = transform
        self.images_dir = os.path.join(self.root, 'images', 'city_gt_fine', self.split)
        self.targets_dir = os.path.join(self.root, 'annotation', 'city_gt_fine', self.split)
        self.predictions_dir = os.path.join(predictions_root, self.split) if predictions_root is not None else ""
        self.images = []
        self.targets = []
        self.predictions = []

        img_dir = self.images_dir
        target_dir = self.targets_dir
        pred_dir = self.predictions_dir
        for file_name in os.listdir(img_dir):
            target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                         self._get_target_suffix(self.mode, target_type))
            self.images.append(os.path.join(img_dir, file_name))
            self.targets.append(os.path.join(target_dir, target_name))
            self.predictions.append(os.path.join(pred_dir, file_name.replace("_leftImg8bit", "")))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')

        if self.split in ['train', 'val']:
            target = Image.open(self.targets[index])
        else:
            target = None
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def _get_target_suffix(mode: str, target_type: str) -> str:
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic_id':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'semantic_train_id':
            return '{}.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            print("'%s' is not a valid target type, choose from:\n" % target_type +
                  "['instance', 'semantic_id', 'semantic_train_id', 'color']")
            exit()




class LostAndFound(Dataset):

    LostAndFoundClass = namedtuple('LostAndFoundClass', ['name', 'id', 'train_id', 'category_name',
                                                         'category_id', 'color'])

    labels = [
        LostAndFoundClass('unlabeled', 0, 255, 'Miscellaneous', 0, (0, 0, 0)),
        LostAndFoundClass('ego vehicle', 0, 255, 'Miscellaneous', 0, (0, 0, 0)),
        LostAndFoundClass('rectification border', 0, 255, 'Miscellaneous', 0, (0, 0, 0)),
        LostAndFoundClass('out of roi', 0, 255, 'Miscellaneous', 0, (0, 0, 0)),
        LostAndFoundClass('background', 0, 255, 'Counter hypotheses', 1, (0, 0, 0)),
        LostAndFoundClass('free', 1, 1, 'Counter hypotheses', 1, (128, 64, 128)),
        LostAndFoundClass('Crate (black)', 2, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Crate (black - stacked)', 3, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Crate (black - upright)', 4, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Crate (gray)', 5, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Crate (gray - stacked) ', 6, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Crate (gray - upright)', 7, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Bumper', 8, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Cardboard box 1', 9, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Crate (blue)', 10, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Crate (blue - small)', 11, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Crate (green)', 12, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Crate (green - small)', 13, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Exhaust Pipe', 14, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Headlight', 15, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Euro Pallet', 16, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Pylon', 17, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Pylon (large)', 18, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Pylon (white)', 19, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Rearview mirror', 20, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Tire', 21, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Ball', 22, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Bicycle', 23, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Dog (black)', 24, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Dog (white)', 25, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Kid dummy', 26, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Bobby car (gray)', 27, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Bobby Car (red)', 28, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Bobby Car (yellow)', 29, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Cardboard box 2', 30, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Marker Pole (lying)', 31, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Plastic bag (bloated)', 32, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Post (red - lying)', 33, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Post Stand', 34, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Styrofoam', 35, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Timber (small)', 36, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Timber (squared)', 37, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Wheel Cap', 38, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Wood (thin)', 39, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Kid (walking)', 40, 2, 'Humans', 6, (0, 0, 142)),
        LostAndFoundClass('Kid (on a bobby car)', 41, 2, 'Humans', 6, (0, 0, 142)),
        LostAndFoundClass('Kid (small bobby)', 42, 2, 'Humans', 6, (0, 0, 142)),
        LostAndFoundClass('Kid (crawling)', 43, 2, 'Humans', 6, (0, 0, 142)),
    ]

    train_id_in = 1
    train_id_out = 2
    # cs = Cityscapes()
    # mean = cs.mean
    # std = cs.std
    num_eval_classes = 19

    def __init__(self, split='test', root="/home/yuyuan/work_space/fs_lost_and_found/", transform=None):
        assert os.path.exists(root), "lost&found valid not exists"
        """Load all filenames."""
        self.transform = transform
        self.root = root
        self.split = split      # ['test', 'train']
        self.images = []        # list of all raw input images
        self.targets = []       # list of all ground truth TrainIds images
        self.annotations = []   # list of all ground truth LabelIds images

        for root, _, filenames in os.walk(os.path.join(root, 'leftImg8bit', self.split)):
            for filename in filenames:
                if os.path.splitext(filename)[1] == '.png':
                    filename_base = '_'.join(filename.split('_')[:-1])
                    city = '_'.join(filename.split('_')[:-3])
                    self.images.append(os.path.join(root, filename_base + '_leftImg8bit.png'))
                    target_root = os.path.join(self.root, 'gtCoarse', self.split)
                    self.targets.append(os.path.join(target_root, city, filename_base + '_gtCoarse_labelTrainIds.png'))
                    self.annotations.append(os.path.join(target_root, city, filename_base + '_gtCoarse_labelIds.png'))

    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image and trainIds as PIL image or torch.Tensor"""
        image = Image.open(self.images[i]).convert('RGB')
        target = Image.open(self.targets[i]).convert('L')
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target


    def __repr__(self):
        """Return number of images in each dataset."""
        fmt_str = 'LostAndFound Split: %s\n' % self.split
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()


class Fishyscapes(Dataset):
    FishyscapesClass = namedtuple('FishyscapesClass', ['name', 'id', 'train_id', 'hasinstances',
                                                       'ignoreineval', 'color'])
    # --------------------------------------------------------------------------------
    # A list of all Lost & Found labels
    # --------------------------------------------------------------------------------
    labels = [
        FishyscapesClass('in-distribution', 0, 0, False, False, (144, 238, 144)),
        FishyscapesClass('out-distribution', 1, 1, False, False, (255, 102, 102)),
        FishyscapesClass('unlabeled', 2, 255, False, True, (0, 0, 0)),
    ]

    train_id_in = 0
    train_id_out = 1
    # cs = Cityscapes()
    # mean = cs.mean
    # std = cs.std
    num_eval_classes = 19
    label_id_to_name = {label.id: label.name for label in labels}
    train_id_to_name = {label.train_id: label.name for label in labels}
    trainid_to_color = {label.train_id: label.color for label in labels}
    label_name_to_id = {label.name: label.id for label in labels}

    def __init__(self, split='Static', root="", transform=None):
        """Load all filenames."""
        self.transform = transform
        self.root = root
        self.split = split  # ['Static', 'LostAndFound']
        self.images = []  # list of all raw input images
        self.targets = []  # list of all ground truth TrainIds images
        filenames = os.listdir(os.path.join(root, self.split, 'original'))
        root = os.path.join(root, self.split)
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.png':
                f_name = os.path.splitext(filename)[0]
                filename_base_img = os.path.join("original", f_name)
                filename_base_labels = os.path.join("labels", f_name)

                self.images.append(os.path.join(root, filename_base_img + '.png'))
                self.targets.append(os.path.join(root, filename_base_labels + '.png'))
        self.images = sorted(self.images)
        self.targets = sorted(self.targets)

    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image, trainIds as torch.Tensor or PIL Image"""
        image = Image.open(self.images[i]).convert('RGB')
        target = Image.open(self.targets[i]).convert('L')
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target

    def __repr__(self):
        """Print some information about dataset."""
        fmt_str = 'LostAndFound Split: %s\n' % self.split
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()



class RoadAnomaly(Dataset):
    RoadAnomaly_class = namedtuple('RoadAnomalyClass', ['name', 'id', 'train_id', 'hasinstances',
                                                        'ignoreineval', 'color'])
    # --------------------------------------------------------------------------------
    # A list of all Lost & Found labels
    # --------------------------------------------------------------------------------
    labels = [
        RoadAnomaly_class('in-distribution', 0, 0, False, False, (144, 238, 144)),
        RoadAnomaly_class('out-distribution', 1, 1, False, False, (255, 102, 102)),
    ]

    train_id_in = 0
    train_id_out = 1
    # cs = Cityscapes()
    # mean = cs.mean
    # std = cs.std
    num_eval_classes = 19
    label_id_to_name = {label.id: label.name for label in labels}
    train_id_to_name = {label.train_id: label.name for label in labels}
    trainid_to_color = {label.train_id: label.color for label in labels}
    label_name_to_id = {label.name: label.id for label in labels}

    def __init__(self, root="/home/yu/yu_ssd/road_anomaly", transform=None):
        """Load all filenames."""
        self.transform = transform
        self.root = root
        self.images = []  # list of all raw input images
        self.targets = []  # list of all ground truth TrainIds images
        filenames = os.listdir(os.path.join(root, 'original'))

        for filename in filenames:
            if os.path.splitext(filename)[1] == '.jpg':
                f_name = os.path.splitext(filename)[0]
                filename_base_img = os.path.join("original", f_name)
                filename_base_labels = os.path.join("labels", f_name)

                self.images.append(os.path.join(self.root, filename_base_img + '.jpg'))
                self.targets.append(os.path.join(self.root, filename_base_labels + '.png'))
        self.images = sorted(self.images)
        self.targets = sorted(self.targets)

    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image, trainIds as torch.Tensor or PIL Image"""
        image = Image.open(self.images[i]).convert('RGB')
        target = Image.open(self.targets[i]).convert('L')
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __repr__(self):
        """Print some information about dataset."""
        fmt_str = 'Road anomaly Dataset: \n'
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()

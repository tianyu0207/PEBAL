# Adapted from
# https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import logging
import random
import types
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as trans

from PIL import Image, ImageOps


class Compose(object):
    """Wraps together multiple image augmentations.

    Should also be used with only one augmentation, as it ensures, that input
    images are of type 'PIL.Image' and handles the augmentation process.

    Args:
        augmentations: List of augmentations to be applied.
    """

    def __init__(self, augmentations):
        """Initializes the composer with the given augmentations."""
        self.augmentations = augmentations

    def __call__(self, img, mask, *inputs):
        """Returns images that are augmented with the given augmentations."""
        # img, mask = Image.fromarray(img, mode='RGB'), Image.fromarray(mask, mode='L')
        assert img.size == mask.size
        for a in self.augmentations:
            img, mask, inputs = a(img, mask, *inputs)
        return (img, mask, *inputs)





class RandomCrop(object):
    """Returns an image of size 'size' that is a random crop of the original.

    Args:
        size: Size of the croped image.
        padding: Number of pixels to be placed around the original image.
    """

    def __init__(self, size, padding=0, *inputs, **kwargs):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask, *inputs, **kwargs):
        """Returns randomly cropped image."""
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
            inputs = tuple(ImageOps.expand(i, border=self.padding, fill=0) for i in inputs)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return (img.resize((tw, th), Image.BILINEAR),
                    mask.resize((tw, th), Image.NEAREST),
                    tuple(i.resize((tw, th), Image.NEAREST) for i in inputs))

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return (img.crop((x1, y1, x1 + tw, y1 + th)),
                mask.crop((x1, y1, x1 + tw, y1 + th)),
                tuple(i.crop((x1, y1, x1 + tw, y1 + th)) for i in inputs))


class CenterCrop(object):
    """Returns image of size 'size' that is center cropped.

    Crops an image of size 'size' from the center of an image. If the center
    index is not an integer, the value will be rounded.

    Args:
        size: The size of the output image.
    """

    def __init__(self, size, *inputs, **kwargs):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask, *inputs, **kwargs):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return (img.crop((x1, y1, x1 + tw, y1 + th)),
                mask.crop((x1, y1, x1 + tw, y1 + th)), *inputs)


class RandomHorizontalFlip(object):
    """Returns an image the got flipped with a probability of 'prob'.

    Args:
        prob: Probability with which the horizontal flip is applied.
    """

    def __init__(self, prob=0.5, *inputs, **kwargs):
        if not isinstance(prob, numbers.Number):
            raise TypeError("'prob' needs to be a number.")
        self.prob = prob

    def __call__(self, img, mask, *inputs, **kwargs):
        if random.random() < self.prob:
            return (img.transpose(Image.FLIP_LEFT_RIGHT),
                    mask.transpose(Image.FLIP_LEFT_RIGHT),
                    tuple(i.transpose(Image.FLIP_LEFT_RIGHT) for i in inputs))
        return img, mask, tuple(i for i in inputs)


class FreeScale(object):
    def __init__(self, size, *inputs, **kwargs):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask, *inputs, **kwargs):
        assert img.size == mask.size
        return (img.resize(self.size, Image.BILINEAR),
                mask.resize(self.size, Image.NEAREST), *inputs)


class Scale(object):
    def __init__(self, size, *inputs, **kwargs):
        self.size = size

    def __call__(self, img, mask, *inputs, **kwargs):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return (img, mask, *inputs)
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return (img.resize((ow, oh), Image.BILINEAR),
                    mask.resize((ow, oh), Image.NEAREST), *inputs)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return (img.resize((ow, oh), Image.BILINEAR),
                    mask.resize((ow, oh), Image.NEAREST), *inputs)


class RandomSizedCrop(object):
    def __init__(self, size, *inputs, **kwargs):
        self.size = size

    def __call__(self, img, mask, *inputs, **kwargs):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return (img.resize((self.size, self.size), Image.BILINEAR),
                        mask.resize((self.size, self.size), Image.NEAREST), *inputs)

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask, *inputs))


class RandomRotate(object):
    def __init__(self, degree, *inputs, **kwargs):
        if not isinstance(degree, numbers.Number):
            raise TypeError("'degree' needs to be a number.")
        self.degree = degree

    def __call__(self, img, mask, *inputs, **kwargs):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (img.rotate(rotate_degree, Image.BILINEAR),
                mask.rotate(rotate_degree, Image.NEAREST), *inputs)


class RandomSized(object):
    def __init__(self, size, min_scale=0.5, max_scale=2, *inputs, **kwargs):
        self.size = size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask, *inputs, **kwargs):
        assert img.size == mask.size

        w = int(random.uniform(self.min_scale, self.max_scale) * img.size[0])
        h = int(random.uniform(self.min_scale, self.max_scale) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return self.crop(*self.scale(img, mask, *inputs))


class RandomOcclusion(object):
    def __init__(self, build_prob=0.5, secondary_build_prob=0.99, occlusion_class=-1, start_points=5, min_size=100,
                 *inputs, **kwargs):
        self.log = logging.getLogger(__name__)
        if build_prob > 1 or build_prob < 0:
            self.log.error('build_prob has to be between 0 and 1!')
            raise ValueError('build_prob has to be between 0 and 1!')
        if secondary_build_prob > 1 or secondary_build_prob < 0:
            self.log.error('secondary_build_prob has to be between 0 and 1!')
            raise ValueError('secondary_build_prob has to be between 0 and 1!')
        self.build_prob = build_prob
        self.secondary_build_prob = secondary_build_prob
        self.occlusion_class = occlusion_class
        self.start_points = start_points
        self.min_size = min_size

    def __call__(self, img, mask, *inputs, **kwargs):
        while (mask == self.occlusion_class).sum() < self.min_size:
            self.queue = []
            self.flags = torch.full_like(mask, 0).byte()
            self.occlusion_map = torch.full_like(mask, 0).byte()
            self.img_height = img.shape[-2]
            self.img_width = img.shape[-1]

            # add first elements
            for _ in range(self.start_points):
                x = random.randint(0, self.img_height)
                y = random.randint(0, self.img_width)
                self.queue.append((x, y))
            while len(self.queue) > 0:
                i, j = self.queue.pop(0)
                self._scan_neighborhood(i, j)

            if self.occlusion_map.sum().item() >= self.min_size:
                for c in range(img.shape[0]):
                    img[c][self.occlusion_map] = 0
                mask[self.occlusion_map] = self.occlusion_class

        return (img, mask, *inputs)

    def _scan_neighborhood(self, i, j, *inputs, **kwargs):
        grid = [(i - 1, j - 1),
                (i - 1, j),
                (i - 1, j + 1),
                (i, j - 1),
                (i, j + 1),
                (i + 1, j - 1),
                (i + 1, j),
                (i + 1, j + 1)]
        if random.random() < self.build_prob:
            for ind in grid:
                if 0 <= ind[0] < self.img_height and 0 <= ind[1] < self.img_width:
                    if self.flags[ind] == 0 and random.random() < self.secondary_build_prob:
                        self.queue.append(ind)
                        self.occlusion_map[ind] = 1
                    self.flags[ind] = 1
        else:
            for ind in grid:
                if 0 <= ind[0] < self.img_height and 0 <= ind[1] < self.img_width:
                    self.flags[ind] = 1


class RandomNoise(object):
    def __init__(self, prob=0.5, ratio=0.1, *inputs, **kwargs):
        self.prob = prob
        self.ratio = ratio

    def __call__(self, image, mask, *inputs, **kwargs):
        if random.random() < self.prob:
            image = (1 - self.ratio) * image + self.ratio * torch.rand_like(image)
        return (image, mask, *inputs)


class RandomNoiseImage(object):
    def __init__(self, prob=0.05, class_index=-1, *inputs, **kwargs):
        self.prob = prob
        self.class_index = class_index

    def __call__(self, image, mask, *inputs, **kwargs):
        if random.random() < self.prob:
            image = torch.rand_like(image)
            mask = torch.full_like(mask, self.class_index)
        return (image, mask, *inputs)


class ToTensor(object):
    def __call__(self, image, mask, *inputs, **kwargs):
        t = trans.ToTensor()
        return (t(image), torch.tensor(np.array(mask, dtype=np.uint8), dtype=torch.long),
                tuple(torch.tensor(np.array(i, dtype=np.uint8), dtype=torch.long) for i in inputs))

    def __repr__(self, *inputs, **kwargs):
        return self.__class__.__name__ + '()'


class Normalize(object):
    def __init__(self, mean, std, *inputs, **kwargs):
        self.mean = mean
        self.std = std
        self.t = trans.Normalize(mean=self.mean, std=self.std)

    def __call__(self, tensor, mask, *inputs, **kwargs):
        return self.t(tensor), mask, tuple(i for i in inputs)


class DeStandardize(object):
    def __init__(self, mean, std, *inputs, **kwargs):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, mask, *inputs, **kwargs):
        for i in range(tensor.shape[0]):
            tensor[i] = tensor[i].mul(self.std[i]).add(self.mean[i])
        return (tensor, mask, *inputs)


class Lambda(object):
    def __init__(self, lambd, *inputs, **kwargs):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, mask, *inputs, **kwargs):
        return self.lambd(img, mask, *inputs)

    def __repr__(self, *inputs, **kwargs):
        return self.__class__.__name__ + '()'

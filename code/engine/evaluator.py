import collections
import time

import cv2
import numpy
import torch

from utils.metric import compute_score


class SlidingEval(torch.nn.Module):
    def __init__(self, config, device):
        super(SlidingEval, self).__init__()
        self.config = config
        self.device = device

    # slide the window to evaluate the image
    def forward(self, img, model, device=None):
        ori_rows, ori_cols, c = img.shape
        # fix to be 19, for inlier testing.
        num_class = 19
        processed_pred = numpy.zeros((ori_rows, ori_cols, num_class))

        # it is single scale
        multi_scales = self.config.eval_scale_array
        for s in multi_scales:
            img_scale = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            new_rows, new_cols, _ = img_scale.shape
            processed_pred += self.scale_process(img_scale, (ori_rows, ori_cols),
                                                 self.config.eval_crop_size, self.config.eval_stride_rate,
                                                 model, device)

        pred = processed_pred.argmax(2)
        return pred

    def process_image(self, img, crop_size=None):
        p_img = img

        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = numpy.concatenate((im_b, im_g, im_r), axis=2)

        # p_img = self.normalize(p_img, self.config.image_mean, self.config.image_std)

        if crop_size is not None:
            p_img, margin = self.pad_image_to_shape(p_img, crop_size,
                                                    cv2.BORDER_CONSTANT, value=0)
            p_img = p_img.transpose(2, 0, 1)

            return p_img, margin

        p_img = p_img.transpose(2, 0, 1)

        return p_img

    def val_func_process(self, input_data, val_func, device=None):
        input_data = numpy.ascontiguousarray(input_data[None, :, :, :], dtype=numpy.float32)
        input_data = torch.FloatTensor(input_data).cuda(device)

        with torch.cuda.device(input_data.get_device()):
            val_func.eval()
            val_func.to(input_data.get_device())
            with torch.no_grad():
                # modify for 19 classes
                start_time = time.time()
                score = val_func.module(input_data, output_anomaly=False)
                end_time = time.time()
                time_len = end_time - start_time
                # remove last reservation channel for OoD
                score = score.squeeze()[:19]

                # if self.config.eval_flip:
                #     input_data = input_data.flip(-1)
                #     score_flip = val_func(input_data)
                #     score_flip = score_flip[0]
                #     score += score_flip.flip(-1)

        return score, time_len

    def scale_process(self, img, ori_shape, crop_size, stride_rate, model, device=None):
        new_rows, new_cols, c = img.shape
        long_size = new_cols if new_cols > new_rows else new_rows

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        # remove last reservation channel for OoD
        class_num = 19
        if long_size <= min(crop_size[0], crop_size[1]):
            input_data, margin = self.process_image(img, crop_size)  # pad image
            score, _ = self.val_func_process(input_data, model, device)
            score = score[:class_num, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]
        else:
            stride_0 = int(numpy.ceil(crop_size[0] * stride_rate))
            stride_1 = int(numpy.ceil(crop_size[1] * stride_rate))
            img_pad, margin = self.pad_image_to_shape(img, crop_size, cv2.BORDER_CONSTANT, value=0)
            pad_rows = img_pad.shape[0]
            pad_cols = img_pad.shape[1]
            r_grid = int(numpy.ceil((pad_rows - crop_size[0]) / stride_0)) + 1
            c_grid = int(numpy.ceil((pad_cols - crop_size[1]) / stride_1)) + 1
            data_scale = torch.zeros(class_num, pad_rows, pad_cols).cuda(
                device)
            count_scale = torch.zeros(class_num, pad_rows, pad_cols).cuda(
                device)

            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride_1
                    s_y = grid_yidx * stride_0
                    e_x = min(s_x + crop_size[1], pad_cols)
                    e_y = min(s_y + crop_size[0], pad_rows)
                    s_x = e_x - crop_size[1]
                    s_y = e_y - crop_size[0]
                    img_sub = img_pad[s_y:e_y, s_x: e_x, :]
                    count_scale[:, s_y: e_y, s_x: e_x] += 1

                    input_data, tmargin = self.process_image(img_sub, crop_size)
                    temp_score, _ = self.val_func_process(input_data, model, device)
                    temp_score = temp_score[:class_num,
                                 tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                 tmargin[2]:(temp_score.shape[2] - tmargin[3])]

                    data_scale[:, s_y: e_y, s_x: e_x] += temp_score
            # score = data_scale / count_scale
            score = data_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)
        data_output = cv2.resize(score.cpu().numpy(), (ori_shape[1], ori_shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

        return data_output

    def compute_metric(self, results):
        hist = numpy.zeros((19, 19))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct, labeled)
        return mean_IU, mean_pixel_acc

    def pad_image_to_shape(self, img, shape, border_mode, value):
        margin = numpy.zeros(4, numpy.uint32)
        shape = self.get_2dshape(shape)
        pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
        pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0

        margin[0] = pad_height // 2
        margin[1] = pad_height // 2 + pad_height % 2
        margin[2] = pad_width // 2
        margin[3] = pad_width // 2 + pad_width % 2

        img = cv2.copyMakeBorder(img, margin[0], margin[1], margin[2], margin[3],
                                 border_mode, value=value)

        return img, margin

    def get_2dshape(self, shape, *, zero=True):
        if not isinstance(shape, collections.Iterable):
            shape = int(shape)
            shape = (shape, shape)
        else:
            h, w = map(int, shape)
            shape = (h, w)
        if zero:
            minv = 0
        else:
            minv = 1

        assert min(shape) >= minv, 'invalid shape: {}'.format(shape)
        return shape

    @staticmethod
    def normalize(img, mean, std):
        img = img.astype(numpy.float32) / 255.0
        img = img - mean
        img = img / std
        return img

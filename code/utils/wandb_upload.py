import os

import PIL
import matplotlib.pyplot as plt
import numpy
import seaborn as sns
import torch
import torchvision
import wandb
from matplotlib.lines import Line2D
from sklearn.preprocessing import minmax_scale as scaler


# from utils.visualize import show_img


def get_class_colors(*args):
    return [[128, 64, 128], [244, 35, 232], [70, 70, 70],
            [102, 102, 156], [190, 153, 153], [153, 153, 153],
            [250, 170, 30], [220, 220, 0], [107, 142, 35],
            [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0],
            [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
            [0, 0, 230], [119, 11, 32]]


def set_img_color(colors, background, img, pred):
    for i in range(0, len(colors)):
        img[numpy.where(pred == i)] = colors[i]

    if len(pred[pred > len(colors)]) > 0:
        # original color
        if numpy.any(pred == 255):
            img[numpy.where(pred == 255)] = [0, 0, 0]
        # change to pure white; class == 19
        else:
            img[numpy.where(pred > len(colors))] = [255, 255, 255]
    return img / 255.0


class Tensorboard:
    def __init__(self, config):
        os.environ['WANDB_API_KEY'] = config['wandb_key']
        os.system("wandb login")
        os.system("wandb {}".format("online" if config['wandb_online'] else "offline"))
        self.palette = get_class_colors()
        self.tensor_board = wandb.init(project=config['proj_name'],
                                       name=config['experiment_name'],
                                       config=config)
        self.restore_transform = torchvision.transforms.Compose([
            DeNormalize(config['image_mean'], config['image_std']),
            torchvision.transforms.ToPILImage()])
        self.visual_root_path = 'visuals/'
        if not os.path.exists(self.visual_root_path):
            os.mkdir(path=self.visual_root_path)

        if not os.path.exists(os.path.join(self.visual_root_path, 'confident_heat_map')):
            os.mkdir(path=os.path.join(self.visual_root_path, 'confident_heat_map'))

        if not os.path.exists(os.path.join(self.visual_root_path, 'distribution')):
            os.mkdir(path=os.path.join(self.visual_root_path, 'distribution'))

        if not os.path.exists(os.path.join(self.visual_root_path, 'prediction')):
            os.mkdir(path=os.path.join(self.visual_root_path, 'prediction'))

        if not os.path.exists(os.path.join(self.visual_root_path, 'sampling')):
            os.mkdir(path=os.path.join(self.visual_root_path, 'sampling'))

    def upload_wandb_info(self, current_step, info_dict):
        for i, info in enumerate(info_dict):
            self.tensor_board.log({info: info_dict[info],
                                   "global_step": current_step})
        return

    def upload_sampling_distribution(self, current_epoch, in_layer, out_layer,
                                     data_name="?"):
        assert len(in_layer) == len(out_layer), "expect same value of the in/out pixels"

        if not os.path.exists(os.path.join(self.visual_root_path,
                                           'distribution', 'epoch_{}'.format(str(current_epoch)))):
            os.mkdir(path=os.path.join(self.visual_root_path,
                                       'distribution', 'epoch_{}'.format(str(current_epoch))))

        custom_lines = [Line2D([0], [0], color="cyan", lw=4),
                        Line2D([0], [0], color="darkred", lw=4)]

        total_list = in_layer + out_layer
        total_list = scaler(total_list, feature_range=[0, 1])
        in_layer = total_list[:len(in_layer)]
        out_layer = total_list[len(in_layer):]
        x_axist_in = [i for i in range(0, len(in_layer))]
        x_axist_out = [i for i in range(len(in_layer), len(in_layer) + len(out_layer))]
        plt.bar(x_axist_in, in_layer, align='center', width=1, color="cyan")
        plt.bar(x_axist_out, out_layer, align='center', width=1, color="darkred")
        plt.grid()
        plt.legend(custom_lines, ['in_layer', 'out_layer'])
        plt.title(data_name + " epoch: " + str(current_epoch))
        plt.savefig(os.path.join(self.visual_root_path,
                                 "distribution", 'epoch_{}'.format(str(current_epoch)),
                                 "{}_dist_{}.png".format(data_name,
                                                         str(current_epoch))))
        plt.clf()
        self.tensor_board.log({"{} distribution".format(data_name):
                                   [wandb.Image(os.path.join(self.visual_root_path,
                                                             "distribution",
                                                             'epoch_{}'.format(str(current_epoch)),
                                                             "{}_dist_{}.png".format(data_name,
                                                                                     str(current_epoch))),
                                                caption="{}".format(data_name))]})
        return

    def upload_ood_image(self, current_step, energy_map, reserve_map=None, img_number=4, data_name="?"):
        self.tensor_board.log({"{}_focus_area_map".format(data_name):
                                   [wandb.Image(j, caption="id {}".format(str(i)))
                                    for i, j in enumerate(energy_map[:img_number])],
                               "global_step": current_step})

        # self.tensor_board.log({"{}_reserve_map".format(data_name):
        #                            [wandb.Image(j, caption="id {}".format(str(i)))
        #                             for i, j in enumerate(reserve_map[:img_number])],
        #                        "global_step": current_step})

        return

    def upload_mcmc_sampling(self, x_q, logits_q,
                             current_step, img_number=4, sample_size=None):
        upload_mcmc_output = []
        result = torch.argmax(logits_q, dim=1)

        if not os.path.exists(os.path.join(self.visual_root_path,
                                           'sampling', 'step_{}'.format(str(current_step)))):
            os.mkdir(path=os.path.join(self.visual_root_path,
                                       'sampling', 'step_{}'.format(str(current_step))))

        for i in range(0, img_number):
            clean_pad = numpy.zeros(sample_size)
            upload_mcmc_output.append(set_img_color(get_class_colors(), -1, clean_pad,
                                                    result[i].detach().cpu().numpy()))

        for i in range(min(img_number, x_q.shape[0])):
            plt.imshow(x_q[i][0].squeeze().cpu().numpy())
            plt.savefig(os.path.join(self.visual_root_path,
                                     "sampling", 'step_{}'.format(str(current_step)),
                                     "x_q_{}.png".format(str(i))))
            plt.clf()

            plt.imsave(os.path.join(self.visual_root_path,
                                    "sampling", 'step_{}'.format(str(current_step)),
                                    "logits_q_{}.png".format(str(i))), upload_mcmc_output[i])
            plt.clf()

        self.tensor_board.log({"logits_q_result": [wandb.Image(j, caption="id {}".format(str(i)))
                                                   for i, j in enumerate(upload_mcmc_output)]})

    def upload_wandb_image(self, current_step,
                           images, ground_truth,
                           prediction, img_number=4):
        img_number = min(ground_truth.shape[0], img_number)
        predict_mask_soft = torch.softmax(prediction, dim=1)
        predict_mask_hard = torch.argmax(predict_mask_soft, dim=1)
        predict_mask_soft = predict_mask_soft.max(1)[0]
        predict_mask_soft = predict_mask_soft.detach().cpu().numpy()

        if not os.path.exists(os.path.join(self.visual_root_path,
                                           'confident_heat_map', 'step_{}'.format(str(current_step)))):
            os.mkdir(path=os.path.join(self.visual_root_path,
                                       'confident_heat_map', 'step_{}'.format(str(current_step))))

        if not os.path.exists(os.path.join(self.visual_root_path,
                                           'prediction', 'step_{}'.format(str(current_step)))):
            os.mkdir(path=os.path.join(self.visual_root_path,
                                       'prediction', 'step_{}'.format(str(current_step))))

        upload_weight = []
        upload_prediction = []
        for i in range(0, img_number):
            clean_pad = numpy.zeros_like(images[0].permute(1, 2, 0).cpu().numpy())
            upload_prediction.append(set_img_color(get_class_colors(), -1, clean_pad,
                                                   predict_mask_hard[i].detach().cpu().numpy()))
        upload_ground_truth = []
        for i in range(0, img_number):
            clean_pad = numpy.zeros_like(images[0].permute(1, 2, 0).cpu().numpy())
            upload_ground_truth.append(set_img_color(get_class_colors(), -1, clean_pad,
                                                     ground_truth[i].detach().cpu().numpy()))

        for i in range(0, img_number):
            ax = sns.heatmap(predict_mask_soft[i], vmin=.0, vmax=1., yticklabels=False, xticklabels=False,
                             cbar=True if i == img_number - 1 else False)
            plt.savefig(self.visual_root_path + 'confident_heat_map/' +
                        'step_{}/heatmap_{}'.format(str(current_step), str(i)) + '.png')
            plt.clf()
            upload_weight.append(plt.imread(self.visual_root_path + 'confident_heat_map/' +
                                            'step_{}/heatmap_{}.png'.format(str(current_step), str(i))))

            plt.imsave(os.path.join(self.visual_root_path,
                                    "prediction", 'step_{}'.format(str(current_step)),
                                    "predict_{}.png".format(str(i))), upload_prediction[i])
            plt.clf()

        upload_weight = numpy.asarray(upload_weight)
        upload_prediction = numpy.asarray(upload_prediction)
        self.tensor_board.log({"confident_weight": [wandb.Image(j, caption="id {}".format(str(i)))
                                                    for i, j in enumerate(upload_weight)]})
        self.tensor_board.log({"prediction": [wandb.Image(j, caption="id {}".format(str(i)))
                                              for i, j in enumerate(upload_prediction)]})
        self.tensor_board.log({"ground_truth": [wandb.Image(j, caption="id {}".format(str(i)))
                                                for i, j in enumerate(upload_ground_truth)]})
        images = self.de_normalize(images[:img_number])
        self.tensor_board.log({"images": [wandb.Image(j, caption="id {}".format(str(i)))
                                          for i, j in enumerate(images)]})

    def de_normalize(self, image):
        return [self.restore_transform(i.detach().cpu()) if (isinstance(i, torch.Tensor) and len(i.shape) == 3)
                else colorize_mask(i.detach().cpu().numpy(), self.palette)
                for i in image]


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    palette[-3:] = [255, 255, 255]
    new_mask = PIL.Image.fromarray(mask.astype(numpy.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

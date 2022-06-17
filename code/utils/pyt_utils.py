# encoding: utf-8
import argparse
# from Code.furnace.engine.logger import get_logger
import logging
import os
import random
import time
from collections import Counter
from collections import OrderedDict

import sklearn.metrics as sk
import torch
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, auc

logger = logging.getLogger("cmd")
logger.propagate = False
recall_level_default = 0.95
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def counts_array_to_data_list(counts_array, max_size=None):
    if max_size is None:
        max_size = np.sum(counts_array)  # max of counted array entry
    counts_array = (counts_array / np.sum(counts_array) * max_size).astype("uint32")
    counts_dict = {}
    for i in range(1, len(counts_array) + 1):
        counts_dict[i] = counts_array[i - 1]
    return list(Counter(counts_dict).elements())


def calc_precision_recall(data, balance=False):
    if balance:
        x1 = counts_array_to_data_list(np.array(data["in"]), 1e+5)
        x2 = counts_array_to_data_list(np.array(data["out"]), 1e+5)
    else:
        ratio_in = np.sum(data["in"]) / (np.sum(data["in"]) + np.sum(data["out"]))
        ratio_out = 1 - ratio_in
        x1 = counts_array_to_data_list(np.array(data["in"]), 1e+7 * ratio_in)
        x2 = counts_array_to_data_list(np.array(data["out"]), 1e+7 * ratio_out)
    probas_pred1 = np.array(x1) / 100
    probas_pred2 = np.array(x2) / 100
    y_true = np.concatenate((np.zeros(len(probas_pred1)), np.ones(len(probas_pred2))))
    y_scores = np.concatenate((probas_pred1, probas_pred2))
    return precision_recall_curve(y_true, y_scores) + (average_precision_score(y_true, y_scores),)


def calc_sensitivity_specificity(data, balance=False):
    if balance:
        x1 = counts_array_to_data_list(np.array(data["in"]), max_size=1e+5)
        x2 = counts_array_to_data_list(np.array(data["out"]), max_size=1e+5)
    else:
        x1 = counts_array_to_data_list(np.array(data["in"]))
        x2 = counts_array_to_data_list(np.array(data["out"]))

    probas_pred1 = np.array(x1) / 100
    probas_pred2 = np.array(x2) / 100
    y_true = np.concatenate((np.zeros(len(probas_pred1)), np.ones(len(probas_pred2)))).astype("uint8")
    y_scores = np.concatenate((probas_pred1, probas_pred2))
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    return fpr, tpr, thresholds, auc(fpr, tpr)


def on_load_checkpoint(model, checkpoint: dict):
    if isinstance(checkpoint, str):
        state_dict = torch.load(checkpoint)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = checkpoint
    model_state_dict = model.state_dict()
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.' + k
        new_state_dict[name] = v
    state_dict = new_state_dict
    is_changed = False
    for k in state_dict:
        if k in model_state_dict:
            # print(state_dict[k].shape, model_state_dict[k].shape)
            if state_dict[k].shape != model_state_dict[k].shape:
                logger.warning(f"Skip loading parameter: {k}, "
                               f"required shape: {model_state_dict[k].shape}, "
                               f"pretrained shape: {state_dict[k].shape}")
                state_dict[k] = model_state_dict[k]
                logger.warning(f"change the shape to: {state_dict[k].shape}")
                # print(state_dict[k], value)
                # if k == 'module.branch.classifier.weight':
                #     state_dict[k][:, :-1, :, :] = value
                #     logger.warning("{} partial load the {} parameters ...".format(k,
                #                                                                   model_state_dict[k].shape))
                # if k == 'module.branch.classifier.bias':
                #     state_dict[k][:-1] = value
                #     logger.warning("{} partial load the {} parameters ...".format(k,
                #                                                                   model_state_dict[k].shape))
                is_changed = True
        else:
            logger.info(f"Dropping parameter {k}")
            is_changed = True

    if is_changed:
        checkpoint.pop("optimizer_states", None)

    model = model.load_state_dict(state_dict, strict=True)
    return model


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))  # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def get_and_print_results(out_score, in_score, num_to_avg=1):
    aurocs, auprs, fprs = [], [], []

    measures = get_measures(out_score, in_score)
    aurocs.append(measures[0]);
    auprs.append(measures[1]);
    fprs.append(measures[2])

    auroc = np.mean(aurocs);
    aupr = np.mean(auprs);
    fpr = np.mean(fprs)

    return auroc, aupr, fpr


import numpy as np


def get_metrics(flat_labels, flat_pred, num_points=50):
    # From fishycapes code
    pos = flat_labels == 1
    valid = flat_labels <= 1  # filter out void
    gt = pos[valid]
    del pos
    uncertainty = flat_pred[valid].reshape(-1).astype(np.float32, copy=False)
    del valid

    # Sort the classifier scores (uncertainties)
    sorted_indices = np.argsort(uncertainty, kind='mergesort')[::-1]
    uncertainty, gt = uncertainty[sorted_indices], gt[sorted_indices]
    del sorted_indices

    # Remove duplicates along the curve
    distinct_value_indices = np.where(np.diff(uncertainty))[0]
    threshold_idxs = np.r_[distinct_value_indices, gt.size - 1]
    del distinct_value_indices, uncertainty

    # Accumulate TPs and FPs
    tps = np.cumsum(gt, dtype=np.uint64)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    del threshold_idxs

    # Compute Precision and Recall
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    # stop when full recall attained and reverse the outputs so recall is decreasing
    sl = slice(tps.searchsorted(tps[-1]), None, -1)
    precision = np.r_[precision[sl], 1]
    recall = np.r_[recall[sl], 0]
    average_precision = -np.sum(np.diff(recall) * precision[:-1])

    # select num_points values for a plotted curve
    interval = 1.0 / num_points
    curve_precision = [precision[-1]]
    curve_recall = [recall[-1]]
    idx = recall.size - 1
    for p in range(1, num_points):
        while recall[idx] < p * interval:
            idx -= 1
        curve_precision.append(precision[idx])
        curve_recall.append(recall[idx])
    curve_precision.append(precision[0])
    curve_recall.append(recall[0])
    del precision, recall

    if tps.size == 0 or fps[0] != 0 or tps[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        tps = np.r_[0., tps]
        fps = np.r_[0., fps]

    # Compute TPR and FPR
    tpr = tps / tps[-1]
    del tps
    fpr = fps / fps[-1]
    del fps

    # Compute AUROC
    auroc = np.trapz(tpr, fpr)

    # Compute FPR@95%TPR
    fpr_tpr95 = fpr[np.searchsorted(tpr, 0.95)]
    results = {
        'auroc': auroc,
        'AP': average_precision,
        'FPR@95%TPR': fpr_tpr95,
        'recall': np.array(curve_recall),
        'precision': np.array(curve_precision),
        'fpr': fpr,
        'tpr': tpr
    }

    return results


def eval_ood_measure(conf, seg_label, train_id_in, train_id_out, mask=None):
    in_scores = conf[seg_label == train_id_in]
    out_scores = conf[seg_label == train_id_out]

    # in_scores = conf_in
    # out_scores = conf_out
    # temp = seg_label.copy()
    # temp[temp == train_id_in] = 0
    # temp[temp == train_id_out] = 0
    # in_scores = - conf[np.logical_not(out_label)]
    # out_scores = - conf[out_label]

    if (len(out_scores) != 0) and (len(in_scores) != 0):
        auroc, aupr, fpr = get_and_print_results(out_scores, in_scores)
        return auroc, aupr, fpr
    else:
        return None


def load_model(model, model_file, is_restore=False,
               strict=True, extra_channel=False):
    t_start = time.time()

    if model_file is None:
        return model

    if isinstance(model_file, str):
        state_dict = torch.load(model_file)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()

    if is_restore:
        model_state_dict = model.state_dict()
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k == "module.criterion.nll_loss.weight":
                continue
            if k == "module.dsn.4.0.bias" or k == "module.final2.0.bias" or k == "module.dsn.4.bias":
                continue
            # module.branch.final.6.weight"
            name = "module.branch1." + k.split('module.')[-1]
            # print(k, state_dict[k].shape)
            # print(name, model_state_dict[name].shape)
            new_state_dict[name] = v
            if k == "module.final.6.weight" and extra_channel:
                new_state_dict[name] = torch.nn.init.kaiming_normal_(torch.zeros([20, 256, 1, 1]))
                new_state_dict[name][:19, :, :, :] = v
                print(k, state_dict[k].shape)
                print(name, model_state_dict[name].shape, "partial loading ...")

            if k == "module.final2.0.weight" and extra_channel:
                new_state_dict[name] = torch.nn.init.kaiming_normal_(torch.zeros([20, 256, 1, 1]))
                new_state_dict[name][:19, :, :, :] = v
                print(k, state_dict[k].shape)
                print(name, model_state_dict[name].shape, "partial loading ...")

            # if k ==  "module.final2.0.bias" and extra_channel:
            #     new_state_dict[name] = torch.zeros([20])
            #     new_state_dict[name][:19] = v
            #     print(k, state_dict[k].shape)
            #     print(name, model_state_dict[name].shape, "partial loading ...")

            if k == "module.dsn.4.weight" and extra_channel:
                new_state_dict[name] = torch.nn.init.kaiming_normal_(torch.zeros([20, 512, 1, 1]))
                new_state_dict[name][:19, :, :, :] = v
                print(k, state_dict[k].shape)
                print(name, model_state_dict[name].shape, "partial loading ...")

            # if k ==  "module.dsn.4.0.bias" and extra_channel:
            #     new_state_dict[name] = torch.zeros([20])
            #     new_state_dict[name][:19] = v
            #     print(k, state_dict[k].shape)
            #     print(name, model_state_dict[name].shape, "partial loading ...")

        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=strict)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys
    del state_dict
    t_end = time.time()
    logger.critical(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))
    return model


def load_dualpath_model(model, model_file, is_restore=False):
    # load raw state_dict
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file)

        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file
    # copy to  hha backbone
    state_dict = {}
    for k, v in raw_state_dict.items():
        state_dict[k.replace('.bn.', '.')] = v
        if k.find('conv1') >= 0:
            state_dict[k] = v
            state_dict[k.replace('conv1', 'hha_conv1')] = v
        if k.find('conv2') >= 0:
            state_dict[k] = v
            state_dict[k.replace('conv2', 'hha_conv2')] = v
        if k.find('conv3') >= 0:
            state_dict[k] = v
            state_dict[k.replace('conv3', 'hha_conv3')] = v
        if k.find('bn1') >= 0:
            state_dict[k] = v
            state_dict[k.replace('bn1', 'hha_bn1')] = v
        if k.find('bn2') >= 0:
            state_dict[k] = v
            state_dict[k.replace('bn2', 'hha_bn2')] = v
        if k.find('bn3') >= 0:
            state_dict[k] = v
            state_dict[k.replace('bn3', 'hha_bn3')] = v
        if k.find('downsample') >= 0:
            state_dict[k] = v
            state_dict[k.replace('downsample', 'hha_downsample')] = v
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    # if len(missing_keys) > 0:
    #     logger.warning('Missing key(s) in state_dict: {}'.format(
    #         ', '.join('{}'.format(k) for k in missing_keys)))
    #
    # if len(unexpected_keys) > 0:
    #     logger.warning('Unexpected key(s) in state_dict: {}'.format(
    #         ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

    return model


def parse_devices(input_devices):
    if input_devices.endswith('*'):
        devices = list(range(torch.cuda.device_count()))
        return devices

    devices = []
    for d in input_devices.split(','):
        if '-' in d:
            start_device, end_device = d.split('-')[0], d.split('-')[1]
            assert start_device != ''
            assert end_device != ''
            start_device, end_device = int(start_device), int(end_device)
            assert start_device < end_device
            assert end_device < torch.cuda.device_count()
            for sd in range(start_device, end_device + 1):
                devices.append(sd)
        else:
            device = int(d)
            assert device < torch.cuda.device_count()
            devices.append(device)

    logger.info('using devices {}'.format(
        ', '.join([str(d) for d in devices])))

    return devices


def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def link_file(src, target):
    if os.path.isdir(target) or os.path.isfile(target):
        os.system('rm -rf {}'.format(target))
    os.system('ln -s {} {}'.format(src, target))


def ensure_dir(path):
    if not os.path.isdir(path):
        try:
            sleeptime = random.randint(0, 3)
            time.sleep(sleeptime)
            os.makedirs(path)
        except:
            print('conflict !!!')

# def load_model(model, model_file):
#     if isinstance(model_file, str):
#         print('Load Model: ' + model_file)
#         state_dict = torch.load(model_file)
#     else:
#         state_dict = model_file
#
#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = k
#         if k.split('.')[0] == 'module':
#             name = k[7:]
#         new_state_dict[name] = v
#     model.load_state_dict(new_state_dict, strict=False)
#
#     return model
#
#
# def parse_devices(input_devices):
#     if input_devices.endswith('*'):
#         devices = list(range(torch.cuda.device_count()))
#         return devices
#
#     devices = []
#     for d in input_devices.split(','):
#         if '-' in d:
#             start_device, end_device = d.split('-')[0], d.split('-')[1]
#             assert start_device != ''
#             assert end_device != ''
#             start_device, end_device = int(start_device), int(end_device)
#             assert start_device < end_device
#             assert end_device < torch.cuda.device_count()
#             for sd in range(start_device, end_device + 1):
#                 devices.append(sd)
#         else:
#             device = int(d)
#             assert device < torch.cuda.device_count()
#             devices.append(device)
#
#     return devices

#
# def inspect(var):
#     return CallbackInjector(var, _dbg_interactive)

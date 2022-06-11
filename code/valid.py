import warnings

import numpy as np
import torch.optim
from tqdm import tqdm

from utils.metric import hist_info
from utils.pyt_utils import eval_ood_measure

warnings.filterwarnings('ignore', '.*imshow.*', )


def valid_anomaly(model, test_set, data_name=None, epoch=None, my_wandb=None, logger=None,
                  upload_img_num=4):
    curr_info = {}
    model.eval()

    logger.info("validating {} dataset ...".format(data_name))
    tbar = tqdm(range(len(test_set)), ncols=137, leave=True)

    anomaly_score_list = []
    ood_gts_list = []
    focus_area = []

    with torch.no_grad():
        for idx in tbar:
            img, label = test_set[idx]
            anomaly_score = model.module(img, output_anomaly=True)
            anomaly_score = anomaly_score.cpu().numpy()
            ood_gts_list.append(np.expand_dims(label.detach().cpu().numpy(), 0))
            anomaly_score_list.append(np.expand_dims(anomaly_score, 0))
            if len(focus_area) < upload_img_num:
                anomaly_score[(label != test_set.train_id_out) & (label != test_set.train_id_in)] = 0
                focus_area.append(anomaly_score)

    # evaluation
    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    roc_auc, prc_auc, fpr = eval_ood_measure(anomaly_scores, ood_gts, test_set.train_id_in, test_set.train_id_out)

    curr_info['{}_auroc'.format(data_name)] = roc_auc
    curr_info['{}_fpr95'.format(data_name)] = fpr
    curr_info['{}_auprc'.format(data_name)] = prc_auc
    logger.critical(f'AUROC score for {data_name}: {roc_auc}')
    logger.critical(f'AUPRC score for {data_name}: {prc_auc}')
    logger.critical(f'FPR@TPR95 for {data_name}: {fpr}')

    if my_wandb is not None:
        my_wandb.upload_wandb_info(current_step=epoch, info_dict=curr_info)
        my_wandb.upload_ood_image(current_step=epoch, energy_map=focus_area, reserve_map=None,
                                  img_number=upload_img_num, data_name=data_name)

    del curr_info
    return roc_auc, prc_auc, fpr


def valid_epoch(model, engine, test_set, my_wandb, evaluator=None, logger=None, transform=None):
    model.eval()
    logger.info("validating cityscapes dataset ...")

    curr_info = {}
    all_results = []
    tbar = tqdm(range(0, len(test_set)), ncols=137, leave=True)

    with torch.no_grad():
        for idx in tbar:
            img, label = test_set[idx]
            img, label = img.permute(1, 2, 0).numpy(), label.numpy()
            pred = evaluator(img, model)
            hist_tmp, labeled_tmp, correct_tmp = hist_info(19, pred, label)
            results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}
            all_results.append(results_dict)

            if engine.local_rank <= 0:
                tbar.set_description(" labeled: {}, correct: {}".format(str(labeled_tmp), str(correct_tmp)))

        m_iou, m_acc = evaluator.compute_metric(all_results)
        curr_info['m_iou'] = m_iou
        curr_info['m_acc'] = m_acc

    logger.critical("current mIoU is {}, mAcc is {}".format(curr_info['m_iou'], curr_info['m_acc']))

    if my_wandb is not None:
        my_wandb.upload_wandb_info(info_dict=curr_info, current_step=0)

    return

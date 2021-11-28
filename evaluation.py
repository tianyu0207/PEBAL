import torch.optim
import numpy as np
import cv2
from tqdm import tqdm
from config import config
from utils.img_utils import pad_image_to_shape, normalize
from seg_opr.metric import hist_info, compute_score
from utils.pyt_utils import eval_ood_measure
import warnings

warnings.filterwarnings('ignore', '.*imshow.*', )

def compute_metric(results):
    hist = np.zeros((config.num_classes, config.num_classes))
    correct = 0
    labeled = 0
    count = 0
    for d in results:
        hist += d['hist']
        correct += d['correct']
        labeled += d['labeled']
        count += 1

    iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct,
                                                   labeled)
    # result_line = print_iou(iu, mean_pixel_acc,
    #                         test_set.get_class_names(), True)

    return mean_IU, mean_pixel_acc


def process_image(img, crop_size=None):
    p_img = img

    if img.shape[2] < 3:
        im_b = p_img
        im_g = p_img
        im_r = p_img
        p_img = np.concatenate((im_b, im_g, im_r), axis=2)

    p_img = normalize(p_img, config.image_mean, config.image_std)

    if crop_size is not None:
        p_img, margin = pad_image_to_shape(p_img, crop_size,
                                           cv2.BORDER_CONSTANT, value=0)
        p_img = p_img.transpose(2, 0, 1)

        return p_img, margin

    p_img = p_img.transpose(2, 0, 1)

    return p_img



def val_func_process(input_data, val_func, device=None):
    input_data = np.ascontiguousarray(input_data[None, :, :, :],
                                      dtype=np.float32)
    input_data = torch.FloatTensor(input_data).cuda(device)
    T = 1.

    with torch.cuda.device(input_data.get_device()):
        val_func.eval()
        val_func.to(input_data.get_device())
        with torch.no_grad():
            # modify for 19 classes
            score = val_func(input_data)
            # remove last reservation channel for OoD
            score = score.squeeze()[:19]
            if config.eval_flip:
                input_data = input_data.flip(-1)
                score_flip = val_func(input_data)
                score_flip = score_flip[0]
                score += score_flip.flip(-1)

    return score


def scale_process(img, ori_shape, crop_size, stride_rate, model, device=None):
    new_rows, new_cols, c = img.shape
    long_size = new_cols if new_cols > new_rows else new_rows

    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)

    if long_size <= min(crop_size[0], crop_size[1]):
        input_data, margin = process_image(img, crop_size)  # pad image
        score = val_func_process(input_data, model, device)
        score = score[:, margin[0]:(score.shape[1] - margin[1]),
                margin[2]:(score.shape[2] - margin[3])]
    else:
        stride_0 = int(np.ceil(crop_size[0] * stride_rate))
        stride_1 = int(np.ceil(crop_size[1] * stride_rate))
        img_pad, margin = pad_image_to_shape(img, crop_size,
                                             cv2.BORDER_CONSTANT, value=0)
        pad_rows = img_pad.shape[0]
        pad_cols = img_pad.shape[1]
        r_grid = int(np.ceil((pad_rows - crop_size[0]) / stride_0)) + 1
        c_grid = int(np.ceil((pad_cols - crop_size[1]) / stride_1)) + 1
        # remove last reservation channel for OoD
        class_num = 19
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

                input_data, tmargin = process_image(img_sub, crop_size)
                temp_score = val_func_process(input_data, model, device)
                temp_score = temp_score[:,
                             tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                             tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                data_scale[:, s_y: e_y, s_x: e_x] += temp_score
        # score = data_scale / count_scale
        score = data_scale
        score = score[:, margin[0]:(score.shape[1] - margin[1]),
                margin[2]:(score.shape[2] - margin[3])]

    score = score.permute(1, 2, 0)
    data_output = cv2.resize(score.cpu().numpy(),
                             (ori_shape[1], ori_shape[0]),
                             interpolation=cv2.INTER_LINEAR)

    return data_output


# slide the window to evaluate the image
def sliding_eval(img, crop_size, stride_rate, model, device=None):

    ori_rows, ori_cols, c = img.shape
    num_class = 19
    # remove last reservation channel for OoD
    processed_pred = np.zeros((ori_rows, ori_cols, num_class))
    multi_scales = config.eval_scale_array
    for s in multi_scales:
        img_scale = cv2.resize(img, None, fx=s, fy=s,
                               interpolation=cv2.INTER_LINEAR)

        new_rows, new_cols, _ = img_scale.shape
        processed_pred += scale_process(img_scale, (ori_rows, ori_cols),
                                        crop_size, stride_rate, model, device)
    pred = processed_pred.argmax(2)
    return pred


def valid_anomaly(model, test_set, device=None, data_name=None):

    model.eval()
    start_len = 0
    end_len = len(test_set)
    T = 1.
    tbar = tqdm(range(start_len, end_len))

    anomaly_score_list = []
    ood_gts_list = []

    with torch.no_grad():
        for idx in tbar:
            img, label = test_set[idx]

            anomaly_score = model(img, output_anomaly=True)
            anomaly_score = anomaly_score.cpu().numpy()
            ood_gts_list.append(np.expand_dims(label.detach().cpu().numpy(), 0))
            anomaly_score_list.append(np.expand_dims(anomaly_score, 0))

    # evaluation
    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    roc_auc, prc_auc, fpr = eval_ood_measure(anomaly_scores, ood_gts,
                                   test_set.train_id_in,
                                   test_set.train_id_out)

    print(f'AUROC score for {data_name}: {roc_auc}')
    print(f'AUPRC score for {data_name}: {prc_auc}')
    print(f'FPR@TPR95 for {data_name}: {fpr}')
    return roc_auc, prc_auc, fpr



def valid_city(model, test_set, device=None):
    model.eval()
    print("\n Validating city scape dataset mIoU, "
          "with {} images".format(str(int(len(test_set)))))

    all_results = []
    start_len = 0
    end_len = len(test_set)
    tbar = tqdm(range(start_len, end_len))
    with torch.no_grad():
        for idx in tbar:
            data = test_set[idx]
            img = data['data']
            label = data['label']
            pred = sliding_eval(img, config.eval_crop_size,
                                config.eval_stride_rate, model, device)

            hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes,
                                                           pred,
                                                           label)

            results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp,
                            'correct': correct_tmp}
            all_results.append(results_dict)

        m_iou, m_acc = compute_metric(all_results)

    print(f'mIoU score for Cityscape: {m_iou}')

    return m_iou, m_acc


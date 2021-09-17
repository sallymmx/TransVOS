import torch
from torch.nn import functional as F

import os
import shutil
import logging
import cv2
import numpy as np
from PIL import Image

import sys
import time
import pandas as pd
from davis2017.evaluation import DAVISEvaluation

logger = logging.getLogger(__name__)


def save_checkpoint(state, epoch, is_best, checkpoint_dir='./models', filename='checkpoint', thres=100):
    """
    - state
    - epoch
    - is_best
    - checkpoint_dir: default, ./models
    - filename: default, checkpoint
    - freq: default, 10
    - thres: default, 100
    """
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if epoch >= thres:
        file_path = os.path.join(checkpoint_dir, filename + '_{}'.format(str(epoch)) + '.pth.tar')
    else:
        file_path = os.path.join(checkpoint_dir, filename + '.pth.tar')
    torch.save(state, file_path)
    logger.info('==> save model at {}'.format(file_path))

    if is_best:
        cpy_file = os.path.join(checkpoint_dir, filename+'_model_best.pth.tar')
        shutil.copyfile(file_path, cpy_file)
        logger.info('==> save best model at {}'.format(cpy_file))
        

def mask_iou(pred, target, eps=1e-7, size_average=True):
    r"""
        param: 
            pred: size [N x H x W]
            target: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """

    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)

    inter = torch.min(pred, target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    if size_average:
        iou = torch.sum(inter / (union+eps)) / N
    else:
        iou = inter / (union + eps)

    return iou

class VideoWriter(object):
    def __init__(self, result_dir, video_name):
        self.video_dir = os.path.join(result_dir, video_name)
        if not os.path.exists(self.video_dir):
            os.mkdir(self.video_dir)

    def write(self, output_name, mask, original_size):
        rescale_mask = F.interpolate(mask, original_size)
        rescale_mask = torch.argmax(rescale_mask[0], dim=0).cpu().numpy().astype(np.uint8)
        im = Image.fromarray(rescale_mask).convert('P')
        im.putpalette(Image.open('datasets/mask_palette.png').getpalette())
        im.save(os.path.join(self.video_dir, output_name), format='PNG')


def write_mask(mask, info, result_dir):
    """
    mask: numpy.array of size [T x max_obj x H x W]
    """
    video_dir = os.path.join(result_dir, info['name'])
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
        
    rescale = False
    if 'original_size' in info.keys():
        rescale = True
        h, w = info['original_size']
        th, tw = mask.shape[2:]
        factor = min(th / h, tw / w)
        sh, sw = int(factor*h), int(factor*w)

        pad_l = (tw - sw) // 2
        pad_t = (th - sh) // 2

    is_ytb = False
    if 'basename_list' in info.keys():
        is_ytb = True
        basename_list = info['basename_list']
        basename_to_save = info['basename_to_save']

    for t in range(mask.shape[0]):
        if rescale:
            m = mask[t, :, pad_t:pad_t + sh, pad_l:pad_l + sw]
            m = m.transpose((1, 2, 0))
            rescale_mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            rescale_mask = mask[t]
            rescale_mask = rescale_mask.transpose((1, 2, 0))
        rescale_mask = rescale_mask.argmax(axis=2).astype(np.uint8)
        if is_ytb:
            if basename_list[t] in basename_to_save:
                output_name = basename_list[t] + '.png'
                im = Image.fromarray(rescale_mask).convert('P')
                im.putpalette(Image.open('datasets/mask_palette.png').getpalette())
                im.save(os.path.join(video_dir, output_name), format='PNG')
        else:
            output_name = '{:0>5d}.png'.format(t)
            im = Image.fromarray(rescale_mask).convert('P')
            im.putpalette(Image.open('datasets/mask_palette.png').getpalette())
            im.save(os.path.join(video_dir, output_name), format='PNG')


def davis2017_eval(results_path, davis_path, task='semi-supervised', set='val', version='2017'):
    time_start = time.time()
    print(f'Evaluating sequences for the {task} task...')
    # Create dataset and evaluate
    dataset_eval = DAVISEvaluation(davis_root=davis_path, task=task, gt_set=set, version=version)
    metrics_res = dataset_eval.evaluate(results_path)
    J, F = metrics_res['J'], metrics_res['F']
    
    # Path 
    csv_name_global = f'global_results-{version}{set}.csv'
    csv_name_per_sequence = f'per-sequence_results-{version}{set}.csv'
    csv_name_global_path = os.path.join(results_path, csv_name_global)
    csv_name_per_sequence_path = os.path.join(results_path, csv_name_per_sequence)

    # Generate dataframe for the general results
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"])])
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)
    with open(csv_name_global_path, 'a') as f:
        table_g.to_csv(f, index=False, float_format="%.3f")
    print(f'Global results saved in {csv_name_global_path}')

    # Generate a dataframe for the per sequence results
    seq_names = list(J['M_per_object'].keys())
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
    with open(csv_name_per_sequence_path, 'a') as f:
        table_seq.to_csv(f, index=False, float_format="%.3f")
    print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

    # Print the results
    sys.stdout.write(f"--------------------------- Global results for {set} ---------------------------\n")
    print(table_g.to_string(index=False))
    sys.stdout.write(f"\n---------- Per sequence results for {set} ----------\n")
    print(table_seq.to_string(index=False))
    total_time = time.time() - time_start
    sys.stdout.write('\nTotal time:' + str(total_time))
    
    return final_mean
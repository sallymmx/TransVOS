import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import os
import os.path as osp
import shutil
import time
import argparse
import random
from progress.bar import Bar
from collections import OrderedDict
import logging
import glob

from utils.system import (setup_logging, AverageMeter, setup_seed, set_bn_eval, match_name_keywords)
from utils.utility import *
from utils.losses import SegLoss
from config import cfg
from models.model import build_dtfvos
from datasets import (multibatch_collate_fn, build_pretrain, build_davis, build_ytbvos)


def parse_args():
    parser = argparse.ArgumentParser('Training Mask Segmentation')
    parser.add_argument('--gpu', default='0', type=str,
                        help='set gpu id to train the network, split with comma')
    parser.add_argument('--resume', default='', type=str, help='resume model path')
    parser.add_argument('--initial', default='', type=str, help='initial model path')
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--log_dir', default='./logs', type=str)
    parser.add_argument('--seed', default=1024, type=int)
    parser.add_argument('--exp_name', default='pretrain', type=str)
    return parser.parse_args()


def main(args):
    # Use GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpu_ids = range(torch.cuda.device_count())

    # Data
    logger.info('==> Preparing dataset')
    if args.pretrain:
        trainset = build_pretrain(cfg)
    else:
        data_list = list()
        data_ratio = cfg.DATA.TRAIN.DATASETS_RATIO
        for i, name in enumerate(cfg.DATA.TRAIN.DATASETS_NAME):
            if name == 'DAVIS17':
                data_list += data_ratio[i] * [build_davis(cfg)]
            elif name == 'YTBVOS':
                data_list += data_ratio[i] * [build_ytbvos(cfg)]
            else:
                raise NameError
        trainset = data.ConcatDataset(data_list)
                                     
    trainloader = data.DataLoader(trainset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, pin_memory=True,
                                  num_workers=cfg.TRAIN.NUM_WORKER, collate_fn=multibatch_collate_fn, drop_last=True)

    # Model
    logger.info("==> creating model")
    net = build_dtfvos(cfg)
    logger.info('==> Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1e6))
    net = net.cuda()
    net.train()
    net.apply(set_bn_eval)

    assert cfg.TRAIN.BATCH_SIZE % len(gpu_ids) == 0
    net = nn.DataParallel(net)

    # Strateges
    criterion = SegLoss().cuda()

    # Optimization
    param_dicts = [
        {
            "params":
                [p for n, p in net.named_parameters() if "backbone" not in n  and p.requires_grad],
            "lr": cfg.TRAIN.LR,
        },
        {
            "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # Resume
    minloss = float('inf')

    if args.resume:
        # Load checkpoint.
        logger.info('==> Resuming from checkpoint {}'.format(args.resume))
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'

        checkpoint = torch.load(args.resume)
        minloss = checkpoint['minloss']
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        skips = checkpoint['max_skip']
        try:
            if isinstance(skips, list):
                for idx, skip in enumerate(skips):
                    trainloader.dataset.datasets[idx].set_max_skip(skip)
            else:
                    trainloader.dataset.set_max_skip(skips)
        except:
            logger.warning('Initializing max skip fail')
    else:
        if args.initial:
            logger.info('==> Initialize model with weight file {}'.format(args.initial))
            weight = torch.load(args.initial)
            if isinstance(weight, OrderedDict):
                net.module.load_param(weight)
            else:
                net.module.load_param(weight['state_dict'])
        start_epoch = 0

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.TRAIN.SCHEDULER.STEP_SIZE, gamma=0.5, last_epoch=start_epoch-1)

    # Train
    for epoch in range(start_epoch, cfg.TRAIN.EPOCH):
        lr = scheduler.get_last_lr()[0]
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, cfg.TRAIN.EPOCH, lr))
        train_loss, loss_stats = train(trainloader,
            model=net,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            max_norm=cfg.TRAIN.CLIP_MAX_NORM)

        # append logger file
        log_format = 'Epoch: {}, LR: {}, Loss: {}, Cls Loss: {}, Iou Loss: {}'
        logger.info(log_format.format(epoch+1, lr, train_loss, loss_stats['cls_loss'], loss_stats['iou_loss']))

        # adjust max skip
        if (epoch + 1) % cfg.DATA.TRAIN.EPOCH_PER_INCREMENT == 0:
            if isinstance(trainloader.dataset, data.ConcatDataset):
                for dataset in trainloader.dataset.datasets:
                    dataset.increase_max_skip()
            else:
                trainloader.dataset.increase_max_skip()

        # save model
        is_best = train_loss <= minloss
        minloss = min(minloss, train_loss)
        skips = [ds.max_skip for ds in trainloader.dataset.datasets] \
            if isinstance(trainloader.dataset, data.ConcatDataset) \
            else trainloader.dataset.max_skip

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'loss': train_loss,
            'minloss': minloss,
            'optimizer': optimizer.state_dict(),
            'max_skip': skips,
        }, epoch+1, is_best, checkpoint_dir=args.checkpoint_dir, thres=0)

        scheduler.step()

    logger.info('minimum loss: {}'.format(minloss))


def train(trainloader, model, criterion, optimizer, epoch, max_norm):
    # switch to train mode
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    iou_loss_meter = AverageMeter()

    end = time.time()

    bar = Bar('Processing', max=len(trainloader))

    for batch_idx, data in enumerate(trainloader):
        frames, masks, objs, infos = data
        # measure data loading time
        data_time.update(time.time() - end)

        frames = frames.cuda()
        masks = masks.cuda()
        objs = objs.cuda()

        optimizer.zero_grad()
        out = model(frames=frames, obj_masks=masks, n_objs=objs)
        loss, loss_stats = criterion(out, masks[:,1:], objs)
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # record loss
        if loss.item() > 0.0:
            loss_meter.update(loss.item(), 1)
            cls_loss_meter.update(loss_stats['cls_loss'].item(), 1)
            iou_loss_meter.update(loss_stats['iou_loss'].item(), 1)

        # measure elapsed time
        end = time.time()

        # plot progress
        plot_format = '({batch}/{size})Data:{data:.3f}s|Loss:{loss_val:.5f}({loss_avg:.5f})|Cls Loss:{cls_val:.5f}({cls_avg:.5f})|Iou Loss:{iou_val:.5f}({iou_avg:.5f})'
        plot_info = {
            'batch': batch_idx + 1,
            'size': len(trainloader),
            'data': data_time.val,
            'loss_val': loss_meter.val,
            'loss_avg': loss_meter.avg,
            'cls_val': cls_loss_meter.val,
            'cls_avg': cls_loss_meter.avg,
            'iou_val': iou_loss_meter.val,
            'iou_avg': iou_loss_meter.avg
        }
        bar.suffix = plot_format.format(**plot_info)
        bar.next()
    bar.finish()

    loss_avg_stats = {
            'cls_loss': cls_loss_meter.avg,
            'iou_loss': iou_loss_meter.avg
        }

    return loss_meter.avg, loss_avg_stats


if __name__ == '__main__':
    args = parse_args()

    # Set seed
    setup_seed(args.seed)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Logs
    prefix = args.exp_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    args.log_dir = log_dir

    # Save scripts
    script_path = os.path.join(log_dir, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path)

    scripts_to_save = ['train.py', 'config.py']
    scripts_to_save +=  list(glob.glob(os.path.join('datasets', '*.py')))
    scripts_to_save +=  list(glob.glob(os.path.join('models', '*.py')))
    scripts_to_save +=  list(glob.glob(os.path.join('utils', '*.py')))
    for script in scripts_to_save:
        dst_path = os.path.join(script_path, script)
        try:
            shutil.copy(script, dst_path)
        except IOError:
            os.makedirs(os.path.dirname(dst_path))
            shutil.copy(script, dst_path)

    # Checkpoints directory
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    args.checkpoint_dir = checkpoint_dir

    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    setup_logging(filename=os.path.join(log_path, 'log.txt'), resume=args.resume != '')
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {}'.format(args.exp_name))

    main(args)

import torch
import torch.utils.data as data

import numpy as np
import os
import time
import argparse
import logging

from utils.system import setup_logging, AverageMeter
from utils.utility import *
from config import cfg
from models.model import build_dtfvos
from datasets import (multibatch_collate_fn, build_davis, build_ytbvos)


def parse_args():
    parser = argparse.ArgumentParser('Testing Mask Segmentation')
    parser.add_argument('--checkpoint', default='', type=str, help='checkpoint to test the network')
    parser.add_argument('--results', default='results', type=str, help='result directory')
    parser.add_argument('--gpu', default='0', type=str, help='set gpu id to test the network')

    return parser.parse_args()

def main(args):
    # Use CUDA
    use_gpu = torch.cuda.is_available() and int(args.gpu) >= 0
    device = 'cuda:{}'.format(args.gpu) if use_gpu else 'cpu'
    
    # Data
    print('==> Preparing dataset')
    data_name = cfg.DATA.VAL.DATASET_NAME
    if data_name in ['DAVIS16', 'DAVIS17']:
        testset = build_davis(cfg, train=False)
    else:
        raise NameError
    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.TRAIN.NUM_WORKER, collate_fn=multibatch_collate_fn)

    # Model
    print("==> creating model")
    net = build_dtfvos(cfg)
    print('==> Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))

    # set eval to freeze batchnorm update
    net.eval()
    net.to(device)

    # set testing parameters
    for p in net.parameters():
        p.requires_grad = False

    # Weights
    if args.checkpoint:
        # Load checkpoint.
        logger.info('==> Loading checkpoint {}'.format(args.checkpoint))
        assert os.path.isfile(args.checkpoint), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state = checkpoint['state_dict']
        epoch = checkpoint['epoch']
        net.load_param(state)

    # Test
    print('==> Runing model on dataset, totally {:d} videos'.format(len(testloader)))

    test(testloader,
        model=net,
        device=device)

    print('==> Results are saved at: {}'.format(args.results))

    # Eval DAVIS17
    if data_name == 'DAVIS17':
        res = davis2017_eval(args.results, cfg.DATA.DAVIS_ROOT, version='2017')
        logger.info('Epoch: {}, J&F(DAVIS17): {}'.format(epoch, res))
    elif data_name == 'DAVIS16':
        res = davis2017_eval(args.results, cfg.DATA.DAVIS_ROOT, version='2016')
        logger.info('Epoch: {}, J&F(DAVIS16): {}'.format(epoch, res))
    

def test(testloader, model, device):

    data_time = AverageMeter()
    frame_cnt = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            frames, masks, objs, infos = data

            frames = frames.to(device)
            masks = masks.to(device)
                
            frames = frames[0]
            masks = masks[0]
            n_obj = objs[0]
            info = infos[0]

            original_size = info['original_size']
            
            T, _, _, _ = frames.shape

            print('==>Runing video {}, objects {:d}'.format(info['name'], n_obj-1))
            vid_writer = VideoWriter(args.results, info['name'])
            # process reference pairs
            first_pairs = {
                'frame': frames[0:1],  # [1 x 3 x H x W]
                'obj_mask': masks[0:1, 1:n_obj] # [1 x no x H x W]
            }
            first_feats_dict = model.extract_ref_feats(first_pairs)

            # segment frames
            vid_writer.write(f'{0:05d}.png', masks[0:1], original_size)
            for i in range(1, T):
                tic = time.time()
                if i > 1:
                    previous_feats_dict = model.extract_ref_feats(previous_pairs)
                    ref_feats_dict = model.concat_features([first_feats_dict, previous_feats_dict])
                    # if (i-1) % 5 == 0:
                    #     first_feats_dict = model.concat_features([first_feats_dict, previous_feats_dict])
                else:
                    ref_feats_dict = first_feats_dict
                seg_feats_dict, median_layers = model.extract_seg_feats(frames[i:i+1])
                seq_dict = model.concat_features([ref_feats_dict, seg_feats_dict], expand=True)

                hs, enc_mem = model.forward_transformer(seq_dict)
                logits = model.segment(hs, enc_mem, seq_dict, median_layers, masks[0:1])  # [1, M, H, W]
                out = torch.softmax(logits, dim=1)

                previous_pairs = {
                    'frame': frames[i:i+1],
                    'obj_mask': out[:, 1:n_obj]
                }
                toc = time.time()
                data_time.update(toc-tic, n=1)
                vid_writer.write(f'{i:05d}.png', out, original_size)

            frame_cnt += T-1

        logger.info("Global FPS:{:.1f}".format(frame_cnt/data_time.sum))

    return


if __name__ == '__main__':
    args = parse_args()

    data_name = cfg.DATA.VAL.DATASET_NAME
    print('==> Test dataset: {}'.format(cfg.DATA.VAL.DATASET_NAME))
    args.results = os.path.join(args.results, data_name)
    print('==> Save directory: {}'.format(args.results))
    if not os.path.exists(args.results):
        os.makedirs(args.results)

    setup_logging(filename=os.path.join(args.results, 'result.txt'), resume=True)
    logger = logging.getLogger(__name__)

    if os.path.isdir(args.checkpoint):
        ckp_dir = args.checkpoint
        ckp_list = os.listdir(ckp_dir)
        for ckp in ckp_list:
            args.checkpoint=os.path.join(ckp_dir, ckp_list)
            main(args)
    else:
        main(args)

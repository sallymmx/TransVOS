import torch
import torch.utils.data as data

import os
import argparse

from utils.utility import *
from config import cfg
from models.model import build_dtfvos
from datasets import multibatch_collate_fn
from datasets.davis import DAVIS_Test
from datasets.youtubevos import YouTube_Test


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
    data_name = cfg.DATA.TEST.DATASET_NAME
    if data_name == 'DAVIS17':
        testset = DAVIS_Test(os.path.join(cfg.DATA.DAVIS_ROOT, 'DAVIS-test-dev'), img_set='2017/test-dev.txt')
    elif data_name == 'YTBVOS':
        testset = YouTube_Test(cfg.DATA.YTBVOS_ROOT, (480, 854))
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
        net.load_param(state)

    # Test
    print('==> Runing model on dataset, totally {:d} videos'.format(len(testloader)))

    if data_name == 'DAVIS17':
        test_DAVIS(
            testloader,
            model=net,
            device=device)
    elif data_name == 'YTBVOS':
        test_YouTube(
            testloader,
            model=net,
            device=device)
    else:
        raise NotImplementedError()

    print('==> Results are saved at: {}'.format(args.results))
    

def test_DAVIS(testloader, model, device):
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
            # process reference pairs
            first_pairs = {
                'frame': frames[0:1],  # [1 x 3 x H x W]
                'obj_mask': masks[0:1, 1:n_obj] # [1 x no x H x W]
            }
            first_feats_dict = model.extract_ref_feats(first_pairs)

            # segment frames
            vid_writer = VideoWriter(args.results, info['name'])
            vid_writer.write(f'{0:05d}.png', masks[0:1], original_size)
            for i in range(1, T):
                if i > 1:
                    previous_feats_dict = model.extract_ref_feats(previous_pairs)
                    ref_feats_dict = model.concat_features([first_feats_dict, previous_feats_dict])
                    # if i == 5:
                    #     first_feats_dict = model.concat_features([first_feats_dict, previous_feats_dict])
                else:
                    ref_feats_dict = first_feats_dict
                seg_feats_dict, median_layers = model.extract_seg_feats(frames[i:i+1])
                seq_dict = model.concat_features([ref_feats_dict, seg_feats_dict], expand=True)

                hs, enc_mem = model.forward_transformer(seq_dict)
                logits = model.segment(hs, enc_mem, seq_dict, median_layers, masks[0:1])  # [1, M, H, W]
                out = torch.softmax(logits, dim=1)
                vid_writer.write(f'{i:05d}.png', out, original_size)

                previous_pairs = {
                    'frame': frames[i:i+1],
                    'obj_mask': out[:, 1:n_obj]
                }


def test_YouTube(testloader, model, device):
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            frames, masks, objs, infos = data

            frames = frames.to(device)
            masks = masks.to(device)
                
            frames = frames[0]
            masks = masks[0]
            obj_n = objs[0]
            info = infos[0]
            
            T, _, _, _ = frames.shape
            print('==>Runing video {}, objects {:d}'.format(info['name'], obj_n-1))

            obj_n = obj_n.item()
            obj_st = info['obj_st']
            obj_vis = info['obj_vis']
            vid_name = info['name']
            original_size = info['original_size']
            basename_list = info['basename_list']
            basename_to_save = info['basename_to_save']

            # Compose the first mask
            pred_mask = torch.zeros_like(masks).unsqueeze(0).float()
            for i in range(1, obj_n):
                if obj_st[i] == 0:
                    pred_mask[0, i] = masks[i]
            pred_mask[0, 0] = 1 - pred_mask.sum(dim=1)

            # process reference pairs
            first_pairs = {
                'frame': frames[0:1],  # [1 x 3 x H x W]
                'obj_mask': pred_mask[0:1, 1:obj_n] # [1 x no x H x W]
            }
            previous_pairs = None

            vid_writer = VideoWriter(args.results, vid_name)
            vid_writer.write(basename_list[0]+'.png', pred_mask, original_size)
            for t in range(1, T):
                if previous_pairs is not None:
                    previous_feats_dict = model.extract_ref_feats(previous_pairs)
                    ref_feats_dict = model.concat_features([first_feats_dict, previous_feats_dict])
                else:
                    first_feats_dict = model.extract_ref_feats(first_pairs)
                    ref_feats_dict = first_feats_dict
                seg_feats_dict, median_layers = model.extract_seg_feats(frames[t:t+1])
                seq_dict = model.concat_features([ref_feats_dict, seg_feats_dict], expand=True)

                hs, enc_mem = model.forward_transformer(seq_dict)
                score = model.segment(hs, enc_mem, seq_dict, median_layers, masks.unsqueeze(0))  # [1, M, H, W]

                reset_list = list()
                for i in range(1, obj_n):
                    # If this object is invisible.
                    if obj_vis[t, i] == 0:
                        score[0, i] = -1000
                    # If this object appears, reset the score map
                    if obj_st[i] == t:
                        reset_list.append(i)
                        score[0, i] = -1000
                        score[0, i][masks[i]] = 1000
                        for j in range(obj_n):
                            if j != i:
                                score[0, j][masks[i]] = -1000

                pred_mask = torch.softmax(score, dim=1)
                if basename_list[t] in basename_to_save:
                    vid_writer.write(basename_list[t]+'.png', pred_mask, original_size)

                if len(reset_list) > 0:
                    first_pairs = {
                        'frame': frames[t:t+1],  # [1 x 3 x H x W]
                        'obj_mask': pred_mask[0:1, 1:obj_n] # [1 x no x H x W]
                    }
                    previous_pairs = None
                else:
                    previous_pairs = {
                        'frame': frames[t:t+1],
                        'obj_mask': pred_mask[0:1, 1:obj_n]
                    }


if __name__ == '__main__':
    args = parse_args()

    data_name = cfg.DATA.TEST.DATASET_NAME
    if data_name == 'DAVIS17':
        data_name = 'DAVIS17-test-dev'
    print('==> Test dataset: {}'.format(data_name))
    args.results = os.path.join(args.results, data_name)
    print('==> Save directory: {}'.format(args.results))

    if not os.path.exists(args.results):
        os.makedirs(args.results)

    main(args)

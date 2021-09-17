import os
import random
import json
import numpy as np
from glob import glob
from itertools import compress

import torch
from torch.utils import data
import torchvision.transforms as TF

import datasets.transform as mytrans
from utils.system import load_image_in_PIL, gct
import matplotlib.pyplot as plt
from datasets.data_utils import multibatch_collate_fn, convert_one_hot


class YouTube_Train(data.Dataset):
    MAX_TRAINING_SKIP = 100
    def __init__(self, root, output_size, dataset_file='train/meta.json', clip_n=3, max_obj_n=11,
    max_skip=3, increment=1, samples=2, choice='order', crop=False):
        self.root = root
        self.clip_n = clip_n
        self.output_size = output_size
        self.max_obj_n = max_obj_n
        self.max_skip = max_skip
        self.increment = increment
        self.samples = samples
        self.sample_choice = choice
        self.crop = crop

        dataset_path = os.path.join(root, dataset_file)
        with open(dataset_path, 'r') as json_file:
            meta_data = json.load(json_file)

        self.dataset_list = list(meta_data['videos'])
        print(f'\t"YouTubeVOS": {len(self.dataset_list)} videos.')

        self.random_horizontal_flip = mytrans.RandomHorizontalFlip(0.3)
        self.color_jitter = TF.ColorJitter(0.1, 0.1, 0.1, 0.02)
        self.random_affine = mytrans.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=10)
        if self.crop:
            self.random_resize_crop = mytrans.RandomResizedCrop(400, (0.3, 0.5), (0.95, 1.05))
        else:
            self.resize = mytrans.Resize(output_size)
        self.to_tensor = TF.ToTensor()
        self.normalize = TF.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.to_onehot = mytrans.ToOnehot(max_obj_n, shuffle=True)

    def increase_max_skip(self):
        self.max_skip = min(self.max_skip + self.increment, self.MAX_TRAINING_SKIP)

    def set_max_skip(self, max_skip):
        self.max_skip = max_skip

    def __len__(self):
        return len(self.dataset_list) * self.samples

    def __getitem__(self, idx):
        video_name = self.dataset_list[idx//self.samples]
        img_dir = os.path.join(self.root, 'train/JPEGImages', video_name)
        mask_dir = os.path.join(self.root, 'train/Annotations', video_name)

        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

        img_n = len(img_list)

        obj_n = 1
        while obj_n == 1:
            if self.sample_choice == 'order':
                idx_list = list()
                last_sample = -1
                sample_n = min(self.clip_n, img_n)
                for i in range(sample_n):
                    if i == 0:
                        last_sample = random.choice(range(0, img_n-sample_n+1))
                    else:
                        last_sample = random.choice(
                            range(last_sample+1, min(last_sample+self.max_skip+1, img_n-sample_n+i+1)))
                    idx_list.append(last_sample)
            elif self.sample_choice == 'random':
                idx_list = list(range(img_n))
                random.shuffle(idx_list)
                sample_n = min(self.clip_n, img_n)
                idx_list = idx_list[:sample_n]
            else:
                raise NotImplementedError()
            while len(idx_list) < self.clip_n:
                idx_list.append(idx_list[-1])

            if not self.crop:
                frames = torch.zeros((self.clip_n, 3, *self.output_size), dtype=torch.float)
                masks = torch.zeros((self.clip_n, self.max_obj_n, *self.output_size), dtype=torch.float)
            else:
                frames = torch.zeros((self.clip_n, 3, 400, 400), dtype=torch.float)
                masks = torch.zeros((self.clip_n, self.max_obj_n, 400, 400), dtype=torch.float)

            for i, frame_idx in enumerate(idx_list):
                img = load_image_in_PIL(img_list[frame_idx], 'RGB')
                mask = load_image_in_PIL(mask_list[frame_idx], 'P')

                if i > 0:
                    img = self.color_jitter(img)
                    img, mask = self.random_affine(img, mask)

                if self.crop:
                    img, mask = self.random_resize_crop(img, mask)
                else:
                    img, mask = self.resize(img, mask)             
                mask = np.array(mask, np.uint8)

                if i == 0:
                    mask, obj_list = self.to_onehot(mask)
                    obj_n = len(obj_list) + 1
                else:
                    mask, _ = self.to_onehot(mask, obj_list)

                frames[i] = self.normalize(self.to_tensor(img))
                masks[i] = mask

            info = {
                'name': video_name,
                'idx_list': idx_list
            }

        return frames, masks, obj_n, info


class YouTube_Test(data.Dataset):
    def __init__(self, root, output_size=(495, 880), dataset_file='valid/meta.json', max_obj_n=11):
        self.root = root
        self.max_obj_n = max_obj_n
        self.out_h, self.out_w = output_size

        dataset_path = os.path.join(root, dataset_file)
        with open(dataset_path, 'r') as json_file:
            self.meta_data = json.load(json_file)

        self.dataset_list = list(self.meta_data['videos'])
        self.dataset_size = len(self.dataset_list)

        self.to_tensor = TF.ToTensor()
        self.normalize = TF.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.to_onehot = mytrans.ToOnehot(max_obj_n, shuffle=False)

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        video_name = self.dataset_list[idx]

        img_dir = os.path.join(self.root, 'valid/JPEGImages', video_name)
        mask_dir = os.path.join(self.root, 'valid/Annotations', video_name)

        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        basename_list = [os.path.basename(x)[:-4] for x in img_list]
        video_len = len(img_list)
        selected_idx = np.ones(video_len, np.bool)

        objs = self.meta_data['videos'][video_name]['objects']
        obj_n = 1
        video_obj_appear_st_idx = video_len

        for obj_idx, obj_gt in objs.items():
            obj_n = max(obj_n, int(obj_idx) + 1)
            video_obj_appear_idx = basename_list.index(obj_gt['frames'][0])
            video_obj_appear_st_idx = min(video_obj_appear_st_idx, video_obj_appear_idx)

        selected_idx[:video_obj_appear_st_idx] = False
        selected_idx = selected_idx.tolist()

        img_list = list(compress(img_list, selected_idx))
        basename_list = list(compress(basename_list, selected_idx))

        video_len = len(img_list)
        obj_vis = np.zeros((video_len, obj_n), np.uint8)
        obj_vis[:, 0] = 1
        obj_st = np.zeros(obj_n, np.uint8)

        tmp_img = load_image_in_PIL(img_list[0], 'RGB')
        original_w, original_h = tmp_img.size
        if original_h < self.out_h:
            out_h, out_w = original_h, original_w
        else:
            out_h = self.out_h
            out_w = int(original_w / original_h * self.out_h)
        # out_h = self.out_h
        # out_w = self.out_w
        masks = torch.zeros((obj_n, out_h, out_w), dtype=torch.bool)

        basename_to_save = list()
        for obj_idx, obj_gt in objs.items():
            obj_idx = int(obj_idx)
            basename_to_save += obj_gt['frames']

            frame_idx = basename_list.index(obj_gt['frames'][0])
            obj_st[obj_idx] = frame_idx
            obj_vis[frame_idx:, obj_idx] = 1

            mask_path = os.path.join(mask_dir, obj_gt['frames'][0] + '.png')
            mask_raw = load_image_in_PIL(mask_path, 'P')
            mask_raw = mask_raw.resize((out_w, out_h))
            mask_raw = torch.from_numpy(np.array(mask_raw, np.uint8))

            masks[obj_idx, mask_raw == obj_idx] = 1

        basename_to_save = sorted(list(set(basename_to_save)))

        frames = torch.zeros((video_len, 3, out_h, out_w), dtype=torch.float)
        for i in range(video_len):
            img = load_image_in_PIL(img_list[i], 'RGB')
            img = img.resize((out_w, out_h))
            frames[i] = self.normalize(self.to_tensor(img))

        info = {
            'name': video_name,
            'num_frames': video_len,
            'obj_vis': obj_vis,
            'obj_st': obj_st,
            'basename_list': basename_list,
            'basename_to_save': basename_to_save,
            'original_size': (original_h, original_w)
        }

        return frames, masks, obj_n, info


def build_ytbvos(cfg, train=True):
    if train:
        return YouTube_Train(
            root=cfg.DATA.YTBVOS_ROOT,
            output_size=cfg.DATA.SIZE,
            clip_n=cfg.DATA.TRAIN.FRAMES_PER_CLIP,
            max_obj_n=cfg.DATA.TRAIN.MAX_OBJECTS,
            max_skip=cfg.DATA.TRAIN.YTBVOS_SKIP_INCREMENT[0],
            increment=cfg.DATA.TRAIN.YTBVOS_SKIP_INCREMENT[1],
            samples=cfg.DATA.TRAIN.SAMPLES_PER_VIDEO,
            choice=cfg.DATA.TRAIN.SAMPLE_CHOICE,
            crop=cfg.DATA.TRAIN.CROP
        )
    else:
        return YouTube_Test(
            root=cfg.DATA.YTBVOS_ROOT,
            output_size=cfg.DATA.SIZE
        )


if __name__ == '__main__':
    ds = YouTube_Train('/public/datasets/YTBVOS', output_size=384, max_obj_n=6)
    trainloader = data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=1,
                                    collate_fn=multibatch_collate_fn, drop_last=True)
    i, data = next(enumerate(trainloader))
    print(data[0].shape, data[1].shape, data[2], data[3][0])
    frame, mask, num_obj = data[0][0], data[1][0], data[2][0]
    frame, mask = frame[:3], mask[:3]
    fig = plt.figure()
    for j in range(frame.shape[0]):
        ax = fig.add_subplot(2, 3, i*6+j+1)
        ax.axis('off')
        ax.imshow(frame[j].numpy().transpose(1, 2, 0))
        plt.pause(0.01)
    for k in range(mask.shape[0]):
        ax = fig.add_subplot(2, 3, i*6+4+k)
        ax.axis('off')
        # ax.imshow(np.array(mask[k, 0], dtype=np.uint8))
        ax.imshow(convert_one_hot(np.array(mask[k],dtype=np.uint8).transpose(1, 2, 0), num_obj.item()))
        plt.pause(0.01)
        # plt.imsave('test{}.png'.format(k), convert_one_hot(np.array(mask[k],dtype=np.uint8).transpose(1, 2, 0), num_obj.item()))
    fig.savefig("test.png")
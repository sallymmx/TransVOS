import os
import numpy as np
from glob import glob
import random

import torch
from torch.utils import data
import torchvision.transforms as TF

import datasets.transform as mytrans
from utils.system import load_image_in_PIL, gct
import matplotlib.pyplot as plt
from PIL import Image
from datasets.data_utils import multibatch_collate_fn, convert_one_hot


class DAVIS_Train(data.Dataset):
    r'''
    - root: data root path, str
    - output_size: output size of image and mask, tuple
    - imset
    - clip_n: number of video clip for training, int
    - max_obj_n: maximum number of objects in a image for training, int
    '''
    MAX_TRAINING_SKIP = 100
    def __init__(self, root, output_size, imset='2017/train.txt', clip_n=3, max_obj_n=7,
    max_skip=5, increment=5, samples=2, choice='order', crop=False):
        self.root = root
        self.clip_n = clip_n
        self.output_size = output_size
        self.max_obj_n = max_obj_n
        self.max_skip = max_skip
        self.increment = increment
        self.smaples = samples
        self.sample_choice = choice
        self.crop = crop

        dataset_path = os.path.join(root, 'ImageSets', imset)
        self.dataset_list = list()
        with open(os.path.join(dataset_path), 'r') as lines:
            for line in lines:
                dataset_name = line.strip()
                if len(dataset_name) > 0:
                    self.dataset_list.append(dataset_name)
        print(f'\t"DAVIS17": {len(self.dataset_list)} videos.')

        self.random_horizontal_flip = mytrans.RandomHorizontalFlip(0.3)
        self.color_jitter = TF.ColorJitter(0.1, 0.1, 0.1, 0.02)
        self.random_affine = mytrans.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=10)
        if self.crop:
            self.random_resize_crop = mytrans.RandomResizedCrop(400, (0.8, 1), (0.95, 1.05))
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
        return len(self.dataset_list) * self.smaples

    def __getitem__(self, idx):
        video_name = self.dataset_list[idx//self.smaples]
        img_dir = os.path.join(self.root, 'JPEGImages', '480p', video_name)
        mask_dir = os.path.join(self.root, 'Annotations', '480p', video_name)

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
            while len(idx_list) < self.clip_n:  # short video
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


class DAVIS_Test(data.Dataset):
    r'''
    - root: data root path, str
    - output_size: output size of image and mask, tuple
    - imset
    - max_obj_n: maximum number of objects in a image for training, int
    '''
    def __init__(self, root, output_size=None, img_set='2017/val.txt', max_obj_n=11, single_obj=False):
        self.root = root
        self.single_obj = single_obj
        dataset_path = os.path.join(root, 'ImageSets', img_set)
        self.dataset_list = list()
        self.output_size = output_size

        with open(os.path.join(dataset_path), 'r') as lines:
            for line in lines:
                dataset_name = line.strip()
                if len(dataset_name) > 0:
                    self.dataset_list.append(dataset_name)

        self.to_tensor = TF.ToTensor()
        self.normalize = TF.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.to_onehot = mytrans.ToOnehot(max_obj_n, shuffle=False)

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        video_name = self.dataset_list[idx]

        img_dir = os.path.join(self.root, 'JPEGImages', '480p', video_name)
        mask_dir = os.path.join(self.root, 'Annotations', '480p', video_name)

        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

        first_mask = load_image_in_PIL(mask_list[0], 'P')
        original_w, original_h = first_mask.size

        if self.output_size:
            out_h, out_w = self.output_size
            if original_h < out_h:
                h, w = original_h, original_w
            else:
                h = out_h
                w = int(original_w / original_h * out_h)
            # h = self.out_h
            # w = self.out_w
        else:
            h, w = original_h, original_w

        first_mask = first_mask.resize((w, h), Image.NEAREST)

        first_mask_np = np.array(first_mask, np.uint8)

        if self.single_obj:
            first_mask_np[first_mask_np > 1] = 1

        obj_n = first_mask_np.max() + 1
        video_len = len(img_list)

        frames = torch.zeros((video_len, 3, h, w), dtype=torch.float)
        masks = torch.zeros((1, obj_n, h, w), dtype=torch.float)

        mask, _ = self.to_onehot(first_mask_np)
        masks[0] = mask[:obj_n]

        for i in range(video_len):
            img = load_image_in_PIL(img_list[i], 'RGB')
            img = img.resize((w, h), Image.BILINEAR)
            frames[i] = self.normalize(self.to_tensor(img))

        info = {
            'name': video_name,
            'num_frames': video_len,
            'original_size': (original_h, original_w)
        }

        return frames, masks, obj_n, info


def build_davis(cfg, train=True):
    if train:
        return DAVIS_Train(
            root=cfg.DATA.DAVIS_ROOT,
            output_size=cfg.DATA.SIZE,
            clip_n=cfg.DATA.TRAIN.FRAMES_PER_CLIP,
            max_obj_n=cfg.DATA.TRAIN.MAX_OBJECTS,
            max_skip=cfg.DATA.TRAIN.DAVIS_SKIP_INCREMENT[0],
            increment=cfg.DATA.TRAIN.DAVIS_SKIP_INCREMENT[1],
            samples=cfg.DATA.TRAIN.SAMPLES_PER_VIDEO,
            choice=cfg.DATA.TRAIN.SAMPLE_CHOICE,
            crop=cfg.DATA.TRAIN.CROP
        )
    else:
        single_obj = (cfg.DATA.VAL.DATASET_NAME == 'DAVIS16')
        return DAVIS_Test(
            root=cfg.DATA.DAVIS_ROOT,
            single_obj=single_obj
        )


if __name__ == '__main__':
    ds = DAVIS_Train('/public/datasets/DAVIS', output_size=(240, 427), max_obj_n=6)
    trainloader = data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=1,
                                    collate_fn=multibatch_collate_fn, drop_last=True)
    i, data = next(enumerate(trainloader))
    print(data[0].shape, data[1].shape, data[2], data[3][0])
    frame, mask, num_obj = data[0][0], data[1][0], data[2][0]
    frame, mask = frame[:3], mask[:3]
    print(torch.max(mask), torch.min(mask))
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
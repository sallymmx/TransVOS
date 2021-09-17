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


class PretrainDataset(data.Dataset):
    r'''
    - root: data root path, str
    - output_size: output size of image and mask, tuple
    - clip_n: number of video clip for training, int
    - max_obj_n: maximum number of objects in a image for training, int
    '''
    PRETRAIN_DATA_LIST = ['COCO', 'ECSSD', 'MSRA10K', 'PASCAL-S', 'PASCALVOC2012']
    sample_ratio = 1
    def __init__(self, root, output_size, clip_n=3, max_obj_n=11, crop=False):
        self.root = root
        self.clip_n = clip_n
        self.output_size = output_size
        self.max_obj_n = max_obj_n
        self.max_skip = None
        self.crop = crop

        self.img_list = list()
        self.mask_list = list()

        dataset_list = list()
        for dataset_name in PretrainDataset.PRETRAIN_DATA_LIST:
            img_dir = os.path.join(root, 'JPEGImages', dataset_name)
            mask_dir = os.path.join(root, 'Annotations', dataset_name)

            img_list = sorted(glob(os.path.join(img_dir, '*.jpg'))) + sorted(glob(os.path.join(img_dir, '*.png')))
            mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

            if len(img_list) > 0:
                if len(img_list) == len(mask_list):
                    dataset_list.append(dataset_name)
                    self.img_list += img_list
                    self.mask_list += mask_list
                    print(f'\t{dataset_name}: {len(img_list)} imgs.')
                else:
                    print(f'\tPreTrain dataset {dataset_name} has {len(img_list)} imgs and {len(mask_list)} annots. Not match! Skip.')
            else:
                print(f'\tPreTrain dataset {dataset_name} doesn\'t exist. Skip.')

        print(gct(), f'{len(self.img_list)} imgs are used for PreTrain. They are from {dataset_list}.')

        self.random_horizontal_flip = mytrans.RandomHorizontalFlip(0.3)
        self.color_jitter = TF.ColorJitter(0.1, 0.1, 0.1, 0.03)
        self.random_affine = mytrans.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)
        if self.crop:
            self.random_resize_crop = mytrans.RandomResizedCrop(400, (0.8, 1))
        else:
            self.resize = mytrans.Resize(output_size)
        self.to_tensor = TF.ToTensor()
        self.normalize = TF.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.to_onehot = mytrans.ToOnehot(max_obj_n, shuffle=True)

    def increase_max_skip(self):
        pass

    def set_max_skip(self, max_skip):
        self.max_skip = max_skip

    def __len__(self):
        return int(PretrainDataset.sample_ratio*len(self.img_list))

    def __getitem__(self, idx):
        obj_n = 1
        while obj_n == 1:
            img_pil = load_image_in_PIL(self.img_list[idx], 'RGB')
            mask_pil = load_image_in_PIL(self.mask_list[idx], 'P')

            if not self.crop:
                frames = torch.zeros((self.clip_n, 3, *self.output_size), dtype=torch.float)
                masks = torch.zeros((self.clip_n, self.max_obj_n, *self.output_size), dtype=torch.float)
            else:
                frames = torch.zeros((self.clip_n, 3, 400, 400), dtype=torch.float)
                masks = torch.zeros((self.clip_n, self.max_obj_n, 400, 400), dtype=torch.float)

            for i in range(self.clip_n):
                img, mask = img_pil, mask_pil
                if i > 0:
                    img, mask = self.random_horizontal_flip(img, mask)
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
                'name': self.img_list[idx]
            }

            if obj_n == 1:
                idx = random.choice(range(len(self.img_list)))

        return frames, masks, obj_n, info


def build_pretrain(cfg, train=True):
    return PretrainDataset(
        root=cfg.DATA.PRETRAIN_ROOT,
        output_size=cfg.DATA.SIZE,
        clip_n=cfg.DATA.TRAIN.FRAMES_PER_CLIP,
        max_obj_n=cfg.DATA.TRAIN.MAX_OBJECTS,
        crop=cfg.DATA.TRAIN.CROP
    )


if __name__ == '__main__':
    ds = PretrainDataset('/public/datasets/VOS/pretrain/', output_size=(384, 384), clip_n=3, max_obj_n=6)
    trainloader = data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=1, drop_last=True)
    i, data = next(enumerate(trainloader))
    print(data[0].shape, data[1].shape, data[2], data[3]['name'])
    frame, mask, num_obj = data[0][0], data[1][0], data[2][0]
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
import torch
import numpy as np


def multibatch_collate_fn(batch):

    min_time = min([sample[0].shape[0] for sample in batch])
    frames = torch.stack([sample[0] for sample in batch])
    masks = torch.stack([sample[1] for sample in batch])

    objs = [torch.LongTensor([sample[2]]) for sample in batch]
    objs = torch.cat(objs, dim=0)

    try:
        info = [sample[3] for sample in batch]
    except IndexError as ie:
        info = None

    return frames, masks, objs, info


def convert_mask(mask, max_obj):

    # convert mask to one hot encoded
    oh = []
    for k in range(max_obj):
        oh.append(mask==k)

    oh = np.stack(oh, axis=2)

    return oh


def convert_one_hot(oh, max_obj):

    mask = np.zeros(oh.shape[:2], dtype=np.uint8)
    for k in range(max_obj):
        mask[oh[:, :, k]==1] = k

    return mask
#coding:utf-8

import os
import torch
import torch.utils.data as data
from PIL import Image
from scipy.io import loadmat
import numpy as np
import glob
from torchvision import transforms
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision

import code.utils.ext_transforms as ext_transforms

import matplotlib.pyplot as plt


def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class NYUv2(data.Dataset):
    """NYUv2 depth dataset loader.
    **Parameters:**
        - **root** (string): Root directory path.
        - **split** (string, optional): 'train' for training set, and 'test' for test set. Default: 'train'.
        - **num_classes** (string, optional): The number of classes, must be 40 or 13. Default:13.
        - **transform** (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default: None.
        - **target_transforms** (callable, optional): A list of function/transform that takes in the target and transform it. Default: None.
        - **ds_type** (string, optional): To pick samples with labels or not. Default: 'labeled'.
    """
    cmap = colormap()

    def __init__(self,
                 root,
                 split='train',
                 num_classes=13,
                 transform=None,
                 ds_type='labeled',
                 split_ratio = 0.1):

        assert(split in ('train', 'val', 'test'))
        assert(ds_type in ('labeled', 'unlabeled'))
        self.root = root
        self.split = split
        self.ds_type = ds_type
        self.transform = transform
        self.num_classes = num_classes
        self.train_idx = np.array([255, ] + list(range(num_classes)))

        if ds_type == 'labeled':
            split_mat = loadmat(os.path.join(
                self.root, 'nyuv2-meta-data', 'splits.mat'))

            if self.split == 'test':
                idxs = split_mat['testNdxs'].reshape(-1)
            else:
                idxs = split_mat['trainNdxs'].reshape(-1)


            if self.split == 'val':
                idxs = idxs[:int(split_ratio*len(idxs))]
            elif self.split == 'train':
                idxs = idxs[int(split_ratio*len(idxs)):]

            if self.split == 'val':
                self.split = 'train'

            self.images = [os.path.join(self.root, '%s' % self.split, 'nyu_rgb_%04d.png' % (idx))
                           for idx in idxs]
            if self.num_classes == 13:
                self.targets = [os.path.join(self.root, 'nyuv2-meta-data', '%s_labels_13' % self.split, 'new_nyu_class13_%04d.png' % idx)
                                for idx in idxs]
            else:
                raise ValueError(
                    'Invalid number of classes! Please use 13 or 40')
        else:
            self.images = [glob.glob(os.path.join(
                self.root, 'unlabeled_images/*.png'))]
        print(self.split, len(self.images))


    def __getitem__(self, idx):
        if self.ds_type == 'labeled':
            image = Image.open(self.images[idx])
            target = Image.open(self.targets[idx])

            if self.transform:
                image, target = self.transform(image, target)
            #print(target)
            target = self.train_idx[target]
            return image, target
        else:
            image = Image.open(self.images[idx])
            if self.transforms is not None:
                image = self.transforms(image)
            image = transforms.ToTensor()(image)
            return image, None

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, target):
        target = (target+1).astype('uint8')  # 255 -> 0, 0->1, 1->2
        return cls.cmap[target]


if __name__ == '__main__':

    logger = SummaryWriter('logs/dataset_fullsize')
    pixel_dist = np.zeros(13)
    transform_list = ext_transforms.ExtCompose([
        # ext_transforms.ExtResize(256),
        # ext_transforms.ExtRandomCrop(128),
        ext_transforms.ExtToTensor(),
        ext_transforms.ExtNormalize((0.5,), (0.5,)),
        ])
    dataset = NYUv2(root='data/Nyu', split='train', transform=transform_list)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    for batch_idx, batch in enumerate(tqdm(loader, ascii=True)):
        data, target = batch
        for i in range(len(pixel_dist)):
            pixel_dist[i] += (torch.sum(target == i))
        labels = dataset.decode_target(target.detach().cpu().numpy())
        labels = torch.Tensor(labels).permute(0, 3, 1, 2)
        data = (data+1)/2
        sample = torch.cat([data, labels], axis=0)
        grid = torchvision.utils.make_grid(sample)
        logger.add_image('data', grid, batch_idx)


    dist = pixel_dist / pixel_dist.sum()
    print(dist)
    print(dist.sum())

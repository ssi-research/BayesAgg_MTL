# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from experiments.ChestX_ray14.dataset.custom_image_folder import MyImageFolder
import logging
from experiments.utils import set_logger
from torchvision.utils import save_image
import shutil

set_logger()

IMG_SIZE = 224
INTERPOLATION = 'bilinear'

"""
Code largely taken from: https://github.com/rohban-lab/SwinCheX/tree/main
"""

def merge_folders(img_root):
    """
    data comes distributed to eleven folders. Merge them to one folder prior running
    :param data_path:
    :return:
    """
    data_path = img_root + "/dataset/"
    if os.path.exists(data_path):
        logging.info("data was already merged")
        return data_path

    os.makedirs(data_path, exist_ok=True)
    for folder in os.listdir(img_root):
        if os.path.isfile(img_root + '/' + folder):
            continue

        orig_path = img_root + "/" + folder + '/images'
        cmd = 'mv ' + orig_path + "/* " + data_path
        os.system(cmd)

        cmd = 'rm -r ' + img_root + "/" + folder
        os.system(cmd)

    return data_path


def build_loader(data_root, train_csv_path='./partition/train.csv',
                 validation_csv_path='./partition/validation.csv', test_csv_path='./partition/test.csv',
                 batch_size=128, test_batch_size=256, num_workers=4):

    data_path = resize_images(data_root, train_csv_path, validation_csv_path,
                              test_csv_path, batch_size, test_batch_size, num_workers)

    dataset_train = build_dataset(is_train=True, data_path=data_path, csv_path=train_csv_path)
    dataset_val = build_dataset(is_train=False, data_path=data_path, csv_path=validation_csv_path)
    dataset_test = build_dataset(is_train=False, data_path=data_path, csv_path=test_csv_path)

    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


def build_dataset(is_train, data_path, csv_path):
    transform = build_transform(is_train)
    dataset_folder = MyImageFolder(root=data_path, csv_path=csv_path, transform=transform)
    return dataset_folder


def build_transform(is_train):

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=IMG_SIZE,
            is_training=True,
            color_jitter=0.4,  # Color jitter factor
            auto_augment='rand-m9-mstd0.5-inc1',  # Use AutoAugment policy. "v0" or "original"
            re_prob=0.25,  # Random erase prob
            re_mode='pixel',  # Random erase mode
            re_count=1,  # Random erase count
            mean=IMAGENET_DEFAULT_MEAN,  #,  (0.0, 0.0, 0.0)
            std=IMAGENET_DEFAULT_STD  #, (1.0, 1.0, 1.0)
        )
        return transform

    t = []
    t.append(transforms.CenterCrop(IMG_SIZE))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def resize_images(data_root, train_csv_path='./partition/train.csv',
                  validation_csv_path='./partition/validation.csv', test_csv_path='./partition/test.csv',
                  batch_size=128, test_batch_size=256, num_workers=4):

    data_path = merge_folders(data_root)
    new_data_path = data_root + "/dataset_compressed/"

    if os.path.exists(new_data_path):
        logging.info("data was already resized")
        return new_data_path

    t = []
    size = int((256 / 224) * IMG_SIZE)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
    )
    t.append(transforms.ToTensor())

    t = transforms.Compose(t)
    dataset_train = MyImageFolder(root=data_path, csv_path=train_csv_path, transform=t)
    dataset_val = MyImageFolder(root=data_path, csv_path=validation_csv_path, transform=t)
    dataset_test = MyImageFolder(root=data_path, csv_path=test_csv_path, transform=t)

    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    for i, loader in enumerate([train_loader, val_loader, test_loader]):
        for j, data in enumerate(loader):
            x, _, paths = (datum for datum in data)

            for k in range(x.shape[0]):
                im_name = paths[k].split('/')[-1]
                save_image(tensor=x[k, ...], fp=new_data_path + im_name)

    return new_data_path

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F
    from torchvision.utils import make_grid


    def show(imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


    train_loader, val_loader, test_loader = (
        build_loader(data_root='/Vols/vol_design/tools/swat/datasets_backup/ChestX-ray8', num_workers=0))
    # train_loader, val_loader, test_loader = (
    #     resize_images(data_root='/Vols/vol_design/tools/swat/datasets_backup/ChestX-ray8', num_workers=0))

    x, ys, paths = next(iter(train_loader))

    grid = make_grid(x)
    show(grid)
    plt.show()
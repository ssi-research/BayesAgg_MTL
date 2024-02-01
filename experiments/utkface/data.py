import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
import torchvision.transforms as transforms
import logging
from experiments.utils import set_logger
import argparse
import pandas as pd
import shutil

set_logger()


def split_to_train_val_test(img_root='./dataset', train_size=0.8, val_size=0.125):
    """
    Split UTKFace to train, val, and test folders
    :param img_root: root for UTKFace data
    :param train_size: % of the total data
    :param val_size:  % of the training data
    """

    if os.path.exists(img_root + "/train"):
        logging.info("data splits already created")
        return

    img_path = img_root + "/UTKFace"

    # collect labels - neglect 3 images with missing labels
    paths = [f for f in os.listdir(img_path) if
             len(f.split("/")[-1].split(".")[0].split("_")[:-1]) == 3]
    labels_dict = {"age": [], "gender": [], "race": [], "path": []}
    for f in paths:
        labels = f.split("/")[-1].split(".")[0].split("_")[:-1]
        a, g, r = labels
        labels_dict["age"].append(a)
        labels_dict["gender"].append(g)
        labels_dict["race"].append(r)
        labels_dict["path"].append(f)

    images_df = pd.DataFrame(labels_dict, index=np.arange(len(paths)))
    # take only records with an age that appear at least 3 times
    common_images_df = images_df[images_df.groupby('age')['age'].transform('size') >= 3]
    uncommon_images_df = images_df[images_df.groupby('age')['age'].transform('size') < 3]

    # split to datasets
    train_data, test_data = train_test_split(common_images_df, train_size=train_size,
                                             stratify=common_images_df['age'], random_state=42)

    train_data, val_data = train_test_split(train_data, test_size=val_size,
                                            stratify=train_data['age'], random_state=42)

    # add uncommon images to train data
    train_data = pd.concat([train_data, uncommon_images_df])

    # copy images to new destination
    for d_idx, dataset in enumerate([train_data, val_data, test_data]):
        paths = dataset['path'].tolist()
        split = 'train' if d_idx == 0 else 'val' if d_idx == 1 else 'test'
        for i, f in enumerate(paths):
            dest_path = img_root + '/' + split + '/images'
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)

            orig_path = img_path + '/' + f
            shutil.copy(orig_path, dest_path)


class UTKFaces(ImageFolder):
    """
    Data class for UTKFace.
    The labels of each face image is embedded in the file name, formated like [age]_[gender]_[race]_[date&time].jpg
    [age] is an integer from 0 to 116, indicating the age
    [gender] is either 0 (male) or 1 (female)
    [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
    """
    def __init__(self, root='./dataset', split='train', transform=None, target_transform=None):
        super().__init__(root=os.path.join(root, split), transform=transform, target_transform=target_transform)
        self.mean = 33.327  # pre-computed based on the training set only
        self.std = 19.927  # pre-computed based on the training set only

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        age, gender, race = path.split("/")[-1].split(".")[0].split("_")[:-1]
        age = (float(age) - self.mean) / self.std

        return sample, torch.tensor(age), torch.tensor(int(gender)), torch.tensor(int(race))


class UTKFacesData:
    """
    Base class for generating datasets and data loaders
    """

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.getcwd() + "/dataset"

        self.data_dir = data_dir
        split_to_train_val_test(data_dir)

    def get_datasets(self):

        # define transforms
        valid_transform = transforms.Compose([
            transforms.Resize((140, 140)),
            transforms.CenterCrop(128),
            transforms.ToTensor()
        ])
        train_transform = transforms.Compose([
            transforms.Resize((140, 140)),
            transforms.RandomCrop(128, padding=0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        # load the dataset
        train_dataset = UTKFaces(
            root=self.data_dir,
            split='train',
            transform=train_transform,
        )

        # valid
        valid_dataset = UTKFaces(  # no augmentations
            root=self.data_dir,
            split='val',
            transform=valid_transform,
        )
        # test
        test_dataset = UTKFaces(  # no augmentations
            root=self.data_dir,
            split='test',
            transform=valid_transform,
        )

        num_train = len(train_dataset)
        num_val = len(valid_dataset)
        num_test = len(test_dataset)

        logging.info(
            f"\nTrain size = {num_train}, "
            f"\nValidation size = {num_val}, "
            f"\nTest size = {num_test}"
        )

        return train_dataset, valid_dataset, test_dataset

    def get_loaders(self, batch_size=256, test_batch_size=512, shuffle_train=True,
                    num_workers=4, pin_memory=True):

        train_dataset, valid_dataset, test_dataset = self.get_datasets()

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            valid_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader, test_loader


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

    parser = argparse.ArgumentParser(description="UTKFace data")
    parser.add_argument("--data-path", type=str, default="./dataset", help="data path")
    args = parser.parse_args()

    dataset = UTKFacesData(args.data_path)
    train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=8, test_batch_size=8, num_workers=0)
    x, a, g, r = next(iter(train_loader))
    print(a)
    print(g)
    print(r)

    grid = make_grid(x)
    show(grid)
    plt.show()

# if __name__ == '__main__':
#
#     ds = UTKFaceDataModule()
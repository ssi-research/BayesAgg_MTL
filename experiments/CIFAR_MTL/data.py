from pathlib import Path
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity

import os
import pickle
import json
import numpy as np

import torch


class CIFAR100(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]

    def __init__(
        self,
        root,
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
                ),
            ]
        ),
        download=False,
    ):
        """
        Args:
        root (string): Root directory of dataset where directory
            ``cifar-100-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        """
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.train = train
        self.root = os.path.expanduser(root)

        # check download
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        # R\read the data file
        root = Path(root)
        if train:
            self.data_path = root / "cifar-100-python/train"
        else:
            self.data_path = root / "cifar-100-python/test"
        with open(self.data_path, "rb") as fo:
            self.data_info = pickle.load(fo, encoding="latin1")
        fo.close()

        # calculate data length
        self.data_len = len(self.data_info["data"])

        # first column contains the image paths
        self.image_arr = self.data_info["data"].reshape([self.data_len, 3, 32, 32])
        self.image_arr = self.image_arr.transpose((0, 2, 3, 1))  # convert to HWC

        # 10 Class, build dict from 20 class:
        class_10 = {
            0: 1,
            1: 2,
            2: 5,
            3: 6,
            4: 5,
            5: 6,
            6: 6,
            7: 3,
            8: 0,
            9: 7,
            10: 8,
            11: 0,
            12: 1,
            13: 3,
            14: 4,
            15: 0,
            16: 2,
            17: 5,
            18: 9,
            19: 9,
        }
        # 3 Class, build dict from 10 class:
        class_3 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2}

        # second column is the labels
        self.label_fine = self.data_info["fine_labels"]
        self.label_coarse = self.data_info["coarse_labels"]
        self.label_c10 = np.vectorize(class_10.get)(self.label_coarse)
        self.label_c3 = np.vectorize(class_3.get)(self.label_c10)

    def _check_integrity(self):
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __getitem__(self, index):
        # get image name from the pandas df
        single_image_name = self.image_arr[index]

        # open image
        img_as_img = Image.fromarray(single_image_name)

        # transform image to tensor
        if self.transform is not None:
            img_as_img = self.transform(img_as_img)
        else:
            img_as_img = self.to_tensor(img_as_img)

        # get label(class) of the image (from coarse to fine)
        label = np.zeros(4)
        label[-1] = self.label_fine[index]
        label[-2] = self.label_coarse[index]
        label[-3] = self.label_c10[index]
        label[-4] = self.label_c3[index]

        return img_as_img, self.label_fine[index], self.label_coarse[index]

    def __len__(self):
        return self.data_len


def one_hot(indices, width):
    indices = indices.squeeze().unsqueeze(1)
    oh = torch.zeros(indices.size()[0], width).to(indices.device)
    oh.scatter_(1, indices, 1)
    return oh


if __name__ == "__main__":
    data_path = "./data"
    train_dataset = CIFAR100(data_path, download=True, train=True)
    test_dataset = CIFAR100(data_path, download=True, train=False)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
    )

    batch = next(iter(test_loader))

    print("debug")

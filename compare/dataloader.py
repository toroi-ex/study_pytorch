import os

import torch

import torchvision.transforms as transforms
from torch.utils.data import Subset

from pathlib import Path

from PIL import Image


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, imageSize, dir_path, transform=None):
        self.transform = transforms.Compose([
            transforms.Resize(imageSize, interpolation=Image.BILINEAR),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        self.image_paths = [str(p) for p in Path(dir_path).glob("**/*.png")]

        self.data_num = len(self.image_paths)
        self.classes = ['broken', 'correct']
        self.class_to_idx = {'broken': 1, 'correct': 0}

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        image = Image.open(p).convert('RGB') #画像をRGBに変換することで、3チャネルにする

        if self.transform:
            out_data = self.transform(image)

        out_label = p.split("/") #フォルダの名前を”/”で分割
        out_label = self.class_to_idx[out_label[-2]] #後ろから2番めの値を取ってくる（”correct or broken”）

        return out_data, out_label


class MyDatasetEval(torch.utils.data.Dataset):

    def __init__(self, imageSize, dir_path, transform=None):
        self.transform = transforms.Compose([
            transforms.Resize(imageSize, interpolation=Image.BILINEAR),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        self.image_paths = [str(p) for p in Path(dir_path).glob("**/*.png")]

        self.data_num = len(self.image_paths)
        self.classes = ['broken', 'correct']
        self.class_to_idx = {'broken': 1, 'correct': 0}

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        image = Image.open(p).convert('RGB') #画像をRGBに変換することで、3チャネルにする

        if self.transform:
            out_data = self.transform(image)

        out_label = p.split("/") #フォルダの名前を”/”で分割
        result_label = out_label[-1]
        out_label = self.class_to_idx[out_label[-2]] #後ろから2番めの値を取ってくる（”correct or broken”）

        return out_data, out_label, result_label

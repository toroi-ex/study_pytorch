import torchvision.transforms as transforms
import torch
from PIL import Image
from pathlib import Path


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, imageSize, dir_path, transform=None):
        self.transform = transforms.Compose([
            transforms.Resize(imageSize, interpolation=Image.BILINEAR),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.ColorJitter(brightness=0.3, contrast=0.3,),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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


class MydatasetGan(torch.utils.data.Dataset):

    def __init__(self, imageSize, gan_path, transform=None):
        self.transform = transforms.Compose([
            transforms.Resize(imageSize, interpolation=Image.BILINEAR),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.image_paths = [str(p) for p in Path(gan_path).glob("**/*.png")]

        self.data_num = len(self.image_paths)
        # self.classes = ['broken', 'correct']
        self.class_to_idx = {'broken': 1}

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

class MydatasetMixup(torch.utils.data.Dataset):

    def __init__(self, imageSize, mix_path, transform=None):
        self.transform = transforms.Compose([
            transforms.Resize(imageSize, interpolation=Image.BILINEAR),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.image_paths = [str(p) for p in Path(mix_path).glob("**/*.png")]

        self.data_num = len(self.image_paths)
        # self.classes = ['broken', 'correct']
        self.class_to_idx = {'broken': 1}

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
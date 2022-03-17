import os
from PIL import Image, ImageOps
from PIL import ImageFilter
import random
from matplotlib import cm
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.transforms import functional as tvf

"""
画像の前処理を定義
"""

def blur(img):
    #ガウシアンフィルタ
    return img.filter(ImageFilter.GaussianBlur(1.5))


data_transforms = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.BILINEAR),
    # transforms.Resize((150, 150), interpolation=Image.BILINEAR),
    # transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    # transforms.Lambda(blur),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_val_transforms = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.BILINEAR),
    # transforms.Resize((150, 150), interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CustomDataset(torch.utils.data.Dataset):

    #１番目のクラスを２番めのクラスに近づけようとしてる 逆もしてる
    # classes = ["broken_03", "broken_06"]
    classes = ["img", "img1"]
    # classes = ["HL", "MCF"]

    def __init__(self, root, fold1, fold2, transform=None):

        # ファイルパスorファイル名のリストを初期化
        # root = "./data/train" or root = "./data/test" or root = "./data/val"
        self.transform = transform
        self.images_a = []  # ファイルのパスを入れる
        self.images_b = []

        root_a_path = os.path.join(root, fold1)
        root_b_path = os.path.join(root, fold2)

        images_a0 = os.listdir(root_a_path)
        images_b0 = os.listdir(root_b_path)

        random.shuffle(images_a0)
        random.shuffle(images_b0)

        # images_a0 = sorted(images_a0)
        # images_b0 = sorted(images_b0)

        len_images = int(len(images_a0))
        len_images_b = int(len(images_b0))

        for i in range(len_images_b):
            try:
                self.images_a.append(os.path.join(root_a_path, images_a0[i]))
                self.images_b.append(os.path.join(root_b_path, images_b0[i]))
            except:
                #x_b = random.randint(0, len_images_b-1)
                #self.images_b.append(os.path.join(root_b_path, images_b0[x_b]))
                x_a = random.randint(0, len_images - 1)
                self.images_a.append(os.path.join(root_a_path, images_a0[x_a]))
                self.images_b.append(os.path.join(root_b_path, images_b0[i]))

    def __getitem__(self, index):

        image_a_path = self.images_a[index]
        image_b_path = self.images_b[index]

        file_name1 = os.path.basename(image_a_path)
        file_name2 = os.path.basename(image_b_path)

        #power device
        img_a = Image.open(image_a_path).convert('RGB')
        img_b = Image.open(image_b_path).convert('RGB')

        # other images
        # img_a = Image.open(image_a_path)
        # img_b = Image.open(image_b_path)

        # クロップ位置を乱数で決定
        # i, j, h, w = transforms.RandomCrop.get_params(img_a, output_size=(256, 256))

        # img_a = tvf.crop(img_a, i, j, h, w)
        # img_b = tvf.crop(img_b, i, j, h, w)

        if self.transform is not None:  # 前処理する場合
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)

        return img_a, img_b, file_name1, file_name2

    def __len__(self):

        return len(self.images_a)


class CustomDatasetTestA(torch.utils.data.Dataset):
    #１番目のクラスを２番めのクラスに近づけようとしてる 逆もしてる
    # classes = ["broken_03"]
    # classes = ["img"]

    def __init__(self, root, fold1, transform=None):

        # ファイルパスorファイル名のリストを初期化
        # root = "./data/train" or root = "./data/test" or root = "./data/val"
        self.transform = transform
        self.images_a = []  # ファイルのパスを入れる

        root_a_path = os.path.join(root, fold1)

        images_a0 = os.listdir(root_a_path)

        random.shuffle(images_a0)

        # images_a0 = sorted(images_a0)
        # images_b0 = sorted(images_b0)

        len_images_a = int(len(images_a0))
        # len_images = 10

        for i in range(len_images_a):
            self.images_a.append(os.path.join(root_a_path, images_a0[i]))

    def __getitem__(self, index):

        image_a_path = self.images_a[index]

        file_name1 = os.path.basename(image_a_path)

        img_a = Image.open(image_a_path).convert('RGB')

        # クロップ位置を乱数で決定
        # i, j, h, w = transforms.RandomCrop.get_params(img_a, output_size=(256, 256))

        # img_a = tvf.crop(img_a, i, j, h, w)
        # img_b = tvf.crop(img_b, i, j, h, w)

        if self.transform is not None:  # 前処理する場合
            img_a = self.transform(img_a)

        return img_a, file_name1

    def __len__(self):

        return len(self.images_a)


class CustomDatasetTestB(torch.utils.data.Dataset):
    #１番目のクラスを２番めのクラスに近づけようとしてる 逆もしてる
    # classes = ["broken_03", "broken_06"]
    # classes = ["img1"]

    def __init__(self, root, fold2, transform=None):

        # ファイルパスorファイル名のリストを初期化
        # root = "./data/train" or root = "./data/test" or root = "./data/val"
        self.transform = transform
        self.images_b = []

        root_b_path = os.path.join(root, fold2)

        images_b0 = os.listdir(root_b_path)

        random.shuffle(images_b0)

        # images_a0 = sorted(images_a0)
        # images_b0 = sorted(images_b0)

        len_images_b = int(len(images_b0))
        # len_images = 10

        for i in range(len_images_b):
            self.images_b.append(os.path.join(root_b_path, images_b0[i]))

    def __getitem__(self, index):

        image_b_path = self.images_b[index]

        file_name2 = os.path.basename(image_b_path)

        img_b = Image.open(image_b_path).convert('RGB')
        # クロップ位置を乱数で決定
        # i, j, h, w = transforms.RandomCrop.get_params(img_a, output_size=(256, 256))

        # img_a = tvf.crop(img_a, i, j, h, w)
        # img_b = tvf.crop(img_b, i, j, h, w)

        if self.transform is not None:
            img_b = self.transform(img_b)

        return img_b, file_name2

    def __len__(self):

        return len(self.images_b)

import matplotlib.pyplot as plt
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data import Subset, TensorDataset
from torchvision.utils import save_image
from torchvision import models

#focal_loss
from vgg16_bn.focalLoss import FocalLoss

from torchsummary import summary

from pathlib import Path

from PIL import Image
import cv2

import datetime

from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score


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


class MydatasetGan(torch.utils.data.Dataset):

    def __init__(self, imageSize, gan_path, transform=None):
        self.transform = transforms.Compose([
            transforms.Resize(imageSize, interpolation=Image.BILINEAR),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        self.image_paths = [str(p) for p in Path(gan_path).glob("**/*.png")]

        self.data_num = len(self.image_paths)
        # self.classes = ['broken', 'correct']
        # self.class_to_idx = {'broken': 1, 'correct': 0}
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


def main():

    # training set
    data_set = MyDataset(200, dir_path=train_data_dir)
    data_set_gan = MydatasetGan(200, gan_path=train_only_path)
    data_set_val = MyDatasetEval(200, dir_path=train_data_dir)

    label_list = []
    for i in range(len(data_set)):
        label_list.append(data_set[i][1])

    total_size = 0

    file_list = os.listdir(os.path.join(train_data_dir, 'broken'))

    pred = []
    Y = []

    #cross_validation
    kf = StratifiedKFold(n_splits=len(file_list), shuffle=True)

    #loss function
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(alpha=0.55, gamma=2.0) #alpha=0.65,gamma=2

    for fold_idx, idx in enumerate(kf.split(data_set, label_list)):

        # モデルを構築
        # net = models.vgg16(pretrained=True)
        net = models.resnet34(pretrained=True)

        print("ネットワーク設定完了：学習をtrainモードで開始します")

        #vgg16の全結合
        # num_ftrs = net.classifier[6].in_features
        # net.classifier[6] = nn.Linear(num_ftrs, 2)

        #resnetの全結合
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 2)

        net.to(device)

        # 最適化
        optimizer = optim.SGD(net.parameters(), lr=0.001)

        train_idx, valid_idx = idx

        #ganをtrainingに加えるためにデータセットを合成
        train_dataset = Subset(data_set, train_idx) + data_set_gan

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(Subset(data_set_val, valid_idx), batch_size=1, shuffle=False)

        # trainモードで開始
        net.train()

        # modelの全体像を表示
        summary(net, input_size=(3, 200, 200))

        train_loss_value = []  # trainingのlossを保持するlist
        train_acc_value = []  # trainingのaccuracyを保持するlist

        for epoch in range(13):
            print("epoch =", epoch + 1)

            # 今回の学習効果を保存するための変数
            running_loss = 0.0
            input_num = 0

            for batch_idx, data in enumerate(train_loader):  # dataがラベルと画像情報の2つの情報を持つ
                total_loss = 0
                # データ整理
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前回の勾配情報をリセット
                optimizer.zero_grad()

                # 予測
                outputs = net(inputs)

                # 予測結果と教師ラベルを比べて損失を計算
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                total_loss += loss.item()
                total_size += inputs.size(0)

                # 損失に基づいてネットワークのパラメーターを更新
                loss.backward()
                optimizer.step()

                input_num += len(inputs)

            #     if batch_idx % 1 == 0:
            #         now = datetime.datetime.now()
            #         print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
            #             now, epoch, batch_idx * len(inputs), len(dataloader.dataset), 100. * batch_idx / len(dataloader),
            #                         total_loss / total_size))
            # train_loss_value.append(running_loss * 50 / len(dataloader.dataset))  # traindataのlossをグラフ描画のためにlistに保持
            # print("running_loss=", running_loss * 50 / len(dataloader.dataset))
                if batch_idx % 1 == 0:
                    now = datetime.datetime.now()
                    print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                        now, epoch+1, input_num, len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                                    total_loss))
            train_loss_value.append(running_loss / len(train_loader.dataset))  # traindataのlossをグラフ描画のためにlistに保持
            print("train_loss=", running_loss / len(train_loader))

        #評価モード
        net.eval()
        print("ネットワークをevalに変更します")

        #test
        with torch.no_grad():
            pred_tmp = []
            Y_tmp = []
            target_name = ["correct 0", "broken 1"]
            for batch_idx, data in enumerate(valid_loader):
                input_val, label_val, image_name = data
                input_val = input_val.to(device)
                label_val = label_val.to(device)
                image_name = image_name[0].replace("'", "").replace(",", "")

                output_val = net(input_val)
                pred_tmp += [int(l.argmax()) for l in output_val]
                Y_tmp += [int(l) for l in label_val]

                if Y_tmp[-1] != pred_tmp[-1]:
                    if Y_tmp[-1] == 1:
                        mis_image = cv2.imread(os.path.join(train_data_dir, "broken/{filename}".format(filename=image_name)))
                        # print(os.path.join(train2, "{filename}".format(filename=image_name)))
                        cv2.imwrite(os.path.join(img_save_dir, "broken/{filename}".format(filename=image_name)), mis_image)
                    else:
                        mis_image = cv2.imread(os.path.join(train_data_dir, "correct/{filename}".format(filename=image_name)))
                        # print(os.path.join(train2, "{filename}".format(filename=image_name)))
                        cv2.imwrite(os.path.join(img_save_dir, "correct/{filename}".format(filename=image_name)), mis_image)

            # save_imageはネットの出力を保存する
            # save_image(output_val, os.path.join(img_save_dir, "{filename}".format(filename=image_name)), nrow=1)

            print(classification_report(Y_tmp, pred_tmp, target_names=target_name))
            print(confusion_matrix(Y_tmp, pred_tmp))

            pred.extend(pred_tmp)
            Y.extend(Y_tmp)

        model_path = f'leave_model/model_leave{fold_idx}.pth'
        torch.save(net.state_dict(), model_path)

    pre_score = precision_score(Y, pred)
    re_score = recall_score(Y, pred)
    f_score = f1_score(Y, pred)

    report = classification_report(Y, pred, target_names=target_name)
    conf = confusion_matrix(Y, pred)

    conf_exp = "TN FP\nFN TP"

    print("====leave_one_outの結果====")
    print(report)
    print(conf, "\n", conf_exp)
    print("precision : ", pre_score)
    print("recall : ", re_score)
    print("f1 : ", f_score)


if __name__ == "__main__":

    train_data_dir = '/home/toui/デスクトップ/ori/add_testUseGan2'
    train_only_path = '/home/toui/デスクトップ/ori/val'

    img_save_dir = "/home/toui/PycharmProjects/toui_pytorch/compare/img"

    weight_pash = "vgg16_bn.pth"

    if os.path.exists(img_save_dir):
        shutil.rmtree(img_save_dir)

    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(os.path.join(img_save_dir, "broken"), exist_ok=True)
    os.makedirs(os.path.join(img_save_dir, "correct"), exist_ok=True)

    device = torch.device("cuda:0")

    main()
    print("completed")

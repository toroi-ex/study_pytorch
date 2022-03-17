import matplotlib.pyplot as plt
import os
import shutil
import re

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

from dataload import MyDataset, MyDatasetEval, MydatasetGan, MydatasetMixup



def main(epochs, mode):

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
    count = 0

    #loss function
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(alpha=0.55, gamma=2.0) #alpha=0.65,gamma=2

    for fold_idx, idx in enumerate(kf.split(data_set, label_list)):

        data_name_list = []
        count+=1
        print(str(count)+"number")

        if mode == "vgg16":
            net = models.vgg16(pretrained=True)

            # vgg16の全結合
            num_ftrs = net.classifier[6].in_features
            net.classifier[6] = nn.Linear(num_ftrs, 2)

        elif mode == "resnet":
            net = models.resnet34(pretrained=True)

            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, 2)

        print("ネットワーク設定完了：学習をtrainモードで開始します")

        net.to(device)

        # 最適化
        optimizer = optim.SGD(net.parameters(), lr=0.001)

        train_idx, valid_idx = idx

        # mixup wo tukau baaino yatu
        valid_loader = torch.utils.data.DataLoader(Subset(data_set_val, valid_idx), batch_size=1, shuffle=False)
        for i in valid_loader:
            _, _, data_name = i
            data_name_list.append(data_name[0])

        data_name_list = sorted(data_name_list)

        #mixup image load
        if 'j' in data_name_list[-1]:
            mixup_name = re.sub(r"\D", "", data_name_list[-1])
            mixup_dir = os.path.join(mixup_base_dir, mixup_name)

        else:
            mixup_dir = os.path.join(mixup_base_dir, "all")

        dataset_mixup = MydatasetMixup(200, mixup_dir)

        train_dataset = Subset(data_set, train_idx)# + data_set_gan + dataset_mixup

        #ganをtrainingに加えるためにデータセットを合成
        # train_dataset = Subset(data_set, train_idx) + data_set_gan

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(Subset(data_set_val, valid_idx), batch_size=1, shuffle=False)

        # trainモードで開始
        net.train()

        # modelの全体像を表示
        # summary(net, input_size=(3, 200, 200))

        train_loss_value = []  # trainingのlossを保持するlist
        train_acc_value = []  # trainingのaccuracyを保持するlist

        for epoch in range(epochs):
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

    train_data_dir = '/home/toui/デスクトップ/ori/RealUseGan'
    train_only_path = '/home/toui/デスクトップ/ori/val'

    mixup_base_dir = "/home/toui/デスクトップ/ori/mixup_train/0.6"

    img_save_dir = "/home/toui/PycharmProjects/toui_pytorch/compare/img"

    epochs = 6

    if os.path.exists(img_save_dir):
        shutil.rmtree(img_save_dir)

    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(os.path.join(img_save_dir, "broken"), exist_ok=True)
    os.makedirs(os.path.join(img_save_dir, "correct"), exist_ok=True)

    device = torch.device("cuda:0")

    # vgg16 or resnet
    main(epochs, "resnet")
    print("completed")

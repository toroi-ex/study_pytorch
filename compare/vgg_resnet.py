import torch
import os
import shutil

import torch.nn as nn
import torch.optim as optim

from torchvision import models

from torch.utils.data import Subset

from torchsummary import summary

import cv2

import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

from dataloader import MyDataset, MyDatasetEval


def main():

    # training set
    data_set = MyDataset(200, dir_path=train_data_dir)
    data_set_val = MyDatasetEval(200, dir_path=train_data_dir)
    # dataloader = torch.utils.data.DataLoader(data_set, batch_size=16, shuffle=True)

    label_list = []
    for i in range(len(data_set)):
        label_list.append(data_set[i][1])

    total_loss = 0
    total_size = 0

    #cross_validationごとのclassification_report
    report1 = ""
    report2 = ""
    report3 = ""

    conf1 = ""
    conf2 = ""
    conf3 = ""

    pre_score = 0
    re_score = 0
    f_score = 0

    #cross_validation
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    #loss function
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(alpha=0.25, gamma=2)

    for fold_idx, idx in enumerate(kf.split(data_set, label_list)):

        # モデルを構築
        # net = models.compare(pretrained=True)
        net = models.resnet34(pretrained=True)

        # print(net)

        #vgg16の全結合
        # num_ftrs = net.classifier[6].in_features
        # net.classifier[6] = nn.Linear(num_ftrs, 2)

        #resnetの全結合
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 2)

        # print(net)
        net = net.to(device)

        print("ネットワーク設定完了：学習をtrainモードで開始します")

        # モデルの重みを読み込み
        # net.load_state_dict(torch.load(weight_pash), strict=False)

        # 最適化
        optimizer = optim.SGD(net.parameters(), lr=0.001)

        train_idx, valid_idx = idx

        train_loader = torch.utils.data.DataLoader(Subset(data_set, train_idx), batch_size=16, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(Subset(data_set_val, valid_idx), batch_size=1, shuffle=False)

        # trainモードで開始
        net.train()

        # modelの全体像を表示
        summary(net, input_size=(3, 200, 200))

        train_loss_value = []  # trainingのlossを保持するlist
        train_acc_value = []  # trainingのaccuracyを保持するlist

        # 同じデータを回学習します
        for epoch in range(10):
            print("epoch =", epoch + 1)

            # 今回の学習効果を保存するための変数
            running_loss = 0.0

            for batch_idx, data in enumerate(train_loader):  # dataがラベルと画像情報の2つの情報を持つ

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

            #     if batch_idx % 1 == 0:
            #         now = datetime.datetime.now()
            #         print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
            #             now, epoch, batch_idx * len(inputs), len(dataloader.dataset), 100. * batch_idx / len(dataloader),
            #                         total_loss / total_size))
            # train_loss_value.append(running_loss * 50 / len(dataloader.dataset))  # traindataのlossをグラフ描画のためにlistに保持
            # print("running_loss=", running_loss * 50 / len(dataloader.dataset))
                if batch_idx % 1 == 0:
                    now = datetime.datetime.now()
                    print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                        now, epoch+1, batch_idx * len(inputs), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                                    total_loss / total_size))
            train_loss_value.append(running_loss / len(train_loader.dataset))  # traindataのlossをグラフ描画のためにlistに保持
            print("train_loss=", running_loss / len(train_loader))

        #評価モード
        net.eval()
        print("ネットワークをevalに変更します")

        #test
        with torch.no_grad():
            pred = []
            Y = []
            target_name = ["correct 0", "broken 1"]
            for batch_idx, data in enumerate(valid_loader):
                input_val, label_val, image_name = data
                input_val = input_val.to(device)
                label_val = label_val.to(device)
                image_name = image_name[0].replace("'", "").replace(",", "")

                output_val = net(input_val)
                pred += [int(l.argmax()) for l in output_val]
                Y += [int(l) for l in label_val]

                if Y[-1] != pred[-1]:
                    if Y[-1] == 1:
                        mis_image = cv2.imread(os.path.join(train_data_dir, "broken/{filename}".format(filename=image_name)))
                        # print(os.path.join(train2, "{filename}".format(filename=image_name)))
                        cv2.imwrite(os.path.join(img_save_dir, "broken/{filename}".format(filename=image_name)), mis_image)
                    else:
                        mis_image = cv2.imread(os.path.join(train_data_dir, "correct/{filename}".format(filename=image_name)))
                        # print(os.path.join(train2, "{filename}".format(filename=image_name)))
                        cv2.imwrite(os.path.join(img_save_dir, "correct/{filename}".format(filename=image_name)), mis_image)

            # save_imageはネットの出力を保存する
            # save_image(output_val, os.path.join(img_save_dir, "{filename}".format(filename=image_name)), nrow=1)

            pre_score += precision_score(Y, pred)
            re_score += recall_score(Y, pred)
            f_score += f1_score(Y, pred)

            print(classification_report(Y, pred, target_names=target_name))
            print(confusion_matrix(Y, pred))

            if fold_idx == 0:
                report1 = classification_report(Y, pred, target_names=target_name)
                conf1 = confusion_matrix(Y, pred)
                # モデルを保存
                model_path = 'model_cross1.pth'
                torch.save(net.state_dict(), model_path)
            elif fold_idx == 1:
                report2 = classification_report(Y, pred, target_names=target_name)
                conf2 = confusion_matrix(Y, pred)
                # モデルを保存
                model_path = 'model_cross2.pth'
                torch.save(net.state_dict(), model_path)
            else:
                report3 = classification_report(Y, pred, target_names=target_name)
                conf3 = confusion_matrix(Y, pred)
                # モデルを保存
                model_path = 'model_cross3.pth'
                torch.save(net.state_dict(), model_path)

    conf_exp = "TN FP\nFN TP"

    print("====cross1の結果====")
    print(report1)
    print(conf1, "\n", conf_exp)
    print("====cross2の結果====")
    print(report2)
    print(conf2, "\n", conf_exp)
    print("====cross3の結果====")
    print(report3)
    print(conf3, "\n", conf_exp)

    print("broken_precision=", pre_score / 3)
    print("broken_recall=", re_score / 3)
    print("broken_f1=", f_score / 3)

    # モデルを保存
    # model_path = 'model.pth'
    # torch.save(net.state_dict(), model_path)


if __name__ == "__main__":

    train_data_dir = '/home/toui/デスクトップ/ori/add_testUseGan'
    # train_data_dir = '/home/toui/デスクトップ/ori/add_testUse'

    img_save_dir = "/home/toui/PycharmProjects/toui_pytorch/compare/img"

    shutil.rmtree(img_save_dir)

    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(os.path.join(img_save_dir, "broken"), exist_ok=True)
    os.makedirs(os.path.join(img_save_dir, "correct"), exist_ok=True)

    # weight_pash = "vgg16_bn.pth"

    device = torch.device("cuda:0")

    main()
    print("completed")
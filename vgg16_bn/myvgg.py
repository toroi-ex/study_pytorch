import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms

from torchsummary import summary

from pathlib import Path

from PIL import Image

import datetime

from vgg16_bn.model import vgg16_bn_test


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


train_data_dir = '/home/toui/デスクトップ/ori/add_data2/DATA3_test/test'
valid_data_dir = '/home/toui/デスクトップ/ori/add_data2/DATA3_test/test200'

# training set
data_set = MyDataset(200, dir_path=train_data_dir)
dataloader = torch.utils.data.DataLoader(data_set, batch_size=16, shuffle=True)

# validation set
# validset = torchvision.datasets.ImageFolder(valid_data_dir, transform=transform_valid)
# validloader = torch.utils.data.DataLoader(validset, batch_size=16, shuffle=False)

# classes = ("broken", "correct")



# TRAINING_DATA_RATIO = 0.5  # データの何％を訓練【Training】用に？ (残りは精度検証【Validation】用) ： 50％
# DATA_NOISE = 0.0           # ノイズ： 0％
#
# # 定義済みの定数を引数に指定して、データを生成する
# data_list = pg.generate_data(PROBLEM_DATA_TYPE, DATA_NOISE)
#
# # データを「訓練用」と「精度検証用」を指定の比率で分割し、さらにそれぞれを「データ（X）」と「教師ラベル（y）」に分ける
# X_train, y_train, X_valid, y_valid = pg.split_data(data_list, training_size=TRAINING_DATA_RATIO)

# model_path = "model.pth"

def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        # これを有効にしないと、計算した勾配が毎回異なり、再現性が担保できない。
        torch.backends.cudnn.deterministic = True
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


device = torch.device("cuda:0")
# device = get_device(use_gpu=True)

train_loss_value=[]      #trainingのlossを保持するlist
train_acc_value=[]       #trainingのaccuracyを保持するlist

weight_pash = "vgg16_bn.pth"

if __name__ == "__main__":

    # net = myVGG()
    # model = models.vgg16_bn(pretrained=True).features
    # net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    # url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'

    # state_dict = load_state_dict_from_url(url=url,progress=True)
    # net.load_state_dict(state_dict, strict=False)

    net = vgg16_bn_test().to(device)

    print("ネットワーク設定完了：学習をtrainモードで開始します")

    net.to(device)
    net.load_state_dict(torch.load(weight_pash), strict=False)

    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    summary(net, input_size=(3, 200, 200))

    total_loss = 0
    total_size = 0

    # 同じデータを  回学習します
    for epoch in range(10):
        print("epoch =", epoch+1)

        # 今回の学習効果を保存するための変数
        running_loss = 0.0

        for batch_idx, data in enumerate(dataloader): #dataがラベルと画像情報の2つの情報を持つ、(data,label)と書くこともできる
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

            if batch_idx % 1 == 0:
                now = datetime.datetime.now()
                print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                    now,epoch, batch_idx * len(inputs), len(dataloader.dataset), 100. * batch_idx / len(dataloader),
                    total_loss / total_size))
        train_loss_value.append(running_loss * 50 / len(dataloader.dataset)) #traindataのlossをグラフ描画のためにlistに保持
        print("running_loss=",running_loss * 50 / len(dataloader.dataset))

    plt.plot(range(10), train_loss_value)
    plt.xlim(0, 10)
    plt.ylim(0, 3)
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.legend(['train loss'])
    plt.title('loss')
    plt.savefig("loss_image.png")
    plt.clf()

    # モデルを保存
    model_path = 'model.pth'
    torch.save(net.state_dict(), model_path)

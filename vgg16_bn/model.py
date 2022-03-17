import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class myVGG(nn.Module):

    def __init__(self):
        super(myVGG, self).__init__()

        self.conv01 = nn.Conv2d(3, 64, 3)
        self.conv02 = nn.Conv2d(64, 64, 3)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv03 = nn.Conv2d(64, 128, 3)
        self.conv04 = nn.Conv2d(128, 128, 3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv05 = nn.Conv2d(128, 256, 3)
        self.conv06 = nn.Conv2d(256, 256, 3)
        self.conv07 = nn.Conv2d(256, 256, 3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv08 = nn.Conv2d(256, 512, 3)
        self.conv09 = nn.Conv2d(512, 512, 3)
        self.conv10 = nn.Conv2d(512, 512, 3)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv11 = nn.Conv2d(512, 512, 3)
        self.conv12 = nn.Conv2d(512, 512, 3)
        self.conv13 = nn.Conv2d(512, 512, 3)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.avepool1 = nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 5)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv01(x))
        x = F.relu(self.conv02(x))
        x = self.pool1(x)

        x = F.relu(self.conv03(x))
        x = F.relu(self.conv04(x))
        x = self.pool2(x)

        x = F.relu(self.conv05(x))
        x = F.relu(self.conv06(x))
        x = F.relu(self.conv07(x))
        x = self.pool3(x)

        x = F.relu(self.conv08(x))
        x = F.relu(self.conv09(x))
        x = F.relu(self.conv10(x))
        x = self.pool4(x)

        # x = F.relu(self.conv11(x))
        # x = F.relu(self.conv12(x))
        # x = F.relu(self.conv13(x))
        # x = self.pool5(x)

        x = self.avepool1(x)

        # 行列をベクトルに変換
        x = x.view(-1, 512 * 7 * 7)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

class vgg16_bn_test(nn.Module):
    def __init__(self):
        super(vgg16_bn_test, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            # nn.MaxPool2d(2, 2),

            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=False),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=False),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=False),
            # nn.MaxPool2d(2, 2),
        )

        # self.avepool1 = nn.AdaptiveAvgPool2d((7,7))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))#grobal average pooling

        #grobal average pooling
        self.classifier = nn.Sequential(
            # nn.Linear(512, 2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(25088, 4096),
        #     nn.Linear(4096, 4096),
        #     nn.Linear(4096, 2),
        # )


        # self.features = models.vgg16_bn(pretrained=True).features
    # def _get_conv_output(self, shape):
    #    bs = 1
    #    input_ = Variable(torch.rand(bs, *shape))
    #    output_feat = self._forward_features(input_)
    #    n_size = output_feat.data.size(1)
    #    return n_size

    def forward(self, x):
        x = self.features(x)
        # x = self.avepool1(x)
        x = self.gap(x) #grobal average pooling
        # x = x.view(-1, 512 * 7 * 7)
        x = x.view(-1, 512 * 1 * 1) #grobal average pooling
        x = self.classifier(x)
        return x
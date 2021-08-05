import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class vgg16_bn_test(nn.Module):
    def __init__(self):
        super(vgg16_bn_test, self).__init__()

        self.conv11 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn11 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn12 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool11 = nn.MaxPool2d(2, 2)

        self.short1 = self.shortcut(channel_in=3, channel_out=64)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn21 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn22 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool21 = nn.MaxPool2d(2, 2)

        self.short2 = self.shortcut(channel_in=64, channel_out=128)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn31 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn32 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn33 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool31 = nn.MaxPool2d(2, 2)

        self.short3 = self.shortcut(channel_in=128, channel_out=256)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn41 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn42 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn43 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # nn.MaxPool2d(2, 2),

        self.short4 = self.shortcut(channel_in=256, channel_out=512)

        self.relu = nn.ReLU()

        # self.avepool1 = nn.AdaptiveAvgPool2d((7,7))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # grobal average pooling

        # grobal average pooling
        self.classifier = nn.Sequential(
            nn.Linear(512, 2),
            # nn.Linear(256, 2)
        )

    def shortcut(self, channel_in, channel_out):
        if channel_in != channel_out:
            return self.adjust(channel_in, channel_out)
        else:
            return lambda x: x

    def adjust(self, channel_in, channel_out):
        return nn.Conv2d(channel_in, channel_out, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        y = self.short1(x)
        x = F.relu(self.conv11(x))
        x = self.bn11(x)
        x = self.conv12(x)
        x = self.relu(x + y)
        x = self.bn12(x)
        x = self.pool11(x)

        y = self.short2(x)
        x = F.relu(self.conv21(x))
        x = self.bn21(x)
        x = self.conv22(x)
        x = self.relu(x + y)
        x = self.bn22(x)
        x = self.pool21(x)

        y = self.short3(x)
        x = F.relu(self.conv31(x))
        x = self.bn31(x)
        x = F.relu(self.conv32(x))
        x = self.bn32(x)
        x = self.conv33(x)
        x = self.relu(x + y)
        x = self.bn33(x)
        x = self.pool31(x)

        y = self.short4(x)
        x = F.relu(self.conv41(x))
        x = self.bn41(x)
        x = F.relu(self.conv42(x))
        x = self.bn42(x)
        x = self.conv43(x)
        x = self.relu(x + y)
        x = self.bn43(x)

        x = self.gap(x)  # grobal average pooling
        x = x.view(-1, 512 * 1 * 1)  # grobal average pooling
        x = self.classifier(x)

        return x

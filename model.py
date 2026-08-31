import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channle, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            self, in_channel, out_channle, 3, stride=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channle)
        self.conv2 = nn.Conv2d(
            self, out_channle, out_channle, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channle)
        self.shortcut = nn.Sequential()
        self.use_shortcut = stride != 1 or in_channel != out_channle
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channle, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channle))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x) if self.use_shortcut else x
        out_adder = out + shortcut
        out = torch.relu(out_adder)

        return out


class AudioCNN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(
                64), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1))
        self.layer1 = nn.ModuleList([ResidualBlock(64, 64) for i in range(3)])
        self.layer2 = nn.ModuleList(
            [ResidualBlock(64 if i == 0 else 128, 128, stride=2 if i == 0 else 1) for i in range(4)])
        self.layer3 = nn.ModuleList(
            [ResidualBlock(128 if i == 0 else 256, 256, stride=2 if i == 0 else 1) for i in range(6)])
        self.layer4 = nn.ModuleList(
            [ResidualBlock(256 if i == 0 else 512, 512, stride=2 if i == 0 else 1) for i in range(3)])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.layer2:
            x = block(x)
        for block in self.layer3:
            x = block(x)
        for block in self.layer3:
            x = block(x)
        for block in self.layer4:
            x = block(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

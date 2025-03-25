import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = CNNBlock(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=(3, 1, 1),
            padding=0,
        )
        self.conv2 = CNNBlock(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=(3, 1, 1),
            padding=0,
        )
        self.conv3 = CNNBlock(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=(3, 1, 1),
            padding=0,
        )
        self.conv4 = CNNBlock(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=(3, 1, 1),
            padding=0,
        )
        self.conv5 = CNNBlock(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=(3, 1, 1),
            padding=0,
        )
        self.conv6 = CNNBlock(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=(3, 1, 1),
            padding=0,
        )
        self.conv7 = CNNBlock(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=(3, 1, 1),
            padding=0,
        )
        self.conv8 = CNNBlock(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=(3, 1, 1),
            padding=0,
        )
        self.conv9 = CNNBlock(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=(3, 1, 1),
            padding=0,
        )
        self.conv10 = CNNBlock(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=(3, 1, 1),
            padding=0,
        )
        self.conv11 = CNNBlock(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=(3, 1, 1),
            padding=0,
        )
        self.conv12 = CNNBlock(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=(3, 1, 1),
            padding=0,
        )

        self.pool8 = nn.AvgPool2d(12)
        self.pool9 = nn.AvgPool2d(12)
        self.pool10 = nn.AvgPool2d(12)
        self.pool11 = nn.AvgPool2d(12)
        self.pool12 = nn.AvgPool2d(12)

        self.fc = nn.Linear(9216, 128)

    def forward(self, x):
        identity = x

        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv2(out2)
        out4 = self.conv2(out3)
        out5 = self.conv2(out4)
        out6 = self.conv2(out5)
        out7 = self.conv2(out6)
        out7 += out5
        out8 = self.conv2(out7)
        out8 += out4
        out9 = self.conv2(out8)
        out9 += out3
        out10 = self.conv2(out9)
        out10 += out2
        out11 = self.conv2(out10)
        out11 += out1
        out12 = self.conv2(out11)

        # TODO global pooling then concatenate then classification with fully connected network
        return out12

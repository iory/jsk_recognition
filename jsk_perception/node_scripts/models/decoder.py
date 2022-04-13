#!/usr/bin/env python


import torch.nn as nn


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation=1, bias=True):
        super(UpsampleBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size,
                               stride, padding, dilation=dilation, bias=bias),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        output = self.layers(x)

        return output


class ResNet18Decoder(nn.Module):
    """Decoder module for ResNet18
    Args:
        in_channels (int, optional): number of input channels(default: 512)
        out_channels (int, optional): number of output channels(deafult: 3)
    """

    def __init__(self, in_channels=512, out_channels=3):
        super(ResNet18Decoder, self).__init__()

        if in_channels >= 512:
            num_layers = 5
        elif in_channels >= 256 and in_channels < 512:
            num_layers = 4
        elif in_channels >= 128 and in_channels < 256:
            num_layers = 3

        layers = []
        last_ch = in_channels
        for _ in range(num_layers):
            layers.append(UpsampleBlock(last_ch, last_ch // 2, 4, 2, 1))
            last_ch = last_ch // 2

        layers.append(nn.Conv2d(last_ch, out_channels, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)

        return x


class ResNet50Decoder(nn.Module):
    def __init__(self, in_channels=2048, out_channels=3):
        super(ResNet50Decoder, self).__init__()

        if in_channels >= 2048:
            num_layers = 5
        elif in_channels >= 1024 and in_channels < 2048:
            num_layers = 4
        elif in_channels >= 512 and in_channels < 1024:
            num_layers = 3
        else:
            raise ValueError(
                "unexpected number of channels {}".format(in_channels))

        layers = []
        last_ch = in_channels
        for _ in range(num_layers):
            layers.append(UpsampleBlock(last_ch, last_ch // 2, 4, 2, 1))
            last_ch = last_ch // 2

        layers.append(nn.Conv2d(last_ch, out_channels, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)

        return x

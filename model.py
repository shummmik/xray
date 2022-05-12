import torch
import torch.nn as nn
import torch.nn.functional as F


class UnetConv(nn.Module):
    def __init__(self, in_channels: int, out_features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_features,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.batchn1 = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.batchn2 = nn.BatchNorm2d(out_features)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.batchn1(self.conv1(x)))
        x = F.relu(self.batchn2(self.conv2(x)))
        return x


class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32) -> None:
        super().__init__()

        self.down_conv1 = UnetConv(in_channels, init_features)  # 32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128

        self.down_conv2 = UnetConv(init_features, init_features * 2)  # 64
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64

        self.down_conv3 = UnetConv(init_features * 2, init_features * 4)  # 128
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32

        self.down_conv4 = UnetConv(init_features * 4, init_features * 8)  # 256
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16

        self.down_conv5 = UnetConv(init_features * 8, init_features * 16)  # 512
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8

        self.bottom = UnetConv(init_features * 16, init_features * 32)  # 1024

        self.conv_transp5 = nn.ConvTranspose2d(
            in_channels=init_features * 32,
            out_channels=init_features * 16,
            kernel_size=2,
            stride=2,
        )  # 16
        self.up_conv5 = UnetConv(
            (init_features * 16) * 2, init_features * 16
        )  # 512                                                                                     #2

        self.conv_transp4 = nn.ConvTranspose2d(
            in_channels=init_features * 16,
            out_channels=init_features * 8,
            kernel_size=2,
            stride=2,
        )  # 32
        self.up_conv4 = UnetConv((init_features * 8) * 2, init_features * 8)  # 256

        self.conv_transp3 = nn.ConvTranspose2d(
            in_channels=init_features * 8,
            out_channels=init_features * 4,
            kernel_size=2,
            stride=2,
        )  # 64
        self.up_conv3 = UnetConv((init_features * 4) * 2, init_features * 4)  # 128

        self.conv_transp2 = nn.ConvTranspose2d(
            in_channels=init_features * 4,
            out_channels=init_features * 2,
            kernel_size=2,
            stride=2,
        )  # 128
        self.up_conv2 = UnetConv((init_features * 2) * 2, init_features * 2)  # 64

        self.conv_transp1 = nn.ConvTranspose2d(
            in_channels=init_features * 2,
            out_channels=init_features,
            kernel_size=2,
            stride=2,
        )  # 256
        self.up_conv1 = UnetConv(init_features * 2, init_features)  # 32

        self.conv = nn.Conv2d(
            in_channels=init_features, out_channels=out_channels, kernel_size=1
        )

        self.softnmax = nn.Softmax(dim=1)

    def forward(self, x) -> torch.Tensor:
        down1 = self.down_conv1(x)
        down2 = self.down_conv2(self.pool1(down1))
        down3 = self.down_conv3(self.pool2(down2))
        down4 = self.down_conv4(self.pool3(down3))
        down5 = self.down_conv5(self.pool4(down4))

        bottom = self.bottom(self.pool5(down5))

        up5 = self.conv_transp5(bottom)
        up5 = torch.cat((up5, down5), dim=1)
        up5 = self.up_conv5(up5)

        up4 = self.conv_transp4(up5)
        up4 = torch.cat((up4, down4), dim=1)
        up4 = self.up_conv4(up4)

        up3 = self.conv_transp3(up4)
        up3 = torch.cat((up3, down3), dim=1)
        up3 = self.up_conv3(up3)

        up2 = self.conv_transp2(up3)
        up2 = torch.cat((up2, down2), dim=1)
        up2 = self.up_conv2(up2)

        up1 = self.conv_transp1(up2)
        up1 = torch.cat((up1, down1), dim=1)
        up1 = self.up_conv1(up1)

        out = self.conv(up1)
        return out

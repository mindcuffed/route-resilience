import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(3, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.dec3 = DoubleConv(128 + 64, 64)
        self.dec2 = DoubleConv(64 + 32, 32)
        self.dec1 = DoubleConv(32 + 16, 16)

        self.output = nn.Conv2d(
            16,
            1,
            kernel_size=1
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        bottleneck = self.bottleneck(
            self.pool(enc3)
        )

        dec3 = F.interpolate(
            bottleneck,
            size=enc3.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        dec3 = self.dec3(
            torch.cat([dec3, enc3], dim=1)
        )

        dec2 = F.interpolate(
            dec3,
            size=enc2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        dec2 = self.dec2(
            torch.cat([dec2, enc2], dim=1)
        )

        dec1 = F.interpolate(
            dec2,
            size=enc1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        dec1 = self.dec1(
            torch.cat([dec1, enc1], dim=1)
        )

        return self.output(dec1)
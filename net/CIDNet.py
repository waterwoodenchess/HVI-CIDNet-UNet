import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from net.HVI_transform import RGB_HVI


def build_norm(channels, use_norm):
    if use_norm:
        return nn.BatchNorm2d(channels)
    return nn.Identity()


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_norm(out_channels, use_norm),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_norm(out_channels, use_norm),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, use_norm=use_norm),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_norm=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, use_norm=use_norm)

    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        channels=[36, 36, 72, 144],
        heads=[1, 2, 4, 8],
        norm=False,
    ):
        super().__init__()

        del heads  # kept only for constructor compatibility

        ch1, ch2, ch3, ch4 = channels
        bottleneck_channels = ch4 * 2

        self.inc = DoubleConv(3, ch1, use_norm=norm)
        self.down1 = Down(ch1, ch2, use_norm=norm)
        self.down2 = Down(ch2, ch3, use_norm=norm)
        self.down3 = Down(ch3, ch4, use_norm=norm)
        self.bottleneck = DoubleConv(ch4, bottleneck_channels, use_norm=norm)

        self.up1 = Up(bottleneck_channels, ch4, ch4, use_norm=norm)
        self.up2 = Up(ch4, ch3, ch3, use_norm=norm)
        self.up3 = Up(ch3, ch2, ch2, use_norm=norm)
        self.up4 = Up(ch2, ch1, ch1, use_norm=norm)
        self.outc = OutConv(ch1, 3)

        # Keep the HVI transform object for compatibility with the
        # existing training, evaluation and app scripts.
        self.trans = RGB_HVI()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

    def HVIT(self, x):
        return self.trans.HVIT(x)

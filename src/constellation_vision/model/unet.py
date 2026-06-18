"""A compact UNet-lite for per-pixel defect segmentation, from scratch.

No pretrained weights and no external model download: the network is small
enough that `make train` fits it to a real held-out IoU in a few minutes on CPU.

Architecture (two-level UNet):

    in -> enc1 -----------------------------+--> dec1 -> head -> logits
            |                               |
          pool -> enc2 ---+--> up -> dec1 --+
                          |          ^
                        pool         |
                          |          |
                        bottleneck --+

Width doubles each downsample from ``base`` (config.BASE_WIDTH). Skip
connections concatenate encoder features into the decoder. Output is a
per-pixel class logit map of shape [N, NUM_CLASSES, H, W].
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .. import config, schema


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetLite(nn.Module):
    def __init__(
        self,
        in_channels: int = config.IN_CHANNELS,
        num_classes: int = schema.NUM_CLASSES,
        base: int = config.BASE_WIDTH,
    ):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)                  # [N, base,   H,   W]
        e2 = self.enc2(self.pool(e1))      # [N, 2base,  H/2, W/2]
        b = self.bottleneck(self.pool(e2)) # [N, 4base,  H/4, W/4]

        d2 = self.up2(b)                          # [N, 2base, H/2, W/2]
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)                          # [N, base, H, W]
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)               # [N, num_classes, H, W]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

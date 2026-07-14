"""Anisotropy-aware 3D U-Net.

Voxels in this data are 2nm x 2nm x 30nm (15x coarser in z than in-plane, see
PLAN.md section 0). Pooling therefore reduces the in-plane axes before it
touches z, instead of treating the volume as isotropic.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lsd import LSD_CHANNELS

# One entry per downsampling stage: (z_stride, y_stride, x_stride).
# First stage is in-plane-only; deeper stages become isotropic in voxel space.
DEFAULT_STRIDES = [(1, 2, 2), (2, 2, 2), (2, 2, 2)]


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        groups = min(groups, out_ch)
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: tuple[int, int, int]):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=stride, stride=stride)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, stride: tuple[int, int, int]):
        super().__init__()
        self.stride = stride
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SeededUNet3D(nn.Module):
    """Input: (B, 2, Z, Y, X) = [raw_EM, seed_heatmap].
    Output: (mask_logits, lsd_pred) -- (B, 1, Z, Y, X) and (B, lsd_channels, Z, Y, X),
    or (mask_logits, None) if predict_lsd=False.

    The LSD head (2026-07-13, see lsd.py) shares the whole decoder with the mask
    head and only diverges at the final 1x1 conv, matching the "MTLSD" (multitask)
    architecture in Sheridan et al. 2023 -- the simplest of the paper's three
    variants, since it needs no second network/auto-context cascade."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 24,
        strides: list[tuple[int, int, int]] = None,
        predict_lsd: bool = True,
        lsd_channels: int = LSD_CHANNELS,
    ):
        super().__init__()
        strides = strides or DEFAULT_STRIDES
        chs = [base_channels * (2**i) for i in range(len(strides) + 1)]

        self.stem = DoubleConv(in_channels, chs[0])
        self.downs = nn.ModuleList(
            [Down(chs[i], chs[i + 1], strides[i]) for i in range(len(strides))]
        )
        self.ups = nn.ModuleList(
            [
                Up(chs[i + 1], chs[i], chs[i], strides[i])
                for i in reversed(range(len(strides)))
            ]
        )
        self.mask_head = nn.Conv3d(chs[0], out_channels, kernel_size=1)
        self.predict_lsd = predict_lsd
        self.lsd_head = nn.Conv3d(chs[0], lsd_channels, kernel_size=1) if predict_lsd else None

    def forward(self, x):
        skips = [self.stem(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        x = skips[-1]
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)

        mask_logits = self.mask_head(x)
        lsd_pred = self.lsd_head(x) if self.predict_lsd else None
        return mask_logits, lsd_pred

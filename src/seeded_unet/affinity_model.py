"""Same anisotropy-aware 3D U-Net trunk as model.py, but for the affinity+LSD
model: no seed input (this model is never told which object it's looking at),
and the head predicts per-voxel affinities instead of a single seeded mask."""
from __future__ import annotations

import torch.nn as nn

from .lsd import LSD_CHANNELS
from .model import DEFAULT_STRIDES, DoubleConv, Down, Up


class AffinityLSDUNet3D(nn.Module):
    """Input: (B, 1, Z, Y, X) = [raw_EM] only.
    Output: (affinity_logits, lsd_pred) -- (B, num_offsets, Z, Y, X) and
    (B, lsd_channels, Z, Y, X), or (affinity_logits, None) if predict_lsd=False.

    Shares the exact same trunk/decoder shape as SeededUNet3D (model.py) --
    only the input channel count and head are different -- so LSD continues
    to act as an auxiliary task on the same shared features, matching the
    "MTLSD" architecture in Sheridan et al. 2023."""

    def __init__(
        self,
        in_channels: int = 1,
        num_offsets: int = 6,
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
        self.affinity_head = nn.Conv3d(chs[0], num_offsets, kernel_size=1)
        self.predict_lsd = predict_lsd
        self.lsd_head = nn.Conv3d(chs[0], lsd_channels, kernel_size=1) if predict_lsd else None

    def forward(self, x):
        skips = [self.stem(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        x = skips[-1]
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)

        affinity_logits = self.affinity_head(x)
        lsd_pred = self.lsd_head(x) if self.predict_lsd else None
        return affinity_logits, lsd_pred

"""Local shape descriptors (LSDs): an auxiliary training signal that forces
the network to use its whole receptive field rather than just local
per-voxel boundary evidence (Sheridan et al. 2023, "Local shape descriptors
for neuron segmentation", Nature Methods).

Added 2026-07-13 for a specific reason, not just because the paper looked
useful: real annotation gaps exist in the traced skeletons -- e.g. dust
obscuring a neuron in one section, so no node could be placed there even
though the neuron obviously continues through it. A model that only reads
local per-voxel evidence has no way to notice a gap like that; the paper's
own framing is that predicting local shape statistics (size, center-of-mass
offset, coordinate covariance) forces the network to use surrounding context
rather than a handful of center voxels, which is exactly the situation where
local evidence briefly disappears but the object obviously continues
(the paper's introduction makes this same argument for ambiguous/weak
boundary evidence in general).

This adapts the paper's formulation to our simpler seeded/single-instance
setting. The paper computes LSDs per-segment across a whole densely-labeled
volume (needing per-segment-id bookkeeping via a ball-intersection). Here, a
training patch already comes with exactly one target binary mask (the seeded
instance), so the LSD only ever needs to be computed with respect to that
one mask -- no segment-id bookkeeping needed. The paper's window is a
uniform ball out to a fixed radius; here it's a Gaussian (reusing the same
anisotropic-sigma-in-nm convention as the seed heatmap in seeds.py), since
scipy's separable Gaussian filter is already the pattern used elsewhere in
this codebase and is cheap to compute per-sample.

Normalization: raw offsets/covariances are computed in voxel units, which
have a wide, patch-size-dependent dynamic range -- if the LSD loss is going
to be added to a Dice+BCE loss that lives in roughly [0, 1], it needs to live
on a comparable scale too. Offsets are divided by sigma_voxel (making them
"offset in units of the LSD window size"), covariances by sigma_voxel_i *
sigma_voxel_j; `size` = a normalized Gaussian filter of a 0/1 mask, which is
already naturally in [0, 1] with no further scaling needed.

Performance: measured directly on a real 32x256x256 patch, 10 full-resolution
gaussian_filter passes cost ~2-5s depending on sigma -- dominated by the sheer
2M-voxel array size, not sigma (a much smaller sigma barely moved the needle).
That's a real problem run 8ish times per training instance per epoch. Fixed by
computing at a coarser resolution in x/y (downsampled via true block-averaging,
not naive pixel-skipping -- confirmed empirically that naive-stride sampling
after a mismatched sliding-window filter can miss a thin 2-voxel sliver
entirely, whereas block-averaging captures it as a fractional value every
time) and upsampling the result back with linear interpolation. Both the
offset and covariance channels are pre-normalized by sigma (in the same
downsampled units), so no rescaling is needed after upsampling -- only `size`
and the raw block-average step operate directly on the mask. This measured
~25x faster (2-5s -> ~150-200ms) with visibly identical output for a test
mask by eye. z is left at full resolution (already only ~32 voxels deep, and
the z sigma is usually small anyway given this data's anisotropy).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter

LSD_CHANNELS = 10  # 1 size + 3 center-offset (z,y,x) + 6 covariance (zz,yy,xx,zy,zx,yx)
_AXES = ("z", "y", "x")
DEFAULT_DOWNSAMPLE_YX = 4


@lru_cache(maxsize=8)
def _coord_grids(shape_zyx: tuple[int, int, int]):
    zz, yy, xx = np.meshgrid(
        np.arange(shape_zyx[0]), np.arange(shape_zyx[1]), np.arange(shape_zyx[2]), indexing="ij"
    )
    return {"z": zz.astype(np.float32), "y": yy.astype(np.float32), "x": xx.astype(np.float32)}


def _block_average_yx(a: np.ndarray, factor: int) -> np.ndarray:
    """True block-average downsample in y/x (not a sliding-window filter + stride,
    which can miss thin features if the window and stride happen to misalign --
    confirmed empirically). Requires y and x evenly divisible by `factor`."""
    z, y, x = a.shape
    return a.reshape(z, y // factor, factor, x // factor, factor).mean(axis=(2, 4))


def compute_lsd_target(
    mask: np.ndarray,
    sigma_voxel_zyx: tuple[float, float, float],
    downsample_yx: int = DEFAULT_DOWNSAMPLE_YX,
) -> np.ndarray:
    """mask: (Z, Y, X) binary array for one instance. Returns (LSD_CHANNELS, Z, Y, X)
    float32 -- local shape descriptors of `mask`, computed the same way at every
    voxel regardless of whether that voxel itself is foreground or background
    (there's only ever one mask to describe here, unlike the paper's
    per-segment-id version, so identity doesn't switch by voxel)."""
    y, x = mask.shape[1], mask.shape[2]
    factor = downsample_yx if (y % downsample_yx == 0 and x % downsample_yx == 0) else 1

    if factor > 1:
        b = _block_average_yx(mask.astype(np.float32), factor)
        sigma_voxel_zyx = (sigma_voxel_zyx[0], sigma_voxel_zyx[1] / factor, sigma_voxel_zyx[2] / factor)
    else:
        b = mask.astype(np.float32)

    coords = _coord_grids(b.shape)
    sigma = {"z": sigma_voxel_zyx[0], "y": sigma_voxel_zyx[1], "x": sigma_voxel_zyx[2]}

    def smooth(a):
        return gaussian_filter(a, sigma=sigma_voxel_zyx)

    eps = 1e-6
    # Below this, the window barely overlaps the mask at all -- there's no real
    # local shape to describe there, and dividing by a near-zero denominator
    # blows up into meaningless huge values (confirmed empirically: uncapped,
    # offsets many sigma wide showed up far from any mask). Defined as exactly
    # 0 there instead -- a neutral value carrying no false signal, everywhere the
    # window has essentially no evidence about this instance.
    valid_threshold = 1e-3
    denom = smooth(b)
    size = denom  # gaussian_filter of a 0/1 array with a normalized kernel is already in [0, 1]
    valid = denom > valid_threshold
    safe_denom = np.where(valid, denom, 1.0)

    raw_mean = {}
    for k in _AXES:
        numer = smooth(b * coords[k])
        # Defaulting to coords[k] itself (not 0) where invalid makes the offset
        # below come out exactly 0, without a second np.where.
        raw_mean[k] = np.where(valid, numer / safe_denom, coords[k])
    offset = {k: (raw_mean[k] - coords[k]) / (sigma[k] + eps) for k in _AXES}

    cov = {}
    for i, ki in enumerate(_AXES):
        for kj in _AXES[i:]:
            numer2 = smooth(b * coords[ki] * coords[kj])
            # Same trick: defaulting to raw_mean[ki] * raw_mean[kj] makes cov come out 0.
            raw_second = np.where(valid, numer2 / safe_denom, raw_mean[ki] * raw_mean[kj])
            cov[(ki, kj)] = (raw_second - raw_mean[ki] * raw_mean[kj]) / (sigma[ki] * sigma[kj] + eps)

    channels = [
        size,
        offset["z"], offset["y"], offset["x"],
        cov[("z", "z")], cov[("y", "y")], cov[("x", "x")],
        cov[("z", "y")], cov[("z", "x")], cov[("y", "x")],
    ]
    result = np.stack(channels, axis=0).astype(np.float32)

    if factor > 1:
        # Nearest-neighbor block-repeat, not scipy's linear zoom -- measured zoom's
        # true bilinear upsampling at ~1.7s for this array size, vs ~0.06s for repeat
        # (>25x). offset/cov are already normalized by sigma (in the same downsampled
        # units), so no further rescaling is needed; the resulting mild blockiness is
        # a fine tradeoff for what's already a soft, approximate auxiliary target.
        result = np.repeat(np.repeat(result, factor, axis=2), factor, axis=3)
    return result

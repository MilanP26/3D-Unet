"""Dense (whole-patch, multi-instance) training targets for the affinity+LSD
model -- the alternative to the seeded per-instance model in dataset.py/model.py.

Unlike the seeded model (one binary mask per patch, for one chosen instance),
this model is never told which object it's looking at: it predicts, for every
voxel and every neighbor offset, whether that pair belongs to the same real
instance ("affinity"), plus the same LSD auxiliary channels as before but now
computed per-voxel with respect to *that voxel's own* instance rather than one
fixed instance. Real per-neuron identity comes back in later, at inference
time, via seeded mutex watershed (affinity_infer.py) -- not from anything in
this file.
"""
from __future__ import annotations

import numpy as np

from .lsd import LSD_CHANNELS, compute_lsd_target

# (dz, dy, dx) in voxels. Short-range = adjacent-voxel connectivity (the core
# signal). Long-range offsets help mutex watershed bridge small gaps/noise and
# provide split evidence over a wider neighborhood -- chosen anisotropically
# (bigger steps in y/x than z) since z voxels are already ~15x coarser
# physically (PLAN.md section 0), so a z-offset of 1 already spans much more
# real distance than an x/y offset of 1.
DEFAULT_OFFSETS = [
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (2, 0, 0), (0, 8, 0), (0, 0, 8),
]


def _axis_slices(n: int, d: int) -> tuple[slice, slice]:
    """Source/destination slices for shifting an axis of length n by offset d."""
    if d >= 0:
        return slice(d, n), slice(0, n - d)
    return slice(0, n + d), slice(-d, n)


def compute_affinities(labels: np.ndarray, offsets=DEFAULT_OFFSETS) -> np.ndarray:
    """labels: (Z, Y, X) int array, 0 = background. Returns (len(offsets), Z, Y, X)
    float32, 1.0 where a voxel and its neighbor at that offset are the same
    real (nonzero) instance, else 0.0 -- including at the instance/background
    boundary and at the volume edge, both of which should never be merged."""
    shape = labels.shape
    aff = np.zeros((len(offsets),) + shape, dtype=np.float32)
    for i, (dz, dy, dx) in enumerate(offsets):
        z_src, z_dst = _axis_slices(shape[0], dz)
        y_src, y_dst = _axis_slices(shape[1], dy)
        x_src, x_dst = _axis_slices(shape[2], dx)
        shifted = np.zeros_like(labels)
        shifted[z_dst, y_dst, x_dst] = labels[z_src, y_src, x_src]
        aff[i] = (labels == shifted) & (labels != 0)
    return aff


def compute_lsd_target_dense(
    labels: np.ndarray,
    sigma_voxel_zyx: tuple[float, float, float],
    downsample_yx: int = 4,
) -> np.ndarray:
    """labels: (Z, Y, X) int array, 0 = background. Returns (LSD_CHANNELS, Z, Y, X):
    same per-voxel local shape descriptors as compute_lsd_target, but each
    voxel is described with respect to *its own* instance (Sheridan et al.'s
    actual per-segment formulation) rather than one instance fixed for the
    whole patch. Background voxels get all-zero LSD channels -- there's no
    object there to describe."""
    result = np.zeros((LSD_CHANNELS,) + labels.shape, dtype=np.float32)
    for inst_id in np.unique(labels):
        if inst_id == 0:
            continue
        mask = labels == inst_id
        per_instance = compute_lsd_target(mask, sigma_voxel_zyx, downsample_yx)
        result[:, mask] = per_instance[:, mask]
    return result

"""Synthetic seed-point sampling and seed-channel (heatmap) construction.

Training does not use real VAST skeleton coordinates (see PLAN.md section 3):
since ground-truth instance masks already exist for every training stack, we
simulate a human click by sampling a point inside the mask, biased toward its
interior/medial region via a distance transform (a click near the center of a
neuron is more realistic than one right on the boundary).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def seed_distribution(mask: np.ndarray, interior_bias: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Precomputes (coords, weights) for sampling an interior-biased seed
    point from `mask`. This is the expensive part (a distance transform) --
    callers should compute it ONCE per instance and cache it, not per draw.
    Cropped to the mask's bounding box first: some instances here span nearly
    the entire volume (long neurites), so running the distance transform over
    the full ~1024x1024xZ array on every sample would be far too slow."""
    if not mask.any():
        raise ValueError("Cannot sample a seed from an empty mask")

    nz = np.argwhere(mask)
    mins = nz.min(axis=0)
    maxs = nz.max(axis=0)
    sub_mask = mask[mins[0]:maxs[0] + 1, mins[1]:maxs[1] + 1, mins[2]:maxs[2] + 1]

    dist = distance_transform_edt(sub_mask)
    coords = np.argwhere(sub_mask) + mins  # back to full-volume coordinates
    weights = dist[sub_mask]
    if interior_bias > 0:
        weights = weights**interior_bias
    weights = weights / weights.sum()
    return coords, weights


def sample_from_distribution(
    coords: np.ndarray, weights: np.ndarray, rng: np.random.Generator
) -> tuple[int, int, int]:
    idx = rng.choice(len(coords), p=weights)
    z, y, x = coords[idx]
    return int(z), int(y), int(x)


def sample_seed_point(mask: np.ndarray, rng: np.random.Generator, interior_bias: float = 2.0) -> tuple[int, int, int]:
    """Convenience one-shot version of seed_distribution + sample_from_distribution.
    Recomputes the distance transform every call -- fine for a single lookup,
    but SeededPatchDataset caches seed_distribution() per instance instead of
    calling this repeatedly."""
    coords, weights = seed_distribution(mask, interior_bias)
    return sample_from_distribution(coords, weights, rng)


def gaussian_heatmap(
    shape_zyx: tuple[int, int, int],
    center_zyx: tuple[int, int, int],
    sigma_voxel_zyx: tuple[float, float, float],
) -> np.ndarray:
    """Dense 3D Gaussian blob, peak 1.0 at `center_zyx`, same shape as the
    patch. sigma is given per-axis in voxels so it can be set to compensate
    for anisotropic voxel size (e.g. a physically round blob under 2/2/30 nm
    voxels needs a much smaller z-sigma in voxel units than xy-sigma)."""
    zz, yy, xx = np.meshgrid(
        np.arange(shape_zyx[0]), np.arange(shape_zyx[1]), np.arange(shape_zyx[2]), indexing="ij"
    )
    cz, cy, cx = center_zyx
    sz, sy, sx = sigma_voxel_zyx
    exponent = (
        ((zz - cz) ** 2) / (2 * sz**2 + 1e-8)
        + ((yy - cy) ** 2) / (2 * sy**2 + 1e-8)
        + ((xx - cx) ** 2) / (2 * sx**2 + 1e-8)
    )
    return np.exp(-exponent).astype(np.float32)


def physical_sigma_to_voxels(sigma_nm: float, scale_nm_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert one physical sigma (nm) into per-axis voxel sigma (z, y, x),
    given the dataset's (x, y, z) nm/voxel scale."""
    sx, sy, sz = scale_nm_xyz
    return (sigma_nm / sz, sigma_nm / sy, sigma_nm / sx)

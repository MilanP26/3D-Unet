"""Training-time data augmentation for the affinity+LSD pipeline (affinity_dataset.py).

Sheridan et al. 2023 (the LSD paper) explicitly calls out rotation, reflection, translation,
and simulated section artifacts/misalignments as necessary augmentation for a network that
has to be robust to real serial-section EM defects, not just clean training crops -- the
same real-world motivation already behind adding LSD in the first place (dust obscuring a
neuron in one section, where the network must lean on context rather than per-voxel
evidence). All five are implemented here; rotation is restricted to 90-degree multiples so
it stays an exact voxel-grid operation (no interpolation blur on this data's anisotropic
voxels), unlike an arbitrary-angle rotation.
"""
from __future__ import annotations

import numpy as np


def random_flip(raw: np.ndarray, label: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Independently flips each of z/y/x with 50% probability."""
    for axis in (0, 1, 2):
        if rng.random() < 0.5:
            raw = np.flip(raw, axis=axis)
            label = np.flip(label, axis=axis)
    return np.ascontiguousarray(raw), np.ascontiguousarray(label)


def random_rotate90(raw: np.ndarray, label: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random 0/90/180/270-degree rotation in the x/y in-plane axes only -- rotating into/out
    of z would mix axes with very different physical scale (2nm in-plane vs 30nm z), which
    isn't a real transform this data could ever actually undergo."""
    k = int(rng.integers(0, 4))
    if k:
        raw = np.rot90(raw, k=k, axes=(1, 2))
        label = np.rot90(label, k=k, axes=(1, 2))
    return np.ascontiguousarray(raw), np.ascontiguousarray(label)


def random_translate_crop(
    padded_raw: np.ndarray,
    padded_label: np.ndarray,
    target_shape_zyx: tuple[int, int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """padded_raw/padded_label are larger than target_shape_zyx in y/x by `2*pad`; this picks
    a random sub-crop of exactly target_shape_zyx, i.e. a small random translation on top of
    the already-random patch center, applied identically to raw and label."""
    pz, py, px = target_shape_zyx
    fz, fy, fx = padded_raw.shape
    z0 = rng.integers(0, fz - pz + 1)
    y0 = rng.integers(0, fy - py + 1)
    x0 = rng.integers(0, fx - px + 1)
    sl = (slice(z0, z0 + pz), slice(y0, y0 + py), slice(x0, x0 + px))
    return padded_raw[sl], padded_label[sl]


def random_artifact(
    raw: np.ndarray, rng: np.random.Generator, prob_per_slice: float = 0.05, max_slices: int = 2
) -> np.ndarray:
    """Simulates a lost/corrupted EM section: zeroes out (or heavily degrades) a handful of
    z-slices' raw intensity, leaving the label untouched -- the object obviously continues
    through a real defect like this, which is exactly why LSD's broader-context signal
    matters. Raw only; never touches label."""
    raw = raw.copy()
    z = raw.shape[0]
    n_hits = 0
    for zi in range(z):
        if n_hits >= max_slices:
            break
        if rng.random() < prob_per_slice:
            n_hits += 1
            if rng.random() < 0.5:
                raw[zi] = 0  # full missing section
            else:
                raw[zi] = (raw[zi].astype(np.float32) * 0.15).astype(raw.dtype)  # degraded/low-contrast section
    return raw


def random_misalignment(
    raw: np.ndarray, rng: np.random.Generator, prob_per_slice: float = 0.05, max_shift: int = 12
) -> np.ndarray:
    """Simulates section-to-section registration error: shifts a handful of individual RAW
    z-slices in x/y by a small offset. Label is left untouched -- the real 3D structure didn't
    move, only that section's imaging is offset, which is what an actual misaligned section
    looks like, and exactly the kind of local mismatch between evidence and truth the network
    needs to be robust to."""
    raw = raw.copy()
    z = raw.shape[0]
    for zi in range(z):
        if rng.random() < prob_per_slice:
            dy = int(rng.integers(-max_shift, max_shift + 1))
            dx = int(rng.integers(-max_shift, max_shift + 1))
            if dy or dx:
                raw[zi] = np.roll(raw[zi], shift=(dy, dx), axis=(0, 1))
    return raw

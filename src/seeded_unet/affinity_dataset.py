"""Dense, whole-patch training patches for the affinity+LSD model -- the
non-seeded counterpart to dataset.py's SeededPatchDataset. Patches are random
crops from a stack (not centered on one chosen instance's seed), and the
target describes every instance visible in the patch at once via
affinities + per-voxel LSD (affinity_targets.py), instead of one instance's
binary mask."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .affinity_targets import DEFAULT_OFFSETS, compute_affinities, compute_lsd_target_dense
from .augmentations import random_artifact, random_flip, random_misalignment, random_rotate90, random_translate_crop
from .dataset import Instance, _crop_with_padding
from .seeds import physical_sigma_to_voxels
from .stack_io import Stack


def stacks_from_instances(instances: list[Instance]) -> list[Stack]:
    """Recovers the unique stacks referenced by a (train/val/test) instance
    split, so this dataset can reuse dataset.py's scene-group-aware split
    instead of re-deriving its own -- dense patches are drawn from whole
    stacks, not per-instance, but should still respect the same train/val
    stack boundaries to avoid the same leakage split_instances already
    guards against."""
    seen: dict[str, Stack] = {}
    for inst in instances:
        seen[inst.stack.name] = inst.stack
    return list(seen.values())


class DenseAffinityPatchDataset(Dataset):
    def __init__(
        self,
        stacks: list[Stack],
        patch_shape_zyx: tuple[int, int, int] = (32, 256, 256),
        samples_per_stack: int = 32,
        offsets=DEFAULT_OFFSETS,
        min_labeled_fraction: float = 0.02,
        max_resample_attempts: int = 8,
        rng_seed: int | None = None,
        predict_lsd: bool = True,
        lsd_sigma_nm: float = 60.0,
        augment: bool = True,
        translate_pad_yx: int = 16,
    ):
        self.stacks = stacks
        self.patch_shape_zyx = patch_shape_zyx
        self.samples_per_stack = samples_per_stack
        self.offsets = offsets
        self.min_labeled_fraction = min_labeled_fraction
        self.max_resample_attempts = max_resample_attempts
        self.rng = np.random.default_rng(rng_seed)
        self.predict_lsd = predict_lsd
        self.lsd_sigma_nm = lsd_sigma_nm
        self.augment = augment
        self.translate_pad_yx = translate_pad_yx
        self._bbox_cache: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {}

    def __len__(self) -> int:
        return len(self.stacks) * self.samples_per_stack

    def _labeled_bbox(self, stack: Stack) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        cached = self._bbox_cache.get(stack.name)
        if cached is not None:
            return cached
        zz, yy, xx = np.where(stack.labels != 0)
        lo = (int(zz.min()), int(yy.min()), int(xx.min()))
        hi = (int(zz.max()), int(yy.max()), int(xx.max()))
        self._bbox_cache[stack.name] = (lo, hi)
        return lo, hi

    def _random_center(self, stack: Stack) -> tuple[int, int, int]:
        lo, hi = self._labeled_bbox(stack)
        return tuple(
            int(self.rng.integers(lo[i], hi[i] + 1)) if hi[i] > lo[i] else lo[i] for i in range(3)
        )

    def __getitem__(self, idx: int):
        stack = self.stacks[idx // self.samples_per_stack]
        pad = self.translate_pad_yx if self.augment else 0
        pz, py, px = self.patch_shape_zyx
        padded_shape = (pz, py + 2 * pad, px + 2 * pad)

        label_padded = None
        for attempt in range(self.max_resample_attempts):
            center = self._random_center(stack)
            candidate = _crop_with_padding(stack.labels, center, padded_shape)
            if (candidate != 0).mean() >= self.min_labeled_fraction or attempt == self.max_resample_attempts - 1:
                label_padded = candidate
                break
        raw_padded = _crop_with_padding(stack.raw, center, padded_shape)

        if self.augment:
            # Artifacts/misalignment operate on raw only (label is the "true" 3D structure,
            # unaffected by an imaging defect in one section -- see augmentations.py), and
            # must happen before the translate sub-crop so a hit slice isn't ever guaranteed
            # to land exactly at the crop's own edge.
            raw_padded = random_artifact(raw_padded, self.rng)
            raw_padded = random_misalignment(raw_padded, self.rng)
            raw_patch, label_patch = random_translate_crop(raw_padded, label_padded, self.patch_shape_zyx, self.rng)
            raw_patch, label_patch = random_flip(raw_patch, label_patch, self.rng)
            raw_patch, label_patch = random_rotate90(raw_patch, label_patch, self.rng)
        else:
            raw_patch, label_patch = raw_padded, label_padded

        raw_patch = raw_patch.astype(np.float32) / 255.0

        affinities = compute_affinities(label_patch, self.offsets)
        input_tensor = torch.from_numpy(raw_patch[None])
        target_tensor = torch.from_numpy(affinities)
        if not self.predict_lsd:
            return input_tensor, target_tensor

        lsd_sigma_voxel = physical_sigma_to_voxels(self.lsd_sigma_nm, stack.scale_nm)
        lsd_target = compute_lsd_target_dense(label_patch, lsd_sigma_voxel)
        lsd_target_tensor = torch.from_numpy(lsd_target)
        return input_tensor, target_tensor, lsd_target_tensor

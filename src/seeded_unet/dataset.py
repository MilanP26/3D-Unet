"""Builds per-neuron training instances from loaded Stacks and serves random
seed-centered 3D patches as a torch Dataset.

Ground truth for a patch is just the intersection of the full instance mask
with the patch window -- instances are not required to fit entirely inside a
patch. Real neurons/neurites in this data can span almost the entire crop
(see PLAN.md section 4 update), so a patch only ever has to show local
context around the seed, not the whole cell.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .lsd import compute_lsd_target
from .seeds import gaussian_heatmap, physical_sigma_to_voxels, sample_from_distribution, seed_distribution
from .stack_io import DEFAULT_CACHE_DIR, Stack


@dataclass
class Instance:
    stack: Stack
    instance_id: int
    voxel_count: int

    @property
    def scene_group(self) -> str:
        return self.stack.scene_group

    @property
    def key(self) -> str:
        return f"{self.stack.name}#{self.instance_id}"


def build_instances(stacks: list[Stack], min_voxels: int = 500) -> list[Instance]:
    instances = []
    for stack in stacks:
        ids, counts = np.unique(stack.labels, return_counts=True)
        for sid, count in zip(ids, counts):
            if sid == 0 or count < min_voxels:
                continue
            instances.append(Instance(stack=stack, instance_id=int(sid), voxel_count=int(count)))
    return instances


def split_instances(
    instances: list[Instance],
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 0,
) -> tuple[list[Instance], list[Instance], list[Instance]]:
    """Splits by scene_group whenever there are enough distinct groups to make
    that meaningful; otherwise falls back to an instance-level split and warns
    loudly, since that is only a weak, leakage-prone proxy (see PLAN.md
    section 8)."""
    rng = np.random.default_rng(seed)
    groups = sorted({inst.scene_group for inst in instances})

    if len(groups) >= 3:
        rng.shuffle(groups)
        n_val = max(1, round(len(groups) * val_fraction))
        n_test = round(len(groups) * test_fraction)
        test_groups = set(groups[:n_test])
        val_groups = set(groups[n_test:n_test + n_val])
        train_groups = set(groups[n_test + n_val:])
        train = [i for i in instances if i.scene_group in train_groups]
        val = [i for i in instances if i.scene_group in val_groups]
        test = [i for i in instances if i.scene_group in test_groups]
        return train, val, test

    warnings.warn(
        f"Only {len(groups)} distinct scene group(s) ({groups}) -- a real held-out "
        "generalization split isn't possible yet. Falling back to an INSTANCE-level "
        "split, which is a weak interim estimate: it checks whether the model can "
        "separate different neurons within the same imaged region, not whether it "
        "generalizes to new tissue. Add more independently-imaged stacks to fix this "
        "(see PLAN.md open questions).",
        stacklevel=2,
    )
    idx = rng.permutation(len(instances))
    n_val = max(1, round(len(instances) * val_fraction))
    n_test = round(len(instances) * test_fraction)
    test = [instances[i] for i in idx[:n_test]]
    val = [instances[i] for i in idx[n_test:n_test + n_val]]
    train = [instances[i] for i in idx[n_test + n_val:]]
    return train, val, test


def _crop_with_padding(
    volume: np.ndarray, center_zyx: tuple[int, int, int], patch_shape_zyx: tuple[int, int, int]
) -> np.ndarray:
    starts = [c - p // 2 for c, p in zip(center_zyx, patch_shape_zyx)]
    ends = [s + p for s, p in zip(starts, patch_shape_zyx)]

    pad_before = [max(0, -s) for s in starts]
    pad_after = [max(0, e - vs) for e, vs in zip(ends, volume.shape)]
    clipped_starts = [max(0, s) for s in starts]
    clipped_ends = [min(vs, e) for vs, e in zip(volume.shape, ends)]

    cropped = volume[
        clipped_starts[0]:clipped_ends[0],
        clipped_starts[1]:clipped_ends[1],
        clipped_starts[2]:clipped_ends[2],
    ]
    if any(pad_before) or any(pad_after):
        cropped = np.pad(cropped, list(zip(pad_before, pad_after)), mode="constant", constant_values=0)
    return cropped


class SeededPatchDataset(Dataset):
    def __init__(
        self,
        instances: list[Instance],
        patch_shape_zyx: tuple[int, int, int] = (32, 256, 256),
        seed_sigma_nm: float = 150.0,
        samples_per_instance: int = 8,
        interior_bias: float = 2.0,
        # No z jitter: some neurons here move around a lot slice-to-slice, so
        # jittering the seed in z risked landing it well off the actual traced
        # centerline rather than just nudging it (unlike x/y, where a modest jitter
        # is still a realistic stand-in for an imprecise click). x/y jitter was
        # also tightened from 16 down to 8 -- some instances are quite small, and a
        # 16-voxel jitter could push the seed close to or outside a small one
        # entirely (see PLAN.md 2026-07-13).
        jitter_voxels_zyx: tuple[int, int, int] = (0, 8, 8),
        rng_seed: int | None = None,
        seed_dist_cache_dir: Path | None = DEFAULT_CACHE_DIR / "seed_dist",
        predict_lsd: bool = True,
        # Deliberately much smaller than seed_sigma_nm (150nm, which represents click
        # uncertainty and is fine to be broad) -- this needs to be a genuinely LOCAL
        # neighborhood. At 150nm it converts to a 75-voxel sigma in x/y given this
        # data's 2nm-per-voxel in-plane scale -- barely "local" at all (a third of a
        # 256-voxel patch width) and expensive to compute (gaussian_filter cost scales
        # with kernel radius, ~4*sigma by default). 60nm -> 30 voxels in x/y, 2 in z.
        lsd_sigma_nm: float = 60.0,
    ):
        self.instances = instances
        self.patch_shape_zyx = patch_shape_zyx
        self.seed_sigma_nm = seed_sigma_nm
        self.samples_per_instance = samples_per_instance
        self.interior_bias = interior_bias
        self.jitter_voxels_zyx = jitter_voxels_zyx
        self.rng = np.random.default_rng(rng_seed)
        self._seed_dist_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.seed_dist_cache_dir = seed_dist_cache_dir
        self.predict_lsd = predict_lsd
        self.lsd_sigma_nm = lsd_sigma_nm

    def __len__(self) -> int:
        return len(self.instances) * self.samples_per_instance

    def _seed_dist_disk_path(self, inst: Instance) -> Path | None:
        if self.seed_dist_cache_dir is None:
            return None
        return self.seed_dist_cache_dir / f"{inst.key.replace('#', '__')}__bias{self.interior_bias}.npz"

    def _cached_seed_distribution(self, inst: Instance) -> tuple[np.ndarray, np.ndarray]:
        cached = self._seed_dist_cache.get(inst.key)
        if cached is not None:
            return cached

        disk_path = self._seed_dist_disk_path(inst)
        if disk_path is not None and disk_path.exists():
            with np.load(disk_path) as npz:
                cached = (npz["coords"], npz["weights"])
        else:
            mask = inst.stack.instance_mask(inst.instance_id)
            cached = seed_distribution(mask, self.interior_bias)
            if disk_path is not None:
                disk_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(disk_path, coords=cached[0], weights=cached[1])

        self._seed_dist_cache[inst.key] = cached
        return cached

    def precompute_seed_distributions(self, show_progress: bool = True) -> None:
        """Runs the one-time (distance transform per instance) setup cost up
        front, with a progress bar, instead of paying it silently/lazily
        during the first epoch. Safe to call more than once (already-cached
        instances are skipped)."""
        unique_instances = {inst.key: inst for inst in self.instances}.values()
        iterable = unique_instances
        if show_progress:
            from tqdm import tqdm

            iterable = tqdm(unique_instances, desc="Precomputing seed distributions", unit="instance")
        for inst in iterable:
            self._cached_seed_distribution(inst)

    def __getitem__(self, idx: int):
        inst = self.instances[idx // self.samples_per_instance]
        stack = inst.stack

        coords, weights = self._cached_seed_distribution(inst)
        seed_z, seed_y, seed_x = sample_from_distribution(coords, weights, self.rng)
        jz, jy, jx = self.jitter_voxels_zyx
        center = (
            int(np.clip(seed_z + self.rng.integers(-jz, jz + 1), 0, stack.raw.shape[0] - 1)),
            int(np.clip(seed_y + self.rng.integers(-jy, jy + 1), 0, stack.raw.shape[1] - 1)),
            int(np.clip(seed_x + self.rng.integers(-jx, jx + 1), 0, stack.raw.shape[2] - 1)),
        )

        raw_patch = _crop_with_padding(stack.raw, center, self.patch_shape_zyx).astype(np.float32) / 255.0
        # Compare only within the small cropped patch, not the whole (Z,Y,X) label volume.
        label_patch = _crop_with_padding(stack.labels, center, self.patch_shape_zyx)
        mask_patch = (label_patch == inst.instance_id).astype(np.float32)

        seed_in_patch = tuple(
            s - (c - p // 2) for s, c, p in zip((seed_z, seed_y, seed_x), center, self.patch_shape_zyx)
        )
        sigma_zyx = physical_sigma_to_voxels(self.seed_sigma_nm, stack.scale_nm)
        heatmap = gaussian_heatmap(self.patch_shape_zyx, seed_in_patch, sigma_zyx)

        input_tensor = torch.from_numpy(np.stack([raw_patch, heatmap], axis=0))
        target_tensor = torch.from_numpy(mask_patch[None])
        if not self.predict_lsd:
            return input_tensor, target_tensor

        lsd_sigma_voxel = physical_sigma_to_voxels(self.lsd_sigma_nm, stack.scale_nm)
        lsd_target = compute_lsd_target(mask_patch, lsd_sigma_voxel)
        lsd_target_tensor = torch.from_numpy(lsd_target)
        return input_tensor, target_tensor, lsd_target_tensor

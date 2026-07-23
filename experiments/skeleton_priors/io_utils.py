"""Load a phase-B affinity prediction file + the VAST skeletons into shapes the
consistency oracle can use, without importing torch or the model.

The prediction file is what `seeded_unet.affinity_phase_b_infer` writes:
`outputs/phase_b_affinity/tree_<id>/predictions.npz`, containing, for the one
target tree, a per-seed-patch *binary* mask (the target tree's voxels only) plus
where each patch sits in the full stack's mip0 frame. Fields (verified against a
real file, tree 1: 176 patches of 32x256x256):

  tree_id           () int         the target tree
  patch_shape_zyx   (3,) int       (pz, py, px), e.g. (32, 256, 256)
  seed_local_ids    (N,) int       which node of the tree each patch was centered on
  seed_xyz          (N, 3) int     that node's absolute (x, y, z) in mip0 voxels
  patch_origin_zyx  (N, 3) int     each patch's absolute (z, y, x) origin corner
  packed_masks      (N, ceil(pz*py*px/8)) uint8   np.packbits of the target mask
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Reuse the production skeleton parser rather than re-implementing it.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from seeded_unet.vast_skeleton import DEFAULT_SKELETON_CSV, load_skeletons  # noqa: E402


@dataclass
class TreePredictions:
    """A loaded predictions.npz plus lazy per-patch mask unpacking."""

    tree_id: int
    patch_shape_zyx: tuple[int, int, int]
    seed_local_ids: np.ndarray  # (N,)
    seed_xyz: np.ndarray  # (N, 3) absolute (x, y, z)
    patch_origin_zyx: np.ndarray  # (N, 3) absolute (z, y, x)
    _packed_masks: np.ndarray  # (N, packed_bytes)

    @property
    def num_patches(self) -> int:
        return len(self.seed_local_ids)

    def mask(self, i: int) -> np.ndarray:
        """Unpack patch i's target-tree binary mask to (pz, py, px) bool."""
        n_vox = int(np.prod(self.patch_shape_zyx))
        flat = np.unpackbits(self._packed_masks[i])[:n_vox].astype(bool)
        return flat.reshape(self.patch_shape_zyx)


def load_predictions(npz_path: Path) -> TreePredictions:
    d = np.load(npz_path)
    return TreePredictions(
        tree_id=int(d["tree_id"]),
        patch_shape_zyx=tuple(int(v) for v in d["patch_shape_zyx"]),
        seed_local_ids=d["seed_local_ids"],
        seed_xyz=d["seed_xyz"],
        patch_origin_zyx=d["patch_origin_zyx"],
        _packed_masks=d["packed_masks"],
    )


@dataclass
class NodeCloud:
    """Every skeleton node from every tree as flat arrays, for fast bounding-box
    membership tests (same idea as affinity_phase_b_infer's `in_patch` filter)."""

    tree_id: np.ndarray  # (M,)
    x: np.ndarray  # (M,)
    y: np.ndarray  # (M,)
    z: np.ndarray  # (M,)

    @classmethod
    def from_skeletons(cls, skeletons: dict[int, list]) -> "NodeCloud":
        tids, xs, ys, zs = [], [], [], []
        for tid, nodes in skeletons.items():
            for n in nodes:
                tids.append(tid)
                xs.append(n.x)
                ys.append(n.y)
                zs.append(n.z)
        return cls(np.array(tids), np.array(xs), np.array(ys), np.array(zs))

    def in_box(self, origin_zyx, shape_zyx):
        """Boolean mask over all nodes that fall inside the given patch box."""
        oz, oy, ox = origin_zyx
        pz, py, px = shape_zyx
        return (
            (self.z >= oz) & (self.z < oz + pz)
            & (self.y >= oy) & (self.y < oy + py)
            & (self.x >= ox) & (self.x < ox + px)
        )


def load_all(npz_path: Path, skeleton_csv: Path = DEFAULT_SKELETON_CSV):
    preds = load_predictions(npz_path)
    skeletons = load_skeletons(skeleton_csv)
    cloud = NodeCloud.from_skeletons(skeletons)
    return preds, cloud

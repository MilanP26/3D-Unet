"""#1 -- Skeleton-consistency error oracle.

The idea: we hold complete manual skeletons (hundreds of nodes per neuron), so a
segmentation can be *automatically* checked against them with zero extra
annotation. Every place the segmentation disagrees with the manual traces is a
self-flagged likely error, and we can rank them so a human proofreads the worst
first instead of scrolling a 176-page PDF hoping to spot mistakes. This turns
the objective from "maximize dice" into "minimize human-minutes-per-neuron",
which is the metric connectomics actually cares about.

Runs purely on a saved predictions.npz + the skeleton CSV -- no model, no GPU,
no retrain. It works with what that file stores: the *target tree's* binary mask
per patch. From that alone (plus every tree's nodes) we can catch four error
classes per patch:

  MERGE_LEAK   a *different* tree's traced node lands inside the target mask
               -> the target neuron's mask has swallowed part of another neuron.
               (This is CLAUDE.md section 9's leakage metric, but measured on real
               Phase-B skeletons instead of Phase-A ground truth.)
  ORPHAN_BLOB  a large connected component of the mask contains none of the
               target tree's own nodes -> mask grew into unrelated tissue.
  SPLIT        the target tree's own nodes in this patch fall in >1 connected
               component -> the neuron is broken into pieces here.
  SEED_UNCOVERED / DEGENERATE
               the patch's own seed node isn't inside its mask, or the mask is
               ~empty / ~full -> the watershed didn't grow a sane region here.

Not yet covered (documented TODO): cross-patch stitching disagreement in the
overlap between neighboring patches. It needs the full stitched volume; the
per-patch checks above are the cheap, high-value first pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from .io_utils import NodeCloud, TreePredictions, load_all

# 3D 6-connectivity for connected components (matches the short-range affinity
# neighborhood; a 26-connectivity structure would bridge diagonal touches that
# the affinity model treats as separable).
_CONNECTIVITY = ndimage.generate_binary_structure(3, 1)

# Severity weights -- a merge leak corrupts identity and is the worst; a split is
# recoverable by a human with one merge click; degenerate/uncovered patches are
# usually just uninformative rather than wrong.
SEVERITY = {
    "MERGE_LEAK": 100.0,
    "ORPHAN_BLOB": 40.0,
    "SPLIT": 20.0,
    "SEED_UNCOVERED": 10.0,
    "DEGENERATE": 5.0,
}


@dataclass
class Finding:
    patch_index: int
    seed_local_id: int
    seed_xyz: tuple[int, int, int]
    kind: str
    severity: float
    detail: str


@dataclass
class TreeReport:
    tree_id: int
    num_patches: int
    findings: list[Finding] = field(default_factory=list)
    fg_fractions: list[float] = field(default_factory=list)

    def ranked(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: -f.severity)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.kind] = out.get(f.kind, 0) + 1
        return out


def _to_local(node_xyz_abs, origin_zyx, shape_zyx):
    """Absolute (x,y,z) node -> patch-local (z,y,x) int, or None if outside."""
    x, y, z = node_xyz_abs
    oz, oy, ox = origin_zyx
    lz, ly, lx = z - oz, y - oy, x - ox
    pz, py, px = shape_zyx
    if 0 <= lz < pz and 0 <= ly < py and 0 <= lx < px:
        return lz, ly, lx
    return None


def evaluate_mask(
    mask: np.ndarray,
    origin_zyx,
    shape_zyx,
    tree_id: int,
    cloud: NodeCloud,
    seed_xyz=None,
    min_orphan_voxels: int = 200,
    degenerate_lo: float = 0.005,
    degenerate_hi: float = 0.98,
) -> list[tuple[str, str]]:
    """Pure per-patch consistency checks over one binary mask, independent of how
    the mask was produced -- so it scores both the production watershed and the
    experimental agglomeration on the same footing. Returns (kind, detail) pairs."""
    out: list[tuple[str, str]] = []
    fg = float(mask.mean())
    if fg <= degenerate_lo or fg >= degenerate_hi:
        out.append(("DEGENERATE", f"foreground fraction {fg:.3%} (empty or filled patch)"))
        return out  # a degenerate mask makes the other checks meaningless

    cc, n_cc = ndimage.label(mask, structure=_CONNECTIVITY)
    in_box = cloud.in_box(origin_zyx, shape_zyx)
    box_tids = cloud.tree_id[in_box]
    box_x, box_y, box_z = cloud.x[in_box], cloud.y[in_box], cloud.z[in_box]

    target_cc_labels: set[int] = set()
    for tid, x, y, z in zip(box_tids, box_x, box_y, box_z):
        loc = _to_local((x, y, z), origin_zyx, shape_zyx)
        if loc is None:
            continue
        lz, ly, lx = loc
        if not mask[lz, ly, lx]:
            continue
        if int(tid) == tree_id:
            target_cc_labels.add(int(cc[lz, ly, lx]))
        else:
            out.append((
                "MERGE_LEAK",
                f"tree {int(tid)}'s node at (x={x},y={y},z={z}) lies inside tree "
                f"{tree_id}'s predicted mask",
            ))

    if seed_xyz is not None:
        seed_loc = _to_local(seed_xyz, origin_zyx, shape_zyx)
        if seed_loc is not None and not mask[seed_loc]:
            out.append(("SEED_UNCOVERED", "the patch's own seed node is not inside its predicted mask"))

    if len(target_cc_labels) > 1:
        out.append((
            "SPLIT",
            f"tree {tree_id}'s nodes span {len(target_cc_labels)} connected components in this patch",
        ))

    if n_cc >= 1:
        sizes = ndimage.sum(np.ones_like(cc), cc, index=np.arange(1, n_cc + 1))
        for lbl, size in enumerate(sizes, start=1):
            if size >= min_orphan_voxels and lbl not in target_cc_labels:
                out.append((
                    "ORPHAN_BLOB",
                    f"component #{lbl} ({int(size)} vox) contains no traced node of tree {tree_id}",
                ))
    return out


def skeleton_clean(mask, origin_zyx, shape_zyx, tree_id: int, cloud: NodeCloud):
    """Drop connected components of `mask` that contain no traced node of tree_id
    -- the oracle's ORPHAN_BLOB detection turned into an automatic fix. Provably
    preserves node coverage (any node inside the input mask stays inside a kept
    component) while removing un-skeleton-supported leak blobs. Returns a bool mask.

    Cleans the *production* seeded-watershed output rather than trying to
    re-partition it -- validated on real tree-1 EM (2026-07-22): cut orphan/leak
    voxels to zero while keeping 100% node coverage."""
    cc, n_cc = ndimage.label(mask, structure=_CONNECTIVITY)
    keep: set[int] = set()
    in_box = cloud.in_box(origin_zyx, shape_zyx)
    for tid, x, y, z in zip(cloud.tree_id[in_box], cloud.x[in_box], cloud.y[in_box], cloud.z[in_box]):
        if int(tid) != tree_id:
            continue
        loc = _to_local((x, y, z), origin_zyx, shape_zyx)
        if loc is not None and mask[loc]:
            keep.add(int(cc[loc]))
    return np.isin(cc, list(keep)) if keep else np.zeros_like(mask)


def analyze_patch(
    i: int,
    preds: TreePredictions,
    cloud: NodeCloud,
    **kwargs,
) -> list[Finding]:
    seed_xyz = tuple(int(v) for v in preds.seed_xyz[i])
    seed_local_id = int(preds.seed_local_ids[i])
    raw = evaluate_mask(
        preds.mask(i),
        tuple(int(v) for v in preds.patch_origin_zyx[i]),
        preds.patch_shape_zyx,
        preds.tree_id,
        cloud,
        seed_xyz=seed_xyz,
        **kwargs,
    )
    return [Finding(i, seed_local_id, seed_xyz, kind, SEVERITY[kind], detail) for kind, detail in raw]


def analyze_tree(
    npz_path: Path,
    skeleton_csv: Path | None = None,
    **patch_kwargs,
) -> TreeReport:
    if skeleton_csv is None:
        preds, cloud = load_all(npz_path)
    else:
        preds, cloud = load_all(npz_path, skeleton_csv)
    report = TreeReport(tree_id=preds.tree_id, num_patches=preds.num_patches)
    for i in range(preds.num_patches):
        report.fg_fractions.append(float(preds.mask(i).mean()))
        report.findings.extend(analyze_patch(i, preds, cloud, **patch_kwargs))
    return report

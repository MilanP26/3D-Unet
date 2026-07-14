"""Turns a (raw patch, predicted mask) pair into a human-viewable PNG.

The model's actual output is a numpy boolean array -- not an image. This is
what produces something you can hand someone and say "here's what the model
predicted."
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _pick_slice_idxs(z_total: int, n_slices: int, node_zs: list[int]) -> list[int]:
    """Prefers slices that actually contain a real traced node over blind even
    spacing -- with even spacing, a patch can easily show 6 slices where none
    happen to contain a node, which looks like (but isn't necessarily) the mask
    drifting away from the real trace. If there are more node-slices than
    n_slices, spreads the selection evenly across the available ones instead of
    always showing the first few."""
    if not node_zs:
        return np.linspace(0, z_total - 1, n_slices).round().astype(int).tolist()
    node_zs = sorted(set(node_zs))
    if len(node_zs) >= n_slices:
        pick = np.linspace(0, len(node_zs) - 1, n_slices).round().astype(int)
        return sorted(node_zs[i] for i in pick)
    remaining = n_slices - len(node_zs)
    candidates = [z for z in range(z_total) if z not in set(node_zs)]
    fill_idxs = np.linspace(0, len(candidates) - 1, remaining).round().astype(int)
    fill = [candidates[i] for i in fill_idxs]
    return sorted(node_zs + fill)


def save_overlay_montage(
    raw_patch: np.ndarray,
    mask_patch: np.ndarray,
    out_path: Path,
    n_slices: int = 6,
    seed_zyx: tuple[int, int, int] | None = None,
    node_markers_zyx: list[tuple[int, int, int]] | None = None,
    prefer_node_slices: bool = True,
    mask_color: tuple[float, float, float] = (1.0, 0.15, 0.15),
    mask_alpha: float = 0.45,
) -> None:
    """raw_patch: (Z,Y,X), any numeric range (auto-normalized). mask_patch:
    (Z,Y,X) bool/0-1, same shape. Saves one PNG with `n_slices` z-slices, each
    shown as grayscale EM with the predicted mask overlaid as a translucent
    color, so a single image conveys the 3D result.

    `node_markers_zyx`: real traced skeleton nodes (patch-local z,y,x) to mark
    with a purple circle wherever one lands on a displayed slice -- lets you
    check the prediction against the actual annotator trace, not just the
    single seed (which is marked separately, with a yellow x, only on its own
    slice). When `prefer_node_slices` is True (the default) and any node
    markers are given, the displayed slices are chosen to include real node
    slices rather than blind even spacing across the patch depth -- with even
    spacing, it's easy to land on 6 slices where none happen to contain a
    node, which can look like (without actually being) the mask drifting off
    the real trace."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = raw_patch.astype(np.float32)
    if raw.max() > 1.5:
        raw = raw / 255.0
    mask = mask_patch.astype(bool)

    z_total = raw.shape[0]
    n_slices = min(n_slices, z_total)
    node_zs = [nz for nz, ny, nx in node_markers_zyx] if node_markers_zyx else []
    if prefer_node_slices and node_zs:
        slice_idxs = _pick_slice_idxs(z_total, n_slices, node_zs)
    else:
        slice_idxs = np.linspace(0, z_total - 1, n_slices).round().astype(int)

    fig, axes = plt.subplots(1, n_slices, figsize=(3 * n_slices, 3.2))
    axes = np.atleast_1d(axes)

    for ax, z in zip(axes, slice_idxs):
        gray = raw[z]
        rgb = np.stack([gray, gray, gray], axis=-1)
        m = mask[z]
        for c in range(3):
            rgb[..., c] = np.where(m, (1 - mask_alpha) * rgb[..., c] + mask_alpha * mask_color[c], rgb[..., c])
        ax.imshow(np.clip(rgb, 0, 1))
        title = f"z={z}"
        if seed_zyx is not None and z == seed_zyx[0]:
            ax.scatter([seed_zyx[2]], [seed_zyx[1]], marker="x", c="yellow", s=70, linewidths=2)
            title += " (seed slice)"
        if node_markers_zyx:
            xs = [nx for nz, ny, nx in node_markers_zyx if nz == z]
            ys = [ny for nz, ny, nx in node_markers_zyx if nz == z]
            if xs:
                ax.scatter(
                    xs, ys, marker="o", facecolors="none", edgecolors="mediumorchid",
                    s=110, linewidths=2,
                )
                title += f" ({len(xs)} node{'s' if len(xs) > 1 else ''})"
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(f"Predicted mask overlay -- {int(mask.sum())} foreground voxels")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

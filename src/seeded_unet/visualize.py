"""Turns a (raw patch, predicted mask) pair into a human-viewable PNG.

The model's actual output is a numpy boolean array -- not an image. This is
what produces something you can hand someone and say "here's what the model
predicted."
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_overlay_montage(
    raw_patch: np.ndarray,
    mask_patch: np.ndarray,
    out_path: Path,
    n_slices: int = 6,
    seed_zyx: tuple[int, int, int] | None = None,
    mask_color: tuple[float, float, float] = (1.0, 0.15, 0.15),
    mask_alpha: float = 0.45,
) -> None:
    """raw_patch: (Z,Y,X), any numeric range (auto-normalized). mask_patch:
    (Z,Y,X) bool/0-1, same shape. Saves one PNG with `n_slices` evenly-spaced
    z-slices, each shown as grayscale EM with the predicted mask overlaid as
    a translucent color, so a single image conveys the 3D result."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = raw_patch.astype(np.float32)
    if raw.max() > 1.5:
        raw = raw / 255.0
    mask = mask_patch.astype(bool)

    z_total = raw.shape[0]
    n_slices = min(n_slices, z_total)
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
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(f"Predicted mask overlay -- {int(mask.sum())} foreground voxels")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

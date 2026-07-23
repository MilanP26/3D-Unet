#!/usr/bin/env python
"""Render a local, openable HTML report of the skeleton-consistency oracle (#1)
on the existing pipeline's tree_1 output.

We only have the predicted masks (the raw EM lives on the detached hard drive),
so slices show the predicted mask coloured by connected component: GREEN = a
component that contains one of tree 1's own traced nodes (legit), RED = an
"orphan" component the oracle flagged (no traced node of tree 1 inside it ->
likely a leak/error). Cyan dots are tree 1's traced nodes in that slice; magenta
dots are other trees' nodes; the white star is the patch's own seed node.

Writes experiments/outputs/report/index.html + PNGs. Open index.html in a browser.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from skeleton_priors.io_utils import load_all  # noqa: E402
from skeleton_priors.consistency import analyze_tree, _CONNECTIVITY  # noqa: E402


def render_real_patch(preds, cloud, i, out_png):
    mask = preds.mask(i)
    origin = tuple(int(v) for v in preds.patch_origin_zyx[i])
    pz, py, px = preds.patch_shape_zyx

    cc, n_cc = ndimage.label(mask, structure=_CONNECTIVITY)

    in_box = cloud.in_box(origin, preds.patch_shape_zyx)
    tids = cloud.tree_id[in_box]
    xs, ys, zs = cloud.x[in_box], cloud.y[in_box], cloud.z[in_box]
    target_cc = set()
    nodes_local = []  # (lz, ly, lx, is_target)
    for tid, x, y, z in zip(tids, xs, ys, zs):
        lz, ly, lx = z - origin[0], y - origin[1], x - origin[2]
        if 0 <= lz < pz and 0 <= ly < py and 0 <= lx < px:
            is_t = int(tid) == preds.tree_id
            nodes_local.append((lz, ly, lx, is_t))
            if is_t and mask[lz, ly, lx]:
                target_cc.add(int(cc[lz, ly, lx]))

    sizes = ndimage.sum(np.ones_like(cc), cc, index=np.arange(1, n_cc + 1)) if n_cc else np.array([])
    orphan_labels = [l for l in range(1, n_cc + 1) if l not in target_cc and sizes[l - 1] >= 200]

    sx, sy, sz = (int(v) for v in preds.seed_xyz[i])
    z_seed = int(np.clip(sz - origin[0], 0, pz - 1))
    if orphan_labels:
        per_z = np.isin(cc, orphan_labels).sum(axis=(1, 2))
        z_orphan = int(per_z.argmax())
    else:
        z_orphan = z_seed

    def draw(ax, z, label):
        ccz = cc[z]
        rgb = np.zeros((py, px, 3), dtype=np.uint8)
        rgb[mask[z]] = (90, 90, 90)
        for lbl in range(1, n_cc + 1):
            sel = ccz == lbl
            if not sel.any():
                continue
            if lbl in target_cc:
                rgb[sel] = (40, 170, 70)       # legit: contains a tree-1 node
            elif sizes[lbl - 1] >= 200:
                rgb[sel] = (200, 55, 45)        # flagged orphan blob
        ax.imshow(rgb, interpolation="nearest")
        for (lz, ly, lx, is_t) in nodes_local:
            if abs(lz - z) <= 1:
                ax.plot(lx, ly, "o", ms=5, mec="k", mfc=("cyan" if is_t else "magenta"))
        if z == z_seed:
            ax.plot(sx - origin[2], sy - origin[1], "*", ms=13, mec="k", mfc="white")
        ax.set_title(f"{label} (z={z})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    draw(axes[0], z_seed, "at seed (*)  -- context/quality")
    draw(axes[1], z_orphan, "worst orphan slice  -- flagged")
    fig.suptitle(f"patch {i}   green=legit  red=orphan/flagged  cyan=tree1 nodes  magenta=other trees", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return {"z_seed": z_seed, "z_orphan": z_orphan, "n_components": n_cc,
            "n_target_components": len(target_cc), "n_orphan": len(orphan_labels)}


def main():
    out_dir = REPO_ROOT / "experiments" / "outputs" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = REPO_ROOT / "outputs" / "phase_b_affinity" / "tree_1" / "predictions.npz"
    real_cards = []
    summary = f"(no real predictions found at {pred_path})"
    if pred_path.exists():
        preds, cloud = load_all(pred_path)
        report = analyze_tree(pred_path)
        seen, patches = set(), []
        for f in report.ranked():
            if f.patch_index not in seen:
                seen.add(f.patch_index)
                patches.append(f.patch_index)
            if len(patches) >= 6:
                break
        for i in patches:
            png = out_dir / f"real_patch_{i}.png"
            info = render_real_patch(preds, cloud, i, png)
            real_cards.append((png.name, i, info))
        counts = report.counts()
        summary = (f"tree {report.tree_id}: {report.num_patches} patches, "
                   f"{counts.get('MERGE_LEAK', 0)} merge-leaks, "
                   f"{counts.get('ORPHAN_BLOB', 0)} orphan blobs, "
                   f"{counts.get('SPLIT', 0)} splits")

    html = ["<!doctype html><meta charset=utf-8><title>consistency oracle report</title>",
            "<style>body{font-family:system-ui,Arial;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#111}"
            "img{max-width:100%;border:1px solid #ccc;border-radius:6px}"
            ".card{margin:1.2rem 0}code{background:#f2f2f2;padding:.1rem .3rem;border-radius:3px}</style>",
            "<h1>Skeleton-consistency oracle (#1) on the existing pipeline (tree 1)</h1>",
            f"<p>{summary}. Only predicted masks are available (raw EM is on the detached drive), so "
            "slices are coloured by connected component: <b style='color:#28aa46'>green = legit</b> "
            "(contains a traced node of tree 1), <b style='color:#c8372d'>red = flagged orphan</b> "
            "(no traced node inside → likely leak/error). Cyan dots = tree 1's nodes, magenta = other trees'.</p>"]
    for name, i, info in real_cards:
        html.append(f"<div class=card><img src='{name}'><div>patch {i}: {info['n_components']} components, "
                    f"{info['n_target_components']} contain a tree-1 node, "
                    f"<b>{info['n_orphan']} flagged orphan(s)</b></div></div>")
    if not real_cards:
        html.append("<p><i>No real prediction file was found to render.</i></p>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(f"Report written to {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()

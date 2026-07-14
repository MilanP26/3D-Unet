"""Phase B: run the trained seeded 3D U-Net over real VAST skeleton seeds
against the full hard-drive EM stack (PLAN.md section 10, CLAUDE.md
"Building Phase B").

For each sampled seed along a neuron's skeleton trace, this reads a patch
directly from the tiled full-stack (`phase_b_stack`), builds the same seed
heatmap used in training, and runs the existing trained model
(`infer.run_inference`) -- no new inference logic, only new I/O to get real
patches/seeds in front of it.

Output format intentionally stays a raw, mergeable intermediate (per-seed
packed-bit patch masks + their absolute placement) rather than a final
VAST-importable file: what VAST accepts for import is still an open
question (see PLAN.md/CLAUDE.md) and shouldn't block getting real
predictions on real seeds running end-to-end first.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .infer import run_inference
from .model import SeededUNet3D
from .phase_b_stack import load_vsvi, read_region_centered
from .vast_skeleton import DEFAULT_SKELETON_CSV, load_skeletons, subsample_seeds
from .visualize import save_overlay_montage


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tree-id", type=int, required=True, help="skeleton/tree id from the CSV")
    p.add_argument("--skeleton-csv", type=Path, default=DEFAULT_SKELETON_CSV)
    p.add_argument(
        "--vsvi", type=Path, required=True,
        help=r"path to the full stack's volume.vsvi, e.g. F:\ppa_b4v5s13\aligned_stack\volume.vsvi "
        "(requires the hard drive to be attached)",
    )
    p.add_argument("--target-spacing-nm", type=float, default=500.0)
    p.add_argument("--max-seeds", type=int, default=None, help="cap seeds processed, for a quick test run")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/phase_b"))
    p.add_argument(
        "--save-example-visualizations", type=int, default=3,
        help="save a raw+mask overlay PNG for this many seeds (spread across the trace) while "
        "the hard drive is attached, so there's something viewable without re-reading it later; 0 to disable",
    )
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    cfg = load_vsvi(args.vsvi)
    print(
        f"Full stack: {cfg.size_x}x{cfg.size_y}x{cfg.size_z} voxels (x,y,z), "
        f"scale_nm(x,y,z)={cfg.scale_nm_xyz}"
    )

    # weights_only=False: this is a checkpoint train.py produced itself, not a third-party
    # file -- safe to unpickle fully. Needed for checkpoints saved before args were
    # stringified (see train.py), and harmless for newer ones.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    # predict_lsd must match what the checkpoint was actually trained with, or
    # load_state_dict below will fail on a missing/unexpected lsd_head.
    predict_lsd = not train_args.get("no_lsd", False)
    model = SeededUNet3D(base_channels=train_args["base_channels"], predict_lsd=predict_lsd).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    patch_shape_zyx = tuple(train_args["patch_size"])
    seed_sigma_nm = train_args["seed_sigma_nm"]
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, patch_shape_zyx={patch_shape_zyx}")

    trees = load_skeletons(args.skeleton_csv)
    if args.tree_id not in trees:
        raise ValueError(f"tree_id {args.tree_id} not found in {args.skeleton_csv} ({len(trees)} trees present)")
    nodes = trees[args.tree_id]
    seeds = subsample_seeds(nodes, cfg.scale_nm_xyz, target_spacing_nm=args.target_spacing_nm)
    print(f"Tree {args.tree_id}: {len(nodes)} traced nodes -> {len(seeds)} subsampled seeds")
    if args.max_seeds is not None:
        seeds = seeds[: args.max_seeds]
        print(f"--max-seeds set: only running the first {len(seeds)}")

    out_dir = args.output_dir / f"tree_{args.tree_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_vis = min(args.save_example_visualizations, len(seeds))
    vis_indices = set(np.linspace(0, len(seeds) - 1, n_vis).round().astype(int).tolist()) if n_vis > 0 else set()

    seed_local_ids, seed_xyz, patch_origins_zyx, packed_masks = [], [], [], []
    pz, py, px = patch_shape_zyx
    for i, node in enumerate(tqdm(seeds, desc=f"tree {args.tree_id} seeds", unit="seed")):
        center_zyx = (node.z, node.y, node.x)
        raw_patch = read_region_centered(cfg, center_zyx, patch_shape_zyx)
        patch_center_zyx = (pz // 2, py // 2, px // 2)

        mask = run_inference(
            model,
            raw_patch,
            cfg.scale_nm_xyz,
            patch_center_zyx,
            patch_shape_zyx,
            seed_sigma_nm,
            device,
        )

        seed_local_ids.append(node.local_id)
        seed_xyz.append((node.x, node.y, node.z))
        patch_origins_zyx.append([c - p // 2 for c, p in zip(center_zyx, patch_shape_zyx)])
        packed_masks.append(np.packbits(mask))

        if i in vis_indices:
            vis_path = out_dir / f"example_node{node.local_id}.png"
            origin_zyx = patch_origins_zyx[-1]
            # Every real traced node (not just the subsampled seeds) that happens to fall
            # inside this patch, in patch-local coords -- lets the montage show the actual
            # annotator trace, not just the one seed this patch was centered on.
            node_markers_zyx = [
                (local_z, local_y, local_x)
                for n in nodes
                if 0 <= (local_z := n.z - origin_zyx[0]) < pz
                and 0 <= (local_y := n.y - origin_zyx[1]) < py
                and 0 <= (local_x := n.x - origin_zyx[2]) < px
            ]
            save_overlay_montage(
                raw_patch, mask, vis_path, seed_zyx=patch_center_zyx, node_markers_zyx=node_markers_zyx,
            )

    np.savez_compressed(
        out_dir / "predictions.npz",
        tree_id=args.tree_id,
        patch_shape_zyx=np.array(patch_shape_zyx),
        seed_local_ids=np.array(seed_local_ids),
        seed_xyz=np.array(seed_xyz),
        patch_origin_zyx=np.array(patch_origins_zyx),
        packed_masks=np.stack(packed_masks),
    )
    print(f"Saved {len(seeds)} patch predictions to {out_dir / 'predictions.npz'}")
    if vis_indices:
        print(f"Saved {len(vis_indices)} example overlay PNGs to {out_dir} (example_node*.png)")


if __name__ == "__main__":
    main()
